"""Bounded real CUDA risk resume audit, with independent probability reload.

128 observed training patients, two epochs, complete observed validation cohort.
No test predictions and no gate issuance. Existing full-pool gate required.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from audit_record_resume import tree_equal
from checkpointing import load_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('release', 'data', 'gate', 'out', 'gpu-profile', 'embedding-cache'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--disease', default='ckd')
    parser.add_argument('--tracking', choices=('disabled', 'offline', 'online'), default='disabled')
    parser.add_argument('--wandb-entity')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Choose a new audit directory')
    args.out.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    base = [sys.executable, str(here/'train_risk.py'), '--release', str(args.release), '--data', str(args.data),
            '--gate', str(args.gate), '--embedding-cache', str(args.embedding_cache), '--disease', args.disease,
            '--arm', 'lora', '--recipe', 'lora', '--device', 'cuda', '--bf16', '--batch-size', '32',
            '--seed', '1', '--train-subset', '128', '--epochs', '2', '--patience', '5', '--validation-only',
            '--gpu-profile', str(args.gpu_profile), '--tracking', args.tracking]
    if args.wandb_entity:
        base += ['--wandb-entity', args.wandb_entity]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_ERROR_REPORTING='false')
    durations = {}
    def execute(name, command):
        print(json.dumps(dict(phase=name, status='started')), flush=True)
        start = time.perf_counter()
        with (args.out/(name+'.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True, timeout=1200)
        durations[name] = time.perf_counter()-start
        print(json.dumps(dict(phase=name, status='completed', seconds=durations[name])), flush=True)
    execute('uninterrupted', base+['--out', str(args.out/'uninterrupted'), '--run-name', 'risk-resume-audit-uninterrupted'])
    execute('paused', base+['--out', str(args.out/'paused'), '--stop-after-epoch', '1', '--run-name', 'risk-resume-audit-paused'])
    execute('resumed', base+['--out', str(args.out/'resumed'), '--resume', str(args.out/'paused/last_training.json'),
                            '--run-name', 'risk-resume-audit-resumed'])
    def latest(name):
        path = args.out/name/'last_training.json'
        manifest = json.loads(path.read_text())
        state = torch.load(path.parent/manifest['checkpoint'], weights_only=True, map_location='cpu')
        return load_epoch(path, state['contract'])[0]
    a, b = latest('uninterrupted'), latest('resumed')
    checks = {key: tree_equal(a[key], b[key]) for key in
              ('current_state', 'optimizer', 'best_state', 'best', 'since_best', 'optimizer_steps', 'rng', 'contract')}
    clean = lambda p: [{k:v for k,v in row.items() if k != 'epoch_seconds'} for row in p['epochs']]
    checks['epoch_metrics'] = tree_equal(clean(a), clean(b))
    ref = torch.load(args.out/'uninterrupted/validation_scores.pt', weights_only=True)
    restored = torch.load(args.out/'resumed/validation_scores.pt', weights_only=True)
    checks['validation_scores'] = tree_equal(ref, restored)
    checks['no_test_outputs'] = not any((args.out/run/file).exists() for run in ('uninterrupted', 'paused', 'resumed')
        for file in ('test_scores.pt', 'test_patient_probabilities.csv', 'gate.json', 'summary.json'))
    report = dict(passed=all(checks.values()), checks=checks, disease=args.disease, epochs=2, train_subset=128,
                  validation_patients=len(ref['patient_id']),
                  maximum_absolute_error=float((ref['probabilities']-restored['probabilities']).abs().max()),
                  durations=durations, code_sha256=a['contract']['code'],
                  run_ids={name: json.loads((args.out/name/'tracking/tracking.json').read_text()).get('run_id')
                           for name in ('uninterrupted', 'paused', 'resumed')})
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise ValueError('Risk resume differs; inspect audit.json')
    execute('reload', [sys.executable, str(here/'predict.py'), '--state', str(args.out/'resumed/trainable_state.pt'),
        '--release', str(args.release), '--data', str(args.data), '--out', str(args.out/'reload'), '--device', 'cuda',
        '--split', 'validation', '--batch-size', '32', '--gpu-profile', str(args.gpu_profile),
        '--reference-scores', str(args.out/'resumed/validation_scores.pt')])
    report['reload'] = json.loads((args.out/'reload/reload_audit.json').read_text())
    report['passed'] = report['passed'] and report['reload']['passed']
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
