"""Real CUDA validation-only audit: uninterrupted vs paused/resumed, then reload.

Each training job is limited to two epochs on 128 selected training patients.
The original dataset and training-only scaler pool stay intact. No test scoring.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from checkpointing import load_epoch, sha256


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'data', 'gate', 'out', 'gpu-profile'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--tracking', choices=('disabled', 'online', 'offline'), default='disabled')
    parser.add_argument('--wandb-entity')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Audit output already exists; choose a new directory')
    args.out.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    base = [sys.executable, str(here / 'train_record.py'), '--release', str(args.release),
            '--data', str(args.data), '--gate', str(args.gate), '--arm', 'lora', '--recipe', 'lora',
            '--device', 'cuda', '--bf16', '--batch-size', '32', '--seed', '1', '--train-subset', '128',
            '--epochs', '2', '--patience', '5', '--validation-only', '--gpu-profile', str(args.gpu_profile),
            '--tracking', args.tracking]
    if args.wandb_entity:
        base += ['--wandb-entity', args.wandb_entity]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_ERROR_REPORTING='false')
    durations = {}
    def execute(name, command):
        print(json.dumps(dict(phase=name, status='started')), flush=True)
        started = time.perf_counter()
        with (args.out / (name + '.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True, timeout=600)
        durations[name] = time.perf_counter() - started
        print(json.dumps(dict(phase=name, status='completed', seconds=durations[name])), flush=True)
    execute('uninterrupted', base + ['--out', str(args.out / 'uninterrupted'), '--run-name', 'record-resume-audit-uninterrupted'])
    execute('paused', base + ['--out', str(args.out / 'paused'), '--stop-after-epoch', '1', '--run-name', 'record-resume-audit-paused'])
    execute('resumed', base + ['--out', str(args.out / 'resumed'), '--resume', str(args.out / 'paused/last_training.json'),
                              '--run-name', 'record-resume-audit-resumed'])
    def latest(name):
        manifest_path = args.out / name / 'last_training.json'
        manifest = json.loads(manifest_path.read_text())
        payload = torch.load(manifest_path.parent / manifest['checkpoint'], map_location='cpu', weights_only=True)
        return load_epoch(manifest_path, payload['contract'])[0]
    a, b = latest('uninterrupted'), latest('resumed')
    fields = ('current_state', 'optimizer', 'best_state', 'best', 'since_best', 'optimizer_steps', 'rng', 'contract')
    checks = {key: tree_equal(a[key], b[key]) for key in fields}
    clean_epochs = lambda payload: [{k:v for k,v in row.items() if k != 'epoch_seconds'} for row in payload['epochs']]
    checks['epoch_metrics'] = tree_equal(clean_epochs(a), clean_epochs(b))
    reference = torch.load(args.out / 'uninterrupted/validation_scores.pt', weights_only=True)
    continued = torch.load(args.out / 'resumed/validation_scores.pt', weights_only=True)
    checks['validation_scores'] = tree_equal(reference, continued)
    checks['no_test_outputs'] = not any((args.out / run / file).exists() for run in ('uninterrupted', 'paused', 'resumed')
                                       for file in ('test_scores.pt', 'test_patient_metrics.csv', 'gate.json'))
    report = dict(passed=all(checks.values()), checks=checks, train_subset=128, epochs=2,
                  optimizer_steps=b['optimizer_steps'], validation_patients=len(reference['patient_id']),
                  validation_max_absolute_error=float((reference['logits'] - continued['logits']).abs().max()),
                  code_sha256=a['contract']['code'], durations=durations,
                  run_ids={name: json.loads((args.out / name / 'tracking/tracking.json').read_text()).get('run_id')
                           for name in ('uninterrupted', 'paused', 'resumed')})
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('Uninterrupted/resumed runs differ; inspect audit.json')
    execute('reload', [sys.executable, str(here / 'predict.py'), '--state', str(args.out / 'resumed/trainable_state.pt'),
                      '--release', str(args.release), '--data', str(args.data), '--out', str(args.out / 'reload'),
                      '--device', 'cuda', '--split', 'validation', '--batch-size', '32',
                      '--gpu-profile', str(args.gpu_profile), '--reference-scores', str(args.out / 'resumed/validation_scores.pt')])
    report['reload'] = json.loads((args.out / 'reload/reload_audit.json').read_text())
    report['passed'] = report['passed'] and report['reload']['passed']
    report['durations'] = durations
    report['artifact_sha256'] = {name: sha256(args.out / 'resumed' / name) for name in ('last_training.json', 'trainable_state.pt')}
    (args.out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
