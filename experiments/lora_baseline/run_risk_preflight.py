"""Bounded risk preparation queue: C10 gates, resume audit, three timing pilots.

This command never starts a full-cohort LoRA epoch or the main experiment suite.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from checkpointing import sha256
from prepare_risk_baselines import DISEASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('release', 'data', 'embedding-cache', 'fixed-models', 'gpu-profile', 'out'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--wandb-entity', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Choose a new output')
    args.out.mkdir(parents=True)
    lock = (args.out.parent/'risk_preflight.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    here = Path(__file__).resolve().parent
    paths = sorted(here.glob('*.py'))
    sources = {str(path): sha256(path) for path in paths}
    shared = ['--release', str(args.release), '--data', str(args.data), '--embedding-cache', str(args.embedding_cache),
              '--device', 'cuda', '--bf16', '--batch-size', '32', '--gpu-profile', str(args.gpu_profile),
              '--tracking', 'online', '--wandb-entity', args.wandb_entity]
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_ERROR_REPORTING='false')
    status = dict(status='running', main_experiment_started=False, phases=[], source_sha256=sources)
    def write():
        temp = args.out/'status.json.tmp'
        temp.write_text(json.dumps(status, indent=2)+'\n')
        temp.replace(args.out/'status.json')
    def execute(name, command):
        if any(sha256(Path(path)) != value for path,value in sources.items()):
            raise ValueError('Code changed during preflight')
        if shutil.disk_usage(args.out).free < 30*2**30:
            raise ValueError('Less than 30 GiB free')
        phase = dict(name=name, status='running')
        status['phases'].append(phase); write()
        print(json.dumps(phase), flush=True)
        start = time.perf_counter()
        with (args.out/(name+'.log')).open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        phase.update(status='completed', seconds=time.perf_counter()-start)
        write()
    try:
        for disease, phenotype in DISEASES.items():
            name = 'gate_'+disease
            execute(name, [sys.executable, str(here/'train_risk.py'), *shared, '--disease', disease,
                '--arm', 'head_only', '--recipe', 'v1', '--seed', '0', '--out', str(args.out/name),
                '--reference-model', str(args.fixed_models/f'{phenotype}__fermat_patient_representation.json'),
                '--run-name', name])
            gate = json.loads((args.out/name/'gate.json').read_text())
            if not gate['passed']:
                raise ValueError('C10 gate failed for '+disease)
            track = json.loads((args.out/name/'tracking/tracking.json').read_text())
            if not track.get('connected') or not track.get('success'):
                raise ValueError('W&B unavailable; inspect preserved local outputs')
        execute('resume_audit', [sys.executable, str(here/'audit_risk_resume.py'),
            '--release', str(args.release), '--data', str(args.data), '--embedding-cache', str(args.embedding_cache),
            '--gate', str(args.out/'gate_ckd/gate.json'), '--gpu-profile', str(args.gpu_profile),
            '--out', str(args.out/'resume_audit'), '--tracking', 'online', '--wandb-entity', args.wandb_entity])
        if not json.loads((args.out/'resume_audit/audit.json').read_text())['passed']:
            raise ValueError('CUDA resume audit failed')
        for disease in DISEASES:
            name = 'benchmark_'+disease
            execute(name, [sys.executable, str(here/'train_risk.py'), *shared, '--disease', disease,
                '--arm', 'lora', '--recipe', 'lora', '--seed', '1', '--benchmark-steps', '128', '--benchmark-warmup', '10',
                '--gate', str(args.out/f'gate_{disease}/gate.json'), '--out', str(args.out/name), '--run-name', name])
            report = json.loads((args.out/name/'benchmark.json').read_text())['benchmark']
            if report['test_evaluated'] or report['optimizer_steps'] != 128 or not report['audit']['frozen_base_unchanged']:
                raise ValueError('Timing pilot integrity check failed')
            if any((args.out/name/file).exists() for file in ('test_scores.pt', 'summary.json', 'gate.json', 'last_training.json')):
                raise ValueError('Unexpected pilot output')
        status['status'] = 'completed'
        write()
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        write()
        raise


if __name__ == '__main__':
    main()
