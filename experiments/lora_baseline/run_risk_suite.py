"""Run the authorized C then LoRA risk suite against a frozen training checkout.

No automatic retries, resume, or accuracy-dependent changes. Failed outputs and
epoch checkpoints remain available for recovery into a new directory.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

DISEASES = ('ckd', 'fatty_liver', 'diabetes')


def plan():
    return [dict(name=f'{group}_{disease}_seed{seed}', group=group,
                 disease=disease, seed=seed)
            for group in ('C', 'lora') for seed in (1, 2, 3) for disease in DISEASES]


def selected_epoch(epochs):
    if not epochs or any(not math.isfinite(row[key]) for row in epochs
            for key in ('train_weighted_bce', 'validation_weighted_bce',
                        'validation_weighted_auroc', 'validation_weighted_brier')):
        raise ValueError('Missing or nonfinite epoch metrics')
    # Strict improvement: equal validation losses retain the earlier epoch.
    return min(epochs, key=lambda row: row['validation_weighted_bce'])['epoch']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'release', 'data', 'embedding-cache', 'preflight', 'gpu-profile', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--wandb-entity', required=True)
    args = parser.parse_args()
    here = args.runtime / 'experiments/lora_baseline'
    sys.path.insert(0, str(here))
    import numpy as np
    import torch
    import common
    from checkpointing import atomic_write, sha256, load_epoch
    from audit_record_resume import tree_equal
    from data import PrefixSplit
    from gates import validate_gate
    from metrics import weighted_auroc, weighted_brier
    from runtime_guard import RuntimeGuard
    if args.out.exists():
        raise ValueError('Choose a new output directory')
    args.out.mkdir(parents=True)
    locks = []
    for name in ('risk_preflight.lock', 'risk_main.lock'):
        lock = (args.out.parent/name).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    prior = json.loads((args.preflight/'status.json').read_text())
    if prior['status'] != 'completed' or common.git_commit() != args.expected_commit:
        raise ValueError('Verified preflight and execution commit required')
    sources = dict(prior['source_sha256'])
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    sources.update({str(p): sha256(p) for p in
                    (args.runtime/'releases/fermat_snuh_20261021/fermat_release').glob('*.py')})
    sources[str(here/'host_health.ps1')] = sha256(here/'host_health.ps1')
    gates = {d: json.loads((args.preflight/f'gate_{d}/gate.json').read_text()) for d in DISEASES}
    if not all(g['passed'] for g in gates.values()):
        raise ValueError('All disease gates must pass')
    guard = RuntimeGuard(args.gpu_profile, args.out)
    state = dict(status='running', main_experiment_started=True,
                 started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
                 execution_commit=args.expected_commit, source_sha256=sources,
                 plan=plan(), completed=[], current=None)
    def write(name, value):
        atomic_write(args.out/name, lambda f: f.write((json.dumps(value, indent=2)+'\n').encode()))
    def intact():
        if any(sha256(path) != digest for path,digest in sources.items()):
            raise ValueError('Execution source changed')
        if shutil.disk_usage(args.out).free < 30*2**30:
            raise ValueError('Less than 30 GiB free')
        guard.check(force=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_ERROR_REPORTING='false')
    def execute(command, log, tracking=None):
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            state['child_pid'] = process.pid
            write('status.json', state)
            start = time.monotonic()
            try:
                while process.poll() is None:
                    if tracking and time.monotonic()-start > 45:
                        track = json.loads(tracking.read_text()) if tracking.exists() else {}
                        if not track.get('connected') or track.get('delivery_failed'):
                            raise ValueError('W&B connection/delivery failed; outputs preserved')
                    if time.monotonic()-start > 4*3600:
                        raise TimeoutError('One run exceeded four hours')
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f'Child exited {process.returncode}: {log.name}')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                raise
    try:
        intact()
        baseline_dir = args.out/'baselines'; baseline_dir.mkdir()
        ids = np.load(args.data/'test/patient_id.npy')
        for d in DISEASES:
            with (baseline_dir/f'{d}.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['patient_id', 'probability'])
                writer.writerows(zip(ids, np.load(args.data/'test'/f'baseline_probability_{d}.npy')))
        write('status.json', state)
        for job in plan():
            intact()
            state['current'] = job; write('status.json', state)
            folder = args.out/job['name']; d = job['disease']
            command = [sys.executable, str(here/'train_risk.py'), '--release', str(args.release),
                '--data', str(args.data), '--embedding-cache', str(args.embedding_cache),
                '--out', str(folder), '--disease', d, '--seed', str(job['seed']),
                '--arm', 'lora' if job['group']=='lora' else 'head_only',
                '--recipe', 'lora' if job['group']=='lora' else 'experimental',
                '--gate', str(args.preflight/f'gate_{d}/gate.json'),
                '--device', 'cuda', '--bf16', '--batch-size', '32', '--epochs', '3', '--patience', '1',
                '--warmup-steps', '100', '--grad-clip', '1', '--lr-head', '.001', '--weight-decay', '.01',
                '--gpu-profile', str(args.gpu_profile), '--tracking', 'online',
                '--wandb-entity', args.wandb_entity, '--run-name', 'risk_main_'+job['name']]
            if job['group']=='C':
                command += ['--matched-head']
            else:
                command += ['--baseline-probabilities', str(baseline_dir/f'{d}.csv'),
                            '--baseline-label', 'saved_v1', '--bootstrap-resamples', '2000']
            execute(command, args.out/(job['name']+'.log'), folder/'tracking/tracking.json')
            intact()
            log = json.loads((folder/'summary.json').read_text()); fp=log['fingerprint']; summary=log['summary']
            cfg=fp['resolved']; actual=fp['arguments']
            if (fp['git_commit']!=args.expected_commit or actual['seed']!=job['seed'] or
                actual['disease']!=d or cfg['epochs']!=3 or cfg['patience']!=1 or
                actual['batch_size']!=32 or fp['warmup_steps']!=100 or fp['grad_clip']!=1 or
                actual['matched_head']!=(job['group']=='C') or fp['training_selection']!='all'):
                raise ValueError('Run settings differ from the plan')
            if validate_gate(gates[d],fp['gate_identity']) or not summary['audit']['frozen_base_unchanged']:
                raise ValueError('Identity or frozen-base audit failed')
            if selected_epoch(log['epochs'])!=summary['selected']['epoch']:
                raise ValueError('Wrong validation-selected epoch')
            manifest=json.loads((folder/'last_training.json').read_text())
            payload=torch.load(folder/manifest['checkpoint'],map_location='cpu',weights_only=True)
            payload,_=load_epoch(folder/'last_training.json',payload['contract'])
            best=torch.load(folder/'trainable_state.pt',map_location='cpu',weights_only=True)
            if not tree_equal(best,payload['best_state']) or payload['optimizer_steps']!=summary['optimizer_steps']:
                raise ValueError('Selected checkpoint or update count mismatch')
            scores=torch.load(folder/'test_scores.pt',map_location='cpu',weights_only=True)
            split=PrefixSplit(args.data,'test',diseases=[d]); y,w=split.outcomes[d]; mask=w>0
            p=scores['probabilities'].numpy()
            if not np.array_equal(scores['patient_id'].numpy(),ids) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                raise ValueError('Prediction identity or range invalid')
            recomputed=dict(weighted_auroc=weighted_auroc(y[mask],p[mask],w[mask]),
                            weighted_brier=weighted_brier(y[mask],p[mask],w[mask]))
            if any(abs(v-summary['test'][k])>1e-12 for k,v in recomputed.items()):
                raise ValueError('Metric readback mismatch')
            # New process reloading for each LoRA seed 1 before seeds 2/3.
            # No accuracy threshold participates in continuation.
            if job['group']=='lora' and job['seed']==1:
                reload=folder/'reload'
                execute([sys.executable,str(here/'predict.py'),'--release',str(args.release),
                    '--data',str(args.data),'--state',str(folder/'trainable_state.pt'),
                    '--split','test','--out',str(reload),'--device','cuda','--batch-size','32',
                    '--gpu-profile',str(args.gpu_profile)],args.out/(job['name']+'_reload.log'))
                reloaded=torch.load(reload/'test_scores.pt',map_location='cpu',weights_only=True)
                if not tree_equal(scores,reloaded):
                    raise ValueError('Independent seed-1 reload differs')
            track=json.loads((folder/'tracking/tracking.json').read_text())
            if not track.get('success') or not track.get('connected') or track.get('finish_error') or track.get('delivery_failed'):
                raise ValueError('W&B finish failed')
            import wandb
            for attempt in range(3):
                remote=wandb.Api(timeout=30).run(args.wandb_entity+'/fermat-lora-baseline/'+track['run_id'])
                if remote.state=='finished':
                    break
                time.sleep(5)
            if remote.state!='finished':
                raise ValueError('W&B remote run not finished')
            result=dict(**job,summary=summary,tracking=track,checkpoint_sha256=manifest['sha256'],
                        independent_metrics=recomputed,reload_exact=job['group']=='lora' and job['seed']==1)
            state['completed'].append(result); write('status.json',state)
        state.update(status='completed',current=None,finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write('status.json',state)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
        write('status.json',state)
        raise


if __name__=='__main__':
    main()
