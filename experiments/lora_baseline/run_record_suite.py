"""Fixed A -> C -> LoRA record suite. Abort on failed integrity checks, never on accuracy.

Run from a dedicated, committed execution snapshot. No automatic retry/resume:
failed runs retain their epoch checkpoints for explicit recovery into fresh folders.
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

import numpy as np
import torch

import common
from checkpointing import atomic_write, load_epoch, sha256
from gates import validate_gate
from metrics import mean_difference, paired_bootstrap, rank_classes, record_metrics
from data import PrefixSplit, class_lookup
from recipes import record_improved
from runtime_guard import RuntimeGuard


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2) + '\n').encode()))


def plan():
    return [dict(name=f'{group}_seed{seed}', group=group, seed=seed,
                 arm='lora' if group == 'lora' else 'head_only',
                 recipe='lora' if group == 'lora' else 'experimental',
                 batch=256 if group == 'A' else 32,
                 epochs=60 if group == 'A' else 20,
                 patience=10 if group == 'A' else 5,
                 warmup=0 if group == 'A' else 100,
                 clip=None if group == 'A' else 1.)
            for group in ('A', 'C', 'lora') for seed in (1, 2, 3)]


def source_hashes():
    return {str(p.relative_to(common.REPO)): sha256(p)
            for folder in (Path(__file__).parent, common.DEFAULT_RELEASE / 'fermat_release')
            for p in sorted(folder.glob('*.py'))}


def verify_run(folder, job, args):
    """Structural/numerical checks only; no performance threshold decides continuation."""
    log = json.loads((folder / 'summary.json').read_text())
    fp, summary = log['fingerprint'], log['summary']
    cfg = fp['resolved']
    if any(cfg[k] != job[k] for k in ('batch', 'epochs', 'patience')):
        raise ValueError('Wrong resolved training schedule')
    if fp['warmup_steps'] != job['warmup'] or fp['grad_clip'] != job['clip']:
        raise ValueError('Wrong warmup/clipping schedule')
    if fp['git_commit'] != args.expected_commit or fp['arguments']['seed'] != job['seed']:
        raise ValueError('Wrong execution revision or seed')
    reasons = validate_gate(json.loads(args.gate.read_text()), fp['gate_identity'])
    if reasons:
        raise ValueError('Run identity differs from the verified v1 cohort: ' + '; '.join(reasons))
    if not summary['audit']['frozen_base_unchanged'] or summary['optimizer_steps'] <= 0:
        raise ValueError('Frozen-base or optimizer audit failed')
    best = None
    for row in log['epochs']:
        if not all(math.isfinite(row[k]) for k in ('train_bce', 'validation_bce', 'validation_recall_at_10')):
            raise ValueError('Nonfinite epoch metric')
        if record_improved(row['validation_recall_at_10'], row['validation_bce'], best):
            best = (row['validation_recall_at_10'], row['validation_bce'], row['epoch'])
    if best is None or best[2] != summary['selected']['epoch']:
        raise ValueError('Selected epoch does not follow the frozen validation rule')
    manifest = json.loads((folder / 'last_training.json').read_text())
    payload = torch.load(folder / manifest['checkpoint'], map_location='cpu', weights_only=True)
    payload, _ = load_epoch(folder / 'last_training.json', payload['contract'])
    current_sources = source_hashes()
    if payload['contract']['code'] != {k: current_sources[
            ('releases/fermat_snuh_20261021/fermat_release/' + k[8:]) if k.startswith('release/')
            else 'experiments/lora_baseline/' + k] for k in payload['contract']['code']}:
        raise ValueError('Training source changed')
    if payload['optimizer_steps'] != summary['optimizer_steps']:
        raise ValueError('Checkpoint update count mismatch')
    selected = torch.load(folder / 'trainable_state.pt', map_location='cpu', weights_only=True)
    from audit_record_resume import tree_equal
    if not tree_equal(selected, payload['best_state']):
        raise ValueError('Final inference state differs from selected checkpoint')
    scores = torch.load(folder / 'test_scores.pt', map_location='cpu', weights_only=True)
    test = PrefixSplit(args.data, 'test')
    if not np.array_equal(scores['patient_id'].numpy(), test.patient_id) or not torch.isfinite(scores['logits']).all():
        raise ValueError('Test patient identity or finite-logit check failed')
    classes = selected['preprocessing']['classes'].numpy()
    if not np.array_equal(scores['top10_classes'].numpy(), classes[rank_classes(scores['logits'], 10).numpy()]):
        raise ValueError('Stored top-10 differs from logits')
    config_path = Path(fp['arguments']['smoke_random_init']) if fp['arguments']['smoke_random_init'] else args.release / 'config/model_config.json'
    vocab_size = json.loads(config_path.read_text())['vocab_size']
    values = record_metrics(scores['logits'], test.label_lists(np.arange(len(test))), class_lookup(classes, vocab_size))
    for key, value in values.items():
        if abs(float(value.mean()) - summary['test'][key]) > 1e-8:
            raise ValueError('Summary metric differs from saved predictions')
    paired = summary['paired_recall_at_10_minus_generation50']
    baseline = common.align_baseline(test.patient_id, common.read_baseline_csv(args.baseline_recall, 'recall_at_10'), 'generation50')
    if abs(mean_difference(values['recall_at_10'], baseline) - paired['point']) > 1e-8 or paired['n_resamples'] != 5000:
        raise ValueError('Primary contrast integrity check failed')
    status = json.loads((folder / 'tracking/tracking.json').read_text())
    if not status.get('connected') or not status.get('success') or status.get('delivery_failed') or status.get('finish_error'):
        raise ValueError('W&B delivery failed; preserve local results and stop queue')
    report = dict(passed=True, continuation_depends_on_accuracy=False, run_id=status['run_id'],
                  selected_epoch=best[2], completed_epochs=len(log['epochs']),
                  optimizer_steps=summary['optimizer_steps'], test_patients=len(test),
                  recall_at_10=summary['test']['recall_at_10'],
                  checkpoint_sha256=manifest['sha256'])
    write_json(folder / 'pipeline_check.json', report)
    return report


def final_contrasts(out):
    def rows(folder):
        with (folder / 'test_patient_metrics.csv').open() as stream:
            values = list(csv.DictReader(stream))
        return {int(r['patient_id']): float(r['recall_at_10']) for r in values}
    result = dict(interpretation='Patient bootstrap conditional on fitted models; seed variation reported separately. '
                  'No pooling of patients across seeds. Secondary comparisons are exploratory.', comparisons={})
    for seed in (1, 2, 3):
        lora = rows(out / f'lora_seed{seed}')
        ids = sorted(lora)
        for control in ('A', 'C'):
            reference = rows(out / f'{control}_seed{seed}')
            if set(reference) != set(lora):
                raise ValueError('Secondary contrast patient sets differ')
            result['comparisons'][f'lora_seed{seed}_minus_{control}_seed{seed}'] = paired_bootstrap(
                mean_difference, [np.array([lora[i] for i in ids]), np.array([reference[i] for i in ids])], 5000, seed)
    write_json(out / 'secondary_contrasts.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'release', 'gate', 'baseline-recall', 'gpu-profile', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--wandb-entity', required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (args.out.parent / 'record_suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.out.exists():
            raise ValueError('Suite output already exists; explicit recovery is required')
        args.out.mkdir()
        status = dict(state='preflight', pid=os.getpid(), completed=[], expected_commit=args.expected_commit,
                      started_unix=time.time())
        write_json(args.out / 'status.json', status)
        try:
            if common.git_commit() != args.expected_commit:
                raise ValueError('Execution snapshot revision mismatch')
            dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=common.REPO,
                                   check=True, capture_output=True, text=True).stdout
            if dirty.strip():
                raise ValueError('Execution snapshot has tracked edits')
            hashes = source_hashes()
            inputs = {str(p): sha256(p) for p in (args.gate, args.baseline_recall,
                       args.release / 'config/model_config.json', args.release / 'artifacts/model.pt')}
            write_json(args.out / 'plan.json', dict(jobs=plan(), source_hashes=hashes, input_hashes=inputs,
                       seed1_check='Artifact integrity only; no test-performance threshold',
                       code_policy='Do not edit this execution snapshot while the suite is running'))
            guard = RuntimeGuard(args.gpu_profile, args.out)
            for _ in range(3):
                guard.check(force=True)
                time.sleep(2)
            for job in plan():
                if source_hashes() != hashes or any(sha256(Path(p)) != digest for p, digest in inputs.items()):
                    raise ValueError('Code or immutable input changed; stop before next job')
                if shutil.disk_usage(args.out).free < 70 * 2**30:
                    raise RuntimeError('Less than 70 GiB free; retain checkpoints and stop')
                guard.check(force=True)
                name = job['name']
                status.update(state='running', current=name, phase='training', updated_unix=time.time())
                write_json(args.out / 'status.json', status)
                command = [sys.executable, str(Path(__file__).with_name('train_record.py')),
                    '--data', str(args.data), '--release', str(args.release), '--gate', str(args.gate),
                    '--out', str(args.out / name), '--arm', job['arm'], '--recipe', job['recipe'],
                    '--seed', str(job['seed']), '--batch-size', '32', '--device', 'cuda', '--bf16',
                    '--epochs', str(job['epochs']), '--patience', str(job['patience']),
                    '--lr-head', '.001', '--weight-decay', '.01', '--warmup-steps', str(job['warmup']),
                    '--grad-accum', '1', '--bootstrap-resamples', '5000',
                    '--baseline-recall', str(args.baseline_recall), '--baseline-label', 'generation50',
                    '--gpu-profile', str(args.gpu_profile), '--tracking', 'online',
                    '--wandb-entity', args.wandb_entity, '--run-name', f'record-main-{name}']
                if job['group'] == 'C':
                    command += ['--optimizer-batch-size', '32']
                if job['clip'] is not None:
                    command += ['--grad-clip', str(job['clip'])]
                write_json(args.out / (name + '_command.json'), command)
                with (args.out / (name + '.log')).open('w') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=10800)
                status.update(phase='pipeline_check', updated_unix=time.time())
                write_json(args.out / 'status.json', status)
                if source_hashes() != hashes:
                    raise ValueError('Source changed during training')
                verified = verify_run(args.out / name, job, args)
                status['completed'].append(dict(name=name, **verified))
                write_json(args.out / 'status.json', status)
            final_contrasts(args.out)
            status.update(state='completed', phase='complete', updated_unix=time.time())
            write_json(args.out / 'status.json', status)
        except BaseException as exc:
            status.update(state='failed', error_type=type(exc).__name__, error=str(exc), updated_unix=time.time())
            write_json(args.out / 'status.json', status)
            raise


if __name__ == '__main__':
    main()
