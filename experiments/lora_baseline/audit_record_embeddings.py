"""CUDA BF16 parity against frozen v1 embeddings, with original 32-row batches.

Checks original-batch composition and exported-cohort composition separately.
This audit never fits a head or issues a performance gate.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import common
from data import PrefixSplit
from prepare_record_v1 import pack_prefixes, verify
from runtime_guard import RuntimeGuard


def compare(actual, reference):
    difference = np.abs(actual - reference)
    cosine = (actual * reference).sum(1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(reference, axis=1))
    return dict(passed=bool(np.allclose(actual, reference, rtol=1e-5, atol=1e-6)),
                maximum_absolute_error=float(difference.max()), minimum_cosine=float(cosine.min()),
                exact_fraction=float(np.mean(actual == reference)))


def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'pilot', 'data', 'out', 'gpu-profile'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--probe-seconds', type=float, default=60.)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh audit directory')
    args.out.mkdir(parents=True)
    guard = RuntimeGuard(args.gpu_profile, args.out)
    metadata = json.loads((args.data / 'dataset.json').read_text())
    for name in ('targets.parquet', 'prefixes.npz', 'embeddings.npy'):
        verify(args.pilot / name, metadata['source_sha256']['pilot/' + name])
    model, info = common.build_model(SimpleNamespace(release=args.release, smoke_random_init=None, device='cuda'))
    target = pd.read_parquet(args.pilot / 'targets.parquet')
    packed = np.load(args.pilot / 'prefixes.npz')
    reference = np.load(args.pilot / 'embeddings.npy', mmap_mode='r')
    starts = np.linspace(0, (len(target) - 32) // 32, 8, dtype=int) * 32
    report = dict(batch_size=32, rtol=1e-5, atol=1e-6, model=info, torch=torch.__version__,
                  original_batches=[], exported_batches=[], probe_seconds=args.probe_seconds)
    work = []
    for start in starts:
        selected = np.arange(start, start + 32)
        idx, age, typ = pack_prefixes(packed['rows'], packed['offsets'], selected,
            target.patient_id_dense.to_numpy(), target.index_age_days.to_numpy())
        width = int((idx > 0).sum(1).max())
        work.append((tuple(torch.from_numpy(x[:, :width].copy()).cuda() for x in (idx, age, typ)),
                     reference[selected], 'original_batches'))
    for name in ('train', 'validation', 'test'):
        split = PrefixSplit(args.data, name)
        work.append((split.batch(np.arange(32), 'cuda'),
                     np.load(args.data / name / 'frozen_embeddings.npy', mmap_mode='r')[:32], 'exported_batches'))
    start = time.perf_counter()
    timings = []
    with torch.inference_mode():
        for tensors, expected, kind in work:
            guard.check()
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                actual = common.last_hidden(model, *tensors).float().cpu().numpy()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            timings.append(elapsed)
            report[kind].append(dict(compare(actual, expected), seconds=elapsed))
        while time.perf_counter() - start < args.probe_seconds:
            guard.check()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                value = common.last_hidden(model, *work[0][0])
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite CUDA probe output')
            torch.cuda.synchronize()
            time.sleep(.25)
    guard.check(force=True)
    report.update(passed=all(x['passed'] for key in ('original_batches', 'exported_batches') for x in report[key]),
                  seconds=time.perf_counter() - start, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  median_seconds_per_32=float(np.median(timings[1:])),
                  estimated_full_extraction_seconds=float(np.median(timings[1:]) * 695))
    common.write_json(args.out / 'audit.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
