"""Shared risk embeddings: full source binding, original batch context, CUDA parity."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common
from data import PrefixSplit
from prepare_record_v1 import sha256, verify
from runtime_guard import RuntimeGuard
from tracking import Tracker, add_arguments


def source_identity(data):
    data = Path(data)
    m = json.loads((data / 'preparation.json').read_text())
    for rel, digest in m['array_sha256'].items():
        verify(data / rel, digest)
    return dict(dataset_json=sha256(data / 'dataset.json'), arrays=m['array_sha256'])


def load_cache(path, data, model_info, args, splits):
    path = Path(path)
    manifest = json.loads((path / 'cache.json').read_text())
    if not manifest['parity_exact'] or manifest['model_sha256'] != model_info['model_sha256']:
        raise ValueError('Unverified risk cache or wrong model')
    if manifest['bf16'] != args.bf16 or manifest['batch_size'] != 64:
        raise ValueError('Risk cache precision/batch mismatch')
    if manifest['sources'] != source_identity(data):
        raise ValueError('Risk cache dataset identity mismatch')
    verify(path / 'embeddings.npy', manifest['embeddings_sha256'])
    values = np.load(path / 'embeddings.npy', mmap_mode='r')
    pool = PrefixSplit(data, 'embedding_pool')
    if values.shape != (len(pool), 640) or values.dtype != np.float32:
        raise ValueError('Malformed risk cache')
    result = {}
    for name, split in splits.items():
        mapping = np.load(Path(data) / name / 'pool_rows.npy')
        if mapping.dtype != np.int64 or mapping.shape != (len(split),) or np.any((mapping < 0) | (mapping >= len(pool))):
            raise ValueError('Invalid risk cache mapping')
        if not np.array_equal(split.patient_id, pool.patient_id[mapping]):
            raise ValueError('Risk cache patient ordering mismatch')
        for start in range(0, len(split), 1024):
            for attr in ('idx', 'age', 'token_type'):
                if not np.array_equal(getattr(split, attr)[start:start+1024], getattr(pool, attr)[mapping[start:start+1024]]):
                    raise ValueError('Risk cache prefix mismatch')
        result[name] = values[mapping]
    return result, sha256(path / 'cache.json')


def main():
    parser = common.add_common_arguments(argparse.ArgumentParser(description=__doc__))
    add_arguments(parser)
    parser.add_argument('--gpu-profile', type=Path, required=True)
    args = parser.parse_args()
    if args.arm != 'head_only' or args.batch_size != 64 or not args.bf16 or args.device != 'cuda':
        raise ValueError('Original risk extraction requires CUDA BF16 head_only batch 64')
    if args.out.exists():
        raise ValueError('Choose a new output')
    args.out.mkdir(parents=True)
    tracker = Tracker(args.out / 'tracking', args.tracking, args.wandb_project, args.wandb_entity,
                      args.run_name, dict(task='risk_embedding_parity', batch_size=64, bf16=True))
    guard = RuntimeGuard(args.gpu_profile, args.out, tracker)
    sources = source_identity(args.data)
    model, info = common.build_model(args)
    model.eval()
    pool = PrefixSplit(args.data, 'embedding_pool')
    reference = np.load(args.data / 'embedding_pool/frozen_embeddings.npy', mmap_mode='r')
    values = np.lib.format.open_memmap(args.out / 'embeddings.npy', mode='w+', dtype=np.float32, shape=reference.shape)
    start, last = time.perf_counter(), time.perf_counter()
    maximum, changed = 0., 0
    with torch.no_grad():
        for i in range(0, len(pool), 64):
            guard.check()
            idx, age, typ = pool.batch(np.arange(i, min(i+64, len(pool))), args.device)
            with common.autocast(args):
                z = common.last_hidden(model, idx, age, typ).float().cpu().numpy()
            values[i:i+len(z)] = z
            delta = np.abs(z - reference[i:i+len(z)])
            maximum = max(maximum, float(delta.max()))
            changed += int(np.any(delta != 0, axis=1).sum())
            if changed:
                common.write_json(args.out / 'parity_failure.json', dict(rows_checked=i+len(z), changed=changed, max_error=maximum))
                raise ValueError('CUDA risk embeddings differ from historical values; diagnose before continuing')
            if time.perf_counter()-last > 20 or i+len(z) == len(pool):
                elapsed = time.perf_counter()-start
                tracker.log({'progress/patients': i+len(z), 'time/extraction_eta_seconds': elapsed/(i+len(z))*(len(pool)-i-len(z)), 'audit/max_embedding_error': maximum})
                common.write_json(args.out / 'progress.json', dict(patients=i+len(z), total=len(pool), seconds=elapsed))
                print(json.dumps(dict(patients=i+len(z), total=len(pool), seconds=elapsed)), flush=True)
                last = time.perf_counter()
    values.flush()
    del values
    manifest = dict(parity_exact=changed == 0, maximum_absolute_error=maximum, patients=len(pool),
                    model_sha256=info['model_sha256'], bf16=True, batch_size=64, sources=sources,
                    embeddings_sha256=sha256(args.out / 'embeddings.npy'), seconds=time.perf_counter()-start,
                    extraction_code_sha256={n: sha256(Path(__file__).with_name(n)) for n in ('risk_cache.py', 'common.py', 'data.py')})
    common.write_json(args.out / 'cache.json', manifest)
    guard.check(force=True)
    tracker.finish()


if __name__ == '__main__':
    main()
