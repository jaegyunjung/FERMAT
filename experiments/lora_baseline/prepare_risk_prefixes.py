"""Export original D3 prefixes and original 64-patient extraction order locally."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_record_v1 import sha256, verify
from data import PrefixSplit


def prepare(root, outcomes, out):
    root, outcomes, out = map(lambda p: Path(p).resolve(), (root, outcomes, out))
    if any((p / '.git').exists() for p in (out, *out.parents)):
        raise ValueError('Private export must be outside Git')
    staging = out.with_name(out.name + '.preparing')
    if out.exists() or staging.exists():
        raise ValueError('Choose a new destination')
    started = time.perf_counter()
    m = json.loads((root / 'embeddings/manifest.json').read_text())
    inputs = json.loads((root / 'comparators/manifest.json').read_text())['inputs']
    old = json.loads((outcomes / 'preparation.json').read_text())
    for rel, digest in old['array_sha256'].items():
        verify(outcomes / rel, digest)
    targets = pd.read_parquet(m['target_path']).sort_values(['split', 'patient_id_dense']).reset_index(drop=True)
    verify(m['target_path'], m['target_sha256'])
    verify(inputs['preindex']['path'], inputs['preindex']['sha256'])
    verify(inputs['embeddings']['path'], inputs['embeddings']['sha256'])
    cohort = pd.read_parquet(inputs['preindex']['path']).set_index('person_id')
    embeddings = pd.read_parquet(inputs['embeddings']['path'])
    if not np.array_equal(embeddings.person_id, targets.person_id) or not targets.person_id.is_unique:
        raise ValueError('Original extraction order mismatch')
    if m['batch_size'] != 64 or m['shard_size'] % 64 or not embeddings.has_embedding_sequence.all():
        raise ValueError('Unsupported original batch context')
    data_dir = Path(m['data_dir'])
    bin_hashes = json.loads((data_dir / 'sha256.json').read_text())
    provenance = {}
    for name in ('patient_id_map.parquet', 'train.bin', 'val.bin', 'test.bin'):
        provenance[name] = verify(data_dir / name, bin_hashes[name])
        print(json.dumps(dict(source_verified=name)), flush=True)
    mapping = pd.read_parquet(data_dir / 'patient_id_map.parquet')
    # Check the original mapping, rather than assuming dense identifiers are person IDs.
    if not {'person_id', 'patient_id_dense'}.issubset(mapping.columns):
        raise ValueError('Unexpected patient-map schema')
    mapped = mapping.set_index('person_id').loc[targets.person_id, 'patient_id_dense'].to_numpy()
    if not np.array_equal(mapped, targets.patient_id_dense):
        raise ValueError('Dense patient mapping mismatch')
    staging.mkdir(parents=True)
    pool = staging / 'embedding_pool'
    pool.mkdir()
    arrays = {name: np.lib.format.open_memmap(pool / (name + '.npy'), mode='w+', dtype=dtype, shape=(len(targets), 512))
              for name, dtype in [('idx', np.int64), ('age', np.float32), ('token_type', np.int64)]}
    lengths = np.zeros(len(targets), dtype=np.int32)
    for split in ('train', 'val', 'test'):
        rows = np.memmap(data_dir / f'{split}.bin', dtype=np.uint32, mode='r').reshape(-1, 4)
        chosen = np.flatnonzero(targets.split.eq(split))
        dense = targets.patient_id_dense.to_numpy()[chosen]
        starts = np.searchsorted(rows[:, 0], dense, side='left')
        ends = np.searchsorted(rows[:, 0], dense, side='right')
        for i, start, end in zip(chosen, starts, ends):
            events = rows[start:end]
            if not len(events) or not np.all(events[:, 0] == targets.patient_id_dense.iloc[i]):
                raise ValueError('Missing or mixed patient prefix')
            events = events[events[:, 1] <= targets.index_age_days.iloc[i]][-512:]
            n = len(events)
            if not n or n != embeddings.sequence_length_pre_index.iloc[i]:
                raise ValueError('Prefix length differs from original inference')
            if n != cohort.loc[targets.person_id.iloc[i], 'prefix_rows_block512']:
                raise ValueError('Prefix differs from original cohort audit')
            arrays['idx'][i] = 0
            arrays['age'][i] = -10000
            arrays['token_type'][i] = 0
            arrays['idx'][i, :n] = events[:, 2].astype(np.int64) + 1
            arrays['age'][i, :n] = events[:, 1]
            arrays['token_type'][i, :n] = events[:, 3]
            lengths[i] = n
        print(json.dumps(dict(split=split, prefixes=len(chosen))), flush=True)
    for a in arrays.values():
        a.flush()
    np.save(pool / 'patient_id.npy', targets.person_id.to_numpy(np.int64))
    np.save(pool / 'frozen_embeddings.npy', embeddings[sorted(c for c in embeddings if c.startswith('emb_'))].to_numpy(np.float32))
    position = pd.Series(np.arange(len(targets)), index=targets.person_id)
    for name in ('train', 'validation', 'test'):
        dest = staging / name
        dest.mkdir()
        for file in (outcomes / name).glob('*.npy'):
            shutil.copyfile(file, dest / file.name)
        ids = np.load(dest / 'patient_id.npy')
        indices = position.loc[ids].to_numpy()
        expected_split = 'val' if name == 'validation' else name
        if not targets.split.iloc[indices].eq(expected_split).all():
            raise ValueError('Split mapping changed')
        np.save(dest / 'pool_rows.npy', indices)
        for attr, array in arrays.items():
            target = np.lib.format.open_memmap(dest / (attr + '.npy'), mode='w+', dtype=array.dtype, shape=(len(ids), 512))
            for start in range(0, len(ids), 1024):
                target[start:start+1024] = array[indices[start:start+1024]]
            target.flush()
            del target
        PrefixSplit(staging, name, diseases=('ckd', 'fatty_liver', 'diabetes'))
    PrefixSplit(staging, 'embedding_pool')
    metadata = dict(task='risk', version='risk-prefix-v1', source_hashes=provenance,
                    outcomes_manifest_sha256=sha256(outcomes / 'preparation.json'),
                    embedding_manifest_sha256=sha256(root / 'embeddings/manifest.json'),
                    embedding_batch_size=64, embedding_shard_size=m['shard_size'],
                    historical_checkpoint_sha256=m['checkpoint_sha256'],
                    prefix_rule='age <= index age; last 512; stored token + 1; original split/dense order',
                    label_rule='Original five-calendar-year disease-before-death definitions',
                    weight_provenance=old['weight_provenance'])
    (staging / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    hashes = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*.npy'))}
    report = dict(status='EXPORTED_NOT_CUDA_VERIFIED', patients=len(targets), source_hashes=provenance,
                  array_sha256=hashes, seconds=time.perf_counter()-started,
                  bytes=sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()))
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    del arrays
    staging.rename(out)
    print(json.dumps({k:v for k,v in report.items() if k != 'array_sha256'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'outcomes', 'out'):
        p.add_argument('--'+flag, type=Path, required=True)
    args = p.parse_args()
    prepare(args.root, args.outcomes, args.out)
