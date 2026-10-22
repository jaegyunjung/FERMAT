"""Export the saved v1 next-record cohort without DB access or new generation.

Requires pandas/pyarrow in addition to training dependencies. Writes private
patient arrays outside Git; source artifacts are read-only. This is preparation,
not a reproduction gate. Frozen embedding parity still requires a model check.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from data import PrefixSplit, dataset_content_hash


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    value = sha256(path)
    if value != expected:
        raise ValueError(f'Source hash mismatch: {path}')
    return value


def pack_prefixes(rows, offsets, selected, dense_ids, index_ages):
    """Original collate_rows mapping: stored token +1; age days; type unchanged."""
    n = len(selected)
    idx = np.zeros((n, 512), dtype=np.int64)
    age = np.full((n, 512), -10000., dtype=np.float32)
    typ = np.zeros((n, 512), dtype=np.int64)
    for j, row in enumerate(selected):
        events = rows[offsets[row]:offsets[row + 1]]
        if not 1 <= len(events) <= 512:
            raise ValueError('Saved prefix length must be 1..512; no silent truncation')
        if not np.all(events[:, 0] == dense_ids[row]) or np.any(events[:, 1] > index_ages[row]):
            raise ValueError('Saved prefix patient identity or index cutoff mismatch')
        idx[j, :len(events)] = events[:, 2].astype(np.int64) + 1
        age[j, :len(events)] = events[:, 1].astype(np.float32)
        typ[j, :len(events)] = events[:, 3].astype(np.int64)
    return idx, age, typ


def prepare(pilot, record, out):
    started = time.perf_counter()
    pilot, record, out = map(lambda p: Path(p).resolve(), (pilot, record, out))
    if any((parent / '.git').exists() for parent in (out, *out.parents)):
        raise ValueError('Patient-level export must be outside a Git checkout')
    staging = out.with_name(out.name + '.preparing')
    if out.exists() or staging.exists():
        raise ValueError('Output or staging directory already exists; choose a fresh destination')
    import pandas as pd
    prepared = json.loads((record / 'prepared.json').read_text())
    provenance = {}
    for name in ('targets.parquet', 'prefixes.npz', 'embeddings.npy', 'prepared_manifest.json'):
        registered = [v for k, v in prepared['inputs'].items() if Path(k).name == name]
        if len(registered) != 1:
            raise ValueError(f'Missing or ambiguous registered source: {name}')
        provenance['pilot/' + name] = verify(pilot / name, registered[0])
    for name in ('labels.parquet', 'classes.npy'):
        provenance['record/' + name] = verify(record / name, prepared['outputs'][name])
    result50 = json.loads((record / 'paths50/results.json').read_text())
    provenance['record/paths50/predictions.parquet'] = verify(
        record / 'paths50/predictions.parquet', result50['hashes']['predictions'])
    targets, labels = pd.read_parquet(pilot / 'targets.parquet'), pd.read_parquet(record / 'labels.parquet')
    for column in ('person_id', 'split'):
        if not np.array_equal(targets[column], labels[column]):
            raise ValueError('Saved target/label ordering mismatch')
    if not targets.person_id.is_unique or set(targets.split) != {'train', 'val', 'test'}:
        raise ValueError('Invalid saved cohort identity or splits')
    embeddings = np.load(pilot / 'embeddings.npy', mmap_mode='r')
    if embeddings.shape != (len(targets), 640) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid saved embeddings')
    with np.load(pilot / 'prefixes.npz') as packed:
        rows, offsets = packed['rows'], packed['offsets']
    if (len(offsets) != len(targets) + 1 or offsets[0] != 0 or offsets[-1] != len(rows)
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Invalid packed prefix offsets')
    classes = np.load(record / 'classes.npy')
    staging.mkdir(parents=True)
    np.save(staging / 'classes.npy', classes)
    counts = {}
    for source, split in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        chosen = np.flatnonzero((targets.split == source) & (labels.bundle_size > 0))
        if len(chosen) != prepared['split_evaluable'][source]:
            raise ValueError('Published cohort count mismatch')
        folder = staging / split
        folder.mkdir()
        idx, age, typ = pack_prefixes(rows, offsets, chosen, targets.patient_id_dense.to_numpy(),
                                     targets.index_age_days.to_numpy())
        tokens = [np.asarray(labels.token_ids.iloc[r], dtype=np.int64) for r in chosen]
        flat = np.concatenate(tokens)
        if not np.isin(flat, classes).all():
            raise ValueError('Target outside saved prediction vocabulary')
        arrays = dict(idx=idx, age=age, token_type=typ,
                      patient_id=targets.person_id.to_numpy(np.int64)[chosen],
                      label_indptr=np.concatenate(([0], np.cumsum([len(t) for t in tokens]))).astype(np.int64),
                      label_tokens=flat, frozen_embeddings=embeddings[chosen])
        for name, values in arrays.items():
            np.save(folder / (name + '.npy'), values)
        counts[split] = len(chosen)
    # The original record scaler used ALL training patients, including those
    # without an observed next-record target; head optimization used only 18,474.
    chosen = np.flatnonzero(targets.split == 'train')
    folder = staging / 'preprocessing_train'
    folder.mkdir()
    idx, age, typ = pack_prefixes(rows, offsets, chosen, targets.patient_id_dense.to_numpy(),
                                 targets.index_age_days.to_numpy())
    for name, values in dict(idx=idx, age=age, token_type=typ,
                             patient_id=targets.person_id.to_numpy(np.int64)[chosen],
                             frozen_embeddings=embeddings[chosen]).items():
        np.save(folder / (name + '.npy'), values)
    source_scale = np.load(record / 'standardization.npz')
    reference = embeddings[chosen]
    if not (np.array_equal(reference.mean(0), source_scale['mean']) and
            np.array_equal(np.maximum(reference.std(0), 1e-4), source_scale['sd'])):
        raise ValueError('All-training scaler does not exactly match saved v1 statistics')
    provenance['record/standardization.npz'] = sha256(record / 'standardization.npz')
    baseline = pd.read_parquet(record / 'paths50/predictions.parquet')
    ids = np.load(staging / 'test/patient_id.npy')
    if not baseline.person_id.is_unique or set(baseline.person_id) != set(ids):
        raise ValueError('50-path baseline cohort mismatch')
    recall = baseline.set_index('person_id').loc[ids, 'recall_50'].to_numpy(float)
    if not np.isfinite(recall).all() or np.any((recall < 0) | (recall > 1)):
        raise ValueError('Invalid saved recall')
    if abs(recall.mean() - result50['scores']['50']['recall_at_10']) > 1e-12:
        raise ValueError('50-path aggregate mismatch')
    # Training metrics use percentage points, while the saved v1 table uses fractions.
    pd.DataFrame(dict(patient_id=ids, recall_at_10=100 * recall)).to_csv(
        staging / 'baseline_rollout50.csv', index=False)
    selection = json.loads((record / 'model_selection.json').read_text())
    metadata = dict(task='record', source='saved next_record_bundle_20260908_v1',
                    preprocessing_train_patients=len(chosen),
                    source_sha256=provenance, counts=counts, selected_v1_head=selection['chosen'],
                    prefix_rule='Saved observed prefix through index date, stored token ID + 1; right padding',
                    label_rule='Saved first future DX/RX/PX day bundle; nonempty target cohort',
                    frozen_embeddings='Reference only; historical CUDA BF16. Training does not consume this cache.',
                    preparation_only=True)
    (staging / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    splits = {s: PrefixSplit(staging, s) for s in counts}
    splits['preprocessing_train'] = PrefixSplit(staging, 'preprocessing_train')
    hash_start = time.perf_counter()
    digest = dataset_content_hash(splits, 'record', None)
    report = dict(status='PREPARED_NOT_REPRODUCED', counts=counts, classes=len(classes),
                  preprocessing_train_patients=len(chosen), scaler_matches_saved_v1_exactly=True,
                  dataset_content_sha256=digest, hash_seconds=time.perf_counter() - hash_start,
                  export_bytes=sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()),
                  selected_v1_head=selection['chosen'], rollout50_recall_percent=float(100 * recall.mean()),
                  source_sha256=provenance, elapsed_seconds=time.perf_counter() - started)
    (staging / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    del splits  # close memory maps before the directory rename on Windows
    staging.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.pilot, args.record, args.out), indent=2))


if __name__ == '__main__':
    main()
