"""Write a random dataset in the data.py layout for pipeline smoke tests only.

Nothing here resembles patients. The output exercises file validation, batching,
training loops, metrics and bootstraps, so the real cohort export can be dropped
in without changing any script.
"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--vocab-size', type=int, default=24435)
parser.add_argument('--n-classes', type=int, default=200)
parser.add_argument('--patients', type=int, nargs=3, default=(96, 32, 32), metavar=('TRAIN', 'VAL', 'TEST'))
parser.add_argument('--max-length', type=int, default=48)
parser.add_argument('--diseases', default='ckd,fatty_liver,diabetes')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
diseases = [d for d in args.diseases.split(',') if d]

# Reserve model IDs 0 (pad), 1 (registry pad), 2 (NO_EVENT); SEX tokens get type 7.
clinical_pool = np.arange(10, args.vocab_size)
classes = np.sort(rng.choice(clinical_pool, size=args.n_classes, replace=False))
token_type_of = rng.integers(1, 4, size=args.vocab_size)
sex_tokens = np.array([3, 4])
args.out.mkdir(parents=True, exist_ok=True)
np.save(args.out / 'classes.npy', classes.astype(np.int64))

for split, n in zip(('train', 'validation', 'test'), args.patients):
    t = args.max_length
    idx = np.zeros((n, t), dtype=np.int64)
    age = np.full((n, t), -10000., dtype=np.float32)
    typ = np.zeros((n, t), dtype=np.int64)
    indptr, tokens = [0], []
    for i in range(n):
        length = int(rng.integers(3, t + 1))
        idx[i, 0], typ[i, 0], age[i, 0] = rng.choice(sex_tokens), 7, 0.
        body = rng.choice(np.concatenate([clinical_pool[:50], classes]), size=length - 1)
        idx[i, 1:length] = body
        typ[i, 1:length] = token_type_of[body]
        gaps = rng.choice([0, 0, 1, 30, 365], size=length - 1)
        age[i, 1:length] = 15000 + np.cumsum(gaps)
        labels = np.unique(rng.choice(classes, size=int(rng.integers(1, 6))))
        tokens.extend(labels.tolist())
        indptr.append(len(tokens))
    folder = args.out / split
    folder.mkdir(exist_ok=True)
    np.save(folder / 'idx.npy', idx)
    np.save(folder / 'age.npy', age)
    np.save(folder / 'token_type.npy', typ)
    np.save(folder / 'patient_id.npy', (np.arange(n) + {'train': 0, 'validation': 10**6, 'test': 2 * 10**6}[split]).astype(np.int64))
    np.save(folder / 'label_indptr.npy', np.asarray(indptr, dtype=np.int64))
    np.save(folder / 'label_tokens.npy', np.asarray(tokens, dtype=np.int64))
    clinical = np.column_stack([age[np.arange(n), (idx > 0).sum(1) - 1] / 365.25, rng.integers(0, 2, n),
                                np.log1p(rng.integers(1, 500, n)), np.log1p(rng.integers(1, 60, n))]).astype(np.float32)
    np.save(folder / 'clinical.npy', clinical)
    for disease in diseases:
        y = rng.choice([-1, 0, 0, 0, 1], size=n).astype(np.int8)
        w = np.where(y >= 0, rng.uniform(1., 2., n), 0.).astype(np.float32)
        np.save(folder / f'outcome_{disease}.npy', y)
        np.save(folder / f'weight_{disease}.npy', w)

(args.out / 'dataset.json').write_text(json.dumps(dict(
    source='synthetic smoke fixture from make_synthetic_dataset.py; not patient data',
    seed=args.seed, diseases=diseases, n_classes=args.n_classes), indent=2))
print(json.dumps(dict(out=str(args.out), patients=args.patients, classes=args.n_classes)))
