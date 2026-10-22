"""Dataset contract for the LoRA baseline. Patient-level files never enter git.

A dataset is a directory::

    <root>/dataset.json                  free-form provenance (source, index rule, cutoff)
    <root>/classes.npy                   int64 [C]  prediction vocabulary (model token IDs)
    <root>/<split>/idx.npy               int64 [N, T]  model token IDs, right-padded with 0
    <root>/<split>/age.npy               float32 [N, T]  age in days, -10000 at padding
    <root>/<split>/token_type.npy        int64 [N, T]  token types, 0 at padding
    <root>/<split>/patient_id.npy        int64 [N]
    record task
    <root>/<split>/label_indptr.npy      int64 [N + 1]  CSR row pointer
    <root>/<split>/label_tokens.npy      int64 [nnz]    model token IDs on the target date
    disease-risk task
    <root>/<split>/clinical.npy          float32 [N, 4]  age, sex, log rows, log active days
    <root>/<split>/outcome_<disease>.npy int8 [N]  1 case, 0 noncase, -1 unobserved/ineligible
    <root>/<split>/weight_<disease>.npy  float32 [N]  clipped inverse observation probability

Splits are ``train``, ``validation`` and ``test``. Prefixes are the last at most
512 observed events through the index date, exactly as supplied to the
representation-based predictors; tensors follow INPUT_SPEC.md (model ID =
stored ID + 1, padding ID 0 with age -10000). Building these files from the
source cohorts is the responsibility of the internal pipeline; this module only
validates and serves them.
"""
import json
from pathlib import Path
import numpy as np
import torch

SPLITS = ('train', 'validation', 'test')
PAD_AGE = -10000.


def _load(path, dtype, ndim):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.dtype != dtype or array.ndim != ndim:
        raise ValueError(f'{path}: expected {dtype} with {ndim} dimensions, got {array.dtype} {array.shape}')
    return array


class PrefixSplit:
    def __init__(self, root, split, block_size=512, diseases=()):
        self.root, self.split = Path(root), split
        folder = self.root / split
        self.idx = _load(folder / 'idx.npy', np.int64, 2)
        self.age = _load(folder / 'age.npy', np.float32, 2)
        self.token_type = _load(folder / 'token_type.npy', np.int64, 2)
        self.patient_id = _load(folder / 'patient_id.npy', np.int64, 1)
        n, t = self.idx.shape
        if self.age.shape != (n, t) or self.token_type.shape != (n, t) or len(self.patient_id) != n:
            raise ValueError(f'{folder}: inconsistent array shapes')
        if t > block_size:
            raise ValueError(f'{folder}: prefixes longer than block size {block_size}')
        if len(np.unique(self.patient_id)) != n:
            raise ValueError(f'{folder}: duplicate patient IDs')
        self.lengths = np.asarray((self.idx > 0).sum(1), dtype=np.int64)
        self._validate_prefixes()
        self.labels = None
        if (folder / 'label_indptr.npy').exists():
            self.label_indptr = _load(folder / 'label_indptr.npy', np.int64, 1)
            self.label_tokens = _load(folder / 'label_tokens.npy', np.int64, 1)
            if len(self.label_indptr) != n + 1 or self.label_indptr[-1] != len(self.label_tokens):
                raise ValueError(f'{folder}: CSR label arrays are inconsistent')
            if (np.diff(self.label_indptr) < 1).any():
                raise ValueError(f'{folder}: every record-task patient needs an observed target')
            self.labels = True
        self.clinical = None
        if (folder / 'clinical.npy').exists():
            self.clinical = _load(folder / 'clinical.npy', np.float32, 2)
            if self.clinical.shape != (n, 4):
                raise ValueError(f'{folder}: clinical must be [N, 4]')
        self.outcomes = {}
        for disease in diseases:
            y = _load(folder / f'outcome_{disease}.npy', np.int8, 1)
            w = _load(folder / f'weight_{disease}.npy', np.float32, 1)
            if len(y) != n or len(w) != n:
                raise ValueError(f'{folder}: outcome arrays for {disease} have the wrong length')
            if not np.isin(y, (-1, 0, 1)).all() or (w < 0).any() or not np.isfinite(w).all():
                raise ValueError(f'{folder}: invalid outcome or weight values for {disease}')
            if ((y == -1) & (w > 0)).any() or ((y >= 0) & (w == 0)).any():
                raise ValueError(f'{folder}: observed outcomes must carry positive weights for {disease}')
            self.outcomes[disease] = (y, w)

    def _validate_prefixes(self):
        valid = self.idx > 0
        if (self.lengths < 1).any():
            raise ValueError(f'{self.split}: every prefix needs at least one observed event')
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError(f'{self.split}: padding must be on the right')
        if (self.idx < 0).any() or (self.token_type[valid] <= 0).any() or (self.token_type[~valid] != 0).any():
            raise ValueError(f'{self.split}: invalid token IDs or types')
        age_valid = np.where(valid, self.age, np.nan)
        if not np.isfinite(age_valid[valid]).all() or (self.age[~valid] != PAD_AGE).any():
            raise ValueError(f'{self.split}: ages must be finite at events and {PAD_AGE} at padding')
        if (np.nan_to_num(np.diff(age_valid, axis=1), nan=0.) < 0).any():
            raise ValueError(f'{self.split}: ages must be nondecreasing')

    def __len__(self):
        return len(self.patient_id)

    def label_lists(self, rows):
        return [self.label_tokens[self.label_indptr[r]:self.label_indptr[r + 1]] for r in rows]

    def batch(self, rows, device):
        """Tensors trimmed to the longest prefix in the batch."""
        rows = np.asarray(rows)
        width = int(self.lengths[rows].max())
        idx = torch.from_numpy(np.ascontiguousarray(self.idx[rows, :width]))
        age = torch.from_numpy(np.ascontiguousarray(self.age[rows, :width]))
        typ = torch.from_numpy(np.ascontiguousarray(self.token_type[rows, :width]))
        return idx.to(device), age.to(device), typ.to(device)


def multi_hot(label_lists, class_index, n_classes):
    """Dense float targets; raises if a label is outside the prediction vocabulary."""
    target = torch.zeros(len(label_lists), n_classes)
    for row, tokens in enumerate(label_lists):
        columns = class_index[np.asarray(tokens, dtype=np.int64)]
        if (columns < 0).any():
            raise ValueError('Target token outside the prediction vocabulary')
        target[row, torch.from_numpy(columns)] = 1.
    return target


def class_lookup(classes, vocab_size):
    """Map model token ID -> column, or -1 when the token is not predicted."""
    classes = np.asarray(classes, dtype=np.int64)
    if len(np.unique(classes)) != len(classes) or (classes < 0).any() or (classes >= vocab_size).any():
        raise ValueError('Prediction vocabulary must be unique token IDs inside the model vocabulary')
    index = np.full(vocab_size, -1, dtype=np.int64)
    index[classes] = np.arange(len(classes))
    return index


def fixed_subset(n, size, seed):
    """Deterministic patient subset shared by every arm of a comparison."""
    if size is None or size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def iterate_batches(rows, batch_size, rng=None):
    rows = np.asarray(rows)
    if rng is not None:
        rows = rows[rng.permutation(len(rows))]
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def read_metadata(root):
    path = Path(root) / 'dataset.json'
    return json.loads(path.read_text()) if path.exists() else {}
