"""Shared plumbing for the task scripts: arguments, model loading, hidden states, gates, provenance."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RELEASE = REPO / 'releases/fermat_snuh_20261021'
sys.path.insert(0, str(DEFAULT_RELEASE))
from fermat_release.model import Fermat, FermatConfig  # noqa: E402
from fermat_release.inference import _hidden  # noqa: E402
from lora import (inject_lora, lora_parameters, lora_state_dict, load_lora_state_dict,  # noqa: E402
                  count_parameters, snapshot_parameters, assert_base_unchanged, TARGET_SETS)

ARMS = ('head_only', 'lora')
from gates import PROTOCOL_VERSION, PROTOCOL_TARGETS, GATE_METRICS, validate_gate
# v1 local recipe constants (scripts/run_next_record_bundle.py, train_task30_d3_comparators.py).
EMBEDDING_SD_FLOOR = 1e-4
RECORD_HEAD_WEIGHT_SD = 0.01


def add_common_arguments(parser):
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE,
                        help='Release folder holding artifacts/model.pt (not in git)')
    parser.add_argument('--data', type=Path, required=True, help='Dataset root in the data.py layout')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True,
                        help='head_only reproduces the frozen representation arm inside this pipeline')
    parser.add_argument('--gate', type=Path, default=None,
                        help='gate.json from check_gate.py; the lora arm refuses to run without a matching passed gate')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16.)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--targets', choices=sorted(TARGET_SETS), default='attention')
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    parser.add_argument('--lr-head', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--weight-decay', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='AdamW betas (PyTorch default)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--patience', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--warmup-steps', type=int, default=None, help='Default 0 for head_only, 100 for lora')
    parser.add_argument('--grad-clip', type=float, default=None, help='Default: none for head_only, 1.0 for lora')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--recipe', choices=('v1', 'diagnostic_v1', 'experimental', 'lora'), default=None)
    parser.add_argument('--smoke-seed', type=int, default=1)
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Fixed random subset of training patients, shared by all arms; must be smaller than the pool')
    parser.add_argument('--train-patient-ids', type=Path, default=None,
                        help='.npy or text file of patient IDs that must all be present; exclusive with --train-subset')
    parser.add_argument('--subset-seed', type=int, default=20260910)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bf16', action='store_true', help='CUDA autocast for the transformer body')
    parser.add_argument('--bootstrap-resamples', type=int, default=None)
    parser.add_argument('--smoke-random-init', type=Path, default=None, metavar='MODEL_CONFIG_JSON',
                        help='Pipeline test only: random weights from this config, no release artifacts')
    return parser


def resolve_defaults(args, task_defaults):
    """Fill arm-dependent defaults in place and return the resolved dictionary."""
    d = dict(task_defaults[args.arm])
    d['epochs'] = d['epochs'] if args.epochs is None else args.epochs
    d['patience'] = d['patience'] if args.patience is None else args.patience
    d['lr_head'] = d['lr_head'] if args.lr_head is None else args.lr_head
    d['weight_decay'] = d['weight_decay'] if args.weight_decay is None else args.weight_decay
    args.warmup_steps = (0 if args.arm == 'head_only' else 100) if args.warmup_steps is None else args.warmup_steps
    args.grad_clip = (None if args.arm == 'head_only' else 1.0) if args.grad_clip is None else args.grad_clip
    if args.grad_accum < 1 or args.batch_size < 1:
        raise ValueError('batch size and gradient accumulation must be positive')
    if args.bf16 and not str(args.device).startswith('cuda'):
        raise ValueError('--bf16 requires a CUDA device; CPU runs use float32')
    if any(not np.isfinite(d[k]) or d[k] <= 0 for k in ('epochs', 'patience', 'lr_head')) or not np.isfinite(d['weight_decay']) or d['weight_decay'] < 0:
        raise ValueError('Invalid training hyperparameters')
    if args.warmup_steps < 0 or (args.grad_clip is not None and args.grad_clip <= 0):
        raise ValueError('Invalid warm-up or clipping setting')
    if not np.isfinite(args.lr_lora) or args.lr_lora <= 0 or not all(0 <= b < 1 for b in args.betas):
        raise ValueError('Invalid adapter learning rate or AdamW betas')
    return d


# ----------------------------------------------------------------------------
# Reproduction gate
# ----------------------------------------------------------------------------

def gate_identity(task, disease, model_info, splits, label_hash, train_rows, selection_mode, dataset_metadata, precision='float32'):
    """Everything a gate must agree on with the run that uses it."""
    from data import dataset_content_hash
    hash_started = time.perf_counter()
    data_hash = dataset_content_hash(splits, task, disease)
    hash_seconds = time.perf_counter() - hash_started
    return dict(
        dataset_content_sha256=data_hash, dataset_hash_seconds=hash_seconds, precision=precision,
        protocol_version=PROTOCOL_VERSION,
        task=task,
        disease=disease,
        smoke=bool(model_info.get('smoke', False)),
        model_sha256=model_info.get('model_sha256'),
        dataset_metadata_sha256=sha256_of(json.dumps(dataset_metadata, sort_keys=True).encode()),
        validation_patient_ids_sha256=sha256_of(splits['validation'].patient_id),
        test_patient_ids_sha256=sha256_of(splits['test'].patient_id),
        label_definition_sha256=label_hash,
        training_selection=selection_mode,
        n_train=int(len(train_rows)),
        train_patient_ids_sha256=sha256_of(splits['train'].patient_id[train_rows]),
    )


def require_gate(args, identity):
    """The LoRA arm may only start after a matching head-only reproduction gate passed."""
    if args.arm != 'lora':
        return None
    if args.gate is None:
        raise SystemExit('The lora arm requires --gate gate.json produced by check_gate.py (PROTOCOL.md Section 4/5)')
    gate = json.loads(Path(args.gate).read_text())
    reasons = validate_gate(gate, identity)
    if reasons:
        raise SystemExit('Reproduction gate rejected for this run:\n  - ' + '\n  - '.join(reasons))
    return gate


# ----------------------------------------------------------------------------
# Model, data selection, features
# ----------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    if args.smoke_random_init is not None:
        config = FermatConfig(**json.loads(Path(args.smoke_random_init).read_text()))
        set_seed(args.smoke_seed)
        model = Fermat(config)
        return model.to(args.device).eval(), dict(smoke=True, config=str(args.smoke_random_init), model_sha256=sha256_of(json.dumps(dict(config=vars(config), seed=args.smoke_seed), sort_keys=True).encode()))
    from fermat_release import load_release
    model = load_release(args.release, args.device)
    manifest = json.loads((args.release / 'config/artifact_manifest.json').read_text())
    return model, dict(smoke=False, model_sha256=manifest['artifacts/model.pt']['sha256'])


def check_split_overlap(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = np.intersect1d(splits[a].patient_id, splits[b].patient_id)
            if len(shared):
                raise ValueError(f'{len(shared)} patients appear in both {a} and {b}')


def select_training_rows(split, args):
    """Exact patient list when given, otherwise a fixed random subset that must fit the pool."""
    from data import fixed_subset
    if args.train_patient_ids is not None:
        if args.train_subset is not None:
            raise ValueError('Choose either a training patient list or a random subset')
        path = Path(args.train_patient_ids)
        wanted = (np.load(path, allow_pickle=False) if path.suffix == '.npy'
                  else np.loadtxt(path, dtype=np.int64, ndmin=1)).astype(np.int64)
        if wanted.ndim != 1 or len(wanted) == 0:
            raise ValueError('Training patient list must be nonempty and one-dimensional')
        if len(np.unique(wanted)) != len(wanted):
            raise ValueError('Training patient list contains duplicates')
        position = {int(p): i for i, p in enumerate(split.patient_id)}
        missing = [int(p) for p in wanted if int(p) not in position]
        if missing:
            raise ValueError(f'{len(missing)} listed training patients are not in the training split '
                             f'(first: {missing[:5]})')
        return np.asarray([position[int(p)] for p in wanted], dtype=np.int64), 'patient_list'
    if args.train_subset is not None and args.train_subset > len(split):
        raise ValueError(f'--train-subset {args.train_subset} exceeds the {len(split)} available training patients')
    return fixed_subset(len(split), args.train_subset, args.subset_seed), ('subset' if args.train_subset else 'all')


def autocast(args):
    if args.bf16 and str(args.device).startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def last_hidden(model, idx, age, typ):
    """Final hidden state at the last observed event under the plain causal mask.

    This is the same quantity as ``fermat_release.embedding`` (the manuscript's
    patient representation) but differentiable, so LoRA gradients can flow.
    """
    hidden = _hidden(model, idx, age, typ)
    last = (idx > 0).sum(1) - 1
    return hidden[torch.arange(len(idx), device=idx.device), last]


@torch.no_grad()
def frozen_embeddings(model, split, rows, args, progress=None):
    from data import iterate_batches
    model.eval()
    out = np.zeros((len(rows), model.config.n_embd), dtype=np.float32)
    position = 0
    for batch_rows in iterate_batches(rows, args.batch_size):
        idx, age, typ = split.batch(batch_rows, args.device)
        with autocast(args):
            h = last_hidden(model, idx, age, typ)
        out[position:position + len(batch_rows)] = h.float().cpu().numpy()
        position += len(batch_rows)
        if progress is not None:
            progress(position, len(rows))
    return out


def standardization(values, sd_floor=EMBEDDING_SD_FLOOR):
    """Training-set mean and standard deviation; the sd is floored, as in the v1 head code."""
    mean = values.mean(0)
    sd = np.maximum(values.std(0), sd_floor)
    return mean.astype(np.float32), sd.astype(np.float32)


def init_record_head(head, positives, n_patients):
    """v1 next-record head: N(0, 0.01) weights, bias = logit((positives + 0.5) / (n + 1))."""
    with torch.no_grad():
        torch.nn.init.normal_(head.weight, mean=0., std=RECORD_HEAD_WEIGHT_SD)
        p = (torch.as_tensor(positives, dtype=torch.float32, device=head.bias.device) + .5) / (n_patients + 1)
        head.bias.copy_(torch.logit(p))


def init_risk_head(head, weighted_prevalence):
    """v1 weighted logistic predictor: zero weights, bias = logit of the weighted prevalence."""
    with torch.no_grad():
        head.weight.zero_()
        p = min(max(float(weighted_prevalence), 1e-6), 1 - 1e-6)
        head.bias.fill_(float(np.log(p / (1 - p))))


def record_loss(logits, targets):
    """v1 training loss: BCE summed over output items, averaged over patients."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(1).mean()


def weighted_objective(logits, y, w):
    """Full-cohort objective: sum(w * BCE) / sum(w) over the patients given."""
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction='none')).sum() / w.sum()


def weighted_minibatch_loss(logits, y, w, mean_weight):
    """Per-patient mean of w * BCE / mean_weight.

    ``mean_weight`` is the fixed average weight over the whole training set, so
    averaging this quantity over all patients (whatever the batching) equals the
    full-cohort objective ``sum(w * BCE) / sum(w)``; per-batch weight sums are
    never used as normalisers.
    """
    return (w * F.binary_cross_entropy_with_logits(logits, y, reduction='none')).mean() / mean_weight


def prepare_adapters(model, args):
    """Inject LoRA when requested; return the parameter snapshot for the final audit."""
    snapshot = snapshot_parameters(model)
    wrapped = []
    if args.arm == 'lora':
        wrapped = inject_lora(model, rank=args.rank, alpha=args.alpha,
                              dropout=args.lora_dropout, targets=args.targets)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return snapshot, wrapped


def make_optimizer(model, head, args, lr_head, weight_decay):
    groups = [dict(params=list(head.parameters()), lr=lr_head, weight_decay=weight_decay)]
    if args.arm == 'lora':
        groups.append(dict(params=lora_parameters(model), lr=args.lr_lora, weight_decay=weight_decay))
    optimizer = torch.optim.AdamW(groups, betas=tuple(args.betas))
    base_lrs = [g['lr'] for g in optimizer.param_groups]

    def set_warmup(step):
        factor = 1. if args.warmup_steps <= 0 else min(1., (step + 1) / args.warmup_steps)
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * factor
    return optimizer, set_warmup


class Accumulator:
    """Gradient accumulation over patients.

    Each call to ``backward`` passes a per-patient mean loss and the number of
    patients it averages; the gradient of the patient *sum* is accumulated and
    divided by the total number of patients at update time. The update therefore
    equals the gradient of the mean loss over every accumulated patient, whatever
    the batch sizes and however many batches the trailing accumulation holds.
    Normalisation happens before gradient clipping.
    """

    def __init__(self, optimizer, set_warmup, args):
        self.optimizer, self.set_warmup, self.args = optimizer, set_warmup, args
        self.pending, self.count, self.steps, self.epoch_start = 0, 0, 0, 0

    def parameters(self):
        return [p for g in self.optimizer.param_groups for p in g['params']]

    def backward(self, mean_loss, n_patients):
        if n_patients <= 0 or not torch.isfinite(mean_loss).all():
            raise ValueError('Loss must be finite and patient count positive')
        (mean_loss * float(n_patients)).backward()
        self.count += int(n_patients)
        self.pending += 1
        if self.pending == self.args.grad_accum:
            self.step()

    def step(self):
        if self.pending == 0:
            return
        for p in self.parameters():
            if p.grad is not None:
                p.grad.div_(float(self.count))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
            raise FloatingPointError('Nonfinite gradient; optimizer update refused')
        self.set_warmup(self.steps)
        if self.args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending, self.count = 0, 0
        self.steps += 1

    def end_epoch(self):
        self.step()
        if self.steps == self.epoch_start:
            raise RuntimeError('No optimizer update was performed in this epoch')
        self.epoch_start = self.steps


# ----------------------------------------------------------------------------
# State, provenance, outputs
# ----------------------------------------------------------------------------

def trainable_state(model, head, preprocessing, config):
    """Everything needed to reproduce predictions without the training run."""
    return dict(lora=lora_state_dict(model) if any(True for _ in lora_parameters(model)) else {},
                head=copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}),
                preprocessing={k: torch.as_tensor(np.asarray(v)).clone() for k, v in preprocessing.items()},
                config=dict(config))


def restore_state(model, head, state):
    if state['lora']:
        load_lora_state_dict(model, state['lora'])
    head.load_state_dict(state['head'])


def load_trained(path, model, args_like):
    """Rebuild adapters and head from ``trainable_state.pt`` for prediction scripts."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = state['config']
    if config['arm'] == 'lora':
        inject_lora(model, rank=config['rank'], alpha=config['alpha'], dropout=0., targets=config['targets'])
    head = torch.nn.Linear(config['head_in'], config['head_out']).to(args_like.device)
    restore_state(model, head, state)
    model.eval()
    head.eval()
    return state, head


def git_commit():
    # A Windows-created worktree contains a Windows .git pointer that Linux git
    # cannot resolve. WSL can query the same checkout through Windows git.
    for executable in ('git', 'git.exe'):
        try:
            return subprocess.run([executable, 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                  text=True, check=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def sha256_of(data):
    if isinstance(data, (bytes, bytearray)):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def sha256_of_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def fingerprint(args, model_info, data_metadata, extra=None):
    record = dict(
        script=Path(sys.argv[0]).name,
        arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        git_commit=git_commit(),
        protocol_version=PROTOCOL_VERSION, recipe=args.recipe,
        model=model_info,
        dataset=data_metadata,
        torch=torch.__version__,
        python=platform.python_version(),
        device=str(args.device),
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        pid=os.getpid(),
    )
    if extra:
        record.update(extra)
    return record


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=float))


def finish_audit(model, snapshot, head, state, out_dir):
    """Prove the release tensors are untouched and persist only the trainable pieces."""
    assert_base_unchanged(model, snapshot)
    torch.save(state, Path(out_dir) / 'trainable_state.pt')
    counts = count_parameters(model)
    adapters = counts['trainable']
    head_count = sum(p.numel() for p in head.parameters())
    return dict(frozen_base_unchanged=True,
                base_parameters=counts['total'] - adapters,
                adapter_parameters=adapters,
                adapter_fraction_of_base=adapters / (counts['total'] - adapters),
                head_parameters=head_count,
                trained_parameters_total=adapters + head_count)


def read_baseline_csv(path, column):
    import csv
    values = {}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            patient_id, value = int(row['patient_id']), float(row[column])
            if patient_id in values or not np.isfinite(value):
                raise ValueError('Baseline CSV has duplicate IDs or nonfinite values')
            values[patient_id] = value
    return values


def align_baseline(patient_ids, baseline, label):
    missing = [int(p) for p in patient_ids if int(p) not in baseline]
    if missing:
        raise ValueError(f'{label}: {len(missing)} test patients missing from the baseline file')
    extra = set(baseline) - set(int(p) for p in patient_ids)
    if extra:
        raise ValueError(f'{label}: baseline file has {len(extra)} patients not in the test split')
    return np.asarray([baseline[int(p)] for p in patient_ids])
