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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RELEASE = REPO / 'releases/fermat_snuh_20261021'
sys.path.insert(0, str(DEFAULT_RELEASE))
from fermat_release.model import Fermat, FermatConfig  # noqa: E402
from fermat_release.inference import _hidden  # noqa: E402
from lora import (inject_lora, lora_parameters, lora_state_dict, load_lora_state_dict,  # noqa: E402
                  count_parameters, snapshot_parameters, assert_base_unchanged, TARGET_SETS)

ARMS = ('head_only', 'lora')


def add_common_arguments(parser):
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE,
                        help='Release folder holding artifacts/model.pt (not in git)')
    parser.add_argument('--data', type=Path, required=True, help='Dataset root in the data.py layout')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True,
                        help='head_only reproduces the frozen representation arm inside this pipeline')
    parser.add_argument('--gate', type=Path, default=None,
                        help='gate.json from check_gate.py; the lora arm refuses to run without a passed gate')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16.)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--targets', choices=sorted(TARGET_SETS), default='attention')
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    parser.add_argument('--lr-head', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--weight-decay', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='AdamW betas (PyTorch default)')
    parser.add_argument('--head-bias-init', choices=('frequency', 'zero'), default='frequency',
                        help='frequency: bias = logit of the training-set target prevalence')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--patience', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--warmup-steps', type=int, default=None, help='Default 0 for head_only, 100 for lora')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Fixed random subset of training patients, shared by all arms; must be smaller than the pool')
    parser.add_argument('--train-patient-ids', type=Path, default=None,
                        help='.npy or text file of patient IDs that must all be present; overrides --train-subset')
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
    d['epochs'] = args.epochs or d['epochs']
    d['patience'] = args.patience or d['patience']
    d['lr_head'] = args.lr_head or d['lr_head']
    d['weight_decay'] = d['weight_decay'] if args.weight_decay is None else args.weight_decay
    args.warmup_steps = (0 if args.arm == 'head_only' else 100) if args.warmup_steps is None else args.warmup_steps
    if args.grad_accum < 1 or args.batch_size < 1:
        raise ValueError('batch size and gradient accumulation must be positive')
    return d


def require_gate(args):
    """The LoRA arm may only start after the head-only reproduction gate passed."""
    if args.arm != 'lora':
        return None
    if args.gate is None:
        raise SystemExit('The lora arm requires --gate gate.json produced by check_gate.py (PROTOCOL.md Section 4/5)')
    gate = json.loads(Path(args.gate).read_text())
    if not gate.get('passed'):
        raise SystemExit(f'Reproduction gate failed: {args.gate}')
    return gate


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    if args.smoke_random_init is not None:
        config = FermatConfig(**json.loads(Path(args.smoke_random_init).read_text()))
        model = Fermat(config)
        return model.to(args.device).eval(), dict(smoke=True, config=str(args.smoke_random_init))
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
        path = Path(args.train_patient_ids)
        wanted = (np.load(path, allow_pickle=False) if path.suffix == '.npy'
                  else np.loadtxt(path, dtype=np.int64, ndmin=1)).astype(np.int64)
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
def frozen_embeddings(model, split, rows, args):
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
    return out


def standardization(values):
    mean = values.mean(0)
    sd = values.std(0)
    sd = np.where(sd > 1e-6, sd, 1.)
    return mean.astype(np.float32), sd.astype(np.float32)


def init_head_bias(head, prevalence, mode):
    """Set the output bias to the logit of the training prevalence (clipped), or zero."""
    with torch.no_grad():
        if mode == 'frequency':
            p = np.clip(np.asarray(prevalence, dtype=np.float64), 1e-6, 1 - 1e-6)
            head.bias.copy_(torch.as_tensor(np.log(p / (1 - p)), dtype=head.bias.dtype))
        else:
            head.bias.zero_()


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
    """Gradient accumulation that never drops the trailing partial accumulation."""

    def __init__(self, optimizer, set_warmup, args):
        self.optimizer, self.set_warmup, self.args = optimizer, set_warmup, args
        self.pending, self.steps, self.epoch_start = 0, 0, 0

    def parameters(self):
        return [p for g in self.optimizer.param_groups for p in g['params']]

    def backward(self, loss):
        (loss / self.args.grad_accum).backward()
        self.pending += 1
        if self.pending == self.args.grad_accum:
            self.step()

    def step(self):
        if self.pending == 0:
            return
        self.set_warmup(self.steps)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        self.steps += 1

    def end_epoch(self):
        self.step()
        if self.steps == self.epoch_start:
            raise RuntimeError('No optimizer update was performed in this epoch')
        self.epoch_start = self.steps


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
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def sha256_of(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fingerprint(args, model_info, data_metadata, extra=None):
    record = dict(
        script=Path(sys.argv[0]).name,
        arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        git_commit=git_commit(),
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
            values[int(row['patient_id'])] = float(row[column])
    return values


def align_baseline(patient_ids, baseline, label):
    missing = [int(p) for p in patient_ids if int(p) not in baseline]
    if missing:
        raise ValueError(f'{label}: {len(missing)} test patients missing from the baseline file')
    extra = set(baseline) - set(int(p) for p in patient_ids)
    if extra:
        raise ValueError(f'{label}: baseline file has {len(extra)} patients not in the test split')
    return np.asarray([baseline[int(p)] for p in patient_ids])
