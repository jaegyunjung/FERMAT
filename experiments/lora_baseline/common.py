"""Shared plumbing for the two task scripts: arguments, model loading, hidden states, provenance."""
import argparse
import contextlib
import copy
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
from lora import (inject_lora, lora_parameters, lora_state_dict, count_parameters,  # noqa: E402
                  snapshot_parameters, assert_base_unchanged, TARGET_SETS)

ARMS = ('head_only', 'lora')


def add_common_arguments(parser):
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE,
                        help='Release folder holding artifacts/model.pt (not in git)')
    parser.add_argument('--data', type=Path, required=True, help='Dataset root in the data.py layout')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True,
                        help='head_only reproduces the frozen representation arm inside this pipeline')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16.)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--targets', choices=sorted(TARGET_SETS), default='attention')
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    parser.add_argument('--lr-head', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--weight-decay', type=float, default=None, help='Task default when omitted')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--patience', type=int, default=None, help='Arm default when omitted')
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train-subset', type=int, default=None,
                        help='Fixed random subset of training patients, shared by all arms')
    parser.add_argument('--subset-seed', type=int, default=20260910)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bf16', action='store_true', help='CUDA autocast for the transformer body')
    parser.add_argument('--bootstrap-resamples', type=int, default=None)
    parser.add_argument('--smoke-random-init', type=Path, default=None, metavar='MODEL_CONFIG_JSON',
                        help='Pipeline test only: random weights from this config, no release artifacts')
    return parser


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
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99))
    base_lrs = [g['lr'] for g in optimizer.param_groups]

    def set_warmup(step):
        factor = min(1., (step + 1) / max(1, args.warmup_steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * factor
    return optimizer, set_warmup


def trainable_state(model, head):
    return dict(lora=lora_state_dict(model) if any(True for _ in lora_parameters(model)) else {},
                head=copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()}))


def restore_state(model, head, state):
    from lora import load_lora_state_dict
    if state['lora']:
        load_lora_state_dict(model, state['lora'])
    head.load_state_dict(state['head'])


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


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


def finish_audit(model, snapshot, head, args, out_dir):
    """Prove the release tensors are untouched and persist only the trainable pieces."""
    assert_base_unchanged(model, snapshot)
    state = trainable_state(model, head)
    torch.save(state, Path(out_dir) / 'trainable_state.pt')
    return dict(frozen_base_unchanged=True, parameters=count_parameters(model),
                head_parameters=sum(p.numel() for p in head.parameters()))


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
