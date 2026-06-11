"""
FERMAT training script.
Extends Delphi's train.py to pass token_type to the model.
Backward compatible with Delphi 3-column data (token_type defaults to DX).
"""

import os
import time
import math
import json
import platform
import subprocess
from contextlib import nullcontext

import numpy as np
import torch

from model import Fermat, FermatConfig, TokenType
from utils import get_p2i, get_batch, load_data


# =============================================================================
# Default hyperparameters (overridable via config file or CLI)
# =============================================================================
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
seed = 42

# wandb
wandb_log = False
wandb_project = 'fermat'
wandb_run_name = 'run' + str(time.time())

# data
dataset = 'fermat_demo'
dataset_dir = ''
gradient_accumulation_steps = 1
batch_size = 128
block_size = 48

# model
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.2
bias = False
vocab_size = 256
n_token_types = len(TokenType)

# optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 10000
min_lr = 6e-5

# system
device = 'cpu'
dtype = 'float32'
compile = False

# FERMAT specific
token_dropout = 0.0
t_min = 0.0
mask_ties = True
ignore_tokens = [0]
output_ignore_tokens = []
ignore_types = [TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT]
data_fraction = 1.0
no_event_token_rate = 5
train_select = 'left'
eval_select = 'left'
eval_selects = []
loss_dt_weight = 1.0
train_lifestyle_augmentations = True
checkpoint_metric = 'objective'
save_latest_checkpoint = False
metrics_filename = 'metrics.jsonl'

# =============================================================================
# Config overrides
# =============================================================================
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
with open('configurator.py') as f:
    exec(f.read())
config = {k: globals()[k] for k in config_keys if k in globals()}

# =============================================================================
# Setup
# =============================================================================
os.makedirs(out_dir, exist_ok=True)
metrics_path = os.path.join(out_dir, metrics_filename)
if init_from == 'scratch':
    open(metrics_path, 'w').close()
try:
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_commit = None
run_manifest = {
    'created_at_unix': time.time(),
    'git_commit': git_commit,
    'python_version': platform.python_version(),
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device': (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    'config': config,
}
with open(os.path.join(out_dir, 'run_manifest.json'), 'w', encoding='utf-8') as handle:
    json.dump(run_manifest, handle, indent=2)
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.set_default_dtype(ptdtype)

# =============================================================================
# Data loading
# =============================================================================
data_dir = dataset_dir or os.path.join('data', dataset)
train_data, has_types = load_data(os.path.join(data_dir, 'train.bin'))
val_data, _ = load_data(os.path.join(data_dir, 'val.bin'))

print(f"Data format: {'4-column (FERMAT)' if has_types else '3-column (Delphi compat)'}")
print(f"Train: {len(train_data)} rows, Val: {len(val_data)} rows")

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)

if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]

# =============================================================================
# Model init
# =============================================================================
iter_num = 0
best_val_loss = 1e9

print(f"vocab_size = {vocab_size}, n_token_types = {n_token_types}")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=vocab_size, n_token_types=n_token_types,
    dropout=dropout, token_dropout=token_dropout, t_min=t_min,
    mask_ties=mask_ties, ignore_tokens=ignore_tokens,
    output_ignore_tokens=output_ignore_tokens, ignore_types=ignore_types,
)

if init_from == 'scratch':
    print("Initializing a new FERMAT model from scratch")
    conf = FermatConfig(**model_args)
    model = Fermat(conf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_token_types']:
        model_args[k] = checkpoint_model_args[k]
    conf = FermatConfig(**model_args)
    model = Fermat(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
    print("Compiling model...")
    model = torch.compile(model)


# =============================================================================
# Helpers
# =============================================================================
def _unpack_batch(batch_result, device):
    """Unpack get_batch result, creating default token_type if not present."""
    if len(batch_result) == 6:
        x, a, y, b, xt, yt = batch_result
    else:
        x, a, y, b, xt = batch_result
        yt = None
    if xt is None:
        # Delphi compatibility: default all to DX type
        xt = torch.full_like(x, TokenType.DX)
        yt = torch.full_like(y, TokenType.DX)
        if device == 'cuda':
            xt = xt.to(device)
            yt = yt.to(device)
    elif yt is None:
        yt = torch.full_like(y, TokenType.DX)
        if device == 'cuda':
            yt = yt.to(device)
    return x, a, y, b, xt, yt


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        loss_sums = torch.zeros(2, dtype=torch.float64)
        target_count = 0
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        selectors = (
            eval_selects
            if split == 'val' and eval_selects
            else [eval_select]
        )
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            batch = get_batch(ix, data, p2i, block_size=block_size,
                              device=device, select=selectors[k % len(selectors)],
                              no_event_token_rate=no_event_token_rate,
                              cut_batch=True, return_target_types=True)
            X, A, Y, B, XT, YT = _unpack_batch(batch, device)
            with ctx:
                logits, loss, _ = model(
                    X, A, XT, Y, B,
                    target_token_type=YT,
                    validation_loss_mode=True,
                    return_attention=False,
                )
            batch_targets = int(loss['n_targets'].item())
            if batch_targets:
                loss_sums += torch.tensor(
                    [
                        loss['loss_ce'].item() * batch_targets,
                        loss['loss_dt'].item() * batch_targets,
                    ],
                    dtype=torch.float64,
                )
                target_count += batch_targets
        if target_count == 0:
            raise RuntimeError(f"No valid targets found while evaluating {split}")
        out[split] = loss_sums / target_count
        out[f'{split}_targets'] = target_count
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# =============================================================================
# Training loop
# =============================================================================
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Initial batch
ix = torch.randint(len(train_p2i), (batch_size,))
batch = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                  padding='random', lifestyle_augmentations=train_lifestyle_augmentations, select=train_select,
                  no_event_token_rate=no_event_token_rate, return_target_types=True)
X, A, Y, B, XT, YT = _unpack_batch(batch, device)

t0 = time.time()
local_iter_num = 0
val_loss = None

while True:
    metrics = {"iter": iter_num}
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        train_objective = losses['train'][0].item() + loss_dt_weight * losses['train'][1].item()
        val_objective = losses['val'][0].item() + loss_dt_weight * losses['val'][1].item()
        val_loss = losses['val'][0].item() if checkpoint_metric == 'ce' else val_objective
        print(
            f"step {iter_num}: "
            f"train objective {train_objective:.4f} "
            f"(ce {losses['train'][0].item():.4f}, dt {losses['train'][1].item():.4f}); "
            f"val objective {val_objective:.4f} "
            f"(ce {losses['val'][0].item():.4f}, dt {losses['val'][1].item():.4f})"
        )

        metrics.update({
            "train/agg_loss": train_objective,
            "val/loss": val_loss,
            "val/loss_ce": losses['val'][0].item(),
            "val/loss_dt": losses['val'][1].item(),
            "val/objective_loss": val_objective,
            "train/eval_targets": losses["train_targets"],
            "val/eval_targets": losses["val_targets"],
        })

        improved = best_val_loss > val_loss
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': min(best_val_loss, val_loss),
            'config': config,
        }

        if improved and iter_num > 0:
            print(f"Saving best checkpoint to {out_dir}/ckpt.pt")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if always_save_checkpoint and save_latest_checkpoint and iter_num > 0:
            print(f"Saving latest checkpoint to {out_dir}/ckpt_latest.pt")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))

        if improved:
            best_val_loss = val_loss

        if iter_num % 10_000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
        t0 = time.time()

    if iter_num == 0 and eval_only:
        break

    # Forward / backward
    step_target_count = 0
    step_ce_sum = 0.0
    step_dt_sum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, _ = model(
                X, A, XT, Y, B,
                target_token_type=YT,
                return_attention=False,
            )

        # Prefetch next batch
        ix = torch.randint(len(train_p2i), (batch_size,))
        batch = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                          padding='random', lifestyle_augmentations=train_lifestyle_augmentations, select=train_select,
                          no_event_token_rate=no_event_token_rate, cut_batch=True, return_target_types=True)
        X, A, Y, B, XT, YT = _unpack_batch(batch, device)

        micro_target_count = int(loss['n_targets'].item())
        step_target_count += micro_target_count
        step_ce_sum += loss['loss_ce'].item() * micro_target_count
        step_dt_sum += loss['loss_dt'].item() * micro_target_count
        combined_loss = loss['loss_ce'] + loss_dt_weight * loss['loss_dt']
        scaler.scale(combined_loss / gradient_accumulation_steps).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        mean_ce = step_ce_sum / step_target_count if step_target_count else 0.0
        mean_dt = step_dt_sum / step_target_count if step_target_count else 0.0
        lossf = mean_ce + loss_dt_weight * mean_dt
        tokens_per_second = step_target_count / dt if dt > 0 else 0.0
        max_memory_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device_type == 'cuda'
            else 0.0
        )
        print(
            f"iter {iter_num}: loss {lossf:.4f} "
            f"(ce {mean_ce:.4f}, dt {mean_dt:.4f}, "
            f"targets {step_target_count}), time {dt*1000:.2f}ms"
        )
        metrics.update({
            "train/loss": lossf,
            "train/loss_ce": mean_ce,
            "train/loss_dt": mean_dt,
            "train/targets": step_target_count,
            "train/targets_per_second": tokens_per_second,
            "system/max_cuda_memory_gb": max_memory_gb,
            "lr": lr,
        })

    if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
        wandb.log(metrics)
    if len(metrics) > 1:
        with open(metrics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + '\n')

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
