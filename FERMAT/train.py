"""
FERMAT training script.
Extends Delphi's train.py to pass token_type to the model.
Backward compatible with Delphi 3-column data (token_type defaults to DX).
"""

import os
import time
import math
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
ignore_types = [TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT]
data_fraction = 1.0
no_event_token_rate = 5

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
data_dir = os.path.join('data', dataset)
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
    mask_ties=mask_ties, ignore_tokens=ignore_tokens, ignore_types=ignore_types,
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
    x, a, y, b, xt = batch_result
    if xt is None:
        # Delphi compatibility: default all to DX type
        xt = torch.full_like(x, TokenType.DX)
        if device == 'cuda':
            xt = xt.to(device)
    return x, a, y, b, xt


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 2)
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            batch = get_batch(ix, data, p2i, block_size=block_size,
                              device=device, select='left',
                              no_event_token_rate=no_event_token_rate,
                              cut_batch=True)
            X, A, Y, B, XT = _unpack_batch(batch, device)
            with ctx:
                logits, loss, _ = model(X, A, XT, Y, B, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
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
                  padding='random', lifestyle_augmentations=True, select='left',
                  no_event_token_rate=no_event_token_rate)
X, A, Y, B, XT = _unpack_batch(batch, device)

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
        val_loss = losses['val'].sum().item()
        print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")

        metrics.update({
            "train/agg_loss": losses['train'].sum().item(),
            "val/loss": val_loss,
            "val/loss_ce": losses['val'][0].item(),
            "val/loss_dt": losses['val'][1].item(),
        })

        if always_save_checkpoint or best_val_loss > val_loss:
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if best_val_loss > val_loss:
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

    if iter_num == 0 and eval_only:
        break

    # Forward / backward
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, att = model(X, A, XT, Y, B)

        # Prefetch next batch
        ix = torch.randint(len(train_p2i), (batch_size,))
        batch = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                          padding='random', lifestyle_augmentations=True, select='left',
                          no_event_token_rate=no_event_token_rate, cut_batch=True)
        X, A, Y, B, XT = _unpack_batch(batch, device)

        combined_loss = loss['loss_ce'] + loss['loss_dt']
        scaler.scale(combined_loss).backward()

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
        lossf = combined_loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        metrics.update({"train/loss": lossf, "lr": lr})

    if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
        wandb.log(metrics)

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
