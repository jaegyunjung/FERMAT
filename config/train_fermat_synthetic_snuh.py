"""
FERMAT smoke training config — Synthetic SNUH 4-column data.

Used by scripts/run_smoke_synthetic_snuh.sh after
scripts/preprocess_synthetic_snuh_to_fermat.py produces
data/synthetic_snuh/{train,val}.bin.

Tiny model. The goal here is pipeline verification, not performance.
"""
import time

out_dir = 'FERMAT-synthetic-snuh-v0_1'
eval_interval = 50
eval_iters = 10
log_interval = 10
seed = 42

always_save_checkpoint = True

wandb_log = False
wandb_project = 'fermat'
wandb_run_name = 'synthetic-snuh-' + str(time.time())

# 4-column dataset produced by preprocess_synthetic_snuh_to_fermat.py
dataset = 'synthetic_snuh'
batch_size = 16
block_size = 128

# Small model
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.1
weight_decay = 1e-1

# vocab_size must match vocab.csv produced by preprocess.
# Set conservatively above the 2,000 cap from token_spec_v0.1.md
# (+ reserved slots for PAD and no-event).
vocab_size = 2048

learning_rate = 3e-4
max_iters = 200
lr_decay_iters = 200
min_lr = 3e-5
beta2 = 0.99

warmup_iters = 20
# ignore_tokens: PAD only (token_id 0). NO_EVENT is filtered by
# ignore_types in the model; we keep ignore_tokens narrow here.
ignore_tokens = [0]
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
mask_ties = True
