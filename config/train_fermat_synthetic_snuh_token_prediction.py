"""
FERMAT token prediction training config — full Synthetic SNUH 4-column data.

This config is intended for a small but real next-token prediction demo on the
full `data/synthetic_snuh` dataset. It keeps the same dataset and tokenization
semantics as the existing Synthetic SNUH pipeline while using a slightly larger
model than the smoke test.
"""
import time

out_dir = 'FERMAT-synthetic-snuh-token-prediction'
eval_interval = 100
eval_iters = 25
log_interval = 25
seed = 42

always_save_checkpoint = True

wandb_log = False
wandb_project = 'fermat'
wandb_run_name = 'synthetic-snuh-token-prediction-' + str(time.time())

dataset = 'synthetic_snuh'
batch_size = 32
block_size = 128

# Slightly larger than the smoke-test model, but still modest.
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
weight_decay = 1e-1

# Matches the existing Synthetic SNUH preprocessing cap plus reserved slots.
vocab_size = 2048

learning_rate = 3e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 3e-5
beta2 = 0.99

warmup_iters = 100
ignore_tokens = [0]
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
mask_ties = True
