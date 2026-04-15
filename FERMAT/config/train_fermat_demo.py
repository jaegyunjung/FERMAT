"""
FERMAT demo config — for testing on Delphi's synthetic data (3-column format).
Automatically uses Delphi-compatible mode (all tokens treated as DX type).
"""
import time

out_dir = 'FERMAT-demo'
eval_interval = 250
eval_iters = 25
log_interval = 25
seed = 42

always_save_checkpoint = False

wandb_log = False
wandb_project = 'fermat'
wandb_run_name = 'demo-' + str(time.time())

# Use Delphi's synthetic data for testing
dataset = 'ukb_simulated_data'
batch_size = 64
block_size = 24

# Small model for demo
n_layer = 4
n_head = 4
n_embd = 64
dropout = 0.1
weight_decay = 2e-1
vocab_size = 1270  # Delphi vocabulary

learning_rate = 6e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 200
ignore_tokens = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Delphi convention
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
mask_ties = True
