"""
Longer-context/lower-regularization token-prediction run on full Synthetic SNUH.

This keeps the proven 4-layer shape but gives the model more context and more
optimization steps while preserving the token-prediction-focused objective.
"""
import time

out_dir = 'FERMAT-synthetic-snuh-token-prediction-longer'
eval_interval = 200
eval_iters = 50
log_interval = 25
seed = 42

always_save_checkpoint = True
save_latest_checkpoint = True

wandb_log = False
wandb_project = 'fermat'
wandb_run_name = 'synthetic-snuh-token-prediction-longer-' + str(time.time())

dataset = 'synthetic_snuh'
batch_size = 32
block_size = 256

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.05
weight_decay = 5e-2

vocab_size = 2048

learning_rate = 3e-4
max_iters = 4000
lr_decay_iters = 4000
min_lr = 3e-5
beta2 = 0.99

warmup_iters = 200
ignore_tokens = [0, 2, 3, 4]
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 0
mask_ties = False
train_select = 'random'
eval_select = 'random'
loss_dt_weight = 0.0
train_lifestyle_augmentations = False
checkpoint_metric = 'ce'
