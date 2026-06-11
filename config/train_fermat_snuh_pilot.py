"""Task 10 baseline smoke training on the 1% SNUH ETL pilot."""

import time

from model import TokenType

out_dir = "out/snuh-pilot-baseline"
dataset_dir = "outputs/snuh_tokenization_etl/patient_001pct_seed_42"

eval_interval = 50
eval_iters = 20
log_interval = 10
max_iters = 300
always_save_checkpoint = True
seed = 42

wandb_log = False
wandb_project = "fermat"
wandb_run_name = "snuh-pilot-baseline-" + str(time.time())

batch_size = 16
block_size = 256
gradient_accumulation_steps = 1

n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.1
weight_decay = 0.1
vocab_size = 7699

learning_rate = 3e-4
warmup_iters = 20
lr_decay_iters = max_iters
min_lr = 3e-5
beta2 = 0.99

ignore_tokens = [0]
ignore_types = [TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT]
mask_ties = True
train_select = "random"
eval_select = "middle"
eval_selects = ["left", "middle", "right"]
no_event_token_rate = 0
loss_dt_weight = 1.0
train_lifestyle_augmentations = False
checkpoint_metric = "objective"

device = "cuda"
dtype = "bfloat16"
compile = False
