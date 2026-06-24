"""Task 16 full-cohort two-stage time benchmark.

This is the short hardware-sizing run for the full SNUH-CDM ETL output. The
runner reads `model_vocab_size` from the ETL manifest and passes it as an
override, so the value here is only a fallback for local config inspection.
"""

import time

from model import TokenType


out_dir = "out/snuh-full-two-stage-benchmark"
dataset_dir = "/home/khdp-user/workspace/fermat-data/etl/patient_100pct_seed_42_with_genomics_tokens"

eval_interval = 100
eval_iters = 25
log_interval = 10
max_iters = 1000
always_save_checkpoint = True
save_latest_checkpoint = True
seed = 42

wandb_log = False
wandb_project = "fermat"
wandb_run_name = "snuh-full-two-stage-benchmark-" + str(time.time())

batch_size = 8
block_size = 512
gradient_accumulation_steps = 4

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
weight_decay = 0.1
vocab_size = 24387

learning_rate = 2e-4
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 2e-5
beta2 = 0.99

ignore_tokens = [0]
output_ignore_tokens = [0, 1]
ignore_types = [
    TokenType.PAD,
    TokenType.SEX,
    TokenType.NO_EVENT,
    TokenType.LAB,
    TokenType.GENOMICS,
]
mask_ties = True
train_select = "random"
eval_select = "middle"
eval_selects = ["left", "middle", "right"]
no_event_token_rate = 0
train_lifestyle_augmentations = False

decoupled_time_head = True
two_stage_time_head = True
loss_dt_weight = 0.3
loss_dt_warmup_iters = 100
checkpoint_metric = "objective"

device = "cuda"
dtype = "bfloat16"
compile = False
