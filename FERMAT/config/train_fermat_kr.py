"""
FERMAT training config for Korean nationwide claims data.

Adjust vocab_size, ignore_tokens, and dataset after data preprocessing.
"""
import time
from model import TokenType

out_dir = 'FERMAT-v1'
eval_interval = 500
eval_iters = 50
log_interval = 50
seed = 42

always_save_checkpoint = False

wandb_log = True
wandb_project = 'fermat'
wandb_run_name = 'fermat-v1-' + str(time.time())

dataset = 'fermat_kr'       # data/fermat_kr/train.bin, val.bin
batch_size = 128
block_size = 96              # Longer sequences due to multi-modal tokens

# Model — start with Delphi-scale, then scale up based on GPU budget
n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.1
weight_decay = 2e-1
vocab_size = 3400            # TBD after vocabulary construction

learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 6e-5
beta2 = 0.99

warmup_iters = 2000

# FERMAT specific
ignore_tokens = [0]          # Only padding; type-based filtering handles the rest
ignore_types = [TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT]

t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
mask_ties = True

# System — adjust per available hardware
device = 'cuda'
dtype = 'bfloat16'
compile = True
gradient_accumulation_steps = 4  # Effective batch = 128 * 4 = 512
