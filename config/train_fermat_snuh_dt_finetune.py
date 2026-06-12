"""Add the waiting-time loss on top of a converged CE-only checkpoint.

Used with `--init_from=finetune`, which loads the CE-only weights but resets the
iteration counter, optimizer, and best validation loss so the run saves its own
checkpoints and follows a fresh, gentle learning-rate schedule. The token model
is already good, so the time loss is a light auxiliary ramped in quickly and
the learning rate is lower than a from-scratch run to avoid disturbing the
established token representation.

Selection is on cross-entropy so the reported checkpoint is the best-CE point
reached while the time loss is active, the most favourable comparison against
the CE-only baseline of 6.53.
"""

exec(open("config/train_fermat_snuh_dt.py").read())

out_dir = "out/snuh-dt-finetune"
wandb_run_name = "snuh-dt-finetune-" + str(time.time())

# Light, quickly-ramped time loss on top of an already-trained token head.
loss_dt_weight = 0.1
loss_dt_warmup_iters = 200

# Gentle, short fine-tuning schedule.
learning_rate = 1e-4
min_lr = 1e-5
warmup_iters = 100
max_iters = 1500
lr_decay_iters = max_iters
eval_interval = 150
log_interval = 50

# Keep the best-CE checkpoint while the time loss trains.
checkpoint_metric = "ce"
