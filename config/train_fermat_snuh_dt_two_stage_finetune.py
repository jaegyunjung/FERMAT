"""Two-stage clinical time model warm-started from the adopted Task 13 run."""

exec(open("config/train_fermat_snuh_dt_decoupled_finetune.py").read())

out_dir = "out/snuh-dt-two-stage-finetune"
wandb_run_name = "snuh-dt-two-stage-finetune-" + str(time.time())

# Stage 1 predicts whether the next clinical target occurs on the same day.
# Stage 2 predicts the positive waiting time only for different-day targets.
two_stage_time_head = True

# Keep the adopted Task 13 weighting. It now applies to the sum of same-day
# binary cross-entropy and conditional different-day exponential NLL.
loss_dt_weight = 0.3
checkpoint_metric = "objective"

max_iters = 3000
lr_decay_iters = 3000
eval_interval = 250
