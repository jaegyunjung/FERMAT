"""Re-enable the waiting-time loss on the 1% SNUH pilot with the log-rate fix.

Builds on the validated LAB-context CE-only arm. The learnable global log-rate
scalar in `model.py` decouples the absolute event rate from the vocabulary
size, so the waiting-time loss is the same order of magnitude as cross-entropy
instead of dominating and flattening the token logits. The time loss is ramped
in linearly over the first part of training so cross-entropy can establish
token ranking before the time objective contributes gradient.

This is the diagnostic that closes the time objective: confirm that with the
fix enabled, clinical-only cross-entropy stays at least as good as the
CE-only run (clinical CE 6.53) while the waiting-time metrics also improve.
"""

exec(open("config/train_fermat_snuh_pilot_lab_context.py").read())

out_dir = "out/snuh-dt-lab-context"
wandb_run_name = "snuh-dt-lab-context-" + str(time.time())

# Re-enable the waiting-time objective. t_min caps the effective rate for
# numeric safety; the global log-rate scalar handles the absolute rate.
t_min = 0.1
loss_dt_weight = 1.0
loss_dt_warmup_iters = 500

max_iters = 3000
eval_interval = 250
eval_iters = 50
log_interval = 100
lr_decay_iters = max_iters
save_latest_checkpoint = True

# Select on the full ce + time objective now that both terms train.
checkpoint_metric = "objective"
