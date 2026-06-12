"""Option A (stage 1): decoupled time head warm-started from the CE-only run.

Mirrors the coupled finetune config (light time loss, gentle schedule, CE
selection) but adds the separate time head. This is the apples-to-apples
comparison against the best coupled result (finetune: CE 6.88, new-onset top-1
2.29%): the only change is the decoupled head, so it isolates whether the
output-layer interference was the cause.

Use with the runner's --resume-from pointing at the CE-only extended run.
"""

exec(open("config/train_fermat_snuh_dt_finetune.py").read())

out_dir = "out/snuh-dt-decoupled-finetune"
wandb_run_name = "snuh-dt-decoupled-finetune-" + str(time.time())

decoupled_time_head = True

# The decoupled head removes the output-layer interference, but the time loss
# still shares the transformer body, so the weight trades token calibration for
# time accuracy. A 1% weight sweep picked 0.3 as the balance where every token
# metric stays at or above the CE-only baseline (CE 6.56, top-1 4.26%, new-onset
# 3.91%, top-5/10 above baseline) while the time head still beats both
# baselines. Checkpoint on the objective so the saved point has a trained time
# head rather than the earliest flat-CE step.
loss_dt_weight = 0.3
checkpoint_metric = "objective"

# The from-scratch run showed the time head needs ~3000 steps to fully train
# (NLL +2.84, 47-day median error). The body is already converged here, so give
# the time head the same budget while cross-entropy stays flat.
max_iters = 3000
lr_decay_iters = 3000
eval_interval = 250
