"""Option A (stage 1): decoupled time head on the 1% pilot, from scratch.

A dedicated head predicts the log event-rate from the shared hidden state, so
the waiting-time loss no longer reshapes the token logits at the output layer.
Token and time share the transformer body but have separate output heads. This
should let clinical cross-entropy recover toward the CE-only baseline (6.53)
while the time head still learns useful waiting times (beating the constant-rate
and median baselines reported by the evaluator).

Directly comparable to the coupled from-scratch runs (CE 6.98-7.22).
"""

exec(open("config/train_fermat_snuh_dt.py").read())

out_dir = "out/snuh-dt-decoupled"
wandb_run_name = "snuh-dt-decoupled-" + str(time.time())

decoupled_time_head = True
loss_dt_weight = 1.0
loss_dt_warmup_iters = 500

# Select on the full objective: the decoupled head keeps cross-entropy safe, so
# selecting on CE alone would save an early checkpoint with an untrained time
# head. The objective tracks the time loss while CE stays flat.
checkpoint_metric = "objective"
