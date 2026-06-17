"""Task 16 full-cohort two-stage time training config.

Use after the benchmark fixes model size and batch shape. Defaults are
deliberately conservative and can be overridden by the Task 16 runner without
editing this file.
"""

exec(open("config/train_fermat_snuh_full_two_stage_benchmark.py").read())

out_dir = "out/snuh-full-two-stage-train"
wandb_run_name = "snuh-full-two-stage-train-" + str(time.time())

max_iters = 100000
eval_interval = 1000
eval_iters = 100
log_interval = 50
warmup_iters = 2000
lr_decay_iters = max_iters
save_latest_checkpoint = True
