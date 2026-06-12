"""Extended CE-only LAB-context run after the Task 10 smoke gate."""

exec(open("config/train_fermat_snuh_pilot_lab_context.py").read())

out_dir = "out/snuh-task10-lab-context-long"
wandb_run_name = "snuh-task10-lab-context-long-" + str(time.time())

max_iters = 3000
eval_interval = 250
eval_iters = 50
log_interval = 100
lr_decay_iters = max_iters
save_latest_checkpoint = True
