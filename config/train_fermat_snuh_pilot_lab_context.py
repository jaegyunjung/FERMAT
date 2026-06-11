"""Task 10 smoke arm with LAB retained as context but excluded as a target."""

exec(open("config/train_fermat_snuh_pilot.py").read())

out_dir = "out/snuh-pilot-lab-context"
wandb_run_name = "snuh-pilot-lab-context-" + str(time.time())
ignore_types = [
    TokenType.PAD,
    TokenType.SEX,
    TokenType.NO_EVENT,
    TokenType.LAB,
]
