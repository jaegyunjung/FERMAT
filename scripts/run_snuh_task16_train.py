#!/usr/bin/env python3
"""Run the Task 16 long full-cohort two-stage training job on the Pod."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_snuh_task16_benchmark import (
    POD_DATA_DIR,
    POD_OUTPUT_ROOT,
    add_override,
    default_output_dir,
    prepare_output,
    require_full_etl,
    run,
)


CONFIG = "config/train_fermat_snuh_full_two_stage_train.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=POD_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=POD_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--loss-dt-weight", type=float, default=0.3)
    parser.add_argument("--loss-dt-warmup-iters", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(output_root)
    )
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    _, vocab_size = require_full_etl(data_dir)
    prepare_output(output_dir, args.overwrite)

    init_from = "scratch"
    if args.resume_from:
        resume_from = args.resume_from.expanduser().resolve()
        checkpoint = resume_from / "ckpt.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
        (output_dir / "ckpt.pt").write_bytes(checkpoint.read_bytes())
        init_from = "resume"

    command = [
        sys.executable,
        "train.py",
        args.config,
        f"--dataset_dir={data_dir}",
        f"--out_dir={output_dir}",
        f"--init_from={init_from}",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        f"--vocab_size={vocab_size}",
        "--decoupled_time_head=True",
        "--two_stage_time_head=True",
        f"--loss_dt_weight={args.loss_dt_weight}",
        "--checkpoint_metric=objective",
    ]
    overrides = {
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "log_interval": args.log_interval,
        "learning_rate": args.learning_rate,
        "warmup_iters": args.warmup_iters,
        "loss_dt_warmup_iters": args.loss_dt_warmup_iters,
    }
    for key, value in overrides.items():
        add_override(command, key, value)
    if args.max_iters is not None:
        command.append(f"--lr_decay_iters={args.max_iters}")
    run(command, output_dir / "training.log")
    print(f"Task 16 training output: {output_dir}")


if __name__ == "__main__":
    main()
