"""Run the complete SNUH Task 10 audit, smoke training, and evaluation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "outputs/snuh_tokenization_etl/patient_001pct_seed_42"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("out/snuh-task10"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command, log_path=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if log_path is None:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    args = parse_args()
    if not args.skip_training and args.max_iters < args.eval_interval:
        raise ValueError("max-iters must be at least eval-interval to save a checkpoint")
    python = sys.executable
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and not args.skip_training:
        if not args.overwrite:
            raise FileExistsError(
                f"{output} already exists; use --overwrite or --skip-training"
            )
        shutil.rmtree(output)
    baseline_dir = output / "baseline"
    context_dir = output / "lab-context"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing ETL manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest["model_vocab_size"])

    run([
        python,
        "scripts/audit_snuh_training_windows.py",
        "--data-dir",
        str(data_dir),
        "--block-size",
        str(args.block_size),
        "--batch-size",
        "64",
        "--batches",
        "100",
        "--output",
        str(reports / "window_audit.json"),
    ], output / "window_audit.log")

    if not args.skip_training:
        run([
            python,
            "train.py",
            "config/train_fermat_snuh_pilot.py",
            f"--dataset_dir={data_dir}",
            f"--out_dir={baseline_dir}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--vocab_size={vocab_size}",
            f"--batch_size={args.train_batch_size}",
            f"--block_size={args.block_size}",
            f"--max_iters={args.max_iters}",
            f"--eval_interval={args.eval_interval}",
            f"--eval_iters={args.eval_iters}",
        ], output / "baseline_training.log")
        run([
            python,
            "train.py",
            "config/train_fermat_snuh_pilot_lab_context.py",
            f"--dataset_dir={data_dir}",
            f"--out_dir={context_dir}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--vocab_size={vocab_size}",
            f"--batch_size={args.train_batch_size}",
            f"--block_size={args.block_size}",
            f"--max_iters={args.max_iters}",
            f"--eval_interval={args.eval_interval}",
            f"--eval_iters={args.eval_iters}",
        ], output / "lab_context_training.log")

    for name, run_dir in (
        ("baseline", baseline_dir),
        ("lab_context", context_dir),
    ):
        checkpoint = run_dir / "ckpt.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        run([
            python,
            "scripts/evaluate_snuh_checkpoint.py",
            "--ckpt",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.eval_batch_size),
            "--output",
            str(reports / f"{name}_evaluation.json"),
        ], output / f"{name}_evaluation.log")

    run([
        python,
        "scripts/compare_snuh_smoke_arms.py",
        "--baseline",
        str(reports / "baseline_evaluation.json"),
        "--lab-context",
        str(reports / "lab_context_evaluation.json"),
        "--baseline-metrics",
        str(baseline_dir / "metrics.jsonl"),
        "--lab-context-metrics",
        str(context_dir / "metrics.jsonl"),
        "--output",
        str(reports / "comparison.md"),
    ])

    print(f"Task 10 completed: {reports / 'comparison.md'}")


if __name__ == "__main__":
    main()
