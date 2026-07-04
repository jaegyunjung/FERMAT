#!/usr/bin/env python3
"""Run Task 21 block-size pilot training jobs on the Pod.

The pilot keeps the 10L/640 Task 16 model shape and compares context lengths.
It intentionally keeps the full-run LR horizon while stopping early, so the
pilot answers whether the longer context can train stably and at a practical
speed rather than creating a separate short-run schedule.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
POD_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
POD_TASK21_ROOT = POD_ROOT / "task21"
POD_OUTPUT_ROOT = POD_TASK21_ROOT / "outputs"
CONFIG = "config/train_fermat_snuh_full_two_stage_train.py"

REQUIRED_ARTIFACTS = [
    "train.bin",
    "val.bin",
    "test.bin",
    "manifest.json",
    "token_registry.csv",
    "sha256.json",
]

DEFAULT_BLOCK_SHAPES = {
    1024: {"batch_size": 8, "gradient_accumulation_steps": 4},
    2048: {"batch_size": 2, "gradient_accumulation_steps": 16},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=POD_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=POD_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--blocks", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--full-lr-decay-iters", type=int, default=100000)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--loss-dt-weight", type=float, default=0.3)
    parser.add_argument("--loss-dt-warmup-iters", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def require_full_etl(data_dir: Path):
    missing = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Full SNUH ETL is not complete. Missing: "
            + ", ".join(str(data_dir / name) for name in missing)
        )
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest["model_vocab_size"])
    if vocab_size <= 0:
        raise ValueError(f"Invalid model_vocab_size in {data_dir / 'manifest.json'}")
    return manifest, vocab_size


def prepare_output(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def add_override(command: list[str], key: str, value):
    if value is not None:
        command.append(f"--{key}={value}")


def run(command: list[str], log_path: Path):
    print("+", " ".join(str(part) for part in command), flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    return return_code, time.time() - started


def read_metrics(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_run(output_dir: Path, block_size: int, status: str, elapsed: float):
    rows = read_metrics(output_dir / "metrics.jsonl")
    train = [row for row in rows if "train/targets_per_second" in row]
    val = [row for row in rows if "val/objective_loss" in row]
    best = min(val, key=lambda row: row["val/objective_loss"]) if val else {}
    last_train = train[-1] if train else {}
    last_val = val[-1] if val else {}
    manifest_path = output_dir / "run_manifest.json"
    config = {}
    if manifest_path.exists():
        config = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {})
    return {
        "block_size": block_size,
        "status": status,
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "n_layer": config.get("n_layer"),
        "n_head": config.get("n_head"),
        "n_embd": config.get("n_embd"),
        "batch_size": config.get("batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "max_iters": config.get("max_iters"),
        "warmup_iters": config.get("warmup_iters"),
        "lr_decay_iters": config.get("lr_decay_iters"),
        "last_iter": last_train.get("iter"),
        "last_train_ce": last_train.get("train/loss_ce"),
        "last_train_objective": last_train.get("train/objective_loss"),
        "last_targets_per_sec": last_train.get("train/targets_per_second"),
        "max_cuda_gb": last_train.get("system/max_cuda_memory_gb"),
        "last_val_iter": last_val.get("iter"),
        "last_val_ce": last_val.get("val/loss_ce"),
        "last_val_objective": last_val.get("val/objective_loss"),
        "best_val_iter": best.get("iter"),
        "best_val_ce": best.get("val/loss_ce"),
        "best_val_objective": best.get("val/objective_loss"),
    }


def write_summary(root: Path, rows: list[dict]):
    summary_path = root / "task21_block_pilot_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\n## TASK21_BLOCK_PILOT_SUMMARY")
    if not rows:
        print("(no rows)")
        return
    columns = [
        "block_size",
        "status",
        "batch_size",
        "gradient_accumulation_steps",
        "last_iter",
        "last_targets_per_sec",
        "max_cuda_gb",
        "best_val_iter",
        "best_val_ce",
        "best_val_objective",
        "elapsed_seconds",
        "output_dir",
    ]
    widths = {column: max(len(column), 12) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    print(" ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print(" ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    print("\n## SUMMARY_JSON")
    print(summary_path)


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.run_name or f"block_pilot_10l640_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    root = output_root / run_name
    prepare_output(root, args.overwrite)
    manifest, vocab_size = require_full_etl(data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Output root: {root}")
    print(f"ETL model vocab size: {vocab_size}")
    print(f"ETL patients: {manifest.get('patients', 'NA')}")

    summaries = []
    for block_size in args.blocks:
        shape = DEFAULT_BLOCK_SHAPES.get(block_size)
        if shape is None:
            raise ValueError(
                f"No default batch shape for block_size={block_size}. "
                f"Known: {sorted(DEFAULT_BLOCK_SHAPES)}"
            )
        output_dir = root / f"block_{block_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "train.py",
            args.config,
            f"--dataset_dir={data_dir}",
            f"--out_dir={output_dir}",
            "--init_from=scratch",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--vocab_size={vocab_size}",
            "--decoupled_time_head=True",
            "--two_stage_time_head=True",
            f"--loss_dt_weight={args.loss_dt_weight}",
            "--checkpoint_metric=objective",
            f"--n_layer={args.n_layer}",
            f"--n_head={args.n_head}",
            f"--n_embd={args.n_embd}",
            f"--block_size={block_size}",
            f"--batch_size={shape['batch_size']}",
            f"--gradient_accumulation_steps={shape['gradient_accumulation_steps']}",
            f"--max_iters={args.max_iters}",
            f"--lr_decay_iters={args.full_lr_decay_iters}",
            f"--warmup_iters={args.warmup_iters}",
            f"--eval_interval={args.eval_interval}",
            f"--eval_iters={args.eval_iters}",
            f"--log_interval={args.log_interval}",
            f"--learning_rate={args.learning_rate}",
            f"--loss_dt_warmup_iters={args.loss_dt_warmup_iters}",
            "--save_latest_checkpoint=True",
        ]
        print(f"\n## START block_size={block_size}", flush=True)
        return_code, elapsed = run(command, output_dir / "training.log")
        status = "ok" if return_code == 0 else f"failed_returncode_{return_code}"
        print(f"## DONE block_size={block_size} status={status}", flush=True)
        summaries.append(summarize_run(output_dir, block_size, status, elapsed))
        write_summary(root, summaries)
        if return_code != 0 and args.fail_fast:
            raise subprocess.CalledProcessError(return_code, command)
    write_summary(root, summaries)
    print("\nTask 21 block pilot output:", root)


if __name__ == "__main__":
    main()
