#!/usr/bin/env python3
"""Run Task 26 primary coupled feasibility pilots on the Pod.

Task 26 tests the primary Delphi-like baseline before any expensive full
training: coupled time modeling, no two-stage same-day head, and no global
log-rate correction by default. This is the (F,F) baseline for the architecture
comparison track.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
POD_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
POD_TASK26_ROOT = POD_ROOT / "task26"
POD_OUTPUT_ROOT = POD_TASK26_ROOT / "outputs"
CONFIG = "config/train_fermat_snuh_full_two_stage_train.py"

REQUIRED_ARTIFACTS = [
    "train.bin",
    "val.bin",
    "test.bin",
    "manifest.json",
    "token_registry.csv",
    "sha256.json",
]

VARIANTS = {
    "coupled_pure": {
        "decoupled_time_head": False,
        "two_stage_time_head": False,
        "use_global_log_rate": False,
    },
    "coupled_lograte": {
        "decoupled_time_head": False,
        "two_stage_time_head": False,
        "use_global_log_rate": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=POD_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=POD_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["coupled_pure"],
        help="coupled_pure is the primary Delphi-like baseline; "
        "coupled_lograte is a sensitivity arm.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-layer", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=10)
    parser.add_argument("--n-embd", type=int, default=640)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--full-lr-decay-iters", type=int, default=100000)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=100)
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


def summarize_run(output_dir: Path, variant: str, status: str, elapsed: float):
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
        "variant": variant,
        "status": status,
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "decoupled_time_head": config.get("decoupled_time_head"),
        "two_stage_time_head": config.get("two_stage_time_head"),
        "use_global_log_rate": config.get("use_global_log_rate"),
        "block_size": config.get("block_size"),
        "batch_size": config.get("batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "max_iters": config.get("max_iters"),
        "last_iter": last_train.get("iter"),
        "last_train_ce": last_train.get("train/loss_ce"),
        "last_train_same_day": last_train.get("train/loss_same_day"),
        "last_train_dt": last_train.get("train/loss_dt"),
        "last_train_objective": last_train.get("train/objective_loss"),
        "last_targets_per_sec": last_train.get("train/targets_per_second"),
        "max_cuda_gb": last_train.get("system/max_cuda_memory_gb"),
        "last_val_iter": last_val.get("iter"),
        "last_val_ce": last_val.get("val/loss_ce"),
        "last_val_same_day": last_val.get("val/loss_same_day"),
        "last_val_dt": last_val.get("val/loss_dt"),
        "last_val_objective": last_val.get("val/objective_loss"),
        "best_val_iter": best.get("iter"),
        "best_val_ce": best.get("val/loss_ce"),
        "best_val_same_day": best.get("val/loss_same_day"),
        "best_val_dt": best.get("val/loss_dt"),
        "best_val_objective": best.get("val/objective_loss"),
    }


def write_summary(root: Path, rows: list[dict]):
    summary_path = root / "task26_coupled_primary_pilot_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("\n## TASK26_COUPLED_PRIMARY_PILOT_SUMMARY")
    if not rows:
        print("(no rows)")
        return
    columns = [
        "variant",
        "status",
        "decoupled_time_head",
        "two_stage_time_head",
        "use_global_log_rate",
        "last_iter",
        "last_train_ce",
        "last_train_dt",
        "last_val_ce",
        "last_val_dt",
        "best_val_objective",
        "max_cuda_gb",
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
    run_name = args.run_name or f"coupled_primary_pilot_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    root = output_root / run_name
    prepare_output(root, args.overwrite)
    manifest, vocab_size = require_full_etl(data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Output root: {root}")
    print(f"ETL model vocab size: {vocab_size}")
    print(f"ETL patients: {manifest.get('patients', 'NA')}")

    summaries = []
    for variant in args.variants:
        variant_config = VARIANTS[variant]
        output_dir = root / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "training.log"
        print(f"\n## START variant={variant}")
        command = [
            "/opt/conda/bin/python",
            "train.py",
            args.config,
        ]
        overrides = {
            "dataset_dir": str(data_dir),
            "out_dir": str(output_dir),
            "init_from": "scratch",
            "device": args.device,
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "decoupled_time_head": variant_config["decoupled_time_head"],
            "two_stage_time_head": variant_config["two_stage_time_head"],
            "use_global_log_rate": variant_config["use_global_log_rate"],
            "loss_dt_weight": args.loss_dt_weight,
            "checkpoint_metric": "objective",
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_iters": args.max_iters,
            "lr_decay_iters": args.full_lr_decay_iters,
            "warmup_iters": args.warmup_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "learning_rate": args.learning_rate,
            "loss_dt_warmup_iters": args.loss_dt_warmup_iters,
            "save_latest_checkpoint": True,
        }
        for key, value in overrides.items():
            add_override(command, key, value)
        return_code, elapsed = run(command, log_path)
        status = "ok" if return_code == 0 else f"failed:{return_code}"
        summaries.append(summarize_run(output_dir, variant, status, elapsed))
        if return_code != 0 and args.fail_fast:
            break
    write_summary(root, summaries)


if __name__ == "__main__":
    main()
