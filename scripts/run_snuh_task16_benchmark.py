#!/usr/bin/env python3
"""Run the Task 16 full-cohort two-stage benchmark on a Pod GPU."""

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
POD_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42"
POD_OUTPUT_ROOT = POD_ROOT / "out"
CONFIG = "config/train_fermat_snuh_full_two_stage_benchmark.py"

REQUIRED_ARTIFACTS = [
    "train.bin",
    "val.bin",
    "test.bin",
    "manifest.json",
    "token_registry.csv",
    "train_lab_decile_cutpoints.parquet",
    "sha256.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=POD_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=POD_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--loss-dt-weight", type=float, default=0.3)
    parser.add_argument("--loss-dt-warmup-iters", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-max-patients", type=int, default=20000)
    parser.add_argument("--time-baseline-max-patients", type=int, default=20000)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_bundle_id():
    manifest = ROOT / "bundle_manifest.json"
    if not manifest.exists():
        return "snuh_task16_benchmark"
    return json.loads(manifest.read_text(encoding="utf-8"))["bundle_id"]


def require_full_etl(data_dir):
    missing = [name for name in REQUIRED_ARTIFACTS if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Task 15 full ETL is not complete. Missing: "
            + ", ".join(str(data_dir / name) for name in missing)
        )
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest["model_vocab_size"])
    if vocab_size <= 0:
        raise ValueError(f"Invalid model_vocab_size in {data_dir / 'manifest.json'}")
    return manifest, vocab_size


def default_output_dir(output_root):
    bundle_id = read_bundle_id()
    candidate = output_root / bundle_id
    if not candidate.exists():
        return candidate
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return output_root / f"{bundle_id}_{stamp}"


def prepare_output(output_dir, overwrite):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
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
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def add_override(command, key, value):
    if value is not None:
        command.append(f"--{key}={value}")


def write_benchmark_summary(output_dir, manifest, args):
    metrics_path = output_dir / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if metrics_path.exists() else []
    validation = [row for row in rows if "val/loss_ce" in row]
    best = min(validation, key=lambda row: row["val/objective_loss"]) if validation else None
    last = rows[-1] if rows else {}
    config = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))[
        "config"
    ]
    lines = [
        "# Task 16 full-cohort benchmark",
        "",
        f"- Data directory: `{args.data_dir}`",
        f"- Output directory: `{output_dir}`",
        f"- ETL model vocab size: `{manifest['model_vocab_size']}`",
        f"- Model: `{config['n_layer']}` layers, `{config['n_head']}` heads, `{config['n_embd']}` embedding, block `{config['block_size']}`",
        f"- Batch: `{config['batch_size']}` x grad accumulation `{config['gradient_accumulation_steps']}`",
        f"- Two-stage time head: `{config['two_stage_time_head']}`",
        f"- Decoupled time head: `{config['decoupled_time_head']}`",
        f"- loss_dt_weight: `{config['loss_dt_weight']}`",
        "",
        "## Last training record",
        "",
        f"- Iteration: `{last.get('iter', 'NA')}`",
        f"- Train CE: `{last.get('train/loss_ce', 'NA')}`",
        f"- Train same-day loss: `{last.get('train/loss_same_day', 'NA')}`",
        f"- Train different-day loss: `{last.get('train/loss_dt', 'NA')}`",
        f"- Targets/sec: `{last.get('train/targets_per_second', 'NA')}`",
        f"- Max CUDA memory GB: `{last.get('system/max_cuda_memory_gb', 'NA')}`",
        "",
        "## Best sampled validation objective",
        "",
    ]
    if best:
        lines.extend([
            f"- Iteration: `{best['iter']}`",
            f"- Validation CE: `{best['val/loss_ce']}`",
            f"- Validation same-day loss: `{best['val/loss_same_day']}`",
            f"- Validation different-day loss: `{best['val/loss_dt']}`",
            f"- Validation objective: `{best['val/objective_loss']}`",
        ])
    else:
        lines.append("- No validation record was written.")
    if (output_dir / "evaluation.json").exists():
        evaluation = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
        clinical = evaluation["clinical_only_softmax"]
        waiting = evaluation["clinical_waiting_time"]
        same_day = evaluation["clinical_same_day"]
        lines.extend([
            "",
            "## Deterministic capped evaluation",
            "",
            f"- Clinical CE: `{clinical['cross_entropy']}`",
            f"- Clinical top-1: `{clinical['top1_accuracy']}`",
            f"- Clinical top-5: `{clinical.get('top5_accuracy')}`",
            f"- Different-day NLL: `{waiting['model_nll']}`",
            f"- Different-day MAE days: `{waiting['model_mae_days']}`",
            f"- Same-day AUROC: `{same_day.get('auroc')}`",
            f"- Same-day AUPRC: `{same_day.get('auprc')}`",
            f"- Same-day Brier: `{same_day.get('brier_score')}`",
        ])
    (output_dir / "benchmark_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(output_root)
    )
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    manifest, vocab_size = require_full_etl(args.data_dir)
    prepare_output(output_dir, args.overwrite)

    command = [
        sys.executable,
        "train.py",
        args.config,
        f"--dataset_dir={args.data_dir}",
        f"--out_dir={output_dir}",
        "--init_from=scratch",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        f"--vocab_size={vocab_size}",
        "--decoupled_time_head=True",
        "--two_stage_time_head=True",
        f"--loss_dt_weight={args.loss_dt_weight}",
        "--checkpoint_metric=objective",
    ]
    for key, value in {
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
        "loss_dt_warmup_iters": args.loss_dt_warmup_iters,
    }.items():
        add_override(command, key, value)
    if args.max_iters is not None:
        command.append(f"--lr_decay_iters={args.max_iters}")
    run(command, output_dir / "training.log")

    checkpoint = output_dir / "ckpt.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not produce {checkpoint}")
    if not args.skip_evaluation:
        eval_command = [
            sys.executable,
            "scripts/evaluate_snuh_checkpoint.py",
            "--ckpt",
            str(checkpoint),
            "--data-dir",
            str(args.data_dir),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--batch-size",
            str(args.eval_batch_size),
            "--max-patients",
            str(args.eval_max_patients),
            "--time-baseline-max-patients",
            str(args.time_baseline_max_patients),
            "--output",
            str(output_dir / "evaluation.json"),
        ]
        run(eval_command, output_dir / "evaluation.log")
    write_benchmark_summary(output_dir, manifest, args)
    print(f"Benchmark summary: {output_dir / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
