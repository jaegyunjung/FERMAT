#!/usr/bin/env python3
"""Run the required checks before SNUH full-scale training."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_001pct_seed_42"
DEFAULT_OUTPUT_ROOT = POD_ROOT / "out"


def run(command, log_path):
    print("+", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_args", {})
    runtime_config = checkpoint.get("config", {})
    return checkpoint, config, runtime_config


def find_task13_run(output_root, expected_weight):
    candidates = []
    for checkpoint_path in output_root.glob("snuh_task13*/ckpt.pt"):
        run_dir = checkpoint_path.parent
        if not (run_dir / "metrics.jsonl").exists():
            continue
        try:
            checkpoint, model_config, runtime_config = load_checkpoint(
                checkpoint_path
            )
        except Exception as exc:
            print(f"Skipping unreadable checkpoint {checkpoint_path}: {exc}")
            continue
        weight = float(runtime_config.get("loss_dt_weight", math.nan))
        decoupled = bool(model_config.get("decoupled_time_head", False))
        if not decoupled or not math.isclose(
            weight,
            expected_weight,
            abs_tol=1e-8,
        ):
            continue
        candidates.append(
            (
                checkpoint_path.stat().st_mtime,
                int(checkpoint.get("iter_num", -1)),
                run_dir,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "No compatible Task 13 checkpoint was found below "
            f"{output_root}. Expected decoupled_time_head=True and "
            f"loss_dt_weight={expected_weight}."
        )
    return max(candidates)[2]


def read_validation_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "val/loss_ce" in row:
            rows.append(row)
    return rows


def pct(value):
    return f"{100.0 * value:.4f}%"


def write_summary(output_path, run_dir, audit, evaluation, validation):
    same_day = audit["same_day_target_summary"]
    waiting = evaluation["clinical_waiting_time"]
    lines = [
        "# SNUH pre-full-scale checks",
        "",
        f"- Source Task 13 run: `{run_dir}`",
        f"- Evaluated checkpoint step: `{evaluation['checkpoint_step']}`",
        f"- Time baseline source: `{waiting['baseline_source_split']}`",
        "",
        "## 1. Same-day ratio among sampled training targets",
        "",
        "| Target group | Targets | Same-day | Same-day rate |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("all", "All valid"),
        ("lab", "LAB"),
        ("clinical", "Clinical DX/RX/PX/DTH"),
        ("DX", "DX"),
        ("RX", "RX"),
        ("PX", "PX"),
        ("DTH", "DTH"),
    ):
        item = same_day[key]
        lines.append(
            f"| {label} | {item['targets']} | {item['same_day']} | "
            f"{pct(item['rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. Validation trajectory",
            "",
            "| step | CE | time loss | weighted objective | targets |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation:
        lines.append(
            f"| {row['iter']} | {row['val/loss_ce']:.4f} | "
            f"{row['val/loss_dt']:.4f} | "
            f"{row['val/objective_loss']:.4f} | "
            f"{row['val/eval_targets']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Train-only waiting-time baselines applied to validation",
            "",
            "| Metric | Model | Train-only baseline | Improvement |",
            "|---|---:|---:|---:|",
            f"| NLL | {waiting['model_nll']:.4f} | "
            f"{waiting['constant_rate_baseline_nll']:.4f} | "
            f"{waiting['nll_improvement_over_constant_rate']:+.4f} |",
            f"| Mean absolute error (days) | "
            f"{waiting['model_mae_days']:.1f} | "
            f"{waiting['median_baseline_mae_days']:.1f} | "
            f"{waiting['mae_improvement_over_median']:+.1f} |",
            "",
            f"- Train baseline targets: `{waiting['baseline_targets']}`",
            f"- Train mean gap: `{waiting['baseline_mean_gap_days']:.1f}` days",
            f"- Train median gap: `{waiting['baseline_median_gap_days']:.1f}` days",
            f"- Model beats both train-only baselines: "
            f"`{waiting['beats_baseline']}`",
            "",
            "Same-day rates use training windows sampled with the same random-window "
            "procedure as training. Time baselines are fitted only on deterministic "
            "train windows, then applied unchanged to validation targets.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--loss-dt-weight", type=float, default=0.3)
    parser.add_argument("--audit-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else find_task13_run(output_root, args.loss_dt_weight)
    )
    checkpoint_path = run_dir / "ckpt.pt"
    _, model_config, _ = load_checkpoint(checkpoint_path)
    block_size = int(model_config["block_size"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"snuh_pre_full_scale_checks_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Source Task 13 run: {run_dir}")
    print(f"Output directory: {output_dir}")

    audit_path = output_dir / "clinical_same_day_audit.json"
    evaluation_path = output_dir / "evaluation_train_baseline.json"
    log_path = output_dir / "pre_full_scale_checks.log"

    run(
        [
            sys.executable,
            "scripts/audit_snuh_training_windows.py",
            "--data-dir",
            str(data_dir),
            "--split",
            "train",
            "--block-size",
            str(block_size),
            "--batch-size",
            str(args.batch_size),
            "--batches",
            str(args.audit_batches),
            "--output",
            str(audit_path),
        ],
        log_path,
    )
    run(
        [
            sys.executable,
            "scripts/evaluate_snuh_checkpoint.py",
            "--ckpt",
            str(checkpoint_path),
            "--data-dir",
            str(data_dir),
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(evaluation_path),
        ],
        log_path,
    )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    validation = read_validation_rows(run_dir / "metrics.jsonl")
    summary_path = output_dir / "summary.md"
    write_summary(summary_path, run_dir, audit, evaluation, validation)
    print(f"Pre-full-scale report: {summary_path}")


if __name__ == "__main__":
    main()
