#!/usr/bin/env python3
"""Run a same-split evaluation suite across selected SNUH checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    Path("/home/khdp-user/workspace/fermat-data")
    / "etl"
    / "patient_100pct_seed_42_with_genomics_tokens"
)
METRIC_RULES = {
    "objective_best": ("val/objective_loss", "min"),
    "ce_best": ("val/loss_ce", "min"),
    "time_nll_best": ("val/loss_dt", "min"),
    "same_day_best": ("val/loss_same_day", "min"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--time-baseline-max-patients", type=int, default=20000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["left", "middle", "right"],
        choices=["left", "middle", "right"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def available_checkpoints(run_dir):
    checkpoints = {}
    for path in sorted(run_dir.glob("ckpt_*.pt")):
        stem = path.stem.removeprefix("ckpt_")
        if stem.isdigit():
            checkpoints[int(stem)] = path
    if (run_dir / "ckpt_latest.pt").exists():
        checkpoints["latest"] = run_dir / "ckpt_latest.pt"
    if (run_dir / "ckpt.pt").exists():
        checkpoints["objective_saved_best"] = run_dir / "ckpt.pt"
    return checkpoints


def select_metric_rows(metrics, checkpoint_steps):
    validation_rows = [
        row for row in metrics
        if isinstance(row.get("iter"), int)
        and any(metric in row for metric, _ in METRIC_RULES.values())
    ]
    selected = {}
    for label, (metric, direction) in METRIC_RULES.items():
        candidates = [
            row for row in validation_rows
            if metric in row and int(row["iter"]) in checkpoint_steps
        ]
        if not candidates:
            selected[label] = {
                "metric": metric,
                "status": "no_available_checkpoint_for_metric",
            }
            continue
        if direction != "min":
            raise ValueError(f"Unsupported direction: {direction}")
        best = min(candidates, key=lambda row: row[metric])
        selected[label] = {
            "metric": metric,
            "status": "selected",
            "iter": int(best["iter"]),
            "metric_value": best[metric],
        }
    return selected


def run_evaluation(args, label, checkpoint_path, output_path):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_snuh_checkpoint.py"),
        "--ckpt",
        str(checkpoint_path),
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--time-baseline-max-patients",
        str(args.time_baseline_max_patients),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--output",
        str(output_path),
        "--selectors",
        *args.selectors,
    ]
    if args.max_patients is not None:
        command.extend(["--max-patients", str(args.max_patients)])
    print(f"[{label}] + {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def metric_value(metrics, path):
    value = metrics
    for key in path:
        if value is None:
            return None
        value = value.get(key)
    return value


def write_summary(output_dir, manifest, evaluations):
    rows = []
    for label, payload in evaluations.items():
        metrics = payload["metrics"]
        rows.append({
            "label": label,
            "checkpoint": payload["checkpoint"],
            "checkpoint_step": metrics.get("checkpoint_step"),
            "clinical_ce": metric_value(
                metrics,
                ["clinical_only_softmax", "cross_entropy"],
            ),
            "clinical_top1": metric_value(
                metrics,
                ["clinical_only_softmax", "top1_accuracy"],
            ),
            "clinical_top5": metric_value(
                metrics,
                ["clinical_only_softmax", "top5_accuracy"],
            ),
            "time_nll": metric_value(
                metrics,
                ["clinical_waiting_time", "model_nll"],
            ),
            "time_mae_days": metric_value(
                metrics,
                ["clinical_waiting_time", "model_mae_days"],
            ),
            "same_day_auroc": metric_value(
                metrics,
                ["clinical_same_day", "auroc"],
            ),
            "same_day_auprc": metric_value(
                metrics,
                ["clinical_same_day", "auprc"],
            ),
            "same_day_brier": metric_value(
                metrics,
                ["clinical_same_day", "brier_score"],
            ),
        })

    csv_path = output_dir / "checkpoint_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Checkpoint Evaluation Suite",
        "",
        f"- Run directory: `{manifest['run_dir']}`",
        f"- Data directory: `{manifest['data_dir']}`",
        f"- Split: `{manifest['split']}`",
        f"- Selectors: `{', '.join(manifest['selectors'])}`",
        "",
        "## Selected Checkpoints",
        "",
        "| label | checkpoint | training metric | training value |",
        "|---|---|---:|---:|",
    ]
    for label, selection in manifest["selections"].items():
        lines.append(
            "| {label} | `{checkpoint}` | {metric} | {value} |".format(
                label=label,
                checkpoint=selection.get("checkpoint", "NA"),
                metric=selection.get("metric", "NA"),
                value=selection.get("metric_value", "NA"),
            )
        )
    lines.extend([
        "",
        "## Held-Out Evaluation",
        "",
        "| label | clinical CE | top1 | top5 | time NLL | time MAE days | same-day AUROC | same-day Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {label} | {clinical_ce} | {clinical_top1} | {clinical_top5} | "
            "{time_nll} | {time_mae_days} | {same_day_auroc} | "
            "{same_day_brier} |".format(**row)
        )
    (output_dir / "checkpoint_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.run_dir / f"paper_eval_{args.split}"
    )
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(args.run_dir / "metrics.jsonl")
    checkpoints = available_checkpoints(args.run_dir)
    step_checkpoints = {
        key: value for key, value in checkpoints.items() if isinstance(key, int)
    }
    selections = select_metric_rows(metrics, set(step_checkpoints))
    if "objective_saved_best" in checkpoints:
        selections["objective_saved_best"] = {
            "metric": "checkpoint_file",
            "status": "selected",
            "checkpoint": str(checkpoints["objective_saved_best"]),
        }
    if "latest" in checkpoints:
        selections["latest"] = {
            "metric": "checkpoint_file",
            "status": "selected",
            "checkpoint": str(checkpoints["latest"]),
        }

    evaluations = {}
    for label, selection in selections.items():
        if selection.get("status") != "selected":
            continue
        checkpoint_path = Path(selection["checkpoint"]) if "checkpoint" in selection else None
        if checkpoint_path is None:
            checkpoint_path = step_checkpoints[int(selection["iter"])]
            selection["checkpoint"] = str(checkpoint_path)
        output_path = output_dir / f"{label}.evaluation.json"
        run_evaluation(args, label, checkpoint_path, output_path)
        evaluations[label] = {
            "checkpoint": str(checkpoint_path),
            "metrics": json.loads(output_path.read_text(encoding="utf-8")),
        }

    if not evaluations:
        raise RuntimeError("No checkpoint evaluations were run")

    manifest = {
        "run_dir": str(args.run_dir),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "selectors": args.selectors,
        "max_patients": args.max_patients,
        "selections": selections,
    }
    (output_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_summary(output_dir, manifest, evaluations)
    print(f"Evaluation suite output: {output_dir}")


if __name__ == "__main__":
    main()
