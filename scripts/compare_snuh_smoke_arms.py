"""Compare baseline and LAB-context SNUH smoke outputs."""

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lab-context", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--lab-context-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested_get(document, path):
    value = document
    for key in path.split("."):
        value = value[key]
    return value


def read_metrics(path):
    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def training_summary(records):
    train = [row for row in records if "train/loss" in row]
    validation = [row for row in records if "val/objective_loss" in row]
    speeds = [
        row["train/targets_per_second"]
        for row in train[1:]
        if row.get("train/targets_per_second", 0) > 0
    ]
    return {
        "first_train_loss": train[0]["train/loss"] if train else None,
        "last_train_loss": train[-1]["train/loss"] if train else None,
        "best_val_objective": (
            min(row["val/objective_loss"] for row in validation)
            if validation
            else None
        ),
        "median_targets_per_second": (
            statistics.median(speeds) if speeds else None
        ),
        "max_cuda_memory_gb": (
            max(
                (row.get("system/max_cuda_memory_gb", 0) for row in train),
                default=0,
            )
        ),
    }


def display(value, fmt):
    return "NA" if value is None else format(value, fmt)


def main():
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    context = json.loads(args.lab_context.read_text(encoding="utf-8"))
    baseline_training = training_summary(read_metrics(args.baseline_metrics))
    context_training = training_summary(read_metrics(args.lab_context_metrics))
    rows = [
        ("Objective targets", "objective.targets", ".0f"),
        ("Objective CE", "objective.cross_entropy", ".4f"),
        (
            "Clinical-only CE",
            "clinical_only_softmax.cross_entropy",
            ".4f",
        ),
        (
            "Clinical-only top-1",
            "clinical_only_softmax.top1_accuracy",
            ".4%",
        ),
        ("New clinical top-1", "new_clinical.top1_accuracy", ".4%"),
        (
            "Repeated clinical top-1",
            "repeated_clinical.top1_accuracy",
            ".4%",
        ),
        (
            "Clinical time MAE (days)",
            "clinical_waiting_time.mae_days",
            ".2f",
        ),
    ]

    lines = [
        "# SNUH smoke arm comparison",
        "",
        "| metric | baseline | LAB-context |",
        "|---|---:|---:|",
    ]
    for label, path, fmt in rows:
        baseline_value = nested_get(baseline, path)
        context_value = nested_get(context, path)
        lines.append(
            f"| {label} | {format(baseline_value, fmt)} | "
            f"{format(context_value, fmt)} |"
        )
    lines.extend([
        "",
        "## Training run",
        "",
        "| metric | baseline | LAB-context |",
        "|---|---:|---:|",
        (
            "| First train objective | "
            f"{display(baseline_training['first_train_loss'], '.4f')} | "
            f"{display(context_training['first_train_loss'], '.4f')} |"
        ),
        (
            "| Last train objective | "
            f"{display(baseline_training['last_train_loss'], '.4f')} | "
            f"{display(context_training['last_train_loss'], '.4f')} |"
        ),
        (
            "| Best validation objective | "
            f"{display(baseline_training['best_val_objective'], '.4f')} | "
            f"{display(context_training['best_val_objective'], '.4f')} |"
        ),
        (
            "| Median effective targets/s | "
            f"{display(baseline_training['median_targets_per_second'], '.1f')} | "
            f"{display(context_training['median_targets_per_second'], '.1f')} |"
        ),
        (
            "| Max CUDA memory (GB) | "
            f"{display(baseline_training['max_cuda_memory_gb'], '.2f')} | "
            f"{display(context_training['max_cuda_memory_gb'], '.2f')} |"
        ),
    ])
    lines.extend([
        "",
        "Objective CE is not directly comparable because the target policies differ.",
        "Use clinical-only metrics for the same-target arm comparison.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
