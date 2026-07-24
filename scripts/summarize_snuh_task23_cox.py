"""Summarize Task23 31-phenotype Cox survival results.

The Task23 Cox run writes:
  * `cox_survival_metrics.csv`            one row per (phenotype, model_set)
                                          with c_index and horizon_auc
  * `cox_survival_bootstrap_delta_ci.csv` per-phenotype delta and 95% CI of
                                          (fermat model - baseline) for each metric

This script reports, for the FERMAT model and the baseline:
  * mean / median / min / max of C-index and 5-year (horizon) AUROC over the 31 phenotypes
And from the bootstrap CI file, for each metric (c_index, horizon_auc):
  * count of phenotypes with point improvement
  * count with significant improvement (CI lower > 0)
  * count with significant worsening (CI upper < 0)
  * mean / median delta
  * the full per-phenotype delta [95% CI] table, sorted by delta

Usage on the Pod:

    python scripts/summarize_snuh_task23_cox.py \
      --dir /home/khdp-user/workspace/fermat-data/task23/outputs/\
cox_survival_2018_5y_31phenotypes_block2048_1000ci_20260704
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRIC_COLS = ["c_index", "horizon_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Task23 Cox output directory containing the two CSVs",
    )
    parser.add_argument("--model", default="cox_fermat_baseline")
    parser.add_argument("--baseline", default="cox_baseline")
    return parser.parse_args()


def aggregate(metrics: pd.DataFrame, model_set: str) -> pd.Series:
    sub = metrics[metrics["model_set"] == model_set]
    if sub.empty:
        raise SystemExit(f"No rows for model_set={model_set!r} in metrics file")
    stats = {"model_set": model_set, "phenotypes": int(sub["phenotype"].nunique())}
    for col in METRIC_COLS:
        if col in sub:
            stats[f"{col}_mean"] = sub[col].mean()
            stats[f"{col}_median"] = sub[col].median()
            stats[f"{col}_min"] = sub[col].min()
            stats[f"{col}_max"] = sub[col].max()
    return pd.Series(stats)


def summarize_ci(ci: pd.DataFrame, model_set: str, metric: str) -> None:
    sub = ci[(ci["model_set"] == model_set) & (ci["metric"] == metric)].copy()
    if sub.empty:
        print(f"  (no CI rows for {model_set} / {metric})")
        return
    n = sub["phenotype"].nunique()
    improved = int((sub["delta"] > 0).sum())
    sig_up = int((sub["ci95_lower"] > 0).sum())
    sig_down = int((sub["ci95_upper"] < 0).sum())
    print(f"  phenotypes            : {n}")
    print(f"  point improved        : {improved}/{n}")
    print(f"  significant improved  : {sig_up}/{n}")
    print(f"  significant worsened  : {sig_down}/{n}")
    print(f"  mean delta            : {sub['delta'].mean():+0.4f}")
    print(f"  median delta          : {sub['delta'].median():+0.4f}")
    print(f"  delta range           : {sub['delta'].min():+0.4f} .. {sub['delta'].max():+0.4f}")


def main() -> None:
    args = parse_args()
    metrics_path = args.dir / "cox_survival_metrics.csv"
    ci_path = args.dir / "cox_survival_bootstrap_delta_ci.csv"
    for path in (metrics_path, ci_path):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

    metrics = pd.read_csv(metrics_path)
    ci = pd.read_csv(ci_path)

    model_stats = aggregate(metrics, args.model)
    base_stats = aggregate(metrics, args.baseline)

    print("== Aggregate over all phenotypes (absolute values) ==")
    summary = pd.DataFrame([base_stats, model_stats]).set_index("model_set")
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(summary.T.to_string())

    for metric in METRIC_COLS:
        label = "5-year (horizon) AUROC" if metric == "horizon_auc" else "C-index"
        print()
        print(f"== {label}: {args.model} vs {args.baseline} (bootstrap delta) ==")
        summarize_ci(ci, args.model, metric)

    # full per-phenotype delta [CI] table for horizon AUROC and C-index
    for metric in METRIC_COLS:
        sub = ci[(ci["model_set"] == args.model) & (ci["metric"] == metric)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta", ascending=False)
        cols = [c for c in ["phenotype", "baseline", "model", "delta",
                            "ci95_lower", "ci95_upper"] if c in sub.columns]
        label = "5-year AUROC" if metric == "horizon_auc" else "C-index"
        print()
        print(f"== Per-phenotype {label} delta [95% CI] ==")
        with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
            print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
