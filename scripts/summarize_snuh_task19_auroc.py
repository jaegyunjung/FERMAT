"""Summarize Task19 downstream AUROC across all phenotypes from lightgbm_metrics.csv.

The Task19 LightGBM run writes one row per (phenotype, model_set) to
`lightgbm_metrics.csv`. This script aggregates those rows so the full
31-phenotype mean/median AUROC (and AUPRC/Brier) can be reported instead of the
partial subset quoted in the narrative report.

It reports, for the FERMAT model and the age/sex/counts baseline:
  * number of phenotypes evaluated
  * simple mean, median, min, max of test AUROC / AUPRC / Brier
  * positives-weighted mean AUROC (large phenotypes weigh more)
  * per-phenotype AUROC delta (FERMAT model minus baseline)

Usage on the Pod (defaults point at the block512 all-cohort run):

    python scripts/summarize_snuh_task19_auroc.py

Point at a different run (e.g. the FERMAT-2048 / block2048 output) with:

    python scripts/summarize_snuh_task19_auroc.py \
        --metrics /home/khdp-user/workspace/fermat-data/task21/outputs/\
lightgbm_ci_2018_5y_block2048_best/lightgbm_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_METRICS = (
    POD_ROOT / "task19" / "outputs" / "lightgbm_ci_2018_5y_all" / "lightgbm_metrics.csv"
)

METRIC_COLS = ["test_auroc", "test_auprc", "test_brier"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Path to lightgbm_metrics.csv",
    )
    parser.add_argument(
        "--model",
        default="lgbm_fermat_embedding_counts",
        help="FERMAT model_set to summarize",
    )
    parser.add_argument(
        "--baseline",
        default="lgbm_age_sex_counts",
        help="Baseline model_set to compare against",
    )
    return parser.parse_args()


def summarize(frame: pd.DataFrame, model_set: str) -> pd.Series:
    sub = frame[frame["model_set"] == model_set]
    if sub.empty:
        raise SystemExit(f"No rows for model_set={model_set!r} in metrics file")
    stats = {"model_set": model_set, "phenotypes": int(sub["phenotype"].nunique())}
    for col in METRIC_COLS:
        if col not in sub:
            continue
        stats[f"{col}_mean"] = sub[col].mean()
        stats[f"{col}_median"] = sub[col].median()
        stats[f"{col}_min"] = sub[col].min()
        stats[f"{col}_max"] = sub[col].max()
    # positives-weighted mean AUROC (bigger phenotypes count more)
    if "test_positives" in sub and sub["test_positives"].sum() > 0:
        stats["test_auroc_weighted_mean"] = (
            (sub["test_auroc"] * sub["test_positives"]).sum()
            / sub["test_positives"].sum()
        )
    return pd.Series(stats)


def main() -> None:
    args = parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    frame = pd.read_csv(args.metrics)
    required = {"phenotype", "model_set", "test_auroc"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Metrics file missing columns: {sorted(missing)}")

    model_stats = summarize(frame, args.model)
    base_stats = summarize(frame, args.baseline)

    print(f"metrics file : {args.metrics}")
    print(f"index_date   : {frame.get('index_date', pd.Series(['?'])).iloc[0]}")
    print(f"horizon      : {frame.get('horizon', pd.Series(['?'])).iloc[0]} year(s)")
    print()
    print("== Aggregate over all phenotypes ==")
    summary = pd.DataFrame([base_stats, model_stats]).set_index("model_set")
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(summary.T.to_string())

    # Per-phenotype AUROC delta (model minus baseline)
    pivot = (
        frame[frame["model_set"].isin([args.model, args.baseline])]
        .pivot(index="phenotype", columns="model_set", values="test_auroc")
        .dropna()
    )
    pivot["delta_auroc"] = pivot[args.model] - pivot[args.baseline]
    pivot = pivot.sort_values("delta_auroc", ascending=False)

    print()
    print("== Per-phenotype test AUROC (sorted by delta vs baseline) ==")
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(pivot.to_string())

    print()
    print(
        f"FERMAT model ({args.model}) mean AUROC over "
        f"{int(model_stats['phenotypes'])} phenotypes : "
        f"{model_stats['test_auroc_mean']:0.4f} "
        f"(median {model_stats['test_auroc_median']:0.4f})"
    )
    print(
        f"Baseline ({args.baseline}) mean AUROC : "
        f"{base_stats['test_auroc_mean']:0.4f} "
        f"(median {base_stats['test_auroc_median']:0.4f})"
    )
    print(
        f"Mean AUROC gain : "
        f"{model_stats['test_auroc_mean'] - base_stats['test_auroc_mean']:+0.4f}"
    )


if __name__ == "__main__":
    main()
