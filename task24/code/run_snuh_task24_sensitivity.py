#!/usr/bin/env python3
"""Task24 sensitivity analysis for downstream prediction outputs.

This script does not train new models. It reads saved test predictions and
baseline feature tables, then re-computes model deltas across utilization,
history-length, early-event-exclusion, and detection-sensitive strata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_PREDICTIONS = (
    POD_ROOT
    / "task22"
    / "outputs"
    / "bag_of_codes_lightgbm_2018_5y_block2048_1000ci_20260704"
    / "bag_lightgbm_test_predictions.parquet"
)
DEFAULT_FEATURE_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "baseline_features"
    / "baseline_features_20180101.parquet"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task23"
    / "outputs"
    / "first_phenotype_dates_20180101_31phenotypes_full.parquet"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task24" / "outputs" / "task24_sensitivity_2018_5y_20260704"
DEFAULT_INDEX_DATE = "2018-01-01"
DETECTION_SENSITIVE = ["colon_polyp", "gallbladder_polyp", "cataract"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def auroc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return np.nan
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / total_pos)


def top5_enrichment(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(np.mean(y_true))
    if prevalence <= 0:
        return np.nan
    k = max(1, int(np.ceil(len(y_true) * 0.05)))
    order = np.argsort(-np.asarray(y_score).astype(float))
    return float(np.mean(y_true[order[:k]]) / prevalence)


def metric_value(y_true, y_score, metric):
    if metric == "auroc":
        return auroc_score(y_true, y_score)
    if metric == "auprc":
        return average_precision(y_true, y_score)
    if metric == "brier":
        return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))
    if metric == "top5_enrichment":
        return top5_enrichment(y_true, y_score)
    raise ValueError(metric)


def load_inputs(args):
    for path in [args.predictions, args.feature_file, args.survival_cache]:
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_parquet(args.predictions)
    needed_prediction_cols = {
        "phenotype",
        "model_set",
        "person_id",
        "y_true",
        "y_score",
    }
    missing = needed_prediction_cols - set(predictions.columns)
    if missing:
        raise KeyError(f"Prediction file missing columns: {sorted(missing)}")

    feature_cols = [
        "person_id",
        "split",
        "age_at_index",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
        "clinical_rows",
        "clinical_unique_concepts",
        "clinical_active_days",
        "dx_rows",
        "rx_rows",
        "px_rows",
    ]
    features = pd.read_parquet(args.feature_file, columns=feature_cols)
    features = features.loc[features["split"].eq("test")].copy()
    features["first_activity_date"] = pd.to_datetime(features["first_activity_date"], errors="coerce")
    features["last_activity_date"] = pd.to_datetime(features["last_activity_date"], errors="coerce")
    index_date = pd.Timestamp(args.index_date)
    features["history_days"] = (index_date - features["first_activity_date"]).dt.days.clip(lower=0)
    features["recency_days"] = (index_date - features["last_activity_date"]).dt.days.clip(lower=0)

    # Global test-set deciles keep strata comparable across phenotypes.
    features["clinical_rows_decile"] = pd.qcut(
        features["clinical_rows"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    ).astype(str)
    decile_num = features["clinical_rows_decile"].str.extract(r"D(\d+)")[0].astype(int)
    features["clinical_rows_band"] = np.select(
        [decile_num <= 3, decile_num <= 7],
        ["low_D1_D3", "mid_D4_D7"],
        default="high_D8_D10",
    )
    features["history_band"] = pd.cut(
        features["history_days"],
        bins=[-1, 365, 3 * 365, 5 * 365, np.inf],
        labels=["lt_1y", "1_3y", "3_5y", "ge_5y"],
    ).astype(str)

    survival = pd.read_parquet(args.survival_cache)
    survival["first_phenotype_date"] = pd.to_datetime(
        survival["first_phenotype_date"],
        errors="coerce",
    )
    survival["days_to_event"] = (survival["first_phenotype_date"] - index_date).dt.days
    return predictions, features, survival


def add_features(predictions, features, survival):
    data = predictions.merge(
        features[
            [
                "person_id",
                "clinical_rows",
                "clinical_unique_concepts",
                "clinical_active_days",
                "history_days",
                "recency_days",
                "clinical_rows_decile",
                "clinical_rows_band",
                "history_band",
                "has_pre_index_washout",
            ]
        ],
        on="person_id",
        how="left",
    )
    data = data.merge(
        survival[["phenotype", "person_id", "days_to_event"]],
        on=["phenotype", "person_id"],
        how="left",
    )
    data["early_event_90d"] = data["y_true"].astype(int).eq(1) & data["days_to_event"].between(0, 90)
    data["early_event_180d"] = data["y_true"].astype(int).eq(1) & data["days_to_event"].between(0, 180)
    data["phenotype_group"] = np.where(
        data["phenotype"].isin(DETECTION_SENSITIVE),
        "detection_sensitive",
        "other",
    )
    return data


def stratum_masks(data: pd.DataFrame):
    yield "all", "all", np.ones(len(data), dtype=bool)
    for value in sorted(data["clinical_rows_band"].dropna().unique()):
        yield "clinical_rows_band", str(value), data["clinical_rows_band"].eq(value).to_numpy()
    for value in [f"D{i}" for i in range(1, 11)]:
        yield "clinical_rows_decile", value, data["clinical_rows_decile"].eq(value).to_numpy()
    for value in ["lt_1y", "1_3y", "3_5y", "ge_5y"]:
        yield "history_band", value, data["history_band"].eq(value).to_numpy()
    for value in ["detection_sensitive", "other"]:
        yield "phenotype_group", value, data["phenotype_group"].eq(value).to_numpy()
    yield "exclude_early_event", "exclude_90d", ~data["early_event_90d"].to_numpy()
    yield "exclude_early_event", "exclude_180d", ~data["early_event_180d"].to_numpy()


def evaluate_delta(labels, baseline, model, bootstrap_samples, rng):
    rows = []
    metrics = ["auroc", "auprc", "brier", "top5_enrichment"]
    labels = np.asarray(labels, dtype=np.int8)
    baseline = np.asarray(baseline, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    n = len(labels)
    events = int(labels.sum())
    if n < 20 or events == 0 or events == n:
        return []
    for metric in metrics:
        base_value = metric_value(labels, baseline, metric)
        model_value = metric_value(labels, model, metric)
        if not (np.isfinite(base_value) and np.isfinite(model_value)):
            continue
        if bootstrap_samples > 0:
            boot = np.empty(bootstrap_samples, dtype=np.float64)
            for sample in range(bootstrap_samples):
                index = rng.integers(0, n, size=n)
                boot[sample] = (
                    metric_value(labels[index], model[index], metric)
                    - metric_value(labels[index], baseline[index], metric)
                )
            lo = float(np.nanpercentile(boot, 2.5))
            hi = float(np.nanpercentile(boot, 97.5))
        else:
            lo = np.nan
            hi = np.nan
        rows.append(
            {
                "metric": metric,
                "n": int(n),
                "events": events,
                "baseline": float(base_value),
                "model": float(model_value),
                "delta": float(model_value - base_value),
                "ci95_lower": lo,
                "ci95_upper": hi,
            }
        )
    return rows


def run_sensitivity(data, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = [
        ("F1_vs_B1", "F1", "B1"),
        ("B2_vs_B1", "B2", "B1"),
        ("F2_vs_B2", "F2", "B2"),
        ("F2_vs_B1", "F2", "B1"),
    ]
    rows = []
    for phenotype, sub in data.groupby("phenotype", sort=True):
        wide = sub.pivot_table(
            index="person_id",
            columns="model_set",
            values="y_score",
            aggfunc="first",
        )
        meta = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide.index]
            .reset_index()
        )
        for comparison, model_set, baseline_set in comparisons:
            if model_set not in wide.columns or baseline_set not in wide.columns:
                continue
            labels = meta["y_true"].to_numpy(dtype=np.int8)
            model_scores = wide[model_set].to_numpy(dtype=np.float64)
            baseline_scores = wide[baseline_set].to_numpy(dtype=np.float64)
            for stratum_type, stratum, mask in stratum_masks(meta):
                metric_rows = evaluate_delta(
                    labels[mask],
                    baseline_scores[mask],
                    model_scores[mask],
                    args.bootstrap_samples,
                    rng,
                )
                for row in metric_rows:
                    row.update(
                        {
                            "phenotype": phenotype,
                            "comparison": comparison,
                            "model_set": model_set,
                            "baseline_model_set": baseline_set,
                            "stratum_type": stratum_type,
                            "stratum": stratum,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame):
    rows = []
    for (comparison, metric, stratum_type, stratum), sub in results.groupby(
        ["comparison", "metric", "stratum_type", "stratum"],
        sort=True,
    ):
        if metric == "brier":
            point_better = int((sub["delta"] < 0).sum())
            sig_better = int((sub["ci95_upper"] < 0).sum())
            sig_worse = int((sub["ci95_lower"] > 0).sum())
        else:
            point_better = int((sub["delta"] > 0).sum())
            sig_better = int((sub["ci95_lower"] > 0).sum())
            sig_worse = int((sub["ci95_upper"] < 0).sum())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "phenotype_rows": int(len(sub)),
                "point_better": point_better,
                "sig_better": sig_better,
                "sig_worse": sig_worse,
                "median_delta": float(sub["delta"].median()),
                "median_n": float(sub["n"].median()),
                "median_events": float(sub["events"].median()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    started = time.time()
    predictions, features, survival = load_inputs(args)
    data = add_features(predictions, features, survival)
    log(f"predictions_rows={len(predictions):,}")
    log(f"joined_rows={len(data):,}")
    log(f"phenotypes={data['phenotype'].nunique():,}")
    log(f"model_sets={sorted(data['model_set'].unique().tolist())}")

    results = run_sensitivity(data, args)
    summary = summarize(results)
    results_path = args.output_dir / "task24_sensitivity_results.csv"
    summary_path = args.output_dir / "task24_sensitivity_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "predictions": str(args.predictions),
        "feature_file": str(args.feature_file),
        "survival_cache": str(args.survival_cache),
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "detection_sensitive": DETECTION_SENSITIVE,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
