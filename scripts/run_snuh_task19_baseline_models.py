#!/usr/bin/env python3
"""Run first disease-risk baseline models for Task 19.

This script trains simple logistic-regression baselines on the patient-level
labels and pre-index features created by the Task 19 setup scripts. It is meant
to establish the first benchmark score table before FERMAT embeddings are added.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - depends on local runtime
    pq = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "baseline_model_scores"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZONS = ["5y"]


FEATURE_SETS = {
    "age_sex": {
        "numeric": ["age_at_index"],
        "categorical": ["gender_concept_id"],
    },
    "age_sex_counts": {
        "numeric": [
            "age_at_index",
            "dx_rows",
            "rx_rows",
            "px_rows",
            "dx_unique_concepts",
            "rx_unique_concepts",
            "px_unique_concepts",
            "dx_active_days",
            "rx_active_days",
            "px_active_days",
            "clinical_rows",
            "clinical_unique_concepts",
            "clinical_active_days",
            "log1p_dx_rows",
            "log1p_rx_rows",
            "log1p_px_rows",
            "log1p_dx_unique_concepts",
            "log1p_rx_unique_concepts",
            "log1p_px_unique_concepts",
            "log1p_dx_active_days",
            "log1p_rx_active_days",
            "log1p_px_active_days",
            "log1p_clinical_rows",
            "log1p_clinical_unique_concepts",
            "log1p_clinical_active_days",
        ],
        "categorical": ["gender_concept_id"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizons", nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--phenotypes", nargs="*", default=None)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(FEATURE_SETS),
        default=["age_sex", "age_sex_counts"],
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def load_inputs(args):
    date = safe_date(args.index_date)
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{date}.parquet"
    feature_path = args.feature_dir / f"baseline_features_{date}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    if pq is not None:
        all_columns = pq.read_schema(label_path).names
    else:
        all_columns = pd.read_parquet(label_path).columns
    phenotypes = sorted(
        {
            col.split("__", 1)[1]
            for col in all_columns
            if col.startswith("eligible_") and "__" in col
        }
    )
    if args.phenotypes:
        requested = set(args.phenotypes)
        missing = sorted(requested - set(phenotypes))
        if missing:
            raise ValueError(f"Unknown phenotypes requested: {missing}")
        phenotypes = [p for p in phenotypes if p in requested]

    needed_label_cols = ["person_id", "split"]
    for phenotype in phenotypes:
        for horizon in args.horizons:
            needed_label_cols.append(f"eligible_{horizon}__{phenotype}")
            needed_label_cols.append(f"label_{horizon}__{phenotype}")
    labels = pd.read_parquet(label_path, columns=needed_label_cols)
    features = pd.read_parquet(feature_path)
    merged = labels.merge(features, on=["person_id", "split"], how="inner")
    if len(merged) != len(labels):
        raise RuntimeError(
            f"Feature merge changed row count: labels={len(labels):,} merged={len(merged):,}"
        )
    return merged, phenotypes, str(label_path), str(feature_path)


def fit_transformer(train, feature_set):
    spec = FEATURE_SETS[feature_set]
    transformer = {
        "feature_set": feature_set,
        "numeric": {},
        "categorical": {},
    }
    for column in spec["numeric"]:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=np.float64)
        median = np.nanmedian(values)
        if np.isnan(median):
            median = 0.0
        filled = np.where(np.isnan(values), median, values)
        mean = float(filled.mean())
        std = float(filled.std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        transformer["numeric"][column] = {
            "median": float(median),
            "mean": mean,
            "std": std,
        }
    for column in spec["categorical"]:
        values = train[column].astype("string").fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        transformer["categorical"][column] = categories
    return transformer


def transform_features(frame, transformer):
    columns = []
    names = []
    for column, stats in transformer["numeric"].items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values), stats["median"], values)
        values = (values - stats["mean"]) / stats["std"]
        columns.append(values.reshape(-1, 1))
        names.append(column)
    for column, categories in transformer["categorical"].items():
        values = frame[column].astype("string").fillna("__MISSING__")
        for category in categories:
            columns.append((values == category).to_numpy(dtype=np.float64).reshape(-1, 1))
            names.append(f"{column}={category}")
    if not columns:
        raise ValueError("No model features were produced")
    return np.concatenate(columns, axis=1), names


def sigmoid(values):
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_regression(x, y, max_iter, learning_rate=0.03, l2=1e-4):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_rows, n_cols = x.shape
    weights = np.zeros(n_cols, dtype=np.float64)
    bias = float(np.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = 0.0
    vb = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for step in range(1, max_iter + 1):
        logits = x @ weights + bias
        pred = sigmoid(logits)
        error = pred - y
        grad_w = (x.T @ error) / n_rows + l2 * weights
        grad_b = float(error.mean())

        mw = beta1 * mw + (1.0 - beta1) * grad_w
        vw = beta2 * vw + (1.0 - beta2) * (grad_w ** 2)
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vb = beta2 * vb + (1.0 - beta2) * (grad_b ** 2)

        mw_hat = mw / (1.0 - beta1 ** step)
        vw_hat = vw / (1.0 - beta2 ** step)
        mb_hat = mb / (1.0 - beta1 ** step)
        vb_hat = vb / (1.0 - beta2 ** step)

        weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
        bias -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

    return {"weights": weights, "bias": bias}


def predict_logistic(model, x):
    return sigmoid(np.asarray(x, dtype=np.float64) @ model["weights"] + model["bias"])


def binary_metrics(y_true, y_score, prefix=""):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = int(y_true.sum())
    total = int(len(y_true))
    negatives = total - positives
    prevalence = positives / total if total else np.nan
    result = {
        f"{prefix}n": total,
        f"{prefix}positives": positives,
        f"{prefix}negatives": negatives,
        f"{prefix}prevalence": prevalence,
    }
    if positives > 0 and negatives > 0:
        result[f"{prefix}auroc"] = auroc_score(y_true, y_score)
        result[f"{prefix}auprc"] = average_precision(y_true, y_score)
        result[f"{prefix}brier"] = float(np.mean((y_score - y_true) ** 2))
        clipped = np.clip(y_score, 1e-15, 1 - 1e-15)
        result[f"{prefix}log_loss"] = float(
            -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
        )
    else:
        result[f"{prefix}auroc"] = np.nan
        result[f"{prefix}auprc"] = np.nan
        result[f"{prefix}brier"] = np.nan
        result[f"{prefix}log_loss"] = np.nan
    for fraction in [0.01, 0.05, 0.10]:
        k = max(1, int(np.ceil(total * fraction))) if total else 0
        if k:
            order = np.argsort(-y_score)
            top_rate = float(y_true[order[:k]].mean())
        else:
            top_rate = np.nan
        key = int(fraction * 100)
        result[f"{prefix}top{key}_rate"] = top_rate
        result[f"{prefix}top{key}_enrichment"] = (
            top_rate / prevalence if prevalence and not np.isnan(top_rate) else np.nan
        )
    return result


def auroc_score(y_true, y_score):
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return np.nan
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / total_pos)


def calibration_bins(y_true, y_score, n_bins=10):
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "score": y_score})
    frame["bin"] = pd.qcut(frame["score"], q=n_bins, duplicates="drop")
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted=("score", "mean"),
            observed_rate=("y", "mean"),
            positives=("y", "sum"),
        )
        .reset_index(drop=True)
    )
    grouped.insert(0, "bin_id", range(len(grouped)))
    return grouped


def evaluate_split(data, split, model, transformer):
    part = data[data["split"].eq(split)]
    y = part["label"].astype(int).to_numpy()
    x, _ = transform_features(part, transformer)
    scores = predict_logistic(model, x)
    return binary_metrics(y, scores), calibration_bins(y, scores)


def main():
    args = parse_args()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    data, phenotypes, label_path, feature_path = load_inputs(args)
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"horizons={args.horizons}")

    metric_rows = []
    calibration_rows = []
    started_all = time.time()
    for horizon in args.horizons:
        for phenotype in phenotypes:
            eligible_col = f"eligible_{horizon}__{phenotype}"
            label_col = f"label_{horizon}__{phenotype}"
            task = data.loc[data[eligible_col].astype(bool)].copy()
            task["label"] = task[label_col].astype(int)
            split_counts = task.groupby("split")["label"].agg(["size", "sum"]).to_dict()
            if task["label"].nunique() < 2:
                log(f"[SKIP] {phenotype} {horizon}: only one class")
                continue

            for feature_set in args.feature_sets:
                started = time.time()
                train = task[task["split"].eq("train")]
                if train["label"].nunique() < 2:
                    log(f"[SKIP] {phenotype} {horizon} {feature_set}: train one class")
                    continue
                transformer = fit_transformer(train, feature_set)
                train_x, feature_names = transform_features(train, transformer)
                train_y = train["label"].astype(int).to_numpy()
                model = fit_logistic_regression(train_x, train_y, max_iter=args.max_iter)
                row = {
                    "index_date": args.index_date,
                    "horizon": horizon,
                    "phenotype": phenotype,
                    "feature_set": feature_set,
                    "fit_seconds": time.time() - started,
                    "train_rows": int(split_counts["size"].get("train", 0)),
                    "train_positives": int(split_counts["sum"].get("train", 0)),
                    "val_rows": int(split_counts["size"].get("val", 0)),
                    "val_positives": int(split_counts["sum"].get("val", 0)),
                    "test_rows": int(split_counts["size"].get("test", 0)),
                    "test_positives": int(split_counts["sum"].get("test", 0)),
                    "n_model_features": len(feature_names),
                }
                for split in ["train", "val", "test"]:
                    split_frame = task.loc[task["split"].eq(split)]
                    split_x, _ = transform_features(split_frame, transformer)
                    split_score = predict_logistic(model, split_x)
                    row.update(binary_metrics(
                        split_frame["label"].astype(int),
                        split_score,
                        prefix=f"{split}_",
                    ))
                    _, bins = evaluate_split(task, split, model, transformer)
                    bins.insert(0, "split", split)
                    bins.insert(0, "feature_set", feature_set)
                    bins.insert(0, "phenotype", phenotype)
                    bins.insert(0, "horizon", horizon)
                    bins.insert(0, "index_date", args.index_date)
                    calibration_rows.append(bins)
                metric_rows.append(row)
                log(
                    "[DONE] "
                    f"{phenotype} {horizon} {feature_set} "
                    f"test_auroc={row.get('test_auroc', np.nan):.4f} "
                    f"test_auprc={row.get('test_auprc', np.nan):.4f} "
                    f"{row['fit_seconds']:.1f}s"
                )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = args.output_dir / "baseline_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if calibration_rows:
        calibration = pd.concat(calibration_rows, ignore_index=True)
    else:
        calibration = pd.DataFrame()
    calibration_path = args.output_dir / "baseline_model_calibration_bins.csv"
    calibration.to_csv(calibration_path, index=False)

    manifest = {
        "label_path": label_path,
        "feature_path": feature_path,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizons": args.horizons,
        "phenotypes": phenotypes,
        "feature_sets": args.feature_sets,
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "calibration_bins": str(calibration_path),
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
