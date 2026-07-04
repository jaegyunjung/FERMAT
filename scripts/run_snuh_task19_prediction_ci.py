#!/usr/bin/env python3
"""Save Task 19 test-set predictions and bootstrap CIs for model deltas."""

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
except ModuleNotFoundError:  # pragma: no cover - depends on Pod packages
    pq = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local help can run without torch
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_snuh_task19_baseline_models import (
    FEATURE_SETS,
    binary_metrics,
    fit_logistic_regression,
    fit_transformer as fit_count_transformer,
    predict_logistic,
    transform_features as transform_count_features,
)
from scripts.run_snuh_task19_embedding_models import (
    fit_torch_logistic,
    fit_transformer as fit_embedding_transformer,
    predict_torch,
    transform_features as transform_embedding_features,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "prediction_ci_2018_5y"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"
MODEL_SETS = ["age_sex_counts", "fermat_embedding", "fermat_embedding_counts"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--phenotypes", nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_torch():
    if torch is None:
        raise RuntimeError("torch is required for FERMAT embedding models on the Pod")


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def parquet_columns(path: Path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def resolve_embedding_file(args):
    if args.embedding_file is not None:
        path = args.embedding_file.expanduser().resolve()
    else:
        path = (
            args.embedding_dir.expanduser().resolve()
            / f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizon}_last.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_inputs(args):
    date = safe_date(args.index_date)
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{date}.parquet"
    feature_path = args.feature_dir / f"baseline_features_{date}.parquet"
    embedding_path = resolve_embedding_file(args)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    label_columns = parquet_columns(label_path)
    phenotypes = sorted(
        {
            column.split("__", 1)[1]
            for column in label_columns
            if column.startswith(f"eligible_{args.horizon}__")
        }
    )
    if args.phenotypes:
        requested = set(args.phenotypes)
        missing = sorted(requested - set(phenotypes))
        if missing:
            raise ValueError(f"Unknown phenotypes requested: {missing}")
        phenotypes = [p for p in phenotypes if p in requested]

    label_cols = ["person_id", "split"]
    for phenotype in phenotypes:
        label_cols.append(f"eligible_{args.horizon}__{phenotype}")
        label_cols.append(f"label_{args.horizon}__{phenotype}")
    labels = pd.read_parquet(label_path, columns=label_cols)

    embedding_columns = parquet_columns(embedding_path)
    emb_cols = [column for column in embedding_columns if column.startswith("emb_")]
    embedding_meta = [
        "person_id",
        "split",
        "has_embedding_sequence",
        "sequence_length_pre_index",
    ]
    embeddings = pd.read_parquet(embedding_path, columns=embedding_meta + emb_cols)
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()

    count_cols = (
        ["person_id", "split"]
        + FEATURE_SETS["age_sex_counts"]["numeric"]
        + FEATURE_SETS["age_sex_counts"]["categorical"]
    )
    features = pd.read_parquet(feature_path, columns=count_cols)
    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(features, on=["person_id", "split"], how="left")
    return data, phenotypes, emb_cols, str(label_path), str(feature_path), str(embedding_path)


def train_predict_model(task, model_set, emb_cols, args):
    train = task.loc[task["split"].eq("train")]
    if model_set == "age_sex_counts":
        transformer = fit_count_transformer(train, "age_sex_counts")
        train_x, feature_names = transform_count_features(train, transformer)
        train_y = train["label"].astype(int).to_numpy()
        model = fit_logistic_regression(
            train_x,
            train_y,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )

        def predict(frame):
            x, _ = transform_count_features(frame, transformer)
            return predict_logistic(model, x)

    elif model_set in {"fermat_embedding", "fermat_embedding_counts"}:
        transformer = fit_embedding_transformer(train, model_set, emb_cols)
        train_x, feature_names = transform_embedding_features(train, transformer)
        train_y = train["label"].astype(int).to_numpy()
        model = fit_torch_logistic(train_x, train_y, args)

        def predict(frame):
            x, _ = transform_embedding_features(frame, transformer)
            return predict_torch(model, x, args.device)

    else:
        raise ValueError(model_set)

    test = task.loc[task["split"].eq("test")].copy()
    return test, predict(test), len(feature_names)


def auroc_score(y_true, y_score):
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
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


def top5_enrichment(y_true, y_score):
    prevalence = float(np.mean(y_true))
    if prevalence <= 0:
        return np.nan
    k = max(1, int(np.ceil(len(y_true) * 0.05)))
    order = np.argsort(-y_score)
    return float(np.mean(y_true[order[:k]]) / prevalence)


def metric_value(y_true, y_score, metric):
    if metric == "auroc":
        return auroc_score(y_true, y_score)
    if metric == "auprc":
        return average_precision(y_true, y_score)
    if metric == "brier":
        return float(np.mean((y_score - y_true) ** 2))
    if metric == "top5_enrichment":
        return top5_enrichment(y_true, y_score)
    raise ValueError(metric)


def bootstrap_delta_ci(predictions, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    metrics = ["auroc", "auprc", "brier", "top5_enrichment"]
    for phenotype, sub in predictions.groupby("phenotype", sort=False):
        wide = sub.pivot(
            index="person_id",
            columns="model_set",
            values="y_score",
        )
        labels = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide.index, "y_true"]
            .to_numpy(dtype=np.int8)
        )
        if "age_sex_counts" not in wide.columns:
            continue
        baseline = wide["age_sex_counts"].to_numpy(dtype=np.float64)
        for model_set in ["fermat_embedding", "fermat_embedding_counts"]:
            if model_set not in wide.columns:
                continue
            scores = wide[model_set].to_numpy(dtype=np.float64)
            n = len(labels)
            for metric in metrics:
                observed_baseline = metric_value(labels, baseline, metric)
                observed_model = metric_value(labels, scores, metric)
                observed_delta = observed_model - observed_baseline
                boot = np.empty(args.bootstrap_samples, dtype=np.float64)
                for sample in range(args.bootstrap_samples):
                    index = rng.integers(0, n, size=n)
                    boot[sample] = (
                        metric_value(labels[index], scores[index], metric)
                        - metric_value(labels[index], baseline[index], metric)
                    )
                rows.append({
                    "phenotype": phenotype,
                    "model_set": model_set,
                    "metric": metric,
                    "n": int(n),
                    "positives": int(labels.sum()),
                    "baseline": observed_baseline,
                    "model": observed_model,
                    "delta": observed_delta,
                    "ci95_lower": float(np.nanpercentile(boot, 2.5)),
                    "ci95_upper": float(np.nanpercentile(boot, 97.5)),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    require_torch()

    started_all = time.time()
    data, phenotypes, emb_cols, label_path, feature_path, embedding_path = load_inputs(args)
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"embedding_features={len(emb_cols):,}")
    log(f"horizon={args.horizon}")

    metric_rows = []
    prediction_frames = []
    for phenotype in phenotypes:
        eligible_col = f"eligible_{args.horizon}__{phenotype}"
        label_col = f"label_{args.horizon}__{phenotype}"
        task = data.loc[data[eligible_col].astype(bool)].copy()
        task["label"] = task[label_col].astype(int)
        if task["label"].nunique() < 2:
            log(f"[SKIP] {phenotype}: only one class")
            continue
        train = task.loc[task["split"].eq("train")]
        if train["label"].nunique() < 2:
            log(f"[SKIP] {phenotype}: train one class")
            continue
        split_counts = task.groupby("split")["label"].agg(["size", "sum"]).to_dict()
        for model_set in MODEL_SETS:
            started = time.time()
            test, scores, n_features = train_predict_model(task, model_set, emb_cols, args)
            y_true = test["label"].astype(int).to_numpy()
            row = {
                "index_date": args.index_date,
                "horizon": args.horizon,
                "phenotype": phenotype,
                "model_set": model_set,
                "fit_seconds": time.time() - started,
                "train_rows": int(split_counts["size"].get("train", 0)),
                "train_positives": int(split_counts["sum"].get("train", 0)),
                "val_rows": int(split_counts["size"].get("val", 0)),
                "val_positives": int(split_counts["sum"].get("val", 0)),
                "test_rows": int(split_counts["size"].get("test", 0)),
                "test_positives": int(split_counts["sum"].get("test", 0)),
                "n_model_features": n_features,
            }
            row.update(binary_metrics(y_true, scores, prefix="test_"))
            metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "index_date": args.index_date,
                "horizon": args.horizon,
                "phenotype": phenotype,
                "model_set": model_set,
                "person_id": test["person_id"].to_numpy(),
                "split": "test",
                "y_true": y_true,
                "y_score": scores.astype(np.float32),
                "sequence_length_pre_index": test["sequence_length_pre_index"].to_numpy(),
            }))
            log(
                "[DONE] "
                f"{phenotype} {model_set} "
                f"test_auroc={row['test_auroc']:.4f} "
                f"test_auprc={row['test_auprc']:.4f} "
                f"{row['fit_seconds']:.1f}s"
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ci = bootstrap_delta_ci(predictions, args)

    metrics_path = args.output_dir / "prediction_metrics.csv"
    predictions_path = args.output_dir / "test_predictions.parquet"
    ci_path = args.output_dir / "bootstrap_delta_ci.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    ci.to_csv(ci_path, index=False)

    manifest = {
        "label_path": label_path,
        "feature_path": feature_path,
        "embedding_path": embedding_path,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "phenotypes": phenotypes,
        "model_sets": MODEL_SETS,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
            "bootstrap_delta_ci": str(ci_path),
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
