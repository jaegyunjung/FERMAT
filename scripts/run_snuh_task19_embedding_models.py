#!/usr/bin/env python3
"""Run Task 19 disease-risk models using FERMAT patient embeddings."""

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
    calibration_bins,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "embedding_model_scores"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZONS = ["5y"]


COUNT_NUMERIC = FEATURE_SETS["age_sex_counts"]["numeric"]
COUNT_CATEGORICAL = FEATURE_SETS["age_sex_counts"]["categorical"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizons", nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--phenotypes", nargs="*", default=None)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["fermat_embedding", "fermat_embedding_counts"],
        default=["fermat_embedding", "fermat_embedding_counts"],
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required for embedding logistic regression. "
            "Run this script in the Pod environment."
        )


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
            / f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizons[0]}_last.parquet"
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
            if column.startswith("eligible_") and "__" in column
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

    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    features = pd.read_parquet(
        feature_path,
        columns=["person_id", "split"] + COUNT_NUMERIC + COUNT_CATEGORICAL,
    )
    data = data.merge(features, on=["person_id", "split"], how="left")
    return data, phenotypes, emb_cols, str(label_path), str(feature_path), str(embedding_path)


def fit_transformer(train, feature_set, emb_cols):
    numeric_cols = list(emb_cols)
    categorical_cols = []
    if feature_set == "fermat_embedding_counts":
        numeric_cols += COUNT_NUMERIC
        categorical_cols += COUNT_CATEGORICAL

    transformer = {"numeric": {}, "categorical": {}, "feature_set": feature_set}
    for column in numeric_cols:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=np.float32)
        median = np.nanmedian(values)
        if np.isnan(median):
            median = 0.0
        filled = np.where(np.isnan(values), median, values)
        mean = float(filled.mean())
        std = float(filled.std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        transformer["numeric"][column] = {
            "median": float(median),
            "mean": mean,
            "std": std,
        }
    for column in categorical_cols:
        values = train[column].astype("string").fillna("__MISSING__")
        transformer["categorical"][column] = sorted(values.unique().tolist())
    return transformer


def transform_features(frame, transformer):
    arrays = []
    names = []
    for column, stats in transformer["numeric"].items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        values = np.where(np.isnan(values), stats["median"], values)
        values = (values - stats["mean"]) / stats["std"]
        arrays.append(values.reshape(-1, 1).astype(np.float32))
        names.append(column)
    for column, categories in transformer["categorical"].items():
        values = frame[column].astype("string").fillna("__MISSING__")
        for category in categories:
            arrays.append((values == category).to_numpy(dtype=np.float32).reshape(-1, 1))
            names.append(f"{column}={category}")
    if not arrays:
        raise ValueError("No features produced")
    return np.concatenate(arrays, axis=1), names


def fit_torch_logistic(x, y, args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    n_features = x_t.shape[1]
    linear = torch.nn.Linear(n_features, 1, bias=True, device=device)
    torch.nn.init.zeros_(linear.weight)
    prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    linear.bias.data.fill_(np.log(prior / (1 - prior)))
    opt = torch.optim.AdamW(linear.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    linear.train()
    for _ in range(args.max_iter):
        opt.zero_grad(set_to_none=True)
        logits = linear(x_t).squeeze(-1)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    return linear


def predict_torch(model, x, device):
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        score = torch.sigmoid(model(x_t).squeeze(-1)).detach().cpu().numpy()
    return score


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
    log(f"horizons={args.horizons}")

    metric_rows = []
    calibration_rows = []
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
            train = task[task["split"].eq("train")]
            if train["label"].nunique() < 2:
                log(f"[SKIP] {phenotype} {horizon}: train one class")
                continue

            for feature_set in args.feature_sets:
                started = time.time()
                transformer = fit_transformer(train, feature_set, emb_cols)
                train_x, feature_names = transform_features(train, transformer)
                train_y = train["label"].astype(int).to_numpy()
                model = fit_torch_logistic(train_x, train_y, args)
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
                    split_score = predict_torch(model, split_x, args.device)
                    row.update(binary_metrics(
                        split_frame["label"].astype(int).to_numpy(),
                        split_score,
                        prefix=f"{split}_",
                    ))
                    bins = calibration_bins(
                        split_frame["label"].astype(int).to_numpy(),
                        split_score,
                    )
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
    metrics_path = args.output_dir / "embedding_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if calibration_rows:
        calibration = pd.concat(calibration_rows, ignore_index=True)
    else:
        calibration = pd.DataFrame()
    calibration_path = args.output_dir / "embedding_model_calibration_bins.csv"
    calibration.to_csv(calibration_path, index=False)

    manifest = {
        "label_path": label_path,
        "feature_path": feature_path,
        "embedding_path": embedding_path,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizons": args.horizons,
        "phenotypes": phenotypes,
        "feature_sets": args.feature_sets,
        "embedding_features": len(emb_cols),
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
