#!/usr/bin/env python3
"""Run Task22 sparse bag-of-codes LightGBM baselines.

This script tests whether FERMAT embeddings add predictive information beyond a
strong sparse pre-index code-frequency baseline. It builds a train-only
vocabulary from pre-index DX/RX/PX tokens and evaluates four model sets:

  B1: age + sex + utilization counts
  F1: B1 + FERMAT embedding
  B2: B1 + sparse bag-of-codes
  F2: B2 + FERMAT embedding + sparse bag-of-codes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover
    lgb = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None

try:
    from scipy import sparse
except ModuleNotFoundError:  # pragma: no cover
    sparse = None

POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task22" / "outputs" / "bag_of_codes_lightgbm_2018_5y_block2048"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"

COUNT_NUMERIC = [
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
]
COUNT_CATEGORICAL = ["gender_concept_id"]
DEFAULT_BAG_TOKEN_TYPES = [1, 2, 3]  # DX, RX, PX
MODEL_SETS = ["B1", "F1", "B2", "F2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--phenotypes", nargs="*", default=None)
    parser.add_argument("--model-sets", nargs="+", choices=MODEL_SETS, default=MODEL_SETS)
    parser.add_argument("--bag-token-types", nargs="+", type=int, default=DEFAULT_BAG_TOKEN_TYPES)
    parser.add_argument("--min-train-patients", type=int, default=200)
    parser.add_argument("--top-k-per-token-type", type=int, default=4000)
    parser.add_argument("--max-bag-features", type=int, default=10000)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def load_data(path: Path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.shape[0] % 4 == 0 and raw.shape[0] % 3 != 0:
        data = raw.reshape(-1, 4)
        has_types = True
    elif raw.shape[0] % 3 == 0 and raw.shape[0] % 4 != 0:
        data = raw.reshape(-1, 3)
        has_types = False
    else:
        data4 = raw.reshape(-1, 4)
        if data4[:, 3].max() < 20:
            data = data4
            has_types = True
        else:
            data = raw.reshape(-1, 3)
            has_types = False
    return data, has_types


def get_p2i(data):
    patient_ids = data[:, 0].astype(int)
    starts = []
    start = 0
    current = patient_ids[0]
    for index, patient_id in enumerate(patient_ids):
        if patient_id != current:
            starts.append([start, index - start])
            current = patient_id
            start = index
    starts.append([start, len(patient_ids) - start])
    return np.asarray(starts, dtype=np.int64)


def auroc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
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


def safe_date(index_date: str):
    return index_date.replace("-", "")


def require_packages():
    missing = []
    if lgb is None:
        missing.append("lightgbm")
    if sparse is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "Missing packages: "
            + ", ".join(missing)
            + ". On the Pod, install from the internal mirror before running."
        )


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def parquet_columns(path: Path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def load_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    frame["patient_id_dense"] = frame["patient_id_dense"].astype(np.int64)
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["split"] = frame["split"].astype(str)
    return frame


def load_inputs(args):
    date = safe_date(args.index_date)
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{date}.parquet"
    feature_path = args.feature_dir / f"baseline_features_{date}.parquet"
    embedding_path = args.embedding_file
    for path in [label_path, feature_path, embedding_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    label_columns = parquet_columns(label_path)
    phenotypes = sorted(
        column.split("__", 1)[1]
        for column in label_columns
        if column.startswith(f"eligible_{args.horizon}__")
    )
    if args.phenotypes:
        requested = set(args.phenotypes)
        missing = sorted(requested - set(phenotypes))
        if missing:
            raise ValueError(f"Unknown phenotypes requested: {missing}")
        phenotypes = [p for p in phenotypes if p in requested]

    label_cols = ["person_id", "split", "age_at_index"]
    for phenotype in phenotypes:
        label_cols.append(f"eligible_{args.horizon}__{phenotype}")
        label_cols.append(f"label_{args.horizon}__{phenotype}")
    labels = pd.read_parquet(label_path, columns=label_cols)
    eligible_cols = [f"eligible_{args.horizon}__{p}" for p in phenotypes]
    labels = labels.loc[labels[eligible_cols].astype(bool).any(axis=1)].copy()
    labels["index_age_days"] = np.floor(labels["age_at_index"] * 365.25).astype(np.int64)
    labels = labels.drop(columns=["age_at_index"])

    emb_cols = [c for c in parquet_columns(embedding_path) if c.startswith("emb_")]
    embeddings = pd.read_parquet(
        embedding_path,
        columns=["person_id", "split", "has_embedding_sequence", "sequence_length_pre_index"] + emb_cols,
    )
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()

    count_cols = ["person_id", "split"] + COUNT_NUMERIC + COUNT_CATEGORICAL
    features = pd.read_parquet(feature_path, columns=count_cols)

    patient_map = load_patient_map(args.data_dir)
    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(features, on=["person_id", "split"], how="left")
    missing_count_cols = [
        column for column in COUNT_NUMERIC + COUNT_CATEGORICAL
        if column not in data.columns
    ]
    if missing_count_cols:
        raise KeyError(f"Missing baseline feature columns after merge: {missing_count_cols}")
    data = data.merge(patient_map, on=["person_id", "split"], how="left")
    data = data.loc[data["patient_id_dense"].notna()].copy()
    data["patient_id_dense"] = data["patient_id_dense"].astype(np.int64)
    if args.max_patients and args.max_patients > 0:
        data = data.head(args.max_patients).copy()
    return data, phenotypes, emb_cols, str(label_path), str(feature_path), str(embedding_path)


def load_split_data(data_dir: Path):
    result = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        data, has_types = load_data(path)
        if not has_types:
            raise ValueError(f"{path} is not a 4-column typed FERMAT bin")
        p2i = get_p2i(data)
        patient_ids = data[p2i[:, 0].astype(np.int64), 0].astype(np.int64)
        result[split] = {
            "data": data,
            "index": {
                int(patient_id): (int(start), int(length))
                for patient_id, (start, length) in zip(patient_ids, p2i)
            },
        }
        log(f"loaded {split}: rows={len(data):,} patients={len(patient_ids):,}")
    return result


def preindex_token_counts(split_data, split: str, dense_id: int, index_age_days: int, token_types: set[int]):
    location = split_data[split]["index"].get(int(dense_id))
    if location is None:
        return Counter()
    start, length = location
    rows = split_data[split]["data"][start : start + length]
    rows = rows[rows[:, 1].astype(np.int64) < int(index_age_days)]
    if len(rows) == 0:
        return Counter()
    mask = np.isin(rows[:, 3].astype(np.int64), list(token_types))
    rows = rows[mask]
    if len(rows) == 0:
        return Counter()
    return Counter((int(tt), int(tok)) for tok, tt in zip(rows[:, 2], rows[:, 3]))


def build_train_vocab(targets: pd.DataFrame, split_data: dict, args):
    token_types = set(int(x) for x in args.bag_token_types)
    patient_counts = defaultdict(Counter)
    event_counts = Counter()
    train = targets.loc[targets["split"].eq("train")].copy()
    started = time.time()
    for i, row in enumerate(train.itertuples(index=False), 1):
        counts = preindex_token_counts(
            split_data,
            str(row.split),
            int(row.patient_id_dense),
            int(row.index_age_days),
            token_types,
        )
        for key, value in counts.items():
            patient_counts[key[0]][key] += 1
            event_counts[key] += int(value)
        if args.progress_every and i % args.progress_every == 0:
            log(f"[vocab] train patients processed={i:,}/{len(train):,}")

    rows = []
    for token_type, counter in patient_counts.items():
        for key, patients in counter.items():
            if patients < args.min_train_patients:
                continue
            rows.append({
                "token_type": int(token_type),
                "token_id": int(key[1]),
                "train_patients": int(patients),
                "train_events": int(event_counts[key]),
            })
    vocab = pd.DataFrame(rows)
    if vocab.empty:
        raise RuntimeError("No bag-of-codes features survived thresholds")
    vocab = (
        vocab.sort_values(["token_type", "train_patients", "train_events"], ascending=[True, False, False])
        .groupby("token_type", as_index=False, group_keys=False)
        .head(args.top_k_per_token_type)
        .sort_values(["train_patients", "train_events"], ascending=False)
        .head(args.max_bag_features)
        .reset_index(drop=True)
    )
    vocab["feature_index"] = np.arange(len(vocab), dtype=np.int32)
    log(f"[vocab] features={len(vocab):,} elapsed={time.time() - started:,.1f}s")
    return vocab


def build_bag_matrix(targets: pd.DataFrame, split_data: dict, vocab: pd.DataFrame, args):
    token_types = set(int(x) for x in args.bag_token_types)
    feature_lookup = {
        (int(row.token_type), int(row.token_id)): int(row.feature_index)
        for row in vocab.itertuples(index=False)
    }
    row_indices = []
    col_indices = []
    values = []
    started = time.time()
    for i, row in enumerate(targets.itertuples(index=False), 1):
        counts = preindex_token_counts(
            split_data,
            str(row.split),
            int(row.patient_id_dense),
            int(row.index_age_days),
            token_types,
        )
        for key, value in counts.items():
            col = feature_lookup.get(key)
            if col is None:
                continue
            row_indices.append(i - 1)
            col_indices.append(col)
            values.append(np.log1p(float(value)))
        if args.progress_every and i % args.progress_every == 0:
            log(f"[matrix] patients processed={i:,}/{len(targets):,}")
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, col_indices)),
        shape=(len(targets), len(vocab)),
        dtype=np.float32,
    )
    log(
        f"[matrix] shape={matrix.shape} nnz={matrix.nnz:,} "
        f"density={matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1):.6f} "
        f"elapsed={time.time() - started:,.1f}s"
    )
    return matrix


def fit_count_transformer(train: pd.DataFrame):
    transformer = {"numeric": {}, "categorical": {}}
    for column in COUNT_NUMERIC:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=np.float32)
        median = np.nanmedian(values)
        if not np.isfinite(median):
            median = 0.0
        transformer["numeric"][column] = float(median)
    for column in COUNT_CATEGORICAL:
        values = train[column].astype("string").fillna("__MISSING__")
        transformer["categorical"][column] = sorted(values.unique().tolist())
    return transformer


def transform_base(frame: pd.DataFrame, transformer: dict):
    blocks = []
    for column, median in transformer["numeric"].items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        values = np.where(np.isnan(values), median, values).astype(np.float32)
        blocks.append(values.reshape(-1, 1))
    for column, categories in transformer["categorical"].items():
        lookup = {value: i for i, value in enumerate(categories)}
        values = frame[column].astype("string").fillna("__MISSING__")
        codes = values.map(lookup).fillna(lookup.get("__MISSING__", 0)).to_numpy(dtype=np.float32)
        blocks.append(codes.reshape(-1, 1))
    return np.hstack(blocks).astype(np.float32) if blocks else np.empty((len(frame), 0), dtype=np.float32)


def model_features(frame: pd.DataFrame, row_pos: np.ndarray, emb_cols: list[str], bag_matrix, transformer, model_set: str):
    base_x = transform_base(frame, transformer)
    pieces = [sparse.csr_matrix(base_x)]
    if model_set in {"F1", "F2"}:
        emb = frame[emb_cols].to_numpy(dtype=np.float32)
        pieces.append(sparse.csr_matrix(emb))
    if model_set in {"B2", "F2"}:
        pieces.append(bag_matrix[row_pos])
    return sparse.hstack(pieces, format="csr", dtype=np.float32)


def lightgbm_params(args):
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "seed": args.random_seed,
        "feature_fraction_seed": args.random_seed,
        "bagging_seed": args.random_seed,
        "data_random_seed": args.random_seed,
        "num_threads": args.n_jobs,
        "verbosity": -1,
    }


def train_predict(task: pd.DataFrame, bag_matrix, emb_cols: list[str], model_set: str, args):
    train = task.loc[task["split"].eq("train")].copy()
    val = task.loc[task["split"].eq("val")].copy()
    test = task.loc[task["split"].eq("test")].copy()
    transformer = fit_count_transformer(train)
    train_x = model_features(train, train["row_pos"].to_numpy(), emb_cols, bag_matrix, transformer, model_set)
    val_x = model_features(val, val["row_pos"].to_numpy(), emb_cols, bag_matrix, transformer, model_set)
    test_x = model_features(test, test["row_pos"].to_numpy(), emb_cols, bag_matrix, transformer, model_set)
    train_y = train["label"].astype(int).to_numpy()
    val_y = val["label"].astype(int).to_numpy()

    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    valid_names = None
    if len(np.unique(val_y)) == 2 and args.early_stopping_rounds > 0:
        val_set = lgb.Dataset(val_x, label=val_y, reference=train_set, free_raw_data=False)
        valid_sets = [val_set]
        valid_names = ["val"]
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=False))
    model = lgb.train(
        lightgbm_params(args),
        train_set,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    best_iteration = model.best_iteration or args.n_estimators
    scores = model.predict(test_x, num_iteration=best_iteration)
    return test, scores, train_x.shape[1], best_iteration


def bootstrap_delta_ci(predictions: pd.DataFrame, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    metrics = ["auroc", "auprc", "brier", "top5_enrichment"]
    comparisons = [("F1", "B1"), ("B2", "B1"), ("F2", "B2"), ("F2", "B1")]
    for phenotype, sub in predictions.groupby("phenotype", sort=False):
        wide = sub.pivot(index="person_id", columns="model_set", values="y_score")
        labels = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide.index, "y_true"]
            .to_numpy(dtype=np.int8)
        )
        n = len(labels)
        for model_set, baseline_name in comparisons:
            if model_set not in wide.columns or baseline_name not in wide.columns:
                continue
            baseline = wide[baseline_name].to_numpy(dtype=np.float64)
            scores = wide[model_set].to_numpy(dtype=np.float64)
            for metric in metrics:
                observed_baseline = metric_value(labels, baseline, metric)
                observed_model = metric_value(labels, scores, metric)
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
                    "baseline_model_set": baseline_name,
                    "metric": metric,
                    "n": int(n),
                    "positives": int(labels.sum()),
                    "baseline": observed_baseline,
                    "model": observed_model,
                    "delta": observed_model - observed_baseline,
                    "ci95_lower": float(np.nanpercentile(boot, 2.5)),
                    "ci95_upper": float(np.nanpercentile(boot, 97.5)),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    require_packages()
    started_all = time.time()

    data, phenotypes, emb_cols, label_path, feature_path, embedding_path = load_inputs(args)
    data = data.reset_index(drop=True)
    data["row_pos"] = np.arange(len(data), dtype=np.int64)
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"embedding_features={len(emb_cols):,}")
    log(f"horizon={args.horizon}")
    log(f"lightgbm_version={lgb.__version__}")

    split_data = load_split_data(args.data_dir)
    vocab = build_train_vocab(data, split_data, args)
    bag_matrix = build_bag_matrix(data, split_data, vocab, args)

    vocab_path = args.output_dir / "bag_vocab.csv"
    vocab.to_csv(vocab_path, index=False)
    profile = {
        "rows": int(len(data)),
        "phenotypes": int(len(phenotypes)),
        "bag_features": int(len(vocab)),
        "bag_nnz": int(bag_matrix.nnz),
        "bag_density": float(bag_matrix.nnz / max(bag_matrix.shape[0] * bag_matrix.shape[1], 1)),
        "min_train_patients": int(args.min_train_patients),
        "top_k_per_token_type": int(args.top_k_per_token_type),
        "max_bag_features": int(args.max_bag_features),
        "bag_token_types": [int(x) for x in args.bag_token_types],
    }
    (args.output_dir / "bag_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    log("## BAG PROFILE")
    print(json.dumps(profile, indent=2), flush=True)
    if args.profile_only:
        return

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
        for model_set in args.model_sets:
            started = time.time()
            test, scores, n_features, best_iteration = train_predict(
                task,
                bag_matrix,
                emb_cols,
                model_set,
                args,
            )
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
                "n_model_features": int(n_features),
                "best_iteration": int(best_iteration or args.n_estimators),
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
                f"best_iter={row['best_iteration']} "
                f"{row['fit_seconds']:.1f}s"
            )

    if not prediction_frames:
        raise RuntimeError("No predictions were generated")
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ci = bootstrap_delta_ci(predictions, args)

    metrics_path = args.output_dir / "bag_lightgbm_metrics.csv"
    predictions_path = args.output_dir / "bag_lightgbm_test_predictions.parquet"
    ci_path = args.output_dir / "bag_lightgbm_bootstrap_delta_ci.csv"
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
        "model_sets": args.model_sets,
        "comparisons": [
            "F1_vs_B1",
            "B2_vs_B1",
            "F2_vs_B2",
            "F2_vs_B1",
        ],
        "bag_profile": profile,
        "lightgbm_version": lgb.__version__,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
            "bootstrap_delta_ci": str(ci_path),
            "bag_vocab": str(vocab_path),
            "bag_profile": str(args.output_dir / "bag_profile.json"),
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
