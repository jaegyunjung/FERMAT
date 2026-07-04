#!/usr/bin/env python3
"""Run Task 20 clinical marker comparator models.

This compares FERMAT embeddings with raw clinical marker models for selected
incident disease-risk tasks. It intentionally reports three analysis cohorts:

* all_eligible: every eligible patient, with missing LAB values left missing
* marker_available: patients with at least one phenotype-specific marker
* marker_recent_2y: patients with at least one phenotype-specific marker
  measured within two years before the index date
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
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - depends on Pod packages
    lgb = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - depends on Pod packages
    pq = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_LAB_MARKER_DIR = POD_ROOT / "task20" / "outputs" / "lab_marker_features"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task20" / "outputs" / "clinical_comparator_2018_5y"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"
DEFAULT_PHENOTYPES = [
    "chronic_kidney_disease",
    "diabetes",
    "hepatocellular_carcinoma",
    "chronic_hepatitis_b",
    "fatty_liver",
]

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

PHENOTYPE_MARKERS = {
    "diabetes": {
        "markers": ["hba1c", "serum_glucose"],
        "availability_markers": ["hba1c"],
    },
    "chronic_kidney_disease": {
        "markers": ["serum_creatinine", "egfr_mdrd"],
        "availability_markers": ["egfr_mdrd"],
    },
    "hepatocellular_carcinoma": {
        "markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
        "availability_markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
    },
    "chronic_hepatitis_b": {
        "markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
        "availability_markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
    },
    "fatty_liver": {
        "markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
        "availability_markers": ["ast", "alt", "total_bilirubin", "platelet_count"],
    },
}

MARKER_VALUE_SUFFIXES = [
    "latest_value",
    "days_since_latest",
    "value_min",
    "value_max",
    "value_mean",
    "value_std",
    "value_mean_1y",
    "value_mean_2y",
    "lab_count",
    "lab_count_1y",
    "lab_count_2y",
]

MODEL_SETS = [
    "lgbm_counts_only",
    "lgbm_marker_only",
    "lgbm_marker_counts",
    "lgbm_fermat_embedding",
    "lgbm_fermat_embedding_counts",
    "lgbm_fermat_marker",
    "lgbm_fermat_marker_counts",
]

COHORTS = ["all_eligible", "marker_available", "marker_recent_2y"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--lab-marker-dir", type=Path, default=DEFAULT_LAB_MARKER_DIR)
    parser.add_argument("--lab-marker-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--phenotypes", nargs="+", default=DEFAULT_PHENOTYPES)
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=COHORTS)
    parser.add_argument("--model-sets", nargs="+", choices=MODEL_SETS, default=MODEL_SETS)
    parser.add_argument(
        "--ci-baselines",
        nargs="+",
        choices=MODEL_SETS,
        default=["lgbm_marker_counts", "lgbm_counts_only"],
    )
    parser.add_argument("--recent-days", type=int, default=730)
    parser.add_argument("--min-train-positives", type=int, default=20)
    parser.add_argument("--min-test-positives", type=int, default=20)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_lightgbm():
    if lgb is None:
        raise RuntimeError(
            "lightgbm is required. On the Pod, install it in the current Python "
            "environment or use the environment where `import lightgbm` works."
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
            / f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizon}_last.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_lab_marker_file(args):
    if args.lab_marker_file is not None:
        path = args.lab_marker_file.expanduser().resolve()
    else:
        path = (
            args.lab_marker_dir.expanduser().resolve()
            / f"lab_marker_features_wide_{safe_date(args.index_date)}.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def marker_columns(markers: list[str], available_columns: set[str]):
    columns = []
    missing = []
    for marker in markers:
        for suffix in MARKER_VALUE_SUFFIXES:
            column = f"lab_{marker}__{suffix}"
            if column in available_columns:
                columns.append(column)
            else:
                missing.append(column)
    if not columns:
        raise ValueError(f"No marker columns found for markers={markers}")
    return columns, missing


def load_inputs(args):
    date = safe_date(args.index_date)
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{date}.parquet"
    feature_path = args.feature_dir / f"baseline_features_{date}.parquet"
    embedding_path = resolve_embedding_file(args)
    lab_marker_path = resolve_lab_marker_file(args)
    for path in [label_path, feature_path, embedding_path, lab_marker_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    label_columns = parquet_columns(label_path)
    available_phenotypes = sorted(
        {
            column.split("__", 1)[1]
            for column in label_columns
            if column.startswith(f"eligible_{args.horizon}__")
        }
    )
    requested = set(args.phenotypes)
    unsupported = sorted(requested - set(PHENOTYPE_MARKERS))
    if unsupported:
        raise ValueError(f"No Task 20 marker config for phenotypes: {unsupported}")
    missing = sorted(requested - set(available_phenotypes))
    if missing:
        raise ValueError(f"Unknown phenotypes requested: {missing}")
    phenotypes = [p for p in available_phenotypes if p in requested]

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

    count_cols = ["person_id", "split"] + COUNT_NUMERIC + COUNT_CATEGORICAL
    count_features = pd.read_parquet(feature_path, columns=count_cols)

    lab_columns = parquet_columns(lab_marker_path)
    lab_cols = [column for column in lab_columns if column.startswith("lab_")]
    lab_features = pd.read_parquet(lab_marker_path, columns=["person_id", "split"] + lab_cols)
    for column in lab_cols:
        if "__lab_count" in column:
            lab_features[column] = pd.to_numeric(lab_features[column], errors="coerce").fillna(0)

    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(count_features, on=["person_id", "split"], how="left")
    data = data.merge(lab_features, on=["person_id", "split"], how="left")
    return {
        "data": data,
        "phenotypes": phenotypes,
        "embedding_columns": emb_cols,
        "lab_columns": set(lab_cols),
        "paths": {
            "label_path": str(label_path),
            "feature_path": str(feature_path),
            "embedding_path": str(embedding_path),
            "lab_marker_path": str(lab_marker_path),
        },
    }


def phenotype_marker_info(phenotype: str, available_columns: set[str]):
    config = PHENOTYPE_MARKERS[phenotype]
    markers = config["markers"]
    availability_markers = config["availability_markers"]
    feature_cols, missing_feature_cols = marker_columns(markers, available_columns)
    availability_count_cols = [
        f"lab_{marker}__lab_count"
        for marker in availability_markers
        if f"lab_{marker}__lab_count" in available_columns
    ]
    availability_recent_cols = [
        f"lab_{marker}__days_since_latest"
        for marker in availability_markers
        if f"lab_{marker}__days_since_latest" in available_columns
    ]
    if not availability_count_cols:
        raise ValueError(f"No availability marker count columns for phenotype={phenotype}")
    return {
        "markers": markers,
        "availability_markers": availability_markers,
        "marker_feature_columns": feature_cols,
        "missing_marker_feature_columns": missing_feature_cols,
        "availability_count_columns": availability_count_cols,
        "availability_recent_columns": availability_recent_cols,
    }


def add_cohort_flags(task: pd.DataFrame, marker_info: dict, recent_days: int):
    count_frame = task[marker_info["availability_count_columns"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    task = task.copy()
    task["cohort_all_eligible"] = True
    task["cohort_marker_available"] = count_frame.gt(0).any(axis=1)
    if marker_info["availability_recent_columns"]:
        recent_frame = task[marker_info["availability_recent_columns"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        task["cohort_marker_recent_2y"] = recent_frame.le(recent_days).any(axis=1)
    else:
        task["cohort_marker_recent_2y"] = False
    return task


def feature_columns(model_set: str, emb_cols: list[str], marker_cols: list[str]):
    if model_set == "lgbm_counts_only":
        return list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    if model_set == "lgbm_marker_only":
        return list(marker_cols), []
    if model_set == "lgbm_marker_counts":
        return list(marker_cols) + list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    if model_set == "lgbm_fermat_embedding":
        return list(emb_cols), []
    if model_set == "lgbm_fermat_embedding_counts":
        return list(emb_cols) + list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    if model_set == "lgbm_fermat_marker":
        return list(emb_cols) + list(marker_cols), []
    if model_set == "lgbm_fermat_marker_counts":
        return list(emb_cols) + list(marker_cols) + list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    raise ValueError(model_set)


def fit_transformer(train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]):
    transformer = {"numeric": list(numeric_cols), "categorical": {}}
    for column in categorical_cols:
        values = train[column].astype("string").fillna("__MISSING__")
        transformer["categorical"][column] = sorted(values.unique().tolist())
    return transformer


def transform_features(frame: pd.DataFrame, transformer: dict):
    data = {}
    categorical_feature = []
    for column in transformer["numeric"]:
        data[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
    for column, categories in transformer["categorical"].items():
        values = frame[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(categories), "__MISSING__")
        data[column] = pd.Categorical(values, categories=categories)
        categorical_feature.append(column)
    x = pd.DataFrame(data, index=frame.index)
    return x, categorical_feature


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


def train_predict_model(task, model_set, emb_cols, marker_cols, args):
    train = task.loc[task["split"].eq("train")]
    val = task.loc[task["split"].eq("val")]
    test = task.loc[task["split"].eq("test")].copy()
    numeric_cols, categorical_cols = feature_columns(model_set, emb_cols, marker_cols)
    transformer = fit_transformer(train, numeric_cols, categorical_cols)
    train_x, categorical_feature = transform_features(train, transformer)
    val_x, _ = transform_features(val, transformer)
    test_x, _ = transform_features(test, transformer)
    train_y = train["label"].astype(int).to_numpy()
    val_y = val["label"].astype(int).to_numpy()

    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        categorical_feature=categorical_feature or "auto",
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    valid_names = None
    if len(np.unique(val_y)) == 2 and args.early_stopping_rounds > 0:
        val_set = lgb.Dataset(
            val_x,
            label=val_y,
            categorical_feature=categorical_feature or "auto",
            reference=train_set,
            free_raw_data=False,
        )
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


def bootstrap_delta_ci(predictions: pd.DataFrame, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    metrics = ["auroc", "auprc", "brier", "top5_enrichment"]
    for (phenotype, cohort), sub in predictions.groupby(["phenotype", "analysis_cohort"], sort=False):
        wide = sub.pivot(index="person_id", columns="model_set", values="y_score")
        labels = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide.index, "y_true"]
            .to_numpy(dtype=np.int8)
        )
        n = len(labels)
        for baseline_name in args.ci_baselines:
            if baseline_name not in wide.columns:
                continue
            baseline = wide[baseline_name].to_numpy(dtype=np.float64)
            for model_set in args.model_sets:
                if model_set == baseline_name or model_set not in wide.columns:
                    continue
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
                        "analysis_cohort": cohort,
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
    args.label_dir = args.label_dir.expanduser().resolve()
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.lab_marker_dir = args.lab_marker_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    require_lightgbm()

    started_all = time.time()
    loaded = load_inputs(args)
    data = loaded["data"]
    phenotypes = loaded["phenotypes"]
    emb_cols = loaded["embedding_columns"]
    lab_columns = loaded["lab_columns"]
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"embedding_features={len(emb_cols):,}")
    log(f"horizon={args.horizon}")
    log(f"lightgbm_version={lgb.__version__}")

    metric_rows = []
    prediction_frames = []
    marker_config_rows = []
    skip_rows = []
    for phenotype in phenotypes:
        marker_info = phenotype_marker_info(phenotype, lab_columns)
        marker_cols = marker_info["marker_feature_columns"]
        marker_config_rows.append({
            "phenotype": phenotype,
            "markers": ",".join(marker_info["markers"]),
            "availability_markers": ",".join(marker_info["availability_markers"]),
            "marker_feature_columns": len(marker_cols),
            "missing_marker_feature_columns": ",".join(marker_info["missing_marker_feature_columns"]),
        })
        eligible_col = f"eligible_{args.horizon}__{phenotype}"
        label_col = f"label_{args.horizon}__{phenotype}"
        base_task = data.loc[data[eligible_col].astype(bool)].copy()
        base_task["label"] = base_task[label_col].astype(int)
        base_task = add_cohort_flags(base_task, marker_info, args.recent_days)
        if base_task["label"].nunique() < 2:
            log(f"[SKIP] {phenotype}: only one class")
            continue

        for cohort in args.cohorts:
            flag_col = f"cohort_{cohort}"
            task = base_task.loc[base_task[flag_col].astype(bool)].copy()
            if task.empty or task["label"].nunique() < 2:
                log(f"[SKIP] {phenotype} {cohort}: insufficient classes")
                continue
            split_counts = task.groupby("split")["label"].agg(["size", "sum"]).to_dict()
            train_pos = int(split_counts["sum"].get("train", 0))
            test_pos = int(split_counts["sum"].get("test", 0))
            if train_pos < args.min_train_positives or test_pos < args.min_test_positives:
                skip_rows.append({
                    "phenotype": phenotype,
                    "analysis_cohort": cohort,
                    "train_positives": train_pos,
                    "test_positives": test_pos,
                    "reason": "too_few_positives",
                })
                log(
                    f"[SKIP] {phenotype} {cohort}: "
                    f"train_pos={train_pos} test_pos={test_pos}"
                )
                continue
            for model_set in args.model_sets:
                started = time.time()
                test, scores, n_features, best_iteration = train_predict_model(
                    task,
                    model_set,
                    emb_cols,
                    marker_cols,
                    args,
                )
                y_true = test["label"].astype(int).to_numpy()
                row = {
                    "index_date": args.index_date,
                    "horizon": args.horizon,
                    "phenotype": phenotype,
                    "analysis_cohort": cohort,
                    "model_set": model_set,
                    "fit_seconds": time.time() - started,
                    "train_rows": int(split_counts["size"].get("train", 0)),
                    "train_positives": train_pos,
                    "val_rows": int(split_counts["size"].get("val", 0)),
                    "val_positives": int(split_counts["sum"].get("val", 0)),
                    "test_rows": int(split_counts["size"].get("test", 0)),
                    "test_positives": test_pos,
                    "n_model_features": int(n_features),
                    "best_iteration": int(best_iteration or args.n_estimators),
                    "marker_available_test_rows": int(test["cohort_marker_available"].sum()),
                    "marker_recent_2y_test_rows": int(test["cohort_marker_recent_2y"].sum()),
                }
                row.update(binary_metrics(y_true, scores, prefix="test_"))
                metric_rows.append(row)
                prediction_frames.append(pd.DataFrame({
                    "index_date": args.index_date,
                    "horizon": args.horizon,
                    "phenotype": phenotype,
                    "analysis_cohort": cohort,
                    "model_set": model_set,
                    "person_id": test["person_id"].to_numpy(),
                    "split": "test",
                    "y_true": y_true,
                    "y_score": scores.astype(np.float32),
                    "sequence_length_pre_index": test["sequence_length_pre_index"].to_numpy(),
                    "marker_available": test["cohort_marker_available"].astype(bool).to_numpy(),
                    "marker_recent_2y": test["cohort_marker_recent_2y"].astype(bool).to_numpy(),
                }))
                log(
                    "[DONE] "
                    f"{phenotype} {cohort} {model_set} "
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

    metrics_path = args.output_dir / "clinical_comparator_metrics.csv"
    predictions_path = args.output_dir / "clinical_comparator_test_predictions.parquet"
    ci_path = args.output_dir / "clinical_comparator_bootstrap_delta_ci.csv"
    marker_config_path = args.output_dir / "clinical_comparator_marker_config.csv"
    skip_path = args.output_dir / "clinical_comparator_skips.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    ci.to_csv(ci_path, index=False)
    pd.DataFrame(marker_config_rows).to_csv(marker_config_path, index=False)
    pd.DataFrame(skip_rows).to_csv(skip_path, index=False)

    manifest = {
        **loaded["paths"],
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "phenotypes": phenotypes,
        "cohorts": args.cohorts,
        "model_sets": args.model_sets,
        "ci_baselines": args.ci_baselines,
        "recent_days": args.recent_days,
        "lightgbm_version": lgb.__version__,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "lightgbm_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "subsample_freq": args.subsample_freq,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "n_jobs": args.n_jobs,
            "random_seed": args.random_seed,
        },
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
            "bootstrap_delta_ci": str(ci_path),
            "marker_config": str(marker_config_path),
            "skips": str(skip_path),
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
