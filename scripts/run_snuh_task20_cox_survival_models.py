#!/usr/bin/env python3
"""Run Task 20 APOLLO-style Cox survival comparator models.

This script complements the Task 19/20 binary classifiers. It keeps the same
fixed index-date framing, but uses censored follow-up rather than requiring
complete 5-year observation. Cox coefficients are fit with a ridge penalty in
PyTorch so the Pod does not need an additional survival-analysis package.
"""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    pq = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    torch = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_LAB_MARKER_DIR = POD_ROOT / "task20" / "outputs" / "lab_marker_features"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task20" / "outputs" / "cox_survival_2018_5y"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"
DEFAULT_DB_END_DATE = "2025-02-05"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task20_cox_survival"

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
    "cox_baseline",
    "cox_clinical_baseline",
    "cox_fermat_embedding",
    "cox_fermat_baseline",
    "cox_fermat_clinical_baseline",
]


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
    parser.add_argument("--horizon", default=DEFAULT_HORIZON, choices=["1y", "3y", "5y"])
    parser.add_argument("--db-end-date", default=DEFAULT_DB_END_DATE)
    parser.add_argument("--phenotypes", nargs="+", default=list(PHENOTYPE_MARKERS))
    parser.add_argument("--model-sets", nargs="+", choices=MODEL_SETS, default=MODEL_SETS)
    parser.add_argument(
        "--survival-cache",
        type=Path,
        help="Optional first-phenotype-date cache. Created if missing.",
    )
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-test-events", type=int, default=20)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Optional deterministic cap for sanity checks. 0 means no cap.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="Optional deterministic cap for val/test metrics. 0 means no cap.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required. On the Pod, install it from the internal "
            "PyTorch mirror before running this script."
        )


def require_psycopg():
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required when the survival cache is missing. Install "
            "with `python -m pip install \"psycopg[binary]>=3\"`."
        )


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    if value:
        return value
    return getpass.getpass("SNUH_CDM_PASSWORD: ")


def connect(args):
    require_psycopg()
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        application_name=APPLICATION_NAME,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def horizon_days(index_date_text: str, horizon: str):
    years = {"1y": 1, "3y": 3, "5y": 5}[horizon]
    index = pd.Timestamp(index_date_text)
    return int((index + pd.DateOffset(years=years) - index).days)


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
    if not availability_count_cols:
        raise ValueError(f"No availability marker count columns for phenotype={phenotype}")
    return {
        "markers": markers,
        "availability_markers": availability_markers,
        "marker_feature_columns": feature_cols,
        "missing_marker_feature_columns": missing_feature_cols,
        "availability_count_columns": availability_count_cols,
    }


def load_concept_map(label_dir: Path):
    path = label_dir / "phenotype_group_concept_map.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required to build the survival cache. Run the Task 19 "
            "label builder first, or pass --survival-cache to an existing cache."
        )
    frame = pd.read_csv(path)
    required = {"phenotype", "condition_concept_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["condition_concept_id"] = frame["condition_concept_id"].astype(np.int64)
    return frame.drop_duplicates(["phenotype", "condition_concept_id"])


def build_survival_cache(args, labels: pd.DataFrame, phenotypes: list[str], output_path: Path):
    concept_map = load_concept_map(args.label_dir)
    concept_map = concept_map.loc[concept_map["phenotype"].isin(phenotypes)].copy()
    if concept_map.empty:
        raise ValueError("No phenotype concepts available for requested phenotypes")

    require_psycopg()
    patient_ids = labels["person_id"].drop_duplicates().astype(np.int64).tolist()
    started = time.time()
    with connect(args) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS tmp_task20_cox_patient")
            cur.execute(
                """
                CREATE TEMP TABLE tmp_task20_cox_patient (
                    person_id bigint PRIMARY KEY
                ) ON COMMIT PRESERVE ROWS
                """
            )
            cur.executemany(
                "INSERT INTO tmp_task20_cox_patient VALUES (%s)",
                [(int(person_id),) for person_id in patient_ids],
            )
            cur.execute("DROP TABLE IF EXISTS tmp_task20_cox_concept")
            cur.execute(
                """
                CREATE TEMP TABLE tmp_task20_cox_concept (
                    phenotype text,
                    condition_concept_id bigint
                ) ON COMMIT PRESERVE ROWS
                """
            )
            cur.executemany(
                "INSERT INTO tmp_task20_cox_concept VALUES (%s,%s)",
                [
                    (str(row.phenotype), int(row.condition_concept_id))
                    for row in concept_map.itertuples(index=False)
                ],
            )
            cur.execute("CREATE INDEX ON tmp_task20_cox_concept(condition_concept_id)")
            statement = sql.SQL(
                """
                SELECT
                    g.phenotype,
                    c.person_id::bigint,
                    MIN(c.condition_start_date)::date AS first_phenotype_date
                FROM {}.condition_occurrence c
                JOIN tmp_task20_cox_patient p USING(person_id)
                JOIN tmp_task20_cox_concept g
                  ON g.condition_concept_id = c.condition_concept_id
                WHERE c.condition_start_date IS NOT NULL
                  AND c.condition_start_date <= %s::date
                GROUP BY g.phenotype, c.person_id
                ORDER BY g.phenotype, c.person_id
                """
            ).format(sql.Identifier(args.schema))
            cur.execute(statement, (args.db_end_date,))
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()
        conn.commit()
    frame = pd.DataFrame(rows, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    log(
        f"[DONE] survival cache rows={len(frame):,} "
        f"path={output_path} {time.time() - started:,.1f}s"
    )
    return frame


def load_survival_dates(args, labels: pd.DataFrame, phenotypes: list[str]):
    if args.survival_cache is None:
        cache = args.output_dir / f"first_phenotype_dates_{safe_date(args.index_date)}.parquet"
    else:
        cache = args.survival_cache.expanduser().resolve()
    if cache.exists():
        frame = pd.read_parquet(cache)
        log(f"survival_cache={cache} rows={len(frame):,}")
        return frame, str(cache)
    return build_survival_cache(args, labels, phenotypes, cache), str(cache)


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
        raise ValueError(f"No marker config for phenotypes: {unsupported}")
    missing = sorted(requested - set(available_phenotypes))
    if missing:
        raise ValueError(f"Unknown phenotypes requested: {missing}")
    phenotypes = [p for p in available_phenotypes if p in requested]

    label_cols = [
        "person_id",
        "split",
        "index_date",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
    ]
    for phenotype in phenotypes:
        label_cols.append(f"prior__{phenotype}")
    labels = pd.read_parquet(label_path, columns=label_cols)
    labels["index_date"] = pd.to_datetime(labels["index_date"], errors="coerce")
    labels["first_activity_date"] = pd.to_datetime(labels["first_activity_date"], errors="coerce")
    labels["last_activity_date"] = pd.to_datetime(labels["last_activity_date"], errors="coerce")

    survival_dates, survival_cache = load_survival_dates(args, labels, phenotypes)
    survival_dates["first_phenotype_date"] = pd.to_datetime(
        survival_dates["first_phenotype_date"],
        errors="coerce",
    )

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
        "survival_dates": survival_dates,
        "phenotypes": phenotypes,
        "embedding_columns": emb_cols,
        "lab_columns": set(lab_cols),
        "paths": {
            "label_path": str(label_path),
            "feature_path": str(feature_path),
            "embedding_path": str(embedding_path),
            "lab_marker_path": str(lab_marker_path),
            "survival_cache": survival_cache,
        },
    }


def model_columns(model_set: str, emb_cols: list[str], marker_cols: list[str]):
    if model_set == "cox_baseline":
        return list(COUNT_NUMERIC), list(COUNT_CATEGORICAL), []
    if model_set == "cox_clinical_baseline":
        return list(COUNT_NUMERIC) + list(marker_cols), list(COUNT_CATEGORICAL), list(marker_cols)
    if model_set == "cox_fermat_embedding":
        return list(emb_cols), [], []
    if model_set == "cox_fermat_baseline":
        return list(emb_cols) + list(COUNT_NUMERIC), list(COUNT_CATEGORICAL), []
    if model_set == "cox_fermat_clinical_baseline":
        return list(emb_cols) + list(COUNT_NUMERIC) + list(marker_cols), list(COUNT_CATEGORICAL), list(marker_cols)
    raise ValueError(model_set)


def build_survival_task(data: pd.DataFrame, survival_dates: pd.DataFrame, phenotype: str, args):
    index_date = pd.Timestamp(args.index_date)
    horizon_end = index_date + pd.DateOffset(years={"1y": 1, "3y": 3, "5y": 5}[args.horizon])
    db_end = pd.Timestamp(args.db_end_date)
    censor_limit = min(horizon_end, db_end)
    dates = survival_dates.loc[
        survival_dates["phenotype"].eq(phenotype),
        ["person_id", "first_phenotype_date"],
    ]
    task = data.merge(dates, on="person_id", how="left")
    prior_col = f"prior__{phenotype}"
    prior = task[prior_col].astype(bool)
    at_risk = task["has_pre_index_washout"].astype(bool) & ~prior
    task = task.loc[at_risk].copy()
    first_date = task["first_phenotype_date"]
    event_mask = first_date.notna() & (first_date >= index_date) & (first_date <= censor_limit)
    censor_date = task["last_activity_date"].where(task["last_activity_date"].notna(), censor_limit)
    censor_date = censor_date.clip(upper=censor_limit)
    endpoint = censor_date.where(~event_mask, first_date)
    task["duration_days"] = (endpoint - index_date).dt.days.astype("float32")
    task["event"] = event_mask.astype("int8")
    task = task.loc[task["duration_days"] > 0].copy()
    return task


def deterministic_cap(frame: pd.DataFrame, max_rows: int, seed: int):
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_values("person_id")


def fit_transformer(train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], missing_indicator_cols: list[str]):
    transformer = {
        "numeric": list(numeric_cols),
        "categorical": {},
        "missing_indicator": list(missing_indicator_cols),
        "median": {},
        "mean": {},
        "std": {},
    }
    feature_arrays = []
    feature_names = []
    for column in numeric_cols:
        values = pd.to_numeric(train[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        imputed = values.fillna(median).astype(np.float64)
        mean = float(imputed.mean())
        std = float(imputed.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        transformer["median"][column] = median
        transformer["mean"][column] = mean
        transformer["std"][column] = std
        feature_arrays.append(((imputed - mean) / std).to_numpy(dtype=np.float32))
        feature_names.append(column)
        if column in missing_indicator_cols:
            indicator = values.isna().astype(np.float32).to_numpy()
            feature_arrays.append(indicator)
            feature_names.append(f"{column}__missing")
    for column in categorical_cols:
        values = train[column].astype("string").fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        transformer["categorical"][column] = categories
        for category in categories:
            feature_arrays.append(values.eq(category).astype(np.float32).to_numpy())
            feature_names.append(f"{column}={category}")
    transformer["feature_names"] = feature_names
    return transformer


def transform_features(frame: pd.DataFrame, transformer: dict):
    arrays = []
    for column in transformer["numeric"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        imputed = values.fillna(transformer["median"][column]).astype(np.float64)
        scaled = (imputed - transformer["mean"][column]) / transformer["std"][column]
        arrays.append(scaled.to_numpy(dtype=np.float32))
        if column in transformer["missing_indicator"]:
            arrays.append(values.isna().astype(np.float32).to_numpy())
    for column, categories in transformer["categorical"].items():
        values = frame[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(categories), "__MISSING__")
        for category in categories:
            arrays.append(values.eq(category).astype(np.float32).to_numpy())
    if not arrays:
        raise ValueError("No features selected")
    return np.column_stack(arrays).astype(np.float32), list(transformer["feature_names"])


def cox_partial_loss(linear_predictor, durations, events, ridge, beta):
    order = torch.argsort(durations, descending=True)
    lp = linear_predictor[order]
    ev = events[order]
    log_risk = torch.logcumsumexp(lp, dim=0)
    event_count = torch.clamp(ev.sum(), min=1.0)
    neg_partial = -torch.sum((lp - log_risk) * ev) / event_count
    penalty = ridge * torch.mean(beta * beta)
    return neg_partial + penalty


def fit_cox_model(train_x, train_duration, train_event, val_x, val_duration, val_event, args):
    require_torch()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    d_train = torch.as_tensor(train_duration, dtype=torch.float32, device=device)
    e_train = torch.as_tensor(train_event, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    d_val = torch.as_tensor(val_duration, dtype=torch.float32, device=device)
    e_val = torch.as_tensor(val_event, dtype=torch.float32, device=device)
    beta = torch.zeros(x_train.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([beta], lr=args.learning_rate, weight_decay=0.0)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_lp = x_train @ beta
        loss = cox_partial_loss(train_lp, d_train, e_train, args.ridge, beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = cox_partial_loss(x_val @ beta, d_val, e_val, args.ridge, beta)
        train_value = float(loss.detach().cpu())
        val_value = float(val_loss.detach().cpu())
        history.append({"epoch": epoch, "train_loss": train_value, "val_loss": val_value})
        if val_value < best_val - args.min_delta:
            best_val = val_value
            best_epoch = epoch
            best_state = beta.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        best_state = beta.detach().cpu().clone()
    return best_state.numpy().astype(np.float32), best_epoch, best_val, history


class Fenwick:
    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int):
        index += 1
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def sum(self, index: int):
        index += 1
        total = 0
        while index > 0:
            total += int(self.tree[index])
            index -= index & -index
        return total


def concordance_index(duration, event, score):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(duration) & np.isfinite(score)
    duration = duration[valid]
    event = event[valid]
    score = score[valid]
    if len(duration) == 0 or event.sum() == 0:
        return np.nan
    unique_scores = np.unique(score)
    ranks = np.searchsorted(unique_scores, score)
    order = np.argsort(-duration, kind="mergesort")
    duration = duration[order]
    event = event[order]
    ranks = ranks[order]
    bit = Fenwick(len(unique_scores))
    comparable = 0
    concordant = 0.0
    inserted = 0
    i = 0
    n = len(duration)
    while i < n:
        j = i + 1
        while j < n and duration[j] == duration[i]:
            j += 1
        for k in range(i, j):
            if event[k]:
                lower = bit.sum(ranks[k] - 1) if ranks[k] > 0 else 0
                equal = bit.sum(ranks[k]) - lower
                concordant += lower + 0.5 * equal
                comparable += inserted
        for k in range(i, j):
            bit.add(int(ranks[k]), 1)
            inserted += 1
        i = j
    if comparable == 0:
        return np.nan
    return float(concordant / comparable)


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


def breslow_baseline_survival(train_duration, train_event, train_score, horizon_day):
    duration = np.asarray(train_duration, dtype=np.float64)
    event = np.asarray(train_event, dtype=np.int8)
    score = np.asarray(train_score, dtype=np.float64)
    exp_score = np.exp(np.clip(score, -50, 50))
    event_times = np.unique(duration[(event == 1) & (duration <= horizon_day)])
    cumulative = 0.0
    for event_time in event_times:
        events_at_time = np.sum((event == 1) & (duration == event_time))
        risk_sum = exp_score[duration >= event_time].sum()
        if risk_sum > 0:
            cumulative += events_at_time / risk_sum
    return float(np.exp(-cumulative))


def km_event_probability(duration, event, horizon_day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    order = np.argsort(duration)
    duration = duration[order]
    event = event[order]
    survival = 1.0
    n = len(duration)
    i = 0
    while i < n:
        t = duration[i]
        if t > horizon_day:
            break
        j = i + 1
        while j < n and duration[j] == t:
            j += 1
        at_risk = n - i
        d = int(event[i:j].sum())
        if at_risk > 0 and d > 0:
            survival *= 1.0 - d / at_risk
        i = j
    return float(1.0 - survival)


def horizon_auc(duration, event, score, horizon_day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    cases = (event == 1) & (duration <= horizon_day)
    controls = (event == 0) & (duration >= horizon_day)
    keep = cases | controls
    if keep.sum() == 0:
        return np.nan, 0, 0
    y = cases[keep].astype(np.int8)
    return auroc_score(y, score[keep]), int(cases.sum()), int(controls.sum())


def calibration_bins(duration, event, risk, horizon_day, bins=10):
    frame = pd.DataFrame({
        "duration": np.asarray(duration, dtype=np.float64),
        "event": np.asarray(event, dtype=np.int8),
        "risk": np.asarray(risk, dtype=np.float64),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame()
    unique_risk = frame["risk"].nunique()
    q = min(bins, int(unique_risk))
    if q <= 1:
        frame["bin"] = 0
    else:
        frame["bin"] = pd.qcut(frame["risk"], q=q, labels=False, duplicates="drop")
    rows = []
    for bin_id, sub in frame.groupby("bin", sort=True):
        observed = km_event_probability(sub["duration"].to_numpy(), sub["event"].to_numpy(), horizon_day)
        predicted = float(sub["risk"].mean())
        rows.append({
            "risk_bin": int(bin_id),
            "n": int(len(sub)),
            "events": int(sub["event"].sum()),
            "predicted_risk": predicted,
            "observed_risk": observed,
            "calibration_error": observed - predicted,
        })
    return pd.DataFrame(rows)


def evaluate_predictions(duration, event, score, risk, horizon_day):
    c_index = concordance_index(duration, event, score)
    # Five-year AUROC is a ranking metric. For a Cox model, the linear
    # predictor is monotone with any valid fixed-horizon risk transform, and is
    # more stable when the Breslow baseline-risk transform degenerates.
    auc, cases, controls = horizon_auc(duration, event, score, horizon_day)
    return {
        "n": int(len(duration)),
        "events": int(np.asarray(event).sum()),
        "censored": int(len(duration) - np.asarray(event).sum()),
        "c_index": c_index,
        "horizon_auc": auc,
        "horizon_cases": cases,
        "horizon_controls": controls,
        "mean_predicted_risk": float(np.nanmean(risk)) if len(risk) else np.nan,
    }


def bootstrap_delta_ci(predictions: pd.DataFrame, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    metric_names = ["c_index", "horizon_auc"]
    for phenotype, sub in predictions.groupby("phenotype", sort=False):
        wide_score = sub.pivot(index="person_id", columns="model_set", values="risk_score")
        base = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide_score.index]
        )
        duration = base["duration_days"].to_numpy(dtype=np.float64)
        event = base["event"].to_numpy(dtype=np.int8)
        horizon_day = int(base["horizon_days"].iloc[0])
        n = len(base)
        for baseline in ["cox_baseline", "cox_clinical_baseline"]:
            if baseline not in wide_score.columns:
                continue
            for model_set in args.model_sets:
                if model_set == baseline or model_set not in wide_score.columns:
                    continue
                for metric in metric_names:
                    base_values = wide_score[baseline].to_numpy()
                    model_values = wide_score[model_set].to_numpy()
                    if metric == "c_index":
                        observed_base = concordance_index(duration, event, base_values)
                        observed_model = concordance_index(duration, event, model_values)
                    else:
                        observed_base = horizon_auc(duration, event, base_values, horizon_day)[0]
                        observed_model = horizon_auc(duration, event, model_values, horizon_day)[0]
                    boot = np.empty(args.bootstrap_samples, dtype=np.float64)
                    for sample in range(args.bootstrap_samples):
                        index = rng.integers(0, n, size=n)
                        if metric == "c_index":
                            b0 = concordance_index(duration[index], event[index], base_values[index])
                            b1 = concordance_index(duration[index], event[index], model_values[index])
                        else:
                            b0 = horizon_auc(duration[index], event[index], base_values[index], horizon_day)[0]
                            b1 = horizon_auc(duration[index], event[index], model_values[index], horizon_day)[0]
                        boot[sample] = b1 - b0
                    rows.append({
                        "phenotype": phenotype,
                        "model_set": model_set,
                        "baseline_model_set": baseline,
                        "metric": metric,
                        "n": int(n),
                        "events": int(event.sum()),
                        "baseline": observed_base,
                        "model": observed_model,
                        "delta": observed_model - observed_base,
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
    require_torch()

    started_all = time.time()
    loaded = load_inputs(args)
    data = loaded["data"]
    survival_dates = loaded["survival_dates"]
    phenotypes = loaded["phenotypes"]
    emb_cols = loaded["embedding_columns"]
    lab_columns = loaded["lab_columns"]
    horizon_day = horizon_days(args.index_date, args.horizon)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"embedding_features={len(emb_cols):,}")
    log(f"horizon={args.horizon} horizon_days={horizon_day}")
    log(f"torch_version={torch.__version__} device={device_name}")

    metric_rows = []
    prediction_frames = []
    calibration_frames = []
    history_frames = []
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
        task_all = build_survival_task(data, survival_dates, phenotype, args)
        if task_all.empty:
            skip_rows.append({"phenotype": phenotype, "model_set": "", "reason": "no_at_risk_rows"})
            log(f"[SKIP] {phenotype}: no at-risk rows")
            continue
        for model_set in args.model_sets:
            started = time.time()
            train = task_all.loc[task_all["split"].eq("train")].copy()
            val = task_all.loc[task_all["split"].eq("val")].copy()
            test = task_all.loc[task_all["split"].eq("test")].copy()
            train = deterministic_cap(train, args.max_train_rows, args.bootstrap_seed)
            if args.max_eval_rows > 0:
                val = deterministic_cap(val, args.max_eval_rows, args.bootstrap_seed + 1)
                test = deterministic_cap(test, args.max_eval_rows, args.bootstrap_seed + 2)
            train_events = int(train["event"].sum())
            test_events = int(test["event"].sum())
            if train_events < args.min_train_events or test_events < args.min_test_events:
                skip_rows.append({
                    "phenotype": phenotype,
                    "model_set": model_set,
                    "reason": "too_few_events",
                    "train_events": train_events,
                    "test_events": test_events,
                })
                log(
                    f"[SKIP] {phenotype} {model_set}: "
                    f"train_events={train_events} test_events={test_events}"
                )
                continue
            numeric_cols, categorical_cols, missing_cols = model_columns(model_set, emb_cols, marker_cols)
            transformer = fit_transformer(train, numeric_cols, categorical_cols, missing_cols)
            train_x, feature_names = transform_features(train, transformer)
            val_x, _ = transform_features(val, transformer)
            test_x, _ = transform_features(test, transformer)
            beta, best_epoch, best_val_loss, history = fit_cox_model(
                train_x,
                train["duration_days"].to_numpy(dtype=np.float32),
                train["event"].to_numpy(dtype=np.int8),
                val_x,
                val["duration_days"].to_numpy(dtype=np.float32),
                val["event"].to_numpy(dtype=np.int8),
                args,
            )
            train_score = train_x @ beta
            test_score = test_x @ beta
            s0 = breslow_baseline_survival(
                train["duration_days"].to_numpy(dtype=np.float32),
                train["event"].to_numpy(dtype=np.int8),
                train_score,
                horizon_day,
            )
            test_risk = 1.0 - np.power(s0, np.exp(np.clip(test_score, -50, 50)))
            row = {
                "index_date": args.index_date,
                "horizon": args.horizon,
                "phenotype": phenotype,
                "model_set": model_set,
                "fit_seconds": time.time() - started,
                "train_rows": int(len(train)),
                "train_events": train_events,
                "val_rows": int(len(val)),
                "val_events": int(val["event"].sum()),
                "test_rows": int(len(test)),
                "test_events": test_events,
                "n_model_features": int(len(feature_names)),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "baseline_survival_5y": float(s0),
            }
            row.update(
                {
                    f"test_{key}": value
                    for key, value in evaluate_predictions(
                        test["duration_days"].to_numpy(dtype=np.float32),
                        test["event"].to_numpy(dtype=np.int8),
                        test_score,
                        test_risk,
                        horizon_day,
                    ).items()
                }
            )
            metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "index_date": args.index_date,
                "horizon": args.horizon,
                "horizon_days": horizon_day,
                "phenotype": phenotype,
                "model_set": model_set,
                "person_id": test["person_id"].to_numpy(),
                "split": "test",
                "duration_days": test["duration_days"].to_numpy(dtype=np.float32),
                "event": test["event"].to_numpy(dtype=np.int8),
                "risk_score": test_score.astype(np.float32),
                "risk_5y": test_risk.astype(np.float32),
                "sequence_length_pre_index": test["sequence_length_pre_index"].to_numpy(),
            }))
            calibration = calibration_bins(
                test["duration_days"].to_numpy(dtype=np.float32),
                test["event"].to_numpy(dtype=np.int8),
                test_risk,
                horizon_day,
            )
            if not calibration.empty:
                calibration.insert(0, "model_set", model_set)
                calibration.insert(0, "phenotype", phenotype)
                calibration.insert(0, "horizon", args.horizon)
                calibration.insert(0, "index_date", args.index_date)
                calibration_frames.append(calibration)
            hist = pd.DataFrame(history)
            hist.insert(0, "model_set", model_set)
            hist.insert(0, "phenotype", phenotype)
            history_frames.append(hist)
            log(
                "[DONE] "
                f"{phenotype} {model_set} "
                f"c_index={row['test_c_index']:.4f} "
                f"auc5y={row['test_horizon_auc']:.4f} "
                f"epoch={best_epoch} "
                f"{row['fit_seconds']:.1f}s"
            )

    if not prediction_frames:
        raise RuntimeError("No Cox predictions were generated")
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ci = bootstrap_delta_ci(predictions, args) if args.bootstrap_samples > 0 else pd.DataFrame()
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    metrics_path = args.output_dir / "cox_survival_metrics.csv"
    predictions_path = args.output_dir / "cox_survival_test_predictions.parquet"
    ci_path = args.output_dir / "cox_survival_bootstrap_delta_ci.csv"
    calibration_path = args.output_dir / "cox_survival_calibration_bins.csv"
    history_path = args.output_dir / "cox_survival_training_history.csv"
    marker_config_path = args.output_dir / "cox_survival_marker_config.csv"
    skip_path = args.output_dir / "cox_survival_skips.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    ci.to_csv(ci_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    history.to_csv(history_path, index=False)
    pd.DataFrame(marker_config_rows).to_csv(marker_config_path, index=False)
    pd.DataFrame(skip_rows).to_csv(skip_path, index=False)

    manifest = {
        **loaded["paths"],
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "horizon_days": horizon_day,
        "db_end_date": args.db_end_date,
        "phenotypes": phenotypes,
        "model_sets": args.model_sets,
        "ridge": args.ridge,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "max_train_rows": args.max_train_rows,
        "max_eval_rows": args.max_eval_rows,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "torch_version": torch.__version__,
        "device": device_name,
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
            "bootstrap_delta_ci": str(ci_path),
            "calibration_bins": str(calibration_path),
            "training_history": str(history_path),
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
