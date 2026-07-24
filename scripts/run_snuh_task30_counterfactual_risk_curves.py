#!/usr/bin/env python3
"""Measure how token edits change FERMAT + Cox risk curves.

The Cox model is fit once for one phenotype.  For each test patient, this
script then keeps the patient fixed, edits one visible pre-index token history,
recomputes the FERMAT embedding, and compares the original and edited risks.

Supported edits are:

* add: insert a token N days before the index date;
* delete: remove the first, last, or every visible occurrence;
* move: move the first or last visible occurrence to N days before index.

This is a model-response analysis.  It does not claim that the edit is a
causal treatment effect.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    torch = None

from scripts.extract_snuh_task19_fermat_embeddings import (
    autocast_context,
    collate_rows,
    extract_hidden,
)
from scripts.run_snuh_task20_cox_survival_models import (
    build_survival_task,
    concordance_index,
    fit_cox_model,
    fit_transformer,
    horizon_auc,
    horizon_days,
    load_survival_dates,
    transform_features,
)
POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "cox_survival_2018_5y_block2048_1000ci_20260703"
    / "first_phenotype_dates_20180101.parquet"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "counterfactual_risk_curves"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
DEFAULT_CURVE_DAYS = sorted(
    {1, 7, 365, 730, 1095, 1460, 1826, *range(30, 1827, 30)}
)
DEFAULT_MAX_ATTENTION_CELLS = 4 * 2048 * 2048


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phenotype", required=True)
    parser.add_argument(
        "--interventions-file",
        type=Path,
        required=True,
        help=(
            "CSV columns: intervention_id, operation, token_key, occurrence, "
            "days_before_index, require_absent"
        ),
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--horizon", default="5y", choices=["1y", "3y", "5y"])
    parser.add_argument("--curve-days", nargs="+", type=int, default=DEFAULT_CURVE_DAYS)
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-attention-cells",
        type=int,
        default=DEFAULT_MAX_ATTENTION_CELLS,
        help=(
            "Automatic GPU-memory guard. A padded batch is split until "
            "batch_patients * max_sequence_length^2 is no larger than this value."
        ),
    )
    parser.add_argument("--write-every", type=int, default=5000)
    parser.add_argument(
        "--max-patients-per-intervention",
        "--max-candidate-patients",
        dest="max_patients_per_intervention",
        type=int,
        default=0,
        help=(
            "Stop after this many patients actually receive each token edit. "
            "0 uses every eligible at-risk test patient."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--embedding-check-patients", type=int, default=128)
    parser.add_argument("--embedding-check-atol", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-test-events", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def safe_date(value: str):
    return value.replace("-", "")


def safe_name(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"Unsafe empty file name derived from {value!r}")
    return cleaned


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(parents=True, exist_ok=True)


def require_torch():
    if torch is None:
        raise RuntimeError("torch is required; run this script in the FERMAT Pod")


def parquet_columns(path: Path):
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except ModuleNotFoundError:
        return pd.read_parquet(path).columns.tolist()


def load_model(path: Path, device: str):
    from model import Fermat, FermatConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Fermat(FermatConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_prefixes = (
        "transformer.",
        "lm_head.",
        "time_head.",
        "same_day_head.",
        "log_rate",
    )
    critical_missing = [key for key in missing if key.startswith(critical_prefixes)]
    critical_unexpected = [key for key in unexpected if key.startswith(critical_prefixes)]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Critical checkpoint/model mismatch:\n"
            + json.dumps(
                {
                    "ckpt": str(path),
                    "critical_missing": critical_missing,
                    "critical_unexpected": critical_unexpected,
                    "all_missing": list(missing),
                    "all_unexpected": list(unexpected),
                },
                indent=2,
            )
        )
    checkpoint_metadata = {
        "iter": int(checkpoint.get("iter", -1)),
        "model_args": dict(checkpoint["model_args"]),
    }
    del checkpoint
    model.to(device)
    model.eval()
    return model, checkpoint_metadata


def load_split_data(data_dir: Path, split: str):
    from utils import get_p2i, load_data

    path = data_dir / f"{split}.bin"
    data, has_types = load_data(path)
    if not has_types:
        raise ValueError(f"{path} is not a 4-column typed FERMAT bin")
    p2i = get_p2i(data)
    patient_ids = data[p2i[:, 0].astype(np.int64), 0].astype(np.int64)
    index = {
        int(patient_id): (int(start), int(length))
        for patient_id, (start, length) in zip(patient_ids, p2i)
    }
    return {"path": str(path), "data": data, "index": index}


def load_registry(data_dir: Path):
    for filename in ["token_registry.csv", "vocab.csv"]:
        path = data_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle)), str(path)
    raise FileNotFoundError("Expected token_registry.csv or vocab.csv")


def registry_type(row):
    value = row.get("token_type_id", row.get("token_type"))
    try:
        return int(value)
    except (TypeError, ValueError):
        from model import TokenType

        return int(TokenType[row["token_type"]])


def parse_bool(value, default=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_resolved_interventions(path: Path, registry_rows):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str).replace({np.nan: ""})
    required = {"intervention_id", "operation", "token_key"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} has no intervention rows")
    if frame["intervention_id"].duplicated().any():
        values = frame.loc[frame["intervention_id"].duplicated(), "intervention_id"].tolist()
        raise ValueError(f"Duplicate intervention_id values: {values}")

    registry = {str(row.get("token_key")): row for row in registry_rows}
    rows = []
    for raw in frame.to_dict("records"):
        intervention_id = str(raw["intervention_id"]).strip()
        operation = str(raw["operation"]).strip().lower()
        token_key = str(raw["token_key"]).strip()
        if operation not in {"add", "delete", "move"}:
            raise ValueError(f"{intervention_id}: operation must be add, delete, or move")
        if token_key not in registry:
            raise ValueError(f"{intervention_id}: token_key not found in registry: {token_key}")
        occurrence = str(raw.get("occurrence", "last")).strip().lower() or "last"
        allowed_occurrences = {"first", "last", "all"} if operation == "delete" else {"first", "last"}
        if occurrence not in allowed_occurrences:
            raise ValueError(
                f"{intervention_id}: occurrence={occurrence!r} is invalid for {operation}"
            )
        days_text = str(raw.get("days_before_index", "")).strip()
        days_before_index = None
        if operation in {"add", "move"}:
            if not days_text:
                raise ValueError(f"{intervention_id}: days_before_index is required for {operation}")
            days_before_index = int(days_text)
            if days_before_index < 1:
                raise ValueError(f"{intervention_id}: days_before_index must be at least 1")
        registry_row = registry[token_key]
        rows.append(
            {
                "intervention_id": intervention_id,
                "operation": operation,
                "token_key": token_key,
                "raw_token_id": int(registry_row["token_id"]),
                "model_token_id": int(registry_row["token_id"]) + 1,
                "token_type_id": int(registry_type(registry_row)),
                "occurrence": occurrence,
                "days_before_index": days_before_index,
                "require_absent": parse_bool(raw.get("require_absent", ""), default=False),
            }
        )
    return pd.DataFrame(rows)


def load_task_data(args):
    label_path = args.label_dir / (
        f"patient_phenotype_labels_wide_{safe_date(args.index_date)}.parquet"
    )
    for path in [label_path, args.embedding_file]:
        if not path.exists():
            raise FileNotFoundError(path)
    prior_col = f"prior__{args.phenotype}"
    label_cols = [
        "person_id",
        "split",
        "index_date",
        "age_at_index",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
        prior_col,
    ]
    available_label_cols = set(parquet_columns(label_path))
    missing = sorted(set(label_cols) - available_label_cols)
    if missing:
        raise ValueError(f"{label_path} is missing columns: {missing}")
    labels = pd.read_parquet(label_path, columns=label_cols)
    labels["index_date"] = pd.to_datetime(labels["index_date"], errors="coerce")
    labels["first_activity_date"] = pd.to_datetime(labels["first_activity_date"], errors="coerce")
    labels["last_activity_date"] = pd.to_datetime(labels["last_activity_date"], errors="coerce")
    labels["index_age_days"] = np.floor(labels["age_at_index"] * 365.25).astype(np.int64)

    embedding_cols_all = parquet_columns(args.embedding_file)
    emb_cols = [column for column in embedding_cols_all if column.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"No emb_* columns in {args.embedding_file}")
    embedding_meta = [
        "person_id",
        "split",
        "patient_id_dense",
        "has_embedding_sequence",
        "sequence_length_pre_index",
    ]
    embeddings = pd.read_parquet(args.embedding_file, columns=embedding_meta + emb_cols)
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()
    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    survival_dates, survival_cache = load_survival_dates(args, labels, [args.phenotype])
    survival_dates["first_phenotype_date"] = pd.to_datetime(
        survival_dates["first_phenotype_date"], errors="coerce"
    )
    task = build_survival_task(data, survival_dates, args.phenotype, args)
    return task, emb_cols, {
        "label_path": str(label_path),
        "embedding_file": str(args.embedding_file),
        "survival_cache": survival_cache,
    }


def deterministic_cap(frame: pd.DataFrame, max_rows: int, seed: int):
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.sort_values("person_id")
    return frame.sample(n=max_rows, random_state=seed).sort_values("person_id")


def breslow_baseline_curve(duration, event, score, max_day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(duration) & np.isfinite(score) & (duration > 0)
    duration = duration[valid]
    event = event[valid]
    score = score[valid]
    if event.sum() == 0:
        raise ValueError("Cannot estimate a baseline hazard without training events")
    order = np.argsort(duration, kind="mergesort")
    duration_sorted = duration[order]
    exp_score_sorted = np.exp(np.clip(score[order], -50, 50))
    suffix_risk = np.cumsum(exp_score_sorted[::-1], dtype=np.float64)[::-1]
    event_times, event_counts = np.unique(
        duration[(event == 1) & (duration <= max_day)], return_counts=True
    )
    first_at_risk = np.searchsorted(duration_sorted, event_times, side="left")
    risk_sums = suffix_risk[first_at_risk]
    increments = np.divide(
        event_counts.astype(np.float64),
        risk_sums,
        out=np.zeros_like(risk_sums, dtype=np.float64),
        where=risk_sums > 0,
    )
    return event_times.astype(np.float64), np.cumsum(increments)


def cumulative_hazard_at_days(event_times, cumulative_hazard, days):
    days = np.asarray(days, dtype=np.float64)
    index = np.searchsorted(event_times, days, side="right") - 1
    output = np.zeros(len(days), dtype=np.float64)
    valid = index >= 0
    output[valid] = cumulative_hazard[index[valid]]
    return output


def risk_matrix(scores, baseline_hazard):
    scores = np.asarray(scores, dtype=np.float64)
    baseline_hazard = np.asarray(baseline_hazard, dtype=np.float64)
    relative_hazard = np.exp(np.clip(scores, -50, 50))[:, None]
    return -np.expm1(-relative_hazard * baseline_hazard[None, :])


def visible_rows(split_data, dense_id: int, index_age_days: int, block_size: int):
    location = split_data["index"].get(int(dense_id))
    if location is None:
        return None
    start, length = location
    rows = split_data["data"][start : start + length]
    rows = rows[rows[:, 1].astype(np.int64) < int(index_age_days)]
    if len(rows) > block_size:
        rows = rows[-block_size:]
    return np.asarray(rows).copy()


def apply_intervention(rows, dense_id: int, index_age_days: int, intervention, block_size: int):
    if rows is None:
        return None, {"status": "missing_patient_sequence"}
    if len(rows) == 0:
        return None, {"status": "empty_pre_index_sequence"}
    token_id = int(intervention["raw_token_id"])
    matches = np.flatnonzero(rows[:, 2].astype(np.int64) == token_id)
    operation = intervention["operation"]
    changed = rows.copy()
    edited_occurrences = 0
    oldest_token_dropped = False
    original_age = None
    new_age = None
    original_days_before_index = None
    new_days_before_index = None

    if operation == "add":
        if bool(intervention["require_absent"]) and len(matches):
            return None, {"status": "token_already_present"}
        new_age = int(index_age_days) - int(intervention["days_before_index"])
        new_days_before_index = int(intervention["days_before_index"])
        if new_age < 0 or new_age >= int(index_age_days):
            return None, {"status": "new_age_outside_pre_index_history"}
        new_row = np.asarray(
            [[dense_id, new_age, token_id, int(intervention["token_type_id"])]],
            dtype=changed.dtype,
        )
        changed = np.concatenate([changed, new_row], axis=0)
        changed = changed[np.argsort(changed[:, 1], kind="stable")]
        if len(changed) > block_size:
            changed = changed[-block_size:]
            oldest_token_dropped = True
        edited_occurrences = 1
    elif operation == "delete":
        if not len(matches):
            return None, {"status": "token_not_present"}
        occurrence = intervention["occurrence"]
        selected = matches if occurrence == "all" else np.asarray(
            [matches[0] if occurrence == "first" else matches[-1]]
        )
        original_age = ",".join(str(int(x)) for x in rows[selected, 1])
        original_days_before_index = ",".join(
            str(int(index_age_days) - int(x)) for x in rows[selected, 1]
        )
        changed = np.delete(changed, selected, axis=0)
        if len(changed) == 0:
            return None, {"status": "empty_sequence_after_delete"}
        edited_occurrences = int(len(selected))
    elif operation == "move":
        if not len(matches):
            return None, {"status": "token_not_present"}
        selected = int(matches[0] if intervention["occurrence"] == "first" else matches[-1])
        original_age = str(int(changed[selected, 1]))
        original_days_before_index = int(index_age_days) - int(changed[selected, 1])
        new_age = int(index_age_days) - int(intervention["days_before_index"])
        new_days_before_index = int(intervention["days_before_index"])
        if new_age < 0 or new_age >= int(index_age_days):
            return None, {"status": "new_age_outside_pre_index_history"}
        changed[selected, 1] = new_age
        changed = changed[np.argsort(changed[:, 1], kind="stable")]
        edited_occurrences = 1
    else:  # pragma: no cover - validated while loading
        raise ValueError(operation)

    if np.array_equal(rows, changed):
        return None, {"status": "model_input_unchanged"}
    return changed, {
        "status": "included",
        "edited_occurrences": edited_occurrences,
        "token_occurrences_before": int(len(matches)),
        "token_occurrences_after": int(
            np.sum(changed[:, 2].astype(np.int64) == token_id)
        ),
        "sequence_length_before": int(len(rows)),
        "sequence_length_after": int(len(changed)),
        "oldest_token_dropped": bool(oldest_token_dropped),
        "original_token_age_days": original_age,
        "new_token_age_days": new_age,
        "original_token_days_before_index": original_days_before_index,
        "new_token_days_before_index": new_days_before_index,
    }


def attention_safe_batches(items, batch_size, max_attention_cells):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_attention_cells < 1:
        raise ValueError("max_attention_cells must be at least 1")
    batch = []
    max_length = 0
    for item in items:
        item_length = len(item["rows"])
        proposed_max = max(max_length, item_length)
        proposed_size = len(batch) + 1
        proposed_cells = proposed_size * proposed_max * proposed_max
        if batch and (
            proposed_size > batch_size or proposed_cells > max_attention_cells
        ):
            yield batch
            batch = []
            max_length = 0
        batch.append(item)
        max_length = max(max_length, item_length)
    if batch:
        yield batch


def embed_one_batch(model, batch, device, dtype):
    idx, age, token_type, lengths = collate_rows(batch, device)
    with torch.no_grad(), autocast_context(device, dtype):
        hidden = extract_hidden(model, idx, age, token_type, lengths, "last")
    return hidden.detach().float().cpu().numpy()


def embed_batch_with_oom_retry(model, batch, device, dtype):
    try:
        return embed_one_batch(model, batch, device, dtype)
    except torch.OutOfMemoryError:
        if device != "cuda" or len(batch) <= 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(batch) // 2
        max_length = max(len(item["rows"]) for item in batch)
        log(
            f"[GPU OOM RETRY] patients={len(batch)} max_sequence_length={max_length} "
            f"retry_as={midpoint}+{len(batch) - midpoint}"
        )
        left = embed_batch_with_oom_retry(model, batch[:midpoint], device, dtype)
        right = embed_batch_with_oom_retry(model, batch[midpoint:], device, dtype)
        return np.concatenate([left, right], axis=0)


def embed_row_items(model, items, device, dtype, batch_size, max_attention_cells):
    if not items:
        return np.zeros((0, int(model.config.n_embd)), dtype=np.float32)
    output = np.zeros((len(items), int(model.config.n_embd)), dtype=np.float32)
    start = 0
    for batch in attention_safe_batches(items, batch_size, max_attention_cells):
        hidden = embed_batch_with_oom_retry(model, batch, device, dtype)
        output[start : start + len(batch)] = hidden
        start += len(batch)
    return output


def verify_saved_embeddings(args, model, split_data, test, emb_cols, block_size):
    n = min(int(args.embedding_check_patients), len(test))
    if n <= 0:
        return {"patients": 0, "max_abs_error": np.nan, "mean_abs_error": np.nan}
    probe = test.sample(n=n, random_state=args.random_seed).sort_values("person_id")
    items = []
    keep = []
    for row in probe.itertuples(index=False):
        rows = visible_rows(split_data, int(row.patient_id_dense), int(row.index_age_days), block_size)
        if rows is None or len(rows) == 0:
            continue
        items.append({"rows": rows})
        keep.append(row)
    recomputed = embed_row_items(
        model,
        items,
        args.device,
        args.dtype,
        args.batch_size,
        args.max_attention_cells,
    )
    saved = np.asarray([[float(getattr(row, col)) for col in emb_cols] for row in keep])
    absolute = np.abs(recomputed - saved)
    result = {
        "patients": int(len(keep)),
        "max_abs_error": float(absolute.max()) if absolute.size else np.nan,
        "mean_abs_error": float(absolute.mean()) if absolute.size else np.nan,
    }
    if absolute.size and result["max_abs_error"] > args.embedding_check_atol:
        raise RuntimeError(
            "Recomputed original embeddings do not match the saved embedding file: "
            + json.dumps(result)
            + f"; allowed max_abs_error={args.embedding_check_atol}. "
            "Check that --ckpt, --data-dir, block_size, pooling, and dtype match the "
            "embedding extraction run."
        )
    return result


def score_embeddings(embeddings, emb_cols, transformer, beta):
    frame = pd.DataFrame(embeddings, columns=emb_cols)
    features, names = transform_features(frame, transformer)
    if names != transformer["feature_names"]:
        raise RuntimeError("Counterfactual feature order differs from Cox training order")
    return features @ beta


def write_intervention_part(
    args,
    intervention,
    part_number,
    buffered,
    model,
    emb_cols,
    transformer,
    beta,
    curve_days,
    baseline_hazard,
    output_dir,
):
    original_items = [{"rows": item["original_rows"]} for item in buffered]
    changed_items = [{"rows": item["changed_rows"]} for item in buffered]
    original_embeddings = embed_row_items(
        model,
        original_items,
        args.device,
        args.dtype,
        args.batch_size,
        args.max_attention_cells,
    )
    changed_embeddings = embed_row_items(
        model,
        changed_items,
        args.device,
        args.dtype,
        args.batch_size,
        args.max_attention_cells,
    )
    original_score = score_embeddings(original_embeddings, emb_cols, transformer, beta)
    changed_score = score_embeddings(changed_embeddings, emb_cols, transformer, beta)
    score_difference = changed_score - original_score
    hazard_ratio = np.exp(np.clip(score_difference, -50, 50))
    original_risk = risk_matrix(original_score, baseline_hazard)
    changed_risk = risk_matrix(changed_score, baseline_hazard)

    metadata = []
    for item in buffered:
        row = {
            "phenotype": args.phenotype,
            **intervention.to_dict(),
            "person_id": int(item["person_id"]),
            "patient_id_dense": int(item["patient_id_dense"]),
            "index_age_days": int(item["index_age_days"]),
            "duration_days": float(item["duration_days"]),
            "event": int(item["event"]),
            **item["edit_metadata"],
        }
        metadata.append(row)
    pairs = pd.DataFrame(metadata)
    pairs["original_risk_score"] = original_score.astype(np.float32)
    pairs["counterfactual_risk_score"] = changed_score.astype(np.float32)
    pairs["risk_score_difference"] = score_difference.astype(np.float32)
    pairs["counterfactual_hazard_ratio"] = hazard_ratio.astype(np.float32)

    curves = []
    for day_index, day in enumerate(curve_days):
        frame = pairs[["phenotype", "intervention_id", "person_id"]].copy()
        frame["day"] = int(day)
        frame["original_risk"] = original_risk[:, day_index].astype(np.float32)
        frame["counterfactual_risk"] = changed_risk[:, day_index].astype(np.float32)
        frame["risk_difference"] = (
            changed_risk[:, day_index] - original_risk[:, day_index]
        ).astype(np.float32)
        curves.append(frame)
    curves = pd.concat(curves, ignore_index=True)

    part_dir = output_dir / "raw" / safe_name(intervention["intervention_id"])
    part_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = part_dir / f"patient_pairs_part_{part_number:05d}.parquet"
    curves_path = part_dir / f"patient_curves_part_{part_number:05d}.parquet"
    pairs.to_parquet(pairs_path, index=False)
    curves.to_parquet(curves_path, index=False)
    log(
        f"[RAW SAVED] {intervention['intervention_id']} part={part_number} "
        f"patients={len(pairs):,}"
    )
    return pairs_path, curves_path


def summarize_outputs(pair_paths, curve_paths, output_dir):
    pairs = pd.concat([pd.read_parquet(path) for path in pair_paths], ignore_index=True)
    curves = pd.concat([pd.read_parquet(path) for path in curve_paths], ignore_index=True)
    pair_summary = (
        pairs.groupby(["phenotype", "intervention_id", "operation", "token_key"], as_index=False)
        .agg(
            patients=("person_id", "size"),
            events=("event", "sum"),
            mean_risk_score_difference=("risk_score_difference", "mean"),
            median_risk_score_difference=("risk_score_difference", "median"),
            mean_hazard_ratio=("counterfactual_hazard_ratio", "mean"),
            median_hazard_ratio=("counterfactual_hazard_ratio", "median"),
            proportion_score_increased=(
                "risk_score_difference",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
            oldest_token_dropped_patients=("oldest_token_dropped", "sum"),
        )
    )
    curve_summary = (
        curves.groupby(["phenotype", "intervention_id", "day"], as_index=False)
        .agg(
            patients=("person_id", "size"),
            mean_original_risk=("original_risk", "mean"),
            mean_counterfactual_risk=("counterfactual_risk", "mean"),
            mean_risk_difference=("risk_difference", "mean"),
            median_risk_difference=("risk_difference", "median"),
            proportion_risk_increased=(
                "risk_difference",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
    )
    pair_path = output_dir / "counterfactual_pair_summary.csv"
    curve_path = output_dir / "counterfactual_curve_summary.csv"
    pair_summary.to_csv(pair_path, index=False)
    curve_summary.to_csv(curve_path, index=False)
    return pair_summary, curve_summary, pair_path, curve_path


def make_curve_plot(curve_summary, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    interventions = curve_summary["intervention_id"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(interventions), 1, figsize=(8, max(4, 3.5 * len(interventions))), squeeze=False
    )
    for axis, intervention_id in zip(axes[:, 0], interventions):
        sub = curve_summary.loc[curve_summary["intervention_id"].eq(intervention_id)]
        axis.plot(sub["day"], sub["mean_original_risk"], label="original", linewidth=2)
        axis.plot(
            sub["day"], sub["mean_counterfactual_risk"], label="token edited", linewidth=2
        )
        axis.set_title(intervention_id)
        axis.set_xlabel("days after index")
        axis.set_ylabel("mean predicted risk")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    path = output_dir / "counterfactual_mean_risk_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main():
    args = parse_args()
    require_torch()
    for name in [
        "ckpt",
        "data_dir",
        "label_dir",
        "embedding_file",
        "interventions_file",
        "output_dir",
    ]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.survival_cache is not None:
        args.survival_cache = args.survival_cache.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    started = time.time()

    registry_rows, registry_path = load_registry(args.data_dir)
    interventions = load_resolved_interventions(args.interventions_file, registry_rows)
    interventions_path = args.output_dir / "interventions_resolved.csv"
    interventions.to_csv(interventions_path, index=False)
    task, emb_cols, input_paths = load_task_data(args)
    train = deterministic_cap(
        task.loc[task["split"].eq("train")].copy(), args.max_train_rows, args.random_seed
    )
    val = deterministic_cap(
        task.loc[task["split"].eq("val")].copy(), args.max_eval_rows, args.random_seed + 1
    )
    test = deterministic_cap(
        task.loc[task["split"].eq("test")].copy(), args.max_eval_rows, args.random_seed + 2
    )
    if int(train["event"].sum()) < args.min_train_events:
        raise RuntimeError(f"Too few train events: {int(train['event'].sum())}")
    if int(test["event"].sum()) < args.min_test_events:
        raise RuntimeError(f"Too few test events: {int(test['event'].sum())}")
    if val.empty or int(val["event"].sum()) == 0:
        raise RuntimeError("Validation split has no events")

    # Confirm the checkpoint/data/embedding combination before spending time on
    # Cox fitting.  Release the model while Cox is fit so GPU memory is not
    # shared with the full training feature matrix.
    model, checkpoint = load_model(args.ckpt, args.device)
    checkpoint_step = int(checkpoint.get("iter", -1))
    block_size = int(model.config.block_size)
    full_length_batch_cap = max(
        1,
        min(
            args.batch_size,
            args.max_attention_cells // max(block_size * block_size, 1),
        ),
    )
    log(
        f"[GPU MEMORY GUARD] requested_batch_size={args.batch_size} "
        f"block_size={block_size} full_length_batch_cap={full_length_batch_cap} "
        f"max_attention_cells={args.max_attention_cells}"
    )
    if int(model.config.n_embd) != len(emb_cols):
        raise RuntimeError(
            f"Checkpoint n_embd={model.config.n_embd} but embedding file has {len(emb_cols)} columns"
        )
    split_data = load_split_data(args.data_dir, args.split)
    embedding_check = verify_saved_embeddings(
        args, model, split_data, test, emb_cols, block_size
    )
    check_path = args.output_dir / "original_embedding_check.json"
    check_path.write_text(
        json.dumps(embedding_check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(
        f"[EMBEDDING MATCH] patients={embedding_check['patients']} "
        f"max_abs_error={embedding_check['max_abs_error']:.6g}"
    )
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    transformer = fit_transformer(train, emb_cols, [], [])
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
    max_curve_day = horizon_days(args.index_date, args.horizon)
    curve_days = sorted(set(day for day in args.curve_days if 0 < day <= max_curve_day))
    if not curve_days:
        raise ValueError(f"No --curve-days fall within the {max_curve_day}-day horizon")
    event_times, cumulative_hazard = breslow_baseline_curve(
        train["duration_days"], train["event"], train_score, max_curve_day
    )
    baseline_hazard = cumulative_hazard_at_days(event_times, cumulative_hazard, curve_days)

    artifact_path = args.output_dir / "cox_embedding_model.npz"
    np.savez_compressed(
        artifact_path,
        beta=beta,
        event_times=event_times,
        cumulative_baseline_hazard=cumulative_hazard,
        curve_days=np.asarray(curve_days, dtype=np.int32),
        baseline_hazard_at_curve_days=baseline_hazard,
    )
    transformer_path = args.output_dir / "cox_embedding_transformer.json"
    transformer_path.write_text(
        json.dumps(transformer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    history_path = args.output_dir / "cox_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    horizon_day = max_curve_day
    test_metrics = {
        "phenotype": args.phenotype,
        "train_rows": int(len(train)),
        "train_events": int(train["event"].sum()),
        "val_rows": int(len(val)),
        "val_events": int(val["event"].sum()),
        "test_rows": int(len(test)),
        "test_events": int(test["event"].sum()),
        "features": int(len(feature_names)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_c_index": float(concordance_index(test["duration_days"], test["event"], test_score)),
        "test_horizon_auc": float(
            horizon_auc(test["duration_days"], test["event"], test_score, horizon_day)[0]
        ),
    }
    metrics_path = args.output_dir / "cox_test_metrics.json"
    metrics_path.write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(
        f"[COX SAVED] c_index={test_metrics['test_c_index']:.4f} "
        f"auc={test_metrics['test_horizon_auc']:.4f}"
    )

    test = test[
        [
            "person_id",
            "patient_id_dense",
            "index_age_days",
            "duration_days",
            "event",
        ]
    ].copy()
    del task, train, val, train_x, val_x, test_x, train_score, test_score
    if args.device == "cuda":
        torch.cuda.empty_cache()

    model, _ = load_model(args.ckpt, args.device)

    pair_paths = []
    curve_paths = []
    status_rows = []
    for _, intervention in interventions.iterrows():
        if args.max_patients_per_intervention > 0:
            candidates = test.sample(frac=1.0, random_state=args.random_seed + 1000)
        else:
            candidates = test.sort_values("person_id")
        status_counts = Counter()
        buffered = []
        part_number = 0

        def flush():
            nonlocal buffered, part_number
            if not buffered:
                return
            pair_path, curve_path = write_intervention_part(
                args,
                intervention,
                part_number,
                buffered,
                model,
                emb_cols,
                transformer,
                beta,
                curve_days,
                baseline_hazard,
                args.output_dir,
            )
            pair_paths.append(pair_path)
            curve_paths.append(curve_path)
            part_number += 1
            buffered = []

        for row in candidates.itertuples(index=False):
            original_rows = visible_rows(
                split_data,
                int(row.patient_id_dense),
                int(row.index_age_days),
                block_size,
            )
            changed_rows, edit_metadata = apply_intervention(
                original_rows,
                int(row.patient_id_dense),
                int(row.index_age_days),
                intervention,
                block_size,
            )
            status_counts[edit_metadata["status"]] += 1
            if changed_rows is None:
                continue
            buffered.append(
                {
                    "person_id": int(row.person_id),
                    "patient_id_dense": int(row.patient_id_dense),
                    "index_age_days": int(row.index_age_days),
                    "duration_days": float(row.duration_days),
                    "event": int(row.event),
                    "original_rows": original_rows,
                    "changed_rows": changed_rows,
                    "edit_metadata": edit_metadata,
                }
            )
            if len(buffered) >= args.write_every:
                flush()
            if (
                args.max_patients_per_intervention > 0
                and status_counts["included"] >= args.max_patients_per_intervention
            ):
                break
        flush()
        for status, count in sorted(status_counts.items()):
            status_rows.append(
                {
                    "intervention_id": intervention["intervention_id"],
                    "status": status,
                    "patients": int(count),
                }
            )
        if status_counts["included"] == 0:
            log(f"[NO INCLUDED PATIENTS] {intervention['intervention_id']}")

    status_path = args.output_dir / "intervention_patient_status.csv"
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    if not pair_paths:
        raise RuntimeError(
            "No counterfactual pairs were generated. See intervention_patient_status.csv."
        )
    pair_summary, curve_summary, pair_summary_path, curve_summary_path = summarize_outputs(
        pair_paths, curve_paths, args.output_dir
    )
    plot_path = make_curve_plot(curve_summary, args.output_dir)

    manifest = {
        **input_paths,
        "registry_path": registry_path,
        "interventions_file": str(args.interventions_file),
        "resolved_interventions": str(interventions_path),
        "ckpt": str(args.ckpt),
        "checkpoint_step": checkpoint_step,
        "checkpoint_block_size": block_size,
        "requested_embedding_batch_size": args.batch_size,
        "full_length_embedding_batch_cap": full_length_batch_cap,
        "max_attention_cells": args.max_attention_cells,
        "phenotype": args.phenotype,
        "index_date": args.index_date,
        "horizon": args.horizon,
        "curve_days": curve_days,
        "analysis_definition": (
            "Fit embedding-only Cox once; edit visible pre-index tokens; recompute FERMAT "
            "embedding; compare paired Cox scores and risks."
        ),
        "causal_claim": False,
        "embedding_check": embedding_check,
        "cox_metrics": test_metrics,
        "outputs": {
            "cox_model": str(artifact_path),
            "cox_transformer": str(transformer_path),
            "cox_history": str(history_path),
            "cox_metrics": str(metrics_path),
            "intervention_status": str(status_path),
            "pair_summary": str(pair_summary_path),
            "curve_summary": str(curve_summary_path),
            "curve_plot": str(plot_path) if plot_path is not None else None,
            "raw_pair_parts": [str(path) for path in pair_paths],
            "raw_curve_parts": [str(path) for path in curve_paths],
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"[DONE] output_dir={args.output_dir}")
    print(pair_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
