#!/usr/bin/env python3
"""Audit candidate outcomes for Task 30 without fitting models or running rollouts.

The audit reuses the fixed-index Task 19 cohort, reviewed phenotype concept
groups, the full first-phenotype-date cache, baseline features, block-2048
embeddings, and the typed FERMAT ETL bins.  It reports whether each candidate
has enough incident events, a stable observed cumulative-incidence curve, a
usable target-token definition, and adequate baseline/embedding coverage.

Condition outcomes use Aalen-Johansen cumulative incidence with all-cause death
as a competing event.  All-cause mortality uses Kaplan-Meier mortality.  This
script deliberately does not fit Cox models, load a checkpoint, or generate
future trajectories.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    pq = None

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK_DIR = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK_DIR / "config" / "snuh_task30_outcome_feasibility.csv"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "patient_phenotype_labels_wide_20180101.parquet"
)
DEFAULT_CONCEPT_MAP = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task23"
    / "outputs"
    / "first_phenotype_dates_20180101_31phenotypes_full.parquet"
)
DEFAULT_FEATURE_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "baseline_features"
    / "baseline_features_20180101.parquet"
)
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_OUTPUT_DIR = TASK_DIR / "outputs" / "outcome_feasibility_audit"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_outcome_feasibility"

REQUIRED_CONFIG_COLUMNS = {
    "outcome_name",
    "audit_enabled",
    "definition_status",
    "source_kind",
    "source_phenotype",
    "index_date",
    "horizon_years",
    "washout_years",
    "event_definition",
    "existing_outcome_exclusion",
    "censoring_rule",
    "competing_event_rule",
    "token_rule",
    "clinical_comparator_status",
    "notes",
}
RESOLVED_SOURCE_KINDS = {"condition_group", "death_table"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--death-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--token-scan-chunk-rows", type=int, default=5_000_000)
    parser.add_argument("--token-date-tolerance-days", type=int, default=7)
    parser.add_argument("--min-train-events", type=int, default=100)
    parser.add_argument("--min-val-test-events", type=int, default=50)
    parser.add_argument("--min-token-capture-rate", type=float, default=0.90)
    parser.add_argument("--min-model-input-coverage", type=float, default=0.90)
    parser.add_argument("--etl-review-start-year", type=int, default=2010)
    parser.add_argument("--etl-spike-ratio", type=float, default=5.0)
    parser.add_argument("--etl-spike-min-current-events", type=int, default=20)
    parser.add_argument(
        "--resume-from-raw",
        action="store_true",
        help="Reuse raw source-date and completed split token scans only when run_config.json matches.",
    )
    parser.add_argument(
        "--skip-etl-token-scan",
        action="store_true",
        help="Diagnostic-only partial run. Token coverage remains unmeasured.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def timed_log(label: str, started: float, extra: str = ""):
    suffix = f" {extra}" if extra else ""
    log(f"[DONE] {label} seconds={time.time() - started:,.1f}{suffix}")


def write_json(value, path: Path):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def parquet_columns(path: Path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_outcome_config(path: Path):
    require_file(path)
    config = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_CONFIG_COLUMNS - set(config.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    config = config[list(REQUIRED_CONFIG_COLUMNS)].copy()
    config["audit_enabled"] = config["audit_enabled"].map(parse_bool)
    config = config.loc[config["audit_enabled"]].copy()
    if config.empty:
        raise ValueError("Outcome config has no enabled candidates")
    if config["outcome_name"].eq("").any():
        raise ValueError("Every enabled config row needs outcome_name")
    duplicates = config.loc[config["outcome_name"].duplicated(), "outcome_name"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate outcome_name values: {duplicates}")

    resolved = config["definition_status"].eq("resolved")
    bad_kind = config.loc[resolved & ~config["source_kind"].isin(RESOLVED_SOURCE_KINDS)]
    if not bad_kind.empty:
        raise ValueError(
            "Resolved candidates must use condition_group or death_table: "
            + ", ".join(bad_kind["outcome_name"])
        )
    condition = resolved & config["source_kind"].eq("condition_group")
    if config.loc[condition, "source_phenotype"].eq("").any():
        names = config.loc[condition & config["source_phenotype"].eq(""), "outcome_name"]
        raise ValueError("Condition candidates missing source_phenotype: " + ", ".join(names))
    if not config.loc[condition, "token_rule"].eq("dx_from_concept_map").all():
        raise ValueError("Resolved condition candidates must use token_rule=dx_from_concept_map")
    death = resolved & config["source_kind"].eq("death_table")
    if not config.loc[death, "token_rule"].eq("all_dth_tokens").all():
        raise ValueError("Resolved death candidates must use token_rule=all_dth_tokens")

    resolved_rows = config.loc[resolved]
    for column in ("index_date", "horizon_years", "washout_years"):
        if resolved_rows[column].eq("").any():
            raise ValueError(f"Resolved candidates require {column}")
        if resolved_rows[column].nunique() != 1:
            raise ValueError(
                f"This audit requires one shared {column}; found "
                f"{resolved_rows[column].drop_duplicates().tolist()}"
            )
    index_date = pd.Timestamp(resolved_rows["index_date"].iloc[0])
    horizon_years = int(resolved_rows["horizon_years"].iloc[0])
    washout_years = int(resolved_rows["washout_years"].iloc[0])
    if horizon_years < 5:
        raise ValueError("Task 30 feasibility audit requires horizon_years >= 5")
    if washout_years < 1:
        raise ValueError("washout_years must be positive")
    return config.reset_index(drop=True), index_date, horizon_years, washout_years


def normalize_paths(args):
    for name in (
        "config_file",
        "data_dir",
        "label_file",
        "concept_map",
        "survival_cache",
        "feature_file",
        "embedding_file",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.death_cache is not None:
        args.death_cache = args.death_cache.expanduser().resolve()


def run_fingerprint(args, config_sha, index_date, horizon_years, washout_years):
    return {
        "config_file": str(args.config_file),
        "config_sha256": config_sha,
        "data_dir": str(args.data_dir),
        "label_file": str(args.label_file),
        "concept_map": str(args.concept_map),
        "survival_cache": str(args.survival_cache),
        "feature_file": str(args.feature_file),
        "embedding_file": str(args.embedding_file),
        "death_cache": str(args.death_cache) if args.death_cache else None,
        "index_date": index_date.strftime("%Y-%m-%d"),
        "horizon_years": horizon_years,
        "washout_years": washout_years,
        "db_end_date": args.db_end_date,
        "token_date_tolerance_days": args.token_date_tolerance_days,
        "token_scan_chunk_rows": args.token_scan_chunk_rows,
        "min_train_events": args.min_train_events,
        "min_val_test_events": args.min_val_test_events,
        "min_token_capture_rate": args.min_token_capture_rate,
        "min_model_input_coverage": args.min_model_input_coverage,
        "etl_review_start_year": args.etl_review_start_year,
        "etl_spike_ratio": args.etl_spike_ratio,
        "etl_spike_min_current_events": args.etl_spike_min_current_events,
        "skip_etl_token_scan": bool(args.skip_etl_token_scan),
    }


def prepare_output(args, fingerprint):
    output = args.output_dir
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume_from_raw:
            raise FileExistsError(
                f"{output} exists and is not empty; use a new output directory or --resume-from-raw"
            )
        if not config_path.is_file():
            raise RuntimeError(f"Cannot resume without {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise RuntimeError(
                "Cannot resume because run_config.json does not match the requested inputs/settings"
            )
        log(f"[RESUME VALIDATED] {config_path}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "raw").mkdir(parents=True, exist_ok=True)
        write_json(fingerprint, config_path)
        log(f"[RAW SAVED] {config_path}")
    (output / "raw").mkdir(parents=True, exist_ok=True)


def registry_type_name(row):
    value = str(row.get("token_type", "")).strip().upper()
    if value:
        return value
    token_key = str(row.get("token_key", ""))
    return token_key.split(":", 1)[0].upper() if ":" in token_key else ""


def load_token_mapping(config, concept_map, registry):
    token_key_to_id = {
        str(row.token_key): int(row.token_id)
        for row in registry.itertuples(index=False)
        if str(row.token_key)
    }
    dth_tokens = sorted(
        {
            int(row.token_id)
            for row in registry.itertuples(index=False)
            if registry_type_name(row._asdict()) == "DTH"
        }
    )
    mapping = {}
    rows = []
    for item in config.itertuples(index=False):
        name = str(item.outcome_name)
        if item.definition_status != "resolved":
            mapping[name] = []
            rows.append(
                {
                    "outcome_name": name,
                    "source_kind": item.source_kind,
                    "source_phenotype": item.source_phenotype,
                    "configured_concepts": 0,
                    "matched_tokens": 0,
                    "concept_token_coverage": np.nan,
                    "stored_token_ids": "",
                    "token_keys": "",
                    "mapping_status": "not_resolved",
                }
            )
            continue
        if item.source_kind == "death_table":
            token_ids = dth_tokens
            keys = registry.loc[registry["token_id"].isin(token_ids), "token_key"].astype(str).tolist()
            configured_concepts = len(token_ids)
        else:
            group = concept_map.loc[
                concept_map["phenotype"].eq(item.source_phenotype)
            ].copy()
            concept_ids = sorted(group["condition_concept_id"].astype(np.int64).unique().tolist())
            keys = [f"DX:{concept_id}" for concept_id in concept_ids]
            token_ids = sorted({token_key_to_id[key] for key in keys if key in token_key_to_id})
            configured_concepts = len(concept_ids)
        mapping[name] = token_ids
        coverage = len(token_ids) / configured_concepts if configured_concepts else np.nan
        rows.append(
            {
                "outcome_name": name,
                "source_kind": item.source_kind,
                "source_phenotype": item.source_phenotype,
                "configured_concepts": configured_concepts,
                "matched_tokens": len(token_ids),
                "concept_token_coverage": coverage,
                "stored_token_ids": "|".join(str(value) for value in token_ids),
                "token_keys": "|".join(keys),
                "mapping_status": "mapped" if token_ids else "no_tokens",
            }
        )
    return mapping, pd.DataFrame(rows)


def require_psycopg():
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required when --death-cache is not supplied. Install "
            "with `python -m pip install \"psycopg[binary]>=3\"`."
        )


def db_password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


def query_death_dates(args):
    require_psycopg()
    started = time.time()
    log("[START] query all-cause death dates")
    with psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=db_password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name=APPLICATION_NAME,
        autocommit=True,
    ) as conn, conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
        statement = sql.SQL(
            """
            SELECT person_id::bigint, MIN(death_date)::date AS death_date
            FROM {}.death
            WHERE death_date IS NOT NULL
              AND death_date <= %s::date
            GROUP BY person_id
            ORDER BY person_id
            """
        ).format(sql.Identifier(args.schema))
        cur.execute(statement, (args.db_end_date,))
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    frame["death_date"] = pd.to_datetime(frame["death_date"], errors="coerce")
    timed_log("query all-cause death dates", started, f"rows={len(frame):,}")
    return frame


def load_death_dates(args):
    if args.death_cache is not None:
        require_file(args.death_cache)
        frame = pd.read_parquet(args.death_cache)
        date_column = "death_date"
        if date_column not in frame.columns:
            raise ValueError(f"{args.death_cache} is missing {date_column}")
        frame = frame[["person_id", date_column]].copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        log(f"[DEATH CACHE] {args.death_cache} rows={len(frame):,}")
        return frame
    return query_death_dates(args)


def build_or_load_source_dates(args, config):
    raw_path = args.output_dir / "raw" / "source_first_outcome_dates.parquet"
    death_path = args.output_dir / "raw" / "all_cause_death_dates.parquet"
    if args.resume_from_raw and raw_path.is_file() and death_path.is_file():
        source = pd.read_parquet(raw_path)
        death = pd.read_parquet(death_path)
        source["first_outcome_date"] = pd.to_datetime(source["first_outcome_date"], errors="coerce")
        death["death_date"] = pd.to_datetime(death["death_date"], errors="coerce")
        log(f"[RESUME] source dates rows={len(source):,}")
        return source, death

    started = time.time()
    cache = pd.read_parquet(args.survival_cache)
    required = {"phenotype", "person_id", "first_phenotype_date"}
    missing = sorted(required - set(cache.columns))
    if missing:
        raise ValueError(f"{args.survival_cache} is missing columns: {missing}")
    cache["first_phenotype_date"] = pd.to_datetime(cache["first_phenotype_date"], errors="coerce")

    death = load_death_dates(args)
    if death["person_id"].duplicated().any():
        raise ValueError("Death dates contain duplicate person_id rows")
    death.to_parquet(death_path, index=False)
    log(f"[RAW SAVED] {death_path}")

    parts = []
    for item in config.itertuples(index=False):
        if item.definition_status != "resolved":
            continue
        if item.source_kind == "condition_group":
            sub = cache.loc[
                cache["phenotype"].eq(item.source_phenotype),
                ["person_id", "first_phenotype_date"],
            ].copy()
            if sub.empty:
                raise ValueError(
                    f"No survival-cache rows for resolved outcome={item.outcome_name} "
                    f"source_phenotype={item.source_phenotype}"
                )
            sub = sub.rename(columns={"first_phenotype_date": "first_outcome_date"})
        else:
            sub = death.rename(columns={"death_date": "first_outcome_date"}).copy()
        sub.insert(0, "outcome_name", item.outcome_name)
        sub.insert(1, "source_kind", item.source_kind)
        parts.append(sub)
    source = pd.concat(parts, ignore_index=True)
    source = source.dropna(subset=["person_id", "first_outcome_date"])
    if source.duplicated(["outcome_name", "person_id"]).any():
        raise ValueError("Source dates contain duplicate outcome_name/person_id rows")
    source.to_parquet(raw_path, index=False)
    log(f"[RAW SAVED] {raw_path}")
    timed_log("build source first-outcome dates", started, f"rows={len(source):,}")
    return source, death


def open_typed_bin(path: Path):
    require_file(path)
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.size % 4 != 0:
        raise ValueError(f"{path} is not a four-column uint32 FERMAT bin")
    data = raw.reshape(-1, 4)
    if len(data):
        sample_size = min(len(data), 100_000)
        sample_index = np.linspace(0, len(data) - 1, sample_size, dtype=np.int64)
        if int(np.max(data[sample_index, 3])) >= 64:
            raise ValueError(f"{path} fourth column does not look like token_type_id")
    return data


def grouped_min(patient_ids, ages):
    if len(patient_ids) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(patient_ids, kind="mergesort")
    patient_ids = patient_ids[order].astype(np.int64, copy=False)
    ages = ages[order].astype(np.int64, copy=False)
    starts = np.flatnonzero(np.r_[True, patient_ids[1:] != patient_ids[:-1]])
    return patient_ids[starts], np.minimum.reduceat(ages, starts)


def scan_split_tokens(path: Path, split: str, outcome_bits, token_mask, chunk_rows: int):
    data = open_typed_bin(path)
    started = time.time()
    total_rows = len(data)
    accum = {name: ([], []) for name in outcome_bits}
    chunks = max(1, int(np.ceil(total_rows / chunk_rows)))
    log(f"[START] scan {split}.bin rows={total_rows:,} chunks={chunks:,}")
    for chunk_index, start in enumerate(range(0, total_rows, chunk_rows), start=1):
        stop = min(total_rows, start + chunk_rows)
        stored_tokens = np.asarray(data[start:stop, 2], dtype=np.int64)
        valid = stored_tokens < len(token_mask)
        masks = np.zeros(len(stored_tokens), dtype=np.uint64)
        masks[valid] = token_mask[stored_tokens[valid]]
        target = masks != 0
        if target.any():
            patient = np.asarray(data[start:stop, 0], dtype=np.int64)[target]
            age = np.asarray(data[start:stop, 1], dtype=np.int64)[target]
            selected_masks = masks[target]
            for name, bit in outcome_bits.items():
                selected = (selected_masks & bit) != 0
                if not selected.any():
                    continue
                unique_patient, first_age = grouped_min(patient[selected], age[selected])
                accum[name][0].append(unique_patient)
                accum[name][1].append(first_age)
        if chunk_index == 1 or chunk_index == chunks or chunk_index % max(1, chunks // 10) == 0:
            log(
                f"[PROGRESS] split={split} chunk={chunk_index:,}/{chunks:,} "
                f"rows={stop:,}/{total_rows:,}"
            )

    rows = []
    for name, (patient_parts, age_parts) in accum.items():
        if not patient_parts:
            continue
        patient, age = grouped_min(np.concatenate(patient_parts), np.concatenate(age_parts))
        rows.append(
            pd.DataFrame(
                {
                    "outcome_name": name,
                    "split": split,
                    "patient_id_dense": patient,
                    "first_token_age_days": age,
                }
            )
        )
    result = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=["outcome_name", "split", "patient_id_dense", "first_token_age_days"]
        )
    )
    elapsed = time.time() - started
    stats = {
        "split": split,
        "bin_path": str(path),
        "bin_bytes": int(path.stat().st_size),
        "bin_rows": int(total_rows),
        "target_patients_outcome_pairs": int(len(result)),
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed if elapsed > 0 else np.nan,
    }
    timed_log(f"scan {split}.bin", started, f"target_pairs={len(result):,}")
    return result, stats


def build_token_mask(mapping, registry):
    resolved = {name: tokens for name, tokens in mapping.items() if tokens}
    if len(resolved) > 63:
        raise ValueError("Token scan supports at most 63 resolved tokenized outcomes")
    max_token = int(pd.to_numeric(registry["token_id"], errors="raise").max())
    lookup = np.zeros(max_token + 1, dtype=np.uint64)
    bits = {}
    for index, (name, tokens) in enumerate(resolved.items()):
        bit = np.uint64(1) << np.uint64(index)
        bits[name] = bit
        for token in tokens:
            if token < 0 or token >= len(lookup):
                raise ValueError(f"Token ID outside registry bounds: {token}")
            lookup[token] |= bit
    return bits, lookup


def build_or_load_etl_token_dates(args, mapping, registry):
    if args.skip_etl_token_scan:
        log("[SKIP] ETL token scan; token capture will be reported as unmeasured")
        empty = pd.DataFrame(
            columns=["outcome_name", "split", "patient_id_dense", "first_token_age_days"]
        )
        return empty, pd.DataFrame()

    outcome_bits, token_mask = build_token_mask(mapping, registry)
    parts = []
    stats = []
    for split in SPLITS:
        raw_path = args.output_dir / "raw" / f"etl_first_outcome_token_age_{split}.parquet"
        stats_path = args.output_dir / "raw" / f"etl_token_scan_stats_{split}.json"
        if args.resume_from_raw and raw_path.is_file() and stats_path.is_file():
            part = pd.read_parquet(raw_path)
            stat = json.loads(stats_path.read_text(encoding="utf-8"))
            log(f"[RESUME] token scan split={split} rows={len(part):,}")
        else:
            part, stat = scan_split_tokens(
                args.data_dir / f"{split}.bin",
                split,
                outcome_bits,
                token_mask,
                args.token_scan_chunk_rows,
            )
            part.to_parquet(raw_path, index=False)
            write_json(stat, stats_path)
            log(f"[RAW SAVED] {raw_path}")
            log(f"[RAW SAVED] {stats_path}")
        parts.append(part)
        stats.append(stat)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(stats)


def validate_key_frame(frame, name, keys=("person_id", "split")):
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    duplicates = int(frame.duplicated(list(keys)).sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates:,} duplicate key rows")


def load_cohort_inputs(args, config, index_date):
    label_columns = set(parquet_columns(args.label_file))
    base_columns = [
        "person_id",
        "split",
        "index_date",
        "age_at_index",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
    ]
    condition_rows = config.loc[
        config["definition_status"].eq("resolved")
        & config["source_kind"].eq("condition_group")
    ]
    prior_columns = [f"prior__{value}" for value in condition_rows["source_phenotype"]]
    required = set(base_columns + prior_columns)
    missing = sorted(required - label_columns)
    if missing:
        raise ValueError(f"{args.label_file} is missing columns: {missing}")
    labels = pd.read_parquet(args.label_file, columns=base_columns + prior_columns)
    validate_key_frame(labels, "labels")
    for column in ("index_date", "first_activity_date", "last_activity_date"):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")
    unexpected = labels["index_date"].notna() & labels["index_date"].ne(index_date)
    if unexpected.any():
        raise ValueError(
            f"Label file has {int(unexpected.sum()):,} rows outside index_date={index_date.date()}"
        )

    features = pd.read_parquet(args.feature_file, columns=["person_id", "split"])
    validate_key_frame(features, "baseline features")
    features["has_baseline_features"] = True

    embedding_columns = set(parquet_columns(args.embedding_file))
    required_embedding = {"person_id", "split", "has_embedding_sequence"}
    missing = sorted(required_embedding - embedding_columns)
    if missing:
        raise ValueError(f"{args.embedding_file} is missing columns: {missing}")
    embeddings = pd.read_parquet(
        args.embedding_file,
        columns=["person_id", "split", "has_embedding_sequence"],
    )
    validate_key_frame(embeddings, "embeddings")
    embeddings["has_fermat_embedding"] = embeddings["has_embedding_sequence"].astype(bool)
    embeddings = embeddings[["person_id", "split", "has_fermat_embedding"]]

    patient_map = pd.read_parquet(
        args.data_dir / "patient_id_map.parquet",
        columns=["patient_id_dense", "person_id", "split"],
    )
    validate_key_frame(patient_map, "patient map", keys=("patient_id_dense", "split"))
    if patient_map["person_id"].duplicated().any():
        raise ValueError("Patient map has duplicate person_id values")

    base = labels.merge(features, on=["person_id", "split"], how="left")
    base = base.merge(embeddings, on=["person_id", "split"], how="left")
    base["has_baseline_features"] = base["has_baseline_features"].fillna(False).astype(bool)
    base["has_fermat_embedding"] = base["has_fermat_embedding"].fillna(False).astype(bool)
    base["has_both_model_inputs"] = base["has_baseline_features"] & base["has_fermat_embedding"]
    base["index_age_days"] = np.floor(
        pd.to_numeric(base["age_at_index"], errors="coerce") * 365.25
    ).astype("Int64")
    return base, patient_map


def validate_source_label_consistency(config, base, source_dates, index_date):
    rows = []
    condition_rows = config.loc[
        config["definition_status"].eq("resolved")
        & config["source_kind"].eq("condition_group")
    ]
    for item in condition_rows.itertuples(index=False):
        prior_column = f"prior__{item.source_phenotype}"
        source = source_dates.loc[
            source_dates["outcome_name"].eq(item.outcome_name),
            ["person_id", "first_outcome_date"],
        ]
        merged = base[["person_id", "split", prior_column]].merge(
            source,
            on="person_id",
            how="left",
            validate="one_to_one",
        )
        expected = merged["first_outcome_date"].notna() & (
            merged["first_outcome_date"] < index_date
        )
        observed = merged[prior_column].fillna(0).astype(bool)
        mismatch = expected != observed
        rows.append(
            {
                "outcome_name": item.outcome_name,
                "source_phenotype": item.source_phenotype,
                "patients": int(len(merged)),
                "source_prior_patients": int(expected.sum()),
                "label_prior_patients": int(observed.sum()),
                "prior_flag_mismatches": int(mismatch.sum()),
            }
        )
    result = pd.DataFrame(rows)
    return result


def attach_token_person_dates(token_dates, patient_map, base, index_date):
    if token_dates.empty:
        return pd.DataFrame(
            columns=[
                "outcome_name",
                "split",
                "patient_id_dense",
                "person_id",
                "first_token_age_days",
                "token_day_from_index",
            ]
        )
    merged = token_dates.merge(
        patient_map,
        on=["patient_id_dense", "split"],
        how="left",
        validate="many_to_one",
    )
    if merged["person_id"].isna().any():
        raise ValueError(
            f"Token scan has {int(merged['person_id'].isna().sum()):,} rows missing patient-map IDs"
        )
    age = base[["person_id", "split", "index_age_days"]]
    merged = merged.merge(age, on=["person_id", "split"], how="left", validate="many_to_one")
    if merged["index_age_days"].isna().any():
        raise ValueError(
            f"Token scan has {int(merged['index_age_days'].isna().sum()):,} rows missing index age"
        )
    merged["token_day_from_index"] = (
        merged["first_token_age_days"].astype(np.int64)
        - merged["index_age_days"].astype(np.int64)
    )
    return merged.drop(columns=["index_age_days"])


def build_followup(base, source_sub, death_dates, item, index_date, horizon_end, db_end):
    censor_limit = min(horizon_end, db_end)
    task = base.merge(source_sub, on="person_id", how="left", validate="one_to_one")
    task = task.merge(death_dates, on="person_id", how="left", validate="one_to_one")
    has_washout = task["has_pre_index_washout"].fillna(False).astype(bool)
    outcome_prior = task["first_outcome_date"].notna() & (
        task["first_outcome_date"] < index_date
    )
    death_prior = task["death_date"].notna() & (task["death_date"] <= index_date)
    at_risk = has_washout & ~outcome_prior & ~death_prior
    task = task.loc[at_risk].copy()

    base_censor = task["last_activity_date"].where(
        task["last_activity_date"].notna(), censor_limit
    )
    base_censor = base_censor.clip(upper=censor_limit)
    if item.source_kind == "death_table":
        outcome_event = (
            task["death_date"].notna()
            & (task["death_date"] > index_date)
            & (task["death_date"] <= censor_limit)
        )
        competing = pd.Series(False, index=task.index)
        endpoint = base_censor.where(~outcome_event, task["death_date"])
    else:
        possible_outcome = (
            task["first_outcome_date"].notna()
            & (task["first_outcome_date"] >= index_date)
            & (task["first_outcome_date"] <= base_censor)
        )
        possible_death = (
            task["death_date"].notna()
            & (task["death_date"] > index_date)
            & (task["death_date"] <= base_censor)
        )
        outcome_event = possible_outcome & (
            ~possible_death | (task["first_outcome_date"] <= task["death_date"])
        )
        competing = possible_death & (
            ~possible_outcome | (task["death_date"] < task["first_outcome_date"])
        )
        endpoint = base_censor.copy()
        endpoint = endpoint.where(~outcome_event, task["first_outcome_date"])
        endpoint = endpoint.where(~competing, task["death_date"])

    task["duration_days"] = (endpoint - index_date).dt.days.astype("float64")
    task["event_type"] = np.select([outcome_event, competing], [1, 2], default=0).astype("int8")
    task["source_day_from_index"] = (
        task["first_outcome_date"] - index_date
    ).dt.days.astype("float64")
    task = task.loc[task["duration_days"] > 0].copy()
    task["duration_days"] = task["duration_days"].astype(np.int32)
    return task


def cumulative_incidence_curve(task, max_day):
    if task.empty:
        return pd.DataFrame(
            {
                "day": np.arange(max_day + 1),
                "at_risk": 0,
                "outcome_events_on_day": 0,
                "competing_deaths_on_day": 0,
                "censored_on_day": 0,
                "observed_cumulative_incidence": np.nan,
                "event_free_survival": np.nan,
            }
        )
    duration = task["duration_days"].to_numpy(dtype=np.int64)
    event_type = task["event_type"].to_numpy(dtype=np.int8)
    outcome_counts = np.bincount(duration[event_type == 1], minlength=max_day + 1)[: max_day + 1]
    competing_counts = np.bincount(duration[event_type == 2], minlength=max_day + 1)[: max_day + 1]
    censor_counts = np.bincount(duration[event_type == 0], minlength=max_day + 1)[: max_day + 1]
    at_risk = len(task)
    survival = 1.0
    cumulative_incidence = 0.0
    rows = []
    for day in range(max_day + 1):
        events = int(outcome_counts[day])
        competing = int(competing_counts[day])
        censored = int(censor_counts[day])
        before = at_risk
        if before > 0:
            previous_survival = survival
            cumulative_incidence += previous_survival * events / before
            survival *= 1.0 - (events + competing) / before
        rows.append(
            {
                "day": day,
                "at_risk": before,
                "outcome_events_on_day": events,
                "competing_deaths_on_day": competing,
                "censored_on_day": censored,
                "observed_cumulative_incidence": cumulative_incidence,
                "event_free_survival": survival,
            }
        )
        at_risk -= events + competing + censored
        if at_risk < 0:
            raise RuntimeError("Risk set became negative")
    return pd.DataFrame(rows)


def quantile_or_nan(values, quantile):
    return float(np.quantile(values, quantile)) if len(values) else np.nan


def summarize_task(
    task,
    split,
    outcome_name,
    horizon_years,
    horizon_day,
    token_sub,
    tolerance_days,
    token_scan_measured,
):
    if split == "all":
        sub = task
    else:
        sub = task.loc[task["split"].eq(split)]
    events = sub.loc[sub["event_type"].eq(1)].copy()
    if token_scan_measured:
        merge_columns = ["person_id", "token_day_from_index"]
        merged = events.merge(token_sub[merge_columns], on="person_id", how="left", validate="one_to_one")
        any_token = (
            merged["token_day_from_index"].notna()
            & (merged["token_day_from_index"] >= -tolerance_days)
            & (merged["token_day_from_index"] <= horizon_day + tolerance_days)
        )
        exact_token = any_token & (
            (merged["token_day_from_index"] - merged["source_day_from_index"]).abs()
            <= tolerance_days
        )
        any_rate = float(any_token.mean()) if len(merged) else np.nan
        exact_rate = float(exact_token.mean()) if len(merged) else np.nan
        exact_count = int(exact_token.sum())
    else:
        any_rate = np.nan
        exact_rate = np.nan
        exact_count = np.nan
    durations = events["duration_days"].to_numpy(dtype=np.float64)
    return {
        "outcome_name": outcome_name,
        "split": split,
        "horizon_years": horizon_years,
        "eligible_patients": int(len(sub)),
        "outcome_events": int(len(events)),
        "competing_deaths": int(sub["event_type"].eq(2).sum()),
        "censored": int(sub["event_type"].eq(0).sum()),
        "raw_event_fraction": float(len(events) / len(sub)) if len(sub) else np.nan,
        "median_followup_days": float(sub["duration_days"].median()) if len(sub) else np.nan,
        "event_day_p10": quantile_or_nan(durations, 0.10),
        "event_day_p25": quantile_or_nan(durations, 0.25),
        "event_day_median": quantile_or_nan(durations, 0.50),
        "event_day_p75": quantile_or_nan(durations, 0.75),
        "event_day_p90": quantile_or_nan(durations, 0.90),
        "baseline_feature_patients": int(sub["has_baseline_features"].sum()),
        "fermat_embedding_patients": int(sub["has_fermat_embedding"].sum()),
        "both_model_input_patients": int(sub["has_both_model_inputs"].sum()),
        "both_model_input_coverage": float(sub["has_both_model_inputs"].mean()) if len(sub) else np.nan,
        "source_events_with_exact_etl_token": exact_count,
        "source_event_any_etl_token_rate": any_rate,
        "source_event_exact_etl_token_rate": exact_rate,
        "token_scan_status": "measured" if token_scan_measured else "not_measured",
    }


def build_outcome_summaries(
    args,
    config,
    base,
    source_dates,
    death_dates,
    token_person_dates,
    index_date,
    horizon_years,
):
    db_end = pd.Timestamp(args.db_end_date)
    summary_rows = []
    curve_parts = []
    max_horizon_end = index_date + pd.DateOffset(years=horizon_years)
    max_day = int((max_horizon_end - index_date).days)
    resolved = config.loc[config["definition_status"].eq("resolved")]
    for item in resolved.itertuples(index=False):
        started = time.time()
        source_sub = source_dates.loc[
            source_dates["outcome_name"].eq(item.outcome_name),
            ["person_id", "first_outcome_date"],
        ]
        token_sub = token_person_dates.loc[
            token_person_dates["outcome_name"].eq(item.outcome_name),
            ["person_id", "token_day_from_index"],
        ]
        if token_sub["person_id"].duplicated().any():
            raise ValueError(f"Duplicate token-person dates for outcome={item.outcome_name}")
        for years in (1, 3, 5):
            horizon_end = index_date + pd.DateOffset(years=years)
            horizon_day = int((horizon_end - index_date).days)
            task = build_followup(
                base,
                source_sub,
                death_dates,
                item,
                index_date,
                horizon_end,
                db_end,
            )
            for split in (*SPLITS, "all"):
                row = summarize_task(
                    task,
                    split,
                    item.outcome_name,
                    years,
                    horizon_day,
                    token_sub,
                    args.token_date_tolerance_days,
                    not args.skip_etl_token_scan,
                )
                sub = task if split == "all" else task.loc[task["split"].eq(split)]
                curve = cumulative_incidence_curve(sub, horizon_day)
                row["observed_cumulative_incidence"] = float(
                    curve["observed_cumulative_incidence"].iloc[-1]
                )
                summary_rows.append(row)
                if years == horizon_years:
                    curve.insert(0, "split", split)
                    curve.insert(0, "outcome_name", item.outcome_name)
                    curve_parts.append(curve)
        timed_log(f"summarize outcome={item.outcome_name}", started)
    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_parts, ignore_index=True)
    return summary, curves


def build_year_counts(config, base, source_dates, db_end_date, review_start_year):
    merged = source_dates.merge(
        base[["person_id", "split"]],
        on="person_id",
        how="inner",
        validate="many_to_one",
    )
    merged = merged.loc[merged["first_outcome_date"] <= pd.Timestamp(db_end_date)].copy()
    merged["event_year"] = merged["first_outcome_date"].dt.year.astype(int)
    grouped = (
        merged.groupby(["outcome_name", "split", "event_year"], as_index=False)
        .agg(first_event_patients=("person_id", "nunique"))
    )
    total = (
        merged.groupby(["outcome_name", "event_year"], as_index=False)
        .agg(first_event_patients=("person_id", "nunique"))
    )
    total["split"] = "all"
    counts = pd.concat([grouped, total], ignore_index=True)

    resolved_names = config.loc[
        config["definition_status"].eq("resolved"), "outcome_name"
    ].tolist()
    end_year = pd.Timestamp(db_end_date).year
    observed_min_year = (
        int(counts["event_year"].min()) if len(counts) else review_start_year
    )
    start_year = min(observed_min_year, review_start_year)
    grid = pd.MultiIndex.from_product(
        [resolved_names, (*SPLITS, "all"), range(start_year, end_year + 1)],
        names=["outcome_name", "split", "event_year"],
    ).to_frame(index=False)
    counts = grid.merge(counts, on=["outcome_name", "split", "event_year"], how="left")
    counts["first_event_patients"] = counts["first_event_patients"].fillna(0).astype(int)
    return counts.sort_values(["outcome_name", "split", "event_year"])


def build_spike_flags(
    year_counts, ratio_threshold, min_current_events, review_start_year
):
    rows = []
    for outcome_name, group in year_counts.loc[year_counts["split"].eq("all")].groupby(
        "outcome_name", sort=True
    ):
        group = group.loc[group["event_year"] >= review_start_year].sort_values("event_year")
        values = group["first_event_patients"].to_numpy(dtype=float)
        years = group["event_year"].to_numpy(dtype=int)
        best_ratio = np.nan
        best_year = np.nan
        best_count = np.nan
        best_previous_median = np.nan
        for index in range(3, len(group)):
            previous_median = float(np.median(values[index - 3 : index]))
            current = float(values[index])
            ratio = current / max(previous_median, 1.0)
            if np.isnan(best_ratio) or ratio > best_ratio:
                best_ratio = ratio
                best_year = int(years[index])
                best_count = int(current)
                best_previous_median = previous_median
        flagged = bool(
            np.isfinite(best_ratio)
            and best_ratio >= ratio_threshold
            and best_count >= min_current_events
        )
        rows.append(
            {
                "outcome_name": outcome_name,
                "max_three_year_median_ratio": best_ratio,
                "max_ratio_year": best_year,
                "events_in_max_ratio_year": best_count,
                "previous_three_year_median": best_previous_median,
                "etl_spike_flag": flagged,
                "flag_rule": (
                    f"ratio>={ratio_threshold:g} and current_events>={min_current_events}"
                ),
            }
        )
    return pd.DataFrame(rows)


def lookup_summary(summary, outcome_name, split, years, column):
    row = summary.loc[
        summary["outcome_name"].eq(outcome_name)
        & summary["split"].eq(split)
        & summary["horizon_years"].eq(years),
        column,
    ]
    return row.iloc[0] if len(row) else np.nan


def build_feasibility_summary(args, config, token_mapping, summary, spike_flags):
    mapping_lookup = token_mapping.set_index("outcome_name").to_dict("index")
    spike_lookup = spike_flags.set_index("outcome_name").to_dict("index")
    rows = []
    for item in config.itertuples(index=False):
        name = item.outcome_name
        row = {
            "outcome_name": name,
            "definition_status": item.definition_status,
            "source_kind": item.source_kind,
            "source_phenotype": item.source_phenotype,
            "event_definition": item.event_definition,
            "censoring_rule": item.censoring_rule,
            "competing_event_rule": item.competing_event_rule,
            "clinical_comparator_status": item.clinical_comparator_status,
            "notes": item.notes,
            "rollout_hit_rate": np.nan,
            "rollout_hit_rate_status": "not_measured_until_benchmark",
            "rollout_runtime_status": "not_measured_until_benchmark",
        }
        if item.definition_status != "resolved":
            row.update(
                {
                    "configured_concepts": 0,
                    "matched_tokens": 0,
                    "concept_token_coverage": np.nan,
                    "train_events_5y": np.nan,
                    "val_events_5y": np.nan,
                    "test_events_1y": np.nan,
                    "test_events_3y": np.nan,
                    "test_events_5y": np.nan,
                    "test_cumulative_incidence_1y": np.nan,
                    "test_cumulative_incidence_3y": np.nan,
                    "test_cumulative_incidence_5y": np.nan,
                    "test_exact_etl_token_capture_5y": np.nan,
                    "test_model_input_coverage_5y": np.nan,
                    "etl_spike_flag": np.nan,
                    "screening_status": "definition_required",
                    "screening_reasons": "reviewed source concept set is not yet available",
                }
            )
            rows.append(row)
            continue

        mapping = mapping_lookup[name]
        spike = spike_lookup.get(name, {"etl_spike_flag": np.nan})
        train_events = lookup_summary(summary, name, "train", 5, "outcome_events")
        val_events = lookup_summary(summary, name, "val", 5, "outcome_events")
        test_events = lookup_summary(summary, name, "test", 5, "outcome_events")
        token_capture = lookup_summary(
            summary, name, "test", 5, "source_event_exact_etl_token_rate"
        )
        model_coverage = lookup_summary(
            summary, name, "test", 5, "both_model_input_coverage"
        )
        reasons = []
        if int(mapping["matched_tokens"]) == 0:
            reasons.append("no mapped target token")
        if train_events < args.min_train_events:
            reasons.append(f"train 5y events < {args.min_train_events}")
        if val_events < args.min_val_test_events:
            reasons.append(f"validation 5y events < {args.min_val_test_events}")
        if test_events < args.min_val_test_events:
            reasons.append(f"test 5y events < {args.min_val_test_events}")
        if args.skip_etl_token_scan:
            reasons.append("ETL token capture not measured")
        elif not np.isfinite(token_capture) or token_capture < args.min_token_capture_rate:
            reasons.append(f"exact ETL token capture < {args.min_token_capture_rate:.2f}")
        if bool(spike.get("etl_spike_flag", False)):
            reasons.append("first-event year spike needs ETL review")
        if not np.isfinite(model_coverage) or model_coverage < args.min_model_input_coverage:
            reasons.append(f"test model-input coverage < {args.min_model_input_coverage:.2f}")
        status = "review_ready" if not reasons else "needs_review"
        row.update(
            {
                "configured_concepts": int(mapping["configured_concepts"]),
                "matched_tokens": int(mapping["matched_tokens"]),
                "concept_token_coverage": mapping["concept_token_coverage"],
                "train_events_5y": train_events,
                "val_events_5y": val_events,
                "test_events_1y": lookup_summary(summary, name, "test", 1, "outcome_events"),
                "test_events_3y": lookup_summary(summary, name, "test", 3, "outcome_events"),
                "test_events_5y": test_events,
                "test_cumulative_incidence_1y": lookup_summary(
                    summary, name, "test", 1, "observed_cumulative_incidence"
                ),
                "test_cumulative_incidence_3y": lookup_summary(
                    summary, name, "test", 3, "observed_cumulative_incidence"
                ),
                "test_cumulative_incidence_5y": lookup_summary(
                    summary, name, "test", 5, "observed_cumulative_incidence"
                ),
                "test_exact_etl_token_capture_5y": token_capture,
                "test_model_input_coverage_5y": model_coverage,
                "etl_spike_flag": spike.get("etl_spike_flag", np.nan),
                "screening_status": status,
                "screening_reasons": " | ".join(reasons),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_definition_status(config, concept_map):
    known_phenotypes = set(concept_map["phenotype"].astype(str))
    rows = []
    for item in config.itertuples(index=False):
        resolved = item.definition_status == "resolved"
        concept_available = (
            item.source_kind == "death_table"
            or (item.source_kind == "condition_group" and item.source_phenotype in known_phenotypes)
        )
        status = "ready" if resolved and concept_available else item.definition_status
        rows.append(
            {
                "outcome_name": item.outcome_name,
                "definition_status": item.definition_status,
                "source_kind": item.source_kind,
                "source_phenotype": item.source_phenotype,
                "concept_definition_available": bool(concept_available),
                "audit_status": status,
                "event_definition": item.event_definition,
                "existing_outcome_exclusion": item.existing_outcome_exclusion,
                "censoring_rule": item.censoring_rule,
                "competing_event_rule": item.competing_event_rule,
                "token_rule": item.token_rule,
                "clinical_comparator_status": item.clinical_comparator_status,
                "notes": item.notes,
            }
        )
    return pd.DataFrame(rows)


def run_self_test():
    patient, age = grouped_min(
        np.array([2, 1, 2, 1], dtype=np.int64),
        np.array([9, 7, 3, 8], dtype=np.int64),
    )
    if patient.tolist() != [1, 2] or age.tolist() != [7, 3]:
        raise AssertionError("grouped_min failed")
    task = pd.DataFrame(
        {
            "duration_days": [10, 10, 20],
            "event_type": [1, 2, 0],
        }
    )
    curve = cumulative_incidence_curve(task, 20)
    if not np.isclose(curve.loc[curve["day"].eq(10), "observed_cumulative_incidence"].iloc[0], 1 / 3):
        raise AssertionError("Aalen-Johansen outcome increment failed")
    if int(curve.iloc[-1]["at_risk"]) != 1:
        raise AssertionError("risk-set accounting failed")
    index_date = pd.Timestamp("2018-01-01")
    base = pd.DataFrame(
        {
            "person_id": [1, 2],
            "split": ["test", "test"],
            "first_activity_date": [pd.Timestamp("2015-01-01")] * 2,
            "last_activity_date": [pd.Timestamp("2018-04-11")] * 2,
            "has_pre_index_washout": [True, True],
            "has_baseline_features": [True, True],
            "has_fermat_embedding": [True, True],
            "has_both_model_inputs": [True, True],
        }
    )
    source = pd.DataFrame(
        {
            "person_id": [1],
            "first_outcome_date": [pd.Timestamp("2018-01-31")],
        }
    )
    death = pd.DataFrame(
        {
            "person_id": [2],
            "death_date": [pd.Timestamp("2018-01-21")],
        }
    )
    followup = build_followup(
        base,
        source,
        death,
        SimpleNamespace(source_kind="condition_group"),
        index_date,
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2025-02-05"),
    )
    if sorted(followup["event_type"].tolist()) != [1, 2]:
        raise AssertionError("outcome-versus-competing-death accounting failed")
    log(
        "[SELF-TEST PASS] grouped minimum, cumulative incidence, and competing-event follow-up"
    )


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    if args.token_scan_chunk_rows <= 0:
        raise ValueError("--token-scan-chunk-rows must be positive")
    for path in (
        args.config_file,
        args.data_dir / "patient_id_map.parquet",
        args.data_dir / "token_registry.csv",
        args.label_file,
        args.concept_map,
        args.survival_cache,
        args.feature_file,
        args.embedding_file,
    ):
        require_file(path)
    if not args.skip_etl_token_scan:
        for split in SPLITS:
            require_file(args.data_dir / f"{split}.bin")

    overall_started = time.time()
    config, index_date, horizon_years, washout_years = load_outcome_config(args.config_file)
    if horizon_years != 5:
        raise ValueError(
            f"This implementation reports fixed 1y/3y/5y summaries and requires horizon_years=5; "
            f"got {horizon_years}"
        )
    fingerprint = run_fingerprint(
        args,
        sha256_file(args.config_file),
        index_date,
        horizon_years,
        washout_years,
    )
    prepare_output(args, fingerprint)

    concept_map = pd.read_csv(args.concept_map)
    required_concept = {"phenotype", "condition_concept_id"}
    missing = sorted(required_concept - set(concept_map.columns))
    if missing:
        raise ValueError(f"{args.concept_map} is missing columns: {missing}")
    concept_map["condition_concept_id"] = pd.to_numeric(
        concept_map["condition_concept_id"], errors="raise"
    ).astype(np.int64)
    registry = pd.read_csv(args.data_dir / "token_registry.csv", dtype={"token_key": str})
    required_registry = {"token_id", "token_key"}
    missing = sorted(required_registry - set(registry.columns))
    if missing:
        raise ValueError(f"token_registry.csv is missing columns: {missing}")
    registry["token_id"] = pd.to_numeric(registry["token_id"], errors="raise").astype(np.int64)

    definitions = candidate_definition_status(config, concept_map)
    definitions.to_csv(args.output_dir / "candidate_definition_status.csv", index=False)
    token_mapping_dict, token_mapping = load_token_mapping(config, concept_map, registry)
    token_mapping.to_csv(args.output_dir / "outcome_token_mapping.csv", index=False)

    source_dates, death_dates = build_or_load_source_dates(args, config)
    base, patient_map = load_cohort_inputs(args, config, index_date)
    source_label_consistency = validate_source_label_consistency(
        config, base, source_dates, index_date
    )
    source_label_consistency.to_csv(
        args.output_dir / "source_label_prior_consistency.csv", index=False
    )
    total_prior_mismatch = int(source_label_consistency["prior_flag_mismatches"].sum())
    if total_prior_mismatch:
        bad = source_label_consistency.loc[
            source_label_consistency["prior_flag_mismatches"] > 0
        ]
        raise RuntimeError(
            "Survival cache and fixed-index label prior flags disagree; refusing to audit.\n"
            + bad.to_string(index=False)
        )
    token_dates, scan_stats = build_or_load_etl_token_dates(args, token_mapping_dict, registry)
    token_person_dates = attach_token_person_dates(
        token_dates, patient_map, base, index_date
    )
    if not token_person_dates.empty:
        token_person_path = args.output_dir / "raw" / "etl_first_outcome_token_person_dates.parquet"
        token_person_dates.to_parquet(token_person_path, index=False)
        log(f"[RAW SAVED] {token_person_path}")
    if not scan_stats.empty:
        scan_stats.to_csv(args.output_dir / "etl_token_scan_runtime.csv", index=False)

    summary, curves = build_outcome_summaries(
        args,
        config,
        base,
        source_dates,
        death_dates,
        token_person_dates,
        index_date,
        horizon_years,
    )
    summary.to_csv(args.output_dir / "outcome_split_horizon_summary.csv", index=False)
    curves.to_csv(args.output_dir / "observed_cumulative_incidence_curves.csv", index=False)

    year_counts = build_year_counts(
        config,
        base,
        source_dates,
        args.db_end_date,
        args.etl_review_start_year,
    )
    year_counts.to_csv(args.output_dir / "outcome_first_event_year_counts.csv", index=False)
    spike_flags = build_spike_flags(
        year_counts,
        args.etl_spike_ratio,
        args.etl_spike_min_current_events,
        args.etl_review_start_year,
    )
    spike_flags.to_csv(args.output_dir / "outcome_etl_spike_flags.csv", index=False)

    feasibility = build_feasibility_summary(
        args,
        config,
        token_mapping,
        summary,
        spike_flags,
    )
    feasibility.to_csv(args.output_dir / "outcome_feasibility_summary.csv", index=False)

    manifest = {
        **fingerprint,
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - overall_started,
        "resolved_outcomes": int(config["definition_status"].eq("resolved").sum()),
        "definition_required_outcomes": int((config["definition_status"] != "resolved").sum()),
        "curve_method": {
            "condition_group": "Aalen-Johansen cumulative incidence with all-cause death as competing event",
            "death_table": "Kaplan-Meier all-cause mortality (equivalent single-event cumulative incidence)",
        },
        "screening_thresholds": {
            "min_train_events": args.min_train_events,
            "min_val_test_events": args.min_val_test_events,
            "min_token_capture_rate": args.min_token_capture_rate,
            "min_model_input_coverage": args.min_model_input_coverage,
            "etl_review_start_year": args.etl_review_start_year,
            "etl_spike_ratio": args.etl_spike_ratio,
            "etl_spike_min_current_events": args.etl_spike_min_current_events,
        },
        "explicitly_not_run": ["Cox fitting", "checkpoint loading", "rollout generation"],
        "outputs": {
            "primary": str(args.output_dir / "outcome_feasibility_summary.csv"),
            "split_horizon": str(args.output_dir / "outcome_split_horizon_summary.csv"),
            "curves": str(args.output_dir / "observed_cumulative_incidence_curves.csv"),
            "token_mapping": str(args.output_dir / "outcome_token_mapping.csv"),
            "year_counts": str(args.output_dir / "outcome_first_event_year_counts.csv"),
            "etl_flags": str(args.output_dir / "outcome_etl_spike_flags.csv"),
            "definition_status": str(args.output_dir / "candidate_definition_status.csv"),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    log("[COMPLETE] Task 30 outcome feasibility audit finished")
    log(feasibility.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
