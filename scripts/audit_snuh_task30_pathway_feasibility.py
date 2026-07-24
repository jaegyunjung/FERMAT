#!/usr/bin/env python3
"""CPU-only feasibility audit for recorded disease pathways in Task 30.

For each configured pathway, the audit selects patients with the required
source diagnosis or diagnoses recorded before a fixed index date and without
the target diagnosis on or before that date.  It reports eligible patients,
1/3/5-year target events, source-to-index recency, two-source ordering, and
observed target cumulative-incidence curves with all-cause death as a competing
event.

The script reads saved Task 19/23 Parquet files and reviewed phenotype concept
groups.  It does not import torch, load a model checkpoint, fit a model, or run
future generation.  If a death-date cache is not supplied, it performs one
read-only aggregate query against the CDM death table.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
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
except ModuleNotFoundError:  # pragma: no cover - optional when cache supplied
    psycopg = None
    sql = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK_DIR = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK_DIR / "config" / "snuh_task30_pathway_feasibility.csv"
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
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_OUTPUT_DIR = TASK_DIR / "outputs" / "pathway_feasibility_audit"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_pathway_feasibility"
SPLITS = ("train", "val", "test")
LANDMARK_YEARS = (1, 3, 5)

REQUIRED_CONFIG_COLUMNS = {
    "pathway_id",
    "audit_enabled",
    "source_a",
    "source_b",
    "source_requirement",
    "target",
    "index_date",
    "horizon_years",
    "washout_years",
    "definition_status",
    "clinical_question",
    "interpretation_limit",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--death-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432"))
    )
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument(
        "--recency-cut-days",
        nargs=3,
        type=int,
        default=[180, 730, 1826],
        metavar=("RECENT", "MIDDLE", "REMOTE"),
        help="Positive source-to-index recency cut points for four strata.",
    )
    parser.add_argument("--min-test-eligible", type=int, default=500)
    parser.add_argument("--min-test-events-5y", type=int, default=50)
    parser.add_argument("--min-recency-stratum-patients", type=int, default=100)
    parser.add_argument("--min-populated-recency-strata", type=int, default=3)
    parser.add_argument("--resume-from-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def atomic_to_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)


def parquet_columns(path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def normalize_paths(args):
    for name in (
        "config_file",
        "label_file",
        "concept_map",
        "survival_cache",
        "data_dir",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.death_cache is not None:
        args.death_cache = args.death_cache.expanduser().resolve()


def load_config(path):
    require_file(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_CONFIG_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[list(REQUIRED_CONFIG_COLUMNS)].copy()
    frame["audit_enabled"] = frame["audit_enabled"].map(parse_bool)
    frame = frame.loc[frame["audit_enabled"]].copy()
    if frame.empty:
        raise ValueError("No enabled pathways")
    if frame["pathway_id"].eq("").any() or frame["pathway_id"].duplicated().any():
        raise ValueError("pathway_id must be nonempty and unique")
    if not frame["source_requirement"].isin({"single", "both"}).all():
        raise ValueError("source_requirement must be single or both")
    single = frame["source_requirement"].eq("single")
    both = frame["source_requirement"].eq("both")
    if frame.loc[single, "source_b"].ne("").any():
        raise ValueError("single-source pathways must leave source_b blank")
    if frame.loc[both, "source_b"].eq("").any():
        raise ValueError("both-source pathways require source_b")
    if (frame["source_a"].eq("") | frame["target"].eq("")).any():
        raise ValueError("Every pathway requires source_a and target")
    if (frame["source_a"] == frame["target"]).any():
        raise ValueError("source_a and target must be distinct")
    if (both & (frame["source_b"] == frame["target"])).any():
        raise ValueError("source_b and target must be distinct")
    for column in ("index_date", "horizon_years", "washout_years"):
        if frame[column].nunique() != 1:
            raise ValueError(f"All pathways must share one {column}")
    index_date = pd.Timestamp(frame["index_date"].iloc[0])
    horizon_years = int(frame["horizon_years"].iloc[0])
    washout_years = int(frame["washout_years"].iloc[0])
    if horizon_years != 5:
        raise ValueError("This audit requires horizon_years=5")
    if washout_years < 1:
        raise ValueError("washout_years must be positive")
    return frame.sort_values("pathway_id").reset_index(drop=True), index_date, washout_years


def fingerprint(args, config_sha, index_date, washout_years):
    return {
        "config_file": str(args.config_file),
        "config_sha256": config_sha,
        "label_file": str(args.label_file),
        "concept_map": str(args.concept_map),
        "survival_cache": str(args.survival_cache),
        "data_dir": str(args.data_dir),
        "death_cache": str(args.death_cache) if args.death_cache else None,
        "index_date": index_date.strftime("%Y-%m-%d"),
        "washout_years": int(washout_years),
        "db_end_date": args.db_end_date,
        "recency_cut_days": list(args.recency_cut_days),
        "min_test_eligible": int(args.min_test_eligible),
        "min_test_events_5y": int(args.min_test_events_5y),
        "min_recency_stratum_patients": int(args.min_recency_stratum_patients),
        "min_populated_recency_strata": int(args.min_populated_recency_strata),
    }


def prepare_output(args, run_config):
    output = args.output_dir
    config_path = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            shutil.rmtree(output)
        elif args.resume_from_raw:
            if not config_path.is_file():
                raise RuntimeError(f"Cannot resume without {config_path}")
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != run_config:
                raise RuntimeError("Saved run_config.json does not match this invocation")
            log(f"[RESUME VALIDATED] {config_path}")
            return
        else:
            raise FileExistsError(
                f"{output} exists and is not empty; use a new directory, "
                "--resume-from-raw, or --overwrite"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    write_json(run_config, config_path)
    log(f"[RAW SAVED] {config_path}")


def db_password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


def query_death_dates(args):
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required without --death-cache; install psycopg[binary] "
            "or supply a saved death-date Parquet file"
        )
    started = time.time()
    log("[START] read all-cause death dates")
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
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=["person_id", "death_date"])
    frame["death_date"] = pd.to_datetime(frame["death_date"], errors="coerce")
    log(f"[DONE] death dates rows={len(frame):,} seconds={time.time() - started:,.1f}")
    return frame


def load_death_dates(args):
    saved = args.output_dir / "raw" / "all_cause_death_dates.parquet"
    if args.resume_from_raw and saved.is_file():
        frame = pd.read_parquet(saved)
        frame["death_date"] = pd.to_datetime(frame["death_date"], errors="coerce")
        log(f"[RESUME] death dates rows={len(frame):,}")
        return frame
    if args.death_cache is not None:
        require_file(args.death_cache)
        frame = pd.read_parquet(args.death_cache, columns=["person_id", "death_date"])
        frame["death_date"] = pd.to_datetime(frame["death_date"], errors="coerce")
    else:
        frame = query_death_dates(args)
    if frame["person_id"].duplicated().any():
        raise ValueError("Death cache contains duplicate person_id rows")
    atomic_to_parquet(frame, saved)
    log(f"[RAW SAVED] {saved}")
    return frame


def load_base(args, index_date):
    required = {
        "person_id",
        "split",
        "index_date",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
    }
    available = set(parquet_columns(args.label_file))
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{args.label_file} is missing columns: {missing}")
    base = pd.read_parquet(args.label_file, columns=sorted(required))
    if base.duplicated(["person_id", "split"]).any() or base["person_id"].duplicated().any():
        raise ValueError("Label file contains duplicate patients")
    for column in ("index_date", "first_activity_date", "last_activity_date"):
        base[column] = pd.to_datetime(base[column], errors="coerce")
    unexpected = base["index_date"].notna() & base["index_date"].ne(index_date)
    if unexpected.any():
        raise ValueError(f"Label rows outside index date: {int(unexpected.sum()):,}")
    if not set(base["split"].dropna().unique()).issubset(set(SPLITS)):
        raise ValueError("Unexpected split values in label file")
    return base


def required_phenotypes(config):
    values = set(config["source_a"]) | set(config["target"])
    values |= {value for value in config["source_b"] if value}
    return sorted(values)


def load_first_dates(args, phenotypes):
    cache = pd.read_parquet(
        args.survival_cache,
        columns=["phenotype", "person_id", "first_phenotype_date"],
    )
    cache = cache.loc[cache["phenotype"].isin(phenotypes)].copy()
    cache["first_phenotype_date"] = pd.to_datetime(
        cache["first_phenotype_date"], errors="coerce"
    )
    cache = cache.dropna(subset=["person_id", "first_phenotype_date"])
    duplicates = cache.duplicated(["phenotype", "person_id"])
    if duplicates.any():
        raise ValueError(
            f"Survival cache has {int(duplicates.sum()):,} duplicate phenotype/person rows"
        )
    missing = sorted(set(phenotypes) - set(cache["phenotype"].unique()))
    if missing:
        raise ValueError(f"Survival cache has no rows for: {missing}")
    wide = cache.pivot(index="person_id", columns="phenotype", values="first_phenotype_date")
    wide.columns = [f"first__{column}" for column in wide.columns]
    return wide.reset_index()


def build_token_mapping(config, concept_map, registry):
    phenotypes = required_phenotypes(config)
    known = set(concept_map["phenotype"].astype(str))
    missing = sorted(set(phenotypes) - known)
    if missing:
        raise ValueError(f"Concept map has no reviewed group for: {missing}")
    token_lookup = {
        str(row.token_key): int(row.token_id)
        for row in registry.itertuples(index=False)
        if str(row.token_key)
    }
    rows = []
    for phenotype in phenotypes:
        concept_ids = sorted(
            concept_map.loc[
                concept_map["phenotype"].eq(phenotype), "condition_concept_id"
            ]
            .astype(np.int64)
            .unique()
            .tolist()
        )
        keys = [f"DX:{value}" for value in concept_ids]
        stored = sorted({token_lookup[key] for key in keys if key in token_lookup})
        rows.append(
            {
                "phenotype": phenotype,
                "reviewed_concepts": len(concept_ids),
                "matched_registry_tokens": len(stored),
                "concept_to_token_coverage": len(stored) / len(concept_ids) if concept_ids else np.nan,
                "stored_token_ids": "|".join(str(value) for value in stored),
                "model_token_ids": "|".join(str(value + 1) for value in stored),
                "token_keys": "|".join(key for key in keys if key in token_lookup),
            }
        )
    return pd.DataFrame(rows)


def recency_labels(cuts):
    first, second, third = cuts
    return [
        f"0001_{first}d",
        f"{first + 1}_{second}d",
        f"{second + 1}_{third}d",
        f"gt_{third}d",
    ]


def assign_recency(days, cuts):
    labels = recency_labels(cuts)
    return pd.cut(
        days,
        bins=[0, cuts[0], cuts[1], cuts[2], np.inf],
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("string")


def build_pathway_patients(base, dates, deaths, item, index_date, db_end, cuts):
    source_a_col = f"first__{item.source_a}"
    target_col = f"first__{item.target}"
    columns = ["person_id", source_a_col, target_col]
    source_b_col = None
    if item.source_requirement == "both":
        source_b_col = f"first__{item.source_b}"
        columns.append(source_b_col)
    task = base.merge(dates[columns], on="person_id", how="left", validate="one_to_one")
    task = task.merge(deaths, on="person_id", how="left", validate="one_to_one")
    required_source = task[source_a_col].notna() & (task[source_a_col] < index_date)
    if source_b_col:
        required_source &= task[source_b_col].notna() & (task[source_b_col] < index_date)
    target_prior = task[target_col].notna() & (task[target_col] <= index_date)
    death_prior = task["death_date"].notna() & (task["death_date"] <= index_date)
    washout = task["has_pre_index_washout"].fillna(False).astype(bool)
    task = task.loc[required_source & ~target_prior & ~death_prior & washout].copy()

    task["source_a_date"] = task[source_a_col]
    if source_b_col:
        task["source_b_date"] = task[source_b_col]
        task["pathway_anchor_date"] = task[["source_a_date", "source_b_date"]].max(axis=1)
        task["source_gap_days"] = (
            task["source_a_date"] - task["source_b_date"]
        ).abs().dt.days.astype("Int64")
        task["source_order"] = np.select(
            [
                task["source_a_date"] < task["source_b_date"],
                task["source_b_date"] < task["source_a_date"],
            ],
            [
                f"{item.source_a}_before_{item.source_b}",
                f"{item.source_b}_before_{item.source_a}",
            ],
            default="same_day",
        )
    else:
        task["source_b_date"] = pd.NaT
        task["pathway_anchor_date"] = task["source_a_date"]
        task["source_gap_days"] = pd.Series(pd.NA, index=task.index, dtype="Int64")
        task["source_order"] = "single_source"
    task["source_recency_days"] = (
        index_date - task["pathway_anchor_date"]
    ).dt.days.astype(np.int32)
    if (task["source_recency_days"] <= 0).any():
        raise RuntimeError(f"Nonpositive source recency for pathway={item.pathway_id}")
    task["source_recency_stratum"] = assign_recency(task["source_recency_days"], cuts)

    horizon_end = index_date + pd.DateOffset(years=5)
    censor_limit = min(horizon_end, db_end)
    base_censor = task["last_activity_date"].where(
        task["last_activity_date"].notna(), censor_limit
    )
    base_censor = base_censor.clip(upper=censor_limit)
    target_possible = (
        task[target_col].notna()
        & (task[target_col] > index_date)
        & (task[target_col] <= base_censor)
    )
    death_possible = (
        task["death_date"].notna()
        & (task["death_date"] > index_date)
        & (task["death_date"] <= base_censor)
    )
    outcome = target_possible & (
        ~death_possible | (task[target_col] <= task["death_date"])
    )
    competing = death_possible & (
        ~target_possible | (task["death_date"] < task[target_col])
    )
    endpoint = base_censor.copy()
    endpoint = endpoint.where(~outcome, task[target_col])
    endpoint = endpoint.where(~competing, task["death_date"])
    task["duration_days"] = (endpoint - index_date).dt.days.astype("float64")
    task["event_type"] = np.select([outcome, competing], [1, 2], default=0).astype(np.int8)
    task = task.loc[task["duration_days"] > 0].copy()
    task["duration_days"] = task["duration_days"].astype(np.int32)
    task["pathway_id"] = item.pathway_id
    task["source_a"] = item.source_a
    task["source_b"] = item.source_b
    task["target"] = item.target
    task["definition_status"] = item.definition_status
    keep = [
        "pathway_id",
        "person_id",
        "split",
        "source_a",
        "source_b",
        "target",
        "definition_status",
        "source_a_date",
        "source_b_date",
        "pathway_anchor_date",
        "source_recency_days",
        "source_recency_stratum",
        "source_order",
        "source_gap_days",
        "duration_days",
        "event_type",
        "first_activity_date",
        "last_activity_date",
    ]
    return task[keep].sort_values(["split", "person_id"]).reset_index(drop=True)


def cumulative_incidence_curve(frame, max_day):
    days = np.arange(max_day + 1, dtype=np.int32)
    if frame.empty:
        return pd.DataFrame(
            {
                "day": days,
                "at_risk": 0,
                "target_events_on_day": 0,
                "competing_deaths_on_day": 0,
                "censored_on_day": 0,
                "observed_cumulative_incidence": np.nan,
                "event_free_survival": np.nan,
            }
        )
    duration = frame["duration_days"].to_numpy(dtype=np.int64)
    event_type = frame["event_type"].to_numpy(dtype=np.int8)
    outcome = np.bincount(duration[event_type == 1], minlength=max_day + 1)[: max_day + 1]
    death = np.bincount(duration[event_type == 2], minlength=max_day + 1)[: max_day + 1]
    censor = np.bincount(duration[event_type == 0], minlength=max_day + 1)[: max_day + 1]
    risk = len(frame)
    survival = 1.0
    incidence = 0.0
    rows = []
    for day in days:
        target_events = int(outcome[day])
        competing_deaths = int(death[day])
        censored = int(censor[day])
        before = risk
        if before > 0:
            incidence += survival * target_events / before
            survival *= 1.0 - (target_events + competing_deaths) / before
        rows.append(
            {
                "day": int(day),
                "at_risk": int(before),
                "target_events_on_day": target_events,
                "competing_deaths_on_day": competing_deaths,
                "censored_on_day": censored,
                "observed_cumulative_incidence": float(incidence),
                "event_free_survival": float(survival),
            }
        )
        risk -= target_events + competing_deaths + censored
        if risk < 0:
            raise RuntimeError("Risk set became negative")
    return pd.DataFrame(rows)


def one_group_summary(frame, pathway_id, split, group_type, group_value, index_date):
    result = {
        "pathway_id": pathway_id,
        "split": split,
        "group_type": group_type,
        "group_value": group_value,
        "eligible_patients": int(len(frame)),
        "competing_deaths_5y": int(frame["event_type"].eq(2).sum()),
        "censored_5y": int(frame["event_type"].eq(0).sum()),
        "median_followup_days": float(frame["duration_days"].median()) if len(frame) else np.nan,
        "source_recency_days_median": float(frame["source_recency_days"].median()) if len(frame) else np.nan,
        "source_recency_days_p10": float(frame["source_recency_days"].quantile(0.10)) if len(frame) else np.nan,
        "source_recency_days_p90": float(frame["source_recency_days"].quantile(0.90)) if len(frame) else np.nan,
        "source_gap_days_median": float(frame["source_gap_days"].median())
        if len(frame) and frame["source_gap_days"].notna().any()
        else np.nan,
    }
    curve = cumulative_incidence_curve(frame, int(((index_date + pd.DateOffset(years=5)) - index_date).days))
    for years in LANDMARK_YEARS:
        day = int(((index_date + pd.DateOffset(years=years)) - index_date).days)
        result[f"target_events_{years}y"] = int(
            (frame["event_type"].eq(1) & (frame["duration_days"] <= day)).sum()
        )
        result[f"observed_cumulative_incidence_{years}y"] = float(
            curve.loc[curve["day"].eq(day), "observed_cumulative_incidence"].iloc[0]
        )
    return result, curve


def build_summaries(patients, config, index_date, recency_strata):
    summary_rows = []
    curve_parts = []
    order_rows = []
    for item in config.itertuples(index=False):
        pathway = patients.loc[patients["pathway_id"].eq(item.pathway_id)]
        for split in (*SPLITS, "all"):
            split_frame = pathway if split == "all" else pathway.loc[pathway["split"].eq(split)]
            row, curve = one_group_summary(
                split_frame, item.pathway_id, split, "overall", "all", index_date
            )
            summary_rows.append(row)
            if split in {"test", "all"}:
                curve.insert(0, "group_value", "all")
                curve.insert(0, "group_type", "overall")
                curve.insert(0, "split", split)
                curve.insert(0, "pathway_id", item.pathway_id)
                curve_parts.append(curve)
            for stratum in recency_strata:
                sub = split_frame.loc[split_frame["source_recency_stratum"].eq(stratum)]
                row, curve = one_group_summary(
                    sub, item.pathway_id, split, "source_recency", stratum, index_date
                )
                summary_rows.append(row)
                if split in {"test", "all"}:
                    curve.insert(0, "group_value", stratum)
                    curve.insert(0, "group_type", "source_recency")
                    curve.insert(0, "split", split)
                    curve.insert(0, "pathway_id", item.pathway_id)
                    curve_parts.append(curve)
            if item.source_requirement == "both":
                for order, sub in split_frame.groupby("source_order", sort=True):
                    row, _ = one_group_summary(
                        sub, item.pathway_id, split, "source_order", str(order), index_date
                    )
                    order_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    curves = pd.concat(curve_parts, ignore_index=True)
    orders = pd.DataFrame(order_rows)
    return summary, curves, orders


def build_screening(config, summary, token_mapping, args):
    token_lookup = token_mapping.set_index("phenotype").to_dict("index")
    rows = []
    for item in config.itertuples(index=False):
        overall = summary.loc[
            summary["pathway_id"].eq(item.pathway_id)
            & summary["split"].eq("test")
            & summary["group_type"].eq("overall")
        ].iloc[0]
        strata = summary.loc[
            summary["pathway_id"].eq(item.pathway_id)
            & summary["split"].eq("test")
            & summary["group_type"].eq("source_recency")
        ]
        populated = int(
            (strata["eligible_patients"] >= args.min_recency_stratum_patients).sum()
        )
        phenotype_roles = [item.source_a, item.target]
        if item.source_b:
            phenotype_roles.append(item.source_b)
        unmapped = [
            name
            for name in phenotype_roles
            if int(token_lookup[name]["matched_registry_tokens"]) == 0
        ]
        reasons = []
        if int(overall["eligible_patients"]) < args.min_test_eligible:
            reasons.append(f"test eligible < {args.min_test_eligible}")
        if int(overall["target_events_5y"]) < args.min_test_events_5y:
            reasons.append(f"test 5y target events < {args.min_test_events_5y}")
        if populated < args.min_populated_recency_strata:
            reasons.append(
                f"recency strata with >= {args.min_recency_stratum_patients} patients "
                f"< {args.min_populated_recency_strata}"
            )
        if unmapped:
            reasons.append("no registry target for " + ",".join(unmapped))
        if item.definition_status != "reviewed_groups":
            reasons.append("target/source concept review required")
        rows.append(
            {
                "pathway_id": item.pathway_id,
                "source_a": item.source_a,
                "source_b": item.source_b,
                "target": item.target,
                "definition_status": item.definition_status,
                "test_eligible_patients": int(overall["eligible_patients"]),
                "test_target_events_1y": int(overall["target_events_1y"]),
                "test_target_events_3y": int(overall["target_events_3y"]),
                "test_target_events_5y": int(overall["target_events_5y"]),
                "test_cumulative_incidence_1y": overall["observed_cumulative_incidence_1y"],
                "test_cumulative_incidence_3y": overall["observed_cumulative_incidence_3y"],
                "test_cumulative_incidence_5y": overall["observed_cumulative_incidence_5y"],
                "test_populated_recency_strata": populated,
                "screening_status": "count_ready" if not reasons else "needs_review",
                "screening_reasons": " | ".join(reasons),
                "clinical_question": item.clinical_question,
                "interpretation_limit": item.interpretation_limit,
                "rollout_hit_rate_status": "not_measured_cpu_audit_only",
            }
        )
    return pd.DataFrame(rows)


def build_return_summary(screening, summary, orders, output_dir):
    test_recency = summary.loc[
        summary["split"].eq("test") & summary["group_type"].eq("source_recency"),
        [
            "pathway_id",
            "group_value",
            "eligible_patients",
            "target_events_1y",
            "target_events_3y",
            "target_events_5y",
            "observed_cumulative_incidence_5y",
        ],
    ]
    lines = [
        "## STATUS",
        "COMPLETE_TASK30_CPU_PATHWAY_FEASIBILITY",
        "## PATHWAY_SCREENING",
        screening.to_csv(index=False).rstrip(),
        "## TEST_RECENCY_COUNTS",
        test_recency.to_csv(index=False).rstrip(),
    ]
    if not orders.empty:
        test_orders = orders.loc[orders["split"].eq("test")]
        lines.extend(["## TEST_SOURCE_ORDER_COUNTS", test_orders.to_csv(index=False).rstrip()])
    lines.extend(
        [
            "## NOT_RUN",
            "NO_TORCH_NO_CHECKPOINT_NO_MODEL_FIT_NO_ROLLOUT",
            "## OUTPUT_DIR",
            str(output_dir),
        ]
    )
    text = "\n".join(lines) + "\n"
    (Path(output_dir) / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    index_date = pd.Timestamp("2018-01-01")
    base = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "split": ["test"] * 5,
            "index_date": [index_date] * 5,
            "first_activity_date": [pd.Timestamp("2014-01-01")] * 5,
            "last_activity_date": [pd.Timestamp("2023-01-01")] * 5,
            "has_pre_index_washout": [True] * 5,
        }
    )
    dates = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "first__source_a": pd.to_datetime(
                ["2017-12-01", "2017-01-01", "2016-01-01", "2015-01-01", "2016-01-01"]
            ),
            "first__source_b": pd.to_datetime(
                ["2017-11-01", "2016-01-01", "2017-01-01", "2015-01-01", "2017-01-01"]
            ),
            "first__target": pd.to_datetime(
                ["2018-02-01", None, "2017-01-01", None, None]
            ),
        }
    )
    deaths = pd.DataFrame(
        {"person_id": [2], "death_date": pd.to_datetime(["2018-03-01"])}
    )
    item = SimpleNamespace(
        pathway_id="test_path",
        source_a="source_a",
        source_b="source_b",
        source_requirement="both",
        target="target",
        definition_status="reviewed_groups",
    )
    task = build_pathway_patients(
        base,
        dates,
        deaths,
        item,
        index_date,
        pd.Timestamp("2025-02-05"),
        [180, 730, 1826],
    )
    if set(task["person_id"]) != {1, 2, 4, 5}:
        raise AssertionError("Prior-target exclusion failed")
    if int(task["event_type"].eq(1).sum()) != 1 or int(task["event_type"].eq(2).sum()) != 1:
        raise AssertionError("Outcome/death assignment failed")
    if set(task["source_order"]) != {
        "source_b_before_source_a",
        "source_a_before_source_b",
        "same_day",
    }:
        raise AssertionError("Two-source ordering failed")
    curve = cumulative_incidence_curve(task, 365)
    if not np.isclose(curve.iloc[-1]["observed_cumulative_incidence"], 0.25):
        raise AssertionError("Competing-risk cumulative incidence failed")
    log("[SELF-TEST PASS] eligibility, ordering, recency, and cumulative incidence")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    if args.resume_from_raw and args.overwrite:
        raise ValueError("--resume-from-raw and --overwrite are mutually exclusive")
    if sorted(args.recency_cut_days) != list(args.recency_cut_days):
        raise ValueError("--recency-cut-days must be strictly increasing")
    if args.recency_cut_days[0] <= 0 or len(set(args.recency_cut_days)) != 3:
        raise ValueError("--recency-cut-days must contain three distinct positive values")
    for path in (
        args.config_file,
        args.label_file,
        args.concept_map,
        args.survival_cache,
        args.data_dir / "token_registry.csv",
    ):
        require_file(path)

    started = time.time()
    config, index_date, washout_years = load_config(args.config_file)
    run_config = fingerprint(
        args, sha256_file(args.config_file), index_date, washout_years
    )
    prepare_output(args, run_config)
    raw_patients_path = args.output_dir / "raw" / "pathway_patient_level.parquet"

    concept_map = pd.read_csv(args.concept_map)
    required_concept = {"phenotype", "condition_concept_id"}
    missing = sorted(required_concept - set(concept_map.columns))
    if missing:
        raise ValueError(f"Concept map missing columns: {missing}")
    concept_map["condition_concept_id"] = pd.to_numeric(
        concept_map["condition_concept_id"], errors="raise"
    ).astype(np.int64)
    registry = pd.read_csv(args.data_dir / "token_registry.csv", dtype={"token_key": str})
    missing = sorted({"token_id", "token_key"} - set(registry.columns))
    if missing:
        raise ValueError(f"Token registry missing columns: {missing}")
    registry["token_id"] = pd.to_numeric(registry["token_id"], errors="raise").astype(np.int64)
    token_mapping = build_token_mapping(config, concept_map, registry)
    token_mapping.to_csv(args.output_dir / "pathway_phenotype_token_mapping.csv", index=False)

    if args.resume_from_raw and raw_patients_path.is_file():
        patients = pd.read_parquet(raw_patients_path)
        for column in (
            "source_a_date",
            "source_b_date",
            "pathway_anchor_date",
            "first_activity_date",
            "last_activity_date",
        ):
            patients[column] = pd.to_datetime(patients[column], errors="coerce")
        log(f"[RESUME] pathway patient rows={len(patients):,}")
    else:
        base = load_base(args, index_date)
        dates = load_first_dates(args, required_phenotypes(config))
        deaths = load_death_dates(args)
        parts = []
        for item in config.itertuples(index=False):
            part_started = time.time()
            part = build_pathway_patients(
                base,
                dates,
                deaths,
                item,
                index_date,
                pd.Timestamp(args.db_end_date),
                args.recency_cut_days,
            )
            parts.append(part)
            log(
                f"[PATHWAY] {item.pathway_id} patients={len(part):,} "
                f"seconds={time.time() - part_started:,.1f}"
            )
        patients = pd.concat(parts, ignore_index=True)
        atomic_to_parquet(patients, raw_patients_path)
        log(f"[RAW SAVED] {raw_patients_path}")

    unexpected = set(patients["pathway_id"].unique()) - set(config["pathway_id"])
    if unexpected:
        raise RuntimeError(f"Unexpected pathway IDs in raw output: {unexpected}")
    summary, curves, orders = build_summaries(
        patients, config, index_date, recency_labels(args.recency_cut_days)
    )
    summary.to_csv(args.output_dir / "pathway_group_horizon_summary.csv", index=False)
    curves.to_csv(args.output_dir / "observed_pathway_incidence_curves.csv", index=False)
    orders.to_csv(args.output_dir / "two_source_order_summary.csv", index=False)
    screening = build_screening(config, summary, token_mapping, args)
    screening.to_csv(args.output_dir / "pathway_feasibility_summary.csv", index=False)

    manifest = {
        **run_config,
        "status": "COMPLETE_TASK30_CPU_PATHWAY_FEASIBILITY",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "pathways": int(len(config)),
        "patient_pathway_rows": int(len(patients)),
        "curve_method": "Aalen-Johansen cumulative incidence with all-cause death as a competing event",
        "source_time": "days from first recorded source diagnosis, or later of two first diagnoses, to fixed index date",
        "explicitly_not_run": [
            "torch import",
            "GPU use",
            "checkpoint loading",
            "model fitting",
            "future generation",
        ],
        "outputs": {
            "primary": str(args.output_dir / "pathway_feasibility_summary.csv"),
            "group_summary": str(args.output_dir / "pathway_group_horizon_summary.csv"),
            "observed_curves": str(args.output_dir / "observed_pathway_incidence_curves.csv"),
            "source_order": str(args.output_dir / "two_source_order_summary.csv"),
            "token_mapping": str(args.output_dir / "pathway_phenotype_token_mapping.csv"),
            "raw_patient_level": str(raw_patients_path),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(screening, summary, orders, args.output_dir)
    log("[COMPLETE] Task 30 CPU pathway feasibility audit finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
