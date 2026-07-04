#!/usr/bin/env python3
"""Build patient-level disease-risk labels for Task 19.

This script turns a reviewed phenotype group plan into fixed-index disease
onset labels. It uses only source dates up to the configured database end date:

* patients must have enough pre-index observation for the strict label set;
* patients must have enough post-index observation for each horizon;
* patients with a phenotype before the index date are excluded for that
  phenotype;
* outcomes are incident phenotype diagnoses within 1/3/5 years.
"""

from __future__ import annotations

import argparse
import getpass
import json
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


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_GROUP_PLAN = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "phenotype_candidates"
    / "phenotype_group_plan_v0.csv"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task19_patient_labels"
DEFAULT_INDEX_DATES = ["2015-01-01", "2018-01-01", "2020-01-01"]
HORIZONS = {"1y": 1, "3y": 3, "5y": 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--group-plan", type=Path, default=DEFAULT_GROUP_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dates", nargs="+", default=DEFAULT_INDEX_DATES)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--washout-years", type=int, default=2)
    parser.add_argument(
        "--wide-labels",
        action="store_true",
        help="Write patient-level wide parquet files. Summaries are always written.",
    )
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument(
        "--statement-timeout",
        default="0",
        help="PostgreSQL statement timeout. Use 0 to disable timeout.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_psycopg():
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required on the Pod. Install with "
            "`python -m pip install \"psycopg[binary]>=3\"`."
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
        cur.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (args.statement_timeout,),
        )
    return conn


def execute(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params)
    conn.commit()
    if label:
        log(f"[DONE] {label} {time.time() - started:,.1f}s")


def query_df(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if label:
        log(f"[DONE] {label} rows={len(frame):,} {time.time() - started:,.1f}s")
    return frame


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def load_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["person_id", "split"])
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["split"] = frame["split"].astype(str)
    return frame


def load_group_plan(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    plan = pd.read_csv(path)
    required = {"phenotype", "concept_ids"}
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    rows = []
    for row in plan.itertuples(index=False):
        phenotype = str(getattr(row, "phenotype"))
        concept_ids = str(getattr(row, "concept_ids")).split("|")
        for concept_id in concept_ids:
            concept_id = concept_id.strip()
            if concept_id:
                rows.append(
                    {
                        "phenotype": phenotype,
                        "condition_concept_id": int(concept_id),
                    }
                )
    group_map = pd.DataFrame(rows).drop_duplicates()
    if group_map.empty:
        raise ValueError(f"{path} produced no phenotype concept rows")
    return plan, group_map


def upload_patient_map(conn, patient_map):
    execute(conn, "DROP TABLE IF EXISTS tmp_task19_patient_map")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_task19_patient_map (
            person_id bigint,
            split text
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    rows = [(int(person_id), str(split)) for person_id, split in patient_map.itertuples(index=False)]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO tmp_task19_patient_map VALUES (%s,%s)", rows)
    conn.commit()
    log(f"uploaded patient map: {len(rows):,}")
    execute(
        conn,
        "CREATE INDEX ON tmp_task19_patient_map(person_id)",
        label="Index patient map",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_task19_patient_map(split, person_id)",
        label="Index patient split",
    )


def upload_group_map(conn, group_map):
    execute(conn, "DROP TABLE IF EXISTS tmp_task19_group_concept")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_task19_group_concept (
            phenotype text,
            condition_concept_id bigint
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    rows = [
        (str(phenotype), int(condition_concept_id))
        for phenotype, condition_concept_id in group_map.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO tmp_task19_group_concept VALUES (%s,%s)", rows)
    conn.commit()
    log(f"uploaded phenotype concept rows: {len(rows):,}")
    execute(
        conn,
        "CREATE INDEX ON tmp_task19_group_concept(condition_concept_id)",
        label="Index phenotype concept map by concept",
    )


def fetch_patient_profile(conn, args):
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT
                p.person_id::bigint,
                p.split,
                person.year_of_birth::int,
                person.month_of_birth::int,
                person.day_of_birth::int,
                person.gender_concept_id::bigint
            FROM tmp_task19_patient_map p
            JOIN {}.person person USING(person_id)
            ORDER BY p.person_id
            """
        ).format(sql.Identifier(args.schema)),
        label="Fetch patient profile",
    )


def fetch_patient_activity(conn, args):
    return query_df(
        conn,
        sql.SQL(
            """
            WITH events AS (
                SELECT
                    c.person_id,
                    MIN(c.condition_start_date)::date AS first_date,
                    MAX(c.condition_start_date)::date AS last_date
                FROM {}.condition_occurrence c
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE c.condition_start_date IS NOT NULL
                GROUP BY c.person_id

                UNION ALL

                SELECT
                    d.person_id,
                    MIN(d.drug_exposure_start_date)::date AS first_date,
                    MAX(d.drug_exposure_start_date)::date AS last_date
                FROM {}.drug_exposure d
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE d.drug_exposure_start_date IS NOT NULL
                GROUP BY d.person_id

                UNION ALL

                SELECT
                    pr.person_id,
                    MIN(pr.procedure_date)::date AS first_date,
                    MAX(pr.procedure_date)::date AS last_date
                FROM {}.procedure_occurrence pr
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE pr.procedure_date IS NOT NULL
                GROUP BY pr.person_id
            )
            SELECT
                person_id::bigint,
                MIN(first_date)::date AS first_activity_date,
                MAX(last_date)::date AS last_activity_date
            FROM events
            GROUP BY person_id
            ORDER BY person_id
            """
        ).format(
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
        ),
        label="Fetch patient first/last clinical activity",
    )


def fetch_phenotype_first_dates(conn, args):
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT
                g.phenotype,
                c.person_id::bigint,
                MIN(c.condition_start_date)::date AS first_phenotype_date
            FROM {}.condition_occurrence c
            JOIN tmp_task19_patient_map p USING(person_id)
            JOIN tmp_task19_group_concept g
              ON g.condition_concept_id = c.condition_concept_id
            WHERE c.condition_start_date IS NOT NULL
              AND c.condition_start_date <= %s::date
            GROUP BY g.phenotype, c.person_id
            ORDER BY g.phenotype, c.person_id
            """
        ).format(sql.Identifier(args.schema)),
        params=(args.db_end_date,),
        label="Fetch first diagnosis date per phenotype",
    )


def coerce_dates(frame, columns):
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def add_patient_base(profile, activity, index_date: pd.Timestamp, washout_years: int):
    base = profile.merge(activity, on="person_id", how="left")
    base = coerce_dates(base, ["first_activity_date", "last_activity_date"])

    month = pd.to_numeric(base["month_of_birth"], errors="coerce").fillna(7).astype(int)
    day = pd.to_numeric(base["day_of_birth"], errors="coerce").fillna(1).astype(int)
    month = month.where(month.between(1, 12), 7)
    day = day.where(day.between(1, 28), 1)
    birth = pd.to_datetime(
        {
            "year": pd.to_numeric(base["year_of_birth"], errors="coerce").astype("Int64"),
            "month": month,
            "day": day,
        },
        errors="coerce",
    )

    base["index_date"] = index_date.strftime("%Y-%m-%d")
    base["birth_date_proxy"] = birth
    base["age_at_index"] = ((index_date - birth).dt.days / 365.25).astype("float32")
    base["has_pre_index_washout"] = (
        base["first_activity_date"] <= index_date - pd.DateOffset(years=washout_years)
    )
    for horizon, years in HORIZONS.items():
        horizon_end = index_date + pd.DateOffset(years=years)
        base[f"has_followup_{horizon}"] = base["last_activity_date"] >= horizon_end
    return base


def summarize_index(base, first_dates, group_plan, index_date: pd.Timestamp):
    summaries = []
    for row in group_plan.itertuples(index=False):
        phenotype = str(getattr(row, "phenotype"))
        phenotype_dates = first_dates.loc[
            first_dates["phenotype"].eq(phenotype),
            ["person_id", "first_phenotype_date"],
        ]
        merged = base.merge(phenotype_dates, on="person_id", how="left")
        first_date = merged["first_phenotype_date"]
        prior = first_date.notna() & (first_date < index_date)
        after_index = first_date.notna() & (first_date >= index_date)
        at_risk_base = merged["has_pre_index_washout"] & ~prior

        for horizon, years in HORIZONS.items():
            horizon_end = index_date + pd.DateOffset(years=years)
            eligible = at_risk_base & merged[f"has_followup_{horizon}"]
            label = eligible & after_index & (first_date < horizon_end)
            grouped = (
                pd.DataFrame(
                    {
                        "split": merged["split"],
                        "eligible": eligible.astype("int64"),
                        "positive": label.astype("int64"),
                    }
                )
                .groupby("split", as_index=False)
                .agg(
                    eligible_patients=("eligible", "sum"),
                    incident_patients=("positive", "sum"),
                )
            )
            grouped["index_date"] = index_date.strftime("%Y-%m-%d")
            grouped["phenotype"] = phenotype
            grouped["horizon"] = horizon
            grouped["prior_patients"] = int(prior.sum())
            grouped["pre_index_washout_patients"] = int(merged["has_pre_index_washout"].sum())
            grouped["followup_complete_patients"] = int(merged[f"has_followup_{horizon}"].sum())
            summaries.append(grouped)

    summary = pd.concat(summaries, ignore_index=True)
    total = (
        summary.groupby(["index_date", "phenotype", "horizon"], as_index=False)
        .agg(
            eligible_patients=("eligible_patients", "sum"),
            incident_patients=("incident_patients", "sum"),
            prior_patients=("prior_patients", "max"),
            pre_index_washout_patients=("pre_index_washout_patients", "max"),
            followup_complete_patients=("followup_complete_patients", "max"),
        )
    )
    total["split"] = "all"
    summary = pd.concat([summary, total], ignore_index=True)
    summary["incident_rate"] = (
        summary["incident_patients"] / summary["eligible_patients"].replace(0, np.nan)
    )
    return summary.sort_values(["index_date", "phenotype", "horizon", "split"])


def build_wide_labels(base, first_dates, group_plan, index_date: pd.Timestamp):
    labels = base[
        [
            "person_id",
            "split",
            "index_date",
            "gender_concept_id",
            "age_at_index",
            "first_activity_date",
            "last_activity_date",
            "has_pre_index_washout",
        ]
        + [f"has_followup_{horizon}" for horizon in HORIZONS]
    ].copy()

    for row in group_plan.itertuples(index=False):
        phenotype = str(getattr(row, "phenotype"))
        phenotype_dates = first_dates.loc[
            first_dates["phenotype"].eq(phenotype),
            ["person_id", "first_phenotype_date"],
        ]
        merged = base[["person_id", "has_pre_index_washout"] + [f"has_followup_{h}" for h in HORIZONS]].merge(
            phenotype_dates,
            on="person_id",
            how="left",
        )
        first_date = merged["first_phenotype_date"]
        prior = first_date.notna() & (first_date < index_date)
        after_index = first_date.notna() & (first_date >= index_date)
        at_risk_base = merged["has_pre_index_washout"] & ~prior
        labels[f"prior__{phenotype}"] = prior.astype("int8")
        for horizon, years in HORIZONS.items():
            horizon_end = index_date + pd.DateOffset(years=years)
            eligible = at_risk_base & merged[f"has_followup_{horizon}"]
            label = eligible & after_index & (first_date < horizon_end)
            labels[f"eligible_{horizon}__{phenotype}"] = eligible.astype("int8")
            labels[f"label_{horizon}__{phenotype}"] = label.astype("int8")
    return labels


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.group_plan = args.group_plan.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    patient_map = load_patient_map(args.data_dir)
    group_plan, group_map = load_group_plan(args.group_plan)
    log(f"patients={len(patient_map):,}")
    log(f"phenotype groups={group_plan['phenotype'].nunique():,}")
    log(f"phenotype concept rows={len(group_map):,}")

    with connect(args) as conn:
        upload_patient_map(conn, patient_map)
        upload_group_map(conn, group_map)
        profile = fetch_patient_profile(conn, args)
        activity = fetch_patient_activity(conn, args)
        first_dates = fetch_phenotype_first_dates(conn, args)

    profile = profile.merge(patient_map[["person_id", "split"]], on=["person_id", "split"], how="right")
    activity = coerce_dates(activity, ["first_activity_date", "last_activity_date"])
    first_dates = coerce_dates(first_dates, ["first_phenotype_date"])

    summaries = []
    for index_date_text in args.index_dates:
        index_date = pd.Timestamp(index_date_text)
        log(f"[START] Build labels for index_date={index_date_text}")
        base = add_patient_base(profile, activity, index_date, args.washout_years)
        summary = summarize_index(base, first_dates, group_plan, index_date)
        summaries.append(summary)
        safe_date = index_date_text.replace("-", "")
        base_path = args.output_dir / f"patient_index_base_{safe_date}.parquet"
        base.to_parquet(base_path, index=False)
        if args.wide_labels:
            wide = build_wide_labels(base, first_dates, group_plan, index_date)
            wide.to_parquet(
                args.output_dir / f"patient_phenotype_labels_wide_{safe_date}.parquet",
                index=False,
            )
        log(f"[DONE] Build labels for index_date={index_date_text}")

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_csv(args.output_dir / "patient_phenotype_label_summary.csv", index=False)
    group_map.to_csv(args.output_dir / "phenotype_group_concept_map.csv", index=False)
    group_plan.to_csv(args.output_dir / "phenotype_benchmark_set_v1.csv", index=False)

    manifest = {
        "data_dir": str(args.data_dir),
        "group_plan": str(args.group_plan),
        "output_dir": str(args.output_dir),
        "index_dates": args.index_dates,
        "db_end_date": args.db_end_date,
        "washout_years": args.washout_years,
        "horizons": HORIZONS,
        "wide_labels": bool(args.wide_labels),
        "phenotype_groups": int(group_plan["phenotype"].nunique()),
        "phenotype_concept_rows": int(len(group_map)),
        "outputs": {
            "summary": str(args.output_dir / "patient_phenotype_label_summary.csv"),
            "benchmark_set": str(args.output_dir / "phenotype_benchmark_set_v1.csv"),
            "concept_map": str(args.output_dir / "phenotype_group_concept_map.csv"),
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
