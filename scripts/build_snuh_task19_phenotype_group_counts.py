#!/usr/bin/env python3
"""Count Task 19 phenotype groups with patient-level de-duplication.

The previous candidate screen reports concept-level counts. This script takes a
reviewed phenotype group plan, maps multiple condition concepts to one
phenotype, and recomputes prior/incident counts per patient from the source
condition_occurrence table.
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
DEFAULT_INPUT = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "phenotype_candidates"
    / "phenotype_group_plan_v0.csv"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "phenotype_group_counts"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task19_phenotype_group_counts"
DEFAULT_INDEX_DATES = ["2015-01-01", "2018-01-01", "2020-01-01"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--group-plan", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dates", nargs="+", default=DEFAULT_INDEX_DATES)
    parser.add_argument("--db-end-date", default="2025-02-05")
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
            if not concept_id:
                continue
            rows.append(
                {
                    "phenotype": phenotype,
                    "condition_concept_id": int(concept_id),
                }
            )
    mapping = pd.DataFrame(rows).drop_duplicates()
    if mapping.empty:
        raise ValueError(f"{path} produced no phenotype concept rows")
    return plan, mapping


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
    execute(
        conn,
        "CREATE INDEX ON tmp_task19_group_concept(phenotype)",
        label="Index phenotype concept map by phenotype",
    )


def fetch_group_counts(conn, args, index_date: str):
    return query_df(
        conn,
        sql.SQL(
            """
            WITH eligible AS (
                SELECT
                    p.person_id,
                    p.split
                FROM tmp_task19_patient_map p
                JOIN {}.person person USING(person_id)
                WHERE make_date(
                    person.year_of_birth,
                    CASE
                        WHEN person.month_of_birth BETWEEN 1 AND 12
                        THEN person.month_of_birth
                        ELSE 7
                    END,
                    CASE
                        WHEN person.day_of_birth BETWEEN 1 AND 28
                        THEN person.day_of_birth
                        ELSE 1
                    END
                ) <= %s::date
            ),
            eligible_count AS (
                SELECT split, COUNT(*)::bigint AS eligible_patients
                FROM eligible
                GROUP BY split
            ),
            phenotype_base AS (
                SELECT DISTINCT phenotype
                FROM tmp_task19_group_concept
            ),
            events AS (
                SELECT
                    g.phenotype,
                    e.person_id,
                    p.split,
                    MIN(e.condition_start_date)
                        FILTER (WHERE e.condition_start_date < %s::date)
                        AS first_prior_date,
                    MIN(e.condition_start_date)
                        FILTER (
                            WHERE e.condition_start_date >= %s::date
                              AND e.condition_start_date <= %s::date
                        ) AS first_future_date
                FROM {}.condition_occurrence e
                JOIN eligible p USING(person_id)
                JOIN tmp_task19_group_concept g
                  ON g.condition_concept_id = e.condition_concept_id
                WHERE e.condition_start_date IS NOT NULL
                  AND e.condition_start_date <= LEAST(
                      %s::date,
                      %s::date + INTERVAL '5 years'
                  )
                GROUP BY g.phenotype, e.person_id, p.split
            ),
            base AS (
                SELECT
                    b.phenotype,
                    e.split,
                    e.eligible_patients
                FROM phenotype_base b
                CROSS JOIN eligible_count e
            ),
            agg AS (
                SELECT
                    e.phenotype,
                    e.split,
                    COUNT(*) FILTER (WHERE first_prior_date IS NOT NULL)::bigint
                        AS prior_patients,
                    COUNT(*) FILTER (
                        WHERE first_prior_date IS NULL
                          AND first_future_date >= %s::date
                          AND first_future_date < %s::date + INTERVAL '1 year'
                    )::bigint AS incident_1y,
                    COUNT(*) FILTER (
                        WHERE first_prior_date IS NULL
                          AND first_future_date >= %s::date
                          AND first_future_date < %s::date + INTERVAL '3 years'
                    )::bigint AS incident_3y,
                    COUNT(*) FILTER (
                        WHERE first_prior_date IS NULL
                          AND first_future_date >= %s::date
                          AND first_future_date < LEAST(
                              %s::date,
                              %s::date + INTERVAL '5 years'
                          )
                    )::bigint AS incident_5y
                FROM events e
                GROUP BY e.phenotype, e.split
            )
            SELECT
                %s::date AS index_date,
                b.phenotype,
                b.split,
                b.eligible_patients,
                COALESCE(a.prior_patients, 0)::bigint AS prior_patients,
                (b.eligible_patients - COALESCE(a.prior_patients, 0))::bigint
                    AS at_risk_patients,
                COALESCE(a.incident_1y, 0)::bigint AS incident_1y,
                COALESCE(a.incident_3y, 0)::bigint AS incident_3y,
                COALESCE(a.incident_5y, 0)::bigint AS incident_5y
            FROM base b
            LEFT JOIN agg a
              ON a.phenotype = b.phenotype
             AND a.split = b.split
            ORDER BY b.phenotype, b.split
            """
        ).format(sql.Identifier(args.schema), sql.Identifier(args.schema)),
        params=(
            index_date,
            index_date,
            index_date,
            args.db_end_date,
            args.db_end_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
            args.db_end_date,
            index_date,
            index_date,
        ),
        label=f"Fetch group counts for index_date={index_date}",
    )


def add_rates(frame):
    frame = frame.copy()
    for column in [
        "eligible_patients",
        "prior_patients",
        "at_risk_patients",
        "incident_1y",
        "incident_3y",
        "incident_5y",
    ]:
        frame[column] = pd.to_numeric(frame[column])
    frame["prior_prevalence"] = (
        frame["prior_patients"] / frame["eligible_patients"].replace(0, np.nan)
    )
    for horizon in ["1y", "3y", "5y"]:
        count_col = f"incident_{horizon}"
        rate_col = f"incident_{horizon}_rate"
        frame[rate_col] = frame[count_col] / frame["at_risk_patients"].replace(0, np.nan)
    return frame


def summarize_counts(split_counts, group_plan):
    total = (
        split_counts.groupby(["index_date", "phenotype"], as_index=False)
        .agg(
            eligible_patients=("eligible_patients", "sum"),
            prior_patients=("prior_patients", "sum"),
            at_risk_patients=("at_risk_patients", "sum"),
            incident_1y=("incident_1y", "sum"),
            incident_3y=("incident_3y", "sum"),
            incident_5y=("incident_5y", "sum"),
        )
    )
    total = add_rates(total)
    plan_cols = [c for c in ["phenotype", "concept_ids", "concept_names", "n_concepts"] if c in group_plan.columns]
    total = total.merge(group_plan[plan_cols], on="phenotype", how="left")
    total["usable_1y"] = total["incident_1y"] >= 200
    total["usable_3y"] = total["incident_3y"] >= 500
    total["usable_5y"] = total["incident_5y"] >= 800
    total["suggested_priority"] = np.select(
        [
            total["usable_1y"] & total["usable_3y"] & total["usable_5y"],
            total["usable_3y"] | total["usable_5y"],
            total["incident_5y"] >= 200,
        ],
        ["high", "medium", "low"],
        default="insufficient",
    )
    priority_rank = {"high": 0, "medium": 1, "low": 2, "insufficient": 3}
    total["priority_rank"] = total["suggested_priority"].map(priority_rank)
    return total.sort_values(
        ["index_date", "priority_rank", "incident_5y", "phenotype"],
        ascending=[True, True, False, True],
    ).drop(columns=["priority_rank"])


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
        counts = []
        for index_date in args.index_dates:
            counts.append(fetch_group_counts(conn, args, index_date))

    split_counts = pd.concat(counts, ignore_index=True)
    split_counts = add_rates(split_counts)
    summary = summarize_counts(split_counts, group_plan)

    group_map.to_csv(args.output_dir / "phenotype_group_concept_map.csv", index=False)
    split_counts.to_csv(args.output_dir / "phenotype_group_split_counts.csv", index=False)
    summary.to_csv(args.output_dir / "phenotype_group_summary.csv", index=False)
    manifest = {
        "data_dir": str(args.data_dir),
        "group_plan": str(args.group_plan),
        "output_dir": str(args.output_dir),
        "index_dates": args.index_dates,
        "db_end_date": args.db_end_date,
        "phenotype_groups": int(group_plan["phenotype"].nunique()),
        "phenotype_concept_rows": int(len(group_map)),
        "outputs": {
            "concept_map": str(args.output_dir / "phenotype_group_concept_map.csv"),
            "split_counts": str(args.output_dir / "phenotype_group_split_counts.csv"),
            "summary": str(args.output_dir / "phenotype_group_summary.csv"),
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
