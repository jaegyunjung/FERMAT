#!/usr/bin/env python3
"""Build candidate phenotypes for an SNUH disease-risk benchmark.

This is the first Task 19 artifact. It does not train a model and does not
extract embeddings. It screens frequent DX concepts and reports whether each
concept has enough incident outcome counts after fixed index dates to support
1/3/5-year disease-risk prediction tasks.
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
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "phenotype_candidates"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task19_disease_risk_candidates"

DEFAULT_INDEX_DATES = ["2015-01-01", "2018-01-01", "2020-01-01"]
HORIZONS = {
    "y1": "1 year",
    "y3": "3 years",
    "y5": "5 years",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-concepts", type=int, default=500)
    parser.add_argument("--min-train-patients", type=int, default=1000)
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
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    frame["person_id"] = frame["person_id"].astype(np.int64)
    return frame


def load_candidate_concepts(data_dir: Path, max_concepts: int, min_train_patients: int):
    frequency_path = data_dir / "train_dx_frequency.parquet"
    registry_path = data_dir / "token_registry.csv"
    if not frequency_path.exists():
        raise FileNotFoundError(frequency_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    freq = pd.read_parquet(frequency_path)
    freq = freq.rename(columns={"concept_id": "condition_concept_id"})
    freq["condition_concept_id"] = freq["condition_concept_id"].astype(np.int64)
    freq = freq.loc[freq["patients"] >= min_train_patients].copy()

    registry = pd.read_csv(registry_path)
    registry = registry.loc[registry["token_type"] == "DX"].copy()
    registry["condition_concept_id"] = (
        registry["token_key"].str.removeprefix("DX:").astype(np.int64)
    )
    merged = freq.merge(
        registry[["condition_concept_id", "token_id", "token_key"]],
        on="condition_concept_id",
        how="inner",
    )
    merged = merged.sort_values(
        ["patients", "rows", "condition_concept_id"],
        ascending=[False, False, True],
    ).head(max_concepts)
    return merged.reset_index(drop=True)


def upload_dataframe(conn, frame, table_name, columns_sql, insert_sql, label):
    execute(conn, sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
    execute(
        conn,
        sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT PRESERVE ROWS").format(
            sql.Identifier(table_name),
            sql.SQL(columns_sql),
        ),
    )
    rows = list(frame.itertuples(index=False, name=None))
    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)
    conn.commit()
    log(f"uploaded {label}: {len(rows):,}")


def upload_reference_tables(conn, patient_map, candidates):
    upload_dataframe(
        conn,
        patient_map[["person_id", "split"]],
        "tmp_task19_patient_map",
        "person_id bigint, split text",
        "INSERT INTO tmp_task19_patient_map VALUES (%s,%s)",
        "patient map",
    )
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

    upload_dataframe(
        conn,
        candidates[["condition_concept_id"]],
        "tmp_task19_candidate_concept",
        "condition_concept_id bigint",
        "INSERT INTO tmp_task19_candidate_concept VALUES (%s)",
        "candidate DX concepts",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_task19_candidate_concept(condition_concept_id)",
        label="Index candidate concepts",
    )


def fetch_concept_metadata(conn, args):
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT
                c.concept_id::bigint AS condition_concept_id,
                c.concept_name,
                c.domain_id,
                c.vocabulary_id,
                c.concept_code,
                c.standard_concept
            FROM {}.concept c
            JOIN tmp_task19_candidate_concept x
              ON x.condition_concept_id = c.concept_id
            ORDER BY c.concept_id
            """
        ).format(sql.Identifier(args.schema)),
        label="Fetch concept metadata",
    )


def fetch_index_counts(conn, args, index_date: str):
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
            events AS (
                SELECT
                    e.person_id,
                    p.split,
                    e.condition_concept_id::bigint,
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
                JOIN tmp_task19_candidate_concept c
                  ON c.condition_concept_id = e.condition_concept_id
                WHERE e.condition_start_date IS NOT NULL
                  AND e.condition_start_date <= %s::date + INTERVAL '5 years'
                GROUP BY e.person_id, p.split, e.condition_concept_id
            ),
            base AS (
                SELECT
                    c.condition_concept_id,
                    e.split,
                    e.eligible_patients
                FROM tmp_task19_candidate_concept c
                CROSS JOIN eligible_count e
            ),
            agg AS (
                SELECT
                    e.condition_concept_id,
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
                          AND first_future_date < %s::date + INTERVAL '5 years'
                    )::bigint AS incident_5y
                FROM events e
                GROUP BY e.condition_concept_id, e.split
            )
            SELECT
                %s::date AS index_date,
                b.condition_concept_id,
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
              ON a.condition_concept_id = b.condition_concept_id
             AND a.split = b.split
            ORDER BY b.condition_concept_id, b.split
            """
        ).format(sql.Identifier(args.schema), sql.Identifier(args.schema)),
        params=(
            index_date,
            index_date,
            index_date,
            args.db_end_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
            index_date,
        ),
        label=f"Fetch incidence counts for index_date={index_date}",
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
    for horizon in ["1y", "3y", "5y"]:
        count_col = f"incident_{horizon}"
        rate_col = f"incident_{horizon}_rate"
        denominator = frame["at_risk_patients"].replace(0, np.nan)
        frame[rate_col] = frame[count_col] / denominator
    frame["prior_prevalence"] = (
        frame["prior_patients"] / frame["eligible_patients"].replace(0, np.nan)
    )
    return frame


def summarize_candidates(counts, candidates, metadata):
    total = (
        counts.groupby(["index_date", "condition_concept_id"], as_index=False)
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
    total = total.merge(candidates, on="condition_concept_id", how="left")
    total = total.merge(metadata, on="condition_concept_id", how="left")
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
        ["priority_rank", "incident_5y", "patients"],
        ascending=[True, False, False],
    ).drop(columns=["priority_rank"])


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    patient_map = load_patient_map(args.data_dir)
    candidates = load_candidate_concepts(
        args.data_dir,
        args.max_concepts,
        args.min_train_patients,
    )
    log(f"patients={len(patient_map):,}")
    log(f"candidate_concepts={len(candidates):,}")
    candidates.to_csv(args.output_dir / "candidate_dx_concepts_seed.csv", index=False)

    with connect(args) as conn:
        upload_reference_tables(conn, patient_map, candidates)
        metadata = fetch_concept_metadata(conn, args)
        counts = []
        for index_date in args.index_dates:
            counts.append(fetch_index_counts(conn, args, index_date))

    split_counts = pd.concat(counts, ignore_index=True)
    split_counts = add_rates(split_counts)
    summary = summarize_candidates(split_counts, candidates, metadata)

    split_counts.to_csv(args.output_dir / "phenotype_candidate_split_counts.csv", index=False)
    metadata.to_csv(args.output_dir / "phenotype_candidate_concept_metadata.csv", index=False)
    summary.to_csv(args.output_dir / "phenotype_candidate_summary.csv", index=False)
    manifest = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "max_concepts": args.max_concepts,
        "min_train_patients": args.min_train_patients,
        "index_dates": args.index_dates,
        "db_end_date": args.db_end_date,
        "candidate_concepts": int(len(candidates)),
        "outputs": {
            "seed": str(args.output_dir / "candidate_dx_concepts_seed.csv"),
            "split_counts": str(args.output_dir / "phenotype_candidate_split_counts.csv"),
            "metadata": str(args.output_dir / "phenotype_candidate_concept_metadata.csv"),
            "summary": str(args.output_dir / "phenotype_candidate_summary.csv"),
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
