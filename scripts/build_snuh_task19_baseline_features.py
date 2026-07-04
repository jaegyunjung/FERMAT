#!/usr/bin/env python3
"""Build pre-index baseline features for Task 19 disease-risk prediction.

The output is one patient-level parquet per index date. It intentionally uses
only source records before the index date, so the features can be joined to the
Task 19 label tables without leaking future information.
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
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task19_baseline_features"
DEFAULT_INDEX_DATES = ["2015-01-01", "2018-01-01", "2020-01-01"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dates", nargs="+", default=DEFAULT_INDEX_DATES)
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


def fetch_pre_index_counts(conn, args, index_date: str):
    return query_df(
        conn,
        sql.SQL(
            """
            WITH dx AS (
                SELECT
                    c.person_id,
                    COUNT(*)::bigint AS dx_rows,
                    COUNT(DISTINCT c.condition_concept_id)::bigint AS dx_unique_concepts,
                    COUNT(DISTINCT c.condition_start_date)::bigint AS dx_active_days,
                    MIN(c.condition_start_date)::date AS dx_first_date,
                    MAX(c.condition_start_date)::date AS dx_last_date
                FROM {}.condition_occurrence c
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE c.condition_start_date IS NOT NULL
                  AND c.condition_start_date < %s::date
                GROUP BY c.person_id
            ),
            rx AS (
                SELECT
                    d.person_id,
                    COUNT(*)::bigint AS rx_rows,
                    COUNT(DISTINCT d.drug_concept_id)::bigint AS rx_unique_concepts,
                    COUNT(DISTINCT d.drug_exposure_start_date)::bigint AS rx_active_days,
                    MIN(d.drug_exposure_start_date)::date AS rx_first_date,
                    MAX(d.drug_exposure_start_date)::date AS rx_last_date
                FROM {}.drug_exposure d
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE d.drug_exposure_start_date IS NOT NULL
                  AND d.drug_exposure_start_date < %s::date
                GROUP BY d.person_id
            ),
            px AS (
                SELECT
                    pr.person_id,
                    COUNT(*)::bigint AS px_rows,
                    COUNT(DISTINCT pr.procedure_concept_id)::bigint AS px_unique_concepts,
                    COUNT(DISTINCT pr.procedure_date)::bigint AS px_active_days,
                    MIN(pr.procedure_date)::date AS px_first_date,
                    MAX(pr.procedure_date)::date AS px_last_date
                FROM {}.procedure_occurrence pr
                JOIN tmp_task19_patient_map p USING(person_id)
                WHERE pr.procedure_date IS NOT NULL
                  AND pr.procedure_date < %s::date
                GROUP BY pr.person_id
            )
            SELECT
                p.person_id::bigint,
                p.split,
                COALESCE(dx.dx_rows, 0)::bigint AS dx_rows,
                COALESCE(dx.dx_unique_concepts, 0)::bigint AS dx_unique_concepts,
                COALESCE(dx.dx_active_days, 0)::bigint AS dx_active_days,
                dx.dx_first_date,
                dx.dx_last_date,
                COALESCE(rx.rx_rows, 0)::bigint AS rx_rows,
                COALESCE(rx.rx_unique_concepts, 0)::bigint AS rx_unique_concepts,
                COALESCE(rx.rx_active_days, 0)::bigint AS rx_active_days,
                rx.rx_first_date,
                rx.rx_last_date,
                COALESCE(px.px_rows, 0)::bigint AS px_rows,
                COALESCE(px.px_unique_concepts, 0)::bigint AS px_unique_concepts,
                COALESCE(px.px_active_days, 0)::bigint AS px_active_days,
                px.px_first_date,
                px.px_last_date
            FROM tmp_task19_patient_map p
            LEFT JOIN dx USING(person_id)
            LEFT JOIN rx USING(person_id)
            LEFT JOIN px USING(person_id)
            ORDER BY p.person_id
            """
        ).format(
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
        ),
        params=(index_date, index_date, index_date),
        label=f"Fetch pre-index counts for index_date={index_date}",
    )


def add_derived_features(features, labels, index_date: str):
    frame = labels[
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
    ].merge(features, on=["person_id", "split"], how="left")

    date_cols = [
        "first_activity_date",
        "last_activity_date",
        "dx_first_date",
        "dx_last_date",
        "rx_first_date",
        "rx_last_date",
        "px_first_date",
        "px_last_date",
    ]
    for column in date_cols:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    idx = pd.Timestamp(index_date)
    for prefix in ["dx", "rx", "px"]:
        rows_col = f"{prefix}_rows"
        unique_col = f"{prefix}_unique_concepts"
        days_col = f"{prefix}_active_days"
        first_col = f"{prefix}_first_date"
        last_col = f"{prefix}_last_date"
        for column in [rows_col, unique_col, days_col]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame[f"{prefix}_history_days"] = (idx - frame[first_col]).dt.days.clip(lower=0)
        frame[f"{prefix}_recency_days"] = (idx - frame[last_col]).dt.days.clip(lower=0)
        frame[f"log1p_{rows_col}"] = np.log1p(frame[rows_col])
        frame[f"log1p_{unique_col}"] = np.log1p(frame[unique_col])
        frame[f"log1p_{days_col}"] = np.log1p(frame[days_col])

    frame["clinical_rows"] = frame["dx_rows"] + frame["rx_rows"] + frame["px_rows"]
    frame["clinical_unique_concepts"] = (
        frame["dx_unique_concepts"]
        + frame["rx_unique_concepts"]
        + frame["px_unique_concepts"]
    )
    frame["clinical_active_days"] = (
        frame["dx_active_days"] + frame["rx_active_days"] + frame["px_active_days"]
    )
    frame["log1p_clinical_rows"] = np.log1p(frame["clinical_rows"])
    frame["log1p_clinical_unique_concepts"] = np.log1p(frame["clinical_unique_concepts"])
    frame["log1p_clinical_active_days"] = np.log1p(frame["clinical_active_days"])
    return frame


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    patient_map = load_patient_map(args.data_dir)
    log(f"patients={len(patient_map):,}")

    outputs = {}
    with connect(args) as conn:
        upload_patient_map(conn, patient_map)
        for index_date in args.index_dates:
            safe_date = index_date.replace("-", "")
            label_path = args.label_dir / f"patient_phenotype_labels_wide_{safe_date}.parquet"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            labels = pd.read_parquet(
                label_path,
                columns=[
                    "person_id",
                    "split",
                    "index_date",
                    "gender_concept_id",
                    "age_at_index",
                    "first_activity_date",
                    "last_activity_date",
                    "has_pre_index_washout",
                ],
            )
            features = fetch_pre_index_counts(conn, args, index_date)
            features = add_derived_features(features, labels, index_date)
            out_path = args.output_dir / f"baseline_features_{safe_date}.parquet"
            features.to_parquet(out_path, index=False)
            outputs[index_date] = str(out_path)
            log(f"[DONE] wrote {out_path}")

    manifest = {
        "data_dir": str(args.data_dir),
        "label_dir": str(args.label_dir),
        "output_dir": str(args.output_dir),
        "index_dates": args.index_dates,
        "outputs": outputs,
        "feature_scope": "source records strictly before each index date",
        "tables": ["condition_occurrence", "drug_exposure", "procedure_occurrence"],
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
