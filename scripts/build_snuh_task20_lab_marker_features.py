#!/usr/bin/env python3
"""Build raw LAB marker features for Task 20 clinical comparators."""

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
DEFAULT_OUTPUT_DIR = POD_ROOT / "task20" / "outputs" / "lab_marker_features"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task20_lab_marker_features"
DEFAULT_INDEX_DATES = ["2018-01-01"]

# Curated from Task 20 marker profile. These are raw numeric marker-unit pairs,
# not LAB quantile token ids.
DEFAULT_MARKER_PAIRS = [
    {
        "marker": "hba1c",
        "measurement_concept_id": 3004410,
        "unit_concept_id": 8554,
        "description": "Hemoglobin A1c/Hemoglobin.total in Blood, percent",
    },
    {
        "marker": "serum_glucose",
        "measurement_concept_id": 3004501,
        "unit_concept_id": 8840,
        "description": "Glucose in Serum or Plasma, mg/dL",
    },
    {
        "marker": "serum_creatinine",
        "measurement_concept_id": 3016723,
        "unit_concept_id": 8840,
        "description": "Creatinine in Serum or Plasma, mg/dL",
    },
    {
        "marker": "egfr_mdrd",
        "measurement_concept_id": 46236952,
        "unit_concept_id": 720870,
        "description": "eGFR MDRD, mL/min/1.73m2",
    },
    {
        "marker": "egfr_ckdepi",
        "measurement_concept_id": 40764999,
        "unit_concept_id": 0,
        "description": "eGFR CKD-EPI, unit missing in OMOP row",
    },
    {
        "marker": "alt",
        "measurement_concept_id": 3006923,
        "unit_concept_id": 8923,
        "description": "ALT, IU/L",
    },
    {
        "marker": "ast",
        "measurement_concept_id": 3013721,
        "unit_concept_id": 8923,
        "description": "AST, IU/L",
    },
    {
        "marker": "total_bilirubin",
        "measurement_concept_id": 3024128,
        "unit_concept_id": 8840,
        "description": "Total bilirubin, mg/dL",
    },
    {
        "marker": "platelet_count",
        "measurement_concept_id": 3024929,
        "unit_concept_id": 8848,
        "description": "Platelet count, thousand/uL",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-dates", nargs="+", default=DEFAULT_INDEX_DATES)
    parser.add_argument("--marker-config", type=Path)
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
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
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


def load_marker_config(path: Path | None):
    if path is None:
        frame = pd.DataFrame(DEFAULT_MARKER_PAIRS)
    else:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
    required = {"marker", "measurement_concept_id", "unit_concept_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"marker config is missing columns: {missing}")
    frame = frame.copy()
    frame["marker"] = frame["marker"].astype(str)
    frame["measurement_concept_id"] = frame["measurement_concept_id"].astype(np.int64)
    frame["unit_concept_id"] = frame["unit_concept_id"].fillna(0).astype(np.int64)
    frame = frame.drop_duplicates(["marker", "measurement_concept_id", "unit_concept_id"])
    return frame


def upload_patient_map(conn, patient_map: pd.DataFrame):
    execute(conn, "DROP TABLE IF EXISTS tmp_task20_patient_map")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_task20_patient_map (
            person_id bigint,
            split text
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    rows = [(int(person_id), str(split)) for person_id, split in patient_map.itertuples(index=False)]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO tmp_task20_patient_map VALUES (%s,%s)", rows)
    conn.commit()
    log(f"uploaded patient map: {len(rows):,}")
    execute(conn, "CREATE INDEX ON tmp_task20_patient_map(person_id)", label="Index patient map")
    execute(conn, "CREATE INDEX ON tmp_task20_patient_map(split, person_id)", label="Index patient split")


def upload_marker_config(conn, markers: pd.DataFrame):
    execute(conn, "DROP TABLE IF EXISTS tmp_task20_marker_pair")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_task20_marker_pair (
            marker text,
            measurement_concept_id bigint,
            unit_concept_id bigint
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    rows = [
        (str(row.marker), int(row.measurement_concept_id), int(row.unit_concept_id))
        for row in markers.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO tmp_task20_marker_pair VALUES (%s,%s,%s)", rows)
    conn.commit()
    log(f"uploaded marker pairs: {len(rows):,}")
    execute(
        conn,
        "CREATE INDEX ON tmp_task20_marker_pair(measurement_concept_id, unit_concept_id)",
        label="Index marker pairs",
    )


def fetch_marker_features(conn, args, index_date: str):
    return query_df(
        conn,
        sql.SQL(
            """
            WITH selected_measurements AS (
                SELECT
                    p.person_id,
                    p.split,
                    mp.marker,
                    m.measurement_date::date AS measurement_date,
                    m.value_as_number::double precision AS value_as_number
                FROM {}.measurement m
                JOIN tmp_task20_patient_map p
                  ON p.person_id = m.person_id
                JOIN tmp_task20_marker_pair mp
                  ON mp.measurement_concept_id = m.measurement_concept_id
                 AND mp.unit_concept_id = COALESCE(m.unit_concept_id, 0)
                WHERE m.measurement_date IS NOT NULL
                  AND m.measurement_date < %s::date
                  AND m.value_as_number IS NOT NULL
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY person_id, marker
                        ORDER BY measurement_date DESC, value_as_number DESC
                    ) AS reverse_rank
                FROM selected_measurements
            ),
            aggregated AS (
                SELECT
                    person_id,
                    marker,
                    COUNT(*)::bigint AS lab_count,
                    COUNT(*) FILTER (
                        WHERE measurement_date >= %s::date - INTERVAL '365 days'
                    )::bigint AS lab_count_1y,
                    COUNT(*) FILTER (
                        WHERE measurement_date >= %s::date - INTERVAL '730 days'
                    )::bigint AS lab_count_2y,
                    MIN(value_as_number)::double precision AS value_min,
                    MAX(value_as_number)::double precision AS value_max,
                    AVG(value_as_number)::double precision AS value_mean,
                    STDDEV_POP(value_as_number)::double precision AS value_std,
                    MIN(measurement_date)::date AS first_date,
                    MAX(measurement_date)::date AS latest_date,
                    AVG(value_as_number) FILTER (
                        WHERE measurement_date >= %s::date - INTERVAL '365 days'
                    )::double precision AS value_mean_1y,
                    AVG(value_as_number) FILTER (
                        WHERE measurement_date >= %s::date - INTERVAL '730 days'
                    )::double precision AS value_mean_2y
                FROM selected_measurements
                GROUP BY person_id, marker
            ),
            latest AS (
                SELECT
                    person_id,
                    marker,
                    value_as_number AS latest_value,
                    measurement_date AS latest_date
                FROM ranked
                WHERE reverse_rank = 1
            )
            SELECT
                p.person_id::bigint,
                p.split,
                a.marker,
                a.lab_count,
                a.lab_count_1y,
                a.lab_count_2y,
                a.value_min,
                a.value_max,
                a.value_mean,
                COALESCE(a.value_std, 0)::double precision AS value_std,
                a.value_mean_1y,
                a.value_mean_2y,
                a.first_date,
                a.latest_date,
                l.latest_value,
                (%s::date - l.latest_date)::integer AS days_since_latest
            FROM aggregated a
            JOIN latest l
              ON l.person_id = a.person_id
             AND l.marker = a.marker
            JOIN tmp_task20_patient_map p
              ON p.person_id = a.person_id
            ORDER BY p.person_id, a.marker
            """
        ).format(sql.Identifier(args.schema)),
        params=(index_date, index_date, index_date, index_date, index_date, index_date),
        label=f"Fetch raw LAB marker features for index_date={index_date}",
    )


def long_to_wide(features: pd.DataFrame, patient_map: pd.DataFrame):
    index = patient_map[["person_id", "split"]].copy()
    if features.empty:
        return index
    value_cols = [
        "lab_count",
        "lab_count_1y",
        "lab_count_2y",
        "value_min",
        "value_max",
        "value_mean",
        "value_std",
        "value_mean_1y",
        "value_mean_2y",
        "latest_value",
        "days_since_latest",
    ]
    wide_parts = [index.set_index("person_id")]
    for value_col in value_cols:
        pivot = features.pivot(index="person_id", columns="marker", values=value_col)
        pivot.columns = [f"lab_{marker}__{value_col}" for marker in pivot.columns]
        wide_parts.append(pivot)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    for column in wide.columns:
        if column.startswith("lab_") and column.endswith("__lab_count"):
            wide[column] = wide[column].fillna(0)
    return wide


def summarize_long(features: pd.DataFrame):
    if features.empty:
        return pd.DataFrame(columns=["marker"])
    return (
        features.groupby("marker", dropna=False)
        .agg(
            patients=("person_id", "nunique"),
            total_rows=("lab_count", "sum"),
            median_latest_value=("latest_value", "median"),
            p05_latest_value=("latest_value", lambda x: float(np.nanpercentile(x, 5))),
            p95_latest_value=("latest_value", lambda x: float(np.nanpercentile(x, 95))),
            median_days_since_latest=("days_since_latest", "median"),
            p95_days_since_latest=("days_since_latest", lambda x: float(np.nanpercentile(x, 95))),
        )
        .reset_index()
        .sort_values("patients", ascending=False)
    )


def safe_date(index_date: str):
    return index_date.replace("-", "")


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    patient_map = load_patient_map(args.data_dir)
    markers = load_marker_config(args.marker_config.expanduser().resolve() if args.marker_config else None)
    markers_path = args.output_dir / "marker_config_used.csv"
    markers.to_csv(markers_path, index=False)
    log(f"patients={len(patient_map):,}")
    log(f"marker_pairs={len(markers):,}")

    outputs = {}
    summaries = []
    with connect(args) as conn:
        upload_patient_map(conn, patient_map)
        upload_marker_config(conn, markers)
        for index_date in args.index_dates:
            date = safe_date(index_date)
            long_features = fetch_marker_features(conn, args, index_date)
            if not long_features.empty:
                long_features["person_id"] = long_features["person_id"].astype(np.int64)
            wide_features = long_to_wide(long_features, patient_map)
            summary = summarize_long(long_features)
            summary.insert(0, "index_date", index_date)

            long_path = args.output_dir / f"lab_marker_features_long_{date}.parquet"
            wide_path = args.output_dir / f"lab_marker_features_wide_{date}.parquet"
            summary_path = args.output_dir / f"lab_marker_feature_summary_{date}.csv"
            long_features.to_parquet(long_path, index=False)
            wide_features.to_parquet(wide_path, index=False)
            summary.to_csv(summary_path, index=False)
            summaries.append(summary)
            outputs[index_date] = {
                "long": str(long_path),
                "wide": str(wide_path),
                "summary": str(summary_path),
            }
            print(f"\n## marker feature summary {index_date}", flush=True)
            print(summary.to_string(index=False), flush=True)

    combined_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined_summary.to_csv(args.output_dir / "lab_marker_feature_summary_all.csv", index=False)
    manifest = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "index_dates": args.index_dates,
        "marker_config": str(markers_path),
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n## outputs", flush=True)
    print(json.dumps(outputs, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
