#!/usr/bin/env python3
"""Audit whether genomic measurement results can support FERMAT LAB_GEN tokens.

This script is intentionally defensive for long-running Pod/database work:

- every query is labeled, timed, and prints the PostgreSQL backend pid;
- stdout/stderr are mirrored to logs/task17_run.log by default;
- only likely genomic LAB test concepts are scanned by default;
- measurement scans are chunked and checkpointed as JSONL;
- each chunk has a statement timeout;
- exceptions cancel the active PostgreSQL connection before exiting.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import json
import os
import platform
import re
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:
    psycopg = None
    sql = None


DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_genomic_variant_feasibility_audit_v2"

GENOMIC_KEYWORDS = [
    "genetic",
    "genomic",
    "genotype",
    "germline",
    "somatic",
    "mutation",
    "variant",
    "molecular",
    "biomarker",
    "ngs",
    "sequencing",
    "fish",
    "pcr",
    "ihc",
]

GENE_KEYWORDS = [
    "ALK",
    "BRAF",
    "BRCA",
    "EGFR",
    "ERBB2",
    "HER2",
    "KRAS",
    "MET",
    "MLH1",
    "MSH2",
    "MSH6",
    "NRAS",
    "NTRK",
    "PD-L1",
    "PDL1",
    "PIK3CA",
    "PMS2",
    "RET",
    "ROS1",
    "MSI",
]

TEST_CLASS_PREDICATE = """
(
       (vocabulary_id = 'LOINC' AND concept_class_id IN ('Lab Test', 'Clinical Observation'))
    OR (vocabulary_id = 'EDI' AND concept_class_id = 'Meas Class')
    OR (vocabulary_id = 'SNUBH generated' AND concept_class_id = 'Lab Test')
    OR (vocabulary_id = 'OMOP Extension' AND concept_class_id = 'Lab Test')
    OR (vocabulary_id = 'SNOMED' AND concept_class_id = 'Observable Entity')
    OR (vocabulary_id = 'CIEL' AND concept_class_id = 'Test')
)
"""

VARIANT_ONTOLOGY_PREDICATE = """
(
       vocabulary_id IN ('OMOP Genomic', 'JAX', 'ClinVar', 'OncoKB', 'CIViC', 'NCIt')
    OR concept_class_id IN (
        'Variant',
        'Genetic Variation',
        'RNA Variant',
        'Protein Variant',
        'DNA Variant'
    )
)
"""


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task17_genomic_variant_audit"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/task17_run.log"))
    parser.add_argument(
        "--etl-dir",
        type=Path,
        default=Path("/home/khdp-user/workspace/fermat-data/etl/patient_100pct_seed_42"),
        help=(
            "Task 15 ETL output directory. The default audit uses its LAB "
            "frequency parquet files instead of rescanning the measurement table."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--statement-timeout", default="20min")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument(
        "--max-scan-concepts",
        type=int,
        default=0,
        help=(
            "Abort before scanning measurement if the filtered scan candidate "
            "set is broader than this. Default 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--scan-all-candidates",
        action="store_true",
        help=(
            "Scan every discovered concept. Default scans only likely genomic "
            "LAB test concepts and leaves variant ontologies as review material."
        ),
    )
    parser.add_argument(
        "--db-measurement-scan",
        action="store_true",
        help=(
            "Run the slower chunked DB measurement scan. Default is off; use "
            "Task 15 ETL frequency files instead."
        ),
    )
    parser.add_argument(
        "--skip-source-value-scan",
        action="store_true",
        help=(
            "Skip source-value regex checks on OMOP rows. By default Task 17 "
            "checks source/value text columns with bounded top-value queries."
        ),
    )
    parser.add_argument("--source-top-values", type=int, default=50)
    parser.add_argument(
        "--source-patient-buckets",
        type=int,
        default=1,
        help=(
            "Percent of patients used for source/value text checks. Default 1 "
            "keeps regex scans bounded; use 100 only intentionally."
        ),
    )
    parser.add_argument("--source-sample-seed", type=int, default=20260617)
    parser.add_argument("--source-statement-timeout", default="5min")
    parser.add_argument(
        "--top-concepts",
        type=int,
        default=0,
        help=(
            "Optional final top-concept query. Default 0 avoids a late "
            "unchunked scan after the checkpointed presence pass."
        ),
    )
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--no-log-tee", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def install_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, handle)
    sys.stderr = Tee(sys.__stderr__, handle)
    print(f"[LOG] tee stdout/stderr -> {log_file.resolve()}", flush=True)
    return handle


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_iso()}] {message}", flush=True)


def pg_word_regex(terms):
    escaped = [re.escape(term) for term in sorted(set(terms), key=lambda item: (len(item), item))]
    return r"\m(" + "|".join(escaped) + r")\M"


def keyword_regexes(extra_keywords):
    extra = [kw.strip() for kw in extra_keywords if kw.strip()]
    broad_regex = pg_word_regex(GENOMIC_KEYWORDS + extra)
    gene_regex = pg_word_regex(GENE_KEYWORDS + extra)
    return broad_regex, gene_regex


def connect(args, password):
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password,
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name=APPLICATION_NAME,
        options=(
            f"-c statement_timeout={args.statement_timeout} "
            "-c idle_in_transaction_session_timeout=5min "
            "-c tcp_keepalives_idle=60 "
            "-c tcp_keepalives_interval=30 "
            "-c tcp_keepalives_count=5"
        ),
    )


def fetch_rows(conn, query, params=None, label=None, heartbeat_seconds=60):
    if label:
        with conn.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            backend_pid = cursor.fetchone()[0]
        log(f"[START] {label} backend_pid={backend_pid}")
    started = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(heartbeat_seconds):
            log(f"[RUNNING] {label} elapsed_min={(time.time() - started) / 60:.1f}")

    thread = None
    if label and heartbeat_seconds > 0:
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            columns = [column.name for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2)
    if label:
        log(f"[DONE] {label} elapsed_sec={time.time() - started:.1f} rows={len(rows)}")
    return rows


def execute_sql(conn, query, params=None, label=None):
    if label:
        log(f"[START] {label}")
    started = time.time()
    with conn.cursor() as cursor:
        cursor.execute(query, params or ())
    if label:
        log(f"[DONE] {label} elapsed_sec={time.time() - started:.1f}")


def fetch_one(conn, query, params=None, label=None, heartbeat_seconds=60):
    rows = fetch_rows(conn, query, params, label, heartbeat_seconds)
    return rows[0] if rows else {}


def try_fetch_rows(conn, query, params=None, label=None, heartbeat_seconds=60):
    try:
        rows = fetch_rows(conn, query, params, label, heartbeat_seconds)
    except Exception as error:
        log(f"[WARN] {label} failed: {error!r}")
        try:
            conn.cancel()
        except Exception:
            pass
        return {
            "status": "failed",
            "error": repr(error),
            "rows": [],
        }
    return {
        "status": "ok",
        "error": None,
        "rows": rows,
    }


def table_columns(conn, schema, table):
    rows = fetch_rows(
        conn,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
        label=f"Inspect {schema}.{table} columns",
        heartbeat_seconds=0,
    )
    return {row["column_name"] for row in rows}


def find_candidate_concepts(conn, schema, broad_regex, gene_regex, heartbeat_seconds):
    rows = fetch_rows(
        conn,
        sql.SQL(
            """
            WITH found AS (
                SELECT
                    concept_id::bigint AS concept_id,
                    concept_name,
                    domain_id,
                    vocabulary_id,
                    concept_class_id,
                    standard_concept,
                    concept_code,
                    {test_class_predicate} AS is_lab_test_class,
                    {variant_ontology_predicate} AS is_variant_ontology
                FROM {schema}.concept
                WHERE invalid_reason IS NULL
                  AND (
                        concept_name ~* %s
                     OR concept_name ~* %s
                     OR concept_code ~* %s
                  )
            )
            SELECT
                *,
                (
                    domain_id = 'Measurement'
                    AND is_lab_test_class
                    AND NOT is_variant_ontology
                    AND (
                           concept_name ~* %s
                        OR concept_name ~* %s
                        OR concept_code ~* %s
                    )
                ) AS default_scan_candidate,
                CASE
                    WHEN domain_id = 'Measurement'
                     AND is_lab_test_class
                     AND NOT is_variant_ontology
                    THEN 'LAB_GEN_TEST_CANDIDATE'
                    WHEN is_variant_ontology
                    THEN 'VARIANT_ONTOLOGY_REVIEW_ONLY'
                    ELSE 'RELATED_CONCEPT_REVIEW_ONLY'
                END AS audit_role
            FROM found
            ORDER BY default_scan_candidate DESC, audit_role, domain_id,
                     vocabulary_id, concept_class_id, concept_name, concept_id
            """
        ).format(
            schema=sql.Identifier(schema),
            test_class_predicate=sql.SQL(TEST_CLASS_PREDICATE),
            variant_ontology_predicate=sql.SQL(VARIANT_ONTOLOGY_PREDICATE),
        ),
        (
            broad_regex,
            gene_regex,
            gene_regex,
            broad_regex,
            gene_regex,
            gene_regex,
        ),
        label="Find candidate molecular/genomic concepts",
        heartbeat_seconds=heartbeat_seconds,
    )
    log(f"[INFO] candidate molecular concepts: {len(rows)}")
    return rows


def chunks(items, size):
    for index in range(0, len(items), size):
        yield index // size, items[index:index + size]


def load_completed_chunks(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["chunk_index"])
            except Exception:
                continue
    return completed


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def read_parquet(path):
    try:
        import pandas as pd
    except Exception as error:
        raise RuntimeError(
            "Missing dependency: pandas/pyarrow is required to read Task 15 "
            "frequency parquet files."
        ) from error
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def add_concept_metadata(frame, candidate_rows, concept_col):
    import pandas as pd

    meta = pd.DataFrame(candidate_rows)
    if meta.empty:
        return frame
    meta = meta.rename(columns={"concept_id": concept_col})
    keep = [
        concept_col,
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "audit_role",
    ]
    keep = [column for column in keep if column in meta.columns]
    return frame.merge(meta[keep], on=concept_col, how="left")


def summarize_with_task15_etl(args, candidate_rows, scan_rows):
    import pandas as pd

    etl_dir = args.etl_dir.expanduser()
    numeric_path = etl_dir / "train_numeric_lab_stats.parquet"
    categorical_path = etl_dir / "train_lab_categorical_frequency.parquet"
    log(f"[TASK17] using Task 15 ETL frequencies from {etl_dir}")
    numeric = read_parquet(numeric_path)
    categorical = read_parquet(categorical_path)

    scan_ids = {int(row["concept_id"]) for row in scan_rows}
    candidate_ids = {int(row["concept_id"]) for row in candidate_rows}
    numeric["measurement_concept_id"] = numeric["measurement_concept_id"].astype("int64")
    categorical["measurement_concept_id"] = categorical["measurement_concept_id"].astype("int64")
    categorical["value_as_concept_id"] = categorical["value_as_concept_id"].astype("int64")

    numeric_hits = numeric.loc[numeric["measurement_concept_id"].isin(scan_ids)].copy()
    categorical_hits = categorical.loc[categorical["measurement_concept_id"].isin(scan_ids)].copy()

    numeric_by_test = (
        numeric_hits
        .groupby("measurement_concept_id", as_index=False)
        .agg(
            numeric_daily_rows=("daily_rows", "sum"),
            numeric_patient_unit_sum=("patients", "sum"),
            numeric_unit_count=("unit_concept_id", "nunique"),
            frequent_numeric_unit_count=("is_frequent", "sum"),
        )
        if not numeric_hits.empty else
        pd.DataFrame(columns=[
            "measurement_concept_id",
            "numeric_daily_rows",
            "numeric_patient_unit_sum",
            "numeric_unit_count",
            "frequent_numeric_unit_count",
        ])
    )
    categorical_by_test = (
        categorical_hits
        .groupby("measurement_concept_id", as_index=False)
        .agg(
            categorical_rows=("rows", "sum"),
            categorical_patient_value_sum=("patients", "sum"),
            categorical_value_count=("value_as_concept_id", "nunique"),
        )
        if not categorical_hits.empty else
        pd.DataFrame(columns=[
            "measurement_concept_id",
            "categorical_rows",
            "categorical_patient_value_sum",
            "categorical_value_count",
        ])
    )

    observed = pd.DataFrame(scan_rows).rename(columns={"concept_id": "measurement_concept_id"})
    observed["measurement_concept_id"] = observed["measurement_concept_id"].astype("int64")
    observed = observed.merge(numeric_by_test, on="measurement_concept_id", how="left")
    observed = observed.merge(categorical_by_test, on="measurement_concept_id", how="left")
    for column in [
        "numeric_daily_rows",
        "numeric_patient_unit_sum",
        "numeric_unit_count",
        "frequent_numeric_unit_count",
        "categorical_rows",
        "categorical_patient_value_sum",
        "categorical_value_count",
    ]:
        observed[column] = observed[column].fillna(0).astype("int64")
    observed = observed.loc[
        (observed["numeric_daily_rows"] > 0) | (observed["categorical_rows"] > 0)
    ].copy()
    observed["recommended_token_rule"] = "review"
    observed.loc[
        observed["frequent_numeric_unit_count"] > 0,
        "recommended_token_rule",
    ] = "LAB:<measurement_concept_id>:<unit_concept_id>:Qxx"
    observed.loc[
        (observed["frequent_numeric_unit_count"] == 0)
        & (observed["numeric_daily_rows"] > 0),
        "recommended_token_rule",
    ] = "LAB_TEST:<measurement_concept_id>"
    observed.loc[
        observed["categorical_rows"] > 0,
        "recommended_token_rule",
    ] = "LAB_CAT:<measurement_concept_id>:<value_as_concept_id>"

    variant_value_hits = categorical.loc[
        categorical["value_as_concept_id"].isin(candidate_ids)
    ].copy()
    variant_value_hits = add_concept_metadata(
        variant_value_hits,
        candidate_rows,
        "value_as_concept_id",
    )

    observed_csv = args.output_dir / "lab_gen_observed_test_candidates.csv"
    observed.to_csv(observed_csv, index=False)
    numeric_csv = args.output_dir / "lab_gen_numeric_unit_candidates.csv"
    numeric_hits.to_csv(numeric_csv, index=False)
    categorical_csv = args.output_dir / "lab_gen_categorical_result_candidates.csv"
    categorical_hits.to_csv(categorical_csv, index=False)
    variant_value_csv = args.output_dir / "variant_value_as_concept_hits.csv"
    variant_value_hits.to_csv(variant_value_csv, index=False)

    summary = {
        "etl_dir": str(etl_dir),
        "numeric_frequency_path": str(numeric_path),
        "categorical_frequency_path": str(categorical_path),
        "scan_candidate_concepts": len(scan_rows),
        "all_numeric_lab_unit_rows": int(len(numeric)),
        "all_categorical_lab_value_rows": int(len(categorical)),
        "all_numeric_lab_daily_rows": int(numeric["daily_rows"].sum()) if not numeric.empty else 0,
        "all_categorical_lab_rows": int(categorical["rows"].sum()) if not categorical.empty else 0,
        "observed_lab_gen_test_candidates": int(len(observed)),
        "observed_lab_gen_test_candidate_rate": safe_rate(len(observed), len(scan_rows)),
        "missing_lab_gen_test_candidates": int(len(scan_rows) - len(observed)),
        "missing_lab_gen_test_candidate_rate": safe_rate(len(scan_rows) - len(observed), len(scan_rows)),
        "numeric_unit_candidate_rows": int(len(numeric_hits)),
        "categorical_result_candidate_rows": int(len(categorical_hits)),
        "variant_value_as_concept_hits": int(len(variant_value_hits)),
        "numeric_daily_rows": int(numeric_hits["daily_rows"].sum()) if not numeric_hits.empty else 0,
        "categorical_rows": int(categorical_hits["rows"].sum()) if not categorical_hits.empty else 0,
        "genomic_candidate_numeric_lab_row_rate": safe_rate(
            int(numeric_hits["daily_rows"].sum()) if not numeric_hits.empty else 0,
            int(numeric["daily_rows"].sum()) if not numeric.empty else 0,
        ),
        "genomic_candidate_categorical_lab_row_rate": safe_rate(
            int(categorical_hits["rows"].sum()) if not categorical_hits.empty else 0,
            int(categorical["rows"].sum()) if not categorical.empty else 0,
        ),
        "note": (
            "Counts come from Task 15 train-split frequency parquet files. "
            "Patient counts summed across units or categorical values may double-count."
        ),
    }
    outputs = {
        "lab_gen_observed_test_candidates_csv": str(observed_csv),
        "lab_gen_numeric_unit_candidates_csv": str(numeric_csv),
        "lab_gen_categorical_result_candidates_csv": str(categorical_csv),
        "variant_value_as_concept_hits_csv": str(variant_value_csv),
    }
    log(
        "[TASK17] observed LAB_GEN test candidates from ETL: "
        f"{summary['observed_lab_gen_test_candidates']} "
        f"(numeric unit rows={summary['numeric_unit_candidate_rows']}, "
        f"categorical rows={summary['categorical_result_candidate_rows']})"
    )
    return summary, outputs


def source_value_regex():
    terms = [
        "EGFR",
        "KRAS",
        "NRAS",
        "BRAF",
        "ALK",
        "ROS1",
        "BRCA",
        "ERBB2",
        "HER2",
        "MSI",
        "MMR",
        "MLH1",
        "MSH2",
        "MSH6",
        "PMS2",
        "NTRK",
        "RET",
        "MET",
        "PIK3CA",
        "PD-L1",
        "PDL1",
        "NGS",
        "sequencing",
        "genotype",
        "mutation",
        "variant",
        "fusion",
        "rearrangement",
        "amplification",
        "deletion",
        "microsatellite",
    ]
    return pg_word_regex(terms)


def source_text_query(schema, table, text_col, date_col):
    return sql.SQL(
        """
        SELECT
            %s::text AS source_table,
            %s::text AS source_column,
            NULLIF(t.{text_col}::text, '') AS source_value,
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT t.person_id)::bigint AS patients,
            MIN(t.{date_col})::text AS first_date,
            MAX(t.{date_col})::text AS last_date
        FROM {schema}.{table} AS t
        JOIN tmp_task17_source_person AS p USING (person_id)
        WHERE t.{text_col} IS NOT NULL
          AND NULLIF(t.{text_col}::text, '') IS NOT NULL
          AND t.{text_col}::text ~* %s
          AND t.{date_col} BETWEEN DATE '1900-01-01' AND %s::date
        GROUP BY NULLIF(t.{text_col}::text, '')
        ORDER BY rows DESC, source_value
        LIMIT %s
        """
    ).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        text_col=sql.Identifier(text_col),
        date_col=sql.Identifier(date_col),
    )


def run_source_value_audit(conn, args):
    if not 1 <= args.source_patient_buckets <= 100:
        raise ValueError("--source-patient-buckets must be between 1 and 100")
    regex = source_value_regex()
    execute_sql(
        conn,
        sql.SQL("SET statement_timeout = {}").format(
            sql.Literal(args.source_statement_timeout)
        ),
        label=f"Set source scan statement_timeout={args.source_statement_timeout}",
    )
    execute_sql(conn, "DROP TABLE IF EXISTS tmp_task17_source_person")
    execute_sql(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_task17_source_person
            ON COMMIT PRESERVE ROWS AS
            SELECT person_id
            FROM {schema}.person
            WHERE mod(
                hashtextextended(person_id::text, %s) & 9223372036854775807,
                100
            ) < %s
            """
        ).format(schema=sql.Identifier(args.schema)),
        (args.source_sample_seed, args.source_patient_buckets),
        label=(
            "Create source-value sampled patient table "
            f"({args.source_patient_buckets}% patients)"
        ),
    )
    execute_sql(
        conn,
        "CREATE INDEX ON tmp_task17_source_person(person_id)",
        label="Index source-value sampled patients",
    )
    execute_sql(
        conn,
        "ANALYZE tmp_task17_source_person",
        label="Analyze source-value sampled patients",
    )
    sample_count = fetch_one(
        conn,
        "SELECT COUNT(*)::bigint AS patients FROM tmp_task17_source_person",
        label="Count source-value sampled patients",
        heartbeat_seconds=0,
    )
    specs = [
        (
            "measurement",
            "measurement_date",
            [
                "measurement_source_value",
                "value_source_value",
            ],
        ),
        (
            "procedure_occurrence",
            "procedure_date",
            [
                "procedure_source_value",
            ],
        ),
        (
            "observation",
            "observation_date",
            [
                "observation_source_value",
                "value_source_value",
            ],
        ),
    ]
    all_rows = []
    statuses = []
    for table, date_col, text_cols in specs:
        columns = table_columns(conn, args.schema, table)
        if "person_id" not in columns or date_col not in columns:
            statuses.append({
                "table": table,
                "status": "missing_required_columns",
            })
            continue
        for text_col in text_cols:
            if text_col not in columns:
                statuses.append({
                    "table": table,
                    "column": text_col,
                    "status": "missing_column",
                })
                continue
            label = f"Source text top values {table}.{text_col}"
            result = try_fetch_rows(
                conn,
                source_text_query(args.schema, table, text_col, date_col),
                (table, text_col, regex, args.db_end_date, args.source_top_values),
                label=label,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            statuses.append({
                "table": table,
                "column": text_col,
                "status": result["status"],
                "error": result["error"],
                "returned_top_values": len(result["rows"]),
            })
            all_rows.extend(result["rows"])
    output = args.output_dir / "source_value_genomic_regex_top_values.csv"
    write_csv(
        output,
        all_rows,
        [
            "source_table",
            "source_column",
            "source_value",
            "rows",
            "patients",
            "first_date",
            "last_date",
        ],
    )
    status_output = args.output_dir / "source_value_scan_status.csv"
    write_csv(
        status_output,
        statuses,
        [
            "table",
            "column",
            "status",
            "error",
            "returned_top_values",
        ],
    )
    ok_rows = [row for row in statuses if row.get("status") == "ok"]
    failed_rows = [row for row in statuses if row.get("status") == "failed"]
    log(
        "[TASK17] source-value audit finished: "
        f"ok_columns={len(ok_rows)} failed_columns={len(failed_rows)} "
        f"top_values={len(all_rows)}"
    )
    return {
        "regex": regex,
        "patient_sample_percent": args.source_patient_buckets,
        "sample_patients": int(sample_count.get("patients") or 0),
        "statement_timeout": args.source_statement_timeout,
        "ok_columns": len(ok_rows),
        "failed_columns": len(failed_rows),
        "top_values": len(all_rows),
    }, {
        "source_value_genomic_regex_top_values_csv": str(output),
        "source_value_scan_status_csv": str(status_output),
    }


def create_temp_candidate_table(conn, concept_ids):
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS tmp_task17_candidate_concept")
        cursor.execute(
            """
            CREATE TEMP TABLE tmp_task17_candidate_concept(
                concept_id bigint PRIMARY KEY
            ) ON COMMIT PRESERVE ROWS
            """
        )
        cursor.executemany(
            "INSERT INTO tmp_task17_candidate_concept(concept_id) VALUES (%s)",
            [(int(concept_id),) for concept_id in concept_ids],
        )
        cursor.execute("ANALYZE tmp_task17_candidate_concept")


def measurement_chunk_presence(conn, args, columns, concept_ids, chunk_index):
    create_temp_candidate_table(conn, concept_ids)
    value_as_number = (
        sql.SQL("m.value_as_number::double precision")
        if "value_as_number" in columns else sql.SQL("NULL::double precision")
    )
    value_as_concept = (
        sql.SQL("m.value_as_concept_id::bigint")
        if "value_as_concept_id" in columns else sql.SQL("NULL::bigint")
    )
    unit_concept = (
        sql.SQL("COALESCE(m.unit_concept_id, 0)::bigint")
        if "unit_concept_id" in columns else sql.SQL("0::bigint")
    )
    source_select = sql.SQL("")
    if "measurement_source_concept_id" in columns:
        source_select = sql.SQL(
            """
            UNION ALL
            SELECT
                m.person_id,
                m.measurement_concept_id::bigint AS measurement_concept_id,
                {value_as_number} AS value_as_number,
                {value_as_concept} AS value_as_concept,
                {unit_concept} AS unit_concept,
                m.measurement_date
            FROM {schema}.measurement AS m
            JOIN tmp_task17_candidate_concept AS c
              ON m.measurement_source_concept_id::bigint = c.concept_id
            WHERE m.measurement_date <= %s::date
              AND (
                    m.measurement_concept_id IS NULL
                 OR m.measurement_concept_id::bigint <> c.concept_id
              )
            """
        ).format(
            schema=sql.Identifier(args.schema),
            value_as_number=value_as_number,
            value_as_concept=value_as_concept,
            unit_concept=unit_concept,
        )
    query = sql.SQL(
        """
        WITH matched AS (
            SELECT
                m.person_id,
                m.measurement_concept_id::bigint AS measurement_concept_id,
                {value_as_number} AS value_as_number,
                {value_as_concept} AS value_as_concept,
                {unit_concept} AS unit_concept,
                m.measurement_date
            FROM {schema}.measurement AS m
            JOIN tmp_task17_candidate_concept AS c
              ON m.measurement_concept_id::bigint = c.concept_id
            WHERE m.measurement_date <= %s::date
            {source_select}
        )
        SELECT
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT person_id)::bigint AS patients,
            COUNT(DISTINCT measurement_concept_id)::bigint AS measurement_concepts,
            COUNT(*) FILTER (WHERE value_as_number IS NOT NULL)::bigint AS numeric_rows,
            COUNT(*) FILTER (
                WHERE value_as_concept IS NOT NULL AND value_as_concept <> 0
            )::bigint AS categorical_rows,
            COUNT(*) FILTER (
                WHERE value_as_number IS NULL
                  AND value_as_concept IS NOT NULL
                  AND value_as_concept <> 0
            )::bigint AS recoverable_categorical_rows,
            COUNT(*) FILTER (
                WHERE value_as_number IS NULL
                  AND (value_as_concept IS NULL OR value_as_concept = 0)
            )::bigint AS empty_rows,
            COUNT(DISTINCT unit_concept)::bigint AS units,
            MIN(measurement_date)::text AS first_date,
            MAX(measurement_date)::text AS last_date
        FROM matched
        """
    ).format(
        schema=sql.Identifier(args.schema),
        value_as_number=value_as_number,
        value_as_concept=value_as_concept,
        unit_concept=unit_concept,
        source_select=source_select,
    )
    params = (
        (args.db_end_date, args.db_end_date)
        if "measurement_source_concept_id" in columns else
        (args.db_end_date,)
    )
    return fetch_one(
        conn,
        query,
        params,
        label=(
            f"measurement presence chunk={chunk_index} "
            f"concepts={len(concept_ids)} timeout={args.statement_timeout}"
        ),
        heartbeat_seconds=args.heartbeat_seconds,
    )


def top_measurement_concepts(conn, args, columns, candidate_ids, limit):
    if not limit:
        return []
    create_temp_candidate_table(conn, candidate_ids)
    value_as_number = (
        sql.SQL("m.value_as_number::double precision")
        if "value_as_number" in columns else sql.SQL("NULL::double precision")
    )
    value_as_concept = (
        sql.SQL("m.value_as_concept_id::bigint")
        if "value_as_concept_id" in columns else sql.SQL("NULL::bigint")
    )
    query = sql.SQL(
        """
        SELECT
            m.measurement_concept_id::bigint AS measurement_concept_id,
            co.concept_name AS measurement_concept_name,
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT m.person_id)::bigint AS patients,
            COUNT(*) FILTER (WHERE {value_as_number} IS NOT NULL)::bigint AS numeric_rows,
            COUNT(*) FILTER (
                WHERE {value_as_concept} IS NOT NULL AND {value_as_concept} <> 0
            )::bigint AS categorical_rows,
            MIN(m.measurement_date)::text AS first_date,
            MAX(m.measurement_date)::text AS last_date
        FROM {schema}.measurement AS m
        JOIN tmp_task17_candidate_concept AS c
          ON m.measurement_concept_id::bigint = c.concept_id
        LEFT JOIN {schema}.concept AS co
          ON co.concept_id = m.measurement_concept_id::bigint
        WHERE m.measurement_date <= %s::date
        GROUP BY m.measurement_concept_id, co.concept_name
        ORDER BY rows DESC, measurement_concept_id
        LIMIT %s
        """
    ).format(
        schema=sql.Identifier(args.schema),
        value_as_number=value_as_number,
        value_as_concept=value_as_concept,
    )
    return fetch_rows(
        conn,
        query,
        (args.db_end_date, limit),
        label=f"Top {limit} matched measurement concepts",
        heartbeat_seconds=args.heartbeat_seconds,
    )


def summarize_chunks(chunk_rows):
    totals = {
        "rows": 0,
        "patients_sum_across_chunks": 0,
        "measurement_concepts_sum_across_chunks": 0,
        "numeric_rows": 0,
        "categorical_rows": 0,
        "recoverable_categorical_rows": 0,
        "empty_rows": 0,
        "first_date": None,
        "last_date": None,
    }
    for row in chunk_rows:
        presence = row.get("presence") or {}
        for key in [
            "rows",
            "numeric_rows",
            "categorical_rows",
            "recoverable_categorical_rows",
            "empty_rows",
        ]:
            totals[key] += int(presence.get(key) or 0)
        totals["patients_sum_across_chunks"] += int(presence.get("patients") or 0)
        totals["measurement_concepts_sum_across_chunks"] += int(presence.get("measurement_concepts") or 0)
        first_date = presence.get("first_date")
        last_date = presence.get("last_date")
        if first_date and (totals["first_date"] is None or first_date < totals["first_date"]):
            totals["first_date"] = first_date
        if last_date and (totals["last_date"] is None or last_date > totals["last_date"]):
            totals["last_date"] = last_date
    totals["note"] = (
        "Patient and concept counts are summed across chunks and may double-count "
        "entities appearing in multiple chunks."
    )
    return totals


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if psycopg is None:
        raise RuntimeError(
            "Missing dependency: psycopg. Install with "
            "`python -m pip install \"psycopg[binary]>=3\"`."
        )
    log_handle = None
    if not args.no_log_tee:
        log_handle = install_logging(args.log_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log("[TASK17] genomic variant feasibility audit v2 starting")
    log(f"[TASK17] output_dir={args.output_dir.resolve()}")
    log(f"[TASK17] chunk_size={args.chunk_size} statement_timeout={args.statement_timeout}")

    password = os.environ.get("SNUH_CDM_PASSWORD")
    if not password:
        password = getpass.getpass("SNUH CDM password: ")

    conn = connect(args, password)
    conn.autocommit = True
    result = {
        "started_at": now_iso(),
        "application_name": APPLICATION_NAME,
        "schema": args.schema,
        "db_end_date": args.db_end_date,
        "chunk_size": args.chunk_size,
        "statement_timeout": args.statement_timeout,
        "python_version": platform.python_version(),
        "outputs": {},
    }
    checkpoint_path = args.output_dir / "measurement_presence_chunks.jsonl"
    try:
        broad_regex, gene_regex = keyword_regexes(args.keyword)
        result["broad_keyword_regex"] = broad_regex
        result["gene_keyword_regex"] = gene_regex
        columns = table_columns(conn, args.schema, "measurement")
        result["measurement_columns_checked"] = sorted(columns)
        candidate_rows = find_candidate_concepts(
            conn,
            args.schema,
            broad_regex,
            gene_regex,
            args.heartbeat_seconds,
        )
        discovered_ids = [int(row["concept_id"]) for row in candidate_rows]
        if args.scan_all_candidates:
            scan_rows = candidate_rows
        else:
            scan_rows = [row for row in candidate_rows if row.get("default_scan_candidate")]
        candidate_ids = [int(row["concept_id"]) for row in scan_rows]
        result["candidate_concepts"] = len(discovered_ids)
        result["scan_candidate_concepts"] = len(candidate_ids)
        result["scan_all_candidates"] = args.scan_all_candidates
        log(
            "[INFO] measurement scan candidates: "
            f"{len(candidate_ids)} / {len(discovered_ids)} discovered"
        )
        if args.max_scan_concepts and len(candidate_ids) > args.max_scan_concepts:
            raise RuntimeError(
                "Measurement scan candidate set is too broad: "
                f"{len(candidate_ids)} concepts exceeds --max-scan-concepts "
                f"{args.max_scan_concepts}. Refine keywords or raise the cap "
                "intentionally."
            )
        candidate_csv = args.output_dir / "candidate_molecular_concepts.csv"
        write_csv(
            candidate_csv,
            candidate_rows,
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "is_lab_test_class",
                "is_variant_ontology",
                "default_scan_candidate",
                "audit_role",
            ],
        )
        result["outputs"]["candidate_concepts_csv"] = str(candidate_csv)

        scan_candidate_csv = args.output_dir / "lab_gen_test_candidates.csv"
        write_csv(
            scan_candidate_csv,
            scan_rows,
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "is_lab_test_class",
                "is_variant_ontology",
                "default_scan_candidate",
                "audit_role",
            ],
        )
        result["outputs"]["lab_gen_test_candidates_csv"] = str(scan_candidate_csv)
        review_rows = [
            row for row in candidate_rows
            if row.get("audit_role") == "VARIANT_ONTOLOGY_REVIEW_ONLY"
        ]
        review_csv = args.output_dir / "variant_ontology_review_only.csv"
        write_csv(
            review_csv,
            review_rows,
            [
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
                "is_lab_test_class",
                "is_variant_ontology",
                "default_scan_candidate",
                "audit_role",
            ],
        )
        result["outputs"]["variant_ontology_review_only_csv"] = str(review_csv)
        if not args.db_measurement_scan:
            etl_summary, etl_outputs = summarize_with_task15_etl(
                args,
                candidate_rows,
                scan_rows,
            )
            result["task15_etl_frequency_summary"] = etl_summary
            result["outputs"].update(etl_outputs)
            if args.skip_source_value_scan:
                result["source_value_audit_summary"] = {
                    "status": "skipped",
                }
            else:
                source_summary, source_outputs = run_source_value_audit(conn, args)
                result["source_value_audit_summary"] = source_summary
                result["outputs"].update(source_outputs)
            result["finished_at"] = now_iso()
            summary_path = args.output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            log(f"[TASK17] wrote summary: {summary_path}")
            log("[TASK17] completed without DB measurement scan")
            return

        if not candidate_ids:
            log("[TASK17] no measurement scan candidates; writing empty summary")
            result["measurement_presence"] = summarize_chunks([])
            result["top_matched_measurement_concepts"] = 0
            result["finished_at"] = now_iso()
            summary_path = args.output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            log(f"[TASK17] wrote summary: {summary_path}")
            return

        completed = set() if args.no_resume else load_completed_chunks(checkpoint_path)
        log(f"[TASK17] completed_chunks_loaded={len(completed)} checkpoint={checkpoint_path}")
        for chunk_index, concept_chunk in chunks(candidate_ids, args.chunk_size):
            if chunk_index in completed:
                log(f"[SKIP] measurement presence chunk={chunk_index} already checkpointed")
                continue
            payload = {
                "chunk_index": chunk_index,
                "concept_count": len(concept_chunk),
                "concept_id_min": min(concept_chunk),
                "concept_id_max": max(concept_chunk),
                "started_at": now_iso(),
            }
            try:
                payload["presence"] = measurement_chunk_presence(
                    conn,
                    args,
                    columns,
                    concept_chunk,
                    chunk_index,
                )
                payload["status"] = "ok"
            except Exception as error:
                payload["status"] = "failed"
                payload["error"] = repr(error)
                payload["finished_at"] = now_iso()
                append_jsonl(checkpoint_path, payload)
                raise
            payload["finished_at"] = now_iso()
            append_jsonl(checkpoint_path, payload)

        chunk_rows = []
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("status") == "ok":
                            chunk_rows.append(row)
        result["measurement_presence"] = summarize_chunks(chunk_rows)
        result["outputs"]["measurement_presence_chunks_jsonl"] = str(checkpoint_path)

        matched_ids = []
        if args.top_concepts:
            for row in chunk_rows:
                if int((row.get("presence") or {}).get("rows") or 0) > 0:
                    # Optional only: this final query is intentionally disabled
                    # by default because it is not checkpointed.
                    matched_ids = candidate_ids
                    break
        if matched_ids:
            top_rows = top_measurement_concepts(
                conn,
                args,
                columns,
                matched_ids,
                args.top_concepts,
            )
        else:
            top_rows = []
        top_csv = args.output_dir / "top_matched_measurement_concepts.csv"
        write_csv(
            top_csv,
            top_rows,
            [
                "measurement_concept_id",
                "measurement_concept_name",
                "rows",
                "patients",
                "numeric_rows",
                "categorical_rows",
                "first_date",
                "last_date",
            ],
        )
        result["outputs"]["top_matched_measurement_concepts_csv"] = str(top_csv)
        result["top_matched_measurement_concepts"] = len(top_rows)
        result["finished_at"] = now_iso()
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        log(f"[TASK17] wrote summary: {summary_path}")
        log("[TASK17] completed")
    except Exception:
        log("[TASK17] FAILED; cancelling active connection if possible")
        try:
            conn.cancel()
        except Exception as cancel_error:
            log(f"[TASK17] conn.cancel() failed: {cancel_error!r}")
        traceback.print_exc()
        raise
    finally:
        try:
            conn.close()
        finally:
            if log_handle is not None:
                log_handle.flush()


if __name__ == "__main__":
    main()
