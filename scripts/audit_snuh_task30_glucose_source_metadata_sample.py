#!/usr/bin/env python3
"""Fast source-code audit for SNUH generic glucose (OMOP concept 3004501).

The audit avoids an exhaustive GROUP BY over the measurement table.  It:

1. searches vocabulary mapping metadata exhaustively; and
2. reads a bounded SYSTEM sample of measurement storage blocks.

No patient identifiers are selected or written.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
except (ImportError, ModuleNotFoundError):  # self-test does not need pandas
    pd = None

try:
    import psycopg
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # self-test does not need psycopg
    psycopg = None
    sql = None


GENERIC_GLUCOSE_CONCEPT_ID = 3004501
DEFAULT_OUTPUT_DIR = Path(
    "/home/khdp-user/workspace/fermat-data/task30/outputs/glucose_source_metadata_sample"
)
FASTING_PATTERNS = (
    ("FAST_WORD", re.compile(r"\bfast(?:ing)?\b", re.IGNORECASE)),
    ("FBS_TOKEN", re.compile(r"(^|[^A-Z0-9])FBS([^A-Z0-9]|$)", re.IGNORECASE)),
    ("FBG_TOKEN", re.compile(r"(^|[^A-Z0-9])FBG([^A-Z0-9]|$)", re.IGNORECASE)),
    ("KOREAN_FASTING", re.compile(r"공복")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2012-01-01")
    parser.add_argument("--end-date", default="2025-02-05")
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=0.1,
        help="Percentage of physical measurement blocks sampled with TABLESAMPLE SYSTEM.",
    )
    parser.add_argument("--sample-row-limit", type=int, default=100000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "SNUH_CDM_HOST", "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
        ),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432"))
    )
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument(
        "--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung")
    )
    parser.add_argument(
        "--schema", default=os.environ.get("SNUH_CDM_SCHEMA", "cdm2024_official")
    )
    parser.add_argument(
        "--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable")
    )
    parser.add_argument(
        "--statement-timeout",
        default="120000",
        help="Per-query PostgreSQL timeout in milliseconds; default is 2 minutes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def fasting_evidence(*values):
    text = " | ".join("" if value is None else str(value) for value in values)
    return "|".join(name for name, pattern in FASTING_PATTERNS if pattern.search(text))


def self_test():
    assert fasting_evidence("FBS") == "FBS_TOKEN"
    assert fasting_evidence("fasting glucose") == "FAST_WORD"
    assert fasting_evidence("공복혈당") == "KOREAN_FASTING"
    assert fasting_evidence("Glucose in Serum or Plasma") == ""
    print("SELF_TEST_OK")


def require_dependencies():
    missing = []
    if pd is None:
        missing.append("pandas")
    if psycopg is None:
        missing.append("psycopg")
    if missing:
        raise RuntimeError("Missing Pod Python dependencies: " + ", ".join(missing))


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


def query_frame(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        print(f"[START] {label}", flush=True)
    with conn.cursor() as cursor:
        cursor.execute(statement, params or ())
        columns = [item.name for item in cursor.description]
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if label:
        print(
            f"[DONE] {label}: rows={len(frame):,}, seconds={time.time() - started:,.1f}",
            flush=True,
        )
    return frame


def write_csv(frame, path):
    frame.to_csv(path, index=False)
    print(f"[WRITE] {path} rows={len(frame):,}", flush=True)


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().rstrip()


def classify(frame, columns):
    result = frame.copy()
    if result.empty:
        result["explicit_fasting_evidence"] = pd.Series(dtype=str)
        result["explicit_fasting_label"] = pd.Series(dtype=bool)
        return result
    result["explicit_fasting_evidence"] = result.apply(
        lambda row: fasting_evidence(*(row.get(column) for column in columns)),
        axis=1,
    )
    result["explicit_fasting_label"] = result["explicit_fasting_evidence"].ne("")
    return result


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require_dependencies()
    if not (0 < args.sample_percent <= 10):
        raise ValueError("--sample-percent must be greater than 0 and at most 10")
    if args.sample_row_limit < 1:
        raise ValueError("--sample-row-limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    try:
        conn = psycopg.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=password(),
            sslmode=args.sslmode,
            connect_timeout=15,
            application_name="fermat_task30_glucose_source_metadata_sample",
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=6,
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (args.statement_timeout,),
            )
        conn.commit()
        schema = sql.Identifier(args.schema)

        indexes = query_frame(
            conn,
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = %s AND tablename = 'measurement'
            ORDER BY indexname
            """,
            (args.schema,),
            label="inspect measurement indexes",
        )
        write_csv(indexes, args.output_dir / "measurement_indexes.csv")

        relationships = query_frame(
            conn,
            sql.SQL(
                """
                SELECT source.concept_id AS source_concept_id,
                       source.concept_name AS source_concept_name,
                       source.vocabulary_id AS source_vocabulary_id,
                       source.concept_code AS source_concept_code,
                       source.standard_concept AS source_standard_concept,
                       source.invalid_reason AS source_invalid_reason,
                       r.relationship_id,
                       r.invalid_reason AS relationship_invalid_reason,
                       target.concept_id AS target_concept_id,
                       target.concept_name AS target_concept_name,
                       target.vocabulary_id AS target_vocabulary_id,
                       target.concept_code AS target_concept_code
                FROM {}.concept_relationship r
                JOIN {}.concept source ON source.concept_id = r.concept_id_1
                JOIN {}.concept target ON target.concept_id = r.concept_id_2
                WHERE r.concept_id_2 = %s
                  AND r.relationship_id IN ('Maps to', 'Maps to value')
                ORDER BY source.vocabulary_id, source.concept_code,
                         source.concept_id, r.relationship_id
                """
            ).format(schema, schema, schema),
            (GENERIC_GLUCOSE_CONCEPT_ID,),
            label="find concepts mapped to generic glucose",
        )
        relationships = classify(
            relationships,
            ["source_concept_name", "source_concept_code"],
        )
        write_csv(relationships, args.output_dir / "concept_relationship_to_generic_glucose.csv")

        table_check = query_frame(
            conn,
            "SELECT to_regclass(%s) IS NOT NULL AS table_exists",
            (f"{args.schema}.source_to_concept_map",),
            label="check source_to_concept_map availability",
        )
        has_source_map = bool(table_check.iloc[0]["table_exists"])
        if has_source_map:
            source_map = query_frame(
                conn,
                sql.SQL(
                    """
                    SELECT source_code, source_concept_id, source_vocabulary_id,
                           source_code_description, target_concept_id,
                           target_vocabulary_id, valid_start_date,
                           valid_end_date, invalid_reason
                    FROM {}.source_to_concept_map
                    WHERE target_concept_id = %s
                    ORDER BY source_vocabulary_id, source_code, source_concept_id
                    """
                ).format(schema),
                (GENERIC_GLUCOSE_CONCEPT_ID,),
                label="find source codes mapped to generic glucose",
            )
        else:
            source_map = pd.DataFrame(
                columns=[
                    "source_code", "source_concept_id", "source_vocabulary_id",
                    "source_code_description", "target_concept_id",
                    "target_vocabulary_id", "valid_start_date", "valid_end_date",
                    "invalid_reason",
                ]
            )
        source_map = classify(source_map, ["source_code", "source_code_description"])
        write_csv(source_map, args.output_dir / "source_to_concept_map_generic_glucose.csv")

        sample_statement = sql.SQL(
            """
            SELECT m.measurement_source_value,
                   COALESCE(m.measurement_source_concept_id, 0)::bigint
                       AS measurement_source_concept_id,
                   source.concept_name AS source_concept_name,
                   source.vocabulary_id AS source_vocabulary_id,
                   source.concept_code AS source_concept_code,
                   m.unit_concept_id::bigint,
                   unit.concept_name AS unit_concept_name,
                   m.unit_source_value
            FROM {}.measurement m
                 TABLESAMPLE SYSTEM ({}) REPEATABLE ({})
            LEFT JOIN {}.concept source
              ON source.concept_id = m.measurement_source_concept_id
            LEFT JOIN {}.concept unit
              ON unit.concept_id = m.unit_concept_id
            WHERE m.measurement_concept_id = %s
              AND m.measurement_date BETWEEN %s::date AND %s::date
            LIMIT %s
            """
        ).format(
            schema,
            sql.Literal(args.sample_percent),
            sql.Literal(args.sample_seed),
            schema,
            schema,
        )
        sample = query_frame(
            conn,
            sample_statement,
            (
                GENERIC_GLUCOSE_CONCEPT_ID,
                args.start_date,
                args.end_date,
                args.sample_row_limit,
            ),
            label=f"sample {args.sample_percent}% of measurement storage blocks",
        )
        conn.close()

        sample = classify(
            sample,
            ["measurement_source_value", "source_concept_name", "source_concept_code"],
        )
        group_columns = [
            "measurement_source_value", "measurement_source_concept_id",
            "source_concept_name", "source_vocabulary_id", "source_concept_code",
            "unit_concept_id", "unit_concept_name", "unit_source_value",
            "explicit_fasting_evidence", "explicit_fasting_label",
        ]
        if sample.empty:
            sample_summary = pd.DataFrame(columns=group_columns + ["sample_rows"])
        else:
            sample_summary = (
                sample.groupby(group_columns, dropna=False)
                .size()
                .rename("sample_rows")
                .reset_index()
                .sort_values("sample_rows", ascending=False)
            )
        write_csv(sample_summary, args.output_dir / "sampled_measurement_source_values.csv")

        metadata_evidence = pd.concat(
            [
                relationships.loc[relationships["explicit_fasting_label"]].assign(
                    evidence_source="concept_relationship"
                ),
                source_map.loc[source_map["explicit_fasting_label"]].assign(
                    evidence_source="source_to_concept_map"
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        sample_evidence = sample_summary.loc[
            sample_summary["explicit_fasting_label"].fillna(False)
        ].copy()
        status = (
            "EXPLICIT_FASTING_SOURCE_FOUND"
            if not metadata_evidence.empty or not sample_evidence.empty
            else "EXPLICIT_FASTING_SOURCE_NOT_IDENTIFIED_IN_METADATA_OR_BOUNDED_SAMPLE"
        )
        summary = {
            "status": status,
            "generic_glucose_concept_id": GENERIC_GLUCOSE_CONCEPT_ID,
            "concept_relationship_rows": int(len(relationships)),
            "source_to_concept_map_table_exists": has_source_map,
            "source_to_concept_map_rows": int(len(source_map)),
            "sample_percent_of_storage_blocks": args.sample_percent,
            "sampled_generic_glucose_rows": int(len(sample)),
            "sampled_source_combinations": int(len(sample_summary)),
            "explicit_metadata_candidates": int(len(metadata_evidence)),
            "explicit_sample_candidates": int(len(sample_evidence)),
            "patient_identifiers_selected_or_written": False,
            "claim_boundary": (
                "A positive explicit label supports a fasting subset. "
                "A negative bounded sample does not prove that no fasting source code exists anywhere."
            ),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sections = [
            "## STATUS",
            status,
            "## SUMMARY",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "## EXPLICIT_METADATA_CANDIDATES",
            csv_text(metadata_evidence),
            "## EXPLICIT_SAMPLE_CANDIDATES",
            csv_text(sample_evidence),
            "## ALL_CONCEPT_RELATIONSHIPS_TO_GENERIC_GLUCOSE",
            csv_text(relationships),
            "## SOURCE_TO_CONCEPT_MAP_ROWS",
            csv_text(source_map),
            "## SAMPLED_SOURCE_VALUES",
            csv_text(sample_summary),
            "## MEASUREMENT_INDEXES",
            csv_text(indexes),
        ]
        text = "\n".join(sections) + "\n"
        (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
        print(text, end="", flush=True)
        print(f"[COMPLETE] {args.output_dir}", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "return_summary.txt").write_text(
            "## STATUS\nFAILED\n"
            f"error_type={type(exc).__name__}\n"
            f"error={exc}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
