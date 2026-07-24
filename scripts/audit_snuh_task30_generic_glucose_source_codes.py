#!/usr/bin/env python3
"""Inspect source test codes behind SNUH's generic serum/plasma glucose concept.

The script reads only measurement rows whose standard concept is LOINC 2345-7
(``measurement_concept_id=3004501``), aggregates their source-code metadata,
and flags explicit fasting labels.  It does not export patient-level rows and
does not infer fasting from the numeric value or collection time.
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
except (ImportError, ModuleNotFoundError):  # self-test does not require pandas
    pd = None

try:
    import psycopg
    from psycopg import sql
except (ImportError, ModuleNotFoundError):  # self-test does not require psycopg
    psycopg = None
    sql = None


GENERIC_GLUCOSE_CONCEPT_ID = 3004501
DEFAULT_OUTPUT_DIR = Path(
    "/home/khdp-user/workspace/fermat-data/task30/outputs/generic_glucose_source_audit"
)

# These patterns are deliberately conservative.  Collection time, a glucose
# value >=126, or a health-check visit alone is not accepted as proof of fasting.
EXPLICIT_FASTING_PATTERNS = (
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
        default="900000",
        help="PostgreSQL timeout in milliseconds; default is 15 minutes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def explicit_fasting_evidence(*values):
    text = " | ".join("" if value is None else str(value) for value in values)
    matches = [name for name, pattern in EXPLICIT_FASTING_PATTERNS if pattern.search(text)]
    return "|".join(matches)


def self_test():
    assert explicit_fasting_evidence("FBS") == "FBS_TOKEN"
    assert explicit_fasting_evidence("fasting glucose") == "FAST_WORD"
    assert explicit_fasting_evidence("공복혈당") == "KOREAN_FASTING"
    assert explicit_fasting_evidence("Glucose in Serum or Plasma") == ""
    assert explicit_fasting_evidence("blood sugar") == ""
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


def source_inventory_query(schema):
    return sql.SQL(
        """
        SELECT
            COALESCE(NULLIF(btrim(m.measurement_source_value), ''), '<EMPTY>')
                AS measurement_source_value,
            COALESCE(m.measurement_source_concept_id, 0)::bigint
                AS measurement_source_concept_id,
            source.concept_name AS source_concept_name,
            source.vocabulary_id AS source_vocabulary_id,
            source.concept_code AS source_concept_code,
            source.standard_concept AS source_standard_concept,
            source.invalid_reason AS source_invalid_reason,
            m.unit_concept_id::bigint,
            unit.concept_name AS unit_concept_name,
            COALESCE(NULLIF(btrim(m.unit_source_value), ''), '<EMPTY>')
                AS unit_source_value,
            COUNT(*)::bigint AS measurement_rows,
            MIN(m.measurement_date)::text AS first_measurement_date,
            MAX(m.measurement_date)::text AS last_measurement_date
        FROM {}.measurement m
        LEFT JOIN {}.concept source
          ON source.concept_id = m.measurement_source_concept_id
        LEFT JOIN {}.concept unit
          ON unit.concept_id = m.unit_concept_id
        WHERE m.measurement_concept_id = %s
          AND m.measurement_date BETWEEN %s::date AND %s::date
        GROUP BY 1,2,3,4,5,6,7,8,9,10
        ORDER BY measurement_rows DESC,
                 measurement_source_value,
                 measurement_source_concept_id,
                 unit_concept_id
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema), sql.Identifier(schema))


def source_relationship_query(schema, source_ids):
    if not source_ids:
        return pd.DataFrame(
            columns=[
                "source_concept_id", "source_concept_name", "source_vocabulary_id",
                "source_concept_code", "relationship_id", "target_concept_id",
                "target_concept_name", "target_vocabulary_id", "target_concept_code",
                "relationship_invalid_reason",
            ]
        )
    return sql.SQL(
        """
        SELECT source.concept_id AS source_concept_id,
               source.concept_name AS source_concept_name,
               source.vocabulary_id AS source_vocabulary_id,
               source.concept_code AS source_concept_code,
               r.relationship_id,
               target.concept_id AS target_concept_id,
               target.concept_name AS target_concept_name,
               target.vocabulary_id AS target_vocabulary_id,
               target.concept_code AS target_concept_code,
               r.invalid_reason AS relationship_invalid_reason
        FROM {}.concept_relationship r
        JOIN {}.concept source ON source.concept_id = r.concept_id_1
        JOIN {}.concept target ON target.concept_id = r.concept_id_2
        WHERE r.concept_id_1 = ANY(%s)
          AND r.relationship_id IN ('Maps to', 'Maps to value')
        ORDER BY source.concept_id, r.relationship_id,
                 target.vocabulary_id, target.concept_code, target.concept_id
        """
    ).format(sql.Identifier(schema), sql.Identifier(schema), sql.Identifier(schema))


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require_dependencies()
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
            application_name="fermat_task30_generic_glucose_source_audit",
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

        inventory = query_frame(
            conn,
            source_inventory_query(args.schema),
            (GENERIC_GLUCOSE_CONCEPT_ID, args.start_date, args.end_date),
            label="aggregate source codes for generic serum/plasma glucose",
        )
        write_csv(inventory, args.output_dir / "generic_glucose_source_inventory_raw.csv")

        evidence_columns = [
            "measurement_source_value",
            "source_concept_name",
            "source_concept_code",
        ]
        inventory["explicit_fasting_evidence"] = inventory.apply(
            lambda row: explicit_fasting_evidence(
                *(row.get(column) for column in evidence_columns)
            ),
            axis=1,
        )
        inventory["explicit_fasting_label"] = inventory[
            "explicit_fasting_evidence"
        ].ne("")
        write_csv(inventory, args.output_dir / "generic_glucose_source_inventory_classified.csv")

        explicit = inventory.loc[inventory["explicit_fasting_label"]].copy()
        explicit = explicit.sort_values("measurement_rows", ascending=False)
        write_csv(explicit, args.output_dir / "explicit_fasting_source_candidates.csv")

        source_ids = (
            pd.to_numeric(inventory["measurement_source_concept_id"], errors="coerce")
            .dropna()
            .astype("int64")
        )
        source_ids = sorted(set(int(value) for value in source_ids if int(value) != 0))
        relationship_statement = source_relationship_query(args.schema, source_ids)
        if isinstance(relationship_statement, pd.DataFrame):
            relationships = relationship_statement
        else:
            relationships = query_frame(
                conn,
                relationship_statement,
                (source_ids,),
                label="resolve mappings from observed source concepts",
            )
        write_csv(relationships, args.output_dir / "observed_source_concept_mappings.csv")
        conn.close()

        total_rows = int(inventory["measurement_rows"].sum()) if not inventory.empty else 0
        explicit_rows = int(explicit["measurement_rows"].sum()) if not explicit.empty else 0
        summary = {
            "period": [args.start_date, args.end_date],
            "generic_glucose_concept_id": GENERIC_GLUCOSE_CONCEPT_ID,
            "generic_glucose_source_combinations": int(len(inventory)),
            "generic_glucose_measurement_rows": total_rows,
            "explicit_fasting_source_combinations": int(len(explicit)),
            "explicit_fasting_labeled_rows": explicit_rows,
            "explicit_fasting_labeled_row_fraction": (
                explicit_rows / total_rows if total_rows else None
            ),
            "exact_fasting_concept_usage_requeried": False,
            "patient_rows_exported": False,
            "rule": (
                "Only explicit source labels containing fasting/fast/FBS/FBG/공복 are flagged. "
                "Numeric value, collection time, and generic health-check context are not used as proof."
            ),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        top_inventory = inventory.head(50)
        sections = [
            "## STATUS",
            "GENERIC_GLUCOSE_SOURCE_AUDIT_COMPLETE",
            "## SUMMARY",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "## EXPLICIT_FASTING_SOURCE_CANDIDATES",
            csv_text(explicit),
            "## TOP_50_GENERIC_GLUCOSE_SOURCE_COMBINATIONS",
            csv_text(top_inventory),
            "## OBSERVED_SOURCE_CONCEPT_MAPPINGS",
            csv_text(relationships),
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
