#!/usr/bin/env python3
"""Search OMOP vocabulary metadata for fasting-glucose concepts.

This script reads only ``concept`` and ``concept_relationship``.  It does not
read measurement rows or patient data.
"""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql


DEFAULT_OUTPUT_DIR = Path(
    "/home/khdp-user/workspace/fermat-data/task30/outputs/fasting_glucose_concept_search"
)
EXACT_FASTING_LOINC_CODES = ("1558-6", "14771-0", "35184-1")
REFERENCE_CONCEPT_IDS = (3004410, 3004501)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


def query_frame(conn, statement, params=None):
    with conn.cursor() as cursor:
        cursor.execute(statement, params or ())
        columns = [item.name for item in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def write_csv(frame, path):
    frame.to_csv(path, index=False)
    print(f"[WRITE] {path} rows={len(frame):,}", flush=True)


def csv_text(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().rstrip()


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name="fermat_task30_fasting_glucose_concept_search",
    )
    schema = sql.Identifier(args.schema)

    with conn:
        exact = query_frame(
            conn,
            sql.SQL(
                """
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       concept_class_id, standard_concept, concept_code,
                       valid_start_date, valid_end_date, invalid_reason
                FROM {}.concept
                WHERE vocabulary_id = 'LOINC'
                  AND concept_code = ANY(%s)
                ORDER BY concept_code, invalid_reason NULLS FIRST, concept_id
                """
            ).format(schema),
            (list(EXACT_FASTING_LOINC_CODES),),
        )

        named = query_frame(
            conn,
            sql.SQL(
                """
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       concept_class_id, standard_concept, concept_code,
                       valid_start_date, valid_end_date, invalid_reason
                FROM {}.concept
                WHERE domain_id = 'Measurement'
                  AND lower(concept_name) LIKE '%%glucose%%'
                  AND (
                       lower(concept_name) LIKE '%%fast%%'
                    OR lower(concept_name) LIKE '%%post cfst%%'
                    OR lower(concept_name) LIKE '%%calorie fast%%'
                  )
                ORDER BY invalid_reason NULLS FIRST, vocabulary_id,
                         concept_name, concept_id
                """
            ).format(schema),
        )

        mappings = query_frame(
            conn,
            sql.SQL(
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
                WHERE target.vocabulary_id = 'LOINC'
                  AND target.concept_code = ANY(%s)
                  AND r.relationship_id IN ('Maps to', 'Maps to value')
                ORDER BY target.concept_code, source.vocabulary_id,
                         source.concept_code, source.concept_id
                """
            ).format(schema, schema, schema),
            (list(EXACT_FASTING_LOINC_CODES),),
        )

        references = query_frame(
            conn,
            sql.SQL(
                """
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       concept_class_id, standard_concept, concept_code,
                       valid_start_date, valid_end_date, invalid_reason
                FROM {}.concept
                WHERE concept_id = ANY(%s)
                ORDER BY concept_id
                """
            ).format(schema),
            (list(REFERENCE_CONCEPT_IDS),),
        )

    conn.close()

    write_csv(exact, args.output_dir / "exact_fasting_loinc_lookup.csv")
    write_csv(named, args.output_dir / "fasting_glucose_name_search.csv")
    write_csv(mappings, args.output_dir / "source_to_fasting_loinc_mappings.csv")
    write_csv(references, args.output_dir / "task20_reference_concepts.csv")

    valid_exact = exact.loc[
        exact["invalid_reason"].isna()
        & exact["domain_id"].eq("Measurement")
        & exact["standard_concept"].fillna("").eq("S")
    ]
    active_mappings = mappings.loc[mappings["relationship_invalid_reason"].isna()]
    status = {
        "valid_exact_fasting_loinc_concepts": int(len(valid_exact)),
        "fasting_glucose_name_matches": int(len(named)),
        "active_source_concepts_mapping_to_exact_fasting_loinc": int(
            active_mappings["source_concept_id"].nunique()
        ),
        "measurement_or_patient_tables_read": False,
        "interpretation": (
            "Vocabulary existence does not prove that SNUH measurement rows use the concept. "
            "Occurrence in patient data must be checked separately."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            "## STATUS",
            "CONCEPT_METADATA_SEARCH_COMPLETE",
            "## SUMMARY",
            json.dumps(status, ensure_ascii=False, indent=2),
            "## EXACT_FASTING_LOINC_LOOKUP",
            csv_text(exact),
            "## TASK20_REFERENCE_CONCEPTS",
            csv_text(references),
            "## FASTING_GLUCOSE_NAME_SEARCH",
            csv_text(named),
            "## SOURCE_TO_EXACT_FASTING_LOINC_MAPPINGS",
            csv_text(mappings),
        ]
    ) + "\n"
    summary_path = args.output_dir / "return_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
