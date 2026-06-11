"""Measure one-scan measurement staging feasibility inside PostgreSQL."""

import argparse
import getpass
import json
import os
import platform
import time
from pathlib import Path

import psycopg
from psycopg import sql


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="pg-2vge6u.vpc-cdb-kr.gov-ntruss.com")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="cdm")
    parser.add_argument("--user", default="jaegyun_jung")
    parser.add_argument("--schema", default="cdm2024_official")
    parser.add_argument("--sslmode", default="disable")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pilot-seed", type=int, default=20260611)
    parser.add_argument("--patient-buckets", type=int, default=1)
    parser.add_argument("--create-indexes", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/snuh_measurement_staging_audit.json"),
    )
    return parser.parse_args()


def fetch_one(conn, query, params=None):
    with conn.cursor() as cursor:
        cursor.execute(query, params or ())
        columns = [column.name for column in cursor.description]
        return dict(zip(columns, cursor.fetchone()))


def execute_timed(conn, query, params=None):
    started = time.time()
    with conn.cursor() as cursor:
        cursor.execute(query, params or ())
    return time.time() - started


def main():
    args = parse_args()
    if not 1 <= args.patient_buckets <= 100:
        raise ValueError("patient-buckets must be between 1 and 100")
    password = os.environ.get("SNUH_CDM_PASSWORD")
    if not password:
        password = getpass.getpass("SNUH CDM password: ")

    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password,
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name="fermat_measurement_staging_audit",
        options="-c statement_timeout=0 -c temp_buffers=256MB",
    )
    conn.autocommit = True
    result = {
        "schema": args.schema,
        "db_end_date": args.db_end_date,
        "split_seed": args.split_seed,
        "pilot_seed": args.pilot_seed,
        "patient_buckets": args.patient_buckets,
        "create_indexes": args.create_indexes,
        "python_version": platform.python_version(),
    }

    try:
        execute_timed(conn, "DROP TABLE IF EXISTS tmp_fermat_stage_person")
        result["cohort_create_seconds"] = execute_timed(
            conn,
            sql.SQL("""
            CREATE TEMP TABLE tmp_fermat_stage_person
            ON COMMIT PRESERVE ROWS AS
            WITH assigned AS (
                SELECT
                    person_id,
                    mod(
                        hashtextextended(person_id::text, %s)
                            & 9223372036854775807,
                        100
                    )::integer AS split_bucket,
                    mod(
                        hashtextextended(person_id::text, %s)
                            & 9223372036854775807,
                        100
                    )::integer AS pilot_bucket
                FROM {}.person
            )
            SELECT
                person_id,
                CASE
                    WHEN split_bucket < 70 THEN 'train'
                    WHEN split_bucket < 85 THEN 'val'
                    ELSE 'test'
                END AS split
            FROM assigned
            WHERE pilot_bucket < %s
            """).format(sql.Identifier(args.schema)),
            (args.split_seed, args.pilot_seed, args.patient_buckets),
        )
        execute_timed(
            conn,
            "CREATE INDEX ON tmp_fermat_stage_person(person_id)",
        )
        execute_timed(conn, "ANALYZE tmp_fermat_stage_person")
        result["cohort"] = fetch_one(
            conn,
            """
            SELECT COUNT(*)::bigint AS patients
            FROM tmp_fermat_stage_person
            """,
        )

        execute_timed(conn, "DROP TABLE IF EXISTS tmp_measurement_stage")
        result["stage_create_seconds"] = execute_timed(
            conn,
            sql.SQL("""
            CREATE TEMP TABLE tmp_measurement_stage
            ON COMMIT PRESERVE ROWS AS
            SELECT
                m.person_id,
                p.split,
                m.measurement_date,
                m.measurement_concept_id::bigint,
                COALESCE(m.unit_concept_id, 0)::bigint AS unit_concept_id,
                m.value_as_number::double precision,
                m.value_as_concept_id::bigint
            FROM {}.measurement AS m
            JOIN tmp_fermat_stage_person AS p USING (person_id)
            WHERE m.measurement_date
                    BETWEEN DATE '1900-01-01' AND %s::date
              AND m.measurement_concept_id IS NOT NULL
              AND m.measurement_concept_id <> 0
              AND (
                    m.value_as_number IS NOT NULL
                    OR (
                        m.value_as_concept_id IS NOT NULL
                        AND m.value_as_concept_id <> 0
                    )
              )
            """).format(sql.Identifier(args.schema)),
            (args.db_end_date,),
        )
        result["stage_analyze_seconds"] = execute_timed(
            conn,
            "ANALYZE tmp_measurement_stage",
        )

        if args.create_indexes:
            started = time.time()
            execute_timed(
                conn,
                "CREATE INDEX ON tmp_measurement_stage(person_id, measurement_date)",
            )
            execute_timed(
                conn,
                """
                CREATE INDEX ON tmp_measurement_stage(
                    split, measurement_concept_id, unit_concept_id
                ) WHERE value_as_number IS NOT NULL
                """,
            )
            execute_timed(
                conn,
                """
                CREATE INDEX ON tmp_measurement_stage(
                    split, measurement_concept_id, value_as_concept_id
                ) WHERE value_as_concept_id IS NOT NULL
                  AND value_as_concept_id <> 0
                """,
            )
            result["stage_index_seconds"] = time.time() - started

        result["stage_counts"] = fetch_one(
            conn,
            """
            SELECT
                COUNT(*)::bigint AS rows,
                COUNT(*) FILTER (
                    WHERE value_as_number IS NOT NULL
                )::bigint AS numeric_rows,
                COUNT(*) FILTER (
                    WHERE value_as_concept_id IS NOT NULL
                      AND value_as_concept_id <> 0
                )::bigint AS categorical_rows,
                COUNT(*) FILTER (
                    WHERE value_as_number IS NOT NULL
                      AND value_as_concept_id IS NOT NULL
                      AND value_as_concept_id <> 0
                )::bigint AS numeric_and_categorical_rows,
                COUNT(DISTINCT person_id)::bigint AS patients
            FROM tmp_measurement_stage
            """,
        )
        result["stage_storage"] = fetch_one(
            conn,
            """
            SELECT
                pg_relation_size('tmp_measurement_stage'::regclass)::bigint
                    AS table_bytes,
                pg_indexes_size('tmp_measurement_stage'::regclass)::bigint
                    AS index_bytes,
                pg_total_relation_size('tmp_measurement_stage'::regclass)::bigint
                    AS total_bytes
            """,
        )
        for key in ("table_bytes", "index_bytes", "total_bytes"):
            result["stage_storage"][key.replace("_bytes", "_gb")] = (
                result["stage_storage"][key] / 1024**3
            )

        multiplier = 100 / args.patient_buckets
        result["linear_full_scale_projection"] = {
            "rows": int(result["stage_counts"]["rows"] * multiplier),
            "table_gb": result["stage_storage"]["table_gb"] * multiplier,
            "index_gb": result["stage_storage"]["index_gb"] * multiplier,
            "total_gb": result["stage_storage"]["total_gb"] * multiplier,
            "warning": "Linear projection only; verify with larger patient buckets.",
        }
    finally:
        conn.close()
        password = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
