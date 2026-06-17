#!/usr/bin/env python3
"""Profile Task 17 source/value columns from a terminal session.

This is a small diagnostic script. It does not create genomic tokens and does
not use marker regex by default. The default run only fetches first examples
and their mapped concepts. Aggregates are opt-in because they can be slow on
large CDM tables.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import json
import os
import sys
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
APPLICATION_NAME = "fermat_task17_source_value_profile"

TARGETS = [
    {
        "label": "measurement.measurement_source_value",
        "table": "measurement",
        "date_col": "measurement_date",
        "value_col": "measurement_source_value",
        "concept_col": "measurement_concept_id",
        "source_concept_col": "measurement_source_concept_id",
        "value_as_concept_col": "value_as_concept_id",
    },
    {
        "label": "measurement.value_source_value",
        "table": "measurement",
        "date_col": "measurement_date",
        "value_col": "value_source_value",
        "concept_col": "measurement_concept_id",
        "source_concept_col": "measurement_source_concept_id",
        "value_as_concept_col": "value_as_concept_id",
    },
    {
        "label": "procedure_occurrence.procedure_source_value",
        "table": "procedure_occurrence",
        "date_col": "procedure_date",
        "value_col": "procedure_source_value",
        "concept_col": "procedure_concept_id",
        "source_concept_col": "procedure_source_concept_id",
        "value_as_concept_col": None,
    },
    {
        "label": "observation.observation_source_value",
        "table": "observation",
        "date_col": "observation_date",
        "value_col": "observation_source_value",
        "concept_col": "observation_concept_id",
        "source_concept_col": "observation_source_concept_id",
        "value_as_concept_col": "value_as_concept_id",
    },
    {
        "label": "observation.value_source_value",
        "table": "observation",
        "date_col": "observation_date",
        "value_col": "value_source_value",
        "concept_col": "observation_concept_id",
        "source_concept_col": "observation_source_concept_id",
        "value_as_concept_col": "value_as_concept_id",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--sample-per-mille", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--top-limit", type=int, default=80)
    parser.add_argument("--statement-timeout", default="10min")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task17_source_value_profile"))
    parser.add_argument(
        "--with-aggregate",
        action="store_true",
        help="Also run COUNT/GROUP BY profiles. Default only writes first examples.",
    )
    return parser.parse_args()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_rows(title, rows, max_rows=30):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    headers = list(rows[0].keys())
    print("\t".join(headers))
    for row in rows[:max_rows]:
        print("\t".join("" if row.get(h) is None else str(row.get(h)) for h in headers))


def connect(args):
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install psycopg in the Pod environment.")
    password = os.environ.get("SNUH_CDM_PASSWORD") or getpass.getpass("SNUH CDM password: ")
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password,
        sslmode=args.sslmode,
        application_name=APPLICATION_NAME,
    )


def fetch_dicts(conn, query, params=()):
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def execute(conn, query, params=()):
    with conn.cursor() as cursor:
        cursor.execute(query, params)


def set_timeout(conn, timeout):
    execute(conn, "SELECT set_config('statement_timeout', %s, false)", (timeout,))


def create_sample(conn, args):
    if not 0 < args.sample_per_mille <= 1000:
        raise ValueError("--sample-per-mille must be between 1 and 1000")
    log(f"creating deterministic patient sample: {args.sample_per_mille}/1000")
    execute(conn, "DROP TABLE IF EXISTS tmp_task17_profile_person")
    query = sql.SQL(
        """
        CREATE TEMP TABLE tmp_task17_profile_person ON COMMIT PRESERVE ROWS AS
        SELECT person_id
        FROM {}.person
        WHERE mod(
            hashtextextended(person_id::text, %s) & 9223372036854775807,
            1000
        ) < %s
        """
    ).format(sql.Identifier(args.schema))
    execute(conn, query, (args.seed, args.sample_per_mille))
    execute(conn, "CREATE INDEX ON tmp_task17_profile_person(person_id)")
    execute(conn, "ANALYZE tmp_task17_profile_person")
    rows = fetch_dicts(conn, "SELECT COUNT(*)::bigint AS patients FROM tmp_task17_profile_person")
    print_rows("sample patients", rows)
    return rows[0]["patients"]


def value_expr(target):
    return sql.SQL("t.{}::text").format(sql.Identifier(target["value_col"]))


def concept_selects(target):
    fields = [
        sql.SQL("t.{}::bigint AS concept_id").format(sql.Identifier(target["concept_col"])),
        sql.SQL("c.concept_name AS concept_name"),
        sql.SQL("c.domain_id AS domain_id"),
        sql.SQL("c.vocabulary_id AS vocabulary_id"),
        sql.SQL("c.concept_class_id AS concept_class_id"),
        sql.SQL("t.{}::bigint AS source_concept_id").format(sql.Identifier(target["source_concept_col"])),
        sql.SQL("sc.concept_name AS source_concept_name"),
        sql.SQL("sc.vocabulary_id AS source_vocabulary_id"),
    ]
    if target["value_as_concept_col"]:
        fields.extend(
            [
                sql.SQL("t.{}::bigint AS value_as_concept_id").format(
                    sql.Identifier(target["value_as_concept_col"])
                ),
                sql.SQL("vc.concept_name AS value_as_concept_name"),
            ]
        )
    else:
        fields.extend(
            [
                sql.SQL("NULL::bigint AS value_as_concept_id"),
                sql.SQL("NULL::text AS value_as_concept_name"),
            ]
        )
    return sql.SQL(",\n        ").join(fields)


def value_as_concept_join(target, schema):
    if not target["value_as_concept_col"]:
        return sql.SQL("")
    return sql.SQL("LEFT JOIN {}.concept vc ON vc.concept_id = t.{}::bigint").format(
        sql.Identifier(schema),
        sql.Identifier(target["value_as_concept_col"]),
    )


def value_as_concept_group_by(target):
    if not target["value_as_concept_col"]:
        return sql.SQL("NULL::bigint, NULL::text")
    return sql.SQL("t.{}::bigint, vc.concept_name").format(
        sql.Identifier(target["value_as_concept_col"])
    )


def first_examples(conn, args, target):
    label = target["label"]
    log(f"first examples: {label}")
    query = sql.SQL(
        """
        WITH picked AS MATERIALIZED (
            SELECT
                t.person_id,
                {value_expr} AS value,
                t.{concept_col}::bigint AS concept_id,
                t.{source_concept_col}::bigint AS source_concept_id,
                {value_as_concept_expr} AS value_as_concept_id,
                t.{date_col}::text AS event_date
            FROM {schema}.{table} t
            WHERE t.{date_col} BETWEEN DATE '1900-01-01' AND %s::date
              AND {value_expr} IS NOT NULL
              AND NULLIF({value_expr}, '') IS NOT NULL
            LIMIT %s
        )
        SELECT
            picked.value,
            length(picked.value) AS value_len,
            picked.concept_id,
            c.concept_name AS concept_name,
            c.domain_id AS domain_id,
            c.vocabulary_id AS vocabulary_id,
            c.concept_class_id AS concept_class_id,
            picked.source_concept_id,
            sc.concept_name AS source_concept_name,
            sc.vocabulary_id AS source_vocabulary_id,
            picked.value_as_concept_id,
            vc.concept_name AS value_as_concept_name,
            picked.event_date
        FROM picked
        LEFT JOIN {schema}.concept c ON c.concept_id = picked.concept_id
        LEFT JOIN {schema}.concept sc ON sc.concept_id = picked.source_concept_id
        LEFT JOIN {schema}.concept vc ON vc.concept_id = picked.value_as_concept_id
        """
    ).format(
        value_expr=value_expr(target),
        value_as_concept_expr=(
            sql.SQL("t.{}::bigint").format(sql.Identifier(target["value_as_concept_col"]))
            if target["value_as_concept_col"]
            else sql.SQL("NULL::bigint")
        ),
        schema=sql.Identifier(args.schema),
        table=sql.Identifier(target["table"]),
        date_col=sql.Identifier(target["date_col"]),
        concept_col=sql.Identifier(target["concept_col"]),
        source_concept_col=sql.Identifier(target["source_concept_col"]),
    )
    rows = fetch_dicts(conn, query, (args.db_end_date, args.limit))
    write_tsv(args.output_dir / f"{label}.examples.tsv", rows)
    print_rows(f"{label} first {args.limit} non-null examples", rows)
    return rows


def column_profile(conn, args, target):
    label = target["label"]
    log(f"column profile: {label}")
    v = value_expr(target)
    query = sql.SQL(
        """
        SELECT
            %s::text AS label,
            COUNT(*)::bigint AS non_null_rows,
            COUNT(*) FILTER (WHERE {v} ~ '^[0-9]+(\\.[0-9]+)?$')::bigint AS numeric_like_rows,
            COUNT(*) FILTER (WHERE {v} ~ '^[A-Za-z0-9_.:/+-]+$')::bigint AS code_like_ascii_rows,
            COUNT(*) FILTER (WHERE length({v}) <= 4)::bigint AS len_le_4_rows,
            COUNT(*) FILTER (WHERE length({v}) BETWEEN 5 AND 20)::bigint AS len_5_20_rows,
            COUNT(*) FILTER (WHERE length({v}) > 20)::bigint AS len_gt_20_rows,
            ROUND(AVG(length({v}))::numeric, 2) AS avg_len,
            MAX(length({v}))::bigint AS max_len
        FROM {schema}.{table} t
        JOIN tmp_task17_profile_person p ON p.person_id = t.person_id
        WHERE t.{date_col} BETWEEN DATE '1900-01-01' AND %s::date
          AND {v} IS NOT NULL
          AND NULLIF({v}, '') IS NOT NULL
        """
    ).format(
        v=v,
        schema=sql.Identifier(args.schema),
        table=sql.Identifier(target["table"]),
        date_col=sql.Identifier(target["date_col"]),
    )
    rows = fetch_dicts(conn, query, (label, args.db_end_date))
    return rows[0]


def top_mappings(conn, args, target):
    label = target["label"]
    log(f"top source/value to concept mapping: {label}")
    query = sql.SQL(
        """
        SELECT
            %s::text AS label,
            {value_expr} AS value,
            {concept_selects},
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT t.person_id)::bigint AS patients,
            MIN(t.{date_col})::text AS first_date,
            MAX(t.{date_col})::text AS last_date
        FROM {schema}.{table} t
        JOIN tmp_task17_profile_person p ON p.person_id = t.person_id
        LEFT JOIN {schema}.concept c ON c.concept_id = t.{concept_col}::bigint
        LEFT JOIN {schema}.concept sc ON sc.concept_id = t.{source_concept_col}::bigint
        {value_as_concept_join}
        WHERE t.{date_col} BETWEEN DATE '1900-01-01' AND %s::date
          AND {value_expr} IS NOT NULL
          AND NULLIF({value_expr}, '') IS NOT NULL
        GROUP BY
            {value_expr},
            t.{concept_col},
            c.concept_name,
            c.domain_id,
            c.vocabulary_id,
            c.concept_class_id,
            t.{source_concept_col},
            sc.concept_name,
            sc.vocabulary_id,
            {value_as_concept_group_by}
        ORDER BY rows DESC
        LIMIT %s
        """
    ).format(
        value_expr=value_expr(target),
        concept_selects=concept_selects(target),
        schema=sql.Identifier(args.schema),
        table=sql.Identifier(target["table"]),
        date_col=sql.Identifier(target["date_col"]),
        concept_col=sql.Identifier(target["concept_col"]),
        source_concept_col=sql.Identifier(target["source_concept_col"]),
        value_as_concept_join=value_as_concept_join(target, args.schema),
        value_as_concept_group_by=value_as_concept_group_by(target),
    )
    rows = fetch_dicts(conn, query, (label, args.db_end_date, args.top_limit))
    write_tsv(args.output_dir / f"{label}.top_mappings.tsv", rows)
    print_rows(f"{label} top {args.top_limit} value-to-concept mappings", rows, max_rows=args.limit)
    return rows


def classify(profile):
    rows = int(profile["non_null_rows"] or 0)
    if rows == 0:
        return "empty_or_not_observed_in_sample"
    numeric = int(profile["numeric_like_rows"] or 0) / rows
    code = int(profile["code_like_ascii_rows"] or 0) / rows
    short = int(profile["len_le_4_rows"] or 0) / rows
    long_text = int(profile["len_gt_20_rows"] or 0) / rows
    if numeric >= 0.8:
        return "numeric_string"
    if code >= 0.8 and long_text < 0.05:
        return "local_code"
    if short >= 0.8 and long_text < 0.05:
        return "short_categorical"
    if long_text >= 0.2:
        return "text_like"
    return "mixed"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": now(),
        "schema": args.schema,
        "db_end_date": args.db_end_date,
        "sample_per_mille": args.sample_per_mille,
        "seed": args.seed,
        "statement_timeout": args.statement_timeout,
        "with_aggregate": args.with_aggregate,
        "output_dir": str(args.output_dir.resolve()),
        "application_name": APPLICATION_NAME,
        "profiles": [],
    }
    log("Task 17 source/value terminal profile starting")
    log(f"output_dir={args.output_dir.resolve()}")
    conn = None
    try:
        conn = connect(args)
        conn.autocommit = True
        set_timeout(conn, args.statement_timeout)
        if args.with_aggregate:
            sample_patients = create_sample(conn, args)
            summary["sample_patients"] = sample_patients
        else:
            summary["sample_patients"] = None

        all_profiles = []
        for target in TARGETS:
            first_examples(conn, args, target)
            if args.with_aggregate:
                profile = column_profile(conn, args, target)
                profile["shape_class"] = classify(profile)
                all_profiles.append(profile)
                top_mappings(conn, args, target)

        if all_profiles:
            write_tsv(args.output_dir / "column_profile.tsv", all_profiles)
        summary["profiles"] = all_profiles
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if all_profiles:
            print_rows("column profile", all_profiles, max_rows=len(all_profiles))
        log(f"wrote summary: {args.output_dir / 'summary.json'}")
        log("completed")
    except Exception:
        if conn is not None:
            try:
                conn.cancel()
            except Exception:
                pass
        traceback.print_exc()
        return 1
    finally:
        if conn is not None:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
