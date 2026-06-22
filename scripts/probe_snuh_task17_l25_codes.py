#!/usr/bin/env python3
"""Probe SNUH L25 molecular genetics codes from a terminal session.

This script is for the Pod environment where the SNUH CDM is reachable. It is
intentionally code-prefix driven, not marker-regex driven: the default positive
control is EGFR exon 18-21 sequencing (L25372), and the default family prefix is
L25.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import json
import os
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:
    psycopg = None
    sql = None


DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task17_l25_code_probe"

TARGETS = [
    {
        "label": "procedure_occurrence.procedure_source_value",
        "table": "procedure_occurrence",
        "id_col": "procedure_occurrence_id",
        "date_col": "procedure_date",
        "value_col": "procedure_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "procedure_concept_id",
            "procedure_source_concept_id",
            "modifier_source_value",
            "quantity",
        ],
    },
    {
        "label": "measurement.measurement_source_value",
        "table": "measurement",
        "id_col": "measurement_id",
        "date_col": "measurement_date",
        "value_col": "measurement_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "measurement_concept_id",
            "measurement_source_concept_id",
            "value_as_number",
            "value_as_concept_id",
            "value_source_value",
            "unit_source_value",
            "range_low",
            "range_high",
        ],
    },
    {
        "label": "measurement.value_source_value",
        "table": "measurement",
        "id_col": "measurement_id",
        "date_col": "measurement_date",
        "value_col": "value_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "measurement_source_value",
            "measurement_concept_id",
            "measurement_source_concept_id",
            "value_as_number",
            "value_as_concept_id",
            "unit_source_value",
        ],
    },
    {
        "label": "observation.observation_source_value",
        "table": "observation",
        "id_col": "observation_id",
        "date_col": "observation_date",
        "value_col": "observation_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "observation_concept_id",
            "observation_source_concept_id",
            "value_as_number",
            "value_as_string",
            "value_as_concept_id",
            "value_source_value",
            "qualifier_source_value",
        ],
    },
    {
        "label": "observation.value_source_value",
        "table": "observation",
        "id_col": "observation_id",
        "date_col": "observation_date",
        "value_col": "value_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "observation_source_value",
            "observation_concept_id",
            "observation_source_concept_id",
            "value_as_number",
            "value_as_string",
            "value_as_concept_id",
        ],
    },
    {
        "label": "note.note_source_value",
        "table": "note",
        "id_col": "note_id",
        "date_col": "note_date",
        "value_col": "note_source_value",
        "person_col": "person_id",
        "extra_cols": [
            "note_title",
            "note_source_value2",
            "note_class_concept_id",
            "note_type_concept_id",
            "encoding_concept_id",
            "language_concept_id",
            "note_text",
        ],
    },
    {
        "label": "note.note_source_value2",
        "table": "note",
        "id_col": "note_id",
        "date_col": "note_date",
        "value_col": "note_source_value2",
        "person_col": "person_id",
        "extra_cols": [
            "note_title",
            "note_source_value",
            "note_class_concept_id",
            "note_type_concept_id",
            "note_text",
        ],
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
    parser.add_argument("--code", default="L25372")
    parser.add_argument("--prefix", default="L25")
    parser.add_argument("--statement-timeout", default="2min")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task17_l25_molecular_code_probe"),
    )
    return parser.parse_args()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_iso()}] {message}", flush=True)


def connect(args):
    if psycopg is None:
        raise RuntimeError("psycopg is required. Run this in the Pod environment.")
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


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_columns(conn, schema, table):
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
    """
    return {row["column_name"] for row in fetch_dicts(conn, query, (schema, table))}


def target_available(conn, schema, target):
    columns = table_columns(conn, schema, target["table"])
    required = {
        target["id_col"],
        target["date_col"],
        target["value_col"],
        target["person_col"],
    }
    missing = sorted(required - columns)
    if missing:
        return False, columns, missing
    return True, columns, []


def relation(schema, table):
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def compact_expression(column):
    if column == "note_text":
        return sql.SQL("left(regexp_replace({}::text, '\\s+', ' ', 'g'), 500) AS {}").format(
            sql.Identifier(column),
            sql.Identifier("note_text_prefix"),
        )
    return sql.SQL("{}::text AS {}").format(sql.Identifier(column), sql.Identifier(column))


def count_rows(conn, args, target, mode):
    operator = "=" if mode == "exact" else "LIKE"
    pattern = args.code if mode == "exact" else f"{args.prefix}%"
    query = sql.SQL(
        """
        SELECT
            {label} AS target,
            {mode} AS mode,
            {pattern} AS pattern,
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT {person_col})::bigint AS persons,
            MIN({date_col})::date AS min_date,
            MAX({date_col})::date AS max_date,
            COUNT(*) FILTER (WHERE {date_col} >= DATE '2021-01-01')::bigint AS post2021_rows,
            COUNT(DISTINCT {person_col}) FILTER (
                WHERE {date_col} >= DATE '2021-01-01'
            )::bigint AS post2021_persons
        FROM {table}
        WHERE {value_col} {operator} %s
        """
    ).format(
        label=sql.Literal(target["label"]),
        mode=sql.Literal(mode),
        pattern=sql.Literal(pattern),
        person_col=sql.Identifier(target["person_col"]),
        date_col=sql.Identifier(target["date_col"]),
        table=relation(args.schema, target["table"]),
        value_col=sql.Identifier(target["value_col"]),
        operator=sql.SQL(operator),
    )
    return fetch_dicts(conn, query, (pattern,))[0]


def year_counts(conn, args, target, mode):
    operator = "=" if mode == "exact" else "LIKE"
    pattern = args.code if mode == "exact" else f"{args.prefix}%"
    query = sql.SQL(
        """
        SELECT
            {label} AS target,
            {mode} AS mode,
            date_part('year', {date_col})::int AS year,
            COUNT(*)::bigint AS rows,
            COUNT(DISTINCT {person_col})::bigint AS persons
        FROM {table}
        WHERE {value_col} {operator} %s
        GROUP BY date_part('year', {date_col})::int
        ORDER BY year
        """
    ).format(
        label=sql.Literal(target["label"]),
        mode=sql.Literal(mode),
        date_col=sql.Identifier(target["date_col"]),
        person_col=sql.Identifier(target["person_col"]),
        table=relation(args.schema, target["table"]),
        value_col=sql.Identifier(target["value_col"]),
        operator=sql.SQL(operator),
    )
    return fetch_dicts(conn, query, (pattern,))


def sample_rows(conn, args, target, columns, mode):
    operator = "=" if mode == "exact" else "LIKE"
    pattern = args.code if mode == "exact" else f"{args.prefix}%"
    composables = [
        sql.SQL("{} AS target").format(sql.Literal(target["label"])),
        sql.SQL("{} AS mode").format(sql.Literal(mode)),
        sql.SQL("{}::text AS id").format(sql.Identifier(target["id_col"])),
        sql.SQL("{}::text AS person_id").format(sql.Identifier(target["person_col"])),
        sql.SQL("{}::date AS event_date").format(sql.Identifier(target["date_col"])),
        sql.SQL("{}::text AS matched_value").format(sql.Identifier(target["value_col"])),
    ]
    for column in target["extra_cols"]:
        if column in columns:
            composables.append(compact_expression(column))
    query = sql.SQL(
        """
        SELECT {selects}
        FROM {table}
        WHERE {value_col} {operator} %s
        ORDER BY {date_col} DESC, {id_col} DESC
        LIMIT %s
        """
    ).format(
        selects=sql.SQL(", ").join(composables),
        table=relation(args.schema, target["table"]),
        value_col=sql.Identifier(target["value_col"]),
        operator=sql.SQL(operator),
        date_col=sql.Identifier(target["date_col"]),
        id_col=sql.Identifier(target["id_col"]),
    )
    return fetch_dicts(conn, query, (pattern, args.sample_limit))


def write_report(path: Path, args, counts, years, samples, skipped):
    lines = [
        "# Task 17 L25 Molecular Code Probe",
        f"generated_at: {now_iso()}",
        f"schema: {args.schema}",
        f"exact_code: {args.code}",
        f"prefix: {args.prefix}",
        "",
        "## Counts",
        "",
        "| target | mode | pattern | rows | persons | min_date | max_date | post2021_rows | post2021_persons |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in counts:
        lines.append(
            "| {target} | {mode} | {pattern} | {rows} | {persons} | {min_date} | "
            "{max_date} | {post2021_rows} | {post2021_persons} |".format(**row)
        )
    if skipped:
        lines.extend(["", "## Skipped Targets", ""])
        for row in skipped:
            lines.append(f"- {row['target']}: missing columns {row['missing_columns']}")
    lines.extend(["", "## Year Counts", ""])
    for row in years:
        lines.append(
            "{target} | {mode} | {year}: rows={rows}, persons={persons}".format(**row)
        )
    lines.extend(["", "## Recent Samples", ""])
    for row in samples[:200]:
        preview = " | ".join(
            f"{key}={value}" for key, value in row.items() if value not in (None, "")
        )
        lines.append(f"- {preview}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"output_dir={args.output_dir}")
    log(f"probing exact code {args.code} and prefix {args.prefix}%")

    summary = {
        "generated_at": now_iso(),
        "schema": args.schema,
        "application_name": APPLICATION_NAME,
        "code": args.code,
        "prefix": args.prefix,
        "statement_timeout": args.statement_timeout,
    }
    counts = []
    years = []
    samples = []
    skipped = []

    with connect(args) as conn:
        execute(conn, "SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
        for target in TARGETS:
            available, columns, missing = target_available(conn, args.schema, target)
            if not available:
                log(f"skip {target['label']}: missing {missing}")
                skipped.append({"target": target["label"], "missing_columns": missing})
                continue
            for mode in ["exact", "prefix"]:
                log(f"{target['label']} {mode}")
                counts.append(count_rows(conn, args, target, mode))
                years.extend(year_counts(conn, args, target, mode))
                samples.extend(sample_rows(conn, args, target, columns, mode))

    count_fields = [
        "target",
        "mode",
        "pattern",
        "rows",
        "persons",
        "min_date",
        "max_date",
        "post2021_rows",
        "post2021_persons",
    ]
    year_fields = ["target", "mode", "year", "rows", "persons"]
    write_csv(args.output_dir / "l25_code_counts.csv", counts, count_fields)
    write_csv(args.output_dir / "l25_code_year_counts.csv", years, year_fields)
    write_csv(args.output_dir / "l25_code_recent_samples.csv", samples)
    write_csv(args.output_dir / "l25_code_skipped_targets.csv", skipped)
    write_report(args.output_dir / "l25_code_probe_report.md", args, counts, years, samples, skipped)
    summary.update(
        {
            "counts_csv": str(args.output_dir / "l25_code_counts.csv"),
            "year_counts_csv": str(args.output_dir / "l25_code_year_counts.csv"),
            "samples_csv": str(args.output_dir / "l25_code_recent_samples.csv"),
            "report": str(args.output_dir / "l25_code_probe_report.md"),
            "skipped": skipped,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {args.output_dir / 'l25_code_probe_report.md'}")


if __name__ == "__main__":
    main()
