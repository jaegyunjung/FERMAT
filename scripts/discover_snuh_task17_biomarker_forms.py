#!/usr/bin/env python3
"""Discover SNUH CDM forms that carry tumor biomarker result text.

This is a form-discovery tool, not a token generator. It first finds source
signatures such as observation source/concept or note title/source that contain
high-confidence biomarker result language, then prints representative rows.

Run on the SNUH Pod. The script prints to the terminal by default and does not
write output files unless --report is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import os
from pathlib import Path

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None
    dict_row = None


DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task17_biomarker_form_discovery"

# The pattern is intentionally result-shape oriented. RET/MET alone are not
# included because they match common non-biomarker words such as interpretation,
# preterm, ureter, metastasis, and endometrial.
BIOMARKER_RESULT_RE = (
    r"(?i)("
    r"\mEGFR\M|\mALK\M|\mKRAS\M|\mNRAS\M|\mBRAF\M|\mROS1\M|"
    r"\mBRCA1\M|\mBRCA2\M|\mBRCA\M|\mHER2\M|\mERBB2\M|"
    r"PD[- ]?L1|22C3|SP263|CPS|TPS|"
    r"\mMSI[- ]?[HL]\M|\mMSS\M|\mMMR[dpi]?\M|"
    r"\mNTRK[123]?\M|"
    r"\mRET\M.{0,30}(fusion|rearrangement|mutation|mut|positive|negative|wild|wt)|"
    r"\mMET\M.{0,30}(exon|skipping|amplification|amp|mutation|mut|positive|negative|wild|wt)|"
    r"L858R|E19del|exon ?19|del ?19|G12C|G12D|G12V|G12F|V600E|T790M"
    r")"
)

NOISE_RE = (
    r"(?i)("
    r"\mBrCa\M[, ]*(Lt|Rt|Left|Right)?\M|"
    r"\mMSi\M\s*\(rheumatic\)|"
    r"\mCPSP\M|"
    r"Poor data quality|Normal ECG|Abnormal ECG|"
    r"interpretation may be adversely affected"
    r")"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--statement-timeout", default="8min")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--sample-limit", type=int, default=80)
    parser.add_argument("--include-measurement", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


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


def table_columns(conn, schema_name, table_name):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, table_name),
        )
        return {row["column_name"] for row in cur.fetchall()}


def format_rows(title, rows, max_rows=None):
    lines = ["", f"## {title}"]
    if not rows:
        lines.append("(no rows)")
        return lines
    if max_rows is None:
        max_rows = len(rows)
    headers = list(rows[0].keys())
    lines.append("\t".join(headers))
    for row in rows[:max_rows]:
        values = []
        for header in headers:
            value = row.get(header)
            if value is None:
                value = ""
            value = str(value).replace("\n", " ").replace("\r", " ")
            if len(value) > 1200:
                value = value[:1200] + "..."
            values.append(value)
        lines.append("\t".join(values))
    return lines


def fetch_dicts(conn, query, params):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def text_expr(table_alias, columns):
    pieces = []
    for column_name in columns:
        pieces.append(
            sql.SQL("coalesce({}.{}::text, '')").format(
                sql.Identifier(table_alias),
                sql.Identifier(column_name),
            )
        )
    return sql.SQL(" || ' ' || ").join(pieces)


def observation_sections(conn, args, schema_ident):
    columns = table_columns(conn, args.schema, "observation")
    value_columns = [
        col
        for col in [
            "value_source_value",
            "value_as_string",
            "qualifier_source_value",
            "observation_source_value",
            "ext_etc_source_value",
        ]
        if col in columns
    ]
    expr = text_expr("o", value_columns)
    base_filter = sql.SQL("({expr}) ~* %s AND NOT (({expr}) ~* %s)").format(expr=expr)

    counts_query = sql.SQL(
        """
        SELECT
            'observation' AS source_table,
            coalesce(o.observation_source_value, '') AS source_signature,
            o.observation_concept_id,
            extract(year FROM o.observation_date)::int AS year,
            count(*)::bigint AS rows,
            count(DISTINCT o.person_id)::bigint AS persons,
            min(o.observation_date) AS min_date,
            max(o.observation_date) AS max_date
        FROM {schema}.observation o
        WHERE o.observation_date >= %s::date
          AND {base_filter}
        GROUP BY coalesce(o.observation_source_value, ''), o.observation_concept_id,
                 extract(year FROM o.observation_date)::int
        ORDER BY persons DESC, rows DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, base_filter=base_filter)
    rows = fetch_dicts(conn, counts_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.limit))

    sample_query = sql.SQL(
        """
        SELECT
            o.observation_id AS source_id,
            o.person_id,
            o.observation_date AS event_date,
            o.observation_datetime AS event_datetime,
            coalesce(o.observation_source_value, '') AS source_signature,
            o.observation_concept_id,
            left(regexp_replace(({expr}), '\\s+', ' ', 'g'), 1500) AS text_prefix
        FROM {schema}.observation o
        WHERE o.observation_date >= %s::date
          AND {base_filter}
        ORDER BY o.observation_date DESC, o.observation_id DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, expr=expr, base_filter=base_filter)
    samples = fetch_dicts(conn, sample_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.sample_limit))
    return rows, samples


def note_sections(conn, args, schema_ident):
    columns = table_columns(conn, args.schema, "note")
    value_columns = [
        col
        for col in [
            "note_title",
            "note_source_value",
            "note_source_value2",
            "note_source_value3",
            "note_source_value4",
            "note_text",
        ]
        if col in columns
    ]
    expr = text_expr("n", value_columns)
    base_filter = sql.SQL("({expr}) ~* %s AND NOT (({expr}) ~* %s)").format(expr=expr)

    counts_query = sql.SQL(
        """
        SELECT
            'note' AS source_table,
            coalesce(n.note_title, '') AS source_signature,
            coalesce(n.note_source_value, '') AS note_source_value,
            extract(year FROM n.note_date)::int AS year,
            count(*)::bigint AS rows,
            count(DISTINCT n.person_id)::bigint AS persons,
            min(n.note_date) AS min_date,
            max(n.note_date) AS max_date
        FROM {schema}.note n
        WHERE n.note_date >= %s::date
          AND {base_filter}
        GROUP BY coalesce(n.note_title, ''), coalesce(n.note_source_value, ''),
                 extract(year FROM n.note_date)::int
        ORDER BY persons DESC, rows DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, base_filter=base_filter)
    rows = fetch_dicts(conn, counts_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.limit))

    sample_query = sql.SQL(
        """
        SELECT
            n.note_id AS source_id,
            n.person_id,
            n.note_date AS event_date,
            n.note_title AS source_signature,
            n.note_source_value,
            n.note_source_value2,
            left(regexp_replace(({expr}), '\\s+', ' ', 'g'), 1800) AS text_prefix
        FROM {schema}.note n
        WHERE n.note_date >= %s::date
          AND {base_filter}
        ORDER BY n.note_date DESC, n.note_id DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, expr=expr, base_filter=base_filter)
    samples = fetch_dicts(conn, sample_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.sample_limit))
    return rows, samples


def condition_sections(conn, args, schema_ident):
    columns = table_columns(conn, args.schema, "condition_occurrence")
    if "ext_cond_source_value_cc_text" not in columns:
        return [], []
    expr = text_expr("c", ["ext_cond_source_value_cc_text"])
    base_filter = sql.SQL("({expr}) ~* %s AND NOT (({expr}) ~* %s)").format(expr=expr)

    counts_query = sql.SQL(
        """
        SELECT
            'condition_occurrence' AS source_table,
            coalesce(c.condition_source_value, '') AS source_signature,
            c.condition_concept_id,
            extract(year FROM c.condition_start_date)::int AS year,
            count(*)::bigint AS rows,
            count(DISTINCT c.person_id)::bigint AS persons,
            min(c.condition_start_date) AS min_date,
            max(c.condition_start_date) AS max_date
        FROM {schema}.condition_occurrence c
        WHERE c.condition_start_date >= %s::date
          AND {base_filter}
        GROUP BY coalesce(c.condition_source_value, ''), c.condition_concept_id,
                 extract(year FROM c.condition_start_date)::int
        ORDER BY persons DESC, rows DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, base_filter=base_filter)
    rows = fetch_dicts(conn, counts_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.limit))

    sample_query = sql.SQL(
        """
        SELECT
            c.condition_occurrence_id AS source_id,
            c.person_id,
            c.condition_start_date AS event_date,
            c.condition_source_value AS source_signature,
            c.condition_concept_id,
            left(regexp_replace(({expr}), '\\s+', ' ', 'g'), 1200) AS text_prefix
        FROM {schema}.condition_occurrence c
        WHERE c.condition_start_date >= %s::date
          AND {base_filter}
        ORDER BY c.condition_start_date DESC, c.condition_occurrence_id DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, expr=expr, base_filter=base_filter)
    samples = fetch_dicts(conn, sample_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.sample_limit))
    return rows, samples


def measurement_sections(conn, args, schema_ident):
    columns = table_columns(conn, args.schema, "measurement")
    value_columns = [
        col
        for col in [
            "measurement_source_value",
            "value_source_value",
            "unit_source_value",
        ]
        if col in columns
    ]
    expr = text_expr("m", value_columns)
    base_filter = sql.SQL("({expr}) ~* %s AND NOT (({expr}) ~* %s)").format(expr=expr)

    sample_query = sql.SQL(
        """
        SELECT
            m.measurement_id AS source_id,
            m.person_id,
            m.measurement_date AS event_date,
            m.measurement_concept_id,
            m.measurement_source_concept_id,
            m.measurement_source_value AS source_signature,
            left(regexp_replace(({expr}), '\\s+', ' ', 'g'), 1200) AS text_prefix
        FROM {schema}.measurement m
        WHERE m.measurement_date >= %s::date
          AND {base_filter}
        ORDER BY m.measurement_date DESC, m.measurement_id DESC
        LIMIT %s
        """
    ).format(schema=schema_ident, expr=expr, base_filter=base_filter)
    samples = fetch_dicts(conn, sample_query, (args.since, BIOMARKER_RESULT_RE, NOISE_RE, args.sample_limit))
    return [], samples


def main():
    args = parse_args()
    output = [
        "# SNUH Task 17 Biomarker Form Discovery",
        f"generated_at: {now()}",
        f"schema: {args.schema}",
        f"since: {args.since}",
        "",
        "## Purpose",
        "Find source/form signatures that carry biomarker result text before any token integration.",
        "This avoids treating EGFR/ALK/KRAS keyword hits as the final unit of discovery.",
    ]

    with connect(args) as conn:
        schema_ident = sql.Identifier(args.schema)
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))

        obs_counts, obs_samples = observation_sections(conn, args, schema_ident)
        note_counts, note_samples = note_sections(conn, args, schema_ident)
        cond_counts, cond_samples = condition_sections(conn, args, schema_ident)

        output.extend(format_rows("Observation Form Counts", obs_counts))
        output.extend(format_rows("Observation Form Samples", obs_samples))
        output.extend(format_rows("Note Form Counts", note_counts))
        output.extend(format_rows("Note Form Samples", note_samples))
        output.extend(format_rows("Condition Form Counts", cond_counts))
        output.extend(format_rows("Condition Form Samples", cond_samples))

        if args.include_measurement:
            output.append("")
            output.append("## Measurement Samples")
            output.append("Requested explicitly. This may time out on unindexed SNUH measurement.")
            meas_counts, meas_samples = measurement_sections(conn, args, schema_ident)
            output.extend(format_rows("Measurement Form Counts", meas_counts))
            output.extend(format_rows("Measurement Form Samples", meas_samples))

    text = "\n".join(output) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
