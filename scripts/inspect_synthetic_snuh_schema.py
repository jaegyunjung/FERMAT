"""
Inspect Synthetic SNUH OMOP CDM schema from a DuckDB file.

Usage:
    python scripts/inspect_synthetic_snuh_schema.py \
        --duckdb data/synthetic_snuh_raw.duckdb \
        --out logs/synthetic_snuh_schema_summary.txt

Outputs a human-readable summary of:
  - tables present
  - row counts per table
  - columns (name, dtype) per table
  - fill rates for source_value / concept_id / date / birth columns
  - death table accessibility
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb


OMOP_TABLES = [
    "person",
    "visit_occurrence",
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "measurement",
    "death",
    "observation",
    "observation_period",
]

KEY_COLUMNS = {
    "person": [
        "person_id", "gender_concept_id", "gender_source_value",
        "year_of_birth", "month_of_birth", "day_of_birth", "birth_datetime",
    ],
    "condition_occurrence": [
        "person_id", "condition_concept_id", "condition_source_value",
        "condition_start_date", "condition_start_datetime",
    ],
    "drug_exposure": [
        "person_id", "drug_concept_id", "drug_source_value",
        "drug_exposure_start_date", "drug_exposure_start_datetime",
    ],
    "procedure_occurrence": [
        "person_id", "procedure_concept_id", "procedure_source_value",
        "procedure_date", "procedure_datetime",
    ],
    "measurement": [
        "person_id", "measurement_concept_id", "measurement_source_value",
        "measurement_date", "measurement_datetime",
        "value_as_number", "value_as_concept_id",
        "unit_concept_id", "unit_source_value",
    ],
    "death": [
        "person_id", "death_date", "death_datetime",
        "cause_concept_id", "cause_source_value",
    ],
    "visit_occurrence": [
        "person_id", "visit_concept_id", "visit_source_value",
        "visit_start_date", "visit_end_date",
    ],
}


def list_tables(con):
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def describe_table(con, table):
    cols = con.execute(f"DESCRIBE {table}").fetchall()
    return [(c[0], c[1]) for c in cols]


def row_count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def fill_rate(con, table, column):
    total = row_count(con, table)
    if total == 0:
        return 0, 0, 0.0
    non_null = con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL"
    ).fetchone()[0]
    return total, non_null, non_null / total


def write_section(out, title, body=""):
    out.write(f"\n## {title}\n")
    if body:
        out.write(body + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb",
        default="data/synthetic_snuh_raw.duckdb",
        help="path to DuckDB file (default: data/synthetic_snuh_raw.duckdb)",
    )
    parser.add_argument(
        "--out",
        default="logs/synthetic_snuh_schema_summary.txt",
        help="output summary file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print(f"ERROR: DuckDB file not found: {args.duckdb}", file=sys.stderr)
        print(
            "If you have not yet downloaded Synthetic SNUH from KHDP, "
            "this script will be skipped. The fallback path "
            "(generate_self_synthetic_4col.py) does not need it.",
            file=sys.stderr,
        )
        sys.exit(1)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.duckdb, read_only=True)
    tables = list_tables(con)

    with open(args.out, "w") as out:
        out.write("# Synthetic SNUH Schema Summary\n")
        out.write(f"Source: {args.duckdb}\n")
        out.write(f"Tables found: {len(tables)}\n")

        write_section(out, "Tables present")
        for t in tables:
            marker = "[OMOP]" if t in OMOP_TABLES else "      "
            out.write(f"  {marker} {t}\n")

        write_section(out, "Required OMOP tables")
        for t in OMOP_TABLES:
            present = "YES" if t in tables else "NO"
            count = row_count(con, t) if t in tables else "-"
            out.write(f"  {present:3s}  {t:25s}  rows={count}\n")

        write_section(out, "Schema and fill rates")
        for t in OMOP_TABLES:
            if t not in tables:
                continue
            cols = describe_table(con, t)
            count = row_count(con, t)
            out.write(f"\n### {t} ({count} rows)\n")
            colnames = {c[0]: c[1] for c in cols}
            out.write("  Columns:\n")
            for name, dtype in cols:
                out.write(f"    - {name}: {dtype}\n")
            out.write("  Fill rates for key columns:\n")
            for kc in KEY_COLUMNS.get(t, []):
                if kc not in colnames:
                    out.write(f"    {kc:32s}  MISSING\n")
                    continue
                total, non_null, rate = fill_rate(con, t, kc)
                out.write(
                    f"    {kc:32s}  {non_null}/{total} ({rate*100:.1f}%)\n"
                )

        write_section(out, "Death table accessibility")
        if "death" in tables:
            n = row_count(con, "death")
            out.write(f"  death table present, {n} rows\n")
        else:
            out.write("  death table NOT present\n")

        write_section(out, "Patient sample size suggestions")
        if "person" in tables:
            n_persons = row_count(con, "person")
            out.write(f"  Total persons: {n_persons}\n")
            out.write(f"  For smoke test: use all {min(n_persons, 10000)} persons\n")

    con.close()
    print(f"Schema summary written to {args.out}")


if __name__ == "__main__":
    main()
