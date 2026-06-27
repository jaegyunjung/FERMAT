#!/usr/bin/env python3
"""Audit source event dates for SNUH clinical evaluation rows.

This script reads the real CDM event-date columns for DX/RX/PX/DTH events and
matches them back to the existing FERMAT shard rows. It is intended for
date-based evaluation of the clinical targets used by next-token, same-day, and
waiting-time metrics.
"""

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
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_snuh_clinical_event_date_audit"

CLINICAL_TYPES = {
    1: "DX",
    2: "RX",
    3: "PX",
    6: "DTH",
}
DOMAIN_QUERIES = [
    {
        "label": "DX",
        "table": "condition_occurrence",
        "concept_col": "condition_concept_id",
        "date_col": "condition_start_date",
        "token_prefix": "DX:",
        "fallback_token_id": 2,
        "token_type_id": 1,
    },
    {
        "label": "RX",
        "table": "drug_exposure",
        "concept_col": "drug_concept_id",
        "date_col": "drug_exposure_start_date",
        "token_prefix": "RX:",
        "fallback_token_id": 3,
        "token_type_id": 2,
    },
    {
        "label": "PX",
        "table": "procedure_occurrence",
        "concept_col": "procedure_concept_id",
        "date_col": "procedure_date",
        "token_prefix": "PX:",
        "fallback_token_id": 4,
        "token_type_id": 3,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--patient-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--db-end-date", default="2025-02-05")
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
    parser.add_argument("--fetch-size", type=int, default=250_000)
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
        cur.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (args.statement_timeout,),
        )
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


def load_shard(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.size % 4:
        raise ValueError(f"{path} does not contain 4-column uint32 rows")
    return np.asarray(raw.reshape(-1, 4))


def load_patient_map(path: Path, splits: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    frame = frame.loc[frame["split"].isin(splits)].copy()
    if frame.empty:
        raise RuntimeError(f"No patients found for splits: {sorted(splits)}")
    frame["patient_id_dense"] = frame["patient_id_dense"].astype(np.int64)
    frame["person_id"] = frame["person_id"].astype(np.int64)
    return frame.sort_values("patient_id_dense", kind="mergesort")


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    registry = pd.read_csv(path)
    registry = registry[["token_id", "token_key", "token_type_id"]].copy()
    registry["token_id"] = registry["token_id"].astype(np.int64)
    registry["token_type_id"] = registry["token_type_id"].astype(np.int64)
    registry = registry.loc[registry["token_type_id"].isin(CLINICAL_TYPES)]
    return registry


def upload_dataframe(conn, frame, table_name, columns_sql, insert_sql, label):
    execute(conn, sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
    execute(
        conn,
        sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT PRESERVE ROWS").format(
            sql.Identifier(table_name),
            sql.SQL(columns_sql),
        ),
    )
    rows = list(frame.itertuples(index=False, name=None))
    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)
    conn.commit()
    log(f"uploaded {label}: {len(rows):,}")


def upload_reference_tables(conn, patient_map, registry):
    upload_dataframe(
        conn,
        patient_map[["patient_id_dense", "person_id", "split"]],
        "tmp_event_audit_patient_map",
        "patient_id_dense bigint, person_id bigint, split text",
        "INSERT INTO tmp_event_audit_patient_map VALUES (%s,%s,%s)",
        "patient map",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_event_audit_patient_map(person_id)",
        label="Index patient map",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_event_audit_patient_map(split, person_id)",
        label="Index patient split",
    )

    upload_dataframe(
        conn,
        registry,
        "tmp_event_audit_registry",
        "token_id bigint, token_key text, token_type_id integer",
        "INSERT INTO tmp_event_audit_registry VALUES (%s,%s,%s)",
        "clinical token registry",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_event_audit_registry(token_key)",
        label="Index token registry",
    )


def clinical_rows_from_shard(shard: np.ndarray, split: str) -> pd.DataFrame:
    mask = np.isin(shard[:, 3].astype(np.int64), list(CLINICAL_TYPES))
    row_index = np.flatnonzero(mask).astype(np.int64)
    rows = shard[mask]
    frame = pd.DataFrame(
        {
            "split": split,
            "row_index": row_index,
            "patient_id_dense": rows[:, 0].astype(np.int64),
            "age_in_days": rows[:, 1].astype(np.int64),
            "token_id": rows[:, 2].astype(np.int64),
            "token_type_id": rows[:, 3].astype(np.int64),
        }
    )
    key_cols = ["patient_id_dense", "age_in_days", "token_id", "token_type_id"]
    frame["key_ordinal"] = frame.groupby(key_cols, sort=False).cumcount()
    return frame


def source_query_for_domain(args, spec):
    return sql.SQL(
        """
        WITH birth AS (
            SELECT
                p.patient_id_dense,
                p.person_id,
                p.split,
                make_date(
                    person.year_of_birth,
                    CASE
                        WHEN person.month_of_birth BETWEEN 1 AND 12
                        THEN person.month_of_birth
                        ELSE 7
                    END,
                    CASE
                        WHEN person.day_of_birth BETWEEN 1 AND 28
                        THEN person.day_of_birth
                        ELSE 1
                    END
                ) AS birth_date
            FROM tmp_event_audit_patient_map p
            JOIN {}.person AS person USING(person_id)
            WHERE p.split = %s
        )
        SELECT
            b.split,
            b.patient_id_dense,
            (e.{} - b.birth_date)::integer AS age_in_days,
            COALESCE(r.token_id, %s)::bigint AS token_id,
            %s::integer AS token_type_id,
            e.{}::date AS event_date,
            %s::text AS source_table
        FROM {}.{} e
        JOIN birth b USING(person_id)
        LEFT JOIN tmp_event_audit_registry r
          ON r.token_key = %s || e.{}::text
        WHERE e.{} BETWEEN b.birth_date AND %s::date
          AND e.{} IS NOT NULL
          AND e.{} <> 0
        ORDER BY b.patient_id_dense, e.{}, token_id
        """
    ).format(
        sql.Identifier(args.schema),
        sql.Identifier(spec["date_col"]),
        sql.Identifier(spec["date_col"]),
        sql.Identifier(args.schema),
        sql.Identifier(spec["table"]),
        sql.Identifier(spec["concept_col"]),
        sql.Identifier(spec["date_col"]),
        sql.Identifier(spec["concept_col"]),
        sql.Identifier(spec["concept_col"]),
        sql.Identifier(spec["date_col"]),
    )


def death_query(args):
    return sql.SQL(
        """
        WITH birth AS (
            SELECT
                p.patient_id_dense,
                p.person_id,
                p.split,
                make_date(
                    person.year_of_birth,
                    CASE
                        WHEN person.month_of_birth BETWEEN 1 AND 12
                        THEN person.month_of_birth
                        ELSE 7
                    END,
                    CASE
                        WHEN person.day_of_birth BETWEEN 1 AND 28
                        THEN person.day_of_birth
                        ELSE 1
                    END
                ) AS birth_date
            FROM tmp_event_audit_patient_map p
            JOIN {}.person AS person USING(person_id)
            WHERE p.split = %s
        )
        SELECT
            b.split,
            b.patient_id_dense,
            (d.death_date - b.birth_date)::integer AS age_in_days,
            COALESCE(r.token_id, 5)::bigint AS token_id,
            6::integer AS token_type_id,
            d.death_date::date AS event_date,
            'death'::text AS source_table
        FROM {}.death d
        JOIN birth b USING(person_id)
        LEFT JOIN tmp_event_audit_registry r
          ON r.token_key = 'SPECIAL:DEATH'
        WHERE d.death_date BETWEEN b.birth_date AND %s::date
        ORDER BY b.patient_id_dense, d.death_date, token_id
        """
    ).format(sql.Identifier(args.schema), sql.Identifier(args.schema))


def fetch_source_events(conn, args, split):
    chunks = []
    for spec in DOMAIN_QUERIES:
        started = time.time()
        log(f"[START] Fetch {split} {spec['label']} source events")
        with conn.cursor(name=f"source_{split}_{spec['label'].lower()}") as cur:
            cur.execute(
                source_query_for_domain(args, spec),
                (
                    split,
                    spec["fallback_token_id"],
                    spec["token_type_id"],
                    spec["table"],
                    spec["token_prefix"],
                    args.db_end_date,
                ),
            )
            while True:
                rows = cur.fetchmany(args.fetch_size)
                if not rows:
                    break
                chunks.append(
                    pd.DataFrame(
                        rows,
                        columns=[
                            "split",
                            "patient_id_dense",
                            "age_in_days",
                            "token_id",
                            "token_type_id",
                            "event_date",
                            "source_table",
                        ],
                    )
                )
        conn.commit()
        log(f"[DONE] Fetch {split} {spec['label']} {time.time() - started:,.1f}s")

    started = time.time()
    log(f"[START] Fetch {split} DTH source events")
    with conn.cursor(name=f"source_{split}_dth") as cur:
        cur.execute(death_query(args), (split, args.db_end_date))
        while True:
            rows = cur.fetchmany(args.fetch_size)
            if not rows:
                break
            chunks.append(
                pd.DataFrame(
                    rows,
                    columns=[
                        "split",
                        "patient_id_dense",
                        "age_in_days",
                        "token_id",
                        "token_type_id",
                        "event_date",
                        "source_table",
                    ],
                )
            )
    conn.commit()
    log(f"[DONE] Fetch {split} DTH {time.time() - started:,.1f}s")

    if not chunks:
        return pd.DataFrame(
            columns=[
                "split",
                "patient_id_dense",
                "age_in_days",
                "token_id",
                "token_type_id",
                "event_date",
                "source_table",
            ]
        )
    source = pd.concat(chunks, ignore_index=True)
    for column in ["patient_id_dense", "age_in_days", "token_id", "token_type_id"]:
        source[column] = source[column].astype(np.int64)
    source["event_date"] = pd.to_datetime(source["event_date"])
    key_cols = ["patient_id_dense", "age_in_days", "token_id", "token_type_id"]
    source = source.sort_values(
        key_cols + ["event_date", "source_table"],
        kind="mergesort",
    )
    source["key_ordinal"] = source.groupby(key_cols, sort=False).cumcount()
    return source


def merge_and_validate(shard_rows, source_rows):
    key_cols = [
        "patient_id_dense",
        "age_in_days",
        "token_id",
        "token_type_id",
        "key_ordinal",
    ]
    merged = shard_rows.merge(
        source_rows,
        on=key_cols,
        how="outer",
        suffixes=("_shard", "_source"),
        indicator=True,
    )
    missing_source = merged["_merge"].eq("left_only")
    extra_source = merged["_merge"].eq("right_only")
    summary = {
        "shard_clinical_rows": int(len(shard_rows)),
        "source_clinical_rows": int(len(source_rows)),
        "matched_rows": int(merged["_merge"].eq("both").sum()),
        "missing_source_rows": int(missing_source.sum()),
        "extra_source_rows": int(extra_source.sum()),
    }
    if missing_source.any() or extra_source.any():
        examples = merged.loc[
            missing_source | extra_source,
            key_cols + ["_merge"],
        ].head(50)
        return merged, summary, examples
    matched = merged.loc[merged["_merge"].eq("both")].copy()
    matched["split"] = matched["split_shard"]
    matched = matched[
        [
            "split",
            "row_index",
            "patient_id_dense",
            "age_in_days",
            "token_id",
            "token_type_id",
            "event_date",
            "source_table",
        ]
    ].sort_values("row_index", kind="mergesort")
    matched["calendar_year"] = matched["event_date"].dt.year.astype("int16")
    matched["days_since_previous_clinical_event"] = (
        matched.groupby("patient_id_dense")["event_date"].diff().dt.days.astype("Int32")
    )
    matched["same_day_as_previous_clinical"] = (
        matched["days_since_previous_clinical_event"].eq(0).fillna(False)
    )
    matched["clinical_gap_over_365_days"] = (
        matched["days_since_previous_clinical_event"].gt(365).fillna(False)
    )
    return matched, summary, None


def write_split_outputs(output_dir, split, matched, summary, examples):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{split}_clinical_source_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if examples is not None:
        examples_path = output_dir / f"{split}_clinical_source_audit_mismatches.csv"
        examples.to_csv(examples_path, index=False)
        raise RuntimeError(
            f"{split}: source/shard mismatch. See {examples_path}"
        )
    rows_path = output_dir / f"{split}_clinical_event_dates.parquet"
    matched.to_parquet(rows_path, index=False)
    counts = (
        matched.assign(token_type=matched["token_type_id"].map(CLINICAL_TYPES))
        .groupby(["token_type", "calendar_year"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["token_type", "calendar_year"])
    )
    counts_path = output_dir / f"{split}_clinical_event_year_counts.csv"
    counts.to_csv(counts_path, index=False)
    summary.update(
        {
            "rows_path": str(rows_path),
            "year_counts_path": str(counts_path),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    patient_map_path = (
        args.patient_map.expanduser().resolve()
        if args.patient_map
        else args.data_dir / "patient_id_map.parquet"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.data_dir / "clinical_event_date_audit"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = set(args.splits)
    patient_map = load_patient_map(patient_map_path, splits)
    registry = load_registry(args.data_dir / "token_registry.csv")
    log(f"data_dir={args.data_dir}")
    log(f"patient_map={patient_map_path}")
    log(f"output_dir={output_dir}")
    log(f"patients={len(patient_map):,}")
    log(f"clinical registry tokens={len(registry):,}")

    summaries = []
    with connect(args) as conn:
        upload_reference_tables(conn, patient_map, registry)
        for split in args.splits:
            log(f"\n## SPLIT {split}")
            shard = load_shard(args.data_dir / f"{split}.bin")
            shard_rows = clinical_rows_from_shard(shard, split)
            log(f"shard clinical rows={len(shard_rows):,}")
            del shard
            source_rows = fetch_source_events(conn, args, split)
            log(f"source clinical rows={len(source_rows):,}")
            matched, summary, examples = merge_and_validate(shard_rows, source_rows)
            log(json.dumps(summary, ensure_ascii=False))
            summaries.append(
                write_split_outputs(output_dir, split, matched, summary, examples)
            )

    manifest = {
        "method": "source CDM event-date audit for DX/RX/PX/DTH clinical rows",
        "data_dir": str(args.data_dir),
        "patient_map": str(patient_map_path),
        "output_dir": str(output_dir),
        "db_end_date": args.db_end_date,
        "splits": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"\n[DONE] wrote clinical source-date audit to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
