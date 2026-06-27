#!/usr/bin/env python3
"""Build per-row event-date sidecars for SNUH FERMAT shards.

The training shards keep only `(patient_id, age_in_days, token_id,
token_type_id)`. This script reconstructs the original CDM event date using the
same tokenization rules and writes one Parquet sidecar per split. When the
target data directory contains appended GENOMICS tokens, their retained
provenance file is merged into the same row order as the augmented shard.
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
DEFAULT_BASE_ETL_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"

DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_snuh_event_date_sidecar"
NONNEGATIVE_MEASUREMENT_IDS = [3012888, 3013650, 3025315, 3004249, 3004410]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-etl-dir", type=Path, default=DEFAULT_BASE_ETL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="10min")
    parser.add_argument("--lab-buckets", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=500_000)
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


def query_df(conn, statement, params=None, label=None):
    started = time.time()
    if label:
        log(f"[START] {label}")
    with conn.cursor() as cur:
        cur.execute(statement, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    if label:
        log(f"[DONE] {label} rows={len(frame):,} {time.time() - started:,.1f}s")
    return frame


def load_uint32_shard(path: Path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.size % 4:
        raise ValueError(f"{path} does not contain 4-column uint32 rows")
    return raw.reshape(-1, 4)


def iter_patient_blocks(arr: np.ndarray):
    if len(arr) == 0:
        return
    patient_ids = arr[:, 0]
    starts = np.r_[0, np.flatnonzero(np.diff(patient_ids)) + 1]
    ends = np.r_[starts[1:], len(arr)]
    for start, end in zip(starts, ends):
        yield int(patient_ids[start]), int(start), int(end)


def load_required_files(args):
    for path in [
        args.base_etl_dir / "patient_id_map.parquet",
        args.base_etl_dir / "token_registry.csv",
        args.base_etl_dir / "train_lab_decile_cutpoints.parquet",
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    for split in args.splits:
        for root in [args.base_etl_dir, args.data_dir]:
            path = root / f"{split}.bin"
            if not path.exists():
                raise FileNotFoundError(path)


def prepare_output(args):
    output_dir = args.output_dir or (args.data_dir / "event_date_sidecar")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def upload_dataframe(conn, frame, table_name, columns_sql, insert_sql, rows_label):
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
    log(f"uploaded {rows_label}: {len(rows):,}")


def upload_reference_tables(conn, args):
    patient_map = pd.read_parquet(args.base_etl_dir / "patient_id_map.parquet")
    patient_map = patient_map[["patient_id_dense", "person_id", "split"]].copy()
    upload_dataframe(
        conn,
        patient_map,
        "tmp_sidecar_patient_map",
        "patient_id_dense bigint, person_id bigint, split text",
        "INSERT INTO tmp_sidecar_patient_map VALUES (%s,%s,%s)",
        "patient map",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_sidecar_patient_map(person_id)",
        label="Index patient map",
    )

    registry = pd.read_csv(args.base_etl_dir / "token_registry.csv")
    registry = registry[["token_id", "token_key", "token_type", "token_type_id"]]
    upload_dataframe(
        conn,
        registry,
        "tmp_sidecar_token_registry",
        "token_id bigint, token_key text, token_type text, token_type_id integer",
        "INSERT INTO tmp_sidecar_token_registry VALUES (%s,%s,%s,%s)",
        "token registry",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_sidecar_token_registry(token_key)",
        label="Index token registry",
    )

    cutpoints = pd.read_parquet(args.base_etl_dir / "train_lab_decile_cutpoints.parquet")
    upload_dataframe(
        conn,
        cutpoints[["measurement_concept_id", "unit_concept_id", "cutpoints"]],
        "tmp_sidecar_lab_cutpoint",
        "measurement_concept_id bigint, unit_concept_id bigint, cutpoints double precision[]",
        "INSERT INTO tmp_sidecar_lab_cutpoint VALUES (%s,%s,%s)",
        "LAB cutpoints",
    )


def materialize_numeric_labs(conn, args):
    execute(conn, "DROP TABLE IF EXISTS tmp_sidecar_numeric_lab_daily")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_sidecar_numeric_lab_daily (
            person_id bigint,
            event_date date,
            measurement_concept_id bigint,
            unit_concept_id bigint,
            value_as_number double precision
        ) ON COMMIT PRESERVE ROWS
        """,
        label="Create numeric LAB table",
    )
    for bucket in range(args.lab_buckets):
        execute(
            conn,
            sql.SQL(
                """
                INSERT INTO tmp_sidecar_numeric_lab_daily
                SELECT
                    m.person_id,
                    m.measurement_date AS event_date,
                    m.measurement_concept_id,
                    COALESCE(m.unit_concept_id, 0) AS unit_concept_id,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY m.value_as_number)
                        AS value_as_number
                FROM {}.measurement AS m
                JOIN tmp_sidecar_patient_map AS p USING (person_id)
                WHERE mod(p.patient_id_dense, %s) = %s
                  AND m.value_as_number IS NOT NULL
                  AND m.measurement_concept_id IS NOT NULL
                  AND m.measurement_concept_id <> 0
                  AND m.measurement_date BETWEEN DATE '1900-01-01' AND %s::date
                  AND NOT (
                      m.measurement_concept_id = ANY(%s)
                      AND m.value_as_number < 0
                  )
                GROUP BY
                    m.person_id,
                    m.measurement_date,
                    m.measurement_concept_id,
                    COALESCE(m.unit_concept_id, 0)
                """
            ).format(sql.Identifier(args.schema)),
            (
                args.lab_buckets,
                bucket,
                args.db_end_date,
                NONNEGATIVE_MEASUREMENT_IDS,
            ),
            label=f"Collapse numeric LAB bucket {bucket + 1}/{args.lab_buckets}",
        )
    execute(conn, "ANALYZE tmp_sidecar_numeric_lab_daily", label="Analyze numeric LAB")
    execute(
        conn,
        "CREATE INDEX ON tmp_sidecar_numeric_lab_daily(person_id)",
        label="Index numeric LAB",
    )


def materialize_base_events(conn, args):
    execute(conn, "DROP TABLE IF EXISTS tmp_sidecar_base_event")
    execute(
        conn,
        sql.SQL(
            """
            CREATE TEMP TABLE tmp_sidecar_base_event ON COMMIT PRESERVE ROWS AS
            WITH birth AS (
                SELECT
                    p.person_id,
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
                    ) AS birth_date,
                    person.gender_concept_id,
                    p.split
                FROM tmp_sidecar_patient_map p
                JOIN {}.person AS person USING (person_id)
            ),
            domain_event AS (
                SELECT e.person_id, e.condition_start_date AS event_date,
                       'DX:' || e.condition_concept_id::text AS token_key,
                       1 AS type_order
                FROM {}.condition_occurrence e JOIN birth b USING(person_id)
                WHERE e.condition_start_date BETWEEN b.birth_date AND %s::date
                  AND e.condition_concept_id <> 0
                UNION ALL
                SELECT e.person_id, e.drug_exposure_start_date,
                       'RX:' || e.drug_concept_id::text, 2
                FROM {}.drug_exposure e JOIN birth b USING(person_id)
                WHERE e.drug_exposure_start_date BETWEEN b.birth_date AND %s::date
                  AND e.drug_concept_id <> 0
                UNION ALL
                SELECT e.person_id, e.procedure_date,
                       'PX:' || e.procedure_concept_id::text, 3
                FROM {}.procedure_occurrence e JOIN birth b USING(person_id)
                WHERE e.procedure_date BETWEEN b.birth_date AND %s::date
                  AND e.procedure_concept_id <> 0
            ),
            numeric_lab AS (
                SELECT
                    n.person_id,
                    n.event_date,
                    CASE
                        WHEN c.cutpoints IS NULL THEN
                            'LAB_TEST:' || n.measurement_concept_id::text
                        ELSE
                            'LAB:' || n.measurement_concept_id::text || ':' ||
                            n.unit_concept_id::text || ':Q' ||
                            lpad((1 + (
                                SELECT COUNT(*) FROM unnest(c.cutpoints) AS q(v)
                                WHERE n.value_as_number > q.v
                            ))::text, 2, '0')
                    END AS token_key,
                    4 AS type_order
                FROM tmp_sidecar_numeric_lab_daily n
                LEFT JOIN tmp_sidecar_lab_cutpoint c
                  ON c.measurement_concept_id = n.measurement_concept_id
                 AND c.unit_concept_id = n.unit_concept_id
            ),
            categorical_lab AS (
                SELECT DISTINCT
                    m.person_id,
                    m.measurement_date AS event_date,
                    'LAB_CAT:' || m.measurement_concept_id::text || ':' ||
                        m.value_as_concept_id::text AS token_key,
                    4 AS type_order
                FROM {}.measurement m
                JOIN birth b USING(person_id)
                WHERE m.measurement_date BETWEEN b.birth_date AND %s::date
                  AND m.measurement_concept_id <> 0
                  AND m.value_as_concept_id IS NOT NULL
                  AND m.value_as_concept_id <> 0
            ),
            death_event AS (
                SELECT d.person_id, d.death_date AS event_date,
                       'SPECIAL:DEATH' AS token_key, 6 AS type_order
                FROM {}.death d JOIN birth b USING(person_id)
                WHERE d.death_date BETWEEN b.birth_date AND %s::date
            ),
            sex_event AS (
                SELECT person_id, birth_date AS event_date,
                       CASE gender_concept_id
                           WHEN 8507 THEN 'SEX:M'
                           WHEN 8532 THEN 'SEX:F'
                           ELSE 'SEX:U'
                       END AS token_key,
                       0 AS type_order
                FROM birth
            ),
            all_event AS (
                SELECT * FROM sex_event
                UNION ALL SELECT * FROM domain_event
                UNION ALL SELECT * FROM numeric_lab
                UNION ALL SELECT * FROM categorical_lab
                UNION ALL SELECT * FROM death_event
            )
            SELECT
                p.patient_id_dense,
                a.person_id,
                (a.event_date - b.birth_date)::integer AS age_in_days,
                COALESCE(r.token_id,
                    CASE
                        WHEN a.token_key LIKE 'DX:%%' THEN 2
                        WHEN a.token_key LIKE 'RX:%%' THEN 3
                        WHEN a.token_key LIKE 'PX:%%' THEN 4
                        ELSE 5
                    END
                )::bigint AS token_id,
                COALESCE(r.token_type_id, a.type_order)::integer AS token_type_id,
                b.split,
                a.event_date,
                a.type_order
            FROM all_event a
            JOIN birth b USING(person_id)
            JOIN tmp_sidecar_patient_map p USING(person_id)
            LEFT JOIN tmp_sidecar_token_registry r USING(token_key)
            WHERE a.event_date IS NOT NULL
              AND a.event_date >= b.birth_date
            """
        ).format(
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
            sql.Identifier(args.schema),
        ),
        (
            args.db_end_date,
            args.db_end_date,
            args.db_end_date,
            args.db_end_date,
            args.db_end_date,
        ),
        label="Materialize base events with dates",
    )
    execute(
        conn,
        "CREATE INDEX ON tmp_sidecar_base_event(split, patient_id_dense)",
        label="Index base events",
    )


def fetch_base_sidecar(conn, split, chunk_size):
    statement = """
        SELECT
            patient_id_dense,
            age_in_days,
            token_id,
            token_type_id,
            event_date
        FROM tmp_sidecar_base_event
        WHERE split = %s
        ORDER BY person_id, event_date, type_order, token_id
    """
    chunks = []
    with conn.cursor(name=f"sidecar_{split}") as cur:
        cur.execute(statement, (split,))
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            chunks.append(
                pd.DataFrame(
                    rows,
                    columns=[
                        "patient_id_dense",
                        "age_in_days",
                        "token_id",
                        "token_type_id",
                        "event_date",
                    ],
                )
            )
    conn.commit()
    if not chunks:
        return pd.DataFrame(
            columns=[
                "patient_id_dense",
                "age_in_days",
                "token_id",
                "token_type_id",
                "event_date",
            ]
        )
    return pd.concat(chunks, ignore_index=True)


def load_genomics_events(data_dir, split):
    path = data_dir / "genomics_token_events.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "patient_id_dense",
                "age_in_days",
                "token_id",
                "token_type_id",
                "event_date",
            ]
        )
    events = pd.read_parquet(path)
    events = events.loc[events["split"] == split].copy()
    if events.empty:
        return events[
            ["patient_id_dense", "age_in_days", "token_id", "token_type_id", "event_date"]
        ]
    return events[
        ["patient_id_dense", "age_in_days", "token_id", "token_type_id", "event_date"]
    ].copy()


def merge_like_augmented_shard(base, genomics):
    if genomics.empty:
        return base.reset_index(drop=True)
    base_arrays = {
        int(pid): group
        for pid, group in base.groupby("patient_id_dense", sort=False)
    }
    add_arrays = {
        int(pid): group.sort_values(
            ["age_in_days", "token_type_id", "token_id"],
            kind="mergesort",
        )
        for pid, group in genomics.groupby("patient_id_dense", sort=False)
    }
    merged_parts = []
    for patient_id in base["patient_id_dense"].drop_duplicates():
        patient_id = int(patient_id)
        block = base_arrays[patient_id]
        extra = add_arrays.pop(patient_id, None)
        if extra is None:
            merged_parts.append(block)
            continue
        merged = pd.concat([block, extra], ignore_index=True, sort=False)
        merged = merged.sort_values(
            ["age_in_days", "token_type_id", "token_id"],
            kind="mergesort",
        )
        merged_parts.append(merged)
    for patient_id in sorted(add_arrays):
        merged_parts.append(add_arrays[patient_id])
    return pd.concat(merged_parts, ignore_index=True, sort=False)


def add_gap_columns(frame):
    frame = frame.copy()
    dates = pd.to_datetime(frame["event_date"])
    frame["calendar_year"] = dates.dt.year.astype("int16")
    previous = dates.groupby(frame["patient_id_dense"]).shift(1)
    gap = (dates - previous).dt.days
    frame["days_since_previous_event"] = gap.astype("Int32")
    frame["same_day_as_previous"] = gap.eq(0).fillna(False)
    frame["gap_over_365_days"] = gap.gt(365).fillna(False)
    return frame


def validate_against_shard(frame, shard_path):
    shard = load_uint32_shard(shard_path)
    keys = frame[
        ["patient_id_dense", "age_in_days", "token_id", "token_type_id"]
    ].to_numpy(dtype=np.uint32)
    if len(keys) != len(shard):
        raise RuntimeError(
            f"{shard_path.name}: sidecar rows={len(keys):,}, shard rows={len(shard):,}"
        )
    if not np.array_equal(keys, np.asarray(shard)):
        mismatch = np.flatnonzero(np.any(keys != np.asarray(shard), axis=1))[0]
        raise RuntimeError(
            f"{shard_path.name}: sidecar key mismatch at row {int(mismatch):,}; "
            f"sidecar={keys[mismatch].tolist()} shard={np.asarray(shard)[mismatch].tolist()}"
        )
    del shard


def write_split_sidecar(conn, args, output_dir, split):
    log(f"\n## SPLIT {split}")
    base = fetch_base_sidecar(conn, split, args.chunk_size)
    log(f"base sidecar rows={len(base):,}")
    validate_against_shard(base, args.base_etl_dir / f"{split}.bin")

    genomics = load_genomics_events(args.data_dir, split)
    if not genomics.empty:
        log(f"genomics sidecar rows={len(genomics):,}")
    merged = merge_like_augmented_shard(base, genomics)
    merged = add_gap_columns(merged)
    merged.insert(0, "row_index", np.arange(len(merged), dtype=np.int64))
    merged.insert(0, "split", split)
    validate_against_shard(merged, args.data_dir / f"{split}.bin")

    path = output_dir / f"{split}_event_dates.parquet"
    merged.to_parquet(path, index=False)
    summary = {
        "split": split,
        "rows": int(len(merged)),
        "patients": int(merged["patient_id_dense"].nunique()),
        "first_event_date": str(merged["event_date"].min()),
        "last_event_date": str(merged["event_date"].max()),
        "same_day_rows": int(merged["same_day_as_previous"].sum()),
        "gap_over_365_rows": int(merged["gap_over_365_days"].sum()),
        "path": str(path),
    }
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    args.base_etl_dir = args.base_etl_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    load_required_files(args)
    output_dir = prepare_output(args)

    with connect(args) as conn:
        upload_reference_tables(conn, args)
        materialize_numeric_labs(conn, args)
        materialize_base_events(conn, args)
        summaries = [
            write_split_sidecar(conn, args, output_dir, split)
            for split in args.splits
        ]

    manifest = {
        "base_etl_dir": str(args.base_etl_dir),
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "db_end_date": args.db_end_date,
        "splits": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"\n[DONE] event-date sidecars written to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
