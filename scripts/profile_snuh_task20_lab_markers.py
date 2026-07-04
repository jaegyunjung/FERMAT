#!/usr/bin/env python3
"""Profile LAB marker candidates for Task 20 clinical comparators.

This is intentionally a metadata-first step. It does not scan the full
measurement table. Instead, it reads the existing Task 15/16 numeric LAB
frequency artifacts and joins their measurement/unit concept IDs to OMOP
concept metadata. The output is a reviewed candidate list for later raw-value
feature extraction.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
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
DEFAULT_OUTPUT_DIR = POD_ROOT / "task20" / "outputs" / "lab_marker_profile"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task20_lab_marker_profile"

MARKER_PATTERNS = {
    "hba1c": [
        r"\bhba1c\b",
        r"hemoglobin\s*a1c",
        r"haemoglobin\s*a1c",
        r"glycated\s*hemoglobin",
        r"glycohemoglobin",
    ],
    "glucose": [
        r"\bglucose\b",
    ],
    "creatinine": [
        r"\bcreatinine\b",
    ],
    "egfr": [
        r"\begfr\b",
        r"\bgfr\b",
        r"glomerular\s+filtration",
    ],
    "ast": [
        r"\bast\b",
        r"\bsgot\b",
        r"aspartate\s+aminotransferase",
    ],
    "alt": [
        r"\balt\b",
        r"\bsgpt\b",
        r"alanine\s+aminotransferase",
    ],
    "total_bilirubin": [
        r"total.*bilirubin",
        r"bilirubin.*total",
    ],
    "platelet": [
        r"\bplatelet",
        r"\bplt\b",
        r"thrombocyte",
    ],
    "afp": [
        r"\bafp\b",
        r"alpha.?fetoprotein",
        r"alpha.?foetoprotein",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--top-n-per-marker", type=int, default=30)
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
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
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


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def load_lab_artifacts(data_dir: Path):
    stats_path = data_dir / "train_numeric_lab_stats.parquet"
    cutpoints_path = data_dir / "train_lab_decile_cutpoints.parquet"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    if not cutpoints_path.exists():
        raise FileNotFoundError(cutpoints_path)

    stats = pd.read_parquet(stats_path)
    cutpoints = pd.read_parquet(cutpoints_path)
    for frame in (stats, cutpoints):
        frame["measurement_concept_id"] = frame["measurement_concept_id"].astype(np.int64)
        frame["unit_concept_id"] = frame["unit_concept_id"].fillna(0).astype(np.int64)
    labs = stats.merge(
        cutpoints,
        on=["measurement_concept_id", "unit_concept_id"],
        how="left",
        suffixes=("", "_cutpoint"),
    )
    return labs, str(stats_path), str(cutpoints_path)


def upload_concept_ids(conn, ids: list[int]):
    execute(conn, "DROP TABLE IF EXISTS tmp_task20_concept_id")
    execute(
        conn,
        """
        CREATE TEMP TABLE tmp_task20_concept_id (
            concept_id bigint PRIMARY KEY
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    rows = [(int(value),) for value in sorted(set(ids)) if int(value) != 0]
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO tmp_task20_concept_id VALUES (%s)", rows)
    conn.commit()
    log(f"uploaded concept ids: {len(rows):,}")


def fetch_concepts(conn, args, ids: list[int]):
    upload_concept_ids(conn, ids)
    if not ids:
        return pd.DataFrame(
            columns=[
                "concept_id",
                "concept_name",
                "concept_code",
                "vocabulary_id",
                "domain_id",
                "concept_class_id",
            ]
        )
    return query_df(
        conn,
        sql.SQL(
            """
            SELECT
                c.concept_id::bigint,
                c.concept_name,
                c.concept_code,
                c.vocabulary_id,
                c.domain_id,
                c.concept_class_id
            FROM {}.concept c
            JOIN tmp_task20_concept_id t
              ON t.concept_id = c.concept_id
            ORDER BY c.concept_id
            """
        ).format(sql.Identifier(args.schema)),
        label="Fetch OMOP concept names",
    )


def add_concept_metadata(labs: pd.DataFrame, concepts: pd.DataFrame):
    measurement = concepts.rename(
        columns={
            "concept_id": "measurement_concept_id",
            "concept_name": "measurement_concept_name",
            "concept_code": "measurement_concept_code",
            "vocabulary_id": "measurement_vocabulary_id",
            "domain_id": "measurement_domain_id",
            "concept_class_id": "measurement_concept_class_id",
        }
    )
    unit = concepts.rename(
        columns={
            "concept_id": "unit_concept_id",
            "concept_name": "unit_concept_name",
            "concept_code": "unit_concept_code",
            "vocabulary_id": "unit_vocabulary_id",
            "domain_id": "unit_domain_id",
            "concept_class_id": "unit_concept_class_id",
        }
    )
    merged = labs.merge(measurement, on="measurement_concept_id", how="left")
    merged = merged.merge(unit, on="unit_concept_id", how="left")
    merged["measurement_concept_name"] = merged["measurement_concept_name"].fillna("")
    merged["measurement_concept_code"] = merged["measurement_concept_code"].fillna("")
    merged["unit_concept_name"] = merged["unit_concept_name"].fillna("")
    merged["unit_concept_code"] = merged["unit_concept_code"].fillna("")
    return merged


def cutpoint_columns(frame: pd.DataFrame):
    if "cutpoints" not in frame.columns:
        return frame
    labels = [f"p{value}" for value in range(10, 100, 10)]
    values = frame["cutpoints"].tolist()
    expanded = []
    for item in values:
        if isinstance(item, str):
            item = item.strip("{}[]")
            parts = [part.strip() for part in item.split(",") if part.strip()]
            row = [float(part) for part in parts[:9]]
        elif isinstance(item, (list, tuple, np.ndarray)):
            row = [float(x) for x in list(item)[:9]]
        else:
            row = []
        row = row + [np.nan] * (9 - len(row))
        expanded.append(row[:9])
    cut = pd.DataFrame(expanded, columns=labels)
    return pd.concat([frame.drop(columns=["cutpoints"]), cut], axis=1)


def marker_matches(row, compiled):
    text = " ".join(
        str(row.get(column, "") or "")
        for column in [
            "measurement_concept_name",
            "measurement_concept_code",
            "measurement_vocabulary_id",
            "measurement_concept_class_id",
        ]
    ).lower()
    matches = []
    for marker, patterns in compiled.items():
        if any(pattern.search(text) for pattern in patterns):
            matches.append(marker)
    return matches, text


def build_candidates(labs: pd.DataFrame):
    compiled = {
        marker: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for marker, patterns in MARKER_PATTERNS.items()
    }
    rows = []
    for row in labs.to_dict("records"):
        markers, text = marker_matches(row, compiled)
        for marker in markers:
            out = dict(row)
            out["marker"] = marker
            out["match_text"] = text
            rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["marker"])
    candidates = pd.DataFrame(rows)
    return candidates.sort_values(
        ["marker", "daily_rows", "patients"],
        ascending=[True, False, False],
    )


def summarize_candidates(candidates: pd.DataFrame):
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "marker",
                "candidate_pairs",
                "frequent_pairs",
                "train_daily_rows",
                "train_patients_max",
                "top_measurement_concept_id",
                "top_measurement_name",
                "top_unit_name",
            ]
        )
    rows = []
    for marker, sub in candidates.groupby("marker", sort=True):
        sub = sub.sort_values(["daily_rows", "patients"], ascending=False)
        top = sub.iloc[0]
        rows.append(
            {
                "marker": marker,
                "candidate_pairs": int(len(sub)),
                "frequent_pairs": int(sub.get("is_frequent", pd.Series(False, index=sub.index)).fillna(False).sum()),
                "train_daily_rows": int(sub["daily_rows"].sum()),
                "train_patients_max": int(sub["patients"].max()),
                "top_measurement_concept_id": int(top["measurement_concept_id"]),
                "top_measurement_name": top.get("measurement_concept_name", ""),
                "top_unit_concept_id": int(top.get("unit_concept_id", 0)),
                "top_unit_name": top.get("unit_concept_name", ""),
                "top_daily_rows": int(top["daily_rows"]),
                "top_patients": int(top["patients"]),
            }
        )
    return pd.DataFrame(rows).sort_values("train_daily_rows", ascending=False)


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)

    labs, stats_path, cutpoints_path = load_lab_artifacts(args.data_dir)
    log(f"numeric lab pairs={len(labs):,}")
    concept_ids = (
        labs["measurement_concept_id"].fillna(0).astype(np.int64).tolist()
        + labs["unit_concept_id"].fillna(0).astype(np.int64).tolist()
    )
    with connect(args) as conn:
        concepts = fetch_concepts(conn, args, concept_ids)

    labs = add_concept_metadata(labs, concepts)
    labs = cutpoint_columns(labs)
    candidates = build_candidates(labs)
    summary = summarize_candidates(candidates)

    all_pairs_path = args.output_dir / "numeric_lab_pairs_with_concepts.csv"
    candidates_path = args.output_dir / "candidate_marker_lab_pairs.csv"
    top_candidates_path = args.output_dir / "candidate_marker_lab_pairs_top.csv"
    summary_path = args.output_dir / "marker_summary.csv"
    labs.to_csv(all_pairs_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    (
        candidates.groupby("marker", group_keys=False)
        .head(args.top_n_per_marker)
        .to_csv(top_candidates_path, index=False)
        if not candidates.empty
        else candidates.to_csv(top_candidates_path, index=False)
    )
    summary.to_csv(summary_path, index=False)

    manifest = {
        "data_dir": str(args.data_dir),
        "stats_path": stats_path,
        "cutpoints_path": cutpoints_path,
        "output_dir": str(args.output_dir),
        "marker_patterns": MARKER_PATTERNS,
        "outputs": {
            "all_pairs": str(all_pairs_path),
            "candidate_pairs": str(candidates_path),
            "top_candidate_pairs": str(top_candidates_path),
            "summary": str(summary_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n## marker summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n## top marker candidates", flush=True)
    if candidates.empty:
        print("no candidates", flush=True)
    else:
        columns = [
            "marker",
            "measurement_concept_id",
            "measurement_concept_name",
            "measurement_vocabulary_id",
            "measurement_concept_code",
            "unit_concept_id",
            "unit_concept_name",
            "daily_rows",
            "patients",
            "is_frequent",
            "p50",
        ]
        available = [column for column in columns if column in candidates.columns]
        print(
            candidates.groupby("marker", group_keys=False)
            .head(args.top_n_per_marker)[available]
            .to_string(index=False),
            flush=True,
        )
    print("\n## outputs", flush=True)
    print(json.dumps(manifest["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
