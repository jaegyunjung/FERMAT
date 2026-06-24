#!/usr/bin/env python3
"""Append SNUH Task 17 GENOMICS tokens to a completed Task 15 ETL directory.

This script creates a new ETL directory, preserving the original Task 15
artifacts. It adds a unified GENOMICS token namespace from:

- 2004-2020 note/NGS parser CSVs under Task 17 outputs
- 2021+ observation clinical-summary biomarker text from observation 기타/1340204

Source provenance is retained in genomics_token_events.csv. The training shards
receive only token_id and token_type_id, as usual.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    psycopg = None
    dict_row = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_ETL_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42"
DEFAULT_TASK17_DIR = POD_ROOT / "scripts" / "outputs" / "task17_molecular_biomarker_audit"
DEFAULT_OUTPUT_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"

DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task17_add_genomics_tokens"

GENOMICS_TOKEN_TYPE_ID = 9
SNUH_CDM_PASSWORD_CACHE = None

NGS_EVENTS = "ngs_fermat_events.csv"
MOLECULAR_FINAL = "molecular_event_tokens_final.csv"

DB_PREFILTER = (
    r"(?i)(EGFR|ALK|KRAS|NRAS|BRAF|ROS1|BRCA1|BRCA2|HER2|ERBB2|"
    r"PD[- ]?L1|22C3|SP263|CPS|TPS|MSS|MMRd|MMRp|"
    r"L858R|E19del|exon ?19|del ?19|G12C|G12D|G12V|G12F|V600E|T790M)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--etl-dir", type=Path, default=DEFAULT_ETL_DIR)
    parser.add_argument("--task17-output-dir", type=Path, default=DEFAULT_TASK17_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="8min")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-db-observation", action="store_true")
    return parser.parse_args()


def compact(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sanitize_part(value: str) -> str:
    value = compact(value, 120).upper()
    value = value.replace(" ", "_").replace("/", "_").replace(":", "_")
    value = re.sub(r"[^A-Z0-9_.+<>\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "UNKNOWN"


def genomics_key(gene: str, result_type: str, result_value: str) -> str:
    return "GENOMICS:" + ":".join(
        [sanitize_part(gene), sanitize_part(result_type), sanitize_part(result_value)]
    )


def source_key(prefix: str, value: str) -> str:
    return "GENOMICS:" + sanitize_part(prefix) + ":" + sanitize_part(value)


def add_event(events, *, person_id, event_date, token_key, source_tier,
              source_table, source_id, source_detail="", evidence=""):
    if not person_id or not event_date or not token_key:
        return
    events.append(
        {
            "person_id": int(person_id),
            "event_date": str(event_date)[:10],
            "token_key": token_key,
            "source_tier": source_tier,
            "source_table": source_table,
            "source_id": "" if source_id is None else str(source_id),
            "source_detail": compact(source_detail, 240),
            "evidence": compact(evidence, 700),
        }
    )


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def normalize_molecular_final_token(token: str) -> str | None:
    parts = [part for part in (token or "").split(":") if part]
    if len(parts) < 2 or parts[0] != "MOL":
        return source_key("LEGACY", token)
    if len(parts) == 3:
        _, marker, value = parts
        if marker == "MSI":
            return genomics_key("MSI", "STATUS", value)
        return genomics_key(marker, "STATUS", value)
    if len(parts) >= 5:
        _, gene, exon, variant, status = parts[:5]
        result_value = variant
        if status and status.lower() not in {"positive", "detected"}:
            result_value = f"{variant}_{status}"
        return genomics_key(gene, "MUTATION", result_value)
    return source_key("LEGACY", token)


def load_legacy_events(task17_dir: Path):
    events = []

    for row in read_csv_rows(task17_dir / NGS_EVENTS):
        token = row.get("token_id") or ""
        if not token:
            continue
        add_event(
            events,
            person_id=row.get("person_id"),
            event_date=row.get("event_date"),
            token_key=source_key("NGS", token),
            source_tier="canonical_report",
            source_table=row.get("source_table") or "note",
            source_id=row.get("source_id"),
            source_detail=row.get("source_detail") or row.get("parser_source") or "",
            evidence=token,
        )

    for row in read_csv_rows(task17_dir / MOLECULAR_FINAL):
        token = row.get("token") or ""
        token_key = normalize_molecular_final_token(token)
        if not token_key:
            continue
        evidence = " | ".join(
            part
            for part in [
                row.get("test_item") or "",
                row.get("gene") or "",
                row.get("exon") or "",
                row.get("variant_label") or "",
                row.get("result_value") or "",
                row.get("diagnosis") or "",
            ]
            if part
        )
        add_event(
            events,
            person_id=row.get("person_id"),
            event_date=row.get("note_date"),
            token_key=token_key,
            source_tier="canonical_report",
            source_table="note",
            source_id=row.get("note_id"),
            source_detail=row.get("note_source_value2") or row.get("pathology_no") or "",
            evidence=evidence or token,
        )

    return events


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def classify_observation_text(text: str):
    text = norm(text)
    upper = text.upper()
    output = []

    def emit(token_key):
        output.append(token_key)

    if re.search(r"\b(L858R|T790M|E19DEL|EXON ?19|DEL ?19|19 ?DEL)\b", upper) or re.search(r"\bEGFR\b", upper):
        if "L858R" in upper:
            emit(genomics_key("EGFR", "MUTATION", "L858R"))
        if "T790M" in upper:
            emit(genomics_key("EGFR", "MUTATION", "T790M"))
        if re.search(r"E19DEL|EXON ?19|DEL ?19|19 ?DEL", upper):
            emit(genomics_key("EGFR", "MUTATION", "EXON19DEL"))
        if re.search(r"\bEGFR\b.{0,15}\b(WT|WILD|\-)\b|\bWT\s*EGFR\b", upper):
            emit(genomics_key("EGFR", "STATUS", "WT"))

    ordered = re.search(r"\bEGFR/KRAS/ALK/ROS1\s*[: ]*([+\-WT/ ]{3,20})", upper)
    if ordered:
        parts = re.findall(r"\+|\-|WT", ordered.group(1))
        if len(parts) >= 4:
            for gene, value in zip(["EGFR", "KRAS", "ALK", "ROS1"], parts[:4]):
                emit(genomics_key(gene, "STATUS", "POSITIVE" if value == "+" else "NEGATIVE_OR_WT"))

    if re.search(r"\bALK\b|EML4[- ]ALK", upper):
        if re.search(r"EML4[- ]ALK|ALK\s*(FISH)?\s*(\+|POSITIVE|POS)\b|ALK-POSITIVE|ALK\(\+\)", upper):
            emit(genomics_key("ALK", "STATUS", "POSITIVE"))
        elif re.search(r"\bALK\b.{0,12}(WT|WILD|NEGATIVE|NEG|\-)|ALK-NEGATIVE", upper):
            emit(genomics_key("ALK", "STATUS", "NEGATIVE_OR_WT"))

    for gene in ["KRAS", "NRAS"]:
        if re.search(rf"\b{gene}\b", upper) or (gene == "KRAS" and re.search(r"\bG12[CDVF]\b", upper)):
            for mutation in ["G12C", "G12D", "G12V", "G12F"]:
                if re.search(rf"\b{mutation}\b", upper):
                    emit(genomics_key(gene, "MUTATION", mutation))

    if re.search(r"\bBRAF\b|\bV600E\b", upper):
        if "V600E" in upper:
            emit(genomics_key("BRAF", "MUTATION", "V600E"))
        elif re.search(r"\bBRAF\b.{0,12}(WT|WILD|\-)", upper):
            emit(genomics_key("BRAF", "STATUS", "WT"))
        elif re.search(r"\bBRAF\b.{0,12}(MT|MUT|MUTATION|\+)", upper):
            emit(genomics_key("BRAF", "STATUS", "MUTATED_OR_POSITIVE"))

    if re.search(r"\bROS1\b", upper):
        if re.search(r"\bROS1\b.{0,12}(\+|POSITIVE|POS)", upper):
            emit(genomics_key("ROS1", "STATUS", "POSITIVE"))
        elif re.search(r"\bROS1\b.{0,12}(WT|WILD|NEGATIVE|NEG|\-)", upper):
            emit(genomics_key("ROS1", "STATUS", "NEGATIVE_OR_WT"))

    if re.search(r"\bBRCA1\b|\bBRCA2\b|\bBRCA\b", upper):
        if re.search(r"\bBRCA1\b.{0,25}(MT|MUT|MUTATION|\+)", upper):
            emit(genomics_key("BRCA1", "STATUS", "MUTATED_OR_POSITIVE"))
        elif re.search(r"\bBRCA2\b.{0,25}(MT|MUT|MUTATION|\+)", upper):
            emit(genomics_key("BRCA2", "STATUS", "MUTATED_OR_POSITIVE"))
        elif re.search(r"\bBRCA\b.{0,25}(MT|MUT|MUTATION|\+)", upper):
            emit(genomics_key("BRCA", "STATUS", "MUTATED_OR_POSITIVE"))
        elif re.search(r"\bBRCA\b.{0,25}\(-\)", upper):
            emit(genomics_key("BRCA", "STATUS", "NEGATIVE"))

    for match in re.finditer(r"\b(HER2|ERBB2)\b.{0,25}", upper):
        fragment = match.group(0)
        if re.search(r"3\+|\+\+\+|POSITIVE|FISH\+|AMP", fragment):
            emit(genomics_key("HER2", "STATUS", "POSITIVE"))
        elif re.search(r"(^|[^A-Z0-9])(NEGATIVE|NEG|\-)([^A-Z0-9]|$)", fragment):
            emit(genomics_key("HER2", "STATUS", "NEGATIVE"))

    if re.search(r"PD[- ]?L1|22C3|SP263|TPS|CPS", upper):
        for label in ["TPS", "CPS"]:
            for match in re.finditer(rf"{label}\s*[:=]?\s*(<\s*)?(\d+)", upper):
                emit(genomics_key("PDL1", label, (match.group(1) or "").replace(" ", "") + match.group(2)))
        for match in re.finditer(r"PD[- ]?L1[^0-9<]{0,30}(<\s*)?(\d+)\s*%", upper):
            emit(genomics_key("PDL1", "PERCENT", (match.group(1) or "").replace(" ", "") + match.group(2)))

    if re.search(r"\bMSS\b", upper):
        emit(genomics_key("MSI", "STATUS", "MSS"))
    if re.search(r"\bMMRD\b", upper):
        emit(genomics_key("MMR", "STATUS", "D"))
    if re.search(r"\bMMRP\b", upper):
        emit(genomics_key("MMR", "STATUS", "P"))
    if re.search(r"\bMSI[- ]?H\b", upper):
        emit(genomics_key("MSI", "STATUS", "MSI-H"))
    if re.search(r"\bMSI[- ]?L\b", upper):
        emit(genomics_key("MSI", "STATUS", "MSI-L"))

    return list(dict.fromkeys(output))


def connect(args):
    if psycopg is None:
        raise RuntimeError("psycopg is required. Run this in the Pod environment.")
    global SNUH_CDM_PASSWORD_CACHE
    if SNUH_CDM_PASSWORD_CACHE is None:
        SNUH_CDM_PASSWORD_CACHE = os.environ.get("SNUH_CDM_PASSWORD") or getpass.getpass("SNUH CDM password: ")
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=SNUH_CDM_PASSWORD_CACHE,
        sslmode=args.sslmode,
        application_name=APPLICATION_NAME,
    )


def fetch_observation_events(args):
    if args.skip_db_observation:
        return []
    events = []
    with connect(args) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
            cur.execute(
                f"""
                SELECT
                    observation_id,
                    person_id,
                    observation_date,
                    coalesce(value_source_value,'') || ' ' ||
                    coalesce(value_as_string,'') || ' ' ||
                    coalesce(qualifier_source_value,'') AS text_value
                FROM {args.schema}.observation
                WHERE observation_date >= %s::date
                  AND observation_date <= %s::date
                  AND observation_source_value = '기타'
                  AND observation_concept_id = 1340204
                  AND (
                        coalesce(value_source_value,'') || ' ' ||
                        coalesce(value_as_string,'') || ' ' ||
                        coalesce(qualifier_source_value,'')
                      ) ~* %s
                ORDER BY observation_date, observation_id
                """,
                (args.since, args.db_end_date, DB_PREFILTER),
            )
            for row in cur.fetchall():
                for token_key in classify_observation_text(row["text_value"]):
                    add_event(
                        events,
                        person_id=row["person_id"],
                        event_date=row["observation_date"],
                        token_key=token_key,
                        source_tier="weak_clinical_summary",
                        source_table="observation",
                        source_id=row["observation_id"],
                        source_detail="observation_source_value=기타; observation_concept_id=1340204",
                        evidence=row["text_value"],
                    )
    return events


def require_files(etl_dir: Path):
    required = [
        "train.bin",
        "val.bin",
        "test.bin",
        "manifest.json",
        "token_registry.csv",
        "patient_id_map.parquet",
        "event_summary.csv",
        "shard_validation.csv",
        "sha256.json",
    ]
    missing = [name for name in required if not (etl_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing ETL artifacts: " + ", ".join(missing))


def prepare_output(input_dir: Path, output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for path in input_dir.iterdir():
        if path.is_file() and path.name not in {"train.bin", "val.bin", "test.bin"}:
            shutil.copy2(path, output_dir / path.name)


def load_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint32)
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
        yield int(patient_ids[start]), start, end


def write_augmented_shards(input_dir: Path, output_dir: Path, genomics_events: pd.DataFrame):
    validation = pd.read_csv(output_dir / "shard_validation.csv")
    event_summary = pd.read_csv(output_dir / "event_summary.csv")
    max_new_token_id = int(genomics_events["token_id"].max()) if len(genomics_events) else None

    for split in ["train", "val", "test"]:
        base_path = input_dir / f"{split}.bin"
        base_raw = np.memmap(base_path, dtype=np.uint32, mode="r")
        if base_raw.size % 4:
            raise ValueError(f"{base_path} does not contain 4-column uint32 rows")
        base = base_raw.reshape(-1, 4)
        add = genomics_events.loc[genomics_events["split"] == split]
        add_arrays = {}
        if not add.empty:
            for patient_id, group in add.groupby("patient_id_dense", sort=False):
                rows = group[
                    ["patient_id_dense", "age_in_days", "token_id", "token_type_id"]
                ].to_numpy(dtype=np.uint32)
                order = np.lexsort((rows[:, 2], rows[:, 3], rows[:, 1]))
                add_arrays[int(patient_id)] = rows[order]

        out_path = output_dir / f"{split}.bin"
        total_rows = len(base) + len(add)
        out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_rows, 4))
        cursor = 0
        for patient_id, start, end in iter_patient_blocks(base):
            block = np.asarray(base[start:end])
            extra = add_arrays.pop(patient_id, None)
            if extra is None:
                out[cursor: cursor + len(block)] = block
                cursor += len(block)
                continue
            merged = np.vstack([block, extra])
            order = np.lexsort((merged[:, 2], merged[:, 3], merged[:, 1]))
            merged = merged[order]
            out[cursor: cursor + len(merged)] = merged
            cursor += len(merged)
        for patient_id in sorted(add_arrays):
            extra = add_arrays[patient_id]
            out[cursor: cursor + len(extra)] = extra
            cursor += len(extra)
        if cursor != total_rows:
            raise RuntimeError(f"{split}: wrote {cursor} rows, expected {total_rows}")
        out.flush()
        del out
        del base
        del base_raw

        split_mask = validation["split"] == split
        validation.loc[split_mask, "events"] = validation.loc[split_mask, "events"].astype(int) + int(len(add))
        validation.loc[split_mask, "file_gb"] = out_path.stat().st_size / 1024**3
        if max_new_token_id is not None:
            validation.loc[split_mask, "max_token_id"] = np.maximum(
                validation.loc[split_mask, "max_token_id"].astype(int),
                max_new_token_id,
            )

        event_summary = pd.concat(
            [
                event_summary,
                pd.DataFrame(
                    [
                        {
                            "split": split,
                            "token_type_id": GENOMICS_TOKEN_TYPE_ID,
                            "events": int(len(add)),
                            "patients": int(add["patient_id_dense"].nunique()),
                            "unknown_events": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    validation.to_csv(output_dir / "shard_validation.csv", index=False)
    event_summary.sort_values(["split", "token_type_id"]).to_csv(output_dir / "event_summary.csv", index=False)


def write_checksums(output_dir: Path):
    checksums = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"task15.log", "sha256.json"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[path.name] = digest.hexdigest()
    (output_dir / "sha256.json").write_text(
        json.dumps(checksums, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    input_dir = args.etl_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    task17_dir = args.task17_output_dir.expanduser().resolve()
    require_files(input_dir)

    print("## INPUT")
    print(f"etl_dir={input_dir}")
    print(f"task17_output_dir={task17_dir}")
    print(f"output_dir={output_dir}")
    print(f"genomics_token_type_id={GENOMICS_TOKEN_TYPE_ID}")

    legacy_events = load_legacy_events(task17_dir)
    observation_events = fetch_observation_events(args)
    all_events = legacy_events + observation_events
    if not all_events:
        raise RuntimeError("No genomics events were found.")

    events = pd.DataFrame(all_events).drop_duplicates(
        ["person_id", "event_date", "token_key", "source_table", "source_id"]
    )
    print("\n## DISCOVERED GENOMICS EVENTS")
    print(f"legacy_events={len(legacy_events)}")
    print(f"observation_events={len(observation_events)}")
    print(f"deduplicated_events={len(events)}")
    print(f"persons={events.person_id.nunique()}")
    print(f"unique_token_keys={events.token_key.nunique()}")
    print("source_tier_counts=" + json.dumps(events.source_tier.value_counts().to_dict(), ensure_ascii=False))

    patient_map = pd.read_parquet(input_dir / "patient_id_map.parquet")
    patient_map = patient_map[["patient_id_dense", "person_id", "split"]].copy()
    events = events.merge(patient_map, on="person_id", how="inner")
    dropped = len(all_events) - len(events)
    print(f"events_after_patient_map={len(events)}")
    print(f"events_dropped_not_in_etl_patient_map={dropped}")

    if events.empty:
        raise RuntimeError("No genomics events matched ETL patient_id_map.")

    prepare_output(input_dir, output_dir, args.overwrite)

    registry = pd.read_csv(input_dir / "token_registry.csv")
    next_id = int(registry["token_id"].max()) + 1
    token_counts = events.loc[events["split"] == "train", "token_key"].value_counts()
    new_tokens = pd.DataFrame(
        [
            {
                "token_id": next_id + index,
                "token_key": token_key,
                "token_type": "GENOMICS",
                "token_type_id": GENOMICS_TOKEN_TYPE_ID,
                "frequency": int(token_counts.get(token_key, 0)),
            }
            for index, token_key in enumerate(sorted(events["token_key"].unique()))
        ]
    )
    registry_aug = pd.concat([registry, new_tokens], ignore_index=True, sort=False)
    registry_aug.to_csv(output_dir / "token_registry.csv", index=False)
    registry_aug.to_parquet(output_dir / "token_registry.parquet", index=False)

    token_map = new_tokens.set_index("token_key")["token_id"].to_dict()
    events["token_id"] = events["token_key"].map(token_map).astype(np.uint32)
    events["token_type_id"] = GENOMICS_TOKEN_TYPE_ID

    birth_year = patient_map[["person_id"]].drop_duplicates()
    # Age is already available only in the ETL shards, so use an approximate
    # lookup from existing patient events: first event age plus date order is not
    # recoverable from the shard. Querying person birth date would require DB
    # access even when --skip-db-observation is used, so use a deterministic
    # direct DB lookup here.
    with connect(args) as conn:
        person_ids = [int(value) for value in events["person_id"].drop_duplicates()]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT
                    person_id,
                    make_date(
                        year_of_birth,
                        CASE WHEN month_of_birth BETWEEN 1 AND 12 THEN month_of_birth ELSE 7 END,
                        CASE WHEN day_of_birth BETWEEN 1 AND 28 THEN day_of_birth ELSE 1 END
                    ) AS birth_date
                FROM {args.schema}.person
                WHERE person_id = ANY(%s)
                """,
                (person_ids,),
            )
            birth = pd.DataFrame([dict(row) for row in cur.fetchall()])
    events = events.merge(birth, on="person_id", how="left")
    if events["birth_date"].isna().any():
        raise RuntimeError("Missing birth_date for genomics events after person lookup.")
    events["event_date_dt"] = pd.to_datetime(events["event_date"])
    events["birth_date_dt"] = pd.to_datetime(events["birth_date"])
    events["age_in_days"] = (events["event_date_dt"] - events["birth_date_dt"]).dt.days
    events = events.loc[events["age_in_days"] >= 0].copy()
    events["age_in_days"] = events["age_in_days"].astype(np.uint32)

    provenance_columns = [
        "person_id",
        "patient_id_dense",
        "split",
        "event_date",
        "age_in_days",
        "token_id",
        "token_key",
        "source_tier",
        "source_table",
        "source_id",
        "source_detail",
        "evidence",
    ]
    events[provenance_columns].sort_values(
        ["person_id", "event_date", "token_key", "source_table", "source_id"]
    ).to_csv(output_dir / "genomics_token_events.csv", index=False)
    events[provenance_columns].to_parquet(output_dir / "genomics_token_events.parquet", index=False)

    write_augmented_shards(input_dir, output_dir, events)

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    manifest.update(
        {
            "genomics_tokens_added": True,
            "genomics_token_type_id": GENOMICS_TOKEN_TYPE_ID,
            "genomics_source_note": (
                "Unified GENOMICS tokens combine canonical report-derived Task 17 "
                "events and post-2021 weak clinical-summary observation events; "
                "source provenance is retained in genomics_token_events.csv."
            ),
            "genomics_input_etl_dir": str(input_dir),
            "genomics_task17_output_dir": str(task17_dir),
            "genomics_events": int(len(events)),
            "genomics_persons": int(events["person_id"].nunique()),
            "genomics_unique_token_keys": int(events["token_key"].nunique()),
            "stored_token_id_min": int(registry_aug["token_id"].min()),
            "stored_token_id_max": int(registry_aug["token_id"].max()),
            "model_vocab_size": int(registry_aug["token_id"].max()) + 2,
            "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = (
        events.groupby(["source_tier", "token_key"], dropna=False)
        .agg(events=("token_key", "size"), persons=("person_id", "nunique"))
        .reset_index()
        .sort_values(["persons", "events"], ascending=False)
    )
    summary.to_csv(output_dir / "genomics_token_summary.csv", index=False)
    write_checksums(output_dir)

    print("\n## OUTPUT")
    print(f"output_dir={output_dir}")
    print(f"genomics_events={len(events)}")
    print(f"genomics_persons={events.person_id.nunique()}")
    print(f"genomics_unique_token_keys={events.token_key.nunique()}")
    print(f"new_model_vocab_size={manifest['model_vocab_size']}")
    print("\n## TOP GENOMICS TOKENS")
    print(summary.head(40).to_csv(index=False).strip())
    print("\n## DONE")


if __name__ == "__main__":
    main()
