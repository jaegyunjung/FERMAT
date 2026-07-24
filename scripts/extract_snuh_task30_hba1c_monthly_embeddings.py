#!/usr/bin/env python3
"""Extract observed monthly FERMAT embeddings for the ADM mortality head.

One parquet part is written per patient chunk and validated before resume skips
it.  The output contains dense FERMAT patient IDs only.  Embeddings are stored
as float16 to keep the roughly patient-by-60-month development table tractable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELF_TEST_ONLY = "--self-test" in sys.argv
if not SELF_TEST_ONLY:
    import torch
    from scripts.extract_snuh_task19_fermat_embeddings import (
        autocast_context,
        collate_rows,
        extract_hidden,
        patient_rows_before_index,
    )
    from scripts.run_snuh_task27_primary_direct_risk_dry_run import (
        load_model,
        load_split_data,
    )


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_INPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_inputs"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CKPT = (
    POD_ROOT / "task21" / "outputs" / "block2048_full_10l640_20260629" / "block_2048" / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_monthly_embeddings"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--patient-chunk-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def month_schedule(index_date, index_age_days, months=60):
    index_date = pd.Timestamp(index_date)
    values = []
    for month in range(int(months)):
        date = index_date + pd.DateOffset(months=month)
        next_date = index_date + pd.DateOffset(months=month + 1)
        values.append(
            {
                "month": month,
                "cutoff_age_days": int(index_age_days + (date - index_date).days),
                "month_end_day": int((next_date - index_date).days),
            }
        )
    return values


def build_snapshot_rows(patient, split_data, block_size):
    death = pd.NaT if pd.isna(patient.death_date) else pd.Timestamp(patient.death_date)
    observation_end = pd.Timestamp(patient.observation_end_date)
    index_date = pd.Timestamp(patient.index_date)
    horizon_end = index_date + pd.Timedelta(days=1826)
    stop = min(observation_end, horizon_end, death if not pd.isna(death) else horizon_end)
    rows = []
    for item in month_schedule(index_date, int(patient.index_age_days)):
        month_start = index_date + pd.DateOffset(months=item["month"])
        if month_start >= stop:
            break
        prefix = patient_rows_before_index(
            split_data,
            int(patient.patient_key),
            int(item["cutoff_age_days"]),
            int(block_size),
        )
        if prefix is None or len(prefix) == 0:
            continue
        month_end = index_date + pd.DateOffset(months=item["month"] + 1)
        event = int(not pd.isna(death) and death >= month_start and death < month_end and death <= stop)
        if not event and month_end > stop:
            # Do not treat a partially observed final month as a full
            # event-free month in the discrete-time likelihood.
            break
        rows.append(
            {
                "patient_key": int(patient.patient_key),
                "split": str(patient.split),
                "month": int(item["month"]),
                "death_event": event,
                "rows": prefix,
            }
        )
        if event:
            break
    return rows


def validate_part(path, expected_keys, n_embd):
    if not Path(path).is_file():
        return False
    try:
        frame = pd.read_parquet(path, columns=["patient_key", "month", "death_event"])
    except Exception:
        return False
    if frame[["patient_key", "month"]].duplicated().any():
        return False
    if not set(frame["patient_key"].unique()).issubset(set(expected_keys)):
        return False
    try:
        columns = pd.read_parquet(path).columns
    except Exception:
        return False
    return sum(str(column).startswith("emb_") for column in columns) == int(n_embd)


def self_test():
    schedule = month_schedule("2020-01-31", 20_000, 3)
    assert schedule[0]["cutoff_age_days"] == 20_000
    assert schedule[1]["cutoff_age_days"] - 20_000 == 29
    assert schedule[2]["cutoff_age_days"] - 20_000 == 60
    print("SELF_TEST_OK calendar_month_schedule")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = args.output_dir / "parts"
    part_dir.mkdir(exist_ok=True)
    patients = pd.read_parquet(args.input_dir / "ccw_patients.parquet")
    patients = patients.loc[patients["split"].astype(str).isin(args.splits)].copy()
    required = {"patient_key", "split", "index_date", "index_age_days", "death_date", "observation_end_date"}
    missing = sorted(required - set(patients.columns))
    if missing:
        raise ValueError(f"ccw_patients missing columns: {missing}")
    patients = patients.sort_values(["split", "patient_key"]).reset_index(drop=True)
    model, checkpoint = load_model(args.fermat_ckpt, args.device)
    block_size = int(model.config.block_size)
    n_embd = int(model.config.n_embd)
    split_data = {split: load_split_data(args.data_dir, split) for split in args.splits}
    completed_rows = 0
    started = time.time()
    for chunk_start in range(0, len(patients), int(args.patient_chunk_size)):
        chunk = patients.iloc[chunk_start : chunk_start + int(args.patient_chunk_size)]
        part_number = chunk_start // int(args.patient_chunk_size)
        path = part_dir / f"monthly_embeddings_part_{part_number:05d}.parquet"
        expected_keys = chunk["patient_key"].astype(int).tolist()
        if args.resume and validate_part(path, expected_keys, n_embd):
            log(f"[RESUME] {path}")
            continue
        tasks = []
        for patient in chunk.itertuples(index=False):
            tasks.extend(build_snapshot_rows(patient, split_data[str(patient.split)], block_size))
        output_rows = []
        for start in range(0, len(tasks), int(args.embedding_batch_size)):
            batch = tasks[start : start + int(args.embedding_batch_size)]
            idx, age, token_type, lengths = collate_rows(batch, args.device)
            with torch.no_grad(), autocast_context(args.device, args.dtype):
                embedding = extract_hidden(model, idx, age, token_type, lengths, "last")
            values = embedding.detach().float().cpu().numpy().astype(np.float16)
            for task, vector in zip(batch, values):
                row = {key: task[key] for key in ("patient_key", "split", "month", "death_event")}
                row.update({f"emb_{i:04d}": vector[i] for i in range(n_embd)})
                output_rows.append(row)
        frame = pd.DataFrame(output_rows)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        if not validate_part(path, expected_keys, n_embd):
            raise RuntimeError(f"written part failed validation: {path}")
        completed_rows += len(frame)
        log(
            f"[RAW SAVED] part={part_number} patients={len(chunk)} rows={len(frame):,} "
            f"elapsed={time.time() - started:,.1f}s"
        )
    manifest = {
        "status": "MONTHLY_EMBEDDINGS_COMPLETE",
        "patients": int(len(patients)),
        "splits": args.splits,
        "months": 60,
        "n_embd": n_embd,
        "checkpoint_step": int(checkpoint.get("iter", -1)),
        "parts": len(list(part_dir.glob("monthly_embeddings_part_*.parquet"))),
        "new_rows_this_run": int(completed_rows),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
