#!/usr/bin/env python3
"""Profile SNUH FERMAT sequence lengths for Task 21 block-size decisions.

The goal is to quantify how much patient history is truncated by block_size=512
and whether 1024 or 2048 is the better next context length. It reports both the
full ETL patient trajectory length and the Task 19/20 pre-index length used for
downstream disease-risk evaluation.
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

from utils import get_p2i, load_data  # noqa: E402


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task21" / "outputs" / "sequence_length_profile_2018_5y"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"
THRESHOLDS = [128, 256, 512, 1024, 2048, 4096]
BLOCK_SIZES = [512, 1024, 2048, 4096]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--thresholds", nargs="+", type=int, default=THRESHOLDS)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=BLOCK_SIZES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def parquet_columns(path: Path):
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except ModuleNotFoundError:
        return pd.read_parquet(path).columns.tolist()


def resolve_embedding_file(args):
    if args.embedding_file is not None:
        path = args.embedding_file.expanduser().resolve()
    else:
        path = (
            args.embedding_dir.expanduser().resolve()
            / f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizon}_last.parquet"
        )
    return path if path.exists() else None


def length_stats(values, thresholds):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    row = {"n": int(len(values))}
    if len(values) == 0:
        for key in ["mean", "p50", "p75", "p90", "p95", "p99", "max"]:
            row[key] = np.nan
        for threshold in thresholds:
            row[f"gt_{threshold}"] = 0
            row[f"pct_gt_{threshold}"] = np.nan
        return row
    row.update(
        {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
    )
    for threshold in thresholds:
        count = int(np.sum(values > threshold))
        row[f"gt_{threshold}"] = count
        row[f"pct_gt_{threshold}"] = float(count / len(values))
    return row


def summarize_by(frame, group_cols, length_col, thresholds):
    rows = []
    if not group_cols:
        row = length_stats(frame[length_col].to_numpy(), thresholds)
        row["group"] = "all"
        rows.append(row)
        return pd.DataFrame(rows)
    for key, sub in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(length_stats(sub[length_col].to_numpy(), thresholds))
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(frame, length_col, block_sizes, label):
    values = frame[length_col].to_numpy(dtype=np.float64)
    rows = []
    for block_size in block_sizes:
        covered = values <= block_size
        truncated = values > block_size
        rows.append(
            {
                "scope": label,
                "block_size": int(block_size),
                "patients": int(len(values)),
                "covered_patients": int(covered.sum()),
                "covered_pct": float(covered.mean()) if len(values) else np.nan,
                "truncated_patients": int(truncated.sum()),
                "truncated_pct": float(truncated.mean()) if len(values) else np.nan,
                "truncated_tokens": int(np.maximum(values - block_size, 0).sum()),
                "mean_tokens_used": float(np.minimum(values, block_size).mean()) if len(values) else np.nan,
                "total_tokens_used_pct": (
                    float(np.minimum(values, block_size).sum() / values.sum())
                    if values.sum() > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def load_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    frame["patient_id_dense"] = frame["patient_id_dense"].astype(np.int64)
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["split"] = frame["split"].astype(str)
    return frame


def load_split_indices(data_dir: Path, patient_map: pd.DataFrame):
    split_frames = []
    split_data = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        data, has_types = load_data(path)
        if not has_types:
            raise ValueError(f"{path} is not a 4-column typed FERMAT bin")
        p2i = get_p2i(data)
        patient_ids = data[p2i[:, 0].astype(np.int64), 0].astype(np.int64)
        frame = pd.DataFrame(
            {
                "patient_id_dense": patient_ids,
                "split": split,
                "full_sequence_length": p2i[:, 1].astype(np.int64),
            }
        )
        split_frames.append(frame)
        split_data[split] = {
            "data": data,
            "index": {
                int(patient_id): (int(start), int(length))
                for patient_id, (start, length) in zip(patient_ids, p2i)
            },
        }
        log(f"loaded {split}: rows={len(data):,} patients={len(frame):,}")
    lengths = pd.concat(split_frames, ignore_index=True)
    lengths = lengths.merge(patient_map, on=["patient_id_dense", "split"], how="left")
    return lengths, split_data


def load_labels(args):
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{safe_date(args.index_date)}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    columns = parquet_columns(label_path)
    eligible_cols = [
        column for column in columns
        if column.startswith(f"eligible_{args.horizon}__")
    ]
    if not eligible_cols:
        raise ValueError(f"No eligible columns found for horizon={args.horizon}")
    labels = pd.read_parquet(
        label_path,
        columns=["person_id", "split", "age_at_index"] + eligible_cols,
    )
    labels["split"] = labels["split"].astype(str)
    labels["index_age_days"] = np.floor(labels["age_at_index"] * 365.25).astype(np.int64)
    return labels, label_path, eligible_cols


def preindex_lengths(targets: pd.DataFrame, split_data: dict):
    lengths = np.zeros(len(targets), dtype=np.int32)
    missing = np.zeros(len(targets), dtype=bool)
    for index, row in enumerate(targets.itertuples(index=False)):
        location = split_data[str(row.split)]["index"].get(int(row.patient_id_dense))
        if location is None:
            missing[index] = True
            continue
        start, length = location
        rows = split_data[str(row.split)]["data"][start : start + length]
        lengths[index] = int(np.sum(rows[:, 1].astype(np.int64) < int(row.index_age_days)))
    return lengths, missing


def print_table(title, frame, columns=None, max_rows=80):
    print(f"\n## {title}")
    if frame.empty:
        print("(no rows)")
        return
    if columns is not None:
        frame = frame[columns]
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
    print(frame.to_string(index=False))


def main():
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    started = time.time()

    patient_map = load_patient_map(args.data_dir)
    full_lengths, split_data = load_split_indices(args.data_dir, patient_map)
    labels, label_path, eligible_cols = load_labels(args)
    targets = labels.merge(patient_map, on=["person_id", "split"], how="left")
    if targets["patient_id_dense"].isna().any():
        missing = int(targets["patient_id_dense"].isna().sum())
        raise RuntimeError(f"{missing:,} label rows are missing patient_id_dense")
    targets["patient_id_dense"] = targets["patient_id_dense"].astype(np.int64)
    eligible_any = targets[eligible_cols].astype(bool).any(axis=1)
    targets_any = targets.loc[eligible_any].copy()
    targets_any["preindex_sequence_length"], missing_sequence = preindex_lengths(targets_any, split_data)
    targets_any["missing_sequence"] = missing_sequence

    phenotype_rows = []
    for eligible_col in eligible_cols:
        phenotype = eligible_col.split("__", 1)[1]
        sub = targets.loc[targets[eligible_col].astype(bool)].copy()
        sub["phenotype"] = phenotype
        sub["preindex_sequence_length"], missing = preindex_lengths(sub, split_data)
        sub["missing_sequence"] = missing
        phenotype_rows.append(sub[["person_id", "split", "phenotype", "preindex_sequence_length", "missing_sequence"]])
    phenotype_lengths = pd.concat(phenotype_rows, ignore_index=True)

    full_summary = summarize_by(full_lengths, ["split"], "full_sequence_length", args.thresholds)
    full_summary_all = summarize_by(full_lengths, [], "full_sequence_length", args.thresholds)
    preindex_summary = summarize_by(targets_any, ["split"], "preindex_sequence_length", args.thresholds)
    preindex_summary_all = summarize_by(targets_any, [], "preindex_sequence_length", args.thresholds)
    phenotype_summary = summarize_by(
        phenotype_lengths,
        ["phenotype"],
        "preindex_sequence_length",
        args.thresholds,
    )
    block_coverage = pd.concat(
        [
            coverage_table(full_lengths, "full_sequence_length", args.block_sizes, "full_etl"),
            coverage_table(targets_any, "preindex_sequence_length", args.block_sizes, "task19_any_eligible_preindex"),
        ],
        ignore_index=True,
    )

    embedding_path = resolve_embedding_file(args)
    embedding_summary = pd.DataFrame()
    if embedding_path is not None:
        emb_cols = ["person_id", "split", "sequence_length_pre_index", "has_embedding_sequence"]
        emb = pd.read_parquet(embedding_path, columns=emb_cols)
        emb["split"] = emb["split"].astype(str)
        emb = emb.merge(
            targets_any[["person_id", "split", "preindex_sequence_length"]],
            on=["person_id", "split"],
            how="inner",
        )
        emb["embedding_length_shortfall"] = (
            emb["preindex_sequence_length"] - emb["sequence_length_pre_index"]
        ).clip(lower=0)
        embedding_summary = pd.DataFrame(
            [
                {
                    "embedding_path": str(embedding_path),
                    "patients": int(len(emb)),
                    "has_embedding_sequence": int(emb["has_embedding_sequence"].astype(bool).sum()),
                    "max_saved_sequence_length": int(emb["sequence_length_pre_index"].max()),
                    "patients_capped_at_512": int((emb["sequence_length_pre_index"] >= 512).sum()),
                    "patients_with_preindex_gt_saved": int((emb["embedding_length_shortfall"] > 0).sum()),
                    "total_shortfall_tokens": int(emb["embedding_length_shortfall"].sum()),
                }
            ]
        )

    full_lengths.to_parquet(args.output_dir / "full_sequence_lengths.parquet", index=False)
    targets_any.to_parquet(args.output_dir / "task19_any_eligible_preindex_lengths.parquet", index=False)
    phenotype_lengths.to_parquet(args.output_dir / "task19_phenotype_preindex_lengths.parquet", index=False)
    full_summary_all.to_csv(args.output_dir / "full_sequence_summary_all.csv", index=False)
    full_summary.to_csv(args.output_dir / "full_sequence_summary_by_split.csv", index=False)
    preindex_summary_all.to_csv(args.output_dir / "task19_preindex_summary_all.csv", index=False)
    preindex_summary.to_csv(args.output_dir / "task19_preindex_summary_by_split.csv", index=False)
    phenotype_summary.to_csv(args.output_dir / "task19_preindex_summary_by_phenotype.csv", index=False)
    block_coverage.to_csv(args.output_dir / "block_size_coverage.csv", index=False)
    if not embedding_summary.empty:
        embedding_summary.to_csv(args.output_dir / "existing_embedding_cap_check.csv", index=False)

    print_table("FULL_SEQUENCE_LENGTH_ALL", full_summary_all)
    print_table("FULL_SEQUENCE_LENGTH_BY_SPLIT", full_summary)
    print_table("TASK19_ANY_ELIGIBLE_PREINDEX_LENGTH_ALL", preindex_summary_all)
    print_table("TASK19_ANY_ELIGIBLE_PREINDEX_LENGTH_BY_SPLIT", preindex_summary)
    print_table("TASK19_PREINDEX_LENGTH_BY_PHENOTYPE", phenotype_summary)
    print_table("BLOCK_SIZE_COVERAGE", block_coverage)
    if not embedding_summary.empty:
        print_table("EXISTING_EMBEDDING_CAP_CHECK", embedding_summary)

    manifest = {
        "data_dir": str(args.data_dir),
        "label_path": str(label_path),
        "embedding_path": str(embedding_path) if embedding_path else None,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "thresholds": args.thresholds,
        "block_sizes": args.block_sizes,
        "eligible_columns": eligible_cols,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "full_sequence_lengths": str(args.output_dir / "full_sequence_lengths.parquet"),
            "task19_any_eligible_preindex_lengths": str(args.output_dir / "task19_any_eligible_preindex_lengths.parquet"),
            "task19_phenotype_preindex_lengths": str(args.output_dir / "task19_phenotype_preindex_lengths.parquet"),
            "block_size_coverage": str(args.output_dir / "block_size_coverage.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n## OUTPUT_DIR")
    print(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
