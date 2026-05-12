"""
Validate a FERMAT 4-column binary file.

Checks invariants from docs/token_spec_v0.1.md:
  - uint32 layout, divisible into 4-tuples
  - patient_id is dense (0..N-1) and contiguous (rows for same pid are adjacent)
  - age_in_days is non-negative and <= 150 years in days
  - token_id matches an entry in vocab.csv
  - token_type is in 0..8
  - train/val patient sets do not overlap

Usage:
    python scripts/check_fermat_bin.py \
        --data_dir data/synthetic_snuh \
        --out logs/fermat_bin_check.log
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_4col_bin(path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.shape[0] % 4 != 0:
        raise ValueError(
            f"{path}: byte length not divisible by 4 uint32 (rows*4)"
        )
    return np.array(raw.reshape(-1, 4))


def check_one(name, arr, vocab_ids, out, max_age_days=150 * 365):
    issues = 0
    out.write(f"\n## {name}\n")
    out.write(f"  rows: {len(arr)}\n")

    if len(arr) == 0:
        out.write("  EMPTY — skipping further checks\n")
        return 1, set()

    pids = arr[:, 0]
    ages = arr[:, 1]
    toks = arr[:, 2]
    types = arr[:, 3]

    # Patient contiguity
    diffs = np.diff(pids.astype(np.int64))
    if (diffs < 0).any():
        out.write(f"  ERROR: patient_id is not non-decreasing\n")
        issues += 1
    else:
        out.write(f"  OK: patient_id non-decreasing\n")

    # Patient density
    unique_pids = np.unique(pids)
    expected = np.arange(unique_pids[0], unique_pids[-1] + 1)
    if len(unique_pids) != len(expected) or (unique_pids != expected).any():
        out.write(
            f"  WARN: patient_id not perfectly dense "
            f"(min={unique_pids[0]}, max={unique_pids[-1]}, "
            f"unique={len(unique_pids)})\n"
        )
    else:
        out.write(
            f"  OK: patient_id dense ({len(unique_pids)} patients)\n"
        )

    # Age range
    if (ages < 0).any():
        out.write(f"  ERROR: negative age_in_days\n")
        issues += 1
    else:
        out.write(f"  OK: age_in_days >= 0 (min={ages.min()}, max={ages.max()})\n")
    if (ages > max_age_days).any():
        out.write(f"  WARN: age_in_days > {max_age_days} for {(ages > max_age_days).sum()} rows\n")

    # Token range
    unknown_toks = ~np.isin(toks, list(vocab_ids))
    if unknown_toks.any():
        out.write(
            f"  ERROR: {int(unknown_toks.sum())} token_ids not in vocab.csv "
            f"(e.g. {toks[unknown_toks][:5].tolist()})\n"
        )
        issues += 1
    else:
        out.write(f"  OK: all token_ids in vocab\n")

    # Token type range
    if (types > 8).any() or (types < 0).any():
        bad = types[(types > 8) | (types < 0)]
        out.write(f"  ERROR: invalid token_type values (e.g. {bad[:5].tolist()})\n")
        issues += 1
    else:
        out.write(
            f"  OK: token_type in 0..8 (counts: "
            f"{dict(zip(*np.unique(types, return_counts=True)))})\n"
        )

    # Age sortedness within patient
    bad_pids = 0
    for pid in unique_pids:
        sel = pids == pid
        if not np.all(np.diff(ages[sel].astype(np.int64)) >= 0):
            bad_pids += 1
    if bad_pids > 0:
        out.write(f"  WARN: {bad_pids} patients have non-monotonic ages\n")
    else:
        out.write(f"  OK: ages monotonic within each patient\n")

    return issues, set(unique_pids.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/synthetic_snuh")
    parser.add_argument("--out", default="logs/fermat_bin_check.log")
    args = parser.parse_args()

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    vocab_csv = os.path.join(args.data_dir, "vocab.csv")

    for p in (train_bin, val_bin, vocab_csv):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)

    vocab = pd.read_csv(vocab_csv)
    vocab_ids = set(int(t) for t in vocab["token_id"].tolist())

    train_arr = load_4col_bin(train_bin)
    val_arr = load_4col_bin(val_bin)

    with open(args.out, "w") as out:
        out.write("# FERMAT bin check\n")
        out.write(f"data_dir: {args.data_dir}\n")
        out.write(f"vocab size: {len(vocab_ids)}\n")

        train_issues, train_pids = check_one("train.bin", train_arr, vocab_ids, out)
        val_issues, val_pids = check_one("val.bin", val_arr, vocab_ids, out)

        out.write("\n## train/val patient overlap\n")
        overlap = train_pids & val_pids
        if overlap:
            out.write(
                f"  WARN: {len(overlap)} patient_ids overlap (they are "
                f"redensed independently per split; this is expected if "
                f"both splits start at 0 — but person_id-level overlap "
                f"should be 0 in patient_split.csv).\n"
            )
        else:
            out.write("  OK: no patient_id overlap\n")

        # Cross-check person-level overlap from patient_split.csv
        ps_path = os.path.join(args.data_dir, "patient_split.csv")
        if os.path.exists(ps_path):
            ps = pd.read_csv(ps_path)
            id_col = "person_id" if "person_id" in ps.columns else (
                "patient_id" if "patient_id" in ps.columns else None
            )
            if id_col and "split" in ps.columns:
                tr_p = set(ps.loc[ps["split"] == "train", id_col])
                va_p = set(ps.loc[ps["split"] == "val", id_col])
                p_overlap = tr_p & va_p
                if p_overlap:
                    out.write(
                        f"  ERROR: {len(p_overlap)} {id_col}s appear in "
                        f"both train and val\n"
                    )
                else:
                    out.write(
                        f"  OK: {id_col}-level train/val are disjoint "
                        f"(train={len(tr_p)}, val={len(va_p)})\n"
                    )

        total = train_issues + val_issues
        out.write(f"\n## Summary\n")
        out.write(f"  total ERROR-level issues: {total}\n")

    print(f"Bin check written to {args.out}")
    sys.exit(0 if total == 0 else 2)


if __name__ == "__main__":
    main()
