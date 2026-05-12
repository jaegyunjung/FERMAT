"""
Generate dataset summary statistics for a FERMAT 4-column dataset.

Outputs:
  - logs/fermat_dataset_summary.txt   (human-readable)
  - logs/fermat_dataset_summary.md    (markdown table, for PI brief)

Usage:
    python scripts/summarize_fermat_dataset.py \
        --data_dir data/synthetic_snuh \
        --out_txt logs/fermat_dataset_summary.txt \
        --out_md  logs/fermat_dataset_summary.md
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


TT_NAME = {0: "PAD", 1: "DX", 2: "RX", 3: "PX", 4: "LAB",
           5: "LIFESTYLE", 6: "DTH", 7: "SEX", 8: "NO_EVENT"}


def load_4col(path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    return np.array(raw.reshape(-1, 4))


def sequence_lengths(arr):
    """Lengths of consecutive runs of the same patient_id."""
    if len(arr) == 0:
        return np.array([], dtype=int)
    pids = arr[:, 0]
    change = np.where(np.diff(pids) != 0)[0] + 1
    boundaries = np.concatenate([[0], change, [len(pids)]])
    return np.diff(boundaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/synthetic_snuh")
    parser.add_argument("--out_txt", default="logs/fermat_dataset_summary.txt")
    parser.add_argument("--out_md",  default="logs/fermat_dataset_summary.md")
    args = parser.parse_args()

    Path(os.path.dirname(args.out_txt) or ".").mkdir(parents=True, exist_ok=True)

    train = load_4col(os.path.join(args.data_dir, "train.bin"))
    val = load_4col(os.path.join(args.data_dir, "val.bin"))
    vocab = pd.read_csv(os.path.join(args.data_dir, "vocab.csv"))

    dropped_path = os.path.join(args.data_dir, "dropped_events.csv")
    if os.path.exists(dropped_path):
        dropped = pd.read_csv(dropped_path)
    else:
        dropped = pd.DataFrame(columns=["reason", "table", "count"])

    ps_path = os.path.join(args.data_dir, "patient_split.csv")
    ps = pd.read_csv(ps_path) if os.path.exists(ps_path) else None

    # Per-type counts
    def by_type(arr):
        types = arr[:, 3]
        return {TT_NAME.get(int(t), str(t)): int((types == t).sum())
                for t in sorted(np.unique(types).tolist())}

    train_types = by_type(train)
    val_types = by_type(val)

    train_lens = sequence_lengths(train)
    val_lens = sequence_lengths(val)

    train_ages = train[:, 1] if len(train) else np.array([0], dtype=int)
    val_ages = val[:, 1] if len(val) else np.array([0], dtype=int)

    # Person-level counts
    if ps is not None and "split" in ps.columns:
        n_train_p = int((ps["split"] == "train").sum())
        n_val_p = int((ps["split"] == "val").sum())
        id_col = "person_id" if "person_id" in ps.columns else "patient_id"
        overlap = (
            set(ps.loc[ps["split"] == "train", id_col])
            & set(ps.loc[ps["split"] == "val", id_col])
        )
    else:
        n_train_p = len(np.unique(train[:, 0])) if len(train) else 0
        n_val_p = len(np.unique(val[:, 0])) if len(val) else 0
        overlap = set()

    # ---- Plain text ----
    with open(args.out_txt, "w") as f:
        f.write("# FERMAT dataset summary\n")
        f.write(f"data_dir: {args.data_dir}\n\n")

        f.write("## Sizes\n")
        f.write(f"  total patients (train + val): {n_train_p + n_val_p}\n")
        f.write(f"    train: {n_train_p}\n")
        f.write(f"    val:   {n_val_p}\n")
        f.write(f"  total events:\n")
        f.write(f"    train: {len(train)}\n")
        f.write(f"    val:   {len(val)}\n")
        f.write(f"  vocab size: {len(vocab)}\n")
        f.write(f"  max token_id: {int(vocab['token_id'].max())}\n\n")

        f.write("## Events by token_type (train)\n")
        for k, v in train_types.items():
            f.write(f"  {k:10s} {v}\n")
        f.write("## Events by token_type (val)\n")
        for k, v in val_types.items():
            f.write(f"  {k:10s} {v}\n")
        f.write("\n")

        f.write("## Sequence length per patient\n")
        if len(train_lens):
            f.write(
                f"  train: median={int(np.median(train_lens))}, "
                f"p95={int(np.percentile(train_lens, 95))}, "
                f"max={int(train_lens.max())}\n"
            )
        if len(val_lens):
            f.write(
                f"  val:   median={int(np.median(val_lens))}, "
                f"p95={int(np.percentile(val_lens, 95))}, "
                f"max={int(val_lens.max())}\n"
            )
        f.write("\n")

        f.write("## age_in_days range\n")
        f.write(f"  train: min={int(train_ages.min())}, max={int(train_ages.max())}\n")
        f.write(f"  val:   min={int(val_ages.min())}, max={int(val_ages.max())}\n\n")

        f.write("## Dropped events (from preprocess)\n")
        if len(dropped):
            for _, r in dropped.iterrows():
                f.write(
                    f"  {r['reason']:32s} {r.get('table', ''):24s} "
                    f"count={int(r['count'])}\n"
                )
        else:
            f.write("  (none; self-synthetic path has no source data to drop from)\n")
        f.write("\n")

        f.write("## Train/val patient overlap\n")
        f.write(f"  person-level overlap: {len(overlap)}\n")

    # ---- Markdown for PI brief ----
    with open(args.out_md, "w") as f:
        f.write("# FERMAT dataset summary\n\n")
        f.write(f"_Source:_ `{args.data_dir}`\n\n")
        f.write("| metric | train | val |\n|---|---:|---:|\n")
        f.write(f"| patients | {n_train_p} | {n_val_p} |\n")
        f.write(f"| events   | {len(train)} | {len(val)} |\n")
        if len(train_lens):
            f.write(
                f"| seq len (median / p95 / max) | "
                f"{int(np.median(train_lens))} / "
                f"{int(np.percentile(train_lens, 95))} / "
                f"{int(train_lens.max())} | "
            )
        else:
            f.write(f"| seq len (median / p95 / max) | — | ")
        if len(val_lens):
            f.write(
                f"{int(np.median(val_lens))} / "
                f"{int(np.percentile(val_lens, 95))} / "
                f"{int(val_lens.max())} |\n"
            )
        else:
            f.write(f"— |\n")
        f.write(f"| age_in_days range | "
                f"{int(train_ages.min())}–{int(train_ages.max())} | "
                f"{int(val_ages.min())}–{int(val_ages.max())} |\n")
        f.write(f"| vocab size | {len(vocab)} | — |\n")
        f.write(f"| person-level train/val overlap | {len(overlap)} | — |\n\n")

        f.write("## Events by token_type\n\n")
        type_keys = sorted(set(train_types) | set(val_types))
        f.write("| token_type | train | val |\n|---|---:|---:|\n")
        for k in type_keys:
            f.write(
                f"| {k} | {train_types.get(k, 0)} | {val_types.get(k, 0)} |\n"
            )
        f.write("\n")

        if len(dropped):
            f.write("## Dropped events during preprocessing\n\n")
            f.write("| reason | table | count |\n|---|---|---:|\n")
            for _, r in dropped.iterrows():
                f.write(
                    f"| {r['reason']} | {r.get('table', '')} | "
                    f"{int(r['count'])} |\n"
                )

    print(f"Summary written to {args.out_txt} and {args.out_md}")


if __name__ == "__main__":
    main()
