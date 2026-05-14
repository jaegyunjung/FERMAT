"""
Audit semantic mapping from Synthetic SNUH OMOP tables into FERMAT 4-column data.

Checks:
  1. Source row counts for person/condition/drug/procedure/measurement/death
  2. FERMAT token_type counts from train.bin and val.bin
  3. Expected source->token_type mapping
  4. Zero person_id overlap between train and val in patient_split.csv
  5. dropped_events.csv explanation for source-vs-mapped differences
  6. ERROR if measurement rows > 0 but LAB count == 0
  7. ERROR if death rows > 0 but DTH count == 0

Writes both markdown and text reports, and exits nonzero if any ERROR is found.
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TT_NAME = {
    0: "PAD",
    1: "DX",
    2: "RX",
    3: "PX",
    4: "LAB",
    5: "LIFESTYLE",
    6: "DTH",
    7: "SEX",
    8: "NO_EVENT",
}

SOURCE_TO_TOKEN = {
    "person": "SEX",
    "condition_occurrence": "DX",
    "drug_exposure": "RX",
    "procedure_occurrence": "PX",
    "measurement": "LAB",
    "death": "DTH",
}


def load_4col(path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.shape[0] % 4 != 0:
        raise ValueError(f"{path}: not divisible into 4-column uint32 rows")
    return np.array(raw.reshape(-1, 4))


def count_rows(con, table):
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def token_counts(arr):
    if len(arr) == 0:
        return {}
    vals, counts = np.unique(arr[:, 3], return_counts=True)
    return {TT_NAME.get(int(v), str(v)): int(c) for v, c in zip(vals, counts)}


def format_dropped_summary(dropped_df, table):
    if dropped_df.empty or "table" not in dropped_df.columns:
        return "none"
    sub = dropped_df.loc[dropped_df["table"] == table].copy()
    if sub.empty:
        return "none"
    parts = []
    for _, row in sub.sort_values(["reason"]).iterrows():
        parts.append(f"{row['reason']}={int(row['count'])}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default="data/synthetic_snuh_raw.duckdb")
    parser.add_argument("--data-dir", default="data/synthetic_snuh")
    parser.add_argument("--out-md", default="logs/fermat_mapping_audit.md")
    parser.add_argument("--out-txt", default="logs/fermat_mapping_audit.txt")
    args = parser.parse_args()

    for path in [args.duckdb,
                 os.path.join(args.data_dir, "train.bin"),
                 os.path.join(args.data_dir, "val.bin"),
                 os.path.join(args.data_dir, "patient_split.csv"),
                 os.path.join(args.data_dir, "dropped_events.csv")]:
        if not os.path.exists(path):
            print(f"ERROR: missing required input: {path}", file=sys.stderr)
            sys.exit(1)

    Path(os.path.dirname(args.out_md) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_txt) or ".").mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.duckdb, read_only=True)
    source_counts = {
        "person": count_rows(con, "person"),
        "condition_occurrence": count_rows(con, "condition_occurrence"),
        "drug_exposure": count_rows(con, "drug_exposure"),
        "procedure_occurrence": count_rows(con, "procedure_occurrence"),
        "measurement": count_rows(con, "measurement"),
        "death": count_rows(con, "death"),
    }
    con.close()

    train = load_4col(os.path.join(args.data_dir, "train.bin"))
    val = load_4col(os.path.join(args.data_dir, "val.bin"))
    all_arr = np.vstack([train, val]) if len(val) else train.copy()
    all_token_counts = token_counts(all_arr)
    train_token_counts = token_counts(train)
    val_token_counts = token_counts(val)

    patient_split = pd.read_csv(os.path.join(args.data_dir, "patient_split.csv"))
    dropped = pd.read_csv(os.path.join(args.data_dir, "dropped_events.csv"))

    train_persons = set(
        patient_split.loc[patient_split["split"] == "train", "person_id"].astype(int).tolist()
    )
    val_persons = set(
        patient_split.loc[patient_split["split"] == "val", "person_id"].astype(int).tolist()
    )
    overlap = train_persons & val_persons

    issues = []
    findings = []

    if overlap:
        issues.append(
            f"ERROR: patient_split.csv has {len(overlap)} overlapping person_id values between train and val"
        )
    else:
        findings.append(
            f"OK: patient_split.csv train/val person_id overlap is 0 (train={len(train_persons)}, val={len(val_persons)})"
        )

    if source_counts["measurement"] > 0 and all_token_counts.get("LAB", 0) == 0:
        issues.append(
            f"ERROR: measurement rows={source_counts['measurement']} but LAB token count is 0"
        )
    else:
        findings.append(
            f"OK: measurement rows={source_counts['measurement']}, LAB tokens={all_token_counts.get('LAB', 0)}"
        )

    if source_counts["death"] > 0 and all_token_counts.get("DTH", 0) == 0:
        issues.append(
            f"ERROR: death rows={source_counts['death']} but DTH token count is 0"
        )
    else:
        findings.append(
            f"OK: death rows={source_counts['death']}, DTH tokens={all_token_counts.get('DTH', 0)}"
        )

    mapping_rows = []
    for source_table, token_name in SOURCE_TO_TOKEN.items():
        mapping_rows.append({
            "source_table": source_table,
            "source_rows": source_counts[source_table],
            "expected_token_type": token_name,
            "train_tokens": train_token_counts.get(token_name, 0),
            "val_tokens": val_token_counts.get(token_name, 0),
            "total_tokens": all_token_counts.get(token_name, 0),
            "dropped_explanation": format_dropped_summary(dropped, source_table),
        })

    # Text report
    with open(args.out_txt, "w") as f:
        f.write("# FERMAT mapping audit\n")
        f.write(f"duckdb: {args.duckdb}\n")
        f.write(f"data_dir: {args.data_dir}\n\n")

        f.write("## Source row counts\n")
        for table in ["person", "condition_occurrence", "drug_exposure",
                      "procedure_occurrence", "measurement", "death"]:
            f.write(f"  {table:24s} {source_counts[table]}\n")
        f.write("\n")

        f.write("## FERMAT token_type counts\n")
        for token_name in ["SEX", "DX", "RX", "PX", "LAB", "DTH"]:
            f.write(
                f"  {token_name:10s} train={train_token_counts.get(token_name, 0)} "
                f"val={val_token_counts.get(token_name, 0)} "
                f"total={all_token_counts.get(token_name, 0)}\n"
            )
        f.write("\n")

        f.write("## Expected mapping audit\n")
        for row in mapping_rows:
            f.write(
                f"  {row['source_table']:24s} -> {row['expected_token_type']:4s} "
                f"source_rows={row['source_rows']} "
                f"mapped_total={row['total_tokens']} "
                f"dropped=({row['dropped_explanation']})\n"
            )
        f.write("\n")

        f.write("## Split integrity\n")
        if overlap:
            f.write(
                f"  ERROR: train/val person_id overlap = {len(overlap)}\n"
            )
        else:
            f.write(
                f"  OK: train/val person_id overlap = 0 "
                f"(train={len(train_persons)}, val={len(val_persons)})\n"
            )
        f.write("\n")

        f.write("## Findings\n")
        for line in findings:
            f.write(f"  {line}\n")
        for line in issues:
            f.write(f"  {line}\n")
        f.write("\n")

        f.write("## Summary\n")
        f.write(f"  ERROR-level issues: {len(issues)}\n")

    # Markdown report
    with open(args.out_md, "w") as f:
        f.write("# FERMAT mapping audit\n\n")
        f.write(f"_DuckDB:_ `{args.duckdb}`  \n")
        f.write(f"_Data dir:_ `{args.data_dir}`\n\n")

        f.write("## Source Row Counts\n\n")
        f.write("| source_table | rows |\n|---|---:|\n")
        for table in ["person", "condition_occurrence", "drug_exposure",
                      "procedure_occurrence", "measurement", "death"]:
            f.write(f"| {table} | {source_counts[table]} |\n")
        f.write("\n")

        f.write("## FERMAT Token Counts\n\n")
        f.write("| token_type | train | val | total |\n|---|---:|---:|---:|\n")
        for token_name in ["SEX", "DX", "RX", "PX", "LAB", "DTH"]:
            f.write(
                f"| {token_name} | {train_token_counts.get(token_name, 0)} | "
                f"{val_token_counts.get(token_name, 0)} | "
                f"{all_token_counts.get(token_name, 0)} |\n"
            )
        f.write("\n")

        f.write("## Expected Mapping\n\n")
        f.write("| source_table | expected_token_type | source_rows | mapped_total | dropped_explanation |\n")
        f.write("|---|---|---:|---:|---|\n")
        for row in mapping_rows:
            f.write(
                f"| {row['source_table']} | {row['expected_token_type']} | "
                f"{row['source_rows']} | {row['total_tokens']} | "
                f"{row['dropped_explanation']} |\n"
            )
        f.write("\n")

        f.write("## Split Integrity\n\n")
        if overlap:
            f.write(
                f"- ERROR: `patient_split.csv` has {len(overlap)} overlapping original `person_id` values.\n"
            )
        else:
            f.write(
                f"- OK: original `person_id` overlap between train and val is 0 "
                f"(train={len(train_persons)}, val={len(val_persons)}).\n"
            )
        f.write("\n")

        f.write("## Findings\n\n")
        for line in findings:
            f.write(f"- {line}\n")
        for line in issues:
            f.write(f"- {line}\n")
        if not findings and not issues:
            f.write("- No findings.\n")
        f.write("\n")

        f.write("## Result\n\n")
        if issues:
            f.write(f"ERROR: {len(issues)} ERROR-level issue(s) found.\n")
        else:
            f.write("PASS: no ERROR-level issues found.\n")

    print(f"Mapping audit written to {args.out_txt} and {args.out_md}")
    sys.exit(0 if not issues else 2)


if __name__ == "__main__":
    main()
