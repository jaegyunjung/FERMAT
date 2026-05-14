"""
Convert Synthetic SNUH OMOP CDM (DuckDB) into FERMAT 4-column binary.

Implements docs/token_spec_v0.1.md.

Usage:
    python scripts/preprocess_synthetic_snuh_to_fermat.py \
        --duckdb data/synthetic_snuh_raw.duckdb \
        --out_dir data/synthetic_snuh \
        --vocab_cap 2000 \
        --val_frac 0.1 \
        --seed 42

Outputs:
    data/synthetic_snuh/train.bin
    data/synthetic_snuh/val.bin
    data/synthetic_snuh/vocab.csv
    data/synthetic_snuh/patient_split.csv
    data/synthetic_snuh/dropped_events.csv
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TT = {
    "PAD": 0, "DX": 1, "RX": 2, "PX": 3, "LAB": 4,
    "LIFESTYLE": 5, "DTH": 6, "SEX": 7, "NO_EVENT": 8,
}


# ---------------------------------------------------------------------------
# OMOP query helpers
# ---------------------------------------------------------------------------

def query_persons(con):
    return con.execute("""
        SELECT
            person_id,
            gender_concept_id,
            gender_source_value,
            year_of_birth,
            month_of_birth,
            day_of_birth,
            birth_datetime
        FROM person
    """).df()


def get_table_columns(con, table):
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {r[1] for r in rows}


def query_events_table(con, table, src_col, fallback_col, date_col):
    """Generic event extractor; returns DataFrame with person_id, event_date, source_value."""
    fallback_sql = f"CAST({fallback_col} AS VARCHAR)" if fallback_col else "NULL"
    return con.execute(f"""
        SELECT
            person_id,
            CAST({date_col} AS DATE) AS event_date,
            COALESCE(NULLIF({src_col}, ''), {fallback_sql}) AS source_value
        FROM {table}
        WHERE {date_col} IS NOT NULL
          AND person_id IS NOT NULL
    """).df()


def query_measurements(con):
    cols = get_table_columns(con, "measurement")
    value_as_concept_sql = (
        "value_as_concept_id"
        if "value_as_concept_id" in cols else
        "NULL::BIGINT AS value_as_concept_id"
    )
    unit_concept_sql = (
        "unit_concept_id"
        if "unit_concept_id" in cols else
        "NULL::BIGINT AS unit_concept_id"
    )
    unit_source_sql = (
        "unit_source_value"
        if "unit_source_value" in cols else
        "NULL::VARCHAR AS unit_source_value"
    )
    return con.execute(f"""
        SELECT
            person_id,
            CAST(measurement_date AS DATE) AS event_date,
            COALESCE(NULLIF(measurement_source_value, ''),
                     CAST(measurement_concept_id AS VARCHAR)) AS source_value,
            value_as_number,
            {value_as_concept_sql},
            {unit_concept_sql},
            {unit_source_sql}
        FROM measurement
        WHERE measurement_date IS NOT NULL
          AND person_id IS NOT NULL
    """).df()


def query_death(con):
    cols = get_table_columns(con, "death")
    source_expr = (
        "NULLIF(cause_source_value, '')"
        if "cause_source_value" in cols else
        "NULL"
    )
    return con.execute(f"""
        SELECT
            person_id,
            CAST(death_date AS DATE) AS event_date,
            COALESCE({source_expr}, CAST(cause_concept_id AS VARCHAR), 'UNK') AS source_value
        FROM death
        WHERE death_date IS NOT NULL
          AND person_id IS NOT NULL
    """).df()


# ---------------------------------------------------------------------------
# age_in_days per token_spec section 4
# ---------------------------------------------------------------------------

def compute_birth_date(row):
    if pd.notna(row.get("birth_datetime")):
        return pd.to_datetime(row["birth_datetime"]).date()
    if pd.notna(row.get("year_of_birth")):
        m = int(row["month_of_birth"]) if pd.notna(row.get("month_of_birth")) else 7
        d = int(row["day_of_birth"]) if pd.notna(row.get("day_of_birth")) else 1
        try:
            return date(int(row["year_of_birth"]), m, d)
        except ValueError:
            return None
    return None


def build_birth_map(persons):
    out = {}
    drops = 0
    for _, r in persons.iterrows():
        bd = compute_birth_date(r)
        if bd is None:
            drops += 1
            continue
        out[int(r["person_id"])] = bd
    return out, drops


def add_age_in_days(df, birth_map, drop_log, table_name):
    """Attach age_in_days; drop rows with missing/invalid age."""
    bd_series = df["person_id"].map(birth_map)
    has_birth = bd_series.notna()
    drop_log.append({
        "reason": "birth_unknown_patient",
        "table": table_name,
        "source_value": "",
        "count": int((~has_birth).sum()),
    })
    df = df.loc[has_birth].copy()
    bd_series = bd_series.loc[has_birth]
    age = (pd.to_datetime(df["event_date"]) -
           pd.to_datetime(bd_series)).dt.days

    neg = age < 0
    far = age > 150 * 365
    drop_log.append({
        "reason": "age_negative", "table": table_name,
        "source_value": "", "count": int(neg.sum()),
    })
    drop_log.append({
        "reason": "age_unrealistic", "table": table_name,
        "source_value": "", "count": int(far.sum()),
    })

    keep = ~(neg | far)
    df = df.loc[keep].copy()
    df["age_in_days"] = age.loc[keep].astype(int)
    return df


# ---------------------------------------------------------------------------
# Measurement → quantile binning (train-split only)
# ---------------------------------------------------------------------------

def measurement_to_lab_labels(meas_df, train_person_set, drop_log):
    """
    Turn measurement rows into LAB:<key>:<bin> labels.
    Quantile cutpoints are computed on training persons only.
    """
    has_number = meas_df["value_as_number"].notna()
    drop_log.append({
        "reason": "measurement_value_missing", "table": "measurement",
        "source_value": "", "count": int((~has_number).sum()),
    })
    meas_df = meas_df.loc[has_number].copy()

    labels = pd.Series(index=meas_df.index, dtype=object)

    # Numeric path: compute quantiles on training persons
    num_mask = has_number.loc[meas_df.index]
    num_df = meas_df.loc[num_mask]
    if len(num_df):
        train_mask_num = num_df["person_id"].isin(train_person_set)
        train_num = num_df.loc[train_mask_num]
        # Compute tertile cutpoints per measurement key on train only
        cutpoints = {}
        for key, grp in train_num.groupby("source_value"):
            vals = grp["value_as_number"].dropna().values
            if len(vals) == 0:
                continue
            q1 = np.quantile(vals, 1 / 3.0)
            q2 = np.quantile(vals, 2 / 3.0)
            cutpoints[key] = (q1, q2)

        # Apply
        def to_bin(row):
            key = row["source_value"]
            if key not in cutpoints:
                return None
            q1, q2 = cutpoints[key]
            v = row["value_as_number"]
            if v <= q1:
                return f"LAB:{key}:Q1"
            elif v <= q2:
                return f"LAB:{key}:Q2"
            else:
                return f"LAB:{key}:Q3"

        applied = num_df.apply(to_bin, axis=1)
        below = applied.isna()
        drop_log.append({
            "reason": "measurement_cutpoint_unavailable", "table": "measurement",
            "source_value": "", "count": int(below.sum()),
        })
        labels.loc[num_df.index] = applied

    meas_df["label"] = labels
    meas_df = meas_df.loc[meas_df["label"].notna()].copy()
    return meas_df[["person_id", "age_in_days", "label"]], len(meas_df)


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(events_by_type, vocab_cap):
    """
    events_by_type: dict[token_type_name] -> DataFrame with column 'label'
    Returns vocab DataFrame and a dict[(token_type_name, label)] -> token_id.
    """
    # PAD=0, NO_EVENT reserved as id=1 (Delphi convention via get_batch +1 shift).
    rows = []
    next_id = 0

    def add(label, ttn, source_value=None, quantile_bin=None, freq=0,
            extra=None):
        nonlocal next_id
        rows.append({
            "token_id": next_id,
            "token_type": TT[ttn],
            "token_type_name": ttn,
            "source_table": (extra or {}).get("table", ""),
            "source_column": (extra or {}).get("column", ""),
            "source_value": source_value or "",
            "concept_id": 0,
            "label": label,
            "unit_concept_id": 0,
            "unit_source_value": "",
            "quantile_bin": quantile_bin or "",
            "frequency": freq,
        })
        tid = next_id
        next_id += 1
        return tid

    add("PAD", "PAD", "<PAD>")
    add("NO_EVENT", "NO_EVENT", "<NO_EVENT>")

    # Compute frequencies and pick top-N
    freq_records = []
    for ttn, df in events_by_type.items():
        if ttn == "SEX":
            # always-include SEX tokens (small set)
            continue
        if df is None or len(df) == 0:
            continue
        vc = df["label"].value_counts()
        for label, count in vc.items():
            freq_records.append({
                "token_type_name": ttn, "label": label, "frequency": int(count),
            })
    freq_df = pd.DataFrame(freq_records)
    if len(freq_df) > 0:
        freq_df = freq_df.sort_values("frequency", ascending=False).reset_index(drop=True)
        # Keep top vocab_cap minus SEX tokens (counted separately)
        kept = freq_df.head(vocab_cap).copy()
    else:
        kept = pd.DataFrame(columns=["token_type_name", "label", "frequency"])

    # SEX tokens always included
    sex_df = events_by_type.get("SEX")
    if sex_df is not None and len(sex_df):
        for label in sorted(sex_df["label"].unique()):
            add(label, "SEX", source_value=label, freq=int((sex_df["label"] == label).sum()),
                extra={"table": "person", "column": "gender_source_value"})

    # Insert kept tokens grouped by type (DX -> RX -> PX -> LAB -> LIFESTYLE -> DTH)
    for ttn in ["DX", "RX", "PX", "LAB", "LIFESTYLE", "DTH"]:
        sub = kept.loc[kept["token_type_name"] == ttn]
        for _, r in sub.iterrows():
            label = r["label"]
            sv = label.split(":", 1)[1] if ":" in label else label
            qbin = ""
            if ttn == "LAB":
                parts = label.split(":")
                if len(parts) >= 3:
                    qbin = parts[2]
            add(
                label, ttn, source_value=sv, quantile_bin=qbin,
                freq=int(r["frequency"]),
                extra={
                    "table": {
                        "DX": "condition_occurrence",
                        "RX": "drug_exposure",
                        "PX": "procedure_occurrence",
                        "LAB": "measurement",
                        "LIFESTYLE": "observation",
                        "DTH": "death",
                    }[ttn],
                    "column": "source_value",
                },
            )

    vocab = pd.DataFrame(rows)
    label_to_id = {
        (r["token_type_name"], r["label"]): int(r["token_id"])
        for _, r in vocab.iterrows()
    }
    return vocab, label_to_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb",
        default="data/synthetic_snuh_raw.duckdb",
        help="path to DuckDB file (default: data/synthetic_snuh_raw.duckdb)",
    )
    parser.add_argument("--out_dir", default="data/synthetic_snuh")
    parser.add_argument("--vocab_cap", type=int, default=2000)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.duckdb):
        print(f"ERROR: DuckDB file not found: {args.duckdb}", file=sys.stderr)
        sys.exit(1)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    con = duckdb.connect(args.duckdb, read_only=True)
    drop_log = []

    # --- 1. Persons & birth map ---
    persons = query_persons(con)
    birth_map, n_birth_drop = build_birth_map(persons)
    drop_log.append({
        "reason": "birth_unknown_patient", "table": "person",
        "source_value": "", "count": n_birth_drop,
    })

    eligible_persons = sorted(birth_map.keys())
    if not eligible_persons:
        print("ERROR: no persons with valid birth date.", file=sys.stderr)
        sys.exit(1)

    # SEX labels
    sex_rows = []
    for _, r in persons.iterrows():
        pid = int(r["person_id"])
        if pid not in birth_map:
            continue
        sv = r.get("gender_source_value")
        if not sv or pd.isna(sv):
            gid = r.get("gender_concept_id")
            sv = "M" if gid == 8507 else "F" if gid == 8532 else "U"
        sex_rows.append({
            "person_id": pid, "age_in_days": 0,
            "label": f"SEX:{sv}",
        })
    sex_df = pd.DataFrame(sex_rows)

    # --- 2. Train/val split on person_id ---
    persons_arr = np.array(eligible_persons)
    rng.shuffle(persons_arr)
    n_val = int(len(persons_arr) * args.val_frac)
    val_set = set(persons_arr[:n_val].tolist())
    train_set = set(persons_arr[n_val:].tolist())

    # --- 3. Extract events per table ---
    def fetch_and_label(table, src, fallback, date_col, ttn, prefix):
        try:
            df = query_events_table(con, table, src, fallback, date_col)
        except duckdb.Error as e:
            print(f"WARN: could not query {table}: {e}", file=sys.stderr)
            return pd.DataFrame(columns=["person_id", "age_in_days", "label"])
        if len(df) == 0:
            return pd.DataFrame(columns=["person_id", "age_in_days", "label"])
        # filter to eligible persons
        df = df.loc[df["person_id"].isin(birth_map.keys())]
        # unknown source_value
        unk = df["source_value"].isna() | (df["source_value"] == "")
        drop_log.append({
            "reason": "unknown_source_value", "table": table,
            "source_value": "", "count": int(unk.sum()),
        })
        df = df.loc[~unk].copy()
        df = add_age_in_days(df, birth_map, drop_log, table)
        df["label"] = f"{prefix}:" + df["source_value"].astype(str)
        return df[["person_id", "age_in_days", "label"]]

    dx_df = fetch_and_label(
        "condition_occurrence", "condition_source_value",
        "condition_concept_id", "condition_start_date", "DX", "DX",
    )
    rx_df = fetch_and_label(
        "drug_exposure", "drug_source_value",
        "drug_concept_id", "drug_exposure_start_date", "RX", "RX",
    )
    px_df = fetch_and_label(
        "procedure_occurrence", "procedure_source_value",
        "procedure_concept_id", "procedure_date", "PX", "PX",
    )

    # Measurement
    try:
        meas_raw = query_measurements(con)
        meas_raw = meas_raw.loc[meas_raw["person_id"].isin(birth_map.keys())]
        meas_raw = add_age_in_days(meas_raw, birth_map, drop_log, "measurement")
        lab_df, _ = measurement_to_lab_labels(meas_raw, train_set, drop_log)
    except duckdb.Error as e:
        print(f"WARN: measurement: {e}", file=sys.stderr)
        lab_df = pd.DataFrame(columns=["person_id", "age_in_days", "label"])

    # Death
    try:
        dth_raw = query_death(con)
        dth_raw = dth_raw.loc[dth_raw["person_id"].isin(birth_map.keys())]
        dth_raw = add_age_in_days(dth_raw, birth_map, drop_log, "death")
        dth_raw["label"] = "DTH:DEATH"
        dth_df = dth_raw[["person_id", "age_in_days", "label"]]
    except duckdb.Error as e:
        print(f"WARN: death: {e}", file=sys.stderr)
        dth_df = pd.DataFrame(columns=["person_id", "age_in_days", "label"])

    # LIFESTYLE: not present in standard OMOP v5.4 measurement; v0.1 skips.
    ls_df = pd.DataFrame(columns=["person_id", "age_in_days", "label"])

    events_by_type = {
        "DX": dx_df, "RX": rx_df, "PX": px_df,
        "LAB": lab_df, "LIFESTYLE": ls_df, "DTH": dth_df, "SEX": sex_df,
    }

    # --- 4. Vocab ---
    vocab, label_to_id = build_vocab(events_by_type, args.vocab_cap)

    # --- 5. Map labels to token_ids, drop OOV ---
    def attach_token(df, ttn):
        if len(df) == 0:
            return df.assign(token_id=pd.Series(dtype=int),
                             token_type=pd.Series(dtype=int))
        ids = df["label"].map(lambda lab: label_to_id.get((ttn, lab)))
        oov = ids.isna()
        drop_log.append({
            "reason": "not_in_top_2000", "table": ttn,
            "source_value": "", "count": int(oov.sum()),
        })
        df = df.loc[~oov].copy()
        df["token_id"] = ids.loc[~oov].astype(int)
        df["token_type"] = TT[ttn]
        return df

    parts = []
    for ttn, df in events_by_type.items():
        parts.append(attach_token(df, ttn))
    all_events = pd.concat(parts, axis=0, ignore_index=True)

    # --- 6. Same-day deterministic ordering (per token_spec §7) ---
    type_priority = [TT["SEX"], TT["DX"], TT["RX"], TT["PX"],
                     TT["LAB"], TT["LIFESTYLE"], TT["DTH"]]
    type_rank = {t: i for i, t in enumerate(type_priority)}
    all_events["_type_rank"] = all_events["token_type"].map(type_rank).fillna(99).astype(int)
    all_events = all_events.sort_values(
        ["person_id", "age_in_days", "_type_rank", "token_id"]
    ).drop(columns=["_type_rank"]).reset_index(drop=True)

    # --- 7. Patient-level dense ids (preserve order) ---
    pid_order = pd.Series(all_events["person_id"].unique())
    pid_map = {int(pid): i for i, pid in enumerate(pid_order)}
    all_events["patient_id"] = all_events["person_id"].map(pid_map).astype(int)

    # --- 8. Split by patient ---
    is_val = all_events["person_id"].isin(val_set)
    train_arr = all_events.loc[~is_val, ["patient_id", "age_in_days", "token_id", "token_type"]].to_numpy(dtype=np.uint32)
    val_arr = all_events.loc[is_val, ["patient_id", "age_in_days", "token_id", "token_type"]].to_numpy(dtype=np.uint32)

    # Re-dense patient_id within each split for FERMAT's get_p2i contiguity
    def redense(arr):
        if len(arr) == 0:
            return arr
        old = arr[:, 0]
        unique_sorted = np.unique(old)
        remap = {int(u): i for i, u in enumerate(unique_sorted)}
        # Preserve original ordering by stable sort on the original id
        new = np.array([remap[int(x)] for x in old], dtype=np.uint32)
        arr = arr.copy()
        arr[:, 0] = new
        # Sort by (new_pid, age, ...)
        order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
        return arr[order]

    train_arr = redense(train_arr)
    val_arr = redense(val_arr)

    train_arr.tofile(os.path.join(args.out_dir, "train.bin"))
    val_arr.tofile(os.path.join(args.out_dir, "val.bin"))

    # Update vocab frequency from actual training tokens
    if len(train_arr):
        train_token_counts = pd.Series(train_arr[:, 2]).value_counts()
        vocab["frequency"] = vocab["token_id"].map(train_token_counts).fillna(0).astype(int)
    vocab.to_csv(os.path.join(args.out_dir, "vocab.csv"), index=False)

    # patient_split.csv
    split_records = []
    for pid in pid_order:
        split_records.append({
            "person_id": int(pid),
            "patient_id": pid_map[int(pid)],
            "split": "val" if int(pid) in val_set else "train",
        })
    pd.DataFrame(split_records).to_csv(
        os.path.join(args.out_dir, "patient_split.csv"), index=False
    )

    # Drop log
    drop_df = pd.DataFrame(drop_log)
    drop_df = drop_df.groupby(["reason", "table"], as_index=False)["count"].sum()
    drop_df["source_value"] = ""
    drop_df = drop_df[["reason", "table", "source_value", "count"]]
    drop_df.to_csv(os.path.join(args.out_dir, "dropped_events.csv"), index=False)

    print(f"Preprocess complete -> {args.out_dir}")
    print(f"  train: {len(train_arr)} rows from {(~is_val).sum()} events / "
          f"{len(set(train_arr[:, 0])) if len(train_arr) else 0} patients")
    print(f"  val:   {len(val_arr)} rows from {is_val.sum()} events / "
          f"{len(set(val_arr[:, 0])) if len(val_arr) else 0} patients")
    print(f"  vocab: {len(vocab)} tokens (cap={args.vocab_cap})")
    con.close()


if __name__ == "__main__":
    main()
