"""
Fallback path: generate a self-contained 4-column synthetic dataset
that exercises every TokenType (DX/RX/PX/LAB/LIFESTYLE/DTH/SEX).

This script is the fallback used when Synthetic SNUH is not yet
available. It writes data/synthetic_snuh/{train,val,vocab,patient_split}
in the exact same format as preprocess_synthetic_snuh_to_fermat.py, so
downstream scripts (check, summarize, train) treat both paths
identically.

Usage:
    python scripts/generate_self_synthetic_4col.py \
        --out_dir data/synthetic_snuh \
        --n_patients 5000 \
        --seed 42
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Token type ids must match model.TokenType
TT = {
    "PAD": 0,
    "DX": 1,
    "RX": 2,
    "PX": 3,
    "LAB": 4,
    "LIFESTYLE": 5,
    "DTH": 6,
    "SEX": 7,
    "NO_EVENT": 8,
}


# A small synthetic Korean-flavored vocabulary so the bin file looks
# like real CDM tokens (rather than meaningless ids).
DX_CODES = [
    ("I10", "essential hypertension"),
    ("E11", "type 2 diabetes mellitus"),
    ("E78", "disorders of lipoprotein metabolism"),
    ("J45", "asthma"),
    ("K21", "GERD"),
    ("F32", "depressive episode"),
    ("M54", "back pain"),
    ("N18", "chronic kidney disease"),
    ("I21", "acute MI"),
    ("C50", "breast neoplasm"),
    ("C16", "stomach neoplasm"),
    ("C18", "colon neoplasm"),
    ("I50", "heart failure"),
    ("J44", "COPD"),
    ("G30", "Alzheimer disease"),
]

RX_CODES = [
    ("C09AA02", "enalapril"),
    ("A10BA02", "metformin"),
    ("A10BJ02", "liraglutide"),
    ("C10AA05", "atorvastatin"),
    ("R03AC02", "salbutamol"),
    ("A02BC05", "esomeprazole"),
    ("N06AB06", "sertraline"),
    ("M01AE01", "ibuprofen"),
    ("B01AC06", "aspirin low-dose"),
    ("C07AB07", "bisoprolol"),
]

PX_CODES = [
    ("M0010", "office visit"),
    ("E5500", "ECG"),
    ("E7220", "spirometry"),
    ("D0030", "abdominal US"),
    ("HA670", "CT chest"),
    ("HC322", "colonoscopy"),
    ("HZ381", "PCI"),
]

LAB_KEYS = [
    "GLU",   # fasting glucose
    "HBA1C",
    "LDL",
    "HDL",
    "TG",
    "CR",    # creatinine
    "ALT",
    "AST",
    "SBP",
    "DBP",
    "BMI",
]

LIFESTYLE_KEYS = [
    "SMOKE_NEVER", "SMOKE_PAST", "SMOKE_CURRENT",
    "ALC_NONE", "ALC_LIGHT", "ALC_HEAVY",
    "EXERCISE_NONE", "EXERCISE_REGULAR",
]


def build_vocab():
    """Build a globally-unique vocab and return as a DataFrame."""
    rows = []
    next_id = 0

    def add(label, token_type, source_value=None, concept_id=None,
            extra=None, quantile_bin=None):
        nonlocal next_id
        tid = next_id
        next_id += 1
        rows.append({
            "token_id": tid,
            "token_type": TT[token_type],
            "token_type_name": token_type,
            "source_table": extra.get("table") if extra else "",
            "source_column": extra.get("column") if extra else "",
            "source_value": source_value or "",
            "concept_id": concept_id or 0,
            "label": label,
            "unit_concept_id": 0,
            "unit_source_value": "",
            "quantile_bin": quantile_bin or "",
            "frequency": 0,
        })
        return tid

    # Reserved
    pad_id = add("PAD", "PAD", source_value="<PAD>")
    no_event_id = add("NO_EVENT", "NO_EVENT", source_value="<NO_EVENT>")

    # SEX block
    sex_ids = {
        "M": add("SEX_MALE", "SEX", source_value="M"),
        "F": add("SEX_FEMALE", "SEX", source_value="F"),
    }

    # DX block
    dx_ids = {}
    for code, label in DX_CODES:
        dx_ids[code] = add(
            f"DX:{code}", "DX", source_value=code,
            extra={"table": "condition_occurrence", "column": "condition_source_value"},
        )

    # RX block
    rx_ids = {}
    for code, label in RX_CODES:
        rx_ids[code] = add(
            f"RX:{code}", "RX", source_value=code,
            extra={"table": "drug_exposure", "column": "drug_source_value"},
        )

    # PX block
    px_ids = {}
    for code, label in PX_CODES:
        px_ids[code] = add(
            f"PX:{code}", "PX", source_value=code,
            extra={"table": "procedure_occurrence", "column": "procedure_source_value"},
        )

    # LAB block (one token per <key, quantile>)
    lab_ids = {}
    for key in LAB_KEYS:
        for q in ("Q1", "Q2", "Q3"):
            lab_ids[(key, q)] = add(
                f"LAB:{key}:{q}", "LAB", source_value=key,
                extra={"table": "measurement", "column": "measurement_source_value"},
                quantile_bin=q,
            )

    # LIFESTYLE block
    ls_ids = {}
    for k in LIFESTYLE_KEYS:
        ls_ids[k] = add(
            f"LIFESTYLE:{k}", "LIFESTYLE", source_value=k,
            extra={"table": "observation", "column": "observation_source_value"},
        )

    # DTH block
    dth_ids = {}
    for code in ["I21", "C16", "C18", "J44", "I50", "UNK"]:
        dth_ids[code] = add(
            f"DTH:{code}", "DTH", source_value=code,
            extra={"table": "death", "column": "cause_source_value"},
        )

    vocab = pd.DataFrame(rows)
    blocks = {
        "PAD": pad_id, "NO_EVENT": no_event_id,
        "SEX": sex_ids, "DX": dx_ids, "RX": rx_ids,
        "PX": px_ids, "LAB": lab_ids, "LIFESTYLE": ls_ids, "DTH": dth_ids,
    }
    return vocab, blocks


def simulate_patient(rng, pid, blocks):
    """Simulate one patient's event sequence."""
    events = []  # list of (age_in_days, token_id, token_type)

    # SEX prefix at age 0
    sex = rng.choice(["M", "F"])
    events.append((0, blocks["SEX"][sex], TT["SEX"]))

    # Age range
    age_years = rng.integers(20, 80)
    death = rng.random() < 0.10
    if death:
        death_age_days = int(age_years * 365 + rng.integers(-365, 365))
    else:
        death_age_days = None

    # Hypertension/diabetes onset prob increases with age
    has_htn = rng.random() < min(0.6, age_years / 100)
    has_dm = rng.random() < min(0.4, age_years / 120)

    def rand_age_in(lo, hi):
        lo = max(0, min(lo, hi - 1))
        hi = max(lo + 1, hi)
        return int(rng.integers(lo, hi))

    # Generate diagnoses
    n_dx = rng.poisson(3) + 1
    dx_choices = list(blocks["DX"].keys())
    if has_htn:
        events.append((
            rand_age_in(30, min(60, age_years)) * 365,
            blocks["DX"]["I10"], TT["DX"]
        ))
    if has_dm and age_years > 35:
        events.append((
            rand_age_in(35, min(65, age_years)) * 365,
            blocks["DX"]["E11"], TT["DX"]
        ))
    for _ in range(n_dx):
        code = rng.choice(dx_choices)
        ev_age = rand_age_in(20, age_years) * 365 + int(rng.integers(-180, 181))
        events.append((ev_age, blocks["DX"][code], TT["DX"]))

    # Drug prescriptions: clustered around diagnosis age
    if has_htn:
        for _ in range(int(rng.integers(2, 8))):
            ev_age = rand_age_in(30, age_years) * 365 + int(rng.integers(-90, 91))
            code = rng.choice(["C09AA02", "C07AB07"])
            events.append((ev_age, blocks["RX"][code], TT["RX"]))
    if has_dm and age_years > 35:
        for _ in range(int(rng.integers(2, 6))):
            ev_age = rand_age_in(35, age_years) * 365 + int(rng.integers(-90, 91))
            code = rng.choice(["A10BA02", "A10BJ02"])
            events.append((ev_age, blocks["RX"][code], TT["RX"]))
    for _ in range(int(rng.integers(0, 4))):
        ev_age = rand_age_in(20, age_years) * 365 + int(rng.integers(-180, 181))
        code = rng.choice(list(blocks["RX"].keys()))
        events.append((ev_age, blocks["RX"][code], TT["RX"]))

    # Procedures (sparse)
    for _ in range(int(rng.integers(0, 4))):
        ev_age = rand_age_in(20, age_years) * 365 + int(rng.integers(-180, 181))
        code = rng.choice(list(blocks["PX"].keys()))
        events.append((ev_age, blocks["PX"][code], TT["PX"]))

    # Labs (biennial-ish, age 40+)
    for screening_age in range(40, age_years, 2):
        for key in LAB_KEYS:
            ev_age = int(screening_age * 365 + rng.integers(-30, 30))
            # Quantile correlated with HTN/DM
            if key in ("SBP", "DBP") and has_htn:
                q = "Q3" if rng.random() < 0.7 else "Q2"
            elif key in ("GLU", "HBA1C") and has_dm:
                q = "Q3" if rng.random() < 0.7 else "Q2"
            else:
                q = rng.choice(["Q1", "Q2", "Q3"])
            tok = blocks["LAB"][(key, q)]
            events.append((ev_age, tok, TT["LAB"]))

    # Lifestyle (one snapshot at first screening)
    if age_years >= 40:
        smoke = rng.choice(["SMOKE_NEVER", "SMOKE_PAST", "SMOKE_CURRENT"])
        alc = rng.choice(["ALC_NONE", "ALC_LIGHT", "ALC_HEAVY"])
        ex = rng.choice(["EXERCISE_NONE", "EXERCISE_REGULAR"])
        for key in (smoke, alc, ex):
            events.append((40 * 365, blocks["LIFESTYLE"][key], TT["LIFESTYLE"]))

    # Death
    if death_age_days is not None:
        cause = rng.choice(list(blocks["DTH"].keys()))
        events.append((death_age_days, blocks["DTH"][cause], TT["DTH"]))

    # Filter pre-death events and clip age
    events = [
        e for e in events
        if 0 <= e[0] <= 150 * 365
        and (death_age_days is None or e[0] <= death_age_days
             or e[2] == TT["DTH"])
    ]

    # Same-day deterministic ordering: by (age, type, token_id)
    type_priority = [TT["SEX"], TT["DX"], TT["RX"], TT["PX"],
                     TT["LAB"], TT["LIFESTYLE"], TT["DTH"]]
    type_rank = {t: i for i, t in enumerate(type_priority)}
    events.sort(key=lambda e: (e[0], type_rank.get(e[2], 99), e[1]))

    rows = np.array(
        [(pid, age, tok, ttype) for (age, tok, ttype) in events],
        dtype=np.uint32,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/synthetic_snuh")
    parser.add_argument("--n_patients", type=int, default=5000)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vocab, blocks = build_vocab()

    n_train = int(args.n_patients * (1 - args.val_frac))
    n_val = args.n_patients - n_train

    split_records = []
    train_chunks = []
    val_chunks = []

    for pid in range(n_train):
        rows = simulate_patient(rng, pid, blocks)
        if len(rows):
            train_chunks.append(rows)
            split_records.append({"patient_id": pid, "split": "train",
                                  "n_events": len(rows)})

    for j in range(n_val):
        pid = n_train + j
        rows = simulate_patient(rng, pid, blocks)
        if len(rows):
            val_chunks.append(rows)
            split_records.append({"patient_id": pid, "split": "val",
                                  "n_events": len(rows)})

    train_arr = np.concatenate(train_chunks, axis=0)
    val_arr = np.concatenate(val_chunks, axis=0)

    # Update vocab frequency by counting train tokens
    train_token_counts = pd.Series(train_arr[:, 2]).value_counts()
    vocab["frequency"] = vocab["token_id"].map(train_token_counts).fillna(0).astype(int)

    # Persist
    train_arr.tofile(os.path.join(args.out_dir, "train.bin"))
    val_arr.tofile(os.path.join(args.out_dir, "val.bin"))
    vocab.to_csv(os.path.join(args.out_dir, "vocab.csv"), index=False)
    pd.DataFrame(split_records).to_csv(
        os.path.join(args.out_dir, "patient_split.csv"), index=False
    )

    # Drop log (empty for self-synthetic — no source data to drop from)
    pd.DataFrame(columns=["reason", "table", "source_value", "count"]).to_csv(
        os.path.join(args.out_dir, "dropped_events.csv"), index=False
    )

    print(f"Self-synthetic 4-column dataset written to {args.out_dir}")
    print(f"  train: {len(train_arr)} rows from {n_train} patients")
    print(f"  val:   {len(val_arr)} rows from {n_val} patients")
    print(f"  vocab: {len(vocab)} tokens "
          f"(max token_id = {vocab['token_id'].max()})")


if __name__ == "__main__":
    main()
