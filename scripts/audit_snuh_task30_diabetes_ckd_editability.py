#!/usr/bin/env python3
"""CPU-only editability audit for same-patient diabetes-to-CKD experiments.

This script does not test whether patients with a longer recorded diabetes
history have more CKD.  It asks whether the *same patient's* pre-index FERMAT
input can be edited in a controlled way before a later GPU experiment.

For each eligible test patient it checks:

* whether reviewed diabetes diagnosis tokens are present in the model-visible
  last 2,048 pre-index records;
* whether the first diabetes diagnosis day, all diabetes diagnosis rows, or
  the first-diagnosis +/- 30-day record window can be moved 1 or 3 years
  earlier/later;
* whether the moved diabetes anchor remains visible after chronological sort
  and 2,048-row truncation;
* how many DX/RX/PX/LAB rows surround the first diabetes diagnosis;
* what remains visible if all reviewed diabetes diagnosis rows are deleted.

Nearby RX/LAB/PX rows are not labelled as diabetes-specific.  The audit only
measures how much other recorded activity would be left behind by a
diagnosis-only edit.  It imports neither torch nor the FERMAT checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_diabetes_ckd_editability.json"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CONCEPT_MAP = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
DEFAULT_PATIENT_FILE = (
    TASK30
    / "outputs"
    / "diabetes_ckd_cpu_followup_20260717_180641"
    / "raw"
    / "diabetes_ckd_analysis_patients.parquet"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_ckd_editability"

TOKEN_TYPE_NAMES = {
    0: "PAD",
    1: "DX",
    2: "RX",
    3: "PX",
    4: "LAB",
    5: "LIFESTYLE",
    6: "DTH",
    7: "SEX",
    8: "NO_EVENT",
    9: "GENOMICS",
}
CLINICAL_TYPES = {1, 2, 3, 4}
REQUIRED_PATIENT_COLUMNS = {
    "pathway_id",
    "person_id",
    "split",
    "age_at_index",
    "source_recency_days",
    "event_type",
    "renal_screen_status",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--patient-file", type=Path, default=DEFAULT_PATIENT_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def atomic_to_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "pathway_id",
        "phenotype",
        "index_date",
        "split",
        "block_size",
        "onset_window_days",
        "shift_days",
        "patient_chunk_size",
        "minimum_visible_diabetes_patients",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    shifts = [int(value) for value in config["shift_days"]]
    if not shifts or 0 in shifts or len(shifts) != len(set(shifts)):
        raise ValueError("shift_days must contain distinct non-zero integers")
    if int(config["block_size"]) < 1 or int(config["patient_chunk_size"]) < 1:
        raise ValueError("block_size and patient_chunk_size must be positive")
    return config


def find_patient_file(requested):
    requested = requested.expanduser()
    if requested.is_file():
        return requested.resolve()
    pattern = "diabetes_ckd_cpu_followup_*/raw/diabetes_ckd_analysis_patients.parquet"
    candidates = sorted((TASK30 / "outputs").glob(pattern), key=lambda path: path.stat().st_mtime)
    if candidates:
        selected = candidates[-1].resolve()
        log(f"[AUTO-SELECT] patient_file={selected}")
        return selected
    raise FileNotFoundError(requested)


def normalize_args(args):
    args.config_file = args.config_file.expanduser().resolve()
    args.patient_file = find_patient_file(args.patient_file)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.concept_map = args.concept_map.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()


def prepare_output(args, fingerprint):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            shutil.rmtree(output)
        elif not args.resume:
            raise FileExistsError(f"{output} exists and is not empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw" / "patient_parts").mkdir(parents=True, exist_ok=True)
    (output / "raw" / "shift_parts").mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.is_file() and args.resume:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != fingerprint:
            raise ValueError("Existing run_config.json does not match this run")
    else:
        write_json(fingerprint, config_path)


def load_patients(path, config, patient_map_path):
    columns = pd.read_parquet(path).columns
    missing = sorted(REQUIRED_PATIENT_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"Patient file missing columns: {missing}")
    wanted = list(REQUIRED_PATIENT_COLUMNS)
    for optional in ("source_a_date", "duration_days", "any_known_renal_abnormality"):
        if optional in columns:
            wanted.append(optional)
    patients = pd.read_parquet(path, columns=sorted(set(wanted)))
    patients = patients.loc[
        patients["pathway_id"].eq(config["pathway_id"])
        & patients["split"].eq(config["split"])
    ].copy()
    if patients.empty:
        raise ValueError("No patients remain after pathway and split filtering")
    if patients["person_id"].duplicated().any():
        raise ValueError("Patient file contains duplicate person_id rows")
    patient_map = pd.read_parquet(
        patient_map_path,
        columns=["patient_id_dense", "person_id", "split"],
    )
    patient_map = patient_map.loc[patient_map["split"].eq(config["split"])].copy()
    patients = patients.merge(
        patient_map,
        on=["person_id", "split"],
        how="left",
        validate="one_to_one",
    )
    if patients["patient_id_dense"].isna().any():
        raise ValueError(
            f"Missing patient_id_dense for {int(patients['patient_id_dense'].isna().sum()):,} patients"
        )
    patients["patient_id_dense"] = patients["patient_id_dense"].astype(np.int64)
    patients["age_at_index"] = pd.to_numeric(patients["age_at_index"], errors="raise")
    patients["index_age_days"] = np.floor(patients["age_at_index"] * 365.25).astype(np.int64)
    patients["source_recency_days"] = pd.to_numeric(
        patients["source_recency_days"], errors="raise"
    ).astype(np.int64)
    return patients.sort_values("patient_id_dense").reset_index(drop=True)


def load_diabetes_tokens(concept_map_path, registry_path, phenotype):
    concept_map = pd.read_csv(concept_map_path)
    required = {"phenotype", "condition_concept_id"}
    missing = sorted(required - set(concept_map.columns))
    if missing:
        raise ValueError(f"Concept map missing columns: {missing}")
    concepts = (
        pd.to_numeric(
            concept_map.loc[concept_map["phenotype"].eq(phenotype), "condition_concept_id"],
            errors="raise",
        )
        .astype(np.int64)
        .unique()
    )
    if len(concepts) == 0:
        raise ValueError(f"No reviewed concepts for phenotype={phenotype}")
    registry = pd.read_csv(registry_path, dtype={"token_key": str})
    missing = sorted({"token_id", "token_key"} - set(registry.columns))
    if missing:
        raise ValueError(f"Token registry missing columns: {missing}")
    registry["token_id"] = pd.to_numeric(registry["token_id"], errors="raise").astype(np.int64)
    keys = {f"DX:{int(value)}" for value in concepts}
    matched = registry.loc[registry["token_key"].isin(keys), ["token_id", "token_key"]].copy()
    if matched.empty:
        raise ValueError("No reviewed diabetes concepts matched token_registry.csv")
    mapping = {
        "phenotype": phenotype,
        "reviewed_concepts": int(len(concepts)),
        "matched_registry_tokens": int(len(matched)),
        "coverage": float(len(matched) / len(concepts)),
        "token_ids": sorted(matched["token_id"].astype(int).tolist()),
        "token_keys": sorted(matched["token_key"].tolist()),
    }
    return set(mapping["token_ids"]), mapping


def load_typed_bin(path):
    require_file(path)
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.size % 4:
        raise ValueError(f"{path} is not divisible into four uint32 columns")
    data = raw.reshape(-1, 4)
    sample = data[: min(len(data), 1_000_000), 3]
    if len(sample) and int(sample.max()) >= 20:
        raise ValueError(f"{path} does not look like a typed four-column FERMAT bin")
    return data


def patient_boundaries(data, chunk_rows=5_000_000):
    if len(data) == 0:
        raise ValueError("Split bin has no rows")
    starts = [0]
    previous = int(data[0, 0])
    for start in range(0, len(data), chunk_rows):
        end = min(start + chunk_rows, len(data))
        values = np.asarray(data[start:end, 0], dtype=np.int64)
        prior = np.empty_like(values)
        prior[0] = previous
        prior[1:] = values[:-1]
        changed = np.flatnonzero(values != prior)
        starts.extend((start + changed).tolist())
        previous = int(values[-1])
    starts = np.asarray(sorted(set(starts)), dtype=np.int64)
    ends = np.r_[starts[1:], len(data)].astype(np.int64)
    ids = np.asarray(data[starts, 0], dtype=np.int64)
    if len(ids) != len(np.unique(ids)):
        raise ValueError("Patient rows are not contiguous in the split bin")
    return ids, starts, ends


def make_location_lookup(ids, starts, ends):
    if np.all(ids[1:] > ids[:-1]):
        def lookup(patient_id):
            position = int(np.searchsorted(ids, int(patient_id)))
            if position >= len(ids) or int(ids[position]) != int(patient_id):
                return None
            return int(starts[position]), int(ends[position])
        return lookup
    mapping = {
        int(patient_id): (int(start), int(end))
        for patient_id, start, end in zip(ids, starts, ends)
    }
    return mapping.get


def type_counts(rows):
    types = rows[:, 3].astype(np.int64) if len(rows) else np.array([], dtype=np.int64)
    result = {f"rows_{name.lower()}": int(np.sum(types == token_type)) for token_type, name in TOKEN_TYPE_NAMES.items()}
    result["rows_other"] = int(np.sum(~np.isin(types, list(TOKEN_TYPE_NAMES))))
    return result


def visible_after_edit(rows, moved_mask, block_size):
    order = np.argsort(rows[:, 1], kind="stable")
    sorted_rows = rows[order]
    sorted_moved = moved_mask[order]
    if len(sorted_rows) > block_size:
        sorted_rows = sorted_rows[-block_size:]
        sorted_moved = sorted_moved[-block_size:]
    return sorted_rows, sorted_moved


def shift_label(days):
    return f"earlier_{abs(days)}d" if days < 0 else f"later_{days}d"


def evaluate_shift(rows, selected_mask, diabetes_mask, index_age_days, block_size, days):
    moved_rows = int(selected_mask.sum())
    diabetes_selected = selected_mask & diabetes_mask
    edited = rows.copy()
    shifted_ages = edited[selected_mask, 1].astype(np.int64) + int(days)
    technical = bool(
        moved_rows > 0
        and len(shifted_ages)
        and shifted_ages.min() >= 0
        and shifted_ages.max() < int(index_age_days)
    )
    within_activity_span = bool(
        technical
        and shifted_ages.min() >= int(rows[:, 1].min())
        and shifted_ages.max() <= int(rows[:, 1].max())
    )
    if technical:
        edited[selected_mask, 1] = shifted_ages.astype(np.uint32)
        _, visible_moved = visible_after_edit(edited, selected_mask, block_size)
        # Recompute from the original diabetes mask after sorting so token sets are not needed here.
        order = np.argsort(edited[:, 1], kind="stable")
        sorted_diabetes = diabetes_mask[order]
        if len(sorted_diabetes) > block_size:
            sorted_diabetes = sorted_diabetes[-block_size:]
        visible_diabetes_count = int(sorted_diabetes.sum())
        visible_moved_diabetes = int((visible_moved & sorted_diabetes).sum())
        moved_visible = int(visible_moved.sum())
    else:
        visible_diabetes_count = 0
        visible_moved_diabetes = 0
        moved_visible = 0
    return {
        "shift_days": int(days),
        "shift_label": shift_label(days),
        "moved_rows": moved_rows,
        "moved_diabetes_rows": int(diabetes_selected.sum()),
        "technical_feasible": technical,
        "within_original_activity_span": within_activity_span,
        "moved_rows_visible_after_edit": moved_visible,
        "all_moved_rows_visible_after_edit": bool(technical and moved_visible == moved_rows),
        "moved_diabetes_rows_visible_after_edit": visible_moved_diabetes,
        "diabetes_rows_visible_after_edit": visible_diabetes_count,
        "diabetes_anchor_visible_after_edit": bool(visible_moved_diabetes > 0),
    }


def analyze_patient(row, rows, diabetes_token_ids, config):
    index_age = int(row.index_age_days)
    pre = rows[rows[:, 1].astype(np.int64) < index_age].copy()
    base = {
        "person_id": int(row.person_id),
        "patient_id_dense": int(row.patient_id_dense),
        "split": str(row.split),
        "renal_screen_status": str(row.renal_screen_status),
        "event_type": int(row.event_type),
        "source_recency_days": int(row.source_recency_days),
        "index_age_days": index_age,
        "full_preindex_rows": int(len(pre)),
    }
    if len(pre) == 0:
        base.update(
            {
                "history_start_age_days": np.nan,
                "history_end_age_days": np.nan,
                "diabetes_rows_full": 0,
                "diabetes_rows_visible": 0,
                "first_diabetes_age_days": np.nan,
                "last_diabetes_age_days": np.nan,
                "first_diabetes_vs_pathway_date_days": np.nan,
                "visible_original_rows": 0,
                "onset_window_rows": 0,
                "onset_window_non_diabetes_rows": 0,
                "delete_all_diabetes_rows_visible_after": 0,
                "delete_all_diabetes_technical_feasible": False,
            }
        )
        for key, value in type_counts(pre).items():
            base[f"onset_window_{key}"] = value
        return base, []

    diabetes_mask = np.isin(pre[:, 2].astype(np.int64), list(diabetes_token_ids))
    visible = pre[-int(config["block_size"]):]
    visible_diabetes = np.isin(visible[:, 2].astype(np.int64), list(diabetes_token_ids))
    diabetes_ages = pre[diabetes_mask, 1].astype(np.int64)
    base.update(
        {
            "history_start_age_days": int(pre[:, 1].min()),
            "history_end_age_days": int(pre[:, 1].max()),
            "visible_original_rows": int(len(visible)),
            "diabetes_rows_full": int(diabetes_mask.sum()),
            "diabetes_rows_visible": int(visible_diabetes.sum()),
        }
    )
    if len(diabetes_ages) == 0:
        first_age = None
        onset_mask = np.zeros(len(pre), dtype=bool)
        base.update(
            {
                "first_diabetes_age_days": np.nan,
                "last_diabetes_age_days": np.nan,
                "first_diabetes_vs_pathway_date_days": np.nan,
            }
        )
    else:
        first_age = int(diabetes_ages.min())
        onset_mask = (
            np.isin(pre[:, 3].astype(np.int64), list(CLINICAL_TYPES))
            & (np.abs(pre[:, 1].astype(np.int64) - first_age) <= int(config["onset_window_days"]))
        )
        expected_first_age = index_age - int(row.source_recency_days)
        base.update(
            {
                "first_diabetes_age_days": first_age,
                "last_diabetes_age_days": int(diabetes_ages.max()),
                "first_diabetes_vs_pathway_date_days": int(first_age - expected_first_age),
            }
        )
    onset = pre[onset_mask]
    onset_diabetes = diabetes_mask[onset_mask]
    base["onset_window_rows"] = int(len(onset))
    base["onset_window_diabetes_rows"] = int(onset_diabetes.sum())
    base["onset_window_non_diabetes_rows"] = int(len(onset) - onset_diabetes.sum())
    for key, value in type_counts(onset).items():
        base[f"onset_window_{key}"] = value

    deleted = pre[~diabetes_mask]
    deleted_visible = deleted[-int(config["block_size"]):]
    deleted_visible_diabetes = np.isin(
        deleted_visible[:, 2].astype(np.int64), list(diabetes_token_ids)
    )
    base["delete_all_diabetes_technical_feasible"] = bool(diabetes_mask.any())
    base["delete_all_diabetes_rows_visible_after"] = int(deleted_visible_diabetes.sum())
    base["delete_all_diabetes_visible_rows_after"] = int(len(deleted_visible))

    shift_rows = []
    first_diabetes_day_mask = (
        diabetes_mask & (pre[:, 1].astype(np.int64) == first_age)
        if first_age is not None
        else np.zeros(len(pre), dtype=bool)
    )
    strategies = {
        "first_diabetes_day_dx": first_diabetes_day_mask,
        "all_diabetes_dx": diabetes_mask,
        "first_diagnosis_window": onset_mask,
    }
    for strategy, selected in strategies.items():
        for days in config["shift_days"]:
            result = evaluate_shift(
                pre,
                selected,
                diabetes_mask,
                index_age,
                int(config["block_size"]),
                int(days),
            )
            result.update(
                {
                    "person_id": int(row.person_id),
                    "patient_id_dense": int(row.patient_id_dense),
                    "renal_screen_status": str(row.renal_screen_status),
                    "event_type": int(row.event_type),
                    "strategy": strategy,
                }
            )
            shift_rows.append(result)
    return base, shift_rows


def process_chunk(chunk, data, locate, diabetes_tokens, config):
    patient_rows = []
    shift_rows = []
    for row in chunk.itertuples(index=False):
        location = locate(int(row.patient_id_dense))
        if location is None:
            raise ValueError(f"patient_id_dense={row.patient_id_dense} missing from split bin")
        start, end = location
        patient_result, patient_shifts = analyze_patient(
            row,
            np.asarray(data[start:end]),
            diabetes_tokens,
            config,
        )
        patient_rows.append(patient_result)
        shift_rows.extend(patient_shifts)
    return pd.DataFrame(patient_rows), pd.DataFrame(shift_rows)


def combine_parts(paths, output):
    frames = [pd.read_parquet(path) for path in sorted(paths)]
    if not frames:
        raise ValueError(f"No raw parts found for {output}")
    combined = pd.concat(frames, ignore_index=True)
    atomic_to_parquet(combined, output)
    return combined


def summarize_shifts(frame, group_columns):
    summary = (
        frame.groupby(group_columns, sort=True, dropna=False)
        .agg(
            patients=("person_id", "size"),
            technically_feasible=("technical_feasible", "sum"),
            within_original_activity_span=("within_original_activity_span", "sum"),
            diabetes_anchor_visible_after_edit=("diabetes_anchor_visible_after_edit", "sum"),
            all_moved_rows_visible_after_edit=("all_moved_rows_visible_after_edit", "sum"),
            median_moved_rows=("moved_rows", "median"),
        )
        .reset_index()
    )
    for column in (
        "technically_feasible",
        "within_original_activity_span",
        "diabetes_anchor_visible_after_edit",
        "all_moved_rows_visible_after_edit",
    ):
        summary[f"fraction_{column}"] = summary[column] / summary["patients"]
    return summary


def build_summaries(patients, shifts, config, output_dir):
    cohort_rows = []
    group_columns = ["renal_screen_status"]
    for values, group in patients.groupby(group_columns, dropna=False, sort=True):
        status = values[0] if isinstance(values, tuple) else values
        cohort_rows.append(
            {
                "renal_screen_status": status,
                "patients": int(len(group)),
                "ckd_events_5y": int(group["event_type"].eq(1).sum()),
                "diabetes_token_in_full_history": int(group["diabetes_rows_full"].gt(0).sum()),
                "diabetes_token_visible_at_baseline": int(group["diabetes_rows_visible"].gt(0).sum()),
                "median_visible_rows": float(group["visible_original_rows"].median()),
                "median_onset_window_rows": float(group["onset_window_rows"].median()),
                "median_onset_window_non_diabetes_rows": float(
                    group["onset_window_non_diabetes_rows"].median()
                ),
            }
        )
    cohort_summary = pd.DataFrame(cohort_rows)

    shift_summary = summarize_shifts(
        shifts,
        ["strategy", "shift_label", "shift_days"],
    )
    shift_by_renal = summarize_shifts(
        shifts,
        ["renal_screen_status", "strategy", "shift_label", "shift_days"],
    )

    pilot = shifts.loc[
        shifts["technical_feasible"]
        & shifts["diabetes_anchor_visible_after_edit"]
    ].merge(
        patients[
            [
                "person_id",
                "diabetes_rows_visible",
                "onset_window_rows",
                "onset_window_non_diabetes_rows",
                "source_recency_days",
            ]
        ],
        on="person_id",
        how="left",
        validate="many_to_one",
    )
    pilot = pilot.sort_values(
        ["strategy", "shift_days", "renal_screen_status", "patient_id_dense"]
    )

    cohort_summary.to_csv(output_dir / "cohort_editability_summary.csv", index=False)
    shift_summary.to_csv(output_dir / "shift_editability_summary.csv", index=False)
    shift_by_renal.to_csv(
        output_dir / "shift_editability_by_renal_status.csv", index=False
    )
    pilot.to_csv(output_dir / "gpu_pilot_candidates.csv", index=False)
    return cohort_summary, shift_summary, pilot


def build_return_summary(patients, cohort, shifts, pilot, config, output_dir):
    visible = int(patients["diabetes_rows_visible"].gt(0).sum())
    minimum = int(config["minimum_visible_diabetes_patients"])
    status = "PASS" if visible >= minimum else "REVIEW"
    lines = [
        "## STATUS COMPLETE_TASK30_DIABETES_CKD_EDITABILITY",
        f"EDITABILITY_GATE {status}",
        f"patients {len(patients)}",
        f"patients_with_visible_diabetes_tokens {visible}",
        f"minimum_visible_diabetes_patients {minimum}",
        "",
        "## COHORT_EDITABILITY",
        cohort.to_string(index=False),
        "",
        "## SHIFT_EDITABILITY",
        shifts.to_string(index=False),
        "",
        f"gpu_pilot_candidate_rows {len(pilot)}",
        "",
        "## INTERPRETATION",
        "This audit tests same-patient editability, not cross-patient diabetes-duration association.",
        "Nearby RX/LAB/PX rows are not assumed to be diabetes-specific.",
        "The next GPU step is a small CKD rollout hit-rate pilot on technically editable patients.",
        "",
        "## OUTPUTS",
        str(output_dir / "cohort_editability_summary.csv"),
        str(output_dir / "shift_editability_summary.csv"),
        str(output_dir / "shift_editability_by_renal_status.csv"),
        str(output_dir / "gpu_pilot_candidates.csv"),
        str(output_dir / "raw" / "editability_patient_level.parquet"),
        str(output_dir / "raw" / "shift_patient_level.parquet"),
    ]
    text = "\n".join(lines) + "\n"
    (output_dir / "RETURN_THIS.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    config = {
        "block_size": 5,
        "onset_window_days": 30,
        "shift_days": [-365, 365],
    }
    row = type(
        "Row",
        (),
        {
            "person_id": 1,
            "patient_id_dense": 10,
            "split": "test",
            "renal_screen_status": "measured_no_abnormality",
            "event_type": 0,
            "source_recency_days": 500,
            "index_age_days": 10_000,
        },
    )()
    rows = np.asarray(
        [
            [10, 8_000, 20, 1],
            [10, 9_480, 99, 4],
            [10, 9_500, 10, 1],
            [10, 9_510, 30, 2],
            [10, 9_700, 40, 3],
            [10, 9_900, 50, 4],
        ],
        dtype=np.uint32,
    )
    base, shifts = analyze_patient(row, rows, {10}, config)
    if base["diabetes_rows_full"] != 1 or base["onset_window_rows"] != 3:
        raise AssertionError("Diabetes/onset-window counting failed")
    if len(shifts) != 6 or not all(item["technical_feasible"] for item in shifts):
        raise AssertionError("Shift feasibility failed")
    if base["delete_all_diabetes_rows_visible_after"] != 0:
        raise AssertionError("Delete-all check failed")
    log("[SELF-TEST PASS] same-patient editability calculations")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_args(args)
    config = load_config(args.config_file)
    split_bin = args.data_dir / f"{config['split']}.bin"
    patient_map = args.data_dir / "patient_id_map.parquet"
    registry = args.data_dir / "token_registry.csv"
    for path in (args.config_file, args.patient_file, args.concept_map, split_bin, patient_map, registry):
        require_file(path)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "patient_file": str(args.patient_file),
        "data_dir": str(args.data_dir),
        "concept_map": str(args.concept_map),
        "split_bin": str(split_bin),
    }
    prepare_output(args, fingerprint)
    started = time.time()
    patients = load_patients(args.patient_file, config, patient_map)
    diabetes_tokens, token_mapping = load_diabetes_tokens(
        args.concept_map, registry, config["phenotype"]
    )
    write_json(token_mapping, args.output_dir / "diabetes_token_mapping.json")
    log(
        f"[INPUT] patients={len(patients):,} diabetes_tokens={len(diabetes_tokens):,} split={config['split']}"
    )
    data = load_typed_bin(split_bin)
    ids, starts, ends = patient_boundaries(data)
    locate = make_location_lookup(ids, starts, ends)
    log(f"[BIN] rows={len(data):,} patients={len(ids):,}")

    chunk_size = int(config["patient_chunk_size"])
    part_count = (len(patients) + chunk_size - 1) // chunk_size
    for part_index in range(part_count):
        patient_part = args.output_dir / "raw" / "patient_parts" / f"part_{part_index:05d}.parquet"
        shift_part = args.output_dir / "raw" / "shift_parts" / f"part_{part_index:05d}.parquet"
        if args.resume and patient_part.is_file() and shift_part.is_file():
            log(f"[RESUME] part={part_index + 1}/{part_count}")
            continue
        start = part_index * chunk_size
        end = min(start + chunk_size, len(patients))
        patient_frame, shift_frame = process_chunk(
            patients.iloc[start:end], data, locate, diabetes_tokens, config
        )
        atomic_to_parquet(patient_frame, patient_part)
        atomic_to_parquet(shift_frame, shift_part)
        log(
            f"[RAW SAVED] part={part_index + 1}/{part_count} patients={len(patient_frame):,}"
        )

    patient_level = combine_parts(
        (args.output_dir / "raw" / "patient_parts").glob("part_*.parquet"),
        args.output_dir / "raw" / "editability_patient_level.parquet",
    )
    shift_level = combine_parts(
        (args.output_dir / "raw" / "shift_parts").glob("part_*.parquet"),
        args.output_dir / "raw" / "shift_patient_level.parquet",
    )
    if patient_level["person_id"].duplicated().any():
        raise ValueError("Combined patient output contains duplicates")
    expected_shift_rows = len(patient_level) * 3 * len(config["shift_days"])
    if len(shift_level) != expected_shift_rows:
        raise ValueError(
            f"Expected {expected_shift_rows:,} shift rows, found {len(shift_level):,}"
        )
    cohort, shifts, pilot = build_summaries(
        patient_level, shift_level, config, args.output_dir
    )
    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_DIABETES_CKD_EDITABILITY",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "patients": int(len(patient_level)),
        "shift_rows": int(len(shift_level)),
        "uses_gpu": False,
        "imports_torch": False,
        "loads_checkpoint": False,
        "fits_outcome_model": False,
        "cross_patient_association_used_as_gate": False,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(patient_level, cohort, shifts, pilot, config, args.output_dir)
    log("[COMPLETE] Task 30 diabetes-to-CKD editability audit finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
