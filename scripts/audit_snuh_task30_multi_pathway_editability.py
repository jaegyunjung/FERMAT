#!/usr/bin/env python3
"""CPU-only same-patient editability audit for four Task 30 pathways.

The script reads the test patients already defined by the pathway feasibility
audit, but deliberately does not read their future target-event labels or
event times.  It scans test.bin once, checks whether first diagnosis-day DX
rows remain in the model-visible 2,048-record history after controlled timing
edits, and saves full eligible cohorts plus deterministic GPU pilot samples.

Single-source pathways use first diagnosis-day shifts of -365 and +365 days.
The hypertension+dyslipidemia pathway additionally checks each diagnosis
separately and swaps their first diagnosis dates.  No torch import, model
checkpoint, outcome model, risk ratio, or rollout is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_snuh_task30_diabetes_ckd_editability import (
    atomic_to_parquet,
    load_typed_bin,
    make_location_lookup,
    patient_boundaries,
    sha256_file,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_multi_pathway_editability.json"
DEFAULT_PATHWAY_FILE = (
    TASK30
    / "outputs"
    / "pathway_feasibility_20260717_175059"
    / "raw"
    / "pathway_patient_level.parquet"
)
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "patient_phenotype_labels_wide_20180101.parquet"
)
DEFAULT_CONCEPT_MAP = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "phenotype_group_concept_map.csv"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "multi_pathway_editability"

REQUIRED_PATHWAY_COLUMNS = {
    "pathway_id",
    "person_id",
    "split",
    "source_a",
    "source_b",
    "target",
    "source_a_date",
    "source_b_date",
    "source_order",
    "source_gap_days",
    "source_recency_days",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pathway-file", type=Path, default=DEFAULT_PATHWAY_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def atomic_to_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "index_date",
        "split",
        "block_size",
        "paired_shift_days",
        "gpu_sample_patients_per_pathway",
        "planned_rollouts_per_patient_arm",
        "random_seed",
        "pathways",
        "selection_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    shifts = [int(value) for value in config["paired_shift_days"]]
    if set(shifts) != {-365, 365}:
        raise ValueError("paired_shift_days must be exactly [-365, 365]")
    if int(config["block_size"]) < 1:
        raise ValueError("block_size must be positive")
    if int(config["gpu_sample_patients_per_pathway"]) < 1:
        raise ValueError("gpu_sample_patients_per_pathway must be positive")
    if int(config["planned_rollouts_per_patient_arm"]) < 1:
        raise ValueError("planned_rollouts_per_patient_arm must be positive")
    pathways = pd.DataFrame(config["pathways"])
    required_pathway = {
        "pathway_id",
        "source_a",
        "source_b",
        "target",
        "preferred_family",
    }
    missing = sorted(required_pathway - set(pathways.columns))
    if missing:
        raise ValueError(f"Configured pathways missing keys: {missing}")
    if pathways["pathway_id"].duplicated().any() or len(pathways) != 4:
        raise ValueError("Exactly four unique pathways are required")
    return config


def find_pathway_file(requested):
    requested = requested.expanduser()
    if requested.is_file():
        return requested.resolve()
    candidates = sorted(
        (TASK30 / "outputs").glob(
            "pathway_feasibility_*/raw/pathway_patient_level.parquet"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(requested)
    selected = candidates[-1].resolve()
    log(f"[AUTO-SELECT] pathway_file={selected}")
    return selected


def normalize_args(args):
    args.config_file = args.config_file.expanduser().resolve()
    args.pathway_file = find_pathway_file(args.pathway_file)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_file = args.label_file.expanduser().resolve()
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
    for child in ("raw", "eligible", "samples", "arms"):
        (output / child).mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != fingerprint:
            raise ValueError("Existing run_config.json does not match this run")
    else:
        write_json(fingerprint, config_path)


def require_columns(frame, required, label):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_patients(args, config):
    available = pd.read_parquet(args.pathway_file).columns
    missing = sorted(REQUIRED_PATHWAY_COLUMNS - set(available))
    if missing:
        raise ValueError(f"Pathway file missing columns: {missing}")
    pathways = {item["pathway_id"] for item in config["pathways"]}
    patients = pd.read_parquet(
        args.pathway_file,
        columns=sorted(REQUIRED_PATHWAY_COLUMNS),
    )
    patients = patients.loc[
        patients["pathway_id"].isin(pathways)
        & patients["split"].eq(config["split"])
    ].copy()
    if patients.empty:
        raise ValueError("No configured test pathway patients found")
    if patients.duplicated(["pathway_id", "person_id"]).any():
        raise ValueError("Pathway input contains duplicate pathway/person rows")

    labels = pd.read_parquet(
        args.label_file,
        columns=["person_id", "split", "age_at_index"],
    )
    labels = labels.loc[labels["split"].eq(config["split"])].copy()
    if labels["person_id"].duplicated().any():
        raise ValueError("Label file contains duplicate test person_id rows")
    patient_map = pd.read_parquet(
        args.data_dir / "patient_id_map.parquet",
        columns=["person_id", "split", "patient_id_dense"],
    )
    patient_map = patient_map.loc[patient_map["split"].eq(config["split"])].copy()
    if patient_map["person_id"].duplicated().any():
        raise ValueError("Patient map contains duplicate test person_id rows")
    patients = patients.merge(labels, on=["person_id", "split"], how="left", validate="many_to_one")
    patients = patients.merge(
        patient_map,
        on=["person_id", "split"],
        how="left",
        validate="many_to_one",
    )
    if patients[["age_at_index", "patient_id_dense"]].isna().any().any():
        raise ValueError("Pathway patients are missing age_at_index or patient_id_dense")
    patients["patient_id_dense"] = patients["patient_id_dense"].astype(np.int64)
    patients["index_age_days"] = np.floor(
        pd.to_numeric(patients["age_at_index"], errors="raise") * 365.25
    ).astype(np.int64)
    for column in ("source_a_date", "source_b_date"):
        patients[column] = pd.to_datetime(patients[column], errors="coerce")
    return patients.sort_values(["pathway_id", "patient_id_dense"]).reset_index(drop=True)


def load_token_sets(config, concept_map_path, registry_path):
    phenotypes = sorted(
        {
            source
            for item in config["pathways"]
            for source in (item["source_a"], item.get("source_b"))
            if source
        }
    )
    concept_map = pd.read_csv(concept_map_path)
    require_columns(
        concept_map,
        {"phenotype", "condition_concept_id"},
        "concept map",
    )
    registry = pd.read_csv(registry_path, dtype={"token_key": str})
    require_columns(registry, {"token_id", "token_key"}, "token registry")
    registry["token_id"] = pd.to_numeric(registry["token_id"], errors="raise").astype(np.int64)
    key_to_id = dict(zip(registry["token_key"], registry["token_id"]))
    token_sets = {}
    mapping_rows = []
    for phenotype in phenotypes:
        concepts = (
            pd.to_numeric(
                concept_map.loc[
                    concept_map["phenotype"].eq(phenotype),
                    "condition_concept_id",
                ],
                errors="raise",
            )
            .astype(np.int64)
            .unique()
        )
        if len(concepts) == 0:
            raise ValueError(f"No reviewed concepts for source phenotype={phenotype}")
        keys = [f"DX:{int(value)}" for value in concepts]
        matched = sorted({int(key_to_id[key]) for key in keys if key in key_to_id})
        if not matched:
            raise ValueError(f"No registry DX tokens for source phenotype={phenotype}")
        token_sets[phenotype] = set(matched)
        mapping_rows.append(
            {
                "phenotype": phenotype,
                "reviewed_concepts": int(len(concepts)),
                "matched_registry_tokens": int(len(matched)),
                "coverage": float(len(matched) / len(concepts)),
                "stored_token_ids": "|".join(str(value) for value in matched),
            }
        )
    return token_sets, pd.DataFrame(mapping_rows)


def stable_rank(pathway_id, person_id, seed):
    value = f"{int(seed)}:{pathway_id}:{int(person_id)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def first_day_mask(rows, token_ids):
    token_mask = np.isin(rows[:, 2].astype(np.int64), list(token_ids))
    if not token_mask.any():
        return token_mask, np.zeros(len(rows), dtype=bool), None
    first_age = int(rows[token_mask, 1].astype(np.int64).min())
    day_mask = token_mask & (rows[:, 1].astype(np.int64) == first_age)
    return token_mask, day_mask, first_age


def visible_markers(rows, markers, block_size):
    order = np.argsort(rows[:, 1], kind="stable")
    if len(order) > block_size:
        order = order[-block_size:]
    return {name: marker[order] for name, marker in markers.items()}


def finalize_edit(
    edited,
    moved_mask,
    markers,
    block_size,
    technical,
):
    result = {
        "technical_feasible": bool(technical),
        "moved_rows": int(moved_mask.sum()),
    }
    if not technical:
        result.update(
            {
                "all_moved_rows_visible_after_edit": False,
                "source_a_first_day_visible_after_edit": 0,
                "source_a_first_day_all_visible_after_edit": False,
                "source_b_first_day_visible_after_edit": 0,
                "source_b_first_day_all_visible_after_edit": False,
            }
        )
        return result
    visible = visible_markers(
        edited,
        {**markers, "moved": moved_mask},
        block_size,
    )
    moved_visible = int(visible["moved"].sum())
    result["all_moved_rows_visible_after_edit"] = bool(
        moved_visible == int(moved_mask.sum())
    )
    for source in ("source_a", "source_b"):
        marker = markers[f"{source}_first_day"]
        count = int(visible[f"{source}_first_day"].sum())
        total = int(marker.sum())
        result[f"{source}_first_day_visible_after_edit"] = count
        result[f"{source}_first_day_all_visible_after_edit"] = bool(
            total > 0 and count == total
        )
    return result


def evaluate_shift(rows, source_mask, markers, index_age, block_size, days):
    edited = rows.copy()
    shifted_ages = edited[source_mask, 1].astype(np.int64) + int(days)
    technical = bool(
        source_mask.any()
        and shifted_ages.min() >= 0
        and shifted_ages.max() < int(index_age)
    )
    if technical:
        edited[source_mask, 1] = shifted_ages.astype(np.uint32)
    return finalize_edit(edited, source_mask, markers, block_size, technical)


def evaluate_swap(rows, source_a_mask, source_b_mask, markers, index_age, block_size):
    edited = rows.copy()
    if not source_a_mask.any() or not source_b_mask.any():
        technical = False
    else:
        age_a = int(rows[source_a_mask, 1].min())
        age_b = int(rows[source_b_mask, 1].min())
        technical = bool(age_a != age_b and max(age_a, age_b) < int(index_age))
        if technical:
            edited[source_a_mask, 1] = np.uint32(age_b)
            edited[source_b_mask, 1] = np.uint32(age_a)
    moved = source_a_mask | source_b_mask
    return finalize_edit(edited, moved, markers, block_size, technical)


def base_output(row, rows, first_a, first_b, config):
    block_size = int(config["block_size"])
    markers = {
        "source_a_first_day": first_a,
        "source_b_first_day": first_b,
    }
    visible = visible_markers(rows, markers, block_size) if len(rows) else markers
    result = {
        "pathway_id": row.pathway_id,
        "person_id": int(row.person_id),
        "patient_id_dense": int(row.patient_id_dense),
        "split": row.split,
        "source_a": row.source_a,
        "source_b": row.source_b if pd.notna(row.source_b) else "",
        "target": row.target,
        "source_order": row.source_order,
        "source_gap_days": row.source_gap_days,
        "source_recency_days": int(row.source_recency_days),
        "full_preindex_rows": int(len(rows)),
        "visible_original_rows": int(min(len(rows), block_size)),
        "source_a_first_day_rows": int(first_a.sum()),
        "source_a_first_day_visible_original": int(
            visible["source_a_first_day"].sum()
        ),
        "source_b_first_day_rows": int(first_b.sum()),
        "source_b_first_day_visible_original": int(
            visible["source_b_first_day"].sum()
        ),
        "future_target_outcome_read": False,
    }
    result["source_a_first_day_all_visible_original"] = bool(
        result["source_a_first_day_rows"] > 0
        and result["source_a_first_day_visible_original"]
        == result["source_a_first_day_rows"]
    )
    result["source_b_first_day_all_visible_original"] = bool(
        result["source_b_first_day_rows"] > 0
        and result["source_b_first_day_visible_original"]
        == result["source_b_first_day_rows"]
    )
    return result, markers


def analyze_patient(row, all_rows, token_sets, config):
    index_age = int(row.index_age_days)
    rows = all_rows[all_rows[:, 1].astype(np.int64) < index_age].copy()
    source_a_tokens = token_sets[row.source_a]
    _, first_a, first_a_age = first_day_mask(rows, source_a_tokens)
    source_b_name = row.source_b if pd.notna(row.source_b) and str(row.source_b) else None
    if source_b_name:
        _, first_b, first_b_age = first_day_mask(rows, token_sets[source_b_name])
    else:
        first_b = np.zeros(len(rows), dtype=bool)
        first_b_age = None
    base, markers = base_output(row, rows, first_a, first_b, config)
    base["source_a_first_age_days"] = first_a_age
    base["source_b_first_age_days"] = first_b_age
    edits = []
    for source_label, first_mask in (("source_a", first_a), ("source_b", first_b)):
        if source_label == "source_b" and not source_b_name:
            continue
        for days in config["paired_shift_days"]:
            result = evaluate_shift(
                rows,
                first_mask,
                markers,
                index_age,
                int(config["block_size"]),
                int(days),
            )
            result.update(
                {
                    "pathway_id": row.pathway_id,
                    "person_id": int(row.person_id),
                    "patient_id_dense": int(row.patient_id_dense),
                    "edit_family": f"{source_label}_shift",
                    "edit_name": (
                        f"{source_label}_earlier_{abs(int(days))}d"
                        if int(days) < 0
                        else f"{source_label}_later_{int(days)}d"
                    ),
                    "shift_days": int(days),
                    "future_target_outcome_read": False,
                }
            )
            edits.append(result)
    if source_b_name:
        result = evaluate_swap(
            rows,
            first_a,
            first_b,
            markers,
            index_age,
            int(config["block_size"]),
        )
        result.update(
            {
                "pathway_id": row.pathway_id,
                "person_id": int(row.person_id),
                "patient_id_dense": int(row.patient_id_dense),
                "edit_family": "first_diagnosis_day_swap",
                "edit_name": "swap_source_a_source_b_first_days",
                "shift_days": 0,
                "future_target_outcome_read": False,
            }
        )
        edits.append(result)
    return base, edits


def edit_row_eligible(row, require_b):
    eligible = bool(
        row.technical_feasible
        and row.all_moved_rows_visible_after_edit
        and row.source_a_first_day_all_visible_after_edit
    )
    if require_b:
        eligible = eligible and bool(row.source_b_first_day_all_visible_after_edit)
    return eligible


def paired_ids(edits, family, shifts, require_b):
    subset = edits.loc[edits["edit_family"].eq(family)].copy()
    subset["eligible_edit"] = [
        edit_row_eligible(row, require_b) for row in subset.itertuples(index=False)
    ]
    wide = subset.pivot(index="person_id", columns="shift_days", values="eligible_edit")
    wide = wide.reindex(columns=shifts, fill_value=False).fillna(False)
    return set(wide.index[wide.all(axis=1)].astype(np.int64))


def build_eligible(base, edits, item, config):
    shifts = [int(value) for value in config["paired_shift_days"]]
    require_b = bool(item.get("source_b"))
    baseline = base["source_a_first_day_all_visible_original"].astype(bool)
    if require_b:
        baseline &= base["source_b_first_day_all_visible_original"].astype(bool)
    baseline_ids = set(base.loc[baseline, "person_id"].astype(np.int64))

    family_ids = {}
    family_ids["source_a_paired_shift"] = baseline_ids & paired_ids(
        edits, "source_a_shift", shifts, require_b
    )
    if require_b:
        family_ids["source_b_paired_shift"] = baseline_ids & paired_ids(
            edits, "source_b_shift", shifts, True
        )
        swap = edits.loc[edits["edit_family"].eq("first_diagnosis_day_swap")].copy()
        swap["eligible_edit"] = [
            edit_row_eligible(row, True) for row in swap.itertuples(index=False)
        ]
        swap_ids = set(swap.loc[swap["eligible_edit"], "person_id"].astype(np.int64))
        family_ids["first_diagnosis_day_swap"] = baseline_ids & swap_ids
        family_ids["joint_timing_set"] = (
            family_ids["source_a_paired_shift"]
            & family_ids["source_b_paired_shift"]
            & family_ids["first_diagnosis_day_swap"]
        )

    parts = []
    metadata_columns = [
        "pathway_id",
        "person_id",
        "patient_id_dense",
        "split",
        "source_a",
        "source_b",
        "target",
        "source_order",
        "source_gap_days",
        "source_recency_days",
        "visible_original_rows",
        "future_target_outcome_read",
    ]
    for family, ids in family_ids.items():
        part = base.loc[base["person_id"].isin(ids), metadata_columns].copy()
        part["edit_family"] = family
        part["eligible_all_required_edits"] = True
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def arm_definitions(item):
    if item.get("source_b"):
        return [
            ("original", "original", 0),
            ("source_a_earlier_365d", "source_a_shift", -365),
            ("source_a_later_365d", "source_a_shift", 365),
            ("source_b_earlier_365d", "source_b_shift", -365),
            ("source_b_later_365d", "source_b_shift", 365),
            ("swap_first_diagnosis_days", "first_diagnosis_day_swap", 0),
        ]
    return [
        ("original", "original", 0),
        ("source_a_earlier_365d", "source_a_shift", -365),
        ("source_a_later_365d", "source_a_shift", 365),
    ]


def sample_preferred(eligible, item, config):
    family = item["preferred_family"]
    group = eligible.loc[eligible["edit_family"].eq(family)].copy()
    requested = int(config["gpu_sample_patients_per_pathway"])
    group["selection_hash"] = group["person_id"].map(
        lambda person_id: stable_rank(
            item["pathway_id"], person_id, config["random_seed"]
        )
    )
    group = group.sort_values(["selection_hash", "person_id"])
    selected = group.head(min(requested, len(group))).copy()
    selected["pilot_rank"] = np.arange(1, len(selected) + 1)
    selected["requested_sample_patients"] = requested
    selected["sample_complete"] = bool(len(selected) == requested)
    selected["sampling_used_future_target_outcome"] = False
    return selected


def expand_arms(sample, item, config):
    definitions = pd.DataFrame(
        arm_definitions(item),
        columns=["arm", "operation", "shift_days"],
    )
    keys = sample[
        [
            "pathway_id",
            "person_id",
            "patient_id_dense",
            "split",
            "source_a",
            "source_b",
            "target",
            "pilot_rank",
        ]
    ].copy()
    keys["_join"] = 1
    definitions["_join"] = 1
    arms = keys.merge(definitions, on="_join", how="inner").drop(columns="_join")
    arms["rollouts_per_patient_arm"] = int(config["planned_rollouts_per_patient_arm"])
    arms["sampling_used_future_target_outcome"] = False
    return arms.sort_values(["pilot_rank", "arm"]).reset_index(drop=True)


def process_pathway(pathway, data, locate, token_sets, item, config):
    base_rows = []
    edit_rows = []
    for row in pathway.itertuples(index=False):
        location = locate(int(row.patient_id_dense))
        if location is None:
            raise ValueError(f"patient_id_dense={row.patient_id_dense} missing from test.bin")
        start, end = location
        base, edits = analyze_patient(
            row,
            np.asarray(data[start:end]),
            token_sets,
            config,
        )
        base_rows.append(base)
        edit_rows.extend(edits)
    base = pd.DataFrame(base_rows)
    edits = pd.DataFrame(edit_rows)
    eligible = build_eligible(base, edits, item, config)
    sample = sample_preferred(eligible, item, config)
    arms = expand_arms(sample, item, config)
    return base, edits, eligible, sample, arms


def build_summary(base_parts, eligible_parts, sample_parts, arm_parts, config):
    rows = []
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        base = base_parts[pathway_id]
        eligible = eligible_parts[pathway_id]
        sample = sample_parts[pathway_id]
        preferred = item["preferred_family"]
        preferred_eligible = eligible.loc[eligible["edit_family"].eq(preferred)]
        baseline_visible = base["source_a_first_day_all_visible_original"].astype(bool)
        if item.get("source_b"):
            baseline_visible &= base["source_b_first_day_all_visible_original"].astype(bool)
        rows.append(
            {
                "pathway_id": pathway_id,
                "source_a": item["source_a"],
                "source_b": item.get("source_b") or "",
                "target": item["target"],
                "test_pathway_input_patients": int(len(base)),
                "all_source_first_days_visible_original": int(baseline_visible.sum()),
                "preferred_family": preferred,
                "preferred_family_eligible_patients": int(len(preferred_eligible)),
                "gpu_sample_requested": int(config["gpu_sample_patients_per_pathway"]),
                "gpu_sample_selected": int(len(sample)),
                "gpu_sample_complete": bool(
                    len(sample) == int(config["gpu_sample_patients_per_pathway"])
                ),
                "planned_arms": int(arm_parts[pathway_id]["arm"].nunique()),
                "planned_trajectories": int(
                    arm_parts[pathway_id]["rollouts_per_patient_arm"].sum()
                ),
                "future_target_outcome_used_for_selection": False,
            }
        )
    pathway_summary = pd.DataFrame(rows)
    family_summary = (
        pd.concat(eligible_parts.values(), ignore_index=True)
        .groupby(["pathway_id", "edit_family"], sort=True)
        .agg(eligible_patients=("person_id", "nunique"))
        .reset_index()
    )
    return pathway_summary, family_summary


def build_return_summary(pathway_summary, family_summary, samples, arms, output):
    lines = [
        "## STATUS COMPLETE_TASK30_MULTI_PATHWAY_EDITABILITY",
        f"pathways {len(pathway_summary)}",
        f"gpu_sample_patients_total {len(samples)}",
        f"planned_patient_arm_rows {len(arms)}",
        f"planned_total_trajectories {int(arms['rollouts_per_patient_arm'].sum())}",
        "future_target_outcome_used_for_selection false",
        "",
        "## PATHWAY_SUMMARY",
        pathway_summary.to_string(index=False),
        "",
        "## ELIGIBLE_BY_EDIT_FAMILY",
        family_summary.to_string(index=False),
        "",
        "## INTERPRETATION",
        "Counts begin from the existing test-split pathway cohorts, not all SNUH patients.",
        "This is an editability and sampling audit; no generated target hit rate was measured.",
        "",
        "## OUTPUTS",
        str(output / "multi_pathway_editability_summary.csv"),
        str(output / "eligible_family_summary.csv"),
        str(output / "gpu_pilot_samples.csv"),
        str(output / "gpu_pilot_arms.csv"),
        str(output / "raw" / "multi_pathway_patient_level.parquet"),
        str(output / "raw" / "multi_pathway_edit_level.parquet"),
    ]
    text = "\n".join(lines) + "\n"
    (output / "RETURN_THIS.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    config = {
        "block_size": 8,
        "paired_shift_days": [-365, 365],
        "gpu_sample_patients_per_pathway": 1,
        "planned_rollouts_per_patient_arm": 2,
        "random_seed": 42,
    }
    rows = np.asarray(
        [
            [10, 7000, 90, 4],
            [10, 8000, 11, 1],
            [10, 8200, 21, 1],
            [10, 9000, 91, 2],
            [10, 9500, 92, 3],
        ],
        dtype=np.uint32,
    )
    row = SimpleNamespace(
        pathway_id="two_source",
        person_id=1,
        patient_id_dense=10,
        split="test",
        source_a="a",
        source_b="b",
        target="target",
        source_order="a_before_b",
        source_gap_days=200,
        source_recency_days=1800,
        index_age_days=10000,
    )
    base, edits = analyze_patient(row, rows, {"a": {11}, "b": {21}}, config)
    base_frame = pd.DataFrame([base])
    edit_frame = pd.DataFrame(edits)
    item = {
        "pathway_id": "two_source",
        "source_a": "a",
        "source_b": "b",
        "target": "target",
        "preferred_family": "joint_timing_set",
    }
    eligible = build_eligible(base_frame, edit_frame, item, config)
    if set(eligible["edit_family"]) != {
        "source_a_paired_shift",
        "source_b_paired_shift",
        "first_diagnosis_day_swap",
        "joint_timing_set",
    }:
        raise AssertionError("Two-source eligibility families failed")
    sample = sample_preferred(eligible, item, config)
    arms = expand_arms(sample, item, config)
    if len(sample) != 1 or len(arms) != 6:
        raise AssertionError("Sample or arm expansion failed")
    if any(column in sample.columns for column in ("event_type", "duration_days")):
        raise AssertionError("Future outcome columns leaked into sample")

    single_row = SimpleNamespace(**{**row.__dict__, "pathway_id": "single", "source_b": ""})
    single_base, single_edits = analyze_patient(
        single_row, rows, {"a": {11}}, config
    )
    single_item = {
        "pathway_id": "single",
        "source_a": "a",
        "source_b": None,
        "target": "target",
        "preferred_family": "source_a_paired_shift",
    }
    single_eligible = build_eligible(
        pd.DataFrame([single_base]), pd.DataFrame(single_edits), single_item, config
    )
    single_sample = sample_preferred(single_eligible, single_item, config)
    single_arms = expand_arms(single_sample, single_item, config)
    if len(single_sample) != 1 or len(single_arms) != 3:
        raise AssertionError("Single-source eligibility or arms failed")
    log("[SELF-TEST PASS] four-pathway edit logic, paired eligibility, and outcome-blind sampling")


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
    for path in (
        args.config_file,
        args.pathway_file,
        args.label_file,
        args.concept_map,
        split_bin,
        patient_map,
        registry,
    ):
        require_file(path)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "pathway_file": str(args.pathway_file),
        "data_dir": str(args.data_dir),
        "label_file": str(args.label_file),
        "concept_map": str(args.concept_map),
        "split_bin": str(split_bin),
    }
    prepare_output(args, fingerprint)
    started = time.time()
    patients = load_patients(args, config)
    token_sets, mapping = load_token_sets(config, args.concept_map, registry)
    atomic_to_csv(mapping, args.output_dir / "source_token_mapping.csv")
    log(
        f"[INPUT] pathway_rows={len(patients):,} unique_patients={patients['person_id'].nunique():,}"
    )

    data = load_typed_bin(split_bin)
    ids, starts, ends = patient_boundaries(data)
    locate = make_location_lookup(ids, starts, ends)
    log(f"[BIN LOADED ONCE] rows={len(data):,} patients={len(ids):,}")

    base_parts = {}
    edit_parts = {}
    eligible_parts = {}
    sample_parts = {}
    arm_parts = {}
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        base_path = args.output_dir / "raw" / f"{pathway_id}__patient_level.parquet"
        edit_path = args.output_dir / "raw" / f"{pathway_id}__edit_level.parquet"
        eligible_path = args.output_dir / "eligible" / f"{pathway_id}__eligible.csv"
        sample_path = args.output_dir / "samples" / f"{pathway_id}__gpu_sample.csv"
        arms_path = args.output_dir / "arms" / f"{pathway_id}__gpu_arms.csv"
        if args.resume and all(
            path.is_file()
            for path in (base_path, edit_path, eligible_path, sample_path, arms_path)
        ):
            base = pd.read_parquet(base_path)
            edits = pd.read_parquet(edit_path)
            eligible = pd.read_csv(eligible_path)
            sample = pd.read_csv(sample_path)
            arms = pd.read_csv(arms_path)
            log(f"[RESUME] pathway={pathway_id} patients={len(base):,}")
        else:
            pathway = patients.loc[patients["pathway_id"].eq(pathway_id)].copy()
            if pathway.empty:
                raise ValueError(f"No input rows for pathway={pathway_id}")
            base, edits, eligible, sample, arms = process_pathway(
                pathway, data, locate, token_sets, item, config
            )
            atomic_to_parquet(base, base_path)
            atomic_to_parquet(edits, edit_path)
            atomic_to_csv(eligible, eligible_path)
            atomic_to_csv(sample, sample_path)
            atomic_to_csv(arms, arms_path)
            log(
                f"[RAW SAVED] pathway={pathway_id} patients={len(base):,} "
                f"sample_selected={len(sample):,}"
            )
        base_parts[pathway_id] = base
        edit_parts[pathway_id] = edits
        eligible_parts[pathway_id] = eligible
        sample_parts[pathway_id] = sample
        arm_parts[pathway_id] = arms

    all_base = pd.concat(base_parts.values(), ignore_index=True)
    all_edits = pd.concat(edit_parts.values(), ignore_index=True)
    all_eligible = pd.concat(eligible_parts.values(), ignore_index=True)
    all_samples = pd.concat(sample_parts.values(), ignore_index=True)
    all_arms = pd.concat(arm_parts.values(), ignore_index=True)
    atomic_to_parquet(
        all_base,
        args.output_dir / "raw" / "multi_pathway_patient_level.parquet",
    )
    atomic_to_parquet(
        all_edits,
        args.output_dir / "raw" / "multi_pathway_edit_level.parquet",
    )
    atomic_to_parquet(
        all_eligible,
        args.output_dir / "raw" / "multi_pathway_eligible_cohorts.parquet",
    )
    atomic_to_csv(all_samples, args.output_dir / "gpu_pilot_samples.csv")
    atomic_to_csv(all_arms, args.output_dir / "gpu_pilot_arms.csv")

    pathway_summary, family_summary = build_summary(
        base_parts,
        eligible_parts,
        sample_parts,
        arm_parts,
        config,
    )
    atomic_to_csv(
        pathway_summary,
        args.output_dir / "multi_pathway_editability_summary.csv",
    )
    atomic_to_csv(
        family_summary,
        args.output_dir / "eligible_family_summary.csv",
    )
    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_MULTI_PATHWAY_EDITABILITY",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "pathways": int(len(pathway_summary)),
        "pathway_patient_rows": int(len(all_base)),
        "gpu_sample_patients_total": int(len(all_samples)),
        "planned_patient_arm_rows": int(len(all_arms)),
        "planned_total_trajectories": int(
            all_arms["rollouts_per_patient_arm"].sum()
        ),
        "test_bin_loaded_once": True,
        "uses_gpu": False,
        "imports_torch": False,
        "loads_checkpoint": False,
        "future_target_outcome_used_for_selection": False,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(
        pathway_summary,
        family_summary,
        all_samples,
        all_arms,
        args.output_dir,
    )
    log("[COMPLETE] Task 30 multi-pathway editability audit finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
