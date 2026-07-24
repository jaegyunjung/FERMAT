#!/usr/bin/env python3
"""Calculate patient-specific stable diagnosis-time shift ranges on CPU.

For each previously eligible patient in five Task 30 pathways, this script
moves the first diagnosis-day DX rows while holding every other row fixed.  It
finds the earliest and latest day shifts for which:

* every moved row remains before the index date and at a nonnegative age;
* the exact same row identities remain in FERMAT's visible 2,048-row context.

The second rule separates timing changes from changes caused by a different
record entering or leaving the truncated context.  The script then identifies
a symmetric range supported by at least a configured fraction of patients and
creates 30-day analysis grids and 90-day initial rollout grids.  It loads
test.bin once and does not read future outcomes, import torch, load a model, or
run generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

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
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_continuous_shift_ranges.json"
DEFAULT_DIABETES_DIR = (
    TASK30 / "outputs" / "diabetes_ckd_paired_pilot_selection_20260718_053743"
)
DEFAULT_MULTI_DIR = (
    TASK30 / "outputs" / "multi_pathway_editability_20260718_061535"
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
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "continuous_shift_ranges"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diabetes-selection-dir", type=Path, default=DEFAULT_DIABETES_DIR)
    parser.add_argument("--multi-pathway-dir", type=Path, default=DEFAULT_MULTI_DIR)
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


def require_columns(frame, required, label):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "split",
        "block_size",
        "maximum_recommended_range_days",
        "minimum_coverage_fraction",
        "minimum_supported_patients",
        "dense_grid_step_days",
        "initial_rollout_grid_step_days",
        "initial_rollout_maximum_range_days",
        "random_seed",
        "gpu_sample_patients_per_pathway",
        "diabetes_gpu_sample_strata",
        "pathways",
        "selection_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    if not 0 < float(config["minimum_coverage_fraction"]) <= 1:
        raise ValueError("minimum_coverage_fraction must be in (0, 1]")
    for name in (
        "block_size",
        "maximum_recommended_range_days",
        "minimum_supported_patients",
        "dense_grid_step_days",
        "initial_rollout_grid_step_days",
        "initial_rollout_maximum_range_days",
        "gpu_sample_patients_per_pathway",
    ):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    pathways = pd.DataFrame(config["pathways"])
    require_columns(
        pathways,
        {"pathway_id", "source_a", "source_b", "target", "preferred_family"},
        "configured pathways",
    )
    if len(pathways) != 5 or pathways["pathway_id"].duplicated().any():
        raise ValueError("Exactly five unique pathways are required")
    return config


def find_output_dir(requested, pattern, required_relative):
    requested = requested.expanduser()
    if (requested / required_relative).is_file():
        return requested.resolve()
    candidates = sorted(
        (TASK30 / "outputs").glob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
    candidates = [path for path in candidates if (path / required_relative).is_file()]
    if not candidates:
        raise FileNotFoundError(requested / required_relative)
    selected = candidates[-1].resolve()
    log(f"[AUTO-SELECT] {selected}")
    return selected


def normalize_args(args):
    args.config_file = args.config_file.expanduser().resolve()
    args.diabetes_selection_dir = find_output_dir(
        args.diabetes_selection_dir,
        "diabetes_ckd_paired_pilot_selection_*",
        Path("raw/paired_eligible_patients.parquet"),
    )
    args.multi_pathway_dir = find_output_dir(
        args.multi_pathway_dir,
        "multi_pathway_editability_*",
        Path("raw/multi_pathway_eligible_cohorts.parquet"),
    )
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
    (output / "raw" / "pathway_parts").mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != fingerprint:
            raise ValueError("Existing run_config.json does not match this run")
    else:
        write_json(fingerprint, config_path)


def load_eligible_patients(args, config):
    pathway_lookup = {item["pathway_id"]: item for item in config["pathways"]}
    diabetes_path = (
        args.diabetes_selection_dir / "raw" / "paired_eligible_patients.parquet"
    )
    multi_path = (
        args.multi_pathway_dir / "raw" / "multi_pathway_eligible_cohorts.parquet"
    )
    diabetes = pd.read_parquet(diabetes_path)
    require_columns(
        diabetes,
        {"person_id", "patient_id_dense", "split"},
        "diabetes eligible file",
    )
    diabetes = diabetes.copy()
    diabetes["pathway_id"] = "diabetes_to_ckd"
    diabetes["source_a"] = "diabetes"
    diabetes["source_b"] = ""
    diabetes["target"] = "chronic_kidney_disease"
    diabetes["edit_family"] = "source_a_paired_shift"
    if "renal_screen_status" not in diabetes:
        diabetes["renal_screen_status"] = "unknown"

    multi = pd.read_parquet(multi_path)
    require_columns(
        multi,
        {
            "pathway_id",
            "person_id",
            "patient_id_dense",
            "split",
            "source_a",
            "source_b",
            "target",
            "edit_family",
        },
        "multi-pathway eligible file",
    )
    keep_parts = []
    for pathway_id, item in pathway_lookup.items():
        if pathway_id == "diabetes_to_ckd":
            continue
        keep_parts.append(
            multi.loc[
                multi["pathway_id"].eq(pathway_id)
                & multi["edit_family"].eq(item["preferred_family"])
            ].copy()
        )
    multi = pd.concat(keep_parts, ignore_index=True)
    multi["renal_screen_status"] = "not_applicable"

    common = [
        "pathway_id",
        "person_id",
        "patient_id_dense",
        "split",
        "source_a",
        "source_b",
        "target",
        "edit_family",
        "renal_screen_status",
    ]
    patients = pd.concat([diabetes[common], multi[common]], ignore_index=True)
    patients = patients.loc[patients["split"].eq(config["split"])].copy()
    if patients.duplicated(["pathway_id", "person_id"]).any():
        raise ValueError("Eligible inputs contain duplicate pathway/person rows")
    expected = set(pathway_lookup)
    missing = sorted(expected - set(patients["pathway_id"]))
    if missing:
        raise ValueError(f"No eligible patients for pathways: {missing}")

    labels = pd.read_parquet(
        args.label_file,
        columns=["person_id", "split", "age_at_index"],
    )
    labels = labels.loc[labels["split"].eq(config["split"])].copy()
    if labels["person_id"].duplicated().any():
        raise ValueError("Label file contains duplicate test person_id")
    patients = patients.merge(
        labels,
        on=["person_id", "split"],
        how="left",
        validate="many_to_one",
    )
    if patients["age_at_index"].isna().any():
        raise ValueError("Eligible patients are missing age_at_index")
    patients["patient_id_dense"] = patients["patient_id_dense"].astype(np.int64)
    patients["index_age_days"] = np.floor(
        pd.to_numeric(patients["age_at_index"], errors="raise") * 365.25
    ).astype(np.int64)
    patients["future_target_outcome_read"] = False
    return patients.sort_values(["pathway_id", "patient_id_dense"]).reset_index(drop=True)


def load_token_sets(config, concept_map_path, registry_path):
    sources = sorted(
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
    lookup = dict(zip(registry["token_key"], registry["token_id"]))
    token_sets = {}
    rows = []
    for source in sources:
        concepts = (
            pd.to_numeric(
                concept_map.loc[
                    concept_map["phenotype"].eq(source),
                    "condition_concept_id",
                ],
                errors="raise",
            )
            .astype(np.int64)
            .unique()
        )
        tokens = sorted(
            {
                int(lookup[f"DX:{int(concept)}"])
                for concept in concepts
                if f"DX:{int(concept)}" in lookup
            }
        )
        if not tokens:
            raise ValueError(f"No registry DX tokens for source={source}")
        token_sets[source] = set(tokens)
        rows.append(
            {
                "source": source,
                "reviewed_concepts": int(len(concepts)),
                "matched_tokens": int(len(tokens)),
                "coverage": float(len(tokens) / len(concepts)) if len(concepts) else np.nan,
            }
        )
    return token_sets, pd.DataFrame(rows)


def first_day_mask(rows, token_ids):
    token_mask = np.isin(rows[:, 2].astype(np.int64), list(token_ids))
    if not token_mask.any():
        return np.zeros(len(rows), dtype=bool), None
    first_age = int(rows[token_mask, 1].astype(np.int64).min())
    return token_mask & (rows[:, 1].astype(np.int64) == first_age), first_age


def context_ids(ages, block_size):
    order = np.argsort(ages, kind="stable")
    if len(order) > block_size:
        order = order[-block_size:]
    return np.sort(order.astype(np.int64))


def stable_context_predicate(rows, moved_mask, index_age, block_size):
    ages = rows[:, 1].astype(np.int64)
    moved_ages = ages[moved_mask]
    original_ids = context_ids(ages, block_size)
    moved_ids = np.flatnonzero(moved_mask).astype(np.int64)
    original_valid = bool(
        len(moved_ids)
        and np.all(np.isin(moved_ids, original_ids))
    )

    def stable(delta):
        if not original_valid:
            return False
        shifted = moved_ages + int(delta)
        if shifted.min() < 0 or shifted.max() >= int(index_age):
            return False
        edited_ages = ages.copy()
        edited_ages[moved_mask] = shifted
        return bool(np.array_equal(context_ids(edited_ages, block_size), original_ids))

    return stable, original_valid, moved_ages, original_ids


def minimum_true_delta(stable, technical_min):
    technical_min = int(technical_min)
    if stable(technical_min):
        return technical_min
    low = technical_min
    high = 0
    if not stable(high):
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        if stable(middle):
            high = middle
        else:
            low = middle
    return int(high)


def maximum_true_delta(stable, technical_max):
    technical_max = int(technical_max)
    if stable(technical_max):
        return technical_max
    low = 0
    high = technical_max
    if not stable(low):
        return None
    while low < high:
        middle = (low + high + 1) // 2
        if stable(middle):
            low = middle
        else:
            high = middle - 1
    return int(low)


def calculate_range(rows, moved_mask, index_age, block_size):
    stable, original_valid, moved_ages, original_ids = stable_context_predicate(
        rows, moved_mask, index_age, block_size
    )
    if not original_valid:
        return {
            "original_context_valid": False,
            "moved_rows": int(moved_mask.sum()),
            "visible_context_rows": int(len(original_ids)),
            "stable_min_delta_days": np.nan,
            "stable_max_delta_days": np.nan,
            "stable_earlier_days": 0,
            "stable_later_days": 0,
            "stable_symmetric_days": 0,
        }
    technical_min = -int(moved_ages.min())
    technical_max = int(index_age) - 1 - int(moved_ages.max())
    if len(rows) <= int(block_size):
        stable_min = technical_min
        stable_max = technical_max
    else:
        stable_min = minimum_true_delta(stable, technical_min)
        stable_max = maximum_true_delta(stable, technical_max)
    if stable_min is None or stable_max is None:
        raise RuntimeError("Stable range could not be found around delta=0")
    if not stable(stable_min) or not stable(stable_max):
        raise RuntimeError("Stable range endpoint validation failed")
    earlier = max(0, -int(stable_min))
    later = max(0, int(stable_max))
    return {
        "original_context_valid": True,
        "moved_rows": int(moved_mask.sum()),
        "visible_context_rows": int(len(original_ids)),
        "stable_min_delta_days": int(stable_min),
        "stable_max_delta_days": int(stable_max),
        "stable_earlier_days": earlier,
        "stable_later_days": later,
        "stable_symmetric_days": int(min(earlier, later)),
    }


def analyze_pathway(pathway, item, data, locate, token_sets, config):
    rows_out = []
    roles = [("source_a", item["source_a"])]
    if item.get("source_b"):
        roles.append(("source_b", item["source_b"]))
    for patient in pathway.itertuples(index=False):
        location = locate(int(patient.patient_id_dense))
        if location is None:
            raise ValueError(f"patient_id_dense={patient.patient_id_dense} missing from test.bin")
        start, end = location
        all_rows = np.asarray(data[start:end])
        rows = all_rows[
            all_rows[:, 1].astype(np.int64) < int(patient.index_age_days)
        ].copy()
        for source_role, source in roles:
            moved_mask, first_age = first_day_mask(rows, token_sets[source])
            result = calculate_range(
                rows,
                moved_mask,
                int(patient.index_age_days),
                int(config["block_size"]),
            )
            result.update(
                {
                    "pathway_id": item["pathway_id"],
                    "person_id": int(patient.person_id),
                    "patient_id_dense": int(patient.patient_id_dense),
                    "split": patient.split,
                    "source_role": source_role,
                    "source": source,
                    "target": item["target"],
                    "first_source_age_days": first_age,
                    "full_preindex_rows": int(len(rows)),
                    "renal_screen_status": patient.renal_screen_status,
                    "future_target_outcome_read": False,
                }
            )
            rows_out.append(result)
    return pd.DataFrame(rows_out)


def add_joint_two_source_ranges(ranges, config):
    parts = [ranges]
    for item in config["pathways"]:
        if not item.get("source_b"):
            continue
        pathway = ranges.loc[ranges["pathway_id"].eq(item["pathway_id"])].copy()
        wide = pathway.pivot(
            index=["person_id", "patient_id_dense", "split", "target"],
            columns="source_role",
            values=[
                "original_context_valid",
                "stable_earlier_days",
                "stable_later_days",
                "stable_symmetric_days",
            ],
        )
        joint_rows = []
        for index, row in wide.iterrows():
            person_id, dense_id, split, target = index
            valid = bool(
                row[("original_context_valid", "source_a")]
                and row[("original_context_valid", "source_b")]
            )
            earlier = int(
                min(
                    row[("stable_earlier_days", "source_a")],
                    row[("stable_earlier_days", "source_b")],
                )
            )
            later = int(
                min(
                    row[("stable_later_days", "source_a")],
                    row[("stable_later_days", "source_b")],
                )
            )
            joint_rows.append(
                {
                    "pathway_id": item["pathway_id"],
                    "person_id": int(person_id),
                    "patient_id_dense": int(dense_id),
                    "split": split,
                    "source_role": "joint",
                    "source": f"{item['source_a']}|{item['source_b']}",
                    "target": target,
                    "first_source_age_days": np.nan,
                    "full_preindex_rows": np.nan,
                    "renal_screen_status": "not_applicable",
                    "future_target_outcome_read": False,
                    "original_context_valid": valid,
                    "moved_rows": np.nan,
                    "visible_context_rows": np.nan,
                    "stable_min_delta_days": -earlier,
                    "stable_max_delta_days": later,
                    "stable_earlier_days": earlier,
                    "stable_later_days": later,
                    "stable_symmetric_days": int(min(earlier, later)),
                }
            )
        parts.append(pd.DataFrame(joint_rows))
    return pd.concat(parts, ignore_index=True)


def coverage_curve(ranges, config):
    maximum = int(config["maximum_recommended_range_days"])
    step = int(config["dense_grid_step_days"])
    points = sorted(set([0, 365, 730, 1095, *range(0, maximum + 1, step)]))
    points = [value for value in points if value <= maximum]
    rows = []
    for (pathway_id, source_role), group in ranges.groupby(
        ["pathway_id", "source_role"], sort=True
    ):
        valid = group.loc[group["original_context_valid"].astype(bool)].copy()
        for days in points:
            symmetric = valid["stable_symmetric_days"].ge(days)
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "source_role": source_role,
                    "shift_days_absolute": int(days),
                    "valid_patients": int(len(valid)),
                    "patients_supporting_symmetric_range": int(symmetric.sum()),
                    "coverage_fraction": float(symmetric.mean()) if len(valid) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def recommended_ranges(ranges, config):
    maximum = int(config["maximum_recommended_range_days"])
    fraction = float(config["minimum_coverage_fraction"])
    minimum = int(config["minimum_supported_patients"])
    dense_step = int(config["dense_grid_step_days"])
    rollout_step = int(config["initial_rollout_grid_step_days"])
    rollout_cap = int(config["initial_rollout_maximum_range_days"])
    rows = []
    for (pathway_id, source_role), group in ranges.groupby(
        ["pathway_id", "source_role"], sort=True
    ):
        valid = group.loc[group["original_context_valid"].astype(bool)].copy()
        symmetric = valid["stable_symmetric_days"].to_numpy(dtype=np.int64)
        supported = 0
        for days in range(0, maximum + 1):
            count = int(np.sum(symmetric >= days))
            coverage = count / len(symmetric) if len(symmetric) else 0.0
            if count >= minimum and coverage >= fraction:
                supported = days
        dense_range = (supported // dense_step) * dense_step
        rollout_range = (min(supported, rollout_cap) // rollout_step) * rollout_step
        rows.append(
            {
                "pathway_id": pathway_id,
                "source_role": source_role,
                "valid_patients": int(len(valid)),
                "coverage_requirement": fraction,
                "minimum_patient_requirement": minimum,
                "maximum_supported_symmetric_days": int(supported),
                "recommended_dense_range_days": int(dense_range),
                "recommended_dense_step_days": dense_step,
                "initial_rollout_range_days": int(rollout_range),
                "initial_rollout_step_days": rollout_step,
                "patients_supporting_initial_rollout_range": int(
                    np.sum(symmetric >= rollout_range)
                ),
            }
        )
    return pd.DataFrame(rows)


def centered_grid(range_days, step_days):
    range_days = int(range_days)
    step_days = int(step_days)
    if range_days <= 0:
        return [0]
    positive = list(range(step_days, range_days + 1, step_days))
    return [-value for value in reversed(positive)] + [0] + positive


def build_grid_table(recommendations):
    rows = []
    for item in recommendations.itertuples(index=False):
        # The joint row is only the common bound for selecting the same
        # two-source patients.  Actual continuous edits move one source at a
        # time, so joint is not emitted as a GPU operation.
        if item.source_role == "joint":
            continue
        for grid_type, range_days, step_days in (
            (
                "dense_analysis",
                item.recommended_dense_range_days,
                item.recommended_dense_step_days,
            ),
            (
                "initial_rollout",
                item.initial_rollout_range_days,
                item.initial_rollout_step_days,
            ),
        ):
            for days in centered_grid(range_days, step_days):
                rows.append(
                    {
                        "pathway_id": item.pathway_id,
                        "source_role": item.source_role,
                        "operation": f"shift_{item.source_role}_first_diagnosis_day",
                        "grid_type": grid_type,
                        "shift_days": int(days),
                    }
                )
    return pd.DataFrame(rows)


def stable_rank(pathway_id, person_id, seed):
    payload = f"{int(seed)}:{pathway_id}:{int(person_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_gpu_samples(ranges, recommendations, patients, config):
    rows = []
    pathway_lookup = {item["pathway_id"]: item for item in config["pathways"]}
    seed = int(config["random_seed"])
    requested = int(config["gpu_sample_patients_per_pathway"])
    for pathway_id, item in pathway_lookup.items():
        role = "joint" if item.get("source_b") else "source_a"
        recommendation = recommendations.loc[
            recommendations["pathway_id"].eq(pathway_id)
            & recommendations["source_role"].eq(role)
        ]
        if len(recommendation) != 1:
            raise ValueError(f"Missing recommendation for pathway={pathway_id} role={role}")
        required_range = int(recommendation.iloc[0]["initial_rollout_range_days"])
        candidates = ranges.loc[
            ranges["pathway_id"].eq(pathway_id)
            & ranges["source_role"].eq(role)
            & ranges["original_context_valid"].astype(bool)
            & ranges["stable_symmetric_days"].ge(required_range)
        ].copy()
        metadata = patients.loc[
            patients["pathway_id"].eq(pathway_id),
            ["person_id", "renal_screen_status"],
        ].drop_duplicates("person_id")
        candidates = candidates.merge(
            metadata,
            on="person_id",
            how="left",
            suffixes=("", "_patient"),
            validate="one_to_one",
        )
        if pathway_id == "diabetes_to_ckd":
            strata_parts = []
            for stratum, count in config["diabetes_gpu_sample_strata"].items():
                group = candidates.loc[
                    candidates["renal_screen_status_patient"].eq(stratum)
                ].copy()
                count = int(count)
                if len(group) < count:
                    raise ValueError(
                        f"{pathway_id} stratum={stratum} supports {len(group)} patients; requested {count}"
                    )
                group["selection_hash"] = group["person_id"].map(
                    lambda value: stable_rank(
                        f"{pathway_id}:{stratum}", value, seed
                    )
                )
                group = group.sort_values(["selection_hash", "person_id"]).head(count)
                group["pilot_stratum"] = stratum
                strata_parts.append(group)
            selected = pd.concat(strata_parts, ignore_index=True)
        else:
            if len(candidates) < requested:
                raise ValueError(
                    f"{pathway_id} supports {len(candidates)} patients at range={required_range}; requested {requested}"
                )
            candidates["selection_hash"] = candidates["person_id"].map(
                lambda value: stable_rank(pathway_id, value, seed)
            )
            selected = candidates.sort_values(
                ["selection_hash", "person_id"]
            ).head(requested)
            selected["pilot_stratum"] = "unstratified"
        selected = selected.copy()
        selected["pathway_id"] = pathway_id
        selected["required_symmetric_range_days"] = required_range
        selected["sample_rank"] = np.arange(1, len(selected) + 1)
        selected["sampling_used_future_target_outcome"] = False
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["pathway_id", "sample_rank"]).reset_index(drop=True)


def build_summary(ranges, recommendations, samples, config):
    rows = []
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        role = "joint" if item.get("source_b") else "source_a"
        group = ranges.loc[
            ranges["pathway_id"].eq(pathway_id)
            & ranges["source_role"].eq(role)
        ]
        recommendation = recommendations.loc[
            recommendations["pathway_id"].eq(pathway_id)
            & recommendations["source_role"].eq(role)
        ].iloc[0]
        symmetric = group["stable_symmetric_days"]
        rows.append(
            {
                "pathway_id": pathway_id,
                "range_role": role,
                "eligible_patients_input": int(len(group)),
                "original_context_valid": int(group["original_context_valid"].sum()),
                "median_stable_symmetric_days": float(symmetric.median()),
                "p10_stable_symmetric_days": float(symmetric.quantile(0.10)),
                "maximum_supported_symmetric_days_at_80pct": int(
                    recommendation["maximum_supported_symmetric_days"]
                ),
                "recommended_dense_range_days": int(
                    recommendation["recommended_dense_range_days"]
                ),
                "dense_step_days": int(recommendation["recommended_dense_step_days"]),
                "initial_rollout_range_days": int(
                    recommendation["initial_rollout_range_days"]
                ),
                "initial_rollout_step_days": int(
                    recommendation["initial_rollout_step_days"]
                ),
                "gpu_sample_patients": int(
                    samples["pathway_id"].eq(pathway_id).sum()
                ),
                "future_target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows)


def build_return_summary(summary, recommendations, output):
    lines = [
        "## STATUS COMPLETE_TASK30_CONTINUOUS_SHIFT_RANGES",
        f"pathways {len(summary)}",
        f"gpu_sample_patients_total {int(summary['gpu_sample_patients'].sum())}",
        "future_target_outcome_used false",
        "",
        "## PATHWAY_RANGE_SUMMARY",
        summary.to_string(index=False),
        "",
        "## SOURCE_ROLE_RECOMMENDATIONS",
        recommendations.to_string(index=False),
        "",
        "## INTERPRETATION",
        "Stable means the exact same 2048 row identities remain visible while only diagnosis ages change.",
        "Dense grids are for month-scale analysis; initial rollout grids use 90-day steps to limit GPU cost.",
        "",
        "## OUTPUTS",
        str(output / "continuous_shift_range_summary.csv"),
        str(output / "source_role_range_recommendations.csv"),
        str(output / "shift_range_coverage_curve.csv"),
        str(output / "recommended_shift_grids.csv"),
        str(output / "gpu_continuous_pilot_samples.csv"),
        str(output / "raw" / "patient_source_stable_ranges.parquet"),
    ]
    text = "\n".join(lines) + "\n"
    (output / "RETURN_THIS.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    rows = np.asarray(
        [
            [1, 100, 10, 1],
            [1, 200, 20, 2],
            [1, 300, 30, 1],
            [1, 400, 40, 4],
            [1, 500, 50, 3],
        ],
        dtype=np.uint32,
    )
    moved = np.asarray([False, False, True, False, False])
    result = calculate_range(rows, moved, index_age=1000, block_size=3)
    if result["stable_min_delta_days"] != -100:
        raise AssertionError(f"Expected stable minimum -100, got {result}")
    if result["stable_max_delta_days"] != 699:
        raise AssertionError(f"Expected stable maximum 699, got {result}")
    if result["stable_symmetric_days"] != 100:
        raise AssertionError("Symmetric stable range failed")
    for delta in (-100, 0, 699):
        stable, _, _, _ = stable_context_predicate(rows, moved, 1000, 3)
        if not stable(delta):
            raise AssertionError(f"Expected delta={delta} to be stable")
    stable, _, _, _ = stable_context_predicate(rows, moved, 1000, 3)
    if stable(-101) or stable(700):
        raise AssertionError("Out-of-range delta incorrectly accepted")
    log("[SELF-TEST PASS] stable-context bounds and symmetric time range")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_args(args)
    config = load_config(args.config_file)
    split_bin = args.data_dir / f"{config['split']}.bin"
    registry = args.data_dir / "token_registry.csv"
    diabetes_file = (
        args.diabetes_selection_dir / "raw" / "paired_eligible_patients.parquet"
    )
    multi_file = (
        args.multi_pathway_dir / "raw" / "multi_pathway_eligible_cohorts.parquet"
    )
    for path in (
        args.config_file,
        split_bin,
        registry,
        args.label_file,
        args.concept_map,
        diabetes_file,
        multi_file,
    ):
        require_file(path)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "diabetes_selection_dir": str(args.diabetes_selection_dir),
        "multi_pathway_dir": str(args.multi_pathway_dir),
        "data_dir": str(args.data_dir),
        "label_file": str(args.label_file),
        "concept_map": str(args.concept_map),
        "split_bin": str(split_bin),
    }
    prepare_output(args, fingerprint)
    started = time.time()
    patients = load_eligible_patients(args, config)
    token_sets, mapping = load_token_sets(config, args.concept_map, registry)
    atomic_to_csv(mapping, args.output_dir / "source_token_mapping.csv")
    log(
        f"[INPUT] pathway_patient_rows={len(patients):,} unique_patients={patients['person_id'].nunique():,}"
    )

    data = load_typed_bin(split_bin)
    ids, starts, ends = patient_boundaries(data)
    locate = make_location_lookup(ids, starts, ends)
    log(f"[BIN LOADED ONCE] rows={len(data):,} patients={len(ids):,}")

    pathway_parts = []
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        part_path = (
            args.output_dir
            / "raw"
            / "pathway_parts"
            / f"{pathway_id}__stable_ranges.parquet"
        )
        if args.resume and part_path.is_file():
            part = pd.read_parquet(part_path)
            log(f"[RESUME] pathway={pathway_id} rows={len(part):,}")
        else:
            pathway = patients.loc[patients["pathway_id"].eq(pathway_id)].copy()
            if pathway.empty:
                raise ValueError(f"No eligible patients for pathway={pathway_id}")
            part = analyze_pathway(
                pathway,
                item,
                data,
                locate,
                token_sets,
                config,
            )
            atomic_to_parquet(part, part_path)
            log(f"[RAW SAVED] pathway={pathway_id} source_rows={len(part):,}")
        pathway_parts.append(part)

    source_ranges = pd.concat(pathway_parts, ignore_index=True)
    ranges = add_joint_two_source_ranges(source_ranges, config)
    raw_path = args.output_dir / "raw" / "patient_source_stable_ranges.parquet"
    atomic_to_parquet(ranges, raw_path)
    coverage = coverage_curve(ranges, config)
    recommendations = recommended_ranges(ranges, config)
    grids = build_grid_table(recommendations)
    samples = select_gpu_samples(ranges, recommendations, patients, config)
    summary = build_summary(ranges, recommendations, samples, config)

    atomic_to_csv(summary, args.output_dir / "continuous_shift_range_summary.csv")
    atomic_to_csv(
        recommendations,
        args.output_dir / "source_role_range_recommendations.csv",
    )
    atomic_to_csv(coverage, args.output_dir / "shift_range_coverage_curve.csv")
    atomic_to_csv(grids, args.output_dir / "recommended_shift_grids.csv")
    atomic_to_csv(samples, args.output_dir / "gpu_continuous_pilot_samples.csv")
    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_CONTINUOUS_SHIFT_RANGES",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "pathways": int(len(summary)),
        "source_role_range_rows": int(len(ranges)),
        "gpu_sample_patients_total": int(len(samples)),
        "test_bin_loaded_once": True,
        "uses_gpu": False,
        "imports_torch": False,
        "loads_checkpoint": False,
        "future_target_outcome_used": False,
        "stable_context_definition": "Exact equality of visible pre-index row identity sets before and after moving first diagnosis-day DX rows.",
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(summary, recommendations, args.output_dir)
    log("[COMPLETE] Task 30 continuous diagnosis-time range audit finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
