#!/usr/bin/env python3
"""Select a paired same-patient diabetes-to-CKD GPU pilot cohort on CPU.

The input is the completed Task 30 editability audit.  A patient is eligible
only when moving the first diabetes diagnosis day both 365 days earlier and
365 days later is technically possible and the moved diagnosis remains in the
model-visible history.  The script samples prespecified renal-screen strata
without using the patient's future CKD outcome.

Outputs include the full paired-eligible cohort, a deterministic pilot sample,
and one row per selected patient and planned arm.  No torch import, checkpoint
loading, embedding extraction, or rollout occurs.
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
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_diabetes_ckd_paired_pilot_selection.json"
DEFAULT_EDITABILITY_DIR = (
    TASK30 / "outputs" / "diabetes_ckd_editability_20260717_182124"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_ckd_paired_pilot_selection"

REQUIRED_PATIENT_COLUMNS = {
    "person_id",
    "patient_id_dense",
    "split",
    "renal_screen_status",
    "source_recency_days",
    "diabetes_rows_visible",
    "onset_window_rows",
    "onset_window_non_diabetes_rows",
}
REQUIRED_SHIFT_COLUMNS = {
    "person_id",
    "patient_id_dense",
    "renal_screen_status",
    "strategy",
    "shift_days",
    "technical_feasible",
    "all_moved_rows_visible_after_edit",
    "diabetes_anchor_visible_after_edit",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--editability-dir", type=Path, default=DEFAULT_EDITABILITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def atomic_to_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_to_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "strategy",
        "required_shift_days",
        "require_technical_feasible",
        "require_anchor_visible",
        "require_all_moved_rows_visible",
        "pilot_strata",
        "random_seed",
        "planned_arms",
        "planned_rollouts_per_patient_arm",
        "minimum_baseline_ckd_hits_to_continue",
        "selection_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    shifts = [int(value) for value in config["required_shift_days"]]
    if len(shifts) < 2 or 0 in shifts or len(shifts) != len(set(shifts)):
        raise ValueError("required_shift_days must contain distinct non-zero shifts")
    if not any(value < 0 for value in shifts) or not any(value > 0 for value in shifts):
        raise ValueError("required_shift_days must include earlier and later shifts")
    if not config["pilot_strata"]:
        raise ValueError("pilot_strata must not be empty")
    for stratum, count in config["pilot_strata"].items():
        if not str(stratum).strip() or int(count) < 1:
            raise ValueError("pilot_strata names and counts must be valid")
    thresholds = config["minimum_baseline_ckd_hits_to_continue"]
    if not isinstance(thresholds, dict):
        raise ValueError("minimum_baseline_ckd_hits_to_continue must be a stratum mapping")
    if set(thresholds) != set(config["pilot_strata"]):
        raise ValueError(
            "minimum_baseline_ckd_hits_to_continue must match pilot_strata"
        )
    if any(int(value) < 1 for value in thresholds.values()):
        raise ValueError("Baseline CKD hit thresholds must be positive")
    arms = pd.DataFrame(config["planned_arms"])
    if set(arms.columns) != {"arm", "shift_days"}:
        raise ValueError("Each planned arm must contain arm and shift_days")
    if arms["arm"].duplicated().any() or arms["shift_days"].duplicated().any():
        raise ValueError("planned arm names and shift_days must be unique")
    if set(shifts) - set(arms["shift_days"].astype(int)):
        raise ValueError("Every required shift must have a planned arm")
    if 0 not in set(arms["shift_days"].astype(int)):
        raise ValueError("planned_arms must include the original shift_days=0 arm")
    return config


def find_editability_dir(requested):
    requested = requested.expanduser()
    required = requested / "raw" / "editability_patient_level.parquet"
    if required.is_file():
        return requested.resolve()
    candidates = sorted(
        (TASK30 / "outputs").glob("diabetes_ckd_editability_*"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates = [
        path
        for path in candidates
        if (path / "raw" / "editability_patient_level.parquet").is_file()
        and (path / "raw" / "shift_patient_level.parquet").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(required)
    selected = candidates[-1].resolve()
    log(f"[AUTO-SELECT] editability_dir={selected}")
    return selected


def prepare_output(path, overwrite, fingerprint):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} exists and is not empty; use a new directory")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(parents=True, exist_ok=True)
    write_json(fingerprint, path / "run_config.json")


def require_columns(frame, required, label):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def stable_rank(person_id, seed):
    value = f"{int(seed)}:{int(person_id)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_paired_eligible(patients, shifts, config):
    require_columns(patients, REQUIRED_PATIENT_COLUMNS, "patient-level file")
    require_columns(shifts, REQUIRED_SHIFT_COLUMNS, "shift-level file")
    if patients["person_id"].duplicated().any():
        raise ValueError("Patient-level file contains duplicate person_id")

    required_shifts = [int(value) for value in config["required_shift_days"]]
    selected = shifts.loc[
        shifts["strategy"].eq(config["strategy"])
        & shifts["shift_days"].isin(required_shifts)
    ].copy()
    duplicates = selected.duplicated(["person_id", "shift_days"])
    if duplicates.any():
        raise ValueError("Shift-level file contains duplicate patient/shift rows")
    observed_shifts = set(pd.to_numeric(selected["shift_days"], errors="raise").astype(int))
    missing_shifts = sorted(set(required_shifts) - observed_shifts)
    if missing_shifts:
        raise ValueError(f"Shift-level file has no rows for shifts: {missing_shifts}")

    selected["eligible_this_shift"] = True
    if bool(config["require_technical_feasible"]):
        selected["eligible_this_shift"] &= selected["technical_feasible"].astype(bool)
    if bool(config["require_anchor_visible"]):
        selected["eligible_this_shift"] &= selected[
            "diabetes_anchor_visible_after_edit"
        ].astype(bool)
    if bool(config["require_all_moved_rows_visible"]):
        selected["eligible_this_shift"] &= selected[
            "all_moved_rows_visible_after_edit"
        ].astype(bool)

    wide = selected.pivot(
        index="person_id",
        columns="shift_days",
        values="eligible_this_shift",
    )
    wide = wide.reindex(columns=required_shifts, fill_value=False).fillna(False)
    paired_ids = wide.index[wide.all(axis=1)]
    eligible = patients.loc[patients["person_id"].isin(paired_ids)].copy()
    eligible = eligible.loc[eligible["diabetes_rows_visible"].gt(0)].copy()
    eligible = eligible.drop(columns=["event_type", "duration_days"], errors="ignore")
    selected = selected.drop(columns=["event_type", "duration_days"], errors="ignore")
    for days in required_shifts:
        eligible[f"eligible_shift_{days:+d}d"] = True
    eligible["paired_eligible"] = True
    eligible["selection_used_future_ckd_outcome"] = False
    return eligible.sort_values("patient_id_dense").reset_index(drop=True), selected


def sample_strata(eligible, config):
    parts = []
    seed = int(config["random_seed"])
    for stratum, requested in config["pilot_strata"].items():
        group = eligible.loc[eligible["renal_screen_status"].eq(stratum)].copy()
        requested = int(requested)
        if len(group) < requested:
            raise ValueError(
                f"Stratum {stratum} has {len(group):,} paired-eligible patients; requested {requested:,}"
            )
        group["selection_hash"] = group["person_id"].map(
            lambda value: stable_rank(value, seed)
        )
        group = group.sort_values(["selection_hash", "person_id"]).head(requested).copy()
        group["pilot_stratum"] = stratum
        group["pilot_rank_within_stratum"] = np.arange(1, len(group) + 1)
        parts.append(group)
    sample = pd.concat(parts, ignore_index=True)
    if sample["person_id"].duplicated().any():
        raise ValueError("Pilot sample contains duplicate patients")
    return sample.sort_values(
        ["pilot_stratum", "pilot_rank_within_stratum"]
    ).reset_index(drop=True)


def expand_arms(sample, config):
    arms = pd.DataFrame(config["planned_arms"]).copy()
    arms["shift_days"] = arms["shift_days"].astype(int)
    sample_keys = sample[
        [
            "person_id",
            "patient_id_dense",
            "split",
            "pilot_stratum",
            "pilot_rank_within_stratum",
            "renal_screen_status",
            "source_recency_days",
        ]
    ].copy()
    sample_keys["_join"] = 1
    arms["_join"] = 1
    expanded = sample_keys.merge(arms, on="_join", how="inner").drop(columns="_join")
    expanded["strategy"] = np.where(
        expanded["shift_days"].eq(0), "original", config["strategy"]
    )
    expanded["rollouts_per_patient_arm"] = int(
        config["planned_rollouts_per_patient_arm"]
    )
    expanded["sampling_used_future_ckd_outcome"] = False
    return expanded.sort_values(
        ["pilot_stratum", "pilot_rank_within_stratum", "shift_days"]
    ).reset_index(drop=True)


def build_summaries(patients, shift_rows, eligible, sample, arms, config):
    eligibility_rows = []
    configured_strata = set(config["pilot_strata"])
    all_strata = sorted(set(patients["renal_screen_status"].astype(str)) | configured_strata)
    for stratum in all_strata:
        patient_group = patients.loc[patients["renal_screen_status"].eq(stratum)]
        eligible_group = eligible.loc[eligible["renal_screen_status"].eq(stratum)]
        sample_group = sample.loc[sample["pilot_stratum"].eq(stratum)]
        eligibility_rows.append(
            {
                "renal_screen_status": stratum,
                "input_patients": int(len(patient_group)),
                "paired_eligible_patients": int(len(eligible_group)),
                "paired_eligible_fraction": (
                    float(len(eligible_group) / len(patient_group))
                    if len(patient_group)
                    else np.nan
                ),
                "pilot_requested": int(config["pilot_strata"].get(stratum, 0)),
                "pilot_selected": int(len(sample_group)),
            }
        )
    eligibility_summary = pd.DataFrame(eligibility_rows)

    shift_summary = (
        shift_rows.groupby("shift_days", sort=True)
        .agg(
            candidate_rows=("person_id", "size"),
            eligible_rows=("eligible_this_shift", "sum"),
        )
        .reset_index()
    )
    shift_summary["eligible_fraction"] = (
        shift_summary["eligible_rows"] / shift_summary["candidate_rows"]
    )

    arm_summary = (
        arms.groupby(["arm", "shift_days"], sort=True)
        .agg(
            patients=("person_id", "nunique"),
            planned_trajectories=("rollouts_per_patient_arm", "sum"),
        )
        .reset_index()
    )
    return eligibility_summary, shift_summary, arm_summary


def build_return_summary(eligible, sample, arms, eligibility, shift_summary, arm_summary, config, output):
    lines = [
        "## STATUS COMPLETE_TASK30_DIABETES_CKD_PAIRED_PILOT_SELECTION",
        f"paired_eligible_patients {len(eligible)}",
        f"pilot_selected_patients {len(sample)}",
        f"planned_patient_arm_rows {len(arms)}",
        f"planned_total_trajectories {int(arms['rollouts_per_patient_arm'].sum())}",
        "future_ckd_outcome_used_for_selection false",
        "",
        "## ELIGIBILITY_BY_RENAL_STATUS",
        eligibility.to_string(index=False),
        "",
        "## REQUIRED_SHIFT_ELIGIBILITY",
        shift_summary.to_string(index=False),
        "",
        "## PLANNED_GPU_ARMS",
        arm_summary.to_string(index=False),
        "",
        "## GPU_CONTINUATION_RULE",
        "Run the original arm first. Continue only if each stratum reaches its CKD-hit threshold:",
        *[
            f"{stratum} >= {int(threshold)} hits"
            for stratum, threshold in config[
                "minimum_baseline_ckd_hits_to_continue"
            ].items()
        ],
        "",
        "## OUTPUTS",
        str(output / "raw" / "paired_eligible_patients.parquet"),
        str(output / "gpu_pilot_sample.csv"),
        str(output / "gpu_pilot_arms.csv"),
        str(output / "paired_eligibility_summary.csv"),
    ]
    text = "\n".join(lines) + "\n"
    (output / "RETURN_THIS.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    patients = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "patient_id_dense": [10, 20, 30, 40],
            "split": ["test"] * 4,
            "renal_screen_status": [
                "measured_no_abnormality",
                "measured_no_abnormality",
                "recent_abnormality",
                "recent_abnormality",
            ],
            "event_type": [0, 1, 0, 1],
            "source_recency_days": [500, 600, 700, 800],
            "diabetes_rows_visible": [1, 1, 1, 1],
            "onset_window_rows": [5, 6, 7, 8],
            "onset_window_non_diabetes_rows": [4, 5, 6, 7],
        }
    )
    rows = []
    for person_id in patients["person_id"]:
        for days in (-365, 365):
            rows.append(
                {
                    "person_id": person_id,
                    "patient_id_dense": person_id * 10,
                    "renal_screen_status": patients.loc[
                        patients["person_id"].eq(person_id), "renal_screen_status"
                    ].iloc[0],
                    "strategy": "first_diabetes_day_dx",
                    "shift_days": days,
                    "technical_feasible": not (person_id == 2 and days == 365),
                    "all_moved_rows_visible_after_edit": True,
                    "diabetes_anchor_visible_after_edit": True,
                }
            )
    shifts = pd.DataFrame(rows)
    config = {
        "strategy": "first_diabetes_day_dx",
        "required_shift_days": [-365, 365],
        "require_technical_feasible": True,
        "require_anchor_visible": True,
        "require_all_moved_rows_visible": True,
        "pilot_strata": {"measured_no_abnormality": 1, "recent_abnormality": 1},
        "random_seed": 42,
        "planned_arms": [
            {"arm": "original", "shift_days": 0},
            {"arm": "earlier", "shift_days": -365},
            {"arm": "later", "shift_days": 365},
        ],
        "planned_rollouts_per_patient_arm": 16,
    }
    eligible, selected_shifts = select_paired_eligible(patients, shifts, config)
    if set(eligible["person_id"]) != {1, 3, 4}:
        raise AssertionError("Paired intersection failed")
    sample = sample_strata(eligible, config)
    arms = expand_arms(sample, config)
    if len(sample) != 2 or len(arms) != 6:
        raise AssertionError("Sampling or arm expansion failed")
    if "event_type" in sample.columns or "event_type" in selected_shifts.columns:
        raise AssertionError("Future outcome leaked into selection outputs")
    log("[SELF-TEST PASS] paired eligibility, outcome-blind sampling, and arm expansion")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    args.config_file = args.config_file.expanduser().resolve()
    args.editability_dir = find_editability_dir(args.editability_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    config = load_config(args.config_file)
    patient_file = args.editability_dir / "raw" / "editability_patient_level.parquet"
    shift_file = args.editability_dir / "raw" / "shift_patient_level.parquet"
    for path in (args.config_file, patient_file, shift_file):
        require_file(path)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "editability_dir": str(args.editability_dir),
        "patient_file": str(patient_file),
        "shift_file": str(shift_file),
    }
    prepare_output(args.output_dir, args.overwrite, fingerprint)
    started = time.time()
    patients = pd.read_parquet(patient_file)
    shifts = pd.read_parquet(shift_file)
    eligible, selected_shift_rows = select_paired_eligible(patients, shifts, config)

    raw_path = args.output_dir / "raw" / "paired_eligible_patients.parquet"
    atomic_to_parquet(eligible, raw_path)
    atomic_to_parquet(
        selected_shift_rows,
        args.output_dir / "raw" / "required_shift_patient_rows.parquet",
    )
    log(f"[RAW SAVED] paired_eligible_patients={len(eligible):,} path={raw_path}")

    sample = sample_strata(eligible, config)
    arms = expand_arms(sample, config)
    eligibility, shift_summary, arm_summary = build_summaries(
        patients, selected_shift_rows, eligible, sample, arms, config
    )
    atomic_to_csv(sample, args.output_dir / "gpu_pilot_sample.csv")
    atomic_to_csv(arms, args.output_dir / "gpu_pilot_arms.csv")
    atomic_to_csv(eligibility, args.output_dir / "paired_eligibility_summary.csv")
    atomic_to_csv(shift_summary, args.output_dir / "required_shift_summary.csv")
    atomic_to_csv(arm_summary, args.output_dir / "planned_gpu_arm_summary.csv")

    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_DIABETES_CKD_PAIRED_PILOT_SELECTION",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "paired_eligible_patients": int(len(eligible)),
        "pilot_selected_patients": int(len(sample)),
        "planned_patient_arm_rows": int(len(arms)),
        "planned_total_trajectories": int(arms["rollouts_per_patient_arm"].sum()),
        "uses_gpu": False,
        "imports_torch": False,
        "loads_checkpoint": False,
        "future_ckd_outcome_used_for_selection": False,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(
        eligible,
        sample,
        arms,
        eligibility,
        shift_summary,
        arm_summary,
        config,
        args.output_dir,
    )
    log("[COMPLETE] Task 30 paired pilot cohort selection finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
