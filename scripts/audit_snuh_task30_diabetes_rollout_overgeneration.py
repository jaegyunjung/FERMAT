#!/usr/bin/env python3
"""Audit why the completed Task 30 diabetes rollout overpredicts risk.

This is a CPU-only post-processing step.  It reads the durable 1,000 x 32
trajectory summaries from the completed run and does not load a checkpoint or
generate any new futures.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_INPUT_DIR = (
    TASK30 / "outputs" / "diabetes_main_rollout_20260716_1000x32"
)
DEFAULT_OUTPUT_DIR = (
    TASK30 / "outputs" / "diabetes_rollout_overgeneration_audit_20260718"
)
LANDMARKS = [(0, "index"), (365, "1y"), (1095, "3y"), (1826, "5y")]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def kaplan_meier_risk_at(duration, event, day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    valid = np.isfinite(duration) & (duration >= 0)
    duration = duration[valid]
    event = event[valid]
    event_times = np.unique(duration[event & (duration <= day)])
    event_times.sort()
    survival = 1.0
    for event_time in event_times:
        at_risk = int(np.sum(duration >= event_time))
        events = int(np.sum(event & np.isclose(duration, event_time, atol=1e-7)))
        if at_risk and events:
            survival *= 1.0 - events / at_risk
    return 1.0 - survival


def validate_trajectories(frame):
    required = {
        "person_id",
        "rollout_index",
        "first_diabetes_day",
        "generated_death_day",
        "generated_followup_end_day",
        "reached_horizon_or_death",
        "max_token_cap_before_horizon",
        "valid_generated_events",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trajectory file is missing columns: {missing}")
    if frame.duplicated(["person_id", "rollout_index"]).any():
        raise ValueError("Duplicate person_id/rollout_index trajectory keys")
    counts = frame.groupby("person_id").size()
    if len(counts) != 1000 or not counts.eq(32).all() or len(frame) != 32000:
        raise ValueError(
            "Expected exactly 1,000 patients x 32 rollouts; "
            f"found patients={len(counts):,}, trajectories={len(frame):,}, "
            f"per-patient range={counts.min()}..{counts.max()}"
        )


def trajectory_landmarks(frame):
    hit_day = frame["first_diabetes_day"].to_numpy(dtype=np.float64)
    death_day = frame["generated_death_day"].to_numpy(dtype=np.float64)
    follow_day = frame["generated_followup_end_day"].to_numpy(dtype=np.float64)
    cap = frame["max_token_cap_before_horizon"].astype(bool).to_numpy()
    rows = []
    for day, label in LANDMARKS:
        hit = np.isfinite(hit_day) & (hit_day <= day)
        death = np.isfinite(death_day) & (death_day <= day)
        covered = follow_day >= day
        usable = hit | death | covered
        unresolved_cap = cap & ~hit & ~death & ~covered
        duration = np.where(
            hit,
            hit_day,
            np.minimum(
                np.where(np.isfinite(death_day), death_day, float(day)),
                np.minimum(follow_day, float(day)),
            ),
        )
        rows.append(
            {
                "day": day,
                "landmark": label,
                "trajectories": len(frame),
                "diabetes_hits": int(hit.sum()),
                "deaths_by_landmark": int(death.sum()),
                "covered_through_landmark": int(covered.sum()),
                "usable_trajectories": int(usable.sum()),
                "usable_fraction": float(usable.mean()),
                "unresolved_token_cap_trajectories": int(unresolved_cap.sum()),
                "unresolved_token_cap_fraction": float(unresolved_cap.mean()),
                "hit_fraction_all_trajectories": float(hit.mean()),
                "hit_fraction_among_usable": (
                    float(hit.sum() / usable.sum()) if usable.any() else np.nan
                ),
                "cause_specific_km_risk": kaplan_meier_risk_at(
                    duration, hit, day
                ),
            }
        )
    return pd.DataFrame(rows)


def hit_timing(frame):
    hit = pd.to_numeric(frame["first_diabetes_day"], errors="coerce")
    bins = [-np.inf, 0, 30, 90, 180, 365, 730, 1095, 1460, 1826, np.inf]
    labels = [
        "day_0",
        "day_1_30",
        "day_31_90",
        "day_91_180",
        "day_181_365",
        "day_366_730",
        "day_731_1095",
        "day_1096_1460",
        "day_1461_1826",
        "after_1826",
    ]
    category = pd.cut(hit, bins=bins, labels=labels, right=True)
    counts = category.value_counts(sort=False).reindex(labels, fill_value=0)
    total_5y = int(((hit >= 0) & (hit <= 1826)).sum())
    rows = []
    for label, count in counts.items():
        rows.append(
            {
                "interval": label,
                "hits": int(count),
                "fraction_of_5y_hits": (
                    float(count / total_5y) if total_5y and label != "after_1826" else np.nan
                ),
                "fraction_of_all_trajectories": float(count / len(frame)),
            }
        )
    return pd.DataFrame(rows)


def patient_concentration(frame):
    work = frame[["person_id", "first_diabetes_day"]].copy()
    work["hit_1y"] = work["first_diabetes_day"].between(0, 365, inclusive="both")
    work["hit_3y"] = work["first_diabetes_day"].between(0, 1095, inclusive="both")
    work["hit_5y"] = work["first_diabetes_day"].between(0, 1826, inclusive="both")
    patient = work.groupby("person_id", as_index=False)[["hit_1y", "hit_3y", "hit_5y"]].sum()
    for horizon in ("1y", "3y", "5y"):
        patient[f"risk_{horizon}"] = patient[f"hit_{horizon}"] / 32.0

    summary_rows = []
    for horizon in ("1y", "3y", "5y"):
        hits = patient[f"hit_{horizon}"].to_numpy(dtype=np.int64)
        total_hits = int(hits.sum())
        ordered = np.sort(hits)[::-1]
        summary_rows.append(
            {
                "horizon": horizon,
                "patients_with_at_least_one_hit": int((hits > 0).sum()),
                "patients_with_at_least_four_hits": int((hits >= 4).sum()),
                "patients_with_at_least_eight_hits": int((hits >= 8).sum()),
                "mean_hits_of_32": float(hits.mean()),
                "median_hits_of_32": float(np.median(hits)),
                "p90_hits_of_32": float(np.quantile(hits, 0.90)),
                "p95_hits_of_32": float(np.quantile(hits, 0.95)),
                "maximum_hits_of_32": int(hits.max()),
                "top_10_percent_share_of_hits": (
                    float(ordered[:100].sum() / total_hits) if total_hits else np.nan
                ),
            }
        )
    return patient, pd.DataFrame(summary_rows)


def event_count_association(frame):
    work = frame[["valid_generated_events", "first_diabetes_day"]].copy()
    work["hit_5y"] = work["first_diabetes_day"].between(0, 1826, inclusive="both")
    ranked = work["valid_generated_events"].rank(method="first")
    work["event_count_quartile"] = pd.qcut(ranked, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    result = (
        work.groupby("event_count_quartile", observed=True, as_index=False)
        .agg(
            trajectories=("hit_5y", "size"),
            diabetes_hits_5y=("hit_5y", "sum"),
            mean_generated_events=("valid_generated_events", "mean"),
            median_generated_events=("valid_generated_events", "median"),
            minimum_generated_events=("valid_generated_events", "min"),
            maximum_generated_events=("valid_generated_events", "max"),
        )
    )
    result["diabetes_hit_fraction_5y"] = (
        result["diabetes_hits_5y"] / result["trajectories"]
    )
    return result


def calibration_table(input_dir, raw_landmarks):
    path = input_dir / "population_curve_landmarks.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    reported = pd.read_csv(path)
    required = {
        "day",
        "observed_km",
        "clinical_cox_mean_predicted",
        "fermat_clinical_cox_mean_predicted",
        "fermat_rollout_cause_specific_mean",
        "fermat_rollout_competing_cif_mean",
    }
    missing = sorted(required - set(reported.columns))
    if missing:
        raise ValueError(f"Population landmarks are missing columns: {missing}")
    result = reported.merge(
        raw_landmarks[
            [
                "day",
                "usable_fraction",
                "unresolved_token_cap_fraction",
                "hit_fraction_all_trajectories",
                "hit_fraction_among_usable",
            ]
        ],
        on="day",
        how="left",
        validate="one_to_one",
    )
    observed = result["observed_km"].to_numpy(dtype=np.float64)
    for column in (
        "clinical_cox_mean_predicted",
        "fermat_clinical_cox_mean_predicted",
        "fermat_rollout_cause_specific_mean",
        "fermat_rollout_competing_cif_mean",
        "hit_fraction_all_trajectories",
    ):
        values = result[column].to_numpy(dtype=np.float64)
        result[f"{column}_minus_observed"] = values - observed
        result[f"{column}_divided_by_observed"] = np.divide(
            values,
            observed,
            out=np.full(len(result), np.nan),
            where=observed > 0,
        )
    return result


def overall_summary(frame, raw_landmarks, calibration, concentration):
    five = raw_landmarks.loc[raw_landmarks["day"].eq(1826)].iloc[0]
    five_cal = calibration.loc[calibration["day"].eq(1826)].iloc[0]
    five_con = concentration.loc[concentration["horizon"].eq("5y")].iloc[0]
    follow = pd.to_numeric(frame["generated_followup_end_day"], errors="coerce")
    events = pd.to_numeric(frame["valid_generated_events"], errors="coerce")
    event_days = np.maximum(follow.to_numpy(dtype=np.float64), 1.0)
    events_per_year = events.to_numpy(dtype=np.float64) / (event_days / 365.25)
    raw_excess = float(five_cal["hit_fraction_all_trajectories_divided_by_observed"])
    conditional_excess = float(
        five_cal["fermat_rollout_competing_cif_mean_divided_by_observed"]
    )
    coverage_effect = float(
        five["hit_fraction_among_usable"] - five["hit_fraction_all_trajectories"]
    )
    return {
        "status": "COMPLETE_TASK30_DIABETES_ROLLOUT_OVERGENERATION_AUDIT",
        "cpu_only": True,
        "new_generation_performed": False,
        "patients": int(frame["person_id"].nunique()),
        "trajectories": int(len(frame)),
        "observed_risk_5y": float(five_cal["observed_km"]),
        "rollout_reported_competing_risk_5y": float(
            five_cal["fermat_rollout_competing_cif_mean"]
        ),
        "rollout_all_trajectory_hit_fraction_5y": float(
            five["hit_fraction_all_trajectories"]
        ),
        "reported_rollout_to_observed_ratio_5y": conditional_excess,
        "all_trajectory_hit_to_observed_ratio_5y": raw_excess,
        "usable_fraction_5y": float(five["usable_fraction"]),
        "unresolved_token_cap_fraction_5y": float(
            five["unresolved_token_cap_fraction"]
        ),
        "coverage_conditioning_increase_5y": coverage_effect,
        "patients_with_at_least_one_hit_5y": int(
            five_con["patients_with_at_least_one_hit"]
        ),
        "top_10_percent_share_of_5y_hits": float(
            five_con["top_10_percent_share_of_hits"]
        ),
        "valid_generated_events": {
            "mean": float(events.mean()),
            "median": float(events.median()),
            "p90": float(events.quantile(0.90)),
            "maximum": int(events.max()),
        },
        "generated_events_per_followup_year": {
            "median": float(np.nanmedian(events_per_year)),
            "p90": float(np.nanquantile(events_per_year, 0.90)),
        },
        "what_saved_outputs_can_resolve": [
            "whether incomplete horizon coverage materially inflates the reported risk",
            "whether excess hits are concentrated in a small set of patients",
            "whether first diabetes hits are generated unusually early",
            "whether trajectories with more generated events have more diabetes hits",
        ],
        "what_saved_outputs_cannot_resolve": [
            "which diabetes token or code produced each hit",
            "whether token logits or sampled time gaps are the primary model component causing excess",
            "the complete generated event sequence because it was not stored",
        ],
        "initial_read": {
            "coverage_alone_can_explain_excess": bool(
                np.isfinite(raw_excess) and raw_excess < 1.5 and conditional_excess >= 2.0
            ),
            "excess_remains_when_every_trajectory_is_kept_in_denominator": bool(
                np.isfinite(raw_excess) and raw_excess >= 2.0
            ),
        },
    }


def run_audit(input_dir, output_dir):
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    trajectory_path = input_dir / "raw" / "main_trajectories.parquet"
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(trajectory_path)
    validate_trajectories(frame)
    landmarks = trajectory_landmarks(frame)
    timing = hit_timing(frame)
    patient, concentration = patient_concentration(frame)
    event_counts = event_count_association(frame)
    calibration = calibration_table(input_dir, landmarks)
    summary = overall_summary(frame, landmarks, calibration, concentration)

    atomic_csv(landmarks, output_dir / "trajectory_landmark_diagnostics.csv")
    atomic_csv(timing, output_dir / "first_diabetes_timing.csv")
    atomic_csv(patient, output_dir / "patient_rollout_hit_counts.csv")
    atomic_csv(concentration, output_dir / "patient_hit_concentration.csv")
    atomic_csv(event_counts, output_dir / "hit_rate_by_generated_event_count.csv")
    atomic_csv(calibration, output_dir / "calibration_and_coverage.csv")
    atomic_json(summary, output_dir / "audit_summary.json")

    return_text = "\n".join(
        [
            "## STATUS COMPLETE_TASK30_DIABETES_ROLLOUT_OVERGENERATION_AUDIT",
            f"patients {summary['patients']}",
            f"trajectories {summary['trajectories']}",
            "## FIVE_YEAR_SUMMARY",
            f"observed_risk {summary['observed_risk_5y']:.8f}",
            "reported_rollout_competing_risk "
            f"{summary['rollout_reported_competing_risk_5y']:.8f}",
            "all_trajectory_hit_fraction "
            f"{summary['rollout_all_trajectory_hit_fraction_5y']:.8f}",
            f"usable_fraction {summary['usable_fraction_5y']:.8f}",
            "unresolved_token_cap_fraction "
            f"{summary['unresolved_token_cap_fraction_5y']:.8f}",
            "all_trajectory_hit_to_observed_ratio "
            f"{summary['all_trajectory_hit_to_observed_ratio_5y']:.4f}",
            "## PATIENT_CONCENTRATION",
            concentration.to_csv(index=False).rstrip(),
            "## HIT_TIMING",
            timing.to_csv(index=False).rstrip(),
            "## GENERATED_EVENT_COUNT",
            event_counts.to_csv(index=False).rstrip(),
            "## INTERPRETATION_FLAGS",
            json.dumps(summary["initial_read"], ensure_ascii=False),
            "## OUTPUT_DIR",
            str(output_dir),
        ]
    ) + "\n"
    (output_dir / "return_summary.txt").write_text(return_text, encoding="utf-8")
    print(return_text, end="", flush=True)


def self_test():
    rows = []
    for patient in range(1000):
        for rollout in range(32):
            hit = 100.0 if patient == 0 and rollout < 2 else np.nan
            rows.append(
                {
                    "person_id": patient,
                    "rollout_index": rollout,
                    "first_diabetes_day": hit,
                    "generated_death_day": np.nan,
                    "generated_followup_end_day": 1826.0,
                    "reached_horizon_or_death": True,
                    "max_token_cap_before_horizon": False,
                    "valid_generated_events": 100,
                }
            )
    frame = pd.DataFrame(rows)
    validate_trajectories(frame)
    landmarks = trajectory_landmarks(frame)
    five = landmarks.loc[landmarks["day"].eq(1826)].iloc[0]
    if not np.isclose(five["hit_fraction_all_trajectories"], 2 / 32000):
        raise AssertionError("All-denominator hit fraction failed")
    patient, concentration = patient_concentration(frame)
    if int(patient["hit_5y"].sum()) != 2:
        raise AssertionError("Patient hit aggregation failed")
    if int(concentration.loc[concentration["horizon"].eq("5y"), "patients_with_at_least_one_hit"].iloc[0]) != 1:
        raise AssertionError("Patient concentration failed")
    log("SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run_audit(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
