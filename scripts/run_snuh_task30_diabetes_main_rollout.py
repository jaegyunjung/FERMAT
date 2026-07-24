#!/usr/bin/env python3
"""Run the approved Task 30 diabetes population rollout (1,000 x 32).

The sampling cohort is selected without reading future diabetes outcomes.  One
raw parquet file is written and validated after every patient, so an interrupted
GPU run can resume without losing completed work.  Only after all raw parts are
present does the script attach observed outcomes and saved Cox predictions to
build the population and individual curves.

This stage validates population -> individual curve construction.  It does not
select perturbation patients or make a causal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_snuh_task30_diabetes_rollout_speed_benchmark as bench


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_diabetes_main_rollout_1000x32.json"
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_main_rollout_20260716_1000x32"
LANDMARK_NAMES = {0: "index", 365: "1y", 1095: "3y", 1826: "5y"}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-dir", type=Path, default=bench.DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--baseline-audit-dir", type=Path, default=bench.DEFAULT_BASELINE_AUDIT_DIR
    )
    parser.add_argument("--data-dir", type=Path, default=bench.DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=bench.DEFAULT_LABEL_DIR)
    parser.add_argument("--label-file", type=Path, default=bench.DEFAULT_LABEL_FILE)
    parser.add_argument("--fermat-ckpt", type=Path, default=bench.DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild CPU summaries from complete raw patient parts; requires --resume.",
    )
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


def normalize_paths(args):
    for name in (
        "config_file",
        "baseline_dir",
        "baseline_audit_dir",
        "data_dir",
        "label_dir",
        "label_file",
        "fermat_ckpt",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "outcomes",
        "index_date",
        "horizon_days",
        "landmark_days",
        "split",
        "cohort_rule",
        "expected_sampling_frame_patients",
        "future_outcome_labels_loaded_for_sampling",
        "sampling",
        "patients",
        "rollouts_per_patient",
        "rollout_batch_size",
        "total_trajectories",
        "max_new_tokens",
        "top_k",
        "temperature",
        "same_day_repeat_penalty",
        "same_day_temperature",
        "same_day_probability_cap",
        "random_seed",
        "primary_estimand",
        "sensitivity_estimand",
        "runtime_basis",
        "stage_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing fields: {missing}")
    if config["outcomes"] != ["diabetes"]:
        raise ValueError("Main Task 30 rollout requires outcomes=['diabetes']")
    if config["index_date"] != "2018-01-01" or config["split"] != "test":
        raise ValueError("Main rollout is fixed to the 2018-01-01 held-out test split")
    if int(config["horizon_days"]) != 1826:
        raise ValueError("Main rollout requires horizon_days=1826")
    if [int(x) for x in config["landmark_days"]] != [0, 365, 1095, 1826]:
        raise ValueError("Main rollout landmarks must be [0, 365, 1095, 1826]")
    if bool(config["future_outcome_labels_loaded_for_sampling"]):
        raise ValueError("Future outcome labels must not be loaded for sampling")
    patients = int(config["patients"])
    rollouts = int(config["rollouts_per_patient"])
    batch_size = int(config["rollout_batch_size"])
    if (patients, rollouts, batch_size) != (1000, 32, 8):
        raise ValueError("Approved main setting is fixed to patients=1000, rollouts=32, batch=8")
    if rollouts % batch_size:
        raise ValueError("rollouts_per_patient must be divisible by rollout_batch_size")
    if int(config["total_trajectories"]) != patients * rollouts:
        raise ValueError("total_trajectories does not equal patients * rollouts")
    return config


def small_file_identity(path):
    path = Path(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": bench.sha256_file(path),
    }


def run_fingerprint(args, config):
    return {
        "analysis": config["analysis_name"],
        "config": small_file_identity(args.config_file),
        "runner": small_file_identity(Path(__file__).resolve()),
        "generation_helper": small_file_identity(
            ROOT / "scripts" / "run_snuh_task30_multi_outcome_rollout_pilot.py"
        ),
        "baseline_audit": small_file_identity(
            args.baseline_audit_dir / "baseline_reuse_audit.json"
        ),
        "cox_test_predictions": bench.tracked_file(
            args.baseline_dir / "cox_test_predictions.parquet"
        ),
        "label_file": bench.tracked_file(args.label_file),
        "patient_map": bench.tracked_file(args.data_dir / "patient_id_map.parquet"),
        "test_bin": bench.tracked_file(args.data_dir / "test.bin"),
        "registry": bench.tracked_file(bench.registry_path(args.data_dir)),
        "fermat_ckpt": bench.tracked_file(args.fermat_ckpt),
        "device": args.device,
        "dtype": args.dtype,
        "patients": int(config["patients"]),
        "rollouts_per_patient": int(config["rollouts_per_patient"]),
        "rollout_batch_size": int(config["rollout_batch_size"]),
        "random_seed": int(config["random_seed"]),
    }


def prepare_output(args, fingerprint):
    run_config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"{args.output_dir} exists and is not empty; use a new directory or --resume"
            )
        if not run_config_path.is_file():
            raise RuntimeError(f"Cannot resume without {run_config_path}")
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise RuntimeError("Resume inputs/configuration do not match run_config.json")
        log(f"[RESUME VALIDATED] {run_config_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "raw" / "patient_parts").mkdir(parents=True, exist_ok=True)
        atomic_json(fingerprint, run_config_path)
        log(f"[RAW SAVED] {run_config_path}")
    (args.output_dir / "raw" / "patient_parts").mkdir(parents=True, exist_ok=True)


def require_paths(args):
    bench.require_paths(args)
    required = []
    for model in ("clinical_cox", "fermat_clinical_cox"):
        required.append(args.baseline_dir / "models" / model / "cox_model.npz")
    required.extend(
        [
            args.baseline_dir / "full_test_observed_diabetes_curve.csv",
            args.baseline_dir / "cox_population_curves.csv",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Cox curve inputs:\n" + "\n".join(missing))


def validate_sample(sample, config):
    required = {
        "person_id",
        "split",
        "patient_id_dense",
        "index_age_days",
        "has_pre_index_washout",
        "prior__diabetes",
        "sample_order",
    }
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Sample missing columns: {missing}")
    if len(sample) != int(config["patients"]):
        raise ValueError(f"Sample rows={len(sample):,} expected={config['patients']:,}")
    if sample["person_id"].duplicated().any() or sample["sample_order"].duplicated().any():
        raise ValueError("Sample contains duplicate person_id or sample_order")
    expected_order = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(sample["sample_order"].to_numpy(dtype=np.int64), expected_order):
        raise ValueError("Sample order is not exactly 0..patients-1")
    if not sample["split"].eq("test").all():
        raise ValueError("Sample contains a non-test patient")
    if sample["prior__diabetes"].fillna(0).astype(bool).any():
        raise ValueError("Sample contains prior diabetes")


def build_or_load_sample(args, config):
    path = args.output_dir / "raw" / "main_sample_cohort.parquet"
    summary_path = args.output_dir / "sampling_frame_summary.json"
    if args.resume and path.is_file() and summary_path.is_file():
        sample = pd.read_parquet(path)
        validate_sample(sample, config)
        log(f"[RESUME] outcome-blind sample rows={len(sample):,}")
        return sample, json.loads(summary_path.read_text(encoding="utf-8"))

    frame = bench.load_sampling_frame(args, config)
    sample = frame.sample(
        n=int(config["patients"]),
        replace=False,
        random_state=int(config["random_seed"]),
    ).reset_index(drop=True)
    sample["sample_order"] = np.arange(len(sample), dtype=np.int32)
    sample = sample[
        [
            "person_id",
            "split",
            "patient_id_dense",
            "index_age_days",
            "has_pre_index_washout",
            "prior__diabetes",
            "sample_order",
        ]
    ]
    validate_sample(sample, config)
    bench.atomic_to_parquet(sample, path)
    summary = {
        "validated_baseline_sampling_frame_patients": int(len(frame)),
        "selected_patients": int(len(sample)),
        "sampling": config["sampling"],
        "random_seed": int(config["random_seed"]),
        "future_outcome_labels_loaded_for_sampling": False,
        "future_event_column_loaded": False,
    }
    atomic_json(summary, summary_path)
    log(f"[RAW SAVED] outcome-blind main sample rows={len(sample):,} path={path}")
    return sample, summary


def validate_part(part, row, config):
    required = {
        "person_id",
        "patient_id_dense",
        "sample_order",
        "rollout_index",
        "first_diabetes_day",
        "generated_death_day",
        "generated_followup_end_day",
        "reached_horizon_or_death",
        "max_token_cap_before_horizon",
        "valid_generated_events",
        "requested_batch_size",
        "effective_batch_size",
        "patient_generation_seconds",
        "peak_gpu_memory_gib",
    }
    missing = sorted(required - set(part.columns))
    if missing:
        raise ValueError(f"Raw patient part missing columns: {missing}")
    rollouts = int(config["rollouts_per_patient"])
    if len(part) != rollouts:
        raise ValueError(f"Raw patient part rows={len(part)} expected={rollouts}")
    for column, expected in (
        ("person_id", int(row.person_id)),
        ("patient_id_dense", int(row.patient_id_dense)),
        ("sample_order", int(row.sample_order)),
        ("requested_batch_size", int(config["rollout_batch_size"])),
    ):
        if not part[column].eq(expected).all():
            raise ValueError(f"Raw patient part {column} mismatch")
    indexes = np.sort(part["rollout_index"].to_numpy(dtype=np.int64))
    if not np.array_equal(indexes, np.arange(rollouts, dtype=np.int64)):
        raise ValueError("Raw patient part rollout indexes are incomplete")
    if part.duplicated(["person_id", "rollout_index"]).any():
        raise ValueError("Raw patient part contains duplicate trajectory keys")


def run_or_load_raw(
    args,
    config,
    sample,
    model=None,
    split_data=None,
    targets=None,
    death_tokens=None,
    token_type_lookup=None,
    clinical_mask=None,
):
    parts_dir = args.output_dir / "raw" / "patient_parts"
    rows = list(sample.itertuples(index=False))
    started = time.perf_counter()
    completed_seconds = 0.0
    for patient_number, row in enumerate(rows, start=1):
        part_path = parts_dir / f"patient_{int(row.sample_order):04d}.parquet"
        if part_path.is_file():
            if not args.resume:
                raise RuntimeError(f"Unexpected existing raw part without --resume: {part_path}")
            part = pd.read_parquet(part_path)
            validate_part(part, row, config)
            completed_seconds += float(part["patient_generation_seconds"].iloc[0])
            if patient_number == 1 or patient_number % 25 == 0 or patient_number == len(rows):
                log(
                    f"[RESUME] patient={patient_number}/{len(rows)} raw part validated "
                    f"path={part_path}"
                )
            continue
        if args.summary_only:
            raise FileNotFoundError(f"Summary-only mode missing raw part: {part_path}")

        part = bench.generate_one(
            args,
            config,
            row,
            model,
            split_data,
            targets,
            death_tokens,
            token_type_lookup,
            clinical_mask,
            int(config["rollouts_per_patient"]),
            int(config["rollout_batch_size"]),
            seed_offset=320000,
        )
        part["run_phase"] = "main_population"
        validate_part(part, row, config)
        bench.atomic_to_parquet(part, part_path)
        patient_seconds = float(part["patient_generation_seconds"].iloc[0])
        completed_seconds += patient_seconds
        done = patient_number
        mean_seconds = completed_seconds / done
        eta_hours = mean_seconds * (len(rows) - done) / 3600.0
        progress = {
            "status": "RUNNING_RAW_GENERATION",
            "patients_complete": done,
            "patients_total": len(rows),
            "trajectories_complete": done * int(config["rollouts_per_patient"]),
            "trajectories_total": int(config["total_trajectories"]),
            "last_sample_order": int(row.sample_order),
            "last_person_id": int(row.person_id),
            "last_patient_seconds": patient_seconds,
            "stored_generation_seconds": completed_seconds,
            "estimated_remaining_hours": eta_hours,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(progress, args.output_dir / "progress.json")
        hits = int(part["first_diabetes_day"].notna().sum())
        log(
            f"[RAW SAVED] patient={done}/{len(rows)} rows={len(part)} "
            f"seconds={patient_seconds:,.1f} hits={hits} eta_hours={eta_hours:,.2f} "
            f"path={part_path}"
        )

    parts = []
    for row in rows:
        part_path = parts_dir / f"patient_{int(row.sample_order):04d}.parquet"
        part = pd.read_parquet(part_path)
        validate_part(part, row, config)
        parts.append(part)
    trajectories = pd.concat(parts, ignore_index=True)
    if len(trajectories) != int(config["total_trajectories"]):
        raise RuntimeError(
            f"Combined trajectories={len(trajectories):,} expected={config['total_trajectories']:,}"
        )
    if trajectories.duplicated(["person_id", "rollout_index"]).any():
        raise RuntimeError("Combined raw trajectories contain duplicate keys")
    combined_path = args.output_dir / "raw" / "main_trajectories.parquet"
    bench.atomic_to_parquet(trajectories, combined_path)
    log(f"[RAW SAVED] combined trajectories rows={len(trajectories):,} path={combined_path}")
    atomic_json(
        {
            "status": "RAW_GENERATION_COMPLETE",
            "patients_complete": int(len(sample)),
            "patients_total": int(len(sample)),
            "trajectories_complete": int(len(trajectories)),
            "trajectories_total": int(config["total_trajectories"]),
            "stored_generation_seconds": float(
                trajectories.drop_duplicates("person_id")["patient_generation_seconds"].sum()
            ),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        args.output_dir / "progress.json",
    )
    return trajectories


def kaplan_meier_risk(duration, event, max_day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    valid = np.isfinite(duration) & (duration >= 0)
    duration = np.clip(duration[valid], 0.0, float(max_day))
    event = event[valid]
    event_times = np.unique(duration[event])
    event_times.sort()
    survival_values = []
    survival = 1.0
    for event_time in event_times:
        at_risk = int(np.sum(duration >= event_time))
        events = int(np.sum(event & np.isclose(duration, event_time, rtol=0, atol=1e-7)))
        if at_risk > 0 and events > 0:
            survival *= 1.0 - events / at_risk
        survival_values.append(survival)
    days = np.arange(max_day + 1, dtype=np.float64)
    if not survival_values:
        return np.zeros(max_day + 1, dtype=np.float64)
    index = np.searchsorted(event_times, days, side="right") - 1
    curve = np.zeros(max_day + 1, dtype=np.float64)
    known = index >= 0
    curve[known] = 1.0 - np.asarray(survival_values)[index[known]]
    return curve


def rollout_patient_curves(frame, max_day):
    hit = frame["first_diabetes_day"].to_numpy(dtype=np.float64)
    death = frame["generated_death_day"].to_numpy(dtype=np.float64)
    follow = frame["generated_followup_end_day"].to_numpy(dtype=np.float64)

    event = np.isfinite(hit) & (hit <= max_day)
    censor = np.minimum(
        np.where(np.isfinite(death), death, float(max_day)),
        np.where(np.isfinite(follow), follow, 0.0),
    )
    duration = np.where(event, hit, censor)
    cause_specific = kaplan_meier_risk(duration, event, max_day)

    days = np.arange(max_day + 1, dtype=np.float64)
    competing = np.full(max_day + 1, np.nan, dtype=np.float64)
    usable_counts = np.zeros(max_day + 1, dtype=np.int32)
    for start in range(0, max_day + 1, 128):
        stop = min(start + 128, max_day + 1)
        day = days[start:stop, None]
        hit_known = np.isfinite(hit)[None, :] & (hit[None, :] <= day)
        death_known = np.isfinite(death)[None, :] & (death[None, :] <= day)
        covered = follow[None, :] >= day
        denominator = np.sum(covered | hit_known | death_known, axis=1)
        numerator = np.sum(hit_known, axis=1)
        usable_counts[start:stop] = denominator
        competing[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    return cause_specific, competing, usable_counts


def cox_baseline_at_days(model_path, max_day):
    with np.load(model_path) as saved:
        event_times = saved["event_times"].astype(np.float64)
        cumulative = saved["cumulative_baseline_hazard"].astype(np.float64)
    days = np.arange(max_day + 1, dtype=np.float64)
    indexes = np.searchsorted(event_times, days, side="right") - 1
    baseline = np.zeros(max_day + 1, dtype=np.float64)
    known = indexes >= 0
    baseline[known] = cumulative[indexes[known]]
    return baseline


def cox_risk_matrix(scores, baseline):
    scores = np.asarray(scores, dtype=np.float64)
    relative = np.exp(np.clip(scores, -50, 50))[:, None]
    return -np.expm1(-relative * baseline[None, :])


def monotonic(curve, tolerance=1e-10):
    curve = np.asarray(curve, dtype=np.float64)
    valid = np.isfinite(curve)
    return bool(valid.all() and np.all(np.diff(curve) >= -tolerance))


def attach_evaluation_data(args, sample):
    predictions = pd.read_parquet(args.baseline_dir / "cox_test_predictions.parquet")
    required = {
        "person_id",
        "duration_days",
        "event",
        "clinical_cox_score",
        "clinical_cox_risk_5y",
        "fermat_clinical_cox_score",
        "fermat_clinical_cox_risk_5y",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Cox predictions missing columns: {missing}")
    evaluation = sample.merge(predictions, on="person_id", how="inner", validate="one_to_one")
    if len(evaluation) != len(sample):
        raise RuntimeError("Not every sampled patient has a saved Cox prediction")
    return evaluation.sort_values("sample_order").reset_index(drop=True)


def build_full_cohort_reference(args):
    observed = pd.read_csv(args.baseline_dir / "full_test_observed_diabetes_curve.csv")
    cox = pd.read_csv(args.baseline_dir / "cox_population_curves.csv")
    required_observed = {"day", "observed"}
    required_cox = {
        "day",
        "clinical_cox_mean_predicted",
        "fermat_clinical_cox_mean_predicted",
    }
    if not required_observed.issubset(observed.columns):
        raise ValueError("Full observed curve is missing day/observed")
    if not required_cox.issubset(cox.columns):
        raise ValueError("Full Cox population curves are missing required columns")
    reference = observed.merge(cox, on="day", how="inner", validate="one_to_one")
    return reference


def svg_population_curve(frame, output_path):
    width, height = 1100, 700
    left, right, top, bottom = 95, 40, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    series = [
        ("Observed KM", "observed_km", "#111827", ""),
        ("Clinical Cox", "clinical_cox_mean_predicted", "#2563eb", ""),
        ("FERMAT + Clinical Cox", "fermat_clinical_cox_mean_predicted", "#7c3aed", ""),
        ("FERMAT rollout cause-specific", "fermat_rollout_cause_specific_mean", "#dc2626", ""),
        ("FERMAT rollout competing CIF", "fermat_rollout_competing_cif_mean", "#f59e0b", "6,5"),
    ]
    y_max = max(float(np.nanmax(frame[column])) for _, column, _, _ in series)
    y_max = max(0.01, y_max * 1.12)

    def x(day):
        return left + float(day) / 1826.0 * plot_w

    def y(value):
        return top + plot_h - float(value) / y_max * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="30" font-family="sans-serif" font-size="20" font-weight="700">Task 30 diabetes population risk curves: common 1,000-patient sample</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        yy = y(tick)
        lines.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick * 100:.1f}%</text>')
    for day, label in ((0, "0"), (365, "1y"), (1095, "3y"), (1826, "5y")):
        xx = x(day)
        lines.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{xx:.2f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')
    for name, column, color, dash in series:
        points = " ".join(
            f"{x(day):.2f},{y(value):.2f}"
            for day, value in zip(frame["day"], frame[column])
            if np.isfinite(value)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>' )
    legend_x, legend_y = left + 20, top + 20
    for index, (name, _, color, dash) in enumerate(series):
        yy = legend_y + index * 24
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{legend_x}" y1="{yy}" x2="{legend_x + 36}" y2="{yy}" stroke="{color}" stroke-width="3"{dash_attr}/>' )
        lines.append(f'<text x="{legend_x + 46}" y="{yy + 4}" font-family="sans-serif" font-size="13">{html.escape(name)}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">Time from 2018-01-01 index</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.2f}" transform="rotate(-90 20 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">Cumulative risk</text>')
    lines.append("</svg>")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summaries(args, config, sample, trajectories):
    max_day = int(config["horizon_days"])
    evaluation = attach_evaluation_data(args, sample)
    observed = kaplan_meier_risk(evaluation["duration_days"], evaluation["event"], max_day)

    rollout_cause = []
    rollout_competing = []
    patient_rows = []
    landmarks = [int(day) for day in config["landmark_days"]]
    grouped = trajectories.groupby("person_id", sort=False)
    for row in evaluation.itertuples(index=False):
        part = grouped.get_group(int(row.person_id))
        cause, competing, usable = rollout_patient_curves(part, max_day)
        rollout_cause.append(cause)
        rollout_competing.append(competing)
        summary = {
            "person_id": int(row.person_id),
            "sample_order": int(row.sample_order),
            "duration_days": float(row.duration_days),
            "event": int(row.event),
            "generated_death_fraction": float(part["generated_death_day"].notna().mean()),
            "max_token_cap_before_horizon_fraction": float(
                part["max_token_cap_before_horizon"].astype(bool).mean()
            ),
            "mean_valid_generated_events": float(part["valid_generated_events"].mean()),
            "clinical_cox_risk_5y": float(row.clinical_cox_risk_5y),
            "fermat_clinical_cox_risk_5y": float(row.fermat_clinical_cox_risk_5y),
        }
        for day in landmarks:
            suffix = LANDMARK_NAMES[day]
            summary[f"rollout_cause_specific_risk_{suffix}"] = float(cause[day])
            summary[f"rollout_competing_cif_{suffix}"] = float(competing[day])
            summary[f"usable_rollouts_{suffix}"] = int(usable[day])
        patient_rows.append(summary)
    cause_matrix = np.stack(rollout_cause)
    competing_matrix = np.stack(rollout_competing)

    clinical_baseline = cox_baseline_at_days(
        args.baseline_dir / "models" / "clinical_cox" / "cox_model.npz", max_day
    )
    fermat_baseline = cox_baseline_at_days(
        args.baseline_dir / "models" / "fermat_clinical_cox" / "cox_model.npz", max_day
    )
    clinical_matrix = cox_risk_matrix(evaluation["clinical_cox_score"], clinical_baseline)
    fermat_matrix = cox_risk_matrix(
        evaluation["fermat_clinical_cox_score"], fermat_baseline
    )

    population = pd.DataFrame(
        {
            "day": np.arange(max_day + 1, dtype=np.int32),
            "observed_km": observed,
            "clinical_cox_mean_predicted": clinical_matrix.mean(axis=0),
            "fermat_clinical_cox_mean_predicted": fermat_matrix.mean(axis=0),
            "fermat_rollout_cause_specific_mean": np.nanmean(cause_matrix, axis=0),
            "fermat_rollout_competing_cif_mean": np.nanmean(competing_matrix, axis=0),
        }
    )
    population.to_csv(args.output_dir / "population_curve_comparison.csv", index=False)
    landmark_frame = population.loc[population["day"].isin(landmarks)].copy()
    landmark_frame.insert(1, "landmark", landmark_frame["day"].map(LANDMARK_NAMES))
    landmark_frame.to_csv(args.output_dir / "population_curve_landmarks.csv", index=False)

    patient_summary = pd.DataFrame(patient_rows).sort_values("sample_order")
    bench.atomic_to_parquet(
        patient_summary, args.output_dir / "individual_curve_landmarks.parquet"
    )

    days = np.arange(max_day + 1, dtype=np.int32)
    individual = pd.DataFrame(
        {
            "person_id": np.repeat(evaluation["person_id"].to_numpy(dtype=np.int64), max_day + 1),
            "sample_order": np.repeat(
                evaluation["sample_order"].to_numpy(dtype=np.int32), max_day + 1
            ),
            "day": np.tile(days, len(evaluation)),
            "clinical_cox_risk": clinical_matrix.reshape(-1).astype(np.float32),
            "fermat_clinical_cox_risk": fermat_matrix.reshape(-1).astype(np.float32),
            "fermat_rollout_cause_specific_risk": cause_matrix.reshape(-1).astype(np.float32),
            "fermat_rollout_competing_cif": competing_matrix.reshape(-1).astype(np.float32),
        }
    )
    bench.atomic_to_parquet(individual, args.output_dir / "individual_risk_curves.parquet")

    full_reference = build_full_cohort_reference(args)
    full_reference.to_csv(args.output_dir / "full_cohort_reference_curves.csv", index=False)
    svg_population_curve(population, args.output_dir / "population_curve_comparison.svg")

    curve_columns = [column for column in population.columns if column != "day"]
    checks = {
        "raw_trajectory_rows": int(len(trajectories)),
        "expected_raw_trajectory_rows": int(config["total_trajectories"]),
        "raw_duplicate_keys": int(
            trajectories.duplicated(["person_id", "rollout_index"]).sum()
        ),
        "sample_patients": int(len(evaluation)),
        "sample_observed_events_5y": int(evaluation["event"].sum()),
        "generated_diabetes_hits_5y": int(trajectories["first_diabetes_day"].notna().sum()),
        "generated_deaths_5y": int(trajectories["generated_death_day"].notna().sum()),
        "max_token_cap_before_horizon_fraction": float(
            trajectories["max_token_cap_before_horizon"].astype(bool).mean()
        ),
        "all_population_curves_in_unit_interval": bool(
            all(
                np.nanmin(population[column]) >= -1e-10
                and np.nanmax(population[column]) <= 1 + 1e-10
                for column in curve_columns
            )
        ),
        "all_population_curves_monotonic": bool(
            all(monotonic(population[column]) for column in curve_columns)
        ),
        "individual_curve_rows": int(len(individual)),
        "expected_individual_curve_rows": int(len(evaluation) * (max_day + 1)),
    }
    hard_pass = (
        checks["raw_trajectory_rows"] == checks["expected_raw_trajectory_rows"]
        and checks["raw_duplicate_keys"] == 0
        and checks["sample_patients"] == int(config["patients"])
        and checks["generated_diabetes_hits_5y"] > 0
        and checks["all_population_curves_in_unit_interval"]
        and checks["all_population_curves_monotonic"]
        and checks["individual_curve_rows"] == checks["expected_individual_curve_rows"]
    )
    validation = {"status": "PASS" if hard_pass else "FAIL", **checks}
    atomic_json(validation, args.output_dir / "curve_validation.json")
    return evaluation, population, landmark_frame, patient_summary, validation


def run_self_test():
    duration = np.array([1.0, 2.0, 2.0, 4.0])
    event = np.array([1, 0, 1, 0], dtype=np.int8)
    km = kaplan_meier_risk(duration, event, 5)
    if not monotonic(km) or not np.isclose(km[1], 0.25):
        raise AssertionError(f"Unexpected KM curve: {km}")
    synthetic = pd.DataFrame(
        {
            "first_diabetes_day": [1.0, np.nan, np.nan, 4.0],
            "generated_death_day": [np.nan, 2.0, np.nan, np.nan],
            "generated_followup_end_day": [5.0, 2.0, 3.0, 5.0],
        }
    )
    cause, competing, usable = rollout_patient_curves(synthetic, 5)
    if not monotonic(cause) or not monotonic(competing):
        raise AssertionError("Synthetic rollout curves are not monotonic")
    if usable[5] != 3 or not np.isclose(competing[5], 2.0 / 3.0):
        raise AssertionError("Competing-risk denominator logic failed")
    log("SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.summary_only and not args.resume:
        raise ValueError("--summary-only requires --resume")
    normalize_paths(args)
    config = load_config(args.config_file)
    require_paths(args)
    baseline_audit = bench.validate_baseline_audit(args)
    fingerprint = run_fingerprint(args, config)
    prepare_output(args, fingerprint)
    if not args.summary_only and args.device == "cuda" and not bench.torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    started = datetime.now(timezone.utc)
    endpoint = {
        "status": "MAIN_POPULATION_STAGE",
        "outcome": "diabetes",
        "index_date": config["index_date"],
        "horizon_days": int(config["horizon_days"]),
        "cohort": config["cohort_rule"],
        "baseline_audit": baseline_audit,
        "patients": int(config["patients"]),
        "rollouts_per_patient": int(config["rollouts_per_patient"]),
        "total_trajectories": int(config["total_trajectories"]),
        "sampling_uses_future_outcome": False,
        "primary_estimand": config["primary_estimand"],
        "sensitivity_estimand": config["sensitivity_estimand"],
        "stage_boundary": config["stage_boundary"],
    }
    atomic_json(endpoint, args.output_dir / "endpoint_definition.json")
    sample, sampling_summary = build_or_load_sample(args, config)

    if args.summary_only:
        trajectories = run_or_load_raw(args, config, sample)
        checkpoint = {}
        registry_source = "not_reloaded_summary_only"
    else:
        log("[START] load FERMAT block-2048 checkpoint")
        model, checkpoint = bench.load_model(args.fermat_ckpt, args.device)
        if int(model.config.block_size) != 2048:
            raise RuntimeError(
                f"FERMAT checkpoint block_size={model.config.block_size}; expected 2048"
            )
        registry, registry_source = bench.load_registry(args.data_dir)
        token_type_lookup, _, clinical_mask = bench.registry_maps(
            registry, int(model.config.vocab_size), args.device
        )
        targets = bench.load_target_tokens(
            args, config, registry, int(model.config.vocab_size)
        )
        death_tokens = bench.death_model_tokens(registry, int(model.config.vocab_size))
        if not death_tokens:
            raise RuntimeError("No DTH model tokens were found")
        split_data = bench.load_split_data(args.data_dir, config["split"])
        log(
            f"[START] main population rollout patients={config['patients']} "
            f"rollouts={config['rollouts_per_patient']} batch={config['rollout_batch_size']} "
            f"trajectories={config['total_trajectories']}"
        )
        trajectories = run_or_load_raw(
            args,
            config,
            sample,
            model,
            split_data,
            targets,
            death_tokens,
            token_type_lookup,
            clinical_mask,
        )
        bench.synchronize(args.device)
        del model
        if bench.torch.cuda.is_available():
            bench.torch.cuda.empty_cache()

    log("[START] CPU curve summaries from durable raw trajectories")
    evaluation, population, landmarks, patients, validation = build_summaries(
        args, config, sample, trajectories
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "COMPLETE_TASK30_DIABETES_MAIN_POPULATION_ROLLOUT"
        if validation["status"] == "PASS"
        else "FAIL_TASK30_DIABETES_MAIN_POPULATION_ROLLOUT",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds_this_invocation": (finished - started).total_seconds(),
        "checkpoint": str(args.fermat_ckpt),
        "checkpoint_iter": int(checkpoint.get("iter_num", checkpoint.get("iter", -1))),
        "registry": registry_source,
        "sampling": sampling_summary,
        "validation": validation,
        "perturbation_executed": False,
        "source_baseline_modified": False,
        "outputs": {
            "raw_parts": str(args.output_dir / "raw" / "patient_parts"),
            "raw_combined": str(args.output_dir / "raw" / "main_trajectories.parquet"),
            "population_curves": str(args.output_dir / "population_curve_comparison.csv"),
            "population_svg": str(args.output_dir / "population_curve_comparison.svg"),
            "individual_curves": str(args.output_dir / "individual_risk_curves.parquet"),
            "individual_landmarks": str(args.output_dir / "individual_curve_landmarks.parquet"),
        },
    }
    atomic_json(manifest, args.output_dir / "manifest.json")

    return_summary = [
        "## STATUS",
        manifest["status"],
        "## SAMPLING",
        json.dumps(sampling_summary, ensure_ascii=False),
        "## VALIDATION",
        json.dumps(validation, ensure_ascii=False),
        "## POPULATION_LANDMARKS",
        landmarks.to_csv(index=False).rstrip(),
        "## INDIVIDUAL_OUTPUT",
        f"patients={len(patients):,}",
        str(args.output_dir / "individual_risk_curves.parquet"),
        "## PERTURBATION",
        "NOT_RUN_SELECT_PATIENTS_AND_PERTURBATIONS_AFTER_POPULATION_VALIDATION",
        "## OUTPUT_DIR",
        str(args.output_dir),
    ]
    text = "\n".join(return_summary) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log("[COMPLETE] Task 30 diabetes main population rollout finished")
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
