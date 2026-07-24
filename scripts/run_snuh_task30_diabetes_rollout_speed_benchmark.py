#!/usr/bin/env python3
"""Benchmark Task 30 diabetes rollout speed before choosing the main run size.

The benchmark samples patients without reading future outcome labels. Sampling
is restricted to the exact test-cohort person IDs validated by the saved Cox
baseline audit. Batch sizes 4 and 8 are timed on the same four patients, with
raw patient-level trajectory files saved immediately after each generation.

This is a technical benchmark, not a calibrated risk-curve analysis.
"""

from __future__ import annotations

import argparse
import hashlib
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

SELF_TEST_ONLY = "--self-test" in sys.argv
if SELF_TEST_ONLY:
    torch = None
else:
    import torch

    from scripts.run_snuh_task27_primary_direct_risk_dry_run import (
        load_model,
        load_patient_map,
        load_registry,
        load_split_data,
        registry_maps,
    )
    from scripts.run_snuh_task30_multi_outcome_rollout_pilot import (
        atomic_to_parquet,
        death_model_tokens,
        generate_patient_rollouts,
        load_target_tokens,
        prefix_for_row,
        synchronize,
    )


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_diabetes_rollout_speed_benchmark.json"
DEFAULT_BASELINE_DIR = (
    TASK30 / "outputs" / "diabetes_rollout_stratification_20260715_075746"
)
DEFAULT_BASELINE_AUDIT_DIR = (
    TASK30 / "outputs" / "diabetes_baseline_reuse_audit_20260716_014633"
)
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_LABEL_FILE = DEFAULT_LABEL_DIR / "patient_phenotype_labels_wide_20180101.parquet"
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_rollout_speed_benchmark"
LANDMARKS = {"1y": 365, "3y": 1095, "5y": 1826}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--baseline-audit-dir", type=Path, default=DEFAULT_BASELINE_AUDIT_DIR
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "outcomes",
        "index_date",
        "horizon_days",
        "split",
        "cohort_rule",
        "expected_sampling_frame_patients",
        "future_outcome_labels_loaded_for_sampling",
        "sampling",
        "patients",
        "rollouts_per_patient_per_batch_size",
        "batch_sizes",
        "measured_trajectories",
        "warmup",
        "max_new_tokens",
        "top_k",
        "temperature",
        "same_day_repeat_penalty",
        "same_day_temperature",
        "same_day_probability_cap",
        "random_seed",
        "runtime_basis",
        "benchmark_rationale",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing fields: {missing}")
    if config["outcomes"] != ["diabetes"]:
        raise ValueError("This benchmark requires outcomes=['diabetes']")
    if config["index_date"] != "2018-01-01":
        raise ValueError("This benchmark is fixed to index_date=2018-01-01")
    if int(config["horizon_days"]) != 1826 or config["split"] != "test":
        raise ValueError("This benchmark requires the held-out test split and 1826 days")
    if bool(config["future_outcome_labels_loaded_for_sampling"]):
        raise ValueError("Future outcome labels must not be loaded for sampling")
    patients = int(config["patients"])
    rollouts = int(config["rollouts_per_patient_per_batch_size"])
    batch_sizes = [int(value) for value in config["batch_sizes"]]
    if patients <= 0 or rollouts <= 0 or not batch_sizes:
        raise ValueError("patients, rollouts, and batch_sizes must be positive")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch_sizes must be unique")
    if any(rollouts % batch_size for batch_size in batch_sizes):
        raise ValueError("rollouts must be exactly divisible by every batch size")
    expected = patients * rollouts * len(batch_sizes)
    if int(config["measured_trajectories"]) != expected:
        raise ValueError(
            f"measured_trajectories={config['measured_trajectories']} expected={expected}"
        )
    warmup = config["warmup"]
    if int(warmup["patients"]) != 1 or int(warmup["rollouts"]) <= 0:
        raise ValueError("The warmup must use one patient and at least one rollout")
    return config


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


def registry_path(data_dir):
    for filename in ("token_registry.csv", "vocab.csv"):
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No token_registry.csv or vocab.csv in {data_dir}")


def require_paths(args):
    required = [
        args.config_file,
        args.baseline_audit_dir / "baseline_reuse_audit.json",
        args.baseline_dir / "cox_test_predictions.parquet",
        args.data_dir / "test.bin",
        args.data_dir / "patient_id_map.parquet",
        args.label_dir / "phenotype_group_concept_map.csv",
        args.label_file,
        args.fermat_ckpt,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    registry_path(args.data_dir)


def validate_baseline_audit(args):
    path = args.baseline_audit_dir / "baseline_reuse_audit.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    status = str(result.get("status", ""))
    if not status.startswith("PASS_") or int(result.get("hard_failures", -1)) != 0:
        raise RuntimeError(f"Baseline audit is not reusable: {result}")
    audited_baseline = Path(result["baseline_dir"]).resolve()
    if audited_baseline != args.baseline_dir:
        raise RuntimeError(
            f"Audited baseline mismatch: audit={audited_baseline} arg={args.baseline_dir}"
        )
    return {
        "status": status,
        "hard_failures": 0,
        "audit_json": str(path),
        "baseline_dir": str(args.baseline_dir),
    }


def tracked_file(path, include_hash=False):
    stat = Path(path).stat()
    result = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256_file(path)
    return result


def run_fingerprint(args, config):
    return {
        "config": tracked_file(args.config_file, include_hash=True),
        "baseline_audit": tracked_file(
            args.baseline_audit_dir / "baseline_reuse_audit.json", include_hash=True
        ),
        "cox_test_predictions": tracked_file(
            args.baseline_dir / "cox_test_predictions.parquet"
        ),
        "label_file": tracked_file(args.label_file),
        "patient_map": tracked_file(args.data_dir / "patient_id_map.parquet"),
        "test_bin": tracked_file(args.data_dir / "test.bin"),
        "registry": tracked_file(registry_path(args.data_dir)),
        "fermat_ckpt": tracked_file(args.fermat_ckpt),
        "device": args.device,
        "dtype": args.dtype,
        "patients": int(config["patients"]),
        "rollouts_per_patient_per_batch_size": int(
            config["rollouts_per_patient_per_batch_size"]
        ),
        "batch_sizes": [int(value) for value in config["batch_sizes"]],
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
        (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
        write_json(fingerprint, run_config_path)
        log(f"[RAW SAVED] {run_config_path}")
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)


def load_sampling_frame(args, config):
    baseline_ids = pd.read_parquet(
        args.baseline_dir / "cox_test_predictions.parquet", columns=["person_id"]
    )
    if baseline_ids["person_id"].duplicated().any():
        raise ValueError("Cox test predictions contain duplicate person_id")
    baseline_ids["person_id"] = baseline_ids["person_id"].astype(np.int64)
    expected = int(config["expected_sampling_frame_patients"])
    if len(baseline_ids) != expected:
        raise RuntimeError(f"Validated baseline IDs={len(baseline_ids):,} expected={expected:,}")

    label_columns = [
        "person_id",
        "split",
        "age_at_index",
        "has_pre_index_washout",
        "prior__diabetes",
    ]
    labels = pd.read_parquet(args.label_file, columns=label_columns)
    labels = labels.loc[labels["split"].eq(config["split"])].copy()
    if labels["person_id"].duplicated().any():
        raise ValueError("Test labels contain duplicate person_id")
    frame = baseline_ids.merge(labels, on="person_id", how="inner", validate="one_to_one")
    if len(frame) != expected:
        raise RuntimeError(f"Baseline/label sampling frame={len(frame):,} expected={expected:,}")
    if not frame["has_pre_index_washout"].fillna(False).astype(bool).all():
        raise RuntimeError("Validated baseline frame contains a patient without washout")
    if frame["prior__diabetes"].fillna(0).astype(bool).any():
        raise RuntimeError("Validated baseline frame contains prior diabetes")

    patient_map = load_patient_map(args.data_dir)
    patient_map = patient_map.loc[patient_map["split"].eq(config["split"])].copy()
    if patient_map["person_id"].duplicated().any():
        raise ValueError("Test patient map contains duplicate person_id")
    frame = frame.merge(
        patient_map[["person_id", "patient_id_dense"]],
        on="person_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != expected:
        raise RuntimeError(f"Mapped sampling frame={len(frame):,} expected={expected:,}")
    frame["index_age_days"] = np.floor(
        pd.to_numeric(frame["age_at_index"], errors="coerce") * 365.25
    ).astype(np.int64)
    frame = frame.sort_values("person_id").reset_index(drop=True)
    return frame


def build_or_load_sample(args, config):
    path = args.output_dir / "raw" / "benchmark_sample_cohort.parquet"
    summary_path = args.output_dir / "sampling_frame_summary.json"
    if args.resume and path.is_file() and summary_path.is_file():
        sample = pd.read_parquet(path)
        if len(sample) != int(config["patients"]):
            raise ValueError("Saved benchmark sample row count is invalid")
        log(f"[RESUME] sample rows={len(sample):,}")
        return sample, json.loads(summary_path.read_text(encoding="utf-8"))

    frame = load_sampling_frame(args, config)
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
    atomic_to_parquet(sample, path)
    summary = {
        "validated_baseline_sampling_frame_patients": int(len(frame)),
        "selected_patients": int(len(sample)),
        "sampling": config["sampling"],
        "random_seed": int(config["random_seed"]),
        "future_outcome_labels_loaded_for_sampling": False,
        "future_event_column_loaded": False,
    }
    write_json(summary, summary_path)
    log(f"[RAW SAVED] outcome-blind benchmark sample rows={len(sample):,} path={path}")
    return sample, summary


def validate_part(part, row, rollouts, requested_batch_size):
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
        raise ValueError(f"Raw benchmark part missing columns: {missing}")
    if len(part) != int(rollouts):
        raise ValueError(f"Raw benchmark part rows={len(part)} expected={rollouts}")
    if not part["person_id"].eq(int(row.person_id)).all():
        raise ValueError("Raw benchmark part person_id mismatch")
    if not part["requested_batch_size"].eq(int(requested_batch_size)).all():
        raise ValueError("Raw benchmark part requested batch size mismatch")
    if not np.array_equal(
        np.sort(part["rollout_index"].to_numpy(dtype=np.int64)),
        np.arange(int(rollouts), dtype=np.int64),
    ):
        raise ValueError("Raw benchmark part rollout indexes are incomplete")


def generate_one(
    args,
    config,
    row,
    model,
    split_data,
    targets,
    death_tokens,
    token_type_lookup,
    clinical_mask,
    rollouts,
    batch_size,
    seed_offset,
):
    prefix = prefix_for_row(split_data, row, int(model.config.block_size))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(args.device)
    started = time.perf_counter()
    frame, effective_batch_size = generate_patient_rollouts(
        model,
        prefix,
        targets,
        death_tokens,
        token_type_lookup,
        clinical_mask,
        int(row.index_age_days),
        config,
        args.device,
        args.dtype,
        int(row.sample_order),
        int(rollouts),
        int(batch_size),
        seed_offset=int(seed_offset),
        allow_oom_fallback=True,
    )
    synchronize(args.device)
    seconds = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0
    frame.insert(0, "person_id", int(row.person_id))
    frame.insert(1, "patient_id_dense", int(row.patient_id_dense))
    frame.insert(2, "sample_order", int(row.sample_order))
    frame["requested_batch_size"] = int(batch_size)
    frame["effective_batch_size"] = int(effective_batch_size)
    frame["patient_generation_seconds"] = float(seconds)
    frame["peak_gpu_memory_gib"] = peak_bytes / (1024**3)
    return frame


def run_warmup(
    args,
    config,
    sample,
    model,
    split_data,
    targets,
    death_tokens,
    token_type_lookup,
    clinical_mask,
):
    warmup = config["warmup"]
    path = args.output_dir / "raw" / "warmup_trajectories.parquet"
    row = next(sample.head(1).itertuples(index=False))
    if args.resume and path.is_file():
        frame = pd.read_parquet(path)
        validate_part(frame, row, int(warmup["rollouts"]), int(warmup["batch_size"]))
        log(f"[RESUME] warmup rows={len(frame):,}")
        return frame
    frame = generate_one(
        args,
        config,
        row,
        model,
        split_data,
        targets,
        death_tokens,
        token_type_lookup,
        clinical_mask,
        int(warmup["rollouts"]),
        int(warmup["batch_size"]),
        seed_offset=900000,
    )
    frame["benchmark_phase"] = "warmup_excluded"
    atomic_to_parquet(frame, path)
    log(f"[RAW SAVED] warmup rows={len(frame):,} path={path}")
    return frame


def run_measured_benchmark(
    args,
    config,
    sample,
    model,
    split_data,
    targets,
    death_tokens,
    token_type_lookup,
    clinical_mask,
):
    frames = []
    rollouts = int(config["rollouts_per_patient_per_batch_size"])
    for batch_size in [int(value) for value in config["batch_sizes"]]:
        batch_dir = args.output_dir / "raw" / f"batch_size_{batch_size}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for patient_number, row in enumerate(sample.itertuples(index=False), start=1):
            part_path = batch_dir / f"patient_{int(row.sample_order):03d}.parquet"
            if args.resume and part_path.is_file():
                part = pd.read_parquet(part_path)
                validate_part(part, row, rollouts, batch_size)
                log(
                    f"[RESUME] batch={batch_size} patient={patient_number}/{len(sample)} "
                    f"rows={len(part):,}"
                )
            else:
                part = generate_one(
                    args,
                    config,
                    row,
                    model,
                    split_data,
                    targets,
                    death_tokens,
                    token_type_lookup,
                    clinical_mask,
                    rollouts,
                    batch_size,
                    seed_offset=batch_size * 100000,
                )
                part["benchmark_phase"] = "measured"
                validate_part(part, row, rollouts, batch_size)
                atomic_to_parquet(part, part_path)
                hits = int(part["first_diabetes_day"].notna().sum())
                log(
                    f"[RAW SAVED] batch={batch_size} "
                    f"patient={patient_number}/{len(sample)} rows={len(part):,} "
                    f"seconds={part['patient_generation_seconds'].iloc[0]:,.1f} "
                    f"hits={hits} path={part_path}"
                )
            frames.append(part)
        batch_frame = pd.concat(
            [frame for frame in frames if int(frame["requested_batch_size"].iloc[0]) == batch_size],
            ignore_index=True,
        )
        atomic_to_parquet(
            batch_frame,
            args.output_dir / "raw" / f"batch_size_{batch_size}_trajectories.parquet",
        )
    trajectories = pd.concat(frames, ignore_index=True)
    atomic_to_parquet(
        trajectories, args.output_dir / "raw" / "benchmark_trajectories.parquet"
    )
    log(
        f"[RAW SAVED] combined measured trajectories rows={len(trajectories):,} "
        f"path={args.output_dir / 'raw' / 'benchmark_trajectories.parquet'}"
    )
    return trajectories


def runtime_summary(trajectories):
    patient_runs = trajectories.drop_duplicates(
        ["requested_batch_size", "person_id"]
    )
    rows = []
    for batch_size, group in trajectories.groupby("requested_batch_size", sort=True):
        runs = patient_runs.loc[patient_runs["requested_batch_size"].eq(batch_size)]
        seconds = float(runs["patient_generation_seconds"].sum())
        rows.append(
            {
                "requested_batch_size": int(batch_size),
                "effective_batch_sizes": ",".join(
                    str(value)
                    for value in sorted(group["effective_batch_size"].astype(int).unique())
                ),
                "oom_fallback_occurred": bool(
                    (group["effective_batch_size"].astype(int) < int(batch_size)).any()
                ),
                "patients": int(group["person_id"].nunique()),
                "trajectories": int(len(group)),
                "stored_generation_seconds": seconds,
                "seconds_per_patient": seconds / group["person_id"].nunique(),
                "seconds_per_trajectory": seconds / len(group),
                "trajectories_per_second": len(group) / seconds,
                "max_peak_gpu_memory_gib": float(runs["peak_gpu_memory_gib"].max()),
                "mean_valid_generated_events": float(
                    group["valid_generated_events"].mean()
                ),
                "max_token_cap_before_horizon_fraction": float(
                    group["max_token_cap_before_horizon"].astype(bool).mean()
                ),
                "reached_horizon_or_death_fraction": float(
                    group["reached_horizon_or_death"].astype(bool).mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    base = float(
        result.loc[
            result["requested_batch_size"].eq(4), "trajectories_per_second"
        ].iloc[0]
    )
    result["throughput_relative_to_batch4"] = result["trajectories_per_second"] / base
    return result


def hit_rate_summary(trajectories):
    groups = [
        (str(int(batch_size)), group)
        for batch_size, group in trajectories.groupby("requested_batch_size", sort=True)
    ]
    groups.append(("pooled", trajectories))
    rows = []
    for batch_label, group in groups:
        hit_day = group["first_diabetes_day"].to_numpy(dtype=np.float64)
        death_day = group["generated_death_day"].to_numpy(dtype=np.float64)
        followup = group["generated_followup_end_day"].to_numpy(dtype=np.float64)
        for horizon, day in LANDMARKS.items():
            hit = np.isfinite(hit_day) & (hit_day <= day)
            usable = (
                (followup >= day)
                | hit
                | (np.isfinite(death_day) & (death_day <= day))
            )
            patient_hits = (
                pd.DataFrame({"person_id": group["person_id"].to_numpy(), "hit": hit})
                .groupby("person_id")["hit"]
                .any()
            )
            rows.append(
                {
                    "requested_batch_size": batch_label,
                    "horizon": horizon,
                    "horizon_days": day,
                    "patients": int(group["person_id"].nunique()),
                    "trajectories": int(len(group)),
                    "usable_trajectories": int(usable.sum()),
                    "diabetes_hits": int(hit.sum()),
                    "trajectory_hit_rate_among_usable": (
                        float(hit.sum() / usable.sum()) if usable.any() else np.nan
                    ),
                    "patients_with_at_least_one_hit": int(patient_hits.sum()),
                    "claim_boundary": "technical hit-rate signal; not calibration",
                }
            )
    return pd.DataFrame(rows)


def validate_benchmark(trajectories, sample, config):
    expected = int(config["measured_trajectories"])
    batch_sizes = sorted(int(value) for value in config["batch_sizes"])
    duplicates = int(
        trajectories.duplicated(
            ["requested_batch_size", "person_id", "rollout_index"]
        ).sum()
    )
    checks = {
        "status": "PASS",
        "sampling_frame_patients": int(config["expected_sampling_frame_patients"]),
        "sample_patients": int(len(sample)),
        "expected_measured_trajectories": expected,
        "actual_measured_trajectories": int(len(trajectories)),
        "requested_batch_sizes": batch_sizes,
        "observed_requested_batch_sizes": sorted(
            trajectories["requested_batch_size"].astype(int).unique().tolist()
        ),
        "duplicate_trajectory_keys": duplicates,
        "future_outcome_labels_loaded_for_sampling": False,
        "nonpositive_generation_seconds": int(
            (trajectories["patient_generation_seconds"] <= 0).sum()
        ),
    }
    if (
        len(trajectories) != expected
        or checks["observed_requested_batch_sizes"] != batch_sizes
        or duplicates
        or checks["nonpositive_generation_seconds"]
    ):
        checks["status"] = "FAIL"
    return checks


def run_self_test():
    config = load_config(
        ROOT / "config" / "snuh_task30_diabetes_rollout_speed_benchmark.json"
    )
    if int(config["measured_trajectories"]) != 128:
        raise AssertionError("Benchmark config validation failed")
    rows = []
    for batch_size in (4, 8):
        for person_id in (1, 2):
            for rollout_index in range(2):
                rows.append(
                    {
                        "requested_batch_size": batch_size,
                        "effective_batch_size": batch_size,
                        "person_id": person_id,
                        "rollout_index": rollout_index,
                        "first_diabetes_day": 100.0
                        if person_id == 1 and rollout_index == 0
                        else np.nan,
                        "generated_death_day": np.nan,
                        "generated_followup_end_day": 1826.0,
                        "max_token_cap_before_horizon": False,
                        "reached_horizon_or_death": True,
                        "valid_generated_events": 10,
                        "patient_generation_seconds": float(batch_size),
                        "peak_gpu_memory_gib": 1.0,
                    }
                )
    trajectories = pd.DataFrame(rows)
    runtime = runtime_summary(trajectories)
    hits = hit_rate_summary(trajectories)
    pooled_5y = hits.loc[
        hits["requested_batch_size"].eq("pooled") & hits["horizon"].eq("5y")
    ].iloc[0]
    if len(runtime) != 2 or int(pooled_5y["diabetes_hits"]) != 2:
        raise AssertionError("Benchmark summary self-test failed")
    log("[SELF-TEST PASS] runtime and diabetes hit-rate summaries")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    require_paths(args)
    config = load_config(args.config_file)
    baseline_audit = validate_baseline_audit(args)
    fingerprint = run_fingerprint(args, config)
    prepare_output(args, fingerprint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    started = datetime.now(timezone.utc)
    endpoint = {
        "status": "technical_benchmark_not_calibration",
        "outcome": "diabetes",
        "index_date": config["index_date"],
        "horizon_days": int(config["horizon_days"]),
        "cohort": config["cohort_rule"],
        "baseline_audit": baseline_audit,
        "patients": int(config["patients"]),
        "rollouts_per_patient_per_batch_size": int(
            config["rollouts_per_patient_per_batch_size"]
        ),
        "batch_sizes": config["batch_sizes"],
        "measured_trajectories": int(config["measured_trajectories"]),
        "future_outcome_labels_loaded_for_sampling": False,
        "claim_boundary": config["benchmark_rationale"]["claim_boundary"],
    }
    write_json(endpoint, args.output_dir / "endpoint_definition.json")

    sample, sampling_summary = build_or_load_sample(args, config)
    log("[START] load FERMAT block-2048 checkpoint")
    model, checkpoint = load_model(args.fermat_ckpt, args.device)
    if int(model.config.block_size) != 2048:
        raise RuntimeError(
            f"FERMAT checkpoint block_size={model.config.block_size}; expected 2048"
        )
    registry, registry_source = load_registry(args.data_dir)
    token_type_lookup, _, clinical_mask = registry_maps(
        registry, int(model.config.vocab_size), args.device
    )
    targets = load_target_tokens(args, config, registry, int(model.config.vocab_size))
    death_tokens = death_model_tokens(registry, int(model.config.vocab_size))
    if not death_tokens:
        raise RuntimeError("No DTH model tokens were found")
    split_data = load_split_data(args.data_dir, config["split"])

    run_warmup(
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
    log(
        f"[START] measured benchmark patients={config['patients']} "
        f"rollouts_per_batch_size={config['rollouts_per_patient_per_batch_size']} "
        f"batch_sizes={config['batch_sizes']} "
        f"trajectories={config['measured_trajectories']}"
    )
    trajectories = run_measured_benchmark(
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
    synchronize(args.device)

    runtime = runtime_summary(trajectories)
    hit_rates = hit_rate_summary(trajectories)
    validation = validate_benchmark(trajectories, sample, config)
    runtime.to_csv(args.output_dir / "batch_size_runtime.csv", index=False)
    hit_rates.to_csv(args.output_dir / "diabetes_rollout_hit_rate.csv", index=False)
    write_json(validation, args.output_dir / "benchmark_validation.json")
    diagnostics = {
        "max_token_cap_before_horizon_fraction": float(
            trajectories["max_token_cap_before_horizon"].astype(bool).mean()
        ),
        "reached_horizon_or_death_fraction": float(
            trajectories["reached_horizon_or_death"].astype(bool).mean()
        ),
        "generated_death_fraction": float(
            trajectories["generated_death_day"].notna().mean()
        ),
        "mean_valid_generated_events": float(
            trajectories["valid_generated_events"].mean()
        ),
        "multiple_patient_concurrency_tested": False,
        "multiple_patient_concurrency_reason": (
            "Current generation API batches futures from one identical patient prefix; "
            "different patient prefixes and absolute horizon ages are processed sequentially."
        ),
        "generation_batch_semantics": (
            "Batch size is parallel futures for one patient, not different patients."
        ),
    }
    write_json(diagnostics, args.output_dir / "generation_diagnostics.json")

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "COMPLETE_DIABETES_ROLLOUT_SPEED_BENCHMARK"
        if validation["status"] == "PASS"
        else "FAIL_DIABETES_ROLLOUT_SPEED_BENCHMARK",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds_total": (finished - started).total_seconds(),
        "checkpoint": str(args.fermat_ckpt),
        "checkpoint_iter": int(checkpoint.get("iter_num", checkpoint.get("iter", -1))),
        "checkpoint_block_size": int(model.config.block_size),
        "registry": registry_source,
        "sampling": sampling_summary,
        "benchmark_validation": validation,
        "generation_diagnostics": diagnostics,
        "source_baseline_modified": False,
    }
    write_json(manifest, args.output_dir / "manifest.json")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return_summary = [
        "## STATUS",
        manifest["status"],
        "## BASELINE_AUDIT",
        json.dumps(baseline_audit, ensure_ascii=False),
        "## SAMPLING",
        json.dumps(sampling_summary, ensure_ascii=False),
        "## BATCH_SIZE_RUNTIME",
        runtime.to_csv(index=False).rstrip(),
        "## DIABETES_ROLLOUT_HIT_RATE",
        hit_rates.to_csv(index=False).rstrip(),
        "## GENERATION_DIAGNOSTICS",
        json.dumps(diagnostics, ensure_ascii=False),
        "## BENCHMARK_VALIDATION",
        json.dumps(validation, ensure_ascii=False),
        "## CLAIM_BOUNDARY",
        config["benchmark_rationale"]["claim_boundary"],
        "## OUTPUT_DIR",
        str(args.output_dir),
    ]
    text = "\n".join(return_summary) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log("[COMPLETE] Task 30 diabetes rollout speed benchmark finished")
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
