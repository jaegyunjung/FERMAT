#!/usr/bin/env python3
"""Run the Task 30 shared-rollout multi-outcome feasibility pilot.

This runner generates one set of future trajectories for an outcome-blind
random sample from the common diabetes/CKD/dyslipidemia at-risk test cohort.
Each generated future is scored for all three outcomes.  It is a runtime and
hit-rate pilot, not a calibrated risk-curve analysis.

Raw trajectory parts are written before any summary.  Resume skips only parts
that pass row-count, patient, rollout-index, and duplicate-key validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    # Summary logic is deliberately testable on a CPU host without PyTorch.
    torch = None
    TokenType = None
else:
    import torch

    from model import TokenType
    from scripts.run_snuh_task27_primary_direct_risk_dry_run import (
        collate_prefix,
        dtype_context,
        load_model,
        load_patient_map,
        load_phenotype_token_map,
        load_registry,
        load_split_data,
        registry_maps,
        registry_type,
        rows_before_index,
    )


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK_DIR = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK_DIR / "config" / "snuh_task30_multi_outcome_rollout_pilot.json"
DEFAULT_AUDIT_DIR = (
    TASK_DIR / "outputs" / "outcome_feasibility_audit_20260715_193937"
)
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_LABEL_FILE = DEFAULT_LABEL_DIR / "patient_phenotype_labels_wide_20180101.parquet"
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = TASK_DIR / "outputs" / "multi_outcome_rollout_pilot"
LANDMARKS = {"1y": 365, "3y": 1095, "5y": 1826}
REQUIRED_OUTCOMES = ["diabetes", "chronic_kidney_disease", "dyslipidemia"]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-batch-benchmark", action="store_true")
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


def atomic_to_parquet(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "outcomes",
        "index_date",
        "horizon_days",
        "split",
        "cohort_rule",
        "sampling",
        "future_outcome_labels_used_for_sampling",
        "patients",
        "rollouts_per_patient",
        "total_trajectories",
        "patient_chunk_size",
        "rollout_batch_size",
        "max_new_tokens",
        "top_k",
        "temperature",
        "same_day_repeat_penalty",
        "same_day_temperature",
        "same_day_probability_cap",
        "random_seed",
        "batch_size_benchmark",
        "runtime_basis",
        "monte_carlo_reference",
        "claim_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing fields: {missing}")
    if config["outcomes"] != REQUIRED_OUTCOMES:
        raise ValueError(
            f"Expected outcomes in fixed order {REQUIRED_OUTCOMES}; got {config['outcomes']}"
        )
    if bool(config["future_outcome_labels_used_for_sampling"]):
        raise ValueError("Future outcome labels must not be used for sampling")
    expected = int(config["patients"]) * int(config["rollouts_per_patient"])
    if expected != int(config["total_trajectories"]):
        raise ValueError(
            f"total_trajectories mismatch: {config['total_trajectories']} != {expected}"
        )
    if int(config["horizon_days"]) != LANDMARKS["5y"]:
        raise ValueError("This pilot requires horizon_days=1826")
    if str(config["split"]) != "test":
        raise ValueError("This pilot is fixed to the held-out test split")
    for field in (
        "patients",
        "rollouts_per_patient",
        "patient_chunk_size",
        "rollout_batch_size",
        "max_new_tokens",
    ):
        if int(config[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    return config


def normalize_paths(args):
    for name in (
        "config_file",
        "audit_dir",
        "data_dir",
        "label_dir",
        "label_file",
        "embedding_file",
        "fermat_ckpt",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def require_paths(args):
    required = [
        args.config_file,
        args.audit_dir / "outcome_feasibility_summary.csv",
        args.audit_dir / "source_label_prior_consistency.csv",
        args.data_dir / "test.bin",
        args.data_dir / "patient_id_map.parquet",
        args.data_dir / "token_registry.csv",
        args.label_dir / "phenotype_group_concept_map.csv",
        args.label_file,
        args.embedding_file,
        args.fermat_ckpt,
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def run_fingerprint(args, config):
    tracked_inputs = {
        "config_file": args.config_file,
        "audit_summary": args.audit_dir / "outcome_feasibility_summary.csv",
        "audit_consistency": args.audit_dir / "source_label_prior_consistency.csv",
        "test_bin": args.data_dir / "test.bin",
        "patient_map": args.data_dir / "patient_id_map.parquet",
        "token_registry": args.data_dir / "token_registry.csv",
        "phenotype_concept_map": args.label_dir / "phenotype_group_concept_map.csv",
        "label_file": args.label_file,
        "embedding_file": args.embedding_file,
        "fermat_ckpt": args.fermat_ckpt,
    }
    input_files = {}
    for name, path in tracked_inputs.items():
        stat = path.stat()
        input_files[name] = {
            "path": str(path),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    input_files["config_file"]["sha256"] = sha256_file(args.config_file)
    input_files["audit_summary"]["sha256"] = sha256_file(
        tracked_inputs["audit_summary"]
    )
    input_files["audit_consistency"]["sha256"] = sha256_file(
        tracked_inputs["audit_consistency"]
    )
    return {
        "audit_dir": str(args.audit_dir),
        "data_dir": str(args.data_dir),
        "label_dir": str(args.label_dir),
        "input_files": input_files,
        "device": args.device,
        "dtype": args.dtype,
        "skip_batch_benchmark": bool(args.skip_batch_benchmark),
        "patients": int(config["patients"]),
        "rollouts_per_patient": int(config["rollouts_per_patient"]),
        "total_trajectories": int(config["total_trajectories"]),
    }


def prepare_output(args, fingerprint):
    config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"{args.output_dir} exists and is not empty; use a new directory or --resume"
            )
        if not config_path.is_file():
            raise RuntimeError(f"Cannot resume without {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise RuntimeError("Resume configuration does not match run_config.json")
        log(f"[RESUME VALIDATED] {config_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "raw" / "rollout_parts").mkdir(parents=True, exist_ok=True)
        write_json(fingerprint, config_path)
        log(f"[RAW SAVED] {config_path}")
    (args.output_dir / "raw" / "rollout_parts").mkdir(parents=True, exist_ok=True)


def truthy(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_audit(args, config):
    summary = pd.read_csv(args.audit_dir / "outcome_feasibility_summary.csv")
    consistency = pd.read_csv(args.audit_dir / "source_label_prior_consistency.csv")
    if int(consistency["prior_flag_mismatches"].sum()) != 0:
        raise RuntimeError("Outcome audit has source/label prior mismatches")
    rows = []
    for outcome in config["outcomes"]:
        match = summary.loc[summary["outcome_name"].eq(outcome)]
        if len(match) != 1:
            raise RuntimeError(f"Audit must contain exactly one row for {outcome}")
        row = match.iloc[0]
        if row["definition_status"] != "resolved":
            raise RuntimeError(f"Outcome is not resolved in audit: {outcome}")
        if truthy(row["etl_spike_flag"]):
            raise RuntimeError(f"Outcome has ETL spike flag: {outcome}")
        capture = float(row["test_exact_etl_token_capture_5y"])
        if not np.isfinite(capture) or capture < 0.99:
            raise RuntimeError(f"Outcome token capture is below 0.99: {outcome}={capture}")
        rows.append(
            {
                "outcome": outcome,
                "test_events_1y": int(row["test_events_1y"]),
                "test_events_3y": int(row["test_events_3y"]),
                "test_events_5y": int(row["test_events_5y"]),
                "audit_full_test_cumulative_incidence_5y": float(
                    row["test_cumulative_incidence_5y"]
                ),
                "etl_token_capture_5y": capture,
                "etl_spike_flag": False,
                "model_input_coverage_note": (
                    "Audit model-input coverage is structural and is not used as an outcome exclusion."
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "audit_precheck.csv", index=False)
    log("[AUDIT PRECHECK PASS]\n" + result.to_string(index=False))
    return result


def load_common_cohort(args, config):
    columns = [
        "person_id",
        "split",
        "age_at_index",
        "has_pre_index_washout",
    ] + [f"prior__{outcome}" for outcome in config["outcomes"]]
    labels = pd.read_parquet(args.label_file, columns=columns)
    labels = labels.loc[labels["split"].eq(config["split"])].copy()
    counts = {"test_label_rows": int(len(labels))}
    eligible = labels["has_pre_index_washout"].fillna(False).astype(bool)
    for outcome in config["outcomes"]:
        eligible &= ~labels[f"prior__{outcome}"].fillna(0).astype(bool)
    labels = labels.loc[eligible].copy()
    counts["test_no_prior_all_outcomes_with_washout"] = int(len(labels))

    embeddings = pd.read_parquet(
        args.embedding_file,
        columns=["person_id", "split", "has_embedding_sequence"],
    )
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)]
    embeddings = embeddings[["person_id", "split"]].drop_duplicates()
    patient_map = load_patient_map(args.data_dir)
    patient_map = patient_map.loc[patient_map["split"].eq(config["split"])]
    cohort = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    counts["after_embedding_sequence_intersection"] = int(len(cohort))
    cohort = cohort.merge(patient_map, on=["person_id", "split"], how="inner")
    counts["common_modelable_sampling_frame"] = int(len(cohort))
    cohort["index_age_days"] = np.floor(
        pd.to_numeric(cohort["age_at_index"], errors="coerce") * 365.25
    ).astype(np.int64)
    cohort = cohort.sort_values("person_id").reset_index(drop=True)
    if len(cohort) < int(config["patients"]):
        raise RuntimeError(
            f"Common eligible modelable cohort has {len(cohort):,} patients; "
            f"need {config['patients']:,}"
        )
    return cohort, counts


def validate_sample(sample, config):
    required = {
        "person_id",
        "split",
        "patient_id_dense",
        "index_age_days",
        "sample_order",
        "has_pre_index_washout",
    } | {
        f"prior__{outcome}" for outcome in config["outcomes"]
    }
    missing = sorted(required - set(sample.columns))
    if missing:
        raise ValueError(f"Sample missing columns: {missing}")
    if len(sample) != int(config["patients"]):
        raise ValueError(f"Sample rows={len(sample)} expected={config['patients']}")
    if sample["person_id"].duplicated().any():
        raise ValueError("Sample has duplicate person_id")
    if not sample["split"].eq(config["split"]).all():
        raise ValueError("Sample contains a non-test row")
    if not sample["has_pre_index_washout"].fillna(False).astype(bool).all():
        raise ValueError("Sample contains a row without pre-index washout")
    for outcome in config["outcomes"]:
        if sample[f"prior__{outcome}"].fillna(0).astype(bool).any():
            raise ValueError(f"Sample contains prior outcome: {outcome}")
    if sample[["person_id", "patient_id_dense", "index_age_days"]].isna().any().any():
        raise ValueError("Sample has missing patient or index fields")
    expected_order = np.arange(len(sample), dtype=np.int64)
    if not np.array_equal(sample["sample_order"].to_numpy(dtype=np.int64), expected_order):
        raise ValueError("Sample order is not contiguous")


def build_or_load_sample(args, config):
    path = args.output_dir / "raw" / "pilot_sample_cohort.parquet"
    summary_path = args.output_dir / "raw" / "sampling_frame_summary.json"
    if args.resume and path.is_file():
        sample = pd.read_parquet(path)
        validate_sample(sample, config)
        if not summary_path.is_file():
            raise RuntimeError(f"Cannot resume without {summary_path}")
        sampling_frame = json.loads(summary_path.read_text(encoding="utf-8"))
        log(f"[RESUME] sample cohort rows={len(sample):,}")
        return sample, sampling_frame
    cohort, counts = load_common_cohort(args, config)
    sample = cohort.sample(
        n=int(config["patients"]),
        replace=False,
        random_state=int(config["random_seed"]),
    ).reset_index(drop=True)
    sample["sample_order"] = np.arange(len(sample), dtype=np.int32)
    keep = [
        "person_id",
        "split",
        "patient_id_dense",
        "index_age_days",
        "sample_order",
        "has_pre_index_washout",
    ] + [f"prior__{outcome}" for outcome in config["outcomes"]]
    sample = sample[keep]
    validate_sample(sample, config)
    sampling_frame = {
        **counts,
        "sampling": config["sampling"],
        "random_seed": int(config["random_seed"]),
        "selected_patients": int(len(sample)),
        "future_outcome_labels_used_for_sampling": False,
    }
    write_json(sampling_frame, summary_path)
    log(f"[RAW SAVED] {summary_path}")
    atomic_to_parquet(sample, path)
    log(
        f"[RAW SAVED] outcome-blind common-cohort sample rows={len(sample):,} path={path}"
    )
    return sample, sampling_frame


def death_model_tokens(registry, vocab_size):
    tokens = []
    for row in registry:
        model_token = int(row["token_id"]) + 1
        if model_token < vocab_size and registry_type(row) == int(TokenType.DTH):
            tokens.append(model_token)
    return sorted(set(tokens))


def load_target_tokens(args, config, registry, vocab_size):
    stored, mapping, concept_path = load_phenotype_token_map(
        args.label_dir, registry, config["outcomes"]
    )
    targets = {
        outcome: {int(token) + 1 for token in tokens}
        for outcome, tokens in stored.items()
    }
    for outcome, tokens in targets.items():
        invalid = sorted(token for token in tokens if token >= vocab_size)
        if invalid:
            raise RuntimeError(f"Target tokens outside model vocab for {outcome}: {invalid}")
    mapping["model_token_ids"] = mapping["token_ids"].map(
        lambda text: ",".join(str(int(value) + 1) for value in str(text).split(",") if value)
    )
    mapping.to_csv(args.output_dir / "outcome_token_mapping.csv", index=False)
    log(f"target_concept_map={concept_path}")
    return targets


def synchronize(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_patient_rollouts(
    model,
    prefix,
    targets,
    death_tokens,
    token_type_lookup,
    clinical_mask,
    index_age_days,
    config,
    device,
    dtype,
    person_seed,
    rollouts,
    batch_size,
    seed_offset=0,
    allow_oom_fallback=True,
):
    idx0, age0, type0 = collate_prefix(prefix, device)
    max_day = int(config["horizon_days"])
    horizon_age = float(index_age_days + max_day)
    rows = []
    rollout_index = 0
    effective_batch_size = max(1, int(batch_size))
    while rollout_index < int(rollouts):
        current = min(effective_batch_size, int(rollouts) - rollout_index)
        seed = int(
            config["random_seed"]
            + seed_offset
            + int(person_seed) * 1009
            + rollout_index
        )
        torch.manual_seed(seed)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            with torch.no_grad(), dtype_context(device, dtype):
                idx_out, age_out, type_out, _ = model.generate(
                    idx0.repeat(current, 1),
                    age0.repeat(current, 1),
                    type0.repeat(current, 1),
                    max_new_tokens=int(config["max_new_tokens"]),
                    max_age=horizon_age,
                    no_repeat=False,
                    termination_tokens=death_tokens,
                    token_type_lookup=token_type_lookup,
                    top_k=int(config["top_k"]),
                    temperature=float(config["temperature"]),
                    allowed_token_mask=clinical_mask,
                    same_day_no_repeat=False,
                    same_day_repeat_penalty=float(config["same_day_repeat_penalty"]),
                    same_day_temperature=float(config["same_day_temperature"]),
                    same_day_prob_cap=float(config["same_day_probability_cap"]),
                    return_final_logits=False,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not allow_oom_fallback or current <= 1:
                raise
            effective_batch_size = max(1, current // 2)
            log(f"[OOM RETRY] rollout_batch_size={effective_batch_size}")
            continue

        prefix_len = idx0.shape[1]
        generated_idx = idx_out[:, prefix_len:].detach().cpu().numpy().astype(np.int64)
        generated_age = age_out[:, prefix_len:].detach().cpu().numpy().astype(np.float64)
        generated_type = type_out[:, prefix_len:].detach().cpu().numpy().astype(np.int64)
        for batch_row in range(current):
            token = generated_idx[batch_row]
            age = generated_age[batch_row]
            token_type = generated_type[batch_row]
            valid = (
                (token != 0)
                & (token_type != int(TokenType.PAD))
                & (age >= float(index_age_days) - 1e-5)
                & (age <= horizon_age + 1e-5)
            )
            valid_positions = np.flatnonzero(valid)
            death_positions = np.flatnonzero(valid & (token_type == int(TokenType.DTH)))
            first_death_pos = int(death_positions[0]) if len(death_positions) else None
            result = {
                "rollout_index": rollout_index + batch_row,
                "generated_death_day": (
                    float(age[first_death_pos] - index_age_days)
                    if first_death_pos is not None
                    else np.nan
                ),
                "valid_generated_events": int(valid.sum()),
            }
            any_hit = False
            for outcome, outcome_tokens in targets.items():
                positions = np.flatnonzero(valid & np.isin(token, list(outcome_tokens)))
                first_position = int(positions[0]) if len(positions) else None
                if (
                    first_position is not None
                    and first_death_pos is not None
                    and first_position > first_death_pos
                ):
                    first_position = None
                result[f"first_{outcome}_day"] = (
                    float(age[first_position] - index_age_days)
                    if first_position is not None
                    else np.nan
                )
                any_hit |= first_position is not None
            followup_end = (
                float(np.max(age[valid_positions]) - index_age_days)
                if len(valid_positions)
                else 0.0
            )
            padded = bool(
                np.any(token_type == int(TokenType.PAD)) or np.any(token == 0)
            )
            reached = bool(
                first_death_pos is not None or padded or followup_end >= max_day - 1e-5
            )
            if padded and first_death_pos is None:
                followup_end = float(max_day)
            result["generated_followup_end_day"] = followup_end
            result["reached_horizon_or_death"] = reached
            result["max_token_cap_before_horizon"] = bool(
                first_death_pos is None and not padded and followup_end < max_day - 1e-5
            )
            result["any_target_hit"] = any_hit
            rows.append(result)
        rollout_index += current
    return pd.DataFrame(rows), effective_batch_size


def prefix_for_row(split_data, row, block_size):
    prefix = rows_before_index(
        split_data,
        int(row.patient_id_dense),
        int(row.index_age_days),
        int(block_size),
    )
    if prefix is None or len(prefix) == 0:
        raise RuntimeError(f"No pre-index rows for person_id={row.person_id}")
    return prefix


def run_batch_benchmark(
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
    benchmark = config["batch_size_benchmark"]
    if args.skip_batch_benchmark or not bool(benchmark["enabled"]):
        return pd.DataFrame(), pd.DataFrame()
    raw_parts = []
    summary_rows = []
    selected = sample.head(int(benchmark["patients"]))
    for batch_size in benchmark["batch_sizes"]:
        part_path = args.output_dir / "raw" / f"batch_benchmark_size_{batch_size}.parquet"
        summary_path = args.output_dir / "raw" / f"batch_benchmark_size_{batch_size}.json"
        if args.resume and summary_path.is_file():
            stats = json.loads(summary_path.read_text(encoding="utf-8"))
            if part_path.is_file():
                part = pd.read_parquet(part_path)
            elif stats.get("status") == "OOM":
                part = pd.DataFrame()
            else:
                raise RuntimeError(
                    f"Benchmark summary says {stats.get('status')} but raw part is missing: "
                    f"{part_path}"
                )
            log(f"[RESUME] batch benchmark size={batch_size}")
            raw_parts.append(part)
            summary_rows.append(stats)
            continue
        frames = []
        started = time.perf_counter()
        oom = False
        peak_bytes = 0
        try:
            for row in selected.itertuples(index=False):
                prefix = prefix_for_row(split_data, row, int(model.config.block_size))
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                frame, effective = generate_patient_rollouts(
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
                    int(benchmark["rollouts_per_patient"]),
                    int(batch_size),
                    seed_offset=10_000_000 + int(batch_size) * 100_000,
                    allow_oom_fallback=False,
                )
                frame.insert(0, "person_id", int(row.person_id))
                frame.insert(1, "requested_batch_size", int(batch_size))
                frame.insert(2, "effective_batch_size", int(effective))
                frames.append(frame)
                if args.device == "cuda":
                    peak_bytes = max(peak_bytes, int(torch.cuda.max_memory_allocated()))
        except torch.cuda.OutOfMemoryError:
            oom = True
            torch.cuda.empty_cache()
        synchronize(args.device)
        elapsed = time.perf_counter() - started
        part = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        stats = {
            "requested_batch_size": int(batch_size),
            "status": "OOM" if oom else "PASS",
            "patients": int(part["person_id"].nunique()) if len(part) else 0,
            "trajectories": int(len(part)),
            "elapsed_seconds": elapsed,
            "seconds_per_trajectory": elapsed / len(part) if len(part) else np.nan,
            "trajectories_per_second": len(part) / elapsed if elapsed > 0 else np.nan,
            "peak_gpu_memory_gib": peak_bytes / (1024**3),
            "mean_valid_generated_events": (
                float(part["valid_generated_events"].mean()) if len(part) else np.nan
            ),
            "cap_before_horizon_fraction": (
                float(part["max_token_cap_before_horizon"].mean())
                if len(part)
                else np.nan
            ),
            "estimated_main_hours_at_measured_rate": (
                elapsed / len(part) * int(config["total_trajectories"]) / 3600
                if len(part)
                else np.nan
            ),
        }
        if len(part):
            atomic_to_parquet(part, part_path)
            log(f"[RAW SAVED] {part_path}")
        write_json(stats, summary_path)
        log(f"[RAW SAVED] {summary_path}")
        raw_parts.append(part)
        summary_rows.append(stats)
    nonempty_parts = [frame for frame in raw_parts if len(frame)]
    raw = (
        pd.concat(nonempty_parts, ignore_index=True)
        if nonempty_parts
        else pd.DataFrame()
    )
    summary = pd.DataFrame(summary_rows)
    if len(raw):
        atomic_to_parquet(raw, args.output_dir / "raw" / "batch_benchmark_trajectories.parquet")
    summary.to_csv(args.output_dir / "batch_size_benchmark.csv", index=False)
    return raw, summary


def validate_rollout_part(part, chunk, config):
    required = {
        "person_id",
        "patient_id_dense",
        "sample_order",
        "rollout_index",
        "generated_death_day",
        "generated_followup_end_day",
        "reached_horizon_or_death",
        "max_token_cap_before_horizon",
        "valid_generated_events",
        "patient_generation_seconds",
        "effective_batch_size",
        "peak_gpu_memory_gib",
    } | {f"first_{outcome}_day" for outcome in config["outcomes"]}
    missing = sorted(required - set(part.columns))
    if missing:
        raise ValueError(f"Rollout part missing columns: {missing}")
    expected_rows = len(chunk) * int(config["rollouts_per_patient"])
    if len(part) != expected_rows:
        raise ValueError(f"Rollout part rows={len(part)} expected={expected_rows}")
    expected_people = set(chunk["person_id"].astype(np.int64))
    actual_people = set(part["person_id"].astype(np.int64))
    if expected_people != actual_people:
        raise ValueError("Rollout part patient IDs do not match its sample chunk")
    counts = part.groupby("person_id")["rollout_index"].agg(["count", "min", "max"])
    rollouts = int(config["rollouts_per_patient"])
    if not (
        counts["count"].eq(rollouts).all()
        and counts["min"].eq(0).all()
        and counts["max"].eq(rollouts - 1).all()
    ):
        raise ValueError("Rollout indexes are incomplete")
    if part.duplicated(["person_id", "rollout_index"]).any():
        raise ValueError("Rollout part has duplicate person_id/rollout_index rows")


def run_main_rollouts(
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
    parts_dir = args.output_dir / "raw" / "rollout_parts"
    chunk_size = int(config["patient_chunk_size"])
    total_chunks = math.ceil(len(sample) / chunk_size)
    for chunk_id, start in enumerate(range(0, len(sample), chunk_size)):
        chunk = sample.iloc[start : start + chunk_size].copy()
        part_path = parts_dir / f"rollout_trajectories_part_{chunk_id:05d}.parquet"
        if args.resume and part_path.is_file():
            part = pd.read_parquet(part_path)
            validate_rollout_part(part, chunk, config)
            log(
                f"[RESUME] chunk={chunk_id + 1}/{total_chunks} "
                f"patients={len(chunk):,} rows={len(part):,}"
            )
            continue
        frames = []
        chunk_started = time.perf_counter()
        for local_index, row in enumerate(chunk.itertuples(index=False), start=1):
            prefix = prefix_for_row(split_data, row, int(model.config.block_size))
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            synchronize(args.device)
            patient_started = time.perf_counter()
            frame, effective_batch = generate_patient_rollouts(
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
                int(config["rollouts_per_patient"]),
                int(config["rollout_batch_size"]),
                allow_oom_fallback=True,
            )
            synchronize(args.device)
            seconds = time.perf_counter() - patient_started
            peak_bytes = (
                int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0
            )
            frame.insert(0, "person_id", int(row.person_id))
            frame.insert(1, "patient_id_dense", int(row.patient_id_dense))
            frame.insert(2, "sample_order", int(row.sample_order))
            frame["patient_generation_seconds"] = seconds
            frame["effective_batch_size"] = int(effective_batch)
            frame["peak_gpu_memory_gib"] = peak_bytes / (1024**3)
            frames.append(frame)
            completed = start + local_index
            hits = {
                outcome: int(frame[f"first_{outcome}_day"].notna().sum())
                for outcome in config["outcomes"]
            }
            log(
                f"[PATIENT DONE] {completed}/{len(sample)} person_id={int(row.person_id)} "
                f"seconds={seconds:,.1f} effective_batch={effective_batch} hits={hits}"
            )
        part = pd.concat(frames, ignore_index=True)
        validate_rollout_part(part, chunk, config)
        atomic_to_parquet(part, part_path)
        log(
            f"[RAW SAVED] chunk={chunk_id + 1}/{total_chunks} "
            f"patients={len(chunk):,} rows={len(part):,} "
            f"seconds={time.perf_counter() - chunk_started:,.1f} path={part_path}"
        )

    part_paths = sorted(parts_dir.glob("rollout_trajectories_part_*.parquet"))
    if len(part_paths) != total_chunks:
        raise RuntimeError(f"Expected {total_chunks} parts, found {len(part_paths)}")
    frames = []
    for chunk_id, path in enumerate(part_paths):
        start = chunk_id * chunk_size
        chunk = sample.iloc[start : start + chunk_size]
        part = pd.read_parquet(path)
        validate_rollout_part(part, chunk, config)
        frames.append(part)
    trajectories = pd.concat(frames, ignore_index=True)
    if len(trajectories) != int(config["total_trajectories"]):
        raise RuntimeError(
            f"Combined trajectories={len(trajectories)} expected={config['total_trajectories']}"
        )
    combined_path = args.output_dir / "raw" / "rollout_trajectories.parquet"
    atomic_to_parquet(trajectories, combined_path)
    log(f"[RAW SAVED] {combined_path}")
    return trajectories


def bootstrap_mean_ci(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        boot[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def build_patient_risks(trajectories, config):
    patient_rows = []
    for person_id, group in trajectories.groupby("person_id", sort=False):
        row = {
            "person_id": int(person_id),
            "patient_id_dense": int(group["patient_id_dense"].iloc[0]),
            "sample_order": int(group["sample_order"].iloc[0]),
            "rollouts": int(len(group)),
            "patient_generation_seconds": float(group["patient_generation_seconds"].iloc[0]),
            "effective_batch_size": int(group["effective_batch_size"].iloc[0]),
            "peak_gpu_memory_gib": float(group["peak_gpu_memory_gib"].iloc[0]),
        }
        death = group["generated_death_day"].to_numpy(dtype=np.float64)
        followup = group["generated_followup_end_day"].to_numpy(dtype=np.float64)
        for outcome in config["outcomes"]:
            hit_day = group[f"first_{outcome}_day"].to_numpy(dtype=np.float64)
            for label, day in LANDMARKS.items():
                hit = np.isfinite(hit_day) & (hit_day <= day)
                usable = (
                    (followup >= day)
                    | hit
                    | (np.isfinite(death) & (death <= day))
                )
                hits = int(hit.sum())
                denominator = int(usable.sum())
                row[f"{outcome}_hits_{label}"] = hits
                row[f"{outcome}_usable_{label}"] = denominator
                row[f"{outcome}_risk_{label}"] = (
                    hits / denominator if denominator else np.nan
                )
        patient_rows.append(row)
    return pd.DataFrame(patient_rows).sort_values("sample_order")


def summarize_outcomes(patient_risks, trajectories, audit_precheck, config):
    audit_lookup = audit_precheck.set_index("outcome").to_dict("index")
    rows = []
    for outcome_index, outcome in enumerate(config["outcomes"]):
        for label, day in LANDMARKS.items():
            risks = patient_risks[f"{outcome}_risk_{label}"].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(
                risks, 1000, int(config["random_seed"]) + outcome_index * 100 + day
            )
            hits = int(patient_risks[f"{outcome}_hits_{label}"].sum())
            usable = int(patient_risks[f"{outcome}_usable_{label}"].sum())
            hit_patients = int((patient_risks[f"{outcome}_hits_{label}"] > 0).sum())
            rows.append(
                {
                    "outcome": outcome,
                    "horizon": label,
                    "horizon_days": day,
                    "patients": int(len(patient_risks)),
                    "trajectories": int(len(trajectories)),
                    "usable_trajectories": usable,
                    "outcome_hits": hits,
                    "trajectory_hit_rate_among_usable": hits / usable if usable else np.nan,
                    "mean_patient_rollout_risk": float(np.nanmean(risks)),
                    "median_patient_rollout_risk": float(np.nanmedian(risks)),
                    "patient_bootstrap_ci95_lower": ci_low,
                    "patient_bootstrap_ci95_upper": ci_high,
                    "patients_with_at_least_one_hit": hit_patients,
                    "zero_hit_patient_fraction": float(
                        (patient_risks[f"{outcome}_hits_{label}"] == 0).mean()
                    ),
                    "per_patient_probability_resolution": 1.0
                    / int(config["rollouts_per_patient"]),
                    "audit_full_test_cumulative_incidence_context": (
                        audit_lookup[outcome]["audit_full_test_cumulative_incidence_5y"]
                        if label == "5y"
                        else np.nan
                    ),
                    "comparison_warning": (
                        "Audit incidence uses the full eligible cohort; pilot rollout risk uses "
                        "the random common modelable cohort and is not a calibration comparison."
                    ),
                }
            )
    return pd.DataFrame(rows)


def generation_diagnostics(patient_risks, trajectories, config, wall_seconds):
    unique_runtime = patient_risks["patient_generation_seconds"]
    measured_seconds = float(unique_runtime.sum())
    return {
        "patients": int(len(patient_risks)),
        "trajectories": int(len(trajectories)),
        "rollouts_per_patient": int(config["rollouts_per_patient"]),
        "current_invocation_generation_wall_seconds": wall_seconds,
        "stored_sum_patient_generation_seconds": measured_seconds,
        "stored_sum_patient_generation_hours": measured_seconds / 3600,
        "seconds_per_trajectory": (
            measured_seconds / len(trajectories) if measured_seconds > 0 else np.nan
        ),
        "trajectories_per_second": (
            len(trajectories) / measured_seconds if measured_seconds > 0 else np.nan
        ),
        "median_patient_generation_seconds": float(unique_runtime.median()),
        "p90_patient_generation_seconds": float(unique_runtime.quantile(0.90)),
        "reached_horizon_or_death_fraction": float(
            trajectories["reached_horizon_or_death"].astype(bool).mean()
        ),
        "max_token_cap_before_horizon_fraction": float(
            trajectories["max_token_cap_before_horizon"].astype(bool).mean()
        ),
        "generated_death_fraction": float(trajectories["generated_death_day"].notna().mean()),
        "mean_valid_generated_events": float(trajectories["valid_generated_events"].mean()),
        "median_valid_generated_events": float(trajectories["valid_generated_events"].median()),
        "p90_valid_generated_events": float(trajectories["valid_generated_events"].quantile(0.90)),
        "max_peak_gpu_memory_gib": float(patient_risks["peak_gpu_memory_gib"].max()),
        "effective_batch_sizes": sorted(
            patient_risks["effective_batch_size"].astype(int).unique().tolist()
        ),
        "generation_batch_semantics": "parallel futures for one patient; not concurrent different patients",
    }


def run_self_test():
    trajectories = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "patient_id_dense": [11, 11, 22, 22],
            "sample_order": [0, 0, 1, 1],
            "rollout_index": [0, 1, 0, 1],
            "generated_death_day": [np.nan, np.nan, 200.0, np.nan],
            "generated_followup_end_day": [1826.0, 1826.0, 200.0, 100.0],
            "patient_generation_seconds": [2.0, 2.0, 3.0, 3.0],
            "effective_batch_size": [2, 2, 2, 2],
            "peak_gpu_memory_gib": [1.0, 1.0, 1.2, 1.2],
            "first_diabetes_day": [100.0, np.nan, np.nan, np.nan],
            "first_chronic_kidney_disease_day": [np.nan] * 4,
            "first_dyslipidemia_day": [50.0, 500.0, np.nan, np.nan],
        }
    )
    config = {"outcomes": REQUIRED_OUTCOMES, "rollouts_per_patient": 2}
    patients = build_patient_risks(trajectories, config)
    if not np.isclose(patients.loc[patients["person_id"].eq(1), "diabetes_risk_5y"].iloc[0], 0.5):
        raise AssertionError("patient rollout risk failed")
    if int(patients.loc[patients["person_id"].eq(2), "diabetes_usable_5y"].iloc[0]) != 1:
        raise AssertionError("death-censored usability failed")
    log("[SELF-TEST PASS] patient risk and death/horizon usability")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    require_paths(args)
    config = load_config(args.config_file)
    fingerprint = run_fingerprint(args, config)
    prepare_output(args, fingerprint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    started = datetime.now(timezone.utc)
    endpoint = {
        "status": "pilot_not_calibration",
        "outcomes": config["outcomes"],
        "cohort": config["cohort_rule"],
        "future_outcome_labels_used_for_sampling": False,
        "patients": int(config["patients"]),
        "rollouts_per_patient": int(config["rollouts_per_patient"]),
        "total_main_trajectories": int(config["total_trajectories"]),
        "horizon_days": int(config["horizon_days"]),
        "runtime_basis": config["runtime_basis"],
        "claim_boundary": config["claim_boundary"],
    }
    write_json(endpoint, args.output_dir / "endpoint_definition.json")

    audit_precheck = validate_audit(args, config)
    sample, sampling_frame = build_or_load_sample(args, config)
    log("[START] load FERMAT block-2048 checkpoint")
    model, checkpoint = load_model(args.fermat_ckpt, args.device)
    if int(model.config.block_size) != 2048:
        raise RuntimeError(
            f"FERMAT checkpoint block_size={model.config.block_size}; expected 2048"
        )
    registry, registry_path = load_registry(args.data_dir)
    token_type_lookup, _, clinical_mask = registry_maps(
        registry, int(model.config.vocab_size), args.device
    )
    targets = load_target_tokens(args, config, registry, int(model.config.vocab_size))
    death_tokens = death_model_tokens(registry, int(model.config.vocab_size))
    if not death_tokens:
        raise RuntimeError("No DTH model tokens were found")
    split_data = load_split_data(args.data_dir, config["split"])

    _, batch_summary = run_batch_benchmark(
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
    if len(batch_summary):
        log("## BATCH_SIZE_BENCHMARK\n" + batch_summary.to_csv(index=False).rstrip())

    log(
        f"[START] main pilot patients={config['patients']} "
        f"rollouts={config['rollouts_per_patient']} "
        f"trajectories={config['total_trajectories']} "
        f"prior_measured_estimate_hours={config['runtime_basis']['planned_main_hours']:.3f}"
    )
    generation_started = time.perf_counter()
    trajectories = run_main_rollouts(
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
    generation_wall_seconds = time.perf_counter() - generation_started
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    patient_risks = build_patient_risks(trajectories, config)
    patient_risks = sample.merge(
        patient_risks,
        on=["person_id", "patient_id_dense", "sample_order"],
        how="inner",
        validate="one_to_one",
    )
    atomic_to_parquet(patient_risks, args.output_dir / "rollout_patient_risks.parquet")
    outcome_summary = summarize_outcomes(
        patient_risks, trajectories, audit_precheck, config
    )
    outcome_summary.to_csv(args.output_dir / "outcome_rollout_hit_summary.csv", index=False)
    diagnostics = generation_diagnostics(
        patient_risks, trajectories, config, generation_wall_seconds
    )
    write_json(diagnostics, args.output_dir / "generation_diagnostics.json")

    runtime_by_patient = patient_risks[
        [
            "person_id",
            "sample_order",
            "patient_generation_seconds",
            "effective_batch_size",
            "peak_gpu_memory_gib",
        ]
    ].copy()
    runtime_by_patient.to_csv(args.output_dir / "patient_runtime.csv", index=False)

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "COMPLETE_MULTI_OUTCOME_ROLLOUT_PILOT",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds_total": (finished - started).total_seconds(),
        "checkpoint": str(args.fermat_ckpt),
        "checkpoint_iter": int(checkpoint.get("iter_num", checkpoint.get("iter", -1))),
        "checkpoint_block_size": int(checkpoint["model_args"]["block_size"]),
        "registry": registry_path,
        "config": config,
        "sampling_frame": sampling_frame,
        "generation_diagnostics": diagnostics,
        "multiple_patient_concurrency": {
            "tested": False,
            "reason": (
                "Current generate path batches multiple futures from one identical prefix. "
                "Different-patient prefixes are not mixed in this pilot."
            ),
        },
        "outputs": {
            "audit_precheck": str(args.output_dir / "audit_precheck.csv"),
            "sampling_frame_summary": str(
                args.output_dir / "raw" / "sampling_frame_summary.json"
            ),
            "sample_cohort": str(
                args.output_dir / "raw" / "pilot_sample_cohort.parquet"
            ),
            "raw_rollout_parts": str(args.output_dir / "raw" / "rollout_parts"),
            "raw_trajectories": str(
                args.output_dir / "raw" / "rollout_trajectories.parquet"
            ),
            "patient_risks": str(args.output_dir / "rollout_patient_risks.parquet"),
            "outcome_summary": str(args.output_dir / "outcome_rollout_hit_summary.csv"),
            "generation_diagnostics": str(
                args.output_dir / "generation_diagnostics.json"
            ),
            "batch_benchmark": str(args.output_dir / "batch_size_benchmark.csv"),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")

    five_year = outcome_summary.loc[outcome_summary["horizon"].eq("5y")]
    return_summary = [
        "## STATUS",
        "COMPLETE_MULTI_OUTCOME_ROLLOUT_PILOT",
        "## CONFIG",
        json.dumps(
            {
                "patients": config["patients"],
                "rollouts_per_patient": config["rollouts_per_patient"],
                "total_trajectories": config["total_trajectories"],
                "outcomes": config["outcomes"],
                "future_outcome_labels_used_for_sampling": False,
                "common_modelable_sampling_frame": sampling_frame[
                    "common_modelable_sampling_frame"
                ],
            },
            ensure_ascii=False,
        ),
        "## BATCH_SIZE_BENCHMARK",
        batch_summary.to_csv(index=False).rstrip() if len(batch_summary) else "SKIPPED",
        "## FIVE_YEAR_ROLLOUT_HITS",
        five_year.to_csv(index=False).rstrip(),
        "## GENERATION_DIAGNOSTICS",
        json.dumps(diagnostics, ensure_ascii=False),
        "## OUTPUT_DIR",
        str(args.output_dir),
    ]
    text = "\n".join(return_summary) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log("[COMPLETE] Task 30 multi-outcome rollout pilot finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
