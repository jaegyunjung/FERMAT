#!/usr/bin/env python3
"""Compare observed, Cox, and FERMAT-rollout diabetes risk curves.

The held-out test cohort is fixed at 2018-01-01 and excludes patients with
pre-index diabetes.  Cox models are fit on the complete modelable cohort.  The
expensive rollout evaluation uses an outcome-stratified test sample with inverse
sampling weights, matching the population event prevalence.  Raw trajectory
first-hit times are checkpointed before any curve summarization.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import math
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import TokenType
from scripts.run_snuh_task20_cox_survival_models import (
    COUNT_CATEGORICAL,
    COUNT_NUMERIC,
    build_survival_task,
    concordance_index,
    fit_cox_model,
    fit_transformer,
    horizon_auc,
    parquet_columns,
    phenotype_marker_info,
    transform_features,
)
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
from scripts.run_snuh_task30_group_mortality_risk_curves import (
    breslow_baseline_curve,
    cumulative_hazard_at_days,
    mean_risk_curve,
    observed_km_curve,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_LABEL_FILE = DEFAULT_LABEL_DIR / "patient_phenotype_labels_wide_20180101.parquet"
DEFAULT_FEATURE_FILE = (
    POD_ROOT / "task19" / "outputs" / "baseline_features" / "baseline_features_20180101.parquet"
)
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_LAB_FILE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "lab_marker_features"
    / "lab_marker_features_wide_20180101.parquet"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "cox_survival_2018_5y_block2048_1000ci_20260703"
    / "first_phenotype_dates_20180101.parquet"
)
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "diabetes_rollout_stratification"
LANDMARKS = (0, 365, 1095, 1826)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--lab-file", type=Path, default=DEFAULT_LAB_FILE)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--positive-patients", type=int, default=300)
    parser.add_argument("--negative-patients", type=int, default=900)
    parser.add_argument("--rollouts", type=int, default=60)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--patient-chunk-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--same-day-repeat-penalty", type=float, default=1.0)
    parser.add_argument("--same-day-temperature", type=float, default=1.0)
    parser.add_argument("--same-day-prob-cap", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="Run the complete observed/Cox cohort analysis and stop before any rollout generation.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def prepare_output(path, resume, overwrite):
    if path.exists() and any(path.iterdir()):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"{path} exists and is not empty; use --resume or a new output directory"
            )
    path.mkdir(parents=True, exist_ok=True)


def require_paths(args):
    required = [
        args.label_file,
        args.feature_file,
        args.embedding_file,
        args.lab_file,
        args.survival_cache,
        args.data_dir / "test.bin",
        args.data_dir / "patient_id_map.parquet",
        args.fermat_ckpt,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def load_modeling_data(args):
    label_cols = [
        "person_id",
        "split",
        "index_date",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
        "prior__diabetes",
    ]
    labels = pd.read_parquet(args.label_file, columns=label_cols)
    for column in ("index_date", "first_activity_date", "last_activity_date"):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")

    feature_cols = ["person_id", "split"] + COUNT_NUMERIC + COUNT_CATEGORICAL
    features = pd.read_parquet(args.feature_file, columns=feature_cols)

    embedding_columns = parquet_columns(args.embedding_file)
    emb_cols = [column for column in embedding_columns if column.startswith("emb_")]
    embeddings = pd.read_parquet(
        args.embedding_file,
        columns=[
            "person_id",
            "split",
            "has_embedding_sequence",
            "sequence_length_pre_index",
        ]
        + emb_cols,
    )
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()

    lab_columns = set(parquet_columns(args.lab_file))
    marker_info = phenotype_marker_info("diabetes", lab_columns)
    lab_cols = marker_info["marker_feature_columns"]
    labs = pd.read_parquet(args.lab_file, columns=["person_id", "split"] + lab_cols)

    dates = pd.read_parquet(args.survival_cache)
    dates = dates.loc[
        dates["phenotype"].eq("diabetes"), ["person_id", "phenotype", "first_phenotype_date"]
    ].copy()
    dates["first_phenotype_date"] = pd.to_datetime(
        dates["first_phenotype_date"], errors="coerce"
    )

    data = labels.merge(features, on=["person_id", "split"], how="inner")
    data = data.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(labs, on=["person_id", "split"], how="left")
    task = build_survival_task(data, dates, "diabetes", args)
    return task, emb_cols, lab_cols


def cohort_summary(task):
    rows = []
    for split in ("train", "val", "test"):
        sub = task.loc[task["split"].eq(split)]
        rows.append(
            {
                "split": split,
                "patients": int(len(sub)),
                "diabetes_events_within_5y": int(sub["event"].sum()),
                "median_followup_days": float(sub["duration_days"].median()),
                "max_followup_days": float(sub["duration_days"].max()),
            }
        )
    return pd.DataFrame(rows)


def fit_one_cox(name, train, val, test, numeric, categorical, missing, args, max_day):
    started = time.time()
    log(f"[START] fit {name}")
    transformer = fit_transformer(train, numeric, categorical, missing)
    train_x, feature_names = transform_features(train, transformer)
    val_x, _ = transform_features(val, transformer)
    test_x, _ = transform_features(test, transformer)
    beta, best_epoch, best_val_loss, history = fit_cox_model(
        train_x,
        train["duration_days"].to_numpy(dtype=np.float32),
        train["event"].to_numpy(dtype=np.int8),
        val_x,
        val["duration_days"].to_numpy(dtype=np.float32),
        val["event"].to_numpy(dtype=np.int8),
        args,
    )
    train_score = train_x @ beta
    test_score = test_x @ beta
    event_times, cumulative_hazard = breslow_baseline_curve(
        train["duration_days"], train["event"], train_score, max_day
    )
    days = np.arange(max_day + 1, dtype=np.int32)
    baseline = cumulative_hazard_at_days(event_times, cumulative_hazard, days)
    test_risk_5y = -np.expm1(
        -np.exp(np.clip(test_score, -50, 50)) * baseline[-1]
    )
    mean_curve = mean_risk_curve(test_score, baseline)
    auc, cases, controls = horizon_auc(
        test["duration_days"], test["event"], test_score, max_day
    )
    metric = {
        "model": name,
        "features": int(len(feature_names)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_c_index": float(
            concordance_index(test["duration_days"], test["event"], test_score)
        ),
        "test_5y_auc": float(auc),
        "test_5y_cases": int(cases),
        "test_5y_controls": int(controls),
        "fit_seconds": time.time() - started,
    }
    model_dir = args.output_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "cox_model.npz",
        beta=beta,
        event_times=event_times,
        cumulative_baseline_hazard=cumulative_hazard,
        days=days,
        mean_test_risk=mean_curve,
    )
    write_json(transformer, model_dir / "transformer.json")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    write_json(metric, model_dir / "metrics.json")
    predictions = pd.DataFrame(
        {
            "person_id": test["person_id"].to_numpy(dtype=np.int64),
            "duration_days": test["duration_days"].to_numpy(dtype=np.float32),
            "event": test["event"].to_numpy(dtype=np.int8),
            f"{name}_score": test_score.astype(np.float32),
            f"{name}_risk_5y": test_risk_5y.astype(np.float32),
        }
    )
    log(
        f"[DONE] fit {name}: c_index={metric['test_c_index']:.4f}, "
        f"auc5y={metric['test_5y_auc']:.4f}, seconds={metric['fit_seconds']:,.1f}"
    )
    return metric, predictions, mean_curve


def build_or_load_cox(args, task, emb_cols, lab_cols, max_day):
    prediction_path = args.output_dir / "cox_test_predictions.parquet"
    curves_path = args.output_dir / "cox_population_curves.csv"
    metrics_path = args.output_dir / "cox_model_metrics.csv"
    if args.resume and prediction_path.exists() and curves_path.exists() and metrics_path.exists():
        saved_predictions = pd.read_parquet(prediction_path)
        saved_curves = pd.read_csv(curves_path)
        saved_metrics = pd.read_csv(metrics_path)
        required_prediction_columns = {
            "clinical_cox_risk_5y",
            "fermat_clinical_cox_risk_5y",
        }
        required_curve_columns = {
            "clinical_cox_mean_predicted",
            "fermat_clinical_cox_mean_predicted",
        }
        if required_prediction_columns.issubset(saved_predictions.columns) and required_curve_columns.issubset(saved_curves.columns):
            log("[RESUME] load saved Cox outputs")
            return saved_predictions, saved_curves, saved_metrics

    train = task.loc[task["split"].eq("train")].copy()
    val = task.loc[task["split"].eq("val")].copy()
    test = task.loc[task["split"].eq("test")].copy()
    clinical_numeric = list(COUNT_NUMERIC) + list(lab_cols)
    clinical_missing = list(lab_cols)
    specs = {
        "clinical_cox": (clinical_numeric, list(COUNT_CATEGORICAL), clinical_missing),
        "fermat_clinical_cox": (
            list(emb_cols) + clinical_numeric,
            list(COUNT_CATEGORICAL),
            clinical_missing,
        ),
    }
    metrics = []
    predictions = None
    curve = pd.DataFrame({"day": np.arange(max_day + 1, dtype=np.int32)})
    for name, (numeric, categorical, missing) in specs.items():
        metric, pred, mean_curve = fit_one_cox(
            name, train, val, test, numeric, categorical, missing, args, max_day
        )
        metrics.append(metric)
        if predictions is None:
            predictions = pred
        else:
            predictions = predictions.merge(
                pred.drop(columns=["duration_days", "event"]), on="person_id", how="inner"
            )
        curve[f"{name}_mean_predicted"] = mean_curve
        predictions.to_parquet(prediction_path, index=False)
        curve.to_csv(curves_path, index=False)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, curve, pd.DataFrame(metrics)


def build_rollout_sample(test, patient_map, args):
    sample_path = args.output_dir / "rollout_sample_cohort.parquet"
    if args.resume and sample_path.exists():
        log("[RESUME] load rollout sample cohort")
        return pd.read_parquet(sample_path)

    mapped = test.merge(
        patient_map.loc[patient_map["split"].eq("test")],
        on=["person_id", "split"],
        how="inner",
    )
    mapped["index_age_days"] = np.floor(
        pd.to_numeric(mapped["age_at_index"], errors="coerce") * 365.25
    ).astype(np.int64)
    positives = mapped.loc[mapped["event"].eq(1)]
    negatives = mapped.loc[mapped["event"].eq(0)]
    n_pos = min(args.positive_patients, len(positives))
    n_neg = min(args.negative_patients, len(negatives))
    pos = positives.sample(n=n_pos, random_state=args.random_seed).copy()
    neg = negatives.sample(n=n_neg, random_state=args.random_seed + 1).copy()
    pos["sampling_weight"] = len(positives) / max(n_pos, 1)
    neg["sampling_weight"] = len(negatives) / max(n_neg, 1)
    sample = pd.concat([pos, neg], ignore_index=True)
    sample = sample.sample(frac=1, random_state=args.random_seed + 2).reset_index(drop=True)
    sample["sample_order"] = np.arange(len(sample), dtype=np.int32)
    keep = [
        "person_id",
        "split",
        "patient_id_dense",
        "index_age_days",
        "duration_days",
        "event",
        "sampling_weight",
        "sample_order",
    ]
    sample = sample[keep]
    sample.to_parquet(sample_path, index=False)
    log(
        f"[RAW SAVED] {sample_path} rows={len(sample):,} "
        f"events={int(sample['event'].sum()):,}"
    )
    return sample


def death_model_tokens(registry, vocab_size):
    result = []
    for row in registry:
        model_token = int(row["token_id"]) + 1
        if model_token < vocab_size and registry_type(row) == int(TokenType.DTH):
            result.append(model_token)
    return sorted(set(result))


def generate_patient_rollouts(
    model,
    prefix,
    target_model_tokens,
    death_tokens,
    token_type_lookup,
    clinical_mask,
    index_age_days,
    max_day,
    args,
    person_seed,
):
    idx0, age0, type0 = collate_prefix(prefix, args.device)
    horizon_age = float(index_age_days + max_day)
    rows = []
    rollout_index = 0
    batch_size = max(1, int(args.rollout_batch_size))
    while rollout_index < args.rollouts:
        current = min(batch_size, args.rollouts - rollout_index)
        seed = int(args.random_seed + person_seed * 1009 + rollout_index)
        torch.manual_seed(seed)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            with torch.no_grad(), dtype_context(args.device, args.dtype):
                idx_out, age_out, type_out, _ = model.generate(
                    idx0.repeat(current, 1),
                    age0.repeat(current, 1),
                    type0.repeat(current, 1),
                    max_new_tokens=args.max_new_tokens,
                    max_age=horizon_age,
                    no_repeat=False,
                    termination_tokens=death_tokens,
                    token_type_lookup=token_type_lookup,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    allowed_token_mask=clinical_mask,
                    same_day_no_repeat=False,
                    same_day_repeat_penalty=args.same_day_repeat_penalty,
                    same_day_temperature=args.same_day_temperature,
                    same_day_prob_cap=args.same_day_prob_cap,
                    return_final_logits=False,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current <= 1:
                raise
            batch_size = max(1, current // 2)
            log(f"[OOM RETRY] rollout_batch_size={batch_size}")
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
            hit_positions = np.flatnonzero(valid & np.isin(token, list(target_model_tokens)))
            death_positions = np.flatnonzero(valid & (token_type == int(TokenType.DTH)))
            first_death_pos = int(death_positions[0]) if len(death_positions) else None
            first_hit_pos = int(hit_positions[0]) if len(hit_positions) else None
            if first_hit_pos is not None and first_death_pos is not None and first_hit_pos > first_death_pos:
                first_hit_pos = None
            first_hit_day = (
                float(age[first_hit_pos] - index_age_days) if first_hit_pos is not None else np.nan
            )
            death_day = (
                float(age[first_death_pos] - index_age_days)
                if first_death_pos is not None
                else np.nan
            )
            followup_end = (
                float(np.max(age[valid_positions]) - index_age_days)
                if len(valid_positions)
                else 0.0
            )
            padded = bool(np.any(token_type == int(TokenType.PAD)) or np.any(token == 0))
            reached_horizon = bool(
                first_death_pos is not None or padded or followup_end >= max_day - 1e-5
            )
            if padded and first_death_pos is None:
                followup_end = float(max_day)
            cap_before_horizon = bool(
                first_hit_pos is None and first_death_pos is None and not reached_horizon
            )
            rows.append(
                {
                    "rollout_index": rollout_index + batch_row,
                    "first_diabetes_day": first_hit_day,
                    "generated_death_day": death_day,
                    "generated_followup_end_day": followup_end,
                    "reached_5y_or_death": reached_horizon,
                    "cap_before_5y": cap_before_horizon,
                    "valid_generated_events": int(valid.sum()),
                }
            )
        rollout_index += current
    return pd.DataFrame(rows)


def run_or_load_rollouts(args, sample, max_day):
    parts_dir = args.output_dir / "rollout_trajectory_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.fermat_ckpt, args.device)
    if int(model.config.block_size) != 2048:
        raise RuntimeError(
            f"FERMAT checkpoint block_size={model.config.block_size}; expected 2048"
        )
    registry, _ = load_registry(args.data_dir)
    token_type_lookup, _, clinical_mask = registry_maps(
        registry, int(model.config.vocab_size), args.device
    )
    phenotype_tokens, mapping, _ = load_phenotype_token_map(
        args.label_dir, registry, ["diabetes"]
    )
    mapping.to_csv(args.output_dir / "diabetes_token_mapping.csv", index=False)
    targets = {int(token) + 1 for token in phenotype_tokens["diabetes"]}
    deaths = death_model_tokens(registry, int(model.config.vocab_size))
    split_data = load_split_data(args.data_dir, "test")

    total_chunks = math.ceil(len(sample) / args.patient_chunk_size)
    for chunk_id, start in enumerate(range(0, len(sample), args.patient_chunk_size)):
        part_path = parts_dir / f"rollout_trajectories_part_{chunk_id:05d}.parquet"
        if args.resume and part_path.exists():
            log(f"[RESUME] chunk={chunk_id + 1}/{total_chunks}")
            continue
        chunk = sample.iloc[start : start + args.patient_chunk_size]
        frames = []
        for row in chunk.itertuples(index=False):
            prefix = rows_before_index(
                split_data,
                int(row.patient_id_dense),
                int(row.index_age_days),
                int(model.config.block_size),
            )
            if prefix is None or len(prefix) == 0:
                raise RuntimeError(f"No pre-index FERMAT rows for person_id={row.person_id}")
            frame = generate_patient_rollouts(
                model,
                prefix,
                targets,
                deaths,
                token_type_lookup,
                clinical_mask,
                int(row.index_age_days),
                max_day,
                args,
                int(row.sample_order),
            )
            frame.insert(0, "person_id", int(row.person_id))
            frames.append(frame)
        part = pd.concat(frames, ignore_index=True)
        part.to_parquet(part_path, index=False)
        log(
            f"[RAW SAVED] chunk={chunk_id + 1}/{total_chunks} "
            f"patients={len(chunk):,} rows={len(part):,} path={part_path}"
        )

    part_paths = sorted(parts_dir.glob("rollout_trajectories_part_*.parquet"))
    if len(part_paths) != total_chunks:
        raise RuntimeError(f"Expected {total_chunks} rollout parts, found {len(part_paths)}")
    trajectories = pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)
    expected = len(sample) * args.rollouts
    if len(trajectories) != expected:
        raise RuntimeError(f"Expected {expected} rollout rows, found {len(trajectories)}")
    all_path = args.output_dir / "rollout_trajectories.parquet"
    trajectories.to_parquet(all_path, index=False)
    write_json(
        {
            "checkpoint": str(args.fermat_ckpt),
            "checkpoint_iter": int(checkpoint.get("iter_num", checkpoint.get("iter", -1))),
            "block_size": int(model.config.block_size),
            "rollouts_per_patient": int(args.rollouts),
            "patients": int(len(sample)),
            "trajectory_rows": int(len(trajectories)),
            "max_new_tokens": int(args.max_new_tokens),
        },
        args.output_dir / "rollout_manifest.json",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trajectories


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_nanmean_matrix(matrix, weights):
    matrix = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)[:, None]
    valid = np.isfinite(matrix)
    numerator = np.sum(np.where(valid, matrix * weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def weighted_observed_curve(frame, max_day):
    duration = frame["duration_days"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=np.int8)
    weight = frame["sampling_weight"].to_numpy(dtype=np.float64)
    survival = 1.0
    rows = []
    for day in range(max_day + 1):
        at_risk = float(weight[duration >= day].sum())
        deaths = float(weight[(duration == day) & (event == 1)].sum())
        if at_risk > 0 and deaths > 0:
            survival *= 1.0 - deaths / at_risk
        rows.append(
            {
                "day": day,
                "observed_cumulative_incidence": 1.0 - survival,
                "weighted_patients_at_risk": at_risk,
            }
        )
    return pd.DataFrame(rows)


def patient_rollout_curve(frame, max_day):
    hit = frame["first_diabetes_day"].to_numpy(dtype=np.float64)
    death = frame["generated_death_day"].to_numpy(dtype=np.float64)
    follow = frame["generated_followup_end_day"].to_numpy(dtype=np.float64)
    days = np.arange(max_day + 1, dtype=np.float64)
    output = np.full(max_day + 1, np.nan, dtype=np.float64)
    for start in range(0, max_day + 1, 128):
        stop = min(start + 128, max_day + 1)
        d = days[start:stop, None]
        numerator = np.sum(np.isfinite(hit)[None, :] & (hit[None, :] <= d), axis=1)
        covered = follow[None, :] >= d
        hit_known = np.isfinite(hit)[None, :] & (hit[None, :] <= d)
        death_known = np.isfinite(death)[None, :] & (death[None, :] <= d)
        denominator = np.sum(covered | hit_known | death_known, axis=1)
        output[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
    return output


def summarize_rollouts(sample, trajectories, cox_predictions, max_day, args):
    curves = {}
    patient_rows = []
    for person_id, sub in trajectories.groupby("person_id", sort=False):
        curve = patient_rollout_curve(sub, max_day)
        curves[int(person_id)] = curve
        row = {"person_id": int(person_id)}
        for day in LANDMARKS[1:]:
            row[f"rollout_risk_day_{day}"] = float(curve[day])
        row["usable_rollout_fraction_5y"] = float(
            1.0 - sub["cap_before_5y"].astype(bool).mean()
        )
        row["cap_before_5y_fraction"] = float(sub["cap_before_5y"].astype(bool).mean())
        row["mean_valid_generated_events"] = float(sub["valid_generated_events"].mean())
        patient_rows.append(row)
    patients = sample.merge(pd.DataFrame(patient_rows), on="person_id", how="inner")
    patients = patients.merge(cox_predictions, on=["person_id", "duration_days", "event"], how="left")
    patients.to_parquet(args.output_dir / "rollout_patient_predictions.parquet", index=False)

    weights = patients["sampling_weight"].to_numpy(dtype=np.float64)
    rollout_matrix = np.stack([curves[int(pid)] for pid in patients["person_id"]])
    population_rollout = weighted_nanmean_matrix(rollout_matrix, weights)
    return patients, curves, population_rollout


def assign_risk_groups(patients):
    risk = patients["rollout_risk_day_1826"]
    try:
        quintile = pd.qcut(risk, q=5, labels=False, duplicates="drop")
    except ValueError:
        quintile = pd.Series(np.zeros(len(patients), dtype=int), index=patients.index)
    patients = patients.copy()
    patients["risk_quantile_bin"] = quintile.astype("Int64")
    bins = sorted(patients["risk_quantile_bin"].dropna().astype(int).unique().tolist())
    if len(bins) < 3:
        raise RuntimeError(
            "Rollout 5-year risks produced fewer than three distinct quantile bins; "
            "increase --rollouts before risk-group interpretation"
        )
    selected = {bins[0]: "low", bins[len(bins) // 2]: "middle", bins[-1]: "high"}
    patients["risk_group"] = patients["risk_quantile_bin"].map(selected)
    return patients


def build_group_outputs(patients, curves, max_day):
    patients = assign_risk_groups(patients)
    rows = []
    summaries = []
    for group_name in ("low", "middle", "high"):
        sub = patients.loc[patients["risk_group"].eq(group_name)].copy()
        observed = weighted_observed_curve(sub, max_day)
        matrix = np.stack([curves[int(pid)] for pid in sub["person_id"]])
        predicted = weighted_nanmean_matrix(
            matrix, sub["sampling_weight"].to_numpy(dtype=np.float64)
        )
        group_curve = observed.copy()
        group_curve.insert(0, "risk_group", group_name)
        group_curve["rollout_predicted_cumulative_incidence"] = predicted
        rows.append(group_curve)
        summaries.append(
            {
                "risk_group": group_name,
                "sample_patients": int(len(sub)),
                "sample_events": int(sub["event"].sum()),
                "weighted_population_patients": float(sub["sampling_weight"].sum()),
                "mean_rollout_risk_5y": weighted_mean(
                    sub["rollout_risk_day_1826"], sub["sampling_weight"]
                ),
                "observed_risk_5y": float(observed.iloc[-1]["observed_cumulative_incidence"]),
            }
        )
    return patients, pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def calibration_deciles(patients, max_day):
    frame = patients.copy()
    try:
        frame["calibration_bin"] = pd.qcut(
            frame["rollout_risk_day_1826"], q=10, labels=False, duplicates="drop"
        )
    except ValueError:
        frame["calibration_bin"] = 0
    rows = []
    for bin_id, sub in frame.groupby("calibration_bin", sort=True):
        observed = weighted_observed_curve(sub, max_day).iloc[-1]
        rows.append(
            {
                "calibration_bin": int(bin_id),
                "sample_patients": int(len(sub)),
                "sample_events": int(sub["event"].sum()),
                "predicted_risk_5y": weighted_mean(
                    sub["rollout_risk_day_1826"], sub["sampling_weight"]
                ),
                "observed_risk_5y": float(observed["observed_cumulative_incidence"]),
            }
        )
    return pd.DataFrame(rows)


def esc(value):
    return html.escape(str(value), quote=True)


def svg_line_chart(curve, output_path):
    width, height = 1100, 680
    left, right, top, bottom = 100, 40, 90, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    all_series = [
        ("observed", "Observed", "#222222", ""),
        ("clinical_cox_mean_predicted", "Clinical Cox", "#6b7280", "6 5"),
        ("fermat_clinical_cox_mean_predicted", "FERMAT + clinical Cox", "#2563eb", "6 5"),
        ("fermat_rollout_mean_predicted", "FERMAT rollout", "#dc2626", ""),
    ]
    series = [item for item in all_series if item[0] in curve.columns]
    max_y = max(float(curve[column].max()) for column, *_ in series)
    max_y = max(0.01, math.ceil(max_y * 1000) / 1000 * 1.12)
    x = lambda day: left + plot_w * float(day) / 1826.0
    y = lambda value: top + plot_h * (1.0 - float(value) / max_y)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:28px;font-weight:700}.sub{font-size:16px;fill:#555}.axis{font-size:14px}.legend{font-size:15px;font-weight:600}</style>',
        f'<text x="{left}" y="38" class="title">Diabetes cumulative incidence: observed vs predicted</text>',
        f'<text x="{left}" y="65" class="sub">Held-out 2018 cohort; all non-rollout curves use the complete modelable test cohort</text>',
    ]
    for frac in np.linspace(0, 1, 6):
        yy = y(max_y * frac)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" class="axis">{100*max_y*frac:.1f}%</text>')
    for day, label in [(0, "Start"), (365, "1 year"), (1095, "3 years"), (1826, "5 years")]:
        xx = x(day)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+28}" text-anchor="middle" class="axis">{label}</text>')
    for column, label, color, dash in series:
        points = " ".join(f"{x(d):.2f},{y(v):.2f}" for d, v in zip(curve["day"], curve[column]))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>' )
    legend_x = left + 25
    for i, (_, label, color, dash) in enumerate(series):
        lx = legend_x + i * 230
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{height-28}" x2="{lx+32}" y2="{height-28}" stroke="{color}" stroke-width="3"{dash_attr}/>' )
        parts.append(f'<text x="{lx+40}" y="{height-23}" class="legend">{esc(label)}</text>')
    parts.append(f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 28 {top+plot_h/2})" class="axis">Cumulative incidence</text>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_risk_groups(group_curve, output_path):
    width, height = 1260, 520
    margin, top, bottom, gap = 70, 90, 70, 50
    panel_w = (width - 2 * margin - 2 * gap) / 3
    plot_h = height - top - bottom
    max_y = max(
        float(group_curve["observed_cumulative_incidence"].max()),
        float(group_curve["rollout_predicted_cumulative_incidence"].max()),
        0.01,
    ) * 1.12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:27px;font-weight:700}.panel{font-size:18px;font-weight:700}.axis{font-size:13px}.legend{font-size:14px;font-weight:600}</style>',
        f'<text x="{margin}" y="38" class="title">Observed and rollout-predicted diabetes incidence by rollout risk group</text>',
        f'<line x1="{width-330}" y1="36" x2="{width-300}" y2="36" stroke="#222" stroke-width="3"/><text x="{width-292}" y="41" class="legend">Observed</text>',
        f'<line x1="{width-190}" y1="36" x2="{width-160}" y2="36" stroke="#dc2626" stroke-width="3" stroke-dasharray="6 5"/><text x="{width-152}" y="41" class="legend">Rollout</text>',
    ]
    for panel_i, group in enumerate(("low", "middle", "high")):
        sub = group_curve.loc[group_curve["risk_group"].eq(group)]
        left = margin + panel_i * (panel_w + gap)
        x = lambda day: left + panel_w * float(day) / 1826.0
        y = lambda value: top + plot_h * (1.0 - float(value) / max_y)
        parts.append(f'<text x="{left+panel_w/2:.1f}" y="{top-22}" text-anchor="middle" class="panel">{group.capitalize()} risk</text>')
        for frac in np.linspace(0, 1, 5):
            yy = y(max_y * frac)
            parts.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{left+panel_w:.1f}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
            if panel_i == 0:
                parts.append(f'<text x="{left-8:.1f}" y="{yy+4:.1f}" text-anchor="end" class="axis">{100*max_y*frac:.1f}%</text>')
        for day, label in [(0, "0"), (365, "1y"), (1095, "3y"), (1826, "5y")]:
            parts.append(f'<text x="{x(day):.1f}" y="{top+plot_h+24}" text-anchor="middle" class="axis">{label}</text>')
        for column, color, dash in [
            ("observed_cumulative_incidence", "#222222", ""),
            ("rollout_predicted_cumulative_incidence", "#dc2626", "6 5"),
        ]:
            points = " ".join(f"{x(d):.2f},{y(v):.2f}" for d, v in zip(sub["day"], sub[column]))
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>' )
    parts.append(f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2})" class="axis">Cumulative incidence</text>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def validate_outputs(population, group_curve, patients):
    checks = {
        "rollout_population_curve_decreases": int(
            np.sum(np.diff(population["fermat_rollout_mean_predicted"]) < -1e-12)
        ),
        "invalid_rollout_probability_values": int(
            np.sum(
                ~np.isfinite(patients["rollout_risk_day_1826"])
                | (patients["rollout_risk_day_1826"] < 0)
                | (patients["rollout_risk_day_1826"] > 1)
            )
        ),
        "mean_cap_before_5y_fraction": float(patients["cap_before_5y_fraction"].mean()),
        "risk_groups": int(group_curve["risk_group"].nunique()),
    }
    checks["status"] = (
        "PASS"
        if checks["rollout_population_curve_decreases"] == 0
        and checks["invalid_rollout_probability_values"] == 0
        and checks["risk_groups"] == 3
        and checks["mean_cap_before_5y_fraction"] <= 0.05
        else "FAIL"
    )
    return checks


def main():
    args = parse_args()
    for name in (
        "label_dir",
        "label_file",
        "feature_file",
        "embedding_file",
        "lab_file",
        "survival_cache",
        "data_dir",
        "fermat_ckpt",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.horizon = "5y"
    prepare_output(args.output_dir, args.resume, args.overwrite)
    require_paths(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    started = datetime.now(timezone.utc)
    max_day = 1826

    endpoint = {
        "cohort": "patients without recorded diabetes before 2018-01-01, with pre-index history",
        "outcome": "first recorded diabetes diagnosis after index",
        "horizon": "5 years (1826 days)",
        "observed_curve": "Kaplan-Meier 1-S(t); loss to follow-up is censored",
        "predicted_curves": [
            "clinical Cox",
            "FERMAT block-2048 embedding + same clinical Cox covariates",
            "FERMAT autoregressive rollout",
        ],
        "rollout_risk": "proportion of usable simulated futures with first diabetes by day t",
        "risk_groups": "low, middle, high groups from rollout-predicted 5-year risk quintiles",
    }
    write_json(endpoint, args.output_dir / "endpoint_definition.json")

    log("[START] load diabetes cohort")
    task, emb_cols, lab_cols = load_modeling_data(args)
    summary = cohort_summary(task)
    summary.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    print("## COHORT_SUMMARY_PRECHECK", flush=True)
    print(summary.to_csv(index=False).rstrip(), flush=True)
    if int(summary.loc[summary["split"].eq("train"), "diabetes_events_within_5y"].iloc[0]) < 50:
        raise RuntimeError("Too few train diabetes events for this comparison")
    if int(summary.loc[summary["split"].eq("test"), "diabetes_events_within_5y"].iloc[0]) < 50:
        raise RuntimeError("Too few test diabetes events for this comparison")

    test = task.loc[task["split"].eq("test")].copy()
    observed = observed_km_curve(test["duration_days"], test["event"], max_day)
    observed = observed.rename(columns={"observed_mortality": "observed"})
    observed.to_csv(args.output_dir / "full_test_observed_diabetes_curve.csv", index=False)
    log(f"[RAW SAVED] {args.output_dir / 'full_test_observed_diabetes_curve.csv'}")

    cox_predictions, cox_curves, cox_metrics = build_or_load_cox(
        args, task, emb_cols, lab_cols, max_day
    )
    cohort_population = observed[["day", "observed", "patients_at_risk_after_day"]].merge(
        cox_curves, on="day", how="inner"
    )
    cohort_population.to_csv(
        args.output_dir / "cohort_observed_and_cox_curves.csv", index=False
    )
    cohort_landmarks = cohort_population.loc[
        cohort_population["day"].isin(LANDMARKS)
    ].copy()
    cohort_landmarks.to_csv(
        args.output_dir / "cohort_observed_and_cox_landmarks.csv", index=False
    )
    svg_line_chart(
        cohort_population, args.output_dir / "cohort_observed_and_cox_curves.svg"
    )

    if args.skip_rollout:
        finished = datetime.now(timezone.utc)
        manifest = {
            "status": "complete_cohort_stage",
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "rollout_executed": False,
            "complete_test_cohort_used": True,
            "inputs": {
                "label_file": str(args.label_file),
                "feature_file": str(args.feature_file),
                "embedding_file": str(args.embedding_file),
                "lab_file": str(args.lab_file),
                "survival_cache": str(args.survival_cache),
            },
            "outputs": {
                "cohort_summary": str(args.output_dir / "cohort_summary.csv"),
                "cox_metrics": str(args.output_dir / "cox_model_metrics.csv"),
                "cohort_landmarks": str(
                    args.output_dir / "cohort_observed_and_cox_landmarks.csv"
                ),
                "cohort_svg": str(
                    args.output_dir / "cohort_observed_and_cox_curves.svg"
                ),
            },
        }
        write_json(manifest, args.output_dir / "manifest.json")
        return_summary = [
            "## STAGE",
            "COMPLETE_COHORT_STAGE_NO_ROLLOUT",
            "## COHORT_SUMMARY",
            summary.to_csv(index=False).rstrip(),
            "## COX_METRICS",
            cox_metrics.to_csv(index=False).rstrip(),
            "## COHORT_LANDMARKS",
            cohort_landmarks.to_csv(index=False).rstrip(),
            "## JUPYTER_SVG_PATH",
            str(args.output_dir / "cohort_observed_and_cox_curves.svg"),
        ]
        text = "\n".join(return_summary) + "\n"
        (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
        print(text, end="", flush=True)
        log("[COMPLETE] Cohort stage finished; no rollout was generated")
        return 0

    patient_map = load_patient_map(args.data_dir)
    sample = build_rollout_sample(test, patient_map, args)
    trajectories = run_or_load_rollouts(args, sample, max_day)
    patients, patient_curves, rollout_population = summarize_rollouts(
        sample, trajectories, cox_predictions, max_day, args
    )
    patients, group_curve, group_summary = build_group_outputs(
        patients, patient_curves, max_day
    )
    patients.to_parquet(args.output_dir / "rollout_patient_predictions.parquet", index=False)
    group_curve.to_csv(args.output_dir / "rollout_risk_group_curves.csv", index=False)
    group_summary.to_csv(args.output_dir / "rollout_risk_group_summary.csv", index=False)
    calibration = calibration_deciles(patients, max_day)
    calibration.to_csv(args.output_dir / "rollout_calibration_bins.csv", index=False)

    population = observed[["day", "observed", "patients_at_risk_after_day"]].merge(
        cox_curves, on="day", how="inner"
    )
    population["fermat_rollout_mean_predicted"] = rollout_population
    population.to_csv(args.output_dir / "population_curve_comparison.csv", index=False)
    landmark = population.loc[population["day"].isin(LANDMARKS)].copy()
    landmark.to_csv(args.output_dir / "population_curve_landmarks.csv", index=False)

    validation = validate_outputs(population, group_curve, patients)
    write_json(validation, args.output_dir / "curve_validation.json")
    svg_line_chart(population, args.output_dir / "population_curve_comparison.svg")
    svg_risk_groups(group_curve, args.output_dir / "rollout_risk_group_curves.svg")

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": validation["status"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "inputs": {
            "label_file": str(args.label_file),
            "feature_file": str(args.feature_file),
            "embedding_file": str(args.embedding_file),
            "lab_file": str(args.lab_file),
            "survival_cache": str(args.survival_cache),
            "fermat_ckpt": str(args.fermat_ckpt),
        },
        "rollout_sample": {
            "patients": int(len(sample)),
            "sample_events": int(sample["event"].sum()),
            "rollouts_per_patient": int(args.rollouts),
            "max_new_tokens": int(args.max_new_tokens),
            "inverse_sampling_weighted": True,
        },
        "outputs": {
            "population_svg": str(args.output_dir / "population_curve_comparison.svg"),
            "risk_group_svg": str(args.output_dir / "rollout_risk_group_curves.svg"),
            "population_landmarks": str(args.output_dir / "population_curve_landmarks.csv"),
            "risk_group_summary": str(args.output_dir / "rollout_risk_group_summary.csv"),
            "calibration_bins": str(args.output_dir / "rollout_calibration_bins.csv"),
            "validation": str(args.output_dir / "curve_validation.json"),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")

    return_summary = [
        "## CURVE_VALIDATION",
        json.dumps(validation, ensure_ascii=False),
        "## COHORT_SUMMARY",
        summary.to_csv(index=False).rstrip(),
        "## COX_METRICS",
        cox_metrics.to_csv(index=False).rstrip(),
        "## POPULATION_LANDMARKS",
        landmark.to_csv(index=False).rstrip(),
        "## ROLLOUT_RISK_GROUP_SUMMARY",
        group_summary.to_csv(index=False).rstrip(),
        "## ROLLOUT_CALIBRATION_BINS",
        calibration.to_csv(index=False).rstrip(),
        "## JUPYTER_SVG_PATHS",
        str(args.output_dir / "population_curve_comparison.svg"),
        str(args.output_dir / "rollout_risk_group_curves.svg"),
    ]
    text = "\n".join(return_summary) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log("[COMPLETE] Task30 diabetes rollout risk stratification finished")
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
