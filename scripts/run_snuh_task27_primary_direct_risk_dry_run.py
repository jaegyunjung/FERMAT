#!/usr/bin/env python3
"""Task27 dry run for primary direct-risk scoring.

This script does not try to finalize the primary table. It checks whether the
two intended scoring paths produce usable, non-degenerate scores on a small
sample:

* FERMAT (decoupled + two-stage): horizon-based rollout risk.
* Coupled pure (Delphi-like): closed-form exponential race risk from logits.

The main outputs are score sanity, horizon-reach generation diagnostics,
runtime, and a small AUROC/AUPRC readout for two phenotypes.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Fermat, FermatConfig, TokenType
from utils import get_p2i, load_data


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FERMAT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_COUPLED_CKPT = (
    POD_ROOT
    / "task26"
    / "outputs"
    / "coupled_primary_full_100k_20260705"
    / "coupled_pure"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task27" / "outputs" / "primary_direct_risk_dry_run_20260706"
MASK_TIME = -10000.0
PAD_TOKEN_TYPE = 0
CLINICAL_TYPES = {
    int(TokenType.DX): "DX",
    int(TokenType.RX): "RX",
    int(TokenType.PX): "PX",
    int(TokenType.DTH): "DTH",
}
HORIZON_DAYS = {"1y": 365.25, "5y": 365.25 * 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_FERMAT_CKPT)
    parser.add_argument("--coupled-ckpt", type=Path, default=DEFAULT_COUPLED_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--horizons", nargs="+", default=["1y", "5y"], choices=["1y", "5y"])
    parser.add_argument(
        "--phenotypes",
        nargs="+",
        default=["chronic_kidney_disease", "diabetes"],
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-patients-per-task", type=int, default=500)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--same-day-repeat-penalty", type=float, default=1.0)
    parser.add_argument("--same-day-temperature", type=float, default=1.0)
    parser.add_argument("--same-day-prob-cap", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def dtype_context(device: str, dtype: str):
    if dtype == "float32" or device == "cpu":
        return nullcontext()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Fermat(FermatConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_prefixes = ("transformer.", "lm_head.", "time_head.", "same_day_head.", "log_rate")
    critical_missing = [key for key in missing if key.startswith(critical_prefixes)]
    critical_unexpected = [key for key in unexpected if key.startswith(critical_prefixes)]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            "Critical checkpoint/model mismatch:\n"
            + json.dumps(
                {
                    "ckpt": str(path),
                    "critical_missing": critical_missing,
                    "critical_unexpected": critical_unexpected,
                    "all_missing": list(missing),
                    "all_unexpected": list(unexpected),
                },
                indent=2,
            )
        )
    model.to(device)
    model.eval()
    return model, checkpoint


def load_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    frame = pd.read_parquet(path, columns=["patient_id_dense", "person_id", "split"])
    frame["patient_id_dense"] = frame["patient_id_dense"].astype(np.int64)
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["split"] = frame["split"].astype(str)
    return frame


def load_split_data(data_dir: Path, split: str):
    path = data_dir / f"{split}.bin"
    data, has_types = load_data(path)
    if not has_types:
        raise ValueError(f"{path} is not a 4-column typed FERMAT bin")
    p2i = get_p2i(data)
    patient_ids = data[p2i[:, 0].astype(np.int64), 0].astype(np.int64)
    index = {
        int(patient_id): (int(start), int(length))
        for patient_id, (start, length) in zip(patient_ids, p2i)
    }
    return {"path": str(path), "data": data, "index": index}


def load_registry(data_dir: Path):
    for filename in ["token_registry.csv", "vocab.csv"]:
        path = data_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            return rows, str(path)
    raise FileNotFoundError("Expected token_registry.csv or vocab.csv")


def registry_type(row):
    value = row.get("token_type_id", row.get("token_type"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(TokenType[row["token_type"]])


def registry_maps(registry, vocab_size: int, device: str):
    token_type_lookup = {}
    token_label_lookup = {}
    clinical_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for row in registry:
        token_id = int(row["token_id"])
        model_token_id = token_id + 1
        if model_token_id >= vocab_size:
            continue
        token_type = registry_type(row)
        token_type_lookup[model_token_id] = token_type
        token_label_lookup[model_token_id] = row.get("token_key") or f"token_id={token_id}"
        if token_type in CLINICAL_TYPES:
            clinical_mask[model_token_id] = True
    return token_type_lookup, token_label_lookup, clinical_mask


def load_phenotype_token_map(label_dir: Path, registry, phenotypes):
    concept_path = label_dir / "phenotype_group_concept_map.csv"
    concept_map = pd.read_csv(concept_path)
    concept_map = concept_map.loc[concept_map["phenotype"].isin(phenotypes)].copy()
    token_key_to_id = {
        str(row.get("token_key")): int(row["token_id"])
        for row in registry
        if row.get("token_key")
    }
    mapping = {}
    rows = []
    for phenotype, group in concept_map.groupby("phenotype", sort=True):
        token_ids = []
        for concept_id in group["condition_concept_id"].astype(int).tolist():
            key = f"DX:{concept_id}"
            if key in token_key_to_id:
                token_ids.append(token_key_to_id[key])
        token_ids = sorted(set(token_ids))
        mapping[phenotype] = token_ids
        rows.append(
            {
                "phenotype": phenotype,
                "concept_ids": int(group["condition_concept_id"].nunique()),
                "matched_tokens": len(token_ids),
                "token_ids": ",".join(str(x) for x in token_ids),
            }
        )
    missing = [phenotype for phenotype in phenotypes if not mapping.get(phenotype)]
    if missing:
        raise RuntimeError(f"Phenotypes without mapped DX tokens: {missing}")
    return mapping, pd.DataFrame(rows), str(concept_path)


def load_label_targets(args):
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{safe_date(args.index_date)}.parquet"
    needed = ["person_id", "split", "age_at_index"]
    for horizon in args.horizons:
        for phenotype in args.phenotypes:
            needed.extend([f"eligible_{horizon}__{phenotype}", f"label_{horizon}__{phenotype}"])
    labels = pd.read_parquet(label_path, columns=sorted(set(needed)))
    labels = labels.loc[labels["split"].eq(args.split)].copy()
    labels["index_age_days"] = np.floor(labels["age_at_index"] * 365.25).astype(np.int64)
    return labels, str(label_path)


def rows_before_index(split_data, dense_id: int, index_age_days: int, block_size: int):
    location = split_data["index"].get(int(dense_id))
    if location is None:
        return None
    start, length = location
    rows = split_data["data"][start:start + length]
    rows = rows[rows[:, 1].astype(np.int64) < int(index_age_days)]
    if len(rows) > block_size:
        rows = rows[-block_size:]
    return rows


def collate_prefix(rows, device):
    idx = torch.from_numpy(rows[:, 2].astype(np.int64) + 1).unsqueeze(0).to(device)
    age = torch.from_numpy(rows[:, 1].astype(np.float32)).unsqueeze(0).to(device)
    token_type = torch.from_numpy(rows[:, 3].astype(np.int64)).unsqueeze(0).to(device)
    return idx, age, token_type


def closed_form_score(model, rows, target_stored_tokens, horizon_days: float, device: str, dtype: str):
    idx, age, token_type = collate_prefix(rows, device)
    target_model_tokens = torch.tensor(
        [int(token) + 1 for token in target_stored_tokens],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad(), dtype_context(device, dtype):
        logits, _, _ = model(idx, age, token_type, return_attention=False)
        last = logits[:, -1, :].float().squeeze(0)
        ignored = sorted(set(model.config.ignore_tokens + model.config.output_ignore_tokens))
        if ignored:
            last[torch.tensor(ignored, dtype=torch.long, device=device)] = -torch.inf
        target_logits = last[target_model_tokens]
        log_rate = torch.logsumexp(target_logits, dim=0)
        if (not model.config.decoupled_time_head) and getattr(model.config, "use_global_log_rate", True):
            log_rate = log_rate + model.log_rate.float()
        log_hazard = log_rate + math.log(float(horizon_days))
        if float(log_hazard.detach().cpu()) > 30:
            return 1.0, float(log_rate.detach().cpu()), float(log_hazard.detach().cpu())
        hazard = torch.exp(log_hazard)
        score = -torch.expm1(-hazard)
        return float(score.detach().cpu()), float(log_rate.detach().cpu()), float(log_hazard.detach().cpu())


def rollout_score(
    model,
    rows,
    target_stored_tokens,
    horizon_days: float,
    horizon_end_age: float,
    token_type_lookup,
    clinical_mask,
    args,
    device: str,
    rng: np.random.Generator,
):
    idx0, age0, token_type0 = collate_prefix(rows, device)
    target_model_tokens = {int(token) + 1 for token in target_stored_tokens}
    hits = 0
    event_counts = []
    cap_hits = 0
    dth_stops = 0
    age_reversals = 0
    final_ages = []
    for rollout_index in range(args.rollouts):
        torch.manual_seed(int(args.random_seed + rollout_index + rng.integers(0, 1_000_000)))
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.random_seed + rollout_index + 17))
        with torch.no_grad(), dtype_context(device, args.dtype):
            idx_out, age_out, type_out, _ = model.generate(
                idx0.clone(),
                age0.clone(),
                token_type0.clone(),
                max_new_tokens=args.max_new_tokens,
                max_age=horizon_end_age,
                no_repeat=False,
                termination_tokens=[],
                token_type_lookup=token_type_lookup,
                top_k=args.top_k,
                temperature=args.temperature,
                allowed_token_mask=clinical_mask,
                same_day_no_repeat=False,
                same_day_repeat_penalty=args.same_day_repeat_penalty,
                same_day_temperature=args.same_day_temperature,
                same_day_prob_cap=args.same_day_prob_cap,
            )
        generated_idx = idx_out[0, idx0.size(1):].detach().cpu().numpy().astype(int)
        generated_age = age_out[0, idx0.size(1):].detach().cpu().numpy().astype(float)
        generated_type = type_out[0, idx0.size(1):].detach().cpu().numpy().astype(int)
        if len(generated_age) > 1 and np.any(np.diff(generated_age) < -1e-5):
            age_reversals += 1
        within = generated_age <= horizon_end_age + 1e-5
        event_counts.append(int(within.sum()))
        final_ages.append(float(generated_age[-1]) if len(generated_age) else float(rows[-1, 1]))
        if len(generated_idx) >= args.max_new_tokens and (len(generated_age) == 0 or generated_age[-1] <= horizon_end_age):
            cap_hits += 1
        if np.any((generated_type == int(TokenType.DTH)) & within):
            dth_stops += 1
        hit = any(int(token) in target_model_tokens for token in generated_idx[within])
        hits += int(hit)
    score = hits / max(args.rollouts, 1)
    return {
        "score": float(score),
        "hit_rollouts": int(hits),
        "rollouts": int(args.rollouts),
        "events_mean": float(np.mean(event_counts)) if event_counts else 0.0,
        "events_p50": float(np.percentile(event_counts, 50)) if event_counts else 0.0,
        "events_p95": float(np.percentile(event_counts, 95)) if event_counts else 0.0,
        "events_max": int(max(event_counts)) if event_counts else 0,
        "cap_hit_rate": float(cap_hits / max(args.rollouts, 1)),
        "dth_stop_rate": float(dth_stops / max(args.rollouts, 1)),
        "age_reversal_rate": float(age_reversals / max(args.rollouts, 1)),
        "final_age_mean": float(np.mean(final_ages)) if final_ages else float(rows[-1, 1]),
    }


def binary_auroc(y_true, scores):
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_s[end] == sorted_s[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y_true, scores):
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2.0))


def score_summary(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "score_min": float(np.min(scores)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_zero_frac": float(np.mean(scores <= 1e-12)),
        "score_one_frac": float(np.mean(scores >= 1 - 1e-12)),
        "score_unique": int(len(np.unique(scores))),
    }


def summarize_predictions(predictions: pd.DataFrame):
    rows = []
    for keys, group in predictions.groupby(["horizon", "phenotype", "model"], sort=True):
        horizon, phenotype, model = keys
        y = group["label"].astype(int).to_numpy()
        scores = group["score"].astype(float).to_numpy()
        row = {
            "horizon": horizon,
            "phenotype": phenotype,
            "model": model,
            "n": int(len(group)),
            "positives": int(y.sum()),
            "auroc": binary_auroc(y, scores),
            "auprc": binary_auprc(y, scores),
        }
        row.update(score_summary(scores))
        for column in [
            "rollout_events_mean",
            "rollout_events_p50",
            "rollout_events_p95",
            "rollout_events_max",
            "cap_hit_rate",
            "dth_stop_rate",
            "age_reversal_rate",
        ]:
            if column in group:
                row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.fermat_ckpt = args.fermat_ckpt.expanduser().resolve()
    args.coupled_ckpt = args.coupled_ckpt.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(args.random_seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    started = time.time()
    log("## TASK27 PRIMARY DIRECT-RISK DRY RUN")
    log(f"fermat_ckpt={args.fermat_ckpt}")
    log(f"coupled_ckpt={args.coupled_ckpt}")
    log(f"output_dir={args.output_dir}")

    fermat, fermat_ckpt = load_model(args.fermat_ckpt, args.device)
    coupled, coupled_ckpt = load_model(args.coupled_ckpt, args.device)
    if coupled.config.decoupled_time_head:
        raise RuntimeError("coupled checkpoint has decoupled_time_head=True; expected coupled pure")
    if coupled.config.two_stage_time_head:
        raise RuntimeError("coupled checkpoint has two_stage_time_head=True; expected two_stage=False")

    registry, registry_path = load_registry(args.data_dir)
    token_type_lookup, token_label_lookup, clinical_mask = registry_maps(
        registry,
        int(fermat.config.vocab_size),
        args.device,
    )
    phenotype_tokens, mapping_frame, concept_map_path = load_phenotype_token_map(
        args.label_dir,
        registry,
        args.phenotypes,
    )
    labels, label_path = load_label_targets(args)
    patient_map = load_patient_map(args.data_dir)
    labels = labels.merge(patient_map, on=["person_id", "split"], how="left")
    labels = labels.dropna(subset=["patient_id_dense"]).copy()
    labels["patient_id_dense"] = labels["patient_id_dense"].astype(np.int64)
    split_data = load_split_data(args.data_dir, args.split)

    prediction_rows = []
    sample_rows = []
    for horizon in args.horizons:
        horizon_days = float(HORIZON_DAYS[horizon])
        for phenotype in args.phenotypes:
            eligible_col = f"eligible_{horizon}__{phenotype}"
            label_col = f"label_{horizon}__{phenotype}"
            task = labels.loc[labels[eligible_col].astype(bool)].copy()
            if len(task) > args.max_patients_per_task:
                task = task.sample(
                    n=args.max_patients_per_task,
                    random_state=args.random_seed + len(phenotype) + len(horizon),
                )
            task = task.sort_values("person_id")
            log(
                f"[TASK] horizon={horizon} phenotype={phenotype} "
                f"patients={len(task):,} positives={int(task[label_col].sum())}"
            )
            for row_index, row in enumerate(task.itertuples(index=False), 1):
                prefix = rows_before_index(
                    split_data,
                    int(row.patient_id_dense),
                    int(row.index_age_days),
                    max(int(fermat.config.block_size), int(coupled.config.block_size)),
                )
                if prefix is None or len(prefix) == 0:
                    continue
                tokens = phenotype_tokens[phenotype]
                label = int(getattr(row, label_col))
                coupled_started = time.time()
                c_score, c_log_rate, c_log_hazard = closed_form_score(
                    coupled,
                    prefix[-int(coupled.config.block_size):],
                    tokens,
                    horizon_days,
                    args.device,
                    args.dtype,
                )
                prediction_rows.append(
                    {
                        "horizon": horizon,
                        "phenotype": phenotype,
                        "person_id": int(row.person_id),
                        "label": label,
                        "model": "coupled_pure_closed_form",
                        "score": c_score,
                        "prefix_len": int(len(prefix)),
                        "runtime_sec": time.time() - coupled_started,
                        "closed_form_log_rate": c_log_rate,
                        "closed_form_log_hazard": c_log_hazard,
                    }
                )
                rollout_started = time.time()
                r = rollout_score(
                    fermat,
                    prefix[-int(fermat.config.block_size):],
                    tokens,
                    horizon_days,
                    float(row.index_age_days) + horizon_days,
                    token_type_lookup,
                    clinical_mask,
                    args,
                    args.device,
                    rng,
                )
                prediction_rows.append(
                    {
                        "horizon": horizon,
                        "phenotype": phenotype,
                        "person_id": int(row.person_id),
                        "label": label,
                        "model": "fermat_rollout",
                        "score": r["score"],
                        "prefix_len": int(len(prefix)),
                        "runtime_sec": time.time() - rollout_started,
                        "rollout_events_mean": r["events_mean"],
                        "rollout_events_p50": r["events_p50"],
                        "rollout_events_p95": r["events_p95"],
                        "rollout_events_max": r["events_max"],
                        "cap_hit_rate": r["cap_hit_rate"],
                        "dth_stop_rate": r["dth_stop_rate"],
                        "age_reversal_rate": r["age_reversal_rate"],
                        "hit_rollouts": r["hit_rollouts"],
                        "rollouts": r["rollouts"],
                    }
                )
                if row_index <= 3:
                    sample_rows.append(
                        {
                            "horizon": horizon,
                            "phenotype": phenotype,
                            "person_id": int(row.person_id),
                            "label": label,
                            "prefix_len": int(len(prefix)),
                            "coupled_score": c_score,
                            "fermat_score": r["score"],
                            "fermat_events_mean": r["events_mean"],
                            "fermat_cap_hit_rate": r["cap_hit_rate"],
                        }
                    )
                if row_index % 50 == 0:
                    log(f"  processed={row_index:,}/{len(task):,}")

    predictions = pd.DataFrame(prediction_rows)
    mapping_path = args.output_dir / "phenotype_token_mapping.csv"
    predictions_path = args.output_dir / "task27_dry_run_predictions.csv"
    summary_path = args.output_dir / "task27_dry_run_summary.csv"
    samples_path = args.output_dir / "task27_dry_run_sample_scores.csv"
    mapping_frame.to_csv(mapping_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    summary = summarize_predictions(predictions)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    manifest = {
        "fermat_ckpt": str(args.fermat_ckpt),
        "fermat_iter": int(fermat_ckpt.get("iter_num", fermat_ckpt.get("iter", -1))),
        "fermat_model_args": fermat_ckpt.get("model_args", {}),
        "coupled_ckpt": str(args.coupled_ckpt),
        "coupled_iter": int(coupled_ckpt.get("iter_num", coupled_ckpt.get("iter", -1))),
        "coupled_model_args": coupled_ckpt.get("model_args", {}),
        "data_dir": str(args.data_dir),
        "label_path": label_path,
        "concept_map_path": concept_map_path,
        "registry_path": registry_path,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizons": args.horizons,
        "phenotypes": args.phenotypes,
        "split": args.split,
        "max_patients_per_task": args.max_patients_per_task,
        "rollouts": args.rollouts,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "same_day_repeat_penalty": args.same_day_repeat_penalty,
        "same_day_temperature": args.same_day_temperature,
        "same_day_prob_cap": args.same_day_prob_cap,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "phenotype_token_mapping": str(mapping_path),
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "sample_scores": str(samples_path),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n## SUMMARY", flush=True)
    if summary.empty:
        print("(empty)")
    else:
        print(summary.to_string(index=False), flush=True)
    print("\n## SAMPLE SCORES", flush=True)
    if sample_rows:
        print(pd.DataFrame(sample_rows).to_string(index=False), flush=True)
    else:
        print("(none)", flush=True)
    print("\n## OUTPUTS", flush=True)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
