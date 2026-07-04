#!/usr/bin/env python3
"""Task25 generative smoke test for FERMAT.

This is a proof-of-concept sampler, not a calibrated clinical simulator. It
uses model.generate(), including the checkpoint's token head and decoupled
two-stage time head, to generate clinical future events.
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

from model import Fermat, FermatConfig, TokenType


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task25" / "outputs" / "generation_smoke_20260704"
MASK_TIME = -10000.0
CLINICAL_TYPES = {
    int(TokenType.DX): "DX",
    int(TokenType.RX): "RX",
    int(TokenType.PX): "PX",
    int(TokenType.DTH): "DTH",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-patients", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--min-prefix-tokens", type=int, default=10)
    parser.add_argument("--min-future-events", type=int, default=5)
    parser.add_argument("--stratify-by-clinical-rows-band", action="store_true")
    parser.add_argument("--frequency-baseline-max-events", type=int, default=2_000_000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--same-day-no-repeat", action="store_true")
    parser.add_argument("--same-day-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--same-day-temperature", type=float, default=1.0)
    parser.add_argument("--same-day-prob-cap", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--max-age-years", type=float, default=95.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.shape[0] % 4 == 0 and raw.shape[0] % 3 != 0:
        data = raw.reshape(-1, 4)
        has_types = True
    elif raw.shape[0] % 3 == 0 and raw.shape[0] % 4 != 0:
        data = raw.reshape(-1, 3)
        has_types = False
    else:
        data4 = raw.reshape(-1, 4)
        if data4[:, 3].max() < 20:
            data = data4
            has_types = True
        else:
            data = raw.reshape(-1, 3)
            has_types = False
    return data, has_types


def get_p2i(data):
    patient_ids = data[:, 0].astype(int)
    starts = []
    start = 0
    current = patient_ids[0]
    for index, patient_id in enumerate(patient_ids):
        if patient_id != current:
            starts.append([start, index - start])
            current = patient_id
            start = index
    starts.append([start, len(patient_ids) - start])
    return np.asarray(starts, dtype=np.int64)


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Fermat(FermatConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    critical_prefixes = ("transformer.", "lm_head.", "time_head.", "same_day_head.", "log_rate")
    critical_missing = [key for key in missing if key.startswith(critical_prefixes)]
    critical_unexpected = [key for key in unexpected if key.startswith(critical_prefixes)]
    if missing or unexpected:
        report = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "critical_missing_keys": critical_missing,
            "critical_unexpected_keys": critical_unexpected,
        }
        raise RuntimeError(
            "Checkpoint/model key mismatch; refusing to run generation with "
            "partially initialized weights:\n" + json.dumps(report, indent=2)
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


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


def registry_label(row):
    for key in ["token_key", "concept_name", "label", "source_value", "token"]:
        value = row.get(key)
        if value:
            return value
    return f"token_id={row.get('token_id')}"


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
        token_label_lookup[model_token_id] = registry_label(row)
        if token_type in CLINICAL_TYPES:
            clinical_mask[model_token_id] = True
    return token_type_lookup, token_label_lookup, clinical_mask


def clinical_future_count(patient, prefix_len):
    future = patient[prefix_len:]
    return int(np.isin(future[:, 3].astype(np.int64), list(CLINICAL_TYPES)).sum())


def clinical_rows_band(length):
    if length <= 512:
        return "le_512"
    if length <= 1024:
        return "513_1024"
    if length <= 2048:
        return "1025_2048"
    if length <= 4096:
        return "2049_4096"
    return "gt_4096"


def choose_seed_patients(data, args, rng):
    p2i = get_p2i(data)
    candidates = []
    for start, length in p2i:
        if length < args.min_prefix_tokens + 2:
            continue
        patient = data[int(start): int(start + length)]
        prefix_len = min(args.prefix_tokens, len(patient) - 1)
        if prefix_len < args.min_prefix_tokens:
            continue
        future_events = clinical_future_count(patient, prefix_len)
        if future_events >= args.min_future_events:
            candidates.append((int(start), int(length), int(prefix_len), int(future_events), clinical_rows_band(int(length))))
    if not candidates:
        raise RuntimeError(
            "No candidate seed patients found; lower --min-future-events or --min-prefix-tokens"
        )
    if not args.stratify_by_clinical_rows_band:
        order = rng.permutation(len(candidates))[: args.seed_patients]
        return [candidates[int(i)] for i in order]

    by_band = {}
    for candidate in candidates:
        by_band.setdefault(candidate[-1], []).append(candidate)
    bands = sorted(by_band)
    selected = []
    per_band = max(1, math.ceil(args.seed_patients / len(bands)))
    for band in bands:
        choices = by_band[band]
        order = rng.permutation(len(choices))[:per_band]
        selected.extend([choices[int(i)] for i in order])
    if len(selected) < args.seed_patients:
        selected_keys = {(row[0], row[1]) for row in selected}
        remaining = [row for row in candidates if (row[0], row[1]) not in selected_keys]
        order = rng.permutation(len(remaining))[: args.seed_patients - len(selected)]
        selected.extend([remaining[int(i)] for i in order])
    selected = selected[: args.seed_patients]
    rng.shuffle(selected)
    return selected


def collate_prefix(rows, device):
    idx = torch.from_numpy(rows[:, 2].astype(np.int64) + 1).unsqueeze(0).to(device)
    age = torch.from_numpy(rows[:, 1].astype(np.float32)).unsqueeze(0).to(device)
    token_type = torch.from_numpy(rows[:, 3].astype(np.int64)).unsqueeze(0).to(device)
    return idx, age, token_type


def dtype_context(device: str, dtype: str):
    if dtype == "float32" or device == "cpu":
        return nullcontext()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def top_k_sample(logits, top_k: int, temperature: float, rng: np.random.Generator):
    logits = logits.detach().float().cpu().numpy()
    logits = logits / max(float(temperature), 1e-6)
    finite = np.isfinite(logits)
    if not finite.any():
        raise RuntimeError("No finite logits to sample from")
    if top_k and top_k > 0:
        finite_indices = np.flatnonzero(finite)
        top = finite_indices[np.argsort(logits[finite_indices])[-top_k:]]
        mask = np.full_like(logits, False, dtype=bool)
        mask[top] = True
        finite = finite & mask
    values = logits[finite]
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / probs.sum()
    choices = np.flatnonzero(finite)
    return int(rng.choice(choices, p=probs))


def generate_one(model, prefix_rows, clinical_mask, token_type_lookup, token_label_lookup, args, device, generation_count):
    idx, age, token_type = collate_prefix(prefix_rows, device)
    original_len = idx.size(1)
    with torch.no_grad():
        with dtype_context(device, args.dtype):
            idx_out, age_out, type_out, _ = model.generate(
                idx,
                age,
                token_type,
                max_new_tokens=generation_count,
                max_age=args.max_age_years * 365.25,
                no_repeat=False,
                termination_tokens=[],
                token_type_lookup=token_type_lookup,
                top_k=args.top_k,
                temperature=args.temperature,
                allowed_token_mask=clinical_mask,
                same_day_no_repeat=args.same_day_no_repeat,
                same_day_repeat_penalty=args.same_day_repeat_penalty,
                same_day_temperature=args.same_day_temperature,
                same_day_prob_cap=args.same_day_prob_cap,
            )
    idx_values = idx_out[0, original_len:].detach().cpu().numpy().astype(int)
    age_values = age_out[0, original_len:].detach().cpu().numpy().astype(float)
    type_values = type_out[0, original_len:].detach().cpu().numpy().astype(int)
    previous_age = float(age[0, -1].detach().cpu().item())
    generated = []
    for step, (model_token, event_age, event_type) in enumerate(
        zip(idx_values, age_values, type_values),
        1,
    ):
        if model_token <= 0:
            continue
        generated.append(
            {
                "step": step,
                "token_id": int(model_token - 1),
                "model_token_id": int(model_token),
                "token_type_id": int(event_type),
                "token_type": CLINICAL_TYPES.get(int(event_type), str(int(event_type))),
                "age_in_days": float(event_age),
                "delta_days": max(float(event_age) - previous_age, 0.0),
                "token_label": token_label_lookup.get(int(model_token), f"token_id={int(model_token - 1)}"),
            }
        )
        previous_age = float(event_age)
        if int(event_type) == int(TokenType.DTH):
            break
    return generated


def build_frequency_baseline(data_dir: Path, args):
    data, has_types = load_data(data_dir / "train.bin")
    if not has_types:
        raise ValueError("train.bin is not a 4-column typed FERMAT bin")
    token_counts = {}
    gap_values = []
    seen_events = 0
    p2i = get_p2i(data)
    max_events = int(args.frequency_baseline_max_events)
    for start, length in p2i:
        patient = data[int(start): int(start + length)]
        clinical = patient[np.isin(patient[:, 3].astype(np.int64), list(CLINICAL_TYPES))]
        if len(clinical) == 0:
            continue
        tokens, counts = np.unique(clinical[:, 2].astype(np.int64), return_counts=True)
        for token, count in zip(tokens, counts):
            token_counts[int(token)] = token_counts.get(int(token), 0) + int(count)
        if len(clinical) > 1:
            gaps = np.clip(np.diff(clinical[:, 1].astype(np.int64)), 0, 365 * 80)
            gap_values.extend(gaps.tolist())
        seen_events += int(len(clinical))
        if max_events > 0 and seen_events >= max_events:
            break
    if not token_counts:
        raise RuntimeError("No clinical tokens found for frequency baseline")
    stored_tokens = np.asarray(sorted(token_counts), dtype=np.int64)
    counts = np.asarray([token_counts[int(t)] for t in stored_tokens], dtype=np.float64)
    probabilities = counts / counts.sum()
    gaps = np.asarray(gap_values if gap_values else [0, 1, 7, 30, 90, 365], dtype=np.float64)
    return stored_tokens, probabilities, gaps, seen_events


def frequency_future_rows(prefix_rows, generation_count, stored_tokens, probabilities, gaps, token_type_lookup, token_label_lookup, rng):
    previous_age = float(prefix_rows[-1, 1])
    sampled_tokens = rng.choice(stored_tokens, size=generation_count, replace=True, p=probabilities)
    sampled_gaps = rng.choice(gaps, size=generation_count, replace=True)
    rows = []
    for step, (stored_token, gap) in enumerate(zip(sampled_tokens, sampled_gaps), 1):
        model_token = int(stored_token) + 1
        event_age = previous_age + max(float(gap), 0.0)
        event_type = int(token_type_lookup.get(model_token, int(TokenType.DX)))
        rows.append(
            {
                "step": step,
                "token_id": int(stored_token),
                "model_token_id": model_token,
                "token_type_id": event_type,
                "token_type": CLINICAL_TYPES.get(event_type, str(event_type)),
                "age_in_days": float(event_age),
                "delta_days": max(float(gap), 0.0),
                "token_label": token_label_lookup.get(model_token, f"token_id={int(stored_token)}"),
            }
        )
        previous_age = event_age
        if event_type == int(TokenType.DTH):
            break
    return rows


def real_future_rows(patient_rows, prefix_len, max_new_tokens):
    future = patient_rows[prefix_len:]
    future = future[np.isin(future[:, 3].astype(np.int64), list(CLINICAL_TYPES))]
    future = future[:max_new_tokens]
    if len(future) == 0:
        return []
    previous_age = int(patient_rows[prefix_len - 1, 1])
    rows = []
    for step, row in enumerate(future, 1):
        age = int(row[1])
        rows.append(
            {
                "step": step,
                "token_id": int(row[2]),
                "token_type_id": int(row[3]),
                "token_type": CLINICAL_TYPES.get(int(row[3]), str(int(row[3]))),
                "age_in_days": age,
                "delta_days": max(age - previous_age, 0),
            }
        )
        previous_age = age
    return rows


def bucket_gap(days):
    if days == 0:
        return "0d"
    if days == 1:
        return "1d"
    if days <= 7:
        return "2_7d"
    if days <= 30:
        return "8_30d"
    if days <= 90:
        return "31_90d"
    if days <= 365:
        return "91_365d"
    return "gt_365d"


def sequence_quality(events: pd.DataFrame, label: str):
    if events.empty:
        return []
    rows = []
    by_seed = events.sort_values(["seed_index", "step"]).groupby("seed_index")
    total_transitions = 0
    age_reversals = 0
    repeat_transitions = 0
    for _, sub in by_seed:
        if len(sub) < 2:
            continue
        ages = sub["age_in_days"].to_numpy(dtype=float)
        tokens = sub["token_id"].to_numpy(dtype=int)
        total_transitions += len(sub) - 1
        age_reversals += int(np.sum(np.diff(ages) < 0))
        repeat_transitions += int(np.sum(tokens[1:] == tokens[:-1]))
    denom = max(total_transitions, 1)
    return [
        {"source": label, "summary_type": "quality", "name": "age_reversal_rate", "events": age_reversals, "fraction": age_reversals / denom},
        {"source": label, "summary_type": "quality", "name": "consecutive_repeat_rate", "events": repeat_transitions, "fraction": repeat_transitions / denom},
    ]


def source_diagnostics(events: pd.DataFrame, label: str):
    if events.empty:
        return []
    token_counts = events["token_id"].value_counts()
    top10_fraction = float(token_counts.head(10).sum() / len(events))
    dth_fraction = float((events["token_type"] == "DTH").mean())
    return [
        {"source": label, "summary_type": "diagnostic", "name": "unique_token_count", "events": int(token_counts.size), "fraction": np.nan},
        {"source": label, "summary_type": "diagnostic", "name": "top10_token_fraction", "events": int(token_counts.head(10).sum()), "fraction": top10_fraction},
        {"source": label, "summary_type": "diagnostic", "name": "dth_fraction", "events": int((events["token_type"] == "DTH").sum()), "fraction": dth_fraction},
    ]


def summarize_events(events: pd.DataFrame, label: str):
    if events.empty:
        return []
    rows = []
    for token_type, sub in events.groupby("token_type", dropna=False):
        rows.append(
            {
                "source": label,
                "summary_type": "token_type",
                "name": str(token_type),
                "events": int(len(sub)),
                "fraction": float(len(sub) / len(events)),
            }
        )
    events = events.copy()
    events["gap_bucket"] = events["delta_days"].fillna(0).map(bucket_gap)
    for bucket, sub in events.groupby("gap_bucket", dropna=False):
        rows.append(
            {
                "source": label,
                "summary_type": "gap_bucket",
                "name": str(bucket),
                "events": int(len(sub)),
                "fraction": float(len(sub) / len(events)),
            }
        )
    return rows


def distribution_vector(frame: pd.DataFrame, column: str, labels):
    if frame.empty:
        return np.zeros(len(labels), dtype=np.float64)
    counts = frame[column].value_counts()
    values = np.asarray([counts.get(label, 0) for label in labels], dtype=np.float64)
    total = values.sum()
    if total == 0:
        return values
    return values / total


def js_divergence(p, q):
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(np.where(p > 0, p * np.log2((p + eps) / (m + eps)), 0.0))
        + 0.5 * np.sum(np.where(q > 0, q * np.log2((q + eps) / (m + eps)), 0.0))
    )


def distribution_distances(frames):
    real = frames.get("real_future", pd.DataFrame())
    rows = []
    for column in ["token_id", "token_type", "gap_bucket"]:
        prepared = {}
        labels = set()
        for source, frame in frames.items():
            if frame.empty:
                prepared[source] = frame
                continue
            work = frame.copy()
            if column == "gap_bucket":
                work[column] = work["delta_days"].fillna(0).map(bucket_gap)
            prepared[source] = work
            labels.update(work[column].dropna().unique().tolist())
        labels = sorted(labels, key=lambda x: str(x))
        real_vec = distribution_vector(prepared.get("real_future", pd.DataFrame()), column, labels)
        for source, frame in prepared.items():
            if source == "real_future":
                continue
            vec = distribution_vector(frame, column, labels)
            rows.append(
                {
                    "comparison": f"{source}_vs_real_future",
                    "feature": column,
                    "total_variation": float(0.5 * np.abs(vec - real_vec).sum()),
                    "js_divergence": js_divergence(vec, real_vec),
                }
            )
    return pd.DataFrame(rows)


def repeat_rate(events: pd.DataFrame):
    if events.empty:
        return np.nan
    total = 0
    repeats = 0
    for _, sub in events.sort_values(["seed_index", "step"]).groupby("seed_index"):
        if len(sub) < 2:
            continue
        tokens = sub["token_id"].to_numpy(dtype=int)
        total += len(sub) - 1
        repeats += int(np.sum(tokens[1:] == tokens[:-1]))
    return repeats / total if total else np.nan


def scalar_source_metrics(frames):
    rows = []
    real = frames.get("real_future", pd.DataFrame())
    real_same_day = float((real["delta_days"] == 0).mean()) if not real.empty else np.nan
    real_repeat = repeat_rate(real)
    real_top10 = (
        float(real["token_id"].value_counts().head(10).sum() / len(real))
        if not real.empty
        else np.nan
    )
    for source in ["generated_fermat", "generated_frequency"]:
        frame = frames.get(source, pd.DataFrame())
        if frame.empty:
            continue
        same_day = float((frame["delta_days"] == 0).mean())
        repeat = repeat_rate(frame)
        top10 = float(frame["token_id"].value_counts().head(10).sum() / len(frame))
        rows.extend(
            [
                {
                    "comparison": f"{source}_vs_real_future",
                    "metric": "same_day_abs_error",
                    "value": abs(same_day - real_same_day),
                },
                {
                    "comparison": f"{source}_vs_real_future",
                    "metric": "consecutive_repeat_abs_error",
                    "value": abs(repeat - real_repeat),
                },
                {
                    "comparison": f"{source}_vs_real_future",
                    "metric": "top10_fraction_abs_error",
                    "value": abs(top10 - real_top10),
                },
            ]
        )
    return pd.DataFrame(rows)


def source_distance_metrics(frames):
    distance = distribution_distances(frames)
    rows = []
    for _, row in distance.iterrows():
        rows.append(
            {
                "comparison": row["comparison"],
                "metric": f"{row['feature']}_total_variation",
                "value": float(row["total_variation"]),
            }
        )
        rows.append(
            {
                "comparison": row["comparison"],
                "metric": f"{row['feature']}_js_divergence",
                "value": float(row["js_divergence"]),
            }
        )
    scalar = scalar_source_metrics(frames)
    if not scalar.empty:
        rows.extend(scalar.to_dict("records"))
    return pd.DataFrame(rows)


def bootstrap_source_comparison(frames, seed_ids, rng, n_bootstrap):
    observed = source_distance_metrics(frames)
    if n_bootstrap <= 0 or len(seed_ids) == 0:
        return observed.assign(ci95_lower=np.nan, ci95_upper=np.nan, bootstrap_samples=0)

    metrics = sorted(observed["metric"].unique())
    boot_values = {metric: [] for metric in metrics}
    seed_ids = np.asarray(seed_ids)
    for _ in range(n_bootstrap):
        sampled = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        sampled_frames = {}
        for source, frame in frames.items():
            parts = []
            for draw_index, seed_id in enumerate(sampled):
                sub = frame[frame["seed_index"] == seed_id].copy()
                if sub.empty:
                    continue
                sub["seed_index"] = draw_index + 1
                parts.append(sub)
            sampled_frames[source] = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
        sampled_metrics = source_distance_metrics(sampled_frames)
        pivot = sampled_metrics.pivot(index="metric", columns="comparison", values="value")
        for metric in metrics:
            if (
                metric in pivot.index
                and "generated_fermat_vs_real_future" in pivot.columns
                and "generated_frequency_vs_real_future" in pivot.columns
            ):
                value = (
                    pivot.loc[metric, "generated_frequency_vs_real_future"]
                    - pivot.loc[metric, "generated_fermat_vs_real_future"]
                )
                boot_values[metric].append(float(value))
    pivot_observed = observed.pivot(index="metric", columns="comparison", values="value")
    rows = []
    for metric in metrics:
        if (
            metric not in pivot_observed.index
            or "generated_fermat_vs_real_future" not in pivot_observed.columns
            or "generated_frequency_vs_real_future" not in pivot_observed.columns
        ):
            continue
        delta = (
            pivot_observed.loc[metric, "generated_frequency_vs_real_future"]
            - pivot_observed.loc[metric, "generated_fermat_vs_real_future"]
        )
        values = np.asarray(boot_values.get(metric, []), dtype=np.float64)
        if values.size:
            lo, hi = np.quantile(values, [0.025, 0.975])
        else:
            lo, hi = np.nan, np.nan
        rows.append(
            {
                "metric": metric,
                "fermat_vs_real": float(pivot_observed.loc[metric, "generated_fermat_vs_real_future"]),
                "frequency_vs_real": float(pivot_observed.loc[metric, "generated_frequency_vs_real_future"]),
                "frequency_minus_fermat": float(delta),
                "ci95_lower": float(lo),
                "ci95_upper": float(hi),
                "bootstrap_samples": int(n_bootstrap),
                "fermat_closer": bool(delta > 0),
                "significant_fermat_closer": bool(lo > 0) if values.size else False,
                "significant_frequency_closer": bool(hi < 0) if values.size else False,
                "winner": (
                    "FERMAT"
                    if values.size and lo > 0
                    else "frequency"
                    if values.size and hi < 0
                    else "tie"
                ),
            }
        )
    return pd.DataFrame(rows)


def infer_early_stop_reason(generated, target_count):
    if len(generated) >= target_count:
        return "matched_count"
    if generated and generated[-1].get("token_type") == "DTH":
        return "dth"
    return "short_generation"


def numeric_summary(series, prefix):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        f"{prefix}_p50": float(values.quantile(0.50)),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_p99": float(values.quantile(0.99)),
        f"{prefix}_max": float(values.max()),
    }


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    rng = np.random.default_rng(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    device = args.device
    started = time.time()

    model, checkpoint = load_model(args.ckpt, device)
    registry, registry_path = load_registry(args.data_dir)
    token_type_lookup, token_label_lookup, clinical_mask = registry_maps(
        registry,
        int(model.config.vocab_size),
        device,
    )
    data_path = args.data_dir / f"{args.split}.bin"
    data, has_types = load_data(data_path)
    if not has_types:
        raise ValueError(f"{data_path} is not a 4-column typed FERMAT bin")
    seeds = choose_seed_patients(data, args, rng)
    freq_tokens, freq_probs, freq_gaps, freq_events = build_frequency_baseline(args.data_dir, args)
    model_args = checkpoint.get("model_args", {})
    log(f"checkpoint_iter={checkpoint.get('iter_num')}")
    log("model_time_config=" + json.dumps({k: model_args.get(k) for k in ["t_min", "mask_ties", "decoupled_time_head", "two_stage_time_head"]}, sort_keys=True))
    log(f"split={args.split} rows={len(data):,} seed_patients={len(seeds):,}")
    log(f"vocab_size={model.config.vocab_size} clinical_tokens={int(clinical_mask.sum().item()):,}")
    log(f"frequency_baseline_events={freq_events:,} unique_tokens={len(freq_tokens):,} gap_samples={len(freq_gaps):,}")

    generated_rows = []
    frequency_rows = []
    real_rows = []
    seed_rows = []
    for seed_index, (start, length, prefix_len, candidate_future_events, band) in enumerate(seeds, 1):
        patient = data[start : start + length]
        if prefix_len < args.min_prefix_tokens:
            continue
        prefix = patient[:prefix_len]
        person_dense = int(patient[0, 0])
        real = real_future_rows(patient, prefix_len, args.max_new_tokens)
        generation_count = len(real)
        if generation_count == 0:
            continue
        generated = generate_one(
            model,
            prefix,
            clinical_mask,
            token_type_lookup,
            token_label_lookup,
            args,
            device,
            generation_count,
        )
        frequency = frequency_future_rows(
            prefix,
            generation_count,
            freq_tokens,
            freq_probs,
            freq_gaps,
            token_type_lookup,
            token_label_lookup,
            rng,
        )
        generated_stop_reason = infer_early_stop_reason(generated, generation_count)
        frequency_stop_reason = infer_early_stop_reason(frequency, generation_count)
        seed_rows.append(
            {
                "seed_index": seed_index,
                "patient_id_dense": person_dense,
                "patient_length": int(length),
                "clinical_rows_band": band,
                "prefix_len": int(prefix_len),
                "prefix_last_age_days": int(prefix[-1, 1]),
                "candidate_future_events": int(candidate_future_events),
                "matched_event_count": int(generation_count),
                "generated_fermat_events": int(len(generated)),
                "generated_frequency_events": int(len(frequency)),
                "real_future_events": int(len(real)),
                "generated_fermat_count_match": bool(len(generated) == len(real)),
                "generated_frequency_count_match": bool(len(frequency) == len(real)),
                "generated_fermat_early_stop_reason": generated_stop_reason,
                "generated_frequency_early_stop_reason": frequency_stop_reason,
            }
        )
        for row in generated:
            row.update({"seed_index": seed_index, "patient_id_dense": person_dense, "source": "generated_fermat"})
            generated_rows.append(row)
        for row in frequency:
            row.update({"seed_index": seed_index, "patient_id_dense": person_dense, "source": "generated_frequency"})
            frequency_rows.append(row)
        for row in real:
            row.update({"seed_index": seed_index, "patient_id_dense": person_dense, "source": "real_future"})
            real_rows.append(row)
        if seed_index % 25 == 0:
            log(f"[generated] seeds={seed_index:,}/{len(seeds):,}")

    generated_df = pd.DataFrame(generated_rows)
    frequency_df = pd.DataFrame(frequency_rows)
    real_df = pd.DataFrame(real_rows)
    seeds_df = pd.DataFrame(seed_rows)
    summary_rows = []
    summary_rows.extend(summarize_events(generated_df, "generated_fermat"))
    summary_rows.extend(summarize_events(frequency_df, "generated_frequency"))
    summary_rows.extend(summarize_events(real_df, "real_future"))
    summary_rows.extend(sequence_quality(generated_df, "generated_fermat"))
    summary_rows.extend(sequence_quality(frequency_df, "generated_frequency"))
    summary_rows.extend(sequence_quality(real_df, "real_future"))
    summary_rows.extend(source_diagnostics(generated_df, "generated_fermat"))
    summary_rows.extend(source_diagnostics(frequency_df, "generated_frequency"))
    summary_rows.extend(source_diagnostics(real_df, "real_future"))
    summary = pd.DataFrame(summary_rows)
    distance = distribution_distances(
        {
            "generated_fermat": generated_df,
            "generated_frequency": frequency_df,
            "real_future": real_df,
        }
    )
    bootstrap = bootstrap_source_comparison(
        {
            "generated_fermat": generated_df,
            "generated_frequency": frequency_df,
            "real_future": real_df,
        },
        sorted(seeds_df["seed_index"].unique().tolist()) if not seeds_df.empty else [],
        rng,
        args.bootstrap_samples,
    )
    seed_band_summary = (
        seeds_df.groupby("clinical_rows_band", dropna=False)
        .agg(
            seed_patients=("seed_index", "size"),
            median_patient_length=("patient_length", "median"),
            p95_patient_length=("patient_length", lambda x: float(pd.Series(x).quantile(0.95))),
            p99_patient_length=("patient_length", lambda x: float(pd.Series(x).quantile(0.99))),
            max_patient_length=("patient_length", "max"),
            median_candidate_future_events=("candidate_future_events", "median"),
            p95_candidate_future_events=("candidate_future_events", lambda x: float(pd.Series(x).quantile(0.95))),
            p99_candidate_future_events=("candidate_future_events", lambda x: float(pd.Series(x).quantile(0.99))),
            max_candidate_future_events=("candidate_future_events", "max"),
            median_matched_event_count=("matched_event_count", "median"),
            generated_fermat_count_match_rate=("generated_fermat_count_match", "mean"),
            generated_frequency_count_match_rate=("generated_frequency_count_match", "mean"),
        )
        .reset_index()
        if not seeds_df.empty
        else pd.DataFrame()
    )
    seed_length_summary = pd.DataFrame(
        [
            {
                "seed_patients": int(len(seeds_df)),
                **numeric_summary(seeds_df["patient_length"], "patient_length"),
                **numeric_summary(seeds_df["candidate_future_events"], "candidate_future_events"),
                **numeric_summary(seeds_df["matched_event_count"], "matched_event_count"),
                "generated_fermat_count_match_rate": float(seeds_df["generated_fermat_count_match"].mean()) if not seeds_df.empty else np.nan,
                "generated_frequency_count_match_rate": float(seeds_df["generated_frequency_count_match"].mean()) if not seeds_df.empty else np.nan,
            }
        ]
    )

    generated_path = args.output_dir / "generated_events.csv"
    frequency_path = args.output_dir / "generated_frequency_events.csv"
    real_path = args.output_dir / "real_future_events.csv"
    seeds_path = args.output_dir / "seed_patients.csv"
    summary_path = args.output_dir / "generation_summary.csv"
    distance_path = args.output_dir / "distribution_distances.csv"
    bootstrap_path = args.output_dir / "bootstrap_source_comparison.csv"
    seed_band_path = args.output_dir / "seed_band_summary.csv"
    seed_length_summary_path = args.output_dir / "seed_length_summary.csv"
    generated_df.to_csv(generated_path, index=False)
    frequency_df.to_csv(frequency_path, index=False)
    real_df.to_csv(real_path, index=False)
    seeds_df.to_csv(seeds_path, index=False)
    summary.to_csv(summary_path, index=False)
    distance.to_csv(distance_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    seed_band_summary.to_csv(seed_band_path, index=False)
    seed_length_summary.to_csv(seed_length_summary_path, index=False)
    manifest = {
        "ckpt": str(args.ckpt),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "data_dir": str(args.data_dir),
        "registry": registry_path,
        "split": args.split,
        "output_dir": str(args.output_dir),
        "seed_patients": args.seed_patients,
        "max_new_tokens": args.max_new_tokens,
        "min_future_events": args.min_future_events,
        "prefix_tokens": args.prefix_tokens,
        "stratify_by_clinical_rows_band": bool(args.stratify_by_clinical_rows_band),
        "top_k": args.top_k,
        "temperature": args.temperature,
        "same_day_no_repeat": bool(args.same_day_no_repeat),
        "same_day_repeat_penalty": args.same_day_repeat_penalty,
        "same_day_temperature": args.same_day_temperature,
        "same_day_prob_cap": args.same_day_prob_cap,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
        "dtype": args.dtype,
        "time_sampling": "model.generate decoupled/two-stage time head",
        "model_time_config": {k: model_args.get(k) for k in ["t_min", "mask_ties", "decoupled_time_head", "two_stage_time_head"]},
        "frequency_baseline": {
            "train_clinical_events_seen": int(freq_events),
            "unique_tokens": int(len(freq_tokens)),
            "gap_samples": int(len(freq_gaps)),
            "max_events": int(args.frequency_baseline_max_events),
        },
        "interpretation": "Smoke test only; FERMAT token and time sampling use model.generate(); generated_frequency is a train-frequency clinical token/gap baseline.",
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "generated_events": str(generated_path),
            "generated_frequency_events": str(frequency_path),
            "real_future_events": str(real_path),
            "seed_patients": str(seeds_path),
            "generation_summary": str(summary_path),
            "distribution_distances": str(distance_path),
            "bootstrap_source_comparison": str(bootstrap_path),
            "seed_band_summary": str(seed_band_path),
            "seed_length_summary": str(seed_length_summary_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
