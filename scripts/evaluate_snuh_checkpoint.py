"""Evaluate an SNUH FERMAT checkpoint on deterministic trajectory windows."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import (
    Fermat,
    FermatConfig,
    TokenType,
    align_time_deltas,
    build_attention_mask,
    build_target_mask,
)
from utils import get_p2i, load_data


MASK_TIME = -10000.0
CLINICAL_TYPES = {
    int(TokenType.DX): "DX",
    int(TokenType.RX): "RX",
    int(TokenType.PX): "PX",
    int(TokenType.DTH): "DTH",
}
NON_CLINICAL_TYPES = {
    int(TokenType.PAD),
    int(TokenType.SEX),
    int(TokenType.NO_EVENT),
    int(TokenType.LAB),
    int(TokenType.LIFESTYLE),
    int(TokenType.GENOMICS),
}
FREQUENCY_BUCKETS = [
    ("head", 10000, None),
    ("medium", 1000, 9999),
    ("tail", 100, 999),
    ("rare", 1, 99),
    ("unseen_near_rare", 0, 0),
]
AGE_GROUPS = [
    ("age_000_017", 0, 18 * 365.25),
    ("age_018_039", 18 * 365.25, 40 * 365.25),
    ("age_040_064", 40 * 365.25, 65 * 365.25),
    ("age_065_079", 65 * 365.25, 80 * 365.25),
    ("age_080_plus", 80 * 365.25, None),
]
SEQUENCE_LENGTH_BUCKETS = [
    ("seq_000_127", 0, 127),
    ("seq_128_511", 128, 511),
    ("seq_512_1023", 512, 1023),
    ("seq_1024_plus", 1024, None),
]
VISIT_DENSITY_BUCKETS = [
    ("density_000_004_per_year", 0, 4),
    ("density_005_019_per_year", 5, 19),
    ("density_020_049_per_year", 20, 49),
    ("density_050_plus_per_year", 50, None),
]
TIME_HORIZON_BUCKETS = [
    ("0_days", 0, 0),
    ("1_7_days", 1, 7),
    ("8_30_days", 8, 30),
    ("31_90_days", 31, 90),
    ("91_365_days", 91, 365),
    ("over_365_days", 366, None),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["left", "middle", "right"],
        choices=["left", "middle", "right"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--time-baseline-max-patients", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Fermat(FermatConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    # Reconcile architecture differences (coupled log_rate vs decoupled
    # time_head, or pre-fix checkpoints): drop keys the model does not have and
    # backfill the ones it expects but the checkpoint lacks.
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    for missing in model_state.keys() - state_dict.keys():
        state_dict[missing] = model_state[missing]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def load_registry(data_dir):
    for filename in ("token_registry.csv", "vocab.csv"):
        path = data_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError("Expected token_registry.csv or vocab.csv")


def registry_type(row):
    value = row.get("token_type_id", row.get("token_type"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(TokenType[row["token_type"]])


def clinical_output_mask(registry, vocab_size, device):
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for row in registry:
        if registry_type(row) in CLINICAL_TYPES:
            model_token_id = int(row["token_id"]) + 1
            if model_token_id < vocab_size:
                mask[model_token_id] = True
    return mask


def build_clinical_unigram(data, clinical_vocab_mask, alpha=1.0):
    """Build an add-one-smoothed clinical-token baseline from train data."""
    clinical_types = np.array(list(CLINICAL_TYPES), dtype=np.int64)
    target_rows = np.zeros(len(data), dtype=bool)
    target_rows[1:] = data[1:, 0] == data[:-1, 0]
    rows = target_rows & np.isin(
        data[:, 3].astype(np.int64),
        clinical_types,
    )
    model_token_ids = data[rows, 2].astype(np.int64) + 1
    counts = np.bincount(
        model_token_ids,
        minlength=clinical_vocab_mask.numel(),
    )[:clinical_vocab_mask.numel()]
    counts = torch.as_tensor(
        counts,
        dtype=torch.float64,
        device=clinical_vocab_mask.device,
    )
    smoothed = counts + alpha * clinical_vocab_mask.to(torch.float64)
    smoothed = smoothed.masked_fill(~clinical_vocab_mask, 0)
    total = smoothed.sum()
    if total <= 0:
        raise ValueError("No clinical tokens were found in the train split")
    log_probs = torch.full_like(smoothed, -torch.inf)
    log_probs[clinical_vocab_mask] = torch.log(
        smoothed[clinical_vocab_mask] / total
    )
    top1_token = int(torch.argmax(smoothed).item())
    return log_probs, top1_token


def build_train_token_counts(data, vocab_size):
    model_token_ids = data[:, 2].astype(np.int64) + 1
    return np.bincount(model_token_ids, minlength=vocab_size)[:vocab_size]


def window_start(length, block_size, selector):
    available = max(length - block_size - 1, 0)
    if selector == "left":
        return 0
    if selector == "middle":
        return available // 2
    if selector == "right":
        return available
    raise ValueError(selector)


def iter_windows(data, block_size, selectors, max_patients=None):
    p2i = get_p2i(data)
    if max_patients is not None:
        p2i = p2i[:max_patients]
    for patient_start, patient_length in p2i:
        patient = data[
            int(patient_start):int(patient_start + patient_length)
        ]
        if len(patient) < 2:
            continue
        sex_tokens = patient[patient[:, 3] == int(TokenType.SEX), 2]
        sex_token = int(sex_tokens[0]) if sex_tokens.size else None
        age_span_years = max(
            (int(patient[:, 1].max()) - int(patient[:, 1].min())) / 365.25,
            1.0 / 365.25,
        )
        visit_density = float(len(patient) / age_span_years)
        earliest_age = {}
        for row in patient:
            token_id = int(row[2])
            age = int(row[1])
            earliest_age[token_id] = min(earliest_age.get(token_id, age), age)
        for selector in selectors:
            start = window_start(len(patient), block_size, selector)
            window = patient[start:start + block_size + 1]
            repeated = np.array(
                [
                    earliest_age[int(row[2])] < int(row[1])
                    for row in window[1:]
                ],
                dtype=bool,
            )
            yield {
                "selector": selector,
                "patient_id": int(patient[0, 0]),
                "rows": window,
                "repeated": repeated,
                "sex_token": sex_token,
                "patient_length": int(len(patient)),
                "visit_density_per_year": visit_density,
            }


def collate(windows, device):
    max_targets = max(len(window["rows"]) - 1 for window in windows)
    batch_size = len(windows)
    x = torch.zeros((batch_size, max_targets), dtype=torch.long)
    y = torch.zeros_like(x)
    a = torch.full((batch_size, max_targets), MASK_TIME)
    b = torch.full_like(a, MASK_TIME)
    xt = torch.full_like(x, int(TokenType.PAD))
    yt = torch.full_like(x, int(TokenType.PAD))
    repeated = torch.zeros((batch_size, max_targets), dtype=torch.bool)

    for index, window in enumerate(windows):
        rows = window["rows"]
        length = len(rows) - 1
        raw_tokens = torch.from_numpy(rows[:, 2].astype(np.int64))
        raw_ages = torch.from_numpy(rows[:, 1].astype(np.float32))
        raw_types = torch.from_numpy(rows[:, 3].astype(np.int64))
        x[index, :length] = raw_tokens[:-1] + 1
        y[index, :length] = raw_tokens[1:] + 1
        a[index, :length] = raw_ages[:-1]
        b[index, :length] = raw_ages[1:]
        xt[index, :length] = raw_types[:-1]
        yt[index, :length] = raw_types[1:]
        repeated[index, :length] = torch.from_numpy(window["repeated"])

    return tuple(
        tensor.to(device)
        for tensor in (x, a, y, b, xt, yt, repeated)
    )


def collate_metadata(windows, max_targets, device):
    patient_ids = torch.full((len(windows), max_targets), -1, dtype=torch.long)
    sex_tokens = torch.full_like(patient_ids, -1)
    patient_lengths = torch.zeros_like(patient_ids)
    visit_density = torch.zeros((len(windows), max_targets), dtype=torch.float32)
    for index, window in enumerate(windows):
        length = len(window["rows"]) - 1
        patient_ids[index, :length] = int(window["patient_id"])
        sex_token = -1 if window["sex_token"] is None else int(window["sex_token"])
        sex_tokens[index, :length] = sex_token
        patient_lengths[index, :length] = int(window["patient_length"])
        visit_density[index, :length] = float(window["visit_density_per_year"])
    return tuple(
        tensor.to(device)
        for tensor in (patient_ids, sex_tokens, patient_lengths, visit_density)
    )


def clinical_target_mask(targets, target_types):
    mask = targets > 0
    for token_type in NON_CLINICAL_TYPES:
        mask &= target_types != token_type
    return mask


def collect_clinical_waiting_gaps(
    data,
    block_size,
    selectors,
    batch_size,
    max_patients,
    mask_ties,
):
    chunks = []
    buffer = []

    def collect(windows):
        x, age, targets, target_age, _, target_types, _ = collate(
            windows,
            "cpu",
        )
        attention_mask = build_attention_mask(
            x,
            age,
            targets_age=target_age,
            mask_ties=mask_ties,
        )
        actual_dt = align_time_deltas(
            age,
            target_age,
            attention_mask,
            mask_ties,
        )
        mask = clinical_target_mask(targets, target_types) & (target_age > age)
        if mask.any():
            chunks.append(actual_dt[mask].float().numpy())

    for window in iter_windows(
        data,
        block_size,
        selectors,
        max_patients,
    ):
        buffer.append(window)
        if len(buffer) < batch_size:
            continue
        collect(buffer)
        buffer = []
    if buffer:
        collect(buffer)

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def build_train_waiting_baseline(
    train_data,
    block_size,
    selectors,
    batch_size,
    max_patients,
    mask_ties,
):
    gaps = collect_clinical_waiting_gaps(
        train_data,
        block_size,
        selectors,
        batch_size,
        max_patients,
        mask_ties,
    )
    if not gaps.size:
        return None
    return {
        "source_split": "train",
        "targets": int(gaps.size),
        "mean_gap_days": float(gaps.mean()),
        "median_gap_days": float(np.median(gaps)),
    }


def update_accuracy(stats, logits, targets, mask, prefix):
    count = int(mask.sum())
    if count == 0:
        return
    selected_logits = logits[mask].float()
    selected_targets = targets[mask]
    stats[f"{prefix}_ce_sum"] += float(
        F.cross_entropy(selected_logits, selected_targets, reduction="sum")
    )
    stats[f"{prefix}_count"] += count
    max_k = min(10, selected_logits.shape[-1])
    topk = torch.topk(selected_logits, k=max_k, dim=-1).indices
    for k in (1, 5, 10):
        use_k = min(k, max_k)
        stats[f"{prefix}_top{k}"] += int(
            (topk[:, :use_k] == selected_targets[:, None])
            .any(dim=-1)
            .sum()
        )


def update_group_accuracy(stats, logits, targets, base_mask, groups, prefix):
    for name, group_mask in groups.items():
        update_accuracy(
            stats,
            logits,
            targets,
            base_mask & group_mask,
            f"{prefix}_{name}",
        )


def finalize_accuracy(stats, prefix):
    count = int(stats[f"{prefix}_count"])
    if count == 0:
        return {"targets": 0}
    ce = stats[f"{prefix}_ce_sum"] / count
    return {
        "targets": count,
        "cross_entropy": ce,
        "perplexity": math.exp(ce) if ce < 700 else float("inf"),
        "top1_accuracy": stats[f"{prefix}_top1"] / count,
        "top5_accuracy": stats[f"{prefix}_top5"] / count,
        "top10_accuracy": stats[f"{prefix}_top10"] / count,
    }


def finalize_group_accuracy(stats, prefix, names):
    return {name: finalize_accuracy(stats, f"{prefix}_{name}") for name in names}


def observed_group_names(stats, prefix):
    marker = f"{prefix}_"
    suffix = "_count"
    names = []
    for key in stats:
        if key.startswith(marker) and key.endswith(suffix):
            names.append(key[len(marker):-len(suffix)])
    return sorted(set(names))


def range_masks(values, ranges):
    masks = {}
    for name, lower, upper in ranges:
        mask = values >= lower
        if upper is not None:
            mask &= values <= upper
        masks[name] = mask
    return masks


def frequency_bucket_masks(targets, token_counts):
    counts = torch.as_tensor(
        token_counts,
        dtype=torch.long,
        device=targets.device,
    )
    clipped = targets.clamp(min=0, max=counts.numel() - 1)
    target_counts = counts[clipped]
    masks = {}
    for name, lower, upper in FREQUENCY_BUCKETS:
        mask = target_counts >= lower
        if upper is not None:
            mask &= target_counts <= upper
        masks[name] = mask
    return masks


def update_unigram(stats, log_probs, top1_token, targets, mask):
    selected_targets = targets[mask]
    count = int(selected_targets.numel())
    if count == 0:
        return
    stats["unigram_ce_sum"] += float(
        -log_probs[selected_targets].sum().item()
    )
    stats["unigram_top1"] += int((selected_targets == top1_token).sum())
    stats["unigram_count"] += count


def finalize_unigram(stats):
    count = int(stats["unigram_count"])
    if count == 0:
        return {"targets": 0}
    ce = stats["unigram_ce_sum"] / count
    return {
        "targets": count,
        "cross_entropy": ce,
        "perplexity": math.exp(ce) if ce < 700 else float("inf"),
        "top1_accuracy": stats["unigram_top1"] / count,
    }


def patient_record(patient_acc, patient_id):
    return patient_acc[int(patient_id)]


def update_patient_clinical(patient_acc, logits, targets, mask, patient_ids):
    if not mask.any():
        return
    selected_logits = logits[mask].float()
    selected_targets = targets[mask]
    selected_patients = patient_ids[mask]
    ce = F.cross_entropy(selected_logits, selected_targets, reduction="none")
    max_k = min(10, selected_logits.shape[-1])
    topk = torch.topk(selected_logits, k=max_k, dim=-1).indices
    top1 = (topk[:, :1] == selected_targets[:, None]).any(dim=-1)
    top5 = (topk[:, :min(5, max_k)] == selected_targets[:, None]).any(dim=-1)
    top10 = (topk[:, :min(10, max_k)] == selected_targets[:, None]).any(dim=-1)
    for patient_id, loss, hit1, hit5, hit10 in zip(
        selected_patients.detach().cpu().numpy(),
        ce.detach().cpu().numpy(),
        top1.detach().cpu().numpy(),
        top5.detach().cpu().numpy(),
        top10.detach().cpu().numpy(),
    ):
        record = patient_record(patient_acc, patient_id)
        record["clinical_ce_sum"] += float(loss)
        record["clinical_count"] += 1
        record["clinical_top1"] += int(hit1)
        record["clinical_top5"] += int(hit5)
        record["clinical_top10"] += int(hit10)


def update_patient_same_day(patient_acc, patient_ids, labels, probabilities):
    for patient_id, label, probability in zip(
        patient_ids.detach().cpu().numpy(),
        labels.detach().cpu().numpy(),
        probabilities.detach().cpu().numpy(),
    ):
        record = patient_record(patient_acc, patient_id)
        record["same_day_count"] += 1
        record["same_day_positive"] += int(label)
        record["same_day_brier_sum"] += float((probability - label) ** 2)


def update_patient_time(patient_acc, patient_ids, errors, nll):
    for patient_id, error, target_nll in zip(
        patient_ids.detach().cpu().numpy(),
        errors.detach().cpu().numpy(),
        nll.detach().cpu().numpy(),
    ):
        record = patient_record(patient_acc, patient_id)
        record["time_count"] += 1
        record["time_error_sum"] += float(error)
        record["time_nll_sum"] += float(target_nll)


def summarize_patient_sample(records):
    clinical_count = sum(record["clinical_count"] for record in records)
    time_count = sum(record["time_count"] for record in records)
    same_day_count = sum(record["same_day_count"] for record in records)
    result = {}
    if clinical_count:
        result.update({
            "clinical_ce": (
                sum(record["clinical_ce_sum"] for record in records)
                / clinical_count
            ),
            "clinical_top1": (
                sum(record["clinical_top1"] for record in records)
                / clinical_count
            ),
            "clinical_top5": (
                sum(record["clinical_top5"] for record in records)
                / clinical_count
            ),
            "clinical_top10": (
                sum(record["clinical_top10"] for record in records)
                / clinical_count
            ),
        })
    if time_count:
        result.update({
            "time_mae_days": (
                sum(record["time_error_sum"] for record in records)
                / time_count
            ),
            "time_nll": (
                sum(record["time_nll_sum"] for record in records)
                / time_count
            ),
        })
    if same_day_count:
        result.update({
            "same_day_brier": (
                sum(record["same_day_brier_sum"] for record in records)
                / same_day_count
            ),
            "same_day_prevalence": (
                sum(record["same_day_positive"] for record in records)
                / same_day_count
            ),
        })
    return result


def summarize_patient_bootstrap(patient_acc, samples, seed):
    records = [
        dict(record)
        for record in patient_acc.values()
        if (
            record["clinical_count"]
            or record["time_count"]
            or record["same_day_count"]
        )
    ]
    if samples <= 0 or not records:
        return {
            "enabled": False,
            "patients": len(records),
            "samples": int(samples),
        }
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(samples):
        indices = rng.integers(0, len(records), size=len(records))
        metrics = summarize_patient_sample([records[index] for index in indices])
        for key, value in metrics.items():
            draws[key].append(value)
    intervals = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        intervals[key] = {
            "mean": float(arr.mean()),
            "ci95_lower": float(np.quantile(arr, 0.025)),
            "ci95_upper": float(np.quantile(arr, 0.975)),
        }
    return {
        "enabled": True,
        "patients": len(records),
        "samples": int(samples),
        "seed": int(seed),
        "metrics": intervals,
    }


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def binary_auprc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, labels.size + 1)
    return float(precision[sorted_labels == 1].sum() / positives)


def summarize_same_day(time_acc, n_bins=10):
    if not time_acc["same_day_labels"]:
        return {"targets": 0}
    labels = np.concatenate(time_acc["same_day_labels"]).astype(np.int64)
    probabilities = np.concatenate(time_acc["same_day_probabilities"]).astype(
        np.float64
    )
    bins = []
    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    prevalence = float(labels.mean())
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(labels[mask].mean())
            ece += count / labels.size * abs(mean_probability - observed_rate)
            reliability += count / labels.size * (
                mean_probability - observed_rate
            ) ** 2
            resolution += count / labels.size * (
                observed_rate - prevalence
            ) ** 2
        else:
            mean_probability = None
            observed_rate = None
        bins.append({
            "lower": lower,
            "upper": upper,
            "targets": count,
            "mean_probability": mean_probability,
            "observed_same_day_rate": observed_rate,
        })
    uncertainty = prevalence * (1.0 - prevalence)
    return {
        "targets": int(labels.size),
        "same_day_targets": int(labels.sum()),
        "same_day_prevalence": prevalence,
        "auroc": binary_auroc(labels, probabilities),
        "auprc": binary_auprc(labels, probabilities),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "brier_decomposition": {
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty),
        },
        "expected_calibration_error": float(ece),
        "calibration_bins": bins,
    }


def update_time_bucket(time_acc, name, actual, errors, nll):
    bucket = time_acc["horizon_buckets"][name]
    bucket["targets"] += int(actual.numel())
    bucket["actual"].append(actual.detach().cpu().numpy())
    bucket["errors"].append(errors.detach().cpu().numpy())
    bucket["nll"].append(nll.detach().cpu().numpy())


def summarize_time_buckets(time_acc):
    summary = {}
    for name, bucket in time_acc["horizon_buckets"].items():
        if not bucket["actual"]:
            summary[name] = {"targets": 0}
            continue
        actual = np.concatenate(bucket["actual"])
        errors = np.concatenate(bucket["errors"])
        nll = np.concatenate(bucket["nll"])
        summary[name] = {
            "targets": int(actual.size),
            "actual_median_days": float(np.median(actual)),
            "model_mae_days": float(errors.mean()),
            "model_median_absolute_error_days": float(np.median(errors)),
            "model_p95_absolute_error_days": float(np.quantile(errors, 0.95)),
            "model_nll": float(nll.mean()),
        }
    return summary


def evaluate_batch(
    model,
    tensors,
    windows,
    clinical_vocab_mask,
    train_token_counts,
    unigram_log_probs,
    unigram_top1_token,
    stats,
    time_acc,
    patient_acc,
    evaluate_time,
    autocast_context_factory,
):
    x, age, targets, target_age, token_types, target_types, repeated = tensors
    patient_ids, sex_tokens, patient_lengths, visit_density = collate_metadata(
        windows,
        targets.shape[1],
        targets.device,
    )
    with autocast_context_factory():
        logits, loss, _ = model(
            x,
            age,
            token_types,
            targets,
            target_age,
            target_token_type=target_types,
            validation_loss_mode=True,
            return_attention=False,
            compute_time_loss=evaluate_time,
        )
    flat_targets = targets.reshape(-1)
    flat_types = target_types.reshape(-1)
    objective_mask = build_target_mask(
        flat_targets,
        flat_types,
        list(model.config.ignore_tokens) + [1],
        model.config.ignore_types,
    ).reshape_as(targets)
    clinical_mask = clinical_target_mask(targets, target_types)

    update_unigram(
        stats,
        unigram_log_probs,
        unigram_top1_token,
        targets,
        clinical_mask,
    )
    update_accuracy(stats, logits, targets, objective_mask, "objective")
    update_accuracy(stats, logits, targets, clinical_mask, "clinical_full")

    clinical_logits = logits.masked_fill(
        ~clinical_vocab_mask.view(1, 1, -1),
        -torch.inf,
    )
    update_accuracy(
        stats,
        clinical_logits,
        targets,
        clinical_mask,
        "clinical_only",
    )
    update_patient_clinical(
        patient_acc,
        clinical_logits,
        targets,
        clinical_mask,
        patient_ids,
    )

    for token_type, name in CLINICAL_TYPES.items():
        type_mask = clinical_mask & (target_types == token_type)
        update_accuracy(
            stats,
            clinical_logits,
            targets,
            type_mask,
            f"type_{name}",
        )

    update_accuracy(
        stats,
        clinical_logits,
        targets,
        clinical_mask & ~repeated,
        "new_clinical",
    )
    update_accuracy(
        stats,
        clinical_logits,
        targets,
        clinical_mask & repeated,
        "repeated_clinical",
    )
    group_masks = {
        "frequency": frequency_bucket_masks(targets, train_token_counts),
        "age": range_masks(target_age, AGE_GROUPS),
        "sex": {
            f"sex_token_{int(token)}": sex_tokens == int(token)
            for token in sorted(
                int(token)
                for token in torch.unique(sex_tokens.detach().cpu())
                if int(token) >= 0
            )
        },
        "sequence_length": range_masks(patient_lengths, SEQUENCE_LENGTH_BUCKETS),
        "visit_density": range_masks(visit_density, VISIT_DENSITY_BUCKETS),
    }
    for group_name, masks in group_masks.items():
        update_group_accuracy(
            stats,
            clinical_logits,
            targets,
            clinical_mask,
            masks,
            f"stratified_{group_name}",
        )

    if evaluate_time:
        attention_mask = build_attention_mask(
            x,
            age,
            targets_age=target_age,
            mask_ties=model.config.mask_ties,
        )
        actual_dt = align_time_deltas(
            age,
            target_age,
            attention_mask,
            model.config.mask_ties,
        )
        # Use the rate the model actually computed (coupled or decoupled head),
        # exposed by forward(), instead of re-deriving it here.
        effective_log_rate = loss["effective_log_rate"].float()
        rate = torch.exp(effective_log_rate)
        # Point prediction for the error metrics is the exponential *median*
        # (ln2 / rate), the MAE-optimal estimator, so the comparison against the
        # median-gap baseline is fair. The mean (1/rate) is inflated by the long
        # tail. NLL below still uses the full rate.
        predicted_dt = torch.clamp(
            math.log(2.0) * torch.exp(-effective_log_rate), min=1.0
        )
        same_day_mask = target_age == age
        different_day_mask = clinical_mask & ~same_day_mask
        if loss["same_day_logits"] is not None:
            same_day_probabilities = torch.sigmoid(
                loss["same_day_logits"].float()
            )
            same_day_prob = same_day_probabilities[clinical_mask]
            same_day_labels = same_day_mask[clinical_mask]
            time_acc["same_day_probabilities"].append(
                same_day_prob.detach().cpu().numpy()
            )
            time_acc["same_day_labels"].append(
                same_day_labels.detach().cpu().numpy()
            )
            update_patient_same_day(
                patient_acc,
                patient_ids[clinical_mask],
                same_day_labels,
                same_day_prob,
            )
        actual = actual_dt[different_day_mask]
        if actual.numel():
            errors = torch.abs(predicted_dt[different_day_mask] - actual)
            # Per-target negative log-likelihood of the exponential waiting time:
            # -log(rate) + rate * dt. Used to compare against a constant-rate
            # baseline so "computable" is distinguished from "useful".
            nll = (
                -effective_log_rate[different_day_mask]
                + rate[different_day_mask] * actual
            )
            time_acc["errors"].append(errors.detach().cpu().numpy())
            time_acc["actual"].append(actual.detach().cpu().numpy())
            time_acc["nll"].append(nll.detach().cpu().numpy())
            update_patient_time(
                patient_acc,
                patient_ids[different_day_mask],
                errors,
                nll,
            )
            for name, lower, upper in TIME_HORIZON_BUCKETS:
                horizon_mask = actual >= lower
                if upper is not None:
                    horizon_mask &= actual <= upper
                if horizon_mask.any():
                    update_time_bucket(
                        time_acc,
                        name,
                        actual[horizon_mask],
                        errors[horizon_mask],
                        nll[horizon_mask],
                    )

    stats["batches"] += 1
    stats["objective_model_targets"] += int(loss["n_targets"])


def summarize_waiting_time(time_acc, baseline):
    """Model waiting-time quality against constant-rate and median baselines.

    The time head is only useful if it beats a global constant-rate exponential
    on likelihood (NLL) and a median-gap predictor on absolute error, so a
    finite MAE alone does not pass.
    """
    if not time_acc["actual"]:
        return {
            "targets": 0,
            "model_mae_days": None,
            "model_median_absolute_error_days": None,
            "model_p95_absolute_error_days": None,
            "model_nll": None,
            "constant_rate_baseline_nll": None,
            "median_baseline_mae_days": None,
            "nll_improvement_over_constant_rate": None,
            "mae_improvement_over_median": None,
            "baseline_source_split": None,
            "baseline_targets": 0,
            "baseline_mean_gap_days": None,
            "baseline_median_gap_days": None,
            "beats_baseline": None,
            "horizon_buckets": summarize_time_buckets(time_acc),
        }
    errors = np.concatenate(time_acc["errors"])
    actual = np.concatenate(time_acc["actual"])
    nll = np.concatenate(time_acc["nll"])
    model_nll = float(nll.mean())
    model_mae = float(errors.mean())
    if baseline is None:
        constant_rate_nll = None
        median_baseline_mae = None
    else:
        baseline_rate = 1.0 / max(baseline["mean_gap_days"], 1e-12)
        constant_rate_nll = float(
            (-math.log(baseline_rate) + baseline_rate * actual).mean()
        )
        median_baseline_mae = float(
            np.abs(actual - baseline["median_gap_days"]).mean()
        )
    nll_gain = (
        constant_rate_nll - model_nll if constant_rate_nll is not None else None
    )
    mae_gain = (
        median_baseline_mae - model_mae
        if median_baseline_mae is not None
        else None
    )
    return {
        "targets": int(actual.size),
        "model_mae_days": model_mae,
        "model_median_absolute_error_days": float(np.median(errors)),
        "model_p95_absolute_error_days": float(np.quantile(errors, 0.95)),
        "model_nll": model_nll,
        "constant_rate_baseline_nll": constant_rate_nll,
        "median_baseline_mae_days": median_baseline_mae,
        "nll_improvement_over_constant_rate": nll_gain,
        "mae_improvement_over_median": mae_gain,
        "baseline_source_split": baseline["source_split"] if baseline else None,
        "baseline_targets": baseline["targets"] if baseline else 0,
        "baseline_mean_gap_days": baseline["mean_gap_days"] if baseline else None,
        "baseline_median_gap_days": (
            baseline["median_gap_days"] if baseline else None
        ),
        "beats_baseline": bool(
            nll_gain is not None
            and mae_gain is not None
            and nll_gain > 0
            and mae_gain > 0
        ),
        "horizon_buckets": summarize_time_buckets(time_acc),
    }


def main():
    args = parse_args()
    model, checkpoint = load_model(args.ckpt, args.device)
    data, has_types = load_data(args.data_dir / f"{args.split}.bin")
    if not has_types:
        raise ValueError("SNUH checkpoint evaluation requires 4-column data")
    registry = load_registry(args.data_dir)
    output_mask = clinical_output_mask(
        registry,
        model.config.vocab_size,
        args.device,
    )
    if not output_mask.any():
        raise ValueError("No clinical output tokens were found in the registry")
    train_data, train_has_types = load_data(args.data_dir / "train.bin")
    if not train_has_types:
        raise ValueError("SNUH unigram baseline requires 4-column train data")
    unigram_log_probs, unigram_top1_token = build_clinical_unigram(
        train_data,
        output_mask,
    )
    train_token_counts = build_train_token_counts(
        train_data,
        model.config.vocab_size,
    )
    evaluate_time = (
        float(checkpoint.get("config", {}).get("loss_dt_weight", 1.0)) != 0
    )
    waiting_baseline = None
    if evaluate_time:
        print("Building train-only clinical waiting-time baselines...")
        waiting_baseline = build_train_waiting_baseline(
            train_data,
            model.config.block_size,
            args.selectors,
            args.batch_size,
            args.time_baseline_max_patients,
            model.config.mask_ties,
        )

    stats = defaultdict(float)
    patient_acc = defaultdict(
        lambda: {
            "clinical_ce_sum": 0.0,
            "clinical_count": 0,
            "clinical_top1": 0,
            "clinical_top5": 0,
            "clinical_top10": 0,
            "time_count": 0,
            "time_error_sum": 0.0,
            "time_nll_sum": 0.0,
            "same_day_count": 0,
            "same_day_positive": 0,
            "same_day_brier_sum": 0.0,
        }
    )
    time_acc = {
        "errors": [],
        "actual": [],
        "nll": [],
        "same_day_probabilities": [],
        "same_day_labels": [],
        "horizon_buckets": {
            name: {
                "targets": 0,
                "actual": [],
                "errors": [],
                "nll": [],
            }
            for name, _, _ in TIME_HORIZON_BUCKETS
        },
    }
    if args.device.startswith("cuda") and args.dtype != "float32":
        autocast_context_factory = lambda: torch.amp.autocast(
            "cuda",
            dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[args.dtype],
        )
    else:
        autocast_context_factory = nullcontext
    buffer = []
    with torch.no_grad():
        for window in iter_windows(
            data,
            model.config.block_size,
            args.selectors,
            args.max_patients,
        ):
            buffer.append(window)
            if len(buffer) < args.batch_size:
                continue
            evaluate_batch(
                model,
                collate(buffer, args.device),
                buffer,
                output_mask,
                train_token_counts,
                unigram_log_probs,
                unigram_top1_token,
                stats,
                time_acc,
                patient_acc,
                evaluate_time,
                autocast_context_factory,
            )
            buffer = []
        if buffer:
            evaluate_batch(
                model,
                collate(buffer, args.device),
                buffer,
                output_mask,
                train_token_counts,
                unigram_log_probs,
                unigram_top1_token,
                stats,
                time_acc,
                patient_acc,
                evaluate_time,
                autocast_context_factory,
            )

    waiting_time = summarize_waiting_time(time_acc, waiting_baseline)
    metrics = {
        "checkpoint": str(args.ckpt),
        "checkpoint_step": checkpoint.get("iter_num"),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "selectors": args.selectors,
        "max_patients": args.max_patients,
        "target_policy": {
            "ignore_tokens": [int(value) for value in model.config.ignore_tokens],
            "output_ignore_tokens": [
                int(value) for value in model.config.output_ignore_tokens
            ],
            "ignore_types": [int(value) for value in model.config.ignore_types],
        },
        "time_loss_enabled": evaluate_time,
        "two_stage_time_head": bool(model.config.two_stage_time_head),
        "objective": finalize_accuracy(stats, "objective"),
        "clinical_full_softmax": finalize_accuracy(stats, "clinical_full"),
        "clinical_only_softmax": finalize_accuracy(stats, "clinical_only"),
        "train_clinical_unigram": finalize_unigram(stats),
        "new_clinical": finalize_accuracy(stats, "new_clinical"),
        "repeated_clinical": finalize_accuracy(stats, "repeated_clinical"),
        "type_specific": {
            name: finalize_accuracy(stats, f"type_{name}")
            for name in CLINICAL_TYPES.values()
        },
        "stratified_clinical_only_softmax": {
            "token_frequency": finalize_group_accuracy(
                stats,
                "stratified_frequency",
                [name for name, _, _ in FREQUENCY_BUCKETS],
            ),
            "age_group": finalize_group_accuracy(
                stats,
                "stratified_age",
                [name for name, _, _ in AGE_GROUPS],
            ),
            "sex": finalize_group_accuracy(
                stats,
                "stratified_sex",
                observed_group_names(stats, "stratified_sex"),
            ),
            "visit_density": finalize_group_accuracy(
                stats,
                "stratified_visit_density",
                [name for name, _, _ in VISIT_DENSITY_BUCKETS],
            ),
            "sequence_length": finalize_group_accuracy(
                stats,
                "stratified_sequence_length",
                [name for name, _, _ in SEQUENCE_LENGTH_BUCKETS],
            ),
            "calendar_year": {
                "status": (
                    "not_available_from age-only binary events; provide an "
                    "event-date sidecar to enable calendar-year stratification"
                ),
            },
        },
        "clinical_waiting_time": waiting_time,
        "clinical_same_day": summarize_same_day(time_acc),
        "patient_level_bootstrap_ci": summarize_patient_bootstrap(
            patient_acc,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
        "evaluated_batches": int(stats["batches"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
