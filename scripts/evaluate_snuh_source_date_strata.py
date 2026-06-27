#!/usr/bin/env python3
"""Evaluate an SNUH checkpoint by source-date strata.

Inputs:
- a trained checkpoint;
- the 4-column FERMAT split shard;
- a clinical source-date audit parquet from build_snuh_event_date_sidecar.py.

The script re-runs deterministic evaluation windows and joins each clinical
target by its global row_index, so calendar-year and source clinical-gap
metrics are grounded in CDM event dates rather than inferred from age alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_snuh_checkpoint import (  # noqa: E402
    binary_auprc,
    binary_auroc,
    clinical_output_mask,
    clinical_target_mask,
    load_model,
    load_registry,
    window_start,
)
from model import (  # noqa: E402
    TokenType,
    align_time_deltas,
    build_attention_mask,
)
from utils import get_p2i, load_data  # noqa: E402


CLINICAL_TYPES = {
    int(TokenType.DX): "DX",
    int(TokenType.RX): "RX",
    int(TokenType.PX): "PX",
    int(TokenType.DTH): "DTH",
}
MASK_TIME = -10000.0
GAP_BUCKETS = [
    ("no_previous_clinical", None, None),
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
    parser.add_argument("--date-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["left", "middle", "right"],
        choices=["left", "middle", "right"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    return parser.parse_args()


def iter_windows_with_indices(data, block_size, selectors, max_patients=None):
    p2i = get_p2i(data)
    if max_patients is not None:
        p2i = p2i[:max_patients]
    for patient_start, patient_length in p2i:
        patient_start = int(patient_start)
        patient_length = int(patient_length)
        if patient_length < 2:
            continue
        patient = data[patient_start:patient_start + patient_length]
        for selector in selectors:
            relative_start = window_start(patient_length, block_size, selector)
            absolute_start = patient_start + relative_start
            rows = data[absolute_start:absolute_start + block_size + 1]
            row_indices = np.arange(
                absolute_start,
                absolute_start + len(rows),
                dtype=np.int64,
            )
            yield {
                "selector": selector,
                "patient_id": int(patient[0, 0]),
                "rows": rows,
                "row_indices": row_indices,
            }


def collate_with_row_indices(windows, device):
    max_targets = max(len(window["rows"]) - 1 for window in windows)
    batch_size = len(windows)
    x = torch.zeros((batch_size, max_targets), dtype=torch.long)
    y = torch.zeros_like(x)
    a = torch.full((batch_size, max_targets), MASK_TIME)
    b = torch.full_like(a, MASK_TIME)
    xt = torch.full_like(x, int(TokenType.PAD))
    yt = torch.full_like(x, int(TokenType.PAD))
    row_indices = torch.full((batch_size, max_targets), -1, dtype=torch.long)

    for index, window in enumerate(windows):
        rows = window["rows"]
        length = len(rows) - 1
        raw_tokens = torch.from_numpy(rows[:, 2].astype(np.int64))
        raw_ages = torch.from_numpy(rows[:, 1].astype(np.float32))
        raw_types = torch.from_numpy(rows[:, 3].astype(np.int64))
        raw_indices = torch.from_numpy(window["row_indices"].astype(np.int64))
        x[index, :length] = raw_tokens[:-1] + 1
        y[index, :length] = raw_tokens[1:] + 1
        a[index, :length] = raw_ages[:-1]
        b[index, :length] = raw_ages[1:]
        xt[index, :length] = raw_types[:-1]
        yt[index, :length] = raw_types[1:]
        row_indices[index, :length] = raw_indices[1:]

    return tuple(
        tensor.to(device)
        for tensor in (x, a, y, b, xt, yt, row_indices)
    )


def load_sidecar(path: Path):
    columns = [
        "row_index",
        "token_type_id",
        "calendar_year",
        "days_since_previous_clinical_event",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    return {
        "row_index": frame["row_index"].to_numpy(dtype=np.int64),
        "token_type_id": frame["token_type_id"].to_numpy(dtype=np.int16),
        "calendar_year": frame["calendar_year"].to_numpy(dtype=np.int16),
        "gap": frame["days_since_previous_clinical_event"].to_numpy(dtype=np.float64),
    }


def lookup_sidecar(sidecar, row_indices):
    positions = np.searchsorted(sidecar["row_index"], row_indices)
    valid = (positions >= 0) & (positions < len(sidecar["row_index"]))
    matched = np.zeros_like(valid, dtype=bool)
    matched[valid] = sidecar["row_index"][positions[valid]] == row_indices[valid]
    valid &= matched
    if not np.all(valid):
        missing = row_indices[~valid][:10].tolist()
        raise RuntimeError(f"Missing sidecar row_index values: {missing}")
    return {
        "calendar_year": sidecar["calendar_year"][positions],
        "token_type_id": sidecar["token_type_id"][positions],
        "gap": sidecar["gap"][positions],
    }


def empty_stats():
    return {
        "targets": 0,
        "ce_sum": 0.0,
        "top1": 0,
        "top5": 0,
        "top10": 0,
        "same_day_labels": [],
        "same_day_probs": [],
        "time_errors": [],
        "time_nll": [],
    }


def update_stats(stats, logits, targets, labels, probs, errors, nll):
    count = int(targets.numel())
    if count == 0:
        return
    stats["targets"] += count
    stats["ce_sum"] += float(F.cross_entropy(logits, targets, reduction="sum"))
    max_k = min(10, logits.shape[-1])
    topk = torch.topk(logits, k=max_k, dim=-1).indices
    stats["top1"] += int((topk[:, :1] == targets[:, None]).any(dim=-1).sum())
    stats["top5"] += int(
        (topk[:, :min(5, max_k)] == targets[:, None]).any(dim=-1).sum()
    )
    stats["top10"] += int(
        (topk[:, :min(10, max_k)] == targets[:, None]).any(dim=-1).sum()
    )
    if labels is not None and probs is not None:
        stats["same_day_labels"].append(labels.astype(np.int8))
        stats["same_day_probs"].append(probs.astype(np.float32))
    if errors is not None and nll is not None and errors.size:
        stats["time_errors"].append(errors.astype(np.float32))
        stats["time_nll"].append(nll.astype(np.float32))


def finalize_stats(name, stats):
    count = int(stats["targets"])
    row = {"group": name, "targets": count}
    if not count:
        return row
    row.update(
        {
            "cross_entropy": stats["ce_sum"] / count,
            "top1_accuracy": stats["top1"] / count,
            "top5_accuracy": stats["top5"] / count,
            "top10_accuracy": stats["top10"] / count,
        }
    )
    if stats["same_day_labels"]:
        labels = np.concatenate(stats["same_day_labels"])
        probs = np.concatenate(stats["same_day_probs"])
        row.update(
            {
                "same_day_targets": int(labels.size),
                "same_day_prevalence": float(labels.mean()),
                "same_day_brier": float(np.mean((probs - labels) ** 2)),
                "same_day_auroc": binary_auroc(labels, probs),
                "same_day_auprc": binary_auprc(labels, probs),
            }
        )
    if stats["time_errors"]:
        errors = np.concatenate(stats["time_errors"])
        nll = np.concatenate(stats["time_nll"])
        row.update(
            {
                "different_day_targets": int(errors.size),
                "time_mae_days": float(errors.mean()),
                "time_median_absolute_error_days": float(np.median(errors)),
                "time_p95_absolute_error_days": float(np.quantile(errors, 0.95)),
                "time_nll": float(nll.mean()),
            }
        )
    return row


def gap_bucket_name(value):
    if np.isnan(value):
        return "no_previous_clinical"
    for name, lower, upper in GAP_BUCKETS[1:]:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "unbucketed"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "group",
        "targets",
        "cross_entropy",
        "top1_accuracy",
        "top5_accuracy",
        "top10_accuracy",
        "same_day_targets",
        "same_day_prevalence",
        "same_day_brier",
        "same_day_auroc",
        "same_day_auprc",
        "different_day_targets",
        "time_nll",
        "time_mae_days",
        "time_median_absolute_error_days",
        "time_p95_absolute_error_days",
    ]
    fieldnames = [name for name in preferred if name in fieldnames] + [
        name for name in fieldnames if name not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_grouped(
    grouped,
    group_name,
    logits,
    targets,
    labels,
    probs,
    errors,
    nll,
):
    update_stats(
        grouped[group_name],
        logits,
        targets,
        labels,
        probs,
        errors,
        nll,
    )


def update_group_from_numpy_mask(
    grouped,
    group_name,
    mask,
    selected_logits,
    selected_targets,
    labels,
    probs_np,
    all_errors,
    all_nll,
):
    torch_mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=selected_logits.device,
    )
    time_mask = mask & ~np.isnan(all_errors)
    update_grouped(
        grouped,
        group_name,
        selected_logits[torch_mask],
        selected_targets[torch_mask],
        labels[mask],
        None if probs_np is None else probs_np[mask],
        all_errors[time_mask],
        all_nll[time_mask],
    )


def evaluate_batch(
    model,
    tensors,
    sidecar,
    clinical_vocab_mask,
    grouped,
    autocast_context_factory,
):
    x, age, targets, target_age, token_types, target_types, row_indices = tensors
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
            compute_time_loss=True,
        )
    clinical_mask = clinical_target_mask(targets, target_types)
    if not clinical_mask.any():
        return 0

    clinical_logits = logits.masked_fill(
        ~clinical_vocab_mask.view(1, 1, -1),
        -torch.inf,
    )
    flat_row_indices = row_indices[clinical_mask].detach().cpu().numpy()
    meta = lookup_sidecar(sidecar, flat_row_indices)

    selected_logits = clinical_logits[clinical_mask].float()
    selected_targets = targets[clinical_mask]
    same_day_mask = target_age == age
    labels = same_day_mask[clinical_mask].detach().cpu().numpy().astype(np.int8)

    if loss["same_day_logits"] is not None:
        probs = torch.sigmoid(loss["same_day_logits"].float())[clinical_mask]
        probs_np = probs.detach().cpu().numpy().astype(np.float32)
    else:
        probs_np = None

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
    effective_log_rate = loss["effective_log_rate"].float()
    rate = torch.exp(effective_log_rate)
    predicted_dt = torch.clamp(
        math.log(2.0) * torch.exp(-effective_log_rate),
        min=1.0,
    )
    different_day = clinical_mask & ~same_day_mask
    all_errors = np.full(int(clinical_mask.sum()), np.nan, dtype=np.float32)
    all_nll = np.full_like(all_errors, np.nan)
    if different_day.any():
        errors = torch.abs(predicted_dt[different_day] - actual_dt[different_day])
        nll = (
            -effective_log_rate[different_day]
            + rate[different_day] * actual_dt[different_day]
        )
        different_positions = different_day[clinical_mask].detach().cpu().numpy()
        all_errors[different_positions] = errors.detach().cpu().numpy()
        all_nll[different_positions] = nll.detach().cpu().numpy()

    selected_count = int(selected_targets.numel())
    for group_kind, values in [
        ("calendar_year", meta["calendar_year"]),
        ("token_type", np.array([CLINICAL_TYPES[int(v)] for v in meta["token_type_id"]])),
        ("clinical_gap_bucket", np.array([gap_bucket_name(v) for v in meta["gap"]])),
    ]:
        for value in sorted(set(values.tolist())):
            mask = values == value
            update_group_from_numpy_mask(
                grouped[group_kind],
                str(value),
                mask,
                selected_logits,
                selected_targets,
                labels,
                probs_np,
                all_errors,
                all_nll,
            )

    time_mask = ~np.isnan(all_errors)
    update_grouped(
        grouped["source_date_same_day"],
        "overall",
        selected_logits,
        selected_targets,
        labels,
        probs_np,
        all_errors[time_mask],
        all_nll[time_mask],
    )
    for year in sorted(set(meta["calendar_year"].tolist())):
        mask = meta["calendar_year"] == year
        update_group_from_numpy_mask(
            grouped["source_date_same_day"],
            f"year_{int(year)}",
            mask,
            selected_logits,
            selected_targets,
            labels,
            probs_np,
            all_errors,
            all_nll,
        )

    return selected_count


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(args.ckpt, args.device)
    data, has_types = load_data(args.data_dir / f"{args.split}.bin")
    if not has_types:
        raise ValueError("Source-date strata evaluation requires 4-column data")
    registry = load_registry(args.data_dir)
    output_mask = clinical_output_mask(
        registry,
        model.config.vocab_size,
        args.device,
    )
    sidecar = load_sidecar(args.date_sidecar)

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

    grouped = {
        "calendar_year": defaultdict(empty_stats),
        "clinical_gap_bucket": defaultdict(empty_stats),
        "source_date_same_day": defaultdict(empty_stats),
        "token_type": defaultdict(empty_stats),
    }
    processed_targets = 0
    buffer = []
    with torch.no_grad():
        for window in iter_windows_with_indices(
            data,
            model.config.block_size,
            args.selectors,
            args.max_patients,
        ):
            buffer.append(window)
            if len(buffer) < args.batch_size:
                continue
            processed_targets += evaluate_batch(
                model,
                collate_with_row_indices(buffer, args.device),
                sidecar,
                output_mask,
                grouped,
                autocast_context_factory,
            )
            print(f"clinical_targets_processed={processed_targets:,}", flush=True)
            buffer = []
        if buffer:
            processed_targets += evaluate_batch(
                model,
                collate_with_row_indices(buffer, args.device),
                sidecar,
                output_mask,
                grouped,
                autocast_context_factory,
            )
            print(f"clinical_targets_processed={processed_targets:,}", flush=True)

    write_rows(
        args.output_dir / "calendar_year_clinical_metrics.csv",
        [
            finalize_stats(group, stats)
            for group, stats in sorted(grouped["calendar_year"].items())
        ],
    )
    write_rows(
        args.output_dir / "clinical_gap_bucket_metrics.csv",
        [
            finalize_stats(group, grouped["clinical_gap_bucket"][group])
            for group, _, _ in GAP_BUCKETS
            if group in grouped["clinical_gap_bucket"]
        ],
    )
    write_rows(
        args.output_dir / "source_date_same_day_metrics.csv",
        [
            finalize_stats(group, stats)
            for group, stats in sorted(grouped["source_date_same_day"].items())
        ],
    )
    write_rows(
        args.output_dir / "token_type_source_date_metrics.csv",
        [
            finalize_stats(group, stats)
            for group, stats in sorted(grouped["token_type"].items())
        ],
    )
    manifest = {
        "checkpoint": str(args.ckpt),
        "checkpoint_step": checkpoint.get("iter_num"),
        "data_dir": str(args.data_dir),
        "date_sidecar": str(args.date_sidecar),
        "split": args.split,
        "selectors": args.selectors,
        "max_patients": args.max_patients,
        "clinical_targets_processed": processed_targets,
        "outputs": {
            "calendar_year": str(args.output_dir / "calendar_year_clinical_metrics.csv"),
            "clinical_gap_bucket": str(args.output_dir / "clinical_gap_bucket_metrics.csv"),
            "same_day": str(args.output_dir / "source_date_same_day_metrics.csv"),
            "token_type": str(args.output_dir / "token_type_source_date_metrics.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
