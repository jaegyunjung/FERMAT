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
}


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


def evaluate_batch(
    model,
    tensors,
    clinical_vocab_mask,
    stats,
    time_errors,
    autocast_context_factory,
):
    x, age, targets, target_age, token_types, target_types, repeated = tensors
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
        )
    flat_targets = targets.reshape(-1)
    flat_types = target_types.reshape(-1)
    objective_mask = build_target_mask(
        flat_targets,
        flat_types,
        list(model.config.ignore_tokens) + [1],
        model.config.ignore_types,
    ).reshape_as(targets)
    clinical_mask = targets > 0
    for token_type in NON_CLINICAL_TYPES:
        clinical_mask &= target_types != token_type

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
    raw_log_rate = torch.logsumexp(logits.float(), dim=-1)
    predicted_dt = torch.clamp(torch.exp(-raw_log_rate), min=1.0)
    errors = torch.abs(predicted_dt - actual_dt)[clinical_mask]
    if errors.numel():
        time_errors.append(errors.detach().cpu().numpy())

    stats["batches"] += 1
    stats["objective_model_targets"] += int(loss["n_targets"])


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

    stats = defaultdict(float)
    time_errors = []
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
                output_mask,
                stats,
                time_errors,
                autocast_context_factory,
            )
            buffer = []
        if buffer:
            evaluate_batch(
                model,
                collate(buffer, args.device),
                output_mask,
                stats,
                time_errors,
                autocast_context_factory,
            )

    errors = (
        np.concatenate(time_errors)
        if time_errors
        else np.array([], dtype=np.float32)
    )
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
        "objective": finalize_accuracy(stats, "objective"),
        "clinical_full_softmax": finalize_accuracy(stats, "clinical_full"),
        "clinical_only_softmax": finalize_accuracy(stats, "clinical_only"),
        "new_clinical": finalize_accuracy(stats, "new_clinical"),
        "repeated_clinical": finalize_accuracy(stats, "repeated_clinical"),
        "type_specific": {
            name: finalize_accuracy(stats, f"type_{name}")
            for name in CLINICAL_TYPES.values()
        },
        "clinical_waiting_time": {
            "targets": int(errors.size),
            "mae_days": float(errors.mean()) if errors.size else None,
            "median_absolute_error_days": (
                float(np.median(errors)) if errors.size else None
            ),
            "p95_absolute_error_days": (
                float(np.quantile(errors, 0.95)) if errors.size else None
            ),
        },
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
