"""Measure the windows actually seen by SNUH smoke training."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import TokenType
from utils import get_batch, get_p2i, load_data


TYPE_NAMES = {int(token_type): token_type.name for token_type in TokenType}
BASELINE_IGNORED = {TokenType.PAD, TokenType.SEX, TokenType.NO_EVENT}
CONTEXT_IGNORED = BASELINE_IGNORED | {TokenType.LAB}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    args = parse_args()
    data, has_types = load_data(args.data_dir / f"{args.split}.bin")
    if not has_types:
        raise ValueError("Window audit requires FERMAT 4-column data")

    p2i = get_p2i(data)
    rng = torch.Generator().manual_seed(args.seed)
    type_counts = {name: 0 for name in TYPE_NAMES.values()}
    valid_input_counts = []
    lab_fractions = []
    baseline_targets = []
    context_targets = []
    same_day_transitions = []

    for _ in range(args.batches):
        patient_ix = torch.randint(
            len(p2i), (args.batch_size,), generator=rng
        )
        x, age, _, target_age, input_types, target_types = get_batch(
            patient_ix,
            data,
            p2i,
            select="random",
            padding="none",
            block_size=args.block_size,
            device="cpu",
            cut_batch=False,
            return_target_types=True,
        )

        for row in range(len(patient_ix)):
            input_row = input_types[row]
            target_row = target_types[row]
            valid_input = input_row != TokenType.PAD
            valid_target = target_row != TokenType.PAD

            valid_count = int(valid_input.sum())
            valid_input_counts.append(valid_count)
            lab_fractions.append(
                float((input_row[valid_input] == TokenType.LAB).float().mean())
                if valid_count else 0.0
            )

            for type_id, name in TYPE_NAMES.items():
                type_counts[name] += int((input_row[valid_input] == type_id).sum())

            baseline_mask = valid_target.clone()
            context_mask = valid_target.clone()
            for ignored_type in BASELINE_IGNORED:
                baseline_mask &= target_row != ignored_type
            for ignored_type in CONTEXT_IGNORED:
                context_mask &= target_row != ignored_type

            baseline_targets.append(int(baseline_mask.sum()))
            context_targets.append(int(context_mask.sum()))
            same_day_transitions.append(
                int(((target_age[row] == age[row]) & valid_target).sum())
            )

    total_inputs = sum(type_counts.values())
    result = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "patients": int(len(p2i)),
        "events": int(len(data)),
        "sampling": "patient-uniform, within-patient random window",
        "block_size": args.block_size,
        "sampled_windows": args.batch_size * args.batches,
        "input_type_fraction": {
            name: count / total_inputs if total_inputs else 0.0
            for name, count in type_counts.items()
        },
        "valid_input_tokens_per_window": summarize(valid_input_counts),
        "lab_fraction_per_window": summarize(lab_fractions),
        "baseline_effective_targets_per_window": summarize(baseline_targets),
        "lab_context_effective_targets_per_window": summarize(context_targets),
        "same_day_transitions_per_window": summarize(same_day_transitions),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
