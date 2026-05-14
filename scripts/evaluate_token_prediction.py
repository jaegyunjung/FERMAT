"""
Evaluate next-token prediction on the full Synthetic SNUH validation split.

Outputs:
  - logs/fermat_token_prediction_eval.md
  - logs/fermat_token_prediction_eval.csv

This script uses model.forward() logits directly and evaluates next-token
prediction on the 4-column FERMAT validation data without changing the
preprocessing pipeline.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Fermat, FermatConfig, TokenType
from utils import get_p2i, load_data


MASK_TIME = -10000.0
EVAL_TOKEN_TYPES = {
    int(TokenType.DX): "DX",
    int(TokenType.RX): "RX",
    int(TokenType.PX): "PX",
    int(TokenType.LAB): "LAB",
    int(TokenType.DTH): "DTH",
}
IGNORED_TARGET_TYPES = {
    int(TokenType.PAD),
    int(TokenType.SEX),
    int(TokenType.NO_EVENT),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="FERMAT-synthetic-snuh-token-prediction/ckpt.pt",
        help="checkpoint path",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        default="data/synthetic_snuh",
        help="directory containing train.bin/val.bin/vocab.csv",
    )
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--batch-size", type=int, default=64, help="number of chunks per eval batch")
    parser.add_argument(
        "--out-md",
        default="logs/fermat_token_prediction_eval.md",
        help="markdown report output path",
    )
    parser.add_argument(
        "--out-csv",
        default="logs/fermat_token_prediction_eval.csv",
        help="csv report output path",
    )
    return parser.parse_args()


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    model = Fermat(FermatConfig(**model_args))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_vocab_rows(vocab_path):
    with open(vocab_path, newline="") as f:
        return list(csv.DictReader(f))


def iter_patient_chunks(data, block_size):
    p2i = get_p2i(data)
    for row in p2i:
        start = int(row[0])
        length = int(row[1])
        if length < 2:
            continue
        patient = data[start:start + length]
        for chunk_start in range(0, length - 1, block_size):
            chunk_end = min(chunk_start + block_size + 1, length)
            if chunk_end - chunk_start < 2:
                continue
            chunk = patient[chunk_start:chunk_end]
            yield {
                "patient_id": int(patient[0, 0]),
                "token_ids": chunk[:, 2].astype("int64"),
                "ages": chunk[:, 1].astype("float32"),
                "token_types": chunk[:, 3].astype("int64"),
            }


def collate_chunks(chunks, device):
    max_len = max(len(chunk["token_ids"]) - 1 for chunk in chunks)
    batch_size = len(chunks)

    x = torch.zeros((batch_size, max_len), dtype=torch.long)
    y = torch.zeros((batch_size, max_len), dtype=torch.long)
    xt = torch.full((batch_size, max_len), int(TokenType.PAD), dtype=torch.long)
    yt = torch.full((batch_size, max_len), int(TokenType.PAD), dtype=torch.long)
    a = torch.full((batch_size, max_len), MASK_TIME, dtype=torch.float32)
    b = torch.full((batch_size, max_len), MASK_TIME, dtype=torch.float32)

    for i, chunk in enumerate(chunks):
        seq_len = len(chunk["token_ids"]) - 1
        raw_tokens = torch.tensor(chunk["token_ids"], dtype=torch.long)
        raw_ages = torch.tensor(chunk["ages"], dtype=torch.float32)
        raw_types = torch.tensor(chunk["token_types"], dtype=torch.long)

        x[i, :seq_len] = raw_tokens[:-1] + 1
        y[i, :seq_len] = raw_tokens[1:] + 1
        xt[i, :seq_len] = raw_types[:-1]
        yt[i, :seq_len] = raw_types[1:]
        a[i, :seq_len] = raw_ages[:-1]
        b[i, :seq_len] = raw_ages[1:]

    return (
        x.to(device),
        a.to(device),
        y.to(device),
        b.to(device),
        xt.to(device),
        yt.to(device),
    )


def metric_mask(y, yt):
    mask = y > 0
    for token_type in IGNORED_TARGET_TYPES:
        mask = mask & (yt != token_type)
    return mask


def update_type_stats(type_hits, top1_correct, masked_yt):
    for token_type, token_name in EVAL_TOKEN_TYPES.items():
        type_mask = masked_yt == token_type
        count = int(type_mask.sum().item())
        if count == 0:
            continue
        correct = int(top1_correct[type_mask].sum().item())
        stats = type_hits[token_name]
        stats["correct"] += correct
        stats["count"] += count


def write_markdown(out_path, metrics, checkpoint, ckpt_path, data_dir):
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# FERMAT token prediction evaluation\n\n")
        f.write(f"- Checkpoint: `{ckpt_path}`\n")
        f.write(f"- Data: `{data_dir}`\n")
        f.write(f"- Checkpoint step: {checkpoint.get('iter_num', 'unknown')}\n\n")
        f.write(
            "Note: training shifts raw `vocab.csv` token ids by `+1` before they are fed into the model. "
            "Metric decoding therefore maps model token index `k` back to raw token id `k-1`.\n\n"
        )

        f.write("## Overall Metrics\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        f.write(f"| validation_loss | {metrics['validation_loss']:.6f} |\n")
        f.write(f"| cross_entropy_loss | {metrics['cross_entropy_loss']:.6f} |\n")
        f.write(f"| perplexity | {metrics['perplexity']:.6f} |\n")
        f.write(f"| top1_accuracy | {metrics['top1_accuracy']:.6f} |\n")
        f.write(f"| top5_accuracy | {metrics['top5_accuracy']:.6f} |\n")
        f.write(f"| top10_accuracy | {metrics['top10_accuracy']:.6f} |\n")
        f.write(f"| evaluated_targets | {metrics['evaluated_targets']} |\n")
        f.write("\n")

        f.write("## Token-Type Specific Top-1 Accuracy\n\n")
        f.write("| token_type | top1_accuracy | count |\n|---|---:|---:|\n")
        for token_name in ["DX", "RX", "PX", "LAB", "DTH"]:
            type_metrics = metrics["type_specific"].get(token_name, {"accuracy": float("nan"), "count": 0})
            acc = type_metrics["accuracy"]
            if math.isnan(acc):
                acc_str = "NA"
            else:
                acc_str = f"{acc:.6f}"
            f.write(f"| {token_name} | {acc_str} | {type_metrics['count']} |\n")


def write_csv(out_path, metrics):
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    row = {
        "validation_loss": metrics["validation_loss"],
        "cross_entropy_loss": metrics["cross_entropy_loss"],
        "perplexity": metrics["perplexity"],
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "top10_accuracy": metrics["top10_accuracy"],
        "evaluated_targets": metrics["evaluated_targets"],
    }
    for token_name in ["DX", "RX", "PX", "LAB", "DTH"]:
        type_metrics = metrics["type_specific"].get(token_name, {"accuracy": float("nan"), "count": 0})
        row[f"{token_name.lower()}_top1_accuracy"] = type_metrics["accuracy"]
        row[f"{token_name.lower()}_count"] = type_metrics["count"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()

    model, checkpoint = load_model(args.ckpt, args.device)
    val_data, has_types = load_data(os.path.join(args.data_dir, "val.bin"))
    if not has_types:
        raise ValueError("evaluate_token_prediction.py requires 4-column FERMAT data")
    _ = load_vocab_rows(os.path.join(args.data_dir, "vocab.csv"))

    chunks_iter = iter_patient_chunks(val_data, model.config.block_size)

    total_tokens = 0
    ce_loss_sum = 0.0
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    type_hits = {name: {"correct": 0, "count": 0} for name in EVAL_TOKEN_TYPES.values()}

    buffer = []
    with torch.no_grad():
        for chunk in chunks_iter:
            buffer.append(chunk)
            if len(buffer) < args.batch_size:
                continue

            x, a, y, b, xt, yt = collate_chunks(buffer, args.device)
            logits, loss, _ = model(x, a, xt, y, b, validation_loss_mode=True)
            mask = metric_mask(y, yt)
            count = int(mask.sum().item())
            if count > 0:
                masked_logits = logits[mask]
                masked_targets = y[mask]
                masked_target_types = yt[mask]

                ce_loss_sum += F.cross_entropy(masked_logits, masked_targets, reduction="sum").item()

                topk = torch.topk(masked_logits, k=10, dim=-1).indices
                top1 = topk[:, :1]
                top5 = topk[:, :5]
                top10 = topk[:, :10]

                top1_correct = top1.squeeze(-1) == masked_targets
                top1_hits += int(top1_correct.sum().item())
                top5_hits += int((top5 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item())
                top10_hits += int((top10 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item())
                total_tokens += count

                update_type_stats(type_hits, top1_correct, masked_target_types)

            buffer = []

        if buffer:
            x, a, y, b, xt, yt = collate_chunks(buffer, args.device)
            logits, loss, _ = model(x, a, xt, y, b, validation_loss_mode=True)
            mask = metric_mask(y, yt)
            count = int(mask.sum().item())
            if count > 0:
                masked_logits = logits[mask]
                masked_targets = y[mask]
                masked_target_types = yt[mask]

                ce_loss_sum += F.cross_entropy(masked_logits, masked_targets, reduction="sum").item()

                topk = torch.topk(masked_logits, k=10, dim=-1).indices
                top1 = topk[:, :1]
                top5 = topk[:, :5]
                top10 = topk[:, :10]

                top1_correct = top1.squeeze(-1) == masked_targets
                top1_hits += int(top1_correct.sum().item())
                top5_hits += int((top5 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item())
                top10_hits += int((top10 == masked_targets.unsqueeze(-1)).any(dim=-1).sum().item())
                total_tokens += count

                update_type_stats(type_hits, top1_correct, masked_target_types)

    if total_tokens == 0:
        raise RuntimeError("No validation targets were available for evaluation")

    type_specific = {}
    for token_name, stats in type_hits.items():
        if stats["count"] == 0:
            type_specific[token_name] = {"accuracy": float("nan"), "count": 0}
        else:
            type_specific[token_name] = {
                "accuracy": stats["correct"] / stats["count"],
                "count": stats["count"],
            }

    metrics = {
        "validation_loss": ce_loss_sum / total_tokens,
        "cross_entropy_loss": ce_loss_sum / total_tokens,
        "perplexity": math.exp(ce_loss_sum / total_tokens),
        "top1_accuracy": top1_hits / total_tokens,
        "top5_accuracy": top5_hits / total_tokens,
        "top10_accuracy": top10_hits / total_tokens,
        "evaluated_targets": total_tokens,
        "type_specific": type_specific,
    }

    write_markdown(args.out_md, metrics, checkpoint, args.ckpt, args.data_dir)
    write_csv(args.out_csv, metrics)

    print(f"Evaluation written to {args.out_md} and {args.out_csv}")


if __name__ == "__main__":
    main()
