#!/usr/bin/env python3
"""Fit a monthly death-hazard head on observed FERMAT embeddings.

All death months are retained.  Non-event person-months are sampled at a fixed
probability and receive inverse-sampling weights, so the intercept and absolute
monthly risk remain estimable.  Train fits the model; val controls early
stopping.  Test is not read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SELF_TEST_ONLY = "--self-test" in sys.argv
if not SELF_TEST_ONLY:
    import torch


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_INPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_monthly_embeddings"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_embedding_hazard"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-sample-rate", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--random-seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def deterministic_keep(patient_key, month, rate, seed):
    value = (
        np.asarray(patient_key, dtype=np.uint64) * np.uint64(11400714819323198485)
        + np.asarray(month, dtype=np.uint64) * np.uint64(14029467366897019727)
        + np.uint64(seed)
    )
    threshold = np.uint64(float(rate) * np.iinfo(np.uint64).max)
    return value <= threshold


def load_sampled_parts(input_dir, split, negative_rate, seed):
    parts = []
    embedding_columns = None
    for path in sorted((Path(input_dir) / "parts").glob("monthly_embeddings_part_*.parquet")):
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["split"].eq(split)].copy()
        if frame.empty:
            continue
        current = sorted(column for column in frame.columns if column.startswith("emb_"))
        if embedding_columns is None:
            embedding_columns = current
        elif current != embedding_columns:
            raise ValueError(f"embedding columns differ in {path}")
        event = frame["death_event"].to_numpy(dtype=np.int8) == 1
        keep_negative = deterministic_keep(
            frame["patient_key"].to_numpy(), frame["month"].to_numpy(), negative_rate, seed
        )
        frame = frame.loc[event | keep_negative].copy()
        frame["sampling_weight"] = np.where(
            frame["death_event"].eq(1), 1.0, 1.0 / float(negative_rate)
        )
        parts.append(frame)
    if not parts or not embedding_columns:
        raise ValueError(f"no sampled {split} embeddings found")
    return pd.concat(parts, ignore_index=True), embedding_columns


def make_features(frame, embedding_columns, mean, scale):
    embedding = frame[embedding_columns].to_numpy(dtype=np.float32)
    embedding = (embedding - mean) / scale
    month = frame["month"].to_numpy(dtype=np.float32) / 60.0
    return np.column_stack([embedding, month, month * month]).astype(np.float32)


def weighted_loss(model, x, y, weight, ridge):
    logits = model(x).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, weight=weight, reduction="sum"
    ) / weight.sum().clamp_min(1.0)
    penalty = float(ridge) * torch.square(model.weight).sum()
    return loss + penalty


def evaluate(model, x, y, weight, batch_size):
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            stop = start + int(batch_size)
            logits = model(x[start:stop]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[start:stop], weight=weight[start:stop], reduction="sum"
            )
            total_loss += float(loss.detach().cpu())
            total_weight += float(weight[start:stop].sum().detach().cpu())
    return total_loss / max(total_weight, 1.0)


def self_test():
    key = np.arange(100_000, dtype=np.uint64)
    month = np.zeros(len(key), dtype=np.uint64)
    rate = deterministic_keep(key, month, 0.1, 42).mean()
    assert 0.095 < rate < 0.105
    print(f"SELF_TEST_OK deterministic_negative_rate={rate:.4f}")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 0 < args.negative_sample_rate <= 1:
        raise ValueError("negative-sample-rate must be in (0,1]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, embedding_columns = load_sampled_parts(
        args.input_dir, "train", args.negative_sample_rate, args.random_seed
    )
    val, val_columns = load_sampled_parts(
        args.input_dir, "val", args.negative_sample_rate, args.random_seed
    )
    if val_columns != embedding_columns:
        raise ValueError("train/val embedding columns differ")
    train_embedding = train[embedding_columns].to_numpy(dtype=np.float32)
    mean = train_embedding.mean(axis=0)
    scale = train_embedding.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    train_x = make_features(train, embedding_columns, mean, scale)
    val_x = make_features(val, embedding_columns, mean, scale)

    device = args.device
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train["death_event"].to_numpy(np.float32)).to(device)
    train_w = torch.from_numpy(train["sampling_weight"].to_numpy(np.float32)).to(device)
    val_x = torch.from_numpy(val_x).to(device)
    val_y = torch.from_numpy(val["death_event"].to_numpy(np.float32)).to(device)
    val_w = torch.from_numpy(val["sampling_weight"].to_numpy(np.float32)).to(device)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    torch.manual_seed(args.random_seed)
    torch.nn.init.zeros_(model.weight)
    prevalence = float(
        np.average(train["death_event"].to_numpy(float), weights=train["sampling_weight"])
    )
    prevalence = np.clip(prevalence, 1e-7, 1 - 1e-7)
    model.bias.data.fill_(float(np.log(prevalence / (1 - prevalence))))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    generator = torch.Generator(device="cpu").manual_seed(args.random_seed)
    best = None
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        permutation = torch.randperm(len(train_x), generator=generator)
        model.train()
        for start in range(0, len(permutation), int(args.batch_size)):
            index = permutation[start : start + int(args.batch_size)].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(
                model, train_x[index], train_y[index], train_w[index], args.ridge
            )
            loss.backward()
            optimizer.step()
        model.eval()
        train_nll = evaluate(model, train_x, train_y, train_w, args.batch_size)
        val_nll = evaluate(model, val_x, val_y, val_w, args.batch_size)
        history.append({"epoch": epoch, "train_weighted_nll": train_nll, "val_weighted_nll": val_nll})
        log(f"epoch={epoch} train_nll={train_nll:.8f} val_nll={val_nll:.8f}")
        if best is None or val_nll < best[0] - 1e-7:
            best = (
                val_nll,
                model.weight.detach().cpu().numpy().copy(),
                model.bias.detach().cpu().numpy().copy(),
                epoch,
            )
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best is None:
        raise RuntimeError("no hazard model was fit")
    np.savez_compressed(
        args.output_dir / "embedding_monthly_hazard_model.npz",
        weight=best[1],
        bias=best[2],
        embedding_mean=mean,
        embedding_scale=scale,
        embedding_columns=np.asarray(embedding_columns),
        best_epoch=np.asarray(best[3]),
        val_weighted_nll=np.asarray(best[0]),
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    manifest = {
        "status": "EMBEDDING_MONTHLY_HAZARD_COMPLETE",
        "train_sampled_rows": int(len(train)),
        "train_events": int(train["death_event"].sum()),
        "val_sampled_rows": int(len(val)),
        "val_events": int(val["death_event"].sum()),
        "negative_sample_rate": args.negative_sample_rate,
        "negative_inverse_sampling_weight": 1.0 / args.negative_sample_rate,
        "best_epoch": best[3],
        "best_val_weighted_nll": best[0],
        "test_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
