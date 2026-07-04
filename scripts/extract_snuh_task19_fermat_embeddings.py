#!/usr/bin/env python3
"""Extract FERMAT patient embeddings for Task 19 disease-risk benchmarks.

For a fixed index date and horizon, this script finds patients who are eligible
for at least one phenotype in the Task 19 label table. It then feeds each
patient's token history strictly before the index age through the trained FERMAT
checkpoint and writes the last valid hidden state as the patient embedding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local help can run without torch
    torch = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings"
MASK_TIME = -10000.0
PAD_TOKEN_TYPE = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--horizon", default="5y")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--pooling",
        choices=["last", "mean"],
        default="last",
        help="Embedding pooling over valid pre-index tokens.",
    )
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


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is required for FERMAT embedding extraction. "
            "Run this script in the Pod environment used for FERMAT evaluation."
        )


def load_model(path: Path, device: str):
    from model import Fermat, FermatConfig

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Fermat(FermatConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model_state = model.state_dict()
    state_dict = {key: value for key, value in state_dict.items() if key in model_state}
    for missing in model_state.keys() - state_dict.keys():
        state_dict[missing] = model_state[missing]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def load_patient_map(data_dir: Path):
    path = data_dir / "patient_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    columns = ["patient_id_dense", "person_id", "split"]
    frame = pd.read_parquet(path, columns=columns)
    frame["patient_id_dense"] = frame["patient_id_dense"].astype(np.int64)
    frame["person_id"] = frame["person_id"].astype(np.int64)
    frame["split"] = frame["split"].astype(str)
    return frame


def label_columns(path: Path):
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except ModuleNotFoundError:
        return pd.read_parquet(path).columns.tolist()


def load_embedding_targets(args):
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{safe_date(args.index_date)}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    columns = label_columns(label_path)
    eligible_cols = [
        column for column in columns
        if column.startswith(f"eligible_{args.horizon}__")
    ]
    if not eligible_cols:
        raise ValueError(f"No eligible columns found for horizon={args.horizon}")
    labels = pd.read_parquet(
        label_path,
        columns=["person_id", "split", "age_at_index"] + eligible_cols,
    )
    eligible_any = labels[eligible_cols].astype(bool).any(axis=1)
    labels = labels.loc[eligible_any, ["person_id", "split", "age_at_index"]].copy()
    labels["index_age_days"] = np.floor(labels["age_at_index"] * 365.25).astype(np.int64)
    if args.max_patients is not None:
        labels = labels.head(args.max_patients).copy()
    return labels, str(label_path), eligible_cols


def build_patient_index(data):
    from utils import get_p2i

    p2i = get_p2i(data)
    patient_ids = data[p2i[:, 0].astype(np.int64), 0].astype(np.int64)
    return {
        int(patient_id): (int(start), int(length))
        for patient_id, (start, length) in zip(patient_ids, p2i)
    }


def load_split_indices(data_dir: Path):
    from utils import load_data

    result = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(path)
        data, has_types = load_data(path)
        if not has_types:
            raise ValueError(f"{path} is not a 4-column typed FERMAT bin")
        result[split] = {
            "path": str(path),
            "data": data,
            "index": build_patient_index(data),
        }
        log(f"loaded {split}: rows={len(data):,} patients={len(result[split]['index']):,}")
    return result


def patient_rows_before_index(split_data, dense_id: int, index_age_days: int, block_size: int):
    location = split_data["index"].get(int(dense_id))
    if location is None:
        return None
    start, length = location
    rows = split_data["data"][start:start + length]
    rows = rows[rows[:, 1].astype(np.int64) < int(index_age_days)]
    if len(rows) == 0:
        return rows
    if len(rows) > block_size:
        rows = rows[-block_size:]
    return rows


def collate_rows(row_items, device):
    batch_size = len(row_items)
    lengths = torch.tensor([len(item["rows"]) for item in row_items], dtype=torch.long)
    max_len = int(lengths.max().item()) if batch_size else 0
    x = torch.zeros((batch_size, max_len), dtype=torch.long)
    age = torch.full((batch_size, max_len), MASK_TIME, dtype=torch.float32)
    token_type = torch.full((batch_size, max_len), PAD_TOKEN_TYPE, dtype=torch.long)

    for index, item in enumerate(row_items):
        rows = item["rows"]
        if len(rows) == 0:
            continue
        length = len(rows)
        x[index, :length] = torch.from_numpy(rows[:, 2].astype(np.int64)) + 1
        age[index, :length] = torch.from_numpy(rows[:, 1].astype(np.float32))
        token_type[index, :length] = torch.from_numpy(rows[:, 3].astype(np.int64))
    return x.to(device), age.to(device), token_type.to(device), lengths.to(device)


def extract_hidden(model, idx, age, token_type, lengths, pooling):
    from model import build_attention_mask

    tok_emb = model.transformer.wte(idx)
    age_emb = model.transformer.wae(age.unsqueeze(-1))
    type_emb = model.transformer.wtype(token_type)
    x = model.transformer.token_drop(tok_emb) * (1 - model.config.token_dropout)
    x = x + age_emb + type_emb
    x = model.transformer.drop(x)
    attention_mask = build_attention_mask(idx, age, mask_ties=False)
    for block in model.transformer.h:
        x, _ = block(x, attention_mask)
    x = model.transformer.ln_f(x)

    valid = idx > 0
    if pooling == "last":
        gather_index = torch.clamp(lengths - 1, min=0)
        pooled = x[torch.arange(x.size(0), device=x.device), gather_index]
    elif pooling == "mean":
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        pooled = (x * valid.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
    else:
        raise ValueError(pooling)
    pooled = pooled.masked_fill((lengths == 0).view(-1, 1), 0)
    return pooled


def autocast_context(device, dtype):
    if device == "cuda" and dtype in {"bfloat16", "float16"}:
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        )
    return torch.amp.autocast(device_type="cpu", enabled=False)


def main():
    args = parse_args()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    require_torch()

    started = time.time()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    labels, label_path, eligible_cols = load_embedding_targets(args)
    patient_map = load_patient_map(args.data_dir)
    targets = labels.merge(patient_map, on=["person_id", "split"], how="left")
    if targets["patient_id_dense"].isna().any():
        missing = int(targets["patient_id_dense"].isna().sum())
        raise RuntimeError(f"{missing:,} target patients were missing patient_id_dense")
    targets["patient_id_dense"] = targets["patient_id_dense"].astype(np.int64)
    log(f"embedding target patients={len(targets):,}")

    model, checkpoint = load_model(args.ckpt, device)
    split_indices = load_split_indices(args.data_dir)
    block_size = int(model.config.block_size)
    n_embd = int(model.config.n_embd)
    log(f"checkpoint_step={checkpoint.get('iter')} block_size={block_size} n_embd={n_embd}")

    embeddings = np.zeros((len(targets), n_embd), dtype=np.float32)
    lengths = np.zeros(len(targets), dtype=np.int32)
    missing_sequence = np.zeros(len(targets), dtype=bool)

    buffer = []
    row_indices = []
    processed = 0

    def flush():
        nonlocal buffer, row_indices, processed
        if not buffer:
            return
        idx, age, token_type, batch_lengths = collate_rows(buffer, device)
        with torch.no_grad(), autocast_context(device, args.dtype):
            pooled = extract_hidden(model, idx, age, token_type, batch_lengths, args.pooling)
        embeddings[np.asarray(row_indices)] = pooled.detach().float().cpu().numpy()
        lengths[np.asarray(row_indices)] = batch_lengths.detach().cpu().numpy().astype(np.int32)
        processed += len(buffer)
        if processed % max(args.batch_size * 20, 1) == 0:
            log(f"processed={processed:,}/{len(targets):,}")
        buffer = []
        row_indices = []

    for output_index, row in enumerate(targets.itertuples(index=False)):
        split_data = split_indices[str(row.split)]
        rows = patient_rows_before_index(
            split_data,
            int(row.patient_id_dense),
            int(row.index_age_days),
            block_size,
        )
        if rows is None:
            missing_sequence[output_index] = True
            rows = np.zeros((0, 4), dtype=np.uint32)
        elif len(rows) == 0:
            missing_sequence[output_index] = True
        buffer.append({"rows": rows})
        row_indices.append(output_index)
        if len(buffer) >= args.batch_size:
            flush()
    flush()

    embed_cols = [f"emb_{i:04d}" for i in range(n_embd)]
    output = pd.DataFrame(embeddings, columns=embed_cols)
    output.insert(0, "sequence_length_pre_index", lengths)
    output.insert(0, "has_embedding_sequence", ~missing_sequence)
    output.insert(0, "index_date", args.index_date)
    output.insert(0, "horizon", args.horizon)
    output.insert(0, "split", targets["split"].to_numpy())
    output.insert(0, "patient_id_dense", targets["patient_id_dense"].to_numpy())
    output.insert(0, "person_id", targets["person_id"].to_numpy())
    out_path = args.output_dir / (
        f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizon}_{args.pooling}.parquet"
    )
    output.to_parquet(out_path, index=False)

    norms = np.linalg.norm(embeddings, axis=1)
    manifest = {
        "ckpt": str(args.ckpt),
        "checkpoint_step": int(checkpoint.get("iter", -1)),
        "data_dir": str(args.data_dir),
        "label_path": label_path,
        "output_path": str(out_path),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "pooling": args.pooling,
        "target_patients": int(len(targets)),
        "missing_sequence_patients": int(missing_sequence.sum()),
        "block_size": block_size,
        "n_embd": n_embd,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "eligible_columns": eligible_cols,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_p50": float(np.percentile(norms, 50)),
        "embedding_norm_p95": float(np.percentile(norms, 95)),
        "elapsed_seconds": time.time() - started,
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
