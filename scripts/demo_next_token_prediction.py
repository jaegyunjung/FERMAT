"""
Print next-token prediction examples from the Synthetic SNUH validation split.

Outputs:
  - logs/fermat_next_token_examples.md

This script uses model.forward() logits directly and decodes predictions through
vocab.csv. It does not use model.generate().
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Fermat, FermatConfig, TokenType
from utils import get_p2i, load_data


DISPLAY_IGNORED_TYPES = {
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
        help="directory containing val.bin and vocab.csv",
    )
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--num-examples", type=int, default=5, help="number of examples to write")
    parser.add_argument("--context-len", type=int, default=12, help="maximum context events to display")
    parser.add_argument("--top-k", type=int, default=5, help="number of predictions to display")
    parser.add_argument(
        "--out-md",
        default="logs/fermat_next_token_examples.md",
        help="markdown output path",
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


def load_vocab_maps(vocab_path):
    with open(vocab_path, newline="") as f:
        vocab = list(csv.DictReader(f))
    id_to_label = {int(row["token_id"]): row["label"] for row in vocab}
    id_to_type = {int(row["token_id"]): row["token_type_name"] for row in vocab}
    ignored_prediction_ids = []
    for row in vocab:
        token_id = int(row["token_id"])
        token_type = int(row["token_type"])
        if token_type in DISPLAY_IGNORED_TYPES:
            ignored_prediction_ids.append(token_id + 1)
    return id_to_label, id_to_type, sorted(set(ignored_prediction_ids))


def decode_token(raw_token_id, id_to_label, id_to_type):
    label = id_to_label.get(int(raw_token_id), f"<UNK:{raw_token_id}>")
    token_type = id_to_type.get(int(raw_token_id), "UNK")
    return f"{label} [{token_type}]"


def select_examples(val_data, num_examples, context_len):
    examples = []
    p2i = get_p2i(val_data)
    for row in p2i:
        start = int(row[0])
        length = int(row[1])
        if length < 2:
            continue
        patient = val_data[start:start + length]
        target_type = int(patient[-1, 3])
        if target_type in DISPLAY_IGNORED_TYPES:
            continue
        context_start = max(0, length - 1 - context_len)
        context = patient[context_start:length - 1]
        target = patient[length - 1]
        examples.append({
            "patient_id": int(patient[0, 0]),
            "context": context,
            "target": target,
        })
        if len(examples) >= num_examples:
            break
    return examples


def predict_topk(model, context, device, top_k, ignored_prediction_ids):
    block_size = model.config.block_size
    if len(context) > block_size:
        context = context[-block_size:]

    x = torch.tensor(context[:, 2].astype("int64") + 1, dtype=torch.long, device=device).unsqueeze(0)
    a = torch.tensor(context[:, 1].astype("float32"), dtype=torch.float32, device=device).unsqueeze(0)
    xt = torch.tensor(context[:, 3].astype("int64"), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _, _ = model(x, a, xt)
    next_logits = logits[0, -1].clone()
    next_logits[0] = float("-inf")
    for shifted_id in ignored_prediction_ids:
        if shifted_id < next_logits.numel():
            next_logits[shifted_id] = float("-inf")

    topk = torch.topk(next_logits, k=top_k, dim=-1).indices.tolist()
    return [idx - 1 for idx in topk]


def write_markdown(out_path, ckpt_path, data_dir, checkpoint, examples, predictions, id_to_label, id_to_type, top_k):
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# FERMAT next-token prediction examples\n\n")
        f.write(f"- Checkpoint: `{ckpt_path}`\n")
        f.write(f"- Data: `{data_dir}`\n")
        f.write(f"- Checkpoint step: {checkpoint.get('iter_num', 'unknown')}\n\n")
        f.write(
            "Note: the training pipeline shifts raw `vocab.csv` token ids by `+1` for model input/output. "
            "Predicted model indices shown here are decoded back to raw token ids by subtracting 1.\n\n"
        )

        for i, example in enumerate(examples, start=1):
            f.write(f"## Example {i}\n\n")
            f.write(f"- validation patient_id: {example['patient_id']}\n")
            f.write("- context tokens:\n")
            for event in example["context"]:
                raw_token_id = int(event[2])
                age = int(event[1])
                f.write(f"  - age={age}: {decode_token(raw_token_id, id_to_label, id_to_type)}\n")

            target_token_id = int(example["target"][2])
            target_age = int(example["target"][1])
            f.write(f"- actual next token: age={target_age}: {decode_token(target_token_id, id_to_label, id_to_type)}\n")
            f.write(f"- model top-{top_k} predicted next tokens:\n")
            for rank, pred_token_id in enumerate(predictions[i - 1], start=1):
                f.write(f"  - {rank}. {decode_token(pred_token_id, id_to_label, id_to_type)}\n")
            f.write("\n")


def main():
    args = parse_args()

    model, checkpoint = load_model(args.ckpt, args.device)
    val_data, has_types = load_data(os.path.join(args.data_dir, "val.bin"))
    if not has_types:
        raise ValueError("demo_next_token_prediction.py requires 4-column FERMAT data")

    id_to_label, id_to_type, ignored_prediction_ids = load_vocab_maps(
        os.path.join(args.data_dir, "vocab.csv")
    )
    examples = select_examples(val_data, args.num_examples, args.context_len)
    if not examples:
        raise RuntimeError("No eligible validation examples found")

    predictions = []
    for example in examples:
        pred_ids = predict_topk(
            model,
            example["context"],
            args.device,
            args.top_k,
            ignored_prediction_ids,
        )
        predictions.append(pred_ids)

    write_markdown(
        args.out_md,
        args.ckpt,
        args.data_dir,
        checkpoint,
        examples,
        predictions,
        id_to_label,
        id_to_type,
        args.top_k,
    )
    print(f"Demo examples written to {args.out_md}")


if __name__ == "__main__":
    main()
