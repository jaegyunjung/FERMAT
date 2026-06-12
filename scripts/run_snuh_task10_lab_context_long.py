"""Extend the Task 10 LAB-context CE-only arm and evaluate its best checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POD_STORAGE = Path("/home/khdp-user/workspace/fermat-data")
POD_DATA_DIR = POD_STORAGE / "etl/patient_001pct_seed_42"
POD_OUTPUT_ROOT = POD_STORAGE / "out"
POD_CHECKPOINT_ROOTS = [
    POD_STORAGE / "out",
    POD_STORAGE / "task10",
    POD_STORAGE / "task10-ce-only",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Defaults to the SNUH 1%% pilot under fermat-data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a versioned directory under fermat-data/out",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Task 10 LAB-context directory containing the best ckpt.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_bundle_id():
    manifest = ROOT / "bundle_manifest.json"
    if not manifest.exists():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return document.get("bundle_id")


def default_data_dir():
    if POD_DATA_DIR.exists():
        return POD_DATA_DIR
    local = ROOT / "outputs/snuh_tokenization_etl/patient_001pct_seed_42"
    if local.exists():
        return local
    raise FileNotFoundError(
        "Could not find the 1% SNUH ETL pilot. Expected "
        f"{POD_DATA_DIR} or {local}"
    )


def default_output_dir():
    bundle_id = read_bundle_id() or "snuh_task10_lab_context_long"
    base = POD_OUTPUT_ROOT if POD_STORAGE.exists() else ROOT / "out"
    candidate = base / bundle_id
    if not candidate.exists():
        return candidate
    return base / f"{bundle_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def checkpoint_is_lab_context_ce_only(checkpoint_path):
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    ignore_types = {int(value) for value in config.get("ignore_types", [])}
    compatible = (
        float(config.get("loss_dt_weight", 1.0)) == 0.0
        and int(4) in ignore_types
        and config.get("checkpoint_metric") == "ce"
    )
    return compatible, int(checkpoint.get("iter_num", 0))


def discover_resume_dir(output, max_iters):
    search_roots = (
        POD_CHECKPOINT_ROOTS
        if POD_STORAGE.exists()
        else [ROOT.parent, ROOT]
    )
    candidates = []
    seen = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for checkpoint in search_root.rglob("ckpt.pt"):
            if checkpoint in seen or output in checkpoint.parents:
                continue
            seen.add(checkpoint)
            path_text = str(checkpoint).lower()
            if "lab-context" not in path_text and "lab_context" not in path_text:
                continue
            try:
                compatible, step = checkpoint_is_lab_context_ce_only(checkpoint)
                if compatible and step < max_iters:
                    candidates.append((step, checkpoint))
            except Exception as exc:
                print(f"Skipping unreadable checkpoint {checkpoint}: {exc}")
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (item[0], item[1].stat().st_mtime, str(item[1])),
        reverse=True,
    )
    step, selected = candidates[0]
    print(
        "Auto-selected LAB-context CE-only checkpoint: "
        f"{selected} (step {step})"
    )
    return selected.parent


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def prepare_output(output, resume_from, overwrite):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    if resume_from is None:
        return "scratch", None

    resume_from = resume_from.resolve()
    checkpoint = resume_from / "ckpt.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
    shutil.copy2(checkpoint, output / "ckpt.pt")
    return "resume", resume_from


def read_metrics(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_summary(output, evaluation_path, resumed_from):
    records = read_metrics(output / "metrics.jsonl")
    validation = [
        row for row in records
        if "val/loss_ce" in row
    ]
    if not validation:
        raise RuntimeError("No validation records were written to metrics.jsonl")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    best_validation = min(validation, key=lambda row: row["val/loss_ce"])

    lines = [
        "# Task 10 LAB-context CE-only extended run",
        "",
        f"- Resumed from: `{resumed_from}`" if resumed_from else "- Started from scratch",
        f"- Best checkpoint step: `{evaluation['checkpoint_step']}`",
        f"- Best sampled validation CE: `{best_validation['val/loss_ce']:.4f}`",
        f"- Best sampled validation step: `{best_validation['iter']}`",
        (
            "- Deterministic clinical-only CE: "
            f"`{evaluation['clinical_only_softmax']['cross_entropy']:.4f}`"
        ),
        (
            "- Deterministic clinical-only top-1: "
            f"`{evaluation['clinical_only_softmax']['top1_accuracy']:.4%}`"
        ),
        (
            "- Train-unigram clinical top-1: "
            f"`{evaluation['train_clinical_unigram']['top1_accuracy']:.4%}`"
        ),
        f"- New clinical top-1: `{evaluation['new_clinical']['top1_accuracy']:.4%}`",
        (
            "- Repeated clinical top-1: "
            f"`{evaluation['repeated_clinical']['top1_accuracy']:.4%}`"
        ),
        "",
        "## Validation CE trajectory",
        "",
        "| step | validation CE | validation targets |",
        "|---:|---:|---:|",
    ]
    for row in validation:
        lines.append(
            f"| {row['iter']} | {row['val/loss_ce']:.4f} | "
            f"{row['val/eval_targets']} |"
        )
    lines.extend([
        "",
        "Waiting-time metrics remain disabled in this CE-only diagnostic.",
        "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.max_iters < args.eval_interval:
        raise ValueError("max-iters must be at least eval-interval")

    data_dir = (args.data_dir or default_data_dir()).resolve()
    output = (args.output_dir or default_output_dir()).resolve()
    resume_from = args.resume_from
    if resume_from is None:
        resume_from = discover_resume_dir(output, args.max_iters)
    if resume_from is None:
        print("No compatible checkpoint found; starting LAB-context from scratch.")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output}")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing ETL manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest["model_vocab_size"])

    init_from, resumed_from = prepare_output(
        output,
        resume_from,
        args.overwrite,
    )
    run([
        sys.executable,
        "train.py",
        "config/train_fermat_snuh_lab_context_long.py",
        f"--dataset_dir={data_dir}",
        f"--out_dir={output}",
        f"--init_from={init_from}",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        f"--vocab_size={vocab_size}",
        f"--batch_size={args.batch_size}",
        f"--block_size={args.block_size}",
        f"--max_iters={args.max_iters}",
        f"--lr_decay_iters={args.max_iters}",
        f"--eval_interval={args.eval_interval}",
        f"--eval_iters={args.eval_iters}",
        f"--log_interval={args.log_interval}",
    ], output / "training.log")

    checkpoint = output / "ckpt.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not produce {checkpoint}")
    evaluation_path = output / "evaluation.json"
    run([
        sys.executable,
        "scripts/evaluate_snuh_checkpoint.py",
        "--ckpt",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.eval_batch_size),
        "--output",
        str(evaluation_path),
    ], output / "evaluation.log")
    write_summary(output, evaluation_path, resumed_from)
    print(f"Extended Task 10 report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
