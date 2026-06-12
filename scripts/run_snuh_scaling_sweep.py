"""Run short SNUH model/context scaling trials and summarize efficiency."""

import argparse
import csv
import json
import os
import shutil
import statistics
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
DEFAULT_TRIALS = [
    {
        "name": "tiny_ctx256",
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 256,
        "batch_size": 16,
    },
    {
        "name": "tiny_ctx512",
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 512,
        "batch_size": 8,
    },
    {
        "name": "small_ctx256",
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 256,
        "block_size": 256,
        "batch_size": 8,
    },
    {
        "name": "small_ctx512",
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 256,
        "block_size": 512,
        "batch_size": 4,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--policy",
        choices=["baseline", "lab-context"],
        default="lab-context",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--trials-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_bundle_id():
    manifest_path = ROOT / "bundle_manifest.json"
    if not manifest_path.exists():
        return "snuh_task12_scaling_sweep"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("bundle_id", "snuh_task12_scaling_sweep")


def default_data_dir():
    if POD_DATA_DIR.exists():
        return POD_DATA_DIR
    local = ROOT / "outputs/snuh_tokenization_etl/patient_001pct_seed_42"
    if local.exists():
        return local
    raise FileNotFoundError(
        f"Missing SNUH 1% pilot at {POD_DATA_DIR} or {local}"
    )


def default_output_dir():
    base = POD_OUTPUT_ROOT if POD_STORAGE.exists() else ROOT / "out"
    bundle_id = read_bundle_id()
    candidate = base / bundle_id
    if not candidate.exists():
        return candidate
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return base / f"{bundle_id}_{timestamp}"


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
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
        return process.wait()


def read_metrics(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_evaluation(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_trial(trial, trial_dir, return_code):
    records = read_metrics(trial_dir / "metrics.jsonl")
    train = [row for row in records if "train/loss" in row]
    validation = [row for row in records if "val/objective_loss" in row]
    speeds = [
        row["train/targets_per_second"]
        for row in train[1:]
        if row.get("train/targets_per_second", 0) > 0
    ]
    manifest_path = trial_dir / "run_manifest.json"
    parameter_count = None
    checkpoint_path = trial_dir / "ckpt.pt"
    if checkpoint_path.exists():
        import torch
        from model import Fermat, FermatConfig

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        parameter_count = Fermat(
            FermatConfig(**checkpoint["model_args"])
        ).get_num_params()
    evaluation = read_evaluation(trial_dir / "evaluation.json")
    clinical = evaluation.get("clinical_only_softmax", {})
    new_clinical = evaluation.get("new_clinical", {})
    repeated_clinical = evaluation.get("repeated_clinical", {})
    return {
        **trial,
        "status": "ok" if return_code == 0 else "failed",
        "return_code": return_code,
        "parameter_count": parameter_count,
        "first_train_objective": (
            train[0]["train/loss"] if train else None
        ),
        "last_train_objective": (
            train[-1]["train/loss"] if train else None
        ),
        "best_val_objective": (
            min(row["val/objective_loss"] for row in validation)
            if validation
            else None
        ),
        "clinical_ce": clinical.get("cross_entropy"),
        "clinical_top1": clinical.get("top1_accuracy"),
        "clinical_top5": clinical.get("top5_accuracy"),
        "new_clinical_top1": new_clinical.get("top1_accuracy"),
        "repeated_clinical_top1": repeated_clinical.get("top1_accuracy"),
        "median_effective_targets_per_second": (
            statistics.median(speeds) if speeds else None
        ),
        "max_cuda_memory_gb": max(
            (row.get("system/max_cuda_memory_gb", 0) for row in train),
            default=None,
        ),
        "manifest_present": manifest_path.exists(),
    }


def write_reports(rows, output_dir):
    csv_path = output_dir / "scaling_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SNUH scaling sweep",
        "",
        (
            "| trial | status | parameters | context | batch | best val CE | "
            "clinical CE | top-1 | top-5 | targets/s | VRAM GB |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key, fmt):
            item = row[key]
            return "NA" if item is None else format(item, fmt)

        lines.append(
            f"| {row['name']} | {row['status']} | "
            f"{value('parameter_count', ',')} | {row['block_size']} | "
            f"{row['batch_size']} | {value('best_val_objective', '.4f')} | "
            f"{value('clinical_ce', '.4f')} | "
            f"{value('clinical_top1', '.4%')} | "
            f"{value('clinical_top5', '.4%')} | "
            f"{value('median_effective_targets_per_second', '.1f')} | "
            f"{value('max_cuda_memory_gb', '.2f')} |"
        )
    (output_dir / "scaling_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.max_iters < args.eval_interval:
        raise ValueError("max-iters must be at least eval-interval")
    data_dir = (args.data_dir or default_data_dir()).resolve()
    output = (args.output_dir or default_output_dir()).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8-sig")
    )
    vocab_size = int(manifest["model_vocab_size"])
    trials = (
        json.loads(args.trials_json.read_text(encoding="utf-8-sig"))
        if args.trials_json
        else DEFAULT_TRIALS
    )
    config = (
        "config/train_fermat_snuh_pilot.py"
        if args.policy == "baseline"
        else "config/train_fermat_snuh_pilot_lab_context.py"
    )

    rows = []
    for trial in trials:
        trial_dir = output / trial["name"]
        trial_dir.mkdir()
        command = [
            sys.executable,
            "train.py",
            config,
            f"--dataset_dir={data_dir}",
            f"--out_dir={trial_dir}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--vocab_size={vocab_size}",
            f"--n_layer={trial['n_layer']}",
            f"--n_head={trial['n_head']}",
            f"--n_embd={trial['n_embd']}",
            f"--block_size={trial['block_size']}",
            f"--batch_size={trial['batch_size']}",
            f"--max_iters={args.max_iters}",
            f"--eval_interval={args.eval_interval}",
            f"--eval_iters={args.eval_iters}",
            f"--log_interval={args.log_interval}",
            f"--lr_decay_iters={args.max_iters}",
        ]
        return_code = run(command, trial_dir / "training.log")
        checkpoint = trial_dir / "ckpt.pt"
        if return_code == 0 and checkpoint.exists():
            evaluation_code = run([
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
                str(trial_dir / "evaluation.json"),
            ], trial_dir / "evaluation.log")
            if evaluation_code:
                return_code = evaluation_code
        rows.append(summarize_trial(trial, trial_dir, return_code))
        write_reports(rows, output)

    print(f"Scaling report: {output / 'scaling_summary.md'}")


if __name__ == "__main__":
    main()
