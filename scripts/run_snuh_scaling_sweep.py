"""Run short SNUH model/context scaling trials and summarize efficiency."""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("out/snuh-scaling"))
    parser.add_argument(
        "--policy",
        choices=["baseline", "lab-context"],
        default="baseline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--trials-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        "| trial | status | parameters | context | batch | best val | targets/s | VRAM GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(key, fmt):
            item = row[key]
            return "NA" if item is None else format(item, fmt)

        lines.append(
            f"| {row['name']} | {row['status']} | "
            f"{value('parameter_count', ',')} | {row['block_size']} | "
            f"{row['batch_size']} | {value('best_val_objective', '.4f')} | "
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
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
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
        ]
        return_code = run(command, trial_dir / "training.log")
        rows.append(summarize_trial(trial, trial_dir, return_code))
        write_reports(rows, output)

    print(f"Scaling report: {output / 'scaling_summary.md'}")


if __name__ == "__main__":
    main()
