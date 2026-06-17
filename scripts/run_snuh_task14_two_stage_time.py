#!/usr/bin/env python3
"""Run the two-stage same-day plus different-day time diagnostic on the Pod."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
POD_DATA_DIR = POD_ROOT / "etl" / "patient_001pct_seed_42"
POD_OUTPUT_ROOT = POD_ROOT / "out"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=POD_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=POD_OUTPUT_ROOT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--loss-dt-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    return parser.parse_args()


def read_bundle_id():
    path = ROOT / "bundle_manifest.json"
    if not path.exists():
        return "snuh_task14_two_stage_time"
    return json.loads(path.read_text(encoding="utf-8"))["bundle_id"]


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def find_task13_run(output_root, expected_weight):
    candidates = []
    for checkpoint_path in output_root.glob("snuh_task13*/ckpt.pt"):
        try:
            checkpoint = load_checkpoint(checkpoint_path)
        except Exception as exc:
            print(f"Skipping unreadable checkpoint {checkpoint_path}: {exc}")
            continue
        model_args = checkpoint.get("model_args", {})
        runtime_config = checkpoint.get("config", {})
        if not model_args.get("decoupled_time_head", False):
            continue
        if model_args.get("two_stage_time_head", False):
            continue
        weight = float(runtime_config.get("loss_dt_weight", math.nan))
        if not math.isclose(weight, expected_weight, abs_tol=1e-8):
            continue
        candidates.append(
            (
                checkpoint_path.stat().st_mtime,
                int(checkpoint.get("iter_num", -1)),
                checkpoint_path.parent,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "No adopted Task 13 checkpoint was found below "
            f"{output_root}. Expected decoupled_time_head=True, "
            f"two_stage_time_head=False, and loss_dt_weight={expected_weight}."
        )
    return max(candidates)[2]


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    resume_from = (
        args.resume_from.expanduser().resolve()
        if args.resume_from
        else find_task13_run(output_root, args.loss_dt_weight)
    )
    bundle_id = read_bundle_id()
    output_dir = output_root / bundle_id
    if output_dir.exists():
        output_dir = output_root / (
            f"{bundle_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        )

    command = [
        sys.executable,
        "scripts/run_snuh_task13_dt.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--resume-from",
        str(resume_from),
        "--config",
        "config/train_fermat_snuh_dt_two_stage_finetune.py",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    print(f"Warm-start Task 13 run: {resume_from}")
    print(f"Output directory: {output_dir}")
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=True,
    )


if __name__ == "__main__":
    main()
