#!/usr/bin/env python3
"""Build the Task 30 observed-vs-generated event-density CPU bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/compare_snuh_task30_observed_generated_event_density.py"]


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    return f"{commit}{'_dirty' if dirty else ''}"


def main():
    digest = hashlib.sha256()
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    content_hash = digest.hexdigest()[:12]
    bundle_id = f"snuh_task30_event_density_comparison_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis": "CPU-only observed versus generated clinical-event density comparison",
        "loads_checkpoint": False,
        "uses_gpu": False,
        "runs_rollout": False,
        "observed_types": ["DX", "RX", "PX", "DTH"],
        "observed_input": (
            "/home/khdp-user/workspace/fermat-data/etl/"
            "patient_100pct_seed_42_with_genomics_tokens/test.bin"
        ),
        "generated_input": (
            "/home/khdp-user/workspace/fermat-data/task30/outputs/"
            "diabetes_main_rollout_20260716_1000x32/raw/main_trajectories.parquet"
        ),
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
