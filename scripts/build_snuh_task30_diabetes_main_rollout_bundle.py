#!/usr/bin/env python3
"""Build the approved Task 30 diabetes main population-rollout Pod bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "model.py",
    "utils.py",
    "scripts/run_snuh_task27_primary_direct_risk_dry_run.py",
    "scripts/run_snuh_task30_multi_outcome_rollout_pilot.py",
    "scripts/run_snuh_task30_diabetes_rollout_speed_benchmark.py",
    "scripts/run_snuh_task30_diabetes_main_rollout.py",
    "config/snuh_task30_diabetes_main_rollout_1000x32.json",
]


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
    bundle_id = f"snuh_task30_diabetes_main_rollout_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_repo_dir": "/home/khdp-user/workspace/fermat",
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis": "Task 30 diabetes main population rollout",
        "approved_design": {
            "sampling": "outcome-blind simple random sample",
            "patients": 1000,
            "rollouts_per_patient": 32,
            "batch_size": 8,
            "total_trajectories": 32000,
        },
        "raw_first": True,
        "resume_validation": "run fingerprint plus per-patient trajectory keys",
        "stage_boundary": "Population and saved individual curves only; perturbations deferred.",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
