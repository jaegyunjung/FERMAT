#!/usr/bin/env python3
"""Build the versioned Task 30 shared-rollout multi-outcome pilot bundle."""

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
    "config/snuh_task30_multi_outcome_rollout_pilot.json",
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
    bundle_id = f"snuh_task30_multi_outcome_rollout_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis": "Shared FERMAT rollout hit-rate and runtime pilot for diabetes, CKD, and dyslipidemia",
        "planned_main": {
            "patients": 160,
            "rollouts_per_patient": 60,
            "trajectories": 9600,
            "prior_measured_runtime_hours": 5.3755555556,
        },
        "raw_first": True,
        "claim_boundary": "Pilot only; not a calibrated risk curve or causal analysis",
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
