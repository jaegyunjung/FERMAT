#!/usr/bin/env python3
"""Build the HbA1c-defined ADM-timing feasibility bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/audit_snuh_task30_hba1c_adm_timing_feasibility.py",
    "scripts/audit_snuh_task30_adm_timing_feasibility.py",
]


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=ROOT, check=False
    ).returncode != 0
    return f"{commit}{'_dirty' if dirty else ''}"


def main():
    digest = hashlib.sha256()
    for relative in FILES:
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    content_hash = digest.hexdigest()[:12]
    bundle_id = f"snuh_task30_hba1c_adm_timing_feasibility_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis_type": "train_val_hba1c_restricted_feasibility",
        "time_zero": "first HbA1c >=6.5%",
        "test_data_queried": False,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
