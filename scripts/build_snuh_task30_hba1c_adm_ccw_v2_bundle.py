#!/usr/bin/env python3
"""Build the corrected Task 30 monthly CCW v2 Pod bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/audit_snuh_task30_adm_timing_feasibility.py",
    "scripts/audit_snuh_task30_hba1c_adm_timing_feasibility.py",
    "scripts/snuh_task30_adm_ccw_core.py",
    "scripts/extract_snuh_task30_hba1c_adm_ccw_inputs.py",
    "scripts/run_snuh_task30_hba1c_adm_ccw.py",
    "scripts/audit_snuh_task30_hba1c_adm_ccw_results.py",
    "scripts/run_snuh_task30_hba1c_adm_ccw_v2_pod.sh",
]


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
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
    bundle_id = f"snuh_task30_hba1c_adm_ccw_v2_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "method_fixes": [
            "end-of-grace IPCW for initiation arms and monthly cumulative IPCW for no-initiation",
            "train-derived five-knot restricted cubic splines for age and follow-up days",
            "ADM excluded from baseline and monthly generic drug utilization",
            "reviewed comorbidity and prespecified ATC comedication covariates",
            "separate train/val patient bootstrap with IPCW and P99-cap refit in every draw",
            "single-pass design-matrix construction with one categorical reference level",
            "redundant lag age/calendar variables excluded when baseline and follow-up time are present",
            "damped Newton ridge-logistic fit with explicit convergence diagnostics",
            "existing completed v3 input can be reused without another database extraction",
            "stage-aware Pod failure summary prints the actual censor-model diagnostic file",
            "denominator-only unstabilized IPCW is calculated and saved separately for balance diagnostics",
            "balance-only mode reuses completed inputs and does not repeat the completed bootstrap",
        ],
        "reuses_v2_checkpoints_without_reusing_old_drug_counts": True,
        "test_used": False,
        "raw_first": True,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
