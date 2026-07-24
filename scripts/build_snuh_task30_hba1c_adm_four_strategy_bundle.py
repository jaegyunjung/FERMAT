#!/usr/bin/env python3
"""Build the complete four-strategy HbA1c ADM Task 30 Pod bundle."""

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
    "scripts/audit_snuh_task30_adm_timing_feasibility.py",
    "scripts/audit_snuh_task30_hba1c_adm_timing_feasibility.py",
    "scripts/extract_snuh_task19_fermat_embeddings.py",
    "scripts/run_snuh_task27_primary_direct_risk_dry_run.py",
    "scripts/snuh_task30_adm_strategy.py",
    "scripts/snuh_task30_adm_ccw_core.py",
    "scripts/extract_snuh_task30_hba1c_adm_ccw_inputs.py",
    "scripts/run_snuh_task30_hba1c_adm_ccw.py",
    "scripts/extract_snuh_task30_hba1c_monthly_embeddings.py",
    "scripts/fit_snuh_task30_hba1c_embedding_hazard.py",
    "scripts/run_snuh_task30_hba1c_adm_conditioned_fermat.py",
]


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "nogit_dirty"
    return f"{commit}{'_dirty' if dirty else ''}"


def main():
    digest = hashlib.sha256()
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    content_hash = digest.hexdigest()[:12]
    bundle_id = f"snuh_task30_hba1c_adm_four_strategy_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "time_zero": "first observed HbA1c >=6.5%",
        "strategies": [
            "initiate ADM within 3 months",
            "initiate ADM within 6 months",
            "initiate ADM within 12 months",
            "no ADM initiation within 12 months",
        ],
        "real_data_estimator": "clone-censor-weight with train-fit censoring models and val diagnostics",
        "fermat_estimators": [
            "conditioned rollout generated-death curve",
            "conditioned context rollout plus monthly FERMAT-embedding death-hazard curve",
        ],
        "test_used_by_default": False,
        "raw_first": True,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
