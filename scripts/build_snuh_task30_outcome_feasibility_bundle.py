#!/usr/bin/env python3
"""Build the versioned Task 30 multi-outcome feasibility-audit Pod bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "scripts/audit_snuh_task30_outcome_feasibility.py": (
        "code/audit_snuh_task30_outcome_feasibility.py"
    ),
    "config/snuh_task30_outcome_feasibility.csv": (
        "config/snuh_task30_outcome_feasibility.csv"
    ),
}


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return f"{commit}{'_dirty' if dirty else ''}"


def main():
    digest = hashlib.sha256()
    for source, archive_name in FILES.items():
        path = ROOT / source
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(source.encode("utf-8"))
        digest.update(archive_name.encode("utf-8"))
        digest.update(path.read_bytes())
    content_hash = digest.hexdigest()[:12]
    bundle_id = f"snuh_task30_outcome_feasibility_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis": "CPU-only multi-outcome feasibility audit; no Cox fitting and no rollout",
        "default_outcomes": {
            "resolved": "31 reviewed Task19 phenotype groups plus all-cause mortality",
            "definition_required": [
                "heart_failure",
                "acute_myocardial_infarction",
                "dementia",
                "copd",
                "broad_cancer",
            ],
        },
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, archive_name in FILES.items():
            archive.write(ROOT / source, archive_name)
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
