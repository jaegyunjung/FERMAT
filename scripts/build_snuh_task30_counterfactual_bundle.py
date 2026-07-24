#!/usr/bin/env python3
"""Build a versioned Pod bundle for Task 30 counterfactual risk curves."""

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
    "config/snuh_task30_diabetes_hba1c_q10_counterfactual.csv",
    "scripts/extract_snuh_task19_fermat_embeddings.py",
    "scripts/run_snuh_task20_cox_survival_models.py",
    "scripts/run_snuh_task30_counterfactual_risk_curves.py",
]


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    commit = git_output("rev-parse", "--short=7", "HEAD")
    dirty = bool(git_output("status", "--porcelain"))
    digest = hashlib.sha256()
    for relative_path in FILES:
        digest.update(relative_path.encode())
        digest.update((ROOT / relative_path).read_bytes())
    content_hash = digest.hexdigest()[:12]
    state = f"{commit}{'_dirty' if dirty else ''}"
    bundle_id = f"snuh_task30_counterfactual_risk_curves_{state}_{content_hash}"
    output = ROOT / "dist" / "task30" / f"{bundle_id}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_id": bundle_id,
        "commit": commit,
        "dirty": dirty,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "pod_extract_command": (
            "TASK_DIR=/home/khdp-user/workspace/fermat-data/task30\n"
            "mkdir -p \"$TASK_DIR\"/code \"$TASK_DIR\"/configs \"$TASK_DIR\"/outputs "
            "\"$TASK_DIR\"/logs \"$TASK_DIR\"/zips\n"
            "cd \"$TASK_DIR\"/code\n"
            f"unzip -o \"$TASK_DIR\"/zips/{output.name} -d {bundle_id}-code\n"
            f"cd {bundle_id}-code"
        ),
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            archive.write(ROOT / relative_path, relative_path)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(output)
    print(manifest["pod_extract_command"])


if __name__ == "__main__":
    main()
