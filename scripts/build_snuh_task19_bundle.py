#!/usr/bin/env python3
"""Build a versioned Pod bundle for SNUH disease-risk benchmark setup."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/build_snuh_task19_disease_risk_candidates.py",
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
    bundle_id = f"snuh_task19_disease_risk_{state}_{content_hash}"
    output = ROOT / "dist" / "task19_disease_risk" / f"{bundle_id}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_id": bundle_id,
        "commit": commit,
        "dirty": dirty,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task19",
        "pod_extract_command": (
            "TASK_DIR=/home/khdp-user/workspace/fermat-data/task19\n"
            "mkdir -p \"$TASK_DIR\"/code \"$TASK_DIR\"/outputs \"$TASK_DIR\"/logs \"$TASK_DIR\"/zips\n"
            "cd \"$TASK_DIR\"/code\n"
            f"unzip -o \"$TASK_DIR\"/zips/{output.name} -d {bundle_id}-code\n"
            f"cd {bundle_id}-code"
        ),
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            archive.write(ROOT / relative_path, relative_path)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    print(output)
    print(manifest["pod_extract_command"])


if __name__ == "__main__":
    main()
