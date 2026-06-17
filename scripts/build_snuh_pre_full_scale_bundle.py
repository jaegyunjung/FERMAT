#!/usr/bin/env python3
"""Build a versioned Pod bundle for pre-full-scale SNUH checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
FILES = [
    "model.py",
    "utils.py",
    "scripts/audit_snuh_training_windows.py",
    "scripts/evaluate_snuh_checkpoint.py",
    "scripts/run_snuh_pre_full_scale_checks.py",
]


def git_output(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def main():
    commit = git_output("rev-parse", "--short=7", "HEAD")
    dirty = bool(git_output("status", "--porcelain"))
    state = f"{commit}{'_dirty' if dirty else ''}"

    digest = hashlib.sha256()
    for relative_path in FILES:
        path = ROOT / relative_path
        digest.update(relative_path.encode())
        digest.update(path.read_bytes())
    content_hash = digest.hexdigest()[:12]

    stem = f"snuh_pre_full_scale_checks_{state}_{content_hash}"
    bundle_path = DIST / f"{stem}.zip"
    DIST.mkdir(exist_ok=True)
    manifest = {
        "bundle": bundle_path.name,
        "commit": commit,
        "dirty": dirty,
        "content_hash": content_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": FILES,
        "pod_extract_command": (
            "cd /home/khdp-user/workspace/fermat-data && "
            f"mkdir -p {stem}-code && "
            f"unzip -o {bundle_path.name} -d {stem}-code && "
            f"cd {stem}-code"
        ),
        "pod_run_command": "python scripts/run_snuh_pre_full_scale_checks.py",
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for relative_path in FILES:
            bundle.write(ROOT / relative_path, relative_path)
        bundle.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2) + "\n",
        )

    print(bundle_path)
    print(manifest["pod_extract_command"])
    print(manifest["pod_run_command"])


if __name__ == "__main__":
    main()
