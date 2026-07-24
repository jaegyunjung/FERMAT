#!/usr/bin/env python3
"""Build the no-dependency Task 30 curve validation and SVG bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/verify_snuh_task30_observed_mace_curve.py"]


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
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
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    content_hash = digest.hexdigest()[:12]
    bundle_id = f"snuh_task30_curve_check_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "reruns_database_query": False,
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
