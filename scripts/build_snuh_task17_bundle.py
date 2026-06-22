#!/usr/bin/env python3
"""Build a versioned Pod bundle for Task 17 genomic variant audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/build_snuh_task17_bundle.py",
    "scripts/audit_snuh_genomic_variant_feasibility.py",
    "scripts/task17_profile_source_value_columns.py",
    "scripts/probe_snuh_task17_l25_codes.py",
    "scripts/check_snuh_task17_status.py",
    "scripts/review_task17_biomarker_outputs.py",
    "docs/snuh_task17_genomic_variant_audit.md",
]


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    commit = git_output("rev-parse", "--short=7", "HEAD")
    dirty = bool(git_output("status", "--porcelain"))
    digest = hashlib.sha256()
    for relative_path in FILES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    content_hash = digest.hexdigest()[:12]
    state = f"{commit}{'_dirty' if dirty else ''}"
    bundle_id = f"snuh_task17_genomic_variant_audit_{state}_{content_hash}"
    output = ROOT / "dist" / f"{bundle_id}.zip"
    output.parent.mkdir(exist_ok=True)
    manifest = {
        "bundle_id": bundle_id,
        "commit": commit,
        "dirty": dirty,
        "content_hash": content_hash,
        "files": FILES,
        "pod_extract_command": (
            "cd /home/khdp-user/workspace/fermat-data && "
            f"mkdir -p {bundle_id}-code && "
            f"unzip -o {output.name} -d {bundle_id}-code && "
            f"cd {bundle_id}-code"
        ),
        "pod_run_command": (
            "mkdir -p logs && "
            "python scripts/audit_snuh_genomic_variant_feasibility.py 2>&1 | "
            "tee logs/task17_run.log"
        ),
        "pod_resume_command": (
            "mkdir -p logs && "
            "python scripts/audit_snuh_genomic_variant_feasibility.py 2>&1 | "
            "tee logs/task17_run.log"
        ),
        "pod_source_value_profile_command": (
            "python scripts/task17_profile_source_value_columns.py"
        ),
        "pod_l25_code_probe_command": "python scripts/probe_snuh_task17_l25_codes.py",
        "pod_status_command": "python scripts/check_snuh_task17_status.py",
        "pod_review_command": (
            "python scripts/review_task17_biomarker_outputs.py "
            "--report /home/khdp-user/workspace/fermat-data/scripts/outputs/"
            "task17_molecular_biomarker_audit/task17_biomarker_review.md"
        ),
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            archive.write(ROOT / relative_path, relative_path)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    print(output)
    print(manifest["pod_extract_command"])
    print(manifest["pod_run_command"])
    print(manifest["pod_l25_code_probe_command"])
    print(manifest["pod_status_command"])
    print(manifest["pod_review_command"])


if __name__ == "__main__":
    main()
