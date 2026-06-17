#!/usr/bin/env python3
"""Build a versioned Pod bundle for the Task 14 two-stage time model."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "model.py",
    "train.py",
    "utils.py",
    "configurator.py",
    "config/train_fermat_snuh_pilot.py",
    "config/train_fermat_snuh_pilot_lab_context.py",
    "config/train_fermat_snuh_dt.py",
    "config/train_fermat_snuh_dt_finetune.py",
    "config/train_fermat_snuh_dt_decoupled_finetune.py",
    "config/train_fermat_snuh_dt_two_stage_finetune.py",
    "scripts/evaluate_snuh_checkpoint.py",
    "scripts/run_snuh_task13_dt.py",
    "scripts/run_snuh_task14_two_stage_time.py",
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
    bundle_id = f"snuh_task14_two_stage_time_{state}_{content_hash}"
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
        "pod_run_command": "python scripts/run_snuh_task14_two_stage_time.py",
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
    print(manifest["pod_run_command"])


if __name__ == "__main__":
    main()
