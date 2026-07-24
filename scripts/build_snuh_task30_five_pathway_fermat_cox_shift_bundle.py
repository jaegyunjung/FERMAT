#!/usr/bin/env python3
"""Build the Task 30 five-pathway FERMAT+Cox shift Pod bundle."""

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
    "scripts/extract_snuh_task19_fermat_embeddings.py",
    "scripts/run_snuh_task20_cox_survival_models.py",
    "scripts/run_snuh_task30_counterfactual_risk_curves.py",
    "scripts/run_snuh_task30_five_pathway_fermat_cox_shift.py",
    "scripts/run_snuh_task30_five_pathway_fermat_cox_shift_pod.sh",
    "config/snuh_task30_five_pathway_fermat_cox_shift.json",
]


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "nogit"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = True
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
    bundle_id = f"snuh_task30_five_pathway_fermat_cox_shift_{git_state()}_{content_hash}"
    output_dir = ROOT / "dist" / "task30"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "content_hash": content_hash,
        "files": FILES,
        "pod_task_dir": "/home/khdp-user/workspace/fermat-data/task30",
        "contains_patient_data": False,
        "analysis": "Five-pathway continuous diagnosis-time shift with FERMAT embedding-only Cox",
        "runner_version": "20260718_paired_embedding_delta_v3",
        "pathways": 5,
        "unique_targets": 4,
        "patient_arm_rows": 6912,
        "nonzero_shift_embeddings": 6144,
        "effect_embedding": "saved original plus within-run paired edited-minus-original embedding delta",
        "embedding_preflight": "manifest, sequence order and length, mean absolute error, relative L2, and cosine similarity",
        "runs_rollout": False,
        "raw_first": True,
        "resume_unit": "one pathway/source-role/shift arm",
        "interactive_launcher_pauses_before_shell_exit": True,
        "future_rollout_requirement": "save every generated token ID, type, date, and position before summary",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    print(output)
    print(f"bundle_id={bundle_id}")
    print(f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
