"""Build a small upload bundle for SNUH Pod training and ETL audits."""

import argparse
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
    "config/train_fermat_snuh_lab_context_long.py",
    "scripts/audit_snuh_training_windows.py",
    "scripts/evaluate_snuh_checkpoint.py",
    "scripts/compare_snuh_smoke_arms.py",
    "scripts/run_snuh_task10.py",
    "scripts/run_snuh_task10_lab_context_long.py",
    "scripts/run_snuh_scaling_sweep.py",
    "scripts/audit_snuh_measurement_staging.py",
    "docs/snuh_task10_smoke.md",
    "docs/snuh_task10_lab_context_long.md",
    "docs/snuh_pretraining_runbook.md",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="snuh_task10_lab_context_long",
        help="Human-readable bundle purpose used in the versioned filename",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dist",
    )
    return parser.parse_args()


def content_hash():
    digest = hashlib.sha256()
    for relative_path in FILES:
        source = ROOT / relative_path
        if not source.exists():
            raise FileNotFoundError(source)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    args = parse_args()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip())
    except Exception:
        commit = None
        dirty = None
    bundle_hash = content_hash()
    short_commit = commit[:7] if commit else "nogit"
    state = "dirty" if dirty else "clean" if dirty is not None else "unknown"
    bundle_id = f"{args.label}_{short_commit}_{state}_{bundle_hash[:12]}"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{bundle_id}.zip"
    manifest = {
        "bundle_id": bundle_id,
        "bundle_sha256": bundle_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "files": FILES,
        "entrypoints": {
            "task10_smoke": "python scripts/run_snuh_task10.py",
            "task10_lab_context_long": (
                "python scripts/run_snuh_task10_lab_context_long.py"
            ),
        },
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            source = ROOT / relative_path
            archive.write(source, relative_path)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2),
        )
    print(output)


if __name__ == "__main__":
    main()
