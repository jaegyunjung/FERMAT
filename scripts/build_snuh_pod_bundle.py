"""Build a small upload bundle for SNUH Pod training and ETL audits."""

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
    "scripts/audit_snuh_training_windows.py",
    "scripts/evaluate_snuh_checkpoint.py",
    "scripts/compare_snuh_smoke_arms.py",
    "scripts/run_snuh_task10.py",
    "scripts/run_snuh_scaling_sweep.py",
    "scripts/audit_snuh_measurement_staging.py",
    "docs/snuh_task10_smoke.md",
    "docs/snuh_pretraining_runbook.md",
]


def main():
    output = ROOT / "dist" / "snuh_task10_pod_bundle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        commit = None
    manifest = {
        "git_commit": commit,
        "files": FILES,
        "entrypoint": "python scripts/run_snuh_task10.py",
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            source = ROOT / relative_path
            if not source.exists():
                raise FileNotFoundError(source)
            archive.write(source, relative_path)
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2),
        )
    print(output)


if __name__ == "__main__":
    main()
