"""Build a versioned SNUH Task 11 pod bundle."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "scripts/audit_snuh_measurement_staging.py",
    "scripts/run_snuh_task11_measurement_staging.py",
    "docs/snuh_pretraining_runbook.md",
]


def git_state():
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
        return commit, dirty
    except Exception:
        return None, None


def content_hash():
    digest = hashlib.sha256()
    for relative_path in FILES:
        source = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    commit, dirty = git_state()
    digest = content_hash()
    short_commit = commit[:7] if commit else "nogit"
    state = "dirty" if dirty else "clean" if dirty is not None else "unknown"
    bundle_id = (
        f"snuh_task11_measurement_staging_"
        f"{short_commit}_{state}_{digest[:12]}"
    )
    output = ROOT / "dist" / f"{bundle_id}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "bundle_id": bundle_id,
        "bundle_sha256": digest,
        "git_commit": commit,
        "git_dirty": dirty,
        "files": FILES,
        "entrypoint": "python scripts/run_snuh_task11_measurement_staging.py",
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
