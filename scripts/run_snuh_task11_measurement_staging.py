"""Run the SNUH Task 11 measurement staging gate on block storage."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POD_STORAGE = Path("/home/khdp-user/workspace/fermat-data")
POD_OUTPUT_ROOT = POD_STORAGE / "out"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-buckets", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-indexes", action="store_true")
    return parser.parse_args()


def read_bundle_id():
    manifest_path = ROOT / "bundle_manifest.json"
    if not manifest_path.exists():
        return "snuh_task11_measurement_staging"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("bundle_id", "snuh_task11_measurement_staging")


def default_output_dir():
    base = POD_OUTPUT_ROOT if POD_STORAGE.exists() else ROOT / "out"
    bundle_id = read_bundle_id()
    candidate = base / bundle_id
    if not candidate.exists():
        return candidate
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return base / f"{bundle_id}_{timestamp}"


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def ensure_dependencies():
    if importlib.util.find_spec("psycopg") is not None:
        return
    print("psycopg is not installed; installing psycopg[binary]>=3 ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "psycopg[binary]>=3",
        ],
        cwd=ROOT,
        check=True,
    )
    if importlib.util.find_spec("psycopg") is None:
        raise RuntimeError("psycopg installation completed but import is unavailable")


def format_gb(value):
    return f"{value:.2f} GB"


def write_summary(result, output_dir):
    counts = result["stage_counts"]
    storage = result["stage_storage"]
    projection = result["linear_full_scale_projection"]
    overlap_rate = (
        counts["numeric_and_categorical_rows"] / counts["rows"]
        if counts["rows"]
        else 0.0
    )
    gate = (
        "REVIEW_5PCT"
        if result["patient_buckets"] == 1
        else "READY_FOR_FULL_ETL_REVIEW"
    )
    lines = [
        "# Task 11 measurement staging gate",
        "",
        f"- Gate status: **{gate}**",
        f"- Patient sample: `{result['patient_buckets']}%`",
        f"- Patients: `{counts['patients']:,}`",
        f"- Staged rows: `{counts['rows']:,}`",
        f"- Numeric rows: `{counts['numeric_rows']:,}`",
        f"- Categorical rows: `{counts['categorical_rows']:,}`",
        f"- Numeric/categorical overlap: `{overlap_rate:.4%}`",
        f"- Cohort creation: `{result['cohort_create_seconds']:.1f}s`",
        f"- Stage creation: `{result['stage_create_seconds']:.1f}s`",
        f"- Stage analyze: `{result['stage_analyze_seconds']:.1f}s`",
        (
            f"- Index creation: `{result.get('stage_index_seconds', 0.0):.1f}s`"
            if result["create_indexes"]
            else "- Index creation: `disabled`"
        ),
        f"- Sample table size: `{format_gb(storage['table_gb'])}`",
        f"- Sample index size: `{format_gb(storage['index_gb'])}`",
        f"- Linear full-scale table projection: `{format_gb(projection['table_gb'])}`",
        f"- Linear full-scale index projection: `{format_gb(projection['index_gb'])}`",
        f"- Linear full-scale total projection: `{format_gb(projection['total_gb'])}`",
        "",
    ]
    if result["patient_buckets"] == 1:
        lines.extend([
            "The 1% result is a screening estimate. Run the same gate at 5%",
            "before adopting the staging design for the full ETL.",
            "",
        ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if not 1 <= args.patient_buckets <= 100:
        raise ValueError("patient-buckets must be between 1 and 100")
    ensure_dependencies()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "measurement_staging_audit.json"
    print(f"Output directory: {output_dir}")

    command = [
        sys.executable,
        "scripts/audit_snuh_measurement_staging.py",
        "--patient-buckets",
        str(args.patient_buckets),
        "--output",
        str(result_path),
    ]
    if not args.no_indexes:
        command.append("--create-indexes")
    run(command, output_dir / "task11.log")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    write_summary(result, output_dir)
    print(f"Task 11 report: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
