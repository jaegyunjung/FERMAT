#!/usr/bin/env python3
"""Run the validated SNUH tokenization notebook as a full-cohort ETL job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "02_snuh_tokenization_etl.ipynb"
POD_STORAGE = Path("/home/khdp-user/workspace/fermat-data")
POD_ETL_ROOT = POD_STORAGE / "etl"
REQUIRED_PACKAGES = ("psycopg", "numpy", "pandas", "pyarrow", "psutil", "IPython")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-buckets", type=int, default=100)
    parser.add_argument("--lab-bucket-width", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=POD_ETL_ROOT)
    parser.add_argument("--minimum-free-gb", type=float, default=140.0)
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="Run even when free block-storage space is below the safety gate.",
    )
    return parser.parse_args()


def ensure_dependencies():
    missing = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return
    install = []
    for package in missing:
        install.append("psycopg[binary]>=3" if package == "psycopg" else package)
    print("Installing missing packages:", ", ".join(install), flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *install],
        cwd=ROOT,
        check=True,
    )


def execute_notebook_cells(notebook_path):
    from IPython.display import display

    document = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "display": display,
    }
    code_cells = [
        cell for cell in document["cells"] if cell.get("cell_type") == "code"
    ]
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(
            f"\n[TASK 15] Starting notebook code cell {index}/{len(code_cells)}",
            flush=True,
        )
        started = time.time()
        exec(
            compile(source, f"{notebook_path.name}:cell-{index}", "exec"),
            namespace,
        )
        print(
            f"[TASK 15] Completed cell {index} in "
            f"{time.time() - started:,.1f}s",
            flush=True,
        )


def main():
    args = parse_args()
    if not 1 <= args.patient_buckets <= 100:
        raise ValueError("patient-buckets must be between 1 and 100")
    if not 1 <= args.lab_bucket_width <= args.patient_buckets:
        raise ValueError(
            "lab-bucket-width must be between 1 and patient-buckets"
        )
    ensure_dependencies()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = f"patient_{args.patient_buckets:03d}pct_seed_42"
    result_dir = output_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (result_dir / "task15.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    free_gb = shutil.disk_usage(output_root).free / 1024**3
    print(f"ETL output root: {output_root}")
    print(f"Patient cohort: {args.patient_buckets}%")
    print(f"Numeric LAB bucket width: {args.lab_bucket_width}%")
    print(f"Free storage: {free_gb:.1f} GB")
    if free_gb < args.minimum_free_gb and not args.allow_low_disk:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB is free; Task 15 requires at least "
            f"{args.minimum_free_gb:.1f} GB by default. Free storage or pass "
            "--allow-low-disk after reviewing the risk."
        )

    os.environ["FERMAT_PATIENT_BUCKETS"] = str(args.patient_buckets)
    os.environ["FERMAT_ETL_OUTPUT_ROOT"] = str(output_root)
    os.environ["FERMAT_LAB_BUCKET_WIDTH"] = str(args.lab_bucket_width)
    started = time.time()
    try:
        execute_notebook_cells(NOTEBOOK)
    except Exception:
        print("\n[TASK 15] FAILED", flush=True)
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("SNUH_CDM_PASSWORD", None)

    print(f"\nTask 15 completed in {(time.time() - started) / 3600:.2f} hours")
    print(f"ETL artifacts: {result_dir}")
    print(f"Manifest: {result_dir / 'manifest.json'}")
    print(f"Checksums: {result_dir / 'sha256.json'}")


if __name__ == "__main__":
    main()
