#!/usr/bin/env python3
"""Audit a FERMAT Pod for Task 24/27 execution evidence and outputs.

This script is intentionally read-only except for its own timestamped audit
report. It distinguishes finished outputs from code/bundle preparation and
from execution traces such as logs or shell history.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TEXT_SUFFIXES = {".log", ".out", ".txt", ".md", ".json", ".jsonl", ".sh", ".py"}
CONTENT_PATTERN = re.compile(
    r"task\s*24|task24_sensitivity|first.record|utilization sensitivity|"
    r"task\s*27|primary_direct_risk|direct.risk|task27_dry_run|np\.trapz",
    re.IGNORECASE,
)
FAILURE_PATTERN = re.compile(
    r"traceback|error|exception|failed|killed|out of memory|oom|no space|np\.trapz",
    re.IGNORECASE,
)
NAME_PATTERN = re.compile(
    r"task\s*24|task24|sensitivity|task\s*27|task27|direct.?risk",
    re.IGNORECASE,
)

TASK_SPECS = {
    "task24": {
        "strong": {"task24_sensitivity_results.csv", "task24_sensitivity_summary.csv"},
        "runner": "run_snuh_task24_sensitivity.py",
    },
    "task27": {
        "strong": {"task27_dry_run_predictions.csv", "task27_dry_run_summary.csv"},
        "runner": "run_snuh_task27_primary_direct_risk_dry_run.py",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: ROOT/audits/task24_task27_pod_audit_TIMESTAMP.txt",
    )
    parser.add_argument("--max-content-mb", type=int, default=32)
    parser.add_argument("--max-content-hits", type=int, default=300)
    return parser.parse_args()


class Reporter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def emit(self, message=""):
        message = str(message)
        print(message, flush=True)
        self.handle.write(message + "\n")

    def close(self):
        self.handle.close()


def human_bytes(value: int):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024


def file_info(path: Path):
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return f"{human_bytes(stat.st_size):>10}  {modified}  {path}"


def walk_files(root: Path):
    if not root.exists():
        return
    if root.is_file():
        yield root
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
        base = Path(current)
        for name in files:
            yield base / name


def count_csv_rows(path: Path):
    with path.open("rb") as handle:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""))
    return max(0, lines - 1)


def inspect_csv(path: Path, report: Reporter):
    report.emit(f"\n[CSV] {path}")
    report.emit(f"rows={count_csv_rows(path):,}  {file_info(path)}")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        report.emit("columns=" + ", ".join(columns))
        group_columns = [
            column
            for column in ["comparison", "metric", "stratum_type", "horizon", "phenotype", "model"]
            if column in columns
        ]
        groups = defaultdict(lambda: {"rows": 0, "positives": 0, "score_min": None, "score_max": None})
        preview = []
        for index, row in enumerate(reader):
            if index < 5:
                preview.append(row)
            if group_columns:
                key = tuple(row.get(column, "") for column in group_columns)
                item = groups[key]
                item["rows"] += 1
                try:
                    item["positives"] += int(float(row.get("label", row.get("y_true", 0)) or 0))
                except ValueError:
                    pass
                score_text = row.get("score", row.get("y_score", ""))
                try:
                    score = float(score_text)
                except (TypeError, ValueError):
                    score = None
                if score is not None:
                    item["score_min"] = score if item["score_min"] is None else min(item["score_min"], score)
                    item["score_max"] = score if item["score_max"] is None else max(item["score_max"], score)
        if groups:
            report.emit(f"groups_by={group_columns} unique_groups={len(groups):,}")
            for key, item in list(sorted(groups.items()))[:80]:
                report.emit(
                    f"  {dict(zip(group_columns, key))} rows={item['rows']:,} "
                    f"positives={item['positives']:,} score_range={item['score_min']}..{item['score_max']}"
                )
        report.emit("preview=" + json.dumps(preview, ensure_ascii=False, default=str))


def inspect_json(path: Path, report: Reporter):
    report.emit(f"\n[JSON] {file_info(path)}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception as error:
        report.emit(f"json_parse_error={error}")
        return
    if isinstance(data, dict):
        report.emit("keys=" + ", ".join(map(str, data.keys())))
    rendered = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    report.emit(rendered[:20000])
    if len(rendered) > 20000:
        report.emit(f"... truncated {len(rendered) - 20000:,} characters")


def inspect_parquet(path: Path, report: Reporter):
    report.emit(f"\n[PARQUET] {file_info(path)}")
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        report.emit(f"rows={metadata.num_rows:,} row_groups={metadata.num_row_groups:,}")
        report.emit("schema=" + str(metadata.schema).replace("\n", " | "))
    except Exception as error:
        report.emit(f"metadata_unavailable={error}")


def tail_text(path: Path, max_lines=80):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return lines[-max_lines:]


def task_tree(task_root: Path, report: Reporter):
    report.emit(f"\n## TREE {task_root}")
    if not task_root.exists():
        report.emit("NOT_FOUND")
        return []
    files = sorted(walk_files(task_root), key=lambda path: path.stat().st_mtime)
    if not files:
        report.emit("DIRECTORY_EXISTS_BUT_EMPTY")
        return []
    for path in files:
        try:
            report.emit(file_info(path))
        except OSError as error:
            report.emit(f"STAT_ERROR {path}: {error}")
    return files


def inspect_task_outputs(task: str, files: list[Path], report: Reporter):
    report.emit(f"\n## STRUCTURED OUTPUT INSPECTION {task}")
    candidates = []
    for path in files:
        lower_parts = {part.lower() for part in path.parts}
        if "outputs" in lower_parts or path.name in TASK_SPECS[task]["strong"]:
            candidates.append(path)
    if not candidates:
        report.emit("NO_OUTPUT_FILES_FOUND")
        return
    for path in candidates:
        try:
            if path.suffix.lower() == ".csv":
                inspect_csv(path, report)
            elif path.suffix.lower() == ".json":
                inspect_json(path, report)
            elif path.suffix.lower() in {".parquet", ".pq"}:
                inspect_parquet(path, report)
            elif path.suffix.lower() in {".log", ".out", ".txt"}:
                report.emit(f"\n[TEXT TAIL] {file_info(path)}")
                for line in tail_text(path):
                    report.emit("  " + line.rstrip())
        except Exception as error:
            report.emit(f"INSPECT_ERROR {path}: {error}")


def search_global(root: Path, max_bytes: int, max_hits: int, report: Reporter):
    report.emit("\n## GLOBAL NAME AND CONTENT SEARCH")
    all_files = list(walk_files(root))
    name_hits = []
    content_hits = []
    for path in all_files:
        relative = str(path.relative_to(root))
        if NAME_PATTERN.search(relative):
            name_hits.append(path)
    content_candidates = []
    for path in all_files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or size > max_bytes:
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".out", ".txt"} or "nohup" in path.name.lower():
            priority = 0
        elif suffix in {".json", ".jsonl", ".md"}:
            priority = 1
        else:
            priority = 2
        content_candidates.append((priority, path))
    for _, path in sorted(content_candidates, key=lambda item: (item[0], str(item[1]))):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if CONTENT_PATTERN.search(line):
                        content_hits.append((path, line_number, line.strip()[:1000]))
                        if len(content_hits) >= max_hits:
                            break
        except OSError:
            pass
        if len(content_hits) >= max_hits:
            break
    report.emit(f"files_scanned={len(all_files):,}")
    report.emit(f"name_hits={len(name_hits):,}")
    for path in sorted(set(name_hits)):
        try:
            report.emit("  " + file_info(path))
        except OSError:
            report.emit(f"  {path}")
    report.emit(f"content_hits={len(content_hits):,} max={max_hits:,}")
    for path, line_number, line in content_hits:
        report.emit(f"  {path}:{line_number}: {line}")
    if len(content_hits) >= max_hits:
        report.emit("CONTENT_HIT_LIMIT_REACHED; narrow the root or raise --max-content-hits if needed")
    return name_hits, content_hits


def inspect_histories(report: Reporter):
    report.emit("\n## SHELL HISTORY SEARCH")
    candidates = [
        Path.home() / ".bash_history",
        Path.home() / ".zsh_history",
        Path("/root/.bash_history"),
        Path("/root/.zsh_history"),
    ]
    found = False
    hits = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        found = True
        report.emit(f"[{path}]")
        try:
            for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if CONTENT_PATTERN.search(line):
                    report.emit(f"  {line_number}: {line[:1000]}")
                    hits.append((path, line_number, line[:1000]))
        except OSError as error:
            report.emit(f"  READ_ERROR: {error}")
    if not found:
        report.emit("NO_HISTORY_FILE_FOUND (this does not prove that a task was not run)")
    return hits


def classify(task: str, files: list[Path], name_hits, content_hits, report: Reporter):
    spec = TASK_SPECS[task]
    output_files = [path for path in files if "outputs" in {part.lower() for part in path.parts}]
    strong = {path.name for path in output_files if path.stat().st_size > 0} & spec["strong"]
    log_files = [path for path in files if path.suffix.lower() in {".log", ".out", ".txt"}]
    trace_hits = [
        hit
        for hit in content_hits
        if (task in str(hit[0]).lower() or task in hit[2].lower())
        and (
            hit[0].suffix.lower() in {".log", ".out", ".txt"}
            or "history" in hit[0].name.lower()
            or "nohup" in hit[0].name.lower()
        )
    ]
    prepared_paths = files + [path for path in name_hits if task in str(path).lower()]
    prepared = any(path.name == spec["runner"] or path.suffix.lower() == ".zip" for path in prepared_paths)
    failures = []
    for path in log_files:
        try:
            for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if FAILURE_PATTERN.search(line):
                    failures.append((path, line_number, line[:1000]))
        except OSError:
            pass
    if strong == spec["strong"]:
        status = "COMPLETED_OUTPUTS_FOUND"
    elif strong or output_files:
        status = "PARTIAL_OUTPUTS_FOUND"
    elif trace_hits or log_files:
        status = "EXECUTION_TRACES_ONLY"
    elif prepared:
        status = "PREPARED_ONLY"
    else:
        status = "NOT_FOUND"
    report.emit(f"\n## VERDICT {task}")
    report.emit(f"status={status}")
    report.emit(f"expected_nonempty_outputs_found={sorted(strong)}")
    report.emit(f"all_output_files={len(output_files):,} log_files={len(log_files):,} trace_hits={len(trace_hits):,}")
    if failures:
        report.emit(f"failure_lines={len(failures):,}")
        for path, line_number, line in failures[-40:]:
            report.emit(f"  {path}:{line_number}: {line}")
    else:
        report.emit("failure_lines=0")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    requested_report = args.report or root / "audits" / f"task24_task27_pod_audit_{timestamp}.txt"
    try:
        report = Reporter(requested_report.expanduser().resolve())
    except OSError:
        fallback = Path("/tmp") / f"task24_task27_pod_audit_{timestamp}.txt"
        report = Reporter(fallback)
    try:
        report.emit("# SNUH TASK24/TASK27 POD AUDIT")
        report.emit(f"started_at={datetime.now().astimezone().isoformat(timespec='seconds')}")
        report.emit(f"root={root}")
        report.emit(f"report={report.path}")
        report.emit(f"python={sys.version.split()[0]}")
        if not root.exists():
            report.emit("ROOT_NOT_FOUND")
            return 2

        task_files = {}
        for task in TASK_SPECS:
            task_files[task] = task_tree(root / task, report)
            inspect_task_outputs(task, task_files[task], report)

        name_hits, content_hits = search_global(
            root,
            max_bytes=args.max_content_mb * 1024 * 1024,
            max_hits=args.max_content_hits,
            report=report,
        )
        content_hits.extend(inspect_histories(report))
        for task in TASK_SPECS:
            classify(task, task_files[task], name_hits, content_hits, report)

        report.emit(f"\nfinished_at={datetime.now().astimezone().isoformat(timespec='seconds')}")
        report.emit(f"AUDIT_REPORT={report.path}")
        return 0
    finally:
        report.close()


if __name__ == "__main__":
    raise SystemExit(main())
