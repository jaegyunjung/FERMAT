#!/usr/bin/env python3
"""Check the last Task 17 genomic audit run from a Pod terminal.

This script is meant for the hospital-network terminal where the SNUH CDM and
block-storage outputs are reachable. It reports:

- local Python processes that look like Task 17 audit/profile jobs;
- active PostgreSQL sessions for the Task 17 application names;
- the tail of the Task 17 log file;
- the latest summary.json and output file timestamps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import psycopg
except ModuleNotFoundError:
    psycopg = None


DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
DEFAULT_BASE_DIR = Path("/home/khdp-user/workspace/fermat-data")
TASK17_APP_NAMES = [
    "fermat_genomic_variant_feasibility_audit_v2",
    "fermat_task17_source_value_profile",
    "fermat_task17_l25_code_probe",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--tail-lines", type=int, default=80)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def fmt_mtime(path: Path):
    try:
        return dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return None


def rel_or_str(path: Path, root: Path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_ps():
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,stat,command"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {"error": repr(error), "matches": []}

    matches = []
    needles = [
        "audit_snuh_genomic_variant_feasibility.py",
        "task17_profile_source_value_columns.py",
        "probe_snuh_task17_l25_codes.py",
        *TASK17_APP_NAMES,
    ]
    for line in proc.stdout.splitlines():
        if "check_snuh_task17_status.py" in line:
            continue
        if any(needle in line for needle in needles):
            matches.append(line.strip())
    return {"returncode": proc.returncode, "matches": matches}


def connect(args):
    if psycopg is None:
        raise RuntimeError("psycopg is not installed in this environment")
    password = os.environ.get("SNUH_CDM_PASSWORD")
    if password is None:
        password = getpass.getpass("SNUH CDM password: ")
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password,
        sslmode=args.sslmode,
        application_name="fermat_task17_status_check",
    )


def check_db_sessions(args):
    query = """
        SELECT pid, leader_pid, backend_type, client_addr, client_port,
               state, wait_event_type, wait_event,
               now() - query_start AS query_age,
               now() - xact_start AS xact_age,
               application_name,
               left(regexp_replace(query, '\\s+', ' ', 'g'), 240) AS query
        FROM pg_stat_activity
        WHERE application_name = ANY(%s)
        ORDER BY COALESCE(leader_pid, pid), pid
    """
    with connect(args) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '15s'")
            cur.execute(query, (TASK17_APP_NAMES,))
            columns = [desc.name for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def tail_file(path: Path, lines: int):
    if not path.exists():
        return {"exists": False, "path": str(path), "lines": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return {"exists": True, "path": str(path), "error": repr(error), "lines": []}
    return {
        "exists": True,
        "path": str(path),
        "mtime": fmt_mtime(path),
        "size_bytes": path.stat().st_size,
        "lines": text.splitlines()[-lines:],
    }


def load_summary(path: Path):
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "exists": True,
            "path": str(path),
            "mtime": fmt_mtime(path),
            "error": repr(error),
        }
    return {
        "exists": True,
        "path": str(path),
        "mtime": fmt_mtime(path),
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(data.keys()),
        "preview": data,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "application_name": data.get("application_name"),
        "outputs": data.get("outputs", {}),
        "measurement_presence": data.get("measurement_presence"),
        "task15_etl_frequency_summary": data.get("task15_etl_frequency_summary"),
        "source_value_audit_summary": data.get("source_value_audit_summary"),
    }


def list_outputs(output_dir: Path, root: Path):
    if not output_dir.exists():
        return []
    rows = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": rel_or_str(path, root),
                    "size_bytes": path.stat().st_size,
                    "mtime": fmt_mtime(path),
                }
            )
    return rows


def run_root_from_output_dir(output_dir: Path):
    parts = output_dir.parts
    if len(parts) >= 2 and parts[-2] == "outputs":
        return output_dir.parents[1]
    return output_dir.parent


def discover_runs(base_dir: Path):
    roots = []
    for root in [Path.cwd(), base_dir]:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in roots and resolved.exists():
            roots.append(resolved)

    summaries = []
    logs = []
    for root in roots:
        for path in root.glob("**/outputs/task17*/summary.json"):
            if path.is_file():
                output_dir = path.parent
                summaries.append(
                    {
                        "summary_path": path,
                        "output_dir": output_dir,
                        "run_root": run_root_from_output_dir(output_dir),
                        "mtime_epoch": path.stat().st_mtime,
                        "mtime": fmt_mtime(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        for path in root.glob("**/logs/task17_run.log"):
            if path.is_file():
                logs.append(
                    {
                        "log_file": path,
                        "run_root": path.parents[1],
                        "mtime_epoch": path.stat().st_mtime,
                        "mtime": fmt_mtime(path),
                        "size_bytes": path.stat().st_size,
                    }
                )

    summaries.sort(key=lambda row: row["mtime_epoch"], reverse=True)
    logs.sort(key=lambda row: row["mtime_epoch"], reverse=True)
    return summaries, logs


def compact_discovered(summaries, logs, root: Path):
    return {
        "summaries": [
            {
                "summary_path": rel_or_str(row["summary_path"], root),
                "output_dir": rel_or_str(row["output_dir"], root),
                "run_root": rel_or_str(row["run_root"], root),
                "mtime": row["mtime"],
                "size_bytes": row["size_bytes"],
            }
            for row in summaries[:12]
        ],
        "logs": [
            {
                "log_file": rel_or_str(row["log_file"], root),
                "run_root": rel_or_str(row["run_root"], root),
                "mtime": row["mtime"],
                "size_bytes": row["size_bytes"],
            }
            for row in logs[:12]
        ],
    }


def print_section(title):
    print(f"\n## {title}")


def print_human(report):
    print(f"Task 17 status check at {report['checked_at']}")
    print(f"base_dir: {report['base_dir']}")
    print(f"output_dir: {report['output_dir']}")
    print(f"log_file: {report['log_file']}")

    print_section("Discovered Task 17 runs")
    discovered = report.get("discovered_runs") or {}
    summaries = discovered.get("summaries") or []
    logs = discovered.get("logs") or []
    if summaries:
        print("summaries:")
        for row in summaries:
            print(f"  {row['mtime']} {row['size_bytes']:>10} {row['summary_path']}")
    else:
        print("no summary.json candidates found")
    if logs:
        print("logs:")
        for row in logs:
            print(f"  {row['mtime']} {row['size_bytes']:>10} {row['log_file']}")
    else:
        print("no task17_run.log candidates found")

    print_section("Local process matches")
    matches = report["local_processes"].get("matches") or []
    if matches:
        for line in matches:
            print(line)
    else:
        print("no local Task 17 Python process found")

    print_section("Database sessions")
    if report.get("db_error"):
        print(f"DB check failed: {report['db_error']}")
    else:
        sessions = report.get("db_sessions") or []
        if not sessions:
            print("no active Task 17 PostgreSQL session found")
        for row in sessions:
            print(
                "pid={pid} leader={leader_pid} state={state} wait={wait_event_type}/{wait_event} "
                "query_age={query_age} app={application_name}".format(**row)
            )
            print(f"  query: {row.get('query')}")

    print_section("Summary")
    summary = report["summary"]
    if not summary.get("exists"):
        print(f"missing: {summary['path']}")
    elif summary.get("error"):
        print(f"could not read {summary['path']}: {summary['error']}")
    else:
        print(f"path: {summary['path']}")
        print(f"mtime: {summary.get('mtime')}")
        print(f"started_at: {summary.get('started_at')}")
        print(f"finished_at: {summary.get('finished_at')}")
        print(f"application_name: {summary.get('application_name')}")
        print(f"top_level_keys: {summary.get('top_level_keys')}")
        if summary.get("measurement_presence") is not None:
            print(f"measurement_presence: {json.dumps(summary['measurement_presence'], ensure_ascii=False)}")
        if summary.get("task15_etl_frequency_summary") is not None:
            print(
                "task15_etl_frequency_summary: "
                f"{json.dumps(summary['task15_etl_frequency_summary'], ensure_ascii=False)}"
            )
        if summary.get("source_value_audit_summary") is not None:
            print(
                "source_value_audit_summary: "
                f"{json.dumps(summary['source_value_audit_summary'], ensure_ascii=False)}"
            )
        if not any(
            summary.get(key) is not None
            for key in [
                "measurement_presence",
                "task15_etl_frequency_summary",
                "source_value_audit_summary",
            ]
        ):
            print(
                "summary_preview: "
                f"{json.dumps(summary.get('preview'), ensure_ascii=False, default=str)[:4000]}"
            )

    print_section("Output files")
    outputs = report["output_files"]
    if outputs:
        for row in outputs:
            print(f"{row['mtime']} {row['size_bytes']:>10} {row['path']}")
    else:
        print("no output files found")

    print_section("Log tail")
    log_tail = report["log_tail"]
    if not log_tail.get("exists"):
        print(f"missing: {log_tail['path']}")
    elif log_tail.get("error"):
        print(f"could not read {log_tail['path']}: {log_tail['error']}")
    else:
        print(f"path: {log_tail['path']}")
        print(f"mtime: {log_tail.get('mtime')} size={log_tail.get('size_bytes')} bytes")
        for line in log_tail["lines"]:
            print(line)


def main():
    args = parse_args()
    output_dir = args.output_dir or Path("outputs/task17_genomic_variant_audit")
    log_file = args.log_file or Path("logs/task17_run.log")
    summaries, logs = discover_runs(args.base_dir)
    if args.output_dir is None and not (output_dir / "summary.json").exists() and summaries:
        output_dir = summaries[0]["output_dir"]
    if args.log_file is None and not log_file.exists():
        run_root = run_root_from_output_dir(output_dir)
        paired_log = run_root / "logs" / "task17_run.log"
        if paired_log.exists():
            log_file = paired_log
        elif logs:
            log_file = logs[0]["log_file"]
    report = {
        "checked_at": now_iso(),
        "base_dir": str(args.base_dir),
        "output_dir": str(output_dir),
        "log_file": str(log_file),
        "application_names": TASK17_APP_NAMES,
        "discovered_runs": compact_discovered(summaries, logs, Path.cwd()),
        "local_processes": run_ps(),
        "summary": load_summary(output_dir / "summary.json"),
        "output_files": list_outputs(output_dir, Path.cwd()),
        "log_tail": tail_file(log_file, args.tail_lines),
    }
    if args.no_db:
        report["db_skipped"] = True
    else:
        try:
            report["db_sessions"] = check_db_sessions(args)
        except Exception as error:
            report["db_error"] = repr(error)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
