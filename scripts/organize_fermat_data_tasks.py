#!/usr/bin/env python3
"""Organize Pod-side fermat-data task files into task directories.

Default mode is dry-run. Add --apply only after reviewing the planned moves.

Example:
    python scripts/organize_fermat_data_tasks.py \
      --root /home/khdp-user/workspace/fermat-data

    python scripts/organize_fermat_data_tasks.py \
      --root /home/khdp-user/workspace/fermat-data \
      --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


TASK_PATTERN = re.compile(r"(?:^|_)task(?P<num>\d+)(?:[_\-.]|$)", re.IGNORECASE)
TASK_DIR_PATTERN = re.compile(r"^task(?P<num>\d+)(?:[-_].*)?$", re.IGNORECASE)
CODE_DIR_SUFFIX = "-code"

KEEP_AT_ROOT = {
    "etl",
    "lost+found",
    "notebooks",
    "out",
    "outputs",
    "reference",
    "scripts",
}


@dataclass(frozen=True)
class Move:
    src: Path
    dst: Path
    reason: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/khdp-user/workspace/fermat-data"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Omit for dry-run.",
    )
    parser.add_argument(
        "--include-existing-task-dirs",
        action="store_true",
        help=(
            "Also move existing task-named directories such as task10-ce-only "
            "into task10/legacy_dirs. Plain taskNN directories are left in place."
        ),
    )
    return parser.parse_args()


def task_number(name: str) -> str | None:
    match = TASK_PATTERN.search(name)
    if match:
        return match.group("num").lstrip("0") or "0"
    match = TASK_DIR_PATTERN.match(name)
    if match:
        return match.group("num").lstrip("0") or "0"
    return None


def destination_subdir(path: Path) -> str:
    name = path.name
    if path.is_dir() and name.endswith(CODE_DIR_SUFFIX):
        return "code"
    if path.suffix == ".zip":
        return "zips"
    if path.suffix in {".log", ".out", ".err"}:
        return "logs"
    if path.suffix in {".md", ".txt"}:
        return "docs"
    if path.suffix in {".py", ".ipynb", ".sh"}:
        return "scripts"
    if path.is_dir():
        return "legacy_dirs"
    return "misc"


def task_root(root: Path, number: str) -> Path:
    return root / f"task{number}"


def is_plain_task_dir(path: Path) -> bool:
    match = re.fullmatch(r"task\d+", path.name, flags=re.IGNORECASE)
    return bool(match and path.is_dir())


def plan_moves(root: Path, include_existing_task_dirs: bool) -> list[Move]:
    moves: list[Move] = []
    for src in sorted(root.iterdir(), key=lambda p: p.name):
        if src.name in KEEP_AT_ROOT:
            continue
        if is_plain_task_dir(src):
            continue
        number = task_number(src.name)
        if number is None:
            continue
        if src.is_dir() and not include_existing_task_dirs:
            if TASK_DIR_PATTERN.match(src.name) and not src.name.endswith(CODE_DIR_SUFFIX):
                continue
        dst = task_root(root, number) / destination_subdir(src) / src.name
        if src.resolve() == dst.resolve():
            continue
        moves.append(Move(src=src, dst=dst, reason=f"task{number} artifact"))
    return moves


def ensure_no_collisions(moves: list[Move]):
    destinations = {}
    for move in moves:
        key = str(move.dst)
        destinations.setdefault(key, []).append(str(move.src))
        if move.dst.exists():
            raise FileExistsError(f"Destination already exists: {move.dst}")
    collisions = {
        dst: srcs for dst, srcs in destinations.items() if len(srcs) > 1
    }
    if collisions:
        raise RuntimeError(
            "Multiple sources map to the same destination:\n"
            + json.dumps(collisions, indent=2, ensure_ascii=False)
        )


def apply_moves(moves: list[Move]):
    ensure_no_collisions(moves)
    for move in moves:
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.src), str(move.dst))


def print_plan(moves: list[Move], apply: bool):
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"# fermat-data task organization plan ({mode})")
    print(f"planned_moves={len(moves)}")
    for move in moves:
        print(f"{move.src} -> {move.dst}")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    moves = plan_moves(root, args.include_existing_task_dirs)
    print_plan(moves, args.apply)
    if args.apply:
        apply_moves(moves)
        print(f"moved={len(moves)}")
    else:
        print("No files moved. Re-run with --apply after reviewing the plan.")


if __name__ == "__main__":
    main()
