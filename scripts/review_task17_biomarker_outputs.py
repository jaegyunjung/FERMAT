#!/usr/bin/env python3
"""Review Task 17 molecular/biomarker output CSVs.

Run this on the Pod in the directory that contains the Task 17 audit outputs,
or pass --output-dir explicitly. The script does not query the database; it
only summarizes existing CSV artifacts so the review is cheap and repeatable.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(
    "/home/khdp-user/workspace/fermat-data/scripts/outputs/task17_molecular_biomarker_audit"
)

OBS_EVENTS = "post2021_observation_onco_biomarker_events_v2.csv"
OBS_FIRST_EVENTS = "post2021_observation_onco_biomarker_first_events_v2.csv"
OBS_SUMMARY = "post2021_observation_onco_biomarker_event_summary_v2.csv"
OBS_REVIEW = "post2021_observation_onco_biomarker_manual_review_v2.csv"
NOTE_SAMPLES = "note_molecular_keyword_sample_rows.csv"
NGS_RELAXED = "post2021_ngs_relaxed_hits.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-per-token", type=int, default=3)
    parser.add_argument("--sample-notes", type=int, default=40)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path):
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, reader.fieldnames or []


def year_of(value: str):
    if not value or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def compact(text: str, limit: int = 240):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def classify_note(row):
    title = (row.get("note_title") or "").upper()
    source = (row.get("note_source_value") or "").upper()
    source2 = (row.get("note_source_value2") or "").upper()
    text = " ".join(
        [
            row.get("note_text_prefix") or "",
            row.get("sample_header") or "",
            row.get("text_prefix") or "",
        ]
    ).upper()

    joined = " ".join([title, source, source2, text])
    if any(term in joined for term in ["FIRST CANCER PANEL", "NEXT GENERATION SEQUENCING", "SNV/INDEL", "COPY NUMBER ALTERATION"]):
        return "likely_ngs_report"
    if any(term in joined for term in ["EGFR GENE MUTATION", "ALK TRANSLOCATION", "MICROSATELLITE INSTABILITY", "MSI"]):
        return "likely_molecular_report"
    if "WHOLEBODY SCAN" in joined or "I-131" in joined or "I-123" in joined:
        return "imaging_i131_false_positive"
    if source.startswith("KL") or source.startswith("CTCKL") or any(term in joined for term in ["QRS", "QTC", "NORMAL ECG"]):
        return "ecg_false_positive"
    if source.startswith("R") and ("BREAST" in joined or source in {"R5364", "R5519", "R5515", "R5277", "R5133"}):
        return "breast_imaging_false_positive"
    if any(term in joined for term in ["BODY MASS", "SKELETAL MUSCLE", "BMI", "WEIGHT(KG)", "체지방"]):
        return "body_composition_false_positive"
    return "needs_review"


def row_count_summary(rows, date_col, person_col="person_id"):
    persons = {row.get(person_col) for row in rows if row.get(person_col)}
    years = Counter()
    for row in rows:
        y = year_of(row.get(date_col, ""))
        if y is not None:
            years[y] += 1
    return {
        "rows": len(rows),
        "persons": len(persons),
        "years": dict(sorted(years.items())),
        "post2021_rows": sum(count for year, count in years.items() if year >= 2021),
    }


def write(line="", out=None):
    if out is None:
        print(line)
    else:
        out.append(line)


def section(title, out):
    write("", out)
    write(f"## {title}", out)


def summarize_observation_events(output_dir: Path, out, sample_per_token: int):
    rows, fields = read_csv(output_dir / OBS_EVENTS)
    section(OBS_EVENTS, out)
    if not rows:
        write("missing or empty", out)
        return

    summary = row_count_summary(rows, "event_date")
    write(f"rows: {summary['rows']}", out)
    write(f"persons: {summary['persons']}", out)
    write(f"years: {summary['years']}", out)
    write(f"post2021_rows: {summary['post2021_rows']}", out)
    write(f"columns: {fields}", out)

    for label, key in [
        ("token_id", "token_id"),
        ("biomarker_token", "biomarker_token"),
        ("biomarker_value", "biomarker_value"),
        ("observation_source_value", "observation_source_value"),
    ]:
        counter = Counter(row.get(key, "") for row in rows)
        section(f"Top {label}", out)
        for value, count in counter.most_common(40):
            write(f"{count:>5}  {value}", out)

    samples_by_token = defaultdict(list)
    for row in rows:
        token = row.get("token_id") or row.get("biomarker_token") or "<blank>"
        if len(samples_by_token[token]) < sample_per_token:
            samples_by_token[token].append(row)

    section("Evidence samples by token", out)
    for token in sorted(samples_by_token):
        write(f"\n### {token}", out)
        for row in samples_by_token[token]:
            write(
                " | ".join(
                    [
                        f"person={row.get('person_id')}",
                        f"date={row.get('event_date')}",
                        f"source_id={row.get('source_id')}",
                        f"value={row.get('biomarker_value')}",
                        f"source={row.get('observation_source_value')}",
                        f"evidence={compact(row.get('evidence') or row.get('value_source_value') or row.get('ext_etc_source_value'))}",
                    ]
                ),
                out,
            )


def summarize_first_events(output_dir: Path, out):
    rows, fields = read_csv(output_dir / OBS_FIRST_EVENTS)
    section(OBS_FIRST_EVENTS, out)
    if not rows:
        write("missing or empty", out)
        return
    summary = row_count_summary(rows, "event_date")
    write(f"rows: {summary['rows']}", out)
    write(f"persons: {summary['persons']}", out)
    write(f"years: {summary['years']}", out)
    write(f"columns: {fields}", out)


def summarize_manual_review(output_dir: Path, out):
    rows, fields = read_csv(output_dir / OBS_REVIEW)
    section(OBS_REVIEW, out)
    if not rows:
        write("missing or empty", out)
        return
    write(f"rows: {len(rows)}", out)
    write(f"columns: {fields}", out)
    reasons = Counter(row.get("reason", "") for row in rows)
    for reason, count in reasons.most_common():
        write(f"{count:>5}  {reason}", out)
    write("", out)
    for row in rows[:30]:
        write(
            " | ".join(
                [
                    f"reason={row.get('reason')}",
                    f"person={row.get('person_id')}",
                    f"date={row.get('observation_date')}",
                    f"observation_id={row.get('observation_id')}",
                    compact(row.get("text"), 260),
                ]
            ),
            out,
        )


def summarize_summary_file(output_dir: Path, out):
    rows, fields = read_csv(output_dir / OBS_SUMMARY)
    section(OBS_SUMMARY, out)
    if not rows:
        write("missing or empty", out)
        return
    write(f"rows: {len(rows)}", out)
    write(f"columns: {fields}", out)
    for row in rows[:80]:
        write(
            " | ".join(
                [
                    row.get("token_id", ""),
                    f"events={row.get('n_events')}",
                    f"persons={row.get('n_persons')}",
                    f"{row.get('min_date')}..{row.get('max_date')}",
                ]
            ),
            out,
        )


def summarize_note_candidates(output_dir: Path, out, sample_notes: int):
    for filename in [NGS_RELAXED, NOTE_SAMPLES]:
        rows, fields = read_csv(output_dir / filename)
        section(filename, out)
        if not rows:
            write("missing or empty", out)
            continue
        write(f"rows: {len(rows)}", out)
        write(f"columns: {fields}", out)

        classes = Counter(classify_note(row) for row in rows)
        write("classification counts:", out)
        for label, count in classes.most_common():
            write(f"{count:>5}  {label}", out)

        source_counts = Counter(
            (
                row.get("note_title", ""),
                row.get("note_source_value", ""),
                row.get("note_source_value2", ""),
            )
            for row in rows
        )
        section(f"{filename} Top title/source/source2", out)
        for key, count in source_counts.most_common(40):
            write(f"{count:>5}  {key}", out)

        section(f"{filename} Samples", out)
        for row in rows[:sample_notes]:
            write(
                " | ".join(
                    [
                        classify_note(row),
                        f"note={row.get('note_id')}",
                        f"person={row.get('person_id')}",
                        f"date={row.get('note_date')}",
                        f"title={row.get('note_title')}",
                        f"source={row.get('note_source_value')}",
                        f"source2={row.get('note_source_value2')}",
                        compact(row.get("note_text_prefix") or row.get("sample_header") or row.get("text_prefix"), 300),
                    ]
                ),
                out,
            )


def main():
    args = parse_args()
    output = []
    write(f"# Task 17 Biomarker Output Review", output)
    write(f"generated_at: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}", output)
    write(f"output_dir: {args.output_dir}", output)

    summarize_observation_events(args.output_dir, output, args.sample_per_token)
    summarize_first_events(args.output_dir, output)
    summarize_summary_file(args.output_dir, output)
    summarize_manual_review(args.output_dir, output)
    summarize_note_candidates(args.output_dir, output, args.sample_notes)

    text = "\n".join(output) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
    print(text)


if __name__ == "__main__":
    main()
