#!/usr/bin/env python3
"""Validate and render the existing Task 30 observed event curves.

This post-processing step uses only Python's standard library.  It does not
query SNUH-CDM, refit a model, or require matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = {
    "treatment_class",
    "day",
    "patients_at_risk_after_day",
    "mi_or_stroke_events_on_day",
    "competing_deaths_on_day",
    "censored_on_day",
    "event_free_survival",
    "mi_or_stroke_cumulative_incidence",
}
LANDMARKS = [(0, "Start"), (365, "1 year"), (1095, "3 years"), (1826, "5 years")]
COLORS = {"ACEI": "#D1495B", "DHP_CCB": "#00798C"}
TOLERANCE = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and render an existing observed MI/stroke curve CSV."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_curve(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        groups = {}
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {
                    "treatment_class": raw["treatment_class"],
                    "day": int(raw["day"]),
                    "patients_at_risk_after_day": int(raw["patients_at_risk_after_day"]),
                    "mi_or_stroke_events_on_day": int(raw["mi_or_stroke_events_on_day"]),
                    "competing_deaths_on_day": int(raw["competing_deaths_on_day"]),
                    "censored_on_day": int(raw["censored_on_day"]),
                    "event_free_survival": float(raw["event_free_survival"]),
                    "mi_or_stroke_cumulative_incidence": float(
                        raw["mi_or_stroke_cumulative_incidence"]
                    ),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value on CSV line {line_number}: {exc}") from exc
            groups.setdefault(row["treatment_class"], []).append(row)
    if not groups:
        raise ValueError("input CSV contains no curve rows")
    for rows in groups.values():
        rows.sort(key=lambda item: item["day"])
    return groups


def validate_group(treatment, rows):
    days = [row["day"] for row in rows]
    cif = [row["mi_or_stroke_cumulative_incidence"] for row in rows]
    survival = [row["event_free_survival"] for row in rows]
    at_risk = [row["patients_at_risk_after_day"] for row in rows]

    duplicate_days = len(days) - len(set(days))
    missing_daily_rows = (days[-1] - days[0] + 1) - len(set(days))
    cif_decreases = sum(b + TOLERANCE < a for a, b in zip(cif, cif[1:]))
    survival_increases = sum(b > a + TOLERANCE for a, b in zip(survival, survival[1:]))
    at_risk_increases = sum(b > a for a, b in zip(at_risk, at_risk[1:]))
    invalid_probabilities = sum(
        not (math.isfinite(value) and -TOLERANCE <= value <= 1.0 + TOLERANCE)
        for value in cif + survival
    )
    negative_counts = sum(
        any(
            row[column] < 0
            for column in (
                "patients_at_risk_after_day",
                "mi_or_stroke_events_on_day",
                "competing_deaths_on_day",
                "censored_on_day",
            )
        )
        for row in rows
    )

    increase_without_event = 0
    event_without_increase = 0
    step_count = 0
    for previous, current in zip(rows, rows[1:]):
        delta = (
            current["mi_or_stroke_cumulative_incidence"]
            - previous["mi_or_stroke_cumulative_incidence"]
        )
        events = current["mi_or_stroke_events_on_day"]
        if delta > TOLERANCE:
            step_count += 1
            if events == 0:
                increase_without_event += 1
        elif events > 0:
            event_without_increase += 1

    landmark_rows = {}
    by_day = {row["day"]: row for row in rows}
    for day, label in LANDMARKS:
        if day in by_day:
            landmark_rows[label] = {
                "day": day,
                "patients_at_risk_after_day": by_day[day]["patients_at_risk_after_day"],
                "cumulative_incidence": by_day[day][
                    "mi_or_stroke_cumulative_incidence"
                ],
            }

    checks = {
        "duplicate_days": duplicate_days,
        "missing_daily_rows": missing_daily_rows,
        "cumulative_incidence_decreases": cif_decreases,
        "event_free_survival_increases": survival_increases,
        "patients_at_risk_increases": at_risk_increases,
        "invalid_probabilities": invalid_probabilities,
        "negative_count_rows": negative_counts,
        "curve_increases_without_event": increase_without_event,
        "event_days_without_curve_increase": event_without_increase,
    }
    passed = all(value == 0 for value in checks.values())
    return {
        "treatment_class": treatment,
        "status": "PASS" if passed else "FAIL",
        "rows": len(rows),
        "first_day": days[0],
        "last_day": days[-1],
        "curve_step_count": step_count,
        "events_within_curve_window": sum(
            row["mi_or_stroke_events_on_day"] for row in rows
        ),
        "final_cumulative_incidence": cif[-1],
        "checks": checks,
        "landmarks": landmark_rows,
    }


def nice_y_max(max_value):
    target = max(0.01, max_value * 1.15)
    step = 0.0025 if target <= 0.025 else 0.005 if target <= 0.05 else 0.01
    return math.ceil(target / step) * step


def svg_text(x, y, value, **attrs):
    rendered = " ".join(
        f'{key.rstrip("_").replace("_", "-")}="{html.escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {rendered}>{html.escape(str(value))}</text>'


def render_svg(groups, path):
    width, height = 1080, 720
    left, right, top, bottom = 100, 45, 85, 195
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_day = max(rows[-1]["day"] for rows in groups.values())
    max_cif = max(
        row["mi_or_stroke_cumulative_incidence"]
        for rows in groups.values()
        for row in rows
    )
    y_max = nice_y_max(max_cif)

    def x(day):
        return left + plot_width * day / max_day

    def y(value):
        return top + plot_height * (1.0 - value / y_max)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{font-size:13px}.small{font-size:12px}.label{font-size:14px;font-weight:600}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#555}</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        svg_text(left, 34, "Observed cumulative incidence of acute MI or stroke", class_="title"),
        svg_text(
            left,
            57,
            "After first-line ACEi or DHP-CCB initiation; all-cause death is a competing event",
            class_="subtitle",
        ),
    ]

    tick_count = 5
    for index in range(tick_count + 1):
        value = y_max * index / tick_count
        yy = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#E3E6E8" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, yy + 4, f"{100 * value:.1f}%", class_="axis", text_anchor="end"))

    visible_landmarks = [(day, label) for day, label in LANDMARKS if day <= max_day]
    for day, label in visible_landmarks:
        xx = x(day)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_height}" stroke="#E3E6E8" stroke-width="1"/>'
        )
        parts.append(svg_text(xx, top + plot_height + 24, label, class_="axis", text_anchor="middle"))

    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#333" stroke-width="1.2"/>',
            f'<text x="25" y="{top + plot_height/2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 25 {top + plot_height/2:.1f})">Cumulative incidence</text>',
        ]
    )

    legend_x = width - right - 250
    for index, (treatment, rows) in enumerate(sorted(groups.items())):
        color = COLORS.get(treatment, ["#6A4C93", "#F4A261", "#2A9D8F"][index % 3])
        commands = [f"M {x(rows[0]['day']):.2f} {y(rows[0]['mi_or_stroke_cumulative_incidence']):.2f}"]
        previous_value = rows[0]["mi_or_stroke_cumulative_incidence"]
        for row in rows[1:]:
            xx = x(row["day"])
            value = row["mi_or_stroke_cumulative_incidence"]
            commands.append(f"H {xx:.2f}")
            if abs(value - previous_value) > TOLERANCE:
                commands.append(f"V {y(value):.2f}")
            previous_value = value
        parts.append(
            f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        legend_y = top + 12 + index * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+30}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(svg_text(legend_x + 38, legend_y + 5, treatment, class_="label"))
        by_day = {row["day"]: row for row in rows}
        for day, _ in visible_landmarks[1:]:
            if day not in by_day:
                continue
            value = by_day[day]["mi_or_stroke_cumulative_incidence"]
            parts.append(
                f'<circle cx="{x(day):.1f}" cy="{y(value):.1f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
            )

    table_top = top + plot_height + 65
    parts.append(svg_text(left, table_top, "Patients at risk after day", class_="label"))
    header_y = table_top + 25
    for day, label in visible_landmarks:
        parts.append(svg_text(x(day), header_y, label, class_="small", text_anchor="middle"))
    for row_index, (treatment, rows) in enumerate(sorted(groups.items())):
        yy = header_y + 25 + row_index * 23
        color = COLORS.get(treatment, "#444")
        parts.append(svg_text(left, yy, treatment, class_="label", fill=color))
        by_day = {row["day"]: row for row in rows}
        for day, _ in visible_landmarks:
            value = by_day.get(day, {}).get("patients_at_risk_after_day", "-")
            parts.append(svg_text(x(day), yy, value, class_="small", text_anchor="middle"))

    parts.append(
        svg_text(
            left,
            height - 25,
            "Observed, unadjusted cohort curves. This is not a FERMAT counterfactual result.",
            class_="subtitle",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def summary_text(report, svg_path):
    lines = [f"OVERALL_STATUS={report['overall_status']}", f"SVG={svg_path}"]
    for group in report["groups"]:
        lines.extend(
            [
                f"[{group['treatment_class']}] status={group['status']}",
                f"rows={group['rows']} day_range={group['first_day']}..{group['last_day']}",
                f"events_within_curve_window={group['events_within_curve_window']}",
                f"curve_step_count={group['curve_step_count']}",
                f"final_cumulative_incidence={100 * group['final_cumulative_incidence']:.4f}%",
                "checks=" + json.dumps(group["checks"], sort_keys=True),
            ]
        )
        for label, landmark in group["landmarks"].items():
            lines.append(
                f"{label}: incidence={100 * landmark['cumulative_incidence']:.4f}% "
                f"at_risk={landmark['patients_at_risk_after_day']}"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    groups = read_curve(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_reports = [
        validate_group(treatment, rows) for treatment, rows in sorted(groups.items())
    ]
    report = {
        "input_csv": str(args.input_csv.resolve()),
        "overall_status": (
            "PASS" if all(item["status"] == "PASS" for item in group_reports) else "FAIL"
        ),
        "meaning": "structural validation of the already calculated observed curves",
        "does_not_mean": "causal effect or FERMAT counterfactual validation",
        "groups": group_reports,
    }
    svg_path = args.output_dir / "observed_mi_stroke_cumulative_incidence.svg"
    json_path = args.output_dir / "curve_validation.json"
    summary_path = args.output_dir / "curve_check_summary.txt"
    render_svg(groups, svg_path)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = summary_text(report, svg_path)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"VALIDATION_JSON={json_path}")
    print(f"SUMMARY={summary_path}")
    return 0 if report["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
