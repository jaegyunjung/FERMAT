#!/usr/bin/env python3
"""Compare observed and model-predicted group all-cause mortality curves.

This is Task 30 stage 1.  It uses the fixed 2018-01-01 downstream cohort,
queries all-cause death dates, fits two Cox models (clinical counts vs FERMAT
embedding), and compares their mean test-set predictions with the observed
Kaplan-Meier mortality curve.  It does not edit tokens or generate individual
counterfactual curves.
"""

from __future__ import annotations

import argparse
import gc
import getpass
import html
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    psycopg = None
    sql = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    torch = None

from scripts.run_snuh_task20_cox_survival_models import (
    COUNT_CATEGORICAL,
    COUNT_NUMERIC,
    concordance_index,
    fit_cox_model,
    fit_transformer,
    horizon_auc,
    parquet_columns,
    transform_features,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "patient_phenotype_labels_wide_20180101.parquet"
)
DEFAULT_FEATURE_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "baseline_features"
    / "baseline_features_20180101.parquet"
)
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "group_all_cause_mortality_curves"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
APPLICATION_NAME = "fermat_task30_group_mortality_curves"
MODEL_SPECS = {
    "clinical_cox": (list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)),
    "fermat_embedding_cox": (None, []),
}
LANDMARKS = (365, 1095, 1826)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--death-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--horizon-years", type=int, default=5)
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=os.environ.get("SNUH_CDM_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SNUH_CDM_PORT", "5432")))
    parser.add_argument("--dbname", default=os.environ.get("SNUH_CDM_DATABASE", "cdm"))
    parser.add_argument("--user", default=os.environ.get("SNUH_CDM_USER", "jaegyun_jung"))
    parser.add_argument("--schema", default=os.environ.get("SNUH_CDM_SCHEMA", DEFAULT_SCHEMA))
    parser.add_argument("--sslmode", default=os.environ.get("SNUH_CDM_SSLMODE", "disable"))
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--min-train-events", type=int, default=50)
    parser.add_argument("--min-val-events", type=int, default=10)
    parser.add_argument("--min-test-events", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log(f"[WRITE] {path}")


def prepare_output(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(exist_ok=True)


def require_dependencies():
    missing = []
    if psycopg is None:
        missing.append("psycopg")
    if torch is None:
        missing.append("torch")
    if missing:
        raise RuntimeError("Missing Pod dependencies: " + ", ".join(missing))


def password():
    value = os.environ.get("SNUH_CDM_PASSWORD")
    return value if value else getpass.getpass("SNUH_CDM_PASSWORD: ")


def connect(args):
    conn = psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=password(),
        sslmode=args.sslmode,
        connect_timeout=15,
        application_name=APPLICATION_NAME,
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (args.statement_timeout,))
    return conn


def validate_columns(path, required):
    available = set(parquet_columns(path))
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def load_death_dates(args):
    if args.death_cache is not None:
        path = args.death_cache.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        log(f"[DEATH CACHE] {path} rows={len(frame):,}")
        return frame, str(path)

    started = time.time()
    log("[START] query all-cause death dates")
    with connect(args) as conn, conn.cursor() as cur:
        statement = sql.SQL(
            """
            SELECT person_id::bigint, MIN(death_date)::date AS death_date
            FROM {}.death
            WHERE death_date IS NOT NULL
              AND death_date <= %s::date
            GROUP BY person_id
            """
        ).format(sql.Identifier(args.schema))
        cur.execute(statement, (args.db_end_date,))
        columns = [item.name for item in cur.description]
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    frame["death_date"] = pd.to_datetime(frame["death_date"], errors="coerce")
    cache_path = args.output_dir / "all_cause_death_dates.parquet"
    frame.to_parquet(cache_path, index=False)
    log(
        f"[DONE] query all-cause death dates: rows={len(frame):,}, "
        f"seconds={time.time() - started:,.1f}"
    )
    log(f"[RAW SAVED] {cache_path}")
    return frame, str(cache_path)


def load_modeling_data(args, death_dates):
    for path in (args.label_file, args.feature_file, args.embedding_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    label_cols = [
        "person_id",
        "split",
        "index_date",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
    ]
    feature_cols = ["person_id", "split", *COUNT_NUMERIC, *COUNT_CATEGORICAL]
    embedding_available = parquet_columns(args.embedding_file)
    emb_cols = [column for column in embedding_available if column.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"No emb_* columns in {args.embedding_file}")
    embedding_cols = [
        "person_id",
        "split",
        "has_embedding_sequence",
        "sequence_length_pre_index",
        *emb_cols,
    ]
    validate_columns(args.label_file, label_cols)
    validate_columns(args.feature_file, feature_cols)
    validate_columns(args.embedding_file, embedding_cols)

    log("[START] load fixed-index labels")
    labels = pd.read_parquet(args.label_file, columns=label_cols)
    for column in ("index_date", "first_activity_date", "last_activity_date"):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")
    log(f"[DONE] load fixed-index labels: rows={len(labels):,}")
    fixed_index = pd.Timestamp(args.index_date)
    unexpected_index = labels["index_date"].notna() & labels["index_date"].ne(fixed_index)
    if unexpected_index.any():
        raise ValueError(
            f"Label file contains {int(unexpected_index.sum()):,} rows outside index date {args.index_date}"
        )

    log("[START] load clinical baseline features")
    features = pd.read_parquet(args.feature_file, columns=feature_cols)
    log(f"[DONE] load clinical baseline features: rows={len(features):,}")

    log("[START] load block-2048 FERMAT embeddings")
    embeddings = pd.read_parquet(args.embedding_file, columns=embedding_cols)
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()
    log(f"[DONE] load block-2048 FERMAT embeddings: rows={len(embeddings):,}")

    for name, frame in (("labels", labels), ("features", features), ("embeddings", embeddings)):
        duplicates = int(frame.duplicated(["person_id", "split"]).sum())
        if duplicates:
            raise ValueError(f"{name} contains {duplicates:,} duplicate person_id/split rows")
    if death_dates["person_id"].duplicated().any():
        raise ValueError("Death-date input contains duplicate person_id rows")

    data = labels.merge(features, on=["person_id", "split"], how="inner")
    data = data.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(death_dates, on="person_id", how="left")
    return data, emb_cols


def build_mortality_task(data, args):
    index_date = pd.Timestamp(args.index_date)
    horizon_end = index_date + pd.DateOffset(years=int(args.horizon_years))
    db_end = pd.Timestamp(args.db_end_date)
    censor_limit = min(horizon_end, db_end)

    task = data.loc[data["has_pre_index_washout"].astype(bool)].copy()
    task = task.loc[task["death_date"].isna() | (task["death_date"] > index_date)].copy()
    event = (
        task["death_date"].notna()
        & (task["death_date"] > index_date)
        & (task["death_date"] <= censor_limit)
    )
    censor_date = task["last_activity_date"].where(
        task["last_activity_date"].notna(), censor_limit
    )
    censor_date = censor_date.clip(upper=censor_limit)
    endpoint = censor_date.where(~event, task["death_date"])
    task["duration_days"] = (endpoint - index_date).dt.days.astype("float32")
    task["event"] = event.astype("int8")
    task = task.loc[task["duration_days"] > 0].copy()
    return task, int((horizon_end - index_date).days)


def deterministic_cap(frame, max_rows, seed):
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.sort_values("person_id")
    return frame.sample(n=max_rows, random_state=seed).sort_values("person_id")


def split_task(task, args):
    train = deterministic_cap(
        task.loc[task["split"].eq("train")].copy(), args.max_train_rows, args.random_seed
    )
    val = deterministic_cap(
        task.loc[task["split"].eq("val")].copy(), args.max_eval_rows, args.random_seed + 1
    )
    test = deterministic_cap(
        task.loc[task["split"].eq("test")].copy(), args.max_eval_rows, args.random_seed + 2
    )
    return train, val, test


def validate_event_counts(train, val, test, args):
    thresholds = {
        "train": (train, args.min_train_events),
        "val": (val, args.min_val_events),
        "test": (test, args.min_test_events),
    }
    for name, (frame, minimum) in thresholds.items():
        events = int(frame["event"].sum())
        if events < minimum:
            raise RuntimeError(f"Too few {name} deaths: {events} < {minimum}")


def cohort_summary(train, val, test):
    rows = []
    for name, frame in (("train", train), ("val", val), ("test", test)):
        rows.append(
            {
                "split": name,
                "patients": int(len(frame)),
                "deaths_within_horizon": int(frame["event"].sum()),
                "censored": int((frame["event"] == 0).sum()),
                "median_followup_days": float(frame["duration_days"].median()),
                "max_followup_days": int(frame["duration_days"].max()),
            }
        )
    return pd.DataFrame(rows)


def observed_km_curve(duration, event, max_day):
    frame = pd.DataFrame(
        {
            "duration_days": np.asarray(duration, dtype=np.int32),
            "event": np.asarray(event, dtype=np.int8),
        }
    )
    counts = frame.groupby(["duration_days", "event"]).size().unstack(fill_value=0)
    at_risk = int(len(frame))
    survival = 1.0
    rows = []
    for day in range(max_day + 1):
        deaths = censored = 0
        if day > 0 and day in counts.index:
            values = counts.loc[day]
            censored = int(values.get(0, 0))
            deaths = int(values.get(1, 0))
            if at_risk > 0 and deaths > 0:
                survival *= 1.0 - deaths / at_risk
            at_risk -= deaths + censored
        rows.append(
            {
                "day": day,
                "patients_at_risk_after_day": at_risk,
                "deaths_on_day": deaths,
                "censored_on_day": censored,
                "observed_mortality": 1.0 - survival,
            }
        )
    return pd.DataFrame(rows)


def breslow_baseline_curve(duration, event, score, max_day):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(duration) & np.isfinite(score) & (duration > 0)
    duration = duration[valid]
    event = event[valid]
    score = score[valid]
    order = np.argsort(duration, kind="mergesort")
    duration_sorted = duration[order]
    exp_score_sorted = np.exp(np.clip(score[order], -50, 50))
    suffix_risk = np.cumsum(exp_score_sorted[::-1], dtype=np.float64)[::-1]
    event_times, event_counts = np.unique(
        duration[(event == 1) & (duration <= max_day)], return_counts=True
    )
    first_at_risk = np.searchsorted(duration_sorted, event_times, side="left")
    risk_sums = suffix_risk[first_at_risk]
    increments = np.divide(
        event_counts.astype(np.float64),
        risk_sums,
        out=np.zeros_like(risk_sums, dtype=np.float64),
        where=risk_sums > 0,
    )
    return event_times.astype(np.float64), np.cumsum(increments)


def cumulative_hazard_at_days(event_times, cumulative_hazard, days):
    days = np.asarray(days, dtype=np.float64)
    index = np.searchsorted(event_times, days, side="right") - 1
    output = np.zeros(len(days), dtype=np.float64)
    valid = index >= 0
    output[valid] = cumulative_hazard[index[valid]]
    return output


def mean_risk_curve(scores, baseline_hazard, day_chunk=64):
    relative_hazard = np.exp(np.clip(np.asarray(scores, dtype=np.float64), -50, 50))
    baseline_hazard = np.asarray(baseline_hazard, dtype=np.float64)
    output = np.zeros(len(baseline_hazard), dtype=np.float64)
    for start in range(0, len(baseline_hazard), day_chunk):
        stop = min(start + day_chunk, len(baseline_hazard))
        block = -np.expm1(
            -relative_hazard[:, None] * baseline_hazard[None, start:stop]
        )
        output[start:stop] = block.mean(axis=0)
    return output


def fit_one_model(name, train, val, test, feature_cols, categorical_cols, args, max_day):
    started = time.time()
    log(f"[START] fit {name}")
    transformer = fit_transformer(train, feature_cols, categorical_cols, [])
    train_x, feature_names = transform_features(train, transformer)
    val_x, _ = transform_features(val, transformer)
    test_x, _ = transform_features(test, transformer)
    beta, best_epoch, best_val_loss, history = fit_cox_model(
        train_x,
        train["duration_days"].to_numpy(dtype=np.float32),
        train["event"].to_numpy(dtype=np.int8),
        val_x,
        val["duration_days"].to_numpy(dtype=np.float32),
        val["event"].to_numpy(dtype=np.int8),
        args,
    )
    train_score = train_x @ beta
    test_score = test_x @ beta
    event_times, cumulative_hazard = breslow_baseline_curve(
        train["duration_days"], train["event"], train_score, max_day
    )
    days = np.arange(max_day + 1, dtype=np.int32)
    baseline_at_days = cumulative_hazard_at_days(event_times, cumulative_hazard, days)
    mean_curve = mean_risk_curve(test_score, baseline_at_days)
    auc, cases, controls = horizon_auc(
        test["duration_days"], test["event"], test_score, max_day
    )
    metric = {
        "model": name,
        "features": int(len(feature_names)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "test_c_index": float(
            concordance_index(test["duration_days"], test["event"], test_score)
        ),
        "test_5y_auc": float(auc),
        "test_5y_cases": int(cases),
        "test_5y_controls": int(controls),
        "fit_seconds": time.time() - started,
    }
    model_dir = args.output_dir / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "cox_model.npz",
        beta=beta,
        event_times=event_times,
        cumulative_baseline_hazard=cumulative_hazard,
        days=days,
        mean_test_risk=mean_curve,
    )
    write_json(transformer, model_dir / "transformer.json")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    pd.DataFrame({"day": days, f"{name}_mean_predicted": mean_curve}).to_csv(
        model_dir / "mean_test_risk_curve.csv", index=False
    )
    write_json(metric, model_dir / "metrics.json")
    log(
        f"[DONE] fit {name}: c_index={metric['test_c_index']:.4f}, "
        f"auc5y={metric['test_5y_auc']:.4f}, seconds={metric['fit_seconds']:,.1f}"
    )
    return metric, mean_curve


def validate_curve(curve):
    checks = {}
    for column in (
        "observed_mortality",
        "clinical_cox_mean_predicted",
        "fermat_embedding_cox_mean_predicted",
    ):
        values = curve[column].to_numpy(dtype=np.float64)
        checks[column] = {
            "decreases": int(np.sum(np.diff(values) < -1e-12)),
            "invalid_probability_values": int(
                np.sum(~np.isfinite(values) | (values < -1e-12) | (values > 1 + 1e-12))
            ),
        }
    checks["patients_at_risk_increases"] = int(
        np.sum(np.diff(curve["patients_at_risk_after_day"].to_numpy()) > 0)
    )
    passed = (
        all(
            item["decreases"] == 0 and item["invalid_probability_values"] == 0
            for item in checks.values()
            if isinstance(item, dict)
        )
        and checks["patients_at_risk_increases"] == 0
    )
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def nice_y_max(value):
    target = max(0.02, float(value) * 1.15)
    magnitude = 10 ** math.floor(math.log10(target))
    normalized = target / magnitude
    step = 0.2 if normalized <= 1 else 0.5 if normalized <= 2.5 else 1.0
    return math.ceil(normalized / step) * step * magnitude


def svg_text(x, y, value, **attrs):
    rendered = " ".join(
        f'{key.rstrip("_").replace("_", "-")}="{html.escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {rendered}>{html.escape(str(value))}</text>'


def render_svg(curve, path):
    width, height = 1180, 760
    left, right, top, bottom = 115, 55, 95, 210
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_day = int(curve["day"].max())
    columns = [
        ("observed_mortality", "Observed", "#222222", "4", ""),
        ("clinical_cox_mean_predicted", "Clinical Cox", "#D1495B", "3", "9 6"),
        ("fermat_embedding_cox_mean_predicted", "FERMAT embedding + Cox", "#00798C", "3", ""),
    ]
    max_value = max(float(curve[column].max()) for column, *_ in columns)
    y_max = nice_y_max(max_value)

    def x(day):
        return left + plot_width * day / max_day

    def y(value):
        return top + plot_height * (1.0 - value / y_max)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.axis{font-size:14px}.small{font-size:13px}.label{font-size:15px;font-weight:600}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#555}</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        svg_text(left, 38, "Observed and predicted all-cause mortality", class_="title"),
        svg_text(
            left,
            64,
            "Held-out test patients; index date 2018-01-01; five-year follow-up",
            class_="subtitle",
        ),
    ]
    for index in range(6):
        value = y_max * index / 5
        yy = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#E1E5E8" stroke-width="1"/>'
        )
        parts.append(
            svg_text(left - 14, yy + 5, f"{100 * value:.1f}%", class_="axis", text_anchor="end")
        )
    ticks = [(0, "Start"), (365, "1 year"), (1095, "3 years"), (1826, "5 years")]
    for day, label in ticks:
        if day > max_day:
            continue
        xx = x(day)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_height}" stroke="#E1E5E8" stroke-width="1"/>'
        )
        parts.append(svg_text(xx, top + plot_height + 27, label, class_="axis", text_anchor="middle"))
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#333"/>',
            f'<text x="30" y="{top + plot_height / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 30 {top + plot_height / 2:.1f})">All-cause mortality risk</text>',
        ]
    )
    days = curve["day"].to_numpy(dtype=np.int32)
    legend_x = width - right - 330
    for index, (column, label, color, stroke_width, dash) in enumerate(columns):
        values = curve[column].to_numpy(dtype=np.float64)
        commands = [f"M {x(days[0]):.2f} {y(values[0]):.2f}"]
        previous = values[0]
        for day, value in zip(days[1:], values[1:]):
            xx = x(day)
            commands.append(f"H {xx:.2f}")
            if abs(value - previous) > 1e-12:
                commands.append(f"V {y(value):.2f}")
            previous = value
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>'
        )
        legend_y = top + 16 + index * 27
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+38}" y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>'
        )
        parts.append(svg_text(legend_x + 48, legend_y + 5, label, class_="label"))

    table_top = top + plot_height + 78
    parts.append(svg_text(left, table_top, "Patients at risk after day", class_="label"))
    for day, label in ticks:
        if day > max_day:
            continue
        xx = x(day)
        parts.append(svg_text(xx, table_top + 29, label, class_="small", text_anchor="middle"))
        row = curve.loc[curve["day"].eq(day)].iloc[0]
        parts.append(
            svg_text(
                xx,
                table_top + 57,
                f"{int(row['patients_at_risk_after_day']):,}",
                class_="label",
                text_anchor="middle",
            )
        )
    parts.append(
        svg_text(
            left,
            height - 27,
            "Observed: Kaplan-Meier mortality. Predicted: mean of individual Cox risks in the same test patients.",
            class_="subtitle",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    log(f"[WRITE] {path}")


def main():
    args = parse_args()
    require_dependencies()
    for name in ("label_file", "feature_file", "embedding_file", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.death_cache is not None:
        args.death_cache = args.death_cache.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    started = datetime.now(timezone.utc)

    endpoint = {
        "endpoint": "all-cause mortality",
        "source": f"{args.schema}.death.death_date",
        "index_date": args.index_date,
        "horizon_years": args.horizon_years,
        "observed_curve": "Kaplan-Meier 1-S(t) in held-out test patients",
        "predicted_curves": ["clinical Cox", "FERMAT embedding + Cox"],
        "token_edits": False,
        "individual_curves": False,
    }
    write_json(endpoint, args.output_dir / "endpoint_definition.json")
    death_dates, death_cache_path = load_death_dates(args)
    data, emb_cols = load_modeling_data(args, death_dates)
    task, max_day = build_mortality_task(data, args)
    train, val, test = split_task(task, args)
    summary = cohort_summary(train, val, test)
    summary.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    log(f"[WRITE] {args.output_dir / 'cohort_summary.csv'}")
    print("## COHORT_SUMMARY_PRECHECK", flush=True)
    print(summary.to_csv(index=False).rstrip(), flush=True)
    validate_event_counts(train, val, test, args)

    observed = observed_km_curve(test["duration_days"], test["event"], max_day)
    observed.to_csv(args.output_dir / "observed_test_mortality_curve.csv", index=False)
    log(f"[RAW SAVED] {args.output_dir / 'observed_test_mortality_curve.csv'}")

    model_metrics = []
    model_curves = {}
    for name, (feature_cols, categorical_cols) in MODEL_SPECS.items():
        if feature_cols is None:
            feature_cols = emb_cols
        metric, mean_curve = fit_one_model(
            name,
            train,
            val,
            test,
            feature_cols,
            categorical_cols,
            args,
            max_day,
        )
        model_metrics.append(metric)
        model_curves[name] = mean_curve
        gc.collect()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve = observed.copy()
    curve["clinical_cox_mean_predicted"] = model_curves["clinical_cox"]
    curve["fermat_embedding_cox_mean_predicted"] = model_curves[
        "fermat_embedding_cox"
    ]
    curve.to_csv(args.output_dir / "group_mortality_curves.csv", index=False)

    landmark = curve.loc[curve["day"].isin(LANDMARKS)].copy()
    for column in (
        "clinical_cox_mean_predicted",
        "fermat_embedding_cox_mean_predicted",
    ):
        landmark[f"{column}_minus_observed"] = (
            landmark[column] - landmark["observed_mortality"]
        )
    landmark.to_csv(args.output_dir / "group_mortality_landmarks.csv", index=False)
    metrics_frame = pd.DataFrame(model_metrics)
    observed_values = curve["observed_mortality"].to_numpy(dtype=np.float64)
    for name in MODEL_SPECS:
        column = f"{name}_mean_predicted"
        index = metrics_frame["model"].eq(name)
        metrics_frame.loc[index, "mean_absolute_curve_error"] = float(
            np.mean(np.abs(curve[column].to_numpy(dtype=np.float64) - observed_values))
        )
        metrics_frame.loc[index, "observed_5y_mortality"] = float(observed_values[-1])
        metrics_frame.loc[index, "mean_predicted_5y_mortality"] = float(curve[column].iloc[-1])
    metrics_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)

    validation = validate_curve(curve)
    write_json(validation, args.output_dir / "curve_validation.json")
    render_svg(curve, args.output_dir / "group_all_cause_mortality_curves.svg")

    finished = datetime.now(timezone.utc)
    manifest = {
        "status": "complete",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "inputs": {
            "label_file": str(args.label_file),
            "feature_file": str(args.feature_file),
            "embedding_file": str(args.embedding_file),
            "death_cache": death_cache_path,
        },
        "embedding_features": len(emb_cols),
        "max_curve_day": max_day,
        "patient_identifiers_in_summary_outputs": False,
    }
    write_json(manifest, args.output_dir / "manifest.json")

    print("## ENDPOINT", flush=True)
    print(json.dumps(endpoint, indent=2, ensure_ascii=False), flush=True)
    print("## COHORT_SUMMARY", flush=True)
    print(summary.to_csv(index=False).rstrip(), flush=True)
    print("## MODEL_METRICS", flush=True)
    print(metrics_frame.to_csv(index=False).rstrip(), flush=True)
    print("## GROUP_MORTALITY_LANDMARKS", flush=True)
    print(landmark.to_csv(index=False).rstrip(), flush=True)
    print("## CURVE_VALIDATION", flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    print("## JUPYTER_SVG", flush=True)
    print(args.output_dir / "group_all_cause_mortality_curves.svg", flush=True)
    log("[COMPLETE] Group all-cause mortality risk-curve test finished")
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
