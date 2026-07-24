#!/usr/bin/env python3
"""CPU-only analysis of Task 30 five-pathway diagnosis-date shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_INPUT_DIR = TASK30 / "outputs" / "five_pathway_fermat_cox_shift_20260718"
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "five_pathway_shift_cpu_analysis_20260718"
SHIFTS = np.arange(-360, 361, 90, dtype=np.int64)
HORIZONS = (365, 1095, 1826)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def parquet_columns(path):
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except ModuleNotFoundError:
        return pd.read_parquet(path).columns.tolist()


def prepare_output(path, overwrite):
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_rng(seed, key):
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(derived)


def percentile_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def bootstrap_mean_interval(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite and non-empty")
    indexes = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indexes].mean(axis=1)
    lower, upper = percentile_interval(means)
    return float(values.mean()), lower, upper


def rank_correlation(x, y):
    x = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def path_efficiency(values, tolerance=1e-12):
    values = np.asarray(values, dtype=np.float64)
    total_variation = float(np.abs(np.diff(values)).sum())
    endpoint_change = float(values[-1] - values[0])
    if total_variation <= tolerance:
        return 1.0, total_variation, endpoint_change
    return abs(endpoint_change) / total_variation, total_variation, endpoint_change


def validate_shift_frame(frame):
    key = ["pathway_id", "source_role", "person_id", "shift_days"]
    if frame.duplicated(key).any():
        raise RuntimeError("Duplicate pathway/source-role/person/shift rows")
    observed_shifts = sorted(frame["shift_days"].astype(int).unique())
    if observed_shifts != SHIFTS.tolist():
        raise RuntimeError(f"Unexpected shift grid: {observed_shifts}")
    combinations = frame[["pathway_id", "source_role"]].drop_duplicates()
    if len(combinations) != 6:
        raise RuntimeError(f"Expected 6 pathway/source-role combinations, got {len(combinations)}")
    counts = frame.groupby(["pathway_id", "source_role"])["person_id"].nunique()
    if not counts.eq(128).all():
        raise RuntimeError(f"Each pathway/source role must contain 128 patients: {counts.to_dict()}")
    rows_per_patient = frame.groupby(["pathway_id", "source_role", "person_id"]).size()
    if not rows_per_patient.eq(len(SHIFTS)).all():
        raise RuntimeError("Every patient must have all nine shifts")
    if len(frame) != 6912:
        raise RuntimeError(f"Expected 6,912 patient-shift rows, got {len(frame)}")
    zero = frame.loc[frame["shift_days"].eq(0)]
    for horizon in HORIZONS:
        original_range = frame.groupby(
            ["pathway_id", "source_role", "person_id"]
        )[f"original_risk_day_{horizon}"].agg(lambda values: values.max() - values.min())
        if not np.allclose(original_range.to_numpy(dtype=np.float64), 0.0, atol=1e-12):
            raise RuntimeError(
                f"Original risk changes across shift arms at day {horizon}"
            )
        reconstructed = (
            frame[f"edited_risk_day_{horizon}"]
            - frame[f"original_risk_day_{horizon}"]
        )
        if not np.allclose(
            reconstructed.to_numpy(dtype=np.float64),
            frame[f"risk_difference_day_{horizon}"].to_numpy(dtype=np.float64),
            atol=1e-7,
        ):
            raise RuntimeError(f"Risk-difference identity failed at day {horizon}")
        if not np.allclose(
            zero[f"risk_difference_day_{horizon}"].to_numpy(dtype=np.float64),
            0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Shift-zero risk difference is not zero at day {horizon}")
    return {
        "rows": int(len(frame)),
        "patients_per_combination": 128,
        "combinations": int(len(combinations)),
        "shift_grid": observed_shifts,
    }


def summarize_shift_responses(frame, bootstrap_samples, seed):
    curve_rows = []
    patient_rows = []
    group_rows = []
    group_columns = ["pathway_id", "source_role", "source", "target"]
    for keys, group in frame.groupby(group_columns, sort=True):
        pathway_id, source_role, source, target = keys
        group = group.sort_values(["person_id", "shift_days"])
        pivot = group.pivot(
            index="person_id",
            columns="shift_days",
            values="risk_difference_day_1826",
        ).reindex(columns=SHIFTS)
        if pivot.isna().any().any():
            raise RuntimeError(f"Incomplete 5-year response grid for {pathway_id}/{source_role}")
        matrix = pivot.to_numpy(dtype=np.float64)
        x = SHIFTS.astype(np.float64)
        slopes_per_day = matrix @ x / float(np.sum(x * x))
        slopes_per_year = slopes_per_day * 365.0
        endpoint_changes = matrix[:, -1] - matrix[:, 0]
        patient_ranges = matrix.max(axis=1) - matrix.min(axis=1)
        efficiencies = np.zeros(len(pivot), dtype=np.float64)
        correlations = np.zeros(len(pivot), dtype=np.float64)
        adjacent_consistency = np.zeros(len(pivot), dtype=np.float64)
        for index, values in enumerate(matrix):
            efficiencies[index] = path_efficiency(values)[0]
            correlations[index] = rank_correlation(x, values)
            adjacent = np.diff(values)
            expected_sign = np.sign(slopes_per_day[index])
            if expected_sign == 0:
                adjacent_consistency[index] = float(np.mean(np.abs(adjacent) <= 1e-12))
            else:
                adjacent_consistency[index] = float(
                    np.mean(expected_sign * adjacent >= -1e-12)
                )
        for index, person_id in enumerate(pivot.index.astype(np.int64)):
            patient_rows.append(
                {
                    "pathway_id": pathway_id,
                    "source_role": source_role,
                    "source": source,
                    "target": target,
                    "person_id": int(person_id),
                    "slope_per_year": float(slopes_per_year[index]),
                    "endpoint_change_5y_risk": float(endpoint_changes[index]),
                    "response_range_5y_risk": float(patient_ranges[index]),
                    "rank_correlation_shift_vs_risk": float(correlations[index]),
                    "path_efficiency": float(efficiencies[index]),
                    "adjacent_direction_consistency": float(adjacent_consistency[index]),
                }
            )
        for horizon in HORIZONS:
            value_column = f"risk_difference_day_{horizon}"
            for shift, shift_group in group.groupby("shift_days", sort=True):
                values = shift_group[value_column].to_numpy(dtype=np.float64)
                rng = stable_rng(seed, f"curve:{keys}:{horizon}:{int(shift)}")
                mean, lower, upper = bootstrap_mean_interval(
                    values, bootstrap_samples, rng
                )
                curve_rows.append(
                    {
                        "pathway_id": pathway_id,
                        "source_role": source_role,
                        "source": source,
                        "target": target,
                        "horizon_days": int(horizon),
                        "shift_days": int(shift),
                        "patients": int(len(values)),
                        "mean_risk_difference": mean,
                        "bootstrap_lower_95": lower,
                        "bootstrap_upper_95": upper,
                        "median_risk_difference": float(np.median(values)),
                        "p10_risk_difference": float(np.quantile(values, 0.10)),
                        "p90_risk_difference": float(np.quantile(values, 0.90)),
                        "proportion_increased": float(np.mean(values > 0)),
                    }
                )
        mean_curve = matrix.mean(axis=0)
        efficiency, total_variation, endpoint_change = path_efficiency(mean_curve)
        endpoint_rng = stable_rng(seed, f"endpoint:{keys}")
        endpoint_mean, endpoint_lower, endpoint_upper = bootstrap_mean_interval(
            endpoint_changes, bootstrap_samples, endpoint_rng
        )
        slope_rng = stable_rng(seed, f"slope:{keys}")
        slope_mean, slope_lower, slope_upper = bootstrap_mean_interval(
            slopes_per_year, bootstrap_samples, slope_rng
        )
        maximum_index = int(np.argmax(np.abs(mean_curve)))
        original_5y = group.loc[group["shift_days"].eq(0), "original_risk_day_1826"].mean()
        group_rows.append(
            {
                "pathway_id": pathway_id,
                "source_role": source_role,
                "source": source,
                "target": target,
                "patients": int(len(pivot)),
                "mean_original_5y_risk": float(original_5y),
                "mean_difference_at_minus_360": float(mean_curve[0]),
                "mean_difference_at_plus_360": float(mean_curve[-1]),
                "mean_endpoint_contrast": endpoint_mean,
                "endpoint_contrast_lower_95": endpoint_lower,
                "endpoint_contrast_upper_95": endpoint_upper,
                "mean_slope_per_year": slope_mean,
                "slope_per_year_lower_95": slope_lower,
                "slope_per_year_upper_95": slope_upper,
                "group_curve_rank_correlation": rank_correlation(x, mean_curve),
                "group_curve_path_efficiency": float(efficiency),
                "group_curve_total_variation": float(total_variation),
                "maximum_absolute_mean_difference": float(abs(mean_curve[maximum_index])),
                "maximum_absolute_mean_difference_shift_days": int(SHIFTS[maximum_index]),
                "median_patient_rank_correlation": float(np.median(correlations)),
                "median_absolute_patient_rank_correlation": float(
                    np.median(np.abs(correlations))
                ),
                "median_patient_path_efficiency": float(np.median(efficiencies)),
                "median_adjacent_direction_consistency": float(
                    np.median(adjacent_consistency)
                ),
                "median_patient_response_range": float(np.median(patient_ranges)),
                "proportion_patient_endpoint_increased": float(
                    np.mean(endpoint_changes > 0)
                ),
                "endpoint_contrast_ci_excludes_zero": bool(
                    endpoint_lower > 0 or endpoint_upper < 0
                ),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(patient_rows), pd.DataFrame(group_rows)


def km_risk_at_horizon(duration, event, horizon):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    valid = np.isfinite(duration) & (duration > 0)
    duration = duration[valid]
    event = event[valid]
    survival = 1.0
    for event_time in np.unique(duration[event & (duration <= horizon)]):
        at_risk = int(np.sum(duration >= event_time))
        events = int(np.sum(event & np.isclose(duration, event_time, atol=1e-7)))
        if at_risk:
            survival *= 1.0 - events / at_risk
    return float(1.0 - survival)


def summarize_calibration(frame, bootstrap_samples, seed):
    rows = []
    original = frame.loc[
        frame["source_role"].eq("source_a") & frame["shift_days"].eq(0)
    ].copy()
    for (pathway_id, target), group in original.groupby(
        ["pathway_id", "target"], sort=True
    ):
        if group["person_id"].duplicated().any():
            raise RuntimeError(f"Duplicate calibration patients for {pathway_id}")
        duration = group["duration_days"].to_numpy(dtype=np.float64)
        event = group["event"].to_numpy(dtype=np.int8)
        for horizon in HORIZONS:
            model_values = group[f"original_risk_day_{horizon}"].to_numpy(
                dtype=np.float64
            )
            observed = km_risk_at_horizon(duration, event, horizon)
            model_mean = float(model_values.mean())
            gap = model_mean - observed
            rng = stable_rng(seed, f"calibration:{pathway_id}:{horizon}")
            indexes = rng.integers(
                0, len(group), size=(int(bootstrap_samples), len(group))
            )
            bootstrap_gaps = np.empty(int(bootstrap_samples), dtype=np.float64)
            for bootstrap_index, sampled in enumerate(indexes):
                bootstrap_gaps[bootstrap_index] = float(
                    model_values[sampled].mean()
                    - km_risk_at_horizon(
                        duration[sampled], event[sampled], horizon
                    )
                )
            lower, upper = percentile_interval(bootstrap_gaps)
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "target": target,
                    "horizon_days": int(horizon),
                    "patients": int(len(group)),
                    "events_by_horizon": int(np.sum(event.astype(bool) & (duration <= horizon))),
                    "observed_km_risk": observed,
                    "mean_original_model_risk": model_mean,
                    "model_minus_observed": gap,
                    "gap_lower_95": lower,
                    "gap_upper_95": upper,
                    "gap_ci_includes_zero": bool(lower <= 0 <= upper),
                }
            )
    return pd.DataFrame(rows)


def build_screening_summary(group_summary, calibration):
    five_year = calibration.loc[calibration["horizon_days"].eq(1826), [
        "pathway_id",
        "observed_km_risk",
        "mean_original_model_risk",
        "model_minus_observed",
        "gap_lower_95",
        "gap_upper_95",
        "events_by_horizon",
    ]]
    result = group_summary.merge(five_year, on="pathway_id", how="left", validate="many_to_one")
    probability_columns = [
        "mean_original_5y_risk",
        "mean_difference_at_minus_360",
        "mean_difference_at_plus_360",
        "mean_endpoint_contrast",
        "endpoint_contrast_lower_95",
        "endpoint_contrast_upper_95",
        "maximum_absolute_mean_difference",
        "median_patient_response_range",
        "observed_km_risk",
        "mean_original_model_risk",
        "model_minus_observed",
        "gap_lower_95",
        "gap_upper_95",
    ]
    for column in probability_columns:
        result[f"{column}_percentage_points"] = 100.0 * result[column]
    result["absolute_endpoint_contrast"] = result["mean_endpoint_contrast"].abs()
    return result.sort_values(
        ["absolute_endpoint_contrast", "maximum_absolute_mean_difference"],
        ascending=False,
    ).reset_index(drop=True)


def make_plots(curves, calibration, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {"created": False, "reason": f"matplotlib unavailable: {error}"}

    five_year = curves.loc[curves["horizon_days"].eq(1826)].copy()
    combinations = list(
        five_year.groupby(["pathway_id", "source_role", "source"], sort=True)
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, (keys, group) in zip(axes.flat, combinations):
        pathway_id, source_role, source = keys
        group = group.sort_values("shift_days")
        x = group["shift_days"].to_numpy()
        mean = 100.0 * group["mean_risk_difference"].to_numpy()
        lower = 100.0 * group["bootstrap_lower_95"].to_numpy()
        upper = 100.0 * group["bootstrap_upper_95"].to_numpy()
        axis.axhline(0, color="black", linewidth=0.8)
        axis.plot(x, mean, marker="o")
        axis.fill_between(x, lower, upper, alpha=0.2)
        axis.set_title(f"{pathway_id}\n{source_role}: {source}", fontsize=9)
        axis.set_xlabel("Diagnosis-date shift (days)")
        axis.set_ylabel("5-year risk change (percentage points)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    shift_path = output_dir / "five_year_shift_response.png"
    figure.savefig(shift_path, dpi=180)
    plt.close(figure)

    combinations = list(calibration.groupby(["pathway_id", "target"], sort=True))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, (keys, group) in zip(axes.flat, combinations):
        axis.set_visible(True)
        pathway_id, target = keys
        group = group.sort_values("horizon_days")
        years = group["horizon_days"].to_numpy() / 365.25
        axis.plot(years, 100 * group["observed_km_risk"], marker="o", label="Observed")
        axis.plot(
            years,
            100 * group["mean_original_model_risk"],
            marker="o",
            label="FERMAT+Cox",
        )
        axis.set_title(f"{pathway_id}\n{target}", fontsize=9)
        axis.set_xlabel("Years")
        axis.set_ylabel("Cumulative risk (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    calibration_path = output_dir / "observed_vs_original_model.png"
    figure.savefig(calibration_path, dpi=180)
    plt.close(figure)
    return {
        "created": True,
        "five_year_shift_response": str(shift_path),
        "observed_vs_original_model": str(calibration_path),
    }


def self_test():
    rows = []
    for pathway_id, role, sign in (("linear_up", "source_a", 1.0), ("linear_down", "source_a", -1.0)):
        for person_id in range(20):
            for shift in SHIFTS:
                difference = sign * (shift / 360.0) * (0.001 + person_id * 0.000001)
                row = {
                    "pathway_id": pathway_id,
                    "source_role": role,
                    "source": pathway_id,
                    "target": "target",
                    "person_id": person_id,
                    "shift_days": int(shift),
                    "duration_days": 1826.0,
                    "event": 0,
                }
                for horizon in HORIZONS:
                    row[f"original_risk_day_{horizon}"] = 0.1
                    row[f"risk_difference_day_{horizon}"] = difference
                rows.append(row)
    frame = pd.DataFrame(rows)
    curves, patients, groups = summarize_shift_responses(frame, 200, 42)
    if len(curves) != 2 * len(HORIZONS) * len(SHIFTS):
        raise AssertionError("Curve summary self-test row count failed")
    if not np.allclose(groups["group_curve_path_efficiency"], 1.0):
        raise AssertionError("Linear response efficiency self-test failed")
    if not set(np.sign(groups["mean_endpoint_contrast"])) == {-1.0, 1.0}:
        raise AssertionError("Endpoint direction self-test failed")
    if not np.allclose(patients["adjacent_direction_consistency"], 1.0):
        raise AssertionError("Patient direction self-test failed")
    duration = np.asarray([100, 200, 400], dtype=float)
    event = np.asarray([1, 0, 1], dtype=int)
    if not np.isclose(km_risk_at_horizon(duration, event, 365), 1 / 3):
        raise AssertionError("Kaplan-Meier self-test failed")
    validation_rows = []
    for combination in range(6):
        for person_id in range(128):
            for shift in SHIFTS:
                difference = float(shift) * 1e-7
                row = {
                    "pathway_id": f"pathway_{combination}",
                    "source_role": "source_a",
                    "source": "source",
                    "target": "target",
                    "person_id": person_id,
                    "shift_days": int(shift),
                    "duration_days": 1826.0,
                    "event": 0,
                }
                for horizon in HORIZONS:
                    row[f"original_risk_day_{horizon}"] = 0.1
                    row[f"edited_risk_day_{horizon}"] = 0.1 + difference
                    row[f"risk_difference_day_{horizon}"] = difference
                validation_rows.append(row)
    validation = validate_shift_frame(pd.DataFrame(validation_rows))
    if validation["rows"] != 6912 or validation["combinations"] != 6:
        raise AssertionError("Full input-contract self-test failed")
    log("SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = prepare_output(args.output_dir, args.overwrite)
    input_path = input_dir / "patient_shift_embeddings_and_risks.parquet"
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    available = set(parquet_columns(input_path))
    columns = [
        "person_id",
        "pathway_id",
        "source_role",
        "source",
        "target",
        "shift_days",
        "duration_days",
        "event",
        "risk_score_difference",
    ]
    for horizon in HORIZONS:
        columns.extend(
            [
                f"original_risk_day_{horizon}",
                f"edited_risk_day_{horizon}",
                f"risk_difference_day_{horizon}",
            ]
        )
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    started = time.time()
    frame = pd.read_parquet(input_path, columns=columns)
    validation = validate_shift_frame(frame)
    curves, patients, groups = summarize_shift_responses(
        frame, args.bootstrap_samples, args.random_seed
    )
    calibration = summarize_calibration(
        frame, args.bootstrap_samples, args.random_seed
    )
    screening = build_screening_summary(groups, calibration)
    atomic_csv(curves, output_dir / "shift_curve_bootstrap_summary.csv")
    atomic_parquet(patients, output_dir / "patient_response_metrics.parquet")
    atomic_csv(groups, output_dir / "group_response_summary.csv")
    atomic_csv(calibration, output_dir / "observed_model_calibration_summary.csv")
    atomic_csv(screening, output_dir / "pathway_screening_summary.csv")
    plots = make_plots(curves, calibration, output_dir)
    manifest = {
        "status": "COMPLETE_TASK30_FIVE_PATHWAY_SHIFT_CPU_ANALYSIS",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "bootstrap_samples": int(args.bootstrap_samples),
        "random_seed": int(args.random_seed),
        "validation": validation,
        "elapsed_seconds": time.time() - started,
        "uses_gpu": False,
        "plots": plots,
    }
    write_json(manifest, output_dir / "manifest.json")
    report_columns = [
        "pathway_id",
        "source_role",
        "source",
        "target",
        "mean_endpoint_contrast_percentage_points",
        "endpoint_contrast_lower_95_percentage_points",
        "endpoint_contrast_upper_95_percentage_points",
        "endpoint_contrast_ci_excludes_zero",
        "group_curve_rank_correlation",
        "group_curve_path_efficiency",
        "median_patient_rank_correlation",
        "median_patient_path_efficiency",
        "maximum_absolute_mean_difference_percentage_points",
        "observed_km_risk_percentage_points",
        "mean_original_model_risk_percentage_points",
        "model_minus_observed_percentage_points",
        "gap_lower_95_percentage_points",
        "gap_upper_95_percentage_points",
        "events_by_horizon",
    ]
    return_text = "\n".join(
        [
            "## STATUS COMPLETE_TASK30_FIVE_PATHWAY_SHIFT_CPU_ANALYSIS",
            f"pathway_source_combinations {len(screening)}",
            f"bootstrap_samples {args.bootstrap_samples}",
            f"elapsed_seconds {manifest['elapsed_seconds']:.2f}",
            "## PATHWAY_SCREENING",
            screening[report_columns].to_csv(index=False).rstrip(),
            "## OUTPUT_DIR",
            str(output_dir),
        ]
    ) + "\n"
    (output_dir / "RETURN_THIS.txt").write_text(return_text, encoding="utf-8")
    print(return_text, end="", flush=True)
    log("[COMPLETE] Task 30 five-pathway CPU result analysis finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
