#!/usr/bin/env python3
"""Estimate four adjusted ADM-timing mortality curves by clone-censor-weight.

The nuisance censoring models are fit on train.  The same fitted models and
train-derived weight caps are applied to val.  The script reports every
configured truncation candidate; it does not choose a candidate by comparing
against FERMAT or by selecting the largest treatment contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from snuh_task30_adm_ccw_core import (  # noqa: E402
    LANDMARK_DAYS,
    STRATEGIES,
    apply_censor_weights,
    build_month_intervals,
    clone_patients,
    curve_landmarks,
    effective_sample_size,
    fit_censor_models,
    weighted_km,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_INPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_inputs"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw"
BASELINE_NUMERIC = [
    "age",
    "index_year",
    "index_hba1c",
    "visit_rows_1y",
    "visit_days_1y",
    "condition_rows_1y",
    "condition_days_1y",
    "nonadm_drug_rows_1y",
    "nonadm_drug_days_1y",
    "inpatient_visits_1y",
    "ambulatory_ed_visits_1y",
]
BASELINE_CATEGORICAL = ["gender_concept_id"]
TIME_VARYING_NUMERIC = [
    "lag_hba1c",
    "lag_visit_rows_30d",
    "lag_visit_days_30d",
    "lag_condition_rows_30d",
    "lag_condition_days_30d",
    "lag_nonadm_drug_rows_30d",
    "lag_nonadm_drug_days_30d",
    "lag_inpatient_visits_30d",
    "lag_ambulatory_ed_visits_30d",
]
DEFAULT_TRIM_CANDIDATES = {
    "UNTRUNCATED": (0.0, 1.0),
    "P99_5": (0.0, 0.995),
    "P99": (0.0, 0.99),
    "P98": (0.0, 0.98),
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log(f"[WRITE] {path}")


def write_frame(frame, path):
    path = Path(path)
    if path.suffix == ".parquet":
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    else:
        frame.to_csv(path, index=False)
    log(f"[WRITE] {path} rows={len(frame):,}")


def prepare_output(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(exist_ok=True)


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_covariate_spec(input_dir):
    path = Path(input_dir) / "covariate_spec.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {"baseline_numeric", "baseline_categorical", "time_varying_numeric"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"{path} missing keys: {missing}")
        if not value.get("treatment_exposure_excluded_from_drug_utilization", False):
            raise ValueError("covariate spec does not confirm ADM exclusion from drug utilization")
        # These two columns are deterministic transformations of baseline age /
        # index year and follow-up time. Including all of them together created
        # the near-collinearity seen in the first Pod fit. Keep the source
        # columns in the parquet, but exclude them from the nuisance model.
        redundant = [
            name
            for name in ("lag_age", "lag_calendar_year")
            if name in value["time_varying_numeric"]
        ]
        value = dict(value)
        value["time_varying_numeric"] = [
            name for name in value["time_varying_numeric"] if name not in redundant
        ]
        value["excluded_redundant_time_varying_covariates"] = redundant
        return value, path
    # Local synthetic tests exercise the runner without a Pod extractor.
    return {
        "baseline_numeric": list(BASELINE_NUMERIC),
        "baseline_categorical": list(BASELINE_CATEGORICAL),
        "time_varying_numeric": list(TIME_VARYING_NUMERIC),
        "treatment_exposure_excluded_from_drug_utilization": True,
        "source": "local_test_fallback",
    }, None


def restricted_cubic_spline(values, knots):
    values = np.asarray(values, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    if len(knots) != 5 or not np.all(np.diff(knots) > 0):
        raise ValueError(f"age spline requires five distinct increasing knots: {knots}")
    low, penultimate, high = knots[0], knots[-2], knots[-1]
    scale = max((high - low) ** 2, 1.0)
    columns = []
    for knot in knots[:-2]:
        first = np.maximum(values - knot, 0.0) ** 3
        second = np.maximum(values - penultimate, 0.0) ** 3
        third = np.maximum(values - high, 0.0) ** 3
        basis = (
            first
            - ((high - knot) / (high - penultimate)) * second
            + ((penultimate - knot) / (high - penultimate)) * third
        ) / scale
        columns.append(basis)
    return np.column_stack(columns)


def add_train_age_spline(patients, monthly, covariate_spec):
    train_age = pd.to_numeric(
        patients.loc[patients["split"].astype(str).eq("train"), "age"], errors="raise"
    ).to_numpy(float)
    knots = np.quantile(train_age, [0.05, 0.275, 0.5, 0.725, 0.95])
    if len(np.unique(knots)) < 5:
        if covariate_spec.get("source") == "local_test_fallback":
            knots = np.array([20.0, 35.0, 50.0, 65.0, 85.0])
        else:
            raise ValueError(f"train age distribution cannot support five-knot spline: {knots}")
    patient_basis = restricted_cubic_spline(patients["age"], knots)
    baseline_numeric = list(covariate_spec["baseline_numeric"])
    time_varying_numeric = list(covariate_spec["time_varying_numeric"])
    for index in range(3):
        baseline_name = f"age_rcs_{index + 1}"
        patients[baseline_name] = patient_basis[:, index]
        baseline_numeric.append(baseline_name)
    covariate_spec = dict(covariate_spec)
    covariate_spec["baseline_numeric"] = baseline_numeric
    covariate_spec["time_varying_numeric"] = time_varying_numeric
    covariate_spec["age_spline_knots_train"] = [float(value) for value in knots]
    return patients, monthly, covariate_spec


def add_followup_time_spline(intervals):
    """Add the paper-specified five-knot spline of follow-up days."""
    result = intervals.copy()
    result["followup_day"] = (
        pd.to_datetime(result["interval_start"]) - pd.to_datetime(result["index_date"])
    ).dt.days.astype(float)
    reference = result.loc[
        result["split"].astype(str).eq("train")
        & result["strategy"].eq("NO_INIT_WITHIN_12M")
        & result["month"].lt(12),
        "followup_day",
    ].to_numpy(float)
    knots = np.quantile(reference, [0.05, 0.275, 0.5, 0.725, 0.95])
    if len(np.unique(knots)) < 5:
        raise ValueError(f"follow-up distribution cannot support five-knot spline: {knots}")
    basis = restricted_cubic_spline(result["followup_day"], knots)
    columns = ["followup_day"]
    for index in range(3):
        name = f"followup_day_rcs_{index + 1}"
        result[name] = basis[:, index]
        columns.append(name)
    return result, columns, [float(value) for value in knots]


def load_inputs(input_dir):
    patients_path = Path(input_dir) / "ccw_patients.parquet"
    monthly_path = Path(input_dir) / "ccw_monthly_covariates.parquet"
    patients = pd.read_parquet(patients_path)
    monthly = pd.read_parquet(monthly_path)
    covariate_spec, covariate_spec_path = load_covariate_spec(input_dir)
    baseline_numeric = covariate_spec["baseline_numeric"]
    baseline_categorical = covariate_spec["baseline_categorical"]
    time_varying_numeric = covariate_spec["time_varying_numeric"]
    required_patients = {
        "patient_key", "split", "index_date", "first_adm_date", "death_date",
        "observation_end_date", *baseline_numeric, *baseline_categorical,
    }
    required_monthly = {"patient_key", "split", "month", *time_varying_numeric}
    require_columns(patients, required_patients, "patients")
    require_columns(monthly, required_monthly, "monthly covariates")
    for column in ("index_date", "first_adm_date", "death_date", "observation_end_date"):
        patients[column] = pd.to_datetime(patients[column])
    patients["patient_key"] = patients["patient_key"].astype("int64")
    monthly["patient_key"] = monthly["patient_key"].astype("int64")
    if patients["patient_key"].duplicated().any():
        raise ValueError("duplicate patient_key")
    patients, monthly, covariate_spec = add_train_age_spline(
        patients, monthly, covariate_spec
    )
    return patients, monthly, patients_path, monthly_path, covariate_spec, covariate_spec_path


def attach_monthly_covariates(intervals, monthly, time_varying_numeric=None):
    time_varying_numeric = list(time_varying_numeric or TIME_VARYING_NUMERIC)
    result = intervals.merge(
        monthly.drop(columns=["split"]),
        on=["patient_key", "month"],
        how="left",
        validate="many_to_one",
    )
    first_year = result["month"].lt(12)
    missing = result.loc[first_year, time_varying_numeric].isna().all(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum()):,} first-year clone-months lack covariates")
    # Artificial censoring ends after month 12, so later values are irrelevant
    # to the censoring models.  Carry month-11 values only to keep the saved
    # counting-process table complete and explicit.
    result[time_varying_numeric] = result.groupby(
        ["patient_key", "strategy"], sort=False
    )[time_varying_numeric].ffill()
    return result


def summarize_weights(weighted, scope, trim_name, caps):
    rows = []
    for strategy in STRATEGIES:
        group = weighted.loc[weighted["strategy"].eq(strategy)]
        final = group.sort_values("month").groupby("patient_key", as_index=False).tail(1)
        values = final["weight"].to_numpy(float)
        raw = final["weight_untrimmed"].to_numpy(float)
        unstabilized = final["unstabilized_weight_untrimmed"].to_numpy(float)
        rows.append(
            {
                "scope": scope,
                "trim": trim_name,
                "strategy": strategy,
                "patients": int(len(final)),
                "artificially_censored": int(final["artificial_censor"].sum()),
                "weight_min": float(values.min()),
                "weight_p50": float(np.quantile(values, 0.5)),
                "weight_p95": float(np.quantile(values, 0.95)),
                "weight_p99": float(np.quantile(values, 0.99)),
                "weight_max": float(values.max()),
                "untrimmed_max": float(raw.max()),
                "effective_sample_size": effective_sample_size(values),
                "unstabilized_weight_min": float(unstabilized.min()),
                "unstabilized_weight_p50": float(np.quantile(unstabilized, 0.5)),
                "unstabilized_weight_p95": float(np.quantile(unstabilized, 0.95)),
                "unstabilized_weight_p99": float(np.quantile(unstabilized, 0.99)),
                "unstabilized_weight_max": float(unstabilized.max()),
                "unstabilized_effective_sample_size": effective_sample_size(unstabilized),
                "cap_low": float(caps[strategy]["low"]),
                "cap_high": float(caps[strategy]["high"]),
            }
        )
    return pd.DataFrame(rows)


def analyze_scope(
    intervals,
    models,
    trim_candidates,
    baseline_numeric,
    baseline_categorical,
    time_varying_numeric,
    training_caps=None,
):
    all_curves, all_landmarks, all_weights, caps_by_trim = [], [], [], []
    for trim_name, quantiles in trim_candidates.items():
        caps = None if training_caps is None else training_caps[trim_name]
        weighted, derived_caps = apply_censor_weights(
            intervals,
            models,
            baseline_numeric=baseline_numeric,
            baseline_categorical=baseline_categorical,
            time_varying_numeric=time_varying_numeric,
            trim_quantiles=quantiles,
            trim_caps=caps,
        )
        curve = weighted_km(weighted)
        landmarks = curve_landmarks(curve, LANDMARK_DAYS)
        curve.insert(0, "trim", trim_name)
        landmarks.insert(0, "trim", trim_name)
        all_curves.append(curve)
        all_landmarks.append(landmarks)
        all_weights.append((trim_name, weighted, derived_caps))
        caps_by_trim.append((trim_name, derived_caps))
    return (
        pd.concat(all_curves, ignore_index=True),
        pd.concat(all_landmarks, ignore_index=True),
        all_weights,
        dict(caps_by_trim),
    )


def self_test():
    from snuh_task30_adm_ccw_core import self_test as core_self_test

    core_self_test()
    print("RUNNER_SELF_TEST_OK")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    prepare_output(args.output_dir, args.overwrite)
    (
        patients,
        monthly,
        patients_path,
        monthly_path,
        covariate_spec,
        covariate_spec_path,
    ) = load_inputs(args.input_dir)
    baseline_numeric = covariate_spec["baseline_numeric"]
    baseline_categorical = covariate_spec["baseline_categorical"]
    time_varying_numeric = covariate_spec["time_varying_numeric"]
    clones = clone_patients(patients)
    intervals = attach_monthly_covariates(
        build_month_intervals(clones), monthly, time_varying_numeric
    )
    intervals, followup_numeric, followup_knots = add_followup_time_spline(intervals)
    write_frame(
        intervals[
            [
                "patient_key", "split", "strategy", "month", "interval_start",
                "interval_stop", "stop_reason", "death_event", "artificial_censor",
                "event_in_interval", "artificial_censor_in_interval",
            ]
        ],
        args.output_dir / "raw" / "clone_month_structure.parquet",
    )

    train = intervals.loc[intervals["split"].eq("train")].copy()
    val = intervals.loc[intervals["split"].eq("val")].copy()
    if train.empty or val.empty:
        raise ValueError("both train and val clone-month rows are required")
    models = fit_censor_models(
        train,
        baseline_numeric=baseline_numeric,
        baseline_categorical=baseline_categorical,
        time_varying_numeric=time_varying_numeric,
        ridge=args.ridge,
        followup_numeric=followup_numeric,
    )
    model_diagnostics = []
    for strategy, value in models.items():
        model_diagnostics.append(
            {
                "strategy": strategy,
                "rows": value["rows"],
                "weighted_rows": value["weighted_rows"],
                "remained": value["remained"],
                "censored": value["censored"],
                "numerator_converged": value["numerator"].converged,
                "denominator_converged": value["denominator"].converged,
                "numerator_iterations": value["numerator"].iterations,
                "denominator_iterations": value["denominator"].iterations,
                "numerator_predictors": value["numerator_predictors"],
                "denominator_predictors": value["denominator_predictors"],
                "numerator_ridge": value["numerator"].ridge,
                "denominator_ridge": value["denominator"].ridge,
                "numerator_max_abs_step": value["numerator"].max_abs_step,
                "denominator_max_abs_step": value["denominator"].max_abs_step,
                "numerator_max_abs_gradient": value["numerator"].max_abs_gradient,
                "denominator_max_abs_gradient": value["denominator"].max_abs_gradient,
                "numerator_line_search_reductions": value["numerator"].line_search_reductions,
                "denominator_line_search_reductions": value["denominator"].line_search_reductions,
            }
        )
    model_diagnostics = pd.DataFrame(model_diagnostics)
    if not model_diagnostics[["numerator_converged", "denominator_converged"]].all().all():
        write_frame(model_diagnostics, args.output_dir / "censor_model_diagnostics.csv")
        raise RuntimeError("at least one censoring model did not converge")

    train_curve, train_landmarks, train_weight_sets, train_caps = analyze_scope(
        train,
        models,
        DEFAULT_TRIM_CANDIDATES,
        baseline_numeric,
        baseline_categorical,
        time_varying_numeric,
    )
    val_curve, val_landmarks, val_weight_sets, _ = analyze_scope(
        val,
        models,
        DEFAULT_TRIM_CANDIDATES,
        baseline_numeric,
        baseline_categorical,
        time_varying_numeric,
        training_caps=train_caps,
    )
    train_curve.insert(0, "scope", "train")
    val_curve.insert(0, "scope", "val")
    train_landmarks.insert(0, "scope", "train")
    val_landmarks.insert(0, "scope", "val")

    weight_summaries = []
    for trim_name, weighted, caps in train_weight_sets:
        weight_summaries.append(summarize_weights(weighted, "train", trim_name, caps))
        if trim_name in {"UNTRUNCATED", "P99"}:
            write_frame(
                weighted[
                    [
                        "patient_key", "strategy", "month", "weight_untrimmed",
                        "weight", "unstabilized_weight_untrimmed",
                    ]
                ],
                args.output_dir / "raw" / f"train_weights_{trim_name.lower()}.parquet",
            )
    for trim_name, weighted, caps in val_weight_sets:
        weight_summaries.append(summarize_weights(weighted, "val", trim_name, caps))
        if trim_name in {"UNTRUNCATED", "P99"}:
            write_frame(
                weighted[
                    [
                        "patient_key", "strategy", "month", "weight_untrimmed",
                        "weight", "unstabilized_weight_untrimmed",
                    ]
                ],
                args.output_dir / "raw" / f"val_weights_{trim_name.lower()}.parquet",
            )

    curves = pd.concat([train_curve, val_curve], ignore_index=True)
    landmarks = pd.concat([train_landmarks, val_landmarks], ignore_index=True)
    weights = pd.concat(weight_summaries, ignore_index=True)
    write_frame(model_diagnostics, args.output_dir / "censor_model_diagnostics.csv")
    write_frame(weights, args.output_dir / "weight_diagnostics.csv")
    write_frame(curves, args.output_dir / "ccw_mortality_curves.csv")
    write_frame(landmarks, args.output_dir / "ccw_mortality_landmarks.csv")
    write_json(train_caps, args.output_dir / "train_weight_caps.json")
    write_json(covariate_spec, args.output_dir / "resolved_covariate_spec.json")

    manifest = {
        "status": "CCW_FOUR_STRATEGY_CURVES_COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_patients": str(patients_path),
        "input_monthly": str(monthly_path),
        "input_covariate_spec": (
            None if covariate_spec_path is None else str(covariate_spec_path)
        ),
        "resolved_covariate_spec": covariate_spec,
        "strategies": list(STRATEGIES),
        "outcome": "all-cause mortality",
        "model_fit_split": "train",
        "validation_split": "val",
        "test_used": False,
        "trim_candidates": DEFAULT_TRIM_CANDIDATES,
        "trim_caps_source": "train",
        "selection_rule": "choose/freeze from val weight stability and effective sample size; never from FERMAT agreement",
        "ipcw_schedule": {
            "initiation_strategies": "single IPCW at each grace-period endpoint",
            "no_initiation_strategy": "interval-specific monthly IPCW accumulated through month 12",
        },
        "followup_time_model": (
            "follow-up days with restricted cubic spline at train-derived "
            "5/27.5/50/72.5/95 percentile knots"
        ),
        "followup_time_spline_knots_train": followup_knots,
        "age_model": "restricted cubic spline with train-derived 5/27.5/50/72.5/95 percentile knots",
        "ridge_penalty": float(args.ridge),
        "weight_roles": {
            "outcome_curve": "P(adherence|baseline)/P(adherence|baseline,time-varying), P99 candidates reported",
            "selection_balance": "1/P(adherence|baseline,time-varying), untrimmed and not used for outcome curves",
        },
        "redundant_covariates_excluded": covariate_spec.get(
            "excluded_redundant_time_varying_covariates", []
        ),
        "treatment_exposure_excluded_from_drug_utilization": True,
        "claim_boundary": (
            "SNUH-adapted CCW using measured demographics, HbA1c, reviewed "
            "comorbidities, ATC comedications, and health-care use; unmeasured "
            "confounding remains and this is not a literal replication"
        ),
    }
    write_json(manifest, args.output_dir / "manifest.json")
    summary = [
        "## STATUS",
        manifest["status"],
        "## CENSOR_MODELS",
        model_diagnostics.to_csv(index=False).rstrip(),
        "## WEIGHT_DIAGNOSTICS",
        weights.to_csv(index=False).rstrip(),
        "## LANDMARKS",
        landmarks.to_csv(index=False).rstrip(),
        "## CLAIM_BOUNDARY",
        manifest["claim_boundary"],
    ]
    (args.output_dir / "return_summary.txt").write_text("\n".join(summary) + "\n")
    print((args.output_dir / "return_summary.txt").read_text(), end="", flush=True)
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
