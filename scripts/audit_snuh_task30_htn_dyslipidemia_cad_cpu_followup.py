#!/usr/bin/env python3
"""CPU-only adjusted audit of hypertension/dyslipidemia order and later CAD.

The input is the patient-level pathway file from Task 30. Cause-specific Cox
models compare diagnosis order at 1, 3, and 5 years while adjusting for the
time from the second diagnosis to index, the gap between diagnoses, age, sex,
and utilization. A trimmed analysis retains only the overlapping recency/gap
range of the two non-same-day order groups.

No torch import, checkpoint loading, GPU use, or rollout occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_htn_dyslipidemia_cad_cpu_followup.json"
DEFAULT_PATHWAY_PATIENT_FILE = (
    TASK30
    / "outputs"
    / "pathway_feasibility_20260717_175059"
    / "raw"
    / "pathway_patient_level.parquet"
)
DEFAULT_FEATURE_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "baseline_features"
    / "baseline_features_20180101.parquet"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "htn_dyslipidemia_cad_cpu_followup"
NON_SAME_DAY_ORDERS = (
    "dyslipidemia_before_hypertension",
    "hypertension_before_dyslipidemia",
)
REQUIRED_PATHWAY_COLUMNS = {
    "pathway_id",
    "person_id",
    "split",
    "source_recency_days",
    "source_gap_days",
    "source_order",
    "duration_days",
    "event_type",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pathway-patient-file", type=Path, default=DEFAULT_PATHWAY_PATIENT_FILE)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)


def parquet_columns(path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def atomic_to_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_paths(args):
    for name in ("config_file", "pathway_patient_file", "feature_file", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "pathway_id",
        "index_date",
        "horizons_years",
        "order_reference",
        "minimum_model_patients",
        "minimum_model_events",
        "common_support_quantiles",
        "adjustment_numeric",
        "adjustment_categorical",
        "model_scopes",
        "analysis_cohorts",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    if config["horizons_years"] != [1, 3, 5]:
        raise ValueError("horizons_years must be [1, 3, 5]")
    if config["order_reference"] not in NON_SAME_DAY_ORDERS:
        raise ValueError("order_reference must be a non-same-day order")
    expected_cohorts = {
        "all_order_groups",
        "non_same_day",
        "non_same_day_common_support",
    }
    if set(config["analysis_cohorts"]) != expected_cohorts:
        raise ValueError(f"analysis_cohorts must equal {sorted(expected_cohorts)}")
    low, high = map(float, config["common_support_quantiles"])
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid common_support_quantiles")
    return config


def prepare_output(args, fingerprint):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists and is not empty")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
    write_json(fingerprint, args.output_dir / "run_config.json")


def load_inputs(args, config):
    available = set(parquet_columns(args.pathway_patient_file))
    missing = sorted(REQUIRED_PATHWAY_COLUMNS - available)
    if missing:
        raise ValueError(f"Pathway patient file missing columns: {missing}")
    pathway = pd.read_parquet(args.pathway_patient_file)
    pathway = pathway.loc[pathway["pathway_id"].eq(config["pathway_id"])].copy()
    if pathway.empty or pathway["person_id"].duplicated().any():
        raise ValueError("Selected pathway is empty or has duplicate patients")
    if not set(pathway["source_order"].unique()).issubset(
        set(NON_SAME_DAY_ORDERS) | {"same_day"}
    ):
        raise ValueError("Unexpected source_order values")

    feature_columns = [
        "person_id",
        "split",
        *config["adjustment_numeric"],
        *config["adjustment_categorical"],
    ]
    available_features = set(parquet_columns(args.feature_file))
    missing = sorted(set(feature_columns) - available_features)
    if missing:
        raise ValueError(f"Baseline feature file missing columns: {missing}")
    features = pd.read_parquet(args.feature_file, columns=feature_columns)
    if features.duplicated(["person_id", "split"]).any():
        raise ValueError("Baseline feature file contains duplicate keys")
    data = pathway.merge(features, on=["person_id", "split"], how="left", validate="one_to_one")
    if data[config["adjustment_numeric"]].isna().all(axis=1).any():
        raise ValueError("Some pathway patients are missing all adjustment features")
    data["source_gap_days"] = pd.to_numeric(data["source_gap_days"], errors="coerce").fillna(0)
    return data


def common_support_bounds(frame, config):
    low_q, high_q = map(float, config["common_support_quantiles"])
    rows = []
    bounds = {}
    for scope in config["model_scopes"]:
        scoped = frame if scope == "all" else frame.loc[frame["split"].eq(scope)]
        scoped = scoped.loc[scoped["source_order"].isin(NON_SAME_DAY_ORDERS)]
        scope_bounds = {}
        for variable in ("source_recency_days", "source_gap_days"):
            group_bounds = []
            for order in NON_SAME_DAY_ORDERS:
                values = pd.to_numeric(
                    scoped.loc[scoped["source_order"].eq(order), variable], errors="coerce"
                ).dropna()
                if values.empty:
                    raise ValueError(f"No values for scope={scope} order={order}")
                group_bounds.append((float(values.quantile(low_q)), float(values.quantile(high_q))))
            lower = max(value[0] for value in group_bounds)
            upper = min(value[1] for value in group_bounds)
            if lower >= upper:
                raise RuntimeError(f"No common support for scope={scope} variable={variable}")
            scope_bounds[variable] = (lower, upper)
            rows.append(
                {
                    "scope": scope,
                    "variable": variable,
                    "lower": lower,
                    "upper": upper,
                    "quantile_low": low_q,
                    "quantile_high": high_q,
                }
            )
        bounds[scope] = scope_bounds
    return bounds, pd.DataFrame(rows)


def analysis_mask(frame, cohort, scope_bounds=None):
    if cohort == "all_order_groups":
        return pd.Series(True, index=frame.index)
    mask = frame["source_order"].isin(NON_SAME_DAY_ORDERS)
    if cohort == "non_same_day":
        return mask
    if cohort == "non_same_day_common_support":
        if scope_bounds is None:
            raise ValueError("Common-support bounds are required")
        for variable, (lower, upper) in scope_bounds.items():
            values = pd.to_numeric(frame[variable], errors="coerce")
            mask &= values.between(lower, upper)
        return mask
    raise ValueError(cohort)


def cumulative_incidence_at(frame, horizon_day):
    if frame.empty:
        return np.nan
    duration = np.minimum(frame["duration_days"].to_numpy(dtype=np.int64), horizon_day)
    original_duration = frame["duration_days"].to_numpy(dtype=np.int64)
    event_type = frame["event_type"].to_numpy(dtype=np.int8).copy()
    event_type[original_duration > horizon_day] = 0
    target = np.bincount(duration[event_type == 1], minlength=horizon_day + 1)[: horizon_day + 1]
    death = np.bincount(duration[event_type == 2], minlength=horizon_day + 1)[: horizon_day + 1]
    censor = np.bincount(duration[event_type == 0], minlength=horizon_day + 1)[: horizon_day + 1]
    risk = len(frame)
    survival = 1.0
    incidence = 0.0
    for day in range(horizon_day + 1):
        if risk > 0:
            incidence += survival * int(target[day]) / risk
            survival *= 1.0 - (int(target[day]) + int(death[day])) / risk
        risk -= int(target[day]) + int(death[day]) + int(censor[day])
    return float(incidence)


def build_descriptive(data, config, bounds):
    index_date = pd.Timestamp(config["index_date"])
    rows = []
    for scope in config["model_scopes"]:
        scoped = data if scope == "all" else data.loc[data["split"].eq(scope)]
        for cohort in config["analysis_cohorts"]:
            selected = scoped.loc[
                analysis_mask(scoped, cohort, bounds[scope] if "common_support" in cohort else None)
            ]
            for order, group in selected.groupby("source_order", sort=True):
                row = {
                    "scope": scope,
                    "analysis_cohort": cohort,
                    "source_order": order,
                    "patients": int(len(group)),
                    "anchor_recency_days_median": float(group["source_recency_days"].median()),
                    "source_gap_days_median": float(group["source_gap_days"].median()),
                }
                for years in config["horizons_years"]:
                    day = int(((index_date + pd.DateOffset(years=years)) - index_date).days)
                    row[f"target_events_{years}y"] = int(
                        (group["event_type"].eq(1) & (group["duration_days"] <= day)).sum()
                    )
                    row[f"observed_cumulative_incidence_{years}y"] = cumulative_incidence_at(
                        group, day
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def prepare_design(frame, config, cohort):
    columns = []
    names = []
    scales = []
    reference = config["order_reference"]
    categories = sorted(frame["source_order"].astype(str).unique().tolist())
    if reference not in categories:
        raise ValueError(f"Missing order reference={reference}")
    for category in categories:
        if category == reference:
            continue
        columns.append(frame["source_order"].astype(str).eq(category).to_numpy(float))
        names.append(f"order={category}_vs_{reference}")
        scales.append("indicator")

    anchor = np.log2(1.0 + frame["source_recency_days"].to_numpy(float) / 365.25)
    columns.append(anchor)
    names.append("log2_1plus_anchor_recency_years")
    scales.append("per_one_unit")

    if cohort != "all_order_groups":
        gap = np.log2(1.0 + frame["source_gap_days"].to_numpy(float) / 365.25)
        columns.append(gap)
        names.append("log2_1plus_source_gap_years")
        scales.append("per_one_unit")

    for column in config["adjustment_numeric"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        array = values.fillna(median).to_numpy(float)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        columns.append((array - mean) / std)
        names.append(column)
        scales.append(f"per_sd={std:.8g}")

    for column in config["adjustment_categorical"]:
        values = frame[column].astype("string").fillna("__MISSING__")
        values_list = sorted(values.unique().tolist())
        categorical_reference = values_list[0]
        for category in values_list[1:]:
            columns.append(values.eq(category).to_numpy(float))
            names.append(f"{column}={category}_vs_{categorical_reference}")
            scales.append("indicator")
    matrix = np.column_stack(columns).astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite design matrix")
    return matrix, names, scales


def cox_components(beta, x, duration, event, ridge):
    order = np.argsort(-duration, kind="mergesort")
    d = duration[order]
    e = event[order]
    xs = x[order]
    eta = np.clip(xs @ beta, -30, 30)
    weight = np.exp(eta)
    cum0 = np.cumsum(weight)
    cum1 = np.cumsum(weight[:, None] * xs, axis=0)
    cum2 = np.cumsum(weight[:, None, None] * xs[:, :, None] * xs[:, None, :], axis=0)
    stops = np.r_[np.flatnonzero(d[1:] != d[:-1]), len(d) - 1]
    starts = np.r_[0, stops[:-1] + 1]
    loglik = 0.0
    gradient = np.zeros(x.shape[1])
    information = np.zeros((x.shape[1], x.shape[1]))
    for start, stop in zip(starts, stops):
        event_in_group = e[start : stop + 1].astype(bool)
        count = int(event_in_group.sum())
        if count == 0:
            continue
        event_x = xs[start : stop + 1][event_in_group]
        risk0 = float(cum0[stop])
        risk1 = cum1[stop]
        risk2 = cum2[stop]
        mean = risk1 / risk0
        loglik += float(eta[start : stop + 1][event_in_group].sum()) - count * np.log(risk0)
        gradient += event_x.sum(axis=0) - count * mean
        information += count * (risk2 / risk0 - np.outer(mean, mean))
    loglik -= 0.5 * ridge * float(beta @ beta)
    gradient -= ridge * beta
    information += ridge * np.eye(x.shape[1])
    return loglik, gradient, information


def fit_cox(x, duration, event, ridge, max_iterations, tolerance):
    beta = np.zeros(x.shape[1])
    loglik, gradient, information = cox_components(beta, x, duration, event, ridge)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        scale = 1.0
        accepted = False
        while scale >= 1 / 1024:
            candidate = beta + scale * step
            new_loglik, new_gradient, new_information = cox_components(
                candidate, x, duration, event, ridge
            )
            if new_loglik >= loglik - 1e-10:
                beta = candidate
                loglik, gradient, information = new_loglik, new_gradient, new_information
                accepted = True
                break
            scale /= 2
        if not accepted:
            break
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break
    covariance = np.linalg.pinv(information)
    se = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    return beta, se, converged, iteration, loglik


def run_models(data, config, bounds, args):
    index_date = pd.Timestamp(config["index_date"])
    coefficient_rows = []
    diagnostic_rows = []
    for scope in config["model_scopes"]:
        scoped = data if scope == "all" else data.loc[data["split"].eq(scope)]
        for cohort in config["analysis_cohorts"]:
            frame = scoped.loc[
                analysis_mask(scoped, cohort, bounds[scope] if "common_support" in cohort else None)
            ].copy()
            for years in config["horizons_years"]:
                horizon_day = int(((index_date + pd.DateOffset(years=years)) - index_date).days)
                duration = np.minimum(frame["duration_days"].to_numpy(float), horizon_day)
                event = (
                    frame["event_type"].eq(1)
                    & frame["duration_days"].le(horizon_day)
                ).to_numpy(np.int8)
                event_count = int(event.sum())
                if (
                    len(frame) < int(config["minimum_model_patients"])
                    or event_count < int(config["minimum_model_events"])
                ):
                    diagnostic_rows.append(
                        {
                            "scope": scope,
                            "analysis_cohort": cohort,
                            "horizon_years": years,
                            "patients": int(len(frame)),
                            "events": event_count,
                            "status": "insufficient_data",
                            "iterations": 0,
                            "penalized_loglik": np.nan,
                        }
                    )
                    continue
                x, names, scales = prepare_design(frame, config, cohort)
                beta, se, converged, iterations, loglik = fit_cox(
                    x, duration, event, args.ridge, args.max_iterations, args.tolerance
                )
                diagnostic_rows.append(
                    {
                        "scope": scope,
                        "analysis_cohort": cohort,
                        "horizon_years": years,
                        "patients": int(len(frame)),
                        "events": event_count,
                        "status": "converged" if converged else "not_converged",
                        "iterations": iterations,
                        "penalized_loglik": loglik,
                    }
                )
                for name, feature_scale, estimate, error in zip(names, scales, beta, se):
                    coefficient_rows.append(
                        {
                            "scope": scope,
                            "analysis_cohort": cohort,
                            "horizon_years": years,
                            "feature": name,
                            "feature_scale": feature_scale,
                            "log_hazard_ratio": estimate,
                            "standard_error": error,
                            "hazard_ratio": float(np.exp(np.clip(estimate, -30, 30))),
                            "ci95_low": float(np.exp(np.clip(estimate - 1.96 * error, -30, 30))),
                            "ci95_high": float(np.exp(np.clip(estimate + 1.96 * error, -30, 30))),
                            "patients": int(len(frame)),
                            "events": event_count,
                            "model_converged": converged,
                        }
                    )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostic_rows)


def build_return_summary(descriptive, coefficients, diagnostics, bounds_frame, output):
    test_description = descriptive.loc[descriptive["scope"].eq("test")]
    test_terms = coefficients.loc[
        coefficients["scope"].eq("test")
        & coefficients["feature"].str.startswith(("order=", "log2_1plus_source_gap"), na=False)
    ]
    lines = [
        "## STATUS",
        "COMPLETE_TASK30_HTN_DYSLIPIDEMIA_CAD_CPU_FOLLOWUP",
        "## TEST_ORDER_OUTCOMES",
        test_description.to_csv(index=False).rstrip(),
        "## TEST_ADJUSTED_ORDER_AND_GAP",
        test_terms.to_csv(index=False).rstrip(),
        "## COMMON_SUPPORT_BOUNDS",
        bounds_frame.to_csv(index=False).rstrip(),
        "## MODEL_DIAGNOSTICS",
        diagnostics.to_csv(index=False).rstrip(),
        "## CLAIM_BOUNDARY",
        "ASSOCIATION_AUDIT_ONLY_NOT_CAUSAL_NOT_A_TREATMENT_EFFECT",
        "NO_GPU_NO_CHECKPOINT_NO_ROLLOUT",
        "## OUTPUT_DIR",
        str(output),
    ]
    text = "\n".join(lines) + "\n"
    (Path(output) / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    rng = np.random.default_rng(7)
    n = 600
    exposure = rng.integers(0, 2, n).astype(float)
    rate = 0.01 * np.exp(0.7 * exposure)
    event_time = rng.exponential(1 / rate)
    duration = np.minimum(event_time, 200)
    event = (event_time <= 200).astype(np.int8)
    beta, se, converged, _, _ = fit_cox(
        exposure[:, None], duration, event, 1e-6, 60, 1e-8
    )
    if not converged or not (0.3 < beta[0] < 1.1) or not (0 < se[0] < 0.5):
        raise AssertionError(f"Cox self-test failed beta={beta[0]} se={se[0]}")
    frame = pd.DataFrame(
        {
            "split": ["test"] * 8,
            "source_order": list(NON_SAME_DAY_ORDERS) * 4,
            "source_recency_days": [100, 120, 200, 220, 300, 320, 400, 420],
            "source_gap_days": [50, 60, 100, 110, 150, 160, 200, 210],
        }
    )
    config = {
        "model_scopes": ["test"],
        "common_support_quantiles": [0.0, 1.0],
    }
    bounds, _ = common_support_bounds(frame, config)
    selected = analysis_mask(frame, "non_same_day_common_support", bounds["test"])
    if int(selected.sum()) != 6:
        raise AssertionError("Common-support trimming failed")
    log("[SELF-TEST PASS] CPU Cox fit and two-order common-support trimming")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    for path in (args.config_file, args.pathway_patient_file, args.feature_file):
        require_file(path)
    config = load_config(args.config_file)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "pathway_patient_file": str(args.pathway_patient_file),
        "feature_file": str(args.feature_file),
        "ridge": args.ridge,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    prepare_output(args, fingerprint)
    started = time.time()
    data = load_inputs(args, config)
    bounds, bounds_frame = common_support_bounds(data, config)
    data["test_common_support"] = analysis_mask(
        data, "non_same_day_common_support", bounds.get("test")
    ) & data["split"].eq("test")
    data["all_common_support"] = analysis_mask(
        data, "non_same_day_common_support", bounds.get("all")
    )
    raw_path = args.output_dir / "raw" / "htn_dyslipidemia_cad_analysis_patients.parquet"
    atomic_to_parquet(data, raw_path)

    descriptive = build_descriptive(data, config, bounds)
    coefficients, diagnostics = run_models(data, config, bounds, args)
    descriptive.to_csv(args.output_dir / "order_outcome_summary.csv", index=False)
    coefficients.to_csv(args.output_dir / "adjusted_cox_coefficients.csv", index=False)
    diagnostics.to_csv(args.output_dir / "adjusted_cox_diagnostics.csv", index=False)
    bounds_frame.to_csv(args.output_dir / "common_support_bounds.csv", index=False)

    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_HTN_DYSLIPIDEMIA_CAD_CPU_FOLLOWUP",
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": time.time() - started,
        "patients": int(len(data)),
        "test_patients": int(data["split"].eq("test").sum()),
        "model": "cause-specific Cox at 1/3/5 years; death treated as censoring",
        "explicitly_not_run": ["torch import", "GPU use", "checkpoint loading", "rollout"],
        "outputs": {
            "descriptive": str(args.output_dir / "order_outcome_summary.csv"),
            "coefficients": str(args.output_dir / "adjusted_cox_coefficients.csv"),
            "diagnostics": str(args.output_dir / "adjusted_cox_diagnostics.csv"),
            "common_support": str(args.output_dir / "common_support_bounds.csv"),
            "raw_patient_level": str(raw_path),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(descriptive, coefficients, diagnostics, bounds_frame, args.output_dir)
    log("[COMPLETE] Task 30 HTN/dyslipidemia-to-CAD CPU follow-up finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
