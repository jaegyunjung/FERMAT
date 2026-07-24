#!/usr/bin/env python3
"""CPU-only follow-up audit for the Task 30 diabetes-to-CKD pathway.

The script reuses the patient-level pathway file produced by the Task 30 CPU
pathway feasibility audit. It performs two checks before any GPU rollout:

1. Fit cause-specific Cox models for CKD with diabetes-record recency plus age,
   sex, and utilization adjustment.
2. Identify patients with pre-index renal abnormality signals from saved eGFR
   and serum-creatinine features, then repeat the recency analysis after those
   patients are excluded.

No torch import, checkpoint loading, model fitting on GPU, or rollout occurs.
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
except ModuleNotFoundError:  # pragma: no cover - Pod dependency
    pq = None


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_diabetes_ckd_cpu_followup.json"
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
DEFAULT_LAB_FILE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "lab_marker_features"
    / "lab_marker_features_wide_20180101.parquet"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_ckd_cpu_followup"

EGFR_MARKERS = ("egfr_mdrd", "egfr_ckdepi")
CREATININE_MARKER = "serum_creatinine"
REQUIRED_PATHWAY_COLUMNS = {
    "pathway_id",
    "person_id",
    "split",
    "source_recency_days",
    "source_recency_stratum",
    "duration_days",
    "event_type",
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pathway-patient-file", type=Path, default=DEFAULT_PATHWAY_PATIENT_FILE
    )
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--lab-file", type=Path, default=DEFAULT_LAB_FILE)
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
    for name in (
        "config_file",
        "pathway_patient_file",
        "feature_file",
        "lab_file",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def load_config(path):
    require_file(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "pathway_id",
        "index_date",
        "horizon_years",
        "recency_reference",
        "minimum_model_patients",
        "minimum_model_events",
        "adjustment_numeric",
        "adjustment_categorical",
        "renal_screen",
        "analysis_cohorts",
        "model_scopes",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    renal_required = {
        "egfr_cutoff",
        "recent_days",
        "creatinine_male_cutoff",
        "creatinine_female_cutoff",
        "creatinine_other_cutoff",
        "male_gender_concept_id",
        "female_gender_concept_id",
    }
    missing = sorted(renal_required - set(config["renal_screen"]))
    if missing:
        raise ValueError(f"renal_screen missing keys: {missing}")
    expected_cohorts = {
        "full",
        "exclude_any_known_renal_abnormality",
        "measured_no_renal_abnormality",
    }
    if set(config["analysis_cohorts"]) != expected_cohorts:
        raise ValueError(f"analysis_cohorts must equal {sorted(expected_cohorts)}")
    if not set(config["model_scopes"]).issubset({"test", "all"}):
        raise ValueError("model_scopes may only contain test and all")
    if int(config["horizon_years"]) != 5:
        raise ValueError("This audit requires a 5-year horizon")
    return config


def prepare_output(args, fingerprint):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output} exists and is not empty; use a new directory")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    write_json(fingerprint, output / "run_config.json")


def load_inputs(args, config):
    pathway_columns = set(parquet_columns(args.pathway_patient_file))
    missing = sorted(REQUIRED_PATHWAY_COLUMNS - pathway_columns)
    if missing:
        raise ValueError(f"Pathway patient file missing columns: {missing}")
    pathway = pd.read_parquet(args.pathway_patient_file)
    pathway = pathway.loc[pathway["pathway_id"].eq(config["pathway_id"])].copy()
    if pathway.empty:
        raise ValueError(f"No rows for pathway_id={config['pathway_id']}")
    if pathway["person_id"].duplicated().any():
        raise ValueError("Selected pathway contains duplicate person_id rows")

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

    wanted_labs = ["person_id", "split"]
    available_labs = set(parquet_columns(args.lab_file))
    for marker in (*EGFR_MARKERS, CREATININE_MARKER):
        for suffix in ("lab_count", "value_min", "value_max", "latest_value", "days_since_latest"):
            column = f"lab_{marker}__{suffix}"
            if column in available_labs:
                wanted_labs.append(column)
    labs = pd.read_parquet(args.lab_file, columns=wanted_labs)
    if labs.duplicated(["person_id", "split"]).any():
        raise ValueError("Lab feature file contains duplicate keys")
    if not any(column.startswith("lab_egfr_") for column in wanted_labs):
        raise ValueError("No eGFR features found")
    if not any(column.startswith("lab_serum_creatinine__") for column in wanted_labs):
        raise ValueError("No serum-creatinine features found")

    data = pathway.merge(features, on=["person_id", "split"], how="left", validate="one_to_one")
    data = data.merge(labs, on=["person_id", "split"], how="left", validate="one_to_one")
    missing_features = data[config["adjustment_numeric"]].isna().all(axis=1)
    if missing_features.any():
        raise ValueError(
            f"Pathway rows missing all baseline features: {int(missing_features.sum()):,}"
        )
    return data


def numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def derive_renal_flags(data, config):
    result = data.copy()
    renal = config["renal_screen"]
    egfr_cutoff = float(renal["egfr_cutoff"])
    recent_days = int(renal["recent_days"])

    any_egfr_measured = pd.Series(False, index=result.index)
    any_low_egfr = pd.Series(False, index=result.index)
    recent_low_egfr = pd.Series(False, index=result.index)
    for marker in EGFR_MARKERS:
        minimum = numeric(result, f"lab_{marker}__value_min")
        latest = numeric(result, f"lab_{marker}__latest_value")
        days = numeric(result, f"lab_{marker}__days_since_latest")
        count = numeric(result, f"lab_{marker}__lab_count").fillna(0)
        measured = count.gt(0) | minimum.notna() | latest.notna()
        any_egfr_measured |= measured
        any_low_egfr |= minimum.lt(egfr_cutoff)
        recent_low_egfr |= latest.lt(egfr_cutoff) & days.between(0, recent_days)

    gender = numeric(result, "gender_concept_id")
    creatinine_cutoff = pd.Series(
        float(renal["creatinine_other_cutoff"]), index=result.index, dtype=float
    )
    creatinine_cutoff.loc[
        gender.eq(int(renal["male_gender_concept_id"]))
    ] = float(renal["creatinine_male_cutoff"])
    creatinine_cutoff.loc[
        gender.eq(int(renal["female_gender_concept_id"]))
    ] = float(renal["creatinine_female_cutoff"])
    creatinine_max = numeric(result, "lab_serum_creatinine__value_max")
    creatinine_latest = numeric(result, "lab_serum_creatinine__latest_value")
    creatinine_days = numeric(result, "lab_serum_creatinine__days_since_latest")
    creatinine_count = numeric(result, "lab_serum_creatinine__lab_count").fillna(0)
    creatinine_measured = (
        creatinine_count.gt(0) | creatinine_max.notna() | creatinine_latest.notna()
    )
    any_high_creatinine = creatinine_max.gt(creatinine_cutoff)
    recent_high_creatinine = (
        creatinine_latest.gt(creatinine_cutoff)
        & creatinine_days.between(0, recent_days)
    )

    result["renal_lab_measured"] = any_egfr_measured | creatinine_measured
    result["any_low_egfr"] = any_low_egfr
    result["recent_low_egfr"] = recent_low_egfr
    result["any_high_creatinine"] = any_high_creatinine
    result["recent_high_creatinine"] = recent_high_creatinine
    result["any_known_renal_abnormality"] = any_low_egfr | any_high_creatinine
    result["recent_renal_abnormality"] = recent_low_egfr | recent_high_creatinine
    result["renal_screen_status"] = np.select(
        [
            result["recent_renal_abnormality"],
            result["any_known_renal_abnormality"],
            result["renal_lab_measured"],
        ],
        [
            "recent_abnormality",
            "historical_only_abnormality",
            "measured_no_abnormality",
        ],
        default="not_measured",
    )
    return result


def cohort_mask(frame, cohort):
    if cohort == "full":
        return pd.Series(True, index=frame.index)
    if cohort == "exclude_any_known_renal_abnormality":
        return ~frame["any_known_renal_abnormality"].astype(bool)
    if cohort == "measured_no_renal_abnormality":
        return frame["renal_lab_measured"].astype(bool) & ~frame[
            "any_known_renal_abnormality"
        ].astype(bool)
    raise ValueError(cohort)


def cumulative_incidence_at(frame, max_day):
    if frame.empty:
        return np.nan
    duration = frame["duration_days"].to_numpy(dtype=np.int64)
    event_type = frame["event_type"].to_numpy(dtype=np.int8)
    target = np.bincount(duration[event_type == 1], minlength=max_day + 1)[: max_day + 1]
    death = np.bincount(duration[event_type == 2], minlength=max_day + 1)[: max_day + 1]
    censor = np.bincount(duration[event_type == 0], minlength=max_day + 1)[: max_day + 1]
    risk = len(frame)
    survival = 1.0
    incidence = 0.0
    for day in range(max_day + 1):
        if risk > 0:
            incidence += survival * int(target[day]) / risk
            survival *= 1.0 - (int(target[day]) + int(death[day])) / risk
        risk -= int(target[day]) + int(death[day]) + int(censor[day])
        if risk < 0:
            raise RuntimeError("Risk set became negative")
    return float(incidence)


def build_descriptive_summaries(data, config):
    horizon_day = int(
        (pd.Timestamp(config["index_date"]) + pd.DateOffset(years=5) - pd.Timestamp(config["index_date"])).days
    )
    renal_rows = []
    recency_rows = []
    balance_rows = []
    for scope in config["model_scopes"]:
        scoped = data if scope == "all" else data.loc[data["split"].eq(scope)]
        for status, group in scoped.groupby("renal_screen_status", sort=True):
            renal_rows.append(
                {
                    "scope": scope,
                    "renal_screen_status": status,
                    "patients": int(len(group)),
                    "target_events_5y": int(group["event_type"].eq(1).sum()),
                    "observed_cumulative_incidence_5y": cumulative_incidence_at(group, horizon_day),
                }
            )
        for cohort in config["analysis_cohorts"]:
            selected = scoped.loc[cohort_mask(scoped, cohort)]
            for stratum, group in selected.groupby("source_recency_stratum", sort=True):
                recency_rows.append(
                    {
                        "scope": scope,
                        "analysis_cohort": cohort,
                        "source_recency_stratum": str(stratum),
                        "patients": int(len(group)),
                        "target_events_5y": int(group["event_type"].eq(1).sum()),
                        "observed_cumulative_incidence_5y": cumulative_incidence_at(group, horizon_day),
                    }
                )
                row = {
                    "scope": scope,
                    "analysis_cohort": cohort,
                    "source_recency_stratum": str(stratum),
                    "patients": int(len(group)),
                    "renal_abnormality_fraction": float(group["any_known_renal_abnormality"].mean())
                    if len(group)
                    else np.nan,
                }
                for column in config["adjustment_numeric"]:
                    values = pd.to_numeric(group[column], errors="coerce")
                    row[f"{column}__mean"] = float(values.mean()) if len(group) else np.nan
                balance_rows.append(row)
    return pd.DataFrame(renal_rows), pd.DataFrame(recency_rows), pd.DataFrame(balance_rows)


def prepare_design(frame, config, recency_model):
    columns = []
    names = []
    scales = []
    if recency_model == "continuous":
        values = np.log2(1.0 + frame["source_recency_days"].to_numpy(dtype=float) / 365.25)
        columns.append(values)
        names.append("log2_1plus_diabetes_record_years")
        scales.append("per_one_unit")
    elif recency_model == "categorical":
        reference = config["recency_reference"]
        categories = sorted(frame["source_recency_stratum"].astype(str).unique().tolist())
        if reference not in categories:
            raise ValueError(f"Missing recency reference={reference}")
        for category in categories:
            if category == reference:
                continue
            columns.append(frame["source_recency_stratum"].astype(str).eq(category).to_numpy(float))
            names.append(f"recency={category}_vs_{reference}")
            scales.append("indicator")
    else:
        raise ValueError(recency_model)

    for column in config["adjustment_numeric"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        array = values.fillna(median).to_numpy(dtype=float)
        mean = float(array.mean())
        std = float(array.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        columns.append((array - mean) / std)
        names.append(column)
        scales.append(f"per_sd={std:.8g}")

    for column in config["adjustment_categorical"]:
        values = frame[column].astype("string").fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        reference = categories[0]
        for category in categories[1:]:
            columns.append(values.eq(category).to_numpy(float))
            names.append(f"{column}={category}_vs_{reference}")
            scales.append("indicator")
    matrix = np.column_stack(columns).astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Design matrix contains nonfinite values")
    return matrix, names, scales


def cox_loglik_gradient_information(beta, x, duration, event, ridge):
    order = np.argsort(-duration, kind="mergesort")
    d = duration[order]
    e = event[order]
    xs = x[order]
    eta = np.clip(xs @ beta, -30.0, 30.0)
    weight = np.exp(eta)
    cum0 = np.cumsum(weight)
    cum1 = np.cumsum(weight[:, None] * xs, axis=0)
    outer = weight[:, None, None] * xs[:, :, None] * xs[:, None, :]
    cum2 = np.cumsum(outer, axis=0)
    boundaries = np.r_[np.flatnonzero(d[1:] != d[:-1]), len(d) - 1]
    starts = np.r_[0, boundaries[:-1] + 1]
    loglik = 0.0
    gradient = np.zeros(x.shape[1], dtype=float)
    information = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for start, stop in zip(starts, boundaries):
        group_event = e[start : stop + 1].astype(bool)
        count = int(group_event.sum())
        if count == 0:
            continue
        event_x = xs[start : stop + 1][group_event]
        risk0 = float(cum0[stop])
        risk1 = cum1[stop]
        risk2 = cum2[stop]
        mean = risk1 / risk0
        loglik += float(eta[start : stop + 1][group_event].sum()) - count * np.log(risk0)
        gradient += event_x.sum(axis=0) - count * mean
        information += count * (risk2 / risk0 - np.outer(mean, mean))
    penalized = loglik - 0.5 * ridge * float(beta @ beta)
    gradient -= ridge * beta
    information += ridge * np.eye(x.shape[1])
    return penalized, gradient, information


def fit_cox(x, duration, event, ridge, max_iterations, tolerance):
    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    final_iteration = 0
    loglik, gradient, information = cox_loglik_gradient_information(
        beta, x, duration, event, ridge
    )
    for iteration in range(1, max_iterations + 1):
        final_iteration = iteration
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        scale = 1.0
        accepted = False
        while scale >= 1 / 1024:
            candidate = beta + scale * step
            new_loglik, new_gradient, new_information = cox_loglik_gradient_information(
                candidate, x, duration, event, ridge
            )
            if new_loglik >= loglik - 1e-10:
                beta = candidate
                loglik = new_loglik
                gradient = new_gradient
                information = new_information
                accepted = True
                break
            scale /= 2.0
        if not accepted:
            break
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break
    covariance = np.linalg.pinv(information)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
    return beta, standard_error, converged, final_iteration, loglik


def run_models(data, config, args):
    coefficient_rows = []
    diagnostic_rows = []
    for scope in config["model_scopes"]:
        scoped = data if scope == "all" else data.loc[data["split"].eq(scope)]
        for cohort in config["analysis_cohorts"]:
            frame = scoped.loc[cohort_mask(scoped, cohort)].copy()
            events = int(frame["event_type"].eq(1).sum())
            if (
                len(frame) < int(config["minimum_model_patients"])
                or events < int(config["minimum_model_events"])
            ):
                diagnostic_rows.append(
                    {
                        "scope": scope,
                        "analysis_cohort": cohort,
                        "recency_model": "not_fit",
                        "patients": int(len(frame)),
                        "events": events,
                        "status": "insufficient_data",
                        "iterations": 0,
                        "penalized_loglik": np.nan,
                    }
                )
                continue
            duration = frame["duration_days"].to_numpy(dtype=float)
            event = frame["event_type"].eq(1).to_numpy(dtype=np.int8)
            for recency_model in ("continuous", "categorical"):
                x, names, scales = prepare_design(frame, config, recency_model)
                beta, se, converged, iterations, loglik = fit_cox(
                    x,
                    duration,
                    event,
                    args.ridge,
                    args.max_iterations,
                    args.tolerance,
                )
                diagnostic_rows.append(
                    {
                        "scope": scope,
                        "analysis_cohort": cohort,
                        "recency_model": recency_model,
                        "patients": int(len(frame)),
                        "events": events,
                        "status": "converged" if converged else "not_converged",
                        "iterations": iterations,
                        "penalized_loglik": loglik,
                    }
                )
                for name, scale, estimate, error in zip(names, scales, beta, se):
                    coefficient_rows.append(
                        {
                            "scope": scope,
                            "analysis_cohort": cohort,
                            "recency_model": recency_model,
                            "feature": name,
                            "feature_scale": scale,
                            "log_hazard_ratio": estimate,
                            "standard_error": error,
                            "hazard_ratio": float(np.exp(estimate)),
                            "ci95_low": float(np.exp(estimate - 1.96 * error)),
                            "ci95_high": float(np.exp(estimate + 1.96 * error)),
                            "patients": int(len(frame)),
                            "events": events,
                            "model_converged": converged,
                        }
                    )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostic_rows)


def build_return_summary(data, renal_summary, recency_summary, coefficients, diagnostics, output):
    renal_test = renal_summary.loc[renal_summary["scope"].eq("test")]
    recency_test = recency_summary.loc[recency_summary["scope"].eq("test")]
    recency_terms = coefficients.loc[
        coefficients["scope"].eq("test")
        & coefficients["feature"].str.startswith(("log2_1plus", "recency="), na=False)
    ]
    lines = [
        "## STATUS",
        "COMPLETE_TASK30_DIABETES_CKD_CPU_FOLLOWUP",
        "## TEST_RENAL_SCREEN",
        renal_test.to_csv(index=False).rstrip(),
        "## TEST_RECENCY_BY_RENAL_COHORT",
        recency_test.to_csv(index=False).rstrip(),
        "## TEST_ADJUSTED_RECENCY_HAZARD_RATIOS",
        recency_terms.to_csv(index=False).rstrip(),
        "## MODEL_DIAGNOSTICS",
        diagnostics.to_csv(index=False).rstrip(),
        "## CLAIM_BOUNDARY",
        "RENAL_LAB_FLAGS_ARE_SCREENING_SIGNALS_NOT_CKD_DIAGNOSES",
        "NO_GPU_NO_CHECKPOINT_NO_ROLLOUT",
        "## OUTPUT_DIR",
        str(output),
    ]
    text = "\n".join(lines) + "\n"
    (Path(output) / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def run_self_test():
    rng = np.random.default_rng(42)
    n = 600
    x = rng.integers(0, 2, size=n).astype(float)
    rate = 0.01 * np.exp(0.9 * x)
    event_time = rng.exponential(1 / rate)
    censor_time = np.full(n, 200.0)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.int8)
    matrix = x[:, None]
    beta, se, converged, _, _ = fit_cox(matrix, duration, event, 1e-6, 60, 1e-8)
    if not converged or not (0.5 < beta[0] < 1.3) or not (0 < se[0] < 0.5):
        raise AssertionError(f"Cox self-test failed beta={beta[0]} se={se[0]}")

    frame = pd.DataFrame(
        {
            "gender_concept_id": [8507, 8532, 8507],
            "lab_egfr_mdrd__lab_count": [2, 1, 0],
            "lab_egfr_mdrd__value_min": [45.0, 80.0, np.nan],
            "lab_egfr_mdrd__latest_value": [55.0, 80.0, np.nan],
            "lab_egfr_mdrd__days_since_latest": [100.0, 100.0, np.nan],
            "lab_serum_creatinine__lab_count": [2, 2, 0],
            "lab_serum_creatinine__value_max": [1.5, 0.9, np.nan],
            "lab_serum_creatinine__latest_value": [1.4, 0.9, np.nan],
            "lab_serum_creatinine__days_since_latest": [100.0, 100.0, np.nan],
        }
    )
    config = {
        "renal_screen": {
            "egfr_cutoff": 60,
            "recent_days": 730,
            "creatinine_male_cutoff": 1.3,
            "creatinine_female_cutoff": 1.1,
            "creatinine_other_cutoff": 1.2,
            "male_gender_concept_id": 8507,
            "female_gender_concept_id": 8532,
        }
    }
    screened = derive_renal_flags(frame, config)
    if screened["renal_screen_status"].tolist() != [
        "recent_abnormality",
        "measured_no_abnormality",
        "not_measured",
    ]:
        raise AssertionError("Renal-screen classification failed")
    log("[SELF-TEST PASS] CPU Cox fit and renal-abnormality classification")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    for path in (
        args.config_file,
        args.pathway_patient_file,
        args.feature_file,
        args.lab_file,
    ):
        require_file(path)
    config = load_config(args.config_file)
    fingerprint = {
        "config_file": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "pathway_patient_file": str(args.pathway_patient_file),
        "feature_file": str(args.feature_file),
        "lab_file": str(args.lab_file),
        "ridge": args.ridge,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    prepare_output(args, fingerprint)
    started = time.time()
    data = derive_renal_flags(load_inputs(args, config), config)
    raw_path = args.output_dir / "raw" / "diabetes_ckd_analysis_patients.parquet"
    atomic_to_parquet(data, raw_path)
    log(f"[RAW SAVED] {raw_path}")

    renal_summary, recency_summary, balance = build_descriptive_summaries(data, config)
    coefficients, diagnostics = run_models(data, config, args)
    renal_summary.to_csv(args.output_dir / "renal_abnormality_summary.csv", index=False)
    recency_summary.to_csv(args.output_dir / "recency_outcomes_by_renal_cohort.csv", index=False)
    balance.to_csv(args.output_dir / "covariates_by_recency.csv", index=False)
    coefficients.to_csv(args.output_dir / "adjusted_cox_coefficients.csv", index=False)
    diagnostics.to_csv(args.output_dir / "adjusted_cox_diagnostics.csv", index=False)

    manifest = {
        **fingerprint,
        "status": "COMPLETE_TASK30_DIABETES_CKD_CPU_FOLLOWUP",
        "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "patients": int(len(data)),
        "test_patients": int(data["split"].eq("test").sum()),
        "test_events": int(
            (data["split"].eq("test") & data["event_type"].eq(1)).sum()
        ),
        "model": "cause-specific Cox; death treated as censoring",
        "renal_screen_boundary": config["renal_screen"].get("interpretation", ""),
        "explicitly_not_run": [
            "torch import",
            "GPU use",
            "checkpoint loading",
            "FERMAT rollout",
        ],
        "outputs": {
            "renal_summary": str(args.output_dir / "renal_abnormality_summary.csv"),
            "recency_summary": str(args.output_dir / "recency_outcomes_by_renal_cohort.csv"),
            "covariates": str(args.output_dir / "covariates_by_recency.csv"),
            "cox_coefficients": str(args.output_dir / "adjusted_cox_coefficients.csv"),
            "cox_diagnostics": str(args.output_dir / "adjusted_cox_diagnostics.csv"),
            "raw_patient_level": str(raw_path),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    build_return_summary(
        data,
        renal_summary,
        recency_summary,
        coefficients,
        diagnostics,
        args.output_dir,
    )
    log("[COMPLETE] Task 30 diabetes-to-CKD CPU follow-up finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
