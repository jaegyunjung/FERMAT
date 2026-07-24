#!/usr/bin/env python3
"""Audit completed four-strategy HbA1c ADM CCW mortality curves.

This script reads existing CCW inputs and outputs only and does not query the
SNUH database.  Point-estimate censoring models are not reused by the default
bootstrap; they are refit within every resample.  The script adds:

* unweighted and weighted risk-set sizes at 1, 3, and 5 years;
* cumulative death and artificial-censor counts at those landmarks;
* a paired train/validation patient bootstrap that refits censoring models and
  P99 caps in every draw (fixed-weight mode remains diagnostic-only);
* adherence-selection balance diagnostics at each strategy grace-period end.

The default full-refit bootstrap includes nuisance-model estimation by
resampling train and validation patients separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from snuh_task30_adm_ccw_core import (  # noqa: E402
    LANDMARK_DAYS,
    STRATEGIES,
    curve_landmarks,
    effective_sample_size,
    apply_censor_weights,
    build_month_intervals,
    clone_patients,
    fit_censor_models,
    weighted_km,
)
from run_snuh_task30_hba1c_adm_ccw import (  # noqa: E402
    BASELINE_CATEGORICAL,
    BASELINE_NUMERIC,
    TIME_VARYING_NUMERIC,
    attach_monthly_covariates,
    add_followup_time_spline,
    load_inputs as load_runner_inputs,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_INPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_inputs_20260719_v1"
DEFAULT_CCW_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_20260720_013822"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_audit"
STRATEGY_MONTHS = {
    "INIT_WITHIN_3M": 3,
    "INIT_WITHIN_6M": 6,
    "INIT_WITHIN_12M": 12,
    "NO_INIT_WITHIN_12M": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--ccw-dir", type=Path, default=DEFAULT_CCW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trim", default="P99")
    parser.add_argument("--bootstrap-scope", choices=["train", "val"], default="val")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="same prespecified ridge penalty used for the point-estimate censor models",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=["full_refit", "fixed_weights"],
        default="full_refit",
        help="full_refit resamples train and validation patients and refits IPCW each draw",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--balance-only",
        action="store_true",
        help="write corrected unstabilized-IPCW balance diagnostics without rerunning bootstrap",
    )
    parser.add_argument(
        "--resume-bootstrap",
        action="store_true",
        help="resume a checkpointed full-refit bootstrap in the same output directory",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def prepare_output(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    log(f"[WRITE] {path} rows={len(frame):,}")


def write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    log(f"[WRITE] {path}")


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(input_dir, ccw_dir, trim):
    input_dir, ccw_dir = Path(input_dir), Path(ccw_dir)
    paths = {
        "patients": input_dir / "ccw_patients.parquet",
        "monthly": input_dir / "ccw_monthly_covariates.parquet",
        "structure": ccw_dir / "raw" / "clone_month_structure.parquet",
        "landmarks": ccw_dir / "ccw_mortality_landmarks.csv",
        "covariate_spec": ccw_dir / "resolved_covariate_spec.json",
    }
    for name, path in paths.items():
        if name == "covariate_spec" and not path.is_file():
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    for scope in ("train", "val"):
        paths[f"weights_{scope}"] = ccw_dir / "raw" / f"{scope}_weights_{trim.lower()}.parquet"
        if not paths[f"weights_{scope}"].is_file():
            raise FileNotFoundError(paths[f"weights_{scope}"])

    patients = pd.read_parquet(paths["patients"])
    monthly = pd.read_parquet(paths["monthly"])
    structure = pd.read_parquet(paths["structure"])
    landmarks = pd.read_csv(paths["landmarks"])
    weights = pd.concat(
        [pd.read_parquet(paths["weights_train"]), pd.read_parquet(paths["weights_val"])],
        ignore_index=True,
    )
    if paths["covariate_spec"].is_file():
        covariate_spec = json.loads(paths["covariate_spec"].read_text(encoding="utf-8"))
    else:
        covariate_spec = {
            "baseline_numeric": list(BASELINE_NUMERIC),
            "baseline_categorical": list(BASELINE_CATEGORICAL),
            "time_varying_numeric": list(TIME_VARYING_NUMERIC),
        }
        paths.pop("covariate_spec")
    baseline_numeric = covariate_spec["baseline_numeric"]
    baseline_categorical = covariate_spec["baseline_categorical"]
    time_varying_numeric = covariate_spec["time_varying_numeric"]
    raw_time_varying = [
        value for value in time_varying_numeric if not value.startswith("lag_age_rcs_")
    ]
    raw_baseline = [
        value for value in baseline_numeric if not value.startswith("age_rcs_")
    ]
    require_columns(
        patients,
        [
            "patient_key", "split", "index_date", "first_adm_date", "death_date",
            "observation_end_date", *raw_baseline, *baseline_categorical,
        ],
        "patients",
    )
    # Age spline columns are created by the CCW runner and need not exist in
    # the raw input files used for this descriptive balance audit.
    require_columns(monthly, ["patient_key", "split", "month", *raw_time_varying], "monthly")
    require_columns(
        structure,
        [
            "patient_key", "split", "strategy", "month", "interval_start",
            "interval_stop", "stop_reason", "event_in_interval",
            "artificial_censor_in_interval",
        ],
        "clone structure",
    )
    require_columns(
        weights,
        [
            "patient_key", "strategy", "month", "weight", "weight_untrimmed",
            "unstabilized_weight_untrimmed",
        ],
        "weights",
    )
    require_columns(landmarks, ["scope", "trim", "strategy", "day", "risk"], "landmarks")

    patients = patients.copy()
    for column in ("index_date", "first_adm_date", "death_date", "observation_end_date"):
        patients[column] = pd.to_datetime(patients[column])
    for column in ("interval_start", "interval_stop"):
        structure[column] = pd.to_datetime(structure[column])
    if patients["patient_key"].duplicated().any():
        raise ValueError("patients contain duplicate patient_key")
    if structure[["patient_key", "strategy", "month"]].duplicated().any():
        raise ValueError("clone structure contains duplicate patient-strategy-month")
    if weights[["patient_key", "strategy", "month"]].duplicated().any():
        raise ValueError("weight files contain duplicate patient-strategy-month")

    patient_index = patients[["patient_key", "split", "index_date"]].rename(
        columns={"split": "patient_split"}
    )
    intervals = structure.merge(patient_index, on="patient_key", how="left", validate="many_to_one")
    if intervals["patient_split"].isna().any():
        raise ValueError("clone structure has patients missing from patient input")
    if not intervals["split"].astype(str).eq(intervals["patient_split"].astype(str)).all():
        raise ValueError("split mismatch between clone structure and patient input")
    intervals = intervals.drop(columns=["patient_split"])
    intervals = intervals.merge(
        weights,
        on=["patient_key", "strategy", "month"],
        how="left",
        validate="one_to_one",
    )
    if intervals[
        ["weight", "weight_untrimmed", "unstabilized_weight_untrimmed"]
    ].isna().any().any():
        raise ValueError("clone rows are missing outcome or balance weights")
    intervals["start_day"] = (intervals["interval_start"] - intervals["index_date"]).dt.days
    intervals["stop_day"] = (intervals["interval_stop"] - intervals["index_date"]).dt.days
    intervals["stop_date"] = intervals["interval_stop"]
    selected_landmarks = landmarks.loc[landmarks["trim"].astype(str).eq(str(trim))].copy()
    if len(selected_landmarks) != 2 * len(STRATEGIES) * len(LANDMARK_DAYS):
        raise ValueError("selected landmark grid is incomplete")
    balance_spec = {
        "baseline_numeric": raw_baseline,
        "baseline_categorical": baseline_categorical,
        "time_varying_numeric": raw_time_varying,
    }
    return patients, monthly, intervals, selected_landmarks, paths, balance_spec


def landmark_diagnostics(intervals, landmarks):
    rows = []
    risk_lookup = landmarks.set_index(["scope", "strategy", "day"])["risk"]
    for scope in ("train", "val"):
        for strategy in STRATEGIES:
            group = intervals.loc[
                intervals["split"].astype(str).eq(scope)
                & intervals["strategy"].astype(str).eq(strategy)
            ]
            for day in LANDMARK_DAYS:
                at_risk = group.loc[group["start_day"].lt(day) & group["stop_day"].ge(day)]
                deaths = group.loc[
                    group["event_in_interval"].eq(1) & group["stop_day"].le(day)
                ]
                artificial = group.loc[
                    group["artificial_censor_in_interval"].eq(1)
                    & group["stop_day"].le(day)
                ]
                rows.append(
                    {
                        "scope": scope,
                        "strategy": strategy,
                        "day": int(day),
                        "risk": float(risk_lookup.loc[(scope, strategy, day)]),
                        "at_risk_patients": int(at_risk["patient_key"].nunique()),
                        "weighted_risk_set": float(at_risk["weight"].sum()),
                        "cumulative_deaths": int(deaths["patient_key"].nunique()),
                        "weighted_cumulative_deaths": float(deaths["weight"].sum()),
                        "cumulative_artificial_censors": int(
                            artificial["patient_key"].nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def weighted_mean_variance(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[valid], weights[valid]
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    variance = float(np.average(np.square(values - mean), weights=weights))
    return mean, variance


def smd(reference_values, selected_values, selected_weights=None):
    ref = pd.to_numeric(pd.Series(reference_values), errors="coerce").to_numpy(float)
    sel = pd.to_numeric(pd.Series(selected_values), errors="coerce").to_numpy(float)
    ref = ref[np.isfinite(ref)]
    if selected_weights is None:
        weights = np.ones(len(sel), dtype=float)
    else:
        weights = np.asarray(selected_weights, dtype=float)
    sel_mean, sel_var = weighted_mean_variance(sel, weights)
    if len(ref) == 0 or not np.isfinite(sel_mean):
        return np.nan
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    pooled = np.sqrt(max(0.0, (ref_var + sel_var) / 2.0))
    if pooled < 1e-12:
        return 0.0 if abs(ref_mean - sel_mean) < 1e-12 else np.nan
    return float((sel_mean - ref_mean) / pooled)


def add_calendar_months(series, months):
    return series.map(lambda value: pd.Timestamp(value) + pd.DateOffset(months=int(months)))


def adherence_balance(
    patients,
    monthly,
    intervals,
    baseline_numeric=None,
    baseline_categorical=None,
    time_varying_numeric=None,
):
    baseline_numeric = list(baseline_numeric or BASELINE_NUMERIC)
    baseline_categorical = list(baseline_categorical or BASELINE_CATEGORICAL)
    time_varying_numeric = list(time_varying_numeric or TIME_VARYING_NUMERIC)
    balance_rows, support_rows = [], []
    final_month_weights = intervals[
        [
            "patient_key", "split", "strategy", "month",
            "unstabilized_weight_untrimmed",
        ]
    ].copy()
    for scope in ("train", "val"):
        scope_patients = patients.loc[patients["split"].astype(str).eq(scope)].copy()
        scope_monthly = monthly.loc[monthly["split"].astype(str).eq(scope)].copy()
        for strategy in STRATEGIES:
            deadline_months = STRATEGY_MONTHS[strategy]
            decision_month = deadline_months - 1
            work = scope_patients.copy()
            work["deadline_date"] = add_calendar_months(work["index_date"], deadline_months)
            work["eligible_at_deadline"] = (
                (work["death_date"].isna() | work["death_date"].gt(work["deadline_date"]))
                & work["observation_end_date"].ge(work["deadline_date"])
            )
            if strategy == "NO_INIT_WITHIN_12M":
                work["adherent"] = work["first_adm_date"].isna() | work[
                    "first_adm_date"
                ].gt(work["deadline_date"])
            else:
                work["adherent"] = work["first_adm_date"].notna() & work[
                    "first_adm_date"
                ].le(work["deadline_date"])
            reference = work.loc[work["eligible_at_deadline"]].copy()
            selected = reference.loc[reference["adherent"]].copy()
            month_values = scope_monthly.loc[scope_monthly["month"].eq(decision_month)]
            reference = reference.merge(
                month_values[["patient_key", *time_varying_numeric]],
                on="patient_key",
                how="left",
                validate="one_to_one",
            )
            selected = selected.merge(
                month_values[["patient_key", *time_varying_numeric]],
                on="patient_key",
                how="left",
                validate="one_to_one",
            )
            strategy_weights = final_month_weights.loc[
                final_month_weights["split"].astype(str).eq(scope)
                & final_month_weights["strategy"].astype(str).eq(strategy)
                & final_month_weights["month"].eq(decision_month),
                ["patient_key", "unstabilized_weight_untrimmed"],
            ]
            selected = selected.merge(
                strategy_weights.rename(
                    columns={"unstabilized_weight_untrimmed": "balance_weight"}
                ),
                on="patient_key",
                how="left",
                validate="one_to_one",
            )
            if selected["balance_weight"].isna().any():
                raise ValueError(
                    f"{scope} {strategy}: adherent patients lack decision-month weights"
                )
            support_rows.append(
                {
                    "scope": scope,
                    "strategy": strategy,
                    "deadline_months": deadline_months,
                    "eligible_at_deadline": int(len(reference)),
                    "adherent_at_deadline": int(len(selected)),
                    "adherent_fraction": float(len(selected) / len(reference)),
                    "unstabilized_weighted_adherent_ess": effective_sample_size(
                        selected["balance_weight"]
                    ),
                    "unstabilized_weight_min": float(selected["balance_weight"].min()),
                    "unstabilized_weight_p99": float(
                        np.quantile(selected["balance_weight"], 0.99)
                    ),
                    "unstabilized_weight_max": float(selected["balance_weight"].max()),
                }
            )
            for feature in [*baseline_numeric, *time_varying_numeric]:
                balance_rows.append(
                    {
                        "scope": scope,
                        "strategy": strategy,
                        "feature": feature,
                        "level": "",
                        "unweighted_smd": smd(reference[feature], selected[feature]),
                        "weighted_smd": smd(
                            reference[feature], selected[feature], selected["balance_weight"]
                        ),
                        "balance_weight": "unstabilized_ipcw_untrimmed",
                    }
                )
            for feature in baseline_categorical:
                levels = sorted(reference[feature].astype(str).fillna("__MISSING__").unique())
                for level in levels:
                    ref_indicator = reference[feature].astype(str).eq(level).astype(float)
                    sel_indicator = selected[feature].astype(str).eq(level).astype(float)
                    balance_rows.append(
                        {
                            "scope": scope,
                            "strategy": strategy,
                            "feature": feature,
                            "level": level,
                            "unweighted_smd": smd(ref_indicator, sel_indicator),
                            "weighted_smd": smd(
                                ref_indicator, sel_indicator, selected["balance_weight"]
                            ),
                            "balance_weight": "unstabilized_ipcw_untrimmed",
                        }
                    )
    return pd.DataFrame(balance_rows), pd.DataFrame(support_rows)


def bootstrap_fixed_weights(
    intervals,
    scope,
    samples,
    seed,
    checkpoint_path=None,
    checkpoint_every=25,
):
    scope_frame = intervals.loc[intervals["split"].astype(str).eq(str(scope))].copy()
    patient_ids = np.sort(scope_frame["patient_key"].unique())
    patient_index = pd.Series(np.arange(len(patient_ids)), index=patient_ids)
    row_patient_index = scope_frame["patient_key"].map(patient_index).to_numpy(dtype=int)
    work = scope_frame[
        [
            "strategy", "interval_start", "interval_stop", "index_date", "stop_date",
            "event_in_interval", "weight",
        ]
    ].copy()
    rng = np.random.default_rng(int(seed))
    draws = []
    for replicate in range(int(samples)):
        sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
        multiplicity = np.bincount(sampled, minlength=len(patient_ids))
        work["bootstrap_weight"] = (
            work["weight"].to_numpy(float) * multiplicity[row_patient_index]
        )
        landmarks = curve_landmarks(
            weighted_km(work, weight_column="bootstrap_weight"), LANDMARK_DAYS
        )
        landmarks.insert(0, "replicate", replicate)
        landmarks.insert(0, "scope", scope)
        draws.append(landmarks)
        completed = replicate + 1
        if checkpoint_path is not None and (
            completed % int(checkpoint_every) == 0 or completed == int(samples)
        ):
            atomic_csv(pd.concat(draws, ignore_index=True), checkpoint_path)
            log(f"[BOOTSTRAP] completed={completed}/{samples}")
    return pd.concat(draws, ignore_index=True)


def patient_bootstrap_multiplicity(patient_ids, rng):
    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
    counts = np.bincount(sampled, minlength=len(patient_ids)).astype(float)
    return pd.Series(counts, index=patient_ids)


def bootstrap_full_refit(
    input_dir,
    samples,
    seed,
    checkpoint_path=None,
    checkpoint_every=10,
    ridge=1e-2,
    trim_quantiles=(0.0, 0.99),
    existing_draws=None,
):
    """Paired split bootstrap with censor-model refitting in every draw."""
    patients, monthly, _, _, spec, _ = load_runner_inputs(input_dir)
    baseline_numeric = spec["baseline_numeric"]
    baseline_categorical = spec["baseline_categorical"]
    time_varying_numeric = spec["time_varying_numeric"]
    intervals = attach_monthly_covariates(
        build_month_intervals(clone_patients(patients)),
        monthly,
        time_varying_numeric,
    )
    intervals, followup_numeric, _ = add_followup_time_spline(intervals)
    train = intervals.loc[intervals["split"].astype(str).eq("train")].copy()
    validation = intervals.loc[intervals["split"].astype(str).eq("val")].copy()
    train_model = train.loc[train["month"].lt(12)].copy()
    validation_model = validation.loc[validation["month"].lt(12)].copy()
    train_ids = np.sort(train["patient_key"].unique())
    validation_ids = np.sort(validation["patient_key"].unique())
    if len(train_ids) == 0 or len(validation_ids) == 0:
        raise ValueError("full bootstrap requires nonempty train and val patients")
    rng = np.random.default_rng(int(seed))
    draws = [] if existing_draws is None else [existing_draws.copy()]
    completed_replicates = (
        set()
        if existing_draws is None
        else set(pd.to_numeric(existing_draws["replicate"], errors="raise").astype(int))
    )
    for replicate in range(int(samples)):
        train_multiplier = patient_bootstrap_multiplicity(train_ids, rng)
        validation_multiplier = patient_bootstrap_multiplicity(validation_ids, rng)
        if replicate in completed_replicates:
            continue
        train_work = train_model.copy()
        validation_work = validation_model.copy()
        train_work["bootstrap_multiplicity"] = (
            train_work["patient_key"].map(train_multiplier).fillna(0.0)
        )
        validation_work["bootstrap_multiplicity"] = (
            validation_work["patient_key"].map(validation_multiplier).fillna(0.0)
        )
        models = fit_censor_models(
            train_work,
            baseline_numeric=baseline_numeric,
            baseline_categorical=baseline_categorical,
            time_varying_numeric=time_varying_numeric,
            ridge=ridge,
            sample_weight_column="bootstrap_multiplicity",
            followup_numeric=followup_numeric,
        )
        if not all(
            value["numerator"].converged and value["denominator"].converged
            for value in models.values()
        ):
            raise RuntimeError(f"bootstrap replicate {replicate}: censor model did not converge")
        _, train_caps = apply_censor_weights(
            train_work,
            models,
            baseline_numeric,
            baseline_categorical,
            time_varying_numeric,
            trim_quantiles=trim_quantiles,
            trim_quantile_weight_column="bootstrap_multiplicity",
        )
        validation_first_year, _ = apply_censor_weights(
            validation_work,
            models,
            baseline_numeric,
            baseline_categorical,
            time_varying_numeric,
            trim_quantiles=trim_quantiles,
            trim_caps=train_caps,
        )
        weight_lookup = validation_first_year[
            ["patient_key", "strategy", "month", "weight"]
        ]
        validation_weighted = validation.merge(
            weight_lookup,
            on=["patient_key", "strategy", "month"],
            how="left",
            validate="one_to_one",
        ).sort_values(["patient_key", "strategy", "month"])
        validation_weighted["weight"] = validation_weighted.groupby(
            ["patient_key", "strategy"], sort=False
        )["weight"].ffill()
        if validation_weighted["weight"].isna().any():
            raise RuntimeError(f"bootstrap replicate {replicate}: missing carried weight")
        validation_weighted["bootstrap_multiplicity"] = (
            validation_weighted["patient_key"].map(validation_multiplier).fillna(0.0)
        )
        validation_weighted["bootstrap_weight"] = (
            validation_weighted["weight"]
            * validation_weighted["bootstrap_multiplicity"]
        )
        landmarks = curve_landmarks(
            weighted_km(validation_weighted, weight_column="bootstrap_weight"),
            LANDMARK_DAYS,
        )
        landmarks.insert(0, "replicate", replicate)
        landmarks.insert(0, "scope", "val")
        draws.append(landmarks)
        completed = replicate + 1
        if checkpoint_path is not None and (
            completed % int(checkpoint_every) == 0 or completed == int(samples)
        ):
            atomic_csv(pd.concat(draws, ignore_index=True), checkpoint_path)
            log(f"[FULL REFIT BOOTSTRAP] completed={completed}/{samples}")
    return pd.concat(draws, ignore_index=True).sort_values(
        ["replicate", "strategy", "day"]
    ).reset_index(drop=True)


def summarize_bootstrap(draws, observed_landmarks, scope):
    observed = observed_landmarks.loc[observed_landmarks["scope"].astype(str).eq(str(scope))]
    observed_lookup = observed.set_index(["strategy", "day"])["risk"]
    risk_rows = []
    for (strategy, day), group in draws.groupby(["strategy", "day"], sort=False):
        values = group["risk"].to_numpy(float)
        risk_rows.append(
            {
                "scope": scope,
                "strategy": strategy,
                "day": int(day),
                "observed_risk": float(observed_lookup.loc[(strategy, day)]),
                "bootstrap_median": float(np.quantile(values, 0.5)),
                "ci_lower_2_5": float(np.quantile(values, 0.025)),
                "ci_upper_97_5": float(np.quantile(values, 0.975)),
            }
        )
    pivot = draws.pivot(index=["replicate", "day"], columns="strategy", values="risk")
    contrast_rows = []
    for strategy in STRATEGIES:
        if strategy == "NO_INIT_WITHIN_12M":
            continue
        difference = pivot[strategy] - pivot["NO_INIT_WITHIN_12M"]
        for day, values in difference.groupby(level="day"):
            array = values.to_numpy(float)
            observed_difference = float(
                observed_lookup.loc[(strategy, day)]
                - observed_lookup.loc[("NO_INIT_WITHIN_12M", day)]
            )
            contrast_rows.append(
                {
                    "scope": scope,
                    "strategy": strategy,
                    "reference": "NO_INIT_WITHIN_12M",
                    "day": int(day),
                    "observed_risk_difference": observed_difference,
                    "bootstrap_median": float(np.quantile(array, 0.5)),
                    "ci_lower_2_5": float(np.quantile(array, 0.025)),
                    "ci_upper_97_5": float(np.quantile(array, 0.975)),
                }
            )
    return pd.DataFrame(risk_rows), pd.DataFrame(contrast_rows)


def synthetic_inputs(n=80):
    rng = np.random.default_rng(11)
    n = int(n)
    index = pd.Timestamp("2018-01-01")
    split = np.where(np.arange(n) < int(round(n * 0.75)), "train", "val")
    first_days = np.where(
        np.arange(n) % 4 == 0,
        45,
        np.where(np.arange(n) % 4 == 1, 140, np.where(np.arange(n) % 4 == 2, 280, -1)),
    )
    first_adm = [
        index + pd.Timedelta(days=int(day)) if day >= 0 else pd.NaT for day in first_days
    ]
    death_days = np.where(
        np.arange(n) % 13 == 0, 300 + (np.arange(n) % 10) * 100, -1
    )
    death = [
        index + pd.Timedelta(days=int(day)) if day >= 0 else pd.NaT for day in death_days
    ]
    patients = pd.DataFrame(
        {
            "patient_key": np.arange(n),
            "split": split,
            "index_date": [index] * n,
            "first_adm_date": first_adm,
            "death_date": death,
            "observation_end_date": [index + pd.Timedelta(days=1826)] * n,
            "age": rng.integers(30, 85, n),
            "index_year": [2018] * n,
            "index_hba1c": rng.normal(7.2, 0.6, n),
            "visit_rows_1y": rng.integers(1, 20, n),
            "visit_days_1y": rng.integers(1, 10, n),
            "condition_rows_1y": rng.integers(0, 20, n),
            "condition_days_1y": rng.integers(0, 10, n),
            "nonadm_drug_rows_1y": rng.integers(0, 20, n),
            "nonadm_drug_days_1y": rng.integers(0, 10, n),
            "inpatient_visits_1y": rng.integers(0, 3, n),
            "ambulatory_ed_visits_1y": rng.integers(0, 10, n),
            "gender_concept_id": np.where(np.arange(n) % 2, "8507", "8532"),
        }
    )
    monthly_rows = []
    for patient in patients.itertuples(index=False):
        for month in range(12):
            monthly_rows.append(
                {
                    "patient_key": patient.patient_key,
                    "split": patient.split,
                    "month": month,
                    "lag_hba1c": patient.index_hba1c,
                    "lag_age": patient.age + month / 12.0,
                    "lag_calendar_year": 2018 + month / 12.0,
                    "lag_visit_rows_30d": month % 4,
                    "lag_visit_days_30d": month % 3,
                    "lag_condition_rows_30d": month % 5,
                    "lag_condition_days_30d": month % 2,
                    "lag_nonadm_drug_rows_30d": month % 6,
                    "lag_nonadm_drug_days_30d": month % 4,
                    "lag_inpatient_visits_30d": month % 2,
                    "lag_ambulatory_ed_visits_30d": month % 3,
                }
            )
    monthly = pd.DataFrame(monthly_rows)
    from snuh_task30_adm_ccw_core import build_month_intervals, clone_patients  # noqa: E402

    structure = build_month_intervals(clone_patients(patients))
    structure["weight"] = 1.0
    structure["weight_untrimmed"] = 1.0
    structure["unstabilized_weight_untrimmed"] = 1.0
    structure["start_day"] = (structure["interval_start"] - structure["index_date"]).dt.days
    structure["stop_day"] = (structure["interval_stop"] - structure["index_date"]).dt.days
    structure["stop_date"] = structure["stop_date"]
    landmarks = []
    for scope in ("train", "val"):
        curve = curve_landmarks(
            weighted_km(structure.loc[structure["split"].eq(scope)]), LANDMARK_DAYS
        )
        curve.insert(0, "scope", scope)
        curve.insert(1, "trim", "P99")
        landmarks.append(curve)
    return patients, monthly, structure, pd.concat(landmarks, ignore_index=True)


def self_test():
    patients, monthly, intervals, landmarks = synthetic_inputs()
    diagnostics = landmark_diagnostics(intervals, landmarks)
    balance, support = adherence_balance(patients, monthly, intervals)
    draws = bootstrap_fixed_weights(intervals, "val", 12, 7)
    risk_ci, contrast_ci = summarize_bootstrap(draws, landmarks, "val")
    assert len(diagnostics) == 24
    assert len(support) == 8
    assert not balance.empty
    assert len(draws) == 12 * len(STRATEGIES) * len(LANDMARK_DAYS)
    assert len(risk_ci) == len(STRATEGIES) * len(LANDMARK_DAYS)
    assert len(contrast_ci) == 3 * len(LANDMARK_DAYS)
    assert risk_ci["ci_upper_97_5"].max() > 0
    with TemporaryDirectory() as directory:
        input_dir = Path(directory)
        patients.to_parquet(input_dir / "ccw_patients.parquet", index=False)
        monthly.to_parquet(input_dir / "ccw_monthly_covariates.parquet", index=False)
        spec = {
            "baseline_numeric": list(BASELINE_NUMERIC),
            "baseline_categorical": list(BASELINE_CATEGORICAL),
            "time_varying_numeric": [
                "lag_age",
                "lag_calendar_year",
                *TIME_VARYING_NUMERIC,
            ],
            "treatment_exposure_excluded_from_drug_utilization": True,
        }
        (input_dir / "covariate_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        full_draws = bootstrap_full_refit(
            input_dir, samples=3, seed=19, ridge=0.1, checkpoint_every=1
        )
        assert len(full_draws) == 3 * len(STRATEGIES) * len(LANDMARK_DAYS)
    print(
        "SELF_TEST_OK diagnostics=24 support=8 bootstrap_draws="
        f"{len(draws)} full_refit_draws={len(full_draws)} "
        f"risk_ci={len(risk_ci)} contrasts={len(contrast_ci)}"
    )


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.bootstrap_samples < 50:
        raise ValueError("bootstrap-samples must be at least 50 for a reported interval")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.resume_bootstrap:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        prepare_output(args.output_dir, args.overwrite)
    log("[START] load and validate completed CCW artifacts")
    patients, monthly, intervals, landmarks, paths, balance_spec = load_inputs(
        args.input_dir, args.ccw_dir, args.trim
    )
    log(
        f"[DONE] patients={len(patients):,} clone_months={len(intervals):,} "
        f"trim={args.trim}"
    )

    diagnostics = landmark_diagnostics(intervals, landmarks)
    atomic_csv(diagnostics, args.output_dir / "landmark_risk_set_event_diagnostics.csv")
    balance, support = adherence_balance(
        patients,
        monthly,
        intervals,
        balance_spec["baseline_numeric"],
        balance_spec["baseline_categorical"],
        balance_spec["time_varying_numeric"],
    )
    atomic_csv(support, args.output_dir / "strategy_adherence_support.csv")
    atomic_csv(balance, args.output_dir / "adherence_selection_balance.csv")

    balance_ranked = balance.assign(
        abs_unweighted=balance["unweighted_smd"].abs(),
        abs_weighted=balance["weighted_smd"].abs(),
    )
    balance_max = (
        balance_ranked.groupby(["scope", "strategy"], as_index=False)[
            ["abs_unweighted", "abs_weighted"]
        ]
        .max()
    )
    if args.balance_only:
        source_hashes = {name: file_sha256(path) for name, path in paths.items()}
        manifest = {
            "status": "CCW_UNSTABILIZED_IPCW_BALANCE_AUDIT_COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(args.input_dir),
            "ccw_dir": str(args.ccw_dir),
            "trim": args.trim,
            "source_sha256": source_hashes,
            "database_queried": False,
            "bootstrap_rerun": False,
            "outcome_weight": (
                "stabilized IPCW remains unchanged for the mortality curves"
            ),
            "balance_weight": (
                "untrimmed denominator-only IPCW: 1/P(adherence|baseline,time-varying)"
            ),
            "claim_boundary": (
                "This checks measured-covariate selection balance at each strategy "
                "deadline; it does not prove exchangeability or eliminate unmeasured "
                "confounding."
            ),
        }
        write_json(manifest, args.output_dir / "manifest.json")
        worst = balance_ranked.sort_values(
            ["scope", "strategy", "abs_weighted"],
            ascending=[True, True, False],
        ).groupby(["scope", "strategy"], as_index=False).head(5)
        return_summary = [
            "## STATUS",
            manifest["status"],
            "## STRATEGY_SUPPORT_AND_UNSTABILIZED_WEIGHT_DIAGNOSTICS",
            support.to_csv(index=False).rstrip(),
            "## BALANCE_MAX_ABS_SMD_UNSTABILIZED_IPCW",
            balance_max.to_csv(index=False).rstrip(),
            "## FIVE_WORST_WEIGHTED_SMD_PER_SCOPE_STRATEGY",
            worst[
                [
                    "scope", "strategy", "feature", "level", "unweighted_smd",
                    "weighted_smd", "balance_weight",
                ]
            ].to_csv(index=False).rstrip(),
            "## CLAIM_BOUNDARY",
            manifest["claim_boundary"],
        ]
        summary_path = args.output_dir / "return_summary.txt"
        summary_path.write_text("\n".join(return_summary) + "\n", encoding="utf-8")
        print(summary_path.read_text(encoding="utf-8"), end="", flush=True)
        log(f"[COMPLETE] {args.output_dir}")
        return 0

    if args.bootstrap_mode == "full_refit":
        if args.bootstrap_scope != "val":
            raise ValueError("full_refit bootstrap evaluates val; use --bootstrap-scope val")
        draws_path = args.output_dir / "bootstrap_full_refit_draws.csv"
        existing_draws = None
        if args.resume_bootstrap and draws_path.is_file():
            existing_draws = pd.read_csv(draws_path)
            log(
                f"[RESUME] full-refit bootstrap replicates="
                f"{existing_draws['replicate'].nunique():,}"
            )
        draws = bootstrap_full_refit(
            args.input_dir,
            args.bootstrap_samples,
            args.bootstrap_seed,
            checkpoint_path=draws_path,
            checkpoint_every=args.checkpoint_every,
            ridge=args.ridge,
            existing_draws=existing_draws,
        )
    else:
        draws_path = args.output_dir / "bootstrap_fixed_weight_draws.csv"
        draws = bootstrap_fixed_weights(
            intervals,
            args.bootstrap_scope,
            args.bootstrap_samples,
            args.bootstrap_seed,
            checkpoint_path=draws_path,
            checkpoint_every=args.checkpoint_every,
        )
    risk_ci, contrast_ci = summarize_bootstrap(draws, landmarks, args.bootstrap_scope)
    atomic_csv(risk_ci, args.output_dir / "bootstrap_risk_ci.csv")
    atomic_csv(contrast_ci, args.output_dir / "bootstrap_contrast_ci.csv")

    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    manifest = {
        "status": "CCW_RESULT_AUDIT_COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "ccw_dir": str(args.ccw_dir),
        "trim": args.trim,
        "bootstrap_scope": args.bootstrap_scope,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_mode": args.bootstrap_mode,
        "censor_model_ridge": float(args.ridge),
        "source_sha256": source_hashes,
        "database_queried": False,
        "ccw_refit": args.bootstrap_mode == "full_refit",
        "bootstrap_claim_boundary": (
            "Train and validation patients are resampled separately; censoring models "
            "and train-derived P99 caps are refit in every draw."
            if args.bootstrap_mode == "full_refit"
            else "Paired patient bootstrap with P99 censoring weights held fixed; "
            "does not include censor-model estimation uncertainty."
        ),
        "balance_claim_boundary": (
            "Measured-covariate adherence-selection SMD using untrimmed, "
            "denominator-only IPCW at each grace-period end; not proof of "
            "exchangeability or absence of residual confounding."
        ),
    }
    write_json(manifest, args.output_dir / "manifest.json")
    return_summary = [
        "## STATUS",
        manifest["status"],
        "## LANDMARK_RISK_SET_EVENTS",
        diagnostics.to_csv(index=False).rstrip(),
        "## STRATEGY_SUPPORT",
        support.to_csv(index=False).rstrip(),
        f"## {args.bootstrap_mode.upper()}_RISK_CI",
        risk_ci.to_csv(index=False).rstrip(),
        f"## {args.bootstrap_mode.upper()}_CONTRAST_CI",
        contrast_ci.to_csv(index=False).rstrip(),
        "## BALANCE_MAX_ABS_SMD_UNSTABILIZED_IPCW",
        balance_max.to_csv(index=False).rstrip(),
        "## CLAIM_BOUNDARY",
        manifest["bootstrap_claim_boundary"],
        manifest["balance_claim_boundary"],
    ]
    summary_path = args.output_dir / "return_summary.txt"
    summary_path.write_text("\n".join(return_summary) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"), end="", flush=True)
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
