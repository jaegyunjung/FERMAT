#!/usr/bin/env python3
"""Core clone-censor-weight utilities for the Task 30 ADM-timing study.

The functions in this module do not connect to SNUH CDM and do not import
FERMAT.  They turn an already-eligible patient table into four treatment
strategy clones, estimate stabilized inverse-probability-of-artificial-
censoring weights, and calculate weighted Kaplan-Meier risk curves.

Time zero is the first observed HbA1c >=6.5%.  The four strategies are:

* initiate ADM within 3 calendar months;
* initiate ADM within 6 calendar months;
* initiate ADM within 12 calendar months;
* do not initiate ADM within 12 calendar months.

The module is deliberately data-source agnostic so its state transitions can
be self-tested locally before a Pod/database run.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


STRATEGIES = (
    "INIT_WITHIN_3M",
    "INIT_WITHIN_6M",
    "INIT_WITHIN_12M",
    "NO_INIT_WITHIN_12M",
)
INIT_DEADLINE_MONTHS = {
    "INIT_WITHIN_3M": 3,
    "INIT_WITHIN_6M": 6,
    "INIT_WITHIN_12M": 12,
}
HORIZON_DAYS = 1826
LANDMARK_DAYS = (365, 1095, 1826)


@dataclass(frozen=True)
class LogisticFit:
    coefficients: np.ndarray
    converged: bool
    iterations: int
    ridge: float
    max_abs_step: float
    max_abs_gradient: float
    penalized_log_likelihood: float
    line_search_reductions: int


def expit(value):
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def fit_logistic_irls(
    x,
    y,
    ridge=1e-4,
    max_iter=300,
    tol=1e-7,
    objective_tol=1e-10,
    sample_weight=None,
):
    """Fit a ridge logistic model by damped Newton iterations.

    The earlier implementation used undamped IRLS and required an absolute
    coefficient change below 1e-8 within 100 iterations.  That is needlessly
    brittle for sparse artificial-censoring outcomes.  This implementation
    maximizes the penalized log likelihood, halves a Newton step when needed,
    and records enough diagnostics to distinguish slow convergence from a
    genuinely unstable fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("x must be 2-D and have the same rows as y")
    if len(np.unique(y)) < 2:
        raise ValueError("logistic outcome has only one class")
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.shape != y.shape:
        raise ValueError("sample_weight must have one value per row")
    if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
        raise ValueError("sample_weight must be finite and nonnegative")
    if sample_weight.sum() <= 0:
        raise ValueError("sample_weight has zero total weight")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    if float(ridge) <= 0:
        raise ValueError("ridge must be positive")

    beta = np.zeros(x.shape[1], dtype=np.float64)
    weighted_rate = np.clip(np.average(y, weights=sample_weight), 1e-6, 1.0 - 1e-6)
    beta[0] = np.log(weighted_rate / (1.0 - weighted_rate))
    penalty_diagonal = np.full(x.shape[1], float(ridge), dtype=np.float64)
    penalty_diagonal[0] = 0.0

    def objective(coefficients):
        linear = x @ coefficients
        likelihood = np.sum(sample_weight * (y * linear - np.logaddexp(0.0, linear)))
        penalty_value = 0.5 * np.sum(penalty_diagonal * np.square(coefficients))
        return float(likelihood - penalty_value)

    converged = False
    current_objective = objective(beta)
    max_abs_step = np.inf
    max_abs_gradient = np.inf
    total_line_search_reductions = 0
    for iteration in range(1, int(max_iter) + 1):
        probability = np.clip(expit(x @ beta), 1e-9, 1.0 - 1e-9)
        variance = probability * (1.0 - probability) * sample_weight
        gradient = x.T @ (sample_weight * (y - probability)) - penalty_diagonal * beta
        hessian = (x.T * variance) @ x
        hessian.flat[:: hessian.shape[0] + 1] += penalty_diagonal
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.isfinite(step).all():
            break

        step_scale = 1.0
        reductions = 0
        candidate = beta + step
        candidate_objective = objective(candidate)
        while (
            (not np.isfinite(candidate_objective) or candidate_objective < current_objective)
            and reductions < 30
        ):
            step_scale *= 0.5
            reductions += 1
            candidate = beta + step_scale * step
            candidate_objective = objective(candidate)
        total_line_search_reductions += reductions
        if not np.isfinite(candidate_objective) or candidate_objective < current_objective:
            break

        applied_step = candidate - beta
        max_abs_step = float(np.max(np.abs(applied_step)))
        max_abs_gradient = float(np.max(np.abs(gradient)))
        relative_objective_change = abs(candidate_objective - current_objective) / (
            1.0 + abs(current_objective)
        )
        beta = candidate
        current_objective = candidate_objective
        coefficient_scale = 1.0 + float(np.max(np.abs(beta)))
        if max_abs_step <= float(tol) * coefficient_scale:
            converged = True
            break
        if relative_objective_change <= float(objective_tol) and max_abs_gradient <= 1e-5:
            converged = True
            break
    return LogisticFit(
        beta,
        converged,
        iteration,
        float(ridge),
        max_abs_step,
        max_abs_gradient,
        current_objective,
        total_line_search_reductions,
    )


def predict_logistic(fit, x):
    return np.clip(expit(np.asarray(x, float) @ fit.coefficients), 1e-6, 1 - 1e-6)


def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if sample_weight is None:
        return np.quantile(values, quantiles)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values, sample_weight = values[valid], sample_weight[valid]
    if len(values) == 0:
        raise ValueError("weighted quantile has no positive-weight observations")
    order = np.argsort(values, kind="mergesort")
    values, sample_weight = values[order], sample_weight[order]
    positions = (np.cumsum(sample_weight) - 0.5 * sample_weight) / sample_weight.sum()
    return np.interp(quantiles, positions, values, left=values[0], right=values[-1])


def add_calendar_month(date_value, months):
    return pd.Timestamp(date_value) + pd.DateOffset(months=int(months))


def strategy_artificial_censor_date(index_date, first_adm_date, strategy):
    """Return the first date on which observed care violates a strategy."""
    index_date = pd.Timestamp(index_date)
    first_adm = pd.NaT if pd.isna(first_adm_date) else pd.Timestamp(first_adm_date)
    if strategy in INIT_DEADLINE_MONTHS:
        deadline = add_calendar_month(index_date, INIT_DEADLINE_MONTHS[strategy])
        if pd.isna(first_adm) or first_adm > deadline:
            return deadline
        return pd.NaT
    if strategy == "NO_INIT_WITHIN_12M":
        deadline = add_calendar_month(index_date, 12)
        if not pd.isna(first_adm) and first_adm <= deadline:
            return first_adm
        return pd.NaT
    raise ValueError(f"Unknown strategy: {strategy}")


def clone_patients(patients, horizon_days=HORIZON_DAYS):
    """Create one survival row for every patient-strategy combination.

    Required patient columns are patient_key, index_date, first_adm_date,
    death_date, and observation_end_date.  Death is the current primary
    outcome.  Artificial censoring is applied only when it precedes death and
    administrative censoring.
    """
    required = {
        "patient_key",
        "index_date",
        "first_adm_date",
        "death_date",
        "observation_end_date",
    }
    missing = sorted(required - set(patients.columns))
    if missing:
        raise ValueError(f"patient table missing columns: {missing}")
    rows = []
    for patient in patients.itertuples(index=False):
        index_date = pd.Timestamp(patient.index_date)
        horizon_date = index_date + pd.Timedelta(days=int(horizon_days))
        death_date = pd.NaT if pd.isna(patient.death_date) else pd.Timestamp(patient.death_date)
        observation_end = pd.Timestamp(patient.observation_end_date)
        natural_end = min(observation_end, horizon_date)
        for strategy in STRATEGIES:
            artificial = strategy_artificial_censor_date(
                index_date, patient.first_adm_date, strategy
            )
            candidates = [(natural_end, "ADMIN")]
            if not pd.isna(artificial):
                candidates.append((pd.Timestamp(artificial), "ARTIFICIAL"))
            if not pd.isna(death_date):
                candidates.append((death_date, "DEATH"))
            # Event wins an exact-date tie; this is explicit and self-tested.
            priority = {"DEATH": 0, "ARTIFICIAL": 1, "ADMIN": 2}
            stop_date, reason = min(candidates, key=lambda item: (item[0], priority[item[1]]))
            if stop_date < index_date:
                raise ValueError(f"negative follow-up for patient {patient.patient_key}")
            base = patient._asdict()
            base.update(
                {
                    "strategy": strategy,
                    "artificial_censor_date": artificial,
                    "stop_date": stop_date,
                    "stop_day": min((stop_date - index_date).days, int(horizon_days)),
                    "stop_reason": reason,
                    "death_event": int(reason == "DEATH" and stop_date <= horizon_date),
                    "artificial_censor": int(reason == "ARTIFICIAL"),
                }
            )
            rows.append(base)
    clones = pd.DataFrame(rows)
    if len(clones) != len(patients) * len(STRATEGIES):
        raise AssertionError("clone count mismatch")
    return clones


def build_month_intervals(clones, max_months=60):
    """Expand clone survival rows to calendar-month counting-process rows."""
    rows = []
    for clone in clones.itertuples(index=False):
        index_date = pd.Timestamp(clone.index_date)
        stop_date = pd.Timestamp(clone.stop_date)
        for month in range(int(max_months)):
            start = add_calendar_month(index_date, month)
            if start >= stop_date and month > 0:
                break
            nominal_stop = add_calendar_month(index_date, month + 1)
            interval_stop = min(nominal_stop, stop_date)
            if interval_stop < start:
                break
            row = clone._asdict()
            row.update(
                {
                    "month": month,
                    "interval_start": start,
                    "interval_stop": interval_stop,
                    "event_in_interval": int(
                        clone.death_event == 1 and stop_date <= nominal_stop
                    ),
                    "artificial_censor_in_interval": int(
                        clone.artificial_censor == 1 and stop_date <= nominal_stop
                    ),
                }
            )
            rows.append(row)
            if interval_stop >= stop_date:
                break
    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise ValueError("no clone-month intervals were created")
    return intervals


def design_matrix(frame, numeric, categorical, reference=None):
    """Build a stable matrix; reuse `reference` columns/standardization on val."""
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(f"covariates cannot be both numeric and categorical: {overlap}")
    parts = {}
    if reference is None:
        means = {}
        scales = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            mean = float(values.mean()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().any() else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            means[column], scales[column] = mean, scale
            parts[column] = values.fillna(mean).sub(mean).div(scale).to_numpy(float)
            parts[f"{column}__missing"] = values.isna().to_numpy(dtype=float)
        for column in categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            dummies = pd.get_dummies(values, prefix=column, dtype=float)
            # Drop one reference level. Keeping every level together with an
            # intercept creates an exact linear dependency.
            for name in sorted(dummies.columns)[1:]:
                parts[name] = dummies[name].to_numpy(float)
        work = pd.DataFrame(parts, index=frame.index)
        columns = ["intercept"] + work.columns.tolist()
        reference = {"means": means, "scales": scales, "columns": columns}
    else:
        means = reference["means"]
        scales = reference["scales"]
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            parts[column] = (
                values.fillna(means[column]).sub(means[column]).div(scales[column]).to_numpy(float)
            )
            parts[f"{column}__missing"] = values.isna().to_numpy(dtype=float)
        for column in categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            dummies = pd.get_dummies(values, prefix=column, dtype=float)
            for name in dummies.columns:
                parts[name] = dummies[name].to_numpy(float)
        work = pd.DataFrame(parts, index=frame.index)
        expected = [name for name in reference["columns"] if name != "intercept"]
        work = work.reindex(columns=expected, fill_value=0.0)
        columns = reference["columns"]
    matrix = np.column_stack([np.ones(len(work)), work.to_numpy(dtype=np.float64)])
    if matrix.shape[1] != len(columns):
        raise AssertionError("design matrix column mismatch")
    return matrix, reference


def _censor_model_rows(intervals, strategy):
    """Return the paper-specified artificial-censoring risk set.

    Initiation strategies receive IPCW once, at their grace-period endpoint.
    The no-initiation strategy can deviate in any of the first 12 months and
    therefore receives interval-specific cumulative IPCW.
    """
    group = intervals.loc[intervals["strategy"].eq(strategy)].copy()
    if strategy in INIT_DEADLINE_MONTHS:
        deadline_month = INIT_DEADLINE_MONTHS[strategy] - 1
        return group.loc[group["month"].eq(deadline_month)].copy()
    return group.loc[group["month"].lt(12)].copy()


def fit_censor_models(
    train_intervals,
    baseline_numeric,
    baseline_categorical,
    time_varying_numeric,
    ridge=1e-4,
    sample_weight_column=None,
    followup_numeric=None,
):
    """Fit numerator and denominator models separately for each strategy."""
    models = {}
    followup_numeric = list(followup_numeric or ["month"])
    for strategy in STRATEGIES:
        rows = _censor_model_rows(train_intervals, strategy)
        y = 1 - rows["artificial_censor_in_interval"].to_numpy(dtype=np.int8)
        sample_weight = (
            None
            if sample_weight_column is None
            else rows[sample_weight_column].to_numpy(dtype=np.float64)
        )
        numerator_x, numerator_reference = design_matrix(
            rows, baseline_numeric + followup_numeric, baseline_categorical
        )
        denominator_x, denominator_reference = design_matrix(
            rows,
            baseline_numeric + followup_numeric + list(time_varying_numeric),
            baseline_categorical,
        )
        models[strategy] = {
            "numerator": fit_logistic_irls(
                numerator_x, y, ridge=ridge, sample_weight=sample_weight
            ),
            "denominator": fit_logistic_irls(
                denominator_x, y, ridge=ridge, sample_weight=sample_weight
            ),
            "numerator_reference": numerator_reference,
            "denominator_reference": denominator_reference,
            "rows": int(len(rows)),
            "remained": int(y.sum()),
            "censored": int((1 - y).sum()),
            "weighted_rows": float(len(rows) if sample_weight is None else sample_weight.sum()),
            "followup_numeric": followup_numeric,
            "numerator_predictors": int(numerator_x.shape[1] - 1),
            "denominator_predictors": int(denominator_x.shape[1] - 1),
        }
    return models


def apply_censor_weights(
    intervals,
    models,
    baseline_numeric,
    baseline_categorical,
    time_varying_numeric,
    trim_quantiles=(0.0, 0.99),
    trim_caps=None,
    trim_quantile_weight_column=None,
):
    """Attach stabilized time-varying weights to clone-month rows."""
    output = intervals.copy()
    output["weight_untrimmed"] = 1.0
    # Keep a second, denominator-only IPCW specifically for diagnostics.
    # The stabilized weight above is used for the outcome curve.  It is not
    # appropriate for checking whether censoring selection has been removed
    # against the original eligible population because its numerator retains
    # baseline-dependent selection.
    output["unstabilized_weight_untrimmed"] = 1.0
    for strategy in STRATEGIES:
        model = models[strategy]
        rows = _censor_model_rows(output, strategy)
        numerator_x, _ = design_matrix(
            rows,
            baseline_numeric + model.get("followup_numeric", ["month"]),
            baseline_categorical,
            model["numerator_reference"],
        )
        denominator_x, _ = design_matrix(
            rows,
            baseline_numeric
            + model.get("followup_numeric", ["month"])
            + list(time_varying_numeric),
            baseline_categorical,
            model["denominator_reference"],
        )
        p_num = predict_logistic(model["numerator"], numerator_x)
        p_den = predict_logistic(model["denominator"], denominator_x)
        remained = 1 - rows["artificial_censor_in_interval"].to_numpy(dtype=np.int8)
        # The inverse probability factor belongs to clones that remain under
        # follow-up. A clone censored during this interval contributes with its
        # prior weight up to the censoring time and receives no new factor.
        increment = np.where(remained == 1, p_num / p_den, 1.0)
        unstabilized_increment = np.where(remained == 1, 1.0 / p_den, 1.0)
        prediction = rows[["patient_key", "month"]].copy()
        prediction["increment"] = increment
        prediction["unstabilized_increment"] = unstabilized_increment
        strategy_index = output["strategy"].eq(strategy)
        current = output.loc[strategy_index].copy()
        current["_row_order"] = np.arange(len(current))
        current = current.merge(
            prediction, on=["patient_key", "month"], how="left", validate="one_to_one"
        ).sort_values("_row_order")
        current["increment"] = current["increment"].fillna(1.0)
        current["unstabilized_increment"] = current[
            "unstabilized_increment"
        ].fillna(1.0)
        current["weight_untrimmed"] = current.groupby("patient_key")["increment"].cumprod()
        current["unstabilized_weight_untrimmed"] = current.groupby("patient_key")[
            "unstabilized_increment"
        ].cumprod()
        output.loc[strategy_index, "weight_untrimmed"] = current["weight_untrimmed"].to_numpy()
        output.loc[strategy_index, "unstabilized_weight_untrimmed"] = current[
            "unstabilized_weight_untrimmed"
        ].to_numpy()
    lo, hi = map(float, trim_quantiles)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("trim_quantiles must satisfy 0 <= low < high <= 1")
    output["weight"] = output["weight_untrimmed"]
    caps = {}
    for strategy in STRATEGIES:
        mask = output["strategy"].eq(strategy)
        values = output.loc[mask, "weight_untrimmed"].to_numpy(float)
        if trim_caps is None:
            quantile_weights = (
                None
                if trim_quantile_weight_column is None
                else output.loc[mask, trim_quantile_weight_column].to_numpy(float)
            )
            low_cap, high_cap = weighted_quantile(
                values, [lo, hi], quantile_weights
            )
        else:
            low_cap = float(trim_caps[strategy]["low"])
            high_cap = float(trim_caps[strategy]["high"])
        output.loc[mask, "weight"] = np.clip(values, low_cap, high_cap)
        caps[strategy] = {"low": float(low_cap), "high": float(high_cap)}
    return output, caps


def weighted_km(intervals, weight_column="weight", horizon_days=HORIZON_DAYS):
    """Calculate strategy-specific weighted KM curves from interval rows."""
    curve_rows = []
    for strategy in STRATEGIES:
        group = intervals.loc[intervals["strategy"].eq(strategy)].copy()
        start_day = (group["interval_start"] - group["index_date"]).dt.days.to_numpy(int)
        stop_day = (group["interval_stop"] - group["index_date"]).dt.days.to_numpy(int)
        weights = group[weight_column].to_numpy(float)
        # Each interval contributes to the risk set on integer days
        # start_day < day <= stop_day.  A difference array makes this O(rows +
        # horizon), rather than repeatedly scanning every interval at each event.
        risk_delta = np.zeros(int(horizon_days) + 2, dtype=np.float64)
        add_day = np.clip(start_day + 1, 0, int(horizon_days) + 1)
        remove_day = np.clip(stop_day + 1, 0, int(horizon_days) + 1)
        np.add.at(risk_delta, add_day, weights)
        np.add.at(risk_delta, remove_day, -weights)
        risk_weight_by_day = np.cumsum(risk_delta)[: int(horizon_days) + 1]

        event_rows = group.loc[group["event_in_interval"].eq(1)].copy()
        event_days_array = (
            event_rows["stop_date"] - event_rows["index_date"]
        ).dt.days.to_numpy(int)
        event_weights_array = event_rows[weight_column].to_numpy(float)
        valid = (event_days_array >= 0) & (event_days_array <= int(horizon_days))
        event_weight_by_day = np.bincount(
            event_days_array[valid],
            weights=event_weights_array[valid],
            minlength=int(horizon_days) + 1,
        )
        event_days = np.flatnonzero(event_weight_by_day > 0).tolist()
        survival = 1.0
        curve_rows.append({"strategy": strategy, "day": 0, "risk": 0.0, "survival": 1.0})
        for day in event_days:
            risk_weight = float(risk_weight_by_day[day])
            event_weight = float(event_weight_by_day[day])
            if risk_weight > 0:
                survival *= max(0.0, 1.0 - event_weight / risk_weight)
            curve_rows.append(
                {"strategy": strategy, "day": day, "risk": 1.0 - survival, "survival": survival}
            )
    return pd.DataFrame(curve_rows)


def curve_landmarks(curve, days=LANDMARK_DAYS):
    rows = []
    for strategy in STRATEGIES:
        group = curve.loc[curve["strategy"].eq(strategy)].sort_values("day")
        for day in days:
            eligible = group.loc[group["day"].le(day)]
            risk = 0.0 if eligible.empty else float(eligible.iloc[-1]["risk"])
            rows.append({"strategy": strategy, "day": int(day), "risk": risk})
    return pd.DataFrame(rows)


def effective_sample_size(values):
    values = np.asarray(values, dtype=np.float64)
    numerator = values.sum() ** 2
    denominator = np.square(values).sum()
    return float(numerator / denominator) if denominator > 0 else 0.0


def self_test():
    patients = pd.DataFrame(
        {
            "patient_key": [1, 2, 3, 4, 5, 6],
            "index_date": pd.to_datetime(["2020-01-01"] * 6),
            "first_adm_date": pd.to_datetime(
                ["2020-02-15", "2020-05-01", "2020-10-01", None, None, "2020-04-01"]
            ),
            "death_date": pd.to_datetime(
                [None, None, None, "2020-01-31", "2021-06-01", None]
            ),
            "observation_end_date": pd.to_datetime(["2025-01-01"] * 6),
            "age": [40, 50, 60, 70, 80, 55],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "index_hba1c": [6.6, 6.8, 7.2, 6.5, 8.0, 7.0],
            "baseline_visits": [1, 2, 3, 4, 5, 3],
        }
    )
    clones = clone_patients(patients)
    assert len(clones) == 24
    # Day-30 death occurs before any grace-period decision in all four arms.
    early = clones.loc[clones["patient_key"].eq(4)]
    assert early["death_event"].sum() == 4
    # Day-60 initiation fulfills all initiation strategies and violates no-init.
    p1 = clones.loc[clones["patient_key"].eq(1)].set_index("strategy")
    assert p1.loc["INIT_WITHIN_3M", "artificial_censor"] == 0
    assert p1.loc["NO_INIT_WITHIN_12M", "artificial_censor"] == 1

    intervals = build_month_intervals(clones)
    month_covariates = []
    for row in intervals.itertuples(index=False):
        month_covariates.append(
            {
                "patient_key": row.patient_key,
                "strategy": row.strategy,
                "month": row.month,
                "lag_hba1c": float(row.index_hba1c) + 0.02 * row.month,
                "lag_visits_30d": float((row.patient_key + row.month) % 4),
            }
        )
    covariates = pd.DataFrame(month_covariates)
    intervals = intervals.merge(
        covariates, on=["patient_key", "strategy", "month"], how="left", validate="one_to_one"
    )
    models = fit_censor_models(
        intervals,
        baseline_numeric=["age", "index_hba1c", "baseline_visits"],
        baseline_categorical=["sex"],
        time_varying_numeric=["lag_hba1c", "lag_visits_30d"],
        ridge=0.1,
    )
    assert models["INIT_WITHIN_3M"]["rows"] <= len(patients)
    assert models["INIT_WITHIN_6M"]["rows"] <= len(patients)
    assert models["INIT_WITHIN_12M"]["rows"] <= len(patients)
    assert models["NO_INIT_WITHIN_12M"]["rows"] > len(patients)
    weighted, caps = apply_censor_weights(
        intervals,
        models,
        baseline_numeric=["age", "index_hba1c", "baseline_visits"],
        baseline_categorical=["sex"],
        time_varying_numeric=["lag_hba1c", "lag_visits_30d"],
        trim_quantiles=(0.0, 0.99),
    )
    assert np.isfinite(weighted["weight"]).all()
    assert (weighted["weight"] > 0).all()
    assert np.isfinite(weighted["unstabilized_weight_untrimmed"]).all()
    assert (weighted["unstabilized_weight_untrimmed"] > 0).all()
    curve = weighted_km(weighted)
    landmarks = curve_landmarks(curve)
    assert set(landmarks["strategy"]) == set(STRATEGIES)
    assert landmarks.groupby("strategy")["risk"].apply(lambda x: x.is_monotonic_increasing).all()
    assert set(caps) == set(STRATEGIES)

    # Reproduce the shape that previously emitted DataFrame fragmentation
    # warnings and stressed the censoring fit: many columns, sparse outcomes,
    # and an exactly duplicated predictor.  PerformanceWarning is promoted to
    # an error so this test fails if column-by-column insertion returns.
    rng = np.random.default_rng(20260722)
    high_dimensional = pd.DataFrame(
        {f"x_{index}": rng.normal(size=1200) for index in range(110)}
    )
    high_dimensional["x_duplicate"] = high_dimensional["x_0"]
    high_dimensional["group"] = rng.choice(["A", "B", "C"], size=len(high_dimensional))
    sparse_probability = expit(
        -3.2
        + 0.7 * high_dimensional["x_0"].to_numpy()
        - 0.4 * high_dimensional["x_1"].to_numpy()
    )
    sparse_outcome = rng.binomial(1, sparse_probability)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        high_x, _ = design_matrix(
            high_dimensional,
            numeric=[f"x_{index}" for index in range(110)] + ["x_duplicate"],
            categorical=["group"],
        )
    high_fit = fit_logistic_irls(high_x, sparse_outcome, ridge=1e-2)
    assert high_fit.converged
    assert np.isfinite(high_fit.coefficients).all()
    print(
        "SELF_TEST_OK "
        f"clones={len(clones)} intervals={len(intervals)} "
        f"weights={weighted['weight'].min():.4f}-{weighted['weight'].max():.4f} "
        f"high_dimensional_iterations={high_fit.iterations}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This module is a library; use --self-test or the Pod runner")
    self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
