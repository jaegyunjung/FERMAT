#!/usr/bin/env python3
"""CPU-only: recency-of-recorded-diagnosis group risk curves (real-data side).

Study name: recency-group risk-curve validation. NOT causal, NOT a
counterfactual "answer key". It splits real patients by how long ago the source
disease was RECORDED before index (`source_recency_days`) and describes the
future target-disease risk of each real group. The two groups differ by
selection as well as timing (the cohort excludes patients with the target before
index, so the long-standing group is enriched for target-free survivors), so the
contrast is a between-group difference, not an effect of diagnosis timing.

Task B (this file) builds the REAL-DATA side, per group:
  competing-risk cumulative incidence (all-cause death is the competing event)
    - unweighted (primary) and IPW-weighted (base / extended adjustment).
  death-as-censoring cause-specific risk (matches the existing FERMAT+Cox)
    - unweighted and IPW-weighted (base / extended).
Per-patient IPW weights are written out so Task C applies the SAME weights to
FERMAT's predicted curves. Non-converged weight models yield NaN weighted curves
AND NaN saved weights (never silently reused). Bootstrap CIs are optional.

Input contract (same as the htn/dyslipidemia audit):
  - pathway_patient_level.parquet : person_id, split, source_recency_days,
    duration_days, event_type, pathway_id
  - baseline_features.parquet     : person_id, split, + adjustment columns

No torch import, checkpoint loading, GPU use, or rollout occurs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover
    pq = None


REQUIRED_PATHWAY_COLUMNS = {
    "pathway_id",
    "person_id",
    "split",
    "source_recency_days",
    "duration_days",
    "event_type",
}
HORIZON_DAYS = 1826
LANDMARK_DAYS = (365, 1096, 1826)  # 1y, 3y (leap-correct), 5y from 2018-01-01
CURVE_DAYS = sorted(set(list(range(0, HORIZON_DAYS + 1, 30)) + list(LANDMARK_DAYS)))
ADJUSTMENT_SETS = ("base", "extended")

# internal curve key -> output column name
CURVE_COLUMN = {
    "cif_unweighted": "observed_cif_unweighted",
    "cif_base": "observed_cif_weighted_base",
    "cif_extended": "observed_cif_weighted_extended",
    "dc_unweighted": "observed_risk_death_censored",
    "dc_base": "observed_risk_death_censored_weighted_base",
    "dc_extended": "observed_risk_death_censored_weighted_extended",
}
CURVE_KEYS = tuple(CURVE_COLUMN)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--config-file", type=Path)
    p.add_argument("--pathway-patient-file", type=Path)
    p.add_argument("--feature-file", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--weight-trim-quantiles", type=float, nargs=2, default=(0.01, 0.99))
    p.add_argument("--bootstrap-samples", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=20260718)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


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


def load_config(path):
    require_file(path)
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "pathway_id",
        "condition_variable",
        "condition_split_days",
        "minimum_group_patients",
        "minimum_group_events",
        "adjustment_sets",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    for name in ADJUSTMENT_SETS:
        block = config["adjustment_sets"].get(name)
        if not block or "numeric" not in block or "categorical" not in block:
            raise ValueError(f"adjustment_sets['{name}'] needs 'numeric' and 'categorical'")
    return config


# --------------------------------------------------------------------------- #
# Curve estimators
# --------------------------------------------------------------------------- #
def weighted_competing_cif(duration, event_type, weight, horizon=HORIZON_DAYS):
    """Cumulative incidence of cause 1 with death (cause 2) as competing event."""
    dur = np.clip(np.asarray(duration).astype(np.int64), 0, horizon)
    event_type = np.asarray(event_type).astype(np.int8)
    weight = np.ones(len(dur)) if weight is None else np.asarray(weight, float)
    target = np.bincount(dur[event_type == 1], weights=weight[event_type == 1], minlength=horizon + 1)[: horizon + 1]
    death = np.bincount(dur[event_type == 2], weights=weight[event_type == 2], minlength=horizon + 1)[: horizon + 1]
    censor = np.bincount(dur[event_type == 0], weights=weight[event_type == 0], minlength=horizon + 1)[: horizon + 1]
    risk = float(weight.sum())
    survival, incidence = 1.0, 0.0
    curve = np.zeros(horizon + 1)
    for day in range(horizon + 1):
        if risk > 1e-12:
            incidence += survival * float(target[day]) / risk
            survival *= 1.0 - (float(target[day]) + float(death[day])) / risk
        curve[day] = incidence
        risk -= float(target[day]) + float(death[day]) + float(censor[day])
    return curve


def death_censored_risk(duration, event_type, weight, horizon=HORIZON_DAYS):
    """Cause-specific risk = 1 - KM survival, death treated as censoring."""
    dur = np.clip(np.asarray(duration).astype(np.int64), 0, horizon)
    event_type = np.asarray(event_type).astype(np.int8)
    weight = np.ones(len(dur)) if weight is None else np.asarray(weight, float)
    target = np.bincount(dur[event_type == 1], weights=weight[event_type == 1], minlength=horizon + 1)[: horizon + 1]
    leaving = np.bincount(dur, weights=weight, minlength=horizon + 1)[: horizon + 1]
    risk = float(weight.sum())
    survival = 1.0
    curve = np.zeros(horizon + 1)
    for day in range(horizon + 1):
        if risk > 1e-12:
            survival *= 1.0 - float(target[day]) / risk
        curve[day] = 1.0 - survival
        risk -= float(leaving[day])
    return curve


# --------------------------------------------------------------------------- #
# Logistic (IRLS) for IPW weights
# --------------------------------------------------------------------------- #
def design_matrix(frame, numeric, categorical):
    columns = [np.ones(len(frame))]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        array = values.fillna(median).to_numpy(float)
        std = float(array.std(ddof=0)) or 1.0
        columns.append((array - float(array.mean())) / std)
    for column in categorical:
        values = frame[column].astype("string").fillna("__MISSING__")
        levels = sorted(values.unique().tolist())
        for level in levels[1:]:
            columns.append(values.eq(level).to_numpy(float))
    matrix = np.column_stack(columns).astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite logistic design matrix")
    return matrix


def fit_logistic(x, y, ridge, iters=100, tol=1e-8):
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    converged = False
    for _ in range(iters):
        eta = np.clip(x @ beta, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(prob * (1.0 - prob), 1e-6, None)
        gradient = x.T @ (y - prob) - penalty @ beta
        hessian = (x * w[:, None]).T @ x + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    prob = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    return prob, converged


def stabilized_weights(prob, is_long, trim_quantiles):
    marginal = float(is_long.mean())
    prob = np.clip(prob, 1e-4, 1 - 1e-4)
    weight = np.where(is_long == 1, marginal / prob, (1.0 - marginal) / (1.0 - prob))
    lo, hi = trim_quantiles
    low_cap, high_cap = float(np.quantile(weight, lo)), float(np.quantile(weight, hi))
    trimmed = np.clip(weight, low_cap, high_cap)
    return trimmed, prob, low_cap, high_cap


def effective_sample_size(weight):
    s1 = float(weight.sum())
    s2 = float((weight ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def standardized_mean_difference(values, is_long, weight):
    values = np.asarray(values, float)
    g1, g0 = is_long == 1, is_long == 0
    w1 = weight[g1] if weight is not None else np.ones(int(g1.sum()))
    w0 = weight[g0] if weight is not None else np.ones(int(g0.sum()))

    def wmean(v, w):
        return float(np.average(v, weights=w)) if w.sum() > 0 else np.nan

    def wvar(v, w):
        m = wmean(v, w)
        return float(np.average((v - m) ** 2, weights=w)) if w.sum() > 0 else np.nan

    m1, m0 = wmean(values[g1], w1), wmean(values[g0], w0)
    pooled = np.sqrt((wvar(values[g1], w1) + wvar(values[g0], w0)) / 2.0)
    return float((m1 - m0) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else np.nan


# --------------------------------------------------------------------------- #
# Core computation (shared by point estimate and bootstrap)
# --------------------------------------------------------------------------- #
def compute_curves(frame, config, args):
    is_long = (frame["condition"].to_numpy() == "long_standing").astype(int)
    dur = frame["duration_days"].to_numpy(float)
    evt = frame["event_type"].to_numpy(np.int8)

    weights, converged, propensity, caps = {}, {}, {}, {}
    for name in ADJUSTMENT_SETS:
        block = config["adjustment_sets"][name]
        x = design_matrix(frame, block["numeric"], block["categorical"])
        prob, conv = fit_logistic(x, is_long.astype(float), args.ridge)
        w, prob, low_cap, high_cap = stabilized_weights(prob, is_long, tuple(args.weight_trim_quantiles))
        weights[name] = w
        converged[name] = bool(conv)
        propensity[name] = prob
        caps[name] = (low_cap, high_cap)

    nan_curve = np.full(HORIZON_DAYS + 1, np.nan)
    curves = {}
    for cond in ("recent", "long_standing"):
        mask = frame["condition"].eq(cond).to_numpy()
        d, e = dur[mask], evt[mask]
        curves[(cond, "cif_unweighted")] = weighted_competing_cif(d, e, None)
        curves[(cond, "dc_unweighted")] = death_censored_risk(d, e, None)
        for name in ADJUSTMENT_SETS:
            if converged[name]:
                curves[(cond, f"cif_{name}")] = weighted_competing_cif(d, e, weights[name][mask])
                curves[(cond, f"dc_{name}")] = death_censored_risk(d, e, weights[name][mask])
            else:
                curves[(cond, f"cif_{name}")] = nan_curve
                curves[(cond, f"dc_{name}")] = nan_curve
    return curves, weights, converged, propensity, caps, is_long


def scope_status(converged):
    ok = [converged[name] for name in ADJUSTMENT_SETS]
    if all(ok):
        return "ok"
    if any(ok):
        return "partial"
    return "unweighted_only"


def point_analysis(frame, config, args, scope_label):
    curves, weights, converged, propensity, caps, is_long = compute_curves(frame, config, args)

    curve_rows, landmark_rows = [], []
    for cond in ("recent", "long_standing"):
        mask = frame["condition"].eq(cond).to_numpy()
        dur = frame["duration_days"].to_numpy(float)[mask]
        evt = frame["event_type"].to_numpy(np.int8)[mask]
        for day in CURVE_DAYS:
            row = {"scope": scope_label, "condition": cond, "day": day}
            for key in CURVE_KEYS:
                row[CURVE_COLUMN[key]] = float(curves[(cond, key)][day])
            curve_rows.append(row)
        for day in LANDMARK_DAYS:
            row = {
                "scope": scope_label, "condition": cond, "day": day,
                "patients": int(mask.sum()),
                "target_events_by_day": int(((evt == 1) & (dur <= day)).sum()),
                "death_events_by_day": int(((evt == 2) & (dur <= day)).sum()),
            }
            for key in CURVE_KEYS:
                row[CURVE_COLUMN[key]] = float(curves[(cond, key)][day])
            landmark_rows.append(row)

    # balance: numeric AND categorical (each level as a 0/1 indicator)
    balance_rows = []
    ext = config["adjustment_sets"]["extended"]
    for column in ext["numeric"]:
        series = pd.to_numeric(frame[column], errors="coerce")
        values = series.fillna(series.median()).to_numpy(float)
        row = {"scope": scope_label, "variable": column, "kind": "numeric",
               "smd_unweighted": standardized_mean_difference(values, is_long, None)}
        for name in ADJUSTMENT_SETS:
            row[f"smd_weighted_{name}"] = (
                standardized_mean_difference(values, is_long, weights[name]) if converged[name] else np.nan
            )
        balance_rows.append(row)
    for column in ext["categorical"]:
        series = frame[column].astype("string").fillna("__MISSING__")
        for level in sorted(series.unique().tolist()):
            indicator = series.eq(level).to_numpy(float)
            row = {"scope": scope_label, "variable": f"{column}={level}", "kind": "categorical",
                   "smd_unweighted": standardized_mean_difference(indicator, is_long, None)}
            for name in ADJUSTMENT_SETS:
                row[f"smd_weighted_{name}"] = (
                    standardized_mean_difference(indicator, is_long, weights[name]) if converged[name] else np.nan
                )
            balance_rows.append(row)

    weight_summary = {}
    for name in ADJUSTMENT_SETS:
        w, prob = weights[name], propensity[name]
        weight_summary[name] = {
            "converged": converged[name],
            "ess_recent": effective_sample_size(w[is_long == 0]) if converged[name] else None,
            "ess_long_standing": effective_sample_size(w[is_long == 1]) if converged[name] else None,
            "propensity_min": float(prob.min()),
            "propensity_max": float(prob.max()),
            "trim_low_cap": caps[name][0],
            "trim_high_cap": caps[name][1],
            "n_at_trim_cap": int(((w <= caps[name][0]) | (w >= caps[name][1])).sum()),
        }

    # per-patient weights: NaN for a set that did not converge (never reuse them)
    weights_frame = frame[["person_id", "split", "condition"]].copy()
    for name in ADJUSTMENT_SETS:
        weights_frame[f"w_{name}"] = weights[name] if converged[name] else np.nan
        weights_frame[f"w_{name}_converged"] = converged[name]
    weights_frame["scope"] = scope_label

    diagnostics = {
        "scope": scope_label,
        "status": scope_status(converged),
        "patients": int(len(frame)),
        "recent_patients": int((is_long == 0).sum()),
        "long_standing_patients": int((is_long == 1).sum()),
        "recent_target_events": int(((frame["condition"] == "recent") & (frame["event_type"] == 1)).sum()),
        "long_standing_target_events": int(((frame["condition"] == "long_standing") & (frame["event_type"] == 1)).sum()),
        "death_events": int((frame["event_type"] == 2).sum()),
        "weights": weight_summary,
        "death_handling": "competing (observed_cif_*) and death-censored (observed_risk_death_censored*)",
    }
    return pd.DataFrame(curve_rows), pd.DataFrame(landmark_rows), pd.DataFrame(balance_rows), weights_frame, diagnostics


def bootstrap_ci(frame, config, args, scope_label):
    if args.bootstrap_samples <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(args.bootstrap_seed)
    index = frame.index.to_numpy()
    draws = {(cond, day, key): [] for cond in ("recent", "long_standing")
             for day in LANDMARK_DAYS for key in CURVE_KEYS}
    for _ in range(args.bootstrap_samples):
        sample = frame.loc[rng.choice(index, size=len(index), replace=True)].reset_index(drop=True)
        if sample["condition"].nunique() < 2:
            continue
        curves, _, _, _, _, _ = compute_curves(sample, config, args)
        for cond in ("recent", "long_standing"):
            for day in LANDMARK_DAYS:
                for key in CURVE_KEYS:
                    draws[(cond, day, key)].append(float(curves[(cond, key)][day]))
    rows = []
    for (cond, day, key), values in draws.items():
        arr = np.array([v for v in values if np.isfinite(v)])
        if arr.size < max(10, args.bootstrap_samples // 5):
            ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
        rows.append({"scope": scope_label, "condition": cond, "day": day,
                     "curve_type": CURVE_COLUMN[key], "boot_n": int(arr.size),
                     "ci95_low": ci_low, "ci95_high": ci_high})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Load / condition assignment
# --------------------------------------------------------------------------- #
def assign_condition(frame, config):
    values = pd.to_numeric(frame[config["condition_variable"]], errors="coerce")
    if values.isna().any():
        n = int(values.isna().sum())
        raise ValueError(
            f"{n} patients have missing/non-numeric {config['condition_variable']}; "
            "these must not be silently assigned to 'recent'. Fix the cohort upstream."
        )
    return np.where(values > float(config["condition_split_days"]), "long_standing", "recent")


def load_inputs(args, config):
    available = set(parquet_columns(args.pathway_patient_file))
    missing = sorted(REQUIRED_PATHWAY_COLUMNS - available)
    if missing:
        raise ValueError(f"Pathway patient file missing columns: {missing}")
    pathway = pd.read_parquet(args.pathway_patient_file)
    pathway = pathway.loc[pathway["pathway_id"].eq(config["pathway_id"])].copy()
    if pathway.empty or pathway["person_id"].duplicated().any():
        raise ValueError("Selected pathway is empty or has duplicate patients")

    needed = {"person_id", "split"}
    for block in config["adjustment_sets"].values():
        needed |= set(block["numeric"]) | set(block["categorical"])
    available_features = set(parquet_columns(args.feature_file))
    fmissing = sorted(needed - available_features)
    if fmissing:
        raise ValueError(f"Baseline feature file missing columns: {fmissing}")
    features = pd.read_parquet(args.feature_file, columns=sorted(needed))
    data = pathway.merge(features, on=["person_id", "split"], how="left", validate="one_to_one")
    data["condition"] = assign_condition(data, config)
    return data


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def run_self_test():
    rng = np.random.default_rng(3)
    n = 2000
    age = rng.normal(60, 10, n)
    is_long = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.06 * (age - 60))))).astype(int)
    rate = 0.0003 * np.exp(0.05 * (age - 60) + 0.15 * is_long)
    t_event = rng.exponential(1 / rate)
    t_death = rng.exponential(1 / (0.00015 * np.ones(n)))
    dur = np.minimum.reduce([t_event, t_death, np.full(n, 1826.0)])
    evt = np.where((t_event <= t_death) & (t_event <= 1826), 1,
                   np.where((t_death < t_event) & (t_death <= 1826), 2, 0)).astype(np.int8)
    sex = np.where(age + rng.normal(0, 5, n) > 60, "F", "M")  # sex correlated with age -> imbalanced
    frame = pd.DataFrame({
        "person_id": np.arange(n), "split": "test",
        "condition": np.where(is_long == 1, "long_standing", "recent"),
        "duration_days": dur, "event_type": evt, "age": age, "sex": sex,
    })
    config = {"adjustment_sets": {
        "base": {"numeric": ["age"], "categorical": ["sex"]},
        "extended": {"numeric": ["age"], "categorical": ["sex"]}}}
    args = argparse.Namespace(ridge=1e-4, weight_trim_quantiles=(0.01, 0.99),
                              bootstrap_samples=0, bootstrap_seed=1)
    _, landmarks, balance, weights, diag = point_analysis(frame, config, args, "test")
    lm = landmarks.loc[landmarks.day == 1826].set_index("condition")
    gap_unw = abs(lm.loc["long_standing", "observed_cif_unweighted"] - lm.loc["recent", "observed_cif_unweighted"])
    gap_w = abs(lm.loc["long_standing", "observed_cif_weighted_base"] - lm.loc["recent", "observed_cif_weighted_base"])
    smd_age_u = abs(balance.loc[balance.variable == "age", "smd_unweighted"].iloc[0])
    smd_age_w = abs(balance.loc[balance.variable == "age", "smd_weighted_base"].iloc[0])
    sex_rows = balance.loc[balance.kind == "categorical"]
    assert not sex_rows.empty, "categorical (sex) balance missing"
    assert diag["status"] == "ok"
    assert gap_w < gap_unw, f"IPW did not shrink outcome gap {gap_unw:.4f}->{gap_w:.4f}"
    assert smd_age_w < smd_age_u, "IPW did not shrink age SMD"
    assert diag["recent_target_events"] > 0 and diag["long_standing_target_events"] > 0
    assert {365, 1096, 1826}.issubset(set(CURVE_DAYS))
    assert "observed_risk_death_censored" in landmarks.columns
    assert "observed_risk_death_censored_weighted_base" in landmarks.columns
    assert weights["w_base_converged"].all()
    # recency guard
    bad = frame.copy(); bad["source_recency_days"] = 100.0; bad.loc[0, "source_recency_days"] = np.nan
    try:
        assign_condition(bad, {"condition_variable": "source_recency_days", "condition_split_days": 730})
        raise AssertionError("missing recency was not rejected")
    except ValueError:
        pass
    log(f"[SELF-TEST PASS] gap {gap_unw:.4f}->{gap_w:.4f}; age SMD {smd_age_u:.3f}->{smd_age_w:.3f}; "
        f"sex balance rows={len(sex_rows)}; death-censored+weighted cols present; guards OK")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    for name in ("config_file", "pathway_patient_file", "feature_file", "output_dir"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_','-')} is required")
    config = load_config(args.config_file)
    for path in (args.config_file, args.pathway_patient_file, args.feature_file):
        require_file(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)

    started = time.time()
    data = load_inputs(args, config)

    curves_all, landmarks_all, balance_all, weights_all, ci_all, diag_all = [], [], [], [], [], []
    for scope in ("test", "all"):
        scoped = data if scope == "all" else data.loc[data["split"].eq(scope)].copy()
        counts = pd.Series(scoped["condition"]).value_counts()
        recent_events = int(((scoped["condition"] == "recent") & (scoped["event_type"] == 1)).sum())
        long_events = int(((scoped["condition"] == "long_standing") & (scoped["event_type"] == 1)).sum())
        min_pat = int(config["minimum_group_patients"])
        min_evt = int(config["minimum_group_events"])
        if (counts.get("recent", 0) < min_pat or counts.get("long_standing", 0) < min_pat
                or recent_events < min_evt or long_events < min_evt):
            diag_all.append({
                "scope": scope, "status": "insufficient_data",
                "recent_patients": int(counts.get("recent", 0)),
                "long_standing_patients": int(counts.get("long_standing", 0)),
                "recent_target_events": recent_events,
                "long_standing_target_events": long_events,
            })
            continue
        curves, landmarks, balance, weights, diag = point_analysis(scoped, config, args, scope)
        ci = bootstrap_ci(scoped, config, args, scope)
        curves_all.append(curves); landmarks_all.append(landmarks)
        balance_all.append(balance); weights_all.append(weights)
        if not ci.empty:
            ci_all.append(ci)
        diag_all.append(diag)

    def concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    curves = concat(curves_all)
    landmarks = concat(landmarks_all)
    balance = concat(balance_all)
    weights = concat(weights_all)
    ci = concat(ci_all)

    curves.to_csv(args.output_dir / "recency_group_curves.csv", index=False)
    landmarks.to_csv(args.output_dir / "recency_group_landmarks.csv", index=False)
    balance.to_csv(args.output_dir / "covariate_balance.csv", index=False)
    if not ci.empty:
        ci.to_csv(args.output_dir / "recency_group_landmark_ci.csv", index=False)
    if not weights.empty:
        weights.to_parquet(args.output_dir / "raw" / "patient_ipw_weights.parquet", index=False)
    write_json(diag_all, args.output_dir / "recency_group_diagnostics.json")

    lines = [
        "## STATUS", "COMPLETE_TASK30_RECENCY_GROUP_CURVES",
        f"pathway={config['pathway_id']} split_variable={config['condition_variable']}"
        f" split_days={config['condition_split_days']} bootstrap={args.bootstrap_samples}",
        "## LANDMARKS", landmarks.to_csv(index=False).rstrip() if not landmarks.empty else "(none)",
        "## COVARIATE_BALANCE", balance.to_csv(index=False).rstrip() if not balance.empty else "(none)",
        "## LANDMARK_CI", ci.to_csv(index=False).rstrip() if not ci.empty else "(bootstrap not run)",
        "## DIAGNOSTICS", json.dumps(diag_all, ensure_ascii=False, default=str),
        "## CLAIM_BOUNDARY",
        "RECENCY_GROUP_DIFFERENCE_NOT_A_DIAGNOSIS_TIMING_EFFECT_NOT_CAUSAL",
        "IPW_WEIGHTS_WRITTEN_FOR_SYMMETRIC_REUSE; NON_CONVERGED_SETS_SAVED_AS_NAN",
        "COMPETING_RISK_IS_PRIMARY; DEATH_CENSORED_COLUMNS_MATCH_EXISTING_FERMAT_COX",
        "NO_GPU_NO_CHECKPOINT_NO_ROLLOUT",
        "## OUTPUT_DIR", str(args.output_dir),
    ]
    text = "\n".join(lines) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log(f"[COMPLETE] recency-group curves finished in {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
