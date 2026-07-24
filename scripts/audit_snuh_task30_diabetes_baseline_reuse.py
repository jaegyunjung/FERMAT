#!/usr/bin/env python3
"""Audit whether the saved Task 30 diabetes observed/Cox baseline is reusable.

This is a read-only, CPU-only audit. It reconstructs the exact modelable
cause-specific diabetes cohort from the saved inputs and verifies the saved
observed curve, two Cox prediction tables, and metrics. Missing derived
combined-curve/landmark files are rebuilt in the new audit directory; the
source baseline directory is never modified. Optional provenance/model
artifacts are reported as warnings when absent. The audit also makes the
competing-risk limitation explicit: the saved observed curve is Kaplan-Meier
1-S(t), whereas the outcome feasibility audit used Aalen-Johansen with death
as a competing event.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_BASELINE_DIR = (
    TASK30 / "outputs" / "diabetes_rollout_stratification_20260715_075746"
)
DEFAULT_AUDIT_DIR = (
    TASK30 / "outputs" / "outcome_feasibility_audit_20260715_193937"
)
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
DEFAULT_LAB_FILE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "lab_marker_features"
    / "lab_marker_features_wide_20180101.parquet"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task20"
    / "outputs"
    / "cox_survival_2018_5y_block2048_1000ci_20260703"
    / "first_phenotype_dates_20180101.parquet"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "diabetes_baseline_reuse_audit"
INDEX_DATE = pd.Timestamp("2018-01-01")
HORIZON_END = pd.Timestamp("2023-01-01")
MAX_DAY = 1826
LANDMARKS = [0, 365, 1095, 1826]
MODELS = ["clinical_cox", "fermat_clinical_cox"]
REFERENCE_METRICS = {
    "clinical_cox": {"test_c_index": 0.7764, "test_5y_auc": 0.7806},
    "fermat_clinical_cox": {"test_c_index": 0.7990, "test_5y_auc": 0.8034},
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--feasibility-audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--lab-file", type=Path, default=DEFAULT_LAB_FILE)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def normalize_paths(args):
    for name in (
        "baseline_dir",
        "feasibility_audit_dir",
        "label_file",
        "feature_file",
        "embedding_file",
        "lab_file",
        "survival_cache",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def required_paths(args):
    """Require only files needed to reconstruct and validate the core result."""
    paths = [
        args.label_file,
        args.feature_file,
        args.embedding_file,
        args.lab_file,
        args.survival_cache,
        args.feasibility_audit_dir / "outcome_feasibility_summary.csv",
        args.feasibility_audit_dir / "outcome_split_horizon_summary.csv",
        args.baseline_dir / "endpoint_definition.json",
        args.baseline_dir / "cohort_summary.csv",
        args.baseline_dir / "full_test_observed_diabetes_curve.csv",
        args.baseline_dir / "cox_test_predictions.parquet",
        args.baseline_dir / "cox_population_curves.csv",
        args.baseline_dir / "cox_model_metrics.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def validate_unique(frame, name, columns=("person_id", "split")):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")
    duplicates = int(frame.duplicated(list(columns)).sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates:,} duplicate key rows")


def truthy(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def reconstruct_modelable_cohort(args):
    label_columns = [
        "person_id",
        "split",
        "index_date",
        "first_activity_date",
        "last_activity_date",
        "has_pre_index_washout",
        "prior__diabetes",
    ]
    labels = pd.read_parquet(args.label_file, columns=label_columns)
    validate_unique(labels, "labels")
    for column in ("index_date", "first_activity_date", "last_activity_date"):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")
    unexpected_index = labels["index_date"].notna() & labels["index_date"].ne(INDEX_DATE)
    if unexpected_index.any():
        raise ValueError(f"labels contain {int(unexpected_index.sum()):,} unexpected index dates")

    features = pd.read_parquet(args.feature_file, columns=["person_id", "split"])
    validate_unique(features, "baseline features")
    embeddings = pd.read_parquet(
        args.embedding_file,
        columns=["person_id", "split", "has_embedding_sequence"],
    )
    validate_unique(embeddings, "embeddings")
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)]

    dates = pd.read_parquet(args.survival_cache)
    required_dates = {"phenotype", "person_id", "first_phenotype_date"}
    missing = sorted(required_dates - set(dates.columns))
    if missing:
        raise ValueError(f"survival cache missing columns: {missing}")
    dates = dates.loc[
        dates["phenotype"].eq("diabetes"),
        ["person_id", "first_phenotype_date"],
    ].copy()
    if dates["person_id"].duplicated().any():
        raise ValueError("diabetes survival dates contain duplicate person_id")
    dates["first_phenotype_date"] = pd.to_datetime(
        dates["first_phenotype_date"], errors="coerce"
    )

    data = labels.merge(
        features, on=["person_id", "split"], how="inner", validate="one_to_one"
    )
    data = data.merge(
        embeddings[["person_id", "split"]],
        on=["person_id", "split"],
        how="inner",
        validate="one_to_one",
    )
    task = data.merge(dates, on="person_id", how="left", validate="one_to_one")

    source_prior = task["first_phenotype_date"].notna() & task[
        "first_phenotype_date"
    ].lt(INDEX_DATE)
    label_prior = task["prior__diabetes"].fillna(0).astype(bool)
    prior_mismatch = int((source_prior != label_prior).sum())
    at_risk = task["has_pre_index_washout"].fillna(False).astype(bool) & ~label_prior
    task = task.loc[at_risk].copy()

    first_date = task["first_phenotype_date"]
    event = first_date.notna() & first_date.ge(INDEX_DATE) & first_date.le(HORIZON_END)
    censor = task["last_activity_date"].where(
        task["last_activity_date"].notna(), HORIZON_END
    )
    censor = censor.clip(upper=HORIZON_END)
    endpoint = censor.where(~event, first_date)
    task["duration_days"] = (endpoint - INDEX_DATE).dt.days.astype("float64")
    task["event"] = event.astype("int8")
    task = task.loc[task["duration_days"] > 0].copy()
    task["duration_days"] = task["duration_days"].astype("float32")

    rows = []
    for split in ("train", "val", "test"):
        sub = task.loc[task["split"].eq(split)]
        rows.append(
            {
                "split": split,
                "patients": int(len(sub)),
                "diabetes_events_within_5y": int(sub["event"].sum()),
                "median_followup_days": float(sub["duration_days"].median()),
                "max_followup_days": float(sub["duration_days"].max()),
            }
        )
    return task, pd.DataFrame(rows), prior_mismatch


def add_check(checks, name, passed, detail, severity="hard"):
    checks.append(
        {
            "check": name,
            "severity": severity,
            "passed": bool(passed),
            "detail": str(detail),
        }
    )


def exact_day_axis(frame):
    return np.array_equal(
        frame["day"].to_numpy(dtype=np.int64), np.arange(MAX_DAY + 1, dtype=np.int64)
    )


def bounded_monotone(values):
    values = np.asarray(values, dtype=np.float64)
    return bool(
        np.isfinite(values).all()
        and (values >= -1e-12).all()
        and (values <= 1 + 1e-12).all()
        and (np.diff(values) >= -1e-12).all()
    )


def frames_equivalent(left, right):
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                left[column].to_numpy(dtype=np.float64),
                right[column].to_numpy(dtype=np.float64),
                equal_nan=True,
            ):
                return False
        elif not left[column].fillna("<NA>").astype(str).equals(
            right[column].fillna("<NA>").astype(str)
        ):
            return False
    return True


def audit_saved_baseline(args, task, reconstructed, prior_mismatch):
    checks = []
    saved_summary = pd.read_csv(args.baseline_dir / "cohort_summary.csv")
    add_check(
        checks,
        "cohort_summary_splits",
        set(saved_summary["split"]) == {"train", "val", "test"},
        saved_summary["split"].tolist(),
    )
    for row in reconstructed.itertuples(index=False):
        saved = saved_summary.loc[saved_summary["split"].eq(row.split)]
        same = len(saved) == 1
        if same:
            saved_row = saved.iloc[0]
            same = (
                int(saved_row["patients"]) == int(row.patients)
                and int(saved_row["diabetes_events_within_5y"])
                == int(row.diabetes_events_within_5y)
                and np.isclose(
                    float(saved_row["median_followup_days"]),
                    float(row.median_followup_days),
                )
                and np.isclose(
                    float(saved_row["max_followup_days"]),
                    float(row.max_followup_days),
                )
            )
        add_check(
            checks,
            f"reconstructed_cohort_matches_{row.split}",
            same,
            f"saved={saved.to_dict('records')} reconstructed={row._asdict()}",
        )
    add_check(
        checks,
        "source_label_prior_consistency",
        prior_mismatch == 0,
        f"mismatches={prior_mismatch}",
    )

    test = task.loc[task["split"].eq("test"), ["person_id", "duration_days", "event"]]
    test = test.sort_values("person_id").reset_index(drop=True)
    predictions = pd.read_parquet(args.baseline_dir / "cox_test_predictions.parquet")
    required_prediction_columns = {
        "person_id",
        "duration_days",
        "event",
        "clinical_cox_score",
        "clinical_cox_risk_5y",
        "fermat_clinical_cox_score",
        "fermat_clinical_cox_risk_5y",
    }
    add_check(
        checks,
        "prediction_columns",
        required_prediction_columns.issubset(predictions.columns),
        sorted(predictions.columns),
    )
    predictions = predictions.sort_values("person_id").reset_index(drop=True)
    same_keys = len(test) == len(predictions)
    if same_keys:
        same_keys = (
            np.array_equal(test["person_id"], predictions["person_id"])
            and np.allclose(test["duration_days"], predictions["duration_days"])
            and np.array_equal(test["event"], predictions["event"])
        )
    add_check(
        checks,
        "prediction_rows_match_reconstructed_test",
        same_keys,
        f"reconstructed={len(test)} saved={len(predictions)}",
    )
    add_check(
        checks,
        "prediction_person_id_unique",
        not predictions["person_id"].duplicated().any(),
        f"duplicates={int(predictions['person_id'].duplicated().sum())}",
    )
    for model in MODELS:
        score = predictions[f"{model}_score"].to_numpy(dtype=np.float64)
        risk = predictions[f"{model}_risk_5y"].to_numpy(dtype=np.float64)
        add_check(
            checks,
            f"{model}_scores_finite",
            np.isfinite(score).all(),
            f"finite={int(np.isfinite(score).sum())}/{len(score)}",
        )
        add_check(
            checks,
            f"{model}_risks_bounded",
            np.isfinite(risk).all() and (risk >= 0).all() and (risk <= 1).all(),
            f"min={np.nanmin(risk):.8f} max={np.nanmax(risk):.8f}",
        )

    observed = pd.read_csv(args.baseline_dir / "full_test_observed_diabetes_curve.csv")
    cox_curves = pd.read_csv(args.baseline_dir / "cox_population_curves.csv")
    for name, frame in (("observed", observed), ("cox", cox_curves)):
        add_check(checks, f"{name}_day_axis", exact_day_axis(frame), f"rows={len(frame)}")
    required_observed_columns = {
        "day",
        "observed",
        "patients_at_risk_after_day",
        "deaths_on_day",
    }
    required_cox_curve_columns = {
        "day",
        "clinical_cox_mean_predicted",
        "fermat_clinical_cox_mean_predicted",
    }
    if not required_observed_columns.issubset(observed.columns):
        raise ValueError(
            "observed curve missing columns: "
            f"{sorted(required_observed_columns - set(observed.columns))}"
        )
    if not required_cox_curve_columns.issubset(cox_curves.columns):
        raise ValueError(
            "Cox population curves missing columns: "
            f"{sorted(required_cox_curve_columns - set(cox_curves.columns))}"
        )

    combined = observed[
        ["day", "observed", "patients_at_risk_after_day"]
    ].merge(cox_curves, on="day", how="inner", validate="one_to_one")
    landmarks = combined.loc[combined["day"].isin(LANDMARKS)].reset_index(drop=True)
    recovered_curves_path = (
        args.output_dir / "recovered_cohort_observed_and_cox_curves.csv"
    )
    recovered_landmarks_path = (
        args.output_dir / "recovered_cohort_observed_and_cox_landmarks.csv"
    )
    combined.to_csv(recovered_curves_path, index=False)
    landmarks.to_csv(recovered_landmarks_path, index=False)
    add_check(
        checks,
        "recovered_combined_day_axis",
        exact_day_axis(combined),
        f"rows={len(combined)}",
    )

    missing_optional = []
    saved_combined_path = args.baseline_dir / "cohort_observed_and_cox_curves.csv"
    if saved_combined_path.is_file():
        saved_combined = pd.read_csv(saved_combined_path)
        add_check(
            checks,
            "saved_combined_curve_matches_recovered",
            frames_equivalent(saved_combined, combined),
            "saved file vs deterministic merge of observed and Cox curves",
        )
    else:
        missing_optional.append(str(saved_combined_path))
        add_check(
            checks,
            "saved_combined_curve_present",
            False,
            f"recovered={recovered_curves_path}",
            severity="soft",
        )

    saved_landmarks_path = args.baseline_dir / "cohort_observed_and_cox_landmarks.csv"
    if saved_landmarks_path.is_file():
        saved_landmarks = pd.read_csv(saved_landmarks_path)
        add_check(
            checks,
            "saved_landmarks_match_recovered",
            frames_equivalent(saved_landmarks, landmarks),
            "saved file vs recovered 0/1/3/5y rows",
        )
    else:
        missing_optional.append(str(saved_landmarks_path))
        add_check(
            checks,
            "saved_landmarks_present",
            False,
            f"recovered={recovered_landmarks_path}",
            severity="soft",
        )

    saved_svg_path = args.baseline_dir / "cohort_observed_and_cox_curves.svg"
    if not saved_svg_path.is_file():
        missing_optional.append(str(saved_svg_path))
    add_check(
        checks,
        "saved_curve_svg_present",
        saved_svg_path.is_file(),
        str(saved_svg_path),
        severity="soft",
    )
    for column in (
        "observed",
        "clinical_cox_mean_predicted",
        "fermat_clinical_cox_mean_predicted",
    ):
        add_check(
            checks,
            f"curve_{column}_bounded_monotone",
            bounded_monotone(combined[column]),
            f"start={combined[column].iloc[0]:.8f} end={combined[column].iloc[-1]:.8f}",
        )
    add_check(
        checks,
        "observed_event_sum_matches_test",
        int(observed["deaths_on_day"].sum()) == int(test["event"].sum()),
        f"curve={int(observed['deaths_on_day'].sum())} test={int(test['event'].sum())}",
    )
    add_check(
        checks,
        "landmark_days",
        landmarks["day"].astype(int).tolist() == LANDMARKS,
        landmarks["day"].astype(int).tolist(),
    )
    expected_landmarks = combined.loc[combined["day"].isin(LANDMARKS)].reset_index(drop=True)
    add_check(
        checks,
        "landmark_values_match_curves",
        frames_equivalent(landmarks, expected_landmarks),
        "recovered 0/1/3/5y rows",
    )

    metrics = pd.read_csv(args.baseline_dir / "cox_model_metrics.csv")
    add_check(
        checks,
        "metric_model_rows",
        set(metrics["model"]) == set(MODELS) and len(metrics) == 2,
        metrics["model"].tolist(),
    )
    test_events = int(test["event"].sum())
    for model in MODELS:
        row = metrics.loc[metrics["model"].eq(model)]
        exists = len(row) == 1
        add_check(checks, f"{model}_metric_row_unique", exists, f"rows={len(row)}")
        if not exists:
            continue
        row = row.iloc[0]
        metric_valid = all(
            np.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1
            for field in ("test_c_index", "test_5y_auc")
        )
        add_check(
            checks,
            f"{model}_metrics_bounded",
            metric_valid,
            f"c_index={row['test_c_index']} auc={row['test_5y_auc']}",
        )
        add_check(
            checks,
            f"{model}_cases_match_test_events",
            int(row["test_5y_cases"]) == test_events,
            f"metric_cases={int(row['test_5y_cases'])} test_events={test_events}",
        )
        reference = REFERENCE_METRICS[model]
        reference_match = all(
            abs(float(row[field]) - value) <= 0.001
            for field, value in reference.items()
        )
        add_check(
            checks,
            f"{model}_matches_reported_reference",
            reference_match,
            f"saved_c={row['test_c_index']:.6f} saved_auc={row['test_5y_auc']:.6f}",
            severity="soft",
        )
        model_dir = args.baseline_dir / "models" / model
        artifact_paths = [
            model_dir / "cox_model.npz",
            model_dir / "transformer.json",
            model_dir / "training_history.csv",
            model_dir / "metrics.json",
        ]
        missing_model_artifacts = [
            str(path) for path in artifact_paths if not path.is_file()
        ]
        missing_optional.extend(missing_model_artifacts)
        add_check(
            checks,
            f"{model}_model_artifacts_complete",
            not missing_model_artifacts,
            (
                "all present"
                if not missing_model_artifacts
                else "missing=" + ",".join(missing_model_artifacts)
            ),
            severity="soft",
        )
        metrics_json_path = model_dir / "metrics.json"
        if metrics_json_path.is_file():
            model_metric = json.loads(metrics_json_path.read_text(encoding="utf-8"))
            json_match = all(
                np.isclose(float(row[field]), float(model_metric[field]))
                for field in ("test_c_index", "test_5y_auc", "best_val_loss")
            )
            add_check(
                checks,
                f"{model}_csv_json_metrics_match",
                json_match,
                "CSV vs JSON",
            )

    manifest_path = args.baseline_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add_check(
            checks,
            "manifest_cohort_stage_complete",
            manifest.get("status") == "complete_cohort_stage"
            and manifest.get("rollout_executed") is False
            and manifest.get("complete_test_cohort_used") is True,
            json.dumps(
                {
                    "status": manifest.get("status"),
                    "rollout_executed": manifest.get("rollout_executed"),
                    "complete_test_cohort_used": manifest.get(
                        "complete_test_cohort_used"
                    ),
                }
            ),
        )
    else:
        missing_optional.append(str(manifest_path))
        add_check(
            checks,
            "manifest_present",
            False,
            "core outputs remain auditable; run provenance is incomplete",
            severity="soft",
        )
    endpoint = json.loads(
        (args.baseline_dir / "endpoint_definition.json").read_text(encoding="utf-8")
    )
    endpoint_text = json.dumps(endpoint, ensure_ascii=False).lower()
    add_check(
        checks,
        "endpoint_is_diabetes_2018_5y",
        "diabetes" in endpoint_text
        and "2018-01-01" in endpoint_text
        and "1826" in str(endpoint.get("horizon", "")),
        json.dumps(endpoint, ensure_ascii=False),
    )
    recovery = {
        "recovery_needed": bool(missing_optional),
        "missing_optional_source_files": sorted(set(missing_optional)),
        "recovered_curves": str(recovered_curves_path),
        "recovered_landmarks": str(recovered_landmarks_path),
        "source_baseline_modified": False,
    }
    return checks, saved_summary, metrics, landmarks, recovery


def build_audit_context(args, reconstructed):
    feasibility = pd.read_csv(
        args.feasibility_audit_dir / "outcome_feasibility_summary.csv"
    )
    split_summary = pd.read_csv(
        args.feasibility_audit_dir / "outcome_split_horizon_summary.csv"
    )
    candidate = feasibility.loc[feasibility["outcome_name"].eq("diabetes")]
    full = split_summary.loc[
        split_summary["outcome_name"].eq("diabetes")
        & split_summary["split"].eq("test")
        & split_summary["horizon_years"].eq(5)
    ]
    if len(candidate) != 1 or len(full) != 1:
        raise ValueError("Feasibility audit must contain one diabetes test 5y row")
    candidate = candidate.iloc[0]
    full = full.iloc[0]
    modelable = reconstructed.loc[reconstructed["split"].eq("test")].iloc[0]
    return pd.DataFrame(
        [
            {
                "outcome": "diabetes",
                "audit_full_test_eligible": int(full["eligible_patients"]),
                "audit_full_test_events_5y": int(full["outcome_events"]),
                "audit_full_test_cumulative_incidence_5y": float(
                    full["observed_cumulative_incidence"]
                ),
                "audit_exact_etl_token_capture_5y": float(
                    candidate["test_exact_etl_token_capture_5y"]
                ),
                "audit_etl_spike_flag": truthy(candidate["etl_spike_flag"]),
                "saved_modelable_test_patients": int(modelable["patients"]),
                "saved_modelable_test_events_5y": int(
                    modelable["diabetes_events_within_5y"]
                ),
                "modelable_patient_fraction_of_full_at_risk": float(
                    modelable["patients"] / full["eligible_patients"]
                ),
                "modelable_event_fraction_of_full_events": float(
                    modelable["diabetes_events_within_5y"] / full["outcome_events"]
                ),
            }
        ]
    )


def run_self_test():
    good = np.array([0.0, 0.0, 0.1, 0.1, 0.3])
    bad = np.array([0.0, 0.2, 0.1])
    if not bounded_monotone(good):
        raise AssertionError("valid curve was rejected")
    if bounded_monotone(bad):
        raise AssertionError("decreasing curve was accepted")
    frame = pd.DataFrame({"day": np.arange(MAX_DAY + 1)})
    if not exact_day_axis(frame):
        raise AssertionError("valid day axis was rejected")
    log("[SELF-TEST PASS] curve and day-axis validation")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    normalize_paths(args)
    required_paths(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("[START] reconstruct saved diabetes modelable cohort")
    task, reconstructed, prior_mismatch = reconstruct_modelable_cohort(args)
    reconstructed.to_csv(args.output_dir / "reconstructed_cohort_summary.csv", index=False)
    checks, saved_summary, metrics, landmarks, recovery = audit_saved_baseline(
        args, task, reconstructed, prior_mismatch
    )
    context = build_audit_context(args, reconstructed)
    context_row = context.iloc[0]
    add_check(
        checks,
        "feasibility_audit_diabetes_precheck",
        float(context_row["audit_exact_etl_token_capture_5y"]) >= 0.99
        and not bool(context_row["audit_etl_spike_flag"]),
        (
            f"token_capture={context_row['audit_exact_etl_token_capture_5y']} "
            f"etl_spike={context_row['audit_etl_spike_flag']}"
        ),
    )
    add_check(
        checks,
        "modelable_cohort_is_subset_by_counts",
        int(context_row["saved_modelable_test_patients"])
        <= int(context_row["audit_full_test_eligible"])
        and int(context_row["saved_modelable_test_events_5y"])
        <= int(context_row["audit_full_test_events_5y"]),
        (
            f"modelable_n={context_row['saved_modelable_test_patients']} "
            f"full_n={context_row['audit_full_test_eligible']} "
            f"modelable_events={context_row['saved_modelable_test_events_5y']} "
            f"full_events={context_row['audit_full_test_events_5y']}"
        ),
    )
    context.to_csv(args.output_dir / "baseline_vs_feasibility_audit.csv", index=False)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(args.output_dir / "baseline_reuse_checks.csv", index=False)

    hard_failures = checks_frame.loc[
        checks_frame["severity"].eq("hard") & ~checks_frame["passed"]
    ]
    soft_failures = checks_frame.loc[
        checks_frame["severity"].eq("soft") & ~checks_frame["passed"]
    ]
    if not hard_failures.empty:
        internal_status = "FAIL_BASELINE_REUSE"
    elif recovery["recovery_needed"]:
        internal_status = "PASS_RECOVERED_CORE_CAUSE_SPECIFIC_BASELINE"
    else:
        internal_status = "PASS_INTERNAL_CAUSE_SPECIFIC_BASELINE"
    methodology = {
        "saved_observed_estimator": "Kaplan-Meier 1-S(t)",
        "saved_cox_estimand": "cause-specific diabetes hazard with loss to follow-up censored",
        "feasibility_audit_observed_estimator": (
            "Aalen-Johansen cumulative incidence with all-cause death competing"
        ),
        "death_competing_event_explicitly_modeled_in_saved_baseline": False,
        "safe_reuse_scope": (
            "Saved cohort, predictions, C-index, and 5y AUC are reusable for the "
            "existing cause-specific definition if all hard checks pass. Missing "
            "derived curve/landmark files are deterministic and are recovered in "
            "the audit directory."
        ),
        "provenance_note": (
            "A missing source manifest does not invalidate matching core outputs, "
            "but it means the prior run must not be described as a fully completed run."
        ),
        "not_yet_safe_for_final_four_curve_claim": (
            "The saved observed/Cox curves are not estimator-aligned with the audit's "
            "death-competing cumulative-incidence definition."
        ),
        "next_required_decision": (
            "Retain the cause-specific endpoint consistently for all four methods, or "
            "rebuild observed and comparator curves as competing-risk CIFs."
        ),
    }
    result = {
        "status": internal_status,
        "hard_checks": int((checks_frame["severity"] == "hard").sum()),
        "hard_failures": int(len(hard_failures)),
        "soft_failures": int(len(soft_failures)),
        "baseline_dir": str(args.baseline_dir),
        "output_dir": str(args.output_dir),
        "recovery": recovery,
        "methodology": methodology,
    }
    write_json(result, args.output_dir / "baseline_reuse_audit.json")

    return_summary = [
        "## STATUS",
        internal_status,
        "## SAVED_COHORT_SUMMARY",
        saved_summary.to_csv(index=False).rstrip(),
        "## RECONSTRUCTED_COHORT_SUMMARY",
        reconstructed.to_csv(index=False).rstrip(),
        "## COX_METRICS",
        metrics.to_csv(index=False).rstrip(),
        "## CURVE_LANDMARKS",
        landmarks.to_csv(index=False).rstrip(),
        "## RECOVERY",
        json.dumps(recovery, ensure_ascii=False),
        "## BASELINE_VS_FEASIBILITY_AUDIT",
        context.to_csv(index=False).rstrip(),
        "## METHODOLOGY_LIMITATION",
        json.dumps(methodology, ensure_ascii=False),
        "## FAILED_HARD_CHECKS",
        hard_failures.to_csv(index=False).rstrip() if len(hard_failures) else "NONE",
        "## FAILED_SOFT_CHECKS",
        soft_failures.to_csv(index=False).rstrip() if len(soft_failures) else "NONE",
        "## OUTPUT_DIR",
        str(args.output_dir),
    ]
    text = "\n".join(return_summary) + "\n"
    (args.output_dir / "return_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    log("[COMPLETE] diabetes baseline reuse audit finished")
    return 0 if hard_failures.empty else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
