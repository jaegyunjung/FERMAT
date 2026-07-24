#!/usr/bin/env python3
"""Compare observed and generated clinical-event density for Task 30.

CPU only.  The script reuses the exact 1,000 patients from the completed
diabetes rollout, counts observed post-index DX/RX/PX/DTH rows in test.bin,
and compares them with the saved generated-event counts from 32 rollouts per
patient.  It does not load a model or generate new trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_INPUT_DIR = (
    TASK30 / "outputs" / "diabetes_main_rollout_20260716_1000x32"
)
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_FILE = (
    POD_ROOT
    / "task19"
    / "outputs"
    / "patient_phenotype_labels_wide"
    / "patient_phenotype_labels_wide_20180101.parquet"
)
DEFAULT_OUTPUT_DIR = (
    TASK30 / "outputs" / "diabetes_observed_generated_event_density_20260718"
)
INDEX_DATE = pd.Timestamp("2018-01-01")
HORIZON_DAYS = 1826
CLINICAL_TYPES = {1: "DX", 2: "RX", 3: "PX", 6: "DTH"}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_parquet(frame, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_typed_bin(path):
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    if raw.size % 4:
        raise ValueError(f"Typed bin size is not divisible by four: {path}")
    data = raw.reshape(-1, 4)
    sample_types = np.asarray(data[: min(len(data), 1_000_000), 3], dtype=np.int64)
    if len(sample_types) and sample_types.max() >= 20:
        raise ValueError(f"File does not look like a typed FERMAT bin: {path}")
    return data


def patient_boundaries(data, chunk_rows=5_000_000):
    if len(data) == 0:
        raise ValueError("test.bin has no rows")
    starts = [0]
    previous = int(data[0, 0])
    for chunk_start in range(0, len(data), chunk_rows):
        chunk_end = min(chunk_start + chunk_rows, len(data))
        values = np.asarray(data[chunk_start:chunk_end, 0], dtype=np.int64)
        prior = np.empty_like(values)
        prior[0] = previous
        prior[1:] = values[:-1]
        changed = np.flatnonzero(values != prior)
        starts.extend((chunk_start + changed).tolist())
        previous = int(values[-1])
    starts = np.asarray(sorted(set(starts)), dtype=np.int64)
    ends = np.r_[starts[1:], len(data)].astype(np.int64)
    patient_ids = np.asarray(data[starts, 0], dtype=np.int64)
    if len(patient_ids) != len(np.unique(patient_ids)):
        raise ValueError("Patient rows are not contiguous in test.bin")
    return {
        int(patient_id): (int(start), int(end))
        for patient_id, start, end in zip(patient_ids, starts, ends)
    }


def count_observed_rows(rows, index_age_days, followup_days):
    followup_days = float(np.clip(followup_days, 0, HORIZON_DAYS))
    ages = rows[:, 1].astype(np.int64)
    types = rows[:, 3].astype(np.int64)
    selected = (
        (ages >= int(index_age_days))
        & (ages <= int(index_age_days) + int(np.floor(followup_days)))
        & np.isin(types, list(CLINICAL_TYPES))
    )
    selected_ages = ages[selected]
    selected_types = types[selected]
    result = {
        "observed_clinical_events": int(selected.sum()),
        "observed_unique_event_days": int(len(np.unique(selected_ages))),
        "observed_same_day_excess_events": int(
            selected.sum() - len(np.unique(selected_ages))
        ),
    }
    for token_type, name in CLINICAL_TYPES.items():
        result[f"observed_{name.lower()}_events"] = int(
            np.sum(selected_types == token_type)
        )
    return result


def validate_inputs(sample, trajectories):
    sample_required = {
        "person_id",
        "patient_id_dense",
        "index_age_days",
        "sample_order",
    }
    trajectory_required = {
        "person_id",
        "rollout_index",
        "valid_generated_events",
        "generated_followup_end_day",
        "generated_death_day",
        "max_token_cap_before_horizon",
    }
    missing_sample = sorted(sample_required - set(sample.columns))
    missing_trajectory = sorted(trajectory_required - set(trajectories.columns))
    if missing_sample or missing_trajectory:
        raise ValueError(
            f"Missing sample columns={missing_sample}; trajectory columns={missing_trajectory}"
        )
    if len(sample) != 1000 or sample["person_id"].duplicated().any():
        raise ValueError("Expected exactly 1,000 unique sample patients")
    counts = trajectories.groupby("person_id").size()
    if len(trajectories) != 32000 or len(counts) != 1000 or not counts.eq(32).all():
        raise ValueError("Expected exactly 1,000 patients x 32 trajectories")
    if trajectories.duplicated(["person_id", "rollout_index"]).any():
        raise ValueError("Duplicate trajectory keys")
    if set(sample["person_id"]) != set(trajectories["person_id"]):
        raise ValueError("Sample and trajectory patient sets differ")


def build_observed_patient_table(sample, labels, data, locations):
    labels = labels.copy()
    labels["last_activity_date"] = pd.to_datetime(
        labels["last_activity_date"], errors="coerce"
    )
    patients = sample.merge(
        labels[["person_id", "last_activity_date", "has_followup_5y"]],
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    if patients["last_activity_date"].isna().any():
        raise ValueError("Missing last_activity_date for sampled patients")
    patients["observed_followup_days"] = (
        patients["last_activity_date"] - INDEX_DATE
    ).dt.days.clip(lower=0, upper=HORIZON_DAYS)
    patients["observed_complete_5y"] = (
        patients["has_followup_5y"].fillna(False).astype(bool)
        & patients["observed_followup_days"].ge(HORIZON_DAYS)
    )

    rows = []
    ordered = patients.sort_values("sample_order")
    for number, patient in enumerate(ordered.itertuples(index=False), start=1):
        location = locations.get(int(patient.patient_id_dense))
        if location is None:
            raise ValueError(
                f"patient_id_dense={int(patient.patient_id_dense)} missing from test.bin"
            )
        start, end = location
        counts = count_observed_rows(
            data[start:end],
            int(patient.index_age_days),
            float(patient.observed_followup_days),
        )
        followup = float(patient.observed_followup_days)
        events = int(counts["observed_clinical_events"])
        rows.append(
            {
                "person_id": int(patient.person_id),
                "sample_order": int(patient.sample_order),
                "patient_id_dense": int(patient.patient_id_dense),
                "index_age_days": int(patient.index_age_days),
                "last_activity_date": patient.last_activity_date,
                "observed_followup_days": followup,
                "observed_complete_5y": bool(patient.observed_complete_5y),
                **counts,
                "observed_events_per_year": (
                    events / (followup / 365.25) if followup > 0 else np.nan
                ),
                "observed_days_per_event": (
                    followup / events if events > 0 else np.nan
                ),
            }
        )
        if number == 1 or number % 100 == 0 or number == len(ordered):
            log(f"[OBSERVED COUNT] patient={number}/{len(ordered)}")
    return pd.DataFrame(rows)


def build_generated_table(trajectories):
    result = trajectories[
        [
            "person_id",
            "rollout_index",
            "valid_generated_events",
            "generated_followup_end_day",
            "generated_death_day",
            "max_token_cap_before_horizon",
        ]
    ].copy()
    result["generated_followup_days"] = pd.to_numeric(
        result["generated_followup_end_day"], errors="coerce"
    ).clip(lower=0, upper=HORIZON_DAYS)
    result["generated_clinical_events"] = pd.to_numeric(
        result["valid_generated_events"], errors="raise"
    ).astype(np.int64)
    result["generated_complete_5y_no_death"] = (
        result["generated_followup_days"].ge(HORIZON_DAYS)
        & result["generated_death_day"].isna()
        & ~result["max_token_cap_before_horizon"].fillna(False).astype(bool)
    )
    years = result["generated_followup_days"] / 365.25
    result["generated_events_per_year"] = np.divide(
        result["generated_clinical_events"],
        years,
        out=np.full(len(result), np.nan),
        where=years > 0,
    )
    result["generated_days_per_event"] = np.divide(
        result["generated_followup_days"],
        result["generated_clinical_events"],
        out=np.full(len(result), np.nan),
        where=result["generated_clinical_events"] > 0,
    )
    return result


def distribution_row(source, unit, event_count, followup_days, event_rate):
    event_count = pd.to_numeric(event_count, errors="coerce")
    followup_days = pd.to_numeric(followup_days, errors="coerce")
    event_rate = pd.to_numeric(event_rate, errors="coerce")
    return {
        "source": source,
        "unit": unit,
        "units": int(len(event_count)),
        "mean_followup_days": float(followup_days.mean()),
        "median_followup_days": float(followup_days.median()),
        "mean_events": float(event_count.mean()),
        "median_events": float(event_count.median()),
        "p25_events": float(event_count.quantile(0.25)),
        "p75_events": float(event_count.quantile(0.75)),
        "p90_events": float(event_count.quantile(0.90)),
        "p95_events": float(event_count.quantile(0.95)),
        "p99_events": float(event_count.quantile(0.99)),
        "maximum_events": int(event_count.max()),
        "mean_events_per_year": float(event_rate.mean()),
        "median_events_per_year": float(event_rate.median()),
        "p90_events_per_year": float(event_rate.quantile(0.90)),
    }


def build_distribution_summary(observed, generated):
    observed_complete = observed.loc[observed["observed_complete_5y"]]
    generated_complete = generated.loc[generated["generated_complete_5y_no_death"]]
    rows = [
        distribution_row(
            "observed_available_followup",
            "patient",
            observed["observed_clinical_events"],
            observed["observed_followup_days"],
            observed["observed_events_per_year"],
        ),
        distribution_row(
            "observed_complete_5y",
            "patient",
            observed_complete["observed_clinical_events"],
            observed_complete["observed_followup_days"],
            observed_complete["observed_events_per_year"],
        ),
        distribution_row(
            "generated_available_followup",
            "trajectory",
            generated["generated_clinical_events"],
            generated["generated_followup_days"],
            generated["generated_events_per_year"],
        ),
        distribution_row(
            "generated_complete_5y_no_death",
            "trajectory",
            generated_complete["generated_clinical_events"],
            generated_complete["generated_followup_days"],
            generated_complete["generated_events_per_year"],
        ),
    ]
    return pd.DataFrame(rows)


def build_paired_patient_table(observed, generated):
    generated_patient = (
        generated.groupby("person_id", as_index=False)
        .agg(
            generated_mean_events=("generated_clinical_events", "mean"),
            generated_median_events=("generated_clinical_events", "median"),
            generated_mean_events_per_year=("generated_events_per_year", "mean"),
            generated_median_events_per_year=("generated_events_per_year", "median"),
            generated_complete_5y_fraction=("generated_complete_5y_no_death", "mean"),
        )
    )
    paired = observed.merge(
        generated_patient, on="person_id", how="inner", validate="one_to_one"
    )
    paired["generated_to_observed_event_rate_ratio"] = np.divide(
        paired["generated_mean_events_per_year"],
        paired["observed_events_per_year"],
        out=np.full(len(paired), np.nan),
        where=paired["observed_events_per_year"] > 0,
    )
    return paired


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def interpret(distribution, observed, generated, paired):
    indexed = distribution.set_index("source")
    observed_complete = indexed.loc["observed_complete_5y"]
    generated_complete = indexed.loc["generated_complete_5y_no_death"]
    median_count_ratio = ratio(
        generated_complete["median_events"], observed_complete["median_events"]
    )
    mean_count_ratio = ratio(
        generated_complete["mean_events"], observed_complete["mean_events"]
    )
    median_rate_ratio = ratio(
        generated_complete["median_events_per_year"],
        observed_complete["median_events_per_year"],
    )
    valid_pair_ratio = paired["generated_to_observed_event_rate_ratio"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    result = {
        "status": "COMPLETE_TASK30_OBSERVED_GENERATED_EVENT_DENSITY",
        "cpu_only": True,
        "new_rollout_performed": False,
        "clinical_token_types_counted": list(CLINICAL_TYPES.values()),
        "sample_patients": int(len(observed)),
        "observed_complete_5y_patients": int(observed["observed_complete_5y"].sum()),
        "generated_trajectories": int(len(generated)),
        "generated_complete_5y_no_death_trajectories": int(
            generated["generated_complete_5y_no_death"].sum()
        ),
        "complete_5y_generated_to_observed": {
            "median_event_count_ratio": median_count_ratio,
            "mean_event_count_ratio": mean_count_ratio,
            "median_event_rate_ratio": median_rate_ratio,
        },
        "paired_patient_event_rate_ratio": {
            "patients_with_defined_ratio": int(len(valid_pair_ratio)),
            "median": float(valid_pair_ratio.median()),
            "p25": float(valid_pair_ratio.quantile(0.25)),
            "p75": float(valid_pair_ratio.quantile(0.75)),
            "p90": float(valid_pair_ratio.quantile(0.90)),
        },
        "interpretation_flags": {
            "generated_event_density_at_least_twice_observed": bool(
                np.isfinite(median_rate_ratio) and median_rate_ratio >= 2.0
            ),
            "generated_event_density_roughly_similar_to_observed": bool(
                np.isfinite(median_rate_ratio) and 0.67 <= median_rate_ratio <= 1.5
            ),
        },
        "claim_boundary": (
            "A higher generated event rate supports excessive event opportunity as a mechanism, "
            "but does not by itself isolate the time head from token-selection behavior."
        ),
    }
    return result


def run(args):
    input_dir = args.input_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    label_file = args.label_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sample_path = input_dir / "raw" / "main_sample_cohort.parquet"
    trajectory_path = input_dir / "raw" / "main_trajectories.parquet"
    bin_path = data_dir / "test.bin"
    required = [sample_path, trajectory_path, label_file, bin_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_parquet(sample_path)
    trajectories = pd.read_parquet(trajectory_path)
    validate_inputs(sample, trajectories)
    labels = pd.read_parquet(
        label_file,
        columns=["person_id", "last_activity_date", "has_followup_5y"],
    )
    labels = labels.drop_duplicates("person_id")
    data = load_typed_bin(bin_path)
    log("[START] index patient boundaries in test.bin")
    locations = patient_boundaries(data)
    log(f"[INDEXED] test patients={len(locations):,}")

    observed = build_observed_patient_table(sample, labels, data, locations)
    atomic_parquet(observed, output_dir / "observed_patient_event_counts.parquet")
    log("[RAW SAVED] observed patient event counts")
    generated = build_generated_table(trajectories)
    atomic_parquet(generated, output_dir / "generated_trajectory_event_counts.parquet")
    log("[RAW SAVED] generated trajectory event counts")

    distribution = build_distribution_summary(observed, generated)
    paired = build_paired_patient_table(observed, generated)
    summary = interpret(distribution, observed, generated, paired)
    atomic_csv(distribution, output_dir / "event_density_distribution_summary.csv")
    atomic_parquet(paired, output_dir / "paired_patient_event_density.parquet")
    atomic_json(summary, output_dir / "comparison_summary.json")

    return_text = "\n".join(
        [
            "## STATUS COMPLETE_TASK30_OBSERVED_GENERATED_EVENT_DENSITY",
            f"sample_patients {summary['sample_patients']}",
            f"observed_complete_5y_patients {summary['observed_complete_5y_patients']}",
            f"generated_trajectories {summary['generated_trajectories']}",
            "generated_complete_5y_no_death_trajectories "
            f"{summary['generated_complete_5y_no_death_trajectories']}",
            "## EVENT_DENSITY_DISTRIBUTIONS",
            distribution.to_csv(index=False).rstrip(),
            "## COMPLETE_5Y_RATIOS",
            json.dumps(summary["complete_5y_generated_to_observed"], ensure_ascii=False),
            "## PAIRED_PATIENT_RATIOS",
            json.dumps(summary["paired_patient_event_rate_ratio"], ensure_ascii=False),
            "## INTERPRETATION_FLAGS",
            json.dumps(summary["interpretation_flags"], ensure_ascii=False),
            "## OUTPUT_DIR",
            str(output_dir),
        ]
    ) + "\n"
    (output_dir / "return_summary.txt").write_text(return_text, encoding="utf-8")
    print(return_text, end="", flush=True)


def self_test():
    rows = np.asarray(
        [
            [7, 100, 1, 1],
            [7, 100, 2, 4],
            [7, 101, 3, 2],
            [7, 102, 4, 3],
            [7, 103, 5, 6],
            [7, 104, 6, 7],
        ],
        dtype=np.uint32,
    )
    counts = count_observed_rows(rows, index_age_days=100, followup_days=3)
    locations = patient_boundaries(rows, chunk_rows=2)
    if locations != {7: (0, 6)}:
        raise AssertionError("Patient boundary indexing failed")
    if counts["observed_clinical_events"] != 4:
        raise AssertionError("Clinical event type filter failed")
    if counts["observed_unique_event_days"] != 4:
        raise AssertionError("Unique event-day count failed")
    generated = pd.DataFrame(
        {
            "person_id": [1, 1],
            "rollout_index": [0, 1],
            "valid_generated_events": [10, 20],
            "generated_followup_end_day": [1826.0, 100.0],
            "generated_death_day": [np.nan, 100.0],
            "max_token_cap_before_horizon": [False, False],
        }
    )
    result = build_generated_table(generated)
    if int(result["generated_complete_5y_no_death"].sum()) != 1:
        raise AssertionError("Generated complete-followup classification failed")
    log("SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
