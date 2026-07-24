#!/usr/bin/env python3
"""Run five diagnosis-time shift experiments with FERMAT embeddings plus Cox.

The same selected patients and the same visible 2,048 row identities are kept
at every shift.  Only the ages of DX rows on the first source-diagnosis day are
changed.  No future trajectory is generated in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from scripts.run_snuh_task20_cox_survival_models import (
    concordance_index,
    fit_cox_model,
    fit_transformer,
    horizon_auc,
    transform_features,
)
from scripts.run_snuh_task30_counterfactual_risk_curves import (
    breslow_baseline_curve,
    cumulative_hazard_at_days,
    deterministic_cap,
    embed_row_items,
    load_model,
    load_split_data,
    load_task_data,
    parquet_columns,
    risk_matrix,
    score_embeddings,
)


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
TASK30 = POD_ROOT / "task30"
DEFAULT_CONFIG = TASK30 / "config" / "snuh_task30_five_pathway_fermat_cox_shift.json"
DEFAULT_RANGE_DIR = TASK30 / "outputs" / "continuous_shift_ranges_20260718_075041"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_EMBEDDING_FILE = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "fermat_embeddings_2018_5y_block2048_best"
    / "fermat_embeddings_20180101_5y_last.parquet"
)
DEFAULT_SURVIVAL_CACHE = (
    POD_ROOT
    / "task23"
    / "outputs"
    / "first_phenotype_dates_20180101_31phenotypes_full.parquet"
)
DEFAULT_CKPT = (
    POD_ROOT
    / "task21"
    / "outputs"
    / "block2048_full_10l640_20260629"
    / "block_2048"
    / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = TASK30 / "outputs" / "five_pathway_fermat_cox_shift_20260718"
DEFAULT_HOST = "pg-2vge6u.vpc-cdb-kr.gov-ntruss.com"
DEFAULT_SCHEMA = "cdm2024_official"
LANDMARKS = [365, 1095, 1826]
RUNNER_VERSION = "20260718_paired_embedding_delta_v3"
DIABETES_SAMPLE_STRATA = {
    "measured_no_abnormality": 64,
    "recent_abnormality": 64,
}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--range-dir", type=Path, default=DEFAULT_RANGE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--embedding-file", type=Path, default=DEFAULT_EMBEDDING_FILE)
    parser.add_argument("--survival-cache", type=Path, default=DEFAULT_SURVIVAL_CACHE)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--min-train-events", type=int, default=20)
    parser.add_argument("--min-test-events", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--index-date", default="2018-01-01")
    parser.add_argument("--horizon", default="5y", choices=["5y"])
    parser.add_argument("--db-end-date", default="2025-02-05")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="cdm")
    parser.add_argument("--user", default="jaegyun_jung")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--sslmode", default="disable")
    parser.add_argument("--statement-timeout", default="0")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def require_torch():
    if torch is None:
        raise RuntimeError("torch is required in the FERMAT Pod")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def safe_name(value):
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))


def find_range_dir(path):
    path = path.expanduser()
    required = path / "gpu_continuous_pilot_samples.csv"
    if required.is_file():
        return path.resolve()
    candidates = sorted(
        (TASK30 / "outputs").glob("continuous_shift_ranges_*"),
        key=lambda item: item.stat().st_mtime,
    )
    candidates = [
        item for item in candidates if (item / "gpu_continuous_pilot_samples.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(required)
    selected = candidates[-1].resolve()
    log(f"[AUTO-SELECT] range_dir={selected}")
    return selected


def normalize_args(args):
    for name in (
        "config_file",
        "data_dir",
        "label_dir",
        "embedding_file",
        "survival_cache",
        "ckpt",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.range_dir = find_range_dir(args.range_dir)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "index_date",
        "horizon",
        "horizon_days",
        "split",
        "block_size",
        "shift_days",
        "patients_per_pathway",
        "embedding_batch_size",
        "max_attention_cells",
        "embedding_check_patients",
        "embedding_check_atol",
        "pathways",
        "future_rollout_raw_output_contract",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    if config["index_date"] != "2018-01-01" or int(config["horizon_days"]) != 1826:
        raise ValueError("This run is fixed to 2018-01-01 and five years")
    if sorted(int(value) for value in config["shift_days"]) != list(range(-360, 361, 90)):
        raise ValueError("Shift grid must be -360..360 in 90-day increments")
    pathways = pd.DataFrame(config["pathways"])
    if len(pathways) != 5 or pathways["pathway_id"].duplicated().any():
        raise ValueError("Exactly five unique pathways are required")
    return config


def resume_fingerprint_matches(saved, current, allow_runner_version_mismatch=False):
    saved = dict(saved)
    current = dict(current)
    # A corrected bundle is extracted to a new directory, so the absolute config
    # path changes even when its verified contents and every analysis input do not.
    saved.pop("config", None)
    current.pop("config", None)
    if allow_runner_version_mismatch:
        saved.pop("runner_version", None)
        current.pop("runner_version", None)
    return saved == current


def prepare_output(args, config):
    fingerprint = {
        "config": str(args.config_file),
        "config_sha256": sha256_file(args.config_file),
        "range_dir": str(args.range_dir),
        "data_dir": str(args.data_dir),
        "embedding_file": str(args.embedding_file),
        "survival_cache": str(args.survival_cache),
        "ckpt": str(args.ckpt),
        "device": args.device,
        "dtype": args.dtype,
        "analysis_name": config["analysis_name"],
        "runner_version": RUNNER_VERSION,
    }
    run_path = args.output_dir / "run_config.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"{args.output_dir} exists and is not empty; pass --resume"
            )
        if not run_path.is_file():
            raise RuntimeError(f"Cannot resume without {run_path}")
        saved_fingerprint = json.loads(run_path.read_text(encoding="utf-8"))
        arm_dir = args.output_dir / "raw" / "arms"
        existing_arm_outputs = bool(arm_dir.is_dir() and next(arm_dir.glob("*.parquet"), None))
        saved_runner_version = saved_fingerprint.get("runner_version")
        matches = resume_fingerprint_matches(
            saved_fingerprint,
            fingerprint,
            allow_runner_version_mismatch=not existing_arm_outputs,
        )
        if not matches:
            raise RuntimeError("Resume fingerprint does not match")
        if saved_runner_version != RUNNER_VERSION:
            write_json(fingerprint, run_path)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(fingerprint, run_path)
    for relative in (
        "models",
        "raw/arms",
        "raw/edited_tokens",
        "raw/curves",
        "raw/original_recomputed",
    ):
        (args.output_dir / relative).mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "current_stage_generates_future_tokens": False,
            "current_stage_saved_raw_outputs": [
                "Cox risk-set eligibility and exclusion reason for every stable-range candidate",
                "original-sample retention and replacement counts by pathway and stratum",
                "edited diagnosis token IDs, types, original dates, and shifted dates",
                "patient embeddings for every pathway, source role, and shift",
                "patient Cox scores and 1y, 3y, and 5y risks",
                "daily population mean risk curves",
                "paired recomputed-original and recomputed-edited embedding change anchored to the saved original embedding",
            ],
            "mandatory_for_any_later_rollout": config[
                "future_rollout_raw_output_contract"
            ],
        },
        args.output_dir / "raw_output_contract.json",
    )


def load_samples(args, config):
    samples = pd.read_csv(args.range_dir / "gpu_continuous_pilot_samples.csv")
    ranges = pd.read_parquet(
        args.range_dir / "raw" / "patient_source_stable_ranges.parquet"
    )
    expected = {item["pathway_id"] for item in config["pathways"]}
    if set(samples["pathway_id"]) != expected:
        raise ValueError("Sample file does not contain the configured five pathways")
    counts = samples.groupby("pathway_id")["person_id"].nunique()
    if not counts.eq(int(config["patients_per_pathway"])).all():
        raise ValueError(f"Each pathway must have 128 patients: {counts.to_dict()}")
    if samples.duplicated(["pathway_id", "person_id"]).any():
        raise ValueError("Duplicate pathway/person sample rows")
    return samples, ranges


def stable_rank(pathway_id, person_id, seed):
    payload = f"{int(seed)}:{pathway_id}:{int(person_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_cox_candidate_pool(ranges, config):
    parts = []
    minimum_shift = min(int(value) for value in config["shift_days"])
    maximum_shift = max(int(value) for value in config["shift_days"])
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        selection_role = "joint" if item.get("source_b") else "source_a"
        candidates = ranges.loc[
            ranges["pathway_id"].eq(pathway_id)
            & ranges["source_role"].eq(selection_role)
            & ranges["original_context_valid"].astype(bool)
            & pd.to_numeric(ranges["stable_min_delta_days"], errors="coerce").le(
                minimum_shift
            )
            & pd.to_numeric(ranges["stable_max_delta_days"], errors="coerce").ge(
                maximum_shift
            )
        ].copy()
        if candidates["person_id"].duplicated().any():
            raise ValueError(f"Duplicate candidate patients for {pathway_id}")
        if len(candidates) < int(config["patients_per_pathway"]):
            raise RuntimeError(
                f"Only {len(candidates)} stable-range candidates for {pathway_id}"
            )
        candidates["selection_role"] = selection_role
        parts.append(candidates)
    return pd.concat(parts, ignore_index=True)


def select_cox_eligible_samples(
    original_samples, candidate_pool, cox_by_target, config, seed
):
    selected_parts = []
    audit_rows = []
    patient_audit_parts = []
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        target = item["target"]
        candidates = candidate_pool.loc[
            candidate_pool["pathway_id"].eq(pathway_id)
        ].copy()
        target_audit = cox_by_target[target]["candidate_audit"][
            [
                "person_id",
                "cox_risk_set_eligible",
                "cox_risk_set_exclusion_reason",
            ]
        ].copy()
        candidates = candidates.merge(
            target_audit,
            on="person_id",
            how="left",
            validate="one_to_one",
        )
        if candidates["cox_risk_set_eligible"].isna().any():
            raise RuntimeError(f"Missing Cox risk-set audit rows for {pathway_id}")
        candidates["cox_risk_set_eligible"] = candidates[
            "cox_risk_set_eligible"
        ].astype(bool)
        eligible_ids = set(
            candidates.loc[candidates["cox_risk_set_eligible"], "person_id"].astype(
                np.int64
            )
        )
        initial_ids = set(
            original_samples.loc[
                original_samples["pathway_id"].eq(pathway_id), "person_id"
            ].astype(np.int64)
        )
        candidates["in_original_continuous_sample"] = candidates["person_id"].isin(
            initial_ids
        )
        candidates["selected_for_cox_shift"] = False

        if pathway_id == "diabetes_to_ckd":
            strata = DIABETES_SAMPLE_STRATA
        else:
            strata = {"unstratified": int(config["patients_per_pathway"])}

        pathway_selected = []
        for stratum, requested in strata.items():
            if stratum == "unstratified":
                group = candidates.loc[candidates["cox_risk_set_eligible"]].copy()
                rank_key = pathway_id
            else:
                group = candidates.loc[
                    candidates["cox_risk_set_eligible"]
                    & candidates["renal_screen_status"].eq(stratum)
                ].copy()
                rank_key = f"{pathway_id}:{stratum}"
            requested = int(requested)
            if len(group) < requested:
                raise RuntimeError(
                    f"Only {len(group)} Cox-eligible stable candidates for "
                    f"{pathway_id}/{stratum}; need {requested}"
                )
            group["selection_hash"] = group["person_id"].map(
                lambda value: stable_rank(rank_key, value, seed)
            )
            chosen = group.sort_values(["selection_hash", "person_id"]).head(
                requested
            )
            chosen["pilot_stratum"] = stratum
            pathway_selected.append(chosen)
            chosen_ids = set(chosen["person_id"].astype(np.int64))
            initial_stratum = original_samples.loc[
                original_samples["pathway_id"].eq(pathway_id)
                & original_samples["pilot_stratum"].eq(stratum)
            ]
            if stratum == "unstratified" and initial_stratum.empty:
                initial_stratum = original_samples.loc[
                    original_samples["pathway_id"].eq(pathway_id)
                ]
            initial_stratum_ids = set(initial_stratum["person_id"].astype(np.int64))
            audit_rows.append(
                {
                    "pathway_id": pathway_id,
                    "target": target,
                    "pilot_stratum": stratum,
                    "stable_candidates": int(
                        len(candidates)
                        if stratum == "unstratified"
                        else candidates["renal_screen_status"].eq(stratum).sum()
                    ),
                    "cox_risk_set_eligible_candidates": int(len(group)),
                    "original_sample_patients": int(len(initial_stratum_ids)),
                    "original_sample_cox_eligible": int(
                        len(initial_stratum_ids & eligible_ids)
                    ),
                    "final_selected_patients": int(len(chosen)),
                    "retained_from_original_sample": int(
                        len(chosen_ids & initial_stratum_ids)
                    ),
                    "replacement_patients": int(
                        len(chosen_ids - initial_stratum_ids)
                    ),
                    "selection_used_future_target_event_or_time": False,
                    "selection_used_cox_risk_set_eligibility": True,
                }
            )

        selected = pd.concat(pathway_selected, ignore_index=True)
        selected["sample_rank"] = np.arange(1, len(selected) + 1)
        selected["sampling_used_future_target_outcome"] = False
        candidates.loc[
            candidates["person_id"].isin(selected["person_id"]),
            "selected_for_cox_shift",
        ] = True
        selected_parts.append(selected)
        patient_audit_parts.append(candidates)

    selected = pd.concat(selected_parts, ignore_index=True)
    counts = selected.groupby("pathway_id")["person_id"].nunique()
    if not counts.eq(int(config["patients_per_pathway"])).all():
        raise RuntimeError(f"Final Cox-eligible sample counts are invalid: {counts.to_dict()}")
    return (
        selected.sort_values(["pathway_id", "sample_rank"]).reset_index(drop=True),
        pd.DataFrame(audit_rows),
        pd.concat(patient_audit_parts, ignore_index=True),
    )


def load_source_token_sets(args, config):
    concept_map = pd.read_csv(args.label_dir / "phenotype_group_concept_map.csv")
    registry = pd.read_csv(args.data_dir / "token_registry.csv", dtype={"token_key": str})
    lookup = dict(
        zip(
            registry["token_key"].astype(str),
            pd.to_numeric(registry["token_id"], errors="raise").astype(np.int64),
        )
    )
    sources = sorted(
        {
            source
            for item in config["pathways"]
            for source in (item["source_a"], item.get("source_b"))
            if source
        }
    )
    token_sets = {}
    mapping_rows = []
    for source in sources:
        concepts = (
            pd.to_numeric(
                concept_map.loc[
                    concept_map["phenotype"].eq(source), "condition_concept_id"
                ],
                errors="raise",
            )
            .astype(np.int64)
            .unique()
        )
        tokens = sorted(
            {
                int(lookup[f"DX:{int(concept)}"])
                for concept in concepts
                if f"DX:{int(concept)}" in lookup
            }
        )
        if not tokens:
            raise ValueError(f"No source DX tokens for {source}")
        token_sets[source] = set(tokens)
        for token in tokens:
            mapping_rows.append(
                {"source": source, "stored_token_id": token, "model_token_id": token + 1}
            )
    return token_sets, pd.DataFrame(mapping_rows)


def pathway_roles(item):
    roles = [("source_a", item["source_a"])]
    if item.get("source_b"):
        roles.append(("source_b", item["source_b"]))
    return roles


def build_arm_plan(samples, ranges, config):
    rows = []
    for item in config["pathways"]:
        pathway_id = item["pathway_id"]
        selected = samples.loc[samples["pathway_id"].eq(pathway_id)].copy()
        for source_role, source in pathway_roles(item):
            stable = ranges.loc[
                ranges["pathway_id"].eq(pathway_id)
                & ranges["source_role"].eq(source_role),
                ["person_id", "stable_min_delta_days", "stable_max_delta_days"],
            ]
            # The pilot-sample CSV also carries the range used during sampling.
            # Remove those columns before attaching the source-role-specific range;
            # otherwise pandas suffixes both copies with _x/_y.
            selected_without_ranges = selected.drop(
                columns=[column for column in selected.columns if column.startswith("stable_")]
            )
            selected_role = selected_without_ranges.merge(
                stable, on="person_id", how="left", validate="one_to_one"
            )
            if selected_role[["stable_min_delta_days", "stable_max_delta_days"]].isna().any().any():
                raise ValueError(f"Missing stable ranges for {pathway_id}/{source_role}")
            for shift in config["shift_days"]:
                shift = int(shift)
                supported = (
                    selected_role["stable_min_delta_days"].le(shift)
                    & selected_role["stable_max_delta_days"].ge(shift)
                )
                if not supported.all():
                    raise ValueError(
                        f"Shift {shift} is not stable for all {pathway_id}/{source_role} patients"
                    )
                rows.append(
                    {
                        "pathway_id": pathway_id,
                        "source_role": source_role,
                        "source": source,
                        "target": item["target"],
                        "shift_days": shift,
                        "patients": int(len(selected_role)),
                    }
                )
    plan = pd.DataFrame(rows)
    if int(plan["patients"].sum()) != 6912:
        raise ValueError(f"Expected 6,912 patient-arm rows, got {int(plan['patients'].sum())}")
    return plan


def candidate_id_hash(candidate_ids):
    values = sorted(set(int(value) for value in candidate_ids))
    payload = ",".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_candidate_risk_set_audit(
    args, target, candidate_ids, eligible_ids, label_path
):
    candidate_ids = sorted(set(int(value) for value in candidate_ids))
    prior_column = f"prior__{target}"
    labels = pd.read_parquet(
        label_path,
        columns=[
            "person_id",
            "split",
            "index_date",
            "last_activity_date",
            "has_pre_index_washout",
            prior_column,
        ],
    )
    labels = labels.loc[labels["person_id"].isin(candidate_ids)].copy()
    embeddings = pd.read_parquet(
        args.embedding_file,
        columns=["person_id", "split", "has_embedding_sequence"],
    )
    embeddings = embeddings.loc[embeddings["person_id"].isin(candidate_ids)].copy()
    embeddings = embeddings.rename(
        columns={"has_embedding_sequence": "candidate_has_embedding_sequence"}
    )
    audit = pd.DataFrame({"person_id": candidate_ids})
    audit = audit.merge(labels, on="person_id", how="left", validate="one_to_one")
    audit = audit.merge(
        embeddings,
        on=["person_id", "split"],
        how="left",
        validate="one_to_one",
    )
    audit["cox_risk_set_eligible"] = audit["person_id"].isin(eligible_ids)
    index_date = pd.to_datetime(audit["index_date"], errors="coerce")
    last_activity = pd.to_datetime(audit["last_activity_date"], errors="coerce")
    reasons = np.full(len(audit), "other_target_task_exclusion", dtype=object)
    reasons[audit["cox_risk_set_eligible"].to_numpy(dtype=bool)] = "eligible"
    reasons[audit["split"].isna().to_numpy()] = "missing_target_label_row"
    reasons[
        audit["split"].notna().to_numpy()
        & ~audit["split"].eq("test").to_numpy()
    ] = "not_in_test_split"
    reasons[
        audit["split"].eq("test").to_numpy()
        & ~audit["candidate_has_embedding_sequence"].fillna(False).astype(bool).to_numpy()
    ] = "missing_usable_original_embedding"
    reasons[
        audit["split"].eq("test").to_numpy()
        & audit["candidate_has_embedding_sequence"].fillna(False).astype(bool).to_numpy()
        & ~audit["has_pre_index_washout"].fillna(False).astype(bool).to_numpy()
    ] = "insufficient_pre_index_washout"
    reasons[
        audit["split"].eq("test").to_numpy()
        & audit["candidate_has_embedding_sequence"].fillna(False).astype(bool).to_numpy()
        & audit["has_pre_index_washout"].fillna(False).astype(bool).to_numpy()
        & audit[prior_column].fillna(False).astype(bool).to_numpy()
    ] = "target_present_before_index"
    reasons[
        audit["split"].eq("test").to_numpy()
        & audit["candidate_has_embedding_sequence"].fillna(False).astype(bool).to_numpy()
        & audit["has_pre_index_washout"].fillna(False).astype(bool).to_numpy()
        & ~audit[prior_column].fillna(False).astype(bool).to_numpy()
        & (last_activity.le(index_date) | last_activity.isna()).to_numpy()
    ] = "no_positive_post_index_followup"
    reasons[audit["cox_risk_set_eligible"].to_numpy(dtype=bool)] = "eligible"
    audit["cox_risk_set_exclusion_reason"] = reasons
    audit["target"] = target
    return audit


def fit_or_load_cox(args, target, candidate_ids):
    model_dir = args.output_dir / "models" / safe_name(target)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cox_model.npz"
    transformer_path = model_dir / "transformer.json"
    metrics_path = model_dir / "metrics.json"
    selected_path = model_dir / "selected_original_embeddings.parquet"
    audit_path = model_dir / "candidate_risk_set_eligibility.parquet"
    if all(
        path.is_file()
        for path in (
            model_path,
            transformer_path,
            metrics_path,
            selected_path,
            audit_path,
        )
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_hash = candidate_id_hash(candidate_ids)
        if metrics.get("candidate_ids_sha256") != expected_hash:
            raise RuntimeError(
                f"Cached Cox candidate set does not match for {target}; "
                "use a new output directory"
            )
        saved = np.load(model_path)
        return {
            "target": target,
            "beta": saved["beta"],
            "baseline": saved["baseline"],
            "days": saved["days"],
            "transformer": json.loads(transformer_path.read_text(encoding="utf-8")),
            "metrics": metrics,
            "selected": pd.read_parquet(selected_path),
            "candidate_audit": pd.read_parquet(audit_path),
        }

    args.phenotype = target
    task, emb_cols, input_paths = load_task_data(args)
    train = deterministic_cap(
        task.loc[task["split"].eq("train")].copy(),
        args.max_train_rows,
        args.random_seed,
    )
    val = deterministic_cap(
        task.loc[task["split"].eq("val")].copy(),
        args.max_eval_rows,
        args.random_seed + 1,
    )
    test = deterministic_cap(
        task.loc[task["split"].eq("test")].copy(),
        args.max_eval_rows,
        args.random_seed + 2,
    )
    if int(train["event"].sum()) < args.min_train_events:
        raise RuntimeError(f"Too few training events for {target}")
    if int(val["event"].sum()) == 0 or int(test["event"].sum()) < args.min_test_events:
        raise RuntimeError(f"Too few validation/test events for {target}")
    transformer = fit_transformer(train, emb_cols, [], [])
    train_x, _ = transform_features(train, transformer)
    val_x, _ = transform_features(val, transformer)
    test_x, _ = transform_features(test, transformer)
    started = time.time()
    beta, best_epoch, best_val_loss, history = fit_cox_model(
        train_x,
        train["duration_days"].to_numpy(dtype=np.float32).copy(),
        train["event"].to_numpy(dtype=np.int8).copy(),
        val_x,
        val["duration_days"].to_numpy(dtype=np.float32).copy(),
        val["event"].to_numpy(dtype=np.int8).copy(),
        args,
    )
    train_score = train_x @ beta
    test_score = test_x @ beta
    event_times, cumulative = breslow_baseline_curve(
        train["duration_days"], train["event"], train_score, 1826
    )
    days = np.arange(1827, dtype=np.int32)
    baseline = cumulative_hazard_at_days(event_times, cumulative, days)
    candidate_ids = set(int(value) for value in candidate_ids)
    selected = test.loc[test["person_id"].isin(candidate_ids), [
        "person_id",
        "patient_id_dense",
        "index_age_days",
        "duration_days",
        "event",
        *emb_cols,
    ]].copy()
    eligible_ids = set(selected["person_id"].astype(np.int64))
    candidate_audit = build_candidate_risk_set_audit(
        args,
        target,
        candidate_ids,
        eligible_ids,
        input_paths["label_path"],
    )
    atomic_parquet(selected, selected_path)
    atomic_parquet(candidate_audit, audit_path)
    metrics = {
        "target": target,
        "train_rows": int(len(train)),
        "train_events": int(train["event"].sum()),
        "val_rows": int(len(val)),
        "val_events": int(val["event"].sum()),
        "test_rows": int(len(test)),
        "test_events": int(test["event"].sum()),
        "test_c_index": float(concordance_index(test["duration_days"], test["event"], test_score)),
        "test_5y_auc": float(horizon_auc(test["duration_days"], test["event"], test_score, 1826)[0]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "fit_seconds": float(time.time() - started),
        "embedding_features": int(len(emb_cols)),
        "candidate_patients": int(len(candidate_ids)),
        "candidate_cox_risk_set_eligible": int(len(eligible_ids)),
        "candidate_cox_risk_set_ineligible": int(len(candidate_ids - eligible_ids)),
        "candidate_ids_sha256": candidate_id_hash(candidate_ids),
        "candidate_selection_used_future_target_event_or_time": False,
        "inputs": input_paths,
    }
    np.savez_compressed(
        model_path,
        beta=beta,
        event_times=event_times,
        cumulative_baseline_hazard=cumulative,
        days=days,
        baseline=baseline,
    )
    write_json(transformer, transformer_path)
    write_json(metrics, metrics_path)
    atomic_csv(pd.DataFrame(history), model_dir / "training_history.csv")
    log(
        f"[COX SAVED] target={target} train_events={metrics['train_events']} "
        f"test_events={metrics['test_events']} c_index={metrics['test_c_index']:.4f}"
    )
    return {
        "target": target,
        "beta": beta,
        "baseline": baseline,
        "days": days,
        "transformer": transformer,
        "metrics": metrics,
        "selected": selected,
        "candidate_audit": candidate_audit,
    }


def edit_first_diagnosis_day(full_rows, index_age, token_ids, shift_days, block_size):
    full_rows = np.asarray(full_rows).copy()
    ages = full_rows[:, 1].astype(np.int64)
    tokens = full_rows[:, 2].astype(np.int64)
    token_mask = np.isin(tokens, list(token_ids))
    if not token_mask.any():
        raise ValueError("Source diagnosis token missing")
    first_age = int(ages[token_mask].min())
    moved_ids = np.flatnonzero(token_mask & (ages == first_age)).astype(np.int64)
    original_order = np.argsort(ages, kind="stable")
    original_ids = original_order[-int(block_size):]
    shifted_ages = ages.copy()
    shifted_ages[moved_ids] += int(shift_days)
    if shifted_ages[moved_ids].min() < 0 or shifted_ages[moved_ids].max() >= int(index_age):
        raise ValueError("Shifted diagnosis is outside the pre-index history")
    changed_order = np.argsort(shifted_ages, kind="stable")
    changed_ids = changed_order[-int(block_size):]
    if not np.array_equal(np.sort(original_ids), np.sort(changed_ids)):
        raise RuntimeError("Visible 2048 row identities changed")
    if not np.all(np.isin(moved_ids, original_ids)):
        raise RuntimeError("Moved first-day diagnosis row is not model-visible")
    edited = full_rows[changed_ids].copy()
    edited[:, 1] = shifted_ages[changed_ids]
    token_rows = []
    for row_id in moved_ids:
        token_rows.append(
            {
                "source_stored_token_id": int(full_rows[row_id, 2]),
                "source_model_token_id": int(full_rows[row_id, 2]) + 1,
                "source_token_type_id": int(full_rows[row_id, 3]),
                "original_age_days": int(full_rows[row_id, 1]),
                "shifted_age_days": int(shifted_ages[row_id]),
                "original_days_before_index": int(index_age - full_rows[row_id, 1]),
                "shifted_days_before_index": int(index_age - shifted_ages[row_id]),
                "original_row_identity": int(row_id),
            }
        )
    return edited, token_rows


def full_preindex_rows(split_data, patient_id_dense, index_age):
    location = split_data["index"].get(int(patient_id_dense))
    if location is None:
        raise ValueError(f"patient_id_dense={patient_id_dense} missing from test.bin")
    start, length = location
    rows = split_data["data"][start : start + length]
    rows = rows[rows[:, 1].astype(np.int64) < int(index_age)]
    if len(rows) == 0:
        raise ValueError("Empty pre-index history")
    return np.asarray(rows).copy()


def original_visible_rows(full_rows, block_size):
    full_rows = np.asarray(full_rows)
    ages = full_rows[:, 1].astype(np.int64)
    if np.any(np.diff(ages) < 0):
        raise RuntimeError("Patient rows in test.bin are not ordered by age")
    return full_rows[-int(block_size):].copy()


def selected_for_target(cox_artifact):
    frame = cox_artifact["selected"].copy()
    emb_cols = list(cox_artifact["transformer"]["numeric"])
    return frame, emb_cols


def anchor_saved_embedding_with_paired_delta(
    saved_original, recomputed_original, recomputed_edited
):
    saved_original = np.asarray(saved_original, dtype=np.float32)
    recomputed_original = np.asarray(recomputed_original, dtype=np.float32)
    recomputed_edited = np.asarray(recomputed_edited, dtype=np.float32)
    if not (
        saved_original.shape
        == recomputed_original.shape
        == recomputed_edited.shape
    ):
        raise ValueError("Paired embedding arrays have different shapes")
    if not (
        np.isfinite(saved_original).all()
        and np.isfinite(recomputed_original).all()
        and np.isfinite(recomputed_edited).all()
    ):
        raise ValueError("Paired embedding arrays contain non-finite values")
    delta = recomputed_edited - recomputed_original
    return saved_original + delta, delta


def run_arm(
    args,
    config,
    item,
    source_role,
    source,
    shift_days,
    selected_samples,
    token_ids,
    cox,
    split_data,
    model,
):
    arm_id = f"{item['pathway_id']}__{source_role}__shift_{int(shift_days):+05d}d"
    arm_path = args.output_dir / "raw" / "arms" / f"{safe_name(arm_id)}.parquet"
    token_path = args.output_dir / "raw" / "edited_tokens" / f"{safe_name(arm_id)}.parquet"
    curve_path = args.output_dir / "raw" / "curves" / f"{safe_name(arm_id)}.parquet"
    if args.resume and arm_path.is_file() and token_path.is_file() and curve_path.is_file():
        arm = pd.read_parquet(arm_path)
        if len(arm) != int(config["patients_per_pathway"]):
            raise RuntimeError(f"Invalid resumed arm rows: {arm_path}")
        log(f"[RESUME] arm={arm_id}")
        return arm_path, token_path, curve_path, 0.0

    target_frame, emb_cols = selected_for_target(cox)
    target_frame = target_frame.rename(
        columns={"patient_id_dense": "target_patient_id_dense"}
    )
    patients = selected_samples.merge(
        target_frame,
        on="person_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_target"),
    ).sort_values("sample_rank")
    if patients[emb_cols].isna().any().any():
        raise RuntimeError(f"Missing original embedding for arm={arm_id}")
    if not patients["patient_id_dense"].astype(np.int64).eq(
        patients["target_patient_id_dense"].astype(np.int64)
    ).all():
        raise RuntimeError(f"patient_id_dense mismatch for arm={arm_id}")
    original_embeddings = patients[emb_cols].to_numpy(dtype=np.float32)

    original_items = []
    edited_items = []
    token_rows = []
    for patient in patients.itertuples(index=False):
        dense_id = int(patient.patient_id_dense)
        index_age = int(patient.index_age_days)
        full_rows = full_preindex_rows(split_data, dense_id, index_age)
        original_visible = original_visible_rows(full_rows, int(config["block_size"]))
        edited, moved = edit_first_diagnosis_day(
            full_rows,
            index_age,
            token_ids,
            int(shift_days),
            int(config["block_size"]),
        )
        original_items.append({"rows": original_visible})
        edited_items.append({"rows": edited})
        for moved_index, moved_row in enumerate(moved):
            token_rows.append(
                {
                    "arm_id": arm_id,
                    "pathway_id": item["pathway_id"],
                    "source_role": source_role,
                    "source": source,
                    "target": item["target"],
                    "shift_days": int(shift_days),
                    "person_id": int(patient.person_id),
                    "patient_id_dense": dense_id,
                    "moved_token_number": moved_index,
                    **moved_row,
                }
            )

    started = time.time()
    original_cache_path = (
        args.output_dir
        / "raw"
        / "original_recomputed"
        / f"{safe_name(item['pathway_id'])}.parquet"
    )
    if original_cache_path.is_file():
        original_cache = pd.read_parquet(original_cache_path)
        original_cache = patients[["person_id"]].merge(
            original_cache,
            on="person_id",
            how="left",
            validate="one_to_one",
        )
        if original_cache[emb_cols].isna().any().any():
            raise RuntimeError(
                f"Invalid recomputed-original cache for {item['pathway_id']}"
            )
        recomputed_original = original_cache[emb_cols].to_numpy(dtype=np.float32)
    else:
        recomputed_original = embed_row_items(
            model,
            original_items,
            args.device,
            args.dtype,
            int(config["embedding_batch_size"]),
            int(config["max_attention_cells"]),
        )
        original_cache = pd.DataFrame(recomputed_original, columns=emb_cols)
        original_cache.insert(
            0, "person_id", patients["person_id"].to_numpy(dtype=np.int64)
        )
        atomic_parquet(original_cache, original_cache_path)
    if int(shift_days) == 0:
        recomputed_edited = recomputed_original.copy()
    else:
        recomputed_edited = embed_row_items(
            model,
            edited_items,
            args.device,
            args.dtype,
            int(config["embedding_batch_size"]),
            int(config["max_attention_cells"]),
        )
    # The saved embeddings were produced in different BF16 batch geometry.
    # Anchor absolute risk at that saved value, and apply only the paired
    # within-run change caused by moving the diagnosis date.
    edited_embeddings, embedding_delta = anchor_saved_embedding_with_paired_delta(
        original_embeddings,
        recomputed_original,
        recomputed_edited,
    )
    original_score = score_embeddings(
        original_embeddings, emb_cols, cox["transformer"], cox["beta"]
    )
    edited_score = score_embeddings(
        edited_embeddings, emb_cols, cox["transformer"], cox["beta"]
    )
    original_risk = risk_matrix(original_score, cox["baseline"])
    edited_risk = risk_matrix(edited_score, cox["baseline"])

    arm = patients[
        ["person_id", "sample_rank", "pilot_stratum", "duration_days", "event"]
    ].copy()
    arm.insert(0, "arm_id", arm_id)
    arm["pathway_id"] = item["pathway_id"]
    arm["source_role"] = source_role
    arm["source"] = source
    arm["target"] = item["target"]
    arm["shift_days"] = int(shift_days)
    arm["original_risk_score"] = original_score.astype(np.float32)
    arm["edited_risk_score"] = edited_score.astype(np.float32)
    arm["risk_score_difference"] = (edited_score - original_score).astype(np.float32)
    original_absolute_error = np.abs(recomputed_original - original_embeddings)
    arm["recomputed_original_mean_abs_error"] = original_absolute_error.mean(
        axis=1
    ).astype(np.float32)
    arm["recomputed_original_max_abs_error"] = original_absolute_error.max(
        axis=1
    ).astype(np.float32)
    arm["paired_embedding_delta_l2"] = np.linalg.norm(
        embedding_delta, axis=1
    ).astype(np.float32)
    arm["edited_embedding_is_saved_original_plus_paired_delta"] = True
    for day in LANDMARKS:
        arm[f"original_risk_day_{day}"] = original_risk[:, day].astype(np.float32)
        arm[f"edited_risk_day_{day}"] = edited_risk[:, day].astype(np.float32)
        arm[f"risk_difference_day_{day}"] = (
            edited_risk[:, day] - original_risk[:, day]
        ).astype(np.float32)
    arm = pd.concat(
        [
            arm.reset_index(drop=True),
            pd.DataFrame(
                edited_embeddings.astype(np.float32),
                columns=emb_cols,
            ),
        ],
        axis=1,
    )

    curve = pd.DataFrame(
        {
            "arm_id": arm_id,
            "pathway_id": item["pathway_id"],
            "source_role": source_role,
            "source": source,
            "target": item["target"],
            "shift_days": int(shift_days),
            "day": cox["days"],
            "patients": len(arm),
            "mean_original_risk": original_risk.mean(axis=0),
            "mean_edited_risk": edited_risk.mean(axis=0),
            "mean_risk_difference": (edited_risk - original_risk).mean(axis=0),
        }
    )
    atomic_parquet(arm, arm_path)
    atomic_parquet(pd.DataFrame(token_rows), token_path)
    atomic_parquet(curve, curve_path)
    elapsed = time.time() - started
    log(
        f"[RAW SAVED] arm={arm_id} patients={len(arm)} "
        f"moved_token_rows={len(token_rows)} seconds={elapsed:,.1f}"
    )
    return arm_path, token_path, curve_path, elapsed


def validate_embedding_manifest(args, config, model, checkpoint):
    manifest_path = args.embedding_file.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding extraction manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "ckpt": str(args.ckpt),
        "data_dir": str(args.data_dir),
        "output_path": str(args.embedding_file),
        "index_date": config["index_date"],
        "horizon": config["horizon"],
        "pooling": "last",
        "block_size": int(config["block_size"]),
        "n_embd": int(model.config.n_embd),
        "dtype": args.dtype,
        "checkpoint_step": int(checkpoint.get("iter", -1)),
    }
    mismatches = {}
    for key, expected_value in expected.items():
        observed = manifest.get(key)
        if key in {"ckpt", "data_dir", "output_path"} and observed is not None:
            observed = str(Path(observed).expanduser().resolve())
            expected_value = str(Path(expected_value).expanduser().resolve())
        if observed != expected_value:
            mismatches[key] = {"manifest": observed, "current": expected_value}
    result = {
        "manifest_path": str(manifest_path),
        "checked_fields": expected,
        "informational_extraction_batch_size": manifest.get("batch_size"),
        "informational_extraction_device": manifest.get("device"),
        "mismatches": mismatches,
    }
    write_json(result, args.output_dir / "embedding_manifest_check.json")
    if mismatches:
        raise RuntimeError(f"Embedding provenance mismatch: {mismatches}")
    return result


def select_unique_embedding_probe(candidate_pool, pathway_ids, requested):
    requested = int(requested)
    if requested < 1:
        raise ValueError("Embedding probe size must be positive")
    queues = {}
    for pathway_id in pathway_ids:
        queue = (
            candidate_pool.loc[candidate_pool["pathway_id"].eq(pathway_id)]
            .sort_values("person_id")
            .drop_duplicates("person_id")
            .to_dict("records")
        )
        if not queue:
            raise RuntimeError(f"No embedding probe candidates for {pathway_id}")
        queues[pathway_id] = queue
    selected = []
    used_person_ids = set()
    positions = {pathway_id: 0 for pathway_id in pathway_ids}
    while len(selected) < requested:
        made_progress = False
        for pathway_id in pathway_ids:
            queue = queues[pathway_id]
            position = positions[pathway_id]
            while (
                position < len(queue)
                and int(queue[position]["person_id"]) in used_person_ids
            ):
                position += 1
            positions[pathway_id] = position
            if position >= len(queue):
                continue
            row = queue[position]
            positions[pathway_id] += 1
            used_person_ids.add(int(row["person_id"]))
            selected.append(row)
            made_progress = True
            if len(selected) == requested:
                break
        if not made_progress:
            raise RuntimeError(
                f"Only {len(selected)} unique patients available for a "
                f"{requested}-patient embedding probe"
            )
    result = pd.DataFrame(selected)
    if result["person_id"].duplicated().any():
        raise AssertionError("Embedding probe contains duplicate person_id")
    return result


def attach_embedding_preflight_eligibility(candidate_pool, labels, embedding_meta):
    label_columns = ["person_id", "split", "age_at_index"]
    embedding_columns = [
        "person_id",
        "split",
        "patient_id_dense",
        "has_embedding_sequence",
        "sequence_length_pre_index",
    ]
    if labels.duplicated(["person_id", "split"]).any():
        raise RuntimeError("Duplicate person/split rows in the label table")
    if embedding_meta.duplicated(["person_id", "split"]).any():
        raise RuntimeError("Duplicate person/split rows in the embedding table")
    result = candidate_pool.merge(
        labels[label_columns],
        on=["person_id", "split"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        embedding_meta[embedding_columns],
        on=["person_id", "split"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_saved"),
    )
    result["label_row_available"] = result["age_at_index"].notna()
    result["embedding_row_available"] = result[
        "patient_id_dense_saved"
    ].notna()
    result["embedding_sequence_available"] = result[
        "has_embedding_sequence"
    ].eq(True)
    result["patient_id_dense_matches"] = (
        pd.to_numeric(result["patient_id_dense"], errors="coerce")
        .eq(pd.to_numeric(result["patient_id_dense_saved"], errors="coerce"))
        .fillna(False)
    )
    result["embedding_preflight_eligible"] = (
        result["label_row_available"]
        & result["embedding_row_available"]
        & result["embedding_sequence_available"]
        & result["patient_id_dense_matches"]
    )
    reasons = np.full(len(result), "eligible", dtype=object)
    reasons[~result["label_row_available"].to_numpy()] = "missing_label_row"
    reasons[
        result["label_row_available"].to_numpy()
        & ~result["embedding_row_available"].to_numpy()
    ] = "missing_embedding_row"
    reasons[
        result["embedding_row_available"].to_numpy()
        & ~result["embedding_sequence_available"].to_numpy()
    ] = "missing_embedding_sequence"
    reasons[
        result["embedding_row_available"].to_numpy()
        & result["embedding_sequence_available"].to_numpy()
        & ~result["patient_id_dense_matches"].to_numpy()
    ] = "patient_id_dense_mismatch"
    result["embedding_preflight_exclusion_reason"] = reasons
    result["index_age_days"] = np.floor(
        pd.to_numeric(result["age_at_index"], errors="coerce") * 365.25
    )
    return result


def verify_original_embeddings(
    args, config, candidate_pool, split_data, model, checkpoint
):
    check_path = args.output_dir / "original_embedding_check.json"
    manifest_check = validate_embedding_manifest(args, config, model, checkpoint)
    requested = int(config["embedding_check_patients"])
    pathway_ids = [item["pathway_id"] for item in config["pathways"]]
    candidate_ids = candidate_pool["person_id"].astype(np.int64).unique().tolist()
    label_path = args.label_dir / "patient_phenotype_labels_wide_20180101.parquet"
    labels = pd.read_parquet(
        label_path,
        columns=["person_id", "split", "age_at_index"],
    )
    labels = labels.loc[
        labels["person_id"].isin(candidate_ids) & labels["split"].eq(config["split"])
    ].copy()
    embedding_meta = pd.read_parquet(
        args.embedding_file,
        columns=[
            "person_id",
            "split",
            "patient_id_dense",
            "has_embedding_sequence",
            "sequence_length_pre_index",
        ],
        filters=[("person_id", "in", candidate_ids)],
    )
    embedding_meta = embedding_meta.loc[
        embedding_meta["split"].eq(config["split"])
    ].copy()
    preflight_candidates = attach_embedding_preflight_eligibility(
        candidate_pool,
        labels,
        embedding_meta,
    )
    atomic_parquet(
        preflight_candidates,
        args.output_dir / "raw" / "embedding_preflight_candidate_eligibility.parquet",
    )
    preflight_summary = (
        preflight_candidates.groupby(
            ["pathway_id", "embedding_preflight_exclusion_reason"],
            dropna=False,
        )
        .size()
        .rename("patients")
        .reset_index()
    )
    atomic_csv(
        preflight_summary,
        args.output_dir / "embedding_preflight_candidate_summary.csv",
    )
    log("[EMBEDDING PREFLIGHT CANDIDATES]\n" + preflight_summary.to_csv(index=False).rstrip())
    eligible_pool = preflight_candidates.loc[
        preflight_candidates["embedding_preflight_eligible"]
    ].copy()
    probe = select_unique_embedding_probe(eligible_pool, pathway_ids, requested)
    probe_ids = probe["person_id"].astype(np.int64).tolist()
    emb_cols = [
        column
        for column in parquet_columns(args.embedding_file)
        if column.startswith("emb_")
    ]
    saved_probe = pd.read_parquet(
        args.embedding_file,
        columns=[
            "person_id",
            "split",
            "patient_id_dense",
            "sequence_length_pre_index",
            *emb_cols,
        ],
        filters=[("person_id", "in", probe_ids)],
    )
    saved_probe = saved_probe.loc[saved_probe["split"].eq(config["split"])].copy()
    probe = probe.merge(
        saved_probe.drop(
            columns=[
                "split",
                "patient_id_dense",
                "sequence_length_pre_index",
            ]
        ),
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    if len(probe) != requested:
        raise RuntimeError(
            f"Embedding preflight selected {len(probe)} patients; expected {requested}"
        )
    missing_probe_columns = probe[["index_age_days", *emb_cols]].isna().sum()
    missing_probe_columns = missing_probe_columns.loc[missing_probe_columns.gt(0)]
    if len(missing_probe_columns):
        raise RuntimeError(
            "Embedding preflight probe has missing fields: "
            + missing_probe_columns.to_dict().__repr__()
        )
    rows = []
    saved = []
    metadata = []
    for patient in probe.itertuples(index=False):
        dense_id = int(patient.patient_id_dense)
        if dense_id != int(patient.patient_id_dense_saved):
            raise RuntimeError("Embedding preflight patient_id_dense mismatch")
        index_age = int(patient.index_age_days)
        full = full_preindex_rows(split_data, dense_id, index_age)
        ages = full[:, 1].astype(np.int64)
        nonmonotonic_transitions = int(np.sum(np.diff(ages) < 0))
        visible = full[-int(config["block_size"]):].copy()
        if len(visible) != int(patient.sequence_length_pre_index):
            raise RuntimeError("Embedding preflight sequence length mismatch")
        rows.append({"rows": visible})
        saved.append(
            np.asarray([getattr(patient, column) for column in emb_cols], dtype=np.float32)
        )
        metadata.append(
            {
                "pathway_id": patient.pathway_id,
                "person_id": int(patient.person_id),
                "patient_id_dense": dense_id,
                "visible_rows": int(len(visible)),
                "nonmonotonic_age_transitions": nonmonotonic_transitions,
            }
        )
    computed = embed_row_items(
        model,
        rows,
        args.device,
        args.dtype,
        int(config["embedding_batch_size"]),
        int(config["max_attention_cells"]),
    )
    saved = np.stack(saved)
    difference = computed - saved
    absolute = np.abs(difference)
    saved_norm = np.linalg.norm(saved, axis=1)
    computed_norm = np.linalg.norm(computed, axis=1)
    difference_norm = np.linalg.norm(difference, axis=1)
    cosine = np.sum(saved * computed, axis=1) / np.maximum(
        saved_norm * computed_norm, 1e-12
    )
    relative_l2 = difference_norm / np.maximum(saved_norm, 1e-12)
    patient_level = pd.DataFrame(metadata)
    patient_level["mean_abs_error"] = absolute.mean(axis=1)
    patient_level["max_abs_error"] = absolute.max(axis=1)
    patient_level["relative_l2_error"] = relative_l2
    patient_level["cosine_similarity"] = cosine
    atomic_parquet(
        patient_level,
        args.output_dir / "raw" / "original_embedding_check_patient_level.parquet",
    )
    result = {
        "patients": int(len(rows)),
        "max_abs_error": float(absolute.max()),
        "mean_abs_error": float(absolute.mean()),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "median_relative_l2_error": float(np.median(relative_l2)),
        "p95_relative_l2_error": float(np.quantile(relative_l2, 0.95)),
        "minimum_cosine_similarity": float(np.min(cosine)),
        "median_cosine_similarity": float(np.median(cosine)),
        "nonmonotonic_age_transitions": int(
            patient_level["nonmonotonic_age_transitions"].sum()
        ),
        "gates": {
            "maximum_mean_abs_error": float(config["embedding_check_atol"]),
            "maximum_p95_relative_l2_error": 0.05,
            "minimum_cosine_similarity": 0.995,
            "require_monotonic_input_age": True,
        },
        "single_coordinate_max_is_diagnostic_only": True,
        "manifest_check": manifest_check,
    }
    write_json(result, check_path)
    failures = []
    if result["mean_abs_error"] > result["gates"]["maximum_mean_abs_error"]:
        failures.append("mean_abs_error")
    if (
        result["p95_relative_l2_error"]
        > result["gates"]["maximum_p95_relative_l2_error"]
    ):
        failures.append("p95_relative_l2_error")
    if (
        result["minimum_cosine_similarity"]
        < result["gates"]["minimum_cosine_similarity"]
    ):
        failures.append("minimum_cosine_similarity")
    if result["nonmonotonic_age_transitions"]:
        failures.append("nonmonotonic_input_age")
    if failures:
        raise RuntimeError(f"Original embedding contract failed {failures}: {result}")
    log(f"[EMBEDDING MATCH] {result}")
    return result


def kaplan_meier_curve(duration, event, max_day=1826):
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    valid = np.isfinite(duration) & (duration >= 0)
    duration = np.clip(duration[valid], 0, max_day)
    event = event[valid]
    event_times = np.unique(duration[event])
    event_times.sort()
    survival = 1.0
    values = []
    for event_time in event_times:
        at_risk = int(np.sum(duration >= event_time))
        events = int(np.sum(event & np.isclose(duration, event_time, atol=1e-7)))
        if at_risk and events:
            survival *= 1.0 - events / at_risk
        values.append(1.0 - survival)
    days = np.arange(max_day + 1, dtype=np.int32)
    curve = np.zeros(max_day + 1, dtype=np.float64)
    if values:
        indexes = np.searchsorted(event_times, days, side="right") - 1
        known = indexes >= 0
        curve[known] = np.asarray(values)[indexes[known]]
    return days, curve


def summarize(args, config, arm_paths, token_paths, curve_paths, cox_by_target, timing):
    arms = pd.concat([pd.read_parquet(path) for path in arm_paths], ignore_index=True)
    tokens = pd.concat([pd.read_parquet(path) for path in token_paths], ignore_index=True)
    curves = pd.concat([pd.read_parquet(path) for path in curve_paths], ignore_index=True)
    expected = 6912
    if len(arms) != expected:
        raise RuntimeError(f"Expected {expected} patient-arm rows, got {len(arms)}")
    summary_rows = []
    for keys, group in arms.groupby(
        ["pathway_id", "source_role", "source", "target", "shift_days"], sort=True
    ):
        pathway_id, source_role, source, target, shift_days = keys
        row = {
            "pathway_id": pathway_id,
            "source_role": source_role,
            "source": source,
            "target": target,
            "shift_days": int(shift_days),
            "patients": int(len(group)),
            "mean_risk_score_difference": float(group["risk_score_difference"].mean()),
            "median_risk_score_difference": float(group["risk_score_difference"].median()),
            "proportion_score_increased": float((group["risk_score_difference"] > 0).mean()),
        }
        for day in LANDMARKS:
            column = f"risk_difference_day_{day}"
            row[f"mean_risk_difference_day_{day}"] = float(group[column].mean())
            row[f"median_risk_difference_day_{day}"] = float(group[column].median())
            row[f"proportion_risk_increased_day_{day}"] = float((group[column] > 0).mean())
            row[f"mean_edited_risk_day_{day}"] = float(group[f"edited_risk_day_{day}"].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    atomic_parquet(arms, args.output_dir / "patient_shift_embeddings_and_risks.parquet")
    atomic_parquet(tokens, args.output_dir / "edited_diagnosis_token_rows.parquet")
    atomic_parquet(curves, args.output_dir / "population_risk_curves_daily.parquet")
    atomic_csv(summary, args.output_dir / "shift_effect_summary.csv")
    observed_parts = []
    for item in config["pathways"]:
        pathway = arms.loc[
            arms["pathway_id"].eq(item["pathway_id"])
            & arms["source_role"].eq("source_a")
            & arms["shift_days"].eq(0)
        ].copy()
        days, observed = kaplan_meier_curve(pathway["duration_days"], pathway["event"])
        original_curve = curves.loc[
            curves["pathway_id"].eq(item["pathway_id"])
            & curves["source_role"].eq("source_a")
            & curves["shift_days"].eq(0),
            ["day", "mean_original_risk"],
        ].copy()
        if len(original_curve) != len(days):
            raise RuntimeError(f"Missing original daily curve for {item['pathway_id']}")
        original_curve = original_curve.sort_values("day")
        observed_parts.append(
            pd.DataFrame(
                {
                    "pathway_id": item["pathway_id"],
                    "target": item["target"],
                    "day": days,
                    "observed_km": observed,
                    "mean_original_fermat_cox_risk": original_curve[
                        "mean_original_risk"
                    ].to_numpy(),
                    "sample_patients": len(pathway),
                    "sample_events": int(pathway["event"].sum()),
                }
            )
        )
    observed_curves = pd.concat(observed_parts, ignore_index=True)
    atomic_parquet(
        observed_curves,
        args.output_dir / "observed_vs_original_model_curves_daily.parquet",
    )
    metrics = pd.DataFrame([value["metrics"] for value in cox_by_target.values()])
    atomic_csv(metrics, args.output_dir / "cox_target_model_metrics.csv")
    write_json(timing, args.output_dir / "timing.json")
    sample_audit = pd.read_csv(args.output_dir / "sample_replacement_audit.csv")

    landmarks = summary.loc[summary["shift_days"].isin([-360, -180, 0, 180, 360])]
    return_text = "\n".join(
        [
            "## STATUS COMPLETE_TASK30_FIVE_PATHWAY_FERMAT_COX_SHIFT",
            f"pathways {len(config['pathways'])}",
            f"unique_targets {len(cox_by_target)}",
            f"patient_arm_rows {len(arms)}",
            f"edited_token_rows {len(tokens)}",
            "future_rollout_executed false",
            "## COX_TARGET_MODELS",
            metrics.to_csv(index=False).rstrip(),
            "## SAMPLE_REPLACEMENT_AUDIT",
            sample_audit.to_csv(index=False).rstrip(),
            "## SHIFT_LANDMARK_SUMMARY",
            landmarks.to_csv(index=False).rstrip(),
            "## TIMING",
            json.dumps(timing, ensure_ascii=False),
            "## OUTPUT_DIR",
            str(args.output_dir),
        ]
    ) + "\n"
    (args.output_dir / "RETURN_THIS.txt").write_text(return_text, encoding="utf-8")
    print(return_text, end="", flush=True)


def self_test():
    rows = np.asarray(
        [
            [1, 100, 10, 1],
            [1, 150, 20, 2],
            [1, 200, 10, 1],
            [1, 250, 30, 3],
        ],
        dtype=np.uint32,
    )
    edited, tokens = edit_first_diagnosis_day(rows, 500, {10}, 90, 4)
    if len(tokens) != 1 or tokens[0]["shifted_age_days"] != 190:
        raise AssertionError("First diagnosis-day edit failed")
    if set(map(tuple, edited[:, [0, 2, 3]])) != set(map(tuple, rows[:, [0, 2, 3]])):
        raise AssertionError("Row identities changed in edit self-test")

    test_pathways = [
        {"pathway_id": f"single_{index}", "source_a": f"source_{index}", "target": "target"}
        for index in range(4)
    ] + [
        {
            "pathway_id": "double",
            "source_a": "source_a",
            "source_b": "source_b",
            "target": "target",
        }
    ]
    sample_rows = []
    range_rows = []
    for pathway_number, item in enumerate(test_pathways):
        for patient_number in range(128):
            person_id = pathway_number * 1000 + patient_number
            sample_rows.append(
                {
                    "pathway_id": item["pathway_id"],
                    "person_id": person_id,
                    # Reproduce the columns present in the real pilot-sample CSV.
                    "stable_min_delta_days": -360,
                    "stable_max_delta_days": 360,
                    "stable_symmetric_days": 360,
                }
            )
            for source_role, _ in pathway_roles(item):
                range_rows.append(
                    {
                        "pathway_id": item["pathway_id"],
                        "source_role": source_role,
                        "person_id": person_id,
                        "stable_min_delta_days": -400,
                        "stable_max_delta_days": 400,
                    }
                )
    plan = build_arm_plan(
        pd.DataFrame(sample_rows),
        pd.DataFrame(range_rows),
        {
            "pathways": test_pathways,
            "shift_days": [-360, -270, -180, -90, 0, 90, 180, 270, 360],
        },
    )
    if len(plan) != 54 or int(plan["patients"].sum()) != 6912:
        raise AssertionError("Stable-range arm planning self-test failed")
    saved_fingerprint = {"config": "/old/config.json", "config_sha256": "same"}
    moved_fingerprint = {"config": "/new/config.json", "config_sha256": "same"}
    changed_fingerprint = {"config": "/new/config.json", "config_sha256": "changed"}
    if not resume_fingerprint_matches(saved_fingerprint, moved_fingerprint):
        raise AssertionError("Moved config path should remain resumable")
    if resume_fingerprint_matches(saved_fingerprint, changed_fingerprint):
        raise AssertionError("Changed config contents must not be resumable")
    versioned_fingerprint = {
        **moved_fingerprint,
        "runner_version": RUNNER_VERSION,
    }
    if resume_fingerprint_matches(saved_fingerprint, versioned_fingerprint):
        raise AssertionError("Unversioned run resumed without an explicit migration")
    if not resume_fingerprint_matches(
        saved_fingerprint,
        versioned_fingerprint,
        allow_runner_version_mismatch=True,
    ):
        raise AssertionError("Safe unversioned pre-arm migration failed")

    selection_pathways = [
        {"pathway_id": "diabetes_to_ckd", "target": "ckd"},
        {"pathway_id": "other_pathway", "target": "other_target"},
    ]
    selection_candidates = []
    selection_original = []
    target_eligible = {"ckd": [], "other_target": []}
    for pathway_number, item in enumerate(selection_pathways):
        if item["pathway_id"] == "diabetes_to_ckd":
            strata = [("measured_no_abnormality", 150), ("recent_abnormality", 150)]
            original_per_stratum = 64
        else:
            strata = [("not_applicable", 180)]
            original_per_stratum = 128
        for stratum_number, (stratum, count) in enumerate(strata):
            ids = [
                100000 * (pathway_number + 1) + 1000 * stratum_number + number
                for number in range(count)
            ]
            for person_id in ids:
                selection_candidates.append(
                    {
                        "pathway_id": item["pathway_id"],
                        "person_id": person_id,
                        "renal_screen_status": stratum,
                    }
                )
            for person_id in ids[:original_per_stratum]:
                selection_original.append(
                    {
                        "pathway_id": item["pathway_id"],
                        "person_id": person_id,
                        "pilot_stratum": (
                            stratum
                            if item["pathway_id"] == "diabetes_to_ckd"
                            else "unstratified"
                        ),
                    }
                )
            target_eligible[item["target"]].extend(ids[5:])
    mock_cox = {}
    for target, ids in target_eligible.items():
        all_target_ids = [
            row["person_id"]
            for row in selection_candidates
            if next(
                item["target"]
                for item in selection_pathways
                if item["pathway_id"] == row["pathway_id"]
            )
            == target
        ]
        eligible_set = set(ids)
        mock_cox[target] = {
            "selected": pd.DataFrame({"person_id": ids}),
            "candidate_audit": pd.DataFrame(
                {
                    "person_id": all_target_ids,
                    "cox_risk_set_eligible": [
                        value in eligible_set for value in all_target_ids
                    ],
                    "cox_risk_set_exclusion_reason": [
                        "eligible" if value in eligible_set else "synthetic_ineligible"
                        for value in all_target_ids
                    ],
                }
            ),
        }
    selected, audit, _ = select_cox_eligible_samples(
        pd.DataFrame(selection_original),
        pd.DataFrame(selection_candidates),
        mock_cox,
        {"pathways": selection_pathways, "patients_per_pathway": 128},
        42,
    )
    if selected.groupby("pathway_id")["person_id"].nunique().to_dict() != {
        "diabetes_to_ckd": 128,
        "other_pathway": 128,
    }:
        raise AssertionError("Cox-eligible replacement sampling failed")
    if int(audit["replacement_patients"].sum()) < 15:
        raise AssertionError("Ineligible original patients were not replaced")
    saved_embedding = np.asarray([[1.0, 2.0]], dtype=np.float32)
    recomputed_original = np.asarray([[1.1, 1.9]], dtype=np.float32)
    recomputed_unchanged = recomputed_original.copy()
    unchanged, unchanged_delta = anchor_saved_embedding_with_paired_delta(
        saved_embedding, recomputed_original, recomputed_unchanged
    )
    if not np.array_equal(unchanged, saved_embedding) or np.any(unchanged_delta):
        raise AssertionError("Paired anchoring did not cancel recomputation offset")
    recomputed_edited = np.asarray([[1.3, 1.8]], dtype=np.float32)
    edited, delta = anchor_saved_embedding_with_paired_delta(
        saved_embedding, recomputed_original, recomputed_edited
    )
    if not np.allclose(delta, [[0.2, -0.1]], atol=1e-6) or not np.allclose(
        edited, [[1.2, 1.9]], atol=1e-6
    ):
        raise AssertionError("Paired embedding delta was not anchored correctly")
    overlapping_candidates = []
    overlapping_pathways = [f"pathway_{index}" for index in range(5)]
    for pathway_number, pathway_id in enumerate(overlapping_pathways):
        for person_id in [1, 2, 3, *range(1000 + 20 * pathway_number, 1020 + 20 * pathway_number)]:
            overlapping_candidates.append(
                {
                    "pathway_id": pathway_id,
                    "person_id": person_id,
                    "split": "test",
                    "patient_id_dense": person_id + 5000,
                }
            )
    overlapping_frame = pd.DataFrame(overlapping_candidates)
    unique_overlapping_ids = sorted(overlapping_frame["person_id"].unique())
    synthetic_labels = pd.DataFrame(
        {
            "person_id": unique_overlapping_ids,
            "split": "test",
            "age_at_index": 60.0,
        }
    )
    synthetic_embedding_meta = pd.DataFrame(
        {
            "person_id": unique_overlapping_ids[1:],
            "split": "test",
            "patient_id_dense": [value + 5000 for value in unique_overlapping_ids[1:]],
            "has_embedding_sequence": True,
            "sequence_length_pre_index": 100,
        }
    )
    attached_overlap = attach_embedding_preflight_eligibility(
        overlapping_frame,
        synthetic_labels,
        synthetic_embedding_meta,
    )
    if attached_overlap.loc[
        attached_overlap["person_id"].eq(unique_overlapping_ids[0]),
        "embedding_preflight_exclusion_reason",
    ].ne("missing_embedding_row").any():
        raise AssertionError("Missing embedding rows were not classified correctly")
    unique_probe = select_unique_embedding_probe(
        attached_overlap.loc[attached_overlap["embedding_preflight_eligible"]],
        overlapping_pathways,
        16,
    )
    if len(unique_probe) != 16 or unique_probe["person_id"].nunique() != 16:
        raise AssertionError("Cross-pathway duplicate patients remained in embedding probe")
    if set(unique_probe["pathway_id"]) != set(overlapping_pathways):
        raise AssertionError("Embedding probe did not cover every pathway")
    log("SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require_torch()
    normalize_args(args)
    config = load_config(args.config_file)
    required = [
        args.config_file,
        args.data_dir / "test.bin",
        args.data_dir / "token_registry.csv",
        args.label_dir / "phenotype_group_concept_map.csv",
        args.embedding_file,
        args.survival_cache,
        args.ckpt,
        args.range_dir / "gpu_continuous_pilot_samples.csv",
        args.range_dir / "raw" / "patient_source_stable_ranges.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    prepare_output(args, config)
    overall_started = time.time()
    original_samples, ranges = load_samples(args, config)
    candidate_pool = build_cox_candidate_pool(ranges, config)
    token_sets, token_mapping = load_source_token_sets(args, config)
    atomic_csv(token_mapping, args.output_dir / "source_token_mapping.csv")

    # Fail on checkpoint/data/embedding incompatibility before fitting Cox.
    model, checkpoint = load_model(args.ckpt, args.device)
    if int(model.config.block_size) != int(config["block_size"]):
        raise RuntimeError("Checkpoint block_size does not match config")
    split_data = load_split_data(args.data_dir, config["split"])
    embedding_check = verify_original_embeddings(
        args, config, candidate_pool, split_data, model, checkpoint
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cox_by_target = {}
    cox_started = time.time()
    for target in sorted({item["target"] for item in config["pathways"]}):
        candidate_ids = candidate_pool.loc[
            candidate_pool["pathway_id"].isin(
                [item["pathway_id"] for item in config["pathways"] if item["target"] == target]
            ),
            "person_id",
        ].astype(np.int64).unique()
        cox_by_target[target] = fit_or_load_cox(args, target, candidate_ids)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cox_seconds = time.time() - cox_started

    samples, sample_audit, patient_eligibility = select_cox_eligible_samples(
        original_samples,
        candidate_pool,
        cox_by_target,
        config,
        args.random_seed,
    )
    atomic_csv(samples, args.output_dir / "cox_eligible_pilot_samples.csv")
    atomic_csv(sample_audit, args.output_dir / "sample_replacement_audit.csv")
    atomic_parquet(
        patient_eligibility,
        args.output_dir / "raw" / "candidate_cox_risk_set_eligibility.parquet",
    )
    plan = build_arm_plan(samples, ranges, config)
    atomic_csv(plan, args.output_dir / "arm_plan.csv")
    log(
        f"[PLAN] arms={len(plan)} patient_arm_rows={int(plan['patients'].sum())} "
        "nonzero_shift_embeddings=6144"
    )

    model, inference_checkpoint = load_model(args.ckpt, args.device)
    if int(inference_checkpoint.get("iter", -1)) != int(checkpoint.get("iter", -1)):
        raise RuntimeError("Checkpoint changed between preflight and inference")

    arm_paths = []
    token_paths = []
    curve_paths = []
    inference_started = time.time()
    measured_seconds = 0.0
    measured_arms = 0
    for arm_number, arm in enumerate(plan.itertuples(index=False), start=1):
        item = next(
            value for value in config["pathways"] if value["pathway_id"] == arm.pathway_id
        )
        selected = samples.loc[samples["pathway_id"].eq(arm.pathway_id)].copy()
        arm_path, token_path, curve_path, elapsed = run_arm(
            args,
            config,
            item,
            arm.source_role,
            arm.source,
            int(arm.shift_days),
            selected,
            token_sets[arm.source],
            cox_by_target[arm.target],
            split_data,
            model,
        )
        arm_paths.append(arm_path)
        token_paths.append(token_path)
        curve_paths.append(curve_path)
        if elapsed > 0 and int(arm.shift_days) != 0:
            measured_seconds += elapsed
            measured_arms += 1
        remaining_nonzero = int(
            np.sum((plan.iloc[arm_number:]["shift_days"].to_numpy(dtype=np.int64) != 0))
        )
        eta_minutes = (
            measured_seconds / measured_arms * remaining_nonzero / 60.0
            if measured_arms
            else np.nan
        )
        log(
            f"[PROGRESS] arm={arm_number}/{len(plan)} "
            f"measured_gpu_arms={measured_arms} eta_minutes={eta_minutes:,.1f}"
        )
    inference_seconds = time.time() - inference_started
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timing = {
        "cox_fit_or_load_seconds": cox_seconds,
        "arm_inference_seconds": inference_seconds,
        "total_seconds_before_summary": time.time() - overall_started,
        "measured_nonzero_arm_seconds": measured_seconds,
        "measured_nonzero_arms": measured_arms,
        "checkpoint_iter": int(checkpoint.get("iter", -1)),
        "embedding_check": embedding_check,
    }
    summarize(args, config, arm_paths, token_paths, curve_paths, cox_by_target, timing)
    log("[COMPLETE] Task 30 five-pathway FERMAT+Cox shift experiment finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise
