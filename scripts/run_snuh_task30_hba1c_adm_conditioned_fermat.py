#!/usr/bin/env python3
"""Generate four ADM-policy FERMAT mortality curves.

Two estimators are produced from separate constrained generations:

1. rollout-event: empirical first generated death-token time;
2. embedding-hazard: death tokens are suppressed, monthly FERMAT hidden states
   are scored by the train-fitted monthly hazard head.

Raw patient parts are saved before population summaries and resume validates
both parts.  This development runner defaults to val; held-out test is reserved
for the locked final comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELF_TEST_ONLY = "--self-test" in sys.argv
if not SELF_TEST_ONLY:
    import torch
    import torch.nn.functional as F
    from model import TokenType, build_attention_mask
    from scripts.run_snuh_task27_primary_direct_risk_dry_run import (
        collate_prefix,
        dtype_context,
        load_model,
        load_registry,
        load_split_data,
        registry_maps,
        rows_before_index,
    )

from scripts.extract_snuh_task30_hba1c_monthly_embeddings import month_schedule
from scripts.snuh_task30_adm_strategy import STRATEGY_BY_NAME, STRATEGY_SPECS, strategy_action


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_INPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_ccw_inputs"
DEFAULT_DATA_DIR = POD_ROOT / "etl" / "patient_100pct_seed_42_with_genomics_tokens"
DEFAULT_ADM_DIR = DEFAULT_INPUT_DIR
DEFAULT_HAZARD_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_embedding_hazard"
DEFAULT_CKPT = (
    POD_ROOT / "task21" / "outputs" / "block2048_full_10l640_20260629" / "block_2048" / "ckpt.pt"
)
DEFAULT_OUTPUT_DIR = POD_ROOT / "task30" / "outputs" / "hba1c_adm_conditioned_fermat"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--adm-dir", type=Path, default=DEFAULT_ADM_DIR)
    parser.add_argument("--hazard-dir", type=Path, default=DEFAULT_HAZARD_DIR)
    parser.add_argument("--fermat-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--same-day-repeat-penalty", type=float, default=2.0)
    parser.add_argument("--same-day-temperature", type=float, default=1.0)
    parser.add_argument("--same-day-probability-cap", type=float, default=0.35)
    parser.add_argument("--random-seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size < 50_000_000 else None
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def load_adm_tokens(adm_dir, registry, vocab_size):
    inventory = pd.read_csv(Path(adm_dir) / "observed_adm_inventory.csv")
    required = {"expected_token_key", "token_in_fermat_registry"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"ADM inventory missing columns: {missing}")
    key_to_model = {
        str(row["token_key"]): int(row["token_id"]) + 1
        for row in registry
        if int(row["token_id"]) + 1 < int(vocab_size)
    }
    covered = inventory["token_in_fermat_registry"]
    if covered.dtype != bool:
        covered = covered.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    keys = inventory.loc[covered, "expected_token_key"].astype(str)
    tokens = sorted({key_to_model[key] for key in keys if key in key_to_model})
    if not tokens:
        raise ValueError("no observed ADM product maps to a FERMAT model token")
    return tokens


def death_model_tokens(registry, vocab_size):
    tokens = []
    for row in registry:
        token = int(row["token_id"]) + 1
        text = str(row.get("token_key", ""))
        if token < int(vocab_size) and text.startswith("DTH:"):
            tokens.append(token)
    if not tokens:
        raise ValueError("no death tokens in FERMAT registry")
    return sorted(set(tokens))


def _hidden_logits_and_wait(model, idx, age, token_type, args):
    block_size = int(model.config.block_size)
    idx_in, age_in, type_in = idx[:, -block_size:], age[:, -block_size:], token_type[:, -block_size:]
    x = model.transformer.wte(idx_in)
    x = model.transformer.token_drop(x) * (1 - model.config.token_dropout)
    x = x + model.transformer.wae(age_in.unsqueeze(-1)) + model.transformer.wtype(type_in)
    x = model.transformer.drop(x)
    mask = build_attention_mask(idx_in, age_in, mask_ties=False)
    for block in model.transformer.h:
        x, _ = block(x, mask)
    x = model.transformer.ln_f(x)
    hidden = x[:, -1, :]
    logits = model.lm_head(hidden)
    ignored = sorted(set(model.config.ignore_tokens + model.config.output_ignore_tokens))
    if ignored:
        logits[:, ignored] = -torch.inf
    if not model.config.decoupled_time_head:
        raise RuntimeError("ADM conditioned rollout currently requires the checkpoint's decoupled time head")
    raw_log_rate = model.time_head(hidden).squeeze(-1)
    log_rate = -torch.log(torch.exp(-raw_log_rate) + model.config.t_min)
    rate = torch.exp(log_rate).clamp_min(1e-12)
    if model.config.two_stage_time_head:
        same_logit = model.same_day_head(hidden).squeeze(-1)
        same_prob = torch.sigmoid(same_logit / max(float(args.same_day_temperature), 1e-6))
        same_prob = torch.clamp(same_prob, max=float(args.same_day_probability_cap))
        same_day = torch.rand_like(same_prob) < same_prob
    else:
        same_day = torch.zeros_like(rate, dtype=torch.bool)
    wait = -torch.rand_like(rate).clamp_min(1e-12).log() / rate - model.config.t_min
    wait = torch.where(same_day, torch.zeros_like(wait), torch.clamp(wait, min=1.0))
    return hidden, logits, torch.clamp(wait, min=0.0, max=365 * 80)


def _apply_same_day_penalty(logits, idx, age, penalty):
    if float(penalty) <= 0:
        return logits
    adjusted = logits.clone()
    for batch_index in range(len(idx)):
        positions = torch.isclose(age[batch_index], age[batch_index, -1], atol=1e-4, rtol=0.0)
        tokens = idx[batch_index, positions]
        tokens = tokens[tokens > 1]
        if tokens.numel():
            adjusted[batch_index, tokens] -= float(penalty)
    return adjusted


def _sample_tokens(logits, top_k, temperature):
    values = logits / max(float(temperature), 1e-6)
    if top_k and 0 < int(top_k) < values.shape[1]:
        threshold = torch.topk(values, int(top_k), dim=-1).values[:, [-1]]
        values = values.masked_fill(values < threshold, -torch.inf)
    if torch.isinf(values).all(dim=1).any():
        raise RuntimeError("a rollout row has no allowed token")
    return torch.multinomial(F.softmax(values, dim=-1), 1)


def conditioned_generate(
    model,
    prefix,
    strategy,
    adm_tokens,
    death_tokens,
    token_type_lookup,
    allowed_mask,
    index_age_days,
    deadline_day,
    snapshot_days,
    args,
    batch_size,
    suppress_death,
    seed,
):
    idx0, age0, type0 = collate_prefix(prefix, args.device)
    idx = idx0.repeat(int(batch_size), 1)
    age = age0.repeat(int(batch_size), 1)
    token_type = type0.repeat(int(batch_size), 1)
    torch.manual_seed(int(seed))
    if args.device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    adm_tensor = torch.tensor(adm_tokens, device=args.device, dtype=torch.long)
    death_tensor = torch.tensor(death_tokens, device=args.device, dtype=torch.long)
    base_mask = allowed_mask.to(device=args.device, dtype=torch.bool).clone()
    if suppress_death:
        base_mask[death_tensor] = False
    seen_adm = torch.zeros(int(batch_size), dtype=torch.bool, device=args.device)
    first_death = torch.full((int(batch_size),), np.nan, dtype=torch.float32, device=args.device)
    forced_count = torch.zeros(int(batch_size), dtype=torch.int64, device=args.device)
    snapshot_days = np.asarray(snapshot_days, dtype=np.float32)
    snapshots = torch.zeros(
        (int(batch_size), len(snapshot_days), int(model.config.n_embd)),
        dtype=torch.float32,
        device="cpu",
    )
    next_snapshot = np.zeros(int(batch_size), dtype=np.int64)
    horizon_age = float(index_age_days + 1826)
    with torch.no_grad(), dtype_context(args.device, args.dtype):
        for _ in range(int(args.max_new_tokens)):
            hidden, logits, wait = _hidden_logits_and_wait(model, idx, age, token_type, args)
            logits = _apply_same_day_penalty(logits, idx, age, args.same_day_repeat_penalty)
            candidate_age = age[:, -1] + wait
            row_logits = logits.clone()
            row_logits[:, ~base_mask] = -torch.inf
            forced = torch.zeros(int(batch_size), dtype=torch.bool, device=args.device)
            died_before_step = ~torch.isnan(first_death)
            for row in range(int(batch_size)):
                candidate_day = float(candidate_age[row].detach().cpu()) - float(index_age_days)
                action = (
                    {"force_adm_at_deadline": False, "suppress_adm": False}
                    if bool(died_before_step[row])
                    else strategy_action(strategy, candidate_day, deadline_day, bool(seen_adm[row]))
                )
                while (
                    next_snapshot[row] < len(snapshot_days)
                    and snapshot_days[next_snapshot[row]] < candidate_day
                ):
                    snapshots[row, next_snapshot[row]] = hidden[row].detach().float().cpu()
                    next_snapshot[row] += 1
                if action["suppress_adm"]:
                    row_logits[row, adm_tensor] = -torch.inf
                if candidate_day < 0:
                    # Eligibility establishes that the patient is alive and has
                    # no prior ADM at time zero. Do not let bridge generation
                    # between the last observed token and time zero contradict it.
                    row_logits[row, adm_tensor] = -torch.inf
                    row_logits[row, death_tensor] = -torch.inf
                if action["force_adm_at_deadline"]:
                    forced[row] = True
                    row_logits[row, :] = -torch.inf
                    row_logits[row, adm_tensor] = logits[row, adm_tensor]
                    candidate_age[row] = float(index_age_days + deadline_day)
            sampled = _sample_tokens(row_logits, args.top_k if not forced.any() else 0, args.temperature)
            # A mixed batch needs top-k for natural rows and unrestricted ADM
            # sampling for forced rows. Re-sample natural rows with top-k.
            if forced.any() and (~forced).any():
                natural = _sample_tokens(row_logits[~forced], args.top_k, args.temperature)
                sampled[~forced] = natural
            type_next = torch.tensor(
                [[token_type_lookup.get(int(value), int(TokenType.DX))] for value in sampled.squeeze(-1)],
                dtype=torch.long,
                device=args.device,
            )
            idx = torch.cat([idx, sampled], dim=1)
            age = torch.cat([age, candidate_age[:, None]], dim=1)
            token_type = torch.cat([token_type, type_next], dim=1)
            is_adm = torch.isin(sampled.squeeze(-1), adm_tensor)
            within_deadline = (
                (candidate_age >= float(index_age_days) - 1e-5)
                & (candidate_age <= float(index_age_days + deadline_day) + 1e-5)
            )
            seen_adm |= is_adm & within_deadline & ~died_before_step
            forced_count += forced.to(torch.int64)
            is_death = torch.isin(sampled.squeeze(-1), death_tensor)
            new_death = (
                is_death
                & (candidate_age >= float(index_age_days) - 1e-5)
                & torch.isnan(first_death)
            )
            first_death[new_death] = candidate_age[new_death] - float(index_age_days)
            if float(candidate_age.min().detach().cpu()) > horizon_age:
                break
        # Fill remaining monthly states using the final valid generated history.
        final_hidden, _, _ = _hidden_logits_and_wait(model, idx, age, token_type, args)
        for row in range(int(batch_size)):
            while next_snapshot[row] < len(snapshot_days):
                snapshots[row, next_snapshot[row]] = final_hidden[row].detach().float().cpu()
                next_snapshot[row] += 1
    return {
        "first_death_day": first_death.detach().cpu().numpy(),
        "followup_end_day": (age[:, -1] - float(index_age_days)).detach().cpu().numpy(),
        "forced_adm_count": forced_count.detach().cpu().numpy(),
        "seen_adm_by_deadline": seen_adm.detach().cpu().numpy(),
        "snapshots": snapshots.numpy(),
    }


def load_hazard_model(path):
    data = np.load(path, allow_pickle=False)
    return {name: data[name] for name in data.files}


def score_embedding_hazard(snapshot, hazard):
    mean = hazard["embedding_mean"].astype(np.float32)
    scale = hazard["embedding_scale"].astype(np.float32)
    normalized = (snapshot.astype(np.float32) - mean) / scale
    month = np.arange(snapshot.shape[1], dtype=np.float32) / 60.0
    time = np.stack([month, month * month], axis=1)
    time = np.broadcast_to(time[None, :, :], (snapshot.shape[0], len(month), 2))
    features = np.concatenate([normalized, time], axis=2)
    weight = hazard["weight"].reshape(-1).astype(np.float32)
    bias = float(hazard["bias"].reshape(-1)[0])
    logits = features @ weight + bias
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    survival = np.cumprod(1.0 - probability, axis=1)
    return 1.0 - survival


def validate_parts(direct_path, embedding_path, rollouts):
    if not direct_path.is_file() or not embedding_path.is_file():
        return False
    direct = pd.read_parquet(direct_path)
    embedding = pd.read_parquet(embedding_path)
    expected_direct = len(STRATEGY_SPECS) * int(rollouts)
    expected_embedding = expected_direct * 60
    return (
        len(direct) == expected_direct
        and len(embedding) == expected_embedding
        and not direct[["strategy", "rollout_index"]].duplicated().any()
        and not embedding[["strategy", "rollout_index", "month"]].duplicated().any()
    )


def self_test():
    fake = np.zeros((2, 60, 3), dtype=np.float32)
    hazard = {
        "embedding_mean": np.zeros(3),
        "embedding_scale": np.ones(3),
        "weight": np.zeros((1, 5)),
        "bias": np.asarray([-4.0]),
    }
    risk = score_embedding_hazard(fake, hazard)
    assert risk.shape == (2, 60)
    assert np.all(np.diff(risk, axis=1) >= 0)
    print("SELF_TEST_OK embedding_hazard_curve")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    fingerprint = {
        "runner": file_identity(Path(__file__)),
        "patients": file_identity(args.input_dir / "ccw_patients.parquet"),
        "adm_inventory": file_identity(args.adm_dir / "observed_adm_inventory.csv"),
        "hazard_model": file_identity(args.hazard_dir / "embedding_monthly_hazard_model.npz"),
        "checkpoint": file_identity(args.fermat_ckpt),
        "split": args.split,
        "patients_requested": int(args.patients),
        "rollouts": int(args.rollouts),
        "rollout_batch_size": int(args.rollout_batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "random_seed": int(args.random_seed),
    }
    fingerprint_path = args.output_dir / "run_config.json"
    if fingerprint_path.is_file():
        existing = json.loads(fingerprint_path.read_text())
        if not args.resume:
            raise FileExistsError(f"{args.output_dir} already contains a run; pass --resume")
        if existing != fingerprint:
            raise RuntimeError("resume configuration or tracked inputs differ from run_config.json")
    else:
        if args.resume and any(raw_dir.iterdir()):
            raise RuntimeError("cannot resume raw parts without run_config.json")
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
    patients = pd.read_parquet(args.input_dir / "ccw_patients.parquet")
    patients = patients.loc[patients["split"].eq(args.split)].copy()
    patients = patients.sort_values("patient_key")
    if int(args.patients) > 0:
        patients = patients.sample(
            n=min(int(args.patients), len(patients)), random_state=int(args.random_seed)
        ).sort_values("patient_key")
    model, checkpoint = load_model(args.fermat_ckpt, args.device)
    registry, _ = load_registry(args.data_dir)
    token_type_lookup, _, clinical_mask = registry_maps(
        registry, int(model.config.vocab_size), args.device
    )
    adm_tokens = load_adm_tokens(args.adm_dir, registry, model.config.vocab_size)
    death_tokens = death_model_tokens(registry, model.config.vocab_size)
    split_data = load_split_data(args.data_dir, args.split)
    hazard = load_hazard_model(args.hazard_dir / "embedding_monthly_hazard_model.npz")
    completed = 0
    for patient_number, patient in enumerate(patients.itertuples(index=False), start=1):
        direct_path = raw_dir / f"patient_{int(patient.patient_key):09d}_direct.parquet"
        embedding_path = raw_dir / f"patient_{int(patient.patient_key):09d}_embedding.parquet"
        if args.resume and validate_parts(direct_path, embedding_path, args.rollouts):
            log(f"[RESUME] patient={patient_number}/{len(patients)}")
            continue
        prefix = rows_before_index(
            split_data,
            int(patient.patient_key),
            int(patient.index_age_days),
            int(model.config.block_size),
        )
        if prefix is None or len(prefix) == 0:
            raise RuntimeError(f"missing prefix for patient_key={patient.patient_key}")
        schedule = month_schedule(patient.index_date, int(patient.index_age_days), 60)
        snapshot_days = [item["cutoff_age_days"] - int(patient.index_age_days) for item in schedule]
        direct_rows, embedding_rows = [], []
        for strategy_index, spec in enumerate(STRATEGY_SPECS):
            deadline_date = pd.Timestamp(patient.index_date) + pd.DateOffset(months=spec.deadline_months)
            deadline_day = int((deadline_date - pd.Timestamp(patient.index_date)).days)
            rollout_start = 0
            while rollout_start < int(args.rollouts):
                current = min(int(args.rollout_batch_size), int(args.rollouts) - rollout_start)
                paired_seed = int(args.random_seed + int(patient.patient_key) * 1009 + rollout_start)
                direct = conditioned_generate(
                    model, prefix, spec.name, adm_tokens, death_tokens, token_type_lookup,
                    clinical_mask, int(patient.index_age_days), deadline_day, snapshot_days,
                    args, current, False, paired_seed,
                )
                context = conditioned_generate(
                    model, prefix, spec.name, adm_tokens, death_tokens, token_type_lookup,
                    clinical_mask, int(patient.index_age_days), deadline_day, snapshot_days,
                    args, current, True, paired_seed,
                )
                embedding_risk = score_embedding_hazard(context["snapshots"], hazard)
                for row in range(current):
                    if context["followup_end_day"][row] < 1826:
                        beyond = np.asarray(snapshot_days) > context["followup_end_day"][row]
                        embedding_risk[row, beyond] = np.nan
                for row in range(current):
                    rollout_index = rollout_start + row
                    direct_rows.append(
                        {
                            "patient_key": int(patient.patient_key),
                            "strategy": spec.name,
                            "rollout_index": rollout_index,
                            "index_date": str(pd.Timestamp(patient.index_date).date()),
                            "first_death_day": float(direct["first_death_day"][row]),
                            "followup_end_day": float(direct["followup_end_day"][row]),
                            "forced_adm_count": int(direct["forced_adm_count"][row]),
                            "adm_by_deadline": bool(direct["seen_adm_by_deadline"][row]),
                        }
                    )
                    for month, risk in enumerate(embedding_risk[row]):
                        embedding_rows.append(
                            {
                                "patient_key": int(patient.patient_key),
                                "strategy": spec.name,
                                "rollout_index": rollout_index,
                                "month": month,
                                "day": int(snapshot_days[month]),
                                "risk": float(risk),
                                "followup_end_day": float(context["followup_end_day"][row]),
                                "forced_adm_count": int(context["forced_adm_count"][row]),
                                "adm_by_deadline": bool(context["seen_adm_by_deadline"][row]),
                            }
                        )
                rollout_start += current
        atomic_parquet(pd.DataFrame(direct_rows), direct_path)
        atomic_parquet(pd.DataFrame(embedding_rows), embedding_path)
        if not validate_parts(direct_path, embedding_path, args.rollouts):
            raise RuntimeError(f"patient part validation failed: {patient.patient_key}")
        completed += 1
        log(f"[RAW SAVED] patient={patient_number}/{len(patients)} key={patient.patient_key}")

    direct = pd.concat([pd.read_parquet(path) for path in sorted(raw_dir.glob("patient_*_direct.parquet"))])
    embedding = pd.concat([pd.read_parquet(path) for path in sorted(raw_dir.glob("patient_*_embedding.parquet"))])
    curve_rows = []
    for strategy in STRATEGY_BY_NAME:
        group = direct.loc[direct["strategy"].eq(strategy)]
        index_dates = pd.to_datetime(group["index_date"])
        for month in range(60):
            thresholds = (
                index_dates + pd.DateOffset(months=month + 1) - index_dates
            ).dt.days.to_numpy(float)
            death_day = group["first_death_day"].to_numpy(float)
            followup = group["followup_end_day"].to_numpy(float)
            usable = (np.isfinite(death_day) & (death_day <= thresholds)) | (followup >= thresholds)
            events = np.isfinite(death_day) & (death_day <= thresholds)
            curve_rows.append(
                {
                    "estimator": "rollout_event",
                    "strategy": strategy,
                    "month": month,
                    "day": int(np.median(thresholds)),
                    "risk": float(events[usable].mean()) if usable.any() else np.nan,
                    "usable_trajectories": int(usable.sum()),
                }
            )
    rollout_curve = pd.DataFrame(curve_rows)
    embedding_curve = (
        embedding.groupby(["strategy", "month", "day"], as_index=False)["risk"].mean()
    )
    embedding_curve.insert(0, "estimator", "embedding_monthly_hazard")
    curves = pd.concat([rollout_curve, embedding_curve], ignore_index=True)
    curves.to_csv(args.output_dir / "fermat_four_strategy_mortality_curves.csv", index=False)
    pd.DataFrame(
        {
            "status": ["FERMAT_FOUR_STRATEGY_CURVES_COMPLETE"],
            "split": [args.split],
            "patients": [len(patients)],
            "rollouts_per_patient_strategy": [args.rollouts],
            "new_patients_this_run": [completed],
            "checkpoint_step": [int(checkpoint.get("iter", -1))],
            "adm_model_tokens": [len(adm_tokens)],
            "death_model_tokens": [len(death_tokens)],
        }
    ).to_csv(args.output_dir / "run_summary.csv", index=False)
    manifest = {
        "status": "FERMAT_FOUR_STRATEGY_CURVES_COMPLETE",
        "split": args.split,
        "test_used": args.split == "test",
        "strategies": list(STRATEGY_BY_NAME),
        "estimators": ["rollout_event", "embedding_monthly_hazard"],
        "embedding_generation_suppresses_death_tokens": True,
        "rollout_generation_allows_death_tokens": True,
        "init_policy": "natural ADM generation before deadline; otherwise force an ADM token at the deadline using ADM-restricted model probabilities",
        "no_init_policy": "suppress all observed ADM model tokens through the 12-month deadline",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"[COMPLETE] {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
