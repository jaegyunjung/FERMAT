"""Run the SNUH waiting-time (loss_dt) diagnostic on the 1% pilot.

This closes the time objective. The model now carries a learnable global
log-rate scalar that decouples the absolute event rate from vocabulary size, so
the waiting-time loss is the same order of magnitude as cross-entropy instead
of flattening the token logits. This runner trains the LAB-context arm with the
time loss enabled (ramped in over a warmup) and evaluates the best checkpoint.

By default it trains from scratch, which is the cleanest test of the fix and is
directly comparable to the CE-only extended run. Pass --resume-from to continue
an existing LAB-context CE-only checkpoint instead.

Pass criteria (compared against the CE-only extended run):
  - clinical-only CE does not collapse toward ln(vocab); it stays near or below
    the CE-only baseline of 6.53.
  - clinical-only top-1 stays at or above the CE-only baseline of ~3.64%.
  - waiting-time metrics are finite and reported (no longer NA).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POD_STORAGE = Path("/home/khdp-user/workspace/fermat-data")
POD_DATA_DIR = POD_STORAGE / "etl/patient_001pct_seed_42"
POD_OUTPUT_ROOT = POD_STORAGE / "out"

CE_ONLY_BASELINE_CE = 6.53
CE_ONLY_BASELINE_TOP1 = 0.0364
CE_ONLY_BASELINE_NEW_TOP1 = 0.0307


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Defaults to the SNUH 1%% pilot under fermat-data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a versioned directory under fermat-data/out",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "CE-only LAB-context directory with a ckpt.pt to warm-start from. "
            "The run loads its weights but resets the iteration counter, "
            "optimizer, and best validation loss (init_from=finetune)."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/train_fermat_snuh_dt.py",
        help=(
            "Training config. Use config/train_fermat_snuh_dt_finetune.py with "
            "--resume-from."
        ),
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["ce", "objective"],
        help="Override the config's checkpoint selection metric.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    # The training hyper-parameters below default to None: when unset the config
    # file's value is used, so a different --config (e.g. the finetune config)
    # keeps its own settings. Pass a flag only to override the config.
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--loss-dt-weight", type=float)
    parser.add_argument("--loss-dt-warmup-iters", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_bundle_id():
    manifest = ROOT / "bundle_manifest.json"
    if not manifest.exists():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return document.get("bundle_id")


def default_data_dir():
    if POD_DATA_DIR.exists():
        return POD_DATA_DIR
    local = ROOT / "outputs/snuh_tokenization_etl/patient_001pct_seed_42"
    if local.exists():
        return local
    raise FileNotFoundError(
        "Could not find the 1% SNUH ETL pilot. Expected "
        f"{POD_DATA_DIR} or {local}"
    )


def default_output_dir():
    bundle_id = read_bundle_id() or "snuh_task13_dt"
    base = POD_OUTPUT_ROOT if POD_STORAGE.exists() else ROOT / "out"
    candidate = base / bundle_id
    if not candidate.exists():
        return candidate
    return base / f"{bundle_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def run(command, log_path):
    print("+", " ".join(str(part) for part in command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def prepare_output(output, resume_from, overwrite):
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    if resume_from is None:
        return "scratch", None

    resume_from = resume_from.resolve()
    checkpoint = resume_from / "ckpt.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
    shutil.copy2(checkpoint, output / "ckpt.pt")
    return "finetune", resume_from


def read_metrics(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt(value, spec):
    if value is None:
        return "NA"
    return format(value, spec)


def write_summary(output, evaluation_path, resumed_from):
    records = read_metrics(output / "metrics.jsonl")
    validation = [row for row in records if "val/loss_ce" in row]
    if not validation:
        raise RuntimeError("No validation records were written to metrics.jsonl")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    best_validation = min(validation, key=lambda row: row["val/loss_ce"])

    clinical = evaluation["clinical_only_softmax"]
    unigram = evaluation["train_clinical_unigram"]
    waiting = evaluation["clinical_waiting_time"]
    same_day = evaluation.get("clinical_same_day", {"targets": 0})
    clinical_ce = clinical["cross_entropy"]
    clinical_top1 = clinical["top1_accuracy"]
    new_top1 = evaluation["new_clinical"]["top1_accuracy"]

    ce_pass = clinical_ce <= CE_ONLY_BASELINE_CE + 0.1
    top1_pass = clinical_top1 >= CE_ONLY_BASELINE_TOP1 - 0.001
    # New-onset prediction is the clinically important subtask and the one that
    # regressed when the time loss shared the token logits; hold it explicitly.
    new_pass = new_top1 >= CE_ONLY_BASELINE_NEW_TOP1 - 0.001
    # A finite MAE is not enough: the time head must beat the constant-rate and
    # median baselines.
    time_pass = bool(evaluation["time_loss_enabled"]) and bool(waiting["beats_baseline"])
    if evaluation.get("two_stage_time_head"):
        prevalence = same_day.get("same_day_prevalence")
        constant_brier = (
            prevalence * (1.0 - prevalence)
            if prevalence is not None
            else None
        )
        same_day_pass = bool(
            same_day.get("auroc") is not None
            and same_day["auroc"] > 0.5
            and same_day.get("auprc") is not None
            and same_day["auprc"] > prevalence
            and same_day.get("brier_score") is not None
            and same_day["brier_score"] < constant_brier
        )
    else:
        constant_brier = None
        same_day_pass = True
    verdict = (
        "PASS"
        if (ce_pass and top1_pass and new_pass and time_pass and same_day_pass)
        else "REVIEW"
    )

    lines = [
        (
            "# Task 14 two-stage time diagnostic"
            if evaluation.get("two_stage_time_head")
            else "# Task 13 waiting-time diagnostic (loss_dt re-enabled)"
        ),
        "",
        f"- Verdict: **{verdict}**",
        f"- Resumed from: `{resumed_from}`" if resumed_from else "- Started from scratch",
        f"- Best checkpoint step: `{evaluation['checkpoint_step']}`",
        f"- Time loss enabled: `{evaluation['time_loss_enabled']}`",
        "",
        "## Pass criteria vs CE-only extended run",
        "",
        "| Check | Baseline | This run | Pass |",
        "|---|---:|---:|:--:|",
        (
            f"| Clinical-only CE | {CE_ONLY_BASELINE_CE:.2f} | "
            f"{clinical_ce:.4f} | {'yes' if ce_pass else 'no'} |"
        ),
        (
            f"| Clinical-only top-1 | {CE_ONLY_BASELINE_TOP1:.4%} | "
            f"{clinical_top1:.4%} | {'yes' if top1_pass else 'no'} |"
        ),
        (
            f"| New-onset top-1 | {CE_ONLY_BASELINE_NEW_TOP1:.4%} | "
            f"{new_top1:.4%} | {'yes' if new_pass else 'no'} |"
        ),
        (
            f"| Waiting-time beats baseline | yes | "
            f"{'yes' if waiting['beats_baseline'] else 'no'} | "
            f"{'yes' if time_pass else 'no'} |"
        ),
        *(
            [
                (
                    f"| Same-day beats non-informative baselines | yes | "
                    f"{'yes' if same_day_pass else 'no'} | "
                    f"{'yes' if same_day_pass else 'no'} |"
                )
            ]
            if evaluation.get("two_stage_time_head")
            else []
        ),
        "",
        "## Deterministic clinical-only metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Clinical-only CE | {clinical_ce:.4f} |",
        f"| Clinical-only perplexity | {fmt(clinical.get('perplexity'), '.2f')} |",
        f"| Clinical-only top-1 | {clinical_top1:.4%} |",
        f"| Clinical-only top-5 | {fmt(clinical.get('top5_accuracy'), '.4%')} |",
        f"| Clinical-only top-10 | {fmt(clinical.get('top10_accuracy'), '.4%')} |",
        f"| Train-unigram clinical top-1 | {unigram['top1_accuracy']:.4%} |",
        f"| New clinical top-1 | {evaluation['new_clinical']['top1_accuracy']:.4%} |",
        f"| Repeated clinical top-1 | {evaluation['repeated_clinical']['top1_accuracy']:.4%} |",
        "",
        "## Different-day waiting time: model vs train-only baselines",
        "",
        "| Metric | Model | Baseline |",
        "|---|---:|---:|",
        (
            f"| NLL | {fmt(waiting['model_nll'], '.4f')} | "
            f"{fmt(waiting['constant_rate_baseline_nll'], '.4f')} (constant rate) |"
        ),
        (
            f"| Mean absolute error (days) | {fmt(waiting['model_mae_days'], '.1f')} | "
            f"{fmt(waiting['median_baseline_mae_days'], '.1f')} (median gap) |"
        ),
        f"| Median absolute error (days) | {fmt(waiting['model_median_absolute_error_days'], '.1f')} | |",
        f"| p95 absolute error (days) | {fmt(waiting['model_p95_absolute_error_days'], '.1f')} | |",
        f"| Targets | {waiting['targets']} | |",
        "",
        (
            f"NLL improvement over constant rate: "
            f"`{fmt(waiting['nll_improvement_over_constant_rate'], '+.4f')}`; "
            f"MAE improvement over median: "
            f"`{fmt(waiting['mae_improvement_over_median'], '+.1f')}` days "
            f"(positive = model better)."
        ),
        "",
        "## Same-day classification (clinical targets)",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Targets | {same_day.get('targets', 0)} |",
        f"| Same-day prevalence | {fmt(same_day.get('same_day_prevalence'), '.4%')} |",
        f"| AUROC | {fmt(same_day.get('auroc'), '.4f')} |",
        f"| AUPRC | {fmt(same_day.get('auprc'), '.4f')} |",
        f"| Brier score | {fmt(same_day.get('brier_score'), '.4f')} |",
        f"| Constant-prevalence Brier score | {fmt(constant_brier, '.4f')} |",
        (
            f"| Expected calibration error | "
            f"{fmt(same_day.get('expected_calibration_error'), '.4f')} |"
        ),
        "",
        "### Same-day calibration bins",
        "",
        "| Predicted probability bin | Targets | Mean predicted | Observed same-day |",
        "|---|---:|---:|---:|",
        *[
            (
                f"| {item['lower']:.1f}-{item['upper']:.1f} | "
                f"{item['targets']} | "
                f"{fmt(item['mean_probability'], '.4f')} | "
                f"{fmt(item['observed_same_day_rate'], '.4f')} |"
            )
            for item in same_day.get("calibration_bins", [])
        ],
        "",
        "## Validation trajectory",
        "",
        "| step | validation CE | same-day loss | different-day loss | validation objective | validation targets |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation:
        lines.append(
            f"| {row['iter']} | {row['val/loss_ce']:.4f} | "
            f"{row.get('val/loss_same_day', 0.0):.4f} | "
            f"{row['val/loss_dt']:.4f} | "
            f"{row['val/objective_loss']:.4f} | "
            f"{row['val/eval_targets']} |"
        )
    lines.extend([
        "",
        f"Best sampled validation CE `{best_validation['val/loss_ce']:.4f}` "
        f"at step `{best_validation['iter']}`.",
        "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verdict: {verdict}")


def main():
    args = parse_args()

    data_dir = (args.data_dir or default_data_dir()).resolve()
    output = (args.output_dir or default_output_dir()).resolve()
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output}")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing ETL manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    vocab_size = int(manifest["model_vocab_size"])

    init_from, resumed_from = prepare_output(
        output,
        args.resume_from,
        args.overwrite,
    )
    command = [
        sys.executable,
        "train.py",
        args.config,
        f"--dataset_dir={data_dir}",
        f"--out_dir={output}",
        f"--init_from={init_from}",
        f"--device={args.device}",
        f"--dtype={args.dtype}",
        f"--vocab_size={vocab_size}",
    ]
    # Append only the hyper-parameters the caller overrode; otherwise the
    # config file's values stand. Keep lr_decay_iters aligned with max_iters.
    overrides = {
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "log_interval": args.log_interval,
        "loss_dt_weight": args.loss_dt_weight,
        "loss_dt_warmup_iters": args.loss_dt_warmup_iters,
        "checkpoint_metric": args.checkpoint_metric,
    }
    for key, value in overrides.items():
        if value is not None:
            command.append(f"--{key}={value}")
    if args.max_iters is not None:
        command.append(f"--lr_decay_iters={args.max_iters}")
    run(command, output / "training.log")

    checkpoint = output / "ckpt.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not produce {checkpoint}")
    evaluation_path = output / "evaluation.json"
    run([
        sys.executable,
        "scripts/evaluate_snuh_checkpoint.py",
        "--ckpt",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.eval_batch_size),
        "--output",
        str(evaluation_path),
    ], output / "evaluation.log")
    write_summary(output, evaluation_path, resumed_from)
    print(f"Task 13 report: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
