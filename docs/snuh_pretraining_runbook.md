# SNUH Foundation Pretraining Runbook

## Task 10: smoke comparison

On an L4 GPU pod:

```bash
python scripts/run_snuh_task10.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --output-dir out/snuh-task10
```

The command runs the window audit, baseline arm, LAB-context arm, deterministic
checkpoint evaluation, and comparison report.

Task 10 is currently a CE-only diagnostic. Both arms use
`loss_dt_weight=0.0`, select checkpoints by validation CE, and compare against
a train-only clinical unigram baseline. Do not start the scaling sweep unless
the model improves meaningfully over that baseline.

Primary output:

```text
out/snuh-task10/reports/comparison.md
```

Use clinical-only metrics for arm comparison. Objective CE uses different
target sets and is not directly comparable. Waiting-time metrics are `NA` in
this diagnostic because the time loss is disabled.

## Task 10 extension: LAB-context CE-only

Before Task 11 or Task 12, extend the selected LAB-context arm to 3,000 total
steps. On the SNUH pod, the runner automatically finds the 1% ETL directory,
selects the newest compatible LAB-context CE-only checkpoint, and creates a
versioned output directory:

```bash
python scripts/run_snuh_task10_lab_context_long.py
```

If no compatible checkpoint exists, the runner starts the same LAB-context
configuration from scratch. The run remains CE-only and evaluates every 250
steps. Training progress is printed every 100 steps; validation and checkpoint
messages are printed every 250 steps. It prints the selected checkpoint and
output directory before training. Outputs are written under:

```text
/home/khdp-user/workspace/fermat-data/out/<bundle-id>/
```

Keep the extracted code under the same block storage, for example:

```text
/home/khdp-user/workspace/fermat-data/task10-lab-context-long-code/
```

Review the validation CE trajectory and deterministic clinical-only, new, and
repeated target metrics before starting the scaling sweep.

Build a versioned pod bundle after committing the code:

```bash
python scripts/build_snuh_pod_bundle.py
```

The filename contains the task label, Git SHA, clean/dirty state, and a content
hash. Record the bundle ID alongside the pod output directory and checkpoint.
Checkpoints and `dist/` bundles are intentionally excluded from Git.

## Task 11: measurement staging gate

On the CPU/ETL pod:

```bash
python scripts/run_snuh_task11_measurement_staging.py
```

The runner prompts for the SNUH CDM password, runs the indexed 1% staging
audit, and writes versioned outputs under
`/home/khdp-user/workspace/fermat-data/out/<bundle-id>/`.
If `psycopg` is absent, the runner installs `psycopg[binary]>=3` into the
active Pod Python environment before connecting.

Review:

- stage creation and index runtime
- table and index size
- numeric/categorical overlap
- linear full-scale size projection

The projection is only a first estimate. Increase to 5% before adopting a
single-scan staging design for the full ETL.

## Task 12: scaling sweep

Run this only after choosing the baseline or LAB-context target policy:

```bash
python scripts/run_snuh_scaling_sweep.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --policy lab-context \
  --output-dir out/snuh-scaling-lab-context
```

Primary output:

```text
out/snuh-scaling-lab-context/scaling_summary.md
```

Choose a production candidate using validation trend, effective targets per
second, and peak VRAM. The 1% pilot is for relative comparison, not final
perplexity.
