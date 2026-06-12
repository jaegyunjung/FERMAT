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

Run the LAB-context CE-only sweep on the L4 GPU pod:

```bash
python scripts/run_snuh_scaling_sweep.py
```

The runner automatically uses the 1% ETL pilot on block storage and compares
the following four candidates for 500 steps each:

```text
tiny:  2 layers, 128 embedding, context 256 and 512
small: 4 layers, 256 embedding, context 256 and 512
```

Each successful trial receives deterministic clinical-only evaluation. Results
are written under
`/home/khdp-user/workspace/fermat-data/out/<bundle-id>/scaling_summary.md`.
Choose the next production candidate using validation CE, clinical top-k,
effective targets per second, and peak VRAM. The 1% pilot is for relative
comparison, not final perplexity.

## Task 13: waiting-time diagnostic (loss_dt re-enabled)

Tasks 10-12 ran CE-only because the waiting-time loss used an event rate equal
to `sum(exp(logit)) ~ vocab_size`, several orders of magnitude above the true
rate, which flattened the token logits and produced a near-uniform model. The
model now carries a learnable global log-rate scalar that sets the absolute
event rate independently of the per-token logits, so the time loss is the same
order of magnitude as cross-entropy. This task confirms the token head survives
when the time loss is turned back on.

Run on the L4 GPU pod:

```bash
python scripts/run_snuh_task13_dt.py
```

The runner finds the 1% ETL pilot, trains the LAB-context arm from scratch with
the time loss ramped in over the first 500 steps, evaluates the best
checkpoint, and writes a pass/review summary under
`/home/khdp-user/workspace/fermat-data/out/<bundle-id>/summary.md`. Pass
`--resume-from <ce-only-dir>` to continue an existing CE-only checkpoint
instead of training from scratch.

Pass criteria, compared against the CE-only extended run:

- clinical-only CE stays near or below `6.53` (does not collapse toward
  `ln(vocab)`)
- clinical-only top-1 stays at or above `3.64%`
- waiting-time error metrics are finite and reported (no longer `NA`)

Only after this passes should the time loss be enabled in the Task 10 extension
and Task 12 sweep configurations for full-cohort scaling.

### Result and decision

The structural fix works: with the global log-rate scalar the time loss no
longer collapses the model to a uniform distribution. Three 1% runs confirmed
the token head learns with the time loss enabled:

| Run | clinical CE | top-1 | time median error |
|---|---:|---:|---:|
| CE-only baseline | 6.53 | 3.64% | (off) |
| scratch, dt weight 1.0 | 7.22 | 3.21% | 83.6 d |
| scratch, dt weight 0.1 | 6.98 | 3.09% | 77.4 d |
| finetune from CE-only, dt weight 0.1 | 6.88 | 3.57% | 29.8 d |

Enabling the time loss costs about 0.35 clinical CE that neither a lower weight
nor warm-starting from the converged checkpoint removes. The cause is
structural: the event rate is derived from the shared token logits, so the time
gradient perturbs token ranking and settles at an equilibrium near CE 6.9. A
fully decoupled time head (predicting the log-rate from the hidden state with a
separate linear layer) would remove this cost but departs from FERMAT's coupled
point-process formulation.

Decision: proceed with the coupled formulation (option B). A ~30-day median
waiting-time error is acceptable for the SNUH cohort (~3.77M patients, ~20
years). The decoupled time head (option A) is deferred; it is a global
architecture switch, not a per-phenotype change, and the 1% runs are throwaway
diagnostics, so the choice can be made later at no extra cost. Revisit option A
only if full-scale training increases the error or a phenotype needs tighter
time accuracy; phenotype-specific time accuracy is more naturally addressed by
downstream fine-tuning. Define the time-accuracy target when the gastric-cancer
downstream task is designed.

The recommended single-pass recipe for scaled runs is to train cross-entropy
first and ramp the time loss in (warmup), with a light `loss_dt_weight` around
`0.1`; warm-starting the time loss onto a converged CE checkpoint gave the best
waiting-time accuracy (29.8-day median) at nearly baseline top-1.
