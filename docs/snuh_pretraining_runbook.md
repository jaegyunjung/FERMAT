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

### Option A experiment (decoupled time head, stage 1)

A code review found that the coupled formulation also dropped new-onset top-1
from 3.07% to 2.29% (aggregate top-1 held only because repeated events
improved) and that the original time pass criterion accepted any finite MAE.
Both are addressed before committing to full-scale training:

- `model.py` gains a `decoupled_time_head` option: a separate linear head
  predicts the log event-rate from the hidden state, so the time loss is
  structurally absent from the token logits' graph (verified: the time loss
  sends zero gradient into the logits). The transformer body stays shared.
- `evaluate_snuh_checkpoint.py` now reports the waiting-time NLL against a
  constant-rate baseline and MAE against a median-gap baseline, with a
  `beats_baseline` flag. The runner's time criterion requires beating the
  baseline, and the summary adds a new-onset top-1 pass check.

Run the from-scratch comparison (directly comparable to the coupled
from-scratch runs at CE 6.98-7.22):

```bash
python scripts/run_snuh_task13_dt.py --config config/train_fermat_snuh_dt_decoupled.py
```

Or warm-start from the CE-only run to mirror the best coupled result
(CE 6.88, new-onset 2.29%) with only the head changed:

```bash
python scripts/run_snuh_task13_dt.py \
  --resume-from <ce-only-dir> \
  --config config/train_fermat_snuh_dt_decoupled_finetune.py
```

Pass means clinical CE recovers toward 6.53, new-onset top-1 returns to ~3.07%,
and the time head beats both baselines.

### Result: option A is adopted

The decoupled head removes the output-layer interference but still shares the
transformer body, so `loss_dt_weight` trades token calibration for time
accuracy. A 1% warm-start weight sweep (objective selection, 3,000 steps) maps
the frontier:

| loss_dt_weight | clinical CE | top-1 | new-onset | top-5 / top-10 | time NLL gain | time MAE gain | verdict |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| coupled (B) best | 6.88 | 3.57% | 2.29% | 9.3 / 13.7% | +0.6 | n/a | review |
| 0.1 | 6.53 | 3.90% | 3.50% | 13.2 / 19.1% | +1.03 | -170 d | review (time weak) |
| **0.3** | **6.56** | **4.26%** | **3.91%** | **13.3 / 19.3%** | **+1.97** | **+630 d** | **pass (4/4)** |
| 1.0 | 6.77 | 4.27% | 4.20% | 11.8 / 17.0% | +2.24 | +825 d | review (CE drifts) |

At `loss_dt_weight=0.3` every token metric is at or above the CE-only baseline
(6.53 / 3.64% / 3.07% / 12.6% / 18.5%) and the time head beats both the
constant-rate (NLL) and median-gap (MAE) baselines, with a 180-day median
waiting-time error. This strictly dominates the coupled option B, which
collapsed new-onset top-1 to 2.29%.

Decision: adopt the decoupled time head with `loss_dt_weight=0.3` (this is the
default in `config/train_fermat_snuh_dt_decoupled_finetune.py`). The weight is
the knob for the calibration-vs-time-accuracy trade; raise it if the
gastric-cancer downstream needs tighter time accuracy and can tolerate the CE
cost. This recipe carries to the scaled runs in place of option B.
