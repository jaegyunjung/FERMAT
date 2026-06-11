# SNUH Task 10 Smoke Training

Run these checks against the existing 1% ETL artifacts before full ETL.

## One-command run

```bash
python scripts/run_snuh_task10.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --output-dir out/snuh-task10
```

This runs all steps below and writes the final comparison to
`out/snuh-task10/reports/comparison.md`.

## 1. Audit sampled windows

```bash
python scripts/audit_snuh_training_windows.py \
  --data-dir outputs/snuh_tokenization_etl/patient_001pct_seed_42 \
  --block-size 256 \
  --batch-size 64 \
  --batches 100 \
  --output outputs/snuh_tokenization_etl/patient_001pct_seed_42/window_audit.json
```

Review:

- LAB fraction per sampled window
- baseline effective targets per window
- LAB-context effective targets per window
- same-day transitions per window

The sampler is patient-uniform and then chooses a random window within each
selected patient. It is not event-uniform.

## 2. Baseline arm

```bash
python train.py config/train_fermat_snuh_pilot.py
```

LAB tokens are inputs and prediction targets.

## 3. LAB-context arm

```bash
python train.py config/train_fermat_snuh_pilot_lab_context.py
```

LAB tokens remain in the input context but LAB target positions are excluded
from cross-entropy and waiting-time losses. This arm does not yet relink each
position to the next non-LAB event or remove LAB tokens from the output
softmax.

## Gate

Both arms must:

- load the 4-column shards without mapping errors
- produce finite CE and time losses
- show decreasing training objective
- complete validation over fixed left, middle, and right windows
- fit in GPU memory at `block_size=256`

Compare the arms using convergence, throughput, GPU memory, and validation
loss trends. Their CE values are not directly interchangeable because the
target sets differ.

The checkpoint evaluator therefore also reports clinical-only softmax metrics
over the same DX/RX/PX/DTH target and output set for both arms.

Use the window audit to decide whether the next experiment should use LAB
event dropout/downsampling, longer context, or next-non-LAB target relinking.
