# LoRA fine-tuning baseline

Code for the fine-tuning baseline requested in the NeurIPS GenAI4Health review
(reviewer RRnf, meta-review item 4). [PROTOCOL.md](PROTOCOL.md) fixes the design;
read it before running anything on real data.

The scripts import the frozen model code from `releases/fermat_snuh_20261021`
and never modify the release. Weights (`artifacts/model.pt`) and patient data are
not in git; point `--release` at an unpacked collaborator archive.

```
lora.py                    LoRA wrapper, injection, state export, frozen-base audit
check_gate.py              reproduction gate: head_only summary vs protocol_targets.json -> gate.json with run identity
protocol_targets.json      v1 reference values and tolerances per task and disease
predict.py                 rebuild a trained arm from trainable_state.pt and reproduce its outputs
data.py                    dataset layout (documented in the module docstring) and validation
metrics.py                 Recall@K / F1 / precision / hit, weighted AUROC and Brier, paired bootstrap
common.py                  arguments, model loading, hidden states, provenance fingerprint
train_record.py            task R: items on the first clinical date after the history
train_risk.py              task D: five-year disease risk for one disease
make_synthetic_dataset.py  random fixture for pipeline tests (not patient data)
tests/                     unit tests and an end-to-end CPU smoke test, no artifacts needed
```

## Check the pipeline without data or weights

```bash
pip install -r releases/fermat_snuh_20261021/requirements.txt
python -m unittest discover -s experiments/lora_baseline/tests -v
```

## Run on the exported cohorts

```bash
cd experiments/lora_baseline
R=/path/to/unpacked/FERMAT_SNUH_20261021_compared   # holds artifacts/model.pt
D=/path/to/exported/record_dataset                   # data.py layout

# Task R: reproduction arm, gate, then LoRA arm (seed 1)
python train_record.py --release $R --data $D --out runs/record/head_only_s1 --arm head_only --seed 1 \
    --train-patient-ids /path/to/v1_record_train_ids.npy --device cuda --bf16
python check_gate.py --summary runs/record/head_only_s1/summary.json --out runs/record/gate.json
python train_record.py --release $R --data $D --out runs/record/lora_s1 --arm lora --seed 1 --gate runs/record/gate.json \
    --train-patient-ids /path/to/v1_record_train_ids.npy --device cuda --bf16 \
    --baseline-recall /path/to/generation50_recall.csv --baseline-label generation50
python predict.py --release $R --data $D --state runs/record/lora_s1/trainable_state.pt --out runs/record/lora_s1_reload --device cuda --bf16

# Task D, one disease at a time: full-pool reproduction, gate, then both arms on the same 100k list
E=/path/to/exported/risk_dataset
python train_risk.py --release $R --data $E --out runs/risk/ckd/head_only_full --arm head_only --disease ckd --device cuda --bf16
python check_gate.py --summary runs/risk/ckd/head_only_full/summary.json --out runs/risk/ckd/gate.json
python train_risk.py --release $R --data $E --out runs/risk/ckd/head_only_100k --arm head_only --disease ckd \
    --train-patient-ids /path/to/risk_train_100k_ids.npy --device cuda --bf16
python train_risk.py --release $R --data $E --out runs/risk/ckd/lora_s1 --arm lora --disease ckd --seed 1 --gate runs/risk/ckd/gate.json \
    --train-patient-ids /path/to/risk_train_100k_ids.npy --device cuda --bf16 \
    --baseline-probabilities runs/risk/ckd/head_only_100k/test_patient_probabilities.csv --baseline-label head_only
```

Before the first gate run, complete the v1 recipe comparison in PROTOCOL.md
Section 12 against the local v1 scripts; synthetic-data test passes are not a
v1 reproduction.

Every run writes `summary.json`, a per-patient test CSV and `trainable_state.pt`
(adapters, head and preprocessing constants; `predict.py` reproduces the run's
test outputs from it). Commit only `summary.json` and `gate.json` files, under
`results/`.
