# LoRA fine-tuning baseline

Code for the fine-tuning baseline requested in the NeurIPS GenAI4Health review
(reviewer RRnf, meta-review item 4). [PROTOCOL.md](PROTOCOL.md) fixes the design;
read it before running anything on real data.

The scripts import the frozen model code from `releases/fermat_snuh_20261021`
and never modify the release. Weights (`artifacts/model.pt`) and patient data are
not in git; point `--release` at an unpacked collaborator archive.

```
lora.py                    LoRA wrapper, injection, state export, frozen-base audit
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

# Task R, reproduction arm then LoRA arm, seed 1
python train_record.py --release $R --data $D --out runs/record/head_only_s1 --arm head_only --seed 1 --device cuda --bf16
python train_record.py --release $R --data $D --out runs/record/lora_s1 --arm lora --seed 1 --device cuda --bf16 \
    --baseline-recall /path/to/generation50_recall.csv --baseline-label generation50

# Task D, one disease at a time, same fixed 100k training subset for both arms
E=/path/to/exported/risk_dataset
python train_risk.py --release $R --data $E --out runs/risk/ckd/head_only --arm head_only --disease ckd --train-subset 100000 --device cuda --bf16
python train_risk.py --release $R --data $E --out runs/risk/ckd/lora --arm lora --disease ckd --train-subset 100000 --device cuda --bf16 \
    --baseline-probabilities runs/risk/ckd/head_only/test_patient_probabilities.csv --baseline-label head_only
```

Every run writes `summary.json`, a per-patient test CSV and `trainable_state.pt`
(adapters and head only). Commit only `summary.json` files, under `results/`.
