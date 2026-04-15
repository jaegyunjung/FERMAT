# FERMAT

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

FERMAT is a generative transformer that learns the temporal progression of an individual's health from nationwide multimodal clinical data. It models the interplay between diagnoses, drug prescriptions, procedures, repeated biomarker measurements, and lifestyle changes as a single autoregressive sequence — predicting what clinical event comes next and when it will occur.

## Motivation

A patient's health trajectory is not a list of diagnoses. Before diabetic nephropathy appears, there are years of gradually worsening fasting glucose levels, antihyperglycemic prescriptions, dose escalations, and screening results that collectively define the clinical path toward that outcome. Existing models for health trajectory prediction discard most of this pre-diagnostic context, operating only on diagnosis codes.

FERMAT addresses this by treating the full spectrum of clinical events — diagnoses, prescriptions, procedures, lab results, and lifestyle assessments — as a unified token sequence ordered by the patient's age. This design is enabled by Korea's National Healthcare Big Data Integration Platform, which links data across multiple public institutions including the National Health Insurance Service (NHIS), the Health Insurance Review and Assessment Service (HIRA), the national death registry, the Korea Disease Control and Prevention Agency (KDCA), the National Cancer Center, and university hospital CDMs, covering the complete healthcare utilization of the Korean population across all providers.

## Data Sources

FERMAT is designed to train on linked multimodal data from Korea's single-payer healthcare system. The platform provides:

| Source | Data | What it captures |
|--------|------|-----------------|
| **NHIS** (National Health Insurance Service) | Eligibility (BFC), health screening with 113 variables (G1EQ), death records (TG_DTH), cancer screening (5 types) | Biennial biomarker trajectories (BP, glucose, cholesterol, BMI, liver function), lifestyle questionnaires (smoking, alcohol, exercise), demographics, mortality |
| **HIRA** (Health Insurance Review & Assessment Service) | Claims summary (TWJHC200), services (TWJHC300), diagnoses (TWJHC400), prescriptions (TWJHC530) | Every diagnosis, drug prescription, and procedure across all healthcare providers nationwide |
| **National Death Registry** (Statistics Korea) | Cause-of-death records (DTH, KCD 8th edition) | Out-of-hospital deaths with cause codes |
| **KDCA** (Korea Disease Control & Prevention Agency) | KNHANES (600–1,100 vars/year), KoGES, vaccination records, TB registry | Detailed nutrition, physical activity, mental health, cohort-level longitudinal data |
| **National Cancer Center** | Cancer registry (1999–2022) | Pathologically confirmed cancer diagnoses (24 major cancer types) |
| **University Hospital CDMs** (Pusan, Chonnam, Kyungpook) | OMOP standard tables (CONDITION, DRUG, MEASUREMENT, PROCEDURE, DEATH, PERSON, VISIT) | Hospital-level clinical detail including lab values — used for external validation |

These sources are linked at the individual level through a trusted third party (TTP) mechanism, creating longitudinal patient trajectories that span all levels of care.

## Multimodal Token Vocabulary

FERMAT represents every clinical event as a token with three attributes: **what** (token ID), **when** (age in days), and **what kind** (token type). This is stored as a 4-column binary format:

```
patient_id | age_in_days | token_id | token_type
---------- | ----------- | -------- | ----------
001        | 9131        | 42       | DX          ← hypertension diagnosed
001        | 9496        | 815      | RX          ← amlodipine prescribed
001        | 9861        | 1102     | LAB         ← fasting glucose 110 (prediabetes range)
001        | 10227       | 1103     | LAB         ← fasting glucose 130 (diabetes range)
001        | 10592       | 55       | DX          ← type 2 diabetes diagnosed
001        | 10957       | 830      | RX          ← metformin prescribed
```

### Token types

| Type | ID | Source | Role in sequence | Temporal behavior |
|------|----|--------|-----------------|-------------------|
| DX | 1 | KCD/ICD-10 (HIRA TWJHC400) | Predicted — primary outcome | Longitudinal: each diagnosis is a distinct event |
| RX | 2 | ATC code (HIRA TWJHC530) | Predicted — pharmacological intervention | Longitudinal: each prescription is a distinct event |
| PX | 3 | EDI procedure code (HIRA TWJHC300) | Predicted — procedural intervention | Longitudinal |
| LAB | 4 | NHIS screening results (G1EQ) | Contextual — biomarker state at screening | Longitudinal: updated at each biennial screening |
| LIFESTYLE | 5 | NHIS screening questionnaire | Contextual — behavioral state | Longitudinal: updated at each biennial screening |
| DTH | 6 | Death registry + KCD cause code | Predicted — terminal event | Once per patient; terminates trajectory |
| SEX | 7 | NHIS BFC | Conditioning — static attribute | Static: entered once, never predicted |
| NO_EVENT | 8 | Synthetic | Structural — fills long event-free gaps | Synthetic: prevents artificial long-range dependencies |
| PAD | 0 | N/A | Structural — sequence padding | N/A |

The estimated vocabulary size is 2,000–3,400 tokens across all types.

## Architecture

Each token is embedded as the sum of three learned representations:

```
Event representation = TokenEmb(what) + AgeEncoding(when) + TypeEmb(what kind)
```

- **TokenEmb**: learned embedding for each clinical code (shared with output projection via weight tying)
- **AgeEncoding**: continuous sinusoidal encoding of age in days, followed by a learned linear projection
- **TypeEmb**: learned embedding for each token type — enables the model to learn type-specific interaction patterns (e.g., how a prior RX token modifies the probability of a subsequent DX token)

The embedded sequence passes through a stack of transformer blocks with causal attention (each event attends only to prior events). Co-occurring events at the same age are additionally masked from attending to each other.

Two output heads produce:
- **Next-token head**: probability distribution over the full vocabulary — which clinical event is most likely next
- **Time-to-event head**: expected time until the next event, modeled as an exponential waiting time

Training minimizes the sum of cross-entropy loss (next token identity) and exponential log-likelihood loss (time to next event).

## Ablation Design

The central experiment is a 6-step ablation that progressively adds pre-diagnostic context:

| Step | Tokens included | Question answered |
|------|----------------|-------------------|
| 1 | DX only | Baseline: what can diagnosis sequences alone predict? |
| 2 | DX + RX | Does knowing what drugs were prescribed improve prediction? |
| 3 | DX + RX + PX | Does procedural history add information beyond drug history? |
| 4 | DX + RX + PX + LAB (static) | Does a one-time biomarker snapshot help? |
| 5 | DX + RX + PX + LAB (dynamic) | Does the *trajectory* of biomarker change outperform a static snapshot? |
| 6 | Full (+ LIFESTYLE) | Does behavioral context add further value? |

The comparison between steps 4 and 5 is the key experiment: it isolates whether it is the *level* of a biomarker or the *temporal change* that carries predictive information.

## Current Implementation Status

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed evidence.

| Component | Status |
|-----------|--------|
| Model architecture (`model.py`) | Prototype complete. Smoke-tested on synthetic 3-column data. |
| Data loader (`utils.py`) | Prototype complete. 4-column format defined but not tested with real data. |
| Training loop (`train.py`) | Prototype complete. Runs on synthetic data with checkpoint save/load. |
| Ablation configs | Defined. Not executable until real data is available. |
| Data preprocessing (NHIS/HIRA → tokens) | **Not implemented.** |
| Token vocabulary mapping tables | **Not implemented.** |
| Evaluation and generation | **Not implemented.** |

## Reproducing the Smoke Test

```bash
# Clone Delphi's public synthetic data for code verification
git clone https://github.com/gerstung-lab/Delphi.git
cp -r Delphi/data/ukb_simulated_data data/

# Run smoke test (CPU, ~1 minute)
python train.py config/train_fermat_demo.py --device=cpu --max_iters=50
```

This verifies that the code runs without errors. It does not validate model performance — the synthetic data contains only diagnosis tokens (no RX/PX/LAB), so the token type embedding receives no meaningful signal.

## Related Work

FERMAT is a transformer-based model for longitudinal clinical event modeling.

Related work on generative modeling of disease trajectories includes:

```bibtex
@article{shmatko2025delphi,
  title={Learning the natural history of human disease with generative transformers},
  author={Shmatko, Artem and Jung, Alexander Wolfgang and Gaurav, Kumar and others},
  journal={Nature},
  volume={647},
  pages={248--256},
  year={2025}
}

## License

MIT
