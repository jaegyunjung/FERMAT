# FERMAT

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

FERMAT is a generative transformer for longitudinal clinical trajectories. It
represents a patient's medical history as an age-ordered sequence of events and
learns patterns across DX, RX, PX, LAB, GENOMICS, LIFESTYLE, DTH, and SEX
tokens.

The project is designed for real-world Korean healthcare data from
National Health Insurance Service, Health Insurance Review & Assessment Service,
Kyungpook National University Hospital, Pusan National University Hospital,
Chonnam National University Hospital, and Seoul National University Hospital.

## Motivation

Clinical risk and disease progression are rarely visible in DX codes alone.
Long before an outcome appears, patients may accumulate RX, PX, LAB, GENOMICS,
LIFESTYLE, DTH, and SEX context. FERMAT models these records in one temporal
sequence so that downstream analyses can use the broader clinical context
rather than a single data modality.

## Data Scope

FERMAT is intended for linked, de-identified healthcare data sources, including:

| Source | Examples of captured information |
|--------|----------------------------------|
| National Health Insurance Service | eligibility, health screening, death and cancer screening records |
| Health Insurance Review & Assessment Service | claims, diagnoses, procedures, prescriptions |
| Kyungpook National University Hospital | hospital-level clinical records |
| Pusan National University Hospital | hospital-level clinical records |
| Chonnam National University Hospital | hospital-level clinical records |
| Seoul National University Hospital | hospital-CDM clinical events, LAB records, and local clinical context |

Raw private data, credentials, and site-specific runtime artifacts are not
stored in this repository.

## Event Vocabulary

FERMAT converts multimodal records into a unified event vocabulary. Each event
keeps the patient sequence, event age, vocabulary token, and event modality.

Current token types include:

| Token type | Meaning |
|------------|---------|
| DX | diagnosis |
| RX | drug prescription |
| PX | procedure or surgery |
| LAB | lab or screening result |
| GENOMICS | genomics or tumor biomarker token |
| LIFESTYLE | screening questionnaire context |
| DTH | death event with cause code |
| SEX | static sex token |

This representation allows the model to learn from clinical order, age, event
co-occurrence, and modality jointly.

## Model

FERMAT uses a causal transformer over longitudinal health events. Inputs combine
event identity, event age, and modality information. The model is trained for
next-event prediction and temporal prediction so that evaluation can examine
both the likely next clinical event and the expected timing of future events.

The current research model supports:

- multimodal event embeddings;
- age-aware longitudinal sequence modeling;
- clinical next-event prediction;
- event-time prediction; and
- checkpoint-based evaluation on held-out patients.

## Current Progress

The current implementation includes private hospital-CDM pretraining,
multimodal tokenization, transformer training, and checkpoint evaluation. A
full-cohort model has been trained and is under evaluation.

## Repository Contents

| Path | Purpose |
|------|---------|
| `model.py` | FERMAT transformer model |
| `train.py` | training loop and checkpointing |
| `utils.py` | data loading and batching utilities |
| `config/` | training configurations |
| `scripts/` | preprocessing, training, evaluation, and audit utilities |
| `docs/` | internal development notes and runbooks |
| `foundation_tests/` | focused regression tests |

## License

MIT
