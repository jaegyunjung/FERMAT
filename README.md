# FERMAT

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

## What FERMAT Does

FERMAT learns to predict a patient's next clinical event from their past medical history. "Clinical event" includes not only diagnoses, but also drug prescriptions, procedures, and changes in biomarkers over time — the full pre-diagnostic clinical context that precedes disease onset.

Given a patient's history up to the present, FERMAT outputs:
1. **What happens next** — a probability distribution over ~3,400 possible clinical events
2. **When it happens** — estimated time until the next event

By repeating this process, FERMAT can generate synthetic future health trajectories spanning up to 20 years.

## Key Differences from Delphi

| Feature | Delphi | FERMAT |
|---------|--------|--------|
| Vocabulary | ~1,258 (diagnosis + static lifestyle) | ~2,000–3,400 (DX + RX + PX + dynamic LAB + LIFESTYLE + DTH) |
| Token type embedding | None | Learnable embedding per type (DX, RX, PX, LAB, etc.) |
| Data format | 3-column `(pid, age, token_id)` | 4-column `(pid, age, token_id, token_type_id)` |
| Biomarkers | Static (one-time baseline) | Dynamic (updated at each biennial screening) |
| Training data | UK Biobank (400K, volunteer cohort) | Korean NHIS/HIRA claims (1M+, nationwide single-payer) |

## How It Works

### Step 1: Converting clinical events to numbers

A patient's medical history is stored as a sequence of rows:

```
patient_id | age_in_days | token_id | token_type
---------- | ----------- | -------- | ----------
001        | 9131        | 42       | 1 (DX)       ← hypertension diagnosed at age 25
001        | 9496        | 815      | 2 (RX)       ← amlodipine prescribed at age 26
001        | 9861        | 1102     | 4 (LAB)      ← fasting glucose 110 at age 27
001        | 10227       | 1103     | 4 (LAB)      ← fasting glucose 130 at age 28
001        | 10592       | 55       | 1 (DX)       ← type 2 diabetes diagnosed at age 29
001        | 10957       | 830      | 2 (RX)       ← metformin prescribed at age 30
```

The 4th column (`token_type`) tells the model *what kind* of event each row represents:

| Type ID | Name | Meaning |
|---------|------|---------|
| 0 | PAD | Empty padding |
| 1 | DX | Diagnosis |
| 2 | RX | Drug prescription |
| 3 | PX | Procedure / surgery |
| 4 | LAB | Lab / screening result (discretized) |
| 5 | LIFESTYLE | Smoking, alcohol, exercise |
| 6 | DTH | Death (with cause) |
| 7 | SEX | Sex (static) |
| 8 | NO_EVENT | No-event padding (Delphi-style) |

Delphi uses only 3 columns (no `token_type`), so it cannot distinguish a diagnosis from a prescription. FERMAT's data loader auto-detects 3-column vs. 4-column format for backward compatibility.

### Step 2: Three-way embedding

Each event is converted to a meaningful vector by summing three components:

```
Event representation = TokenEmb(token_id) + AgeEncoding(age) + TypeEmb(token_type)
                       ├─ "what code"       ├─ "when"           ├─ "what kind"
                       │  (E11 = T2DM)      │  (age 29)         │  (DX = diagnosis)
```

The **Type Embedding** is FERMAT's key architectural addition. Without it, the model cannot tell that token 42 (hypertension) is a diagnosis while token 815 (amlodipine) is a prescription — it treats them as interchangeable symbols. With it, the attention mechanism learns type-specific interaction patterns: how a preceding RX token modifies the probability of a subsequent DX token, how a worsening LAB trajectory conditions future disease onset.

### Step 3: Transformer processing

The embedded sequence passes through 12 transformer blocks. Each block uses causal attention — each event can only "look at" events that came before it in time. Through this process, each position accumulates context from all prior events.

### Step 4: Dual prediction heads

Two output heads produce:
- **Next-token head** (cross-entropy loss): probability distribution over all ~3,400 tokens
- **Time-to-event head** (exponential log-likelihood loss): expected days until the next event

### Step 5: Generation

For trajectory generation, the model samples from the predicted distributions:
1. Convert each logit to a rate (events per day)
2. Sample waiting times from exponential distributions
3. The event with the shortest sampled waiting time becomes the next token
4. Append it, advance the age, repeat

## Ablation Experiment Design

The central experimental result is the 6-step ablation showing how each modality of pre-diagnostic context improves prediction:

| # | Configuration | What it tests | Expected insight |
|---|--------------|---------------|-----------------|
| 1 | DX only | Diagnosis sequences alone | Delphi-equivalent baseline |
| 2 | DX + RX | + drug prescriptions | Which diseases benefit from treatment history? |
| 3 | DX + RX + PX | + procedures/surgeries | Does procedural history add beyond drug history? |
| 4 | DX + RX + PX + LAB (static) | + screening results, one-time | Value of biomarkers as static features |
| 5 | DX + RX + PX + LAB (dynamic) | + screening results, updated biennially | **Value of temporal biomarker change vs. static snapshot** |
| 6 | Full model | All modalities + lifestyle | Complete pre-diagnostic context |

The comparison between experiments 4 and 5 is particularly important — it isolates whether it's the *level* of a biomarker or the *trajectory of change* that matters for prediction.

## Quick Start

### Test with Delphi synthetic data (3-column compatibility)
```bash
# Copy Delphi's publicly available synthetic data
cp -r path/to/Delphi/data/ukb_simulated_data data/

# Run demo training (CPU, ~2 minutes)
python train.py config/train_fermat_demo.py --device=cpu
```

### Train on claims data from South Korea
```bash
# After data preprocessing
python train.py config/train_fermat_kr.py --device=cuda
```

### Run ablation experiments
```bash
python train.py config/train_fermat_kr.py --ablation=dx_only --device=cuda
python train.py config/train_fermat_kr.py --ablation=dx_rx --device=cuda
python train.py config/train_fermat_kr.py --ablation=full --device=cuda
```

## Project Structure

```
FERMAT/
├── model.py                      # Fermat model + FermatConfig + TokenType enum
├── train.py                      # Training loop (handles 3-col and 4-col data)
├── utils.py                      # Data loading, batching, patient indexing
├── configurator.py               # CLI config override (from nanoGPT)
├── config/
│   ├── train_fermat_demo.py      # Demo config (Delphi synthetic data)
│   ├── train_fermat_kr.py        # Full training config template
│   └── ablation_configs.py       # 6-step ablation experiment definitions
├── data/
│   └── ukb_simulated_data/       # Delphi synthetic data for code testing
└── README.md
```

## Citation

For prior work on generative modeling of disease trajectories, see:
```bibtex
@article{shmatko2025delphi,
  title={Learning the natural history of human disease with generative transformers},
  author={Shmatko, Artem and Jung, Alexander Wolfgang and Gaurav, Kumar and others},
  journal={Nature},
  volume={647},
  pages={248--256},
  year={2025}
}
```

## License

MIT
