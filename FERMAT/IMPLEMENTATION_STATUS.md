# FERMAT Implementation Status Report

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

Date: 2026-04-15

---

## A. Current Status Summary

| Category | Status |
|----------|--------|
| Concept design (architecture, token types, ablation plan) | **Completed** |
| Prototype code (model, dataloader, training loop) | **Completed** |
| Smoke test on synthetic data (Delphi 3-col compat) | **Completed** |
| Smoke test on real 4-col multi-modal data | **Not completed** — no 4-col data exists yet |
| Data preprocessing pipeline (NHIS/HIRA → 4-col bin) | **Not implemented** |
| Token vocabulary mapping (KCD→DX, ATC→RX, etc.) | **Not implemented** — design only |
| Real data training | **Not started** — blocked on data access |
| Evaluation / ablation experiments | **Not started** |
| Git repo with runnable code | **Completed** — runs on Delphi synthetic data |

**One-line summary:**

- **Completed:** Architecture design, prototype code (model.py, train.py, utils.py), smoke test on Delphi synthetic 3-col data, checkpoint save/load
- **Partially completed:** 4-col data format spec (defined but no real data to test with), ablation config (defined but not run)
- **Not yet implemented:** All data preprocessing, vocabulary construction, real-data training, evaluation, generation, interpretability analysis

---

## B. File Structure

```
FERMAT/
├── model.py                  # Fermat class, FermatConfig, TokenType enum
├── train.py                  # Training loop with 3-col/4-col auto-detection
├── utils.py                  # load_data(), get_p2i(), get_batch()
├── configurator.py           # CLI config override (from nanoGPT, unmodified)
├── plotting.py               # Plotting utilities (from Delphi, unmodified)
├── requirements.txt          # Dependencies (from Delphi)
├── LICENSE                   # MIT
├── README.md                 # Project documentation
├── config/
│   ├── train_fermat_demo.py  # Demo config for Delphi synthetic data
│   ├── train_fermat_kr.py    # Template config for Korean data (not runnable yet)
│   └── ablation_configs.py   # 6-step ablation definitions (not runnable yet)
└── data/
    └── ukb_simulated_data/   # Delphi's public synthetic data (for smoke test)
        ├── train.bin         # 181,293 rows, 7,143 patients, 3-col uint32
        ├── val.bin           # Same format
        └── labels.csv        # 1,270 token labels
```

### File roles

| File | What it does | Modified from Delphi? |
|------|-------------|----------------------|
| `model.py` | Defines `Fermat` model class, `FermatConfig` dataclass, `TokenType` enum. Contains all neural network layers, forward pass logic, loss computation, optimizer setup, and trajectory generation. | **Yes** — added `TokenType` enum, `nn.Embedding(n_token_types, n_embd)` as `wtype`, modified forward to accept `token_type` tensor, modified generate to track types |
| `train.py` | Main training script. Loads data, initializes model, runs training loop with eval/checkpoint. | **Yes** — passes `token_type` through all calls, uses `_unpack_batch()` to handle 3-col/4-col |
| `utils.py` | `load_data()`: reads .bin file and auto-detects 3 vs 4 columns. `get_p2i()`: builds patient→index mapping. `get_batch()`: constructs (input, target) pairs with padding and augmentation. | **Yes** — `load_data()` new function, `get_batch()` returns 5th element `xt` (token types or None) |
| `config/ablation_configs.py` | Dict of 6 ablation experiment configurations varying which token types are included in training | **New file** |
| `configurator.py` | Parses CLI arguments to override config values | No (from nanoGPT) |
| `plotting.py` | Plotting helpers | No (from Delphi) |

---

## C. Interface Specification

### C.1 Data format

**3-column (Delphi format):**
```
np.ndarray, dtype=uint32, shape=(N, 3)
Column 0: patient_id      — sequential grouping key
Column 1: age_in_days     — patient age when event occurred
Column 2: token_id        — clinical code (1-indexed; 0 reserved for padding)
```

**4-column (FERMAT format):**
```
np.ndarray, dtype=uint32, shape=(N, 4)
Column 0: patient_id
Column 1: age_in_days
Column 2: token_id
Column 3: token_type_id   — which kind of clinical event (see table below)
```

**Auto-detection logic** (in `utils.py:load_data()`):
- If total uint32 count is divisible by 4 but not 3 → 4-column
- If divisible by 3 but not 4 → 3-column
- If divisible by both → checks if 4th column values are all < 20 (type IDs are small integers) → 4-column; otherwise → 3-column

This is a **heuristic** and may fail on edge cases. When real 4-col data is produced, explicit format specification should replace this.

### C.2 TokenType specification

| Name | Int value | Meaning | In loss? | Static/Longitudinal | Status |
|------|-----------|---------|----------|-------------------|--------|
| PAD | 0 | Empty padding position | No — excluded via `ignore_tokens=[0]` | N/A | Confirmed |
| DX | 1 | Diagnosis (KCD/ICD-10 code) | **Yes** — this is the primary prediction target | Longitudinal (each diagnosis is a new event) | Confirmed |
| RX | 2 | Drug prescription (ATC code) | **Yes** — predicting prescriptions is part of the task | Longitudinal | Confirmed |
| PX | 3 | Procedure/surgery (EDI code) | **Yes** — same as RX | Longitudinal | Confirmed |
| LAB | 4 | Discretized screening result (e.g., FBS_Q3) | **Decision needed** — include in loss or use as context only? | Longitudinal (updated every 2 years at health screening) | **Partially confirmed** — type defined, loss inclusion TBD |
| LIFESTYLE | 5 | Discretized lifestyle from screening (smoking, alcohol, exercise) | **Decision needed** — same as LAB | Longitudinal (same 2-year cycle) | **Partially confirmed** |
| DTH | 6 | Death event with cause code | **Yes** — terminal event, must be predicted | Longitudinal (occurs once) | Confirmed |
| SEX | 7 | Biological sex | **No** — excluded via `ignore_types`. Static attribute that conditions predictions but is not itself predicted | Static (entered once at sequence start) | Confirmed |
| NO_EVENT | 8 | Synthetic no-event token inserted at random intervals to fill gaps (Delphi method) | **No** — excluded via `ignore_types`. Exists only to give the model a gradient signal during long event-free periods | Synthetic | Confirmed |

**Why SEX is not in loss:** Sex is a conditioning variable — the model should learn that male/female patients have different disease rates, but it should not try to "predict" sex as a next event. Delphi handles this the same way (tokens 2-3 in ignore_tokens).

**Why NO_EVENT is not in loss:** No-event tokens are synthetic artifacts inserted to prevent the model from seeing artificially long gaps between real events. They are not real clinical events. Delphi handles this identically (token 1 excluded during validation).

**Why DTH IS in loss:** Death is a real clinical outcome that the model should predict. The time-to-death estimate is clinically meaningful. In generation, death tokens serve as trajectory terminators.

**Open design decisions on LAB/LIFESTYLE:** Two valid approaches exist:
1. Include in loss → model learns to predict "next screening will show elevated glucose," which is clinically interpretable
2. Exclude from loss → model uses them as context only, similar to how Delphi treats lifestyle tokens

This must be decided before real data training. The ablation experiments (config 5 vs 6) can empirically test both approaches.

### C.3 Model interface (`Fermat.forward`)

```python
def forward(self, idx, age, token_type, targets=None, targets_age=None,
            validation_loss_mode=False):
```

**Inputs:**

| Argument | Shape | Type | Description |
|----------|-------|------|-------------|
| `idx` | (B, T) | int64 | Token IDs for input sequence |
| `age` | (B, T) | float32 | Age in days for each token |
| `token_type` | (B, T) | int64 | Token type ID for each token (0-8) |
| `targets` | (B, T) | int64 | Target token IDs (= idx shifted right by 1). None during inference. |
| `targets_age` | (B, T) | float32 | Target ages (= age shifted right by 1). None during inference. |

**Internal computation:**

1. `tok_emb = wte(idx)` — look up token embedding, shape (B, T, n_embd)
2. `age_emb = wae(age)` — sinusoidal age encoding + linear projection, shape (B, T, n_embd)
3. `type_emb = wtype(token_type)` — look up type embedding, shape (B, T, n_embd)
4. `x = dropout(tok_emb) + age_emb + type_emb` — sum of three embeddings
5. Build causal attention mask (lower-triangular + padding mask + optional co-occurrence mask)
6. Pass through N transformer blocks, each: LayerNorm → CausalSelfAttention → residual → LayerNorm → MLP → residual
7. Final LayerNorm
8. `logits = lm_head(x)` — project to vocabulary size

**Outputs:**

| Output | Shape | Description |
|--------|-------|-------------|
| `logits` | (B, T, vocab_size) | Raw scores for each possible next token at each position |
| `loss` | dict or None | `{'loss_ce': scalar, 'loss_dt': scalar}` during training; None during inference |
| `att` | (n_layer, B, n_head, T, T) | Attention weights from all layers (for interpretability) |

**Loss computation (when targets provided):**

- `loss_ce`: Cross-entropy between predicted token distribution and actual next token. Tokens in `ignore_tokens` list are masked out (do not contribute to gradient).
- `loss_dt`: Exponential log-likelihood for time-to-next-event. `logsumexp(logits)` gives the aggregate event rate; this is compared against the actual time gap `targets_age - age`.
- Total loss used for backprop: `loss_ce + loss_dt`

### C.4 What "forward pass works" means and does NOT mean

**Verified:**
- Given tensors of the correct shape and dtype, the model produces logits of shape (B, T, V) without errors
- Loss computation runs and produces finite scalar values for both loss_ce and loss_dt
- Gradients flow backward through the network (training loop runs, loss changes over iterations)
- Checkpoint save/load works
- 3-column Delphi data is auto-detected and default token_type (all DX) is created

**NOT verified:**
- Whether the model learns anything meaningful (30 iterations on synthetic data is not enough)
- Whether 4-column real data loads correctly (no real 4-col data exists)
- Whether the token type embedding actually improves prediction (requires ablation on real data)
- Whether the vocabulary size, block_size, and model dimensions are appropriate for Korean claims data
- Whether the time-to-event loss behaves correctly with multi-modal token sequences (only tested with diagnosis-only Delphi data)
- Whether the generation function produces valid multi-modal trajectories
- Whether the `token_type_lookup` dict needed for generation works correctly

---

## D. Execution Evidence

### D.1 Batch tensor shapes
```
x  (input tokens):      torch.Size([4, 24]), dtype=torch.int64
a  (input ages):         torch.Size([4, 24]), dtype=torch.float32
y  (target tokens):      torch.Size([4, 24]), dtype=torch.int64
b  (target ages):        torch.Size([4, 24]), dtype=torch.float32
xt (input token types):  None (3-col data → created as all-DX default)
```

### D.2 Forward pass output
```
logits: torch.Size([4, 24, 1270])  — (batch=4, seq_len=24, vocab=1270)
loss_ce: 7.1575
loss_dt: 369.6253
att: torch.Size([2, 4, 2, 24, 24]) — (layers=2, batch=4, heads=2, 24, 24)
```

### D.3 Training log (30 iterations, CPU, Delphi synthetic data)
```
Data format: 3-column (Delphi compat)
Train: 181293 rows, Val: 181293 rows
vocab_size = 1270, n_token_types = 9
Initializing a new FERMAT model from scratch
FERMAT parameters: 0.28M

iter 0: loss 4449.2280, time 269.90ms
iter 5: loss 4917.4717, time 196.57ms
iter 10: loss 4933.0142, time 192.76ms
step 15: train loss 3499.2642, val loss 3388.6655
Saving checkpoint to FERMAT-demo
iter 15: loss 4536.5591, time 495.71ms
iter 20: loss 4591.7896, time 202.75ms
iter 25: loss 4321.2480, time 197.43ms
step 30: train loss 3430.0945, val loss 3758.9873
```

### D.4 Checkpoint
```
File: FERMAT-demo/ckpt.pt (3.3MB)
Contents: model state_dict (34 keys), optimizer state, model_args, iter_num=15, best_val_loss=3388.6655
Includes transformer.wtype.weight: torch.Size([9, 64]) — confirms type embedding is saved
```

### D.5 3-col vs 4-col auto-detection
```
3-col input (9 uint32 values) → detected shape (3, 3), has_types=False
4-col input (12 uint32 values, 4th col < 20) → detected shape (3, 4), has_types=True
```

---

## E. Unresolved Design Issues

| Issue | Options | Impact | Decision needed by |
|-------|---------|--------|--------------------|
| LAB/LIFESTYLE in loss | (a) Include → model predicts future screening results (b) Exclude → context only | Changes what the model learns; affects ablation interpretation | Before first real training run |
| ATC granularity | Level 3 (~300 groups) vs Level 4 (~1,000+) | Vocabulary size, compute cost, pharmacological resolution | Before vocabulary construction |
| HIRA EDI→ATC mapping | Need mapping table from HIRA or construct from drug product codes | Without this, RX tokens cannot be created from HIRA data | Before data preprocessing |
| 3-col/4-col detection heuristic | Current: divisibility + max value check. Fragile. | Could silently misinterpret data | Replace with explicit format flag before real data use |
| block_size for multi-modal data | Delphi uses 48. Multi-modal sequences are 3-5x longer. 96? 128? | Memory usage, context window, training speed | After seeing real sequence length distribution |
| Data linkage method | TTP-based random key → how does merged patient_id work? | Determines whether patient_id is consistent across sources | Confirm with platform operators |

---

## F. Next Implementation Priorities

| Priority | Task | Blocked on | Estimated effort |
|----------|------|-----------|-----------------|
| 1 | **Platform data access application** — finalize which institutions/tables to request, submit by 4.29 deadline | PI decision on institution combination (max 4) | 1-2 days (paperwork) |
| 2 | **Token vocabulary mapping table** — KCD 3-char → DX token IDs, ATC level → RX token IDs, screening items → LAB token IDs with cutpoints, procedure groups → PX token IDs | Data catalog Excel (Y-marked variables) | 3-5 days |
| 3 | **Data preprocessing scripts** — one per source (NHIS, HIRA, Statistics Korea, etc.), each converting raw tables to 4-col uint32 rows, then merge/sort/split | Vocabulary table (#2) + actual data access | 1-2 weeks |
| 4 | **4-col smoke test** — create small synthetic 4-col dataset, verify all code paths that are currently untested (type embedding gradient flow, type-aware ignore, multi-modal generation) | Vocabulary table (#2) | 2-3 days |
| 5 | **Platform environment setup** — specify GPU/PyTorch/shell requirements to platform operators, test code execution in restricted environment | Platform access confirmation | Depends on platform timeline |
