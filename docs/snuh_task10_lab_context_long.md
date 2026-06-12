# SNUH Task 10 LAB-context CE-only Extended Run

## Run

- Data: 1% SNUH ETL pilot
- Policy: LAB retained as context and excluded as a target
- Objective: CE-only
- Maximum step: 3,000
- Best checkpoint step: 2,750
- Model size: 1.40M parameters
- Checkpoint selection: validation CE

## Deterministic Validation Results

| Metric | Result |
|---|---:|
| Clinical-only CE | 6.5278 |
| Clinical-only perplexity | 683.88 |
| Clinical-only top-1 | 3.6422% |
| Clinical-only top-5 | 12.6087% |
| Clinical-only top-10 | 18.5070% |
| Train-unigram clinical CE | 7.4270 |
| Train-unigram clinical top-1 | 1.0219% |
| New clinical top-1 | 3.0701% |
| Repeated clinical top-1 | 4.0053% |

Type-specific top-1:

| Token type | Top-1 |
|---|---:|
| DX | 6.6669% |
| RX | 1.3977% |
| PX | 3.4560% |
| DTH | 0.0000% |

The DTH result is not interpretable because the validation set contained only
56 DTH targets. Waiting-time metrics were intentionally disabled for this
CE-only diagnostic.

## Interpretation

The extended run improved clinical-only CE from 7.3392 at the 300-step smoke
gate to 6.5278. Clinical top-1 improved from 3.0467% to 3.6422%, which is about
3.56 times the train-only unigram top-1 baseline.

This result closes the Task 10 training diagnostic. It does not represent the
final SNUH foundation model: the run used a 1% patient pilot and a deliberately
small 1.40M-parameter model. The next gates are measurement staging and model
scaling before full-cohort ETL and pretraining.
