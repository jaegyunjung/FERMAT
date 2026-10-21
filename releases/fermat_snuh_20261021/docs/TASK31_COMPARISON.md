# Task 31 matched checkpoint comparison — 2026-10-21

**Retain the original 93,000-update FERMAT with corrected generation as the default
collaborator model.** Task 31's completed 100,000-update checkpoint is a useful
comparison model, with a specific advantage in direct next-record item selection.
Neither checkpoint was retrained or calibrated for this comparison.

## Test results

| Measure | FERMAT | Task 31 | Task 31 minus FERMAT, paired 95% interval |
|---|---:|---:|---:|
| Joint observed event/day NLL, lower better | 4.5678 | 17.1734 | +12.6056 [9.1165, 17.0665] |
| Same-day probability Brier, lower better | 0.1513 | 0.2894 | +0.1380 [0.1277, 0.1487] |
| Next-token top-1 accuracy | 47.88% | 40.23% | -7.65 pp [-8.55, -6.72] |
| Direct next-record item Recall@10 | 69.89% | 73.20% | +3.31 pp [2.48, 4.14] |
| Generated next-record item Recall@10 | 63.75% | 60.59% | -3.17 pp [-6.85, 0.63] |

Observed-transition metrics use 1,861 test patients and 13,592 clinical targets.
Direct selection uses 1,862 patients with observed next-record labels. Generation
uses the same fixed subset of 256 patients for both models, ten paths each. Its
test interval includes zero; do not claim a conclusive generation advantage.
Direct selection ranks the foundation model's next-token probabilities without
sampling paths or fitting an embedding head.

## Selection and matching

The protocol was frozen before full evaluation. On validation data (1,866 patients
with observed transitions), joint NLL was 4.6289 versus 16.6381. Generation Recall@10
was 61.20% versus 57.16% on 256 validation patients, difference -4.04 pp
[-7.13, -0.94]. The original model was selected before test evaluation started.
Selection required Task 31 to improve joint NLL with a paired interval excluding
zero, and not worsen marginal event NLL or cap-aware generation recall. These
were operational selection criteria, not formal equivalence tests.

Both arms reused identical observed patient histories, up to 512 events, the
same clinical output vocabulary, temperature 1, no top-k restriction and no
repetition penalty. The two checkpoints share the training data and core model
configuration. Task 31 removes the separate same-day head and fits a single
zero-inclusive geometric waiting-time distribution. Each arm's time inference
matches its training implementation. The common day likelihood is the integer-day
bin probability of each arm's actual sampler; different training loss totals
were not compared directly.

Observed targets are the first up to eight clinical events among the first 64
post-index source events, before death and within five years. Patient means
receive equal weight. No absent follow-up was converted to a negative outcome.

Generation ended when the first evaluated clinical-day bundle was complete, at
death, at the horizon, or at 256 emitted events. **No path reached that cap** in
either split or model. No failed or capped paths were dropped. The predeclared
fallback would bound the entire affected patient's recall between zero and one.

Generation retained the existing `floor(raw_elapsed_days + 1)` date convention.
Consequently, raw day-zero generated events contribute to the first evaluation
day. This recall measures content under that convention, not accuracy of the
actual next visit date. Observed-transition day likelihood does not use this shift.

## Interpretation and verification

Task 31 underestimates same-day occurrence in these selected observed transitions:
67.28% are actually same-day, versus average predicted probabilities of 77.69%
for FERMAT and 40.09% for Task 31 (event-weighted descriptive values). The mean
joint-NLL gap is amplified by long-gap outliers, but 84.26% of test patients also
have lower joint NLL under FERMAT. Recomputing the five largest validation day
losses in CPU FP32 preserved the large losses; relative numerical differences
were at most 4.28% from GPU BF16.

The comparison verifies common inputs for all 3,999 candidate patients, unchanged
checkpoint hashes, raw-source reconstruction of all 512 generation label sets,
and independent recomputation of saved bundle scores and patient-mean joint scores.
Aggregate evidence is in `evidence/task31/`; patient-level data are not published.

These are reused internal validation/test cohorts, not external or prospectively
untouched confirmation cohorts. Intervals use 5,000 paired patient resamples with
fixed checkpoints and paths. No Task 31 five-year disease-risk evaluation or new
embedding-head training was performed. The existing collaborator archive retains
its original weights, inference implementation and optional matching embedding head.
