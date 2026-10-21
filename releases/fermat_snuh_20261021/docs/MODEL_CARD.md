# Model and checkpoint selection

## Selected foundation model

FERMAT SNUH is a causal clinical-event transformer with 10 layers, 10 attention
heads, width 640, and a context of 512 events. It has **65,221,122 unique
parameters**, including tied input/output embeddings; summing state-dict tensors
double-counts tied weights. Vocabulary size is 24,435 including runtime padding.
The backbone combines token, event-type, and continuous age embeddings. A binary
head predicts same-day occurrence and a separate head predicts positive waiting time.

Development used a single hospital's SNUH OMOP records, split by patient:

| Split | Patients | Events |
|---|---:|---:|
| Train | 2,642,237 | 865,804,608 |
| Validation | 565,534 | 185,728,141 |
| Test | 565,785 | 186,797,128 |
| Total | 3,773,556 | 1,238,329,877 |

These counts describe the final training dataset including the small genomics
context extension. They are not evidence of multi-institution or Samsung Life validation.
This release does not claim to have trained on nationwide linked insurance records.

Stage 1 trained event cross-entropy for 100,000 updates. Its validation-best
checkpoint at 76,000 updates initialized stage 2. Stage 2 trained another 100,000
updates with `CE + 0.3 * (same_day_BCE + positive_wait_NLL)`; the selected checkpoint
is at **93,000 updates**, validation objective **4.1976882089725684**. At that
evaluation, CE was 2.624752983646105, waiting loss 5.043902980753805, and same-day
loss 0.19921443700107358. The final update was not the best validation checkpoint.
See `config/training_recipe.json` and `config/provenance.json` for provenance.

The subsequent Task 31 ablation completed 100,000 updates and achieved CE
2.6134053610592685 and total objective 3.0856635315887377, but removed the separate
same-day head and changed the waiting-time objective/target positions. Its total
loss is not comparable to 4.1976882. Its slightly better event CE alone does not
establish better joint event/time generation or downstream performance.

The matched frozen-checkpoint comparison completed on 2026-10-21 retained the
93,000-update FERMAT as the default. On 1,861 evaluable test patients, patient-mean
joint event/day NLL was 4.5678 for FERMAT versus 17.1734 for Task 31 (lower is
better), and next-token top-1 accuracy was 47.88% versus 40.23%. Task 31 did better
at directly selecting ten next-record items without generating paths: 73.20%
versus 69.89% Recall@10 on 1,862 patients. This is a task-specific advantage, not
an embedding-head evaluation. See [the full comparison](TASK31_COMPARISON.md)
for validation selection, uncertainty, generation results and date conventions.

## What changed in the corrected release

The 93,000-step weights were **not retrained** for the corrected generation run.
The correction is in inference: it evaluates the two possible next-date cases
(same day / different day), combines their event distributions using the learned
same-day probability, and samples the event and waiting time from the same case.
Historical dates come only from the observed prefix. No future test date is supplied.
The exact adapter version is `observed-history-two-date-branches-v1`.

Generation uses temperature 1, the full allowed vocabulary (DX/RX/PX/death), no
top-k truncation, and no repetition penalty or fitted probability multiplier.
Positive gaps are clipped to 1–29,200 days; same-day gaps are zero. These are
explicit inherited inference settings, not post-hoc probability calibration.
Death terminates a path. A token cap means the path is incomplete; the example
prints `stopped_at_token_cap`. Do not interpret an incomplete path as disease-free.

## Optional next-record-day prediction head

The selected head is a 640-to-17,460 linear layer trained using 100,000 patients'
frozen embeddings. Selection compared linear/MLP models and three seeds using
validation Recall@10, with validation BCE as tie-breaker. Test data did not select
the checkpoint. It uses the last observed event's embedding, standardized with
the training mean and standard deviation. Those two arrays are exported without
the source file's training-patient indices.

On the historical 1,862-patient next-record-day evaluation, the head obtained
**Recall@10 63.42%**, versus **55.68%** for the original paper head and **71.00%**
for the saved historical 50-path generation result. The rerun at 18,474 training
patients was 55.85%, with eligible-training-only standardization; it is not the
original paper point. The difference between historical generation and the
100,000-patient head was 7.58 percentage points (paired bootstrap 95% interval
6.34–8.88). This is an exploratory learning curve, not new corrected-generation
validation. It does not establish superiority for every kind of prediction.

The published experiment extracted embeddings under CUDA BF16 autocast before
FP32 head training. The CPU example uses FP32 for portability; it checks executable
consistency rather than reproducing the cohort metric. For matching the historical
embedding precision, use `with torch.autocast('cuda', dtype=torch.bfloat16):` around
`embedding(...)`, then pass `embedding.float()` to the head outside that context.
Historical head ranking used a patient-seeded tie break; the example uses stable
class order. Existing cohort predictions, not this fabricated example, support the
reported metric. The head probabilities are multilabel scores, not calibrated
long-term disease risks.

## Transfer and further training

The package supports loading, embedding extraction, corrected generation, and
starting additional training. It is not a restart checkpoint containing optimizer,
random-generator, or data-loader state. The loss recipe is shown in a runnable
one-step example; new-data validation determines further-training choices.
Registry keys use OMOP standard concept IDs, not unconverted ICD/KCD or insurer
codes. The receiver must audit concept/unit coverage and evaluate on their own
data. The tiny genomics extension is context only; this release establishes no
genomics-specific or external clinical benefit.
