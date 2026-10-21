# Input specification: frozen SNUH release

This replaces the old synthetic-data v0.1 specification **for this release only**.
Do not apply a 2,000-token limit or include LAB targets in its prediction loss.

## IDs and time

`token_registry.csv` contains stored IDs 0–24,433. The model ID is **stored ID + 1**.
Runtime model ID 0 is padding. Thus `SPECIAL:PAD` in the registry is model ID 1,
and `SPECIAL:NO_EVENT` is model ID 2; neither is an observed clinical event.
Age is days since birth, not days since first encounter. Stored events have integer
ages; generated ages may be continuous. Inputs are sorted chronologically. The
model uses the last 512 events, with right padding only when batching.

| Type | ID | Use | Key example / construction |
|---|---:|---|---|
| PAD | 0 | padding | runtime ID 0 |
| DX | 1 | context and target | `DX:<standard concept ID>` |
| RX | 2 | context and target | `RX:<standard concept ID>` |
| PX | 3 | context and target | `PX:<standard concept ID>` |
| LAB | 4 | context only | forms below |
| LIFESTYLE | 5 | reserved architecture type; no tokens in this dataset | none |
| DTH | 6 | target and terminal event | `SPECIAL:DEATH` |
| SEX | 7 | context only at birth | `SEX:M`, `SEX:F`, `SEX:U` |
| NO_EVENT | 8 | excluded | `SPECIAL:NO_EVENT` |
| GENOMICS | 9 | context only, 51 frozen keys | exact registry keys |

SNUH used gender concepts 8507/8532 for M/F, otherwise U. Birth dates use supplied
year, month if 1–12 else July, day if 1–28 else 1. Events before birth or after the
extraction cutoff are excluded. For new datasets supply an explicit observation
cutoff and do not include future records in a prediction prefix.

Same-date order is SEX, DX, RX, PX, LAB, DTH; ties within a type are sorted by stored
token ID. DX/RX/PX rows are **retained**, including repeated same-date occurrences:
the source extraction used UNION ALL, not daily deduplication. Numeric labs use
one daily median per patient/date/test/unit. Categorical labs use distinct
patient/date/test/value rows. Genomics context uses the frozen registry's type 9;
do not manufacture a new token number for an unfamiliar genomic key.

## Labs and unknown concepts

For a numeric test/unit pair with exported cutpoints, use
`LAB:<test concept>:<unit concept>:Q01` through `Q10`. Quantile is
`1 + number of cutpoints strictly less than value`; equality stays in the lower
bin. Keep the training cutpoints when transferring; do not recompute them on the
receiver's test dataset. Harmonize units before binning.

If no cutpoints exist for a pair, use `LAB_TEST:<test concept>` (test occurrence
only). Categorical labs use `LAB_CAT:<test concept>:<value concept>`. A numeric and
a categorical result can both occur. Source extraction removed negative numeric
values for these test concept IDs:
`3001376, 3004249, 3012888, 3020891, 3024171, 3025315, 3027018, 3036277,
40762499, 21492241, 21490852, 44789316`.

Missing registry keys in DX/RX/PX/LAB map to their corresponding `SPECIAL:UNK_*`
token. Unknown sex/genomic/custom domains are rejected rather than assigned a new
embedding. The portable adapter accepts already-harmonized events and performs
lookup, lab binning, sorting, truncation, and coverage reporting. Raw OMOP extraction,
daily lab aggregation, birth-date imputation, and insurer-code mapping remain
upstream responsibilities described above; the adapter does not silently perform them.

## Tensors and generation

`idx`: int64 `[batch, sequence]`, model IDs. `age`: float32 same shape.
`token_type`: int64 same shape. Padding is idx/type 0 with age -10000.
Use observed history only. `next_distribution` accepts right-padded prefixes;
`generate_consistent` requires initially unpadded equal-length prefixes. A batch
may be grouped by prefix length; one patient at a time works directly.

Frozen embeddings use the ordinary causal mask without masking past same-day
events. Training event logits use the next-date-aware mask; corrected inference
marginalizes the unknown final next date. Training targets exclude types
0, 4, 7, 8, 9. Generation explicitly restricts outputs to types 1, 2, 3, 6.

The generated raw timeline and the disease-evaluation timeline are distinct:
existing disease analyses map raw offsets to `floor(offset + 1)` days. The inference
example preserves raw ages. Changing this analysis convention creates a different
evaluation and must be reported.
