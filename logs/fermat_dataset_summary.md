# FERMAT dataset summary

_Source:_ `data/synthetic_snuh`

| metric | train | val |
|---|---:|---:|
| patients | 9000 | 1000 |
| events   | 541154 | 59713 |
| seq len (median / p95 / max) | 57 / 133 / 234 | 56 / 133 / 180 |
| age_in_days range | 0–34987 | 0–34924 |
| vocab size | 422 | — |
| person-level train/val overlap | 0 | — |

## Events by token_type

| token_type | train | val |
|---|---:|---:|
| DTH | 611 | 59 |
| DX | 160161 | 17568 |
| LAB | 184578 | 20516 |
| PX | 1338 | 119 |
| RX | 185466 | 20451 |
| SEX | 9000 | 1000 |

## Dropped events during preprocessing

| reason | table | count |
|---|---|---:|
| age_negative | condition_occurrence | 0 |
| age_negative | death | 0 |
| age_negative | drug_exposure | 0 |
| age_negative | measurement | 0 |
| age_negative | procedure_occurrence | 0 |
| age_unrealistic | condition_occurrence | 0 |
| age_unrealistic | death | 0 |
| age_unrealistic | drug_exposure | 0 |
| age_unrealistic | measurement | 0 |
| age_unrealistic | procedure_occurrence | 0 |
| birth_unknown_patient | condition_occurrence | 0 |
| birth_unknown_patient | death | 0 |
| birth_unknown_patient | drug_exposure | 0 |
| birth_unknown_patient | measurement | 0 |
| birth_unknown_patient | person | 0 |
| birth_unknown_patient | procedure_occurrence | 0 |
| measurement_cutpoint_unavailable | measurement | 0 |
| measurement_value_missing | measurement | 0 |
| not_in_top_2000 | DTH | 0 |
| not_in_top_2000 | DX | 0 |
| not_in_top_2000 | LAB | 0 |
| not_in_top_2000 | PX | 0 |
| not_in_top_2000 | RX | 0 |
| not_in_top_2000 | SEX | 0 |
| unknown_source_value | condition_occurrence | 0 |
| unknown_source_value | drug_exposure | 0 |
| unknown_source_value | procedure_occurrence | 0 |
