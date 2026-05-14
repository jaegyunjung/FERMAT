# FERMAT mapping audit

_DuckDB:_ `data/synthetic_snuh_raw.duckdb`  
_Data dir:_ `data/synthetic_snuh`

## Source Row Counts

| source_table | rows |
|---|---:|
| person | 10000 |
| condition_occurrence | 177729 |
| drug_exposure | 205917 |
| procedure_occurrence | 1457 |
| measurement | 205094 |
| death | 670 |

## FERMAT Token Counts

| token_type | train | val | total |
|---|---:|---:|---:|
| SEX | 9000 | 1000 | 10000 |
| DX | 160161 | 17568 | 177729 |
| RX | 185466 | 20451 | 205917 |
| PX | 1338 | 119 | 1457 |
| LAB | 184578 | 20516 | 205094 |
| DTH | 611 | 59 | 670 |

## Expected Mapping

| source_table | expected_token_type | source_rows | mapped_total | dropped_explanation |
|---|---|---:|---:|---|
| person | SEX | 10000 | 10000 | birth_unknown_patient=0 |
| condition_occurrence | DX | 177729 | 177729 | age_negative=0, age_unrealistic=0, birth_unknown_patient=0, unknown_source_value=0 |
| drug_exposure | RX | 205917 | 205917 | age_negative=0, age_unrealistic=0, birth_unknown_patient=0, unknown_source_value=0 |
| procedure_occurrence | PX | 1457 | 1457 | age_negative=0, age_unrealistic=0, birth_unknown_patient=0, unknown_source_value=0 |
| measurement | LAB | 205094 | 205094 | age_negative=0, age_unrealistic=0, birth_unknown_patient=0, measurement_cutpoint_unavailable=0, measurement_value_missing=0 |
| death | DTH | 670 | 670 | age_negative=0, age_unrealistic=0, birth_unknown_patient=0 |

## Split Integrity

- OK: original `person_id` overlap between train and val is 0 (train=9000, val=1000).

## Findings

- OK: patient_split.csv train/val person_id overlap is 0 (train=9000, val=1000)
- OK: measurement rows=205094, LAB tokens=205094
- OK: death rows=670, DTH tokens=670

## Result

PASS: no ERROR-level issues found.
