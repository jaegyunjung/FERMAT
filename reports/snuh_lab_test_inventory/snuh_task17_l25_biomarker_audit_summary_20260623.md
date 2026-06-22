# SNUH Task 17 L25 Biomarker Audit Summary

Generated on 2026-06-23 from the SNUH Department of Laboratory Medicine public
test guide and Pod-side Task 17 audit outputs.

## L25 inventory

- Source URL pattern:
  `http://www.snuhlab.org/checkup/check_list.aspx?ins_class_code=L25&searchfield=TOTAL&searchword=`
- Public page total for L25 molecular genetics: 387 current/orderable tests.
- Inventory artifact:
  `snuh_lab_test_inventory_20260623_021501.csv`
- Oncology/hematology biomarker candidate subset:
  `snuh_lab_test_inventory_L25_oncology_candidates_20260623_021501.csv`

## CDM vocabulary mapping

Pod-side terminal check using the L25 inventory showed:

- Inventory item codes: 387
- Mapped to `cdm2024_official.source_to_concept_map`: 346
- Unmapped item codes: 41
- Distinct target concept IDs: 249
- Target vocabularies: EDI 346 mappings, LOINC 30 mappings

For the 29 oncology/hematology candidate codes:

- Mapped codes: 26
- Unmapped codes: 3
- Distinct target concept IDs: 24
- Target vocabulary: EDI only
- Task 15 LAB ETL hits: 0 in both numeric and categorical LAB frequency files

Interpretation: L25 oncology biomarker codes are mostly preserved in the CDM
vocabulary mapping, but they are not represented in the current Task 15 LAB ETL
frequency outputs.

## Event path checks

Focused post-2021 checks for mapped EDI concept IDs for EGFR, BRAF, KRAS, NRAS,
and MSI found:

- `procedure_occurrence`: no post-2021 samples
- `observation`: no post-2021 samples for the mapped EDI concept IDs
- `measurement`: not resolved by direct concept-ID lookup because the table is
  319 GB and has no indexes; `EXPLAIN` showed sequential scans and timed out
  for sample queries
- `note`: 22,055,802 estimated rows, 3.2 GB, no indexes; exact L25 code matches
  in `note_source_value` / `note_source_value2` and title matches for the
  selected test names returned no post-2021 samples

Interpretation: the mapped L25 oncology concept IDs do not currently reveal a
usable post-2021 structured event/result path outside measurement, and
measurement requires a different anchor than direct full-table concept lookup.

## Post-2021 observation biomarker signal

Existing Task 17 output:
`post2021_observation_onco_biomarker_events_v2.csv`

Summary:

- All rows: 328
- All persons: 160
- Value-bearing rows: 231
- Value-bearing persons: 135
- Mention-only rows: 97
- Mention-only persons: 76
- First value-bearing person-token rows: 185
- Same person-date-token duplicates among value-bearing rows: 0

Main value-bearing groups:

- PDL1 TPS: 71 rows, 57 persons
- PDL1 percent: 35 rows, 28 persons
- EGFR mutation: 28 rows, 22 persons
- ALK status: 29 rows, 22 persons
- EGFR status: 19 rows, 15 persons
- BRAF mutation: 8 rows, 5 persons
- BRAF status: 8 rows, 8 persons
- ROS1 status: 7 rows, 6 persons
- MSI status: 6 rows, 4 persons
- KRAS mutation: 6 rows, 4 persons
- HER2 status: 5 rows, 5 persons

Policy decision:

- Use value-bearing observation biomarker events as the post-2021 weak-signal
  candidate artifact.
- Use all value-bearing events for sequence-style analysis when needed.
- Use first-by-person-token for coverage reporting.
- Exclude mention-only rows from training/token integration by default; keep
  them for QA.
- Exclude `PDL1_MENTION` when value-bearing PDL1 exists for the same person or
  same date.

## Overall unique person coverage

Pod-side union check across existing Task 17 artifacts showed:

- `ngs_fermat_events.csv`: 57 persons, 2019-2020
- `molecular_event_tokens_final.csv`: 390 persons, 2004-2020
- `general_molecular_coarse_tokens_fixed.csv`: 520 persons, 2004-2020
- Post-2021 observation value-bearing biomarker events: 135 persons, 2021-2024

Unique person unions:

- NGS + molecular final: 430
- NGS + molecular final + post-2021 observation value-bearing: 564
- Coarse + post-2021 observation value-bearing: 655
- All four artifacts: 704

Interpretation: current discovered coverage is approximately 500-700 persons,
depending on strict versus coarse inclusion. This is too small to treat as a
high-coverage full-cohort pretraining feature, but meaningful enough to preserve
as a Task 17 audit artifact and as a possible downstream oncology annotation or
weak-signal token source.

## Next direction

The next unresolved task is to find a higher-coverage authoritative report/result
path, if one exists. The current evidence supports keeping these artifacts while
continuing to search for the canonical molecular pathology or genomics report
route rather than treating the 135-person post-2021 observation signal as the
final genomic-token integration path.
