# FERMAT v0.1 Changelog

Records every change made in the v0.1 PoC cycle, per the
`Code patch policy` section of the work order.

---

## Added (new files)

- `docs/token_spec_v0.1.md` — concrete OMOP → token mapping rules.
- `docs/changelog_v0.1.md` — this file.
- `docs/cdm_playground_schema_checklist.md` — pre-flight checklist.
- `docs/cdm_playground_execution_plan.md` — air-lock execution plan.
- `scripts/inspect_synthetic_snuh_schema.py` — DuckDB schema inspector.
- `scripts/preprocess_synthetic_snuh_to_fermat.py` — OMOP → 4-column converter.
- `scripts/generate_self_synthetic_4col.py` — fallback synthetic generator.
- `scripts/check_fermat_bin.py` — 4-column bin invariant checker.
- `scripts/summarize_fermat_dataset.py` — dataset summary in txt + markdown.
- `scripts/run_smoke_synthetic_snuh.sh` — Bash end-to-end runner.
- `scripts/run_smoke_synthetic_snuh.ps1` — PowerShell end-to-end runner.
- `config/train_fermat_synthetic_snuh.py` — tiny model config for smoke test.
- `sql/extract_cdm_playground_sample.sql` — OMOP cohort extraction queries.
- `reports/FERMAT_v0.1_PI_brief.md` — 2–4 page PI memo.

## Changed (existing files)

- `.gitignore`:
  - Added `data/*.duckdb`, `data/*raw*`,
    `data/synthetic_snuh_raw.duckdb`, `data/synthetic_snuh/*.bin`,
    `*.duckdb`, `logs/` so Synthetic SNUH raw data, derived bins, and
    harness logs are not committed.

No changes to `model.py`, `utils.py`, `train.py`, `configurator.py`.

## Compatibility verified

- Existing `config/train_fermat_demo.py` (3-column Delphi smoke test)
  continues to run unchanged. Verified by `logs/fermat_3col_regression_test.log`.
- `utils.py::load_data` auto-detection (3-col vs 4-col uint32) was
  observed to choose 4-col correctly for our `train.bin` / `val.bin`
  outputs (token type ids in the small range 0..8).

## Known limitations carried into v0.2

- Unit harmonization on `measurement.unit_*` is not performed; quantile
  binning is per `(source_value)` only.
- VISIT_START / VISIT_END boundary tokens are not produced.
- LIFESTYLE token coverage depends on availability of an `observation`
  table or screening questionnaire schema. Synthetic SNUH may or may
  not expose this; preprocess produces zero LIFESTYLE tokens by default.
- `source_value` → standard vocabulary mapping (ATC, LOINC, SNOMED) is
  deferred; FERMAT v0.1 treats every distinct `source_value` as an
  opaque token.
- Vocabulary cap of 2,000 tokens; long-tail codes are dropped (counted
  in `dropped_events.csv`).
- LAB is included in the loss in v0.1 to keep type-embedding gradients
  flowing; the context-only LAB ablation is deferred to v0.2.

## Smoke test snapshot

Recorded from `logs/fermat_synthetic_snuh_smoke_test.log` (self-synthetic
fallback path, n_patients=2000):

| metric                     | value          |
|----------------------------|----------------|
| iter 0 loss                | ~4,000         |
| step 100 train / val loss  | ~990 / ~894    |
| step 200 train / val loss  | ~922 / ~883    |
| TypeEmb gradient observed  | yes            |
| 3-col regression status    | runs unchanged |
| bin check ERRORS           | 0              |

The numbers above are for the fallback path. Real Synthetic SNUH
numbers will be appended after the first CDM Playground or
local-DuckDB run.
