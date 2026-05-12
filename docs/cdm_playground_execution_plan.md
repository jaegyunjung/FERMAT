# CDM Playground Execution Plan (FERMAT v0.1)

When CDM Playground access is granted, this plan walks through what to
run, in what order, and what to bring back out.

The plan assumes:

- The schema compatibility check in
  [`cdm_playground_schema_checklist.md`](./cdm_playground_schema_checklist.md)
  has been completed.
- The Synthetic SNUH smoke harness has already been validated on the
  local machine, so the only new variables in this run are the data
  and the environment — not the code.

---

## Stage 0. Pre-flight (outside CDM Playground)

Before requesting an air-lock session, prepare a single transferable
bundle:

```
fermat_v0_1_bundle/
├── README_FOR_OPERATOR.md
├── requirements.txt
├── model.py
├── utils.py
├── train.py
├── configurator.py
├── config/
│   ├── train_fermat_demo.py
│   └── train_fermat_synthetic_snuh.py
├── scripts/
│   ├── inspect_synthetic_snuh_schema.py
│   ├── preprocess_synthetic_snuh_to_fermat.py
│   ├── check_fermat_bin.py
│   ├── summarize_fermat_dataset.py
│   └── run_smoke_synthetic_snuh.sh
├── sql/
│   └── extract_cdm_playground_sample.sql
└── docs/
    ├── token_spec_v0.1.md
    ├── cdm_playground_schema_checklist.md
    └── cdm_playground_execution_plan.md
```

`README_FOR_OPERATOR.md` (one page) tells the data operator:

1. The only network call this code makes is `pip install` of the
   packages in `requirements.txt`. If pip is blocked, the operator
   should install offline from a pre-approved mirror.
2. The only file the bundle *writes* outside the working directory is
   under `data/synthetic_snuh/`, `logs/`, and
   `FERMAT-synthetic-snuh-v0_1/`. Nothing else.
3. The bundle *does not* try to upload anywhere. All output is
   reviewed in the air-lock by the operator before export.

This addresses the BOBIC concern voiced in 회의록 1 about "안전한
아웃풋 예상." We tell them up-front what files will appear.

---

## Stage 1. Cohort extraction (inside CDM Playground)

Open the SQL editor for the target CDM and run
`sql/extract_cdm_playground_sample.sql` in segments:

| Step | What | Verify |
|------|------|--------|
| 1.1  | `CREATE TABLE sample_cohort` (10,000 persons) | `SELECT COUNT(*) FROM sample_cohort = 10000` |
| 1.2  | Each `*_sample` table | sanity-count query at end of SQL file |
| 1.3  | Export to `fermat_export.duckdb` per the bottom of the SQL file | DuckDB file size < 2 GB |

If the environment forbids creating new tables inside the CDM schema,
the equivalent is to wrap each `CREATE TABLE` body in a single
`COPY (...) TO 'person.parquet'` etc., then load Parquet into a fresh
DuckDB via `read_parquet`.

---

## Stage 2. Preprocess (inside CDM Playground)

Place `fermat_export.duckdb` at `data/synthetic_snuh_raw.duckdb`
(this is the default the script expects) and run:

```bash
bash scripts/run_smoke_synthetic_snuh.sh
```

Inside CDM Playground, the runner will pick up the DuckDB at the
default path and walk through:

1. `inspect_synthetic_snuh_schema.py` — sanity check
2. `preprocess_synthetic_snuh_to_fermat.py` — produce
   `data/synthetic_snuh/{train,val,vocab}`
3. `check_fermat_bin.py` — invariant check
4. `summarize_fermat_dataset.py` — `logs/fermat_dataset_summary.md`
5. `train.py config/train_fermat_synthetic_snuh.py` — 200-iter smoke
6. `train.py config/train_fermat_demo.py --max_iters=50` — 3-col regression

If any step fails: stop, capture the log, do not try to "fix in place"
inside the analysis environment. Bring the log back out and fix on the
outside.

**Expected runtime**: ~5 minutes for 10,000 persons on CPU. <30 s on a
single GPU.

---

## Stage 3. What to export (out of the air-lock)

The following files are safe to export. They contain **no row-level
patient data** beyond the dataset summary statistics that are routinely
shared at the cohort level.

| file | reason it can leave | check before export |
|------|---------------------|---------------------|
| `logs/synthetic_snuh_schema_summary.txt` | aggregate counts only | open and confirm |
| `logs/fermat_bin_check.log`              | invariants, no values  | open and confirm |
| `logs/fermat_dataset_summary.txt`        | aggregate counts       | open and confirm |
| `logs/fermat_dataset_summary.md`         | aggregate counts       | open and confirm |
| `logs/fermat_synthetic_snuh_smoke_test.log` | training loss only  | grep for any `print(X[…])` |
| `logs/fermat_3col_regression_test.log`   | training loss only     | grep similarly |
| `FERMAT-synthetic-snuh-v0_1/ckpt.pt`     | model weights          | model weights only, no data — operator confirms |
| `data/synthetic_snuh/vocab.csv`          | token vocabulary       | source_value column is the only sensitive field — review |

The following files **must not** be exported:

| file | reason |
|------|--------|
| `data/synthetic_snuh/train.bin`, `val.bin` | row-level event sequences |
| `data/synthetic_snuh/patient_split.csv`    | person_id mapping |
| `data/synthetic_snuh/dropped_events.csv`   | per-row drop reasons (small risk; default exclude) |
| `data/synthetic_snuh_raw.duckdb`           | raw CDM extract |

Operator (정집민 단장님 / 보건의료정보원) reviews and stamps the
"allowed" list before any USB / email transfer, per the air-lock policy
described in 회의록 1.

---

## Stage 4. Post-export reconciliation (outside)

Back on the development machine:

1. Place exported artifacts under `cdm_playground_run_<YYYYMMDD>/`.
2. Diff `vocab.csv` against the self-synthetic vocab to confirm the
   token coverage shift (DX/RX/PX/LAB count by token_type).
3. Read `fermat_dataset_summary.md` into the PI brief (the markdown is
   already formatted for this).
4. Load the exported checkpoint into FERMAT locally with `init_from =
   'resume'` and run a short generation pass on the **self-synthetic**
   data (not the CDM data, which stays inside).
5. If anything looks off (e.g., vocab nearly all collapses into a few
   tokens, or sequence lengths are pathologically short), file an
   issue and re-run Stage 1 with a larger sample.

---

## Stage 5. Contingency

| event | response |
|-------|----------|
| CDM Playground access delayed > Day 5 | Run only Stage 0; deliver bundle to operator for proxy execution |
| `death` table inaccessible | Run with `--vocab_cap 2000` unchanged; DTH counts will be zero, PI brief notes the gap |
| `measurement` table empty | LAB token count will be zero; pipeline still runs; PI brief flags this as a v0.2 data-acquisition task |
| Training step crashes | Bring `logs/*.log` out; reproduce on self-synthetic outside; do not patch in place |
| Export blocked at operator review | Treat the run as inconclusive; record what was learned in `docs/changelog_v0.1.md` and propose a smaller-scope re-run |

---

## Reporting

After Stages 1–4 complete, append a short subsection to
`reports/FERMAT_v0.1_PI_brief.md` titled **"CDM Playground run results"**
covering:

- sample size requested vs. delivered
- per-table row counts (from Stage 1.2 sanity count)
- `fermat_dataset_summary.md` table
- final smoke training loss curve summary (3 numbers: iter 0, mid, last)
- list of exported artifacts (the "allowed" column above)
- list of items deferred to v0.2 because of CDM constraints

This becomes the deliverable for the next status meeting with 이형철 PI
and 정집민 단장님.
