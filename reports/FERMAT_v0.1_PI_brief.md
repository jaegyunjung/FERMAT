# FERMAT v0.1 — PI Brief

이형철 교수님께 드리는 v0.1 진행 메모입니다.

---

## 1. 목적

회의록 2에서 말씀하신 "서울대병원 CDM으로 일단 한 번 돌려보자"의
**선행 단계**로, Synthetic SNUH OMOP CDM을 FERMAT의 4-column event
sequence로 변환하고 training loop가 multimodal 입력에서 정상 작동함을
검증하였습니다.

이번 단계의 목표는 모델 성능 입증이 아니라 **파이프라인 검증과
재현 가능성 확보**이며, CDM Playground / 실제 SNUH CDM 적용을 위한
dry run에 해당합니다.

---

## 2. 토큰 스키마 (이형철 교수님 질문에 대한 답)

질문하신 "토큰을 어떻게 정의했냐"에 대한 v0.1 답변입니다. 전체 규칙은
`docs/token_spec_v0.1.md`에 정리했으며, 핵심만 추리면 다음과 같습니다.

- **4-column 입력 포맷**: `patient_id | age_in_days | token_id | token_type`
  (uint32, FERMAT의 기존 `utils.py::load_data`가 자동 인식)
- **9개 token type**: PAD / DX / RX / PX / LAB / LIFESTYLE / DTH / SEX / NO_EVENT
- **token_id는 전체 vocabulary에서 globally unique**. `DX:I10`과
  `RX:I10`은 서로 다른 정수
- **OMOP 매핑**:
  - `condition_source_value` → DX (fallback: `condition_concept_id`)
  - `drug_source_value` → RX (fallback: `drug_concept_id`)
  - `procedure_source_value` → PX (fallback: `procedure_concept_id`)
  - `measurement` → LAB (Section 3 참조)
  - `death.cause_source_value` → DTH
  - `person.gender_source_value` → SEX
- **age_in_days 계산**: `birth_datetime` 우선, 없으면 `year_of_birth`
  + `month_of_birth`(없으면 7월) + `day_of_birth`(없으면 1일).
  음수이거나 150세 초과면 drop
- **Same-day deterministic ordering**: SEX → DX → RX → PX → LAB →
  LIFESTYLE → DTH
- **Loss 제외**: PAD, SEX, NO_EVENT (LAB은 v0.1에서는 loss에 포함시켜
  TypeEmb 학습 신호 확보)
- **Vocabulary cap**: top-frequency 2,000 tokens (드물게 등장하는
  코드는 drop하고 `dropped_events.csv`에 기록)

핵심 디자인 의도는 **코드 체계 호환성**입니다. KCD든 EDI든 ATC든
SNOMED든 일단 `source_value`로 받고, FERMAT 내부에서는 token type별
projection만 학습합니다. v0.2에서 ATC/LOINC 매핑을 도입할 때 모델 수정
없이 vocabulary만 교체하면 됩니다.

---

## 3. MEASUREMENT (LAB) 처리

회의록 2의 "타임스탬프 + 타입 + 밸류" 발언에서 가장 디테일이 필요한
부분입니다. v0.1 규칙:

1. `value_as_concept_id`가 있으면 categorical token으로 사용
   (e.g. `LAB:GLU:C45876384`)
2. `value_as_number`만 있으면 **train split에서 계산한 tertile**
   기준으로 Q1/Q2/Q3 binning (e.g. `LAB:GLU:Q3`)
3. 둘 다 없으면 drop
4. 단위 변환은 v0.1에서 안 함 (`unit_concept_id`만 vocabulary metadata에
   기록). v0.2 task

이 방식은 raw numeric embedding보다 단순하지만, smoke test 단계에서
"같은 LDL 검사도 unit에 따라 값이 다르다" 같은 함정을 피하기 위한
의도적 보수 선택입니다.

---

## 4. 실행 결과 (self-synthetic fallback path)

Synthetic SNUH의 KHDP 다운로드를 기다리는 동안, 모든 token type을
강제로 포함하는 자체 4-column generator로 전체 파이프라인을 검증했습니다.

자세한 통계는 `logs/fermat_dataset_summary.md`에 자동 생성되며, 일부를
옮기면 다음과 같습니다.

| metric                          | train  | val   |
|--------------------------------|--------|-------|
| patients                       | 1,800  | 200   |
| events                         | 156,656| 19,969|
| seq length (median / p95 / max)| 70/219/243 | 104/219/245 |
| age_in_days range              | 0–29,076 | 0–28,498 |
| vocab size                     | 83     | —     |
| person-level train/val overlap | 0      | —     |

Events by token_type (train):

| token_type | count   |
|------------|---------|
| DX         | 8,447   |
| RX         | 8,178   |
| PX         | 2,626   |
| LAB        | 131,811 |
| LIFESTYLE  | 3,633   |
| DTH        | 161     |
| SEX        | 1,800   |

Smoke training (200 iter, n_layer=2 / n_head=2 / n_embd=64, CPU):

| step | train loss | val loss |
|------|------------|----------|
| 0    | ~4,000     | —        |
| 100  | 990        | 894      |
| 200  | 922        | 883      |

- TypeEmb gradient flow 확인됨 (LAB token이 압도적으로 많지만
  DX/RX/PX/SEX/DTH 모두 loss에 기여)
- 4-column path → checkpoint save/load 성공
- 기존 3-column Delphi smoke test (`config/train_fermat_demo.py`)
  여전히 작동 (`logs/fermat_3col_regression_test.log`)
- bin invariant check 0 errors

성능 자체는 의미 없는 합성 데이터이므로 절대값은 보지 마시고, 다만
loss가 finite하게 떨어지고 train/val gap이 작은 것은 데이터 누수와
batch 구성 모두 정상이라는 신호입니다.

---

## 5. CDM Playground 적용 계획

회의록 1에서 약속하신 "일주일 내 서울대병원 CDM 시연"의 후속입니다.
Synthetic SNUH 또는 CDM Playground sample로 위 파이프라인을 그대로
재실행하는 절차를 `docs/cdm_playground_execution_plan.md`에 정리했고,
사전 점검 항목은 `docs/cdm_playground_schema_checklist.md`에 있습니다.

핵심 흐름은 다음과 같습니다.

1. SQL (`sql/extract_cdm_playground_sample.sql`)로 10,000명 random
   cohort 추출 → DuckDB로 export
2. DuckDB를 `data/synthetic_snuh_raw.duckdb`에 두고
   `bash scripts/run_smoke_synthetic_snuh.sh` (또는 PowerShell 버전)
3. 6-step harness가 자동 실행: inspect → preprocess → check → summarize
   → smoke training → 3-col regression
4. **공기-락(air-lock) 정책 준수**:
   - 들어가는 것: 코드 번들과 모델 정의만
   - 나오는 것: 학습 로그, 데이터 요약, vocabulary 파일, model
     checkpoint
   - **나오지 않는 것**: 환자별 event sequence (`*.bin`),
     `patient_split.csv`, raw DuckDB
5. 결과는 본 메모의 "CDM Playground run results" 절에 추가

타이밍에 대한 솔직한 보고:

- **Synthetic SNUH 우선** 이유: KHDP에서 즉시 받을 수 있고, IRB
  단계가 필요 없어서 dry run의 잡음을 최소화합니다.
- **실제 SNUH CDM 적용**: IRB 면제 통과 + CDM Playground 접근권한
  확보 시점에 같은 코드를 재실행하면 됩니다. 코드를 손댈 필요는
  없습니다.

---

## 6. 회의 정치 맥락 대응

회의록 1에서 단장님께서 정리하신 복지부 / 건보 / 심평원 측 반대
논리 세 가지에 대한 v0.1 차원의 응대 카드는 다음과 같습니다.

| 반대 논리                          | v0.1이 보여줄 수 있는 것 |
|------------------------------------|-------------------------|
| (1) 목적 불명확                    | "어떤 OMOP CDM이든 token_spec_v0.1만 따르면 같은 모델을 학습시킬 수 있다"는 구체적 실행 가능성 |
| (2) 쓰레기 데이터 학습 우려        | source_value 우선 + 단위 차이는 quantile로 흡수 + drop 기록 (`dropped_events.csv`) |
| (3) 안전한 outcome 예상 불가       | "들어가는 코드와 나오는 파일이 모두 명시되어 있다"는 air-lock 친화 설계 |

단장님께서 말씀하신 **"송곳처럼 찌르는 설명"**의 근거 자료로 본 메모
+ `token_spec_v0.1.md` + `cdm_playground_execution_plan.md` 세 문서를
함께 전달하시면 됩니다.

---

## 7. 다음 단계 (v0.2 이후)

본 v0.1에서 의도적으로 보류한 항목:

- **외국 모델 비교 / 부적합성 시연** (Delphi-2M, ETHOS 등을 한국
  데이터에 inference) — v0.1 입증 범위 밖
- **Distillation pretraining** (외국 공개 모델로 합성 환자 생성 후
  선행 학습) — 회의록 2에서 언급되었지만 별도 트랙
- **6-step ablation 본 실험** (LAB static vs dynamic 등) — 실데이터
  확보 후
- **ATC / LOINC / SNOMED 완전 매핑** — v0.2 vocabulary 확장
- **VISIT boundary token** (`VISIT_START`, `VISIT_END`) — 효과 검증
  후 도입
- **LAB raw numeric embedding** — 현재 quantile 방식이 부족하다고
  판단되면 도입
- **다기관 데이터 통합** — BOBIC 데이터 결합 일정에 맞춤

각 항목은 `docs/changelog_v0.1.md`의 "Known limitations" 절에도
명시되어 있습니다.

---

## 8. 부록: 본 작업 산출물 위치

```
FERMAT/
├── docs/
│   ├── token_spec_v0.1.md
│   ├── changelog_v0.1.md
│   ├── cdm_playground_schema_checklist.md
│   └── cdm_playground_execution_plan.md
├── scripts/
│   ├── inspect_synthetic_snuh_schema.py
│   ├── preprocess_synthetic_snuh_to_fermat.py
│   ├── generate_self_synthetic_4col.py
│   ├── check_fermat_bin.py
│   ├── summarize_fermat_dataset.py
│   ├── run_smoke_synthetic_snuh.sh
│   └── run_smoke_synthetic_snuh.ps1
├── config/
│   └── train_fermat_synthetic_snuh.py
├── sql/
│   └── extract_cdm_playground_sample.sql
└── reports/
    └── FERMAT_v0.1_PI_brief.md   ← 본 메모
```

자동 생성되는 결과물 (gitignore됨):

```
data/synthetic_snuh/{train.bin, val.bin, vocab.csv,
                     patient_split.csv, dropped_events.csv}
logs/{01_inspect, 02_preprocess, 03_check_bin, 04_summary,
      fermat_synthetic_snuh_smoke_test, fermat_3col_regression_test,
      fermat_bin_check, fermat_dataset_summary{.txt,.md},
      synthetic_snuh_schema_summary.txt}.log
FERMAT-synthetic-snuh-v0_1/ckpt.pt
```

이상입니다. 다음 미팅 전에 Synthetic SNUH DuckDB로 재실행한 결과를
본 메모의 §4와 §5 사이에 "Synthetic SNUH run results" 절로 추가해
드리겠습니다.
