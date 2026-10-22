# PR #2 실행 준비 및 재현 상태 — 2026-10-22

**기록 과제 v1 gate와 A/C/LoRA 9개 본 실험 완료. 질병 gate·질병 LoRA 및 100k 확대는 미실행이다.**
프로토콜은 `2026-10-22.5`이며 PR은 draft다. 상세 결과, 실패 진단과 수정 이력,
v1 결과는 [RUN_RESULTS_KO.md](RUN_RESULTS_KO.md), LoRA 속도·메모리·ETA는
[LORA_BENCHMARK_KO.md](LORA_BENCHMARK_KO.md)에 있다.
체크포인트 저장·재개 및 CUDA 재로딩 검증은 [CHECKPOINT_RESUME_KO.md](CHECKPOINT_RESUME_KO.md)를 따른다.
독립 검토 후 본 실험 시작 승인을 받았다. 실행 조건과 A/C 통제군은
[RECORD_MAIN_PLAN_KO.md](RECORD_MAIN_PLAN_KO.md)에 고정했다. 같은 문서의 최종 결과 절에
19:15:40 완료 기록, 성능·bootstrap과 원고/답변 제안문을 정리했다.

## 1. 논문과 실행 경로 대조

다른 리뷰의 C 지적이 맞았다. 원본 `train_task30_d3_comparators.py`의 grid는
최종 원고를 만든 경로와 다르다. 다음 로컬 근거를 확인했다.

- `reports/genai4health_paper_20260908/FIXED_C10_REVISION_KO.md`: CKD·지방간은
  기존 C=10 모델 재사용, 당뇨병 두 모델은 C=10으로 다시 학습했다고 명시.
- 같은 보고서의 `build/recompute_fixed_c.py`, `build/fixed_c10_analysis.json`:
  당뇨병의 기존 선택은 C=.01. C=10 재학습의 검증 Brier가 저장 후보와 오차 0으로 일치.
- 실제 제한 폴더의 여섯 모델 모두 `fixed_C=10`. 최종 manifest와 환자별 확률
  파일의 SHA-256이 원고 빌드에 기록된 값과 일치한다.

따라서 v1은 **C=10, weight decay=1e-5, 200회 전체 배치 AdamW**로 고정했다.
grid는 `--recipe experimental`에서만 사용한다. 지표가 같아도 grid/다른 C의
실제 자료 gate는 거부한다. 기존 .3 gate를 버전 문자열만 바꾸어 재사용하면 안 된다.
C=10은 기존 결과를 검토한 후 정한 수정 사항이므로 사전 지정했다고 쓰지 않는다.

기록 과제의 저장 `model_selection.json`은 `linear_20260909`를 선택했다.
재현 명령에 seed 20260909를 명시한다. 다른 두 역사적 seed도 코드에서 허용하지만,
이는 원래의 linear/MLP × 3 seed 선택 전체를 다시 수행한 것과 같지 않다.

## 2. 현재 사용해야 할 입력

WSL 제한 폴더 `PREP=/home/fermat/fermat-data/restricted/lora_baseline_20261022`.
환자 자료, 임베딩, 예측, 가중치는 모두 Git 밖에 둔다.

| 폴더 | 용도 |
|---|---|
| `record_v1_original_batches/` | 최종 기록 입력: 학습 18,474명, 검증 1,867명, 시험 1,862명; vocabulary 17,460개 |
| 그 안의 `preprocessing_train/` | 원본 학습 환자 19,992명 전원으로 정규화 통계 계산; 정답 없는 환자 포함 |
| 그 안의 `embedding_pool/` | 원본 23,991명 순서, batch 32를 보존하는 추출 전용 입력; held-out 자료는 정규화 학습에 사용하지 않음 |
| 각 split의 `frozen_embeddings.npy` | 원본 저장 임베딩; diagnostic_v1 입력 또는 fresh v1의 대조 기준 |
| `baseline_rollout50.csv` | 기존 생성 50경로 시험 1,862명, 평균 Recall@10 70.9954203%; 신규 생성 결과 아님 |
| `risk_baselines/` | 기존 C10 표현 모델의 시험 10,000명 확률 CSV; 질병 학습 입력이 아님 |

최종 기록 export는 848,899,736 bytes(약 849 MB)다. 입력 content hash는
`5eb20ab9b04a1619711733dca0db67332cd3d0ecfbcccce233f8f7343b2b983e`.
전체 hash 읽기는 warm cache에서 약 0.22초였다. cold cache나 질병 입력의
시간으로 일반화하지 않는다. 매 실행 `dataset_hash_seconds`를 기록한다.
기존 `record_v1/`, `record_v1_full_preprocessing/`는 초기 진단용으로 보존했으며,
이 경로로 최종 gate를 실행하지 않는다. .3/.4 gate를 버전만 바꾸어 재사용하지 않는다.

기존 고정 C10 결과(새 재현 결과 아님):

| 질환 | 학습 적격 / 관측 | 검증 적격 / 관측 | 시험 적격 / 관측 | AUROC | Brier |
|---|---:|---:|---:|---:|---:|
| CKD | 256,536 / 148,048 | 54,858 / 31,925 | 9,741 / 5,665 | 0.8705996312 | 0.0204001220 |
| 지방간 | 253,697 / 145,782 | 54,319 / 31,461 | 9,632 / 5,578 | 0.7519826423 | 0.0272524609 |
| 당뇨병 | 235,670 / 133,947 | 50,477 / 28,974 | 8,858 / 5,082 | 0.7840186600 | 0.0451849993 |

## 3. 다음 실행 순서와 중단 조건

아래 1번 기록 실험은 완료했다. 현재 작업은 5절의 **질병 전체 코호트 계획**이다.
2~4번 중 기존 자료의 CPU 확률·지표 replay와 정답/가중치 준비는 완료했지만,
prefix export, 신규 CUDA 임베딩, C10 재학습 gate는 아직 남아 있다.

1. **기록 LoRA 소량 실행:** 현재 동일 dataset/model/precision의 .5 gate를 먼저 검증한다.
   소량 실행에서 속도·메모리를 측정한 뒤 전체 epoch/seed 소요 시간을 계산한다.
   본 실험 비교 arm과 seed 1, 2, 3을 고정한다. 128-step 측정은 완료했고, LoRA 20 epochs × 3 seeds 예약 예산은 약 3시간 30~45분이다. 18:02 KST에 A/C/LoRA 순차 큐를 시작했다. RECORD_MAIN_PLAN_KO.md의 시작 기록을 참조한다.
2. **질병 학습 입력 export:** 원본 `task30_d3_final_analysis_20260901_v1/preopen/`
   코호트·정답, `embeddings/fermat_embeddings_20180101_5y_last.parquet`,
   `final_evaluation_v2_raw_dx/restricted_patient_level/selected_test_outcomes_raw_dx_once.parquet`
   를 대조한다. 마지막 512 prefix, 임상변수 4개, eligibility, observed status, ipcw를
   보존한다. `eligible = weight > 0`으로 만들지 않는다.
3. **질병 전처리·확률 검증:** 저장 C10 임상변수 중앙값/평균/표준편차의 float64 완전 일치,
   독립 원본 함수 대비 FP32 특징 완전 일치, 저장 모델 확률 replay 최대 절대 오차
   ≤1e-6을 요구한다. 실패하면 원인을 설명하고 gate를 실행하지 않는다.
4. **질병 C10 gate:** 전체 적격 학습 집단으로 전처리, 관측 집단으로 C10·200회
   전체 배치 학습. AUROC와 Brier를 모두 검사한다. 실제 자료에서 아직 미검증이다.
   반복 추출을 줄이는 질병 공용 cache는 미구현이다. 향후 모델·prefix 내용·정밀도·
   추출 설정으로 키를 만들고 질환별 eligibility/정답과 분리해 검증한다.
5. **100k 확대:** 기존 `weekend_head_scaling_20260912_v1/n100000` 명단부터 검증하고
   새로운 prefix export가 필요하다. validation/test가 같아도 training content hash가
   달라지므로 현재 기록 gate를 재사용할 수 없다. 역사적 reference와 확대 pool의
   호환성을 검사하는 별도 프로토콜 개정·테스트 전까지 100k LoRA는 보류한다.

실행 중 입력 불일치, GPU 한계 초과, 새 WHEA/driver/System 오류 또는 재부팅 대기
상태가 발견되면 중단한다. 기존 결과는 덮어쓰지 않는다. 테스트 결과에 맞춰
허용오차, C, 코호트 또는 seed를 바꾸지 않는다.

## 4. 기록 재실행 명령 (WSL bash)

현재 PC의 경로이며 다른 호스트에서는 바꾼다. OUT은 항상 새로운 빈 경로를 쓴다.
W&B에는 집계 지표와 선택한 비환자 설정만 보낸다. 원자료·예측·모델·코드·콘솔은
업로드하지 않으며, 업로드 실패 시 로컬 JSONL을 보존한다.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_ERROR_REPORTING=false
PY=/home/fermat/venvs/fermat/bin/python
CODE=/mnt/c/Users/HP/.codex/worktrees/lora-reproduction-refactor/FERMAT/experiments/lora_baseline
PREP=/home/fermat/fermat-data/restricted/lora_baseline_20261022
DATA=$PREP/record_v1_original_batches
RELEASE=/mnt/c/Users/HP/Documents/FERMAT/exports/FERMAT_SNUH_20261021_compared/FERMAT_SNUH_20261021
PROFILE=/mnt/c/Users/HP/Documents/FERMAT/reports/ethos_disease_comparison_20260914/crash_20260914/clock_setting.json
OUT=$PREP/runs/record_v1_new_run

# 먼저 저장 임베딩으로 head 레시피만 진단: gate를 발급하지 않는다.
$PY "$CODE/train_record.py" --release "$RELEASE" --data "$DATA" \
  --out "${OUT}_diagnostic" --arm head_only --recipe diagnostic_v1 \
  --embeddings-cache "$DATA" --seed 20260909 --batch-size 32 --device cuda --bf16 \
  --gpu-profile "$PROFILE" --tracking online \
  --wandb-entity jaegyunjung-seoul-national-university-hospital

# 신규 추출의 전체 임베딩 대조가 통과해야 head 학습을 시작한다.
timeout --kill-after=15s 1200s $PY "$CODE/train_record.py" \
  --release "$RELEASE" --data "$DATA" --out "$OUT" --arm head_only --recipe v1 \
  --reference-embeddings "$DATA" --seed 20260909 --batch-size 32 --device cuda --bf16 \
  --gpu-profile "$PROFILE" --tracking online \
  --wandb-entity jaegyunjung-seoul-national-university-hospital
$PY "$CODE/check_gate.py" --summary "$OUT/summary.json" --out "${OUT}_verified_gate.json"
```

`--batch-size 32`는 추출·점수 계산용이다. head 학습 batch 256은 v1 recipe가 고정한다.
현재 부팅 후 검증한 GPU profile이 필요하며, 재부팅 뒤 예전 profile을 그대로 쓰지 않는다.

## 5. 질병 LoRA 실행 계획과 시작한 준비 — 2026-10-22

목표는 **CKD·지방간·당뇨병 각각에서 LoRA가 기존 frozen embedding 예측보다
잘 맞히는지 확인하는 것**이다. 신규 생성이나 DB 조회는 필요하지 않다.

### 비교 기준과 학습 범위

| 질환 | 학습 관측 환자 | 검증 관측 환자 | 시험 관측 환자 / 발생 사례 | 기존 AUROC | 기존 Brier |
|---|---:|---:|---:|---:|---:|
| CKD | 148,048 | 31,925 | 5,665 / 155 | 0.8705996312 | 0.0204001220 |
| 지방간 | 145,782 | 31,461 | 5,578 / 168 | 0.7519826423 | 0.0272524609 |
| 당뇨병 | 133,947 | 28,974 | 5,082 / 280 | 0.7840186600 | 0.0451849993 |

공통 원본 코호트는 train 263,245 / validation 56,338 / test 10,000명이다.
질환별 기존 질병 보유자와 관측 여부 때문에 위 평가 인원이 다르다.
각 arm에서는 같은 환자·정답·가중치를 사용한다. 학습에 쓰는 정규화 통계는
관측 환자만이 아니라 전체 적격 학습 환자에서 계산한다.

이전 100,000명 부분집합 제안은 폐기하고 전체 관측 학습 집단을 사용한다.
기존 논문 결과와 직접 비교하기 위한 변경이며, 질병 LoRA 결과를 보기 전에 정했다.

- **A: 기존 frozen embedding 예측기.** 임상변수 4개 + 임베딩 640개,
  C=10, AdamW lr .03, weight decay 1e-5, 전체 배치 200 updates.
  질환별 1회 재학습하고 저장된 기존 확률과 나란히 보존한다.
- **B: LoRA + 같은 645-parameter logistic head.** 질환별 seeds 1/2/3,
  batch 32, rank 8, alpha 16, attention, adapter dropout .05, base dropout .1,
  adapter lr 2e-4, head lr .001, weight decay .01, warmup 100, clip 1,
  accumulation 1, 최대 3 epochs, patience 1. 검증 weighted BCE가 가장 낮은
  epoch를 선택한다. 현재 질병 프로토콜을 유지한다. 시험 결과로 설정을 바꾸지 않는다.
  학습 파라미터는 adapter 307,200 + head 645 = **307,845개**다.
- **C: 최적화 일정 대조군.** frozen embedding 위 head만 B의 batch/LR/weight decay/
  warmup/clip/epoch/선택 규칙으로 학습한다. seeds 1/2/3을 모두 보고한다.
  현재 risk trainer에는 이 경로가 없어 구현·검증이 필요하다. 주 비교는 B 대 A이며,
  C는 학습 일정의 차이가 결과에 얼마나 기여하는지 확인한다.
  B의 transformer dropout 등까지 모두 같은 실험은 아니다.

### 평가와 판정

AUROC는 **올라갈수록**, Brier는 **내려갈수록** 좋다. 각 질환에서 두 지표를
따로 보고한다. 예를 들어 AUROC만 오르면 “환자 위험도 순위 구분이 좋아졌다”라고
쓴다. Brier도 내려가면 “발생 확률의 제곱오차도 줄었다”라고 쓴다.
Brier 하나만으로 보정(calibration) 자체가 좋아졌다고 단정하지 않는다.

1. 주 비교: seed 1 B − 기존 저장 A. 재학습 A와의 차이도 함께 제시한다.
2. seeds 2/3은 반복 실행의 일관성을 확인한다. 가장 좋은 seed만 선택하지 않는다.
3. 같은 환자를 짝지어 2,000회 bootstrap, 95% percentile CI를 계산한다.
   AUROC 차이 CI가 0보다 높으면 AUROC 개선, Brier 차이 CI가 0보다 낮으면 Brier 개선이다.
   0을 포함하면 차이가 불확실하다고 적는다. 점추정과 CI를 질환별로 모두 공개한다.
4. 위 95% CI는 각 비교별 구간이며 여섯 지표 전체의 동시 95% 보장은 아니다.
   세 질환 전체에 대한 하나의 유의성 주장으로 합치지 않는다.
   기존 관측 가중치와 학습된 모델을 고정한 구간이다. v1의 관측 모델 재추정
   bootstrap과 구분하고 필요하면 저장 확률에 원래 v1 분석을 추가한다.
5. seed 평균±SD와 환자 bootstrap CI는 별개로 표기한다. seed를 환자처럼 합치지 않는다.

### 실행 순서와 통과 조건

| 순서 | 할 일 | 완료 기준 / 실패 시 처리 |
|---|---|---|
| 1 | 원본 자료, 기존 모델, 정답/가중치 CPU 확인 | **완료**, 아래 실제 준비 결과 참조 |
| 2 | 원본 bin에서 마지막 512 prefix export | 기준일 포함(age ≤ index_age), token mapping, 환자 순서, split/정답/eligibility 대조. 원본 hash 확인. 다른 코호트나 누락 환자로 대체 금지 |
| 3 | 공용 임베딩 cache와 CUDA BF16 대조 | 원본은 **batch 64**, shard 4096, split/dense ID 순서. 전체 추출 pool에서 원본 batch 문맥 보존. 기록 batch 32를 복사하지 않음. 모델/내용/precision/추출설정 hash 검증. 동일 cache에서 질환별 적격 명단만 선택 |
| 4 | 질환별 C10 재학습 및 gate | 임상 통계 float64 오차 0, FP32 특징 오차 0, 저장 모델 replay 확률 오차 ≤1e-6. 신규 임베딩 차이는 먼저 진단. AUROC ±.01, Brier ±.002 둘 다 충족. 현재 CPU replay는 이 gate가 아님 |
| 5 | 실행 기능 보완 | risk trainer에 W&B, GPU guard, validation-only, checkpoint/resume, C arm, 시간/메모리 계측, 원시 확률 float64 저장. risk head·전처리·optimizer·best·RNG 복원 및 실제 CUDA 재로딩/연속 대 재개 대조 통과 |
| 6 | 질환별 128-step 소량 측정 | 시험 지표 계산 없이 step 시간, 검증 시간, peak memory 측정. 전체 명단을 유지하고 별도 output 사용. 1~5 통과 후 실행 |
| 7 | 본 실험 | A 3개 → C 9개 → B seed 1 세 질환 → 파일·재로딩 점검 → B seeds 2/3. 중간 시험 성능 때문에 뒤 seed를 취소/변경하지 않음 |
| 8 | 최종 분석 | 환자 정렬/지표 독립 재계산, paired bootstrap, 3 seeds 모두 표, W&B finished 및 체크포인트 확인 |

GPU 실행 전 pending reboot, 현재 profile, 다른 GPU 작업, 여유 공간을 확인한다.
비유한 손실/gradient, 모델·환자 hash 불일치, 재개 불일치, GPU guard 위반이면 중단한다.
개선이 없다는 이유로 다른 설정을 자동 탐색하지 않는다. 3 epochs 끝에도 검증 손실이
계속 감소한다면 학습을 충분히 했는지 추가 실험에서 확인하며 최초 결과와 구분한다.
실행 코드는 별도 고정 checkout에서 사용하고 실행 중 수정하지 않는다.

### 이전 예상 시간: 기록 pilot 환산 (아래 6절 실측 ETA로 대체)

기록 pilot의 step .292037초(batch 32), validation 1,867명/6.774초를 환자 수에
비례해 환산했다. 질병 head가 작고 prefix 분포가 달라 실제 속도는 다를 수 있다.

`epoch = ceil(n_train / 32) × .292037 + n_validation / 1867 × 6.774`.

| 질환 | steps/epoch | 학습+검증 1 epoch | 3 epochs × 1 seed | 3 epochs × 3 seeds |
|---|---:|---:|---:|---:|
| CKD | 4,627 | 24.45분 | 73.35분 | 3.67시간 |
| 지방간 | 4,556 | 24.08분 | 72.23분 | 3.61시간 |
| 당뇨병 | 4,186 | 22.13분 | 66.38분 | 3.32시간 |

세 질환 **1 seed씩 약 3시간 32분**, **3 seeds 전부 약 10시간 36분**이 기본값이다.
cache 준비, gate/head fit, C 대조군, 체크포인트·최종 평가·bootstrap 및 여유를 더해
**본 실행 큐 12~15시간**을 임시 예약 예산으로 잡는다. 개발·실패 원인 수정 시간은
포함하지 않는다. 조기 종료 시 짧아진다. 소량 CUDA 측정 후 ETA를 다시 고정한다.

원본 임베딩 manifest의 329,583명 추출은 당시 686초였지만 현재 GPU profile의
새 실측이 아니므로 그대로 완료 ETA로 쓰지 않는다. 준비/코드 보완은 별도 작업이며
단순 실행 수십 분 외에 개발 시간이 필요하다. 현 단계에서 시작·완료 시각을 약속하지 않는다.

### 이전 단계: CPU 정답/가중치 준비 기록

`prepare_risk_outcomes.py`를 작성하고 실제 로컬 자료로 실행했다.
최종 출력은 `$PREP/risk_outcomes_02/`다. 이전 `_01`은 보존했다.
출력은 prefix 없는 **정답/가중치 준비 자료**이며 `risk_training_dataset_ready=false`다.

- 원본 cohort/outcomes/embedding 파일, 원본 분석 함수, 공개 결과 manifest/점수의 hash 대조.
- train/validation/test 정답·eligibility·임상변수·가중치 및 test 기존 확률을 Git 밖에 저장.
- 세 질환 모두 적격/관측 인원 일치, 임상 중앙값/평균/표준편차 오차 0.
- 저장 전처리 통계를 적용한 실제 test FP32 특징: 원본 함수 대비 오차 0.
- 저장 모델 확률 replay 최대 절대 오차: CKD 1.89e-15, 지방간 1.22e-15, 당뇨병 1.55e-15.
- 기존 AUROC/Brier replay 오차 ≤2.23e-16. **새 CUDA 임베딩/새 C10 학습 결과는 아니다.**
- 기존 manifest에는 가중치 배열이 없어 hash 검증한 원본 함수로 관측 모델을 다시
  계산했다. 동일 원본 test 지표가 재현됐다. train/validation 가중치를 역사적 배열과
  직접 비교한 것은 아니므로 다음 C10 재학습을 건너뛰지 않는다.
- 원본 bin/map 파일 존재·크기 일치 확인. 전체 bin content hash와 prefix export는 다음 단계다.
- 디스크 여유 약 775 GiB, 조회 시 GPU 사용률 0%, 전력 제한 400 W 확인.
  이 조회가 본 실행 직전 guard를 대신하지 않는다.

집계 결과는 `results/risk_preparation_20261022/summary.json`에 저장한다.
W&B는 본 학습과 소량 GPU 실행에서 집계 지표·시간·메모리만 계속 기록한다.
이번 CPU 준비는 로컬 provenance JSON에 기록했으며 W&B 학습 run은 새로 만들지 않았다.
환자 ID·정답·확률·모델·콘솔·코드는 W&B에 올리지 않는다.

## 6. 질병 실행 기능 보완 및 CUDA 검증 완료

2026-10-22 추가 작업. 본 LoRA 실험은 시작하지 않는다.

- `prepare_risk_prefixes.py`: 원본 bin/map의 SHA-256 확인 후 329,583명 prefix export 완료.
  원본 추출 순서·길이·환자 map·분할과 일치. 출력 `risk_v1_01`, 약 7.62 GB, 88.91초.
- `risk_cache.py`: 원본 batch 64 / shard 4096 문맥에서 새 CUDA BF16 임베딩을
  추출하고 모든 원본 벡터와 정확히 대조한다. 성공해야 `cache.json`이 생긴다.
  cache 사용 시 데이터 전체 내용·patient mapping·prefix·모델·정밀도 및 cache 해시를 검증한다.
- `train_risk.py`: 원본 모델 JSON과 통계·FP32 특징·저장 예측 replay를 검사한 뒤 C10 gate 발급.
  `--matched-head`, `--validation-only`, `--benchmark-steps`, `--resume`,
  `--stop-after-epoch`, W&B 집계 기록과 host/GPU guard 추가.
- 원본 pandas의 연속 열 합산 순서를 `fit_risk_scaler`에서 보존하도록 수정했다.
  기존 NumPy 행 선택 후 C-order 합산은 임상 통계에 최대 1.34e-12 오차를 만들었다.
  수정 후 세 질환의 저장 임베딩/임상 median·mean·sd가 모두 정확히 일치했다.
- checkpoint 계약은 질병 학습 소스·cache·질병·설정·전처리·모델·자료·환경을 포함한다.
  저장/재개 구현은 기존 원자적 epoch checkpoint를 사용한다. validation-only checkpoint와
  benchmark state는 시험 예측에 사용할 수 없다. 원시 확률은 float64로 보존한다.
- `audit_risk_resume.py`: 실제 CUDA에서 128명 × 2 epochs 연속 실행과 새 프로세스
  중단/재개를 대조한다. 전체 관측 검증 환자의 확률 및 별도 모델 재로딩을 확인한다.
- `run_risk_preflight.py`: C10 gate 3개 → CKD CUDA 재개 감사 → 질환별 128-step pilot.
  실행 중 소스 변경/공간 부족/실패 시 중단한다. **본 실험 실행 기능은 없다.**
- CPU 테스트 **59개 통과**, torch 2.12.1+cu132, 197.50초. 새 risk 재개/재로딩,
  다른 seed·validation-only 계약 거부, terminal resume, matched-head, pilot 격리,
  cache 입력 변조 거부, 원본 통계 합산 순서를 포함한다.

### 실제 CUDA 결과

실행 커밋 `95713c679f6a11de1d0201f6bcfecdc70b39788b`, RTX 5090,
torch 2.12.1+cu132, BF16, 학습 batch 32. 고정 실행 checkout은
`C:/Users/HP/.codex/worktrees/risk-preflight-runtime/FERMAT`이며 수정하지 않는다.
제한 자료/출력은 `$PREP/risk_v1_01`, `risk_cache_01`, `risk_preflight_01`에 보존한다.

- 329,583명 전체 CUDA 임베딩이 저장 원본과 **정확히 일치**, 최대 오차 0.
  공용 cache 추출 1,104.53초(18분 25초). 이 비용은 이미 끝났으며 질환/seed마다 반복하지 않는다.
- 세 질환의 임상/임베딩 float64 전처리 통계와 FP32 학습 특징이 정확히 일치했다.
  저장 모델 replay 확률 오차 ≤1.45e-15. 새 C10 학습은 아래와 같이 gate를 모두 통과했다.

| 질환 | C10 재학습 AUROC | Brier | 기존 확률 대비 최대 절대 차이 |
|---|---:|---:|---:|
| CKD | 0.8705996312 | 0.0204001221 | 8.27e-8 |
| 지방간 | 0.7519826423 | 0.0272524609 | 5.54e-8 |
| 당뇨병 | 0.7840186600 | 0.0451849993 | 7.64e-8 |

- CKD 128명 × 2 epochs: 연속 실행 대 새 프로세스 재개에서 현재/best 가중치,
  AdamW, 누적 update, patience, RNG, 시간 외 epoch 지표가 모두 정확히 일치했다.
  검증 31,925명 확률 및 독립 재로딩 확률 최대 오차 **0**. 감사 총 933.70초.
  질병 checkpoint는 epoch당 약 5.04 MB다. 시험 출력은 만들지 않았다.
- 질환별 4,096명/128 updates 소량 학습 완료. 기반 가중치 불변, adapter tensor 40개 변경.
  모든 pilot batch와 전체 학습 batch의 실제 패딩 폭이 512임을 확인했다.
  속도 계산은 첫 10 step을 제외한 평균이며 LR warmup 100 step과 별개다.
- W&B cache 1개, gate 3개, 재개 감사 3개, pilot 3개가 서버에서 모두 finished.
  업로드 파일은 config.yaml / wandb-summary.json뿐이다. 환자별 자료/모델 업로드 없음.

### 실측에 근거한 본 실험 ETA

아래는 완료된 소량 측정으로 **추정한** 전체 학습 시간이다. 본 실험 결과가 아니다.

| 질환 | 평균 초/step | 검증 전체 | 최대 할당 GPU 메모리 | 학습+검증 1 epoch | 3 epochs × 1 seed | 3 epochs × 3 seeds |
|---|---:|---:|---:|---:|---:|---:|
| CKD | 0.273 | 103.1초 | 14.47 GiB | 22.76분 | 69.13분 | 3.46시간 |
| 지방간 | 0.270 | 101.3초 | 14.47 GiB | 22.19분 | 67.39분 | 3.37시간 |
| 당뇨병 | 0.264 | 95.1초 | 14.47 GiB | 20.04분 | 60.95분 | 3.05시간 |

세 질환 × 3 seeds × 최대 3 epochs는 **9시간 52분**이다. 초기 준비·시험 점수 산출
추정을 포함한다. 체크포인트/최종 저장, bootstrap, 네트워크 및 속도 변동에 15~25%를
추가하고, C 대조군 9개에 별도 30분(미실측 예산)을 더하면 **11.86~12.84시간**,
일정상 **12~13시간**을 확보한다. 조기 종료 시간은 예측하지 않았다.

본 실험은 시작하지 않았다. 다음 순서는 완료된 A gate를 검증한 후 C 9개 →
LoRA seed 1의 세 질환 → 결과 파일/재로딩 점검 → seeds 2/3이다.
시작 직전에 pending reboot, GPU profile/사용 상태와 디스크를 다시 확인한다.
Windows 업데이트 일시중지 기한은 조회 당시 이미 지났으며 OS 설정은 변경하지 않았다.
이번 dataset hash 1~2.3초는 현재 캐시 상태의 측정이다. cold-cache 시간으로 해석하지 않는다.

본 실험에서 기존 frozen embedding보다 좋아졌는지는 아직 알 수 없다.
이번 pilot은 속도/메모리만 측정했으며 시험 성능과 bootstrap을 계산하지 않았다.
