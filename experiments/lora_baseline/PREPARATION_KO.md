# PR #2 실행 준비 및 재현 상태 — 2026-10-22

**기록 과제 v1 실제 재현 gate 통과. LoRA 128-step 소량 학습 완료; 질병 gate와 LoRA 본 실험은 미실행이다.**
프로토콜은 `2026-10-22.5`이며 PR은 draft다. 상세 결과, 실패 진단과 수정 이력,
v1 결과는 [RUN_RESULTS_KO.md](RUN_RESULTS_KO.md), LoRA 속도·메모리·ETA는
[LORA_BENCHMARK_KO.md](LORA_BENCHMARK_KO.md)에 있다.
체크포인트 저장·재개 및 CUDA 재로딩 검증은 [CHECKPOINT_RESUME_KO.md](CHECKPOINT_RESUME_KO.md)를 따른다.
독립 검토 후 본 실험 시작 승인을 받았다. 실행 조건과 A/C 통제군은
[RECORD_MAIN_PLAN_KO.md](RECORD_MAIN_PLAN_KO.md)에 고정한다.

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
