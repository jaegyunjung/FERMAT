# PR #2: 실제 재현과 LoRA 실행 준비 — 2026-10-22

현재 상태는 **입력 준비 및 코드 검증 완료, 실제 재현 gate와 LoRA 학습 미실행**이다.
아래 저장 결과 수치는 기존 v1 산출물을 확인한 값이며 새 실험 결과가 아니다.
프로토콜 버전은 `2026-10-22.4`. PR은 draft로 유지한다.

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

## 2. 이번에 실제 준비한 것

환자 자료는 모두 Git 밖에 있다. WSL 기준 준비 폴더:
`/home/fermat/fermat-data/restricted/lora_baseline_20261022/`

| 산출물 | 현재 확인한 내용 |
|---|---|
| `record_v1/` | train 18,474명, validation 1,867명, test 1,862명; vocabulary 17,460개; prefixes/정답/원본 순서 보존 |
| `record_v1/baseline_rollout50.csv` | 시험 1,862명 모두 일치; Recall@10 평균 70.9954203%; 새 코드와 같은 % 단위로 변환 |
| `record_v1/*/frozen_embeddings.npy` | 기존 임베딩의 동일 행을 복사한 대조용 파일. 학습 스크립트는 아직 이 cache를 소비하지 않음 |
| `risk_baselines/{ckd,fatty_liver,diabetes}_v1_probabilities.csv` | 고정 C=10 표현 예측기의 기존 시험 10,000명 확률; 새 모델 결과 아님 |
| 각 준비 폴더 `preparation.json` | 원본 해시, 환자 수, 작업 상태. 환자별 행은 포함하지 않음 |
| `release_cpu_preflight.json` | 릴리스 artifact 6개 해시 일치, 실제 가중치 로딩 및 6명 CPU forward 정상 |

기록 export는 약 286 MB(대조용 임베딩 포함)다. `PrefixSplit` 검증을 통과했으며,
전체 입력 content hash는
`3bba0fedeec726ede3cbab579eaac0d67c8cd4c438abdc7e41ecd596147e4835`다.
이번 warm-cache 전체 content hash 읽기는 약 0.09초였다. 더 큰 질병 코호트나
cold cache의 시간 추정에 그대로 적용하지 않는다. 매 학습 실행의 전체 입력 해시
읽기 시간도 `gate_identity.dataset_hash_seconds`에 기록한다.

CPU FP32 임베딩 6개와 저장 CUDA BF16 임베딩의 최소 cosine은 0.999959,
최대 절대 차이는 0.02714 미만이었다. 이는 진단용이며 BF16 재현 통과 증거가 아니다.
동일 CUDA BF16 설정과 batch 구성을 맞춘 비교가 남았다.

검증한 기존 표현 예측기 결과:

| 질환 | 학습 적격 / 관측 | 검증 적격 / 관측 | 시험 적격 / 관측 | AUROC | Brier |
|---|---:|---:|---:|---:|---:|
| CKD | 256,536 / 148,048 | 54,858 / 31,925 | 9,741 / 5,665 | 0.8705996312 | 0.0204001220 |
| 지방간 | 253,697 / 145,782 | 54,319 / 31,461 | 9,632 / 5,578 | 0.7519826423 | 0.0272524609 |
| 당뇨병 | 235,670 / 133,947 | 50,477 / 28,974 | 8,858 / 5,082 | 0.7840186600 | 0.0451849993 |

CPU 자동 테스트 37개가 torch `2.12.1+cu132`에서 통과했다.
다른 환경의 torch 2.14.1 검증 결과로 표현하지 않는다.

## 3. 아직 필요한 것과 실행 순서

1. **동일 CUDA BF16 임베딩 확인.** 기존 batch 32 구성과 입력 순서를 맞춰
   저장 임베딩과 대조한다. 차이가 나면 release 모델, 마지막 hidden state,
   attention mask, dtype부터 확인한다. 위 CPU 비교를 gate로 사용하지 않는다.
2. **기록 v1 재학습 및 gate.** 아래 명령의 seed 20260909, batch 256,
   v1 설정을 사용한다. `--batch-size 32`는 추출/점수 계산용이며, head 학습 batch는
   v1 recipe에서 256으로 고정된다. 먼저 `--recipe diagnostic_v1 --embeddings-cache`
   실행으로 저장 임베딩 위의 동일 레시피를 진단한다. 일반 experimental은 epoch
   순서까지 달라지므로 이 진단에 사용하지 않는다. diagnostic_v1은 gate를 발급하지 못한다.
   stable ranking tie 처리는 원본과 다른 점이므로
   저장 예측과 환자별 차이를 별도로 정리한다. gate 실패 시 LoRA를 시작하지 않는다.
3. **질병 입력 export.** `task30_d3_final_analysis_20260901_v1` 아래 원본
   `preopen/d3_preindex_cohort_train_val_testselected.parquet`,
   `preopen/d3_train_validation_outcomes.parquet`,
   `embeddings/fermat_embeddings_20180101_5y_last.parquet`가 로컬에 있다.
   시험 정답은 `final_evaluation_v2_raw_dx/restricted_patient_level/selected_test_outcomes_raw_dx_once.parquet`다.
   원본 `observed_status`/`ipcw` 정의를 따라 eligible, outcome, weight를 만들고,
   임상변수 4개와 원본 마지막 512행 prefix를 연결한다. 가중치/임상변수 float64를 보존한다.
   `eligible = weight > 0`으로 만들면 안 된다. 위 표의 수와 저장 C10 전처리
   통계·확률을 대조한다. **현재 `risk_baselines`는 학습 dataset이 아니다.**
4. **질병별 전체 집단 C10 gate.** AUROC와 Brier를 모두 검사한다. 시험 결과에
   맞추어 C, cohort, 허용오차를 조절하지 않는다.
5. **비교할 학습 환자 목록 고정.** 현재 기록 입력은 18,474명이다.
   기존 `weekend_head_scaling_20260912_v1/n100000`의 contract/선택 결과/예측 파일은
   있으나, 10만 명 prefix export 및 기존 명단 대조는 아직 하지 않았다.
   18,474명 입력에 `--train-subset 100000`을 주거나 새로 임의 추출하지 않는다.
   **현재 .4 프로토콜은 확장된 학습 pool에 기존 gate 재사용을 허용하지 않는다.**
   validation/test가 같아도 전체 content hash가 바뀐다. 이번 실행 범위는 18,474명이며,
   100k 비교는 새 학습 pool과 역사적 reference cohort를 각각 검증하는 gate 설계 및
   회귀 테스트를 별도 프로토콜 개정으로 확정할 때까지 보류한다. experimental head를
   v1으로 이름만 바꾸거나 hash 검사를 우회하지 않는다.
   질병의 10만 명 실험도 질환별 관측 환자 명단을 고정하고 두 arm이 공유해야 한다.
6. **GPU 소량 실행 후 본 실험.** 일치하는 gate가 생긴 뒤 한 seed의 작은 실행으로
   메모리와 step 시간을 측정한다. 이후 사전 고정한 arm/seed 1, 2, 3을 실행한다.
   원래 head/rollout 결과는 덮어쓰지 않는다. 본 실험 전 WSL GPU probe,
   다른 작업·Windows 재시작 예약·장시간 실행 guard와 모니터링을 확인한다.

이번에는 GPU 학습을 시작하지 않았다. 최초 `nvidia-smi`에서 RTX 5090 32 GB,
사용 약 2.2 GB, 실행 중 학습 Python 프로세스 없음이 관찰됐지만 이는 순간 상태다.
장시간 실행 적합성이나 소요 시간의 실측을 뜻하지 않는다.

## 4. 재실행 명령 (WSL bash)

아래 경로는 현재 PC의 로컬 배치다. 다른 호스트에서는 위치를 바꾼다.
export가 이미 있으면 스크립트가 실패한다. 재검사 때문에 기존 결과를 지우지 않는다.

```bash
PY=/home/fermat/venvs/fermat/bin/python
CODE=/mnt/c/Users/HP/.codex/worktrees/lora-reproduction-refactor/FERMAT/experiments/lora_baseline
BASE=/home/fermat/fermat-data/restricted/task30_local_v1
PREP=/home/fermat/fermat-data/restricted/lora_baseline_20261022
RELEASE=/mnt/c/Users/HP/Documents/FERMAT/exports/FERMAT_SNUH_20261021_compared/FERMAT_SNUH_20261021

# 이미 완료한 export. 새로 만들 때만 비어 있는 별도 --out을 사용한다.
$PY "$CODE/prepare_record_v1.py" \
  --pilot "$BASE/token_time_application_pilot_20260908_v1" \
  --record "$BASE/next_record_bundle_20260908_v1" --out "$PREP/record_v1"
$PY "$CODE/prepare_risk_baselines.py" \
  --fixed-dir "$BASE/task30_d3_final_analysis_20260901_v1/fixed_c10_20260909" \
  --publication-evidence /mnt/c/Users/HP/Documents/FERMAT/reports/genai4health_paper_20260908/build/fixed_c10_analysis.json \
  --out "$PREP/risk_baselines"

# 아래 학습 명령은 CUDA BF16 대조 및 GPU 실행 전 확인을 마친 뒤 실행한다.
$PY "$CODE/train_record.py" --release "$RELEASE" --data "$PREP/record_v1" \
  --out "$PREP/runs/record_v1_seed20260909" --arm head_only --recipe v1 \
  --seed 20260909 --batch-size 32 --device cuda --bf16
$PY "$CODE/check_gate.py" --summary "$PREP/runs/record_v1_seed20260909/summary.json" \
  --out "$PREP/runs/record_v1_gate.json"
```

완료 기준은 새로운 학습의 `summary.json`과 같은 입력/모델/정밀도를 식별하는
통과 `gate.json`이다. 기존 원고의 성능표나 이번 준비 보고서로 대체하지 않는다.
