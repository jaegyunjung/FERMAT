# Task 30: 토큰 변경 전후 위험곡선 첫 실험

## 1. 로컬에 생성한 파일 링크

- 실행 코드: `scripts/run_snuh_task30_counterfactual_risk_curves.py`
- 첫 실험 설정: `config/snuh_task30_diabetes_hba1c_q10_counterfactual.csv`
- Pod bundle: `dist/task30/snuh_task30_counterfactual_risk_curves_108fa38_dirty_9b3b14616caa.zip`
- bundle SHA256: `3406078b0ecc2642d1dca176a69aeec1f4d688a8cdae801082bdd0ef2388cda3`

Pod에는 Python 파일과 CSV를 따로 옮기지 않는다. 위 zip 하나만 다음 위치로 옮긴다.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_counterfactual_risk_curves_108fa38_dirty_9b3b14616caa.zip
```

## 2. Pod Terminal에 입력할 코드

```bash
set -euo pipefail

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_counterfactual_risk_curves_108fa38_dirty_9b3b14616caa
OUT="$TASK_DIR/outputs/diabetes_hba1c_q10_pilot500_20260714_v2"
LOG="$TASK_DIR/logs/diabetes_hba1c_q10_pilot500_20260714_v2.log"

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/configs "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips

test -f "$TASK_DIR/zips/$BUNDLE.zip"
cd "$TASK_DIR/code"
unzip -o "$TASK_DIR/zips/$BUNDLE.zip" -d "$BUNDLE-code"
cd "$BUNDLE-code"

python3 -m py_compile scripts/run_snuh_task30_counterfactual_risk_curves.py

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/run_snuh_task30_counterfactual_risk_curves.py \
  --phenotype diabetes \
  --interventions-file config/snuh_task30_diabetes_hba1c_q10_counterfactual.csv \
  --output-dir "$OUT" \
  --max-patients-per-intervention 500 \
  --batch-size 64 \
  --max-attention-cells 16777216 \
  --write-every 500 \
  2>&1 | tee "$LOG"
```

## 3. 그 코드의 목적

Cox를 날짜별로 다시 학습하지 않는다. 당뇨에 대한 embedding-only Cox를 한 번 학습한 뒤, 동일 환자의 index 이전 기록에서 HbA1c 최고 10분위 토큰 `LAB:3004410:8554:Q10`만 변경한다.

비교는 다섯 개다.

1. 토큰이 없는 동일 환자군에 index 30일 전 추가
2. 토큰이 없는 동일 환자군에 index 365일 전 추가
3. 토큰이 있는 환자에서 마지막 토큰 삭제
4. 토큰이 있는 동일 환자군에서 마지막 토큰을 index 30일 전으로 이동
5. 토큰이 있는 동일 환자군에서 마지막 토큰을 index 365일 전으로 이동

각 환자에서 변경 전후 FERMAT 임베딩과 Cox 위험점수를 계산한다. Cox의 기본위험을 이용해 index 후 1일, 7일, 이후 30일 간격, 1년·3년·5년 위험을 계산한다.

이 분석은 “기록 토큰을 바꿨을 때 모델 예측이 어떻게 움직이는가”를 본다. HbA1c 토큰 추가를 실제 치료나 인과효과로 해석하지 않는다.

환자별 결과는 500명 단위 Parquet로 먼저 저장한다. 뒤의 요약 코드가 실패해도 완료된 환자 결과는 `raw/` 아래에 남는다.

## 4. 실행 후 다시 전달할 outcome

정상 종료 후 아래 명령을 그대로 실행하고, `##`로 시작하는 전체 출력만 전달한다.

```bash
OUT=/home/khdp-user/workspace/fermat-data/task30/outputs/diabetes_hba1c_q10_pilot500_20260714_v2

echo '## ORIGINAL_EMBEDDING_CHECK'
cat "$OUT/original_embedding_check.json"

echo '## COX_TEST_METRICS'
cat "$OUT/cox_test_metrics.json"

echo '## INTERVENTION_PATIENT_STATUS'
cat "$OUT/intervention_patient_status.csv"

echo '## COUNTERFACTUAL_PAIR_SUMMARY'
cat "$OUT/counterfactual_pair_summary.csv"

echo '## COUNTERFACTUAL_CURVE_LANDMARKS'
awk -F, 'NR==1 || $3==365 || $3==1095 || $3==1826' \
  "$OUT/counterfactual_curve_summary.csv"

echo '## OUTPUT_FILES'
find "$OUT" -maxdepth 3 -type f -printf '%p\t%s bytes\n' | sort
```

오류로 종료되면 새로 실행하지 말고 아래 출력만 전달한다.

```bash
tail -n 120 /home/khdp-user/workspace/fermat-data/task30/logs/diabetes_hba1c_q10_pilot500_20260714_v2.log
```

정상 종료 시에는 환자별 raw Parquet나 PNG 파일 자체를 다시 옮길 필요가 없다. 위 요약을 먼저 보고 30일과 365일 설정의 포함 환자 수가 같은지, 위험 방향이 일관적인지 확인한 뒤 전체 시험 환자 실행 여부를 결정한다.
