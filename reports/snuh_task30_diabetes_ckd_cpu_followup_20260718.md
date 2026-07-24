# Task 30 당뇨→CKD CPU 후속 점검

## 1. 로컬 파일 링크

Pod에는 아래 zip 하나만 옮겨 주세요.

- `dist/task30/snuh_task30_diabetes_ckd_cpu_followup_108fa38_dirty_1f02b648aff8.zip`

참고 파일:

- `scripts/audit_snuh_task30_diabetes_ckd_cpu_followup.py`
- `config/snuh_task30_diabetes_ckd_cpu_followup.json`

SHA256:

```text
4ca4ac8007428fe3b8bb83fe71eeda29dccb446612dc80279670da378ccebcef
```

zip을 Pod의 다음 위치로 옮겨 주세요.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_diabetes_ckd_cpu_followup_108fa38_dirty_1f02b648aff8.zip
```

## 2. Pod에 파일을 옮긴 후 Terminal에 입력하실 코드

```bash
set -u

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_diabetes_ckd_cpu_followup_108fa38_dirty_1f02b648aff8
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/diabetes_ckd_cpu_followup_$RUN_TAG"
LOG="$TASK_DIR/logs/diabetes_ckd_cpu_followup_$RUN_TAG.log"
CODE="$TASK_DIR/code/$BUNDLE-code"
ZIP="$TASK_DIR/zips/$BUNDLE.zip"
PATHWAY_PATIENTS="$TASK_DIR/outputs/pathway_feasibility_20260717_175059/raw/pathway_patient_level.parquet"
FEATURES=/home/khdp-user/workspace/fermat-data/task19/outputs/baseline_features/baseline_features_20180101.parquet
LABS=/home/khdp-user/workspace/fermat-data/task20/outputs/lab_marker_features/lab_marker_features_wide_20180101.parquet

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips

for REQUIRED in "$ZIP" "$PATHWAY_PATIENTS" "$FEATURES" "$LABS"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "MISSING_REQUIRED_FILE: $REQUIRED"
    exit 1
  fi
done

unzip -oq "$ZIP" -d "$CODE"

python3 -m py_compile \
  "$CODE/scripts/audit_snuh_task30_diabetes_ckd_cpu_followup.py"

python3 \
  "$CODE/scripts/audit_snuh_task30_diabetes_ckd_cpu_followup.py" \
  --self-test

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export ARROW_NUM_THREADS=2

set +e
nice -n 10 python3 \
  "$CODE/scripts/audit_snuh_task30_diabetes_ckd_cpu_followup.py" \
  --config-file "$CODE/config/snuh_task30_diabetes_ckd_cpu_followup.json" \
  --pathway-patient-file "$PATHWAY_PATIENTS" \
  --feature-file "$FEATURES" \
  --lab-file "$LABS" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

echo '## EXIT_STATUS'
echo "$STATUS"
echo '## OUTPUT_DIR'
echo "$OUT"
echo '## LOG_FILE'
echo "$LOG"

if [ "$STATUS" -eq 0 ]; then
  echo '## RETURN_SUMMARY'
  cat "$OUT/return_summary.txt"
  echo '## OUTPUT_FILES'
  find "$OUT" -maxdepth 2 -type f -printf '%p\t%s bytes\n' | sort
else
  echo '## LOG_TAIL'
  tail -n 200 "$LOG"
  echo '## SURVIVING_OUTPUTS'
  find "$OUT" -maxdepth 2 -type f -printf '%p\t%s bytes\n' 2>&1 | sort
fi

read -r -p 'Press Enter to close this shell...' _
```

## 3. 코드의 작업과 의미

입력 자료:

- 앞서 생성한 `diabetes_to_ckd` 환자별 경로 자료
- Task 19 연령·성별·진단/처방/검사 이용량
- Task 20 기준일 이전 eGFR·serum creatinine 요약값

두 가지를 확인합니다.

1. 당뇨 첫 기록부터 기준일까지의 기간과 CKD 발생 관계를 연령·성별·의료이용량으로
   보정한 cause-specific Cox 위험비를 계산합니다.
2. 기준일 전에 eGFR 또는 creatinine 이상 신호가 있던 환자를 제외한 뒤 같은 분석을
   반복합니다.

신장 이상 신호는 다음과 같이 표시합니다.

- pre-index eGFR 최솟값 `<60`
- serum creatinine 최댓값이 남성 `>1.3`, 여성 `>1.1`, 기타/미상 `>1.2 mg/dL`
- 최근 730일 이내 마지막 검사값의 이상 여부도 별도로 표시
- 검사자료가 없는 환자는 `not_measured`로 분리

분석 코호트는 세 개입니다.

- 전체 환자
- 알려진 신장 이상 신호가 있는 환자 제외
- 신장검사를 받았고 이상 신호가 없는 환자만 포함

단일 이상 검사값은 CKD 진단이 아닙니다. 이 기준은 이미 신장기능이 저하된 환자가
결과를 만들고 있는지 확인하기 위한 민감도 분석입니다.

## 4. 실행 후 전달해 주실 내용

정상 종료 시 Terminal의 다음 부분 전체를 저에게 전달해 주세요.

```text
## EXIT_STATUS
...
## OUTPUT_DIR
...
## LOG_FILE
...
## RETURN_SUMMARY
...
## OUTPUT_FILES
...
```

실패하면 다음 부분 전체를 저에게 전달해 주세요.

```text
## EXIT_STATUS
...
## OUTPUT_DIR
...
## LOG_FILE
...
## LOG_TAIL
...
## SURVIVING_OUTPUTS
...
```

`raw/diabetes_ckd_analysis_patients.parquet`는 Pod 밖으로 옮기지 않으셔도 됩니다.
