# Task 30 고혈압·이상지질혈증 순서 → CAD CPU 후속 점검

## 1. 로컬 파일 링크

Pod에는 아래 zip 하나만 옮겨 주세요.

- `dist/task30/snuh_task30_htn_dyslipidemia_cad_cpu_followup_108fa38_dirty_182bfe91f980.zip`

참고 파일:

- `scripts/audit_snuh_task30_htn_dyslipidemia_cad_cpu_followup.py`
- `config/snuh_task30_htn_dyslipidemia_cad_cpu_followup.json`

SHA256:

```text
3608737c27460e03ce805b3fff911c8c4669b80d4f78cb19dd3510b5248e4052
```

Pod의 다음 위치로 옮겨 주세요.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_htn_dyslipidemia_cad_cpu_followup_108fa38_dirty_182bfe91f980.zip
```

## 2. Pod에 파일을 옮긴 후 Terminal에 입력하실 코드

```bash
set -u

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_htn_dyslipidemia_cad_cpu_followup_108fa38_dirty_182bfe91f980
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/htn_dyslipidemia_cad_cpu_followup_$RUN_TAG"
LOG="$TASK_DIR/logs/htn_dyslipidemia_cad_cpu_followup_$RUN_TAG.log"
CODE="$TASK_DIR/code/$BUNDLE-code"
ZIP="$TASK_DIR/zips/$BUNDLE.zip"
PATHWAY_PATIENTS="$TASK_DIR/outputs/pathway_feasibility_20260717_175059/raw/pathway_patient_level.parquet"
FEATURES=/home/khdp-user/workspace/fermat-data/task19/outputs/baseline_features/baseline_features_20180101.parquet

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips

for REQUIRED in "$ZIP" "$PATHWAY_PATIENTS" "$FEATURES"; do
  if [ ! -f "$REQUIRED" ]; then
    echo "MISSING_REQUIRED_FILE: $REQUIRED"
    exit 1
  fi
done

unzip -oq "$ZIP" -d "$CODE"

python3 -m py_compile \
  "$CODE/scripts/audit_snuh_task30_htn_dyslipidemia_cad_cpu_followup.py"

python3 \
  "$CODE/scripts/audit_snuh_task30_htn_dyslipidemia_cad_cpu_followup.py" \
  --self-test

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export ARROW_NUM_THREADS=2

set +e
nice -n 10 python3 \
  "$CODE/scripts/audit_snuh_task30_htn_dyslipidemia_cad_cpu_followup.py" \
  --config-file "$CODE/config/snuh_task30_htn_dyslipidemia_cad_cpu_followup.json" \
  --pathway-patient-file "$PATHWAY_PATIENTS" \
  --feature-file "$FEATURES" \
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

## 3. 코드가 하는 일과 의미

다음 세 기록군을 비교합니다.

- 이상지질혈증이 먼저 기록된 환자
- 고혈압이 먼저 기록된 환자
- 두 질환이 같은 날 처음 기록된 환자

연령·성별·의료이용량과 다음 두 시간 차이를 보정합니다.

- 두 번째 질환 기록부터 기준일까지의 기간
- 고혈압과 이상지질혈증 첫 기록 사이의 간격

두 비동시 순서군의 기록 시점과 간격이 서로 겹치는 범위만 남긴 분석도 함께 수행합니다.
또한 1·3·5년 CAD 결과를 따로 계산하여 순서에 따른 차이가 시간에 따라 뒤집히는지
확인합니다.

이 분석은 질환 기록 순서가 CAD를 일으킨다는 인과효과를 추정하지 않습니다. GPU,
FERMAT 체크포인트 및 미래 생성은 사용하지 않습니다.

## 4. 실행 후 저에게 전달해 주실 내용

정상 종료하면 다음 부분 전체를 저에게 전달해 주세요.

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

환자별 raw Parquet은 Pod 밖으로 옮기지 않으셔도 됩니다.
