# Task 30: 실제 치료 시작군의 심혈관 사건 곡선 가능성 확인

## 1. 코드가 하는 일

이 코드는 FERMAT이나 Cox를 사용하지 않는다. 먼저 실제 SNUH 자료만으로 다음
곡선이 계산되는지 확인한다.

- ACEi를 일차 약제로 시작한 환자의 급성 심근경색 또는 뇌졸중 누적발생 곡선
- DHP-CCB를 일차 약제로 시작한 환자의 급성 심근경색 또는 뇌졸중 누적발생 곡선

앞선 실행에서 이미 확인한 실제 SNUH 제품만 재사용한다.

- ACEi 제품 28개
- DHP-CCB 제품 48개
- 전체 항고혈압제 제품 316개

전 세계 OMOP 약제 vocabulary를 다시 펼치지 않는다. PostgreSQL 임시 테이블을
단계별로 만들고 각 단계의 환자 수를 즉시 `stage_counts.csv`에 저장한다.

사망은 이후 심근경색·뇌졸중을 관찰할 수 없게 만드는 경쟁사건으로 반영한다.
현재 곡선에는 급성 심근경색과 뇌졸중만 들어간다. 이전 초록에서 사용한
심혈관 사망 정의를 아직 확인하지 않았으므로 이 결과를 최종 MACE 곡선이라고
부르지 않는다.

## 2. 로컬 파일과 Pod로 옮길 파일

- 실행 코드: `scripts/run_snuh_task30_observed_mace_feasibility.py`
- bundle 생성 코드: `scripts/build_snuh_task30_observed_mace_bundle.py`
- Pod로 옮길 ZIP: `dist/task30/snuh_task30_observed_mace_feasibility_108fa38_dirty_f23e6c8d1ac8.zip`
- SHA256: `c6f5806804f41042e76b54e8fe9a849dd69948110ace92c28c1fa384ef8ff214`

ZIP 하나만 다음 위치로 옮긴다.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_observed_mace_feasibility_108fa38_dirty_f23e6c8d1ac8.zip
```

현재 중단한 v3 실행에서 아래 매핑 폴더는 그대로 재사용한다.

```text
/home/khdp-user/workspace/fermat-data/task30/outputs/acei_ccb_mace_feasibility_v3_20260714_014127
```

## 3. Pod 실행 코드

실패해도 창이 닫히지 않고, 성공 결과 또는 오류를 자동으로 출력한다.

```bash
set +e

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_observed_mace_feasibility_108fa38_dirty_f23e6c8d1ac8
MAPPING="$TASK_DIR/outputs/acei_ccb_mace_feasibility_v3_20260714_014127"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/observed_mi_stroke_curves_$RUN_TAG"
LOG="$TASK_DIR/logs/observed_mi_stroke_curves_$RUN_TAG.log"
CODE="$TASK_DIR/code/$BUNDLE-code"

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips

cd "$TASK_DIR/code"
unzip -o "$TASK_DIR/zips/$BUNDLE.zip" -d "$BUNDLE-code"

python3 -m py_compile "$CODE/scripts/run_snuh_task30_observed_mace_feasibility.py"

python3 "$CODE/scripts/run_snuh_task30_observed_mace_feasibility.py" \
  --mapping-dir "$MAPPING" \
  --output-dir "$OUT" \
  --lookback-days 365 \
  --db-end-date 2025-02-05 \
  --max-curve-days 1826 \
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
else
  echo '## LOG_TAIL'
  tail -n 200 "$LOG"
  echo '## SURVIVING_OUTPUTS'
  find "$OUT" -maxdepth 1 -type f -printf '%f\t%s bytes\n' 2>&1 | sort
fi

read -r -p 'Press Enter to close this shell...' _
```

## 4. 실행 후 전달할 결과

위 명령이 출력하는 `## EXIT_STATUS`부터 마지막 줄까지 그대로 전달한다.

정상 종료하면 다음 내용이 포함된다.

- 각 환자 선정 단계의 환자 수
- ACEi 시작군과 DHP-CCB 시작군의 환자 수
- 각 군의 심근경색·뇌졸중 사건 수
- 경쟁사건으로 처리된 사망 수
- 1년·3년·5년 누적발생률
- 곡선 계산 가능 여부

오류가 발생하면 재실행하지 않는다. `stage_counts.csv`와 로그를 보고 어느
단계에서 실패했는지 먼저 확인한다.
