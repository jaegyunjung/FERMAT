# Task 30: ACEi 대 CCB 분석 가능성 확인

> 2026-07-14 수정: 최초 bundle `..._308e47af7319.zip`은 ATC에서 모든 종류의
> `concept_relationship`을 따라가면서 ACEi/CCB 83,186개, 전체 항고혈압제
> 991,290개를 포함한 오류가 있었다. 해당 실행은 중단하고 사용하지 않는다.
> 두 번째 bundle `..._06466d47cc9a.zip`도 전 세계 OMOP vocabulary의 제품을
> 먼저 전부 펼쳐 ACEi 16,659개, DHP-CCB 13,875개를 만든 뒤 SNUH 자료와
> 대조했다. 제품 concept 수로는 가능한 값이지만 SNUH 분석의 약제군을 만드는
> 순서가 잘못됐으므로 해당 실행도 중단하고 사용하지 않는다.
>
> 아래 v3 bundle은 SNUH `drug_exposure`에 실제 등장한 제품만 먼저 찾고,
> 그 제품에 한해서 ACEi·DHP-CCB·기타 항고혈압제 여부와 성분 수를 확인한다.

## 1. 코드가 하는 일

이 코드는 모델을 학습하거나 위험곡선을 그리지 않는다. 먼저 SNUH CDM에서 아래 숫자를 확정한다.

1. ACEi와 혈관선택성 CCB(DHP-CCB, 예: amlodipine)에 해당하는 실제 처방 concept와 FERMAT `RX:` 토큰
2. 365일 동안 항고혈압제 처방이 없다가 ACEi 또는 DHP-CCB를 시작한 환자 수
3. 같은 날 두 계열을 함께 시작했거나 다른 항고혈압제를 같이 시작한 환자 수
4. 치료 전 365일 관찰, 최근 고혈압 진단, 과거 심근경색·뇌졸중 없음 조건을 통과한 환자 수
5. 각 군의 1년·3년·5년 심근경색 또는 뇌졸중 사건 수와 관찰 가능한 환자 수
6. 사망원인 자료가 실제 MACE의 심혈관 사망을 만들 만큼 채워져 있는지

환자별 자료는 파일로 내보내지 않는다. 집계표와 사용 가능한 약제 토큰만 저장한다.

이 단계에서 `심근경색 또는 뇌졸중`은 표본 수 확인용 임시 결과다. 이를 최종 MACE라고 부르지 않는다. 심혈관 사망 포함 여부와 정확한 MACE 구성은 이전 초록의 정의를 확인한 뒤 고정한다.

## 2. 로컬 파일

- 실행 코드: `scripts/audit_snuh_task30_acei_ccb_mace_feasibility.py`
- bundle 생성 코드: `scripts/build_snuh_task30_acei_ccb_audit_bundle.py`
- Pod로 옮길 zip: `dist/task30/snuh_task30_acei_ccb_audit_108fa38_dirty_4058294dd122.zip`
- SHA256: `fab53e7928c0e51fc5d60b8a8c273c9bac5b3aabbe79f45525881eb35ecfbc99`

Pod에는 Python 파일을 따로 옮기지 않는다. zip 하나만 아래 위치로 옮긴다.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_acei_ccb_audit_108fa38_dirty_4058294dd122.zip
```

## 3. Pod에서 실행할 코드

```bash
set +e

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_acei_ccb_audit_108fa38_dirty_4058294dd122
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/acei_ccb_mace_feasibility_v3_$RUN_TAG"
LOG="$TASK_DIR/logs/acei_ccb_mace_feasibility_v3_$RUN_TAG.log"
CODE="$TASK_DIR/code/$BUNDLE-code"

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips
test -f "$TASK_DIR/zips/$BUNDLE.zip"

cd "$TASK_DIR/code"
unzip -o "$TASK_DIR/zips/$BUNDLE.zip" -d "$BUNDLE-code"

python3 -m py_compile "$CODE/scripts/audit_snuh_task30_acei_ccb_mace_feasibility.py"

python3 "$CODE/scripts/audit_snuh_task30_acei_ccb_mace_feasibility.py" \
  --output-dir "$OUT" \
  --lookback-days 365 \
  --db-end-date 2025-02-05 \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
echo "## EXIT_STATUS"
echo "$STATUS"
echo "## OUTPUT_DIR"
echo "$OUT"
echo "## LOG_FILE"
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

이 작업은 GPU를 사용하지 않는다. CDM의 처방·진단 테이블을 집계하므로 CPU와 데이터베이스 읽기 시간이 든다. 현재 실행 중인 HbA1c GPU pilot과 동시에 실행해도 GPU 메모리는 추가로 사용하지 않는다.

## 4. 실행 후 전달할 결과

위 실행 코드는 성공하면 `return_summary.txt`를, 실패하면 오류 로그와 남은 파일을
자동으로 출력한다. `## EXIT_STATUS`부터 마지막 줄까지 그대로 전달한다.

필요하면 실행 직후 화면에 출력된 `## OUTPUT_DIR` 값을 사용해 결과를 다시 볼 수 있다.

```bash
cat "$OUT/return_summary.txt"
```

`return_summary.txt`에는 다음 내용이 모두 들어 있다.

- OMOP에서 실제로 확인된 ACEi·DHP-CCB 정의
- FERMAT registry에 존재하는 상위 처방 토큰
- 두 약제군의 대상 환자 수와 제외 이유
- 1년·3년·5년 심근경색·뇌졸중 사건 수
- 연도별 환자 수
- 사망원인 입력률

오류로 종료됐으면 재실행하지 않는다. 실행 코드가 이미 아래 두 내용을 자동으로
출력한다.

```bash
echo '## LOG_TAIL'
tail -n 160 "$LOG"

echo '## SURVIVING_OUTPUTS'
find "$OUT" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
```

이 결과를 받은 다음에만 ACEi와 CCB를 같은 환자의 대안으로 넣어 1년·3년·5년 MACE 위험곡선을 비교하는 코드를 만든다.
