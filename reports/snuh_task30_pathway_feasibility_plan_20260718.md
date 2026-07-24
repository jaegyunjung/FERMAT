# Task 30: 질병 경로 타당성 CPU 점검

## 목적

현재 실행 중인 당뇨 미래 생성과 별개로, GPU를 사용하지 않고 다음 질문에 필요한
환자 수를 먼저 확정한다.

- 기준일 전에 선행질환이 기록된 환자가 얼마나 있는가?
- 기준일 전에는 결과질환이 없었던 환자 중 1년·3년·5년 사건이 얼마나 발생하는가?
- 선행질환 첫 기록부터 기준일까지의 기간별로 환자 수와 결과 발생이 충분한가?
- 두 선행질환을 사용하는 경우 기록 순서와 간격을 비교할 수 있는가?
- 선행질환과 결과질환의 reviewed condition group이 FERMAT DX 토큰으로 매핑되는가?

이 작업은 `torch`를 import하지 않고, 체크포인트·모델 학습·미래 생성을 전혀 사용하지
않는다. 저장된 Task 19/23 Parquet, phenotype concept map, token registry를 읽는다. 사망
Parquet를 지정하지 않으면 CDM의 death 테이블을 한 번 읽어 최소 사망일만 집계한다.

## 포함한 경로

1. 당뇨 진단 → 만성신장질환
2. 당뇨 진단 → 망막질환
3. 고혈압 진단 → 관상동맥질환
4. 고혈압 진단 → 허혈성 뇌졸중
5. 고혈압·이상지질혈증의 순서와 간격 → 관상동맥질환
6. 만성 B형간염 진단 → 간세포암

망막질환은 현재 phenotype이 당뇨망막병증보다 넓으므로, 환자 수가 충분하더라도
`needs_review`로 남는다. 다른 경로도 기록된 진단일을 생물학적 발병일로 해석하지 않는다.

## 산출물

- `pathway_feasibility_summary.csv`: 경로별 test 대상 환자와 1·3·5년 사건 수
- `pathway_group_horizon_summary.csv`: split·선행질환 기간 구간별 환자 수와 실제 누적발생
- `observed_pathway_incidence_curves.csv`: test 및 전체 환자의 실제 5년 누적발생곡선
- `two_source_order_summary.csv`: 두 선행질환의 기록 순서별 환자와 사건 수
- `pathway_phenotype_token_mapping.csv`: reviewed concept와 FERMAT DX 토큰 매핑
- `raw/pathway_patient_level.parquet`: 환자별 경로 구성 결과; Pod 밖으로 옮기지 않음
- `return_summary.txt`: 결과를 전달하기 위한 요약

기본 선행질환 기간 구간은 다음과 같다.

- 1–180일
- 181–730일
- 731–1,826일
- 1,826일 초과

두 선행질환 경로에서는 두 질환 중 나중에 처음 기록된 날짜를 해당 경로가 완성된
날짜로 사용한다.

## Pod Terminal 실행

아래 zip 하나를 다음 위치에 둔다.

```text
/home/khdp-user/workspace/fermat-data/task30/zips/snuh_task30_pathway_feasibility_108fa38_dirty_ecb2d1233bbe.zip
```

현재 GPU 실행과 함께 돌릴 때 CPU 경쟁을 줄이기 위해 연산 스레드를 2개로 제한하고
낮은 우선순위로 실행한다.

```bash
set +e

TASK_DIR=/home/khdp-user/workspace/fermat-data/task30
BUNDLE=snuh_task30_pathway_feasibility_108fa38_dirty_ecb2d1233bbe
RUN_TAG=$(date +%Y%m%d_%H%M%S)
OUT="$TASK_DIR/outputs/pathway_feasibility_$RUN_TAG"
LOG="$TASK_DIR/logs/pathway_feasibility_$RUN_TAG.log"
CODE="$TASK_DIR/code/$BUNDLE-code"

mkdir -p "$TASK_DIR"/code "$TASK_DIR"/outputs "$TASK_DIR"/logs "$TASK_DIR"/zips
test -f "$TASK_DIR/zips/$BUNDLE.zip"

cd "$TASK_DIR/code"
unzip -o "$TASK_DIR/zips/$BUNDLE.zip" -d "$BUNDLE-code"

python3 -m py_compile "$CODE/scripts/audit_snuh_task30_pathway_feasibility.py"
python3 "$CODE/scripts/audit_snuh_task30_pathway_feasibility.py" --self-test

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

nice -n 10 python3 "$CODE/scripts/audit_snuh_task30_pathway_feasibility.py" \
  --config-file "$CODE/config/snuh_task30_pathway_feasibility.csv" \
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
else
  echo '## LOG_TAIL'
  tail -n 200 "$LOG"
  echo '## SURVIVING_OUTPUTS'
  find "$OUT" -maxdepth 2 -type f -printf '%p\t%s bytes\n' 2>&1 | sort
fi
```

실행 중 CDM 비밀번호를 물으면 기존 SNUH CDM 비밀번호를 입력한다. 이전 Task 30
outcome feasibility 실행에서 저장한 `all_cause_death_dates.parquet` 위치를 알고 있다면
다음 인수를 추가하여 DB 조회도 생략할 수 있다.

```text
--death-cache /기존/output/raw/all_cause_death_dates.parquet
```

## 중단 또는 요약 실패 시

같은 `OUT`을 사용하고 마지막 실행 명령에 `--resume-from-raw`를 추가한다. 저장된 사망일과
환자별 경로 자료의 실행 설정이 일치할 때만 재사용한다.

새 output directory로 다시 시작하거나 기존 원자료를 삭제하지 않는다.

## 결과를 받은 뒤의 결정

`return_summary.txt`를 먼저 확인한다. 아래 조건은 최종 통계적 검정 기준이 아니라 다음
GPU 시험에 올릴 후보를 거르는 기본값이다.

- test 대상 환자 500명 이상
- test 5년 결과 사건 50건 이상
- 환자 100명 이상인 선행질환 기간 구간이 3개 이상
- source와 target 모두 registry DX 토큰이 존재
- 별도 concept review가 필요하지 않음

통과한 경로 중 실제 기간별 발생곡선이 구분되고 임상적 해석이 가장 명확한 경로를
하나 선택한다. 그다음에만 작은 생성 빈도·속도 시험을 설계한다.
