# 기록 과제 본 실험 실행 계획 — 2026-10-22

독립 검토를 받은 뒤 사용자가 본 실험 시작을 승인했다. 기록 과제의 동일한
18,474/1,867/1,862명 학습/검증/시험 자료만 사용한다. 기존 v1 gate는 역사적
재현 근거로 유지하고, 100k 확대와 질병 과제는 이번 실행에 포함하지 않는다.

## 실행 순서와 고정 조건

| 그룹 | 구현 | 학습 batch | 최대 epochs / patience | head LR / weight decay | warmup / clip | seeds |
|---|---|---:|---|---|---|---|
| A | frozen head, experimental | 256 | 60 / 10 | .001 / .01 | 0 / 없음 | 1, 2, 3 |
| C | frozen head, experimental, 일정 통제군 | 32 | 20 / 5 | .001 / .01 | 100 updates / 1.0 | 1, 2, 3 |
| LoRA | adapter + head, lora | 32 | 20 / 5 | .001 / .01 | 100 updates / 1.0 | 1, 2, 3 |

A 1·2·3 → C 1·2·3 → LoRA 1 → 결과 파이프라인 점검 → LoRA 2·3 순서다.
각 실행 뒤에도 같은 무결성 검사를 적용하며 실패하면 뒤 실행을 시작하지 않는다.
역사적 v1 seed 20260909의 통과 결과는 A와 별도의 재현 기준이다.

기록 추출/평가 batch는 모든 그룹에서 32다. `--batch-size`는 frozen head의
학습 batch를 바꾸지 않으므로 C에는 새 `--optimizer-batch-size 32`를 명시한다.
이 옵션은 `head_only --recipe experimental`에만 허용하며, resolved batch가
resume 계약에 포함된다. v1 recipe의 batch 256과 기존 gate 정의는 바꾸지 않는다.

공통 설정: AdamW betas (.9, .999), accumulation 1, 같은 head 초기화,
train-only frozen scaler(원래 학습 19,992명), 같은 환자 순서 생성 규칙과
검증 Recall@10/BCE tie-break 선택. LoRA는 rank 8, alpha 16, attention,
adapter LR 2e-4, adapter dropout .05, 기반 모델 dropout .1을 유지한다.

C는 **최적화 일정을 맞춘 통제군**이다. frozen head는 eval-mode의 고정 임베딩을
사용하고 LoRA는 train-mode transformer와 dropout을 사용하므로, C–LoRA 차이를
adapter 파라미터만의 순수 효과라고 주장하지 않는다. global gradient clipping의
대상도 각각 head와 head+adapter로 다르다. 추가 조정은 결과를 본 뒤 하지 않는다.

## 평가와 다음 seed 진입 기준

- 모든 실행은 검증으로 선택한 모델로 시험 집합을 한 번 평가한다. 이후 검사는
  저장된 logits/지표를 대조하며, 다른 모델을 고르거나 성능에 따라 재학습하지 않는다.
- 1차 대비는 사전 지정한 LoRA seed 1 − 저장 generation50의 환자별 Recall@10이다.
  동일 환자 1,862명을 paired bootstrap 5,000회 재표집하고 percentile 95% CI를 쓴다.
- LoRA seed 2·3은 seed 민감도이며 최상의 seed로 1차 결과를 대체하지 않는다.
  각 seed의 LoRA − 동일 seed A/C는 탐색적 2차 대비다. 환자를 seed 수만큼 복제하여
  표본 수를 늘리지 않는다. CI는 학습된 모델과 저장 rollout에 조건부이며 학습 seed나
  생성 경로의 불확실성까지 포함하지 않는다. 별도 검정 없이 다중 비교 확증 주장을 하지 않는다.
- 계속 실행할 조건: 같은 데이터/gate identity, base 가중치 불변, 유한한 loss/logits,
  검증 선택 규칙 준수, checkpoint checksum/source/선택 모델 일치, 시험 환자·top-10·
  집계 지표 일치, W&B 정상 종료. **정확도의 방향이나 크기는 계속 실행 조건이 아니다.**

## 실행 코드와 복구

실행 전 커밋을 별도 로컬 Git snapshot에 고정한다. 실행 중 이 폴더의 코드와
릴리스/입력 자료를 수정하지 않는다. 매 run 전에 source 및 주요 입력 hash를 다시
대조한다. 작업용 브랜치 변경은 실행 snapshot에 반영하지 않는다.

`run_record_suite.py`가 단일 실행 lock과 순차 subprocess를 관리한다. 임의 재시도나
자동 resume는 하지 않는다. 실패 시 status.json에 실패를 기록하고 큐를 멈추며,
복구는 원래 snapshot과 마지막 manifest로 새 출력 폴더에서 수행한다.
각 run timeout은 3시간이다. 이는 ETA가 아니라 무한 대기를 막는 상한이다.

epoch checkpoint를 전부 보존한다. A는 60 epochs × 3 seeds, C/LoRA는 각각
20 epochs × 3 seeds이므로 최대 **약 55 GB**에 best/final 파일·예측·로그 공간이
추가된다. 실행 전과 각 run 전 최소 70 GiB 여유를 요구한다. 11 GB 추정은
LoRA 세 seeds만의 크기이며 A/C 전체를 포함하지 않는다.

현재 Windows 부팅 이후 검증된 GPU profile과 guard를 사용한다. 새 GPU/WHEA 오류,
GPU 운영 한계 초과, 재부팅 대기를 발견하면 중단한다. 업데이트 pause는 만료되어
있지만 시작 점검에서 재부팅 대기는 없었다. OS 업데이트 정책을 임의로 바꾸지 않았다.

## W&B와 ETA

사용자가 계속하도록 승인한 기존 W&B 프로젝트/계정을 그대로 사용한다. 집계 수치와
허용된 비환자 설정만 전송하며 patient ID, 개별 예측, 가중치, 코드, 콘솔은 올리지 않는다.
기관 차원의 승인 여부를 이 코드 검증만으로 입증한 것으로 해석하지 않는다.
전송 실패는 로컬 JSONL에 남기고 다음 run을 시작하지 않는다.

기존 실측 기준 LoRA는 최대 20 epochs에서 seed당 약 60분, 3 seeds 기본 약 3시간,
여유 포함 약 3시간 30–45분이다. A/C는 별도이며 실제 완료 시간으로 갱신한다.
초기 전체 예약 예산은 **약 4시간**으로 잡되 보장된 완료 시각은 아니다.
early stopping 시점과 C의 실제 소요 시간은 아직 측정 전이다.

실행 상태는 제한 폴더의 `status.json`, 각 run의 `progress.json`, `tracking/`에 남긴다.
최종 `secondary_contrasts.json`과 run별 `pipeline_check.json`은 집계 결과이며,
전체 원자료와 모델은 제한 폴더 안에 유지한다.

## 시작 기록

- 2026-10-22 **18:02:21 KST** 큐 시작. 실행 커밋
  `24a8912e28e2486f7f107716681fcd27be7dfeaa`; 이후 작업 브랜치의 문서 변경과 분리한다.
- 실행 snapshot: `C:/Users/HP/.codex/runtimes/fermat-record-24a8912`.
  WSL 안의 snapshot은 Windows PowerShell의 네트워크 경로 실행 정책에 막혀 사용하지
  않았다. 동일 커밋의 Windows 로컬 snapshot을 사용하며 보안 정책을 바꾸지 않았다.
- 결과 루트: `/home/fermat/fermat-data/restricted/lora_baseline_20261022/record_main_suite_01`.
  큐 상태 `status.json`, 시작 manifest는 Windows
  `C:/Users/HP/.codex/runtimes/record-main-20261022.launch.json`에 있다.
- 사전 점검 통과: gate/data/model identity 일치, CUDA BF16 원본 batch 32명
  임베딩 최대 오차 0, GPU/host guard 통과, 799.23 GiB 여유. CPU **55 tests 통과**
  (72.174초, torch 2.12.1+cu132). [집계 근거](results/record_main_launch_20261022/summary.json).
- 첫 실행 [A seed 1 W&B](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/ky63fpr8).
  이 시작 기록은 완료 보고가 아니다. 결과/완료 상태는 로컬 status와 W&B에서 확인한다.
