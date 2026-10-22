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

## 최종 결과 2026년 10월 22일

19:15:40 KST에 9개 실행이 모두 완료됐다. 큐 기준 73.28분이 걸렸다.
W&B의 9개 finished 상태, 실행 커밋과 시험 Recall@10, 모든 마지막 epoch checkpoint
SHA-256을 20:13 KST에 대조했다. 각 실행의 결과 파이프라인 검사를 통과했다.
LoRA 세 seeds는 모두 검증으로 epoch 2를 선택하고 epoch 7에서 조기 종료했다.
실제 LoRA 실행은 seed당 약 20.6–21.8분이었다.

[전체 집계와 독립 검증](results/record_main_20261022/summary.json),
[bootstrap 그림 PNG](figures/record_lora_bootstrap_20261022.png),
[편집 가능한 SVG](figures/record_lora_bootstrap_20261022.svg).

### 시험 성능

동일한 시험 환자 1,862명, 단위는 Recall@10 %다. 평균±SD는 세 training seeds의
기술통계이며 신뢰구간이나 세 환자 집단의 독립 반복을 의미하지 않는다.

| 방식 | Seed 1 | Seed 2 | Seed 3 | 평균 ± 표본 SD |
|---|---:|---:|---:|---:|
| A 원래 최적화 일정의 frozen head | 56.08 | 55.96 | 56.30 | 56.11 ± 0.17 |
| C LoRA 일정과 맞춘 frozen head | 53.86 | 54.30 | 54.41 | 54.19 ± 0.29 |
| LoRA + head | 67.17 | 67.05 | 66.66 | 66.96 ± 0.27 |
| 저장 generation50 | 71.00 | 동일 출력 | 동일 출력 | seed 평균 아님 |

v1의 선택된 frozen-head 55.68%는 기존 재현 기준이며 위 A seeds 1·2·3과
구분한다. 새로운 seed 평균으로 기존 v1 수치를 조용히 교체하지 않는다.

### Paired bootstrap

차이의 단위는 percentage points(%p), 대괄호는 95% percentile CI다.
모두 환자 단위 5,000회 paired resampling이며 양수는 LoRA가 높다는 뜻이다.

| 비교 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| LoRA − generation50 | −3.83 [−5.05, −2.61] | −3.95 [−5.21, −2.65] | −4.34 [−5.58, −3.09] |
| LoRA − A | +11.09 [9.98, 12.15] | +11.09 [9.94, 12.23] | +10.35 [9.27, 11.42] |
| LoRA − C | +13.31 [12.18, 14.46] | +12.75 [11.57, 13.93] | +12.25 [11.12, 13.36] |

사전 지정 1차 대비는 seed 1 LoRA − generation50뿐이다. Seeds 2·3은 민감도,
A/C 대비는 탐색적 2차 비교이며 다중 비교 보정을 하지 않았다. CI는 학습된 모델과
저장 생성 출력에 조건부이고 학습/생성의 모든 불확실성을 포함하지 않는다.
환자 수는 1,862명이며 세 seeds를 합쳐 5,586명으로 처리하지 않았다.

NumPy stable ranking과 정답 집합 교집합으로 raw logits에서 Recall@10을 독립
재계산했고, 환자별 차이의 bootstrap을 다시 계산했다. 원래 resampling 순서
(generation50 대비는 dataset 순서, A/C 대비는 환자 ID 정렬)를 유지했다.
기존 추정치/CI와 최대 차이는 1.44e-8 %p 미만으로, A/C CSV의 소수 6자리 반올림과
부동소수점 연산 차이 범위다. 논문 표는 기존 저장 결과를 유지한다.

그림의 선은 위 조건부 CI이며 마름모는 1차 대비다. 나머지 표식은 민감도/탐색적
비교다. 학습 seed 간 SD와 환자 bootstrap CI를 같은 오차막대로 혼동하지 않는다.

### 원고 반영 권고

1. Methods에 parameter-efficient supervised adaptation을 추가한다. 동일 코호트,
   선형 head, A/C 일정, LoRA rank 8/alpha 16/attention/dropout .05, 검증 선택,
   3 seeds, 고정 비교 및 bootstrap 범위를 적는다. Backbone adapter는 307,200개
   (기반 모델의 0.471%)지만 head 11,191,860개도 함께 학습했다. 전체 학습 파라미터를
   307,200개라고 쓰지 않는다. LoRA+head 총수는 11,499,060개다.
2. Results의 next-record 표에 LoRA와 A/C를 추가한다. 본문에서 1차 seed 결과와
   bootstrap을 먼저 쓰고, 세 seeds 평균±SD와 전체 CI를 보충표로 제공한다.
   C만 비교하면 개선량을 더 크게 보이게 할 수 있으므로 성능이 높은 A도 함께 제시한다.
3. Discussion은 고정 표현보다 지도 적응이 성능을 높였고 생성과의 차이를 줄였다는
   점을 인정하되, 이 next-record 과제의 검증된 설정에서는 generation50의 이점이
   남았다고 쓴다. 두 응용의 과제별 성격이라는 연구 관점을 유지한다.
4. 기존 fine-tuning 미평가 한계를 기록 과제에 대해서는 갱신한다. 완전 미세조정,
   모든 LoRA 설정에 대한 최적화, 외부 검증, 질병 과제 LoRA까지 검증했다고 쓰지 않는다.
   C는 일정 통제군이지 dropout/global clipping 차이까지 제거한 순수 adapter 효과
   실험은 아니다. 이 결과만으로 생성의 시간 정확도·임상 유용성도 입증하지 않는다.

### Results 문장 초안

Supervised LoRA adaptation improved next-record-day Recall@10 relative to the
frozen-representation predictors. In the prespecified seed-1 comparison, LoRA
achieved 67.17%, compared with 56.08% for the frozen head under the original
optimization schedule and 53.86% for the schedule-matched frozen-head control.
Aggregation of 50 generated trajectories achieved 71.00%; the paired difference
for LoRA minus generation was −3.83 percentage points (95% CI, −5.05 to −2.61).
Across three training seeds, LoRA achieved 66.96% ± 0.27% (mean ± sample SD).
Confidence intervals used 5,000 paired patient bootstrap resamples and were
conditional on the fitted predictors and saved generation outputs.

### 리뷰어 답변 초안

To address the concern that frozen representations may understate the value of
supervised adaptation, we evaluated LoRA fine-tuning on the same next-record-day
training, validation, and test cohorts. We included frozen-head baselines under
both the original and LoRA-matched optimization schedules and evaluated three
training seeds. For the prespecified seed, LoRA increased Recall@10 from 56.08%
to 67.17% relative to the original-schedule frozen head (difference, 11.09
percentage points; exploratory paired 95% CI, 9.98 to 12.15). The corresponding
gain over the schedule-matched control was 13.31 points (12.18 to 14.46).
Generation with 50 trajectories retained higher Recall@10 at 71.00% (LoRA minus
generation, −3.83 points; 95% CI, −5.05 to −2.61). The direction was consistent
across all three seeds. These results support a task-specific advantage for
generation under the evaluated adaptation configuration, rather than a general
claim that generation outperforms all fine-tuned predictors. This experiment
addresses next-record prediction; disease-risk fine-tuning remains unevaluated.

위 두 영문 문단은 원고와 답변에 넣을 제안문이다. 실제 원고 파일은 아직 수정하지 않았다.
