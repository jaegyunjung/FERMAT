# 기록 과제 v1 실제 재현 결과 — 2026-10-22

이 문서는 v1 gate 완료 시점의 기록이다. 이후 LoRA 소량 측정은
[LORA_BENCHMARK_KO.md](LORA_BENCHMARK_KO.md)에 별도로 기록했다.

**CUDA BF16 전체 임베딩 대조, 기록 head 재학습 및 재현 gate 통과.**
질병 과제 실제 재현과 LoRA 학습은 아직 하지 않았다. PR #2는 draft로 유지한다.
프로토콜 `2026-10-22.5`, 실행 커밋 `3903e92f2f64b2511c1ce7220cd0f486cee0ce3b`.

## 최종 실행

[W&B 최종 실행](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/7bgw0cey).
로컬 `PREP=/home/fermat/fermat-data/restricted/lora_baseline_20261022`,
최종 run은 `$PREP/runs/record_fresh_v1_03`이다.
Git에는 [summary.json](results/record_v1_20261022/summary.json)과
[gate.json](results/record_v1_20261022/gate.json)만 복사했다.
환자별 자료·모델·임베딩·예측 파일은 제한 폴더에 남긴다.

| 확인 항목 | 결과 |
|---|---|
| 입력 | 학습 18,474명 / 검증 1,867명 / 시험 1,862명, 출력 항목 17,460개 |
| 전체 임베딩 | 원본 23,991명 CUDA BF16 벡터 모두 원소 단위 완전 일치; 최대 절대 오차 0 |
| 정규화 | 정답 없는 환자를 포함한 원본 학습 19,992명; 저장 평균·표준편차와 완전 일치 |
| 설정 | seed 20260909, 추출·점수 계산 batch 32, head 학습 batch 256 |
| 학습 | 29 epoch, 2,117 optimizer updates; 검증으로 선택한 epoch 19 |
| 선택 epoch 검증 Recall@10 / BCE | 55.205024719% / 21.214193344 |
| 시험 Recall@10 | **55.68168739085007%**; 기존 v1 55.68% 재현 |
| 공식 gate | **PASS**, 기준 55.68 ±1.0%p; 허용오차 변경 없음 |
| 독립 재검사 | 공식 targets로 gate 재생성 일치, validate_gate 거부 사유 없음 |
| 예측 대조 | 저장 임베딩 진단 및 이전 fresh 실행과 시험 환자 순서·전체 logits·top-10 모두 완전 일치 |
| 기반 모델 | 변경 없음; 이번 실행의 adapter 파라미터 0 |

원본 saved head의 같은 점수에 원본 환자별 seeded tie와 새 stable tie를 적용했다.
시험 1,862명에서 경계 동점 0명, top-10 변경 0명, Recall@10 차이 0%p였다.
이 결과를 다른 head나 코호트의 동점 영향 상한으로 일반화하지 않는다.

## 진단에서 발견해 고친 문제

1. **정규화 대상 환자:** 정답 있는 18,474명만으로 계산하던 통계를 원본 코드의
   전체 학습 19,992명으로 복구했다. 초기 진단은 Recall@10 55.851275%로 넓은
   gate 오차 안에 있었지만, 원본 레시피와 달라 재현 증거로 채택하지 않았다.
2. **BF16 추출 batch 문맥:** batch 크기 32만 같아도 split을 먼저 나누면 마지막
   batch의 구성과 패딩 길이가 달라졌다. 초기 fresh 실행은 일부 학습/검증 벡터의
   최대 절대 오차 0.012985/0.018103으로 head 학습 전에 중단됐다. 원본 23,991명
   순서로 한 번 추출한 뒤 각 split에 매핑하자 모든 벡터가 완전히 일치했다.
3. **실행 코드 식별:** Windows가 만든 worktree의 .git 경로를 Linux git이 해석하지
   못해 첫 성공 실행의 commit 필드가 null이었다. WSL의 Windows git fallback을
   추가하고 재실행했다. 최종 summary, gate와 W&B config에 실행 커밋이 기록된다.

.5 개정은 위 source/artifact 기반 재현 수정이며 초기 진단 이후 이루어졌다.
사전 고정했다고 표현하지 않는다. 시험 성능을 높이기 위한 설정 탐색은 하지 않았다.
모든 이전 실행은 보존했다.

| 로컬 runs 하위 폴더 / W&B ID | 상태 |
|---|---|
| `record_cached_diagnostic_01` / `351g0b55` | 정규화 코호트 오류; superseded 표시, gate 없음 |
| `record_cached_diagnostic_02` / `772g6z5h` | 원본 head 레시피 진단 통과, 55.681687%; 진단 recipe라 gate 없음 |
| `record_fresh_v1_01` / `7ecsk8n1` | 전체 BF16 대조 실패로 학습 전 중단; 실패 상태 보존 |
| `record_fresh_v1_02` / `vct5aed3` | 전체 대조·gate 통과, 커밋 자동 식별 null이므로 최종 실행으로 대체 |
| `record_fresh_v1_03` / `7bgw0cey` | 최종 실행; 전체 대조·gate 통과, 커밋 기록 확인 |

## 실측 시간과 트래킹

환경: RTX 5090 32 GB, 400 W power limit / 1500 MHz 상한, driver 616.92,
WSL Ubuntu-24.04, torch 2.12.1+cu132, OMP/MKL threads=2.

- 최종 pipeline 136.791초(약 2분 17초); W&B 초기 연결·종료 시간은 제외.
- 모델·자료 로딩, 전체 추출·대조·hash를 포함한 준비 단계 79.609초.
- 준비 이후 head 학습·선택·시험 평가·저장 약 57.181초. epoch 중앙값 1.671초.
- 저장 임베딩 진단은 60.204초. 동일 환경 재실행 예산은 fresh 약 3분,
  진단 포함 약 4~5분으로 잡을 수 있다. 최초 CUDA 안정성 probe·export는 별도다.
- 전체 데이터 hash 읽기는 warm cache 약 0.22초; cold cache 시간은 미측정.
- 감시 표본 최대 전력 246.13 W, 온도 48°C, 장치 전체 사용 메모리 5,304 MiB.
  마지막 값은 프로세스별 peak allocation이 아니며 표본 사이 peak를 보장하지 않는다.
- 실행 중 guard가 검사한 새 System GPU/WHEA 오류나 재부팅 대기 상태는 없었다.
- LoRA는 backward 비용이 다르므로 위 head 시간으로 ETA를 계산하지 않는다.
  작은 LoRA 실행에서 step 시간·메모리를 측정한 후 전체 환자 수/epoch/seed로 산출한다.

W&B는 이후 arm·seed 비교와 실패 원인 추적에 유용하므로 online 집계 추적을 사용했다.
다만 실행 자체의 필수 의존성은 아니며 연결 실패 시 로컬 JSONL을 남긴다.
최종 원격 run의 `finished`, `gate/passed=true`, 시험 Recall@10을 API로 다시 확인했다.
업로드 파일은 `config.yaml`, `wandb-summary.json` 두 개였다. 환자별 표, 가중치,
코드, 콘솔, 자동 시스템 메타데이터를 업로드하지 않았다.

## 남은 범위

- 44개 CPU 테스트 통과(28.859초, torch 2.12.1). 다른 에이전트의 2.14.1 결과와 구분한다.
- 실제 기록 gate는 현재 동일 dataset/model/precision만 허용한다. 100k export는
  새로운 입력 hash를 가지므로 별도 호환성 설계·검증 없이 이 gate로 실행할 수 없다.
- 질병 입력 export, 고정 C10 전처리·확률 대조와 실제 질병 gate는 남아 있다.
- LoRA 소량 학습 및 본 실험은 미실행이다. 이번 gate 통과는 LoRA 성능 결과가 아니다.
- synthetic checkpoint reload는 검증했으나, real CUDA BF16의 임의 batch 크기나
  split-only `predict.py` replay는 이 결과로 보장하지 않는다. 그러한 replay가 필요하면
  original embedding-pool 문맥을 보존하는 경로를 추가하고 전체 logits를 비교해야 한다.
- 재실행 명령과 중단 기준은 PREPARATION_KO.md를 따른다.
