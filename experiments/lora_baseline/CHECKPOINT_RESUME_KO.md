# 기록 학습 체크포인트와 재개 검증 — 2026-10-22

이번 변경은 저장·복구와 예측 재현 검증이다. 본 실험은 외부 검토가 끝날 때까지
시작하지 않는다. 기록 과제에만 epoch 재개를 구현했으며 질병 과제는 대상이 아니다.
통계 프로토콜 .5, 학습 recipe, 재현 gate 허용오차는 변경하지 않았다.

## 저장 방식

| 파일 | 의미 |
|---|---|
| `checkpoints/epoch_NNNN.pt` | epoch 종료 후의 현재 adapter/head, optimizer, best 모델, best 지표, 조기 종료 counter, 전체 epoch 이력, RNG와 실행 계약을 담은 불변 파일 |
| `last_training.json` | 파일 경로·SHA-256·완료 epoch를 가리키는 원자적 commit 지점. 재개의 기준 |
| `best_model.pt` | 개선 시 갱신하는 추론용 편의 사본. 재개 기준은 위 manifest/payload |
| `trainable_state.pt` | 완료 후 선택 모델의 추론용 저장 파일. 단독으로는 optimizer 재개 불가 |
| `paused.json` | 요청한 epoch에서 저장하고 종료했음을 기록. 시험 평가 없음 |

같은 파일시스템의 임시 파일에 저장하고 flush/fsync 후 rename한다. 체크포인트를
다시 읽고 SHA-256을 계산한 뒤 manifest를 교체한다. 저장 중 실패하면 이전 manifest가
계속 유효하다. manifest 교체 전 생성된 고아 파일이나 앞서 갱신된 best 편의 사본을
재개 기준으로 사용하지 않는다. 이전 epoch 파일은 자동 삭제하지 않는다.

재개는 **새 출력 폴더**에 한다. 중단된 epoch의 미완료 gradient는 복구하지 않고
마지막으로 commit한 epoch 다음부터 다시 실행한다. 기존 결과는 덮어쓰지 않는다.

## 복원 및 호환성 검사

복원 대상은 현재 모델, AdamW 상태, optimizer update 횟수, best 모델/지표/epoch,
patience counter, Python random, NumPy 전역 RNG와 shuffle generator, Torch CPU/CUDA RNG다.
gradient accumulation은 epoch 끝에서 모두 반영한 후 저장한다. warmup은 복원한
전체 update 횟수에서 이어간다.

재개 시 데이터 내용·선택 환자·정답 정의·precision·모델, 릴리스 model_config.json의
별도 hash(동일 가중치의 dropout 설정 변경도 거부), seed와 모든 학습 설정,
epoch 상한·patience, 학습 핵심 소스와 릴리스 Python 코드 hash, PyTorch/NumPy/Python/
CUDA 버전, GPU·profile, thread 수 및 주요 수치 설정을 대조한다. frozen 전처리 통계도
다시 계산하여 저장값과 정확히 비교한다. 불일치에 대한 강제 우회 옵션은 없다.

학습 소스의 hash를 쓰므로 문서만 바꾼 커밋은 재개할 수 있지만, 학습 소스를 바꾼
경우에는 버전 문자열이 같아도 거부한다. GPU profile은 부팅 시각 대신 driver/전력/
clock 설정을 비교하며 현재 부팅에서 유효한지는 runtime guard가 별도로 확인한다.

W&B는 재개 실행마다 별도 run을 만들고 `resumed=true`, 이어지는 epoch를 기록한다.
매 epoch의 누적 optimizer update 수를 갱신한다. 로컬 fingerprint에는 이전 W&B run ID와 checkpoint SHA를 남긴다. 기존 W&B 이력을
수정하거나 덮어쓰지 않는다. 환자 자료·가중치는 업로드하지 않는다.

## 사용 예시

첫 실행과 같은 입력·gate·seed·recipe·정밀도·epoch 상한·patience를 지정한다.
아래는 명령 형태이며 이번에 본 실험을 실행했다는 뜻이 아니다.

```bash
# 선택적으로 epoch 경계에서 안전하게 멈춘다.
python train_record.py [동일한 학습 인자] --out /private/run_A --stop-after-epoch 3

# 중단되지 않은 실행과 동일한 총 epoch 상한을 유지한다.
python train_record.py [동일한 학습 인자] --out /private/run_A_resume \
  --resume /private/run_A/last_training.json
```

`--epochs`는 재개 이후 추가 epoch 수가 아니라 **원래의 총 상한**이다.
이미 상한/조기 종료 조건에 도달한 checkpoint라면 추가 학습 없이 선택 모델의
최종 처리를 수행한다. 스케줄러가 자동으로 재시작하는 기능은 포함하지 않는다.
이전 버전 최종 checkpoint 및 benchmark-only 파일은 이 재개 형식으로 변환하지 않는다.

## 검증 전용 모드

`--validation-only`는 LoRA 또는 experimental head-only에서만 허용한다.
시험 점수·시험 예측·gate를 생성하지 않고 `validation_scores.pt`와
`validation_summary.json`을 저장한다. 이 모드는 resume 계약에 포함되므로
검증 전용 checkpoint를 본 실험으로 바꿔 이어갈 수 없다.

```bash
python predict.py --state /private/run/trainable_state.pt \
  --release /private/release --data /private/data --out /private/reload \
  --device cuda --split validation --batch-size 32 --gpu-profile /private/profile.json \
  --reference-scores /private/run/validation_scores.pt
```

별도 프로세스에서 모델을 다시 만들고, 환자 순서·top-10의 정확한 일치와 logits의
rtol=1e-5/atol=1e-5 일치를 요구한다. 실제 최대 오차 및 bitwise 일치 여부도 기록한다.
원본과 같은 BF16, batch 크기, 환자 순서가 조건이다. 임의 batch 문맥의 재현을
보장하지 않으며, validation-only checkpoint로 시험 예측을 요청하면 거부한다.

실제 자료 검증용 `audit_record_resume.py`는 동일 128명에 2 epochs만 학습한다.
연속 실행, 1 epoch 후 종료, 새 프로세스에서 재개, 별도 prediction 프로세스를 순서대로
실행한다. 최종 가중치·optimizer·best/patience·RNG·epoch 지표·검증 예측을 비교한다.
전체 검증 1,867명을 사용하되 시험 집합은 평가하지 않는다.

## 검증 결과

CPU 52개 테스트 통과(64.969초, torch 2.12.1+cu132). 실패한 원자적 쓰기가 기존
파일을 보존하는지, manifest 게시 실패가 이전 epoch를 유지하는지, 손상 checksum 및
변경된 설정을 거부하는지 검사했다. head-only/LoRA의 중단·재개 대조에는 부분 gradient
accumulation, 별도 프로세스 validation reload, epoch 상한 이후 추가 학습 방지가 포함된다.

### 실제 CUDA 대조

환경은 RTX 5090 32 GB, driver 616.92, torch 2.12.1+cu132, BF16,
batch 32, seed 1이다. 기존 400 W / 1,500 MHz profile과 runtime guard를 사용했다.
학습 128명 × 2 epochs(총 8 updates), 검증 1,867명만 사용하는 기능 감사다.
성능 비교나 최종 seed 결과로 사용하지 않는다.

실행 커밋은 `70c974c62071ace72168e5a6d746be9fb56daff1`이다.
최종 결과는 [집계 JSON](results/record_resume_audit_20261022/summary.json)에 있다.

| 대조 항목 | 결과 |
|---|---|
| 연속 2 epochs vs 1 epoch 저장 후 새 프로세스에서 재개 | 현재 가중치, AdamW 상태, best 모델/지표, patience, 총 update 수, 모든 RNG, epoch 지표 정확히 일치 |
| 두 실행의 검증 예측 | 1,867명 전체 logits와 top-10 정확히 일치, 최대 절대 오차 0 |
| 최종 checkpoint를 별도 예측 프로세스에서 재로딩 | 환자 순서·logits·top-10 정확히 일치, 최대 절대 오차 0 |
| 시험 출력·재현 gate | 생성하지 않음 |
| W&B | 세 실행 모두 finished; validation_only=true, 재개 실행 resumed=true; 최종 update 수 8/4/8 확인 |

연속 실행 118.283초, 저장 후 종료 98.712초, 재개 실행 110.409초,
별도 재로딩 15.554초였다. 각 학습 실행은 원본 전처리용 임베딩 추출을 다시
수행하므로 이 시간은 순수 학습 시간이나 본 실험 ETA가 아니다.
체크포인트 게시 구간은 0.294–0.349초였다. 이 구간은 쓰기·재로딩·checksum·manifest
게시를 포함하지만, 그 전에 수행한 GPU→CPU 상태 복사 시간은 제외한다.

W&B: [연속 실행](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/eq1zp219),
[저장 후 종료](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/8yd6erl0),
[재개 실행](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/ciyav8e2).
원격 상태·실행 커밋·update 수를 확인했고, 업로드 파일은 config.yaml과
wandb-summary.json뿐이다. 환자별 자료·예측·가중치는 로컬 제한 폴더에 유지했다.

앞선 검증에서 발견한 W&B 누적 update 표시의 정체도 수정했다. 재개 시점의 4에
머무르던 표시가 최종 8로 갱신되며, 실제 학습 상태와 일치함을 CPU/CUDA에서 확인했다.

### 범위와 남은 검토

- 이번 CUDA 대조는 epoch 경계에서 정상 저장·종료한 뒤 새 프로세스로 재개했다.
  실제 전원 차단이나 OS 재부팅을 유발하지 않았다. 쓰기/manifest 게시 실패는
  CPU 테스트에서 주입해 이전 checkpoint가 보존되는지 확인했다.
- 같은 코드·수치 환경의 작은 학습에서 일치함을 확인한다. 전체 코호트의 장시간
  학습, 다른 GPU/driver, 임의 batch 변경까지 재현됨을 의미하지 않는다.
- epoch 파일은 약 184.36 MB다. 모두 보존하면 20 epochs당 약 3.69 GB,
  3 seeds 약 11.06 GB이며 best/final 파일과 로그 공간은 별도다.
- epoch 중간에 중단되면 마지막 저장 이후의 epoch를 다시 계산한다. 저장 전
  첫 epoch에서 중단되면 처음부터 시작한다. 자동 재시작과 질병 과제 재개는 미구현이다.
- 본 실험 전에는 독립 리뷰에서 비교 arm의 학습/선택 조건을 확정하고 OS 업데이트,
  재부팅 대기, GPU 상태와 저장 공간을 다시 확인한다. 100k 확대 실험의 gate 호환성과
  질병 과제 실제 재현은 별도 미완 항목이다.
