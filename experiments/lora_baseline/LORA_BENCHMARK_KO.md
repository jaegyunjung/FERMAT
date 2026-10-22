# 기록 과제 LoRA 소량 학습 및 ETA — 2026-10-22

**실제 128 optimizer step 측정 완료. 본 실험은 아직 시작하지 않았다.**
[W&B 실행](https://wandb.ai/jaegyunjung-seoul-national-university-hospital/fermat-lora-baseline/runs/mlpdq16w)
및 [집계 결과](results/record_lora_benchmark_20261022/summary.json).
실행 코드 `cd29589611d385d3004afa83696a98f3de3229a4`, 프로토콜 `2026-10-22.5`.

## 측정 조건과 검증

- RTX 5090 32 GB, 기존 400 W power limit / 1500 MHz 상한, driver 616.92.
  WSL Ubuntu-24.04, torch 2.12.1+cu132, OMP/MKL threads=2.
- 최종 기록 입력 `record_v1_original_batches`, 전체 학습 명단 18,474명을 유지했다.
  `record_fresh_v1_03/gate.json`의 모델·전체 dataset·정밀도·정답 정의를 검사한 뒤 실행했다.
- seed 1의 첫 epoch shuffle 중 128 batches, 4,096명만 학습했다.
  batch 32, BF16, rank 8, alpha 16, attention 대상, LoRA dropout .05,
  기반 모델 dropout .1, adapter lr 2e-4, head lr 1e-3, weight decay .01,
  gradient clip 1.0, LR warmup 100 updates, accumulation 1을 그대로 사용했다.
- 초기 10 step은 시간 평균에서 제외했다. 이는 100-step LR warmup과 다른 개념이다.
  나머지 118 step의 synchronized wall time은 batch 준비·forward·backward·optimizer·guard를 포함한다.
- seed 1의 측정 128 batches와 전체 578 batches 모두 최장 prefix가 512였다.
  따라서 측정 구간이 짧은 prefix 위주라 속도가 과대 추정된 경우는 아니다.
- LoRA 307,200개 파라미터(기반 모델 대비 0.471%)와 head 11,191,860개를 학습했다.
  adapter tensor 40개가 바뀌었고, 기반 모델 65,221,122개 파라미터는 완전히 동일했다.
- 전체 검증 1,867명의 평가 시간만 측정했다. 시험 성능·최적 epoch를 선택하지 않았고
  새 gate도 발급하지 않았다. 이것은 LoRA 성능 결과가 아니다.
- 비유한 손실·gradient, GPU 한계 초과, 새 System GPU/WHEA 오류, 재부팅 대기는 없었다.

## 속도와 메모리

| 항목 | 실측 |
|---|---:|
| 안정 구간 평균 step 시간 | 0.292037초 |
| 중앙값 / 10~90 백분위 step 시간 | 0.251090초 / 0.244158~0.326724초 |
| 처리량 | 109.58명/초 |
| 128 step 학습 총시간(초기 step 포함) | 40.540초 |
| 전체 검증 1회 | 6.774초 |
| 임베딩 준비·학습·검증·감사·저장 전체 | 127.906초, 약 2분 8초; W&B 초기화/종료 제외 |
| 최대 PyTorch CUDA allocated | **14.597 GiB** |
| 최대 PyTorch CUDA reserved | **14.887 GiB** |
| 표본 nvidia-smi 장치 전체 사용 메모리 최대 | 18,606 MiB = 18.170 GiB |
| 감시 표본 최대 전력 / 온도 | 245.03 W / 49°C |

CUDA peak는 학습 직전 통계를 reset하고 측정했으며 기존 모델 allocation을 포함한다.
장치 전체 메모리는 데스크톱·다른 프로세스·드라이버 비용도 포함하는 별도 지표다.
현재 batch 32로 메모리 부족은 없었다. 다른 모델/코호트/동시 작업에 일반화하지 않는다.

## 본 실험 ETA

동일 18,474명, batch 32, accumulation 1, 현재 GPU 설정이 조건이다.
`ceil(18474/32) = 578` batches/epoch.

- 학습 1 epoch: `578 × 0.292037 = 168.797초`.
- 검증 포함 1 epoch: `168.797 + 6.774 = 175.571초`, 약 **2분 56초**.
- 각 seed의 임베딩 준비 비용을 포함한다. 최종 시험 평가 시간은 검증 시간에
  환자 수 비율을 곱해 약 6.76초로 추정했다. 시험 성능 계산으로 측정한 값은 아니다.

| 실행 길이 | 1 seed 기본 추정 | 3 seeds 순차 실행 | 15~25% 여유 포함, 3 seeds |
|---|---:|---:|---:|
| 10 epochs | 30.7분 | 92.1분 | 약 1시간 46~55분 |
| 20 epochs(현재 상한) | 60.0분 | 179.9분 | **약 3시간 27~45분** |

실행 예약에는 **3시간 30~45분**을 권장한다. 실제 early stopping 시점은 예측할 수 없다.
기준 설정은 최대 20 epochs, 개선 없는 5 epochs 후 중단이다. 표는 10/20 epochs를
실제로 수행한다는 가정이며, 통계적 신뢰구간이나 완료 보장이 아니다.
체크포인트 복사/저장, 5,000회 paired bootstrap, W&B 종료, 순간적인 자원 경쟁은
기본 공식에서 빠져 있어 여유를 더했다. matched head-only 3 seeds는 별도 작업이다.
100k 확대는 gate 호환성 문제가 남아 있으므로 이 ETA를 100k 실행 허가로 사용하지 않는다.

## W&B와 산출물

원격 run `finished`, `progress/benchmark_completed=true`, 실행 커밋, 메모리/시간 지표를
API로 다시 확인했다. 5/10/20 epoch의 1-seed/3-seed ETA도 원격 summary에 기록했다.
원격 파일은 `config.yaml`, `wandb-summary.json` 두 개다. 환자별 정보·예측·가중치·코드는
업로드하지 않았다.

로컬 제한 폴더:
`/home/fermat/fermat-data/restricted/lora_baseline_20261022/runs/record_lora_benchmark_01/`

`benchmark.json`, `verification.json`, `gpu_telemetry.jsonl`, 집계 tracking JSONL과
`benchmark_only=true` 체크포인트를 보존했다. 체크포인트는 성능 선택 모델이 아니며
optimizer resume 상태도 없다. 본 실험은 seed별로 원래 초기화에서 새로 시작한다.

## 재실행 및 다음 단계

PREPARATION_KO.md의 환경변수를 설정한 뒤 다음 인자를 사용하는 gated LoRA 명령을 실행한다:

```bash
$PY "$CODE/train_record.py" --release "$RELEASE" --data "$DATA" \
  --out "$PREP/runs/record_lora_benchmark_new" --arm lora --recipe lora \
  --gate "$PREP/runs/record_fresh_v1_03/gate.json" --seed 1 \
  --batch-size 32 --device cuda --bf16 --benchmark-steps 128 --benchmark-warmup 10 \
  --gpu-profile "$PROFILE" --tracking online \
  --wandb-entity jaegyunjung-seoul-national-university-hospital
```

본 실험에는 `--benchmark-*`를 제거하고 새로운 seed별 출력 경로와 generation50 비교 CSV를
지정한다. 20 epochs 상한과 patience 5를 유지한다. 이번 작업에서는 실행하지 않았다.
장시간 실행 전에는 현재 Windows 업데이트/재부팅 일정과 GPU 상태를 다시 확인한다.
현재 업데이트 pause 만료일은 이미 지났지만, 이번 짧은 측정에는 재부팅 대기가 없었다.
이 측정 당시에는 resume가 미구현이었다. 이후 추가한 epoch 재개와 별도 CUDA
검증은 CHECKPOINT_RESUME_KO.md에 기록했다. 이 benchmark checkpoint에는 optimizer
상태가 없으므로 이후 기능으로도 본 실험을 이어 시작할 수 없다.

코드 검증은 CPU 46개 테스트 통과(32.277초). 새 테스트는 실제 adapter 갱신,
base 불변, 시험 출력/gate 미생성, ETA의 마지막 부분 batch 및 평가 비용을 확인한다.
