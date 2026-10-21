# FERMAT SNUH collaborator release · 2026-10-21

이 폴더는 이형철 교수님께 전달하는 실행 패키지의 코드입니다. 별도 전달 ZIP에는
`artifacts/`의 가중치·토큰 사전·검사 구간·선택적 예측기도 포함됩니다.
GitHub에서 코드만 내려받으면 가중치는 포함되지 않습니다.

기본 모델은 **65,221,122개 파라미터**, 2단계 학습의 **93,000 step**입니다.
기존 평가에 사용한 가중치에 **수정된 생성 코드**를 결합했습니다.
모든 과제에서 최고의 모델이라는 뜻이 아니라, 현재 검증 기록과 후속 평가를 갖춘
주 모델의 최적 검증 체크포인트를 전달하는 것입니다.

## 바로 실행

Python 3.10 이상을 사용하고, 이 폴더에서 실행합니다. 실제 확인한 환경은
`config/test_environment.json`에 기록합니다. CPU 실행에는 GPU가 필요하지 않습니다.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell에서는 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python examples/run_inference.py --device cpu --new-events 8
python -m unittest discover -s tests -v
python examples/finetune_step.py --device cpu
```

CUDA 환경에서는 해당 GPU에 맞는 PyTorch를 설치하고 `--device cuda`를 지정합니다.
예제 입력은 새로 만든 가상 환자이며, 실제 환자의 기록이 아닙니다. 첫 예제는
640차원 임베딩, 다음 항목의 확률, 다음 기록일 예측기 결과, 짧은 생성 기록을 출력합니다.
두 번째 예제는 실제 역전파와 optimizer 갱신을 한 번 수행하며 전달 가중치를 덮어쓰지 않습니다.
전체 학습을 중단 지점에서 재개하는 optimizer 상태는 포함하지 않습니다.

## 자신의 데이터에 적용

1. 진단·처방·시술 코드를 **OMOP 표준 concept ID**로 매핑합니다.
2. 검사 단위와 날짜를 맞추고 [입력 사양](docs/INPUT_SPEC.md)에 따라 변환합니다.
3. `examples/synthetic_patient.json` 형태로 `age_days`, `token_key`를 제공합니다.
4. `python examples/run_inference.py --patient my_patient.json`을 실행합니다.
5. 출력의 `unknown_fraction`과 항목별 매핑률을 먼저 확인한 후 새 데이터의 정답으로 평가합니다.

삼성생명 데이터의 ICD/KCD·수가·약제 코드를 그대로 토큰 번호로 넣으면 안 됩니다.
새 데이터의 사전을 다시 만들어 기존 번호를 바꾸는 것도 안 됩니다.
필요한 코드 매핑표는 해당 데이터에 맞게 준비해야 합니다.

## 포함 파일

| 파일 | 내용 |
|---|---|
| `artifacts/model.pt` | tensor-only `state_dict`; 원본 BF16 수치를 보존 |
| `config/model_config.json` | 구조 및 학습 시 마스킹 설정 |
| `fermat_release/model.py` | 학습 모델 구현의 정확한 스냅샷 |
| `fermat_release/inference.py` | 학습의 날짜 처리와 맞춘 수정 생성 코드 |
| `artifacts/token_registry.csv`, `lab_cutpoints.json` | 고정 토큰 사전, 학습 자료에서 정한 검사 구간 |
| `artifacts/next_record_head.pt`, `head_*.np*` | 10만 명으로 학습한 선택적 다음 기록일 예측기 |
| `config/artifact_manifest.json` | 파일 크기 및 SHA-256 검증값 |
| `docs/PROGRESS.md` | 완료한 작업, 성능, 남은 평가 |
| `docs/MODEL_CARD.md` | 모델 선정 근거, 사용 범위, 평가 버전 |

일반적인 사용은 `load_release()`와 `generate_consistent()`를 사용합니다.
학습 스냅샷에 남아 있는 `model.generate()`는 이전 방식이므로 이 배포본의 실행 경로로 사용하지 않습니다.
5년 발병확률은 위 짧은 생성 예제의 출력이 아닙니다. 해당 평가의 경로 종료 처리 기준은
[진행 요약](docs/PROGRESS.md)에 따로 기록했습니다.
