# 질병 LoRA 최종 결과 및 원고 반영안

2026-10-23 오전 08:09:04 KST에 18회 모두 완료했다. 시작은 전날 21:31:41,
총 벽시계 시간은 **10시간 37분 23초**다. C 대조군 9회와 LoRA 9회 모두 W&B finished다.
본 문서는 분석 보고서와 원고 제안이다. 기존 논문 원문은 수정하지 않았다.

## 결과를 읽는 기준

A는 기존 frozen embedding + fixed-C10 로지스틱 모델이다. C는 임베딩을 고정한 채
LoRA와 같은 미니배치 학습 일정으로 head만 학습한 대조군이다. B는 LoRA + head다.
A의 과거 수치를 새 B 수치로 덮어쓰지 않고 세 arm을 함께 보고한다.
AUROC는 높을수록 좋다. Brier는 확률 예측의 평균 제곱오차로 낮을수록 좋다.
Brier만으로 확률 보정(calibration)의 모든 측면을 평가했다고 쓰지 않는다.

## 세 seed의 최종 시험 성능

아래는 검증 weighted BCE로 고른 checkpoint의 시험 성능이다. 시험 성능으로 seed를 고르지 않았다.

| 질환 | 지표 | 기존 A | LoRA seed 1 | seed 2 | seed 3 | LoRA 평균 ± 표본 SD |
|---|---|---:|---:|---:|---:|---:|
| CKD | AUROC ↑ | 0.8706 | 0.9016 | 0.9003 | 0.9072 | 0.9030 ± 0.0036 |
| CKD | Brier ↓ | 0.02040 | 0.01914 | 0.02076 | 0.02014 | 0.02001 ± 0.00081 |
| 지방간 | AUROC ↑ | 0.7520 | 0.7903 | 0.7801 | 0.7753 | 0.7819 ± 0.0076 |
| 지방간 | Brier ↓ | 0.02725 | 0.02763 | 0.02747 | 0.02708 | 0.02739 ± 0.00028 |
| 당뇨병 | AUROC ↑ | 0.7840 | 0.8516 | 0.8568 | 0.8585 | 0.8557 ± 0.0036 |
| 당뇨병 | Brier ↓ | 0.04518 | 0.04217 | 0.04137 | 0.04121 | 0.04158 ± 0.00052 |

## 기존 A 대비 paired bootstrap

차이는 **LoRA − 기존 A**다. 대괄호는 95% percentile CI.
질환별 관측 환자(가중치 > 0)를 동일하게 짝지어 2,000회 재표집했고 관측 가중치는 고정했다.
seed 1이 사전에 정한 주 분석이다. seeds 2/3은 반복 실행 확인이다.

| 질환 | seed | ΔAUROC [95% CI] | ΔBrier [95% CI] |
|---|---:|---:|---:|
| CKD | 1 | +0.0310 [+0.0111, +0.0511] | -0.00126 [-0.00229, -0.00021] |
| CKD | 2 | +0.0297 [+0.0138, +0.0461] | +0.00036 [-0.00072, +0.00147] |
| CKD | 3 | +0.0366 [+0.0211, +0.0538] | -0.00026 [-0.00150, +0.00101] |
| 지방간 | 1 | +0.0383 [+0.0091, +0.0693] | +0.00038 [-0.00071, +0.00147] |
| 지방간 | 2 | +0.0281 [-0.0036, +0.0620] | +0.00021 [-0.00101, +0.00144] |
| 지방간 | 3 | +0.0233 [-0.0105, +0.0553] | -0.00018 [-0.00119, +0.00080] |
| 당뇨병 | 1 | +0.0676 [+0.0485, +0.0872] | -0.00302 [-0.00465, -0.00148] |
| 당뇨병 | 2 | +0.0728 [+0.0513, +0.0942] | -0.00382 [-0.00558, -0.00213] |
| 당뇨병 | 3 | +0.0745 [+0.0541, +0.0952] | -0.00398 [-0.00611, -0.00170] |

CKD는 세 seed 모두 AUROC 차이의 CI가 0보다 높다. Brier는 seed 1에서만 감소 구간이
0보다 낮다. seed 2/3의 Brier 차이는 불확실하다.
지방간은 세 seed 모두 AUROC 점추정이 높지만 seed 2/3 CI는 0을 포함한다.
Brier 차이는 세 seed 모두 불확실하다.
당뇨병은 세 seed 모두 AUROC 증가와 Brier 감소의 CI가 각각 0 위/아래에 있다.

구간은 각 비교별 95%이며 다중비교 보정은 하지 않았다. 세 seed를 독립 환자로 합치지 않았다.
평균±SD는 학습 seed 변동이고 bootstrap CI와 다르다. 이 CI는 학습된 모델과 가중치가
고정된 상태의 시험 환자 표집 불확실성을 나타내며 전체 모델 재학습 불확실성을 포함하지 않는다.

## C 대조군 및 C10 재학습 비교

| 질환 | C seed 1 AUROC / Brier | C seed 2 | C seed 3 | C 평균 AUROC / Brier |
|---|---:|---:|---:|---:|
| CKD | 0.8568 / 0.02072 | 0.8534 / 0.02107 | 0.8607 / 0.02059 | 0.8569 / 0.02079 |
| 지방간 | 0.7402 / 0.02734 | 0.7365 / 0.02788 | 0.7308 / 0.02745 | 0.7358 / 0.02756 |
| 당뇨병 | 0.7661 / 0.04582 | 0.7755 / 0.04510 | 0.7784 / 0.04570 | 0.7733 / 0.04554 |

주 seed 1의 보조 비교는 아래와 같다. 모든 seed의 보조 비교는 summary.json에 보존했다.

| 질환 | 비교 | ΔAUROC [95% CI] | ΔBrier [95% CI] |
|---|---|---:|---:|
| CKD | LoRA − 같은 seed의 C | +0.0449 [+0.0220, +0.0684] | -0.00157 [-0.00289, -0.00028] |
| CKD | LoRA − C10 재학습 A | +0.0310 [+0.0111, +0.0511] | -0.00126 [-0.00229, -0.00021] |
| 지방간 | LoRA − 같은 seed의 C | +0.0500 [+0.0193, +0.0831] | +0.00029 [-0.00091, +0.00150] |
| 지방간 | LoRA − C10 재학습 A | +0.0383 [+0.0091, +0.0693] | +0.00038 [-0.00071, +0.00147] |
| 당뇨병 | LoRA − 같은 seed의 C | +0.0855 [+0.0638, +0.1079] | -0.00365 [-0.00547, -0.00199] |
| 당뇨병 | LoRA − C10 재학습 A | +0.0676 [+0.0485, +0.0872] | -0.00302 [-0.00465, -0.00148] |

C의 평균 AUROC는 세 질환 모두 기존 A보다 낮았다. head의 최적화 일정을 바꾸는 것만으로
LoRA의 AUROC 개선이 재현되지는 않았다. C는 optimizer 일정 대조이며 backbone dropout과
clipping 대상 파라미터까지 동일하게 만드는 대조는 아니다.

## 실행과 검증

- 학습 커밋 `95713c679f6a11de1d0201f6bcfecdc70b39788b`, 실행기 `ad0a581`.
- 18개 checkpoint checksum, 저장 best 상태, 환자 순서와 시험 지표를 확인했다.
- 세 질환 seed 1에서 독립 CUDA checkpoint 재로딩 결과가 정확히 일치했다.
- W&B 18개 모두 finished, 서버 시험 지표와 로컬 결과 일치. config.yaml/summary 외 파일 업로드 없음.
- 독립 NumPy 그룹 누적 가중치 계산과 환자 재표집 횟수 방식으로 9개 bootstrap을 다시 계산했다.
  기존 계산과의 최대 절대 차이는 1.23e-15 미만. 동점·가중치·중복 환자에 대한 별도 대조도 통과했다.
- 후반 seed 3은 epoch당 약 32~36분으로, 전반의 19~25분보다 느렸다. 이 때문에 중간에
  갱신한 05:20~06:50 ETA보다 늦은 08:09에 종료됐다. 속도 저하 원인은 이 분석에서 확인하지 않았다.
- 당뇨병 seed 3은 마지막 허용 epoch 3을 선택했다. 학습 상한에서 최적화가 수렴했다고 단정하지 않는다.
- 최종 원시 예측과 가중치는 제한 폴더에 보존하고 이 저장소에는 집계만 저장한다.

## 논문에 반영할 내용

Methods에는 기존 fixed-C10 A, 같은 학습 일정의 frozen-head C, LoRA B를 구분한다.
질환별 동일 split/관측 가중치/적격 학습 환자 기반 전처리, rank 8/alpha 16,
학습 파라미터 307,845개, seeds 1/2/3, 최대 3 epochs/patience 1,
검증 weighted BCE 선택 규칙, seed 1 주 분석 및 paired bootstrap 방식을 명시한다.
C=10은 기존 원고 수정에서 고정한 값이며 이번에 사전 탐색 없이 새로 선정한 값이라고 쓰지 않는다.

Results 본문에는 기존 A와 LoRA 평균±SD 및 seed 1 CI를 싣는다. 부록에는 세 seed 전부,
C 결과와 보조 비교를 싣는다. 기존 frozen 결과를 유지하고 새 LoRA 열을 추가한다.
medRxiv 다음 버전에는 이 실험을 새 분석으로 명시하고 기존 v1에서 이미 했던 분석처럼 쓰지 않는다.

영문 Results 제안:

> Across three training seeds, LoRA-adapted predictors achieved mean AUROCs of
> 0.9030 ± 0.0036 for CKD, 0.7819 ± 0.0076 for fatty liver, and 0.8557 ± 0.0036
> for diabetes, compared with 0.8706, 0.7520, and 0.7840 for the original
> frozen-representation predictors. In the prespecified seed-1 comparison,
> AUROC improvements were 0.0310 (95% CI 0.0111–0.0511), 0.0383
> (0.0091–0.0693), and 0.0676 (0.0485–0.0872), respectively. Brier scores
> improved consistently across seeds for diabetes; improvement was seed-dependent
> for CKD and was not consistently observed for fatty liver.

영문 reviewer 답변 제안 (실제 원고 반영 후 사용하는 문안):

> We thank the reviewer for requesting a supervised fine-tuning comparison.
> We added LoRA-adapted predictors for all three disease outcomes, trained with
> three seeds on the same training, validation, and test cohorts as the original
> frozen-representation predictors. We also added a frozen-head control using
> the LoRA optimization schedule. Model selection used validation weighted BCE.
> LoRA increased AUROC point estimates for all three diseases in every seed.
> The prespecified seed-1 AUROC differences had positive 95% paired-bootstrap
> intervals for all three diseases. Across seeds, the AUROC improvement was most
> consistent for CKD and diabetes; the fatty-liver intervals crossed zero in
> seeds 2 and 3. Brier scores improved consistently for diabetes, whereas CKD
> and fatty-liver probability-error results varied across seeds. We report all
> seeds and both metrics in the revised results and supplementary tables.

연구 해석: 기존 frozen embedding 성능이 이 모델에서 얻을 수 있는 최고 성능은 아니었다.
지도 미세조정으로 질병 위험도 구분을 더 잘하게 만들 수 있었다. 특히 당뇨병은
위험도 구분과 확률 오차가 함께 개선됐다. 따라서 기존 논문의 frozen 성능을 모델의
성능 상한으로 설명한 문장은 수정한다. 생성 방식과의 수치 비교는 같은 시험 환자의
기존 생성 결과를 해당 표에 함께 놓고 판단한다. 이번 분석만으로 생성 대비 새로운 차이를 계산한 것은 아니다.
