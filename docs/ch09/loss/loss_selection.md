# 로스 함수의 선택
알맞은 로스 함수를 고르는 것은 데이터, 과제, 바라는 최적화 거동에 대한 가정을 담는 모델링의 결정이다. 이 절은 로스 함수를 고르는 체계적인 틀을 제시하며 이론적 기초와 실용적인 판단 기준을 이어 준다.

---

## 1. 판단의 틀

### 1단계: 과제의 종류

첫 클래스는 회귀와 분류를 가른다.

```
Task Type?
├── Regression (continuous output)
│   → MSE, MAE, or Huber
├── Binary Classification (2 classes)
│   → BCE, Focal Loss, or Hinge
├── Multi-Class Classification (K > 2, mutually exclusive)
│   → Cross-Entropy or Hinge
├── Multi-Label Classification (multiple labels per sample)
│   → BCE (per label)
└── Distributional Matching
    → KL Divergence
```

### 2단계: 데이터의 성격

과제의 종류 안에서는 데이터의 성질이 최적의 선택을 정한다.

**회귀의 경우:**

| 특성 | 권장 로스 | 이유 |
|---------------|-----------------|--------|
| 깨끗한 데이터, 정규 잡음 | MSE | 고정 분산 가우시안 NLL과 같은 평균 모수의 최소점 |
| 이상점이 있을 때 | MAE 또는 후버 | 극단값의 영향이 유계 |
| 대체로 깨끗하고 이따금 이상점 | 후버 | 최적점 근처는 MSE의 정밀함, 멀리서는 MAE의 견고함 |
| 두꺼운 꼬리 잡음 | MAE | 라플라스 최대가능도. 중앙값 기반 추정 |
| 잡음의 분포를 모를 때 | 후버 (출발점) | 하나의 강건한 후보. 목표 스케일에 맞춰 전환점을 검증한다 |

**분류의 경우:**

| 특성 | 권장 로스 | 이유 |
|---------------|-----------------|--------|
| 균형 잡힌 클래스 | 교차 엔트로피 / BCE | 표준 최대가능도 |
| 보통의 불균형 | 가중 교차 엔트로피 | 클래스 균형 맞추기 |
| 많은 쉬운 예가 지배하는 불균형 | Focal Loss | 쉬운 다수 클래스 예의 비중을 낮춘다 |
| 보정된 확률이 필요할 때 | 교차 엔트로피 / BCE | 확률적 출력 |
| 최대 여백이 필요할 때 | 힌지 로스 | SVM 같은 결정 경계 |

### 3단계: 응용에 따른 고려

| 분야 | 흔히 쓰는 로스 | 근거 |
|--------|------------|-----------|
| 물체 검출 (상자) | Smooth L1 / GIoU | 이상점에 견디는 경계 상자 회귀 |
| 의미 분할 | 다이스 + BCE | 불균형한 영역의 겹침 최적화 |
| 지식 증류 | KL 발산 | 교사의 분포에 맞추기 |
| VAE 학습 | 복원 + KL | ELBO 최대화 |
| GAN 학습 | BCE / 바서슈타인 | 판별기와 생성기의 목적 함수 |
| 순위 매기기 | 여백 기반 로스 | 쌍별 순서 |
| 언어 모델링 | 교차 엔트로피 | 다음 토큰 예측 |

---

## 2. PyTorch 빠른 참조

### 회귀 로스

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)

# MSE: 깨끗한 회귀의 기본값
mse = nn.MSELoss()(predictions, targets)

# MAE: 이상점에 견고하다
mae = nn.L1Loss()(predictions, targets)

# 후버: MSE와 MAE의 혼합
huber = nn.HuberLoss(delta=1.0)(predictions, targets)

# Smooth L1: 물체 검출의 관례
smooth_l1 = nn.SmoothL1Loss(beta=1.0)(predictions, targets)
```

### 분류 로스

```python
logits_binary = torch.randn(32)           # 이진: 로짓 하나
labels_binary = torch.randint(0, 2, (32,)).float()

logits_multi = torch.randn(32, 10)        # 다중 클래스: 로짓 K개
labels_multi = torch.randint(0, 10, (32,))

# 이진 분류
bce = nn.BCEWithLogitsLoss()(logits_binary, labels_binary)

# 다중 클래스 분류
ce = nn.CrossEntropyLoss()(logits_multi, labels_multi)

# 클래스 웨이트와 함께 (불균형을 위해)
weights = torch.ones(10)
weights[0] = 5.0  # 드문 클래스 0의 비중 올리기
ce_weighted = nn.CrossEntropyLoss(weight=weights)(logits_multi, labels_multi)
```

### 분포에 대한 로스

```python
# 이산 분포에 대한 KL 발산
log_probs = F.log_softmax(logits_multi, dim=1)
target_probs = F.softmax(torch.randn(32, 10), dim=1)
kl = nn.KLDivLoss(reduction='batchmean')(log_probs, target_probs)

# VAE를 위한 KL 발산 (정규분포)
mu = torch.randn(32, 20)
logvar = torch.randn(32, 20)
kl_vae = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
```

---

## 3. 확률적 해석 안내

일부 로스는 특정 확률모형의 음의 로그우도에 대응한다. 이 해석은 유용하지만 모든 로스가 정규화된 확률모형의 NLL인 것은 아니다. 비용·순위·겹침 같은 목표에서 직접 설계한 로스도 있다.

| 로스 | 함의된 모델 | $p(y \mid x, \theta)$ |
|------|--------------|----------------------|
| MSE | 정규 잡음 | $\mathcal{N}(f_\theta(x), \sigma^2)$ |
| MAE | 라플라스 잡음 | $\text{Laplace}(f_\theta(x), b)$ |
| 후버 | 가운데는 정규, 꼬리는 라플라스 | 후버 분포 |
| BCE | 베르누이 | $\text{Bern}(\sigma(f_\theta(x)))$ |
| 교차 엔트로피 | 범주형 | $\text{Cat}(\text{softmax}(f_\theta(x)))$ |
| 힌지 | — | 확률적 해석이 없다 (기하적이다) |

데이터를 만들어 낸 과정에 대한 믿음을 정할 수 있다면 로스 함수는 최대가능도 추정에서 따라 나온다.

$$\mathcal{L}(\theta) = -\frac{1}{m}\sum_{i=1}^m \log p(y^{(i)} \mid x^{(i)}, \theta)$$

---

## 4. 흔한 실수

### 실수 1: 로스 앞에 활성화를 적용하기

```python
# 틀림: 시그모이드가 두 번
probs = torch.sigmoid(logits)
loss = nn.BCEWithLogitsLoss()(probs, targets)  # 시그모이드가 두 번 적용된다!

# 올바름: 날 로짓
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

```python
# 틀림: CrossEntropyLoss 앞의 소프트맥스
probs = F.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, targets)  # 확률에 내부 log_softmax + NLL이 적용된다!

# 올바름: 날 로짓
loss = nn.CrossEntropyLoss()(logits, targets)
```

### 실수 2: 레이블의 형식이 틀림

```python
# 틀림: CrossEntropyLoss에 원-핫 레이블
one_hot = F.one_hot(targets, num_classes=10).float()
loss = nn.CrossEntropyLoss()(logits, one_hot)  # 정수 인덱스를 받는다!

# 올바름: 정수 클래스 인덱스
loss = nn.CrossEntropyLoss()(logits, targets)
```

### 실수 3: 인수의 순서가 틀림

```python
# PyTorch의 관례: (예측, 목푯값)
loss = nn.MSELoss()(predictions, targets)  # ✓

# 어떤 프레임워크는 (목푯값, 예측)을 쓴다 — 조심하라
```

### 선택 4: 분류에서 MSE와 교차 엔트로피 비교

```python
# 틀림: 분류에 MSE
probs = F.softmax(logits, dim=1)
loss = nn.MSELoss()(probs, one_hot_targets)  # 그래디언트가 나빠 수렴이 느리다

# 올바름: 분류에 교차 엔트로피
loss = nn.CrossEntropyLoss()(logits, targets)
```

시그모이드 확률에 MSE를 적용하면 경사에 시그모이드 도함수가 추가로 곱해져, 확신한 오답에서도 작아질 수 있다. 확률에 대한 제곱 오차는 Brier score 등으로 사용할 수 있으므로 분류에 쓰는 것 자체가 오류는 아니다. 로짓 BCE·CE는 그 출력 레이어 상쇄를 활용하지만 전체 뉴럴 넷트웍의 경사 소실까지 해결하지는 않는다.

---

## 5. 진단 기준

학습이 기대대로 나아가지 않으면 로스 함수가 원인일 수 있다. 다음을 점검해 보라.

**로스가 줄지 않을 때:**

- 분류라면 로스 앞에 소프트맥스나 시그모이드를 적용하고 있지 않은지 확인하라
- 이상점이 있는 회귀라면 MSE에서 후버로 바꾸라
- 레이블과 예측의 모양과 자료형이 맞는지 확인하라

**학습이 불안정할 때 (로스가 진동하거나 NaN):**

- MAE에서 후버로 바꾸라 (최적점 근처에서 그래디언트가 매끄럽다)
- 그래디언트 자르기를 넣으라
- 사용자 정의 로스라면 로그의 인수에 엡실론을 더하라

**수렴은 하는데 성능이 나쁠 때:**

- 불균형한 분류라면 Focal Loss이나 클래스 웨이트를 쓰라
- 분할이라면 BCE에 다이스 로스를 더하라
- 이분산 잡음이 있는 회귀라면 분산을 학습하는 방법을 고려하라($\sigma$을 출력하는 음의 로그우도)

**모델이 이상점에 과적합할 때:**

- MSE에서 MAE나 후버로 바꾸라
- 정규화(웨이트 감쇠)를 넣으라
- 로버스트 로스(절사평균, 윈저화 로스)을 고려하라

---

## 6. 요약표

| 로스 함수 | PyTorch 클래스 | 과제 | 주요 성질 |
|--------------|---------------|------|-------------|
| MSE | `nn.MSELoss` | 회귀 | 매끄러운 그래디언트, 정규 최대가능도 |
| MAE | `nn.L1Loss` | 회귀 | 이상점에 견고, 라플라스 최대가능도 |
| 후버 | `nn.HuberLoss` | 회귀 | MSE와 MAE의 혼합 |
| Smooth L1 | `nn.SmoothL1Loss` | 회귀 | 물체 검출의 표준 |
| BCE | `nn.BCEWithLogitsLoss` | 이진 분류 | 베르누이 최대가능도 |
| 교차 엔트로피 | `nn.CrossEntropyLoss` | 다중 클래스 분류 | 범주형 최대가능도 |
| NLL | `nn.NLLLoss` | 다중 클래스 | 정규화된 로그 확률을 입력으로 받음 |
| Focal Loss | 직접 구현 | 불균형 분류 | 쉬운 예의 비중을 낮춘다 |
| 힌지 | `nn.MultiMarginLoss` | 최대 여백 분류 | 희소한 그래디언트 |
| KL 발산 | `nn.KLDivLoss` | 분포 맞추기 | 지식 증류 |
| 다이스 | 직접 구현 | 분할 | 겹침 최적화 |

---

## 7. 핵심 정리

로스는 예측 목적, 오류 비용과 데이터 가정을 함께 반영한다. 출력이 로짓인지 확률인지에 맞춰 API를 선택하고 목표 모양·자료형·집계를 맞춘다. BCEWithLogitsLoss와 CrossEntropyLoss는 로짓, BCELoss는 확률, NLLLoss는 로그 확률을 받는다. 불균형이나 이상치 기법의 추가 효과는 검증해야 한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로스 선택의 판단 기준을 과제·비용·입력 규약으로 나누어 설명하라.

</div>

??? success "풀이"
    먼저 회귀·분류·순위·분할의 예측 대상을 정한다. 큰 오차나 특정 오분류의 비용을 반영하고, 로스가 받는 로짓·확률·로그 확률과 목표 표현을 맞춘다. 확률모형 해석은 가능한 경우 활용하되 모든 선택을 잡음 분포 하나로 설명하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MSE·MAE·CE의 우도 해석과 후버의 해석에서 조건을 설명하라.

</div>

??? success "풀이"
    고정 분산 가우시안은 제곱 오차, 고정 스케일 라플라스는 절대 오차, 범주분포는 CE와 연결된다. 후버는 exp(-Huber)를 정규화한 구간별 밀도로 해석할 수 있지만 정규–라플라스 혼합은 아니다. 모수에 의존하는 정규화 상수나 스케일을 생략하면 같은 우도 목적이 아닐 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
비대칭 비용을 반영한 로스를 선택할 때 예측 확률과 결정 임계값을 어떻게 구분하는가?

</div>

??? success "풀이"
    오분류 비용을 가중 로스로 반영하거나 확률을 학습한 뒤 결정 임계값에서 반영할 수 있다. 가중 로스의 출력이 원래 데이터 분포의 보정된 확률과 같은 것은 아니므로, 목표 비용과 확률 보정·임계값을 검증한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
잔차가 클 때 MSE, MAE, 후버 로스의 그래디언트 거동을 견주고 학습 안정성에 어떤 뜻을 갖는지 설명하라.

</div>

??? success "풀이"
    MSE의 기울기는 $r$이다(잔차와 함께 커지므로 이상점에서 불안정할 수 있다). MAE의 기울기는 $\text{sign}(r)$이다(상수라 안정적이지만 0 근처에서 요동친다). 후버의 그래디언트는 잔차가 작으면 $r$, 크면 $\delta\cdot\text{sign}(r)$이다(유계라 안정적이다). 후버가 두 세계의 좋은 점을 준다.

## 정리하며

로스 선택은 목표와 출력 규약을 함께 정하는 작업이다. 음의 로그우도 해석의 조건을 명시하고, 이상치·불균형·비대칭 비용의 효과는 같은 검증 지표에서 비교한다. 수치 안정성은 로짓 기반 구현 등 적절한 식으로 확보하며 작은 상수를 더하는 것만으로 모든 문제를 해결하지 않는다.
