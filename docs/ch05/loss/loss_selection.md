# 손실 함수의 선택
알맞은 손실 함수를 고르는 것은 데이터, 과제, 바라는 최적화 거동에 대한 가정을 담는 모형화의 결정이다. 이 절은 손실 함수를 고르는 체계적인 틀을 제시하며 이론적 바탕과 실용적인 판단 기준을 이어 준다.

## 판단의 틀

### 1단계: 과제의 종류

첫 갈래는 회귀와 분류를 가른다.

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

| 특성 | 권장 손실 | 이유 |
|---------------|-----------------|--------|
| 깨끗한 데이터, 정규 잡음 | MSE | 통계적으로 최적 (크라메르-라오) |
| 이상점이 있을 때 | MAE 또는 후버 | 극단값의 영향이 유계 |
| 대체로 깨끗하고 이따금 이상점 | 후버 | 최적점 근처는 MSE의 정밀함, 멀리서는 MAE의 견고함 |
| 두꺼운 꼬리 잡음 | MAE | 라플라스 최대가능도. 중앙값 기반 추정 |
| 잡음의 분포를 모를 때 | 후버 (출발점) | 가장 안전한 기본값. $\delta$을 조율한다 |

**분류의 경우:**

| 특성 | 권장 손실 | 이유 |
|---------------|-----------------|--------|
| 균형 잡힌 클래스 | 교차 엔트로피 / BCE | 표준 최대가능도 |
| 보통의 불균형 | 가중 교차 엔트로피 | 클래스 균형 맞추기 |
| 심한 불균형 ($>\$10:1) | 초점 손실 | 쉬운 다수 클래스 예의 비중을 낮춘다 |
| 보정된 확률이 필요할 때 | 교차 엔트로피 / BCE | 확률적 출력 |
| 최대 여백이 필요할 때 | 힌지 손실 | SVM 같은 결정 경계 |

### 3단계: 응용에 따른 고려

| 분야 | 흔히 쓰는 손실 | 근거 |
|--------|------------|-----------|
| 물체 검출 (상자) | Smooth L1 / GIoU | 이상점에 견디는 경계 상자 회귀 |
| 의미 분할 | 다이스 + BCE | 불균형한 영역의 겹침 최적화 |
| 지식 증류 | KL 발산 | 교사의 분포에 맞추기 |
| VAE 학습 | 복원 + KL | ELBO 최대화 |
| GAN 학습 | BCE / 바서슈타인 | 판별기와 생성기의 목적 함수 |
| 순위 매기기 | 여백 기반 손실 | 쌍별 순서 |
| 언어 모형화 | 교차 엔트로피 | 다음 토큰 예측 |

## PyTorch 빠른 참조

### 회귀 손실

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

### 분류 손실

```python
logits_binary = torch.randn(32)           # 이진: 로짓 하나
labels_binary = torch.randint(0, 2, (32,)).float()

logits_multi = torch.randn(32, 10)        # 다중 클래스: 로짓 K개
labels_multi = torch.randint(0, 10, (32,))

# 이진 분류
bce = nn.BCEWithLogitsLoss()(logits_binary, labels_binary)

# 다중 클래스 분류
ce = nn.CrossEntropyLoss()(logits_multi, labels_multi)

# 클래스 가중치와 함께 (불균형을 위해)
weights = torch.ones(10)
weights[0] = 5.0  # 드문 클래스 0의 비중 올리기
ce_weighted = nn.CrossEntropyLoss(weight=weights)(logits_multi, labels_multi)
```

### 분포에 대한 손실

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

## 확률적 해석 안내

모든 손실 함수는 확률 모형을 함의한다. 손실을 고르는 것은 잡음의 분포를 고르는 것이다.

| 손실 | 함의된 모형 | $p(y \mid x, \theta)$ |
|------|--------------|----------------------|
| MSE | 정규 잡음 | $\mathcal{N}(f_\theta(x), \sigma^2)$ |
| MAE | 라플라스 잡음 | $\text{Laplace}(f_\theta(x), b)$ |
| 후버 | 가운데는 정규, 꼬리는 라플라스 | 후버 분포 |
| BCE | 베르누이 | $\text{Bern}(\sigma(f_\theta(x)))$ |
| 교차 엔트로피 | 범주형 | $\text{Cat}(\text{softmax}(f_\theta(x)))$ |
| 힌지 | — | 확률적 해석이 없다 (기하적이다) |

데이터를 만들어 낸 과정에 대한 믿음을 정할 수 있다면 손실 함수는 최대가능도 추정에서 따라 나온다.

$$\mathcal{L}(\theta) = -\frac{1}{m}\sum_{i=1}^m \log p(y^{(i)} \mid x^{(i)}, \theta)$$

## 흔한 실수

### 실수 1: 손실 앞에 활성화를 적용하기

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

### 실수 4: 분류에 MSE를 쓰기

```python
# 틀림: 분류에 MSE
probs = F.softmax(logits, dim=1)
loss = nn.MSELoss()(probs, one_hot_targets)  # 기울기가 나빠 수렴이 느리다

# 올바름: 분류에 교차 엔트로피
loss = nn.CrossEntropyLoss()(logits, targets)
```

분류에서 MSE의 기울기는 $p \approx 0$이나 $p \approx 1$일 때(시그모이드/소프트맥스의 포화 구간에서) 사라져 학습이 몹시 느려진다. 교차 엔트로피의 기울기 $p - y$에는 이런 문제가 없다.

## 진단 기준

학습이 기대대로 나아가지 않으면 손실 함수가 원인일 수 있다. 다음을 점검해 보라.

**손실이 줄지 않을 때:**

- 분류라면 손실 앞에 소프트맥스나 시그모이드를 적용하고 있지 않은지 확인하라
- 이상점이 있는 회귀라면 MSE에서 후버로 바꾸라
- 레이블과 예측의 모양과 자료형이 맞는지 확인하라

**학습이 불안정할 때 (손실이 진동하거나 NaN):**

- MAE에서 후버로 바꾸라 (최적점 근처에서 기울기가 매끄럽다)
- 기울기 자르기를 넣으라
- 사용자 정의 손실이라면 로그의 인수에 엡실론을 더하라

**수렴은 하는데 성능이 나쁠 때:**

- 불균형한 분류라면 초점 손실이나 클래스 가중치를 쓰라
- 분할이라면 BCE에 다이스 손실을 더하라
- 이분산 잡음이 있는 회귀라면 분산을 학습하는 방법을 고려하라($\sigma$을 출력하는 음의 로그가능도)

**모델이 이상점에 과적합할 때:**

- MSE에서 MAE나 후버로 바꾸라
- 정칙화(가중치 감쇠)를 넣으라
- 로버스트 손실(절사평균, 윈저화 손실)을 고려하라

## 요약표

| 손실 함수 | PyTorch 클래스 | 과제 | 주요 성질 |
|--------------|---------------|------|-------------|
| MSE | `nn.MSELoss` | 회귀 | 매끄러운 기울기, 정규 최대가능도 |
| MAE | `nn.L1Loss` | 회귀 | 이상점에 견고, 라플라스 최대가능도 |
| 후버 | `nn.HuberLoss` | 회귀 | MSE와 MAE의 혼합 |
| Smooth L1 | `nn.SmoothL1Loss` | 회귀 | 물체 검출의 표준 |
| BCE | `nn.BCEWithLogitsLoss` | 이진 분류 | 베르누이 최대가능도 |
| 교차 엔트로피 | `nn.CrossEntropyLoss` | 다중 클래스 분류 | 범주형 최대가능도 |
| NLL | `nn.NLLLoss` | 다중 클래스 (로그 확률 입력) | 빔 탐색이나 사용자 정의 소프트맥스에 |
| 초점 | 직접 구현 | 불균형 분류 | 쉬운 예의 비중을 낮춘다 |
| 힌지 | `nn.MultiMarginLoss` | 최대 여백 분류 | 희소한 기울기 |
| KL 발산 | `nn.KLDivLoss` | 분포 맞추기 | 지식 증류 |
| 다이스 | 직접 구현 | 분할 | 겹침 최적화 |

## 핵심 정리

손실 함수의 선택은 단순한 기술적 선택이 아니라 모형화의 결정이다. 확률적 해석이 가장 뚜렷한 길잡이가 된다. 함의된 잡음의 분포가 데이터에 대한 믿음과 맞는 손실을 고르라. 회귀에서는 안전한 기본값인 후버에서 시작한 뒤 (깨끗한 데이터라면) MSE나 (이상점이 많다면) MAE로 특화하라. 분류에서는 다중 클래스에 `CrossEntropyLoss`을, 이진에 `BCEWithLogitsLoss`을 쓰고 클래스 불균형이 심하면 초점 손실을 더하라. PyTorch의 분류 손실에는 언제나 (확률이 아니라) 날 로짓을 넣고, 레이블의 형식이 맞는지 언제나 확인하라. 표준 손실로 모자랄 때에는 `nn.Module`을 상속한 사용자 정의 손실이 PyTorch 학습 생태계와의 호환을 지키면서 얼마든지 유연함을 준다.

## 연습문제

**연습문제 1.**
과제의 종류에 따라 알맞은 손실 함수를 고르는 판단 나무를 만들라.

??? success "연습문제 1 풀이"
    회귀는 MSE(정규 잡음), MAE(라플라스 잡음이나 이상점), 후버(섞인 경우)를 쓴다. 이진 분류는 BCE를, 다중 클래스는 교차 엔트로피를, 불균형에는 초점 손실을 쓴다. 순서형에는 누적 연결 함수를, 순위 매기기에는 삼중항/대조 손실을 쓴다. 핵심 원리는 손실이 가정한 잡음 모형과 맞아야 한다는 것이다.

---

**연습문제 2.**
최대가능도 추정을 통해 손실 함수마다 어떤 잡음 가정에 대응하는지 설명하라.

??? success "연습문제 2 풀이"
    MSE는 정규 잡음 아래의 최대가능도, MAE는 라플라스 잡음 아래의 최대가능도, 교차 엔트로피는 범주형 분포 아래의 최대가능도, 후버는 정규-라플라스 혼합 아래의 최대가능도이다. 모든 손실 함수는 확률 모형을 암묵적으로 가정하며, 알맞은 손실을 고르는 것은 알맞은 잡음 모형을 고르는 것이다.

---

**연습문제 3.**
사용자 정의 손실 함수는 언제 설계해야 하는가? 예를 들라.

??? success "연습문제 3 풀이"
    표준 손실이 과제에 특유한 비용 구조를 담지 못할 때이다. 예를 들어 의료 진단에서는 거짓 음성(질병을 놓치는 것)이 거짓 양성보다 훨씬 나쁘다. `pos_weight >> 1`인 가중 BCE나 사용자 정의 비대칭 손실이 이를 다룬다.

---

**연습문제 4.**
잔차가 클 때 MSE, MAE, 후버 손실의 기울기 거동을 견주고 학습 안정성에 어떤 뜻을 갖는지 설명하라.

??? success "연습문제 4 풀이"
    MSE의 기울기는 $r$이다(잔차와 함께 커지므로 이상점에서 불안정할 수 있다). MAE의 기울기는 $\text{sign}(r)$이다(상수라 안정적이지만 0 근처에서 요동친다). 후버의 기울기는 잔차가 작으면 $r$, 크면 $\delta\cdot\text{sign}(r)$이다(유계라 안정적이다). 후버가 두 세계의 좋은 점을 준다.
