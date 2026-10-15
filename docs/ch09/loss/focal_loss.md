# Focal Loss

Focal Loss는 정답 클래스 확률에 따라 교차 엔트로피의 비중을 조절한다. Lin 등은 조밀한 물체 검출에서 많은 쉬운 배경 예의 영향을 줄이기 위해 제안했다. 모든 불균형 문제에서 교차 엔트로피가 실패하거나 Focal Loss가 더 좋은 것은 아니다.

## 1. 클래스 불균형 문제

심하게 불균형한 데이터셋(예: 검출에서 배경 99%에 물체 1%, 사기 탐지에서 정상 99.9%에 사기 0.1%)에서는 표준 교차 엔트로피를 쓰면 모델이 다수 클래스만 내놓고도 높은 정확도를 얻는다. 핵심 문제는 정확도가 아니라 **기울기의 지배**이다. 수많은 쉬운 음성이 다 합쳐져 큰 기울기 신호를 만들고, 이것이 몇 안 되는 어려운 양성의 학습 신호를 삼켜 버린다.

이미지마다 후보 영역 $10^5$개를 살피는데 그중 물체가 든 것은 몇 개뿐인 검출기를 생각해 보자. 쉬운 음성 하나하나의 손실이 작더라도 $10^5$개가 쌓인 기울기는 몇 안 되는 양성의 신호를 훨씬 뛰어넘는다. 모델은 어디서나 "배경"을 내놓는 법을 배우는데, 이는 퇴화한 해이지만 손실은 낮다.

---

## 2. 수학적 정식화

### 표준 교차 엔트로피 기준선

이진 분류에서 표본 하나에 대한 교차 엔트로피 손실은 다음과 같다.

$$\text{CE}(p, y) = -y\log(p) - (1 - y)\log(1 - p)$$

$p_t$을 모델이 **참 클래스**에 매긴 확률이라 하자.

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

그러면 교차 엔트로피는 $\text{CE}(p_t) = -\log(p_t)$으로 간단해진다.

### 균형 잡힌 교차 엔트로피

불균형을 다루는 첫 시도는 클래스 균형 가중치 $\alpha_t$을 넣는 것이다.

$$\text{CE}_{\text{balanced}} = -\alpha_t \log(p_t)$$

여기서 양성 클래스에는 $\alpha_t = \alpha$, 음성 클래스에는 $\alpha_t = 1 - \alpha$이다. 이는 클래스마다 손실의 몫을 조정하지만 한 클래스 안에서 쉬운 예와 어려운 예를 가리지는 못한다.

### Focal Loss의 정의

Focal Loss은 교차 엔트로피에 **조절 인수** $(1 - p_t)^\gamma$을 더한다.

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

여기서 각 기호는 다음과 같다.

- $p_t$은 모델이 참 클래스에 매긴 확률이다
- $\gamma \geq 0$은 비중을 얼마나 낮출지 정하는 **집중 매개변수**이다
- $\alpha_t$은 선택적인 클래스 균형 가중치이다

### 조절 인수가 하는 일

인수 $(1 - p_t)^\gamma$은 예측의 확신도에 따라 손실을 매끄럽게 조정한다.

- **잘 분류된 예** ($p_t \to 1$): 인수 $(1 - p_t)^\gamma \to 0$이므로 손실에 대한 기여가 0에 가까워진다. 쉬운 예는 사실상 무시된다.
- **잘못 분류된 예** ($p_t \to 0$): 인수 $(1 - p_t)^\gamma \to 1$이므로 손실이 그대로 남는다. 어려운 예는 기울기 신호를 온전히 지킨다.
- **확신이 어중간한 예** ($p_t \approx 0.5$): 인수가 중간 정도로 비중을 낮춘다.

$p_t = 0.9$, $\gamma = 2$인 잘 분류된 예에 대해 다음과 같다.

$$(1 - 0.9)^2 = 0.01$$

표준 교차 엔트로피에 비교하여 손실이 100분의 1로 줄어든다. $p_t = 0.1$인 잘못 분류된 예에 대해서는 다음과 같다.

$$(1 - 0.1)^2 = 0.81$$

손실이 $\sim 19\%$만 줄어든다. 이렇게 차등적으로 비중을 낮추면 최적화의 힘이 가장 중요한 예로 옮겨 간다.

### gamma의 효과

| $\gamma$ | 거동 |
|----------|----------|
| $\gamma = 0$ | 클래스 가중 교차 엔트로피. 가중치가 1이면 표준 CE |
| $\gamma = 1$ | 쉬운 예의 비중을 조금 낮춤 |
| $\gamma = 2$ | 표준적인 선택. 비중을 강하게 낮춤 (권장 기본값) |
| $\gamma = 5$ | 비중을 아주 세게 낮춤. 쉬운 예를 거의 무시함 |

---

## 3. 경사 분석

로짓 $z$($p = \sigma(z)$)에 대한 Focal Loss의 기울기는 다음과 같다.

$$\frac{\partial\mathcal L_{FL}}{\partial z}=\alpha_t(2y-1)(1-p_t)^\gamma\left[\gamma p_t\log p_t-(1-p_t)\right]$$

여기서는 $y\in\{0,1\}$, $\partial p_t/\partial z=(2y-1)p_t(1-p_t)$를 사용했다. 조절 인수도 예측값에 의존하므로 그 미분까지 포함해야 한다. 손실에 곱한 인수를 그대로 기존 경사에 곱하는 계산과는 다르다.

---

## 4. PyTorch 구현

### 이진 Focal Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """이진 분류를 위한 초점 손실.

    참고: "Focal Loss for Dense Object Detection" (Lin 등, 2017)
    https://arxiv.org/abs/1708.02002

    인수:
        alpha: 양성 클래스의 균형 가중치. 기본값: 0.25.
        gamma: 집중 매개변수. 값이 클수록 쉬운 예의 비중을
               더 낮춘다. 기본값: 2.0.
        reduction: 'mean', 'sum', 또는 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        인수:
            logits: (시그모이드 전의) 모델의 날 출력, 모양 (N,).
            targets: 이진 레이블 (0 또는 1), 모양 (N,).
        """
        probs = torch.sigmoid(logits)

        # p_t: 참 클래스에 매긴 확률
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # alpha_t: 클래스 균형 가중치
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # 초점 조절 인수
        focal_weight = (1 - p_t) ** self.gamma

        # 표준 BCE (표본별, 수치적으로 안정)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 초점 손실
        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### 다중 클래스 Focal Loss

```python
class MultiClassFocalLoss(nn.Module):
    """다중 클래스 분류를 위한 초점 손실.

    인수:
        alpha: 클래스별 가중치, 모양 (K,). None이면 클래스 가중을 하지 않는다.
        gamma: 집중 매개변수. 기본값: 2.0.
        reduction: 'mean', 'sum', 또는 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        인수:
            logits: 모델의 날 출력, 모양 (N, K).
            targets: 클래스 인덱스, 모양 (N,).
        """
        # 소프트맥스 확률 계산
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # p_t 모으기: 참 클래스의 확률
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 초점 조절 인수
        focal_weight = (1 - p_t) ** self.gamma

        # 표본별 음의 로그가능도 손실
        nll = F.nll_loss(log_probs, targets, reduction='none')

        # 초점 가중치 적용
        loss = focal_weight * nll

        # 클래스 가중치 적용
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

---

## 5. 시연: gamma의 효과

```python
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

# 난이도가 다른 예측
logits = torch.tensor([2.0, -1.5, -2.0, 3.0, -1.0])
targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
probs = torch.sigmoid(logits)

print("Per-sample comparison:")
bce_per_sample = bce_criterion(logits, targets)
focal_per_sample = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')(logits, targets)

for i in range(len(targets)):
    p = probs[i].item()
    p_t = p if targets[i] == 1 else 1 - p
    print(f"  Sample {i+1}: p_t={p_t:.3f}, "
          f"BCE={bce_per_sample[i]:.4f}, "
          f"Focal={focal_per_sample[i]:.4f}, "
          f"Ratio={focal_per_sample[i]/bce_per_sample[i]:.4f}")
```

**출력:**

```
Per-sample comparison:
  Sample 1: p_t=0.881, BCE=0.1269, Focal=0.0005, Ratio=0.0036
  Sample 2: p_t=0.818, BCE=0.2014, Focal=0.0050, Ratio=0.0250
  Sample 3: p_t=0.881, BCE=0.1269, Focal=0.0014, Ratio=0.0107
  Sample 4: p_t=0.953, BCE=0.0486, Focal=0.0000, Ratio=0.0006
  Sample 5: p_t=0.731, BCE=0.3133, Focal=0.0170, Ratio=0.0542
```

```python
# gamma가 손실 곡선에 미치는 영향
import matplotlib.pyplot as plt

p_t = torch.linspace(0.01, 0.99, 200)
gammas = [0, 0.5, 1, 2, 5]

plt.figure(figsize=(8, 5))
for gamma in gammas:
    focal = -((1 - p_t) ** gamma) * torch.log(p_t)
    plt.plot(p_t.numpy(), focal.numpy(), label=f'γ={gamma}')

plt.xlabel('p_t (probability of true class)')
plt.ylabel('Focal Loss')
plt.title('Focal Loss for Different γ Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. 하이퍼파라미터 고르기

**$\gamma$ (집중 매개변수).** 기본값 $\gamma = 2$에서 시작한다. 불균형이 더 심하거나 쉬운 예가 학습을 지배하면 $\gamma$을 올린다. 모델이 아무것도 배우지 못하면(교육 과정이 지나치게 과격할 수 있다) $\gamma$을 0 쪽으로 낮춘다.

**$\alpha$ (클래스 가중치).** 빈도 역수에 비례하는 정규화 가중치는 한 후보이지만 필수 정의는 아니다. 원 논문의 RetinaNet 설정은 양성 가중치 0.25와 감마 2를 사용한다. 이 값을 다른 데이터의 최적값으로 일반화하지 말고 손실·샘플링 가중치와 함께 검증한다.

---

## 7. 다른 전략과의 비교

| 전략 | 작동 원리 | 한계 |
|----------|-----------|------------|
| **과표집** (SMOTE) | 소수 클래스 표본을 늘린다 | 소수 클래스에 과적합한다 |
| **과소표집** | 다수 클래스 표본을 줄인다 | 쓸모 있을 수 있는 데이터를 버린다 |
| **클래스 가중치** ($\alpha_t$만) | 클래스마다 손실을 조정한다 | 쉬움과 어려움을 가리지 못한다 |
| **Focal Loss** | 클래스와 무관하게 쉬운 예의 비중을 낮춘다 | $\gamma$을 조율해야 한다 |

Focal Loss은 샘플링 전략과 서로 보완하며 함께 쓸 수 있다.

---

## 8. 핵심 정리

Focal Loss은 잘 분류된 예의 비중을 낮추는 인수 $(1 - p_t)^\gamma$으로 교차 엔트로피 손실을 조절하여 클래스 불균형을 다룬다. 집중 매개변수 $\gamma$이 비중을 낮추는 세기를 정하며 $\gamma = 2$이 표준 기본값이다. 단순한 클래스 가중과 달리 Focal Loss은 클래스 안에서 쉬운 예와 어려운 예를 가려 최적화의 힘을 가장 중요한 곳으로 돌린다. 구현은 수치 안정성을 위해 `BCEWithLogitsLoss` 위에 세우며 조절 인수만 더 계산하면 된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
$p_t$에 대한 도함수와 로짓에 대한 도함수를 구분해 Focal Loss를 미분하라.

</div>

??? success "연습문제 1 풀이"
    $\partial FL/\partial p_t=\alpha_t\gamma(1-p_t)^{\gamma-1}\log p_t-\alpha_t(1-p_t)^\gamma/p_t$이다. 여기에 $(2y-1)p_t(1-p_t)$를 곱하면 본문의 로짓 경사를 얻는다. 잘 맞힌 예에서 경사는 더 빠르게 작아지지만, 큰 손실의 원인이 레이블 오류라면 그 표본을 상대적으로 강조할 위험도 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Focal Loss을 쓰는 물체 검출에서 $\gamma$과 $\alpha$은 보통 어떤 값을 쓰는가?

</div>

??? success "연습문제 2 풀이"
    RetinaNet 논문(Lin 등, 2017)의 기본값은 $\gamma = 2$과 $\alpha = 0.25$이다. $\gamma = 2$이 좋은 균형을 준다. 쉬운 예($p_t > 0.5$)의 비중이 $4$배 넘게 낮아진다. $\alpha$은 양성 클래스와 음성 클래스의 균형을 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Focal Loss을 PyTorch로 구현하고, 불균형한 데이터셋에서 표준 BCE와 학습 곡선을 비교하여 보라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        loss = alpha * (1 - p_t)**gamma * bce
        return loss.mean()
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Focal Loss를 사용할 때 교차 엔트로피 기준과 비교해야 할 한계를 설명하라.

</div>

??? success "연습문제 4 풀이"
    어려운 예에는 중요한 소수 표본뿐 아니라 오표기·모호한 표본도 포함된다. 감마와 클래스 가중치는 검증해야 하고, 순위·문턱값 성능과 확률 보정도 함께 평가한다. 클래스 비율만으로 사용 여부나 보편적 우열을 정하지 않는다.

## 정리하며

Focal Loss는 $-\alpha_t(1-p_t)^\gamma\log p_t$로 쉬운 예의 손실을 낮춘다. 감마 0은 클래스 가중 CE에 대응한다. 경사는 조절 인수의 미분까지 포함하며, 실제 효과와 확률 보정은 검증에서 판단한다.
