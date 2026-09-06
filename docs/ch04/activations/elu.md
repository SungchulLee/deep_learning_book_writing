# ELU (지수 선형 단위)
## 개요

Clevert 등(2016)이 제안한 **ELU(지수 선형 단위)**는 양수 입력에서 포화하지 않는 ReLU의 성질과 음수 입력에서 매끄럽게 포화하는 지수 곡선을 결합한다. 이 설계는 평균 활성화를 0에 가깝게 밀어(경사의 흐름이 좋아진다), 모든 곳에서 매끄러운 함수를 제공하며(최적화 지형이 나아진다), 부드러운 포화를 통해 잡음에 대한 견고함을 준다.

## 학습 목표

이 절을 마치면 다음을 이해하게 된다.

1. ELU의 수학적 정의와 도함수
2. 음수 입력에서의 매끄러운 포화가 이로운 이유
3. ELU가 ReLU와 Leaky ReLU보다 나은 점
4. 계산 비용의 절충
5. PyTorch 구현 방식

---

## 수학적 정의

$$\operatorname{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

여기서 $\alpha > 0$(기본값 1.0)은 음수 입력에서의 포화값을 조절한다.

### 도함수

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x \leq 0 \end{cases}$$

$x \leq 0$일 때 도함수를 $\operatorname{ELU}(x) + \alpha$으로도 쓸 수 있어, 순전파의 출력만으로 효율적으로 계산할 수 있음에 유의하라.

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \operatorname{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

### 거동

- **양수 영역** ($x > 0$): 항등함수이며 경사는 1이다 (ReLU와 같다)
- **음수 영역** ($x \leq 0$): $-\alpha$에서 포화하는 매끄러운 지수 곡선
- **원점에서** ($x = 0$): 연속이며 일계도함수도 연속이다. 함수가 $C^1$급으로 매끄럽다

---

## 성질

| 성질 | 값 |
|----------|-------|
| **출력 범위** | $(-\alpha, +\infty)$ |
| **매끄러움** | 그렇다 (모든 곳에서 일계도함수가 연속) |
| **0을 중심에 가깝게** | 그렇다 (음수 쪽이 출력을 0으로 당긴다) |
| **부드러운 포화** | $x \to -\infty$일 때 $-\alpha$에서 |
| **죽은 뉴런** | 없다 (모든 $x$에서 경사 $> 0$) |
| **단조성** | 그렇다 |
| **계산 비용** | 보통 (음수 입력에 $\exp$이 필요하다) |

---

## ReLU보다 나은 점

### 1. 원점에서의 매끄러움

ReLU는 $x = 0$에서 도함수가 정의되지 않는 날카로운 모서리를 갖는다. ELU는 양수 영역과 음수 영역 사이를 매끄럽게 넘나들어 더 나은 최적화 지형을 제공한다.

$$\lim_{x \to 0^-} \operatorname{ELU}'(x) = \alpha \cdot e^0 = \alpha$$

$\alpha = 1$이면 이것이 양수 쪽의 경사(1)와 맞아떨어져 일계도함수가 연속이 된다.

### 2. 음수 값이 평균을 0으로 당긴다

(언제나 $\geq 0$인) ReLU나 (작은 음수 값만 내는) Leaky ReLU와 달리 ELU는 제법 큰 음수 출력을 내어 평균 활성화를 0에 더 가깝게 옮긴다. 이는 편향 이동 효과를 줄이고 수렴을 빠르게 할 수 있다.

### 3. 포화를 통한 잡음 견고성

아주 큰 음수 입력에서 ELU는 (Leaky ReLU처럼) 선형으로 자라는 대신 $-\alpha$에서 포화한다. 이 부드러운 포화가 음수 영역의 잡음과 이상치에 대해 신경망을 더 견고하게 만든다.

---

## 단점

- **계산 비용**: 지수함수는 ReLU/Leaky ReLU의 단순한 비교보다 비싸다
- **정확히 0을 중심에 두지는 않는다**: ReLU보다는 0에 가깝지만 출력 분포가 완벽하게 대칭은 아니다

---

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 함수형 API
y = F.elu(x, alpha=1.0)

# 모듈 API
elu = nn.ELU(alpha=1.0)
y = elu(x)

# 제자리 판본
elu_inplace = nn.ELU(alpha=1.0, inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
# 출력: ['-0.8647', '-0.6321', '0.0000', '1.0000', '2.0000']
```

### alpha 고르기

매개변수 $\alpha$은 포화의 바닥을 정한다.

| $\alpha$ | 포화하는 값 | 효과 |
|----------|---------------|--------|
| 0.5 | $-0.5$ | 음수 범위가 좁다 |
| 1.0 (기본값) | $-1.0$ | 표준적이고 균형 잡혀 있다 |
| 2.0 | $-2.0$ | 음수 범위가 넓다 |

실무에서는 대부분의 응용에 $\alpha = 1.0$이 잘 통한다.

---

## 시각화

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# alpha 값을 달리한 ELU
for alpha in [0.5, 1.0, 2.0]:
    y = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    axes[0].plot(x.numpy(), y.numpy(), 
                 label=f'ELU (α={alpha})', linewidth=2)

axes[0].plot(x.numpy(), torch.relu(x).numpy(), 
             '--', label='ReLU', linewidth=1.5, alpha=0.5)
axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[0].set_title('ELU with Different α Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ELU의 도함수
elu_grad = torch.where(x > 0, torch.ones_like(x), torch.exp(x))
relu_grad = (x > 0).float()

axes[1].plot(x.numpy(), elu_grad.numpy(), label="ELU' (α=1)", linewidth=2)
axes[1].plot(x.numpy(), relu_grad.numpy(), '--', 
             label="ReLU'", linewidth=1.5, alpha=0.5)
axes[1].set_title('Derivatives')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## 신경망 예제

```python
import torch.nn as nn

class ConvNetELU(nn.Module):
    """더 매끄러운 학습 동역학을 위해 ELU를 쓰는 CNN."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

## 비슷한 함수들과의 비교

| 항목 | ReLU | Leaky ReLU | ELU |
|--------|------|-----------|-----|
| **음수에서의 거동** | 0 | 선형 ($\alpha x$) | 지수적 포화 |
| **원점에서 매끄러움** | ❌ | ❌ | ✅ |
| **0을 중심에 가깝게** | ❌ | ≈ | ✅ |
| **죽은 뉴런** | 있다 | 없다 | 없다 |
| **잡음 견고성** | 낮음 | 낮음 | 높음 |
| **계산 비용** | 아주 낮음 | 아주 낮음 | 보통 |

---

## 요약

| 항목 | ELU |
|--------|-----|
| **공식** | $\max(x, 0) + \min(0, \alpha(e^x - 1))$ |
| **치역** | $(-\alpha, +\infty)$ |
| **죽은 뉴런** | ✅ 없음 |
| **매끄러움** | ✅ 그렇다 ($C^1$ 연속) |
| **0을 중심에 둠** | ✅ 대체로 그렇다 |
| **계산 비용** | ⚠️ 보통 (지수함수) |
| **알맞은 상황** | 매끄러운 경사와 잡음 견고성이 중요할 때 |
| **초기화** | He(Kaiming) 정규 |

!!! tip "실무적인 권고"
    ELU는 ReLU로 학습이 불안정하거나, GELU 같은 더 복잡한 활성화로 갈아타지 않고도 더 매끄러운 경사의 흐름이 필요할 때 좋은 선택이다. 지수함수의 계산 부담은 신경망의 나머지에 견주면 대개 무시할 만하다. 배치 정규화 없이 스스로 정규화하는 거동을 원한다면 SELU를 고려해 보라.

## 연습문제

**연습문제 1.**
ELU 함수와 그 도함수를 수학적으로 쓰라.

??? success "연습문제 1 풀이"
    ELU: $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$. 도함수: $f'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x & x \leq 0 \end{cases} = \begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \leq 0 \end{cases}$.

---

**연습문제 2.**
경사 소실과 죽은 뉴런의 관점에서 ELU, ReLU, Leaky ReLU를 비교하라.

??? success "연습문제 2 풀이"
    ReLU는 $x < 0$에서 경사가 0이다(죽은 뉴런). Leaky ReLU는 $x < 0$에서 작은 상수 경사를 준다(죽은 뉴런은 없지만 0에서 경사가 불연속이다). ELU는 매끄럽게 넘어가며 $x < 0$에서 경사가 $\alpha e^x$으로 매끄럽게 0에 다가간다(죽은 뉴런이 없고 경사가 매끄럽지만 exp 때문에 계산이 느리다).

---

**연습문제 3.**
ELU를 바닥부터 구현하고 `torch.nn.ELU`와 일치하는지 확인하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    def manual_elu(x, alpha=1.0):
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    assert torch.allclose(manual_elu(x), torch.nn.ELU()(x))
    ```

---

**연습문제 4.**
ELU의 음수 쪽 포화가 평균 활성화를 0에 더 가깝게 당길 수 있는 이유와, 그것이 학습에 왜 중요한지 설명하라.

??? success "연습문제 4 풀이"
    (평균이 0보다 큰) ReLU나 Leaky ReLU와 달리 ELU는 음수 값을 내어 활성화를 0 주위에 모은다. 0을 중심으로 한 활성화는 뒤따르는 층의 편향 이동을 줄여 수렴을 앞당긴다. 이는 배치 정규화의 효과와 비슷하지만 그만한 계산 부담이 없다.
