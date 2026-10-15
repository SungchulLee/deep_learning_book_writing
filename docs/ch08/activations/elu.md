# ELU (지수 선형 단위)

ELU(exponential linear unit)는 양수 입력에서 항등 함수이고 음수 입력에서 지수적으로 포화한다. 음수 출력을 허용하면서 ReLU와 다른 경사를 제공한다. 원점의 매끄러움은 alpha 값에 따라 달라지며 학습 효과는 데이터와 구조에서 확인해야 한다.

## 1. 학습 목표

이 절을 마치면 다음을 이해하게 된다.

1. ELU의 수학적 정의와 도함수
2. 음수 입력에서의 매끄러운 포화가 이로운 이유
3. ReLU·Leaky ReLU와의 차이와 적용 조건
4. 계산 비용의 절충
5. PyTorch 구현 방식

---

## 2. 수학적 정의

$$\operatorname{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

여기서 $\alpha > 0$(기본값 1.0)은 음수 입력에서의 포화값을 조절한다.

### 도함수

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x < 0 \end{cases}$$

$x \leq 0$일 때 도함수를 $\operatorname{ELU}(x) + \alpha$으로도 쓸 수 있어, 순전파의 출력만으로 효율적으로 계산할 수 있음에 유의하라.

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \operatorname{ELU}(x) + \alpha & \text{if } x < 0 \end{cases}$$

### 거동

- **양수 영역** ($x > 0$): 항등함수이며 경사는 1이다 (ReLU와 같다)
- **음수 영역** ($x \leq 0$): $-\alpha$에서 포화하는 매끄러운 지수 곡선
- **원점에서** ($x = 0$): 모든 $\alpha>0$에서 연속이다. 좌우 도함수가 일치하는 $\alpha=1$일 때만 $C^1$이다.

---

## 3. 성질

| 성질 | 값 |
|----------|-------|
| **출력 범위** | $(-\alpha, +\infty)$ |
| **매끄러움** | $\alpha=1$일 때 $C^1$. 일반적으로 $C^2$는 아님 |
| **0을 중심에 가깝게** | 그렇다 (음수 쪽이 출력을 0으로 당긴다) |
| **부드러운 포화** | $x \to -\infty$일 때 $-\alpha$에서 |
| **음수 영역의 경사** | 유한한 음수에서 양수이지만 포화하면 매우 작아짐 |
| **단조성** | 그렇다 |
| **계산 비용** | 보통 (음수 입력에 $\exp$이 필요하다) |

---

## 4. ReLU보다 나은 점

### 1. 원점에서의 매끄러움

ReLU는 0에서 미분 가능하지 않다. ELU는 alpha=1이면 원점에서 1차 도함수가 연속이지만, 이 사실만으로 최적화가 더 쉽거나 좋은 해를 찾는다고 보장할 수는 없다.

$$\lim_{x \to 0^-} \operatorname{ELU}'(x) = \alpha \cdot e^0 = \alpha$$

$\alpha = 1$이면 이것이 양수 쪽의 경사(1)와 맞아떨어져 일계도함수가 연속이 된다.

### 2. 음수 값이 평균을 0으로 당긴다

ELU는 음수 입력에 음수 값을 내므로 ReLU와 다른 활성화 평균을 만든다. 평균이 0에 얼마나 가까운지는 입력 분포와 alpha에 따라 달라진다. Leaky ReLU도 음수 출력을 내며 크기는 음수 기울기와 입력에 달려 있다.

### 3. 포화를 통한 잡음 견고성

큰 음수 입력에서 출력의 변화가 작아져 작은 입력 잡음의 영향이 줄어들 수 있다. 같은 포화로 도함수도 작아지므로 학습이 느려질 수 있다. 전체 모델의 강건함은 이 성질 하나로 보장되지 않는다.

---

## 5. 단점

- **계산 비용**: 지수함수는 ReLU/Leaky ReLU의 단순한 비교보다 비싸다
- **정확히 0을 중심에 두지는 않는다**: ReLU보다는 0에 가깝지만 출력 분포가 완벽하게 대칭은 아니다

---

## 6. PyTorch 구현

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

**출력:**

```
Input:  [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: ['-0.8647', '-0.6321', '0.0000', '1.0000', '2.0000']
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

## 7. 시각화

```python
import torch
import matplotlib.pyplot as plt

# 함수 자체(왼쪽)와 그 도함수(오른쪽)를 나란히 그린다.
# 활성 함수를 볼 때는 도함수가 더 중요하다. 역전파에서 실제로
# 곱해지는 것이 도함수이므로, 그 값이 0이 되는 구간이 곧 학습이 멎는 구간이다.
x = torch.linspace(-4, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ── 왼쪽 그림: alpha를 달리한 ELU ───────────────────────────────────
for alpha in [0.5, 1.0, 2.0]:
    # ELU의 정의. 양수 쪽은 x 그대로, 음수 쪽은 alpha*(e^x - 1).
    # torch.where(조건, 참일 때, 거짓일 때)로 두 갈래를 한 번에 셈한다
    y = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    # alpha는 음수 쪽이 내려가 멈추는 바닥값(-alpha)을 정한다
    axes[0].plot(x.numpy(), y.numpy(), 
                 label=f'ELU (α={alpha})', linewidth=2)

# 견줄 기준으로 ReLU를 점선으로 겹쳐 그린다.
# 양수 쪽은 완전히 겹치고 음수 쪽에서만 갈린다는 점이 한눈에 보인다
axes[0].plot(x.numpy(), torch.relu(x).numpy(), 
             '--', label='ReLU', linewidth=1.5, alpha=0.5)
axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)   # y=0 기준선
axes[0].set_title('ELU with Different α Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── 오른쪽 그림: 도함수 ─────────────────────────────────────────────
# ELU의 도함수는 양수 쪽에서 1, 음수 쪽에서 e^x 이다(alpha=1일 때).
# e^x 는 x가 아무리 작아도 0보다 크므로 기울기가 완전히 죽지 않는다
elu_grad = torch.where(x > 0, torch.ones_like(x), torch.exp(x))

# ReLU의 도함수는 음수 쪽에서 딱 0이다. 이것이 "죽은 ReLU" 문제의 뿌리로,
# 한 번 음수 영역에 갇힌 유닛은 기울기가 0이라 되살아날 길이 없다
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

## 8. 신경망 예제

```python
import torch.nn as nn

class ConvNetELU(nn.Module):
    """더 매끄러운 학습 동역학을 위해 ELU를 쓰는 CNN."""
    def __init__(self, num_classes=10):
        super().__init__()
        # 입력은 32x32 RGB 이미지(CIFAR 크기)를 가정한다.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),        # 32x32 크기 유지, 채널 3 -> 64
            # inplace=True: 새 텐서를 만들지 않고 입력을 덮어써 메모리를 아낀다.
            # 이 값이 다른 곳에서 다시 쓰이지 않을 때만 안전하며,
            # 잔차 연결이 있는 구조에서는 켜서는 안 된다
            nn.ELU(alpha=1.0, inplace=True),
            # 여기서는 ELU 뒤에 BatchNorm을 두었다. 흔한 차례는 그 반대이며
            # (Conv -> BN -> 활성) 그쪽이 활성 함수에 들어가는 값의 자를
            # 맞춰 주므로 보통 더 안정적이다. ELU는 음수 쪽에서도 기울기가
            # 살아 있어 이 차례에서도 학습이 되지만, 새로 짤 때는 BN을 앞에 두는 편이 낫다
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                       # 32x32 -> 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),                       # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            # 앞의 두 번의 풀링으로 8x8이 되었고 채널이 128이므로 128*8*8
            nn.Linear(128 * 8 * 8, 256),
            nn.ELU(alpha=1.0, inplace=True),
            # 완전연결층은 매개변수가 가장 많이 몰린 곳이라 과적합의 주범이다.
            # 그래서 여기서만 드롭아웃 비율을 0.5로 세게 건다
            nn.Dropout(0.5),
            # 마지막 층에는 활성을 두지 않는다. CrossEntropyLoss가
            # 소프트맥스를 안에서 걸기 때문이다
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # (배치, 128, 8, 8) -> (배치, 8192). 완전연결층에 넣기 전에 편다
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

## 9. 비슷한 함수들과의 비교

| 항목 | ReLU | Leaky ReLU | ELU |
|--------|------|-----------|-----|
| **음수에서의 거동** | 0 | 선형 ($\alpha x$) | 지수적 포화 |
| **원점에서 미분 가능** | 아니오 | 기울기가 1이 아니면 아니오 | $\alpha=1$일 때 가능 |
| **0을 중심에 가깝게** | ❌ | ≈ | ✅ |
| **음수 도함수** | 0 | 음수 기울기 | 포화하면 매우 작아짐 |
| **음수 입력의 국소 반응** | 출력 0 | 선형 반응 | 포화하면서 반응이 작아짐 |
| **계산 비용** | 아주 낮음 | 아주 낮음 | 보통 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ELU의 좌우 도함수를 구하고 원점에서 미분 가능한 $\alpha$의 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    $x>0$에서 도함수는 1, $x<0$에서 $\alpha e^x=f(x)+\alpha$이다. 0에서 좌미분은 $\alpha$, 우미분은 1이므로 $\alpha=1$일 때만 미분 가능하다. 이때 1차 도함수는 연속이지만 좌우 2차 도함수는 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
세 활성화의 음수 영역 도함수와 경사 소실의 한계를 비교하라.

</div>

??? success "연습문제 2 풀이"
    ReLU는 음수에서 도함수가 0이다. 양의 음수 기울기를 가진 Leaky ReLU는 상수 도함수를 전달한다. ELU는 $\alpha e^x$라 유한한 음수에서는 양수지만 큰 음수에서 0에 가까워진다. 셋 모두 가중치 행렬과 다른 층의 영향까지 포함한 전체 경사 보존을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
ELU를 바닥부터 구현하고 `torch.nn.ELU`와 일치하는지 확인하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    def manual_elu(x, alpha=1.0):
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    assert torch.allclose(manual_elu(x), torch.nn.ELU()(x))
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
ELU의 음수 출력이 활성화 평균에 미치는 영향이 입력 분포에 따라 달라지는 이유를 설명하라.

</div>

??? success "연습문제 4 풀이"
    출력 평균은 입력의 양수·음수 비중과 크기, alpha에 따라 결정된다. 음수 출력을 허용한다고 평균이 항상 0이 되는 것은 아니다. 배치 정규화처럼 관측 통계량을 직접 빼는 연산과도 다르다.

## 정리하며

ELU는 음수 입력에서 $-\alpha$로 포화하고 양수 입력에서 항등 함수이다. $\alpha=1$이면 원점에서 1차 도함수가 연속이다. 음수 영역에서 도함수가 작아질 수 있고, 활성화 평균·속도·학습 효과는 입력과 구현에 따라 달라진다. ReLU 계열 초기화를 출발점으로 삼더라도 실제 활성화와 경사를 점검해야 한다.
