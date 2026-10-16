# 보편 근사 정리

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 보편 근사 정리를 고전적 형태, 일반화된 형태, ReLU 형태로 정확히 진술하기
    - 시그모이드 뉴럴 넷트웍과 ReLU 뉴럴 넷트웍이 임의의 연속 함수를 근사할 수 있는 이유를 기하학적으로 설명하기
    - 좋은 해의 존재(정리)와 찾아냄(학습)을 구별하기
    - 얕은 근사와 깊은 근사의 너비 복잡도, 그리고 차원의 저주 분석하기
    - PyTorch 실험으로 보편 근사를 경험적으로 보이기

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| MLP 구조 (9.2절) | 이 정리는 은닉 레이어가 하나인 뉴럴 넷트웍에 적용된다 |
| 연속성과 컴팩트성 | 정리의 가정은 컴팩트 정의역 위의 연속 함수를 다룬다 |
| 푸리에 해석 (선택) | 배런의 정리가 푸리에 적률을 쓴다 |

---

## 3. 개요

**보편 근사 정리**는 뉴럴 넷트웍 이론에서 가장 중요한 이론적 결과 중 하나이다. 은닉 레이어가 하나인 순방향 뉴럴 넷트웍도 너비가 충분하면 임의의 연속 함수를 원하는 정확도로 근사할 수 있음을 확립한다. 이는 뉴럴 넷트웍을 유연한 함수 근사기로 쓰는 이론적 토대가 되지만, 알맞은 파라메터를 어떻게 찾을지에 대해서는 아무 말도 하지 않는다.

---

## 4. 정리의 진술

### 고전적 형태 (Cybenko, 1989)

!!! abstract "정리 (보편 근사 — 너비 판)"
    $\sigma: \mathbb{R} \to \mathbb{R}$을 연속인 **시그모이드형** 함수라 하자. 즉 다음과 같다.
    
    $$
    \sigma(t) \to \begin{cases} 1 & t \to +\infty \\ 0 & t \to -\infty \end{cases}
    $$
    
    $I_n = [0,1]^n$을 $n$차원 단위 초입방체라 하고, $C(I_n)$을 상한 노름을 갖춘 $I_n$ 위의 연속 함수 공간이라 하자.
    
    그러면 임의의 $f \in C(I_n)$과 임의의 $\varepsilon > 0$에 대해, $N \in \mathbb{N}$과 실수 $v_i, b_i \in \mathbb{R}$, 벡터 $\mathbf{w}_i \in \mathbb{R}^n$($i = 1, \ldots, N$)이 존재하여 다음 함수가
    
    $$
    F(\mathbf{x}) = \sum_{i=1}^{N} v_i \, \sigma\!\left(\mathbf{w}_i^\top \mathbf{x} + b_i\right)
    $$
    
    $\|F - f\|_\infty = \sup_{\mathbf{x} \in I_n} |F(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$을 만족한다.

뉴럴 넷트웍의 언어로 말하면, 시그모이드 뉴런 $N$개를 갖고 (출력 활성화 없이) 출력 웨이트 $v_i$을 쓰는 은닉 레이어 하나짜리 뉴럴 넷트웍이 컴팩트 정의역 위의 임의의 연속 함수를 균등하게 근사할 수 있다는 것이다.

### 비다항 활성화로의 일반화

연속 엑티베이션 함수가 다항식이 아니면, 바이어스를 허용한 은닉 레이어 하나짜리 뉴럴 넷트웍의 선형 결합은 컴팩트 집합 위의 연속 함수를 균등하게 근사할 수 있다. 이 비다항 조건에 관한 일반적인 결과는 [Leshno 등(1993)](https://pinkus.net.technion.ac.il/files/2021/02/neural.pdf)에 따른다. Hornik의 연구는 유계 활성화와 여러 함수 공간에서의 근사 능력을 확장했다. 도함수 근사에는 별도의 매끄러움과 노름 조건이 필요하므로 여기의 연속 함수 정리만으로 결론 내릴 수 없다.

고정된 차수의 다항 활성화를 쓰고 깊이도 고정하면, 너비를 늘려도 출력 다항식의 차수에는 상한이 있다. 활성화의 차수가 $d$이고 활성화를 적용하는 레이어가 $L$개이면 차수는 최대 $d^L$이다. 따라서 일반적인 연속 함수 전체를 이 고정 깊이 모형으로 근사할 수는 없다. 깊이까지 늘리는 경우는 별도로 분석해야 한다.

### ReLU 형태 (현대적)

!!! abstract "정리 (ReLU를 쓰는 보편 근사)"
    $\sigma(x) = \max(0, x)$을 ReLU 활성화라 하자. 컴팩트 정의역 $K \subset \mathbb{R}^n$ 위의 임의의 연속 함수 $f: K \to \mathbb{R}$과 임의의 $\varepsilon > 0$에 대해, 은닉 레이어가 하나인 ReLU 뉴럴 넷트웍
    
    $$
    F(\mathbf{x}) = \sum_{i=1}^{N} v_i \max\!\left(0,\; \mathbf{w}_i^\top \mathbf{x} + b_i\right)
    $$
    
    이 존재하여 $\sup_{\mathbf{x} \in K} |F(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$을 만족한다.

ReLU는 유계도 매끄러운 함수도 아니지만 연속인 비다항 활성화이므로 위 정리를 적용할 수 있다. 1차원에서는 조각별 선형 보간을 ReLU 합으로 직접 표현할 수 있다. 다차원의 모든 조각별 선형 함수가 은닉 레이어 하나로 정확히 표현된다는 주장과는 구별해야 한다.

---

## 5. 기하학적 직관

### 시그모이드 뉴럴 넷트웍: 부드러운 계단의 합

각 은닉 뉴런 $\sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$은 **부드러운 계단 함수**를 만든다. $\mathbf{w}_i$ 방향을 따라 0에서 1로 매끄럽게 넘어가는 함수이다.

- 벡터 $\mathbf{w}_i$은 계단의 **방향**을 정한다
- 바이어스 $b_i$은 ($\mathbf{w}_i$을 따르는) **위치**를 정한다
- 크기 $\|\mathbf{w}_i\|$은 **가파름**을 조절한다 ($\|\mathbf{w}_i\|$이 클수록 계단이 날카롭다)

서로 반대 방향의 계단 둘로 **혹 함수**를 만들 수 있다.

$$
\text{bump}(x) \approx v \cdot \sigma(w x + b_1) - v \cdot \sigma(w x + b_2), \quad b_1 > b_2
$$

위치, 높이($v_i$), 너비가 서로 다른 이런 혹을 여럿 결합하면 임의의 연속 함수를 다시 만들어 낼 수 있다. 계단 함수가 적분을 근사하는 것과 닮았다.

### ReLU 뉴럴 넷트웍: 조각별 선형 근사

각 ReLU 뉴런은 "경첩"을 만든다. 꺾이는 점(매듭)이 하나뿐인 조각별 선형 함수이다.

$$
\max(0, w x + b) = \begin{cases} 0 & \text{if } wx + b \leq 0 \\ wx + b & \text{if } wx + b > 0 \end{cases}
$$

눈여겨볼 점은 다음과 같다.

- 1차원에서 ReLU 뉴런 $N$개는 최대 $N+1$개의 선형 영역을 갖는 함수를 만든다
- $n$차원에서 뉴런 $N$개는 (초평면 배열로) 최대 $O(N^n)$개의 선형 영역을 만든다
- 컴팩트 정의역 위의 임의의 연속 함수는 조각별 선형 함수로 균등하게 근사할 수 있다 (실해석의 표준적인 결과이다)

따라서 ReLU 뉴럴 넷트웍은 꺾이는 점과 그래디언트를 파라메터로 조정하는 **학습 가능한 조각별 선형 근사기**이다.

---

## 6. 증명의 구조

### 함수해석을 이용한 증명의 개요

핵심 착상은 함수해석의 논증으로 은닉 레이어 하나짜리 뉴럴 넷트웍의 집합이 $C(I_n)$에서 **조밀**함을 보이는 것이다.

**1단계.** 함수 모임을 정의한다.

$$
\mathcal{S} = \left\{ \sum_{i=1}^{N} v_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i) \;\Big|\; N \in \mathbb{N},\, v_i, b_i \in \mathbb{R},\, \mathbf{w}_i \in \mathbb{R}^n \right\}
$$

**2단계.** 모순을 위해 $\overline{\mathcal{S}} \neq C(I_n)$이라 가정하자. 그러면 $f \in C(I_n) \setminus \overline{\mathcal{S}}$이 존재한다.

**3단계.** 한-바나흐 정리에 의해 다음을 만족하는 유계 선형 범함수 $\mu \in C(I_n)^*$이 존재한다.

$$
\int_{I_n} g(\mathbf{x}) \, d\mu(\mathbf{x}) = 0 \quad \forall\, g \in \mathcal{S}, \qquad \text{but} \qquad \int_{I_n} f(\mathbf{x}) \, d\mu(\mathbf{x}) \neq 0
$$

리스 표현 정리에 의해 $\mu$은 부호 측도에 대응한다.

**4단계.** 모든 $\mathbf{w}, b$에 대해 $\int \sigma(\mathbf{w}^\top \mathbf{x} + b) \, d\mu = 0$임에서 시그모이드형이라는 성질을 활용해 $\mu = 0$임을 보이면 모순이 나온다.

결정적인 단계는 웨이트의 배율이 커질 때 $\sigma(t) \to \mathbf{1}_{t > 0}$이라는 사실을 쓴다. 그러면 $\mu$에 대한 적분이 모든 반공간에서 0이 되어야 하고, 따라서 $\mu = 0$이 강제된다.

### ReLU에 대한 구성적 접근

1차원 ReLU 뉴럴 넷트웍에서는 증명이 구성적이다.

**1단계.** 임의의 연속 함수 $f: [a,b] \to \mathbb{R}$과 $\varepsilon > 0$에 대해 $N$을 충분히 크게 잡고 $[a,b]$을 $N$개의 같은 소구간으로 나눈다.

**2단계.** 각 소구간 $[x_i, x_{i+1}]$에서 균등연속성에 의해 $f$의 선형 보간이 $|f(x) - L_i(x)| < \varepsilon$을 만족한다.

**3단계.** 조각별 선형 보간은 ReLU로 정확히 쓸 수 있다.

$$
F(x) = f(x_0) + s_0\max(0,x-x_0) + \sum_{i=1}^{N-1}(s_i-s_{i-1})\max(0,x-x_i)
$$

여기서 $s_i=[f(x_{i+1})-f(x_i)]/(x_{i+1}-x_i)$이며 $i=0,\ldots,N-1$이다. 첫 ReLU 항은 구간 $[x_0,x_N]$에서 초기 그래디언트를 만들고, 나머지는 각 내부 매듭에서 그래디언트 변화를 더한다. 상수항은 출력 바이어스로 둔다. 출력 편향이 없다면 상수 출력을 내는 뉴런 하나로 대신할 수 있다.

---

## 7. 너비 복잡도와 차원의 저주

### 일반 함수와 차원의 저주

립시츠 함수의 정의역 $[0,1]^n$을 격자로 나누면, 균등 오차를 줄이는 데 필요한 셀 수는 대략 해상도의 $n$제곱으로 증가한다. 이는 차원의 저주에 대한 직관이다. 격자의 셀 수를 곧바로 은닉 레이어 하나의 뉴런 수나 모든 함수에 필요한 하한으로 읽으면 안 된다. 뉴럴 넷트웍의 근사율은 함수족, 활성화, 노름, 웨이트 제약에 따라 달라진다.

### 깊이가 효율적인 함수족

특정 함수족에서는 깊은 뉴럴 넷트웍이 얕은 뉴럴 넷트웍보다 적은 뉴런으로 같은 근사 오차를 달성한다. [Telgarsky(2016)](https://proceedings.mlr.press/v49/telgarsky16.html)는 일정한 너비의 깊은 네트워크로 표현되는 함수가 더 얕은 네트워크에서는 지수적으로 많은 뉴런을 요구하는 예를 보였다. 이는 **그런 함수가 존재한다**는 결과이며, 임의의 과제에서 깊은 모델이 더 정확하다는 뜻은 아니다.

곱셈의 트리처럼 반복되는 부분 계산을 재사용하는 구조는 깊이의 장점을 이해하는 직관을 준다. 다만 유한한 ReLU 뉴럴 넷트웍은 조각별 아핀 함수이므로, 연속 영역에서 $xy$나 $x^2$을 정확히 나타낼 수 없다. 근사하려면 오차 허용치와 입력 범위를 정해야 하며, 필요한 크기에도 그 조건이 반영된다.

### 배런의 정리 (1993)

배런은 특정 함수 모임에 대해 차원과 무관한 오차 상계를 제시했다.

!!! abstract "배런의 정리"
    $f$의 **배런 노름**(1차 푸리에 적률)을 다음과 같이 정의한다.
    
    $$
    C_f = \int_{\mathbb{R}^n} \|\boldsymbol{\omega}\| \, |\hat{f}(\boldsymbol{\omega})| \, d\boldsymbol{\omega}
    $$
    
    여기서 $\hat{f}$은 푸리에 변환이다. $C_f < \infty$이면 뉴런 $N$개를 갖는 은닉 레이어 하나짜리 뉴럴 넷트웍이 다음을 달성한다.
    
    $$
    \inf_{F_N} \|F_N - f\|_{L^2(\mu)}^2 \leq C\frac{C_f^2}{N}
    $$
    
    **핵심 통찰:** 근사 속도 $O(1/N)$이 입력 차원 $n$과 **무관하다**.

위 근사율의 핵심은 제곱 오차가 $1/N$ 차수로 줄어드는 함수족이 있다는 점이다. 정확한 상수는 정의역의 반지름, 푸리에 변환의 관례, 사용하는 확률측도에 달려 있다. 이 절의 $L^2$ 오차는 컴팩트 정의역 위의 확률측도에 대한 오차로 읽는다. 단순히 매끄러운 모든 함수에 차원과 무관한 상수를 보장하는 것은 아니며, $C_f$ 자체가 차원에 따라 커질 수 있다.

---

## 8. 실무적 함의

### 이 정리가 보장하는 것

보편 근사 정리는 다음을 말해 준다.

- 뉴럴 넷트웍은 **어떤** 연속적인 입출력 관계든 모델링할 만큼 표현력이 있다
- **너비를 충분히 허용하면** 컴팩트 정의역 위의 연속 함수를 원하는 균등 오차 이내로 근사할 수 있다. 고정 크기 모델의 정확한 표현을 보장하지는 않는다
- **원리적으로는** 은닉 레이어 하나로 충분하다 (실무에서도 그렇다는 뜻은 아니다)

### 이 정리가 보장하지 않는 것

!!! warning "존재한다고 찾을 수 있는 것은 아니다"
    이 정리는 **순수한 존재 결과**이다. 좋은 웨이트가 존재함은 보장하지만 다음에 대해서는 아무 말이 없다.
    
    - 특정 문제와 정확도에 **뉴런이 몇 개 필요한지**
    - 경사 기반 학습이 그 웨이트를 **찾아낼 수 있는지** (최적화 지형)
    - 학습이 다항 시간 안에 **수렴할지**
    - 보지 않은 데이터로의 **일반화** (이 정리는 통계적 학습이 아니라 근사를 다룬다)
    - **계산 효율.** 요구되는 너비가 천문학적으로 클 수 있다

깊은 구조는 합성적인 관계를 효율적으로 표현할 수 있지만, 최적화가 어려워지거나 작은 데이터에서 불필요한 복잡도를 더할 수도 있다. 깊이와 너비의 선택은 표현력, 검증 성능, 학습 비용을 함께 비교해야 한다.

---

## 9. PyTorch로 보이기

### 복잡한 1차원 함수 근사하기

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    """복잡한 비선형 목표 함수: 진동과 가우스 포락선의 곱."""
    return np.sin(3 * x) * np.exp(-x**2) + 0.5 * np.cos(5 * x)

class ShallowApproximator(nn.Module):
    """은닉 레이어 하나 — 정리가 보장하는 구조."""
    def __init__(self, width: int, activation: nn.Module = nn.Tanh):
        super().__init__()
        # 정리가 말하는 최소 구조를 그대로 옮긴 것이다. 은닉 레이어는 딱 하나,
        # 늘리는 것은 깊이가 아니라 너비뿐이다. tanh를 기본으로 둔 까닭은
        # 고전적인 정리가 다항식이 아닌 유계 활성화를 요구하기 때문이다
        self.net = nn.Sequential(
            nn.Linear(1, width),
            activation(),
            nn.Linear(width, 1),
        )
    
    def forward(self, x):
        return self.net(x)

# ── 데이터 생성 ──
np.random.seed(42)
# 정리는 유계 폐구간에서만 근사를 보장한다. 여기서는 그 구간이
# [-3, 3]이며, 아래 그림도 같은 구간만 그린다. 이 밖에서 뉴럴 넷트웍이
# 무엇을 하는지는 정리가 아무것도 말해 주지 않는다
x_train = np.random.uniform(-3, 3, 1000).reshape(-1, 1).astype(np.float32)
# 목표에 잡음을 섞는다. 그래서 아래 MSE는 0으로 내려갈 수 없고
# 잡음의 분산인 0.05^2 = 0.0025 언저리가 바닥이다. 이 값에 닿았다면
# 못 배운 것이 아니라 배울 수 있는 데까지 배운 것이다
y_train = target_function(x_train) + np.random.normal(0, 0.05, x_train.shape).astype(np.float32)

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)

# ── 서로 다른 너비 비교 ──
widths = [5, 20, 100, 500]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, width in enumerate(widths):
    ax = axes[idx // 2, idx % 2]
    
    # 너비를 뺀 나머지 조건을 같게 맞춘다. 이 실험이 보려는 것은
    # "너비를 키우면 근사가 좋아지는가" 하나뿐이다
    torch.manual_seed(0)
    model = ShallowApproximator(width)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 학습
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()
    
    # 평가한다
    x_plot = torch.linspace(-3, 3, 500).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(x_plot).numpy()
    
    ax.scatter(x_train[:200], y_train[:200], alpha=0.2, s=5, color='gray', label='Train data')
    ax.plot(x_plot.numpy(), target_function(x_plot.numpy()), 'g-', lw=2, label='True $f(x)$')
    ax.plot(x_plot.numpy(), y_pred, 'r--', lw=2, label='NN approx')
    # 주의: 여기 찍히는 loss는 학습 집합에 대한 값이고, 그것도 마지막
    # optimizer.step() 앞에서 잰 값이다. 시험 오차가 아니므로 너비를
    # 키울수록 작아지는 것이 당연하고, 일반화가 나아졌다는 뜻은 아니다.
    # 정리가 보장하는 것도 "근사할 수 있다"이지 "학습으로 찾아진다"가
    # 아니라는 점을 함께 새겨 두라
    ax.set_title(f'Width $N = {width}$  |  Final MSE = {loss.item():.4f}')
    ax.legend(fontsize=8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.grid(True, alpha=0.3)

plt.suptitle('Universal Approximation: Effect of Width (1 hidden layer, Tanh)', fontsize=14)
plt.tight_layout()
plt.savefig('universal_approximation_width.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 깊이와 너비의 비교

```python
class DeepNarrowNet(nn.Module):
    """너비가 적당한 여러 은닉 레이어."""
    def __init__(self, width: int = 32, depth: int = 4):
        super().__init__()
        # 입력 1차원 -> width 로 시작한다
        layers = [nn.Linear(1, width), nn.ReLU()]
        # 가운데 레이어를 depth-1 번 더 쌓는다. 첫 레이어를 이미 만들었으므로 -1이다
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 1))   # 출력에는 활성을 두지 않는다(회귀)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 중첩 합성 목표 함수 — 자연히 깊이에 유리하다.
# sin을 세 번 겹쳐 놓았다. 레이어를 쌓는 것이 곧 함수를 합성하는 것이므로,
# 이런 목표는 깊은 망의 구조와 결이 맞는다. 얕은 망은 같은 함수를
# 흉내 내려면 훨씬 많은 조각(뉴런)을 이어 붙여야 한다.
# 보편 근사 정리는 은닉 레이어 하나로 "가능하다"고만 말할 뿐,
# 그것이 "값싸다"고는 말하지 않는다. 이 실험이 그 차이를 보인다.
def nested_target(x):
    return torch.sin(torch.sin(torch.sin(x * 3) * 2) * 4)

x = torch.linspace(-2, 2, 1000).reshape(-1, 1)   # (1000, 1) 꼴로 만든다
y = nested_target(x)

def train_model(model, x, y, epochs=5000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        # 여기서는 데이터 전체를 한 배치로 쓴다(표본이 1000개뿐이라 가능하다)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

torch.manual_seed(42)
# 두 모델의 파라메터 수를 비슷하게 맞추는 것이 이 비교의 핵심이다.
# 그래야 "깊이 덕분인지 크기 덕분인지"를 가릴 수 있다
wide_model = ShallowApproximator(500, activation=nn.ReLU)   # 은닉 레이어 1개, 너비 500
deep_model = DeepNarrowNet(width=32, depth=5)                # 은닉 레이어 5개, 너비 32

print(f"Wide-shallow params: {sum(p.numel() for p in wide_model.parameters()):,}")
print(f"Deep-narrow  params: {sum(p.numel() for p in deep_model.parameters()):,}")

wide_losses = train_model(wide_model, x, y)
deep_losses = train_model(deep_model, x, y)

# 파라메터가 비슷한데도 깊은 쪽의 MSE가 작으면, 그 이득은 크기가 아니라
# 깊이에서 온 것이다. 다만 이 결과는 목표 함수가 합성 꼴이라는 데
# 기대고 있으므로, 모든 문제에서 깊이가 이긴다는 뜻은 아니다
print(f"Wide-shallow final MSE: {wide_losses[-1]:.6f}")
print(f"Deep-narrow  final MSE: {deep_losses[-1]:.6f}")
```

**출력:**

```
Wide-shallow params: 1,501
Deep-narrow  params: 4,321
Wide-shallow final MSE: 0.031728
Deep-narrow  final MSE: 0.000256
```

---

## 10. 핵심 정리

!!! success "요약"

    1. **보편 근사**는 은닉 레이어 하나짜리 뉴럴 넷트웍이 컴팩트 정의역 위의 임의의 연속 함수를 원하는 정확도로 근사할 수 있음을 보장한다
    2. 시그모이드형 활성화의 고전적 결과는 Cybenko에, 연속 비다항 활성화에 대한 일반화는 Leshno 등에 따른다
    3. 이는 **순수한 존재 결과**이다. 뉴런이 몇 개 필요한지도, 학습이 알맞은 웨이트를 찾아낼지도 말해 주지 않는다
    4. 일반적인 고차원 함수의 근사는 어려울 수 있지만, 격자 셀 수와 뉴럴 넷트웍의 너비 하한은 구별해야 한다
    5. **깊은 뉴럴 넷트웍**은 합성으로 이루어진 함수에 대해 다항 개수의 파라메터만으로 같은 근사를 이룰 수 있다 (지수적 분리)
    6. **배런의 정리**는 (푸리에 적률이 유한한) 매끄러운 함수에 대해 차원과 무관한 $O(1/N)$의 속도를 준다
    7. 깊이와 너비의 효과는 함수의 구조와 학습 조건에 따라 달라진다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ReLU 활성화를 쓰는 뉴럴 넷트웍에 대해 보편 근사 정리를 정확히 진술하라.

</div>

??? success "연습문제 1 풀이"
    임의의 연속 함수 $f: [0,1]^d \to \mathbb{R}$과 $\epsilon > 0$에 대해, $\sup_{x \in [0,1]^d} |f(x) - g(x)| < \epsilon$을 만족하는 은닉 레이어 하나짜리 뉴럴 넷트웍 $g(x) = \sum_{i=1}^N c_i \max(0, w_i^\top x + b_i)$이 존재한다. 필요한 너비 $N$은 $f$, $d$, $\epsilon$에 따라 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
보편 근사 정리가 경사 하강법이 좋은 근사를 찾아낼 것임을 보장하지 못하는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    이 정리는 구성적이지 않은 존재 결과이다. 좋은 웨이트가 존재함은 보장하지만 (1) 경사 하강법이 그것을 찾을 수 있는지(로스 지형에 나쁜 국소 최솟값이 있을 수 있다), (2) 일반화에 표본이 몇 개 필요한지, (3) 학습의 계산 효율이 어떤지에 대해서는 아무 말이 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
은닉 뉴런 두 개로 함수 $f(x) = |x|$을 정확히 표현하는 ReLU 뉴럴 넷트웍을 구성하라.

</div>

??? success "연습문제 3 풀이"
    $|x| = \max(x, 0) + \max(-x, 0) = \text{ReLU}(x) + \text{ReLU}(-x)$이다. 뉴럴 넷트웍: 은닉 레이어는 $w_1 = 1, b_1 = 0$과 $w_2 = -1, b_2 = 0$, 출력 레이어는 $c_1 = c_2 = 1$이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
1차원 입력과 너비 $d$인 은닉 레이어 $L$개를 가진 ReLU 뉴럴 넷트웍의 선형 구간 수에 대한 단순 상한 $(d+1)^L$을 유도하라.

</div>

??? success "연습문제 4 풀이"
    기존 선형 구간 안에서 다음 레이어의 각 활성화 전 값은 입력의 아핀 함수이다. 뉴런 하나는 그 구간을 최대 한 번 나누므로, 뉴런 $d$개가 만들 수 있는 부분 구간은 최대 $d+1$개이다. 이를 레이어마다 반복하면 $(d+1)^L$이라는 상한을 얻는다. 이는 1차원에 대한 단순 상한이며 모든 구조에서 달성되는 값이나 다차원 공식은 아니다.

## 정리하며

보편 근사 정리는 적절한 활성화와 충분한 너비를 허용할 때 좋은 근사가 존재함을 보인다. 학습 알고리즘이 그 해를 찾는지, 데이터 밖에서도 정확한지, 필요한 계산량이 감당 가능한지는 별도의 문제이다. ReLU의 1차원 보간 구성과 일반적인 근사 정리, 특정 함수족의 깊이 이점을 구분해서 이해한다.

**참고 문헌**

- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
- Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Transactions on Information Theory*, 39(3), 930–945.
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
