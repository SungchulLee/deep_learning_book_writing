# 깊이와 너비

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 넓고 얕은 구조와 깊고 좁은 구조 사이의 표현력 절충을 수치로 재기
    - 지수적 분리 결과 증명하기: 깊은 신경망으로 효율적으로 표현되는 함수가 얕은 신경망에서는 지수적인 너비를 요구한다
    - 깊이가 계층적인 특징 합성과 매개변수 효율을 어떻게 가능하게 하는지 분석하기
    - 실무적인 절충 이해하기: 학습 가능성, 최적화 지형, 경사의 흐름
    - 조건을 통제한 PyTorch 실험으로 구조들을 경험적으로 비교하기

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| MLP 구조 (§4.2.1) | 비교 대상이 되는 신경망을 정의한다 |
| 보편 근사 (§4.2.2) | 너비에 기댄 근사가 기준선이 된다 |
| 경사의 흐름 (§4.2.6) | 깊은 신경망은 경사 소실/폭발 문제를 겪는다 |

---

## 3. 개요

신경망의 **깊이**(층의 수)와 **너비**(층당 뉴런 수)는 용량을 결정하는 두 개의 기본 축이다. 보편 근사 정리(§4.2.2)는 너비만으로도 충분함을 보장한다. 그런데 그 대가는 무엇인가? 이 절은 깊이가 여러 함수 모임에 대해 지수적으로 더 효율적인 표현을 제공하면서도 너비에는 없는 최적화의 어려움을 들여온다는 수학적·실험적 증거를 살펴본다.

---

## 4. 정의와 매개변수 예산

### 깊이와 너비

층이 $l = 0, 1, \ldots, L$인 신경망에 대해 다음이 성립한다.

$$
\text{Depth} = L \quad \text{(층의 개수)}
$$

$$
\text{Width of layer } l = n^{[l]} \quad \text{(층 } l \text{의 뉴런 수)}
$$

**너비가 일정한** 신경망에서는 모든 은닉층의 너비가 $d$으로 같다. $l = 1, \ldots, L-1$에 대해 $n^{[l]} = d$이다.

### 전체 매개변수 개수

구조가 $(n^{[0]}, n^{[1]}, \ldots, n^{[L]})$인 완전 연결 신경망에서 다음이 성립한다.

$$
|\boldsymbol{\theta}| = \sum_{l=1}^{L} n^{[l]} \left(n^{[l-1]} + 1\right)
$$

입력 차원이 $n^{[0]}$, 은닉 너비가 $d$, 출력 차원이 $n^{[L]}$이고 층이 모두 $L$개인 균일 너비 신경망에서 다음이 성립한다.

$$
|\boldsymbol{\theta}| = d(n^{[0]} + 1) + (L-2) \cdot d(d+1) + n^{[L]}(d+1)
$$

눈여겨볼 점은 이것이다. 층을 하나 더할 때 깊이는 매개변수를 $O(d^2)$개 더하고, 뉴런을 하나 더할 때 너비는 $O(d \cdot L)$개를 더한다. 공정하게 비교하려면 전체 매개변수 예산을 고정한다.

---

## 5. 이론적 분석

### 표현력: 너비

보편 근사 정리는 **은닉층 하나짜리** 신경망이 임의의 연속 함수를 근사할 수 있다고 말한다. 다만 요구되는 너비가 지수적일 수 있다.

립시츠 상수가 $L$인 함수 $f: [0,1]^n \to \mathbb{R}$을 깊이 2인 신경망으로 $\varepsilon$의 정확도로 근사하려면 다음이 필요하다.

$$
N_{\text{width}} = O\!\left(\left(\frac{L}{\varepsilon}\right)^n\right)
$$

차원 $n$에 대한 이 **지수적 의존**이 얕은 신경망의 차원의 저주이다.

### 표현력: 깊이

얕은 신경망이 지수 개의 매개변수를 요구하는 함수를 깊은 신경망은 **다항** 개수로 표현할 수 있다.

!!! abstract "지수적 분리 (Telgarsky, 2016)"
    임의의 양의 정수 $k$에 대해 다음을 만족하는 함수 $f_k: [0,1] \to [0,1]$이 존재한다.
    
    1. 층이 $O(k)$개이고 층당 뉴런이 $O(1)$개인 ReLU 신경망으로 계산할 수 있다 (매개변수는 모두 $O(k)$개)
    2. 층이 $O(k^{1/3})$개인 ReLU 신경망으로는 뉴런이 $\Omega(2^{k^{1/3}})$개가 아닌 한 $\frac{1}{3}$의 정확도로도 근사할 수 없다

Telgarsky의 구성은 삼각파를 되풀이해 합성한 "톱니" 함수를 쓴다. 이 함수는 $2^k$번 진동하지만 너비 2인 층이 $O(k)$개만 있으면 된다. 얕은 신경망은 진동마다 뉴런이 하나씩 필요하다.

### 합성으로 이루어진 함수

현실의 많은 함수는 **합성 구조**를 갖는다.

$$
f(\mathbf{x}) = g_1 \circ g_2 \circ \cdots \circ g_k(\mathbf{x})
$$

그런 함수에서는 깊은 신경망이 구조적으로 유리하다.

**정석적인 예: 곱 함수** $f(x_1, \ldots, x_n) = \prod_{i=1}^n x_i$

깊은 신경망은 두 개씩 곱하는 이진 트리로 이를 계산한다.

| 성질 | 깊은 신경망 | 얕은 신경망 |
|----------|-------------|-----------------|
| 깊이 | $O(\log n)$ | $2$ |
| 너비 | $O(n)$ | $\Omega(2^n)$ |
| 매개변수 | $O(n \log n)$ | $O(n \cdot 2^n)$ |

두 수의 곱 $xy$은 다음을 통해 ReLU 뉴런으로 정확히 구현할 수 있다.

$$
xy = \frac{1}{4}\!\left[(x+y)^2 - (x-y)^2\right]
$$

여기서 $z^2$은 ReLU 경첩의 합으로 원하는 정밀도까지 근사된다.

### ReLU 신경망의 선형 영역

ReLU 신경망은 입력 공간을 **선형 영역**으로 나눈다. 각 영역은 신경망이 아핀으로 동작하는 볼록 다포체이다. 선형 영역의 최대 개수는 표현력을 재는 정확한 척도이다.

!!! abstract "선형 영역의 개수"
    입력 차원이 $n$이고 너비가 각각 $d$인 은닉층 $L$개를 갖는 ReLU 신경망은 최대
    
    $$
    R(n, d, L) \leq \left(\prod_{l=1}^{L-1} \left\lfloor \frac{d}{n} \right\rfloor^n \right) \cdot \sum_{j=0}^{n} \binom{d}{j}
    $$
    
    개의 선형 영역을 갖는 조각별 선형 함수를 계산한다. 눈여겨볼 점은 이것이 **깊이에 대해서는 지수적으로** 커지지만 **너비에 대해서는 다항적으로만** 커진다는 것이다.

전체 뉴런 수 $N = dL$을 고정했을 때, 좁은 층을 여럿 두는 편이 넓은 층 하나보다 **지수적으로 더 많은** 선형 영역을 만든다.

| 구조 | 선형 영역 수 (1차원) |
|-------------|-------------------|
| 깊이 1, 너비 $N$ | $N + 1$ |
| 깊이 $L$, 너비 $N/L$ | $O\!\left((N/L)^L\right)$ |

---

## 6. 실무적인 절충

### 깊이: 이점과 대가

**깊이의 이점:**

1. **매개변수 효율.** 더 적은 매개변수로 같은 표현력을 얻는다 (지수적 분리)
2. **계층적 특징.** 각 층이 앞 층 위에 쌓여 여러 수준의 추상이 가능해진다.
    - 앞쪽 층: 모서리, 질감, 단순한 무늬
    - 중간 층: 부분, 모양, 물체의 구성 요소
    - 뒤쪽 층: 물체, 장면, 추상적 개념
3. **합성적 귀납 편향.** 현실의 많은 함수가 합성 구조를 가지며 깊이가 이를 자연스럽게 포착한다

**깊이의 대가:**

1. **경사 소실.** 여러 층을 지나며 신호가 지수적으로 줄어든다 (§4.2.6)
2. **최적화의 어려움.** 깊을수록 손실 지형에 안장점과 좁은 골짜기가 많아진다
3. **학습의 불안정.** 신중한 초기화(He/Xavier)와 흔히 배치 정규화가 필요하다

### 너비: 이점과 대가

**너비의 이점:**

1. **더 쉬운 최적화.** 넓은 신경망일수록 손실 지형이 매끄럽다 (Li 등, 2018)
2. **게으른 학습.** 아주 넓은 신경망은 커널 방법(신경 접선 커널)으로 수렴하여 이론적 보장을 준다
3. **경사 소실이 없다.** 얕은 신경망에서는 경사가 곱셈적으로 줄어드는 층이 둘뿐이다

**너비의 대가:**

1. **매개변수의 비효율.** 같은 표현력에 지수적으로 더 많은 매개변수가 필요하다
2. **계층이 없다.** 한 수준의 특징 추출만 가능하며 합성적인 표현을 쌓을 수 없다
3. **메모리.** 이웃한 층 한 쌍마다 매개변수가 $O(d^2)$으로 늘어난다

### 최적화와 표현력의 절충

여기에는 근본적인 긴장이 있다.

$$
\underbrace{\text{More depth}}_{\text{more expressive}} \quad \longleftrightarrow \quad \underbrace{\text{Harder optimization}}_{\text{vanishing gradients, saddle points}}
$$

현대적인 구조는 **건너뛰기 연결**(잔차 신경망)로 이를 푼다. 건너뛰기 연결은 깊이의 표현력 이점을 지키면서 경사 소실 문제를 우회하는 직접적인 경사 경로를 제공한다.

---

## 7. PyTorch 실험

### 조건을 통제한 구조 비교

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def create_mlp(architecture: list[int]) -> nn.Sequential:
    """층 크기 목록으로 평범한 MLP를 만든다."""
    layers = []
    for i in range(len(architecture) - 1):
        layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
) -> tuple[float, float]:
    """모델을 학습시키고 (시험 정확도, 실행 시간)을 돌려준다."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    wall_time = time.time() - t0
    
    # 평가한다
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            preds = model(data).argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total, wall_time

# ── 데이터 ──
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256)

# ── 구조: 매개변수 예산을 대략 같게 맞춘다 ──
architectures = {
    'Wide-Shallow (1 hidden)':   [784, 1024, 10],
    'Medium (2 hidden)':         [784, 256, 128, 10],
    'Deep-Narrow (4 hidden)':    [784, 64, 64, 64, 64, 10],
    'Deep-Wide (3 hidden)':      [784, 256, 256, 256, 10],
}

print("Architecture Comparison on MNIST")
print("=" * 75)
print(f"{'Name':<28s} {'Params':>10s} {'Accuracy':>10s} {'Time (s)':>10s}")
print("-" * 75)

for name, arch in architectures.items():
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)
    acc, t = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    print(f"{name:<28s} {params:>10,d} {acc:>9.2f}% {t:>10.1f}")

print("=" * 75)
```

### 예산 고정: 깊이와 너비의 증가

```python
import matplotlib.pyplot as plt

def compute_uniform_width(input_dim, output_dim, depth, budget):
    """주어진 깊이에서 전체 매개변수가 예산에 가깝게 되는 너비 d를 찾는다."""
    # params = d*(input_dim+1) + (depth-2)*d*(d+1) + output_dim*(d+1)
    # depth >= 3에 대해 d의 이차방정식을 푼다
    if depth == 2:
        # params = input_dim * d + d + output_dim * d + output_dim
        d = (budget - output_dim) // (input_dim + 1 + output_dim)
        return max(16, d)
    a = depth - 2
    b = input_dim + 1 + (depth - 2) + output_dim
    c = output_dim - budget
    d = int((-b + (b**2 - 4*a*c)**0.5) / (2*a))
    return max(16, d)

param_budget = 100_000
depths = [2, 3, 4, 5, 6, 8, 10]
results = []

print(f"\nFixed budget ≈ {param_budget:,} parameters")
print(f"{'Depth':<8s} {'Width':<8s} {'Params':<12s} {'Accuracy':<10s}")
print("-" * 40)

for L in depths:
    d = compute_uniform_width(784, 10, L, param_budget)
    arch = [784] + [d] * (L - 1) + [10]
    
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)
    acc, _ = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    results.append((L, d, params, acc))
    print(f"{L:<8d} {d:<8d} {params:<12,d} {acc:<10.2f}%")

# 그래프 그리기
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r[0] for r in results], [r[3] for r in results], 'bo-', lw=2, ms=8)
ax.set_xlabel('Depth (number of layers)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Effect of Depth (fixed ≈{param_budget:,} params)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_vs_width_fixed_budget.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 잔차 연결을 갖춘 깊은 신경망

```python
class ResidualBlock(nn.Module):
    """2층 잔차 블록: x + F(x)."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        return torch.relu(x + self.block(x))

class ResidualMLP(nn.Module):
    """아주 깊은 신경망을 학습시키기 위해 잔차 연결을 쓰는 MLP."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)

# ── 평범한 깊은 신경망과 잔차 깊은 신경망 비교 ──
torch.manual_seed(42)
plain_deep = create_mlp([784] + [128] * 20 + [10])      # 은닉층 20개, 건너뛰기 없음
residual   = ResidualMLP(784, 128, 10, num_blocks=10)    # 블록 10개로 20개 층

print(f"Plain deep:    {count_params(plain_deep):>10,d} params")
print(f"Residual deep: {count_params(residual):>10,d} params")

plain_acc, _  = train_and_evaluate(plain_deep, train_loader, test_loader, epochs=5)
resid_acc, _  = train_and_evaluate(residual,   train_loader, test_loader, epochs=5)

print(f"\nPlain deep accuracy:    {plain_acc:.2f}%")
print(f"Residual deep accuracy: {resid_acc:.2f}%")
```

---

## 8. 최근의 통찰

### 너비와 신경 접선 커널

너비가 무한대로 가는 극한에서 신경망은 신경 접선 커널(NTK)이라 불리는 **커널 방법**으로 수렴한다. 이 "게으른 학습" 영역에서는 신경망의 매개변수가 초기값에서 거의 변하지 않고 학습의 움직임이 선형이 된다. 이는 이론적 보장(전역 수렴, 일반화 상계)을 주지만 유한 너비 신경망이 갖는 특징 학습의 이점을 내준다.

### 복권 가설

Frankle과 Carlin(2019)은 조밀한 신경망 안에 희소한 **부분 신경망**("당첨 복권")이 들어 있어, 같은 초기값에서 따로 학습시켜도 비슷한 정확도에 이른다는 것을 관찰했다. 이는 너비가 주로 좋은 초기값을 찾는 데 중요하며 최종적으로 유효한 신경망은 훨씬 작을 수 있음을 시사한다.

### 실무적인 권고

1. **검증된 구조로 시작하라** (예: 깊이에는 ResNet 방식, 표 형태 데이터에는 표준적인 MLP 너비)
2. MLP에는 **깔때기(피라미드) 모양**을 쓴다. 출력 쪽으로 갈수록 너비를 점차 줄인다
3. GPU 메모리 정렬을 위해 너비에 **2의 거듭제곱**(64, 128, 256, 512)을 쓴다
4. 대략 5층보다 깊은 신경망에는 **잔차 연결을 더한다**
5. **문제의 복잡도에 맞춰 깊이를 키우되** 경사가 충분히 흐르는지 확인한다

---

## 9. 핵심 정리

!!! success "요약"

    1. **깊이는 계층적인 특징 학습을 가능하게 하며**, 합성으로 이루어진 함수에서 너비보다 지수적으로 나은 매개변수 효율을 준다
    2. **너비는 용량**과 더 쉬운 최적화를 주지만, 수확이 점점 줄고 합성 구조가 없다
    3. **지수적 분리**가 증명되어 있다. 깊이 2에서는 너비 $\Omega(2^k)$을 요구하지만 깊이 $O(k)$에서는 매개변수 $O(k)$개면 되는 함수가 존재한다
    4. **ReLU의 선형 영역**은 깊이에 대해서는 지수적으로, 너비에 대해서는 다항적으로만 늘어난다
    5. **깊이는 최적화의 어려움**(경사 소실, 안장점)을 들여오며, 잔차 연결과 정규화가 이를 푼다
    6. **매개변수 예산이 고정되어 있을 때** 적당한 깊이와 적당한 너비가 대체로 극단보다 낫다
    7. **오늘날의 실무:** 잔차 연결, 배치 정규화, 신중한 초기화를 갖춘 깊은 신경망

---

## 연습문제

**연습문제 1.**
특정 함수를 표현할 때 깊은 신경망이 얕은 신경망보다 지수적으로 효율적인 이유를 설명하라.

??? success "연습문제 1 풀이"
    깊은 신경망은 특징을 계층적으로 합성할 수 있다. 뉴런이 각각 $d$개인 층 $L$개는 은닉층 하나로는 $O(d^L)$개의 뉴런이 필요한 함수를 표현할 수 있다. 예를 들어 $n$비트의 패리티 함수는 층이 하나면 은닉 단위가 $2^n$개 필요하지만 층이 $\log n$개면 $O(n)$개면 된다.

---

**연습문제 2.**
보편 근사 정리란 무엇이며 실무적인 한계는 무엇인가?

??? success "연습문제 2 풀이"
    이 정리는 너비가 충분한 은닉층 하나로 옹골집합 위의 임의의 연속 함수를 원하는 정밀도로 근사할 수 있다고 말한다. 한계는 다음과 같다. (1) 필요한 너비의 상계를 주지 않는다(지수적으로 클 수 있다), (2) 학습 가능성(경사 하강법으로 알맞은 가중치를 찾는 일)에 대해 아무 말이 없다, (3) 일반화를 다루지 않는다.

---

**연습문제 3.**
MNIST에서 넓고 얕은 신경망(1층, 1024개 단위)과 깊고 좁은 신경망(4층, 층마다 64개 단위)의 학습을 비교하라.

??? success "연습문제 3 풀이"
    매개변수는 넓은 쪽이 $784 \times 1024 + 1024 \times 10 \approx 813K$, 깊은 쪽이 $784 \times 64 + 3 \times 64^2 + 64 \times 10 \approx 63K$이다. 깊은 신경망은 대체로 약 10배 적은 매개변수로 비슷하거나 더 나은 정확도에 이르며, 이는 깊이의 효율을 보여준다.

---

**연습문제 4.**
'복권 가설'의 개념과 그것이 신경망 설계에 갖는 함의를 설명하라.

??? success "연습문제 4 풀이"
    복권 가설(Frankle & Carlin, 2019)은 조밀한 신경망 안에 따로 학습시켜도 온전한 정확도에 이르는 희소한 부분 신경망('당첨 복권')이 들어 있다고 말한다. 이는 최적화(알맞은 부분 신경망을 찾는 일)에는 과매개화된 신경망이 필요하지만 실효 용량은 훨씬 작음을 시사한다.

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、개요、정의와 매개변수 예산을 차례로 짚었다.

**참고 문헌**

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Montufar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. *NeurIPS*.
- Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *NeurIPS*.
- Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR*.
