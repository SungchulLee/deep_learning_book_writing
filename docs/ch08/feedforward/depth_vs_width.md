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
| MLP 구조 (9.2절) | 비교 대상이 되는 신경망을 정의한다 |
| 보편 근사 (9.2절) | 너비에 기댄 근사가 기준선이 된다 |
| 경사의 흐름 (9.2절) | 깊은 신경망은 경사 소실/폭발 문제를 겪는다 |

---

## 3. 개요

신경망의 **깊이**(층의 수)와 **너비**(층당 뉴런 수)는 용량을 결정하는 두 개의 기본 축이다. 보편 근사 정리(9.2절)는 너비만으로도 충분함을 보장한다. 그런데 그 대가는 무엇인가? 이 절은 깊이가 여러 함수 모임에 대해 지수적으로 더 효율적인 표현을 제공하면서도 너비에는 없는 최적화의 어려움을 들여온다는 수학적·실험적 증거를 살펴본다.

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

### 너비에 따른 보편 근사와 깊이의 효율

적절한 활성화를 쓰고 너비를 충분히 늘리면 은닉층 하나로 컴팩트 집합 위의 연속 함수를 근사할 수 있다. 이 존재 정리가 모든 함수의 필요한 너비를 정해 주는 것은 아니다. 깊이에 따른 효율을 비교하려면 입력 영역, 함수족, 오차 노름, 허용 정확도를 명시해야 한다.

### 지수적 분리의 의미

[Telgarsky(2016)](https://proceedings.mlr.press/v49/telgarsky16.html)는 너비가 일정하고 깊이가 $\Theta(k^3)$인 네트워크로 표현되지만, 깊이를 $O(k)$로 제한하면 같은 기준으로 근사하는 데 $\Omega(2^k)$개의 노드가 필요한 함수족을 제시했다. 반복 합성으로 진동이 늘어나는 함수가 그 직관을 보여 준다. 이 결과는 모든 실무 데이터에서 깊이가 유리하다는 보장이 아니다.

### 합성 구조와 곱셈 근사

함수가 여러 간단한 연산의 합성으로 주어지면 중간 결과를 재사용하는 구조가 효율적일 수 있다. 예를 들어 여러 입력의 곱은 두 수씩 곱하는 트리로 구성할 수 있다. 그러나 ReLU 신경망의 출력은 조각별 아핀 함수이므로 연속 영역의 곱셈을 유한한 크기로 정확히 구현하지는 못한다.

$$
xy=\frac{(x+y)^2-(x-y)^2}{4}
$$

이 항등식과 제곱 함수의 근사를 결합하면 곱셈을 근사할 수 있다. 이때 모델 크기와 깊이는 입력 범위와 허용 오차에도 의존한다. 정확도 조건을 빼고 얕은 곱셈 모델이 무조건 지수 크기를 요구한다고 결론 내릴 수는 없다.

### ReLU 신경망의 선형 영역

활성 패턴을 고정하면 ReLU 신경망은 입력의 아핀 함수가 된다. 활성 패턴별 영역은 볼록 다포체로 기술할 수 있다. 서로 이웃한 영역에서 같은 아핀 식이 나올 수도 있으므로 패턴 수와 서로 다른 출력 영역 수는 구별한다.

입력 차원이 $n$, 각 은닉층 너비가 $d\geq n$, 은닉층 수가 $L$일 때, 다음 식은 특정 가중치 구성으로 **달성 가능한 영역 수의 하한**이다.

$$
R_{\max}\geq
\left(\prod_{l=1}^{L-1}\left\lfloor\frac{d}{n}\right\rfloor^n\right)
\sum_{j=0}^{n}\binom{d}{j}
$$

이는 모든 가중치가 만드는 영역 수에 대한 상한이 아니다. [Montúfar 등(2014)](https://arxiv.org/abs/1402.1869)의 구성은 깊이에 따라 영역 수를 크게 늘릴 수 있음을 보여 준다. 영역의 개수는 표현력의 한 지표이며, 근사 오차나 일반화 성능을 단독으로 결정하지 않는다.

1차원에서는 은닉층 하나의 뉴런 $N$개가 만드는 구간 수는 최대 $N+1$이다. 여러 층에서는 각 기존 구간을 다시 나눌 수 있어 구간 수가 깊이에 따라 빠르게 증가할 수 있지만, 모든 구조와 매개변수에서 최대치가 달성되는 것은 아니다.

## 6. 실무적인 절충

### 깊이의 효과

깊이를 늘리면 중간 표현을 여러 단계로 변환할 수 있다. 합성 구조가 과제에 맞을 때 표현 효율이 좋아질 수 있지만, 야코비안의 곱 때문에 경사가 작아지거나 커질 가능성도 있다. 초기화, 활성화, 잔차 연결과 정규화 층은 이 문제를 완화하는 방법이다. 어느 방법도 최적화 성공을 보장하지 않는다.

### 너비의 효과

너비를 늘리면 표현할 수 있는 특징의 수와 매개변수 수가 늘어난다. 특정 조건에서는 과매개화가 최적화를 쉽게 하지만, 넓을수록 손실 지형이 항상 매끄럽거나 일반화가 항상 좋아지는 것은 아니다. 얕은 모델에서도 포화 활성화나 부적절한 초기화로 경사 소실이 생길 수 있다.

### 비교할 조건

매개변수 수를 맞추는 것은 유용한 비교 방식이지만 계산량과 학습 시간이 같아지지는 않는다. 같은 데이터 분할, 전처리, 학습 예산과 탐색 예산을 정하고, 여러 초기값에서 검증 성능과 변동을 확인한다. 최종 테스트셋은 구조 선택에 반복 사용하지 않는다.

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
        # 마지막 층 뒤에만 ReLU를 빼는 조건이다. 출력이 로짓이어야
        # CrossEntropyLoss에 넣을 수 있는데, ReLU를 걸면 음수 로짓이
        # 모두 0으로 눌려 버린다
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
    
    # 주의: GPU에서는 이 시간이 부정확하다. CUDA 연산은 비동기라
    # 파이썬이 먼저 앞서 나가므로, 제대로 재려면 t0 앞뒤로
    # torch.cuda.synchronize()를 불러야 한다
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            # MNIST는 (1, 28, 28)로 오므로 784짜리 벡터로 편다.
            # MLP에는 공간 구조를 다룰 재간이 없어 그냥 펴서 넣는다
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

# ── 구조: 흔히 쓰는 네 가지 모양을 견준다 ──
# 주의: 매개변수 수는 맞추어져 있지 않다. 아래에서 찍히는 값은 각각
# 약 814k, 235k, 63k, 335k로 13배 가까이 벌어진다. 첫 층이
# 784 x 은닉폭이라 전체를 좌우하기 때문이다. 그러므로 이 표에서
# "깊이가 정확도를 얼마나 올리는가"를 곧바로 읽어 낼 수는 없다.
# 정확도 차이에는 깊이의 효과와 크기의 효과가 섞여 있다.
# 예산을 맞추어 깊이만 견주는 실험은 바로 다음 절
# "예산 고정: 깊이와 너비의 증가"에서 다룬다
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
    # 모델을 만들기 직전마다 씨앗을 다시 심는다. 이렇게 해야 초기화의
    # 운이 아니라 구조의 차이를 보는 것이 된다
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)
    acc, t = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    print(f"{name:<28s} {params:>10,d} {acc:>9.2f}% {t:>10.1f}")

print("=" * 75)
```

**출력:**

```
Architecture Comparison on MNIST
===========================================================================
Name                             Params   Accuracy   Time (s)
---------------------------------------------------------------------------
Wide-Shallow (1 hidden)         814,090     97.99%       30.8
Medium (2 hidden)               235,146     97.71%       21.1
Deep-Narrow (4 hidden)           63,370     96.34%       14.4
Deep-Wide (3 hidden)            335,114     98.03%       15.5
===========================================================================
```

### 예산 고정: 깊이와 너비의 증가

```python
import matplotlib.pyplot as plt

def compute_uniform_width(input_dim, output_dim, depth, budget):
    """주어진 깊이에서 전체 매개변수가 예산에 가깝게 되는 너비 d를 찾는다."""
    # 깊이를 바꾸면서 매개변수 수는 그대로 두려는 것이다. 그래야
    # 성능 차이를 "깊이 덕분"이라고 말할 수 있다. 그냥 층만 늘리면
    # 매개변수도 함께 늘어 무엇이 이겼는지 알 수 없다.
    #
    # params = d*(input_dim+1) + (depth-2)*d*(d+1) + output_dim*(d+1)
    # 가운데 항에 d^2 이 있으므로 d에 대한 이차방정식이 된다
    if depth == 2:
        # 은닉층이 하나뿐이면 d^2 항이 없어 일차식으로 풀린다
        # params = input_dim * d + d + output_dim * d + output_dim
        d = (budget - output_dim) // (input_dim + 1 + output_dim)
        return max(16, d)

    # 이차방정식 a*d^2 + b*d + c = 0 의 계수
    a = depth - 2
    b = input_dim + 1 + (depth - 2) + output_dim
    c = output_dim - budget
    # 근의 공식에서 + 쪽만 쓴다. - 쪽은 음수가 되어 너비로 쓸 수 없다
    d = int((-b + (b**2 - 4*a*c)**0.5) / (2*a))

    # 너무 좁아지면 예산과 무관하게 성능이 무너지므로 16에서 끊는다.
    # 곧 깊이가 아주 클 때는 예산을 맞추지 못하고 넘길 수 있다
    return max(16, d)

param_budget = 100_000
depths = [2, 3, 4, 5, 6, 8, 10]
results = []

print(f"\nFixed budget ≈ {param_budget:,} parameters")
print(f"{'Depth':<8s} {'Width':<8s} {'Params':<12s} {'Accuracy':<10s}")
print("-" * 40)

for L in depths:
    d = compute_uniform_width(784, 10, L, param_budget)
    arch = [784] + [d] * (L - 1) + [10]   # 은닉층을 모두 같은 너비로

    # 깊이마다 씨앗을 같은 값으로 되돌린다. 초기화의 운이 아니라
    # 깊이가 성능을 갈랐다고 말하려면 이 줄이 있어야 한다
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)   # 예산에 실제로 얼마나 가까운지 함께 찍는다
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

**출력:**

```
Fixed budget ≈ 100,000 parameters
Depth    Width    Params       Accuracy  
----------------------------------------
2        125      99,385       97.60     %
3        110      99,670       97.57     %
4        100      99,710       97.70     %
5        92       98,818       97.24     %
6        87       99,799       96.86     %
8        78       98,992       96.80     %
10       72       99,298       96.54     %
```

### 잔차 연결을 갖춘 깊은 신경망

```python
class ResidualBlock(nn.Module):
    """2층 잔차 블록: x + F(x)."""
    def __init__(self, dim: int):
        super().__init__()
        # 마지막에 ReLU를 두지 않는 것이 요점이다. 활성은 더한 뒤에 건다.
        # 블록 안에서 걸어 버리면 F(x)가 음수를 낼 수 없어 잔차가
        # 값을 빼는 쪽으로 고칠 방법이 사라진다
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        # x + F(x) 가 잔차 연결이다. 덧셈이므로 역전파에서 기울기가
        # 1의 몫으로 그대로 지나간다. 그래서 층을 아무리 쌓아도
        # 기울기가 층 수만큼 곱해지며 사그라들지 않는다.
        # 입력과 출력의 차원이 같아야 더할 수 있어 dim -> dim으로 잡았다
        return torch.relu(x + self.block(x))

class ResidualMLP(nn.Module):
    """아주 깊은 신경망을 학습시키기 위해 잔차 연결을 쓰는 MLP."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        # 784 -> 128 로 먼저 맞춘다. 이 사영이 있어야 뒤의 블록들이
        # 모두 같은 차원에서 더하기를 할 수 있다
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
# 블록 하나가 선형층 2개이므로 블록 10개면 층 20개다. 곧 두 모델의
# 깊이가 같고 다른 것은 건너뛰기 연결이 있느냐뿐이다.
# 매개변수 수를 함께 찍는 까닭도 그 점을 확인하기 위해서다
torch.manual_seed(42)
plain_deep = create_mlp([784] + [128] * 20 + [10])      # 은닉층 20개, 건너뛰기 없음
residual   = ResidualMLP(784, 128, 10, num_blocks=10)    # 블록 10개로 20개 층

# 잔차 쪽이 조금 더 많다. 배치 정규화의 감마와 베타가 더해지기 때문이며,
# 건너뛰기 연결 자체는 매개변수를 하나도 쓰지 않는다
print(f"Plain deep:    {count_params(plain_deep):>10,d} params")
print(f"Residual deep: {count_params(residual):>10,d} params")

plain_acc, _  = train_and_evaluate(plain_deep, train_loader, test_loader, epochs=5)
resid_acc, _  = train_and_evaluate(residual,   train_loader, test_loader, epochs=5)

# 층 20개짜리 평범한 MLP는 기울기가 입력 쪽까지 닿지 못해 학습이
# 잘 되지 않는다. 깊이가 같은데도 잔차 쪽이 크게 앞선다면,
# 그 차이는 표현력이 아니라 "학습이 되느냐"에서 온 것이다
print(f"\nPlain deep accuracy:    {plain_acc:.2f}%")
print(f"Residual deep accuracy: {resid_acc:.2f}%")
```

**출력:**

```
Plain deep:       415,498 params
Residual deep:    437,130 params

Plain deep accuracy:    11.35%
Residual deep accuracy: 97.49%
```

---

## 8. 최근의 통찰

### 너비와 신경 접선 커널

특정 초기화와 매개변수화, 학습률 스케일에서 너비를 무한대로 보내면 신경 접선 커널(NTK)로 학습을 기술할 수 있다. 이런 극한에서는 특징의 변화가 제한되는 학습 영역이 나타난다. 모든 넓은 유한 신경망이 같은 방식으로 동작하거나 자동으로 전역 수렴·좋은 일반화를 얻는다는 뜻은 아니다.

### 복권 가설

[Frankle과 Carbin(2019)](https://arxiv.org/abs/1803.03635)은 실험한 조밀한 네트워크 안에, 원래 초기값을 유지해 별도로 학습하면 비슷한 성능에 이르는 희소 부분 네트워크가 있음을 관찰했다. 이는 초기값과 가지치기의 역할에 대한 가설이며, 모든 과제에 적용되는 정리나 과매개화가 반드시 필요하다는 증명은 아니다. 작은 모델을 처음부터 찾는 비용도 고려해야 한다.

### 구조를 선택할 때

검증된 작은 기준 모델부터 시작하고 과소적합 여부와 계산 비용을 확인한다. 너비를 점차 줄이는 MLP나 잔차 연결을 후보로 삼을 수 있지만, 특정 층 수나 2의 거듭제곱 너비가 모든 장치와 과제에 최적이라는 규칙은 없다. 선택은 검증 성능과 실제 자원 사용량으로 판단한다.

## 9. 핵심 정리

깊이와 너비는 모델의 표현 방식과 비용을 함께 바꾼다. 지수적 분리는 특정 함수족의 표현 효율에 관한 결과이고, 선형 영역의 하한은 달성 가능한 복잡도에 관한 결과이다. 두 결과 모두 실험에서의 정확도나 최적화 성공을 직접 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
깊이에 따른 지수적 분리의 결과를 설명하고, 이를 모든 실무 과제의 성능 비교로 일반화할 수 없는 이유를 밝혀라.

</div>

??? success "연습문제 1 풀이"
    지수적 분리는 특정 함수족과 정한 근사 기준에서, 깊은 구조가 적은 뉴런으로 달성하는 오차를 더 얕은 구조가 달성하려면 지수적으로 많은 뉴런을 요구한다는 존재 결과이다. 실제 목표 함수가 그 함수족에 속하는지, 학습이 적절한 계수를 찾는지, 제한된 표본으로 일반화하는지는 별도로 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
보편 근사 정리란 무엇이며 실무적인 한계는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    이 정리는 너비가 충분한 은닉층 하나로 옹골집합 위의 임의의 연속 함수를 원하는 정밀도로 근사할 수 있다고 말한다. 한계는 다음과 같다. (1) 필요한 너비의 상계를 주지 않는다(지수적으로 클 수 있다), (2) 학습 가능성(경사 하강법으로 알맞은 가중치를 찾는 일)에 대해 아무 말이 없다, (3) 일반화를 다루지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MNIST에서 은닉층 하나의 너비가 1024인 MLP와 은닉층 네 개의 너비가 각각 64인 MLP를 비교한다고 하자. 편향을 포함한 매개변수 수를 구하고 비교 실험의 조건을 제안하라.

</div>

??? success "연습문제 3 풀이"
    구조 $784\to1024\to10$의 매개변수 수는 $(784+1)1024+(1024+1)10=814090$개이다. $784\to64\to64\to64\to64\to10$은 $(784+1)64+3(64+1)64+(64+1)10=63370$개이다. 약 12.8배 차이가 나므로 같은 매개변수 예산의 비교는 아니다. 데이터 분할과 학습 조건을 맞추고 검증 정확도, 학습 시간, 여러 초기값의 변동을 함께 비교해야 한다. 크기만으로 어느 모델이 더 정확할지 결정할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
복권 가설이 말하는 부분 네트워크와 초기값의 역할을 설명하라. 작은 모델의 존재와 효율적으로 찾을 수 있다는 주장은 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    원래 조밀한 모델의 연결 일부와 그 초기값을 유지한 부분 네트워크가 비슷한 정확도로 학습될 수 있다는 가설이다. 관측된 부분 네트워크는 가지치기와 반복 학습으로 찾아낸 것이므로, 처음부터 같은 구조를 저렴하게 알아낼 수 있다는 결론은 아니다. 모든 구조와 데이터에 보장되는 정리도 아니다.

## 정리하며

깊이는 합성을, 너비는 한 층에서 다룰 특징의 수를 늘린다. 특정 함수족에서는 깊이의 표현 효율이 크지만, 실제 구조 선택에는 학습 가능성, 데이터 크기, 검증 성능과 비용이 함께 중요하다. 같은 매개변수 수가 같은 계산 비용을 뜻하지 않는다는 점도 비교에 반영한다.

**참고 문헌**

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Montufar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. *NeurIPS*.
- Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *NeurIPS*.
- Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR*.
