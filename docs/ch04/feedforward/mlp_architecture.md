# MLP 구조

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 뉴런 하나에서 완전한 신경망까지 다층 퍼셉트론(MLP)의 수학적 구조를 정의하기
    - 입력층, 은닉층, 출력층의 역할과 각각의 설계 고려 사항을 구별하기
    - 임의의 구조에 대한 매개변수 개수를 유도하고 그 증가 양상 이해하기
    - `nn.Module`과 `nn.Sequential` 둘 다로 설정 가능한 MLP를 PyTorch에서 구현하기
    - 분류와 회귀 과제에 구조 설계 원리를 적용하기

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| 선형대수 (행렬 곱) | 층의 계산이 아핀 변환이다 |
| 활성화 함수 (4.1절) | 각 선형 층 뒤에 비선형 활성화가 적용된다 |
| PyTorch 텐서 기초 | 구현 연습에 PyTorch를 쓴다 |

---

## 3. 개요

순방향 신경망이라고도 하는 **다층 퍼셉트론(MLP)**은 딥러닝의 토대가 되는 구조이다. MLP는 입력층 하나, 은닉층 하나 이상, 출력층 하나로 이루어지며 각 층이 다음 층과 **완전히 연결**(밀집)되어 있다. 단순한데도 MLP는 현대의 모든 신경망 구조를 이해하는 구성 요소 노릇을 한다.

"순방향"이라는 말은 정보가 순환이나 되먹임 없이 입력에서 출력으로 한 방향으로만 흐른다는 점을 강조한다. 이 점이 MLP를 순환 신경망(7장)과 구별해 준다.

---

## 4. 수학적 정식화

### 뉴런 하나

뉴런 하나는 입력의 가중합을 구한 뒤 비선형 활성화를 적용한다.

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b
$$

$$
a = \sigma(z)
$$

여기서 각 기호는 다음과 같다.

- $\mathbf{x} \in \mathbb{R}^n$은 입력 벡터이다
- $\mathbf{w} \in \mathbb{R}^n$은 가중치 벡터이다
- $b \in \mathbb{R}$은 편향 항이다
- $\sigma(\cdot)$은 활성화 함수이다
- $z \in \mathbb{R}$은 **활성화 전 값**(선형결합)이다
- $a \in \mathbb{R}$은 **활성화 후** 출력이다

뉴런은 기하학적으로도 이해할 수 있다. $\mathbf{w}^\top \mathbf{x} + b = 0$은 $\mathbb{R}^n$의 초평면을 정의하며, 활성화 함수는 이 초평면의 양쪽에 있는 입력에 뉴런이 어떻게 반응할지를 정한다.

### 층의 계산

$n^{[l-1]}$개의 뉴런에서 입력을 받는 $n^{[l]}$개 뉴런의 층 $l$에서 계산은 뉴런 하나를 **벡터화**한 것이다.

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right)
$$

여기서 각 기호는 다음과 같다.

- $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$은 층 $l$의 가중치 행렬이다
- $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$은 편향 벡터이다
- $\mathbf{a}^{[l-1]} \in \mathbb{R}^{n^{[l-1]}}$은 앞 층의 활성화이다
- $\mathbf{z}^{[l]} \in \mathbb{R}^{n^{[l]}}$은 활성화 전 벡터이다
- $\mathbf{a}^{[l]} \in \mathbb{R}^{n^{[l]}}$은 활성화 출력이다

가중치 행렬의 각 행 $\mathbf{W}^{[l]}_{j,:}$은 층 $l$의 뉴런 $j$에 대한 가중치를 담는다. 행렬-벡터 곱이 $n^{[l]}$개 뉴런을 한꺼번에 계산한다.

### 신경망 전체

층이 $L$개인 신경망에서 순전파 계산은 **아핀 변환과 비선형성의 합성**이다.

$$
\mathbf{a}^{[0]} = \mathbf{x} \quad \text{(input)}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad l = 1, \ldots, L
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right), \quad l = 1, \ldots, L
$$

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]} \quad \text{(output)}
$$

간결하게 쓰면 신경망 전체가 함수의 합성이다.

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \sigma^{[L]} \circ g^{[L]} \circ \sigma^{[L-1]} \circ g^{[L-1]} \circ \cdots \circ \sigma^{[1]} \circ g^{[1]}(\mathbf{x})
$$

여기서 $g^{[l]}(\mathbf{a}) = \mathbf{W}^{[l]} \mathbf{a} + \mathbf{b}^{[l]}$은 층 $l$의 아핀 사상이고 $\boldsymbol{\theta} = \{\mathbf{W}^{[l]}, \mathbf{b}^{[l]}\}_{l=1}^L$은 학습 가능한 모든 매개변수를 나타낸다.

!!! warning "비선형성이 반드시 필요한 이유"
    활성화 함수가 없으면 아핀 사상의 합성은 그 자체로 아핀이다.
    
    $$
    \mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = \underbrace{(\mathbf{W}^{[2]}\mathbf{W}^{[1]})}_{\mathbf{W}'}\mathbf{x} + \underbrace{(\mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]})}_{\mathbf{b}'}
    $$
    
    층을 아무리 쌓아도 신경망은 선형 함수만 표현할 수 있다. 깊이에 힘을 주는 것이 바로 활성화 함수 $\sigma$이다.

---

## 5. 구조의 구성 요소

### 입력층

입력층은 날것의 특징을 받아 첫 은닉층으로 넘긴다. 계산은 하지 않고 입력값 $\mathbf{a}^{[0]} = \mathbf{x}$을 담고 있을 뿐이다.

**설계 고려 사항:**

- 뉴런의 개수는 입력 특징의 개수와 같다. $n^{[0]} = \dim(\mathbf{x})$이다
- 이미지에서는 (평탄화하여) $n^{[0]} = H \times W \times C$이다. 예컨대 MNIST는 $n^{[0]} = 28 \times 28 = 784$이다
- 표 형태 데이터에서는 $n^{[0]}$이 전처리 후의 특징 개수이다

### 은닉층

은닉층이 계산과 특징 추출의 대부분을 맡는다. 각 은닉층은 두 단계의 변환을 적용한다.

1. **아핀 사상** (선형 변환 + 편향): $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
2. **비선형 활성화**: $\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$

은닉층의 목적은 입력 데이터에 대해 점점 더 추상적이고 계층적인 표현을 배우는 것이다. 앞쪽 층은 저수준 특징(모서리, 단순한 무늬)을 잡아내고, 깊은 층은 이를 모아 더 높은 수준의 추상으로 엮는다.

### 출력층

출력층은 최종 예측을 낸다. 그 설계는 과제에 달려 있다.

| 과제 | $n^{[L]}$ | 활성화 $\sigma^{[L]}$ | 손실 함수 |
|------|-----------|--------------------------|---------------|
| 이진 분류 | 1 | 시그모이드 | 이진 교차 엔트로피 |
| 다중 클래스 분류 | $K$ (클래스 수) | 소프트맥스 | 범주형 교차 엔트로피 |
| 회귀 | 1 (또는 $d$) | 항등 (없음) | MSE / MAE |
| 다중 이름표 분류 | $K$ | 시그모이드 (각각) | 이진 교차 엔트로피 (각각) |

!!! tip "PyTorch의 소프트맥스와 교차 엔트로피"
    PyTorch의 `nn.CrossEntropyLoss`은 수치적 안정성을 위해 내부에서 로그 소프트맥스를 적용한다. 이를 쓸 때 출력층은 (소프트맥스 활성화 없이) **날것의 로짓**을 내야 한다. 그래야 소프트맥스를 두 번 계산하지 않고 소프트맥스의 로그에서 오는 수치 문제도 피할 수 있다.

---

## 6. 신경망의 표기

### 표준 표기

층이 $l = 0, 1, \ldots, L$인 신경망에 대해 다음이 성립한다.

| 기호 | 설명 | 모양 |
|--------|-------------|-------|
| $n^{[l]}$ | 층 $l$의 뉴런 개수 | 스칼라 |
| $\mathbf{W}^{[l]}$ | 층 $l$의 가중치 행렬 | $(n^{[l]}, n^{[l-1]})$ |
| $\mathbf{b}^{[l]}$ | 층 $l$의 편향 벡터 | $(n^{[l]},)$ |
| $\mathbf{z}^{[l]}$ | 층 $l$의 활성화 전 값 | $(n^{[l]},)$ |
| $\mathbf{a}^{[l]}$ | 층 $l$의 활성화 후 값 | $(n^{[l]},)$ |
| $\sigma^{[l]}$ | 층 $l$의 활성화 함수 | — |

### 배치 처리

표본 $B$개의 미니배치에서는 모든 벡터가 행렬이 된다. 각 **열**이 표본인 관례를 쓰면 다음과 같다.

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \mathbf{1}_B^\top
$$

여기서 각 기호는 다음과 같다.

- $\mathbf{A}^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times B}$ — 각 열이 표본 하나의 활성화이다
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times B}$
- $\mathbf{1}_B^\top$은 편향을 $B$개 표본 전체에 브로드캐스팅한다

!!! note "PyTorch의 관례"
    PyTorch는 각 **행**이 표본인 **행 우선** 형식을 쓴다.
    
    - 입력 모양: $(B, n^{[0]})$
    - `nn.Linear`의 가중치 행렬: 모양 $(n^{[l]}, n^{[l-1]})$
    - 계산: `output = input @ weight.T + bias`이며 모양은 $(B, n^{[l]})$이다
    
    이는 위의 수학적 관례를 전치한 것이다. 둘은 동등하며, 행 우선 형식이 GPU 메모리 배치에 더 자연스럽다.

---

## 7. 매개변수 개수

### 유도

각 층 $l$은 다음을 갖는다.

- **가중치:** $n^{[l]} \times n^{[l-1]}$개 (연결마다 하나)
- **편향:** $n^{[l]}$개 (뉴런마다 하나)

신경망 전체의 매개변수 총수는 다음과 같다.

$$
|\boldsymbol{\theta}| = \sum_{l=1}^{L} \left( n^{[l]} \cdot n^{[l-1]} + n^{[l]} \right) = \sum_{l=1}^{L} n^{[l]}\!\left(n^{[l-1]} + 1\right)
$$

### 증가 양상

너비가 일정한($모든 은닉층에서 n^{[l]} = d$) 층 $L$개짜리 신경망에서 다음이 성립한다.

$$
|\boldsymbol{\theta}| = n^{[0]} \cdot d + (L-2) \cdot d^2 + d \cdot n^{[L]} + L \cdot d
$$

지배적인 항은 $(L-2) \cdot d^2$이므로 매개변수는 너비에 대해 **이차적으로**, 깊이에 대해 **선형적으로** 늘어난다.

### 풀이 예제

구조가 $[784, 256, 128, 10]$인 신경망을 보자.

| 층 $l$ | $n^{[l-1]} \to n^{[l]}$ | 가중치 | 편향 | 합계 |
|-----------|--------------------------|---------|--------|-------|
| 1 | $784 \to 256$ | $256 \times 784 = 200{,}704$ | $256$ | $200{,}960$ |
| 2 | $256 \to 128$ | $128 \times 256 = 32{,}768$ | $128$ | $32{,}896$ |
| 3 | $128 \to 10$ | $10 \times 128 = 1{,}280$ | $10$ | $1{,}290$ |
| **합계** | | $234{,}752$ | $394$ | **$235{,}146$** |

1층 하나가 전체 매개변수의 $200{,}960 / 235{,}146 \approx 85\%$을 차지한다. 입력 차원이 크기 때문이다.

---

## 8. PyTorch 구현

### `nn.Module`로 만드는 설정 가능한 MLP

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    구조를 설정할 수 있는 다층 퍼셉트론.
    
    인수:
        input_size: 입력 특징의 수
        hidden_sizes: 은닉층 너비의 목록, 예: [256, 128]
        output_size: 출력 뉴런의 수
        activation: 활성화 함수 클래스 (기본값: nn.ReLU)
        output_activation: 출력 활성화 (로짓이면 None)
        dropout: 은닉층 사이의 드롭아웃 확률 (0이면 드롭아웃 없음)
    """
    def __init__(
        self, 
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: type = nn.ReLU,
        output_activation: type | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers: list[nn.Module] = []
        prev_size = input_size
        
        # 은닉층
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 출력층
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def count_parameters(self) -> int:
        """학습 가능한 전체 매개변수의 수를 센다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── 예: MNIST 다중 클래스 분류 ──
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    activation=nn.ReLU,
    output_activation=None,   # CrossEntropyLoss를 위한 날 로짓
    dropout=0.2,
)

print(f"Architecture: 784 → 256 → 128 → 10")
print(f"Total parameters: {model.count_parameters():,}")

# 모양 확인
x = torch.randn(32, 784)     # 크기 32인 배치
logits = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {logits.shape}")
```

**출력:**
```
Architecture: 784 → 256 → 128 → 10
Total parameters: 235,146
Input shape:  torch.Size([32, 784])
Output shape: torch.Size([32, 10])
```

### `nn.Sequential`을 곧바로 쓰기

구조가 단순하다면 `nn.Sequential`을 쓰면 클래스를 따로 정의하지 않아도 된다.

```python
# 간단한 MLP 정의
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
)

# 층별 모양 살펴보기
x = torch.randn(1, 784)
for i, layer in enumerate(model):
    x = layer(x)
    print(f"Layer {i} ({layer.__class__.__name__:>10s}): {x.shape}")
```

**출력:**
```
Layer 0 (    Linear): torch.Size([1, 256])
Layer 1 (      ReLU): torch.Size([1, 256])
Layer 2 (   Dropout): torch.Size([1, 256])
Layer 3 (    Linear): torch.Size([1, 128])
Layer 4 (      ReLU): torch.Size([1, 128])
Layer 5 (   Dropout): torch.Size([1, 128])
Layer 6 (    Linear): torch.Size([1, 10])
```

### 완전한 학습 예제

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── 초매개변수 ──
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

# ── 데이터 ──
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST의 평균과 표준편차
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── 모델 ──
model = MLP(
    input_size=784,
    hidden_sizes=[512, 256],
    output_size=10,
    activation=nn.ReLU,
    dropout=0.2,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── 학습 루프 ──
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for data, target in train_loader:
        data = data.view(data.size(0), -1)   # 28×28 → 784로 펼치기
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    
    # ── 평가 ──
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            test_correct += output.argmax(dim=1).eq(target).sum().item()
            test_total += data.size(0)
    
    print(
        f"Epoch {epoch+1:2d}/{EPOCHS} | "
        f"Train Loss: {total_loss/total:.4f} | "
        f"Train Acc: {100*correct/total:.2f}% | "
        f"Test Acc: {100*test_correct/test_total:.2f}%"
    )
```

---

## 9. 구조 설계의 원리

### 층 크기를 정하는 지침

1. **단순하게 시작하라.** 은닉층 하나나 둘로 시작하고, 과소적합할 때에만 복잡도를 더하라.
2. **피라미드(깔때기) 구조.** 출력 쪽으로 갈수록 층의 너비를 점차 줄여라. 이는 신경망이 정보를 압축하도록 만든다. 예: $784 \to 256 \to 128 \to 10$.
3. **2의 거듭제곱.** GPU 메모리 정렬 효율을 위해 64, 128, 256, 512 같은 너비를 쓰라.
4. **병목에 유의하라.** 가장 좁은 은닉층이 정보의 흐름을 제한한다. 과제에 충분할 만큼 넓은지 확인하라.

### 흔한 구조 패턴

```
Small dataset:    Input → 128 → 64 → Output
Medium dataset:   Input → 256 → 128 → 64 → Output
Large dataset:    Input → 512 → 256 → 128 → Output
```

### 평범한 MLP를 쓰지 말아야 할 때

MLP는 완전 연결이어서 모든 뉴런이 이웃한 층의 모든 뉴런과 이어진다. 여기에는 한계가 있다.

- **이미지:** 공간 구조를 살리지 못한다 → CNN(5장)이 낫다
- **열:** 시간 구조를 살리지 못한다 → RNN/트랜스포머(7-8장)가 낫다
- **그래프:** 관계 구조를 살리지 못한다 → GNN이 낫다

그럼에도 MLP는 표 형태 데이터에, 그리고 더 큰 구조 안의 구성 요소(예: 분류 갈래)로 여전히 훌륭한 선택이다.

---

## 10. 핵심 정리

!!! success "요약"

    1. **MLP는 입력층, 은닉층, 출력층을 갖춘 완전 연결 순방향 신경망이다**
    2. **각 층은** 아핀 변환 뒤에 비선형 활성화를 적용한다. $\mathbf{a}^{[l]} = \sigma(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$
    3. **활성화 함수는 필수적이다.** 그것이 없으면 아무리 깊은 신경망도 아핀 사상 하나로 주저앉는다
    4. **매개변수 개수**는 너비가 일정한 신경망에서 $O(d^2 L)$으로 늘어나며, 가장 넓은 층의 연결이 대부분을 차지한다
    5. **출력층의 설계**는 과제에 달려 있다. 이진 분류에는 시그모이드, 다중 클래스에는 (`CrossEntropyLoss`을 통한) 소프트맥스, 회귀에는 항등함수를 쓴다
    6. **PyTorch의 `nn.Linear`**은 행 우선 관례를 쓴다. 입력이 $(B, n_\text{in})$, 출력이 $(B, n_\text{out})$이다

---

## 연습문제

**연습문제 1.**
입력 차원이 784, 은닉층이 256과 128, 출력 차원이 10인 3층 MLP의 (편향을 포함한) 전체 매개변수 개수를 계산하라.

??? success "연습문제 1 풀이"
    1층: $784 \times 256 + 256 = 200{,}960$. 2층: $256 \times 128 + 128 = 32{,}896$. 3층: $128 \times 10 + 10 = 1{,}290$. 합계: 매개변수 $235{,}146$개.

---

**연습문제 2.**
활성화 함수가 없는 MLP가 깊이와 무관하게 선형 변환 하나와 동등한 이유를 설명하라.

??? success "연습문제 2 풀이"
    선형 함수의 합성은 선형이다. $f(x) = W_L(W_{L-1}(\cdots W_1 x)) = (W_L W_{L-1} \cdots W_1)x = W'x$이다. 비선형성이 없으면 층을 쌓아도 표현력이 늘지 않는다.

---

**연습문제 3.**
`nn.Sequential`을 써서 2층 MLP를 PyTorch로 구현하고 간단한 분류 과제로 학습시켜라.

??? success "연습문제 3 풀이"
    ```python
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 10)
    )
    ```

---

**연습문제 4.**
은닉층의 너비 선택이 모델의 용량에 어떤 영향을 주는가? 층을 넓힐 때 무엇을 주고 무엇을 얻는가?

??? success "연습문제 4 풀이"
    층을 넓히면 용량이 커지지만(매개변수가 늘고 표현이 풍부해진다) 과적합의 위험이 있고 계산과 메모리를 더 쓴다. 보편 근사 정리는 충분히 넓은 은닉층 하나로 임의의 연속 함수를 근사할 수 있음을 보장하지만, 실무에서는 깊이가 더 효율적인 경우가 많다.

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、개요、수학적 정식화을 차례로 짚었다.

**참고 문헌**

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 5.
