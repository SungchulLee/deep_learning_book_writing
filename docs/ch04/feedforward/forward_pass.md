# 순전파

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 표본 하나와 미니배치에 대한 순전파 방정식 유도하기
    - 구체적인 수치로 신경망을 지나는 데이터의 흐름 따라가기
    - 계산 그래프의 관점을 이해하고 중간값을 저장해 두는 것이 왜 필수적인지 알기
    - 순전파를 직접 구현하고 `nn.Module`과 비교하기
    - 순전파의 시간·메모리 복잡도 분석하기
    - 소프트맥스와 교차 엔트로피에 수치적 안정성 기법 적용하기

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| MLP 구조 (§4.2.1) | 여기서 실행되는 층의 계산을 정의한다 |
| 행렬 곱 | 순전파는 행렬-벡터 곱의 연속이다 |
| 활성화 함수 (4.1절) | 각 선형 변환 뒤에 원소별로 적용된다 |

---

## 3. 개요

**순전파**는 입력이 주어졌을 때 신경망의 출력을 계산하는 과정이다. 데이터가 입력층에서 각 은닉층을 거쳐 출력층까지 차례로 흐르며, 모든 층이 선형 변환 뒤에 비선형 활성화를 적용한다.

순전파는 두 가지 역할을 한다. **추론**에서는 예측을 내고, **학습**에서는 그와 함께 **계산 그래프**를 만들고 역전파(§4.2.5)에 필요한 중간값을 저장해 둔다.

---

## 4. 수학적 정식화

### 표본 하나

층이 $L$개이고 매개변수가 $\boldsymbol{\theta} = \{(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})\}_{l=1}^L$인 신경망에서 다음과 같다.

**입력 배정:**

$$
\mathbf{a}^{[0]} = \mathbf{x} \in \mathbb{R}^{n^{[0]}}
$$

$l = 1, 2, \ldots, L$에 대한 **층별 계산:**

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right) \in \mathbb{R}^{n^{[l]}}
$$

**출력:**

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]}
$$

따라서 각 층은 두 가지 연산을 한다. (1) 앞 층의 활성화를 새로운 공간으로 사영하는 **아핀 변환**, (2) 비선형적인 능력을 들여오는 **점별 비선형성**이다.

### 미니배치 처리

표본 $B$개의 미니배치에서는 벡터 계산이 모두 행렬 계산이 된다. 각 행이 표본인 PyTorch의 행 우선 관례를 쓰면 다음과 같다.

**입력:** $\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{B \times n^{[0]}}$

$l = 1, \ldots, L$에 대한 **층의 계산:**

$$
\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top + \mathbf{1}_B \, (\mathbf{b}^{[l]})^\top \in \mathbb{R}^{B \times n^{[l]}}
$$

$$
\mathbf{A}^{[l]} = \sigma^{[l]}\!\left(\mathbf{Z}^{[l]}\right) \in \mathbb{R}^{B \times n^{[l]}}
$$

여기서 편향 항은 $B$개의 행 전체에 브로드캐스팅된다. PyTorch에서는 `nn.Linear`이 자동 브로드캐스팅과 함께 `output = input @ weight.T + bias`으로 이를 처리한다.

!!! tip "배치 처리가 효율적인 이유"
    행렬 곱 $\mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top$은 고도로 최적화된 BLAS/cuBLAS 루틴으로 $B$개의 표본을 한꺼번에 계산한다. 표본을 하나씩 반복문으로 도는 것은 자릿수만큼 느릴 것이다.

---

## 5. 단계별 수치 예제

이진 분류를 위한 2층 신경망을 생각하자.

- 입력: $\mathbf{x} \in \mathbb{R}^2$
- 은닉층: ReLU를 쓰는 뉴런 3개
- 출력: 시그모이드를 쓰는 뉴런 1개

### 층의 차원

| 층 $l$ | $n^{[l-1]} \to n^{[l]}$ | $\mathbf{W}^{[l]}$의 모양 | $\mathbf{b}^{[l]}$의 모양 |
|-----------|--------------------------|---------------------------|---------------------------|
| 1 (은닉) | $2 \to 3$ | $(3, 2)$ | $(3,)$ |
| 2 (출력) | $3 \to 1$ | $(1, 3)$ | $(1,)$ |

### 구체적인 값

$\mathbf{x} = \begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix}$이라 하고 다음과 같다고 하자.

$$
\mathbf{W}^{[1]} = \begin{bmatrix} 0.2 & -0.3 \\ 0.4 & 0.1 \\ -0.5 & 0.6 \end{bmatrix}, \quad
\mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.0 \end{bmatrix}
$$

$$
\mathbf{W}^{[2]} = \begin{bmatrix} 0.7 & -0.4 & 0.3 \end{bmatrix}, \quad
b^{[2]} = -0.1
$$

**1단계 — 은닉층의 활성화 전 값:**

$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}
= \begin{bmatrix} 0.2(0.5) + (-0.3)(0.8) + 0.1 \\ 0.4(0.5) + 0.1(0.8) + (-0.2) \\ -0.5(0.5) + 0.6(0.8) + 0.0 \end{bmatrix}
= \begin{bmatrix} -0.04 \\ 0.08 \\ 0.23 \end{bmatrix}
$$

**2단계 — 은닉층의 활성화 (ReLU):**

$$
\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \begin{bmatrix} \max(0, -0.04) \\ \max(0, 0.08) \\ \max(0, 0.23) \end{bmatrix} = \begin{bmatrix} 0.00 \\ 0.08 \\ 0.23 \end{bmatrix}
$$

참고: 이 입력에서 첫 번째 뉴런은 "죽어" 있다(출력 = 0).

**3단계 — 출력층의 활성화 전 값:**

$$
z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}
= 0.7(0.00) + (-0.4)(0.08) + 0.3(0.23) + (-0.1)
= -0.069
$$

**4단계 — 출력층의 활성화 (시그모이드):**

$$
\hat{y} = \sigma(z^{[2]}) = \frac{1}{1 + e^{0.069}} = \frac{1}{1.0714} \approx 0.4828
$$

$\hat{y} < 0.5$이므로 예측은 클래스 0이다.

---

## 6. 계산 그래프

순전파는 데이터에 수행된 모든 연산을 기록하는 **유향 비순환 그래프(DAG)**, 즉 계산 그래프를 만든다.

```
x ──→ [z¹ = W¹x + b¹] ──→ [a¹ = ReLU(z¹)] ──→ [z² = W²a¹ + b²] ──→ [ŷ = σ(z²)]
       ↑                                          ↑
      W¹, b¹                                     W², b²
```

그래프의 각 노드는 다음을 저장한다.

1. **연산** (행렬 곱, 덧셈, ReLU, 시그모이드 등)
2. **그 입력** (부모 노드를 가리키는 포인터)
3. **그 출력값** (계산된 텐서)
4. **국소 경사를 계산하는 방법** (역전파에 쓰인다)

!!! info "역전파를 위한 저장"
    학습 중에 순전파는 모든 중간값 $\{\mathbf{z}^{[l]}, \mathbf{a}^{[l]}\}_{l=1}^L$을 저장해 두어야 한다. 저장된 이 값들은 역전파에서 경사를 계산하는 데 쓰인다. 학습이 추론보다 메모리를 약 2~3배 쓰는 이유가 여기 있다.
    
    **추론만 할 때**는 중간값이 필요 없으므로 곧바로 버릴 수 있다.
    ```python
    with torch.no_grad():      # 기울기 추적을 끈다
        output = model(input)   # 계산 그래프를 만들지 않는다
    ```

---

## 7. 순전파에서의 활성화 함수

### ReLU

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

계산: 원소마다 비교 한 번. 가장 빠른 활성화이며, 이것이 ReLU가 은닉층의 기본 선택인 이유이다.

### 시그모이드

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

계산: 원소마다 지수 한 번, 덧셈 한 번, 나눗셈 한 번. 출력이 $(0, 1)$으로 유계이며 이진 분류의 출력에 쓰인다.

### 소프트맥스

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K
$$

계산: 지수 $K$번과 정규화 한 번. 유효한 확률 분포(음이 아니고 합이 1)를 낸다.

---

## 8. 수치적 안정성

### 소프트맥스의 넘침

**문제:** $z_i$이 크면(예: $z_i = 1000$) $e^{z_i}$이 넘쳐 `inf`가 된다.

**해결 — log-sum-exp 기법:** 지수를 취하기 전에 $\max_j z_j$을 뺀다.

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - z_{\max}}}{\sum_{j} e^{z_j - z_{\max}}}
$$

이는 (상수가 상쇄되므로) 수학적으로 동일하지만, 가장 큰 지수가 $e^0 = 1$이 되므로 수치적으로 안정하다.

```python
def stable_softmax(z: torch.Tensor) -> torch.Tensor:
    z_shifted = z - z.max(dim=-1, keepdim=True).values
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=-1, keepdim=True)
```

### 로짓을 쓰는 교차 엔트로피

**문제:** `log(softmax(z))`을 계산하면 `log(exp(...))`이 들어가 정밀도를 잃을 수 있다.

**해결:** PyTorch의 `nn.CrossEntropyLoss`은 log-sum-exp 항등식을 써서 로그 소프트맥스와 음의 로그가능도를 수치적으로 안정한 하나의 연산으로 융합한다.

$$
\log \text{softmax}(\mathbf{z})_i = z_i - \log \sum_{j} e^{z_j} = z_i - z_{\max} - \log \sum_{j} e^{z_j - z_{\max}}
$$

`nn.CrossEntropyLoss`을 쓸 때 출력층이 (소프트맥스 없이) **날것의 로짓**을 내야 하는 이유가 여기 있다.

### 시그모이드 + BCE의 안정성

**문제:** $\hat{y} = \sigma(z)$이 0이나 1에 아주 가까우면 $\log(\hat{y})$이나 $\log(1 - \hat{y})$이 발산한다.

**해결:** 내부에서 수치적 안전장치와 함께 시그모이드를 적용하는 `nn.BCEWithLogitsLoss`을 쓴다.

$$
\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})
$$

```python
# ✗ 수치적으로 불안정하다
loss = nn.BCELoss()(torch.sigmoid(logits), targets)

# ✓ 수치적으로 안정하다 (시그모이드가 안에 녹아 있다)
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

---

## 9. PyTorch 구현

### 직접 구현한 순전파

```python
import torch
import torch.nn.functional as F

def forward_pass_manual(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    L개 층 신경망의 순전파를 직접 계산한다.
    
    은닉층은 ReLU를, 출력층은 시그모이드를 쓴다(이진 분류).
    
    인수:
        x:       입력 텐서, 모양 (B, n_in)
        weights: [W1, W2, ..., WL], Wl의 모양은 (n_l, n_{l-1})
        biases:  [b1, b2, ..., bL], bl의 모양은 (n_l,)
    
    반환값:
        output: 예측값, 모양 (B, n_out)
        cache:  모든 중간 z와 a 값의 사전
    """
    L = len(weights)
    cache = {'a0': x}
    a = x
    
    for l in range(L):
        # 아핀: z = a @ W^T + b   (PyTorch의 행 우선 관례)
        z = a @ weights[l].T + biases[l]
        cache[f'z{l+1}'] = z
        
        # 활성화
        if l < L - 1:
            a = F.relu(z)          # 은닉층
        else:
            a = torch.sigmoid(z)   # 출력층
        cache[f'a{l+1}'] = a
    
    return a, cache

# ── 예시 ──
torch.manual_seed(42)

W1 = torch.randn(3, 2) * 0.5     # 은닉: 2 → 3
b1 = torch.zeros(3)
W2 = torch.randn(1, 3) * 0.5     # 출력: 3 → 1
b2 = torch.zeros(1)

x = torch.tensor([[0.5, 0.8],
                   [0.1, 0.2],
                   [0.9, 0.4]])    # 크기 3인 배치

output, cache = forward_pass_manual(x, [W1, W2], [b1, b2])

print("=== Forward Pass Trace ===")
for key in sorted(cache.keys()):
    print(f"  {key}: shape {cache[key].shape}")
    print(f"        {cache[key].detach()}")
print(f"\nPredictions: {output.detach().squeeze()}")
```

### 중간값을 추적하는 순전파 (`nn.Module`)

```python
import torch
import torch.nn as nn

class TrackedMLP(nn.Module):
    """필요하면 모든 중간 활성화를 함께 돌려주는 MLP."""
    
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        # 너비 목록을 받아 이웃한 쌍마다 층을 하나씩 만든다.
        # [784, 256, 128, 10] 이면 층이 3개가 된다.
        # 파이썬 리스트가 아니라 ModuleList라야 안의 매개변수가
        # model.parameters()와 state_dict에 잡힌다
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        # 기본값이 False인 것이 중요하다. 중간값을 늘 들고 있으면
        # 메모리를 크게 쓰므로, 디버깅할 때만 켜도록 해 둔다
        cache = {'a0': x} if return_intermediates else None
        a = x

        for i, layer in enumerate(self.layers):
            z = layer(a)   # 활성 함수 앞의 값. 흔히 로짓 또는 pre-activation이라 한다
            if return_intermediates:
                cache[f'z{i+1}'] = z

            # 활성화: 은닉층에는 ReLU, 출력층에는 항등함수.
            # 마지막 층에 ReLU를 걸면 음수 로짓이 모두 0이 되어
            # 갈래 사이의 차이가 뭉개진다. 소프트맥스는 손실 쪽에서 건다
            a = torch.relu(z) if i < len(self.layers) - 1 else z

            if return_intermediates:
                cache[f'a{i+1}'] = a   # z와 a를 따로 담아 두면 활성 함수가
                                       # 무엇을 얼마나 죽였는지 견줄 수 있다

        return (a, cache) if return_intermediates else a

# ── 사용법 ──
model = TrackedMLP([784, 256, 128, 10])
x = torch.randn(32, 784)   # 배치 32개

# 표준 추론 (중간값 없음)
logits = model(x)
print(f"Output shape: {logits.shape}")

# 디버그 모드 (중간값 포함).
# z와 a의 모양이 같고 층마다 너비가 줄어드는 것을 확인할 수 있다.
# 활성이 죽거나 터지는지 볼 때 이 캐시를 그대로 히스토그램으로 그리면 된다
logits, cache = model(x, return_intermediates=True)
print("\nIntermediate shapes:")
for k, v in cache.items():
    print(f"  {k}: {v.shape}")
```

### `torch.no_grad()`으로 효율적인 추론하기

```python
model = TrackedMLP([784, 256, 128, 10])
model.eval()   # 드롭아웃과 배치 정규화를 평가 모드로

x = torch.randn(64, 784)

# ── 학습 모드: 계산 그래프를 만든다 ──
logits_train = model(x)
print(f"Grad tracking: {logits_train.requires_grad}")  # True

# ── 추론 모드: 그래프 없음, 메모리 절약 ──
with torch.no_grad():
    logits_infer = model(x)
    print(f"Grad tracking: {logits_infer.requires_grad}")  # False

# 출력이 같은지 확인
print(f"Max difference: {(logits_train - logits_infer).abs().max().item():.1e}")  # 0.0
```

**출력:**

```
Grad tracking: True
Grad tracking: False
Max difference: 0.0e+00
```

---

## 10. 계산 복잡도

### 시간 복잡도

입력이 $n_{\text{in}}$개, 출력이 $n_{\text{out}}$개인 층 하나가 표본 $B$개의 배치를 처리할 때 다음이 성립한다.

$$
T_{\text{layer}} = O(B \cdot n_{\text{in}} \cdot n_{\text{out}})
$$

이는 행렬 곱 $\mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top$의 비용이다. 활성화 함수가 $O(B \cdot n_{\text{out}})$을 더하지만 행렬 곱에 묻힌다.

신경망 전체에 대해서는 다음과 같다.

$$
T_{\text{forward}} = O\!\left(B \sum_{l=1}^{L} n^{[l-1]} \cdot n^{[l]}\right) = O(B \cdot |\boldsymbol{\theta}|)
$$

순전파의 비용은 매개변수의 개수와 배치 크기에 **선형**이다.

### 메모리 복잡도

| 모드 | 저장하는 것 | 메모리 |
|------|---------------|--------|
| **학습** | 역전파를 위한 모든 $\mathbf{z}^{[l]}, \mathbf{a}^{[l]}$ | $O\!\left(B \sum_{l=0}^{L} n^{[l]}\right)$ |
| **추론** | 현재 층의 활성화만 | $O\!\left(B \cdot \max_l n^{[l]}\right)$ |

학습 메모리는 매개변수 자체가 아니라 저장해 둔 활성화가 대부분을 차지한다. 아주 깊거나 넓은 신경망에서는 이것이 병목이 될 수 있다. **경사 체크포인트** 같은 기법은 활성화를 저장하는 대신 역전파 중에 다시 계산하여 계산을 내주고 메모리를 얻는다.

---

## 11. 시각화: 활성화의 분포

```python
import matplotlib.pyplot as plt

# 층을 지나며 활성값의 분포가 어떻게 달라지는지 눈으로 확인한다.
# 층이 깊어질수록 분포가 좁아지면 신호가 죽어 가는 것이고(기울기 소실),
# 자꾸 넓어지면 터지는 쪽이다. 초기화가 잘 되었는지 보는 가장 빠른 방법이다.
model = TrackedMLP([784, 512, 256, 128, 10])
x = torch.randn(500, 784)  # 무작위 표본 500개

# return_intermediates=True로 각 층의 활성을 cache에 받아 둔다
logits, cache = model(x, return_intermediates=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))   # 층 4개를 나란히 그린다
for i, l in enumerate([1, 2, 3, 4]):
    # detach: 그림을 그릴 뿐이므로 계산 그래프에서 떼어 낸다
    # flatten: 표본 500개 x 유닛 수를 한 줄로 펴서 히스토그램으로 만든다
    vals = cache[f'a{l}'].detach().numpy().flatten()

    # density=True로 그리면 층마다 유닛 수가 달라도 높이를 견줄 수 있다
    axes[i].hist(vals, bins=50, edgecolor='black', alpha=0.7, density=True)

    # 제목에 평균과 표준편차를 함께 적어 층 사이의 변화를 수로도 본다
    axes[i].set_title(f'Layer {l} activations\n'
                      f'mean={vals.mean():.3f}, std={vals.std():.3f}')
    axes[i].set_xlabel('Activation value')

plt.suptitle('Activation Distributions Through the Network', fontsize=14)
plt.tight_layout()   # 제목과 축 이름이 겹치지 않게 여백을 다시 잡는다
plt.savefig('forward_pass_activations.png', dpi=150, bbox_inches='tight')
plt.show()
```

활성화의 분포를 지켜보면 흔한 문제가 드러난다. 값이 0으로 주저앉거나(활성화 소실), 경계에서 포화하거나(시그모이드/tanh), 끝없이 커지는(활성화 폭발) 경우이다. 건강한 활성화는 모든 층에서 평균과 표준편차가 적당하다.

---

## 12. 핵심 정리

!!! success "요약"

    1. **순전파**는 각 층에서 아핀 변환과 활성화를 차례로 적용하여 신경망의 출력을 계산한다
    2. **미니배치 처리**는 행렬 곱으로 표본에 걸쳐 병렬화한다. $\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} (\mathbf{W}^{[l]})^\top + \mathbf{b}^{[l]}$이다
    3. 순전파는 역전파를 위해 연산을 기록하는 **계산 그래프**를 만든다
    4. 학습 중에는 역전파를 위해 **중간값을 저장해 두어야 한다**. 추론에서는 버려도 된다
    5. **수치적 안정성**을 위해 소프트맥스에는 log-sum-exp 기법이, 손실에는 융합된 함수(`CrossEntropyLoss`, `BCEWithLogitsLoss`)가 필요하다
    6. **시간 복잡도**는 $O(B \cdot |\boldsymbol{\theta}|)$이고, 학습 중 **메모리**는 $O(B \sum_l n^{[l]})$으로 늘어난다

---

## 연습문제

**연습문제 1.**
입력 $\mathbf{x} = [1, -1]$, $W_1 = [[1, 0], [0, 1]]$, $b_1 = [0, 0]$, $W_2 = [1, 1]$, $b_2 = 0$일 때 ReLU 활성화를 쓰는 2층 MLP의 순전파를 따라가라.

??? success "연습문제 1 풀이"
    $\mathbf{h}_1 = \text{ReLU}(W_1\mathbf{x} + b_1) = \text{ReLU}([1, -1]) = [1, 0]$이다. 출력: $W_2\mathbf{h}_1 + b_2 = 1(1) + 1(0) = 1$.

---

**연습문제 2.**
층이 $L$개이고 최대 너비가 $d$인 MLP를 지나는 순전파의 계산 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    각 층이 행렬-벡터 곱 $O(d^2)$과 활성화 $O(d)$을 수행한다. 합계는 $O(Ld^2)$이다.

---

**연습문제 3.**
텐서 연산만 써서 (`nn.Module` 없이) 순전파를 직접 구현하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    def forward(x, weights, biases):
        for W, b in zip(weights, biases):
            x = torch.relu(x @ W.T + b)
        return x
    ```

---

**연습문제 4.**
추론 시점에는 순전파가 결정적이어야 하지만 학습 중에는 (드롭아웃 같은) 확률적 요소가 들어갈 수 있는 이유를 설명하라.

??? success "연습문제 4 풀이"
    학습 중에 드롭아웃 같은 확률적 요소는 뉴런을 무작위로 0으로 만들어 공적응을 막는 정칙화 노릇을 한다. 추론에서는 결정적이고 재현 가능한 예측을 원하므로 드롭아웃을 끄고 모든 뉴런을 쓴다(기댓값과 맞도록 가중치의 배율을 조정한다).

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、개요、수학적 정식화을 차례로 짚었다.

**참고 문헌**

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.4.
- PyTorch Documentation: [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
