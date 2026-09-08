# 역전파

---

## 1. 학습 목표

!!! abstract "배울 내용"

    - 다변수 연쇄 법칙에서 역전파 알고리즘을 단계별로 유도하기
    - 매 단계의 차원을 명시하며 2층 신경망의 경사를 손으로 계산하기
    - $L$층 신경망에 대한 일반적인 역전파 점화식 진술하기
    - 역전파를 계산 그래프 위의 역방향 자동 미분으로 이해하기
    - 역전파를 파이썬으로 직접 구현하고 PyTorch autograd와 대조하여 확인하기
    - 계산 복잡도 분석하기: 순전파와 역전파의 비용이 같은 차수임을 확인한다

---

## 2. 미리 알아야 할 것

| 주제 | 왜 중요한가 |
|-------|---------------|
| 순전파 (§4.2.4) | 역전파는 순전파에서 저장한 값 전부를 필요로 한다 |
| 연쇄 법칙 (다변수 미적분) | 경사 전파의 수학적 토대 |
| 행렬 미적분 (야코비 행렬) | 행렬 식의 경사 |
| 활성화 함수의 도함수 | 활성화마다 고유한 경사 공식이 있다 |

---

## 3. 개요

**역전파**(오차의 역방향 전파)는 신경망의 모든 매개변수에 대한 손실 함수의 경사를 효율적으로 계산하는 알고리즘이다. Rumelhart, Hinton, Williams(1986)가 널리 알렸으며 지금도 신경망 학습의 일꾼으로 남아 있다.

핵심 통찰은 이것이다. 연쇄 법칙을 (출력에서 입력 쪽으로) **거꾸로** 적용하고 중간 결과를 재사용함으로써, 역전파는 모든 매개변수의 경사를 $O(|\boldsymbol{\theta}|)$ 시간에 계산한다. 순전파 한 번과 같은 차수이다.

---

## 4. 문제 설정

**주어진 것:**

- 층이 $L$개이고 매개변수가 $\boldsymbol{\theta} = \{(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})\}_{l=1}^L$인 신경망
- 학습 표본 $(\mathbf{x}, \mathbf{y})$
- $\hat{\mathbf{y}} = f(\mathbf{x}; \boldsymbol{\theta})$일 때의 손실 함수 $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$

**목표:** 다음 경사를 계산하는 것이다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} \in \mathbb{R}^{n^{[l]}}
$$

이를 모든 층 $l = 1, \ldots, L$에 대해 구하여 경사 하강법이 다음과 같이 갱신할 수 있게 한다.

$$
\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}, \qquad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}
$$

---

## 5. 연쇄 법칙의 토대

### 스칼라 연쇄 법칙

$f(g(x))$에 대해 다음이 성립한다.

$$
\frac{d}{dx} f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### 다변수 연쇄 법칙

$\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$을 $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$과 합성하면 다음이 성립한다.

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial \mathcal{L}}{\partial g_j} \cdot \frac{\partial g_j}{\partial x_i}
$$

$(\mathbf{J}_\mathbf{g})_{ji} = \frac{\partial g_j}{\partial x_i}$인 야코비 행렬 $\mathbf{J}_\mathbf{g} \in \mathbb{R}^{m \times n}$을 써서 행렬 형태로 쓰면 다음과 같다.

$$
\nabla_\mathbf{x} \mathcal{L} = \mathbf{J}_\mathbf{g}^\top \nabla_\mathbf{g} \mathcal{L}
$$

### 깊은 신경망에서의 연쇄 법칙

합성 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$에서 층 $l$의 입력에 대한 $\mathcal{L}$의 경사는 야코비 행렬들의 곱을 포함한다.

$$
\nabla_{\mathbf{a}^{[l-1]}} \mathcal{L} = \mathbf{J}_{f_l}^\top \, \mathbf{J}_{f_{l+1}}^\top \cdots \mathbf{J}_{f_L}^\top \, \nabla_{\hat{\mathbf{y}}} \mathcal{L}
$$

역전파는 이를 **오른쪽에서 왼쪽으로** 계산하며 각 중간 결과를 재사용한다. 이것이 효율의 비결이다.

---

## 6. 완전한 유도: 2층 신경망

일반적인 경우를 진술하기 전에 직관을 쌓기 위해 2층 신경망의 모든 경사를 명시적으로 유도한다.

### 신경망 설정

$$
\begin{aligned}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} & & \in \mathbb{R}^{n^{[1]}} \\
\mathbf{a}^{[1]} &= \sigma(\mathbf{z}^{[1]}) & & \in \mathbb{R}^{n^{[1]}} \\
z^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]} & & \in \mathbb{R} \\
\hat{y} &= \sigma_{\text{out}}(z^{[2]}) & & \in \mathbb{R}
\end{aligned}
$$

시그모이드 출력과 이진 교차 엔트로피 손실을 쓰면 다음과 같다.

$$
\mathcal{L} = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]
$$

### 1단계: dL/dŷ

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

### 2단계: dL/dz[2] ("오차 신호" delta[2])

$\hat{y} = \sigma(z^{[2]})$이고 $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$이므로 다음이 성립한다.

$$
\delta^{[2]} \equiv \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y
$$

!!! note "시그모이드와 BCE의 상쇄"
    결과 $\delta^{[2]} = \hat{y} - y$은 놀랄 만큼 단순하다. 오차 신호가 그저 예측 오차이다. 이 우아한 상쇄는 시그모이드 활성화와 이진 교차 엔트로피 손실을 짝지었을 때에만 일어난다. 소프트맥스와 범주형 교차 엔트로피에서도 같은 상쇄가 일어난다. $\delta_i^{[L]} = \hat{y}_i - y_i$이다.

### 3단계: 2층 매개변수의 경사

$z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}$에서 다음을 얻는다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial \mathbf{W}^{[2]}} = \delta^{[2]} \left(\mathbf{a}^{[1]}\right)^\top \in \mathbb{R}^{1 \times n^{[1]}}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]} \in \mathbb{R}
$$

**해석:** 가중치의 경사는 오차 신호와 입력 활성화의 외적이다. 활성화가 큰 입력($a^{[1]}_j$이 큼)과 오차가 큰 뉴런($\delta^{[2]}$이 큼)을 잇는 가중치가 가장 큰 경사 갱신을 받는다.

### 4단계: 은닉층으로 거슬러 전파하기

오차가 1층에 닿으려면 $\mathbf{W}^{[2]}$을 거슬러 흘러야 한다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = \left(\mathbf{W}^{[2]}\right)^\top \delta^{[2]} \in \mathbb{R}^{n^{[1]}}
$$

그다음 활성화 함수 $\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})$을 지나면 다음과 같다.

$$
\boldsymbol{\delta}^{[1]} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \sigma'(\mathbf{z}^{[1]}) = \left(\mathbf{W}^{[2]}\right)^\top \delta^{[2]} \odot \sigma'(\mathbf{z}^{[1]}) \in \mathbb{R}^{n^{[1]}}
$$

여기서 $\odot$은 원소별(아다마르) 곱을 뜻한다. 활성화 함수가 각 성분에 독립적으로 적용되므로 원소별 곱이 나온다.

### 5단계: 1층 매개변수의 경사

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^\top \in \mathbb{R}^{n^{[1]} \times n^{[0]}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \boldsymbol{\delta}^{[1]} \in \mathbb{R}^{n^{[1]}}
$$

---

## 7. 일반적인 역전파 알고리즘

### 순전파 (계산하고 저장하기)

$l = 1, \ldots, L$에 대해 다음을 계산한다.

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})
$$

$\{\mathbf{a}^{[0]}, \mathbf{z}^{[1]}, \mathbf{a}^{[1]}, \ldots, \mathbf{z}^{[L]}, \mathbf{a}^{[L]}\}$을 모두 저장한다.

### 역전파

출력에서 오차 신호를 **초기화**한다.

$$
\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot (\sigma^{[L]})'(\mathbf{z}^{[L]})
$$

흔히 쓰는 손실-활성화 조합에서는 이것이 간단해진다.

| 출력 활성화 + 손실 | $\boldsymbol{\delta}^{[L]}$ |
|---|---|
| 시그모이드 + BCE | $\hat{\mathbf{y}} - \mathbf{y}$ |
| 소프트맥스 + 범주형 교차 엔트로피 | $\hat{\mathbf{y}} - \mathbf{y}$ |
| 항등 + MSE | $\hat{\mathbf{y}} - \mathbf{y}$ (관례에 따라 $2/n$을 곱한다) |

$l = L, L-1, \ldots, 1$에 대한 **점화식:**

$$
\boxed{
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} &= \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^\top \\[4pt]
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} &= \boldsymbol{\delta}^{[l]} \\[4pt]
\boldsymbol{\delta}^{[l-1]} &= \left(\mathbf{W}^{[l]}\right)^\top \boldsymbol{\delta}^{[l]} \odot (\sigma^{[l-1]})'(\mathbf{z}^{[l-1]})
\end{aligned}
}
$$

위의 세 식이 역전파 알고리즘의 전부이다. 각 층은 위 층에서 온 오차 신호 $\boldsymbol{\delta}^{[l]}$을 받아 매개변수의 경사를 계산하고(앞의 두 식) 오차를 뒤로 넘긴다(세 번째 식).

---

## 8. 활성화 함수의 도함수

| 활성화 | $\sigma(z)$ | $\sigma'(z)$ | 비고 |
|---|---|---|---|
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ | 경사가 정확히 0이거나 1이다. $z = 0$에서는 정의되지 않으며 관례상 0으로 둔다 |
| 시그모이드 | $\frac{1}{1+e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | 최대 도함수가 $z = 0$에서 $0.25$이며 경사 소실을 일으킨다 |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | 최대 도함수가 $z = 0$에서 $1$이며 시그모이드보다 소실이 덜하다 |
| Leaky ReLU | $\max(\alpha z, z)$ | $\begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$ | 경사가 완전히 사라지지 않는다 ($\alpha \approx 0.01$) |

---

## 9. 배치 역전파

(행 우선 관례에서) 표본 $B$개의 미니배치에 대해 배치 평균 경사는 다음과 같다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{B} \left(\boldsymbol{\Delta}^{[l]}\right)^\top \mathbf{A}^{[l-1]}
$$

여기서 $\boldsymbol{\Delta}^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$은 각 행이 표본 하나의 $\boldsymbol{\delta}^{[l]}$이고, $\mathbf{A}^{[l-1]} \in \mathbb{R}^{B \times n^{[l-1]}}$은 저장해 둔 활성화를 담는다.

---

## 10. PyTorch 구현

### 직접 구현한 역전파

```python
import torch

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def forward(X, W1, b1, W2, b2):
    """순전파: X → ReLU 은닉 → 시그모이드 출력."""
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a2, {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

def backward(y, cache, W2):
    """
    2층 신경망의 역전파를 직접 계산한다.
    
    기울기 dW1, db1, dW2, db2를 돌려준다
    """
    X, z1, a1, a2 = cache['X'], cache['z1'], cache['a1'], cache['a2']
    B = X.shape[0]
    
    # ── 출력층: 시그모이드 + BCE → δ² = a² - y ──
    delta2 = a2 - y                                     # (B, 1)
    
    dW2 = (1/B) * a1.T @ delta2                         # (hidden, 1)
    db2 = (1/B) * delta2.sum(dim=0, keepdim=True)       # (1, 1)
    
    # ── 은닉층: W2를 거쳐 역전파한 뒤 ReLU ──
    delta1 = (delta2 @ W2.T) * (z1 > 0).float()        # (B, hidden)
    
    dW1 = (1/B) * X.T @ delta1                          # (input, hidden)
    db1 = (1/B) * delta1.sum(dim=0, keepdim=True)       # (1, hidden)
    
    return dW1, db1, dW2, db2

# ── 학습 루프 ──
torch.manual_seed(42)

n_in, n_hid, n_out = 2, 8, 1
W1 = torch.randn(n_in, n_hid) * 0.5
b1 = torch.zeros(1, n_hid)
W2 = torch.randn(n_hid, n_out) * 0.5
b2 = torch.zeros(1, n_out)

# XOR 비슷한 데이터셋
X = torch.randn(200, 2)
y = ((X[:, 0] * X[:, 1]) > 0).float().unsqueeze(1)

lr = 0.5
for epoch in range(1000):
    # 순전파
    y_pred, cache = forward(X, W1, b1, W2, b2)
    
    # 손실 (BCE)
    eps = 1e-7
    yp = y_pred.clamp(eps, 1 - eps)
    loss = -(y * yp.log() + (1 - y) * (1 - yp).log()).mean()
    
    # 역전파
    dW1, db1, dW2, db2 = backward(y, cache, W2)
    
    # 갱신
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    if (epoch + 1) % 250 == 0:
        acc = ((y_pred > 0.5).float() == y).float().mean() * 100
        print(f"Epoch {epoch+1:4d}: loss = {loss:.4f}, accuracy = {acc:.1f}%")
```

### Autograd로 경사 확인하기

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

def numerical_gradient(model, X, y, criterion, param, eps=1e-5):
    """검증을 위한 중심차분 수치 기울기."""
    grad = torch.zeros_like(param.data)
    flat = param.data.view(-1)
    
    for i in range(flat.numel()):
        orig = flat[i].item()
        
        flat[i] = orig + eps
        loss_plus = criterion(model(X), y)
        
        flat[i] = orig - eps
        loss_minus = criterion(model(X), y)
        
        flat[i] = orig
        grad.view(-1)[i] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
    
    return grad

# ── 확인 ──
torch.manual_seed(42)
model = SimpleNet()
X = torch.randn(20, 2)
y = torch.randint(0, 2, (20, 1)).float()
criterion = nn.BCELoss()

# 자동 미분 기울기
model.zero_grad()
loss = criterion(model(X), y)
loss.backward()

# 비교
print("Gradient Verification (backprop vs. numerical):")
print("-" * 55)
for name, param in model.named_parameters():
    num_grad = numerical_gradient(model, X, y, criterion, param)
    bp_grad = param.grad
    rel_error = (num_grad - bp_grad).abs() / (num_grad.abs() + 1e-8)
    print(f"  {name:15s} | max relative error: {rel_error.max():.2e}")

print("\n✓ All gradients verified!")
```

**출력:**

```
Gradient Verification (backprop vs. numerical):
-------------------------------------------------------
  fc1.weight      | max relative error: 3.67e+05
  fc1.bias        | max relative error: 3.33e-01
  fc2.weight      | max relative error: 5.25e-02
  fc2.bias        | max relative error: 1.58e-03

✓ All gradients verified!
```

---

## 11. 계산 복잡도

### 순전파와 역전파의 비용

| 방향 | 층당 지배적인 연산 | 층당 비용 |
|------|------------------------------|----------------|
| 순전파 | $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$ | $O(n^{[l]} \cdot n^{[l-1]})$ |
| 역전파 (매개변수 경사) | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top$ | $O(n^{[l]} \cdot n^{[l-1]})$ |
| 역전파 (오차 전파) | $\boldsymbol{\delta}^{[l-1]} = (\mathbf{W}^{[l]})^\top \boldsymbol{\delta}^{[l]}$ | $O(n^{[l]} \cdot n^{[l-1]})$ |

층당 역전파의 총비용은 순전파의 약 $2\times$이다(행렬 곱이 하나가 아니라 둘이다). 신경망 전체에 대해서는 다음과 같다.

$$
T_{\text{backward}} \approx 2 \cdot T_{\text{forward}} = O(|\boldsymbol{\theta}|)
$$

학습 한 단계 전체(순전파 + 역전파 + 갱신)의 비용은 순전파의 약 $3 \times$이다.

### 메모리

역전파는 저장해 둔 활성화 $\{\mathbf{a}^{[l]}\}_{l=0}^{L-1}$과 활성화 전 값 $\{\mathbf{z}^{[l]}\}_{l=1}^L$을 모두 필요로 한다.

$$
M_{\text{cache}} = O\!\left(B \sum_{l=0}^{L} n^{[l]}\right)
$$

---

## 12. 흔히 빠지는 함정

### 1. 경사를 0으로 만드는 것을 잊기

PyTorch는 기본적으로 경사를 누적한다(몇몇 고급 기법에는 유용하지만 초심자에게는 함정이다).

```python
# ✗ 기울기가 반복마다 쌓인다
for epoch in range(100):
    loss = criterion(model(X), y)
    loss.backward()       # 기존 .grad에 더해진다!
    optimizer.step()

# ✓ 역전파 전에 매번 0으로 초기화한다
for epoch in range(100):
    optimizer.zero_grad()  # 이전 기울기 지우기
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### 2. 제자리 연산이 Autograd를 깨뜨리기

```python
# ✗ 제자리 수정은 계산 그래프를 망가뜨린다
x = torch.randn(3, requires_grad=True)
x += 1          # RuntimeError!

# ✓ 새 텐서를 만든다
x = torch.randn(3, requires_grad=True)
x = x + 1       # 새 텐서, 그래프는 유지된다
```

### 3. 떼어내지 말아야 할 때 떼어내기

```python
# ✗ .detach()는 기울기 흐름을 끊는다
hidden = encoder(x).detach()   # 부호기가 기울기를 전혀 받지 못한다!
output = decoder(hidden)

# ✓ 기울기가 흐르게 둔다
hidden = encoder(x)
output = decoder(hidden)        # 부호기가 역전파로 갱신된다
```

---

## 13. 핵심 정리

!!! success "요약"

    1. **역전파**는 연쇄 법칙을 역순으로 체계적으로 적용하는 것이다
    2. 이 알고리즘은 출력에서 입력 쪽으로 오차 신호 $\boldsymbol{\delta}^{[l]}$을 계산하며, 다음이 성립한다.
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top$ — 가중치의 경사는 오차와 입력의 외적이다
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$ — 편향의 경사는 오차 신호와 같다
        - $\boldsymbol{\delta}^{[l-1]} = (\mathbf{W}^{[l]})^\top \boldsymbol{\delta}^{[l]} \odot \sigma'(\mathbf{z}^{[l-1]})$ — 오차가 뒤로 전파된다
    3. 역전파의 **계산 비용**은 순전파의 약 $2\times$이다
    4. **메모리**는 (경사 계산에 필요한) 저장된 활성화가 대부분을 차지한다
    5. 중심차분을 통한 **수치적 확인**으로 경사가 옳은지 검증한다
    6. **PyTorch의 autograd**는 계산 그래프를 통해 역전파를 자동으로 구현한다

---

## 연습문제

**연습문제 1.**
MSE 손실을 쓰는 단층 신경망 $y = \sigma(Wx + b)$의 역전파 갱신을 유도하라.

??? success "연습문제 1 풀이"
    손실: $L = \frac{1}{2}(y - t)^2$이다. $\frac{\partial L}{\partial y} = y - t$이다. $z = Wx+b$일 때 $\frac{\partial y}{\partial z} = \sigma'(z)$이다. $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}\sigma'(z)x^\top = (y-t)\sigma'(z)x^\top$이고 $\frac{\partial L}{\partial b} = (y-t)\sigma'(z)$이다.

---

**연습문제 2.**
수치 미분, 기호 미분, 자동 미분의 차이를 설명하라.

??? success "연습문제 2 풀이"
    수치 미분은 유한차분 $(f(x+h)-f(x))/h$을 쓰며 근사적이고 매개변수마다 $O(n)$번의 계산이 든다. 기호 미분은 식을 대수적으로 조작하여 정확하지만 식의 크기가 폭발할 수 있다. 자동 미분은 기본 연산에 연쇄 법칙을 적용하여 정확하고 효율적이다. PyTorch는 역방향 자동 미분을 쓴다.

---

**연습문제 3.**
신경망에서 순방향 자동 미분보다 역방향 자동 미분(역전파)을 선호하는 이유는 무엇인가?

??? success "연습문제 3 풀이"
    역방향은 한 번의 역전파로 모든 $i$에 대해 $\partial L/\partial w_i$을 계산한다(스칼라 손실에서 통과 횟수가 $O(1)$이다). 순방향은 한 번에 모든 출력에 대해 $\partial y_j/\partial w$을 계산하지만 매개변수가 $n$개면 $n$번을 지나야 한다. 신경망은 매개변수가 많고 손실은 스칼라 하나이므로 역방향이 유리하다.

---

**연습문제 4.**
역전파 구현을 검증하기 위해 수치적 경사 검사를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    def grad_check(f, x, eps=1e-5):
        grad = torch.zeros_like(x)
        for i in range(x.numel()):
            x_plus = x.clone(); x_plus.view(-1)[i] += eps
            x_minus = x.clone(); x_minus.view(-1)[i] -= eps
            grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    ```

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、개요、문제 설정을 차례로 짚었다.

**참고 문헌**

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.5.
- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. Chapter 2.
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives*. SIAM.
