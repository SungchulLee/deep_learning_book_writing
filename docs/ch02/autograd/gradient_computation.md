# 경사 계산

PyTorch의 autograd 엔진은 딥러닝에서 자동 미분의 토대이다. 텐서에 대한 연산을 기록하고 **후진 모드 자동 미분**(역전파)으로 경사를 계산하는 테이프 기반 체계를 제공한다. 이 절에서는 경사 계산의 수학적 틀, 전진 모드와 후진 모드 자동 미분의 구분, 그리고 PyTorch의 벡터-야코비 곱(VJP) 기법이 연쇄 법칙을 어떻게 효율적으로 구현하는지를 다룬다.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. `.backward()`와 `torch.autograd.grad`로 경사를 계산한다
2. 야코비 행렬과 벡터-야코비 곱(VJP)의 틀을 이해한다
3. 전진 모드(JVP)와 후진 모드(VJP) 자동 미분을 구분한다
4. 신경망 학습에 후진 모드가 선호되는 이유를 설명한다
5. 두 모드 모두로 전체 야코비 행렬을 계산한다

---

## 2. 수학적 기초

### 연쇄 법칙

자동 미분은 연쇄 법칙을 체계적으로 적용한다. 합성 함수 $y = f(g(x))$에 대해 다음과 같다.

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

연산의 수열 $x \rightarrow z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow z_n \rightarrow y$에 대해서는 다음과 같다.

$$\frac{dy}{dx} = \frac{dy}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdot \ldots \cdot \frac{dz_2}{dz_1} \cdot \frac{dz_1}{dx}$$

이 행렬 곱들을 계산하는 **순서** 가 자동 미분의 모드를 결정한다.

### 야코비 행렬

함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$에 대해 **야코비안** 은 모든 1차 편도함수를 모은 행렬이다.

$$J = \frac{\partial f}{\partial x} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

합성 $y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$에 대해 전체 야코비안은 다음과 같다.

$$J = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$$

---

## 3. PyTorch에서 경사 계산하기

### 스칼라 손실: `.backward()` 메서드

스칼라 손실 $L: \mathbb{R}^n \rightarrow \mathbb{R}$에 대해 `L.backward()`를 호출하면 `requires_grad=True`인 모든 잎 텐서 $x$에 대한 경사 $\nabla_x L$이 계산된다.

$$\nabla_x L = \left[\frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, \ldots, \frac{\partial L}{\partial x_n}\right]^T$$

```python
import torch

torch.manual_seed(0)

x = torch.randn(3, requires_grad=True)
print(f"x: {x}")

# 순전파: loss = sum(x²)
loss = (x ** 2).sum()
print(f"loss: {loss}")

# 역전파: d(loss)/dx = 2x
loss.backward()

print(f"x.grad: {x.grad}")
print(f"Expected (2x): {2 * x.detach()}")
print(f"Match: {torch.allclose(x.grad, 2 * x.detach())}")
```

**출력:**
```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
loss: tensor(7.2274, grad_fn=<SumBackward0>)
x.grad: tensor([ 3.0820, -0.5868, -4.3576])
Expected (2x): tensor([ 3.0820, -0.5868, -4.3576])
Match: True
```

### 경사 저장: 잎 텐서와 비잎 텐서

기본적으로 경사는 잎 텐서에 대해서**만** 저장된다.

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x         # Non-leaf
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # tensor([ 8., 16., 24.]) — stored
print(f"y.grad: {y.grad}")  # None — not stored by default
```

잎이 아닌 텐서의 경사를 저장하려면 역전파 전에 `.retain_grad()`를 쓴다.

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x
y.retain_grad()       # Request gradient storage
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # tensor([ 8., 16., 24.])
print(f"y.grad: {y.grad}")  # tensor([ 4.,  8., 12.])
```

**출력:**

```
x.grad: tensor([ 8., 16., 24.])
y.grad: tensor([ 4.,  8., 12.])
```

**검증:** $y = 2x = [2, 4, 6]$이고 $z = \sum y_i^2$일 때 다음과 같다.

- $\frac{\partial z}{\partial y_i} = 2y_i$이므로 $\nabla_y z = [4, 8, 12]$이다
- $\frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_i} = 2y_i \cdot 2 = 4y_i$이므로 $\nabla_x z = [8, 16, 24]$이다

### `torch.autograd.grad` 인터페이스

좀 더 세밀하게 제어하려면 `torch.autograd.grad`를 쓴다. `.grad` 속성을 채우지 않고 경사를 곧바로 반환한다.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 경사들의 튜플을 반환한다
(grad_y,) = torch.autograd.grad(y, x)
print(f"dy/dx = 3x² = {grad_y}")  # tensor([12.])
```

**출력:**

```
dy/dx = 3x² = tensor([12.])
```

이는 고계 도함수(고계 경사 참고)를 다룰 때, 그리고 `retain_grad()`를 호출하지 않고 잎이 아닌 텐서의 경사가 필요할 때 특히 유용하다.

---

## 4. 벡터-야코비 곱의 틀

### 왜 VJP인가?

PyTorch의 역전파는 전체 야코비 행렬을 계산하지 **않는다.** 고차원 매개변수 공간에서는 그 비용이 감당할 수 없이 크기 때문이다. 대신 경사 기반 최적화에 필요한 전부인 **벡터-야코비 곱**(VJP)을 계산한다.

### 수학적 정의

야코비안이 $J \in \mathbb{R}^{m \times n}$인 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$과 상류 경사(수반) $\bar{y} \in \mathbb{R}^m$이 주어졌을 때 다음과 같다.

$$\text{VJP:} \quad \bar{x} = J^T \bar{y} \in \mathbb{R}^n$$

동등하게 $\bar{x}^T = \bar{y}^T J$이다. 이것이 행렬 형태의 연쇄 법칙이다. $\bar{y}$가 하류에서 온 경사를 나르고, $J^T$를 곱하는 것이 그것을 한 단계 더 상류로 전파한다.

### 스칼라 출력에서의 암묵적 VJP

출력이 스칼라이면($m = 1$) 야코비안이 행벡터(즉 경사)로 줄어들고 상류 "벡터"는 스칼라 $\bar{y} = 1$이 된다. `loss.backward()`에 인수가 필요 없는 이유가 이것이다.

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x ** 2).sum()

# 동등한 호출:
# loss.backward()
loss.backward(torch.tensor(1.0))   # Explicit v = 1

print(f"x.grad: {x.grad}")  # tensor([2., 4., 6.])
```

**출력:**

```
x.grad: tensor([2., 4., 6.])
```

### 스칼라가 아닌 출력에서의 명시적 VJP

스칼라가 아닌 출력에 대해서는 상류 경사 벡터 $\bar{y}$를 반드시 넘겨야 한다.

```python
import torch

x = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)
y = torch.sin(x)   # y: R³ → R³, elementwise

# y와 같은 모양의 경사 벡터 v를 반드시 넘겨야 한다
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(v)

# 원소별 sin에 대해 J = diag(cos(x))
# x.grad = J^T v = v ⊙ cos(x)
expected = v * torch.cos(x.detach())
print(f"x.grad:   {x.grad}")
print(f"Expected: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

`v` 없이 스칼라가 아닌 텐서에 `.backward()`를 호출하면 `RuntimeError`가 발생한다.

```python
x = torch.randn(3, requires_grad=True)
y = torch.sin(x)

try:
    y.backward()  # Fails — non-scalar output
except RuntimeError as e:
    print(f"Error: {e}")
```

**출력:**

```
Error: grad can be implicitly created only for scalar outputs
```

### 선형 변환에서의 VJP

$A \in \mathbb{R}^{m \times n}$인 선형 사상 $y = Ax$에 대해 야코비안은 단순히 $J = A$이다.

```python
import torch

A = torch.tensor([[2.0, 0.0, -1.0],
                  [0.5, 3.0,  1.0]])   # (2, 3)
x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
y = A @ x   # (2,)

v = torch.tensor([3.0, -1.0])
y.backward(v)

# x.grad = A^T @ v
expected = A.T @ v
print(f"x.grad: {x.grad}")
print(f"A^T @ v: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

**출력:**

```
x.grad: tensor([ 5.5000, -3.0000, -4.0000])
A^T @ v: tensor([ 5.5000, -3.0000, -4.0000])
Match: True
```

---

## 5. 전진 모드와 후진 모드 자동 미분

야코비안의 곱 $J = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$은 두 가지 순서로 계산할 수 있으며, 각각이 자동 미분의 한 모드를 정의한다.

### 전진 모드: 야코비-벡터 곱(JVP)

전진 모드는 도함수를 **입력에서 출력으로** 전파하며, 입력에서의 접벡터 $\dot{x}$에 대해 $\dot{y} = J \cdot \dot{x}$를 계산한다.

**계산 순서(오른쪽에서 왼쪽으로):**

$$J \cdot \dot{x} = J_L \cdot (J_{L-1} \cdot (\ldots \cdot (J_2 \cdot (J_1 \cdot \dot{x}))))$$

특정한 $\dot{x}$로 순전파를 한 번 하면 방향도함수 하나를 얻는다. 입력이 $n$개일 때 전체 야코비안을 얻으려면 **순전파 $n$번** 이 필요하다.

```python
import torch
from torch.autograd.functional import jvp

def f(x):
    """f(x) = [sin(x₁·x₂), x₁² + x₂]"""
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0])

# 접벡터 [1, 0]으로 JVP를 하면 야코비안의 첫 열을 얻는다
tangent = torch.tensor([1.0, 0.0])
output, jvp_result = jvp(f, (x,), (tangent,))

print(f"f(x) = {output}")
print(f"J @ [1,0] = {jvp_result}")  # First column of J
```

**출력:**

```
f(x) = tensor([0.9093, 3.0000])
J @ [1,0] = tensor([-0.8323,  2.0000])
```

### 후진 모드: 벡터-야코비 곱(VJP)

후진 모드는 도함수를 **출력에서 입력으로** 전파하며 $\bar{x} = J^T \bar{y}$를 계산한다.

**계산 순서(왼쪽에서 오른쪽으로):**

$$\bar{x}^T = ((((\bar{y}^T \cdot J_L) \cdot J_{L-1}) \cdot \ldots) \cdot J_1)$$

특정한 $\bar{y}$로 역전파를 한 번 하면 $J^T$의 한 행(동등하게 $J$의 한 열)을 얻는다. **스칼라** 출력($m = 1$)에서는 **역전파 한 번** 으로 전체 경사를 얻는다.

```python
import torch
from torch.autograd.functional import vjp

def f(x):
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 수반 벡터 [1, 0]으로 VJP를 하면 J의 첫 행을 얻는다
adjoint = torch.tensor([1.0, 0.0])
output, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(adjoint)[0]

print(f"f(x) = {output}")
print(f"[1,0] @ J = {vjp_result}")
```

### 신경망에 후진 모드를 쓰는 이유

딥러닝의 전형적인 상황은 수백만 개의 매개변수($n \gg 1$)가 하나의 스칼라 손실($m = 1$)로 대응되는 것이다.

| 모드 | 필요한 통과 횟수 | 복잡도 |
|------|-----------------|------------|
| 전진 | $n$(매개변수마다 한 번) | $O(n \cdot T)$ |
| **후진** | $m = 1$ | $O(T)$ |

여기서 $T$는 순전파 한 번의 비용이다. 후진 모드가 **압도적으로 효율적** 이며, PyTorch가 역전파를 기본으로 삼는 이유가 이것이다.

```python
import torch
import torch.nn as nn

# 매개변수가 약 20만 개인 신경망
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

logits = model(x)
loss = nn.functional.cross_entropy(logits, y)

# 한 번의 역전파가 모든 매개변수의 경사를 계산한다
loss.backward()

for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
```

**출력:**

```
0.weight: grad shape = torch.Size([256, 784])
0.bias: grad shape = torch.Size([256])
2.weight: grad shape = torch.Size([10, 256])
2.bias: grad shape = torch.Size([10])
```

### 메모리와 계산의 절충

후진 모드는 역전파를 위해 **중간 활성값을 저장해야** 하므로 신경망 깊이에 비례하는 메모리 부담이 생긴다.

| 측면 | 전진 모드 | 후진 모드 |
|--------|--------------|--------------|
| **전파 방향** | 입력 → 출력 | 출력 → 입력 |
| **계산 대상** | JVP: $J \dot{x}$ | VJP: $J^T \bar{y}$ |
| **효율적인 경우** | $n \ll m$ | $m \ll n$ |
| **전체 야코비안** | $n$번 통과 | $m$번 통과 |
| **추가 메모리** | $O(1)$ | $O(T)$(활성값) |

아주 깊은 신경망에서는 **경사 체크포인팅** 이 활성값을 저장하는 대신 역전파 중에 다시 계산함으로써 계산을 대가로 메모리를 아낀다.

```python
from torch.utils.checkpoint import checkpoint

def expensive_layer(x):
    return torch.relu(x @ x.T)

x = torch.randn(1000, 1000, requires_grad=True)

# 체크포인팅: 역전파 중에 활성값을 다시 계산한다
y = checkpoint(expensive_layer, x, use_reentrant=False)
```

---

## 6. 전체 야코비안 계산하기

### `torch.autograd.functional.jacobian` 사용하기

```python
import torch
from torch.autograd.functional import jacobian

def f(x):
    """f: R³ → R²"""
    return torch.stack([
        x[0] * x[1] + x[2],
        torch.sin(x[0]) + x[1]**2
    ])

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, x)

print(f"Input dim:  {x.shape[0]}")
print(f"Output dim: {f(x).shape[0]}")
print(f"Jacobian shape: {J.shape}")
print(f"Jacobian:\n{J}")
```

**출력:**

```
Input dim:  3
Output dim: 2
Jacobian shape: torch.Size([2, 3])
Jacobian:
tensor([[2.0000, 1.0000, 1.0000],
        [0.5403, 4.0000, 0.0000]])
```

### VJP로 야코비안 직접 구하기(행 단위)

원핫 수반 벡터 $e_i$로 역전파를 하면 야코비안의 $i$번 행을 뽑아낼 수 있다.

```python
import torch

def compute_jacobian_via_vjp(f, x):
    """거꾸로 방식 벡터-야코비 곱으로 야코비를 한 줄씩 짓는다."""
    y = f(x.detach().requires_grad_(True))
    m, n = y.numel(), x.numel()
    J = torch.zeros(m, n)
    
    for i in range(m):
        x_copy = x.detach().requires_grad_(True)
        y_copy = f(x_copy)
        
        v = torch.zeros(m)
        v[i] = 1.0
        
        y_copy.backward(v)
        J[i] = x_copy.grad
    
    return J

def f(x):
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

x = torch.tensor([2.0, 3.0])
J = compute_jacobian_via_vjp(f, x)
print(f"Jacobian:\n{J}")
# [[4, 0],    d(x₁²)/d(x₁, x₂)
#  [3, 2],    d(x₁x₂)/d(x₁, x₂)
#  [0, 6]]    d(x₂²)/d(x₁, x₂)
```

**출력:**

```
Jacobian:
tensor([[4., 0.],
        [3., 2.],
        [0., 6.]])
```

### JVP로 야코비안 직접 구하기(열 단위)

원핫 접벡터 $e_j$로 순전파를 하면 $j$번 열을 뽑아낼 수 있다.

```python
import torch
from torch.autograd.functional import jvp

def f(x):
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0])

J_col0 = jvp(f, (x,), (torch.tensor([1.0, 0.0]),))[1]
J_col1 = jvp(f, (x,), (torch.tensor([0.0, 1.0]),))[1]
J_forward = torch.stack([J_col0, J_col1], dim=1)

print(f"Full Jacobian (forward mode):\n{J_forward}")
```

**출력:**

```
Full Jacobian (forward mode):
tensor([[-0.8323, -0.4161],
        [ 2.0000,  1.0000]])
```

---

## 7. 주요 속성과 메서드

| 속성 / 메서드 | 설명 |
|--------------------|-------------|
| `x.requires_grad` | `x`에 대한 연산이 미분을 위해 추적되는지 여부 |
| `x.grad` | 누적된 경사(`.backward()` 후에 채워진다) |
| `x.grad_fn` | `x`를 만든 역전파 함수(잎 텐서에서는 `None`) |
| `x.is_leaf` | `x`가 잎 텐서이면 `True` |
| `x.backward(v)` | VJP를 계산하여 `.grad`에 누적한다 |
| `x.retain_grad()` | 잎이 아닌 `x`의 경사를 저장하도록 요청한다 |
| `x.detach()` | 데이터는 공유하되 그래프에서 분리된 텐서를 반환한다 |
| `torch.autograd.grad(y, x)` | `.grad`를 채우지 않고 경사를 계산한다 |

---

## 연습문제

**연습문제 1.**
점 $(1, 2)$에서 함수 $f(x_1, x_2) = (x_1^2 + x_2, x_1 x_2^3)$의 야코비 행렬을 계산하라. 그런 다음 $\bar{y} = (1, 1)$에 대한 VJP를 계산하라.

??? success "연습문제 1 풀이"
    야코비안은 다음과 같다.

    $$
    J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 2x_1 & 1 \\ x_2^3 & 3x_1 x_2^2 \end{bmatrix}
    $$

    $(1, 2)$에서 $J = \begin{bmatrix} 2 & 1 \\ 8 & 12 \end{bmatrix}$이다.

    VJP는 $J^T \bar{y} = \begin{bmatrix} 2 & 8 \\ 1 & 12 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 10 \\ 13 \end{bmatrix}$이다.

---

**연습문제 2.**
신경망 학습에서 전진 모드보다 후진 모드 자동 미분이 선호되는 이유를 설명하라. 어떤 상황에서 전진 모드가 더 효율적인가?

??? success "연습문제 2 풀이"
    후진 모드 자동 미분은 역전파 한 번으로 스칼라 손실의 $n$개 매개변수 전부에 대한 경사를 계산하며, 연산 횟수를 $T$라 할 때 비용이 $O(T)$이다. 전진 모드는 매개변수마다 한 번씩 총 $n$번의 통과가 필요하다.

    신경망은 매개변수가 수백만 개지만 손실은 스칼라 하나이므로 후진 모드가 압도적으로 효율적이다. 전진 모드는 입력이 아주 적고 출력이 많은 함수의 야코비안을 계산할 때 선호된다. 전진 모드는 입력 차원마다 한 번씩만 통과하면 되기 때문이다.

---

**연습문제 3.**
`torch.autograd.functional.jacobian`을 사용하여 $f(x_1, x_2) = (x_1^2, x_1 x_2, x_2^2)$의 전체 $3 \times 2$ 야코비안을 계산하는 PyTorch 코드를 작성하라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    from torch.autograd.functional import jacobian

    def f(x):
        return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

    x = torch.tensor([1.0, 2.0])
    J = jacobian(f, x)
    print(J)
    # tensor([[2., 0.],
    #         [2., 1.],
    #         [0., 4.]])
    ```

---

**연습문제 4.**
어떤 학생이 사이에 `optimizer.zero_grad()`를 호출하지 않고 `loss.backward()`를 연달아 두 번 썼다. `.grad` 속성에는 어떤 일이 생기며, PyTorch에서 이 동작이 기본값인 이유는 무엇인가?

??? success "연습문제 4 풀이"
    경사가 더해지면서 누적된다. 두 번째 `backward()` 호출 후 각 매개변수의 `.grad`는 두 번의 통과에서 나온 경사의 합을 담는다. 이것이 기본값인 이유는 경사 누적이 큰 배치 크기를 흉내 내는 데 유용하기 때문이다. 여러 미니배치에 대해 `backward()`를 호출하고 `optimizer.step()`은 한 번만 호출하면 사실상 더 큰 실효 배치에 대해 평균을 내는 셈이 된다. 다만 표준적인 학습 루프에서는 오래된 경사가 갱신을 망치지 않도록 매 `backward()` 전에 `optimizer.zero_grad()`를 호출해야 한다.

## 정리하며

| 개념 | 핵심 |
|---------|------------|
| **경사** | 스칼라 $L$에 대해 후진 모드 자동 미분으로 계산한 $\nabla_x L$ |
| **VJP** | $J^T \bar{y}$ — PyTorch가 역전파 중에 계산하는 것 |
| **JVP** | $J \dot{x}$ — 전진 모드. 입력이 적을 때 효율적이다 |
| **후진 모드** | 스칼라 손실에 대해 역전파 한 번. 비용 $O(T)$ |
| **전진 모드** | 순전파 $n$번이 필요. 추가 메모리 $O(1)$ |
| **전체 야코비안** | 역전파 $m$번(후진) 또는 순전파 $n$번 |
| **경사 저장** | 기본적으로 잎 텐서만 `.grad`를 저장한다 |

**참고 문헌**

- Baydin, A.G., et al. (2018). Automatic Differentiation in Machine Learning: A Survey. *JMLR*.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*.
- PyTorch Autograd Documentation: [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
