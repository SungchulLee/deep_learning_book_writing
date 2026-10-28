# 경사 계산

PyTorch의 autograd 엔진은 딥러닝에서 자동 미분의 토대이다. 텐서에 대한 연산을 기록하고 **후진 모드 자동 미분**(역전파)으로 경사를 계산하는 테이프 기반 체계를 제공한다. 이 절에서는 경사 계산의 수학적 틀, 전진 모드와 후진 모드 자동 미분의 구분, 그리고 PyTorch의 벡터-야코비안 곱(VJP) 기법이 연쇄 법칙을 어떻게 효율적으로 구현하는지를 다룬다.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

1. `.backward()`와 `torch.autograd.grad`로 경사를 계산한다
2. 야코비안 행렬과 벡터-야코비안 곱(VJP)의 틀을 이해한다
3. 전진 모드(JVP)와 후진 모드(VJP) 자동 미분을 구분한다
4. 뉴럴 넷트웍 학습에 후진 모드가 선호되는 이유를 설명한다
5. 두 모드 모두로 전체 야코비안 행렬을 계산한다

---

## 2. 수학적 기초

<div class="defn" markdown>

### 정의 1. 야코비안 행렬 { .dfn }

미분 가능한 $f : \mathbb{R}^n \to \mathbb{R}^m$에 대해 모든 1차 편도함수를 모은 행렬

$$
J_f(x) = \frac{\partial f}{\partial x} = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

을 $f$의 **야코비안**이라 한다. $(i,j)$칸은 $j$번째 입력을 조금 흔들었을 때 $i$번째 출력이 얼마나 따라 움직이는지다.

</div>

<div class="thmbox" markdown>

### 정리 1. 연쇄 법칙 — 합성의 야코비안은 야코비안의 곱이다 { .thm }

$y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$이고 각 $f_\ell$이 미분 가능하면

$$
J = J_L \, J_{L-1} \cdots J_1
$$

이다. 여기서 $J_\ell$은 $f_\ell$을 그 자리의 입력 $z_{\ell-1}$에서 잰 야코비안이다.

</div>

??? proof "증명"

    $L = 2$에서 보이면 나머지는 귀납이다. $z = f_1(x)$, $y = f_2(z)$라 하고 성분으로 적는다. 다변수 연쇄 법칙에서

    $$
    \frac{\partial y_i}{\partial x_j} = \sum_{k} \frac{\partial y_i}{\partial z_k}\,\frac{\partial z_k}{\partial x_j}
    $$

    이다. 오른쪽은 $(i,k)$칸이 $\partial y_i / \partial z_k$인 행렬과 $(k,j)$칸이 $\partial z_k / \partial x_j$인 행렬의 **곱의 $(i,j)$칸**과 정확히 같은 꼴이다. 곧 $J = J_2 J_1$이다.

    $L-1$개까지 성립한다고 하면 $g = f_{L-1}\circ\cdots\circ f_1$에 대해 $J_g = J_{L-1}\cdots J_1$이고, 위의 두 겹 결과를 $f_L$과 $g$에 쓰면 $J = J_L J_g = J_L J_{L-1} \cdots J_1$이다. $\square$

**곱하는 차례가 자동 미분의 모드를 가른다.** 행렬 곱은 결합법칙이 성립하므로 $J_L(J_{L-1}(\cdots J_1))$로 묶든 $((J_L J_{L-1})\cdots)J_1$로 묶든 **결과는 같다.** 그러나 드는 비용이 다르다. 오른쪽부터 묶는 것이 전진 모드, 왼쪽부터 묶는 것이 후진 모드이며, 로스처럼 출력이 스칼라 하나인 경우($m = 1$) 후진 모드가 훨씬 싸다. 딥러닝이 역전파를 쓰는 까닭이 이것이다.

---

## 3. PyTorch에서 경사 계산하기

### 스칼라 로스: `.backward()` 메서드

스칼라 로스 $L: \mathbb{R}^n \rightarrow \mathbb{R}$에 대해 `L.backward()`를 호출하면 `requires_grad=True`인 모든 잎 텐서 $x$에 대한 경사 $\nabla_x L$이 계산된다.

$$\nabla_x L = \left[\frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, \ldots, \frac{\partial L}{\partial x_n}\right]^T$$

```python
import torch

torch.manual_seed(0)

x = torch.randn(3, requires_grad=True)
print(f"x: {x}")

# 순전파: loss = sum(x²)
loss = (x ** 2).sum()
print("loss:", loss)   # f-string을 쓰면 0차원 텐서가 맨 실수로 찍혀 grad_fn이 사라진다

# 역전파: d(loss)/dx = 2x
loss.backward()

print(f"x.grad: {x.grad}")
print(f"Expected (2x): {2 * x.detach()}")
print(f"Match: {torch.allclose(x.grad, 2 * x.detach())}")
```

**출력:**
```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
loss: tensor(7.2079, grad_fn=<SumBackward0>)
x.grad: tensor([ 3.0820, -0.5869, -4.3576])
Expected (2x): tensor([ 3.0820, -0.5869, -4.3576])
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

**출력:**

```
x.grad: tensor([ 8., 16., 24.])
y.grad: None
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

## 4. 벡터-야코비안 곱의 틀

### 왜 VJP인가?

PyTorch의 역전파는 전체 야코비안 행렬을 계산하지 **않는다.** 고차원 파라메터 공간에서는 그 비용이 감당할 수 없이 크기 때문이다. 대신 경사 기반 최적화에 필요한 전부인 **벡터-야코비안 곱**(VJP)을 계산한다.

### 수학적 정의

야코비안이 $J \in \mathbb{R}^{m \times n}$인 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$과 상류 경사(수반) $\bar{y} \in \mathbb{R}^m$이 주어졌을 때 다음과 같다.

$$\text{VJP:} \quad \bar{x} = J^T \bar{y} \in \mathbb{R}^n$$

동등하게 $\bar{x}^T = \bar{y}^T J$이다. 이것이 행렬 형태의 연쇄 법칙이다. $\bar{y}$가 하류에서 온 경사를 나르고, $J^T$를 곱하는 것이 그것을 한 단계 더 상류로 전파한다.

### 스칼라 출력에서의 암묵적 VJP

출력이 스칼라가면($m = 1$) 야코비안이 행벡터(즉 경사)로 줄어들고 상류 "벡터"는 스칼라 $\bar{y} = 1$이 된다. `loss.backward()`에 인수가 필요 없는 이유가 이것이다.

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

**출력:**

```
x.grad:   tensor([0.0878, 0.5403, 0.0088])
Expected: tensor([0.0878, 0.5403, 0.0088])
Match: True
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

### 후진 모드: 벡터-야코비안 곱(VJP)

후진 모드는 도함수를 **출력에서 입력으로** 전파하며 $\bar{x} = J^T \bar{y}$를 계산한다.

**계산 순서(왼쪽에서 오른쪽으로):**

$$\bar{x}^T = ((((\bar{y}^T \cdot J_L) \cdot J_{L-1}) \cdot \ldots) \cdot J_1)$$

일반적인 $\bar{y}$로는 $J$의 행들의 가중합을 얻는다. $\bar{y}=e_i$인 기저 벡터를 쓰면 $J^T$의 i번 열, 즉 $J$의 i번 행을 열벡터 형태로 얻는다. **스칼라** 출력($m = 1$)에서는 **역전파 한 번** 으로 전체 경사를 얻는다.

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
# torch.autograd.functional.vjp는 수반 벡터를 세 번째 인자로 직접 받고
# (출력, VJP 결과)를 돌려준다. 함수를 돌려주는 torch.func.vjp와 혼동하기 쉽다.
adjoint = torch.tensor([1.0, 0.0])
output, vjp_result = vjp(f, x, adjoint)

print(f"f(x) = {output}")
print(f"[1,0] @ J = {vjp_result}")
```

**출력:**

```
f(x) = tensor([0.9093, 3.0000])
[1,0] @ J = tensor([-0.8323, -0.4161])
```

### 뉴럴 넷트웍에 후진 모드를 쓰는 이유

딥러닝의 전형적인 상황은 수백만 개의 파라메터($n \gg 1$)가 하나의 스칼라 로스($m = 1$)로 대응되는 것이다.

| 모드 | 필요한 통과 횟수 | 복잡도 |
|------|-----------------|------------|
| 전진 | $n$(파라메터마다 한 번) | $O(n \cdot T)$ |
| **후진** | $m = 1$ | $O(T)$ |

여기서 $T$는 순전파 한 번의 비용이다. 후진 모드가 **압도적으로 효율적** 이며, PyTorch가 역전파를 기본으로 삼는 이유가 이것이다.

```python
import torch
import torch.nn as nn

# 파라메터가 약 20만 개인 뉴럴 넷트웍
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

logits = model(x)
loss = nn.functional.cross_entropy(logits, y)

# 한 번의 역전파가 모든 파라메터의 경사를 계산한다
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

후진 모드는 역전파를 위해 **중간 엑티베이션을 저장해야** 하므로 뉴럴 넷트웍 깊이에 비례하는 메모리 부담이 생긴다.

| 측면 | 전진 모드 | 후진 모드 |
|--------|--------------|--------------|
| **전파 방향** | 입력 → 출력 | 출력 → 입력 |
| **계산 대상** | JVP: $J \dot{x}$ | VJP: $J^T \bar{y}$ |
| **효율적인 경우** | $n \ll m$ | $m \ll n$ |
| **전체 야코비안** | $n$번 통과 | $m$번 통과 |
| **추가 메모리** | 한 방향의 접선과 현재 연산 상태 | 역전파용 저장 텐서(구조에 따라 다름) |

아주 깊은 뉴럴 넷트웍에서는 **엑티베이션 체크포인팅** 이 엑티베이션을 저장하는 대신 역전파 중에 다시 계산함으로써 계산을 대가로 메모리를 아낀다.

```python
from torch.utils.checkpoint import checkpoint

def expensive_layer(x):
    return torch.relu(x @ x.T)

x = torch.randn(1000, 1000, requires_grad=True)

# 체크포인팅: 역전파 중에 엑티베이션을 다시 계산한다
y = checkpoint(expensive_layer, x, use_reentrant=False)
```

---

## 6. 셈을 재어 본다

위의 표는 후진 모드가 $O(T)$, 전진 모드가 $O(n \cdot T)$라고 적는다. 파라미터가
4,225개면 4,225배 차이가 나야 한다는 말이다. 재어 보면 **60배**다.

셈이 틀린 것이 아니라 **상수가 숨어 있다.** 아래 코드는 이 쪽 옆의
`gradient_computation.py`에 있다.

```python
"""전진 모드와 후진 모드 — 셈으로 말고 재어 본다.

본문의 표는 후진 모드가 $O(T)$, 전진 모드가 $O(n \\cdot T)$ 라고
적는다. 파라미터가 4,225 개면 4,225 배 차이가 나야 한다는 말이다.

재어 보면 55 배다. 셈이 틀린 것이 아니라 **상수가 숨어 있다.**
전진 모드도 한 번에 여러 방향을 묶어 처리하므로 통과 횟수만큼
곱해지지 않는다.

그래도 자릿수가 같고, 방향을 뒤집으면 **정확히 거꾸로** 뒤집힌다.

실행: python gradient_computation.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacfwd, jacrev


def header(title):
    print(f"\n=== {title} ===")


def timeit(fn, rep=3):
    fn()                                   # 예열
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


# ========================================================================
# 1. 딥러닝의 꼴 — 입력 많고 출력 하나
# ========================================================================

def deep_learning_shape():
    header("1. 파라미터는 많고 로스는 하나")
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    n_param = sum(p.numel() for p in net.parameters())
    x, y = torch.randn(8, 64), torch.randn(8, 1)
    params = {k: v.detach() for k, v in net.named_parameters()}

    def loss_fn(p):
        return F.mse_loss(functional_call(net, p, (x,)), y)

    rev = timeit(lambda: jacrev(loss_fn)(params))
    fwd = timeit(lambda: jacfwd(loss_fn)(params))
    print(f"  파라미터 {n_param:,}개, 로스는 홑값 하나")
    print(f"  jacrev (후진) {rev:8.2f} ms")
    print(f"  jacfwd (전진) {fwd:8.2f} ms")
    print(f"  전진이 {fwd / rev:.0f} 배 느리다")
    print(f"  표의 셈으로는 {n_param:,} 배여야 하는데 {fwd / rev:.0f} 배다")
    print("  전진 모드도 여러 방향을 묶어 처리하므로 통과 횟수만큼 안 곱해진다")


# ========================================================================
# 2. 거꾸로 — 입력 하나에 출력 많기
# ========================================================================

def the_other_shape():
    header("2. 거꾸로 뒤집으면")
    torch.manual_seed(0)
    A = torch.randn(2048, 1)

    def h(t):
        return (A @ t.view(1, 1)).squeeze(1).tanh()

    xin = torch.randn(1)
    rev = timeit(lambda: jacrev(h)(xin))
    fwd = timeit(lambda: jacfwd(h)(xin))
    print(f"  입력 1 개 -> 출력 2048 개")
    print(f"  jacrev (후진) {rev:8.2f} ms")
    print(f"  jacfwd (전진) {fwd:8.2f} ms")
    print(f"  이번에는 전진이 {rev / fwd:.0f} 배 빠르다")
    print("  고르는 기준은 라이브러리가 아니라 함수의 모양이다")


# ========================================================================
# 3. 후진 모드가 치르는 값 — 활성값을 들고 있어야 한다
# ========================================================================

def the_memory_cost():
    header("3. 후진 모드가 치르는 값")
    torch.manual_seed(0)
    net = nn.Sequential(*[nn.Sequential(nn.Linear(256, 256), nn.ReLU())
                          for _ in range(8)])
    x = torch.randn(64, 256)
    with torch.no_grad():
        no_graph = net(x)
    with_graph = net(x)
    print("  no_grad 로 돌린 결과 grad_fn :", no_graph.grad_fn)
    print("  그냥 돌린 결과 grad_fn       :",
          type(with_graph.grad_fn).__name__)
    act = 64 * 256 * 4 * 8 * 2 / 1e6
    par = sum(p.numel() * 4 for p in net.parameters()) / 1e6
    print(f"  파라미터 {par:.2f} MB   층마다 저장되는 활성값 대략 {act:.2f} MB")
    print("  전진 모드는 활성값을 안 들고 있어도 된다 — 앞으로만 가니까")
    print("  후진 모드의 속도는 공짜가 아니라 메모리와 바꾼 것이다")


# ========================================================================
# 4. 야코비를 모으는 두 방향
# ========================================================================

def two_directions():
    header("4. 야코비를 줄로 모으기 vs 열로 모으기")

    def f(t):
        return torch.stack([t[0] ** 2, t[0] * t[1], t[1] ** 2])

    x = torch.tensor([1.0, 2.0])
    print("  f: R^2 -> R^3,  야코비는 3x2")
    print("  jacrev (줄 단위, 출력 3 번):")
    for r in jacrev(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    print("  jacfwd (열 단위, 입력 2 번):")
    for r in jacfwd(f)(x).tolist():
        print("    ", [round(v, 4) for v in r])
    print("  같은가:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))
    print("  손으로: [[2x, 0], [y, x], [0, 2y]] = [[2,0],[2,1],[0,4]]")
    print("  출력 3 > 입력 2 이므로 여기서는 전진이 한 번 덜 돈다")


def main():
    deep_learning_shape()
    the_other_shape()
    the_memory_cost()
    two_directions()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 파라미터는 많고 로스는 하나 ===
  파라미터 4,225개, 로스는 홑값 하나
  jacrev (후진)     0.24 ms
  jacfwd (전진)    14.62 ms
  전진이 60 배 느리다
  표의 셈으로는 4,225 배여야 하는데 60 배다
  전진 모드도 여러 방향을 묶어 처리하므로 통과 횟수만큼 안 곱해진다

=== 2. 거꾸로 뒤집으면 ===
  입력 1 개 -> 출력 2048 개
  jacrev (후진)     5.42 ms
  jacfwd (전진)     0.11 ms
  이번에는 전진이 48 배 빠르다
  고르는 기준은 라이브러리가 아니라 함수의 모양이다

=== 3. 후진 모드가 치르는 값 ===
  no_grad 로 돌린 결과 grad_fn : None
  그냥 돌린 결과 grad_fn       : ReluBackward0
  파라미터 2.11 MB   층마다 저장되는 활성값 대략 1.05 MB
  전진 모드는 활성값을 안 들고 있어도 된다 — 앞으로만 가니까
  후진 모드의 속도는 공짜가 아니라 메모리와 바꾼 것이다

=== 4. 야코비를 줄로 모으기 vs 열로 모으기 ===
  f: R^2 -> R^3,  야코비는 3x2
  jacrev (줄 단위, 출력 3 번):
     [2.0, 0.0]
     [2.0, 1.0]
     [0.0, 4.0]
  jacfwd (열 단위, 입력 2 번):
     [2.0, 0.0]
     [2.0, 1.0]
     [0.0, 4.0]
  같은가: True
  손으로: [[2x, 0], [y, x], [0, 2y]] = [[2,0],[2,1],[0,4]]
  출력 3 > 입력 2 이므로 여기서는 전진이 한 번 덜 돈다
```

### 읽기

```
파라미터 4,225개, 로스는 홑값 하나
jacrev (후진)     0.24 ms
jacfwd (전진)    14.62 ms      전진이 60 배 느리다

입력 1 개 -> 출력 2048 개
jacrev (후진)     5.42 ms
jacfwd (전진)     0.11 ms      전진이 48 배 빠르다
```

**방향이 정확히 뒤집힌다.** 같은 두 함수가 함수의 모양에 따라 60배 손해를 보기도
하고 48배 이득을 보기도 한다.

$n$배가 아니라 60배인 까닭은 전진 모드가 **여러 방향을 묶어** 처리하기 때문이다.
통과 횟수가 $n$이어도 각 통과가 통째로 되풀이되는 것은 아니다. 그래도 자릿수의
이야기는 맞는다 — 파라미터가 수백만 개인 실제 모델에서는 전진 모드로 학습하는
것이 불가능하다.

### 후진 모드가 치르는 값

```
파라미터 2.11 MB   층마다 저장되는 활성값 대략 1.05 MB
```

후진 모드가 빠른 것은 공짜가 아니다. **역전파가 쓸 중간값을 들고 있어야** 한다.
전진 모드는 앞으로만 가므로 그럴 필요가 없다.

「경사 추적을 멈추는 시점」에서 `no_grad`가 아끼는 메모리가 바로 이것이고,
위 출력의 `grad_fn: None`이 그 표시다.

---

## 7. 전체 야코비안 계산하기

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

## 8. 주요 속성과 메서드

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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$f(x_1, x_2) = (x_1^2, x_1 x_2, x_2^2)$의 $3 \times 2$ 야코비안을 손으로 적고
`jacrev`와 `jacfwd`로 각각 확인하라. 둘이 같은가?

</div>

??? success "풀이"
    $$
    J = \begin{bmatrix}
    2x_1 & 0 \\ x_2 & x_1 \\ 0 & 2x_2
    \end{bmatrix}
    \;\xrightarrow{(1,2)}\;
    \begin{bmatrix} 2 & 0 \\ 2 & 1 \\ 0 & 4 \end{bmatrix}
    $$

    ```python
    from torch.func import jacrev, jacfwd

    def f(t):
        return torch.stack([t[0] ** 2, t[0] * t[1], t[1] ** 2])

    x = torch.tensor([1.0, 2.0])
    print("  jacrev:", jacrev(f)(x).tolist())
    print("  jacfwd:", jacfwd(f)(x).tolist())
    print("  같은가:", torch.allclose(jacrev(f)(x), jacfwd(f)(x)))
    ```

    ```
      jacrev: [[2.0, 0.0], [2.0, 1.0], [0.0, 4.0]]
      jacfwd: [[2.0, 0.0], [2.0, 1.0], [0.0, 4.0]]
      같은가: True
    ```

    **값은 같고 구하는 길만 다르다.** `jacrev`는 출력 3개에 대해 VJP를 세 번,
    `jacfwd`는 입력 2개에 대해 JVP를 두 번 부른다.

    출력(3)이 입력(2)보다 많으니 **여기서는 전진이 한 번 덜 돈다.** 6절에서
    본 규칙 그대로다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점 $(1, 2)$에서 $f(x_1, x_2) = (x_1^2 + x_2,\ x_1 x_2^3)$의 야코비안을 구하고,
$\bar{y} = (1, 1)$에 대한 VJP를 **손으로 셈한 뒤** 확인하라.

</div>

??? success "풀이"
    $$
    J = \begin{bmatrix} 2x_1 & 1 \\ x_2^3 & 3x_1 x_2^2 \end{bmatrix}
    \;\xrightarrow{(1,2)}\;
    \begin{bmatrix} 2 & 1 \\ 8 & 12 \end{bmatrix},
    \qquad
    J^\top \bar{y} = \begin{bmatrix} 2 & 8 \\ 1 & 12 \end{bmatrix}
    \begin{bmatrix} 1 \\ 1 \end{bmatrix}
    = \begin{bmatrix} 10 \\ 13 \end{bmatrix}
    $$

    ```python
    def f(t):
        return torch.stack([t[0] ** 2 + t[1], t[0] * t[1] ** 3])

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = f(x)
    y.backward(torch.tensor([1.0, 1.0]))
    print("  야코비:", jacrev(f)(x.detach()).tolist())
    print("  VJP   :", x.grad.tolist())
    ```

    ```
      야코비: [[2.0, 1.0], [8.0, 12.0]]
      VJP   : [10.0, 13.0]
    ```

    손으로 센 것과 맞는다.

    VJP가 $J^\top \bar{y}$이지 $J \bar{y}$가 아니라는 것을 모양으로도 확인할 수
    있다. $J$가 $2 \times 2$라 여기서는 둘 다 되지만, $J$가 $3 \times 2$였다면
    $J\bar{y}$는 모양이 안 맞는다. 「선형 사상의 VJP」에서 본 그 점검이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
6절에서 전진 모드가 60배 느렸다. **파라미터 수를 바꿔 가며** 그 배수가 어떻게
변하는지 재고, 이론의 $O(n)$과 견주어라.

</div>

??? success "풀이"
    ```python
    import time
    from torch.func import functional_call, jacfwd, jacrev

    def timeit(fn, rep=3):
        fn()
        ts = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    print(f"  {'폭':>5} {'파라미터':>10} {'후진 ms':>9} {'전진 ms':>9} {'배수':>7}")
    for width in (8, 16, 32, 64):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(width, width), nn.Tanh(),
                            nn.Linear(width, 1))
        n = sum(p.numel() for p in net.parameters())
        x, y = torch.randn(8, width), torch.randn(8, 1)
        params = {k: v.detach() for k, v in net.named_parameters()}

        def loss_fn(p):
            return F.mse_loss(functional_call(net, p, (x,)), y)

        r = timeit(lambda: jacrev(loss_fn)(params))
        f = timeit(lambda: jacfwd(loss_fn)(params))
        print(f"  {width:>5} {n:>10,} {r:>9.2f} {f:>9.2f} {f / r:>7.1f}")
    ```

    ```
         폭     파라미터    후진 ms    전진 ms      배수
          8         81      0.23      0.33     1.4
         16        289      0.22      0.57     2.6
         32      1,089      0.22      1.78     8.2
         64      4,225      0.24     14.72    61.8
    ```

    **후진 쪽은 거의 일정하고(0.22~0.24 ms) 전진 쪽만 자란다.** 그것이 표가
    말하는 바다 — 후진은 $O(T)$, 전진은 $O(n \cdot T)$.

    배수를 보면 1.4 → 2.6 → 8.2 → 61.8이다. 파라미터가 네 배가 될 때마다
    대략 두 배, 세 배, 일곱 배씩 는다. **$n$에 비례하지는 않지만 $n$과 함께
    자라고, 커질수록 가팔라진다.**

    $n$에 못 미치는 까닭은 6절에서 적은 대로다. 그리고 작은 모델에서는 고정
    비용이 섞여 있어 배수가 더 눌린다 — 폭 8에서는 1.4배뿐이다.

    실제 모델로 외삽하면 뜻이 분명해진다. 파라미터가 백만 개면 이 추세로도
    수천 배이고, 그것이 **전진 모드로는 학습이 불가능하다**는 말의 내용이다.
    전진 모드가 쓰이는 자리는 입력이 적은 민감도 분석이나, 후진 모드와 섞어
    쓰는 고차 미분이다.

## 정리하며

| 개념 | 핵심 |
|---------|------------|
| **경사** | 스칼라 $L$에 대해 후진 모드 자동 미분으로 계산한 $\nabla_x L$ |
| **VJP** | $J^T \bar{y}$ — PyTorch가 역전파 중에 계산하는 것 |
| **JVP** | $J \dot{x}$ — 전진 모드. 입력이 적을 때 효율적이다 |
| **후진 모드** | 스칼라 로스에 대해 역전파 한 번. 비용 $O(T)$ |
| **전진 모드** | 전체 야코비안에 방향별 $n$번 통과. 현재 값과 접선의 메모리는 필요 |
| **전체 야코비안** | 역전파 $m$번(후진) 또는 순전파 $n$번 |
| **경사 저장** | 기본적으로 잎 텐서만 `.grad`를 저장한다 |

**참고 문헌**

- Baydin, A.G., et al. (2018). Automatic Differentiation in Machine Learning: A Survey. *JMLR*.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*.
- PyTorch Autograd Documentation: [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
