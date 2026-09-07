# Autograd 바닥부터 만들기

이 스크립트는 autograd를 바닥부터 만드는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""밑바닥부터 짜는 자동 미분."""
# ---
# title: "밑바닥부터 짜는 자동 미분"
# description: "파이썬으로 가장 단출한 자동 미분 엔진 짓기 — 셈 그래프,
#               역전파, 경사 누적, PyTorch와의 비교"
# ---
#
# PyTorch의 autograd 체계는 강력하지만, 그 속에서 어떻게 동작하는지 이해하는 것이
# 딥러닝을 제대로 익히는 데 필수적이다. 이 스크립트는 최소한의
# 자동 미분 엔진을 바닥부터 만든 뒤 PyTorch와 대조하여 검증한다.
#
#   1부 – NumberWithGrad: 덧셈/곱셈을 지원하는 스칼라 자동 미분
#   2부 – 경사 누적과 여러 경로를 가진 그래프
#   3부 – 뺄셈, 나눗셈, 거듭제곱으로 확장
#   4부 – TensorWithGrad: numpy를 이용한 벡터화된 자동 미분
#   5부 – PyTorch autograd와 대조 검증
#   6부 – 우리 엔진으로 작은 신경망 만들기
#
# 출처 각색: O'Reilly "Deep Learning from Scratch" 6장

from __future__ import annotations
from typing import Union, List, Optional
import numpy as np

Numberable = Union[float, int]


# =====================================================================
# 1부 – 스칼라 자동 미분
# =====================================================================
print("=" * 60)
print("Part 1: Scalar Automatic Differentiation")
print("=" * 60)


def ensure_number(num):
    """날 파이썬 수를 NumberWithGrad으로 감싼다."""
    if isinstance(num, NumberWithGrad):
        return num
    return NumberWithGrad(num)


class NumberWithGrad:
    """자동 미분을 위해 제 셈 그래프를 좇는 홑값.

    인스턴스마다 다음을 담는다.
      - num:         홑값
      - grad:        쌓인 기울기(.backward()이 채운다)
      - depends_on:  어버이 NumberWithGrad 객체의 목록
      - creation_op: 이 마디를 만든 셈을 가리키는 글자열 표
    """

    def __init__(
        self,
        num: Numberable,
        depends_on: Optional[List] = None,
        creation_op: str = "",
    ):
        self.num = num
        self.grad: Optional[float] = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op

    # ── 순전파 연산자 ────────────────────────────────────────────────

    def __add__(self, other: Numberable) -> NumberWithGrad:
        other = ensure_number(other)
        return NumberWithGrad(
            self.num + other.num,
            depends_on=[self, other],
            creation_op="add",
        )

    def __radd__(self, other: Numberable) -> NumberWithGrad:
        return self.__add__(other)

    def __mul__(self, other: Numberable) -> NumberWithGrad:
        other = ensure_number(other)
        return NumberWithGrad(
            self.num * other.num,
            depends_on=[self, other],
            creation_op="mul",
        )

    def __rmul__(self, other: Numberable) -> NumberWithGrad:
        return self.__mul__(other)

    def __sub__(self, other: Numberable) -> NumberWithGrad:
        other = ensure_number(other)
        return NumberWithGrad(
            self.num - other.num,
            depends_on=[self, other],
            creation_op="sub",
        )

    def __truediv__(self, other: Numberable) -> NumberWithGrad:
        other = ensure_number(other)
        return NumberWithGrad(
            self.num / other.num,
            depends_on=[self, other],
            creation_op="div",
        )

    def __pow__(self, exponent: Numberable) -> NumberWithGrad:
        exponent = ensure_number(exponent)
        return NumberWithGrad(
            self.num ** exponent.num,
            depends_on=[self, exponent],
            creation_op="pow",
        )

    def __neg__(self) -> NumberWithGrad:
        return NumberWithGrad(-self.num, depends_on=[self], creation_op="neg")

    def __repr__(self) -> str:
        return f"NumberWithGrad({self.num})"

    # ── 역전파 ───────────────────────────────────────────────────────

    def backward(self, backward_grad: Optional[float] = None) -> None:
        """셈 그래프를 따라 기울기를 되부르며 퍼뜨린다.

        사슬 법칙을 쓴다:  dL/dx = dL/dy * dy/dx
        여기서 y은 이 마디의 값이고 x은 어버이의 값이다.
        """
        if backward_grad is None:
            # 역전파의 뿌리 — 출력의 자기 자신에 대한 경사는 1이다
            self.grad = 1.0
        else:
            # 경사 누적(팬아웃/여러 경로 그래프를 처리한다)
            if self.grad is None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad

        # 생성 연산에 따라 부모로 전파한다
        if self.creation_op == "add":
            # d(a + b)/da = 1,  d(a + b)/db = 1
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)

        elif self.creation_op == "mul":
            # d(a * b)/da = b,  d(a * b)/db = a
            self.depends_on[0].backward(self.grad * self.depends_on[1].num)
            self.depends_on[1].backward(self.grad * self.depends_on[0].num)

        elif self.creation_op == "sub":
            # d(a - b)/da = 1,  d(a - b)/db = -1
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(-self.grad)

        elif self.creation_op == "div":
            # d(a / b)/da = 1/b,  d(a / b)/db = -a/b^2
            a, b = self.depends_on
            a.backward(self.grad / b.num)
            b.backward(-self.grad * a.num / (b.num ** 2))

        elif self.creation_op == "pow":
            # d(a^n)/da = n * a^(n-1),  d(a^n)/dn = a^n * ln(a)
            base, exp = self.depends_on
            base.backward(self.grad * exp.num * base.num ** (exp.num - 1))
            if base.num > 0:
                exp.backward(self.grad * self.num * np.log(base.num))

        elif self.creation_op == "neg":
            self.depends_on[0].backward(-self.grad)


# 시연: 간단한 사슬  c = (a * 4) + 3
a = NumberWithGrad(3)
b = a * 4          # b = 12
c = b + 3          # c = 15
c.backward()
print(f"  c = (a * 4) + 3,  a = 3")
print(f"  dc/da = {a.grad}   (expected 4)")
print(f"  dc/db = {b.grad}   (expected 1)")
print()


# =====================================================================
# 2부 – 경사 누적(여러 경로를 가진 그래프)
# =====================================================================
print("=" * 60)
print("Part 2: Multi-path computation graphs")
print("=" * 60)

# e = c * d 이며 c도 a에 의존하고 d도 a에 의존한다
# a가 두 번 쓰인다 → 경사가 누적되어야 한다
a = NumberWithGrad(3)
b = a * 4           # b = 12
c = b + 3           # c = 15
d = a + 2           # d = 5   (a is re-used here!)
e = c * d           # e = 75
e.backward()

print(f"  e = ((a*4)+3) * (a+2),  a = 3")
print(f"  de/da = {a.grad}  (expected 35)")
# 확인: e = (4a+3)(a+2) = 4a² + 11a + 6
# de/da = 8a + 11 = 8*3 + 11 = 35  ✓
print(f"  Manual check: 8*3 + 11 = {8*3 + 11}")
print()


# =====================================================================
# 3부 – 확장 연산: 뺄셈, 나눗셈, 거듭제곱
# =====================================================================
print("=" * 60)
print("Part 3: Extended operations")
print("=" * 60)

# x = 3에서 f(x) = (x^2 - 1) / (x + 1)
x = NumberWithGrad(3.0)
f = (x ** 2 - 1) / (x + 1)
f.backward()
print(f"  f(x) = (x² - 1)/(x + 1) at x = 3")
print(f"  f(3) = {f.num:.4f}  (expected 2.0)")
print(f"  df/dx = {x.grad:.4f}")
# f(x) = (x-1)(x+1)/(x+1) = x-1  →  x ≠ -1일 때 df/dx = 1
# 그러나 원식에 몫의 미분법을 쓰면: ((2x)(x+1) - (x²-1)) / (x+1)²
#   = (2x² + 2x - x² + 1) / (x+1)²  = (x² + 2x + 1)/(x+1)² = (x+1)²/(x+1)² = 1
print(f"  Expected: 1.0")
print()


# =====================================================================
# 4부 – TensorWithGrad: 벡터화된 자동 미분
# =====================================================================
print("=" * 60)
print("Part 4: Tensor-level automatic differentiation")
print("=" * 60)


def ensure_tensor(t):
    if isinstance(t, TensorWithGrad):
        return t
    return TensorWithGrad(np.asarray(t, dtype=np.float64))


class TensorWithGrad:
    """행렬 곱, 원소별 셈, 합을 받치는 텐서 수준 자동 미분."""

    def __init__(self, data, depends_on=None, creation_op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op
        self.shape = self.data.shape

    def __repr__(self):
        return f"TensorWithGrad(shape={self.shape})"

    # ── 순전파 연산 ──────────────────────────────────────────────────

    def __add__(self, other):
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data + other.data,
            depends_on=[self, other],
            creation_op="add",
        )

    def __mul__(self, other):
        """원소별 곱하기."""
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data * other.data,
            depends_on=[self, other],
            creation_op="mul",
        )

    def matmul(self, other):
        """행렬 곱하기: self @ other."""
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data @ other.data,
            depends_on=[self, other],
            creation_op="matmul",
        )

    def sum(self):
        """홑값으로 줄인다."""
        return TensorWithGrad(
            np.array(self.data.sum()),
            depends_on=[self],
            creation_op="sum",
        )

    def relu(self):
        """ReLU 살림."""
        return TensorWithGrad(
            np.maximum(self.data, 0),
            depends_on=[self],
            creation_op="relu",
        )

    def sigmoid(self):
        """시그모이드 살림."""
        s = 1.0 / (1.0 + np.exp(-self.data))
        return TensorWithGrad(s, depends_on=[self], creation_op="sigmoid")

    # ── 역전파 ───────────────────────────────────────────────────────

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        if self.creation_op == "add":
            self.depends_on[0].backward(self.grad)
            self.depends_on[1].backward(self.grad)

        elif self.creation_op == "mul":
            self.depends_on[0].backward(self.grad * self.depends_on[1].data)
            self.depends_on[1].backward(self.grad * self.depends_on[0].data)

        elif self.creation_op == "matmul":
            # d(A @ B)/dA = grad @ B^T
            # d(A @ B)/dB = A^T @ grad
            A, B = self.depends_on
            A.backward(self.grad @ B.data.T)
            B.backward(A.data.T @ self.grad)

        elif self.creation_op == "sum":
            self.depends_on[0].backward(
                np.ones_like(self.depends_on[0].data) * self.grad
            )

        elif self.creation_op == "relu":
            self.depends_on[0].backward(
                self.grad * (self.depends_on[0].data > 0).astype(float)
            )

        elif self.creation_op == "sigmoid":
            s = self.data
            self.depends_on[0].backward(self.grad * s * (1 - s))


# 시연: 선형 층  y = X @ W + b, loss = sum(y)
np.random.seed(42)
X = TensorWithGrad(np.random.randn(4, 3))
W = TensorWithGrad(np.random.randn(3, 2))
b = TensorWithGrad(np.random.randn(1, 2))

y = X.matmul(W) + b
loss = y.sum()
loss.backward()

print(f"  X shape: {X.shape}, W shape: {W.shape}")
print(f"  y = X @ W + b, loss = sum(y)")
print(f"  loss = {loss.data:.4f}")
print(f"  dL/dW shape: {W.grad.shape}")
print(f"  dL/dW:\n{W.grad}")
print(f"  dL/db: {b.grad}")
print()


# =====================================================================
# 5부 – PyTorch와 대조 검증
# =====================================================================
print("=" * 60)
print("Part 5: Validation against PyTorch autograd")
print("=" * 60)

try:
    import torch

    # --- 스칼라 검사 ---
    x_pt = torch.tensor(3.0, requires_grad=True)
    f_pt = (x_pt ** 2 - 1) / (x_pt + 1)
    f_pt.backward()

    x_ours = NumberWithGrad(3.0)
    f_ours = (x_ours ** 2 - 1) / (x_ours + 1)
    f_ours.backward()

    print(f"  Scalar test:  f(x) = (x²-1)/(x+1) at x=3")
    print(f"    PyTorch grad:  {x_pt.grad.item():.6f}")
    print(f"    Our grad:      {x_ours.grad:.6f}")
    print(f"    Match: {abs(x_pt.grad.item() - x_ours.grad) < 1e-10}")
    print()

    # --- 텐서 검사: 선형 층 ---
    np.random.seed(42)
    X_np = np.random.randn(4, 3)
    W_np = np.random.randn(3, 2)
    b_np = np.random.randn(1, 2)

    # PyTorch
    X_pt = torch.tensor(X_np, dtype=torch.float64)
    W_pt = torch.tensor(W_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    y_pt = X_pt @ W_pt + b_pt
    loss_pt = y_pt.sum()
    loss_pt.backward()

    # 우리 구현
    X_t = TensorWithGrad(X_np)
    W_t = TensorWithGrad(W_np)
    b_t = TensorWithGrad(b_np)
    y_t = X_t.matmul(W_t) + b_t
    loss_t = y_t.sum()
    loss_t.backward()

    W_diff = np.abs(W_pt.grad.numpy() - W_t.grad).max()
    b_diff = np.abs(b_pt.grad.numpy() - b_t.grad).max()
    print(f"  Tensor test: y = X @ W + b, loss = sum(y)")
    print(f"    dL/dW max |diff| = {W_diff:.2e}")
    print(f"    dL/db max |diff| = {b_diff:.2e}")
    print(f"    Match: {W_diff < 1e-10 and b_diff < 1e-10}")
    print()

    # --- ReLU 검사 ---
    np.random.seed(123)
    X_np = np.random.randn(4, 3)
    W_np = np.random.randn(3, 2)

    X_pt = torch.tensor(X_np, dtype=torch.float64)
    W_pt = torch.tensor(W_np, dtype=torch.float64, requires_grad=True)
    h_pt = torch.relu(X_pt @ W_pt)
    loss_pt = h_pt.sum()
    loss_pt.backward()

    X_t = TensorWithGrad(X_np)
    W_t = TensorWithGrad(W_np)
    h_t = X_t.matmul(W_t).relu()
    loss_t = h_t.sum()
    loss_t.backward()

    W_diff = np.abs(W_pt.grad.numpy() - W_t.grad).max()
    print(f"  ReLU test: h = relu(X @ W), loss = sum(h)")
    print(f"    dL/dW max |diff| = {W_diff:.2e}")
    print(f"    Match: {W_diff < 1e-10}")
    print()

except ImportError:
    print("  PyTorch not available — skipping validation\n")


# =====================================================================
# 6부 – 우리 autograd 엔진으로 만드는 작은 신경망
# =====================================================================
print("=" * 60)
print("Part 6: Training a tiny NN with our autograd")
print("=" * 60)

np.random.seed(42)

# XOR 데이터셋 — 선형 분리 불가능
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)

# 가중치 초기화(2층 신경망: 2 → 4 → 1)
W1_data = np.random.randn(2, 4) * 0.5
b1_data = np.zeros((1, 4))
W2_data = np.random.randn(4, 1) * 0.5
b2_data = np.zeros((1, 1))

lr = 0.5
losses = []

for epoch in range(500):
    # 데이터를 TensorWithGrad로 감싼다
    X = TensorWithGrad(X_data)
    y = TensorWithGrad(y_data)
    W1 = TensorWithGrad(W1_data.copy())
    b1 = TensorWithGrad(b1_data.copy())
    W2 = TensorWithGrad(W2_data.copy())
    b2 = TensorWithGrad(b2_data.copy())

    # 순전파
    h = X.matmul(W1) + b1          # hidden layer pre-activation
    h_act = h.sigmoid()             # hidden activation
    out = h_act.matmul(W2) + b2    # output pre-activation
    pred = out.sigmoid()            # output activation

    # MSE 손실 = sum((pred - y)^2) / n
    diff = pred + (y * TensorWithGrad(np.full_like(y_data, -1.0)))
    sq = diff * diff
    loss = sq.sum()

    # 역전파
    loss.backward()

    # SGD 갱신
    W1_data -= lr * W1.grad
    b1_data -= lr * b1.grad
    W2_data -= lr * W2.grad
    b2_data -= lr * b2.grad

    losses.append(loss.data.item())
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d}: loss = {loss.data.item():.4f}")

# 최종 예측
X = TensorWithGrad(X_data)
W1 = TensorWithGrad(W1_data)
b1 = TensorWithGrad(b1_data)
W2 = TensorWithGrad(W2_data)
b2 = TensorWithGrad(b2_data)

h_act = X.matmul(W1) + b1
h_act = h_act.sigmoid()
out = h_act.matmul(W2) + b2
pred = out.sigmoid()

print(f"\n  Final predictions (XOR):")
for i in range(4):
    print(f"    {X_data[i]} → {pred.data[i, 0]:.4f}  (target: {y_data[i, 0]:.0f})")

print("\nDone.")


if __name__ == "__main__":
    pass
```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```
