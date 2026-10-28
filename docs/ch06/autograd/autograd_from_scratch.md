# Autograd 직접 구현하기

스칼라와 NumPy 배열을 감싸는 작은 클래스로 자동 미분의 원리를 살펴본다. 교육용 구현의 지원 범위와 한계도 함께 구분한다.

## 1. 코드

??? note "코드 (541줄)"

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
    #   6부 – 우리 엔진으로 작은 뉴럴 넷트웍 만들기
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
          - grad:        쌓인 그래디언트(.backward()이 채운다)
          - depends_on:  어버이 NumberWithGrad 객체의 목록
          - creation_op: 이 절를 만든 셈을 가리키는 글자열 표
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
            """셈 그래프를 따라 그래디언트를 되부르며 퍼뜨린다.

            사슬 법칙을 쓴다:  dL/dx = dL/dy * dy/dx
            여기서 y은 이 절의 값이고 x은 어버이의 값이다.
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


    def unbroadcast(grad, shape):
        """방송으로 늘어난 축을 되접어 grad를 shape에 맞춘다.

        numpy는 (4, 4) + (1, 4)처럼 모양이 다른 배열을 자동으로 늘려 더한다.
        순전파에서 늘어난 만큼, 역전파에서는 그 축을 따라 그래디언트를 더해
        원래 모양으로 되돌려야 한다. 이것을 빠뜨리면 치우침 b가 (1, 4)인데
        그래디언트는 (4, 4)로 나와 b -= lr * b.grad에서 오류가 난다.
        실제 자동 미분 엔진이 모두 이 되접기를 안에 품고 있다.
        """
        # 순전파에서 앞쪽에 축이 새로 생긴 경우: 그 축을 모두 더해 없앤다
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # 길이 1이던 축이 늘어난 경우: 그 축을 더하되 자리는 남겨 둔다
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad


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
                # 두 피연산자의 모양이 다를 수 있으므로 각자의 모양으로 되접는다.
                # 치우침을 더할 때가 바로 이 경우다
                a, b = self.depends_on
                a.backward(unbroadcast(self.grad, a.data.shape))
                b.backward(unbroadcast(self.grad, b.data.shape))

            elif self.creation_op == "mul":
                a, b = self.depends_on
                a.backward(unbroadcast(self.grad * b.data, a.data.shape))
                b.backward(unbroadcast(self.grad * a.data, b.data.shape))

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


    # 시연: 선형 레이어  y = X @ W + b, loss = sum(y)
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

        # --- 텐서 검사: 선형 레이어 ---
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
    # 6부 – 우리 autograd 엔진으로 만드는 작은 뉴럴 넷트웍
    # =====================================================================
    print("=" * 60)
    print("Part 6: Training a tiny NN with our autograd")
    print("=" * 60)

    np.random.seed(42)

    # XOR 데이터셋 — 선형 분리 불가능
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # 웨이트 초기화(2레이어 뉴럴 넷트웍: 2 → 4 → 1)
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

        # MSE 로스 = sum((pred - y)^2) / n
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


**출력:**

```
============================================================
Part 1: Scalar Automatic Differentiation
============================================================
  c = (a * 4) + 3,  a = 3
  dc/da = 4.0   (expected 4)
  dc/db = 1.0   (expected 1)

============================================================
Part 2: Multi-path computation graphs
============================================================
  e = ((a*4)+3) * (a+2),  a = 3
  de/da = 35.0  (expected 35)
  Manual check: 8*3 + 11 = 35

============================================================
Part 3: Extended operations
============================================================
  f(x) = (x² - 1)/(x + 1) at x = 3
  f(3) = 2.0000  (expected 2.0)
  df/dx = 1.0000
  Expected: 1.0

============================================================
Part 4: Tensor-level automatic differentiation
============================================================
  X shape: (4, 3), W shape: (3, 2)
  y = X @ W + b, loss = sum(y)
  loss = -15.6822
  dL/dW shape: (3, 2)
  dL/dW:
[[ 4.14151687  4.14151687]
 [-0.06840064 -0.06840064]
 [-0.52165256 -0.52165256]]
  dL/db: [[4. 4.]]

============================================================
Part 5: Validation against PyTorch autograd
============================================================
  Scalar test:  f(x) = (x²-1)/(x+1) at x=3
    PyTorch grad:  1.000000
    Our grad:      1.000000
    Match: True

  Tensor test: y = X @ W + b, loss = sum(y)
    dL/dW max |diff| = 0.00e+00
    dL/db max |diff| = 0.00e+00
    Match: True

  ReLU test: h = relu(X @ W), loss = sum(h)
    dL/dW max |diff| = 0.00e+00
    Match: True

============================================================
Part 6: Training a tiny NN with our autograd
============================================================
  Epoch    0: loss = 1.0227
  Epoch  100: loss = 0.9977
  Epoch  200: loss = 0.3222
  Epoch  300: loss = 0.0221
  Epoch  400: loss = 0.0094

  Final predictions (XOR):
    [0. 0.] → 0.0433  (target: 0)
    [0. 1.] → 0.9636  (target: 1)
    [1. 0.] → 0.9637  (target: 1)
    [1. 1.] → 0.0363  (target: 0)

Done.
```

## 2. 논의

### 값과 미분 경로를 함께 저장하기

`NumberWithGrad`는 값, 경사, 부모 노드, 자신을 만든 연산을 저장한다. 덧셈과 곱셈 등으로 새 객체를 만들 때 부모 연결을 남기고, `backward()`에서 각 연산의 국소 미분에 상위 경사를 곱해 전달한다. 한 입력에 여러 경로가 연결되면 각 경로의 기여를 합해야 한다.

### 텐서 연산으로 확장하기

`TensorWithGrad`는 배열 단위로 같은 원리를 적용한다. 이 예제의 이차원 행렬 곱 $Y=XW$에서 상위 경사를 $G$라 하면 입력 경사는 $GW^\top$, 웨이트 경사는 $X^\top G$다. 브로드캐스팅한 덧셈의 역전파에서는 확장된 축을 합쳐 원래 입력 모양으로 되돌려야 하며, `unbroadcast()`가 이를 담당한다. 여기의 행렬 곱 미분은 일반적인 모든 차원과 배치 형태를 지원하는 구현은 아니다.

### 교육용 구현의 한계

이 구현은 위상 정렬로 각 노드를 한 번씩 처리하지 않고 부모로 재귀 호출한다. 특히 이미 누적한 `self.grad` 전체를 다시 부모로 전달하므로, 공유된 중간 노드가 여러 번 방문되면 일부 경로의 기여를 중복 전달할 수 있다. 예를 들어 $b=3a$, $L=b+b$에서는 참값 $dL/da=6$과 다른 값이 나올 수 있다. XOR 예제의 `diff * diff`처럼 같은 중간 결과를 두 입력으로 사용하는 연산도 이 문제의 영향을 받을 수 있다.

따라서 일부 PyTorch 비교 예제가 일치한다고 해서 모든 계산 그래프에서 올바른 구현으로 일반화할 수 없다. 일반적인 엔진을 만들려면 역위상 순서 처리 또는 새로 들어온 경사만 전달하는 방식과 경사 초기화 규칙 등을 설계해야 한다. 제공된 출력은 원래 실행 기록이며, 이 구현의 일반적인 정확성이나 XOR 학습 성능을 보장하는 근거로 사용하지 않는다.

## 3. 그 한계를 재어 본다

위의 「교육용 구현의 한계」는 설명이다. 돌려 보면 숫자가 나온다. 아래 코드는
이 쪽 옆의 `autograd_from_scratch.py`에 있고, 본문 구현을 가장 작게 줄인 것과
위상 순서로 고친 것을 나란히 돌린다.

```python
"""밑바닥 엔진의 버그를 재어 본다 — 그리고 고친다.

이 쪽의 본문 구현은 역전파를 **재귀**로 한다. 노드가 제 경사를
부모에게 넘기고, 부모가 다시 제 부모에게 넘긴다. 쓰기 쉽고 작은
그래프에서는 맞는 답을 준다.

그런데 **한 노드가 여러 곳에 쓰이면** 틀린다. 본문의 「교육용
구현의 한계」가 그것을 설명하는데, 설명만 하고 돌려 보지는 않는다.
여기서 돌려 본다.

틀리는 까닭은 하나다. 두 번째로 방문할 때 **새로 들어온 몫**이
아니라 **지금까지 쌓인 전부**를 부모에게 다시 넘긴다.

실행: python autograd_from_scratch.py
"""


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 본문과 같은 방식 — 재귀로 넘긴다
# ========================================================================

class Recursive:
    """본문 구현을 가장 작게 줄인 것. 더하기와 상수배만 있다."""

    def __init__(self, num, parents=(), op=""):
        self.num = num
        self.grad = None
        self.parents = list(parents)
        self.op = op

    def __add__(self, other):
        # type(self) 로 만들어야 물려받은 클래스가 제 backward 를 쓴다
        return type(self)(self.num + other.num, (self, other), "add")

    def __rmul__(self, k):                       # 상수 * 노드
        return type(self)(k * self.num, (self,), f"mul{k}")

    def backward(self, upstream=1.0):
        self.grad = upstream if self.grad is None else self.grad + upstream
        for p in self.parents:
            if self.op == "add":
                p.backward(self.grad)            # <- 쌓인 전부를 넘긴다
            else:
                p.backward(self.grad * float(self.op[3:]))


# ========================================================================
# 고친 것 — 위상 순서로 한 번씩만
# ========================================================================

class Topological:
    """역위상 순서로 노드마다 한 번씩만 처리한다."""

    def __init__(self, num, parents=(), op=""):
        self.num = num
        self.grad = 0.0
        self.parents = list(parents)
        self.op = op

    def __add__(self, other):
        return type(self)(self.num + other.num, (self, other), "add")

    def __rmul__(self, k):
        return type(self)(k * self.num, (self,), f"mul{k}")

    def _order(self):
        seen, order = set(), []

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                visit(p)
            order.append(node)                   # 부모가 먼저 들어간다

        visit(self)
        return order

    def backward(self):
        order = self._order()
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):             # 자식부터 처리한다
            for p in node.parents:
                if node.op == "add":
                    p.grad += node.grad
                else:
                    p.grad += node.grad * float(node.op[3:])


# ========================================================================
# 1. 공유가 없으면 둘 다 맞는다
# ========================================================================

def no_sharing():
    header("1. 공유가 없을 때")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        (3 * a + 3 * a).backward()               # 3a 를 두 번 따로 만든다
        print(f"  {name}  L = 3a + 3a (따로)  a.grad = {a.grad}   참값 6")
    print("  두 3a 가 서로 다른 객체라 공유가 아니다")


# ========================================================================
# 2. 공유하면 재귀 쪽이 틀린다
# ========================================================================

def with_sharing():
    header("2. b 를 두 번 쓰면")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        b = 3 * a
        (b + b).backward()
        print(f"  {name}  L = b + b, b = 3a   a.grad = {a.grad}   참값 6")
    print("  재귀 쪽이 9 다. 두 번째 방문에서 b.grad=2 를 통째로 다시 넘긴다")
    print("  3*1 + 3*2 = 9 — 첫 몫이 두 번 세어졌다")


# ========================================================================
# 3. 깊어지면 더 벌어진다
# ========================================================================

def deeper():
    header("3. 더 깊이 공유하면")
    for cls, name in ((Recursive, "재귀"), (Topological, "위상")):
        a = cls(2.0)
        b = 3 * a
        c = b + b
        d = c + c
        d.backward()
        print(f"  {name}  d = (b+b)+(b+b)   a.grad = {a.grad}   참값 12")
    print("  재귀 쪽이 39 다. 틀린 정도가 깊이에 따라 커진다")


# ========================================================================
# 4. 왜 작은 보기에서는 맞았는가
# ========================================================================

def why_it_passed():
    header("4. 본문의 보기들은 왜 맞았는가")
    print("  본문 1~3부의 식들은 중간 노드를 한 번씩만 쓴다")
    print("    c = 4a + 3        a 는 한 번")
    print("    e = (4a+3)(a+2)   a 가 두 번이지만 잎이라 더 넘길 데가 없다")
    print("  잎에서 쌓이는 것은 맞는 동작이다 — 경사 누적")
    print("  틀리는 것은 **중간 노드**가 공유될 때뿐이다")
    print("  그래서 PyTorch 와 대조한 검사들이 전부 통과했다")
    print("  통과한 검사가 구현의 일반적 정확성을 보증하지 않는다")


# ========================================================================
# 5. 고침의 핵심
# ========================================================================

def the_fix():
    header("5. 고치는 두 가지 길")
    print("  (가) 위상 순서 — 자식을 다 처리한 뒤에 부모로 넘긴다")
    print("       노드마다 정확히 한 번씩만 넘기므로 중복이 없다")
    print("       파이토치가 쓰는 방법이기도 하다")
    print("  (나) 새로 들어온 몫만 넘기기 — upstream 을 그대로 쓴다")
    print("       재귀를 유지할 수 있지만 같은 노드를 여러 번 방문한다")
    print("       그래프가 깊고 넓으면 방문 횟수가 지수로 는다")
    a = Recursive(2.0)
    b = 3 * a
    (b + b).backward()
    print(f"  (나) 를 본문 코드에 적용하려면 p.backward(self.grad) 를")
    print(f"       p.backward(upstream) 으로 고치면 된다")


def main():
    no_sharing()
    with_sharing()
    deeper()
    why_it_passed()
    the_fix()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 공유가 없을 때 ===
  재귀  L = 3a + 3a (따로)  a.grad = 6.0   참값 6
  위상  L = 3a + 3a (따로)  a.grad = 6.0   참값 6
  두 3a 가 서로 다른 객체라 공유가 아니다

=== 2. b 를 두 번 쓰면 ===
  재귀  L = b + b, b = 3a   a.grad = 9.0   참값 6
  위상  L = b + b, b = 3a   a.grad = 6.0   참값 6
  재귀 쪽이 9 다. 두 번째 방문에서 b.grad=2 를 통째로 다시 넘긴다
  3*1 + 3*2 = 9 — 첫 몫이 두 번 세어졌다

=== 3. 더 깊이 공유하면 ===
  재귀  d = (b+b)+(b+b)   a.grad = 39.0   참값 12
  위상  d = (b+b)+(b+b)   a.grad = 12.0   참값 12
  재귀 쪽이 39 다. 틀린 정도가 깊이에 따라 커진다

=== 4. 본문의 보기들은 왜 맞았는가 ===
  본문 1~3부의 식들은 중간 노드를 한 번씩만 쓴다
    c = 4a + 3        a 는 한 번
    e = (4a+3)(a+2)   a 가 두 번이지만 잎이라 더 넘길 데가 없다
  잎에서 쌓이는 것은 맞는 동작이다 — 경사 누적
  틀리는 것은 **중간 노드**가 공유될 때뿐이다
  그래서 PyTorch 와 대조한 검사들이 전부 통과했다
  통과한 검사가 구현의 일반적 정확성을 보증하지 않는다

=== 5. 고치는 두 가지 길 ===
  (가) 위상 순서 — 자식을 다 처리한 뒤에 부모로 넘긴다
       노드마다 정확히 한 번씩만 넘기므로 중복이 없다
       파이토치가 쓰는 방법이기도 하다
  (나) 새로 들어온 몫만 넘기기 — upstream 을 그대로 쓴다
       재귀를 유지할 수 있지만 같은 노드를 여러 번 방문한다
       그래프가 깊고 넓으면 방문 횟수가 지수로 는다
  (나) 를 본문 코드에 적용하려면 p.backward(self.grad) 를
       p.backward(upstream) 으로 고치면 된다
```

### 틀리는 자리와 틀리는 정도

| | 재귀 (본문) | 위상 (고침) | 참값 |
|---|---|---|---|
| `3a + 3a` (따로 만듦) | 6.0 | 6.0 | 6 |
| `b + b`, `b = 3a` | **9.0** | 6.0 | 6 |
| `(b+b) + (b+b)` | **39.0** | 12.0 | 12 |

공유가 없으면 둘 다 맞는다. **중간 노드를 두 번 쓰는 순간** 재귀 쪽이 9를
내놓고, 한 겹 더 깊어지면 39가 된다. 틀린 정도가 깊이에 따라 커진다.

9가 나오는 까닭은 셈으로 따라갈 수 있다. 첫 방문에서 `b.grad = 1`을 부모에게
넘겨 `a.grad = 3`이 된다. 두 번째 방문에서 `b.grad`가 2로 쌓였는데, **새로 들어온
1이 아니라 쌓인 2를** 다시 넘긴다. $3 \times 1 + 3 \times 2 = 9$다.

### 본문의 검사들은 왜 전부 통과했는가

1~3부의 식들은 **중간 노드를 한 번씩만** 쓴다. $e = (4a+3)(a+2)$에서 $a$가 두 번
나오지만 $a$는 잎이고, 잎에서 경사가 쌓이는 것은 **맞는 동작**이다. 틀리는 것은
중간 노드가 공유될 때뿐이다.

그래서 PyTorch와 대조한 5부의 검사가 전부 통과했다. **통과한 검사가 구현의
일반적 정확성을 보증하지 않는다**는 것이 이 쪽에서 가져갈 가장 큰 교훈이다.

### 고치는 두 길

| | 어떻게 | 값 |
|---|---|---|
| **위상 순서** | 자식을 다 처리한 뒤 부모로 넘긴다 | 노드마다 한 번. 파이토치가 쓰는 방법 |
| **새 몫만 넘기기** | `p.backward(self.grad)` → `p.backward(upstream)` | 재귀를 유지하지만 방문이 지수로 는다 |

본문 코드를 한 글자만 고치면 두 번째가 된다. 그런데 그래프가 깊고 넓으면
같은 노드를 여러 번 방문하므로, 실제 엔진은 모두 첫 번째를 쓴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`a * a`에서 본문 구현이 맞는 답을 주는가? 왜 그런지 3절의 설명으로 말하라.

</div>

??? success "풀이"
    맞는다. $a$가 두 번 쓰이지만 **잎**이기 때문이다.

    ```python
    a = NumberWithGrad(3.0)
    (a * a).backward()
    print("  a*a 의 a.grad =", a.grad, "  참값 2a = 6")
    ```

    ```
      a*a 의 a.grad = 6.0  참값 2a = 6
    ```

    잎은 부모가 없으므로 넘길 곳이 없다. 두 번 방문해서 각각 3을 더해 6이 되고,
    그것이 **바로 맞는 동작**이다 — 「경사 누적」에서 본 그 누적이다.

    틀리는 것은 **중간 노드**일 때다. 중간 노드는 쌓인 값을 다시 위로 넘기므로,
    쌓임이 그대로 전파된다.

    그래서 "공유하면 틀린다"가 아니라 **"중간 노드를 공유하면 틀린다"**가 정확한
    말이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
3절의 재귀 구현을 **한 글자만 고쳐** 바로잡아라. 고친 뒤 세 경우를 다시 확인하라.

</div>

??? success "풀이"
    `backward`가 부모에게 넘기는 것을 `self.grad`에서 `upstream`으로 바꾼다.

    ```python
    class Fixed(Recursive):
        def backward(self, upstream=1.0):
            self.grad = upstream if self.grad is None else self.grad + upstream
            for p in self.parents:
                if self.op == "add":
                    p.backward(upstream)             # self.grad 가 아니라
                else:
                    p.backward(upstream * float(self.op[3:]))

    a = Fixed(2.0); (3 * a + 3 * a).backward()
    print(f"  3a + 3a        a.grad = {a.grad}   참값 6")
    a = Fixed(2.0); b = 3 * a; (b + b).backward()
    print(f"  b + b          a.grad = {a.grad}   참값 6")
    a = Fixed(2.0); b = 3 * a; ((b + b) + (b + b)).backward()
    print(f"  (b+b)+(b+b)    a.grad = {a.grad}   참값 12")
    ```

    ```
      3a + 3a        a.grad = 6.0   참값 6
      b + b          a.grad = 6.0   참값 6
      (b+b)+(b+b)    a.grad = 12.0   참값 12
    ```

    셋 다 맞는다. **`self.grad`는 "지금까지 쌓인 전부"이고 `upstream`은 "이번에
    들어온 몫"**이다. 부모에게 넘겨야 하는 것은 뒤쪽이다.

    > 물려받기가 먹으려면 `__add__`와 `__rmul__`이 `Recursive(...)`가 아니라
    > `type(self)(...)`로 새 노드를 만들어야 한다. 그러지 않으면 잎만 `Fixed`가
    > 되고 중간 노드는 전부 원래 클래스라, 고친 `backward`가 한 번도 안 불린다.
    > 처음에 그렇게 적었다가 `b + b`가 여전히 9를 주는 것을 보고 알았다.

    그런데 이 고침에는 값이 따른다. 같은 노드를 **여러 번 방문**하게 된다.
    `(b+b)+(b+b)`에서 `backward`가 **11번** 불린다 — 노드가 다섯 개뿐인
    그래프에서. 공유가 겹칠수록 방문 횟수가 지수로 늘고, 층이 $L$개면
    $2^L$번이 될 수 있다.

    그래서 실제 엔진은 **위상 순서**를 쓴다. 한 번씩만 방문하면서도 맞는 답을
    준다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
5부의 PyTorch 대조 검사를 **통과시키면서도 틀린** 구현을 만들 수 있는가?
그 사실이 검사에 대해 무엇을 말하는가?

</div>

??? success "풀이"
    이미 그런 구현이 이 쪽의 본문이다. 5부의 세 검사(스칼라, 선형층, ReLU)를
    전부 통과하면서 `b + b`에서 틀린다.

    까닭은 **검사가 한 종류의 그래프만 밟기 때문**이다.

    | 5부의 검사 | 중간 노드 공유 |
    |---|---|
    | $(x^2-1)/(x+1)$ | $x$는 잎 |
    | $XW + b$ | 한 줄로 지나간다 |
    | $\text{relu}(XW)$ | 한 줄로 지나간다 |

    셋 다 **분기가 없는 그래프**다. 갈라졌다 다시 만나는 구조가 하나도 없으므로,
    바로 그 경우에만 나타나는 버그가 안 걸린다.

    여기서 가져갈 것이 둘이다.

    **검사는 밟은 길만 보증한다.** "PyTorch와 일치했다"는 "이 세 식에 대해
    일치했다"는 뜻이고, 다른 식으로 일반화할 근거가 아니다. 「실전 예제」에서
    파이프라인의 조용한 고장을 찾을 때와 같은 이야기다 — 숫자가 맞게 나오는 것이
    맞다는 증거가 아니다.

    **그러니 검사는 틀릴 법한 자리를 겨냥해야 한다.** 자동 미분 엔진이라면
    분기, 공유, 반복, 고차 미분이 그 자리다. 무작위로 식을 만들어 PyTorch와
    견주는 것(fuzzing)이 이런 버그를 찾는 가장 싼 길이고, 실제로
    `torch.autograd.gradcheck`가 하는 일이 그 일부다.

## 정리하며

자동 미분의 핵심은 국소 미분과 연쇄 법칙, 여러 경로의 경사 합산이다. 공유 노드의 방문 순서와 누적 처리는 엔진의 정확성을 좌우하므로 단순 재귀 구현의 한계를 함께 이해한다.
