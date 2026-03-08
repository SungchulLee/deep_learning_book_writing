"""Autograd from scratch."""
# ---
# title: "Automatic Differentiation from Scratch"
# description: "Building a minimal autograd engine in Python — computation graphs,
#               backward pass, gradient accumulation, and PyTorch comparison"
# ---
#
# PyTorch's autograd system is powerful, but understanding how it works under the
# hood is essential for deep learning mastery.  This script builds a minimal
# automatic differentiation engine from scratch, then validates it against PyTorch.
#
#   Part 1 – NumberWithGrad: scalar autodiff with add/mul
#   Part 2 – Gradient accumulation and multi-path graphs
#   Part 3 – Extend to subtraction, division, power
#   Part 4 – TensorWithGrad: vectorised autodiff with numpy
#   Part 5 – Validate against PyTorch autograd
#   Part 6 – Build a tiny neural network using our engine
#
# Adapted from: O'Reilly "Deep Learning from Scratch" Ch 6

from __future__ import annotations
from typing import Union, List, Optional
import numpy as np

Numberable = Union[float, int]


# =====================================================================
# Part 1 – Scalar Automatic Differentiation
# =====================================================================
print("=" * 60)
print("Part 1: Scalar Automatic Differentiation")
print("=" * 60)


def ensure_number(num):
    """Wrap raw Python numbers into NumberWithGrad."""
    if isinstance(num, NumberWithGrad):
        return num
    return NumberWithGrad(num)


class NumberWithGrad:
    """A scalar value that tracks its computation graph for autodiff.

    Each instance stores:
      - num:         the scalar value
      - grad:        accumulated gradient (filled by .backward())
      - depends_on:  list of parent NumberWithGrad objects
      - creation_op: string tag identifying the operation that created this node
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

    # ── forward operators ────────────────────────────────────────────

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

    # ── backward pass ────────────────────────────────────────────────

    def backward(self, backward_grad: Optional[float] = None) -> None:
        """Recursively propagate gradients through the computation graph.

        Uses the chain rule:  dL/dx = dL/dy * dy/dx
        where y is this node's value and x is a parent's value.
        """
        if backward_grad is None:
            # Root of backward — gradient of output w.r.t. itself is 1
            self.grad = 1.0
        else:
            # Accumulate gradients (handles fan-out / multi-path graphs)
            if self.grad is None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad

        # Propagate to parents based on the creation operation
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


# Demo: simple chain  c = (a * 4) + 3
a = NumberWithGrad(3)
b = a * 4          # b = 12
c = b + 3          # c = 15
c.backward()
print(f"  c = (a * 4) + 3,  a = 3")
print(f"  dc/da = {a.grad}   (expected 4)")
print(f"  dc/db = {b.grad}   (expected 1)")
print()


# =====================================================================
# Part 2 – Gradient Accumulation (multi-path graphs)
# =====================================================================
print("=" * 60)
print("Part 2: Multi-path computation graphs")
print("=" * 60)

# e = c * d  where c depends on a and d depends on a
# a is used twice → gradients must accumulate
a = NumberWithGrad(3)
b = a * 4           # b = 12
c = b + 3           # c = 15
d = a + 2           # d = 5   (a is re-used here!)
e = c * d           # e = 75
e.backward()

print(f"  e = ((a*4)+3) * (a+2),  a = 3")
print(f"  de/da = {a.grad}  (expected 35)")
# Verify: e = (4a+3)(a+2) = 4a² + 11a + 6
# de/da = 8a + 11 = 8*3 + 11 = 35  ✓
print(f"  Manual check: 8*3 + 11 = {8*3 + 11}")
print()


# =====================================================================
# Part 3 – Extended operations: sub, div, pow
# =====================================================================
print("=" * 60)
print("Part 3: Extended operations")
print("=" * 60)

# f(x) = (x^2 - 1) / (x + 1)  at x = 3
x = NumberWithGrad(3.0)
f = (x ** 2 - 1) / (x + 1)
f.backward()
print(f"  f(x) = (x² - 1)/(x + 1) at x = 3")
print(f"  f(3) = {f.num:.4f}  (expected 2.0)")
print(f"  df/dx = {x.grad:.4f}")
# f(x) = (x-1)(x+1)/(x+1) = x-1  →  df/dx = 1  for x ≠ -1
# But using quotient rule on original: ((2x)(x+1) - (x²-1)) / (x+1)²
#   = (2x² + 2x - x² + 1) / (x+1)²  = (x² + 2x + 1)/(x+1)² = (x+1)²/(x+1)² = 1
print(f"  Expected: 1.0")
print()


# =====================================================================
# Part 4 – TensorWithGrad: vectorised autodiff
# =====================================================================
print("=" * 60)
print("Part 4: Tensor-level automatic differentiation")
print("=" * 60)


def ensure_tensor(t):
    if isinstance(t, TensorWithGrad):
        return t
    return TensorWithGrad(np.asarray(t, dtype=np.float64))


class TensorWithGrad:
    """Tensor-level autodiff supporting matmul, element-wise ops, and sum."""

    def __init__(self, data, depends_on=None, creation_op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op
        self.shape = self.data.shape

    def __repr__(self):
        return f"TensorWithGrad(shape={self.shape})"

    # ── forward ops ──────────────────────────────────────────────────

    def __add__(self, other):
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data + other.data,
            depends_on=[self, other],
            creation_op="add",
        )

    def __mul__(self, other):
        """Element-wise multiplication."""
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data * other.data,
            depends_on=[self, other],
            creation_op="mul",
        )

    def matmul(self, other):
        """Matrix multiplication: self @ other."""
        other = ensure_tensor(other)
        return TensorWithGrad(
            self.data @ other.data,
            depends_on=[self, other],
            creation_op="matmul",
        )

    def sum(self):
        """Reduce to scalar."""
        return TensorWithGrad(
            np.array(self.data.sum()),
            depends_on=[self],
            creation_op="sum",
        )

    def relu(self):
        """ReLU activation."""
        return TensorWithGrad(
            np.maximum(self.data, 0),
            depends_on=[self],
            creation_op="relu",
        )

    def sigmoid(self):
        """Sigmoid activation."""
        s = 1.0 / (1.0 + np.exp(-self.data))
        return TensorWithGrad(s, depends_on=[self], creation_op="sigmoid")

    # ── backward pass ────────────────────────────────────────────────

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


# Demo: linear layer  y = X @ W + b, loss = sum(y)
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
# Part 5 – Validate against PyTorch
# =====================================================================
print("=" * 60)
print("Part 5: Validation against PyTorch autograd")
print("=" * 60)

try:
    import torch

    # --- Scalar test ---
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

    # --- Tensor test: linear layer ---
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

    # Ours
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

    # --- ReLU test ---
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
# Part 6 – Tiny neural network with our autograd engine
# =====================================================================
print("=" * 60)
print("Part 6: Training a tiny NN with our autograd")
print("=" * 60)

np.random.seed(42)

# XOR dataset — linearly inseparable
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Initialise weights (2-layer network: 2 → 4 → 1)
W1_data = np.random.randn(2, 4) * 0.5
b1_data = np.zeros((1, 4))
W2_data = np.random.randn(4, 1) * 0.5
b2_data = np.zeros((1, 1))

lr = 0.5
losses = []

for epoch in range(500):
    # Wrap data in TensorWithGrad
    X = TensorWithGrad(X_data)
    y = TensorWithGrad(y_data)
    W1 = TensorWithGrad(W1_data.copy())
    b1 = TensorWithGrad(b1_data.copy())
    W2 = TensorWithGrad(W2_data.copy())
    b2 = TensorWithGrad(b2_data.copy())

    # Forward pass
    h = X.matmul(W1) + b1          # hidden layer pre-activation
    h_act = h.sigmoid()             # hidden activation
    out = h_act.matmul(W2) + b2    # output pre-activation
    pred = out.sigmoid()            # output activation

    # MSE loss = sum((pred - y)^2) / n
    diff = pred + (y * TensorWithGrad(np.full_like(y_data, -1.0)))
    sq = diff * diff
    loss = sq.sum()

    # Backward pass
    loss.backward()

    # SGD update
    W1_data -= lr * W1.grad
    b1_data -= lr * b1.grad
    W2_data -= lr * W2.grad
    b2_data -= lr * b2.grad

    losses.append(loss.data.item())
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d}: loss = {loss.data.item():.4f}")

# Final predictions
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
