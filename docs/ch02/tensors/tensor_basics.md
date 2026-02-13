# Tensor Basics

## Overview

Modern machine learning frameworks (such as PyTorch and JAX) generalize NumPy arrays into **tensors**, which support automatic differentiation and GPU acceleration. Tensors are the fundamental data structure in PyTorch: all data—images, text, audio, and numerical features—must be represented as tensors before processing.

---

## What Is a Tensor?

A tensor is a multi-dimensional array with a fixed dtype living on a specific device (CPU or GPU):

| Rank | Name | Shape Example | Description |
|------|------|---------------|-------------|
| 0 | Scalar | `[]` | Single number |
| 1 | Vector | `[n]` | 1D array |
| 2 | Matrix | `[m, n]` | 2D array |
| 3 | 3D Tensor | `[d, m, n]` | 3D array (e.g., RGB image) |
| N | N-D Tensor | `[d₁, d₂, ..., dₙ]` | N-dimensional array |

### Mathematical Notation

A tensor $\mathcal{T}$ of rank $n$ with dimensions $(d_1, d_2, \ldots, d_n)$ contains elements:

$$\mathcal{T}_{i_1, i_2, \ldots, i_n} \quad \text{where} \quad 0 \leq i_k < d_k$$

For example, a 4D tensor representing a batch of images has shape $(B, C, H, W)$ where:

- $B$ = batch size
- $C$ = number of channels
- $H$ = height
- $W$ = width

The **total number of elements** (numel) is the product of all dimensions:

$$\text{numel}(\mathcal{T}) = \prod_{k=1}^{n} d_k$$

---

## Creating Tensors

```python
import torch

# From Python data
x = torch.tensor([1.0, 2.0, 3.0])
A = torch.zeros((2, 3))

# With explicit dtype
B = torch.tensor([1, 2, 3], dtype=torch.float32)
```

Tensors closely resemble NumPy arrays but additionally support gradient tracking and GPU placement.

---

## Scalar Tensors

A scalar tensor has rank 0 (empty shape) and contains a single value.

```python
# From a Python integer
scalar_int = torch.tensor(42)
print(f"Value: {scalar_int}")           # tensor(42)
print(f"Shape: {scalar_int.shape}")     # torch.Size([])
print(f"Dtype: {scalar_int.dtype}")     # torch.int64

# From a Python float
scalar_float = torch.tensor(3.14)
print(f"Value: {scalar_float}")         # tensor(3.1400)
print(f"Dtype: {scalar_float.dtype}")   # torch.float32

# With explicit dtype
scalar_f32 = torch.tensor(42, dtype=torch.float32)
print(f"Dtype: {scalar_f32.dtype}")     # torch.float32
```

### Scalar vs 1-Element Tensor

A critical distinction exists between a true scalar (rank-0) and a 1-element vector (rank-1):

```python
scalar = torch.tensor(42)      # Shape: []     — rank 0
vector = torch.tensor([42])    # Shape: [1]   — rank 1

print(f"Scalar shape: {scalar.shape}")  # torch.Size([])
print(f"Vector shape: {vector.shape}")  # torch.Size([1])

# Dimensionality affects broadcasting and indexing behavior
print(scalar.dim())   # 0
print(vector.dim())   # 1
```

### Alternative Scalar Creation

```python
# Using torch.scalar_tensor()
t = torch.scalar_tensor(42)
print(f"Shape: {t.shape}")  # torch.Size([])

# Using torch.full() with empty shape
t = torch.full((), 7.7)  # Empty tuple () means scalar
print(f"Value: {t}, Shape: {t.shape}")  # tensor(7.7000), torch.Size([])
```

### Extracting Python Values with `item()`

The `item()` method converts a single-element tensor to a Python scalar:

```python
t = torch.tensor(42)
python_val = t.item()
print(f"Python value: {python_val}, type: {type(python_val)}")
# Python value: 42, type: <class 'int'>

# Works for any single-element tensor regardless of shape
t_2d = torch.tensor([[3.14]])  # Shape [1, 1]
print(t_2d.item())  # 3.14

# Fails for multi-element tensors
t_multi = torch.tensor([1, 2, 3])
# t_multi.item()  # ValueError: only one element tensors can be converted
```

---

## Shape, Dtype, Device

Every tensor has three fundamental attributes plus a gradient-tracking flag:

```python
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 1. Shape — dimensions of the tensor
print(f"Shape: {t.shape}")           # torch.Size([2, 2])

# 2. Dtype — data type of elements
print(f"Dtype: {t.dtype}")           # torch.float32

# 3. Device — where tensor is stored
print(f"Device: {t.device}")         # cpu

# 4. requires_grad — whether to track operations for autodiff
print(f"Requires grad: {t.requires_grad}")  # False
```

These attributes control memory layout, numerical precision, and computation placement.

### Shape Inspection Utilities

```python
t = torch.randn(2, 3, 4)

print(f"shape:   {t.shape}")          # torch.Size([2, 3, 4])
print(f"size():  {t.size()}")         # torch.Size([2, 3, 4])  (equivalent)
print(f"dim():   {t.dim()}")          # 3  (rank/number of dimensions)
print(f"numel(): {t.numel()}")        # 24  (total element count)
print(f"size(1): {t.size(1)}")        # 3  (size of dimension 1)
```

---

## Tensor Operations Preview

Operations are vectorized to avoid Python loops and exploit hardware parallelism:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise arithmetic
print(a + b)    # tensor([5., 7., 9.])
print(a * b)    # tensor([4., 10., 18.])
print(a ** 2)   # tensor([1., 4., 9.])

# Reductions
print(a.sum())    # tensor(6.)
print(a.mean())   # tensor(2.)
```

!!! tip "See Also"
    For comprehensive coverage of arithmetic, matrix, reduction, comparison, and
    in-place operations, see [Tensor Operations](tensor_operations.md).

---

## Autograd: Automatic Differentiation

Tensors can track operations for gradient computation, which is the foundation of neural network training:

```python
# Create tensor that tracks gradients
x = torch.tensor(5.0, requires_grad=True)

# Perform computation: y = ½x²
y = 0.5 * (x ** 2)

# Compute gradient: dy/dx = x
y.backward()
print(f"x.grad: {x.grad}")  # tensor(5.)
```

!!! note "Integer Tensors Cannot Require Gradients"
    The `requires_grad=True` flag only works with floating-point and complex dtypes.
    Integer tensors cannot track gradients because differentiation is undefined for
    discrete values.

### Vector-Valued Functions

When $\mathbf{x} \in \mathbb{R}^n$ and $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(\mathbf{x})$ has the same shape as $\mathbf{x}$:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# f(x) = sum(x²) = x₁² + x₂² + x₃²
y = (x ** 2).sum()

# ∂f/∂xᵢ = 2xᵢ
y.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

### Disabling Gradient Tracking

```python
# Context manager — useful during inference
with torch.no_grad():
    y = x * 2  # No gradient tracking

# Decorator — useful for evaluation functions
@torch.no_grad()
def evaluate(model, data):
    return model(data)
```

---

## Practical Example: Loss Computation

Scalar tensors are fundamental in training, where the loss function produces a scalar:

```python
# Simulated prediction and target
prediction = torch.tensor(2.5, requires_grad=True)
target = torch.tensor(3.0)

# Mean Squared Error loss (scalar output)
loss = (prediction - target) ** 2
print(f"Loss: {loss}")  # tensor(0.2500, grad_fn=<PowBackward0>)

# Backpropagate
loss.backward()
print(f"Gradient: {prediction.grad}")  # tensor(-1.)
# Gradient = 2(pred - target) = 2(2.5 - 3.0) = -1.0
```

---

## Complete Neural Network Example

### 1. Data Preparation

```python
import numpy as np

# NumPy data
X_np = np.random.randn(100, 10).astype(np.float32)
y_np = np.random.randint(0, 2, 100)

# Convert to tensors
X = torch.from_numpy(X_np)            # float32, zero-copy
y = torch.from_numpy(y_np).long()     # int64 for class labels
```

### 2. Model Definition

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

### 3. Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Financial Relevance

Tensors enable GPU-accelerated computation for:

- Large-scale Monte Carlo simulations (batched path generation)
- Neural-network-based pricing and hedging models
- Differentiable calibration of stochastic volatility surfaces
- Portfolio optimization with gradient-based solvers
- Real-time risk computation across thousands of instruments

---

## Summary

| Concept | Description |
|---------|-------------|
| Scalar Tensor | Rank-0 tensor with shape `[]` |
| `item()` | Extracts Python scalar from single-element tensor |
| `requires_grad` | Enables automatic differentiation |
| `shape` / `size()` | Tensor dimensions as `torch.Size` |
| `dim()` | Number of dimensions (rank) |
| `numel()` | Total number of elements |
| `dtype` | Element data type |
| `device` | Storage location (CPU/GPU) |

## See Also

- [Tensor Creation](tensor_creation.md) — Comprehensive creation methods and dtype system
- [Tensor Operations](tensor_operations.md) — Arithmetic, reductions, and matrix operations
- [Indexing and Slicing](indexing_slicing.md) — Accessing and modifying tensor elements
- [NumPy Interoperability](numpy_interop.md) — Converting between NumPy/Pandas and PyTorch
