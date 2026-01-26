# Tensor Basics

## Overview

Tensors are the fundamental data structure in PyTorch, serving as multi-dimensional arrays that form the backbone of all deep learning computations. Understanding tensors is essential for anyone working with neural networks, as all data—images, text, audio, and numerical features—must be represented as tensors before processing.

## What is a Tensor?

A tensor is a generalization of scalars, vectors, and matrices to potentially higher dimensions:

| Rank | Name | Shape Example | Description |
|------|------|---------------|-------------|
| 0 | Scalar | `[]` | Single number |
| 1 | Vector | `[n]` | 1D array |
| 2 | Matrix | `[m, n]` | 2D array |
| 3 | 3D Tensor | `[d, m, n]` | 3D array (e.g., RGB image) |
| N | N-D Tensor | `[d₁, d₂, ..., dₙ]` | N-dimensional array |

## Mathematical Notation

In mathematical notation, a tensor $\mathcal{T}$ of rank $n$ with dimensions $(d_1, d_2, \ldots, d_n)$ contains elements:

$$\mathcal{T}_{i_1, i_2, \ldots, i_n} \quad \text{where} \quad 0 \leq i_k < d_k$$

For example, a 3D tensor representing a batch of images might have shape $(B, C, H, W)$ where:

- $B$ = batch size
- $C$ = number of channels
- $H$ = height
- $W$ = width

## Creating Scalar Tensors

A scalar tensor has rank 0 (empty shape) and contains a single value.

```python
import torch

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
scalar = torch.tensor(42)      # Shape: []     - rank 0
vector = torch.tensor([42])    # Shape: [1]   - rank 1

print(f"Scalar shape: {scalar.shape}")  # torch.Size([])
print(f"Vector shape: {vector.shape}")  # torch.Size([1])
```

### Extracting Python Values with `item()`

The `item()` method converts a single-element tensor to a Python scalar:

```python
t = torch.tensor(42)
python_val = t.item()
print(f"Python value: {python_val}, type: {type(python_val)}")
# Python value: 42, type: <class 'int'>

# Works for any single-element tensor
t_2d = torch.tensor([[3.14]])  # Shape [1, 1]
print(t_2d.item())  # 3.14

# Fails for multi-element tensors
t_multi = torch.tensor([1, 2, 3])
# t_multi.item()  # ValueError: only one element tensors can be converted
```

## Key Tensor Attributes

Every tensor has four fundamental attributes:

```python
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 1. Shape - dimensions of the tensor
print(f"Shape: {t.shape}")           # torch.Size([2, 2])

# 2. Dtype - data type of elements
print(f"Dtype: {t.dtype}")           # torch.float32

# 3. Device - where tensor is stored
print(f"Device: {t.device}")         # cpu

# 4. requires_grad - whether to track operations for autodiff
print(f"Requires grad: {t.requires_grad}")  # False
```

## Tensor Creation with Autograd

For gradient computation in neural networks, tensors must track operations:

```python
# Create tensor that tracks gradients
x = torch.tensor(5.0, requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}")  # True

# Perform computation
y = 0.5 * (x ** 2)  # y = ½x²

# Compute gradient
y.backward()        # dy/dx = x

print(f"x.grad: {x.grad}")  # tensor(5.)
```

!!! note "Integer Tensors Cannot Require Gradients"
    The `requires_grad=True` flag only works with floating-point and complex dtypes.
    Integer tensors cannot track gradients because differentiation is undefined for
    discrete values.

## Alternative Creation Methods

### Using `torch.scalar_tensor()`

Explicit scalar creation:

```python
t = torch.scalar_tensor(42)
print(f"Shape: {t.shape}")  # torch.Size([])
```

### Using `torch.full()` with Empty Shape

Create a scalar by specifying an empty shape:

```python
t = torch.full((), 7.7)  # Empty tuple () means scalar
print(f"Value: {t}, Shape: {t.shape}")  # tensor(7.7000), torch.Size([])
```

## Practical Example: Loss Computation

Scalar tensors are fundamental in training neural networks, where the loss function produces a scalar:

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

## Summary

| Concept | Description |
|---------|-------------|
| Scalar Tensor | Rank-0 tensor with shape `[]` |
| `item()` | Extracts Python scalar from single-element tensor |
| `requires_grad` | Enables automatic differentiation |
| `shape` | Tensor dimensions as `torch.Size` |
| `dtype` | Element data type |
| `device` | Storage location (CPU/GPU) |

## See Also

- [Tensor Creation and dtypes](tensor_creation_dtypes.md) - Comprehensive creation methods
- [Memory Layout and Strides](memory_layout_strides.md) - How tensors are stored in memory
- [NumPy Interoperability](numpy_interop.md) - Converting between NumPy and PyTorch
