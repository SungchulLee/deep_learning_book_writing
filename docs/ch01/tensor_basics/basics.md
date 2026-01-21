# Tensor Basics

Modern machine learning frameworks (such as PyTorch and JAX) generalize NumPy arrays into **tensors**, which support automatic differentiation and GPU acceleration.

---

## What is a tensor?

A tensor is a:
- multi-dimensional array,
- with a fixed dtype,
- living on a specific device (CPU/GPU).

Conceptually:
- scalar → 0D tensor
- vector → 1D tensor
- matrix → 2D tensor
- higher-order tensors → ND tensors

---

## Creating tensors

Example (PyTorch-style):

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
A = torch.zeros((2, 3))
```

Tensors closely resemble NumPy arrays.

---

## Shape, dtype, device

```python
x.shape
x.dtype
x.device
```

These attributes control:
- memory layout,
- numerical precision,
- computation placement.

---

## Tensor operations

Operations are vectorized:

```python
y = x * 2 + 1
```

They are designed to:
- avoid Python loops,
- exploit hardware parallelism.

---

## Financial

Tensors are used for:
- large-scale simulations,
- neural-network-based models,
- differentiable pricing and calibration.

---

## Key takeaways

- Tensors generalize arrays.
- Shape, dtype, and device matter.
- They enable scalable numerical computing.
