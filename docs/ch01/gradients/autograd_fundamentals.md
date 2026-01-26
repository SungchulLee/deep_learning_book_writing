# Autograd Fundamentals

## Overview

PyTorch's autograd engine is the foundation of automatic differentiation in deep learning. It provides a tape-based differentiation system that records operations performed on tensors and automatically computes gradients through backpropagation. Understanding autograd is essential for training neural networks, debugging gradient-related issues, and implementing custom operations.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the concept of leaf tensors and their role in gradient computation
2. Enable and disable gradient tracking on tensors
3. Compute gradients using `.backward()` on scalar losses
4. Distinguish between leaf and non-leaf tensors
5. Use `.retain_grad()` to store gradients for intermediate tensors
6. Understand the vector-Jacobian product (VJP) framework

## Core Concepts

### Leaf Tensors

A **leaf tensor** is a tensor created directly by the user rather than as the result of an operation. Leaf tensors are the "starting points" of a computational graph.

**Mathematical Definition:**
A tensor $x$ is a leaf tensor if and only if:
- It was created by the user (not from an operation on other tensors)
- It has `grad_fn = None` (no gradient function)
- By default, only leaf tensors store gradients after `.backward()`

```python
import torch

# Leaf tensor - created directly by user
x = torch.randn(3, requires_grad=True)

print(f"x.is_leaf: {x.is_leaf}")        # True
print(f"x.grad_fn: {x.grad_fn}")        # None
print(f"x.requires_grad: {x.requires_grad}")  # True
```

### Non-Leaf Tensors

Non-leaf tensors are created as the result of operations on other tensors. They have a `grad_fn` attribute pointing to the backward function.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)  # Leaf
y = 2 * x    # Non-leaf (result of multiplication)
z = y ** 2   # Non-leaf (result of power)

print(f"y.is_leaf: {y.is_leaf}")        # False
print(f"y.grad_fn: {y.grad_fn}")        # <MulBackward0>
print(f"z.grad_fn: {z.grad_fn}")        # <PowBackward0>
```

### The `requires_grad` Attribute

The `requires_grad` attribute controls whether PyTorch tracks operations on a tensor for gradient computation.

**Propagation Rule:** If any input to an operation has `requires_grad=True`, the output will also have `requires_grad=True`.

```python
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=False)
c = a + b

print(f"c.requires_grad: {c.requires_grad}")  # True (inherits from a)
```

## Computing Gradients

### The Backward Pass

For a scalar loss function $L$, calling `L.backward()` computes $\frac{\partial L}{\partial x}$ for all leaf tensors $x$ with `requires_grad=True`.

**Mathematical Framework:**
Given:
- $x \in \mathbb{R}^n$ (input tensor)
- $L: \mathbb{R}^n \rightarrow \mathbb{R}$ (loss function)

The gradient is:
$$\nabla_x L = \left[\frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, \ldots, \frac{\partial L}{\partial x_n}\right]^T$$

**Example: Computing Gradients**

```python
import torch

torch.manual_seed(0)

# Create leaf tensor with gradient tracking
x = torch.randn(3, requires_grad=True)
print(f"x: {x}")

# Forward pass: loss = sum(x^2)
loss = (x ** 2).sum()
print(f"loss: {loss}")

# Backward pass: compute d(loss)/dx = 2*x
loss.backward()

print(f"x.grad: {x.grad}")
print(f"Expected (2*x): {2*x.detach()}")
print(f"Match: {torch.allclose(x.grad, 2*x.detach())}")
```

**Output:**
```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
loss: tensor(7.2274, grad_fn=<SumBackward0>)
x.grad: tensor([ 3.0820, -0.5868, -4.3576])
Expected (2*x): tensor([ 3.0820, -0.5868, -4.3576])
Match: True
```

### Gradient Storage Behavior

By default, gradients are only stored for **leaf tensors**:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x       # Non-leaf
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # Stored: tensor([4., 8., 12.])
print(f"y.grad: {y.grad}")  # None (not stored by default)
```

### Retaining Gradients for Non-Leaf Tensors

Use `.retain_grad()` before the backward pass to store gradients for non-leaf tensors:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x
y.retain_grad()  # Request gradient storage
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # tensor([8., 16., 24.])
print(f"y.grad: {y.grad}")  # tensor([4., 8., 12.]) - Now stored!
```

**Mathematical Verification:**
- $y = 2x$, so $y = [2, 4, 6]$
- $z = \sum y_i^2 = 4 + 16 + 36 = 56$
- $\frac{\partial z}{\partial y_i} = 2y_i$, so $\frac{\partial z}{\partial y} = [4, 8, 12]$
- $\frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_i} = 2y_i \cdot 2 = 4y_i$
- $\frac{\partial z}{\partial x} = [8, 16, 24]$

## The Vector-Jacobian Product Framework

PyTorch uses **reverse-mode automatic differentiation** (backpropagation), which computes vector-Jacobian products (VJPs) efficiently.

### Mathematical Foundation

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with input $x$ and output $y = f(x)$:

The **Jacobian matrix** is:
$$J = \frac{\partial y}{\partial x} = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

For a scalar loss $L$ depending on $y$, let $v = \frac{\partial L}{\partial y} \in \mathbb{R}^m$ be the **upstream gradient**.

The **vector-Jacobian product** computes:
$$\frac{\partial L}{\partial x} = J^T v = v^T J \in \mathbb{R}^n$$

This is the chain rule in vector form, and it's what PyTorch computes during backpropagation.

### Implicit VJP for Scalar Outputs

When calling `loss.backward()` on a scalar, PyTorch implicitly uses $v = 1$:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x ** 2).sum()

# This is equivalent to:
# loss.backward(torch.tensor(1.0))
loss.backward()

print(f"x.grad: {x.grad}")  # tensor([2., 4., 6.])
```

### Explicit VJP for Non-Scalar Outputs

For non-scalar outputs, you must provide the upstream gradient explicitly:

```python
x = torch.randn(3, requires_grad=True)
y = torch.sin(x)  # Non-scalar output

# Must provide gradient vector v
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(v)

# x.grad = v^T * J where J = diag(cos(x))
# x.grad[i] = v[i] * cos(x[i])
print(f"x.grad: {x.grad}")
```

## Key Attributes and Methods

| Attribute/Method | Description |
|------------------|-------------|
| `x.requires_grad` | Whether gradients should be computed |
| `x.grad` | Stores computed gradient (None until backward) |
| `x.grad_fn` | Function that created this tensor (None for leaf) |
| `x.is_leaf` | Whether this is a leaf tensor |
| `x.backward()` | Compute gradients |
| `x.retain_grad()` | Request gradient storage for non-leaf |
| `x.detach()` | Return tensor without gradient tracking |

## Common Patterns

### Creating Trainable Parameters

```python
# Method 1: Direct creation
w = torch.randn(3, 4, requires_grad=True)

# Method 2: From existing tensor
w = torch.randn(3, 4)
w.requires_grad_(True)  # In-place modification
```

### Safe Gradient Access

```python
def safe_grad_norm(param):
    """Safely compute gradient norm."""
    if param.grad is not None:
        return param.grad.norm().item()
    return 0.0
```

## Summary

| Concept | Key Points |
|---------|------------|
| Leaf Tensor | Created by user, `grad_fn=None`, stores gradients |
| Non-Leaf Tensor | Result of operations, has `grad_fn` |
| `requires_grad` | Enables gradient tracking; propagates through operations |
| `.backward()` | Computes gradients via VJP; requires scalar for implicit use |
| `.retain_grad()` | Enables gradient storage for non-leaf tensors |
| VJP | Efficient computation of $J^T v$ during backpropagation |

## References

- PyTorch Autograd Documentation: https://pytorch.org/docs/stable/notes/autograd.html
- Automatic Differentiation in Machine Learning: A Survey (Baydin et al., 2018)
