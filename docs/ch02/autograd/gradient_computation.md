# Gradient Computation

## Overview

PyTorch's autograd engine is the foundation of automatic differentiation in deep learning. It provides a tape-based system that records operations on tensors and computes gradients through **reverse-mode automatic differentiation** (backpropagation). This section covers the mathematical framework behind gradient computation, the distinction between forward and reverse mode AD, and how PyTorch's vector-Jacobian product (VJP) machinery implements the chain rule efficiently.

## Learning Objectives

By the end of this section, you will be able to:

1. Compute gradients using `.backward()` and `torch.autograd.grad`
2. Understand the Jacobian matrix and vector-Jacobian product (VJP) framework
3. Distinguish between forward mode (JVP) and reverse mode (VJP) automatic differentiation
4. Explain why reverse mode is preferred for neural network training
5. Compute full Jacobian matrices using both modes

## Mathematical Foundation

### The Chain Rule

Automatic differentiation systematically applies the chain rule. For a composite function $y = f(g(x))$:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

For a sequence of operations $x \rightarrow z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow z_n \rightarrow y$:

$$\frac{dy}{dx} = \frac{dy}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdot \ldots \cdot \frac{dz_2}{dz_1} \cdot \frac{dz_1}{dx}$$

The **order** in which these matrix products are evaluated determines the mode of AD.

### The Jacobian Matrix

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, the **Jacobian** is the matrix of all first-order partial derivatives:

$$J = \frac{\partial f}{\partial x} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

For a composition $y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$, the full Jacobian is:

$$J = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$$

## Computing Gradients in PyTorch

### Scalar Loss: The `.backward()` Method

For a scalar loss $L: \mathbb{R}^n \rightarrow \mathbb{R}$, calling `L.backward()` computes the gradient $\nabla_x L$ for all leaf tensors $x$ with `requires_grad=True`:

$$\nabla_x L = \left[\frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, \ldots, \frac{\partial L}{\partial x_n}\right]^T$$

```python
import torch

torch.manual_seed(0)

x = torch.randn(3, requires_grad=True)
print(f"x: {x}")

# Forward pass: loss = sum(x²)
loss = (x ** 2).sum()
print(f"loss: {loss}")

# Backward pass: d(loss)/dx = 2x
loss.backward()

print(f"x.grad: {x.grad}")
print(f"Expected (2x): {2 * x.detach()}")
print(f"Match: {torch.allclose(x.grad, 2 * x.detach())}")
```

**Output:**
```
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
loss: tensor(7.2274, grad_fn=<SumBackward0>)
x.grad: tensor([ 3.0820, -0.5868, -4.3576])
Expected (2x): tensor([ 3.0820, -0.5868, -4.3576])
Match: True
```

### Gradient Storage: Leaf vs Non-Leaf

By default, gradients are stored **only** for leaf tensors:

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x         # Non-leaf
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # tensor([ 8., 16., 24.]) — stored
print(f"y.grad: {y.grad}")  # None — not stored by default
```

Use `.retain_grad()` before backward to store gradients for non-leaf tensors:

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

**Verification:** With $y = 2x = [2, 4, 6]$ and $z = \sum y_i^2$:

- $\frac{\partial z}{\partial y_i} = 2y_i$, giving $\nabla_y z = [4, 8, 12]$
- $\frac{\partial z}{\partial x_i} = \frac{\partial z}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_i} = 2y_i \cdot 2 = 4y_i$, giving $\nabla_x z = [8, 16, 24]$

### The `torch.autograd.grad` Interface

For finer control, `torch.autograd.grad` returns gradients directly without populating `.grad` attributes:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# Returns a tuple of gradients
(grad_y,) = torch.autograd.grad(y, x)
print(f"dy/dx = 3x² = {grad_y}")  # tensor([12.])
```

This is particularly useful for higher-order derivatives (see [Higher-Order Gradients](higher_order_gradients.md)) and when you need gradients for non-leaf tensors without calling `retain_grad()`.

## The Vector-Jacobian Product Framework

### Why VJPs?

PyTorch's backward pass does **not** compute the full Jacobian matrix — that would be prohibitively expensive for high-dimensional parameter spaces. Instead, it computes **vector-Jacobian products** (VJPs), which is all that's needed for gradient-based optimization.

### Mathematical Definition

Given a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with Jacobian $J \in \mathbb{R}^{m \times n}$, and an upstream gradient (adjoint) $\bar{y} \in \mathbb{R}^m$:

$$\text{VJP:} \quad \bar{x} = J^T \bar{y} \in \mathbb{R}^n$$

Equivalently, $\bar{x}^T = \bar{y}^T J$. This is the chain rule in matrix form: $\bar{y}$ carries the gradient from downstream, and multiplication by $J^T$ propagates it one step further upstream.

### Implicit VJP for Scalar Outputs

When the output is scalar ($m = 1$), the Jacobian reduces to a row vector (the gradient), and the upstream "vector" is the scalar $\bar{y} = 1$. This is why `loss.backward()` needs no argument:

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x ** 2).sum()

# Equivalent calls:
# loss.backward()
loss.backward(torch.tensor(1.0))   # Explicit v = 1

print(f"x.grad: {x.grad}")  # tensor([2., 4., 6.])
```

### Explicit VJP for Non-Scalar Outputs

For non-scalar outputs, you must supply the upstream gradient vector $\bar{y}$:

```python
import torch

x = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)
y = torch.sin(x)   # y: R³ → R³, elementwise

# Must provide gradient vector v with same shape as y
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(v)

# For elementwise sin, J = diag(cos(x))
# x.grad = J^T v = v ⊙ cos(x)
expected = v * torch.cos(x.detach())
print(f"x.grad:   {x.grad}")
print(f"Expected: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

Without `v`, calling `.backward()` on a non-scalar tensor raises `RuntimeError`:

```python
x = torch.randn(3, requires_grad=True)
y = torch.sin(x)

try:
    y.backward()  # Fails — non-scalar output
except RuntimeError as e:
    print(f"Error: {e}")
```

### VJP for Linear Transformations

For a linear map $y = Ax$ where $A \in \mathbb{R}^{m \times n}$, the Jacobian is simply $J = A$:

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

## Forward vs Reverse Mode AD

The Jacobian product $J = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$ can be evaluated in two orders, each defining a mode of automatic differentiation.

### Forward Mode: Jacobian-Vector Product (JVP)

Forward mode propagates derivatives **from inputs to outputs**, computing $\dot{y} = J \cdot \dot{x}$ where $\dot{x}$ is a tangent vector at the input:

**Evaluation order (right to left):**
$$J \cdot \dot{x} = J_L \cdot (J_{L-1} \cdot (\ldots \cdot (J_2 \cdot (J_1 \cdot \dot{x}))))$$

One forward pass with a specific $\dot{x}$ yields one directional derivative. To obtain the full Jacobian for $n$ inputs requires **$n$ forward passes**.

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

# JVP with tangent [1, 0] gives first column of Jacobian
tangent = torch.tensor([1.0, 0.0])
output, jvp_result = jvp(f, (x,), (tangent,))

print(f"f(x) = {output}")
print(f"J @ [1,0] = {jvp_result}")  # First column of J
```

### Reverse Mode: Vector-Jacobian Product (VJP)

Reverse mode propagates derivatives **from outputs to inputs**, computing $\bar{x} = J^T \bar{y}$:

**Evaluation order (left to right):**
$$\bar{x}^T = ((((\bar{y}^T \cdot J_L) \cdot J_{L-1}) \cdot \ldots) \cdot J_1)$$

One backward pass with a specific $\bar{y}$ yields one row of $J^T$ (equivalently one column of $J$). For **scalar** output ($m = 1$), a **single backward pass** gives the full gradient.

```python
import torch
from torch.autograd.functional import vjp

def f(x):
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# VJP with adjoint [1, 0] gives first row of J
adjoint = torch.tensor([1.0, 0.0])
output, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(adjoint)[0]

print(f"f(x) = {output}")
print(f"[1,0] @ J = {vjp_result}")
```

### Why Reverse Mode for Neural Networks

In deep learning, the typical setting is millions of parameters ($n \gg 1$) mapped to a single scalar loss ($m = 1$):

| Mode | Passes Required | Complexity |
|------|-----------------|------------|
| Forward | $n$ (one per parameter) | $O(n \cdot T)$ |
| **Reverse** | $m = 1$ | $O(T)$ |

where $T$ is the cost of one forward pass. Reverse mode is **dramatically more efficient**, which is why PyTorch defaults to backpropagation.

```python
import torch
import torch.nn as nn

# Network with ~200K parameters
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

logits = model(x)
loss = nn.functional.cross_entropy(logits, y)

# Single backward pass computes gradients for ALL parameters
loss.backward()

for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
```

### Memory-Compute Tradeoff

Reverse mode requires **storing intermediate activations** for the backward pass, leading to memory overhead proportional to network depth:

| Aspect | Forward Mode | Reverse Mode |
|--------|--------------|--------------|
| **Propagation** | Input → Output | Output → Input |
| **Computes** | JVP: $J \dot{x}$ | VJP: $J^T \bar{y}$ |
| **Efficient when** | $n \ll m$ | $m \ll n$ |
| **Full Jacobian** | $n$ passes | $m$ passes |
| **Extra memory** | $O(1)$ | $O(T)$ (activations) |

For very deep networks, **gradient checkpointing** trades compute for memory by recomputing activations during the backward pass instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

def expensive_layer(x):
    return torch.relu(x @ x.T)

x = torch.randn(1000, 1000, requires_grad=True)

# Checkpointing: recomputes activations during backward
y = checkpoint(expensive_layer, x, use_reentrant=False)
```

## Computing Full Jacobians

### Using `torch.autograd.functional.jacobian`

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

### Manual Jacobian via VJP (Row by Row)

Each backward pass with a one-hot adjoint $e_i$ extracts row $i$ of the Jacobian:

```python
import torch

def compute_jacobian_via_vjp(f, x):
    """Build Jacobian row by row using reverse-mode VJPs."""
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

### Manual Jacobian via JVP (Column by Column)

Each forward pass with a one-hot tangent $e_j$ extracts column $j$:

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

## Key Attributes and Methods

| Attribute / Method | Description |
|--------------------|-------------|
| `x.requires_grad` | Whether operations on `x` are tracked for differentiation |
| `x.grad` | Accumulated gradient (populated after `.backward()`) |
| `x.grad_fn` | The backward function that created `x` (`None` for leaf tensors) |
| `x.is_leaf` | `True` if `x` is a leaf tensor |
| `x.backward(v)` | Compute VJP and accumulate into `.grad` |
| `x.retain_grad()` | Request gradient storage for non-leaf `x` |
| `x.detach()` | Return a tensor sharing data but detached from the graph |
| `torch.autograd.grad(y, x)` | Compute gradient without populating `.grad` |

## Summary

| Concept | Key Points |
|---------|------------|
| **Gradient** | $\nabla_x L$ computed via reverse-mode AD for scalar $L$ |
| **VJP** | $J^T \bar{y}$ — what PyTorch computes during backpropagation |
| **JVP** | $J \dot{x}$ — forward-mode, efficient for few inputs |
| **Reverse mode** | One backward pass for scalar loss; $O(T)$ cost |
| **Forward mode** | $n$ forward passes needed; $O(1)$ extra memory |
| **Full Jacobian** | $m$ backward passes (reverse) or $n$ forward passes |
| **Gradient storage** | By default, only leaf tensors store `.grad` |

## References

- Baydin, A.G., et al. (2018). Automatic Differentiation in Machine Learning: A Survey. *JMLR*.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*.
- PyTorch Autograd Documentation: [https://pytorch.org/docs/stable/autograd.html](https://pytorch.org/docs/stable/autograd.html)
