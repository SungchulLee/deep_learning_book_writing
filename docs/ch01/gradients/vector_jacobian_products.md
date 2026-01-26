# Vector-Jacobian Products

## Overview

When outputs are non-scalar (vectors or tensors), PyTorch's `.backward()` method requires an explicit gradient argument. This section explores vector-Jacobian products (VJPs), their mathematical foundation, and practical applications including computing full Jacobians and handling batched operations.

## Learning Objectives

1. Understand why non-scalar outputs require explicit gradients
2. Compute VJPs using `backward(v)` and `torch.autograd.grad`
3. Build full Jacobian matrices using VJPs
4. Handle batched VJP computations

## Mathematical Foundation

### The Jacobian Matrix

For $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, the Jacobian is:

$$J = \frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}$$

### Vector-Jacobian Product

Given upstream gradient $v \in \mathbb{R}^m$, the VJP computes:

$$v^T J = \sum_i v_i \frac{\partial f_i}{\partial x} \in \mathbb{R}^n$$

This is what `x.grad` receives during backpropagation.

## Basic VJP Computation

### Why Scalar Loss Works Without Arguments

```python
import torch

x = torch.randn(3, requires_grad=True)
loss = (x ** 2).sum()  # Scalar output

# No argument needed - implicit v = 1
loss.backward()
print(f"x.grad: {x.grad}")
```

### Non-Scalar Outputs Require v

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.sin(x)  # Vector output (shape [3])

# This fails!
try:
    y.backward()
except RuntimeError as e:
    print(f"Error: {e}")

# Must provide v with same shape as y
v = torch.tensor([0.1, 1.0, 0.01])
y.backward(v)
print(f"x.grad: {x.grad}")  # v^T * J where J = diag(cos(x))
```

### Mathematical Verification

For $y = \sin(x)$:
- Jacobian: $J = \text{diag}(\cos(x))$ (diagonal for elementwise ops)
- $v^T J = [v_1 \cos(x_1), v_2 \cos(x_2), v_3 \cos(x_3)]$

```python
import torch

x = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)
y = torch.sin(x)
v = torch.tensor([0.1, 1.0, 0.01])

y.backward(v)

expected = v * torch.cos(x.detach())
print(f"x.grad: {x.grad}")
print(f"Expected: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

## Linear Transformations

For linear $y = Ax$, the Jacobian is simply $J = A$:

```python
import torch

A = torch.tensor([[2.0, 0.0, -1.0],
                  [0.5, 3.0,  1.0]])  # Shape (2, 3)
x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)
y = A @ x  # Shape (2,)

v = torch.tensor([3.0, -1.0])
y.backward(v)

# x.grad = A^T @ v
expected = A.T @ v
print(f"x.grad: {x.grad}")
print(f"A^T @ v: {expected}")
print(f"Match: {torch.allclose(x.grad, expected)}")
```

## Computing Full Jacobians

### Using torch.autograd.functional

```python
import torch
from torch.autograd.functional import jacobian

def f(x):
    return torch.stack([
        x[0] * x[1] + x[2],
        torch.sin(x[0]) + x[1]**2
    ])

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, x)
print(f"Jacobian shape: {J.shape}")  # (2, 3)
print(f"Jacobian:\n{J}")
```

### Manual Construction via VJPs

```python
import torch

def compute_jacobian_via_vjp(f, x):
    """Build Jacobian row by row using VJPs."""
    y = f(x.detach().requires_grad_(True))
    m, n = y.numel(), x.numel()
    J = torch.zeros(m, n)
    
    for i in range(m):
        x_copy = x.detach().requires_grad_(True)
        y_copy = f(x_copy)
        
        # One-hot vector selects row i
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
# Expected:
# [[4, 0],    # d(x1^2)/d(x1, x2)
#  [3, 2],   # d(x1*x2)/d(x1, x2)
#  [0, 6]]   # d(x2^2)/d(x1, x2)
```

## Batched Operations

### Batched Linear VJP

```python
import torch

B, m, n = 2, 3, 2
A = torch.randn(n, m)  # Shared weight matrix

# Batched inputs (treating as trainable)
x_batch = torch.randn(B, m, requires_grad=True)

# Batched forward: output shape (B, n)
y_batch = x_batch @ A.T

# Batched upstream gradients
v_batch = torch.randn(B, n)

# VJP
y_batch.backward(v_batch)

# Expected: grad[b] = v_batch[b] @ A
expected = v_batch @ A
print(f"x_batch.grad:\n{x_batch.grad}")
print(f"Expected:\n{expected}")
```

## Practical Applications

### Computing Gradient of Loss w.r.t. Intermediate Layers

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

x = torch.randn(32, 10)
y_true = torch.randint(0, 10, (32,))

# Get intermediate activations
activations = []
def hook(module, input, output):
    activations.append(output)

model[1].register_forward_hook(hook)

# Forward pass
logits = model(x)
loss = nn.functional.cross_entropy(logits, y_true)

# Backward with create_graph for intermediate gradients
loss.backward()

# Activations after ReLU were captured
hidden = activations[0]
print(f"Hidden activation shape: {hidden.shape}")
```

### Gradient Penalty (WGAN-GP)

```python
import torch

def gradient_penalty(critic, real, fake, device):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    critic_output = critic(interpolated)
    
    # Need gradients w.r.t. interpolated (not scalar loss!)
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
```

## Summary

| Scenario | Method | Returns |
|----------|--------|---------|
| Scalar loss | `loss.backward()` | Gradients in `.grad` |
| Vector output | `y.backward(v)` | VJP: $v^T J$ in `.grad` |
| Full Jacobian | `jacobian(f, x)` | $J \in \mathbb{R}^{m \times n}$ |
| Intermediate gradients | `torch.autograd.grad` | Specified outputs |

## Common Pitfalls

1. **Shape mismatch**: `v` must have same shape as output
2. **Forgetting `create_graph`**: Needed for gradients of gradients
3. **Graph consumption**: Use `retain_graph=True` for multiple VJPs

## References

- PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
- Automatic Differentiation in ML: A Survey (Baydin et al., 2018)
