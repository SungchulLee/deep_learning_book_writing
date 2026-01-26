# Higher-Order Derivatives

## Overview

Higher-order derivatives (second derivatives, Hessians, etc.) are essential for understanding loss surface curvature, implementing second-order optimization methods, and computing quantities like the Fisher Information Matrix. PyTorch supports higher-order derivatives through the `create_graph` parameter, enabling differentiation of gradients themselves.

## Learning Objectives

By the end of this section, you will be able to:

1. Compute second-order derivatives using `create_graph=True`
2. Understand the Hessian matrix and its computation
3. Implement Hessian-vector products efficiently
4. Apply higher-order derivatives in optimization contexts
5. Understand computational costs and practical limitations

## First-Order Review

Before diving into higher-order derivatives, let's review the standard gradient computation:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³

# First derivative: dy/dx = 3x²
y.backward()
print(f"First derivative at x=2: {x.grad}")  # tensor([12.]) = 3(2)² = 12
```

By default, the gradient computation does **not** create a computational graph for the gradients themselves.

## Computing Second Derivatives

### The `create_graph` Parameter

To compute higher-order derivatives, use `create_graph=True`:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³

# Compute first derivative WITH graph for the gradient
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 3x² = {grad_y}")  # tensor([12.])
print(f"grad_y.requires_grad: {grad_y.requires_grad}")  # True!

# Now differentiate the gradient itself
grad2_y = torch.autograd.grad(grad_y, x)[0]
print(f"d²y/dx² = 6x = {grad2_y}")  # tensor([12.]) = 6(2) = 12
```

### Mathematical Verification

For $y = x^3$:
- First derivative: $\frac{dy}{dx} = 3x^2$
- Second derivative: $\frac{d^2y}{dx^2} = 6x$

At $x = 2$: $\frac{d^2y}{dx^2} = 6(2) = 12$ ✓

### Using `backward()` with `create_graph`

```python
import torch

x = torch.tensor([3.0], requires_grad=True)
y = torch.sin(x)  # y = sin(x)

# First backward with create_graph=True
y.backward(create_graph=True)
print(f"dy/dx = cos(x) = {x.grad}")  # cos(3) ≈ -0.99

# The gradient is now part of the graph
first_grad = x.grad.clone()

# Zero gradient before second backward
x.grad = None

# Differentiate the first gradient
first_grad.backward()
print(f"d²y/dx² = -sin(x) = {x.grad}")  # -sin(3) ≈ -0.14
```

## The Hessian Matrix

### Definition

For a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the **Hessian** is the matrix of second-order partial derivatives:

$$H = \nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} \in \mathbb{R}^{n \times n}$$

### Computing the Full Hessian

```python
import torch
from torch.autograd.functional import hessian

def f(x):
    """f(x) = x₁² + x₁x₂ + x₂²"""
    return x[0]**2 + x[0]*x[1] + x[1]**2

x = torch.tensor([1.0, 2.0])
H = hessian(f, x)

print(f"Hessian:\n{H}")
# Expected:
# [[2, 1],
#  [1, 2]]
# Because ∂²f/∂x₁² = 2, ∂²f/∂x₁∂x₂ = 1, ∂²f/∂x₂² = 2
```

### Manual Hessian Computation

```python
import torch

def compute_hessian_manual(f, x):
    """Compute Hessian by differentiating the gradient."""
    n = x.numel()
    H = torch.zeros(n, n)
    
    # Compute gradient with graph
    x_grad = x.detach().requires_grad_(True)
    y = f(x_grad)
    grad = torch.autograd.grad(y, x_grad, create_graph=True)[0]
    
    # Compute each row of Hessian
    for i in range(n):
        # Differentiate i-th component of gradient
        grad2 = torch.autograd.grad(
            grad[i], x_grad, 
            retain_graph=True,
            create_graph=False
        )[0]
        H[i] = grad2
    
    return H

def f(x):
    return x[0]**3 + x[0]*x[1]**2 + torch.sin(x[1])

x = torch.tensor([1.0, 2.0])
H = compute_hessian_manual(f, x)
print(f"Hessian:\n{H}")
```

## Hessian-Vector Products

### Motivation

Computing the full Hessian is $O(n^2)$ in both time and memory, prohibitive for neural networks with millions of parameters. However, **Hessian-vector products** can be computed in $O(n)$:

$$Hv = \nabla(\nabla f \cdot v)$$

This is sufficient for many second-order methods (conjugate gradient, Newton-CG, etc.).

### Implementation

```python
import torch

def hvp(f, x, v):
    """
    Compute Hessian-vector product: H @ v
    
    Args:
        f: Scalar function
        x: Point at which to compute Hessian
        v: Vector to multiply
    
    Returns:
        Hv: Hessian-vector product
    """
    x = x.detach().requires_grad_(True)
    
    # Compute gradient
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Compute gradient-vector dot product
    grad_v = (grad * v).sum()
    
    # Differentiate to get Hv
    Hv = torch.autograd.grad(grad_v, x)[0]
    
    return Hv

# Example
def f(x):
    return 0.5 * (x ** 2).sum()  # f(x) = 0.5 * ||x||², H = I

x = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([1.0, 0.0, 0.0])

Hv = hvp(f, x, v)
print(f"Hv = {Hv}")  # Should be [1, 0, 0] since H = I
```

### Efficient HVP with `torch.autograd.functional`

```python
from torch.autograd.functional import hvp as torch_hvp

def f(x):
    return (x ** 3).sum()

x = torch.tensor([1.0, 2.0])
v = torch.tensor([1.0, 1.0])

_, Hv = torch_hvp(f, x, v)
print(f"Hv = {Hv}")
# H = diag(6x), so Hv = [6*1*1, 6*2*1] = [6, 12]
```

## Applications

### Newton's Method

Newton's method uses the Hessian for optimization:

$$x_{k+1} = x_k - H^{-1} \nabla f(x_k)$$

For large-scale problems, use HVP with conjugate gradient instead of computing $H^{-1}$:

```python
import torch

def newton_step_cg(f, x, max_iter=10, tol=1e-5):
    """
    Approximate Newton step using conjugate gradient with HVP.
    Solves H @ step = -grad for step.
    """
    x = x.detach().requires_grad_(True)
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Target: solve H @ step = -grad
    b = -grad.detach()
    step = torch.zeros_like(x)
    r = b.clone()  # residual
    p = r.clone()  # search direction
    
    for _ in range(max_iter):
        # Compute Hp
        grad_p = (grad * p).sum()
        Hp = torch.autograd.grad(grad_p, x, retain_graph=True)[0]
        
        # CG update
        alpha = (r @ r) / (p @ Hp + 1e-8)
        step = step + alpha * p
        r_new = r - alpha * Hp
        
        if r_new.norm() < tol:
            break
            
        beta = (r_new @ r_new) / (r @ r + 1e-8)
        p = r_new + beta * p
        r = r_new
    
    return step

# Example: minimize f(x) = 0.5 * x^T A x
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
def f(x):
    return 0.5 * x @ A @ x

x = torch.tensor([1.0, 1.0])
step = newton_step_cg(f, x)
x_new = x + step
print(f"Step: {step}")
print(f"New x (should be near [0,0]): {x_new}")
```

### Fisher Information Matrix

The Fisher Information Matrix is related to the Hessian of the negative log-likelihood:

```python
import torch
import torch.nn as nn

def compute_fisher_diagonal(model, x, y, criterion):
    """
    Compute diagonal of Fisher Information Matrix.
    For classification: F ≈ E[∇log p(y|x) ∇log p(y|x)^T]
    """
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    fisher_diag = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher_diag[name] = param.grad ** 2
    
    return fisher_diag

# Example usage
model = nn.Linear(10, 2)
x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

fisher = compute_fisher_diagonal(model, x, y, nn.CrossEntropyLoss())
for name, diag in fisher.items():
    print(f"{name}: shape={diag.shape}, mean={diag.mean():.4f}")
```

### Gradient Penalty (WGAN-GP)

WGAN-GP requires computing gradients of gradients:

```python
import torch
import torch.nn as nn

def gradient_penalty(discriminator, real, fake):
    """
    Compute gradient penalty for WGAN-GP.
    Penalizes gradients of D that deviate from unit norm.
    """
    batch_size = real.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    # Discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients (with graph for penalty gradient)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,  # Need graph for penalty gradient
        retain_graph=True
    )[0]
    
    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty: (||∇D|| - 1)²
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

## Third and Higher Derivatives

The pattern extends to any order:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 4  # y = x⁴

# dy/dx = 4x³
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 4x³ = {grad1}")  # 32

# d²y/dx² = 12x²
grad2 = torch.autograd.grad(grad1, x, create_graph=True)[0]
print(f"d²y/dx² = 12x² = {grad2}")  # 48

# d³y/dx³ = 24x
grad3 = torch.autograd.grad(grad2, x, create_graph=True)[0]
print(f"d³y/dx³ = 24x = {grad3}")  # 48

# d⁴y/dx⁴ = 24 (constant)
grad4 = torch.autograd.grad(grad3, x)[0]
print(f"d⁴y/dx⁴ = 24 = {grad4}")  # 24
```

## Computational Considerations

### Memory Cost

Each `create_graph=True` call stores additional computation graph:

| Order | Memory Overhead |
|-------|-----------------|
| First derivative | $O(T)$ - activations |
| Second derivative | $O(T^2)$ - graph of graph |
| $k$-th derivative | $O(T^k)$ |

### Practical Recommendations

1. **Use HVP instead of full Hessian** when possible
2. **Limit to second-order** for most applications
3. **Consider approximations** (diagonal Hessian, KFAC, etc.)
4. **Use `retain_graph=True`** carefully to avoid memory leaks

## Summary

| Concept | Implementation |
|---------|----------------|
| **Second Derivative** | `torch.autograd.grad(grad, x, create_graph=True)` |
| **Hessian** | `torch.autograd.functional.hessian(f, x)` |
| **HVP** | `torch.autograd.functional.hvp(f, x, v)` |
| **Key Parameter** | `create_graph=True` enables higher-order |
| **Memory** | Scales exponentially with derivative order |

## Common Use Cases

- **Newton's method**: Second-order optimization
- **Natural gradient**: Fisher Information Matrix
- **WGAN-GP**: Gradient penalty term
- **Physics-informed NNs**: PDE constraints involving derivatives
- **Meta-learning**: Differentiating through optimization

## References

- Martens, J. (2010). Deep Learning via Hessian-free Optimization. ICML.
- Pearlmutter, B.A. (1994). Fast Exact Multiplication by the Hessian. Neural Computation.
- PyTorch Higher-Order Gradients: https://pytorch.org/tutorials/intermediate/per_sample_grads.html
