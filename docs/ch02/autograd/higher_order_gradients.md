# Higher-Order Gradients

## Overview

Higher-order derivatives (second derivatives, Hessians, etc.) are essential for understanding loss surface curvature, implementing second-order optimization methods, and computing quantities like the Fisher Information Matrix. PyTorch supports higher-order derivatives through the `create_graph` parameter, which builds a computational graph of the backward pass itself, enabling differentiation of gradients.

## Learning Objectives

By the end of this section, you will be able to:

1. Compute second-order derivatives using `create_graph=True`
2. Understand and compute the Hessian matrix
3. Implement efficient Hessian-vector products (HVPs)
4. Apply higher-order derivatives in optimization, GANs, and meta-learning
5. Reason about computational costs and practical limitations

## Second Derivatives via `create_graph`

### The Key Idea

By default, `backward()` and `torch.autograd.grad()` do **not** build a computational graph for the gradient computation itself. Setting `create_graph=True` changes this: the resulting gradient tensor is itself differentiable, enabling further differentiation.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3    # y = x³

# First derivative WITH graph retention
grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = 3x² = {grad_y}")                    # tensor([12.])
print(f"grad_y.requires_grad: {grad_y.requires_grad}")  # True!

# Second derivative: differentiate the gradient
grad2_y = torch.autograd.grad(grad_y, x)[0]
print(f"d²y/dx² = 6x = {grad2_y}")                  # tensor([12.])
```

**Verification** for $y = x^3$ at $x = 2$:

- $\frac{dy}{dx} = 3x^2 = 3(4) = 12$ ✓
- $\frac{d^2y}{dx^2} = 6x = 6(2) = 12$ ✓

### Using `backward()` with `create_graph`

The same mechanism works with `.backward()`:

```python
import torch

x = torch.tensor([3.0], requires_grad=True)
y = torch.sin(x)

# First backward with create_graph=True
y.backward(create_graph=True)
first_grad = x.grad.clone()    # cos(3) ≈ -0.99

# Zero and differentiate the first gradient
x.grad = None
first_grad.backward()
print(f"d²y/dx² = -sin(x) = {x.grad}")   # -sin(3) ≈ -0.14
```

## The Hessian Matrix

### Definition

For a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the **Hessian** is the $n \times n$ matrix of second-order partial derivatives:

$$H = \nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix} \in \mathbb{R}^{n \times n}$$

The Hessian is symmetric for twice continuously differentiable functions (Schwarz's theorem).

### Computing the Full Hessian

Using `torch.autograd.functional.hessian`:

```python
import torch
from torch.autograd.functional import hessian

def f(x):
    """f(x) = x₁² + x₁x₂ + x₂²"""
    return x[0]**2 + x[0]*x[1] + x[1]**2

x = torch.tensor([1.0, 2.0])
H = hessian(f, x)

print(f"Hessian:\n{H}")
# [[2, 1],
#  [1, 2]]
# ∂²f/∂x₁² = 2,  ∂²f/∂x₁∂x₂ = 1,  ∂²f/∂x₂² = 2
```

### Manual Hessian Computation

Row $i$ of the Hessian is obtained by differentiating the $i$-th component of the gradient:

```python
import torch

def compute_hessian_manual(f, x):
    """Compute Hessian by differentiating the gradient."""
    n = x.numel()
    H = torch.zeros(n, n)
    
    x_grad = x.detach().requires_grad_(True)
    y = f(x_grad)
    grad = torch.autograd.grad(y, x_grad, create_graph=True)[0]
    
    for i in range(n):
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

The full Hessian is $O(n^2)$ in both time and memory — prohibitive for neural networks with millions of parameters. However, a **Hessian-vector product** $Hv$ can be computed in $O(n)$ time using the identity:

$$Hv = \nabla_x \left( (\nabla_x f)^T v \right)$$

This is sufficient for many second-order methods (conjugate gradient, Newton-CG, Lanczos, etc.).

### Implementation

```python
import torch

def hvp(f, x, v):
    """
    Compute Hessian-vector product Hv in O(n) time.
    
    Uses the identity: Hv = ∇(∇f · v)
    """
    x = x.detach().requires_grad_(True)
    
    # Step 1: gradient with graph
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Step 2: directional derivative of gradient
    grad_v = (grad * v).sum()
    Hv = torch.autograd.grad(grad_v, x)[0]
    
    return Hv

# Example: f(x) = ½||x||², H = I
def f(x):
    return 0.5 * (x ** 2).sum()

x = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([1.0, 0.0, 0.0])

Hv = hvp(f, x, v)
print(f"Hv = {Hv}")  # tensor([1., 0., 0.]) since H = I
```

### Using `torch.autograd.functional.hvp`

```python
from torch.autograd.functional import hvp as torch_hvp

def f(x):
    return (x ** 3).sum()

x = torch.tensor([1.0, 2.0])
v = torch.tensor([1.0, 1.0])

_, Hv = torch_hvp(f, x, v)
print(f"Hv = {Hv}")
# H = diag(6x), so Hv = [6·1·1, 6·2·1] = [6, 12]
```

## Applications

### Newton's Method with Conjugate Gradient

Newton's method updates parameters using the Hessian: $x_{k+1} = x_k - H^{-1} \nabla f$. For large-scale problems, use HVP with conjugate gradient to solve $Hs = -\nabla f$ without forming $H$ explicitly:

```python
import torch

def newton_step_cg(f, x, max_iter=10, tol=1e-5):
    """
    Approximate Newton step via conjugate gradient with HVP.
    Solves H @ step = -grad for step.
    """
    x = x.detach().requires_grad_(True)
    y = f(x)
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    
    b = -grad.detach()
    step = torch.zeros_like(x)
    r = b.clone()
    p = r.clone()
    
    for _ in range(max_iter):
        # HVP: H @ p
        grad_p = (grad * p).sum()
        Hp = torch.autograd.grad(grad_p, x, retain_graph=True)[0]
        
        alpha = (r @ r) / (p @ Hp + 1e-8)
        step = step + alpha * p
        r_new = r - alpha * Hp
        
        if r_new.norm() < tol:
            break
        
        beta = (r_new @ r_new) / (r @ r + 1e-8)
        p = r_new + beta * p
        r = r_new
    
    return step

# Minimize f(x) = ½ xᵀAx
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
def f(x):
    return 0.5 * x @ A @ x

x = torch.tensor([1.0, 1.0])
step = newton_step_cg(f, x)
x_new = x + step
print(f"Newton step: {step}")
print(f"x_new (should be near [0,0]): {x_new}")
```

### Gradient Penalty (WGAN-GP)

WGAN-GP penalizes the discriminator's gradient norm, requiring differentiation through the gradient itself:

```python
import torch
import torch.nn as nn

def gradient_penalty(discriminator, real, fake):
    """
    WGAN-GP gradient penalty: ((||∇D(x̃)||₂ - 1)²)
    where x̃ is a random interpolation between real and fake.
    """
    batch_size = real.size(0)
    
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,     # Need graph so penalty is differentiable
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return penalty
```

### Fisher Information Matrix (Diagonal Approximation)

The Fisher Information Matrix is related to the Hessian of the negative log-likelihood. The diagonal approximation is computationally tractable:

```python
import torch
import torch.nn as nn

def compute_fisher_diagonal(model, x, y, criterion):
    """Compute diagonal of empirical Fisher: F ≈ E[∇log p · (∇log p)ᵀ]."""
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    fisher_diag = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher_diag[name] = param.grad ** 2
    
    return fisher_diag

model = nn.Linear(10, 2)
x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

fisher = compute_fisher_diagonal(model, x, y, nn.CrossEntropyLoss())
for name, diag in fisher.items():
    print(f"{name}: shape={diag.shape}, mean={diag.mean():.4f}")
```

## Higher-Order Derivatives (Third and Beyond)

The pattern extends to arbitrary order by chaining `create_graph=True`:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 4    # y = x⁴

grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx   = 4x³  = {grad1}")     # 32

grad2 = torch.autograd.grad(grad1, x, create_graph=True)[0]
print(f"d²y/dx² = 12x² = {grad2}")     # 48

grad3 = torch.autograd.grad(grad2, x, create_graph=True)[0]
print(f"d³y/dx³ = 24x  = {grad3}")     # 48

grad4 = torch.autograd.grad(grad3, x)[0]
print(f"d⁴y/dx⁴ = 24   = {grad4}")     # 24
```

## Computational Considerations

### Memory Cost

Each `create_graph=True` call stores an additional layer of computational graph:

| Derivative Order | Memory Overhead |
|-----------------|-----------------|
| First | $O(T)$ — activations |
| Second | $O(T^2)$ — graph of graph |
| $k$-th | $O(T^k)$ |

### Practical Recommendations

1. **Use HVP instead of the full Hessian** — $O(n)$ vs $O(n^2)$
2. **Limit to second order** for most applications
3. **Consider approximations** — diagonal Hessian, KFAC, empirical Fisher
4. **Use `retain_graph=True` carefully** — release as soon as possible to avoid memory leaks

## Summary

| Concept | Implementation |
|---------|----------------|
| **Second derivative** | `torch.autograd.grad(grad, x, create_graph=True)` |
| **Full Hessian** | `torch.autograd.functional.hessian(f, x)` |
| **Hessian-vector product** | `torch.autograd.functional.hvp(f, x, v)` |
| **Key parameter** | `create_graph=True` enables higher-order differentiation |
| **Memory** | Scales exponentially with derivative order |

## Common Use Cases

- **Newton's method / Natural gradient** — second-order optimization
- **WGAN-GP** — gradient penalty requiring differentiable gradient norm
- **Physics-informed neural networks** — PDE constraints involving spatial/temporal derivatives
- **Meta-learning (MAML)** — differentiating through the inner optimization loop
- **Loss surface analysis** — Hessian eigenvalues characterize curvature

## References

- Martens, J. (2010). Deep Learning via Hessian-free Optimization. *ICML*.
- Pearlmutter, B.A. (1994). Fast Exact Multiplication by the Hessian. *Neural Computation*.
- Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. *NeurIPS*.
- PyTorch Higher-Order Gradients: [https://pytorch.org/tutorials/intermediate/per_sample_grads.html](https://pytorch.org/tutorials/intermediate/per_sample_grads.html)
