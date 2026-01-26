# Forward and Reverse Mode Automatic Differentiation

## Overview

Automatic differentiation (AD) is the foundation of gradient-based optimization in deep learning. There are two fundamental modes: **forward mode** and **reverse mode** (backpropagation). Understanding both modes illuminates why PyTorch uses reverse mode for training neural networks and when each approach is computationally advantageous.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the mathematical foundation of automatic differentiation
2. Distinguish between forward and reverse mode AD
3. Analyze the computational complexity of each mode
4. Explain why reverse mode is preferred for neural network training
5. Implement basic examples demonstrating both modes

## Mathematical Foundation

### The Chain Rule

Automatic differentiation is based on the chain rule. For a composite function:

$$y = f(g(x))$$

The derivative is:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

For a sequence of operations $x \rightarrow z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow z_n \rightarrow y$:

$$\frac{dy}{dx} = \frac{dy}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdot \ldots \cdot \frac{dz_2}{dz_1} \cdot \frac{dz_1}{dx}$$

### Jacobian Matrices

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, the **Jacobian** is:

$$J = \frac{\partial f}{\partial x} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

For a composition $y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$, the full Jacobian is:

$$J = J_L \cdot J_{L-1} \cdot \ldots \cdot J_1$$

The **order of multiplication** determines the mode of AD.

## Forward Mode AD

### Concept

Forward mode computes derivatives **from inputs to outputs**, propagating a "tangent" alongside the primal computation.

**Computation Order:**
$$\frac{dy}{dx} = J_L \cdot (J_{L-1} \cdot (\ldots \cdot (J_2 \cdot J_1)))$$

Parenthesization shows right-to-left (forward) accumulation.

### Jacobian-Vector Product (JVP)

Forward mode efficiently computes the **Jacobian-vector product**:

$$\dot{y} = J \cdot \dot{x}$$

where $\dot{x}$ is a "tangent vector" at the input.

**Complexity:** For $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, one forward pass with a specific $\dot{x}$ gives one column of the Jacobian. To get the full gradient with respect to $n$ inputs requires $n$ forward passes.

### Implementation Example

```python
import torch
from torch.autograd.functional import jvp

def f(x):
    """Example function: f(x) = [sin(x1*x2), x1^2 + x2]"""
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0])

# Forward mode: JVP with tangent vector [1, 0] gives first column of Jacobian
tangent = torch.tensor([1.0, 0.0])
output, jvp_result = jvp(f, (x,), (tangent,))

print(f"f(x) = {output}")
print(f"J @ [1,0] = {jvp_result}")  # First column of Jacobian

# To get full Jacobian, need n forward passes (n=2 here)
J_col1 = jvp(f, (x,), (torch.tensor([1.0, 0.0]),))[1]
J_col2 = jvp(f, (x,), (torch.tensor([0.0, 1.0]),))[1]
J_forward = torch.stack([J_col1, J_col2], dim=1)

print(f"Full Jacobian (forward mode):\n{J_forward}")
```

### When Forward Mode is Efficient

Forward mode is efficient when:
- **Few inputs, many outputs**: $n \ll m$
- Computing directional derivatives
- Sensitivity analysis with specific input perturbations

**Cost:** $O(n)$ forward passes for full Jacobian

## Reverse Mode AD (Backpropagation)

### Concept

Reverse mode computes derivatives **from outputs to inputs**, propagating an "adjoint" (gradient) backward through the computation.

**Computation Order:**
$$\frac{dy}{dx} = (((J_L \cdot J_{L-1}) \cdot J_{L-2}) \cdot \ldots) \cdot J_1$$

Parenthesization shows left-to-right (backward) accumulation.

### Vector-Jacobian Product (VJP)

Reverse mode efficiently computes the **vector-Jacobian product**:

$$\bar{x}^T = \bar{y}^T \cdot J \quad \Leftrightarrow \quad \bar{x} = J^T \bar{y}$$

where $\bar{y}$ is the "adjoint" (upstream gradient).

**Complexity:** For $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, one backward pass with a specific $\bar{y}$ gives one row of $J^T$ (equivalently, one column of $J$). To get the full Jacobian with respect to $m$ outputs requires $m$ backward passes.

For a scalar output ($m=1$), **one backward pass** gives the full gradient!

### Implementation Example

```python
import torch
from torch.autograd.functional import vjp

def f(x):
    """Example function: f(x) = [sin(x1*x2), x1^2 + x2]"""
    return torch.stack([
        torch.sin(x[0] * x[1]),
        x[0]**2 + x[1]
    ])

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Reverse mode: VJP with adjoint [1, 0] gives first row of Jacobian
adjoint = torch.tensor([1.0, 0.0])
output, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(adjoint)[0]

print(f"f(x) = {output}")
print(f"[1,0] @ J = {vjp_result}")  # First row of Jacobian

# To get full Jacobian, need m backward passes (m=2 here)
J_row1 = vjp(f, x.detach().requires_grad_(True))[1](torch.tensor([1.0, 0.0]))[0]
J_row2 = vjp(f, x.detach().requires_grad_(True))[1](torch.tensor([0.0, 1.0]))[0]
J_reverse = torch.stack([J_row1, J_row2], dim=0)

print(f"Full Jacobian (reverse mode):\n{J_reverse}")
```

### Why Reverse Mode for Neural Networks?

In deep learning:
- **Inputs**: Millions of parameters ($n \gg 1$)
- **Outputs**: Single scalar loss ($m = 1$)

| Mode | Passes Required | Complexity |
|------|-----------------|------------|
| Forward | $n$ (once per parameter) | $O(n \cdot T)$ |
| **Reverse** | $m = 1$ (one backward pass) | $O(T)$ |

where $T$ is the cost of one forward pass.

**Conclusion:** Reverse mode is dramatically more efficient for training neural networks.

### Standard Backpropagation in PyTorch

```python
import torch
import torch.nn as nn

# Neural network with millions of parameters
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

# Forward pass
logits = model(x)
loss = nn.functional.cross_entropy(logits, y)

# Single backward pass computes gradients for ALL parameters!
loss.backward()

# Check: gradients exist for all parameters
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
```

## Computing Full Jacobians

### Using `torch.autograd.functional.jacobian`

```python
import torch
from torch.autograd.functional import jacobian

def f(x):
    """f: R^3 -> R^2"""
    return torch.stack([
        x[0] * x[1] + x[2],
        torch.sin(x[0]) + x[1]**2
    ])

x = torch.tensor([1.0, 2.0, 3.0])
J = jacobian(f, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {f(x).shape}")
print(f"Jacobian shape: {J.shape}")
print(f"Jacobian:\n{J}")
```

### Manual Full Jacobian via VJP (Reverse Mode)

```python
import torch

def compute_jacobian_reverse(f, x):
    """Compute full Jacobian using reverse mode (m backward passes)."""
    y = f(x)
    m = y.numel()
    n = x.numel()
    
    J = torch.zeros(m, n)
    for i in range(m):
        # Create one-hot adjoint
        v = torch.zeros(m)
        v[i] = 1.0
        
        # Recompute to get fresh graph
        x_new = x.detach().requires_grad_(True)
        y_new = f(x_new)
        y_new.backward(v)
        J[i] = x_new.grad
    
    return J

x = torch.tensor([1.0, 2.0, 3.0])
J = compute_jacobian_reverse(lambda x: torch.stack([x[0]*x[1], x[1]+x[2]]), x)
print(f"Jacobian:\n{J}")
```

## Comparison Summary

| Aspect | Forward Mode | Reverse Mode |
|--------|--------------|--------------|
| **Propagation** | Input → Output | Output → Input |
| **Computes** | JVP: $J \cdot \dot{x}$ | VJP: $J^T \cdot \bar{y}$ |
| **Efficient When** | $n \ll m$ | $m \ll n$ |
| **Deep Learning** | Rarely used | Standard (backprop) |
| **Full Jacobian** | $n$ passes | $m$ passes |
| **Memory** | $O(1)$ extra | $O(T)$ (stores activations) |

## Mixed Mode Considerations

### Memory-Compute Tradeoff

Reverse mode requires storing intermediate activations for the backward pass:

```python
import torch

x = torch.randn(1000, 1000, requires_grad=True)

# Forward pass - activations stored
y1 = x @ x.T
y2 = torch.relu(y1)
y3 = y2.sum()

# Backward pass - uses stored activations
y3.backward()

# Memory for activations: O(intermediate tensors)
```

### Gradient Checkpointing

For very deep networks, **gradient checkpointing** trades compute for memory:

```python
from torch.utils.checkpoint import checkpoint

def expensive_layer(x):
    return torch.relu(x @ x.T)

x = torch.randn(1000, 1000, requires_grad=True)

# Normal: stores intermediate activations
# y = expensive_layer(x)

# Checkpointing: recomputes during backward, saves memory
y = checkpoint(expensive_layer, x, use_reentrant=False)
```

## Practical Implications

### For Training

1. **Use reverse mode** (default in PyTorch) for standard training
2. **Scalar loss required** for efficient single backward pass
3. **Memory scales** with network depth and batch size

### For Analysis

1. **Full Jacobians** needed for: Fisher information, Hessian approximations, sensitivity analysis
2. **Choose mode** based on input/output dimensions
3. **torch.autograd.functional** provides both JVP and VJP utilities

## Summary

| Concept | Key Points |
|---------|------------|
| **Automatic Differentiation** | Systematic application of chain rule |
| **Forward Mode** | JVP, efficient for few inputs, $O(n)$ passes |
| **Reverse Mode** | VJP, efficient for few outputs, $O(m)$ passes |
| **Backpropagation** | Reverse mode with scalar loss ($m=1$) |
| **Why Reverse for DL** | $n$ (parameters) $\gg$ $m$ (loss = 1) |
| **Memory** | Reverse mode stores activations |

## References

- Baydin, A.G., et al. (2018). Automatic Differentiation in Machine Learning: A Survey. JMLR.
- Griewank, A. & Walther, A. (2008). Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.
- PyTorch Autograd Documentation: https://pytorch.org/docs/stable/autograd.html
