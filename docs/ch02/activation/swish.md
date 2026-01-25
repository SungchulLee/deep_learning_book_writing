# Swish / SiLU (Sigmoid Linear Unit)

## Overview

**Swish** (also known as **SiLU** - Sigmoid Linear Unit) is a self-gating activation function discovered through automated architecture search by Google Brain. It has become the preferred activation for efficient convolutional networks like EfficientNet and MobileNetV3.

## Learning Objectives

By the end of this section, you will understand:

1. Mathematical definition and the self-gating concept
2. Why Swish outperforms ReLU in many vision tasks
3. The relationship between Swish and GELU
4. Hardswish as an efficient approximation
5. Implementation patterns in PyTorch

---

## Mathematical Definition

### Swish / SiLU Formula

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\sigma(x)$ is the sigmoid function.

### Self-Gating Interpretation

Swish uses the input itself to gate the output:

$$\text{Swish}(x) = x \cdot \underbrace{\sigma(x)}_{\text{gate}}$$

- **Gate value** $\sigma(x)$: Ranges from 0 to 1
- **Gated value** $x$: The input itself
- **Result**: Input is scaled by a learned, input-dependent factor

This self-gating allows the function to adaptively filter information based on input magnitude.

### Derivative

$$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$

Or more elegantly:

$$\text{Swish}'(x) = \text{Swish}(x) + \sigma(x)(1 - \text{Swish}(x))$$

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-0.278, +\infty)$ approximately |
| **Minimum value** | $\approx -0.278$ at $x \approx -1.28$ |
| **Monotonic** | No |
| **Smooth** | Yes (infinitely differentiable) |
| **Bounded below** | Yes |
| **Bounded above** | No |

### Non-Monotonicity

Like GELU, Swish is non-monotonic—it dips below zero for negative inputs:

```python
import torch
import torch.nn.functional as F

# Find the minimum
x = torch.linspace(-3, 0, 1000, requires_grad=True)
y = F.silu(x)

# Minimum is around x ≈ -1.28
min_idx = y.argmin()
print(f"Minimum at x = {x[min_idx].item():.3f}")
print(f"Minimum value = {y[min_idx].item():.3f}")
# Minimum at x ≈ -1.28, value ≈ -0.278
```

### Asymptotic Behavior

- As $x \to +\infty$: $\text{Swish}(x) \to x$ (approaches identity)
- As $x \to -\infty$: $\text{Swish}(x) \to 0$ (approaches zero)

---

## PyTorch Implementation

### Naming Convention

PyTorch uses **SiLU** (Sigmoid Linear Unit) as the official name, while the research community often calls it **Swish**. They are identical functions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Functional API
y = F.silu(x)  # PyTorch name: SiLU

# Module API
silu = nn.SiLU()
y = silu(x)

# In-place version
silu_inplace = nn.SiLU(inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
# Output: ['-0.1423', '-0.2689', '0.0000', '0.7311', '2.8577']
```

### Manual Implementation

```python
# Swish is simply x * sigmoid(x)
def swish(x):
    return x * torch.sigmoid(x)

# Verify
x = torch.randn(100)
assert torch.allclose(swish(x), F.silu(x))
```

---

## Swish vs ReLU

### Visual Comparison

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Activation functions
ax1.plot(x.numpy(), F.silu(x).numpy(), label='Swish/SiLU', linewidth=2.5)
ax1.plot(x.numpy(), torch.relu(x).numpy(), label='ReLU', linewidth=2, linestyle='--')
ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
ax1.axvline(0, color='k', linestyle=':', alpha=0.3)
ax1.set_title('Swish vs ReLU')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Components of Swish
ax2.plot(x.numpy(), x.numpy(), label='x (identity)', linewidth=2, alpha=0.7)
ax2.plot(x.numpy(), torch.sigmoid(x).numpy(), label='σ(x) (gate)', linewidth=2, alpha=0.7)
ax2.plot(x.numpy(), F.silu(x).numpy(), label='x·σ(x) = Swish', linewidth=2.5)
ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
ax2.set_title('Swish Decomposition')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
```

### Key Differences

| Aspect | ReLU | Swish |
|--------|------|-------|
| **Formula** | $\max(0, x)$ | $x \cdot \sigma(x)$ |
| **Smoothness** | Non-smooth at 0 | Smooth everywhere |
| **Negative outputs** | Never | Yes, bounded |
| **Monotonic** | Yes | No |
| **Computational cost** | Very low | Higher (requires sigmoid) |

---

## Why Swish Works

### Discovery Through Neural Architecture Search

Swish was discovered by searching over a space of activation functions using reinforcement learning. The search optimized for:

- Accuracy on image classification
- Training stability
- Generalization

Swish emerged as the top performer, outperforming both ReLU and many hand-designed alternatives.

### Theoretical Advantages

1. **Smooth gradients**: Better optimization landscape
2. **Non-monotonicity**: Captures more complex patterns
3. **Self-gating**: Adaptive feature selection
4. **Bounded below**: Prevents extreme negative activations

### Empirical Advantages

In the original paper, Swish outperformed ReLU on:

- ImageNet classification
- CIFAR-10/100
- Machine translation
- Various other benchmarks

Improvements typically range from 0.3% to 1.5% on ImageNet.

---

## Swish vs GELU

### Comparison

Both Swish and GELU are smooth, non-monotonic activations discovered around the same time:

| Aspect | Swish | GELU |
|--------|-------|------|
| **Formula** | $x \cdot \sigma(x)$ | $x \cdot \Phi(x)$ |
| **Gate function** | Sigmoid | Gaussian CDF |
| **Minimum** | ≈ -0.278 | ≈ -0.170 |
| **Primary use** | CNNs, mobile nets | Transformers |
| **Computation** | 1 sigmoid, 1 mult | Error function |

### Visual Comparison

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), F.silu(x).numpy(), label='Swish', linewidth=2.5)
plt.plot(x.numpy(), F.gelu(x).numpy(), label='GELU', linewidth=2.5)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='ReLU', linewidth=2, linestyle='--', alpha=0.5)
plt.axhline(0, color='k', linestyle=':', alpha=0.3)
plt.title('Swish vs GELU vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)
```

### When to Use Each

- **Swish**: CNNs, EfficientNet-style networks, mobile models
- **GELU**: Transformers, BERT, GPT, attention-based models

---

## Hardswish: Efficient Approximation

### Definition

Hardswish is a piecewise linear approximation of Swish, designed for mobile and edge deployment:

$$\text{Hardswish}(x) = \begin{cases}
0 & \text{if } x \leq -3 \\
x & \text{if } x \geq 3 \\
x \cdot \frac{x + 3}{6} & \text{otherwise}
\end{cases}$$

Or equivalently:

$$\text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-4.0, -3.0, 0.0, 3.0, 4.0])

# Functional API
y_hard = F.hardswish(x)

# Module API
hardswish = nn.Hardswish()
y_hard = hardswish(x)

# Compare to Swish
y_swish = F.silu(x)

print(f"Input:     {x.tolist()}")
print(f"Hardswish: {y_hard.tolist()}")
print(f"Swish:     {[f'{v:.4f}' for v in y_swish.tolist()]}")
```

### Comparison

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), F.silu(x).numpy(), label='Swish (original)', linewidth=3, alpha=0.5)
plt.plot(x.numpy(), F.hardswish(x).numpy(), label='Hardswish (efficient)', linewidth=2, linestyle='--')
plt.axhline(0, color='k', linestyle=':', alpha=0.3)
plt.title('Swish vs Hardswish')
plt.legend()
plt.grid(True, alpha=0.3)
```

### Why Hardswish?

| Aspect | Swish | Hardswish |
|--------|-------|-----------|
| **Operations** | Sigmoid, multiply | ReLU6, add, multiply, divide |
| **Sigmoid required** | Yes | No |
| **Hardware friendly** | Less | More |
| **Accuracy** | Slightly better | Nearly identical |
| **Speed (mobile)** | Slower | Faster |

---

## Usage in Architectures

### EfficientNet

```python
import torch.nn as nn

class EfficientNetBlock(nn.Module):
    """Simplified EfficientNet MBConv block with Swish."""
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        self.block = nn.Sequential(
            # Expansion
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),  # Swish!
            
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),  # Swish!
            
            # Squeeze-and-Excitation
            # ... (omitted for brevity)
            
            # Projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.block(x)
```

### MobileNetV3

```python
class MobileNetV3Block(nn.Module):
    """MobileNetV3 block with Hardswish."""
    def __init__(self, in_channels, out_channels, kernel_size, 
                 expand_ratio, use_se=True, use_hs=True):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        # Use Hardswish for efficiency
        activation = nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True)
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation,
            ])
        
        layers.extend([
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation,
        ])
        
        # ... SE and projection
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
```

---

## Parameterized Swish (Beta Swish)

### Definition

A parameterized version of Swish with a learnable or fixed scaling factor:

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

where $\beta$ controls the shape:

- $\beta = 0$: Linear function (scaled by 0.5)
- $\beta = 1$: Standard Swish
- $\beta \to \infty$: Approaches ReLU

### Implementation

```python
import torch
import torch.nn as nn

class ParametricSwish(nn.Module):
    """Swish with learnable beta parameter."""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Usage
param_swish = ParametricSwish()
x = torch.randn(32, 64)
y = param_swish(x)

# Beta is learned during training
print(f"Initial beta: {param_swish.beta.item():.4f}")
```

---

## Mish: A Related Activation

### Definition

Mish is similar to Swish but uses softplus and tanh:

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

### PyTorch Implementation

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

y_mish = F.mish(x)
y_swish = F.silu(x)

print(f"Input:  {x.tolist()}")
print(f"Mish:   {[f'{v:.4f}' for v in y_mish.tolist()]}")
print(f"Swish:  {[f'{v:.4f}' for v in y_swish.tolist()]}")
```

### Comparison

| Aspect | Swish | Mish |
|--------|-------|------|
| **Formula** | $x \cdot \sigma(x)$ | $x \cdot \tanh(\text{softplus}(x))$ |
| **Smoothness** | Smooth | Smoother |
| **Computation** | Lighter | Heavier |
| **Used in** | EfficientNet | YOLOv4 |

---

## Summary

| Activation | Formula | Best For |
|------------|---------|----------|
| **Swish/SiLU** | $x \cdot \sigma(x)$ | CNNs, EfficientNet |
| **Hardswish** | $x \cdot \text{ReLU6}(x+3)/6$ | Mobile, edge deployment |
| **Mish** | $x \cdot \tanh(\text{softplus}(x))$ | YOLOv4, experiments |

!!! tip "Practical Recommendations"
    - **EfficientNet-style networks**: Use Swish/SiLU
    - **Mobile deployment**: Use Hardswish
    - **General CNNs**: Try Swish; fall back to ReLU if no improvement
    - **Transformers**: Use GELU instead

---

## References

1. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for Activation Functions". arXiv:1710.05941
2. Howard, A., et al. (2019). "Searching for MobileNetV3". ICCV 2019
3. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for CNNs". ICML 2019
4. Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Activation Function". arXiv:1908.08681
