# Swish / SiLU (Sigmoid Linear Unit)

## Overview

**Swish** (also known as **SiLU** — Sigmoid Linear Unit) is a self-gating activation function discovered through automated architecture search by Google Brain (Ramachandran et al., 2017). It has become the preferred activation for efficient convolutional networks like EfficientNet and MobileNetV3, and via its gated variant **SwiGLU**, it powers the FFN blocks of modern LLMs including LLaMA and PaLM.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and the self-gating concept
2. Why Swish outperforms ReLU in many vision tasks
3. The relationship between Swish and GELU
4. Hardswish as an efficient approximation for mobile deployment
5. SwiGLU and gated FFN blocks in modern LLMs
6. Parameterized (Beta) Swish

---

## Mathematical Definition

### Swish / SiLU Formula

$$\operatorname{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\sigma(x)$ is the sigmoid function.

### Self-Gating Interpretation

Swish uses the input itself to gate the output:

$$\operatorname{Swish}(x) = x \cdot \underbrace{\sigma(x)}_{\text{gate}}$$

- **Gate value** $\sigma(x) \in (0, 1)$: Controls how much of the input passes through
- **Gated value** $x$: The input itself
- **Result**: An input-dependent, learned scaling

For large positive $x$, $\sigma(x) \approx 1$, so $\operatorname{Swish}(x) \approx x$ (identity). For large negative $x$, $\sigma(x) \approx 0$, so $\operatorname{Swish}(x) \approx 0$ (suppressed). Near zero, the gate is approximately 0.5, creating a smooth transition.

### Derivative

$$\operatorname{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)\bigl(1 + x(1 - \sigma(x))\bigr)$$

Or equivalently in terms of Swish itself:

$$\operatorname{Swish}'(x) = \operatorname{Swish}(x) + \sigma(x)\bigl(1 - \operatorname{Swish}(x)\bigr)$$

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $\approx (-0.278, +\infty)$ |
| **Minimum value** | $\approx -0.278$ at $x \approx -1.28$ |
| **Monotonic** | No |
| **Smooth** | Yes (infinitely differentiable) |
| **Bounded below** | Yes |
| **Bounded above** | No |

### Asymptotic Behavior

- As $x \to +\infty$: $\operatorname{Swish}(x) \to x$ (approaches identity)
- As $x \to -\infty$: $\operatorname{Swish}(x) \to 0$ (approaches zero)

---

## PyTorch Implementation

PyTorch uses **SiLU** (Sigmoid Linear Unit) as the official name. Swish and SiLU are identical.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Functional API
y = F.silu(x)

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
def swish(x):
    return x * torch.sigmoid(x)

# Verify equivalence
x = torch.randn(100)
assert torch.allclose(swish(x), F.silu(x))
```

---

## Why Swish Works

### Discovery Through Neural Architecture Search

Swish was discovered by searching over a space of activation functions using reinforcement learning, optimizing for accuracy on image classification, training stability, and generalization. Swish emerged as the top performer, outperforming both ReLU and many hand-designed alternatives.

### Theoretical Advantages

1. **Smooth gradients**: Better optimization landscape than ReLU's sharp corner
2. **Non-monotonicity**: Captures more complex patterns via the negative dip
3. **Self-gating**: Adaptive feature selection without additional parameters
4. **Bounded below**: Prevents extreme negative activations

### Empirical Results

In the original paper, Swish outperformed ReLU on ImageNet, CIFAR-10/100, and machine translation benchmarks. Improvements typically range from 0.3% to 1.5% on ImageNet.

---

## Swish vs ReLU

| Aspect | ReLU | Swish |
|--------|------|-------|
| **Formula** | $\max(0, x)$ | $x \cdot \sigma(x)$ |
| **Smoothness** | Non-smooth at 0 | Smooth everywhere |
| **Negative outputs** | Never | Yes, bounded |
| **Monotonic** | Yes | No |
| **Computational cost** | Very low | Higher (requires sigmoid) |

---

## Swish vs GELU

Both are smooth, non-monotonic activations discovered around the same time:

| Aspect | Swish | GELU |
|--------|-------|------|
| **Formula** | $x \cdot \sigma(x)$ | $x \cdot \Phi(x)$ |
| **Gate function** | Sigmoid | Gaussian CDF |
| **Minimum** | $\approx -0.278$ | $\approx -0.170$ |
| **Primary use** | CNNs, efficient networks | Transformers |
| **Computation** | 1 sigmoid, 1 multiply | Error function |

The two are closely related: GELU's sigmoid approximation is $x \cdot \sigma(1.702\,x)$, while Swish is $x \cdot \sigma(x)$. They differ only in the scaling of the sigmoid argument.

**When to use each:**

- **Swish**: CNNs, EfficientNet-style networks, mobile models
- **GELU**: Transformers, BERT, GPT, attention-based models

---

## Hardswish: Efficient Approximation

### Definition

Hardswish is a piecewise linear approximation of Swish, designed for mobile and edge deployment where sigmoid computation is expensive:

$$\operatorname{Hardswish}(x) = \begin{cases}
0 & \text{if } x \leq -3 \\
x & \text{if } x \geq 3 \\
x \cdot \frac{x + 3}{6} & \text{otherwise}
\end{cases}$$

Or equivalently:

$$\operatorname{Hardswish}(x) = x \cdot \frac{\operatorname{ReLU6}(x + 3)}{6}$$

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

# Compare with Swish
y_swish = F.silu(x)

print(f"Input:     {x.tolist()}")
print(f"Hardswish: {y_hard.tolist()}")
print(f"Swish:     {[f'{v:.4f}' for v in y_swish.tolist()]}")
```

### Comparison

| Aspect | Swish | Hardswish |
|--------|-------|-----------|
| **Operations** | Sigmoid + multiply | ReLU6 + add + multiply + divide |
| **Sigmoid required** | Yes | No |
| **Hardware friendly** | Less | More (no transcendentals) |
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
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),  # Swish
            
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),  # Swish
            
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.block(x)
```

### MobileNetV3

```python
class MobileNetV3Block(nn.Module):
    """MobileNetV3 uses Hardswish for efficiency."""
    def __init__(self, in_channels, out_channels, expand_ratio, use_hs=True):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        activation = nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.block(x)
```

---

## SwiGLU: Swish-Gated FFN for LLMs

SwiGLU combines Swish with a **gated linear unit** mechanism, using two separate linear projections where one is gated by Swish:

$$\operatorname{SwiGLU}(\mathbf{x}) = \operatorname{Swish}(\mathbf{x}\mathbf{W}_1) \otimes (\mathbf{x}\mathbf{W}_3)$$

This has become the standard FFN block in modern LLMs (LLaMA, PaLM, Falcon):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """SwiGLU-based FFN (LLaMA style)."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Reduce hidden dim by 2/3 to match standard FFN parameter count
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Align for efficiency
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

!!! note "Parameter Count"
    SwiGLU uses 3 weight matrices instead of 2, increasing parameters by 50%. To match the parameter count of a standard FFN, the hidden dimension is typically reduced by a factor of $\frac{2}{3}$.

---

## Parameterized Swish (Beta Swish)

A parameterized version with a learnable or fixed scaling factor:

$$\operatorname{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

The parameter $\beta$ controls the activation shape:

- $\beta = 0$: Linear function (scaled by 0.5)
- $\beta = 1$: Standard Swish
- $\beta \to \infty$: Approaches ReLU

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

param_swish = ParametricSwish()
x = torch.randn(32, 64)
y = param_swish(x)
print(f"Initial beta: {param_swish.beta.item():.4f}")
```

---

## Summary

| Activation | Formula | Best For |
|------------|---------|----------|
| **Swish/SiLU** | $x \cdot \sigma(x)$ | CNNs, EfficientNet |
| **Hardswish** | $x \cdot \operatorname{ReLU6}(x+3)/6$ | Mobile, edge deployment |
| **SwiGLU** | $\operatorname{Swish}(\mathbf{x}\mathbf{W}_1) \otimes \mathbf{x}\mathbf{W}_3$ | Modern LLMs (LLaMA, PaLM) |

!!! tip "Practical Recommendations"
    - **EfficientNet-style networks**: Use Swish/SiLU
    - **Mobile deployment**: Use Hardswish
    - **General CNNs**: Try Swish; fall back to ReLU if no improvement
    - **Modern LLMs**: Use SwiGLU in FFN blocks
    - **Standard transformers**: Use [GELU](gelu.md) instead

---

## References

1. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for Activation Functions". arXiv:1710.05941
2. Howard, A., et al. (2019). "Searching for MobileNetV3". ICCV 2019
3. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for CNNs". ICML 2019
4. Shazeer, N. (2020). "GLU Variants Improve Transformer"
5. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
