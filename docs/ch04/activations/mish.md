# Mish Activation Function

## Overview

**Mish** is a smooth, non-monotonic activation function proposed by Misra (2019). It is closely related to Swish but uses a different self-gating mechanism based on softplus and tanh. Mish gained prominence through its adoption in **YOLOv4** and has been shown to provide small but consistent improvements over ReLU and Swish in certain object detection tasks.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and properties of Mish
2. How Mish relates to Swish and GELU
3. Computational considerations
4. When Mish provides benefits over alternatives
5. PyTorch implementation

---

## Mathematical Definition

$$\operatorname{Mish}(x) = x \cdot \tanh\!\bigl(\operatorname{softplus}(x)\bigr) = x \cdot \tanh\!\bigl(\ln(1 + e^x)\bigr)$$

### Decomposition

Like Swish, Mish has a self-gating structure:

$$\operatorname{Mish}(x) = x \cdot \underbrace{\tanh(\operatorname{softplus}(x))}_{\text{gate}}$$

The gate function $\tanh(\operatorname{softplus}(x))$ is a smooth, bounded function that:

- Approaches 1 for large positive $x$ (since $\operatorname{softplus}(x) \to x$ and $\tanh(x) \to 1$)
- Approaches 0 for large negative $x$ (since $\operatorname{softplus}(x) \to 0$ and $\tanh(0) = 0$)
- Provides a smooth transition near zero

### Derivative

$$\operatorname{Mish}'(x) = \tanh(\operatorname{softplus}(x)) + x \cdot \operatorname{sech}^2(\operatorname{softplus}(x)) \cdot \sigma(x)$$

where $\sigma(x)$ is the sigmoid function (the derivative of softplus).

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $\approx (-0.309, +\infty)$ |
| **Minimum value** | $\approx -0.309$ at $x \approx -1.19$ |
| **Monotonic** | No |
| **Smooth** | Yes (infinitely differentiable) |
| **Bounded below** | Yes |
| **Bounded above** | No |

### Comparison with Swish and GELU

| Property | Mish | Swish | GELU |
|----------|------|-------|------|
| **Gate function** | $\tanh(\operatorname{softplus}(x))$ | $\sigma(x)$ | $\Phi(x)$ |
| **Minimum** | $\approx -0.309$ | $\approx -0.278$ | $\approx -0.170$ |
| **Smoothness** | Smoother | Smooth | Smooth |
| **Computation** | Heaviest | Medium | Medium |
| **Used in** | YOLOv4 | EfficientNet | Transformers |

All three are smooth, non-monotonic, and self-gating. The practical differences are small; the choice is often dictated by the architecture lineage.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Functional API
y_mish = F.mish(x)

# Module API
mish = nn.Mish()
y_mish = mish(x)

# In-place version
mish_inplace = nn.Mish(inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Mish:   {[f'{v:.4f}' for v in y_mish.tolist()]}")
# Output: ['-0.2525', '-0.3034', '0.0000', '0.8651', '1.9440']
```

### Comparison with Swish

```python
y_swish = F.silu(x)

print(f"Input:  {x.tolist()}")
print(f"Mish:   {[f'{v:.4f}' for v in y_mish.tolist()]}")
print(f"Swish:  {[f'{v:.4f}' for v in y_swish.tolist()]}")
# The two are very similar, with Mish having slightly deeper negative values
```

---

## Visualization

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), F.mish(x).numpy(), label='Mish', linewidth=2.5)
plt.plot(x.numpy(), F.silu(x).numpy(), label='Swish/SiLU', linewidth=2, alpha=0.7)
plt.plot(x.numpy(), F.gelu(x).numpy(), label='GELU', linewidth=2, alpha=0.7)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='ReLU', linewidth=1.5, 
         linestyle='--', alpha=0.5)
plt.axhline(0, color='k', linestyle=':', alpha=0.3)
plt.title('Mish vs Swish vs GELU vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

---

## Manual Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MishManual(nn.Module):
    """Manual Mish implementation for understanding."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Verify against PyTorch
x = torch.randn(100)
manual = MishManual()
assert torch.allclose(manual(x), F.mish(x), atol=1e-6)
```

---

## Usage in YOLOv4

Mish was adopted as the activation function in the **CSPDarknet53** backbone of YOLOv4, replacing Leaky ReLU used in earlier YOLO versions:

```python
import torch.nn as nn

class CSPDarknetBlock(nn.Module):
    """Simplified CSPDarknet block with Mish (YOLOv4 style)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
```

---

## Computational Considerations

Mish is the most computationally expensive among the smooth activations:

| Activation | Operations | Relative Cost |
|------------|-----------|---------------|
| ReLU | 1 comparison | 1.0× |
| Swish | 1 exp + 1 div + 1 mul | ~2.0× |
| GELU (approx) | 1 tanh + arithmetic | ~1.5× |
| Mish | 1 exp + 1 log + 1 tanh + 1 mul | ~2.5× |

In practice, the activation function is rarely the bottleneck — convolutions and matrix multiplications dominate. The difference is most noticeable on edge devices with limited compute.

---

## When to Use Mish

### Recommended

- **Object detection** (YOLO family): Established track record in YOLOv4
- **Experimentation**: When comparing smooth activations, Mish is worth trying

### Usually Not Necessary

- **Transformers**: GELU is the established standard
- **EfficientNet-style CNNs**: Swish is the native choice
- **Mobile/edge**: Hardswish or ReLU for efficiency
- **General CNNs**: The improvement over Swish is often marginal

---

## Summary

| Aspect | Mish |
|--------|------|
| **Formula** | $x \cdot \tanh(\ln(1 + e^x))$ |
| **Range** | $\approx (-0.309, +\infty)$ |
| **Smoothness** | ✅ Very smooth |
| **Non-monotonic** | ✅ Yes |
| **Computational cost** | ⚠️ Highest among smooth activations |
| **Best for** | YOLOv4, object detection experiments |
| **vs Swish** | Slightly smoother, slightly more expensive |
| **vs GELU** | Similar properties, different domain |

!!! tip "Practical Recommendation"
    Mish provides marginal improvements over Swish in some settings, but the differences are small. Use Mish when following the YOLOv4 architecture or when experimenting with activation functions. For most new projects, Swish (for CNNs) or GELU (for transformers) are the safer defaults.

---

## References

1. Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Activation Function". arXiv:1908.08681
2. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection". arXiv:2004.10934
