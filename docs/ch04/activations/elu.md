# ELU (Exponential Linear Unit)

## Overview

**ELU (Exponential Linear Unit)**, proposed by Clevert et al. (2016), combines the non-saturating property of ReLU for positive inputs with a smooth, saturating exponential curve for negative inputs. This design pushes mean activations closer to zero (improving gradient flow), provides a smooth function everywhere (better optimization landscape), and offers noise robustness through soft saturation.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and derivative of ELU
2. Why smooth saturation for negative inputs is beneficial
3. Advantages of ELU over ReLU and Leaky ReLU
4. The computational cost trade-off
5. PyTorch implementation patterns

---

## Mathematical Definition

$$\operatorname{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

where $\alpha > 0$ (default: 1.0) controls the saturation value for negative inputs.

### Derivative

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x \leq 0 \end{cases}$$

Note that for $x \leq 0$, the derivative can also be written as $\operatorname{ELU}(x) + \alpha$, allowing efficient computation from the forward-pass output:

$$\operatorname{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \operatorname{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

### Behavior

- **Positive regime** ($x > 0$): Identity function, gradient = 1 (same as ReLU)
- **Negative regime** ($x \leq 0$): Smooth exponential curve saturating at $-\alpha$
- **At origin** ($x = 0$): Continuous with continuous first derivative — the function is $C^1$ smooth

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-\alpha, +\infty)$ |
| **Smooth** | Yes (continuous first derivative everywhere) |
| **Near zero-centered** | Yes (negative mean pushes output toward zero) |
| **Soft saturation** | At $-\alpha$ for $x \to -\infty$ |
| **Dead neurons** | No (gradient $> 0$ for all $x$) |
| **Monotonic** | Yes |
| **Computational cost** | Medium (requires $\exp$ for negative inputs) |

---

## Advantages over ReLU

### 1. Smoothness at the Origin

ReLU has a sharp corner at $x = 0$ with an undefined derivative. ELU transitions smoothly between the positive and negative regimes, providing a better optimization landscape:

$$\lim_{x \to 0^-} \operatorname{ELU}'(x) = \alpha \cdot e^0 = \alpha$$

For $\alpha = 1$, this matches the gradient from the positive side (which is 1), giving a continuous first derivative.

### 2. Negative Values Push Mean Toward Zero

Unlike ReLU (always $\geq 0$) or Leaky ReLU (with small negative values), ELU produces substantial negative outputs that shift the mean activation closer to zero. This reduces the bias shift effect and can speed convergence.

### 3. Noise Robustness Through Saturation

For very negative inputs, ELU saturates at $-\alpha$ instead of growing linearly (as Leaky ReLU does). This soft saturation makes the network more robust to noise and outliers in the negative regime.

---

## Disadvantages

- **Computational cost**: The exponential function is more expensive than the simple comparison in ReLU/Leaky ReLU
- **Not exactly zero-centered**: While closer to zero-centered than ReLU, the output distribution is not perfectly symmetric

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Functional API
y = F.elu(x, alpha=1.0)

# Module API
elu = nn.ELU(alpha=1.0)
y = elu(x)

# In-place version
elu_inplace = nn.ELU(alpha=1.0, inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
# Output: ['-0.8647', '-0.6321', '0.0000', '1.0000', '2.0000']
```

### Choosing $\alpha$

The parameter $\alpha$ controls the saturation floor:

| $\alpha$ | Saturation at | Effect |
|----------|---------------|--------|
| 0.5 | $-0.5$ | Less negative range |
| 1.0 (default) | $-1.0$ | Standard, balanced |
| 2.0 | $-2.0$ | More negative range |

In practice, $\alpha = 1.0$ works well for most applications.

---

## Visualization

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ELU with different alpha values
for alpha in [0.5, 1.0, 2.0]:
    y = torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    axes[0].plot(x.numpy(), y.numpy(), 
                 label=f'ELU (α={alpha})', linewidth=2)

axes[0].plot(x.numpy(), torch.relu(x).numpy(), 
             '--', label='ReLU', linewidth=1.5, alpha=0.5)
axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)
axes[0].set_title('ELU with Different α Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ELU derivative
elu_grad = torch.where(x > 0, torch.ones_like(x), torch.exp(x))
relu_grad = (x > 0).float()

axes[1].plot(x.numpy(), elu_grad.numpy(), label="ELU' (α=1)", linewidth=2)
axes[1].plot(x.numpy(), relu_grad.numpy(), '--', 
             label="ReLU'", linewidth=1.5, alpha=0.5)
axes[1].set_title('Derivatives')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## Network Example

```python
import torch.nn as nn

class ConvNetELU(nn.Module):
    """CNN using ELU for smoother training dynamics."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(alpha=1.0, inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

---

## Comparison with Relatives

| Aspect | ReLU | Leaky ReLU | ELU |
|--------|------|-----------|-----|
| **Negative behavior** | Zero | Linear ($\alpha x$) | Exponential saturation |
| **Smooth at origin** | ❌ | ❌ | ✅ |
| **Near zero-centered** | ❌ | ≈ | ✅ |
| **Dead neurons** | Yes | No | No |
| **Noise robustness** | Low | Low | High |
| **Compute cost** | Very low | Very low | Medium |

---

## Summary

| Aspect | ELU |
|--------|-----|
| **Formula** | $\max(x, 0) + \min(0, \alpha(e^x - 1))$ |
| **Range** | $(-\alpha, +\infty)$ |
| **Dead neurons** | ✅ None |
| **Smooth** | ✅ Yes ($C^1$ continuous) |
| **Zero-centered** | ✅ Approximately |
| **Computational cost** | ⚠️ Medium (exponential) |
| **Best for** | When smooth gradients and noise robustness are important |
| **Initialization** | He (Kaiming) Normal |

!!! tip "Practical Recommendation"
    ELU is a good choice when training is unstable with ReLU or when you need smoother gradient flow without switching to more complex activations like GELU. The computational overhead of the exponential is usually negligible compared to the rest of the network. For self-normalizing behavior without BatchNorm, consider [SELU](selu.md) instead.
