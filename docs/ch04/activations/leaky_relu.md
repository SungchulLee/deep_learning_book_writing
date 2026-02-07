# Leaky ReLU

## Overview

**Leaky ReLU** is a variant of ReLU that addresses the dead neuron problem by allowing a small, non-zero gradient for negative inputs. Instead of mapping all negative values to zero, Leaky ReLU multiplies them by a small positive slope $\alpha$, ensuring that gradients always flow and neurons never permanently die.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and derivative of Leaky ReLU
2. How it solves the dead neuron problem
3. How to choose the negative slope hyperparameter $\alpha$
4. When to prefer Leaky ReLU over standard ReLU
5. PyTorch implementation patterns

---

## Mathematical Definition

$$\operatorname{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases} = \max(\alpha x,\; x)$$

where $\alpha$ is a small positive constant, typically 0.01 or 0.1.

### Derivative

$$\operatorname{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

The key insight: the gradient is **never zero**, always at least $\alpha > 0$. This guarantees that every neuron receives a gradient signal during backpropagation, regardless of the sign of its pre-activation.

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-\infty, +\infty)$ |
| **Non-saturating** | Yes (everywhere) |
| **Dead neurons** | No (gradient $\geq \alpha > 0$ always) |
| **Zero-centered** | Approximately (for symmetric input distributions) |
| **Smooth** | No (non-differentiable at $x = 0$) |
| **Computational cost** | Very low (same as ReLU) |
| **Negative slope** | Fixed hyperparameter $\alpha$ |

---

## Why Leaky ReLU Works

### Solving the Dead Neuron Problem

With standard ReLU, if a neuron's pre-activation is negative for all inputs, the gradient is zero and the neuron can never recover. Leaky ReLU fixes this:

- ReLU gradient for $x < 0$: **0** → neuron is dead, no recovery possible
- Leaky ReLU gradient for $x < 0$: **$\alpha > 0$** → gradient flows, weights can update

Even a small $\alpha = 0.01$ is sufficient to keep neurons alive and allow recovery from unfavorable weight configurations.

### Gradient Flow Comparison

For a 10-layer network where all pre-activations happen to be negative:

| Activation | Gradient per layer | 10-layer product |
|-----------|-------------------|-----------------|
| ReLU | 0 | 0 (completely dead) |
| Leaky ReLU ($\alpha = 0.01$) | 0.01 | $10^{-20}$ (small but nonzero) |
| Leaky ReLU ($\alpha = 0.1$) | 0.1 | $10^{-10}$ (recoverable) |

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Functional API
y = F.leaky_relu(x, negative_slope=0.1)

# Module API
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
y = leaky_relu(x)

# In-place version
leaky_relu_inplace = nn.LeakyReLU(negative_slope=0.1, inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {y.tolist()}")
# Output: [-0.2, -0.1, 0.0, 1.0, 2.0]
```

---

## Choosing the Negative Slope

The negative slope $\alpha$ controls the trade-off between sparsity and gradient flow:

| $\alpha$ Value | Effect | Use Case |
|----------------|--------|----------|
| `0.01` | Minimal negative flow, near-ReLU behavior | Default, general use |
| `0.1` | Moderate negative flow | Often better empirically |
| `0.2` | Standard for GANs | DCGAN, WGAN discriminators |
| `0.3+` | Aggressive, approaches linearity | Rarely used |

!!! tip "Practical Guidance"
    $\alpha = 0.01$ is the PyTorch default. In practice, $\alpha = 0.1$ often works better. For GANs, $\alpha = 0.2$ is the established convention. If you want the network to learn the optimal slope, use [PReLU](prelu.md).

---

## Network Example: GAN Discriminator

Leaky ReLU is the standard activation for GAN discriminators, where it prevents dead neurons while maintaining useful gradient flow:

```python
import torch.nn as nn

class Discriminator(nn.Module):
    """DCGAN-style discriminator with Leaky ReLU."""
    def __init__(self, img_channels=3, features_d=64):
        super().__init__()
        self.model = nn.Sequential(
            # Input: [batch, 3, 64, 64]
            nn.Conv2d(img_channels, features_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 4, 1, 4, 1, 0),
            # No activation — use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.model(x).view(-1)
```

---

## Network Example: CNN with Leaky ReLU

```python
import torch.nn as nn

class ConvNetLeakyReLU(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## Weight Initialization

Like ReLU, Leaky ReLU networks should use He (Kaiming) initialization, with the nonlinearity type specified:

```python
import torch.nn as nn

def init_leaky_relu_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(
            module.weight, 
            a=0.1,  # Negative slope for Leaky ReLU
            mode='fan_in', 
            nonlinearity='leaky_relu'
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_leaky_relu_weights)
```

The `a` parameter in `kaiming_normal_` adjusts the variance calculation to account for the non-zero negative slope.

---

## Summary

| Aspect | Leaky ReLU |
|--------|------------|
| **Formula** | $\max(\alpha x, x)$ |
| **Range** | $(-\infty, +\infty)$ |
| **Dead neurons** | ✅ None |
| **Smooth** | ❌ No (non-differentiable at $x=0$) |
| **Computational cost** | ✅ Very low |
| **Best for** | GANs, CNNs with dead neuron issues |
| **Hyperparameter** | Negative slope $\alpha$ (typically 0.01–0.2) |
| **Initialization** | He (Kaiming) with `nonlinearity='leaky_relu'` |

!!! tip "Practical Recommendation"
    Leaky ReLU is a safe drop-in replacement for ReLU that eliminates the dead neuron problem with negligible computational overhead. Use it when you observe dead neurons with standard ReLU, or as a default for GAN discriminators. If you want the network to learn the optimal negative slope, use [PReLU](prelu.md) instead.
