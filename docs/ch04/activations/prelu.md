# PReLU (Parametric ReLU)

## Overview

**PReLU (Parametric Rectified Linear Unit)** extends Leaky ReLU by making the negative slope a **learnable parameter** that is optimized during training via backpropagation. This allows the network to discover the optimal negative slope for each channel, often outperforming both ReLU and Leaky ReLU, particularly in image classification tasks.

## Learning Objectives

By the end of this section, you will understand:

1. How PReLU differs from Leaky ReLU
2. Per-channel vs shared parameter modes
3. Training dynamics and learned slope analysis
4. PyTorch implementation patterns
5. When to use PReLU vs fixed-slope alternatives

---

## Mathematical Definition

$$\operatorname{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ a \cdot x & \text{if } x \leq 0 \end{cases}$$

where $a$ is a **learnable parameter** optimized via gradient descent.

### Derivative

$$\frac{\partial \operatorname{PReLU}(x)}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ a & \text{if } x \leq 0 \end{cases}$$

The gradient with respect to the learnable parameter $a$:

$$\frac{\partial \operatorname{PReLU}(x)}{\partial a} = \begin{cases} 0 & \text{if } x > 0 \\ x & \text{if } x \leq 0 \end{cases}$$

This means $a$ is updated only by negative pre-activations, learning the optimal slope for the negative regime.

---

## Key Difference from Leaky ReLU

| Aspect | Leaky ReLU | PReLU |
|--------|------------|-------|
| **Negative slope** | Fixed hyperparameter $\alpha$ | Learned parameter $a$ |
| **Tuning** | Manual (grid search) | Automatic (backpropagation) |
| **Additional parameters** | 0 | 1 per channel (or 1 shared) |
| **Adaptability** | Same slope everywhere | Different slopes per channel |

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-\infty, +\infty)$ |
| **Dead neurons** | No (gradient $\geq a > 0$) |
| **Learnable** | Yes (negative slope $a$) |
| **Default initialization** | $a = 0.25$ |
| **Additional parameters** | Minimal (1 per channel) |

---

## PyTorch Implementation

### Basic Usage

```python
import torch
import torch.nn as nn

# Single learnable parameter shared across all channels
prelu_shared = nn.PReLU(num_parameters=1)
print(f"Initial a: {prelu_shared.weight.item():.4f}")  # Default: 0.25

# One learnable parameter per channel (recommended for conv layers)
prelu_per_channel = nn.PReLU(num_parameters=64)
print(f"Parameters shape: {prelu_per_channel.weight.shape}")  # [64]

# Custom initialization
prelu_custom = nn.PReLU(num_parameters=64, init=0.1)
```

### Per-Channel vs Shared Mode

**Per-channel** (recommended): Each channel learns its own slope, allowing different features to have different negative behaviors.

**Shared**: A single slope parameter for all channels, reducing parameters but limiting expressiveness.

```python
# Per-channel: one parameter per output feature
prelu = nn.PReLU(num_parameters=128)  # For 128-channel layer

# Shared: single parameter for all features
prelu = nn.PReLU(num_parameters=1)
```

---

## Network Example

```python
import torch
import torch.nn as nn

class PReLUNetwork(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.prelu1 = nn.PReLU(num_parameters=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU(num_parameters=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        return self.fc3(x)

model = PReLUNetwork()

# PReLU parameters are included in model.parameters()
optimizer = torch.optim.Adam(model.parameters())  # Includes PReLU weights
```

### Inspecting Learned Slopes

After training, inspecting the learned slopes provides insight into what the network discovered:

```python
def print_prelu_stats(model):
    """Display learned negative slopes."""
    for name, module in model.named_modules():
        if isinstance(module, nn.PReLU):
            w = module.weight.data
            print(f"{name}: mean={w.mean():.4f}, std={w.std():.4f}, "
                  f"min={w.min():.4f}, max={w.max():.4f}")

# After training
print_prelu_stats(model)
# Typical output:
# prelu1: mean=0.12, std=0.08, min=-0.02, max=0.35
# prelu2: mean=0.18, std=0.06, min=0.05, max=0.31
```

Common observations after training:

- Learned slopes often differ from the default 0.25
- Different channels learn different slopes
- Some slopes may approach 0 (ReLU-like) while others approach larger values
- Negative slopes are rare but possible (indicating the neuron prefers to flip sign)

---

## CNN with PReLU

```python
import torch.nn as nn

class ConvNetPReLU(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.PReLU(num_parameters=64),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.PReLU(num_parameters=128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.PReLU(num_parameters=512),
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

## When to Use PReLU

### Recommended

- **Image classification**: PReLU was introduced in the context of ImageNet and consistently helps
- **When Leaky ReLU helps but optimal slope is unknown**: Let the network learn it
- **Sufficient training data**: PReLU adds parameters, so it benefits from larger datasets

### Caution

- **Very small datasets**: The extra parameters (one per channel) could contribute to overfitting, though the effect is usually negligible
- **When reproducibility with fixed architectures matters**: Learned slopes add training-dependent variation

### Parameter Count Perspective

The additional parameters from PReLU are minimal:

| Layer | Weights | PReLU Params | Overhead |
|-------|---------|-------------|----------|
| `Linear(784, 256)` | 200,960 | 256 | 0.13% |
| `Conv2d(64, 128, 3)` | 73,856 | 128 | 0.17% |

The overhead is negligible, making PReLU a low-cost improvement over Leaky ReLU.

---

## Summary

| Aspect | PReLU |
|--------|-------|
| **Formula** | $\max(x, ax)$ where $a$ is learned |
| **Range** | $(-\infty, +\infty)$ |
| **Dead neurons** | ✅ None |
| **Learnable** | ✅ Yes (negative slope per channel) |
| **Initialization** | $a = 0.25$ (default) |
| **Best for** | Image classification, when optimal slope is unknown |
| **Parameter overhead** | Negligible (1 per channel) |

!!! tip "Practical Recommendation"
    PReLU is a strong choice when you suspect Leaky ReLU would help but don't want to tune the negative slope manually. The parameter overhead is negligible, and the network will learn appropriate slopes for each channel. Use per-channel mode (`num_parameters=out_channels`) for best results.
