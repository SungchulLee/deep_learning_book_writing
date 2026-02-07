# ReLU (Rectified Linear Unit)

## Overview

The **Rectified Linear Unit (ReLU)** is the most widely used activation function in deep learning. Introduced to neural networks by Nair & Hinton (2010) and popularized by Krizhevsky et al. (2012) in AlexNet, ReLU solved the vanishing gradient problem that plagued sigmoid and tanh, enabling the training of much deeper networks. It remains the default choice for hidden layers in most convolutional neural networks.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and derivative of ReLU
2. Why ReLU solved the vanishing gradient problem
3. The dead ReLU problem: causes, detection, and prevention
4. Proper weight initialization (He/Kaiming) for ReLU networks
5. PyTorch implementation patterns

---

## Mathematical Definition

$$\operatorname{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Derivative

$$\operatorname{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, the subgradient at $x = 0$ is defined as 0 (PyTorch convention). Since the set $\{x = 0\}$ has measure zero, the choice does not affect training.

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $[0, +\infty)$ |
| **Non-saturating** | Yes (for positive inputs) |
| **Sparse activation** | Yes (~50% zeros with standard initialization) |
| **Computational cost** | Very low (single comparison) |
| **Zero-centered** | No (outputs are always $\geq 0$) |
| **Smooth** | No (non-differentiable at $x = 0$) |

---

## Advantages

### 1. No Vanishing Gradient for Positive Values

The gradient of ReLU is exactly 1 for all positive inputs. Unlike sigmoid (max gradient 0.25) or tanh (max gradient 1.0 but decaying rapidly), ReLU propagates gradients without attenuation through the positive regime:

$$\frac{\partial \operatorname{ReLU}(x)}{\partial x} = 1 \quad \text{for } x > 0$$

### 2. Computational Efficiency

ReLU requires only a single comparison operation `max(0, x)`, making it significantly faster than activations requiring exponentials (sigmoid, tanh, ELU) or error functions (GELU).

### 3. Sparse Representations

With standard initialization, approximately 50% of neurons output zero at any given time. This **activation sparsity** can improve computational efficiency and has been linked to better generalization in some settings.

### 4. Faster Convergence

Networks with ReLU activations typically converge ~6× faster than equivalent networks with sigmoid or tanh activations, as demonstrated in the original AlexNet paper.

---

## Disadvantages

### 1. Dead ReLU Problem

Neurons can permanently output 0 for all inputs. See the detailed analysis below.

### 2. Not Zero-Centered

All outputs are non-negative, which can cause zig-zag gradient dynamics (same issue as sigmoid, though less severe in practice due to BatchNorm).

### 3. Unbounded Output

Without normalization, activations can grow unboundedly, potentially causing numerical instability.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Functional API
y = torch.relu(x)
# or: y = F.relu(x)

# Module API
relu = nn.ReLU()
y = relu(x)

# In-place operation (saves memory, but cannot be used if input is needed later)
relu_inplace = nn.ReLU(inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {y.tolist()}")
# Output: [0.0, 0.0, 0.0, 1.0, 2.0]
```

---

## The Dead ReLU Problem

### Mechanism

A "dead" ReLU neuron always outputs zero for **all** inputs in the dataset:

1. During training, a large gradient update pushes weights such that the pre-activation $z = \mathbf{w}^T\mathbf{x} + b$ is negative for all training inputs
2. Since $\operatorname{ReLU}(z) = 0$ for $z < 0$, the gradient $\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial z} \cdot \operatorname{ReLU}'(z) \cdot \mathbf{x} = \mathbf{0}$
3. With zero gradient, the weights never update, and the neuron remains permanently dead

### Causes

- **High learning rate**: Large weight updates overshoot into the all-negative regime
- **Poor initialization**: Neurons start with weights that produce persistently negative pre-activations
- **Large negative bias**: A persistent negative shift that overwhelms inputs

### Detection

```python
import torch
import torch.nn as nn

def detect_dead_neurons(model, dataloader, threshold=0.0):
    """Detect neurons that are always inactive across the dataset."""
    model.eval()
    
    activation_max = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_max:
                activation_max[name] = output.detach().max(dim=0)[0]
            else:
                activation_max[name] = torch.max(
                    activation_max[name],
                    output.detach().max(dim=0)[0]
                )
        return hook
    
    # Register hooks on all ReLU layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # Process entire dataset
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Analyze results
    total_dead = 0
    total_neurons = 0
    
    for name, max_acts in activation_max.items():
        dead = (max_acts <= threshold).sum().item()
        total = max_acts.numel()
        total_dead += dead
        total_neurons += total
        print(f"{name}: {dead}/{total} dead neurons ({100*dead/total:.1f}%)")
    
    print(f"\nTotal: {total_dead}/{total_neurons} dead "
          f"({100*total_dead/total_neurons:.1f}%)")
    
    return total_dead, total_neurons
```

### Prevention Strategies

**1. Use Leaky ReLU or ELU** — These variants allow gradient flow for negative inputs, eliminating the dead neuron problem entirely. See [Leaky ReLU](leaky_relu.md), [ELU](elu.md).

**2. He (Kaiming) Initialization** — Designed specifically for ReLU networks to maintain variance across layers (see below).

**3. Lower Learning Rate** — Reduces the chance of drastic weight updates that push neurons into the dead regime.

**4. Batch Normalization** — Normalizes pre-activations to keep them centered around zero, reducing the probability of persistent negative values.

**5. Small Positive Bias** — Initializing biases to a small positive value (e.g., 0.01) shifts pre-activations slightly positive:

```python
def init_relu_network(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
```

---

## Weight Initialization: He (Kaiming)

### Motivation

For a layer with input $\mathbf{x}$ and weights $\mathbf{W}$, the pre-activation variance is:

$$\operatorname{Var}(z) = n_{\text{in}} \cdot \operatorname{Var}(w) \cdot \operatorname{Var}(x)$$

After ReLU, half the values are zeroed, so the output variance is approximately halved. To compensate and maintain $\operatorname{Var}(\text{output}) \approx \operatorname{Var}(\text{input})$, He initialization sets:

$$W \sim \mathcal{N}\!\left(0,\; \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

The factor of 2 (instead of 1 in Xavier initialization) accounts for ReLU zeroing ~50% of activations.

### PyTorch Implementation

```python
import torch.nn as nn

# He initialization for ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Full network initialization
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model.apply(init_weights)
```

---

## Complete Network Example

### CNN with ReLU + BatchNorm

The most common and well-tested pattern for convolutional networks:

```python
import torch.nn as nn

class ConvNetReLU(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # Raw logits — use CrossEntropyLoss
```

### Layer Ordering Convention

The standard ordering for convolutional blocks is:

```
Conv → BatchNorm → ReLU → Pooling
```

This places normalization before the activation, keeping pre-activations well-conditioned.

---

## Monitoring Activation Health

```python
class ActivationMonitor:
    """Monitor activation statistics during training."""
    
    def __init__(self, model):
        self.stats = {}
        self.handles = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_forward_hook(self._create_hook(name))
                self.handles.append(handle)
    
    def _create_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'mean': output.detach().mean().item(),
                'std': output.detach().std().item(),
                'sparsity': (output.detach() == 0).float().mean().item(),
                'max': output.detach().max().item(),
            }
        return hook
    
    def print_stats(self):
        for name, stat in self.stats.items():
            print(f"{name}: mean={stat['mean']:.4f}, "
                  f"std={stat['std']:.4f}, sparsity={stat['sparsity']:.2%}")
    
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
```

---

## Summary

| Aspect | ReLU |
|--------|------|
| **Formula** | $\max(0, x)$ |
| **Range** | $[0, +\infty)$ |
| **Gradient (positive)** | 1 |
| **Gradient (negative)** | 0 |
| **Dead neurons** | ❌ High risk |
| **Zero-centered** | ❌ No |
| **Computational cost** | ✅ Very low |
| **Best for** | CNNs, general hidden layers |
| **Initialization** | He (Kaiming) Normal |

!!! tip "Practical Recommendation"
    **ReLU + BatchNorm** is the proven default for CNNs. If you observe dead neurons (loss plateaus, sparse activations > 70%), switch to [Leaky ReLU](leaky_relu.md) or [ELU](elu.md). For transformers, use [GELU](gelu.md) instead.
