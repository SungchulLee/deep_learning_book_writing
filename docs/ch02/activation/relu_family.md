# ReLU Family: ReLU, Leaky ReLU, PReLU, ELU, SELU

## Overview

The **Rectified Linear Unit (ReLU)** family revolutionized deep learning by solving the vanishing gradient problem that plagued sigmoid and tanh activations. ReLU and its variants remain the default choice for hidden layers in most modern neural network architectures.

## Learning Objectives

By the end of this section, you will understand:

1. Mathematical definitions and properties of each ReLU variant
2. The dead ReLU problem and solutions
3. When to use each variant
4. Implementation patterns in PyTorch
5. Proper weight initialization for ReLU networks

---

## ReLU (Rectified Linear Unit)

### Mathematical Definition

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Derivative

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, the derivative at $x = 0$ is defined as either 0 or 1 (PyTorch uses 0).

### Properties

| Property | Value |
|----------|-------|
| **Output range** | $[0, +\infty)$ |
| **Non-saturating** | Yes (for positive inputs) |
| **Sparse activation** | Yes (~50% zeros) |
| **Computational cost** | Very low (single comparison) |
| **Zero-centered** | No |

### Advantages

1. **No vanishing gradient for positive values**: Gradient is exactly 1
2. **Computational efficiency**: Simple `max(0, x)` operation
3. **Sparse representations**: Zero outputs create sparse activations
4. **Faster convergence**: Networks train ~6× faster than sigmoid/tanh

### Disadvantages

1. **Dead ReLU problem**: Neurons can permanently output 0
2. **Not zero-centered**: Can cause zig-zag gradient dynamics
3. **Unbounded output**: Can cause exploding activations without proper normalization

### PyTorch Implementation

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

# In-place operation (saves memory)
relu_inplace = nn.ReLU(inplace=True)

print(f"Input:  {x.tolist()}")
print(f"Output: {y.tolist()}")
# Output: [0.0, 0.0, 0.0, 1.0, 2.0]
```

---

## The Dead ReLU Problem

### Mechanism

A "dead" ReLU neuron always outputs zero for all inputs:

1. During training, a large gradient update pushes weights such that the pre-activation is always negative
2. Since ReLU outputs 0 for negative inputs, the gradient is also 0
3. With zero gradient, weights never update, and the neuron stays dead

### Causes

- **High learning rate**: Large weight updates overshoot
- **Poor initialization**: Neurons start in dead state
- **Large negative bias**: Persistent negative pre-activation

### Detection

```python
import torch
import torch.nn as nn

def count_dead_neurons(model, test_input, threshold=0.0):
    """Count neurons that are always zero across a batch."""
    model.eval()
    dead_count = 0
    
    def hook(module, input, output):
        nonlocal dead_count
        # Check if max activation across batch is zero
        max_activation = output.max(dim=0)[0]
        dead_count += (max_activation <= threshold).sum().item()
    
    handles = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(hook))
    
    with torch.no_grad():
        model(test_input)
    
    for h in handles:
        h.remove()
    
    return dead_count

# Example usage
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
x = torch.randn(1000, 10)  # Large batch for reliable detection
dead = count_dead_neurons(model, x)
print(f"Dead neurons: {dead}")
```

### Solutions

1. **Use Leaky ReLU or ELU**: Allow gradient flow for negative inputs
2. **Lower learning rate**: Prevent drastic weight updates
3. **Proper initialization**: Use He initialization
4. **Batch Normalization**: Keep activations in favorable range

---

## Leaky ReLU

### Mathematical Definition

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases} = \max(\alpha x, x)$$

where $\alpha$ is a small positive constant (typically 0.01 or 0.1).

### Derivative

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

### Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-\infty, +\infty)$ |
| **Non-saturating** | Yes (everywhere) |
| **Dead neurons** | No (gradient always $\geq \alpha$) |
| **Negative slope** | Fixed hyperparameter $\alpha$ |

### PyTorch Implementation

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

print(f"Input:  {x.tolist()}")
print(f"Output: {y.tolist()}")
# Output: [-0.2, -0.1, 0.0, 1.0, 2.0]
```

### Choosing the Negative Slope

| Value | Effect |
|-------|--------|
| `0.01` | Standard, minimal negative flow |
| `0.1` | More negative flow, often works better |
| `0.2-0.3` | Aggressive, closer to linear |

---

## PReLU (Parametric ReLU)

### Mathematical Definition

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ a x & \text{if } x \leq 0 \end{cases}$$

where $a$ is a **learnable parameter** optimized during training.

### Key Difference from Leaky ReLU

- Leaky ReLU: $\alpha$ is a fixed hyperparameter
- PReLU: $a$ is learned from data via backpropagation

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Single learnable parameter for all channels
prelu_shared = nn.PReLU(num_parameters=1)
print(f"Initial a: {prelu_shared.weight.item():.4f}")  # Default: 0.25

# One learnable parameter per channel
prelu_per_channel = nn.PReLU(num_parameters=64)  # For 64-channel input
print(f"Parameters shape: {prelu_per_channel.weight.shape}")  # [64]

# Example network with PReLU
class PReLUNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.prelu1 = nn.PReLU(num_parameters=64)
        self.fc2 = nn.Linear(64, 32)
        self.prelu2 = nn.PReLU(num_parameters=32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        return self.fc3(x)

model = PReLUNetwork()

# Train the network - PReLU parameters are optimized too
optimizer = torch.optim.Adam(model.parameters())  # Includes PReLU weights

# Check learned values after training
print(f"Learned a (layer 1): {model.prelu1.weight.data[:5].tolist()}")
```

### When to Use PReLU

- When you want the network to learn optimal negative slopes
- Often better than fixed Leaky ReLU for image classification
- Adds minimal parameters (one per channel)

---

## ELU (Exponential Linear Unit)

### Mathematical Definition

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

where $\alpha > 0$ (default: 1.0) controls the saturation value for negative inputs.

### Derivative

$$\text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & \text{if } x \leq 0 \end{cases}$$

### Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-\alpha, +\infty)$ |
| **Smooth** | Yes (continuous first derivative) |
| **Near zero-centered** | Yes (negative mean for negative inputs) |
| **Saturation** | Soft saturation at $-\alpha$ |

### Advantages over ReLU

1. **Smooth everywhere**: No discontinuity at $x=0$
2. **Negative values**: Pushes mean activation closer to zero
3. **Noise robustness**: Saturates for large negative values

### Disadvantage

- **Computational cost**: Requires exponential function

### PyTorch Implementation

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

print(f"Input:  {x.tolist()}")
print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
# Output: ['-0.8647', '-0.6321', '0.0000', '1.0000', '2.0000']
```

---

## SELU (Scaled Exponential Linear Unit)

### Mathematical Definition

$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

with specific constants:
- $\lambda \approx 1.0507$
- $\alpha \approx 1.6733$

### Self-Normalizing Property

SELU is designed to automatically maintain:
- Mean ≈ 0
- Variance ≈ 1

throughout the network, without explicit normalization layers.

### Requirements for Self-Normalization

1. Weights initialized with LeCun normal: $\mathcal{N}(0, 1/\text{fan\_in})$
2. Sequential fully-connected architecture
3. No batch normalization (it interferes)

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# SELU with proper initialization
class SELUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.selu = nn.SELU()
        
        # LeCun normal initialization (required for self-normalizing)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
    
    def forward(self, x):
        x = self.selu(self.fc1(x))
        x = self.selu(self.fc2(x))
        return self.fc3(x)

model = SELUNetwork(784, 512, 10)
```

### When to Use SELU

- Deep fully-connected networks
- When batch normalization is undesirable
- Self-normalizing neural networks (SNNs)

!!! warning "Limitations"
    SELU's self-normalizing property does not hold for:
    - Convolutional networks (use Batch/Layer Norm instead)
    - Networks with skip connections
    - Recurrent networks
    - Networks with dropout (use AlphaDropout instead)

---

## Comparison and Selection Guide

### Performance Comparison

| Activation | Dead Neurons | Zero-Centered | Smoothness | Compute Cost |
|------------|--------------|---------------|------------|--------------|
| ReLU | ❌ High risk | ❌ No | ❌ Non-smooth | ✅ Very low |
| Leaky ReLU | ✅ None | ❌ No | ❌ Non-smooth | ✅ Very low |
| PReLU | ✅ None | ❌ No | ❌ Non-smooth | ✅ Low |
| ELU | ✅ None | ✅ Near-zero | ✅ Smooth | ⚠️ Medium |
| SELU | ✅ None | ✅ Yes | ✅ Smooth | ⚠️ Medium |

### Selection Decision Tree

```
Start
  │
  ├─ Deep CNN? ──────────────> ReLU (default) or Leaky ReLU
  │
  ├─ Experiencing dead neurons?
  │     └─ Yes ──────────────> Leaky ReLU or ELU
  │
  ├─ Training instability?
  │     └─ Yes ──────────────> ELU (smoother) or add Batch Norm
  │
  ├─ Deep fully-connected?
  │     └─ Consider SELU with proper initialization
  │
  ├─ Want learnable slopes?
  │     └─ PReLU
  │
  └─ Transformer/NLP? ──────> GELU (see next section)
```

---

## Weight Initialization

### He Initialization (Kaiming)

For ReLU networks, use He initialization to maintain variance across layers:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

```python
import torch.nn as nn

# He initialization for ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# He initialization for Leaky ReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')

# Full network example
def init_weights(module):
    if isinstance(module, nn.Linear):
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

## Complete Network Examples

### CNN with Leaky ReLU

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
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### MLP with PReLU

```python
import torch.nn as nn

class MLPWithPReLU(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.PReLU(num_parameters=hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

---

## Summary

| Activation | Best For | Avoid When |
|------------|----------|------------|
| **ReLU** | Default choice, CNNs, general use | High dead neuron rate |
| **Leaky ReLU** | When ReLU causes issues, GANs | - |
| **PReLU** | Image classification, when data can inform slope | Limited data (overfitting) |
| **ELU** | Smooth gradients needed, noise robustness | Compute-constrained |
| **SELU** | Deep MLPs without normalization | CNNs, RNNs |

!!! tip "Practical Recommendation"
    Start with **ReLU** or **Leaky ReLU**. If training is unstable or you notice dead neurons, try **ELU**. For CNNs, **ReLU + BatchNorm** is the proven combination. For transformers, consider **GELU** (next section).
