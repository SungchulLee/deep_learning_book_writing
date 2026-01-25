# Sigmoid and Tanh Activation Functions

## Overview

**Sigmoid** and **Tanh** are the classical activation functions that dominated early neural network research. While largely superseded by ReLU variants in deep networks, they remain essential for specific applications: sigmoid for binary output probabilities and tanh for gating mechanisms in recurrent networks.

## Learning Objectives

By the end of this section, you will understand:

1. Mathematical formulations and derivatives of sigmoid and tanh
2. Output ranges and their implications
3. The vanishing gradient problem
4. Modern use cases where these functions remain essential
5. PyTorch implementation patterns

---

## Sigmoid Function

### Mathematical Definition

The sigmoid function (also called the logistic function) maps any real number to the range $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Properties

| Property | Value |
|----------|-------|
| **Output range** | $(0, 1)$ |
| **Domain** | $(-\infty, +\infty)$ |
| **Monotonic** | Yes (strictly increasing) |
| **Centered** | No (mean output ≈ 0.5) |
| **Maximum gradient** | 0.25 (at $x = 0$) |

### Derivative

The sigmoid has an elegant derivative that can be expressed in terms of itself:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

This property makes backpropagation efficient as the derivative can be computed from the forward pass output.

**Proof:**
$$\begin{aligned}
\sigma'(x) &= \frac{d}{dx}\left[\frac{1}{1 + e^{-x}}\right] \\
&= \frac{e^{-x}}{(1 + e^{-x})^2} \\
&= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
&= \sigma(x) \cdot (1 - \sigma(x))
\end{aligned}$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Functional API
y_functional = torch.sigmoid(x)
# or: y_functional = F.sigmoid(x)

# Module API
sigmoid_module = nn.Sigmoid()
y_module = sigmoid_module(x)

print(f"Input:   {x.tolist()}")
print(f"Sigmoid: {[f'{v:.4f}' for v in y_functional.tolist()]}")
# Output: ['0.0474', '0.2689', '0.5000', '0.7311', '0.9526']
```

### Gradient Computation

```python
import torch

x = torch.tensor([0.0], requires_grad=True)
y = torch.sigmoid(x)
y.backward()

print(f"σ(0) = {y.item():.4f}")           # 0.5
print(f"σ'(0) = {x.grad.item():.4f}")     # 0.25

# Verify: σ(x) * (1 - σ(x)) = 0.5 * 0.5 = 0.25
```

---

## Tanh Function

### Mathematical Definition

The hyperbolic tangent maps any real number to the range $(-1, 1)$:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

### Relationship to Sigmoid

Tanh is a scaled and shifted version of sigmoid:

$$\tanh(x) = 2\sigma(2x) - 1$$

Or equivalently:

$$\sigma(x) = \frac{\tanh(x/2) + 1}{2}$$

### Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-1, 1)$ |
| **Domain** | $(-\infty, +\infty)$ |
| **Monotonic** | Yes (strictly increasing) |
| **Centered** | Yes (zero-centered) |
| **Maximum gradient** | 1.0 (at $x = 0$) |
| **Symmetry** | Odd function: $\tanh(-x) = -\tanh(x)$ |

### Derivative

$$\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Functional API
y_functional = torch.tanh(x)

# Module API
tanh_module = nn.Tanh()
y_module = tanh_module(x)

print(f"Input: {x.tolist()}")
print(f"Tanh:  {[f'{v:.4f}' for v in y_functional.tolist()]}")
# Output: ['-0.9951', '-0.7616', '0.0000', '0.7616', '0.9951']
```

---

## Comparison: Sigmoid vs Tanh

### Output Distribution

| Aspect | Sigmoid | Tanh |
|--------|---------|------|
| **Range** | $(0, 1)$ | $(-1, 1)$ |
| **Zero crossing** | $\sigma(0) = 0.5$ | $\tanh(0) = 0$ |
| **Zero-centered** | No | Yes |
| **Gradient at origin** | 0.25 | 1.0 |

### Why Zero-Centered Matters

When activations are not zero-centered (as with sigmoid), the gradients of weights in subsequent layers will all have the same sign, leading to **zig-zag dynamics** during gradient descent:

```python
# Sigmoid outputs are always positive
# If h = sigmoid(Wx) and all h_i > 0
# Then ∂L/∂w_i will have the same sign for all i
# This causes inefficient optimization
```

!!! tip "Practical Recommendation"
    Between sigmoid and tanh for hidden layers, **prefer tanh** due to its zero-centered output and stronger gradients. However, for most modern applications, **ReLU variants are preferred** for both.

### Visualization Comparison

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Activations
axes[0].plot(x.numpy(), torch.sigmoid(x).numpy(), label='Sigmoid', linewidth=2)
axes[0].plot(x.numpy(), torch.tanh(x).numpy(), label='Tanh', linewidth=2)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Activation Functions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Derivatives (computed numerically for illustration)
x_grad = torch.linspace(-5, 5, 1000, requires_grad=True)
sigmoid_grad = torch.sigmoid(x_grad) * (1 - torch.sigmoid(x_grad))
tanh_grad = 1 - torch.tanh(x_grad)**2

axes[1].plot(x.numpy(), sigmoid_grad.detach().numpy(), 
             label="Sigmoid' (max=0.25)", linewidth=2)
axes[1].plot(x.numpy(), tanh_grad.detach().numpy(), 
             label="Tanh' (max=1.0)", linewidth=2)
axes[1].set_title('Derivatives')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Saturation regions
axes[2].fill_between(x.numpy(), 0, 1, where=x.numpy() < -3, 
                      alpha=0.3, color='red', label='Saturation')
axes[2].fill_between(x.numpy(), 0, 1, where=x.numpy() > 3, 
                      alpha=0.3, color='red')
axes[2].plot(x.numpy(), sigmoid_grad.detach().numpy(), linewidth=2)
axes[2].set_title('Gradient Saturation (Sigmoid)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## The Vanishing Gradient Problem

### Mechanism

Both sigmoid and tanh **saturate** at extreme values:

- As $x \to +\infty$: gradients approach 0
- As $x \to -\infty$: gradients approach 0

In deep networks, these small gradients compound multiplicatively through the chain rule:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_n} \cdot \prod_{i=1}^{n-1} \sigma'(z_i) \cdot \prod_{i=1}^{n} W_i$$

If each $\sigma'(z_i) < 0.25$ (sigmoid) or $< 1$ (tanh), the product rapidly approaches zero.

### Quantitative Analysis

For a 10-layer network with sigmoid activations, if each gradient is 0.25:

$$\text{Gradient magnitude} \propto (0.25)^{10} = 9.5 \times 10^{-7}$$

This means early layers receive virtually no learning signal!

### PyTorch Demonstration

```python
import torch
import torch.nn as nn

class DeepSigmoidNetwork(nn.Module):
    def __init__(self, depth=10):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(64, 64), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test gradient flow
model = DeepSigmoidNetwork(depth=10)
x = torch.randn(32, 64, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()

# Gradient at input is extremely small
print(f"Input gradient norm: {x.grad.norm().item():.2e}")
# Typically: ~1e-7 or smaller
```

---

## Modern Use Cases

Despite limitations in hidden layers, sigmoid and tanh remain essential for specific purposes:

### Sigmoid: Binary Classification Output

For binary classification, the final layer should output a probability:

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()  # Hidden layer uses ReLU
        # No sigmoid in forward - use BCEWithLogitsLoss
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)  # Return logits
        return logits
    
    def predict_proba(self, x):
        """For inference, convert logits to probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

# Training
model = BinaryClassifier(10, 32)
criterion = nn.BCEWithLogitsLoss()  # Applies sigmoid internally

x = torch.randn(8, 10)
y = torch.randint(0, 2, (8, 1)).float()

logits = model(x)
loss = criterion(logits, y)  # Numerically stable
```

!!! warning "Best Practice"
    **Never apply sigmoid before BCEWithLogitsLoss!** The loss function applies sigmoid internally with numerical stability improvements. Doing it twice will cause incorrect training.

### Tanh: LSTM and GRU Gates

Tanh is used in recurrent network architectures for bounded state representations:

```python
# Inside LSTM cell (simplified)
def lstm_cell(x, h_prev, c_prev, W):
    # Gates use sigmoid (bounded 0-1)
    i = torch.sigmoid(W_i @ torch.cat([h_prev, x]))  # Input gate
    f = torch.sigmoid(W_f @ torch.cat([h_prev, x]))  # Forget gate
    o = torch.sigmoid(W_o @ torch.cat([h_prev, x]))  # Output gate
    
    # Candidate uses tanh (bounded -1 to 1)
    c_tilde = torch.tanh(W_c @ torch.cat([h_prev, x]))
    
    # Cell state update
    c = f * c_prev + i * c_tilde
    
    # Hidden state uses tanh
    h = o * torch.tanh(c)
    
    return h, c
```

### Tanh: Output Normalization

When outputs need to be bounded between -1 and 1:

```python
# Actor network in reinforcement learning
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh bounds output to [-1, 1], then scale
        action = self.max_action * torch.tanh(self.fc3(x))
        return action
```

---

## Implementation Details

### Numerical Stability

Both functions can overflow for extreme inputs. PyTorch handles this automatically:

```python
x_large = torch.tensor([100.0, -100.0])

# PyTorch handles overflow gracefully
print(torch.sigmoid(x_large))  # [1.0, 0.0] - no NaN
print(torch.tanh(x_large))     # [1.0, -1.0] - no NaN
```

### Module vs Functional

```python
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(32, 64)

# Module API - use when storing as class attribute
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()  # Stored activation
    
    def forward(self, x):
        return self.sigmoid(x)

# Functional API - use inline
def forward(x):
    return torch.sigmoid(x)  # Direct call
```

---

## Summary

| Aspect | Sigmoid | Tanh |
|--------|---------|------|
| **Range** | $(0, 1)$ | $(-1, 1)$ |
| **Zero-centered** | ❌ No | ✅ Yes |
| **Max gradient** | 0.25 | 1.0 |
| **Vanishing gradients** | ❌ Severe | ⚠️ Moderate |
| **Hidden layers (deep)** | ❌ Avoid | ⚠️ Avoid |
| **Binary output** | ✅ Essential | ❌ Not used |
| **RNN gates** | ✅ Essential | ✅ Essential |

!!! tip "Modern Practice"
    - **Hidden layers**: Use ReLU family or GELU instead
    - **Binary classification output**: Use BCEWithLogitsLoss (applies sigmoid internally)
    - **Multiclass output**: Use CrossEntropyLoss (applies softmax internally)
    - **RNN architectures**: Sigmoid and tanh remain standard for gates
