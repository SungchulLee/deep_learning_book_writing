# Tanh Activation Function

## Overview

The **hyperbolic tangent** (tanh) is a classical activation function that maps inputs to the range $(-1, 1)$. While largely superseded by ReLU variants in deep hidden layers, tanh remains essential inside recurrent network architectures (LSTM, GRU) for bounded state representations and is preferred over sigmoid when a zero-centered activation is needed.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical formulation and derivative of tanh
2. Its relationship to the sigmoid function
3. Why zero-centered output matters for optimization
4. The vanishing gradient problem with tanh (and why it is less severe than sigmoid)
5. Modern use cases: RNN cell states, bounded outputs, and normalization

---

## Mathematical Definition

The hyperbolic tangent maps any real number to the range $(-1, 1)$:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{e^{2x} - 1}{e^{2x} + 1}$$

---

## Relationship to Sigmoid

Tanh is a scaled and shifted version of the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$:

$$\tanh(x) = 2\sigma(2x) - 1$$

Or equivalently:

$$\sigma(x) = \frac{\tanh(x/2) + 1}{2}$$

This relationship means that any theoretical result about sigmoid transfers directly to tanh (and vice versa) via an affine transformation. In particular, both saturate for large $|x|$, but tanh's output is centered around zero.

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(-1, 1)$ |
| **Domain** | $(-\infty, +\infty)$ |
| **Monotonic** | Yes (strictly increasing) |
| **Zero-centered** | Yes |
| **Maximum gradient** | 1.0 (at $x = 0$) |
| **Symmetry** | Odd function: $\tanh(-x) = -\tanh(x)$ |
| **Fixed point** | $\tanh(0) = 0$ |

---

## Derivative

$$\tanh'(x) = 1 - \tanh^2(x) = \operatorname{sech}^2(x)$$

**Proof:**

$$\begin{aligned}
\tanh'(x) &= \frac{d}{dx}\left[\frac{e^x - e^{-x}}{e^x + e^{-x}}\right] \\[6pt]
&= \frac{(e^x + e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})(e^x - e^{-x})}{(e^x + e^{-x})^2} \\[6pt]
&= \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} \\[6pt]
&= 1 - \left(\frac{e^x - e^{-x}}{e^x + e^{-x}}\right)^2 \\[6pt]
&= 1 - \tanh^2(x)
\end{aligned}$$

Like sigmoid, the derivative can be computed directly from the forward-pass output, saving computation during backpropagation.

### Gradient Bounds

Since $\tanh(x) \in (-1,1)$, we have $\tanh^2(x) \in [0,1)$, so:

$$\tanh'(x) = 1 - \tanh^2(x) \in (0, 1]$$

The maximum gradient of **1.0** (at $x = 0$) is four times larger than sigmoid's maximum of 0.25. This is a significant advantage for gradient flow, though saturation still occurs for $|x| \gg 0$.

---

## PyTorch Implementation

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

### Gradient Computation

```python
import torch

x = torch.tensor([0.0], requires_grad=True)
y = torch.tanh(x)
y.backward()

print(f"tanh(0) = {y.item():.4f}")          # 0.0
print(f"tanh'(0) = {x.grad.item():.4f}")    # 1.0

# Verify: 1 - tanh²(0) = 1 - 0 = 1.0
```

---

## Comparison with Sigmoid

| Aspect | Sigmoid | Tanh |
|--------|---------|------|
| **Range** | $(0, 1)$ | $(-1, 1)$ |
| **Zero crossing** | $\sigma(0) = 0.5$ | $\tanh(0) = 0$ |
| **Zero-centered** | No | Yes |
| **Gradient at origin** | 0.25 | 1.0 |
| **Saturates** | Yes (both tails) | Yes (both tails) |
| **Vanishing gradient severity** | Severe | Moderate |

### Why Zero-Centered Matters

When activations are not zero-centered (as with sigmoid), the gradients of weights in subsequent layers will all have the same sign. Consider a neuron computing $z = \sum_j w_j h_j + b$ where $h_j = \sigma(\cdot) > 0$ for all $j$. The weight gradient is:

$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \cdot h_j$$

Since every $h_j > 0$, the gradient components $\frac{\partial L}{\partial w_j}$ all share the sign of $\frac{\partial L}{\partial z}$, restricting update directions to a single orthant and producing **zig-zag optimization paths**.

Tanh avoids this by producing both positive and negative outputs, allowing gradient components to have mixed signs.

### Visualization

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Activations
axes[0].plot(x.numpy(), torch.sigmoid(x).numpy(), label='Sigmoid', linewidth=2)
axes[0].plot(x.numpy(), torch.tanh(x).numpy(), label='Tanh', linewidth=2)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Sigmoid vs Tanh')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Derivatives
sigmoid_grad = torch.sigmoid(x) * (1 - torch.sigmoid(x))
tanh_grad = 1 - torch.tanh(x)**2

axes[1].plot(x.numpy(), sigmoid_grad.numpy(), 
             label="Sigmoid' (max=0.25)", linewidth=2)
axes[1].plot(x.numpy(), tanh_grad.numpy(), 
             label="Tanh' (max=1.0)", linewidth=2)
axes[1].set_title('Derivatives')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## The Vanishing Gradient Problem

### Tanh vs Sigmoid

Both tanh and sigmoid saturate for large $|x|$, leading to vanishing gradients in deep networks. However, tanh is **less severe** because:

1. **Larger maximum gradient**: 1.0 vs 0.25
2. **Zero-centered**: Does not force same-sign gradients

For a 10-layer network where each gradient is at its maximum:

| Activation | Max Gradient | 10-layer Product |
|-----------|-------------|-----------------|
| Sigmoid | 0.25 | $(0.25)^{10} \approx 10^{-6}$ |
| Tanh | 1.0 | $(1.0)^{10} = 1.0$ |

In practice, pre-activations are rarely at exactly zero, so tanh gradients are typically between 0 and 1, still compounding to small values in very deep networks. Nevertheless, the improvement over sigmoid is substantial.

!!! tip "Practical Recommendation"
    Between sigmoid and tanh for hidden layers, **prefer tanh** due to its zero-centered output and stronger gradients. However, for most modern applications, **ReLU variants or GELU are preferred** for both.

---

## Modern Use Cases

### LSTM and GRU Cell States

Tanh is used in recurrent architectures for bounded state representations. In an LSTM, tanh appears in two roles:

1. **Candidate cell state**: Produces values in $(-1, 1)$ for the new information to store
2. **Output squashing**: Bounds the cell state before producing the hidden state

```python
# Inside LSTM cell (simplified)
def lstm_cell(x, h_prev, c_prev, W_i, W_f, W_o, W_c):
    # Gates use sigmoid (bounded 0-1)
    i = torch.sigmoid(W_i @ torch.cat([h_prev, x]))  # Input gate
    f = torch.sigmoid(W_f @ torch.cat([h_prev, x]))  # Forget gate
    o = torch.sigmoid(W_o @ torch.cat([h_prev, x]))  # Output gate
    
    # Candidate uses tanh (bounded -1 to 1)
    c_tilde = torch.tanh(W_c @ torch.cat([h_prev, x]))
    
    # Cell state update
    c = f * c_prev + i * c_tilde
    
    # Hidden state: tanh squashes cell state
    h = o * torch.tanh(c)
    
    return h, c
```

### Bounded Output for Reinforcement Learning

When network outputs need to be bounded, tanh provides a smooth mapping to $[-1, 1]$ that can then be rescaled:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor network in continuous-action RL (e.g., DDPG, TD3, SAC)."""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh bounds output to [-1, 1], then scale to action range
        action = self.max_action * torch.tanh(self.fc3(x))
        return action
```

### Image Generation (GAN Generator Output)

For GANs where images are normalized to $[-1, 1]$:

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # ... hidden layers with ReLU ...
            nn.Linear(512, img_channels * 64 * 64),
            nn.Tanh()  # Output in [-1, 1] matching image normalization
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 3, 64, 64)
```

---

## Numerical Stability

PyTorch handles extreme inputs gracefully:

```python
x_large = torch.tensor([100.0, -100.0])

# PyTorch handles overflow gracefully
print(torch.tanh(x_large))  # tensor([1., -1.]) — no NaN
```

---

## Module vs Functional API

```python
import torch.nn as nn

x = torch.randn(32, 64)

# Module API — use when storing as a class attribute
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        return self.tanh(x)

# Functional API — use inline
def forward(x):
    return torch.tanh(x)
```

Both are functionally identical; the module form is conventional when the activation appears in `nn.Sequential` or needs to be visible in `model.modules()`.

---

## Summary

| Aspect | Tanh |
|--------|------|
| **Formula** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ |
| **Range** | $(-1, 1)$ |
| **Zero-centered** | ✅ Yes |
| **Max gradient** | 1.0 |
| **Vanishing gradients** | ⚠️ Moderate |
| **Hidden layers (deep)** | ⚠️ Avoid (use ReLU/GELU) |
| **RNN cell states** | ✅ Essential |
| **Bounded output** | ✅ Useful for $[-1, 1]$ range |
| **GAN generator output** | ✅ Standard choice |

!!! tip "Modern Practice"
    - **Hidden layers**: Use ReLU family or GELU instead
    - **RNN architectures**: Tanh remains standard for cell states and candidates
    - **Bounded outputs**: Tanh is the natural choice when output must be in $[-1, 1]$
    - **Between sigmoid and tanh for hidden layers**: Always prefer tanh (zero-centered, stronger gradients)
