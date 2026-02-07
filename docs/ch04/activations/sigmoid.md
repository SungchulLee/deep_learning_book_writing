# Sigmoid Activation Function

## Overview

The **sigmoid** function (also called the logistic function) is one of the oldest activation functions in neural network history. While largely superseded by ReLU variants in hidden layers of deep networks, sigmoid remains essential for **binary output probabilities** and as a gating mechanism inside recurrent architectures (LSTM, GRU).

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical formulation and derivative of the sigmoid function
2. Its output range, shape, and key properties
3. The vanishing gradient problem and why sigmoid causes it
4. Modern use cases where sigmoid remains essential
5. PyTorch implementation patterns and numerical stability considerations

---

## Mathematical Definition

The sigmoid function maps any real number to the interval $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Equivalent Forms

The sigmoid can be written in several equivalent ways, each highlighting a different interpretation:

$$\sigma(x) = \frac{e^x}{e^x + 1} = \frac{1}{2} + \frac{1}{2}\tanh\!\left(\frac{x}{2}\right)$$

The last form makes explicit that sigmoid is an affine rescaling of the hyperbolic tangent (see [Tanh](tanh.md) for details on that relationship).

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(0, 1)$ |
| **Domain** | $(-\infty, +\infty)$ |
| **Monotonic** | Yes (strictly increasing) |
| **Zero-centered** | No (mean output $\approx 0.5$) |
| **Maximum gradient** | 0.25 (at $x = 0$) |
| **Symmetry** | $\sigma(-x) = 1 - \sigma(x)$ |
| **Fixed point** | $\sigma(0) = 0.5$ |

---

## Derivative

The sigmoid has an elegant derivative that can be expressed in terms of itself:

$$\sigma'(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)$$

This property makes backpropagation efficient because the derivative can be computed directly from the forward pass output without re-evaluating exponentials.

**Proof:**

$$\begin{aligned}
\sigma'(x) &= \frac{d}{dx}\left[\frac{1}{1 + e^{-x}}\right] \\[6pt]
&= \frac{e^{-x}}{(1 + e^{-x})^2} \\[6pt]
&= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\[6pt]
&= \frac{1}{1 + e^{-x}} \cdot \frac{(1 + e^{-x}) - 1}{1 + e^{-x}} \\[6pt]
&= \sigma(x) \cdot \bigl(1 - \sigma(x)\bigr)
\end{aligned}$$

### Gradient Bounds

Since $\sigma(x) \in (0,1)$, the product $\sigma(x)(1-\sigma(x))$ is maximized when $\sigma(x) = 0.5$, i.e., at $x = 0$:

$$\max_x \sigma'(x) = 0.5 \times 0.5 = 0.25$$

This means the gradient is **always at most 0.25**, which is central to the vanishing gradient problem discussed below.

---

## PyTorch Implementation

### Functional and Module APIs

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# Functional API
y_functional = torch.sigmoid(x)
# or equivalently: y_functional = F.sigmoid(x)

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

## The Vanishing Gradient Problem

### Mechanism

Sigmoid **saturates** at extreme values: as $|x|$ grows, the gradient $\sigma'(x)$ approaches zero exponentially. In deep networks, these small gradients compound multiplicatively through the chain rule:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_n} \cdot \prod_{i=1}^{n-1} \sigma'(z_i) \cdot \prod_{i=1}^{n} W_i$$

### Quantitative Analysis

For a 10-layer network with sigmoid activations, if each gradient is at its maximum of 0.25:

$$\text{Gradient magnitude} \propto (0.25)^{10} = 9.5 \times 10^{-7}$$

This means early layers receive virtually no learning signal. In practice, pre-activations are rarely exactly zero, so the actual gradients are often even smaller.

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

!!! warning "Do Not Use Sigmoid in Hidden Layers of Deep Networks"
    The vanishing gradient problem makes sigmoid unsuitable as a hidden-layer activation in networks deeper than 2–3 layers. Use ReLU variants or GELU instead.

---

## Why Sigmoid Is Not Zero-Centered

Sigmoid outputs are always positive: $\sigma(x) \in (0, 1)$. When activations from the previous layer are all positive, the gradients of the weights in the next layer must all have the same sign:

$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \cdot h_j$$

where $h_j = \sigma(\cdot) > 0$. This forces weight updates into a constrained subspace, producing **zig-zag gradient dynamics** that slow convergence.

This is one reason why zero-centered activations like Tanh and modern alternatives (GELU, Swish) are preferred for hidden layers.

---

## Modern Use Cases

Despite its limitations for hidden layers, sigmoid remains essential in several contexts:

### Binary Classification Output

For binary classification, the final layer should produce a probability. However, best practice is to **not apply sigmoid explicitly** and instead use `BCEWithLogitsLoss`, which fuses the sigmoid and binary cross-entropy for numerical stability:

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()  # Hidden layer uses ReLU
        # No sigmoid in forward — use BCEWithLogitsLoss
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)  # Return raw logits
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
    **Never apply sigmoid before `BCEWithLogitsLoss`!** The loss function applies sigmoid internally with numerical stability improvements (log-sum-exp trick). Applying sigmoid twice will cause incorrect training.

### Gating Mechanisms in LSTMs and GRUs

Sigmoid produces values in $(0, 1)$, making it ideal for **gates** that control information flow:

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
    
    # Hidden state uses tanh
    h = o * torch.tanh(c)
    
    return h, c
```

### Attention Weights and Multi-label Classification

Sigmoid is also used when outputs should be **independent** probabilities (not summing to 1):

```python
# Multi-label classification: each class independent
logits = model(x)  # Shape: [batch, num_labels]
probs = torch.sigmoid(logits)  # Each label independently in (0, 1)
```

---

## Numerical Stability

PyTorch's built-in sigmoid handles extreme inputs gracefully:

```python
x_large = torch.tensor([100.0, -100.0])

# PyTorch handles overflow gracefully
print(torch.sigmoid(x_large))  # tensor([1., 0.]) — no NaN or Inf
```

For manual implementations, the numerically stable form is:

$$\sigma(x) = \begin{cases} \frac{1}{1+e^{-x}} & \text{if } x \geq 0 \\ \frac{e^x}{1+e^x} & \text{if } x < 0 \end{cases}$$

This avoids computing $e^{|x|}$ for large $|x|$ in the denominator.

---

## Visualization

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-6, 6, 1000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sigmoid function
axes[0].plot(x.numpy(), torch.sigmoid(x).numpy(), linewidth=2.5, color='#2196F3')
axes[0].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('Sigmoid: σ(x) = 1 / (1 + e⁻ˣ)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('σ(x)')
axes[0].grid(True, alpha=0.3)

# Sigmoid derivative
sig = torch.sigmoid(x)
sig_grad = sig * (1 - sig)
axes[1].plot(x.numpy(), sig_grad.numpy(), linewidth=2.5, color='#FF5722')
axes[1].axhline(y=0.25, color='k', linestyle='--', alpha=0.3, label='max = 0.25')
axes[1].set_title("Sigmoid Derivative: σ'(x) = σ(x)(1 - σ(x))")
axes[1].set_xlabel('x')
axes[1].set_ylabel("σ'(x)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## Summary

| Aspect | Sigmoid |
|--------|---------|
| **Formula** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| **Range** | $(0, 1)$ |
| **Zero-centered** | ❌ No |
| **Max gradient** | 0.25 |
| **Vanishing gradients** | ❌ Severe |
| **Hidden layers (deep)** | ❌ Avoid |
| **Binary output** | ✅ Essential (via `BCEWithLogitsLoss`) |
| **RNN gates** | ✅ Essential |
| **Multi-label output** | ✅ Useful |

!!! tip "Modern Practice"
    - **Hidden layers**: Use ReLU family or GELU instead
    - **Binary classification output**: Use `BCEWithLogitsLoss` (applies sigmoid internally)
    - **Multi-label classification**: Use `BCEWithLogitsLoss` per label
    - **Gating mechanisms**: Sigmoid remains the standard choice
