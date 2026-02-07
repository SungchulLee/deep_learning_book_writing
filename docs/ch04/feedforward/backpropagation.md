# Backpropagation

## Learning Objectives

!!! abstract "What You Will Learn"
    - Derive the backpropagation algorithm from the multivariate chain rule, step by step
    - Compute gradients for a 2-layer network by hand with explicit dimensions at every step
    - State the general backpropagation recurrence for an $L$-layer network
    - Understand backpropagation as reverse-mode automatic differentiation on the computational graph
    - Implement manual backpropagation in Python and verify against PyTorch autograd
    - Analyze the computational complexity: forward and backward passes have the same cost

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| Forward Pass (§4.2.4) | Backprop requires all cached forward-pass values |
| Chain rule (multivariable calculus) | Mathematical foundation of gradient propagation |
| Matrix calculus (Jacobians) | Gradients of matrix expressions |
| Activation function derivatives | Each activation has a specific gradient formula |

---

## Overview

**Backpropagation** (backward propagation of errors) is the algorithm that efficiently computes gradients of the loss function with respect to all network parameters. It was popularized by Rumelhart, Hinton, and Williams (1986) and remains the workhorse of neural network training.

The key insight: by applying the chain rule **in reverse** (from output to input) and reusing intermediate results, backpropagation computes all parameter gradients in $O(|\boldsymbol{\theta}|)$ time — the same order as a single forward pass.

---

## Problem Setup

**Given:**

- Network with $L$ layers, parameters $\boldsymbol{\theta} = \{(\mathbf{W}^{[l]}, \mathbf{b}^{[l]})\}_{l=1}^L$
- Training sample $(\mathbf{x}, \mathbf{y})$
- Loss function $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$ where $\hat{\mathbf{y}} = f(\mathbf{x}; \boldsymbol{\theta})$

**Goal:** Compute the gradients

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} \in \mathbb{R}^{n^{[l]}}
$$

for all layers $l = 1, \ldots, L$, so that gradient descent can update:

$$
\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}, \qquad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}
$$

---

## Chain Rule Foundation

### Scalar Chain Rule

For $f(g(x))$:

$$
\frac{d}{dx} f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### Multivariate Chain Rule

For $\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$ composed with $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial \mathcal{L}}{\partial g_j} \cdot \frac{\partial g_j}{\partial x_i}
$$

In matrix form using the Jacobian $\mathbf{J}_\mathbf{g} \in \mathbb{R}^{m \times n}$ where $(\mathbf{J}_\mathbf{g})_{ji} = \frac{\partial g_j}{\partial x_i}$:

$$
\nabla_\mathbf{x} \mathcal{L} = \mathbf{J}_\mathbf{g}^\top \nabla_\mathbf{g} \mathcal{L}
$$

### Chain Rule for Deep Networks

For a composition $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$, the gradient of $\mathcal{L}$ with respect to the input of layer $l$ involves a product of Jacobians:

$$
\nabla_{\mathbf{a}^{[l-1]}} \mathcal{L} = \mathbf{J}_{f_l}^\top \, \mathbf{J}_{f_{l+1}}^\top \cdots \mathbf{J}_{f_L}^\top \, \nabla_{\hat{\mathbf{y}}} \mathcal{L}
$$

Backpropagation computes this **right to left**, reusing each intermediate result — this is the efficiency trick.

---

## Complete Derivation: 2-Layer Network

We derive all gradients explicitly for a 2-layer network to build intuition before stating the general case.

### Network Setup

$$
\begin{aligned}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} & & \in \mathbb{R}^{n^{[1]}} \\
\mathbf{a}^{[1]} &= \sigma(\mathbf{z}^{[1]}) & & \in \mathbb{R}^{n^{[1]}} \\
z^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]} & & \in \mathbb{R} \\
\hat{y} &= \sigma_{\text{out}}(z^{[2]}) & & \in \mathbb{R}
\end{aligned}
$$

Using sigmoid output and binary cross-entropy loss:

$$
\mathcal{L} = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]
$$

### Step 1: $\partial \mathcal{L} / \partial \hat{y}$

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

### Step 2: $\partial \mathcal{L} / \partial z^{[2]}$ (the "error signal" $\delta^{[2]}$)

Since $\hat{y} = \sigma(z^{[2]})$ and $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$:

$$
\delta^{[2]} \equiv \frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y
$$

!!! note "The Sigmoid + BCE Cancellation"
    The result $\delta^{[2]} = \hat{y} - y$ is remarkably simple: the error signal is just the prediction error. This elegant cancellation arises specifically from the pairing of sigmoid activation with binary cross-entropy loss. The same cancellation occurs for softmax + categorical cross-entropy: $\delta_i^{[L]} = \hat{y}_i - y_i$.

### Step 3: Gradients for Layer 2 Parameters

From $z^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + b^{[2]}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \frac{\partial \mathcal{L}}{\partial z^{[2]}} \cdot \frac{\partial z^{[2]}}{\partial \mathbf{W}^{[2]}} = \delta^{[2]} \left(\mathbf{a}^{[1]}\right)^\top \in \mathbb{R}^{1 \times n^{[1]}}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{[2]}} = \delta^{[2]} \in \mathbb{R}
$$

**Interpretation:** The weight gradient is the outer product of the error signal and the input activation. Weights connecting highly active inputs ($a^{[1]}_j$ large) to neurons with large error ($\delta^{[2]}$ large) receive the largest gradient updates.

### Step 4: Backpropagate to Hidden Layer

The error must flow backward through $\mathbf{W}^{[2]}$ to reach layer 1:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = \left(\mathbf{W}^{[2]}\right)^\top \delta^{[2]} \in \mathbb{R}^{n^{[1]}}
$$

Then through the activation function $\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})$:

$$
\boldsymbol{\delta}^{[1]} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \sigma'(\mathbf{z}^{[1]}) = \left(\mathbf{W}^{[2]}\right)^\top \delta^{[2]} \odot \sigma'(\mathbf{z}^{[1]}) \in \mathbb{R}^{n^{[1]}}
$$

where $\odot$ denotes element-wise (Hadamard) multiplication. The element-wise product arises because each activation function is applied independently to each component.

### Step 5: Gradients for Layer 1 Parameters

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^\top \in \mathbb{R}^{n^{[1]} \times n^{[0]}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \boldsymbol{\delta}^{[1]} \in \mathbb{R}^{n^{[1]}}
$$

---

## General Backpropagation Algorithm

### Forward Pass (compute and cache)

For $l = 1, \ldots, L$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})
$$

Cache all $\{\mathbf{a}^{[0]}, \mathbf{z}^{[1]}, \mathbf{a}^{[1]}, \ldots, \mathbf{z}^{[L]}, \mathbf{a}^{[L]}\}$.

### Backward Pass

**Initialize** the error signal at the output:

$$
\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot (\sigma^{[L]})'(\mathbf{z}^{[L]})
$$

For common loss-activation pairings, this simplifies:

| Output activation + Loss | $\boldsymbol{\delta}^{[L]}$ |
|---|---|
| Sigmoid + BCE | $\hat{\mathbf{y}} - \mathbf{y}$ |
| Softmax + Categorical CE | $\hat{\mathbf{y}} - \mathbf{y}$ |
| Identity + MSE | $\hat{\mathbf{y}} - \mathbf{y}$ (times $2/n$ depending on convention) |

**Recurrence** for $l = L, L-1, \ldots, 1$:

$$
\boxed{
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} &= \boldsymbol{\delta}^{[l]} \left(\mathbf{a}^{[l-1]}\right)^\top \\[4pt]
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} &= \boldsymbol{\delta}^{[l]} \\[4pt]
\boldsymbol{\delta}^{[l-1]} &= \left(\mathbf{W}^{[l]}\right)^\top \boldsymbol{\delta}^{[l]} \odot (\sigma^{[l-1]})'(\mathbf{z}^{[l-1]})
\end{aligned}
}
$$

The three equations above are the entire backpropagation algorithm. Each layer uses the error signal $\boldsymbol{\delta}^{[l]}$ from the layer above, computes parameter gradients (first two equations), and passes the error backward (third equation).

---

## Activation Function Derivatives

| Activation | $\sigma(z)$ | $\sigma'(z)$ | Notes |
|---|---|---|---|
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ | Gradient is exactly 0 or 1; undefined at $z = 0$ (set to 0 by convention) |
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Max derivative is $0.25$ at $z = 0$; causes vanishing gradients |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Max derivative is $1$ at $z = 0$; less severe vanishing than sigmoid |
| Leaky ReLU | $\max(\alpha z, z)$ | $\begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$ | Gradient never fully vanishes ($\alpha \approx 0.01$) |

---

## Batch Backpropagation

For a mini-batch of $B$ samples (row-major convention), the gradient averaged over the batch:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{B} \left(\boldsymbol{\Delta}^{[l]}\right)^\top \mathbf{A}^{[l-1]}
$$

where $\boldsymbol{\Delta}^{[l]} \in \mathbb{R}^{B \times n^{[l]}}$ has each row being the $\boldsymbol{\delta}^{[l]}$ for one sample, and $\mathbf{A}^{[l-1]} \in \mathbb{R}^{B \times n^{[l-1]}}$ has the cached activations.

---

## PyTorch Implementation

### Manual Backpropagation

```python
import torch


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def forward(X, W1, b1, W2, b2):
    """Forward pass: X → ReLU hidden → sigmoid output."""
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a2, {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}


def backward(y, cache, W2):
    """
    Manual backpropagation for 2-layer network.
    
    Returns gradients: dW1, db1, dW2, db2
    """
    X, z1, a1, a2 = cache['X'], cache['z1'], cache['a1'], cache['a2']
    B = X.shape[0]
    
    # ── Output layer: sigmoid + BCE → δ² = a² - y ──
    delta2 = a2 - y                                     # (B, 1)
    
    dW2 = (1/B) * a1.T @ delta2                         # (hidden, 1)
    db2 = (1/B) * delta2.sum(dim=0, keepdim=True)       # (1, 1)
    
    # ── Hidden layer: backprop through W2, then ReLU ──
    delta1 = (delta2 @ W2.T) * (z1 > 0).float()        # (B, hidden)
    
    dW1 = (1/B) * X.T @ delta1                          # (input, hidden)
    db1 = (1/B) * delta1.sum(dim=0, keepdim=True)       # (1, hidden)
    
    return dW1, db1, dW2, db2


# ── Training loop ──
torch.manual_seed(42)

n_in, n_hid, n_out = 2, 8, 1
W1 = torch.randn(n_in, n_hid) * 0.5
b1 = torch.zeros(1, n_hid)
W2 = torch.randn(n_hid, n_out) * 0.5
b2 = torch.zeros(1, n_out)

# XOR-like dataset
X = torch.randn(200, 2)
y = ((X[:, 0] * X[:, 1]) > 0).float().unsqueeze(1)

lr = 0.5
for epoch in range(1000):
    # Forward
    y_pred, cache = forward(X, W1, b1, W2, b2)
    
    # Loss (BCE)
    eps = 1e-7
    yp = y_pred.clamp(eps, 1 - eps)
    loss = -(y * yp.log() + (1 - y) * (1 - yp).log()).mean()
    
    # Backward
    dW1, db1, dW2, db2 = backward(y, cache, W2)
    
    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    if (epoch + 1) % 250 == 0:
        acc = ((y_pred > 0.5).float() == y).float().mean() * 100
        print(f"Epoch {epoch+1:4d}: loss = {loss:.4f}, accuracy = {acc:.1f}%")
```

### Gradient Verification with Autograd

```python
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def numerical_gradient(model, X, y, criterion, param, eps=1e-5):
    """Central-difference numerical gradient for verification."""
    grad = torch.zeros_like(param.data)
    flat = param.data.view(-1)
    
    for i in range(flat.numel()):
        orig = flat[i].item()
        
        flat[i] = orig + eps
        loss_plus = criterion(model(X), y)
        
        flat[i] = orig - eps
        loss_minus = criterion(model(X), y)
        
        flat[i] = orig
        grad.view(-1)[i] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
    
    return grad


# ── Verify ──
torch.manual_seed(42)
model = SimpleNet()
X = torch.randn(20, 2)
y = torch.randint(0, 2, (20, 1)).float()
criterion = nn.BCELoss()

# Autograd gradients
model.zero_grad()
loss = criterion(model(X), y)
loss.backward()

# Compare
print("Gradient Verification (backprop vs. numerical):")
print("-" * 55)
for name, param in model.named_parameters():
    num_grad = numerical_gradient(model, X, y, criterion, param)
    bp_grad = param.grad
    rel_error = (num_grad - bp_grad).abs() / (num_grad.abs() + 1e-8)
    print(f"  {name:15s} | max relative error: {rel_error.max():.2e}")

print("\n✓ All gradients verified!")
```

---

## Computational Complexity

### Forward vs. Backward Cost

| Pass | Dominant operation per layer | Cost per layer |
|------|------------------------------|----------------|
| Forward | $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$ | $O(n^{[l]} \cdot n^{[l-1]})$ |
| Backward (param grad) | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top$ | $O(n^{[l]} \cdot n^{[l-1]})$ |
| Backward (propagate) | $\boldsymbol{\delta}^{[l-1]} = (\mathbf{W}^{[l]})^\top \boldsymbol{\delta}^{[l]}$ | $O(n^{[l]} \cdot n^{[l-1]})$ |

Total backward cost per layer is $\approx 2\times$ the forward cost (two matrix multiplications vs. one). For the full network:

$$
T_{\text{backward}} \approx 2 \cdot T_{\text{forward}} = O(|\boldsymbol{\theta}|)
$$

Total training step (forward + backward + update): $\approx 3 \times$ forward cost.

### Memory

The backward pass requires all cached activations $\{\mathbf{a}^{[l]}\}_{l=0}^{L-1}$ and pre-activations $\{\mathbf{z}^{[l]}\}_{l=1}^L$:

$$
M_{\text{cache}} = O\!\left(B \sum_{l=0}^{L} n^{[l]}\right)
$$

---

## Common Pitfalls

### 1. Forgetting to Zero Gradients

PyTorch accumulates gradients by default (useful for some advanced techniques, but a trap for beginners):

```python
# ✗ Gradients accumulate across iterations
for epoch in range(100):
    loss = criterion(model(X), y)
    loss.backward()       # adds to existing .grad!
    optimizer.step()

# ✓ Zero before each backward pass
for epoch in range(100):
    optimizer.zero_grad()  # clear previous gradients
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### 2. In-Place Operations Breaking Autograd

```python
# ✗ In-place modification breaks the computational graph
x = torch.randn(3, requires_grad=True)
x += 1          # RuntimeError!

# ✓ Create a new tensor
x = torch.randn(3, requires_grad=True)
x = x + 1       # new tensor, graph preserved
```

### 3. Detaching When You Shouldn't

```python
# ✗ .detach() severs gradient flow
hidden = encoder(x).detach()   # encoder receives NO gradients!
output = decoder(hidden)

# ✓ Let gradients flow through
hidden = encoder(x)
output = decoder(hidden)        # encoder is updated via backprop
```

---

## Key Takeaways

!!! success "Summary"
    1. **Backpropagation** is the systematic application of the chain rule in reverse order
    2. The algorithm computes error signals $\boldsymbol{\delta}^{[l]}$ from output to input, where:
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top$ — weight gradient is outer product of error and input
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$ — bias gradient equals the error signal
        - $\boldsymbol{\delta}^{[l-1]} = (\mathbf{W}^{[l]})^\top \boldsymbol{\delta}^{[l]} \odot \sigma'(\mathbf{z}^{[l-1]})$ — error propagates backward
    3. **Computational cost** of the backward pass is $\approx 2\times$ the forward pass
    4. **Memory** is dominated by cached activations (needed for gradient computation)
    5. **Numerical verification** via central differences confirms gradient correctness
    6. **PyTorch autograd** implements backpropagation automatically via the computational graph

---

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.5.
- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. Chapter 2.
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives*. SIAM.
