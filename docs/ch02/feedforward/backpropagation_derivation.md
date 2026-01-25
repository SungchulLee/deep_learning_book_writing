# Backpropagation Derivation

## Overview

**Backpropagation** (backward propagation of errors) is the algorithm that computes gradients of the loss function with respect to all network parameters. It enables efficient training of neural networks by systematically applying the chain rule to propagate error signals from the output layer back to the input layer.

## Historical Context

Backpropagation was popularized by Rumelhart, Hinton, and Williams (1986), though the core ideas had been developed earlier. The algorithm revolutionized neural network training by providing an efficient $O(n)$ method to compute gradients for networks with millions of parameters.

## Problem Setup

Given:

- Network with $L$ layers
- Parameters $\boldsymbol{\theta} = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$
- Training example $(\mathbf{x}, \mathbf{y})$
- Loss function $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$

**Goal:** Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$ for all layers $l$.

## Chain Rule Foundation

The chain rule is the mathematical foundation of backpropagation.

### Scalar Chain Rule

For composed functions $f(g(x))$:

$$
\frac{d}{dx}f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### Multivariate Chain Rule

For $f(\mathbf{g}(\mathbf{x}))$ where $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ and $f: \mathbb{R}^m \to \mathbb{R}$:

$$
\frac{\partial f}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial f}{\partial g_j} \cdot \frac{\partial g_j}{\partial x_i}
$$

In matrix form:

$$
\nabla_\mathbf{x} f = \mathbf{J}_\mathbf{g}^T \nabla_\mathbf{g} f
$$

where $\mathbf{J}_\mathbf{g} \in \mathbb{R}^{m \times n}$ is the Jacobian matrix.

## Derivation for a 2-Layer Network

Let's derive backpropagation for a simple 2-layer network to build intuition.

### Network Structure

$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = \sigma(\mathbf{z}^{[1]})
$$
$$
\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{[2]} = \sigma(\mathbf{z}^{[2]})
$$

### Loss Function

Using binary cross-entropy:

$$
\mathcal{L} = -\left[y \log(\hat{y}) + (1-y) \log(1-\hat{y})\right]
$$

### Step 1: Gradient at Output

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

### Step 2: Gradient w.r.t. $z^{[2]}$ (using sigmoid)

Sigmoid derivative: $\sigma'(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$

$$
\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \hat{y} - y
$$

Define $\boldsymbol{\delta}^{[2]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \hat{\mathbf{y}} - \mathbf{y}$

!!! note "Lucky Cancellation"
    The beautiful result $\boldsymbol{\delta}^{[2]} = \hat{\mathbf{y}} - \mathbf{y}$ comes from the specific combination of sigmoid activation and binary cross-entropy loss. This is why this pairing is commonly used.

### Step 3: Gradients for Layer 2 Parameters

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \boldsymbol{\delta}^{[2]} (\mathbf{a}^{[1]})^T
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} = \boldsymbol{\delta}^{[2]}
$$

### Step 4: Backpropagate to Hidden Layer

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = (\mathbf{W}^{[2]})^T \boldsymbol{\delta}^{[2]}
$$

$$
\boldsymbol{\delta}^{[1]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \sigma'(\mathbf{z}^{[1]})
$$

where $\odot$ denotes element-wise multiplication.

### Step 5: Gradients for Layer 1 Parameters

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \boldsymbol{\delta}^{[1]}
$$

## General Backpropagation Algorithm

### Forward Pass

For $l = 1, \ldots, L$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})
$$

Cache: $\{\mathbf{z}^{[l]}, \mathbf{a}^{[l]}\}$ for all $l$.

### Backward Pass

**Initialize:**

$$
\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot (\sigma^{[L]})'(\mathbf{z}^{[L]})
$$

**For $l = L, L-1, \ldots, 1$:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}
$$

$$
\boldsymbol{\delta}^{[l-1]} = \left[(\mathbf{W}^{[l]})^T \boldsymbol{\delta}^{[l]}\right] \odot (\sigma^{[l-1]})'(\mathbf{z}^{[l-1]})
$$

## Activation Function Derivatives

### ReLU

$$
\text{ReLU}(z) = \max(0, z)
$$

$$
\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases} = \mathbf{1}_{z > 0}
$$

### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

### Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

$$
\tanh'(z) = 1 - \tanh^2(z)
$$

### Softmax (with Cross-Entropy)

For $\mathbf{a} = \text{softmax}(\mathbf{z})$ and cross-entropy loss $\mathcal{L} = -\sum_i y_i \log(a_i)$:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = a_i - y_i
$$

## Batch Backpropagation

For a batch of $B$ samples:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{B} \boldsymbol{\Delta}^{[l]} (\mathbf{A}^{[l-1]})^T
$$

where $\boldsymbol{\Delta}^{[l]} \in \mathbb{R}^{n^{[l]} \times B}$ contains $\boldsymbol{\delta}$ vectors for all samples.

## PyTorch Implementation

### Manual Backpropagation (Educational)

```python
import torch
import numpy as np

torch.manual_seed(42)

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid: σ'(z) = σ(z)(1-σ(z)) = a(1-a)"""
    return a * (1 - a)

def relu(z):
    return torch.maximum(z, torch.tensor(0.0))

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0"""
    return (z > 0).float()

def forward_pass(X, W1, b1, W2, b2):
    """Forward propagation with caching."""
    # Layer 1
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Layer 2
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    
    cache = {
        'X': X,
        'z1': z1, 'a1': a1,
        'z2': z2, 'a2': a2
    }
    return a2, cache

def backward_pass(y_true, y_pred, cache, W1, W2):
    """
    Backpropagation: Compute gradients for all parameters.
    
    Returns:
        dW1, db1, dW2, db2: Gradients for each parameter
    """
    n = y_true.shape[0]
    
    # Retrieve cached values
    X = cache['X']
    z1, a1 = cache['z1'], cache['a1']
    z2, a2 = cache['z2'], cache['a2']
    
    # ============ Output Layer Gradients ============
    # For sigmoid + BCE: δ² = a² - y (lucky cancellation)
    delta2 = a2 - y_true  # Shape: (n, 1)
    
    # Gradients for W2, b2
    dW2 = (1/n) * a1.T @ delta2  # Shape: (hidden, 1)
    db2 = (1/n) * torch.sum(delta2, dim=0, keepdim=True)  # Shape: (1, 1)
    
    # ============ Hidden Layer Gradients ============
    # Backpropagate through W2
    da1 = delta2 @ W2.T  # Shape: (n, hidden)
    
    # Apply ReLU derivative
    delta1 = da1 * relu_derivative(z1)  # Shape: (n, hidden)
    
    # Gradients for W1, b1
    dW1 = (1/n) * X.T @ delta1  # Shape: (input, hidden)
    db1 = (1/n) * torch.sum(delta1, dim=0, keepdim=True)  # Shape: (1, hidden)
    
    return dW1, db1, dW2, db2


# Generate data
n_samples = 100
X = torch.randn(n_samples, 2)
y = ((X[:, 0] * X[:, 1]) > 0).float().reshape(-1, 1)

# Initialize parameters
input_size, hidden_size, output_size = 2, 8, 1
W1 = torch.randn(input_size, hidden_size) * 0.5
b1 = torch.zeros(1, hidden_size)
W2 = torch.randn(hidden_size, output_size) * 0.5
b2 = torch.zeros(1, output_size)

# Training loop
learning_rate = 0.5
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    y_pred, cache = forward_pass(X, W1, b1, W2, b2)
    
    # Compute loss
    eps = 1e-7
    y_pred_clipped = torch.clamp(y_pred, eps, 1 - eps)
    loss = -torch.mean(y * torch.log(y_pred_clipped) + 
                       (1 - y) * torch.log(1 - y_pred_clipped))
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(y, y_pred, cache, W1, W2)
    
    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if (epoch + 1) % 200 == 0:
        accuracy = ((y_pred > 0.5).float() == y).float().mean() * 100
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.1f}%")

print("\n✓ Manual backpropagation successful!")
```

### Verify Against PyTorch Autograd

```python
import torch
import torch.nn as nn

# Same network with autograd
class AutogradNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Compare gradients
torch.manual_seed(42)
model = AutogradNet()

# Use same weights as manual implementation
with torch.no_grad():
    model.fc1.weight.copy_(torch.randn(8, 2) * 0.5)
    model.fc1.bias.zero_()
    model.fc2.weight.copy_(torch.randn(1, 8) * 0.5)
    model.fc2.bias.zero_()

# Forward and backward with autograd
X_auto = X.clone().requires_grad_(False)
y_pred_auto = model(X_auto)

criterion = nn.BCELoss()
loss_auto = criterion(y_pred_auto, y)
loss_auto.backward()

print("Autograd gradients computed!")
print(f"fc1.weight.grad shape: {model.fc1.weight.grad.shape}")
print(f"fc2.weight.grad shape: {model.fc2.weight.grad.shape}")

# Verify manual vs autograd gradients are close
# (would need to use identical initialization to compare exactly)
```

### Complete Training with Gradient Verification

```python
import torch
import torch.nn as nn
import torch.optim as optim

def numerical_gradient(model, X, y, criterion, param, eps=1e-5):
    """
    Compute numerical gradient for verification.
    
    grad ≈ (f(θ + ε) - f(θ - ε)) / (2ε)
    """
    grad = torch.zeros_like(param.data)
    
    for i in range(param.numel()):
        # Store original value
        orig = param.data.view(-1)[i].item()
        
        # f(θ + ε)
        param.data.view(-1)[i] = orig + eps
        loss_plus = criterion(model(X), y)
        
        # f(θ - ε)
        param.data.view(-1)[i] = orig - eps
        loss_minus = criterion(model(X), y)
        
        # Restore and compute gradient
        param.data.view(-1)[i] = orig
        grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
    
    return grad

# Verification
torch.manual_seed(42)
model = AutogradNet()
X_small = torch.randn(10, 2)
y_small = torch.randint(0, 2, (10, 1)).float()
criterion = nn.BCELoss()

# Compute backprop gradient
model.zero_grad()
loss = criterion(model(X_small), y_small)
loss.backward()

# Compare with numerical gradient
for name, param in model.named_parameters():
    if param.grad is not None:
        numerical_grad = numerical_gradient(model, X_small, y_small, criterion, param)
        backprop_grad = param.grad
        
        # Relative error
        error = torch.abs(numerical_grad - backprop_grad) / (torch.abs(numerical_grad) + 1e-8)
        max_error = error.max().item()
        
        print(f"{name}: max relative error = {max_error:.2e}")
        assert max_error < 1e-4, f"Gradient check failed for {name}!"

print("\n✓ Gradient check passed! Backprop gradients are correct.")
```

## Computational Complexity

### Forward Pass

$$
O\left(\sum_{l=1}^{L} n^{[l-1]} \cdot n^{[l]}\right) = O(\text{parameters})
$$

### Backward Pass

**Same complexity as forward pass!**

$$
O\left(\sum_{l=1}^{L} n^{[l-1]} \cdot n^{[l]}\right) = O(\text{parameters})
$$

### Memory Requirements

Must store all intermediate activations:

$$
O\left(\sum_{l=0}^{L} n^{[l]}\right) = O(\text{neurons})
$$

## Common Pitfalls

### 1. Forgetting to Zero Gradients

```python
# ❌ Wrong: gradients accumulate
for epoch in range(100):
    loss = criterion(model(X), y)
    loss.backward()  # Gradients add to existing!
    optimizer.step()

# ✅ Correct: zero before backward
for epoch in range(100):
    optimizer.zero_grad()  # Clear previous gradients
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### 2. In-Place Operations Breaking Autograd

```python
# ❌ Wrong: in-place operation
x = torch.randn(3, requires_grad=True)
x += 1  # In-place modification!

# ✅ Correct: create new tensor
x = torch.randn(3, requires_grad=True)
x = x + 1  # New tensor
```

### 3. Detaching When You Shouldn't

```python
# ❌ Wrong: detach breaks gradient flow
hidden = model.encoder(x).detach()  # No gradients to encoder!
output = model.decoder(hidden)

# ✅ Correct: let gradients flow
hidden = model.encoder(x)
output = model.decoder(hidden)
```

## Key Takeaways

!!! success "Summary"
    1. **Backpropagation** is the efficient application of the chain rule
    2. **Forward pass** computes outputs and caches intermediate values
    3. **Backward pass** propagates $\boldsymbol{\delta}$ from output to input
    4. **Gradient formulas:**
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$
        - $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$
        - $\boldsymbol{\delta}^{[l-1]} = [(\mathbf{W}^{[l]})^T \boldsymbol{\delta}^{[l]}] \odot \sigma'(\mathbf{z}^{[l-1]})$
    5. **Complexity** is $O(\text{parameters})$ — same as forward pass
    6. **PyTorch autograd** implements this automatically via computational graphs

## References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.5.
- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. Chapter 2.
