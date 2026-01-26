# Backpropagation Through Time (BPTT)

## Introduction

Training recurrent neural networks requires computing gradients through the unrolled computational graph—a procedure called **Backpropagation Through Time (BPTT)**. Understanding BPTT illuminates both how RNNs learn and why they struggle with long sequences.

## The Unrolled Computational Graph

Consider an RNN processing a sequence of length $T$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
$$y_t = W_{hy} h_t$$

Unrolling creates a deep feedforward network where:
- Depth equals sequence length $T$
- Weights are shared across all layers
- Each "layer" corresponds to one timestep

```
Loss = L₁ + L₂ + L₃ + L₄
        ↑    ↑    ↑    ↑
       y₁   y₂   y₃   y₄
        ↑    ↑    ↑    ↑
h₀ → h₁ → h₂ → h₃ → h₄
      ↑    ↑    ↑    ↑
     x₁   x₂   x₃   x₄
```

## Loss Computation

The total loss sums contributions from each timestep:

$$\mathcal{L} = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)$$

For classification with cross-entropy at each step:
$$L_t = -\sum_{c} \hat{y}_{t,c} \log(y_{t,c})$$

For sequence classification, we might only have loss at the final step:
$$\mathcal{L} = L_T(y_T, \hat{y})$$

## Gradient Computation

### Gradients w.r.t. Output Weights

The gradient $\frac{\partial \mathcal{L}}{\partial W_{hy}}$ accumulates contributions from all timesteps:

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot \frac{\partial y_t}{\partial W_{hy}}$$

Since $y_t = W_{hy} h_t$:
$$\frac{\partial y_t}{\partial W_{hy}} = h_t^T$$

### Gradients w.r.t. Hidden State

The gradient at hidden state $h_t$ receives contributions from:
1. The loss at time $t$: $\frac{\partial L_t}{\partial h_t}$
2. The hidden state at time $t+1$: $\frac{\partial L_{t+1:T}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$$

This recursive formula propagates gradients backward through time.

### Gradients w.r.t. Recurrent Weights

The gradient $\frac{\partial \mathcal{L}}{\partial W_{hh}}$ requires summing contributions across all timesteps:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}$$

However, $h_t$ depends on $W_{hh}$ both directly and through $h_{t-1}, h_{t-2}, \ldots, h_1$. The full gradient requires the chain rule through time:

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t} \cdot \frac{\partial^+ h_t}{\partial W_{hh}}$$

Where $\frac{\partial^+ h_t}{\partial W_{hh}}$ denotes the immediate (non-recursive) partial derivative.

## The Chain Rule Through Time

The key term $\frac{\partial h_T}{\partial h_t}$ expands as a product of Jacobians:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Each Jacobian $\frac{\partial h_{k+1}}{\partial h_k}$ is:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(1 - h_{k+1}^2) \cdot W_{hh}$$

Where $\text{diag}(1 - h_{k+1}^2)$ is the derivative of $\tanh$ evaluated at the pre-activation.

## Implementation in PyTorch

PyTorch handles BPTT automatically through its autograd system:

```python
import torch
import torch.nn as nn

# Model
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
fc = nn.Linear(20, 5)

# Data
x = torch.randn(32, 15, 10, requires_grad=True)
targets = torch.randint(0, 5, (32,))

# Forward pass (builds computational graph)
outputs, h_n = rnn(x)
logits = fc(h_n.squeeze(0))

# Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)

# Backward pass (BPTT happens automatically)
loss.backward()

# Gradients are now available
print(f"Input grad shape: {x.grad.shape}")
print(f"RNN weight grad: {rnn.weight_ih_l0.grad.shape}")
```

## Manual BPTT Implementation

For pedagogical clarity:

```python
def bptt_manual(xs, hs, ys, targets, W_xh, W_hh, W_hy):
    """
    Manual BPTT computation.
    
    Args:
        xs: List of inputs [x_1, ..., x_T]
        hs: List of hidden states [h_0, h_1, ..., h_T]
        ys: List of outputs [y_1, ..., y_T]
        targets: List of targets
        W_xh, W_hh, W_hy: Weight matrices
    
    Returns:
        Gradients for all parameters
    """
    T = len(xs)
    
    # Initialize gradients
    dW_xh = torch.zeros_like(W_xh)
    dW_hh = torch.zeros_like(W_hh)
    dW_hy = torch.zeros_like(W_hy)
    
    # Backward pass through time
    dh_next = torch.zeros_like(hs[0])
    
    for t in reversed(range(T)):
        # Gradient from output
        dy = ys[t] - targets[t]  # Assuming softmax + cross-entropy
        
        # Gradient for output weights
        dW_hy += torch.outer(dy, hs[t+1])
        
        # Gradient flowing to hidden state
        dh = W_hy.T @ dy + dh_next
        
        # Gradient through tanh
        dh_raw = dh * (1 - hs[t+1] ** 2)
        
        # Gradient for input weights
        dW_xh += torch.outer(dh_raw, xs[t])
        
        # Gradient for recurrent weights
        dW_hh += torch.outer(dh_raw, hs[t])
        
        # Propagate gradient to previous timestep
        dh_next = W_hh.T @ dh_raw
    
    return dW_xh, dW_hh, dW_hy
```

## Truncated BPTT

For very long sequences, full BPTT becomes computationally prohibitive. **Truncated BPTT** limits backpropagation to $k$ timesteps:

```python
def truncated_bptt(model, sequence, chunk_size, optimizer):
    """
    Truncated BPTT: backpropagate through chunks.
    """
    hidden = None
    total_loss = 0
    
    for i in range(0, len(sequence) - 1, chunk_size):
        # Get chunk
        inputs = sequence[i:i + chunk_size]
        targets = sequence[i + 1:i + chunk_size + 1]
        
        # Detach hidden state to truncate gradient flow
        if hidden is not None:
            hidden = hidden.detach()
        
        # Forward pass
        outputs, hidden = model(inputs.unsqueeze(0), hidden)
        loss = criterion(outputs.squeeze(0), targets)
        
        # Backward pass (only through this chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss
```

Truncation trades off:
- **Computational efficiency**: Reduced memory and time
- **Gradient accuracy**: Cannot learn dependencies longer than $k$ steps

## Computational Complexity

For a sequence of length $T$ with hidden size $H$:

| Operation | Time | Space |
|-----------|------|-------|
| Forward pass | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| Backward pass | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| Total BPTT | $O(T \cdot H^2)$ | $O(T \cdot H)$ |

Space complexity is dominated by storing hidden states for gradient computation.

## Gradient Flow Visualization

```python
import matplotlib.pyplot as plt

def visualize_gradient_flow(model, sequence):
    """Visualize gradient magnitudes through time."""
    gradients = []
    
    # Register hooks to capture gradients
    def hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].norm().item())
    
    handle = model.rnn.register_backward_hook(hook)
    
    # Forward and backward
    outputs, _ = model(sequence)
    loss = outputs.sum()
    loss.backward()
    
    handle.remove()
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(gradients[::-1])  # Reverse for chronological order
    plt.xlabel('Timestep')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow Through Time')
    plt.show()
```

## Summary

BPTT extends backpropagation to recurrent networks by:

1. Unrolling the RNN into a deep feedforward network
2. Computing gradients through the chain rule across timesteps
3. Accumulating gradients for shared weights

Key insights:
- Gradients must traverse $T$ timesteps to reach early inputs
- Each timestep multiplies by $W_{hh}$ and $\tanh'$
- This multiplication chain creates vanishing/exploding gradient problems

Understanding BPTT motivates LSTM and GRU architectures, which modify the gradient flow to enable learning over longer sequences.
