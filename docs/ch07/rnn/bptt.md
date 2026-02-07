# Backpropagation Through Time

## Introduction

Training recurrent neural networks requires computing gradients through the unrolled computational graph—a procedure called **Backpropagation Through Time (BPTT)**. Understanding BPTT illuminates both how RNNs learn and why they struggle with long sequences.

## The Unrolled Computational Graph

Consider an RNN processing a sequence of length $T$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
$$y_t = W_{hy} h_t$$

Unrolling creates a deep feedforward network where depth equals the sequence length $T$, weights are shared across all layers, and each "layer" corresponds to one timestep:

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

For per-step classification with cross-entropy:

$$L_t = -\sum_{c} \hat{y}_{t,c} \log(y_{t,c})$$

For sequence classification using only the final output:

$$\mathcal{L} = L_T(y_T, \hat{y})$$

## Gradient Derivation

### Output Weights

The gradient $\frac{\partial \mathcal{L}}{\partial W_{hy}}$ accumulates contributions from all timesteps:

$$\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot \frac{\partial y_t}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial y_t} \cdot h_t^T$$

Since $y_t = W_{hy} h_t$, the derivative is straightforward—no temporal chain rule is needed because $W_{hy}$ acts locally at each timestep.

### Hidden State Gradients

The gradient at hidden state $h_t$ receives contributions from two sources: the local loss at time $t$, and the gradient flowing back from future timesteps through $h_{t+1}$:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial \mathcal{L}}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}$$

This recursive formula is the heart of BPTT—it propagates gradients backward through time.

### Recurrent Weights

The gradient $\frac{\partial \mathcal{L}}{\partial W_{hh}}$ requires the full chain rule through time because $h_t$ depends on $W_{hh}$ both directly and through all previous hidden states:

$$\frac{\partial L_T}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t} \cdot \frac{\partial^+ h_t}{\partial W_{hh}}$$

where $\frac{\partial^+ h_t}{\partial W_{hh}}$ denotes the immediate (non-recursive) partial derivative—the contribution from the direct use of $W_{hh}$ at timestep $t$ only.

## The Jacobian Product

The key term $\frac{\partial h_T}{\partial h_t}$ expands as a product of Jacobians:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Each Jacobian is:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(1 - h_{k+1}^2) \cdot W_{hh}$$

where $\text{diag}(1 - h_{k+1}^2)$ is the derivative of $\tanh$ evaluated at the pre-activation. This product of matrices over $T - t$ timesteps is what makes or breaks gradient flow: when $T - t$ is large, the product tends to either vanish or explode.

## PyTorch Automatic BPTT

PyTorch handles BPTT automatically through its autograd engine. The forward pass builds the computational graph, and `.backward()` performs BPTT:

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

# Loss and backward pass (BPTT happens automatically)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
loss.backward()

# Gradients are now available
print(f"Input grad shape: {x.grad.shape}")
print(f"RNN weight grad: {rnn.weight_ih_l0.grad.shape}")
```

## Manual BPTT Implementation

For pedagogical clarity, here is an explicit BPTT computation:

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
    
    dW_xh = torch.zeros_like(W_xh)
    dW_hh = torch.zeros_like(W_hh)
    dW_hy = torch.zeros_like(W_hy)
    
    # Backward pass through time
    dh_next = torch.zeros_like(hs[0])
    
    for t in reversed(range(T)):
        # Gradient from output loss
        dy = ys[t] - targets[t]  # Softmax + cross-entropy derivative
        
        # Output weight gradient
        dW_hy += torch.outer(dy, hs[t + 1])
        
        # Gradient flowing into hidden state
        dh = W_hy.T @ dy + dh_next
        
        # Gradient through tanh nonlinearity
        dh_raw = dh * (1 - hs[t + 1] ** 2)
        
        # Input and recurrent weight gradients
        dW_xh += torch.outer(dh_raw, xs[t])
        dW_hh += torch.outer(dh_raw, hs[t])
        
        # Propagate to previous timestep
        dh_next = W_hh.T @ dh_raw
    
    return dW_xh, dW_hh, dW_hy
```

The loop proceeds in reverse time order. At each step, the gradient `dh_next` carries information from all future timesteps back through the recurrence—this is the temporal gradient flow that suffers from vanishing and exploding problems.

## Truncated BPTT

For very long sequences, full BPTT becomes computationally prohibitive because memory scales linearly with sequence length (all hidden states must be stored). **Truncated BPTT** limits backpropagation to $k$ timesteps:

```python
def truncated_bptt(model, sequence, chunk_size, optimizer, criterion):
    """
    Truncated BPTT: process sequence in chunks, backpropagate
    only within each chunk.
    """
    hidden = None
    total_loss = 0
    
    for i in range(0, len(sequence) - 1, chunk_size):
        inputs = sequence[i:i + chunk_size]
        targets = sequence[i + 1:i + chunk_size + 1]
        
        # Detach hidden state to truncate gradient flow
        if hidden is not None:
            hidden = hidden.detach()
        
        # Forward pass through chunk
        outputs, hidden = model(inputs.unsqueeze(0), hidden)
        loss = criterion(outputs.squeeze(0), targets)
        
        # Backward pass (only through this chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss
```

The `.detach()` call is the mechanism that truncates gradient flow: while the hidden state value carries forward from chunk to chunk, gradients cannot propagate back through the detachment point. This trades gradient accuracy for computational efficiency—dependencies longer than $k$ steps cannot be learned.

## Computational Complexity

For a sequence of length $T$ with hidden size $H$:

| Operation | Time | Space |
|-----------|------|-------|
| Forward pass | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| Backward pass | $O(T \cdot H^2)$ | $O(T \cdot H)$ |
| Truncated BPTT (chunk $k$) | $O(T \cdot H^2)$ | $O(k \cdot H)$ |

Space complexity is dominated by storing hidden states at every timestep for gradient computation. Truncated BPTT reduces the space requirement from $O(T \cdot H)$ to $O(k \cdot H)$.

## Gradient Flow Visualization

```python
import matplotlib.pyplot as plt


def visualize_gradient_flow(model, x):
    """
    Visualize gradient magnitudes from final output to each input position.
    """
    model.train()
    x = x.requires_grad_(True)
    
    outputs, _ = model(x)
    loss = outputs[0, -1, :].sum()
    loss.backward()
    
    grad_norms = x.grad[0].norm(dim=-1).detach().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(grad_norms)), grad_norms)
    plt.xlabel('Timestep')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Flow Through Time')
    plt.yscale('log')
    plt.show()
```

## Summary

BPTT extends backpropagation to recurrent networks by unrolling the RNN into a deep feedforward network, computing gradients through the chain rule across timesteps, and accumulating gradients for shared weights. The critical insight is that gradients must traverse $T$ timesteps to reach early inputs, with each step multiplying by $W_{hh}$ and $\tanh'$. This multiplication chain creates the vanishing and exploding gradient problems examined in the next sections.
