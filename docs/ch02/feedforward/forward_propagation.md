# Forward Propagation

## Overview

**Forward propagation** (or forward pass) is the process of computing the output of a neural network given an input. Data flows from the input layer through hidden layers to the output layer, with each layer applying a linear transformation followed by a nonlinear activation.

## Mathematical Formulation

### Single Sample

For a network with $L$ layers, forward propagation computes:

$$
\mathbf{a}^{[0]} = \mathbf{x}
$$

For each layer $l = 1, 2, \ldots, L$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})
$$

Final output:

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]}
$$

### Batch Processing

For a batch of $B$ samples organized as columns in matrix $\mathbf{X} \in \mathbb{R}^{n^{[0]} \times B}$:

$$
\mathbf{A}^{[0]} = \mathbf{X}
$$

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{A}^{[l]} = \sigma^{[l]}(\mathbf{Z}^{[l]})
$$

!!! note "Broadcasting"
    The bias $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$ is broadcast across all $B$ samples when added to $\mathbf{Z}^{[l]}$.

## Step-by-Step Example

Consider a 2-layer network for binary classification:

- Input: $\mathbf{x} \in \mathbb{R}^2$
- Hidden layer: 4 neurons with ReLU
- Output: 1 neuron with Sigmoid

### Layer Dimensions

| Layer | Input Dim | Output Dim | $\mathbf{W}$ Shape | $\mathbf{b}$ Shape |
|-------|-----------|------------|--------------------|--------------------|
| 1 (Hidden) | 2 | 4 | $(4, 2)$ | $(4,)$ |
| 2 (Output) | 4 | 1 | $(1, 4)$ | $(1,)$ |

### Forward Pass Walkthrough

```
Input: x = [0.5, 0.8]ᵀ

Step 1: Hidden Layer Linear
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
     = [w₁₁ w₁₂] [0.5]   [b₁]
       [w₂₁ w₂₂] [0.8] + [b₂]
       [w₃₁ w₃₂]         [b₃]
       [w₄₁ w₄₂]         [b₄]

Step 2: Hidden Layer Activation
a⁽¹⁾ = ReLU(z⁽¹⁾) = max(0, z⁽¹⁾)  (element-wise)

Step 3: Output Layer Linear
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
     = [w₁ w₂ w₃ w₄] [a₁⁽¹⁾]   [b]
                      [a₂⁽¹⁾] + 
                      [a₃⁽¹⁾]
                      [a₄⁽¹⁾]

Step 4: Output Layer Activation
ŷ = a⁽²⁾ = σ(z⁽²⁾) = 1/(1 + e^(-z⁽²⁾))
```

## Computational Graph Perspective

Forward propagation builds a **computational graph** that records all operations:

```
x ──┬─→ z¹ ─→ ReLU ─→ a¹ ──┬─→ z² ─→ Sigmoid ─→ ŷ
    │                       │
   W¹                      W²
   b¹                      b²
```

Each node stores:

1. **Input values** (received from parent nodes)
2. **Output values** (computed and sent to children)
3. **Operation type** (for gradient computation)

This graph is essential for **backpropagation**—gradients flow backward through the same structure.

## Activation Functions in Forward Pass

### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

```python
def relu(z):
    return torch.maximum(z, torch.tensor(0.0))
```

**Computation:** Simple comparison, very fast.

### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

```python
def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))
```

**Computation:** Requires exponentiation.

### Softmax (for multi-class)

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

```python
def softmax(z):
    exp_z = torch.exp(z - z.max(dim=-1, keepdim=True)[0])  # Numerical stability
    return exp_z / exp_z.sum(dim=-1, keepdim=True)
```

**Computation:** Requires exponentiation and normalization.

## PyTorch Implementation

### Manual Forward Pass

```python
import torch
import torch.nn.functional as F

def forward_pass_manual(x, weights, biases):
    """
    Manual forward propagation through a neural network.
    
    Args:
        x: Input tensor, shape (batch_size, input_dim)
        weights: List of weight matrices [W1, W2, ..., WL]
        biases: List of bias vectors [b1, b2, ..., bL]
    
    Returns:
        output: Final prediction
        cache: Dictionary of intermediate values for backprop
    """
    cache = {'a0': x}
    a = x
    
    num_layers = len(weights)
    
    for l in range(num_layers):
        # Linear transformation
        z = torch.mm(a, weights[l].T) + biases[l]
        cache[f'z{l+1}'] = z
        
        # Activation
        if l < num_layers - 1:  # Hidden layers: ReLU
            a = F.relu(z)
        else:  # Output layer: Sigmoid (for binary classification)
            a = torch.sigmoid(z)
        
        cache[f'a{l+1}'] = a
    
    return a, cache


# Example usage
torch.manual_seed(42)

# Initialize parameters
input_dim, hidden_dim, output_dim = 2, 4, 1
batch_size = 3

W1 = torch.randn(hidden_dim, input_dim) * 0.5
b1 = torch.zeros(hidden_dim)
W2 = torch.randn(output_dim, hidden_dim) * 0.5
b2 = torch.zeros(output_dim)

# Sample input
x = torch.tensor([[0.5, 0.8],
                  [0.1, 0.2],
                  [0.9, 0.4]])

# Forward pass
output, cache = forward_pass_manual(x, [W1, W2], [b1, b2])

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("\nIntermediate values:")
for key, value in cache.items():
    print(f"  {key}: shape {value.shape}")
print("\nOutput (predictions):")
print(output)
```

**Output:**
```
Input shape: torch.Size([3, 2])
Output shape: torch.Size([3, 1])

Intermediate values:
  a0: shape torch.Size([3, 2])
  z1: shape torch.Size([3, 4])
  a1: shape torch.Size([3, 4])
  z2: shape torch.Size([3, 1])
  a2: shape torch.Size([3, 1])

Output (predictions):
tensor([[0.4521],
        [0.4892],
        [0.4376]])
```

### Using nn.Module

```python
import torch
import torch.nn as nn

class ForwardPassNetwork(nn.Module):
    """
    Neural network that exposes intermediate computations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, return_intermediates=False):
        """
        Forward pass with optional intermediate value tracking.
        
        Args:
            x: Input tensor
            return_intermediates: If True, return cache dictionary
        
        Returns:
            output: Final prediction
            cache (optional): Intermediate values
        """
        # Layer 1
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        
        # Layer 2
        z2 = self.fc2(a1)
        output = torch.sigmoid(z2)
        
        if return_intermediates:
            cache = {
                'z1': z1,
                'a1': a1,
                'z2': z2,
                'output': output
            }
            return output, cache
        
        return output


# Usage
model = ForwardPassNetwork(input_dim=2, hidden_dim=4, output_dim=1)
x = torch.randn(5, 2)

# Standard forward pass
output = model(x)
print("Output shape:", output.shape)

# Forward pass with intermediates (for debugging/visualization)
output, cache = model(x, return_intermediates=True)
print("\nCache keys:", cache.keys())
```

### Efficient Batch Processing

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EfficientMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Not last layer
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Create model
model = EfficientMLP([784, 256, 128, 10])

# Batch processing
batch_size = 64
x_batch = torch.randn(batch_size, 784)

# Single forward pass for entire batch
output = model(x_batch)
print(f"Batch input: {x_batch.shape}")
print(f"Batch output: {output.shape}")

# Memory-efficient forward pass with DataLoader
dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient tracking for inference
    for batch_x, batch_y in dataloader:
        output = model(batch_x)
        # Process outputs...
```

## Computational Complexity

### Time Complexity

For a single layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:

$$
\text{Time} = O(n_{\text{in}} \cdot n_{\text{out}} \cdot B)
$$

where $B$ is the batch size.

For the full network:

$$
\text{Time} = O\left(B \sum_{l=1}^{L} n^{[l-1]} \cdot n^{[l]}\right)
$$

### Memory Complexity

During forward pass, we must store:

1. **Input activations** for each layer (needed for backprop)
2. **Pre-activations** $\mathbf{z}^{[l]}$ (for gradient computation)

$$
\text{Memory} = O\left(B \sum_{l=0}^{L} n^{[l]}\right)
$$

!!! tip "Memory Optimization"
    During **inference only** (no training), intermediate values aren't needed:
    
    ```python
    with torch.no_grad():
        output = model(input)  # Much less memory!
    ```

## Numerical Stability

### Overflow in Exponentials

**Problem:** Large values in $\mathbf{z}$ can cause `exp(z)` to overflow.

**Solution:** Log-sum-exp trick for softmax:

```python
def stable_softmax(z):
    """Numerically stable softmax."""
    z_max = z.max(dim=-1, keepdim=True)[0]
    exp_z = torch.exp(z - z_max)  # Subtract max for stability
    return exp_z / exp_z.sum(dim=-1, keepdim=True)
```

### Underflow in Sigmoid

**Problem:** Very negative values give $\sigma(z) \approx 0$, causing log(0) issues.

**Solution:** Clamp values:

```python
def stable_bce(y_pred, y_true, eps=1e-7):
    """Numerically stable binary cross-entropy."""
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    return -torch.mean(y_true * torch.log(y_pred) + 
                       (1 - y_true) * torch.log(1 - y_pred))
```

## Visualization

### Tracing Values Through the Network

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TrackedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        
    def forward(self, x):
        self.activations = {'input': x}
        
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        self.activations['hidden1_pre'] = z1
        self.activations['hidden1_post'] = a1
        
        z2 = self.fc2(a1)
        a2 = torch.relu(z2)
        self.activations['hidden2_pre'] = z2
        self.activations['hidden2_post'] = a2
        
        z3 = self.fc3(a2)
        output = torch.sigmoid(z3)
        self.activations['output_pre'] = z3
        self.activations['output'] = output
        
        return output


# Create and run
model = TrackedMLP()
x = torch.randn(100, 2)
output = model(x)

# Visualize activation distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (name, values) in enumerate(model.activations.items()):
    if idx >= len(axes):
        break
    ax = axes[idx]
    values_flat = values.detach().numpy().flatten()
    ax.hist(values_flat, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'{name}\nmean={values_flat.mean():.3f}, std={values_flat.std():.3f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('forward_pass_activations.png', dpi=150)
plt.show()
```

## Key Takeaways

!!! success "Summary"
    1. **Forward propagation** computes network output by passing data through layers sequentially
    2. **Each layer** applies: linear transformation → nonlinear activation
    3. **Batch processing** is efficient: matrix operations parallelize across samples
    4. **Caching intermediate values** is essential for backpropagation
    5. **Numerical stability** requires careful handling of exponentials and logarithms
    6. **Memory usage** scales with batch size and network width

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.4.
- PyTorch Documentation: [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
