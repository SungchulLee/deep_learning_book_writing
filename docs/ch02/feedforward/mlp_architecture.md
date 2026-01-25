# MLP Architecture

## Overview

The **Multi-Layer Perceptron (MLP)**, also known as a feedforward neural network, is the foundational architecture in deep learning. An MLP consists of an input layer, one or more hidden layers, and an output layer, with each layer fully connected to the next.

## Mathematical Formulation

### Single Neuron

A single neuron computes a weighted sum of inputs followed by a nonlinear activation:

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b
$$

$$
a = \sigma(z)
$$

where:

- $\mathbf{x} \in \mathbb{R}^n$ is the input vector
- $\mathbf{w} \in \mathbb{R}^n$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\sigma(\cdot)$ is the activation function
- $z$ is the pre-activation (linear combination)
- $a$ is the post-activation output

### Layer Computation

For a layer with $m$ neurons receiving input from $n$ neurons:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})
$$

where:

- $\mathbf{W}^{[l]} \in \mathbb{R}^{m \times n}$ is the weight matrix for layer $l$
- $\mathbf{b}^{[l]} \in \mathbb{R}^m$ is the bias vector
- $\mathbf{a}^{[l-1]} \in \mathbb{R}^n$ is the activation from the previous layer
- $\mathbf{z}^{[l]} \in \mathbb{R}^m$ is the pre-activation vector
- $\mathbf{a}^{[l]} \in \mathbb{R}^m$ is the activation output

### Full Network

For an $L$-layer network:

$$
\mathbf{a}^{[0]} = \mathbf{x} \quad \text{(input)}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad l = 1, \ldots, L
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]}), \quad l = 1, \ldots, L
$$

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]} \quad \text{(output)}
$$

## Architecture Components

### Input Layer

The input layer receives raw features and passes them to the first hidden layer. It performs no computation—it simply holds the input values.

**Design considerations:**

- Number of neurons equals the number of input features
- For images: $n = H \times W \times C$ (flattened)
- For tabular data: $n$ = number of features

### Hidden Layers

Hidden layers perform the bulk of computation and feature extraction. Each hidden layer:

1. Applies a linear transformation (affine map)
2. Applies a nonlinear activation function

**Purpose of hidden layers:**

- Learn hierarchical representations
- Extract increasingly abstract features
- Enable modeling of complex, nonlinear relationships

### Output Layer

The output layer produces the final predictions. Its design depends on the task:

| Task | Output Neurons | Activation | Loss Function |
|------|----------------|------------|---------------|
| Binary Classification | 1 | Sigmoid | BCE |
| Multi-class Classification | $K$ (number of classes) | Softmax | Cross-Entropy |
| Regression | 1 (or $d$ for multi-output) | None (Linear) | MSE |
| Multi-label Classification | $K$ | Sigmoid (each) | BCE (each) |

## Network Notation

### Standard Notation

For a network with layers $l = 0, 1, \ldots, L$:

| Symbol | Description |
|--------|-------------|
| $n^{[l]}$ | Number of neurons in layer $l$ |
| $\mathbf{W}^{[l]}$ | Weight matrix, shape $(n^{[l]}, n^{[l-1]})$ |
| $\mathbf{b}^{[l]}$ | Bias vector, shape $(n^{[l]},)$ |
| $\mathbf{z}^{[l]}$ | Pre-activation, shape $(n^{[l]},)$ |
| $\mathbf{a}^{[l]}$ | Activation, shape $(n^{[l]},)$ |
| $\sigma^{[l]}$ | Activation function for layer $l$ |

### Batch Processing

For a batch of $B$ samples:

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}
$$

where:

- $\mathbf{A}^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times B}$ (each column is a sample)
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times B}$

!!! note "PyTorch Convention"
    PyTorch uses row-major format where each **row** is a sample:
    
    - Input shape: $(B, n^{[0]})$
    - Weight shape: $(n^{[l]}, n^{[l-1]})$ for `nn.Linear`
    - Computation: `output = input @ weight.T + bias`

## Parameter Count

Total parameters in a network:

$$
\text{Total} = \sum_{l=1}^{L} \left( n^{[l]} \times n^{[l-1]} + n^{[l]} \right)
$$

**Example:** Network with architecture [784, 256, 128, 10]:

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| 1 | $256 \times 784 = 200,704$ | 256 | 200,960 |
| 2 | $128 \times 256 = 32,768$ | 128 | 32,896 |
| 3 | $10 \times 128 = 1,280$ | 10 | 1,290 |
| **Total** | 234,752 | 394 | **235,146** |

## PyTorch Implementation

### Basic MLP with nn.Module

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output neurons
        activation: Activation function class (default: nn.ReLU)
        output_activation: Output activation (None for regression)
    """
    def __init__(
        self, 
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = None
    ):
        super().__init__()
        
        # Build layer list
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example: Binary classification on MNIST
model = MLP(
    input_size=784,          # 28x28 flattened
    hidden_sizes=[256, 128], # Two hidden layers
    output_size=1,           # Binary output
    activation=nn.ReLU,
    output_activation=nn.Sigmoid
)

print(f"Architecture: 784 -> 256 -> 128 -> 1")
print(f"Total parameters: {model.count_parameters():,}")

# Example forward pass
x = torch.randn(32, 784)  # Batch of 32 samples
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

**Output:**
```
Architecture: 784 -> 256 -> 128 -> 1
Total parameters: 235,137
Input shape: torch.Size([32, 784])
Output shape: torch.Size([32, 1])
```

### Using nn.Sequential Directly

```python
# Quick MLP definition for simple cases
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

# Named layers for better readability
model = nn.Sequential(
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(128, 10)),
    ('softmax', nn.Softmax(dim=1))
)
```

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model for multi-class classification
model = MLP(
    input_size=784,
    hidden_sizes=[512, 256],
    output_size=10,  # 10 digits
    activation=nn.ReLU,
    output_activation=None  # Use CrossEntropyLoss which includes softmax
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten images
        data = data.view(data.size(0), -1)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
```

## Architectural Design Principles

### Rule of Thumb for Hidden Layer Sizes

1. **Start simple:** Begin with one hidden layer
2. **Pyramid structure:** Gradually decrease layer sizes
3. **Power of 2:** Use sizes like 64, 128, 256, 512 for GPU efficiency
4. **Input-output interpolation:** Hidden sizes between input and output dimensions

### Common Patterns

**Classification (small dataset):**
```
Input -> 128 -> 64 -> Output
```

**Classification (medium dataset):**
```
Input -> 256 -> 128 -> 64 -> Output
```

**Deep network with skip connections:**
```
Input -> 512 -> 256 -> 256 -> 128 -> Output
           \__________↗     \______↗
```

## Key Takeaways

!!! success "Summary"
    1. **MLPs are fully connected networks** with input, hidden, and output layers
    2. **Each layer performs** a linear transformation followed by nonlinear activation
    3. **The activation function is crucial** — without it, the network collapses to a single linear transformation
    4. **Parameter count scales quadratically** with layer width
    5. **Architecture choice depends on the task**: classification vs. regression, data size, complexity

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
