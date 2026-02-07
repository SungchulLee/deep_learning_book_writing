# MLP Architecture

## Learning Objectives

!!! abstract "What You Will Learn"
    - Define the mathematical structure of a Multi-Layer Perceptron (MLP) from single neuron to full network
    - Distinguish the roles of input, hidden, and output layers and their design considerations
    - Derive parameter counts for arbitrary architectures and understand their scaling behavior
    - Implement configurable MLPs in PyTorch using both `nn.Module` and `nn.Sequential`
    - Apply architectural design principles for classification and regression tasks

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| Linear algebra (matrix multiplication) | Layer computation is affine transformation |
| Activation functions (Ch 4.1) | Nonlinear activations are applied after each linear layer |
| Basic PyTorch tensors | Implementation exercises use PyTorch |

---

## Overview

The **Multi-Layer Perceptron (MLP)**, also known as a feedforward neural network, is the foundational architecture in deep learning. An MLP consists of an input layer, one or more hidden layers, and an output layer, with each layer **fully connected** (dense) to the next. Despite its simplicity, the MLP serves as the building block for understanding all modern neural architectures.

The term "feedforward" emphasizes that information flows in one direction — from input to output — with no cycles or feedback loops. This distinguishes MLPs from recurrent networks (Ch 7).

---

## Mathematical Formulation

### Single Neuron

A single neuron computes a weighted sum of inputs followed by a nonlinear activation:

$$
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b
$$

$$
a = \sigma(z)
$$

where:

- $\mathbf{x} \in \mathbb{R}^n$ is the input vector
- $\mathbf{w} \in \mathbb{R}^n$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\sigma(\cdot)$ is the activation function
- $z \in \mathbb{R}$ is the **pre-activation** (linear combination)
- $a \in \mathbb{R}$ is the **post-activation** output

The neuron can be understood geometrically: $\mathbf{w}^\top \mathbf{x} + b = 0$ defines a hyperplane in $\mathbb{R}^n$, and the activation function determines how the neuron responds to inputs on each side of this hyperplane.

### Layer Computation

For a layer $l$ with $n^{[l]}$ neurons receiving input from $n^{[l-1]}$ neurons, the computation is a **vectorized** version of the single neuron:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right)
$$

where:

- $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$ is the weight matrix for layer $l$
- $\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}$ is the bias vector
- $\mathbf{a}^{[l-1]} \in \mathbb{R}^{n^{[l-1]}}$ is the activation from the previous layer
- $\mathbf{z}^{[l]} \in \mathbb{R}^{n^{[l]}}$ is the pre-activation vector
- $\mathbf{a}^{[l]} \in \mathbb{R}^{n^{[l]}}$ is the activation output

Each row $\mathbf{W}^{[l]}_{j,:}$ of the weight matrix contains the weights for neuron $j$ in layer $l$. The matrix-vector product computes all $n^{[l]}$ neurons simultaneously.

### Full Network

For an $L$-layer network, the forward computation is a **composition of affine transformations and nonlinearities**:

$$
\mathbf{a}^{[0]} = \mathbf{x} \quad \text{(input)}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \quad l = 1, \ldots, L
$$

$$
\mathbf{a}^{[l]} = \sigma^{[l]}\!\left(\mathbf{z}^{[l]}\right), \quad l = 1, \ldots, L
$$

$$
\hat{\mathbf{y}} = \mathbf{a}^{[L]} \quad \text{(output)}
$$

Compactly, the entire network is a function composition:

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \sigma^{[L]} \circ g^{[L]} \circ \sigma^{[L-1]} \circ g^{[L-1]} \circ \cdots \circ \sigma^{[1]} \circ g^{[1]}(\mathbf{x})
$$

where $g^{[l]}(\mathbf{a}) = \mathbf{W}^{[l]} \mathbf{a} + \mathbf{b}^{[l]}$ is the affine map at layer $l$ and $\boldsymbol{\theta} = \{\mathbf{W}^{[l]}, \mathbf{b}^{[l]}\}_{l=1}^L$ denotes all learnable parameters.

!!! warning "Why Nonlinearity Is Essential"
    Without activation functions, the composition of affine maps is itself affine:
    
    $$
    \mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = \underbrace{(\mathbf{W}^{[2]}\mathbf{W}^{[1]})}_{\mathbf{W}'}\mathbf{x} + \underbrace{(\mathbf{W}^{[2]}\mathbf{b}^{[1]} + \mathbf{b}^{[2]})}_{\mathbf{b}'}
    $$
    
    No matter how many layers we stack, the network can only represent linear functions. The activation function $\sigma$ is what gives depth its power.

---

## Architecture Components

### Input Layer

The input layer receives raw features and passes them to the first hidden layer. It performs no computation — it simply holds the input values $\mathbf{a}^{[0]} = \mathbf{x}$.

**Design considerations:**

- Number of neurons equals the number of input features: $n^{[0]} = \dim(\mathbf{x})$
- For images: $n^{[0]} = H \times W \times C$ (flattened), e.g., MNIST gives $n^{[0]} = 28 \times 28 = 784$
- For tabular data: $n^{[0]}$ = number of features after preprocessing

### Hidden Layers

Hidden layers perform the bulk of computation and feature extraction. Each hidden layer applies a two-step transformation:

1. **Affine map** (linear transformation + bias): $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
2. **Nonlinear activation**: $\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$

The purpose of hidden layers is to learn increasingly abstract, hierarchical representations of the input data. Early layers might detect low-level features (edges, simple patterns), while deeper layers combine these into higher-level abstractions.

### Output Layer

The output layer produces the final predictions. Its design depends on the task:

| Task | $n^{[L]}$ | Activation $\sigma^{[L]}$ | Loss Function |
|------|-----------|--------------------------|---------------|
| Binary classification | 1 | Sigmoid | Binary cross-entropy |
| Multi-class classification | $K$ (classes) | Softmax | Categorical cross-entropy |
| Regression | 1 (or $d$) | Identity (none) | MSE / MAE |
| Multi-label classification | $K$ | Sigmoid (each) | Binary cross-entropy (each) |

!!! tip "Softmax and Cross-Entropy in PyTorch"
    `nn.CrossEntropyLoss` in PyTorch applies log-softmax internally for numerical stability. When using it, the output layer should produce **raw logits** (no softmax activation). This avoids computing softmax twice and prevents log-of-softmax numerical issues.

---

## Network Notation

### Standard Notation

For a network with layers $l = 0, 1, \ldots, L$:

| Symbol | Description | Shape |
|--------|-------------|-------|
| $n^{[l]}$ | Number of neurons in layer $l$ | scalar |
| $\mathbf{W}^{[l]}$ | Weight matrix for layer $l$ | $(n^{[l]}, n^{[l-1]})$ |
| $\mathbf{b}^{[l]}$ | Bias vector for layer $l$ | $(n^{[l]},)$ |
| $\mathbf{z}^{[l]}$ | Pre-activation at layer $l$ | $(n^{[l]},)$ |
| $\mathbf{a}^{[l]}$ | Post-activation at layer $l$ | $(n^{[l]},)$ |
| $\sigma^{[l]}$ | Activation function for layer $l$ | — |

### Batch Processing

For a mini-batch of $B$ samples, every vector becomes a matrix. Using the convention where each **column** is a sample:

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \mathbf{1}_B^\top
$$

where:

- $\mathbf{A}^{[l-1]} \in \mathbb{R}^{n^{[l-1]} \times B}$ — each column is one sample's activation
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times B}$
- $\mathbf{1}_B^\top$ broadcasts the bias across all $B$ samples

!!! note "PyTorch Convention"
    PyTorch uses **row-major** format where each **row** is a sample:
    
    - Input shape: $(B, n^{[0]})$
    - Weight matrix in `nn.Linear`: shape $(n^{[l]}, n^{[l-1]})$
    - Computation: `output = input @ weight.T + bias`, giving shape $(B, n^{[l]})$
    
    This is the transpose of the mathematical convention above. Both are equivalent; the row-major format is more natural for GPU memory layout.

---

## Parameter Count

### Derivation

Each layer $l$ has:

- **Weights:** $n^{[l]} \times n^{[l-1]}$ parameters (one per connection)
- **Biases:** $n^{[l]}$ parameters (one per neuron)

Total parameters for the entire network:

$$
|\boldsymbol{\theta}| = \sum_{l=1}^{L} \left( n^{[l]} \cdot n^{[l-1]} + n^{[l]} \right) = \sum_{l=1}^{L} n^{[l]}\!\left(n^{[l-1]} + 1\right)
$$

### Scaling Behavior

For a uniform-width network ($n^{[l]} = d$ for all hidden layers) with $L$ layers:

$$
|\boldsymbol{\theta}| = n^{[0]} \cdot d + (L-2) \cdot d^2 + d \cdot n^{[L]} + L \cdot d
$$

The dominant term is $(L-2) \cdot d^2$, so parameters scale **quadratically** with width and **linearly** with depth.

### Worked Example

Network with architecture $[784, 256, 128, 10]$:

| Layer $l$ | $n^{[l-1]} \to n^{[l]}$ | Weights | Biases | Total |
|-----------|--------------------------|---------|--------|-------|
| 1 | $784 \to 256$ | $256 \times 784 = 200{,}704$ | $256$ | $200{,}960$ |
| 2 | $256 \to 128$ | $128 \times 256 = 32{,}768$ | $128$ | $32{,}896$ |
| 3 | $128 \to 10$ | $10 \times 128 = 1{,}280$ | $10$ | $1{,}290$ |
| **Total** | | $234{,}752$ | $394$ | **$235{,}146$** |

Layer 1 alone accounts for $200{,}960 / 235{,}146 \approx 85\%$ of all parameters — a consequence of the large input dimension.

---

## PyTorch Implementation

### Configurable MLP with `nn.Module`

```python
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer widths, e.g. [256, 128]
        output_size: Number of output neurons
        activation: Activation function class (default: nn.ReLU)
        output_activation: Output activation (None for logits)
        dropout: Dropout probability between hidden layers (0 = no dropout)
    """
    def __init__(
        self, 
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: type = nn.ReLU,
        output_activation: type | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers: list[nn.Module] = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Example: MNIST multi-class classification ──
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    activation=nn.ReLU,
    output_activation=None,   # raw logits for CrossEntropyLoss
    dropout=0.2,
)

print(f"Architecture: 784 → 256 → 128 → 10")
print(f"Total parameters: {model.count_parameters():,}")

# Verify shapes
x = torch.randn(32, 784)     # batch of 32
logits = model(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {logits.shape}")
```

**Output:**
```
Architecture: 784 → 256 → 128 → 10
Total parameters: 235,146
Input shape:  torch.Size([32, 784])
Output shape: torch.Size([32, 10])
```

### Using `nn.Sequential` Directly

For simple architectures, `nn.Sequential` avoids defining a class:

```python
# Quick MLP definition
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
)

# Inspect layer-by-layer shapes
x = torch.randn(1, 784)
for i, layer in enumerate(model):
    x = layer(x)
    print(f"Layer {i} ({layer.__class__.__name__:>10s}): {x.shape}")
```

**Output:**
```
Layer 0 (    Linear): torch.Size([1, 256])
Layer 1 (      ReLU): torch.Size([1, 256])
Layer 2 (   Dropout): torch.Size([1, 256])
Layer 3 (    Linear): torch.Size([1, 128])
Layer 4 (      ReLU): torch.Size([1, 128])
Layer 5 (   Dropout): torch.Size([1, 128])
Layer 6 (    Linear): torch.Size([1, 10])
```

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Hyperparameters ──
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

# ── Data ──
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean, std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── Model ──
model = MLP(
    input_size=784,
    hidden_sizes=[512, 256],
    output_size=10,
    activation=nn.ReLU,
    dropout=0.2,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── Training loop ──
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for data, target in train_loader:
        data = data.view(data.size(0), -1)   # flatten 28×28 → 784
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    
    # ── Evaluation ──
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            test_correct += output.argmax(dim=1).eq(target).sum().item()
            test_total += data.size(0)
    
    print(
        f"Epoch {epoch+1:2d}/{EPOCHS} | "
        f"Train Loss: {total_loss/total:.4f} | "
        f"Train Acc: {100*correct/total:.2f}% | "
        f"Test Acc: {100*test_correct/test_total:.2f}%"
    )
```

---

## Architectural Design Principles

### Layer Sizing Guidelines

1. **Start simple.** Begin with one or two hidden layers; add complexity only when underfitting.
2. **Pyramid (funnel) structure.** Gradually decrease layer widths toward the output — this forces the network to compress information: e.g., $784 \to 256 \to 128 \to 10$.
3. **Powers of 2.** Use widths like 64, 128, 256, 512 for GPU memory alignment efficiency.
4. **Bottleneck awareness.** The narrowest hidden layer bounds the information flow; ensure it is wide enough for the task.

### Common Architecture Patterns

```
Small dataset:    Input → 128 → 64 → Output
Medium dataset:   Input → 256 → 128 → 64 → Output
Large dataset:    Input → 512 → 256 → 128 → Output
```

### When NOT to Use a Plain MLP

MLPs are fully connected, meaning every neuron connects to every neuron in the adjacent layer. This has limitations:

- **Images:** No spatial structure is exploited → prefer CNNs (Ch 5)
- **Sequences:** No temporal structure → prefer RNNs/Transformers (Ch 7-8)
- **Graphs:** No relational structure → prefer GNNs

The MLP remains an excellent choice for tabular data and as a component (e.g., classification head) within larger architectures.

---

## Key Takeaways

!!! success "Summary"
    1. **MLPs are fully connected feedforward networks** with input, hidden, and output layers
    2. **Each layer performs** an affine transformation followed by a nonlinear activation: $\mathbf{a}^{[l]} = \sigma(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$
    3. **The activation function is essential** — without it, any depth of network collapses to a single affine map
    4. **Parameter count** scales as $O(d^2 L)$ for uniform-width networks, dominated by the widest layer connections
    5. **Output layer design** depends on the task: sigmoid for binary, softmax for multi-class (via `CrossEntropyLoss`), identity for regression
    6. **PyTorch `nn.Linear`** uses row-major convention: input $(B, n_\text{in})$, output $(B, n_\text{out})$

---

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 5.
