# Network Depth and Width

## Overview

The **depth** (number of layers) and **width** (neurons per layer) of a neural network are fundamental architectural choices that significantly impact model capacity, training dynamics, and generalization. Understanding the trade-offs between deep narrow networks and shallow wide networks is essential for effective architecture design.

## Definitions

### Depth

**Depth** refers to the number of layers in a network:

- **Shallow networks**: 1-2 hidden layers
- **Deep networks**: Many hidden layers (tens to hundreds)

$$
\text{Depth} = L \quad \text{(number of layers)}
$$

### Width

**Width** refers to the number of neurons in each layer:

$$
\text{Width of layer } l = n^{[l]}
$$

For uniform width: all hidden layers have the same number of neurons.

### Total Parameters

For a fully connected network:

$$
\text{Parameters} = \sum_{l=1}^{L} \left(n^{[l-1]} \cdot n^{[l]} + n^{[l]}\right)
$$

## Depth vs. Width Trade-offs

### Expressive Power

| Property | Wide Shallow | Deep Narrow |
|----------|--------------|-------------|
| Function class | Universal approximator | Hierarchical representations |
| Parameter efficiency | Lower | Higher |
| Feature abstraction | Single level | Multi-level |
| Typical use | Kernel approximation | Modern deep learning |

### Theoretical Results

!!! abstract "Width Result (Universal Approximation)"
    A single hidden layer with sufficient width can approximate any continuous function.
    
    Required width: potentially exponential in input dimension.

!!! abstract "Depth Result (Exponential Separation)"
    There exist functions computable by depth-$k$ networks with polynomial width that require exponential width for depth-$(k-1)$ networks.

### Practical Implications

**Deep networks** are preferred because:

1. **Parameter efficiency**: Achieve same expressivity with fewer parameters
2. **Hierarchical features**: Learn compositional representations
3. **Inductive bias**: Structure encourages meaningful feature hierarchies

## Mathematical Analysis

### Width and Approximation

For approximating a function $f$ with Lipschitz constant $L$ to accuracy $\epsilon$:

**Shallow network (1 hidden layer):**
$$
\text{Width} = O\left(\left(\frac{L}{\epsilon}\right)^d\right)
$$

where $d$ is input dimension — **exponential** in $d$!

**Deep network:**
$$
\text{Width} = O\left(\text{poly}(d) \cdot \log\frac{1}{\epsilon}\right)
$$

**Much more efficient** for high-dimensional problems.

### Depth and Compositional Functions

Consider $f(\mathbf{x}) = g_1(g_2(\ldots g_k(\mathbf{x})))$ (nested composition):

- **Depth-$k$ network**: $O(n)$ neurons per layer
- **Depth-2 network**: $O(n^k)$ neurons required

**Example:** Computing $x_1 \cdot x_2 \cdot \ldots \cdot x_n$

| Depth | Width | Total Parameters |
|-------|-------|------------------|
| $O(\log n)$ | $O(n)$ | $O(n \log n)$ |
| 2 | $O(2^n)$ | $O(n \cdot 2^n)$ |

## Empirical Observations

### Effect of Depth

```
Depth 2:  ████████░░░░░░░░░░░░ 40% accuracy
Depth 5:  ████████████████░░░░ 80% accuracy  
Depth 10: ████████████████████ 95% accuracy
Depth 50: ████████████████████ 96% accuracy (diminishing returns)
```

**Observations:**

1. Increasing depth improves performance up to a point
2. Very deep networks face optimization challenges
3. Skip connections enable training of 100+ layer networks

### Effect of Width

```
Width 32:  ████████████░░░░░░░░ 60% accuracy
Width 128: ████████████████░░░░ 80% accuracy
Width 512: ████████████████████ 92% accuracy
Width 2048:████████████████████ 93% accuracy (diminishing returns)
```

**Observations:**

1. Wider layers increase capacity
2. Diminishing returns after sufficient width
3. Wider networks are easier to optimize (more paths for gradients)

## Design Guidelines

### Rule of Thumb

1. **Start with established architectures** (ResNet, VGG patterns)
2. **Pyramid shape**: Gradually decrease width toward output
3. **Power of 2**: Use widths like 64, 128, 256 for GPU efficiency
4. **Match complexity to data**: More data → deeper/wider networks

### Common Patterns

**Classification (MNIST-scale):**
```
784 → 256 → 128 → 10
```

**Classification (ImageNet-scale):**
```
Input → [64×3] → [128×4] → [256×6] → [512×3] → Output
(ResNet-34 style)
```

**Regression:**
```
Input → 128 → 64 → 32 → 1
```

### Width-to-Depth Ratio

Recent research suggests:

- **Optimal ratio** depends on task and data
- **Very narrow deep** networks are hard to train
- **Minimum width** of 64-256 is often practical
- **Depth** should scale with problem complexity

## PyTorch Implementation

### Comparing Architectures

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def create_network(architecture):
    """Create network from architecture specification."""
    layers = []
    for i in range(len(architecture) - 1):
        layers.append(nn.Linear(architecture[i], architecture[i+1]))
        if i < len(architecture) - 2:  # No ReLU after last layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_loader, test_loader, epochs=5):
    """Train model and return final accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    train_time = time.time() - start_time
    
    return accuracy, train_time


# Load MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Define architectures to compare
architectures = {
    'Wide-Shallow': [784, 1024, 10],           # 1 hidden, wide
    'Medium': [784, 256, 128, 10],             # 2 hidden, medium
    'Deep-Narrow': [784, 64, 64, 64, 64, 10],  # 4 hidden, narrow
    'Deep-Wide': [784, 256, 256, 256, 10],     # 3 hidden, wide
}

print("Architecture Comparison on MNIST")
print("=" * 70)
print(f"{'Architecture':<20} {'Params':>10} {'Accuracy':>10} {'Time (s)':>10}")
print("-" * 70)

results = {}
for name, arch in architectures.items():
    model = create_network(arch)
    params = count_parameters(model)
    accuracy, train_time = train_and_evaluate(model, train_loader, test_loader)
    results[name] = {'params': params, 'accuracy': accuracy, 'time': train_time}
    print(f"{name:<20} {params:>10,} {accuracy:>9.2f}% {train_time:>10.1f}")

print("=" * 70)
```

### Scaling Experiment

```python
import matplotlib.pyplot as plt

def scaling_experiment():
    """Study effect of depth and width on performance."""
    
    # Fixed parameter budget experiments
    param_budget = 100000
    
    # Vary depth (adjust width to match budget)
    depths = [2, 3, 4, 5, 6, 8, 10]
    depth_results = []
    
    for d in depths:
        # Calculate width for approximately param_budget parameters
        # Simplified: assume uniform width w, then params ≈ 784*w + (d-1)*w^2 + 10*w
        # Solve for w (approximate)
        w = int((-794 + (794**2 + 4*(d-1)*param_budget)**0.5) / (2*(d-1)))
        w = max(32, min(w, 1024))  # Clamp to reasonable range
        
        arch = [784] + [w] * (d-1) + [10]
        model = create_network(arch)
        params = count_parameters(model)
        accuracy, _ = train_and_evaluate(model, train_loader, test_loader, epochs=3)
        depth_results.append((d, w, params, accuracy))
        print(f"Depth {d}, Width {w}: {params:,} params, {accuracy:.2f}%")
    
    # Vary width (fixed depth)
    fixed_depth = 3
    widths = [32, 64, 128, 256, 512, 1024]
    width_results = []
    
    for w in widths:
        arch = [784] + [w] * (fixed_depth - 1) + [10]
        model = create_network(arch)
        params = count_parameters(model)
        accuracy, _ = train_and_evaluate(model, train_loader, test_loader, epochs=3)
        width_results.append((w, params, accuracy))
        print(f"Width {w}: {params:,} params, {accuracy:.2f}%")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Depth plot
    ax1 = axes[0]
    depths_plot = [r[0] for r in depth_results]
    accs_depth = [r[3] for r in depth_results]
    ax1.plot(depths_plot, accs_depth, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Depth (number of layers)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Effect of Depth (fixed ~100K params)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Width plot
    ax2 = axes[1]
    widths_plot = [r[0] for r in width_results]
    accs_width = [r[2] for r in width_results]
    ax2.plot(widths_plot, accs_width, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Width (neurons per layer)', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Effect of Width (depth=3)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('depth_width_scaling.png', dpi=150)
    plt.show()

scaling_experiment()
```

### Deep Networks with Skip Connections

```python
class ResidualMLP(nn.Module):
    """MLP with residual connections for training very deep networks."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)  # Skip connection
        
        return self.output(x)


# Compare: Plain deep vs. Residual deep
print("\nPlain Deep vs. Residual Networks")
print("=" * 50)

# Plain deep network (will struggle to train)
plain_deep = create_network([784] + [128]*20 + [10])

# Residual network (same depth, trainable)
residual_deep = ResidualMLP(784, 128, 10, num_blocks=10)  # 20 layers total

print(f"Plain Deep:    {count_parameters(plain_deep):,} params")
print(f"Residual Deep: {count_parameters(residual_deep):,} params")

# Train both
plain_acc, _ = train_and_evaluate(plain_deep, train_loader, test_loader, epochs=5)
resid_acc, _ = train_and_evaluate(residual_deep, train_loader, test_loader, epochs=5)

print(f"\nPlain Deep Accuracy:    {plain_acc:.2f}%")
print(f"Residual Deep Accuracy: {resid_acc:.2f}%")
```

## Modern Architecture Insights

### Width as Regularization

**Observation:** Very wide networks can generalize well despite overparameterization.

**Explanation:** Wide networks have a "lazy training" regime where they behave like kernel methods (Neural Tangent Kernel theory).

### Depth and Feature Hierarchy

Deep networks learn hierarchical features:

| Depth | Features Learned |
|-------|------------------|
| Early layers | Edges, textures, simple patterns |
| Middle layers | Parts, shapes, object components |
| Late layers | Objects, scenes, abstract concepts |

### Lottery Ticket Hypothesis

**Finding:** Dense networks contain sparse subnetworks ("winning tickets") that, when trained in isolation, achieve comparable accuracy.

**Implication:** Width matters for finding good initializations, but final solutions may be much smaller.

## Key Takeaways

!!! success "Summary"
    1. **Depth enables hierarchical feature learning** and parameter efficiency
    2. **Width provides capacity** but with diminishing returns
    3. **Trade-off exists**: same parameters can be allocated to depth or width
    4. **Deep narrow networks** are more parameter-efficient but harder to train
    5. **Skip connections** enable training of very deep networks (100+ layers)
    6. **Practical guidance**: Start with proven architectures, then adjust
    7. **Modern trend**: Deeper networks with residual connections

## References

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Lu, Z., et al. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis. *ICLR*.
