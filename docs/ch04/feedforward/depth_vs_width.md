# Depth vs. Width

## Learning Objectives

!!! abstract "What You Will Learn"
    - Quantify the expressive power trade-off between wide-shallow and deep-narrow architectures
    - Prove exponential separation results: functions efficiently representable by deep networks require exponential width in shallow ones
    - Analyze how depth enables hierarchical feature composition and parameter efficiency
    - Understand the practical trade-offs: trainability, optimization landscape, and gradient flow
    - Compare architectures empirically using controlled PyTorch experiments

## Prerequisites

| Topic | Why It Matters |
|-------|---------------|
| MLP Architecture (§4.2.1) | Defines the networks being compared |
| Universal Approximation (§4.2.2) | Width-based approximation is the baseline |
| Gradient Flow (§4.2.6) | Deep networks face vanishing/exploding gradient issues |

---

## Overview

The **depth** (number of layers) and **width** (neurons per layer) of a neural network are the two fundamental axes of capacity. The Universal Approximation Theorem (§4.2.2) guarantees that width alone suffices — but at what cost? This section examines the mathematical and empirical evidence that depth provides exponentially more efficient representations for many function classes, while also introducing optimization challenges that width does not.

---

## Definitions and Parameter Budget

### Depth and Width

For a network with layers $l = 0, 1, \ldots, L$:

$$
\text{Depth} = L \quad \text{(number of parameterized layers)}
$$

$$
\text{Width of layer } l = n^{[l]} \quad \text{(neurons in layer } l\text{)}
$$

For a **uniform-width** network, all hidden layers share the same width $d$: $n^{[l]} = d$ for $l = 1, \ldots, L-1$.

### Total Parameter Count

For a fully connected network with architecture $(n^{[0]}, n^{[1]}, \ldots, n^{[L]})$:

$$
|\boldsymbol{\theta}| = \sum_{l=1}^{L} n^{[l]} \left(n^{[l-1]} + 1\right)
$$

For a uniform-width network with input dimension $n^{[0]}$, hidden width $d$, output dimension $n^{[L]}$, and $L$ layers total:

$$
|\boldsymbol{\theta}| = d(n^{[0]} + 1) + (L-2) \cdot d(d+1) + n^{[L]}(d+1)
$$

The key observation: depth adds $O(d^2)$ parameters per additional layer, while width adds $O(d \cdot L)$ parameters per additional neuron. For fair comparisons, we fix the total parameter budget.

---

## Theoretical Analysis

### Expressive Power: Width

The Universal Approximation Theorem states that a **single-hidden-layer** network can approximate any continuous function. However, the required width may be exponential.

For a Lipschitz-$L$ function $f: [0,1]^n \to \mathbb{R}$, to achieve $\varepsilon$-accuracy with a depth-2 network:

$$
N_{\text{width}} = O\!\left(\left(\frac{L}{\varepsilon}\right)^n\right)
$$

This **exponential dependence on dimension** $n$ is the curse of dimensionality for shallow networks.

### Expressive Power: Depth

Deep networks can represent certain functions with **polynomially** many parameters where shallow networks require exponentially many.

!!! abstract "Exponential Separation (Telgarsky, 2016)"
    For any positive integer $k$, there exists a function $f_k: [0,1] \to [0,1]$ that:
    
    1. Can be computed by a ReLU network with $O(k)$ layers and $O(1)$ neurons per layer (total $O(k)$ parameters)
    2. Cannot be $\frac{1}{3}$-approximated by any ReLU network with $O(k^{1/3})$ layers unless it has $\Omega(2^{k^{1/3}})$ neurons

Telgarsky's construction uses the "sawtooth" function — iterated compositions of the triangle wave — which has $2^k$ oscillations but only requires $O(k)$ layers of width 2. A shallow network needs one neuron per oscillation.

### Compositional Functions

Many real-world functions have **compositional structure**:

$$
f(\mathbf{x}) = g_1 \circ g_2 \circ \cdots \circ g_k(\mathbf{x})
$$

For such functions, deep networks have a structural advantage.

**Canonical example: product function** $f(x_1, \ldots, x_n) = \prod_{i=1}^n x_i$

A deep network computes this via a binary tree of pairwise products:

| Property | Deep network | Shallow network |
|----------|-------------|-----------------|
| Depth | $O(\log n)$ | $2$ |
| Width | $O(n)$ | $\Omega(2^n)$ |
| Parameters | $O(n \log n)$ | $O(n \cdot 2^n)$ |

Each pairwise product $xy$ can be implemented exactly using ReLU neurons via:

$$
xy = \frac{1}{4}\!\left[(x+y)^2 - (x-y)^2\right]
$$

where $z^2 \approx$ sum of ReLU hinges to arbitrary precision.

### Linear Regions of ReLU Networks

A ReLU network partitions input space into **linear regions** — convex polytopes within which the network is affine. The maximum number of linear regions is a precise measure of expressivity.

!!! abstract "Linear Region Count"
    A ReLU network with input dimension $n$, $L$ hidden layers, each of width $d$, computes a piecewise linear function with at most
    
    $$
    R(n, d, L) \leq \left(\prod_{l=1}^{L-1} \left\lfloor \frac{d}{n} \right\rfloor^n \right) \cdot \sum_{j=0}^{n} \binom{d}{j}
    $$
    
    linear regions. The key observation: this grows **exponentially in depth** but only **polynomially in width**.

For fixed total neurons $N = dL$, allocating them as many narrow layers produces **exponentially more** linear regions than a single wide layer:

| Architecture | Linear regions (1D) |
|-------------|-------------------|
| Depth 1, width $N$ | $N + 1$ |
| Depth $L$, width $N/L$ | $O\!\left((N/L)^L\right)$ |

---

## Practical Trade-offs

### Depth: Benefits and Costs

**Benefits of depth:**

1. **Parameter efficiency.** Same representational power with fewer parameters (exponential separation)
2. **Hierarchical features.** Each layer builds on the previous one, enabling multi-level abstraction:
    - Early layers: edges, textures, simple patterns
    - Middle layers: parts, shapes, object components
    - Late layers: objects, scenes, abstract concepts
3. **Compositional inductive bias.** Many real-world functions have compositional structure that depth naturally captures

**Costs of depth:**

1. **Vanishing gradients.** Signal attenuates exponentially through many layers (§4.2.6)
2. **Optimization difficulty.** Deeper loss landscapes have more saddle points and narrow valleys
3. **Training instability.** Requires careful initialization (He/Xavier) and often batch normalization

### Width: Benefits and Costs

**Benefits of width:**

1. **Easier optimization.** Wider networks have smoother loss landscapes (Li et al., 2018)
2. **Lazy training.** Very wide networks converge to kernel methods (Neural Tangent Kernel), providing theoretical guarantees
3. **No vanishing gradients.** Only two layers of multiplicative gradient decay in a shallow network

**Costs of width:**

1. **Parameter inefficiency.** Exponentially more parameters needed for the same expressivity
2. **No hierarchy.** Single-level feature extraction — cannot build compositional representations
3. **Memory.** Parameter count scales as $O(d^2)$ per pair of adjacent layers

### The Optimization-Expressivity Trade-off

There is a fundamental tension:

$$
\underbrace{\text{More depth}}_{\text{more expressive}} \quad \longleftrightarrow \quad \underbrace{\text{Harder optimization}}_{\text{vanishing gradients, saddle points}}
$$

Modern architectures resolve this via **skip connections** (residual networks), which provide direct gradient paths that bypass the vanishing gradient problem while preserving the expressivity benefits of depth.

---

## PyTorch Experiments

### Controlled Architecture Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


def create_mlp(architecture: list[int]) -> nn.Sequential:
    """Create a plain MLP from a list of layer sizes."""
    layers = []
    for i in range(len(architecture) - 1):
        layers.append(nn.Linear(architecture[i], architecture[i + 1]))
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
) -> tuple[float, float]:
    """Train model, return (test_accuracy, wall_time)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    wall_time = time.time() - t0
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            preds = model(data).argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total, wall_time


# ── Data ──
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256)

# ── Architectures: same approximate parameter budget ──
architectures = {
    'Wide-Shallow (1 hidden)':   [784, 1024, 10],
    'Medium (2 hidden)':         [784, 256, 128, 10],
    'Deep-Narrow (4 hidden)':    [784, 64, 64, 64, 64, 10],
    'Deep-Wide (3 hidden)':      [784, 256, 256, 256, 10],
}

print("Architecture Comparison on MNIST")
print("=" * 75)
print(f"{'Name':<28s} {'Params':>10s} {'Accuracy':>10s} {'Time (s)':>10s}")
print("-" * 75)

for name, arch in architectures.items():
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)
    acc, t = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    print(f"{name:<28s} {params:>10,d} {acc:>9.2f}% {t:>10.1f}")

print("=" * 75)
```

### Fixed Budget: Depth vs. Width Scaling

```python
import matplotlib.pyplot as plt


def compute_uniform_width(input_dim, output_dim, depth, budget):
    """Find width d such that total params ≈ budget for given depth."""
    # params = d*(input_dim+1) + (depth-2)*d*(d+1) + output_dim*(d+1)
    # Solve quadratic in d for depth >= 3
    if depth == 2:
        # params = input_dim * d + d + output_dim * d + output_dim
        d = (budget - output_dim) // (input_dim + 1 + output_dim)
        return max(16, d)
    a = depth - 2
    b = input_dim + 1 + (depth - 2) + output_dim
    c = output_dim - budget
    d = int((-b + (b**2 - 4*a*c)**0.5) / (2*a))
    return max(16, d)


param_budget = 100_000
depths = [2, 3, 4, 5, 6, 8, 10]
results = []

print(f"\nFixed budget ≈ {param_budget:,} parameters")
print(f"{'Depth':<8s} {'Width':<8s} {'Params':<12s} {'Accuracy':<10s}")
print("-" * 40)

for L in depths:
    d = compute_uniform_width(784, 10, L, param_budget)
    arch = [784] + [d] * (L - 1) + [10]
    
    torch.manual_seed(42)
    model = create_mlp(arch)
    params = count_params(model)
    acc, _ = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    results.append((L, d, params, acc))
    print(f"{L:<8d} {d:<8d} {params:<12,d} {acc:<10.2f}%")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r[0] for r in results], [r[3] for r in results], 'bo-', lw=2, ms=8)
ax.set_xlabel('Depth (number of layers)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'Effect of Depth (fixed ≈{param_budget:,} params)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_vs_width_fixed_budget.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Deep Networks with Residual Connections

```python
class ResidualBlock(nn.Module):
    """Two-layer residual block: x + F(x)."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x):
        return torch.relu(x + self.block(x))


class ResidualMLP(nn.Module):
    """MLP with residual connections for training very deep networks."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# ── Compare plain deep vs. residual deep ──
torch.manual_seed(42)
plain_deep = create_mlp([784] + [128] * 20 + [10])      # 20 hidden layers, no skip
residual   = ResidualMLP(784, 128, 10, num_blocks=10)    # 20 layers via 10 blocks

print(f"Plain deep:    {count_params(plain_deep):>10,d} params")
print(f"Residual deep: {count_params(residual):>10,d} params")

plain_acc, _  = train_and_evaluate(plain_deep, train_loader, test_loader, epochs=5)
resid_acc, _  = train_and_evaluate(residual,   train_loader, test_loader, epochs=5)

print(f"\nPlain deep accuracy:    {plain_acc:.2f}%")
print(f"Residual deep accuracy: {resid_acc:.2f}%")
```

---

## Modern Insights

### Width and the Neural Tangent Kernel

In the infinite-width limit, neural networks converge to a **kernel method** called the Neural Tangent Kernel (NTK). In this "lazy training" regime, the network parameters barely change from initialization, and training dynamics become linear. This provides theoretical guarantees (global convergence, generalization bounds) but sacrifices the feature-learning advantage of finite-width networks.

### The Lottery Ticket Hypothesis

Frankle & Carlin (2019) observed that dense networks contain sparse **subnetworks** ("winning tickets") that, when trained in isolation from the same initialization, achieve comparable accuracy. This suggests that width matters primarily for finding good initializations, while the final effective network may be much smaller.

### Practical Recommendations

1. **Start with proven architectures** (e.g., ResNet patterns for depth, or standard MLP widths for tabular data)
2. **Funnel (pyramid) shape** for MLPs: gradually decreasing width toward the output
3. **Use powers of 2** for width (64, 128, 256, 512) for GPU memory alignment
4. **Add residual connections** for any network deeper than ~5 layers
5. **Scale depth with problem complexity**, but ensure adequate gradient flow

---

## Key Takeaways

!!! success "Summary"
    1. **Depth enables hierarchical feature learning** and provides exponential parameter efficiency over width for compositional functions
    2. **Width provides capacity** and easier optimization, but with diminishing returns and no compositional structure
    3. **Exponential separation** is proven: there exist functions requiring $\Omega(2^k)$ width at depth 2 but only $O(k)$ parameters at depth $O(k)$
    4. **ReLU linear regions** grow exponentially with depth but only polynomially with width
    5. **Depth introduces optimization challenges** (vanishing gradients, saddle points) resolved by residual connections and normalization
    6. **Fixed parameter budget:** moderate depth with moderate width typically outperforms extremes
    7. **Modern practice:** deep networks with residual connections, batch normalization, and careful initialization

---

## References

- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Montufar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. *NeurIPS*.
- Lu, Z., Pu, H., Wang, F., Hu, Z., & Wang, L. (2017). The expressive power of neural networks: A view from the width. *NeurIPS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *NeurIPS*.
- Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *ICLR*.
