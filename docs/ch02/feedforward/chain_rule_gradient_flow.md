# Chain Rule and Gradient Flow

## Overview

The **chain rule** is the mathematical foundation that enables gradient computation in deep neural networks. Understanding how gradients flow through network layers is crucial for designing architectures that train effectively and avoiding pathological behaviors like vanishing or exploding gradients.

## The Chain Rule

### Scalar Form

For a composite function $f(g(x))$:

$$
\frac{d}{dx}f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### Multivariate Chain Rule

For $f: \mathbb{R}^m \to \mathbb{R}$ and $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\frac{\partial f}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial f}{\partial g_j} \cdot \frac{\partial g_j}{\partial x_i}
$$

In matrix notation:

$$
\nabla_\mathbf{x} f = \mathbf{J}_\mathbf{g}^T \nabla_\mathbf{g} f
$$

### Chain Rule for Deep Networks

For a network with composition $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{J}_{f_1}^T \mathbf{J}_{f_2}^T \cdots \mathbf{J}_{f_L}^T \nabla_{\mathbf{y}} \mathcal{L}
$$

This **product of Jacobians** is the mathematical core of backpropagation.

## Gradient Flow Through Layers

### Linear Layer

For $\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \mathbf{a}^T, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \mathbf{W}^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}}
$$

### Activation Functions

| Activation | Forward | Backward |
|------------|---------|----------|
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ |
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $a(1-a)$ |
| Tanh | $\tanh(z)$ | $1 - a^2$ |

## Vanishing Gradient Problem

### The Issue

When gradients flow through many layers with saturating activations:

$$
\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}}\right\| \leq \prod_{l=2}^{L} \|\mathbf{J}^{[l]}\| \cdot \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}}\right\|
$$

With sigmoid ($\max |\sigma'| = 0.25$), for $L=20$ layers: $0.25^{19} \approx 10^{-12}$.

### Solutions

1. **ReLU activation**: Gradient = 1 for positive inputs
2. **Proper initialization**: Xavier (tanh/sigmoid), He/Kaiming (ReLU)
3. **Batch normalization**: Keeps activations in non-saturated regime
4. **Residual connections**: Direct gradient path via identity

## Exploding Gradient Problem

### Symptoms

- Loss becomes `NaN` or `Inf`
- Training instability

### Solution: Gradient Clipping

```python
# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

## PyTorch Implementation

### Monitoring Gradient Flow

```python
import torch
import torch.nn as nn

def check_gradient_flow(model):
    """Monitor gradient magnitudes after backward pass."""
    grad_info = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append({
                'name': name,
                'norm': grad_norm,
                'mean': param.grad.mean().item(),
                'max': param.grad.abs().max().item()
            })
    
    print("Gradient Statistics:")
    print("-" * 60)
    for info in grad_info:
        status = "⚠️" if info['norm'] < 1e-7 or info['norm'] > 1e3 else "✓"
        print(f"{status} {info['name']:30s} norm={info['norm']:.2e}")
    
    return grad_info


# Example usage
class DeepNet(nn.Module):
    def __init__(self, depth=10):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(64, 64), nn.ReLU()])
        layers.append(nn.Linear(64, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# Test gradient flow
model = DeepNet(depth=10)
x = torch.randn(32, 64)
y = torch.randn(32, 1)

output = model(x)
loss = nn.MSELoss()(output, y)
loss.backward()

check_gradient_flow(model)
```

### Comparing Activations

```python
import matplotlib.pyplot as plt

def compare_activations():
    """Compare gradient flow with different activations."""
    results = {}
    
    for activation in ['relu', 'sigmoid', 'tanh']:
        # Build network
        layers = []
        for _ in range(10):
            layers.append(nn.Linear(64, 64))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Tanh())
        layers.append(nn.Linear(64, 1))
        model = nn.Sequential(*layers)
        
        # Forward/backward
        x = torch.randn(32, 64)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        # Collect gradient norms
        norms = []
        for i, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.grad.norm().item())
        
        results[activation] = norms
    
    # Plot
    plt.figure(figsize=(10, 6))
    for act, norms in results.items():
        plt.semilogy(range(1, len(norms)+1), norms, 'o-', label=act)
    
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title('Gradient Flow Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gradient_flow.png', dpi=150)
    plt.show()
```

## Residual Connections

### Gradient Flow with Skip Connections

For $\mathbf{a}^{[l]} = f(\mathbf{z}^{[l]}) + \mathbf{a}^{[l-2]}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l-2]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l]}} \left(\frac{\partial f}{\partial \mathbf{a}^{[l-2]}} + \mathbf{I}\right)
$$

The identity term $\mathbf{I}$ ensures gradients can flow directly, even if $\frac{\partial f}{\partial \mathbf{a}}$ vanishes.

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection!
```

## Key Takeaways

!!! success "Summary"
    1. **Chain rule** enables computing gradients through composed functions
    2. **Gradient magnitude** can shrink (vanish) or grow (explode) with depth
    3. **ReLU** prevents saturation but can cause "dead neurons"
    4. **Proper initialization** keeps gradients in reasonable range
    5. **Residual connections** provide direct gradient paths
    6. **Gradient clipping** prevents explosion during training
    7. **Monitor gradients** to diagnose training issues

## References

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- He, K., et al. (2015). Delving deep into rectifiers. *ICCV*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
