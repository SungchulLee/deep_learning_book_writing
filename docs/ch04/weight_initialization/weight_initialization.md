# Weight Initialization

## Overview

Before the first gradient update, a neural network's weights fully determine its forward activations and backward gradients. If the initial weight magnitudes are too large, activations and gradients explode exponentially with depth; if too small, they vanish. In both cases training either diverges or stalls. Weight initialization strategies set the variance of each layer's parameters so that signal magnitude remains approximately constant across layers during the initial forward and backward passes.

This section derives the variance conditions from first principles, motivates the two dominant schemes — Xavier (Glorot) and He (Kaiming) — and provides practical guidance for choosing between them.

## Learning Objectives

By the end of this section, you will understand:

1. Why naive random initialization fails for deep networks
2. The variance propagation analysis for forward and backward passes
3. How activation function choice determines the correct variance formula
4. The relationship between initialization and normalization layers
5. PyTorch's built-in initialization utilities and when to use them

---

## The Initialization Problem

### Signal Propagation in Deep Networks

Consider a feedforward network with $L$ layers. At layer $l$, the pre-activation is:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$

where $h^{(l-1)}$ is the activation from the previous layer (with $h^{(0)} = x$, the input). The activation is:

$$h^{(l)} = f\bigl(z^{(l)}\bigr)$$

for some nonlinearity $f$.

If we initialise weights from $W_{ij}^{(l)} \sim \mathcal{N}(0, \sigma^2)$ with biases at zero, the variance of a single pre-activation unit (assuming inputs are i.i.d. with zero mean and independence from weights) is:

$$\text{Var}\bigl(z_j^{(l)}\bigr) = n_{l-1}\,\sigma^2\,\text{Var}\bigl(h^{(l-1)}\bigr)$$

where $n_{l-1}$ is the fan-in (number of input units to layer $l$).

### Exploding and Vanishing Activations

Across $L$ layers, the variance compounds multiplicatively:

$$\text{Var}\bigl(z^{(L)}\bigr) \propto \prod_{l=1}^{L} \bigl(n_{l-1}\,\sigma^2 \cdot c_l\bigr) \cdot \text{Var}(x)$$

where $c_l$ accounts for the activation function's effect on variance. If $n_{l-1}\,\sigma^2\,c_l > 1$ at each layer, variance grows exponentially; if $< 1$, it decays exponentially. For a 50-layer network, even a 10% deviation per layer produces a factor of $(1.1)^{50} \approx 117$ or $(0.9)^{50} \approx 0.005$.

### Gradient Propagation

The same analysis applies in reverse. During backpropagation, the gradient with respect to layer $l$'s pre-activation involves:

$$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \bigl(W^{(l+1)}\bigr)^\top \frac{\partial \mathcal{L}}{\partial z^{(l+1)}} \odot f'\bigl(z^{(l)}\bigr)$$

The variance of the gradient signal depends on $n_{l+1}\,\sigma^2$ (the fan-out) and the derivative of the activation function. For stable backpropagation, we need:

$$n_{l+1}\,\sigma^2\,\mathbb{E}\bigl[f'(z)^2\bigr] \approx 1$$

### The Core Design Principle

Proper initialization requires choosing $\sigma^2$ so that both conditions are approximately satisfied:

$$n_{\text{in}}\,\sigma^2\,c_{\text{fwd}} \approx 1 \qquad \text{(forward stability)}$$

$$n_{\text{out}}\,\sigma^2\,c_{\text{bwd}} \approx 1 \qquad \text{(backward stability)}$$

where $c_{\text{fwd}}$ and $c_{\text{bwd}}$ depend on the activation function.

## Naive Initialization: What Goes Wrong

### Constant Initialization

Setting all weights to the same value (including zero) is catastrophic: every neuron computes the same output, receives the same gradient, and updates identically. The network is effectively a single neuron per layer regardless of width, a failure known as the **symmetry problem**.

### Standard Normal ($\sigma = 1$)

For a layer with $n_{\text{in}} = 512$, the pre-activation variance is $512 \cdot 1 \cdot \text{Var}(h) = 512\,\text{Var}(h)$. After a few layers, activations overflow `float32` range.

### Too-Small Variance ($\sigma = 0.001$)

Pre-activation variance is $512 \times 10^{-6}\,\text{Var}(h) \approx 5 \times 10^{-4}\,\text{Var}(h)$. Activations collapse to near zero; gradients vanish.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def visualize_initialization_problem(n_layers=20, hidden_dim=256, n_samples=512):
    """Show how different init scales affect activation magnitudes."""
    x = torch.randn(n_samples, hidden_dim)

    scales = {'Too small (0.001)': 0.001, 'Too large (1.0)': 1.0, 'Proper (Xavier)': (2.0 / (hidden_dim + hidden_dim)) ** 0.5}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, scale) in zip(axes, scales.items()):
        h = x.clone()
        means, stds = [], []

        for _ in range(n_layers):
            W = torch.randn(hidden_dim, hidden_dim) * scale
            h = torch.tanh(h @ W)
            means.append(h.mean().item())
            stds.append(h.std().item())

        ax.plot(stds, 'b-', linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Std')
        ax.set_title(label)
        ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('init_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


visualize_initialization_problem()
```

## Variance Analysis Framework

### Assumptions

The derivations below use the following standard assumptions:

1. Weights $W_{ij}^{(l)}$ are i.i.d. with zero mean and variance $\sigma_l^2$
2. Biases are initialised to zero
3. Inputs to each layer have zero mean (approximately true after centering or normalization)
4. Weights and activations are independent (strictly true only at initialization)

These assumptions hold exactly at the first forward pass and provide a good approximation for the first several gradient steps.

### Forward Pass Variance

For a single pre-activation unit at layer $l$:

$$z_j^{(l)} = \sum_{i=1}^{n_{l-1}} W_{ij}^{(l)}\,h_i^{(l-1)}$$

Taking the variance:

$$\text{Var}(z_j^{(l)}) = n_{l-1}\,\text{Var}(W_{ij}^{(l)})\,\mathbb{E}\bigl[(h_i^{(l-1)})^2\bigr]$$

This uses $\text{Var}(XY) = \text{Var}(X)\,\text{Var}(Y) + \text{Var}(X)\,[\mathbb{E}(Y)]^2 + \text{Var}(Y)\,[\mathbb{E}(X)]^2$, which simplifies to $\text{Var}(X)\,\mathbb{E}(Y^2)$ when $\mathbb{E}(X) = 0$.

For the activation $h^{(l)} = f(z^{(l)})$, we need $\mathbb{E}[f(z)^2]$ in terms of $\text{Var}(z)$. This factor depends on the activation function and is where Xavier and He initialization diverge.

### Backward Pass Variance

By analogous reasoning on the gradient signal:

$$\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial h_i^{(l-1)}}\right) = n_l\,\text{Var}(W_{ij}^{(l)})\,\mathbb{E}\bigl[f'(z_j^{(l)})^2\bigr]\,\text{Var}\!\left(\frac{\partial \mathcal{L}}{\partial z_j^{(l)}}\right)$$

Forward and backward stability generally cannot be achieved simultaneously unless $n_{\text{in}} = n_{\text{out}}$, so practical schemes compromise between the two.

## PyTorch Initialization Utilities

PyTorch provides initialization functions in `torch.nn.init`:

```python
import torch.nn as nn
import torch.nn.init as init

linear = nn.Linear(256, 128)

# Xavier (Glorot) initialization
init.xavier_uniform_(linear.weight)
init.xavier_normal_(linear.weight)

# He (Kaiming) initialization
init.kaiming_uniform_(linear.weight, nonlinearity='relu')
init.kaiming_normal_(linear.weight, nonlinearity='relu')

# Other schemes
init.orthogonal_(linear.weight)          # preserves norms exactly
init.sparse_(linear.weight, sparsity=0.1)  # sparse initialization

# Bias initialization (typically zero)
init.zeros_(linear.bias)
```

### Custom Initialization for a Full Network

```python
def init_weights(module):
    """Apply He initialization to all linear and conv layers."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        init.ones_(module.weight)
        init.zeros_(module.bias)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
model.apply(init_weights)
```

## Initialization Selection Guide

| Activation Function | Recommended Init | Variance Formula |
|---------------------|-----------------|-----------------|
| Sigmoid, Tanh | Xavier (Glorot) | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU | He (Kaiming) | $\sigma^2 = \frac{2}{n_{\text{in}}}$ |
| Leaky ReLU (slope $a$) | He (modified) | $\sigma^2 = \frac{2}{(1 + a^2)\,n_{\text{in}}}$ |
| SELU | LeCun Normal | $\sigma^2 = \frac{1}{n_{\text{in}}}$ |
| GELU, Swish, Mish | He or Xavier | Empirically robust to either |

### Practical Rules

1. **ReLU networks without normalization**: He initialization is strongly preferred. Xavier causes activation collapse in deep ReLU networks.
2. **Networks with BatchNorm or LayerNorm**: Initialization matters less because normalization re-scales activations at every layer. Xavier or He both work.
3. **Transformers**: Often use Xavier or scaled normal $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ with special treatment for residual connections (scaling by $1/\sqrt{2L}$).
4. **Residual networks**: The final layer in each residual block is sometimes initialised to zero so that the block initially computes the identity.

## Interaction with Normalization

Normalization layers (BatchNorm, LayerNorm) enforce fixed statistics at each layer boundary, strongly mitigating the signal propagation problem. With normalization:

- The network is less sensitive to initialization variance — the normalization "corrects" the scale at every layer.
- Training still starts faster with proper initialization because the first few steps produce more informative gradients before running statistics have stabilised.
- For very deep networks (100+ layers), combining He initialization with normalization remains best practice.

Without normalization, proper initialization is **critical** — the network may not train at all otherwise.

## Quantitative Finance Application

In quantitative finance, initialization has particular relevance:

**Online learning and warm-starting.** In production systems that update model weights in real time (e.g., online market-making models), proper initialization of new layers or expanded model components determines how quickly the model adapts to regime changes.

**Transfer learning for limited data.** When fine-tuning a pre-trained model on a small financial dataset, the initialization of new task-specific heads affects both convergence speed and the risk of catastrophic forgetting of the pre-trained representations.

**Ensemble diversity.** When building model ensembles for risk management, different random initializations produce diverse learned functions. The quality of this diversity depends on the initialization distribution spanning distinct basins of attraction in the loss landscape.

```python
# Example: Initializing a pricing network with output constraints
class PricingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, 1)

        # He init for ReLU backbone
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.zeros_(m.bias)

        # Small init for output head — predictions start near zero
        init.xavier_normal_(self.output_head.weight)
        init.zeros_(self.output_head.bias)

    def forward(self, x):
        h = self.backbone(x)
        return torch.softplus(self.output_head(h))  # enforce positivity
```

## Summary

| Aspect | Key Insight |
|--------|-------------|
| **Core problem** | Weight variance determines whether activations and gradients remain in a usable range |
| **Forward condition** | $n_{\text{in}} \sigma^2 c_{\text{fwd}} \approx 1$ preserves activation variance |
| **Backward condition** | $n_{\text{out}} \sigma^2 c_{\text{bwd}} \approx 1$ preserves gradient variance |
| **Xavier** | Compromises between forward and backward for symmetric activations |
| **He** | Accounts for ReLU zeroing half the distribution |
| **With normalization** | Initialization is less critical but still beneficial |

## References

1. Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV*.
3. Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." *ICLR*.
4. Mishkin, D., & Matas, J. (2016). "All You Need is a Good Init." *ICLR*.
