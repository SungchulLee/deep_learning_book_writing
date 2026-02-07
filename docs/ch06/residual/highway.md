# Highway Networks

## Overview

Highway Networks, introduced by Srivastava, Greff, and Schmidhuber in 2015, were among the first architectures to enable training of very deep networks (tens to hundreds of layers). They predate ResNet and introduce a *gating mechanism* that learns to regulate information flow through each layer, allowing the network to adaptively decide how much to transform versus pass through the input.

While ResNet uses a fixed, parameter-free identity skip connection ($y = F(x) + x$), Highway Networks learn a *data-dependent* gate that smoothly interpolates between transformation and identity:

$$y = T(x) \cdot H(x) + C(x) \cdot x$$

where $T(x)$ is the *transform gate*, $H(x)$ is the transformed representation, and $C(x)$ is the *carry gate*.

## Mathematical Formulation

### Plain Layer

A standard feedforward layer applies a nonlinear transformation:

$$y = H(x, W_H)$$

where $W_H$ are the layer's weights and $H$ typically includes an affine transformation followed by a nonlinearity.

### Highway Layer

A Highway layer adds two gating mechanisms:

$$y = T(x, W_T) \odot H(x, W_H) + C(x, W_C) \odot x$$

where:

- $H(x, W_H)$: The **transform** — a standard nonlinear layer
- $T(x, W_T) = \sigma(W_T x + b_T)$: The **transform gate** — sigmoid output in $[0, 1]$
- $C(x, W_C)$: The **carry gate** — controls how much of the input to pass through
- $\odot$: Element-wise (Hadamard) product

### Simplified Highway Layer

In practice, the carry gate is typically set as the complement of the transform gate:

$$C(x) = 1 - T(x)$$

This reduces the formulation to:

$$y = T(x) \odot H(x) + (1 - T(x)) \odot x$$

This is a *learned convex combination* of the transformation and the identity. When $T(x) \to 0$, the layer passes the input unchanged (identity). When $T(x) \to 1$, the layer applies the full transformation.

### Connection to ResNet

ResNet can be viewed as a special case of Highway Networks where the gate is fixed at $T = 1$:

$$y_{\text{ResNet}} = H(x) + x = 1 \cdot H(x) + (1 - 1) \cdot x + x$$

More precisely, ResNet learns $y = F(x) + x$ where $F$ is the residual, while Highway Networks learn $y = T(x) \cdot H(x) + (1 - T(x)) \cdot x$. The fixed identity in ResNet turns out to be simpler, more parameter-efficient, and equally effective for most tasks.

## Implementation

### Highway Linear Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HighwayLayer(nn.Module):
    """
    Highway Network layer with learned gating.
    
    y = T(x) * H(x) + (1 - T(x)) * x
    
    where T(x) = sigmoid(W_T x + b_T) is the transform gate
    and H(x) is a standard nonlinear transformation.
    
    Args:
        size: Dimension of input and output (must be equal)
        bias_init: Initial bias for the transform gate.
            Negative values (e.g., -2) bias toward carrying input,
            which helps training in very deep networks.
    """
    
    def __init__(self, size: int, bias_init: float = -2.0):
        super(HighwayLayer, self).__init__()
        
        # Transform: H(x) = ReLU(W_H x + b_H)
        self.transform = nn.Linear(size, size)
        
        # Gate: T(x) = σ(W_T x + b_T)
        self.gate = nn.Linear(size, size)
        
        # Initialize gate bias to negative value
        # This biases the network toward carrying input initially,
        # allowing gradients to flow and enabling training of deep networks
        nn.init.constant_(self.gate.bias, bias_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform
        h = F.relu(self.transform(x))
        
        # Gate
        t = torch.sigmoid(self.gate(x))
        
        # Highway output: gated combination
        return t * h + (1 - t) * x
```

### Highway Convolutional Layer

```python
class HighwayConv2d(nn.Module):
    """
    Highway Network convolutional layer.
    
    Applies spatial gating where each position independently
    decides how much to transform vs carry.
    
    Args:
        channels: Number of channels (in = out for identity path)
        kernel_size: Convolution kernel size
        bias_init: Initial bias for transform gate
    """
    
    def __init__(self, channels: int, kernel_size: int = 3,
                 bias_init: float = -2.0):
        super(HighwayConv2d, self).__init__()
        
        padding = kernel_size // 2
        
        # Transform path
        self.transform_conv = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.transform_bn = nn.BatchNorm2d(channels)
        
        # Gate path
        self.gate_conv = nn.Conv2d(
            channels, channels, kernel_size, padding=padding
        )
        
        # Initialize gate bias
        nn.init.constant_(self.gate_conv.bias, bias_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.transform_bn(self.transform_conv(x)))
        t = torch.sigmoid(self.gate_conv(x))
        return t * h + (1 - t) * x
```

### Stacked Highway Network

```python
class HighwayNetwork(nn.Module):
    """
    Deep Highway Network with multiple highway layers.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Highway layer dimension
        num_layers: Number of highway layers
        output_dim: Output dimension
        bias_init: Initial gate bias (negative = bias toward carry)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        bias_init: float = -2.0
    ):
        super(HighwayNetwork, self).__init__()
        
        # Project input to highway dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Highway layers
        self.highway_layers = nn.ModuleList([
            HighwayLayer(hidden_dim, bias_init=bias_init)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        for layer in self.highway_layers:
            x = layer(x)
        return self.output_proj(x)
```

## Gate Behavior and Analysis

### Gate Initialization

The transform gate bias initialization is critical. Setting $b_T$ to a negative value (typically $-1$ to $-3$) biases $T(x) = \sigma(W_T x + b_T)$ toward 0 at initialization, meaning the network initially acts close to identity. This ensures stable gradient flow during early training, similar to how ResNet's zero-initialized residual branches start at identity.

### Learned Gate Patterns

Srivastava et al. observed that trained Highway Networks develop interpretable gate patterns:

- **Early layers**: Gates tend to be open ($T \approx 1$), performing active transformation
- **Middle layers**: Gates show selective behavior, transforming some dimensions while carrying others
- **Later layers**: Some gates close ($T \approx 0$), effectively shortening the network for certain inputs

This adaptive depth behavior means the network learns to use different effective depths for different inputs—a form of conditional computation.

```python
def analyze_gate_values(model: HighwayNetwork, 
                        x: torch.Tensor) -> list:
    """
    Extract and analyze gate values at each highway layer.
    
    Returns list of gate activation tensors for visualization.
    """
    gate_values = []
    
    h = F.relu(model.input_proj(x))
    for layer in model.highway_layers:
        t = torch.sigmoid(layer.gate(h))
        gate_values.append(t.detach())
        h = layer(h)
    
    return gate_values


def visualize_gates(gate_values: list):
    """Visualize gate activation patterns across layers."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(gate_values), figsize=(3 * len(gate_values), 4))
    if len(gate_values) == 1:
        axes = [axes]
    
    for i, (ax, gates) in enumerate(zip(axes, gate_values)):
        # Average over batch, show per-dimension gate values
        mean_gates = gates.mean(dim=0).cpu().numpy()
        ax.bar(range(len(mean_gates)), mean_gates, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Gate Value (T)')
        ax.set_title(f'Layer {i+1}')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('highway_gates.png', dpi=150)
    plt.close()
```

## Gradient Flow Analysis

### Gradient Through Highway Layers

For the simplified highway layer $y = T \odot H + (1 - T) \odot x$, the Jacobian is:

$$\frac{\partial y}{\partial x} = T \odot \frac{\partial H}{\partial x} + (1 - T) \odot I + \frac{\partial T}{\partial x} \odot (H - x)$$

The key term is $(1 - T) \odot I$: even when the transform gradient $\frac{\partial H}{\partial x}$ vanishes, the carry path provides a gradient proportional to $(1 - T)$.

### Comparison with ResNet Gradient

| Architecture | Gradient through layer | Identity component |
|-------------|----------------------|-------------------|
| Plain | $\frac{\partial H}{\partial x}$ | None |
| ResNet | $I + \frac{\partial F}{\partial x}$ | Always $I$ |
| Highway | $(1-T) I + T \frac{\partial H}{\partial x} + \frac{\partial T}{\partial x}(H-x)$ | Modulated by $(1-T)$ |

ResNet provides a constant identity gradient regardless of the data, while Highway Networks modulate the identity gradient through the learned gate. This makes Highway gradients more flexible but also more prone to vanishing if gates saturate at $T = 1$.

### Depth Limitations

Despite gating, Highway Networks in practice struggle beyond ~100 layers. The learned gate introduces a multiplicative factor $(1-T)$ on the identity path, which can compound across layers. ResNet's fixed identity ($+ x$ rather than $(1-T) \cdot x$) provides a strictly stronger gradient guarantee.

## Highway Networks vs ResNet

| Aspect | Highway Networks | ResNet |
|--------|-----------------|--------|
| Year introduced | May 2015 | December 2015 |
| Skip connection | Learned gate | Fixed identity |
| Parameters per layer | 3× (H, T, C) or 2× (H, T) | ~1× (just H) |
| Gate mechanism | Sigmoid-modulated | None (always open) |
| Maximum practical depth | ~100 layers | 1000+ layers |
| Gradient guarantee | $(1-T)$ modulated | Constant identity |
| Adaptive depth | Yes (per-sample) | No (fixed) |
| ImageNet performance | Not competitive | State-of-the-art |

### Why ResNet Won

Despite Highway Networks' theoretical elegance, ResNet dominates in practice for several reasons:

1. **Parameter efficiency**: Highway gates double or triple parameters without proportional accuracy gains
2. **Stronger gradient flow**: Fixed identity provides a guaranteed, unattenuated gradient path
3. **Simplicity**: No gate initialization tuning required
4. **Scalability**: ResNet scales to 1000+ layers; Highway Networks plateau around 100

However, Highway Networks' contribution is foundational—they demonstrated that very deep networks *could* be trained with appropriate information flow mechanisms, directly inspiring ResNet.

## Modern Legacy: Gating in Other Architectures

While Highway Networks themselves are rarely used today, the gating principle lives on in many modern architectures:

### LSTM and GRU

The original inspiration for Highway Networks came from LSTM gates. The carry gate is analogous to the LSTM forget gate, and the transform gate parallels the input gate.

### Gated Linear Units (GLU)

$$\text{GLU}(x) = (Wx + b) \otimes \sigma(Vx + c)$$

Used in Transformer feedforward layers (e.g., PaLM, LLaMA), GLUs apply element-wise gating without a separate carry path.

### Mixture of Experts (MoE)

MoE routing can be viewed as a generalization of highway gating where the network learns to route inputs to different expert subnetworks, extending the binary transform/carry decision to a multi-way routing decision.

## Applications in Quantitative Finance

### Adaptive Feature Processing

Highway Networks' data-dependent gating is conceptually relevant for financial models where different market regimes require different levels of feature transformation:

```python
class HighwayFactorModel(nn.Module):
    """
    Highway-gated factor model for returns prediction.
    
    The gate learns which market conditions require deep 
    nonlinear processing vs simple linear factor exposure.
    In stable markets, gates may close (carry raw factors).
    In volatile/complex regimes, gates open for nonlinear processing.
    """
    
    def __init__(self, num_factors: int, hidden_dim: int = 128,
                 num_highway: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(num_factors, hidden_dim)
        self.highway = nn.ModuleList([
            HighwayLayer(hidden_dim, bias_init=-1.0)
            for _ in range(num_highway)
        ])
        self.head = nn.Linear(hidden_dim, 1)
    
    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            factors: (batch, num_factors) raw factor exposures
        Returns:
            (batch, 1) predicted returns
        """
        x = F.relu(self.input_proj(factors))
        for layer in self.highway:
            x = layer(x)
        return self.head(x)
    
    def get_gate_values(self, factors: torch.Tensor) -> list:
        """Extract gate values for regime analysis."""
        gates = []
        x = F.relu(self.input_proj(factors))
        for layer in self.highway:
            t = torch.sigmoid(layer.gate(x))
            gates.append(t.detach())
            x = layer(x)
        return gates
```

### Interpretable Gating for Risk Management

The gate values in Highway Networks provide a natural measure of model complexity per input. In a risk management context, samples where gates are mostly closed (near identity) represent "simple" market conditions, while samples with open gates indicate complex, nonlinear regimes. This interpretability can inform position sizing and risk limits.

## Summary

Highway Networks introduced the foundational principle that deep networks need mechanisms to control information flow across layers:

| Contribution | Impact |
|-------------|--------|
| **Learned gating** | First architecture to train 50–100+ layer feedforward networks |
| **Transform/carry decomposition** | Inspired the identity mapping in ResNet |
| **Adaptive depth** | Per-input computation depth, precursor to early exit and MoE |
| **LSTM-to-feedforward bridge** | Connected recurrent gating ideas to deep feedforward networks |

While ResNet's simpler, parameter-free identity skip connection ultimately proved more practical, Highway Networks' intellectual contribution—that deep networks require explicit information highways—remains central to modern deep learning architecture design.

## References

1. Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Training Very Deep Networks. *NeurIPS 2015*.
2. Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway Networks. *arXiv:1505.00387*.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
4. Zilly, J. G., Srivastava, R. K., Koutník, J., & Schmidhuber, J. (2017). Recurrent Highway Networks. *ICML 2017*.
