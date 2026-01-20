# Flow Composition

## Introduction

A single invertible transformation is rarely expressive enough to model complex distributions. The power of normalizing flows comes from **composing** multiple simple transformations into a deep, expressive model. This document covers the theory and practice of flow composition.

## The Composition Principle

### Chaining Transformations

Given invertible functions $f_1, f_2, \ldots, f_K$, their composition is:

$$f = f_K \circ f_{K-1} \circ \cdots \circ f_1$$

Applied step by step:
$$z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} z_2 \xrightarrow{} \cdots \xrightarrow{f_K} z_K = x$$

### Properties Preserved Under Composition

If each $f_k$ is invertible with tractable Jacobian, then $f$ is also:

1. **Invertible**: $(f_K \circ \cdots \circ f_1)^{-1} = f_1^{-1} \circ \cdots \circ f_K^{-1}$
2. **Tractable Jacobian**: $\det J_f = \prod_{k=1}^K \det J_{f_k}$

### Log-Likelihood Decomposition

$$\log p(x) = \log p_Z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|$$

Each layer contributes its own log-determinant term.

## Implementation

### Basic Flow Sequence

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class FlowSequence(nn.Module):
    """Compose multiple flow layers."""
    
    def __init__(self, flows: List[nn.Module], base_distribution):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.base_dist = base_distribution
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: base distribution → data distribution.
        
        z_0 → f_1 → z_1 → f_2 → ... → f_K → x
        
        Returns:
            x: Transformed samples
            log_det: Sum of log-determinants
        """
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        for flow in self.flows:
            z, log_det = flow.forward(z)
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass: data distribution → base distribution.
        
        x → f_K^{-1} → z_{K-1} → ... → f_1^{-1} → z_0
        
        Returns:
            z: Latent samples
            log_det: Sum of log-determinants (for inverse)
        """
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        
        # Reverse order for inverse
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) using change of variables."""
        z, log_det = self.inverse(x)
        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the flow."""
        z = self.base_dist.sample(n_samples, device)
        x, _ = self.forward(z)
        return x
```

### Tracking Intermediate States

For debugging and visualization:

```python
class FlowSequenceWithHistory(FlowSequence):
    """Flow sequence that records intermediate states."""
    
    def forward_with_history(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning all intermediate states."""
        history = [z.clone()]
        
        for flow in self.flows:
            z, _ = flow.forward(z)
            history.append(z.clone())
        
        return z, history
    
    def inverse_with_history(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Inverse pass returning all intermediate states."""
        history = [x.clone()]
        
        for flow in reversed(self.flows):
            x, _ = flow.inverse(x)
            history.append(x.clone())
        
        return x, history
```

## Design Patterns for Composition

### Pattern 1: Homogeneous Stacking

Same layer type repeated multiple times:

```python
def build_planar_flow(dim: int, n_layers: int) -> FlowSequence:
    """Stack multiple planar flow layers."""
    flows = [PlanarFlow(dim) for _ in range(n_layers)]
    base_dist = GaussianBase(dim)
    return FlowSequence(flows, base_dist)
```

**Pros**: Simple, uniform architecture
**Cons**: May need many layers for expressiveness

### Pattern 2: Alternating Coupling

Alternate which dimensions are transformed:

```python
def build_realnvp(dim: int, n_layers: int, hidden_dim: int = 64) -> FlowSequence:
    """Build RealNVP with alternating masks."""
    flows = []
    
    for i in range(n_layers):
        # Create alternating mask
        mask = torch.zeros(dim)
        if i % 2 == 0:
            mask[dim // 2:] = 1  # Transform second half
        else:
            mask[:dim // 2] = 1  # Transform first half
        
        flows.append(AffineCoupling(dim, hidden_dim, mask))
    
    base_dist = GaussianBase(dim)
    return FlowSequence(flows, base_dist)
```

**Rationale**: Each layer only transforms half the dimensions. Alternating ensures all dimensions get transformed.

### Pattern 3: Coupling + Normalization

Interleave coupling layers with normalization:

```python
def build_glow_block(dim: int, hidden_dim: int = 64) -> List[nn.Module]:
    """Single Glow block: ActNorm + 1x1 Conv + Coupling."""
    return [
        ActNorm(dim),
        Invertible1x1Conv(dim),
        AffineCoupling(dim, hidden_dim)
    ]


def build_glow(dim: int, n_blocks: int, hidden_dim: int = 64) -> FlowSequence:
    """Build Glow model."""
    flows = []
    for _ in range(n_blocks):
        flows.extend(build_glow_block(dim, hidden_dim))
    
    base_dist = GaussianBase(dim)
    return FlowSequence(flows, base_dist)
```

### Pattern 4: Multi-Scale Architecture

Process at different resolutions, factoring out dimensions:

```python
class MultiScaleFlow(nn.Module):
    """Multi-scale flow with dimension factoring."""
    
    def __init__(self, input_shape, n_scales: int = 3, n_blocks_per_scale: int = 4):
        super().__init__()
        
        self.scales = nn.ModuleList()
        self.n_scales = n_scales
        
        c, h, w = input_shape
        
        for scale in range(n_scales):
            # Squeeze: (c, h, w) → (4c, h/2, w/2)
            # Then apply flow blocks
            scale_flows = []
            
            for _ in range(n_blocks_per_scale):
                scale_flows.extend([
                    ActNorm(c * 4),
                    Invertible1x1Conv(c * 4),
                    AffineCoupling(c * 4, hidden_dim=256)
                ])
            
            self.scales.append(nn.ModuleList(scale_flows))
            
            # Update dimensions for next scale
            c = c * 2  # Half channels factored out
            h, w = h // 2, w // 2
    
    def forward(self, x):
        """Multi-scale forward pass."""
        log_det_total = 0
        outputs = []  # Factored out dimensions
        
        for scale_idx, scale_flows in enumerate(self.scales):
            # Squeeze operation
            x = self.squeeze(x)
            
            # Apply flow blocks
            for flow in scale_flows:
                x, log_det = flow.forward(x)
                log_det_total += log_det
            
            # Factor out half the channels (except last scale)
            if scale_idx < self.n_scales - 1:
                x, x_out = x.chunk(2, dim=1)
                outputs.append(x_out)
        
        outputs.append(x)
        return outputs, log_det_total
    
    def squeeze(self, x):
        """Squeeze operation: trade spatial for channel dimensions."""
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return x
```

## Expressiveness and Depth

### Universal Approximation

Sufficiently deep flows can approximate any distribution:

**Theorem** (informal): For any continuous target density $p^*(x)$ and error $\epsilon > 0$, there exists a flow $f$ such that the KL divergence $D_{KL}(p^* \| p_f) < \epsilon$.

### Depth vs. Width Trade-off

| Aspect | Deeper (more layers) | Wider (larger hidden dims) |
|--------|---------------------|---------------------------|
| Parameters | Linear growth | Quadratic growth |
| Expressiveness | Compositional | Per-layer capacity |
| Training | Gradient flow issues | More stable |
| Inference | Sequential | Parallelizable |

**Rule of thumb**: Start with moderate depth (8-16 layers) and increase if needed.

### Typical Configurations

| Application | Layers | Hidden Dim | Architecture |
|-------------|--------|------------|--------------|
| 2D toy data | 4-8 | 64 | Planar or Coupling |
| MNIST | 8-16 | 256 | RealNVP |
| CIFAR-10 | 32 | 512 | Glow (multi-scale) |
| High-res images | 48+ | 512 | Glow (multi-scale) |

## Gradient Flow in Deep Compositions

### The Challenge

In a $K$-layer flow, gradients must flow through all layers:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial z_K} \cdot \frac{\partial z_K}{\partial z_{K-1}} \cdots \frac{\partial z_2}{\partial z_1} \cdot \frac{\partial z_1}{\partial \theta_1}$$

Can suffer from vanishing/exploding gradients like deep networks.

### Solutions

**1. ActNorm (Activation Normalization)**
```python
class ActNorm(nn.Module):
    """Data-dependent initialization normalization layer."""
    
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False
    
    def initialize(self, x):
        """Initialize to normalize first batch to zero mean, unit variance."""
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std + 1e-6)
        self.initialized = True
    
    def forward(self, z):
        if not self.initialized:
            self.initialize(z)
        
        x = (z + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(z.shape[0])
        return x, log_det
```

**2. Residual Connections (where applicable)**
```python
class ResidualFlow(nn.Module):
    """Flow with residual structure: f(z) = z + g(z)."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        # Scale output to ensure invertibility
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z):
        g = self.scale * self.net(z)
        x = z + g
        
        # Log det requires careful computation (not triangular)
        # Use Hutchinson trace estimator or matrix det lemma
        log_det = self._compute_log_det(z, g)
        
        return x, log_det
```

**3. Careful Initialization**
```python
def initialize_flow_near_identity(flow):
    """Initialize all layers to be close to identity."""
    for name, param in flow.named_parameters():
        if 'log_scale' in name or 'log_s' in name:
            nn.init.zeros_(param)  # scale = 1
        elif 'bias' in name or 'shift' in name:
            nn.init.zeros_(param)  # no shift
        elif 'weight' in name:
            nn.init.normal_(param, 0, 0.01)  # small weights
```

## Computational Considerations

### Memory in Deep Flows

For $K$ layers, naive implementation stores all intermediate activations:
- Memory: $O(K \cdot \text{batch} \cdot \text{dim})$

**Gradient checkpointing** trades compute for memory:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientFlow(FlowSequence):
    """Flow with gradient checkpointing."""
    
    def forward(self, z):
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        for flow in self.flows:
            # Checkpoint: don't save activations, recompute in backward
            z, log_det = checkpoint(flow.forward, z)
            log_det_sum += log_det
        
        return z, log_det_sum
```

### Parallelization

Forward/inverse passes are inherently sequential across layers. But within each layer:
- Batch dimension is parallelizable
- Some operations (e.g., 1x1 conv) are highly parallel

## Summary

Flow composition principles:

1. **Compose invertible layers** → invertible model
2. **Log-det adds across layers** → efficient likelihood
3. **Alternate masks** in coupling → all dimensions transform
4. **Interleave normalization** → stable training
5. **Multi-scale** for images → computational efficiency

Key design choices:
- Number of layers (depth)
- Type of layers (coupling, autoregressive, etc.)
- Mask patterns
- Normalization strategy
- Width of conditioning networks

## References

1. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
2. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
