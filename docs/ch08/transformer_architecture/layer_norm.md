# Layer Normalization in Transformers

## Overview

Layer normalization is a critical component of the Transformer architecture, applied after every sub-layer (self-attention and feed-forward network) to stabilize training and enable deep stacking. The placement of normalization—before or after the sub-layer—has significant implications for training dynamics, and modern architectures have converged on alternatives like RMSNorm for efficiency.

## Layer Normalization

### Mathematical Formulation

Layer normalization normalizes across the feature dimension for each token independently:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ is the mean across features
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ is the variance across features
- $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters
- $\epsilon$ is a small constant for numerical stability (typically $10^{-5}$ or $10^{-6}$)

### Why Layer Normalization (Not Batch Normalization)

Batch normalization normalizes across the batch dimension, computing statistics over all tokens in a mini-batch for each feature. This is problematic for Transformers because:

1. **Variable sequence lengths**: Different sequences in a batch have different lengths, making batch statistics meaningless for padded positions.
2. **Autoregressive generation**: During inference, the model processes one token at a time (batch size 1), making batch statistics unavailable.
3. **Sequence dependence**: Token representations at different positions have different distributions; normalizing across the batch conflates these.

Layer normalization computes statistics per token, avoiding all three issues.

### PyTorch Implementation

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization as used in Transformers.
    
    Normalizes across the last dimension (feature dimension).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., d_model]
        Returns:
            Normalized tensor [..., d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

## Pre-Norm vs Post-Norm

The placement of layer normalization relative to the sub-layer has significant effects on training stability and model quality.

### Post-Norm (Original Transformer)

The original "Attention Is All You Need" paper places normalization after the residual addition:

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**Properties:**

- Normalization controls the output scale after the residual connection
- Gradients pass through LayerNorm, which can constrain gradient magnitude
- Requires careful learning rate warmup to avoid training instability
- Tends to produce slightly better final performance when training succeeds

### Pre-Norm (Modern Standard)

Modern Transformers (GPT-2+, LLaMA, T5 v1.1) place normalization before the sub-layer:

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

**Properties:**

- Gradients flow freely through the residual connection without passing through normalization
- More stable for training deep models (no warmup required in many cases)
- The residual stream carries unnormalized representations, accumulating contributions from each layer
- Standard choice for models deeper than ~12 layers

### Gradient Flow Analysis

The key difference lies in gradient propagation. With pre-norm, the gradient of the loss with respect to an early layer's output has a direct path:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \sum_{k=l}^{L-1} \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} \cdot \frac{\partial \text{SubLayer}^{(k)}}{\partial \mathbf{x}^{(l)}}
$$

The first term is a direct gradient path (identity through residuals), while the second involves sub-layer Jacobians. In post-norm, this direct path is disrupted by the normalization function.

### PyTorch Implementation Comparison

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PostNormTransformerBlock(nn.Module):
    """Post-norm: LayerNorm after residual addition (original Transformer)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Post-norm: normalize AFTER residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x


class PreNormTransformerBlock(nn.Module):
    """Pre-norm: LayerNorm before sublayer (modern standard)."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm: normalize BEFORE sublayer
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = residual + self.dropout(attn_out)
        
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ffn(x_norm)
        x = residual + ff_out
        return x
```

### Final Layer Norm

In pre-norm architectures, a final layer normalization is applied after the last Transformer block, before the output projection. This is necessary because the residual stream accumulates unnormalized contributions:

```python
class PreNormTransformer(nn.Module):
    def __init__(self, d_model, num_layers, ...):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(d_model, ...) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)  # Critical for pre-norm
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # Normalize before output projection
        return self.output_proj(x)
```

Post-norm architectures do not need this final normalization because each layer's output is already normalized.

## RMSNorm

Root Mean Square Layer Normalization (RMSNorm) simplifies LayerNorm by removing the mean-centering step, used in LLaMA, Mistral, and other modern LLMs:

$$
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}
$$

Where:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

### Motivation

RMSNorm hypothesizes that the re-centering (mean subtraction) in LayerNorm is unnecessary and that the re-scaling (variance normalization) is the primary contributor to training stability. Removing mean-centering:

1. **Reduces computation**: One fewer reduction operation per normalization
2. **Simplifies gradients**: Fewer terms in the backward pass
3. **Empirically equivalent**: No degradation in model quality

### PyTorch Implementation

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Used in LLaMA, Mistral, and modern LLMs.
    Simpler and faster than LayerNorm with equivalent performance.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., d_model]
        Returns:
            Normalized tensor [..., d_model]
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms
```

### Comparison with LayerNorm

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Mean subtraction | Yes | No |
| Learnable shift ($\beta$) | Yes | No |
| Parameters | $2d$ ($\gamma, \beta$) | $d$ ($\gamma$) |
| Computation | 2 reductions (mean, var) | 1 reduction (RMS) |
| Used in | BERT, GPT-2, T5 | LLaMA, Mistral, Gemma |
| Performance | Baseline | Equivalent |

## Normalization Placement Across Architectures

| Model | Norm Type | Placement |
|-------|-----------|-----------|
| Original Transformer | LayerNorm | Post-norm |
| BERT | LayerNorm | Post-norm |
| GPT-2 | LayerNorm | Pre-norm |
| T5 v1.0 | LayerNorm | Pre-norm |
| T5 v1.1 | RMSNorm | Pre-norm |
| LLaMA | RMSNorm | Pre-norm |
| Mistral | RMSNorm | Pre-norm |

The field has converged on pre-norm with RMSNorm as the default for new large-scale models.

## Practical Considerations

### Weight Decay and Normalization

Normalization parameters ($\gamma$, $\beta$) are typically excluded from weight decay:

```python
# Separate parameters for optimizer
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'norm' in name or 'bias' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4)
```

### Numerical Stability

When working with mixed-precision training (FP16/BF16), normalization layers are typically kept in FP32 to maintain numerical stability:

```python
# In mixed-precision training, norms compute in FP32
class StableLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
    
    def forward(self, x):
        # Upcast to FP32 for normalization
        return self.norm(x.float()).type_as(x)
```

## Summary

Layer normalization in Transformers involves three key decisions:

1. **Norm type**: LayerNorm (traditional) vs RMSNorm (modern, faster, equivalent quality)
2. **Placement**: Post-norm (original, slightly better peak performance, harder to train) vs Pre-norm (modern standard, stable training, enables deep models)
3. **Final norm**: Required for pre-norm architectures before the output projection

Modern best practice is **pre-norm RMSNorm**, as used in LLaMA, Mistral, and other state-of-the-art models.

## References

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." arXiv.
2. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
3. Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS.
4. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
