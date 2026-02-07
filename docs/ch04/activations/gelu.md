# GELU (Gaussian Error Linear Unit)

## Overview

**GELU (Gaussian Error Linear Unit)** is the activation function of choice for modern transformer architectures. Introduced by Hendrycks and Gimpel (2016), it has become the standard activation in BERT, GPT, RoBERTa, Vision Transformers (ViT), and virtually all state-of-the-art language models.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and probabilistic interpretation of GELU
2. Why GELU works well in transformers
3. Exact vs approximate computation and the trade-offs
4. Comparison with ReLU and Swish
5. PyTorch implementation patterns

---

## Mathematical Definition

### Exact Formula

$$\operatorname{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution:

$$\Phi(x) = P(X \leq x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^2/2}\, dt$$

### Expanded Form

$$\operatorname{GELU}(x) = \frac{x}{2}\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\operatorname{erf}$ is the error function.

### Approximations

Since computing the exact CDF is expensive, two approximations are commonly used:

**Tanh approximation** (fast, widely adopted):

$$\operatorname{GELU}(x) \approx \frac{x}{2}\left[1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\bigl(x + 0.044715\,x^3\bigr)\right)\right]$$

**Sigmoid approximation** (simpler, slightly less accurate):

$$\operatorname{GELU}(x) \approx x \cdot \sigma(1.702\,x)$$

The sigmoid approximation reveals that GELU is closely related to [Swish](swish.md): Swish is $x \cdot \sigma(x)$, while the GELU sigmoid approximation is $x \cdot \sigma(1.702\,x)$.

### Derivative

$$\operatorname{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ is the standard normal PDF.

---

## Probabilistic Interpretation

### Stochastic Regularization Perspective

GELU can be understood as the **expected value** of a stochastic masking operation:

$$\operatorname{GELU}(x) = x \cdot P(X \leq x) = \mathbb{E}[x \cdot \mathbf{1}_{X \leq x}]$$

where $X \sim \mathcal{N}(0, 1)$.

**Interpretation**: Each input $x$ is kept (multiplied by 1) with probability $\Phi(x)$, or dropped (multiplied by 0) with probability $1 - \Phi(x)$:

- Large positive $x$: $\Phi(x) \approx 1$ → almost always kept (like ReLU)
- Large negative $x$: $\Phi(x) \approx 0$ → almost always dropped (like ReLU)
- Near-zero $x$: $\Phi(x) \approx 0.5$ → probabilistic mixture (smoother than ReLU)

### Connection to Dropout

This interpretation connects GELU to **dropout** with an input-dependent rate:

- Standard dropout: Fixed drop probability $p$ for all neurons
- GELU: Drop probability depends on input magnitude — larger inputs are more likely to be preserved

This provides an implicit form of regularization built into the activation function.

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $\approx (-0.17, +\infty)$ |
| **Monotonic** | No (small dip for negative values) |
| **Smooth** | Yes (infinitely differentiable) |
| **Approximately zero-centered** | Yes |
| **Minimum** | $\approx -0.17$ at $x \approx -0.75$ |

### Non-Monotonicity

GELU is **non-monotonic**: it dips slightly below zero for small negative inputs before asymptoting to zero:

```python
import torch
import torch.nn.functional as F

# GELU is negative for some inputs
x = torch.tensor([-1.0])
print(f"GELU(-1) = {F.gelu(x).item():.4f}")  # ≈ -0.1588

# The minimum is around x ≈ -0.75
x_min = torch.tensor([-0.75])
print(f"GELU(-0.75) = {F.gelu(x_min).item():.4f}")  # ≈ -0.17
```

This non-monotonicity allows neurons to produce different outputs for similar negative inputs, potentially capturing more nuanced features than ReLU's flat zero.

---

## PyTorch Implementation

### Functional API

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])

# Default (exact computation)
y = F.gelu(x)

# Tanh approximation (faster, used in many production systems)
y_approx = F.gelu(x, approximate='tanh')

print(f"Input:   {x.tolist()}")
print(f"Exact:   {[f'{v:.4f}' for v in y.tolist()]}")
print(f"Approx:  {[f'{v:.4f}' for v in y_approx.tolist()]}")
```

### Module API

```python
import torch.nn as nn

# Exact computation
gelu_exact = nn.GELU()

# Tanh approximation
gelu_approx = nn.GELU(approximate='tanh')

x = torch.randn(32, 768)  # Typical transformer hidden size
y = gelu_exact(x)
```

### Exact vs Approximate

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Exact | Slower | Perfect | Research, small models |
| Tanh approx | Faster | ~99.99% | Production, large models |

```python
import torch
import torch.nn.functional as F

x = torch.randn(1000)

# Maximum difference between exact and approximate
exact = F.gelu(x)
approx = F.gelu(x, approximate='tanh')
max_diff = (exact - approx).abs().max().item()
print(f"Maximum difference: {max_diff:.6f}")  # Typically < 0.0002
```

---

## GELU vs ReLU

### Key Differences

| Aspect | ReLU | GELU |
|--------|------|------|
| **At $x=0$** | Sharp corner | Smooth |
| **Negative inputs** | Exactly 0 | Slightly negative |
| **Gradient at $x=0$** | Undefined (0 or 1) | Well-defined (0.5) |
| **Monotonic** | Yes | No |
| **Probabilistic** | No | Yes |
| **Computational cost** | Very low | Higher |

### Why GELU Works Better in Transformers

1. **Smoothness**: Attention mechanisms produce continuous-valued outputs; smooth activations maintain this continuity through the FFN block
2. **Non-zero gradients for negative inputs**: Information flows even for weakly negative activations, preventing information loss
3. **Regularization effect**: The stochastic masking interpretation provides implicit regularization
4. **Empirical success**: Consistently outperforms ReLU in NLP benchmarks across architectures

### Visualization

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-4, 4, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), torch.nn.functional.gelu(x).numpy(), 
         label='GELU', linewidth=2.5)
plt.plot(x.numpy(), torch.relu(x).numpy(), 
         label='ReLU', linewidth=2, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('GELU vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

---

## Usage in Major Architectures

### BERT (Bidirectional Encoder)

```python
import torch.nn as nn

class BertFeedForward(nn.Module):
    """BERT's feed-forward network uses GELU."""
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, hidden_states):
        hidden = self.dense_in(hidden_states)
        hidden = self.activation(hidden)
        output = self.dense_out(hidden)
        return output
```

### GPT (Generative Pre-trained Transformer)

```python
class GPTBlock(nn.Module):
    """Simplified GPT block with GELU."""
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=12)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
```

### Vision Transformer (ViT)

```python
class ViTMLP(nn.Module):
    """Vision Transformer MLP with GELU."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
```

---

## Implementation from Scratch

Understanding GELU's implementation helps with debugging and porting to other frameworks:

```python
import torch
import torch.nn as nn
import math

class GELUCustom(nn.Module):
    """Custom GELU implementation with selectable approximation."""
    
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x):
        if self.approximate == 'tanh':
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            ))
        elif self.approximate == 'sigmoid':
            return x * torch.sigmoid(1.702 * x)
        else:
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Verify against PyTorch
x = torch.randn(100)
custom_gelu = GELUCustom(approximate='tanh')
pytorch_gelu = nn.GELU(approximate='tanh')

assert torch.allclose(custom_gelu(x), pytorch_gelu(x), atol=1e-6)
print("Custom GELU matches PyTorch!")
```

---

## Computational Considerations

### Memory Efficiency

GELU requires storing the input tensor for the backward pass (like most activations). For memory-constrained training, gradient checkpointing can help:

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
    
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self.ffn, x, use_reentrant=False)
        return self.ffn(x)
```

---

## When to Use GELU

### Recommended

- ✅ Transformer architectures (NLP, Vision)
- ✅ BERT-style and GPT-style models
- ✅ Vision Transformers
- ✅ When smoothness matters

### Consider Alternatives

- ⚠️ Very deep CNNs: ReLU + BatchNorm is well-tested and sufficient
- ⚠️ Mobile/edge deployment: ReLU or Hardswish are more efficient
- ⚠️ Modern LLMs at scale: SwiGLU (gated variant) may outperform standard GELU FFN

---

## Summary

| Aspect | GELU |
|--------|------|
| **Formula** | $x \cdot \Phi(x)$ |
| **Smoothness** | ✅ Infinitely differentiable |
| **Best for** | Transformers (NLP, Vision) |
| **Standard in** | BERT, GPT, ViT, RoBERTa |
| **Approximation** | Tanh approx for production speed |
| **vs ReLU** | Better gradients, smoother, non-monotonic |
| **vs Swish** | Very similar; GELU preferred for transformers |

!!! tip "Practical Recommendation"
    If you're building a transformer-based model, **use GELU by default**. It's the industry standard and consistently performs well. Use `approximate='tanh'` for production deployments where speed matters. For cutting-edge LLMs, consider [SwiGLU](swish.md) in the FFN block.

---

## References

1. Hendrycks, D., & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)". arXiv:1606.08415
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition"
