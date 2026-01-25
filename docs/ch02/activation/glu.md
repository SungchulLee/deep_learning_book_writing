# GLU (Gated Linear Unit) and Variants

## Overview

**Gated Linear Units (GLU)** and their variants are a family of activation mechanisms that use learned gating to control information flow. Unlike simple activation functions, GLUs combine multiple linear transformations with element-wise gating, providing more expressive power. They have become important components in modern architectures like GPT-style models, LLaMA, and PaLM.

## Learning Objectives

By the end of this section, you will understand:

1. The gating mechanism and its benefits
2. Original GLU and its variants (GeGLU, SwiGLU, ReGLU)
3. Why gated activations improve transformers
4. Implementation patterns in PyTorch
5. When to use gated vs simple activations

---

## Gated Linear Unit (GLU)

### Mathematical Definition

The original GLU splits the input into two halves and applies gating:

$$\text{GLU}(\mathbf{x}) = (\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) \otimes \sigma(\mathbf{W}_2\mathbf{x} + \mathbf{b}_2)$$

where:
- $\mathbf{W}_1, \mathbf{W}_2$ are learned weight matrices
- $\mathbf{b}_1, \mathbf{b}_2$ are bias vectors
- $\sigma$ is the sigmoid function
- $\otimes$ denotes element-wise multiplication

### Intuition

The GLU can be decomposed into:
- **Value path**: $\mathbf{W}_1\mathbf{x} + \mathbf{b}_1$ — the "content" to pass through
- **Gate path**: $\sigma(\mathbf{W}_2\mathbf{x} + \mathbf{b}_2)$ — controls how much content passes

The gate produces values in $(0, 1)$, allowing the network to learn which features to amplify or suppress.

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """Gated Linear Unit from Dauphin et al. (2017)."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Double the output size for value and gate paths
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x):
        # Split into value and gate
        x = self.linear(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)

# Alternative using F.glu
class GLUFunctional(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x):
        return F.glu(self.linear(x), dim=-1)

# Usage
glu = GLU(768, 3072)
x = torch.randn(32, 128, 768)  # [batch, seq, dim]
y = glu(x)
print(f"Input:  {x.shape}")
print(f"Output: {y.shape}")  # [32, 128, 3072]
```

---

## Modern GLU Variants

### GeGLU (GELU Gated Linear Unit)

Replaces sigmoid with GELU in the gate:

$$\text{GeGLU}(\mathbf{x}) = (\mathbf{W}_1\mathbf{x}) \otimes \text{GELU}(\mathbf{W}_2\mathbf{x})$$

```python
class GeGLU(nn.Module):
    """GELU-Gated Linear Unit."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x):
        x = self.linear(x)
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)
```

### SwiGLU (Swish Gated Linear Unit)

Uses Swish (SiLU) as the gating function:

$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{W}_1\mathbf{x}) \otimes \text{Swish}(\mathbf{W}_2\mathbf{x})$$

```python
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (used in LLaMA, PaLM)."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=False)
        self.w2 = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        return self.w1(x) * F.silu(self.w2(x))
```

!!! note "SwiGLU in Practice"
    SwiGLU has become the standard in modern LLMs like LLaMA, PaLM, and Falcon. It typically outperforms both standard FFN+ReLU and FFN+GELU configurations.

### ReGLU (ReLU Gated Linear Unit)

Uses ReLU as the gating function:

$$\text{ReGLU}(\mathbf{x}) = (\mathbf{W}_1\mathbf{x}) \otimes \text{ReLU}(\mathbf{W}_2\mathbf{x})$$

```python
class ReGLU(nn.Module):
    """ReLU-Gated Linear Unit."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x):
        x = self.linear(x)
        value, gate = x.chunk(2, dim=-1)
        return value * F.relu(gate)
```

---

## Comparison of GLU Variants

| Variant | Gate Function | Used In | Relative Performance |
|---------|---------------|---------|---------------------|
| **GLU** | Sigmoid | ConvS2S | Baseline |
| **ReGLU** | ReLU | Research | Similar to GLU |
| **GeGLU** | GELU | T5, PaLM | Better than GLU |
| **SwiGLU** | Swish | LLaMA, PaLM | Best overall |

---

## GLU in Transformer FFN

### Standard Transformer FFN

The original transformer uses a simple two-layer FFN:

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

```python
class StandardFFN(nn.Module):
    """Standard transformer feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))
```

### Gated FFN (SwiGLU)

Modern LLMs often use SwiGLU-based FFN:

$$\text{FFN}(\mathbf{x}) = (\text{Swish}(\mathbf{x}\mathbf{W}_1) \otimes \mathbf{x}\mathbf{W}_3)\mathbf{W}_2$$

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU-based FFN (LLaMA style)."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Note: Often d_ff is scaled by 2/3 to match parameter count
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
```

### Parameter Count Consideration

GLU variants require more parameters (3 matrices instead of 2). To maintain the same parameter count as standard FFN:

- Standard FFN: $d_\text{model} \times d_\text{ff} + d_\text{ff} \times d_\text{model} = 2 \times d_\text{model} \times d_\text{ff}$
- SwiGLU FFN: $3 \times d_\text{model} \times d_\text{ff}'$

To match: $d_\text{ff}' = \frac{2}{3} d_\text{ff}$

```python
class ParameterEfficientSwiGLU(nn.Module):
    """SwiGLU with reduced hidden dim to match standard FFN params."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Reduce hidden dim by 2/3 to match parameter count
        hidden_dim = int(2 * d_ff / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## Why Gated Activations Help

### Information Highway

GLU creates an "information highway" where:
- Important features pass through (gate → 1)
- Irrelevant features are blocked (gate → 0)

This is more expressive than applying a fixed nonlinearity.

### Gradient Flow

The multiplicative gating provides multiple gradient paths:

$$\frac{\partial}{\partial x} [v(x) \cdot g(x)] = \frac{\partial v}{\partial x} \cdot g(x) + v(x) \cdot \frac{\partial g}{\partial x}$$

Even if one path saturates, gradients can flow through the other.

### Learned Selectivity

Unlike ReLU (which zeros based on sign) or GELU (which zeros probabilistically), GLU learns *what* to select:

```python
# ReLU: Fixed rule - zero if negative
y = F.relu(x)  # Zeros where x < 0

# GLU: Learned rule - gate controls what passes
y = value * sigmoid(gate)  # Gate is learned!
```

---

## Implementation: Complete LLaMA-style Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight

class LLaMABlock(nn.Module):
    """LLaMA-style transformer block with SwiGLU."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ffn_norm = RMSNorm(d_model)
        # SwiGLU FFN
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
    
    def forward(self, x, mask=None):
        # Pre-norm attention
        h = self.attention_norm(x)
        h, _ = self.attention(h, h, h, attn_mask=mask)
        x = x + h
        
        # Pre-norm SwiGLU FFN
        h = self.ffn_norm(x)
        h = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + h
        
        return x

# Example usage
block = LLaMABlock(d_model=768, n_heads=12, d_ff=3072)
x = torch.randn(4, 128, 768)  # [batch, seq, dim]
y = block(x)
print(f"Output shape: {y.shape}")  # [4, 128, 768]
```

---

## Bilinear Layers (Generalization)

GLU can be seen as a special case of **bilinear layers**:

$$\text{Bilinear}(\mathbf{x}) = \mathbf{x}\mathbf{W}_1 \otimes f(\mathbf{x}\mathbf{W}_2)$$

Different choices of $f$ give different variants:

| $f$ | Name |
|-----|------|
| $\sigma(x)$ | GLU |
| $\text{ReLU}(x)$ | ReGLU |
| $\text{GELU}(x)$ | GeGLU |
| $\text{Swish}(x)$ | SwiGLU |
| $x$ | Bilinear |

---

## Benchmarks and Selection

### When to Use GLU Variants

| Scenario | Recommendation |
|----------|----------------|
| **Building an LLM** | SwiGLU (proven in LLaMA, PaLM) |
| **Fine-tuning existing models** | Match original architecture |
| **Parameter-constrained** | Standard GELU FFN |
| **Research/experimentation** | Try SwiGLU vs GELU |

### Empirical Results (from literature)

On language modeling perplexity:
- GeGLU improves over GELU by ~0.5-1%
- SwiGLU improves over GeGLU by ~0.2-0.5%
- All GLU variants improve over standard FFN

---

## Summary

| Concept | Description |
|---------|-------------|
| **GLU** | $v \otimes \sigma(g)$ — multiplicative gating |
| **SwiGLU** | $v \otimes \text{Swish}(g)$ — current best practice |
| **Use case** | Transformer FFN blocks |
| **Benefit** | Better gradient flow, learned feature selection |
| **Cost** | 50% more parameters (mitigate with smaller hidden dim) |

!!! tip "Practical Recommendation"
    For new LLM projects, **use SwiGLU** following the LLaMA architecture. For standard transformer work where compatibility matters, stick with **GELU FFN**. The extra complexity of GLU is mainly worthwhile at scale.

---

## References

1. Dauphin, Y. N., et al. (2017). "Language Modeling with Gated Convolutional Networks"
2. Shazeer, N. (2020). "GLU Variants Improve Transformer"
3. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
4. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways"
