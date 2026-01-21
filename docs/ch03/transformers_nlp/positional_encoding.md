# Positional Encoding

## The Problem

Self-attention is **permutation equivariant**: if we shuffle the input positions, the output is shuffled identically. Attention treats positions symmetrically—it has no built-in notion of order.

Consider: "The cat sat on the mat" vs "mat the on sat cat The"

Without positional information, self-attention sees these as equivalent (same bag of tokens). But word order is crucial for meaning!

## The Solution

Add positional encodings to the input embeddings:

$$\mathbf{X}_{\text{pos}} = \mathbf{X}_{\text{embed}} + \mathbf{PE}$$

where $\mathbf{PE} \in \mathbb{R}^{n \times d}$ contains position information.

## Sinusoidal Positional Encoding

The original Transformer uses sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where:
- $pos$: Position in the sequence (0, 1, 2, ...)
- $i$: Dimension index (0, 1, ..., $d/2 - 1$)
- $d$: Model dimension

### Why Sinusoids?

**1. Unique encoding per position**: Each position gets a distinct pattern.

**2. Bounded values**: Sine and cosine are in $[-1, 1]$, matching typical embedding scales.

**3. Relative position information**: For any fixed offset $k$:

$$PE_{pos+k} = f(PE_{pos})$$

The encoding at position $pos + k$ can be expressed as a linear function of the encoding at position $pos$. This allows the model to learn relative positioning.

**4. Extrapolation**: Can handle sequences longer than seen during training (in principle).

### Geometric Interpretation

Think of each dimension pair as a clock hand rotating at different speeds:

| Dimension | Wavelength | Interpretation |
|-----------|------------|----------------|
| $i = 0$ | $2\pi$ | Fast rotation (local position) |
| $i = d/4$ | $\approx 1000$ | Medium rotation |
| $i = d/2 - 1$ | $\approx 10000 \cdot 2\pi$ | Slow rotation (global position) |

Low dimensions capture fine-grained position; high dimensions capture coarse position.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisor term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Example usage
d_model = 512
max_len = 1000

pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
x = torch.randn(32, 100, d_model)  # batch=32, seq_len=100
x_with_pos = pos_encoder(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {x_with_pos.shape}")  # Same shape
```

## Learned Positional Embeddings

An alternative to sinusoidal: learn the encodings as parameters.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)
```

### Comparison

| Aspect | Sinusoidal | Learned |
|--------|------------|---------|
| Parameters | 0 | $\text{max\_len} \times d$ |
| Extrapolation | Theoretically possible | Limited to training length |
| Performance | Comparable | Comparable |
| Usage | Original Transformer | BERT, GPT-2 |

In practice, both work similarly for typical sequence lengths.

## Relative Positional Encoding

Modern variants encode **relative** rather than **absolute** positions.

### Why Relative?

"The cat sat" at positions 0-2 should have similar relationships as "The cat sat" at positions 100-102. Absolute encodings don't guarantee this; relative encodings do.

### Relative Attention Bias (T5-style)

Add a learned bias based on distance:

$$A_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + b_{|i-j|}$$

where $b$ is a learned bias indexed by relative distance.

### Rotary Position Embedding (RoPE)

Used in LLaMA, GPT-NeoX, and many modern models:

$$\mathbf{q}'_m = R_{\Theta, m} \mathbf{q}_m, \quad \mathbf{k}'_n = R_{\Theta, n} \mathbf{k}_n$$

where $R_{\Theta, m}$ is a rotation matrix depending on position $m$. The dot product naturally encodes relative position:

$$\mathbf{q}'^T_m \mathbf{k}'_n = \mathbf{q}_m^T R_{\Theta, n-m} \mathbf{k}_n$$

## ALiBi: Attention with Linear Biases

Add a linear penalty based on distance:

$$A_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} - m \cdot |i - j|$$

where $m$ is a head-specific slope. Farther tokens get lower attention—simple but effective.

## Position Encoding Summary

| Method | Type | Extrapolation | Modern Usage |
|--------|------|---------------|--------------|
| Sinusoidal | Absolute | Limited | Original Transformer |
| Learned | Absolute | No | BERT, GPT-2 |
| Relative (T5) | Relative | Better | T5 |
| RoPE | Relative | Good | LLaMA, GPT-NeoX |
| ALiBi | Relative | Excellent | BLOOM |

## Why Position Matters

Without positional encoding:

```
"Dog bites man" ≈ "Man bites dog"  (same attention patterns)
```

With positional encoding:

```
"Dog bites man" ≠ "Man bites dog"  (different patterns due to position)
```

The model can learn that the first noun is typically the subject and the second is the object.

## Summary

Positional encoding solves the fundamental limitation that attention is position-agnostic:

1. **Sinusoidal**: Fixed patterns at multiple frequencies
2. **Learned**: Trainable embedding per position
3. **Relative**: Encode relationships between positions
4. **RoPE/ALiBi**: Modern methods with better extrapolation

All methods inject positional information so the model can distinguish "The cat sat" from "sat cat The"—essential for understanding language.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Shaw et al., "Self-Attention with Relative Position Representations" (2018)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Press et al., "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022)
