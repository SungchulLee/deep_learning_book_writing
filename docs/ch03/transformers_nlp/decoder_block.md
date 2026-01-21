# Decoder Block

## Overview

The decoder block is the building unit of the Transformer decoder. Each block consists of **three sublayers**: masked self-attention, cross-attention to encoder, and a feed-forward network—each wrapped with residual connections and layer normalization.

## Architecture

```
Input Y (target sequence)
    │
    ├──────────────────────┐
    ▼                      │
┌─────────────────────────┐│
│ Masked Multi-Head Attn  ││  ← Causal mask prevents future tokens
└─────────────────────────┘│
    │                      │
    ▼                      │
┌────────┐                 │
│   +    │◄────────────────┘ (Residual)
└────────┘
    │
    ▼
┌─────────────────────────┐
│      Layer Norm         │
└─────────────────────────┘
    │
    ├──────────────────────┐
    ▼                      │        ┌─────────────────┐
┌─────────────────────────┐│        │ Encoder Output  │
│    Cross-Attention      │◄────────┤      (M)        │
└─────────────────────────┘│        └─────────────────┘
    │                      │
    ▼                      │
┌────────┐                 │
│   +    │◄────────────────┘ (Residual)
└────────┘
    │
    ▼
┌─────────────────────────┐
│      Layer Norm         │
└─────────────────────────┘
    │
    ├──────────────────────┐
    ▼                      │
┌─────────────────────────┐│
│   Feed-Forward Network  ││
└─────────────────────────┘│
    │                      │
    ▼                      │
┌────────┐                 │
│   +    │◄────────────────┘ (Residual)
└────────┘
    │
    ▼
┌─────────────────────────┐
│      Layer Norm         │
└─────────────────────────┘
    │
    ▼
Output Y'
```

## Mathematical Formulation

Let $\mathbf{Y} \in \mathbb{R}^{n_t \times d}$ be the decoder input and $\mathbf{M} \in \mathbb{R}^{n_s \times d}$ be the encoder output (memory).

**Sublayer 1: Masked Self-Attention**
$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y}))$$

**Sublayer 2: Cross-Attention**
$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{CrossAttn}(\mathbf{Y}', \mathbf{M}))$$

**Sublayer 3: Feed-Forward Network**
$$\mathbf{Y}''' = \text{LayerNorm}(\mathbf{Y}'' + \text{FFN}(\mathbf{Y}''))$$

## Three Types of Attention in Decoder

| Sublayer | Q Source | K, V Source | Mask | Purpose |
|----------|----------|-------------|------|---------|
| Masked Self-Attn | Decoder | Decoder | Causal | Process target context |
| Cross-Attention | Decoder | Encoder | None | Reference source |
| (Encoder has) Self-Attn | Encoder | Encoder | None | Process source |

## Masked Self-Attention

The decoder's self-attention uses a **causal mask** to prevent attending to future positions:

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}\right)$$

where:
$$\mathbf{M}_{\text{causal}} = \begin{pmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{pmatrix}$$

This ensures position $i$ only attends to positions $1, \ldots, i$.

## Cross-Attention

Cross-attention bridges decoder and encoder:

- **Queries**: From current decoder state $\mathbf{Y}'$
- **Keys, Values**: From encoder output $\mathbf{M}$

$$\mathbf{Q} = \mathbf{Y}'\mathbf{W}_Q^{\text{(cross)}}, \quad \mathbf{K} = \mathbf{M}\mathbf{W}_K^{\text{(cross)}}, \quad \mathbf{V} = \mathbf{M}\mathbf{W}_V^{\text{(cross)}}$$

No mask needed—the decoder can attend to any encoder position.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Masked self-attention
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_o = nn.Linear(d_model, d_model)
        
        # Cross-attention
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_o = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(attn_weights), V)
    
    def forward(self, y, encoder_output, src_mask=None, tgt_mask=None):
        batch_size, tgt_len, _ = y.size()
        src_len = encoder_output.size(1)
        
        # Create causal mask for self-attention
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=y.device), diagonal=1
            ).bool()
            tgt_mask = ~tgt_mask  # Invert: True = attend, False = mask
        
        # ====== Sublayer 1: Masked Self-Attention ======
        Q = self.self_attn_q(y).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.self_attn_k(y).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.self_attn_v(y).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        
        self_attn_out = self._attention(Q, K, V, mask=tgt_mask.unsqueeze(0).unsqueeze(0))
        self_attn_out = self_attn_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        self_attn_out = self.self_attn_o(self_attn_out)
        
        y = self.norm1(y + self.dropout(self_attn_out))
        
        # ====== Sublayer 2: Cross-Attention ======
        Q = self.cross_attn_q(y).view(batch_size, tgt_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.cross_attn_k(encoder_output).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.cross_attn_v(encoder_output).view(batch_size, src_len, self.n_heads, self.d_k).transpose(1, 2)
        
        cross_attn_out = self._attention(Q, K, V, mask=src_mask)
        cross_attn_out = cross_attn_out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        cross_attn_out = self.cross_attn_o(cross_attn_out)
        
        y = self.norm2(y + self.dropout(cross_attn_out))
        
        # ====== Sublayer 3: Feed-Forward Network ======
        ffn_out = self.ffn(y)
        y = self.norm3(y + self.dropout(ffn_out))
        
        return y

# Example usage
d_model = 512
n_heads = 8
d_ff = 2048

decoder_block = DecoderBlock(d_model, n_heads, d_ff)

# Encoder output (from encoder stack)
encoder_output = torch.randn(32, 20, d_model)  # batch=32, src_len=20

# Decoder input (target sequence, shifted right)
y = torch.randn(32, 15, d_model)  # batch=32, tgt_len=15

output = decoder_block(y, encoder_output)
print(f"Output shape: {output.shape}")  # (32, 15, 512)
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Self-attention (Q, K, V, O) | $4 \times d_{\text{model}}^2$ |
| Cross-attention (Q, K, V, O) | $4 \times d_{\text{model}}^2$ |
| FFN | $8 \times d_{\text{model}}^2$ |
| LayerNorms (3 total) | $6 \times d_{\text{model}}$ |
| **Total** | $\approx 16 d_{\text{model}}^2$ |

The decoder has more parameters per layer than the encoder due to cross-attention.

## Decoder-Only (GPT-style)

For decoder-only models (no encoder):

```python
class DecoderOnlyBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Only masked self-attention (no cross-attention)
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Masked self-attention
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

Only 2 sublayers (like encoder), but with causal masking.

## Information Flow

During generation of target token at position $t$:

1. **Masked Self-Attention**: Sees positions $1, \ldots, t$ of target (not $t+1, \ldots$)
2. **Cross-Attention**: Sees all positions of source (full encoder output)
3. **FFN**: Position-wise transformation

This allows the decoder to:
- Build context from previously generated tokens
- Reference the full source at any generation step

## Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Target input | Full target (shifted right) | Generated tokens so far |
| Parallelism | All positions at once | One position at a time |
| Causal mask | Enforced | Natural (future doesn't exist) |
| Cross-attention | To encoder output | To encoder output (cached) |

## Summary

The decoder block:

1. **Masked Self-Attention**: Processes target sequence autoregressively
2. **Cross-Attention**: Bridges to encoder output
3. **FFN**: Applies nonlinear transformation
4. **Residual + LayerNorm**: Enables deep stacking

This three-sublayer structure allows the decoder to generate coherent output conditioned on both previous generation and the source sequence.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
