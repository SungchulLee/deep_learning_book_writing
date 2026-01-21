# Encoder Block

## Overview

The encoder block is the fundamental building unit of the Transformer encoder. Each block consists of two sublayers: **multi-head self-attention** and a **position-wise feed-forward network**, each wrapped with residual connections and layer normalization.

## Architecture

```
Input X
    │
    ├──────────────────────┐
    ▼                      │
┌─────────────────────────┐│
│  Multi-Head Self-Attn   ││
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
Output X'
```

## Mathematical Formulation

Let $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$ be the input to the encoder layer.

**Sublayer 1: Self-Attention**
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttn}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

**Sublayer 2: Feed-Forward Network**
$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$

## Step-by-Step Computation

### Step 1: Store Input for Residual

$$\mathbf{X}_{\text{residual}_1} = \mathbf{X}$$

### Step 2: Compute Q, K, V for All Heads

For each head $i \in \{1, \ldots, h\}$:

$$\mathbf{Q}^{(i)} = \mathbf{X}\mathbf{W}_Q^{(i)} \in \mathbb{R}^{n \times d_k}$$
$$\mathbf{K}^{(i)} = \mathbf{X}\mathbf{W}_K^{(i)} \in \mathbb{R}^{n \times d_k}$$
$$\mathbf{V}^{(i)} = \mathbf{X}\mathbf{W}_V^{(i)} \in \mathbb{R}^{n \times d_v}$$

where $d_k = d_v = d_{\text{model}} / h$.

### Step 3: Compute Attention Scores

For each head $i$:

$$\mathbf{S}^{(i)} = \frac{\mathbf{Q}^{(i)} (\mathbf{K}^{(i)})^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

**Note for encoders**: No causal mask—every position attends to every position (bidirectional).

### Step 4: Softmax to Get Attention Weights

$$\mathbf{A}^{(i)} = \text{softmax}(\mathbf{S}^{(i)}) \in \mathbb{R}^{n \times n}$$

### Step 5: Compute Weighted Sum of Values

$$\text{head}_i = \mathbf{A}^{(i)} \mathbf{V}^{(i)} \in \mathbb{R}^{n \times d_v}$$

### Step 6: Concatenate and Project

$$\text{MultiHead} = [\text{head}_1 | \ldots | \text{head}_h] \mathbf{W}_O \in \mathbb{R}^{n \times d_{\text{model}}}$$

### Step 7: First Residual Connection

$$\mathbf{Z}_{\text{attn}}' = \mathbf{X}_{\text{residual}_1} + \text{MultiHead}$$

### Step 8: First Layer Normalization

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{Z}_{\text{attn}}')$$

For each position $j$:
$$\mu_j = \frac{1}{d} \sum_{\ell=1}^{d} Z'_{j\ell}, \quad \sigma_j^2 = \frac{1}{d} \sum_{\ell=1}^{d} (Z'_{j\ell} - \mu_j)^2$$
$$(\mathbf{X}')_{j\ell} = \gamma_\ell \cdot \frac{Z'_{j\ell} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_\ell$$

### Step 9: Feed-Forward Network

$$\mathbf{H} = \text{ReLU}(\mathbf{X}' \mathbf{W}_1 + \mathbf{b}_1)$$
$$\mathbf{Z}_{\text{ffn}} = \mathbf{H} \mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$.

Typically $d_{ff} = 4 \cdot d_{\text{model}}$.

### Step 10: Second Residual Connection

$$\mathbf{Z}_{\text{ffn}}' = \mathbf{X}' + \mathbf{Z}_{\text{ffn}}$$

### Step 11: Second Layer Normalization

$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{Z}_{\text{ffn}}')$$

## Parameter Count

| Component | Parameters |
|-----------|------------|
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ (all heads) | $3 \times d_{\text{model}}^2$ |
| $\mathbf{W}_O$ | $d_{\text{model}}^2$ |
| LayerNorm 1 ($\gamma, \beta$) | $2 \times d_{\text{model}}$ |
| FFN $\mathbf{W}_1, \mathbf{b}_1$ | $d_{\text{model}} \times d_{ff} + d_{ff}$ |
| FFN $\mathbf{W}_2, \mathbf{b}_2$ | $d_{ff} \times d_{\text{model}} + d_{\text{model}}$ |
| LayerNorm 2 ($\gamma, \beta$) | $2 \times d_{\text{model}}$ |

**Total** ≈ $4d_{\text{model}}^2 + 8d_{\text{model}}^2 + 6d_{\text{model}} = 12d_{\text{model}}^2$ (using $d_{ff} = 4d_{\text{model}}$)

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Multi-head self-attention
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Multi-head self-attention
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_o(context)
        
        # First residual + layer norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Second residual + layer norm
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

# Example usage
d_model = 512
n_heads = 8
d_ff = 2048

encoder_block = EncoderBlock(d_model, n_heads, d_ff)
x = torch.randn(32, 100, d_model)  # batch=32, seq_len=100
output = encoder_block(x)
print(f"Output shape: {output.shape}")  # (32, 100, 512)
```

## Pre-Norm vs Post-Norm

The original Transformer uses **Post-Norm** (shown above):
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Sublayer}(\mathbf{X}))$$

Modern variants often use **Pre-Norm**:
$$\mathbf{X}' = \mathbf{X} + \text{Sublayer}(\text{LayerNorm}(\mathbf{X}))$$

Pre-Norm is more stable for deep networks but may have slightly lower final performance.

## Stacking Encoder Blocks

The full encoder stacks $N$ identical blocks:

$$\mathbf{X}^{(0)} = \text{Embedding}(\text{input}) + \text{PositionalEncoding}$$
$$\mathbf{X}^{(l)} = \text{EncoderBlock}^{(l)}(\mathbf{X}^{(l-1)}) \quad \text{for } l = 1, \ldots, N$$

Standard configurations: $N = 6$ (original), $N = 12$ (BERT-base), $N = 24$ (BERT-large).

## Summary

Each encoder block:

1. **Self-Attention**: Enables each token to gather information from all tokens
2. **FFN**: Applies position-wise nonlinear transformation
3. **Residual connections**: Enable gradient flow through deep networks
4. **Layer normalization**: Stabilizes training

The encoder's bidirectional attention allows complete understanding of the input sequence, making it ideal for tasks like classification, NER, and question answering.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)
