# Masked Self-Attention

## Purpose

The causal mask serves one critical purpose: **prevent information leakage from future tokens during training**.

### The Problem

During training with teacher forcing, the entire target sequence is available:

- Input: "The cat sat on"
- We want to predict: "cat" after "The", "sat" after "The cat", etc.

Without masking, position 1 ("The") could "see" position 4 ("on") through attention—information that wouldn't exist during inference.

### The Solution

The causal mask ensures that position $i$ can only attend to positions $1, 2, \ldots, i$.

## Mask Implementation

### The Lower Triangular Mask

For a sequence of length $n$, the causal mask is:

$$\mathbf{M}_{\text{causal}} = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

This is added to the attention scores **before** softmax:

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}\right)$$

### Effect of $-\infty$

After adding the mask:

$$\tilde{S}_{ij} = \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax:

$$A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_k \exp(\tilde{S}_{ik})} = \begin{cases} \text{positive value} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}$$

Because $\exp(-\infty) = 0$, future positions receive zero attention weight.

## Visualizing the Difference

### Encoder (No Mask): Full Attention Matrix

$$\mathbf{A}_{\text{encoder}} = \begin{pmatrix} a_{11} & a_{12} & a_{13} & a_{14} \\ a_{21} & a_{22} & a_{23} & a_{24} \\ a_{31} & a_{32} & a_{33} & a_{34} \\ a_{41} & a_{42} & a_{43} & a_{44} \end{pmatrix}$$

Every position attends to every position. All entries are non-zero.

### Decoder (Causal Mask): Lower Triangular

$$\mathbf{A}_{\text{decoder}} = \begin{pmatrix} a_{11} & 0 & 0 & 0 \\ a_{21} & a_{22} & 0 & 0 \\ a_{31} & a_{32} & a_{33} & 0 \\ a_{41} & a_{42} & a_{43} & a_{44} \end{pmatrix}$$

Position $i$ only attends to positions $1, 2, \ldots, i$.

## Training vs Inference

### Training (Mask Required)

During training:
- The entire target sequence is present (teacher forcing)
- Future tokens physically exist in the batch
- Mask **prevents** the model from seeing them

**Example**: Target = "The black cat \<end\>"

All tokens present simultaneously:

| | The | black | cat | \<end\> |
|---|:---:|:---:|:---:|:---:|
| **The** | ✓ | mask | mask | mask |
| **black** | ✓ | ✓ | mask | mask |
| **cat** | ✓ | ✓ | ✓ | mask |
| **\<end\>** | ✓ | ✓ | ✓ | ✓ |

### Inference (Mask Unnecessary but Harmless)

During inference:
- We generate one token at a time
- Future tokens literally **don't exist yet**
- Nothing to mask—there's nothing there

**Step-by-step generation**:

1. Input: "\<start\>" → Predict: "The"
2. Input: "\<start\> The" → Predict: "black"
3. Input: "\<start\> The black" → Predict: "cat"

No future tokens to mask. The causal structure emerges naturally.

## Summary Table

| Phase | Future tokens exist? | Mask needed? | Why? |
|-------|---------------------|--------------|------|
| Training | Yes (teacher forcing) | Yes | Prevent cheating |
| Inference | No (sequential) | No | Nothing to hide |

## Training-Inference Consistency

The mask makes training **match** inference:

**Training with mask**:
- Position 3 sees: tokens 1, 2, 3
- Predicts: token 4

**Inference without mask**:
- Position 3 sees: tokens 1, 2, 3 (token 4 doesn't exist yet)
- Predicts: token 4

**Same information in both cases.** The model learns under identical constraints.

## What Would Happen Without Mask?

**Training without mask** (hypothetical):
- Position 3 sees: tokens 1, 2, 3, 4, 5, ... (everything)
- Learns to rely on future context

**Inference** (reality):
- Position 3 sees: tokens 1, 2, 3 only
- Model looks for future context that isn't there
- **Performance collapses**

This is **train-test mismatch**. The mask prevents it.

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F
import math

def create_causal_mask(size, device='cpu'):
    """Create a causal mask where position i can attend to positions 0...i"""
    # Upper triangular matrix of 1s (positions to mask)
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    # Convert to -inf for masked positions, 0 for allowed
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def masked_self_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention with optional causal mask.
    
    Args:
        Q: Queries (batch, heads, seq_len, d_k)
        K: Keys (batch, heads, seq_len, d_k)
        V: Values (batch, heads, seq_len, d_v)
        mask: Causal mask (seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply causal mask
    if mask is not None:
        scores = scores + mask  # Broadcasting adds mask to all batches/heads
    
    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# Example
batch_size = 2
n_heads = 8
seq_len = 10
d_k = 64

Q = torch.randn(batch_size, n_heads, seq_len, d_k)
K = torch.randn(batch_size, n_heads, seq_len, d_k)
V = torch.randn(batch_size, n_heads, seq_len, d_k)

# Create causal mask
causal_mask = create_causal_mask(seq_len)

output, weights = masked_self_attention(Q, K, V, causal_mask)

print(f"Output shape: {output.shape}")      # (2, 8, 10, 64)
print(f"Attention shape: {weights.shape}")   # (2, 8, 10, 10)

# Verify: attention weights for position 0 should only be on position 0
print(f"Position 0 weights: {weights[0, 0, 0, :]}")  # Only first element non-zero
```

## Efficient Mask Application

Modern implementations use boolean masks or fused operations:

```python
# Boolean mask (True = attend, False = mask)
def create_causal_mask_bool(size, device='cpu'):
    return torch.tril(torch.ones(size, size, device=device)).bool()

# Apply with masked_fill
def masked_attention_bool(scores, mask):
    return scores.masked_fill(~mask, float('-inf'))
```

## FlashAttention and Causal Masks

FlashAttention implementations have specialized kernels for causal masks:

```python
# Using PyTorch 2.0+ scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention

output = scaled_dot_product_attention(
    Q, K, V,
    is_causal=True  # Efficient causal mask handling
)
```

This is faster than explicit mask multiplication.

## Summary

- **Purpose**: Prevent future information leakage during training
- **Implementation**: Add $-\infty$ to upper triangular before softmax
- **Training**: Required—future tokens exist but must be hidden
- **Inference**: Unnecessary—future tokens don't exist
- **Practice**: Applied in both phases for code simplicity
- **Result**: Training conditions match inference conditions

The causal mask isn't punishment—it's realistic practice that ensures the model learns to generate without peeking at answers.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Dao et al., "FlashAttention" (2022)
