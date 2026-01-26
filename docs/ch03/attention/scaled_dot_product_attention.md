# Scaled Dot-Product Attention

## Introduction

Scaled dot-product attention is the fundamental building block of the Transformer architecture. Introduced by Vaswani et al. (2017) in "Attention Is All You Need," this mechanism provides an efficient and effective way to compute attention weights using matrix operations that can be highly parallelized on modern hardware.

The key innovation lies in combining the computational efficiency of dot-product attention with a scaling factor that ensures stable gradients during training. This seemingly simple modification was crucial for enabling the training of deep attention-based models.

## Mathematical Formulation

### The Attention Function

Given queries $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, and values $\mathbf{V} \in \mathbb{R}^{m \times d_v}$, scaled dot-product attention computes:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

### Step-by-Step Computation

**Step 1: Compute Raw Attention Scores**

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times m}
$$

Each element $s_{ij}$ measures the similarity between query $i$ and key $j$:

$$
s_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} k_{jl}
$$

**Interpretation**: Higher dot product indicates the query and key point in similar directions in the embedding space.

**Step 2: Apply Scaling**

$$
\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}
$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ is critical for training stability.

**Step 3: Apply Softmax (Row-wise)**

$$
\alpha_{ij} = \frac{\exp(s_{ij}^{\text{scaled}})}{\sum_{l=1}^{m} \exp(s_{il}^{\text{scaled}})}
$$

Each row of $\mathbf{A}$ becomes a probability distribution over positions:
- $\sum_j A_{ij} = 1$ for all $i$
- $A_{ij} \geq 0$ for all $i, j$

**Step 4: Weighted Sum of Values**

$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{n \times d_v}
$$

Each output row is a convex combination of value vectors, weighted by attention:

$$
\mathbf{o}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j
$$

### Dimensional Analysis

Understanding tensor shapes is crucial for implementation:

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(n, d_k)$ | $n$ query vectors of dimension $d_k$ |
| $\mathbf{K}$ | $(m, d_k)$ | $m$ key vectors of dimension $d_k$ |
| $\mathbf{V}$ | $(m, d_v)$ | $m$ value vectors of dimension $d_v$ |
| $\mathbf{S}$ | $(n, m)$ | Attention scores |
| $\mathbf{A}$ | $(n, m)$ | Attention weights (after softmax) |
| $\mathbf{O}$ | $(n, d_v)$ | Output vectors |

**Key constraints:**
- Queries and keys must have the same dimension ($d_k$)
- Keys and values must have the same sequence length ($m$)
- Values can have any dimension ($d_v$), which becomes the output dimension
- For self-attention: $n = m$ (same sequence for queries and keys/values)

### With Batches and Heads

In practice, we operate on batched, multi-head tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(B, H, n, d_k)$ | Batched multi-head queries |
| $\mathbf{K}$ | $(B, H, m, d_k)$ | Batched multi-head keys |
| $\mathbf{V}$ | $(B, H, m, d_v)$ | Batched multi-head values |
| $\mathbf{O}$ | $(B, H, n, d_v)$ | Batched multi-head outputs |

Where $B$ is batch size and $H$ is the number of attention heads.

---

## Deep Insight: Why Dot Product?

The dot product $\mathbf{q}^T \mathbf{k}$ has desirable properties that make it ideal for measuring relevance.

### Geometric Interpretation

$$
\mathbf{q}^T \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta
$$

where $\theta$ is the angle between vectors. High dot product when:
- Vectors point in similar directions (small $\theta$)
- Vectors have large magnitudes

This provides an intuitive notion of similarity: queries and keys that are "aligned" in embedding space produce high attention scores.

### Computational Efficiency

Dot products can be computed as matrix multiplication, which is highly optimized on GPUs:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # Single GEMM operation
```

This enables massive parallelization across all query-key pairs simultaneously.

### Comparison to Alternative Attention Mechanisms

| Attention Type | Scoring Function | Complexity | Notes |
|----------------|------------------|------------|-------|
| Dot-product | $\mathbf{q}^T \mathbf{k}$ | $O(d)$ | Simplest, fastest |
| Additive (Bahdanau) | $\mathbf{v}^T \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $O(d)$ + nonlinearity | More expressive, slower |
| Multiplicative | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | $O(d^2)$ | Learnable interaction |

**Key insight**: Scaled dot-product with learned projections (as in multi-head attention) achieves similar expressiveness to additive attention while being significantly faster due to hardware optimization of matrix multiplication.

---

## Deep Insight: The Critical Role of Scaling

### The Vanishing Gradient Problem

Consider what happens without scaling. For random vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with components drawn from $\mathcal{N}(0, 1)$:

$$
\mathbb{E}[\mathbf{q}^T \mathbf{k}] = 0, \quad \text{Var}[\mathbf{q}^T \mathbf{k}] = d_k
$$

As $d_k$ grows, dot products have increasingly large variance. With $d_k = 512$, scores can easily reach $\pm 30$ or more.

### Softmax Saturation

When inputs to softmax have large magnitude:

$$
\text{softmax}([100, 0, 0]) \approx [1.0, 0.0, 0.0]
$$

The distribution becomes nearly one-hot, causing:
1. **Vanishing gradients**: $\frac{\partial \text{softmax}}{\partial x} \approx 0$ in saturated regions
2. **Loss of information**: Nuanced attention patterns collapse to hard selection
3. **Training instability**: Gradients become noisy and unreliable

### The Scaling Solution

Dividing by $\sqrt{d_k}$ normalizes the variance:

$$
\text{Var}\left[\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1
$$

This keeps scores in a moderate range (typically $[-3, 3]$) where softmax gradients are healthy, enabling:
- Smooth gradient flow during backpropagation
- Rich, distributed attention patterns
- Stable training dynamics

### Temperature Interpretation

The scaling factor acts like inverse temperature in statistical mechanics:

$$
\text{softmax}\left(\frac{\mathbf{s}}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{\mathbf{s}}{T}\right) \text{ with } T = \sqrt{d_k}
$$

- **High temperature** ($T$ large): Uniform attention, exploration
- **Low temperature** ($T$ small): Concentrated attention, exploitation

---

## Deep Insight: Attention as Soft Content-Addressable Memory

### The Memory Analogy

Attention can be viewed as a differentiable memory system:

| Traditional Memory | Attention Equivalent |
|--------------------|---------------------|
| Memory address | Key vector $\mathbf{k}$ |
| Memory content | Value vector $\mathbf{v}$ |
| Query/lookup | Query vector $\mathbf{q}$ |
| Hard address match | Soft similarity matching |
| Read operation | Weighted value retrieval |

### Write and Read Operations

**Write** $(k, v)$: Add a key-value pair to the memory bank
- Keys act as "addresses" that content can be retrieved by
- Values contain the actual information to be retrieved

**Read** with query $q$:
$$
\text{output} = \sum_j \text{similarity}(q, k_j) \cdot v_j
$$

This is content-addressable: we retrieve based on *what* we're looking for, not *where* it is stored.

### Why This Matters

1. **Differentiable**: Unlike discrete memory lookup, gradients flow through the entire operation
2. **Parallel**: All memory locations are accessed simultaneously
3. **Soft selection**: Related content can be blended, capturing fuzzy relationships
4. **Learnable**: The model learns what to store (values) and how to index it (keys)

---

## Deep Insight: Gradient Flow and Soft Selection

### Competition Among Positions

The softmax creates competition: attention to one position reduces attention to others. If one position dominates:

$$
A_{ij} \approx 1 \text{ for some } j, \quad A_{ik} \approx 0 \text{ for } k \neq j
$$

Gradients flow primarily through the attended position, implementing a **soft form of discrete selection** that remains differentiable.

### The Softmax Jacobian

The gradient of softmax has a beautiful structure:

$$
\frac{\partial \alpha_i}{\partial s_j} = \alpha_i (\delta_{ij} - \alpha_j)
$$

where $\delta_{ij}$ is the Kronecker delta. This means:
- Diagonal terms: $\frac{\partial \alpha_i}{\partial s_i} = \alpha_i(1 - \alpha_i)$ — maximized when $\alpha_i = 0.5$
- Off-diagonal terms: $\frac{\partial \alpha_i}{\partial s_j} = -\alpha_i \alpha_j$ — competition effect

**Insight**: Gradients are strongest for uncertain attention (around 0.5) and weakest for confident attention (near 0 or 1), naturally focusing learning on ambiguous cases.

---

## PyTorch Implementation

### Core Module Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    The fundamental attention mechanism used in Transformers.
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    This implementation supports:
    - Batched computation
    - Multi-head attention (queries have shape [batch, heads, seq, dim])
    - Optional attention masking
    - Optional dropout on attention weights
    """
    
    def __init__(self, dropout: float = 0.0):
        """
        Args:
            dropout: Dropout probability applied to attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Queries of shape (batch, [heads], seq_q, d_k)
            key: Keys of shape (batch, [heads], seq_k, d_k)
            value: Values of shape (batch, [heads], seq_k, d_v)
            mask: Optional mask of shape broadcastable to (batch, [heads], seq_q, seq_k)
                  Elements with mask=0 are masked out (set to -inf before softmax)
            return_weights: Whether to return attention weights
            
        Returns:
            output: Attention output of shape (batch, [heads], seq_q, d_v)
            attention_weights: If return_weights=True, weights of shape 
                              (batch, [heads], seq_q, seq_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: Q @ K^T
        # Shape: (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Replace masked positions with -inf (becomes 0 after softmax)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle case where entire row is masked (softmax produces nan)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute output: weights @ V
        # Shape: (..., seq_q, seq_k) @ (..., seq_k, d_v) -> (..., seq_q, d_v)
        output = torch.matmul(attention_weights, value)
        
        if return_weights:
            return output, attention_weights
        return output, None
```

### Functional Interface

For simpler use cases:

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> torch.Tensor:
    """
    Functional interface for scaled dot-product attention.
    
    Args:
        query: (..., seq_q, d_k)
        key: (..., seq_k, d_k)
        value: (..., seq_k, d_v)
        mask: Optional mask broadcastable to (..., seq_q, seq_k)
        dropout_p: Dropout probability
        training: Whether in training mode (for dropout)
        
    Returns:
        output: (..., seq_q, d_v)
    """
    d_k = query.size(-1)
    
    # Compute scaled attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax and dropout
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    return torch.matmul(attn_weights, value)
```

### Using PyTorch's Built-in Function

PyTorch 2.0+ provides an optimized implementation:

```python
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

# Usage (note: mask convention may differ)
output = torch_sdpa(
    query, key, value,
    attn_mask=mask,        # Optional attention mask
    dropout_p=0.1,         # Dropout probability
    is_causal=False        # Whether to apply causal masking
)
```

The built-in function automatically selects the most efficient implementation (FlashAttention, Memory-Efficient Attention, or standard) based on hardware and input characteristics.

---

## Masking Strategies

### Why Masking?

Masks control which positions can attend to which other positions:
1. **Padding masks**: Ignore padding tokens in variable-length sequences
2. **Causal masks**: Prevent attending to future tokens (autoregressive models)
3. **Custom masks**: Implement sparse attention patterns

### Mask Implementation

```python
def create_attention_masks(batch_size: int, seq_q: int, seq_k: int):
    """Demonstrate different masking strategies."""
    
    # 1. Padding mask: ignore padding tokens
    # Shape: (batch, 1, 1, seq_k) for broadcasting
    pad_mask = torch.ones(batch_size, 1, 1, seq_k)
    pad_mask[:, :, :, -2:] = 0  # Last 2 positions are padding
    
    # 2. Causal mask: only attend to past (for autoregressive)
    # Shape: (1, 1, seq_q, seq_k)
    causal_mask = torch.tril(torch.ones(1, 1, seq_q, seq_k))
    
    # 3. Combined mask: both padding and causal
    combined_mask = pad_mask * causal_mask
    
    # 4. Block-sparse mask: local attention patterns
    block_size = 4
    block_mask = torch.zeros(1, 1, seq_q, seq_k)
    for i in range(0, seq_q, block_size):
        block_mask[0, 0, i:i+block_size, i:i+block_size] = 1
    
    return pad_mask, causal_mask, combined_mask, block_mask
```

### Mask Visualization

```
Causal Mask (seq=4):          Padding Mask (pad last 2):
┌─────────────┐               ┌─────────────┐
│ 1 0 0 0 │               │ 1 1 0 0 │
│ 1 1 0 0 │               │ 1 1 0 0 │
│ 1 1 1 0 │               │ 1 1 0 0 │
│ 1 1 1 1 │               │ 1 1 0 0 │
└─────────────┘               └─────────────┘
```

**Masked positions**: Setting scores to $-\infty$ before softmax ensures these positions receive exactly zero weight after normalization.

---

## Attention Visualization

For a sequence of length 4:

```
Queries (n=4)          Keys (m=4)           Attention Matrix
    ┌───┐                ┌───┐              ┌─────────────┐
q₁  │ ● │                │ ● │ k₁          │ .8 .1 .05.05│ → sums to 1
    ├───┤                ├───┤              ├─────────────┤
q₂  │ ● │    ×           │ ● │ k₂    =     │ .1 .7 .1 .1 │ → sums to 1
    ├───┤                ├───┤              ├─────────────┤
q₃  │ ● │                │ ● │ k₃          │ .2 .2 .4 .2 │ → sums to 1
    ├───┤                ├───┤              ├─────────────┤
q₄  │ ● │                │ ● │ k₄          │ .1 .1 .2 .6 │ → sums to 1
    └───┘                └───┘              └─────────────┘
```

Each row shows where that query position attends.

---

## Scaling Variants

Several alternative scaling approaches have been proposed:

```python
def attention_scaling_variants(Q, K, V, variant='standard'):
    """Compare different scaling approaches."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    if variant == 'standard':
        # Original Transformer scaling
        scores = scores / math.sqrt(d_k)
        
    elif variant == 'cosine':
        # Cosine similarity (L2 normalized)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        
    elif variant == 'scaled_cosine':
        # Scaled cosine (used in some vision transformers)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        tau = nn.Parameter(torch.ones(1) * 0.07)  # Learnable temperature
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / tau
        
    elif variant == 'query_scaling':
        # Scale queries and keys separately (equivalent formulation)
        Q_scaled = Q / math.sqrt(math.sqrt(d_k))
        K_scaled = K / math.sqrt(math.sqrt(d_k))
        scores = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))
    
    return F.softmax(scores, dim=-1) @ V
```

---

## Computational Analysis

### Complexity

For queries of length $n$, keys/values of length $m$, and dimension $d$:

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| $\mathbf{Q}\mathbf{K}^T$ | $O(nmd)$ | $O(nm)$ |
| Scaling | $O(nm)$ | $O(1)$ |
| Softmax | $O(nm)$ | $O(nm)$ |
| $\mathbf{A}\mathbf{V}$ | $O(nmd)$ | $O(nd)$ |
| **Total** | $O(nmd)$ | $O(nm + nd)$ |

For self-attention where $n = m$:
- **Time**: $O(n^2d)$
- **Space**: $O(n^2)$ for attention matrix

The quadratic memory requirement for storing the attention matrix is the primary bottleneck for processing long sequences.

### Numerical Stability

**Softmax Stability**: Direct computation of softmax can overflow. PyTorch's `F.softmax` automatically applies the log-sum-exp trick:

$$
\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_j \exp(x_j - \max(\mathbf{x}))}
$$

For explicit implementation:

```python
def stable_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

---

## Memory-Efficient Variants

For long sequences, standard attention's $O(n^2)$ memory is prohibitive. Several efficient variants exist:

### Chunked Attention

Process attention in chunks to reduce peak memory:

```python
def chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    chunk_size: int = 64
) -> torch.Tensor:
    """
    Compute attention in chunks to reduce memory usage.
    
    Memory: O(chunk_size * seq_len) instead of O(seq_len^2)
    """
    batch, seq_q, d_k = query.shape
    seq_k = key.size(1)
    d_v = value.size(-1)
    
    output = torch.zeros(batch, seq_q, d_v, device=query.device)
    
    for i in range(0, seq_q, chunk_size):
        q_chunk = query[:, i:i+chunk_size]
        
        # Compute attention for this chunk
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        output[:, i:i+chunk_size] = torch.matmul(weights, value)
    
    return output
```

### FlashAttention

FlashAttention (Dao et al., 2022) computes exact attention with $O(n)$ memory using a tiled algorithm that keeps data in fast SRAM. PyTorch 2.0+ automatically uses FlashAttention when available:

```python
# Check if FlashAttention is available
import torch.backends.cuda

if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
    print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")

# Use optimized attention (automatically selects best implementation)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=True,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(query, key, value)
```

---

## Practical Example: Sequence Alignment

```python
def attention_alignment_example():
    """
    Demonstrate attention as a sequence alignment mechanism.
    """
    torch.manual_seed(42)
    
    # Source sequence: "The cat sat on mat"
    # Target query: Looking for "animal"
    source_embeddings = torch.randn(1, 5, 8)  # (batch, seq, dim)
    query_embedding = torch.randn(1, 1, 8)    # (batch, 1, dim)
    
    # Make "cat" (position 1) more similar to query
    source_embeddings[0, 1] = query_embedding[0, 0] + 0.1 * torch.randn(8)
    
    # Compute attention
    attention = ScaledDotProductAttention()
    output, weights = attention(
        query_embedding,      # What we're looking for
        source_embeddings,    # Where we're looking
        source_embeddings     # What we retrieve
    )
    
    print("Attention Alignment Example")
    print("-" * 40)
    print(f"Query shape: {query_embedding.shape}")
    print(f"Source shape: {source_embeddings.shape}")
    print(f"\nAttention weights:")
    
    words = ["The", "cat", "sat", "on", "mat"]
    for i, (word, weight) in enumerate(zip(words, weights[0, 0].tolist())):
        bar = "█" * int(weight * 40)
        print(f"  {word:4s}: {weight:.3f} {bar}")
    
    print(f"\nOutput (weighted combination of source): {output.shape}")


if __name__ == "__main__":
    attention_alignment_example()
```

**Output:**
```
Attention Alignment Example
----------------------------------------
Query shape: torch.Size([1, 1, 8])
Source shape: torch.Size([1, 5, 8])

Attention weights:
  The : 0.089 ███
  cat : 0.542 █████████████████████
  sat : 0.124 ████
  on  : 0.098 ███
  mat : 0.147 █████

Output (weighted combination of source): torch.Size([1, 1, 8])
```

---

## Summary

Scaled dot-product attention provides an elegant and efficient attention mechanism that forms the foundation of all Transformer architectures.

### Key Formulation

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

### Core Design Principles

1. **Dot-product scoring**: Efficient parallel computation via matrix multiplication, leveraging GPU optimization
2. **Scaling by $\frac{1}{\sqrt{d_k}}$**: Prevents gradient vanishing in softmax by maintaining unit variance
3. **Softmax normalization**: Produces valid probability distribution enabling soft selection
4. **Value aggregation**: Weighted combination preserves information through convex combinations

### Deep Insights

1. **Geometric view**: Dot product measures alignment in embedding space — queries find keys pointing in similar directions
2. **Memory view**: Attention implements differentiable content-addressable memory with soft retrieval
3. **Temperature view**: Scaling factor controls attention sharpness, balancing exploration vs. exploitation
4. **Gradient view**: Softmax creates competition among positions while maintaining gradient flow

### Practical Considerations

- Quadratic complexity ($O(n^2)$) limits sequence length
- Masking enables causal and padding constraints
- Memory-efficient variants (FlashAttention) exist for long sequences
- Modern frameworks provide highly optimized implementations

The mechanism is simple, efficient, and highly parallelizable — three properties that made Transformers the dominant architecture in modern deep learning.

---

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.

3. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.

4. Rabe, M. N., & Staats, C. (2021). Self-attention Does Not Need $O(n^2)$ Memory. *arXiv preprint arXiv:2112.05682*.
