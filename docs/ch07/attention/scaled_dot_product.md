# Scaled Dot-Product Attention

## Introduction

Scaled dot-product attention is the fundamental building block of the Transformer architecture. Introduced by Vaswani et al. (2017) in "Attention Is All You Need," this mechanism provides an efficient and effective way to compute attention weights using matrix operations that can be highly parallelized on modern hardware.

The key innovation lies in combining the computational efficiency of dot-product attention with a scaling factor that ensures stable gradients during training. This seemingly simple modification was crucial for enabling the training of deep attention-based models.

## Mathematical Formulation

### The Attention Function

Given queries $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, keys $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, and values $\mathbf{V} \in \mathbb{R}^{m \times d_v}$, scaled dot-product attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### Step-by-Step Computation

**Step 1: Compute Raw Attention Scores**

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times m}$$

Each element $s_{ij}$ measures the similarity between query $i$ and key $j$:

$$s_{ij} = \mathbf{q}_i^T \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

Higher dot product indicates the query and key point in similar directions in the embedding space.

**Step 2: Apply Scaling**

$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ is critical for training stability (see detailed analysis below).

**Step 3: Apply Softmax (Row-wise)**

$$\alpha_{ij} = \frac{\exp(s_{ij}^{\text{scaled}})}{\sum_{l=1}^{m} \exp(s_{il}^{\text{scaled}})}$$

Each row of $\mathbf{A}$ becomes a probability distribution over positions:
- $\sum_j A_{ij} = 1$ for all $i$
- $A_{ij} \geq 0$ for all $i, j$

**Step 4: Weighted Sum of Values**

$$\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{n \times d_v}$$

Each output row is a convex combination of value vectors, weighted by attention:

$$\mathbf{o}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j$$

### Dimensional Analysis

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

## Why Dot Product?

The dot product $\mathbf{q}^T \mathbf{k}$ has desirable properties that make it ideal for measuring relevance.

### Geometric Interpretation

$$\mathbf{q}^T \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$$

where $\theta$ is the angle between vectors. High dot product when vectors point in similar directions (small $\theta$) or have large magnitudes. This provides an intuitive notion of similarity: queries and keys that are "aligned" in embedding space produce high attention scores.

### Computational Efficiency

Dot products can be computed as matrix multiplication, which is highly optimized on GPUs:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # Single GEMM operation
```

This enables massive parallelization across all query-key pairs simultaneously.

### Comparison to Alternatives

| Attention Type | Scoring Function | Complexity | Notes |
|----------------|------------------|------------|-------|
| Dot-product | $\mathbf{q}^T \mathbf{k}$ | $O(d)$ | Simplest, fastest |
| Additive (Bahdanau) | $\mathbf{v}^T \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $O(d)$ + nonlinearity | More expressive, slower |
| Multiplicative | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | $O(d^2)$ | Learnable interaction |

Scaled dot-product with learned projections (as in multi-head attention) achieves similar expressiveness to additive attention while being significantly faster due to hardware optimization of matrix multiplication.

## The Critical Role of Scaling

### The Variance Explosion Problem

Consider two random vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ where each component is independently drawn from a distribution with mean $\mu = 0$ and variance $\sigma^2 = 1$.

**Mean of the dot product:**
$$\mathbb{E}[\mathbf{q}^T \mathbf{k}] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

**Variance of the dot product:**
$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$$

The variance **scales linearly with dimension**:
$$\mathbf{q}^T \mathbf{k} \sim \mathcal{N}(0, d_k)$$

For typical Transformer dimensions:

| $d_k$ | $\sqrt{d_k}$ | Typical score range ($\pm 2\sigma$) |
|-------|--------------|-------------------------------------|
| 16 | 4 | [-8, 8] |
| 64 | 8 | [-16, 16] |
| 128 | 11.3 | [-22.6, 22.6] |
| 512 | 22.6 | [-45.2, 45.2] |

### Softmax Saturation

When inputs to softmax have large magnitude, the distribution becomes nearly one-hot:

$$\text{softmax}([100, 0, 0]) \approx [1.0, 0.0, 0.0]$$

This causes:

1. **Vanishing gradients**: $\frac{\partial \text{softmax}}{\partial x} \approx 0$ in saturated regions
2. **Loss of information**: Nuanced attention patterns collapse to hard selection
3. **Training instability**: Gradients become noisy and unreliable

### The Softmax Jacobian

The gradient of softmax has the structure:

$$\frac{\partial \alpha_i}{\partial s_j} = \alpha_i (\delta_{ij} - \alpha_j)$$

When softmax saturates ($\alpha_1 \approx 1$, others $\approx 0$):

$$\frac{\partial \alpha_1}{\partial s_j} \approx 1 \cdot (1 - 1) = 0 \quad \text{for } j = 1$$
$$\frac{\partial \alpha_1}{\partial s_j} \approx 1 \cdot (0 - 0) = 0 \quad \text{for } j \neq 1$$

**All gradients vanish.** The model cannot learn to adjust attention weights.

Gradients are strongest for uncertain attention (around 0.5) and weakest for confident attention (near 0 or 1), naturally focusing learning on ambiguous cases.

### The Scaling Solution

Dividing by $\sqrt{d_k}$ normalizes the variance:

$$\text{Var}\left[\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1$$

This keeps scores in a moderate range (typically $[-3, 3]$) where softmax gradients are healthy.

### Numerical Example

Let $d_k = 512$. Consider attention scores: $\mathbf{s} = (20, 22, 18, 21)$.

**Unscaled softmax:**
$$\text{softmax}(20, 22, 18, 21) \approx (0.018, 0.731, 0.002, 0.249)$$

**Scaled (dividing by $\sqrt{512} \approx 22.6$):**
$$\mathbf{s}_{\text{scaled}} = (0.88, 0.97, 0.80, 0.93)$$
$$\text{softmax}(0.88, 0.97, 0.80, 0.93) \approx (0.227, 0.249, 0.210, 0.314)$$

The scaled version has a much smoother distribution, allowing meaningful gradients for all positions.

### Why $\sqrt{d_k}$ Specifically?

| Scaling Factor | Resulting Variance | Effect |
|----------------|-------------------|--------|
| None | $d_k$ | Softmax saturates |
| $d_k$ | $1/d_k$ | Scores too small, near-uniform attention |
| $\sqrt{d_k}$ | $1$ | Just right—unit variance |

The $\sqrt{d_k}$ choice is elegant because it normalizes to unit variance assuming standard initialization.

### Temperature Interpretation

The scaling factor acts like inverse temperature in statistical mechanics:

$$\text{softmax}\left(\frac{\mathbf{s}}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{\mathbf{s}}{T}\right) \text{ with } T = \sqrt{d_k}$$

- **High temperature** ($T$ large): Uniform attention, exploration
- **Low temperature** ($T$ small): Concentrated attention, exploitation

Scaling provides a temperature that adapts to the dimensionality—larger models get higher temperature automatically, preventing saturation as model size grows.

### The Vanishing Gradient Cascade

In a Transformer, gradients must flow through multiple components: output projection → value aggregation → attention weights (softmax) → score computation → Q/K projections → layer normalization → residual connections → previous layers. If attention weights are saturated, gradients cannot effectively propagate through the attention mechanism, creating a bottleneck that stalls learning throughout the entire network.

## Attention as Soft Content-Addressable Memory

Attention can be viewed as a differentiable memory system:

| Traditional Memory | Attention Equivalent |
|--------------------|---------------------|
| Memory address | Key vector $\mathbf{k}$ |
| Memory content | Value vector $\mathbf{v}$ |
| Query/lookup | Query vector $\mathbf{q}$ |
| Hard address match | Soft similarity matching |
| Read operation | Weighted value retrieval |

**Write** $(k, v)$: Add a key-value pair to the memory bank.

**Read** with query $q$: $\text{output} = \sum_j \text{similarity}(q, k_j) \cdot v_j$

This is content-addressable: we retrieve based on *what* we're looking for, not *where* it is stored. Unlike discrete memory lookup, gradients flow through the entire operation, enabling end-to-end learning.

## Gradient Flow and Soft Selection

### Competition Among Positions

The softmax creates competition: attention to one position reduces attention to others. If one position dominates ($A_{ij} \approx 1$ for some $j$), gradients flow primarily through the attended position, implementing a **soft form of discrete selection** that remains differentiable.

### When Scaling is Particularly Important

**Large embedding dimensions.** Modern models use large dimensions:

| Model | $d_{\text{model}}$ | $d_k$ (per head) |
|-------|-------------------|------------------|
| BERT-base | 768 | 64 |
| GPT-2 | 768-1600 | 64 |
| GPT-3 | 12288 | 128 |

Without scaling, scores would have standard deviations of 8-11, pushing softmax into saturation.

**Deep networks.** In deep Transformers (12+ layers), gradient flow is critical. Saturated attention in early layers creates severe gradient bottlenecks.

**Architecture scalability.** Scaling ensures consistent softmax behavior across different model sizes, allowing architectures to be scaled without retuning.

## PyTorch Implementation

### Core Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Supports batched computation, multi-head attention,
    optional masking, and dropout on attention weights.
    """
    
    def __init__(self, dropout: float = 0.0):
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
        Args:
            query: (..., seq_q, d_k)
            key: (..., seq_k, d_k)
            value: (..., seq_k, d_v)
            mask: Broadcastable to (..., seq_q, seq_k). 0 → masked out.
            
        Returns:
            output: (..., seq_q, d_v)
            attention_weights: (..., seq_q, seq_k) if return_weights
        """
        d_k = query.size(-1)
        
        # Score: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        if return_weights:
            return output, attention_weights
        return output, None
```

### Functional Interface

```python
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional scaled dot-product attention.
    
    Args:
        query: (..., seq_q, d_k)
        key: (..., seq_k, d_k)
        value: (..., seq_k, d_v)
        mask: Broadcastable to (..., seq_q, seq_k)
        
    Returns:
        output: (..., seq_q, d_v)
        weights: (..., seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p, training=True)
    
    return torch.matmul(weights, value), weights
```

### Using PyTorch's Built-in Function

PyTorch 2.0+ provides an optimized implementation:

```python
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa

output = torch_sdpa(
    query, key, value,
    attn_mask=mask,
    dropout_p=0.1,
    is_causal=False
)
```

The built-in function automatically selects the most efficient implementation (FlashAttention, Memory-Efficient Attention, or standard) based on hardware and input characteristics.

### Empirical Verification of Scaling

```python
def verify_scaling_effect():
    """Empirically verify that scaling normalizes variance."""
    torch.manual_seed(42)
    
    dims = [16, 64, 256, 512, 1024]
    num_samples = 10000
    
    print("Empirical verification of dot-product variance:")
    print("-" * 70)
    print(f"{'d_k':>6} | {'Unscaled Var':>12} | {'Scaled Var':>12} | {'sqrt(d_k)':>10}")
    print("-" * 70)
    
    for d_k in dims:
        Q = torch.randn(num_samples, d_k)
        K = torch.randn(num_samples, d_k)
        
        unscaled = (Q * K).sum(dim=1)
        scaled = unscaled / (d_k ** 0.5)
        
        print(f"{d_k:>6} | {unscaled.var().item():>12.2f} | "
              f"{scaled.var().item():>12.4f} | {d_k**0.5:>10.2f}")


verify_scaling_effect()
```

**Output:**
```
Empirical verification of dot-product variance:
----------------------------------------------------------------------
   d_k | Unscaled Var |   Scaled Var |   sqrt(d_k)
----------------------------------------------------------------------
    16 |        16.05 |       1.0032 |       4.00
    64 |        63.81 |       0.9970 |       8.00
   256 |       257.42 |       1.0055 |      16.00
   512 |       513.68 |       1.0034 |      22.63
  1024 |      1026.39 |       1.0024 |      32.00
----------------------------------------------------------------------
```

## Masking Strategies

Masks control which positions can attend to which other positions:

### Mask Types

1. **Padding masks**: Ignore padding tokens in variable-length sequences
2. **Causal masks**: Prevent attending to future tokens (autoregressive models)
3. **Custom masks**: Implement sparse attention patterns

### Implementation

```python
def create_attention_masks(batch_size: int, seq_q: int, seq_k: int):
    """Demonstrate different masking strategies."""
    
    # Padding mask: (batch, 1, 1, seq_k)
    pad_mask = torch.ones(batch_size, 1, 1, seq_k)
    pad_mask[:, :, :, -2:] = 0  # Last 2 positions are padding
    
    # Causal mask: (1, 1, seq_q, seq_k)
    causal_mask = torch.tril(torch.ones(1, 1, seq_q, seq_k))
    
    # Combined mask
    combined_mask = pad_mask * causal_mask
    
    return pad_mask, causal_mask, combined_mask
```

```
Causal Mask (seq=4):          Padding Mask (pad last 2):
┌─────────────┐               ┌─────────────┐
│ 1 0 0 0 │               │ 1 1 0 0 │
│ 1 1 0 0 │               │ 1 1 0 0 │
│ 1 1 1 0 │               │ 1 1 0 0 │
│ 1 1 1 1 │               │ 1 1 0 0 │
└─────────────┘               └─────────────┘
```

## Alternative Scaling Strategies

### Query-Key Balanced Scaling

Scale queries and keys separately instead of scaling the final scores:

$$\text{score} = \left(\frac{\mathbf{q}}{d_k^{1/4}}\right)^T \left(\frac{\mathbf{k}}{d_k^{1/4}}\right) = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}$$

Mathematically equivalent, potentially better numerical properties:

```python
class BalancedScaledAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = d_k ** (-0.25)  # Fourth root
    
    def forward(self, Q, K, V, mask=None):
        Q_scaled = Q * self.scale
        K_scaled = K * self.scale
        scores = torch.matmul(Q_scaled, K_scaled.transpose(-2, -1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

### Learnable Temperature

Some models use a learnable scaling factor:

```python
class TemperatureScaledAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(d_k)))
    
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

### Cosine Similarity

Normalizing queries and keys to unit vectors bounds scores to $[-1, 1]$ regardless of dimension:

```python
class CosineAttention(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, Q, K, V, mask=None):
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1) @ V
```

## Computational Analysis

For queries of length $n$, keys/values of length $m$, and dimension $d$:

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| $\mathbf{Q}\mathbf{K}^T$ | $O(nmd)$ | $O(nm)$ |
| Scaling | $O(nm)$ | $O(1)$ |
| Softmax | $O(nm)$ | $O(nm)$ |
| $\mathbf{A}\mathbf{V}$ | $O(nmd)$ | $O(nd)$ |
| **Total** | $O(nmd)$ | $O(nm + nd)$ |

For self-attention where $n = m$: $O(n^2d)$ time, $O(n^2)$ space. The quadratic memory requirement for storing the attention matrix is the primary bottleneck for processing long sequences.

### Numerical Stability

PyTorch's `F.softmax` automatically applies the log-sum-exp trick:

$$\text{softmax}(\mathbf{x})_i = \frac{\exp(x_i - \max(\mathbf{x}))}{\sum_j \exp(x_j - \max(\mathbf{x}))}$$

## Memory-Efficient Variants

### Chunked Attention

Process attention in chunks to reduce peak memory:

```python
def chunked_attention(query, key, value, chunk_size=64):
    """Memory: O(chunk_size * seq_len) instead of O(seq_len^2)."""
    batch, seq_q, d_k = query.shape
    d_v = value.size(-1)
    output = torch.zeros(batch, seq_q, d_v, device=query.device)
    
    for i in range(0, seq_q, chunk_size):
        q_chunk = query[:, i:i+chunk_size]
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        output[:, i:i+chunk_size] = torch.matmul(weights, value)
    
    return output
```

### FlashAttention

FlashAttention (Dao et al., 2022) computes exact attention with $O(n)$ memory using a tiled algorithm that keeps data in fast SRAM. PyTorch 2.0+ automatically uses FlashAttention when available:

```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=True,
    enable_mem_efficient=True
):
    output = F.scaled_dot_product_attention(query, key, value)
```

## High-Dimensional Geometry

### Concentration of Measure

In high dimensions, interesting geometric phenomena occur:

- **Most volume near the surface**: Nearly all the volume of a high-dimensional sphere is concentrated near its surface
- **Near-orthogonality**: Random vectors are nearly orthogonal with high probability
- **Concentration around mean**: Dot products concentrate around their expected value

For random unit vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:
$$\mathbb{E}[\mathbf{u}^T \mathbf{v}] = 0, \quad \text{Var}(\mathbf{u}^T \mathbf{v}) = \frac{1}{d}$$

Without scaling, small differences in alignment get magnified by the growing variance, and softmax sees dramatically different score ranges for different dimensions. With scaling, variance is normalized to a constant regardless of dimension, providing consistent behavior across model sizes.

## Summary

### Key Formulation

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### Core Design Principles

1. **Dot-product scoring**: Efficient parallel computation via matrix multiplication, leveraging GPU optimization
2. **Scaling by $\frac{1}{\sqrt{d_k}}$**: Prevents gradient vanishing in softmax by maintaining unit variance
3. **Softmax normalization**: Produces valid probability distribution enabling soft selection
4. **Value aggregation**: Weighted combination preserves information through convex combinations

### Deep Insights

1. **Geometric view**: Dot product measures alignment in embedding space—queries find keys pointing in similar directions
2. **Memory view**: Attention implements differentiable content-addressable memory with soft retrieval
3. **Temperature view**: Scaling factor controls attention sharpness, balancing exploration vs. exploitation
4. **Gradient view**: Softmax creates competition among positions while maintaining gradient flow

| Aspect | Without Scaling | With $\sqrt{d_k}$ Scaling |
|--------|-----------------|--------------------------|
| Score variance | $d_k$ (grows with dimension) | $1$ (stable) |
| Softmax behavior | Saturates | Smooth gradients |
| Attention distribution | Near one-hot | Distributed |
| Learning | Vanishing gradients | Stable training |

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.
4. Rabe, M. N., & Staats, C. (2021). "Self-attention Does Not Need $O(n^2)$ Memory." *arXiv:2112.05682*.
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*.
6. Noci, L., et al. (2022). "Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse." *NeurIPS*.
