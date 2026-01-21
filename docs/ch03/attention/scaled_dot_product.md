# Scaled Dot-Product Attention

## Definition

Scaled dot-product attention is the core attention mechanism in Transformers:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where:
- $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$: Query matrix
- $\mathbf{K} \in \mathbb{R}^{m \times d_k}$: Key matrix  
- $\mathbf{V} \in \mathbb{R}^{m \times d_v}$: Value matrix
- $d_k$: Key/query dimension

## Step-by-Step Computation

### Step 1: Compute Attention Scores

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n \times m}$$

Each entry $S_{ij} = \mathbf{q}_i^T \mathbf{k}_j$ measures the dot-product similarity between query $i$ and key $j$.

**Interpretation**: Higher dot product indicates the query and key point in similar directions in the embedding space.

### Step 2: Scale

$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}}$$

The scaling factor $\sqrt{d_k}$ normalizes for the key dimension.

### Step 3: Apply Softmax (Row-wise)

$$A_{ij} = \frac{\exp(S_{ij} / \sqrt{d_k})}{\sum_{l=1}^{m} \exp(S_{il} / \sqrt{d_k})}$$

Each row of $\mathbf{A}$ becomes a probability distribution over positions:
- $\sum_j A_{ij} = 1$ for all $i$
- $A_{ij} \geq 0$ for all $i, j$

### Step 4: Weighted Sum of Values

$$\mathbf{z}_i = \sum_{j=1}^{m} A_{ij} \mathbf{v}_j$$

Each output is a convex combination of value vectors, weighted by attention.

## Dimensional Analysis

| Matrix | Shape | Description |
|--------|-------|-------------|
| $\mathbf{Q}$ | $(n, d_k)$ | $n$ queries, each $d_k$-dimensional |
| $\mathbf{K}$ | $(m, d_k)$ | $m$ keys, each $d_k$-dimensional |
| $\mathbf{V}$ | $(m, d_v)$ | $m$ values, each $d_v$-dimensional |
| $\mathbf{Q}\mathbf{K}^T$ | $(n, m)$ | Attention scores |
| $\mathbf{A}$ | $(n, m)$ | Attention weights |
| Output | $(n, d_v)$ | $n$ outputs, each $d_v$-dimensional |

For self-attention: $n = m$ (same sequence for queries and keys/values).

## Why Dot Product?

The dot product $\mathbf{q}^T \mathbf{k}$ has desirable properties:

### Geometric Interpretation

$$\mathbf{q}^T \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$$

where $\theta$ is the angle between vectors. High dot product when:
- Vectors point in similar directions (small $\theta$)
- Vectors have large magnitudes

### Computational Efficiency

Dot products can be computed as matrix multiplication, which is highly optimized on GPUs:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))  # Single GEMM operation
```

### Comparison to Alternatives

| Attention Type | Scoring Function | Complexity |
|----------------|------------------|------------|
| Dot-product | $\mathbf{q}^T \mathbf{k}$ | $O(d)$ |
| Additive | $\mathbf{w}^T \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | $O(d)$ + nonlinearity |
| Multiplicative | $\mathbf{q}^T \mathbf{W} \mathbf{k}$ | $O(d^2)$ |

Scaled dot-product is simplest and fastest, performing comparably to additive attention with proper scaling.

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Queries (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
        K: Keys (batch, seq_len, d_k) or (batch, heads, seq_len, d_k)
        V: Values (batch, seq_len, d_v) or (batch, heads, seq_len, d_v)
        mask: Optional mask (broadcastable to attention shape)
    
    Returns:
        output: Attention output
        attention_weights: Attention distribution
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax over last dimension (keys)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

## Visualization

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

## Properties

### Softmax Temperature

The scaling factor acts like inverse temperature:

$$\text{softmax}\left(\frac{\mathbf{s}}{\sqrt{d_k}}\right) = \text{softmax}\left(\frac{\mathbf{s}}{T}\right) \text{ with } T = \sqrt{d_k}$$

- High temperature ($T$ large): Uniform attention
- Low temperature ($T$ small): Concentrated attention

### Gradient Flow

The softmax creates competition among positions. If one position dominates:

$$A_{ij} \approx 1 \text{ for some } j, \quad A_{ik} \approx 0 \text{ for } k \neq j$$

Gradients flow primarily through the attended position, implementing a soft form of discrete selection.

### Memory as Attention

Attention can be viewed as content-addressable memory:

| Memory Operation | Attention Equivalent |
|------------------|---------------------|
| Write $(k, v)$ | Add key-value pair |
| Read with query $q$ | $\sum_j \text{similarity}(q, k_j) \cdot v_j$ |

## Numerical Stability

For numerical stability, use the log-sum-exp trick:

```python
def stable_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

PyTorch's `F.softmax` handles this internally.

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| $\mathbf{Q}\mathbf{K}^T$ | $O(n \cdot m \cdot d_k)$ | $O(n \cdot m)$ |
| Softmax | $O(n \cdot m)$ | $O(n \cdot m)$ |
| $\mathbf{A}\mathbf{V}$ | $O(n \cdot m \cdot d_v)$ | $O(n \cdot d_v)$ |
| **Total** | $O(n \cdot m \cdot d)$ | $O(n \cdot m)$ |

For self-attention ($n = m$): $O(n^2 d)$ time, $O(n^2)$ space for the attention matrix.

## Summary

Scaled dot-product attention:

1. **Computes similarity** between queries and keys via dot product
2. **Scales** to prevent softmax saturation
3. **Normalizes** via softmax to get a probability distribution
4. **Aggregates** values according to attention weights

The mechanism is simple, efficient, and highly parallelizable—the foundation of all Transformer architectures.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
