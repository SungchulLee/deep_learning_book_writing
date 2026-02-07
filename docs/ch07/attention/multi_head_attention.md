# Multi-Head Attention

## Overview

Multi-head attention allows the model to jointly attend to information from **different representation subspaces** at different positions. Instead of a single attention function, we compute $h$ parallel attention operations, each with its own learned projections.

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

where each head is:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(i)}, \mathbf{X}\mathbf{W}_K^{(i)}, \mathbf{X}\mathbf{W}_V^{(i)})$$

## Motivation: The Single-Head Bottleneck

A fundamental limitation exists in single-head attention: no matter how large $d_{\text{model}}$ is, there is only **one** attention distribution per query position.

### The Rank Constraint

In single-head attention, the attention pattern is determined by:

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q\mathbf{W}_K^T\mathbf{X}^T}{\sqrt{d_k}}\right)$$

The matrix $\mathbf{W}_Q\mathbf{W}_K^T \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ defines a **single bilinear form** for computing relevance.

### Why One Pattern Is Insufficient

Consider: "The animal didn't cross the street because it was too tired."

To resolve "it," the model needs to attend to:

1. **Syntactic antecedent**: "animal" (noun phrase structure)
2. **Semantic constraint**: "tired" (animacy—streets don't get tired)
3. **Local context**: "didn't cross" (action being explained)

A single attention distribution must compromise:
- If it peaks sharply on "animal," it loses information from "tired"
- If it spreads across all relevant tokens, each receives diluted weight

### The Multi-Head Solution

With $h$ heads, we compute $h$ **independent** attention distributions:

$$\mathbf{A}^{(i)} = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q^{(i)}(\mathbf{W}_K^{(i)})^T\mathbf{X}^T}{\sqrt{d_k}}\right)$$

Each head learns a **different notion of relevance**:

- $\text{head}_1$: syntactic dependency (animal)
- $\text{head}_2$: semantic predicate (tired)
- $\text{head}_3$: local verb phrase (didn't cross)

## Architecture

### Dimensions

Given:
- Model dimension: $d_{\text{model}}$
- Number of heads: $h$
- Per-head dimension: $d_k = d_v = d_{\text{model}} / h$

Each head operates on a lower-dimensional subspace.

### Parameter Matrices

For each head $i \in \{1, \ldots, h\}$:

$$\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$
$$\mathbf{W}_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$
$$\mathbf{W}_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$$

Output projection:

$$\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

### Total Parameters

$$\underbrace{h \cdot 3 \cdot d_{\text{model}} \cdot d_k}_{\text{QKV projections}} + \underbrace{d_{\text{model}}^2}_{\mathbf{W}_O} = 4 d_{\text{model}}^2$$

Same parameter count as a single large attention head with dimension $d_{\text{model}}$.

## The Role of $\mathbf{W}_O$

### Why We Need It

The output projection $\mathbf{W}_O$ cannot be absorbed into the FFN because of the **residual connection**:

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots)\mathbf{W}_O)$$

The attention output is **added to $\mathbf{X}$** before FFN. Without $\mathbf{W}_O$:

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{Concat}(\ldots))$$

The FFN's $\mathbf{W}_1$ acts on $\mathbf{X}'$, which already includes the residual. The computation graph separates $\mathbf{W}_O$ from $\mathbf{W}_1$.

### Inter-Head Communication

More importantly, $\mathbf{W}_O$ enables **cross-head mixing**:

**Before $\mathbf{W}_O$**: Heads are completely independent—just vectors stacked side by side.

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_h) = [\mathbf{z}^{(1)} | \mathbf{z}^{(2)} | \ldots | \mathbf{z}^{(h)}]$$

**After $\mathbf{W}_O$**: Information flows between heads:

$$(\text{Concat}(\ldots)\mathbf{W}_O)_i = \sum_{j=1}^{h} \mathbf{z}^{(j)} \mathbf{W}_O^{(j \to \text{out})}$$

This is where the model learns **how to combine different attention patterns**.

### Concrete Example

Suppose:
- Head 1 found syntactic information: "the subject is 'cat'"
- Head 2 found semantic information: "the predicate implies animacy"
- Head 3 found positional information: "nearby tokens are about movement"

**Before $\mathbf{W}_O$** (concatenation):
$$[\text{syntax} | \text{semantics} | \text{position}]$$

**After $\mathbf{W}_O$** (learned mixing):
$$\text{output} = 0.5 \cdot \text{syntax} + 0.3 \cdot \text{semantics} + 0.2 \cdot \text{position}$$

## Information-Theoretic Perspective

### Single-Head: One Point in Convex Hull

The attention output for a single head:

$$\mathbf{z}_i = \sum_{j=1}^{n} A_{ij} \mathbf{v}_j$$

Since $\sum_j A_{ij} = 1$, this is a point in the **convex hull** of $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$.

### Multi-Head: Richer Subspace

Each head reaches a **different point** in its respective (lower-dimensional) convex hull:

$$\mathbf{z}_i = \left[\sum_j A_{ij}^{(1)} \mathbf{v}_j^{(1)}; \ldots; \sum_j A_{ij}^{(h)} \mathbf{v}_j^{(h)}\right]\mathbf{W}_O$$

The concatenation spans a **much richer subspace**, and $\mathbf{W}_O$ can produce **any linear combination** of these $h$ different contextual summaries.

## PyTorch Implementation

### Multi-Head Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention
    
    Splits the embedding dimension into multiple heads, allowing the model
    to attend to information from different representation subspaces.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection (enables inter-head communication)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention for all heads in parallel
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


def demonstrate_multi_head():
    """Demonstrate multi-head self-attention."""
    batch_size, seq_len, d_model, n_heads = 2, 10, 64, 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model, n_heads)
    
    output, weights = mha(x)
    
    print("Multi-Head Attention Demonstration")
    print("-" * 40)
    print(f"Input:    {x.shape}")
    print(f"Output:   {output.shape}")
    print(f"Weights:  {weights.shape}")
    print(f"Heads:    {n_heads}")
    print(f"Head dim: {d_model // n_heads}")


if __name__ == "__main__":
    demonstrate_multi_head()
```

### General Multi-Head Attention (for Cross-Attention)

```python
class MultiHeadAttention(nn.Module):
    """
    General Multi-Head Attention
    
    Can be used for self-attention (Q=K=V=X) or cross-attention (Q from decoder, K/V from encoder).
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query source (batch, seq_q, d_model)
            K: Key source (batch, seq_k, d_model)
            V: Value source (batch, seq_v, d_model), usually seq_k == seq_v
            mask: Optional attention mask
        """
        batch_size = Q.size(0)
        seq_q, seq_k = Q.size(1), K.size(1)
        
        # Project and reshape
        Q = self.W_q(Q).view(batch_size, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        
        # Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights
```

## Empirical Head Specialization

Research analyzing trained Transformers confirms functional specialization:

| Head Type | Pattern | Example |
|-----------|---------|---------|
| **Positional heads** | Attend to previous/next token | "The [cat]" → attend to "The" |
| **Syntactic heads** | Capture subject-verb dependencies | "cats [run]" → attend to "cats" |
| **Coreference heads** | Track pronoun-antecedent relationships | "[it] was tired" → attend to "animal" |
| **Copy heads** | Focus on rare words, proper nouns | Names, technical terms |
| **Separator heads** | Attend to punctuation, sentence boundaries | Periods, [SEP] tokens |

Ablation studies show that removing individual heads degrades specific capabilities while leaving others intact.

## Not Just Parallelization

A common misconception: multi-head attention is simply parallel computation of the same thing.

**Single-head** computes **one** attention-weighted combination per position.

**Multi-head** computes **$h$ independent** combinations, then **learns to mix** them optimally.

The expressive power difference is substantial: multi-head attention can represent functions that single-head cannot, regardless of dimensionality.

## Computational Efficiency

Multi-head attention has the same theoretical complexity as single-head:

| Operation | Single Head | Multi-Head |
|-----------|-------------|------------|
| Projections | $3 \cdot n \cdot d^2$ | $3 \cdot n \cdot d^2$ |
| Attention | $n^2 \cdot d$ | $h \cdot n^2 \cdot (d/h) = n^2 \cdot d$ |
| Output projection | $n \cdot d^2$ | $n \cdot d^2$ |

The parallel structure is highly GPU-friendly—all heads compute simultaneously with a single batched matrix multiplication.

## Hyperparameter: Number of Heads

Common configurations:

| Model | $d_{\text{model}}$ | Heads | $d_k$ |
|-------|-------------------|-------|-------|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 (175B) | 12288 | 96 | 128 |
| LLaMA-7B | 4096 | 32 | 128 |

**Rule of thumb**: $d_k = 64$ or $d_k = 128$ is common; adjust heads to match $d_{\text{model}}$.

## Variants

### Grouped-Query Attention (GQA)

Used in LLaMA-2 and other efficient models. Multiple query heads share key-value heads:

```python
class GroupedQueryAttention(nn.Module):
    """
    GQA: n_heads query heads share n_kv_heads key-value heads.
    Reduces KV cache size during inference.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # Full heads for Q
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)  # Reduced for K
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)  # Reduced for V
        self.W_o = nn.Linear(d_model, d_model)
```

### Multi-Query Attention (MQA)

Extreme case: all query heads share a single KV head ($n_{kv} = 1$).

## Summary

Multi-head attention provides:

1. **Multiple attention patterns**: Capture fundamentally different relationships (syntactic, semantic, positional)
2. **Subspace decomposition**: Specialized feature extraction per head
3. **Learned combination**: $\mathbf{W}_O$ integrates diverse perspectives via cross-head mixing
4. **Functional specialization**: Emerges naturally from training
5. **Same computational cost**: Parameter count equals single large head

**Key insight**: A single attention distribution cannot simultaneously represent syntactic, semantic, and positional relationships. Multi-head attention solves this fundamental limitation while maintaining computational efficiency.

The output projection $\mathbf{W}_O$ is not just a formality—it's where the model learns to combine the different "views" each head provides into a unified representation.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

2. Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL*.

3. Clark, K., et al. (2019). What Does BERT Look At? An Analysis of BERT's Attention. *BlackboxNLP*.

4. Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. *arXiv*.
