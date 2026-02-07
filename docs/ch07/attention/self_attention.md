# Self-Attention

## Introduction

Self-attention is a special case of attention where queries, keys, and values all derive from the **same input sequence**. Each position in the sequence can attend to all positions (including itself), enabling the model to capture relationships and dependencies within a single sequence without any external context.

Self-attention is the core mechanism that gives Transformers their power—forming the foundation for both understanding architectures (BERT, RoBERTa) and generation architectures (GPT). It revolutionized how we model sequential data by replacing sequential processing with parallel, content-based interactions.

## Mathematical Formulation

### Self-Attention Definition

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$ with $n$ positions and embedding dimension $d$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

$$\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{W}^Q, \mathbf{W}^K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}^V \in \mathbb{R}^{d \times d_v}$.

**The key distinction from general attention: Q, K, V all come from the same source X.**

### Why $\sqrt{d_k}$ Scaling?

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing too large in magnitude. When $d_k$ is large, the dot products $\mathbf{q}_i^T \mathbf{k}_j$ tend to have variance proportional to $d_k$ (assuming components are independent with zero mean and unit variance). Large dot products push the softmax into regions of extremely small gradients, effectively making it behave like a hard argmax and stalling learning.

To see this formally, if $q_i, k_j \sim \mathcal{N}(0, 1)$ independently, then:

$$\text{Var}(\mathbf{q}^T \mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Dividing by $\sqrt{d_k}$ normalizes the variance to 1, keeping the softmax in a well-behaved regime.

### Semantic Interpretation of Q, K, V

Each token generates three distinct representations:

| Projection | Semantic Role | Intuition |
|------------|---------------|-----------|
| **Query** $\mathbf{q}_i$ | "What information am I looking for?" | The question a token asks |
| **Key** $\mathbf{k}_i$ | "What information do I offer?" | How a token advertises its content |
| **Value** $\mathbf{v}_i$ | "What will I contribute if selected?" | The actual information to aggregate |

This separation allows a token to seek different information than it provides—critical for modeling asymmetric relationships like subject-verb dependencies.

### Attention Matrix Interpretation

The attention matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ has element $a_{ij}$ representing how much position $i$ attends to position $j$:

$$a_{ij} = \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})}{\sum_{l=1}^{n} \exp(\mathbf{q}_i^T \mathbf{k}_l / \sqrt{d_k})}$$

**Properties:**

- Each row sums to 1 (valid probability distribution over source positions)
- $a_{ii}$ represents self-attention (attending to the same position)
- The matrix is generally **not symmetric** ($a_{ij} \neq a_{ji}$)
- The matrix is **square**—every position attends to every position

### Visualizing the Attention Matrix

For the sentence "The cat sat":

```
         Keys
         The  cat  sat
        ┌────┬────┬────┐
    The │ .6 │ .2 │ .2 │  Query "The" attends mostly to itself
        ├────┼────┼────┤
Q   cat │ .1 │ .7 │ .2 │  Query "cat" attends mostly to itself
        ├────┼────┼────┤
    sat │ .2 │ .5 │ .3 │  Query "sat" attends to "cat" (subject)
        └────┴────┴────┘
```

Each row is a probability distribution (sums to 1). The asymmetry shows that "sat" strongly attends to "cat" (finding its subject), but "cat" doesn't equally attend to "sat".

## Why Self-Attention Enables Global Context

### The Long-Range Dependency Problem

Consider: *"The cat sat on the mat because it was soft."*

To resolve "it" (position 8) to "mat" (position 6), the model needs to connect these distant positions.

**Traditional RNN approach** (sequential processing):
- Information from "mat" must propagate through positions 7, then to 8
- Path length: $O(|i - j|)$
- Gradients must flow through many steps, leading to vanishing/exploding gradients

**Self-attention approach** (parallel processing):
- Position 8 ("it") directly attends to position 6 ("mat")
- Path length: $O(1)$ — constant, regardless of distance
- Direct gradient flow between any two positions

### Path Length Comparison

| Architecture | Path Length (positions $i$ to $j$) | Parallelization |
|--------------|-----------------------------------|-----------------|
| RNN/LSTM | $O(\|i - j\|)$ | Sequential |
| CNN | $O(\log_{k}\|i - j\|)$ with kernel $k$ | Parallel |
| Self-Attention | $O(1)$ | Fully parallel |

Short paths enable better gradient flow for learning long-range dependencies. This is why Transformers excel at tasks requiring global understanding.

## Self-Attention vs Cross-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|----------------|-----------------|
| Q source | Same sequence X | Decoder sequence |
| K, V source | Same sequence X | Encoder sequence |
| Purpose | Internal context modeling | External reference/grounding |
| Attention shape | Square $(n \times n)$ | Rectangular $(n_q \times n_k)$ |
| Typical use | Within encoder or decoder | Decoder attending to encoder |

Self-attention captures relationships **within** a sequence; cross-attention bridges **between** sequences.

## Bidirectional vs Causal Self-Attention

### Bidirectional (Encoder-style)

Every position can see every other position—full context in both directions:

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**Use cases:** Understanding tasks where full context is available (BERT, sentence classification, NER, question answering)

### Causal/Masked (Decoder-style)

Position $i$ can only attend to positions $1, \ldots, i$ (past and present, not future):

$$\mathbf{A} = \begin{pmatrix} a_{11} & 0 & 0 \\ a_{21} & a_{22} & 0 \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

**Use cases:** Generation tasks where we predict the next token autoregressively (GPT, language modeling, text generation)

The causal mask is implemented by setting future positions to $-\infty$ before softmax:

$$\text{scores}_{ij} = \begin{cases} \mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### Masking in Multi-Head Attention

When causal masking is applied in a multi-head setting, the same mask is broadcast across all heads. Each head independently learns different attention patterns, but they all respect the same causal constraint. This means head $h$ computes:

$$\text{head}_h = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}_h$$

where $\mathbf{M}_{ij} = 0$ if $j \leq i$ and $\mathbf{M}_{ij} = -\infty$ otherwise. The additive formulation (rather than multiplicative) ensures clean gradient flow through the softmax.

### Encoder-Decoder Masking Patterns

In a full encoder-decoder Transformer, three distinct masking patterns coexist:

| Component | Masking | Reason |
|-----------|---------|--------|
| Encoder self-attention | None (bidirectional) | Source is fully available |
| Decoder self-attention | Causal mask | Preserve autoregressive property |
| Decoder cross-attention | None (over encoder) | All source positions accessible |

The decoder additionally uses a **padding mask** for both self-attention and cross-attention to ignore padding tokens.

## Fundamental Properties

### Permutation Equivariance

If we permute the input positions, the output is permuted identically:

$$\text{SelfAttn}(\mathbf{P}\mathbf{X}) = \mathbf{P} \cdot \text{SelfAttn}(\mathbf{X})$$

where $\mathbf{P}$ is a permutation matrix.

**Implication:** Self-attention treats positions symmetrically—it has no built-in notion of order. Positional information must be added explicitly via positional encodings (sinusoidal, learned, rotary, etc.).

### No Inductive Bias for Locality or Order

Unlike RNNs (sequential bias) or CNNs (local receptive field), self-attention has no built-in assumptions about:
- Position (which tokens are "close")
- Direction (left vs right context)
- Locality (nearby tokens being more relevant)

This is both a **strength** (flexibility to learn arbitrary patterns) and a **weakness** (requires more data, needs explicit positional encoding).

### Computational Complexity

| Resource | Complexity | Bottleneck |
|----------|------------|------------|
| Time | $O(n^2 d)$ | Quadratic in sequence length |
| Memory | $O(n^2)$ | Storing the attention matrix |

The $O(n^2)$ scaling is the primary limitation for long sequences, motivating efficient variants like Linear Attention, Linformer, Performer, and FlashAttention.

## Self-Attention vs Fully Connected Layers

Self-attention might appear similar to a fully connected layer, but they differ fundamentally:

| Aspect | Fully Connected | Self-Attention |
|--------|-----------------|----------------|
| Weights | Static (fixed after training) | Dynamic (computed from input) |
| Position dependency | Different weights per position | Same Q, K, V matrices for all positions |
| Adaptability | Rigid (same transformation) | Content-adaptive (input-dependent mixing) |
| Parameter count | $O(n^2 d^2)$ for sequence | $O(d^2)$ regardless of sequence length |

**Key insight:** Self-attention computes **dynamic, content-based** mixing weights, while fully connected layers apply **static, learned** transformations.

## What Self-Attention Learns

Research (probing classifiers, attention visualization) has revealed that different layers learn different patterns:

### Early Layers
- Local patterns (adjacent token attention)
- Positional patterns (fixed offsets like "attend to previous word")
- Syntactic basics (punctuation, function words, articles)

### Middle Layers
- Syntactic relationships (subject-verb agreement, modifier-head)
- Coreference resolution (pronoun-antecedent linking)
- Named entity recognition patterns
- Phrase structure

### Later Layers
- Task-specific patterns
- Semantic relationships and reasoning
- Long-range dependencies
- Abstract feature combinations

## PyTorch Implementation

### Basic Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Self-Attention Layer
    
    Computes attention where queries, keys, and values all come from 
    the same input. Used in Transformer encoder layers.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_k: Optional[int] = None, 
        d_v: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = self.d_k ** -0.5
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, d_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
                  0 indicates positions to mask out
            
        Returns:
            output: Self-attended output (batch_size, seq_len, d_model)
            attention_weights: Attention matrix (batch_size, seq_len, seq_len)
        """
        # Project to Q, K, V (all from the same input x)
        Q = self.W_q(x)  # (batch, seq_len, d_k)
        K = self.W_k(x)  # (batch, seq_len, d_k)
        V = self.W_v(x)  # (batch, seq_len, d_v)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over keys (last dimension)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        attended = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


def demonstrate_self_attention():
    """Demonstrate basic self-attention."""
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    self_attn = SelfAttention(d_model)
    X = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = self_attn(X)
    
    print(f"Input shape:     {X.shape}")        # (2, 10, 512)
    print(f"Output shape:    {output.shape}")   # (2, 10, 512)
    print(f"Attention shape: {weights.shape}")  # (2, 10, 10)
    print(f"\nAttention matrix is square: {weights.shape[-2]} x {weights.shape[-1]}")
    print(f"Each row sums to 1: {weights[0, 0].sum().item():.4f}")
```

### Causal Self-Attention (for Autoregressive Models)

```python
class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention with Multi-Head Support
    
    Prevents positions from attending to subsequent positions.
    Used in decoder-only models like GPT for autoregressive generation.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency (single matmul instead of three)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter, but moves with model)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attended output (batch_size, seq_len, embed_dim)
            attention_weights: Optional (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V in one efficient operation
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores: (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (positions can only attend to past and present)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and optional dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        return output, None


def demonstrate_causal_attention():
    """Show causal masking pattern."""
    batch_size, seq_len, embed_dim, num_heads = 1, 5, 64, 4
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    causal_attn = CausalSelfAttention(embed_dim, num_heads)
    
    output, weights = causal_attn(x)
    
    print("Causal Attention Pattern (first head):")
    print("Each row shows what that position attends to.")
    print("Position i can only attend to positions <= i (lower triangular).\n")
    print(weights[0, 0].detach().numpy().round(3))
    print("\nNote: Upper triangle is 0 (future positions masked)")


if __name__ == "__main__":
    demonstrate_causal_attention()
```

**Output:**
```
Causal Attention Pattern (first head):
Each row shows what that position attends to.
Position i can only attend to positions <= i (lower triangular).

[[1.    0.    0.    0.    0.   ]
 [0.423 0.577 0.    0.    0.   ]
 [0.298 0.351 0.351 0.    0.   ]
 [0.221 0.264 0.258 0.257 0.   ]
 [0.178 0.213 0.207 0.201 0.201]]

Note: Upper triangle is 0 (future positions masked)
```

## Comparison with Other Mechanisms

### Self-Attention vs RNNs

| Aspect | Self-Attention | RNN/LSTM |
|--------|---------------|----------|
| Long-range dependencies | $O(1)$ path length | $O(n)$ path length |
| Parallelization | Fully parallel | Sequential (inherently serial) |
| Computation per layer | $O(n^2 d)$ | $O(n d^2)$ |
| Memory | $O(n^2)$ | $O(n)$ |
| Gradient flow | Direct connections | Through recurrent steps |
| Inductive bias | None | Sequential/temporal |

**When to prefer self-attention:** Tasks requiring global context, parallel training, long sequences with important long-range dependencies.

**When to prefer RNNs:** Streaming applications, memory-constrained settings, tasks with strong sequential/temporal structure.

### Self-Attention vs Convolution

| Aspect | Self-Attention | Convolution |
|--------|---------------|-------------|
| Receptive field | Global (full sequence) | Local (kernel size $k$) |
| Parameter sharing | Same Q, K, V everywhere | Same kernel everywhere |
| Inductive bias | None | Translation equivariance, locality |
| Computation | $O(n^2 d)$ | $O(k n d)$ |
| Long-range | Direct | Requires stacking/dilation |

**Insight:** CNNs have strong locality bias (nearby elements interact); self-attention learns which elements should interact regardless of distance.

## Applications

### Transformer Encoder Block (BERT-style)

```python
class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with bidirectional self-attention.
    
    Architecture: Self-Attention → Add & Norm → FFN → Add & Norm
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention (bidirectional)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            src_mask: Attention mask (seq_len, seq_len)
            src_key_padding_mask: Padding mask (batch, seq_len)
        """
        # Self-attention with residual connection (Pre-LN variant)
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### Vision Transformer (ViT) Self-Attention

Self-attention applied to image patches, treating an image as a sequence:

```python
class VisionSelfAttention(nn.Module):
    """
    Self-attention for image patches (Vision Transformer style).
    
    Images are split into patches, flattened, and treated as a sequence.
    Each patch can attend to all other patches globally.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: Flattened image patches (batch, num_patches, embed_dim)
                     Typically includes a [CLS] token as first position
        
        Returns:
            output: Self-attended patches (batch, num_patches, embed_dim)
            weights: Attention weights (batch, num_patches, num_patches)
        """
        # Each patch attends to all other patches (including CLS token)
        attended, weights = self.attention(patches, patches, patches)
        
        # Residual connection
        output = self.norm(patches + attended)
        
        return output, weights
```

### Self-Attention in Different Contexts

| Context | Attention Type | Key Characteristics |
|---------|---------------|---------------------|
| **BERT Encoder** | Bidirectional | Full context, [CLS] token for pooling |
| **GPT Decoder** | Causal | Autoregressive, predicts next token |
| **Vision Transformer** | Bidirectional | Patches as tokens, [CLS] for classification |
| **Audio (Wav2Vec)** | Bidirectional | Raw waveform or spectral features |
| **Protein (ESM)** | Bidirectional | Amino acids as tokens |

## Efficient Self-Attention Variants

The $O(n^2)$ complexity motivates many efficient alternatives:

| Method | Complexity | Key Idea |
|--------|------------|----------|
| **Sparse Attention** | $O(n\sqrt{n})$ | Attend to fixed patterns (local + strided) |
| **Linformer** | $O(n)$ | Low-rank projection of K, V |
| **Performer** | $O(n)$ | Random feature approximation of softmax |
| **Linear Attention** | $O(n)$ | Remove softmax, use kernel trick |
| **FlashAttention** | $O(n^2)$ time, $O(n)$ memory | IO-aware, tiled computation |
| **Sliding Window** | $O(nw)$ | Local attention with window size $w$ |

## Summary

Self-attention is characterized by:

| Property | Description |
|----------|-------------|
| **Source** | Q, K, V all from the same sequence |
| **Scope** | Every position attends to every position |
| **Path length** | $O(1)$ between any two positions |
| **Complexity** | $O(n^2 d)$ time, $O(n^2)$ space |
| **Inductive bias** | None (requires positional encoding) |
| **Key advantage** | Global context with parallel computation |

Self-attention enables direct modeling of relationships within a sequence by allowing each position to attend to all others. It is the mechanism that gives Transformers their power—forming the foundation for both understanding (BERT) and generation (GPT) architectures that have revolutionized NLP, vision, and beyond.

**Key insights:**

1. **Separation of concerns:** Q, K, V allow tokens to ask different questions than they answer, enabling asymmetric relationship modeling.

2. **Content-based addressing:** Unlike fixed connectivity in RNNs/CNNs, attention patterns adapt dynamically to input content.

3. **Position agnostic:** The architecture itself doesn't know about position—this must be injected explicitly, allowing flexibility in how position is encoded.

4. **Scalability trade-off:** Global context comes at quadratic cost, motivating the rich literature on efficient attention variants.

## References

- Vaswani et al., "Attention Is All You Need" (2017) — Original Transformer paper
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2020)
- Clark et al., "What Does BERT Look At?" (2019) — Attention analysis
