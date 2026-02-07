# Attention Patterns

## Introduction

Attention patterns describe the characteristic ways in which attention weights distribute across positions. Understanding these patterns is essential for interpreting model behaviour, debugging architectures, and designing efficient attention variants. This section covers the major pattern types across self-attention, cross-attention, and causal attention, along with the masking strategies that shape them.

## Cross-Attention

Cross-attention (encoder-decoder attention) enables the decoder to reference the encoder's output. Queries come from the decoder while keys and values come from the encoder:

$$\mathbf{Q} = \mathbf{Y}'\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{M}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{M}\mathbf{W}_V$$

$$\text{CrossAttention}(\mathbf{Y}', \mathbf{M}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The attention matrix is **rectangular**: $(n_t \times n_s)$ where $n_t$ is the target length and $n_s$ is the source length. Unlike decoder self-attention, cross-attention has **no causal mask**—the source is fully available.

### Cross-Attention vs Self-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|----------------|-----------------|
| Q source | Same sequence | Decoder |
| K, V source | Same sequence | Encoder memory |
| Attention matrix | Square $(n \times n)$ | Rectangular $(n_t \times n_s)$ |
| Mask | Optional (causal in decoder) | None (source fully available) |
| Purpose | Internal context | External reference |

### Alignment Patterns in Translation

Different language pairs exhibit characteristic alignment patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Monotonic** | Sequential alignment | English ↔ Spanish (similar word order) |
| **Diagonal** | One-to-one with shifts | Closely related languages |
| **Scattered** | Long-range reordering | English ↔ Japanese (SOV vs SVO) |
| **Many-to-one** | Multiple source → one target | Compound words, idioms |
| **One-to-many** | One source → multiple targets | Morphological expansion |

### Cross-Attention in Different Architectures

| Architecture | Cross-Attention Usage |
|--------------|----------------------|
| **Original Transformer** | Decoder cross-attends to encoder at every layer |
| **Decoder-Only (GPT)** | No encoder, no cross-attention |
| **BART, T5** | Encoder-decoder with cross-attention for seq2seq |
| **Multimodal (Flamingo, LLaVA)** | Text decoder cross-attends to image/audio encoder |
| **Whisper** | Audio encoder + text decoder with cross-attention |
| **Stable Diffusion** | Text encoder output injected into U-Net via cross-attention |

The common pattern: the module being conditioned provides **queries**, and the conditioning signal provides **keys and values**.

### K/V Caching

Since encoder memory $\mathbf{M}$ does not change during generation, cross-attention keys and values are computed once and cached. Only the query projection is recomputed at each step—a significant speedup for autoregressive decoding.

## Self-Attention Pattern Emergence

### Layer-wise Specialisation

Research using probing classifiers has revealed that different layers learn systematically different patterns:

**Early layers:** Local patterns (adjacent tokens), positional patterns (fixed offsets), syntactic basics (punctuation, function words).

**Middle layers:** Syntactic relationships (subject-verb agreement), coreference resolution (pronoun-antecedent), named entity patterns, phrase structure.

**Later layers:** Task-specific patterns, semantic relationships, long-range dependencies, abstract feature combinations.

### Head Specialisation

In multi-head attention, different heads develop distinct functional roles:

| Head Type | Pattern | Example |
|-----------|---------|---------|
| **Positional heads** | Attend to previous/next token | "The [cat]" → attend to "The" |
| **Syntactic heads** | Capture subject-verb dependencies | "cats [run]" → attend to "cats" |
| **Coreference heads** | Track pronoun-antecedent | "[it] was tired" → attend to "animal" |
| **Copy heads** | Focus on rare words, proper nouns | Names, technical terms |
| **Separator heads** | Attend to punctuation, boundaries | Periods, [SEP] tokens |

## Masking Patterns

Masking enforces structural constraints by setting scores to $-\infty$ before softmax, guaranteeing zero weight for masked positions.

### Encoder-Decoder Masking Summary

| Component | Masking | Reason |
|-----------|---------|--------|
| Encoder self-attention | None (bidirectional) | Source is fully available |
| Decoder self-attention | Causal mask | Preserve autoregressive property |
| Decoder cross-attention | None (over encoder) | All source positions accessible |

Both decoder sublayers additionally use **padding masks** to ignore pad tokens.

### Causal Mask

Lower-triangular mask ensuring position $i$ only attends to $j \leq i$:

```python
def causal_mask(size):
    """Returns (1, 1, size, size) lower-triangular mask."""
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
```

Applied additively for clean gradient flow:

$$\text{scores}_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + M_{ij}$$

where $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -\infty$ otherwise.

### Padding Mask

Prevents attention to padding tokens in variable-length batches:

```python
def padding_mask(seq_lengths, max_len):
    """(batch,) -> (batch, 1, 1, max_len) broadcastable mask."""
    arange = torch.arange(max_len, device=seq_lengths.device)
    return (arange.unsqueeze(0) < seq_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
```

### Combined Masks

Causal and padding masks are combined multiplicatively:

```python
combined = causal_mask(seq_len) * padding_mask(lengths, seq_len)
```

## Common Pattern Archetypes

Through analysis of trained Transformers, several recurring archetypes have been identified:

### Diagonal Pattern

Positions attend primarily to the same or neighbouring positions. Common in early layers and monotonically aligned cross-attention.

```
┌─────────────┐
│ █ · · · · │
│ · █ · · · │
│ · · █ · · │
│ · · · █ · │
│ · · · · █ │
└─────────────┘
```

### Vertical Stripe Pattern

Many positions attend to a specific token (often [CLS], [SEP], or the first token). These positions act as "information hubs."

```
┌─────────────┐
│ █ · · · █ │
│ █ · · · █ │
│ █ · · · █ │
│ █ · · · █ │
│ █ · · · █ │
└─────────────┘
```

### Block Diagonal Pattern

Positions attend within local blocks, capturing phrase-level structure.

```
┌─────────────┐
│ █ █ · · · │
│ █ █ · · · │
│ · · █ █ █ │
│ · · █ █ █ │
│ · · █ █ █ │
└─────────────┘
```

### Heterogeneous Pattern

Irregular, content-dependent patterns that capture task-specific relationships.

```
┌─────────────┐
│ · █ · · █ │
│ █ · · · · │
│ · · · █ · │
│ · · █ · · │
│ █ · · · · │
└─────────────┘
```

### Uniform Pattern

Near-uniform attention across all positions, often seen in "no-op" heads that contribute little to the output.

```
┌─────────────┐
│ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ │
│ ░ ░ ░ ░ ░ │
└─────────────┘
```

## Efficient Attention Patterns

The $O(n^2)$ complexity of full attention motivates structured sparsity patterns:

| Method | Pattern | Complexity | Key Idea |
|--------|---------|------------|----------|
| **Full attention** | Dense | $O(n^2)$ | All-to-all |
| **Sparse (Longformer)** | Local + global | $O(n)$ | Window + global tokens |
| **Linformer** | Low-rank | $O(n)$ | Project K, V to lower dimension |
| **Performer** | Random features | $O(n)$ | Kernel approximation of softmax |
| **FlashAttention** | Dense (IO-aware) | $O(n^2)$ time, $O(n)$ memory | Tiled computation |
| **Sliding window** | Local band | $O(nw)$ | Fixed window size $w$ |
| **Block-sparse** | Block diagonal | $O(n \cdot b)$ | Fixed block size $b$ |

### Sliding Window Attention

```python
def sliding_window_mask(seq_len, window_size):
    """Each position attends to window_size positions around it."""
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    return mask
```

### Longformer-style: Local + Global

```python
def longformer_mask(seq_len, window_size, global_positions):
    """Local windowed attention with global tokens."""
    mask = torch.zeros(seq_len, seq_len)
    
    # Local window for all positions
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    
    # Global tokens attend to/from everywhere
    for g in global_positions:
        mask[g, :] = 1  # Global attends to all
        mask[:, g] = 1  # All attend to global
    
    return mask
```

## PyTorch Implementation

### Cross-Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention for encoder-decoder models."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        decoder_state: torch.Tensor, 
        encoder_memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: (batch, n_t, d_model)
            encoder_memory: (batch, n_s, d_model)
            mask: (batch, n_t, n_s) — optional padding mask
        """
        batch_size, n_t, _ = decoder_state.shape
        n_s = encoder_memory.shape[1]
        
        Q = self.W_q(decoder_state).view(batch_size, n_t, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(encoder_memory).view(batch_size, n_s, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(encoder_memory).view(batch_size, n_s, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_t, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights
```

### Attention Pattern Analysis

```python
def analyze_attention_patterns(weights: torch.Tensor) -> dict:
    """
    Analyze attention weight patterns.
    
    Args:
        weights: (batch, heads, seq_q, seq_k)
        
    Returns:
        Dictionary of pattern metrics per head.
    """
    batch, n_heads, seq_q, seq_k = weights.shape
    metrics = {}
    
    for h in range(n_heads):
        w = weights[:, h]  # (batch, seq_q, seq_k)
        
        # Entropy: higher = more distributed
        entropy = -(w * (w + 1e-10).log()).sum(-1).mean().item()
        
        # Diagonality: how much weight is on/near diagonal
        if seq_q == seq_k:
            diag_weight = torch.diagonal(w, dim1=-2, dim2=-1).mean().item()
        else:
            diag_weight = 0.0
        
        # Sparsity: max attention weight (higher = more concentrated)
        max_weight = w.max(dim=-1).values.mean().item()
        
        # Positional bias: average attended position relative to query
        positions = torch.arange(seq_k, device=w.device).float()
        attended_pos = (w * positions.unsqueeze(0).unsqueeze(0)).sum(-1)
        query_pos = torch.arange(seq_q, device=w.device).float()
        relative_pos = (attended_pos - query_pos.unsqueeze(0)).mean().item()
        
        metrics[f'head_{h}'] = {
            'entropy': entropy,
            'diag_weight': diag_weight,
            'max_weight': max_weight,
            'mean_relative_position': relative_pos,
        }
    
    return metrics
```

## Gradient Flow Through Cross-Attention

Cross-attention creates two gradient highways:

**To decoder:** $\mathcal{L} \to \text{output} \to \text{FFN} \to \text{cross-attn} \to \mathbf{W}_Q^{\text{(cross)}}$

**To encoder:** $\mathcal{L} \to \text{output} \to \text{FFN} \to \text{cross-attn} \to \mathbf{W}_K, \mathbf{W}_V \to \mathbf{M} \to \text{all encoder layers}$

The encoder-to-decoder gradient path ensures the encoder learns representations that the decoder finds useful—making joint training fundamentally different from training independent models.

## Summary

| Pattern Type | Structure | Where Observed |
|--------------|-----------|----------------|
| **Diagonal** | Identity-like | Early layers, monotonic alignment |
| **Vertical stripe** | Column attention | [CLS]/[SEP] tokens |
| **Block diagonal** | Phrase-level | Middle layers |
| **Causal** | Lower triangular | Decoder self-attention |
| **Cross-attention** | Rectangular, no mask | Decoder-encoder bridge |
| **Sparse** | Structured sparsity | Efficient variants |

Understanding attention patterns provides interpretability, guides architecture design, and motivates efficient attention methods that exploit the observation that most trained attention matrices are naturally sparse.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP*.
3. Voita, E., et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting." *ACL*.
4. Beltagy, I., Peters, M., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." *arXiv:2004.05150*.
5. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*.
6. Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training." *ACL*.
