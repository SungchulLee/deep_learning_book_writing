# Transformer Variants

## Overview

| Architecture | Parallelization | Long-range | Complexity | Inductive Bias |
|--------------|-----------------|------------|------------|----------------|
| RNN/LSTM | Sequential | Difficult | O(n) | Temporal |
| CNN | Parallel | Limited | O(n) | Local patterns |
| Transformer | Parallel | Easy | O(n²) | None (learned) |

## RNN/LSTM

### Architecture

```
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ...
        ↓         ↓         ↓
       y₁        y₂        y₃
```

### Characteristics

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(out)
```

**Pros:**
- O(n) complexity
- Good inductive bias for sequences
- Memory efficient

**Cons:**
- Sequential processing (slow training)
- Vanishing gradients for long sequences
- Difficult to parallelize

## CNN for Sequences

### Architecture

```
[x₁ x₂ x₃ x₄ x₅]
   \_____/
    Conv1
      \_____/
       Conv2
         ...
```

### Characteristics

```python
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), vocab_size)
    
    def forward(self, x):
        x = self.embed(x).transpose(1, 2)  # [B, C, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        return self.fc(torch.cat(pooled, dim=1))
```

**Pros:**
- Fully parallelizable
- Good for local patterns
- Efficient for fixed-size contexts

**Cons:**
- Limited receptive field (need many layers)
- Not ideal for very long dependencies

## Transformer

### Architecture

```
[x₁ x₂ x₃ x₄ x₅]
    ↓↓↓↓↓
Self-Attention (all-to-all)
    ↓↓↓↓↓
[y₁ y₂ y₃ y₄ y₅]
```

### Characteristics

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        x = self.embed(x) + self.pos(pos)
        return self.fc(self.transformer(x))
```

**Pros:**
- Fully parallelizable
- Direct long-range connections
- Highly expressive

**Cons:**
- O(n²) complexity
- No inherent sequential bias
- Memory intensive for long sequences

## Detailed Comparison

### Complexity Analysis

| Operation | RNN | CNN | Transformer |
|-----------|-----|-----|-------------|
| Sequential ops | O(n) | O(1) | O(1) |
| Per-layer complexity | O(n·d²) | O(k·n·d²) | O(n²·d) |
| Max path length | O(n) | O(log_k(n)) | O(1) |

### Memory Usage

For sequence length n, dimension d:

| Model | Training Memory | Inference Memory |
|-------|-----------------|------------------|
| RNN | O(n·d) | O(d) |
| CNN | O(n·d) | O(n·d) |
| Transformer | O(n²·h + n·d) | O(n²·h) with KV-cache |

### Performance by Sequence Length

```
Short (< 512):    Transformer ≈ CNN > RNN
Medium (512-2K):  Transformer > CNN > RNN
Long (2K-8K):     Efficient Transformers > CNN > RNN
Very Long (>8K):  State Space Models / Linear Attention
```

## Benchmarks (Approximate)

### Language Modeling (Perplexity, lower is better)

| Model | WikiText-103 | Parameters |
|-------|--------------|------------|
| LSTM | ~35 | 150M |
| Transformer | ~18 | 150M |
| GPT-2 | ~15 | 1.5B |

### Text Classification (Accuracy)

| Model | IMDB | SST-2 |
|-------|------|-------|
| LSTM | 89% | 87% |
| CNN | 90% | 88% |
| BERT | 95% | 94% |

### Machine Translation (BLEU)

| Model | WMT En-De |
|-------|-----------|
| LSTM Seq2Seq | 28.4 |
| Transformer | 34.4 |

## Hybrid Approaches

### Transformer + CNN

```python
class ConvTransformer(nn.Module):
    """Local convolution + global attention."""
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    def forward(self, x):
        # Local features
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        # Global attention
        global_out, _ = self.attention(x, x, x)
        return local + global_out
```

### Transformer + RNN

```python
class TransformerWithRecurrence(nn.Module):
    """Transformer with recurrent memory."""
    def __init__(self, d_model, num_heads, memory_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.memory_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.memory = None
    
    def forward(self, x):
        # Update memory
        if self.memory is not None:
            x = torch.cat([self.memory, x], dim=1)
        out, _ = self.attention(x, x, x)
        # Compress to memory
        self.memory, _ = self.memory_rnn(out)
        return out[:, -x.size(1):, :]
```

## When to Use Each

### Use RNN When:
- Streaming data (online processing)
- Very memory constrained
- Sequence order is crucial
- Short sequences (<100)

### Use CNN When:
- Local patterns matter most
- Fixed-size classification
- Fast inference needed
- Moderate sequence length

### Use Transformer When:
- Long-range dependencies important
- Parallel training available
- State-of-the-art needed
- Pre-trained models available

## Modern Alternatives

For very long sequences, several architectures offer linear complexity while retaining Transformer-like performance:

| Model | Complexity | Approach | Key Innovation |
|-------|------------|----------|----------------|
| Linear Attention | $O(n)$ | Kernel approximation of softmax | Replaces $\text{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^TV)$ |
| Mamba/S4 | $O(n)$ | Selective state space models | Data-dependent state transitions with hardware-aware scanning |
| RWKV | $O(n)$ | RNN with Transformer-like training | Linear attention formulation that can be computed recurrently |
| Longformer | $O(n \cdot w)$ | Sparse attention with local + global | Sliding window attention with task-specific global tokens |
| Hyena | $O(n \log n)$ | Long convolutions | Replaces attention with implicitly parameterized convolutions |

These approaches are particularly important as context windows extend to 100K+ tokens, where standard Transformer attention becomes impractical.

## Summary

The three architectures represent different design philosophies for sequence modeling:

- **RNNs**: Built on the inductive bias that sequences are inherently temporal. Their sequential nature preserves ordering naturally but creates computation and gradient flow bottlenecks. Still relevant for streaming applications.
- **CNNs**: Apply local receptive field patterns efficiently through parallel convolution. Excel at pattern detection but require many stacked layers for long-range dependencies.
- **Transformers**: Make no assumptions about sequence structure, learning all relationships through attention. Most capable and scalable, at the cost of quadratic complexity.

The field has largely converged on Transformers, but understanding all three architectures helps choose the right tool for specific constraints (latency, memory, sequence length).

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need."
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
3. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification."
4. Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces."

---

## Sparse Attention Patterns

The key insight is that full attention matrices are empirically sparse—most attention weight concentrates on a small subset of positions. Sparse attention formalizes this by restricting the attention pattern a priori, avoiding the computation of low-weight entries entirely.

##### The Complexity Problem

For sequence length $N$ and model dimension $d$:

| Component | Time Complexity | Memory |
|-----------|----------------|--------|
| Full Attention | O(N²d) | O(N²) |
| **Goal** | O(Nd) or O(N log N · d) | O(N) |

For a sequence of $N = 16{,}384$ tokens (a modest document), the full attention matrix has $2.7 \times 10^8$ entries. At 32-bit precision, just storing this matrix for a single head requires ~1 GB.

##### Common Sparse Patterns

##### 1. Local (Sliding Window) Attention

Each position attends only to nearby positions within a window:

$$
\text{Attend}(i) = \{j : |i - j| \leq w\}
$$

**Complexity**: O(Nw) where $w$ is window size

```
Window size = 3:
Position 5 attends to: [2, 3, 4, 5, 6, 7, 8]
```

**Rationale**: Most natural language dependencies are local. Sliding window captures these while ignoring distant positions that rarely receive significant attention weight.

##### 2. Dilated (Strided) Attention

Attend to positions at fixed intervals, enabling long-range coverage without full attention:

$$
\text{Attend}(i) = \{j : (i - j) \mod d = 0, |i-j| \leq w \cdot d\}
$$

**Complexity**: O(Nw)

```
Dilation = 2, Window = 3:
Position 6 attends to: [0, 2, 4, 6, 8, 10, 12]
```

**Rationale**: Analogous to dilated convolutions. By attending to every $d$-th position, the effective receptive field grows by a factor of $d$ without increasing computation.

##### 3. Global Attention

Designate certain positions as "global" that attend to/from all positions:

$$
\text{Attend}(i) = \begin{cases}
\{1, ..., N\} & \text{if } i \in \mathcal{G} \\
\mathcal{G} \cup \text{Local}(i) & \text{otherwise}
\end{cases}
$$

Used in Longformer, BigBird. Typical global tokens include `[CLS]`, task-specific tokens, or the first/last tokens of each segment.

##### 4. Block Sparse Attention

Divide sequence into blocks and attend within/between specific blocks:

```
Block pattern:
[■ ■ □ □ ■]
[■ ■ ■ □ □]
[□ ■ ■ ■ □]
[□ □ ■ ■ ■]
[■ □ □ ■ ■]
```

##### 5. Random Attention

Each position randomly attends to a fixed number of other positions:

$$
\text{Attend}(i) = \text{Random}(k) \cup \text{Local}(i)
$$

**Theoretical significance**: Random attention ensures that the expected path length between any two tokens is $O(\log N)$, providing theoretical guarantees on information propagation (graph expansion properties).

##### 6. Hierarchical (Multi-Scale) Attention

Organize attention across multiple levels of granularity:

**Level 1**: Token-level local attention within segments
**Level 2**: Segment-level attention between segment summaries
**Level 3**: Document-level attention between document summaries

This creates a hierarchical information flow:

$$\text{Local tokens} \to \text{Segment summaries} \to \text{Global representation}$$

Hierarchical attention naturally handles documents with structure (sections, paragraphs) and reduces complexity to $O(N \sqrt{N})$ or better by processing fine-grained information locally and coarse-grained information globally.

**Sparse Transformer** (Child et al., 2019) pioneered this approach by factorizing the attention pattern into two sparse components: one attending to local context and one attending to strided positions, achieving $O(N\sqrt{N})$ complexity.

##### Combining Patterns

The most effective sparse attention methods combine multiple patterns. The BigBird theoretical result (Zaheer et al., 2020) proves that combining random, local, and global attention is sufficient for universal approximation—the sparse model can simulate any full-attention Transformer.

| Component | Purpose | Alone Sufficient? |
|-----------|---------|-------------------|
| Local | Capture adjacent dependencies | No (misses long-range) |
| Global | Aggregate full-sequence information | No (too few positions) |
| Random | Ensure short expected path length | No (misses structure) |
| **Combined** | **All of the above** | **Yes (Turing-complete)** |

##### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


def create_local_attention_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create sliding window attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Size of attention window (one side)
        device: Device for tensor
        
    Returns:
        Mask [seq_len, seq_len] where True = masked (don't attend)
    """
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Compute distances
    distance = torch.abs(rows - cols)
    
    # Mask positions outside window
    mask = distance > window_size
    
    return mask


def create_dilated_attention_mask(
    seq_len: int,
    window_size: int,
    dilation: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create dilated (strided) attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Number of positions to attend to
        dilation: Stride between attended positions
        device: Device for tensor
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Distance in dilated space
    distance = torch.abs(rows - cols)
    
    # Valid if: within dilated window AND aligned with dilation
    within_window = distance <= window_size * dilation
    aligned = (rows - cols) % dilation == 0
    
    mask = ~(within_window & aligned)
    return mask


def create_block_sparse_mask(
    seq_len: int,
    block_size: int,
    num_random_blocks: int = 1,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create block sparse attention mask.
    
    Each block attends to itself, adjacent blocks, and random blocks.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Initialize mask (all masked)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    
    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min((i + 1) * block_size, seq_len)
        
        for j in range(num_blocks):
            j_start = j * block_size
            j_end = min((j + 1) * block_size, seq_len)
            
            # Self block and adjacent blocks
            if abs(i - j) <= 1:
                mask[i_start:i_end, j_start:j_end] = False
            
            # Random blocks
            elif torch.rand(1).item() < num_random_blocks / num_blocks:
                mask[i_start:i_end, j_start:j_end] = False
    
    return mask


class LocalAttention(nn.Module):
    """
    Local (Sliding Window) Attention.
    
    Each position attends only to positions within a fixed window.
    Complexity: O(N * window_size * d)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Forward pass with local attention.
        
        Args:
            x: Input [batch, seq_len, d_model]
            causal: Apply causal masking within window
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create local attention mask
        local_mask = create_local_attention_mask(seq_len, self.window_size, x.device)
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            local_mask = local_mask | causal_mask
        
        # Apply mask
        scores = scores.masked_fill(local_mask, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)


class LongformerAttention(nn.Module):
    """
    Longformer-style attention combining:
    1. Local sliding window attention (for all tokens)
    2. Global attention (for designated tokens like [CLS])
    
    Complexity: O(N * (w + g)) where w = window, g = global tokens
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # Local attention projections
        self.q_local = nn.Linear(d_model, d_model)
        self.k_local = nn.Linear(d_model, d_model)
        self.v_local = nn.Linear(d_model, d_model)
        
        # Global attention projections (separate parameters)
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)
        
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        global_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            global_attention_mask: Boolean [batch, seq_len], True for global tokens
        """
        batch_size, seq_len, _ = x.shape
        
        # Local projections
        q_local = self.q_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_local = self.k_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_local = self.v_local(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Global projections
        q_global = self.q_global(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_global = self.k_global(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute local attention scores
        local_scores = torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale
        
        # Apply local mask
        local_mask = create_local_attention_mask(seq_len, self.window_size, x.device)
        local_scores = local_scores.masked_fill(local_mask, float('-inf'))
        
        # For global positions: compute attention to all positions
        # For other positions: add attention to global positions
        global_indices = global_attention_mask.nonzero(as_tuple=True)
        
        if len(global_indices[0]) > 0:
            # Global positions can attend to everything
            for b, idx in zip(global_indices[0], global_indices[1]):
                local_scores[b, :, idx, :] = torch.matmul(
                    q_global[b, :, idx:idx+1, :],
                    k_global[b].transpose(-2, -1)
                ) * self.scale
                
                # All positions can attend to global positions
                local_scores[b, :, :, idx] = torch.matmul(
                    q_local[b],
                    k_global[b, :, idx:idx+1, :].transpose(-2, -1)
                ).squeeze(-1) * self.scale
        
        # Softmax
        attn_weights = F.softmax(local_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v_local)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)


class BigBirdAttention(nn.Module):
    """
    BigBird-style attention combining:
    1. Random attention
    2. Window (local) attention
    3. Global attention
    
    Achieves O(N) complexity.
    Theoretically proven to be a universal approximator of sequence functions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 64,
        num_global_tokens: int = 2,
        num_random_blocks: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_blocks = num_random_blocks
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_bigbird_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create BigBird sparse attention mask."""
        
        # Start with all masked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # 1. Global tokens (first num_global_tokens positions)
        mask[:self.num_global_tokens, :] = False
        mask[:, :self.num_global_tokens] = False
        
        # 2. Local/sliding window (within block and adjacent blocks)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for i in range(seq_len):
            block_i = i // self.block_size
            
            # Attend to same block
            start = block_i * self.block_size
            end = min((block_i + 1) * self.block_size, seq_len)
            mask[i, start:end] = False
            
            # Attend to adjacent blocks
            if block_i > 0:
                prev_start = (block_i - 1) * self.block_size
                mask[i, prev_start:start] = False
            if block_i < num_blocks - 1:
                next_end = min((block_i + 2) * self.block_size, seq_len)
                mask[i, end:next_end] = False
        
        # 3. Random attention
        for i in range(0, seq_len, self.block_size):
            block_end = min(i + self.block_size, seq_len)
            
            # Select random blocks
            valid_blocks = [b for b in range(num_blocks) 
                          if abs(b - i // self.block_size) > 1]
            
            if valid_blocks:
                random_blocks = torch.tensor(valid_blocks)[
                    torch.randperm(len(valid_blocks))[:self.num_random_blocks]
                ]
                
                for rb in random_blocks:
                    rb_start = rb * self.block_size
                    rb_end = min((rb + 1) * self.block_size, seq_len)
                    mask[i:block_end, rb_start:rb_end] = False
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with BigBird attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply BigBird mask
        mask = self._create_bigbird_mask(seq_len, x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and apply
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output)


def visualize_sparse_patterns(seq_len: int = 64):
    """Visualize different sparse attention patterns."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    patterns = [
        ("Full Attention", torch.zeros(seq_len, seq_len).bool()),
        ("Local (w=8)", create_local_attention_mask(seq_len, window_size=8)),
        ("Dilated (w=4, d=4)", create_dilated_attention_mask(seq_len, 4, 4)),
        ("Block Sparse", create_block_sparse_mask(seq_len, 16, 1)),
        ("Causal Local", create_local_attention_mask(seq_len, 8) | 
         torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()),
    ]
    
    # BigBird pattern
    bigbird = BigBirdAttention(64, 1, block_size=16, num_global_tokens=2)
    bigbird_mask = bigbird._create_bigbird_mask(seq_len, torch.device('cpu'))
    patterns.append(("BigBird", bigbird_mask))
    
    for idx, (name, mask) in enumerate(patterns):
        ax = axes[idx // 3, idx % 3]
        # Invert mask for visualization (white = attend, black = masked)
        ax.imshow(~mask.float().numpy(), cmap='gray', aspect='auto')
        ax.set_title(name)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Count attending pairs
        attending = (~mask).sum().item()
        density = attending / (seq_len * seq_len) * 100
        ax.text(0.02, 0.98, f'Density: {density:.1f}%', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='red')
    
    plt.suptitle('Sparse Attention Patterns (White = Attend, Black = Masked)')
    plt.tight_layout()
    plt.savefig('sparse_attention_patterns.png', dpi=150)
    plt.close()


# Example usage
if __name__ == "__main__":
    d_model = 256
    num_heads = 4
    seq_len = 256
    batch_size = 2
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test Local Attention
    print("--- Local Attention ---")
    local_attn = LocalAttention(d_model, num_heads, window_size=32)
    out_local = local_attn(x)
    print(f"Input: {x.shape}, Output: {out_local.shape}")
    
    # Test Longformer Attention
    print("\n--- Longformer Attention ---")
    longformer_attn = LongformerAttention(d_model, num_heads, window_size=32)
    out_longformer = longformer_attn(x)
    print(f"Input: {x.shape}, Output: {out_longformer.shape}")
    
    # Test BigBird Attention
    print("\n--- BigBird Attention ---")
    bigbird_attn = BigBirdAttention(d_model, num_heads, block_size=32)
    out_bigbird = bigbird_attn(x)
    print(f"Input: {x.shape}, Output: {out_bigbird.shape}")
    
    # Visualize patterns
    visualize_sparse_patterns(64)
    print("\nVisualization saved to 'sparse_attention_patterns.png'")
```

##### Complexity Comparison

| Method | Time | Memory | Global Context |
|--------|------|--------|----------------|
| Full Attention | O(N²) | O(N²) | ✓ Complete |
| Local Window | O(Nw) | O(Nw) | ✗ Limited |
| Dilated | O(Nw) | O(Nw) | ✗ Limited (wider) |
| Sparse Transformer | O(N√N) | O(N√N) | ✓ Via strided pattern |
| Longformer | O(Nw + Ng) | O(N) | ✓ Via global |
| BigBird | O(N) | O(N) | ✓ Random + global |

##### Relationship to Efficient Attention

Sparse attention (structured sparsity in the attention matrix) is one of several approaches to efficient attention. The landscape includes:

| Approach | Strategy | Examples |
|----------|----------|----------|
| **Sparse patterns** | Restrict which pairs compute attention | Longformer, BigBird, Sparse Transformer |
| **Low-rank approximation** | Approximate attention matrix with low-rank factorization | Linformer, Nyström |
| **Kernel methods** | Approximate softmax with linearizable kernels | Performer, Random Feature Attention |
| **IO-aware computation** | Optimize memory access patterns | FlashAttention |

Sparse attention and these other approaches are complementary—for example, FlashAttention can accelerate sparse patterns, and low-rank methods can be combined with local windows.

##### Summary

Sparse attention patterns enable efficient processing of long sequences:

1. **Local attention**: Fast but limited context
2. **Global tokens**: Maintain full-sequence information
3. **Random attention**: Theoretical expressiveness guarantees
4. **Hierarchical attention**: Multi-scale information flow
5. **Combined patterns**: Best of all approaches, provably universal

##### References

1. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
2. Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS.
3. Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers."
4. Kitaev, N., et al. (2020). "Reformer: The Efficient Transformer." ICLR.
5. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS.

---

## Scaling Transformers

##### Training Large-Scale Models

##### GPT-3

GPT-3, developed by OpenAI, demonstrated that scaling decoder-only Transformers yields strong few-shot learning capabilities.

**Architecture**: Decoder-only Transformer with 175 billion parameters, using 96 layers, 12,288 hidden dimensions, and 96 attention heads.

**Training data**: A filtered subset of Common Crawl combined with curated datasets (WebText, Books, Wikipedia), totaling approximately 300 billion tokens. Diversity of training data is critical for generalization across tasks.

**Training compute**: GPT-3 required approximately 3.14 × 10²³ FLOPs, trained across thousands of GPUs using model and data parallelism.

**Key insight**: GPT-3's in-context learning ability—performing tasks from a few examples in the prompt without gradient updates—emerged at scale and was not observed in smaller models.

##### PaLM

PaLM (Pathways Language Model), developed by Google, scaled to 540 billion parameters with several architectural refinements.

**Architecture**: Decoder-only Transformer with SwiGLU activation, parallel attention and FFN computation, multi-query attention, and RoPE positional encodings.

**Training data**: A multilingual corpus of 780 billion tokens spanning web documents, books, code, and conversational data.

**Training infrastructure**: PaLM was trained across 6,144 TPU v4 chips using the Pathways system, which efficiently orchestrates computation across multiple TPU pods.

**Key insight**: PaLM exhibited "breakthrough" capabilities on reasoning tasks (e.g., chain-of-thought prompting) that appeared discontinuously as model scale increased.

##### LLaMA

LLaMA (Large Language Model Meta AI) demonstrated that smaller, well-trained models can match or exceed the performance of much larger models when given sufficient data.

**Architecture**: Decoder-only Transformer using pre-normalization (RMSNorm), SwiGLU activations, and RoPE, with model sizes from 7B to 65B parameters.

**Training data**: 1.4 trillion tokens from publicly available data only, trained for more tokens than typical for the model size.

**Key insight**: The optimal compute-efficient model is trained on significantly more tokens than its parameter count, challenging earlier scaling law assumptions that favored larger models trained on fewer tokens (Chinchilla scaling).

##### Scaling Laws

Kaplan et al. (2020) and Hoffmann et al. (2022) established empirical relationships between model performance and compute budget.

##### Kaplan Scaling Laws

Performance (measured by cross-entropy loss $L$) follows power laws with model size $N$, dataset size $D$, and compute $C$:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

where $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, and $\alpha_C \approx 0.050$.

##### Chinchilla Scaling (Hoffmann et al.)

The compute-optimal approach allocates a fixed compute budget $C$ by scaling both model size $N$ and data size $D$ equally:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

Practically, the optimal token count is approximately 20× the parameter count. A 10B parameter model should be trained on ~200B tokens.

##### Challenges of Training at Scale

##### Data Challenges

**Volume and quality**: Large models require trillions of high-quality tokens. Web-crawled data must be carefully filtered for quality, deduplicated, and balanced across domains. Data contamination (overlap with evaluation benchmarks) must be detected and removed.

**Preprocessing at scale**: Tokenization, filtering, and shuffling of terabyte-scale datasets requires distributed data pipelines. Training data must be served to thousands of accelerators without becoming a bottleneck.

##### Compute Challenges

**Cost**: Training GPT-3 is estimated to have cost several million dollars in compute. The cost scales roughly linearly with total FLOPs, which grows as $O(N \cdot D)$ where $N$ is parameters and $D$ is tokens.

**Training stability**: Large models are prone to loss spikes—sudden increases in training loss that can derail training. Mitigation strategies include lower learning rates, gradient clipping, and careful initialization.

**Reproducibility**: Training runs on thousands of accelerators introduce non-determinism from floating-point order of operations, making exact reproducibility difficult.

##### Memory Bottlenecks

For a model with $N$ parameters trained with Adam in float32:

| Component | Memory per Parameter | Total (175B params) |
|-----------|---------------------|---------------------|
| Parameters | 4 bytes | 700 GB |
| Gradients | 4 bytes | 700 GB |
| Adam optimizer states ($m$, $v$) | 8 bytes | 1,400 GB |
| **Total** | **16 bytes** | **~2.8 TB** |

This exceeds the memory of any single accelerator (typically 40–80 GB), necessitating parallelism strategies.

##### Parallelism Strategies

##### Data Parallelism

Each worker holds a complete copy of the model and processes a different data shard. Gradients are averaged across workers via all-reduce:

$$
g_{\text{avg}} = \frac{1}{K} \sum_{k=1}^{K} g_k
$$

where $K$ is the number of workers and $g_k$ is the gradient from worker $k$.

**Limitation**: Each worker must store the full model, so data parallelism alone cannot handle models that exceed single-device memory.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_data_parallel(rank: int, world_size: int):
    """Initialize distributed process group for data parallelism."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_with_ddp(rank: int, world_size: int, model: nn.Module):
    """
    Wrap model in DistributedDataParallel for multi-GPU training.
    
    Each GPU processes a different mini-batch; gradients are
    automatically averaged via all-reduce before optimizer.step().
    """
    setup_data_parallel(rank, world_size)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])           # Wraps model for gradient sync
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Each rank gets a different data shard via DistributedSampler
    # Gradients are averaged automatically across ranks
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()                              # Gradient all-reduce happens here
        optimizer.step()
```

##### Gradient Accumulation

When the desired effective batch size exceeds GPU memory, gradient accumulation simulates larger batches by accumulating gradients across multiple forward-backward passes before updating parameters:

$$
g_{\text{accumulated}} = \frac{1}{A} \sum_{a=1}^{A} g_a
$$

where $A$ is the number of accumulation steps.

```python
def train_with_gradient_accumulation(
    model: nn.Module,
    dataloader,
    optimizer,
    accumulation_steps: int = 8,
    max_grad_norm: float = 1.0
):
    """
    Training loop with gradient accumulation.
    
    Effective batch size = micro_batch_size × accumulation_steps × num_gpus
    For example: 4 × 8 × 4 GPUs = 128 effective batch size.
    """
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Forward + backward (gradients accumulate in .grad)
        loss = model(**batch).loss
        loss = loss / accumulation_steps              # Normalize by accumulation steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
```

##### Model Parallelism (Tensor Parallelism)

Individual layers are split across devices. For a linear layer $Y = XW$, the weight matrix $W$ is partitioned column-wise across $K$ devices:

$$
W = [W_1 \mid W_2 \mid \cdots \mid W_K]
$$

Each device computes $Y_k = X W_k$ and the results are concatenated (or reduced, depending on the partition strategy). This is the approach used by Megatron-LM.

##### Pipeline Parallelism

Different layers are assigned to different devices. Input micro-batches flow through the pipeline:

$$
\text{Device 1: Layers 1–24} \rightarrow \text{Device 2: Layers 25–48} \rightarrow \text{Device 3: Layers 49–72} \rightarrow \text{Device 4: Layers 73–96}
$$

**Challenge**: Naive pipeline parallelism creates "pipeline bubbles" where devices idle while waiting for forward or backward passes. Micro-batching (GPipe) and interleaved scheduling (PipeDream) reduce idle time.

##### 3D Parallelism

Large-scale training combines all three strategies:

- **Data parallelism** across groups of devices
- **Tensor parallelism** within each group (typically within a single node)
- **Pipeline parallelism** across groups

For example, Megatron-Turing NLG (530B) uses 8-way tensor parallelism within each node, 35-way pipeline parallelism across nodes, and data parallelism across pipeline replicas.

##### Emerging Architectures for Efficient Scaling

##### Megatron-LM

Megatron-LM (NVIDIA) provides efficient tensor parallelism for Transformer layers by partitioning the attention and FFN computations:

- **Attention**: $Q$, $K$, $V$ projections are split column-wise across devices. Each device computes attention independently. The output projection is split row-wise, and results are summed via all-reduce.
- **FFN**: The first linear layer is split column-wise; the second is split row-wise. A single all-reduce synchronizes outputs.

This approach requires only two all-reduce operations per Transformer layer, keeping communication overhead low.

##### Mixture of Experts (MoE)

Mixture of Experts models, such as Switch Transformers, activate only a subset of parameters for each input token, enabling much larger total model capacity with sublinear compute cost.

The gating mechanism selects the top-$k$ experts for each token:

$$
G(x) = \text{TopK}\left(\text{softmax}(x \cdot W_g)\right)
$$

$$
\text{MoE}(x) = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)
$$

where $E_i$ is the $i$-th expert (typically an FFN) and $G(x)_i$ is the gating weight.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: a standard position-wise FFN."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TopKGatingMoE(nn.Module):
    """
    Mixture of Experts layer with Top-K gating.
    
    Replaces the standard FFN in a Transformer block.
    Each token is routed to the top-k experts based on
    a learned gating function.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: projects input to expert scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss: Load balancing loss (scalar)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)                  # [B*S, d_model]
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)                # [B*S, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)    # [B*S, num_experts]
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(       # [B*S, top_k] each
            gate_probs, self.top_k, dim=-1
        )
        
        # Normalize selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simplified; production uses scatter/gather)
        output = torch.zeros_like(x_flat)              # [B*S, d_model]
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]           # [B*S]
            weight = top_k_probs[:, k].unsqueeze(-1)   # [B*S, 1]
            
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weight[mask] * expert_output
        
        # Load balancing auxiliary loss
        # f_i: fraction of tokens routed to expert i
        # p_i: average gating probability for expert i
        f = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                f[i] += (top_k_indices[:, k] == i).float().mean()
        f = f / self.top_k
        
        p = gate_probs.mean(dim=0)                     # [num_experts]
        aux_loss = self.num_experts * (f * p).sum()
        
        return output.view(batch_size, seq_len, d_model), aux_loss
```

**Switch Transformer**: Uses $k=1$ (route to a single expert), achieving up to 7× speedup over dense models of equivalent quality. A Switch Transformer with 1.6 trillion total parameters uses only ~100B parameters per forward pass.

**Load balancing**: Expert routing can lead to uneven load distribution. The auxiliary loss above encourages balanced expert utilization:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N_E \sum_{i=1}^{N_E} f_i \cdot p_i
$$

where $f_i$ is the fraction of tokens routed to expert $i$, $p_i$ is the average gating probability for expert $i$, and $N_E$ is the number of experts.

##### Efficient Scaling Techniques

**Quantization**: Reducing parameter precision from float32 to int8 or int4 cuts memory by 4–8× with minimal quality loss. Post-training quantization (GPTQ, AWQ) and quantization-aware training (QAT) are both used.

**Pruning**: Removing redundant weights reduces model size. Structured pruning (removing entire attention heads or FFN dimensions) is more hardware-friendly than unstructured (individual weight) pruning.

**Distillation**: Training a smaller "student" model to mimic a larger "teacher" model. DistilBERT achieves 97% of BERT's performance with 40% fewer parameters and 60% faster inference.

##### Training Infrastructure Comparison

| System | Model Size | Hardware | Parallelism | Training Time |
|--------|-----------|----------|-------------|---------------|
| GPT-3 | 175B | V100 GPUs | Data + Model | Months |
| PaLM | 540B | TPU v4 (6144 chips) | Data + Model + Pipeline | Weeks |
| LLaMA-65B | 65B | A100 GPUs (2048) | Data + Tensor + Pipeline | ~21 days |
| Chinchilla | 70B | TPU v3/v4 | Data + Model | Weeks |

##### Summary

Scaling Transformers involves navigating a complex trade-off space across data, compute, and memory:

1. **Scaling laws** provide guidance on optimal allocation of compute between model size and training data.
2. **Parallelism strategies** (data, tensor, pipeline) enable training models that exceed single-device memory.
3. **Mixture of Experts** achieves parameter efficiency by activating only a subset of the model per token.
4. **Post-training efficiency** techniques (quantization, pruning, distillation) make large models practical for deployment.
5. **Training stability** requires careful learning rate scheduling, gradient clipping, and initialization.

The field continues to evolve rapidly, with new architectures and training methodologies emerging regularly.

##### References

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3)
2. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv.
3. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv.
4. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv.
5. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS. (Chinchilla)
6. Shoeybi, M., et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv.
7. Fedus, W., et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models." JMLR.
