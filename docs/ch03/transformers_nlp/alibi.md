# ALiBi: Attention with Linear Biases

## Introduction

ALiBi (Attention with Linear Biases), introduced by Press et al. (2022), provides an elegant approach to positional encoding by adding linear biases directly to attention scores. Unlike learned or sinusoidal encodings, ALiBi requires no additional parameters and extrapolates remarkably well to sequences longer than seen during training.

## Key Insight

Instead of encoding positions in the input embeddings, ALiBi adds position-dependent penalties to attention scores:

$$
\text{softmax}\left(\mathbf{q}_i \mathbf{K}^T + m \cdot [-(i-j)]\right)
$$

Where $m$ is a head-specific slope that penalizes attending to distant positions.

## Mathematical Formulation

### Attention Score Modification

For query at position $i$ and key at position $j$:

$$
a_{ij} = \mathbf{q}_i^T \mathbf{k}_j - m \cdot |i - j|
$$

For causal (decoder) attention:

$$
a_{ij} = \begin{cases}
\mathbf{q}_i^T \mathbf{k}_j - m \cdot (i - j) & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

### Head-Specific Slopes

Different attention heads use different slopes, forming a geometric sequence:

$$
m_h = \frac{1}{2^{8h/H}}
$$

For $H$ heads: $m \in \{1/2^1, 1/2^2, 1/2^3, \ldots, 1/2^8\}$

Steeper slopes (smaller $m$) focus on recent context; gentler slopes capture longer dependencies.

### Bias Matrix

The bias matrix $B$ for sequence length $n$:

$$
B = \begin{bmatrix}
0 & -\infty & -\infty & \cdots \\
-1 & 0 & -\infty & \cdots \\
-2 & -1 & 0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix} \times m
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.
    
    Uses geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    
    Args:
        num_heads: Number of attention heads
        
    Returns:
        Tensor of slopes [num_heads]
    """
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(num_heads).is_integer():
        # Power of 2: use standard geometric sequence
        slopes = get_slopes_power_of_2(num_heads)
    else:
        # Non-power of 2: interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base_slopes = get_slopes_power_of_2(closest_power_of_2)
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
        slopes = base_slopes + extra_slopes[:num_heads - closest_power_of_2]
    
    return torch.tensor(slopes)


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    
    Adds linear position-dependent biases to attention scores.
    No learned parameters - only geometric slopes.
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 4096):
        """
        Args:
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length to precompute
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Get slopes for each head
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes.view(num_heads, 1, 1))
        
        # Precompute bias matrix
        self._build_alibi_bias(max_seq_len)
    
    def _build_alibi_bias(self, seq_len: int):
        """Build the ALiBi bias matrix."""
        # Position indices
        positions = torch.arange(seq_len)
        
        # Relative position matrix: [seq_len, seq_len]
        # relative_pos[i, j] = j - i
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # For causal attention: only consider j <= i
        # Bias is -(i - j) = j - i for allowed positions
        # This gives 0, -1, -2, -3, ... for each row
        relative_pos = relative_pos.float()
        
        # Store as [1, seq_len, seq_len] for broadcasting
        self.register_buffer('alibi_bias_base', relative_pos.unsqueeze(0))
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get ALiBi bias for given sequence length.
        
        Args:
            seq_len: Current sequence length
            
        Returns:
            Bias tensor [num_heads, seq_len, seq_len]
        """
        if seq_len > self.max_seq_len:
            self._build_alibi_bias(seq_len)
        
        # Get bias for current sequence length
        bias = self.alibi_bias_base[:, :seq_len, :seq_len]
        
        # Scale by head-specific slopes: [num_heads, seq_len, seq_len]
        return self.slopes * bias
    
    def get_bias_for_kv_cache(
        self,
        query_len: int,
        kv_len: int
    ) -> torch.Tensor:
        """
        Get ALiBi bias for incremental decoding with KV cache.
        
        Args:
            query_len: Number of query positions (usually 1)
            kv_len: Total key/value length including cache
            
        Returns:
            Bias tensor [num_heads, query_len, kv_len]
        """
        # For single query at position kv_len-1 attending to all kv_len keys
        # Relative positions: [-(kv_len-1), -(kv_len-2), ..., -1, 0]
        relative_pos = torch.arange(kv_len, device=self.slopes.device).float()
        relative_pos = relative_pos - (kv_len - 1)  # Shift so last position is 0
        
        # Expand for query dimension: [1, kv_len]
        relative_pos = relative_pos.unsqueeze(0)
        
        # Scale by slopes: [num_heads, 1, kv_len]
        return self.slopes * relative_pos


class ALiBiAttention(nn.Module):
    """
    Multi-Head Attention with ALiBi positional bias.
    
    Used in BLOOM, MPT, and other models.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        causal: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Attention dropout
            causal: Whether to use causal masking
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # ALiBi bias
        self.alibi = ALiBiPositionalBias(num_heads, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        if causal:
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
            self.register_buffer('causal_mask', mask.bool())
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with ALiBi.
        
        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            past_key_value: Cached KV for generation
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        kv_len = k.size(2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi bias
        if past_key_value is not None:
            # Incremental decoding: query is single position
            alibi_bias = self.alibi.get_bias_for_kv_cache(seq_len, kv_len)
        else:
            alibi_bias = self.alibi(kv_len)
        
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)
        
        # Apply causal mask
        if self.causal and seq_len > 1:
            causal_mask = self.causal_mask[kv_len - seq_len:kv_len, :kv_len]
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value


class ALiBiTransformerBlock(nn.Module):
    """Transformer block with ALiBi attention."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = ALiBiAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """Forward with pre-norm architecture."""
        # Attention
        residual = x
        x = self.norm1(x)
        attn_out, present = self.attention(x, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + self.dropout(attn_out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)
        
        return x, present


def visualize_alibi_bias(num_heads: int = 8, seq_len: int = 64):
    """Visualize ALiBi bias patterns for different heads."""
    import matplotlib.pyplot as plt
    
    alibi = ALiBiPositionalBias(num_heads, seq_len)
    bias = alibi(seq_len)  # [num_heads, seq_len, seq_len]
    
    # Apply causal mask for visualization
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    bias = bias.masked_fill(causal_mask, float('nan'))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    slopes = get_alibi_slopes(num_heads)
    
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(bias[h].numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h+1}, slope={slopes[h]:.4f}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('ALiBi Bias Patterns (Different Slopes per Head)')
    plt.tight_layout()
    plt.savefig('alibi_visualization.png', dpi=150)
    plt.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    d_model = 512
    num_heads = 8
    max_seq_len = 2048
    batch_size = 2
    seq_len = 128
    
    # Test ALiBi attention
    attention = ALiBiAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache = attention(x, use_cache=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache shapes: K={cache[0].shape}, V={cache[1].shape}")
    
    # Test length extrapolation
    print("\n--- Testing Length Extrapolation ---")
    long_seq = torch.randn(1, max_seq_len * 2, d_model)
    long_output, _ = attention(long_seq)
    print(f"Long sequence input: {long_seq.shape}")
    print(f"Long sequence output: {long_output.shape}")
    
    # Test incremental generation
    print("\n--- Testing Incremental Generation ---")
    prompt = torch.randn(1, 10, d_model)
    _, kv_cache = attention(prompt, use_cache=True)
    
    for step in range(5):
        new_token = torch.randn(1, 1, d_model)
        output, kv_cache = attention(
            new_token,
            past_key_value=kv_cache,
            use_cache=True
        )
        print(f"Step {step}: cache_len={kv_cache[0].size(2)}")
    
    # Parameters
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("(Note: ALiBi adds 0 extra parameters!)")
    
    # Visualize
    visualize_alibi_bias(num_heads=8, seq_len=64)
    print("\nVisualization saved to 'alibi_visualization.png'")
```

## Advantages of ALiBi

### 1. Zero Extra Parameters

ALiBi only requires precomputed slopes—no learned embeddings:

| Method | Extra Parameters |
|--------|-----------------|
| Learned Positional | $L \times d$ |
| Sinusoidal | 0 (but fixed in embeddings) |
| Relative Bias | $O(L)$ to $O(L^2)$ |
| **ALiBi** | **0** |

### 2. Excellent Length Extrapolation

Models trained with ALiBi on short sequences generalize to much longer ones:

| Training Length | Evaluation Length | Perplexity Increase |
|-----------------|-------------------|---------------------|
| 1024 | 2048 | ~2% |
| 1024 | 4096 | ~5% |
| 1024 | 8192 | ~10% |

### 3. Simple Integration

Just add bias to attention scores—no architecture changes:

```python
attn_scores = q @ k.T / sqrt(d_k)
attn_scores = attn_scores + alibi_bias  # Only change!
attn_weights = softmax(attn_scores)
```

## Comparison with Other Methods

| Method | Parameters | Extrapolation | Relative Position | Complexity |
|--------|------------|---------------|-------------------|------------|
| Sinusoidal | 0 | Moderate | Implicit | O(1) |
| Learned | L×d | Poor | Implicit | O(1) |
| T5 Relative | Buckets | Good | Explicit | O(n²) |
| RoPE | 0 | Good | Explicit | O(n) |
| **ALiBi** | **0** | **Excellent** | **Explicit** | **O(n²) precompute** |

## Summary

ALiBi provides a simple yet effective positional encoding:

1. **No parameters**: Just geometric slopes
2. **Length extrapolation**: Excellent generalization to longer sequences
3. **Easy implementation**: Add bias matrix to attention scores
4. **Proven at scale**: Used in BLOOM (176B), MPT, and others

## References

1. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR.
2. Scao, T., et al. (2022). "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model."
3. MosaicML (2023). "MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs."
