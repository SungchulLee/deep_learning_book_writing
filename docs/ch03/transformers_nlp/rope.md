# Rotary Position Embedding (RoPE)

## Introduction

Rotary Position Embedding (RoPE), introduced in RoFormer and adopted by LLaMA, Mistral, and other modern LLMs, encodes positional information through rotation matrices in complex space. Unlike additive positional encodings, RoPE naturally captures relative positions within the attention computation itself.

## Motivation

Traditional positional encodings have limitations:

| Method | Length Extrapolation | Relative Position | Efficiency |
|--------|---------------------|-------------------|------------|
| Sinusoidal (additive) | Moderate | Implicit | Good |
| Learned (additive) | Poor | Implicit | Good |
| Relative Position Bias | Good | Explicit | Moderate |
| **RoPE** | **Excellent** | **Explicit** | **Good** |

RoPE's key insight: encode positions by rotating query and key vectors, so that the dot product between $q_m$ and $k_n$ depends only on $(m - n)$.

## Mathematical Formulation

### Core Idea

For a 2D vector $\mathbf{x} = [x_0, x_1]^T$, rotation by angle $\theta$ is:

$$
R_\theta \mathbf{x} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}
$$

RoPE applies position-dependent rotations:

$$
R_{\theta,m} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

### Extension to High Dimensions

For $d$-dimensional vectors, we pair dimensions and apply different frequencies:

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1
$$

The full rotation matrix for position $m$:

$$
R_m = \begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

### Relative Position Property

The key property: when computing attention between positions $m$ and $n$:

$$
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

The attention score depends only on the relative position $(n - m)$!

## Efficient Implementation

Instead of constructing full rotation matrices, we use element-wise operations:

$$
\text{RoPE}(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$

$$
\text{RoPE}(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes absolute position through rotation, resulting in
    relative position dependence in attention scores.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000
    ):
        """
        Args:
            dim: Dimension of the embedding (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for frequency computation
        """
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands
        # theta_i = 10000^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute sin and cos values."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate for full dimension: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims.
        
        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]
            
        Returns:
            Rotated (q, k) tuple
        """
        seq_len = q.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # Get cached values
        if position_ids is not None:
            # Custom positions (for KV-cache during generation)
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            # Standard sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated


class RoPEAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding.
    
    Used in LLaMA, Mistral, and other modern LLMs.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        rope_base: int = 10000
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional mask
            position_ids: Position indices for RoPE
            past_key_value: Cached KV for generation
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Attention scores
        kv_seq_len = k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if seq_len > 1:  # Skip for single token generation
            causal_mask = self.causal_mask[
                kv_seq_len - seq_len:kv_seq_len,
                :kv_seq_len
            ]
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and apply to values
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value


class RoPEWithNTKScaling(RotaryPositionEmbedding):
    """
    RoPE with NTK-aware scaling for better length extrapolation.
    
    Scales the base frequency to handle longer sequences than
    seen during training.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        # Adjust base for NTK scaling
        adjusted_base = base * (scaling_factor ** (dim / (dim - 2)))
        super().__init__(dim, max_seq_len, int(adjusted_base))
        self.scaling_factor = scaling_factor


class RoPEWithLinearScaling(RotaryPositionEmbedding):
    """
    RoPE with linear interpolation for length extrapolation.
    
    Simply scales position indices to fit within training range.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        scaling_factor: float = 1.0
    ):
        super().__init__(dim, max_seq_len, base)
        self.scaling_factor = scaling_factor
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply linear scaling to positions."""
        seq_len = q.shape[2]
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device)
        
        # Scale positions
        scaled_positions = position_ids.float() / self.scaling_factor
        
        # Compute cos/sin for scaled positions
        freqs = torch.outer(scaled_positions.flatten(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().view(1, 1, seq_len, -1)
        sin = emb.sin().view(1, 1, seq_len, -1)
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated


# Demonstration and visualization
def visualize_rope_properties():
    """Visualize RoPE's relative position property."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    dim = 64
    seq_len = 100
    rope = RotaryPositionEmbedding(dim, seq_len)
    
    # Create random query and key
    q = torch.randn(1, 1, seq_len, dim)
    k = torch.randn(1, 1, seq_len, dim)
    
    # Apply RoPE
    q_rot, k_rot = rope(q, k)
    
    # Compute attention pattern (shows relative position dependency)
    attn = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (dim ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Attention pattern
    ax1 = axes[0]
    im1 = ax1.imshow(attn[0, 0].detach().numpy(), cmap='viridis')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    ax1.set_title('Attention Pattern with RoPE')
    plt.colorbar(im1, ax=ax1)
    
    # Relative position decay
    ax2 = axes[1]
    # Average attention weight as function of relative position
    relative_weights = []
    for offset in range(-seq_len+1, seq_len):
        weights = []
        for i in range(seq_len):
            j = i + offset
            if 0 <= j < seq_len:
                weights.append(attn[0, 0, i, j].item())
        if weights:
            relative_weights.append((offset, np.mean(weights)))
    
    offsets, weights = zip(*relative_weights)
    ax2.plot(offsets, weights)
    ax2.set_xlabel('Relative Position (k - q)')
    ax2.set_ylabel('Average Attention Weight')
    ax2.set_title('Attention Decay with Relative Distance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rope_visualization.png', dpi=150)
    plt.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    d_model = 512
    num_heads = 8
    head_dim = d_model // num_heads
    batch_size = 2
    seq_len = 128
    
    # Test RoPE module directly
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    q_rot, k_rot = rope(q, k)
    
    print(f"Query shape: {q.shape} -> {q_rot.shape}")
    print(f"Key shape: {k.shape} -> {k_rot.shape}")
    
    # Test full attention layer
    attention = RoPEAttention(d_model, num_heads, max_seq_len=2048)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, cache = attention(x, use_cache=True)
    
    print(f"\nAttention input: {x.shape}")
    print(f"Attention output: {output.shape}")
    print(f"KV cache shapes: K={cache[0].shape}, V={cache[1].shape}")
    
    # Test incremental generation
    print("\n--- Testing Incremental Generation ---")
    
    # First pass: process prompt
    prompt_len = 10
    prompt = torch.randn(1, prompt_len, d_model)
    _, kv_cache = attention(prompt, use_cache=True)
    
    # Generate tokens one by one
    for step in range(5):
        new_token = torch.randn(1, 1, d_model)
        position_ids = torch.tensor([[prompt_len + step]])
        
        output, kv_cache = attention(
            new_token,
            position_ids=position_ids,
            past_key_value=kv_cache,
            use_cache=True
        )
        
        print(f"Step {step}: output={output.shape}, cache_len={kv_cache[0].size(2)}")
    
    # Parameters
    total_params = sum(p.numel() for p in attention.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Visualize
    visualize_rope_properties()
    print("\nVisualization saved to 'rope_visualization.png'")
```

## Comparison with Other Methods

### Additive vs Multiplicative

| Aspect | Additive (Sinusoidal) | Multiplicative (RoPE) |
|--------|----------------------|----------------------|
| Operation | $x + PE$ | $R \cdot x$ |
| Relative position | Implicit | Explicit in dot product |
| Long-range decay | No | Natural decay |

### RoPE Variants for Length Extension

| Method | Approach | Max Extension |
|--------|----------|---------------|
| Linear Scaling | Scale positions by factor | 2-4x |
| NTK-Aware | Adjust frequency base | 4-8x |
| YaRN | Combined approach | 16-32x |

## Summary

RoPE provides elegant position encoding that:

1. **Encodes absolute positions**: Each position gets unique rotation
2. **Captures relative positions**: Dot product depends only on distance
3. **Extends well**: Various scaling methods for longer contexts
4. **Efficient**: Element-wise operations, no extra parameters

## References

1. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
2. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases."
3. Chen, S., et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation."
