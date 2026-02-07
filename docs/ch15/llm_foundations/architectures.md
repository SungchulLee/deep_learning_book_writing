# Architectural Innovations in Large Language Models

## Learning Objectives

- Understand key architectural changes from vanilla Transformer to modern LLMs
- Implement RMSNorm, SwiGLU, Rotary Embeddings, and Grouped-Query Attention
- Analyze the trade-offs of different architectural choices
- Compare architectures across GPT, LLaMA, and Mistral families

## Introduction

Modern LLMs incorporate numerous architectural innovations beyond the original Transformer. These modifications improve training stability, computational efficiency, and model capabilities while maintaining the core attention mechanism.

## Normalization Variants

### Pre-Norm vs Post-Norm

**Original Transformer (Post-Norm)**:
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Modern LLMs (Pre-Norm)**:
$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Pre-norm provides better gradient flow for deep models.

### RMSNorm

Root Mean Square Normalization removes the mean-centering:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

Where:
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS computation
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight
```

**Advantages**: 15% faster than LayerNorm, similar performance.

## Activation Functions

### SwiGLU

Gated Linear Unit with Swish activation:

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2)$$

Where $\text{Swish}(x) = x \cdot \sigma(x)$ and $\sigma$ is sigmoid.

```python
class SwiGLU(nn.Module):
    """SwiGLU activation for FFN."""
    
    def __init__(self, dim: int, hidden_dim: int = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # LLaMA uses 8/3 multiplier
        # Round to multiple of 256 for efficiency
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Up projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W1) * (x @ W3)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

### FFN Comparison

| Activation | Parameters | Performance | Used In |
|------------|------------|-------------|---------|
| ReLU | $2 \cdot d \cdot d_{ff}$ | Baseline | Original Transformer |
| GELU | $2 \cdot d \cdot d_{ff}$ | +1% | GPT-2, BERT |
| SwiGLU | $3 \cdot d \cdot d_{ff}$ | +2% | LLaMA, Mistral |

## Positional Encodings

### Rotary Position Embeddings (RoPE)

RoPE encodes position through rotation in complex space:

$$\text{RoPE}(x_m, m) = x_m e^{im\theta}$$

For real-valued vectors, apply rotation to pairs of dimensions:

$$R_\theta^m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        
        # Compute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q, k: (batch, heads, seq_len, head_dim)
            positions: (seq_len,) position indices
        """
        cos = self.cos_cached[positions]  # (seq_len, head_dim/2)
        sin = self.sin_cached[positions]
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotation to tensor."""
        # Split into pairs
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Rotate
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated.flatten(-2)
```

### RoPE Advantages

1. **Relative position**: Attention depends on $(m - n)$, not absolute positions
2. **Extrapolation**: Better length generalization than absolute embeddings
3. **No additional parameters**: Position encoded through rotation

### Extended Context: NTK-Aware Scaling

For longer sequences than training, scale the base frequency:

$$\theta'_i = \theta_i \cdot \alpha^{-2i/d}$$

```python
def ntk_scaled_rope(dim: int, max_seq_len: int, base: int = 10000, scale: float = 2.0):
    """NTK-aware RoPE scaling for extended context."""
    # Scale base for longer sequences
    scaled_base = base * (scale ** (dim / (dim - 2)))
    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq
```

## Attention Variants

### Multi-Query Attention (MQA)

Single key-value head shared across all query heads:

```python
class MultiQueryAttention(nn.Module):
    """Multi-Query Attention: single KV head, multiple Q heads."""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)  # Multiple query heads
        self.k_proj = nn.Linear(dim, self.head_dim)  # Single KV head
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, 1, self.head_dim)  # Single head
        v = self.v_proj(x).view(B, L, 1, self.head_dim)
        
        # Broadcast k, v across query heads
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)
        
        # Standard attention
        scores = torch.einsum('blhd,bmhd->bhlm', q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhlm,bmhd->blhd', attn, v)
        
        return self.o_proj(out.reshape(B, L, -1))
```

### Grouped-Query Attention (GQA)

Intermediate between MHA and MQA: groups of query heads share KV heads:

```python
class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention: KV heads < Query heads."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_kv_heads: int  # LLaMA-2 70B uses 8 KV heads, 64 Q heads
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        
        # Repeat KV heads to match query heads
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # Standard attention computation
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)
```

### Attention Comparison

| Type | KV Heads | KV Cache Size | Quality | Used In |
|------|----------|---------------|---------|---------|
| MHA | H | $2 \cdot H \cdot d_h$ | Best | GPT-3 |
| GQA | H/G | $2 \cdot H/G \cdot d_h$ | Near-MHA | LLaMA-2 70B |
| MQA | 1 | $2 \cdot d_h$ | Good | PaLM |

## Sliding Window Attention

### Mistral's Approach

Local attention with window size $W$:

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create sliding window attention mask."""
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        start = max(0, i - window_size)
        mask[i, start:i+1] = 1
    
    return mask

class SlidingWindowAttention(nn.Module):
    """Attention with sliding window for efficiency."""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Create sliding window mask
        mask = sliding_window_mask(L, self.window_size).to(x.device)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        
        # Attention with mask
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (self.head_dim ** 0.5)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        return self.out(out.reshape(B, L, -1))
```

### Benefits

- **Memory**: O(n·W) instead of O(n²)
- **Effective context**: Still attends to full sequence via stacked layers

## Complete LLaMA-Style Block

```python
class LLaMABlock(nn.Module):
    """LLaMA-style transformer block with modern innovations."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim_multiplier: float = 8/3,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Pre-norm
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        
        # GQA
        self.attention = GroupedQueryAttention(dim, num_heads, num_kv_heads)
        
        # SwiGLU FFN
        self.ffn = SwiGLU(dim, int(dim * ffn_dim_multiplier))
        
        # RoPE
        self.rope = RotaryEmbedding(dim // num_heads)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Attention with residual
        h = self.attention_norm(x)
        h = self.attention(h, positions, mask)
        x = x + h
        
        # FFN with residual
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        
        return x
```

## Architecture Comparison

| Component | GPT-3 | LLaMA-2 | Mistral |
|-----------|-------|---------|---------|
| Normalization | LayerNorm | RMSNorm | RMSNorm |
| Norm Position | Post | Pre | Pre |
| Activation | GELU | SwiGLU | SwiGLU |
| Position | Learned | RoPE | RoPE |
| Attention | MHA | GQA | SWA + GQA |
| Context | 2K/4K | 4K | 8K (SWA) |

## Summary

Modern LLM architectures incorporate:

1. **RMSNorm**: Faster normalization without mean-centering
2. **SwiGLU**: Gated activation for better expressivity
3. **RoPE**: Rotation-based relative positional encoding
4. **GQA/MQA**: Reduced KV cache for efficient inference
5. **Sliding Window**: Linear complexity attention

## Key Insight

$$\boxed{\text{Modern LLMs} = \text{Transformer} + \text{Pre-Norm} + \text{RoPE} + \text{SwiGLU} + \text{GQA}}$$

## References

1. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
2. Shazeer, N. (2020). GLU Variants Improve Transformer.
3. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
4. Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models.
