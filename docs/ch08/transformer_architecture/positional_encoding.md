# Positional Encoding

## The Position Problem

Self-attention is inherently permutation-invariant—it processes tokens as a set rather than a sequence. Given input tokens $\{x_1, x_2, \ldots, x_n\}$, the attention output is identical regardless of input order:

$$
\text{Attention}(\{x_1, x_2, x_3\}) = \text{Attention}(\{x_3, x_1, x_2\})
$$

This property, while enabling parallelization, loses critical sequential information. The sentence "The cat sat on the mat" and "The mat sat on the cat" would produce identical representations without positional information.

## Sinusoidal Positional Encoding

The original Transformer paper introduces sinusoidal positional encodings using sine and cosine functions of different frequencies:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Where:
- $pos$ is the position in the sequence (0-indexed)
- $i$ is the dimension index (0 to $d_{\text{model}}/2 - 1$)
- $d_{\text{model}}$ is the model dimension

### Frequency Analysis

Each dimension pair $(2i, 2i+1)$ corresponds to a sinusoid with wavelength:

$$
\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}
$$

The wavelengths form a geometric progression from $2\pi$ (for $i=0$) to $2\pi \cdot 10000$ (for $i = d_{\text{model}}/2 - 1$).

### Relative Position Property

A key property is that relative positions can be expressed as linear transformations. For any fixed offset $k$:

$$
\text{PE}_{pos+k} = f(\text{PE}_{pos})
$$

Specifically:

$$
\begin{bmatrix} \sin((pos+k)\omega_i) \\ \cos((pos+k)\omega_i) \end{bmatrix} = 
\begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}
\begin{bmatrix} \sin(pos \cdot \omega_i) \\ \cos(pos \cdot \omega_i) \end{bmatrix}
$$

Where $\omega_i = 1/10000^{2i/d_{\text{model}}}$.

This allows the model to learn to attend to relative positions through linear projections.

## PyTorch Implementation: Sinusoidal Encoding

```python
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Position indices [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Division term for frequencies [d_model/2]
        # Using exp and log for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for visualization."""
        return self.pe[:, :seq_len, :].squeeze(0)


def visualize_positional_encoding(d_model: int = 128, max_len: int = 100):
    """Visualize the sinusoidal positional encoding matrix."""
    
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    encoding = pe.get_encoding(max_len).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full encoding matrix
    ax1 = axes[0, 0]
    im1 = ax1.imshow(encoding, aspect='auto', cmap='RdBu')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Positional Encoding Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # Encoding for specific positions
    ax2 = axes[0, 1]
    positions_to_plot = [0, 10, 20, 50, 99]
    for pos in positions_to_plot:
        ax2.plot(encoding[pos, :50], label=f'pos={pos}', alpha=0.7)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Encoding at Different Positions (first 50 dims)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Encoding for specific dimensions
    ax3 = axes[1, 0]
    dims_to_plot = [0, 1, 10, 11, 50, 51]
    for dim in dims_to_plot:
        ax3.plot(encoding[:, dim], label=f'dim={dim}', alpha=0.7)
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Encoding Value')
    ax3.set_title('Encoding at Different Dimensions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Similarity between positions (dot product)
    ax4 = axes[1, 1]
    similarity = encoding @ encoding.T
    im4 = ax4.imshow(similarity, cmap='viridis')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Position')
    ax4.set_title('Position Similarity (Dot Product)')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    plt.close()
    
    return encoding


# Example usage
if __name__ == "__main__":
    # Create encoding
    d_model = 512
    max_len = 100
    
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    
    # Test with batch of embeddings
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify relative position property
    encoding = pe.get_encoding(100)
    
    # Check that PE[pos+k] can be linearly transformed from PE[pos]
    pos, k = 10, 5
    pe_pos = encoding[pos]
    pe_pos_k = encoding[pos + k]
    
    print(f"\nRelative position test:")
    print(f"PE[{pos}] shape: {pe_pos.shape}")
    print(f"PE[{pos + k}] shape: {pe_pos_k.shape}")
    
    # Visualize
    visualize_positional_encoding(d_model=128, max_len=100)
    print("\nVisualization saved to 'positional_encoding_visualization.png'")
```

## Learned Positional Encoding

An alternative approach learns position embeddings as parameters:

$$
\mathbf{P} = \text{Embedding}(\text{positions}) \in \mathbb{R}^{L \times d_{\text{model}}}
$$

### Advantages and Disadvantages

| Aspect | Sinusoidal | Learned |
|--------|------------|---------|
| Generalization to longer sequences | ✓ Extrapolates | ✗ Fixed length |
| Task-specific optimization | ✗ Fixed | ✓ Adapts |
| Parameter count | 0 | $L \times d_{\text{model}}$ |
| Relative position bias | Implicit | Must be learned |

### PyTorch Implementation: Learned Encoding

```python
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding as used in BERT and GPT.
    
    Each position has a learnable embedding vector.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings with small values."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # Create position indices [seq_len]
        positions = torch.arange(seq_len, device=x.device)
        
        # Get position embeddings [seq_len, d_model]
        pos_embeddings = self.position_embeddings(positions)
        
        # Add to input (broadcasting over batch dimension)
        x = x + pos_embeddings
        
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Get learned encoding for visualization."""
        positions = torch.arange(seq_len)
        return self.position_embeddings(positions)


class LearnedPositionalEncodingWithInterpolation(nn.Module):
    """
    Learned positional encoding with interpolation for variable lengths.
    
    Can handle sequences longer than max_len through interpolation.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding with interpolation support.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len <= self.max_len:
            # Use embeddings directly
            pos_embeddings = self.position_embeddings[:, :seq_len, :]
        else:
            # Interpolate for longer sequences
            pos_embeddings = torch.nn.functional.interpolate(
                self.position_embeddings.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_embeddings
        
        return self.dropout(x)


# Example usage
if __name__ == "__main__":
    d_model = 256
    max_len = 512
    
    # Standard learned encoding
    learned_pe = LearnedPositionalEncoding(d_model, max_len)
    
    # Test
    x = torch.randn(32, 100, d_model)
    output = learned_pe(x)
    print(f"Learned PE output shape: {output.shape}")
    
    # With interpolation
    learned_pe_interp = LearnedPositionalEncodingWithInterpolation(d_model, max_len)
    
    # Test with longer sequence
    x_long = torch.randn(32, 1000, d_model)
    output_long = learned_pe_interp(x_long)
    print(f"Interpolated PE output shape: {output_long.shape}")
```

## Rotary Position Embedding (RoPE)

RoPE, introduced in RoFormer and used in LLaMA, encodes positions through rotation in complex space:

$$
f_q(\mathbf{x}_m, m) = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m
$$

Where $\mathbf{R}_m$ is a rotation matrix encoding position $m$.

### Key Properties

1. **Relative position in attention**: $q_m^T k_n$ depends only on $(m - n)$
2. **Decaying with distance**: Natural decay for distant positions
3. **Flexible sequence length**: Works with arbitrary lengths

### Implementation

```python
import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in LLaMA.
    
    Encodes position through rotation in complex space.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        base: int = 10000
    ):
        """
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute rotation matrices
        self._precompute_freqs(max_len)
    
    def _precompute_freqs(self, seq_len: int):
        """Pre-compute sin and cos for rotation."""
        # Position indices [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device)
        
        # Frequencies [seq_len, d_model/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for sin and cos [seq_len, d_model]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Store cos and sin
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None
    ) -> tuple:
        """
        Apply rotary position embedding to queries and keys.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length (optional, inferred from q if not provided)
            
        Returns:
            Rotated (q, k) tuple
        """
        if seq_len is None:
            seq_len = q.shape[2]
        
        # Get cached values
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Reshape for broadcasting [1, 1, seq_len, d_model]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated


# Example usage
if __name__ == "__main__":
    d_model = 64
    num_heads = 8
    head_dim = d_model // num_heads
    batch_size = 32
    seq_len = 100
    
    rope = RotaryPositionalEncoding(head_dim, max_len=512)
    
    # Create queries and keys
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Apply RoPE
    q_rotated, k_rotated = rope(q, k)
    
    print(f"Query shape: {q_rotated.shape}")
    print(f"Key shape: {k_rotated.shape}")
```

## Alibi (Attention with Linear Biases)

ALiBi adds position-dependent biases directly to attention scores:

$$
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{m} \cdot [-(i-j)]\right)
$$

Where $\mathbf{m}$ is a head-specific slope.

### Implementation

```python
import torch
import torch.nn as nn
import math


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    
    Adds linear position biases to attention scores.
    """
    
    def __init__(self, num_heads: int, max_len: int = 2048):
        """
        Args:
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.num_heads = num_heads
        
        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Pre-compute position difference matrix
        self._precompute_bias(max_len)
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """
        Get ALiBi slopes for each head.
        
        Uses geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            slopes = slopes + extra_slopes[:num_heads - closest_power_of_2]
        
        return torch.tensor(slopes).view(num_heads, 1, 1)
    
    def _precompute_bias(self, max_len: int):
        """Pre-compute the position bias matrix."""
        # Position indices
        positions = torch.arange(max_len)
        
        # Relative positions [max_len, max_len]
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # ALiBi uses negative relative positions for causal attention
        # Bias is -|i - j| for non-causal, -(i - j) for causal (upper triangle masked)
        bias = -torch.abs(relative_positions)
        
        self.register_buffer('alibi_bias', bias)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get ALiBi bias for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Bias tensor [num_heads, seq_len, seq_len]
        """
        # Get bias for current sequence length
        bias = self.alibi_bias[:seq_len, :seq_len]
        
        # Scale by slopes [num_heads, seq_len, seq_len]
        return self.slopes * bias


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi positional bias."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ALiBi bias
        self.alibi = ALiBiPositionalBias(num_heads)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with ALiBi position bias.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi bias [1, num_heads, seq_len, seq_len]
        alibi_bias = self.alibi(seq_len).unsqueeze(0)
        scores = scores + alibi_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output


# Example usage
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    batch_size = 32
    seq_len = 100
    
    alibi_attn = ALiBiAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = alibi_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

## Comparison of Position Encoding Methods

| Method | Length Extrapolation | Relative Position | Parameters | Used In |
|--------|---------------------|-------------------|------------|---------|
| Sinusoidal | Good | Implicit | 0 | Original Transformer |
| Learned | Poor | Implicit | $L \times d$ | BERT, GPT-2 |
| RoPE | Good | Explicit | 0 | LLaMA, GPT-Neo |
| ALiBi | Excellent | Explicit | 0 | BLOOM, MPT |

## Summary

Positional encoding is essential for Transformers to process sequential data. The choice of encoding method affects:

1. **Generalization**: How well the model handles sequences of different lengths
2. **Efficiency**: Computational and memory requirements  
3. **Relative vs. Absolute**: Whether positions are encoded absolutely or relatively
4. **Length extrapolation**: Ability to handle longer sequences than seen during training

Modern architectures increasingly favor relative position methods (RoPE, ALiBi) for their superior length extrapolation capabilities.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Shaw, P., et al. (2018). "Self-Attention with Relative Position Representations." NAACL.
3. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
4. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases." ICLR.

---

## RoPE: Deep Dive

#### Introduction

Rotary Position Embedding (RoPE), introduced in RoFormer and adopted by LLaMA, Mistral, and other modern LLMs, encodes positional information through rotation matrices in complex space. Unlike additive positional encodings, RoPE naturally captures relative positions within the attention computation itself.

#### Motivation

Traditional positional encodings have limitations:

| Method | Length Extrapolation | Relative Position | Efficiency |
|--------|---------------------|-------------------|------------|
| Sinusoidal (additive) | Moderate | Implicit | Good |
| Learned (additive) | Poor | Implicit | Good |
| Relative Position Bias | Good | Explicit | Moderate |
| **RoPE** | **Excellent** | **Explicit** | **Good** |

RoPE's key insight: encode positions by rotating query and key vectors, so that the dot product between $q_m$ and $k_n$ depends only on $(m - n)$.

#### Mathematical Formulation

#### Core Idea

For a 2D vector $\mathbf{x} = [x_0, x_1]^T$, rotation by angle $\theta$ is:

$$
R_\theta \mathbf{x} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}
$$

RoPE applies position-dependent rotations:

$$
R_{\theta,m} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

#### Extension to High Dimensions

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

#### Relative Position Property

The key property: when computing attention between positions $m$ and $n$:

$$
(R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
$$

The attention score depends only on the relative position $(n - m)$!

#### Efficient Implementation

Instead of constructing full rotation matrices, we use element-wise operations:

$$
\text{RoPE}(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i)
$$

$$
\text{RoPE}(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

#### PyTorch Implementation

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

#### Comparison with Other Methods

#### Additive vs Multiplicative

| Aspect | Additive (Sinusoidal) | Multiplicative (RoPE) |
|--------|----------------------|----------------------|
| Operation | $x + PE$ | $R \cdot x$ |
| Relative position | Implicit | Explicit in dot product |
| Long-range decay | No | Natural decay |

#### RoPE Variants for Length Extension

| Method | Approach | Max Extension |
|--------|----------|---------------|
| Linear Scaling | Scale positions by factor | 2-4x |
| NTK-Aware | Adjust frequency base | 4-8x |
| YaRN | Combined approach | 16-32x |

#### Summary

RoPE provides elegant position encoding that:

1. **Encodes absolute positions**: Each position gets unique rotation
2. **Captures relative positions**: Dot product depends only on distance
3. **Extends well**: Various scaling methods for longer contexts
4. **Efficient**: Element-wise operations, no extra parameters

#### References

1. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
2. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases."
3. Chen, S., et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation."

---

## ALiBi: Deep Dive

#### Introduction

ALiBi (Attention with Linear Biases), introduced by Press et al. (2022), provides an elegant approach to positional encoding by adding linear biases directly to attention scores. Unlike learned or sinusoidal encodings, ALiBi requires no additional parameters and extrapolates remarkably well to sequences longer than seen during training.

#### Key Insight

Instead of encoding positions in the input embeddings, ALiBi adds position-dependent penalties to attention scores:

$$
\text{softmax}\left(\mathbf{q}_i \mathbf{K}^T + m \cdot [-(i-j)]\right)
$$

Where $m$ is a head-specific slope that penalizes attending to distant positions.

#### Mathematical Formulation

#### Attention Score Modification

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

#### Head-Specific Slopes

Different attention heads use different slopes, forming a geometric sequence:

$$
m_h = \frac{1}{2^{8h/H}}
$$

For $H$ heads: $m \in \{1/2^1, 1/2^2, 1/2^3, \ldots, 1/2^8\}$

Steeper slopes (smaller $m$) focus on recent context; gentler slopes capture longer dependencies.

#### Bias Matrix

The bias matrix $B$ for sequence length $n$:

$$
B = \begin{bmatrix}
0 & -\infty & -\infty & \cdots \\
-1 & 0 & -\infty & \cdots \\
-2 & -1 & 0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix} \times m
$$

#### PyTorch Implementation

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

#### Advantages of ALiBi

#### 1. Zero Extra Parameters

ALiBi only requires precomputed slopes—no learned embeddings:

| Method | Extra Parameters |
|--------|-----------------|
| Learned Positional | $L \times d$ |
| Sinusoidal | 0 (but fixed in embeddings) |
| Relative Bias | $O(L)$ to $O(L^2)$ |
| **ALiBi** | **0** |

#### 2. Excellent Length Extrapolation

Models trained with ALiBi on short sequences generalize to much longer ones:

| Training Length | Evaluation Length | Perplexity Increase |
|-----------------|-------------------|---------------------|
| 1024 | 2048 | ~2% |
| 1024 | 4096 | ~5% |
| 1024 | 8192 | ~10% |

#### 3. Simple Integration

Just add bias to attention scores—no architecture changes:

```python
attn_scores = q @ k.T / sqrt(d_k)
attn_scores = attn_scores + alibi_bias  # Only change!
attn_weights = softmax(attn_scores)
```

#### Comparison with Other Methods

| Method | Parameters | Extrapolation | Relative Position | Complexity |
|--------|------------|---------------|-------------------|------------|
| Sinusoidal | 0 | Moderate | Implicit | O(1) |
| Learned | L×d | Poor | Implicit | O(1) |
| T5 Relative | Buckets | Good | Explicit | O(n²) |
| RoPE | 0 | Good | Explicit | O(n) |
| **ALiBi** | **0** | **Excellent** | **Explicit** | **O(n²) precompute** |

#### Summary

ALiBi provides a simple yet effective positional encoding:

1. **No parameters**: Just geometric slopes
2. **Length extrapolation**: Excellent generalization to longer sequences
3. **Easy implementation**: Add bias matrix to attention scores
4. **Proven at scale**: Used in BLOOM (176B), MPT, and others

#### References

1. Press, O., et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR.
2. Scao, T., et al. (2022). "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model."
3. MosaicML (2023). "MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs."
