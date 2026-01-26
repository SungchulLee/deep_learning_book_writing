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
