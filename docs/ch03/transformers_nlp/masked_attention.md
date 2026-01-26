# Masked Self-Attention

## Introduction

Masked self-attention is the mechanism that enables autoregressive generation in Transformer decoders. By preventing each position from attending to future positions, it maintains the causal structure required for left-to-right sequence generation.

## The Causality Requirement

In language modeling, we predict the next token given all previous tokens:

$$
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^{n} P(x_t | x_1, \ldots, x_{t-1})
$$

This factorization requires that when predicting $x_t$, we can only use information from $x_1, \ldots, x_{t-1}$. Masked attention enforces this constraint during parallel training.

## Mathematical Formulation

### Standard Self-Attention

Without masking, attention allows each position to attend to all positions:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Causal (Masked) Self-Attention

We add a mask $M$ that sets future positions to $-\infty$ before softmax:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where the mask $M \in \mathbb{R}^{n \times n}$:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \text{ (can attend)} \\ -\infty & \text{if } j > i \text{ (masked)} \end{cases}
$$

### Effect on Attention Weights

After applying softmax with $-\infty$ masking:

$$
\text{softmax}(-\infty) = 0
$$

This ensures zero attention weight to future positions:

$$
\alpha_{ij} = \begin{cases} \frac{\exp(s_{ij})}{\sum_{k \leq i} \exp(s_{ik})} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}
$$

## Mask Construction

### Lower Triangular Mask

```python
def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Returns a mask where mask[i,j] = True means position i CANNOT attend to j.
    
    Example for seq_len=4:
    [[False,  True,  True,  True],
     [False, False,  True,  True],
     [False, False, False,  True],
     [False, False, False, False]]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_causal_mask_float(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal mask with -inf for masked positions.
    
    Returns a mask that can be added directly to attention scores.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device),
        diagonal=1
    )
    return mask
```

### Visualization

```
Attention Pattern for Position 4:
                    Position
              1    2    3    4    5
            ┌───┬───┬───┬───┬───┐
Position 4  │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │
            └───┴───┴───┴───┴───┘
              ↑                ↑
           Attend          Masked
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MaskedSelfAttention(nn.Module):
    """
    Masked (Causal) Self-Attention mechanism.
    
    Implements autoregressive attention where each position can only
    attend to itself and previous positions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_len: int = 2048
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Attention dropout probability
            max_len: Maximum sequence length for pre-computed mask
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute and register causal mask
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split tensor into multiple heads.
        
        Input: [batch, seq_len, d_model]
        Output: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads back to single tensor.
        
        Input: [batch, num_heads, seq_len, head_dim]
        Output: [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            padding_mask: Optional padding mask [batch_size, seq_len]
                True for positions to mask (padding tokens)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
            attention_weights: Optional [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Linear projections
        Q = self.W_q(x)  # [batch, seq, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Step 2: Split into heads
        Q = self._split_heads(Q)  # [batch, heads, seq, head_dim]
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Step 3: Scaled dot-product attention
        # [batch, heads, seq, seq]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Step 4: Apply causal mask (prevent attending to future)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        
        # Step 5: Apply padding mask (optional)
        if padding_mask is not None:
            # padding_mask: [batch, seq] -> [batch, 1, 1, seq]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                padding_mask,
                float('-inf')
            )
        
        # Step 6: Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Handle NaN from all-masked rows (shouldn't happen with causal mask)
        attention_weights = torch.nan_to_num(attention_weights, 0.0)
        
        # Step 7: Dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 8: Apply attention to values
        # [batch, heads, seq, head_dim]
        context = torch.matmul(attention_weights, V)
        
        # Step 9: Merge heads and final projection
        context = self._merge_heads(context)  # [batch, seq, d_model]
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


def visualize_causal_attention(seq_len: int = 10):
    """Visualize the causal attention pattern."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create sample attention weights (uniform within allowed positions)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    attention = torch.ones(seq_len, seq_len)
    attention = attention.masked_fill(mask, 0.0)
    
    # Normalize rows
    attention = attention / attention.sum(dim=-1, keepdim=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Causal mask
    ax1 = axes[0]
    im1 = ax1.imshow(mask.float().numpy(), cmap='Reds')
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_title('Causal Mask (Red = Masked)')
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    plt.colorbar(im1, ax=ax1)
    
    # Attention pattern
    ax2 = axes[1]
    im2 = ax2.imshow(attention.numpy(), cmap='Blues')
    ax2.set_xlabel('Key Position (j)')
    ax2.set_ylabel('Query Position (i)')
    ax2.set_title('Attention Pattern (After Masking)')
    ax2.set_xticks(range(seq_len))
    ax2.set_yticks(range(seq_len))
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('causal_attention_pattern.png', dpi=150)
    plt.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 20
    
    # Create module
    masked_attn = MaskedSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attn_weights = masked_attn(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Verify causality: attention weights should be lower triangular
    # Check that upper triangle (excluding diagonal) is all zeros
    for b in range(batch_size):
        for h in range(num_heads):
            upper = torch.triu(attn_weights[b, h], diagonal=1)
            assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6), \
                "Attention weights should be zero for future positions!"
    
    print("\n✓ Causality verified: No attention to future positions")
    
    # Show attention pattern for first head
    print("\nAttention pattern (first batch, first head):")
    pattern = attn_weights[0, 0].detach()
    print(f"Sum of each row (should be 1.0): {pattern.sum(dim=-1)[:5].tolist()}")
    
    # Visualize
    visualize_causal_attention(10)
    print("\nVisualization saved to 'causal_attention_pattern.png'")
```

## Combining Causal and Padding Masks

In practice, we often combine causal masking with padding masking:

```python
def create_combined_mask(
    seq_len: int,
    padding_mask: torch.Tensor,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create combined causal + padding mask.
    
    Args:
        seq_len: Sequence length
        padding_mask: [batch, seq] True for padding positions
        device: Device for mask tensor
        
    Returns:
        Combined mask [batch, 1, seq, seq]
    """
    # Causal mask: [seq, seq]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    
    # Expand to [1, 1, seq, seq]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Padding mask: [batch, seq] -> [batch, 1, 1, seq]
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    # Combine: mask if EITHER is true
    combined_mask = causal_mask | padding_mask
    
    return combined_mask
```

## Efficient Implementation with Flash Attention

For long sequences, Flash Attention provides memory-efficient causal attention:

```python
# Using PyTorch's scaled_dot_product_attention with is_causal=True
def efficient_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """
    Efficient causal attention using PyTorch's SDPA.
    
    Args:
        query, key, value: [batch, heads, seq, head_dim]
        dropout_p: Dropout probability
        
    Returns:
        Output tensor [batch, heads, seq, head_dim]
    """
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True  # Automatically applies causal mask
    )
```

## Prefix-LM: Bidirectional Prefix with Causal Suffix

Some models (like T5 decoder, UL2) use a hybrid approach:

```python
def create_prefix_lm_mask(
    prefix_len: int,
    total_len: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create Prefix-LM attention mask.
    
    Prefix positions (1 to prefix_len) can attend to each other bidirectionally.
    Suffix positions can attend to prefix and previous suffix positions.
    
    Args:
        prefix_len: Length of bidirectional prefix
        total_len: Total sequence length
        
    Returns:
        Mask tensor [total_len, total_len]
    """
    mask = torch.zeros(total_len, total_len, device=device)
    
    # Suffix positions cannot attend to future suffix
    suffix_mask = torch.triu(
        torch.ones(total_len - prefix_len, total_len - prefix_len, device=device),
        diagonal=1
    )
    mask[prefix_len:, prefix_len:] = suffix_mask
    
    return mask.bool()


# Example: Prefix-LM with 5-token prefix
# Positions 1-5: bidirectional (can see each other)
# Positions 6+: causal (can see prefix + previous suffix)
```

## Sliding Window Attention

For efficiency with long sequences, sliding window limits attention to nearby positions:

```python
def create_sliding_window_causal_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create sliding window causal attention mask.
    
    Each position can attend to at most window_size previous positions.
    
    Args:
        seq_len: Sequence length
        window_size: Size of attention window
        
    Returns:
        Mask tensor [seq_len, seq_len]
    """
    # Start with causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    
    # Also mask positions outside the window
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, :start] = 1
    
    return mask.bool()
```

## Training vs Inference

### Training (Parallel)

During training, all positions are computed in parallel with the causal mask:

```python
# All positions computed simultaneously
output = masked_attention(input_sequence)  # [batch, seq_len, d_model]
```

### Inference (Autoregressive)

During generation, tokens are generated one at a time:

```python
# Generate token by token
for t in range(max_tokens):
    # Only compute for the last position
    logits = model(generated_tokens)[:, -1, :]
    next_token = sample(logits)
    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
```

With KV-caching, we avoid recomputing attention for previous positions.

## Summary

Masked self-attention is essential for autoregressive models:

1. **Enforces Causality**: Prevents information flow from future to past
2. **Enables Parallel Training**: All positions computed simultaneously
3. **Maintains Generation Order**: Tokens generated left-to-right
4. **Combines with Other Masks**: Works with padding, sliding window, etc.

Understanding masked attention is crucial for implementing language models and text generation systems.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
3. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Attention."
4. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
