# Sparse Attention Patterns

## Introduction

Standard self-attention has O(N²) complexity, making it prohibitive for long sequences. Sparse attention patterns reduce this by limiting which positions can attend to each other, achieving O(N) or O(N log N) complexity while maintaining model quality.

## The Complexity Problem

For sequence length $N$ and model dimension $d$:

| Component | Time Complexity | Memory |
|-----------|----------------|--------|
| Full Attention | O(N²d) | O(N²) |
| **Goal** | O(Nd) or O(N log N · d) | O(N) |

## Common Sparse Patterns

### 1. Local (Sliding Window) Attention

Each position attends only to nearby positions within a window:

$$
\text{Attend}(i) = \{j : |i - j| \leq w\}
$$

**Complexity**: O(Nw) where $w$ is window size

```
Window size = 3:
Position 5 attends to: [2, 3, 4, 5, 6, 7, 8]
```

### 2. Dilated (Strided) Attention

Attend to positions at fixed intervals:

$$
\text{Attend}(i) = \{j : (i - j) \mod d = 0, |i-j| \leq w \cdot d\}
$$

**Complexity**: O(Nw)

```
Dilation = 2, Window = 3:
Position 6 attends to: [0, 2, 4, 6, 8, 10, 12]
```

### 3. Global Attention

Designate certain positions as "global" that attend to/from all positions:

$$
\text{Attend}(i) = \begin{cases}
\{1, ..., N\} & \text{if } i \in \mathcal{G} \\
\mathcal{G} \cup \text{Local}(i) & \text{otherwise}
\end{cases}
$$

Used in Longformer, BigBird.

### 4. Block Sparse Attention

Divide sequence into blocks and attend within/between specific blocks:

```
Block pattern:
[■ ■ □ □ ■]
[■ ■ ■ □ □]
[□ ■ ■ ■ □]
[□ □ ■ ■ ■]
[■ □ □ ■ ■]
```

### 5. Random Attention

Each position randomly attends to a fixed number of other positions:

$$
\text{Attend}(i) = \text{Random}(k) \cup \text{Local}(i)
$$

## PyTorch Implementation

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
    Longformer-style attention combining local and global attention.
    
    - Most positions use sliding window attention
    - Global positions attend to/from all positions
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
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Global attention projections (separate)
        self.q_global = nn.Linear(d_model, d_model)
        self.k_global = nn.Linear(d_model, d_model)
        self.v_global = nn.Linear(d_model, d_model)
        
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        global_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with combined local and global attention.
        
        Args:
            x: Input [batch, seq_len, d_model]
            global_attention_mask: Boolean mask indicating global positions
                [batch, seq_len] where True = global attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Default: first token is global (like [CLS])
        if global_attention_mask is None:
            global_attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            global_attention_mask[:, 0] = True
        
        # Local attention for all positions
        q_local = self.q_proj(x)
        k_local = self.k_proj(x)
        v_local = self.v_proj(x)
        
        # Global projections for global positions
        q_global = self.q_global(x)
        k_global = self.k_global(x)
        v_global = self.v_global(x)
        
        # Reshape
        def reshape(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_local = reshape(q_local)
        k_local = reshape(k_local)
        v_local = reshape(v_local)
        q_global = reshape(q_global)
        k_global = reshape(k_global)
        v_global = reshape(v_global)
        
        # Compute local attention
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

## Complexity Comparison

| Method | Time | Memory | Global Context |
|--------|------|--------|----------------|
| Full Attention | O(N²) | O(N²) | ✓ Complete |
| Local Window | O(Nw) | O(Nw) | ✗ Limited |
| Longformer | O(Nw + Ng) | O(N) | ✓ Via global |
| BigBird | O(N) | O(N) | ✓ Random + global |

## Summary

Sparse attention patterns enable efficient processing of long sequences:

1. **Local attention**: Fast but limited context
2. **Global tokens**: Maintain full-sequence information
3. **Random attention**: Theoretical expressiveness guarantees
4. **Combined patterns**: Best of all approaches

## References

1. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer."
2. Zaheer, M., et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS.
3. Child, R., et al. (2019). "Generating Long Sequences with Sparse Transformers."
