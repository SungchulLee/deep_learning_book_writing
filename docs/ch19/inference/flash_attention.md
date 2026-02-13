# Flash Attention

## Introduction

Flash Attention, introduced by Dao et al. (2022), is an IO-aware exact attention algorithm that reduces memory usage from O(N²) to O(N) while also being 2-4x faster than standard attention. It achieves this by carefully orchestrating memory access patterns on GPUs.

## The Memory Bottleneck

### Standard Attention Memory Issue

Standard attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This requires materializing the full $N \times N$ attention matrix:

```
Memory usage: O(N²) for storing attention scores
```

For a 100K token sequence with float16: $100K \times 100K \times 2 = 20$ GB just for attention!

### IO Complexity

GPU operations are often memory-bound, not compute-bound:

| Operation | Compute | Memory Access |
|-----------|---------|---------------|
| Matrix multiply | High | Moderate |
| Softmax | Low | High (read/write full matrix) |
| Dropout | Very low | High |

Flash Attention optimizes for **memory IO**, not FLOPs.

## Key Ideas

### 1. Tiling

Instead of computing full attention, process in tiles:

```
For each tile of Q:
    For each tile of K, V:
        Compute partial attention for this tile
        Update running statistics (online softmax)
```

### 2. Online Softmax

Standard softmax requires two passes:
1. Compute max for numerical stability
2. Compute exp and normalize

Online softmax computes incrementally without storing full matrix:

$$
m_{new} = \max(m_{old}, \max(\mathbf{x}_{new}))
$$

$$
\ell_{new} = e^{m_{old} - m_{new}} \ell_{old} + \sum e^{x_i - m_{new}}
$$

### 3. Recomputation

During backward pass, recompute attention weights instead of storing them:
- Saves memory: Don't store O(N²) attention matrix
- Slightly more compute, but faster due to less memory IO

## Algorithm

### Forward Pass

```
Algorithm: Flash Attention Forward

Input: Q, K, V ∈ ℝ^{N×d}, block sizes Br, Bc
Output: O ∈ ℝ^{N×d}

1. Initialize O = 0, ℓ = 0, m = -∞ (running statistics)
2. Divide Q into Tr blocks of size Br
3. Divide K, V into Tc blocks of size Bc

4. for j = 1 to Tc:  # Iterate over K, V blocks
       Load Kj, Vj from HBM to SRAM
       
       for i = 1 to Tr:  # Iterate over Q blocks
           Load Qi, Oi, ℓi, mi from HBM to SRAM
           
           # Compute attention for this block
           Sij = Qi @ Kj.T / √d
           
           # Update running max
           mij = rowmax(Sij)
           Pij = exp(Sij - mij)
           ℓij = rowsum(Pij)
           
           # Update running statistics
           mi_new = max(mi, mij)
           ℓi_new = exp(mi - mi_new) * ℓi + exp(mij - mi_new) * ℓij
           
           # Update output
           Oi = (ℓi * exp(mi - mi_new) * Oi + exp(mij - mi_new) * Pij @ Vj) / ℓi_new
           
           # Update statistics
           mi = mi_new
           ℓi = ℓi_new
           
           Store Oi, ℓi, mi to HBM

5. Return O
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def flash_attention_forward_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    causal: bool = False
) -> torch.Tensor:
    """
    Reference implementation of Flash Attention (for understanding).
    
    Note: This is a simplified version for educational purposes.
    Real Flash Attention is implemented in CUDA for efficiency.
    
    Args:
        Q: Query [batch, heads, seq_len, head_dim]
        K: Key [batch, heads, seq_len, head_dim]
        V: Value [batch, heads, seq_len, head_dim]
        block_size: Size of tiles for blocking
        causal: Whether to apply causal masking
        
    Returns:
        Output [batch, heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim ** -0.5
    
    # Number of blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Initialize output and statistics
    O = torch.zeros_like(Q)
    L = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)
    M = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=Q.device)
    
    # Process in blocks
    for j in range(num_blocks):
        # Key-Value block indices
        kv_start = j * block_size
        kv_end = min((j + 1) * block_size, seq_len)
        
        Kj = K[:, :, kv_start:kv_end, :]
        Vj = V[:, :, kv_start:kv_end, :]
        
        for i in range(num_blocks):
            # Query block indices
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            
            Qi = Q[:, :, q_start:q_end, :]
            
            # Compute attention scores for this block
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale
            
            # Apply causal mask if needed
            if causal:
                # Create mask for this block
                q_positions = torch.arange(q_start, q_end, device=Q.device)
                kv_positions = torch.arange(kv_start, kv_end, device=Q.device)
                mask = q_positions.unsqueeze(1) < kv_positions.unsqueeze(0)
                Sij = Sij.masked_fill(mask, float('-inf'))
            
            # Online softmax update
            # Current block max
            mij = Sij.max(dim=-1, keepdim=True).values
            
            # Load previous statistics
            mi = M[:, :, q_start:q_end, :]
            li = L[:, :, q_start:q_end, :]
            oi = O[:, :, q_start:q_end, :]
            
            # New max
            mi_new = torch.maximum(mi, mij)
            
            # Compute attention weights with numerical stability
            Pij = torch.exp(Sij - mi_new)
            lij = Pij.sum(dim=-1, keepdim=True)
            
            # Update running sum
            li_new = torch.exp(mi - mi_new) * li + lij
            
            # Update output
            oi_new = (
                torch.exp(mi - mi_new) * li * oi + 
                torch.matmul(Pij, Vj)
            ) / li_new
            
            # Store updated values
            O[:, :, q_start:q_end, :] = oi_new
            L[:, :, q_start:q_end, :] = li_new
            M[:, :, q_start:q_end, :] = mi_new
    
    return O


class FlashAttention(nn.Module):
    """
    Flash Attention module using PyTorch's optimized implementation.
    
    Uses torch.nn.functional.scaled_dot_product_attention which
    automatically uses Flash Attention when available.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.dropout = dropout
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass using Flash Attention.
        
        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional mask
            
        Returns:
            Output [batch, seq_len, d_model]
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
        
        # Use PyTorch's optimized SDPA (includes Flash Attention)
        # This automatically selects the best implementation
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal
        )
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output


class FlashAttentionWithKVCache(nn.Module):
    """
    Flash Attention with KV-cache support for generation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        # Flash attention
        # For generation with cache, only query is short
        is_causal = (past_key_value is None and seq_len > 1)
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)
        
        return output, present_key_value


def benchmark_attention(seq_lengths: list, d_model: int = 512, num_heads: int = 8):
    """Benchmark Flash Attention vs Standard Attention."""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head_dim = d_model // num_heads
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Create inputs
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device)
        
        # Standard attention
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            out_standard = torch.matmul(attn, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_standard = (time.time() - start) / 10
        
        # Flash attention (via SDPA)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        for _ in range(10):
            out_flash = F.scaled_dot_product_attention(q, k, v)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time_flash = (time.time() - start) / 10
        
        # Memory usage
        memory_standard = seq_len * seq_len * 4 / (1024 ** 2)  # MB for float32 attention matrix
        
        print(f"  Standard: {time_standard*1000:.2f}ms, ~{memory_standard:.1f}MB attention matrix")
        print(f"  Flash:    {time_flash*1000:.2f}ms (no attention matrix stored)")
        print(f"  Speedup:  {time_standard/time_flash:.2f}x")
        
        results.append({
            'seq_len': seq_len,
            'standard_ms': time_standard * 1000,
            'flash_ms': time_flash * 1000,
            'speedup': time_standard / time_flash
        })
    
    return results


# Example usage
if __name__ == "__main__":
    print("Flash Attention Demo")
    print("=" * 50)
    
    # Configuration
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 1024
    
    # Test Flash Attention module
    flash_attn = FlashAttention(d_model, num_heads, causal=True)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = flash_attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with KV cache
    print("\n--- Testing with KV Cache ---")
    flash_attn_cache = FlashAttentionWithKVCache(d_model, num_heads)
    
    # Process prompt
    prompt = torch.randn(1, 100, d_model)
    _, kv_cache = flash_attn_cache(prompt, use_cache=True)
    print(f"Prompt processed, cache size: {kv_cache[0].shape}")
    
    # Generate tokens
    for i in range(5):
        new_token = torch.randn(1, 1, d_model)
        out, kv_cache = flash_attn_cache(new_token, past_key_value=kv_cache, use_cache=True)
        print(f"Generated token {i+1}, cache size: {kv_cache[0].shape}")
    
    # Reference implementation test
    print("\n--- Reference Implementation Test ---")
    Q = torch.randn(1, 4, 64, 32)
    K = torch.randn(1, 4, 64, 32)
    V = torch.randn(1, 4, 64, 32)
    
    # Standard attention
    scale = 32 ** -0.5
    standard_out = torch.matmul(
        F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1),
        V
    )
    
    # Flash attention reference
    flash_out = flash_attention_forward_reference(Q, K, V, block_size=16)
    
    # Check they match
    max_diff = (standard_out - flash_out).abs().max().item()
    print(f"Max difference between standard and flash: {max_diff:.2e}")
    print("✓ Outputs match!" if max_diff < 1e-5 else "✗ Outputs differ!")
    
    # Benchmark (if GPU available)
    if torch.cuda.is_available():
        print("\n--- Benchmarking ---")
        benchmark_attention([512, 1024, 2048, 4096])
```

## Memory and Speed Comparison

### Memory Usage

| Sequence Length | Standard Attention | Flash Attention |
|-----------------|-------------------|-----------------|
| 1K | 4 MB | O(block_size) |
| 4K | 64 MB | O(block_size) |
| 16K | 1 GB | O(block_size) |
| 64K | 16 GB | O(block_size) |

### Speed (Typical)

| Sequence Length | Speedup |
|-----------------|---------|
| 512 | 1.5-2x |
| 2048 | 2-3x |
| 8192 | 3-4x |

## Flash Attention 2

Flash Attention 2 introduces further optimizations:

1. **Better parallelization**: Split across sequence length, not batch
2. **Reduced non-matmul FLOPs**: Minimize softmax operations
3. **Better work partitioning**: Optimize for GPU occupancy

Typical speedup: **2x over Flash Attention 1**

## Integration in PyTorch

```python
# PyTorch 2.0+ automatically uses Flash Attention when possible
output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True  # Enables fused causal mask
)

# Check which backend is used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v)
```

## Summary

Flash Attention revolutionizes attention computation:

1. **O(N) memory**: No need to store N×N attention matrix
2. **2-4x faster**: IO-aware algorithm reduces memory bandwidth
3. **Exact computation**: Same results as standard attention
4. **Long sequences**: Enables training on 100K+ token sequences

## References

1. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.
2. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
3. PyTorch Documentation: scaled_dot_product_attention
