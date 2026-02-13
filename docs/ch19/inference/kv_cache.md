# KV-Cache and Inference Optimization

## Introduction

KV-Cache (Key-Value Cache) is a fundamental optimization technique for autoregressive text generation that eliminates redundant computation by caching key and value tensors from previous tokens. This reduces inference time from O(N²) to O(N) per generated token.

## The Redundant Computation Problem

### Without KV-Cache

During autoregressive generation, each new token requires recomputing attention for the entire sequence:

```
Generate token 1: Compute attention for [prompt]
Generate token 2: Compute attention for [prompt, token1]  # Recomputes prompt!
Generate token 3: Compute attention for [prompt, token1, token2]  # Recomputes both!
...
Generate token N: Compute attention for entire sequence
```

**Total attention operations**: O(N³)

### With KV-Cache

Cache K and V from previous positions, only compute for new token:

```
Generate token 1: Compute K, V for [prompt], cache them
Generate token 2: Load cached K, V; compute only for token1
Generate token 3: Load cached K, V; compute only for token2
...
```

**Total attention operations**: O(N²)

## Mathematical Foundation

At timestep $t$, given new token $x_t$ and cached $K_{1:t-1}$, $V_{1:t-1}$:

$$
\begin{aligned}
q_t &= x_t W^Q \\
k_t &= x_t W^K \\
v_t &= x_t W^V \\
K_{1:t} &= [K_{1:t-1}; k_t] \\
V_{1:t} &= [V_{1:t-1}; v_t] \\
\text{out}_t &= \text{softmax}\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right)V_{1:t}
\end{aligned}
$$

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class KVCache:
    """Container for cached key-value pairs."""
    key: torch.Tensor
    value: torch.Tensor
    
    @property
    def seq_len(self) -> int:
        return self.key.size(2)


class AttentionWithKVCache(nn.Module):
    """Causal self-attention with KV-cache."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Update cache
        if kv_cache is not None:
            k = torch.cat([kv_cache.key, k], dim=2)
            v = torch.cat([kv_cache.value, v], dim=2)
        
        new_cache = KVCache(k, v) if use_cache else None
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask (only for multiple tokens)
        if seq_len > 1:
            total_len = k.size(2)
            mask = torch.triu(torch.ones(seq_len, total_len, device=x.device), 
                            diagonal=total_len - seq_len + 1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(output), new_cache


class GPTWithKVCache(nn.Module):
    """GPT model with KV-cache for generation."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            self._make_block(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    
    def _make_block(self, d_model, num_heads, d_ff):
        return nn.ModuleDict({
            'attn': AttentionWithKVCache(d_model, num_heads),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model)
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model)
        })
    
    def forward(self, input_ids, past_caches=None, use_cache=True):
        batch_size, seq_len = input_ids.shape
        past_len = past_caches[0].seq_len if past_caches else 0
        
        pos_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        
        if past_caches is None:
            past_caches = [None] * self.num_layers
        
        new_caches = []
        for i, block in enumerate(self.blocks):
            residual = x
            x = block['norm1'](x)
            x, cache = block['attn'](x, past_caches[i], use_cache)
            x = residual + x
            x = x + block['ffn'](block['norm2'](x))
            new_caches.append(cache)
        
        return self.lm_head(self.ln_f(x)), new_caches if use_cache else None
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        logits, caches = self(prompt_ids, use_cache=True)
        generated = prompt_ids
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float('-inf')
            
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)
            logits, caches = self(next_token, past_caches=caches, use_cache=True)
        
        return generated


# Example
if __name__ == "__main__":
    model = GPTWithKVCache(vocab_size=1000, d_model=256, num_heads=4, 
                           num_layers=4, d_ff=1024)
    
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated: {prompt.shape} -> {generated.shape}")
```

## Memory Considerations

### Cache Size per Layer

$$
\text{Size} = 2 \times B \times H \times L \times d_h \times \text{bytes}
$$

Where $B$ = batch, $H$ = heads, $L$ = sequence length, $d_h$ = head dimension.

### Example: LLaMA-7B

| Sequence Length | Cache Size |
|-----------------|------------|
| 2K | ~1 GB |
| 8K | ~4 GB |
| 32K | ~16 GB |

## Advanced Optimizations

### PagedAttention (vLLM)

Manages KV-cache like virtual memory pages:
- Non-contiguous memory allocation
- Memory sharing across sequences
- Reduces memory fragmentation

### Grouped Query Attention (GQA)

Shares KV heads across multiple query heads:
- Reduces cache size by factor of group size
- Used in LLaMA 2, Mistral

### Sliding Window Cache

Only cache recent tokens for long sequences.

## Summary

KV-Cache is essential for efficient LLM inference:

1. **Eliminates redundant computation**: O(N²) total instead of O(N³)
2. **Memory trade-off**: Faster inference but requires more memory
3. **Foundation for optimization**: Enables batching, speculative decoding

## References

1. Pope, R., et al. (2022). "Efficiently Scaling Transformer Inference."
2. Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention."
