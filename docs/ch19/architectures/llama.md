# LLaMA Family: Open Foundation Models

## Learning Objectives

- Understand LLaMA's design philosophy prioritizing inference efficiency
- Compare LLaMA, LLaMA-2, and LLaMA-3 architectural choices
- Implement key LLaMA components in PyTorch
- Analyze the impact of open weights on the LLM ecosystem

## Introduction

LLaMA (Large Language Model Meta AI) represents Meta's contribution to open-source large language models. Unlike GPT models optimized for peak capability, LLaMA prioritizes inference efficiency—achieving strong performance with smaller, more deployable models.

## LLaMA-1 (February 2023)

### Design Philosophy

**Key insight**: Train smaller models longer on more data.

From Chinchilla scaling laws, 7B parameters optimal for ~140B tokens. LLaMA trains 7B on 1T+ tokens—"over-training" for better inference cost.

### Model Variants

| Model | Parameters | Training Tokens | Context |
|-------|------------|-----------------|---------|
| LLaMA-7B | 6.7B | 1.0T | 2048 |
| LLaMA-13B | 13.0B | 1.0T | 2048 |
| LLaMA-33B | 32.5B | 1.4T | 2048 |
| LLaMA-65B | 65.2B | 1.4T | 2048 |

### Architecture

```python
llama1_7b_config = {
    'dim': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'vocab_size': 32000,
    'multiple_of': 256,  # FFN hidden dim alignment
    'norm_eps': 1e-6,
    
    # Key innovations
    'normalization': 'RMSNorm',
    'activation': 'SwiGLU',
    'position_encoding': 'RoPE',
    'attention': 'Standard MHA'
}
```

### Training Data

```python
llama1_training_data = {
    'CommonCrawl': {'tokens': '67%', 'epochs': 1.1},
    'C4': {'tokens': '15%', 'epochs': 1.06},
    'GitHub': {'tokens': '4.5%', 'epochs': 0.64},
    'Wikipedia': {'tokens': '4.5%', 'epochs': 2.45},
    'Books': {'tokens': '4.5%', 'epochs': 2.23},
    'ArXiv': {'tokens': '2.5%', 'epochs': 1.06},
    'StackExchange': {'tokens': '2%', 'epochs': 1.03},
    'total_tokens': '1.4T'
}
```

## LLaMA-2 (July 2023)

### Improvements Over LLaMA-1

1. **Doubled context**: 2048 → 4096 tokens
2. **More training data**: 1.4T → 2T tokens
3. **Grouped-Query Attention**: For 34B and 70B models
4. **Commercial license**: Open for most uses

### Architecture Updates

```python
llama2_70b_config = {
    'dim': 8192,
    'n_layers': 80,
    'n_heads': 64,
    'n_kv_heads': 8,  # GQA: 8 KV heads, 64 query heads
    'vocab_size': 32000,
    'context_length': 4096,
    'norm_eps': 1e-5,
    
    # FFN dimension
    'intermediate_size': 28672,  # ~3.5x hidden
    
    # GQA reduces KV cache by 8x
    'kv_cache_reduction': 8
}
```

### Grouped-Query Attention Impact

```python
def compute_kv_cache_size(config, batch_size, seq_len, dtype_bytes=2):
    """Compare KV cache sizes: MHA vs GQA."""
    
    # MHA: Full KV heads
    mha_size = (2 * batch_size * config['n_heads'] * seq_len * 
                (config['dim'] // config['n_heads']) * dtype_bytes)
    
    # GQA: Reduced KV heads
    gqa_size = (2 * batch_size * config['n_kv_heads'] * seq_len *
                (config['dim'] // config['n_heads']) * dtype_bytes)
    
    return {
        'mha_gb': mha_size / 1e9,
        'gqa_gb': gqa_size / 1e9,
        'reduction': mha_size / gqa_size
    }

# LLaMA-2 70B, batch=1, seq=4096
# MHA: ~2.6GB, GQA: ~0.33GB (8x reduction)
```

### LLaMA-2-Chat: Aligned Variants

```python
def llama2_chat_format(messages):
    """
    LLaMA-2-Chat uses specific prompt format.
    
    [INST] <<SYS>>
    {system_prompt}
    <</SYS>>
    
    {user_message} [/INST] {assistant_response}
    """
    system = messages[0]['content'] if messages[0]['role'] == 'system' else ""
    
    formatted = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    
    for msg in messages[1:]:
        if msg['role'] == 'user':
            formatted += f"{msg['content']} [/INST] "
        else:
            formatted += f"{msg['content']} </s><s>[INST] "
    
    return formatted
```

## LLaMA-3 (April 2024)

### Major Improvements

1. **Tokenizer**: 32K → 128K vocabulary (4x)
2. **Context**: 4K → 8K tokens
3. **Training**: 2T → 15T+ tokens
4. **GQA everywhere**: All model sizes use GQA

### Architecture

```python
llama3_8b_config = {
    'dim': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'n_kv_heads': 8,  # GQA even for 8B
    'vocab_size': 128256,  # Much larger tokenizer
    'context_length': 8192,
    'rope_theta': 500000,  # Extended RoPE base
    'norm_eps': 1e-5,
}

llama3_70b_config = {
    'dim': 8192,
    'n_layers': 80,
    'n_heads': 64,
    'n_kv_heads': 8,
    'vocab_size': 128256,
    'context_length': 8192,
    'rope_theta': 500000,
}
```

### Tokenizer Improvements

```python
def tokenizer_efficiency_comparison():
    """
    Larger vocabulary = fewer tokens per text = faster inference.
    """
    example = "The quick brown fox jumps over the lazy dog."
    
    # Approximate token counts
    tokenization = {
        'LLaMA-1/2 (32K vocab)': 11,
        'LLaMA-3 (128K vocab)': 9,
        'efficiency_gain': '18% fewer tokens'
    }
    
    # Non-English improvement even larger
    chinese_example = "人工智能正在改变世界"
    chinese_tokens = {
        'LLaMA-2': 15,  # Character-level fallback
        'LLaMA-3': 6,   # Better multilingual coverage
        'efficiency_gain': '60% fewer tokens'
    }
    
    return tokenization, chinese_tokens
```

## PyTorch Implementation

### Complete LLaMA Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    
    def forward(self, x, positions):
        cos = self.cos[positions].unsqueeze(1)  # (seq, 1, dim/2)
        sin = self.sin[positions].unsqueeze(1)
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class LLaMAAttention(nn.Module):
    """LLaMA attention with GQA and RoPE."""
    
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.n_kv_heads = config.get('n_kv_heads', config['n_heads'])
        self.head_dim = config['dim'] // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config['dim'], self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config['dim'], self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config['dim'], self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config['dim'], bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, config['context_length'])
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        B, L, _ = x.shape
        
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = self.rotary(q, positions)
        k = self.rotary(k, positions)
        
        # KV cache handling
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv_cache = (k, v)
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        # Attention
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out), new_kv_cache


class LLaMAMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config['dim'] * 8 / 3)
        hidden_dim = config.get('intermediate_size', 
                                config['multiple_of'] * ((hidden_dim + config['multiple_of'] - 1) // config['multiple_of']))
        
        self.w1 = nn.Linear(config['dim'], hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config['dim'], bias=False)
        self.w3 = nn.Linear(config['dim'], hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(nn.Module):
    """Complete LLaMA transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config['dim'], config['norm_eps'])
        self.attention = LLaMAAttention(config)
        self.ffn_norm = RMSNorm(config['dim'], config['norm_eps'])
        self.ffn = LLaMAMLP(config)
    
    def forward(self, x, positions, mask=None, kv_cache=None):
        h, new_kv = self.attention(self.attention_norm(x), positions, mask, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class LLaMA(nn.Module):
    """Complete LLaMA model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config['vocab_size'], config['dim'])
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config['n_layers'])])
        self.norm = RMSNorm(config['dim'], config['norm_eps'])
        self.output = nn.Linear(config['dim'], config['vocab_size'], bias=False)
        
        # Weight tying
        self.output.weight = self.tok_embeddings.weight
    
    def forward(self, tokens, positions=None, kv_caches=None):
        B, L = tokens.shape
        
        if positions is None:
            positions = torch.arange(L, device=tokens.device)
        
        h = self.tok_embeddings(tokens)
        
        # Causal mask
        mask = torch.triu(torch.full((L, L), float('-inf'), device=tokens.device), diagonal=1)
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            h, new_kv = layer(h, positions, mask, kv_cache)
            new_kv_caches.append(new_kv)
        
        h = self.norm(h)
        logits = self.output(h)
        
        return logits, new_kv_caches
```

## LLaMA Ecosystem

### Fine-tuned Variants

| Model | Base | Method | Focus |
|-------|------|--------|-------|
| Alpaca | LLaMA-7B | SFT | Instructions |
| Vicuna | LLaMA-13B | SFT | Conversations |
| CodeLlama | LLaMA-2 | Continued PT | Code |
| Llama-Guard | LLaMA-2 | SFT | Safety |

### Quantized Versions

```python
quantization_options = {
    'GGUF/GGML': {
        'formats': ['Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q8_0'],
        'tool': 'llama.cpp',
        'use_case': 'CPU inference'
    },
    'GPTQ': {
        'bits': [4, 8],
        'tool': 'AutoGPTQ',
        'use_case': 'GPU inference'
    },
    'AWQ': {
        'bits': [4],
        'tool': 'AutoAWQ',
        'use_case': 'GPU inference (faster)'
    },
    'bitsandbytes': {
        'formats': ['int8', 'nf4'],
        'tool': 'transformers',
        'use_case': 'Training/inference'
    }
}
```

## Version Comparison

| Feature | LLaMA-1 | LLaMA-2 | LLaMA-3 |
|---------|---------|---------|---------|
| Release | Feb 2023 | Jul 2023 | Apr 2024 |
| Sizes | 7-65B | 7-70B | 8-70B |
| Context | 2K | 4K | 8K |
| Vocab | 32K | 32K | 128K |
| GQA | No | 34B+ | All |
| Training | 1.4T | 2T | 15T+ |
| License | Research | Commercial | Commercial |

## Summary

LLaMA family demonstrates that **open models can match proprietary performance** while enabling:

1. **Local deployment**: Run on consumer hardware
2. **Fine-tuning**: Full access to weights
3. **Research**: Reproducible experiments
4. **Customization**: Domain-specific adaptation

## Key Design Choices

$$\boxed{\text{LLaMA} = \text{Pre-Norm} + \text{RMSNorm} + \text{SwiGLU} + \text{RoPE} + \text{GQA}}$$

## References

1. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
2. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.
3. Meta AI. (2024). Introducing Meta Llama 3.
