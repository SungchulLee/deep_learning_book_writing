# GPT: Generative Pre-trained Transformer

## Introduction

GPT (Generative Pre-trained Transformer) is a decoder-only Transformer architecture designed for autoregressive language modeling. Unlike BERT's bidirectional approach, GPT uses unidirectional (left-to-right) attention, making it naturally suited for text generation tasks.

## Evolution of GPT

| Model | Parameters | Training Data | Context Length |
|-------|------------|---------------|----------------|
| GPT-1 (2018) | 117M | BooksCorpus | 512 |
| GPT-2 (2019) | 1.5B | WebText | 1024 |
| GPT-3 (2020) | 175B | 300B tokens | 2048 |
| GPT-4 (2023) | ~1.8T* | Undisclosed | 8K-128K |

*Estimated from reports

## Architecture

GPT uses a stack of decoder-only Transformer blocks with causal self-attention:

$$
\text{GPT} = \text{TransformerDecoder}^L
$$

### Key Differences from BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (unidirectional) | Bidirectional |
| Pre-training | Next token prediction | Masked language modeling |
| Primary use | Generation | Understanding |

### Model Configuration (GPT-2)

| Model | Layers | Heads | $d_{\text{model}}$ | Parameters |
|-------|--------|-------|---------|------------|
| Small | 12 | 12 | 768 | 117M |
| Medium | 24 | 16 | 1024 | 345M |
| Large | 36 | 20 | 1280 | 762M |
| XL | 48 | 25 | 1600 | 1.5B |

## Pre-training Objective

GPT uses standard language modeling (next token prediction):

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, \ldots, x_{t-1}; \theta)
$$

Where the probability is computed via softmax over the vocabulary:

$$
P(x_t | x_{<t}) = \text{softmax}(h_t W_e^T)
$$

Here $h_t$ is the hidden state at position $t$ and $W_e$ is the embedding matrix (weight tying).

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List


class GPTConfig:
    """GPT model configuration."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: int = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.n_positions, config.n_positions))
                .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV-cache."""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Handle cached KV
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present = (k, v) if use_cache else None
        
        # Attention
        kv_seq_len = k.size(2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.bias[:, :, kv_seq_len - seq_len:kv_seq_len, :kv_seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        output = self.c_proj(output)
        output = self.resid_dropout(output)
        
        return output, present


class GPTBlock(nn.Module):
    """Single GPT Transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            nn.GELU(),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with pre-norm architecture."""
        attn_output, present = self.attn(self.ln_1(x), layer_past, use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTModel(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # LM head (weight tied)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        past_length = past_key_values[0][0].size(2) if past_key_values else 0
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Embeddings
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        
        # Blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values else None
            hidden_states, present = block(hidden_states, layer_past, use_cache)
            if use_cache:
                presents.append(present)
        
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits, 'past_key_values': presents}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        past_key_values = None
        
        for _ in range(max_new_tokens):
            model_input = input_ids[:, -1:] if past_key_values else input_ids
            
            outputs = self.forward(model_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs['logits'][:, -1, :] / temperature
            past_key_values = outputs['past_key_values']
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample
            if do_sample:
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


# Sampling strategies
class SamplingStrategies:
    """Text generation sampling strategies."""
    
    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decoding."""
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    @staticmethod
    def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Temperature sampling."""
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Top-k sampling."""
        logits = logits / temperature
        values, indices = torch.topk(logits, k, dim=-1)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, indices, values)
        probs = F.softmax(logits_filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @staticmethod
    def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """Nucleus (top-p) sampling."""
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# Example usage
if __name__ == "__main__":
    config = GPTConfig(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    model = GPTModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    generated = model.generate(input_ids[:, :10], max_new_tokens=30, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## In-Context Learning

GPT-3 introduced in-context learning where the model adapts to tasks through examples in the prompt:

### Zero-Shot
```
Translate English to French:
sea otter =>
```

### Few-Shot
```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese =>
```

## Scaling Laws

GPT-3 demonstrated predictable scaling:

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

Where $N$ is parameters, suggesting larger models are more sample-efficient.

## Summary

GPT established the autoregressive language modeling paradigm:

1. **Decoder-Only Architecture**: Simpler than encoder-decoder
2. **Causal Attention**: Natural for generation
3. **Scaling**: Larger models show emergent capabilities
4. **In-Context Learning**: Task adaptation without fine-tuning

## References

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
2. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
3. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models."
