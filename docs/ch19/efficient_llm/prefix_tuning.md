# Prefix Tuning

## Learning Objectives

- Understand how prefix tuning adapts models without modifying weights
- Implement prefix tuning for both encoder and decoder models
- Compare prefix tuning with prompt tuning and other soft prompt methods
- Configure prefix length and reparameterization strategies

## Introduction

Prefix Tuning (Li & Liang, 2021) is a parameter-efficient fine-tuning method that prepends trainable continuous vectors (the "prefix") to the keys and values at every transformer layer. Unlike discrete prompts, these prefixes are learned embeddings that can encode task-specific information more efficiently.

## Core Concept

### Discrete vs Continuous Prompts

**Discrete prompts** (traditional):
```
"Translate English to French: The cat sat on the mat"
```
Limited to tokens in vocabulary, requires manual engineering.

**Continuous prompts** (prefix tuning):
```
[P1][P2][P3]...[Pn] "The cat sat on the mat"
```
Learned embeddings in continuous space, optimized end-to-end.

### How It Works

Instead of modifying attention weights, prefix tuning modifies what the attention sees:

$$
\text{Attention}(Q, [P_K; K], [P_V; V])
$$

Where $P_K, P_V$ are learnable prefix embeddings prepended to keys and values.

## Mathematical Foundation

### Standard Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Attention with Prefix

For prefix of length $l$:

$$
K' = [P_K; K] \in \mathbb{R}^{(l+n) \times d_k}
$$
$$
V' = [P_V; V] \in \mathbb{R}^{(l+n) \times d_v}
$$
$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{QK'^T}{\sqrt{d_k}}\right)V'
$$

The prefix entries effectively become "virtual tokens" that all real tokens can attend to.

### Parameter Count

For a model with $L$ layers, $H$ attention heads, and head dimension $d_h$:

$$
\text{Prefix params} = l \times L \times 2 \times H \times d_h = 2 \times l \times L \times d_{model}
$$

**Example**: GPT-2 Medium ($L=24$, $d_{model}=1024$) with prefix length 10:
- Prefix parameters: $2 \times 10 \times 24 \times 1024 = 491,520$ (0.14% of 345M)

## Implementation

### Basic Prefix Module

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class PrefixEncoder(nn.Module):
    """
    Encodes the prefix for all layers.
    
    Uses a small MLP to generate prefix embeddings from learnable
    input embeddings. This reparameterization improves training stability.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int,
        hidden_dim: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        
        # Total dimension: layers * 2 (K and V) * heads * head_dim
        self.total_dim = num_layers * 2 * num_heads * head_dim
        
        # Learnable prefix embeddings
        self.prefix_tokens = nn.Embedding(prefix_length, hidden_dim)
        
        # MLP for reparameterization (improves training)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prefix key-value pairs for all layers.
        
        Returns:
            prefix_keys: [batch, num_layers, num_heads, prefix_len, head_dim]
            prefix_values: [batch, num_layers, num_heads, prefix_len, head_dim]
        """
        device = self.prefix_tokens.weight.device
        
        # Get prefix token indices
        prefix_ids = torch.arange(self.prefix_length, device=device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed and transform
        prefix_emb = self.prefix_tokens(prefix_ids)  # [batch, prefix_len, hidden]
        prefix = self.mlp(prefix_emb)  # [batch, prefix_len, total_dim]
        prefix = self.dropout(prefix)
        
        # Reshape: [batch, prefix_len, layers, 2, heads, head_dim]
        prefix = prefix.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            2,
            self.num_heads,
            self.head_dim
        )
        
        # Permute to [batch, layers, heads, prefix_len, head_dim, 2]
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        # Split into keys and values
        prefix_keys = prefix[..., 0].contiguous()
        prefix_values = prefix[..., 1].contiguous()
        
        return prefix_keys, prefix_values


class PrefixTuningModel(nn.Module):
    """
    Wrapper that adds prefix tuning to a pre-trained model.
    
    Freezes base model and only trains prefix parameters.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 20,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.base_model = base_model
        self.prefix_length = prefix_length
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Create prefix encoder
        self.prefix_encoder = PrefixEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            prefix_length=prefix_length,
            hidden_dim=hidden_dim
        )
    
    def get_prefix(self, batch_size: int):
        """Get prefix key-value pairs."""
        return self.prefix_encoder(batch_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        batch_size = input_ids.size(0)
        
        # Generate prefix
        prefix_keys, prefix_values = self.get_prefix(batch_size)
        
        # Extend attention mask for prefix
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward with prefix (implementation depends on base model)
        # This is a simplified interface; actual implementation needs
        # to inject prefix_keys, prefix_values into each attention layer
        return self.base_model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._format_prefix(prefix_keys, prefix_values),
            **kwargs
        )
    
    def _format_prefix(self, prefix_keys, prefix_values):
        """Format prefix for HuggingFace-style past_key_values."""
        # Returns list of (key, value) tuples, one per layer
        past_key_values = []
        for layer_idx in range(prefix_keys.size(1)):
            layer_key = prefix_keys[:, layer_idx]  # [batch, heads, prefix_len, head_dim]
            layer_value = prefix_values[:, layer_idx]
            past_key_values.append((layer_key, layer_value))
        return tuple(past_key_values)
```

### Integrating with Attention Layers

```python
class PrefixAttention(nn.Module):
    """
    Multi-head attention with prefix support.
    
    Prepends prefix to keys and values before computing attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_key: Optional[torch.Tensor] = None,
        prefix_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward with optional prefix.
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len] or [batch, 1, seq_len, total_len]
            prefix_key: [batch, num_heads, prefix_len, head_dim]
            prefix_value: [batch, num_heads, prefix_len, head_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Prepend prefix to K and V
        if prefix_key is not None and prefix_value is not None:
            k = torch.cat([prefix_key, k], dim=2)  # [batch, heads, prefix+seq, head_dim]
            v = torch.cat([prefix_value, v], dim=2)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Extend mask for prefix (always attend to prefix)
            if prefix_key is not None:
                prefix_len = prefix_key.size(2)
                if attention_mask.dim() == 2:
                    # [batch, seq] -> [batch, 1, 1, prefix+seq]
                    prefix_mask = torch.ones(batch_size, prefix_len, device=attention_mask.device)
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
```

## Why Reparameterization?

Direct optimization of prefix embeddings can be unstable. The MLP reparameterization:

1. **Provides a better optimization landscape** - The MLP maps from a smaller space
2. **Enables weight sharing** - Single set of MLP weights generates all prefixes
3. **Improves training stability** - Gradients flow through MLP's nonlinearity

```python
class DirectPrefixEncoder(nn.Module):
    """
    Direct prefix parameterization (without MLP).
    
    Simpler but may be less stable during training.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_length = prefix_length
        
        # Direct learnable parameters
        # Shape: [prefix_len, layers, 2, heads, head_dim]
        self.prefix = nn.Parameter(
            torch.randn(prefix_length, num_layers, 2, num_heads, head_dim) * 0.01
        )
    
    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expand for batch
        prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        
        # Permute to [batch, layers, heads, prefix_len, head_dim, 2]
        prefix = prefix.permute(0, 2, 4, 1, 5, 3)
        
        return prefix[..., 0].contiguous(), prefix[..., 1].contiguous()
```

## Comparison: Prefix Tuning vs Prompt Tuning

| Aspect | Prefix Tuning | Prompt Tuning |
|--------|---------------|---------------|
| Where applied | Every layer (K, V) | Input embeddings only |
| Parameters | $2 \times l \times L \times d$ | $l \times d$ |
| Expressiveness | Higher | Lower |
| Performance | Better (especially small data) | Good for large data |
| Complexity | Higher | Simpler |

### Prompt Tuning Implementation

```python
class PromptTuning(nn.Module):
    """
    Prompt Tuning: Learnable embeddings prepended to input only.
    
    Simpler than prefix tuning but less expressive.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_virtual_tokens: int = 20,
        embedding_dim: int = 768,
        init_from_vocab: bool = True,
        init_text: str = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, embedding_dim)
        
        # Optional: initialize from vocabulary
        if init_from_vocab and init_text is not None:
            self._init_from_text(init_text)
    
    def _init_from_text(self, text: str):
        """Initialize prompt from text tokens (requires tokenizer)."""
        # Implementation would tokenize text and copy embeddings
        pass
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Get input embeddings from base model
        if hasattr(self.base_model, 'get_input_embeddings'):
            input_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            input_embeds = self.base_model.embed_tokens(input_ids)
        
        # Get prompt embeddings
        prompt_ids = torch.arange(self.num_virtual_tokens, device=device)
        prompt_embeds = self.prompt_embeddings(prompt_ids)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate prompt with input
        inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # Extend attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Forward through model
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
```

## Hyperparameter Guide

### Prefix Length

| Prefix Length | Parameters (GPT-2 Medium) | Use Case |
|---------------|---------------------------|----------|
| 5 | ~250K | Simple tasks |
| 10 | ~500K | Default |
| 20 | ~1M | Complex tasks |
| 50 | ~2.5M | High capacity |
| 100 | ~5M | Near fine-tuning |

**Rule of thumb**: Start with 10-20, increase if underfitting.

### Hidden Dimension

For the MLP reparameterization:
- Smaller hidden dim (256-512): Fewer parameters, may underfit
- Larger hidden dim (768-1024): More expressive, risk of overfitting

### Learning Rate

Typically higher than full fine-tuning:
- Prefix tuning: 1e-3 to 5e-3
- Full fine-tuning: 1e-5 to 5e-5

## Use Cases

### Best For

1. **Generation tasks** - Summarization, translation, dialogue
2. **Few-shot learning** - When training data is limited
3. **Multi-task learning** - Different prefix per task
4. **Preserving pretrained knowledge** - Frozen base model

### Less Suitable For

1. **Classification with many labels** - LoRA often better
2. **Very long sequences** - Prefix adds to sequence length
3. **Encoder-only tasks** - Originally designed for generation

## Complete Training Example

```python
def train_prefix_tuning(
    base_model,
    train_dataloader,
    eval_dataloader,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    prefix_length: int = 20,
    num_epochs: int = 5,
    learning_rate: float = 3e-3,
    device: str = 'cuda'
):
    """Train a model with prefix tuning."""
    
    # Create prefix-tuned model
    model = PrefixTuningModel(
        base_model=base_model,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        prefix_length=prefix_length
    ).to(device)
    
    # Only prefix parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer (only prefix parameters)
    optimizer = torch.optim.AdamW(
        model.prefix_encoder.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model
```

## Summary

| Aspect | Details |
|--------|---------|
| **Mechanism** | Prepend learnable K, V to every layer |
| **Parameters** | ~0.1-1% of model |
| **Best for** | Generation, few-shot, multi-task |
| **Key hyperparameter** | Prefix length (10-50) |
| **Training** | Higher LR than fine-tuning |

## References

1. Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL.
2. Lester, B., Al-Rfou, R., & Constant, N. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP.
3. Liu, X., et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks."
