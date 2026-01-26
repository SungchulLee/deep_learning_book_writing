# Transformer Decoder Block

## Overview

The Transformer decoder block is the building unit for autoregressive models like GPT. Unlike the encoder, the decoder uses **causal (masked) self-attention** to prevent positions from attending to future positions, enabling left-to-right generation.

## Architecture Variants

### Full Encoder-Decoder (Original Transformer)

The original decoder has three sub-layers:

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{CrossAttn}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

### Decoder-Only (GPT-style)

Modern language models use decoder-only architecture with two sub-layers:

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MaskedSelfAttn}(\mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

## Causal Self-Attention

The key difference from encoder self-attention is the **causal mask** that prevents attending to future positions.

### Causal Mask

For sequence length $n$, the causal mask $M$ is:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

This creates a lower-triangular attention pattern:

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

### Visual Representation

```
Position can attend to:
     1  2  3  4  5
1  [ ✓  ✗  ✗  ✗  ✗ ]
2  [ ✓  ✓  ✗  ✗  ✗ ]
3  [ ✓  ✓  ✓  ✗  ✗ ]
4  [ ✓  ✓  ✓  ✓  ✗ ]
5  [ ✓  ✓  ✓  ✓  ✓ ]
```

## Cross-Attention (Encoder-Decoder)

In sequence-to-sequence models, the decoder attends to encoder outputs:

$$
\text{CrossAttn}(Z, X_{\text{enc}}) = \text{softmax}\left(\frac{Z W^Q (X_{\text{enc}} W^K)^T}{\sqrt{d_k}}\right) X_{\text{enc}} W^V
$$

Where:
- Query: from decoder ($Z W^Q$)
- Key, Value: from encoder ($X_{\text{enc}} W^K$, $X_{\text{enc}} W^V$)

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention for Transformer Decoder.
    
    Each position can only attend to itself and previous positions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Pre-compute causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block (GPT-style).
    
    Consists of:
    1. Masked Self-Attention + Residual + LayerNorm
    2. Feed-Forward Network + Residual + LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_len: int = 2048,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # Masked self-attention
        self.self_attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        if self.pre_norm:
            # Pre-norm: LayerNorm before sublayer
            residual = x
            x = self.norm1(x)
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = residual + self.dropout(attn_out)
            
            residual = x
            x = self.norm2(x)
            ff_out = self.feed_forward(x)
            x = residual + ff_out
        else:
            # Post-norm: LayerNorm after sublayer
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Full Transformer Decoder (GPT-style, decoder-only).
    
    For language modeling and text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        # Output projection (weight tying with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for LM loss [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_dropout(x)
        
        # Padding mask
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        # Final norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


# Example usage
if __name__ == "__main__":
    # GPT-style decoder configuration
    model = TransformerDecoder(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024
    )
    
    # Sample input
    input_ids = torch.randint(0, 50257, (4, 128))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Generation
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

## KV-Cache for Efficient Inference

During generation, recomputing attention for all previous tokens is wasteful. KV-caching stores key and value tensors from previous steps:

```python
class CausalSelfAttentionWithCache(nn.Module):
    """Causal self-attention with KV-cache for efficient inference."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward with optional KV-cache.
        
        Args:
            x: Input [batch, seq_len, d_model] or [batch, 1, d_model] with cache
            past_kv: Cached (key, value) tensors from previous steps
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V for current input
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Concatenate with past KV if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Store for next iteration
        present_kv = (k, v) if use_cache else None
        
        # Attention (no causal mask needed when using cache for single token)
        full_seq_len = k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask only for initial context (seq_len > 1)
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_seq_len, device=x.device),
                diagonal=full_seq_len - seq_len + 1
            ).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        return output, present_kv
```

## Computational Analysis

### Without KV-Cache

For generating $T$ tokens with context $C$:

$$
\text{Complexity} = O(T \cdot (C + T)^2 \cdot d)
$$

### With KV-Cache

$$
\text{Complexity} = O(T \cdot (C + T) \cdot d)
$$

This reduces quadratic to linear complexity in the generation length.

## Decoder vs Encoder Comparison

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Attention Type | Bidirectional | Causal (masked) |
| Can see future | Yes | No |
| Primary use | Understanding | Generation |
| Examples | BERT, RoBERTa | GPT, LLaMA |

## Summary

The Transformer decoder block enables autoregressive generation through:

1. **Causal Masking**: Prevents attending to future positions
2. **Cross-Attention**: Connects to encoder in seq2seq models
3. **KV-Caching**: Enables efficient incremental generation
4. **Pre-norm Architecture**: Improves training stability

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
