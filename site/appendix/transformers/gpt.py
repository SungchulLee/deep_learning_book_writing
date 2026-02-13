#!/usr/bin/env python3
"""
================================================================================
GPT - Generative Pre-trained Transformer
================================================================================

Paper: "Improving Language Understanding by Generative Pre-Training" (2018)
Authors: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)
Link: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
GPT introduced the paradigm of large-scale unsupervised pre-training followed
by supervised fine-tuning, which became the foundation for modern LLMs.

Evolution:
- GPT (2018): 117M parameters, 12 layers
- GPT-2 (2019): 1.5B parameters, 48 layers, "too dangerous to release"
- GPT-3 (2020): 175B parameters, few-shot learning
- GPT-4 (2023): Multimodal, undisclosed architecture

================================================================================
KEY DIFFERENCES FROM BERT
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ Aspect          │ BERT                    │ GPT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Architecture    │ Encoder-only            │ Decoder-only                   │
│ Attention       │ Bidirectional           │ Unidirectional (causal mask)   │
│ Pre-training    │ MLM + NSP               │ Autoregressive LM              │
│ Generation      │ Fill-in-the-blank       │ Left-to-right generation       │
│ Best for        │ Understanding tasks     │ Generation tasks               │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
CAUSAL (AUTOREGRESSIVE) LANGUAGE MODELING
================================================================================

GPT is trained to predict the next token given all previous tokens:
    P(x₁, x₂, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)

Causal Attention Mask:
    ┌─────────────────────────┐
    │ "The cat sat on"        │
    │   ↓   ↓   ↓   ↓         │
    │  The cat sat on         │
    │  [1   0   0   0] The    │  ← "The" only sees itself
    │  [1   1   0   0] cat    │  ← "cat" sees "The", "cat"
    │  [1   1   1   0] sat    │  ← "sat" sees "The", "cat", "sat"
    │  [1   1   1   1] on     │  ← "on" sees everything before it
    └─────────────────────────┘
    0 = masked (can't attend), 1 = visible

================================================================================
ARCHITECTURE (Pre-Norm Transformer Decoder)
================================================================================

GPT uses Pre-Norm (LayerNorm before attention/FFN) instead of Post-Norm:

    Pre-Norm (GPT-2+):           Post-Norm (Original Transformer):
    ┌────────────────┐           ┌────────────────┐
    │   LayerNorm    │           │  Attention     │
    │   Attention    │           │   Add          │
    │     Add        │           │  LayerNorm     │
    └────────────────┘           └────────────────┘

Pre-Norm benefits:
- More stable training
- Better gradient flow
- Enables training of very deep models

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Transformers for NLP (decoder architecture)
- Ch05: Language Modeling (autoregressive objective)
- Ch06: Large Language Models (GPT family)

Related: bert.py (encoder-only), transformer.py (encoder-decoder)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention for GPT
    
    Implements autoregressive attention where each position can only
    attend to previous positions (including itself).
    
    Args:
        d_model: Model dimension. Default: 768
        n_heads: Number of attention heads. Default: 12
        max_len: Maximum sequence length. Default: 1024
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super(CausalSelfAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask: lower triangular matrix
        # Register as buffer (not a parameter, but saved with model)
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask.view(1, 1, max_len, max_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal self-attention forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Attended output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V in one projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.resid_dropout(self.c_proj(out))


class GPTBlock(nn.Module):
    """
    Single GPT Transformer Block (Pre-Norm)
    
    Structure: LN → Attention → Add → LN → MLP → Add
    
    Args:
        d_model: Model dimension. Default: 768
        n_heads: Number of attention heads. Default: 12
        d_ff: Feed-forward dimension. Default: 3072
        max_len: Maximum sequence length. Default: 1024
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super(GPTBlock, self).__init__()
        
        # Layer norms (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Causal self-attention
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        
        # MLP (feed-forward network)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-2 uses GELU activation
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GPT block
        """
        # Self-attention with residual (pre-norm)
        x = x + self.attn(self.ln_1(x))
        
        # MLP with residual (pre-norm)
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(nn.Module):
    """
    GPT: Generative Pre-trained Transformer
    
    Decoder-only transformer for autoregressive language modeling.
    
    Args:
        vocab_size: Vocabulary size. Default: 50257 (GPT-2)
        d_model: Model dimension. Default: 768
        n_layers: Number of transformer layers. Default: 12
        n_heads: Number of attention heads. Default: 12
        d_ff: Feed-forward dimension. Default: 3072
        max_len: Maximum sequence length. Default: 1024
        dropout: Dropout rate. Default: 0.1
    
    Example:
        >>> model = GPT(vocab_size=50257)
        >>> x = torch.randint(0, 50257, (2, 100))
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([2, 100, 50257])
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1
    ):
        super(GPT, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, max_len, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection (shares weights with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: output projection shares weights with input embedding
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GPT
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Optional target IDs for computing loss
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Optional cross-entropy loss
        """
        batch_size, seq_len = input_ids.size()
        
        assert seq_len <= self.max_len, f"Sequence length {seq_len} > max {self.max_len}"
        
        # Position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0)  # (1, seq_len)
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, d_model)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively
        
        Args:
            input_ids: Starting token IDs (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample only from top-k tokens
            
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_len if necessary
            idx_cond = input_ids if input_ids.size(1) <= self.max_len else input_ids[:, -self.max_len:]
            
            # Get predictions
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature  # Last position
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# DEMO AND TESTING
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GPT Model Summary")
    print("=" * 70)
    
    model = GPT(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072
    )
    
    total_params, _ = count_parameters(model)
    print(f"GPT Configuration (similar to GPT-2 Small):")
    print(f"  vocab_size: 50257")
    print(f"  d_model: 768")
    print(f"  n_layers: 12")
    print(f"  n_heads: 12")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 100
    x = torch.randint(0, 50257, (batch_size, seq_len))
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    
    print(f"Output logits shape: {logits.shape}")
    print("=" * 70)
