#!/usr/bin/env python3
"""
================================================================================
Transformer - Attention Is All You Need
================================================================================

Paper: "Attention Is All You Need" (NeurIPS 2017)
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al. (Google)
Link: https://arxiv.org/abs/1706.03762

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
The Transformer architecture revolutionized NLP and later computer vision:

- **Eliminated Recurrence**: No more sequential processing (RNNs/LSTMs)
- **Parallelization**: Entire sequences processed simultaneously
- **Long-Range Dependencies**: Attention spans entire sequence
- **Foundation for Modern AI**: GPT, BERT, T5, ViT, DALL-E all use Transformers

Impact:
- Machine Translation: State-of-the-art performance
- Language Models: GPT series, BERT, LLaMA
- Vision: Vision Transformer (ViT), CLIP, DALL-E
- Multimodal: Unified architecture for text, image, audio

================================================================================
THE KEY INNOVATION: SELF-ATTENTION
================================================================================

Traditional RNNs process sequences step-by-step:
    h_t = f(h_{t-1}, x_t)  -- Sequential, cannot parallelize!

Transformers use Attention to relate all positions simultaneously:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Self-Attention:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│     Input: "The cat sat on the mat"                                         │
│                                                                             │
│     Each word attends to ALL other words:                                   │
│                                                                             │
│     "cat" → [The: 0.1, cat: 0.2, sat: 0.3, on: 0.1, the: 0.1, mat: 0.2]    │
│                        ↑                                                    │
│           "cat" most strongly attends to "sat" and "mat"                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
SCALED DOT-PRODUCT ATTENTION
================================================================================

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Where:
- Q (Query):  What am I looking for?      Shape: (seq_len, d_k)
- K (Key):    What do I contain?          Shape: (seq_len, d_k)  
- V (Value):  What do I return?           Shape: (seq_len, d_v)

Why scale by √d_k?
────────────────────────────────────────────────────────────────────────────────
Without scaling, dot products grow large for high d_k:
- For random vectors q, k with mean 0, var 1:
  E[q·k] = 0, Var[q·k] = d_k

Large dot products → softmax saturates → tiny gradients!

Scaling by √d_k keeps variance ≈ 1:
  Var[q·k / √d_k] = d_k / d_k = 1

================================================================================
MULTI-HEAD ATTENTION
================================================================================

Instead of one attention function, use h parallel attention "heads":

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

Benefits:
1. Different heads learn different types of relationships
2. More expressive than single attention
3. Computationally similar to single full-sized attention

Typical: h=8 heads, d_k = d_v = d_model/h = 64

================================================================================
TRANSFORMER ARCHITECTURE
================================================================================

ENCODER (Self-Attention):
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input Embedding + Positional Encoding                                      │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────┐                                                    │
│  │  Multi-Head         │◄────┐                                              │
│  │  Self-Attention     │     │ (residual)                                   │
│  └──────────┬──────────┘     │                                              │
│             │                │                                              │
│         Add & Norm ──────────┘                                              │
│             │                                                                │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │  Feed-Forward       │◄────┐                                              │
│  │  Network (FFN)      │     │ (residual)                                   │
│  └──────────┬──────────┘     │                                              │
│             │                │                                              │
│         Add & Norm ──────────┘                                              │
│             │                                                                │
│             ▼                                                               │
│       (Repeat N times)                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

DECODER (with Cross-Attention):
- Masked Self-Attention (prevents looking at future tokens)
- Cross-Attention (attends to encoder output)
- Feed-Forward Network

================================================================================
POSITIONAL ENCODING
================================================================================

Transformers have no inherent notion of position (unlike RNNs).
Solution: Add positional information to embeddings.

Sinusoidal Encoding (original paper):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Properties:
- Each position has unique encoding
- Relative positions can be computed via linear transformation
- Generalizes to longer sequences than seen during training

Alternatives:
- Learned positional embeddings (BERT, GPT)
- Rotary Position Embedding (RoPE) - used in LLaMA
- ALiBi (Attention with Linear Biases)

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================

**Self-Attention:**
    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    
    Q = X · W^Q  (d×d_k)
    K = X · W^K  (d×d_k)
    V = X · W^V  (d×d_v)

**Multi-Head Attention:**
    head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)
    MultiHead = Concat(head_1, ..., head_h) · W^O

**Feed-Forward Network:**
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Typically d_ff = 4 × d_model (expand then contract)

**Layer Normalization:**
    LN(x) = γ · (x - μ) / σ + β
    
    Applied after each sub-layer (Post-LN) or before (Pre-LN)

================================================================================
TRAINING DETAILS (Original Paper)
================================================================================

Base Model:
- d_model = 512
- h = 8 heads
- d_ff = 2048
- N = 6 layers
- ~65M parameters

Big Model:
- d_model = 1024
- h = 16 heads
- d_ff = 4096
- N = 6 layers
- ~213M parameters

Optimizer: Adam with custom learning rate schedule
- Warmup: 4000 steps
- lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

Regularization:
- Dropout: 0.1 (attention, FFN, embeddings)
- Label smoothing: 0.1

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Attention Mechanisms (scaled dot-product, multi-head)
- Ch03: Transformers for NLP (encoder-decoder architecture)
- Ch05: Language Modeling (decoder-only transformers)
- Ch06: Large Language Models (foundation architecture)

Related architectures:
- BERT: Encoder-only (bert.py)
- GPT: Decoder-only (gpt.py)
- Vision Transformer: For images (vision_transformer_vit.py)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need"
    
    Adds positional information to embeddings using sin/cos functions:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length. Default: 5000
        dropout: Dropout rate. Default: 0.1
    
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    
    Computes attention in parallel across h heads, then concatenates.
    
    Args:
        d_model: Total model dimension
        num_heads: Number of attention heads (must divide d_model evenly)
        dropout: Dropout rate. Default: 0.1
    
    Shape:
        - Input Q, K, V: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # ====================================================================
        # LINEAR PROJECTIONS
        # ====================================================================
        # Project input to Q, K, V for all heads simultaneously
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
        
        Args:
            Q: Queries (batch, heads, seq_len, d_k)
            K: Keys (batch, heads, seq_len, d_k)
            V: Values (batch, heads, seq_len, d_v)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention output, attention weights)
        """
        # Compute attention scores: QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale by sqrt(d_k) to prevent softmax saturation
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # ====================================================================
        # STEP 1: Linear projections and reshape for multi-head
        # ====================================================================
        # Project: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, h, d_k)
        # Transpose: → (batch, h, seq_len, d_k)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # ====================================================================
        # STEP 2: Scaled dot-product attention
        # ====================================================================
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # ====================================================================
        # STEP 3: Concatenate heads and project
        # ====================================================================
        # Transpose: (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k)
        # Reshape: → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
    
    Typically d_ff = 4 × d_model (expand then contract)
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension. Default: 2048
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear → ReLU → Dropout → Linear
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Structure:
        x → MultiHeadAttention → Add & Norm → FFN → Add & Norm → output
            ↑__________________|            ↑___|
               (residual)                  (residual)
    
    Args:
        d_model: Model dimension. Default: 512
        num_heads: Number of attention heads. Default: 8
        d_ff: Feed-forward hidden dimension. Default: 2048
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Transformer Encoder for Language Modeling
    
    This is an encoder-only transformer (similar to BERT structure but
    used for language modeling here).
    
    Args:
        vocab_size: Vocabulary size. Default: 10000
        d_model: Model dimension. Default: 512
        num_heads: Number of attention heads. Default: 8
        num_layers: Number of encoder layers. Default: 6
        d_ff: Feed-forward hidden dimension. Default: 2048
        max_len: Maximum sequence length. Default: 5000
        dropout: Dropout rate. Default: 0.1
    
    Example:
        >>> model = Transformer(vocab_size=30000, d_model=512, num_layers=6)
        >>> x = torch.randint(0, 30000, (2, 100))  # (batch, seq_len)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([2, 100, 30000])
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # ====================================================================
        # EMBEDDING LAYERS
        # ====================================================================
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Scale embeddings by sqrt(d_model) as in original paper
        self.embed_scale = math.sqrt(d_model)
        
        # ====================================================================
        # ENCODER LAYERS
        # ====================================================================
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # ====================================================================
        # OUTPUT PROJECTION
        # ====================================================================
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Optional attention mask
            
        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        # Embedding + positional encoding
        x = self.embedding(x) * self.embed_scale
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.fc(x)
        
        return logits


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
    print("Transformer Model Summary")
    print("=" * 70)
    
    # Create base transformer
    model = Transformer(
        vocab_size=30000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    )
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"Configuration:")
    print(f"  vocab_size: 30000")
    print(f"  d_model: 512")
    print(f"  num_heads: 8")
    print(f"  num_layers: 6")
    print(f"  d_ff: 2048")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Forward Pass Test")
    print("=" * 70)
    
    batch_size = 2
    seq_len = 100
    x = torch.randint(0, 30000, (batch_size, seq_len))
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print("=" * 70)
