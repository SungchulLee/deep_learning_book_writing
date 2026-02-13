#!/usr/bin/env python3
"""
================================================================================
BERT - Bidirectional Encoder Representations from Transformers
================================================================================

Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language 
        Understanding" (NAACL 2019)
Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google)
Link: https://arxiv.org/abs/1810.04805

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
BERT revolutionized NLP by demonstrating that bidirectional pre-training
produces powerful language representations for downstream tasks.

Key Achievements:
- **State-of-the-art on 11 NLP tasks** upon release
- **SQuAD v1.1**: First to surpass human performance
- **Foundation for modern NLP**: RoBERTa, ALBERT, DistilBERT, etc.

================================================================================
KEY INNOVATIONS
================================================================================

1. **Bidirectional Context** (vs GPT's unidirectional)
   ─────────────────────────────────────────────────────────────────────────────
   GPT:  "The cat sat on the [MASK]" → only uses left context
   BERT: "The cat sat on the [MASK]" → uses BOTH left AND right context
   
   This is achieved through Masked Language Modeling (MLM).

2. **Pre-training Objectives**
   ─────────────────────────────────────────────────────────────────────────────
   a) Masked Language Modeling (MLM):
      - Randomly mask 15% of tokens
      - Predict original tokens from context
      - Of masked positions: 80% [MASK], 10% random, 10% unchanged
   
   b) Next Sentence Prediction (NSP):
      - Binary classification: Is sentence B the actual next sentence?
      - Helps with tasks requiring sentence relationships
      - (Note: Later work like RoBERTa found NSP may not be necessary)

3. **Input Representation**
   ─────────────────────────────────────────────────────────────────────────────
   Input = Token Embeddings + Segment Embeddings + Position Embeddings
   
   [CLS] Sentence A [SEP] Sentence B [SEP]
     ↓      ↓        ↓       ↓        ↓
   Token:  E_[CLS]  E_A     E_[SEP]  E_B     E_[SEP]
   Segment: E_A     E_A      E_A     E_B      E_B
   Position: E_0    E_1      E_2     E_3      E_4

================================================================================
ARCHITECTURE COMPARISON
================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│ Model      │ Layers │ Hidden │ Attention Heads │ Parameters │               │
├──────────────────────────────────────────────────────────────────────────────┤
│ BERT-Base  │   12   │  768   │       12        │    110M    │               │
│ BERT-Large │   24   │  1024  │       16        │    340M    │               │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
CURRICULUM MAPPING
================================================================================

This implementation supports learning objectives in:
- Ch03: Transformers for NLP (encoder architecture)
- Ch05: Language Modeling (masked LM objective)
- Ch06: Large Language Models (pre-training paradigm)

Related: gpt.py (decoder-only), transformer.py (encoder-decoder)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BERTEmbedding(nn.Module):
    """
    BERT Embedding Layer
    
    Combines three types of embeddings:
    1. Token embeddings: WordPiece vocabulary
    2. Position embeddings: Learned (not sinusoidal)
    3. Segment embeddings: Sentence A vs B
    
    Args:
        vocab_size: Vocabulary size. Default: 30522 (BERT-Base)
        d_model: Embedding dimension. Default: 768
        max_len: Maximum sequence length. Default: 512
        n_segments: Number of segment types. Default: 2
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        max_len: int = 512,
        n_segments: int = 2,
        dropout: float = 0.1
    ):
        super(BERTEmbedding, self).__init__()
        
        # Token embeddings (WordPiece vocabulary)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Position embeddings (learned, not sinusoidal like original Transformer)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Segment embeddings (sentence A = 0, sentence B = 1)
        self.segment_embedding = nn.Embedding(n_segments, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute BERT embeddings
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs (batch, seq_len)
            position_ids: Optional position IDs
            
        Returns:
            Combined embeddings (batch, seq_len, d_model)
        """
        seq_len = input_ids.size(1)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Compute each embedding type
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        segment_emb = self.segment_embedding(segment_ids)
        
        # Combine embeddings
        embeddings = token_emb + position_emb + segment_emb
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTAttention(nn.Module):
    """
    Multi-Head Self-Attention for BERT
    
    Args:
        d_model: Model dimension. Default: 768
        n_heads: Number of attention heads. Default: 12
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super(BERTAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Self-attention forward pass
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, seq_len)
            
        Returns:
            Attended output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out(context)


class BERTLayer(nn.Module):
    """
    Single BERT Encoder Layer
    
    Structure: Self-Attention → Add & Norm → FFN → Add & Norm
    
    Args:
        d_model: Model dimension. Default: 768
        n_heads: Number of attention heads. Default: 12
        d_ff: Feed-forward dimension. Default: 3072 (4 × d_model)
        dropout: Dropout rate. Default: 0.1
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1
    ):
        super(BERTLayer, self).__init__()
        
        # Multi-head self-attention
        self.attention = BERTAttention(d_model, n_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model, eps=1e-12)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # BERT uses GELU activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BERT layer
        """
        # Self-attention with residual
        attn_output = self.attention(x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class BERT(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers
    
    This implementation includes both MLM and NSP heads for pre-training.
    
    Args:
        vocab_size: Vocabulary size. Default: 30522
        d_model: Model dimension. Default: 768
        n_layers: Number of transformer layers. Default: 12
        n_heads: Number of attention heads. Default: 12
        d_ff: Feed-forward dimension. Default: 3072
        max_len: Maximum sequence length. Default: 512
        dropout: Dropout rate. Default: 0.1
    
    Example:
        >>> model = BERT(vocab_size=30522, d_model=768, n_layers=12)
        >>> input_ids = torch.randint(0, 30522, (2, 128))
        >>> segment_ids = torch.zeros(2, 128, dtype=torch.long)
        >>> mlm_logits, nsp_logits = model(input_ids, segment_ids)
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super(BERT, self).__init__()
        
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout=dropout)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            BERTLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # ====================================================================
        # PRE-TRAINING HEADS
        # ====================================================================
        
        # Masked Language Model (MLM) head
        # Predicts original tokens at masked positions
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Linear(d_model, vocab_size)
        )
        
        # Next Sentence Prediction (NSP) head
        # Binary classification: is sentence B the actual next sentence?
        self.nsp_head = nn.Linear(d_model, 2)
        
        # Pooler for [CLS] token representation
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BERT
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            mlm_logits: Logits for MLM (batch, seq_len, vocab_size)
            nsp_logits: Logits for NSP (batch, 2)
        """
        # Get embeddings
        x = self.embedding(input_ids, segment_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # MLM: predict all tokens
        mlm_logits = self.mlm_head(x)
        
        # NSP: use [CLS] token (position 0)
        cls_output = x[:, 0]
        pooled_output = self.pooler(cls_output)
        nsp_logits = self.nsp_head(pooled_output)
        
        return mlm_logits, nsp_logits
    
    def get_sequence_output(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get sequence representations (for fine-tuning on token-level tasks)
        
        Returns:
            Hidden states for all tokens (batch, seq_len, d_model)
        """
        x = self.embedding(input_ids, segment_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x
    
    def get_pooled_output(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get pooled representation (for fine-tuning on sequence-level tasks)
        
        Returns:
            Pooled [CLS] representation (batch, d_model)
        """
        x = self.get_sequence_output(input_ids, segment_ids, attention_mask)
        cls_output = x[:, 0]
        return self.pooler(cls_output)


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
    print("BERT Model Summary")
    print("=" * 70)
    
    # Create BERT-Base
    model = BERT(
        vocab_size=30522,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072
    )
    
    total_params, _ = count_parameters(model)
    print(f"BERT-Base Configuration:")
    print(f"  vocab_size: 30522")
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
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    print(f"Input shape: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        mlm_logits, nsp_logits = model(input_ids, segment_ids)
    
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape}")
    print("=" * 70)
