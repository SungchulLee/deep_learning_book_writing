"""
Self-Attention Mechanism Implementation
========================================
This module implements self-attention mechanisms used in Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Self-Attention Layer
    
    Computes attention where queries, keys, and values all come from the same input.
    Used in Transformer encoder layers.
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim) - Input sequence
            mask: (batch_size, seq_len, seq_len) - Attention mask
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.key_proj(x)    # (batch_size, seq_len, embed_dim)
        V = self.value_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention
    
    Splits the embedding dimension into multiple heads, allowing the model
    to attend to information from different representation subspaces.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        # (batch_size, seq_len, num_heads, head_dim)
        attended = attended.transpose(1, 2).contiguous()
        
        # Reshape to (batch_size, seq_len, embed_dim)
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention
    
    Prevents positions from attending to subsequent positions.
    Used in decoder-only models like GPT.
    """
    
    def __init__(self, embed_dim, num_heads, max_seq_len=512, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for causal mask
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (not a parameter)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V in one shot
        qkv = self.qkv_proj(x)
        
        # Split and reshape
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


def demonstrate_self_attention():
    """Demonstrate self-attention"""
    print("=" * 60)
    print("Self-Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 4
    embed_dim = 64
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create self-attention layer
    self_attn = SelfAttention(embed_dim)
    
    # Forward pass
    output, weights = self_attn(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (first sample):")
    print(weights[0].detach().numpy())


def demonstrate_multi_head():
    """Demonstrate multi-head self-attention"""
    print("\n" + "=" * 60)
    print("Multi-Head Self-Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 4
    embed_dim = 64
    num_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create multi-head self-attention layer
    mh_attn = MultiHeadSelfAttention(embed_dim, num_heads)
    
    # Forward pass
    output, weights = mh_attn(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {embed_dim // num_heads}")


def demonstrate_causal_attention():
    """Demonstrate causal self-attention"""
    print("\n" + "=" * 60)
    print("Causal Self-Attention Demo")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 5
    embed_dim = 64
    num_heads = 4
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create causal self-attention layer
    causal_attn = CausalSelfAttention(embed_dim, num_heads)
    
    # Forward pass
    output, weights = causal_attn(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nCausal attention mask (first head):")
    print("Each position can only attend to current and previous positions:")
    print(weights[0, 0].detach().numpy())


if __name__ == "__main__":
    demonstrate_self_attention()
    demonstrate_multi_head()
    demonstrate_causal_attention()
