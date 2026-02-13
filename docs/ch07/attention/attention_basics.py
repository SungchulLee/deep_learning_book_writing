"""
Basic Attention Mechanism Implementation
=========================================
This module implements the fundamental attention mechanism concepts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicAttention(nn.Module):
    """
    Basic Attention Mechanism (Additive/Bahdanau Attention)
    
    Computes attention weights using a learned alignment model.
    Score(query, key) = v^T * tanh(W_q * query + W_k * key)
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim):
        """
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors
            hidden_dim: Hidden dimension for alignment model
        """
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys, values, mask=None):
        """
        Args:
            query: (batch_size, query_dim)
            keys: (batch_size, seq_len, key_dim)
            values: (batch_size, seq_len, value_dim)
            mask: (batch_size, seq_len) - optional mask for padding
            
        Returns:
            context: (batch_size, value_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = keys.shape
        
        # Project query and keys
        # query: (batch_size, 1, hidden_dim)
        query_proj = self.query_projection(query).unsqueeze(1)
        
        # keys: (batch_size, seq_len, hidden_dim)
        keys_proj = self.key_projection(keys)
        
        # Compute alignment scores
        # (batch_size, seq_len, hidden_dim)
        alignment = torch.tanh(query_proj + keys_proj)
        
        # (batch_size, seq_len)
        scores = self.score_projection(alignment).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector as weighted sum of values
        # (batch_size, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    The fundamental building block of Transformer attention.
    Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_v)
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            
        Returns:
            output: (batch_size, num_heads, seq_len_q, d_v)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


def demonstrate_basic_attention():
    """Demonstrate basic attention mechanism"""
    print("=" * 60)
    print("Basic Attention Mechanism Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 5
    query_dim = 8
    key_dim = 8
    value_dim = 8
    hidden_dim = 16
    
    # Create sample data
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    # Create attention module
    attention = BasicAttention(query_dim, key_dim, hidden_dim)
    
    # Compute attention
    context, weights = attention(query, keys, values)
    
    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")
    print(f"\nOutput shapes:")
    print(f"  Context: {context.shape}")
    print(f"  Attention weights: {weights.shape}")
    print(f"\nAttention weights (first sample):")
    print(f"  {weights[0].detach().numpy()}")
    print(f"  Sum: {weights[0].sum().item():.4f}")


def demonstrate_scaled_dot_product():
    """Demonstrate scaled dot-product attention"""
    print("\n" + "=" * 60)
    print("Scaled Dot-Product Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    num_heads = 4
    seq_len_q = 3
    seq_len_k = 5
    d_k = 16
    d_v = 16
    
    # Create sample data
    query = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    key = torch.randn(batch_size, num_heads, seq_len_k, d_k)
    value = torch.randn(batch_size, num_heads, seq_len_k, d_v)
    
    # Create attention module
    attention = ScaledDotProductAttention()
    
    # Compute attention
    output, weights = attention(query, key, value)
    
    print(f"\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")
    print(f"\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {weights.shape}")
    print(f"\nAttention weights (first sample, first head):")
    print(weights[0, 0].detach().numpy())


if __name__ == "__main__":
    demonstrate_basic_attention()
    demonstrate_scaled_dot_product()
