"""
Cross-Attention Mechanism Implementation
=========================================
This module implements cross-attention mechanisms used in encoder-decoder architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention Layer
    
    Computes attention where queries come from one sequence (e.g., decoder)
    and keys/values come from another sequence (e.g., encoder).
    Used in Transformer decoder layers to attend to encoder outputs.
    """
    
    def __init__(self, query_dim, key_dim, embed_dim, dropout=0.1):
        """
        Args:
            query_dim: Dimension of query vectors (decoder hidden state)
            key_dim: Dimension of key/value vectors (encoder hidden state)
            embed_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Separate projections for queries (from decoder) and keys/values (from encoder)
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: (batch_size, query_len, query_dim) - Decoder sequence
            key_value: (batch_size, kv_len, key_dim) - Encoder sequence
            mask: (batch_size, query_len, kv_len) - Attention mask
            
        Returns:
            output: (batch_size, query_len, embed_dim)
            attention_weights: (batch_size, query_len, kv_len)
        """
        batch_size, query_len, _ = query.shape
        kv_len = key_value.size(1)
        
        # Project queries (from decoder)
        Q = self.query_proj(query)  # (batch_size, query_len, embed_dim)
        
        # Project keys and values (from encoder)
        K = self.key_proj(key_value)   # (batch_size, kv_len, embed_dim)
        V = self.value_proj(key_value) # (batch_size, kv_len, embed_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        # Apply mask if provided (e.g., for padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention
    
    Extends cross-attention with multiple heads for richer representations.
    """
    
    def __init__(self, query_dim, key_dim, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key/value vectors
            embed_dim: Total embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: (batch_size, query_len, query_dim)
            key_value: (batch_size, kv_len, key_dim)
            mask: (batch_size, 1, query_len, kv_len) or (batch_size, 1, 1, kv_len)
            
        Returns:
            output: (batch_size, query_len, embed_dim)
            attention_weights: (batch_size, num_heads, query_len, kv_len)
        """
        batch_size, query_len, _ = query.shape
        kv_len = key_value.size(1)
        
        # Project and reshape for multi-head attention
        Q = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        K = self.key_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        V = self.value_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        
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
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, query_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class EncoderDecoderAttention(nn.Module):
    """
    Complete Encoder-Decoder Attention Block
    
    Includes both self-attention (for decoder) and cross-attention (encoder-decoder).
    This is a typical block in a Transformer decoder.
    """
    
    def __init__(self, decoder_dim, encoder_dim, num_heads, dropout=0.1):
        """
        Args:
            decoder_dim: Decoder hidden dimension
            encoder_dim: Encoder hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention for decoder
        self.self_attention = MultiHeadCrossAttention(
            query_dim=decoder_dim,
            key_dim=decoder_dim,
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention to encoder
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=decoder_dim,
            key_dim=encoder_dim,
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(decoder_dim)
        
    def forward(self, decoder_input, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Args:
            decoder_input: (batch_size, decoder_len, decoder_dim)
            encoder_output: (batch_size, encoder_len, encoder_dim)
            self_attn_mask: Mask for decoder self-attention (causal mask)
            cross_attn_mask: Mask for encoder-decoder attention (padding mask)
            
        Returns:
            output: (batch_size, decoder_len, decoder_dim)
            self_attn_weights: Self-attention weights
            cross_attn_weights: Cross-attention weights
        """
        # Self-attention on decoder
        attn_output, self_attn_weights = self.self_attention(
            decoder_input, decoder_input, self_attn_mask
        )
        decoder_input = self.norm1(decoder_input + attn_output)
        
        # Cross-attention to encoder
        attn_output, cross_attn_weights = self.cross_attention(
            decoder_input, encoder_output, cross_attn_mask
        )
        decoder_input = self.norm2(decoder_input + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(decoder_input)
        output = self.norm3(decoder_input + ffn_output)
        
        return output, self_attn_weights, cross_attn_weights


def demonstrate_cross_attention():
    """Demonstrate basic cross-attention"""
    print("=" * 60)
    print("Cross-Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    query_len = 3  # Decoder sequence length
    kv_len = 5     # Encoder sequence length
    query_dim = 64
    key_dim = 64
    embed_dim = 64
    
    # Create sample inputs
    query = torch.randn(batch_size, query_len, query_dim)  # Decoder
    key_value = torch.randn(batch_size, kv_len, key_dim)   # Encoder
    
    # Create cross-attention layer
    cross_attn = CrossAttention(query_dim, key_dim, embed_dim)
    
    # Forward pass
    output, weights = cross_attn(query, key_value)
    
    print(f"\nQuery (decoder) shape: {query.shape}")
    print(f"Key/Value (encoder) shape: {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nAttention weights (first sample):")
    print("Each decoder position attends to all encoder positions:")
    print(weights[0].detach().numpy())


def demonstrate_multi_head_cross_attention():
    """Demonstrate multi-head cross-attention"""
    print("\n" + "=" * 60)
    print("Multi-Head Cross-Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    query_len = 4
    kv_len = 6
    query_dim = 64
    key_dim = 64
    embed_dim = 64
    num_heads = 8
    
    # Create sample inputs
    query = torch.randn(batch_size, query_len, query_dim)
    key_value = torch.randn(batch_size, kv_len, key_dim)
    
    # Create multi-head cross-attention layer
    mh_cross_attn = MultiHeadCrossAttention(query_dim, key_dim, embed_dim, num_heads)
    
    # Forward pass
    output, weights = mh_cross_attn(query, key_value)
    
    print(f"\nQuery shape: {query.shape}")
    print(f"Key/Value shape: {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Number of heads: {num_heads}")


def demonstrate_encoder_decoder():
    """Demonstrate complete encoder-decoder attention block"""
    print("\n" + "=" * 60)
    print("Encoder-Decoder Attention Block Demo")
    print("=" * 60)
    
    batch_size = 2
    decoder_len = 4
    encoder_len = 6
    decoder_dim = 64
    encoder_dim = 64
    num_heads = 8
    
    # Create sample inputs
    decoder_input = torch.randn(batch_size, decoder_len, decoder_dim)
    encoder_output = torch.randn(batch_size, encoder_len, encoder_dim)
    
    # Create encoder-decoder block
    enc_dec_block = EncoderDecoderAttention(decoder_dim, encoder_dim, num_heads)
    
    # Forward pass
    output, self_weights, cross_weights = enc_dec_block(decoder_input, encoder_output)
    
    print(f"\nDecoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSelf-attention weights shape: {self_weights.shape}")
    print(f"Cross-attention weights shape: {cross_weights.shape}")


if __name__ == "__main__":
    demonstrate_cross_attention()
    demonstrate_multi_head_cross_attention()
    demonstrate_encoder_decoder()
