"""
Comprehensive Attention Mechanisms Examples
============================================
This module demonstrates practical applications of various attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_basics import BasicAttention, ScaledDotProductAttention
from self_attention import MultiHeadSelfAttention, CausalSelfAttention
from cross_attention import MultiHeadCrossAttention, EncoderDecoderAttention


class SimpleEncoder(nn.Module):
    """
    Simple Transformer Encoder using Self-Attention
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - Token indices
            mask: (batch_size, 1, 1, seq_len) - Padding mask
        """
        seq_len = x.size(1)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x, _ = layer(x, mask)
        
        return x


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights


class SimpleDecoder(nn.Module):
    """
    Simple Transformer Decoder using both Self-Attention and Cross-Attention
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x: (batch_size, tgt_len) - Target token indices
            encoder_output: (batch_size, src_len, embed_dim) - Encoder output
            tgt_mask: Causal mask for decoder self-attention
            memory_mask: Padding mask for encoder-decoder attention
        """
        seq_len = x.size(1)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x, _, _ = layer(x, encoder_output, tgt_mask, memory_mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Causal self-attention
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention to encoder
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=embed_dim,
            key_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # Self-attention with residual
        attn_output, self_attn_weights = self.self_attn(x, tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention with residual
        attn_output, cross_attn_weights = self.cross_attn(x, encoder_output, memory_mask)
        x = self.norm2(x + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x, self_attn_weights, cross_attn_weights


class SimpleSeq2Seq(nn.Module):
    """
    Complete Sequence-to-Sequence Model with Attention
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, 
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.encoder = SimpleEncoder(
            src_vocab_size, embed_dim, num_heads, num_layers, dropout=dropout
        )
        
        self.decoder = SimpleDecoder(
            tgt_vocab_size, embed_dim, num_heads, num_layers, dropout=dropout
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, src_len) - Source sequence
            tgt: (batch_size, tgt_len) - Target sequence
            src_mask: Padding mask for source
            tgt_mask: Causal mask for target
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        logits = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        return logits


def create_padding_mask(seq, pad_idx=0):
    """Create padding mask for a sequence"""
    # seq: (batch_size, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """Create causal mask for decoder self-attention"""
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)


def demonstrate_seq2seq():
    """Demonstrate complete sequence-to-sequence model"""
    print("=" * 60)
    print("Sequence-to-Sequence Model Demo")
    print("=" * 60)
    
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embed_dim = 128
    num_heads = 8
    num_layers = 3
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Create model
    model = SimpleSeq2Seq(
        src_vocab_size, tgt_vocab_size, 
        embed_dim, num_heads, num_layers
    )
    
    # Create sample data
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # Create masks
    src_mask = create_padding_mask(src)
    tgt_mask = create_causal_mask(tgt_len)
    
    # Forward pass
    logits = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\nSource sequence shape: {src.shape}")
    print(f"Target sequence shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def demonstrate_vision_attention():
    """Demonstrate attention for vision (image patches)"""
    print("\n" + "=" * 60)
    print("Vision Attention Demo (Image Patches)")
    print("=" * 60)
    
    batch_size = 2
    num_patches = 196  # 14x14 patches for 224x224 image with 16x16 patches
    embed_dim = 256
    num_heads = 8
    
    # Simulate image patches (e.g., from ViT)
    image_patches = torch.randn(batch_size, num_patches, embed_dim)
    
    # Apply self-attention
    attn = MultiHeadSelfAttention(embed_dim, num_heads)
    output, weights = attn(image_patches)
    
    print(f"\nImage patches shape: {image_patches.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("\nEach patch attends to all other patches!")


def demonstrate_text_generation():
    """Demonstrate causal attention for text generation"""
    print("\n" + "=" * 60)
    print("Text Generation with Causal Attention Demo")
    print("=" * 60)
    
    vocab_size = 1000
    batch_size = 2
    seq_len = 10
    embed_dim = 128
    num_heads = 8
    
    # Create embeddings
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    # Create tokens
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    x = embedding(tokens)
    
    # Apply causal attention
    causal_attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len=512)
    output, weights = causal_attn(x)
    
    print(f"\nInput tokens shape: {tokens.shape}")
    print(f"Embedded shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("\nCausal mask ensures autoregressive generation:")
    print("Position i can only attend to positions <= i")


def visualize_attention_weights():
    """Create a simple visualization of attention patterns"""
    print("\n" + "=" * 60)
    print("Attention Pattern Visualization")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create sample attention weights
    seq_len = 8
    x = torch.randn(1, seq_len, 64)
    attn = MultiHeadSelfAttention(64, 4)
    _, weights = attn(x)
    
    # Plot first head
    plt.figure(figsize=(8, 6))
    plt.imshow(weights[0, 0].detach().numpy(), cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Pattern (Head 1)')
    plt.savefig('/home/claude/attention_pattern.png', dpi=150, bbox_inches='tight')
    print("\nAttention pattern visualization saved to 'attention_pattern.png'")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ATTENTION MECHANISMS DEMONSTRATION")
    print("=" * 80)
    
    demonstrate_seq2seq()
    demonstrate_vision_attention()
    demonstrate_text_generation()
    
    try:
        visualize_attention_weights()
    except ImportError:
        print("\nNote: matplotlib not available for visualization")
    
    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
