"""
Self-Attention Demo
"""
import torch
from self_attention import SelfAttention

def demo():
    # Create dummy sequence
    batch_size = 2
    seq_len = 10
    embed_size = 64
    
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # Create self-attention module
    attention = SelfAttention(embed_size, heads=1)
    
    # Forward pass
    output, weights = attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print("\n✓ Self-attention demo complete!")

if __name__ == '__main__':
    demo()
