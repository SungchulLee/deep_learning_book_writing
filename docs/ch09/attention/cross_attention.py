"""cross_attention — cross_attention 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/attention/cross_attention.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
교차 어텐션 장치 구현
=========================================
이 모듈은 부호기-복호기 구조에서 쓰는 교차 어텐션 장치를 구현한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========================================================================
# 메인
# ========================================================================


class CrossAttention(nn.Module):
    """
    교차 어텐션 층
    
    질의는 한 순차열(복호기 따위)에서, 열쇠와 값은 다른 순차열(부호기 따위)에서
    오는 어텐션을 계산한다.
    트랜스포머 복호기 층에서 부호기의 출력에 주목할 때 쓴다.
    """
    
    def __init__(self, query_dim, key_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        batch_size, query_len, _ = query.shape
        kv_len = key_value.size(1)
        
        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        output = self.out_proj(attended)
        
        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """
    다중 머리 교차 어텐션
    
    머리를 여럿 두어 교차 어텐션의 표현을 풍부하게 한다.
    """
    
    def __init__(self, query_dim, key_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        batch_size, query_len, _ = query.shape
        kv_len = key_value.size(1)
        
        Q = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        K = self.key_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        V = self.value_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, query_len, self.embed_dim)
        
        output = self.out_proj(attended)
        
        return output, attention_weights


class EncoderDecoderAttention(nn.Module):
    """
    완전한 부호기-복호기 어텐션 블록
    
    (복호기의) 자기 어텐션과 (부호기-복호기의) 교차 어텐션을 모두 담고 있다.
    트랜스포머 복호기의 전형적인 블록이다.
    """
    
    def __init__(self, decoder_dim, encoder_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadCrossAttention(
            query_dim=decoder_dim, key_dim=decoder_dim,
            embed_dim=decoder_dim, num_heads=num_heads, dropout=dropout
        )
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=decoder_dim, key_dim=encoder_dim,
            embed_dim=decoder_dim, num_heads=num_heads, dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(decoder_dim)
        
    def forward(self, decoder_input, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        attn_output, self_attn_weights = self.self_attention(
            decoder_input, decoder_input, self_attn_mask
        )
        decoder_input = self.norm1(decoder_input + attn_output)
        
        attn_output, cross_attn_weights = self.cross_attention(
            decoder_input, encoder_output, cross_attn_mask
        )
        decoder_input = self.norm2(decoder_input + attn_output)
        
        ffn_output = self.ffn(decoder_input)
        output = self.norm3(decoder_input + ffn_output)
        
        return output, self_attn_weights, cross_attn_weights


def demonstrate_cross_attention():
    """기본 교차 어텐션 시연"""
    print("=" * 60)
    print("Cross-Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    query_len = 3
    kv_len = 5
    query_dim = 64
    key_dim = 64
    embed_dim = 64
    
    query = torch.randn(batch_size, query_len, query_dim)
    key_value = torch.randn(batch_size, kv_len, key_dim)
    
    cross_attn = CrossAttention(query_dim, key_dim, embed_dim)
    output, weights = cross_attn(query, key_value)
    
    print(f"\nQuery (decoder) shape: {query.shape}")
    print(f"Key/Value (encoder) shape: {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")


def demonstrate_multi_head_cross_attention():
    """다중 머리 교차 어텐션 시연"""
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
    
    query = torch.randn(batch_size, query_len, query_dim)
    key_value = torch.randn(batch_size, kv_len, key_dim)
    
    mh_cross_attn = MultiHeadCrossAttention(query_dim, key_dim, embed_dim, num_heads)
    output, weights = mh_cross_attn(query, key_value)
    
    print(f"\nQuery shape: {query.shape}")
    print(f"Key/Value shape: {key_value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")


def demonstrate_encoder_decoder():
    """완전한 부호기-복호기 어텐션 블록 시연"""
    print("\n" + "=" * 60)
    print("Encoder-Decoder Attention Block Demo")
    print("=" * 60)
    
    batch_size = 2
    decoder_len = 4
    encoder_len = 6
    decoder_dim = 64
    encoder_dim = 64
    num_heads = 8
    
    decoder_input = torch.randn(batch_size, decoder_len, decoder_dim)
    encoder_output = torch.randn(batch_size, encoder_len, encoder_dim)
    
    enc_dec_block = EncoderDecoderAttention(decoder_dim, encoder_dim, num_heads)
    output, self_weights, cross_weights = enc_dec_block(decoder_input, encoder_output)
    
    print(f"\nDecoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Self-attention weights shape: {self_weights.shape}")
    print(f"Cross-attention weights shape: {cross_weights.shape}")


if __name__ == "__main__":
    demonstrate_cross_attention()
    demonstrate_multi_head_cross_attention()
    demonstrate_encoder_decoder()
