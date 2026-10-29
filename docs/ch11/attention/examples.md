# 어텐션 예제

토큰·위치 임베딩과 어텐션·피드포워드 레이어를 연결한 간단한 인코더·디코더 예제다. 원문과 타깃을 받아 로짓을 만드는 구조를 보여 주며 학습·생성·이미지 패치화의 전체 구현은 포함하지 않는다.

## 1. 코드

```python
"""
어텐션 장치 종합 예제
============================================
이 모듈은 여러 어텐션 장치의 실제 쓰임을 보인다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

# --- attention_basics ---
"""
기본 어텐션 장치 구현
=========================================
이 모듈은 어텐션 장치의 바탕이 되는 개념을 구현한다.
"""


# ========================================================================
# 메인
# ========================================================================


class BasicAttention(nn.Module):
    """
    기본 어텐션 장치 (덧셈형·바다나우 어텐션)
    
    학습된 정렬 모형으로 어텐션 가중치를 계산한다.
    Score(query, key) = v^T * tanh(W_q * query + W_k * key)
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim):
        """
        인수:
            query_dim: 질의 벡터의 차원
            key_dim: 열쇠 벡터의 차원
            hidden_dim: 정렬 모형의 숨은 차원
        """
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1)
        
    def forward(self, query, keys, values, mask=None):
        """
        인수:
            query: (배치 크기, query_dim)
            keys: (배치 크기, seq_len, key_dim)
            values: (배치 크기, seq_len, value_dim)
            mask: (배치 크기, seq_len) — 덧댐을 가리는 선택적 가림막
            
        반환값:
            context: (배치 크기, value_dim)
            attention_weights: (배치 크기, seq_len)
        """
        batch_size, seq_len, _ = keys.shape
        
        # 질의와 열쇠 사영
        # query: (배치 크기, 1, hidden_dim)
        query_proj = self.query_projection(query).unsqueeze(1)
        
        # keys: (배치 크기, seq_len, hidden_dim)
        keys_proj = self.key_projection(keys)
        
        # 정렬 점수 계산
        # (배치 크기, seq_len, hidden_dim)
        alignment = torch.tanh(query_proj + keys_proj)
        
        # (배치 크기, seq_len)
        scores = self.score_projection(alignment).squeeze(-1)
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 어텐션 가중치 계산
        attention_weights = F.softmax(scores, dim=-1)
        
        # 값의 가중합으로 문맥 벡터 계산
        # (배치 크기, value_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    배율 조정 내적 어텐션
    
    트랜스포머 어텐션의 근본 구성 블록이다.
    Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        인수:
            query: (배치 크기, num_heads, seq_len_q, d_k)
            key: (배치 크기, num_heads, seq_len_k, d_k)
            value: (배치 크기, num_heads, seq_len_v, d_v)
            mask: (배치 크기, 1, seq_len_q, seq_len_k)
            
        반환값:
            output: (배치 크기, num_heads, seq_len_q, d_v)
            attention_weights: (배치 크기, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # 어텐션 점수 계산: Q * K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스로 어텐션 가중치 얻기
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 가중치 적용
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


def demonstrate_basic_attention():
    """기본 어텐션 장치 시연"""
    print("=" * 60)
    print("Basic Attention Mechanism Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 5
    query_dim = 8
    key_dim = 8
    value_dim = 8
    hidden_dim = 16
    
    # 예시 데이터 만들기
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, seq_len, key_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    # 어텐션 모듈 만들기
    attention = BasicAttention(query_dim, key_dim, hidden_dim)
    
    # 어텐션 계산
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
    """배율 조정 내적 어텐션 시연"""
    print("\n" + "=" * 60)
    print("Scaled Dot-Product Attention Demo")
    print("=" * 60)
    
    batch_size = 2
    num_heads = 4
    seq_len_q = 3
    seq_len_k = 5
    d_k = 16
    d_v = 16
    
    # 예시 데이터 만들기
    query = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    key = torch.randn(batch_size, num_heads, seq_len_k, d_k)
    value = torch.randn(batch_size, num_heads, seq_len_k, d_v)
    
    # 어텐션 모듈 만들기
    attention = ScaledDotProductAttention()
    
    # 어텐션 계산
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


# --- cross_attention ---
"""
교차 어텐션 장치 구현
=========================================
이 모듈은 인코더-디코더 구조에서 쓰는 교차 어텐션 장치를 구현한다.
"""


# ========================================================================
# 메인
# ========================================================================


class CrossAttention(nn.Module):
    """
    교차 어텐션 층
    
    질의는 한 순차열(디코더 따위)에서, 열쇠와 값은 다른 순차열(인코더 따위)에서
    오는 어텐션을 계산한다.
    트랜스포머 디코더 층에서 인코더의 출력에 주목할 때 쓴다.
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
    완전한 인코더-디코더 어텐션 블록
    
    (디코더의) 자기 어텐션과 (인코더-디코더의) 교차 어텐션을 모두 담고 있다.
    트랜스포머 디코더의 전형적인 블록이다.
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
    """완전한 인코더-디코더 어텐션 블록 시연"""
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


# --- self_attention ---
class SelfAttention(nn.Module):
    """
    자기 어텐션 층
    
    질의와 열쇠와 값이 모두 같은 입력에서 나오는 어텐션을 계산한다.
    트랜스포머 인코더 층에서 쓴다.
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_k: Optional[int] = None, 
        d_v: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.scale = self.d_k ** -0.5
        
        # Q, K, V의 선형 사영
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_v)
        
        # 출력 사영
        self.out_proj = nn.Linear(self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, d_model)
            mask: 선택적인 어텐션 가림막 (배치 크기, seq_len, seq_len)
                  0은 가릴 자리를 뜻한다
            
        반환값:
            output: 자기 어텐션을 거친 출력 (배치 크기, seq_len, d_model)
            attention_weights: 어텐션 행렬 (배치 크기, seq_len, seq_len)
        """
        # Q, K, V로 사영 (모두 같은 입력 x에서 나온다)
        Q = self.W_q(x)  # (배치, seq_len, d_k)
        K = self.W_k(x)  # (배치, seq_len, d_k)
        V = self.W_v(x)  # (배치, seq_len, d_v)
        
        # 배율 조정 내적 어텐션 점수 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 가림막이 있으면 씌우기
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 열쇠(마지막 차원)에 대해 소프트맥스
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값의 가중합
        attended = torch.matmul(attention_weights, V)
        
        # 출력 사영
        output = self.out_proj(attended)
        
        return output, attention_weights

def demonstrate_self_attention():
    """기본 자기 어텐션 시연."""
    d_model = 512
    seq_len = 10
    batch_size = 2
    
    self_attn = SelfAttention(d_model)
    X = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = self_attn(X)
    
    print(f"Input shape:     {X.shape}")        # (2, 10, 512)
    print(f"Output shape:    {output.shape}")   # (2, 10, 512)
    print(f"Attention shape: {weights.shape}")  # (2, 10, 10)
    print(f"\nAttention matrix is square: {weights.shape[-2]} x {weights.shape[-1]}")
    print(f"Each row sums to 1: {weights[0, 0].sum().item():.4f}")

class CausalSelfAttention(nn.Module):
    """
    다중 머리를 지원하는 인과(가림막) 자기 어텐션
    
    자리마다 뒤의 자리에 주목하지 못하게 막는다.
    자기회귀 생성을 하는 GPT 같은 디코더 전용 모델에서 쓴다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 효율을 위해 QKV 사영을 합침 (행렬 곱 세 번 대신 한 번)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 인과 가림막을 버퍼로 등록 (매개변수는 아니지만 모델과 함께 움직인다)
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인수:
            x: 입력 순차열 (배치 크기, seq_len, embed_dim)
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
        반환값:
            output: 어텐션을 거친 출력 (배치 크기, seq_len, embed_dim)
            attention_weights: 선택적으로 (배치 크기, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 효율적인 연산 한 번으로 Q, K, V 사영
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, 배치, heads, seq, head_dim)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 어텐션 점수 계산: (배치, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 인과 가림막 씌우기 (자리마다 과거와 현재에만 주목할 수 있다)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 소프트맥스와 선택적인 드롭아웃
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 값에 어텐션 적용
        attended = torch.matmul(attention_weights, V)
        
        # 머리를 이어 붙이고 되사영
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        return output, None

def demonstrate_causal_attention():
    """인과 가림막의 무늬를 보인다."""
    batch_size, seq_len, embed_dim, num_heads = 1, 5, 64, 4
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    causal_attn = CausalSelfAttention(embed_dim, num_heads)
    
    output, weights = causal_attn(x)
    
    print("Causal Attention Pattern (first head):")
    print("Each row shows what that position attends to.")
    print("Position i can only attend to positions <= i (lower triangular).\n")
    print(weights[0, 0].detach().numpy().round(3))
    print("\nNote: Upper triangle is 0 (future positions masked)")

if __name__ == "__main__":
    demonstrate_causal_attention()

class TransformerEncoderBlock(nn.Module):
    """
    양방향 자기 어텐션이 있는 트랜스포머 인코더 블록 하나.
    
    구조: 자기 어텐션 → 더하기와 정규화 → 순방향 신경망 → 더하기와 정규화
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 다중 머리 자기 어텐션 (양방향)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 자리별 순방향 신경망
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인수:
            x: 입력 (배치, seq_len, embed_dim)
            src_mask: 어텐션 가림막 (seq_len, seq_len)
            src_key_padding_mask: 덧댐 가림막 (배치, seq_len)
        """
        # 잔차 연결이 있는 자기 어텐션 (사전 층 정규화 판본)
        attn_out, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # 잔차 연결이 있는 순방향 신경망
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class VisionSelfAttention(nn.Module):
    """
    이미지 조각을 위한 자기 어텐션 (비전 트랜스포머 방식).
    
    이미지를 조각으로 나누어 펼친 뒤 순차열로 다룬다.
    조각마다 다른 모든 조각에 전역적으로 주목할 수 있다.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인수:
            patches: 펼친 이미지 조각 (배치, num_patches, embed_dim)
                     대체로 첫 자리에 [CLS] 토큰이 있다
        
        반환값:
            output: 자기 어텐션을 거친 조각 (배치, num_patches, embed_dim)
            weights: 어텐션 가중치 (배치, num_patches, num_patches)
        """
        # 조각마다 (CLS 토큰을 포함한) 다른 모든 조각에 주목한다
        attended, weights = self.attention(patches, patches, patches)
        
        # 잔차 연결
        output = self.norm(patches + attended)
        
        return output, weights


class MultiHeadSelfAttention(nn.Module):
    """머리 여럿을 둔 자기 어텐션.

    위의 SelfAttention은 머리가 하나다. 머리를 여럿 두면 같은 자리에서
    서로 다른 관계를 따로 볼 수 있다. d_model을 num_heads로 나누어
    머리마다 좁은 공간을 주므로, 머리를 늘려도 매개변수는 늘지 않는다.

    입력과 출력 모양은 모두 (batch, seq_len, embed_dim)이다.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim})이 num_heads({num_heads})로 나누어떨어져야 한다")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 머리 전체를 한 번에 사영한 뒤 쪼갠다
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, E) -> (B, H, L, head_dim)"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        q = self._split(self.W_q(x))
        k = self._split(self.W_k(x))
        v = self._split(self.W_v(x))

        scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, L, L)
        if mask is not None:
            # (B, L) 짜리 채움 가리개든 (L, L) 짜리 인과 가리개든 받는다
            if mask.dim() == 2 and mask.shape[0] == B:
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = attn @ v                                     # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(out), attn

# ========================================================================
# 메인
# ========================================================================


class SimpleEncoder(nn.Module):
    """자기 어텐션을 쓰는 간단한 트랜스포머 인코더"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    """트랜스포머 인코더 레이어 하나"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x, attn_weights


class SimpleDecoder(nn.Module):
    """자기 어텐션과 교차 어텐션을 함께 쓰는 간단한 트랜스포머 디코더"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x, _, _ = layer(x, encoder_output, tgt_mask, memory_mask)
        logits = self.output_proj(x)
        return logits


class DecoderLayer(nn.Module):
    """트랜스포머 디코더 레이어 하나"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=embed_dim, key_dim=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        attn_output, self_attn_weights = self.self_attn(x, tgt_mask)
        x = self.norm1(x + attn_output)
        attn_output, cross_attn_weights = self.cross_attn(x, encoder_output, memory_mask)
        x = self.norm2(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x, self_attn_weights, cross_attn_weights


class SimpleSeq2Seq(nn.Module):
    """어텐션을 갖춘 완전한 순차열 대 순차열 모델"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256,
                 num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = SimpleEncoder(src_vocab_size, embed_dim, num_heads, num_layers, dropout=dropout)
        self.decoder = SimpleDecoder(tgt_vocab_size, embed_dim, num_heads, num_layers, dropout=dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        logits = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return logits


def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    # Seq2Seq 시연
    model = SimpleSeq2Seq(1000, 1000, 128, 8, 3)
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 8))
    logits = model(src, tgt, create_padding_mask(src), create_causal_mask(8))
    print(f"Output logits shape: {logits.shape}")
```

## 2. 논의

SimpleEncoder는 토큰 임베딩에 학습 가능한 위치 파라메터를 더하고 레이어를 반복한다. SimpleDecoder는 인과 셀프 어텐션, 크로스 어텐션과 FFN을 거쳐 어휘 로짓을 반환한다. SimpleSeq2Seq는 마스크를 전달할 뿐 타깃을 한 칸 이동시키거나 인과 마스크를 자동 생성하지 않는다.

코드는 MultiHeadSelfAttention을 self_attention에서 가져오지만 그 모듈에는 해당 이름의 클래스가 정의되어 있지 않다. 이 장의 해당 클래스는 multi_head_attention 페이지에 정의되어 있다. 따라서 본문은 현재 가져오기 경로를 그대로 둔 상태에서 완성된 실행 예제로 해석할 수 없다.

마스크는 사용자 정의 어텐션의 0 제외 규칙에 맞춰야 한다. 원문 PAD 마스크를 인코더와 크로스 어텐션에 공유하고, 타깃에는 인과·PAD 조건을 함께 반영한다. 실행 예제의 causal mask는 CPU에서 생성되므로 GPU 입력에서는 장치를 맞춰야 한다. 학습 위치 파라메터 길이 512를 넘는 입력도 별도 처리가 필요하다.

로짓 출력은 다음 토큰 로스와 연결할 수 있지만 학습 루프와 자유 생성은 구현되어 있지 않다. 사인·코사인 위치 예제도 학습하지 않은 별도 모델의 첫 레이어 출력을 계산할 뿐 어텐션 무늬를 시각화하거나 정량 비교하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`src_vocab_size=10000`, `tgt_vocab_size=10000`, `embed_dim=256`, `num_heads=8`, `num_layers=6`인 `SimpleSeq2Seq` 모델의 파라메터 총수를 계산하라. 인코더, 디코더, 출력 사영으로 나누어 세어라.

</div>

??? success "풀이"
    인코더 레이어마다(embed_dim=256) 다음과 같다.

    - 셀프 어텐션: $4 \times (256 \times 256 + 256) = 263{,}168$
    - 순방향 뉴럴 넷트웍: $(256 \times 1024 + 1024) + (1024 \times 256 + 256) = 263{,}168 + 262{,}400 = 525{,}568$
    - 레이어 정규화 (2개): $2 \times 2 \times 256 = 1{,}024$
    - 레이어당 약 $789{,}760$개, 6레이어이면 약 $4{,}738{,}560$개

    인코더 임베딩: $10{,}000 \times 256 = 2{,}560{,}000$, 위치 부호화: $512 \times 256 = 131{,}072$

    디코더 레이어마다 교차 어텐션($263{,}168$)과 레이어 정규화 하나($512$)가 더해져 레이어당 약 $1{,}053{,}440$개가 되고, 6레이어이면 약 $6{,}320{,}640$개이다.

    디코더 임베딩: $10{,}000 \times 256 = 2{,}560{,}000$, 위치 부호화: $131{,}072$, 출력 사영: $256 \times 10{,}000 + 10{,}000 = 2{,}570{,}000$.

    모두 합하면 파라메터가 정확히 $19{,}011{,}344$개이다. 정확한 수는 `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
다음 토큰 학습에서 타깃 이동과 인과 마스크가 각각 필요한 이유를 설명하라.

</div>

??? success "풀이"
    디코더 입력은 이전 토큰, 정답은 다음 토큰으로 정렬한다. 인과 마스크는 각 위치가 이후 입력 토큰을 통해 정답을 미리 보지 못하게 한다. 패딩 제외도 별도로 필요하다. 마스크만 있다고 실제 생성 품질이 보장되지는 않으며 자유 생성 조건의 검증이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`SimpleEncoder`이 학습 가능한 위치 파라메터 대신 사인·코사인 위치 부호화를 쓰도록 고쳐라. 표준 식 $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$과 $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$을 구현하라. 길이 20의 무작위 입력 시퀀스에서 두 방법의 어텐션 웨이트 무늬를 비교하여라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    def sinusoidal_encoding(max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)

    class SinusoidalEncoder(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                     max_seq_len=512, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.register_buffer('pos_encoding', sinusoidal_encoding(max_seq_len, embed_dim))
            self.layers = nn.ModuleList([
                EncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, mask=None):
            seq_len = x.size(1)
            x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
            x = self.dropout(x)
            all_weights = []
            for layer in self.layers:
                x, w = layer(x, mask)
                all_weights.append(w)
            return x, all_weights

    # 비교
    x = torch.randint(1, 1000, (1, 20))
    enc_learn = SimpleEncoder(1000, 128, 8, 2)
    enc_sin = SinusoidalEncoder(1000, 128, 8, 2)
    
    with torch.no_grad():
        out_learn, _ = enc_learn.layers[0](enc_learn.embedding(x) + enc_learn.pos_encoding[:, :20, :])
        out_sin, _ = enc_sin.layers[0](enc_sin.embedding(x) + enc_sin.pos_encoding[:, :20, :])
    ```
    
    사인·코사인 위치 값은 학습 파라메터가 아니지만, 현재 버퍼도 max_seq_len까지로 고정된다. 더 긴 위치를 계산할 수 있다는 사실이 그 길이에 대한 좋은 일반화를 보장하지는 않는다. 함수의 짝수·홀수 슬라이스도 현재 구현에서는 짝수 embed_dim을 가정한다. 두 모델은 초기 웨이트가 다르고 드롭아웃이 켜져 있으며 반환 어텐션 웨이트를 저장하지 않으므로, 이 조각만으로 두 위치 방식의 학습 후 무늬를 비교할 수 없다.

## 정리하며

단순 트랜스포머의 구조를 연결할 때는 모듈 가져오기, 위치 길이와 마스크, 타깃 정렬을 모두 맞춰야 한다. 이 페이지는 구성 예시이며 실제 학습·생성이나 위치 방식의 성능 비교를 완료한 결과는 아니다.
