"""attention_basics — attention_basics 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/attention/attention_basics.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
기본 어텐션 장치 구현
=========================================
이 모듈은 어텐션 장치의 바탕이 되는 개념을 구현한다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
