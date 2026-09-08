# 어텐션의 기초

어텐션 장치는 요즘 딥러닝의 바탕이 되는 부품으로, 출력을 만들 때 모델이 입력 순차열의 쓸모 있는 부분에 그때그때 집중하게 해 준다. 이 모듈은 핵심 판본 두 가지, 곧 덧셈(바다나우) 어텐션과 트랜스포머가 쓰는 배율 조정 내적 어텐션을 구현한다. 어텐션 기반 구조를 다루려면 이 구성 블록을 이해하는 일이 꼭 필요하다.

## 1. 코드

```python
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
```

## 2. 논의

여기서 구현한 두 어텐션 장치는 순차열 대 순차열 모델이 걸어온 길의 중요한 이정표이다. **덧셈(바다나우) 어텐션**은 질의와 열쇠를 $\tanh$ 비선형이 있는 학습된 두 층 신경망에 넣어 정렬 점수를 계산한다. 열쇠 자리마다의 점수는 $\text{score}(q, k) = v^\top \tanh(W_q q + W_k k)$이며 $W_q$, $W_k$, $v$이 학습 가능한 매개변수이다. 질의와 열쇠의 차원이 같지 않아도 되어 유연하지만, 매개변수가 더 들고 자리마다의 계산이 더 비싸다.

트랜스포머 구조에서 나온 **배율 조정 내적 어텐션**은 점수를 그저 $\text{score}(Q, K) = Q K^\top / \sqrt{d_k}$으로 계산한다. 배율 인수 $1/\sqrt{d_k}$은 차원 $d_k$이 커질 때 내적의 크기가 너무 커져 소프트맥스가 기울기가 아주 작은 영역으로 밀려나는 것을 막는다. 이 장치는 병렬 처리가 잘되며, 질의와 열쇠와 값을 여러 머리로 나누어 표현의 서로 다른 부분공간에 주목하게 하는 다중 머리 어텐션의 바탕이 된다.

두 장치는 큰 틀에서 같은 흐름을 따른다. 점수를 계산하고, (덧댐이나 인과 제약을 위해) 필요하면 가림막을 씌우고, 소프트맥스로 정규화하여 어텐션 가중치를 얻고, 값 벡터의 가중합을 계산한다. 실제로 가림막은 매우 중요하다. 덧댐 가림막은 모델이 채움 토큰에 주목하지 않게 하고, 인과 가림막은 생성 중에 자기회귀 제약을 지키게 한다. 어텐션이 내놓는 문맥 벡터는 질의에 조건을 둔 입력의 요약 노릇을 하여, 모델이 아무리 긴 순차열에서도 쓸모 있는 정보를 골라 꺼낼 수 있게 한다.

## 연습문제

**연습문제 1.**
질의 벡터 $q \in \mathbb{R}^4$과 행이 $k_1 = [1, 0, 0, 0]$, $k_2 = [0, 1, 0, 0]$, $k_3 = [0, 0, 1, 0]$인 열쇠 행렬 $K \in \mathbb{R}^{3 \times 4}$, 그리고 $q = [1, 1, 0, 0]$이 주어졌을 때 (가림막 없이) 배율 조정 내적 어텐션 가중치를 계산하라. 어느 열쇠가 가장 큰 주목을 받으며 그 까닭은 무엇인가?

??? success "연습문제 1 풀이"
    날 내적은 $q \cdot k_1 = 1$, $q \cdot k_2 = 1$, $q \cdot k_3 = 0$이다. $1/\sqrt{d_k} = 1/\sqrt{4} = 0.5$을 곱하면 점수가 $[0.5, 0.5, 0]$이다. 소프트맥스를 적용하면 다음과 같다.

    $$
    \alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}} = \frac{e^{0.5}}{2e^{0.5} + e^{0}} \approx \frac{1.6487}{2(1.6487) + 1} \approx \frac{1.6487}{4.2974} \approx 0.3836
    $$

    $k_3$에 대해서는 $\alpha_3 \approx 1/4.2974 \approx 0.2328$이다. 따라서 $k_1$과 $k_2$이 각각 약 38.4%, $k_3$이 약 23.3%의 주목을 받는다. $q$이 $k_1$과 $k_2$에 해당하는 차원에서만 0이 아닌 성분을 가지므로 그 둘이 똑같이 가장 큰 주목을 받는다.

---

**연습문제 2.**
내적 어텐션에 배율 인수 $1/\sqrt{d_k}$이 필요한 까닭을 설명하라. $d_k$이 클 때 그것을 빼면 어떻게 되는가?

??? success "연습문제 2 풀이"
    $d_k$이 크면 내적 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$의 분산이 커지기 쉽다. $q_i$과 $k_i$이 평균 0, 분산 1인 서로 독립인 확률 변수라면 $\text{Var}(q \cdot k) = d_k$이다. 곧 $d_k$이 크면 내적의 크기도 아주 커질 수 있다. 이런 큰 값을 소프트맥스에 넣으면 함수가 포화하여 확률의 거의 전부를 가장 큰 점수에 몰아주고 원-핫에 가까운 어텐션 가중치를 낸다. 그 영역에서는 소프트맥스의 기울기가 아주 작아져(기울기 소실) 모델이 배우기 어렵다. $\sqrt{d_k}$으로 나누면 내적의 분산이 다시 1 언저리로 정규화되어 소프트맥스의 입력이 기울기가 잘 흐르는 범위에 머문다.

---

**연습문제 3.**
`BasicAttention` 클래스가 다중 머리 덧셈 어텐션을 지원하도록 고쳐라. 질의와 열쇠의 사영을 머리 $h$개로 나누고, 머리마다 따로 어텐션을 계산한 뒤 결과를 이어 붙여라. 머리 $h = 4$개와 숨은 차원 64로 구현을 시험하라.

??? success "연습문제 3 풀이"
    ```python
    class MultiHeadAdditiveAttention(nn.Module):
        def __init__(self, query_dim, key_dim, hidden_dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            assert hidden_dim % num_heads == 0
            
            self.query_projection = nn.Linear(query_dim, hidden_dim)
            self.key_projection = nn.Linear(key_dim, hidden_dim)
            self.score_projection = nn.Linear(self.head_dim, 1)
            self.output_projection = nn.Linear(key_dim, key_dim)
            
        def forward(self, query, keys, values, mask=None):
            batch_size, seq_len, _ = keys.shape
            
            query_proj = self.query_projection(query).unsqueeze(1)
            keys_proj = self.key_projection(keys)
            
            # 다중 머리를 위해 모양 바꾸기: (배치, seq_len, num_heads, head_dim)
            query_proj = query_proj.view(batch_size, 1, self.num_heads, self.head_dim)
            keys_proj = keys_proj.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            alignment = torch.tanh(query_proj + keys_proj)
            scores = self.score_projection(alignment).squeeze(-1)  # (배치, seq_len, num_heads)
            scores = scores.permute(0, 2, 1)  # (배치, num_heads, seq_len)
            
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
            
            weights = F.softmax(scores, dim=-1)  # (배치, num_heads, seq_len)
            
            # 머리마다 가중합
            context = torch.bmm(
                weights.view(batch_size * self.num_heads, 1, seq_len),
                values.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                      .reshape(batch_size * self.num_heads, seq_len, -1)
            ).view(batch_size, self.num_heads, -1)
            
            context = context.mean(dim=1)  # 머리에 걸쳐 평균
            context = self.output_projection(context)
            return context, weights

    # 시험
    attn = MultiHeadAdditiveAttention(64, 64, 64, num_heads=4)
    q = torch.randn(2, 64)
    k = torch.randn(2, 5, 64)
    v = torch.randn(2, 5, 64)
    ctx, w = attn(q, k, v)
    print(f"Context shape: {ctx.shape}")  # (2, 64)
    print(f"Weights shape: {w.shape}")    # (2, 4, 5)
    ```

## 정리하며

**다룬 것** — 어텐션의 기초

여기서 구현한 두 어텐션 장치는 순차열 대 순차열 모델이 걸어온 길의 중요한 이정표이다.

핵심 클래스는 `BasicAttention`, `ScaledDotProductAttention`, `MultiHeadAdditiveAttention`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
