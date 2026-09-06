# 교차 어텐션

교차 어텐션은 순차열 대 순차열 구조에서 부호기와 복호기를 잇는 장치이다. 질의와 열쇠와 값이 모두 같은 순차열에서 나오는 자기 어텐션과 달리, 교차 어텐션은 질의를 한 순차열(대개 복호기)에서, 열쇠와 값을 다른 순차열(대개 부호기)에서 얻는다. 덕분에 복호기가 생성 걸음마다 부호화된 입력에서 정보를 골라 꺼낼 수 있어, 기계 번역이나 요약, 이미지 설명 달기 같은 과제에 없어서는 안 된다.

## 코드

```python
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
```

## 논의

부호기-복호기 구조가 통하게 만드는 것이 바로 교차 어텐션이다. 표준 트랜스포머 복호기에서는 층마다 부분층 세 개를 차례로 적용한다. 복호기 자신의 숨은 상태에 대한 가림막 자기 어텐션, 부호기 출력에 대한 교차 어텐션, 자리별 순방향 신경망이다. 자기 어텐션은 복호기가 출력 순차열 안의 의존을 다루게 하고, 교차 어텐션은 출력 토큰마다 입력의 쓸모 있는 부분에 조건을 걸게 한다.

다중 머리 판본은 어텐션 계산을 서로 독립인 머리 $h$개로 나누며, 머리마다 차원이 $d_k = d_{\text{model}} / h$인 부분공간에서 움직인다. 그러면 모델이 서로 다른 자리에서 서로 다른 표현 부분공간의 정보에 한꺼번에 주목할 수 있다. 이를테면 한 머리는 문법적 정렬을, 다른 머리는 의미적 유사성을 잡을 수 있다. 모든 머리의 출력을 이어 붙인 뒤 선형층으로 다시 모델 차원으로 사영한다.

`EncoderDecoderAttention` 블록은 이 부품들이 어떻게 완전한 복호기 층을 이루는지 보인다. 부분층마다 두른 잔차 연결과 층 정규화가 학습을 안정시키고 이런 블록을 깊이 쌓아도 기울기가 흐르게 해 준다. 실제로 트랜스포머 모델은 보통 이런 복호기 층을 6개에서 12개 쓰며, 층마다 같은 부호기 출력에 주목하되 학습된 사영은 저마다 다르다. 가림막 장치도 매우 중요하다. 자기 어텐션 가림막은 인과성을 지키고(복호기가 미래 자리에 주목하지 못하게 한다), 교차 어텐션 가림막은 원본의 덧댐을 처리한다.

## 연습문제

**연습문제 1.**
부호기 출력의 모양이 $(B, 10, 256)$이고 복호기 질의의 모양이 $(B, 1, 256)$이며 머리가 8개인 교차 어텐션 층을 생각해 보자. `MultiHeadCrossAttention` 모듈의 학습 가능한 매개변수 총수를 (편향을 넣어) 계산하라.

??? success "연습문제 1 풀이"
    이 모듈에는 선형층이 네 개 있다.

    - `query_proj`: $256 \times 256 + 256 = 65{,}792$
    - `key_proj`: $256 \times 256 + 256 = 65{,}792$
    - `value_proj`: $256 \times 256 + 256 = 65{,}792$
    - `out_proj`: $256 \times 256 + 256 = 65{,}792$

    모두 $4 \times 65{,}792 = 263{,}168$개의 매개변수이다. 머리의 수는 매개변수의 수를 바꾸지 않는다. 계산할 때 임베딩 차원을 어떻게 나눌지만 정할 뿐이다.

---

**연습문제 2.**
`EncoderDecoderAttention`의 복호기 자기 어텐션이 `decoder_input`을 질의와 key_value에 모두 넘겨 같은 `MultiHeadCrossAttention` 클래스를 쓰는 까닭을 설명하라. 교차 어텐션은 어떤 조건에서 자기 어텐션이 되는가?

??? success "연습문제 2 풀이"
    질의의 출처와 열쇠·값의 출처가 같은 순차열이면 교차 어텐션이 자기 어텐션이 된다. 코드에서 `self.self_attention(decoder_input, decoder_input, self_attn_mask)`은 복호기의 숨은 상태를 질의와 key_value 인자에 모두 넘긴다. 곧 질의와 열쇠와 값이 모두 같은 텐서에서 저마다의 사영 행렬을 거쳐 나온다. 수식은 표준 다중 머리 자기 어텐션과 똑같다. 복호기의 자리마다 (가림막이 허락하는 한) 다른 모든 자리에 주목한다. 전용 자기 어텐션 모듈과의 차이는 개념뿐이다. 이 클래스는 질의와 key_value가 서로 다른 곳에서 오는 일반적인 경우를 위해 설계되었지만 자기 어텐션도 그 특수한 경우로 자연스럽게 다룬다.

---

**연습문제 3.**
모든 머리에 걸쳐 평균한 교차 어텐션 가중치를 돌려주고 그것으로 복호기 자리와 부호기 자리 사이의 "정렬 점수"를 계산하도록 `EncoderDecoderAttention`을 고쳐라. 예제 입력에 대해 이 정렬 행렬을 matplotlib으로 그려 보라.

??? success "연습문제 3 풀이"
    ```python
    import matplotlib.pyplot as plt

    def visualize_alignment(decoder_len=6, encoder_len=10, decoder_dim=64, 
                            encoder_dim=64, num_heads=8):
        # 모델과 임시 데이터 만들기
        block = EncoderDecoderAttention(decoder_dim, encoder_dim, num_heads)
        decoder_input = torch.randn(1, decoder_len, decoder_dim)
        encoder_output = torch.randn(1, encoder_len, encoder_dim)
        
        with torch.no_grad():
            _, _, cross_weights = block(decoder_input, encoder_output)
        
        # 머리에 걸쳐 평균: (1, num_heads, decoder_len, encoder_len) -> (decoder_len, encoder_len)
        alignment = cross_weights[0].mean(dim=0).numpy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(alignment, cmap='viridis', aspect='auto')
        ax.set_xlabel('Encoder Position')
        ax.set_ylabel('Decoder Position')
        ax.set_title('Cross-Attention Alignment (Head-Averaged)')
        plt.colorbar(im, label='Attention Weight')
        plt.tight_layout()
        plt.show()

    visualize_alignment()
    ```
    
    그려 낸 열지도는 복호기 자리마다 부호기 자리에 얼마나 주목하는지 보여 준다. (학습하지 않은) 무작위 가중치에서는 무늬가 거의 고르지만, 번역 과제로 학습한 뒤에는 대각선 비슷한 무늬가 나타나며 원본 언어와 표적 언어 사이의 단조로운 정렬을 드러낸다.
