# 크로스 어텐션

크로스 어텐션은 쿼리와 키·값이 서로 다른 입력 표현에서 나오는 어텐션이다. 디코더 상태가 인코더 출력의 여러 위치를 참조하는 경우가 대표적이다. 이 예제는 단일·멀티헤드 모듈과 디코더 블록의 형태를 확인하며 실제 과제 학습은 수행하지 않는다.

## 1. 코드

```python
"""
교차 어텐션 장치 구현
=========================================
이 모듈은 인코더-디코더 구조에서 쓰는 교차 어텐션 장치를 구현한다.
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
    교차 어텐션 레이어
    
    질의는 한 순차열(디코더 따위)에서, 열쇠와 값은 다른 순차열(인코더 따위)에서
    오는 어텐션을 계산한다.
    트랜스포머 디코더 레이어에서 인코더의 출력에 주목할 때 쓴다.
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
```

**출력:**

```
============================================================
Cross-Attention Demo
============================================================

Query (decoder) shape: torch.Size([2, 3, 64])
Key/Value (encoder) shape: torch.Size([2, 5, 64])
Output shape: torch.Size([2, 3, 64])
Attention weights shape: torch.Size([2, 3, 5])

============================================================
Multi-Head Cross-Attention Demo
============================================================

Query shape: torch.Size([2, 4, 64])
Key/Value shape: torch.Size([2, 6, 64])
Output shape: torch.Size([2, 4, 64])
Attention weights shape: torch.Size([2, 8, 4, 6])

============================================================
Encoder-Decoder Attention Block Demo
============================================================

Decoder input shape: torch.Size([2, 4, 64])
Encoder output shape: torch.Size([2, 6, 64])
Output shape: torch.Size([2, 4, 64])
Self-attention weights shape: torch.Size([2, 8, 4, 4])
Cross-attention weights shape: torch.Size([2, 8, 4, 6])
```

## 2. 논의

CrossAttention은 쿼리·키·값을 공통 차원으로 투영한 뒤 점수와 가중합을 구한다. 쿼리 길이가 Q, 원본 길이가 S이면 단일 헤드 웨이트는 $(B,Q,S)$다. 멀티헤드는 $(B,h,Q,S)$의 웨이트를 만들고 헤드별 문맥을 이어 붙여 출력 레이어에 전달한다. 같은 모델 차원에서 헤드 수를 바꿔도 네 투영레이어의 파라메터 수는 같다.

EncoderDecoderAttention은 셀프 어텐션, 크로스 어텐션과 피드포워드 부분레이어를 잔차·LayerNorm과 결합한 블록이다. 위치 표현과 토큰 임베딩, 출력 어휘레이어를 포함한 전체 디코더는 아니다. self_attn_mask의 기본값은 None이며 실행 예제도 마스크를 주지 않으므로 인과적 디코더로 동작한다고 설명할 수 없다.

각 마스크는 점수 형태에 브로드캐스트 가능해야 하고 이 사용자 정의 구현은 0을 제외 위치로 쓴다. 전부 가려진 행은 NaN이 생길 수 있다. 반환 웨이트는 드롭아웃 적용 후여서 학습 모드에서는 합이 반드시 1이 아니다. 헤드가 각각 문법·의미 역할을 분담한다고 보장되지 않으며 평균하면 헤드별 차이가 사라질 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
인코더 출력의 모양이 $(B, 10, 256)$이고 디코더 쿼리의 모양이 $(B, 1, 256)$이며 헤드가 8개인 크로스 어텐션 레이어를 생각해 보자. `MultiHeadCrossAttention` 모듈의 학습 가능한 파라메터 총수를 (바이어스를 넣어) 계산하라.

</div>

??? success "연습문제 1 풀이"
    이 모듈에는 선형 레이어가 네 개 있다.

    - `query_proj`: $256 \times 256 + 256 = 65{,}792$
    - `key_proj`: $256 \times 256 + 256 = 65{,}792$
    - `value_proj`: $256 \times 256 + 256 = 65{,}792$
    - `out_proj`: $256 \times 256 + 256 = 65{,}792$

    모두 $4 \times 65{,}792 = 263{,}168$개의 파라메터이다. 헤드의 수는 파라메터의 수를 바꾸지 않는다. 계산할 때 임베딩 차원을 어떻게 나눌지만 정할 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`EncoderDecoderAttention`의 디코더 셀프 어텐션이 `decoder_input`을 쿼리와 key_value에 모두 넘겨 같은 `MultiHeadCrossAttention` 클래스를 쓰는 까닭을 설명하라. 크로스 어텐션은 어떤 조건에서 셀프 어텐션이 되는가?

</div>

??? success "연습문제 2 풀이"
    쿼리의 출처와 키·값의 출처가 같은 시퀀스이면 크로스 어텐션이 셀프 어텐션이 된다. 코드에서 `self.self_attention(decoder_input, decoder_input, self_attn_mask)`은 디코더의 은닉 상태를 쿼리와 key_value 인자에 모두 넘긴다. 곧 쿼리와 키와 값이 모두 같은 텐서에서 저마다의 투영 행렬을 거쳐 나온다. 수식은 표준 멀티헤드 셀프 어텐션과 똑같다. 디코더의 자리마다 (마스크가 허락하는 한) 다른 모든 자리에 주목한다. 전용 셀프 어텐션 모듈과의 차이는 개념뿐이다. 이 클래스는 쿼리와 key_value가 서로 다른 곳에서 오는 일반적인 경우를 위해 설계되었지만 셀프 어텐션도 그 특수한 경우로 자연스럽게 다룬다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모든 헤드에 걸쳐 평균한 크로스 어텐션 웨이트를 돌려주고 그것으로 디코더 자리와 인코더 자리 사이의 "정렬 점수"를 계산하도록 `EncoderDecoderAttention`을 고쳐라. 예제 입력에 대해 이 정렬 행렬을 matplotlib으로 그려 보라.

</div>

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
    
    이 코드는 이미 반환된 크로스 웨이트를 헤드 축에서 평균한다. no_grad는 평가 모드를 켜지 않으므로 학습 모드의 드롭아웃이 활성화되어 있다. 무작위 웨이트가 반드시 균등하거나, 번역을 학습한 뒤 반드시 대각선 정렬이 나타나는 것은 아니다. 시각화에는 모델의 평가 모드, 마스크와 입력·출력 토큰 대응을 함께 명시해야 한다.

## 정리하며

크로스 어텐션은 서로 다른 출처의 쿼리와 키·값을 연결한다. 같은 출처를 전달하면 셀프 어텐션으로 쓸 수 있다. 마스크를 생략한 디코더 블록은 인과성을 보장하지 않으며 헤드 평균과 학습 중 드롭아웃을 구분해 해석한다.
