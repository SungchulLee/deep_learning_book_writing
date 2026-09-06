# 어텐션 예제

이 모듈은 여러 어텐션 장치를 완전한 구조로 엮어 실제 쓰임을 보인다. 간단한 트랜스포머 부호기와 복호기, 온전한 순차열 대 순차열 모델을 세우고, 어텐션이 시각(이미지 조각)과 자기회귀 텍스트 생성에 어떻게 쓰이는지도 보인다. 이 예제들은 어텐션이라는 구성 블록이 어떻게 강력한 처음부터 끝까지의 시스템을 이루는지 드러낸다.

## 코드

```python
"""
어텐션 장치 종합 예제
============================================
이 모듈은 여러 어텐션 장치의 실제 쓰임을 보인다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_basics import BasicAttention, ScaledDotProductAttention
from self_attention import MultiHeadSelfAttention, CausalSelfAttention
from cross_attention import MultiHeadCrossAttention, EncoderDecoderAttention

# ========================================================================
# 메인
# ========================================================================


class SimpleEncoder(nn.Module):
    """자기 어텐션을 쓰는 간단한 트랜스포머 부호기"""
    
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
    """트랜스포머 부호기 층 하나"""
    
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
    """자기 어텐션과 교차 어텐션을 함께 쓰는 간단한 트랜스포머 복호기"""
    
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
    """트랜스포머 복호기 층 하나"""
    
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

## 논의

이 모듈은 어텐션 장치가 어떻게 완전한 트랜스포머 구조를 이루는지 보인다. 부호기는 층을 쌓아 만드는데, 층마다 다중 머리 자기 어텐션 뒤에 자리별 순방향 신경망이 온다. 자기 어텐션은 원본 순차열의 모든 자리가 서로 주목하게 하여, 먼 거리의 의존을 담은 문맥 반영 표현을 세운다. 어텐션 자체는 순열에 동변이므로 (여기서는 학습 가능한 매개변수인) 위치 부호화가 위치 정보를 넣어 준다.

복호기는 매우 중요한 둘째 어텐션 부분층, 곧 부호기 출력에 대한 교차 어텐션을 더한다. 복호기가 원본 순차열에 조건을 걸어 예측하는 장치가 바로 이것이다. 복호기의 자기 어텐션에 있는 인과 가림막은 자리 $i$이 자리 $i$ 이하에만 주목하게 하여 생성에 필요한 자기회귀 성질을 지킨다. `SimpleSeq2Seq` 클래스는 부호기와 복호기를 이어 원본 토큰에서 표적 로짓까지의 온전한 순전파를 보인다.

이 예제들은 어텐션이 텍스트를 넘어 두루 쓰인다는 점도 보여 준다. 시각 과제에서는 이미지를 조각으로 나누어 순차열로 다룰 수 있는데, 이것이 바로 비전 트랜스포머(ViT)의 방법이다. 조각 임베딩마다 자기 어텐션으로 다른 모든 조각에 주목하므로 모델이 이미지 전체에 걸친 관계를 붙잡는다. 마찬가지로 인과 자기 어텐션은 토큰마다 앞서 만든 토큰에만 조건을 걸게 하여 자기회귀 텍스트 생성을 가능하게 한다. 이렇게 서로 다른 응용이 모두 같은 근본적인 어텐션 요소에 기댄다.

## 연습문제

**연습문제 1.**
`src_vocab_size=10000`, `tgt_vocab_size=10000`, `embed_dim=256`, `num_heads=8`, `num_layers=6`인 `SimpleSeq2Seq` 모델의 매개변수 총수를 계산하라. 부호기, 복호기, 출력 사영으로 나누어 세어라.

??? success "연습문제 1 풀이"
    부호기 층마다(embed_dim=256) 다음과 같다.

    - 자기 어텐션: $4 \times (256 \times 256 + 256) = 263{,}168$
    - 순방향 신경망: $(256 \times 1024 + 1024) + (1024 \times 256 + 256) = 263{,}168 + 262{,}400 = 525{,}568$
    - 층 정규화 (2개): $2 \times 2 \times 256 = 1{,}024$
    - 층당 약 $789{,}760$개, 6층이면 약 $4{,}738{,}560$개

    부호기 임베딩: $10{,}000 \times 256 = 2{,}560{,}000$, 위치 부호화: $512 \times 256 = 131{,}072$

    복호기 층마다 교차 어텐션($263{,}168$)과 층 정규화 하나($512$)가 더해져 층당 약 $1{,}053{,}440$개가 되고, 6층이면 약 $6{,}320{,}640$개이다.

    복호기 임베딩: $10{,}000 \times 256 = 2{,}560{,}000$, 위치 부호화: $131{,}072$, 출력 사영: $256 \times 10{,}000 + 10{,}000 = 2{,}570{,}000$.

    모두 합하면 매개변수가 약 $18{,}911{,}344$개이다. 정확한 수는 `sum(p.numel() for p in model.parameters())`으로 확인할 수 있다.

---

**연습문제 2.**
복호기의 인과 가림막이 왜 필요한지 설명하라. 그것을 없애면 학습 중에 어떤 일이 일어나는가? 추론할 때도 모델이 올바른 출력을 낼 수 있는가?

??? success "연습문제 2 풀이"
    인과 가림막은 복호기의 자리 $i$이 자리 $j > i$에 주목하지 못하게 한다. 그것이 없으면 학습 중에 복호기가 지금 토큰을 예측하면서 미래의 표적 토큰을 "볼" 수 있는데, 이는 일종의 데이터 누출이다. 모델은 문맥에서 토큰을 만들어 내는 법을 배우는 대신 미래 토큰을 그냥 베끼는 법을 배우게 된다. 추론할 때는 미래 토큰이 아직 만들어지지 않아 쓸 수 없으므로, 인과 가림막 없이 학습한 모델은 학습과 시험의 조건이 크게 어긋나 형편없는 출력을 낸다. 인과 가림막은 학습 조건이 추론 조건과 맞게 해 준다. 자리마다 모델이 과거와 현재의 정보만 쓸 수 있으므로 참된 생성 능력을 배우게 된다.

---

**연습문제 3.**
`SimpleEncoder`이 학습 가능한 위치 매개변수 대신 사인·코사인 위치 부호화를 쓰도록 고쳐라. 표준 식 $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$과 $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$을 구현하라. 길이 20의 무작위 입력 순차열에서 두 방법의 어텐션 가중치 무늬를 견주어라.

??? success "연습문제 3 풀이"
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
    
    사인·코사인 부호화는 배우는 것이 아니라 정해져 있으므로 학습 때 본 것보다 긴 순차열에도 잘 통한다는 이점이 있다. 학습되는 부호화는 길이가 고정된 과제에서 표현력이 더 클 수 있지만 바깥으로 늘려 쓰기는 어렵다.
