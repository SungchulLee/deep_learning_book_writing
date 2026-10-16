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
from attention_basics import BasicAttention, ScaledDotProductAttention
from self_attention import MultiHeadSelfAttention, CausalSelfAttention
from cross_attention import MultiHeadCrossAttention, EncoderDecoderAttention

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

??? success "연습문제 1 풀이"
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

??? success "연습문제 2 풀이"
    디코더 입력은 이전 토큰, 정답은 다음 토큰으로 정렬한다. 인과 마스크는 각 위치가 이후 입력 토큰을 통해 정답을 미리 보지 못하게 한다. 패딩 제외도 별도로 필요하다. 마스크만 있다고 실제 생성 품질이 보장되지는 않으며 자유 생성 조건의 검증이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`SimpleEncoder`이 학습 가능한 위치 파라메터 대신 사인·코사인 위치 부호화를 쓰도록 고쳐라. 표준 식 $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$과 $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$을 구현하라. 길이 20의 무작위 입력 시퀀스에서 두 방법의 어텐션 웨이트 무늬를 비교하여라.

</div>

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
    
    사인·코사인 위치 값은 학습 파라메터가 아니지만, 현재 버퍼도 max_seq_len까지로 고정된다. 더 긴 위치를 계산할 수 있다는 사실이 그 길이에 대한 좋은 일반화를 보장하지는 않는다. 함수의 짝수·홀수 슬라이스도 현재 구현에서는 짝수 embed_dim을 가정한다. 두 모델은 초기 웨이트가 다르고 드롭아웃이 켜져 있으며 반환 어텐션 웨이트를 저장하지 않으므로, 이 조각만으로 두 위치 방식의 학습 후 무늬를 비교할 수 없다.

## 정리하며

단순 트랜스포머의 구조를 연결할 때는 모듈 가져오기, 위치 길이와 마스크, 타깃 정렬을 모두 맞춰야 한다. 이 페이지는 구성 예시이며 실제 학습·생성이나 위치 방식의 성능 비교를 완료한 결과는 아니다.
