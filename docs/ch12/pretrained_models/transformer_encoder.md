# 트랜스포머 인코더

토큰 임베딩과 학습 가능한 위치 표현, 셀프 어텐션·피드포워드 블록을 연결한 인코더 예제다. 모델은 무작위로 초기화되며 사전 학습된 체크포인트를 읽지 않는다. 특히 아래 코드의 배치·위치 축 불일치를 주의해서 읽어야 한다.

## 1. 코드

```python
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 다중 머리 어텐션
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 순전파
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


if __name__ == "__main__":
    pass
```

## 2. 논의

**축 불일치:** TransformerEncoder는 $(B,L,E)$를 만들지만 각 블록의 nn.MultiheadAttention은 batch_first=False 기본값으로 $(L,B,E)$를 기대한다. 전치 없이 전달하므로 B를 시퀀스 길이로 해석해 서로 다른 표본 사이에 어텐션을 계산한다. 출력 모양이 입력과 같다는 사실만으로 계산이 올바른지는 알 수 없다.

원래 의도한 $(B,L)$ 패딩 마스크를 전달하면 내부에서 기대하는 $(L,B)$와 달라 오류가 날 수 있다. batch_first=True로 일관되게 구성하거나 블록 진입·복귀의 축 전치를 맞춰야 한다. 이 문제는 아래 BERTModel 등 이 인코더를 사용하는 모듈에도 영향을 준다.

의도한 구조는 각 토큰에 학습 가능한 위치 벡터를 더하고 Post-LN 블록을 반복하는 인코더다. 위치 표는 (1,max_len,E)의 Parameter로 학습하며 길이는 max_len 이하여야 한다. 마스크의 True는 패딩 키를 제외한다는 뜻이며 패딩 쿼리의 출력·로스까지 0으로 만드는 것은 아니다. 모든 키가 제외되는 행은 별도 처리해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 예제의 출력 모양만 확인하면 축 처리를 검증할 수 있는가? MultiheadAttention의 기본 규약과 패딩 마스크를 검토하라.

</div>

??? success "풀이"
    그렇지 않다. 배치 우선 (4,30,512)가 기본 시퀀스 우선 어텐션에 전달되어 축 해석이 뒤바뀐다. 제시한 (4,30) 마스크도 내부 기대 모양과 다르다. 아래 코드는 그 문제를 드러내는 검토 대상이며 완성된 검증 예제가 아니다.

    ```python
    encoder = TransformerEncoder(
        vocab_size=5000, d_model=512, num_heads=8,
        num_layers=6, d_ff=2048
    )
    x = torch.randint(0, 5000, (4, 30))
    # 채움 가림: True면 무시, False면 주의
    mask = torch.zeros(4, 30, dtype=torch.bool)
    mask[0, 20:] = True  # 첫 수열의 길이는 20
    mask[1, 25:] = True  # 둘째는 길이 25

    output = encoder(x, mask)
    print(f"Output shape: {output.shape}")  # (4, 30, 512)
    ```

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
nn.MultiheadAttention의 key_padding_mask와 attn_mask를 비교하라.

</div>

??? success "풀이"
    배치 우선 여부와 무관하게 key_padding_mask는 (B,S)이며 불리언 True는 해당 키를 제외한다. attn_mask는 (L,S) 또는 (B×H,L,S)로 쿼리·키 쌍을 제한한다. 불리언끼리는 차단 위치를 합칠 수 있고 실수 마스크는 점수에 더하는 값이다. 둘을 함께 쓸 때는 자료형 규약을 맞춘다. 키 마스크만으로 패딩 쿼리의 로스를 제외하지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 사인·코사인 변형이 해결하는 문제와 그대로 남기는 문제를 설명하라.

</div>

??? success "풀이"
    ```python
    import math

    class SinusoidalEncoder(TransformerEncoder):
        def __init__(self, vocab_size, d_model, num_heads,
                     num_layers, d_ff, max_len=5000, dropout=0.1):
            super().__init__(vocab_size, d_model, num_heads,
                             num_layers, d_ff, max_len, dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('sinusoidal_pe', pe.unsqueeze(0))
            del self.pos_encoding

        def forward(self, x, mask=None):
            seq_len = x.size(1)
            x = self.embedding(x) + self.sinusoidal_pe[:, :seq_len, :]
            x = self.dropout(x)
            for layer in self.layers:
                x = layer(x, mask)
            return x
    ```
    고정 위치 표로 바꾸어도 블록의 batch_first 축 불일치는 그대로 남는다. 이 변형은 짝수 d_model과 max_len 이내의 길이를 가정한다. 위치 수식을 더 긴 구간에 계산할 수 있다는 사실이 학습 길이 밖의 예측 품질을 보장하지는 않는다.

## 정리하며

인코더의 입출력 모양뿐 아니라 각 연산이 해석하는 배치·시퀀스 축을 확인한다. 이 구현에는 축 불일치가 있고, 위치 표현을 바꾸는 것만으로 해결되지 않는다. 패딩 마스크·최대 길이·학습 여부를 함께 구분한다.
