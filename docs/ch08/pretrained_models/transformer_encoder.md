# 트랜스포머 인코더

트랜스포머 인코더는 다중 머리 자기 주의와 순전파 신경망을 쌓은 층으로 입력 수열을 양방향으로 처리한다. 디코더와 달리 인코더에는 인과 가림이 없어 자리마다 다른 모든 자리에 주의할 수 있다. 그래서 분류, 개체명 인식, 가린 언어 모형화처럼 맥락 전체를 이해해야 하는 과제에 알맞다.

## 코드

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

## 논의

`TransformerEncoderBlock`은 `attn_mask`가 아니라 `key_padding_mask`를 쓴다. 채움 가림은 열쇠 수열의 어느 자리가 채움 토큰이어서 주의를 셈할 때 무시해야 하는지 알려 준다. 디코더의 인과 가림과 다르다. 인코더는 채움이 아닌 토큰끼리 온전한 양방향 주의를 허락하므로 맥락 전체를 쓸 수 있는 이해 과제에 알맞다.

인코더 블록마다 같은 아래 층 둘, 곧 다중 머리 자기 주의와 자리별 순전파 신경망을 적용한다. 두 아래 층 모두 잔차 연결(`x + self.dropout(sublayer(x))`) 뒤에 층 정규화를 쓴다. 아래 층마다 그 뒤의 드롭아웃이 규제를 주고 잔차 연결이 신경망을 흐르는 기울기를 안정되게 한다.

온전한 `TransformerEncoder`는 토큰 임베딩과 위치 임베딩을 합친 뒤 드롭아웃을 더한다. 이 임베딩 드롭아웃은 합쳐진 표현의 차원을 무작위로 0으로 만드는 규제로, 자리나 토큰 정보가 빠져도 모형이 버티도록 만든다. 학습되는 위치 인코딩은 모형이 학습 중에 가장 좋은 자리 표현을 찾게 해 주는데, 사인파 인코딩도 흔히 쓴다.

## 연습문제

**연습문제 1.**
`vocab_size=5000`, `d_model=512`, `num_heads=8`, `num_layers=6`, `d_ff=2048`으로 `TransformerEncoder`를 세워라. 채움이 있는 배치와 그에 맞는 채움 가림을 넣어라. 출력의 꼴이 올바른지 확인하라.

??? success "연습문제 1 풀이"
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

**연습문제 2.**
파이토치 `nn.MultiheadAttention`의 `key_padding_mask`와 `attn_mask`의 차이를 설명하라. 각각 언제 쓰며 함께 쓸 수 있는가?

??? success "연습문제 2 풀이"
    `key_padding_mask`은 꼴이 `(batch_size, seq_len)`이고 어느 열쇠 자리가 채움인지(True면 무시) 알려 준다. 배치 안의 수열 길이가 다를 때 쓴다. `attn_mask`은 꼴이 `(seq_len, seq_len)`이거나 `(batch_size * num_heads, seq_len, seq_len)`이고 주의 점수를 곧바로 고치며 대개 인과 가림에 쓴다. 함께 쓸 수 있다. 둘 다 주면 어느 한쪽에라도 가려진 자리는 주의 점수가 $-\infty$이 된다. 인코더-디코더 모형에서 인코더는 `key_padding_mask`을 쓰고 디코더는 `attn_mask`(인과)와 `key_padding_mask`(채움)을 모두 쓴다.

---

**연습문제 3.**
학습되는 위치 인코딩을 사인파 위치 인코딩으로 바꾸고 움직임을 견주어라. 사인파 판을 구현하고 각 방식이 언제 더 나은지 논하라.

??? success "연습문제 3 풀이"
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
    사인파 인코딩은 결정론적이므로 따로 학습하지 않고도 본 적 없는 수열 길이로 일반화된다. 학습되는 인코딩은 데이터의 특정 자리 무늬에 맞추어 갈 수 있지만 `max_len` 너머로는 뻗지 못한다. 추론 때의 수열 길이가 학습 때 본 것보다 길 수 있으면 사인파를 쓰라.
