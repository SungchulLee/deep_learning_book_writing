# 트랜스포머 디코더

트랜스포머 디코더는 트랜스포머 구조의 자기 회귀 부분으로, 출력 수열을 한 번에 토큰 하나씩 만드는 일을 맡는다. 디코더 블록마다 (앞으로의 자리에 닿지 못하게 하는) 가린 자기 주의를 하고 이어 순전파 신경망을 적용한다. 이 모듈은 디코더 블록 하나와 디코더 더미 전체를 함께 구현한다.

## 1. 코드

```python
import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
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
        # 가린 자기 주의
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 순전파
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x, mask)

        return self.fc_out(x)


if __name__ == "__main__":
    pass
```

## 2. 논의

`TransformerDecoderBlock`은 (인코더에 대한 교차 주의 없이) 가린 자기 주의만 갖춘 간추린 디코더 층을 구현한다. `attn_mask` 매개변수는 자리마다 뒤따르는 자리에 주의하지 못하게 하는 인과 가림을 받는다. 자기 회귀 생성에 꼭 필요하다. 학습 중에는 모든 자리를 병렬로 셈하지만 가림이 자리마다 앞선 자리만 "보게" 하여 차례대로 만드는 것을 흉내 낸다.

블록 안의 순전파 신경망은 (대개 $4 \times d_{\text{model}}$인 $d_{ff}$으로) 넓히고 ReLU 활성과 드롭아웃을 거쳐 다시 $d_{\text{model}}$으로 줄이는 표준 방식을 따른다. 두 아래 층 모두 잔차 연결과 층 정규화를 뒤 정규화 방식으로 쓰는데, 잔차를 더한 뒤에 정규화를 한다.

온전한 `TransformerDecoder`는 디코더 블록 여럿을 쌓고 임베딩 층과 마지막 선형 사영으로 감싼다. 무작위로 초기화된 학습되는 위치 인코딩을 첫 디코더 블록 앞에서 토큰 임베딩에 더한다. 출력 사영은 어휘 크기로 되잇대어 수열의 자리마다 다음 토큰 예측을 위한 로짓을 낸다.

## 연습문제

**연습문제 1.**
`vocab_size=1000`, `d_model=256`, `num_heads=8`, `num_layers=4`, `d_ff=1024`으로 `TransformerDecoder`를 세워라. 토큰 번호의 배치를 넣어 출력의 꼴이 `(batch_size, seq_len, vocab_size)`과 맞는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    decoder = TransformerDecoder(
        vocab_size=1000, d_model=256, num_heads=8,
        num_layers=4, d_ff=1024
    )
    x = torch.randint(0, 1000, (4, 50))  # batch=4, seq_len=50
    output = decoder(x)
    print(f"Output shape: {output.shape}")  # (4, 50, 1000)
    ```

---

**연습문제 2.**
이 디코더 블록에는 인코더에 대한 교차 주의가 없다. 교차 주의가 언제 필요한지, 그리고 `TransformerDecoderBlock`을 어떻게 고쳐 더할지 설명하라.

??? success "연습문제 2 풀이"
    교차 주의는 디코더가 원문 수열을 조건으로 삼으려고 인코더의 출력에 주의해야 하는 인코더-디코더 모형(이를테면 기계 번역)에서 필요하다. 더하려면 자기 주의와 순전파 아래 층 사이에 둘째 `nn.MultiheadAttention` 층을 끼운다. 질의는 디코더에서 오고 열쇠와 값은 인코더의 출력에서 온다. 이 아래 층의 잔차 연결을 위해 셋째 `nn.LayerNorm`도 필요하다.
    ```python
    self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    self.norm_cross = nn.LayerNorm(d_model)
    ```

---

**연습문제 3.**
여기서 쓴 뒤 정규화 방식(`norm(x + sublayer(x))`)과 앞 정규화 방식(`x + sublayer(norm(x))`)을 견주어라. 앞 정규화를 구현하고 그것이 깊은 모형에서 학습 안정성을 자주 높이는 까닭을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    class PreNormDecoderBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            normed = self.norm1(x)
            attn_output, _ = self.self_attention(normed, normed, normed, attn_mask=mask)
            x = x + self.dropout(attn_output)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
            return x
    ```
    앞 정규화는 층 정규화를 아래 층 뒤가 아니라 앞에 둔다. 그러면 잔차 길이 정규화되지 않은 신호를 날라 기울기가 더 곧게 흐른다. (층이 12개를 넘는) 깊은 모형에서 앞 정규화는 학습 안정성을 크게 높이고 학습률 예열의 필요를 줄인다.

## 정리하며

**다룬 것** — 트랜스포머 디코더

`TransformerDecoderBlock`은 (인코더에 대한 교차 주의 없이) 가린 자기 주의만 갖춘 간추린 디코더 층을 구현한다.

핵심 클래스는 `TransformerDecoderBlock`, `TransformerDecoder`, `PreNormDecoderBlock`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
