# 트랜스포머

멀티헤드 어텐션과 피드포워드·잔차·정규화를 직접 연결한 인코더 예제다. Transformer라는 이름을 쓰지만 기본 클래스에는 디코더와 교차 어텐션이 없다. 위치별 어휘 로짓을 반환하며 사전 학습 웨이트나 학습 결과를 제공하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x = self.scaled_dot_product_attention(Q, K, V, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]

        for layer in self.encoder_layers:
            x = layer(x)

        return self.fc(x)


if __name__ == "__main__":
    model = Transformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 31,724,304
```

## 2. 논의

MultiHeadAttention은 $(B,L,d)$ 입력을 헤드로 나누고 각 점수를 sqrt(d_k)로 스케일한다. d가 num_heads로 나누어떨어져야 하지만 명시적 검사가 없어 reshape에서 실패할 수 있다. 헤드별로 다른 투영을 학습할 수 있으나 문법·의미 등 역할이 자동으로 고정되지는 않는다.

사용자 정의 mask는 0이 차단이며 점수 $(B,H,L_q,L_k)$와 브로드캐스트되어야 한다. 차단 값이 유한한 -1e9이므로 전부 가린 행은 0 출력 대신 균등 분포가 될 수 있다. 축이 맞지 않는 (B,L) 패딩 마스크를 그대로 전달해서도 안 된다.

EncoderLayer는 Post-LN과 ReLU 피드포워드를 사용한다. Transformer의 위치 표는 영으로 초기화된 학습 파라메터이며 길이 5000 이내만 받는다. forward에는 마스크 인수가 없어 기본 모델은 패딩과 미래 토큰을 제외하지 않는다. 어휘 로짓을 낸다는 사실만으로 자기 회귀 언어 모델이 되지는 않는다. 출력의 31,724,304개는 이 인코더 구조의 계수 수다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
d=512, d_ff=2048인 EncoderLayer의 파라메터 수를 계산하라.

</div>

??? success "풀이"
    어텐션 투영 1,050,624개, 피드포워드 2,099,712개, 두 LayerNorm 2,048개로 총 3,152,384개다. 드롭아웃·ReLU는 학습 파라메터를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
내적을 sqrt(d_k)로 나누는 분산 근거와 가정을 설명하라.

</div>

??? success "풀이"
    Q·K 성분이 서로 독립이고 평균 0·분산 1이면 내적의 분산은 $d_k$여서 스케일 뒤 1이 된다. 실제 학습된 피처는 이 가정을 정확히 따르지 않을 수 있다. 스케일은 초기 점수 포화를 줄이는 근거이며 모든 모델의 그래디언트·수렴을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 DecoderLayer 정의만으로 전체 Seq2Seq 모델이 완성되는가? 남은 연결과 마스크를 설명하라.

</div>

??? success "풀이"
    ```python
    class DecoderLayer(nn.Module):
        def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
            super().__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.cross_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
            attn_out = self.self_attn(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout(attn_out))
            cross_out = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(cross_out))
            ff_out = self.feed_forward(x)
            x = self.norm3(x + self.dropout(ff_out))
            return x
    ```
    이 조각은 디코더 레이어만 정의한다. 목표 임베딩·위치 표·디코더 스택·출력 레이어를 상위 모델에 연결하고 원문 메모리를 전달해야 한다. 0=차단인 목표 인과 마스크와 원문 패딩 마스크를 올바른 축으로 전달하며, 학습 목표 시프트도 필요하다. 각 헤드·배치에서 전부 가린 행의 처리도 정해야 한다.

## 정리하며

기본 클래스는 인코더이며, 디코더 레이어를 추가로 정의하는 것과 완성된 번역 모델을 연결하는 것은 다르다. 파라메터 수와 마스크 규약·축·위치 길이를 확인하고 보존된 출력은 모델 생성 결과로 읽는다.
