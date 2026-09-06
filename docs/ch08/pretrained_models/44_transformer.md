# 트랜스포머

2017년 논문 "Attention Is All You Need"에서 나온 트랜스포머 구조는 수열 모형화의 주된 얼개를 순환에서 자기 주의로 바꾸었다. 모든 자리를 병렬로 처리하고 스케일 조정 내적 주의를 써서 학습 효율이 뛰어나며, 자연어 처리와 컴퓨터 비전 모두에서 주된 구조가 되었다.

## 코드

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

## 논의

다중 머리 주의 얼개가 핵심 혁신이다. 머리마다 따로 질의와 열쇠와 값을 $d_k = d_{\text{model}} / h$ 크기의 낮은 차원 부분 공간으로 사영하고 스케일 조정 내적 주의를 셈하며, 그 결과를 이어 붙여 되사영한다. 그래서 머리마다 다른 갈래의 관계를 잡아낼 수 있다. 어떤 머리는 문법 무늬에 집중하고 어떤 머리는 뜻의 비슷함에 주의한다.

자리별 순전파 신경망은 선형 변환 둘 사이에 ReLU 활성을 넣어 적용한다. 자리마다 따로 움직이며 차원을 $d_{\text{model}}$에서 (대개 $4 \times d_{\text{model}}$인) $d_{ff}$으로 넓혔다가 되사영한다. 이 부품이 주의만으로는 낼 수 없는 비선형 변환 능력을 준다.

인코더 층마다 주의와 순전파 아래 층을 모두 잔차 연결과 층 정규화로 감싼다. 잔차 연결은 깊은 신경망에서 기울기가 안정되게 흐르도록 하고, 층 정규화는 특징 차원에 걸쳐 활성을 정규화하여 학습을 안정되게 한다. 드롭아웃과 더불어 이 부품들이 층이 수십, 수백 개인 트랜스포머의 학습을 가능케 한다.

## 연습문제

**연습문제 1.**
$d_{\text{model}} = 512$, $d_{ff} = 2048$인 `EncoderLayer` 하나의 매개변수 수를 셈하라. 아래 부품별(주의 사영, 순전파 층, 층 정규화)로 나누어 보여라.

??? success "연습문제 1 풀이"
    주의는 $(512, 512)$짜리 선형 층 넷으로 $4 \times (512 \times 512 + 512) = 1{,}050{,}624$개이다. 순전파는 $(512 \times 2048 + 2048) + (2048 \times 512 + 512) = 2{,}099{,}712$개이다. 층 정규화 둘은 $2 \times (512 + 512) = 2{,}048$개이다. 층마다 모두 약 315만 개이다.

---

**연습문제 2.**
주의 점수를 왜 $\sqrt{d_k}$으로 나누는지 설명하라. 이 배수를 빼면 어떻게 되며 학습 안정성에 어떤 영향을 주겠는가?

??? success "연습문제 2 풀이"
    $Q$과 $K$의 성분의 분산이 1이면 내적 $Q K^T$의 분산은 $d_k$에 비례한다. 나누지 않으면 $d_k$이 클 때 내적이 커져 소프트맥스가 기울기가 매우 작은 (0이나 1에 가까운) 영역으로 밀려난다. $\sqrt{d_k}$으로 나누면 분산이 대략 1로 정규화되어 소프트맥스가 뜻있는 기울기를 가진 영역에 머문다. 이 나눗셈을 빼면 학습이 흔들리거나 매우 느리게 수렴할 수 있고, 특히 모형 차원이 클수록 그렇다.

---

**연습문제 3.**
가린 자기 주의와 인코더 출력에 대한 교차 주의를 갖춘 디코더 더미를 `Transformer` 클래스에 더하여 수열 대 수열 과제를 위한 온전한 인코더-디코더 트랜스포머를 만들어라.

??? success "연습문제 3 풀이"
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
    디코더는 질의가 디코더에서, 열쇠와 값이 인코더 출력에서 오는 교차 주의 아래 층을 더하여 디코더가 원문 수열의 관련 있는 부분에 주의할 수 있게 한다.
