# 트랜스포머 구조 훑어보기

"Attention Is All You Need"(Vaswani 외, 2017)에서 나온 트랜스포머 구조는 순환을 아주 없애고 자기 주의 얼개를 택하여 수열 모형화를 뒤바꾸어 놓았다. 이 구조의 큰 전환은 학습 중에 전에 없던 병렬 처리를 가능케 했고 요즘 대형 언어 모형의 바탕을 놓았다.

---

## 1. 역사적 배경과 동기

트랜스포머 이전에 수열 대 수열 과제는 주의 얼개를 곁들인 순환 구조(RNN, LSTM, GRU)에 기댔다. 잘 통하기는 했지만 이 방법들에는 근본적인 한계가 있었고 트랜스포머는 그것을 넘어서도록 설계되었다.

### 차례차례 처리해야 하는 병목

순환 신경망은 한 번에 토큰 하나씩 수열을 처리하며 숨은 상태 $h_t$마다 앞선 상태 $h_{t-1}$에 매인다.

$$
h_t = f(h_{t-1}, x_t)
$$

이 차례에 따른 매임이 매우 중요한 문제 둘을 낳는다.

1. **병렬 처리가 안 된다**: 단계마다 앞 단계가 끝나기를 기다려야 하므로 요즘의 병렬 하드웨어(GPU, TPU)에서 학습이 느리다. 쓸 수 있는 계산 자원과 무관하게 학습 시간이 수열 길이에 비례해 늘어난다.
2. **자원을 아끼지 못한다**: 시간 단계마다 제 나름의 연산 묶음이 필요하여 긴 수열에서 계산 비용이 크고 병렬 하드웨어를 잘 쓰지 못한다.

### 먼 거리 의존 문제

순환 신경망은 다음 까닭으로 먼 거리에 걸쳐 정보를 지키기 어려워한다.

1. **기울기 사라짐**: 여러 시간 단계를 거쳐 퍼진 기울기가 지수로 줄어들어 모형이 멀리 떨어진 토큰 사이의 의존을 배우지 못한다. LSTM과 GRU가 문 얼개로 이를 누그러뜨리지만 아주 긴 수열에서는 여전히 나빠진다.
2. **기억의 제약**: 크기가 고정된 숨은 상태가 관련된 내력을 모두 눌러 담아야 하여 정보 병목이 생긴다. 수열이 나아갈수록 앞쪽 토큰의 맥락이 점점 덮여 없어진다.
3. **먼 맥락을 다루기가 비효율적이다**: 중간 토큰이 다루려는 의존과 상관없을 때에도 정보가 중간 숨은 상태를 거쳐 한 걸음씩 전해져야 한다.

### 트랜스포머가 이 어려움을 어떻게 푸는가

트랜스포머 구조는 자기 주의로 이 한계를 푼다.

1. **온전한 병렬 처리**: 자기 주의는 모든 토큰을 한꺼번에 처리하며 토큰 쌍마다의 관계를 한 번의 연산으로 셈한다. 그래서 병렬 하드웨어를 알뜰히 쓸 수 있다.
2. **먼 거리에 곧바로 닿는다**: 토큰마다 거리와 상관없이 다른 모든 토큰에 곧바로 주의한다. 두 토큰 사이의 최대 경로 길이가 순환 신경망의 $O(n)$에 견주어 $O(1)$이다.
3. **차례로 깊어져 생기는 기울기 사라짐이 없다**: 트랜스포머는 토큰을 차례로 처리하지 않으므로 기울기가 여러 시간 단계에 걸쳐 쌓이지 않고 신경망을 더 자유로이 흐른다.
4. **위치 인코딩**: 병렬 처리로 잃은 차례 정보를 입력 임베딩에 더한 드러난 위치 인코딩으로 되찾는다.

### 핵심 혁신 간추리기

트랜스포머는 한데 모여 큰 전환을 이루는 여러 구조적 혁신을 들여왔다.

| 혁신 | 무엇을 대신하는가 | 이점 |
|-----------|----------|---------|
| 자기 주의 | 순환 계산 | 두 토큰 사이의 경로 길이가 $O(1)$ |
| 다중 머리 주의 | 주의 함수 하나 | 갖가지 관계를 병렬로 잡아낸다 |
| 위치 인코딩 | 본디 지닌 차례 | 병렬성을 잃지 않고 자리를 넣는다 |
| 스케일 조정 내적 | 더하기 주의 | 행렬 곱으로 GPU에서 효율적으로 셈한다 |
| 잔차 연결과 층 정규화 | 깊게 쌓기 | 여러 층을 거치는 기울기 흐름이 안정된다 |

---

## 2. 구조 개관

트랜스포머는 인코더-디코더 짜임을 따르지만 요즘 변형은 인코더만(BERT) 또는 디코더만(GPT) 쓰는 경우가 많다.

### 큰 틀의 짜임

$$
\text{Transformer}(X) = \text{Decoder}(\text{Encoder}(X), Y_{\text{shifted}})
$$

온전한 구조는 다음으로 이루어진다.

1. **입력 임베딩 층**: 토큰을 빽빽한 벡터로 바꾼다
2. **위치 인코딩**: 수열의 차례 정보를 넣는다
3. **인코더 더미**: 똑같은 인코더 층 $N$개
4. **디코더 더미**: 똑같은 디코더 층 $N$개
5. **출력 선형 층**: 어휘 크기로 사영한다
6. **소프트맥스**: 확률 분포를 낸다

### 차원의 흐름

다음을 가진 모형에서는

- 어휘 크기 $V$
- 모형 차원 $d_{\text{model}}$
- 수열 길이 $L$
- 배치 크기 $B$

차원이 다음과 같이 바뀐다.

$$
\begin{aligned}
\text{Input tokens} &: (B, L) \\
\text{After embedding} &: (B, L, d_{\text{model}}) \\
\text{After encoder} &: (B, L, d_{\text{model}}) \\
\text{After decoder} &: (B, L, d_{\text{model}}) \\
\text{After output projection} &: (B, L, V)
\end{aligned}
$$

---

## 3. 핵심 부품

### 1. 입력 임베딩

입력 임베딩 층은 띄엄띄엄한 토큰을 이어진 벡터로 잇댄다.

$$
\mathbf{E} = \text{Embedding}(\mathbf{x}) \cdot \sqrt{d_{\text{model}}}
$$

$\sqrt{d_{\text{model}}}$이라는 배수는 임베딩의 크기가 위치 인코딩과 견줄 만하게 한다.

### 2. 위치 인코딩

주의는 순서를 바꾸어도 그대로이므로 자리 정보를 드러내어 더해야 한다.

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

마지막 입력 표현은 다음과 같다.

$$
\mathbf{X}_0 = \mathbf{E} + \text{PE}
$$

### 3. 인코더 층

인코더 층마다 아래 층을 둘 담는다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### 4. 디코더 층

디코더 층마다 아래 층을 셋 담는다.

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHeadAttention}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{MultiHeadAttention}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}}, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

### 5. 순전파 신경망

자리별 순전파 신경망은 다음과 같다.

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

차원은 다음과 같다.

- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$

대개 $d_{ff} = 4 \times d_{\text{model}}$이다.

---

## 4. PyTorch 구현

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """사인파 위치 인코딩."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬을 만든다
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 꼴이 [batch_size, seq_len, d_model]인 텐서
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """트랜스포머 인코더 블록 하나."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 다중 머리 자기 주의
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 순방향 신경망
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 층 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            mask: 주의 가림 [seq_len, seq_len]
            key_padding_mask: 채움 가림 [batch_size, seq_len]
        """
        # 잔차 연결을 곁들인 자기 주의
        attn_output, _ = self.self_attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 잔차 연결이 있는 순방향 신경망
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    """수열 대 수열 과제를 위한 온전한 트랜스포머 모형."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 임베딩
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 부호기
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 디코더 (간추림. 여기서는 인코더만 보인다)
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 출력 사영
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """사비에르 균등으로 가중치를 초기화한다."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """원문 수열을 인코딩한다."""
        # 임베딩과 위치 인코딩
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # 인코더 층을 통과시킨다
        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_key_padding_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """목표 수열을 디코딩한다."""
        # 임베딩과 위치 인코딩
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # 디코더 층을 통과시킨다
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        트랜스포머를 지나는 앞먹임.
        
        인수:
            src: 원문 수열 [batch_size, src_len]
            tgt: 목표 수열 [batch_size, tgt_len]
            src_mask: 원문 주의 가림
            tgt_mask: 목표문 주의 가림(인과)
            src_key_padding_mask: 원문 채움 가림
        
        반환값:
            출력 로짓 [batch_size, tgt_len, tgt_vocab_size]
        """
        # 부호화
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # 디코딩
        output = self.decode(tgt, memory, tgt_mask)
        
        # 어휘로 사영한다
        return self.output_projection(output)
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """자기 회귀 디코딩을 위한 인과 가림을 만든다."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# 쓰는 보기
if __name__ == "__main__":
    # 모형 설정
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048
    )
    
    # 예제 입력
    batch_size, src_len, tgt_len = 32, 20, 15
    src = torch.randint(0, 10000, (batch_size, src_len))
    tgt = torch.randint(0, 10000, (batch_size, tgt_len))
    
    # 디코더를 위한 인과 가림을 만든다
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt_len)
    
    # 순전파
    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Output shape: {output.shape}")  # [32, 15, 10000]
    
    # 매개변수 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

**출력:**

```
Output shape: torch.Size([32, 15, 10000])
Total parameters: 53,198,608
```

---

## 5. 초매개변수

본디 트랜스포머 논문은 설정 둘을 썼다.

| 매개변수 | 기본 모형 | 큰 모형 |
|-----------|------------|-----------|
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $h$ (머리 수) | 8 | 16 |
| $N$ (층 수) | 6 | 6 |
| $d_k = d_v$ | 64 | 64 |
| 매개변수 | 6500만 | 2억 1300만 |

---

## 6. 계산 복잡도

### 자기 주의의 복잡도

수열 길이가 $n$이고 모형 차원이 $d$일 때 다음과 같다.

$$
\text{Time: } O(n^2 \cdot d) \qquad \text{Space: } O(n^2 + n \cdot d)
$$

수열 길이에 대한 이차 복잡도가 긴 수열에서 병목이 된다.

### 순전파의 복잡도

$$
\text{Time: } O(n \cdot d \cdot d_{ff}) \qquad \text{Space: } O(n \cdot d_{ff})
$$

### 순환 신경망과 견주기

| 측면 | 트랜스포머 | 순환 신경망 |
|--------|-------------|-----|
| 차례로 하는 연산 | $O(1)$ | $O(n)$ |
| 층마다의 복잡도 | $O(n^2 \cdot d)$ | $O(n \cdot d^2)$ |
| 최대 경로 길이 | $O(1)$ | $O(n)$ |
| 병렬성 | 높음 | 낮음 |

---

## 7. 핵심 설계 결정

### 1. 잔차 연결

잔차 연결은 깊은 신경망에서 기울기가 흐르게 한다.

$$
\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}^{(l)}))
$$

### 2. 층 정규화

층 정규화는 학습을 안정되게 한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta
$$

여기서 $\mu$과 $\sigma$은 특징 차원에 걸쳐 셈한다.

### 3. 앞 정규화와 뒤 정규화

**뒤 정규화 (본디 방식):**

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**앞 정규화 (요즘 변형에서 흔하다):**

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

앞 정규화는 아주 깊은 모형을 학습할 때 더 안정적인 편이다.

---

## 8. 변형과 확장

### 인코더만 (BERT 방식)
- 양방향 주의
- 이해 과제에 쓴다
- 가린 언어 모형화로 사전 학습한다

### 디코더만 (GPT 방식)
- 인과(한 방향) 주의
- 생성 과제에 쓴다
- 다음 토큰 맞히기로 사전 학습한다

### 인코더-디코더 (T5 방식)
- 온전한 수열 대 수열 능력
- 인코더와 디코더 사이의 교차 주의
- 글에서 글로 옮기는 하나의 틀

---

## 연습문제

**연습문제 1.**
트랜스포머 블록의 핵심 부품을 들고 각각의 몫을 설명하라.

??? success "연습문제 1 풀이"
    다중 머리 자기 주의는 모든 자리 사이의 의존을 잡아낸다. 더하고 정규화하기(잔차와 층 정규화)는 학습을 안정되게 하고 기울기가 흐르게 한다. 순전파 신경망(2층 다층 퍼셉트론)은 자리마다 비선형 변환을 더한다. 이 블록을 $N$번(대개 6~12번) 되풀이한다.

---

**연습문제 2.**
층이 $N=6$개, $d=512$, 머리가 $h=8$개, $d_{ff}=2048$인 트랜스포머 인코더의 전체 매개변수 수를 셈하라.

??? success "연습문제 2 풀이"
    층마다 다중 머리 주의가 $4d^2 \approx 105만$, 순전파가 $2 \times d \times d_{ff} = 2 \times 512 \times 2048 \approx 210만$, 층 정규화가 $4d = 2048$이다. 층마다 모두 약 315만이다. 전체는 $6 \times 315만 \approx 1890만$이다(임베딩은 뺀 값이다).

---

**연습문제 3.**
트랜스포머가 아래 층마다 그 둘레에 잔차 연결을 쓰는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    잔차 연결은 깊은 신경망에서 기울기가 흐르게 한다. $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$이다. 잔차가 없으면 기울기가 주의와 순전파 계산을 거쳐야 하여 깊은 모형에서 기울기가 사라질 위험이 있다. 또 층마다 표현을 조금씩 다듬는 일을 배우기 쉽게 해 준다.

---

**연습문제 4.**
병렬성, 먼 거리 의존, 기억의 면에서 트랜스포머와 순환 신경망을 견주어라.

??? success "연습문제 4 풀이"
    병렬성 면에서 트랜스포머는 모든 자리를 한꺼번에 처리하지만(차례로 하는 단계가 $O(1)$) 순환 신경망은 차례로 하는 단계가 $O(n)$번 든다. 먼 거리 면에서 트랜스포머는 (주의로) 어떤 두 자리도 곧바로 잇지만 순환 신경망은 중간 단계를 모두 거쳐 퍼뜨려야 한다. 기억 면에서 트랜스포머는 주의에 $O(n^2)$을 쓰고 순환 신경망은 $O(n)$을 쓴다. 병렬성과 먼 거리에서는 트랜스포머가, 아주 긴 수열의 기억에서는 순환 신경망이 낫다.

## 정리하며

트랜스포머 구조는 수열 모형화의 근본적인 전환이다.

1. **병렬 처리**: 학습 중의 차례 의존을 없앤다
2. **먼 거리 의존**: 어떤 자리끼리도 곧바로 이어진다
3. **키울 수 있음**: 매개변수가 십억 개인 모형의 바탕이다
4. **쓰임새가 넓음**: 글, 그림, 소리, 여러 양식의 데이터에 두루 쓴다

모듈 방식의 설계(주의, 순전파, 정규화, 잔차)는 여러 분야에서 놀랄 만큼 잘 통함이 드러나 트랜스포머를 요즘 딥러닝의 주된 구조로 자리매김했다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
4. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
