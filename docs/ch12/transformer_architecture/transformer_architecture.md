# 트랜스포머 구조 훑어보기

트랜스포머는 순환 상태 대신 어텐션과 위치별 피드포워드 레이어로 시퀀스를 처리한다. 원 논문의 인코더·디코더 구조를 살펴보고, 병렬 학습의 이점과 어텐션의 계산 비용을 구분한다. 이 페이지의 코드는 구조를 설명하는 불완전한 예제이며 아래 구현의 제한을 함께 읽어야 한다.

## 1. 역사적 배경과 동기

어텐션을 결합한 RNN·LSTM·GRU는 대표적인 Seq2Seq 방식이다. 트랜스포머는 시간축의 순환 의존을 제거해 학습 시 토큰 위치를 병렬로 처리하도록 구성한다. 순환 구조가 모든 과제에서 부적합하다는 뜻은 아니다.

### 차례차례 처리해야 하는 병목

순환 뉴럴 넷트웍은 한 번에 토큰 하나씩 시퀀스을 처리하며 은닉 상태 $h_t$마다 앞선 상태 $h_{t-1}$에 매인다.

$$
h_t = f(h_{t-1}, x_t)
$$

시간 t의 상태를 계산하려면 앞선 상태가 필요하므로 동일한 시퀀스의 시간축을 완전히 병렬화하기 어렵다. 배치·피처·행렬 연산은 병렬화할 수 있어 RNN의 모든 연산이 직렬인 것은 아니다.

긴 시퀀스에서는 순차 의존이 지연 시간에 영향을 준다. 실제 처리 속도는 시퀀스 길이 외에도 배치 크기·장치·커널과 모델 크기에 따라 달라진다.

### 먼 거리 의존 문제

순환 뉴럴 넷트웍은 다음 까닭으로 먼 거리에 걸쳐 정보를 지키기 어려워한다.

1. **그래디언트 소실·폭발**: 여러 시점의 야코비안 곱이 학습을 어렵게 할 수 있다. 게이트와 초기화는 이를 완화하지만 모든 길이에서 해결하지는 않는다.
2. **상태 용량**: 단일 최종 상태로 전체 입력을 전달하는 구성에는 압축 병목이 있다. 어텐션으로 전체 인코더 출력을 읽는 Seq2Seq는 이 병목을 완화한다.
3. **정보 전달 경로**: 기본 RNN에서는 멀리 떨어진 시점 사이의 정보가 여러 상태 갱신을 거친다.

### 트랜스포머가 이 어려움을 어떻게 푸는가

전역 셀프 어텐션은 한 레이어에서 허용된 임의의 두 위치를 직접 연결할 수 있다. 인과·윈도 마스크가 있으면 연결 범위가 달라진다.

1. **학습의 위치별 병렬화**: 정답 입력이 주어지면 여러 쿼리 위치를 행렬 연산으로 처리한다. 자기 회귀 생성은 여전히 순차적이다.
2. **짧은 위치 간 경로**: 전역 어텐션은 먼 위치를 직접 참조할 수 있지만 유용한 의존을 실제로 학습한다는 보장은 없다.
3. **다른 그래디언트 경로**: 시간축 순환은 없지만 깊이·소프트맥스·정규화에 따른 그래디언트 문제는 남는다.
4. **위치 정보**: 위치 벡터, 회전, 바이어스나 마스크로 위치 관계를 반영한다.

### 핵심 혁신 간추리기

트랜스포머는 한데 모여 큰 전환을 이루는 여러 구조적 혁신을 들여왔다.

| 요소 | 역할 | 해석할 때의 조건 |
|---|---|---|
| 셀프 어텐션 | 위치 간 정보 결합 | 마스크가 허용하는 위치만 참조 |
| 멀티헤드 어텐션 | 여러 투영 공간의 병렬 계산 | 헤드별 의미 역할이 고정되지는 않음 |
| 위치 표현 | 순서·거리 정보 반영 | 별도 위치 벡터 외에도 다양한 방식이 있음 |
| 스케일 내적 | 점수 크기 조절 | 모든 입력 분산에서 안정성을 보장하지 않음 |
| 잔차 연결·정규화 | 깊은 블록 구성 | 러닝레이트·초기화와 함께 설계 |

---

## 2. 구조 개관

트랜스포머는 인코더-디코더 구조를 따르지만 요즘 변형은 인코더만(BERT) 또는 디코더만(GPT) 쓰는 경우가 많다.

### 큰 틀의 구조

$$
\text{Transformer}(X,Y_{\text{in}})=\text{OutputProjection}(\text{Decoder}(Y_{\text{in}},\text{Encoder}(X)))
$$

온전한 구조는 다음으로 이루어진다.

1. **입력 임베딩 레이어**: 토큰을 빽빽한 벡터로 바꾼다
2. **위치 인코딩**: 시퀀스의 차례 정보를 넣는다
3. **인코더 스택**: 같은 구조의 인코더 레이어 $N$개이며 보통 레이어별 파라메터는 독립
4. **디코더 스택**: 같은 구조의 디코더 레이어 $N$개이며 보통 레이어별 파라메터는 독립
5. **출력 선형 레이어**: 어휘 크기로 투영한다
6. **소프트맥스**: 확률 분포를 낸다

### 차원의 흐름

다음을 가진 모델에서는

- 어휘 크기 $V$
- 모델 차원 $d_{\text{model}}$
- 시퀀스 길이 $L$
- 배치 크기 $B$

원문 길이를 S, 목표 입력 길이를 T로 구분하면 차원은 다음과 같다. 두 길이가 같을 필요는 없다.

$$
\begin{aligned}
\text{Source tokens} &: (B,S) \\
\text{Encoder input/output} &: (B,S,d_{\text{model}}) \\
\text{Target input} &: (B,T) \\
\text{Decoder output} &: (B,T,d_{\text{model}}) \\
\text{Logits} &: (B,T,V)

---

## 3. 핵심 부품

### 1. 입력 임베딩

입력 임베딩은 정수 토큰 ID를 학습 가능한 실수 벡터로 바꾼다.

$$
\mathbf{E} = \text{Embedding}(\mathbf{x}) \cdot \sqrt{d_{\text{model}}}
$$

원 논문은 임베딩에 모델 차원의 제곱근을 곱한다. 다른 모델의 초기화·위치 표현과 결합할 때도 반드시 같은 배율을 써야 한다는 규칙은 아니다.

### 2. 위치 인코딩

위치 정보나 위치 마스크가 없는 셀프 어텐션은 입력 순열에 따라 출력도 같이 재배열되는 순열 동변성을 갖는다. 다음은 입력 임베딩에 사인·코사인 위치 벡터를 더하는 방식이다.

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

### 3. 인코더 레이어

인코더 레이어마다 하위 레이어를 둘 담는다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### 4. 디코더 레이어

디코더 레이어마다 하위 레이어를 셋 담는다.

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHeadAttention}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{MultiHeadAttention}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}}, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

### 5. 피드포워드 뉴럴 넷트웍

위치별 피드포워드 뉴럴 넷트웍은 다음과 같다.

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

차원은 다음과 같다.

- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$

대개 $d_{ff} = 4 \times d_{\text{model}}$이다.

---

## 4. PyTorch 구현

**구현의 핵심 제한:** Transformer.decode는 memory를 인수로 받지만 사용하지 않는다. decoder_layers도 인코더와 같은 TransformerBlock이어서 교차 어텐션이 없다. 따라서 현재 코드의 디코더 출력은 원문 src와 연결되지 않으며, 원문에 조건부인 번역 모델을 구현한 것이 아니다. 학습 로스를 디코더 출력에서 계산해도 인코더로 경사가 전달되지 않는다.

인코더·디코더의 개념적 설명과 이 코드의 실제 동작을 구분한다. 완전한 구현에는 디코더의 교차 어텐션과 원문 메모리·패딩 마스크 전달이 필요하다. 현재 forward의 기본 tgt_mask=None은 미래 목표 토큰 접근도 허용하므로 인과 마스크를 명시해야 한다.

위치 인코딩은 짝수 d_model과 max_len 이내의 입력을 가정한다. 생성한 목표 마스크는 CPU 텐서이므로 GPU 입력과 함께 사용할 때 장치도 맞춰야 한다. 반환값은 소프트맥스 전 로짓이다. 보존된 출력의 모양·파라메터 수는 번역 성능의 증거가 아니다.

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
        
        # 순방향 뉴럴 넷트웍
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 레이어 정규화
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
        
        # 잔차 연결이 있는 순방향 뉴럴 넷트웍
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
        
        # 인코더
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
        
        # 웨이트 초기화
        self._init_weights()
    
    def _init_weights(self):
        """사비에르 균등으로 웨이트를 초기화한다."""
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
        
        # 인코더 레이어를 통과시킨다
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
        
        # 디코더 레이어를 통과시킨다
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
        트랜스포머를 지나는 피드포워드.
        
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
    
    # 파라메터 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

**출력:**

```
Output shape: torch.Size([32, 15, 10000])
Total parameters: 53,198,608
```

---

## 5. 하이퍼파라미터

원래 트랜스포머 논문은 설정 둘을 썼다.

| 파라메터 | 기본 모델 | 큰 모델 |
|-----------|------------|-----------|
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $h$ (헤드 수) | 8 | 16 |
| $N$ (레이어 수) | 6 | 6 |
| $d_k = d_v$ | 64 | 64 |
| 파라메터 | 6500만 | 2억 1300만 |

---

## 6. 계산 복잡도

### 셀프 어텐션의 복잡도

시퀀스 길이가 $n$이고 모델 차원이 $d$일 때 다음과 같다.

$$
\text{Time: } O(n^2 \cdot d) \qquad \text{Space: } O(n^2 + n \cdot d)
$$

이 식은 QK 내적과 V 가중합 및 명시적인 점수 저장을 나타낸다. Q·K·V와 출력 투영에는 추가 O(nd²) 연산이 든다. 배치·헤드·레이어 수와 역전파 저장은 생략했다. 메모리 효율 커널은 전체 점수 행렬 저장을 피할 수 있다.

### 순전파의 복잡도

$$
\text{Time: } O(n \cdot d \cdot d_{ff}) \qquad \text{Space: } O(n \cdot d_{ff})
$$

### 순환 뉴럴 넷트웍과 비교

| 비교 기준 | 전역 셀프 어텐션 | 기본 RNN |
|---|---|---|
| 학습의 시간축 순차 의존 | 위치별 병렬 계산 가능 | n개 상태 갱신에 의존 |
| 점수·상태 계산 | $O(n^2d)$, 투영은 별도 | 입력·은닉 차원 d 가정 시 $O(nd^2)$ |
| 두 위치의 직접 연결 | 마스크가 허용하면 한 레이어 | 중간 상태를 거침 |
| 자기 회귀 생성 | 순차적 | 순차적 |

---

## 7. 핵심 설계 결정

### 1. 잔차 연결

잔차 연결은 깊은 뉴럴 넷트웍에서 그래디언트가 흐르게 한다.

$$
\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}^{(l)}))
$$

### 2. 레이어 정규화

레이어 정규화는 학습을 안정되게 한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서 $\mu$와 $\sigma^2$는 각 토큰의 피처 축에서 계산한 평균과 분산이다. epsilon은 분산에 더한 뒤 제곱근을 취한다.

### 3. Pre-LN와 Post-LN

**Post-LN (원래 방식):**

$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**Pre-LN (요즘 변형에서 흔하다):**

$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

Pre-LN는 아주 깊은 모델을 학습할 때 더 안정적인 편이다.

---

## 8. 변형과 확장

### 인코더만 (BERT 방식)

- 양방향 어텐션
- 이해 과제에 쓴다
- 마스크 언어 모델링으로 사전 학습한다

### 디코더만 (GPT 방식)

- 인과(한 방향) 주의
- 생성 과제에 쓴다
- 다음 토큰 맞히기로 사전 학습한다

### 인코더-디코더 (T5 방식)

- 온전한 시퀀스 대 시퀀스 능력
- 인코더와 디코더 사이의 크로스 어텐션
- 텍스트에서 텍스트로 옮기는 하나의 틀

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
트랜스포머 인코더와 완전한 디코더의 하위 레이어를 구분하라.

</div>

??? success "연습문제 1 풀이"
    인코더는 셀프 어텐션과 위치별 피드포워드를 갖는다. 인코더·디코더 구조의 디코더에는 인과 셀프 어텐션, 원문을 읽는 교차 어텐션과 피드포워드가 필요하다. 이 페이지 코드는 교차 어텐션을 생략해 설명된 완전한 모델과 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
N=6, d=512, h=8, d_ff=2048인 인코더 블록 스택의 파라메터를 바이어스까지 계산하라.

</div>

??? success "연습문제 2 풀이"
    한 블록의 어텐션은 $4d^2+4d=1,050,624$, 피드포워드는 $2dd_{ff}+d_{ff}+d=2,099,712$, LayerNorm 둘은 $4d=2,048$개다. 블록당 3,152,384개, 6개는 18,914,304개다. 임베딩·최종 정규화·출력 레이어는 제외한다. 고정 d에서 헤드 수를 바꿔도 이 투영 파라메터 수는 바뀌지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Pre-LN 잔차 연결의 야코비안과 그 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"
    $y=x+F(N(x))$이면 $J=I+J_FJ_N$이다. 항등 경로가 있지만 다른 항의 크기와 상쇄 때문에 전체 그래디언트가 항상 보존되는 것은 아니다. Post-LN에서는 잔차 합 뒤의 정규화 야코비안도 곱해진다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
병렬 학습·자기 회귀 생성·메모리를 구분해 RNN과 비교하라.

</div>

??? success "연습문제 4 풀이"
    트랜스포머는 정답 시퀀스가 주어진 학습에서 위치별 계산을 병렬화할 수 있지만 다음 입력을 예측으로 정하는 생성은 순차적이다. 명시적 전역 점수 행렬은 길이의 제곱에 비례하고 기본 RNN의 시점별 활성 저장은 길이에 비례한다. 추론 상태·KV 캐시·메모리 효율 어텐션 등 구현 조건에 따라 실제 메모리는 달라진다.

## 정리하며

트랜스포머의 위치 간 어텐션, 위치별 피드포워드와 잔차·정규화를 연결해 이해한다. 병렬 학습과 순차 생성의 차이, 점수 계산과 투영 비용을 구분한다. 특히 이 예제는 디코더가 인코더 메모리를 사용하지 않으므로 완성된 번역 모델로 해석해서는 안 된다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
4. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
