# 인코더-디코더 구조

트랜스포머는 인코더 전용, 디코더 전용과 인코더·디코더 구조로 구성할 수 있다. 각 구조가 읽을 수 있는 문맥과 학습 목표를 구분한다. 분류·이해·생성의 가능 여부를 구조 이름 하나로 단정할 수는 없다.

## 1. 원래의 인코더-디코더 구조

원래 트랜스포머는 기계 번역을 위한 인코더-디코더 모델로 설계되었다. 간추린 변형을 살피기 전에 이 온전한 구조를 이해해 두어야 한다.

### 인코더 스택

인코더는 같은 구조의 레이어 N개를 쌓는다. 원 논문의 기본 구성은 6개이며, 일반적으로 서로 다른 레이어의 웨이트는 공유하지 않는다. 각 레이어에는 셀프 어텐션과 위치별 피드포워드가 있다.

**하위 레이어 1 — 멀티헤드 셀프 어텐션:**

$$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

패딩 등 마스크가 제외한 위치를 빼고 각 토큰이 원문 전체를 참조할 수 있다. “양방향”은 패딩 마스크도 없다는 뜻이 아니다.

**하위 레이어 2 — 자리별 피드포워드 뉴럴 넷트웍:**

$$\text{FFN}(\mathbf{z}) = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

$$\mathbf{H} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$

피드포워드 뉴럴 넷트웍은 위치마다 똑같이 따로 적용된다. 자리끼리의 주고받음 없이 토큰별 비선형 능력을 더한다.

### 디코더 스택

원래 인코더·디코더 구조의 디코더는 같은 구조의 레이어를 쌓되 레이어별 웨이트는 독립이다. 각 레이어에는 다음 세 하위 레이어가 있다.

**하위 레이어 1 — 마스크 셀프 어텐션:**

$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHead}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y}))$$

인과 마스크가 앞으로의 자리에 주의하지 못하게 막아 자기 회귀 성질을 지킨다.

**하위 레이어 2 — 크로스 어텐션:**

$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{MultiHead}(\mathbf{Y}', \mathbf{M}, \mathbf{M}))$$

쿼리는 디코더에서 오고 키와 값은 인코더의 출력 $\mathbf{M}$에서 온다. 이것이 인코더와 디코더를 잇는 다리이다.

**하위 레이어 3 — 피드포워드 뉴럴 넷트웍:**

$$\mathbf{Y}''' = \text{LayerNorm}(\mathbf{Y}'' + \text{FFN}(\mathbf{Y}''))$$

### 정보 흐름 그림

```
Source tokens          Target tokens (shifted right)
     │                        │
  [Embedding + PE]        [Embedding + PE]
     │                        │
  ┌──▼──┐                ┌───▼───┐
  │ Self │                │Masked │
  │ Attn │                │ Self  │
  │(bidir)│               │ Attn  │
  └──┬──┘                └───┬───┘
     │                        │
  [Add & Norm]           [Add & Norm]
     │                        │
  ┌──▼──┐                ┌───▼───┐
  │ FFN │     Memory M    │Cross  │◄── K, V from encoder
  └──┬──┘  ──────────►   │ Attn  │
     │                    └───┬───┘
  [Add & Norm]           [Add & Norm]
     │                        │
     │    ×N layers       ┌───▼───┐
     │                    │  FFN  │
     ▼                    └───┬───┘
  Encoder                [Add & Norm]
  Output M                    │
                          ×N layers
                              │
                           [Linear]
                           [Softmax]
                              │
                          Output probs
```

### 잔차 연결과 레이어 정규화

트랜스포머의 하위 레이어마다 잔차 연결을 쓴 뒤 레이어 정규화를 한다. 이 설계 선택은 학습의 안정성에 매우 중요하다.

$$\text{output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

**Post-LN**(원래 논문): 위에서 보인 대로 잔차를 더한 뒤 레이어 정규화를 한다.

**Pre-LN**(요즘 방식): 하위 레이어에 들어가기 전에 레이어 정규화를 한다. 깊은 모델에서 더 안정적임이 밝혀졌다.

$$\text{output} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$$

Pre-LN은 항등 잔차 경로가 정규화 바깥에 남는다. 깊은 모델의 학습에 유리할 수 있지만 워밍업 불필요나 그래디언트 크기 보존, 최종 성능 우위를 보장하는 것은 아니다.

---

## 2. 인코더만 (BERT 방식)

아래 짧은 EncoderOnly는 토큰 임베딩과 스택만 연결한다. 위치 표현·패딩 마스크·BERT의 학습 헤드가 없으며 BERT 전체 구현은 아니다. 셀프 어텐션이 순열 동변이라는 점을 이 단순화와 연결해 읽는다.

**어텐션**: 모든 자리가 모든 자리를 본다(양방향)

인코더만 쓰는 구조는 디코더를 아주 없앤다. 모델은 입력을 양방향 셀프 어텐션 레이어 스택로 처리하여 토큰마다 맥락이 담긴 표현을 낸다.

```python
class EncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        return self.encoder(self.embed(x))  # 인과 가림 없음
```

**쓰임새**: 분류, 개체명 인식, 쿼리응답 추출, 문장 임베딩

**사전 학습**: BERT의 마스크 언어 모델링(MLM)은 선택한 토큰을 양방향 문맥으로 예측한다. 이 목표와 순차적인 다음 토큰 생성은 다르지만 인코더가 “이해만 할 수 있다”는 구조적 제약을 뜻하지는 않는다.

---

## 3. 디코더만 (GPT 방식)

**어텐션**: 인과 (위치마다 지난 것만 본다)

디코더 전용 구성은 외부 인코더와 교차 어텐션 없이 인과 셀프 어텐션·피드포워드 블록을 쌓는다.

```python
class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(2048, 2048), 1).bool())
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.layers(self.embed(x), mask=self.mask[:seq_len, :seq_len])
        return self.lm_head(h)
```

**쓰임새**: 글 생성, 대화, 코드, 맥락 안 학습

**사전 학습**: 인과 언어 모델링(CLM)은 앞선 토큰을 조건으로 다음 토큰을 예측한다. 각 유효 위치에 로스를 줄 수 있지만 다른 목표보다 항상 데이터 효율이 좋다는 뜻은 아니다.

**코드의 범위**: 아래처럼 nn.TransformerEncoder에 인과 마스크를 주면 디코더 전용 블록을 구성하는 출발점이 된다. 다만 이 짧은 구현에는 위치 표현과 패딩 처리가 없고, 고정 마스크의 길이는 2048이다. 이 코드가 GPT의 전체 구조·사전 학습을 재현하는 것은 아니다.

---

## 4. 인코더-디코더 (T5 방식)

아래 EncoderDecoder는 교차 어텐션을 갖지만 T5의 상대 위치·정규화·사전 학습을 구현하지 않는다. 원문·목표가 하나의 어휘 임베딩을 공유하고 패딩 마스크·위치 표현이 없다. tgt_mask는 CPU에서 만들어져 GPU 입력과의 장치도 맞춰야 한다.

**어텐션**: 인코더는 양방향, 디코더는 인과, 그리고 크로스 어텐션

인코더-디코더 구조는 원래 트랜스포머 설계를 온전히 지닌다. 인코더가 입력을 양방향으로 처리하고, 디코더는 인코더의 표현에 주의하면서 출력을 자기 회귀로 만들어 낸다.

```python
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        memory = self.encoder(self.embed(src))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), 1).bool()
        h = self.decoder(self.embed(tgt), memory, tgt_mask=tgt_mask)
        return self.lm_head(h)
```

**쓰임새**: 번역, 요약, 텍스트에서 텍스트로 옮기는 과제

**사전 학습**: T5는 일부 구간을 제거하고 특수 표식으로 대체해 제거된 구간을 출력하게 한다. BART는 손상된 입력에서 원문을 복원한다. 둘 다 입력 복원 계열이지만 출력 목표가 정확히 같지는 않다.

---

## 5. 자세히 비교

| 측면 | 인코더 전용 | 디코더 전용 | 인코더·디코더 |
|---|---|---|---|
| 일반적인 입력 문맥 | 양방향 | 과거와 현재 입력 | 원문은 양방향, 목표는 인과 |
| 기본 출력 | 문맥별 토큰 표현 | 다음 토큰 로짓 | 원문 조건부 다음 토큰 로짓 |
| 교차 어텐션 | 기본 구성에 없음 | 기본 구성에 없음 | 디코더가 인코더 메모리를 읽음 |
| 생성 시 캐시 | 적용 방법에 따라 다름 | 셀프 어텐션 K·V | 셀프 K·V와 원문에서 투영한 교차 K·V |
| 파라메터 수 | 레이어·너비·어휘·헤드 구성에 따라 결정 | 같은 기준으로 계산 | 교차 어텐션과 별도 스택 비용 포함 |

### 핵심 트레이드오프

**문맥 접근**: 인코더의 각 토큰 표현은 원문의 앞뒤를 볼 수 있다. 디코더 전용 모델의 현재 출력은 과거 입력과 프롬프트 전체를 활용할 수 있어 생성 외의 분류·추론 과제에도 쓸 수 있다. 인코더·디코더는 원문을 양방향으로 처리하면서 목표문을 순차적으로 생성한다.

**파라메터 수**: 인코더와 디코더를 각각 N레이어씩 두면 단일 N레이어 스택보다 계수가 늘지만 정확히 두 배는 아니다. 교차 어텐션·출력 레이어·어휘 공유 여부도 포함해야 한다. 전체 파라메터와 학습·생성 비용을 같은 예산에서 비교한다.

**학습 목표**: 구조는 사용할 수 있는 문맥을 정하고, 로스는 그 문맥에서 무엇을 예측할지 정한다. 대표 목표가 구조와 잘 맞더라도 한 구조에 오직 한 가지 목표만 가능한 것은 아니다.

**추론 효율**: 디코더의 자기 회귀 캐시는 출력 길이에 따라 커진다. 인코더·디코더에서는 원문을 인코딩하고 교차 어텐션의 원문 K·V를 한 번 계산해 재사용할 수 있다. 어느 구조가 빠른지는 원문·목표 길이, 캐시·투영 구현과 장치에 따라 달라진다.

### 피드포워드 뉴럴 넷트웍의 몫

세 구조의 위치별 피드포워드 레이어는 토큰마다 같은 비선형 변환을 적용한다. 이를 키·값 메모리에 비유하는 분석이 있지만, 모든 사실 지식이 이 레이어에만 저장되거나 검색 결과가 언제나 정확하다는 뜻은 아니다.

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$$

여기서 $W_1\in\mathbb{R}^{d\times d_{ff}}$, $W_2\in\mathbb{R}^{d_{ff}\times d}$다. $d_{ff}=4d$이면 주요 웨이트는 $8d^2$개로, 표준 Q·K·V·출력 투영의 $4d^2$개에 비해 약 두 배다. 바이어스와 게이트형 FFN 등 변형은 별도로 센다.

---

## 6. 역사의 흐름

2017년 원래 트랜스포머는 번역용 인코더·디코더를 제시했다. 이후 BERT는 양방향 인코더와 MLM을, GPT 계열은 인과 언어 모델링을, T5는 텍스트 입력·출력의 통일된 틀을 보여 주었다.

이 흐름을 한 구조가 모든 과제에서 다른 구조를 대체했다는 주장으로 읽지 않는다. 모델 규모·데이터·학습 목표가 함께 달라지므로, 구조의 효과만 비교하려면 조건을 통제해야 한다. 맥락 내 학습도 모든 규모와 과제에서 동일하게 나타나거나 미세 조정을 불필요하게 만드는 보장은 아니다.

## 7. 언제 쓰는가

- **인코더 전용**: 입력 전체의 표현을 활용하는 분류·추출·임베딩의 출발점이다.
- **디코더 전용**: 프롬프트에서 이어지는 생성과 다음 토큰 예측에 자연스럽다.
- **인코더·디코더**: 원문과 출력의 경계가 뚜렷한 번역·요약 등 조건부 생성에 알맞다.

이 선택은 출발점이며 목표 품질·지연 시간·메모리와 사용 가능한 사전 학습 모델을 기준으로 비교한다.

## 8. 인코더 블록 깊이 들여다보기

상세 예제의 입력 attention_mask는 1이 유효, 0이 패딩이며 내부 key_padding_mask는 True가 차단이다. 모든 키가 패딩이면 소프트맥스에서 NaN이 발생할 수 있다. 패딩 쿼리의 출력까지 자동으로 0으로 만들지는 않는다.

Pre-LN의 all_hidden_states는 최종 정규화 전의 마지막 블록 출력까지 담지만 last_hidden_state에는 최종 정규화가 적용된다. 두 값을 곧바로 같다고 비교하면 안 된다. 어텐션 웨이트는 드롭아웃 이후이므로 학습 모드의 행 합은 항상 1이 아니다. 예제의 108,890,112개는 학습 전 구조의 계수 수다.

### 훑어보기

트랜스포머 인코더 블록은 BERT 같은 인코더 기반 구조의 근본 되는 구조 단위이다. 블록마다 셀프 어텐션과 피드포워드 레이어로 입력을 바꾸어, 시퀀스의 길이는 지키면서 표현을 다듬는다.

### 구조

인코더 블록마다 잔차 연결과 레이어 정규화를 갖춘 하위 레이어 둘로 이루어진다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### 부품 나누어 보기

1. **멀티헤드 셀프 어텐션**: 모든 자리 사이의 의존을 잡아낸다
2. **피드포워드 뉴럴 넷트웍**: 넓혔다 줄이는 자리별 변환
3. **잔차 연결**: 깊은 뉴럴 넷트웍에서 그래디언트가 흐르게 한다
4. **레이어 정규화**: 학습을 안정되게 한다

### 인코더의 멀티헤드 셀프 어텐션

인코더는 위치마다 모든 자리에 참조할 수 있는 양방향 셀프 어텐션을 쓴다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

셀프 어텐션에서는 $Q$, $K$, $V$이 모두 같은 입력에서 나온다.

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

### 입력에서 주의까지: 한 걸음씩

날것의 문장이 인코더를 어떻게 흘러가는지 이해하려면 변환을 하나씩 따라가야 한다.

**1단계 — 토큰 나누기**: 입력 문장을 토큰(단어이나 부분 단어)으로 나눈다. 이를테면 "The cat sat"은 토큰 시퀀스 $[\text{The}, \text{cat}, \text{sat}]$이 된다.

**2단계 — 임베딩 찾기**: 임베딩 행렬 $E \in \mathbb{R}^{V \times d_{\text{model}}}$으로 토큰마다 빽빽한 벡터를 잇대어, $n$이 시퀀스의 길이일 때 행렬 $X_{\text{embed}} \in \mathbb{R}^{n \times d_{\text{model}}}$을 만든다. 임베딩에는 $\sqrt{d_{\text{model}}}$을 곱한다.

**3단계 — 위치 인코딩**: 사인·코사인 위치 인코딩을 성분별로 더한다.

$$
X = X_{\text{embed}} \cdot \sqrt{d_{\text{model}}} + \text{PE}
$$

그 결과 $X \in \mathbb{R}^{n \times d_{\text{model}}}$은 (임베딩에서 온) 뜻과 위치 정보를 함께 지닌다.

**4단계 — Q, K, V로의 선형 투영**: 서로 다른 웨이트 행렬 셋이 $X$을 쿼리, 키, 값 공간으로 투영한다.

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

여기서 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$이다. 이 투영은 (표준 정식화에서 바이어스 항이 없는) 선형이며, 무엇을 찾을지(쿼리), 무엇을 내세울지(키), 어떤 정보를 줄지(값)에 대해 모델이 저마다 다른 표현을 배우게 한다.

**5단계 — 여러 헤드로 나누기**: $Q$, $K$, $V$ 각각을 마지막 차원을 따라 헤드 $h$개로 나누며 헤드마다 차원은 $d_k = d_{\text{model}} / h$이다.

$$
Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k} \quad \text{for } i = 1, \ldots, h
$$

**6단계 — 헤드마다 스케일 조정 내적 어텐션**: 헤드마다 따로 주의를 계산한다.

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

$\sqrt{d_k}$으로 나누는 것은 내적의 크기가 너무 커져 소프트맥스가 그래디언트가 매우 작은 영역으로 밀려나는 일을 막는다.

**7단계 — 이어 붙이고 투영하기**: 헤드들의 출력을 이어 붙이고 선형으로 투영한다.

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

여기서 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$이다(역시 표준 정식화에서 바이어스 항이 없다). 출력은 입력과 같은 꼴 $\mathbb{R}^{n \times d_{\text{model}}}$이다.

!!! note "수식의 바이어스 생략과 실제 구현"
    설명 수식에서 Q·K·V와 출력 투영의 바이어스를 생략할 수 있다. 이것이 모든 구현에서 바이어스가 불필요하다는 증명은 아니다. 이 페이지의 사용자 정의 코드와 PyTorch nn.MultiheadAttention 기본값은 바이어스를 사용한다. bias=False는 투영 바이어스를 없애는 설정이며 그 자체로 원 논문의 모든 학습·구조 조건을 재현하지는 않는다.

### 멀티헤드 정식화

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

여기서 헤드마다 다음과 같다.

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

### 피드포워드 뉴럴 넷트웍

자리별 피드포워드 뉴럴 넷트웍은 위치마다 따로 같은 변환을 적용한다.

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

또는 (요즘 모델에서 흔한) GELU 활성으로 다음과 같이 한다.

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

### 차원 넓히기

피드포워드 뉴럴 넷트웍은 대개 숨은 차원을 4배로 넓힌다.

- 입력: $d_{\text{model}}$
- 은닉 레이어: $d_{ff} = 4 \times d_{\text{model}}$
- 출력: $d_{\text{model}}$

### Pre-LN와 Post-LN

### Post-LN (원래 트랜스포머)

$$
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))
$$

### Pre-LN (요즘의 표준)

$$
\mathbf{X}' = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))
$$

Pre-LN은 깊은 모델에서 안정적인 학습에 도움이 될 수 있다. 실제 워밍업 필요성은 러닝레이트·초기화·최적화 설정에 따라 판단한다.

### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadSelfAttention(nn.Module):
    """
    트랜스포머 인코더를 위한 다중 머리 자기 주의.
    
    모든 자리가 다른 모든 자리에 주의할 수 있다(양방향).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        인수:
            d_model: 모형 차원
            num_heads: 주의 머리의 수
            dropout: 드롭아웃 확률
            bias: 선형 사영에 바이어스를 둘지 여부
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 효율을 위해 QKV를 한데 모은 사영
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # 출력 사영
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # 드롭아웃
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 토큰을 위한 가림 [batch_size, seq_len]
                가릴 자리(채움)가 True
            return_attention: 어텐션 웨이트를 돌려줄지 여부
            
        반환값:
            output: 바뀐 텐서 [batch_size, seq_len, d_model]
            attention_weights: 선택으로 돌려주는 주의 웨이트 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 사영 한 번으로 Q, K, V를 셈한다
        qkv = self.qkv_proj(x)  # [batch, seq, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 주의 점수를 셈한다
        # [batch, heads, seq, seq]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 채움 가림이 있으면 적용한다
        if key_padding_mask is not None:
            # 가림을 넓힌다: [batch, seq] -> [batch, 1, 1, seq]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 소프트맥스와 드롭아웃
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 값에 어텐션 적용
        # [batch, heads, seq, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None

class PositionWiseFeedForward(nn.Module):
    """
    자리별 순전파 뉴럴 넷트웍.
    
    FFN(x) = activation(xW1 + b1)W2 + b2
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        인수:
            d_model: 모형 차원
            d_ff: 순전파의 숨은 차원
            dropout: 드롭아웃 확률
            activation: 엑티베이션 함수('relu' 또는 'gelu')
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            
        반환값:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    트랜스포머 인코더 블록 하나.
    
    다음으로 이루어진다:
    1. 다중 머리 자기 주의 + 잔차 + 레이어 정규화
    2. 자리별 순전파 + 잔차 + 레이어 정규화
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        """
        인수:
            d_model: 모형 차원
            num_heads: 주의 머리의 수
            d_ff: 순전파의 숨은 차원
            dropout: 드롭아웃 확률
            activation: 순전파의 엑티베이션 함수
            pre_norm: 앞 정규화(True)를 쓸지 뒤 정규화(False)를 쓸지
        """
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # 자기 주의
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 순방향 뉴럴 넷트웍
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 잔차 연결을 위한 드롭아웃
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 가림 [batch_size, seq_len]
            return_attention: 어텐션 웨이트를 돌려줄지 여부
            
        반환값:
            output: 바뀐 텐서 [batch_size, seq_len, d_model]
            attention_weights: 선택으로 돌려주는 주의 웨이트
        """
        if self.pre_norm:
            # 앞 정규화: 아래 레이어 앞에서 레이어 정규화
            # 주의 블록
            residual = x
            x = self.norm1(x)
            attn_output, attn_weights = self.self_attention(
                x, key_padding_mask, return_attention
            )
            x = residual + self.dropout(attn_output)
            
            # 순전파 블록
            residual = x
            x = self.norm2(x)
            ff_output = self.feed_forward(x)
            x = residual + self.dropout(ff_output)
        else:
            # 뒤 정규화: 아래 레이어 뒤에서 레이어 정규화
            # 주의 블록
            attn_output, attn_weights = self.self_attention(
                x, key_padding_mask, return_attention
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            # 순전파 블록
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """
    온전한 트랜스포머 인코더 더미.
    
    다음으로 이루어진다:
    1. 토큰 임베딩
    2. 위치 인코딩
    3. 인코더 블록 N개의 더미
    4. 선택으로 두는 마지막 레이어 정규화(앞 정규화를 위해)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        """
        인수:
            vocab_size: 어휘 크기
            d_model: 모형 차원
            num_heads: 주의 머리의 수
            num_layers: 인코더 블록의 수
            d_ff: 순전파의 숨은 차원
            max_len: 순차열의 최대 길이
            dropout: 드롭아웃 확률
            activation: 순전파의 엑티베이션 함수
            pre_norm: 앞 정규화 구조를 쓸지 여부
        """
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩 (학습되는 것)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # 임베딩 드롭아웃
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 인코더 블록
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # 마지막 레이어 정규화 (앞 정규화 구조를 위해)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None
        
        # 웨이트 초기화
        self._init_weights()
    
    def _init_weights(self):
        """웨이트를 초기화한다."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> dict:
        """
        피드포워드.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 주의 가림 [batch_size, seq_len]
                실제 토큰은 1, 채움은 0
            return_all_hidden_states: 모든 레이어의 출력을 돌려줄지 여부
            
        반환값:
            다음을 담은 사전:
                - last_hidden_state: 마지막 인코더 출력
                - all_hidden_states: 선택으로 돌려주는 모든 레이어의 출력 목록
        """
        batch_size, seq_len = input_ids.shape
        
        # 자리 번호를 만든다
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 임베딩
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # 토큰 임베딩의 크기를 조정한다
        x = token_embeds * math.sqrt(self.d_model) + position_embeds
        x = self.embedding_dropout(x)
        
        # 주의 가림을 열쇠 채움 가림으로 바꾼다
        # attention_mask: 실제는 1, 채움은 0
        # key_padding_mask: 채움이 True (가릴 자리)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # 인코더 레이어를 통과시킨다
        all_hidden_states = [x] if return_all_hidden_states else None
        
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask)
            if return_all_hidden_states:
                all_hidden_states.append(x)
        
        # 마지막 레이어 정규화
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        return {
            'last_hidden_state': x,
            'all_hidden_states': all_hidden_states
        }

# 쓰는 보기와 시험
if __name__ == "__main__":
    # 설정
    vocab_size = 30522  # BERT 같은 어휘
    d_model = 768
    num_heads = 12
    num_layers = 12
    d_ff = 3072
    max_len = 512
    
    # 인코더 만들기
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1,
        pre_norm=True
    )
    
    # 예제 입력
    batch_size = 8
    seq_len = 128
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -10:] = 0  # 채움을 흉내 낸다
    
    # 순전파
    outputs = encoder(input_ids, attention_mask, return_all_hidden_states=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs['last_hidden_state'].shape}")
    print(f"Number of hidden states: {len(outputs['all_hidden_states'])}")
    
    # 파라메터 개수 세기
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 블록 하나를 시험한다
    print("\n--- Testing single encoder block ---")
    single_block = TransformerEncoderBlock(
        d_model=512,
        num_heads=8,
        d_ff=2048
    )
    
    test_input = torch.randn(4, 32, 512)
    output, attn = single_block(test_input, return_attention=True)
    print(f"Block input shape: {test_input.shape}")
    print(f"Block output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
```

**출력:**

```
Input shape: torch.Size([8, 128])
Output shape: torch.Size([8, 128, 768])
Number of hidden states: 13

Total parameters: 108,890,112
Trainable parameters: 108,890,112

--- Testing single encoder block ---
Block input shape: torch.Size([4, 32, 512])
Block output shape: torch.Size([4, 32, 512])
Attention shape: torch.Size([4, 8, 32, 32])
```

### 그래디언트 흐름 분석

### 잔차 연결의 이점

잔차가 없으면 그래디언트가 하위 레이어를 모두 거쳐 흘러야 한다.

$$
\nabla_{x^{(0)}}\mathcal{L}=J_0^T J_1^T\cdots J_{L-1}^T\nabla_{x^{(L)}}\mathcal{L}

항등 건너뛰기가 있는 y=x+F(x) 또는 Pre-LN 블록의 야코비안에는 I 항이 남는다. 이때 전체 곱을 전개하면 항등 경로에서 오는 다음 항이 포함된다. Post-LN은 정규화가 잔차 합 바깥에 있어 같은 식을 그대로 적용하지 않는다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \text{other terms}
$$

### 레이어 정규화의 효과

레이어 정규화는 피처 차원에 걸쳐 정규화한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$

정규화는 표현의 스케일을 조절한다. 잔차와 함께 학습을 돕지만 모든 레이어의 그래디언트 크기가 같게 유지되는 것은 아니다.

### 계산 복잡도

### 레이어마다의 복잡도

시퀀스 길이 $n$, 모델 차원 $d$, 순전파 차원 $d_{ff}$에 대해 다음과 같다.

| 요소 | 주요 연산 | 활성·점수 저장의 대표 비용 |
|---|---|---|
| 어텐션 내적·가중합 | $O(n^2d)$ | 명시적 점수 $O(hn^2)$와 토큰 표현 $O(nd)$ |
| Q·K·V·출력 투영 | $O(nd^2)$ | 토큰 표현 $O(nd)$ |
| 피드포워드 | $O(ndd_{ff})$ | $O(nd_{ff})$ |
| LayerNorm | $O(nd)$ | $O(nd)$ |

배치 크기와 학습용 저장·웨이트 메모리는 생략했다. 점수 행렬 전체를 저장하지 않는 커널은 메모리 비용이 다르다.

### 인코더 전체의 복잡도

레이어가 $L$개일 때 다음과 같다.

$$
\text{Time: } O(L(n^2d+nd^2+ndd_{ff}))
$$

흔한 $d_{ff} = 4d$이면 다음과 같다.

$$
\text{Time: } O(L(nd(n+d)))
$$

### 핵심 내용

트랜스포머 인코더 블록은 모듈 방식의 강력한 구조 단위로 다음을 한다.

1. **전역 의존을 잡아낸다**: 셀프 어텐션이 모든 자리를 잇는다
2. **비선형 변환을 적용한다**: 피드포워드 뉴럴 넷트웍이 표현 능력을 더한다
3. **그래디언트 흐름을 지킨다**: 잔차가 깊은 구조를 가능케 한다
4. **학습을 안정되게 한다**: 레이어 정규화가 활성을 다스린다

인코더 블록을 이해하는 것은 BERT나 RoBERTa 같은 모델과 인코더 기반 구조를 구현하는 데 꼭 필요하다.

### 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

---

## 9. 디코더 블록 깊이 들여다보기

상세 TransformerDecoder는 교차 어텐션이 없는 디코더 전용 모델이다. labels는 내부에서 한 토큰 시프트하며, attention_mask만으로 패딩 정답을 로스에서 제외하지는 않는다. 제외할 labels는 -100으로 표시해야 한다. 길이가 1이면 시프트 후 유효 정답이 없어 평균 로스가 정의되지 않을 수 있다.

generate는 no_grad를 사용하지만 eval 모드로 전환하지 않는다. 예제 그대로라면 드롭아웃이 켜져 있다. EOS 종료·길이 초과 처리·패딩 마스크 전달과 KV 캐시가 없으므로 매번 전체 접두를 재계산한다. temperature>0, 1≤top_k≤어휘 크기와 0<top_p≤1 조건도 호출자가 확인해야 한다. 예제의 무작위 토큰 로스·생성 길이는 언어 품질 평가가 아니다.

### 훑어보기

트랜스포머 디코더 블록은 GPT 같은 자기 회귀 모델의 구조 단위이다. 인코더와 달리 디코더는 **인과(가린) 셀프 어텐션**을 써서 자리가 앞으로의 자리에 주의하지 못하게 막아 왼쪽에서 오른쪽으로의 생성을 가능케 한다.

### 구조의 변형

### 온전한 인코더-디코더 (원래 트랜스포머)

원래 디코더는 하위 레이어가 셋이다.

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{CrossAttn}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

### 디코더만 (GPT 방식)

요즘 언어 모델은 하위 레이어가 둘인 디코더 전용 구조를 쓴다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MaskedSelfAttn}(\mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### 인과 셀프 어텐션

인코더 셀프 어텐션와의 핵심 차이는 앞으로의 자리에 주의하지 못하게 막는 **인과 마스크**이다.

### 인과 마스크

시퀀스 길이가 $n$일 때 인과 마스크 $M$은 다음과 같다.

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

이는 아래 삼각 꼴의 어텐션 패턴를 만든다.

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

### 눈으로 보기

```
Position can attend to:
     1  2  3  4  5
1  [ ✓  ✗  ✗  ✗  ✗ ]
2  [ ✓  ✓  ✗  ✗  ✗ ]
3  [ ✓  ✓  ✓  ✗  ✗ ]
4  [ ✓  ✓  ✓  ✓  ✗ ]
5  [ ✓  ✓  ✓  ✓  ✓ ]
```

### 크로스 어텐션 (인코더-디코더)

시퀀스 대 시퀀스 모델에서 디코더는 인코더의 출력에 어텐션을 적용한다.

$$
\text{CrossAttn}(Z, X_{\text{enc}}) = \text{softmax}\left(\frac{Z W^Q (X_{\text{enc}} W^K)^T}{\sqrt{d_k}}\right) X_{\text{enc}} W^V
$$

여기서:

- 쿼리: 디코더에서 ($Z W^Q$)
- 키와 값: 인코더에서 ($X_{\text{enc}} W^K$, $X_{\text{enc}} W^V$)

### 파이토치 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CausalSelfAttention(nn.Module):
    """
    트랜스포머 디코더를 위한 인과(가린) 자기 주의.
    
    자리마다 제 자신과 앞선 자리에만 주의할 수 있다.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV를 한데 모은 사영
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 인과 가림을 미리 셈한다
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        인과 가림을 쓰는 피드포워드.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 가림 [batch_size, seq_len]
            return_attention: 어텐션 웨이트를 돌려줄지 여부
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V를 셈한다
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 주의 점수를 셈한다
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 인과 가림을 적용한다
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # 채움 가림이 있으면 적용한다
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 소프트맥스와 드롭아웃
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 값에 어텐션 적용
        context = torch.matmul(attn_weights, v)
        
        # 꼴을 바꾸고 사영한다
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None

class TransformerDecoderBlock(nn.Module):
    """
    트랜스포머 디코더 블록 하나(GPT 방식).
    
    다음으로 이루어진다:
    1. 가린 자기 주의 + 잔차 + 레이어 정규화
    2. 순전파 뉴럴 넷트웍 + 잔차 + 레이어 정규화
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_len: int = 2048,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # 가린 자기 주의
        self.self_attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # 순방향 뉴럴 넷트웍
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """피드포워드."""
        if self.pre_norm:
            # 앞 정규화: 아래 레이어 앞에서 레이어 정규화
            residual = x
            x = self.norm1(x)
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = residual + self.dropout(attn_out)
            
            residual = x
            x = self.norm2(x)
            ff_out = self.feed_forward(x)
            x = residual + ff_out
        else:
            # 뒤 정규화: 아래 레이어 뒤에서 레이어 정규화
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
        
        return x

class TransformerDecoder(nn.Module):
    """
    온전한 트랜스포머 디코더(GPT 방식, 디코더만).
    
    언어 모형화와 글 생성을 위한 것이다.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # 토큰 임베딩과 자리 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 디코더 블록
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # 마지막 레이어 정규화 (앞 정규화를 위해)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        # 출력 사영 (임베딩과 웨이트 묶기)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """웨이트를 초기화한다."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        피드포워드.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 주의 가림 [batch_size, seq_len]
            labels: 언어 모형 로스를 위한 이름표 [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 자리 번호
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 임베딩
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_dropout(x)
        
        # 채움 가림
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # 디코더 레이어를 통과시킨다
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        # 마지막 정규화
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 언어 모형 머리
        logits = self.lm_head(x)
        
        # 이름표가 있으면 로스를 셈한다
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """자기 회귀로 글을 짓는다."""
        for _ in range(max_new_tokens):
            # 마지막 자리의 로짓을 얻는다
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # 상위 k 거르기를 적용한다
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # 상위 p(커널) 거르기를 적용한다
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 뽑기 또는 탐욕
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

# 사용 예
if __name__ == "__main__":
    # GPT 방식 디코더 설정
    model = TransformerDecoder(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024
    )
    
    # 예제 입력
    input_ids = torch.randint(0, 50257, (4, 128))
    labels = input_ids.clone()
    
    # 순전파
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # 생성
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    # 파라메터
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

**출력:**

```
Logits shape: torch.Size([4, 128, 50257])
Loss: 10.9665
Generated shape: torch.Size([1, 30])
Total parameters: 124,439,808
```

### 효율적인 추론을 위한 KV 캐시

다음 CausalSelfAttentionWithCache는 K·V를 이어 붙이는 별도 모듈이며 위 TransformerDecoder.generate에 연결되어 있지 않다. dropout은 정의만 되고 forward에서 적용되지 않으며 패딩 마스크 인수도 없다. 모델 전체의 캐시·위치 처리까지 완성된 것으로 해석하지 않는다.

```python
class CausalSelfAttentionWithCache(nn.Module):
    """효율적인 추론을 위해 KV 캐시를 갖춘 인과 자기 주의."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        KV 캐시를 선택으로 쓰는 피드포워드.
        
        인수:
            x: 입력 [batch, seq_len, d_model], 캐시를 쓰면 [batch, 1, d_model]
            past_kv: 앞 단계에서 담아 둔 (열쇠, 값) 텐서
            use_cache: 고친 캐시를 돌려줄지 여부
        """
        batch_size, seq_len, _ = x.shape
        
        # 지금 입력의 Q, K, V를 셈한다
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 지난 KV가 있으면 이어 붙인다
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 다음 되풀이를 위해 담아 둔다
        present_kv = (k, v) if use_cache else None
        
        # 주의 (토큰 하나에 캐시를 쓸 때는 인과 가림이 필요 없다)
        full_seq_len = k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 처음 맥락(seq_len > 1)에만 인과 가림을 적용한다
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_seq_len, device=x.device),
                diagonal=full_seq_len - seq_len + 1
            ).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        return output, present_kv
```

### 계산 분석

### KV 캐시가 없을 때

맥락이 $C$일 때 토큰 $T$개를 만드는 데 다음과 같다.

$$
\text{Complexity} = O(T \cdot (C + T)^2 \cdot d)
$$

### KV 캐시가 있을 때

$$
\text{Complexity} = O(T \cdot (C + T) \cdot d)
$$

위 식은 프롬프트의 최초 처리와 투영·피드포워드·출력 레이어를 제외한 어텐션 생성 비용의 상한 비교다. C를 고정하면 T가 커질 때 캐시 없이 O(T³), 캐시를 사용하면 O(T²)다. 한 단계 비용을 전체 생성 비용과 혼동하지 않는다.

### 디코더와 인코더 비교

| 측면 | 인코더 | 디코더 |
|--------|---------|---------|
| 어텐션의 종류 | 양방향 | 인과 (가린) |
| 앞을 볼 수 있는가 | 그렇다 | 아니다 |
| 주된 쓰임 | 이해 | 생성 |
| 보기 | BERT, RoBERTa | GPT, LLaMA |

### 핵심 내용

트랜스포머 디코더 블록은 다음으로 자기 회귀 생성을 가능케 한다.

1. **인과 마스크**: 앞으로의 자리에 주의하지 못하게 막는다
2. **크로스 어텐션**: 시퀀스 대 시퀀스 모델에서 인코더와 잇는다
3. **KV 캐싱**: 효율적인 점진 생성을 가능케 한다
4. **Pre-LN 구조**: 학습의 안정성을 높인다

### 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
인코더·디코더 번역 모델의 원문과 목표 입력 흐름을 설명하라.

</div>

??? success "연습문제 1 풀이"
    인코더는 패딩을 제외한 원문 문맥을 처리하고 디코더는 한 토큰 이동한 목표 입력을 인과적으로 읽는다. 교차 어텐션의 쿼리는 디코더, 키·값은 인코더 출력에서 만든다. 목표 입력과 정답의 정렬, 원문 패딩과 목표 패딩의 처리를 따로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
인과 셀프 어텐션과 교차 어텐션의 역할 및 필요한 마스크를 비교하라.

</div>

??? success "연습문제 2 풀이"
    전자는 미래 목표 토큰 접근을 막고 후자는 원문 표현을 읽는다. 교차 어텐션에는 원문 패딩 마스크가 필요하지만 보통 원문 위치의 인과 마스크는 적용하지 않는다. 디코더 전용 모델은 교차 어텐션 없이 프롬프트를 같은 시퀀스 안에서 읽는 다른 구성을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
원문 길이 m, 목표 길이 n인 모델의 명시적 어텐션 웨이트 저장 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"
    배치·헤드·레이어 수를 고정하면 인코더 $O(m^2)$, 디코더 $O(n^2)$, 교차 어텐션 $O(mn)$이다. 합은 $O(m^2+n^2+mn)$이며 토큰 활성·FFN·파라메터·역전파 저장은 별도다. 메모리 효율 커널은 모든 웨이트를 직접 저장하지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 CrossAttention이 기대하는 텐서 축과 패딩 처리의 한계를 설명하라.

</div>

??? success "연습문제 4 풀이"
    nn.MultiheadAttention의 기본 batch_first=False이므로 입력은 (목표 길이,B,d), 메모리는 (원문 길이,B,d)다. 반환값은 출력과 어텐션 웨이트의 튜플이다. 이 래퍼는 원문 패딩 마스크를 받지 않으므로 패딩을 포함한 배치에는 인터페이스 확장이 필요하다.

    ```python
    class CrossAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.mha = nn.MultiheadAttention(d_model, n_heads)
        def forward(self, decoder_hidden, encoder_output):
            # Q는 디코더에서, K와 V는 인코더에서
            return self.mha(decoder_hidden, encoder_output, encoder_output)
    ```

## 정리하며

세 구조는 입력 문맥과 출력 방식이 다르며 학습 목표·마스킹·헤드를 함께 보아야 한다. 예제의 간단한 스택을 BERT·GPT·T5 전체 구현과 구분하고, 상세 구현에서도 패딩·목표 시프트·반환 상태와 생성 모드를 확인한다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5." JMLR.
5. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

---
