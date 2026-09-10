# 인코더-디코더 짜임

트랜스포머 구조(Vaswani 외, 2017)는 크게 세 갈래로 다듬어져 왔다. 갈래마다 주의의 방향과 정보의 흐름을 다르게 골라 저마다 다른 강점을 지닌다.

| 구조 | 주의 | 보기 | 잘 맞는 일 |
|--------------|-----------|----------|----------|
| 인코더만 | 양방향 | BERT, RoBERTa | 이해 |
| 디코더만 | 인과 | GPT, LLaMA | 생성 |
| 인코더-디코더 | 둘 다 | T5, BART | 수열 대 수열 |

---

## 1. 본디의 인코더-디코더 구조

본디 트랜스포머는 기계 번역을 위한 인코더-디코더 모형으로 설계되었다. 간추린 변형을 살피기 전에 이 온전한 구조를 이해해 두어야 한다.

### 인코더 더미

인코더는 똑같은 층 $N$개(대개 $N = 6$)로 이루어지고 층마다 아래 층 둘을 담는다.

**아래 층 1 — 다중 머리 자기 주의:**

$$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

모든 자리가 다른 모든 자리에 주의한다(양방향). 가림을 하지 않으므로 토큰마다 원문 맥락 전체에 닿을 수 있다.

**아래 층 2 — 자리별 순전파 신경망:**

$$\text{FFN}(\mathbf{z}) = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

$$\mathbf{H} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$

순전파 신경망은 자리마다 똑같이 따로 적용된다. 자리끼리의 주고받음 없이 토큰별 비선형 능력을 더한다.

### 디코더 더미

디코더도 똑같은 층 $N$개를 가지지만 층마다 아래 층을 **셋** 담는다.

**아래 층 1 — 가린 자기 주의:**

$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHead}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y}))$$

인과 가림이 앞으로의 자리에 주의하지 못하게 막아 자기 회귀 성질을 지킨다.

**아래 층 2 — 교차 주의:**

$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{MultiHead}(\mathbf{Y}', \mathbf{M}, \mathbf{M}))$$

질의는 디코더에서 오고 열쇠와 값은 인코더의 출력 $\mathbf{M}$에서 온다. 이것이 인코더와 디코더를 잇는 다리이다.

**아래 층 3 — 순전파 신경망:**

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

### 잔차 연결과 층 정규화

트랜스포머의 아래 층마다 잔차 연결을 쓴 뒤 층 정규화를 한다. 이 설계 선택은 학습의 안정성에 매우 중요하다.

$$\text{output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

**뒤 정규화**(본디 논문): 위에서 보인 대로 잔차를 더한 뒤 층 정규화를 한다.

**앞 정규화**(요즘 방식): 아래 층에 들어가기 전에 층 정규화를 한다. 깊은 모형에서 더 안정적임이 밝혀졌다.

$$\text{output} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$$

앞 정규화는 기울기가 정규화를 거치지 않고 잔차 길로 흐르게 하여 훨씬 깊은 신경망을 학습할 수 있게 한다. 요즘 트랜스포머(GPT-2 이후, LLaMA, T5 v1.1) 대부분이 앞 정규화를 쓴다.

---

## 2. 인코더만 (BERT 방식)

**주의**: 모든 자리가 모든 자리를 본다(양방향)

인코더만 쓰는 구조는 디코더를 아주 없앤다. 모형은 입력을 양방향 자기 주의 층 더미로 처리하여 토큰마다 맥락이 담긴 표현을 낸다.

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

**쓰임새**: 분류, 개체명 인식, 질의응답 추출, 문장 임베딩

**사전 학습**: 가린 언어 모형화(MLM). 토큰을 무작위로 가리고 양방향 맥락에서 그것을 맞힌다. 생성을 위한 목표가 아니다. 모형은 생성이 아니라 이해를 배운다.

---

## 3. 디코더만 (GPT 방식)

**주의**: 인과 (자리마다 지난 것만 본다)

디코더만 쓰는 구조는 인코더와 교차 주의 층을 아주 없앤다. 남는 것은 순전파 아래 층을 곁들인, 인과로 가린 자기 주의 층의 더미이다.

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

**사전 학습**: 인과 언어 모형화(CLM). 앞선 토큰을 모두 주고 다음 토큰을 맞힌다. 학습 수열의 토큰마다 학습 신호를 주므로 데이터를 매우 알뜰하게 쓴다.

**이름에 관한 참고**: `nn.TransformerEncoder`를 쓰지만 인과 가림 때문에 기능으로는 디코더이다. 파이토치의 `TransformerEncoderLayer`는 그저 자기 주의와 순전파 블록일 뿐이고, 인과 가림을 더하면 디코더 층이 된다.

---

## 4. 인코더-디코더 (T5 방식)

**주의**: 인코더는 양방향, 디코더는 인과, 그리고 교차 주의

인코더-디코더 구조는 본디 트랜스포머 설계를 온전히 지닌다. 인코더가 입력을 양방향으로 처리하고, 디코더는 인코더의 표현에 주의하면서 출력을 자기 회귀로 만들어 낸다.

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

**쓰임새**: 번역, 요약, 글에서 글로 옮기는 과제

**사전 학습**: 구간 망가뜨리기(T5), 잡음 없애기(BART). 입력을 망가뜨린 뒤 되살린다. 인코더는 망가진 입력을 양방향으로 보고, 디코더는 본디 글이나 망가진 구간을 만들어 낸다.

---

## 5. 자세히 견주기

| 측면 | 인코더만 | 디코더만 | 인코더-디코더 |
|--------|-------------|-------------|---------|
| 맥락 | 양방향 | 인과 | 둘 다 |
| 생성 | ✗ | ✓ | ✓ |
| 이해 | ✓✓ | ✓ | ✓ |
| 교차 주의 | ✗ | ✗ | ✓ |
| 매개변수 | 1배 | 1배 | 약 2배 |
| KV 캐시 (추론) | 해당 없음 | 자기 주의만 | 자기 주의와 교차 주의 |
| 요즘의 인기 | 보통 | **높음** | 보통 |

### 핵심 맞바꿈

**양방향 맥락과 인과 맥락**: 인코더만 쓰는 모형은 토큰마다 표현을 셈할 때 수열 전체를 보므로 이해 과제에 더 넉넉한 맥락을 준다. 디코더만 쓰는 모형은 왼쪽 맥락만 보는데, 생성에는 자연스럽지만 이해 과제는 한 방향 표현에 기대야 한다는 뜻이다. 인코더-디코더 모형은 둘의 좋은 점을 다 가진다. 원문은 양방향으로 인코딩하고 목표문은 인과로 디코딩한다.

**매개변수 효율**: 인코더-디코더 모형은 더미를 둘 따로 담으므로 같은 깊이와 너비에서 인코더만이나 디코더만 쓰는 모형의 대략 두 배의 매개변수를 가진다. 그러나 디코더만 쓰는 모형은 (층을 더하기만 하면 되니) 더 간단히 키울 수 있어 큰 규모 모형의 주류가 되었다.

**학습 목표의 맞춤**: 디코더만 쓰는 모형은 인과 언어 모형화로 학습하는데 이는 생성과 곧바로 들어맞는다. 인코더만 쓰는 모형은 가린 언어 모형화를 쓰는데 이는 이해와 들어맞는다. 인코더-디코더 모형은 수열 대 수열 과제와 들어맞는 구간 망가뜨리기 같은 목표를 쓴다.

**추론 효율**: 디코더만 쓰는 모형은 생성에 따라 커지는 KV 캐시 하나만 있으면 되어 이롭다. 인코더-디코더 모형은 캐시가 둘(커지는 디코더 자기 주의 KV 캐시와 고정된 교차 주의 KV 캐시) 필요하지만, 교차 주의 KV 캐시는 인코더에서 한 번 셈해 두고 다시 쓴다.

### 순전파 신경망의 몫

세 구조 모두에서 순전파 아래 층은 같은 몫을 한다. 토큰별 비선형 변환이다. 연구에 따르면 순전파 층은 열쇠-값 기억 노릇을 하며 사실 지식을 담아 둔다.

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$$

여기서 $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$은 "열쇠"(무늬 탐지기)를 셈하고 $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$은 "값"(딸린 정보)을 담아 둔다. 안쪽 차원 $d_{ff}$은 대개 $4d$이어서 순전파 신경망은 층마다 주의 얼개의 네 배의 매개변수를 가진다.

---

## 6. 역사의 흐름

이 분야는 인코더만 쓰는 모형이 판치던 때(BERT 시대, 2018~2020)에서 디코더만 쓰는 모형이 판치는 때(GPT-3 이후, 2020~지금)로 옮겨 왔다.

- **2017**: 본디 트랜스포머가 번역에 인코더-디코더를 쓴다
- **2018**: BERT가 이해 성능 시험에서 인코더만 쓰는 쪽의 우위를 보인다
- **2018**: GPT-1이 디코더만 쓰는 모형을 갖가지 과제에 맞추어 미세 조정할 수 있음을 보인다
- **2019**: GPT-2가 디코더만 쓰는 모형이 앞뒤가 맞는 글을 지을 수 있음을 보인다
- **2020**: T5가 모든 자연어 처리 과제에 걸친 인코더-디코더의 쓰임새 넓음을 보인다
- **2020 이후**: GPT-3과 뒤이은 큰 디코더 전용 모형이 창발하는 소수 예시 학습 능력을 보이며 분야를 디코더만 쓰는 구조로 옮겨 놓는다
- **2023 이후**: 디코더만 쓰는 모형(LLaMA, Mistral, GPT-4)이 큰 규모에서 생성과 이해 과제를 모두 휘어잡는다

### 디코더만 쓰는 쪽이 이긴 까닭

여러 요인이 디코더만 쓰는 쪽으로의 수렴을 이끌었다.

1. **간결함**: 구조 하나, 목표 하나, 키우는 방법 하나
2. **맥락 안 학습**: 규모가 커지면 창발하여 미세 조정을 선택 사항으로 만든다
3. **하나로 모은 접점**: 다음 토큰 맞히기로 이해와 생성을 모두 한다
4. **키우는 효율**: 모든 매개변수가 모든 예측에 이바지한다(생성 중에 놀고 있는 별도의 인코더가 없다)

---

## 7. 언제 쓰는가

- **인코더만**: 분류, 임베딩, 추출 과제
- **디코더만**: 생성, 대화, 소수 예시 학습 (요즘 가장 인기 있다)
- **인코더-디코더**: 번역, 원문과 목표문이 뚜렷이 구분되는 요약

---

## 8. 인코더 블록 깊이 들여다보기

##### 훑어보기

트랜스포머 인코더 블록은 BERT 같은 인코더 기반 구조의 근본 되는 짜임 단위이다. 블록마다 자기 주의와 순전파 층으로 입력을 바꾸어, 수열의 길이는 지키면서 표현을 다듬는다.

##### 구조

인코더 블록마다 잔차 연결과 층 정규화를 갖춘 아래 층 둘로 이루어진다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

##### 부품 나누어 보기

1. **다중 머리 자기 주의**: 모든 자리 사이의 의존을 잡아낸다
2. **순전파 신경망**: 넓혔다 줄이는 자리별 변환
3. **잔차 연결**: 깊은 신경망에서 기울기가 흐르게 한다
4. **층 정규화**: 학습을 안정되게 한다

##### 인코더의 다중 머리 자기 주의

인코더는 자리마다 모든 자리에 주의할 수 있는 양방향 자기 주의를 쓴다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

자기 주의에서는 $Q$, $K$, $V$이 모두 같은 입력에서 나온다.

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

##### 입력에서 주의까지: 한 걸음씩

날것의 문장이 인코더를 어떻게 흘러가는지 이해하려면 변환을 하나씩 따라가야 한다.

**1단계 — 토큰 나누기**: 입력 문장을 토큰(낱말이나 부분 낱말)으로 나눈다. 이를테면 "The cat sat"은 토큰 수열 $[\text{The}, \text{cat}, \text{sat}]$이 된다.

**2단계 — 임베딩 찾기**: 임베딩 행렬 $E \in \mathbb{R}^{V \times d_{\text{model}}}$으로 토큰마다 빽빽한 벡터를 잇대어, $n$이 수열의 길이일 때 행렬 $X_{\text{embed}} \in \mathbb{R}^{n \times d_{\text{model}}}$을 만든다. 임베딩에는 $\sqrt{d_{\text{model}}}$을 곱한다.

**3단계 — 위치 인코딩**: 사인파 위치 인코딩을 성분별로 더한다.

$$
X = X_{\text{embed}} \cdot \sqrt{d_{\text{model}}} + \text{PE}
$$

그 결과 $X \in \mathbb{R}^{n \times d_{\text{model}}}$은 (임베딩에서 온) 뜻과 자리 정보를 함께 지닌다.

**4단계 — Q, K, V로의 선형 사영**: 서로 다른 가중치 행렬 셋이 $X$을 질의, 열쇠, 값 공간으로 사영한다.

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

여기서 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$이다. 이 사영은 (표준 정식화에서 편향 항이 없는) 선형이며, 무엇을 찾을지(질의), 무엇을 내세울지(열쇠), 어떤 정보를 줄지(값)에 대해 모형이 저마다 다른 표현을 배우게 한다.

**5단계 — 여러 머리로 나누기**: $Q$, $K$, $V$ 각각을 마지막 차원을 따라 머리 $h$개로 나누며 머리마다 차원은 $d_k = d_{\text{model}} / h$이다.

$$
Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k} \quad \text{for } i = 1, \ldots, h
$$

**6단계 — 머리마다 스케일 조정 내적 주의**: 머리마다 따로 주의를 셈한다.

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

$\sqrt{d_k}$으로 나누는 것은 내적의 크기가 너무 커져 소프트맥스가 기울기가 매우 작은 영역으로 밀려나는 일을 막는다.

**7단계 — 이어 붙이고 사영하기**: 머리들의 출력을 이어 붙이고 선형으로 사영한다.

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

여기서 $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$이다(역시 표준 정식화에서 편향 항이 없다). 출력은 입력과 같은 꼴 $\mathbb{R}^{n \times d_{\text{model}}}$이다.

!!! note "주의에는 편향 항이 없다"
    본디 트랜스포머 정식화는 $W^Q$, $W^K$, $W^V$, $W^O$ 사영에 편향 항을 쓰지 않는다. 주의 얼개에는 스케일 조정($\sqrt{d_k}$)과 정규화(소프트맥스)가 들어 있어 신호를 넉넉히 다스리므로 이 단계에서 더하는 편향이 필요하지 않다. 어떤 구현(예: 파이토치의 `nn.MultiheadAttention`)은 기본으로 편향을 두지만 `bias=False`로 하면 본디 설계와 같아진다.

##### 다중 머리 정식화

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

여기서 머리마다 다음과 같다.

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

##### 순전파 신경망

자리별 순전파 신경망은 자리마다 따로 같은 변환을 적용한다.

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

또는 (요즘 모형에서 흔한) GELU 활성으로 다음과 같이 한다.

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

##### 차원 넓히기

순전파 신경망은 대개 숨은 차원을 4배로 넓힌다.

- 입력: $d_{\text{model}}$
- 숨은 층: $d_{ff} = 4 \times d_{\text{model}}$
- 출력: $d_{\text{model}}$

##### 앞 정규화와 뒤 정규화

##### 뒤 정규화 (본디 트랜스포머)

$$
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))
$$

##### 앞 정규화 (요즘의 표준)

$$
\mathbf{X}' = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))
$$

앞 정규화는 깊은 모형을 학습할 때 더 안정적이고 학습률 예열이 필요 없다.

##### 파이토치 구현

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
            bias: 선형 사영에 편향을 둘지 여부
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
        앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 토큰을 위한 가림 [batch_size, seq_len]
                가릴 자리(채움)가 True
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
        반환값:
            output: 바뀐 텐서 [batch_size, seq_len, d_model]
            attention_weights: 선택으로 돌려주는 주의 가중치 [batch_size, num_heads, seq_len, seq_len]
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
    자리별 순전파 신경망.
    
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
            activation: 활성 함수('relu' 또는 'gelu')
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
        앞먹임.
        
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
    1. 다중 머리 자기 주의 + 잔차 + 층 정규화
    2. 자리별 순전파 + 잔차 + 층 정규화
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
            activation: 순전파의 활성 함수
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
        
        # 순방향 신경망
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # 층 정규화
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
        앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 가림 [batch_size, seq_len]
            return_attention: 어텐션 가중치를 돌려줄지 여부
            
        반환값:
            output: 바뀐 텐서 [batch_size, seq_len, d_model]
            attention_weights: 선택으로 돌려주는 주의 가중치
        """
        if self.pre_norm:
            # 앞 정규화: 아래 층 앞에서 층 정규화
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
            # 뒤 정규화: 아래 층 뒤에서 층 정규화
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
    4. 선택으로 두는 마지막 층 정규화(앞 정규화를 위해)
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
            activation: 순전파의 활성 함수
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
        
        # 마지막 층 정규화 (앞 정규화 구조를 위해)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치를 초기화한다."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> dict:
        """
        앞먹임.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 주의 가림 [batch_size, seq_len]
                실제 토큰은 1, 채움은 0
            return_all_hidden_states: 모든 층의 출력을 돌려줄지 여부
            
        반환값:
            다음을 담은 사전:
                - last_hidden_state: 마지막 인코더 출력
                - all_hidden_states: 선택으로 돌려주는 모든 층의 출력 목록
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
        
        # 인코더 층을 통과시킨다
        all_hidden_states = [x] if return_all_hidden_states else None
        
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask)
            if return_all_hidden_states:
                all_hidden_states.append(x)
        
        # 마지막 층 정규화
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
    
    # 부호기 만들기
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
    
    # 매개변수 개수 세기
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

##### 기울기 흐름 분석

##### 잔차 연결의 이점

잔차가 없으면 기울기가 아래 층을 모두 거쳐 흘러야 한다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \prod_{l=0}^{L-1} \frac{\partial \text{SubLayer}^{(l)}}{\partial \mathbf{x}^{(l)}}
$$

잔차가 있으면 기울기가 곧바로 가는 길이 생긴다.

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \text{other terms}
$$

##### 층 정규화의 효과

층 정규화는 특징 차원에 걸쳐 정규화한다.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$

이는 층에 걸친 기울기의 크기를 안정되게 한다.

##### 계산 복잡도

##### 층마다의 복잡도

수열 길이 $n$, 모형 차원 $d$, 순전파 차원 $d_{ff}$에 대해 다음과 같다.

| 부품 | 시간 복잡도 | 공간 복잡도 |
|-----------|-----------------|------------------|
| 자기 주의 | $O(n^2 d)$ | $O(n^2 + nd)$ |
| 순전파 | $O(n d \cdot d_{ff})$ | $O(n d_{ff})$ |
| 층 정규화 | $O(nd)$ | $O(nd)$ |

##### 인코더 전체의 복잡도

층이 $L$개일 때 다음과 같다.

$$
\text{Time: } O(L \cdot (n^2 d + n d \cdot d_{ff}))
$$

흔한 $d_{ff} = 4d$이면 다음과 같다.

$$
\text{Time: } O(L \cdot n \cdot d \cdot (n + 4d))
$$

##### 간추림

트랜스포머 인코더 블록은 모듈 방식의 강력한 짜임 단위로 다음을 한다.

1. **전역 의존을 잡아낸다**: 자기 주의가 모든 자리를 잇는다
2. **비선형 변환을 적용한다**: 순전파 신경망이 표현 능력을 더한다
3. **기울기 흐름을 지킨다**: 잔차가 깊은 구조를 가능케 한다
4. **학습을 안정되게 한다**: 층 정규화가 활성을 다스린다

인코더 블록을 이해하는 것은 BERT나 RoBERTa 같은 모형과 인코더 기반 구조를 구현하는 데 꼭 필요하다.

##### 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

---

## 9. 디코더 블록 깊이 들여다보기

##### 훑어보기

트랜스포머 디코더 블록은 GPT 같은 자기 회귀 모형의 짜임 단위이다. 인코더와 달리 디코더는 **인과(가린) 자기 주의**를 써서 자리가 앞으로의 자리에 주의하지 못하게 막아 왼쪽에서 오른쪽으로의 생성을 가능케 한다.

##### 구조의 변형

##### 온전한 인코더-디코더 (본디 트랜스포머)

본디 디코더는 아래 층이 셋이다.

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{CrossAttn}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

##### 디코더만 (GPT 방식)

요즘 언어 모형은 아래 층이 둘인 디코더 전용 구조를 쓴다.

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MaskedSelfAttn}(\mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

##### 인과 자기 주의

인코더 자기 주의와의 핵심 차이는 앞으로의 자리에 주의하지 못하게 막는 **인과 가림**이다.

##### 인과 가림

수열 길이가 $n$일 때 인과 가림 $M$은 다음과 같다.

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

이는 아래 삼각 꼴의 주의 무늬를 만든다.

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

##### 눈으로 보기

```
Position can attend to:
     1  2  3  4  5
1  [ ✓  ✗  ✗  ✗  ✗ ]
2  [ ✓  ✓  ✗  ✗  ✗ ]
3  [ ✓  ✓  ✓  ✗  ✗ ]
4  [ ✓  ✓  ✓  ✓  ✗ ]
5  [ ✓  ✓  ✓  ✓  ✓ ]
```

##### 교차 주의 (인코더-디코더)

수열 대 수열 모형에서 디코더는 인코더의 출력에 주의한다.

$$
\text{CrossAttn}(Z, X_{\text{enc}}) = \text{softmax}\left(\frac{Z W^Q (X_{\text{enc}} W^K)^T}{\sqrt{d_k}}\right) X_{\text{enc}} W^V
$$

여기서:

- 질의: 디코더에서 ($Z W^Q$)
- 열쇠와 값: 인코더에서 ($X_{\text{enc}} W^K$, $X_{\text{enc}} W^V$)

##### 파이토치 구현

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
        인과 가림을 쓰는 앞먹임.
        
        인수:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            key_padding_mask: 채움 가림 [batch_size, seq_len]
            return_attention: 어텐션 가중치를 돌려줄지 여부
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
    1. 가린 자기 주의 + 잔차 + 층 정규화
    2. 순전파 신경망 + 잔차 + 층 정규화
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
        
        # 순방향 신경망
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
        """앞먹임."""
        if self.pre_norm:
            # 앞 정규화: 아래 층 앞에서 층 정규화
            residual = x
            x = self.norm1(x)
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = residual + self.dropout(attn_out)
            
            residual = x
            x = self.norm2(x)
            ff_out = self.feed_forward(x)
            x = residual + ff_out
        else:
            # 뒤 정규화: 아래 층 뒤에서 층 정규화
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
        
        # 마지막 층 정규화 (앞 정규화를 위해)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        # 출력 사영 (임베딩과 가중치 묶기)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치를 초기화한다."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        앞먹임.
        
        인수:
            input_ids: 토큰 번호 [batch_size, seq_len]
            attention_mask: 주의 가림 [batch_size, seq_len]
            labels: 언어 모형 손실을 위한 이름표 [batch_size, seq_len]
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
        
        # 디코더 층을 통과시킨다
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        # 마지막 정규화
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 언어 모형 머리
        logits = self.lm_head(x)
        
        # 이름표가 있으면 손실을 셈한다
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
            
            # 상위 p(핵) 거르기를 적용한다
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
    
    # 매개변수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

##### 효율적인 추론을 위한 KV 캐시

생성 중에 앞선 토큰의 주의를 다시 셈하는 것은 아깝다. KV 캐싱은 앞 단계의 열쇠와 값 텐서를 담아 둔다.

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
        KV 캐시를 선택으로 쓰는 앞먹임.
        
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

##### 계산 분석

##### KV 캐시가 없을 때

맥락이 $C$일 때 토큰 $T$개를 만드는 데 다음과 같다.

$$
\text{Complexity} = O(T \cdot (C + T)^2 \cdot d)
$$

##### KV 캐시가 있을 때

$$
\text{Complexity} = O(T \cdot (C + T) \cdot d)
$$

이는 생성 길이에 대한 복잡도를 이차에서 일차로 줄인다.

##### 디코더와 인코더 견주기

| 측면 | 인코더 | 디코더 |
|--------|---------|---------|
| 주의의 종류 | 양방향 | 인과 (가린) |
| 앞을 볼 수 있는가 | 그렇다 | 아니다 |
| 주된 쓰임 | 이해 | 생성 |
| 보기 | BERT, RoBERTa | GPT, LLaMA |

##### 간추림

트랜스포머 디코더 블록은 다음으로 자기 회귀 생성을 가능케 한다.

1. **인과 가림**: 앞으로의 자리에 주의하지 못하게 막는다
2. **교차 주의**: 수열 대 수열 모형에서 인코더와 잇는다
3. **KV 캐싱**: 효율적인 점진 생성을 가능케 한다
4. **앞 정규화 구조**: 학습의 안정성을 높인다

##### 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

---

## 연습문제

**연습문제 1.**
기계 번역을 위한 인코더-디코더 트랜스포머의 정보 흐름을 설명하라.

??? success "연습문제 1 풀이"
    인코더는 원문 문장을 처리하며 층마다 자기 주의와 순전파를 적용해 원문 토큰마다 맥락 표현을 낸다. 디코더는 목표 토큰을 자기 회귀로 만들며 층마다 (1) 만들어진 토큰에 대한 가린 자기 주의, (2) 인코더 출력에 대한 교차 주의, (3) 순전파를 적용한다. 교차 주의 덕분에 목표문의 자리마다 원문의 모든 자리에 주의할 수 있다.

---

**연습문제 2.**
디코더에 가린 자기 주의와 교차 주의가 둘 다 필요한 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    가린 자기 주의는 목표문의 자리마다 앞선 목표 자리에 주의하게 해 (자기 회귀 성질을 지킨다) 준다. 교차 주의는 목표문의 자리마다 원문의 모든 자리에 주의하게 해 (입력 정보에 닿게) 준다. 교차 주의가 없으면 디코더가 입력에 닿을 수 없고, 가린 자기 주의가 없으면 디코더가 앞으로의 목표 토큰을 보아 '속임수'를 쓰게 된다.

---

**연습문제 3.**
원문 길이가 $m$이고 목표문 길이가 $n$일 때 인코더-디코더 모형의 기억 복잡도를 셈하라.

??? success "연습문제 3 풀이"
    인코더 자기 주의는 $O(m^2)$, 디코더 자기 주의는 $O(n^2)$, 교차 주의는 $O(mn)$이다. 층마다 전체는 $O(m^2 + n^2 + mn)$이다. 교차 주의가 원문과 목표문의 길이를 엮는 $O(mn)$ 비용을 더한다.

---

**연습문제 4.**
질의는 디코더에서, 열쇠와 값은 인코더에서 오는 교차 주의 얼개를 구현하라.

??? success "연습문제 4 풀이"
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

이 마당은 본디의 인코더-디코더 구조、인코더만 (BERT 방식)、디코더만 (GPT 방식)、인코더-디코더 (T5 방식)을 차례로 짚었다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5." JMLR.
5. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

---
