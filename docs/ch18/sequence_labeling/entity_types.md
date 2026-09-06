# 것의 갈래와 갈래 체계
## 학습 목표

- 분야를 넘나드는 표준 이름 알아보기 갈래 체계를 이해한다
- CoNLL, OntoNotes, 분야별 갈래 체계를 견준다
- 금융을 비롯한 새 분야의 것 갈래 체계를 꾸민다
- 층진 갈래와 결이 고운 갈래 매기기를 다룬다

---

## 표준 갈래 체계

### CoNLL-2003(갈래 4개)

바탕이 되는 잣대 갈래 체계:

| 갈래 | 부호 | 보기 |
|------|------|----------|
| 사람 | PER | "Barack Obama", "Marie Curie" |
| 조직 | ORG | "Apple Inc.", "United Nations" |
| 자리 | LOC | "Paris", "Mount Everest" |
| 그 밖 | MISC | "World Cup", "Nobel Prize" |

### OntoNotes 5.0(갈래 18개)

이름 있는 것, 수, 때를 갈라 놓은 더 풍부한 갈래 체계:

**이름 있는 것**: PERSON, NORP(국적/종교/정치 집단), FAC(시설), ORG, GPE(지정학적 것), LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE

**수로 된 것**: DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

### ACE(자동 내용 뽑기)

아래 갈래를 갖는 것 갈래 일곱: PER, ORG, GPE, LOC, FAC, WEA(무기), VEH(탈것). 갈래마다 아래 갈래가 더 있다(보기로 PER.Individual, PER.Group).

---

## 분야별 갈래 체계

### 생의학 이름 알아보기

| 갈래 | 보기 | 자료 뭉치 |
|------|----------|----------|
| 유전자/단백질 | BRCA1, p53, insulin | JNLPBA, BioCreative |
| 병 | diabetes, COVID-19 | NCBI Disease |
| 약/화학물질 | aspirin, metformin | BC5CDR |
| 종 | *E. coli*, human | LINNAEUS |
| 세포 갈래 | T-cell, neuron | JNLPBA |

### 금융 이름 알아보기

| 갈래 | 보기 | 쓰임새 |
|------|----------|----------|
| 회사 | "Goldman Sachs", "AAPL" | 종목 기호에 잇기 |
| 금융 상품 | "10-year Treasury", "S&P 500" | 자산 꾸러미 살피기 |
| Monetary Amount | "\$2.3 billion", "€50M" | Earnings extraction |
| 경제 지표 | "GDP", "CPI", "unemployment rate" | 거시 살피기 |
| 날짜/기간 | "Q3 2024", "fiscal year" | 때 맞추기 |
| 규제 기관 | "SEC", "Fed", "ECB" | 규정 지킴 살피기 |

### 법률 이름 알아보기

사건 이름, 법원, 법령, 관할, 당사자, 판사, 법률 인용.

---

## 층진 것 갈래 매기기

결이 고운 것 갈래 매기기는 것을 갈래 층위의 마디에 배정한다:

```
Entity
├── Person
│   ├── Politician
│   ├── Athlete
│   ├── Scientist
│   └── Artist
├── Organization
│   ├── Company
│   │   ├── Tech Company
│   │   └── Financial Institution
│   ├── Government Agency
│   └── Educational Institution
└── Location
    ├── Country
    ├── City
    └── Natural Feature
```

### 수식으로 나타내기

Given entity mention $m$ with context $c$, predict a set of types $\mathcal{T}_m \subseteq \mathcal{T}$:

$$P(\mathcal{T}_m | m, c) = \prod_{t \in \mathcal{T}} P(t \in \mathcal{T}_m | m, c)$$

Subject to hierarchical consistency: if $t \in \mathcal{T}_m$ and $t'$ is an ancestor of $t$, then $t' \in \mathcal{T}_m$.

---

## 맞춤 갈래 체계 꾸미기

### 지침

1. **서로 겹치지 않음**: 같은 층위에서 갈래가 겹치는 것을 가장 작게 한다
2. **덮음**: 관심 있는 것이 모두 적어도 한 갈래에 대응되게 한다
3. **결의 균형**: 너무 고우면 표시하는 값이 커지고, 너무 거칠면 쓸모가 줄어든다
4. **뒤따르는 일과의 맞춤**: 목표 쓰임새에 도움이 되는 갈래를 꾸민다
5. **표시 가능함**: 사람이 갈래를 가려낼 수 있어야 한다

### 표시하는 사람들 사이의 일치

Measure taxonomy quality via Cohen's $\kappa$ or Fleiss' $\kappa$ on pilot annotations:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement and $p_e$ is expected agreement by chance. Target $\kappa > 0.8$ for reliable annotation.

---

## 요약

1. **CoNLL-2003**은 이름 알아보기의 표준 4갈래 잣대를 준다
2. **OntoNotes**는 수와 때로 된 것을 넣어 18갈래로 넓힌다
3. **분야별 갈래 체계**는 생의학, 금융, 법률 자연어 다루기에 꼭 필요하다
4. **층진 갈래 매기기**는 결이 고운 것 가르기를 가능하게 한다
5. **갈래 체계 꾸미기**는 결, 덮음, 표시 가능함의 균형을 잡아야 한다

---

## 참고 문헌

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). CoNLL-2003 Shared Task. *CoNLL*.
2. Weischedel, R., et al. (2013). OntoNotes Release 5.0. LDC.
3. Ling, X., & Weld, D. S. (2012). Fine-Grained Entity Recognition. *AAAI*.
4. Alvarado, J., et al. (2015). Domain-Specific Named Entity Recognition in Financial Texts. *ACL FinNLP Workshop*.

## 연습문제

**연습문제 1.**
BIO 이름표 방식을 설명하여라. 월 "Barack Obama visited New York City"에 이름표를 어떻게 붙이겠는가?

??? success "연습문제 1 풀이"
    BIO 이름표에서 **B-X**는 갈래 X인 것의 시작을, **I-X**는 것의 안쪽(이어짐)을, **O**는 어느 것에도 들지 않는 토막을 나타낸다.

    | 토막 | 이름표 |
    |-------|-----|
    | Barack | B-PER |
    | Obama | I-PER |
    | visited | O |
    | New | B-LOC |
    | York | I-LOC |
    | City | I-LOC |

    B 이름표는 같은 갈래의 것이 잇달아 나올 때(보기로 "Obama Trump"를 서로 다른 PER 둘로) 가리는 데 꼭 필요하다.

---

**연습문제 2.**
차례 이름표 붙이기에서 자리마다 따로 소프트맥스로 갈래를 매기는 대신 두 방향 LSTM 위에 CRF 층을 얹는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    Independent softmax treats each position's label independently, ignoring transition constraints. For example, it might predict I-PER following O, which is invalid in BIO tagging. A CRF layer models the joint probability of the entire label sequence, learning a transition matrix $A_{ij}$ that captures which tag transitions are valid. The CRF score for a sequence is $s(x, y) = \sum_t (E_{y_t, t} + A_{y_t, y_{t+1}})$, where $E$ is the emission score from BiLSTM. This ensures globally consistent predictions via Viterbi decoding.

---

**연습문제 3.**
것 수준에서의 이름 알아보기 값매김에 쓰는 정밀도, 재현율, F1 점수를 설명하여라. 것 수준 값매김이 토막 수준보다 왜 더 빡빡한가?

??? success "연습문제 3 풀이"
    **Entity-level** evaluation requires both the entity boundary and type to be exactly correct. Precision = (correctly predicted entities) / (total predicted entities). Recall = (correctly predicted entities) / (total gold entities). $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$. This is stricter than token-level because: a prediction "New York" when the gold entity is "New York City" gets partial credit at the token level (2/3 tokens correct) but zero credit at the entity level (boundary mismatch). Entity-level metrics better reflect real-world utility.

---

**연습문제 4.**
두 방향 LSTM은 양쪽 방향의 맥락을 어떻게 담아내는가? 자리 $t$에서의 숨은 상태 셈을 적어라.

??? success "연습문제 4 풀이"
    A BiLSTM consists of a forward LSTM processing $x_1, \ldots, x_T$ and a backward LSTM processing $x_T, \ldots, x_1$:

    $$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

    The final representation at position $t$ is the concatenation $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$, which captures both left context (through $\overrightarrow{h}_t$) and right context (through $\overleftarrow{h}_t$). This is critical for NER since entity recognition often depends on surrounding words in both directions.
