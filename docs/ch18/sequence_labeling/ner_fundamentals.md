# 이름 알아보기의 근본
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 이름 알아보기를 정의하고 자연어 물길에서의 몫을 이해한다
- 여러 분야에 걸친 표준 것 갈래를 가려내고 갈래 짓는다
- 이름 알아보기를 수학으로 또렷하게 차례 이름표 붙이기 문제로 세운다
- 이름 알아보기와 다른 자연어 일의 관계를 이해한다
- 것 알아보기에 본디 있는 어려움과 복잡함을 살핀다

## 들어가며

이름 알아보기(NER)는 짜임 없는 글에서 이름 있는 것을 찾아 사람 이름, 조직, 자리, 날짜, 그 밖의 분야별 것 같은 미리 정해 둔 갈래로 나누는, 자연어 다루기의 근본 일이다. 이름 알아보기는 앎 뽑기 물길의 결정적인 조각이며 앎 그래프 세우기, 물음 답하기, 글월 이해 같은 뒤따르는 쓰임새를 가능하게 한다.

## 수학적 정식화

### 차례 이름표 붙이기 얼거리

이름 알아보기는 엄밀히 **차례 이름표 붙이기** 문제로 정의된다. 토막의 들임 차례가 주어질 때:

$$
\mathbf{X} = (x_1, x_2, \ldots, x_n)
$$

여기서 $x_i$마다 토막(낱말이나 아래낱말)을 나타내며, 목표는 그에 맞는 이름표 차례를 어림하는 것이다:

$$
\mathbf{Y} = (y_1, y_2, \ldots, y_n)
$$

where each $y_i \in \mathcal{L}$ belongs to a predefined label set $\mathcal{L}$.

### 가장 좋은 차례 어림

목표는 들임이 주어질 때 가장 그럴듯한 이름표 차례를 찾는 것이다:

$$
\mathbf{Y}^* = \arg\max_{\mathbf{Y}} P(\mathbf{Y} | \mathbf{X})
$$

나타내기 방식에 따라 이 확률을 다르게 쪼갠다:

**따로 갈래 매기기(토막 수준)**:

$$
P(\mathbf{Y} | \mathbf{X}) = \prod_{i=1}^{n} P(y_i | \mathbf{X})
$$

**1차 마르코프(선형 사슬 CRF)**:

$$
P(\mathbf{Y} | \mathbf{X}) = \frac{1}{Z(\mathbf{X})} \prod_{i=1}^{n} \psi(y_{i-1}, y_i, \mathbf{X}, i)
$$

where $Z(\mathbf{X})$ is the partition function ensuring proper normalization.

## 것의 갈래와 갈래 체계

### 표준 CoNLL 것 갈래

CoNLL-2003 공동 과제가 바탕이 되는 것 갈래 체계를 세웠다:

| 것 갈래 | 부호 | 설명 | 보기 |
|-------------|------|-------------|----------|
| 사람 | PER | 사람의 이름 | "Barack Obama", "Marie Curie" |
| 조직 | ORG | 회사, 기관, 단체 | "Apple Inc.", "United Nations" |
| 자리 | LOC | 나라, 도시, 지리적 특징 | "Paris", "Mount Everest" |
| 그 밖 | MISC | 그 밖의 이름 있는 것 | "World Cup", "Nobel Prize" |

### 넓힌 OntoNotes 갈래 체계

OntoNotes 5.0은 것 갈래 18개로 더 결이 고운 가르기를 준다:

| 갈래 묶음 | 것 갈래 |
|----------|--------------|
| 이름 있는 것 | PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE |
| 수로 된 것 | DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL |

### 분야별 것 갈래

분야마다 특화된 것 갈래 체계가 필요하다:

**생의학 이름 알아보기**:

- 유전자/단백질 이름(보기로 "BRCA1", "insulin")
- 병 이름(보기로 "diabetes", "COVID-19")
- 약 이름(보기로 "aspirin", "metformin")
- 화합물(보기로 "H2O", "glucose")

**금융 이름 알아보기**:

- 회사 이름과 종목 기호(보기로 "AAPL", "Goldman Sachs")
- 금융 상품(보기로 "10-year Treasury")
- 돈의 양과 화폐
- 경제 지표(보기로 "GDP", "CPI")

**법률 이름 알아보기**:

- 사건 이름과 인용
- 법률상의 것(당사자, 법원)
- 법령과 규정
- 날짜와 관할

## 것 나타내기

### 글자 수준 구간

것은 글자 어긋남을 갖는 구간으로 나타낸다:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Entity:
    """글자 수준 자리를 갖춘 이름 있는 것을 나타낸다."""
    text: str           # 것의 겉모습
    entity_type: str    # 것의 갈래(PER, ORG, LOC 등)
    start: int          # 시작 글자 어긋남(포함)
    end: int            # 끝 글자 어긋남(제외)
    confidence: Optional[float] = None  # 모델의 믿음도 점수
    
    def __post_init__(self):
        """것의 나타냄 확인하기."""
        assert self.end > self.start, "End must be greater than start"
        assert len(self.text) == self.end - self.start, "Text length must match span"
    
    def overlaps(self, other: 'Entity') -> bool:
        """이 것이 다른 것과 겹치는지 살피기."""
        return not (self.end <= other.start or other.start >= self.end)
    
    def contains(self, other: 'Entity') -> bool:
        """이 것이 다른 것을 온전히 담고 있는지 살피기."""
        return self.start <= other.start and self.end >= other.end
```

### 토막 수준 맞추기

토막낸 글을 다룰 때는 것을 토막 경계에 맞춰야 한다:

```python
from typing import List, Tuple

def align_entities_to_tokens(
    text: str,
    entities: List[Entity],
    token_spans: List[Tuple[int, int]]
) -> List[List[str]]:
    """
    글자 수준의 것을 토막 수준의 이름표에 맞추기.
    
    인수:
        text: Original text string
        entities: 글자 어긋남을 담은 Entity 개체의 목록
        token_spans: 토막마다의 (start, end) 글자 자리 목록
        
    반환값:
        토막마다의 것 이름표 목록(IOB2 꼴)
    """
    labels = ['O'] * len(token_spans)
    
    for entity in sorted(entities, key=lambda e: e.start):
        entity_tokens = []
        
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            # 토막이 것과 겹치는지 살피기
            if tok_start < entity.end and tok_end > entity.start:
                entity_tokens.append(idx)
        
        # IOB2 이름표 매기기
        for i, tok_idx in enumerate(entity_tokens):
            prefix = 'B' if i == 0 else 'I'
            labels[tok_idx] = f"{prefix}-{entity.entity_type}"
    
    return labels
```

## 자연어 물길에서의 이름 알아보기

### 앞손질의 달림

이름 알아보기 체계는 대개 몇 가지 앞손질 단계에 달려 있다:

```
Raw Text → Tokenization → (Optional: POS Tagging) → NER → Downstream Tasks
```

**토막내기의 영향**: 어떤 토막내개를 고르느냐가 이름 알아보기 성능에 크게 영향을 준다:

- 낱말 수준 토막내개는 것의 경계를 지키지만 곳간 밖 낱말에 약하다
- 아래낱말 토막내개(BPE, WordPiece)는 곳간 밖 낱말을 다루지만 것을 쪼갤 수 있다

### 뒤따르는 쓰임새

이름 알아보기는 수많은 뒤따르는 쓰임새를 가능하게 한다:

1. **앎 뽑기**: 짜임 없는 글에서 짜임 있는 자료 뽑기
2. **물음 답하기**: 글월에서 답 후보 가려내기
3. **앎 그래프 세우기**: 그래프의 것 마디 채우기
4. **글월 가르기**: 것의 분포를 특징으로 쓰기
5. **같은 것 가리키기 풀기**: 대이름씨를 이름 있는 것에 잇기
6. **관계 뽑기**: 알아낸 것 사이의 관계 찾기

## 이름 알아보기의 어려움

### 아리송함과 맥락에 달림

겉모습이 같아도 서로 다른 것 갈래를 나타낼 수 있다:

| 글 | 맥락 | 것 갈래 |
|------|---------|-------------|
| "Washington" | "Washington crossed the Delaware" | PERSON |
| "Washington" | "I visited Washington D.C." | LOCATION |
| "Washington" | "The Washington Post reported..." | ORGANIZATION |

### 것의 경계 찾기

것의 경계를 또렷이 정하는 일에는 어려움이 있다:

```
"University of California, Berkeley" → Single ORG entity
"New York City Department of Education" → Single ORG entity  
"Dr. Martin Luther King Jr." → Single PER entity with title
```

### 겹겹이 들고 겹치는 것

어떤 글에는 것이 겹겹이 든 짜임이 있다:

```
"Bank of [America]_LOC"  → Contains nested LOC within ORG
"[Bank of America]_ORG"  → Full organization name

"[New York]_LOC University" → LOC within broader context
"[New York University]_ORG" → Full organization name
```

### 드물고 새로 나타나는 것

이름 알아보기 체계는 다음을 다뤄야 한다:

- **본 적 없는 것**: 익힘 자료에 없는 새 회사, 상품, 사람
- **분야 바뀜**: 특화된 분야(보기로 생의학)의 것
- **때에 따른 흘러감**: 모델을 익힌 뒤 새로 나타나는 것

## PyTorch 짜기: 것 자료 짜임새

```python
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class NERExample:
    """이름 알아보기의 익힘/미룸 보기 하나."""
    tokens: List[str]
    labels: Optional[List[str]] = None
    entities: List[Entity] = field(default_factory=list)
    
    def to_tensor(
        self, 
        token_to_idx: Dict[str, int],
        label_to_idx: Dict[str, int],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """보기를 모델 들임용 텐서로 바꾸기."""
        # 토막 덧대거나 잘라 내기
        token_ids = [token_to_idx.get(t, token_to_idx['<UNK>']) 
                     for t in self.tokens[:max_length]]
        padding_length = max_length - len(token_ids)
        token_ids += [token_to_idx['<PAD>']] * padding_length
        
        # 눈길 마스크 만들기
        attention_mask = [1] * min(len(self.tokens), max_length)
        attention_mask += [0] * padding_length
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }
        
        # 있으면 이름표 더하기(익히기 모드)
        if self.labels is not None:
            label_ids = [label_to_idx.get(l, label_to_idx['O']) 
                        for l in self.labels[:max_length]]
            label_ids += [label_to_idx['O']] * padding_length
            result['labels'] = torch.tensor(label_ids, dtype=torch.long)
        
        return result


class NERDataset(torch.utils.data.Dataset):
    """이름 알아보기용 PyTorch 자료 뭉치."""
    
    def __init__(
        self,
        examples: List[NERExample],
        token_to_idx: Dict[str, int],
        label_to_idx: Dict[str, int],
        max_length: int = 512
    ):
        self.examples = examples
        self.token_to_idx = token_to_idx
        self.label_to_idx = label_to_idx
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx].to_tensor(
            self.token_to_idx,
            self.label_to_idx,
            self.max_length
        )
    
    @staticmethod
    def build_vocabularies(
        examples: List[NERExample]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """보기에서 토막 곳간과 이름표 곳간 세우기."""
        tokens = set(['<PAD>', '<UNK>'])
        labels = set(['O'])
        
        for ex in examples:
            tokens.update(ex.tokens)
            if ex.labels:
                labels.update(ex.labels)
        
        token_to_idx = {t: i for i, t in enumerate(sorted(tokens))}
        label_to_idx = {l: i for i, l in enumerate(sorted(labels))}
        
        return token_to_idx, label_to_idx
```

## 값매김 미리보기

이름 알아보기 값매김에는 서로 채워 주는 잣대가 여럿 쓰인다:

### 토막 수준 잣대
- 토막 어림마다의 정밀도, 재현율, F1
- 벌레잡기에 쓸모 있지만 오해를 부를 수 있다

### 것 수준 잣대(표준)
- **딱 맞음**: 것의 경계와 갈래가 모두 딱 맞아야 한다
- **일부 맞음**: 갈래가 맞고 구간이 겹친다
- **갈래 맞음**: 경계와 상관없이 갈래가 맞다

표준 값매김은 **것 수준 딱 맞음 F1**을 쓴다:

$$
\text{Precision} = \frac{|\text{Predicted} \cap \text{Gold}|}{|\text{Predicted}|}
$$

$$
\text{Recall} = \frac{|\text{Predicted} \cap \text{Gold}|}{|\text{Gold}|}
$$

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

## 요약

이름 알아보기는 다음과 같은 바탕이 되는 자연어 일이다:

1. 것 가려내기를 차례 이름표 붙이기로 **세운다**
2. 것을 분야별 갈래 체계로 **나눈다**
3. 것을 글자나 토막 구간으로 **나타낸다**
4. 뒤따르는 앎 뽑기 일을 **가능하게 한다**
5. 아리송함, 겹겹이 든 것, 분야 맞추기 같은 어려움을 **마주한다**

이어지는 절에서는 이름표 방식, 모델 얼개, 익히기 절차를 자세히 살펴본다.

## 참고 문헌

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL*.

2. Weischedel, R., et al. (2013). OntoNotes Release 5.0. Linguistic Data Consortium.

3. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

4. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.

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
