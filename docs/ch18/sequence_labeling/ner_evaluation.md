# 이름 알아보기 값매김 잣대
## 학습 목표

- 토막 수준 값매김과 것 수준 값매김을 가린다
- 딱 맞음 값매김과 일부 맞음 값매김을 짠다
- 미시, 거시, 무게를 준 F1 점수를 셈한다
- 표준 값매김에 seqeval 라이브러리를 쓴다

## 것 수준 값매김(표준)

이름 알아보기 체계는 **것 수준**에서 값매김한다. 곧 경계와 갈래가 모두 딱 맞아야 한다.

### 고갱이 잣대

For predicted entities $\hat{E}$ and gold entities $E^*$:

$$
\text{Precision} = \frac{|\hat{E} \cap E^*|}{|\hat{E}|}, \quad
\text{Recall} = \frac{|\hat{E} \cap E^*|}{|E^*|}, \quad
F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

## PyTorch 구현

```python
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class Entity:
    entity_type: str
    start: int
    end: int

def extract_entities(tags: List[str]) -> Set[Entity]:
    """IOB2 이름표에서 것 뽑기."""
    entities = set()
    current = None
    
    for i, tag in enumerate(tags):
        if tag == 'O':
            if current:
                entities.add(Entity(current[0], current[1], i))
                current = None
        elif tag.startswith('B-'):
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = (tag[2:], i)
        elif tag.startswith('I-') and current and current[0] == tag[2:]:
            pass  # 것 이어 가기
        else:
            if current:
                entities.add(Entity(current[0], current[1], i))
            current = None
    
    if current:
        entities.add(Entity(current[0], current[1], len(tags)))
    return entities

def compute_ner_metrics(
    pred_tags: List[List[str]],
    gold_tags: List[List[str]],
    average: str = 'micro'
) -> Dict[str, float]:
    """이름 알아보기 F1 잣대 셈하기."""
    tp_per_type = defaultdict(int)
    fp_per_type = defaultdict(int)
    fn_per_type = defaultdict(int)
    
    for pred, gold in zip(pred_tags, gold_tags):
        pred_ents = extract_entities(pred)
        gold_ents = extract_entities(gold)
        
        for e in pred_ents:
            if e in gold_ents:
                tp_per_type[e.entity_type] += 1
            else:
                fp_per_type[e.entity_type] += 1
        
        for e in gold_ents:
            if e not in pred_ents:
                fn_per_type[e.entity_type] += 1
    
    if average == 'micro':
        tp = sum(tp_per_type.values())
        fp = sum(fp_per_type.values())
        fn = sum(fn_per_type.values())
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        return {'precision': p, 'recall': r, 'f1': f1}
    
    # 거시 평균
    all_types = set(tp_per_type) | set(fp_per_type) | set(fn_per_type)
    f1s = []
    for t in all_types:
        tp, fp, fn = tp_per_type[t], fp_per_type[t], fn_per_type[t]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
    
    return {'f1': sum(f1s) / len(f1s) if f1s else 0}
```

## seqeval 라이브러리 쓰기

```python
from seqeval.metrics import classification_report, f1_score

# 표준 값매김
y_true = [['B-PER', 'I-PER', 'O', 'B-LOC']]
y_pred = [['B-PER', 'I-PER', 'O', 'B-LOC']]

print(classification_report(y_true, y_pred))
print(f"F1: {f1_score(y_true, y_pred):.4f}")
```

## 맞음 갈래 간추림

| 갈래 | 경계 | 것 갈래 | 쓰임새 |
|------|------------|-------------|----------|
| 딱 맞음(빡빡) | 맞아야 함 | 맞아야 함 | 표준 잣대 |
| 일부 맞음 | 겹쳐야 함 | 맞아야 함 | 너그러운 값매김 |
| 갈래만 | 아무래도 됨 | 맞아야 함 | 갈래 살피기 |

## 핵심 요점

1. **것 수준 딱 맞음**이 이름 알아보기 값매김의 표준이다
2. **미시 F1**은 것의 잦기로 무게를 준다(CoNLL의 표준)
3. **거시 F1**은 모든 것 갈래에 같은 무게를 준다
4. 한결같은 값매김을 위해 **seqeval** 라이브러리를 쓴다

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
