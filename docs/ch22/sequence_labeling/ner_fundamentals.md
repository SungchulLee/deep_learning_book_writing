# 개체명 인식 기초

개체명 인식(NER)은 텍스트에서 대상 언급의 범위를 찾고 미리 정의한 유형을 부여하는 과제다. 사람·조직·장소뿐 아니라 날짜·수량·분야별 표현을 포함할지는 주석 체계에 달렸다. 언급의 이름을 찾는 것과 실제 세계의 특정 개체 ID에 연결하거나 서로 같은 대상을 가리키는 언급을 묶는 것은 다른 작업이다.

## 1. 시퀀스 레이블링과 다른 표현

평면의 연속된 개체는 토큰 열 X=(x_1,…,x_n)에 BIO 같은 태그 열 Y를 부여하는 문제로 구성할 수 있다. 그러나 NER가 본질적으로 이 표현만을 갖는 것은 아니다. span 분류·생성·포인터 모델도 가능하고 중첩·불연속 언급에는 단일 평면 태그 열보다 다른 표현이 필요할 수 있다.

토큰별 독립 분류는 p(Y|X)=∏_i p(y_i|X)로 모델링한다. 레이블의 조건부 분해가 독립이라는 뜻이며 인코더가 다른 입력 토큰을 보지 못한다는 뜻은 아니다. 전체 문맥을 읽는 트랜스포머도 토큰별 softmax head를 사용할 수 있다.

선형 사슬 CRF는 시작·종료를 포함한 시퀀스 점수 s(X,Y)에서 다음 분포를 정의한다.

$$p(Y\mid X)=\frac{\exp s(X,Y)}{\sum_{Y'\in\mathcal Y(X)}\exp s(X,Y')}.$$

가능한 태그 열의 집합과 금지 전이·시작·종료 제약을 명시한다. 학습에서는 정답 열의 음의 로그우도를, 추론에서는 가장 높은 점수의 열을 구할 수 있다. Viterbi의 최적성은 선택한 점수·경로 집합 아래의 것이지 의미적으로 정확한 개체라는 보장은 아니다.

## 2. 유형·문맥과 경계

CoNLL-2003은 PER·ORG·LOC·MISC 네 유형을 사용하고 OntoNotes는 수량·시간을 포함한18유형을 사용한다. 각 범주의 정의는 데이터셋 지침에 달렸으며 MISC를 모든 나머지 개체의 보편적 범주로 취급하지 않는다. 분야별 체계와 계층은 [개체 유형 절](entity_types.md)에서 다룬다.

Washington은 사람·장소를 가리킬 수 있고 The Washington Post는 전체 신문사 언급이 조직일 수 있다. 이때 내부 단어 Washington만 별도로 ORG라고 해석하는 것과 전체 언급을 분류하는 것은 다르다. 직함·기관 접미사·구두점의 포함 여부도 지침이 정한다. 아래 경계 예의 `Dr.` 포함은 가능한 정책의 예이지 모든 NER 정답 규칙이 아니다.

## 3. 문자 구간과 토큰 정렬

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

Entity는 text·유형·문자 start·end를 보관하며 [start,end)의 반열린 구간을 가정한다. Python 문자열 인덱스는 바이트 offset이나 화면의 grapheme 개수와 같지 않을 수 있다. 다른 도구의 UTF-16 offset·Unicode 정규화와 교환할 때 기준을 맞춰야 한다. len(text)=end−start 검사만으로 원문[start:end]가 text와 같은지까지 보장하지 않는다.

현재 `overlaps`는 두 분리 조건에 같은 비교를 중복해 써서 한 방향의 분리만 검사한다. 예를 들어 self=[10,20), other=[0,5)도 겹친다고 반환할 수 있다. 일반적인 겹침은 self.start<other.end와 other.start<self.end를 모두 만족해야 한다. contains는 포함 관계의 경계를 비교하며 동일 구간도 포함으로 본다. 원문과 음수·범위 밖 인덱스 검증은 별도로 필요하다.

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
        text: 본디 글월 문자열
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

정렬 함수는 개체와 **일부라도 겹치는** 토큰을 골라 B·I를 붙인다. 개체 경계가 토큰 경계와 맞지 않으면 원래 문자 span을 정확히 복원하는 정렬이 아니다. 중첩·겹침 언급은 뒤에서 처리한 개체가 같은 토큰의 레이블을 덮어쓴다. text 인수도 실제 계산에 사용하지 않아 원문 일치·범위를 검사하지 않는다. 반환값은 태그 문자열 목록인데 타입 표기는 List[List[str]]로 되어 있는 차이도 있다.

## 4. 전처리와 후속 과제

```
Raw Text → Tokenization → (Optional: POS Tagging) → NER → Downstream Tasks
```

이 도식은 흔한 처리 순서의 예다. 모든 NER에 별도의 품사 태깅이 필수인 것은 아니다. 단어 토큰화도 항상 개체 경계를 보존하지 않고, 서브워드는 희귀 단어의 표현을 돕지만 언급을 여러 조각으로 나누므로 레이블·offset 정렬이 필요하다. 토크나이저의 정규화·특수 토큰·잘림 정책까지 고려한다.

```
"University of California, Berkeley" → Single ORG entity
"New York City Department of Education" → Single ORG entity  
"Dr. Martin Luther King Jr." → Single PER entity with title
```

```
"Bank of [America]_LOC"  → Contains nested LOC within ORG
"[Bank of America]_ORG"  → Full organization name

"[New York]_LOC University" → LOC within broader context
"[New York University]_ORG" → Full organization name
```

위 중첩 예는 가능한 주석을 설명하지만 모든 데이터셋이 회사 이름 내부의 장소까지 별도로 표시하는 것은 아니다. 평면 BIO로 바꿀 때 어떤 층을 남길지 정해야 한다. NER 출력은 정보 추출·질의응답·관계 추출·개체 연결의 입력이 될 수 있으나, 지식 그래프의 ID나 대명사와 이름의 상호참조를 자동으로 해결하지 않는다.

새 이름·분야·시간 변화와 모호한 문맥은 성능을 바꿀 수 있다. 사전만으로 해결되는 문제와 문맥 분류가 필요한 문제를 구분하고 대상 도메인의 별도 평가 자료를 준비한다.

## 5. 데이터 구조와 패딩

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
        
        # 어텐션 마스크 만들기
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

NERExample은 토큰·레이블·Entity를 보관하지만 to_tensor는 entities를 이용해 레이블을 만들지 않는다. 토큰 사전·레이블 사전과 이미 맞춰 둔 labels가 필요하다. 반환 input_ids·attention_mask·labels는 정수 텐서이며 모델·손실·학습 루프는 구현하지 않는다.

토큰은 max_length로 자르고 남은 위치에 PAD를 넣는다. 레이블도 같은 길이·순서여야 하는데 이를 검사하지 않아 길이가 다르면 목표 텐서 길이도 어긋날 수 있다. 모르는 레이블을 O로 바꾸어 오류를 조용히 배경으로 처리하므로 실제 데이터 검증을 대신하지 않는다. 어휘는 학습 분할에서만 만들고 검증·시험 정보가 들어가지 않도록 한다.

패딩 레이블도 O를 넣으므로 attention_mask가 있다고 손실에서 자동 제외되는 것은 아니다. 토큰 손실의 ignore_index나 CRF의 유효 마스크에 연결해야 한다. 잘림이 개체를 중간에서 끊으면 경계·평가 정책도 정한다. 토큰과 레이블 인덱스는 정렬된 목록에서 생성되므로 O나 PAD가 특정 번호라고 외부 코드에서 가정하지 않는다.

## 6. 개체 단위 평가

정확 일치 평가의 TP는 같은 문서·문장·좌표 기준에서 경계와 유형이 모두 맞는 예측이다. P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN)로 집계할 수 있다. 분모0·빈 정답·중복 예측의 정책과 micro/macro 기준을 명시한다. 같은 문자열이라도 다른 위치의 언급은 다른 개체 span이다.

New York City가 정답인데 New York만 예측하면 정확 span 평가에서는 TP가 아니라 FP하나·FN하나다. 일부 토큰 레이블은 맞을 수 있지만 이를 개체 F1의 부분 점수라고 해석하지 않는다. 부분 겹침·유형만의 평가를 쓰려면 대응·점수 규칙을 별도로 정해야 한다. 토큰 정확도는 O 비중 때문에 높아질 수 있어 경계·유형 오류 분석과 함께 본다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Barack Obama visited New York City에 IOB2 태그를 붙이고 span 단위를 설명하라.

</div>

??? success "연습문제 1 풀이"

    B-PER I-PER O B-LOC I-LOC I-LOC다. 토큰 구간은 [0,2)와[3,6)이며 원문 문자 offset과 다르다. 같은 유형의 다른 개체가 붙어 있어도 새B로 경계를 나눈다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
독립 softmax와 CRF의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"

    softmax head는 주어진 입력에서 토큰별 레이블 확률을 분해하지만 입력 인코더는 전체 문맥을 볼 수 있다. CRF는 태그 열을 공동 정규화하고 전이 점수를 사용한다. 유효 BIO를 강제하려면 시작·종료·금지 전이 제약을 별도로 적용해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Entity.overlaps에서 self=[10,20), other=[0,5)의 현재 반환값과 올바른 겹침 조건을 비교하라.

</div>

??? success "연습문제 3 풀이"

    현재는 동일한 방향의 분리 조건을 두 번 써 True를 반환할 수 있지만 두 구간은 겹치지 않는다. self.start<other.end와 other.start<self.end가 모두 참이어야 한다. 접촉하는 끝점만으로는 반열린 구간이 겹치지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
attention_mask가 있으면 패딩 O 레이블이 손실에서 자동 제외되는가?

</div>

??? success "연습문제 4 풀이"

    아니다. 모델의 입력 attention 처리와 손실의 유효 위치 마스크는 별개다. CE ignore_index나 CRF 마스크 등으로 실제 연결해야 한다. 토큰·레이블 길이와 잘림 경계·미등록 레이블의 O 치환도 검사해야 한다.



## 정리하며

NER는 언급의 범위·유형·문맥을 모델링한다. 문자·토큰 offset, 평면·중첩 표현과 유형·개체 연결을 구분하고 데이터 정렬·패딩·손실·span 평가를 일치시킨다. 현재 데이터 구조 코드의 겹침·정렬·검증 한계도 함께 확인한다.

**참고 문헌**

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. *CoNLL*.

2. Weischedel, R., et al. (2013). OntoNotes Release 5.0. Linguistic Data Consortium.

3. Lample, G., et al. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT*.

4. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
