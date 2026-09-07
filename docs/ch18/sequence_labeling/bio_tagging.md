# 차례 이름표 붙이기의 BIO 방식
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- IOB, IOB2, BIOES 이름표 방식을 이해하고 짠다
- 서로 다른 이름표 꼴을 코드로 바꾼다
- 이름표 차례가 어긋나지 않고 올바른지 확인한다
- 모델마다 이름표 방식의 맞바꿈을 살핀다
- 이름표 붙인 차례에서 것을 정확히 뽑아낸다

## 들어가며

이름표 방식은 차례 이름표 안에 것의 경계를 담는 짜임새 있는 길을 준다. 어떤 이름표 방식을 고르느냐가 이름표 자리의 복잡도와 모델이 것의 경계를 배우는 힘 모두에 영향을 준다. 이 절에서는 가장 흔한 방식인 IOB, IOB2, BIOES를 두루 다룬다.

## BIO 갈래의 이름표 방식

### IOB(안-바깥-시작) — 처음 판

Ramshaw와 Marcus(1995)가 내놓은 처음의 IOB 방식은 이름표 앞가지 셋을 쓴다:

| 앞가지 | 뜻 | 쓰임 |
|--------|---------|-------|
| B- | 시작 | 같은 갈래의 다른 것 바로 뒤에 올 **때만** 것의 첫 토막 |
| I- | 안 | 것 안의 다른 모든 토막 |
| O | 바깥 | 어느 것에도 들지 않는 토막 |

**보기**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB:     I-PER  I-PER  O        I-ORG  I-ORG O   I-LOC
```

유의: 처음의 IOB에서 B- 앞가지는 같은 갈래의 것이 잇달아 올 때 가르는 데만 쓴다.

### IOB2(안-바깥-시작, 두 번째 판)

이제 사실상의 표준인 IOB2는 B- 앞가지의 쓰임을 고친다:

| 앞가지 | 뜻 | 쓰임 |
|--------|---------|-------|
| B- | 시작 | 어떤 것이든 첫 토막을 **늘** 나타낸다 |
| I- | 안 | 것 안에서 이어지는 토막 |
| O | 바깥 | 어느 것에도 들지 않는 토막 |

**보기**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB2:    B-PER  I-PER  O        B-ORG  I-ORG O   B-LOC
```

### BIOES(시작-안-바깥-끝-홑)

BIOES(BILOU라고도 한다)는 경계를 드러내는 표시를 준다:

| 앞가지 | 뜻 | 쓰임 |
|--------|---------|-------|
| B- | 시작 | 토막 여럿인 것의 첫 토막 |
| I- | 안 | 것의 가운데 토막(토막 3개 이상) |
| O | 바깥 | 것이 아닌 토막 |
| E- | 끝 | 토막 여럿인 것의 마지막 토막 |
| S- | 홑 | 토막 하나짜리 것 |

**보기**:
```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
BIOES:   B-PER  E-PER  O        B-ORG  E-ORG O   S-LOC
```

## 이름표 자리의 수학적 살핌

### 이름표 자리의 복잡도

것 갈래가 $k$개일 때 이름표 자리의 크기는 다음과 같다:

| 방식 | 이름표 개수 | 식 |
|--------|------------------|---------|
| IOB/IOB2 | $2k + 1$ | 갈래마다 B-, I-와 O |
| BIOES | $4k + 1$ | 갈래마다 B, I, E, S와 O |

**것 갈래가 4개(PER, ORG, LOC, MISC)일 때의 보기**:

- IOB2: $2 \times 4 + 1 = 9$ labels
- BIOES: $4 \times 4 + 1 = 17$ labels

### 넘어가기 제약

옳은 이름표 이음은 정해진 넘어감 규칙을 따른다. $y_{i-1}$과 $y_i$을 잇달은 이름표라 하자.

**IOB2의 올바른 넘어가기**:

$$
\text{Valid}(y_{i-1}, y_i) = \begin{cases}
\text{True} & \text{if } y_i = \text{O} \\
\text{True} & \text{if } y_i = \text{B-}t \text{ for any type } t \\
\text{True} & \text{if } y_i = \text{I-}t \text{ and } y_{i-1} \in \{\text{B-}t, \text{I-}t\} \\
\text{False} & \text{otherwise}
\end{cases}
$$

**BIOES의 올바른 넘어가기**:

BIOES의 넘어가기 행렬은 제약이 더 많다:

| 에서 \ 으로 | O | B-t | I-t | E-t | S-t | B-t' | S-t' |
|-----------|---|-----|-----|-----|-----|------|------|
| O | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| B-t | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| I-t | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| E-t | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| S-t | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |

여기서 $t$과 $t'$은 저마다 같은 것 갈래와 다른 것 갈래를 나타낸다.

## PyTorch 구현

### 이름표 방식 클래스

```python
import torch
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass

class TagScheme(Enum):
    """받치는 이름표 방식."""
    IOB = "IOB"
    IOB2 = "IOB2"
    BIOES = "BIOES"


@dataclass
class TagInfo:
    """이름표에서 뜯어 읽은 앎."""
    prefix: str
    entity_type: Optional[str]
    
    @classmethod
    def parse(cls, tag: str) -> 'TagInfo':
        """이름표 글자열을 앞가지와 것 갈래로 뜯어 읽기."""
        if tag == 'O':
            return cls(prefix='O', entity_type=None)
        
        parts = tag.split('-', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tag format: {tag}")
        
        return cls(prefix=parts[0], entity_type=parts[1])
    
    def __str__(self) -> str:
        if self.entity_type is None:
            return 'O'
        return f"{self.prefix}-{self.entity_type}"


class TagValidator:
    """이름표 방식의 규칙에 따라 이름표 차례를 확인한다."""
    
    @staticmethod
    def validate_iob2(tags: List[str]) -> Tuple[bool, str]:
        """
        IOB2 이름표 차례 확인하기.
        
        규칙:
        1. I-TYPE 앞에는 같은 갈래의 B-TYPE이나 I-TYPE이 와야 한다
        2. B-TYPE은 아무 이름표 뒤에나 올 수 있다
        3. O는 아무 이름표 뒤에나 올 수 있다
        
        인수:
            tags: IOB2 이름표의 목록
            
        반환값:
            (is_valid, error_message) 튜플
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            
            try:
                info = TagInfo.parse(tag)
            except ValueError as e:
                return False, f"Position {i}: {e}"
            
            if info.prefix not in ('B', 'I'):
                return False, f"Position {i}: Invalid prefix '{info.prefix}' for IOB2"
            
            # I- 이름표 앞에는 같은 갈래의 B-나 I-가 와야 한다
            if info.prefix == 'I':
                if i == 0:
                    return False, f"Position {i}: I-{info.entity_type} cannot start sequence"
                
                prev_tag = tags[i - 1]
                if prev_tag == 'O':
                    return False, f"Position {i}: I-{info.entity_type} cannot follow O"
                
                prev_info = TagInfo.parse(prev_tag)
                if prev_info.entity_type != info.entity_type:
                    return False, (f"Position {i}: I-{info.entity_type} cannot follow "
                                   f"{prev_info.prefix}-{prev_info.entity_type}")
        
        return True, "Valid IOB2 sequence"
    
    @staticmethod
    def validate_bioes(tags: List[str]) -> Tuple[bool, str]:
        """
        BIOES 이름표 차례 확인하기.
        
        규칙:
        1. B- 뒤에는 같은 갈래의 I-나 E-가 와야 한다
        2. I- 앞에는 같은 갈래의 B-나 I-가, 뒤에는 I-나 E-가 와야 한다
        3. E- 앞에는 같은 갈래의 B-나 I-가 와야 한다
        4. S-는 토막 하나로 온전한 것을 나타낸다
        
        인수:
            tags: BIOES 이름표의 목록
            
        반환값:
            (is_valid, error_message) 튜플
        """
        for i, tag in enumerate(tags):
            if tag == 'O':
                continue
            
            try:
                info = TagInfo.parse(tag)
            except ValueError as e:
                return False, f"Position {i}: {e}"
            
            if info.prefix not in ('B', 'I', 'O', 'E', 'S'):
                return False, f"Position {i}: Invalid prefix '{info.prefix}' for BIOES"
            
            # 앞선 것 제약 살피기
            if info.prefix in ('I', 'E'):
                if i == 0:
                    return False, f"Position {i}: {tag} cannot start sequence"
                
                prev_info = TagInfo.parse(tags[i - 1])
                valid_prev = prev_info.prefix in ('B', 'I') and \
                             prev_info.entity_type == info.entity_type
                if not valid_prev:
                    return False, f"Position {i}: {tag} cannot follow {tags[i-1]}"
            
            # 뒤따르는 것 제약 살피기
            if info.prefix in ('B', 'I'):
                if i == len(tags) - 1:
                    return False, f"Position {i}: {tag} cannot end sequence (needs E-)"
                
                next_info = TagInfo.parse(tags[i + 1])
                valid_next = next_info.prefix in ('I', 'E') and \
                             next_info.entity_type == info.entity_type
                if not valid_next:
                    return False, f"Position {i}: {tag} cannot be followed by {tags[i+1]}"
        
        return True, "Valid BIOES sequence"
    
    @classmethod
    def validate(cls, tags: List[str], scheme: TagScheme) -> Tuple[bool, str]:
        """정한 방식에 따라 이름표 확인하기."""
        if scheme == TagScheme.IOB2:
            return cls.validate_iob2(tags)
        elif scheme == TagScheme.BIOES:
            return cls.validate_bioes(tags)
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
```

### 이름표 바꾸기

```python
class TagConverter:
    """서로 다른 이름표 방식 사이를 바꾸기."""
    
    @staticmethod
    def iob2_to_bioes(tags: List[str]) -> List[str]:
        """
        IOB2 이름표를 BIOES 이름표로 바꾸기.
        
        바꾸기 규칙:
        - 토막 하나짜리 것: B-TYPE → S-TYPE
        - 토막 여럿의 시작: B-TYPE → B-TYPE(뒤에 I-가 올 때)
        - 토막 여럿의 가운데: I-TYPE → I-TYPE(뒤에 I-가 올 때)
        - 토막 여럿의 끝: I-TYPE → E-TYPE(뒤에 O나 B-가 오거나 끝일 때)
        
        인수:
            tags: IOB2 이름표의 목록
            
        반환값:
            BIOES 이름표의 목록
        """
        bioes_tags = []
        n = len(tags)
        
        for i, tag in enumerate(tags):
            if tag == 'O':
                bioes_tags.append('O')
                continue
            
            info = TagInfo.parse(tag)
            
            # 것의 마지막 토막인지 살피기
            is_last = (i == n - 1) or \
                      (tags[i + 1] == 'O') or \
                      (TagInfo.parse(tags[i + 1]).prefix == 'B')
            
            if info.prefix == 'B':
                if is_last:
                    # 토막 하나짜리 것
                    bioes_tags.append(f'S-{info.entity_type}')
                else:
                    # 토막 여럿짜리 것의 시작
                    bioes_tags.append(f'B-{info.entity_type}')
            else:  # I- 앞가지
                if is_last:
                    # 토막 여럿짜리 것의 끝
                    bioes_tags.append(f'E-{info.entity_type}')
                else:
                    # 토막 여럿짜리 것의 가운데
                    bioes_tags.append(f'I-{info.entity_type}')
        
        return bioes_tags
    
    @staticmethod
    def bioes_to_iob2(tags: List[str]) -> List[str]:
        """
        BIOES 이름표를 IOB2 이름표로 바꾸기.
        
        바꾸기 규칙:
        - S-TYPE → B-TYPE
        - E-TYPE → I-TYPE
        - B-TYPE, I-TYPE, O는 그대로 둔다
        
        인수:
            tags: BIOES 이름표의 목록
            
        반환값:
            IOB2 이름표의 목록
        """
        iob2_tags = []
        
        for tag in tags:
            if tag == 'O':
                iob2_tags.append('O')
                continue
            
            info = TagInfo.parse(tag)
            
            if info.prefix == 'S':
                iob2_tags.append(f'B-{info.entity_type}')
            elif info.prefix == 'E':
                iob2_tags.append(f'I-{info.entity_type}')
            else:  # B 또는 I
                iob2_tags.append(tag)
        
        return iob2_tags
    
    @staticmethod
    def tags_to_entities(
        tokens: List[str],
        tags: List[str],
        scheme: TagScheme = TagScheme.IOB2
    ) -> List[Tuple[str, str, int, int]]:
        """
        이름표 붙인 차례에서 것 뽑기.
        
        인수:
            tokens: 토막의 목록
            tags: 그에 맞는 이름표의 목록
            scheme: 쓴 이름표 방식
            
        반환값:
            (entity_text, entity_type, start_idx, end_idx) 튜플의 목록
        """
        assert len(tokens) == len(tags), "Tokens and tags must have same length"
        
        # 한결같이 다루려 IOB2로 바꾸기
        if scheme == TagScheme.BIOES:
            tags = TagConverter.bioes_to_iob2(tags)
        
        entities = []
        current_entity = None  # (type, start_idx, tokens)
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag == 'O':
                # 있으면 지금 것을 닫기
                if current_entity is not None:
                    ent_type, start_idx, ent_tokens = current_entity
                    entities.append((
                        ' '.join(ent_tokens),
                        ent_type,
                        start_idx,
                        i
                    ))
                    current_entity = None
            
            elif tag.startswith('B-'):
                # 앞 것을 닫고 새것을 시작하기
                if current_entity is not None:
                    ent_type, start_idx, ent_tokens = current_entity
                    entities.append((
                        ' '.join(ent_tokens),
                        ent_type,
                        start_idx,
                        i
                    ))
                
                entity_type = TagInfo.parse(tag).entity_type
                current_entity = (entity_type, i, [token])
            
            elif tag.startswith('I-'):
                # 지금 것을 이어 가기
                if current_entity is not None:
                    current_entity[2].append(token)
        
        # 마지막 것을 잊지 말 것
        if current_entity is not None:
            ent_type, start_idx, ent_tokens = current_entity
            entities.append((
                ' '.join(ent_tokens),
                ent_type,
                start_idx,
                len(tokens)
            ))
        
        return entities
```

### CRF를 위한 넘어가기 행렬 세우기

```python
def build_transition_mask(
    label_to_idx: Dict[str, int],
    scheme: TagScheme = TagScheme.IOB2
) -> torch.Tensor:
    """
    CRF 층의 넘어가기 마스크 세우기.
    
    마스크는 올바른 넘어가기에 0, 올바르지 않은 것에 -inf를 준다.
    
    인수:
        label_to_idx: 이름표 글자열에서 번호로의 대응
        scheme: 이름표 방식
        
    반환값:
        넘어가기 점수를 담은, 꼴이 (num_labels, num_labels)인 텐서
    """
    num_labels = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # 모든 넘어가기를 올바른 것으로 두고 첫자리매김
    mask = torch.zeros(num_labels, num_labels)
    
    # 것 갈래 뽑기
    entity_types = set()
    for label in label_to_idx:
        if label != 'O':
            info = TagInfo.parse(label)
            entity_types.add(info.entity_type)
    
    for i in range(num_labels):
        for j in range(num_labels):
            from_label = idx_to_label[i]
            to_label = idx_to_label[j]
            
            if not _is_valid_transition(from_label, to_label, scheme):
                mask[i, j] = float('-inf')
    
    return mask


def _is_valid_transition(
    from_tag: str, 
    to_tag: str, 
    scheme: TagScheme
) -> bool:
    """from_tag에서 to_tag로 넘어가는 것이 올바른지 살피기."""
    # O는 무엇으로든 넘어갈 수 있다
    if from_tag == 'O':
        if to_tag == 'O':
            return True
        to_info = TagInfo.parse(to_tag)
        if scheme == TagScheme.IOB2:
            return to_info.prefix == 'B'
        else:  # BIOES
            return to_info.prefix in ('B', 'S')
    
    from_info = TagInfo.parse(from_tag)
    
    # 무엇이든 O로 넘어갈 수 있다
    if to_tag == 'O':
        if scheme == TagScheme.IOB2:
            return True
        else:  # BIOES
            return from_info.prefix in ('E', 'S')
    
    to_info = TagInfo.parse(to_tag)
    
    if scheme == TagScheme.IOB2:
        # I-X는 B-X나 I-X 뒤에만 올 수 있다
        if to_info.prefix == 'I':
            return from_info.entity_type == to_info.entity_type
        # B-X는 무엇 뒤에나 올 수 있다
        return True
    
    else:  # BIOES
        # I나 E 앞에는 같은 갈래의 B나 I가 와야 한다
        if to_info.prefix in ('I', 'E'):
            return (from_info.prefix in ('B', 'I') and 
                    from_info.entity_type == to_info.entity_type)
        
        # B나 S는 O, E, S 뒤에만 올 수 있다
        if to_info.prefix in ('B', 'S'):
            return from_info.prefix in ('E', 'S')
        
        return False
```

## 방식 고르기 지침

### IOB2를 쓸 때

**좋은 점**:

- 이름표 자리가 더 단순하다($2k + 1$ 대 $4k + 1$)
- 이름표마다 익힘 보기가 더 많다
- 이미 있는 도구와 자료 뭉치가 널리 받쳐 준다
- 대부분의 차례 이름표 붙이기 일에 넉넉하다

**가장 알맞은 곳**:

- 학습 데이터가 모자랄 때
- 것의 짜임이 단순하다
- 이미 있는 자료 뭉치(CoNLL 꼴)와 맞물린다

### BIOES를 쓸 때

**좋은 점**:

- 드러난 경계 표시가 배움을 낫게 한다
- S- 이름표가 토막 하나짜리 것을 가려내는 데 돕는다
- 일부 잣대에서 성능이 더 낫다(F1 1~2% 나아짐)
- 뒤따르는 CRF 층에 더 알찬 앎을 준다

**가장 알맞은 곳**:

- 익힘 자료가 넉넉하다
- 자료 뭉치에 토막 하나짜리 것이 많다
- CRF나 짜임 있는 어림 층을 쓸 때
- 연구용 잣대 시험

### 실험으로 견주기

여러 연구는 BIOES가 성능을 낫게 할 수 있음을 보였다:

| 모델 | 자료 뭉치 | IOB2 F1 | BIOES F1 | Δ |
|-------|---------|---------|----------|---|
| 두 방향 LSTM-CRF | CoNLL-2003 | 90.94 | 91.21 | +0.27 |
| BERT-base | CoNLL-2003 | 92.4 | 92.8 | +0.4 |

넘어가기 제약을 드러내어 나타낼 수 있는 CRF 층을 쓸 때 나아짐이 더 두드러진다.

## 아래낱말 토막내개 다루기

요즘 변환기는 아래낱말 토막내기를 쓰는데, 이 때문에 이름표 붙이기가 까다로워진다:

```python
def align_labels_to_subwords(
    word_labels: List[str],
    word_to_subword_map: List[List[int]],
    label_first_subword_only: bool = True
) -> List[str]:
    """
    낱말 수준 이름표를 아래낱말 토막에 맞추기.
    
    인수:
        word_labels: 낱말마다의 이름표
        word_to_subword_map: 낱말마다의 아래낱말 번호 목록
        label_first_subword_only: True이면 첫 아래낱말만 이름표를 받는다
        
    반환값:
        아래낱말 토막마다의 이름표
    """
    subword_labels = []
    
    for word_idx, label in enumerate(word_labels):
        subword_indices = word_to_subword_map[word_idx]
        
        for i, subword_idx in enumerate(subword_indices):
            if i == 0:
                # 첫 아래낱말이 본디 이름표를 받는다
                subword_labels.append(label)
            else:
                if label_first_subword_only:
                    # 첫째가 아닌 아래낱말은 특별 이름표를 받는다(손실에서 무시)
                    subword_labels.append('[IGNORE]')
                else:
                    # 이어지는 아래낱말에 I- 이름표 퍼뜨리기
                    if label.startswith('B-'):
                        subword_labels.append('I-' + label[2:])
                    else:
                        subword_labels.append(label)
    
    return subword_labels
```

## 그려 보기와 벌레잡기

```python
def visualize_tags(
    tokens: List[str],
    tags: List[str],
    scheme: TagScheme = TagScheme.IOB2
) -> str:
    """
    이름표 붙인 차례를 눈으로 볼 수 있게 나타내기.
    
    인수:
        tokens: 토막의 목록
        tags: 이름표의 목록
        scheme: 이름표 방식
        
    반환값:
        토막과 이름표를 줄맞춘 글자열
    """
    # 줄맞춤을 위한 최대 너비 찾기
    max_token_len = max(len(t) for t in tokens)
    max_tag_len = max(len(t) for t in tags)
    
    lines = []
    lines.append("Tokens: " + " ".join(f"{t:<{max_token_len}}" for t in tokens))
    lines.append("Tags:   " + " ".join(f"{t:<{max_token_len}}" for t in tags))
    
    # 것 뽑기 더하기
    entities = TagConverter.tags_to_entities(tokens, tags, scheme)
    if entities:
        lines.append("\nExtracted Entities:")
        for text, etype, start, end in entities:
            lines.append(f"  [{start}:{end}] {etype}: '{text}'")
    
    return "\n".join(lines)


# 사용 예
tokens = ["Barack", "Obama", "visited", "New", "York", "City"]
tags_iob2 = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC"]
tags_bioes = TagConverter.iob2_to_bioes(tags_iob2)

print("IOB2 Format:")
print(visualize_tags(tokens, tags_iob2, TagScheme.IOB2))
print("\nBIOES Format:")
print(visualize_tags(tokens, tags_bioes, TagScheme.BIOES))
```

## 요약

BIO 이름표 방식은 것의 경계를 담는 원칙 있는 길을 준다:

1. **IOB2**는 B-가 늘 것의 시작을 나타내는 표준 방식이다
2. **BIOES**는 경계를 더 잘 배우도록 끝 표시와 홑 표시를 드러내어 더한다
3. **넘어가기 제약**은 CRF 층의 마스크로 강제할 수 있다
4. **아래낱말 맞추기**는 이름표를 퍼뜨리는 일을 조심스레 다뤄야 한다
5. **방식 고르기**는 자료 크기, 것의 성질, 모델 얼개에 달렸다

## 참고 문헌

1. Ramshaw, L. A., & Marcus, M. P. (1995). Text Chunking using Transformation-Based Learning. *ACL Workshop on Very Large Corpora*.

2. Ratinov, L., & Roth, D. (2009). Design Challenges and Misconceptions in Named Entity Recognition. *CoNLL*.

3. Sang, E. F. T. K., & Veenstra, J. (1999). Representing Text Chunks. *EACL*.

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
    자리마다 따로 소프트맥스를 쓰면 이름표를 서로 아랑곳없이 다루어 넘어감의 매임을 버린다. 보기로 O 뒤에 I-PER을 미루어 볼 수 있는데, BIO 이름 붙이기에서는 옳지 않다. CRF 켜는 이름표 이음 전체의 함께 확률을 모델로 삼아, 어떤 이름표 넘어감이 옳은지 담는 넘어감 행렬 $A_{ij}$을 배운다. 이음의 CRF 점수는 $s(x, y) = \sum_t (E_{y_t, t} + A_{y_t, y_{t+1}})$이며 $E$은 두 방향 LSTM에서 나온 내보냄 점수다. 이러면 비터비 풀기로 두루 앞뒤가 맞는 미루어 봄을 얻는다.

---

**연습문제 3.**
것 수준에서의 이름 알아보기 값매김에 쓰는 정밀도, 재현율, F1 점수를 설명하여라. 것 수준 값매김이 토막 수준보다 왜 더 빡빡한가?

??? success "연습문제 3 풀이"
    **개체 수준** 따지기는 개체의 테두리와 갈래가 모두 딱 맞아야 한다. 정밀도 = (옳게 미루어 본 개체) / (미루어 본 개체 모두). 재현율 = (옳게 미루어 본 개체) / (참 개체 모두). $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$이다. 이는 토막 수준보다 깐깐하다. 참 개체가 "New York City"인데 "New York"으로 미루어 보면 토막 수준에서는 반쯤 점수를 받지만(토막 3개 가운데 2개가 맞다) 개체 수준에서는 0점이다(테두리가 어긋난다). 개체 수준의 자가 참 세상의 쓸모를 더 잘 비춘다.

---

**연습문제 4.**
두 방향 LSTM은 양쪽 방향의 맥락을 어떻게 담아내는가? 자리 $t$에서의 숨은 상태 셈을 적어라.

??? success "연습문제 4 풀이"
    두 방향 LSTM은 $x_1, \ldots, x_T$을 다루는 앞으로 가는 LSTM과 $x_T, \ldots, x_1$을 다루는 뒤로 가는 LSTM으로 이루어진다.

    $$\overrightarrow{h}_t = \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

    자리 $t$의 마지막 나타냄은 이어 붙인 $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$이며, 왼쪽 앞뒤 흐름($\overrightarrow{h}_t$을 거쳐)과 오른쪽 앞뒤 흐름($\overleftarrow{h}_t$을 거쳐)을 모두 담는다. 개체명 알아내기에서는 둘레 낱말이 양쪽에서 걸리는 일이 잦으므로 이것이 종요롭다.
