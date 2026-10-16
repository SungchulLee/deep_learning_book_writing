# BIO와 BIOES: 개체 경계의 시퀀스 표현

평면 개체명 인식은 토큰마다 개체의 시작·계속·바깥을 표시해 연속 span과 유형을 표현할 수 있다. 같은 유형의 개체가 붙어 있을 때도 경계를 구분해야 한다. 단일 태그 열은 토큰 하나에 한 레이블을 주므로 중첩·불연속 언급을 일반적으로 모두 표현하지 못한다.

## 1. IOB1·IOB2·BIOES의 차이

원래 IOB1에서는 개체 내부를 I-X로 표시하고 같은 유형의 다른 개체가 바로 앞에 있을 때 새 개체의 첫 토큰에 B-X를 사용한다. 따라서 문장 처음의 I-X가 그 체계에서는 반드시 오류가 아니다. 아래 기존 예는 IOB1의 규칙을 보여 준다.

```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB:     I-PER  I-PER  O        I-ORG  I-ORG O   I-LOC
```

IOB2는 모든 개체의 첫 토큰을 B-X, 뒤 토큰을 I-X, 개체 밖을 O로 표시한다. 흔히 BIO라고 부르는 방식이 이것이지만 데이터·도구가 IOB1과 IOB2 중 무엇을 뜻하는지 확인해야 한다.

```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
IOB2:    B-PER  I-PER  O        B-ORG  I-ORG O   B-LOC
```

BIOES는 여러 토큰 개체에 B·I·E, 단일 토큰 개체에 S, 바깥에 O를 쓴다. BILOU는 같은 경계 구분을 B·I·L·U·O로 표기하므로 E↔L, S↔U의 이름 변환이 필요하다. 같은 의미 체계라고 태그 문자열이 그대로 호환되는 것은 아니다.

```
Tokens:  Steve  Jobs   founded  Apple  Inc   in  California
BIOES:   B-PER  E-PER  O        B-ORG  E-ORG O   S-LOC
```

유형 K개이면 IOB1·IOB2의 레이블 목록은2K+1, BIOES는4K+1개다. K4이면9개·17개다. 목록 크기와 실제로 유효한 시퀀스 수는 다르다. 두 체계는 유효한 평면 연속 span을 같은 정보로 표현할 수 있어 BIOES의 레이블이 많다는 사실 자체가 더 풍부한 정답 객체를 담는다는 뜻은 아니다.

## 2. 시작·종료와 전이 제약

IOB2에서 I-X 앞에는 같은 유형의 B-X 또는 I-X가 와야 한다. 문장 처음에는 O·B-X만 가능하고 종료는 O·B-X·I-X 모두 가능하다. BIOES에서는 B-X·I-X 뒤에 같은 유형의 I-X·E-X가 와야 하며, 시작은 O·B-X·S-X, 종료는 O·E-X·S-X다. O·E·S 뒤에는 O·B·S로 새 구간을 시작할 수 있다.

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
        """이름표 글자열을 앞가지와 것 클래스로 뜯어 읽기."""
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
        1. I-TYPE 앞에는 같은 클래스의 B-TYPE이나 I-TYPE이 와야 한다
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
            
            # I- 이름표 앞에는 같은 클래스의 B-나 I-가 와야 한다
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
        1. B- 뒤에는 같은 클래스의 I-나 E-가 와야 한다
        2. I- 앞에는 같은 클래스의 B-나 I-가, 뒤에는 I-나 E-가 와야 한다
        3. E- 앞에는 같은 클래스의 B-나 I-가 와야 한다
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

TagValidator는 문자열 파싱과 연결 규칙의 일부를 검사한다. 그러나 TagInfo.parse는 빈 유형 `B-`를 허용하고 유형 목록의 회원 여부도 검증하지 않는다. BIOES의 허용 prefix 목록에 O가 있어 `O-X`처럼 규격 밖인 문자열을 통과시킬 수도 있다. 일부 이전·다음 태그 파싱은 예외 처리 밖에서 이루어져 언제나 `(False,메시지)`로 반환되는 것도 아니다. TagScheme.IOB는 enum에 있지만 validate에서는 지원하지 않는다.

## 3. 변환과 span 추출의 입력 조건

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

IOB2→BIOES 변환은 다음 태그를 보고 한 토큰 개체를 S로, 마지막 I를 E로 바꾼다. BIOES→IOB2는 S→B, E→I로 바꾼다. **입력이 먼저 유효하다는 전제**가 있어야 경계·유형을 보존하는 변환으로 읽을 수 있다. 잘못된 유형 전환을 자동 교정하거나 IOB1을 일반적으로 변환하는 함수는 아니다.

`tags_to_entities`는 토큰·태그 길이가 같아야 하며 반환 start·end는0부터 시작하는 **토큰 단위 반열린 구간 [start,end)**다. 문자 offset은 아니다. 문자열을 공백으로 join하므로 원문의 공백·구두점 배치를 복원하지도 않는다. 한국어·서브워드에서 원문 복원이 필요하면 원문 offset을 보관한다.

초기 I 태그는 열린 개체가 없어 무시되고, 열린 개체 뒤의 다른 유형 I 태그도 유형 확인 없이 같은 개체에 붙을 수 있다. scheme=IOB라고 해도 IOB1 처리를 따로 하지 않는다. 유효성 검사와 엄격 거부·오류 교정 폴리시를 먼저 정의해야 하며, 추출 결과가 그럴듯하다는 이유로 잘못된 태그를 정상으로 취급하면 안 된다.

## 4. CRF의 전이 마스크

```python
def build_transition_mask(
    label_to_idx: Dict[str, int],
    scheme: TagScheme = TagScheme.IOB2
) -> torch.Tensor:
    """
    CRF 레이어의 넘어가기 마스크 세우기.
    
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
    
    # 것 클래스 뽑기
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
        # I나 E 앞에는 같은 클래스의 B나 I가 와야 한다
        if to_info.prefix in ('I', 'E'):
            return (from_info.prefix in ('B', 'I') and 
                    from_info.entity_type == to_info.entity_type)
        
        # B나 S는 O, E, S 뒤에만 올 수 있다
        if to_info.prefix in ('B', 'S'):
            return from_info.prefix in ('E', 'S')
        
        return False
```

mask[i,j]는 이전 레이블 i에서 다음 j로 가는 점수에 더하는0 또는−∞다. 금지 전이에 실제로 적용해야 제약이 생기며, 학습 가능한 전이 점수만으로 금지 전이가 수학적으로 불가능해지지는 않는다. 시작·종료 제약은 별도의 점수·마스크가 필요하다. 이 코드의 K×K 행렬만으로 문장 처음 I-X나 마지막 BIOES의 B-X를 막지는 못한다.

label_to_idx는0~K−1의 연속 인덱스와 유효한 해당 체계의 태그 집합이어야 한다. 지원되지 않은 IOB enum이 내부에서 BIOES 쪽 분기로 처리되는 것도 완전한 IOB1 지원이 아니다. 모든 경로가 막힌 입력·패딩·빈 문장의 처리는 CRF 쪽에서 정의해야 한다. 마스크를 학습 정규화와 Viterbi 중 어디에 적용하는지도 명시한다.

## 5. 서브워드와 단어 레이블 정렬

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
                    # 첫째가 아닌 아래낱말은 특별 이름표를 받는다(로스에서 무시)
                    subword_labels.append('[IGNORE]')
                else:
                    # 이어지는 아래낱말에 I- 이름표 퍼뜨리기
                    if label.startswith('B-'):
                        subword_labels.append('I-' + label[2:])
                    else:
                        subword_labels.append(label)
    
    return subword_labels
```

첫 서브워드만 지도하는 폴리시는 뒤 조각과 특수 토큰을 loss의 ignore_index로 처리한다. 모든 조각을 지도하면 IOB2의 B-X 뒤 조각은 I-X로 바꾸는 등의 규칙이 필요하다. 두 폴리시의 단어별 웨이트도 달라질 수 있다. BIOES의 S·E를 모든 조각에 그대로 복제하면 유효한 경계 표현이 되지 않으므로 체계별 재부호화가 필요하다.

위 함수는 `[IGNORE]`라는 문자열을 반환할 뿐 실제 로스의 정수 ignore_index로 바꾸지 않는다. 또 word_to_subword_map의 실제 인덱스를 출력 배치에 사용하지 않고 조각 수만큼 순서대로 append한다. 특수 토큰·잘린 단어·순서가 다른 매핑·빈 매핑을 포함한 전체 tokenizer 출력에 자동 정렬되는 함수가 아니다. word_ids·offset 등 실제 토큰화 결과와 함께 목표 배열을 구성해야 한다.

## 6. 표시·평가와 체계 선택

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

**기존 출력:**

```
IOB2 Format:
Tokens: Barack  Obama   visited New     York    City   
Tags:   B-PER   I-PER   O       B-LOC   I-LOC   I-LOC  

Extracted Entities:
  [0:2] PER: 'Barack Obama'
  [3:6] LOC: 'New York City'

BIOES Format:
Tokens: Barack  Obama   visited New     York    City   
Tags:   B-PER   E-PER   O       B-LOC   I-LOC   E-LOC  

Extracted Entities:
  [0:2] PER: 'Barack Obama'
  [3:6] LOC: 'New York City'
```

시연은 유효한 IOB2 예제를 변환·추출해 표시한다. 빈 tokens·tags에는 max()가 실패할 수 있고 표시 폭이 정확한 원문 문자 offset을 뜻하지도 않는다. 평가에는 경계와 유형이 모두 맞는 span 단위 비교, 무효 태그 처리와 분할·집계 규칙을 명시한다. 토큰 정확도는 O 비중이 크면 높아질 수 있다.

IOB2와 BIOES의 선택은 데이터·도구 호환성과 모델·제약 설계에 따른다. 레이블을 바꿔 얻는 고정1~2% 개선이나 출처·동일 조건이 없는 F1 표를 보편적 결과로 제시하지 않는다. [NER 설계 연구](https://aclanthology.org/W09-1119/)처럼 구체적인 실험 조건과 정규화·디코딩을 맞춰 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Barack Obama visited New York City의 IOB2·BIOES를 비교하고 같은 유형의 인접 개체를 어떻게 구분하는지 설명하라.

</div>

??? success "연습문제 1 풀이"

    IOB2는 B-PER I-PER O B-LOC I-LOC I-LOC, BIOES는 B-PER E-PER O B-LOC I-LOC E-LOC다. 한 토큰 개체는 BIOES에서 S-X다. 서로 다른 인접 개체는 새 B 또는 S로 시작하므로 유형이 같아도 경계를 나눈다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CRF의 학습 가능한 전이 행렬만으로 IOB2가 항상 유효한가?

</div>

??? success "연습문제 2 풀이"

    금지 전이에 −∞ 같은 강제 제약을 주지 않으면 부적절한 전이가 여전히 가능하다. 문장 시작·종료 제약도 K×K 마스크와 별도다. 학습 정규화·디코딩에 적용하는 위치와 무효 경로 처리를 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
tags_to_entities의 start·end와 무효 I 태그 처리의 한계를 설명하라.

</div>

??? success "연습문제 3 풀이"

    start·end는 토큰 인덱스의 반열린 구간이며 문자 offset이 아니다. 초기 I는 무시될 수 있고 열린 개체 뒤 다른 유형 I도 붙일 수 있어 먼저 유효성 검사·교정 폴리시를 적용해야 한다. 공백 join은 원문 복원을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
BIOES의 단일 단어가 두 서브워드로 나뉘면 S-X를 두 번 복제해도 되는가?

</div>

??? success "연습문제 4 풀이"

    단일 개체를 두 개의 단일 토큰 개체로 표시하므로 안 된다. 모든 조각을 지도한다면 B-X E-X처럼 실제 조각 수에 맞게 재부호화한다. 첫 조각만 지도하면 나머지·특수 토큰을 실제 정수 ignore_index로 표시하고 원래 단어 수준으로 평가한다.



## 정리하며

BIO는 span 정보를 토큰 열로 표현하는 규칙이다. IOB1·IOB2·BIOES의 시작·종료·전이와 문자열 차이를 구분하고, 유효성 검사·CRF 제약·서브워드 정렬·평가의 계약을 일치시킨다. 이 코드에는 무효 태그·매핑·경계 사례 처리의 한계가 남아 있다.

**참고 문헌**

1. Ramshaw, L. A., & Marcus, M. P. (1995). Text Chunking using Transformation-Based Learning. *ACL Workshop on Very Large Corpora*.

2. Ratinov, L., & Roth, D. (2009). Design Challenges and Misconceptions in Named Entity Recognition. *CoNLL*.

3. Sang, E. F. T. K., & Veenstra, J. (1999). Representing Text Chunks. *EACL*.
