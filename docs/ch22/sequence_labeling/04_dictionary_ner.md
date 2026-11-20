# 사전 기반 개체명 인식과 원문 위치

사전 기반 NER은 등록된 이름·표현을 텍스트와 비교해 개체 후보를 찾는다. 이 예제는 인물·조직·장소 사전의 정확 문자열 일치와 겹침 제거를 보여 준다. 문맥을 학습하거나 철자 유사도를 추정하는 뉴럴 네트워크는 아니며, 사전에 없던 새로운 이름을 자동 발견하는 모델도 아니다.

## 1. 코드와 출력

```python
"""
사전 기반 개체명 인식
==========================================

이 단원은 개체 사전과 조회 테이블을 써서 사전/가제티어 기반 개체명 인식을 구현한다.
이 방식은 단순하고 빠르며, 개체 목록을 뚜렷하게 정할 수 있는 개체 유형에 잘 맞는다.

학습 목표:
- 개체 사전(가제티어)을 세우고 쓴다
- 효율적인 조회 알고리즘 구현하기
- 여러 단어로 된 개체 다루기
- 예측 매칭과 결합하기
- 사전 갱신 관리하기

지은이: 교육 목적
날짜: 2025
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

# ========================================================================
# 메인
# ========================================================================


class EntityDictionary:
    """
    알려진 개체를 조회하는 개체 사전(가제티어).
    
    가제티어는 특정 유형의 알려진 개체 목록이다.
    보기로:
    - 사람 이름: ["Barack Obama", "Steve Jobs", ...]
    - 회사: ["Apple Inc.", "Microsoft", ...]
    - 장소: ["New York", "Paris", ...]
    """
    
    def __init__(self, entity_type: str):
        """
        개체 사전 초기화.
        
        인수:
            entity_type: 이 사전에 담긴 개체의 유형(PER, ORG, LOC 등)
        """
        self.entity_type = entity_type
        self.entities: Set[str] = set()
        self.entities_lower: Dict[str, str] = {}  # 소문자 -> 원래 표기
        self.multi_word_entities: Set[str] = set()
        
    def add_entity(self, entity: str):
        """사전에 개체 더하기."""
        self.entities.add(entity)
        self.entities_lower[entity.lower()] = entity
        
        # 최적화를 위해 여러 단어로 된 개체를 따로 추적하기
        if len(entity.split()) > 1:
            self.multi_word_entities.add(entity)
    
    def add_entities(self, entities: List[str]):
        """개체를 한꺼번에 여럿 더하기."""
        for entity in entities:
            self.add_entity(entity)
    
    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """텍스트가 사전에 있는지 확인하기."""
        if case_sensitive:
            return text in self.entities
        else:
            return text.lower() in self.entities_lower
    
    def __len__(self):
        """사전에 든 개체의 개수 반환하기."""
        return len(self.entities)


class DictionaryNER:
    """
    사전 기반 개체명 인식 시스템.
    
    개체 사전(가제티어)을 조회해 개체를 식별한다.
    
    과정:
    1. 텍스트를 토큰화한다
    2. 구간마다 사전과 맞춰 본다
    3. 일치한 구간을 개체로 반환한다
    
    장점:
    - 아주 빠르다(O(1) 조회)
    - 사전에 있는 개체에는 정밀도가 완벽하다
    - 새 개체를 넣어 고치기 쉽다
    - 학습할 필요가 없다
    
    단점:
    - 사전에 없는 개체에는 재현율이 0이다
    - 포괄적인 사전이 필요하다
    - 변형과 철자 오류에 약하다
    - 개체의 중의성을 해소하지 못한다
    """
    
    def __init__(self):
        """사전 기반 개체명 인식 초기화."""
        self.dictionaries: Dict[str, EntityDictionary] = {}
        self._initialize_default_dictionaries()
    
    def _initialize_default_dictionaries(self):
        """보기 사전으로 초기화."""
        
        # 사람 이름 사전
        person_dict = EntityDictionary("PER")
        person_dict.add_entities([
            "Steve Jobs", "Bill Gates", "Elon Musk",
            "Barack Obama", "Donald Trump", "Joe Biden",
            "Mark Zuckerberg", "Jeff Bezos", "Tim Cook",
            "Satya Nadella", "Sundar Pichai"
        ])
        self.dictionaries["PER"] = person_dict
        
        # 조직 사전
        org_dict = EntityDictionary("ORG")
        org_dict.add_entities([
            "Apple", "Microsoft", "Google", "Amazon", "Facebook", "Meta",
            "Tesla", "SpaceX", "IBM", "Intel", "Nvidia",
            "Harvard University", "Stanford University", "MIT"
        ])
        self.dictionaries["ORG"] = org_dict
        
        # 장소 사전
        loc_dict = EntityDictionary("LOC")
        loc_dict.add_entities([
            "New York", "Los Angeles", "Chicago", "San Francisco",
            "London", "Paris", "Tokyo", "Beijing", "Sydney",
            "California", "Texas", "Florida",
            "United States", "China", "Japan", "Germany", "France"
        ])
        self.dictionaries["LOC"] = loc_dict
    
    def add_dictionary(self, entity_type: str, entities: List[str]):
        """개체 유형의 사전을 더하거나 갱신하기."""
        if entity_type not in self.dictionaries:
            self.dictionaries[entity_type] = EntityDictionary(entity_type)
        self.dictionaries[entity_type].add_entities(entities)
    
    def extract_entities(self, text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        사전 조회로 개체 추출하기.
        
        인수:
            text: 입력 텍스트
            case_sensitive: 대소문자를 구분해 매칭할지 여부
            
        반환값:
            개체 사전의 목록
        """
        entities = []
        words = text.split()
        
        # 가능한 모든 n-그램 확인하기(단어 5개까지)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i:i+n])
                
                # 모든 사전과 맞춰 보기
                for entity_type, dictionary in self.dictionaries.items():
                    if dictionary.contains(span, case_sensitive):
                        # 원본 텍스트에서의 자리 찾기
                        start = text.find(span)
                        if start != -1:
                            entity = {
                                "text": span,
                                "type": entity_type,
                                "start": start,
                                "end": start + len(span),
                                "confidence": 1.0
                            }
                            entities.append(entity)
        
        # 겹치는 개체와 중첩된 개체 없애기
        entities = self._remove_overlaps(entities)
        return entities
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """겹치는 개체를 없애고 긴 쪽을 남기기."""
        if not entities:
            return []
        
        # 시작 자리로 정렬한 뒤 길이로 정렬(긴 것 먼저)
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        
        filtered = []
        for entity in entities:
            # 이미 더한 개체와 겹치는지 확인하기
            overlaps = False
            for added in filtered:
                if not (entity["end"] <= added["start"] or entity["start"] >= added["end"]):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered


def demonstrate_dictionary_ner():
    """사전 기반 개체명 인식 시연."""
    print("="*70)
    print("Dictionary-Based NER Demonstration")
    print("="*70)
    
    ner = DictionaryNER()
    
    text = "Steve Jobs founded Apple in California. Bill Gates started Microsoft."
    print(f"\nText: {text}")
    
    entities = ner.extract_entities(text)
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")


if __name__ == "__main__":
    demonstrate_dictionary_ner()
```

**출력:**

```
======================================================================
Dictionary-Based NER Demonstration
======================================================================

Text: Steve Jobs founded Apple in California. Bill Gates started Microsoft.

Found 3 entities:
  - Steve Jobs (PER)
  - Apple (ORG)
  - Bill Gates (PER)
```

## 2. 사전의 대소문자와 유형

EntityDictionary는 원문 표기 집합과 소문자 키의 사전을 보관한다. case_sensitive=False이면 소문자화해 포함 여부를 검사하지만 문맥의 중의성을 해결하지는 않는다. Apple이 회사인지 과일인지도 같은 문자열이라는 사실만으로 결정할 수 없다. 같은 소문자 키에 다른 표기를 추가하면 대표 표기가 덮어써질 수 있다.

multi_word_entities도 저장하지만 추출에서 별도로 사용하지 않는다. SequenceMatcher를 import해도 fuzzy matching을 호출하지 않으므로 오탈자·약어·새 표기를 유사도로 복구하는 구현은 아니다. 같은 표현을 여러 유형에 등록했을 때의 우선순위도 사전 순서에 영향을 받는다.

## 3. 공백 분리와 정확 일치의 한계

extract_entities는 text.split()의 단어에서 최대 5개부터 1개까지 후보를 만든다. 구두점을 분리하지 않아 예제의 `California.`·`Microsoft.`는 점이 없는 사전 항목과 일치하지 않는다. 일반적인 토큰화나 구두점 허용 matcher를 구현한 것이 아니다. 여섯 단어 이상의 항목은 등록되어 있어도 이 탐색에서 찾지 못한다.

후보를 공백 하나로 join하므로 원문에 여러 공백·탭·줄바꿈이 있으면 후보 문자열이 원문의 연속 부분 문자열로 존재하지 않을 수 있다. Unicode 정규화·하이픈·소유격 등의 표기 변형도 별도 처리하지 않는다. 등록 문자열이 있어도 토큰 경계·검색 표현과 다르면 미탐이 생긴다.

## 4. text.find와 겹침 제거

일치 후보의 시작점은 현재 단어 위치가 아니라 `text.find(span)`로 원문에서 **첫 등장**을 다시 찾는다. 같은 이름이 두 번 나오면 두 후보가 모두 첫 위치를 가리키고 겹침 제거에서 하나로 줄어 뒤 언급을 놓칠 수 있다. text.find는 단어 경계도 알지 못하므로 앞의 다른 단어 내부에서 같은 부분 문자열을 찾을 가능성도 있다. 각 토큰의 원문 offset을 유지해 후보 위치를 직접 연결해야 한다.

_remove_overlaps는 시작점 오름차순, 같은 시작이면 긴 후보 우선으로 정렬한 뒤 이미 선택한 구간과 겹치는 것을 버린다. 같은 시작점의 최장 일치는 선호하지만 나중에 시작하는 더 긴 겹침 후보보다 항상 긴 것을 선택하는 전역 최장 알고리즘은 아니다. 중첩·동일 문자열의 여러 유형도 이 정책에 따라 제거된다.

confidence=1.0은 정확 사전 일치의 고정값이다. 이름이 문맥에서 올바른 유형인지의 확률이나 데이터셋 정확도 100%라는 뜻이 아니다. 출력은 매칭 결과를 보여 줄 뿐 정답 주석과 비교한 정밀도·재현율이 아니다.

## 5. 실제 데이터에 확장할 때

이름 사전은 도메인·시간에 맞춰 갱신하고 변경 버전·별칭·유형 중의성·신규 이름을 관리해야 한다. 검색 최적화를 위해 trie나 다중 패턴 matcher를 사용할 수 있지만 이 코드가 그런 자료 구조를 사용하지는 않는다. 속도는 문장 길이·후보 길이·유형 수·겹침 검사에도 달렸다.

자동 학습이나 역전파가 없으므로 실패는 그래디언트 폭발보다 토큰 경계·원문 위치·사전 누락·유형 충돌을 먼저 살펴야 한다. 규칙·학습 모델과 결합할 때도 같은 offset·유형·평가 기준을 유지한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
사전에 California가 있어도 예제 문장에서 놓칠 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    text.split()은 California.의 마침표를 남기므로 사전 문자열 California와 정확히 같지 않다. 대소문자를 무시하는 설정이 구두점·토큰화 차이까지 해결하지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Apple이 두 번 등장할 때 현재 text.find 방식은 어떤 문제가 있는가?

</div>

??? success "풀이"

    각 후보의 현재 토큰 위치를 사용하지 않고 항상 첫 부분 문자열 위치를 찾는다. 두 언급이 같은 span이 되어 겹침 제거로 하나만 남을 수 있다. 토큰별 원문 offset을 유지해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
confidence=1.0과 SequenceMatcher import를 각각 어떻게 해석해야 하는가?

</div>

??? success "풀이"

    confidence는 사전 일치의 고정값일 뿐 문맥 유형의 정답 확률이 아니다. SequenceMatcher는 호출하지 않아 fuzzy matching도 구현되지 않았다. 오탈자·별칭·다의어 처리는 별도 설계다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 EntityDictionary 시험 조각이 적절한가? 실제 확인할 명세를 설명하라.

</div>

??? success "풀이"

    EntityDictionary는 문자열 집합이며 forward·parameters·학습 경로가 없다. 미정의 입력·backward를 쓴 조각은 완성 시험이 아니다. 대소문자·구두점·반복 언급·공백 변형·최대 5단어·유형 충돌·겹침 정책과 빈 입력을 확인하는 명세가 필요하다.

    ```python
    def test_entitydictionary():
        model = EntityDictionary(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

사전 기반 NER은 이름 목록뿐 아니라 토큰 경계·원문 offset·반복 언급과 겹침 정책이 정확해야 한다. 현재 구현의 구두점·최대 길이·첫 등장 검색 한계를 알고, 고정 일치 점수를 의미적 확신이나 성능으로 확대하지 않는다.
