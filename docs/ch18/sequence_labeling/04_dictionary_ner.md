# 사전 바탕 이름 알아보기

사전 바탕 이름 알아보기. 이 단원은 것 사전을 써서 사전/지명록 바탕 이름 알아보기를 짠다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
사전 바탕 이름 알아보기
==========================================

이 단원은 것 사전과 찾기 표를 써서 사전/지명록 바탕 이름 알아보기를 짠다.
이 방식은 단순하고 빠르며 또렷하게 정해진 것 갈래에 잘 듣는다.
것의 목록.

학습 목표:
- 것 사전(지명록)을 세우고 쓴다
- 효율적인 찾기 알고리즘 짜기
- 낱말 여럿짜리 것 다루기
- 어림 짝짓기와 아우르기
- 사전 고침 다스리기

지은이: 배움 목적
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
    알려진 것을 찾아보는 것 사전(지명록).
    
    지명록은 특정 갈래의 알려진 것의 목록이다.
    보기로:
    - 사람 이름: ["Barack Obama", "Steve Jobs", ...]
    - 회사: ["Apple Inc.", "Microsoft", ...]
    - 자리: ["New York", "Paris", ...]
    """
    
    def __init__(self, entity_type: str):
        """
        것 사전 첫자리매김.
        
        인수:
            entity_type: 이 사전에 담긴 것의 갈래(PER, ORG, LOC 등)
        """
        self.entity_type = entity_type
        self.entities: Set[str] = set()
        self.entities_lower: Dict[str, str] = {}  # 소문자 -> 본디
        self.multi_word_entities: Set[str] = set()
        
    def add_entity(self, entity: str):
        """사전에 것 더하기."""
        self.entities.add(entity)
        self.entities_lower[entity.lower()] = entity
        
        # 다듬기를 위해 낱말 여럿짜리 것을 따로 좇기
        if len(entity.split()) > 1:
            self.multi_word_entities.add(entity)
    
    def add_entities(self, entities: List[str]):
        """것을 한꺼번에 여럿 더하기."""
        for entity in entities:
            self.add_entity(entity)
    
    def contains(self, text: str, case_sensitive: bool = False) -> bool:
        """글이 사전에 있는지 살피기."""
        if case_sensitive:
            return text in self.entities
        else:
            return text.lower() in self.entities_lower
    
    def __len__(self):
        """사전에 든 것의 개수 돌려주기."""
        return len(self.entities)


class DictionaryNER:
    """
    사전 바탕 이름 알아보기 체계.
    
    것 사전(지명록)을 찾아 것을 가려낸다.
    
    과정:
    1. 글을 토막낸다
    2. 구간마다 사전과 맞춰 본다
    3. 맞은 것을 것으로 돌려준다
    
    이점:
    - 아주 빠르다(O(1) 찾기)
    - 사전에 있는 것에는 정밀도가 완벽하다
    - 새 것을 넣어 고치기 쉽다
    - 익힐 필요가 없다
    
    나쁜 점:
    - 사전에 없는 것에는 재현율이 0이다
    - 두루 갖춘 사전이 필요하다
    - 달라진 꼴과 맞춤법 어긋남에 약하다
    - 것의 아리송함을 풀지 못한다
    """
    
    def __init__(self):
        """사전 바탕 이름 알아보기 첫자리매김."""
        self.dictionaries: Dict[str, EntityDictionary] = {}
        self._initialize_default_dictionaries()
    
    def _initialize_default_dictionaries(self):
        """보기 사전으로 첫자리매김."""
        
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
        
        # 자리 사전
        loc_dict = EntityDictionary("LOC")
        loc_dict.add_entities([
            "New York", "Los Angeles", "Chicago", "San Francisco",
            "London", "Paris", "Tokyo", "Beijing", "Sydney",
            "California", "Texas", "Florida",
            "United States", "China", "Japan", "Germany", "France"
        ])
        self.dictionaries["LOC"] = loc_dict
    
    def add_dictionary(self, entity_type: str, entities: List[str]):
        """것 갈래의 사전을 더하거나 고치기."""
        if entity_type not in self.dictionaries:
            self.dictionaries[entity_type] = EntityDictionary(entity_type)
        self.dictionaries[entity_type].add_entities(entities)
    
    def extract_entities(self, text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        사전 찾기로 것 뽑기.
        
        인수:
            text: 입력 텍스트
            case_sensitive: 대소문자를 가려 짝지을지 여부
            
        반환값:
            것 사전의 목록
        """
        entities = []
        words = text.split()
        
        # 가능한 모든 n-그램 살피기(낱말 5개까지)
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                span = " ".join(words[i:i+n])
                
                # 모든 사전과 맞춰 보기
                for entity_type, dictionary in self.dictionaries.items():
                    if dictionary.contains(span, case_sensitive):
                        # 본디 글에서의 자리 찾기
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
        
        # 겹치는 것과 포개진 것 없애기
        entities = self._remove_overlaps(entities)
        return entities
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """겹치는 것을 없애고 긴 쪽을 남기기."""
        if not entities:
            return []
        
        # 시작 자리로 정렬한 뒤 길이로 정렬(긴 것 먼저)
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        
        filtered = []
        for entity in entities:
            # 이미 더한 것과 겹치는지 살피기
            overlaps = False
            for added in filtered:
                if not (entity["end"] <= added["start"] or entity["start"] >= added["end"]):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered


def demonstrate_dictionary_ner():
    """사전 바탕 이름 알아보기 보이기."""
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
    demonstrate_dictionary_ner()```

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 차례 이름표 붙이기에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
사전 바탕 이름 알아보기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_entitydictionary():
        model = EntityDictionary(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 사전 바탕 이름 알아보기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `EntityDictionary`, `DictionaryNER`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
