"""dictionary_04_dictionary_ner — ch20/sequence_labeling/04_dictionary_ner.md 의 코드를
모듈로 쓸 수 있게 옮겨 놓은 것이다.
"""

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
    demonstrate_dictionary_ner()
