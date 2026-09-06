# 규칙 바탕 이름 알아보기

규칙 바탕 이름 알아보기. 이 단원은 무늬 짝짓기와 말 규칙을 써서 규칙 바탕 이름 알아보기를 짠다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 코드

```python
"""
규칙 바탕 이름 알아보기
====================================

이 단원은 무늬 짝짓기와 말 규칙을 써서 규칙 바탕 이름 알아보기를 짠다.
규칙 바탕 체계는 빠르고 읽어 내기 쉬우며 무늬가 또렷한 분야에서 잘 된다.

학습 목표:
- 무늬 바탕 것 뽑기 짜기
- 이름 알아보기에 정규식 쓰기
- 규칙 바탕 체계의 센 점과 한계를 이해한다
- 여러 규칙 갈래 아우르기

핵심 개념:
- 정규식으로 무늬 짝짓기
- 대문자 쓰기 무늬
- 맥락 바탕 규칙
- 규칙 우선순위 매기기

지은이: 배움 목적
날짜: 2025
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

# ========================================================================
# 메인
# ========================================================================


@dataclass
class Rule:
    """
    뽑기 규칙 하나를 나타낸다.
    
    속성:
        name: 규칙의 이름
        pattern: 짝지을 정규식 무늬
        entity_type: 이 규칙이 뽑는 것의 갈래
        priority: 규칙의 우선순위(높을수록 먼저 쓴다)
        context_required: 반드시 있어야 하는 맥락 낱말(없어도 됨)
    """
    name: str
    pattern: str
    entity_type: str
    priority: int = 1
    context_required: List[str] = None
    
    def __post_init__(self):
        """첫자리매김 뒤에 정규식 무늬 컴파일하기."""
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)


class RuleBasedNER:
    """
    규칙 바탕 이름 알아보기 체계.
    
    이 체계는 다음을 비롯한 손수 만든 규칙을 쓴다:
    1. 정규식 무늬
    2. 대문자 쓰기 무늬
    3. 맥락 바탕 규칙
    4. 사전 찾기(규칙 짝짓기와 함께)
    
    이점:
    - 빠르고 효율적이다
    - 익힘 자료가 필요 없다
    - 아주 읽어 내기 쉽다
    - 분야별 규칙을 쓸 수 있다
    - 또렷한 무늬에는 정밀도가 높다
    
    나쁜 점:
    - 재현율이 낮다(달라진 꼴을 놓친다)
    - 규칙을 손수 만들어야 한다
    - 규칙이 많으면 건사하기 어렵다
    - 본 적 없는 무늬에 두루 통하지 못한다
    """
    
    def __init__(self):
        """규칙 바탕 이름 알아보기 체계 첫자리매김."""
        self.rules: List[Rule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """
        흔한 것 갈래의 붙박이 규칙 모음 첫자리매김.
        
        이 규칙이 다루는 것:
        - 사람 이름(대문자 쓰기 무늬)
        - 조직(회사 뒷가지)
        - 자리(자리를 가리키는 말)
        - 날짜와 때
        - 돈과 백분율
        - 전자우편과 URL
        """
        
        # 사람 이름 무늬
        # 무늬: 대문자 낱말 뒤에 대문자 낱말
        # 보기: "John Smith", "Mary Johnson"
        self.add_rule(Rule(
            name="person_full_name",
            pattern=r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
            entity_type="PER",
            priority=3
        ))
        
        # 조직 무늬
        # 무늬: 회사 뒷가지가 붙은 대문자 낱말
        # 보기: "Apple Inc.", "Microsoft Corporation", "Google LLC"
        self.add_rule(Rule(
            name="organization_suffix",
            pattern=r'\b([A-Z][A-Za-z\s&]+)\s+(Inc\.|Corp\.|Corporation|LLC|Ltd\.|Limited|Company|Co\.)\b',
            entity_type="ORG",
            priority=5
        ))
        
        # 무늬: "The" 뒤에 대문자로 시작하는 낱말
        # 보기: "The New York Times", "The Washington Post"
        self.add_rule(Rule(
            name="organization_the",
            pattern=r'\bThe\s+([A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|Times|Post|Bank|University)?)\b',
            entity_type="ORG",
            priority=4
        ))
        
        # 자리 무늬
        # 무늬: 자리를 가리키는 핵심 낱말
        # 보기: "New York City", "San Francisco", "Mount Everest"
        self.add_rule(Rule(
            name="location_indicators",
            pattern=r'\b((?:New|San|Los|Las)\s+[A-Z][a-z]+(?:\s+(?:City|Beach|Angeles|Vegas))?|'
                   r'Mount\s+[A-Z][a-z]+|Lake\s+[A-Z][a-z]+|'
                   r'[A-Z][a-z]+\s+(?:River|Ocean|Sea|Mountain))\b',
            entity_type="LOC",
            priority=4
        ))
        
        # 무늬: 주와 나라
        # 간추린 목록 — 실전에서는 두루 갖춘 지명록을 쓴다
        self.add_rule(Rule(
            name="location_places",
            pattern=r'\b(California|Texas|New\s+York|Florida|Illinois|'
                   r'London|Paris|Tokyo|Beijing|Sydney|Berlin|'
                   r'United\s+States|USA|UK|China|Japan|Germany)\b',
            entity_type="LOC",
            priority=3
        ))
        
        # 날짜 무늬
        # 무늬: 여러 날짜 꼴
        # 보기: "January 1, 2020", "01/01/2020", "2020-01-01"
        self.add_rule(Rule(
            name="date_full",
            pattern=r'\b((?:January|February|March|April|May|June|July|August|'
                   r'September|October|November|December)\s+\d{1,2},?\s+\d{4}|'
                   r'\d{1,2}/\d{1,2}/\d{2,4}|'
                   r'\d{4}-\d{2}-\d{2})\b',
            entity_type="DATE",
            priority=5
        ))
        
        # 때 무늬
        # 보기: "3:30 PM", "14:00", "noon"
        self.add_rule(Rule(
            name="time",
            pattern=r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|noon|midnight)\b',
            entity_type="TIME",
            priority=5
        ))
        
        # 돈 무늬
        # 보기: "$100", "$1,000.00", "€50"
        self.add_rule(Rule(
            name="money",
            pattern=r'\b([$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|'
                   r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|euros|pounds|yen))\b',
            entity_type="MONEY",
            priority=5
        ))
        
        # 백분율 무늬
        # 보기: "25%", "3.14%"
        self.add_rule(Rule(
            name="percentage",
            pattern=r'\b(\d+(?:\.\d+)?%|'
                   r'\d+(?:\.\d+)?\s*percent)\b',
            entity_type="PERCENT",
            priority=5
        ))
        
        # 전자우편 무늬
        # 보기: "user@example.com"
        self.add_rule(Rule(
            name="email",
            pattern=r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
            entity_type="CONTACT",
            priority=6
        ))
        
        # URL 무늬
        # 보기: "https://www.example.com", "www.example.com"
        self.add_rule(Rule(
            name="url",
            pattern=r'\b((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b',
            entity_type="URL",
            priority=6
        ))
    
    def add_rule(self, rule: Rule):
        """
        체계에 새 규칙 더하기.
        
        더 좁은 규칙이 넓은 규칙보다 먼저 쓰이도록
        규칙을 우선순위로 정렬한다(높은 것 먼저).
        
        인수:
            rule: 더할 Rule 개체
        """
        self.rules.append(rule)
        # 규칙을 우선순위로 정렬(높은 것 먼저)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        모든 규칙을 써서 글에서 것 뽑기.
        
        과정:
        1. 우선순위 차례로 규칙을 쓴다
        2. 규칙에 맞을 때마다 것을 만든다
        3. 겹치는 것 다루기(우선순위 높은 쪽을 남긴다)
        4. 정렬한 것의 목록을 돌려준다
        
        인수:
            text: Input text string
            
        반환값:
            다음 항목을 담은 것 사전의 목록:
            - text: 것의 글
            - type: 것의 갈래
            - start: 시작 자리
            - end: 끝 자리
            - rule: 맞은 규칙의 이름
            - confidence: 믿음도 점수
            
        보기:
            >>> ner = RuleBasedNER()
            >>> text = "Apple Inc. was founded by Steve Jobs."
            >>> entities = ner.extract_entities(text)
            >>> for entity in entities:
            ...     print(f"{entity['text']}: {entity['type']}")
            Apple Inc.: ORG
            Steve Jobs: PER
        """
        entities = []
        used_spans = set()  # 겹침을 다루려 쓴 글자 구간 좇기
        
        # 우선순위 차례로 규칙 쓰기
        for rule in self.rules:
            # 이 규칙에 맞는 곳 모두 찾기
            for match in rule.compiled_pattern.finditer(text):
                # 맞은 글과 자리 얻기
                matched_text = match.group(0)
                start = match.start()
                end = match.end()
                
                # 이 구간이 이미 뽑은 것과 겹치는지 살피기
                span_range = range(start, end)
                if any(pos in used_spans for pos in span_range):
                    # 겹치는 것 건너뛰기(우선순위가 낮은 규칙)
                    continue
                
                # 정해졌으면 맥락 요건 살피기
                if rule.context_required:
                    # 맞은 곳 둘레의 맥락 창 뽑기
                    context_start = max(0, start - 50)
                    context_end = min(len(text), end + 50)
                    context = text[context_start:context_end].lower()
                    
                    # 필요한 맥락 낱말이 있는지 살피기
                    if not any(word.lower() in context for word in rule.context_required):
                        continue
                
                # 것 만들기
                entity = {
                    'text': matched_text,
                    'type': rule.entity_type,
                    'start': start,
                    'end': end,
                    'rule': rule.name,
                    'confidence': 1.0 / (10 - rule.priority)  # 우선순위가 높을수록 믿음도가 높다
                }
                
                entities.append(entity)
                
                # 구간을 썼다고 표시하기
                used_spans.update(span_range)
        
        # 것을 자리로 정렬
        entities.sort(key=lambda e: e['start'])
        
        return entities
    
    def extract_by_pattern(self, text: str, pattern: str, entity_type: str) -> List[Dict]:
        """
        맞춤 무늬로 것 뽑기.
        
        덕분에 규칙을 아예 더하지 않고도 무늬를 빠르게 시험할 수 있다.
        
        인수:
            text: 입력 텍스트
            pattern: 정규식 무늬 글자열
            entity_type: 매길 것의 갈래
            
        반환값:
            뽑은 것의 목록
            
        보기:
            >>> ner = RuleBasedNER()
            >>> text = "Call me at 555-1234 or 555-5678"
            >>> entities = ner.extract_by_pattern(
            ...     text, r'\d{3}-\d{4}', 'PHONE'
            ... )
        """
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        entities = []
        
        for match in compiled_pattern.finditer(text):
            entity = {
                'text': match.group(0),
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'rule': 'custom_pattern',
                'confidence': 0.8
            }
            entities.append(entity)
        
        return entities
    
    def get_entity_context(self, text: str, entity: Dict, window_size: int = 5) -> Dict:
        """
        것 둘레의 맥락 낱말 얻기.
        
        맥락은 다음에 쓸모 있다:
        - 것 갈래 확인하기
        - 아리송함 풀기
        - 기계 배움 모델을 위한 특징 뽑기
        
        인수:
            text: Original text
            entity: 것 사전
            window_size: 양쪽에 넣을 낱말의 개수
            
        반환값:
            left_context, entity_text, right_context를 담은 사전
        """
        # 단순 낱말 바탕 토막내기
        words = text.split()
        entity_text = entity['text']
        
        # 낱말 목록에서 것 찾기
        entity_words = entity_text.split()
        entity_start_word = None
        
        for i in range(len(words) - len(entity_words) + 1):
            if words[i:i+len(entity_words)] == entity_words:
                entity_start_word = i
                break
        
        if entity_start_word is None:
            return {'left_context': [], 'entity': entity_text, 'right_context': []}
        
        # 맥락 뽑기
        left_start = max(0, entity_start_word - window_size)
        left_context = words[left_start:entity_start_word]
        
        right_end = min(len(words), entity_start_word + len(entity_words) + window_size)
        right_context = words[entity_start_word + len(entity_words):right_end]
        
        return {
            'left_context': left_context,
            'entity': entity_text,
            'right_context': right_context
        }
    
    def visualize_entities(self, text: str, entities: List[Dict]):
        """
        것 표시를 곁들여 글 찍기.
        
        글 속의 것을 눈으로 볼 수 있게 나타낸다.
        
        인수:
            text: Original text
            entities: 것 사전의 목록
        """
        # 표시한 판 만들기
        result = []
        last_pos = 0
        
        for entity in sorted(entities, key=lambda e: e['start']):
            # 것 앞의 글 더하기
            result.append(text[last_pos:entity['start']])
            
            # 표시한 것 더하기
            result.append(f"[{entity['text']}]_{entity['type']}")
            
            last_pos = entity['end']
        
        # 남은 글 더하기
        result.append(text[last_pos:])
        
        print(''.join(result))


def demonstrate_rule_based_ner():
    """
    규칙 바탕 이름 알아보기를 두루 보이기.
    """
    print("="*70)
    print("Rule-Based Named Entity Recognition Demonstration")
    print("="*70)
    
    # 이름 알아보기 체계 첫자리매김
    ner = RuleBasedNER()
    
    # 보기 1: 기본 것 뽑기
    print("\n1. Basic Entity Extraction")
    print("-" * 70)
    
    text1 = ("Apple Inc. was founded by Steve Jobs in Cupertino, California "
             "on January 1, 1976. The company is valued at $2.5 trillion.")
    
    print(f"Text: {text1}\n")
    
    entities1 = ner.extract_entities(text1)
    
    print(f"Found {len(entities1)} entities:")
    print(f"{'Entity':<25} {'Type':<10} {'Position':<12} {'Rule'}")
    print("-" * 70)
    for entity in entities1:
        pos = f"({entity['start']}:{entity['end']})"
        print(f"{entity['text']:<25} {entity['type']:<10} {pos:<12} {entity['rule']}")
    
    # 시각화한다
    print("\nVisualized:")
    ner.visualize_entities(text1, entities1)
    
    # 보기 2: 서로 다른 것 갈래
    print("\n\n2. Various Entity Types")
    print("-" * 70)
    
    text2 = ("The meeting is scheduled for January 15, 2025 at 2:30 PM. "
             "Please contact us at support@company.com or visit www.company.com. "
             "The discount is 25% off the $199.99 price.")
    
    print(f"Text: {text2}\n")
    
    entities2 = ner.extract_entities(text2)
    
    # 갈래별로 묶기
    by_type = {}
    for entity in entities2:
        if entity['type'] not in by_type:
            by_type[entity['type']] = []
        by_type[entity['type']].append(entity['text'])
    
    print("Entities grouped by type:")
    for entity_type, texts in by_type.items():
        print(f"\n{entity_type}:")
        for text in texts:
            print(f"  - {text}")
    
    # 보기 3: 맞춤 무늬
    print("\n\n3. Custom Pattern Matching")
    print("-" * 70)
    
    text3 = "My phone numbers are 555-1234 and 555-5678. Call me!"
    
    print(f"Text: {text3}\n")
    
    # 맞춤 무늬로 전화번호 뽑기
    phone_pattern = r'\d{3}-\d{4}'
    phone_entities = ner.extract_by_pattern(text3, phone_pattern, 'PHONE')
    
    print(f"Found {len(phone_entities)} phone numbers:")
    for entity in phone_entities:
        print(f"  - {entity['text']}")
    
    # 보기 4: 맥락 뽑기
    print("\n\n4. Entity Context")
    print("-" * 70)
    
    text4 = "Microsoft CEO Satya Nadella announced new products yesterday in Seattle."
    entities4 = ner.extract_entities(text4)
    
    print(f"Text: {text4}\n")
    
    for entity in entities4:
        context = ner.get_entity_context(text4, entity, window_size=3)
        print(f"\nEntity: {entity['text']} ({entity['type']})")
        print(f"Left context: {' '.join(context['left_context'])}")
        print(f"Right context: {' '.join(context['right_context'])}")
    
    # 보기 5: 맞춤 규칙 더하기
    print("\n\n5. Adding Custom Rules")
    print("-" * 70)
    
    # 상품 이름을 위한 맞춤 규칙 더하기
    product_rule = Rule(
            name="product_names",
            pattern=r'\b(iPhone|iPad|MacBook|Windows|Android|Tesla Model [A-Z])\b',
            entity_type="PRODUCT",
            priority=6
        )
    
    ner.add_rule(product_rule)
    
    text5 = "I bought an iPhone 15 and a MacBook Pro. Also considering a Tesla Model 3."
    
    print(f"Text: {text5}\n")
    
    entities5 = ner.extract_entities(text5)
    products = [e for e in entities5 if e['type'] == 'PRODUCT']
    
    print(f"Found {len(products)} products:")
    for entity in products:
        print(f"  - {entity['text']}")
    
    # 보기 6: 아리송함 다루기
    print("\n\n6. Handling Ambiguous Cases")
    print("-" * 70)
    
    ambiguous_texts = [
        "Washington visited Washington.",  # 사람과 자리
        "I love Python programming.",     # 프로그래밍 말과 뱀
        "Apple released new Apple products."  # 회사와 과일
    ]
    
    print("Ambiguous cases (simple rules may incorrectly tag these):")
    for text in ambiguous_texts:
        print(f"\nText: {text}")
        entities = ner.extract_entities(text)
        if entities:
            for entity in entities:
                print(f"  Tagged: '{entity['text']}' as {entity['type']}")
        else:
            print("  No entities found")
        print("  Note: Context-aware rules or ML models needed for disambiguation")
    
    # 보기 7: 성능의 성질
    print("\n\n7. Rule-Based NER Characteristics")
    print("-" * 70)
    
    print("\nStrengths:")
    print("  ✓ Very fast (no model inference)")
    print("  ✓ No training data needed")
    print("  ✓ Highly interpretable (can see exactly why entity was extracted)")
    print("  ✓ Easy to add domain-specific rules")
    print("  ✓ High precision for well-defined patterns")
    print("  ✓ Deterministic (same input always gives same output)")
    
    print("\nWeaknesses:")
    print("  ✗ Low recall (misses variations not covered by rules)")
    print("  ✗ Cannot generalize to unseen patterns")
    print("  ✗ Requires manual rule creation and maintenance")
    print("  ✗ Difficult to handle ambiguity")
    print("  ✗ Rule conflicts can be hard to debug")
    print("  ✗ Does not learn from data")
    
    print("\nBest used for:")
    print("  • Well-defined patterns (dates, emails, URLs)")
    print("  • Domain-specific entities with clear indicators")
    print("  • Quick prototyping before building ML models")
    print("  • Combining with ML models (rule-based + ML hybrid)")
    print("  • High-precision extraction where recall is less critical")


if __name__ == "__main__":
    # 보여 주기를 돌린다
    demonstrate_rule_based_ner()
    
    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Try modifying existing rules")
    print("2. Add custom rules for your domain")
    print("3. Combine with dictionary-based NER (next module)")
    print("4. Experiment with different priority levels")```

## 논의

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
규칙 바탕 이름 알아보기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_rule():
        model = Rule(...)
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
