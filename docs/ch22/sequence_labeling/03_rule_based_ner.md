# 규칙 기반 개체명 인식과 패턴 충돌

규칙 기반 NER은 정규식·문맥 조건·우선순위로 언급을 찾는다. 뉴럴 넷트웍·학습·역전파 없이 동작하지만 규칙의 범위와 충돌 해결에 따라 오탐·미탐이 생긴다. 이 예제는 영어 이름·기관·장소·날짜·금액·연락처 등의 후보 규칙을 보여 주며 정확도가 검증된 범용 추출기가 아니다.

## 1. 코드와 기존 출력

??? note "코드"

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
    - 여러 규칙 클래스 아우르기

    핵심 개념:
    - 정규식으로 무늬 짝짓기
    - 대문자 쓰기 무늬
    - 맥락 바탕 규칙
    - 규칙 우선순위 매기기

    지은이: 러닝 목적
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
            entity_type: 이 규칙이 뽑는 것의 클래스
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
            흔한 것 클래스의 붙박이 규칙 모음 첫자리매김.

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
                text: 들임 글월 문자열

            반환값:
                다음 항목을 담은 것 사전의 목록:
                - text: 것의 글
                - type: 것의 클래스
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
                entity_type: 매길 것의 클래스

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
            - 것 클래스 확인하기
            - 아리송함 풀기
            - 기계 러닝 모델을 위한 피처 뽑기

            인수:
                text: 본디 글월
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
                text: 본디 글월
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

        # 보기 2: 서로 다른 것 클래스
        print("\n\n2. Various Entity Types")
        print("-" * 70)

        text2 = ("The meeting is scheduled for January 15, 2025 at 2:30 PM. "
                 "Please contact us at support@company.com or visit www.company.com. "
                 "The discount is 25% off the $199.99 price.")

        print(f"Text: {text2}\n")

        entities2 = ner.extract_entities(text2)

        # 클래스별로 묶기
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
        print("4. Experiment with different priority levels")
    ```

??? note "기존 전체 출력"

    ```
    ======================================================================
    Rule-Based Named Entity Recognition Demonstration
    ======================================================================

    1. Basic Entity Extraction
    ----------------------------------------------------------------------
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California on January 1, 1976. The company is valued at $2.5 trillion.

    Found 8 entities:
    Entity                    Type       Position     Rule
    ----------------------------------------------------------------------
    Apple Inc                 PER        (0:9)        person_full_name
    was founded               PER        (11:22)      person_full_name
    by Steve                  PER        (23:31)      person_full_name
    Jobs in                   PER        (32:39)      person_full_name
    California on             PER        (51:64)      person_full_name
    January 1, 1976           DATE       (65:80)      date_full
    The company               ORG        (82:93)      organization_suffix
    is valued                 PER        (94:103)     person_full_name

    Visualized:
    [Apple Inc]_PER. [was founded]_PER [by Steve]_PER [Jobs in]_PER Cupertino, [California on]_PER [January 1, 1976]_DATE. [The company]_ORG [is valued]_PER at $2.5 trillion.


    2. Various Entity Types
    ----------------------------------------------------------------------
    Text: The meeting is scheduled for January 15, 2025 at 2:30 PM. Please contact us at support@company.com or visit www.company.com. The discount is 25% off the $199.99 price.

    Entities grouped by type:

    PER:
      - The meeting
      - is scheduled
      - Please contact
      - us at
      - off the

    DATE:
      - January 15, 2025

    TIME:
      - 2:30 PM

    CONTACT:
      - support@company.com

    URL:
      - www.company.com

    ORG:
      - The discount is 


    3. Custom Pattern Matching
    ----------------------------------------------------------------------
    Text: My phone numbers are 555-1234 and 555-5678. Call me!

    Found 2 phone numbers:
      - 555-1234
      - 555-5678


    4. Entity Context
    ----------------------------------------------------------------------
    Text: Microsoft CEO Satya Nadella announced new products yesterday in Seattle.


    Entity: Microsoft CEO (PER)
    Left context: 
    Right context: Satya Nadella announced

    Entity: Satya Nadella (PER)
    Left context: Microsoft CEO
    Right context: announced new products

    Entity: new products (LOC)
    Left context: Satya Nadella announced
    Right context: yesterday in Seattle.

    Entity: in Seattle (PER)
    Left context: 
    Right context: 


    5. Adding Custom Rules
    ----------------------------------------------------------------------
    Text: I bought an iPhone 15 and a MacBook Pro. Also considering a Tesla Model 3.

    Found 2 products:
      - iPhone
      - MacBook


    6. Handling Ambiguous Cases
    ----------------------------------------------------------------------
    Ambiguous cases (simple rules may incorrectly tag these):

    Text: Washington visited Washington.
      Tagged: 'Washington visited' as PER
      Note: Context-aware rules or ML models needed for disambiguation

    Text: I love Python programming.
      Tagged: 'love Python' as PER
      Note: Context-aware rules or ML models needed for disambiguation

    Text: Apple released new Apple products.
      Tagged: 'Apple released' as PER
      Tagged: 'new Apple' as LOC
      Note: Context-aware rules or ML models needed for disambiguation


    7. Rule-Based NER Characteristics
    ----------------------------------------------------------------------

    Strengths:
      ✓ Very fast (no model inference)
      ✓ No training data needed
      ✓ Highly interpretable (can see exactly why entity was extracted)
      ✓ Easy to add domain-specific rules
      ✓ High precision for well-defined patterns
      ✓ Deterministic (same input always gives same output)

    Weaknesses:
      ✗ Low recall (misses variations not covered by rules)
      ✗ Cannot generalize to unseen patterns
      ✗ Requires manual rule creation and maintenance
      ✗ Difficult to handle ambiguity
      ✗ Rule conflicts can be hard to debug
      ✗ Does not learn from data

    Best used for:
      • Well-defined patterns (dates, emails, URLs)
      • Domain-specific entities with clear indicators
      • Quick prototyping before building ML models
      • Combining with ML models (rule-based + ML hybrid)
      • High-precision extraction where recall is less critical

    ======================================================================
    Demonstration complete!
    ======================================================================

    Next steps:
    1. Try modifying existing rules
    2. Add custom rules for your domain
    3. Combine with dictionary-based NER (next module)
    4. Experiment with different priority levels
    ```

## 2. 정규식이 실제로 검사하는 조건

모든 Rule은 re.IGNORECASE로 컴파일된다. 따라서 `[A-Z][a-z]+`처럼 대문자 이름을 의도한 패턴도 소문자 단어에 맞을 수 있다. 두 단어가 연속된 일반 구절이 PER로 잡히는 문제를 뉴럴 넷트웍 문맥 이해와 혼동하지 않는다. 조직 패턴의 `[A-Za-z\s&]+`도 넓고 탐욕적이라 뒤의 불필요한 단어·공백까지 포함할 수 있다.

`\b`는 단어 문자와 비단어 문자 사이의 경계이지 모든 기호의 시작·끝이 아니다. 공백 다음 `$` 앞은 둘 다 비단어라 money 패턴 맨 앞의 경계가 기대대로 맞지 않을 수 있다. `%` 다음 공백, `Inc.`의 마침표 다음 공백도 같은 종류의 문제가 있다. 패턴 모양만 보고 `$199.99`·`25%`·기관 접미사를 항상 추출한다고 설명하면 안 된다.

날짜·시간의 정규식은 숫자 모양을 찾을 뿐 실제 달력·시계의 유효성을 검증하지 않는다. 금액도 소수 자릿수·규모 표현·통화 관례를 제한적으로 처리한다. 제품 예의 `Tesla Model [A-Z]`는 숫자3을 허용하지 않으므로 Model 3 전체를 잡는 규칙이 아니다. iPhone·MacBook 대안도 뒤의15·Pro까지 자동 포함하지 않는다.

## 3. 우선순위와 비중첩 선택

규칙은 priority 내림차순이며 같은 우선순위는 등록 순서를 유지한다. extract_entities는 높은 규칙부터 처리하고 이미 사용한 문자 위치와 하나라도 겹치면 뒤 후보를 버린다. 최종 결과만 원문 시작점 순서로 정렬한다. 이것은 전체 후보에서 가장 긴 개체나 전역 최적 집합을 고르는 알고리즘이 아니다. 중첩 주석도 제거한다.

문맥 조건은 후보 앞뒤50문자의 소문자 문자열에서 키워드 중 **하나라도** 부분 문자열로 존재하는지 확인한다. 단어 경계·형태·구문 관계나 모든 조건의 동시 충족을 검사하지 않는다. 잘못된 큰 후보가 먼저 선택되면 이후 더 정확한 후보를 막을 수 있어 규칙별 충돌을 함께 검토해야 한다.

confidence=1/(10−priority)는 우선순위로 만든 수치일 뿐 정답 확률이 아니다. priority10이면0으로 나누고10보다 크면 음수가 될 수 있어 일반 범위의 신뢰도로 해석할 수 없다. extract_by_pattern의0.8도 고정값이며 해당 함수는 기본 규칙과의 충돌 해결을 따로 수행하지 않는다.

## 4. 문맥 표시와 실제 언급 위치

get_entity_context는 entity.start를 사용하지 않고 공백 분리한 단어 목록에서 같은 문자열의 첫 등장을 찾는다. 같은 이름이 여러 번 나오면 다른 언급도 첫 문맥을 반환할 수 있고, 구두점이 붙어 토큰이 다르면 못 찾을 수 있다. 원문 offset을 토큰 구간에 연결하는 과정이 필요하다.

시각화는 주어진 비중첩 span을 원문에 표시한다. 임의로 겹친 결과를 전달하면 표시 순서가 어긋날 수 있다. 결과에 적용된 rule 이름은 규칙의 출처를 설명하지만 의미적 정답이나 인과적 설명의 충실성을 보장하지 않는다.

## 5. 평가와 확장

규칙은 학습 데이터 없이 만들 수 있지만 설계·유지·검증 자료는 필요하다. 같은 문자열에서도 규칙을 바꾸면 출력이 달라지며 많은 정규식·긴 텍스트·겹침 검사 비용 때문에 항상 빠르다고 보장할 수 없다. 규칙별 FP·FN·경계 오류를 별도 주석 자료에서 평가하고 수정에 사용한 자료와 최종 시험을 구분한다.

이 파일에는 자동미분·생성·모델 최적화가 없다. 규칙과 사전·학습 모델을 결합할 때도 offset·유형·중첩 폴리시와 점수의 의미를 맞춰야 한다. 코드·출력은 현재 규칙의 작동을 읽는 자료이며 다시 실행할 필요는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
사람 이름 패턴의 대문자 조건이 실제로 강제되는가?

</div>

??? success "풀이"

    모든 패턴을 IGNORECASE로 컴파일하므로 그렇지 않다. 소문자 두 단어도 이름 패턴에 맞을 수 있어 대문자 기반 사람 인식으로 설명하면 안 된다. 언어·문맥과 별도 평가가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정규식의 word boundary가 통화·퍼센트 기호에 적합하지 않을 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    word boundary는 단어 문자와 비단어 문자 사이를 검사한다. 공백과 달러 기호, 퍼센트 기호와 공백은 둘 다 비단어여서 해당 위치에 경계가 없을 수 있다. 실제 기호·구두점·문장 경계를 고려한 패턴 계약이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 이름이 두 번 등장할 때 get_entity_context와 confidence의 한계를 설명하라.

</div>

??? success "풀이"

    문맥 함수는 offset이 아니라 같은 단어 열의 첫 등장을 찾아 두 언급을 혼동할 수 있다. confidence도 우선순위 산식일 뿐 확률이 아니고 priority=10에서는 정의되지 않는다. 원문 위치와 점수의 의미를 분리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 Rule 시험 조각이 이 정규식 예제에 맞는가?

</div>

??? success "풀이"

    Rule은 데이터와 컴파일된 패턴을 보관하며 뉴럴 넷트웍 forward·parameters가 없다. 아래 미정의 입력·backward 검사는 실행 가능한 시험이 아니다. 소문자 일반 구절·기호 경계·반복 언급·중첩 우선순위·유효하지 않은 정규식·빈 문자열 등을 명세로 확인하는 것이 적절하다.

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


## 정리하며

규칙 기반 NER은 패턴의 실제 대소문자·단어 경계와 우선순위 충돌을 이해해야 한다. 고정 산식 점수를 확률로 취급하지 않고 원문 offset에 맞춘 문맥과 독립 평가로 오탐·미탐을 확인한다.
