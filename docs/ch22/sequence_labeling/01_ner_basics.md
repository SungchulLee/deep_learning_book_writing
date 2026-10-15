# 개체명 인식 기초 예제: 언급·토큰·문자 위치

이 예제는 직접 지정한 개체 주석을 데이터 구조로 보관하고, 정규식 토큰화 후 IOB2 레이블을 부여한다. 신경망을 학습하거나 새 문장에서 개체를 자동으로 찾는 모델은 아니다. 출력의 ‘Entities found’는 주어진 주석을 표시한 것이며 추론 정확도가 아니다.

## 1. 코드와 기존 출력

??? note "코드"

    ```python
    """
    이름 알아보기 — 기초
    =================================

    이 단원은 이름 알아보기(NER)의 근본 개념을 소개한다.
    것이 무엇인지, 흔한 것 갈래, 이름 알아보기를 위한 기본 글 앞손질을 다룬다.

    학습 목표:
    - 이름 있는 것이 무엇인지 이해한다
    - 흔한 것 갈래와 이름표 방식을 배운다
    - 이름 알아보기를 위한 기본 글 앞손질 살펴보기
    - 이름 알아보기의 어려움을 이해한다

    지은이: 배움 목적
    날짜: 2025
    """

    import re
    from typing import List, Tuple, Dict
    from collections import defaultdict

    # ========================================================================
    # 메인
    # ========================================================================


    class EntityType:
        """
        이름 알아보기에 쓰는 흔한 것 갈래의 열거.

        표준 CoNLL 것 갈래:
        - PER: 사람 이름
        - ORG: 조직 이름
        - LOC: 자리 이름
        - MISC: 그 밖의 것

        넓힌 것 갈래:
        - DATE: 날짜 표현
        - TIME: 때 표현
        - MONEY: 돈의 값
        - PERCENT: 백분율
        """

        # 표준 것 갈래
        PERSON = "PER"
        ORGANIZATION = "ORG"
        LOCATION = "LOC"
        MISCELLANEOUS = "MISC"

        # 넓힌 것 갈래
        DATE = "DATE"
        TIME = "TIME"
        MONEY = "MONEY"
        PERCENT = "PERCENT"
        PRODUCT = "PRODUCT"
        EVENT = "EVENT"

        @staticmethod
        def all_types():
            """정해 둔 것 갈래를 모두 돌려주기."""
            return [
                EntityType.PERSON,
                EntityType.ORGANIZATION,
                EntityType.LOCATION,
                EntityType.MISCELLANEOUS,
                EntityType.DATE,
                EntityType.TIME,
                EntityType.MONEY,
                EntityType.PERCENT,
                EntityType.PRODUCT,
                EntityType.EVENT
            ]


    class Entity:
        """
        글 속의 이름 있는 것을 나타낸다.

        것은 다음으로 이루어진다:
        - text: 실제 글 구간
        - entity_type: 것의 갈래(PER, ORG, LOC 등)
        - start: 본디 글에서의 시작 글자 자리
        - end: 본디 글에서의 끝 글자 자리
        - confidence: 믿음도 점수(없어도 됨, 기계 배움 모델용)
        """

        def __init__(self, text: str, entity_type: str, start: int, end: int, confidence: float = 1.0):
            """
            Entity 첫자리매김.

            인수:
                text: 개체가 걸친 글월 마디
                entity_type: 것의 갈래(PER, ORG, LOC 등)
                start: 시작 글자 자리
                end: 끝 글자 자리
                confidence: 믿음도 점수(0~1), 붙박이 1.0
            """
            self.text = text
            self.entity_type = entity_type
            self.start = start
            self.end = end
            self.confidence = confidence

        def __repr__(self):
            """것의 글자열 나타냄."""
            return f"Entity(text='{self.text}', type='{self.entity_type}', span=({self.start}, {self.end}))"

        def __eq__(self, other):
            """
            두 것이 같은지 살피기.
            두 것은 구간과 갈래가 같으면 같다.
            """
            if not isinstance(other, Entity):
                return False
            return (self.start == other.start and 
                    self.end == other.end and 
                    self.entity_type == other.entity_type)

        def overlaps(self, other: 'Entity') -> bool:
            """
            이 것이 다른 것과 겹치는지 살피기.

            인수:
                other: 다른 Entity 개체

            반환값:
                것이 겹치면 True, 아니면 False
            """
            return not (self.end <= other.start or other.end <= self.start)

        def to_dict(self) -> Dict:
            """것을 사전 꼴로 바꾸기."""
            return {
                'text': self.text,
                'type': self.entity_type,
                'start': self.start,
                'end': self.end,
                'confidence': self.confidence
            }


    class Token:
        """
        토막낸 글의 토막 하나를 나타낸다.

        이름 알아보기에서는 토막의 글뿐 아니라 다음도 좇아야 한다:
        - 본디 글에서의 자리
        - 그 특징(대문자 쓰기, 문장 부호 등)
        - 그 것 이름표(익힘 자료용)
        """

        def __init__(self, text: str, start: int, end: int):
            """
            Token 첫자리매김.

            인수:
                text: 토큰 글월
                start: 본디 글에서의 시작 글자 자리
                end: 본디 글에서의 끝 글자 자리
            """
            self.text = text
            self.start = start
            self.end = end
            self.label = "O"  # 붙박이 이름표는 "O"(것 바깥)

        def __repr__(self):
            """토막의 글자열 나타냄."""
            return f"Token('{self.text}', label='{self.label}')"

        def get_features(self) -> Dict[str, bool]:
            """
            토막에서 말의 특징 뽑기.

            이 특징은 예로부터의 기계 배움 바탕 이름 알아보기에 쓸모 있다:
            - is_capitalized: 첫 글자가 대문자이다
            - is_all_caps: 글자가 모두 대문자이다
            - is_title: 첫 글자는 대문자, 나머지는 소문자
            - contains_digit: 토막에 숫자가 들어 있다
            - contains_punctuation: 토막에 문장 부호가 들어 있다
            - is_alpha: 토막이 글자로만 이루어졌다

            반환값:
                불 특징의 사전
            """
            features = {
                'is_capitalized': self.text[0].isupper() if self.text else False,
                'is_all_caps': self.text.isupper(),
                'is_title': self.text.istitle(),
                'contains_digit': any(c.isdigit() for c in self.text),
                'contains_punctuation': any(not c.isalnum() for c in self.text),
                'is_alpha': self.text.isalpha(),
                'length': len(self.text),
                'is_short': len(self.text) <= 3,
                'is_long': len(self.text) >= 10
            }
            return features


    class SimpleTokenizer:
        """
        이름 알아보기를 위한 단순 빈칸·문장 부호 바탕 토막내개.

        이 토막내개는:
        1. 빈칸에서 쪼갠다
        2. 문장 부호를 낱말에서 갈라낸다
        3. 글자 어긋남을 지킨다(것의 구간에 중요하다)

        유의: 실전에서는 spaCy나 NLTK 토막내개를 헤아려 보라.
        """

        def __init__(self):
            """토막내개 첫자리매김."""
            # 낱말 글자, 숫자, 낱낱의 문장 부호에 맞추는 무늬
            self.pattern = re.compile(r'\w+|[^\w\s]')

        def tokenize(self, text: str) -> List[Token]:
            """
            글을 Token 개체로 토막내기.

            인수:
                text: 들임 글월 문자열

            반환값:
                글과 글자 자리를 담은 Token 개체의 목록

            보기:
                >>> tokenizer = SimpleTokenizer()
                >>> tokens = tokenizer.tokenize("Apple Inc. is great!")
                >>> for token in tokens:
                ...     print(f"{token.text} ({token.start}:{token.end})")
                Apple (0:5)
                Inc (6:9)
                . (9:10)
                is (11:13)
                great (14:19)
                ! (19:20)
            """
            tokens = []

            # 맞는 곳과 그 자리 모두 찾기
            for match in self.pattern.finditer(text):
                token_text = match.group()
                start = match.start()
                end = match.end()

                # Token 개체 만들기
                token = Token(token_text, start, end)
                tokens.append(token)

            return tokens

        def tokenize_with_labels(self, text: str, entities: List[Entity]) -> List[Token]:
            """
            글을 토막내고 토막에 것 이름표 매기기.

            익힘 자료를 갖추는 데 결정적이다. 토막마다 그것이 것의 한 몫인지,
            그리고 어떤 갈래인지 가리키는 이름표를 받는다.

            인수:
                text: 들임 글월 문자열
                entities: 글 속 Entity 개체의 목록

            반환값:
                이름표를 매긴 Token 개체의 목록

            보기:
                >>> tokenizer = SimpleTokenizer()
                >>> text = "Steve Jobs founded Apple"
                >>> entities = [
                ...     Entity("Steve Jobs", "PER", 0, 11, 1.0),
                ...     Entity("Apple", "ORG", 20, 25, 1.0)
                ... ]
                >>> tokens = tokenizer.tokenize_with_labels(text, entities)
            """
            # 먼저 여느 때처럼 토막내기
            tokens = self.tokenize(text)

            # 토막마다 것을 보고 이름표 정하기
            for token in tokens:
                token.label = self._get_token_label(token, entities)

            return tokens

        def _get_token_label(self, token: Token, entities: List[Entity]) -> str:
            """
            토막의 것 이름표 정하기.

            인수:
                token: 이름표를 붙일 토막
                entities: 글 속 것의 목록

            반환값:
                이름표 글자열(보기로 "B-PER", "I-ORG", "O")
            """
            # 토막이 어떤 것과 겹치는지 살피기
            for entity in entities:
                # 토막이 것의 구간 안에 있다
                if token.start >= entity.start and token.end <= entity.end:
                    # 것의 첫 토막인지 살피기
                    is_first = token.start == entity.start

                    # 알맞은 이름표 돌려주기(시작은 B-, 안쪽은 I-)
                    prefix = "B" if is_first else "I"
                    return f"{prefix}-{entity.entity_type}"

            # 토막이 어느 것에도 들지 않는다
            return "O"


    class NERDataset:
        """
        글과 그 것 표시를 담는 이름 알아보기 자료 뭉치 그릇.

        이 클래스는 익히기와 값매김을 위해 이름 알아보기 자료를 갈무리하고 다루도록 돕는다.
        """

        def __init__(self):
            """빈 자료 뭉치 첫자리매김."""
            self.samples = []  # (글, 것) 튜플의 목록

        def add_sample(self, text: str, entities: List[Entity]):
            """
            자료 뭉치에 표본 하나 더하기.

            인수:
                text: 글월 문자열
                entities: 글 속 Entity 개체의 목록
            """
            self.samples.append((text, entities))

        def __len__(self):
            """자료 뭉치의 표본 개수 돌려주기."""
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[str, List[Entity]]:
            """번호로 표본 얻기."""
            return self.samples[idx]

        def get_statistics(self) -> Dict:
            """
            자료 뭉치 통계량 셈하기.

            반환값:
                자료 뭉치 통계량을 담은 사전:
                - num_samples: 표본의 전체 개수
                - num_entities: 것의 전체 개수
                - entity_type_counts: 것 갈래마다의 개수
                - avg_entities_per_sample: 글마다의 평균 것 개수
            """
            total_entities = 0
            entity_type_counts = defaultdict(int)

            for text, entities in self.samples:
                total_entities += len(entities)
                for entity in entities:
                    entity_type_counts[entity.entity_type] += 1

            return {
                'num_samples': len(self.samples),
                'num_entities': total_entities,
                'entity_type_counts': dict(entity_type_counts),
                'avg_entities_per_sample': total_entities / len(self.samples) if self.samples else 0
            }

        def display_sample(self, idx: int):
            """
            표본과 그 것을 읽기 좋은 꼴로 보여 주기.

            인수:
                idx: 표본 번호
            """
            text, entities = self.samples[idx]

            print(f"\n{'='*60}")
            print(f"Sample {idx + 1}:")
            print(f"{'='*60}")
            print(f"Text: {text}")
            print(f"\nEntities found: {len(entities)}")
            print("-" * 60)

            for i, entity in enumerate(entities, 1):
                print(f"{i}. '{entity.text}' -> {entity.entity_type} (pos: {entity.start}-{entity.end})")

            print("="*60)


    def demonstrate_ner_basics():
        """
        기본 이름 알아보기 개념 보이기.

        이 함수가 보여 주는 것:
        1. 것 만들기
        2. 글 토막내기
        3. 토막에 이름표 붙이기
        4. 단순한 자료 뭉치 세우기
        """
        print("="*70)
        print("Named Entity Recognition - Basic Concepts Demonstration")
        print("="*70)

        # 보기 1: 것 만들기
        print("\n1. Creating Named Entities")
        print("-" * 70)

        text1 = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

        # 글 속의 것 정하기
        entities1 = [
            Entity("Apple Inc.", EntityType.ORGANIZATION, 0, 10, 1.0),
            Entity("Steve Jobs", EntityType.PERSON, 26, 37, 1.0),
            Entity("Cupertino", EntityType.LOCATION, 41, 50, 1.0),
            Entity("California", EntityType.LOCATION, 52, 62, 1.0)
        ]

        print(f"Text: {text1}")
        print(f"\nEntities:")
        for entity in entities1:
            print(f"  - {entity}")

        # 보기 2: 토막내기
        print("\n\n2. Tokenization for NER")
        print("-" * 70)

        tokenizer = SimpleTokenizer()
        tokens = tokenizer.tokenize(text1)

        print(f"Tokens extracted: {len(tokens)}")
        for i, token in enumerate(tokens[:10], 1):  # 처음 토막 10개 보이기
            features = token.get_features()
            print(f"  {i}. '{token.text}' at ({token.start}:{token.end})")
            print(f"     Features: capitalized={features['is_capitalized']}, "
                  f"all_caps={features['is_all_caps']}")

        # 보기 3: 토막에 이름표 붙이기
        print("\n\n3. Token Labeling (IOB Scheme)")
        print("-" * 70)

        labeled_tokens = tokenizer.tokenize_with_labels(text1, entities1)

        print(f"{'Token':<15} {'Label':<10} {'Position'}")
        print("-" * 70)
        for token in labeled_tokens[:20]:  # 처음 토막 20개 보이기
            print(f"{token.text:<15} {token.label:<10} ({token.start}:{token.end})")

        # 보기 4: 것끼리 겹침
        print("\n\n4. Checking Entity Overlaps")
        print("-" * 70)

        entity_a = Entity("New York", EntityType.LOCATION, 0, 8, 1.0)
        entity_b = Entity("York University", EntityType.ORGANIZATION, 4, 19, 1.0)

        print(f"Entity A: {entity_a}")
        print(f"Entity B: {entity_b}")
        print(f"Do they overlap? {entity_a.overlaps(entity_b)}")

        # 보기 5: 자료 뭉치 세우기
        print("\n\n5. Building a NER Dataset")
        print("-" * 70)

        dataset = NERDataset()

        # 표본 몇 개 더하기
        samples = [
            ("Apple Inc. was founded by Steve Jobs in Cupertino.",
             [Entity("Apple Inc.", "ORG", 0, 10, 1.0),
              Entity("Steve Jobs", "PER", 27, 37, 1.0),
              Entity("Cupertino", "LOC", 41, 50, 1.0)]),

            ("Google announced a new product in Mountain View.",
             [Entity("Google", "ORG", 0, 6, 1.0),
              Entity("Mountain View", "LOC", 34, 47, 1.0)]),

            ("Barack Obama visited Paris in 2015.",
             [Entity("Barack Obama", "PER", 0, 12, 1.0),
              Entity("Paris", "LOC", 21, 26, 1.0)]),
        ]

        for text, entities in samples:
            dataset.add_sample(text, entities)

        # 통계량 보여 주기
        stats = dataset.get_statistics()
        print(f"Dataset size: {stats['num_samples']} samples")
        print(f"Total entities: {stats['num_entities']}")
        print(f"Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
        print(f"\nEntity type distribution:")
        for entity_type, count in stats['entity_type_counts'].items():
            print(f"  - {entity_type}: {count}")

        # 표본 하나 보여 주기
        dataset.display_sample(0)

        # 보기 6: 흔한 어려움
        print("\n\n6. Common Challenges in NER")
        print("-" * 70)

        challenges = [
            ("Ambiguity", 
             "Washington (person or location?)",
             "Context is crucial for disambiguation"),

            ("Nested entities",
             "Bank of America in New York",
             "'Bank of America' (ORG) contains 'America' (LOC)"),

            ("Entity boundaries",
             "New York City vs. New York",
             "Determining exact entity span"),

            ("Rare entities",
             "Newly founded companies or products",
             "Not in training data"),

            ("Multi-word entities",
             "University of California, Berkeley",
             "Long entity spans are challenging")
        ]

        for challenge_type, example, explanation in challenges:
            print(f"\n{challenge_type}:")
            print(f"  Example: {example}")
            print(f"  Issue: {explanation}")


    if __name__ == "__main__":
        # 보임 돌리기
        demonstrate_ner_basics()

        print("\n" + "="*70)
        print("Demonstration complete!")
        print("="*70)

        # 주고받는 보기 하나 더
        print("\n\nTry it yourself!")
        print("-" * 70)

        # 이 보기는 고쳐 써도 된다
        custom_text = "Microsoft CEO Satya Nadella announced new AI products in Seattle."
        custom_entities = [
            Entity("Microsoft", EntityType.ORGANIZATION, 0, 9, 1.0),
            Entity("Satya Nadella", EntityType.PERSON, 14, 27, 1.0),
            Entity("Seattle", EntityType.LOCATION, 58, 65, 1.0)
        ]

        print(f"\nCustom text: {custom_text}")
        print(f"\nEntities:")
        for entity in custom_entities:
            print(f"  - {entity}")

        # 토막내고 이름표 붙이기
        tokenizer = SimpleTokenizer()
        labeled_tokens = tokenizer.tokenize_with_labels(custom_text, custom_entities)

        print(f"\nLabeled tokens:")
        print(f"{'Token':<20} {'Label':<10}")
        print("-" * 30)
        for token in labeled_tokens:
            print(f"{token.text:<20} {token.label:<10}")
    ```

??? note "기존 전체 출력"

    ```
    ======================================================================
    Named Entity Recognition - Basic Concepts Demonstration
    ======================================================================

    1. Creating Named Entities
    ----------------------------------------------------------------------
    Text: Apple Inc. was founded by Steve Jobs in Cupertino, California.

    Entities:
      - Entity(text='Apple Inc.', type='ORG', span=(0, 10))
      - Entity(text='Steve Jobs', type='PER', span=(26, 37))
      - Entity(text='Cupertino', type='LOC', span=(41, 50))
      - Entity(text='California', type='LOC', span=(52, 62))


    2. Tokenization for NER
    ----------------------------------------------------------------------
    Tokens extracted: 13
      1. 'Apple' at (0:5)
         Features: capitalized=True, all_caps=False
      2. 'Inc' at (6:9)
         Features: capitalized=True, all_caps=False
      3. '.' at (9:10)
         Features: capitalized=False, all_caps=False
      4. 'was' at (11:14)
         Features: capitalized=False, all_caps=False
      5. 'founded' at (15:22)
         Features: capitalized=False, all_caps=False
      6. 'by' at (23:25)
         Features: capitalized=False, all_caps=False
      7. 'Steve' at (26:31)
         Features: capitalized=True, all_caps=False
      8. 'Jobs' at (32:36)
         Features: capitalized=True, all_caps=False
      9. 'in' at (37:39)
         Features: capitalized=False, all_caps=False
      10. 'Cupertino' at (40:49)
         Features: capitalized=True, all_caps=False


    3. Token Labeling (IOB Scheme)
    ----------------------------------------------------------------------
    Token           Label      Position
    ----------------------------------------------------------------------
    Apple           B-ORG      (0:5)
    Inc             I-ORG      (6:9)
    .               I-ORG      (9:10)
    was             O          (11:14)
    founded         O          (15:22)
    by              O          (23:25)
    Steve           B-PER      (26:31)
    Jobs            I-PER      (32:36)
    in              O          (37:39)
    Cupertino       O          (40:49)
    ,               I-LOC      (49:50)
    California      O          (51:61)
    .               I-LOC      (61:62)


    4. Checking Entity Overlaps
    ----------------------------------------------------------------------
    Entity A: Entity(text='New York', type='LOC', span=(0, 8))
    Entity B: Entity(text='York University', type='ORG', span=(4, 19))
    Do they overlap? True


    5. Building a NER Dataset
    ----------------------------------------------------------------------
    Dataset size: 3 samples
    Total entities: 7
    Average entities per sample: 2.33

    Entity type distribution:
      - ORG: 2
      - PER: 2
      - LOC: 3

    ============================================================
    Sample 1:
    ============================================================
    Text: Apple Inc. was founded by Steve Jobs in Cupertino.

    Entities found: 3
    ------------------------------------------------------------
    1. 'Apple Inc.' -> ORG (pos: 0-10)
    2. 'Steve Jobs' -> PER (pos: 27-37)
    3. 'Cupertino' -> LOC (pos: 41-50)
    ============================================================


    6. Common Challenges in NER
    ----------------------------------------------------------------------

    Ambiguity:
      Example: Washington (person or location?)
      Issue: Context is crucial for disambiguation

    Nested entities:
      Example: Bank of America in New York
      Issue: 'Bank of America' (ORG) contains 'America' (LOC)

    Entity boundaries:
      Example: New York City vs. New York
      Issue: Determining exact entity span

    Rare entities:
      Example: Newly founded companies or products
      Issue: Not in training data

    Multi-word entities:
      Example: University of California, Berkeley
      Issue: Long entity spans are challenging

    ======================================================================
    Demonstration complete!
    ======================================================================


    Try it yourself!
    ----------------------------------------------------------------------

    Custom text: Microsoft CEO Satya Nadella announced new AI products in Seattle.

    Entities:
      - Entity(text='Microsoft', type='ORG', span=(0, 9))
      - Entity(text='Satya Nadella', type='PER', span=(14, 27))
      - Entity(text='Seattle', type='LOC', span=(58, 65))

    Labeled tokens:
    Token                Label     
    ------------------------------
    Microsoft            B-ORG     
    CEO                  O         
    Satya                B-PER     
    Nadella              I-PER     
    announced            O         
    new                  O         
    AI                   O         
    products             O         
    in                   O         
    Seattle              O         
    .                    I-LOC     
    ```

## 2. 유형 목록과 언급의 표현

EntityType은 사람·조직·장소 등 문자열 상수를 모은 클래스다. CoNLL 네 유형과 날짜·금액·상품·행사 등을 함께 둔 예제의 목록이며 특정 벤치마크 표준 전체와 동일한 것은 아니다. Entity는 텍스트·유형·start·end·confidence를 보관하지만 원문과 범위·유형을 검사하지 않는다. confidence=1.0도 직접 대입한 값이지 모델의 보정된 확신이 아니다.

start·end는 원문 문자열의 반열린 문자 구간 [start,end)다. 같은 text라도 원문의 서로 다른 위치에 있으면 다른 언급이다. 현재 __eq__는 범위와 유형만 비교하고 문서 ID·text·점수는 비교하지 않는다. 따라서 서로 다른 문서의 객체를 같은 좌표로 직접 비교할 때는 문서 구분이 필요하다. 이 클래스의 overlaps는 두 방향의 분리 조건을 검사하며 맞닿기만 한 구간은 겹치지 않는다.

## 3. 정규식 토큰화와 수동 offset의 한계

SimpleTokenizer는 단어 문자 묶음 또는 공백·단어 문자가 아닌 개별 문장부호를 찾아 원문 offset을 보존한다. 공백을 토큰으로 내지 않으며 형태소·서브워드 분석은 아니다. 대문자·숫자·길이 등의 get_features는 수작업 특징일 뿐 학습한 embedding이 아니고 한국어의 개체를 영어 대문자 규칙으로 판정할 수는 없다.

주석에도 실제 offset 오류가 있다. 첫 문장의 Steve Jobs는 [26,36), Cupertino는 [40,49), California는 [51,61)인데 코드에는 각각 끝이나 시작이 어긋난 범위가 들어 있다. 마지막 사용자 예의 Seattle도 [57,64)가 실제 구간이며 코드의 [58,65)와 다르다. 기존 출력과 코드의 위치를 검증된 정답으로 인용하지 않는다.

같은 단어를 text.find로 한 번 찾는 것만으로 여러 번 등장하는 언급을 모두 정렬할 수도 없다. 원문[start:end]가 지정한 text와 같은지와 문서 범위를 확인해야 한다. 정규화·문자 삭제·다른 인덱스 체계를 사용하면 offset 변환 규칙도 필요하다.

## 4. 토큰에 레이블을 붙이는 규칙

_get_token_label은 토큰 전체가 개체 구간 안에 들어가는 첫 번째 주석을 선택한다. 토큰 시작과 개체 시작이 정확히 같을 때만 B를 붙이고 그렇지 않으면 I를 붙인다. 따라서 개체 경계가 토큰 중간이나 앞 공백에 있으면 첫 선택 토큰도 I가 되거나 일부 토큰이 O로 남을 수 있다. 잘못된 수동 offset이 곧 잘못된 태그 열로 전파된다.

중첩·겹침 주석에서는 목록에서 먼저 만난 개체가 우선한다. 개체 여러 층을 동시에 표현하거나 가장 긴 언급을 자동 선택하는 규칙이 아니다. 주석 지침과 토큰 경계 정렬을 먼저 맞춰야 한다. IOB2의 유효성을 검사하는 단계도 이 파일에는 없다.

## 5. 자료 통계와 코드 범위

NERDataset은 원문·주석 목록을 보관하고 표본 수·유형별 개수·표본당 평균을 계산한다. 레이블 품질이나 새로운 자료에서의 모델 성능은 계산하지 않는다. 빈 데이터의 평균은0으로 처리하지만 그 값이 평가 정확도를 뜻하지는 않는다.

이 파일에는 PyTorch 학습 모델·손실·역전파·생성기가 없다. 기본 시연은 main 조건 아래에서 호출된다. 개체명 인식의 데이터 표현을 이해하는 단계이며, 다음의 규칙·사전·학습 기반 모델과 연결할 때도 동일한 offset·유형·태그 계약이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
직접 지정한 confidence=1.0과 Entities found 출력을 자동 NER 성능으로 해석할 수 있는가?

</div>

??? success "연습문제 1 풀이"

    아니다. 주석 객체를 직접 만들고 표시하며 학습·추론을 하지 않는다. 유형별 통계도 저장된 주석의 개수이고 정밀도·재현율 평가가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
문자 offset과 토큰 레이블을 만들기 전에 확인할 항목을 설명하라.

</div>

??? success "연습문제 2 풀이"

    0≤start<end≤원문길이와 원문[start:end]=entity.text, 유형의 유효성을 확인한다. 토큰 경계와 개체 시작이 맞는지, 겹침 주석의 우선순위가 의도된 것인지도 검사한다. float32 영상 텐서 모양 검사는 이 객체 계약과 관계없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코드의 Cupertino [41,50)이 왜 부적절한지와 레이블에 미치는 영향을 설명하라.

</div>

??? success "연습문제 3 풀이"

    실제는 [40,49)다. 토큰의 시작40이 주석 시작41보다 작아 전체 포함 조건을 만족하지 못하고 O가 될 수 있다. 주석 오류를 신경망의 실패나 과적합으로 해석하지 말고 원문 offset부터 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 EntityType 시험 조각이 본문에 맞는지 검토하라.

</div>

??? success "연습문제 4 풀이"

    EntityType은 문자열 상수를 모은 클래스이며 forward·parameters·학습 경로가 없다. 미정의 입력과 호출·역전파를 쓰는 아래 조각은 완성된 시험이 아니다. 원문 일치·offset·접촉/겹침·반복 언급·중첩 우선순위·빈 표본과 토큰 레이블 계약을 확인하는 명세가 적절하다.

    ```python
    def test_entitytype():
        model = EntityType(...)
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

개체 주석은 유형과 원문 문자 위치가 정확히 맞아야 한다. 이 예제는 수동 주석의 토큰 정렬·표시·통계이며 자동 NER 모델이 아니다. 수동 offset 오류와 첫 주석 우선·전체 포함 규칙의 한계를 이해하고 태그 유효성을 확인한다.
