# NER 종합 데모의 실행 범위와 결과 해석

이 데모는 규칙·사전 기반 추출, 수작업 피처와 토큰 지표를 호출하고 뉴럴 네트워크 구조의 개념을 출력한다. 모든 NER 방식을 학습·평가하거나 하나의 운영 파이프라인으로 결합하는 코드는 아니다. 방법별 장단점 표도 통제 실험 결과가 아니라 고정된 설명 문구다.

## 1. 코드와 출력

??? note "코드 (836줄)"

    ```python
    """
    개체명 인식 종합 데모
    ================================

    이 스크립트는 이 단원에서 다룬 모든 개체명 인식 방식을 보여 준다:
    1. 규칙 기반 개체명 인식
    2. 사전 기반 개체명 인식
    3. CRF 기반 개체명 인식
    4. 딥러닝 방식(양방향 LSTM-CRF, 트랜스포머)

    방식마다의 전체 예제를 보려면 이 스크립트를 실행하라.

    지은이: 교육 목적
    날짜: 2025
    """

    import re
    from typing import List, Dict, Set
    from dataclasses import dataclass

    # ========================================================================
    # 도우미 클래스 — 규칙 기반(RuleBasedNER) · 사전 기반(DictionaryNER)
    # 개체명 인식, 피처 추출(FeatureExtractor), 평가 지표(NERMetrics)
    # ========================================================================

    @dataclass
    class Rule:
        """
        추출 규칙 하나를 나타낸다.

        속성:
            name: 규칙의 이름
            pattern: 매칭할 정규식 패턴
            entity_type: 이 규칙이 추출하는 개체의 유형
            priority: 규칙의 우선순위(높을수록 먼저 쓴다)
            context_required: 반드시 있어야 하는 맥락 단어(없어도 됨)
        """
        name: str
        pattern: str
        entity_type: str
        priority: int = 1
        context_required: List[str] = None

        def __post_init__(self):
            """초기화 뒤에 정규식 패턴 컴파일하기."""
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)


    class RuleBasedNER:
        """
        규칙 기반 개체명 인식 시스템.

        이 시스템은 다음을 비롯한 수작업 규칙을 쓴다:
        1. 정규식 패턴
        2. 대문자 쓰기 패턴
        3. 맥락 기반 규칙
        4. 사전 조회(규칙 매칭과 함께)

        장점:
        - 빠르고 효율적이다
        - 학습 데이터가 필요 없다
        - 해석하기 아주 쉽다
        - 분야별 규칙을 쓸 수 있다
        - 뚜렷한 패턴에는 정밀도가 높다

        단점:
        - 재현율이 낮다(변형을 놓친다)
        - 규칙을 손수 만들어야 한다
        - 규칙이 많으면 유지보수하기 어렵다
        - 본 적 없는 패턴으로 일반화하지 못한다
        """

        def __init__(self):
            """규칙 기반 개체명 인식 시스템 초기화."""
            self.rules: List[Rule] = []
            self._initialize_default_rules()

        def _initialize_default_rules(self):
            """
            흔한 개체 유형의 기본 규칙 모음 초기화.

            이 규칙이 다루는 것:
            - 사람 이름(대문자 쓰기 패턴)
            - 조직(회사 접미사)
            - 장소(장소를 가리키는 말)
            - 날짜와 시간
            - 돈과 백분율
            - 전자우편과 URL
            """

            # 사람 이름 패턴
            # 패턴: 대문자 단어 뒤에 대문자 단어
            # 보기: "John Smith", "Mary Johnson"
            self.add_rule(Rule(
                name="person_full_name",
                pattern=r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',
                entity_type="PER",
                priority=3
            ))

            # 조직 패턴
            # 패턴: 회사 접미사가 붙은 대문자 단어
            # 보기: "Apple Inc.", "Microsoft Corporation", "Google LLC"
            self.add_rule(Rule(
                name="organization_suffix",
                pattern=r'\b([A-Z][A-Za-z\s&]+)\s+(Inc\.|Corp\.|Corporation|LLC|Ltd\.|Limited|Company|Co\.)\b',
                entity_type="ORG",
                priority=5
            ))

            # 패턴: "The" 뒤에 대문자로 시작하는 단어
            # 보기: "The New York Times", "The Washington Post"
            self.add_rule(Rule(
                name="organization_the",
                pattern=r'\bThe\s+([A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|Times|Post|Bank|University)?)\b',
                entity_type="ORG",
                priority=4
            ))

            # 장소 패턴
            # 패턴: 장소를 가리키는 핵심 단어
            # 보기: "New York City", "San Francisco", "Mount Everest"
            self.add_rule(Rule(
                name="location_indicators",
                pattern=r'\b((?:New|San|Los|Las)\s+[A-Z][a-z]+(?:\s+(?:City|Beach|Angeles|Vegas))?|'
                       r'Mount\s+[A-Z][a-z]+|Lake\s+[A-Z][a-z]+|'
                       r'[A-Z][a-z]+\s+(?:River|Ocean|Sea|Mountain))\b',
                entity_type="LOC",
                priority=4
            ))

            # 패턴: 주와 나라
            # 간추린 목록 — 실전에서는 포괄적인 가제티어를 쓴다
            self.add_rule(Rule(
                name="location_places",
                pattern=r'\b(California|Texas|New\s+York|Florida|Illinois|'
                       r'London|Paris|Tokyo|Beijing|Sydney|Berlin|'
                       r'United\s+States|USA|UK|China|Japan|Germany)\b',
                entity_type="LOC",
                priority=3
            ))

            # 날짜 패턴
            # 패턴: 여러 날짜 형식
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

            # 시간 패턴
            # 보기: "3:30 PM", "14:00", "noon"
            self.add_rule(Rule(
                name="time",
                pattern=r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|noon|midnight)\b',
                entity_type="TIME",
                priority=5
            ))

            # 돈 패턴
            # 보기: "$100", "$1,000.00", "€50"
            self.add_rule(Rule(
                name="money",
                pattern=r'\b([$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|'
                       r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|euros|pounds|yen))\b',
                entity_type="MONEY",
                priority=5
            ))

            # 백분율 패턴
            # 보기: "25%", "3.14%"
            self.add_rule(Rule(
                name="percentage",
                pattern=r'\b(\d+(?:\.\d+)?%|'
                       r'\d+(?:\.\d+)?\s*percent)\b',
                entity_type="PERCENT",
                priority=5
            ))

            # 전자우편 패턴
            # 보기: "user@example.com"
            self.add_rule(Rule(
                name="email",
                pattern=r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                entity_type="CONTACT",
                priority=6
            ))

            # URL 패턴
            # 보기: "https://www.example.com", "www.example.com"
            self.add_rule(Rule(
                name="url",
                pattern=r'\b((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)\b',
                entity_type="URL",
                priority=6
            ))

        def add_rule(self, rule: Rule):
            """
            시스템에 새 규칙 더하기.

            더 좁은 규칙이 넓은 규칙보다 먼저 쓰이도록
            규칙을 우선순위로 정렬한다(높은 것 먼저).

            인수:
                rule: 더할 Rule 객체
            """
            self.rules.append(rule)
            # 규칙을 우선순위로 정렬(높은 것 먼저)
            self.rules.sort(key=lambda r: r.priority, reverse=True)

        def extract_entities(self, text: str) -> List[Dict]:
            """
            모든 규칙을 써서 텍스트에서 개체 추출하기.

            과정:
            1. 우선순위 차례로 규칙을 쓴다
            2. 규칙에 맞을 때마다 개체를 만든다
            3. 겹치는 개체 다루기(우선순위 높은 쪽을 남긴다)
            4. 정렬한 개체의 목록을 반환한다

            인수:
                text: 입력 텍스트 문자열

            반환값:
                다음 항목을 담은 개체 사전의 목록:
                - text: 개체의 텍스트
                - type: 개체의 유형
                - start: 시작 자리
                - end: 끝 자리
                - rule: 맞은 규칙의 이름
                - confidence: 신뢰도 점수

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
            used_spans = set()  # 겹침 처리를 위해 사용한 문자 구간 추적

            # 우선순위 차례로 규칙 쓰기
            for rule in self.rules:
                # 이 규칙에 맞는 곳 모두 찾기
                for match in rule.compiled_pattern.finditer(text):
                    # 매칭된 텍스트와 자리 얻기
                    matched_text = match.group(0)
                    start = match.start()
                    end = match.end()

                    # 이 구간이 이미 추출한 개체와 겹치는지 확인하기
                    span_range = range(start, end)
                    if any(pos in used_spans for pos in span_range):
                        # 겹치는 개체 건너뛰기(우선순위가 낮은 규칙)
                        continue

                    # 지정되어 있으면 맥락 요건 확인하기
                    if rule.context_required:
                        # 매칭된 곳 둘레의 맥락 창 추출
                        context_start = max(0, start - 50)
                        context_end = min(len(text), end + 50)
                        context = text[context_start:context_end].lower()

                        # 필요한 맥락 단어가 있는지 확인하기
                        if not any(word.lower() in context for word in rule.context_required):
                            continue

                    # 개체 만들기
                    entity = {
                        'text': matched_text,
                        'type': rule.entity_type,
                        'start': start,
                        'end': end,
                        'rule': rule.name,
                        'confidence': 1.0 / (10 - rule.priority)  # 우선순위가 높을수록 신뢰도가 높다
                    }

                    entities.append(entity)

                    # 구간을 썼다고 표시하기
                    used_spans.update(span_range)

            # 개체를 자리로 정렬
            entities.sort(key=lambda e: e['start'])

            return entities


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


    class FeatureExtractor:
        """전통적인 기계 학습 기반 개체명 인식을 위한 피처 추출."""

        def __init__(self):
            """피처 추출기 초기화."""
            self.gazetteers = {}

        def extract_token_features(self, tokens: List[str], index: int,
                                  window_size: int = 2) -> Dict:
            """
            토큰의 피처를 종합적으로 추출하기.

            인수:
                tokens: 문장 속 토큰의 목록
                index: 현재 토큰의 인덱스
                window_size: 맥락 윈도의 크기

            반환값:
                피처의 사전
            """
            token = tokens[index]
            features = {}

            # 기본 단어 피처
            features['word'] = token.lower()
            features['word_length'] = len(token)

            # 철자 피처
            features['is_capitalized'] = token[0].isupper() if token else False
            features['is_all_caps'] = token.isupper()
            features['is_all_lower'] = token.islower()
            features['is_title'] = token.istitle()
            features['is_alphanumeric'] = token.isalnum()
            features['is_alpha'] = token.isalpha()
            features['is_digit'] = token.isdigit()

            # 단어 형태 피처
            features['word_shape'] = self.get_word_shape(token)
            features['short_word_shape'] = self.get_word_shape(token, short=True)

            # 접두사와 접미사 피처
            for n in range(1, min(5, len(token) + 1)):
                features[f'prefix_{n}'] = token[:n].lower()
                features[f'suffix_{n}'] = token[-n:].lower()

            # 글자 수준 피처
            features['contains_hyphen'] = '-' in token
            features['contains_digit'] = any(c.isdigit() for c in token)
            features['contains_upper'] = any(c.isupper() for c in token)

            # 맥락 피처(앞 토큰들)
            for i in range(1, window_size + 1):
                if index - i >= 0:
                    prev_token = tokens[index - i]
                    features[f'prev_{i}_word'] = prev_token.lower()
                    features[f'prev_{i}_is_cap'] = prev_token[0].isupper() if prev_token else False

            # 맥락 피처(다음 토큰들)
            for i in range(1, window_size + 1):
                if index + i < len(tokens):
                    next_token = tokens[index + i]
                    features[f'next_{i}_word'] = next_token.lower()
                    features[f'next_{i}_is_cap'] = next_token[0].isupper() if next_token else False

            # 자리 피처
            features['is_first'] = (index == 0)
            features['is_last'] = (index == len(tokens) - 1)

            return features

        @staticmethod
        def get_word_shape(word: str, short: bool = False) -> str:
            """
            단어 형태 표현 얻기.

            글자를 형태 기호에 대응시킨다:
            - 대문자: 'X'
            - 소문자: 'x'
            - 숫자: 'd'
            - 그 밖: 'c'

            short=True이면 연속한 같은 글자를 하나로 줄인다.

            보기:
                "iPhone5" -> "xXxxxxd"(긴 형태) 또는 "xXxd"(짧은 형태)
            """
            shape = []
            for char in word:
                if char.isupper():
                    shape.append('X')
                elif char.islower():
                    shape.append('x')
                elif char.isdigit():
                    shape.append('d')
                else:
                    shape.append('c')

            shape_str = ''.join(shape)

            if short:
                # 연속한 같은 글자를 하나로 줄이기
                if not shape_str:
                    return shape_str
                compressed = [shape_str[0]]
                for char in shape_str[1:]:
                    if char != compressed[-1]:
                        compressed.append(char)
                return ''.join(compressed)

            return shape_str


    class NERMetrics:
        """개체명 인식의 평가 지표."""

        @staticmethod
        def compute_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
            """
            토큰 수준의 정밀도, 재현율, F1 계산하기.

            인수:
                y_true: 정답 라벨
                y_pred: 예측 라벨

            반환값:
                정밀도, 재현율, F1 점수를 담은 사전
            """
            tp = 0  # 참 양성
            fp = 0  # 거짓 양성
            fn = 0  # 거짓 음성

            for true_seq, pred_seq in zip(y_true, y_pred):
                for true_label, pred_label in zip(true_seq, pred_seq):
                    if true_label != "O":
                        if pred_label == true_label:
                            tp += 1
                        else:
                            fn += 1
                            if pred_label != "O":
                                fp += 1
                    elif pred_label != "O":
                        fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }


    # ========================================================================
    # 메인
    # ========================================================================

    def demo_rule_based():
        """규칙 기반 개체명 인식 시연."""
        print("="*70)
        print("1. RULE-BASED NER DEMONSTRATION")
        print("="*70)

        ner = RuleBasedNER()

        texts = [
            "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
            "Microsoft CEO Satya Nadella announced new products in Seattle.",
            "The meeting is on January 15, 2025 at 2:30 PM.",
        ]

        for i, text in enumerate(texts, 1):
            print(f"\nExample {i}: {text}")
            entities = ner.extract_entities(text)
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['type']})")

        print("\n" + "-"*70)
        print("Advantages: Fast, interpretable, no training needed")
        print("Disadvantages: Low recall, cannot handle variations")


    def demo_dictionary_based():
        """사전 기반 개체명 인식 시연."""
        print("\n\n" + "="*70)
        print("2. DICTIONARY-BASED NER DEMONSTRATION")
        print("="*70)

        ner = DictionaryNER()

        text = "Steve Jobs founded Apple in California. Bill Gates started Microsoft."

        print(f"\nText: {text}")
        entities = ner.extract_entities(text)

        print(f"\nFound {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity['text']} ({entity['type']})")

        print("\n" + "-"*70)
        print("Advantages: Very fast lookup, perfect precision for known entities")
        print("Disadvantages: Zero recall for entities not in dictionary")


    def demo_feature_extraction():
        """전통적인 머신러닝을 위한 피처 추출 시연."""
        print("\n\n" + "="*70)
        print("3. FEATURE EXTRACTION FOR TRADITIONAL ML")
        print("="*70)

        extractor = FeatureExtractor()
        tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", "."]

        print(f"\nSentence: {' '.join(tokens)}")
        print("\nFeatures for each token:")

        for i, token in enumerate(tokens):
            features = extractor.extract_token_features(tokens, i, window_size=1)
            print(f"\n{token}:")
            print(f"  Word shape: {features['word_shape']}")
            print(f"  Is capitalized: {features['is_capitalized']}")
            print(f"  Prefix 2: {features.get('prefix_2', 'N/A')}")
            print(f"  Total features: {len(features)}")

        print("\n" + "-"*70)
        print("These features are used by CRF and other traditional ML models")


    def demo_evaluation():
        """개체명 인식 평가 지표 시연."""
        print("\n\n" + "="*70)
        print("4. NER EVALUATION METRICS")
        print("="*70)

        # 예측 예시
        y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG"]]
        y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O"]]  # I-ORG를 놓침

        print("\nTrue labels: ", y_true[0])
        print("Predicted:   ", y_pred[0])

        metrics = NERMetrics.compute_metrics(y_true, y_pred)

        print(f"\nToken-level metrics:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")

        print("\n" + "-"*70)
        print("Proper evaluation is crucial for comparing NER systems")


    def demo_architectures():
        """딥러닝 구조 시연."""
        print("\n\n" + "="*70)
        print("5. DEEP LEARNING ARCHITECTURES")
        print("="*70)

        print("\nBiLSTM-CRF Architecture:")
        print("  Input → Embedding → BiLSTM → Linear → CRF → Output")
        print("  - Captures context from both directions")
        print("  - CRF layer enforces valid tag sequences")
        print("  - State-of-the-art for sequence labeling")

        print("\nTransformer Architecture (BERT/RoBERTa):")
        print("  Input → BERT → Linear → Softmax → Tags")
        print("  - Uses pre-trained language understanding")
        print("  - Contextual embeddings")
        print("  - Current state-of-the-art performance")

        print("\n" + "-"*70)
        print("Modern NER systems typically use transformer-based models")


    def print_summary():
        """요약과 다음 단계 출력하기."""
        print("\n\n" + "="*70)
        print("SUMMARY: NER APPROACH COMPARISON")
        print("="*70)

        approaches = [
            ("Rule-based", "High", "Low", "Very Fast", "None", "Domain-specific patterns"),
            ("Dictionary", "High", "Low", "Very Fast", "None", "Known entities"),
            ("CRF", "Medium", "Medium", "Fast", "Moderate", "General NER"),
            ("BiLSTM-CRF", "High", "High", "Medium", "Large", "General NER"),
            ("Transformer", "Very High", "Very High", "Slow", "Very Large", "State-of-the-art"),
        ]

        print(f"\n{'Approach':<15} {'Precision':<12} {'Recall':<10} {'Speed':<12} {'Data Needed':<15} {'Best For'}")
        print("-" * 100)

        for approach, precision, recall, speed, data, best_for in approaches:
            print(f"{approach:<15} {precision:<12} {recall:<10} {speed:<12} {data:<15} {best_for}")

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. For quick prototyping: Start with rule-based or dictionary NER")
        print("2. For better performance: Collect training data and use CRF or BiLSTM-CRF")
        print("3. For state-of-the-art: Fine-tune a transformer model (BERT/RoBERTa)")
        print("4. For production: Combine multiple approaches (ensemble)")

        print("\n" + "="*70)
        print("LEARNING PATH")
        print("="*70)
        print("\nWeek 1 (Beginner):")
        print("  - Understanding NER concepts and entity types")
        print("  - IOB tagging schemes")
        print("  - Rule-based and dictionary-based approaches")

        print("\nWeek 2 (Intermediate):")
        print("  - Feature extraction techniques")
        print("  - CRF for sequence labeling")
        print("  - Evaluation metrics and dataset creation")

        print("\nWeek 3-4 (Advanced):")
        print("  - BiLSTM-CRF architecture")
        print("  - Transformer-based NER (BERT, RoBERTa)")
        print("  - Fine-tuning and production deployment")

        print("\n" + "="*70)


    def main():
        """모든 데모 실행하기."""
        print("\n" + "="*70)
        print("COMPREHENSIVE NER DEMONSTRATION")
        print("Module 38: Named Entity Recognition")
        print("="*70)

        try:
            demo_rule_based()
        except Exception as e:
            print(f"\nNote: Rule-based demo failed. Error: {e}")

        try:
            demo_dictionary_based()
        except Exception as e:
            print(f"\nNote: Dictionary-based demo failed. Error: {e}")

        try:
            demo_feature_extraction()
        except Exception as e:
            print(f"\nNote: Feature extraction demo failed. Error: {e}")

        try:
            demo_evaluation()
        except Exception as e:
            print(f"\nNote: Evaluation demo failed. Error: {e}")

        demo_architectures()
        print_summary()

        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)


    if __name__ == "__main__":
        main()
    ```


??? note "전체 출력 (180줄)"

    ```

    ======================================================================
    COMPREHENSIVE NER DEMONSTRATION
    Module 38: Named Entity Recognition
    ======================================================================
    ======================================================================
    1. RULE-BASED NER DEMONSTRATION
    ======================================================================

    Example 1: Apple Inc. was founded by Steve Jobs in Cupertino, California.
    Found 5 entities:
      - Apple Inc (PER)
      - was founded (PER)
      - by Steve (PER)
      - Jobs in (PER)
      - California (LOC)

    Example 2: Microsoft CEO Satya Nadella announced new products in Seattle.
    Found 3 entities:
      - Microsoft CEO (PER)
      - Satya Nadella (PER)
      - new products (LOC)

    Example 3: The meeting is on January 15, 2025 at 2:30 PM.
    Found 4 entities:
      - The meeting (PER)
      - is on (PER)
      - January 15, 2025 (DATE)
      - 2:30 PM (TIME)

    ----------------------------------------------------------------------
    Advantages: Fast, interpretable, no training needed
    Disadvantages: Low recall, cannot handle variations


    ======================================================================
    2. DICTIONARY-BASED NER DEMONSTRATION
    ======================================================================

    Text: Steve Jobs founded Apple in California. Bill Gates started Microsoft.

    Found 3 entities:
      - Steve Jobs (PER)
      - Apple (ORG)
      - Bill Gates (PER)

    ----------------------------------------------------------------------
    Advantages: Very fast lookup, perfect precision for known entities
    Disadvantages: Zero recall for entities not in dictionary


    ======================================================================
    3. FEATURE EXTRACTION FOR TRADITIONAL ML
    ======================================================================

    Sentence: Steve Jobs founded Apple Inc .

    Features for each token:

    Steve:
      Word shape: Xxxxx
      Is capitalized: True
      Prefix 2: st
      Total features: 26

    Jobs:
      Word shape: Xxxx
      Is capitalized: True
      Prefix 2: jo
      Total features: 28

    founded:
      Word shape: xxxxxxx
      Is capitalized: False
      Prefix 2: fo
      Total features: 28

    Apple:
      Word shape: Xxxxx
      Is capitalized: True
      Prefix 2: ap
      Total features: 28

    Inc:
      Word shape: Xxx
      Is capitalized: True
      Prefix 2: in
      Total features: 26

    .:
      Word shape: c
      Is capitalized: False
      Prefix 2: N/A
      Total features: 20

    ----------------------------------------------------------------------
    These features are used by CRF and other traditional ML models


    ======================================================================
    4. NER EVALUATION METRICS
    ======================================================================

    True labels:  ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG']
    Predicted:    ['B-PER', 'I-PER', 'O', 'B-ORG', 'O']

    Token-level metrics:
      Precision: 1.000
      Recall:    0.750
      F1 Score:  0.857

    ----------------------------------------------------------------------
    Proper evaluation is crucial for comparing NER systems


    ======================================================================
    5. DEEP LEARNING ARCHITECTURES
    ======================================================================

    BiLSTM-CRF Architecture:
      Input → Embedding → BiLSTM → Linear → CRF → Output
      - Captures context from both directions
      - CRF layer enforces valid tag sequences
      - State-of-the-art for sequence labeling

    Transformer Architecture (BERT/RoBERTa):
      Input → BERT → Linear → Softmax → Tags
      - Uses pre-trained language understanding
      - Contextual embeddings
      - Current state-of-the-art performance

    ----------------------------------------------------------------------
    Modern NER systems typically use transformer-based models


    ======================================================================
    SUMMARY: NER APPROACH COMPARISON
    ======================================================================

    Approach        Precision    Recall     Speed        Data Needed     Best For
    ----------------------------------------------------------------------------------------------------
    Rule-based      High         Low        Very Fast    None            Domain-specific patterns
    Dictionary      High         Low        Very Fast    None            Known entities
    CRF             Medium       Medium     Fast         Moderate        General NER
    BiLSTM-CRF      High         High       Medium       Large           General NER
    Transformer     Very High    Very High  Slow         Very Large      State-of-the-art

    ======================================================================
    NEXT STEPS
    ======================================================================

    1. For quick prototyping: Start with rule-based or dictionary NER
    2. For better performance: Collect training data and use CRF or BiLSTM-CRF
    3. For state-of-the-art: Fine-tune a transformer model (BERT/RoBERTa)
    4. For production: Combine multiple approaches (ensemble)

    ======================================================================
    LEARNING PATH
    ======================================================================

    Week 1 (Beginner):
      - Understanding NER concepts and entity types
      - IOB tagging schemes
      - Rule-based and dictionary-based approaches

    Week 2 (Intermediate):
      - Feature extraction techniques
      - CRF for sequence labeling
      - Evaluation metrics and dataset creation

    Week 3-4 (Advanced):
      - BiLSTM-CRF architecture
      - Transformer-based NER (BERT, RoBERTa)
      - Fine-tuning and production deployment

    ======================================================================

    ======================================================================
    DEMONSTRATION COMPLETE
    ======================================================================
    ```

## 2. 도우미 클래스와 실패 처리

규칙·사전·피처·지표를 맡는 네 클래스 `RuleBasedNER` · `DictionaryNER` · `FeatureExtractor` · `NERMetrics` 는 코드 맨 앞에 함께 실려 있다. 각각 「규칙 기반 개체명 인식」·「사전 기반 개체명 인식」·「피처 추출」·「평가 지표」 쪽의 클래스와 같은 코드이고, 이 데모가 부르지 않는 메서드만 뺐다.

각 데모의 예외를 출력하고 다음으로 넘어가므로 마지막 완료 메시지가 앞의 모든 데모가 정상 실행되었다는 증거는 아니다. 어떤 단계가 실패했는지와 실제 결과를 구분해야 한다. 위 출력에서는 1~4번 데모가 모두 예외 없이 끝났으므로 아래 3·4절의 수는 실제 실행 결과다.

## 3. 규칙·사전 결과와 의미적 정확도

규칙 기반 추출은 정규식·우선순위의 결과다. 출력이 그 한계를 그대로 보인다. 예문 1에서 `was founded`·`by Steve`·`Jobs in` 이 사람(PER)으로 잡혔고 Apple Inc 도 조직이 아니라 사람으로 나왔다. 예문 2에서는 `new products` 가 장소(LOC)가 되었다. 대소문자를 무시한 「대문자 두 단어」 규칙이 아무 두 단어에나 맞았기 때문이다. 예문 3의 날짜·시각만 제대로 잡혔다. 같은 입력에 늘 같은 결과가 나온다고 의미적으로 맞는 것은 아니다.

사전 데모는 Steve Jobs · Apple · Bill Gates 셋을 찾았고, 사전에 없는 California 와 Microsoft 는 놓쳤다. ‘known entities에 완벽한 정밀도’ 문구도 부정확하다. Apple 같은 다의어의 유형을 문맥 없이 정할 수 없고, 공백 토큰화의 구두점·반복 언급의 첫 위치 검색 때문에 알려진 항목도 놓치거나 잘못 정렬할 수 있다. 사전 등재 여부와 실제 문맥의 정답을 분리해 평가해야 한다.

피처 추출은 Steve·Jobs 등의 철자·문맥 사전을 보여 줄 뿐 CRF나 뉴럴 네트워크를 학습하지 않는다. 피처 수가 많거나 이름이 익숙하다는 사실을 정확도 비교로 바꾸어 설명하지 않는다.

## 4. 데모의 토큰 지표와 정확 span

평가 예는 정답 B-PER I-PER O B-ORG I-ORG에 대해 마지막 I-ORG를 O로 바꾼 예측을 사용한다. 호출하는 compute_metrics는 비O **토큰 태그**를 세므로 TP 3 · FP 0 · FN 1, 정밀도 1 · 재현율 0.75 · F1 약 0.857이다. 이는 개체 수준의 정확 일치 F1이 아니다.

정확 span으로 보면 사람은 맞지만 조직은 짧게 예측되어 TP 1 · FP 1 · FN 1, 정밀도·재현율·F1이 각각 0.5다. 태그의 일부를 맞힌 것과 전체 언급의 경계·유형을 맞힌 것을 구분한다. 입력 목록 길이·문장 경계·패딩·무효 태그의 정책도 실제 평가에 명시해야 한다.

## 5. 구조 설명·비교표의 한계

demo_architectures는 BiLSTM-CRF·트랜스포머 구조를 **출력할 뿐 생성·학습·비교하지 않는다**. CRF가 늘 유효한 태그 시퀀스를 강제하는 것도 아니다. 시작·종료·금지 전이 제약을 명시하고, 학습의 정규화와 디코딩이 같은 점수를 써야 한다. 관련 코드의 전이 방향·패딩·head 학습 한계는 개별 절의 설명을 참고한다.

정밀도·재현율·속도·필요 데이터를 High·Low 등으로 적은 표는 데이터·분할·구조·사전 학습·장치 조건이 없는 질적 개요다. 실제 측정치나 모든 과제의 고정 순위로 인용하지 않는다. 앙상블 역시 운영에 항상 필요한 단계이거나 자동으로 더 정확한 해결책은 아니다.

실제 모델 선택에는 같은 주석 체계·관찰 데이터·독립 분할·span 지표와 전체 처리 비용을 사용한다. 이 데모는 방법을 연결해 읽는 안내이며 최신 성능이나 배포 준비 상태를 검증한 실험 보고가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 파일이 실제로 실행하는 학습 모델과 출력만 하는 구조를 구분하라.

</div>

??? success "풀이"

    규칙·사전·피처 추출·토큰 지표는 코드 앞에 실린 네 클래스로 실제로 계산하며, 출력의 1~4절이 그 결과다. BiLSTM-CRF·트랜스포머는 설명 문자열만 출력하며 학습하지 않는다. 완료 메시지도 예외 처리된 데모의 성공을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
도우미 클래스·입력·결과에서 확인할 계약을 설명하라.

</div>

??? success "풀이"

    원문 offset·유형·토큰 순서와 태그 길이를 맞춘다. 추출 개수와 정답 평가, 토큰·span 지표를 구분한다. 고정된 텐서 shape 검사로 확인할 수 있는 인터페이스가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
데모의 토큰 F1과 정확 span F1을 계산하라.

</div>

??? success "풀이"

    비O 토큰은 TP 3 · FP 0 · FN 1이라 정밀도 1 · 재현율 0.75 · F1 = 6/7 ≈ 0.857이다. 정확 개체는 사람 하나만 맞고 조직의 짧은 예측과 누락 정답이 있어 TP 1 · FP 1 · FN 1, F1 = 0.5다. 부분 토큰 정답은 정확 span의 부분 점수가 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 테스트 조각이 이 종합 호출 예제에 맞는가?

</div>

??? success "풀이"

    공백이 있는 식별자와 정의되지 않은 Demo Complete NER·입력 때문에 실행 가능한 테스트가 아니다. 이 파일 자체는 뉴럴 네트워크가 아니므로 backward·parameters 검사가 맞지 않는다. 단계별 예외·결과·원문 offset·태그 길이·정확한 지표와 예외 후 완료 메시지를 명세로 확인해야 한다.

    ```python
    def test_demo complete ner():
        model = Demo Complete NER(...)
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

종합 데모는 여러 NER 도구와 구조 설명을 연결한다. 실제 호출·설명 출력·고정 비교표를 구분하고 규칙·사전 일치와 의미적 정답, 토큰 F1과 정확 span F1을 혼동하지 않는다.
