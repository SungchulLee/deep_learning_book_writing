# 이름 알아보기

이름 알아보기 — 기초. 이 단원은 이름 알아보기(NER)의 근본 개념을 소개한다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

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
        print(f"{token.text:<20} {token.label:<10}")```

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
이름 알아보기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 이름 알아보기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `EntityType`, `Entity`, `Token`, `SimpleTokenizer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
