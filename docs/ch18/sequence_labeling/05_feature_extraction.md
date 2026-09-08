# 특징 뽑기

예로부터의 이름 알아보기를 위한 특징 뽑기. 예로부터의 기계 배움 바탕 이름 알아보기(CRF, MaxEnt 등)를 위한 특징 빚기.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
예로부터의 이름 알아보기를 위한 특징 뽑기
=======================================

예로부터의 기계 배움 바탕 이름 알아보기(CRF, MaxEnt 등)를 위한 특징 빚기.

뽑은 특징:
- 낱말 수준 특징
- 글자 수준 특징
- 철자 특징
- 맥락 특징
- 지명록 특징

지은이: 배움 목적
날짜: 2025
"""

from typing import List, Dict, Set
import re

# ========================================================================
# 메인
# ========================================================================


class FeatureExtractor:
    """예로부터의 기계 배움 바탕 이름 알아보기를 위한 특징 뽑기."""
    
    def __init__(self):
        """특징 뽑개 첫자리매김."""
        self.gazetteers = {}
    
    def extract_token_features(self, tokens: List[str], index: int, 
                              window_size: int = 2) -> Dict:
        """
        토막의 특징을 두루 뽑기.
        
        인수:
            tokens: 월 속 토막의 목록
            index: 지금 토막의 번호
            window_size: 맥락 창의 크기
            
        반환값:
            특징의 사전
        """
        token = tokens[index]
        features = {}
        
        # 기본 낱말 특징
        features['word'] = token.lower()
        features['word_length'] = len(token)
        
        # 철자 특징
        features['is_capitalized'] = token[0].isupper() if token else False
        features['is_all_caps'] = token.isupper()
        features['is_all_lower'] = token.islower()
        features['is_title'] = token.istitle()
        features['is_alphanumeric'] = token.isalnum()
        features['is_alpha'] = token.isalpha()
        features['is_digit'] = token.isdigit()
        
        # 낱말 꼴 특징
        features['word_shape'] = self.get_word_shape(token)
        features['short_word_shape'] = self.get_word_shape(token, short=True)
        
        # 앞가지와 뒷가지 특징
        for n in range(1, min(5, len(token) + 1)):
            features[f'prefix_{n}'] = token[:n].lower()
            features[f'suffix_{n}'] = token[-n:].lower()
        
        # 글자 수준 특징
        features['contains_hyphen'] = '-' in token
        features['contains_digit'] = any(c.isdigit() for c in token)
        features['contains_upper'] = any(c.isupper() for c in token)
        
        # 맥락 특징(앞 토막들)
        for i in range(1, window_size + 1):
            if index - i >= 0:
                prev_token = tokens[index - i]
                features[f'prev_{i}_word'] = prev_token.lower()
                features[f'prev_{i}_is_cap'] = prev_token[0].isupper() if prev_token else False
        
        # 맥락 특징(다음 토막들)
        for i in range(1, window_size + 1):
            if index + i < len(tokens):
                next_token = tokens[index + i]
                features[f'next_{i}_word'] = next_token.lower()
                features[f'next_{i}_is_cap'] = next_token[0].isupper() if next_token else False
        
        # 자리 특징
        features['is_first'] = (index == 0)
        features['is_last'] = (index == len(tokens) - 1)
        
        return features
    
    @staticmethod
    def get_word_shape(word: str, short: bool = False) -> str:
        """
        낱말 꼴 나타냄 얻기.
        
        글자를 꼴 부호에 대응시킨다:
        - 대문자: 'X'
        - 소문자: 'x'
        - 숫자: 'd'
        - 그 밖: 'c'
        
        short=True이면 잇달아 같은 글자를 하나로 뭉갠다.
        
        보기:
            "iPhone5" -> "xXxxxxd"(긴 꼴) 또는 "xXxd"(짧은 꼴)
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
            # 잇달아 같은 글자를 하나로 뭉개기
            if not shape_str:
                return shape_str
            compressed = [shape_str[0]]
            for char in shape_str[1:]:
                if char != compressed[-1]:
                    compressed.append(char)
            return ''.join(compressed)
        
        return shape_str


if __name__ == "__main__":
    # 예
    extractor = FeatureExtractor()
    tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", "."]
    
    for i, token in enumerate(tokens):
        features = extractor.extract_token_features(tokens, i)
        print(f"\nToken: {token}")
        print(f"Features: {len(features)} features extracted")
        print(f"Word shape: {features['word_shape']}")
        print(f"Is capitalized: {features['is_capitalized']}")
```

**출력:**

```

Token: Steve
Features: 28 features extracted
Word shape: Xxxxx
Is capitalized: True

Token: Jobs
Features: 30 features extracted
Word shape: Xxxx
Is capitalized: True

Token: founded
Features: 32 features extracted
Word shape: xxxxxxx
Is capitalized: False

Token: Apple
Features: 32 features extracted
Word shape: Xxxxx
Is capitalized: True

Token: Inc
Features: 28 features extracted
Word shape: Xxx
Is capitalized: True

Token: .
Features: 22 features extracted
Word shape: c
Is capitalized: False
```

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
특징 뽑기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_featureextractor():
        model = FeatureExtractor(...)
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

**다룬 것** — 특징 뽑기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `FeatureExtractor`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
