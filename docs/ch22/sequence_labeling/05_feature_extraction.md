# 개체명 인식을 위한 수작업 특징 추출

전통적인 CRF·최대 엔트로피 분류기 등에 단어·철자·주변 문맥 특징을 제공할 수 있다. 이 예제는 토큰 문자열 목록에서 특징 사전을 만드는 순수 Python 코드다. 신경망 표현을 학습하거나 텍스트를 생성·분류하는 모델은 포함하지 않는다.

## 1. 코드와 기존 출력

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

**기존 출력:**

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

## 2. 토큰 자체의 특징

소문자 단어는 표기 변형을 일부 줄이고, 대문자·숫자·길이·하이픈 같은 특징은 형태 단서를 제공한다. 접두·접미는 코드의 range에 따라 최대4문자까지 추출하며 실제 언어학적 형태소 분석은 아니다. 짧은 단어에서는 전체 단어와 접두·접미가 겹칠 수 있다.

word_shape는 대문자X·소문자x·숫자d·그 밖c의 문자열이다. 예를 들어 Steve는 Xxxxx, 축약 형태는 Xx다. short=True는 같은 종류가 연속된 부분을 하나로 줄이므로 길이 정보 일부가 사라진다. Unicode의 isupper·islower는 영문 이외 문자에도 해당 규칙을 쓰지만 한글 음절처럼 대소문자가 없는 문자는 보통 c로 모여 이 형태 특징이 유용하지 않을 수 있다.

이런 특징은 결정적인 유형 판정이 아니다. 대문자는 문장 시작에도 나타나고 숫자·접미사가 같은 서로 다른 개체도 많다. 언어·분야와 토큰화에 맞는 검증이 필요하다. 특징 개수가 많다는 출력도 정보량이나 정확도와 같지 않다.

## 3. 문맥 창과 경계

기본 window_size2는 앞뒤 최대2토큰의 소문자 표기·대문자 여부를 넣는다. 문장 시작·끝에는 존재하는 이웃만 추가하므로 반환 사전의 키 수가 위치에 따라 달라진다. is_first·is_last는 문장 경계의 신호다. 뒤 단어를 참조하므로 미래 텍스트를 사용할 수 없는 스트리밍 계약에는 그대로 적용하지 않는다.

유효한 index와 비음수 window_size를 검사하지 않는다. 음수 index는 Python 목록의 뒤에서 토큰을 선택하면서 이웃·경계의 논리는 음수 인덱스를 기준으로 계산해 의도와 다른 결과를 낼 수 있다. 빈 목록과 범위 밖 index도 계약을 정해야 한다. 빈 토큰 자체는 대문자 검사 전에 예외 처리하지만 원래 토큰화 정책과 일치하는지 확인한다.

## 4. 특징 사전과 모델 연결

반환 값은 문자열·bool·정수의 혼합 사전이다. 모델 라이브러리의 categorical·numeric feature 계약에 맞춰 변환해야 하며 PyTorch float32 영상 텐서를 받는 forward가 아니다. 길이 같은 숫자 특징의 스케일과 희귀 문자열·사전 크기는 학습·정규화에 영향을 줄 수 있다.

gazetteers는 빈 사전으로 저장하지만 특징 추출에서 사용하지 않는다. 따라서 사전 등재 특징이 구현되어 있다고 설명하면 안 된다. 학습 데이터에서 특징 선택·사전 구성을 정하고, 검증·시험 정보를 섞지 않은 동일 전처리로 평가한다. 이 페이지에는 학습·성능 비교가 없으며 신경망 활성화·잔차 연결을 분석할 대상도 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Steve의 word_shape·short_word_shape와 최대 접두·접미 길이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    word_shape는 Xxxxx, 축약은 Xx다. 접두·접미 반복은 최대4문자라 prefix_4는 stev, suffix_4는 teve가 된다. 실제 형태소를 찾아 분할하는 연산은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
반환 특징을 모두 같은 길이 float32 텐서로 assert하는 검사가 적절한가?

</div>

??? success "연습문제 2 풀이"

    문자열·bool·정수 혼합 사전이고 문장 경계에서는 문맥 키가 적어질 수 있어 맞지 않는다. 입력 문자열 목록·유효 index·문맥 범위와 모델 라이브러리가 받는 특징 타입을 검사해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한글 개체명에 대문자와 word_shape 특징을 그대로 적용하면 어떤 한계가 있는가?

</div>

??? success "연습문제 3 풀이"

    한글에는 영문 같은 대소문자 구분이 없어 많은 글자가 c로 모일 수 있다. 철자 형태만으로 구별하지 못하므로 언어에 맞는 토큰·문자·문맥·사전 신호를 별도로 설계하고 평가한다. gazetteers는 현재 사용되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 FeatureExtractor 시험 조각의 한계와 알맞은 확인 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    FeatureExtractor는 nn.Module이 아니고 forward·parameters·역전파가 없다. 아래 미정의 입력의 신경망 검사는 완성 시험이 아니다. 첫·끝·한 토큰 문장, 빈 토큰·유효하지 않은 index, 문맥 창·Unicode·접두 길이와 반환 특징 타입을 명세로 확인한다.

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


## 정리하며

수작업 NER 특징은 단어·철자·문맥 단서를 모델의 입력 계약에 맞춰 제공한다. 최대4문자 접두·접미와 문자 shape는 형태소·의미 분석이 아니며 언어·경계·숫자 스케일과 사전 사용 여부를 확인해야 한다.
