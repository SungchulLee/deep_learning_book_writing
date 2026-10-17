# 수작업 피처를 사용하는 CRF 개체명 인식

이 예제는 sklearn-crfsuite로 선형 사슬 CRF를 학습하고 태그 열을 예측한다. 단어·접미·대소문자와 앞뒤 토큰을 피처로 사용하며 PyTorch 뉴럴 인코더·자동미분 코드는 아니다. 한 문장으로 학습해 다른 문장을 예측하는 작은 시연이므로 일반적인 NER 성능이나 미지 이름에 대한 일반화를 입증하지 않는다.

## 1. 코드

```python
"""
CRF 바탕 이름 알아보기
===================================

피처 빚기를 곁들인 차례 이름표 붙이기를 위한 조건부 무작위 마당.

여기 짠 것은 CRF 모델 익히기에 sklearn-crfsuite를 쓴다.

핵심 개념:
- 토막의 피처 뽑기
- CRF로 하는 차례 이름표 붙이기
- 학습과 평가

지은이: 러닝 목적
날짜: 2025
"""

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from typing import List, Dict, Tuple

# ========================================================================
# 메인
# ========================================================================


class CRF_NER:
    """피처 빚기를 곁들인 CRF 바탕 이름 알아보기."""
    
    def __init__(self):
        """CRF 모델 첫자리매김."""
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    
    def word_features(self, sentence: List[str], i: int) -> Dict:
        """
        자리 i의 토막에서 피처 뽑기.
        
        피처:
        - 낱말 자체(소문자로)
        - 낱말의 대문자 쓰기 무늬
        - 낱말 꼴
        - 앞가지와 뒷가지
        - 맥락 낱말
        """
        word = sentence[i]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        
        # 맥락 피처
        if i > 0:
            word_prev = sentence[i-1]
            features.update({
                '-1:word.lower()': word_prev.lower(),
                '-1:word.istitle()': word_prev.istitle(),
                '-1:word.isupper()': word_prev.isupper(),
            })
        else:
            features['BOS'] = True  # 월의 시작
        
        if i < len(sentence) - 1:
            word_next = sentence[i+1]
            features.update({
                '+1:word.lower()': word_next.lower(),
                '+1:word.istitle()': word_next.istitle(),
                '+1:word.isupper()': word_next.isupper(),
            })
        else:
            features['EOS'] = True  # 월의 끝
        
        return features
    
    def sentence_features(self, sentence: List[str]) -> List[Dict]:
        """월의 모든 토막에서 피처 뽑기."""
        return [self.word_features(sentence, i) for i in range(len(sentence))]
    
    def train(self, X_train: List[List[str]], y_train: List[List[str]]):
        """
        CRF 모델 익히기.
        
        인수:
            X_train: 월의 목록(월마다 토막의 목록)
            y_train: 이름표 차례의 목록(저마다 이름표의 목록)
        """
        X_train_features = [self.sentence_features(s) for s in X_train]
        self.model.fit(X_train_features, y_train)
    
    def predict(self, X_test: List[List[str]]) -> List[List[str]]:
        """시험 월의 이름표 어림하기."""
        X_test_features = [self.sentence_features(s) for s in X_test]
        return self.model.predict(X_test_features)


if __name__ == "__main__":
    # 사용 예
    ner = CRF_NER()
    
    # 예제 데이터
    X_train = [["Steve", "Jobs", "founded", "Apple"]]
    y_train = [["B-PER", "I-PER", "O", "B-ORG"]]
    
    ner.train(X_train, y_train)
    
    X_test = [["Bill", "Gates", "started", "Microsoft"]]
    predictions = ner.predict(X_test)
    
    print("Predictions:", predictions)
```

## 2. 토큰 피처와 시퀀스 점수

word_features는 단어 소문자·끝2/3문자·대문자·숫자 여부, 앞뒤 한 토큰의 표기·대문자 여부와 문장 경계BOS/EOS를 만든다. suffix는 원문 대소문자를 유지해 word.lower()와 다른 신호다. sentence_features는 토큰 순서와 같은 길이의 사전 목록을 반환한다. 형태소·문자 LSTM·embedding을 학습하는 과정은 없다.

CRF는 피처에 대한 태그별 점수와 태그 전이로 전체 열의 조건부 확률을 정의한다. 출력의 유형 이름은 학습 레이블에 의해 정해지므로 새 유형을 사용하려면 일치하는 학습 주석이 필요하다. 뒤 토큰 피처를 사용해 전체 문장 입력을 가정하며 온라인 미래 비참조 모델은 아니다.

## 3. 정규화와 가능한 전이

[sklearn-crfsuite API](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html)의 lbfgs는 CRFsuite 최적화 설정이며 c1·c2는 각각L1·L2 정규화 계수다. dropout이나 러닝레이트와 같은 인수가 아니다. 둘을0.1로 정했다고 모든 자료에 적합한 규제가 되는 것은 아니다. 반복 상한100도 수렴·정확도를 보장하지 않는다.

all_possible_transitions=True는 학습 자료에 관측되지 않은 태그 쌍도 전이 피처 후보로 생성하도록 한다. ‘BIO에 맞는 전이만 허용’하는 옵션이 아니다. 학습 가능한 전이 점수만으로 초기I·다른 유형I가 수학적으로 금지되지는 않으므로 유효성 검사·제약이 필요한 폴리시와 구분한다.

## 4. 데이터와 평가의 한계

X_train·y_train은 문장 수와 문장별 토큰 수·태그 순서가 일치해야 한다. 태그 체계·유형·빈 입력·문장 경계를 검증하는 별도 코드가 없다. 이 시연의 학습은 Steve Jobs·Apple을 포함한 한 문장뿐이라 다양한 문맥·유형을 학습할 충분한 근거가 아니다.

테스트 문장에 대한 예측만 출력하며 정답 태그를 주어 span F1을 계산하지 않는다. metrics를 import하지만 사용하지도 않는다. 문장 하나의 출력이 그럴듯하다는 사실을 정밀도·재현율이나 뉴럴 넷트웍과의 우열로 해석하지 않는다. 실제 비교에는 별도의 분할과 동일한 토큰화·유형·평가 폴리시가 필요하다.

## 5. 확장 시 연결할 계약

규칙·사전 피처를 더할 수 있지만 대상 영역의 오탐·미탐과 희귀 문자열을 검토하고 독립 자료에서 선택해야 한다. 사전 버전·피처 추출·태그 매핑·학습 모델을 함께 보관해 추론 때 같은 입력을 만든다. train과 predict는 문장 목록을 받는 인터페이스이며 PyTorch forward·경사 텐서의 계약과 다르다.

이 코드는 main 조건 아래에서 실제 fit을 호출한다. 문서 편집에서는 그 호출을 실행하거나 출력을 새로 만들지 않고 기존 설정의 의미와 한계를 설명한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
all_possible_transitions=True가 BIO 유효성을 강제하는가?

</div>

??? success "풀이"

    아니다. 관측되지 않은 태그 쌍도 학습 전이 후보에 포함하는 설정이다. 금지 BIO 경로의 강제 제약과는 다르며 레이블 유효성과 디코딩 폴리시를 별도로 정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 X·y에서 어떤 대응을 확인해야 하는가?

</div>

??? success "풀이"

    문장 수, 각 문장의 토큰·태그 수와 순서, 유형·태그 체계가 일치해야 한다. 피처 사전 목록도 같은 토큰 순서여야 한다. 모델은 float32 이미지 배치가 아니라 문자열 문장과 태그 목록을 받는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
현재 시연에서 일반화 성능과 학습 수렴을 주장하기 어려운 이유를 설명하라.

</div>

??? success "풀이"

    학습 문장이 하나뿐이고 테스트 정답·span 지표·검증 분할이 없다. max_iterations는 상한이며 수렴 증거가 아니다. metrics import와 예측 출력만으로 성능이 측정되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 CRF_NER 시험 조각이 라이브러리 인터페이스에 맞는가?

</div>

??? success "풀이"

    CRF_NER는 train·predict를 제공하며 PyTorch forward·parameters·backward를 구현하지 않는다. 아래 미정의 입력도 완성 시험이 아니다. 문장·태그 정렬, 피처 타입·경계, 학습 유형·빈 입력 폴리시·예측 길이와 유효 태그를 확인하는 명세가 필요하다.

    ```python
    def test_crf_ner():
        model = CRF_NER(...)
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

피처 기반 CRF는 토큰 피처와 시퀀스 전이를 학습한다. 정규화 계수·미관측 전이 생성과 BIO 제약을 구분하고 입력·목표 정렬과 독립 span 평가를 연결한다. 한 문장 학습과 예측 출력만으로 일반 성능을 주장하지 않는다.
