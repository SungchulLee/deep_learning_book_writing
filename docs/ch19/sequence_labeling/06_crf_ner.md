# CRF 이름 알아보기

CRF 바탕 이름 알아보기. 특징 빚기를 곁들인 차례 이름표 붙이기를 위한 조건부 무작위 마당.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
CRF 바탕 이름 알아보기
===================================

특징 빚기를 곁들인 차례 이름표 붙이기를 위한 조건부 무작위 마당.

여기 짠 것은 CRF 모델 익히기에 sklearn-crfsuite를 쓴다.

핵심 개념:
- 토막의 특징 뽑기
- CRF로 하는 차례 이름표 붙이기
- 학습과 평가

지은이: 배움 목적
날짜: 2025
"""

import sklearn_crfsuite
from sklearn_crfsuite import metrics
from typing import List, Dict, Tuple

# ========================================================================
# 메인
# ========================================================================


class CRF_NER:
    """특징 빚기를 곁들인 CRF 바탕 이름 알아보기."""
    
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
        자리 i의 토막에서 특징 뽑기.
        
        특징:
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
        
        # 맥락 특징
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
        """월의 모든 토막에서 특징 뽑기."""
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
CRF 이름 알아보기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — CRF 이름 알아보기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `CRF_NER`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
