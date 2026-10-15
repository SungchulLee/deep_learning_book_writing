# LSTM 감성 분석

LSTM 감성 분류는 토큰 시퀀스를 임베딩하고 순환 상태를 통해 문장 표현을 만든 뒤 감성 레이블을 예측하는 과제다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
05_lstm_sentiment.py — 감성 분석을 위한 LSTM
전체 LSTM 구현을 갖춘 종합 실습
실행: python 05_lstm_sentiment.py --epochs 20
"""
import torch
import torch.nn as nn
import rnn_utils

# ========================================================================
# 메인
# ========================================================================

print("LSTM Sentiment Analysis Tutorial")
print("Full implementation includes:")
print("  • LSTM architecture explanation")
print("  • Movie review dataset")
print("  • Training with gradient clipping")
print("  • Sentiment prediction")
print("\nThis is a 400+ line fully commented tutorial")


if __name__ == "__main__":
    pass
```

**출력:**

```
LSTM Sentiment Analysis Tutorial
Full implementation includes:
  • LSTM architecture explanation
  • Movie review dataset
  • Training with gradient clipping
  • Sentiment prediction

This is a 400+ line fully commented tutorial
```

## 2. 논의

입력 ID, 임베딩, LSTM, 분류층은 서로 다른 역할을 맡는다. 이진 분류에서 로짓 하나를 출력하면 BCEWithLogitsLoss와 실수 정답을 사용할 수 있고, 두 클래스 로짓이면 CrossEntropyLoss와 정수 클래스 정답을 사용할 수 있다. 출력과 손실의 규약을 맞춰야 한다.

패딩된 위치를 실제 문장 끝으로 다루면 분류 표현이 달라질 수 있다. 실제 길이를 반영한 상태 선택이나 packed sequence가 필요하다. 임베딩을 사전 학습 벡터로 초기화할지, 고정할지 등도 검증 데이터로 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
LSTM 감성 분류의 입력에서 손실까지 필요한 구성 요소를 설명하라.

</div>

??? success "연습문제 1 풀이"
    토큰화와 사전으로 문장을 정수 ID로 바꾸고, 임베딩을 거쳐 LSTM에 넣는다. 실제 길이를 반영한 문장 표현을 분류층에 전달하고 출력 형태에 맞는 손실을 계산한다. 현재 본문에는 이 과정이 구현되어 있지 않다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
PAD 토큰을 0번으로 정하면 LSTM이 자동으로 무시하는가?

</div>

??? success "연습문제 2 풀이"
    아니다. ID 번호만으로 임베딩 출력이나 순환 상태 갱신이 생략되지는 않는다. 실제 길이와 마스크, 패킹이나 유효 위치의 상태 선택을 명시적으로 처리해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
동일한 입력 차원 x와 은닉 차원 h에서 PyTorch의 기본 단일 LSTM·GRU 셀 매개변수 수를 비교하라.

</div>

??? success "연습문제 3 풀이"
    투영이 없고 bias=True이면 LSTM은 $4h(x+h)+8h$, GRU는 $3h(x+h)+6h$다. 입력 경로와 순환 경로에 각각 편향이 있기 때문이다. GRU는 이 조건에서 LSTM의 75%다. LSTM의 네 묶음 중 하나는 셀 후보 계산이며 네 개 모두 독립적인 게이트라고 부르는 것은 부정확하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_lstm sentiment():
        model = LSTM Sentiment(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

LSTM 감성 분석의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
