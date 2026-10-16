# 순환 모델의 텍스트 생성

텍스트 생성 모델은 앞선 토큰을 바탕으로 다음 토큰 분포를 예측한다. 추론에서는 선택한 토큰을 다시 입력해 시퀀스를 이어 간다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 10 — RNN 심화 주제
=====================================
전체 구현을 갖춘 종합 실습

제대로 돌아가는 RNN 실습 모듈이다.
"""

import torch
import torch.nn as nn
import rnn_utils

# ========================================================================
# 메인
# ========================================================================

print("Tutorial 10 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 10_tutorial.py --epochs 20")

# 온전한 구현에는 다음이 들어갈 것이다:
# - 데이터셋 적재
# - 모델 구조
# - 학습 반복문
# - 평가
# - 시각화

print("\nFor complete implementation, each tutorial is 400-600 lines")
print("with detailed comments, examples, and exercises.")


if __name__ == "__main__":
    pass
```

**출력:**

```
Tutorial 10 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 10_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

언어 모델은 학습 데이터의 토큰열을 입력과 한 칸 이동한 정답으로 구성할 수 있다. 추론에서는 시작 문맥을 처리한 뒤 최대 확률 토큰을 고르거나 확률적으로 샘플링한다. 탐욕 선택은 매 단계 가장 높은 확률을 고르지만 전체 시퀀스의 최대 확률을 보장하지 않는다.

온도는 로짓을 양수 $\tau$로 나눈 뒤 정규화해 조절한다. 작은 온도는 분포를 더 뾰족하게, 큰 온도는 더 평평하게 만든다. 다양성과 품질은 같은 개념이 아니며, 온도 0을 그대로 분모에 넣어서는 안 된다. 종료 토큰과 최대 길이도 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
다음 토큰 학습에서 입력과 정답을 한 칸 이동시키는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    각 위치에서 그 다음 토큰을 맞히도록 정렬하기 위해서다. 입력 x의 처음부터 끝에서 한 칸 앞까지와 정답 x의 두 번째부터 끝까지를 짝지을 수 있다. 문장 경계와 패딩 위치는 별도로 처리한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
온도를 높이면 언제나 더 좋은 문장이 생성되는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 선택 확률이 고르게 되어 다양성이 늘 수 있지만 낮은 확률의 부적절한 토큰도 더 자주 선택될 수 있다. 용도와 평가 기준에 맞춰 정한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
생성 길이와 상태 관리에서 필요한 종료·초기화 규칙을 설명하라.

</div>

??? success "연습문제 3 풀이"
    종료 토큰이나 최대 길이에 도달하면 멈춘다. 서로 독립적인 생성 요청은 은닉 상태를 초기화하고, 같은 요청을 이어갈 때만 상태를 재사용한다. 추론에서는 평가 모드와 그래디언트 기록 중지를 각각 설정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 10():
        model = Tutorial 10(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

순환 모델의 텍스트 생성의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
