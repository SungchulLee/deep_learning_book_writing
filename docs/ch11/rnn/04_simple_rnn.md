# 기본 RNN 시작하기

기본 RNN은 현재 입력과 이전 은닉 상태로 새 은닉 상태를 계산한다. 대표적인 식은 $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 04 — RNN 심화 주제
=====================================
전체 구현을 갖춘 종합 실습

제대로 돌아가는 RNN 실습 모듈이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

print("Tutorial 04 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 04_tutorial.py --epochs 20")

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
Tutorial 04 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 04_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

같은 $W_x,W_h,b$를 여러 시점에 공유하므로, 시퀀스 길이가 늘어도 레이어 하나의 파라메터 수는 늘지 않는다. 계산량과 역전파에 필요한 중간 상태는 길이에 따라 증가한다.

입력이 $(B,L,F)$일 때 batch_first 설정을 사용하면 각 시점의 은닉 출력을 $(B,L,H)$로 받을 수 있다. 시퀀스 분류에는 실제 마지막 토큰의 상태나 풀링 결과를, 시점별 예측에는 각 위치의 출력을 사용할 수 있다. 배치 안의 서로 다른 시퀀스 사이에 은닉 상태를 무조건 이어서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
RNN이 시퀀스 길이마다 다른 파라메터를 만들 필요가 없는 이유를 설명하라.

</div>

??? success "풀이"
    시점마다 같은 상태 갱신 함수를 반복 적용하기 때문이다. 입력과 은닉 피처 차원이 같으면 길이가 바뀌어도 웨이트 크기는 같다. 다만 실행 시간과 중간 상태의 저장량은 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
batch_first=True인 입력과 은닉 출력의 축을 설명하라.

</div>

??? success "풀이"
    입력은 $(B,L,F)$이고 시점별 출력은 $(B,L,H)$다. 최종 은닉 상태는 별도의 레이어·방향 축을 가지며 batch_first가 그 축 순서까지 바꾸지는 않는다. 양방향이나 여러 레이어를 사용하면 해당 축의 의미도 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
서로 관련 없는 두 문장 사이에 은닉 상태를 계속 넘기면 어떤 문제가 생길 수 있는가?

</div>

??? success "풀이"
    앞 문장의 정보가 뒤 문장의 예측에 섞인다. 독립적인 예제라면 상태를 적절히 초기화한다. 연속 스트림을 나눠 처리하는 경우에는 상태를 이어갈 수 있지만 역전파를 어디에서 끊을지도 따로 정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 04():
        model = Tutorial 04(...)
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

기본 RNN 시작하기의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
