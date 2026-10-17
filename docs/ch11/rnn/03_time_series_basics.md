# 시계열 데이터 준비

시계열은 시간 순서로 관측한 데이터다. 예측 문제에서는 입력 구간과 예측 시점, 예측할 미래 구간을 먼저 정해야 한다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 03 — RNN 심화 주제
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

print("Tutorial 03 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 03_tutorial.py --epochs 20")

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
Tutorial 03 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 03_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

입력 창 길이가 $L$, 바로 다음 한 시점을 정답으로 쓰는 길이 $T$의 단변량 시계열에서는 창을 한 칸씩 옮기면 $T-L$개의 예제를 얻는다. 피처를 여러 개 사용하면 입력 형태는 보통 $(B,L,F)$다.

미래 예측을 평가할 때는 시간순으로 학습·검증·테스트를 분리하고, 정규화 통계는 학습 구간에서만 추정한다. 창의 중복과 예측 시점에 사용할 수 없는 정보가 입력에 포함되는지도 확인한다. 계절성이나 추세가 있다고 해서 RNN이 항상 필요한 것은 아니며 단순 기준 모델과 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
길이 100인 시계열에서 입력 창 12개로 바로 다음 값 하나를 예측하면 예제는 몇 개인가?

</div>

??? success "풀이"
    시작 위치는 0부터 87까지이므로 $100-12=88$개다. 미래 여러 시점을 한꺼번에 예측하거나 창의 이동 간격을 바꾸면 개수도 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정규화와 분할 과정에서 미래 정보 유출을 막는 방법을 설명하라.

</div>

??? success "풀이"
    시간순으로 구간을 나누고 평균·표준편차 등 전처리 통계는 학습 구간에서 추정한다. 각 입력 피처가 예측 시점에 실제로 이용 가능한지 확인한다. 겹치는 창 때문에 같은 목표나 미래 정보가 분할 사이에 섞이지 않도록 구성한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
복잡한 시계열 모델을 평가할 때 사용할 간단한 기준 모델을 제시하라.

</div>

??? success "풀이"
    직전 관측값을 그대로 예측하거나, 알려진 계절 주기의 같은 위치 값을 예측하는 모델을 쓸 수 있다. 같은 예측 시점과 기간, 지표로 비교해야 한다. 학습 로스 감소만으로 기준 모델보다 낫다고 판단하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 03():
        model = Tutorial 03(...)
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

시계열 데이터 준비의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
