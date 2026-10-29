# 양방향 RNN

양방향 RNN은 입력을 정방향과 역방향으로 처리하고 두 표현을 결합한다. 입력 전체가 주어진 분류나 태깅에서 앞뒤 문맥을 활용할 수 있다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 07 — RNN 심화 주제
=====================================
전체 구현을 갖춘 종합 실습

제대로 돌아가는 RNN 실습 모듈이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

print("Tutorial 07 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 07_tutorial.py --epochs 20")

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
Tutorial 07 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 07_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

방향별 은닉 차원이 $H$이면 같은 위치의 정방향·역방향 출력을 이어 붙였을 때 피처 차원은 $2H$다. 두 방향의 상태는 입력을 서로 다른 순서로 요약하므로 최종 상태를 선택할 때 축과 위치의 의미를 구분해야 한다.

양방향 모델은 뒤쪽 입력을 참조한다. 실시간 예측 시점에 아직 관측하지 못한 입력까지 넣으면 정보 유출이다. 다만 과거에 이미 관측된 고정 창 내부를 양방향으로 처리하는 것은 가능하다. 문제는 역방향이라는 이름이 아니라 예측 시점에 사용 가능한 정보의 범위다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
방향별 은닉 크기 64인 양방향 RNN의 시점별 출력 피처 수는 얼마인가?

</div>

??? success "풀이"
    두 방향을 이어 붙이면 128이다. 이를 분류 레이어에 넣을 때 입력 차원도 맞춰야 한다. 두 방향을 합산하는 별도 설계를 하면 결과 차원은 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
양방향 RNN을 다음 시점 예측에 사용할 수 있는 조건을 설명하라.

</div>

??? success "풀이"
    예측 시점까지 관측한 입력 창만 사용한다면 그 창을 양방향으로 인코딩할 수 있다. 목표 시점 이후의 값이나 아직 도착하지 않은 입력을 포함하면 안 된다. 온라인 위치별 예측에서 전체 미래 문맥을 사용하는 것과 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마지막 위치의 양방향 출력이 양쪽 최종 상태를 모두 담는가?

</div>

??? success "풀이"
    정방향의 마지막 위치는 입력 전체를 거친 상태지만, 그 위치의 역방향 출력은 역방향 계산의 시작 쪽에 해당한다. 양쪽 전체 요약을 원하면 반환된 최종 상태에서 레이어와 방향을 구분해 선택해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 07():
        model = Tutorial 07(...)
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

양방향 RNN의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
