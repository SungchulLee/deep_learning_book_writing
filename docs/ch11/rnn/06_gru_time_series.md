# GRU 시계열 예측

GRU는 업데이트 게이트와 리셋 게이트로 은닉 상태의 유지와 갱신을 조절한다. 시계열 예측에서는 과거 구간으로 미래 값을 예측하는 모델에 사용할 수 있다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 06 — RNN 심화 주제
=====================================
전체 구현을 갖춘 종합 실습

제대로 돌아가는 RNN 실습 모듈이다.
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

print("Tutorial 06 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 06_tutorial.py --epochs 20")

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
Tutorial 06 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 06_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

입력 시계열의 창, 피처 수와 예측 기간을 먼저 정한다. 과거 $L$개 시점에서 다음 값 하나를 예측하는 문제와 미래 $K$개 값을 예측하는 문제는 출력 형태와 로스 집계가 다르다. GRU를 쓰는 것만으로 이 선택이 결정되지는 않는다.

GRU는 별도 셀 상태가 없지만 은닉 상태를 갖는 순환 모델이다. LSTM보다 파라메터가 적은 설정이 일반적이어도 실제 속도나 정확도의 우열이 항상 같지는 않다. 시간순 검증과 단순 예측 기준 모델을 함께 사용한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GRU와 LSTM이 유지하는 상태의 차이를 설명하라.

</div>

??? success "풀이"
    GRU는 은닉 상태 하나를 사용하고 LSTM은 은닉 상태와 셀 상태를 구분한다. 둘 다 상태를 시점 간 전달하며, 상태 하나라는 이유로 GRU가 장기 의존성을 항상 더 잘 또는 더 못 학습한다고 결론 내릴 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력 피처 수 F와 예측 기간 K가 각각 어떤 차원에 대응하는가?

</div>

??? success "풀이"
    배치 입력을 $(B,L,F)$로 만들면 F는 각 시점의 관측 피처 수다. 미래 K개 값을 직접 예측하는 단변량 모델의 출력은 $(B,K)$ 등으로 설계할 수 있다. 타깃과 출력의 형태를 일치시켜 의도하지 않은 브로드캐스팅을 막는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모든 구간을 함께 정규화한 뒤 시간순 분할하면 왜 문제가 되는가?

</div>

??? success "풀이"
    미래 구간의 분포 통계가 학습 전처리에 반영된다. 시간순 분할 후 학습 구간에서 통계를 추정하고 이후 구간에 같은 변환을 적용해야 한다. 미래의 실제값으로 역정규화 기준을 다시 맞춰서도 안 된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 06():
        model = Tutorial 06(...)
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

GRU 시계열 예측의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
