# 기본 Seq2Seq

Seq2Seq는 인코더로 입력 시퀀스를 표현하고 디코더로 출력 시퀀스를 생성하는 구조다. 입력과 출력의 길이가 달라도 적용할 수 있다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 08 — RNN 심화 주제
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

print("Tutorial 08 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 08_tutorial.py --epochs 20")

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
Tutorial 08 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 08_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

기본 순환 인코더·디코더는 인코더의 최종 상태 등을 디코더의 초기 상태로 전달한다. LSTM이면 은닉 상태와 셀 상태의 연결을 함께 고려해야 한다. 두 모델의 레이어수나 차원이 다르면 변환 레이어가 필요할 수 있다.

학습에서 이전 정답 토큰을 디코더 입력으로 넣는 방식을 티처 포싱이라고 한다. 추론에서는 정답 대신 이전 예측 토큰을 사용한다. 시작 토큰, 종료 토큰과 최대 생성 길이를 정하고, 로스에서 패딩 위치를 제외해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 인코더·디코더에서 입력과 출력 길이가 같을 필요가 없는 이유를 설명하라.

</div>

??? success "풀이"
    인코더가 입력을 상태로 요약하고 디코더는 그 상태와 이전 토큰으로 한 단계씩 생성한다. 출력은 종료 토큰이나 길이 제한까지 계속될 수 있어 입력 길이와 일대일 대응하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
티처 포싱의 학습 입력과 추론 입력을 비교하라.

</div>

??? success "풀이"
    티처 포싱은 이전 정답을 사용한다. 추론은 이전 예측을 사용하므로 오류가 다음 단계의 입력에도 영향을 준다. 학습 로스와 실제 생성 결과의 차이를 해석할 때 이 조건을 고려해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
디코더가 종료 토큰을 내지 않으면 어떻게 해야 하는가?

</div>

??? success "풀이"
    최대 생성 길이 등의 종료 조건을 둔다. 종료 토큰을 학습 목표에 포함했는지, 잘림이나 패딩 처리로 사라지지 않았는지도 점검한다. 길이 제한에 도달한 생성은 자연스럽게 종료한 출력과 구분한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 08():
        model = Tutorial 08(...)
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

기본 Seq2Seq의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
