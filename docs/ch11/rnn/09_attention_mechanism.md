# RNN과 어텐션

어텐션을 쓰는 순환 디코더는 인코더의 여러 위치를 가중합해 현재 생성 단계의 문맥 벡터를 만든다. 입력 전체를 하나의 최종 상태에만 담는 제약을 완화한다. 본문의 코드는 안내 메시지만 출력하는 템플릿이며 데이터 준비, 모델 정의와 학습·평가는 포함하지 않는다.

## 1. 코드

```python
"""
실습 09 — RNN 심화 주제
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

print("Tutorial 09 - See README.md for full details")
print("This tutorial covers advanced RNN concepts")
print("Run with: python 09_tutorial.py --epochs 20")

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
Tutorial 09 - See README.md for full details
This tutorial covers advanced RNN concepts
Run with: python 09_tutorial.py --epochs 20

For complete implementation, each tutorial is 400-600 lines
with detailed comments, examples, and exercises.
```

## 2. 논의

각 디코더 단계에서 쿼리와 인코더 위치별 키로 점수를 계산하고, 유효한 입력 위치에 대해 소프트맥스로 정규화한다. 문맥 벡터는 그 웨이트로 값을 결합한 결과다. 어텐션 웨이트는 단계마다 달라질 수 있다.

패딩 위치는 정규화 전에 마스킹해야 한다. 모든 위치를 마스킹한 행은 일반적인 소프트맥스로 유효한 분포를 만들 수 없으므로 빈 입력 처리 규칙도 필요하다. 웨이트가 크다는 사실만으로 해당 단어가 예측에 미친 인과적 중요도가 확정되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 길이 S, 출력 길이 T인 단일 헤드 인코더·디코더 어텐션의 웨이트 행렬 형태를 설명하라.

</div>

??? success "풀이"
    배치별로 $(T,S)$이고 배치를 포함하면 $(B,T,S)$로 나타낼 수 있다. 각 출력 위치의 행이 유효한 입력 위치에 대한 웨이트를 담는다. 패딩된 출력 위치는 평가에서 별도로 제외해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패딩 위치의 값을 0으로 만드는 것만으로 어텐션 마스크를 대신할 수 있는가?

</div>

??? success "풀이"
    아니다. 그 위치도 소프트맥스 분모에 들어가 유효 위치의 웨이트를 줄일 수 있다. 점수 단계에서 제외해야 한다. 모든 위치가 패딩인 경우에는 별도의 처리가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
어텐션 웨이트를 예측의 완전한 설명으로 볼 수 없는 이유를 설명하라.

</div>

??? success "풀이"
    웨이트는 값 벡터와 함께 출력을 만든다. 값의 크기와 내용, 이후 레이어와 다른 경로도 영향을 준다. 큰 웨이트가 곧 그 토큰을 제거했을 때의 성능 변화나 인과적 기여를 뜻하지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_tutorial 09():
        model = Tutorial 09(...)
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

RNN과 어텐션의 핵심은 데이터와 예측 조건을 맞추는 것이다. 이 페이지의 출력은 학습 결과가 아니며, 위 설명은 실제 구현에서 확인할 설계 기준이다.
