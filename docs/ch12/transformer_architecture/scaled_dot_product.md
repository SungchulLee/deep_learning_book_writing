# 스케일 내적 어텐션

쿼리·키의 내적을 특징 차원의 제곱근으로 나누고 소프트맥스로 값 벡터를 결합하는 함수다. 투영 가중치는 이 함수 밖에서 관리하며, 이 페이지는 학습 모델이 아닌 핵심 연산을 정의한다.

## 1. 코드

```python
"""
배율 조정 내적 어텐션
"""
import torch
import math

# ========================================================================
# 메인
# ========================================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    배율 조정 내적 어텐션
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


if __name__ == "__main__":
    pass
```

## 2. 논의

Q와 K의 마지막 특징 차원은 같아야 하고 K와 V의 위치 수가 같아야 한다. 쿼리 길이 Nq, 키 길이 Nk이면 점수는 마지막 두 축이 $(N_q,N_k)$이며 출력 특징 차원은 V의 마지막 차원이다.

독립적인 평균 0·분산 1 성분의 내적 분산은 d_k여서 sqrt(d_k)로 나누는 설계 근거가 된다. 학습된 Q·K의 분산이나 독립성, 전체 학습의 안정성을 보장하는 식은 아니다.

마스크의 0 위치를 -inf로 바꾸고 소프트맥스를 적용한다. 유효 위치가 하나도 없으면 NaN이 생길 수 있으므로 별도 정책이 필요하다. 함수에 드롭아웃과 학습 모드 인수는 없으며 코드 블록은 호출·평가 결과를 포함하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 함수의 매개변수와 입력별 차원 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    함수 자체에는 학습 매개변수가 없다. Q·K의 특징 차원, K·V의 위치 수와 배치 축을 맞춘다. 필요한 Q·K·V 투영은 외부 모듈에 있다. 입력 dtype·장치와 마스크 브로드캐스트도 호환되어야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
마스크가 한 행의 모든 위치를 제외하면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"
    모든 점수가 -inf가 되어 일반 소프트맥스가 유효한 정규화 값을 계산하지 못한다. 빈 입력을 거부하거나 별도의 0 출력 규칙 등을 정해야 한다. 작은 유한 음수로 바꾸기만 하면 그 행이 균등 분포가 될 수 있어 같은 의미가 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
명시적 점수 행렬을 저장하는 셀프 어텐션의 비용을 설명하라.

</div>

??? success "연습문제 3 풀이"
    길이 n, 특징 d에서 내적과 값 가중합은 $O(n^2d)$이며 가중치 저장은 헤드·배치를 고정하면 $O(n^2)$다. 메모리 효율 커널은 전체 행렬 저장을 피할 수 있지만 이 함수는 직접 생성한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_scaled dot product():
        model = Scaled Dot Product(...)
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

스케일 내적 함수의 입력 규약과 마스크 유효성을 먼저 확인한다. 스케일의 통계적 근거와 실제 학습의 안정성은 구분하며, 가중치·모델 학습은 외부에서 구성해야 한다.
