# 벡터 역전파의 함정

이 스크립트는 벡터 역전파에서 빠지기 쉬운 함정을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""벡터 역전파의 함정."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(3)

    x = torch.randn(4, requires_grad=True)
    y = torch.tanh(x)  # elementwise nonlinearity, output is vector (non-scalar)
    print("x:", x)
    print("y = tanh(x):", y)

    # (a) 잘못된 v 모양 → 오류.
    # 스칼라가 아닌 출력에 대해 역전파는 명시적인 경사 인수 v를 요구한다
    # ("벡터-야코비 곱").
    # 수학적으로: sum_i v_i * ∂y_i/∂x_j
    # 따라서 v의 길이는 y의 길이와 같아야 한다.
    try:
        v_wrong = torch.tensor([1.0, 2.0])  # wrong length → mismatch with y of length 4
        y.backward(v_wrong)
    except RuntimeError as e:
        print("Shape mismatch as expected:", e)

    # (b) 올바른 v 모양.
    # 여기서는 출력에 균등한 가중치를 주는 v를 고른다(시연을 위해서일 뿐이다).
    v = torch.ones_like(y) / y.numel()
    # 역전파는 벡터-야코비 곱 v^T * J를 계산한다
    y.backward(v)
    print("After first backward, x.grad:", x.grad)

    # (c) 지우지 않으면 경사가 누적된다.
    # 다른 출력에 대해 역전파를 또 하면 기존 .grad 값에 더해진다.
    y2 = x ** 2
    y2.backward(v)  # second backward accumulates into x.grad
    print("After second backward (accumulated), x.grad:", x.grad)

    # 새로 시작하려면 경사를 지운다(zero_()를 쓰거나 None으로 설정).
    x.grad.zero_()
    print("After zero_(), x.grad:", x.grad)

if __name__ == "__main__":
    main()
```

## 2. 논의

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
`torch.eye`로 $4 \times 4$ 단위 행렬을 만들고, `torch.diag`로 대각 성분이 $[1, 2, 3, 4]$인 대각 행렬을 만들어라. 둘이 다름을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    I = torch.eye(4)
    D = torch.diag(torch.tensor([1., 2., 3., 4.]))
    print(torch.equal(I, D))  # False -- D has non-unit diagonal
    ```

---


**연습문제 2.**
`torch.empty()`와 `torch.zeros()`의 차이를 설명하라. `empty()`는 언제 쓰는 것이 적절한가?

??? success "연습문제 2 풀이"
    `torch.zeros()`는 모든 원소를 0으로 초기화하지만 `torch.empty()`는 초기화 없이 메모리만 할당한다(값은 그 메모리에 이전에 있던 것이 무엇이든 그대로이다). 초기화되지 않은 내용은 예측할 수 없으므로 `empty()`는 모든 값을 곧바로 덮어쓸 계획일 때만 써야 한다. 0으로 채우는 단계를 건너뛰므로 조금 더 빠르다.

---


**연습문제 3.**
`torch.linspace`로 $-\pi$와 $\pi$ 사이에 균등 간격의 점 100개를 만들어라. 각 점에서 $\sin(x)$를 계산하고 절댓값의 최댓값이 대략 1임을 확인하라.

??? success "연습문제 3 풀이"
    ```python
    import torch, math
    x = torch.linspace(-math.pi, math.pi, 100)
    y = torch.sin(x)
    print(y.abs().max())  # Approximately 1.0
    ```

## 정리하며

**다룬 것** — 벡터 역전파의 함정

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
