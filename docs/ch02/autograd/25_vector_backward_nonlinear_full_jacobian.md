# 벡터 역전파 - 비선형 전체 야코비안

이 스크립트는 비선형 함수의 전체 야코비안을 구하는 벡터 역전파을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""벡터 역전파, 비선형 온 야코비."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(2)

    # 작은 비선형 사상 f: R^2 -> R^2를 정의한다
    def f(x):
        # x: 모양 [2]
        # x의 두 성분에 모두 의존하는 2차원 벡터 출력을 반환한다
        return torch.stack([
            torch.sin(x[0]) + x[1] ** 2,   # first component
            torch.exp(x[0]) - torch.cos(x[1])  # second component
        ])

    x = torch.tensor([0.3, -0.7], requires_grad=True)
    y = f(x)  # output is a vector (shape [2])
    print("x:", x)
    print("y = f(x):", y)

    # 벡터 출력에 대한 경사를 얻으려면 "v" 벡터를 넘겨야 한다.
    # 이는 벡터-야코비 곱 v^T J(x)를 계산한다.
    v = torch.tensor([0.2, -1.5], dtype=torch.float32)
    y.backward(v)  # backward on vector → computes v^T J wrt x
    print("v:", v)
    print("x.grad (v^T J):", x.grad)

    # 검증을 위해: 전체 야코비안 J(x)를 명시적으로 계산한다.
    # 그런 다음 autograd의 v^T J 결과를 직접 계산한 v @ J와 비교한다.
    # 참고: 큰 문제에서는 야코비안을 구성하는 비용이 크다.
    try:
        from torch.autograd.functional import jacobian

        def f_for_jac(x_vec):
            return f(x_vec)  # required to wrap for jacobian API

        J = jacobian(f_for_jac, x.detach().requires_grad_(True))  # shape (2,2)
        with torch.no_grad():
            vT_J = v @ J  # shape (2,) after multiplication
        print("Full Jacobian J:\n", J)
        print("v^T J via full J:", vT_J)
        print("Difference |x.grad - (v^T J)|:", (x.grad - vT_J).abs())
    except Exception as e:
        print("Could not compute full Jacobian (version/platform issue):", e)

if __name__ == "__main__":
    main()```

## 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

## 연습문제

**연습문제 1.**
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    r1 = A @ B
    r2 = torch.mm(A, B)
    r3 = torch.matmul(A, B)
    print(torch.allclose(r1, r2) and torch.allclose(r2, r3))  # True
    ```

---


**연습문제 2.**
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


**연습문제 3.**
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```
