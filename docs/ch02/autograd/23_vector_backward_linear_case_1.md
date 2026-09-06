# 벡터 역전파 - 선형 사례 1

이 스크립트는 벡터 역전파의 선형 사례 1을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Vector backward linear case 1."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(1)

    # 선형 사상 설정:
    # A의 모양은 (2×3), x의 모양은 (3,)이므로 y = A @ x의 모양은 (2,)이다.
    A = torch.tensor([[2.0, 0.0, -1.0],
                      [0.5, 3.0,  1.0]], dtype=torch.float32)  # (2×3)
    x = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)     # (3,), leaf tensor
    y = A @ x                                                  # (2,)
    print("A:\n", A)
    print("x:", x)
    print("y = A @ x:", y)

    # y와 같은 모양(2,)의 상류 경사 v를 고른다.
    # 이는 벡터-야코비 곱 v^T J의 가중치를 고르는 것에 해당한다.
    v = torch.tensor([3.0, -1.0], dtype=torch.float32)

    # 역전파:
    # 선형 사상 y = A x에 대해 야코비안 J = A이다.
    # autograd가 계산한다: x.grad = J^T v = A^T v.
    y.backward(v)
    print("v:", v)
    print("x.grad (autograd):", x.grad)

    # 닫힌 형태로 확인: A^T에 v를 직접 곱한다.
    with torch.no_grad():
        expected = A.t() @ v
    print("A^T v (expected):  ", expected)

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
