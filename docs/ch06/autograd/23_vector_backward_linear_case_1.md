# 벡터 역전파: 선형 사상

일차원 텐서로 표현한 선형 사상 $y=Ax$에서 VJP를 직접 유도한다. autograd의 경사를 $A^\top v$와 비교한다.

## 1. 코드

```python
"""벡터 역전파, 선형 자리 1."""
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
    # 이는 벡터-야코비 곱 v^T J의 웨이트를 고르는 것에 해당한다.
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
    main()
```

**출력:**

```
A:
 tensor([[ 2.0000,  0.0000, -1.0000],
        [ 0.5000,  3.0000,  1.0000]])
x: tensor([ 1.0000, -2.0000,  0.5000], requires_grad=True)
y = A @ x: tensor([ 1.5000, -5.0000], grad_fn=<MvBackward0>)
v: tensor([ 3., -1.])
x.grad (autograd): tensor([ 5.5000, -3.0000, -4.0000])
A^T v (expected):   tensor([ 5.5000, -3.0000, -4.0000])
```

## 2. 논의

### 모양으로 미분 결과 확인하기

$A$는 $(2,3)$, $x$는 $(3,)$, $y$는 $(2,)$이다. $J_{ij}=\partial y_i/\partial x_j=A_{ij}$이므로 야코비안은 $A$ 자체다. 출력 모양의 `v`를 전달하면 입력 경사는 $(3,)$ 모양의 $A^\top v$가 된다.

예제의 $v=[3,-1]$은 $3y_1-y_2$라는 스칼라 목적을 정한다. 따라서 경사는 $[5.5,-3,-4]$이다. 이는 각 출력에 대한 미분을 가중합한 결과이며, 입력 $x$의 값과 무관하다. 선형 사상의 야코비안이 상수이기 때문이다.

비교용 계산은 `no_grad()` 안에서 수행한다. 이는 기대값을 확인하기 위한 계산이며, 새 역전파 경로를 만들 필요가 없다. 예제에서 `A`는 경사를 요구하지 않으므로 `.grad`가 저장되는 대상은 `x`다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$(2, 3)$ 행렬과 $(3, 4)$ 행렬의 곱을 `@`, `torch.mm()`, `torch.matmul()` 세 가지 방법으로 계산하라. 결과가 동일함을 확인하라.

</div>

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


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원소별 곱(`*`)과 행렬 곱(`@`)의 차이를 설명하라. 같은 입력에 대해 서로 다른 결과를 내는 예를 들라.

</div>

??? success "연습문제 2 풀이"
    원소별 곱은 대응하는 원소끼리 곱한다. 같은 모양의 행렬 $A, B$에 대해 $(A * B)_{ij} = A_{ij} B_{ij}$이다. 행렬 곱은 안쪽 차원에 대해 더한다. $(A @ B)_{ij} = \sum_k A_{ik} B_{kj}$이다. $A = B = [[1,2],[3,4]]$일 때 원소별 곱은 $[[1,4],[9,16]]$을, 행렬 곱은 $[[7,10],[15,22]]$을 준다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
`torch.einsum`을 사용하여 모양 $(10, 4, 4)$인 텐서의 배치 대각합을 계산하고 모양 $(10,)$인 텐서를 반환하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    A = torch.randn(10, 4, 4)
    traces = torch.einsum('bii->b', A)
    print(traces.shape)  # torch.Size([10])
    # 직접 계산한 값과 대조 확인:
    manual = torch.stack([torch.trace(A[i]) for i in range(10)])
    print(torch.allclose(traces, manual))  # True
    ```

## 정리하며

선형 사상의 입력 경사는 $A^\top v$다. 행렬과 텐서의 모양을 먼저 확인하면 전치 방향과 경사 인수의 역할을 쉽게 점검할 수 있다.
