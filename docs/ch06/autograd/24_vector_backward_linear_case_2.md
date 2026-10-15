# 벡터 역전파: 열벡터 표현

같은 선형 사상을 $(3,1)$ 열벡터 입력으로 표현한다. 수학적으로 같은 계산도 텐서의 모양이 다르면 경사 인수의 모양을 맞춰야 한다.

## 1. 코드

```python
"""벡터 역전파, 선형 자리 2."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(1)

    # 설정: 선형 사상 A: R^m → R^n
    # A의 모양은 (2,3), x의 모양은 (3,1)이므로 y = A @ x의 모양은 (2,1)이다.
    A = torch.tensor([[2.0, 0.0, -1.0],
                      [0.5, 3.0,  1.0]], dtype=torch.float32)        # (2×3)
    x = torch.tensor([[1.0], [-2.0], [0.5]], requires_grad=True)     # (3×1), leaf param
    y = A @ x                                                        # (2×1)
    print("A:\n", A)
    print("x:", x)
    print("y = A @ x:", y)

    # y와 같은 모양(2×1)의 벡터 v ∈ R^2를 고른다.
    # 이것이 벡터-야코비 곱에서 상류 경사의 역할을 한다.
    v = torch.tensor([[3.0], [-1.0]], dtype=torch.float32)

    # 역전파:
    # autograd가 계산한다: x.grad = J^T v이며 J = ∂y/∂x = A이다(여기서는 상수).
    # 따라서 수학적으로 x.grad = A^T v이다.
    y.backward(v)
    print("v:", v)
    print("x.grad (autograd):", x.grad)

    # 검증을 위한 닫힌 형태의 직접 계산: A^T v
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
x: tensor([[ 1.0000],
        [-2.0000],
        [ 0.5000]], requires_grad=True)
y = A @ x: tensor([[ 1.5000],
        [-5.0000]], grad_fn=<MmBackward0>)
v: tensor([[ 3.],
        [-1.]])
x.grad (autograd): tensor([[ 5.5000],
        [-3.0000],
        [-4.0000]])
A^T v (expected):   tensor([[ 5.5000],
        [-3.0000],
        [-4.0000]])
```

## 2. 논의

### 일차원 벡터와 열벡터

이 예제의 $x$는 $(3,1)$, $y=Ax$는 $(2,1)$이다. 앞 예제의 $(3,)$, $(2,)$와 원소 수는 같지만 차원 수가 다르다. `y.backward(v)`의 `v` 역시 $(2,1)$로 제공하며, 입력 경사는 $(3,1)$로 저장된다.

### 같은 미분을 다른 모양으로 표현하기

야코비안의 성분은 여전히 $A$이고 입력 경사는 $A^\top v$이다. 일반적인 텐서 야코비안 API로 직접 구하면 출력 축과 입력 축을 함께 가진 모양이 될 수 있으나, VJP 결과는 입력과 같은 모양으로 반환된다.

수치적으로는 앞 예제와 같은 $[5.5,-3,-4]$를 열 형태로 얻는다. 이처럼 벡터와 행렬을 오가며 계산할 때는 `shape`를 명시적으로 확인해야 한다. 브로드캐스팅에 맡겨 모양 차이를 무시하면 다른 연산이 만들어질 수 있다.

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

VJP의 수학적 의미는 같아도 텐서 모양은 그대로 유지된다. 입력, 출력, 경사 인수의 모양을 한 세트로 확인한다.
