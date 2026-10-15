# 배치 VJP

배치 입력 전체에 대해 출력 가중합의 입력 경사를 계산한다. 배치 축과 특징 축이 각각 어떻게 보존되는지 살펴본다.

## 1. 코드

```python
"""배치 벡터-야코비 곱."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():
    torch.manual_seed(4)

    B, m, n = 2, 3, 2
    A = torch.randn(n, m)  # fixed weight matrix (not trainable here)

    # ⚠️ 입력을 학습 가능한 매개변수로 취급하기
    x_b = torch.randn(B, m, requires_grad=True)

    # 순전파: batch_output = x_b @ A^T
    batch_output = x_b @ A.t()

    # 상류 경사(batch_output과 같은 모양)
    v_b = torch.tensor([[1.0, 0.0], [0.5, -2.0]], dtype=torch.float32)
    batch_output.backward(v_b)

    # 기댓값: x_b에 대한 경사 = v_b @ A
    expected = v_b @ A
    print("=== Trainable Inputs Demo ===")
    print("A:\n", A)
    print("x_b.grad (autograd):\n", x_b.grad)
    print("Expected v_b @ A:\n", expected)
    print()

if __name__ == "__main__":
    main()
```

**출력:**

```
=== Trainable Inputs Demo ===
A:
 tensor([[-1.6053,  0.2325,  2.2399],
        [ 0.8473,  1.2006, -0.4016]])
x_b.grad (autograd):
 tensor([[-1.6053,  0.2325,  2.2399],
        [-2.4972, -2.2850,  1.9230]])
Expected v_b @ A:
 tensor([[-1.6053,  0.2325,  2.2399],
        [-2.4972, -2.2850,  1.9230]])
```

## 2. 논의

### 행 단위로 표현한 선형 사상

입력 `x_b`의 모양은 $(B,m)$, 고정 행렬 $A$는 $(n,m)$이다. 순전파 `x_b @ A.t()`의 출력은 $(B,n)$이다. 각 행이 한 표본이므로 한 표본의 수학적 열벡터 표현 $y=Ax$를 배치 행렬 곱으로 옮긴 형태다.

### 배치 경사 인수의 의미

`v_b`도 $(B,n)$이며 배치의 모든 출력 성분에 대한 가중치를 지정한다. 역전파 결과는 $(B,m)$ 모양의 `v_b @ A`다. 예제의 표본들은 서로 독립적으로 계산되므로 각 행의 경사가 해당 입력 행에만 전달된다. 배치 통계를 공유하는 연산이 있으면 이런 독립성을 그대로 가정할 수 없다.

입력에 `requires_grad=True`를 주면 `.grad`를 구할 수 있지만 그것만으로 입력이 자동 업데이트되지는 않는다. 이 예제는 입력 민감도를 계산하며, $A$를 학습하거나 입력을 최적화하는 업데이트 단계는 포함하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$A$가 $3 \times 2$ 행렬인 선형 사상 $y = Ax$에 대해, 명시적인 경사 벡터 $v$와 함께 PyTorch의 `.backward()`를 사용하여 VJP $v^\top J$를 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    A = torch.randn(3, 2)
    x = torch.randn(2, requires_grad=True)
    y = A @ x  # shape (3,)
    v = torch.tensor([1.0, 0.0, 0.0])  # select first row of Jacobian
    y.backward(v)
    print(x.grad)  # Should equal A.t() @ v = A[0, :]
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
출력이 스칼라가 아닐 때 PyTorch가 `.backward()`에 경사 인수를 요구하는 이유를 설명하라. 이 인수는 어떤 수학적 대상을 나타내는가?

</div>

??? success "연습문제 2 풀이"
    스칼라 손실 $L$에 대해서는 `.backward()`가 $\partial L / \partial x$를 직접 계산한다. 벡터 출력 $y \in \mathbb{R}^m$에 대해서는 하나의 경사가 아니라 야코비안 행렬 $J \in \mathbb{R}^{m \times n}$이 있다. 경사 인수 $v$는 벡터-야코비안 곱 $v^\top J$를 지정하며, 이는 야코비안을 하나의 경사 벡터로 사영한다. 이는 $\partial (v^\top y) / \partial x$를 계산하는 것과 동등하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`torch.autograd.functional.jacobian`을 사용하여 $x = [1, 2]$에서 $f(x) = [\sin(x_1), x_1 x_2, x_2^2]$의 전체 야코비안을 계산하라. 각 편도함수를 직접 계산하여 검증하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch
    from torch.autograd.functional import jacobian

    def f(x):
        return torch.stack([torch.sin(x[0]), x[0]*x[1], x[1]**2])

    x = torch.tensor([1.0, 2.0])
    J = jacobian(f, x)
    print(J)
    # 기대되는 야코비안:
    # [[cos(1), 0   ],
    #  [2,      1   ],
    #  [0,      4   ]]
    ```

## 정리하며

배치 VJP는 배치 출력 전체의 가중합을 미분한다. 경사는 입력과 같은 모양이며, 표본 간 독립성은 사용한 연산에 따라 달라진다.
