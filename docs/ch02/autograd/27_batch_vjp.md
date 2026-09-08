# 배치 VJP

이 스크립트는 배치 단위의 벡터-야코비 곱을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

벡터-야코비 곱(VJP)은 후진 모드 자동 미분의 핵심 기본 연산이다. 야코비안이 $J$인 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 VJP는 주어진 벡터 $v$에 대한 $v^\top J$를 계산한다. 출력이 스칼라가 아닐 때 PyTorch는 어떤 출력들의 선형 결합을 미분할지 지정하도록 `.backward()`에 경사 인수 $v$를 명시할 것을 요구한다.

행렬 연산은 신경망 계산의 핵심을 이룬다. `@` 연산자와 `torch.matmul()`은 배치 차원에 대한 자동 브로드캐스팅과 함께 행렬 곱을 처리하고, `torch.mm()`이나 `torch.bmm()` 같은 특화된 함수는 명료함과 성능을 위해 텐서의 계수를 특정 값으로 강제한다.

텐서 생성 함수는 데이터를 초기화하는 유연한 방법을 제공한다. `torch.zeros`, `torch.randn`, `torch.arange` 같은 팩토리 함수는 `dtype`, `device`, `requires_grad` 매개변수를 받으므로 불필요한 복사 없이 목표 장치에 곧바로 할당할 수 있다.

## 연습문제

**연습문제 1.**
$A$가 $3 \times 2$ 행렬인 선형 사상 $y = Ax$에 대해, 명시적인 경사 벡터 $v$와 함께 PyTorch의 `.backward()`를 사용하여 VJP $v^\top J$를 계산하라.

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


**연습문제 2.**
출력이 스칼라가 아닐 때 PyTorch가 `.backward()`에 경사 인수를 요구하는 이유를 설명하라. 이 인수는 어떤 수학적 대상을 나타내는가?

??? success "연습문제 2 풀이"
    스칼라 손실 $L$에 대해서는 `.backward()`가 $\partial L / \partial x$를 직접 계산한다. 벡터 출력 $y \in \mathbb{R}^m$에 대해서는 하나의 경사가 아니라 야코비 행렬 $J \in \mathbb{R}^{m \times n}$이 있다. 경사 인수 $v$는 벡터-야코비 곱 $v^\top J$를 지정하며, 이는 야코비안을 하나의 경사 벡터로 사영한다. 이는 $\partial (v^\top y) / \partial x$를 계산하는 것과 동등하다.

---


**연습문제 3.**
`torch.autograd.functional.jacobian`을 사용하여 $x = [1, 2]$에서 $f(x) = [\sin(x_1), x_1 x_2, x_2^2]$의 전체 야코비안을 계산하라. 각 편도함수를 직접 계산하여 검증하라.

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

**다룬 것** — 배치 VJP

벡터-야코비 곱(VJP)은 후진 모드 자동 미분의 핵심 기본 연산이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
