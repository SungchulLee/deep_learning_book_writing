# 벡터 역전파의 함정

경사 인수의 모양 오류와 의도하지 않은 경사 누적을 살펴본다. 올바른 `v`를 전달하는 것과 이전 경사를 지우는 것은 별개의 점검 항목이다.

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
    # 여기서는 출력에 균등한 웨이트를 주는 v를 고른다(시연을 위해서일 뿐이다).
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

**출력:**

```
x: tensor([ 0.8033,  0.1748,  0.0890, -0.6137], requires_grad=True)
y = tanh(x): tensor([ 0.6659,  0.1731,  0.0887, -0.5467], grad_fn=<TanhBackward0>)
Shape mismatch as expected: Mismatch in shape: grad_output[0] has a shape of torch.Size([2]) and output[0] has a shape of torch.Size([4]).
After first backward, x.grad: tensor([0.1392, 0.2425, 0.2480, 0.1753])
After second backward (accumulated), x.grad: tensor([ 0.5408,  0.3299,  0.2925, -0.1316])
After zero_(), x.grad: tensor([0., 0., 0., 0.])
```

## 2. 논의

### 출력과 경사 인수의 모양

길이 4인 `tanh(x)` 출력에 길이 2의 `v`를 주면 모양이 맞지 않아 오류가 발생한다. 예제의 올바른 `v`는 `ones_like(y) / y.numel()`이며, 출력 평균의 경사를 구한다. 수학적으로 첫 경사는 각 성분에서 $(1-\tanh^2 x_i)/4$다.

### 다른 그래프여도 경사는 누적된다

그다음 `y2 = x ** 2`를 새로 만들고 같은 `v`로 역전파한다. 새로 계산한 경사 $2x_i/4$는 기존 `.grad`에 더해진다. 그래프를 새로 만든다고 경사 저장값까지 초기화되는 것은 아니다. 두 결과를 독립적으로 비교하려면 두 번째 역전파 전에 경사를 지워야 한다.

`v`의 모양을 맞춘 뒤에도 어떤 출력 조합을 미분하는지, `.grad`에 이전 값이 남아 있는지 확인한다. 단순히 오류 없이 실행되는 것과 원하는 미분을 계산하는 것은 다를 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`torch.eye`로 $4 \times 4$ 단위 행렬을 만들고, `torch.diag`로 대각 성분이 $[1, 2, 3, 4]$인 대각 행렬을 만들어라. 둘이 다름을 확인하라.

</div>

??? success "풀이"
    ```python
    I = torch.eye(4)
    D = torch.diag(torch.tensor([1., 2., 3., 4.]))
    print(torch.equal(I, D))  # False -- D has non-unit diagonal
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.empty()`와 `torch.zeros()`의 차이를 설명하라. `empty()`는 언제 쓰는 것이 적절한가?

</div>

??? success "풀이"
    `torch.zeros()`는 0으로 초기화한다. `torch.empty()`의 값은 일반적으로 초기화되지 않으므로 읽기 전에 사용할 모든 원소를 채워야 한다. 초기화 비용을 줄일 수 있지만 실제 성능 차이는 작업에 따라 다르며, 결정론적 실행의 설정에 따라 미정의 값 사용을 막기 위한 채움이 수행될 수도 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`torch.linspace`로 $-\pi$와 $\pi$ 사이에 균등 간격의 점 100개를 만들어라. 각 점에서 $\sin(x)$를 계산하고 절댓값의 최댓값이 대략 1임을 확인하라.

</div>

??? success "풀이"
    ```python
    import torch, math
    x = torch.linspace(-math.pi, math.pi, 100)
    y = torch.sin(x)
    print(y.abs().max())  # Approximately 1.0
    ```

## 정리하며

벡터 역전파에서는 출력과 `v`의 모양, `v`가 정하는 목적, 경사 초기화 여부를 함께 확인한다.
