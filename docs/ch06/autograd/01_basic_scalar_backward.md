# 스칼라 로스의 역전파

벡터 입력의 제곱합을 미분하며 `requires_grad`, `grad_fn`, `is_leaf`, `.grad`의 역할을 구분한다. 이어서 중간 텐서의 경사를 보관하는 `retain_grad()`를 살펴본다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
==============================================================================
첫스텝 튜토리얼 01: 홑값 역전파 기초
==============================================================================

학습 목표:
-------------------
1. PyTorch의 자동 미분 구조에서 "잎" 텐서가 무엇인지 이해한다
2. requires_grad=True로 그래디언트 좇기를 켜는 법을 배운다
3. 홑값 로스에 .backward()을 써서 그래디언트를 셈하는 법을 본다
4. 잎 텐서와 잎이 아닌 텐서의 다름을 이해한다
5. 잎이 아닌 텐서를 위한 .retain_grad()을 배운다

핵심 개념:
------------
- 잎 텐서: 쓰는 이가 곧바로 만든 텐서(셈에서 나온 것이 아니다)
- 셈 그래프: PyTorch가 셈을 좇으려고 절로 세운다
- 그래디언트 쌓임: 드러내 놓고 0으로 만들지 않으면 그래디언트가 더해진다
- 벡터-야코비 곱(VJP): PyTorch가 그래디언트를 잘 들게 셈하는 방식

==============================================================================
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def main():
    """
    홑값 로스 함수에 대한 기본 그래디언트 셈을 보인다.
    
    다음을 계산한다: loss = sum(x^2)
    바라는 그래디언트: d(loss)/dx = 2*x
    """
    
    # 재현성을 위한 난수 시드 설정
    torch.manual_seed(0)
    
    print("="*70)
    print("PART 1: Creating a Leaf Tensor with Gradient Tracking")
    print("="*70)
    
    # 경사 추적을 켠 잎 텐서를 만든다
    # "잎" 텐서는 연산이 아니라 사용자가 만든 텐서이다
    x = torch.randn(3, requires_grad=True)  # Shape: (3,)
    
    # 대안: 리스트로부터 만들 수도 있다
    # x = torch.tensor([1., 2., 3.], requires_grad=True)
    
    print(f"x: {x}")
    print(f"x.shape: {x.shape}")
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"x.is_leaf: {x.is_leaf}")  # True because we created it directly
    print(f"x.grad_fn: {x.grad_fn}")  # None for leaf tensors
    print(f"x.grad (before backward): {x.grad}")  # None initially
    print()
    
    print("="*70)
    print("PART 2: Forward Pass - Building the Computational Graph")
    print("="*70)
    
    # 순전파: loss = sum(x^2)을 계산한다
    # PyTorch는 계산 그래프를 자동으로 만든다:
    # x → x**2 → sum → loss
    loss = (x ** 2).sum()  # Shape: scalar ()
    
    print(f"loss: {loss}")
    print(f"loss.shape: {loss.shape}")
    print(f"loss.is_leaf: {loss.is_leaf}")  # False - it's computed from x
    print(f"loss.grad_fn: {loss.grad_fn}")  # Shows the operation that created it
    print()
    
    print("="*70)
    print("PART 3: Backward Pass - Computing Gradients")
    print("="*70)
    print("Calling loss.backward()...")
    print("This computes: d(loss)/dx = d(sum(x^2))/dx = 2*x")
    print()
    
    # 역전파: 경사를 계산한다
    # 스칼라 텐서에 대해 .backward()는 .backward(torch.tensor(1.0))과 동등하다
    loss.backward()
    
    print(f"x.grad (after backward): {x.grad}")
    print(f"Expected gradient (2*x): {2*x.detach()}")
    print(f"Match? {torch.allclose(x.grad, 2*x.detach())}")
    print()
    
    print("="*70)
    print("PART 4: Understanding Leaf vs Non-Leaf Tensors")
    print("="*70)
    
    # 시연을 위한 초기화
    x2 = torch.tensor([1., 2., 3.], requires_grad=True)
    y = 2 * x2  # y is non-leaf (created by operation)
    z = (y ** 2).sum()  # z is also non-leaf
    
    print(f"x2 (leaf): is_leaf={x2.is_leaf}, grad_fn={x2.grad_fn}")
    print(f"y (non-leaf): is_leaf={y.is_leaf}, grad_fn={y.grad_fn}")
    print(f"z (non-leaf): is_leaf={z.is_leaf}, grad_fn={z.grad_fn}")
    print()
    
    # 기본적으로 경사는 잎 텐서에 대해서만 유지된다
    z.backward()
    print(f"x2.grad: {x2.grad}  ← Stored (leaf tensor)")
    print(f"y.grad: {y.grad}  ← Not stored (non-leaf tensor)")
    print()
    
    print("="*70)
    print("PART 5: Retaining Gradients for Non-Leaf Tensors")
    print("="*70)
    
    # 잎이 아닌 텐서의 경사가 필요하면 .retain_grad()를 쓴다
    x3 = torch.tensor([1., 2., 3.], requires_grad=True)
    y3 = 2 * x3
    y3.retain_grad()  # ← Tell PyTorch to keep y3's gradient
    z3 = (y3 ** 2).sum()
    
    z3.backward()
    
    print("With .retain_grad() called on y3:")
    print(f"x3.grad: {x3.grad}")
    print(f"y3.grad: {y3.grad}  ← Now stored!")
    print(f"Expected y3.grad (d(z3)/d(y3) = 2*y3): {2*y3.detach()}")
    print()
    
    print("="*70)
    print("UNDERSTANDING: What is grad_output / upstream gradient?")
    print("="*70)
    print("""
    PyTorch는 벡터-야코비 곱(VJP)으로 사슬 법칙을 쓴다.
    
    y = f(x)이고 loss = g(y)이면
        d(loss)/dx = (d(loss)/dy) * (dy/dx)
                   = 위쪽 그래디언트 * 그 자리 야코비
    
    홑값 loss.backward()에서는
    - 넌지시 쓰이는 위쪽 기울기가 1.0이다
    - 셈마다 들어온 그래디언트 * 그 자리 야코비를 계산한다
    - 이것이 셈 그래프를 따라 거꾸로 퍼진다
    
    홑값이 아닌 출력에는 위쪽 그래디언트를 드러내 놓고 주어야 한다!
    (이는 앞선 학습에서 본다)
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ 잎 텐서: 셈이 아니라 쓰는 이가 만든다
    ✓ requires_grad=True: 그래디언트 좇기를 켠다
    ✓ .backward(): 홑값 로스의 그래디언트를 계산한다
    ✓ .grad: 셈한 그래디언트를 담는다(기본으로는 잎에만)
    ✓ .retain_grad(): 잎이 아닌 텐서의 그래디언트를 남기려면 이를 쓴다
    ✓ .grad_fn: 어떤 셈이 텐서를 만들었는지 보인다(잎이면 None)
    """)


if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
PART 1: Creating a Leaf Tensor with Gradient Tracking
======================================================================
x: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
x.shape: torch.Size([3])
x.requires_grad: True
x.is_leaf: True
x.grad_fn: None
x.grad (before backward): None

======================================================================
PART 2: Forward Pass - Building the Computational Graph
======================================================================
loss: 7.207892417907715
loss.shape: torch.Size([])
loss.is_leaf: False
loss.grad_fn: <SumBackward0 object at 0x117cd7130>

======================================================================
...
```

## 2. 논의

### 로스에서 입력으로 거슬러 올라가기

예제의 로스는 $L=\sum_i x_i^2$이므로 각 입력에 대한 경사는 $\partial L/\partial x_i=2x_i$이다. 입력은 벡터지만 로스는 원소가 하나인 텐서이므로 `loss.backward()`에 별도의 경사 인수가 필요하지 않다. autograd는 로스 쪽의 경사 1에서 출발해 연쇄 법칙을 적용한다.

### 계산에 참여하는 것과 경사를 보관하는 것

`requires_grad=True`인 잎 텐서에는 기본적으로 경사가 `.grad`에 누적된다. `y = 2 * x`처럼 연산으로 만든 텐서는 역전파에 참여하더라도 기본적으로 `.grad`를 보관하지 않는다. 중간 경사를 확인하려면 역전파 전에 `y.retain_grad()`를 호출한다. `grad_fn`은 텐서를 만든 미분 가능한 연산을 가리키며, 경사 값 자체를 담는 속성이 아니다.

`retain_grad()`는 중간 텐서의 경사를 저장하는 기능이다. 같은 순전파의 중간 결과를 다음 역전파에도 쓸 수 있게 보존하는 `retain_graph=True`와 구분해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`requires_grad=True`인 중간 텐서의 `.grad`가 역전파 후에도 `None`일 수 있는 이유는 무엇인가? 중간 경사를 확인하는 방법도 설명하라.

</div>

??? success "풀이"
    비잎 텐서는 역전파에 참여해도 기본적으로 `.grad`를 보관하지 않는다. 역전파 전에 해당 텐서에 `retain_grad()`를 호출하면 중간 경사를 저장할 수 있다. 이는 그래프 재사용을 위한 `retain_graph=True`와 다르다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

역전파의 출발점은 로스며, 결과는 기본적으로 경사를 추적하는 잎 텐서의 `.grad`에 쌓인다. 중간 텐서의 경사를 관찰할 때만 `retain_grad()`를 추가한다.
