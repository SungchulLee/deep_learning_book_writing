# 스칼라에 대한 기본 역전파

이 스크립트는 스칼라에 대한 기본적인 역전파을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
#!/usr/bin/env python3
"""
==============================================================================
첫걸음 익힘 01: 홑값 뒤로 걸음 기초
==============================================================================

배움 목표:
-------------------
1. PyTorch의 자동 미분 얼개에서 "잎" 텐서가 무엇인지 이해한다
2. requires_grad=True으로 기울기 좇기를 켜는 법을 배운다
3. 홑값 잃음에 .backward()을 써서 기울기를 셈하는 법을 본다
4. 잎 텐서와 잎이 아닌 텐서의 다름을 이해한다
5. 잎이 아닌 텐서를 위한 .retain_grad()을 배운다

고갱이 개념:
------------
- 잎 텐서: 쓰는 이가 곧바로 만든 텐서(셈에서 나온 것이 아니다)
- 셈 그래프: PyTorch이 셈을 좇으려고 절로 세운다
- 기울기 쌓임: 드러내 놓고 0으로 만들지 않으면 기울기가 더해진다
- 벡터-야코비 곱(VJP): PyTorch이 기울기를 잘 들게 셈하는 방식

==============================================================================
"""

import torch

# ========================================================================
# 메인
# ========================================================================


def main():
    """
    홑값 잃음 함수에 대한 기본 기울기 셈을 보인다.
    
    다음을 셈한다: loss = sum(x^2)
    바라는 기울기: d(loss)/dx = 2*x
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
    PyTorch은 벡터-야코비 곱(VJP)으로 사슬 법칙을 쓴다.
    
    y = f(x)이고 loss = g(y)이면
        d(loss)/dx = (d(loss)/dy) * (dy/dx)
                   = 위쪽 기울기 * 그 자리 야코비
    
    홑값 loss.backward()에서는
    - 넌지시 쓰이는 위쪽 기울기가 1.0이다
    - 셈마다 들어온 기울기 * 그 자리 야코비를 셈한다
    - 이것이 셈 그래프를 따라 거꾸로 퍼진다
    
    홑값이 아닌 내놓음에는 위쪽 기울기를 드러내 놓고 주어야 한다!
    (이는 앞선 익힘에서 본다)
    """)
    
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    ✓ 잎 텐서: 셈이 아니라 쓰는 이가 만든다
    ✓ requires_grad=True: 기울기 좇기를 켠다
    ✓ .backward(): 홑값 잃음의 기울기를 셈한다
    ✓ .grad: 셈한 기울기를 담는다(기본으로는 잎에만)
    ✓ .retain_grad(): 잎이 아닌 텐서의 기울기를 남기려면 이를 쓴다
    ✓ .grad_fn: 어떤 셈이 텐서를 만들었는지 보인다(잎이면 None)
    """)


if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

벡터-야코비 곱(VJP)은 후진 모드 자동 미분의 핵심 기본 연산이다. 야코비안이 $J$인 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 VJP는 주어진 벡터 $v$에 대한 $v^\top J$를 계산한다. 출력이 스칼라가 아닐 때 PyTorch는 어떤 출력들의 선형 결합을 미분할지 지정하도록 `.backward()`에 경사 인수 $v$를 명시할 것을 요구한다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
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
