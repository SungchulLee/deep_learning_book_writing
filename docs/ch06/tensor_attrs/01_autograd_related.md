# Autograd 관련 속성

기울기가 필요한 텐서라도 `.grad`가 항상 채워지는 것은 아니다. `requires_grad`·`is_leaf`·`grad_fn`으로 추적 여부와 그래프의 위치를 구별한다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
자동 미분에 마음을 둔 텐서의 속성과 움직임.

Covers:
- requires_grad / is_leaf / grad_fn
- 홑값과 홑값 아닌 것에 대한 .backward()(gradient 인자를 주는 벡터-야코비 곱)
- 기울기 쌓기와 지우기(p.grad.zero_과 opt.zero_grad 견주기)
- 안전하게 매개변수를 고치는 torch.no_grad()
- 같은 그래프에서 여러 번 뒤로 걷기 위한 retain_graph=True
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    header("Leaf / non-leaf, requires_grad, grad_fn, is_leaf")
    w = torch.randn(3, requires_grad=True)   # LEAF tensor
    y = (w * 2.0).sum()                      # non-leaf scalar with grad_fn
    print("w:", w)
    print("w.requires_grad:", w.requires_grad, "| w.grad_fn:", w.grad_fn, "| w.is_leaf:", w.is_leaf)
    print("y.requires_grad:", y.requires_grad, "| y.grad_fn:", y.grad_fn, "| y.is_leaf:", y.is_leaf)

    # -------------------------------------------------------------------------
    header("Backward on scalar output → fills w.grad (accumulates)")
    print("Before backward, w.grad:", w.grad)
    y.backward()
    print("After  1st backward, w.grad:", w.grad)
    w.grad.zero_()
    ((w ** 2).sum()).backward()
    print("After  2nd backward, w.grad (fresh):", w.grad)

    # -------------------------------------------------------------------------
    header("Non-scalar output requires gradient arg (VJP)")
    x = torch.randn(4, 3, requires_grad=True)
    A = torch.randn(2, 3)         # fixed linear map (no grads)
    out = x @ A.t()                # shape (4, 2) → non-scalar
    v = torch.tensor([[1., 0.], [0.5, -1.], [0., 0.], [2., 3.]])  # same shape as out
    x.grad = None
    out.backward(v)                # computes VJP wrt x
    print("x.grad shape (expect (4,3)):", x.grad.shape)

    # -------------------------------------------------------------------------
    header("Optimizer-style clearing: zero_grad(set_to_none=True)")
    model = nn.Linear(3, 1, bias=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xb = torch.randn(5, 3)
    yb = torch.randn(5, 1)
    opt.zero_grad(set_to_none=True)   # sets .grad to None (not zeros)
    nn.functional.mse_loss(model(xb), yb).backward()
    # 이제 .grad가 할당되었다. 다시 지우지 않으면 또 한 번의 .backward()는 누적된다.
    print("Param grad is None? ->", [p.grad is None for p in model.parameters()])
    opt.zero_grad(set_to_none=True)
    print("After zero_grad(set_to_none=True):", [p.grad is None for p in model.parameters()])

    # -------------------------------------------------------------------------
    header("torch.no_grad() for parameter updates (avoid graph pollution)")
    p = torch.randn(3, requires_grad=True)
    loss = (p ** 2).sum()
    loss.backward()
    with torch.no_grad():          # update excluded from graph
        p -= 0.1 * p.grad
    print("p.requires_grad stays True:", p.requires_grad)

    # -------------------------------------------------------------------------
    header("retain_graph=True for repeated backward on the SAME graph")
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    z = (a ** 2).sum()             # one graph
    z.backward(retain_graph=True)  # keep graph alive
    z.backward()                   # reuse retained graph, now it will be freed
    print("a.grad (accumulated from two backward passes):", a.grad)

if __name__ == "__main__":
    main()
```

**출력:**

```
================================================================================
Leaf / non-leaf, requires_grad, grad_fn, is_leaf
================================================================================
w: tensor([ 1.5410, -0.2934, -2.1788], requires_grad=True)
w.requires_grad: True | w.grad_fn: None | w.is_leaf: True
y.requires_grad: True | y.grad_fn: <SumBackward0 object at 0x116c0bac0> | y.is_leaf: False

================================================================================
Backward on scalar output → fills w.grad (accumulates)
================================================================================
Before backward, w.grad: None
After  1st backward, w.grad: tensor([2., 2., 2.])
After  2nd backward, w.grad (fresh): tensor([ 3.0820, -0.5869, -4.3576])

================================================================================
Non-scalar output requires gradient arg (VJP)
================================================================================
x.grad shape (expect (4,3)): torch.Size([4, 3])

================================================================================
Optimizer-style clearing: zero_grad(set_to_none=True)
================================================================================
Param grad is None? -> [False, False]
After zero_grad(set_to_none=True): [True, True]

================================================================================
torch.no_grad() for parameter updates (avoid graph pollution)
================================================================================
p.requires_grad stays True: True

================================================================================
retain_graph=True for repeated backward on the SAME graph
================================================================================
a.grad (accumulated from two backward passes): tensor([ 4.,  8., 12.])
```

## 2. 논의

`requires_grad`는 이후 연산을 추적할지 결정하는 속성이고, `grad_fn`은 그 텐서를 만든 미분 연산을 가리킨다. 사용자가 직접 만든 학습 매개변수는 보통 잎 텐서이며 `grad_fn`이 없다.

역전파에서 기본적으로 `.grad`가 누적되는 것은 기울기를 요구하는 잎 텐서다. 비잎 텐서의 중간 기울기를 보려면 `retain_grad()`를 사용한다. `retain_grad`와 그래프 재사용을 위한 `retain_graph`는 다른 기능이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "연습문제 1 풀이"
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
제곱합처럼 저장된 중간값이 필요한 손실에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 손실을 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

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

## 정리하며

기울기가 필요한 텐서라도 `.grad`가 항상 채워지는 것은 아니다. `requires_grad`·`is_leaf`·`grad_fn`으로 추적 여부와 그래프의 위치를 구별한다.
