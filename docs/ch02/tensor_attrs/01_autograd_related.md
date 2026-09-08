# Autograd 관련 속성

이 스크립트는 autograd와 관련된 텐서 속성을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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
    main()```

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

벡터-야코비 곱(VJP)은 후진 모드 자동 미분의 핵심 기본 연산이다. 야코비안이 $J$인 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 VJP는 주어진 벡터 $v$에 대한 $v^\top J$를 계산한다. 출력이 스칼라가 아닐 때 PyTorch는 어떤 출력들의 선형 결합을 미분할지 지정하도록 `.backward()`에 경사 인수 $v$를 명시할 것을 요구한다.

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

## 정리하며

**다룬 것** — Autograd 관련 속성

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
