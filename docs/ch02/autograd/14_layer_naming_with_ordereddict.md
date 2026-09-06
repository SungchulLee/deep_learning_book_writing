# OrderedDict를 이용한 층 이름 짓기

이 스크립트는 `OrderedDict`로 층에 이름을 붙이는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Layer naming with ordereddict."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

def freeze_encoder(model: nn.Sequential):
    """
    Freeze the 'encoder' Linear layer (named explicitly in Sequential).
    Matches 'encoder.weight' / 'encoder.bias'.
    """
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)

def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    # 층에 이름을 붙이려면 OrderedDict를 쓴다
    model = nn.Sequential(OrderedDict([
        ("encoder", nn.Linear(5, 3)),
        ("act",     nn.ReLU()),
        ("head",    nn.Linear(3, 1)),
    ]))
    # OrderedDict 없이 하면 오류:
    # nn.Sequential은 (이름, 모듈) 튜플을 위치 인자로 그대로 받지 않는다.
    # 다음 중 하나를 받는다:
    #   모듈의 리스트/반복 가능 객체만 받거나,
    #   이름 → 모듈로 대응시키는 OrderedDict[str, nn.Module] 하나를 받는다.
    # model = nn.Sequential(
    #     ("encoder", nn.Linear(5, 3)),
    #     ("act",     nn.ReLU()),
    #     ("head",    nn.Linear(3, 1)),
    # )

    grad_status(model, title="Before freezing 'encoder'")

    freeze_encoder(model)
    grad_status(model, title="After freezing 'encoder'")

    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

    grad_status(model, title="After backward")

    unfreeze_all(model)
    grad_status(model, title="After unfreeze")

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

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
