# 매개변수 동결과 해제

이 스크립트는 매개변수를 동결하고 다시 푸는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""매개변수를 얼리고 녹이기."""
# freeze_unfreeze_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

def freeze_layer0(model: nn.Sequential):
    """
    Sequential 안의 첫 선형 층(번호 0)을 얼린다.
    '0.'으로 비롯하는 매개변수 이름을 고른다 → '0.weight', '0.bias'.
    """
    for name, p in model.named_parameters():
        if name.startswith("0."):
            p.requires_grad_(False)

def unfreeze_all(model: nn.Module):
    """모든 매개변수를 다시 익힐 수 있게 한다."""
    for p in model.parameters():
        p.requires_grad_(True)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    # 아주 작은 모델: Linear → ReLU → Linear
    model = nn.Sequential(
        nn.Linear(5, 3),  # index 0 (we'll freeze this one)
        nn.ReLU(),        # index 1
        nn.Linear(3, 1)   # index 2
    )

    grad_status(model, title="Before freezing layer 0")

    # 첫 번째 Linear(인덱스 0)를 동결한다
    freeze_layer0(model)
    grad_status(model, title="After freezing layer 0")

    # 더미 데이터로 순전파 + 역전파
    x = torch.randn(4, 5)  # data tensors default to requires_grad=False
    y = torch.randn(4, 1)
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

    # 역전파 후: 동결된 매개변수는 grad=None을 유지하고 나머지는 경사를 받는다
    grad_status(model, title="After backward")

    # 모두 다시 동결 해제(선택 사항)
    unfreeze_all(model)
    grad_status(model, title="After unfreeze")

if __name__ == "__main__":
    main()
```

**출력:**

```
Before freezing layer 0
0.weight             requires_grad=True  grad_is_None=True
0.bias               requires_grad=True  grad_is_None=True
2.weight             requires_grad=True  grad_is_None=True
2.bias               requires_grad=True  grad_is_None=True

After freezing layer 0
0.weight             requires_grad=False  grad_is_None=True
0.bias               requires_grad=False  grad_is_None=True
2.weight             requires_grad=True  grad_is_None=True
2.bias               requires_grad=True  grad_is_None=True

After backward
0.weight             requires_grad=False  grad_is_None=True
0.bias               requires_grad=False  grad_is_None=True
2.weight             requires_grad=True  grad_is_None=False
2.bias               requires_grad=True  grad_is_None=False

After unfreeze
0.weight             requires_grad=True  grad_is_None=True
0.bias               requires_grad=True  grad_is_None=True
2.weight             requires_grad=True  grad_is_None=False
2.bias               requires_grad=True  grad_is_None=False
```

## 2. 논의

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

## 정리하며

**다룬 것** — 매개변수 동결과 해제

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
