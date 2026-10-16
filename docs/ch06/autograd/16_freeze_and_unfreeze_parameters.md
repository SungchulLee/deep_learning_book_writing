# 파라미터 동결과 해제

모델 일부의 `requires_grad`를 끄고 다시 켜는 과정을 살펴본다. 동결 여부, 경사 유무, 옵티마이저 업데이트 여부를 구분한다.

## 1. 코드

```python
"""파라메터를 얼리고 녹이기."""
# freeze_unfreeze_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

def freeze_layer0(model: nn.Sequential):
    """
    Sequential 안의 첫 선형 레이어(번호 0)을 얼린다.
    '0.'으로 비롯하는 파라메터 이름을 고른다 → '0.weight', '0.bias'.
    """
    for name, p in model.named_parameters():
        if name.startswith("0."):
            p.requires_grad_(False)

def unfreeze_all(model: nn.Module):
    """모든 파라메터를 다시 익힐 수 있게 한다."""
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

    # 역전파 후: 동결된 파라메터는 grad=None을 유지하고 나머지는 경사를 받는다
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

### 파라미터 동결의 의미

예제는 `Sequential`의 첫 계층에 해당하는 `0.`으로 시작하는 파라미터를 동결한다. 동결된 파라미터의 새로운 경사는 계산하지 않지만 순전파 값은 계속 사용한다. 입력이나 앞 계층이 경사를 요구한다면 동결된 계층을 통해서도 그 입력으로 경사가 전달될 수 있다.

### 기존 경사와 옵티마이저 확인

학습 도중 동결할 때는 남아 있는 `.grad`를 `None`으로 지우거나 옵티마이저 대상에서 제외하는 방법을 함께 고려한다. `requires_grad=False`만으로 이미 저장된 경사가 사라지지는 않는다. 동결된 파라미터에 경사가 남으면 옵티마이저가 업데이트할 수 있다.

동결을 풀면 다음 순전파부터 다시 경사를 추적한다. 처음 옵티마이저를 만들 때 해당 파라미터를 제외했다면 다시 포함해야 한다. 옵티마이저의 모멘텀 같은 상태를 유지할지 초기화할지도 학습 목적에 맞게 결정한다.

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
학습 도중 파라미터를 동결할 때 추적 설정 외에 무엇을 확인해야 하는가?

</div>

??? success "연습문제 2 풀이"
    기존 `.grad`가 남아 있으면 옵티마이저가 이를 사용해 업데이트할 수 있으므로 지우거나 대상에서 제외한다. 동결 해제 시에는 옵티마이저에 포함되어 있는지도 확인한다. BatchNorm·Dropout 같은 계층의 학습·평가 모드는 별도로 정한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

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

동결은 새로운 미분을 제어한다. 안전한 선택적 학습을 위해서는 이전 경사, 옵티마이저 대상, 계층의 학습·평가 모드까지 확인한다.
