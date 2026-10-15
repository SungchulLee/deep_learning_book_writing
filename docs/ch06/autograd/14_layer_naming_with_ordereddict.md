# OrderedDict로 계층 이름 지정하기

`nn.Sequential` 안의 계층에 의미 있는 이름을 붙이고 이름을 기준으로 일부 파라미터를 동결한다. 모듈 등록 이름은 점검과 저장에도 사용된다.

## 1. 코드

```python
"""OrderedDict으로 층에 이름 붙이기."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

def freeze_encoder(model: nn.Sequential):
    """
    Sequential에서 이름을 드러내 놓고 붙인 'encoder' 선형 층을 얼린다.
    'encoder.weight'과 'encoder.bias'이 걸린다.
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
    main()
```

**출력:**

```
Before freezing 'encoder'
encoder.weight       requires_grad=True  grad_is_None=True
encoder.bias         requires_grad=True  grad_is_None=True
head.weight          requires_grad=True  grad_is_None=True
head.bias            requires_grad=True  grad_is_None=True

After freezing 'encoder'
encoder.weight       requires_grad=False  grad_is_None=True
encoder.bias         requires_grad=False  grad_is_None=True
head.weight          requires_grad=True  grad_is_None=True
head.bias            requires_grad=True  grad_is_None=True

After backward
encoder.weight       requires_grad=False  grad_is_None=True
encoder.bias         requires_grad=False  grad_is_None=True
head.weight          requires_grad=True  grad_is_None=False
head.bias            requires_grad=True  grad_is_None=False

After unfreeze
encoder.weight       requires_grad=True  grad_is_None=True
encoder.bias         requires_grad=True  grad_is_None=True
head.weight          requires_grad=True  grad_is_None=False
head.bias            requires_grad=True  grad_is_None=False
```

## 2. 논의

### 이름으로 구조를 드러내기

`OrderedDict`에 `encoder`, `act`, `head`를 차례로 넣으면 이 순서로 실행되는 `Sequential`을 만든다. 등록된 파라미터 이름은 `encoder.weight`, `head.bias`처럼 모듈 경로를 포함한다. 이름은 모델의 역할을 드러내고 특정 계층을 선택하는 기준이 된다.

### 이름을 기준으로 동결하기

예제는 `encoder.`로 시작하는 파라미터의 `requires_grad`를 끈다. 입력도 경사를 요구하지 않으므로 이 예제의 인코더 계산은 기록되지 않지만, 학습 가능한 헤드의 계산은 기록된다. 입력에 대한 미분이 필요하면 동결된 계층을 지나는 경로도 추적될 수 있다.

동결 해제는 이후 연산에서 경사를 다시 계산할 수 있게 한다. 이미 수행한 역전파의 결과를 소급해서 만들지는 않는다. 실제 학습 도중 동결한다면 기존 경사를 지우고 옵티마이저의 대상과 상태도 확인한다.

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
`encoder`를 동결한 상태에서도 입력에 대한 미분을 구할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    입력에 `requires_grad=True`를 주고 계산 기록을 유지하면 동결된 인코더를 거쳐 입력으로 경사가 전달될 수 있다. 파라미터 동결은 그 파라미터의 새로운 경사 계산을 제외하는 것이며 인코더 전체를 `no_grad()`로 감싸는 것과 다르다.

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

계층 이름은 모델 점검과 선택적 학습의 기준이 된다. 동결 상태와 `.grad`의 현재 값은 별개의 정보이므로 함께 확인한다.
