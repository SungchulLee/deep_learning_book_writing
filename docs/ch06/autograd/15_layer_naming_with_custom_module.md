# 사용자 정의 모듈의 계층 이름

`nn.Module`을 상속해 모델의 구조와 순전파를 직접 정의한다. 속성으로 등록한 하위 모듈의 이름을 활용해 인코더만 동결한다.

## 1. 코드

```python
"""맞춤 단원으로 레이어에 이름 붙이기."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5, 3)
        self.activation = nn.ReLU()
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        return self.head(x)

def freeze_encoder(model: TinyNet):
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            p.requires_grad_(False)

def grad_status(model: nn.Module, title="Grad status"):
    print(title)
    for name, p in model.named_parameters():
        print(f"{name:20s} requires_grad={p.requires_grad}  grad_is_None={p.grad is None}")
    print()

def main():
    model = TinyNet()
    grad_status(model, "Before freezing 'encoder'")
    freeze_encoder(model)
    grad_status(model, "After freezing 'encoder'")

    x = torch.randn(4, 5)
    y = torch.randn(4, 1)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    grad_status(model, "After backward")

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
```

## 2. 논의

### 등록과 실행의 두 역할

`super().__init__()` 이후 `self.encoder`, `self.activation`, `self.head`에 모듈을 할당하면 하위 모듈로 등록된다. 파라미터 조회와 장치 이동, `state_dict()` 저장은 이 등록 구조를 따른다. `forward()`는 등록한 모듈을 어떤 순서와 계산으로 사용할지 정의한다.

`model(x)`로 호출하면 `Module`의 호출 절차를 거쳐 `forward()`가 실행된다. 단순한 직렬 구성은 `Sequential`로 표현할 수 있지만, 분기나 여러 입력처럼 실행 로직을 직접 제어하려면 사용자 정의 모듈이 유용하다.

### 인코더만 동결하기

예제는 `encoder.`로 시작하는 파라미터만 경사 추적에서 제외한다. 헤드는 계속 학습 가능하다. 동결은 출력값 계산을 생략하는 기능이 아니며, 동결된 계층에 Dropout이나 BatchNorm이 있다면 학습·평가 모드도 별도로 결정해야 한다.

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
속성에 등록한 모듈과 `forward()`에서 수행하는 계산의 역할을 구분하라.

</div>

??? success "연습문제 2 풀이"
    속성으로 할당한 하위 모듈은 파라미터 조회, 장치 이동과 상태 저장의 구조에 포함된다. `forward()`는 입력이 어떤 계산을 거쳐 출력으로 가는지 정한다. 등록만 했다고 모든 모듈이 순전파에서 자동 실행되는 것은 아니다.

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

사용자 정의 모듈에서는 속성 등록이 구조를, `forward()`가 실행을 정의한다. 의미 있는 이름은 일부 계층만 학습하거나 점검할 때 유용하다.
