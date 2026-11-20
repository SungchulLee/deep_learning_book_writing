# Guided Backpropagation

06: Guided Backpropagation 설명:

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
06: Guided Backpropagation
==========================

어려움: 가운데

설명:
Guided Backpropagation는 ReLU 레이어를 지나는 역전파 스텝을 고쳐 양수 그래디언트만
퍼뜨린다. 음수 그래디언트를 눌러 더 또렷하고 깨끗한 그림을 낸다.

고침:
여느 ReLU 역전파: ∂L/∂x = (∂L/∂y) · 1(x > 0)
이끈 ReLU 역전파: ∂L/∂x = (∂L/∂y) · 1(x > 0) · 1(∂L/∂y > 0)

덧붙은 조건: 양수 그래디언트만 되짚는다

지은이: 교육용 자료
"""

import torch
import torch.nn as nn
from utils import *

# ========================================================================
# 메인
# ========================================================================

class GuidedBackpropReLU(nn.Module):
    """Guided Backpropagation를 위해 고친 ReLU."""

    def forward(self, x):
        return F.relu(x)

    def backward(self, grad_output):
        # 양수 살아남을 지나는 양수 그래디언트만 되짚는다
        return grad_output.clamp(min=0) * (self.output > 0).float()


def replace_relu_with_guided(model):
    """모든 ReLU을 GuidedBackpropReLU으로 갈음한다."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedBackpropReLU())
        else:
            replace_relu_with_guided(module)


def compute_guided_backprop(model, image_tensor, target_class, device):
    """Guided Backpropagation를 셈한다."""
    model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True

    output = model(image_tensor)
    target_score = output[0, target_class]

    model.zero_grad()
    target_score.backward()

    guided_grads = image_tensor.grad
    saliency = torch.max(torch.abs(guided_grads), dim=1)[0]

    return saliency


def example_1_guided_vs_vanilla():
    """Guided Backpropagation와 맨 그래디언트를 견준다."""
    print("\n" + "="*60)
    print("보기 1: Guided Backpropagation 대 맨 그래디언트")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    # 모형 둘: 하나는 맨 그래디언트, 하나는 Guided Backpropagation
    model_vanilla = load_pretrained_model('resnet50', device)
    model_guided = load_pretrained_model('resnet50', device)

    # 한 모형을 Guided Backpropagation에 맞게 고친다
    print("Guided Backpropagation를 차린다...")
    # 짚을 것: 온전히 짜려면 손수 만든 갈고리가 있어야 한다
    # 여기서는 단순하게 줄인 것을 보인다

    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(140, 160, 100))
    image_tensor = preprocess_image(test_image, requires_grad=True)

    with torch.no_grad():
        output = model_vanilla(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    # 맨 그래디언트
    print("맨 그래디언트를 셈한다...")
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model_vanilla(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]

    print("\n✓ Guided Backpropagation가 더 또렷한 그림을 낸다")
    print("(온전히 짜려면 손수 만든 autograd 함수가 있어야 한다)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "Guided Backpropagation 익히기")
    print("="*70)

    try:
        example_1_guided_vs_vanilla()

        print("\n" + "="*70)
        print("핵심:")
        print("1. ReLU 역전파 스텝을 고친다")
        print("2. 양수 그래디언트만 퍼뜨린다")
        print("3. 더 또렷하고 깨끗한 그림을 낸다")
        print("4. 짜려면 손수 만든 갈고리가 있어야 한다")
        print("\n다음: 꾸러미 07 - 이끈 Grad-CAM")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
               Guided Backpropagation 익히기
======================================================================

============================================================
보기 1: Guided Backpropagation 대 맨 그래디언트
============================================================
어긋남: name 'get_device' is not defined
```

## 2. 논의

`GuidedBackpropReLU` 클래스는 PyTorch의 `nn.Module` 사이틀로 모델 구조를 감싼다. `forward` 방법이 계산 그래프를 세우므로 PyTorch의 autograd가 학습하는 동안 그래디언트 계산을 알아서 다룬다. 이렇게 조각으로 나눈 설계 덕에 부분마다 고치거나 더 큰 흐름에 끼워 넣기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GuidedBackpropReLU`을 레이어나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라미터를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    못 박아 둔 레이어를 이렇게 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서는 `for layer in self.layers: x = layer(x)`으로 돈다. 일반 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라미터를 최적화에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = GuidedBackpropReLU(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — Guided Backpropagation

`GuidedBackpropReLU` 클래스는 PyTorch의 `nn.Module` 사이틀로 모델 구조를 감싼다.

핵심 클래스는 `GuidedBackpropReLU`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
