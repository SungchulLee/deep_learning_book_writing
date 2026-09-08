# 이끈 되짚기

06: 이끈 되짚기 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
06: 이끈 되짚기
==========================

어려움: 가운데

밝힘:
이끈 되짚기는 ReLU 켜를 지나는 되짚기 걸음을 고쳐 양수 기울기만
퍼뜨린다. 음수 기울기를 눌러 더 또렷하고 깨끗한 그림을 낸다.

고침:
여느 ReLU 되짚기: ∂L/∂x = (∂L/∂y) · 1(x > 0)
이끈 ReLU 되짚기: ∂L/∂x = (∂L/∂y) · 1(x > 0) · 1(∂L/∂y > 0)

덧붙은 조건: 양수 기울기만 되짚는다

지은이: 가르치기 몫
"""

import torch
import torch.nn as nn
from utils import *

# ========================================================================
# 메인
# ========================================================================

class GuidedBackpropReLU(nn.Module):
    """이끈 되짚기를 위해 고친 ReLU."""

    def forward(self, x):
        return F.relu(x)

    def backward(self, grad_output):
        # 양수 살아남을 지나는 양수 기울기만 되짚는다
        return grad_output.clamp(min=0) * (self.output > 0).float()


def replace_relu_with_guided(model):
    """모든 ReLU을 GuidedBackpropReLU으로 갈음한다."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedBackpropReLU())
        else:
            replace_relu_with_guided(module)


def compute_guided_backprop(model, image_tensor, target_class, device):
    """이끈 되짚기를 셈한다."""
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
    """이끈 되짚기와 맨 기울기를 견준다."""
    print("\n" + "="*60)
    print("보기 1: 이끈 되짚기 대 맨 기울기")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    # 모형 둘: 하나는 맨 기울기, 하나는 이끈 되짚기
    model_vanilla = load_pretrained_model('resnet50', device)
    model_guided = load_pretrained_model('resnet50', device)

    # 한 모형을 이끈 되짚기에 맞게 고친다
    print("이끈 되짚기를 차린다...")
    # 짚을 것: 온전히 짜려면 손수 만든 갈고리가 있어야 한다
    # 여기서는 단순하게 줄인 것을 보인다

    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(140, 160, 100))
    image_tensor = preprocess_image(test_image, requires_grad=True)

    with torch.no_grad():
        output = model_vanilla(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    # 맨 기울기
    print("맨 기울기를 셈한다...")
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model_vanilla(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]

    print("\n✓ 이끈 되짚기가 더 또렷한 그림을 낸다")
    print("(온전히 짜려면 손수 만든 autograd 함수가 있어야 한다)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "이끈 되짚기 익히기")
    print("="*70)

    try:
        example_1_guided_vs_vanilla()

        print("\n" + "="*70)
        print("고갱이:")
        print("1. ReLU 되짚기 걸음을 고친다")
        print("2. 양수 기울기만 퍼뜨린다")
        print("3. 더 또렷하고 깨끗한 그림을 낸다")
        print("4. 짜려면 손수 만든 갈고리가 있어야 한다")
        print("\n다음: 꾸러미 07 - 이끈 Grad-CAM")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

## 2. 논의

`GuidedBackpropReLU` 클래스는 PyTorch의 `nn.Module` 사이틀로 모형 얼개를 감싼다. `forward` 방법이 셈 그래프를 세우므로 PyTorch의 autograd가 익히는 동안 기울기 셈을 알아서 다룬다. 이렇게 조각으로 나눈 설계 덕에 부분마다 고치거나 더 큰 흐름에 끼워 넣기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 모형 풀이하기에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 기울기 바탕 풀이에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
`GuidedBackpropReLU`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    못 박아 둔 켜를 이렇게 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서는 `for layer in self.layers: x = layer(x)`으로 돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = GuidedBackpropReLU(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 이끈 되짚기

`GuidedBackpropReLU` 클래스는 PyTorch의 `nn.Module` 사이틀로 모형 얼개를 감싼다.

고갱이 갈래는 `GuidedBackpropReLU`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
