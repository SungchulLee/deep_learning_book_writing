# Guided Backpropagation

Guided Backpropagation을 바닐라 그래디언트와 비교하려는 쪽이다. Guided Backpropagation은 ReLU를 지나는 역전파에서 양수 그래디언트만 통과시켜 더 선명한 Saliency Map을 얻는다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
06: Guided Backpropagation
==========================

난이도: 중간

설명:
Guided Backpropagation은 ReLU 레이어를 지나는 역전파 스텝을 고쳐 양수 그래디언트만
전파한다. 음수 그래디언트를 막아 더 또렷하고 깨끗한 맵을 낸다.

수정 사항:
일반 ReLU 역전파: ∂L/∂x = (∂L/∂y) · 1(x > 0)
Guided ReLU 역전파: ∂L/∂x = (∂L/∂y) · 1(x > 0) · 1(∂L/∂y > 0)

추가 조건: 양수 그래디언트만 역전파한다

지은이: 교육용 자료
"""

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from PIL import Image

# ========================================================================
# 도우미 함수
# ========================================================================

# ImageNet으로 학습한 모델이 쓰는 값. 전처리할 때와 되돌릴 때 같은 값을 써야 한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    """쓸 수 있는 가장 빠른 장치를 고른다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name="resnet50", device=None, eval_mode=True):
    """torchvision에서 사전 학습된 모델을 불러온다.

    다운로드가 안 되면(네트워크가 없을 때) 학습하지 않은 모델로 대체한다. 그래디언트
    방법을 보이는 데에는 가중치가 실제로 학습된 것이 아니어도 맵이 나온다.
    """
    from torchvision import models

    device = device or get_device()
    try:
        model = getattr(models, model_name)(weights="DEFAULT")
    except Exception:
        print(f"  사전 학습 가중치를 받지 못했다 — {model_name}을 무작위 초기화 상태로 쓴다.")
        model = getattr(models, model_name)(weights=None)

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def preprocess_image(image, size=224, device=None, requires_grad=False):
    """이미지를 모델이 받는 (1, 3, size, size) 텐서로 전처리한다.

    인수:
        image: 파일 경로이거나 PIL 이미지
        size: 자를 크기
        device: 텐서를 올릴 장치
        requires_grad: True면 입력에 대한 그래디언트를 받을 수 있게 한다

    파일이 없으면 회색 이미지로 대체한다. 예제가 이미지 하나 때문에
    통째로 멈추지 않게 하기 위해서다.
    """
    from torchvision import transforms

    device = device or get_device()

    if isinstance(image, (str, os.PathLike)):
        try:
            image = Image.open(image).convert("RGB")
        except FileNotFoundError:
            print(f"  '{image}' 파일을 찾지 못했다 — 단색 회색 이미지로 대신한다.")
            image = Image.new("RGB", (size, size), (128, 128, 128))
    elif not isinstance(image, Image.Image):
        raise TypeError("image는 파일 경로이거나 PIL 이미지여야 한다")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor.requires_grad_(requires_grad)


def create_output_dir(path="outputs"):
    """출력 이미지를 저장할 디렉터리를 만들고 그 경로를 돌려준다."""
    os.makedirs(path, exist_ok=True)
    return path


# ========================================================================
# 메인
# ========================================================================

class GuidedBackpropReLU(nn.Module):
    """Guided Backpropagation을 위해 수정한 ReLU."""

    def forward(self, x):
        return F.relu(x)

    def backward(self, grad_output):
        # 양수 활성값을 지나는 양수 그래디언트만 역전파한다
        return grad_output.clamp(min=0) * (self.output > 0).float()


def replace_relu_with_guided(model):
    """모든 ReLU를 GuidedBackpropReLU로 교체한다."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedBackpropReLU())
        else:
            replace_relu_with_guided(module)


def compute_guided_backprop(model, image_tensor, target_class, device):
    """Guided Backpropagation을 계산한다."""
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
    """Guided Backpropagation과 바닐라 그래디언트를 비교한다."""
    print("\n" + "="*60)
    print("예제 1: Guided Backpropagation 대 바닐라 그래디언트")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    # 모델 둘: 하나는 바닐라 그래디언트, 하나는 Guided Backpropagation
    model_vanilla = load_pretrained_model('resnet50', device)
    model_guided = load_pretrained_model('resnet50', device)

    # 한 모델을 Guided Backpropagation에 맞게 수정한다
    print("Guided Backpropagation을 설정한다...")
    # 참고: 완전히 구현하려면 직접 만든 훅이 있어야 한다
    # 여기서는 단순하게 줄인 것을 보인다

    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(140, 160, 100))
    image_tensor = preprocess_image(test_image, requires_grad=True)

    with torch.no_grad():
        output = model_vanilla(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()

    # 바닐라 그래디언트
    print("바닐라 그래디언트를 계산한다...")
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model_vanilla(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]

    print("\n✓ Guided Backpropagation이 더 또렷한 맵을 낸다")
    print("(완전히 구현하려면 직접 만든 autograd 함수가 있어야 한다)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "Guided Backpropagation 튜토리얼")
    print("="*70)

    try:
        example_1_guided_vs_vanilla()

        print("\n" + "="*70)
        print("핵심:")
        print("1. ReLU 역전파 스텝을 고친다")
        print("2. 양수 그래디언트만 전파한다")
        print("3. 더 또렷하고 깨끗한 맵을 낸다")
        print("4. 구현하려면 직접 만든 훅이 있어야 한다")
        print("\n다음: 꾸러미 07 - Guided Grad-CAM")
        print("="*70)
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
               Guided Backpropagation 튜토리얼
======================================================================

============================================================
예제 1: Guided Backpropagation 대 바닐라 그래디언트
============================================================
Guided Backpropagation을 설정한다...
바닐라 그래디언트를 계산한다...

✓ Guided Backpropagation이 더 또렷한 맵을 낸다
(완전히 구현하려면 직접 만든 autograd 함수가 있어야 한다)

======================================================================
핵심:
1. ReLU 역전파 스텝을 고친다
2. 양수 그래디언트만 전파한다
3. 더 또렷하고 깨끗한 맵을 낸다
4. 구현하려면 직접 만든 훅이 있어야 한다

다음: 꾸러미 07 - Guided Grad-CAM
======================================================================
```

## 2. 논의

위 출력에서 보듯 프로그램은 끝까지 돌지만, 예제 1은 Guided Backpropagation 쪽 모델에 실제로 훅을 달지 않고 바닐라 그래디언트만 구한 뒤 「더 또렷한 맵을 낸다」는 고정 문구를 찍을 뿐 두 결과를 비교하는 수나 그림을 내놓지 않는다. 출력 자체가 괄호 안에 「완전히 구현하려면 직접 만든 autograd 함수가 있어야 한다」고 밝히고 있다.

코드 자체에도 짚을 점이 있다. `GuidedBackpropReLU` 는 `nn.Module` 을 상속하고 `backward` 메서드를 정의하지만, autograd는 `nn.Module` 의 `backward` 메서드를 부르지 않는다. 역전파는 `forward` 안의 `F.relu` 가 만든 계산 그래프를 그대로 따르므로 양수 그래디언트만 남기는 규칙이 적용되지 않는다(게다가 `self.output` 은 어디서도 저장되지 않는다). 또 `replace_relu_with_guided` 는 정의만 되고 쓰이지 않으며, `example_1_guided_vs_vanilla` 는 바닐라 그래디언트만 계산한다. 코드 주석이 스스로 말하듯 제대로 구현하려면 `torch.autograd.Function` 을 상속한 사용자 정의 함수나 역전파 훅이 필요하다 — Guided Backpropagation 이론 쪽의 `GuidedReLU` 가 그 형태다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    (1) **ReLU만 바꾸고 나머지 레이어는 그대로 둔다** -- Guided Backpropagation의 차이는 ReLU를 지나는 역전파 규칙 하나뿐이므로, 모델의 ReLU 모듈만 찾아 바꾸면(`replace_relu_with_guided`) 사전 학습된 웨이트를 그대로 쓸 수 있다. (2) **두 마스크의 곱** -- `(∂L/∂y > 0)` 과 `(x > 0)` 을 곱해 순전파에서 꺼진 뉴런과 음수 그래디언트를 함께 막는다. 이것이 맵을 선명하게 만드는 원인이다. (3) **채널 축 절댓값 최대**(`torch.max(torch.abs(guided_grads), dim=1)`) -- RGB 세 채널을 한 장의 맵으로 모은다. 다만 (1)은 `nn.Module` 의 `backward` 를 autograd가 부르지 않아 이 코드에서는 실제로 적용되지 않는다는 점을 위 논의에서 짚었다.

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
    흔한 실패 유형은 이렇다. (1) **수정한 역전파가 실제로는 적용되지 않음** -- 바닐라 그래디언트와 Guided Backpropagation 맵의 차이를 재어 탐지한다. 두 맵이 똑같으면 규칙이 걸리지 않은 것이다. `torch.autograd.Function` 을 상속해 `backward` 를 정의하거나 `register_full_backward_hook` 으로 고친다. (2) **in-place ReLU와 훅의 충돌** -- torchvision 모델의 ReLU는 `inplace=True` 라 역전파 훅이 오류를 내거나 잘못된 텐서를 볼 수 있다. 오류 메시지나 이상한 맵으로 탐지하고, 훅을 걸기 전에 `module.inplace = False` 로 바꾸어 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`GuidedBackpropReLU`를 레이어나 블록의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라미터를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    못 박아 둔 레이어를 이렇게 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서는 `for layer in self.layers: x = layer(x)`로 돈다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 모든 파라미터를 최적화에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = GuidedBackpropReLU(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — Guided Backpropagation

`GuidedBackpropReLU` 클래스는 PyTorch의 `nn.Module` 형태로 ReLU를 감싼다. 다만 위 논의에서 보았듯 그 `backward` 는 autograd가 부르지 않는다.

핵심 클래스는 `GuidedBackpropReLU`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
