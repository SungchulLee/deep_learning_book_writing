# 05: Grad-CAM

Grad-CAM(그래디언트 가중 클래스 활성화 맵)을 구현하고 바닐라 그래디언트와 비교하려는 쪽이다. Grad-CAM은 마지막 컨볼루션 레이어의 피처 맵을 그 그래디언트의 평균으로 가중합해, 어느 영역이 그 클래스를 예측하게 했는지 보이는 히트맵을 만든다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
05: Grad-CAM - 그래디언트 가중 클래스 활성화 맵
========================================================

난이도: 중간

설명:
Grad-CAM은 마지막 컨볼루션 레이어로 흘러드는 그래디언트를 써서 중요한 영역을
보여 주는 거친 위치 맵을 낸다. 픽셀 단위 방법과 달리 Grad-CAM은
어느 영역이 중요한지를 보여 준다.

수학적 배경:
    α_k = (1/Z) Σᵢⱼ (∂y_c/∂A_k^(i,j))    [중요도 가중치]
    L_Grad-CAM = ReLU(Σ_k α_k A_k)         [가중합]

여기서:
- A_k: 마지막 컨볼루션 레이어의 k번째 피처 맵
- α_k: 전역 평균 풀링한 그래디언트
- ReLU: 기여가 양수인 것만

장점:
- 클래스 판별적이다(클래스가 다르면 다른 영역이 보인다)
- 거친 위치 파악("무엇"이 아니라 "어디"를 보인다)
- 어떤 CNN 구조에도 적용된다
- 눈으로 해석할 수 있다

지은이: 교육용 자료
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
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


def denormalize(image_tensor):
    """정규화한 텐서를 [0,1]로 되돌려 시각화할 수 있게 만든다."""
    t = image_tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _as_map(saliency):
    """샐리언시를 (H, W) 넘파이 배열로 맞추고 [0,1]로 정규화한다."""
    s = saliency.detach().cpu() if torch.is_tensor(saliency) else torch.as_tensor(saliency)
    while s.dim() > 2:
        s = s.squeeze(0) if s.shape[0] == 1 else s.max(0).values
    s = s.numpy().astype(np.float32)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def visualize_saliency(image_tensor, saliency, title="Saliency", colormap="jet",
                       alpha=0.4, save_path=None):
    """원본 이미지, 샐리언시 맵, 그 둘을 겹친 것을 나란히 보인다."""
    img = denormalize(image_tensor)
    sal = _as_map(saliency)
    if sal.shape != img.shape[:2]:
        sal = np.array(Image.fromarray((sal * 255).astype(np.uint8))
                       .resize((img.shape[1], img.shape[0]))) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);              axes[0].set_title("Input");    axes[0].axis("off")
    axes[1].imshow(sal, cmap=colormap); axes[1].set_title("Saliency"); axes[1].axis("off")
    axes[2].imshow(img)
    axes[2].imshow(sal, cmap=colormap, alpha=alpha)
    axes[2].set_title("Overlay");     axes[2].axis("off")
    # 제목은 한글이 섞일 수 있다. matplotlib 기본 글꼴에 한글이 없으므로
    # 그림 위가 아니라 표준 출력으로 알린다.
    print(f"  [그림] {title}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_multiple_saliencies(image_tensor, saliencies, colormap="jet",
                                  alpha=0.4, save_path=None):
    """여러 방법의 샐리언시를 한 줄에 늘어놓아 비교한다.

    인수:
        saliencies: {이름: 샐리언시} 딕셔너리. 이름은 표준 출력으로 알린다
    """
    img = denormalize(image_tensor)
    names = list(saliencies)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")

    for ax, name in zip(axes[1:], names):
        sal = _as_map(saliencies[name])
        if sal.shape != img.shape[:2]:
            sal = np.array(Image.fromarray((sal * 255).astype(np.uint8))
                           .resize((img.shape[1], img.shape[0]))) / 255.0
        ax.imshow(img)
        ax.imshow(sal, cmap=colormap, alpha=alpha)
        ax.axis("off")
    print("  [그림] " + " | ".join(n.replace("\n", " ") for n in names))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_output_dir(path="outputs"):
    """출력 이미지를 저장할 디렉터리를 만들고 그 경로를 돌려준다."""
    os.makedirs(path, exist_ok=True)
    return path


# ========================================================================
# 메인
# ========================================================================

class GradCAM:
    """어떤 CNN에도 쓰는 Grad-CAM 구현."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN 모델
            target_layer: 마지막 컨볼루션 레이어(예: model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 훅을 건다
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """순전파 스텝의 활성값을 저장한다."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """역전파 스텝의 그래디언트를 저장한다."""
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_class, device):
        """
        Grad-CAM을 계산한다.

        Returns:
            torch.Tensor: 히트맵 [1, H, W]
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)

        # 순전파 스텝
        output = self.model(image_tensor)
        target_score = output[0, target_class]

        # 역전파 스텝
        self.model.zero_grad()
        target_score.backward()

        # 활성값과 그래디언트를 가져온다
        activations = self.activations  # [1, C, H', W']
        gradients = self.gradients       # [1, C, H', W']

        # 그래디언트를 전역 평균 풀링한다: α_k = (1/Z) Σᵢⱼ ∂y_c/∂A_k^(i,j)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 가중합으로 결합한다: Σ_k α_k A_k
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU를 적용한다: 기여가 양수인 것만
        heatmap = F.relu(weighted_activations)  # [1, 1, H', W']

        # [0, 1]로 정규화한다
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # 입력 크기로 업샘플한다
        heatmap = F.interpolate(
            heatmap,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        return heatmap.squeeze()  # [H, W]


def get_last_conv_layer(model, model_name='resnet50'):
    """구조마다 마지막 컨볼루션 레이어를 골라 준다."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'densenet' in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"지원하지 않는 구조: {model_name}")


def example_1_basic_gradcam():
    """Grad-CAM 기본 사용법."""
    print("\n" + "="*60)
    print("예제 1: 기본 Grad-CAM")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    # 모델을 불러온다
    model_name = 'resnet50'
    model = load_pretrained_model(model_name, device)

    # 마지막 컨볼루션 레이어를 고른다
    target_layer = get_last_conv_layer(model, model_name)
    print(f"대상 레이어: {target_layer.__class__.__name__}")

    # Grad-CAM을 만든다
    gradcam = GradCAM(model, target_layer)

    # 테스트 이미지
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    image_tensor = preprocess_image(test_image, requires_grad=False)

    # 예측을 얻는다
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, target_class].item()

    print(f"\n예측 클래스: {target_class}")
    print(f"확신도: {confidence:.2%}")

    # Grad-CAM을 계산한다
    print("\nGrad-CAM을 계산하는 중...")
    heatmap = gradcam(image_tensor, target_class, device)

    print(f"히트맵 형태: {heatmap.shape}")
    print(f"값 범위: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # 시각화한다
    visualize_saliency(
        image_tensor,
        heatmap,
        title=f"Grad-CAM (클래스 {target_class})",
        colormap='jet',
        alpha=0.4,
        save_path='outputs/05_gradcam_basic.png'
    )

    print("\n✓ Grad-CAM이 거친 위치 파악을 보여 준다!")


def example_2_compare_with_gradients():
    """Grad-CAM과 그래디언트 방법을 비교한다."""
    print("\n" + "="*60)
    print("예제 2: Grad-CAM 대 그래디언트 방법")
    print("="*60)

    device = get_device()
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(180, 120, 80))
    image_tensor_grad = preprocess_image(test_image, requires_grad=True)
    image_tensor_cam = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor_cam.to(device))
        target_class = output.argmax(dim=1).item()

    # 바닐라 그래디언트
    print("바닐라 그래디언트를 계산하는 중...")
    output = model(image_tensor_grad.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_tensor_grad.grad), dim=1)[0]

    # Grad-CAM
    print("Grad-CAM을 계산하는 중...")
    target_layer = get_last_conv_layer(model, 'resnet50')
    gradcam = GradCAM(model, target_layer)
    gradcam_map = gradcam(image_tensor_cam, target_class, device)

    # 비교한다
    saliencies = {
        '바닐라 그래디언트\n(픽셀 단위)': vanilla,
        'Grad-CAM\n(영역 단위)': gradcam_map
    }

    visualize_multiple_saliencies(
        image_tensor_cam, saliencies,
        save_path='outputs/05_gradcam_vs_gradient.png'
    )

    print("\n핵심 차이:")
    print("- 바닐라 그래디언트: 픽셀 단위, 잡음 많음, 세밀함")
    print("- Grad-CAM: 영역 단위, 깨끗함, 거침")
    print("- '어디'에는 Grad-CAM이, '무엇'에는 그래디언트가 낫다")
    print("\n✓ 서로 보완하는 방법이다!")


def main():
    print("\n" + "="*70)
    print(" "*20 + "GRAD-CAM 튜토리얼")
    print("="*70)

    try:
        example_1_basic_gradcam()
        example_2_compare_with_gradients()

        print("\n" + "="*70)
        print("핵심:")
        print("1. Grad-CAM: 거칠지만 클래스 판별적인 위치 파악")
        print("2. 마지막 컨볼루션 레이어의 활성값 + 그래디언트를 쓴다")
        print("3. 히트맵이 깨끗하고 해석하기 쉽다")
        print("4. 공간 구조를 지닌 CNN에만 쓸 수 있다")
        print("\n다음: 꾸러미 06 - Guided Backpropagation")
        print("="*70)
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
                    GRAD-CAM 튜토리얼
======================================================================

============================================================
예제 1: 기본 Grad-CAM
============================================================
대상 레이어: Bottleneck

예측 클래스: 21
확신도: 7.25%

Grad-CAM을 계산하는 중...
히트맵 형태: torch.Size([224, 224])
값 범위: [0.000, 0.995]
  [그림] Grad-CAM (클래스 21)

✓ Grad-CAM이 거친 위치 파악을 보여 준다!

============================================================
예제 2: Grad-CAM 대 그래디언트 방법
============================================================
바닐라 그래디언트를 계산하는 중...
Grad-CAM을 계산하는 중...
  [그림] 바닐라 그래디언트 (픽셀 단위) | Grad-CAM (영역 단위)

핵심 차이:
- 바닐라 그래디언트: 픽셀 단위, 잡음 많음, 세밀함
- Grad-CAM: 영역 단위, 깨끗함, 거침
- '어디'에는 Grad-CAM이, '무엇'에는 그래디언트가 낫다

✓ 서로 보완하는 방법이다!

======================================================================
핵심:
1. Grad-CAM: 거칠지만 클래스 판별적인 위치 파악
2. 마지막 컨볼루션 레이어의 활성값 + 그래디언트를 쓴다
3. 히트맵이 깨끗하고 해석하기 쉽다
4. 공간 구조를 지닌 CNN에만 쓸 수 있다

다음: 꾸러미 06 - Guided Backpropagation
======================================================================
```

## 2. 논의

위 출력에서 보듯 프로그램은 두 예제를 끝까지 돈다. 대상 레이어는 ResNet-50 의 마지막 `Bottleneck` 블록이고, 히트맵은 입력과 같은 224×224 로 업샘플되어 값이 0~1 로 정규화된다(출력의 `[0.000, 0.995]`). 다만 입력이 한 가지 색으로 칠한 합성 이미지라 예측 클래스 21의 확신도가 7.25% 에 그친다. 모델이 무언가를 알아본 이미지가 아니므로 이 히트맵이 「어디를 보았나」를 보여 준다고 읽으면 안 된다. 히트맵 그림은 `outputs/` 에 파일로만 저장되고, 핵심 계산은 `GradCAM.__call__` 의 코드로 읽어야 한다.

`GradCAM` 클래스는 Grad-CAM의 세 단계를 그대로 옮긴다. 순전파 훅으로 피처 맵을, 역전파 훅으로 그 그래디언트를 붙잡고, 그래디언트의 공간 평균을 웨이트로 삼아 피처 맵을 가중합한 뒤 ReLU를 건다. 단계가 나뉘어 있어 부분마다 따로 이해하고 다른 모델이나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    `GradCAM` 에서 셋을 들 수 있다. (1) **훅으로 중간값 붙잡기** -- `register_forward_hook` 과 역전파 훅으로 타깃 레이어의 활성화와 그래디언트를 모델 코드를 고치지 않고 얻는다. 그래서 어떤 CNN에도 쓸 수 있다. (2) **그래디언트의 공간 평균을 웨이트로**(`gradients.mean(dim=(2, 3))`) -- 피처 맵마다 클래스 점수에 대한 전체 중요도를 한 수로 요약한다. (3) **가중합 뒤의 ReLU** -- 클래스 점수를 올리는 영역만 남겨 다른 클래스의 증거가 섞이지 않게 한다. 각 선택은 범용성, 해상도, 해석 용이성 사이의 트레이드오프를 드러낸다.

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
    흔한 실패 유형은 이렇다. (1) **훅이 엉뚱한 값을 붙잡음** -- `register_backward_hook` 은 폐기 예정이고, 여러 연산으로 된 모듈에서는 마지막 연산의 그래디언트만 넘겨줄 수 있다. 히트맵이 모두 0이거나 입력과 무관해 보이면 의심하고, `register_full_backward_hook` 으로 바꾸어 고친다. 같은 레이어에 `GradCAM` 을 여러 번 만들면 훅이 쌓이므로 쓰고 난 훅은 지운다. (2) **타깃 레이어를 잘못 고름** -- `get_last_conv_layer` 는 DenseNet에 `features[-1]` 을 돌려주는데 이는 컨볼루션이 아니라 배치 정규화 레이어다. 해상도가 이상하거나 맵이 의미 없어 보이면 레이어 출력의 형태를 찍어 탐지하고, `features.denseblock4` 처럼 공간 피처 맵을 내는 모듈을 고른다. 입력 크기가 224가 아니면 `size=(224, 224)` 로 못 박은 보간도 틀어지므로 `image_tensor.shape[2:]` 를 쓴다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
05: Grad-CAM 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_05: grad-cam():
        model = 05: Grad-CAM(...)
        # 일반 입력
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 05: Grad-CAM

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 그래디언트 기반 설명의 핵심 아이디어를 보여 준다.

핵심 클래스는 `GradCAM`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
