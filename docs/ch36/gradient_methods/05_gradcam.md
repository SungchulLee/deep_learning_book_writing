# 05: Grad-CAM

05: Grad-CAM - 기울기 짐 실은 갈래 살아남 그림. 밝힘:

신경 그물이 무엇을 배우는지 아는 일은 믿음을 쌓고 모형의 벌레를 잡는 데 종요롭다. 이 꾸러미는 모형이 들임을 어떻게 다루고 어떻게 판단하는지 드러내는 기울기 바탕 풀이 재주를 보이며, 그물의 움직임을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
05: Grad-CAM - 기울기 짐 실은 갈래 살아남 그림
========================================================

어려움: 가운데

밝힘:
Grad-CAM은 마지막 겹치는 켜로 흘러드는 기울기를 써서 중요한 자리를
짚어 주는 성긴 자리 그림을 낸다. 그림점 낱의 방법과 달리 Grad-CAM은
어느 자리가 중요한지를 보여 준다.

수학 밑바탕:
    α_k = (1/Z) Σᵢⱼ (∂y_c/∂A_k^(i,j))    [중요함 짐]
    L_Grad-CAM = ReLU(Σ_k α_k A_k)         [짐 실은 합]

여기서:
- A_k: 마지막 겹치는 켜의 k번째 결 그림
- α_k: 온 세상에 걸쳐 고르게 모은 기울기
- ReLU: 이바지가 양수인 것만

나은 점:
- 갈래를 가려낸다(갈래가 다르면 다른 자리가 보인다)
- 성긴 자리 짚기("무엇"이 아니라 "어디"를 보인다)
- 어떤 CNN 얼개에도 듣는다
- 눈으로 풀이할 수 있다

지은이: 가르치기 몫
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# ========================================================================
# 메인
# ========================================================================

class GradCAM:
    """어떤 CNN에도 쓰는 Grad-CAM 짜보기."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN 모형
            target_layer: 마지막 겹치는 켜(보기로 model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 갈고리를 건다
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """앞으로 걸음의 살아남을 갈무리한다."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """되짚기 걸음의 기울기를 갈무리한다."""
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_class, device):
        """
        Grad-CAM을 셈한다.

        Returns:
            torch.Tensor: 열 그림 [1, H, W]
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)

        # 앞으로 걸음
        output = self.model(image_tensor)
        target_score = output[0, target_class]

        # 되짚기 걸음
        self.model.zero_grad()
        target_score.backward()

        # 살아남과 기울기를 집는다
        activations = self.activations  # [1, C, H', W']
        gradients = self.gradients       # [1, C, H', W']

        # 기울기를 온 세상에 걸쳐 고르게 모은다: α_k = (1/Z) Σᵢⱼ ∂y_c/∂A_k^(i,j)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 짐 실어 아우른다: Σ_k α_k A_k
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU을 건다: 이바지가 양수인 것만
        heatmap = F.relu(weighted_activations)  # [1, 1, H', W']

        # [0, 1]으로 고르게 한다
        heatmap = heatmap / (heatmap.max() + 1e-8)

        # 들임 크기로 키운다
        heatmap = F.interpolate(
            heatmap,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        return heatmap.squeeze()  # [H, W]


def get_last_conv_layer(model, model_name='resnet50'):
    """얼개마다 마지막 겹치는 켜를 집어 준다."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'densenet' in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"모르는 얼개: {model_name}")


def example_1_basic_gradcam():
    """Grad-CAM 기본 쓰임."""
    print("\n" + "="*60)
    print("보기 1: 기본 Grad-CAM")
    print("="*60)

    device = get_device()
    create_output_dir('outputs')

    # 모형을 부른다
    model_name = 'resnet50'
    model = load_pretrained_model(model_name, device)

    # 마지막 겹치는 켜를 집는다
    target_layer = get_last_conv_layer(model, model_name)
    print(f"겨눈 켜: {target_layer.__class__.__name__}")

    # Grad-CAM을 만든다
    gradcam = GradCAM(model, target_layer)

    # 시험 그림
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    image_tensor = preprocess_image(test_image, requires_grad=False)

    # 미루어 봄을 얻는다
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, target_class].item()

    print(f"\n미루어 본 갈래: {target_class}")
    print(f"자신함: {confidence:.2%}")

    # Grad-CAM을 셈한다
    print("\nGrad-CAM을 셈하는 중...")
    heatmap = gradcam(image_tensor, target_class, device)

    print(f"열 그림 꼴: {heatmap.shape}")
    print(f"값 너비: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # 그림으로 보인다
    visualize_saliency(
        image_tensor,
        heatmap,
        title=f"Grad-CAM (갈래 {target_class})",
        colormap='jet',
        alpha=0.4,
        save_path='outputs/05_gradcam_basic.png'
    )

    print("\n✓ Grad-CAM이 성긴 자리 짚기를 보여 준다!")


def example_2_compare_with_gradients():
    """Grad-CAM과 기울기 방법을 견준다."""
    print("\n" + "="*60)
    print("보기 2: Grad-CAM 대 기울기 방법")
    print("="*60)

    device = get_device()
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(180, 120, 80))
    image_tensor_grad = preprocess_image(test_image, requires_grad=True)
    image_tensor_cam = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor_cam.to(device))
        target_class = output.argmax(dim=1).item()

    # 맨 기울기
    print("맨 기울기를 셈하는 중...")
    output = model(image_tensor_grad.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_tensor_grad.grad), dim=1)[0]

    # Grad-CAM
    print("Grad-CAM을 셈하는 중...")
    target_layer = get_last_conv_layer(model, 'resnet50')
    gradcam = GradCAM(model, target_layer)
    gradcam_map = gradcam(image_tensor_cam, target_class, device)

    # 견준다
    saliencies = {
        '맨 기울기\n(그림점 낱)': vanilla,
        'Grad-CAM\n(자리 낱)': gradcam_map
    }

    visualize_multiple_saliencies(
        image_tensor_cam, saliencies,
        save_path='outputs/05_gradcam_vs_gradient.png'
    )

    print("\n고갱이 차이:")
    print("- 맨 기울기: 그림점 낱, 잡음 많음, 결 고움")
    print("- Grad-CAM: 자리 낱, 깨끗함, 성김")
    print("- '어디'에는 Grad-CAM이, '무엇'에는 기울기가 낫다")
    print("\n✓ 서로 채워 주는 방법이다!")


def main():
    print("\n" + "="*70)
    print(" "*20 + "GRAD-CAM 익히기")
    print("="*70)

    try:
        example_1_basic_gradcam()
        example_2_compare_with_gradients()

        print("\n" + "="*70)
        print("고갱이:")
        print("1. Grad-CAM: 성기지만 갈래를 가려내는 자리 짚기")
        print("2. 마지막 겹치는 켜의 살아남 + 기울기를 쓴다")
        print("3. 그림이 깨끗하고 풀이하기 쉽다")
        print("4. 자리 얼개를 지닌 CNN에만 쓸 수 있다")
        print("\n다음: 꾸러미 06 - 이끈 되짚기")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

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
05: Grad-CAM 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_05: grad-cam():
        model = 05: Grad-CAM(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 05: Grad-CAM

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 기울기 바탕 풀이의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `GradCAM`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
