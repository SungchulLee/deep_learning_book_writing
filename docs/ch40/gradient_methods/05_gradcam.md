# 05: Grad-CAM

Grad-CAM(그래디언트 가중 클래스 활성화 맵)을 구현하고 바닐라 그래디언트와 견주려는 쪽이다. Grad-CAM은 마지막 컨볼루션 레이어의 피처 맵을 그 그래디언트의 평균으로 가중합해, 어느 영역이 그 클래스를 예측하게 했는지 보이는 히트맵을 만든다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
05: Grad-CAM - 그래디언트 짐 실은 클래스 살아남 그림
========================================================

어려움: 가운데

설명:
Grad-CAM은 마지막 겹치는 레이어로 흘러드는 그래디언트를 써서 중요한 자리를
짚어 주는 성긴 자리 그림을 낸다. 픽셀 낱의 방법과 달리 Grad-CAM은
어느 자리가 중요한지를 보여 준다.

수학 밑바탕:
    α_k = (1/Z) Σᵢⱼ (∂y_c/∂A_k^(i,j))    [중요함 짐]
    L_Grad-CAM = ReLU(Σ_k α_k A_k)         [짐 실은 합]

여기서:
- A_k: 마지막 겹치는 레이어의 k번째 결 그림
- α_k: 온 세상에 걸쳐 고르게 모은 그래디언트
- ReLU: 기여가 양수인 것만

나은 점:
- 클래스를 가려낸다(클래스가 다르면 다른 자리가 보인다)
- 성긴 자리 짚기("무엇"이 아니라 "어디"를 보인다)
- 어떤 CNN 구조에도 듣는다
- 눈으로 풀이할 수 있다

지은이: 교육용 자료
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# ========================================================================
# 메인
# ========================================================================

class GradCAM:
    """어떤 CNN에도 쓰는 Grad-CAM 구현."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN 모형
            target_layer: 마지막 겹치는 레이어(보기로 model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 갈고리를 건다
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """앞으로 스텝의 살아남을 갈무리한다."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """역전파 스텝의 그래디언트를 갈무리한다."""
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_class, device):
        """
        Grad-CAM을 셈한다.

        Returns:
            torch.Tensor: 열 그림 [1, H, W]
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)

        # 앞으로 스텝
        output = self.model(image_tensor)
        target_score = output[0, target_class]

        # 역전파 스텝
        self.model.zero_grad()
        target_score.backward()

        # 살아남과 그래디언트를 집는다
        activations = self.activations  # [1, C, H', W']
        gradients = self.gradients       # [1, C, H', W']

        # 그래디언트를 온 세상에 걸쳐 고르게 모은다: α_k = (1/Z) Σᵢⱼ ∂y_c/∂A_k^(i,j)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 짐 실어 아우른다: Σ_k α_k A_k
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU을 건다: 기여가 양수인 것만
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
    """구조마다 마지막 겹치는 레이어를 집어 준다."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'densenet' in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"모르는 구조: {model_name}")


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

    # 마지막 겹치는 레이어를 집는다
    target_layer = get_last_conv_layer(model, model_name)
    print(f"겨눈 레이어: {target_layer.__class__.__name__}")

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

    print(f"\n미루어 본 클래스: {target_class}")
    print(f"확신도: {confidence:.2%}")

    # Grad-CAM을 셈한다
    print("\nGrad-CAM을 셈하는 중...")
    heatmap = gradcam(image_tensor, target_class, device)

    print(f"열 그림 형태: {heatmap.shape}")
    print(f"값 너비: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # 그림으로 보인다
    visualize_saliency(
        image_tensor,
        heatmap,
        title=f"Grad-CAM (클래스 {target_class})",
        colormap='jet',
        alpha=0.4,
        save_path='outputs/05_gradcam_basic.png'
    )

    print("\n✓ Grad-CAM이 성긴 자리 짚기를 보여 준다!")


def example_2_compare_with_gradients():
    """Grad-CAM과 그래디언트 방법을 견준다."""
    print("\n" + "="*60)
    print("보기 2: Grad-CAM 대 그래디언트 방법")
    print("="*60)

    device = get_device()
    model = load_pretrained_model('resnet50', device)

    test_image = Image.new('RGB', (224, 224), color=(180, 120, 80))
    image_tensor_grad = preprocess_image(test_image, requires_grad=True)
    image_tensor_cam = preprocess_image(test_image, requires_grad=False)

    with torch.no_grad():
        output = model(image_tensor_cam.to(device))
        target_class = output.argmax(dim=1).item()

    # 맨 그래디언트
    print("맨 그래디언트를 셈하는 중...")
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
        '맨 그래디언트\n(픽셀 낱)': vanilla,
        'Grad-CAM\n(자리 낱)': gradcam_map
    }

    visualize_multiple_saliencies(
        image_tensor_cam, saliencies,
        save_path='outputs/05_gradcam_vs_gradient.png'
    )

    print("\n핵심 차이:")
    print("- 맨 그래디언트: 픽셀 낱, 잡음 많음, 결 고움")
    print("- Grad-CAM: 자리 낱, 깨끗함, 성김")
    print("- '어디'에는 Grad-CAM이, '무엇'에는 그래디언트가 낫다")
    print("\n✓ 서로 채워 주는 방법이다!")


def main():
    print("\n" + "="*70)
    print(" "*20 + "GRAD-CAM 익히기")
    print("="*70)

    try:
        example_1_basic_gradcam()
        example_2_compare_with_gradients()

        print("\n" + "="*70)
        print("핵심:")
        print("1. Grad-CAM: 성기지만 클래스를 가려내는 자리 짚기")
        print("2. 마지막 겹치는 레이어의 살아남 + 그래디언트를 쓴다")
        print("3. 그림이 깨끗하고 풀이하기 쉽다")
        print("4. 자리 구조를 지닌 CNN에만 쓸 수 있다")
        print("\n다음: 꾸러미 06 - Guided Backpropagation")
        print("="*70)
    except Exception as e:
        print(f"어긋남: {e}")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
                    GRAD-CAM 익히기
======================================================================

============================================================
보기 1: 기본 Grad-CAM
============================================================
어긋남: name 'get_device' is not defined
```

## 2. 논의

위 출력에서 보듯 이 프로그램은 첫 보기에서 멈춘다. `get_device`·`load_pretrained_model` 같은 도우미를 `from utils import *` 로 가져오는데, 그 `utils` 모듈이 없어 `NameError` 가 났다. 그러니 히트맵은 이 쪽에 실리지 않았고, 핵심 계산은 `GradCAM.__call__` 의 코드로 읽어야 한다.

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
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
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
