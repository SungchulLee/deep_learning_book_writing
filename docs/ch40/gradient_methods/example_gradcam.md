# Grad-CAM 예제

사전 트레인된 PyTorch 모델에 Grad-CAM을 적용하는 예제다. Grad-CAM으로 CNN의 예측 근거를 시각화하고, Grad-CAM++와 비교하고, 여러 클래스와 여러 구조에 걸어 보는 법을 보인다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
사용 예제: PyTorch 모델에 적용하는 Grad-CAM

이 글은 Grad-CAM으로 CNN의 예측을 시각화하는 법을 보인다.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# --- gradcam ---
class GradCAM:
    """
    어떤 CNN 구조에도 쓸 수 있는 Grad-CAM 구현.

    Args:
        model: PyTorch CNN 모델
        target_layer: 시각화할 컨볼루션 레이어

    사용법:
        gradcam = GradCAM(model, model.layer4[-1])  # ResNet에 쓸 때
        heatmap = gradcam(image_tensor, target_class, device)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 훅을 건다
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """순전파의 활성값을 저장하는 훅."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """역전파의 그래디언트를 저장하는 훅."""
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Grad-CAM 히트맵을 계산한다.

        Args:
            image_tensor: 입력 이미지 [1, C, H, W]
            target_class: 대상 클래스 번호. None이면 예측 클래스를 쓴다.
            device: 계산할 장치

        Returns:
            [0, 1] 범위의 히트맵 텐서 [H, W]
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        image_tensor = image_tensor.to(device)

        # 순전파 - 순전파 훅이 호출된다
        output = self.model(image_tensor)

        # 대상 클래스를 정한다
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 대상 클래스의 점수를 가져온다
        target_score = output[0, target_class]

        # 역전파 - 역전파 훅이 호출된다
        self.model.zero_grad()
        target_score.backward()

        # 그래디언트와 활성값을 가져온다
        gradients = self.gradients[0]    # 형태: [K, H', W']
        activations = self.activations[0] # 형태: [K, H', W']

        # 중요도 웨이트를 계산한다: α_k = 전역 평균 풀링(그래디언트)
        # α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        weights = gradients.mean(dim=(1, 2))  # 형태: [K]

        # 가중 결합: L = Σ_k α_k^c * A^k
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU를 건다
        cam = F.relu(cam)

        # [0, 1]로 정규화한다
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 입력 해상도로 업샘플링한다
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return cam.squeeze()  # [H, W]

    def generate_visualization(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        target_class: int = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        원본 이미지 위에 겹친 Grad-CAM 시각화를 만든다.

        Args:
            image_tensor: 입력 이미지 텐서
            original_image: [0, 255] 범위의 numpy 배열 (H, W, 3) 원본 이미지
            target_class: 대상 클래스 번호
            alpha: 히트맵의 투명도(0이면 이미지만, 1이면 히트맵만)

        Returns:
            [0, 255] 범위의 numpy 배열 (H, W, 3) 이미지
        """
        import cv2

        # CAM을 만든다
        cam = self(image_tensor, target_class)
        cam_np = cam.cpu().numpy()

        # 입력 크기로 맞춘다
        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = image_tensor.shape[2:]

        cam_resized = cv2.resize(cam_np, (w, h))

        # 히트맵으로 바꾼다(OpenCV는 BGR)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 원본 이미지 위에 겹친다
        if original_image is not None:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            visualization = cv2.addWeighted(
                original_image, 1 - alpha, heatmap, alpha, 0
            )
        else:
            visualization = heatmap

        return visualization

def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """
    Grad-CAM에 알맞은 대상 레이어를 고른다.

    Args:
        model: 사전 트레인된 모델
        architecture: 모델 구조 이름

    Returns:
        대상 컨볼루션 레이어
    """
    architecture = architecture.lower()

    if 'resnet' in architecture:
        # ResNet: layer4[-1]이 마지막 bottleneck/basicblock이다
        return model.layer4[-1]

    elif 'vgg' in architecture:
        # VGG: 분류기 앞의 마지막 레이어
        return model.features[-1]

    elif 'densenet' in architecture:
        # DenseNet: 마지막 dense 블록
        return model.features.denseblock4

    elif 'efficientnet' in architecture:
        # EfficientNet: 마지막 컨볼루션 레이어
        return model.features[-1]

    elif 'mobilenet' in architecture:
        # MobileNet: 마지막 컨볼루션 레이어
        return model.features[-1]

    elif 'inception' in architecture:
        # Inception: Mixed 레이어
        return model.Mixed_7c

    else:
        raise ValueError(f"알 수 없는 구조: {architecture}")

def compare_gradcam_classes(
    model: nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list,
    device: torch.device
):
    """
    여러 대상 클래스의 Grad-CAM 히트맵을 비교한다.

    클래스가 다르면 강조하는 영역도 다르다는 것을 보인다.
    """
    n_classes = len(class_indices)
    fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))

    # 표시하려고 이미지의 정규화를 되돌린다
    image_np = denormalize_image(image_tensor)

    # 원본 이미지
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('원본', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for idx, (class_idx, class_name) in enumerate(zip(class_indices, class_names)):
        # 이 클래스의 Grad-CAM을 계산한다
        heatmap = gradcam(image_tensor, class_idx, device)
        heatmap_np = heatmap.cpu().numpy()

        # 히트맵
        axes[0, idx + 1].imshow(heatmap_np, cmap='jet')
        axes[0, idx + 1].set_title(f'{class_name}\n(클래스 {class_idx})', fontsize=10)
        axes[0, idx + 1].axis('off')

        # 오버레이
        overlay = create_overlay(image_np, heatmap_np, alpha=0.5)
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title('오버레이', fontsize=10)
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    return fig

# 예: 고양이와 개가 함께 있는 이미지
# cam_cat은 고양이 영역을 강조한다
# cam_dog은 개 영역을 강조한다

def negative_gradcam(gradcam, image_tensor, target_class, device):
    """
    대상 클래스의 점수를 떨어뜨리는 영역을 계산한다.

    모델이 무엇을 대상 클래스가 '아니라고' 보는지 아는 데 쓸모 있다.
    """
    model = gradcam.model
    model.eval()

    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    target_score = output[0, target_class]

    model.zero_grad()
    target_score.backward()

    # 음수 웨이트를 쓴다(클래스 점수를 떨어뜨리는 영역)
    weights = -gradcam.gradients.mean(dim=(2, 3), keepdim=True)

    weighted = weights * gradcam.activations
    heatmap = weighted.sum(dim=1, keepdim=True)
    heatmap = F.relu(heatmap)

    # 정규화한다
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return F.interpolate(
        heatmap, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze()

def analyze_layer_gradcam(model, image_tensor, target_class, device):
    """
    레이어마다의 Grad-CAM을 비교해 피처의 계층 구조를 알아본다.
    """
    # ResNet이면 layer2, layer3, layer4를 비교한다
    layers = {
        'layer2 (중간 레이어)': model.layer2[-1],
        'layer3 (높은 레이어)': model.layer3[-1],
        'layer4 (의미 레이어)': model.layer4[-1]
    }

    results = {}
    for name, layer in layers.items():
        gradcam = GradCAM(model, layer)
        heatmap = gradcam(image_tensor, target_class, device)
        results[name] = heatmap.cpu().numpy()

    return results


# 페이지들이 쓰는 이름. 위 클래스가 __call__과 generate_visualization으로
# 정의해 둔 것과 같은 것이며, 이름만 맞춘다.
GradCAM.generate_cam = GradCAM.__call__
GradCAM.visualize_cam = GradCAM.generate_visualization


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ — 웨이트를 매기는 방법만 다르다.

    Grad-CAM은 그래디언트를 그냥 평균 내어 채널마다 웨이트 하나를 얻는다(α_k = mean(∂y/∂A^k)).
    그래서 같은 클래스의 객체가 이미지에 여럿 있으면 가장 강한 것 하나에 쏠린다.

    Grad-CAM++는 위치마다 웨이트를 따로 매긴다. 양의 그래디언트만 세되,
    그 위치가 얼마나 드문지를 2계·3계 항으로 나누어 준다.

        α^kij = (∂²y/∂A²) / (2·∂²y/∂A² + Σ A^k · ∂³y/∂A³)
        w_k   = Σ_ij α^kij · relu(∂y/∂A^k_ij)

    점수에 지수를 씌운 경우 ∂ⁿy/∂Aⁿ가 (∂y/∂A)ⁿ이 되므로, 그래디언트 하나로
    위 식을 그대로 계산할 수 있다. 아래가 그 형태.
    """

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        image_tensor = image_tensor.to(device)
        output = self.model(image_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        grads = self.gradients[0]           # [K, H', W']
        acts = self.activations[0]          # [K, H', W']

        g2, g3 = grads ** 2, grads ** 3
        denom = 2.0 * g2 + (acts * g3).sum(dim=(1, 2), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))      # [K]

        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=image_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        return cam.squeeze()


GradCAMPlusPlus.generate_cam = GradCAMPlusPlus.__call__
GradCAMPlusPlus.visualize_cam = GradCAM.generate_visualization

# ========================================================================
# 메인
# ========================================================================


def load_and_preprocess_image(image_path: str, image_size: int = 224):
    """
    모델에 넣을 이미지를 불러와 전처리한다.

    Args:
        image_path: 이미지 파일 경로
        image_size: 목표 이미지 크기

    Returns:
        전처리한 이미지 텐서와 원본 이미지 배열
    """
    # 이미지를 불러온다
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # 전처리 파이프라인을 정의한다
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 전처리한다
    input_tensor = preprocess(image).unsqueeze(0)

    # 오버레이하려고 원본 이미지의 크기를 맞춘다
    original_image = np.array(Image.fromarray(original_image).resize((image_size, image_size)))

    return input_tensor, original_image


def example_resnet_gradcam():
    """
    예제: ResNet50에 Grad-CAM 적용하기
    """
    print("=" * 60)
    print("예제 1: ResNet50에 적용하는 Grad-CAM")
    print("=" * 60)

    # 사전 트레인된 ResNet50을 불러온다
    model = models.resnet50(pretrained=True)
    model.eval()

    # 마지막 컨볼루션 레이어를 고른다
    target_layer = model.layer4[-1]

    # Grad-CAM을 초기화한다
    gradcam = GradCAM(model, target_layer)

    # 예: 무작위 이미지를 만든다(실제 이미지 경로로 바꾸라)
    print("\n예제 입력을 만든다(실제 이미지로 바꾸라)...")
    input_tensor = torch.randn(1, 3, 224, 224)

    # CAM을 만든다
    print("Grad-CAM 히트맵을 만드는 중...")
    cam = gradcam.generate_cam(input_tensor, target_class=None)

    print(f"CAM 형태: {cam.shape}")
    print(f"CAM 최솟값/최댓값: {cam.min():.4f} / {cam.max():.4f}")

    # 시각화한다
    print("시각화를 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor)

    # 표시한다
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title('Grad-CAM 시각화')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_example.png', dpi=150, bbox_inches='tight')
    print("시각화를 'gradcam_example.png'에 저장했다")
    plt.close()


def example_with_real_image(image_path: str):
    """
    예제: 실제 이미지에 적용하는 Grad-CAM

    Args:
        image_path: 입력 이미지 경로
    """
    print("\n" + "=" * 60)
    print("예제 2: 실제 이미지에 적용하는 Grad-CAM")
    print("=" * 60)

    # 모델을 불러온다
    print("ResNet50을 불러오는 중...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # 이미지를 불러와 전처리한다
    print(f"{image_path}에서 이미지를 불러오는 중...")
    input_tensor, original_image = load_and_preprocess_image(image_path)

    # 대상 레이어를 고른다
    target_layer = model.layer4[-1]

    # Grad-CAM을 초기화한다
    gradcam = GradCAM(model, target_layer)

    # 순전파로 예측을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

    print(f"예측 클래스: {pred_class} (확신도: {pred_prob:.2%})")

    # Grad-CAM을 만든다
    print("Grad-CAM을 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor,
                                         target_class=pred_class,
                                         original_image=original_image)

    # 비교 그림을 만든다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM 히트맵')
    axes[1].axis('off')

    axes[2].imshow(visualization)
    axes[2].set_title(f'오버레이 (클래스: {pred_class})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
    print("비교 그림을 'gradcam_comparison.png'에 저장했다")
    plt.close()


def example_gradcam_plusplus():
    """
    예제: Grad-CAM과 Grad-CAM++ 비교하기
    """
    print("\n" + "=" * 60)
    print("예제 3: Grad-CAM 대 Grad-CAM++")
    print("=" * 60)

    # 모델을 불러온다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # 무작위 입력을 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 일반 Grad-CAM
    print("일반 Grad-CAM을 만드는 중...")
    gradcam = GradCAM(model, target_layer)
    cam1 = gradcam.generate_cam(input_tensor)
    vis1 = gradcam.visualize_cam(input_tensor)

    # Grad-CAM++
    print("Grad-CAM++를 만드는 중...")
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam2 = gradcam_pp.generate_cam(input_tensor)
    vis2 = gradcam_pp.visualize_cam(input_tensor)

    # 비교한다
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(cam1, cmap='jet')
    axes[0, 0].set_title('Grad-CAM 히트맵')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis1)
    axes[0, 1].set_title('Grad-CAM 오버레이')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cam2, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++ 히트맵')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vis2)
    axes[1, 1].set_title('Grad-CAM++ 오버레이')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison_methods.png', dpi=150, bbox_inches='tight')
    print("방법 비교를 'gradcam_comparison_methods.png'에 저장했다")
    plt.close()


def example_multiple_classes():
    """
    예제: 여러 클래스에 대한 Grad-CAM 시각화
    """
    print("\n" + "=" * 60)
    print("예제 4: 여러 클래스에 대한 Grad-CAM")
    print("=" * 60)

    # 모델을 불러온다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # Grad-CAM을 초기화한다
    gradcam = GradCAM(model, target_layer)

    # 입력을 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 상위 5개 예측을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        probs, classes = torch.softmax(output, dim=1).topk(5)

    # 상위 클래스마다 CAM을 만든다
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (prob, cls) in enumerate(zip(probs[0], classes[0])):
        cam = gradcam.generate_cam(input_tensor, target_class=cls.item())
        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(f'클래스 {cls.item()}\n({prob.item():.2%})')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_multiple_classes.png', dpi=150, bbox_inches='tight')
    print("여러 클래스 그림을 'gradcam_multiple_classes.png'에 저장했다")
    plt.close()


def example_different_architectures():
    """
    예제: 여러 구조에 Grad-CAM 적용하기
    """
    print("\n" + "=" * 60)
    print("예제 5: 여러 구조에 적용하는 Grad-CAM")
    print("=" * 60)

    architectures = {
        'ResNet50': (models.resnet50(pretrained=True), 'layer4'),
        'VGG16': (models.vgg16(pretrained=True), 'features'),
        'MobileNetV2': (models.mobilenet_v2(pretrained=True), 'features'),
    }

    input_tensor = torch.randn(1, 3, 224, 224)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, (model, layer_name)) in enumerate(architectures.items()):
        print(f"\n{name}을 처리하는 중...")
        model.eval()

        # 대상 레이어를 고른다
        if layer_name == 'features':
            target_layer = list(model.features.children())[-1]
        else:
            target_layer = get_target_layer(model, layer_name)

        # Grad-CAM을 만든다
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)

        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_different_architectures.png', dpi=150, bbox_inches='tight')
    print("\n구조 비교를 'gradcam_different_architectures.png'에 저장했다")
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM 예제\n")

    # 예제를 실행한다
    example_resnet_gradcam()
    example_gradcam_plusplus()
    example_multiple_classes()
    example_different_architectures()

    # 실제 이미지에 쓰려면 주석을 푼다
    # example_with_real_image('path/to/your/image.jpg')

    print("\n" + "=" * 60)
    print("예제를 모두 마쳤다!")
    print("=" * 60)
```

## 2. 논의

시각화는 모델의 행동을 이해하고 문제를 탐지하는 데 큰 역할을 한다. 이 쪽의 예제는 `GradCAM` 과 이를 상속한 `GradCAMPlusPlus` 하나로 다섯 가지 상황을 다룬다. 두 클래스는 중요도 웨이트를 매기는 방식만 다르다 — Grad-CAM은 그래디언트의 공간 평균을, Grad-CAM++는 양수 그래디언트의 픽셀별 가중합을 쓴다.

읽을 때 짚어 둘 점이 있다. 실제 이미지를 쓰는 `example_with_real_image` 는 주석 처리되어 있고, 나머지 예제는 모두 `torch.randn` 으로 만든 잡음 입력을 쓴다. 그러니 여기서 나오는 히트맵은 코드가 돌아가는지 확인하는 용도이지, 객체의 위치를 파악하는 모습을 보여 주지는 않는다. 또 `example_different_architectures` 는 ResNet50에 `get_target_layer(model, 'layer4')` 를 부르는데, 이 함수는 구조 이름에 `'resnet'` 이 들어 있어야 레이어를 돌려주므로 `'layer4'` 로는 `ValueError` 가 난다. 이 쪽에는 실행 출력이 실려 있지 않다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 설명에 알맞은지 설명하라.

</div>

??? success "풀이"
    (1) **훅으로 중간값 붙잡기** -- `register_forward_hook` 과 `register_full_backward_hook` 으로 타깃 레이어의 활성화와 그래디언트를 모델 코드를 고치지 않고 얻는다. 그래서 ResNet·VGG·MobileNet에 같은 클래스를 쓸 수 있다. (2) **상속으로 Grad-CAM++ 만들기** -- `GradCAMPlusPlus` 는 훅과 시각화를 물려받고 웨이트 계산만 바꾼다. 두 방법의 차이가 바로 그 몇 줄이라는 것이 코드에 드러난다. (3) **[0, 1] 정규화와 입력 크기로 보간** -- 서로 다른 레이어·모델의 맵을 같은 색 척도로 비교하고 원본 위에 겹칠 수 있게 한다. 대신 맵의 절대 크기 정보는 잃는다.

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
    흔한 실패 유형은 이렇다. (1) **타깃 레이어가 컨볼루션이 아님** -- VGG16·MobileNetV2에서 `list(model.features.children())[-1]` 은 각각 맥스 풀링과 Conv-BN-ReLU6 블록이라, 기대한 마지막 컨볼루션 출력과 다를 수 있다. 레이어 이름과 출력 형태를 찍어 탐지하고, 마지막 `nn.Conv2d` 를 찾아 고른다. (2) **훅이 쌓이거나 in-place 연산과 충돌함** -- 같은 모델에 `GradCAM` 을 여러 번 만들면 훅이 계속 붙고, torchvision의 `inplace=True` ReLU 뒤에서는 `register_full_backward_hook` 이 오류를 낼 수 있다. 오류 메시지나 이상한 맵으로 탐지하고, 훅 핸들을 저장해 쓰고 난 뒤 `remove()` 하며, 필요하면 ReLU의 `inplace` 를 끈다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
이 쪽의 `GradCAM` 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_example usage():
        model = GradCAM(...)
        # 보통 입력
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
    구조가 엔드투엔드 트레인을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — Grad-CAM 예제

사전 트레인된 CNN에 Grad-CAM과 Grad-CAM++를 걸어 예측 근거를 시각화하는 법을 보았다. 두 방법은 중요도 웨이트를 매기는 방식만 다르다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
