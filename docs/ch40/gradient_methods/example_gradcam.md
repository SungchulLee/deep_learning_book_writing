# Grad-CAM 예제

쓰는 보기: PyTorch 모델에 거는 Grad-CAM. 이 글은 Grad-CAM으로 CNN의 추론을 그리는 법을 보인다.

뉴럴 네트워크가 무엇을 배우는지 아는 일은 신뢰를 쌓고 모델을 디버깅하는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 처리하고 어떻게 판단하는지 드러내는 그래디언트 기반 설명 기법을 보이며, 네트워크의 행동을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
쓰는 보기: PyTorch 모형에 거는 Grad-CAM

이 글은 Grad-CAM으로 CNN의 미루어 봄을 그리는 법을 보인다.
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
    어떤 CNN 얼개에도 쓰는 Grad-CAM 짜보기.

    Args:
        model: PyTorch CNN 모형
        target_layer: 그릴 겹치는 레이어

    쓰임:
        gradcam = GradCAM(model, model.layer4[-1])  # ResNet에 쓸 때
        heatmap = gradcam(image_tensor, target_class, device)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 갈고리를 건다
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """앞으로 걸음의 살아남을 붙드는 갈고리."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """역전파 걸음의 기울기를 붙드는 갈고리."""
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Grad-CAM 열 그림을 셈한다.

        Args:
            image_tensor: 들임 그림 [1, C, H, W]
            target_class: 겨눈 갈래 번호. None이면 미루어 본 갈래를 쓴다.
            device: 셈하는 장치

        Returns:
            [0, 1] 너비의 열 그림 텐서 [H, W]
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        image_tensor = image_tensor.to(device)

        # 앞으로 걸음 - 앞으로 갈고리를 당긴다
        output = self.model(image_tensor)

        # 겨눈 갈래를 정한다
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 겨눈 갈래의 점수를 집는다
        target_score = output[0, target_class]

        # 역전파 걸음 - 역전파 갈고리를 당긴다
        self.model.zero_grad()
        target_score.backward()

        # 기울기와 살아남을 집는다
        gradients = self.gradients[0]    # 형태: [K, H', W']
        activations = self.activations[0] # 형태: [K, H', W']

        # 중요함 짐을 셈한다: α_k = 온세상고르게모으기(기울기)
        # α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        weights = gradients.mean(dim=(1, 2))  # 형태: [K]

        # 짐 실은 아우름: L = Σ_k α_k^c * A^k
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU을 건다
        cam = F.relu(cam)

        # [0, 1]으로 고르게 한다
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 들임 결 고움으로 키운다
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
        본디 그림 위에 겹쳐 보이는 Grad-CAM 그림을 만든다.

        Args:
            image_tensor: 들임 그림 텐서
            original_image: [0, 255] 너비의 numpy 배열 (H, W, 3) 본디 그림
            target_class: 겨눈 갈래 번호
            alpha: 겹침의 비침 정도(0이면 그림만, 1이면 열 그림만)

        Returns:
            [0, 255] 너비의 numpy 배열 (H, W, 3) 그림
        """
        import cv2

        # CAM을 만든다
        cam = self(image_tensor, target_class)
        cam_np = cam.cpu().numpy()

        # 들임 크기로 맞춘다
        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = image_tensor.shape[2:]

        cam_resized = cv2.resize(cam_np, (w, h))

        # 열 그림으로 바꾼다(OpenCV은 BGR)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 본디 그림 위에 겹친다
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
    Grad-CAM에 알맞은 겨눈 레이어를 집는다.

    Args:
        model: 미리 익힌 모형
        architecture: 모형 얼개 이름

    Returns:
        겨눈 겹치는 레이어
    """
    architecture = architecture.lower()

    if 'resnet' in architecture:
        # ResNet: layer4[-1]이 마지막 bottleneck/basicblock이다
        return model.layer4[-1]

    elif 'vgg' in architecture:
        # VGG: 분류기 앞의 마지막 겹치는 레이어
        return model.features[-1]

    elif 'densenet' in architecture:
        # DenseNet: 마지막 빽빽한 덩이
        return model.features.denseblock4

    elif 'efficientnet' in architecture:
        # EfficientNet: 마지막 겹치는 레이어
        return model.features[-1]

    elif 'mobilenet' in architecture:
        # MobileNet: 마지막 겹치는 레이어
        return model.features[-1]

    elif 'inception' in architecture:
        # Inception: Mixed 레이어
        return model.Mixed_7c

    else:
        raise ValueError(f"모르는 얼개: {architecture}")

def compare_gradcam_classes(
    model: nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list,
    device: torch.device
):
    """
    여러 겨눈 갈래의 Grad-CAM 열 그림을 견준다.

    갈래가 다르면 다른 자리를 짚는다는 것을 보인다.
    """
    n_classes = len(class_indices)
    fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))

    # 보여 주려고 그림을 되돌려 고른다
    image_np = denormalize_image(image_tensor)

    # 본디 그림
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('본디', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for idx, (class_idx, class_name) in enumerate(zip(class_indices, class_names)):
        # 이 갈래의 Grad-CAM을 셈한다
        heatmap = gradcam(image_tensor, class_idx, device)
        heatmap_np = heatmap.cpu().numpy()

        # 열 그림
        axes[0, idx + 1].imshow(heatmap_np, cmap='jet')
        axes[0, idx + 1].set_title(f'{class_name}\n(갈래 {class_idx})', fontsize=10)
        axes[0, idx + 1].axis('off')

        # 겹쳐 보이기
        overlay = create_overlay(image_np, heatmap_np, alpha=0.5)
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title('겹쳐 보이기', fontsize=10)
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    return fig

# 보기: 고양이와 개가 함께 있는 그림
# cam_cat은 고양이 자리를 짚는다
# cam_dog은 개 자리를 짚는다

def negative_gradcam(gradcam, image_tensor, target_class, device):
    """
    겨눈 갈래일 낌새를 떨어뜨리는 자리를 셈한다.

    모형이 무엇을 겨눈 갈래가 '아니라고' 여기는지 아는 데 쓸모 있다.
    """
    model = gradcam.model
    model.eval()

    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    target_score = output[0, target_class]

    model.zero_grad()
    target_score.backward()

    # 음수 짐을 쓴다(갈래 점수를 떨어뜨리는 자리)
    weights = -gradcam.gradients.mean(dim=(2, 3), keepdim=True)

    weighted = weights * gradcam.activations
    heatmap = weighted.sum(dim=1, keepdim=True)
    heatmap = F.relu(heatmap)

    # 고르게 한다
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return F.interpolate(
        heatmap, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze()

def analyze_layer_gradcam(model, image_tensor, target_class, device):
    """
    레이어마다의 Grad-CAM을 견주어 결의 레이어 얼개를 알아본다.
    """
    # ResNet이면 layer2, layer3, layer4을 견준다
    layers = {
        'layer2 (가운데 레이어)': model.layer2[-1],
        'layer3 (높은 레이어)': model.layer3[-1],
        'layer4 (뜻 레이어)': model.layer4[-1]
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
    """Grad-CAM++ — 짐을 매기는 방법만 다르다.

    Grad-CAM은 기울기를 그냥 평균내어 레이어마다 짐 하나를 얻는다(α_k = mean(∂y/∂A^k)).
    그래서 같은 갈래의 물체가 그림에 여럿 있으면 가장 센 것 하나에 쏠린다.

    Grad-CAM++는 자리마다 짐을 따로 매긴다. 양의 기울기만 세되,
    그 자리가 얼마나 드문지를 2계·3계 항으로 나누어 준다.

        α^kij = (∂²y/∂A²) / (2·∂²y/∂A² + Σ A^k · ∂³y/∂A³)
        w_k   = Σ_ij α^kij · relu(∂y/∂A^k_ij)

    점수에 지수를 씌운 경우 ∂ⁿy/∂Aⁿ가 (∂y/∂A)ⁿ이 되므로, 기울기 하나로
    위 식을 그대로 셈할 수 있다. 아래가 그 형태.
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
    모형에 넣을 그림을 부르고 미리 다듬는다.

    Args:
        image_path: 그림 두루마리의 길
        image_size: 겨눈 그림 크기

    Returns:
        미리 다듬은 그림 텐서와 본디 그림 배열
    """
    # 그림을 부른다
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # 미리 다듬기를 세운다
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 미리 다듬는다
    input_tensor = preprocess(image).unsqueeze(0)

    # 겹쳐 보이려고 본디 그림의 크기를 맞춘다
    original_image = np.array(Image.fromarray(original_image).resize((image_size, image_size)))

    return input_tensor, original_image


def example_resnet_gradcam():
    """
    보기: ResNet50에 Grad-CAM 걸기
    """
    print("=" * 60)
    print("보기 1: ResNet50에 거는 Grad-CAM")
    print("=" * 60)

    # 미리 익힌 ResNet50을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()

    # 마지막 겹치는 레이어를 집는다
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 보기: 아무 그림이나 만든다(참 그림 길로 갈음하라)
    print("\n보기 들임을 만든다(참 그림으로 갈음하라)...")
    input_tensor = torch.randn(1, 3, 224, 224)

    # CAM을 만든다
    print("Grad-CAM 열 그림을 만드는 중...")
    cam = gradcam.generate_cam(input_tensor, target_class=None)

    print(f"CAM 형태: {cam.shape}")
    print(f"CAM 가장 작음/큼: {cam.min():.4f} / {cam.max():.4f}")

    # 그림으로 보인다
    print("그림을 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor)

    # 보여 준다
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title('Grad-CAM 그림')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_example.png', dpi=150, bbox_inches='tight')
    print("그림을 'gradcam_example.png'에 갈무리했다")
    plt.close()


def example_with_real_image(image_path: str):
    """
    보기: 참 그림에 거는 Grad-CAM

    Args:
        image_path: 들임 그림의 길
    """
    print("\n" + "=" * 60)
    print("보기 2: 참 그림에 거는 Grad-CAM")
    print("=" * 60)

    # 모형을 부른다
    print("ResNet50을 부르는 중...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # 그림을 부르고 미리 다듬는다
    print(f"{image_path}에서 그림을 부르는 중...")
    input_tensor, original_image = load_and_preprocess_image(image_path)

    # 겨눈 레이어를 집는다
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 앞으로 스텝으로 미루어 봄을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

    print(f"미루어 본 클래스: {pred_class} (확신도: {pred_prob:.2%})")

    # Grad-CAM을 만든다
    print("Grad-CAM을 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor,
                                         target_class=pred_class,
                                         original_image=original_image)

    # 견줌 그림을 만든다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('본디 그림')
    axes[0].axis('off')

    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM 열 그림')
    axes[1].axis('off')

    axes[2].imshow(visualization)
    axes[2].set_title(f'겹쳐 보이기 (클래스: {pred_class})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
    print("견줌 그림을 'gradcam_comparison.png'에 갈무리했다")
    plt.close()


def example_gradcam_plusplus():
    """
    보기: Grad-CAM과 Grad-CAM++ 견주기
    """
    print("\n" + "=" * 60)
    print("보기 3: Grad-CAM 대 Grad-CAM++")
    print("=" * 60)

    # 모형을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # 아무 들임이나 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 여느 Grad-CAM
    print("여느 Grad-CAM을 만드는 중...")
    gradcam = GradCAM(model, target_layer)
    cam1 = gradcam.generate_cam(input_tensor)
    vis1 = gradcam.visualize_cam(input_tensor)

    # Grad-CAM++
    print("Grad-CAM++을 만드는 중...")
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam2 = gradcam_pp.generate_cam(input_tensor)
    vis2 = gradcam_pp.visualize_cam(input_tensor)

    # 견준다
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(cam1, cmap='jet')
    axes[0, 0].set_title('Grad-CAM 열 그림')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis1)
    axes[0, 1].set_title('Grad-CAM 그림')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cam2, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++ 열 그림')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vis2)
    axes[1, 1].set_title('Grad-CAM++ 그림')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison_methods.png', dpi=150, bbox_inches='tight')
    print("방법 견줌을 'gradcam_comparison_methods.png'에 갈무리했다")
    plt.close()


def example_multiple_classes():
    """
    보기: 여러 클래스에 대한 Grad-CAM 그리기
    """
    print("\n" + "=" * 60)
    print("보기 4: 여러 클래스에 대한 Grad-CAM")
    print("=" * 60)

    # 모형을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 들임을 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 앞선 5개 미루어 봄을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        probs, classes = torch.softmax(output, dim=1).topk(5)

    # 앞선 클래스마다 CAM을 만든다
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (prob, cls) in enumerate(zip(probs[0], classes[0])):
        cam = gradcam.generate_cam(input_tensor, target_class=cls.item())
        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(f'클래스 {cls.item()}\n({prob.item():.2%})')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_multiple_classes.png', dpi=150, bbox_inches='tight')
    print("여러 클래스 그림을 'gradcam_multiple_classes.png'에 갈무리했다")
    plt.close()


def example_different_architectures():
    """
    보기: 여러 구조에 Grad-CAM 걸기
    """
    print("\n" + "=" * 60)
    print("보기 5: 여러 구조에 거는 Grad-CAM")
    print("=" * 60)

    architectures = {
        'ResNet50': (models.resnet50(pretrained=True), 'layer4'),
        'VGG16': (models.vgg16(pretrained=True), 'features'),
        'MobileNetV2': (models.mobilenet_v2(pretrained=True), 'features'),
    }

    input_tensor = torch.randn(1, 3, 224, 224)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, (model, layer_name)) in enumerate(architectures.items()):
        print(f"\n{name}을 다루는 중...")
        model.eval()

        # 겨눈 레이어를 집는다
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
    print("\n구조 견줌을 'gradcam_different_architectures.png'에 갈무리했다")
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM 보기\n")

    # 보기를 돌린다
    example_resnet_gradcam()
    example_gradcam_plusplus()
    example_multiple_classes()
    example_different_architectures()

    # 참 그림에 쓰려면 주석을 푼다
    # example_with_real_image('path/to/your/image.jpg')

    print("\n" + "=" * 60)
    print("보기를 모두 마쳤다!")
    print("=" * 60)
```

## 2. 논의

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다. 시각화 코드는 학습된 표현, 수렴하는 패턴, 평가 지표를 들여다보게 해서 추상적인 계산을 눈에 보이게 한다.

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
쓰는 보기 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_example usage():
        model = 쓰는 보기(...)
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

**다룬 것** — 쓰는 보기

시각화는 모델의 행동을 이해하고 학습의 문제를 탐지하는 데 큰 역할을 한다.

앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
