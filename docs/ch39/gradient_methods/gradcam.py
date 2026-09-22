"""gradcam — gradcam 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch38/gradient_methods/gradcam.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    어떤 CNN 얼개에도 쓰는 Grad-CAM 짜보기.

    Args:
        model: PyTorch CNN 모형
        target_layer: 그릴 겹치는 켜

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
        """되짚기 걸음의 기울기를 붙드는 갈고리."""
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

        # 되짚기 걸음 - 되짚기 갈고리를 당긴다
        self.model.zero_grad()
        target_score.backward()

        # 기울기와 살아남을 집는다
        gradients = self.gradients[0]    # 꼴: [K, H', W']
        activations = self.activations[0] # 꼴: [K, H', W']

        # 중요함 짐을 셈한다: α_k = 온세상고르게모으기(기울기)
        # α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        weights = gradients.mean(dim=(1, 2))  # 꼴: [K]

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
    Grad-CAM에 알맞은 겨눈 켜를 집는다.

    Args:
        model: 미리 익힌 모형
        architecture: 모형 얼개 이름

    Returns:
        겨눈 겹치는 켜
    """
    architecture = architecture.lower()

    if 'resnet' in architecture:
        # ResNet: layer4[-1]이 마지막 bottleneck/basicblock이다
        return model.layer4[-1]

    elif 'vgg' in architecture:
        # VGG: 가름개 앞의 마지막 겹치는 켜
        return model.features[-1]

    elif 'densenet' in architecture:
        # DenseNet: 마지막 빽빽한 덩이
        return model.features.denseblock4

    elif 'efficientnet' in architecture:
        # EfficientNet: 마지막 겹치는 켜
        return model.features[-1]

    elif 'mobilenet' in architecture:
        # MobileNet: 마지막 겹치는 켜
        return model.features[-1]

    elif 'inception' in architecture:
        # Inception: Mixed 켜
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
    켜마다의 Grad-CAM을 견주어 결의 켜 얼개를 알아본다.
    """
    # ResNet이면 layer2, layer3, layer4을 견준다
    layers = {
        'layer2 (가운데 켜)': model.layer2[-1],
        'layer3 (높은 켜)': model.layer3[-1],
        'layer4 (뜻 켜)': model.layer4[-1]
    }

    results = {}
    for name, layer in layers.items():
        gradcam = GradCAM(model, layer)
        heatmap = gradcam(image_tensor, target_class, device)
        results[name] = heatmap.cpu().numpy()

    return results
