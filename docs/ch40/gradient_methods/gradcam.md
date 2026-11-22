# Grad-CAM: 그래디언트 가중 클래스 활성화 맵

**그래디언트 가중 클래스 활성화 맵(Grad-CAM)**은 CNN 기반 모델의 판단을 시각적으로 해석하는 기법이다. 특정 구조를 요구하던 이전의 클래스 활성화 맵(CAM) 방법과 달리, Grad-CAM은 구조를 고치거나 다시 트레인하지 않고도 어떤 CNN 구조에나 작동한다.

Grad-CAM은 기본적인 질문에 답한다. **입력 이미지의 어느 영역이 어떤 클래스를 예측하는 데 가장 중요한가?**

픽셀 단위의 그래디언트 방법과 달리 Grad-CAM은 클래스를 구별하는 이미지 영역을 짚어 주는 **해상도가 낮은 위치 맵**을 낸다. 이는 바닐라 그래디언트의 두 한계를 다룬다.

1. **클래스 판별력**: 클래스가 다르면 뚜렷이 다른 히트맵이 나온다
2. **시각적 해석 용이성**: 매끄럽고 사람이 이해할 수 있는 위치 파악을 낸다

---

## 1. 수학 기초

### 문제 설정

다음을 지닌 CNN 분류기를 생각하자.

- 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$
- 타깃 레이어(흔히 마지막 컨볼루션 레이어)의 피처 맵 $A^k \in \mathbb{R}^{u \times v}$
- 클래스 $c$의 클래스 점수 $y^c$(소프트맥스 이전 값)

클래스 $c$를 예측하는 데 중요한 이미지 영역을 짚어 주는 히트맵 $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{u \times v}$를 찾는다.

### 핵심 구성

Grad-CAM은 컨볼루션 피처 맵에 남아 있는 공간 정보를 활용한다. 핵심 통찰은 **뒤쪽 컨볼루션 레이어에 의미 정보가 담겨 있고** **그래디언트가 타깃 클래스에 대한 중요도를 알려 준다**는 것이다.

타깃 클래스 $c$에 대해 Grad-CAM은 이렇게 계산한다.

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

여기서

- $A^k \in \mathbb{R}^{H' \times W'}$는 타깃 컨볼루션 레이어의 $k$번째 피처 맵
- $\alpha^c_k$는 클래스 $c$에 대한 피처 맵 $k$의 중요도 웨이트
- ReLU는 기여가 양수인 피처만 남긴다

### 중요도 웨이트 계산

중요도 웨이트 $\alpha^c_k$는 그래디언트를 **전역 평균 풀링해** 계산한다.

$$
\alpha^c_k = \underbrace{\frac{1}{Z} \sum_i \sum_j}_{\text{전역 평균 풀링}} \underbrace{\frac{\partial y^c}{\partial A^k_{ij}}}_{\text{그래디언트}}
$$

여기서

- $y^c$는 클래스 $c$의 클래스 점수(소프트맥스 이전 값)
- $Z = H' \times W'$는 공간 위치의 수
- $\frac{\partial y^c}{\partial A^k_{ij}}$는 위치 $(i, j)$에서 활성화 $A^k$에 대한 클래스 점수의 그래디언트

이렇게 그래디언트를 전역 평균하면 피처 맵 $k$와 클래스 $c$에 대한 **뉴런 중요도 웨이트** $\alpha^c_k$가 나온다.

### 이 형태의 직관

**왜 그래디언트를 전역 평균하는가?**

그래디언트 $\frac{\partial y^c}{\partial A^k_{ij}}$는 피처 맵 $k$의 어느 위치가 얼마나 중요한지를 알려 준다. 모든 위치에 걸쳐 평균하면 클래스 $c$에 대한 피처 맵 $k$의 **전체적인 중요도**를 얻는다.

**왜 ReLU인가?**

기여가 음수인(클래스 점수를 떨어뜨리는) 피처는 걸러 낸다. 우리는 타깃 클래스일 확률을 **올리는** 영역만 보고 싶다. 음수 기여를 하는 피처는 다른 클래스에 속하는 경우가 많아 그 클래스의 예측을 해석하는 데 쓸모가 적다.

**왜 마지막 컨볼루션 레이어인가?**

뒤쪽 레이어에는 다음이 있다.

- 고수준 의미 정보
- 클래스를 구별하는 피처
- 위치를 파악할 만큼의 공간 해상도

### 수학적 유도

Grad-CAM 형태는 **일차 테일러 전개**에서 끌어낼 수 있다. 기본 원리에서 시작해 클래스 점수 $y^c$를 모든 피처 맵의 함수로 보자.

$$
y^c = f(A^1, A^2, \ldots, A^K)
$$

일차 테일러 근사로 클래스 점수를 피처 맵의 선형 함수로 어림하면 대략 이렇다.

$$
y^c \approx \sum_k \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} A_{ij}^k
$$

그래디언트가 위치에 따라 거의 일정하다고 가정하고 그 평균 $\alpha_k^c$로 바꾸어 쓰면

$$
y^c \approx \sum_k \alpha_k^c \sum_i \sum_j A_{ij}^k
$$

이는 $\alpha_k^c$가 피처 맵 $k$가 클래스 점수에 얼마나 기여하는지를 잰다는 뜻이다.

전역 평균 풀링이 공간 정보를 하나로 모은다.

$$
\alpha^c_k = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A^k_{ij}}
$$

가중합이 피처 맵을 중요도에 따라 합친다.

$$
L^c(i,j) = \sum_{k=1}^{K} \alpha^c_k A^k_{ij}
$$

마지막으로 ReLU가 음수 기여를 걷어 낸다.

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}(L^c)
$$

---

## 2. 알고리즘

```
알고리즘: Grad-CAM
입력: 이미지 I, CNN 모델 f, 대상 클래스 c, 대상 레이어 l
출력: 히트맵 L_GradCAM

1. 순전파: 레이어 l의 피처 맵 A^k를 계산한다
2. 순전파: 클래스 점수 y^c를 계산한다
3. 역전파: 그래디언트 ∂y^c/∂A^k를 계산한다
4. 피처 맵 k마다:
   α_k^c = 전역평균풀링(∂y^c/∂A^k)
5. 가중합을 계산한다:
   L_GradCAM = ReLU(Σ_k α_k^c * A^k)
6. L_GradCAM을 입력 해상도로 키운다
7. L_GradCAM을 반환한다
```

---

## 3. PyTorch 구현

### 완전한 GradCAM 클래스

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    어떤 CNN 구조에도 쓰는 Grad-CAM 구현.

    Args:
        model: PyTorch CNN 모델
        target_layer: 시각화할 컨볼루션 레이어

    사용 예:
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

        # 대상 클래스의 점수를 고른다
        target_score = output[0, target_class]

        # 역전파 - 역전파 훅이 호출된다
        self.model.zero_grad()
        target_score.backward()

        # 그래디언트와 활성값을 꺼낸다
        gradients = self.gradients[0]    # 형태: [K, H', W']
        activations = self.activations[0] # 형태: [K, H', W']

        # 중요도 웨이트를 계산한다: α_k = 전역평균풀링(그래디언트)
        # α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        weights = gradients.mean(dim=(1, 2))  # 형태: [K]

        # 가중합: L = Σ_k α_k^c * A^k
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for k, w in enumerate(weights):
            cam += w * activations[k]

        # ReLU를 적용한다
        cam = F.relu(cam)

        # [0, 1]로 정규화한다
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # 입력 해상도로 키운다
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
            alpha: 겹침의 투명도(0이면 이미지만, 1이면 히트맵만)

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
```

### 구조별 타깃 레이어 선택

```python
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
        # VGG: 분류기 앞의 마지막 컨볼루션 레이어
        return model.features[-1]

    elif 'densenet' in architecture:
        # DenseNet: 마지막 덴스 블록
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
```

### 전체 사용 예

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 사전 트레인된 ResNet을 불러온다
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 마지막 컨볼루션 레이어를 대상으로 삼는다
target_layer = model.layer4[-1]

# Grad-CAM을 초기화한다
grad_cam = GradCAM(model, target_layer)

# 입력을 전처리한다
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open('cat.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# 예측 클래스에 대한 Grad-CAM을 만든다
cam = grad_cam(input_tensor)
print(f"CAM 형태: {cam.shape}")  # torch.Size([224, 224])

# 지정한 클래스에 대한 Grad-CAM을 만든다(예: 클래스 281 = 얼룩 고양이)
cam_tabby = grad_cam(input_tensor, target_class=281)

# 시각화를 만든다
original_image = np.array(image.resize((224, 224)))
visualization = grad_cam.generate_visualization(
    input_tensor, 
    original_image,
    target_class=281
)

# 보여 준다
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('원본')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam_tabby.cpu().numpy(), cmap='jet')
plt.title('Grad-CAM 히트맵')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(visualization)
plt.title('겹쳐 보기')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 4. 타깃 레이어 선택

어느 레이어를 타깃으로 삼느냐가 Grad-CAM 맵을 크게 가른다.

### 마지막 컨볼루션 레이어(권장)

마지막 컨볼루션 레이어가 의미 정보가 가장 풍부한 피처를 담는다.

| 항목 | 장점 | 단점 |
|--------|-----|-----|
| 의미 수준 | 고수준 개념 | - |
| 위치 파악 | 객체를 잘 덮는다 | 해상도가 낮다 |
| 클래스 판별력 | 강하다 | - |

```python
# 구조마다의 대상 레이어
target_layers = {
    'resnet': model.layer4[-1],
    'vgg': model.features[-1],
    'mobilenet': model.features[-1],
    'densenet': model.features.denseblock4,
    'efficientnet': model.features[-1],
}
```

### 앞쪽 레이어

앞쪽 레이어는 저수준 피처를 담는다.

| 항목 | 장점 | 단점 |
|--------|-----|-----|
| 해상도 | 더 세밀하다 | - |
| 피처 | 텍스처, 에지 | 의미 정보가 적다 |
| 해석 | - | 기여도 맵에 잡음이 많다 |

### 다중 레이어 Grad-CAM

여러 레이어의 Grad-CAM을 결합하면 더 풍부한 설명이 나온다.

```python
def multi_layer_gradcam(model, image_tensor, layers, target_class=None, device=None):
    """여러 레이어에서 Grad-CAM을 만들어 결합한다."""
    import cv2

    cams = []
    for layer in layers:
        gc = GradCAM(model, layer)
        cam = gc(image_tensor, target_class, device)
        cams.append(cam.cpu().numpy())

    # 모든 CAM을 같은 크기로 맞추고 평균한다
    target_size = (224, 224)
    combined = np.zeros(target_size)
    for cam in cams:
        resized = cv2.resize(cam, target_size)
        combined += resized

    combined /= len(cams)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    return combined

# 예: layer3과 layer4를 결합한다
layers = [model.layer3[-1], model.layer4[-1]]
multi_cam = multi_layer_gradcam(model, input_tensor, layers, target_class=281)
```

---

## 5. 고급 기법

### 여러 클래스 비교

Grad-CAM의 핵심 성질은 **클래스를 구별한다**는 것이다.

```python
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

    클래스가 다르면 다른 영역을 강조한다는 것을 보인다.
    """
    n_classes = len(class_indices)
    fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))

    # 보여 주려고 이미지의 정규화를 되돌린다
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

        # 겹쳐 보기
        overlay = create_overlay(image_np, heatmap_np, alpha=0.5)
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title('겹쳐 보기', fontsize=10)
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    return fig

# 예: 고양이와 개가 함께 있는 이미지
# cam_cat은 고양이 영역을 강조한다
# cam_dog은 개 영역을 강조한다
```

### 음수 Grad-CAM

클래스일 확률을 **떨어뜨리는** 영역(반사실적 설명)을 그리려면 이렇게 한다.

```python
def negative_gradcam(gradcam, image_tensor, target_class, device):
    """
    대상 클래스일 확률을 떨어뜨리는 영역을 계산한다.

    모델이 무엇을 대상 클래스가 '아니라고' 여기는지 아는 데 쓸모 있다.
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
```

### 레이어별 Grad-CAM 분석

```python
def analyze_layer_gradcam(model, image_tensor, target_class, device):
    """
    레이어마다의 Grad-CAM을 비교해 피처의 계층 구조를 알아본다.
    """
    # ResNet이면 layer2, layer3, layer4를 비교한다
    layers = {
        'layer2 (중간 레이어)': model.layer2[-1],
        'layer3 (깊은 레이어)': model.layer3[-1],
        'layer4 (의미 레이어)': model.layer4[-1]
    }

    results = {}
    for name, layer in layers.items():
        gradcam = GradCAM(model, layer)
        heatmap = gradcam(image_tensor, target_class, device)
        results[name] = heatmap.cpu().numpy()

    return results
```

---

## 6. Grad-CAM의 성질

### 클래스 판별력

Grad-CAM의 중요한 성질은 **클래스를 구별한다**는 것이다. 타깃 클래스가 달라지면

- 중요도 웨이트 $\alpha^c_k$가 $c$에 따라 바뀐다
- 다른 피처 맵이 지배적이 된다
- 그 결과 히트맵이 다른 영역을 짚는다

그래서 Grad-CAM은 *"바로 이 클래스에는 어느 영역이 중요한가?"*에 답할 수 있다.

### CAM과의 관계

Grad-CAM은 저우 외(2016)의 **클래스 활성화 맵(CAM)**을 일반화한 것이다.

| 항목 | CAM | Grad-CAM |
|--------|-----|----------|
| 구조 | GAP + FC가 있어야 함 | 아무 CNN |
| 재트레인 | GAP이 없는 모델에는 필요 | 필요 없음 |
| 타깃 레이어 | 고정(마지막 컨볼루션 레이어) | 아무 컨볼루션 레이어 |
| 웨이트 계산 | FC 레이어의 웨이트 | 그래디언트 기반 |
| 설명 품질 | 높음 | 높음 |
| 계산 비용 | 낮음 | 높음(역전파) |

**핵심 통찰**: GAP→FC 구조에서는 Grad-CAM과 CAM이 (상수배를 빼면) 같은 결과를 낸다. Grad-CAM은 CAM을 임의의 구조로 일반화한 것이다.

### 해상도의 트레이드오프

Grad-CAM이 **해상도가 낮은 위치 맵**을 내는 까닭은 이렇다.

1. 깊은 레이어의 피처 맵은 공간 해상도가 줄어 있다
   - ResNet-50 layer4: 224×224 입력에 7×7
   - VGG-16 마지막 컨볼루션 레이어: 224×224 입력에 14×14
2. 입력 해상도로 키우면서 보간 아티팩트가 생긴다
3. 의미 정보를 얻는 대신 세밀한 패턴을 잃는다

이는 픽셀 단위 방법과 **서로 보완하는** 성질이다.

- **Grad-CAM**은 *어디*를 보인다(거친 위치 파악)
- **그래디언트 방법**은 *무엇*을 보인다(세밀한 피처 패턴)
- **Guided Grad-CAM**은 둘을 결합한다

---

## 7. 한계

### 1. 낮은 공간 해상도

가장 큰 한계는 해상도다. 깊은 레이어의 피처 맵은 흔히 입력보다 훨씬 작다.

| 구조 | 입력 크기 | 마지막 피처 맵 크기 | 축소 배율 |
|--------------|------------|-------------|-----------|
| ResNet-50 | 224×224 | 7×7 | 32배 |
| VGG-16 | 224×224 | 14×14 | 16배 |
| EfficientNet-B0 | 224×224 | 7×7 | 32배 |

그래서 Grad-CAM은 다음에 맞지 않는다.

- 경계를 정밀하게 찾기
- 픽셀 단위 위치 파악
- 작은 객체 탐지

### 2. 객체가 여럿일 때의 클래스 혼동

이미지에 객체가 여럿이면 Grad-CAM이 지정한 클래스의 영역을 짚기는 하나 엉뚱한 활성화가 함께 보일 수 있다.

```python
# 이미지에 고양이와 개가 함께 있다
cam_cat = grad_cam(input_tensor, target_class=281)  # 고양이 클래스
cam_dog = grad_cam(input_tensor, target_class=235)  # 개 클래스

# CAM을 비교하면 클래스마다 어느 영역에 어텐션을 두는지 드러난다
# 다만 겹치거나 엉뚱한 활성값이 있을 수 있다
```

### 3. 전역 평균 풀링이라는 가정

Grad-CAM은 피처 맵의 중요도가 위치마다 같다고 가정한다. 다음 경우에는 이것이 맞지 않을 수 있다.

- 피처 맵의 위치마다 다른 의미가 실려 있을 때
- 모델이 안에서 어텐션 구조를 쓸 때
- 피처의 중요도가 위치마다 다를 때

### 4. 그래디언트 포화

확신도가 아주 큰 예측에서는 그래디언트가 포화되어 설명이 약해질 수 있다.

```python
# 확신도가 아주 큰 예측(소프트맥스 ≈ 1.0)은 기울기가 0에 가깝다
# 해결책: 소프트맥스 뒤가 아니라 앞의 점수(로짓)를 쓴다
# 이 구현은 이미 그렇게 한다
```

### 5. 적대적 섭동에 취약함

Grad-CAM 설명은 적대적 섭동으로 조작할 수 있다.

- 적대적으로 섭동한 이미지는 엉뚱한 Grad-CAM을 낼 수 있다
- 모델이 사람이 알아챌 수 없는 적대적 피처를 쓸 수 있다
- 실제 판단 근거는 숨긴 채 "그럴듯해 보이는" 영역만 짚을 수 있다

---

## 8. 응용 분야

### 컴퓨터 비전

```python
# 일반적인 이미지 분류 해석
cam = grad_cam(image_tensor, target_class=predicted_class)
```

### 문서 분류(금융)

금융 문서(차트, 표, 텍스트)의 어느 영역이 분류를 좌우하는지 짚는다.

```python
# 문서 이미지 분류기에 쓴다
cam = grad_cam(document_image, target_class=class_map["quarterly_report"])
# 모델이 어느 부분(표, 차트, 서명)에 어텐션을 두는지 시각화한다
```

### 기술적 분석

CNN 기반 거래 모델이 차트의 어느 패턴에 주목하는지 시각화한다.

```python
# 캔들스틱 패턴 인식기에 쓴다
cam = grad_cam(chart_image, target_class=class_map["bullish_engulfing"])
# 모델이 그 패턴을 이루는 올바른 캔들을 보는지 확인한다
```

### 인공위성/대체 데이터

경제 지표를 추정하는 모델이 위성 이미지에서 무엇을 보는지 확인한다.

```python
# 주차장 이미지로 매장 방문객 수를 추정할 때
cam = grad_cam(parking_lot_image, target_class=0)  # 방문객 많음 클래스
# 모델이 그림자 같은 엉뚱한 패턴이 아니라 주차 공간을 보는지 확인한다
```

### 의료 영상

```python
# X선 진단에 쓴다
cam = grad_cam(xray_image, target_class=class_map["pneumonia"])
# 모델이 표시나 라벨이 아니라 폐 영역을 보는지 확인한다
```

---

## 9. 다른 방법과 비교

| 방법 | 해상도 | 클래스 판별력 | 속도 | 이론적 근거 |
|--------|------------|---------------------|-------|-------------------|
| 바닐라 그래디언트 | 높음 | 중간 | 빠름 | 국소 민감도 |
| Grad-CAM | 낮음 | 강함 | 빠름 | 피처 중요도 |
| Guided Backpropagation | 높음 | 약함 | 빠름 | 수정한 그래디언트 |
| Guided Grad-CAM | 높음 | 강함 | 중간 | 결합 |
| Integrated Gradients | 높음 | 중간 | 느림 | 경로 적분 |
| SHAP | 높음 | 강함 | 아주 느림 | 섀플리 값 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 활성화의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 트레인된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. Grad-CAM은 ReLU로 음수 기여를 버리고 맵을 [0, 1]로 다시 스케일하므로 합이 출력 차이와 같을 까닭이 없어 완전성을 만족하지 못한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 방법이 내놓는 설명이 얼마나 믿을 만한지 평가하는 실험을 설계하라. 방법이 중요하다고 짚은 피처가 실제로 모델에 중요한지를 삽입 곡선과 삭제 곡선으로 재어라.

</div>

??? success "풀이"
    절차는 이렇다. (1) 시험 이미지마다 피처 기여도를 계산한다. (2) 삭제: 기여도가 큰 차례로 피처를 하나씩 가리며 모델의 확신도가 떨어지는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 떨어진다. (3) 삽입: 빈 베이스라인에서 시작해 기여도가 큰 차례로 피처를 하나씩 드러내며 확신도가 오르는 모습을 기록한다. 믿을 만한 설명이면 확신도가 빠르게 오른다. (4) 두 곡선 아래 넓이를 계산한다. (5) 무작위 순서(베이스라인)와 다른 방법과 비교한다. 믿을 만한 방법이면 삭제 곡선의 넓이가 작고 삽입 곡선의 넓이가 커야 한다. 통계적으로 믿을 만하려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 설명 방법을 신용 부도를 예측하는 금융 모델에 어떻게 적용할 수 있는지 논하라. 설명이 채워야 할 규제 요건은 무엇인가?

</div>

??? success "풀이"
    신용 모델에는 규제(ECOA, GDPR 22조)가 불리한 결정마다 그 사람에게 맞춘 설명을 요구한다. 방법은 다음을 내놓아야 한다. (1) 거절에 가장 크게 기여한 요인(불리한 처분 사유). (2) 일관된 설명(비슷한 신청자는 비슷한 설명을 받는다). (3) 실행 가능한 설명(신청자가 무엇을 바꾸어야 하는지 안다). 이 절의 설명 방법으로 피처 중요도를 짚을 수 있으나, 강건성(입력이 조금 바뀌었다고 설명이 확 달라지면 안 된다)과 충실성(중요한 피처를 없애면 예측이 바뀌어야 한다)을 평가해야 한다. 보호 속성은 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

Grad-CAM은 CNN의 판단에 대해 해석할 수 있고 클래스를 구별하는 시각화를 준다.

### 핵심 식

**중요도 웨이트:**

$$
\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

**히트맵:**

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

### 핵심 성질

- **클래스를 구별한다**: 클래스가 다르면 히트맵도 다르다
- **구조에 구애받지 않는다**: 어떤 CNN에도 작동한다
- **거친 위치 파악**: "무엇"이 아니라 "어디"를 보인다
- **빠르게 계산한다**: 순전파·역전파 한 번이면 된다
- **재트레인이 필요 없다**: 사전 트레인된 모델에 그대로 쓴다

### 모범 사례

1. 의미 정보가 풍부하고 클래스를 구별하는 설명을 얻으려면 **마지막 컨볼루션 레이어를 타깃으로 삼는다**
2. 그래디언트 포화를 피하려면 **소프트맥스 이전의 점수(로짓)를 쓴다**
3. 클래스를 구별하는지 보려면 **클래스끼리 비교한다**
4. 세밀하게 들여다보려면 **픽셀 단위 방법과 결합한다**(Guided Grad-CAM)
5. **그 분야 전문가와 함께 검토한다** — 짚어 준 영역이 말이 되는지 확인한다
6. 재현할 수 있도록 **타깃 클래스와 레이어를 적는다**

### Grad-CAM을 쓸 때

**권장:**

- CNN의 판단을 영역 단위로 알고 싶을 때
- 오분류를 디버깅할 때
- 모델이 알맞은 이미지 영역을 쓰는지 확인할 때
- 빠르게 시각화할 때(순전파·역전파 한 번)

**다른 방법을 고려할 때:**

- 픽셀 단위까지 위치를 파악해야 할 때 → Guided Grad-CAM
- 이론적 보장이 필요할 때 → Integrated Gradients
- CNN이 아닌 구조일 때 → 어텐션 시각화, SHAP

**참고 문헌**

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

2. Zhou, B., et al. (2016). "Learning Deep Features for Discriminative Localization." *CVPR 2016*. (처음의 CAM 논문)

3. Chattopadhyay, A., et al. (2018). "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks." *WACV 2018*.

4. Adebayo, J., et al. (2018). "Sanity Checks for Saliency Maps." *NeurIPS 2018*.
