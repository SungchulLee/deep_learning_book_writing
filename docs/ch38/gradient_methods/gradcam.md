# Grad-CAM: 기울기 짐 실은 갈래 살아남 그림

**기울기 짐 실은 갈래 살아남 그림(Grad-CAM)**은 CNN 바탕 모형의 판단을 눈에 보이게 풀이하는 재주다. 정해진 얼개를 바라던 앞선 갈래 살아남 그림(CAM) 방법과 달리, Grad-CAM은 얼개를 고치거나 다시 익히지 않고도 어떤 CNN 얼개에나 듣는다.

Grad-CAM은 밑바탕이 되는 물음에 답한다. **들임 그림의 어느 자리가 어떤 갈래를 미루어 보는 데 가장 중요한가?**

그림점 낱의 기울기 방법과 달리 Grad-CAM은 갈래를 가려내는 그림 자리를 짚어 주는 **성긴 자리 그림**을 낸다. 이는 맨 기울기의 두 걸림돌을 다룬다.

1. **갈래 가려냄**: 갈래가 다르면 또렷이 다른 열 그림이 나온다
2. **눈으로 풀이하기**: 매끄럽고 사람이 알아들을 수 있는 자리 짚기를 낸다

---

## 1. 수학 밑바탕

### 문제 세우기

다음을 지닌 CNN 가름개를 생각하자.

- 들임 그림 $I \in \mathbb{R}^{H \times W \times 3}$
- 겨눈 켜(흔히 마지막 겹치는 켜)의 겹침 결 그림 $A^k \in \mathbb{R}^{u \times v}$
- 갈래 $c$의 갈래 점수 $y^c$(소프트맥스에 앞선 값)

갈래 $c$을 미루어 보는 데 중요한 그림 자리를 짚어 주는 열 그림 $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{u \times v}$을 찾는다.

### 고갱이 세움새

Grad-CAM은 겹침 결 그림에 남아 있는 자리 소식을 살려 쓴다. 고갱이 깨침은 **뒤쪽 겹치는 켜에 뜻이 담겨 있고** **기울기가 겨눈 갈래에 대한 중요함을 알린다**는 것이다.

겨눈 갈래 $c$에 대해 Grad-CAM은 이렇게 셈한다.

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

여기서

- $A^k \in \mathbb{R}^{H' \times W'}$은 겨눈 겹치는 켜의 $k$번째 결 그림
- $\alpha^c_k$은 갈래 $c$에 대한 결 그림 $k$의 중요함 짐
- ReLU은 이바지가 양수인 결에만 눈길을 두게 한다

### 중요함 짐 셈하기

중요함 짐 $\alpha^c_k$은 기울기를 **온 세상에 걸쳐 고르게 모아** 셈한다.

$$
\alpha^c_k = \underbrace{\frac{1}{Z} \sum_i \sum_j}_{\text{온 세상 고르게 모으기}} \underbrace{\frac{\partial y^c}{\partial A^k_{ij}}}_{\text{기울기}}
$$

여기서

- $y^c$은 갈래 $c$의 갈래 점수(소프트맥스에 앞선 값)
- $Z = H' \times W'$은 자리의 수
- $\frac{\partial y^c}{\partial A^k_{ij}}$은 자리 $(i, j)$에서 살아남 $A^k$에 대한 갈래 점수의 기울기

이렇게 기울기를 온 세상에 걸쳐 고르게 모으면 결 그림 $k$과 갈래 $c$에 대한 **신경 세포 중요함 짐** $\alpha^c_k$이 나온다.

### 이 꼴의 속뜻

**왜 기울기를 온 세상에 걸쳐 고르게 모으는가?**

기울기 $\frac{\partial y^c}{\partial A^k_{ij}}$은 결 그림 $k$의 어느 자리가 얼마나 중요한지를 알린다. 모든 자리에 걸쳐 고르게 하면 갈래 $c$에 대한 결 그림 $k$의 **두루 보아 중요함**을 얻는다.

**왜 ReLU인가?**

이바지가 음수인(갈래 점수를 떨어뜨리는) 결은 누른다. 우리는 겨눈 갈래일 낌새를 **올리는** 자리만 그리고 싶다. 눌러야 할 결(음수 기울기)은 그 갈래의 미루어 봄을 풀이하는 데 걸리지 않는다.

**왜 마지막 겹치는 켜인가?**

뒤쪽 켜에는 다음이 있다.

- 켜가 높은 뜻 소식
- 갈래를 가려내는 결
- 자리를 짚을 만한 넉넉한 결 고움

### 수학으로 이끌어 내기

Grad-CAM 꼴은 **일차 테일러 펼침**에서 이끌어 낼 수 있다. 첫 이치에서 시작해 갈래 점수 $y^c$을 모든 결 그림의 함수로 보자.

$$
y^c = f(A^1, A^2, \ldots, A^K)
$$

결 그림을 조금 흔들었을 때 갈래 점수가 바뀌는 만큼은 대략 이렇다.

$$
y^c \approx \sum_k \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} A_{ij}^k
$$

온 세상에 걸쳐 고르게 모은다는 여김으로 다시 쓰면

$$
y^c \approx \sum_k \alpha_k^c \sum_i \sum_j A_{ij}^k
$$

이는 $\alpha_k^c$이 결 그림 $k$이 갈래 점수에 얼마나 이바지하는지를 잰다는 뜻이다.

온 세상 고르게 모으기가 자리 소식을 한데 모은다.

$$
\alpha^c_k = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A^k_{ij}}
$$

짐 실은 합이 결 그림을 중요함에 따라 아우른다.

$$
L^c(i,j) = \sum_{k=1}^{K} \alpha^c_k A^k_{ij}
$$

마지막으로 ReLU이 음수 이바지를 걷어낸다.

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}(L^c)
$$

---

## 2. 알고리즘

```
알고리즘: Grad-CAM
들임: 그림 I, CNN 모형 f, 겨눈 갈래 c, 겨눈 켜 l
내놓기: 열 그림 L_GradCAM

1. 앞으로 걸음: 켜 l의 결 그림 A^k을 셈한다
2. 앞으로 걸음: 갈래 점수 y^c을 셈한다
3. 되짚기 걸음: 기울기 ∂y^c/∂A^k을 셈한다
4. 결 그림 k마다:
   α_k^c = 온세상고르게모으기(∂y^c/∂A^k)
5. 짐 실은 아우름을 셈한다:
   L_GradCAM = ReLU(Σ_k α_k^c * A^k)
6. L_GradCAM을 들임 결 고움으로 키운다
7. L_GradCAM을 내놓는다
```

---

## 3. PyTorch 짜보기

### 온전한 GradCAM 클래스

```python
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
```

### 얼개마다 겨눈 켜 집기

```python
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
```

### 온전히 쓰는 보기

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 미리 익힌 ResNet을 부른다
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 마지막 겹치는 켜를 겨눈다
target_layer = model.layer4[-1]

# Grad-CAM의 첫자리를 잡는다
grad_cam = GradCAM(model, target_layer)

# 들임을 채비한다
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

# 미루어 본 갈래에 대한 Grad-CAM을 만든다
cam = grad_cam(input_tensor)
print(f"CAM 꼴: {cam.shape}")  # torch.Size([224, 224])

# 정한 갈래에 대한 Grad-CAM을 만든다(보기로 갈래 281 = 얼룩 고양이)
cam_tabby = grad_cam(input_tensor, target_class=281)

# 그림을 만든다
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
plt.title('본디')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam_tabby.cpu().numpy(), cmap='jet')
plt.title('Grad-CAM 열 그림')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(visualization)
plt.title('겹쳐 보이기')
plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 4. 겨눈 켜 고르기

어느 켜를 겨누느냐가 Grad-CAM 그림을 크게 가른다.

### 마지막 겹치는 켜(권함)

마지막 겹치는 켜가 뜻이 가장 짙은 결을 담는다.

| 결 | 나은 점 | 못한 점 |
|--------|-----|-----|
| 뜻 속살 | 켜가 높은 개념 | - |
| 자리 짚기 | 물체를 잘 덮는다 | 결이 성기다 |
| 갈래 가려냄 | 세다 | - |

```python
# 얼개마다의 겨눈 켜
target_layers = {
    'resnet': model.layer4[-1],
    'vgg': model.features[-1],
    'mobilenet': model.features[-1],
    'densenet': model.features.denseblock4,
    'efficientnet': model.features[-1],
}
```

### 앞선 켜

앞선 켜는 켜가 낮은 결을 담는다.

| 결 | 나은 점 | 못한 점 |
|--------|-----|-----|
| 결 고움 | 더 곱다 | - |
| 결 | 결무늬, 가장자리 | 뜻이 옅다 |
| 풀이하기 | - | 몫 매기기에 잡음이 많다 |

### 여러 켜 Grad-CAM

여러 켜의 Grad-CAM을 아우르면 더 넉넉한 풀이가 나온다.

```python
def multi_layer_gradcam(model, image_tensor, layers, target_class=None, device=None):
    """여러 켜에서 Grad-CAM을 만들어 아우른다."""
    import cv2

    cams = []
    for layer in layers:
        gc = GradCAM(model, layer)
        cam = gc(image_tensor, target_class, device)
        cams.append(cam.cpu().numpy())

    # 모든 CAM을 같은 크기로 맞추고 고르게 한다
    target_size = (224, 224)
    combined = np.zeros(target_size)
    for cam in cams:
        resized = cv2.resize(cam, target_size)
        combined += resized

    combined /= len(cams)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    return combined

# 보기: layer3과 layer4을 아우른다
layers = [model.layer3[-1], model.layer4[-1]]
multi_cam = multi_layer_gradcam(model, input_tensor, layers, target_class=281)
```

---

## 5. 앞선 재주

### 여러 갈래 견주기

Grad-CAM의 고갱이 됨됨이는 **갈래를 가려낸다**는 것이다.

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
```

### 음수 Grad-CAM

갈래일 낌새를 **떨어뜨리는** 결(되돌려 세운 자리)을 그리려면

```python
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
```

### 켜마다 살피는 Grad-CAM

```python
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
```

---

## 6. Grad-CAM의 됨됨이

### 갈래 가려냄

Grad-CAM의 종요로운 됨됨이는 **갈래를 가려낸다**는 것이다. 겨눈 갈래가 달라지면

- 중요함 짐 $\alpha^c_k$이 $c$에 따라 바뀐다
- 다른 결 그림이 판친다
- 그 열매인 열 그림이 다른 자리를 짚는다

그래서 Grad-CAM은 *"바로 이 갈래에는 어느 자리가 중요한가?"*에 답할 수 있다.

### CAM과의 이어짐

Grad-CAM은 저우 외(2016)의 **갈래 살아남 그림(CAM)**을 넓힌 것이다.

| 결 | CAM | Grad-CAM |
|--------|-----|----------|
| 얼개 | GAP + FC이 있어야 함 | 아무 CNN |
| 다시 익히기 | GAP이 없는 모형에는 있어야 함 | 없어도 됨 |
| 겨눈 켜 | 못 박힘(마지막 겹치는 켜) | 아무 겹치는 켜 |
| 짐 셈하기 | FC 켜의 짐 | 기울기 바탕 |
| 풀이 됨됨이 | 높음 | 높음 |
| 셈 값 | 낮음 | 높음(되짚기) |

**고갱이 깨침**: GAP→FC 얼개에서는 Grad-CAM과 CAM이 똑같은 열매를 낸다. Grad-CAM은 CAM을 아무 얼개로나 넓힌 것이다.

### 결 고움의 맞바꿈

Grad-CAM이 **성긴 자리 짚기**를 내는 까닭은 이렇다.

1. 깊은 켜의 결 그림은 결 고움이 줄어 있다
   - ResNet-50 layer4: 224×224 들임에 7×7
   - VGG-16: 224×224 들임에 14×14
2. 들임 결 고움으로 키우면서 사이 잡기 자국이 생긴다
3. 뜻 소식을 얻는 대신 잔 무늬를 내준다

이는 그림점 낱 방법과 **서로 채워 주는** 됨됨이다.

- **Grad-CAM**은 *어디*를 보인다(성긴 자리 짚기)
- **기울기 방법**은 *무엇*을 보인다(결 고운 무늬)
- **이끈 Grad-CAM**은 둘을 아우른다

---

## 7. 한계

### 1. 성긴 자리 결 고움

가장 큰 걸림돌은 결 고움이다. 깊은 켜의 결 그림은 흔히 들임보다 훨씬 작다.

| 얼개 | 들임 크기 | Layer4 크기 | 줄어든 곱 |
|--------------|------------|-------------|-----------|
| ResNet-50 | 224×224 | 7×7 | 32곱절 |
| VGG-16 | 224×224 | 14×14 | 16곱절 |
| EfficientNet-B0 | 224×224 | 7×7 | 32곱절 |

그래서 Grad-CAM은 다음에 맞지 않는다.

- 테두리를 또렷이 찾기
- 잔 낱까지 자리 짚기
- 작은 물체 찾기

### 2. 물체가 여럿일 때의 갈래 헷갈림

그림에 물체가 여럿이면 Grad-CAM이 정한 갈래의 자리를 짚기는 하나 헛된 살아남이 보일 수 있다.

```python
# 그림에 고양이와 개가 함께 있다
cam_cat = grad_cam(input_tensor, target_class=281)  # 고양이 갈래
cam_dog = grad_cam(input_tensor, target_class=235)  # 개 갈래

# CAM을 견주면 갈래마다 어느 자리에 눈길을 두는지 드러난다
# 다만 겹치거나 헛된 살아남이 있을 수 있다
```

### 3. 온 세상 고르게 모으기라는 여김

Grad-CAM은 결 그림의 중요함이 자리마다 한결같다고 여긴다. 다음에서는 이것이 맞지 않을 수 있다.

- 결 그림의 자리마다 다른 뜻이 실려 있을 때
- 모형이 안에서 눈길 얼개를 쓸 때
- 결의 중요함이 자리마다 다를 때

### 4. 기울기 잦아듦

자신함이 아주 큰 미루어 봄에서는 기울기가 잦아들어 풀이가 옅어질 수 있다.

```python
# 자신함이 아주 큰 미루어 봄(소프트맥스 ≈ 1.0)은 기울기가 0에 가깝다
# 푸는 길: 소프트맥스 뒤가 아니라 앞의 점수(로짓)를 쓴다
# 이 짜보기는 이미 그렇게 맡겨 두었다
```

### 5. 겨눈 흔듦에 약함

Grad-CAM 풀이는 겨눈 흔듦으로 주무를 수 있다.

- 겨누어 흔든 그림은 그르치는 Grad-CAM을 낼 수 있다
- 모형이 알아챌 수 없는 겨눈 결을 쓸 수 있다
- 참 판단 결은 숨은 채 "그럴듯해 보이는" 자리만 짚을 수 있다

---

## 8. 쓰이는 자리

### 셈틀 봄

```python
# 여느 그림 가름 풀이
cam = grad_cam(image_tensor, target_class=predicted_class)
```

### 문서 가름(금융)

금융 문서(그림표, 표, 글)의 어느 자리가 가름을 이끄는지 짚는다.

```python
# 문서 그림 가름개에 쓴다
cam = grad_cam(document_image, target_class=class_map["quarterly_report"])
# 모형이 어느 마디(표, 그림표, 서명)에 눈길을 두는지 그린다
```

### 기술 살피기

CNN 바탕 거래 모형이 어느 그림표 결에 눈길을 두는지 그린다.

```python
# 촛대 결 알아보개에 쓴다
cam = grad_cam(chart_image, target_class=class_map["bullish_engulfing"])
# 모형이 그 결을 이루는 옳은 촛대를 보는지 살핀다
```

### 인공위성/대체 자료

경제 잣대를 얻으려는 모형이 인공위성 그림에서 무엇을 보는지 안다.

```python
# 주차장 그림으로 가게 오감을 미루어 볼 때
cam = grad_cam(parking_lot_image, target_class=0)  # 오감 많음 갈래
# 모형이 그림자 같은 엉뚱한 결이 아니라 주차 자리를 보는지 살핀다
```

### 의료 그림

```python
# X선 진단에 쓴다
cam = grad_cam(xray_image, target_class=class_map["pneumonia"])
# 모형이 자국이나 이름표가 아니라 허파 자리를 보는지 살핀다
```

---

## 9. 다른 방법과 견주기

| 방법 | 결 고움 | 갈래 가려냄 | 빠르기 | 이론 바탕 |
|--------|------------|---------------------|-------|-------------------|
| 맨 기울기 | 높음 | 가운데 | 빠름 | 그 자리 예민함 |
| Grad-CAM | 낮음 | 셈 | 빠름 | 결 중요함 |
| 이끈 되짚기 | 높음 | 약함 | 빠름 | 고친 기울기 |
| 이끈 Grad-CAM | 높음 | 셈 | 가운데 | 아우름 |
| 쌓은 기울기 | 높음 | 가운데 | 느림 | 길 적분 |
| SHAP | 높음 | 셈 | 아주 느림 | 섀플리 값 |

---

## 연습문제

**연습문제 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "연습문제 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**연습문제 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "연습문제 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**연습문제 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "연습문제 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**연습문제 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "연습문제 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$

## 정리하며

Grad-CAM은 CNN의 판단에 대해 풀이할 수 있고 갈래를 가려내는 그림을 준다.

### 고갱이 식

**중요함 짐:**

$$
\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

**열 그림:**

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

### 고갱이 됨됨이

- **갈래를 가려낸다**: 갈래가 다르면 열 그림도 다르다
- **얼개를 가리지 않는다**: 어떤 CNN에도 듣는다
- **성긴 자리 짚기**: "무엇"이 아니라 "어디"를 보인다
- **빠르게 셈한다**: 앞으로-되짚기 한 번이면 된다
- **다시 익히지 않는다**: 미리 익힌 모형에 그대로 쓴다

### 잘 쓰는 길

1. 뜻이 짙고 갈래를 가려내는 풀이를 얻으려면 **마지막 겹치는 켜를 겨눈다**
2. 기울기 잦아듦을 피하려면 **소프트맥스 앞의 점수(로짓)를 쓴다**
3. 갈래를 가려내는지 보려면 **갈래끼리 견준다**
4. 촘촘히 들여다보려면 **그림점 낱 방법과 아우른다**(이끈 Grad-CAM)
5. **밭 밝은 이와 함께 따진다** — 짚어 준 자리가 말이 되는지 살핀다
6. 다시 해 볼 수 있도록 **겨눈 갈래와 켜를 적는다**

### Grad-CAM을 쓸 때

**권한다:**

- CNN의 판단을 자리 낱으로 알고 싶을 때
- 잘못 가른 것을 벌레잡을 때
- 모형이 알맞은 그림 자리를 쓰는지 살필 때
- 빠르게 그려 볼 때(앞으로-되짚기 한 번)

**다른 것을 헤아려 볼 때:**

- 잔 낱까지 자리를 짚어야 할 때 → 이끈 Grad-CAM
- 이론 보장이 있어야 할 때 → 쌓은 기울기
- CNN이 아닌 얼개일 때 → 눈길 그림 그리기, SHAP

**살펴볼 거리**

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

2. Zhou, B., et al. (2016). "Learning Deep Features for Discriminative Localization." *CVPR 2016*. (처음의 CAM 논문)

3. Chattopadhyay, A., et al. (2018). "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks." *WACV 2018*.

4. Adebayo, J., et al. (2018). "Sanity Checks for Saliency Maps." *NeurIPS 2018*.
