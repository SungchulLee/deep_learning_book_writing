# 두드러짐 그림과 맨 기울기
## 들머리

**두드러짐 그림**은 신경 그물의 미루어 봄에 들임의 어느 데가 가장 중요한지 드러내는 그림 그리기 재주다. 뿌리를 보면 두드러짐 방법은 밑바탕이 되는 물음에 답한다. *"모형은 이 판단을 내리는 데 어떤 들임 결에 기대는가?"*

가장 단순하고 가장 밑바탕이 되는 길은 **맨 기울기**를 쓰는 것이다. 곧 모형의 내놓기를 들임으로 미분한다. 밑바탕 생각은 곧다. **어떤 그림점 값을 바꾸었을 때 미루어 봄이 크게 달라지면 그 그림점이 중요하다.**

이 장은 기울기 바탕 두드러짐 방법의 수학 밑바탕, 짜기의 자세한 것, 그리고 참으로 쓸 때 헤아릴 것을 들여온다. 단순하고 손에 잡히지만, 맨 기울기를 아는 일은 더 촘촘한 풀이 방법의 개념 밑바탕이 되므로 꼭 있어야 한다.

## 수학 밑바탕

### 뜻매김

가름 함수 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{C}$($n$차원 들임을 $C$개 갈래 점수로 옮긴다)가 주어졌을 때, 갈래 $c$과 들임 $\mathbf{x}$에 대한 두드러짐 그림은 이렇다.

$$
S_c(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

여기서 $|\cdot|$은 원소마다의 절댓값이다.

### 예민함을 재는 자로서의 기울기

기울기 $\frac{\partial f_c}{\partial x_i}$은 들임 결 $x_i$을 아주 조금 바꿀 때 갈래 점수 $f_c$이 얼마나 흔들리는지, 곧 **예민함**을 잰다.

- **기울기 크기가 크다** → 이 결을 조금만 바꿔도 미루어 봄이 크게 달라진다
- **기울기 크기가 작다** → 이 결은 그 자리에서 미루어 봄에 거의 미치지 않는다
- **기울기가 양수** → 이 결을 올리면 갈래 점수가 오른다
- **기울기가 음수** → 이 결을 올리면 갈래 점수가 내린다

### 테일러 펼침으로 읽기

기울기 바탕 두드러짐은 일차 테일러 펼침으로 자연스레 읽을 수 있다. 작은 흔듦 $\boldsymbol{\epsilon}$에 대해

$$
f_c(\mathbf{x} + \boldsymbol{\epsilon}) \approx f_c(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla_{\mathbf{x}} f_c(\mathbf{x})
$$

여기서 다음이 드러난다.

1. 그림점 $i$에서 **기울기 크기가 크다**는 것은 $x_i$을 조금 바꿀 때 $y_c$이 크게 달라진다는 뜻이다
2. **기울기의 방향**은 그림점 값을 올릴 때 갈래 점수가 오르는지 내리는지를 알린다
3. **절댓값 기울기**는 방향과 상관없이 예민함을 담는다
4. 기울기는 내놓기를 가장 크게 바꾸는 **흔듦의 방향**을 알려 준다

### 여러 통로 모으기

꼴이 $(3, H, W)$인 RGB 그림이면 모든 통로에 대해 기울기를 셈하고 모은다.

$$
\mathbf{G} = \frac{\partial y_c}{\partial \mathbf{x}} \in \mathbb{R}^{3 \times H \times W}
$$

흔히 쓰는 모으기 꾀는 이렇다.

| 방법 | 꼴 | 결 |
|--------|---------|-----------------|
| **가장 큼** | $S_{i,j} = \max_{k \in \{R,G,B\}} \|G_{k,i,j}\|$ | 어느 통로에서든 중요한 그림점을 짚는다 |
| **평균** | $S_{i,j} = \frac{1}{3} \sum_{k} \|G_{k,i,j}\|$ | 통로에 걸쳐 중요함을 고르게 한다 |
| **L2 크기** | $S_{i,j} = \sqrt{\sum_{k} G_{k,i,j}^2}$ | 기울기 벡터의 유클리드 크기 |
| **합** | $S_{i,j} = \sum_{k} \|G_{k,i,j}\|$ | 절댓값 예민함의 온 합 |

## PyTorch 짜보기

### 기본 맨 기울기 두드러짐

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None
) -> np.ndarray:
    """
    맨 기울기 두드러짐 그림을 셈한다.

    Args:
        model: 따지는 결로 놓인 신경 그물 모형
        input_tensor: (1, C, H, W) 꼴 들임 텐서
        target_class: 겨눈 갈래 번호(None이면 미루어 본 갈래를 쓴다)
        device: 셈하는 장치

    Returns:
        (H, W) 꼴 numpy 배열 두드러짐 그림
    """
    if device is None:
        device = next(model.parameters()).device

    # 들임에 기울기 셈을 켠다
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)

    # 앞으로 걸음
    model.eval()
    output = model(input_tensor)

    # 겨눈 갈래를 정한다
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 되짚기 걸음: ∂y_c/∂x을 셈한다
    model.zero_grad()
    output[0, target_class].backward()

    # 들임에 대한 기울기를 집는다
    saliency = input_tensor.grad.data.abs()

    # 빛깔 통로를 가로질러 가장 큰 것을 잡는다
    saliency, _ = saliency.max(dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    return saliency


def compute_vanilla_gradient_saliency(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> torch.Tensor:
    """
    맨 기울기 두드러짐 그림을 셈한다(텐서를 내놓는다).

    Args:
        model: 따지는 결로 놓인 미리 익힌 신경 그물
        image_tensor: 들임 그림 [1, 3, H, W]
        target_class: 두드러짐을 셈할 갈래 번호
        device: 셈하는 장치(CPU/GPU)

    Returns:
        두드러짐 그림 텐서 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 기울기가 들임까지 흐르게 한다
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    # 앞으로 걸음
    output = model(image_tensor)  # [1, num_classes]

    # 겨눈 갈래 점수를 고른다
    target_score = output[0, target_class]

    # 되짚기 걸음
    target_score.backward()

    # 기울기를 집고 절댓값을 잡는다
    gradients = image_tensor.grad  # [1, 3, H, W]
    abs_gradients = torch.abs(gradients)

    # 빛깔 통로를 가로질러 모은다(가장 큰 것 고르기)
    saliency = torch.max(abs_gradients, dim=1)[0]  # [1, H, W]

    return saliency
```

### 부호 있는 두드러짐(양수와 음수 이바지)

```python
def compute_signed_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> tuple:
    """
    양수와 음수 이바지를 보이는 부호 있는 두드러짐을 셈한다.

    Returns:
        positive_saliency: 올리면 갈래 점수를 올리는 결
        negative_saliency: 올리면 갈래 점수를 내리는 결
    """
    input_tensor = input_tensor.clone().requires_grad_(True)

    model.eval()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    gradient = input_tensor.grad.data

    # 양수 기울기와 음수 기울기를 가른다
    positive = gradient.clamp(min=0)
    negative = gradient.clamp(max=0).abs()

    # 통로를 가로질러 가장 큰 것
    positive_saliency = positive.max(dim=1)[0].squeeze().cpu().numpy()
    negative_saliency = negative.max(dim=1)[0].squeeze().cpu().numpy()

    return positive_saliency, negative_saliency
```

### 기울기 × 들임

기울기에 들임 값을 곱하면 **있으면서** **중요한** 결을 짚어 몫 매기기를 또렷하게 할 수 있다.

$$
\text{기울기} \times \text{들임} = \mathbf{x} \odot \frac{\partial f_c}{\partial \mathbf{x}}
$$

```python
def gradient_times_input(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    기울기 × 들임 두드러짐을 셈한다.

    기울기에 들임 값으로 짐을 실어, 있으면서 중요한 결이
    무엇인지 보인다.
    """
    input_tensor = input_tensor.clone().requires_grad_(True)

    model.eval()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    # 기울기 × 들임
    grad_input = input_tensor.grad.data * input_tensor.data

    # 절댓값을 잡고 통로를 가로질러 가장 큰 것
    saliency = grad_input.abs().max(dim=1)[0].squeeze().cpu().numpy()

    return saliency
```

### 모으기 갈래 모두

```python
def compute_all_aggregations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> dict:
    """기울기 모으기 방법을 견준다."""
    model.eval()
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)

    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    gradients = input_tensor.grad
    abs_gradients = torch.abs(gradients)

    return {
        'max': torch.max(abs_gradients, dim=1)[0].squeeze().cpu().numpy(),
        'mean': torch.mean(abs_gradients, dim=1).squeeze().cpu().numpy(),
        'l2': torch.sqrt(torch.sum(abs_gradients ** 2, dim=1)).squeeze().cpu().numpy(),
        'sum': torch.sum(abs_gradients, dim=1).squeeze().cpu().numpy(),
        'squared': (gradients ** 2).max(dim=1)[0].squeeze().cpu().numpy(),
        'positive_only': gradients.clamp(min=0).max(dim=1)[0].squeeze().cpu().numpy(),
    }
```

## 온전히 도는 보기

```python
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 차림
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 미리 익힌 모형을 부른다
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.eval()

# 미리 다듬기(ImageNet으로 고르게 하기)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 그림을 부르고 미리 다듬는다
image = Image.open('dog.jpg').convert('RGB')
image_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
image_tensor.requires_grad = True

# 모형의 미루어 봄을 얻는다
with torch.no_grad():
    output = model(image_tensor.to(device))
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()

print(f"미루어 본 갈래: {pred_class}, 자신함: {confidence:.2%}")

# 여러 두드러짐 그림을 셈한다
saliency = compute_saliency_map(model, image_tensor, pred_class, device)
pos_saliency, neg_saliency = compute_signed_saliency(model, image_tensor, pred_class)
grad_input = gradient_times_input(model, image_tensor, pred_class)

# 그리려고 본디 그림을 채비한다
original = np.array(image.resize((224, 224))) / 255.0

print(f"두드러짐 꼴: {saliency.shape}")
print(f"값 너비: [{saliency.min():.6f}, {saliency.max():.6f}]")
```

## 그림으로 보이기

### 제대로 고르게 하기

뜻있게 그리려면 두드러짐 그림을 $[0, 1]$으로 고르게 해야 한다.

```python
def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    그리려고 두드러짐 그림을 [0, 1]으로 고르게 한다.
    """
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.detach().cpu().numpy()

    if saliency.ndim == 3:
        saliency = saliency.squeeze(0)

    s_min, s_max = saliency.min(), saliency.max()

    if s_max - s_min > 1e-10:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency
```

### 여느 그림 그리기

```python
def visualize_saliency(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    title: str = "두드러짐 그림"
) -> plt.Figure:
    """
    본디 그림과 나란히 두드러짐 그림을 그린다.

    Args:
        original_image: [0, 1] 너비의 본디 그림 (H, W, 3)
        saliency_map: 두드러짐 값 (H, W)
        title: 그림 이름

    Returns:
        Matplotlib 그림
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 본디 그림
    axes[0].imshow(original_image)
    axes[0].set_title('본디 그림')
    axes[0].axis('off')

    # 두드러짐 그림
    saliency_norm = normalize_saliency(saliency_map)
    im = axes[1].imshow(saliency_norm, cmap='hot')
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 겹쳐 보이기
    if original_image.ndim == 3:
        gray_img = np.mean(original_image, axis=2)
    else:
        gray_img = original_image

    axes[2].imshow(gray_img, cmap='gray')
    axes[2].imshow(saliency_norm, cmap='jet', alpha=0.5)
    axes[2].set_title('겹쳐 보이기')
    axes[2].axis('off')

    plt.tight_layout()
    return fig
```

### 부호 있는 두드러짐 그리기

```python
def visualize_signed_saliency(
    original_image: np.ndarray,
    positive_saliency: np.ndarray,
    negative_saliency: np.ndarray
) -> plt.Figure:
    """
    양수 두드러짐과 음수 두드러짐을 따로 그린다.

    - 빨강: 올리면 갈래 점수를 올리는 결
    - 파랑: 올리면 갈래 점수를 내리는 결
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('본디')
    axes[0].axis('off')

    axes[1].imshow(positive_saliency, cmap='Reds')
    axes[1].set_title('양수 (점수를 올림)')
    axes[1].axis('off')

    axes[2].imshow(negative_saliency, cmap='Blues')
    axes[2].set_title('음수 (점수를 내림)')
    axes[2].axis('off')

    # 아우른 봄: 빨강 = 양수, 파랑 = 음수
    combined = np.zeros((*positive_saliency.shape, 3))
    combined[:, :, 0] = positive_saliency / (positive_saliency.max() + 1e-8)
    combined[:, :, 2] = negative_saliency / (negative_saliency.max() + 1e-8)

    axes[3].imshow(combined)
    axes[3].set_title('아우름 (빨강+, 파랑-)')
    axes[3].axis('off')

    plt.tight_layout()
    return fig
```

### 여러 갈래 견주기

고갱이 깨침은 두드러짐 그림이 **갈래마다 다르다**는 것이다. 겨눈 갈래가 다르면 그림의 다른 자리를 짚을 수 있다.

```python
def compare_class_saliencies(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list = None,
    device: torch.device = None
) -> dict:
    """
    여러 겨눈 갈래의 두드러짐 그림을 셈해 견준다.

    갈래가 다르면 다른 자리를 짚는다는 것을 보인다.
    """
    if device is None:
        device = next(model.parameters()).device

    saliencies = {}

    for i, class_idx in enumerate(class_indices):
        # 되짚기마다 새 텐서를 만든다
        img_copy = image_tensor.clone().detach().requires_grad_(True)

        saliency = compute_saliency_map(model, img_copy, class_idx, device)

        name = class_names[i] if class_names else f"갈래 {class_idx}"
        saliencies[name] = saliency

    return saliencies


# 보기: 앞선 3개 미루어 본 갈래를 견준다
with torch.no_grad():
    output = model(image_tensor.to(device))
    top_classes = output[0].topk(3).indices.tolist()

saliencies = compare_class_saliencies(model, image_tensor, top_classes, device=device)
```

## 셈속으로 살피기

두드러짐 값의 퍼짐을 알면 진단에 도움이 된다.

```python
def analyze_saliency_statistics(saliency: np.ndarray) -> dict:
    """두드러짐 퍼짐에 대한 셈속을 두루 셈한다."""
    s = saliency.flatten()

    stats = {
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'median': np.median(s),
        'p75': np.percentile(s, 75),
        'p90': np.percentile(s, 90),
        'p95': np.percentile(s, 95),
        'p99': np.percentile(s, 99),
        'sparsity': (s < s.mean()).mean(),  # 평균 아래인 몫
        'max_mean_ratio': s.max() / (s.mean() + 1e-8)
    }

    return stats
```

**읽는 길잡이:**

| 자 | 읽는 법 |
|--------|----------------|
| 성김이 큼(80% 넘음) | 몇몇 그림점이 판친다. 자리를 뜻있게 짚었을 낌새 |
| 가장 큼/평균 비가 작음 | 중요함이 퍼져 있고 모형이 온 세상 소식을 쓴다 |
| 흩어짐이 큼 | 자리를 세게 짚고 바탕을 누른다 |
| p99/p95 사이가 큼 | 끝자락 값이 몇 개 있다. 자국이 아닌지 살펴라 |

## 한계와 어려움

### 1. 눈에 띄는 잡음

맨 기울기 두드러짐 그림은 **잡음이 많기로** 이름났다. 까닭은 이렇다.

1. **높은 잦기에 예민함**: 신경 그물은 립시츠 상수가 커서 들임이 조금만 달라져도 기울기가 흔들린다
2. **잦아듦**: ReLU 살림이 기울기 터에 끊긴 데를 만든다
3. **매끄럽지 않음**: 기울기는 그 자리의 예민함을 담을 뿐 온 세상 중요함을 담지 않는다

```python
def demonstrate_noise():
    """아무 들임에도 두드러짐 그림에 잡음이 낄 수 있음을 보인다."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # 아무 들임
    input_tensor = torch.randn(1, 3, 224, 224)

    saliency = compute_saliency_map(model, input_tensor)

    # 아무 들임에도 두드러짐이 흩어져 보이는 일이 잦다
    print(f"0이 아닌 그림점: {(saliency > 0.01 * saliency.max()).sum()}")
    print(f"이는 기울기 바탕 두드러짐에 잡음이 있음을 알린다")
```

### 2. 기울기 잦아듦

잦아드는 비선형(시그모이드, tanh)이나 ReLU 그물에서는

$$
\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

살아남이 "죽은" 자리에 있으면 들임이 아무리 중요해도 기울기가 0이 된다. 자신함이 아주 큰 미루어 봄에서는 소프트맥스 기울기도 아주 작아진다.

**푸는 길:** 기울기를 셈할 때 소프트맥스 뒤의 낌새가 아니라 날 로짓(소프트맥스 앞의 점수)을 쓴다.

### 3. 갈래를 잘 가려내지 못함

기울기를 정한 갈래에 대해 셈하는데도, 맨 두드러짐은 갈래가 달라도 비슷한 자리를 짚는 일이 잦다.

```python
def class_discrimination_test(model, input_tensor, device):
    """갈래마다의 두드러짐 그림을 견준다."""
    with torch.no_grad():
        output = model(input_tensor.to(device))
        _, top_classes = output.topk(5)

    saliencies = {}
    for cls in top_classes[0]:
        saliencies[cls.item()] = compute_saliency_map(
            model, input_tensor, cls.item(), device
        )

    # 앞선 켜의 기울기를 모든 갈래가 함께 쓰므로
    # 갈래마다의 두드러짐이 꽤 비슷해 보일 때가 많다
    return saliencies
```

### 4. 들임 흔듦에 예민함

들임을 조금만 흔들어도 두드러짐 그림이 크게 달라질 수 있다.

```python
def sensitivity_test(model, input_tensor, device, noise_scale=0.01):
    """두드러짐이 들임 잡음에 얼마나 예민한지 시험한다."""
    saliency_original = compute_saliency_map(model, input_tensor, device=device)

    # 작은 잡음을 더한다
    noisy_input = input_tensor + noise_scale * torch.randn_like(input_tensor)
    saliency_noisy = compute_saliency_map(model, noisy_input, device=device)

    # 차이를 셈한다
    diff = np.abs(saliency_original - saliency_noisy)

    print(f"고른 절댓값 차이: {diff.mean():.4f}")
    print(f"가장 큰 차이: {diff.max():.4f}")
    print(f"얽힘: {np.corrcoef(saliency_original.flatten(), saliency_noisy.flatten())[0,1]:.4f}")

    return diff
```

## SmoothGrad: 잡음 줄이기

SmoothGrad은 들임에 잡음을 섞은 것들에 걸쳐 기울기를 고르게 해 잡음 문제를 다룬다.

$$
\hat{S}_c(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial f_c(\mathbf{x} + \mathcal{N}(0, \sigma^2))}{\partial \mathbf{x}} \right|
$$

```python
def smoothgrad(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None,
    n_samples: int = 50,
    noise_level: float = 0.1,
    device: torch.device = None
) -> np.ndarray:
    """
    SmoothGrad 두드러짐 그림을 셈한다.

    Args:
        model: 신경 그물
        input_tensor: 들임 텐서
        target_class: 겨눈 갈래
        n_samples: 잡음 표본의 수
        noise_level: 잡음의 잣대 벗어남(들임 너비에 대한 몫)
        device: 셈하는 장치

    Returns:
        매끄럽게 한 두드러짐 그림
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_tensor = input_tensor.to(device)

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    # 잡음 잣대를 잡으려고 들임 셈속을 셈한다
    stdev = noise_level * (input_tensor.max() - input_tensor.min())

    accumulated_grad = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        # 가우스 잡음을 더한다
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = (input_tensor + noise).requires_grad_(True)

        # 앞으로와 되짚기
        output = model(noisy_input)
        model.zero_grad()
        output[0, target_class].backward()

        # 기울기를 쌓는다
        accumulated_grad += noisy_input.grad.data.abs()

    # 고르게 한다
    smoothed_grad = accumulated_grad / n_samples

    # 통로를 가로질러 가장 큰 것
    saliency = smoothed_grad.max(dim=1)[0].squeeze().cpu().numpy()

    return saliency
```

## 금융에 쓰기

### 때 열 두드러짐

차례로 놓인 금융 자료에서는 두드러짐이 어느 때 걸음이 중요한지 드러낸다.

```python
def time_series_saliency(
    model: nn.Module,
    sequence: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    때 열 미루어 봄의 두드러짐을 셈한다.

    지난 어느 때 자리가 미루어 봄에 가장 크게 미치는지 보인다.

    Args:
        model: 열 모형(LSTM, 변환기 등)
        sequence: 들임 열 [1, 열 길이, 결]이나 [1, 열 길이]
        target_class: 가름 모형의 겨눈 갈래

    Returns:
        때 축에 걸친 두드러짐
    """
    sequence = sequence.clone().requires_grad_(True)

    model.eval()
    output = model(sequence)

    if target_class is not None:
        output = output[0, target_class]
    else:
        output = output.squeeze()

    model.zero_grad()
    output.backward()

    # 때 축에 걸친 두드러짐
    saliency = sequence.grad.abs().squeeze().cpu().numpy()

    # 결이 여럿이면 모은다
    if saliency.ndim == 2:
        saliency = saliency.sum(axis=1)  # 결 축으로 더한다

    return saliency
```

### 표 자료의 결 중요함

표로 된 금융 자료(신용 무릅씀, 거래 신호 등)에는

```python
def tabular_saliency(
    model: nn.Module,
    features: torch.Tensor,
    feature_names: list,
    target_class: int = None
) -> np.ndarray:
    """
    기울기로 결 중요함을 셈하고 그린다.

    Args:
        model: 가름이나 되돌이 모형
        features: 결 텐서 [1, n_features]
        feature_names: 결 이름 목록
        target_class: 겨눈 갈래(가름일 때)

    Returns:
        결마다의 중요함 점수
    """
    features = features.clone().requires_grad_(True)

    model.eval()
    output = model(features)

    if target_class is not None:
        output = output[0, target_class]
    else:
        output = output.squeeze()

    model.zero_grad()
    output.backward()

    importance = features.grad.abs().squeeze().cpu().numpy()

    # 중요함으로 줄 세운다
    sorted_idx = np.argsort(importance)[::-1]

    print("결 중요함(기울기 크기로):")
    print("-" * 50)
    for i in sorted_idx[:10]:
        print(f"{feature_names[i]:30s}: {importance[i]:.6f}")

    return importance
```

## 다른 방법과의 이어짐

맨 기울기는 더 촘촘한 방법의 밑바탕이 된다.

| 방법 | 고침 | 다루는 것 |
|--------|--------------|-----------|
| **기울기 × 들임** | 들임 값을 곱한다 | 결을 또렷하게 한다 |
| **SmoothGrad** | 잡음 표본에 걸쳐 고르게 한다 | 잡음을 줄인다 |
| **쌓은 기울기** | 밑금에서 오는 길을 따라 적분한다 | 공리를 채운다 |
| **이끈 되짚기** | ReLU 되짚기 걸음을 고친다 | 그림이 깨끗해진다 |
| **Grad-CAM** | 결 그림의 기울기를 쓴다 | 갈래를 가려낸다 |

### 견줌 표

| 방법 | 나은 점 | 못한 점 |
|--------|------|------|
| 맨 기울기 | 단순하고 빠르며 밑바탕이 된다 | 잡음이 많고 갈래를 잘 못 가려낸다 |
| 기울기 × 들임 | 결이 더 또렷하다 | 여전히 잡음이 많다 |
| SmoothGrad | 잡음이 준다 | 셈이 비싸다 |
| Grad-CAM | 깨끗하고 갈래를 가려낸다 | 결이 성기다 |
| 쌓은 기울기 | 이론 보장이 있다 | 느리고 밑금에 매인다 |

## 참으로 쓸 때 이르는 말

### 맨 기울기를 쓸 때

**알맞은 자리:**

- 빠른 벌레잡기와 제정신인지 살피기
- 기울기 흐름 알아보기
- 다른 방법과 견줄 밑금
- 가르치기 몫
- 처음 둘러보기

**권하지 않는 자리:**

- 논문에 실을 그림(잡음이 너무 많다)
- 서비스에서의 풀이하기(더 매끄러운 방법을 쓰라)
- 걸린 것이 큰 판단(다른 증거와 아울러 쓰라)
- 갈래를 가려내는 풀이(Grad-CAM을 쓰라)

### 짜기 살핌 목록

1. ✅ **모형을 따지는 결로**: `model.eval()`으로 한결같이 돌게 한다
2. ✅ **들임에 기울기를 켠다**: `image_tensor.requires_grad = True`
3. ✅ **남은 기울기를 지운다**: 되짚기 앞에 `model.zero_grad()`
4. ✅ **알맞은 갈래를 쓴다**: 미루어 본 갈래나 눈여겨보는 갈래
5. ✅ **소프트맥스가 아니라 로짓을 쓴다**: 기울기 잦아듦을 피한다
6. ✅ **그리려고 고르게 한다**: $[0, 1]$으로 잣대를 잡는다
7. ✅ **텐서를 베낀다**: 견줄 때는 되짚기마다 새 텐서를 쓴다

## 간추림

맨 기울기 두드러짐은 기울기 바탕 풀이 방법을 알아 가는 개념 밑바탕을 준다. 단순하고 손에 잡히지만, 눈에 띄는 잡음과 갈래를 가려내지 못하는 탓에 참으로 쓸모는 좁다. 뒤따르는 방법들이 이 밑바탕 위에서 더 깨끗하고 뜻있는 몫 매기기를 낸다.

### 고갱이 식

**기본 두드러짐:**

$$
S(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**모은 두드러짐(여러 통로 들임):**

$$
S_{i,j} = \max_{k} |G_{k,i,j}| \quad \text{또는} \quad \sqrt{\sum_k G_{k,i,j}^2}
$$

**기울기 × 들임:**

$$
S(\mathbf{x}) = \left| \mathbf{x} \odot \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**SmoothGrad:**

$$
\hat{S}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial f_c(\mathbf{x} + \epsilon_i)}{\partial \mathbf{x}} \right|, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

### 고갱이

1. **기울기는 그 자리의 예민함을 잰다**. 온 세상 중요함이 아니다
2. 그물이 매끄럽지 않으므로 **잡음이 따라붙는다**
3. 앞선 켜를 함께 쓰므로 **갈래를 가려내는 힘이 약하다**
4. **모으는 길이 여럿** 있고 가장 큰 것 고르기가 가장 흔하다
5. 쌓은 기울기, SmoothGrad 같은 **앞선 방법의 밑바탕이 된다**

## 살펴볼 거리

1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." *ICLR Workshop*.

2. Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). "SmoothGrad: removing noise by adding noise." *ICML Workshop*.

3. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

4. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). "Sanity Checks for Saliency Maps." *NeurIPS*.

5. Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., & Müller, K. R. (2010). "How to Explain Individual Classification Decisions." *Journal of Machine Learning Research*.

## 익힘 문제

**익힘 1.**
이 마디에서 밝힌 풀이 방법을, XOR 들임을 가르는 ReLU 살림의 두 켜 신경 그물에 걸어라. 들임 $x = [1, 1]$에 대한 풀이를 셈하여라.

??? success "익힘 1 풀이"
    짐이 $W_1, b_1, W_2, b_2$인 익힌 XOR 그물에서 내놓기는 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 풀이 방법은 들임 결마다 몫을 내놓는다. $x = [1, 1]$(갈래 0)이면 두 결 모두 음수 가름에 이바지한다. 몫 값은 방법마다 다르다. 기울기 바탕 방법은 $\partial f / \partial x_i$을 셈하고, 흔들어 보는 방법은 결을 가렸을 때 내놓기가 얼마나 바뀌는지 잰다. XOR 문제는 판단 금이 선형이 아니므로 선형 풀이 방법이 그르칠 수 있음을 보여 준다. $\square$

---

**익힘 2.**
이 마디의 풀이 방법이 온전함 공리를 채우는지, 곧 어떤 밑금 $x_0$에 대해 모든 결 몫의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 뒤집어라.

??? success "익힘 2 풀이"
    온전함 공리(섀플리 값 이론에서는 효율이라고도 한다)는 몫의 합이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이와 같다는 것이다. 이 방법이 온전함을 채우는지는 그 세움새에 달렸다. 기울기 방법은 온전함을 채우지 못한다(기울기는 그 자리의 것이고 길을 따라 쌓은 것이 아니다). 쌓은 기울기는 세움새 자체로 온전함을 채운다(길을 따라 미적분의 밑정리를 쓴다). SHAP 값은 섀플리 공리로 효율을 채운다. 온전함을 어기는 방법은 몫을 너무 많거나 적게 매길 수 있어, 온 몫을 온 세상 풀이로 믿기 어렵게 만든다. $\square$

---

**익힘 3.**
이 방법이 내놓는 풀이가 얼마나 미더운지 따지는 시험을 꾸며라. 짚어 준 결이 참으로 모형에 중요한지를 넣기와 빼기 곡선으로 재어라.

??? success "익힘 3 풀이"
    절차는 이렇다. (1) 시험 그림마다 결 몫을 셈한다. (2) 빼기: 몫이 큰 차례로 결을 하나씩 가리며 모형의 자신함이 떨어지는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 떨어진다. (3) 넣기: 빈 밑금에서 시작해 몫이 큰 차례로 결을 하나씩 드러내며 자신함이 오르는 모습을 적는다. 미더운 풀이면 자신함이 빠르게 오른다. (4) 두 곡선의 아래 넓이를 셈한다. (5) 아무렇게나 매긴 차례(밑금)와 다른 방법에 견준다. 미더운 방법이면 빼기 넓이가 작고 넣기 넓이가 커야 한다. 통계로 미더우려면 시험 표본 1000개 넘게 되풀이한다. $\square$

---

**익힘 4.**
이 풀이 방법을 신용 부도를 미루어 보는 금융 모형에 어떻게 걸 수 있는지 다루어라. 풀이가 채워야 할 규정 요건은 무엇인가?

??? success "익힘 4 풀이"
    신용 모형에는 규정(ECOA, GDPR 22조)이 불리한 판단마다 그 사람에게 맞춘 풀이를 바란다. 방법은 다음을 내놓아야 한다. (1) 물리침에 가장 크게 이바지한 인자(불리한 처분 까닭). (2) 한결같은 풀이(비슷한 신청자는 비슷한 풀이를 받는다). (3) 손에 잡히는 풀이(신청자가 무엇을 바꾸어야 하는지 안다). 이 마디의 풀이 방법으로 결의 중요함을 짚을 수 있으나, 든든함(들임이 조금 바뀌었다고 풀이가 확 달라지면 안 된다)과 옳음(중요한 결을 없애면 미루어 봄이 바뀌어야 한다)을 따져 보아야 한다. 지켜야 할 됨됨이는 대리 차별이 드러나지 않도록 조심히 다루어야 한다. $\square$
