# Saliency Map과 기본 그래디언트

**Saliency Map**은 뉴럴 네트워크의 예측에 입력의 어느 부분이 가장 중요한지 드러내는 시각화 기법이다. 근본적으로 샐리언시(saliency) 방법은 다음 질문에 답한다. *"모델은 이 판단을 내리는 데 어떤 입력 피처에 기대는가?"*

가장 단순하고 기본적인 방법은 **바닐라 그래디언트**를 쓰는 것이다. 곧 모델의 출력을 입력으로 미분한다. 기본 아이디어는 단순하다. **어떤 픽셀 값을 바꾸었을 때 예측이 크게 달라지면 그 픽셀이 중요하다.**

이 쪽은 그래디언트 기반 샐리언시 방법의 수학적 기초, 구현의 세부, 그리고 실제로 쓸 때 고려할 점을 다룬다. 바닐라 그래디언트는 단순하고 직관적이지만, 더 정교한 설명 방법의 개념적 기초가 되므로 꼭 알아 두어야 한다.

---

## 1. 수학 기초

### 정의

분류 함수 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{C}$($n$차원 입력을 $C$개 클래스 점수로 보낸다)가 주어졌을 때, 클래스 $c$와 입력 $\mathbf{x}$에 대한 Saliency Map은 이렇다.

$$
S_c(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

여기서 $|\cdot|$은 원소마다의 절댓값이다.

### 민감도를 재는 척도로서의 그래디언트

그래디언트 $\frac{\partial f_c}{\partial x_i}$는 입력 피처 $x_i$를 아주 조금 바꿀 때 클래스 점수 $f_c$가 얼마나 변하는지, 곧 **민감도**를 잰다.

- **그래디언트 크기가 크다** → 이 피처를 조금만 바꿔도 예측이 크게 달라진다
- **그래디언트 크기가 작다** → 이 피처는 현재 입력 근처에서 예측에 거의 영향을 주지 않는다
- **그래디언트가 양수** → 이 피처를 올리면 클래스 점수가 오른다
- **그래디언트가 음수** → 이 피처를 올리면 클래스 점수가 내린다

### 테일러 전개로 읽기

그래디언트 기반 샐리언시는 일차 테일러 전개로 자연스레 읽을 수 있다. 작은 섭동 $\boldsymbol{\epsilon}$에 대해

$$
f_c(\mathbf{x} + \boldsymbol{\epsilon}) \approx f_c(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla_{\mathbf{x}} f_c(\mathbf{x})
$$

여기서 다음이 드러난다.

1. 픽셀 $i$에서 **그래디언트 크기가 크다**는 것은 $x_i$를 조금 바꿀 때 클래스 점수 $y_c = f_c(\mathbf{x})$가 크게 달라진다는 뜻이다
2. **그래디언트의 방향**은 픽셀 값을 올릴 때 클래스 점수가 오르는지 내리는지를 알린다
3. **절댓값 그래디언트**는 방향과 상관없이 민감도를 담는다
4. 그래디언트는 출력을 가장 크게 바꾸는 **섭동의 방향**을 알려 준다

### 여러 채널 집계

형태가 $(3, H, W)$인 RGB 이미지이면 모든 채널에 대해 그래디언트를 계산한 뒤 채널 방향으로 집계한다.

$$
\mathbf{G} = \frac{\partial y_c}{\partial \mathbf{x}} \in \mathbb{R}^{3 \times H \times W}
$$

흔히 쓰는 집계 전략은 이렇다.

| 방법 | 식 | 특징 |
|--------|---------|-----------------|
| **최댓값** | $S_{i,j} = \max_{k \in \{R,G,B\}} \|G_{k,i,j}\|$ | 어느 채널에서든 중요한 픽셀을 짚는다 |
| **평균** | $S_{i,j} = \frac{1}{3} \sum_{k} \|G_{k,i,j}\|$ | 채널에 걸친 중요도를 평균한다 |
| **L2 크기** | $S_{i,j} = \sqrt{\sum_{k} G_{k,i,j}^2}$ | 그래디언트 벡터의 유클리드 크기 |
| **합** | $S_{i,j} = \sum_{k} \|G_{k,i,j}\|$ | 절댓값 민감도의 총합 |

---

## 2. PyTorch 구현

### 기본 바닐라 그래디언트 샐리언시

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
    바닐라 그래디언트 Saliency Map을 계산한다.

    Args:
        model: 평가 모드로 둔 뉴럴 네트워크 모델
        input_tensor: (1, C, H, W) 형태 입력 텐서
        target_class: 대상 클래스 번호(None이면 예측 클래스를 쓴다)
        device: 계산할 장치

    Returns:
        (H, W) 형태 numpy 배열 Saliency Map
    """
    if device is None:
        device = next(model.parameters()).device

    # 입력에 그래디언트 계산을 켠다
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)

    # 순전파
    model.eval()
    output = model(input_tensor)

    # 대상 클래스를 정한다
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 역전파: ∂y_c/∂x를 계산한다
    model.zero_grad()
    output[0, target_class].backward()

    # 입력에 대한 그래디언트를 꺼낸다
    saliency = input_tensor.grad.data.abs()

    # 색 채널 방향으로 최댓값을 취한다
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
    바닐라 그래디언트 Saliency Map을 계산한다(텐서를 반환한다).

    Args:
        model: 평가 모드로 둔 사전 트레인된 뉴럴 네트워크
        image_tensor: 입력 이미지 [1, 3, H, W]
        target_class: 샐리언시를 계산할 클래스 번호
        device: 계산할 장치(CPU/GPU)

    Returns:
        Saliency Map 텐서 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 그래디언트가 입력까지 흐르게 한다
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    # 순전파
    output = model(image_tensor)  # [1, num_classes]

    # 대상 클래스 점수를 고른다
    target_score = output[0, target_class]

    # 역전파
    target_score.backward()

    # 그래디언트를 꺼내 절댓값을 취한다
    gradients = image_tensor.grad  # [1, 3, H, W]
    abs_gradients = torch.abs(gradients)

    # 색 채널 방향으로 집계한다(최댓값)
    saliency = torch.max(abs_gradients, dim=1)[0]  # [1, H, W]

    return saliency
```

### 부호 있는 샐리언시(양수와 음수 기여)

```python
def compute_signed_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> tuple:
    """
    양수와 음수 기여를 보이는 부호 있는 샐리언시를 계산한다.

    Returns:
        positive_saliency: 올리면 클래스 점수를 올리는 피처
        negative_saliency: 올리면 클래스 점수를 내리는 피처
    """
    input_tensor = input_tensor.clone().requires_grad_(True)

    model.eval()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    gradient = input_tensor.grad.data

    # 양수 그래디언트와 음수 그래디언트를 나눈다
    positive = gradient.clamp(min=0)
    negative = gradient.clamp(max=0).abs()

    # 채널 방향 최댓값
    positive_saliency = positive.max(dim=1)[0].squeeze().cpu().numpy()
    negative_saliency = negative.max(dim=1)[0].squeeze().cpu().numpy()

    return positive_saliency, negative_saliency
```

### 그래디언트 × 입력

그래디언트에 입력 값을 곱하면 입력에 실제로 **값이 있으면서** 출력에 **민감한** 피처를 짚어 기여도 분석을 또렷하게 할 수 있다. 값이 0인 피처는 그래디언트가 아무리 커도 기여도가 0이 된다.

$$
\text{그래디언트} \times \text{입력} = \mathbf{x} \odot \frac{\partial f_c}{\partial \mathbf{x}}
$$

```python
def gradient_times_input(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    그래디언트 × 입력 샐리언시를 계산한다.

    그래디언트에 입력 값을 웨이트로 곱해, 값이 있으면서 중요한 피처가
    무엇인지 보인다.
    """
    input_tensor = input_tensor.clone().requires_grad_(True)

    model.eval()
    output = model(input_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    # 그래디언트 × 입력
    grad_input = input_tensor.grad.data * input_tensor.data

    # 절댓값을 취하고 채널 방향 최댓값
    saliency = grad_input.abs().max(dim=1)[0].squeeze().cpu().numpy()

    return saliency
```

### 여러 집계 방식 한꺼번에

```python
def compute_all_aggregations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> dict:
    """그래디언트 집계 방법을 비교한다."""
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

---

## 3. 전체 실행 예제

```python
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 사전 트레인된 모델을 불러온다
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.eval()

# 전처리(ImageNet 통계로 정규화)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 이미지를 불러와 전처리한다
image = Image.open('dog.jpg').convert('RGB')
image_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
image_tensor.requires_grad = True

# 모델의 예측을 얻는다
with torch.no_grad():
    output = model(image_tensor.to(device))
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()

print(f"예측 클래스: {pred_class}, 확신도: {confidence:.2%}")

# 여러 Saliency Map을 계산한다
saliency = compute_saliency_map(model, image_tensor, pred_class, device)
pos_saliency, neg_saliency = compute_signed_saliency(model, image_tensor, pred_class)
grad_input = gradient_times_input(model, image_tensor, pred_class)

# 시각화하려고 원본 이미지를 준비한다
original = np.array(image.resize((224, 224))) / 255.0

print(f"샐리언시 형태: {saliency.shape}")
print(f"값 범위: [{saliency.min():.6f}, {saliency.max():.6f}]")
```

---

## 4. 시각화

### 제대로 정규화하기

의미 있게 그리려면 Saliency Map을 최솟값-최댓값 정규화로 $[0, 1]$ 범위에 맞추어야 한다.

```python
def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    시각화하려고 Saliency Map을 [0, 1]로 정규화한다.
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

### 일반 시각화

```python
def visualize_saliency(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    title: str = "Saliency Map"
) -> plt.Figure:
    """
    원본 이미지와 나란히 Saliency Map을 그린다.

    Args:
        original_image: [0, 1] 범위의 원본 이미지 (H, W, 3)
        saliency_map: 샐리언시 값 (H, W)
        title: 그림 제목

    Returns:
        Matplotlib 그림
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    # Saliency Map
    saliency_norm = normalize_saliency(saliency_map)
    im = axes[1].imshow(saliency_norm, cmap='hot')
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 겹쳐 보기
    if original_image.ndim == 3:
        gray_img = np.mean(original_image, axis=2)
    else:
        gray_img = original_image

    axes[2].imshow(gray_img, cmap='gray')
    axes[2].imshow(saliency_norm, cmap='jet', alpha=0.5)
    axes[2].set_title('겹쳐 보기')
    axes[2].axis('off')

    plt.tight_layout()
    return fig
```

### 부호 있는 샐리언시 그리기

```python
def visualize_signed_saliency(
    original_image: np.ndarray,
    positive_saliency: np.ndarray,
    negative_saliency: np.ndarray
) -> plt.Figure:
    """
    양수 샐리언시와 음수 샐리언시를 따로 그린다.

    - 빨강: 올리면 클래스 점수를 올리는 피처
    - 파랑: 올리면 클래스 점수를 내리는 피처
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('원본')
    axes[0].axis('off')

    axes[1].imshow(positive_saliency, cmap='Reds')
    axes[1].set_title('양수 (점수를 올림)')
    axes[1].axis('off')

    axes[2].imshow(negative_saliency, cmap='Blues')
    axes[2].set_title('음수 (점수를 내림)')
    axes[2].axis('off')

    # 결합한 보기: 빨강 = 양수, 파랑 = 음수
    combined = np.zeros((*positive_saliency.shape, 3))
    combined[:, :, 0] = positive_saliency / (positive_saliency.max() + 1e-8)
    combined[:, :, 2] = negative_saliency / (negative_saliency.max() + 1e-8)

    axes[3].imshow(combined)
    axes[3].set_title('결합 (빨강+, 파랑-)')
    axes[3].axis('off')

    plt.tight_layout()
    return fig
```

### 여러 클래스 비교

핵심 통찰은 Saliency Map이 **클래스마다 다르다**는 것이다. 타깃 클래스가 다르면 이미지의 다른 부분을 짚을 수 있다.

```python
def compare_class_saliencies(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list = None,
    device: torch.device = None
) -> dict:
    """
    여러 대상 클래스의 Saliency Map을 계산해 비교한다.

    클래스가 다르면 다른 영역을 강조한다는 것을 보인다.
    """
    if device is None:
        device = next(model.parameters()).device

    saliencies = {}

    for i, class_idx in enumerate(class_indices):
        # 역전파마다 새 텐서를 만든다
        img_copy = image_tensor.clone().detach().requires_grad_(True)

        saliency = compute_saliency_map(model, img_copy, class_idx, device)

        name = class_names[i] if class_names else f"클래스 {class_idx}"
        saliencies[name] = saliency

    return saliencies

# 예: 상위 3개 예측 클래스를 비교한다
with torch.no_grad():
    output = model(image_tensor.to(device))
    top_classes = output[0].topk(3).indices.tolist()

saliencies = compare_class_saliencies(model, image_tensor, top_classes, device=device)
```

---

## 5. 통계로 살피기

샐리언시 값의 분포를 알면 진단에 도움이 된다.

```python
def analyze_saliency_statistics(saliency: np.ndarray) -> dict:
    """샐리언시 분포의 통계량을 두루 계산한다."""
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
        'sparsity': (s < s.mean()).mean(),  # 평균 아래인 기여도
        'max_mean_ratio': s.max() / (s.mean() + 1e-8)
    }

    return stats
```

**해석 가이드:**

| 지표 | 읽는 법 |
|--------|----------------|
| 희소도가 큼(80% 넘음) | 몇몇 픽셀이 지배한다. 의미 있는 영역을 짚었을 가능성이 크다 |
| 최댓값/평균 비가 작음 | 중요도가 퍼져 있고 모델이 전역 정보를 쓴다 |
| 분산이 큼 | 특정 영역을 강하게 짚고 배경을 억누른다 |
| p99와 p95의 차이가 큼 | 극단값이 몇 개 있다. 아티팩트가 아닌지 살펴라 |

---

## 6. 한계와 어려움

### 1. 눈에 띄는 잡음

바닐라 그래디언트 Saliency Map은 **잡음이 많기로** 이름났다. 까닭은 이렇다.

1. **고주파 성분에 대한 민감도**: 뉴럴 네트워크는 립시츠 상수가 커서 입력이 조금만 달라져도 그래디언트가 흔들린다
2. **포화**: ReLU 엑티베이션이 그래디언트 장(field)에 불연속인 곳을 만든다
3. **매끄럽지 않음**: 그래디언트는 국소 민감도를 담을 뿐 전역 중요도를 담지 않는다

```python
def demonstrate_noise():
    """무작위 입력에도 Saliency Map에 잡음이 낄 수 있음을 보인다."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # 무작위 입력
    input_tensor = torch.randn(1, 3, 224, 224)

    saliency = compute_saliency_map(model, input_tensor)

    # 무작위 입력에도 샐리언시가 흩어져 보이는 일이 잦다
    print(f"0이 아닌 픽셀: {(saliency > 0.01 * saliency.max()).sum()}")
    print(f"이는 그래디언트 기반 샐리언시에 잡음이 있음을 알린다")
```

### 2. 그래디언트 포화

포화되는 비선형(시그모이드, tanh)이나 ReLU 네트워크에서는

$$
\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

엑티베이션이 "죽은" 영역에 있으면 입력이 아무리 중요해도 기울기가 0이 된다. 확신도가 아주 큰 예측에서는 소프트맥스 그래디언트도 아주 작아진다.

**푸는 길:** 그래디언트를 계산할 때 소프트맥스 뒤의 확률이 아니라 원시 로짓(소프트맥스 앞의 점수)을 쓴다.

### 3. 클래스를 잘 가려내지 못함

그래디언트를 특정 클래스에 대해 계산하는데도, 바닐라 샐리언시는 클래스가 달라도 비슷한 영역을 짚는 일이 잦다.

```python
def class_discrimination_test(model, input_tensor, device):
    """클래스마다의 Saliency Map을 비교한다."""
    with torch.no_grad():
        output = model(input_tensor.to(device))
        _, top_classes = output.topk(5)

    saliencies = {}
    for cls in top_classes[0]:
        saliencies[cls.item()] = compute_saliency_map(
            model, input_tensor, cls.item(), device
        )

    # 앞선 레이어의 그래디언트를 모든 클래스가 함께 쓰므로
    # 클래스마다의 샐리언시가 꽤 비슷해 보일 때가 많다
    return saliencies
```

### 4. 입력 섭동에 대한 민감도

입력을 조금만 흔들어도 Saliency Map이 크게 달라질 수 있다.

```python
def sensitivity_test(model, input_tensor, device, noise_scale=0.01):
    """샐리언시가 입력 잡음에 얼마나 민감한지 시험한다."""
    saliency_original = compute_saliency_map(model, input_tensor, device=device)

    # 작은 잡음을 더한다
    noisy_input = input_tensor + noise_scale * torch.randn_like(input_tensor)
    saliency_noisy = compute_saliency_map(model, noisy_input, device=device)

    # 차이를 계산한다
    diff = np.abs(saliency_original - saliency_noisy)

    print(f"평균 절댓값 차이: {diff.mean():.4f}")
    print(f"가장 큰 차이: {diff.max():.4f}")
    print(f"상관계수: {np.corrcoef(saliency_original.flatten(), saliency_noisy.flatten())[0,1]:.4f}")

    return diff
```

---

## 7. SmoothGrad: 잡음 줄이기

SmoothGrad는 잡음을 섞은 입력 여러 개에서 구한 그래디언트를 평균 내어 잡음 문제를 다룬다.

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
    SmoothGrad Saliency Map을 계산한다.

    Args:
        model: 뉴럴 네트워크
        input_tensor: 입력 텐서
        target_class: 대상 클래스
        n_samples: 잡음 표본의 수
        noise_level: 잡음의 표준편차(입력 범위에 대한 비율)
        device: 계산할 장치

    Returns:
        매끄럽게 한 Saliency Map
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_tensor = input_tensor.to(device)

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    # 잡음 스케일을 맞추려고 입력의 범위를 계산한다
    stdev = noise_level * (input_tensor.max() - input_tensor.min())

    accumulated_grad = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        # 가우스 잡음을 더한다
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = (input_tensor + noise).requires_grad_(True)

        # 순전파와 역전파
        output = model(noisy_input)
        model.zero_grad()
        output[0, target_class].backward()

        # 그래디언트를 누적한다
        accumulated_grad += noisy_input.grad.data.abs()

    # 평균한다
    smoothed_grad = accumulated_grad / n_samples

    # 채널 방향 최댓값
    saliency = smoothed_grad.max(dim=1)[0].squeeze().cpu().numpy()

    return saliency
```

---

## 8. 금융 응용

### 시계열 샐리언시

순차 금융 데이터에서는 샐리언시가 어느 타임스텝이 중요한지 드러낸다.

```python
def time_series_saliency(
    model: nn.Module,
    sequence: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    시계열 예측의 샐리언시를 계산한다.

    과거의 어느 타임스텝이 예측에 가장 크게 영향을 주는지 보인다.

    Args:
        model: 시퀀스 모델(LSTM, 트랜스포머 등)
        sequence: 입력 시퀀스 [1, 시퀀스 길이, 피처]나 [1, 시퀀스 길이]
        target_class: 분류 모델의 대상 클래스

    Returns:
        시간 축에 걸친 샐리언시
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

    # 시간 축에 걸친 샐리언시
    saliency = sequence.grad.abs().squeeze().cpu().numpy()

    # 피처가 여럿이면 합친다
    if saliency.ndim == 2:
        saliency = saliency.sum(axis=1)  # 피처 축으로 더한다

    return saliency
```

### 표 데이터의 피처 중요도

표 형식의 금융 데이터(신용 위험, 거래 신호 등)에는 피처마다 그래디언트 크기를 중요도로 쓴다.

```python
def tabular_saliency(
    model: nn.Module,
    features: torch.Tensor,
    feature_names: list,
    target_class: int = None
) -> np.ndarray:
    """
    그래디언트로 피처 중요도를 계산하고 출력한다.

    Args:
        model: 분류나 회귀 모델
        features: 피처 텐서 [1, n_features]
        feature_names: 피처 이름 목록
        target_class: 대상 클래스(분류일 때)

    Returns:
        피처마다의 중요도 점수
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

    # 중요도 순으로 정렬한다
    sorted_idx = np.argsort(importance)[::-1]

    print("피처 중요도(그래디언트 크기 기준):")
    print("-" * 50)
    for i in sorted_idx[:10]:
        print(f"{feature_names[i]:30s}: {importance[i]:.6f}")

    return importance
```

---

## 9. 다른 방법과의 관계

바닐라 그래디언트는 더 정교한 방법의 기초가 된다.

| 방법 | 수정 | 다루는 것 |
|--------|--------------|-----------|
| **그래디언트 × 입력** | 입력 값을 곱한다 | 기여도를 또렷하게 한다 |
| **SmoothGrad** | 잡음 표본에 걸쳐 평균 낸다 | 잡음을 줄인다 |
| **Integrated Gradients** | 베이스라인에서 입력까지의 경로를 따라 적분한다 | 공리를 만족한다 |
| **Guided Backpropagation** | ReLU 역전파 규칙을 고친다 | 맵이 깨끗해진다 |
| **Grad-CAM** | 피처 맵의 그래디언트를 쓴다 | 클래스를 가려낸다 |

### 비교 표

| 방법 | 장점 | 단점 |
|--------|------|------|
| 바닐라 그래디언트 | 단순하고 빠르며 기초가 된다 | 잡음이 많고 클래스를 잘 못 가려낸다 |
| 그래디언트 × 입력 | 기여도가 더 또렷하다 | 여전히 잡음이 많다 |
| SmoothGrad | 잡음이 준다 | 계산 비용이 크다 |
| Grad-CAM | 깨끗하고 클래스를 가려낸다 | 해상도가 낮다 |
| Integrated Gradients | 이론적 보장이 있다 | 느리고 베이스라인 선택에 좌우된다 |

---

## 10. 실제로 쓸 때의 권장 사항

### 바닐라 그래디언트를 쓸 때

**알맞은 경우:**

- 빠른 디버깅과 새니티 체크
- 그래디언트 흐름 확인
- 다른 방법과 비교할 베이스라인
- 교육용
- 초기 탐색

**권하지 않는 경우:**

- 논문에 실을 그림(잡음이 너무 많다)
- 서비스 환경의 해석(더 매끄러운 방법을 쓰라)
- 위험이 큰 판단(다른 증거와 함께 쓰라)
- 클래스를 가려내는 설명(Grad-CAM을 쓰라)

### 구현 체크리스트

1. ✅ **모델을 평가 모드로**: `model.eval()`로 동작을 일관되게 한다
2. ✅ **입력에 그래디언트를 켠다**: `image_tensor.requires_grad = True`
3. ✅ **남은 그래디언트를 지운다**: 역전파 앞에 `model.zero_grad()`
4. ✅ **알맞은 클래스를 쓴다**: 예측한 클래스나 눈여겨보는 클래스
5. ✅ **소프트맥스가 아니라 로짓을 쓴다**: 그래디언트 포화를 피한다
6. ✅ **시각화 전에 정규화한다**: $[0, 1]$ 범위로 스케일을 맞춘다
7. ✅ **텐서를 복제한다**: 비교할 때는 역전파마다 새 텐서를 쓴다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 절에서 다룬 설명 방법을, XOR 입력을 분류하는 ReLU 엑티베이션의 두 레이어 뉴럴 네트워크에 적용하라. 입력 $x = [1, 1]$에 대한 설명을 계산하여라.

</div>

??? success "풀이"
    웨이트가 $W_1, b_1, W_2, b_2$인 트레인된 XOR 네트워크에서 출력은 $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$이다. 설명 방법은 입력 피처마다 기여도를 내놓는다. $x = [1, 1]$(클래스 0)이면 두 피처 모두 음수 쪽 분류에 기여한다. 기여도 값은 방법마다 다르다. 그래디언트 기반 방법은 $\partial f / \partial x_i$을 계산하고, 섭동 기반 방법은 피처를 가렸을 때 출력이 얼마나 바뀌는지 잰다. XOR 문제는 결정 경계가 선형이 아니므로 선형 설명 방법이 틀릴 수 있음을 보여 준다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 절의 설명 방법이 완전성 공리를 만족하는지, 곧 어떤 베이스라인 $x_0$에 대해 모든 피처 기여도의 합이 $f(x) - f(x_0)$과 같은지 증명하거나 반례를 들어라.

</div>

??? success "풀이"
    완전성 공리(섀플리 값 이론에서는 효율성이라고도 한다)는 기여도의 합이 입력에서의 모델 출력과 베이스라인에서의 출력의 차이와 같다는 것이다. 이 방법이 완전성을 만족하는지는 그 구성에 달렸다. 그래디언트 방법은 완전성을 만족하지 못한다(그래디언트는 한 점의 국소적인 양이고 경로를 따라 적분한 것이 아니다). Integrated Gradients는 구성 자체로 완전성을 만족한다(경로를 따라 미적분의 기본정리를 쓴다). SHAP 값은 섀플리 공리로 효율성을 만족한다. 완전성을 어기는 방법은 기여도를 너무 크거나 작게 매길 수 있어, 기여도 전체를 전역 설명으로 믿기 어렵게 만든다. $\square$

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

바닐라 그래디언트 샐리언시는 그래디언트 기반 설명 방법을 이해하는 개념적 기초를 준다. 단순하고 직관적이지만, 눈에 띄는 잡음과 약한 클래스 구별력 탓에 실제 쓸모는 좁다. 뒤따르는 방법들이 이 기초 위에서 더 깨끗하고 의미 있는 기여도 분석을 낸다.

### 핵심 식

**기본 샐리언시:**

$$
S(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**집계한 샐리언시(다채널 입력):**

$$
S_{i,j} = \max_{k} |G_{k,i,j}| \quad \text{또는} \quad \sqrt{\sum_k G_{k,i,j}^2}
$$

**그래디언트 × 입력:**

$$
S(\mathbf{x}) = \left| \mathbf{x} \odot \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**SmoothGrad:**

$$
\hat{S}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial f_c(\mathbf{x} + \epsilon_i)}{\partial \mathbf{x}} \right|, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

### 핵심

1. **그래디언트는 국소 민감도를 잰다**. 전역 중요도가 아니다
2. 네트워크가 매끄럽지 않으므로 **잡음이 따라붙는다**
3. 앞쪽 레이어를 모든 클래스가 함께 쓰므로 **클래스 구별력이 약하다**
4. **채널 집계 방식이 여럿** 있고 최댓값이 가장 흔하다
5. Integrated Gradients, SmoothGrad 같은 **고급 방법의 기초가 된다**

**살펴볼 거리**

1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." *ICLR Workshop*.

2. Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). "SmoothGrad: removing noise by adding noise." *ICML Workshop*.

3. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

4. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). "Sanity Checks for Saliency Maps." *NeurIPS*.

5. Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., & Müller, K. R. (2010). "How to Explain Individual Classification Decisions." *Journal of Machine Learning Research*.
