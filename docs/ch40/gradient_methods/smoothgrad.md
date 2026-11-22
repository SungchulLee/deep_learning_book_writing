# SmoothGrad

**SmoothGrad**는 그래디언트 기반 Saliency Map에 늘 끼는 눈에 띄는 잡음을 줄이는, 단순하면서도 효과적인 기법이다. 핵심 아이디어는 언뜻 역설적이다. 입력에 **잡음을 더하고** 거기서 나온 그래디언트를 평균 내면 **더 또렷하고 깨끗한** 맵을 얻는다는 것이다.

스밀코프 외(2017)가 제안한 SmoothGrad는 바닐라 그래디언트 샐리언시의 큰 걸림돌 하나, 곧 해석하기 어렵게 만드는 잡음 낀 얼룩덜룩한 모습을 다룬다.

---

## 1. 수학 기초

### SmoothGrad 정의

입력 $\mathbf{x}$, 모델 $f$, 타깃 클래스 $c$가 주어지면 SmoothGrad 샐리언시는 이렇다.

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}
$$

여기서

- $n$은 잡음 섞은 표본의 수
- $\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$는 가우스 잡음
- $\sigma$는 잡음의 크기(표준편차)를 정한다

### 잡음을 더했는데 왜 잡음이 주는가

잡음이 하는 일을 따져 보면 이 역설이 풀린다.

**관찰 1: 그래디언트는 국소적으로 크게 흔들린다**

뉴럴 네트워크의 출력 함수에는 고주파 성분이 있다. 입력을 조금만 흔들어도 픽셀마다의 그래디언트가 확 달라질 수 있다.

**관찰 2: 진짜 중요도는 흔들리지 않는다**

어떤 픽셀이 예측에 정말 중요하다면 입력이 조금 달라져도 꾸준히 큰 그래디언트를 낸다. 우연히 그래디언트가 큰 대수롭지 않은 픽셀은 평균을 내면 서로 상쇄된다.

**함의:** SmoothGrad는 **그래디언트 장을 국소적으로 평균 내는** 일을 해서, 일관된 신호는 남기고 잡음은 지운다.

### 그래디언트 평활화와의 관계

SmoothGrad는 그래디언트의 기댓값을 추정하는 것으로 볼 수 있다.

$$
\text{SG}(\mathbf{x}) \approx \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})]
$$

이는 모델을 **매끄럽게 한 것**의 그래디언트를 계산하는 것과 같다.

$$
\tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \int f_c(\mathbf{x} + \boldsymbol{\epsilon}) p(\boldsymbol{\epsilon}) d\boldsymbol{\epsilon}
$$

미분과 적분의 순서를 바꿀 수 있는 조건 아래에서는

$$
\nabla_{\mathbf{x}} \tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \text{SG}(\mathbf{x})
$$

곧 모델을 가우스 커널과 컨볼루션한 함수의 그래디언트이며, 그래서 매끄러워지는 것이다. 다만 위 정의의 SmoothGrad는 그래디언트를 그대로 평균 내고, 코드는 평균 낸 뒤에 절댓값을 잡는다. 「Saliency Map과 기본 그래디언트」 쪽의 `smoothgrad` 함수처럼 절댓값을 먼저 잡고 평균 내면 이 등식은 더 이상 성립하지 않는다.

---

## 2. PyTorch 구현

### 기본 구현

```python
import torch
import torch.nn as nn
import numpy as np

def compute_smoothgrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """
    SmoothGrad 샐리언시 맵을 계산한다.

    Args:
        model: 평가 모드로 둔 뉴럴 네트워크
        image_tensor: 입력 이미지 [1, C, H, W]
        target_class: 대상 클래스 번호
        device: 계산할 장치
        n_samples: 평균할 잡음 표본의 수
        noise_level: 가우스 잡음의 표준편차
                    (입력 값 범위에 대한 비율, 흔히 0.1~0.2)

    Returns:
        샐리언시 맵 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 입력 공간에서의 표준편차
    # 입력은 흔히 정규화되어 있으므로 noise_level ~0.15가 알맞다
    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    # 그래디언트를 누적한다
    accumulated_gradients = torch.zeros_like(image_tensor)

    for _ in range(n_samples):
        # 가우스 잡음을 더한다
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise
        noisy_input.requires_grad_(True)

        # 순전파
        output = model(noisy_input)
        target_score = output[0, target_class]

        # 역전파
        model.zero_grad()
        target_score.backward()

        # 누적한다
        accumulated_gradients += noisy_input.grad

    # 그래디언트를 평균한다
    avg_gradients = accumulated_gradients / n_samples

    # 절댓값을 잡고 채널 축으로 최댓값을 모은다
    saliency = torch.abs(avg_gradients).max(dim=1)[0]

    return saliency
```

### 배치로 하는 구현(효율적이다)

```python
def compute_smoothgrad_batched(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15,
    batch_size: int = 10
) -> torch.Tensor:
    """
    배치로 처리하는 SmoothGrad 계산.

    잡음 표본 여럿을 한꺼번에 처리해 효율을 높인다.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_gradients = torch.zeros_like(image_tensor)

    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)

        # 잡음 섞은 입력의 배치를 만든다
        # 이미지를 배치로 확장한다: [batch_size, C, H, W]
        batch = image_tensor.expand(current_batch_size, -1, -1, -1).clone()
        noise = torch.randn_like(batch) * stdev
        noisy_batch = batch + noise
        noisy_batch.requires_grad_(True)

        # 순전파
        outputs = model(noisy_batch)  # [batch_size, num_classes]

        # 배치 그래디언트를 얻으려고 대상 클래스 점수를 더한다
        target_scores = outputs[:, target_class].sum()

        # 역전파
        model.zero_grad()
        target_scores.backward()

        # 누적한다(배치 축으로 더한다)
        accumulated_gradients += noisy_batch.grad.sum(dim=0, keepdim=True)

    # 평균한다
    avg_gradients = accumulated_gradients / n_samples
    saliency = torch.abs(avg_gradients).max(dim=1)[0]

    return saliency
```

---

## 3. 하이퍼파라미터

### 잡음 크기 (σ)

잡음 크기는 매끄러움과 세밀함 사이의 트레이드오프를 정한다. 아래 값은 입력 값 범위(최댓값 − 최솟값)에 대한 비율이다.

| 잡음 크기 | 효과 |
|-------------|--------|
| 너무 작음 (< 0.05) | 거의 매끄러워지지 않아 여전히 잡음이 많다 |
| 적당함 (0.10~0.20) | 잘 매끄러워지면서 세밀한 패턴도 남는다 |
| 너무 큼 (> 0.30) | 지나치게 매끄러워져 세밀한 패턴을 잃는다 |

```python
def analyze_noise_levels(
    model, image_tensor, target_class, device,
    noise_levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
):
    """잡음 크기를 달리한 SmoothGrad 결과를 비교한다."""
    results = {}

    for noise in noise_levels:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=50, noise_level=noise
        )
        results[noise] = saliency.cpu().numpy()

    return results
```

**잡음 크기 선택 가이드:**

- 기본값으로 **0.15에서 시작한다**
- 샐리언시에 여전히 잡음이 많으면 **올린다**
- 중요한 세밀한 패턴이 뭉개지면 **내린다**
- **입력을 어떻게 정규화했는지**도 고려한다. 입력이 [0, 1]에 있으면 잡음을 ~0.1~0.2로 잡는다

### 표본의 수 (n)

표본이 많을수록 매끄럽고 안정적이지만 계산 비용이 오른다.

| 표본 | 품질 | 쓰임 |
|---------|---------|----------|
| 10~20 | 거침 | 빠른 탐색 |
| 50 | 좋음 | 일반 용도 |
| 100 넘음 | 아주 좋음 | 논문에 실을 품질 |

```python
def analyze_sample_convergence(
    model, image_tensor, target_class, device,
    sample_counts=[10, 20, 30, 50, 75, 100]
):
    """표본 수에 따라 SmoothGrad가 어떻게 수렴하는지 살핀다."""
    results = {}

    for n in sample_counts:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=n, noise_level=0.15
        )
        results[n] = saliency.cpu().numpy()

    # 안정성을 계산한다(표본이 가장 많은 것과의 상관계수)
    reference = results[max(sample_counts)]

    correlations = {}
    for n, sal in results.items():
        corr = np.corrcoef(sal.flatten(), reference.flatten())[0, 1]
        correlations[n] = corr

    return results, correlations
```

---

## 4. 변형과 확장

### SmoothGrad 제곱

평균 내기 전에 그래디언트를 제곱한다(크기가 큰 그래디언트를 강조한다).

$$
\text{SG}^2(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \left( \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}} \right)^2
$$

```python
def compute_smoothgrad_squared(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """그래디언트를 제곱하는 SmoothGrad."""
    model.eval()
    image_tensor = image_tensor.to(device)

    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_squared = torch.zeros_like(image_tensor)

    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = (image_tensor + noise).requires_grad_(True)

        output = model(noisy_input)
        output[0, target_class].backward()

        # 누적하기 전에 제곱한다
        accumulated_squared += noisy_input.grad ** 2
        model.zero_grad()

    avg_squared = accumulated_squared / n_samples
    saliency = torch.sqrt(avg_squared).max(dim=1)[0]

    return saliency
```

### VarGrad(그래디언트의 분산)

평균이 아니라 그래디언트가 얼마나 흩어져 있는지를 잰다.

$$
\text{VarGrad}(\mathbf{x}) = \text{Var}_{\boldsymbol{\epsilon}}\left[ \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon})}{\partial \mathbf{x}} \right]
$$

```python
def compute_vargrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """VarGrad: 잡음 표본에 걸친 그래디언트의 분산."""
    model.eval()
    image_tensor = image_tensor.to(device)

    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    gradients_list = []

    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = (image_tensor + noise).requires_grad_(True)

        output = model(noisy_input)
        output[0, target_class].backward()

        gradients_list.append(noisy_input.grad.clone())
        model.zero_grad()

    # 쌓아 올리고 분산을 계산한다
    all_gradients = torch.stack(gradients_list)  # [n, 1, C, H, W]
    variance = all_gradients.var(dim=0)  # [1, C, H, W]

    saliency = variance.max(dim=1)[0]

    return saliency
```

**읽는 법:** 분산이 크면 그 픽셀에서 모델이 **불확실하거나** 작은 변화에 **민감하다**는 뜻이다.

### SmoothGrad를 다른 방법과 결합하기

SmoothGrad는 다른 기여도 분석 방법을 개선하는 데도 쓸 수 있다. 아래 두 함수는 `compute_integrated_gradients`와 `GradCAM`이 따로 정의되어 있다고 가정한다(각 방법의 쪽에서 다룬다).

```python
def smooth_integrated_gradients(
    model, image_tensor, target_class, device,
    n_smooth_samples: int = 20,
    n_ig_steps: int = 50,
    noise_level: float = 0.1
):
    """
    SmoothGrad와 Integrated Gradients를 결합한다.

    잡음 섞은 입력에서 계산한 IG 기여도를 평균한다.
    """
    accumulated = torch.zeros_like(image_tensor)
    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    for _ in range(n_smooth_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise

        # 잡음 섞은 입력에 IG를 계산한다
        ig_attr = compute_integrated_gradients(
            model, noisy_input, target_class, device,
            steps=n_ig_steps
        )
        accumulated += ig_attr

    return accumulated / n_smooth_samples

def smooth_gradcam(
    model, target_layer, image_tensor, target_class, device,
    n_samples: int = 20,
    noise_level: float = 0.1
):
    """
    SmoothGrad의 아이디어를 Grad-CAM에 적용한다.
    """
    gradcam = GradCAM(model, target_layer)

    accumulated = None
    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise

        heatmap = gradcam(noisy_input, target_class, device)

        if accumulated is None:
            accumulated = heatmap
        else:
            accumulated += heatmap

    return accumulated / n_samples
```

---

## 5. 시각화

### SmoothGrad와 바닐라 그래디언트 비교

```python
import matplotlib.pyplot as plt

def visualize_smoothgrad_comparison(
    model, image_tensor, target_class, device,
    noise_level: float = 0.15,
    n_samples: int = 50
):
    """바닐라 그래디언트와 SmoothGrad를 나란히 비교한다."""

    # 바닐라 그래디언트
    img = image_tensor.clone().requires_grad_(True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.abs(img.grad).max(dim=1)[0]

    # SmoothGrad
    smoothgrad = compute_smoothgrad(
        model, image_tensor, target_class, device,
        n_samples=n_samples, noise_level=noise_level
    )

    # 이미지의 정규화를 되돌린다
    image_np = denormalize_image(image_tensor)

    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    axes[1].imshow(vanilla.squeeze().cpu().numpy(), cmap='hot')
    axes[1].set_title('바닐라 그래디언트\n(잡음 많음)')
    axes[1].axis('off')

    axes[2].imshow(smoothgrad.squeeze().cpu().numpy(), cmap='hot')
    axes[2].set_title(f'SmoothGrad\n(n={n_samples}, σ={noise_level})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig
```

### 하이퍼파라미터 민감도 그리기

```python
def visualize_hyperparameter_effects(
    model, image_tensor, target_class, device
):
    """잡음 크기와 표본 수가 미치는 영향을 시각화한다."""

    noise_levels = [0.05, 0.15, 0.25]
    sample_counts = [10, 50, 100]

    fig, axes = plt.subplots(
        len(noise_levels), len(sample_counts) + 1,
        figsize=(4 * (len(sample_counts) + 1), 4 * len(noise_levels))
    )

    image_np = denormalize_image(image_tensor)

    for i, noise in enumerate(noise_levels):
        # 잡음 크기 레이블을 표시한다
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'σ = {noise}' if i == 0 else '')
        axes[i, 0].set_ylabel(f'잡음: {noise}', fontsize=12)
        axes[i, 0].axis('off')

        for j, n_samples in enumerate(sample_counts):
            saliency = compute_smoothgrad(
                model, image_tensor, target_class, device,
                n_samples=n_samples, noise_level=noise
            )

            axes[i, j + 1].imshow(saliency.squeeze().cpu().numpy(), cmap='hot')
            if i == 0:
                axes[i, j + 1].set_title(f'n = {n_samples}')
            axes[i, j + 1].axis('off')

    plt.suptitle('SmoothGrad: 잡음 크기 대 표본 수', fontsize=14)
    plt.tight_layout()
    return fig
```

---

## 6. 장점과 한계

### 장점

1. **구현이 쉽다**: 잡음 섞은 표본을 뽑아 평균 내기만 하면 된다
2. **어떤 그래디언트 방법에도 작동한다**: 바닐라 그래디언트, 그래디언트×입력 등을 모두 매끄럽게 할 수 있다
3. **병렬 계산에 알맞다**: 배치로 구현하면 효율적이다
4. **눈에 더 깨끗하다**: 해석하기 좋은 맵이 나온다
5. **구조에 손대지 않는다**: 미분 가능한 모델이면 구조를 고칠 일이 없다

### 한계

1. **이론적 보장이 없다**: Integrated Gradients와 달리 정해진 공리를 만족하지 않는다
2. **하이퍼파라미터에 민감하다**: 결과가 잡음 크기와 표본 수에 좌우된다
3. **계산 비용이 늘어난다**: 순전파-역전파를 여러 번 해야 한다
4. **지나치게 매끄러워질 수 있다**: 중요한 세밀한 패턴을 잃을 수 있다
5. **근본 문제를 풀지는 않는다**: 맵만 깨끗해질 뿐 기여도 분석 자체의 충실성이 오르지는 않는다

---

## 7. 실제로 쓸 때의 권장 사항

### SmoothGrad를 쓸 때

**권하는 경우:**

- 빠르게 시각화하고 디버깅할 때
- 바닐라 그래디언트가 너무 잡음이 많아 해석할 수 없을 때
- 다른 그래디언트 방법의 후처리로 쓸 때
- 발표와 시연

**다른 방법을 고려할 경우:**

- 이론적 보장이 중요할 때 → Integrated Gradients
- 영역 단위로 위치를 파악해야 할 때 → Grad-CAM
- 계산 여유가 적을 때 → 바닐라 그래디언트

### 구현 체크리스트

1. **잡음 크기**: 0.15에서 시작해 결과를 보고 조정한다
2. **표본 수**: 일반 분석에는 50, 최종 결과에는 100 넘게
3. **배치 처리**: 효율적으로 하려면 배치 구현을 쓴다
4. **눈으로 확인**: 바닐라 그래디언트와 비교하여 매끄러워졌는지 본다

### 기존 파이프라인에 끼워 넣기

```python
class SaliencyPipeline:
    """샐리언시 계산 전체 파이프라인."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        method: str = 'smoothgrad',
        **kwargs
    ) -> torch.Tensor:
        """
        지정한 방법으로 샐리언시를 계산한다.

        Args:
            method: 'vanilla', 'smoothgrad', 'smoothgrad_squared', 'vargrad'
        """
        if method == 'vanilla':
            return self._vanilla_gradient(image_tensor, target_class)
        elif method == 'smoothgrad':
            return compute_smoothgrad(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        elif method == 'smoothgrad_squared':
            return compute_smoothgrad_squared(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        elif method == 'vargrad':
            return compute_vargrad(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        else:
            raise ValueError(f"알 수 없는 방법: {method}")
```

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

SmoothGrad는 그래디언트 기반 Saliency Map의 잡음 문제에 실용적인 답을 준다.

**핵심 식:**

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$

**핵심 통찰:**

- 잡음을 더해 평균 내면 무의미한 고주파 그래디언트 신호가 상쇄된다
- 진짜 중요도는 입력이 조금 달라져도 일관된다
- 가우스로 매끄럽게 한 모델의 그래디언트를 계산하는 것과 같다

**권하는 기본값:**

- 잡음 크기: σ = 0.15
- 표본 수: n = 50

**살펴볼 거리**

1. Smilkov, D., et al. (2017). *SmoothGrad: Removing Noise by Adding Noise*. arXiv:1706.03825.

2. Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps*. NeurIPS.

3. Hooker, S., et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks*. NeurIPS.
