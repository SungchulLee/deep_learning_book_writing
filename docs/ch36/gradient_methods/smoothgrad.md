# SmoothGrad
## 들머리

**SmoothGrad**은 기울기 바탕 두드러짐 그림에 늘 끼는 눈에 띄는 잡음을 줄이는, 단순하면서도 잘 듣는 재주다. 고갱이 생각은 얼른 보아 얄궂다. 들임에 **잡음을 더하고** 거기서 나온 기울기를 고르게 하면 **더 또렷하고 깨끗한** 그림을 얻는다는 것이다.

스밀코프 외(2017)가 들여온 SmoothGrad은 맨 기울기 두드러짐의 큰 걸림돌 하나, 곧 풀이하기 어렵게 만드는 잡음 낀 얼룩덜룩한 모습을 다룬다.

## 수학 밑바탕

### SmoothGrad 꼴

들임 $\mathbf{x}$, 모형 $f$, 겨눈 갈래 $c$이 주어지면 SmoothGrad 두드러짐은 이렇다.

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}
$$

여기서

- $n$은 잡음 섞은 표본의 수
- $\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$은 가우스 잡음
- $\sigma$은 잡음의 크기(잣대 벗어남)를 다스린다

### 잡음을 더했는데 왜 잡음이 주는가

잡음이 하는 일을 헤아리면 이 얄궂음이 풀린다.

**눈여겨볼 것 1: 기울기는 그 자리에서 흔들린다**

신경 그물의 잃음 터에는 잦기가 높은 결이 있다. 들임을 조금만 흔들어도 그림점마다의 기울기가 확 달라질 수 있다.

**눈여겨볼 것 2: 참된 중요함은 흔들리지 않는다**

어떤 그림점이 미루어 봄에 참으로 중요하다면 들임이 조금 달라져도 한결같이 큰 기울기를 낸다. 우연히 기울기가 큰 대수롭지 않은 그림점은 고르게 하면 서로 지워진다.

**속뜻:** SmoothGrad은 **기울기 밭에서 그 자리를 고르게 하는** 일을 해서, 한결같은 신호는 남기고 잡음은 지운다.

### 기울기 매끄럽게 하기와의 이어짐

SmoothGrad은 기댓값 기울기를 어림하는 것으로 볼 수 있다.

$$
\text{SG}(\mathbf{x}) \approx \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})]
$$

이는 모형을 **매끄럽게 한 것**의 기울기를 셈하는 것과 같다.

$$
\tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \int f_c(\mathbf{x} + \boldsymbol{\epsilon}) p(\boldsymbol{\epsilon}) d\boldsymbol{\epsilon}
$$

어떤 조건 아래에서는

$$
\nabla_{\mathbf{x}} \tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \text{SG}(\mathbf{x})
$$

이는 모형을 가우스 낟알로 겹친 것의 기울기이며, 그래서 매끄러워지는 것이다.

## PyTorch 짜보기

### 기본 짜보기

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
    SmoothGrad 두드러짐 그림을 셈한다.

    Args:
        model: 따지는 결로 놓인 신경 그물
        image_tensor: 들임 그림 [1, C, H, W]
        target_class: 겨눈 갈래 번호
        device: 셈하는 장치
        n_samples: 고르게 할 잡음 표본의 수
        noise_level: 가우스 잡음의 잣대 벗어남
                    (들임 너비에 대한 몫, 흔히 0.1~0.2)

    Returns:
        두드러짐 그림 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 들임 밭에서의 잣대 벗어남
    # 들임은 흔히 고르게 되어 있으므로 noise_level ~0.15이 알맞다
    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    # 기울기를 쌓는다
    accumulated_gradients = torch.zeros_like(image_tensor)

    for _ in range(n_samples):
        # 가우스 잡음을 더한다
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise
        noisy_input.requires_grad_(True)

        # 앞으로 걸음
        output = model(noisy_input)
        target_score = output[0, target_class]

        # 되짚기 걸음
        model.zero_grad()
        target_score.backward()

        # 쌓는다
        accumulated_gradients += noisy_input.grad

    # 기울기를 고르게 한다
    avg_gradients = accumulated_gradients / n_samples

    # 절댓값을 잡고 통로를 가로질러 모은다
    saliency = torch.abs(avg_gradients).max(dim=1)[0]

    return saliency
```

### 묶음으로 하는 짜보기(잘 든다)

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
    묶음으로 다듬은 SmoothGrad 셈.

    잡음 표본 여럿을 함께 다루어 잘 들게 한다.
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_gradients = torch.zeros_like(image_tensor)

    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)

        # 잡음 섞은 들임의 묶음을 만든다
        # 그림을 묶음으로 넓힌다: [batch_size, C, H, W]
        batch = image_tensor.expand(current_batch_size, -1, -1, -1).clone()
        noise = torch.randn_like(batch) * stdev
        noisy_batch = batch + noise
        noisy_batch.requires_grad_(True)

        # 앞으로 걸음
        outputs = model(noisy_batch)  # [batch_size, num_classes]

        # 묶음 기울기를 얻으려고 겨눈 점수를 더한다
        target_scores = outputs[:, target_class].sum()

        # 되짚기 걸음
        model.zero_grad()
        target_scores.backward()

        # 쌓는다(묶음 축으로 더한다)
        accumulated_gradients += noisy_batch.grad.sum(dim=0, keepdim=True)

    # 고르게 한다
    avg_gradients = accumulated_gradients / n_samples
    saliency = torch.abs(avg_gradients).max(dim=1)[0]

    return saliency
```

## 하이퍼파라미터

### 잡음 크기 (σ)

잡음 크기는 매끄러움과 맞음 사이의 맞바꿈을 다스린다.

| 잡음 크기 | 미치는 힘 |
|-------------|--------|
| 너무 작음 (< 0.05) | 거의 매끄러워지지 않아 여전히 잡음이 많다 |
| 알맞음 (0.10~0.20) | 잘 매끄러워지면서 잔 무늬도 남는다 |
| 너무 큼 (> 0.30) | 지나치게 매끄러워져 잔 무늬를 잃는다 |

```python
def analyze_noise_levels(
    model, image_tensor, target_class, device,
    noise_levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
):
    """잡음 크기를 달리한 SmoothGrad 열매를 견준다."""
    results = {}

    for noise in noise_levels:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=50, noise_level=noise
        )
        results[noise] = saliency.cpu().numpy()

    return results
```

**잡음 크기를 고르는 길잡이:**

- 맡긴 값으로 **0.15에서 시작한다**
- 두드러짐이 여전히 잡음 많으면 **올린다**
- 중요한 잔 무늬가 뭉개지면 **내린다**
- **들임을 어떻게 고르게 했는지**도 헤아린다. 들임이 [0, 1]에 있으면 잡음을 ~0.1~0.2로 잡는다

### 표본의 수 (n)

표본이 많을수록 매끄럽고 든든하지만 셈 값이 오른다.

| 표본 | 됨됨이 | 쓰일 자리 |
|---------|---------|----------|
| 10~20 | 거침 | 빠르게 둘러보기 |
| 50 | 좋음 | 여느 쓰임 |
| 100 넘음 | 아주 좋음 | 논문에 실을 됨됨이 |

```python
def analyze_sample_convergence(
    model, image_tensor, target_class, device,
    sample_counts=[10, 20, 30, 50, 75, 100]
):
    """표본 수에 따라 SmoothGrad이 어떻게 모이는지 살핀다."""
    results = {}

    for n in sample_counts:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=n, noise_level=0.15
        )
        results[n] = saliency.cpu().numpy()

    # 든든함을 셈한다(표본이 가장 많은 것과의 얽힘)
    reference = results[max(sample_counts)]

    correlations = {}
    for n, sal in results.items():
        corr = np.corrcoef(sal.flatten(), reference.flatten())[0, 1]
        correlations[n] = corr

    return results, correlations
```

## 갈래와 넓힘

### SmoothGrad 제곱

고르게 하기에 앞서 기울기를 제곱한다(크기가 큰 기울기를 앞세운다).

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
    """기울기를 제곱하는 SmoothGrad."""
    model.eval()
    image_tensor = image_tensor.to(device)

    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_squared = torch.zeros_like(image_tensor)

    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = (image_tensor + noise).requires_grad_(True)

        output = model(noisy_input)
        output[0, target_class].backward()

        # 쌓기에 앞서 제곱한다
        accumulated_squared += noisy_input.grad ** 2
        model.zero_grad()

    avg_squared = accumulated_squared / n_samples
    saliency = torch.sqrt(avg_squared).max(dim=1)[0]

    return saliency
```

### VarGrad(기울기의 흩어짐)

고른 값이 아니라 기울기가 얼마나 들쭉날쭉한지를 잰다.

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
    """VarGrad: 잡음 표본에 걸친 기울기의 흩어짐."""
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

    # 쌓아 올리고 흩어짐을 셈한다
    all_gradients = torch.stack(gradients_list)  # [n, 1, C, H, W]
    variance = all_gradients.var(dim=0)  # [1, C, H, W]

    saliency = variance.max(dim=1)[0]

    return saliency
```

**읽는 법:** 흩어짐이 크면 그 그림점에서 모형이 **아리송해하거나** 작은 바뀜에 **예민하다**는 뜻이다.

### SmoothGrad을 다른 방법과 아우르기

SmoothGrad은 다른 몫 매기기 방법을 낫게 하는 데도 걸 수 있다.

```python
def smooth_integrated_gradients(
    model, image_tensor, target_class, device,
    n_smooth_samples: int = 20,
    n_ig_steps: int = 50,
    noise_level: float = 0.1
):
    """
    SmoothGrad과 쌓은 기울기를 아우른다.

    잡음 섞은 들임에서 셈한 IG 몫을 고르게 한다.
    """
    accumulated = torch.zeros_like(image_tensor)
    stdev = noise_level * (image_tensor.max() - image_tensor.min())

    for _ in range(n_smooth_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise

        # 잡음 섞은 들임에 IG를 셈한다
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
    SmoothGrad의 생각을 Grad-CAM에 건다.
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

## 그림으로 보이기

### SmoothGrad과 맨 기울기 견주기

```python
import matplotlib.pyplot as plt

def visualize_smoothgrad_comparison(
    model, image_tensor, target_class, device,
    noise_level: float = 0.15,
    n_samples: int = 50
):
    """맨 기울기와 SmoothGrad을 나란히 견준다."""

    # 맨 기울기
    img = image_tensor.clone().requires_grad_(True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.abs(img.grad).max(dim=1)[0]

    # SmoothGrad
    smoothgrad = compute_smoothgrad(
        model, image_tensor, target_class, device,
        n_samples=n_samples, noise_level=noise_level
    )

    # 그림을 되돌려 고른다
    image_np = denormalize_image(image_tensor)

    # 그린다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title('본디 그림')
    axes[0].axis('off')

    axes[1].imshow(vanilla.squeeze().cpu().numpy(), cmap='hot')
    axes[1].set_title('맨 기울기\n(잡음 많음)')
    axes[1].axis('off')

    axes[2].imshow(smoothgrad.squeeze().cpu().numpy(), cmap='hot')
    axes[2].set_title(f'SmoothGrad\n(n={n_samples}, σ={noise_level})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig
```

### 하이퍼파라미터 예민함 그리기

```python
def visualize_hyperparameter_effects(
    model, image_tensor, target_class, device
):
    """잡음 크기와 표본 수가 미치는 힘을 그린다."""

    noise_levels = [0.05, 0.15, 0.25]
    sample_counts = [10, 50, 100]

    fig, axes = plt.subplots(
        len(noise_levels), len(sample_counts) + 1,
        figsize=(4 * (len(sample_counts) + 1), 4 * len(noise_levels))
    )

    image_np = denormalize_image(image_tensor)

    for i, noise in enumerate(noise_levels):
        # 잡음 크기 이름표를 보인다
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

## 나은 점과 한계

### 나은 점

1. **짜기 쉽다**: 뽑아서 고르게 하기만 하면 된다
2. **어떤 기울기 방법에도 듣는다**: 맨 기울기, 기울기×들임 등을 모두 매끄럽게 할 수 있다
3. **함께 셈하기 좋다**: 묶음으로 짜면 잘 든다
4. **눈에 더 깨끗하다**: 풀이하기 좋은 그림이 나온다
5. **모형을 가리지 않는다**: 얼개를 고칠 일이 없다

### 한계

1. **이론 보장이 없다**: 쌓은 기울기와 달리 갖춘 공리를 채우지 않는다
2. **하이퍼파라미터에 예민하다**: 잡음 크기와 표본 수에 열매가 달렸다
3. **셈 짐이 붙는다**: 앞으로-되짚기를 여러 번 해야 한다
4. **지나치게 매끄러워질 수 있다**: 중요한 잔 무늬를 잃을 수 있다
5. **밑바탕 문제를 풀지는 않는다**: 그림만 나아질 뿐 몫 매기기 자체의 됨됨이가 오르지는 않는다

## 참으로 쓸 때 이르는 말

### SmoothGrad을 쓸 때

**권한다:**

- 빠르게 그려 보고 벌레잡을 때
- 맨 기울기가 너무 잡음이 많아 풀이할 수 없을 때
- 다른 기울기 방법의 뒷손질로 쓸 때
- 발표하고 보일 때

**다른 것을 헤아려 볼 때:**

- 이론 보장이 중요할 때 → 쌓은 기울기
- 자리를 짚어야 할 때 → Grad-CAM
- 셈할 여유가 적을 때 → 맨 기울기

### 짜기 살핌 목록

1. **잡음 크기**: 0.15에서 시작해 열매를 보고 고친다
2. **표본 수**: 여느 살핌에는 50, 마지막 열매에는 100 넘게
3. **묶음으로 다루기**: 잘 들게 하려면 묶음 짜보기를 쓴다
4. **눈으로 살피기**: 맨 기울기와 견주어 매끄러워졌는지 본다

### 이미 있는 흐름에 끼워 넣기

```python
class SaliencyPipeline:
    """두드러짐 셈의 온 흐름."""

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
        정한 방법으로 두드러짐을 셈한다.

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
            raise ValueError(f"모르는 방법: {method}")
```

## 간추림

SmoothGrad은 기울기 바탕 두드러짐 그림의 잡음 문제에 손에 잡히는 답을 준다.

**고갱이 식:**

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$

**고갱이 깨침:**

- 잡음을 더하면 헛된 높은 잦기 기울기 신호가 지워진다
- 참된 중요함은 들임이 조금 달라져도 한결같다
- 가우스로 매끄럽게 한 모형의 기울기를 셈하는 것과 같다

**권하는 맡긴 값:**

- 잡음 크기: σ = 0.15
- 표본 수: n = 50

## 살펴볼 거리

1. Smilkov, D., et al. (2017). *SmoothGrad: Removing Noise by Adding Noise*. arXiv:1706.03825.

2. Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps*. NeurIPS.

3. Hooker, S., et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks*. NeurIPS.

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
