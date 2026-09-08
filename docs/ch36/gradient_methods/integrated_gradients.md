# 쌓은 기울기

**쌓은 기울기(IG)**는 맨 기울기 바탕 길이 지닌 밑바탕 걸림돌을 다루는, 이치에 닿는 몫 매기기 방법이다. 그 자리의 예민함만 주는 단순한 기울기와 달리, 쌓은 기울기는 **밑금에서 들임으로 가는 길을 따라 기울기를 쌓아** 몫을 셈한다.

순다라라잔, 탈리, 얀(2017)이 **예민함**과 **짜기에 흔들리지 않음**을 채우는 몫 매기기 방법을 만들겠다는 또렷한 뜻으로 들여왔다. 이 둘은 맨 기울기가 채우지 못하는 됨됨이다. 바라는 이론 공리 여럿을 한꺼번에 채우는, 널리 쓰이는 하나뿐인 방법으로 이름이 높다.

---

## 1. 왜 있어야 하는가

### 맨 기울기의 탈

단순한 ReLU 그물 $f(x) = \max(0, x - 1)$을 생각하자.

- $x = 2$이면 $f(2) = 1$, $\nabla f(2) = 1$(기울기가 있다)
- $x = 0$이면 $f(0) = 0$, $\nabla f(0) = 0$(기울기가 0이다)

이제 몫 매기기 물음을 던지자. **"왜 $f(2) = 1$인가?"**

맨 기울기는 $x$의 중요함이 1이라 한다. 그런데 밑금을 $x' = 0$으로 잡으면 어떨까?

차이 $f(2) - f(0) = 1 - 0 = 1$은 온전히 들임에 돌려야 하는데, 기울기만으로는 온 이바지를 담지 못한다. 기울기는 그 들임 자리에서의 **그 자리 예민함**만 잴 뿐이다.

**더 종요롭게는**: $x = 0.5$(문턱 바로 아래)이면 $f(0.5) = 0$이고 $\nabla f(0.5) = 0$이다. 들임 값이 내놓기가 0인지 아닌지를 곧바로 가르는데도 기울기는 이 들임의 중요함이 0이라고 우긴다!

### 잦아듦 문제

ReLU, 시그모이드, tanh 살림을 쓰는 신경 그물에는 기울기가 0이거나 0에 가까운 자리(잦아듦)가 있다. 그런 자리에서는

- 맨 기울기가 그 결의 중요함이 0이라고 한다
- 그런데도 그 결이 내놓기를 또렷이 흔든다

### 길 따라 쌓아 푸는 길

쌓은 기울기는 밑금에서 들임까지의 **온 길**을 따라 기울기를 적분해 이를 푼다.

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

사이 잡는 길을 따라 기울기 소식을 모두 쌓으므로 잦아든 자리를 지나면서도 이바지를 담아낸다.

---

## 2. 수학 밑바탕

### 길 적분 세움새

들임 $\mathbf{x} \in \mathbb{R}^n$, 밑금 $\mathbf{x}' \in \mathbb{R}^n$, 모형 $f: \mathbb{R}^n \rightarrow \mathbb{R}$에 대해 결 $i$의 쌓은 기울기 몫은 이렇다.

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

여기서

- $\mathbf{x}'$은 밑금(견줌 자리, 흔히 0이나 흐린 그림)
- $\alpha \in [0, 1]$은 $\mathbf{x}'$에서 $\mathbf{x}$으로 가는 곧은 길을 매긴다
- $\frac{\partial f}{\partial x_i}$은 $i$번째 들임 결에 대한 기울기
- $(x_i - x'_i)$은 결 $i$이 얼마나 멀리 갔는지로 쌓은 기울기의 잣대를 잡는다

### 길 매기기

밑금에서 들임으로 가는 길은 이렇다.

$$
\gamma(\alpha) = \mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'), \quad \alpha \in [0, 1]
$$

- $\alpha = 0$이면 $\gamma(0) = \mathbf{x}'$(밑금)
- $\alpha = 1$이면 $\gamma(1) = \mathbf{x}$(들임)

이 길 위의 자리마다 기울기를 셈한다. 적분이 이 기울기들을 쌓되, 결마다 밑금에서 들임까지 간 만큼으로 짐을 싣는다.

### 리만 합 어림

참으로는 걸음 $m$개의 리만 합으로 적분을 어림한다.

$$
\text{IG}_i(\mathbf{x}) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f\left(\mathbf{x}' + \frac{k}{m}(\mathbf{x} - \mathbf{x}')\right)}{\partial x_i}
$$

그물을 앞으로-되짚기로 $m$번 지나야 한다(다만 묶음으로 묶어 잘 들게 할 수 있다).

---

## 3. 공리로 본 됨됨이

쌓은 기울기는 순다라라잔 외가 어떤 몫 매기기 방법이든 채워야 한다고 따지는 밑바탕 공리를 채운다. **예민함과 짜기에 흔들리지 않음을 함께 채우는 하나뿐인 방법이다**(곧은 길을 쓰는 길 바탕 방법 가운데).

### 공리 1: 예민함

**말하면**: 들임과 밑금이 결 하나에서만 다르고 모형의 내놓기가 다르면, 그 결은 0이 아닌 몫을 받아야 한다.

**엄밀히**: $f(\mathbf{x}) \neq f(\mathbf{x}')$이고 $x_i \neq x'_i$이며 $j \neq i$마다 $x_j = x'_j$이면 $\text{IG}_i(\mathbf{x}) \neq 0$이다.

**맨 기울기가 왜 어기는가**: $x = 2$, $x' = 0$인 $f(x) = \text{ReLU}(x - 1)$을 보자. $x = 0.5$에서 따지면 (문턱 아래이므로) 기울기가 0이지만, $x$은 내놓기에 또렷이 걸린다.

**IG는 왜 채우는가**: 길을 따라 적분하므로 끝점에서 0이더라도 기울기가 0이 *아닌* 자리를 담아낸다.

### 공리 2: 짜기에 흔들리지 않음

**말하면**: 모든 들임에 같은 내놓기를 내는 두 그물은 안쪽 얼개가 어떻든 같은 몫을 받아야 한다.

**엄밀히**: 모든 $\mathbf{x}$에 대해 $f(\mathbf{x}) = g(\mathbf{x})$이면 $\text{IG}^f_i(\mathbf{x}) = \text{IG}^g_i(\mathbf{x})$이다.

**왜 중요한가**: 시그모이드를 $\sigma(x)$으로 짜든 $1 - \sigma(-x)$으로 짜든 함수가 수학으로 같으므로 몫이 달라져서는 안 된다.

**DeepLIFT은 왜 어기는가**: DeepLIFT은 셈 그래프를 따라 이바지를 퍼뜨리므로, 함수가 같아도 그래프 얼개가 다르면 다른 몫이 나올 수 있다.

### 따라 나오는 됨됨이: 온전함

뜻매김에서 따라 나오는 중요한 결과가 **온전함**(또는 **효율**) 됨됨이다.

$$
\sum_{i=1}^{n} \text{IG}_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}')
$$

몫이 들임에서의 모형 내놓기와 밑금에서의 내놓기의 차이를 **꼭 맞게 갈라 담는다**. 중요함이 "새거나" 없던 것이 생기지 않는다.

**증명:**

미적분의 밑정리와 사슬 규칙을 쓰면

$$
\sum_i \text{IG}_i = \sum_i (x_i - x'_i) \int_0^1 \frac{\partial f}{\partial x_i} d\alpha = \int_0^1 \nabla f \cdot (\mathbf{x} - \mathbf{x}') \, d\alpha = \int_0^1 \frac{d}{d\alpha} f(\gamma(\alpha)) \, d\alpha = f(\mathbf{x}) - f(\mathbf{x}')
$$

이 됨됨이는 풀이하기에 아주 값지다. 몫이 **손에 잡히는 뜻**을 지닌다는 말이니, 미루어 봄의 차이를 나누어 담는 것이다.

---

## 4. PyTorch 짜보기

### 함수로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

def compute_integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: Literal['zeros', 'blur', 'random', 'mean'] = 'zeros',
    steps: int = 50
) -> torch.Tensor:
    """
    쌓은 기울기 몫을 셈한다.

    Args:
        model: 따지는 결로 놓인 신경 그물
        image_tensor: 들임 그림 [1, C, H, W]
        target_class: 겨눈 갈래 번호
        device: 셈하는 장치
        baseline_type: 밑금의 갈래('zeros', 'blur', 'random', 'mean')
        steps: 사이를 잡는 걸음 수(많을수록 더 맞다)

    Returns:
        몫 그림 [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # 밑금을 만든다
    baseline = create_baseline(image_tensor, baseline_type, device)

    # 들임과 밑금의 차이
    delta = image_tensor - baseline  # (x - x')

    # 길을 따라 기울기를 쌓는다
    accumulated_gradients = torch.zeros_like(image_tensor)

    for step in range(1, steps + 1):
        # 사이 잡는 계수: α = k/m
        alpha = step / steps

        # 사이 들임: x' + α(x - x')
        interpolated = baseline + alpha * delta
        interpolated = interpolated.clone().detach().requires_grad_(True)

        # 앞으로 걸음
        output = model(interpolated)
        target_score = output[0, target_class]

        # 되짚기 걸음
        model.zero_grad()
        target_score.backward()

        # 쌓는다: 이 사이 자리에서의 ∂f/∂x
        accumulated_gradients += interpolated.grad

    # 기울기를 고르게 한다(리만 합 어림)
    avg_gradients = accumulated_gradients / steps  # (1/m) Σ_k ∇f

    # 들임 차이로 잣대를 잡는다: (x - x') * avg_gradients
    integrated_grads = delta * avg_gradients

    # 통로를 가로질러 모은다(그리려고 절댓값을 잡는다)
    attribution = torch.abs(integrated_grads)
    saliency = attribution.max(dim=1)[0]  # [1, H, W]

    return saliency

def create_baseline(
    image_tensor: torch.Tensor,
    baseline_type: str,
    device: torch.device
) -> torch.Tensor:
    """
    쌓은 기울기에 쓸 밑금을 만든다.

    Args:
        image_tensor: 들임 그림
        baseline_type: 'zeros', 'blur', 'random', 'mean' 가운데 하나
        device: 셈하는 장치

    Returns:
        들임과 꼴이 같은 밑금 텐서
    """
    if baseline_type == 'zeros':
        # 검은 그림(그림에 가장 흔히 고른다)
        baseline = torch.zeros_like(image_tensor)

    elif baseline_type == 'blur':
        # 들임을 세게 흐리게 한 것(낮은 잦기 얼개를 남긴다)
        from torchvision.transforms.functional import gaussian_blur
        baseline = gaussian_blur(image_tensor, kernel_size=51, sigma=20)

    elif baseline_type == 'random':
        # 아무렇게나 만든 잡음([0, 0.1]에 고루)
        baseline = torch.rand_like(image_tensor) * 0.1

    elif baseline_type == 'mean':
        # 자료의 평균(ImageNet으로 고르게 한 그림에 쓴다)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        mean = mean.view(1, 3, 1, 1).expand_as(image_tensor)
        baseline = mean

    elif baseline_type == 'max_entropy':
        # 엔트로피가 가장 큰 밑금(그림이면 잿빛)
        baseline = torch.ones_like(image_tensor) * 0.5

    else:
        raise ValueError(f"모르는 밑금 갈래: {baseline_type}")

    return baseline.to(device)
```

### 클래스로 짜기

```python
class IntegratedGradients:
    """
    쌓은 기울기 몫 매기기 방법.

    살펴볼 거리: Sundararajan et al., "Axiomatic Attribution for 
    Deep Networks"(ICML 2017)
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch 모형
        """
        self.model = model

    def attribute(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor = None,
        target_class: int = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False
    ) -> torch.Tensor:
        """
        쌓은 기울기 몫을 셈한다.

        Args:
            input_tensor: (1, *들임 차원) 꼴 들임 텐서
            baseline: 꼴이 같은 밑금 텐서. None이면 0을 쓴다.
            target_class: 몫을 매길 겨눈 갈래. None이면 argmax을 쓴다.
            n_steps: 적분 걸음 수(많을수록 더 맞다)
            return_convergence_delta: True이면 어림 어긋남도 내놓는다

        Returns:
            들임과 꼴이 같은 몫 텐서
        """
        self.model.eval()
        device = input_tensor.device

        # 맡긴 밑금: 0(그림이면 검은 그림)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # 겨눈 갈래를 정한다
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # 차이를 셈한다
        diff = input_tensor - baseline

        # 사이 걸음을 만든다: k = 1, ..., m에 대해 α_k = k/m
        scaled_inputs = [
            baseline + (float(i) / n_steps) * diff 
            for i in range(1, n_steps + 1)
        ]

        # 묶음으로 셈하려고 사이 들임을 모두 쌓는다
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)

        # 모든 걸음을 한꺼번에 앞으로 보낸다(묶음)
        outputs = self.model(scaled_inputs)

        # 겨눈 갈래 점수를 뽑는다
        target_scores = outputs[:, target_class]

        # 되짚기 걸음
        self.model.zero_grad()

        # 기울기를 셈한다
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=scaled_inputs,
            create_graph=False
        )[0]

        # 걸음에 걸쳐 기울기를 고르게 한다
        avg_grads = grads.mean(dim=0, keepdim=True)

        # 쌓은 기울기 = (들임 - 밑금) * 고른 기울기
        attributions = diff * avg_grads

        if return_convergence_delta:
            # 온전함을 살핀다: 몫의 합 ≈ f(x) - f(x')
            with torch.no_grad():
                f_x = self.model(input_tensor)[0, target_class]
                f_baseline = self.model(baseline)[0, target_class]
                expected_diff = f_x - f_baseline
                actual_sum = attributions.sum()
                delta = (expected_diff - actual_sum).abs().item()
            return attributions, delta

        return attributions

    def attribute_with_noise(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor = None,
        target_class: int = None,
        n_steps: int = 50,
        n_samples: int = 5,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        잡음에 든든한 쌓은 기울기(기댓값 기울기)를 셈한다.

        잡음 섞은 밑금 여럿에 걸쳐 IG를 고르게 해 더 든든한 몫을 낸다.
        이는 SHAP의 기댓값 기울기 세움새와 이어진다.
        """
        attributions = torch.zeros_like(input_tensor)

        for _ in range(n_samples):
            if baseline is None:
                noisy_baseline = noise_level * torch.randn_like(input_tensor)
            else:
                noisy_baseline = baseline + noise_level * torch.randn_like(baseline)

            attr = self.attribute(
                input_tensor, 
                noisy_baseline, 
                target_class, 
                n_steps
            )
            attributions += attr

        return attributions / n_samples
```

### 묶음으로 다듬은 짜보기

잘 들게 하려고 사이 걸음 여럿을 한 묶음으로 다룬다.

```python
def compute_integrated_gradients_batched(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: str = 'zeros',
    steps: int = 50,
    batch_size: int = 10
) -> torch.Tensor:
    """
    묶음으로 다듬은 쌓은 기울기 셈.

    묶음마다 사이 걸음 여럿을 다루어 잘 들게 한다.
    모든 걸음을 한꺼번에 다루는 것보다 GPU 기억을 덜 쓴다.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    baseline = create_baseline(image_tensor, baseline_type, device)
    delta = image_tensor - baseline

    accumulated_gradients = torch.zeros_like(image_tensor)

    # 알파 값을 만든다: [1/m, 2/m, ..., 1]
    alphas = torch.linspace(1/steps, 1, steps, device=device)

    # 묶음으로 다룬다
    for i in range(0, steps, batch_size):
        batch_alphas = alphas[i:i+batch_size]
        current_batch_size = len(batch_alphas)

        # 사이 들임의 묶음을 만든다
        # 꼴: [batch_size, C, H, W]
        batch_alphas = batch_alphas.view(-1, 1, 1, 1)
        interpolated_batch = baseline + batch_alphas * delta
        interpolated_batch.requires_grad_(True)

        # 묶음을 앞으로 보낸다
        outputs = model(interpolated_batch)  # [batch_size, num_classes]

        # 기울기를 셈하려고 겨눈 점수를 더한다
        target_scores = outputs[:, target_class].sum()

        # 되짚기 걸음
        model.zero_grad()
        target_scores.backward()

        # 기울기를 쌓는다(묶음 축으로 더한다)
        accumulated_gradients += interpolated_batch.grad.sum(dim=0, keepdim=True)

    # 고르게 하고 잣대를 잡는다
    avg_gradients = accumulated_gradients / steps
    integrated_grads = delta * avg_gradients

    # 통로를 가로질러 모은다
    saliency = torch.abs(integrated_grads).max(dim=1)[0]

    return saliency
```

---

## 5. 밑금 고르기

밑금을 어떻게 고르느냐가 몫을 크게 가른다. 밑금은 "치우친 데 없는" 또는 "소식이 없는" 들임을 나타낸다.

### 흔히 고르는 밑금

| 밑금 | 밝힘 | 알맞은 자리 |
|----------|-------------|----------|
| **0(검정)** | 온통 0인 텐서 | 그림(가장 흔하다) |
| **흐림** | 세게 흐리게 한 들임 | 얼개를 남길 때 |
| **아무렇게나** | 고른 아무 잡음 | 모둠으로 고르게 할 때 |
| **평균** | 자료의 평균값 | 고르게 한 들임 |
| **엔트로피 가장 큼** | 0.5(그림이면 잿빛) | 아리송함이 가장 클 때 |

### 밑금 견주기

```python
def compare_baselines(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baselines: list = ['zeros', 'blur', 'random', 'mean']
):
    """밑금을 달리한 쌓은 기울기를 견준다."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(baselines) + 1, figsize=(4 * (len(baselines) + 1), 8))

    # 본디 그림
    image_np = denormalize_image(image_tensor)
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('본디')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for idx, baseline_type in enumerate(baselines):
        # IG를 셈한다
        attr = compute_integrated_gradients(
            model, image_tensor, target_class, device,
            baseline_type=baseline_type, steps=50
        )
        attr_np = attr.squeeze().cpu().numpy()

        # 밑금을 보인다
        baseline = create_baseline(image_tensor, baseline_type, device)
        baseline_np = denormalize_image(baseline)
        axes[0, idx + 1].imshow(baseline_np)
        axes[0, idx + 1].set_title(f'{baseline_type.capitalize()} 밑금')
        axes[0, idx + 1].axis('off')

        # 몫을 보인다
        axes[1, idx + 1].imshow(attr_np, cmap='hot')
        axes[1, idx + 1].set_title('몫')
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    return fig
```

### 밭마다의 밑금 길잡이

| 밭 | 권하는 밑금 | 까닭 |
|--------|---------------------|-----------|
| 그림(RGB) | 0(검정) | 눈에 보이는 소식이 없음 |
| 글(쏘아 넣기) | 채움 낱말의 쏘아 넣기 | 치우친 데 없는 낱말 |
| 표 자료 | 익힘 자료의 평균 | 고른 결 값 |
| 때 열 | 0이나 지난날의 평균 | 밑금 움직임 켜 |
| 소리 | 고요(0) | 소리가 없음 |

---

## 6. 온전함 살피기

IG의 됨됨이를 살피는 종요로운 길은 온전함을 따져 보는 것이다.

```python
def verify_completeness(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: str = 'zeros',
    steps: int = 50
) -> dict:
    """
    IG 몫의 합이 f(x) - f(x')이 되는지 살핀다.

    Returns:
        온전함 살핌 열매를 담은 사전
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    baseline = create_baseline(image_tensor, baseline_type, device)

    # 모형 내놓기를 셈한다
    with torch.no_grad():
        output_input = model(image_tensor)[0, target_class].item()
        output_baseline = model(baseline)[0, target_class].item()

    output_difference = output_input - output_baseline

    # IG를 셈한다(더하려고 통로 축을 남긴다)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(image_tensor, baseline, target_class, steps)

    attribution_sum = attributions.sum().item()

    # 어긋남을 셈한다
    absolute_error = abs(attribution_sum - output_difference)
    relative_error = absolute_error / (abs(output_difference) + 1e-8)

    return {
        'f(x)': output_input,
        'f(x\')': output_baseline,
        'f(x) - f(x\')': output_difference,
        'sum(IG)': attribution_sum,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'completeness_satisfied': relative_error < 0.05  # 5% 너그러움
    }
```

---

## 7. 걸음 수 살피기

적분 걸음의 수가 어림의 맞음을 가른다.

```python
def analyze_steps_convergence(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    step_counts: list = [5, 10, 20, 50, 100, 200, 500]
):
    """걸음이 늘수록 몫이 어떻게 모이는지 살핀다."""
    import matplotlib.pyplot as plt

    errors = []

    for steps in step_counts:
        result = verify_completeness(
            model, image_tensor, target_class, device, steps=steps
        )
        errors.append(result['relative_error'])
        print(f"걸음: {steps:4d}, 견준 어긋남: {result['relative_error']:.6f}")

    # 모여 가는 모습을 그린다
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('걸음 수', fontsize=12)
    plt.ylabel('견준 온전함 어긋남', fontsize=12)
    plt.title('쌓은 기울기가 모여 가는 모습', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

    return dict(zip(step_counts, errors))
```

### 이르는 말

| 걸음 | 쓰일 자리 | 흔한 어긋남 |
|-------|----------|---------------|
| 20~30 | 빠르게 둘러보기, 벌레잡기 | ~5~10% |
| 50 | 여느 쓰임, 넉넉한 맞음 | ~1~5% |
| 100~200 | 논문, 엄밀한 살핌 | 1% 아래 |
| 300 넘음 | 온전함 어긋남이 클 때 | 0.5% 아래 |

---

## 8. 그림으로 보이기

### 여느 그림 그리기

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_integrated_gradients(
    image: np.ndarray,
    attribution: np.ndarray,
    title: str = "쌓은 기울기"
):
    """
    쌓은 기울기 몫을 그린다.

    Args:
        image: [0, 1] 너비의 본디 그림 [H, W, 3]
        attribution: 몫 그림 [H, W]
        title: 그림 이름
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 본디 그림
    axes[0].imshow(image)
    axes[0].set_title('본디')
    axes[0].axis('off')

    # 부호 있는 몫(갈라지는 빛깔 그림)
    vmax = np.abs(attribution).max()
    im = axes[1].imshow(attribution, cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1].set_title('부호 있는 몫')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 절댓값 몫
    axes[2].imshow(np.abs(attribution), cmap='hot')
    axes[2].set_title('절댓값 몫')
    axes[2].axis('off')

    # 겹쳐 보이기
    overlay = image.copy()
    mask = np.abs(attribution)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    axes[3].imshow(image)
    axes[3].imshow(mask, cmap='jet', alpha=0.5)
    axes[3].set_title('겹쳐 보이기')
    axes[3].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return fig
```

---

## 9. 금융에 쓰기

### 표 자료의 결 몫 매기기

```python
def ig_for_tabular(
    model: nn.Module,
    features: torch.Tensor,
    feature_names: list,
    target_class: int = None,
    baseline_type: str = 'zero',
    device: torch.device = None
):
    """
    표로 된 금융 자료에 쌓은 기울기를 건다.

    Args:
        model: 신용 무릅씀이나 돌아옴을 미루어 보는 모형
        features: 들임 결 텐서 [1, n_features]
        feature_names: 결 이름 목록
        target_class: 겨눈 갈래(가름일 때)
        baseline_type: 'zero'나 'mean'

    Returns:
        결마다의 몫 값
    """
    if device is None:
        device = features.device

    ig = IntegratedGradients(model)

    # 밑금
    if baseline_type == 'zero':
        baseline = torch.zeros_like(features)
    else:
        # 익힘 자료의 평균을 쓴다(참으로는 넘겨받아야 한다)
        baseline = features.mean(dim=0, keepdim=True)

    # 몫을 셈한다
    attributions = ig.attribute(features, baseline, target_class, n_steps=100)

    # 값을 뽑는다
    attr_values = attributions.squeeze().cpu().numpy()

    # 몫의 절댓값으로 줄 세운다
    sorted_idx = np.argsort(np.abs(attr_values))[::-1]

    print("결 몫 매기기(|몫|으로 줄 세움):")
    print("-" * 50)
    for i in sorted_idx[:15]:
        print(f"{feature_names[i]:30s}: {attr_values[i]:+.4f}")

    return attr_values, feature_names

# 보기: 신용 무릅씀 모형
feature_names = [
    'credit_score', 'debt_to_income', 'loan_amount', 
    'employment_years', 'num_credit_lines', 'payment_history',
    'total_debt', 'income', 'age', 'months_since_delinquent'
]

# 신용 부도를 미루어 보는 모형이면:
# attributions, names = ig_for_tabular(credit_model, applicant_features, feature_names)
```

### 때 열 몫 매기기

```python
def ig_for_time_series(
    model: nn.Module,
    sequence: torch.Tensor,
    baseline: torch.Tensor = None,
    target_class: int = None,
    n_steps: int = 100
):
    """
    금융 때 열에 쌓은 기울기를 건다.

    Args:
        model: 열 모형(LSTM, 변환기 등)
        sequence: 들임 열 (묶음, 열 길이, 결)이나 (묶음, 열 길이)
        baseline: 밑금 열(맡긴 값: 0)
        target_class: 가름 모형의 겨눈 갈래

    Returns:
        어느 때 걸음이 중요한지 보이는 때 몫
    """
    import matplotlib.pyplot as plt

    ig = IntegratedGradients(model)

    # 맡긴 밑금: 0
    if baseline is None:
        baseline = torch.zeros_like(sequence)

    # 몫을 셈한다
    attributions = ig.attribute(sequence, baseline, target_class, n_steps=n_steps)

    # 때 걸음마다 결이 여럿이면 결 축으로 더한다
    if attributions.dim() == 3:
        temporal_attr = attributions.abs().sum(dim=-1).squeeze()
    else:
        temporal_attr = attributions.abs().squeeze()

    # 그림으로 보인다
    temporal_attr_np = temporal_attr.cpu().numpy()

    plt.figure(figsize=(14, 4))
    plt.bar(range(len(temporal_attr_np)), temporal_attr_np, color='steelblue')
    plt.xlabel('때 걸음 (가장 오래됨 → 가장 최근)', fontsize=12)
    plt.ylabel('몫', fontsize=12)
    plt.title('때 몫 매기기: 어느 때 걸음이 미루어 봄을 흔드는가?', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return temporal_attr_np
```

---

## 10. 다른 방법과 견주기

### 이론에서 견주기

| 방법 | 온전함 | 예민함 | 짜기에 흔들리지 않음 | 셈 |
|--------|:------------:|:-----------:|:----------------:|:-----------:|
| 맨 기울기 | ✗ | ✗ | ✓ | 빠름 |
| 기울기 × 들임 | ✗ | ✗ | ✓ | 빠름 |
| 쌓은 기울기 | ✓ | ✓ | ✓ | 가운데 |
| DeepLIFT | ✓ | ✓ | ✗ | 빠름 |
| SHAP(정확) | ✓ | ✓ | ✓ | 느림 |
| LRP | ✓ | 얼마쯤 | ✗ | 가운데 |

### 참으로 쓸 때 견주기

| 결 | 맨 기울기 | 쌓은 기울기 |
|--------|-----------------|---------------------|
| 셈 | 앞으로-되짚기 한 번 | 여러 번(걸음 m번) |
| 예민함 공리 | ❌ 어김 | ✅ 채움 |
| 온전함 | ❌ 보장 없음 | ✅ 꼭 맞음(어림 안에서) |
| 잦아듦 다루기 | ❌ 나쁨 | ✅ 좋음 |
| 읽는 법 | 그 자리 예민함 | 길 따라 쌓은 이바지 |
| 밑금이 있어야 함 | 아니오 | 예 |

### 다른 방법과 아우르기

**IG + SmoothGrad:** 잡음 섞은 들임에 걸쳐 IG를 고르게 해 더 매끄러운 그림을 얻는다
```python
# 매끄럽게 한 쌓은 기울기
smooth_ig = ig.attribute_with_noise(input_tensor, n_samples=10, noise_level=0.1)
```

**IG + Grad-CAM:** 그림점 낱의 잔 무늬에는 IG를, 자리 낱으로 알아보는 데는 Grad-CAM을 쓴다

---

## 11. 한계

### 1. 밑금에 매임

몫이 밑금을 어떻게 고르느냐에 달렸다. 같은 들임-내놓기 짝이라도 밑금이 다르면 뜻있게 다른 몫이 나올 수 있다.

**눅이는 길:**

- 밑금을 여럿 써서 고르게 한다(기댓값 기울기 / SHAP)
- 그 밭에 알맞은 밑금을 쓴다
- 논문에 고른 밑금을 적는다

### 2. 셈 값

앞으로-되짚기를 $m$번(흔히 50~200) 해야 하므로 맨 기울기보다 훨씬 느리다.

**눅이는 길:**

- 묶음으로 셈한다
- 둘러볼 때는 걸음을 적게 잡는다
- 모여 가는 결을 보고 일찍 멈춘다

### 3. 길 고르기

곧은 길은 하나의 고름일 뿐이다. 다른 길도 공리를 채울 수 있다.

$$
\gamma: [0, 1] \rightarrow \mathbb{R}^n, \quad \gamma(0) = \mathbf{x}', \gamma(1) = \mathbf{x}
$$

**짚을 것:** 곧은 길이 가장 자연스럽고 널리 쓰이며, 맞섬을 지키는 됨됨이를 홀로 채운다.

### 4. 그 자리에서 선형이라는 여김

IG는 모형이 자리마다 거의 선형일 때 가장 잘 듣는다. 몹시 선형이 아닌 자리에서는 얄궂은 몫이 나올 수 있다.

---

## 12. 참으로 쓸 때 이르는 말

1. 그림에는 **0 밑금에서 시작한다**. 단순하고 대개 잘 듣는다

2. 여느 살핌에는 **걸음 50**을 쓰고, 엄밀한 일에는 100 넘게 올린다

3. 셈이 맞는지 보려면 **온전함을 살핀다**(어긋남 5% 아래)

4. 서로 채워 주는 눈으로 **Grad-CAM과 견준다**(IG는 그림점 낱, Grad-CAM은 자리 낱)

5. 그림이 크거나 풀이를 많이 해야 하면 **묶음 셈을 헤아린다**

6. 밑금이 아리송하면 **밑금에 걸쳐 고르게 한다**(기댓값 기울기)

7. 다시 해 볼 수 있도록 **밑금과 걸음 수를 적는다**

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

쌓은 기울기는 밑금에서 들임으로 가는 길을 따라 기울기를 적분해 **이치에 닿고 공리에 뿌리내린 몫 매기기**를 준다.

### 고갱이 식

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

### 고갱이 됨됨이

| 됨됨이 | 밝힘 |
|----------|-------------|
| **예민함** | 내놓기를 흔드는 결에 0이 아닌 몫을 준다 |
| **짜기에 흔들리지 않음** | 함수가 같으면 몫도 같다 |
| **온전함** | $\sum_i \text{IG}_i = f(\mathbf{x}) - f(\mathbf{x}')$ |

### 알맞은 자리

- 엄밀하고 이론에 뿌리내린 몫 매기기
- 맨 기울기가 무너지는 자리(잦아듦, ReLU 그물)
- 온전함 됨됨이가 중요할 때
- 이치에 닿는 풀이를 바라는 규정이나 감사 자리
- 서로 다른 모형의 몫을 고르게 견줄 때

**살펴볼 거리**

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." *ICML 2017*.

2. Sturmfels, P., Lundberg, S., & Lee, S. I. (2020). "Visualizing the Impact of Feature Attribution Baselines." *Distill*.

3. Kapishnikov, A., et al. (2019). "XRAI: Better Attributions Through Regions." *ICCV 2019*.

4. Mudrakarta, P. K., et al. (2018). "Did the Model Understand the Question?" *ACL 2018*.

5. Erion, G., et al. (2021). "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients." *Nature Machine Intelligence*.
