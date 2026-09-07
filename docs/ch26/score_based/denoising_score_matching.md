# 잡음 없애는 점수 맞추기
잡음 없애는 점수 맞추기(Vincent, 2011)는 드러난 점수 맞추기의 비싼 헤세 대각합을 아는 목표에 대한 단순한 되돌아 기대기 손실로 바꾼다. 잡음으로 흔든 분포의 점수를 배워 셈 비용 문제와 밀도 낮은 자리 어림 문제를 함께 푼다. 이는 요즘 모든 퍼짐 모델의 바탕이 되는 익히기 목표이다.

## 왜 필요한가

드러난 점수 맞추기에는 실제로 결정적인 한계가 둘 있다.

| 문제 | 설명 | 영향 |
|---------|-------------|--------|
| **셈 비용** | 라플라시안 $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta)$은 뒤먹임 $D$번이 필요하다 | 그림에서는 쓸 수 없다($D > 10^4$) |
| **밀도 낮은 자리 어림** | $p_{\text{data}}(\mathbf{x}) \approx 0$인 곳에서 점수 어림을 믿기 어렵다 | 봉우리 사이에서 뽑기가 실패한다 |

잡음 없애는 점수 맞추기는 **잡음으로 흔든** 분포의 점수를 배워 둘 다 푼다.

## 흔든 분포

깨끗한 자료 $\mathbf{x} \sim p_{\text{data}}$과 정규 잡음 알맹이가 주어질 때

$$q(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}} \,|\, \mathbf{x}, \, \sigma^2 \mathbf{I})$$

잡음 섞인 표본의 가장자리 분포는 다음과 같다.

$$q_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \, q(\tilde{\mathbf{x}} | \mathbf{x}) \, d\mathbf{x}$$

이 매끄럽게 한 분포는 받침이 온전해(어디서나 뜻매김된다) 자료 다양체에서 멀어도 점수가 잘 뜻매김된다.

## 아는 목표 점수

핵심 통찰은 잡음 알맹이의 점수를 닫힌 꼴로 얻을 수 있다는 것이다.

$$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}$$

$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$일 때 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}$으로 적으면 이는 $-\boldsymbol{\epsilon} / \sigma$이 된다. 잡음 섞인 점마다 목표 점수는 그저 깨끗한 자료 쪽을 가리키며 잡음 수준의 역수로 잣수가 맞춰진다.

## 잡음 없애는 점수 맞추기의 목표

**정리(Vincent, 2011).** 흔든 분포 $q_\sigma$의 가장 좋은 점수 신경망은 다음을 가장 작게 하여 찾는다.

$$\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \, \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \, \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot | \mathbf{x})} \!\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x})\|^2\right]$$

아는 목표를 넣으면:

$$\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]}$$

익히기가 되돌아 기대기 문제가 된다. 곧 잡음 섞인 표본마다 음의 잡음 방향을 헤아린다.

## 드러난 점수 맞추기와 같음

잡음 없애는 점수 맞추기와 드러난 점수 맞추기의 목표는 $\theta$에 매이지 않는 상수만큼만 다르다.

$$\mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{ESM}}(\theta;\, q_\sigma) + C_\sigma$$

밝힘은 $\mathbf{s}_\theta$과 $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})$ 사이의 피셔 벌어짐을 펼친 뒤 온 기댓값 법칙으로 가장자리 점수 $\nabla \log q_\sigma$을 조건부 점수 $\nabla \log q(\tilde{\mathbf{x}} | \mathbf{x})$으로 바꾸어 나아간다. 드러난 점수 맞추기에서 $\mathbf{s}_{\text{data}}$을 없애는 어긋 항이 여기서는 아는 조건부 점수로 바뀌어 부분 적분 없이 같은 효과를 낸다.

$\sigma \to 0$이면 흔든 분포가 자료 분포로 모이므로 $\lim_{\sigma \to 0} \mathcal{L}_{\text{DSM}} = \mathcal{L}_{\text{ESM}} + C$이다. 따라서 $\sigma$이 작은 잡음 없애는 점수 맞추기는 야코비를 전혀 셈하지 않고 드러난 점수 맞추기를 어림한다.

## 여러 잣수의 잡음 없애는 점수 맞추기

### 잣수 하나의 한계

잡음 수준 $\sigma$ 하나는 치우침과 덮기의 맞바꿈을 낳는다. $\sigma$이 작으면 자료 가까이의 자잘한 세부를 담지만 밀도 낮은 자리를 잘 덮지 못한다. $\sigma$이 크면 공간을 채우지만 자료 짜임을 무너뜨린다.

### 잡음 조건 점수 신경망(NCSN)

풀이는 잡음 수준 범위 $\{\sigma_i\}_{i=1}^L$에 걸쳐 점수를 한꺼번에 배우는 것이다.

$$\mathcal{L}_{\text{NCSN}}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \!\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma_i \boldsymbol{\epsilon},\, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\|^2\right]$$

점수 신경망 $\mathbf{s}_\theta(\cdot, \sigma)$은 잡음 수준을 조건으로 삼으므로 $\sigma$으로 어깨수를 매긴 점수 무리를 배운다. $\sigma$이 크면 온마당 짜임을 주고 작으면 자잘한 세부를 담는다. 뽑는 동안(식힘 랑주뱅 움직임) 잡음 수준을 차츰 낮춘다.

### 잡음 차례표

여느 고르기는 **등비 차례표**이다.

$$\sigma_i = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{(i-1)/(L-1)}, \quad i = 1, \ldots, L$$

흔한 값: $\sigma_{\max} \approx$ 자료의 짝마다 최대 거리, $\sigma_{\min} \approx 0.01$, $L \in [10, 1000]$.

## 무게 매기기 방책

$\lambda(\sigma)$을 어떻게 고르느냐가 익히기 움직임에 영향을 준다.

| 무게 | 공식 | 까닭 |
|-----------|---------|-----------|
| 고르게 | $\lambda = 1$ | 바탕. 잡음 낮은 잣수가 기울기를 도맡는다 |
| $\sigma^2$ | $\lambda = \sigma^2$ | 잣수에 걸쳐 기울기 크기를 고르게 한다(NCSN 기본값) |
| 신호 대 잡음비 | $\lambda = 1/(1 + \sigma^2)$ | 자잘한 세부를 위해 잡음 낮은 자리를 무겁게 본다 |

### 시그마 제곱 무게가 통하는 까닭

점수 크기가 $\|\mathbf{s}\| \sim 1/\sigma$으로 커지므로 잣수 $\sigma$에서 무게 없는 손실은 $O(1/\sigma^2)$이다. $\sigma^2$을 곱하면 몫이 고르게 맞춰진다.

$$\sigma^2 \left\|\mathbf{s}_\theta + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2 = \|\sigma \, \mathbf{s}_\theta + \boldsymbol{\epsilon}\|^2$$

이는 잡음 $\boldsymbol{\epsilon}$을 곧바로 헤아리는 것과 같으며 바로 DDPM의 익히기 목표이다.

## 퍼짐 모형과의 이음

### 잡음 없애는 점수 맞추기가 곧 DDPM 익히기이다

DDPM의 목표는 다음과 같다.

$$\mathcal{L}_{\text{DDPM}}(t) = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon}\|^2\right]$$

여기서 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$이다. 다음 대응 아래 이는 바로 $\sigma^2$ 무게를 준 잡음 없애는 점수 맞추기이다.

| DDPM | 잡음 없애는 점수 맞추기 |
|------|-----|
| 잡음 헤아리개 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | $\mathbf{s}_\theta = -\boldsymbol{\epsilon}_\theta / \sqrt{1 - \bar{\alpha}_t}$ |
| 앞 과정 $\mathbf{x}_t$ | 잡음 섞인 표본 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$ |
| 잡음 차례표 $\bar{\alpha}_t$ | $\sigma_t = \sqrt{(1 - \bar{\alpha}_t) / \bar{\alpha}_t}$ |

세 틀, 곧 잡음 없애는 점수 맞추기와 DDPM과 점수 바탕 확률 미분 방정식은 모두 같은 수학의 것을 그린다.

## 다른 잡음 알맹이

정규 잡음이 여느 것이지만 다른 알맹이도 쓸모 있을 수 있다.

**라플라스 알맹이** $q(\tilde{\mathbf{x}} | \mathbf{x}) = \text{Laplace}(\tilde{\mathbf{x}} | \mathbf{x}, b)$: 목표 점수는 $-\text{sign}(\tilde{\mathbf{x}} - \mathbf{x})/b$이다. 꼬리가 두꺼운 자료에 쓸모 있다.

**스튜던트 $t$ 알맹이** $q(\tilde{\mathbf{x}} | \mathbf{x}) = t_\nu(\tilde{\mathbf{x}} | \mathbf{x}, \sigma^2)$: 목표 점수는 $-(\nu + D)(\tilde{\mathbf{x}} - \mathbf{x}) / (\nu\sigma^2 + \|\tilde{\mathbf{x}} - \mathbf{x}\|^2)$이다. 동떨어진 값에 튼튼하다.

## PyTorch 구현

### 기본 잡음 없애는 점수 맞추기 손실

```python
import torch
import torch.nn as nn


def dsm_loss(score_net: nn.Module, x: torch.Tensor,
             sigma: float) -> torch.Tensor:
    """잡음 수준 하나의 잡음 없애는 점수 맞추기 손실.

    인수:
        score_net: [batch, dim]을 [batch, dim]으로 보내는 그물 s(x).
        x: 깨끗한 자료 [batch, dim].
        sigma: 잡음의 표준 편차.

    반환값:
        낱값 손실.
    """
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy)
    target = -noise / sigma
    return 0.5 * ((score - target)**2).sum(dim=-1).mean()
```

### 에너지 바탕 잡음 없애는 점수 맞추기 손실

모델을 곧바로 점수 신경망으로 두지 않고 에너지 함수 $E_\theta(\mathbf{x})$으로 매개변수화할 때:

```python
def dsm_loss_energy(energy_net: nn.Module, x: torch.Tensor,
                    sigma: float) -> torch.Tensor:
    """에너지 바탕 모델의 잡음 없애는 점수 맞추기 손실.

    점수는 자동 미분으로 ∇_x E_θ(x)으로 셈한다.

    인수:
        energy_net: 낱값 에너지 신경망 E(x).
        x: 깨끗한 자료 [batch, dim].
        sigma: 잡음의 표준 편차.

    반환값:
        낱값 손실.
    """
    noise = torch.randn_like(x)
    x_noisy = (x + sigma * noise).requires_grad_(True)

    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        energy.sum(), x_noisy, create_graph=True
    )[0]

    target = -noise / sigma
    return ((score - target)**2).sum(dim=1).mean()
```

### 여러 잣수의 잡음 없애는 점수 맞추기(NCSN 꼴)

```python
def multi_scale_dsm_loss(
    score_net: nn.Module, x: torch.Tensor,
    sigmas: list[float], weights: list[float] | None = None
) -> torch.Tensor:
    """기본으로 σ² 무게를 준 여러 잣수의 잡음 없애는 점수 맞추기 손실.

    인수:
        score_net: 잡음 조건을 갖춘 신경망 s(x, sigma).
        x: 깨끗한 자료 [batch, dim].
        sigmas: 잡음 수준.
        weights: 수준마다 무게(기본: σ²으로 고르게 맞춘 것).

    반환값:
        잣수에 걸친 무게 있는 잡음 없애는 점수 맞추기 손실.
    """
    if weights is None:
        raw = [s**2 for s in sigmas]
        total = sum(raw)
        weights = [w / total for w in raw]

    loss = torch.tensor(0.0, device=x.device)
    for sigma, w in zip(sigmas, weights):
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise
        sigma_t = torch.full((x.shape[0], 1), sigma, device=x.device)
        score = score_net(x_noisy, sigma_t)
        target = -noise / sigma
        loss = loss + w * 0.5 * ((score - target)**2).sum(dim=-1).mean()

    return loss
```

### 효율 좋은 아무 시그마 익히기

```python
def dsm_loss_random_sigma(
    score_net: nn.Module, x: torch.Tensor,
    sigma_min: float = 0.01, sigma_max: float = 10.0
) -> torch.Tensor:
    """표본마다 아무 시그마(로그 고른 분포)를 쓴 잡음 없애는 점수 맞추기 손실.

    모든 시그마 수준을 되풀이하는 것보다 효율이 좋다.
    """
    log_sigma = (
        torch.rand(x.shape[0], 1, device=x.device)
        * (torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min)))
        + torch.log(torch.tensor(sigma_min))
    )
    sigma = torch.exp(log_sigma)

    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy, sigma)
    target = -noise / sigma

    # σ² 무게를 준 손실
    return 0.5 * (sigma**2 * (score - target)**2).sum(dim=-1).mean()
```

### 잡음 조건을 갖춘 점수 신경망

```python
class ScoreNet(nn.Module):
    """잡음 수준을 조건으로 삼는 여러 층 신경망 점수 신경망."""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """인자: x [batch, dim], sigma [batch, 1]."""
        inp = torch.cat([x, torch.log(sigma) / 4.0], dim=-1)
        return self.net(inp)
```

### 익히기 되풀이

```python
import numpy as np

def train_dsm(data: torch.Tensor, n_epochs: int = 5000,
              batch_size: int = 256, lr: float = 1e-3):
    """아무 시그마 잡음 없애는 점수 맞추기로 점수 신경망을 익힌다."""
    model = ScoreNet(data_dim=data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        idx = torch.randperm(len(data))[:batch_size]
        batch = data[idx]

        loss = dsm_loss_random_sigma(model, batch,
                                     sigma_min=0.01, sigma_max=5.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    return model
```

## 잡음 수준 고르기 길잡이

1. 먼저 자료를 평균 0, 흩어짐 1으로 **고르게 맞춘다**.
2. **잣수 하나의 잡음 없애는 점수 맞추기:** 고르게 맞춘 자료에서 $\sigma \approx 0.5$, 또는 가장 가까운 이웃 거리 가운데값의 $0.5$배.
3. **여러 잣수의 잡음 없애는 점수 맞추기:** $\sigma_{\min} = 0.01$에서 $\sigma_{\max} \approx \max_{i,j} \|\mathbf{x}_i - \mathbf{x}_j\|$까지의 등비 차례표.

## 요약

| 항목 | 설명 |
|--------|-------------|
| **목표** | $\frac{1}{2}\mathbb{E}[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \boldsymbol{\epsilon}/\sigma\|^2]$ |
| **목표 값** | $-\boldsymbol{\epsilon}/\sigma$(음의 잡음 방향) |
| **비용** | 앞먹임 + 뒤먹임 한 번(헤세 없음) |
| **같음** | $\sigma \to 0$이면 $\mathcal{L}_{\text{DSM}} \to \mathcal{L}_{\text{ESM}}$ |
| **여러 잣수** | NCSN: 여러 $\sigma$ 수준의 점수를 배운다 |
| **퍼짐과의 이음** | DDPM 익히기 = $\sigma^2$ 무게를 준 잡음 없애는 점수 맞추기 |

## 참고 문헌

1. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., & Ermon, S. (2020). "Improved Techniques for Training Score-Based Generative Models." *NeurIPS*.
4. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.

## 연습문제

1. **라플라스 알맹이 목표.** $q(\tilde{x}|x) = \frac{1}{2b}e^{-|\tilde{x}-x|/b}$에서 $\nabla_{\tilde{x}} \log q = -\text{sign}(\tilde{x} - x)/b$을 이끌어 내라.

2. **여러 잡음 실험.** 2차원 스위스 롤에서 $\sigma \in \{0.1, 0.3, 0.5, 1.0\}$ 하나씩으로 모델을 익혀라. 배운 점수 마당을 그리고 랑주뱅 움직임으로 표본 품질을 견주어라.

3. **무게 연구.** 여러 잣수의 잡음 없애는 점수 맞추기에서 고른 무게와 $\sigma^2$ 무게를 견주어라. 모인 뒤 잣수마다 손실을 재라.

4. **DDPM과 같음.** $\mathcal{L}_{\text{DDPM}}$에서 시작해 같은 뜻의 잡음 없애는 점수 맞추기 적기를 이끌어 내라. $\bar{\alpha}_t$과 $\sigma_t$의 대응을 보여라.
