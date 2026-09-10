# DDPM 뽑기

익힌 DDPM에서 뽑기는 배운 뒤 과정을 거쳐 순수한 가우스 잡음을 거듭 걷어 내며 자료를 만든다. $x_T \sim \mathcal{N}(0, I)$에서 비롯해 걸음마다 모형의 잡음 예측 $\epsilon_\theta(x_t, t)$을 써서 잡음이 덜한 $x_{t-1}$을 셈하고, $T$ 걸음 뒤 깨끗한 표본 $x_0$이 나올 때까지 짜임을 차츰 되살린다. 이 마디는 사후 분포에서 뒤 뽑기 식을 이끌어 내고, 흔히 쓰는 흩어짐 매개변수화 둘을 짚고, 분류기 이끎과 분류기 없는 이끎을 갖춘 온전한 PyTorch 짜보기를 보이며, [DDIM](../ddim/fundamentals.md)의 빠른 뽑기 방법이 나오게 된 셈 값을 살핀다.

---

## 1. 뒤 과정 이끌어 내기

### 1.1 뒤 조건부 분포

앞 과정은 $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$에 따라 잡음을 더한다. 표본을 만들려면 그 반대가 필요하다.

$$q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\, q(x_0 \mid x_t)\, dx_0$$

$q(x_0 \mid x_t)$이 알 수 없는 자료 분포에 매이므로 이는 다룰 수 없다. 그러나 **$x_0$을 조건으로 한 사후 분포**는 다룰 수 있다.

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I)$$

여기서 정규 조건부 분포에 베이즈 정리를 쓰면:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)}\, \beta_t$$

여기서 $\alpha_t = 1 - \beta_t$이고 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$이다.

### 1.2 x_0 헤아리기에서 엡실론 헤아리기로

앞 과정이 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$을 주므로 $x_0$을 $x_t$과 $\epsilon$으로 적을 수 있다.

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon}{\sqrt{\bar{\alpha}_t}}$$

$\epsilon$ 자리에 모델의 잡음 헤아림 $\epsilon_\theta(x_t, t)$을 넣고 $\tilde{\mu}_t$에 넣으면:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

이것이 배운 뒤 걸음 $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$의 평균이다.

### 1.3 x_0 헤아리기 관점

같은 말로 모델은 은근히 $x_0$을 헤아린다.

$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

그리고 평균은 다음과 같다.

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, \hat{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

$\mu_\theta$을 셈하기 앞에 $\hat{x}_0 \in [-1, 1]$으로 자르면(이 범위로 고르게 맞춘 자료에서) 뽑기가 안정되고 표본 품질이 나아진다. 특히 $x_0$ 헤아림이 잡음 섞인 앞 때 걸음에서 그렇다.

---

## 2. 흩어짐 매개변수화

### 2.1 붙박인 흩어짐 고르기

본디 DDPM 논문은 붙박인 뒤 흩어짐 $\sigma_t^2$을 쓴다. 자연스러운 고르기 둘이 서로 다른 가정에 맞물린다.

**아래 한계** — $x_0$을 안다고 볼 때의 사후 흩어짐:

$$\sigma_t^2 = \tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)}\, \beta_t$$

**위 한계** — 앞 과정의 흩어짐:

$$\sigma_t^2 = \beta_t$$

둘 다 표본 품질이 비슷하다. 아래 한계 $\tilde{\beta}_t$은 로그 가능도가 조금 낫고 $\beta_t$은 표본이 조금 더 또렷할 수 있다.

### 2.2 배운 흩어짐(나아진 DDPM)

Nichol와 Dhariwal(2021)은 흩어짐을 로그 공간에서 배운 사이 메우기로 매개변수화한다.

$$\log \sigma_t^2 = v_t \log \beta_t + (1 - v_t) \log \tilde{\beta}_t$$

여기서 $v_t$은 때 걸음마다 신경망이 더 내놓는 낱값이다. 이는 여느 익히기 손실과 변분 한계 항을 아우른 섞은 목표로 익힌다.

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda\, \mathcal{L}_{\text{vlb}}$$

익히기 앞머리에 변분 아래 한계 항이 도맡지 않도록 $\lambda = 0.001$으로 둔다.

---

## 3. 뽑기 알고리즘

### 3.1 밑그림 코드

```
알고리즘: DDPM 조상 뽑기
───────────────────────────────────
Input:  Trained noise predictor ε_θ, noise schedule {β_t, ᾱ_t}_{t=1}^T
내놓기: 만든 표본 x_0

 1. Sample x_T ~ N(0, I)
 2. for t = T, T-1, ..., 1 do
 3.     ε = ε_θ(x_t, t)                                    ▷ Predict noise
 4.     x̂_0 = (x_t − √(1−ᾱ_t) · ε) / √ᾱ_t              ▷ Predict clean image
 5.     x̂_0 = clip(x̂_0, −1, 1)                            ▷ Optional: stabilize
 6.     μ = (√ᾱ_{t-1} · β_t)/(1−ᾱ_t) · x̂_0
            + (√α_t · (1−ᾱ_{t-1}))/(1−ᾱ_t) · x_t         ▷ Posterior mean
 7.     if t > 1 then
 8.         z ~ N(0, I)
 9.         x_{t-1} = μ + σ_t · z                           ▷ Stochastic step
10.     아니면
11.         x_{t-1} = μ                                     ▷ Final step: no noise
12.     조건 끝
13. 되풀이 끝
14. x_0을 돌려준다
```

마지막 걸음($t = 1 \to t = 0$)은 정해져 있다. $t = 0$에서 잡음을 더하면 만든 표본이 망가지기 때문이다.

### 3.2 확률성이 중요한 까닭

걸음마다(8줄) 더하는 잡음 $z$은 군더더기가 아니라 꼭 필요하다. 뒤 과정 $p_\theta(x_{t-1} \mid x_t)$은 정규 분포이며 (평균만 취하지 않고) 거기서 뽑아야 흠 없이 배웠을 때 만든 분포가 자료 분포와 맞는다. 이 잡음을 없애면 DDIM 정해진 뽑기가 되며 다양함을 한결같음과 맞바꾼다.

---

## 4. 구현

```python
"""
DDPM 뽑기
=============
조건 없는 만들어 내기, 가름개 이끌기, 가름개 없는 이끌기를 받쳐 주는
DDPM 조상 뽑기의 온전한 짜기.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm

class DDPMSampler:
    """미리 셈한 차례표 계수를 갖춘 DDPM 조상 뽑개."""

    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_lower",
        clip_denoised: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        매개변수
        ----------
        model : nn.Module
            익힌 잡음 헤아리기 신경망 ε_θ(x_t, t) 또는
            조건 있는 모델의 ε_θ(x_t, t, c).
        n_timesteps : int
            퍼짐 걸음 수 T.
        beta_start, beta_end : float
            잡음 차례표의 끝점.
        beta_schedule : str
            짜임 갈래: 'linear' 또는 'cosine'.
        variance_type : str
            'fixed_lower'(β̃_t), 'fixed_upper'(β_t), 'learned' 가운데 하나.
        clip_denoised : bool
            x̂_0 예측을 [-1, 1]로 자를지 여부.
        device : torch.device
            셈할 기기.
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.variance_type = variance_type
        self.clip_denoised = clip_denoised
        self.device = device

        # --- 잡음 차례표 세우기 ---
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif beta_schedule == "cosine":
            # Nichol와 Dhariwal(2021)의 코사인 차례표
            steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
            f = torch.cos((steps / n_timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = betas.clamp(max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # --- 계수 미리 셈하기(모두 꼴 [T]) ---
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas = torch.sqrt(alphas).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # 사후 평균 계수: μ = coef1 * x̂_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

        # 사후 흩어짐: β̃_t
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        # t=0에서 수치의 안정을 위해 로그를 가둔다
        self.posterior_log_variance = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        ).to(device)

    # ------------------------------------------------------------------
    # 핵심 뽑기 걸음
    # ------------------------------------------------------------------
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """x_t과 헤아린 잡음으로 x_0을 헤아린다."""
        x0 = (
            x_t - self.sqrt_one_minus_alphas_cumprod[t] * eps_pred
        ) / self.sqrt_alphas_cumprod[t]

        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def posterior_mean(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """사후 평균 μ_θ(x_t, t)을 셈한다."""
        return self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x_t

    def get_variance(self, t: int) -> torch.Tensor:
        """뒤 걸음의 σ_t²을 돌려준다."""
        if self.variance_type == "fixed_lower":
            return self.posterior_variance[t]
        elif self.variance_type == "fixed_upper":
            return self.betas[t]
        else:
            raise ValueError(f"Variance type '{self.variance_type}' not supported here. "
                             "Use 'learned' variance via the model output.")

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        거꾸로 한 걸음: x_{t-1} ~ p_θ(x_{t-1} | x_t)에서 표본을 뽑는다.

        매개변수
        ----------
        x_t : Tensor of shape (B, C, H, W) or (B, D)
            지금의 잡음 섞인 표본.
        t : int
            지금 때 걸음.
        condition : Tensor, optional
            조건 앎(갈래 이름표, 글 박아 넣기 등).

        반환값
        -------
        x_{t-1} : x_t와 꼴이 같은 텐서.
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # 잡음을 헤아린다
        if condition is not None:
            eps_pred = self.model(x_t, t_tensor, condition)
        else:
            eps_pred = self.model(x_t, t_tensor)

        # x_0을 헤아리고 사후 평균을 셈한다
        x0_pred = self.predict_x0(x_t, t, eps_pred)
        mean = self.posterior_mean(x_t, x0_pred, t)

        # 뽑기
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.get_variance(t).sqrt()
            return mean + sigma * noise
        else:
            return mean  # 마지막 걸음에는 잡음이 없다

    # ------------------------------------------------------------------
    # 온전한 뽑기 되풀이
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_interval: int = 100,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        조상 뽑기로 표본을 만든다.

        매개변수
        ----------
        shape : tuple
            만들 표본의 꼴. 예컨대 (B, C, H, W).
        condition : Tensor, optional
            조건 앎.
        return_trajectory : bool
            True이면 일정한 사이마다 중간 x_t도 돌려준다.
        trajectory_interval : int
            이만큼의 걸음마다 자취를 갈무리한다.
        show_progress : bool
            tqdm 나아감 막대를 보인다.

        반환값
        -------
        samples : 꼴이 `shape`인 텐서.
        trajectory : 텐서의 목록(있으면).
        """
        self.model.eval()

        # 순수 잡음에서 시작한다
        x = torch.randn(shape, device=self.device)
        trajectory = [x.cpu().clone()] if return_trajectory else None

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling", total=self.n_timesteps)

        for t in timesteps:
            x = self.p_sample(x, t, condition=condition)

            if return_trajectory and t % trajectory_interval == 0 and t > 0:
                trajectory.append(x.cpu().clone())

        if return_trajectory:
            trajectory.append(x.cpu().clone())  # 마지막 표본
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_classifier_guidance(
        self,
        shape: tuple,
        classifier: nn.Module,
        class_label: int,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        가름개 이끌기로 뽑는다(Dhariwal와 Nichol, 2021).

        점수를 고친다: ∇ log p(x_t | y) = ∇ log p(x_t) + s · ∇ log p(y | x_t)

        매개변수
        ----------
        shape : tuple
            만들 표본의 꼴.
        classifier : nn.Module
            Noise-aware classifier p(y | x_t, t).
        class_label : int
            목표 갈래 어깨수.
        guidance_scale : float
            가름개 이끌기의 세기(s).
        show_progress : bool
            나아감 막대를 보인다.

        반환값
        -------
        samples : 꼴이 `shape`인 텐서.
        """
        self.model.eval()
        classifier.eval()

        x = torch.randn(shape, device=self.device)
        labels = torch.full((shape[0],), class_label, device=self.device, dtype=torch.long)

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Classifier-Guided Sampling")

        for t in timesteps:
            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # 가름개 기울기를 셈한다
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                log_probs = torch.log_softmax(classifier(x_in, t_tensor), dim=-1)
                selected = log_probs[range(batch_size), labels]
                grad = torch.autograd.grad(selected.sum(), x_in)[0]

            # 잡음을 헤아리고 이끌기만큼 옮긴다
            eps_pred = self.model(x, t_tensor)
            eps_guided = eps_pred - guidance_scale * self.sqrt_one_minus_alphas_cumprod[t] * grad

            # 이끈 잡음 헤아림으로 잡음을 없앤다
            x0_pred = self.predict_x0(x, t, eps_guided)
            mean = self.posterior_mean(x, x0_pred, t)

            if t > 0:
                sigma = self.get_variance(t).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x

    @torch.no_grad()
    def sample_classifier_free(
        self,
        shape: tuple,
        condition: torch.Tensor,
        guidance_scale: float = 7.5,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        가름개 없는 이끌기로 뽑는다(Ho와 Salimans, 2022).

        조건 있는 예측과 조건 없는 예측을 합친다.
            ε̃ = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, c) − ε_θ(x_t, t, ∅))

        매개변수
        ----------
        shape : tuple
            만들 표본의 꼴.
        condition : Tensor
            조건 신호(갈래 이름표, 글 박아 넣기 등).
        guidance_scale : float
            이끎 세기 s. 1보다 크면 조건 분포가 뾰족해진다.
        show_progress : bool
            나아감 막대를 보인다.

        반환값
        -------
        samples : 꼴이 `shape`인 텐서.
        """
        self.model.eval()

        x = torch.randn(shape, device=self.device)
        # 조건 없는 헤아림을 위한 빈 조건(0이거나 배운 빈 토큰)
        null_condition = torch.zeros_like(condition)

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Classifier-Free Sampling")

        for t in timesteps:
            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # 조건 없는 헤아림과 조건 있는 헤아림
            eps_uncond = self.model(x, t_tensor, null_condition)
            eps_cond = self.model(x, t_tensor, condition)

            # 가름개 없는 이끌기 아우르기
            eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # 잡음을 없앤다
            x0_pred = self.predict_x0(x, t, eps_guided)
            mean = self.posterior_mean(x, x0_pred, t)

            if t > 0:
                sigma = self.get_variance(t).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x
```

---

## 5. 뽑기 자취와 중간 헤아림

### 5.1 잡음 없애기 과정 그려 보기

중간 상태를 갈무리하면 짜임이 어떻게 드러나는지 보인다.

```python
def visualize_trajectory(
    sampler: DDPMSampler,
    shape: tuple,
    save_steps: list = None,
) -> list:
    """
    표본을 만들고 중간 x_t과 x̂_0 헤아림을 갈무리한다.

    매개변수
    ----------
    sampler : DDPMSampler
    shape : tuple
        표본 꼴. 예컨대 (4, 3, 64, 64).
    save_steps : 정수의 목록
        갈무리할 때 걸음. 기본은 고르게 벌린 것이다.

    반환값
    -------
    snapshots : 'timestep', 'x_t', 'x0_pred' 열쇠를 지닌 사전의 목록.
    """
    if save_steps is None:
        save_steps = list(range(0, sampler.n_timesteps, sampler.n_timesteps // 10))

    sampler.model.eval()
    x = torch.randn(shape, device=sampler.device)
    snapshots = []

    for t in range(sampler.n_timesteps - 1, -1, -1):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=sampler.device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = sampler.model(x, t_tensor)
            x0_pred = sampler.predict_x0(x, t, eps_pred)

        if t in save_steps:
            snapshots.append({
                "timestep": t,
                "x_t": x.cpu().clone(),
                "x0_pred": x0_pred.cpu().clone(),
            })

        # 뒤 걸음을 내딛는다
        with torch.no_grad():
            x = sampler.p_sample(x, t)

    # 마지막 표본
    snapshots.append({"timestep": 0, "x_t": x.cpu().clone(), "x0_pred": x.cpu().clone()})
    return snapshots
```

### 5.2 자취가 드러내는 것

| 단계 | 때 걸음 범위 | 살핌 |
|-------|---------------|-------------|
| 앞머리 잡음 없애기 | $t \in [T, 0.7T]$ | 온마당 짜임이 드러난다: 배치, 큰 모양, 으뜸 빛깔 |
| 가운데 잡음 없애기 | $t \in [0.7T, 0.3T]$ | 중간 크기 특징: 물체 가장자리, 결 자리 |
| 뒷머리 잡음 없애기 | $t \in [0.3T, 0]$ | 자잘한 세부: 가장자리, 결, 잦기 높은 내용 |

앞 때 걸음의 $\hat{x}_0$ 헤아림은 흐릿하지만 짜임은 들어맞으며 $t$이 줄수록 차츰 또렷해진다. 이 거친 데서 고운 데로 가는 만들어 내기가 퍼짐 모델의 표징이다.

---

## 6. 셈 살피기

### 6.1 표본마다 비용

뽑기 걸음마다 모델 $\epsilon_\theta$을 앞먹임 한 번 지나야 한다. 매개변수가 $P$개인 U-Net이 크기 $C \times H \times W$인 그림을 다룰 때:

| 부품 | 걸음마다 비용 | 모두($T$걸음) |
|-----------|--------------|-------------------|
| 모델 앞먹임 | $O(P \cdot C \cdot H \cdot W)$ | $T \times O(P \cdot C \cdot H \cdot W)$ |
| 잡음 만들기 | $O(C \cdot H \cdot W)$ | 무시할 만함 |
| 계수 찾아보기 | $O(1)$ | 무시할 만함 |

흔한 U-Net(매개변수 약 1억 개)이 256×256 해상도에서 $T = 1000$일 때:

| 잣대 | 대략의 값 |
|--------|------------------|
| 걸음마다 시간 | 약 15 ms(A100 GPU) |
| 온 뽑기 시간 | 그림마다 약 15초 |
| 기억(묶음=1) | 약 4 GB |
| 기억(묶음=16) | 약 20 GB |

### 6.2 빠르기 문제

여느 DDPM은 표본마다 모델을 차례대로 $T = 1000$번 셈해야 한다. 이는 맞겨루기 만들개처럼 한 번에 만드는 것(그림마다 약 20 ms)보다 몇 자릿수 느리다. 이 바탕이 되는 한계가 다음을 이끌어 냈다.

| 빠르게 하는 방법 | 필요한 걸음 | 품질 | 마디 |
|--------------------|---------------|---------|---------|
| **DDPM**(바탕) | 1000 | 가장 좋음 | 이 쪽 |
| **DDIM** | 50–100 | DDPM에 가깝다 | [DDIM 기초](../ddim/fundamentals.md) |
| **DDIM(정해진 대로)** | 20–50 | 좋다 | 정해진 대로 뽑기 |
| **확률 흐름 상미분 방정식** | 20–100 | 좋다 | 확률 흐름 |
| **차근차근 앎 옮기기** | 4–8 | 좋다 | 얼개에 따라 다르다 |
| **한결같음 모형** | 1–2 | 어지간하다 | 한 걸음 만들기 |

---

## 7. 실전에서 살필 것

### 7.1 수치의 정밀도

익힐 때 섞인 정밀도를 썼더라도 뽑을 때는 `float32`을 쓰라. 반 정밀도의 쌓이는 어긋남이 1000걸음에 걸쳐 겹쳐 흠을 만든다.

```python
# 뽑기에는 float32을 쓴다
model = model.float()
x = torch.randn(shape, device=device, dtype=torch.float32)
```

### 7.2 x-hat_0 자르기

헤아린 $x_0$을 $[-1, 1]$(또는 자료 범위)으로 자르면 잡음이 클 때 헤아림이 부정확해 사후 평균이 떠도는 것을 막는다. 특히 다음에서 중요하다.

- 헤아림을 믿기 어려운 앞 때 걸음($t$이 $T$에 가까움)
- 잡음 차례표를 꼼꼼히 맞추지 않고 익힌 모델
- 이끌기가 헤아림을 범위 밖으로 밀 수 있는 조건 만들어 내기

### 7.3 묶음 뽑기

묶음 차원을 늘려 표본 여럿을 나란히 만든다. 기억이 허락하면 차례대로 만드는 것보다 훨씬 효율이 좋다.

```python
# 표본 64개를 나란히 만든다
samples = sampler.sample(shape=(64, 3, 64, 64), show_progress=True)
```

### 7.4 되풀이할 수 있음

되풀이할 수 있는 만들어 내기를 위해 첫 잡음을 붙박는다. 그러면 모델이나 이끌기 잣수에 걸쳐 다스린 견줌을 할 수 있다.

```python
# 되풀이할 수 있는 만들어 내기
generator = torch.Generator(device=device).manual_seed(42)
x_T = torch.randn(shape, device=device, generator=generator)
```

---

## 8. 돈살림 쓰임새

DDPM 뽑기는 돈살림 자료 만들어 내기로 자연스럽게 넓혀진다.

| 쓰임새 | 뽑기 방식 | 참고 |
|-------------|------------------|-------|
| 인공 수익률 길 | 조건 없는 뽑기 | 그럴듯한 여러 변수 수익률 분포를 만든다 |
| 판세 조건 시나리오 | 가름개 없는 이끌기 | 저자 판세 이름표를 조건으로 삼는다 |
| 버팀 시험 | 가름개 이끌기 | 위험 가름개로 꼬리 사건 쪽으로 이끈다 |
| 꾸러미 시나리오 만들기 | 조건 뽑기 | 앞을 내다보는 시나리오를 위해 거시 변수를 조건으로 삼는다 |
| 빠진 자료 메우기 | 안 그리기 꼴 뽑기 | 본 값은 붙박고 빠진 항목을 뽑는다 |

돈살림 쓰임새의 자세한 것은 시계열 만들어 내기와 시나리오 만들어 내기를 보라.

---

## 9. 핵심 간추리기

1. **DDPM 뽑기는 앞 과정을 거꾸로 돌린다.** 곧 배운 잡음 헤아리개 $\epsilon_\theta(x_t, t)$을 거듭 써서 사후 평균 $\mu_\theta$을 셈하고 정규 뒤 걸음에서 뽑는다.

2. **$\hat{x}_0$ 헤아리기 관점**이 직관을 준다. 곧 걸음마다 모델이 깨끗한 그림을 헤아리고 사후 평균이 그 헤아림과 지금의 잡음 섞인 상태 사이를 메운다.

3. **흩어짐 고르기**($\tilde{\beta}_t$, $\beta_t$, 배운 것)는 표본 품질에는 영향이 작지만 로그 가능도에는 영향을 준다. 코사인 차례표와 배운 흩어짐이 가장 좋은 가능도 한계를 준다.

4. **마지막 걸음은 정해져 있다.** 곧 올바른 뒤 분포를 지키려 $t > 0$에서만 잡음을 더한다.

5. **가름개 이끌기**는 바깥 가름개의 기울기로 점수를 고친다. **가름개 없는 이끌기**는 조건 있는 헤아림과 없는 헤아림 사이를 메워 따로 가름개를 두지 않는다.

6. **1000 걸음이 든다는 것이 가장 큰 한계**이며, 그래서 [DDIM](../ddim/fundamentals.md)을 비롯한 빠른 뽑기 방법이 나왔다.

---

## 연습문제

### 연습 1: 흩어짐 견주기

$\sigma_t = \sqrt{\tilde{\beta}_t}$과 $\sigma_t = \sqrt{\beta_t}$으로 뽑기를 모두 짜라. 각각으로 표본 1000개를 만들어 FID 점수를 견주어라. 차이가 뜻있는가?

### 익힘 2: 자르기 떼어 보기

$\hat{x}_0$ 자르기를 하거나 하지 않고 표본을 만들어라. 두 자리매김 모두에서 $t = 900, 500, 100$의 $\hat{x}_0$ 헤아림을 그려 보라. 어느 때 걸음에서 자르기의 효과가 가장 큰가?

### 익힘 3: 자취 그려 보기

`visualize_trajectory`으로 100걸음마다 장면을 갈무리하라. 갈무리한 걸음마다 $x_t$과 $\hat{x}_0$을 함께 보이는 그림을 만들어라. 거친 데서 고운 데로 가는 만들어 내기 과정을 설명하라.

### 익힘 4: 이끌기 잣수 훑기

가름개 없는 이끌기로 $s \in \{1.0, 3.0, 5.0, 7.5, 10.0, 15.0\}$에서 표본을 만들어라. 다양함과 품질의 맞바꿈(FID와 서로 다른 봉우리 수)을 그려라. 어느 잣수에서 봉우리 무너짐이 눈에 띄는가?

### 익힘 5: 뽑기 살림

매개변수 1억 개짜리 U-넷으로 256×256 표본 10,000개를 만들어야 한다면 DDPM($T = 1000$)과 DDIM($T = 50$)에 드는 온 GPU 시간을 어림하여라. GPU 시간당 \$2이라면 값 차이는 얼마인가?

---

## 정리하며

이 마당은 뒤 과정 이끌어 내기、흩어짐 매개변수화、뽑기 알고리즘、구현을 차례로 짚었다.

**참고 문헌**

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Nichol, A. Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
3. Dhariwal, P. & Nichol, A. Q. (2021). Diffusion Models Beat GANs on Image Synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*.
5. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
