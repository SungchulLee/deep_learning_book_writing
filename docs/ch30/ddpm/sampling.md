# DDPM 샘플링

DDPM은 표준정규 노이즈에서 시작해 학습한 역전이 분포를 순서대로 적용한다. 노이즈 예측을 역전이 평균으로 바꾸는 식, 분산 선택과 마지막 단계의 처리가 핵심이다. 여기서는 [역과정 유도](../foundations/reverse_process.md)와 연결해 샘플러를 읽고, 조건부 생성과 계산 비용의 범위를 살펴본다.

## 1. 조건부 사후분포에서 모델 평균으로

전방과정은 αₜ=1−βₜ, 누적곱 ᾱₜ=∏ₛ₌₁ᵗαₛ를 사용한다. 원본 x₀를 알고 있으면 두 가우스 전이를 결합해 다음 사후분포를 얻는다.

$$q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t,\tilde\beta_t I)$$

$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

생성할 때는 x₀를 모르므로 이 조건부 사후분포를 직접 사용할 수 없다. x₀를 적분한 q(xₜ₋₁|xₜ)는 일반적으로 가우스 혼합이며 단일 가우스와 같지 않다. 모델은 가우스 역전이 pθ의 평균을 노이즈 예측 εθ로 구성한다.

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},
\qquad
\mu_\theta=\frac1{\sqrt{\alpha_t}}\left(x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

x̂₀를 사후평균 식의 x₀ 자리에 대입하면 오른쪽 평균식이 나온다. 단, 코드처럼 x̂₀를 [−1,1]로 자르면 비선형 변환이 추가되므로 자르기 없는 εθ 평균식과 더 이상 정확히 같지 않다. 자르기는 그 범위로 정규화한 데이터에서 사용하는 선택이며 잠재변수나 다른 데이터 범위에 그대로 적용하지 않는다.

## 2. 역전이 분산과 마지막 단계

자주 사용하는 고정 분산은 β̃ₜ와 βₜ다. 전자는 x₀까지 조건으로 준 사후분산이고 후자는 전방전이 분산이다. β̃ₜ≤βₜ이지만 이 둘이 임의의 데이터 분포에서 최적 주변 역전이 분산의 보편적인 하한·상한이라는 뜻은 아니다. 코드의 fixed_lower와 fixed_upper는 이 두 선택을 가리키는 이름이다.

[Improved DDPM](https://arxiv.org/abs/2102.09672)은 log σ²를 log β̃와 log β의 조합으로 예측하고 단순 손실과 변분 항을 결합한다. 예측값은 구현에 따라 위치·채널별일 수 있어 항상 시간마다 스칼라 하나로 제한되는 것은 아니다. 이 페이지의 DDPMSampler는 **두 고정 분산만 지원**하며 learned를 지정하면 오류가 난다.

수식에서 마지막 전이는 t=1→0이다. 구현의 시간 인덱스는 0부터 시작하므로 t=0인 호출에서 추가 노이즈를 넣지 않는다. β̃₁=0과 맞는 관례다. β 분산 선택에서도 마지막은 평균을 반환하도록 정의한다. 이것이 가능한 모든 생성 모델의 최종 확률분포가 결정적이라는 주장은 아니다.

## 3. 기본 알고리즘

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

중간 단계의 가우스 노이즈는 선택한 확률적 역전이에서 샘플을 뽑는 데 필요하다. 그러나 **DDPM 식에서 노이즈만 제거하면 DDIM이 되는 것은 아니다**. DDIM은 시간 간격에 맞게 상태 방향과 분산 계수를 다시 구성한다. 결정적 DDIM도 초기 노이즈가 무작위이므로 여러 초기점에서 다양한 결과를 만들 수 있다.

## 4. 샘플러 구현과 호출 조건

```python
"""
DDPM 뽑기
=============
조건 없는 만들어 내기, 분류기 이끌기, 분류기 없는 이끌기를 받쳐 주는
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
            확산 걸음 수 T.
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
            조건 앎(갈래 이름표, 글 임베딩 등).

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
        분류기 이끌기로 뽑는다(Dhariwal와 Nichol, 2021).

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
            분류기 이끌기의 세기(s).
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

            # 분류기 기울기를 셈한다
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
        분류기 없는 이끌기로 뽑는다(Ho와 Salimans, 2022).

        조건 있는 예측과 조건 없는 예측을 합친다.
            ε̃ = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, c) − ε_θ(x_t, t, ∅))

        매개변수
        ----------
        shape : tuple
            만들 표본의 꼴.
        condition : Tensor
            조건 신호(갈래 이름표, 글 임베딩 등).
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

            # 분류기 없는 이끌기 아우르기
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

model은 배치 시간 텐서와 입력을 받아 같은 모양의 노이즈 예측을 반환해야 한다. 샘플러 생성 시 지정한 스케줄·T가 학습 조건과 일치해야 한다. 계수는 지정 장치에 놓지만 모델과 condition을 자동으로 이동하지 않으므로 호출자가 장치·dtype·배치 크기를 맞춘다.

sample은 eval과 no_grad로 실행하며 이전 학습 모드로 되돌리지 않는다. return_trajectory=True이면 결과 텐서와 CPU 상태 목록을 함께 반환한다. 처음 상태, 지정 간격에 해당하는 **업데이트 후 상태**, 마지막 상태를 저장하므로 목록 위치를 곧 시간 인덱스로 해석하지 않는다. trajectory_interval은 양수여야 한다.

## 5. 분류기 가이던스와 CFG

분류기 가이던스는 노이즈가 섞인 xₜ에서 클래스 로그확률의 입력 기울기를 구한다. 스코어와 노이즈 예측의 관계를 이용하면 다음과 같다.

$$\epsilon_{\mathrm{guided}}=\epsilon_\theta-
 s\sqrt{1-\bar\alpha_t}\nabla_{x_t}\log p_\phi(y\mid x_t,t)$$

분류기도 시간별 노이즈 입력을 처리하도록 학습되어 있어야 한다. 전체 샘플링은 no_grad로 실행하지만 분류기 입력 기울기를 구하는 부분만 enable_grad로 켠다. 이 코드에서는 위 노이즈 예측을 통해 평균을 바꾸며, 자르기가 없을 때 평균 변화는 sβₜ/√αₜ에 분류기 기울기를 곱한 값이다. 선택한 σ²에 기울기를 곱하는 다른 평균 이동 공식과 항상 같지는 않다.

Classifier-Free Guidance는 같은 모델의 무조건·조건부 예측을 결합한다.

$$\epsilon_{\mathrm{CFG}}=\epsilon_{\varnothing}+s(\epsilon_c-\epsilon_{\varnothing})$$

이 관례에서 s=0은 무조건 예측, s=1은 조건부 예측이다. s>1은 두 예측 사이의 보간이 아니라 외삽이다. 함수는 zeros_like(condition)을 null 조건으로 사용하지만 **0이 실제 학습에서 무조건 토큰으로 예약되어 있어야 한다**. 정수 클래스 0이 실제 클래스라면 그대로 사용할 수 없다. CFG는 단계마다 모델을 두 번 호출하며 품질·다양성의 변화는 데이터와 설정에 따라 평가해야 한다.

## 6. 중간 상태와 원본 추정 비교

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

visualize_trajectory는 저장 시점의 xₜ와 x̂₀를 함께 보관한다. 노이즈가 많은 xₜ에서도 x̂₀의 구조를 살펴볼 수 있지만, 고정 시간 구간별로 전역 구조·윤곽·세부가 반드시 순서대로 나타난다고 단정하지 않는다. 스케줄과 모델에 따라 관찰이 달라진다.

이 함수는 각 단계에서 예측을 구한 뒤 p_sample 안에서 다시 구하므로 **단계마다 두 번 모델을 호출**한다. 기본 T=1000에서는 2,000회다. 기본 저장 목록은 t=0의 업데이트 전 상태를 담고 마지막에 업데이트 후 결과도 timestep=0으로 덧붙인다. 같은 레이블의 두 상태가 같다는 뜻이 아니다. T<10에서는 기본 간격 T//10이 0이 되는 제한도 있다.

## 7. 정밀도·배치·난수

```python
# 뽑기에는 float32을 쓴다
model = model.float()
x = torch.randn(shape, device=device, dtype=torch.float32)
```

float32는 계수와 반복 연산의 수치 상태를 점검하는 출발점이 될 수 있다. 모든 모델에서 반정밀도 추론이 잘못된다는 뜻은 아니며 학습 dtype, 연산별 안정성과 실제 오차를 함께 확인해야 한다. 이 짧은 코드만으로 앞서 생성한 샘플러의 모든 상태가 자동으로 같은 dtype으로 바뀌는 것도 아니다.

```python
# 표본 64개를 나란히 만든다
samples = sampler.sample(shape=(64, 3, 64, 64), show_progress=True)
```

배치 처리는 연산 활용률을 높일 수 있지만 메모리와 모델 구조에 제약을 받는다. 일반 샘플링의 모델 호출 수는 배치당 T회이며 이미지 수만큼 T에 다시 곱한 순차 호출을 수행하는 것은 아니다. 벽시계 시간은 배치 크기·해상도·정밀도·어텐션·장치 등에 따라 측정해야 한다. 파라미터 수만으로 U-Net의 FLOPs나 메모리를 결정할 수 없다.

```python
# 되풀이할 수 있는 만들어 내기
generator = torch.Generator(device=device).manual_seed(42)
x_T = torch.randn(shape, device=device, generator=generator)
```

이 난수 예제는 별도 x_T만 만든다. sample은 전달받는 x_T나 generator 인수가 없고 내부에서 초기·중간 노이즈를 새로 생성하므로, **위 x_T를 만들었다고 실제 샘플링 경로가 고정되지는 않는다**. 비교하려는 경로에 초기점과 중간 난수가 실제로 사용되는지 확인해야 한다.

## 8. 빠른 샘플러와 응용 범위

DDIM처럼 시간 부분열을 사용하는 방법은 단계 수를 줄일 수 있지만, 학습된 인덱스를 임의로 건너뛰면서 기존 한 단계 DDPM 식을 그대로 적용해서는 안 된다. 자세한 계수는 [DDIM 기초](../ddim/fundamentals.md)에서 다룬다. 시간 감소율이 호출 수 감소율과 정확히 같거나 DDPM이 언제나 최고 품질이라는 보장은 없다.

시계열 시나리오나 결측값 생성에도 역과정의 개념을 적용할 수 있다. 다만 이 이미지 샘플러는 시간 의존성, 관측값 조건화나 꼬리 분포 검증을 구현하지 않는다. 합성 경로를 실제 미래 예측이나 위험 추정의 검증 결과로 해석해서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1. 고정 분산의 의미**

β̃ₜ와 βₜ의 차이, 마지막 단계에서 추가 노이즈를 넣지 않는 인덱스 관례를 설명하라.

</div>

??? success "연습문제 1 풀이"
    β̃는 x₀를 조건으로 준 사후분산이고 β는 전방전이 분산이다. 둘이 임의의 주변 역전이 분산을 보편적으로 둘러싸지는 않는다. 수식 t=1→0은 코드 t=0 호출에 해당하며 이 호출은 두 선택 모두 평균만 반환한다.

---

<div class="drillbox" markdown>

**연습문제 2. 결정적 DDIM과의 구분**

DDPM 평균만 반복하면 결정적 DDIM과 같은가?

</div>

??? success "연습문제 2 풀이"
    같지 않다. DDIM은 선택한 시간 간격의 누적 계수로 x̂₀와 예측 노이즈의 방향 계수를 다시 구성한다. DDPM 전이의 노이즈 항만 지우는 것은 다른 업데이트다. 결정적 경로여도 초기 노이즈가 다르면 결과가 달라질 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3. CFG 조건**

클래스 레이블을 condition으로 사용할 때 zeros_like가 무조건 조건이 되려면 무엇이 필요한가?

</div>

??? success "연습문제 3 풀이"
    학습 단계에서 0을 null 조건으로 예약해야 한다. 실제 클래스 0을 뜻한다면 그 예측은 무조건 예측이 아니다. 무조건 전용 토큰이나 학습 때 사용한 마스킹 표현이 필요하다. 이 CFG 관례에서 s=1은 조건부 예측이고 s>1은 외삽이다.

---

<div class="drillbox" markdown>

**연습문제 4. 경로 기록의 비용**

기본 T=1000인 visualize_trajectory의 모델 호출 수와 마지막 두 timestep=0 기록의 차이를 설명하라.

</div>

??? success "연습문제 4 풀이"
    단계마다 직접 예측 한 번과 p_sample 안 예측 한 번으로 총 2,000회다. 첫 t=0 기록은 마지막 업데이트 전 상태와 그 원본 추정이며, 마지막 추가 기록은 업데이트 후 결과다. 같은 시간 레이블만 보고 같은 상태로 해석하면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 5. 계산 예산의 조건부 추정**

가상의 배치 1에서 한 번의 모델 호출이 15ms이고 부가 비용을 무시한다고 하자. 10,000개를 1,000회와 50회 호출로 순차 생성할 때 시간과 시간당 2달러의 비용을 계산하라.

</div>

??? success "연습문제 5 풀이"
    1,000회는 이미지당 15초, 전체 150,000초=약 41.67시간이므로 약 83.33달러다. 50회는 이미지당 0.75초, 전체 7,500초=약 2.08시간이므로 약 4.17달러다. 이는 동일 호출 시간과 배치 1을 가정한 산술 예시이며 실제 장치 성능이나 DDIM 품질을 측정한 값이 아니다.

## 정리하며

DDPM은 노이즈 예측으로 역전이 평균을 구성하고 선택한 분산으로 샘플링한다. 시간 인덱스와 마지막 단계, 자르기로 달라지는 평균을 확인한다. 가이던스의 null 조건과 입력 기울기, 경로 기록의 추가 호출, 실제로 사용되는 난수 경로를 구분해야 한다.

**참고 문헌**

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Nichol, A. Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
3. Dhariwal, P. & Nichol, A. Q. (2021). Diffusion Models Beat GANs on Image Synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*.
5. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
