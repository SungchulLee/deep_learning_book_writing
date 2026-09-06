# 퍼짐 도구

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 퍼짐 도구을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""퍼짐 도구."""
# ============================================================================
# diffusion_utils.py - 퍼짐 모델의 수학 바탕
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""
퍼짐 모델 도구 - 핵심 개념
==========================================

이 단원은 퍼짐 모델의 수학 엔진을 담는다.

KEY COMPONENTS:
---------------
1. NOISE SCHEDULES: Define how much noise to add at each timestep (β_t)
2. 퍼짐 매개변수: 효율을 위해 미리 셈한 수학 상수
3. 앞 퍼짐: 닫힌 꼴 공식으로 잡음 더하기
   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
4. REVERSE DIFFUSION: Remove noise iteratively (generation)
5. 익히기 도구: 배우기 위한 손실 셈하기
6. 그려 보기 연장: 퍼짐을 눈으로 이해하기

MATHEMATICAL FOUNDATION:
------------------------
Forward process (data → noise):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

Closed-form solution:
    q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
    where ᾱ_t = ∏_{i=1}^t (1-β_i)

Reverse process (noise → data):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

모델은 잡음 ε_θ(x_t, t)을 헤아려 μ_θ을 배운다.
"""

# ============================================================================
# 잡음 차례표
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, 
                        beta_end: float = 0.02) -> torch.Tensor:
    """
    선형 잡음 차례표를 만든다.
    
    β_t increases linearly: [0.0001, 0.0002, ..., 0.02]
    
    좋은 점: 단순하고 직관적이며 이해하기 쉽다
    나쁜 점: 그림에는 가장 좋지 않고 끝에서 잡음을 너무 빨리 더한다
    
    역사 참고: 본디 DDPM 논문에서 썼다.
    요즘 방식: 그림에는 코사인 차례표가 낫다.
    
    인수:
        timesteps: Total diffusion steps (T)
        beta_start: Initial noise level (≈0.0001)
        beta_end: Final noise level (≈0.02)
    
    반환값:
        betas: Tensor of shape (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    코사인 잡음 차례표를 만든다(선형보다 나아진 것).
    
    왜 코사인인가?
    -----------
    - 퍼지는 내내 더 매끄럽게 나아간다
    - 끝에서 덜 과감하다
    - 신호 대 잡음비를 더 잘 지킨다
    - 그림 만들어 내기 품질이 높다
    
    MATHEMATICAL DEFINITION:
    ------------------------
    f(t) = cos((t/T + s)/(1 + s) · π/2)²
    ᾱ_t = f(t)/f(0)
    β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    
    코사인은 ᾱ_t이 줄어드는 매끄러운 S자 곡선을 만든다.
    Offset 's' prevents numerical instabilities at boundaries.
    
    BENEFITS:
    ---------
    1. 매끄러운 옮아감: 걸음마다 비슷한 양의 잡음을 없앤다
    2. 더 나은 기울기: 때 걸음에 걸쳐 배움 신호가 고르다
    3. 흠이 적다: 갑작스러운 바뀜이 덜하다
    4. 겪어 본 성공: 최고 수준의 모델이 이것을 쓴다
    
    참고: "Improved Denoising Diffusion Probabilistic Models"
               (Nichol & Dhariwal, 2021)
    
    인수:
        timesteps: 온 퍼짐 걸음
        s: Small offset (0.008 default, prevents β_t ≈ 0 at start)
    
    반환값:
        betas: 꼴 (timesteps,)인 텐서이며 매끄럽게 늘어난다
    """
    # 때 어깨수를 만든다(차이를 셈하려면 T+1개 점이 필요하다)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # 코사인 함수로 ᾱ_t을 셈한다
    t_normalized = (x / timesteps + s) / (1 + s)
    alphas_cumprod = torch.cos(t_normalized * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # ᾱ_0=1이 되게 고르게 맞춘다
    
    # ᾱ_t에서 β_t을 셈한다: β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # 수치의 안정을 위해 자른다
    return torch.clip(betas, 0.0001, 0.9999)

# ============================================================================
# 퍼짐 매개변수(효율을 위해 미리 셈함)
# ============================================================================

def get_diffusion_parameters(betas: torch.Tensor) -> dict:
    """
    퍼짐에 필요한 수학 상수를 모두 미리 셈한다.
    
    왜 미리 셈하는가?
    ---------------
    Training requires these values repeatedly. Computing once is O(T),
    되풀이마다 O(T) 대신 O(1) 찾아보기 → 엄청나게 빨라진다!
    
    For T=1000, 10k iterations: 1000 computations + 10k lookups vs 10M computations
    
    DERIVED QUANTITIES:
    -------------------
    α_t = 1 - β_t                    (signal retention)
    ᾱ_t = ∏_{i=1}^t α_i             (cumulative signal)
    √ᾱ_t, √(1-ᾱ_t)                  (forward diffusion coefficients)
    1/√α_t                           (reverse diffusion scaling)
    σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)  (posterior variance)
    
    인수:
        betas: β schedule, shape (T,)
    
    반환값:
        Dictionary with all precomputed parameters:
        - 'betas': β_t values
        - 'alphas': α_t = 1 - β_t
        - 'alphas_cumprod': ᾱ_t (cumulative product)
        - 'alphas_cumprod_prev': ᾱ_{t-1} (shifted)
        - 'sqrt_alphas_cumprod': √ᾱ_t (data coefficient)
        - 'sqrt_one_minus_alphas_cumprod': √(1-ᾱ_t) (noise coefficient)
        - 'sqrt_recip_alphas': 1/√α_t (reverse scaling)
        - 'posterior_variance': σ_t² (reverse noise)
    """
    # α_t = 1 - β_t(걸음마다 남는 신호)
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏ α_i(쌓인 신호 - 핵심 양이다!)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # 성질: 줄어들고 ≈1에서 시작해 ≈0에서 끝난다
    
    # ᾱ_{t-1}(ᾱ_0 = 1이므로 앞에 1.0을 붙인다)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # 앞 퍼짐 계수: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # 뒤 퍼짐 잣수 맞추기: μ = 1/√α_t·(...)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # 사후 흩어짐: 뒤로 갈 때 잡음을 얼마나 더할지
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }

# ============================================================================
# 도구: 퍼뜨리기를 곁들인 묶음 어깨수 찾기
# ============================================================================

def extract(tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    1차원 텐서의 어깨수 t에서 값을 뽑고 퍼뜨리기에 맞게 꼴을 바꾼다.
    
    THE PROBLEM:
    ------------
    Batch has different timesteps: t = [15, 73, 42, ...]
    표본마다 다른 상수가 필요하다: √ᾱ_15, √ᾱ_73, √ᾱ_42, ...
    
    SOLUTION:
    ---------
    1. Index: tensor[t] → get values for each sample
    2. 꼴 바꾸기: 퍼뜨리기를 위해 (묶음,) → (묶음, 1, 1, ...)
    
    BROADCASTING EXAMPLE:
    ---------------------
    x 꼴: (3, 2)          2차원 점 셋
    values: (3, 1)           꼴을 바꾼 뒤
    
    [[x1, y1],     [[v1, 1],      [[v1·x1, v1·y1],
     [x2, y2],  *   [v2, 1],  =    [v2·x2, v2·y2],
     [x3, y3]]      [v3, 1]]       [v3·x3, v3·y3]]
    
    인수:
        tensor: Source values, shape (T,)
        t: Timestep indices, shape (batch_size,)
        x_shape: 퍼뜨리기의 목표 자료 꼴
    
    반환값:
        Extracted and reshaped values, shape (batch_size, 1, 1, ...)
    """
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)  # 어깨수에서 뽑는다
    
    # 퍼뜨리기에 맞게 꼴을 바꾼다: (묶음,) → (묶음, 1, 1, ...)
    num_extra_dims = len(x_shape) - 1
    trailing_dims = (1,) * num_extra_dims
    return out.reshape(batch_size, *trailing_dims)

# ============================================================================
# 앞 퍼짐
# ============================================================================

def forward_diffusion(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                     sqrt_alphas_cumprod: torch.Tensor,
                     sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    앞 퍼짐을 쓴다: 깨끗한 자료에 잡음을 더한다.
    
    FORMULA:
    --------
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    
    풀이:
    ---------------
    Weighted sum of data and noise:
    - t=0:   ᾱ_0=1    → x_0 = x_0 (no noise)
    - t=50:  ᾱ_50≈0.6 → 60% data + 80% noise (magnitude)
    - t=100: ᾱ_100≈0.3 → 30% data + 95% noise
    - t→∞:   ᾱ_∞→0   → 순수 잡음
    
    왜 제곱근인가?
    -----------------
    Variance preservation: √ᾱ_t² + √(1-ᾱ_t)² = 1
    x_t의 흩어짐이 x_0과 같게 한다!
    
    인수:
        x_0: Clean data, shape (batch_size, *data_dims)
        t: Timesteps, shape (batch_size,)
        noise: 정규 잡음 ε ~ N(0,I)이며 x_0과 같은 꼴
        sqrt_alphas_cumprod: Precomputed √ᾱ_t, shape (T,)
        sqrt_one_minus_alphas_cumprod: Precomputed √(1-ᾱ_t), shape (T,)
    
    반환값:
        x_t: 때 걸음 t의 잡음 섞인 자료
    """
    # 표본마다 때 걸음의 계수를 뽑고 퍼뜨리기에 맞게 꼴을 바꾼다
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # 앞 퍼짐 공식을 쓴다
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

# ============================================================================
# 뒤 퍼짐(만들어 내기)
# ============================================================================

@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: int, t_tensor: torch.Tensor,
            diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    Perform one reverse diffusion step: x_t → x_{t-1}.
    
    REVERSE FORMULA:
    ----------------
    μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
    x_{t-1} = μ_θ + σ_t·z  (if t>0)
    x_0 = μ_θ              (if t=0)
    
    PROCESS:
    --------
    1. Model predicts noise: ε_θ(x_t,t)
    2. 잡음 없앤 평균을 셈한다: μ_θ
    3. Add small noise σ_t·z for diversity (except at t=0)
    
    왜 잡음을 없애면서 잡음을 더하는가?
    -------------------------------
    잡음이 없으면: 정해짐 → 봉우리 무너짐, 다양함 없음
    잡음이 있으면: 확률 → 풍부한 표본, 온전한 분포
    Key: noise added < noise removed → net progress!
    
    인수:
        model: 익힌 잡음 없애기 모델
        x_t: 때 걸음 t의 잡음 섞인 자료
        t: Current timestep (integer)
        t_tensor: Timestep as tensor, shape (batch_size,)
        diffusion_params: 미리 셈한 상수
        device: 'cpu' or 'cuda'
    
    반환값:
        x_{t-1}: Less noisy data at previous timestep
    """
    # 모델로 잡음을 헤아린다
    predicted_noise = model(x_t, t_tensor)
    
    # 지금 때 걸음의 매개변수를 뽑는다
    beta_t = extract(diffusion_params['betas'], t_tensor, x_t.shape)
    sqrt_recip_alpha_t = extract(diffusion_params['sqrt_recip_alphas'], t_tensor, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t_tensor, x_t.shape
    )
    
    # 잡음 없앤 평균을 셈한다: μ = 1/√α_t·(x_t - β_t/√(1-ᾱ_t)·ε_pred)
    noise_term = beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
    model_mean = sqrt_recip_alpha_t * (x_t - noise_term)
    
    if t == 0:
        # 마지막 걸음: 잡음을 더하지 않는다
        return model_mean
    else:
        # 다양함을 위해 확률 잡음을 더한다
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t_tensor, x_t.shape)
        noise = torch.randn_like(x_t)
        posterior_std = torch.sqrt(posterior_variance_t)
        return model_mean + posterior_std * noise

@torch.no_grad()
def sample(model: nn.Module, shape: tuple, timesteps: int,
          diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    온전한 뒤 퍼짐으로 표본을 만든다.
    
    알고리즘:
    ----------
    x_T ~ N(0,I)           ← 순수 잡음에서 시작
        ↓ 잡음 없애기
    x_{T-1} = p_θ(· | x_T)
        ↓ ...
    x_0                    ← 마지막 깨끗한 표본
    
    TRADE-OFFS:
    -----------
    걸음이 많으면(T이 큼): 품질이 높고 느리다
    걸음이 적으면(T이 작음): 빠르고 품질이 낮다
    
    Common: T=100 (fast), T=1000 (standard), T=4000 (high quality)
    
    인수:
        model: Trained denoising model (in eval mode)
        shape: Shape of samples, e.g., (32, 2) or (16, 3, 256, 256)
        timesteps: Number of denoising steps (must match training!)
        diffusion_params: 미리 셈한 상수
        device: 'cpu' or 'cuda'
    
    반환값:
        꼴에 맞는 만든 표본
    """
    model.eval()
    
    # 순수 잡음에서 시작한다
    x_t = torch.randn(shape, device=device)
    
    # 거듭 잡음 없애기: T → T-1 → ... → 1 → 0
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_tensor, diffusion_params, device)
    
    return x_t  # x_0: 깨끗한 만든 표본

# ============================================================================
# 익히기 손실
# ============================================================================

def get_loss(model: nn.Module, x_0: torch.Tensor, t: torch.Tensor,
            diffusion_params: dict, noise: torch.Tensor = None) -> torch.Tensor:
    """
    퍼짐 모델의 익히기 손실을 셈한다.
    
    DDPM TRAINING OBJECTIVE:
    ------------------------
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]
    
    x_t을 만들려 더한 바로 그 잡음을 모델이 헤아리기를 바란다.
    
    왜 잡음을 헤아리는가?
    ------------------
    - 잡음은 멈춰 있다: 모든 때 걸음에서 N(0,I)이다
    - t에 걸쳐 배움 목표가 고르다
    - Highest quality samples (empirically)
    - 요즘 퍼짐 모델의 여느 방식이다
    
    RANDOM TIMESTEP SAMPLING:
    -------------------------
    Key trick: Sample t uniformly from [0,T-1] per batch
    → 모델이 모든 잡음 수준에서 잡음 없애는 법을 배운다
    → 특정 때 걸음에 지나치게 맞춰지는 것을 막는다
    
    인수:
        model: 익힐 잡음 없애기 모델
        x_0: Batch of clean data, shape (batch_size, *data_dims)
        t: Random timesteps, shape (batch_size,)
        diffusion_params: 미리 셈한 상수
        noise: Optional pre-generated noise (if None, generate here)
    
    반환값:
        뒤먹임 퍼뜨리기를 위한 낱값 평균 제곱 어긋남 손실
    """
    # 잡음이 주어지지 않으면 만든다
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # 앞 퍼짐을 쓴다: 아는 잡음으로 x_t을 만든다
    x_t = forward_diffusion(
        x_0, t, noise,
        diffusion_params['sqrt_alphas_cumprod'],
        diffusion_params['sqrt_one_minus_alphas_cumprod']
    )
    
    # 모델이 잡음을 헤아린다
    predicted_noise = model(x_t, t)
    
    # 헤아린 잡음과 실제 잡음 사이의 평균 제곱 어긋남 손실을 셈한다
    return nn.functional.mse_loss(predicted_noise, noise)

# ============================================================================
# 그려 보기 도구
# ============================================================================

def visualize_diffusion_process(x_0: torch.Tensor, timesteps: int,
                               diffusion_params: dict, num_images: int = 10):
    """
    그림의 앞 퍼짐을 그려 본다(자료 → 잡음).
    
    차츰 망가지는 모습을 보인다: 깨끗함 → 조금 잡음 → 아주 잡음 → 순수 잡음
    모델이 무엇을 거꾸로 돌려야 하는지 직관을 쌓는 데 도움이 된다.
    """
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    time_steps = np.linspace(0, timesteps - 1, num_images, dtype=int)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    
    for idx, t in enumerate(time_steps):
        noise = torch.randn_like(x_0)
        t_tensor = torch.tensor([t])
        x_t = forward_diffusion(x_0, t_tensor, noise, 
                               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        
        # 보이기
        if x_0.shape[1] == 1:  # 회색
            img = x_t[0, 0].cpu().numpy()
            axes[idx].imshow(img, cmap='gray')
        else:  # RGB
            img = x_t[0].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[idx].imshow(img)
        
        axes[idx].set_title(f't={t}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('diffusion_process.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: diffusion_process.png")

def visualize_samples(samples: torch.Tensor, nrow: int = 8, 
                     filename: str = 'samples.png'):
    """만든 표본을 격자로 보인다."""
    from torchvision.utils import make_grid
    
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    grid = make_grid(samples, nrow=nrow, padding=2)
    
    plt.figure(figsize=(12, 12))
    if samples.shape[1] == 1:
        plt.imshow(grid[0].cpu().numpy(), cmap='gray')
    else:
        plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

# ============================================================================
# 때 박아 넣기(앞선 모델용)
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    때 걸음 조건 주기를 위한 사인 꼴 때 박아 넣기.
    
    왜 사인 꼴인가?
    ---------------
    Better than scalar t:
    - 이어져 있다: 가까운 때 걸음은 박아 넣기가 비슷하다
    - 하나뿐이다: 때 걸음마다 다르다
    - 매끄럽다: 작은 Δt → 작은 Δ박아 넣기
    - 되풀이된다: 사인과 코사인으로 풍부한 나타냄
    
    FORMULA (from Transformers):
    ----------------------------
    PE(t, 2i) = sin(t/10000^(2i/d))      even indices
    PE(t, 2i+1) = cos(t/10000^(2i/d))    odd indices
    
    차원마다 다른 잦기로 흔들린다.
    """
    
    def __init__(self, dim: int):
        """
        인수:
            dim: Embedding dimension (should be even)
                흔히: 64, 128, 256
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        때 걸음을 사인 꼴 박아 넣기로 바꾼다.
        
        인수:
            time: Timesteps, shape (batch_size,)
        
        반환값:
            Embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 잦기 잣수를 셈한다(등비 수열)
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 때 × 잦기를 셈한다
        embeddings = time[:, None] * embeddings[None, :]
        
        # 사인과 코사인을 쓴다
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ============================================================================
# 보여 주기
# ============================================================================

if __name__ == "__main__":
    """잡음 차례표를 견준다."""
    print("=" * 70)
    print("NOISE SCHEDULE COMPARISON")
    print("=" * 70)
    
    timesteps = 1000
    print(f"\nGenerating schedules for T={timesteps}...")
    
    betas_linear = linear_beta_schedule(timesteps)
    betas_cosine = cosine_beta_schedule(timesteps)
    
    print(f"\nLinear: [{betas_linear[0]:.6f}, {betas_linear[-1]:.6f}]")
    print(f"Cosine: [{betas_cosine[0]:.6f}, {betas_cosine[-1]:.6f}]")
    
    # 비교 그리기
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(betas_linear, label="Linear", alpha=0.7)
    axes[0].plot(betas_cosine, label="Cosine", alpha=0.7)
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('β_t')
    axes[0].set_title('Full Schedule')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(betas_linear[-100:], label="Linear (end)", alpha=0.7)
    axes[1].plot(betas_cosine[-100:], label="Cosine (end)", alpha=0.7)
    axes[1].set_xlabel('Timestep t')
    axes[1].set_title('Last 100 Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_schedule_comparison.png', dpi=150)
    print("\n✓ Saved: noise_schedule_comparison.png")
    
    print("\nKEY DIFFERENCES:")
    print("  Linear: Uniform increase, aggressive at end")
    print("  Cosine: Smooth increase, better for images")
    print("  Recommendation: Use cosine for better quality")
    print("=" * 70)```

## 논의

퍼짐 도구의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
