# 체크포인트를 이용한 MNIST 샘플링

트레인된 확산 모델을 생성에 사용하려면 웨이트뿐 아니라 모델 구조와 노이즈 스케줄도 트레인 당시와 맞아야 한다. 이 예제는 저장 파일을 찾아 EMA 모델로 이미지 격자를 만들고 초기 노이즈 사이의 보간을 시도한다. 보간 루프와 파일 이름에는 실제 동작을 따로 확인해야 하는 부분이 있다.

## 1. 코드

```python
"""
샘플링 스크립트 예제

트레인된 확산 모델에서 표본을 생성한다.
이 스크립트는 체크포인트를 불러와 새 MNIST 숫자를 생성한다.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- diffusion_utils ---
"""확산 유틸리티."""
# ============================================================================
# diffusion_utils.py - 확산 모델의 수학적 기초
# ============================================================================


"""
확산 모델 도구 - 핵심 개념
==========================================

이 모듈은 확산 모델의 수학 엔진을 담는다.

핵심 구성 요소:
---------------
1. 노이즈 스케줄: 타임스텝마다 노이즈를 얼마나 더할지 정한다(β_t)
2. 확산 파라메터: 효율을 위해 미리 계산한 수학 상수
3. 정방향 확산: 닫힌 형태 공식으로 노이즈 더하기
   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
4. 역방향 확산: 노이즈를 반복해 제거한다(생성)
5. 트레인 도구: 트레인을 위한 로스 계산
6. 시각화 도구: 확산을 눈으로 이해하기

수학적 기초:
------------------------
정방향 과정(데이터 → 노이즈):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

닫힌 형태 해:
    q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
    여기서 ᾱ_t = ∏_{i=1}^t (1-β_i)

역방향 과정(노이즈 → 데이터):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))

모델은 노이즈 ε_θ(x_t, t)을 예측해 μ_θ을 배운다.
"""

# ============================================================================
# 노이즈 스케줄
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, 
                        beta_end: float = 0.02) -> torch.Tensor:
    """
    선형 노이즈 스케줄를 만든다.
    
    β_t이 선형으로 늘어난다: [0.0001, 0.0002, ..., 0.02]
    
    장점: 단순하고 직관적이며 이해하기 쉽다
    단점: 이미지에는 최선이 아니고 끝에서 노이즈를 너무 빨리 더한다
    
    역사: 원래 DDPM 논문에서 썼다.
    요즘 방식: 이미지에는 코사인 스케줄가 낫다.
    
    인수:
        timesteps: 전체 확산 스텝 수(T)
        beta_start: 처음 노이즈 크기(≈0.0001)
        beta_end: 마지막 노이즈 크기(≈0.02)
    
    반환값:
        betas: 형태가 (timesteps,)인 텐서
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    코사인 노이즈 스케줄를 만든다(선형보다 개선된 것).
    
    왜 코사인인가?
    -----------
    - 확산 내내 더 매끄럽게 진행한다
    - 끝에서 덜 과감하다
    - 신호 대 잡음비를 더 잘 지킨다
    - 이미지 생성 품질이 높다
    
    수학적 정의:
    ------------------------
    f(t) = cos((t/T + s)/(1 + s) · π/2)²
    ᾱ_t = f(t)/f(0)
    β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    
    코사인은 ᾱ_t이 줄어드는 매끄러운 S자 곡선을 만든다.
    오프셋 's'이 가장자리에서 수치가 흔들리는 것을 막는다.
    
    장점:
    ---------
    1. 매끄러운 전이: 스텝마다 비슷한 양의 노이즈를 제거한다
    2. 더 나은 그래디언트: 타임스텝에 걸쳐 트레인 신호가 고르다
    3. 아티팩트가 적다: 급격한 변화가 덜하다
    4. 경험적 성공: 최고 수준의 모델이 이것을 쓴다
    
    참고: "Improved Denoising Diffusion Probabilistic Models"
               (니콜 & 다리왈, 2021)
    
    인수:
        timesteps: 전체 확산 스텝
        s: 작은 오프셋(기본값 0.008. 처음에 β_t ≈ 0이 되는 것을 막는다)
    
    반환값:
        betas: 형태 (timesteps,)인 텐서이며 매끄럽게 늘어난다
    """
    # 시간 인덱스를 만든다(차이를 계산하려면 T+1개 점이 필요하다)
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # 코사인 함수로 ᾱ_t을 계산한다
    t_normalized = (x / timesteps + s) / (1 + s)
    alphas_cumprod = torch.cos(t_normalized * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # ᾱ_0=1이 되게 정규화한다
    
    # ᾱ_t에서 β_t을 계산한다: β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # 수치 안정성을 위해 클리핑한다
    return torch.clip(betas, 0.0001, 0.9999)

# ============================================================================
# 확산 파라메터(효율을 위해 미리 계산함)
# ============================================================================

def get_diffusion_parameters(betas: torch.Tensor) -> dict:
    """
    확산에 필요한 수학 상수를 모두 미리 계산한다.
    
    왜 미리 계산하는가?
    ---------------
    트레인할 때 이 값이 반복해서 쓰인다. 한 번만 계산하면 O(T)이고,
    반복마다 O(T) 대신 O(1) 조회 → 엄청나게 빨라진다!
    
    T=1000, 반복 1만 번이면 계산 1000번 + 조회 1만 번 대 계산 1000만 번이다
    
    유도한 값:
    -------------------
    α_t = 1 - β_t                    (남는 신호)
    ᾱ_t = ∏_{i=1}^t α_i             (누적 신호)
    √ᾱ_t, √(1-ᾱ_t)                  (정방향 확산 계수)
    1/√α_t                           (역방향 확산 스케일)
    σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)  (사후분산)
    
    인수:
        betas: β 스케줄, 형태 (T,)
    
    반환값:
        미리 계산한 파라메터를 모두 담은 사전:
        - 'betas': β_t 값
        - 'alphas': α_t = 1 - β_t
        - 'alphas_cumprod': ᾱ_t (누적곱)
        - 'alphas_cumprod_prev': ᾱ_{t-1} (한 칸 옮김)
        - 'sqrt_alphas_cumprod': √ᾱ_t (데이터 계수)
        - 'sqrt_one_minus_alphas_cumprod': √(1-ᾱ_t) (노이즈 계수)
        - 'sqrt_recip_alphas': 1/√α_t (역방향 스케일)
        - 'posterior_variance': σ_t² (역방향 노이즈)
    """
    # α_t = 1 - β_t(스텝마다 남는 신호)
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏ α_i(누적 신호 - 핵심 양이다!)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # 성질: 줄어들고 ≈1에서 시작해 ≈0에서 끝난다
    
    # ᾱ_{t-1}(ᾱ_0 = 1이므로 앞에 1.0을 붙인다)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # 정방향 확산 계수: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # 역방향 확산 스케일 맞추기: μ = 1/√α_t·(...)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # 사후분산: 역방향으로 갈 때 노이즈를 얼마나 더할지
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
# 도구: 브로드캐스팅을 곁들인 배치 인덱스 조회
# ============================================================================

def extract(tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    1차원 텐서의 인덱스 t에서 값을 뽑고 브로드캐스팅에 맞게 형태를 바꾼다.
    
    문제:
    ------------
    배치마다 타임스텝이 다르다: t = [15, 73, 42, ...]
    표본마다 다른 상수가 필요하다: √ᾱ_15, √ᾱ_73, √ᾱ_42, ...
    
    풀이:
    ---------
    1. 인덱싱: tensor[t] → 표본마다 값을 얻는다
    2. 형태 바꾸기: 브로드캐스팅를 위해 (배치,) → (배치, 1, 1, ...)
    
    브로드캐스팅 예:
    ---------------------
    x 형태: (3, 2)          2차원 점 셋
    values: (3, 1)           형태를 바꾼 뒤
    
    [[x1, y1],     [[v1, 1],      [[v1·x1, v1·y1],
     [x2, y2],  *   [v2, 1],  =    [v2·x2, v2·y2],
     [x3, y3]]      [v3, 1]]       [v3·x3, v3·y3]]
    
    인수:
        tensor: 조회할 값, 형태 (T,)
        t: 타임스텝 번호, 형태 (batch_size,)
        x_shape: 브로드캐스팅의 목표 데이터 형태
    
    반환값:
        뽑아서 형태를 바꾼 값, 형태 (batch_size, 1, 1, ...)
    """
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)  # 인덱스에서 뽑는다
    
    # 브로드캐스팅에 맞게 형태를 바꾼다: (배치,) → (배치, 1, 1, ...)
    num_extra_dims = len(x_shape) - 1
    trailing_dims = (1,) * num_extra_dims
    return out.reshape(batch_size, *trailing_dims)

# ============================================================================
# 정방향 확산
# ============================================================================

def forward_diffusion(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                     sqrt_alphas_cumprod: torch.Tensor,
                     sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    정방향 확산을 적용한다: 깨끗한 데이터에 노이즈를 더한다.
    
    식:
    --------
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    
    풀이:
    ---------------
    데이터와 노이즈의 가중합:
    - t=0:   ᾱ_0=1    → x_0 = x_0 (노이즈 없음)
    - t=50:  ᾱ_50≈0.6 → 데이터 60% + 노이즈 80%(크기 기준)
    - t=100: ᾱ_100≈0.3 → 데이터 30% + 노이즈 95%
    - t→∞:   ᾱ_∞→0   → 순수 노이즈
    
    왜 제곱근인가?
    -----------------
    분산 보존: √ᾱ_t² + √(1-ᾱ_t)² = 1
    x_t의 분산이 x_0과 같게 한다!
    
    인수:
        x_0: 깨끗한 데이터, 형태 (batch_size, *data_dims)
        t: 타임스텝, 형태 (batch_size,)
        noise: 정규 노이즈 ε ~ N(0,I)이며 x_0과 같은 형태
        sqrt_alphas_cumprod: 미리 계산한 √ᾱ_t, 형태 (T,)
        sqrt_one_minus_alphas_cumprod: 미리 계산한 √(1-ᾱ_t), 형태 (T,)
    
    반환값:
        x_t: 타임스텝 t의 노이즈 섞인 데이터
    """
    # 표본마다 타임스텝의 계수를 뽑고 브로드캐스팅에 맞게 형태를 바꾼다
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # 정방향 확산 공식을 적용한다
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

# ============================================================================
# 역방향 확산(생성)
# ============================================================================

@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: int, t_tensor: torch.Tensor,
            diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    역방향 확산을 한 스텝 수행한다: x_t → x_{t-1}.
    
    역방향 식:
    ----------------
    μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
    x_{t-1} = μ_θ + σ_t·z  (t>0이면)
    x_0 = μ_θ              (t=0이면)
    
    과정:
    --------
    1. 모델이 노이즈를 예측한다: ε_θ(x_t,t)
    2. 디노이징한 평균을 계산한다: μ_θ
    3. 다양한 표본이 나오도록 작은 노이즈 σ_t·z을 더한다(t=0은 뺀다)
    
    왜 노이즈를 제거하면서 노이즈를 더하는가?
    -------------------------------
    노이즈가 없으면: 결정적 → 모드 붕괴, 다양성 없음
    노이즈가 있으면: 확률적 → 풍부한 표본, 완전한 분포
    핵심: 더한 노이즈 < 제거한 노이즈 → 전체로는 노이즈가 줄어든다!
    
    인수:
        model: 트레인된 디노이징 모델
        x_t: 타임스텝 t의 노이즈 섞인 데이터
        t: 현재 타임스텝(정수)
        t_tensor: 텐서로 나타낸 타임스텝, 형태 (batch_size,)
        diffusion_params: 미리 계산한 상수
        device: 'cpu' 또는 'cuda'
    
    반환값:
        x_{t-1}: 이전 타임스텝의, 노이즈가 덜한 데이터
    """
    # 모델로 노이즈를 예측한다
    predicted_noise = model(x_t, t_tensor)
    
    # 현재 타임스텝의 파라메터를 뽑는다
    beta_t = extract(diffusion_params['betas'], t_tensor, x_t.shape)
    sqrt_recip_alpha_t = extract(diffusion_params['sqrt_recip_alphas'], t_tensor, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t_tensor, x_t.shape
    )
    
    # 디노이징한 평균을 계산한다: μ = 1/√α_t·(x_t - β_t/√(1-ᾱ_t)·ε_pred)
    noise_term = beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
    model_mean = sqrt_recip_alpha_t * (x_t - noise_term)
    
    if t == 0:
        # 마지막 스텝: 노이즈를 더하지 않는다
        return model_mean
    else:
        # 다양성을 위해 확률적 노이즈를 더한다
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t_tensor, x_t.shape)
        noise = torch.randn_like(x_t)
        posterior_std = torch.sqrt(posterior_variance_t)
        return model_mean + posterior_std * noise

@torch.no_grad()
def sample(model: nn.Module, shape: tuple, timesteps: int,
          diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    전체 역방향 확산으로 표본을 생성한다.
    
    알고리즘:
    ----------
    x_T ~ N(0,I)           ← 순수 노이즈에서 시작
        ↓ 디노이징
    x_{T-1} = p_θ(· | x_T)
        ↓ ...
    x_0                    ← 마지막 깨끗한 표본
    
    트레이드오프:
    -----------
    스텝이 많으면(T이 큼): 품질이 높고 느리다
    스텝이 적으면(T이 작음): 빠르고 품질이 낮다
    
    흔한 값: T=100(빠름), T=1000(표준 값), T=4000(고품질)
    
    인수:
        model: 트레인된 디노이징 모델(평가 모드)
        shape: 표본의 형태. 보기로 (32, 2)이나 (16, 3, 256, 256)
        timesteps: 디노이징 스텝 수(트레인 때와 맞아야 한다!)
        diffusion_params: 미리 계산한 상수
        device: 'cpu' 또는 'cuda'
    
    반환값:
        주어진 형태의 생성 표본
    """
    model.eval()
    
    # 순수 노이즈에서 시작한다
    x_t = torch.randn(shape, device=device)
    
    # 반복 디노이징: T → T-1 → ... → 1 → 0
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t, t_tensor, diffusion_params, device)
    
    return x_t  # x_0: 깨끗한 생성 표본

# ============================================================================
# 트레인 로스
# ============================================================================

def get_loss(model: nn.Module, x_0: torch.Tensor, t: torch.Tensor,
            diffusion_params: dict, noise: torch.Tensor = None) -> torch.Tensor:
    """
    확산 모델의 트레인 로스를 계산한다.
    
    DDPM 트레인 목표:
    ------------------------
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]
    
    x_t을 만들려 더한 바로 그 노이즈를 모델이 예측하기를 바란다.
    
    왜 노이즈를 예측하는가?
    ------------------
    - 노이즈의 분포는 고정되어 있다: 모든 타임스텝에서 N(0,I)이다
    - t에 걸쳐 트레인 목표가 고르다
    - 경험적으로 표본 품질이 가장 높다
    - 요즘 확산 모델의 표준 방식이다
    
    무작위 타임스텝 샘플링:
    -------------------------
    핵심 기법: 배치마다 [0,T-1]에서 t을 균등하게 뽑는다
    → 모델이 모든 노이즈 수준에서 디노이징하는 법을 배운다
    → 특정 타임스텝에 과적합하는 것을 막는다
    
    인수:
        model: 트레인할 디노이징 모델
        x_0: 깨끗한 데이터의 배치, 형태 (batch_size, *data_dims)
        t: 무작위 타임스텝, 형태 (batch_size,)
        diffusion_params: 미리 계산한 상수
        noise: 미리 만든 노이즈(선택 사항. None이면 여기서 만든다)
    
    반환값:
        역전파를 위한 스칼라 평균 제곱 오차(MSE) 로스
    """
    # 노이즈가 주어지지 않으면 만든다
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # 정방향 확산을 적용한다: 알려진 노이즈로 x_t을 만든다
    x_t = forward_diffusion(
        x_0, t, noise,
        diffusion_params['sqrt_alphas_cumprod'],
        diffusion_params['sqrt_one_minus_alphas_cumprod']
    )
    
    # 모델이 노이즈를 예측한다
    predicted_noise = model(x_t, t)
    
    # 예측한 노이즈와 실제 노이즈 사이의 평균 제곱 오차 로스를 계산한다
    return nn.functional.mse_loss(predicted_noise, noise)

# ============================================================================
# 시각화 도구
# ============================================================================

def visualize_diffusion_process(x_0: torch.Tensor, timesteps: int,
                               diffusion_params: dict, num_images: int = 10):
    """
    이미지의 정방향 확산을 시각화한다(데이터 → 노이즈).
    
    차츰 망가지는 모습을 보인다: 깨끗함 → 약한 노이즈 → 강한 노이즈 → 순수 노이즈
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
        
        # 표시
        if x_0.shape[1] == 1:  # 흑백
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
    """생성한 표본을 격자로 표시한다."""
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
# 타임스텝 임베딩(고급 모델용)
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    타임스텝 조건화를 위한 사인파 타임스텝 임베딩.
    
    왜 사인파인가?
    ---------------
    스칼라 t보다 나은 점:
    - 연속적이다: 가까운 타임스텝은 임베딩이 비슷하다
    - 유일하다: 타임스텝마다 다르다
    - 매끄럽다: 작은 Δt → 작은 Δ임베딩
    - 주기적이다: 사인과 코사인으로 풍부한 표현
    
    식(트랜스포머에서 가져옴):
    ----------------------------
    PE(t, 2i) = sin(t/10000^(2i/d))      짝수 번호
    PE(t, 2i+1) = cos(t/10000^(2i/d))    홀수 번호
    
    차원마다 다른 주파수로 진동한다.
    """
    
    def __init__(self, dim: int):
        """
        인수:
            dim: 임베딩 차원(짝수여야 한다)
                흔히: 64, 128, 256
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        타임스텝을 사인파 임베딩으로 바꾼다.
        
        인수:
            time: 타임스텝, 형태 (batch_size,)
        
        반환값:
            임베딩, 형태 (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 주파수 스케일을 계산한다(등비 수열)
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 시간 × 주파수를 계산한다
        embeddings = time[:, None] * embeddings[None, :]
        
        # 사인과 코사인을 적용한다
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ============================================================================
# 데모
# ============================================================================

if __name__ == "__main__":
    """노이즈 스케줄를 견준다."""
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
    print("=" * 70)


# --- unet_architecture ---
"""
확산 모델을 위한 U-Net 아키텍처

이 모듈은 확산 모델에 흔히 쓰는 U-Net 아키텍처를 구현한다.
U-Net은 노이즈 섞인 이미지와 타임스텝 임베딩을 입력으로 받아 노이즈를 예측한다.
"""


# ========================================================================
# 메인
# ========================================================================


class ResidualBlock(nn.Module):
    """
    GroupNorm과 타임스텝 임베딩을 갖춘 잔차 블록.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 num_groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # 타임스텝 임베딩 프로젝션
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 잔차 연결
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 입력 텐서, 형태 (batch, in_channels, height, width)
            time_emb: 타임스텝 임베딩, 형태 (batch, time_emb_dim)
        
        반환값:
            출력 텐서, 형태 (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)
        
        # 첫 합성곱
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # 타임스텝 임베딩을 더한다
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]  # 공간 차원으로 브로드캐스팅한다
        
        # 두 번째 합성곱
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """
    장거리 의존성을 포착하는 셀프 어텐션 블록.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 입력 텐서, 형태 (batch, channels, height, width)
        
        반환값:
            입력과 같은 형태의 출력 텐서
        """
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 멀티 헤드 어텐션에 맞게 형태를 바꾼다
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, 배치, 헤드, hw, 차원)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 어텐션
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (channels // self.num_heads) ** 0.5
        attn = F.softmax(attn, dim=-1)
        
        # 밸류에 어텐션 적용
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        
        # 프로젝션한 뒤 잔차를 더한다
        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    """합성곱을 쓴 다운샘플링 레이어."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """전치 합성곱을 쓴 업샘플링 레이어."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    확산 모델을 위한 U-Net 아키텍처.
    
    신경망은 다음을 지닌다.
    - 다운샘플링을 갖춘 인코더 경로
    - 어텐션을 갖춘 병목
    - 업샘플링과 스킵 연결을 갖춘 디코더 경로
    - 타임스텝을 조건으로 삼는 타임스텝 임베딩
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 base_channels: int = 64,
                 channel_multipliers: tuple = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 attention_resolutions: tuple = (16,),
                 dropout: float = 0.0,
                 time_emb_dim: int = 256):
        """
        인수:
            in_channels: 입력 채널의 개수
            out_channels: 출력 채널의 개수
            base_channels: 기본 채널 수
            channel_multipliers: 해상도 단계마다 채널 배수
            num_res_blocks: 해상도마다 잔차 블록의 수
            attention_resolutions: 어텐션을 쓸 해상도
            dropout: 드롭아웃 확률
            time_emb_dim: 타임스텝 임베딩의 차원
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 타임스텝 임베딩
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 첫 합성곱
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 인코더
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                block = ResidualBlock(now_channels, out_ch, time_emb_dim)
                self.encoder_blocks.append(block)
                now_channels = out_ch
                channels.append(now_channels)
            
            # 마지막 단계만 빼고 다운샘플링을 더한다
            if i != len(channel_multipliers) - 1:
                self.downsamples.append(Downsample(now_channels))
                channels.append(now_channels)
        
        # 병목
        self.bottleneck = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim),
        ])
        
        # 디코더
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                # 인코더에서 오는 스킵 연결
                skip_ch = channels.pop()
                block = ResidualBlock(now_channels + skip_ch, out_ch, time_emb_dim)
                self.decoder_blocks.append(block)
                now_channels = out_ch
            
            # 마지막 단계만 빼고 업샘플링을 더한다
            if i != len(channel_multipliers) - 1:
                self.upsamples.append(Upsample(now_channels))
        
        # 출력
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        U-Net을 지나는 순전파.
        
        인수:
            x: 노이즈 섞인 입력, 형태 (batch, in_channels, height, width)
            time: 타임스텝, 형태 (batch,)
        
        반환값:
            예측한 노이즈, 형태 (batch, out_channels, height, width)
        """
        # 타임스텝 임베딩
        time_emb = self.time_embedding(time)
        
        # 첫 합성곱
        x = self.conv_in(x)
        
        # 인코더
        encoder_outputs = [x]
        
        down_idx = 0
        for block in self.encoder_blocks:
            x = block(x, time_emb)
            encoder_outputs.append(x)
        
        for downsample in self.downsamples:
            x = downsample(x)
            encoder_outputs.append(x)
        
        # 병목
        for block in self.bottleneck:
            if isinstance(block, AttentionBlock):
                x = block(x)
            else:
                x = block(x, time_emb)
        
        # 디코더
        up_idx = 0
        for block in self.decoder_blocks:
            skip = encoder_outputs.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb)
        
        for upsample in self.upsamples:
            x = upsample(x)
        
        # 출력
        x = self.conv_out(x)
        
        return x


class SimpleUNet(nn.Module):
    """
    MNIST 같은 작은 데이터셋에서 더 빨리 트레인하기 위한 단순한 U-Net.
    배우기에 좋은 출발점이다.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        
        # 타임스텝 임베딩
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 인코더
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down1 = Downsample(base_channels * 2)
        
        self.res2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down2 = Downsample(base_channels * 4)
        
        # 병목
        self.res3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # 디코더
        self.up1 = Upsample(base_channels * 4)
        self.res4 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        
        self.up2 = Upsample(base_channels * 2)
        self.res5 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        
        # 출력
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(time)
        
        # 인코더
        x1 = self.conv1(x)
        x2 = self.res1(x1, time_emb)
        x2_down = self.down1(x2)
        
        x3 = self.res2(x2_down, time_emb)
        x3_down = self.down2(x3)
        
        # 병목
        x4 = self.res3(x3_down, time_emb)
        
        # 스킵 연결을 갖춘 디코더
        x5 = self.up1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.res4(x5, time_emb)
        
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.res5(x6, time_emb)
        
        return self.conv_out(x6)


if __name__ == "__main__":
    # 모델을 시험한다
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SimpleUNet을 시험한다
    model = SimpleUNet().to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    out = model(x, t)
    print(f"SimpleUNet output shape: {out.shape}")
    
    # 전체 UNet을 시험한다
    model = UNet(in_channels=1, base_channels=32).to(device)
    out = model(x, t)
    print(f"UNet output shape: {out.shape}")
    
    # 파라메터 개수 세기
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")


# --- mnist_diffusion ---
"""
MNIST 확산 모델

MNIST 숫자 생성을 위한 확산 모델의 완전한 구현.
학부생을 위한 실제적인 이미지 생성 예제다.
"""


# ========================================================================
# 메인
# ========================================================================



class MNISTDiffusion:
    """
    MNIST 확산 모델의 트레인과 샘플링을 감싸는 클래스.
    """
    
    def __init__(self, 
                 timesteps: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 2e-4,
                 device: str = None):
        """
        인수:
            timesteps: 확산 타임스텝 수
            batch_size: 트레인 배치 크기
            learning_rate: 옵티마이저의 러닝레이트
            device: 트레인할 장치('cuda' 또는 'cpu')
        """
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 확산 파라메터를 준비한다
        betas = cosine_beta_schedule(timesteps)
        self.diffusion_params = get_diffusion_parameters(betas)
        
        # 파라메터를 장치로 옮긴다
        for key in self.diffusion_params:
            self.diffusion_params[key] = self.diffusion_params[key].to(self.device)
        
        # 모델을 초기화한다
        self.model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # 더 나은 표본을 위한 지수 이동 평균 모델
        self.ema_model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = 0.9999
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def update_ema(self):
        """모델 파라메터의 지수 이동 평균을 갱신한다."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), 
                                        self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_dataloader(self, train: bool = True):
        """
        MNIST 데이터 로더를 만든다.
        
        인수:
            train: 트레인 배치를 불러올지 시험 배치를 불러올지
        
        반환값:
            MNIST용 DataLoader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]로 정규화
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, dataloader):
        """한 에포크를 트레인한다."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            
            # 무작위 타임스텝을 뽑는다
            t = torch.randint(
                0, self.timesteps,
                (images.shape[0],),
                device=self.device
            )
            
            # 로스를 계산한다
            loss = get_loss(self.model, images, t, self.diffusion_params)
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 지수 이동 평균을 갱신한다
            self.update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def sample_images(self, n_samples: int = 64, use_ema: bool = True):
        """
        표본 이미지를 생성한다.
        
        인수:
            n_samples: 만들 표본의 개수
            use_ema: 지수 이동 평균 모델을 쓸지 여부
        
        반환값:
            생성한 이미지 텐서
        """
        model = self.ema_model if use_ema else self.model
        
        samples = sample(
            model,
            shape=(n_samples, 1, 28, 28),
            timesteps=self.timesteps,
            diffusion_params=self.diffusion_params,
            device=self.device
        )
        
        return samples
    
    def train(self, epochs: int, save_interval: int = 10):
        """
        확산 모델을 트레인한다.
        
        인수:
            epochs: 트레인 에포크 수
            save_interval: N에포크마다 표본을 저장한다
        """
        dataloader = self.get_dataloader(train=True)
        
        # 표본을 담을 자리를 만든다
        os.makedirs('samples', exist_ok=True)
        
        losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # 트레인
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본을 생성해 저장한다
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(n_samples=64)
                visualize_samples(
                    samples,
                    nrow=8,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # 체크포인트 저장
            if epoch % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 트레인 로스 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MNIST Diffusion Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved training loss plot to training_loss.png")
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print("=" * 50)
    
    def save_checkpoint(self, filename: str):
        """모델 체크포인트를 저장한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_params': self.diffusion_params,
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 체크포인트를 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


def main():
    """
    MNIST 확산의 주 트레인 스크립트.
    """
    print("=" * 50)
    print("MNIST Diffusion Model Training")
    print("=" * 50)
    
    # 확산 모델을 초기화한다
    mnist_diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=128,
        learning_rate=2e-4
    )
    
    # 트레인
    mnist_diffusion.train(epochs=100, save_interval=10)
    
    # 마지막 표본을 만든다
    print("\nGenerating final samples...")
    samples = mnist_diffusion.sample_images(n_samples=64)
    visualize_samples(samples, nrow=8, filename='final_samples.png')
    
    # 마지막 모델을 저장한다
    mnist_diffusion.save_checkpoint('mnist_diffusion_final.pt')
    
    print("\nAll done! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()

# ========================================================================
# 메인
# ========================================================================


def generate_grid(diffusion, n_samples=64, filename='generated_grid.png'):
    """표본 격자를 만든다."""
    print(f"Generating {n_samples} samples...")
    samples = diffusion.sample_images(n_samples=n_samples, use_ema=True)
    visualize_samples(samples, nrow=8, filename=filename)
    print(f"Saved to {filename}")


def generate_interpolation(diffusion, n_steps=10):
    """무작위 출발점 사이를 보간해 생성한다."""
    print(f"Generating interpolation with {n_steps} steps...")
    
    # 무작위 노이즈 벡터 둘을 만든다
    noise1 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    noise2 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    
    # 보간
    alphas = torch.linspace(0, 1, n_steps)
    interpolated_samples = []
    
    for alpha in alphas:
        # 노이즈 공간에서의 선형 보간
        noise = (1 - alpha) * noise1 + alpha * noise2
        
        # 이 출발점에서 노이즈를 제거한다
        x_t = noise
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((1,), t, device=diffusion.device, dtype=torch.long)
            
            predicted_noise = diffusion.ema_model(x_t, t_tensor)
            
            beta_t = diffusion.diffusion_params['betas'][t]
            sqrt_recip_alpha_t = diffusion.diffusion_params['sqrt_recip_alphas'][t]
            sqrt_one_minus_alpha_cumprod_t = diffusion.diffusion_params['sqrt_one_minus_alphas_cumprod'][t]
            
            mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            
            if t > 0:
                posterior_variance_t = diffusion.diffusion_params['posterior_variance'][t]
                noise_sample = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance_t) * noise_sample
            else:
                x_t = mean
        
        interpolated_samples.append(x_t)
    
    # 시각화한다
    samples = torch.cat(interpolated_samples, dim=0)
    visualize_samples(samples, nrow=n_steps, filename='interpolation.png')
    print("Saved to interpolation.png")


def main():
    # 체크포인트가 있는지 확인한다
    checkpoint_files = [
        'mnist_diffusion_model.pt',
        'mnist_diffusion_final.pt',
        'checkpoint_epoch_100.pt'
    ]
    
    checkpoint = None
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            checkpoint = ckpt_file
            break
    
    if checkpoint is None:
        print("Error: No checkpoint found!")
        print("Please train a model first using example_train.py")
        print("Looking for one of:", checkpoint_files)
        return
    
    print(f"Loading checkpoint: {checkpoint}")
    print("-" * 50)
    
    # 모델을 초기화한다
    diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=64,
        learning_rate=2e-4
    )
    
    # 체크포인트를 불러온다
    diffusion.load_checkpoint(checkpoint)
    
    print("\nGenerating samples...")
    print("-" * 50)
    
    # 여러 표본을 만든다
    generate_grid(diffusion, n_samples=64, filename='samples_8x8.png')
    generate_grid(diffusion, n_samples=100, filename='samples_10x10.png')
    
    # 보간을 생성한다
    generate_interpolation(diffusion, n_steps=10)
    
    # 다양성을 확인하려 큰 배치를 만든다
    print("\nGenerating large batch for visualization...")
    samples = diffusion.sample_images(n_samples=256)
    visualize_samples(samples, nrow=16, filename='samples_large.png')
    
    print("\n" + "=" * 50)
    print("Sampling complete!")
    print("Generated files:")
    print("  - samples_8x8.png: 64 samples in 8x8 grid")
    print("  - samples_10x10.png: 100 samples in 10x10 grid")
    print("  - samples_large.png: 256 samples")
    print("  - interpolation.png: Smooth interpolation between digits")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

**출력:**

```
Error: No checkpoint found!
Please train a model first using example_train.py
Looking for one of: ['mnist_diffusion_model.pt', 'mnist_diffusion_final.pt', 'checkpoint_epoch_100.pt']
```

이 출력은 후보 체크포인트가 없어 생성 전에 종료한 경우다. 트레인 완료나 샘플 생성 성공을 보여 주지 않는다.

## 2. 체크포인트 선택

작업 디렉터리에서 mnist_diffusion_model.pt, mnist_diffusion_final.pt, checkpoint_epoch_100.pt 순으로 존재 여부를 확인하고 첫 파일을 선택한다. 수정 시각이나 트레인 성능을 비교해 최신·최상 모델을 고르는 방식은 아니다. 모두 없으면 안내 메시지와 함께 반환한다.

로드할 객체는 T=1000과 기본 코사인 스케줄, 같은 SimpleUNet 구조로 생성한다. load_checkpoint는 트레인·EMA 웨이트와 optimizer 상태를 복원하지만 저장된 diffusion_params는 사용하지 않는다. 따라서 다른 스케줄로 트레인한 파일을 선택하면 생성 조건이 어긋날 수 있다.

## 3. 이미지 격자와 표시 범위

generate_grid는 EMA 모델로 지정 수량을 생성하지만 nrow는 항상 8로 전달한다. 64개는 8×8 격자지만 100개는 8열·13행의 격자에 배치되며 나머지 칸은 비어 있다. **samples_10x10.png라는 파일 이름이나 마지막 출력 문구와 달리 실제 10×10 배치는 아니다.** 256개 생성 부분은 nrow=16을 별도로 전달해 16×16 격자를 만든다.

생성 텐서는 B×1×28×28이며 픽셀 범위를 강제로 제한하지 않는다. visualize_samples는 배치 최솟값과 최댓값으로 표시 범위를 조정한다. 서로 다른 격자의 명암이 비슷하게 보이더라도 원래 값의 범위가 같다는 뜻은 아니다.

## 4. 초기 노이즈 보간의 의미

두 독립 표준정규 텐서 z₁, z₂에 대해 z(a)=(1−a)z₁+az₂를 만든다. 보간점의 좌표별 분산은 (1−a)²+a²이고, 중간점 a=1/2에서는 1/2이다. 따라서 선형 보간 경로 전체가 트레인 시 가정한 단위 분산 초기 분포를 유지하지는 않는다.

또한 각 a에 대해 역과정의 모든 중간 단계에서 **새 노이즈를 독립적으로 뽑는다**. 이웃 결과의 차이는 초기점 변화와 역과정 난수 변화가 섞인 결과다. 출력 문구의 “Smooth interpolation”처럼 숫자의 의미가 매끄럽게 변한다고 보장할 수 없다. 조건을 분리하려면 초기점과 역과정 난수의 제어 방법을 함께 설계해야 한다.

## 5. 보간 루프의 계산 그래프

generate_grid가 호출하는 sample_images에는 no_grad가 있지만 generate_interpolation 자체에는 해당 문맥이 없다. EMA 모델도 파라미터의 requires_grad가 켜져 있으므로 1,000단계의 모델 호출을 따라 계산 그래프가 쌓일 수 있다. 단순 추론으로 기대한 것보다 메모리 사용이 커지거나 실패할 수 있다.

main에서는 격자 생성이 먼저 평가 모드를 설정하지만, eval은 자동 미분을 끄지 않는다. 이 두 설정의 역할을 구분해야 한다. 저장된 성공 결과나 실제 메모리·시간 측정이 없는 상태에서 보간 완료를 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
100개를 generate_grid로 생성할 때 실제 격자 크기와 파일 이름을 비교하라.

</div>

??? success "풀이"

    함수는 nrow=8을 고정하므로 13행·8열에 100개를 배치한다. 마지막 네 칸은 비어 있다. samples_10x10.png라는 이름이 10행·10열을 구현하지는 않는다. 코드의 인수와 표시 문구를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
초기 노이즈 선형 보간의 중간점 분산을 구하고, 결과가 매끄럽지 않을 수 있는 다른 이유를 설명하라.

</div>

??? success "풀이"

    독립 z₁,z₂의 각 좌표 분산이 1이면 (z₁+z₂)/2의 분산은 1/2이다. 다른 보간 계수에서도 단위 분산이 유지되지 않는다. 역과정 노이즈도 각 보간점에서 새로 뽑으므로 관찰된 차이가 초기점 변화만의 효과가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
출력이 의미하는 실행 경로와 보간 함수의 추론 설정 문제를 설명하라.

</div>

??? success "풀이"

    출력은 세 후보 파일이 모두 없어 main이 조기 반환한 경우다. 생성 성공의 증거가 아니다. 보간 함수에는 no_grad가 없어 EMA 파라미터를 통한 계산 그래프가 누적될 수 있으며 eval만으로 이를 막을 수 없다. 체크포인트의 스케줄과 생성자의 스케줄도 별도로 맞아야 한다.



## 정리하며

샘플링에서는 체크포인트 선택 순서와 확산 설정, 실제 격자 열 수를 확인한다. 초기 노이즈의 선형 보간은 단위 분산을 유지하지 않으며, 독립적인 역과정 노이즈와 추론 그래프 누적도 결과 해석과 메모리에 영향을 준다.
