# MNIST 확산 모델 학습 파이프라인

이 페이지는 앞에서 정의한 SimpleUNet과 확산 유틸리티를 연결한다. MNIST 이미지에 노이즈를 섞어 예측기를 학습하고, EMA 모델로 이미지를 생성하며, 웨이트와 optimizer 상태를 저장하는 흐름이다. 코드에 적힌 학습 설정과 실제로 검증된 생성 품질은 구분해야 한다.

## 1. 코드

```python
"""
MNIST 확산 모델

MNIST 숫자 만들어 내기를 위한 확산 모델의 온전한 짜기.
학부생을 위한 그럴듯한 그림 만들어 내기 보기를 준다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ========================================================================
# 메인
# ========================================================================

import numpy as np
import torch.nn.functional as F

# --- diffusion_utils ---
"""확산 도구."""
# ============================================================================
# diffusion_utils.py - 확산 모델의 수학 바탕
# ============================================================================


"""
확산 모델 도구 - 핵심 개념
==========================================

이 단원은 확산 모델의 수학 엔진을 담는다.

고갱이 조각:
---------------
1. 잡음 일정: 때 걸음마다 잡음을 얼마나 더할지 매긴다(β_t)
2. 확산 매개변수: 효율을 위해 미리 셈한 수학 상수
3. 앞 확산: 닫힌 꼴 공식으로 잡음 더하기
   x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
4. 뒤 확산: 잡음을 거듭 걷어 낸다(만들기)
5. 익히기 도구: 배우기 위한 손실 셈하기
6. 그려 보기 연장: 확산을 눈으로 이해하기

수학 바탕:
------------------------
앞 과정(자료 → 잡음):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

닫힌 꼴 풀이:
    q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
    여기서 ᾱ_t = ∏_{i=1}^t (1-β_i)

뒤 과정(잡음 → 자료):
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
    
    β_t이 선형으로 늘어난다: [0.0001, 0.0002, ..., 0.02]
    
    좋은 점: 단순하고 직관적이며 이해하기 쉽다
    나쁜 점: 그림에는 가장 좋지 않고 끝에서 잡음을 너무 빨리 더한다
    
    역사 참고: 본디 DDPM 논문에서 썼다.
    요즘 방식: 그림에는 코사인 차례표가 낫다.
    
    인수:
        timesteps: 온 확산 걸음 수(T)
        beta_start: 처음 잡음 크기(≈0.0001)
        beta_end: 마지막 잡음 크기(≈0.02)
    
    반환값:
        betas: 꼴이 (timesteps,)인 텐서
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
    
    수학 뜻매김:
    ------------------------
    f(t) = cos((t/T + s)/(1 + s) · π/2)²
    ᾱ_t = f(t)/f(0)
    β_t = 1 - (ᾱ_t/ᾱ_{t-1})
    
    코사인은 ᾱ_t이 줄어드는 매끄러운 S자 곡선을 만든다.
    치우침 's'이 가장자리에서 수치가 흔들리는 것을 막는다.
    
    좋은 점:
    ---------
    1. 매끄러운 옮아감: 걸음마다 비슷한 양의 잡음을 없앤다
    2. 더 나은 기울기: 때 걸음에 걸쳐 배움 신호가 고르다
    3. 흠이 적다: 갑작스러운 바뀜이 덜하다
    4. 겪어 본 성공: 최고 수준의 모델이 이것을 쓴다
    
    참고: "Improved Denoising Diffusion Probabilistic Models"
               (니콜 & 다리왈, 2021)
    
    인수:
        timesteps: 온 확산 걸음
        s: 작은 치우침(기본값 0.008. 처음에 β_t ≈ 0이 되는 것을 막는다)
    
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
# 확산 매개변수(효율을 위해 미리 셈함)
# ============================================================================

def get_diffusion_parameters(betas: torch.Tensor) -> dict:
    """
    확산에 필요한 수학 상수를 모두 미리 셈한다.
    
    왜 미리 셈하는가?
    ---------------
    익힐 때 이 값이 거듭 든다. 한 번만 셈하면 O(T)이고,
    되풀이마다 O(T) 대신 O(1) 찾아보기 → 엄청나게 빨라진다!
    
    T=1000, 되돌이 1만 번이면 셈 1000번 + 찾아보기 1만 번 대 셈 1000만 번이다
    
    이끌어 낸 값:
    -------------------
    α_t = 1 - β_t                    (남는 신호)
    ᾱ_t = ∏_{i=1}^t α_i             (쌓은 신호)
    √ᾱ_t, √(1-ᾱ_t)                  (앞 확산 계수)
    1/√α_t                           (뒤 확산 잣대)
    σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)  (사후 흩어짐)
    
    인수:
        betas: β 일정, 꼴 (T,)
    
    반환값:
        미리 셈한 매개변수를 모두 담은 사전:
        - 'betas': β_t 값
        - 'alphas': α_t = 1 - β_t
        - 'alphas_cumprod': ᾱ_t (쌓은 곱)
        - 'alphas_cumprod_prev': ᾱ_{t-1} (한 칸 옮김)
        - 'sqrt_alphas_cumprod': √ᾱ_t (자료 계수)
        - 'sqrt_one_minus_alphas_cumprod': √(1-ᾱ_t) (잡음 계수)
        - 'sqrt_recip_alphas': 1/√α_t (뒤 잣대)
        - 'posterior_variance': σ_t² (뒤 잡음)
    """
    # α_t = 1 - β_t(걸음마다 남는 신호)
    alphas = 1.0 - betas
    
    # ᾱ_t = ∏ α_i(쌓인 신호 - 핵심 양이다!)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # 성질: 줄어들고 ≈1에서 시작해 ≈0에서 끝난다
    
    # ᾱ_{t-1}(ᾱ_0 = 1이므로 앞에 1.0을 붙인다)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
    # 앞 확산 계수: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # 뒤 확산 잣수 맞추기: μ = 1/√α_t·(...)
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
# 도구: 퍼뜨리기를 곁들인 배치 어깨수 찾기
# ============================================================================

def extract(tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    1차원 텐서의 어깨수 t에서 값을 뽑고 퍼뜨리기에 맞게 꼴을 바꾼다.
    
    문제:
    ------------
    배치마다 때 걸음이 다르다: t = [15, 73, 42, ...]
    표본마다 다른 상수가 필요하다: √ᾱ_15, √ᾱ_73, √ᾱ_42, ...
    
    풀이:
    ---------
    1. 자리 잡기: tensor[t] → 표본마다 값을 얻는다
    2. 꼴 바꾸기: 퍼뜨리기를 위해 (배치,) → (배치, 1, 1, ...)
    
    펴 맞추기 보기:
    ---------------------
    x 꼴: (3, 2)          2차원 점 셋
    values: (3, 1)           꼴을 바꾼 뒤
    
    [[x1, y1],     [[v1, 1],      [[v1·x1, v1·y1],
     [x2, y2],  *   [v2, 1],  =    [v2·x2, v2·y2],
     [x3, y3]]      [v3, 1]]       [v3·x3, v3·y3]]
    
    인수:
        tensor: 밑 값, 꼴 (T,)
        t: 때 걸음 번호, 꼴 (batch_size,)
        x_shape: 퍼뜨리기의 목표 자료 꼴
    
    반환값:
        뽑아서 꼴을 바꾼 값, 꼴 (batch_size, 1, 1, ...)
    """
    batch_size = t.shape[0]
    out = tensor.gather(-1, t)  # 어깨수에서 뽑는다
    
    # 퍼뜨리기에 맞게 꼴을 바꾼다: (배치,) → (배치, 1, 1, ...)
    num_extra_dims = len(x_shape) - 1
    trailing_dims = (1,) * num_extra_dims
    return out.reshape(batch_size, *trailing_dims)

# ============================================================================
# 앞 확산
# ============================================================================

def forward_diffusion(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                     sqrt_alphas_cumprod: torch.Tensor,
                     sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    앞 확산을 쓴다: 깨끗한 자료에 잡음을 더한다.
    
    식:
    --------
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    
    풀이:
    ---------------
    자료와 잡음의 무게 실은 합:
    - t=0:   ᾱ_0=1    → x_0 = x_0 (잡음 없음)
    - t=50:  ᾱ_50≈0.6 → 자료 60% + 잡음 80%(크기 기준)
    - t=100: ᾱ_100≈0.3 → 자료 30% + 잡음 95%
    - t→∞:   ᾱ_∞→0   → 순수 잡음
    
    왜 제곱근인가?
    -----------------
    흩어짐 지킴: √ᾱ_t² + √(1-ᾱ_t)² = 1
    x_t의 흩어짐이 x_0과 같게 한다!
    
    인수:
        x_0: 깨끗한 자료, 꼴 (batch_size, *data_dims)
        t: 때 걸음, 꼴 (batch_size,)
        noise: 정규 잡음 ε ~ N(0,I)이며 x_0과 같은 꼴
        sqrt_alphas_cumprod: 미리 셈한 √ᾱ_t, 꼴 (T,)
        sqrt_one_minus_alphas_cumprod: 미리 셈한 √(1-ᾱ_t), 꼴 (T,)
    
    반환값:
        x_t: 때 걸음 t의 잡음 섞인 자료
    """
    # 표본마다 때 걸음의 계수를 뽑고 퍼뜨리기에 맞게 꼴을 바꾼다
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # 앞 확산 공식을 쓴다
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

# ============================================================================
# 뒤 확산(만들어 내기)
# ============================================================================

@torch.no_grad()
def p_sample(model: nn.Module, x_t: torch.Tensor, t: int, t_tensor: torch.Tensor,
            diffusion_params: dict, device: str = 'cpu') -> torch.Tensor:
    """
    뒤 확산을 한 걸음 한다: x_t → x_{t-1}.
    
    뒤 식:
    ----------------
    μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
    x_{t-1} = μ_θ + σ_t·z  (t>0이면)
    x_0 = μ_θ              (t=0이면)
    
    과정:
    --------
    1. 모형이 잡음을 예측한다: ε_θ(x_t,t)
    2. 잡음 없앤 평균을 셈한다: μ_θ
    3. 여러 갈래가 나오도록 작은 잡음 σ_t·z을 더한다(t=0은 뺀다)
    
    왜 잡음을 없애면서 잡음을 더하는가?
    -------------------------------
    잡음이 없으면: 정해짐 → 봉우리 무너짐, 다양함 없음
    잡음이 있으면: 확률 → 풍부한 표본, 온전한 분포
    고갱이: 더한 잡음 < 걷어 낸 잡음 → 알짜로 나아간다!
    
    인수:
        model: 익힌 잡음 없애기 모델
        x_t: 때 걸음 t의 잡음 섞인 자료
        t: 이제 때 걸음(정수)
        t_tensor: 텐서로 나타낸 때 걸음, 꼴 (batch_size,)
        diffusion_params: 미리 셈한 상수
        device: 'cpu' 또는 'cuda'
    
    반환값:
        x_{t-1}: 앞선 때 걸음의, 잡음이 덜한 자료
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
    온전한 뒤 확산으로 표본을 만든다.
    
    알고리즘:
    ----------
    x_T ~ N(0,I)           ← 순수 잡음에서 시작
        ↓ 잡음 없애기
    x_{T-1} = p_θ(· | x_T)
        ↓ ...
    x_0                    ← 마지막 깨끗한 표본
    
    맞바꿈:
    -----------
    걸음이 많으면(T이 큼): 품질이 높고 느리다
    걸음이 적으면(T이 작음): 빠르고 품질이 낮다
    
    흔한 값: T=100(빠름), T=1000(여느 값), T=4000(높은 좋음)
    
    인수:
        model: 익힌 잡음 걷개 모형(따짐 모드)
        shape: 표본의 꼴. 보기로 (32, 2)이나 (16, 3, 256, 256)
        timesteps: 잡음을 걷는 걸음 수(익힘과 맞아야 한다!)
        diffusion_params: 미리 셈한 상수
        device: 'cpu' 또는 'cuda'
    
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
    확산 모델의 익히기 손실을 셈한다.
    
    DDPM 익힘 목표:
    ------------------------
    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]
    
    x_t을 만들려 더한 바로 그 잡음을 모델이 헤아리기를 바란다.
    
    왜 잡음을 헤아리는가?
    ------------------
    - 잡음은 멈춰 있다: 모든 때 걸음에서 N(0,I)이다
    - t에 걸쳐 배움 목표가 고르다
    - 겪어 보니 표본의 좋음이 가장 높다
    - 요즘 확산 모델의 여느 방식이다
    
    마구잡이 때 걸음 뽑기:
    -------------------------
    고갱이 솜씨: 배치마다 [0,T-1]에서 t을 고르게 뽑는다
    → 모델이 모든 잡음 수준에서 잡음 없애는 법을 배운다
    → 특정 때 걸음에 지나치게 맞춰지는 것을 막는다
    
    인수:
        model: 익힐 잡음 없애기 모델
        x_0: 깨끗한 자료의 배치, 꼴 (batch_size, *data_dims)
        t: 마구잡이 때 걸음, 꼴 (batch_size,)
        diffusion_params: 미리 셈한 상수
        noise: 미리 만든 잡음(골라 쓴다. None이면 여기서 만든다)
    
    반환값:
        뒤먹임 퍼뜨리기를 위한 낱값 평균 제곱 어긋남 손실
    """
    # 잡음이 주어지지 않으면 만든다
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # 앞 확산을 쓴다: 아는 잡음으로 x_t을 만든다
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
    그림의 앞 확산을 그려 본다(자료 → 잡음).
    
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
# 때 임베딩(앞선 모델용)
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    때 걸음 조건 주기를 위한 사인 꼴 때 임베딩.
    
    왜 사인 꼴인가?
    ---------------
    홑값 t보다 나은 점:
    - 이어져 있다: 가까운 때 걸음은 임베딩이 비슷하다
    - 하나뿐이다: 때 걸음마다 다르다
    - 매끄럽다: 작은 Δt → 작은 Δ임베딩
    - 되풀이된다: 사인과 코사인으로 풍부한 나타냄
    
    식(변환기에서 가져옴):
    ----------------------------
    PE(t, 2i) = sin(t/10000^(2i/d))      짝수 번호
    PE(t, 2i+1) = cos(t/10000^(2i/d))    홀수 번호
    
    차원마다 다른 잦기로 흔들린다.
    """
    
    def __init__(self, dim: int):
        """
        인수:
            dim: 묻힘 차원(짝수여야 한다)
                흔히: 64, 128, 256
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        때 걸음을 사인 꼴 임베딩으로 바꾼다.
        
        인수:
            time: 때 걸음, 꼴 (batch_size,)
        
        반환값:
            묻힘, 꼴 (batch_size, dim)
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
    print("=" * 70)


# --- unet_architecture ---
"""
확산 모델을 위한 U-Net 얼개

이 단원은 확산 모델에 흔히 쓰는 U-Net 얼개를 짠다.
U-Net은 잡음 섞인 그림과 때 걸음 임베딩을 들임으로 받아 잡음을 헤아린다.
"""


# ========================================================================
# 메인
# ========================================================================


class ResidualBlock(nn.Module):
    """
    GroupNorm과 때 임베딩을 갖춘 남은 덩이.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 num_groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # 때 임베딩 쏘기
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 잔차 연결
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: 들임 텐서, 꼴 (batch, in_channels, height, width)
            time_emb: 때 묻힘, 꼴 (batch, time_emb_dim)
        
        반환값:
            내놓음 텐서, 꼴 (batch, out_channels, height, width)
        """
        residual = self.residual_conv(x)
        
        # 첫 누비기
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # 때 임베딩을 더한다
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]  # 공간 차원으로 퍼뜨린다
        
        # 두 번째 누비기
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """
    멀리 떨어진 매임을 담는 셀프 어텐션 덩이.
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
            x: 들임 텐서, 꼴 (batch, channels, height, width)
        
        반환값:
            들임과 같은 꼴의 내놓기 텐서
        """
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 다중 머리 주의에 맞게 꼴을 바꾼다
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, 배치, 머리, hw, 차원)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 어텐션
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (channels // self.num_heads) ** 0.5
        attn = F.softmax(attn, dim=-1)
        
        # 값에 어텐션 적용
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        
        # 쏘고 남은 것을 더한다
        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    """겹말기를 쓴 줄이기 층."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """자리 바꾼 겹말기를 쓴 키우기 층."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    확산 모델을 위한 U-Net 얼개.
    
    그물은 다음을 지닌다.
    - 줄이기를 갖춘 인코더 길
    - 어텐션을 갖춘 병목
    - 키우기와 건너뛰기 이음을 갖춘 디코더 길
    - 때 걸음을 조건으로 삼는 때 임베딩
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
            in_channels: 들임 채널의 개수
            out_channels: 내놓는 채널의 개수
            base_channels: 바탕 채널 수
            channel_multipliers: 해상도 층마다 채널 갑절
            num_res_blocks: 해상도마다 남은 덩이의 수
            attention_resolutions: 어텐션을 쓸 해상도
            dropout: 드롭아웃 확률
            time_emb_dim: 때 임베딩의 차원
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 때 임베딩
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
            
            # 마지막 층만 빼고 줄이기를 더한다
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
                # 인코더에서 오는 건너뛰기 이음
                skip_ch = channels.pop()
                block = ResidualBlock(now_channels + skip_ch, out_ch, time_emb_dim)
                self.decoder_blocks.append(block)
                now_channels = out_ch
            
            # 마지막 층만 빼고 키우기를 더한다
            if i != len(channel_multipliers) - 1:
                self.upsamples.append(Upsample(now_channels))
        
        # 내놓기
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        U-Net을 지나는 앞먹임.
        
        인수:
            x: 잡음 섞인 들임, 꼴 (batch, in_channels, height, width)
            time: 때 걸음, 꼴 (batch,)
        
        반환값:
            예측한 잡음, 꼴 (batch, out_channels, height, width)
        """
        # 때 임베딩
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
        
        # 내놓기
        x = self.conv_out(x)
        
        return x


class SimpleUNet(nn.Module):
    """
    MNIST 같은 작은 자료 묶음에서 더 빨리 익히기 위한 단순한 U-Net.
    배우기에 좋은 출발점이다.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, 
                 base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        
        # 때 임베딩
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
        
        # 내놓기
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
        
        # 건너뛰는 이음을 갖춘 디코더
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
    
    # 온전한 UNet을 시험한다
    model = UNet(in_channels=1, base_channels=32).to(device)
    out = model(x, t)
    print(f"UNet output shape: {out.shape}")
    
    # 매개변수 개수 세기
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")


class MNISTDiffusion:
    """
    MNIST 확산 모델의 익히기와 뽑기를 감싸는 클래스.
    """
    
    def __init__(self, 
                 timesteps: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 2e-4,
                 device: str = None):
        """
        인수:
            timesteps: 확산 때 스텝 수
            batch_size: 익히기 배치 크기
            learning_rate: 최적화기의 러닝레이트
            device: 익힐 장치('cuda' 또는 'cpu')
        """
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 확산 파라메터를 채비한다
        betas = cosine_beta_schedule(timesteps)
        self.diffusion_params = get_diffusion_parameters(betas)
        
        # 파라메터를 기기로 옮긴다
        for key in self.diffusion_params:
            self.diffusion_params[key] = self.diffusion_params[key].to(self.device)
        
        # 모형을 시작한다
        self.model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        
        # 최적화기
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
        """모델 파라메터의 지수 이동 평균을 고친다."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), 
                                        self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_dataloader(self, train: bool = True):
        """
        MNIST 자료 불러오개를 만든다.
        
        인수:
            train: 익히기 배치를 불러올지 시험 배치를 불러올지
        
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
        """한 에폭을 학습한다."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            
            # 아무 때 스텝을 뽑는다
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
            
            # 지수 이동 평균을 고친다
            self.update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def sample_images(self, n_samples: int = 64, use_ema: bool = True):
        """
        표본 그림을 만든다.
        
        인수:
            n_samples: 만들 표본의 개수
            use_ema: 지수 이동 평균 모델을 쓸지 여부
        
        반환값:
            만든 그림 텐서
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
        확산 모델을 익힌다.
        
        인수:
            epochs: 학습 에포크 수
            save_interval: N바퀴마다 표본을 갈무리한다
        """
        dataloader = self.get_dataloader(train=True)
        
        # 표본을 담을 자리를 만든다
        os.makedirs('samples', exist_ok=True)
        
        losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # 학습
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본을 만들어 갈무리한다
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(n_samples=64)
                visualize_samples(
                    samples,
                    nrow=8,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # 검사점 저장
            if epoch % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 학습 로스 그리기
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
        """모델 되짚기 지점을 갈무리한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_params': self.diffusion_params,
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 되짚기 지점을 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


def main():
    """
    MNIST 확산의 으뜸 익히기 대본.
    """
    print("=" * 50)
    print("MNIST Diffusion Model Training")
    print("=" * 50)
    
    # 확산 모델을 첫자리매김한다
    mnist_diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=128,
        learning_rate=2e-4
    )
    
    # 학습
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
```

## 2. 데이터와 노이즈 예측

ToTensor 뒤 Normalize(0.5,0.5)를 적용하므로 원본 픽셀의 [0,1] 범위는 [−1,1]로 바뀐다. 입력 배치는 B×1×28×28이며 숫자 레이블은 학습에 사용하지 않는다. 클래스 조건이 없는 생성 모델이다.

매 배치에서 각 이미지의 시간 인덱스를 0부터 T−1까지 균등하게 뽑는다. get_loss는 독립 표준정규 노이즈 ε를 생성하고 다음 입력으로 모델을 호출한다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

모델 출력과 ε 사이의 MSE는 배치·채널·픽셀 전체의 평균이다. 시간별 웨이트가 모두 1인 단순 노이즈 예측 목적이며, 전체 변분 하한의 모든 항을 그대로 계산하지 않는다. 코드의 인덱스 0은 첫 노이즈 단계이므로 노이즈가 전혀 없는 원본과 같지 않다.

코사인 스케줄과 확산 계수는 CPU에서 만든 다음 사전의 각 텐서를 선택 장치로 옮긴다. 러닝레이트 0.0002의 Adam을 사용하며, 역전파 뒤 모델 파라미터의 전체 그래디언트 노름을 1로 제한한다. 이 제한은 optimizer가 만드는 파라미터 변화량의 노름을 직접 제한하는 연산이 아니다.

## 3. SimpleUNet과 텐서 크기

여기서 가져오는 클래스는 일반 UNet이 아니라 SimpleUNet이다. base_channels=64에서 잔차 블록은 채널을 64→128→256→256→128→64로 바꾸고 공간 크기는 28→14→7→14→28을 따른다. 확대 경로에서 같은 해상도의 피처를 연결한 뒤 채널 수를 줄인다. 최종 출력은 입력과 같은 B×1×28×28이다.

시간은 사인·코사인 임베딩과 MLP로 256차원 조건을 만든 뒤 각 잔차 블록에 더한다. GroupNorm과 시간 조건 잔차 블록을 사용하지만 이 SimpleUNet에는 공간 어텐션이 없다. 출력 활성화로 픽셀 범위를 제한하지 않으며 학습 목표도 깨끗한 이미지 자체가 아니라 노이즈다.

## 4. 에포크 평균과 EMA

훈련 이미지 60,000개와 배치 크기 128에서 한 에포크는 469번의 업데이트다. 마지막 96개 배치를 버리지 않으므로 100에포크를 완료하면 46,900번 업데이트한다. 반환하는 에포크 로스는 각 배치 평균을 같은 비중으로 평균한 값이다. 따라서 마지막 작은 배치까지 샘플 수로 가중한 정확한 전체 평균과는 다르다.

EMA 모델은 시작할 때 학습 모델의 웨이트를 복사하고 첫 optimizer 업데이트부터 다음 식으로 갱신한다.

$$\theta_{\mathrm{EMA}}\leftarrow0.9999\theta_{\mathrm{EMA}}+0.0001\theta$$

별도 대기 기간은 없다. 파라메터를 완만하게 평균해 생성에 활용하는 방식이며, 특정 품질 향상이나 최적 시점을 보장하지는 않는다. 현재 함수는 파라미터만 갱신한다. 다른 구조에 BatchNorm 같은 상태 버퍼를 추가한다면 해당 상태의 처리도 검토해야 한다.

## 5. 생성과 시각화

sample_images는 기본적으로 EMA 모델을 사용한다. 표준정규 B×1×28×28에서 출발해 1,000단계 역과정을 수행하며 유틸리티의 샘플러는 후방분산을 사용한다. 최종 단계에는 추가 노이즈를 넣지 않는다. 생성은 no_grad와 평가 모드에서 수행하지만 출력 텐서가 반드시 [−1,1]에 들어오는 것은 아니다.

기본 실행은 에포크 1과 이후 10에포크마다 64개를 생성하고, 학습이 끝나면 최종 샘플을 별도로 만든다. visualize_samples는 배치 전체 최솟값·최댓값으로 화면 표시 범위를 조정한다. 이는 학습 정규화의 고정 역변환 (x+1)/2와 다르므로 밝기·범위의 차이를 가릴 수 있다. 모든 값이 같으면 분모가 0이 되는 점도 고려한다.

샘플 이미지와 학습 로스는 점검 자료지만 분포의 다양성이나 일반화를 수치로 입증하지 않는다. 이 코드는 검증 로스, FID나 IS를 계산하지 않으며 train=False 로더를 만들 수 있다는 사실만으로 테스트 평가를 수행한 것은 아니다.

## 6. 체크포인트가 복원하는 범위

50에포크마다 저장하고 마지막에도 체크포인트를 남긴다. 저장 항목은 학습 모델, EMA 모델, optimizer 상태와 확산 계수다. 그러나 load_checkpoint는 **저장된 diffusion_params를 복원하지 않는다**. 새 객체를 만들 때 지정한 시간 단계 수와 스케줄이 계속 사용된다.

따라서 웨이트가 로드되어도 학습 당시와 다른 T나 스케줄로 생성할 수 있다. 같은 모델 구조와 확산 설정을 별도로 확인해야 한다. 에포크 번호, 난수 상태, 전체 설정, 로스 이력과 EMA 감쇠율도 복원하지 않으므로 중단 지점과 완전히 같은 학습 궤적을 재개하는 기능으로 설명할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정의 입력·출력 모양과 한 에포크의 업데이트 수를 구하라.

</div>

??? success "풀이"

    입력과 노이즈 예측 출력은 B×1×28×28이다. 60,000=468×128+96이므로 마지막 배치를 포함해 469번 업데이트한다. 100에포크면 46,900번이다. 로스 반환값은 469개 배치 평균의 산술평균이어서 샘플 수에 따른 가중 평균과는 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MSE 로스의 평균 축과 EMA 갱신 시점을 설명하라.

</div>

??? success "풀이"

    로스는 모든 배치·채널·공간 원소의 제곱 오차를 평균한다. 전체 원소 수가 M이면 그래디언트는 (2/M)Σ(εθ−ε)∇θεθ다. EMA는 초기 웨이트를 복사하고 첫 optimizer 업데이트부터 0.9999의 감쇠율로 파라미터를 평균한다. 단순 MSE가 전체 ELBO와 항별로 같은 목적이라는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
다른 노이즈 스케줄로 객체를 만든 뒤 기존 체크포인트를 불러오면 어떤 문제가 생기는가?

</div>

??? success "풀이"

    load_checkpoint는 저장된 확산 계수를 읽어 객체에 대입하지 않으므로 새 생성자의 스케줄이 남는다. 웨이트의 학습 조건과 샘플링 조건이 달라진다. 시간 단계 수와 스케줄, 모델 구조를 함께 일치시켜야 하며, 스케줄의 품질 비교에는 같은 데이터·예산·평가 조건과 별도의 실험이 필요하다.



## 정리하며

MNIST 파이프라인은 노이즈 MSE 학습, 그래디언트 clipping, 첫 단계부터 적용하는 EMA, 후방분산을 이용한 샘플링을 연결한다. 배치 평균의 의미와 표시용 정규화를 구분하고, 체크포인트 로더가 확산 계수를 복원하지 않는 제한을 확인한다.
