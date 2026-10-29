# 확산 모델을 위한 U-Net 구조

확산의 노이즈 예측기는 관측 이미지와 시간 조건을 받아 같은 공간 크기의 텐서를 출력한다. U-Net은 축소 경로에서 넓은 범위의 피처를 만들고 확대 경로에서 같은 해상도의 피처를 다시 결합하는 구조다. 이 페이지의 SimpleUNet과 일반 UNet은 서로 다른 구현이며, 후자의 순방향 순서에는 기본 설정에서도 차원이 맞지 않는 문제가 있다.

## 1. 코드

```python
"""
확산 모델을 위한 U-Net 구조

이 단원은 확산 모델에 흔히 쓰는 U-Net 구조를 짠다.
U-Net은 잡음 섞인 그림과 때 스텝 임베딩을 들임으로 받아 잡음을 헤아린다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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
        
        # 첫 컨볼루션
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # 때 임베딩을 더한다
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]  # 공간 차원으로 퍼뜨린다
        
        # 두 번째 컨볼루션
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
    """겹말기를 쓴 줄이기 레이어."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """자리 바꾼 겹말기를 쓴 키우기 레이어."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    확산 모델을 위한 U-Net 구조.
    
    넷트웍은 다음을 지닌다.
    - 줄이기를 갖춘 인코더 길
    - 어텐션을 갖춘 병목
    - 키우기와 건너뛰기 이음을 갖춘 디코더 길
    - 때 스텝을 조건으로 삼는 때 임베딩
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
            channel_multipliers: 해상도 레이어마다 채널 갑절
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
        
        # 첫 컨볼루션
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
            
            # 마지막 레이어만 빼고 줄이기를 더한다
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
            
            # 마지막 레이어만 빼고 키우기를 더한다
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
        U-Net을 지나는 피드포워드.
        
        인수:
            x: 잡음 섞인 들임, 꼴 (batch, in_channels, height, width)
            time: 때 스텝, 꼴 (batch,)
        
        반환값:
            예측한 잡음, 꼴 (batch, out_channels, height, width)
        """
        # 때 임베딩
        time_emb = self.time_embedding(time)
        
        # 첫 컨볼루션
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
    
    # 파라메터 개수 세기
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {params:,}")
```

## 2. 시간 조건 잔차 블록

ResidualBlock은 GroupNorm→SiLU→3×3 컨볼루션 뒤에 시간 임베딩의 선형 투영을 더한다. 다시 정규화·활성화·컨볼루션을 적용하고 입력의 Identity 또는 1×1 투영을 더한다. 공간 크기는 유지하며 시간 조건은 출력 채널 수만큼 만든 뒤 높이·너비에 브로드캐스팅한다.

이 시간 주입은 다른 페이지의 scale·shift 방식과 달리 **덧셈 방식**이다. GroupNorm의 입력·출력 채널은 그룹 수 8로 나누어져야 한다. 기본 채널 폭은 조건을 만족하지만 임의의 폭을 지원하는 것은 아니다.

SinusoidalPositionEmbedding은 diffusion_utils에서 가져온다. 실제 반환 폭이 2×floor(dim/2)이고 작은 폭에는 분모 문제가 있으므로 기본 base_channels=32처럼 4 이상인 짝수와 계층별 나눗셈 조건을 함께 맞춘다.

## 3. 공간 어텐션

AttentionBlock은 B×C×H×W를 헤드별 위치 시퀀스로 바꾸어 q·k·v를 계산한다. 위치 수 N=HW, 헤드 폭 Ch=C/heads이면 q·k·v는 B×heads×N×Ch다. 웨이트 행렬은 B×heads×N×N이고 Ch의 제곱근으로 스케일한 뒤 softmax를 적용한다.

출력은 원래 이미지 모양으로 복원하고 1×1 투영 뒤 입력에 더한다. C는 헤드 수와 8로 나누어져야 한다. 어텐션은 위치 사이 상호작용을 제공하지만 메모리는 N²에 민감하므로 큰 해상도·배치에서 비용을 확인해야 한다.

## 4. SimpleUNet의 유효한 28×28 경로

main이 먼저 만드는 SimpleUNet은 공간 연산을 잔차 블록 사이에 배치한다. base_channels=32에서 다음과 같다.

| 피처 | 모양 |
|---|---|
| conv1 | B×32×28×28 |
| res1의 x2 | B×64×28×28 |
| down1 후 res2의 x3 | B×128×14×14 |
| down2 후 res3의 x4 | B×128×7×7 |
| up1 후 x3 연결 | B×256×14×14 |
| res4 후 up2 | B×64×28×28 |
| x2 연결 후 res5 | B×32×28×28 |
| 최종 출력 | B×1×28×28 |

up1·up2는 전치 컨볼루션으로 공간을 두 배 키운다. SimpleUNet에는 AttentionBlock이 없으며 생성기 로스나 샘플링 함수도 없다. 다른 이미지 크기에서는 두 번의 올림 축소와 두 배 확대가 skip 크기와 맞는지 확인해야 한다.

## 5. 일반 UNet의 실행 순서 불일치

일반 UNet의 생성자는 수준별 잔차 블록과 downsample을 따로 저장한다. 하지만 forward는 **모든 인코더 잔차 블록을 먼저 실행한 뒤 모든 downsample을 실행**한다. 다중 해상도 경로로 교차 실행하지 않는다.

main의 base_channels=32와 배수 (1,2,4,8)에서는 잔차 블록들을 모두 지난 h가 B×256×28×28이다. 첫 downsample은 32채널용으로 만들어졌으므로 **256채널 입력을 받을 수 없어 실패한다**. 따라서 main의 두 번째 UNet 출력 모양과 파라미터 출력까지 정상 도달한다고 설명할 수 없다.

디코더도 모든 잔차 블록을 먼저 실행한 뒤 모든 upsample을 수행한다. 생성자의 skip 채널 목록은 축소가 중간에 들어가는 순서를 전제로 하지만 실제 저장·소비 순서는 다르다. 인코더 한 줄만 조정해 전체 문제가 해결된다고 볼 수 없으며 수준별 공간 연산과 skip의 대응을 함께 설계해야 한다.

입력 인수 attention_resolutions와 dropout은 현재 UNet 내부에서 사용되지 않는다. 어텐션은 bottleneck 하나에 고정되어 있고 Dropout 계층은 없다. 인수를 바꿔 해상도별 어텐션이나 Dropout 효과를 비교할 수 있는 코드라고 설명하지 않는다. down_idx와 up_idx 변수도 정의만 되어 실행 순서를 제어하지 않는다.

## 6. 구조 확인과 학습 검증의 차이

데모는 임의의 4×1×28×28 입력과 시간으로 순방향을 호출할 뿐 데이터셋·노이즈 로스·optimizer·생성 평가를 갖추지 않는다. SimpleUNet의 출력 모양이 맞아도 올바른 노이즈 예측을 학습했다는 뜻은 아니다. 일반 UNet은 위에서 설명한 채널 문제를 먼저 갖고 있다.

이 코드 블록은 보존하며 실행·모델 학습이나 그림 재생성은 수행하지 않는다. 학습 파이프라인으로 연결할 때는 사용하려는 클래스, 입력 크기, 예측 대상과 시간·장치·정규화 조건을 명시해야 한다. 구조의 이름이나 파라미터 수로 생성 품질을 평가할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SimpleUNet에서 첫 번째 skip 연결과 두 번째 skip 연결의 모양을 설명하라.

</div>

??? success "풀이"

    첫 연결은 up1의 128채널 14×14와 x3의 128채널 14×14를 붙여 256채널을 만든다. 두 번째는 up2의 64채널 28×28과 x2의 64채널 28×28을 붙여 128채널을 만든다. 이후 res5가 이를 32채널로 바꾼다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
일반 UNet의 첫 downsample이 main 설정에서 실패하는 이유는 무엇인가?

</div>

??? success "풀이"

    생성자는 첫 수준의 32채널 downsample을 만들지만 forward는 모든 잔차 블록부터 실행해 256채널 피처를 전달한다. 공간 축소를 수준 사이에 배치하도록 설계된 채널 목록과 실행 순서가 다르다. 디코더의 모든 확대를 뒤로 미룬 순서도 함께 검토해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
attention_resolutions와 dropout을 바꾸면 현재 UNet의 구조가 어떻게 바뀌는가?

</div>

??? success "풀이"

    둘 다 사용되지 않아 그 인수만 바꾸어서는 구조가 달라지지 않는다. 어텐션은 bottleneck에 고정되고 Dropout 계층은 없다. 해당 옵션의 효과나 노이즈 스케줄·학습 품질을 비교하는 실험도 이 페이지에는 구현되어 있지 않다.



## 정리하며

U-Net에서는 해상도 변화, 채널 수와 skip 저장·소비의 순서가 함께 맞아야 한다. SimpleUNet은 28×28의 두 단계 경로를 제공하지만 일반 UNet은 모든 축소·확대를 뒤로 미뤄 차원이 맞지 않는다. 사용되지 않는 옵션과 실제 구현을 구분해 읽는다.
