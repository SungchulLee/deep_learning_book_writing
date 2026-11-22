# 확산 모델 유틸리티

확산 예제들이 같은 정방향 식과 역전이 계수를 사용하려면 스케줄·누적곱·입력 축의 관례를 일치시켜야 한다. 이 페이지는 β 스케줄, 가우시안 관측, 노이즈 MSE와 샘플링을 연결하는 보조 함수들을 살펴본다. 함수 정의와 실제 모델 트레인·품질 평가의 범위도 구분한다.

## 1. 코드와 출력

??? note "코드 (600줄)"

    ```python
    """확산 유틸리티."""
    # ============================================================================
    # diffusion_utils.py - 확산 모델의 수학적 기초
    # ============================================================================

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

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
    ```

```
======================================================================
NOISE SCHEDULE COMPARISON
======================================================================

Generating schedules for T=1000...

Linear: [0.000100, 0.020000]
Cosine: [0.000100, 0.999900]

✓ Saved: noise_schedule_comparison.png

KEY DIFFERENCES:
  Linear: Uniform increase, aggressive at end
  Cosine: Smooth increase, better for images
  Recommendation: Use cosine for better quality
======================================================================
```

## 2. 스케줄과 사전 계산

linear_beta_schedule은 β를 선형 보간한다. cosine_beta_schedule은 정규화한 코사인 누적곱의 이웃 비에서 β를 얻고 **[0.0001,0.9999]**로 클리핑한다. 다른 페이지의 [10⁻⁸,0.999]와 끝값이 다르므로 같은 이름만으로 완전히 같은 노이즈 과정을 가정하지 않는다.

get_diffusion_parameters는 α=1−β, 누적곱과 이전 누적곱, 제곱근·역제곱근과 사후분산을 계산한다. β 배열 길이를 T라 하면 기본 저장 비용은 O(T)이며 반복에서 필요한 배치 계수만 조회한다. 다만 데이터에 노이즈를 더하는 연산 자체가 데이터 크기와 무관한 O(1)이 되는 것은 아니다.

이 함수는 이전 누적곱을 만들 때 CPU의 torch.ones(1)을 사용한다. 따라서 CUDA β를 직접 넘기는 일반 장치 독립 함수는 아니며, 현재 예제처럼 **CPU에서 사전을 만든 뒤 각 값을 같은 장치로 이동**하는 사용과 구분한다. β 범위·유한성과 0 나눗셈에 대한 입력 검증은 없다.

## 3. 인덱스와 브로드캐스팅

extract는 길이 B의 정수 시간 t로 길이 T의 배열을 조회해 B×1×…×1로 바꾼다. 이미지 B×C×H×W에는 B×1×1×1, 벡터 B×d에는 B×1이 된다. 입력·배열·인덱스의 장치와 유효 범위가 맞아야 한다.

코드는 0~T−1 인덱스다. 첫 누적곱은 1이 아니라 1−β[0]이므로 **t=0도 노이즈 관측**이다. 문자열·주석의 이론상 ᾱ₀=1은 배열 인덱스와 분리해 읽는다. 깨끗한 x₀는 함수의 원본 인수이며 배열의 첫 노이즈 단계와 같지 않다.

## 4. 정방향 관측과 로스

forward_diffusion은 주어진 노이즈를 사용해 다음을 계산한다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

계수 제곱의 합이 1이라는 사실은 **원본 좌표의 분산이 1일 때** 관측 분산을 유지한다는 뜻이다. 일반적으로 Cov(xₜ)=ᾱt Cov(x₀)+(1−ᾱt)I다. 모든 데이터의 분산을 그대로 보존한다는 주석 설명은 일반적이지 않다.

get_loss는 외부에서 받은 t와 선택적 ε로 관측을 만들고 실제 ε와 모델 출력의 모든 원소 MSE를 계산한다. **시간을 직접 뽑거나 optimizer를 갱신하지 않는다**. 호출자가 시간 선택·트레인 루프를 정해야 한다. 이 단순 MSE는 전체 ELBO나 이미지별 픽셀 합 NLL을 계산하는 함수가 아니다.

표준 정규 노이즈를 같은 목표 분포로 사용해도 각 시간의 조건부 예측 난이도와 회귀 잔여 분산은 다를 수 있다. 모델이 모든 개별 ε를 정확히 알아낼 수 있다는 보장도 없다. 시간별 노이즈 예측의 조건부 평균과 주변 스코어의 관계를 사용한다.

## 5. 역전이와 최종 단계

p_sample은 노이즈 예측으로 다음 평균을 계산한다.

$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

t>0에서는 사후분산 β̃t의 잡음을 더하고 t=0이면 평균을 반환한다. 정수 t와 배치 t_tensor는 **같은 시간**을 나타내야 한다. sample은 이 조건을 만족하는 배치를 만들어 N(0,I)에서 T단계로 생성한다. p_sample의 device 인수는 함수 내부에서 상태·모델·스케줄을 이동시키지 않는다.

sample은 eval·no_grad를 사용하지만 원래 트레인 모드를 복원하지 않는다. 트레인 중 호출한 뒤에는 모델의 모드를 다시 설정해야 한다. 생성값의 범위를 자동으로 클리핑하거나 이미지 파일로 저장하지도 않는다. 종단 사전분포와 데이터 스케일의 적합성은 스케줄·모델과 함께 검토한다.

## 6. 시각화의 조건과 한계

visualize_diffusion_process는 시점마다 새 노이즈를 만들어 첫 이미지 하나만 표시한다. 따라서 한 마르코프 궤적의 연속 장면이 아니라 시점별 조건부 주변 샘플 비교다. 시간 인덱스를 CPU에 만들므로 GPU에 있는 상수 사전에 바로 대응하는 함수도 아니다. num_images=1에서는 subplot이 단일 axes 객체가 되어 현재 인덱싱과 맞지 않는다.

흑백 패널은 vmin·vmax를 고정하지 않아 각 배열에 따라 눈금이 달라질 수 있다. RGB 패널은 이미지마다 최솟값·최댓값으로 정규화하며 상수 이미지에서는 분모가 0이다. visualize_samples도 배치 전체의 최솟값·최댓값으로 바꾸므로 원래 밝기·대비와 다른 배치의 절대값 비교를 숨길 수 있다. 공통 데이터 범위·변환을 정해 비교해야 한다.

시각화 함수는 .numpy를 직접 사용하므로 필요한 입력이 그래디언트 그래프에서 분리되어 있어야 한다. CPU 이동을 했다는 사실만으로 그래디언트 추적이 꺼지는 것은 아니다. 그림이나 파일 저장 메시지 자체는 생성 품질의 정량 검증이 아니다.

## 7. 타임스텝 임베딩과 실제 데모 범위

SinusoidalPositionEmbedding은 길이 B의 시간을 B×dim 사인·코사인 표현으로 바꾸려는 함수다. 실제 성분 수는 2×floor(dim/2)이므로 **4 이상인 짝수 폭**을 전제로 한다. 홀수 폭을 위한 패딩은 구현하지 않았고 half_dim−1이 0인 작은 폭도 유효하지 않다. 시간 입력의 배치·dtype을 모델과 맞춰야 한다.

main에서는 선형·코사인 β를 만들고 곡선과 양 끝값을 출력한다. 뉴럴 네트워크 트레인, 샘플 생성과 FID 비교는 수행하지 않는다. 따라서 출력의 '코사인이 이미지 품질에 더 좋다'는 일반 권고를 이 데모의 실측 결론으로 설명하지 않는다. 스케줄·트레인량·솔버와 평가 조건을 맞춘 별도 비교가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
extract가 B×3×32×32 입력에 사용할 계수를 어떤 모양으로 반환하는가?

</div>

??? success "풀이"

    시간 t가 길이 B이면 계수 B개를 조회해 B×1×1×1로 만든다. 이를 이미지·노이즈와 브로드캐스팅한다. 코드의 t=0은 첫 β가 적용된 노이즈 단계이며 원본 시점과 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
get_loss와 sample이 각각 무엇을 하고 어떤 트레인 제어는 하지 않는가?

</div>

??? success "풀이"

    get_loss는 주어진 시간에서 노이즈 MSE를 계산하지만 시간 추출·optimizer 업데이트를 하지 않는다. sample은 eval·no_grad로 역전이를 반복하지만 이전 모드나 장치를 자동 복원·동기화하지 않는다. 호출자가 공통 스케줄과 장치·트레인 모드를 관리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
시각화에서 배치·이미지별 min–max 정규화와 새 노이즈가 어떤 해석 차이를 만드는가?

</div>

??? success "풀이"

    각 범위를 따로 0~1에 맞추면 절대 밝기·대비 차이가 가려질 수 있다. 시점마다 새 노이즈를 쓰면 같은 경로의 연속 궤적이 아니라 주변 분포 비교다. 스케줄 곡선을 그렸다는 것만으로 생성 품질을 비교한 실험도 아니다.



## 정리하며

유틸리티는 스케줄과 계수를 재사용하지만 인덱스·배치 축·장치·모드는 호출자가 일치시켜야 한다. 노이즈 MSE와 트레인 루프, 사후분산 샘플러와 표시용 정규화를 구분한다. 현재 main은 β 곡선 시연이며 생성 품질의 비교 결과가 아니다.
