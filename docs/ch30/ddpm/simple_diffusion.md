# 2차원 데이터로 이해하는 확산 모델

2차원 데이터에서는 정방향 노이즈 관측과 생성 샘플을 산점도로 직접 비교할 수 있다. 이 페이지는 나선 데이터에 시간 조건 MLP를 학습하는 작은 DDPM 구성을 살펴본다. 시각화는 학습 동작을 이해하는 도구이며, 분포 일치나 일반화를 증명하는 평가는 아니다.

## 1. 코드와 기존 출력

??? note "코드 (689줄)"

    ```python
    """단순한 확산."""
    # ============================================================================
    # simple_diffusion.py - 2차원 확산 모델(가르치기 위한 짜기)
    # ============================================================================

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from diffusion_utils import (
        cosine_beta_schedule,
        get_diffusion_parameters,
        forward_diffusion,
        get_loss
    )
    from tqdm import tqdm

    # ============================================================================
    # 핵심 개념: 확산 모델
    # ============================================================================

    """
    확산 모델이란 무엇인가?
    ===========================

    확산 모형은 조금씩 잡음을 더하는 과정을 거꾸로 되돌려 자료를 만드는 법을 배운다.

    1. 앞 과정(쉽다. 배울 것이 없다):
       실제 자료 → 차츰 잡음 더하기 → 순수 아무 잡음

    2. 뒤 과정(어렵다. 익히기가 든다):
       아무 잡음 → 차츰 잡음 없애기 → 그럴듯한 자료

    핵심 통찰: 어떤 잡음 수준에서든 잡음 없애는 법을 배우면 순수 잡음에서 시작해
    잡음을 없애 가며 그럴듯한 자료로 돌아갈 수 있다!

    왜 2차원 장난감 자료인가?
    ================
    높은 차원의 그림(256×256×3 = 196,608 차원)을 다루기 앞에 다음을 쓴다
    또렷한 그림과 직관 쌓기를 위해 2차원 자료(x, y뿐)를 쓴다.
    """

    # ============================================================================
    # 수학의 바탕: 닫힌 꼴 앞 확산
    # ============================================================================

    """
    앞 확산 식(온전한 이끌어 내기)
    =================================================

    회귀 꼴(한 번에 한 걸음씩):
        x_t = √α_t · x_{t-1} + √β_t · ε_t,  여기서 ε_t ~ N(0, I)

        α_t = 1 - β_t (남는 신호)
        β_t = 걸음 t에서의 잡음 흩어짐

    마법: 닫힌 꼴(어느 걸음으로든 곧바로 건너뛴다!)
    ====================================================

    되부름을 t 걸음 하고 나면 잡음 마디가 모두 아울러 다음이 된다.

        ╔══════════════════════════════════════════════════════════╗
        ║  x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  여기서 ε ~ N(0, I)   ║
        ╚══════════════════════════════════════════════════════════╝

    여기서 각 기호는 다음과 같다.
        ᾱ_t = ∏_{i=1}^t α_i  (모든 알파의 쌓은 곱)

    느낌으로 보면:
    ----------
        √ᾱ_t: 본디 신호가 얼마나 남았는가(t이 커지면 줄어든다)
        √(1-ᾱ_t): 잡음이 얼마나 있는가(t이 커지면 늘어난다)

        t=0에서:   ᾱ_0 = 1    → x_0 = 1·x_0 + 0·ε = x_0 (잡음 없음)
        t=100에서: ᾱ_100 ≈ 0.37 → x_100 = 0.6·x_0 + 0.8·ε (거의 잡음)
        t=∞에서:   ᾱ_∞ → 0    → x_∞ ≈ 0·x_0 + 1·ε = ε (순수한 잡음)

    이것이 놀라운 까닭:
    --------------------
    1. 효율: O(t) 대신 O(1) - 아무 때 걸음으로 곧바로 뛴다
    2. 나란한 익히기: 배치마다 서로 매이지 않게 다른 때 걸음을 뽑는다
    3. 수학의 아름다움: 복잡한 되풀이가 단순한 무게 합으로 줄어든다

    빠른 살펴보기:
    ----------------
    기호  | 뜻                              | 흔한 범위
    --------|--------------------------------------|---------------
    β_t     | 걸음 t에서 더한 잡음                | 0.0001 ~ 0.02
    α_t     | 남는 신호 (= 1 - β_t)          | 0.98 ~ 0.9999
    ᾱ_t     | 쌓은 신호 (= ∏ α_i)          | 1.0 → 0 (잦아듦)
    √ᾱ_t    | 앞 과정에서 자료의 계수  | 1.0 → 0
    √(1-ᾱ_t)| 앞 과정에서 잡음의 계수 | 0 → 1.0
    """

    # ============================================================================
    # 수학으로 이끌어 내기: 뒤 확산 공식
    # ============================================================================

    """
    밝힘: 뒤 확산 평균 공식
    ======================================

    우리가 밝히려는 것:

        μ_θ(x_t, t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))

    이 식은 잡음을 걷어 낼 때의 평균을 어떻게 셈하는지 알려 준다: x_t → x_{t-1}

    ============================================================================
    1부: 참 사후 분포(x_0을 안다면)
    ============================================================================

    베이즈 규칙에 따라 참 뒤 분포는 다음과 같다.

        q(x_{t-1} | x_t, x_0) = q(x_t | x_{t-1}, x_0) · q(x_{t-1} | x_0) / q(x_t | x_0)

    마르코프 성질에 따라 q(x_t | x_{t-1}, x_0) = q(x_t | x_{t-1})

    So:
        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

    이는 정규 분포의 곱이므로 → 역시 정규 분포이다!

    ============================================================================
    걸음 1: 앞 확산 식을 떠올린다
    ============================================================================

    앞 확산에서 다음을 얻는다.

    (가) 한 걸음 앞으로:
        q(x_t | x_{t-1}) = N(x_t; √α_t · x_{t-1}, β_t · I)

        밀도:
            p(x_t | x_{t-1}) ∝ exp(-1/(2β_t) ||x_t - √α_t · x_{t-1}||²)

    (나) x_0에서 x_t까지 여러 걸음:
        q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)

        밀도:
            p(x_t | x_0) ∝ exp(-1/(2(1-ᾱ_t)) ||x_t - √ᾱ_t · x_0||²)

    (다) x_0에서 x_{t-1}까지 여러 걸음:
        q(x_{t-1} | x_0) = N(x_{t-1}; √ᾱ_{t-1} · x_0, (1-ᾱ_{t-1}) · I)

        밀도:
            p(x_{t-1} | x_0) ∝ exp(-1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1} · x_0||²)

    ============================================================================
    걸음 2: 사후 분포 q(x_{t-1} | x_t, x_0)을 셈한다
    ============================================================================

    밀도를 아우른다.

        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

        ∝ exp(-1/(2β_t) ||x_t - √α_t·x_{t-1}||²) 
          · exp(-1/(2(1-ᾱ_{t-1})) ||x_{t-1} - √ᾱ_{t-1}·x_0||²)

    이차식을 펼친다.

        ||x_t - √α_t·x_{t-1}||² = ||x_t||² - 2√α_t·⟨x_t, x_{t-1}⟩ + α_t||x_{t-1}||²

        ||x_{t-1} - √ᾱ_{t-1}·x_0||² = ||x_{t-1}||² - 2√ᾱ_{t-1}·⟨x_{t-1}, x_0⟩ + ᾱ_{t-1}||x_0||²

    지수는 x_{t-1}에 대한 이차식이다.

        E(x_{t-1}) = A||x_{t-1}||² - 2⟨B, x_{t-1}⟩ + C

    여기서 각 기호는 다음과 같다.
        A = α_t/(2β_t) + 1/(2(1-ᾱ_{t-1}))
        B = √α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})
        C = (x_{t-1}이 들어가지 않는 상수 마디)

    완전제곱으로 만든다.
        E(x_{t-1}) = A||x_{t-1} - B/A||² + (constant)

    그러므로 q(x_{t-1} | x_t, x_0) = N(x_{t-1}; B/A, 1/(2A)·I)

    평균은 μ̃_t(x_t, x_0) = B/A이다

    ============================================================================
    걸음 3: 평균을 얻으려 B/A을 단순하게 한다
    ============================================================================

    A을 셈한다:
        A = α_t/(2β_t) + 1/(2(1-ᾱ_{t-1}))

        = [α_t(1-ᾱ_{t-1}) + β_t] / [2β_t(1-ᾱ_{t-1})]

    분자를 간추린다.
        α_t(1-ᾱ_{t-1}) + β_t = (1-β_t)(1-ᾱ_{t-1}) + β_t
                              = 1 - ᾱ_{t-1} - β_t + β_t·ᾱ_{t-1} + β_t
                              = 1 - ᾱ_{t-1} + β_t·ᾱ_{t-1}

    눈여겨볼 것: ᾱ_t = ᾱ_{t-1}·α_t = ᾱ_{t-1}·(1-β_t)

    그러므로 1 - ᾱ_{t-1} + β_t·ᾱ_{t-1} = 1 - ᾱ_{t-1}(1-β_t)
                                    = 1 - ᾱ_{t-1}·α_t
                                    = 1 - ᾱ_t

    따라서 다음이 성립한다.
        A = (1-ᾱ_t) / [2β_t(1-ᾱ_{t-1})]

    B을 셈한다:
        B = √α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})

    μ̃_t = B/A을 셈한다:
        μ̃_t(x_t, x_0) = [√α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})] / [(1-ᾱ_t)/(2β_t(1-ᾱ_{t-1}))]

        = [√α_t·x_t/β_t + √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})] · [2β_t(1-ᾱ_{t-1})/(1-ᾱ_t)]

    펼쳐 나눈다.
        = 2β_t(1-ᾱ_{t-1})/(1-ᾱ_t) · √α_t·x_t/β_t 
          + 2β_t(1-ᾱ_{t-1})/(1-ᾱ_t) · √ᾱ_{t-1}·x_0/(1-ᾱ_{t-1})

        = 2(1-ᾱ_{t-1})/(1-ᾱ_t) · √α_t·x_t 
          + 2β_t/(1-ᾱ_t) · √ᾱ_{t-1}·x_0

    정리하면 다음과 같다.
        μ̃_t(x_t, x_0) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    이것이 (x_0을 안다면) 참 사후 평균이다.

    ============================================================================
    걸음 4: x_t에서 x_0을 어림한다(핵심 솜씨!)
    ============================================================================

    문제: 만들 때는 x_0을 모른다!

    풀이: 앞 확산으로 x_0을 어림한다.

    다음에서: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε

    x_0에 대해 푼다.
        √ᾱ_t·x_0 = x_t - √(1-ᾱ_t)·ε

        x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t    ... (★)

    모형의 예측 ε_θ(x_t, t)을 쓴다.

        x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t    ... (★★)

    ============================================================================
    걸음 5: 사후 평균에 x̂_0을 넣는다
    ============================================================================

    다음에서 x_0을 x̂_0으로 갈음한다.
        μ̃_t(x_t, x_0) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    그러면 다음이 된다.
        μ_θ(x_t,t) = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x̂_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    x̂_0을 넣는다.
        μ_θ = √ᾱ_{t-1}·β_t/(1-ᾱ_t) · [(x_t - √(1-ᾱ_t)·ε_θ)/√ᾱ_t] 
              + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    첫 마디를 펼친다.
        = [√ᾱ_{t-1}/(√ᾱ_t)] · [β_t/(1-ᾱ_t)] · [x_t - √(1-ᾱ_t)·ε_θ]
        + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

    눈여겨볼 것: ᾱ_t = ᾱ_{t-1}·α_t이므로 √ᾱ_{t-1}/√ᾱ_t = 1/√α_t

        = [1/√α_t] · [β_t/(1-ᾱ_t)] · [x_t - √(1-ᾱ_t)·ε_θ]
        + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t

        = [β_t/((1-ᾱ_t)√α_t)]·x_t - [β_t/(√α_t·√(1-ᾱ_t))]·ε_θ
        + [√α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)]·x_t

    x_t 마디를 아우른다.
        x_t의 계수 = β_t/((1-ᾱ_t)√α_t) + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)
                        = [1/((1-ᾱ_t)√α_t)] · [β_t + α_t(1-ᾱ_{t-1})]

    핵심 항등식: 1 - ᾱ_t = 1 - α_t·ᾱ_{t-1} = (1-α_t) + α_t(1-ᾱ_{t-1})
                           = β_t + α_t(1-ᾱ_{t-1})

    그러므로 x_t의 계수 = [1/((1-ᾱ_t)√α_t)] · (1-ᾱ_t) = 1/√α_t

    마지막 결과:
        μ_θ(x_t,t) = x_t/√α_t - [β_t/(√α_t·√(1-ᾱ_t))]·ε_θ(x_t,t)

    1/√α_t을 밖으로 묶어 낸다.

        ╔═══════════════════════════════════════════════════════════╗
        ║  μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))    ║
        ╚═══════════════════════════════════════════════════════════╝

    ✓✓✓ 밝혔다 ✓✓✓

    이것이 뒤 확산에 쓰는 공식이다!

    ============================================================================
    직관의 풀이
    ============================================================================

    μ_θ(x_t,t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t))
                 └──┬──┘   └─────────┬─────────┘
                  잣수    헤아린 잡음 없애기

    1. 모형이 잡음을 예측한다: ε_θ(x_t,t)
    2. 잡음의 잣수를 맞춘다: β_t/√(1-ᾱ_t)·ε_θ
    3. x_t에서 뺀다: x_t - (잣대 잡은 잡음)
    4. 결과의 잣수를 다시 맞춘다: 1/√α_t을 곱한다

    결과: x_{t-1}에 대한 우리의 가장 좋은 어림!

    ============================================================================
    """

    # ============================================================================
    # 모델: 잡음 없애는 뉴럴 넷트웍
    # ============================================================================

    class Simple2DModel(nn.Module):
        """
        어떤 때 걸음에서든 자료의 잡음을 헤아리는 핵심 잡음 없애기 뉴럴 넷트웍.

        구조:
        -------------
        들임: [잡음 섞인 자료(2차원), 때 걸음 묻힘(64차원)]
        내놓음: 예측한 잡음(2차원)

        모델이 배우는 것: ε_θ(x_t, t) ≈ ε

        왜 (깨끗한 자료가 아니라) 잡음을 헤아리는가?
        -----------------------------------
        1. 잡음이 자료 클래스에 걸쳐 더 고르다
        2. 모든 때 걸음에서 목표가 한결같다
        3. 겪어 보니 표본 품질이 낫다

        때를 조건으로 삼기:
        ---------------------
        때 걸음마다 다른 꾀가 든다.
        - 앞머리(t이 작음): 자료에 잡음이 거의 없다 → 작은 고침
        - 뒷머리(t이 큼): 자료에 잡음이 많다 → 과감한 잡음 없애기

        뉴럴 넷트웍이 때에 따른 잡음 없애기 결을 배울 수 있도록
        때 걸음을 64차원 벡터로 박아 넣는다.
        """

        def __init__(self, hidden_dim: int = 128):
            super().__init__()

            # 때 임베딩: 낱값 때 걸음 → 64차원 나타냄
            self.time_embed = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),  # SiLU(x) = x * 시그모이드(x)
                nn.Linear(64, 64),
            )

            # 으뜸 잡음 없애기 뉴럴 넷트웍: [2차원 자료 + 64차원 때] → 2차원 잡음
            self.network = nn.Sequential(
                nn.Linear(2 + 64, hidden_dim),  # 66 → 128
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),  # 128 → 2(잡음 벡터)
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """
            주어진 때 걸음에서 자료의 잡음을 헤아린다.

            인수:
                x: 잡음 섞인 2차원 점, 꼴 (batch_size, 2)
                t: 때 걸음 번호, 꼴 (batch_size,)

            반환값:
                예측한 잡음: 꼴 (batch_size, 2)
            """
            # 더 잘 배우도록 때 걸음을 [0, 1] 범위로 고르게 맞춘다
            t_normalized = t.float().unsqueeze(-1) / 1000.0

            # 때 걸음을 풍부한 나타냄으로 박아 넣는다
            t_emb = self.time_embed(t_normalized)  # (배치, 1) → (배치, 64)

            # 자료와 때 임베딩을 이어 붙인다
            h = torch.cat([x, t_emb], dim=-1)  # (배치, 66)

            # 잡음을 헤아린다
            return self.network(h)  # (배치, 2)

    # ============================================================================
    # 장난감 자료 묶음
    # ============================================================================

    def generate_swiss_roll(n_samples: int = 1000) -> torch.Tensor:
        """
        2차원 스위스 롤 나선 자료 묶음을 만든다.

        시험하기에 딱 맞는 나선 꼴의 점 구름:
        - 그림으로 보기에 넉넉히 단순하다(2차원)
        - 흥미로울 만큼 넉넉히 복잡하다(비선형)
        - 모델이 굽은 다양체를 배울 수 있는지 시험한다

        수식: x = θ·cos(θ), y = θ·sin(θ), θ ∈ [0, 4π]
        """
        t = torch.linspace(0, 4 * np.pi, n_samples)
        x = t * torch.cos(t)
        y = t * torch.sin(t)
        data = torch.stack([x, y], dim=1)
        return data + torch.randn_like(data) * 0.1  # 작은 잡음을 더한다

    def generate_moons(n_samples: int = 1000) -> torch.Tensor:
        """
        서로 끼워진 초승달 모양 둘을 만든다.
        시험하는 것: 여러 봉우리, 굽은 모양, 가깝지만 떨어진 짜임.
        """
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        return torch.tensor(data, dtype=torch.float32)

    # ============================================================================
    # 그려 보기
    # ============================================================================

    def visualize_2d_diffusion(data, timesteps, diffusion_params, num_steps=10):
        """
        자료가 어떻게 차츰 잡음이 되는지 그려 본다(앞 확산).

        장면 10개를 보인다: 깨끗한 자료 → 일부 잡음 → 순수 잡음
        직관을 쌓는 데 도움이 된다. 곧 이것이 우리가 거꾸로 돌리는 법을 배워야 할 것이다!
        """
        sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
        sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']

        time_steps = np.linspace(0, timesteps - 1, num_steps, dtype=int)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for idx, t in enumerate(time_steps):
            # 닫힌 꼴 앞 확산을 쓴다: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
            noise = torch.randn_like(data)
            t_tensor = torch.full((data.shape[0],), t)
            x_t = forward_diffusion(data, t_tensor, noise, 
                                   sqrt_alphas_cumprod, 
                                   sqrt_one_minus_alphas_cumprod)

            axes[idx].scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), 
                             alpha=0.5, s=1)
            axes[idx].set_title(f't = {t}')
            axes[idx].set_xlim(-15, 15)
            axes[idx].set_ylim(-15, 15)
            axes[idx].set_aspect('equal')

        plt.tight_layout()
        plt.savefig('2d_forward_diffusion.png', dpi=150)
        plt.close()
        print("✓ Saved forward diffusion visualization")

    # ============================================================================
    # 익히기
    # ============================================================================

    def train_2d_diffusion(data, timesteps=100, epochs=1000, batch_size=128):
        """
        어떤 때 걸음에서든 잡음을 헤아리도록 모델을 익힌다.

        익힘 목표:
        -------------------
        회귀마다:
        1. 깨끗한 자료 x_0의 배치를 뽑는다
        2. 아무 때 걸음 t을 뽑는다
        3. 잡음을 더한다: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
        4. 잡음을 예측한다: ε_pred = model(x_t, t)
        5. 잃음: MSE(ε_pred, ε)
        6. 뒤먹임 퍼뜨리고 고친다

        모든 때 걸음에서 잡음 헤아리는 법을 배우면 모델은 은근히
        잡음 없애는 법을 배운다. 만들어 낼 때 이를 거듭 써서
        잡음 → 자료로 바꾼다.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training on {device}")

        # 확산 파라메터를 채비한다
        betas = cosine_beta_schedule(timesteps)
        diffusion_params = get_diffusion_parameters(betas)
        for key in diffusion_params:
            diffusion_params[key] = diffusion_params[key].to(device)
        data = data.to(device)

        # 모델과 가장 좋게 하개를 첫자리매김한다
        model = Simple2DModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            # 아무 배치를 뽑는다
            indices = torch.randint(0, len(data), (batch_size,))
            x_0 = data[indices]

            # 아무 때 걸음을 뽑는다
            t = torch.randint(0, timesteps, (batch_size,), device=device)

            # 로스를 셈한다(앞 확산 + 잡음 헤아리기를 다룬다)
            loss = get_loss(model, x_0, t, diffusion_params)

            # 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.4f}")

        # 익히기 나아감을 그린다
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('2d_training_loss.png', dpi=150)
        plt.close()
        print("✓ Saved training loss plot")

        return model, diffusion_params

    # ============================================================================
    # 뽑기(뒤 확산)
    # ============================================================================

    @torch.no_grad()
    def sample_2d(model, n_samples, timesteps, diffusion_params, device="cpu"):
        """
        뒤 확산으로 새 표본을 만든다(잡음 → 자료).

        알고리즘:
        ----------
        1. 비롯함: x_T ~ N(0, I) [순수한 마구잡이 잡음]
        2. t = T, T-1, ..., 1에 대해:
               ε_pred = model(x_t, t)                      # 잡음을 헤아린다
               μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_pred)  # 평균을 셈한다
               x_{t-1} = μ_θ + σ_t·z  (z ~ N(0,I))         # 작은 잡음을 더한다
        3. 끝: x_0 [깨끗하게 만들어진 표본]

        뒤 확산 식:
        -------------------------------
            x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ(x_t,t)) + σ_t·z
                      └─────────────── μ_θ(x_t,t) ───────────┘  └─잡음─┘

        하나씩 보면:
        ----------
        1. ε_θ(x_t,t): 모델이 x_t의 잡음을 헤아린다
        2. β_t/√(1-ᾱ_t)·ε_θ: 없앨 잡음의 잣수를 맞춘 것
        3. x_t - [잡음]: 어림한 잡음을 뺀다
        4. 1/√α_t · [...]: 흩어짐이 알맞도록 잣대를 다시 잡는다
        5. σ_t·z: 다양함을 위해 다스린 작은 잡음을 더한다

        왜 잡음을 없애면서 잡음을 더하는가?
        ------------------------------
        - 없으면: 정해진 길, 다양함 없음, 봉우리 무너짐
        - 있으면: 확률 길, 다양한 표본, 온전한 분포
        - 핵심: 더한 잡음 < 걷어 낸 잡음이므로 알짜로는 깨끗한 자료 쪽으로 나아간다!

        남다른 자리:
        -------------
        t=0(마지막 걸음)에서는 잡음을 더하지 않는다: x_0 = μ_θ(x_1, 0)
        이것이 마지막 깨끗한 만든 표본을 준다.

        보기(한 걸음):
        -------------------
        x_100 = [0.5, -0.3] (잡음 섞임)
        ε_pred = [0.4, -0.2] (모형의 예측)
        μ_θ = [0.489, -0.295] (잡음을 걷어 낸 평균)
        z = [0.01, 0.02] (마구잡이 잡음)
        x_99 = μ_θ + 0.05·z = [0.4905, -0.294] (조금 더 깨끗해졌다!)
        """
        model.eval()

        # 순수 잡음에서 시작한다
        x_t = torch.randn(n_samples, 2, device=device)

        # 거듭 잡음을 없앤다: T → T-1 → ... → 1 → 0
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)

            # 잡음을 헤아린다
            predicted_noise = model(x_t, t_tensor)

            # 이 때 걸음의 파라메터를 얻는다
            beta_t = diffusion_params['betas'][t]
            sqrt_recip_alpha_t = diffusion_params['sqrt_recip_alphas'][t]
            sqrt_one_minus_alpha_cumprod_t = diffusion_params['sqrt_one_minus_alphas_cumprod'][t]

            # 잡음 없앤 평균을 셈한다: μ_θ = 1/√α_t·(x_t - β_t/√(1-ᾱ_t)·ε_θ)
            mean = sqrt_recip_alpha_t * (
                x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise
            )

            if t > 0:
                # 다양함을 위해 다스린 잡음을 더한다
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(diffusion_params['posterior_variance'][t])
                x_t = mean + sigma * noise
            else:
                # 마지막 걸음: 잡음을 더하지 않는다
                x_t = mean

        return x_t  # x_0: 깨끗한 만든 표본

    def visualize_results(original_data, generated_data):
        """본디 익히기 자료와 만든 표본을 견준다."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(original_data[:, 0].cpu().numpy(), 
                       original_data[:, 1].cpu().numpy(), 
                       alpha=0.5, s=10)
        axes[0].set_title('Original Data')
        axes[0].set_aspect('equal')

        axes[1].scatter(generated_data[:, 0].cpu().numpy(), 
                       generated_data[:, 1].cpu().numpy(), 
                       alpha=0.5, s=10, color='red')
        axes[1].set_title('Generated Data')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.savefig('2d_comparison.png', dpi=150)
        plt.close()
        print("✓ Saved comparison plot")

    # ============================================================================
    # 으뜸 흐름
    # ============================================================================

    def main():
        """
        온전한 2차원 확산 보여 주기.

        흐름:
        ---------
        1. 장난감 2차원 자료 묶음을 만든다(스위스 롤)
        2. 앞 확산을 그림으로 본다(자료 → 잡음)
        3. 잡음을 헤아리도록 모델을 익힌다
        4. 새 표본을 만든다(잡음 → 자료)
        5. 본디 것과 만든 것을 견준다
        """
        print("=" * 50)
        print("Simple 2D Diffusion Model Demo")
        print("=" * 50)

        # 1. 자료를 만든다
        print("\n1. Generating Swiss roll data...")
        data = generate_swiss_roll(n_samples=2000)
        print(f"   ✓ Generated {len(data)} points")

        # 2. 앞 확산을 그려 본다
        print("\n2. Visualizing forward diffusion...")
        timesteps = 100
        betas = cosine_beta_schedule(timesteps)
        diffusion_params = get_diffusion_parameters(betas)
        visualize_2d_diffusion(data, timesteps, diffusion_params)

        # 3. 모델을 익힌다
        print("\n3. Training diffusion model...")
        model, diffusion_params = train_2d_diffusion(
            data, timesteps=timesteps, epochs=2000, batch_size=256
        )
        print("   ✓ Training complete")

        # 4. 표본을 만든다
        print("\n4. Generating samples...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generated = sample_2d(model, 2000, timesteps, diffusion_params, device)
        print(f"   ✓ Generated {len(generated)} samples")

        # 5. 결과를 견준다
        print("\n5. Comparing results...")
        visualize_results(data, generated)

        print("\n" + "=" * 50)
        print("✓ Demo complete! Check generated images:")
        print("  • 2d_forward_diffusion.png - Data dissolving into noise")
        print("  • 2d_training_loss.png - Training progress")
        print("  • 2d_comparison.png - Original vs Generated")
        print("=" * 50)

        print("\nKEY TAKEAWAYS:")
        print("  ✓ Forward: Add noise progressively (no learning)")
        print("  ✓ Reverse: Remove noise progressively (learned)")
        print("  ✓ Model predicts noise at any timestep")
        print("  ✓ Generation: Noise → iterative denoising → data")
        print("=" * 50)

    if __name__ == "__main__":
        main()
    ```

```
==================================================
Simple 2D Diffusion Model Demo
==================================================

1. Generating Swiss roll data...
   ✓ Generated 2000 points

2. Visualizing forward diffusion...
✓ Saved forward diffusion visualization

3. Training diffusion model...
Training on cpu
✓ Saved training loss plot
   ✓ Training complete

4. Generating samples...
   ✓ Generated 2000 samples

5. Comparing results...
✓ Saved comparison plot

==================================================
✓ Demo complete! Check generated images:
  • 2d_forward_diffusion.png - Data dissolving into noise
  • 2d_training_loss.png - Training progress
  • 2d_comparison.png - Original vs Generated
==================================================

KEY TAKEAWAYS:
  ✓ Forward: Add noise progressively (no learning)
  ✓ Reverse: Remove noise progressively (learned)
  ✓ Model predicts noise at any timestep
  ✓ Generation: Noise → iterative denoising → data
==================================================
```

코드는 같은 폴더의 diffusion_utils.py에서 스케줄·누적 계수·정방향 관측과 로스를 가져온다. 이 파일의 수식 설명 중 상당수는 실행하지 않는 문자열 리터럴에 들어 있다. 그 설명도 코드 블록과 함께 보존하며, 아래에서 실제 계산과 수학적 전제를 구분한다.

## 2. 데이터와 시간 조건 모델

`generate_swiss_roll`은 0~4π의 균등한 각도 목록 t에서 (t cos t,t sin t)를 만들고 표준편차 0.1의 노이즈를 더한다. scikit-learn의 3차원 Swiss roll을 투영한 다른 페이지의 데이터와 같지 않다. main은 표본 2,000개를 사용하며 별도로 정의한 반달 데이터 함수는 호출하지 않는다.

`Simple2DModel`은 시간 t/1000을 1→64→64 MLP로 바꾸고 2차원 좌표와 연결한다. B×66 입력을 128차원 은닉 레이어 세 개와 출력 레이어로 보내 B×2 노이즈를 예측한다. 기본 파라미터는 46,146개다. 실제 T=100이므로 시간 조건은 0~0.099이고 T를 바꿔도 정규화 분모는 그대로다.

## 3. 정방향 관측과 코드 인덱스

외부 cosine_beta_schedule은 β를 [0.0001,0.9999]로 clipping한다. get_diffusion_parameters는 같은 β에서 α·누적곱과 사후분산을 한 번 계산한다. 관측은 다음과 같다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

코드의 t=0은 첫 β를 이미 포함하므로 깨끗한 원본이 아니다. 코드 문자열의 이론상 ᾱ₀=1과 0부터 시작하는 배열 인덱스를 구분한다. 't=100이면 ᾱ≈0.37' 같은 문자열 수치도 선택한 스케줄에 관계없이 성립하지 않는다. 원본 데이터가 단위 분산이 아니므로 계수 제곱합이 1이라는 이유만으로 관측 분산이 원본과 같다고 할 수도 없다.

`visualize_2d_diffusion`은 0~99에서 열 시점을 고르고 **각 패널마다 새 ε를 뽑는다**. 따라서 같은 점이 하나의 마르코프 경로를 따라 움직이는 연속 궤적이 아니라 시점별 조건부 주변 샘플의 비교다. 축은 −15~15로 고정해 범위 밖 샘플은 보이지 않을 수 있다. num_steps를 바꾸더라도 subplot 수는 열 개라 열보다 많으면 인덱스가 맞지 않는다.

## 4. 학습 횟수와 로스

train_2d_diffusion은 매 반복 데이터에서 **복원추출로 256개**를 뽑아 한 번 갱신한다. main의 epochs=2000은 전체 데이터 2,000회 순회가 아니라 optimizer 업데이트 2,000번이다. t는 표본별 0~99에서 균등하게 선택하고, get_loss는 실제 ε와 모델 출력의 원소 평균 MSE를 계산한다.

이는 단순 노이즈 예측 로스며 전체 변분 우도를 직접 계산하는 함수가 아니다. 표준 정규 ε를 목표로 써도 시간별 예측 난이도와 조건부 분산이 같은 것은 아니다. 러닝레이트 0.001의 Adam을 사용하며 시드·검증·체크포인트·조기 종료는 없다. 데이터와 누적 계수는 선택한 장치로 옮긴다.

## 5. 보존된 사후평균 유도의 계수

코드 문자열의 완전제곱 전개에는 계수 2가 섞여 있다. 정밀도 P와 선형 계수 b를 다음처럼 정의하면 혼동을 피할 수 있다.

$$P=\frac{\alpha_t}{\beta_t}+\frac1{1-\bar\alpha_{t-1}},\qquad
b=\frac{\sqrt{\alpha_t}}{\beta_t}x_t+
\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0$$

음의 로그 밀도의 이차항은 $(P/2)\|x_{t-1}\|^2-b^Tx_{t-1}$이므로 평균은 **b/P**, 분산은 1/P다. 문자열처럼 A=P/2, B=b라고 정의했다면 평균은 B/(2A)여야 한다. B/A로 전개한 뒤 마지막에 2를 생략하는 과정은 올바른 유도가 아니다.

정리된 역평균의 최종 식 자체와 sample_2d의 계산은 다음과 같다.

$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

원본에 조건을 둔 사후분포 q(xₜ₋₁|xₜ,x₀)와 원본을 모르는 생성용 역전이를 구분한다. 관련 유도는 [역방향 과정](../foundations/reverse_process.md)을 참고한다.

## 6. 샘플링과 결과 해석

sample_2d는 no_grad·eval에서 N×2 표준 정규 초기값을 만들고 T개 역단계를 수행한다. 추가 잡음에는 βt가 아니라 helper의 **사후분산 β̃t**를 사용하며 마지막에는 잡음을 생략한다. model·상수 사전·초기값과 함수의 device 인수가 같아야 한다.

main은 2,000개 샘플을 100단계로 생성하도록 구성되어 있고 마지막 산점도는 원본과 생성 결과를 나란히 보여 준다. CPU 출력과 저장 메시지는 기존 기록이며 여기서 새로 실행한 것이 아니다. 그림의 유사성만으로 모든 나선 영역의 확률 질량, 일반화나 비암기를 확정할 수 없다. 독립 데이터와 적절한 피처·분포 평가를 보완해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
2차원 모델의 시간·좌표 입력이 어떻게 결합되는가?

</div>

??? success "연습문제 1 풀이"

    길이 B의 시간을 B×1로 바꾸고 1000으로 나눈 뒤 64차원 MLP 출력을 얻는다. B×2 좌표와 연결한 B×66이 은닉 레이어 세 개를 거쳐 B×2 노이즈 예측이 된다. T=100에서도 분모는 1000이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
사후 유도에서 이차항이 (P/2)‖u‖²−bᵀu이면 평균과 분산은 무엇인가?

</div>

??? success "연습문제 2 풀이"

    완전제곱은 (P/2)‖u−b/P‖²+상수다. 따라서 평균 b/P, 등방 분산 1/P다. A=P/2라 부르면 평균은 b/(2A)이며 b/A가 아니다. 코드 문자열의 중간 유도와 실제 샘플러의 최종 식을 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
학습 2,000회와 정방향 그림의 열 패널이 각각 무엇을 의미하는가?

</div>

??? success "연습문제 3 풀이"

    학습은 복원추출 미니배치 하나씩 총 2,000번 갱신한다. 정방향 패널은 열 시점마다 새 노이즈로 만든 주변 샘플이며 하나의 연속 경로가 아니다. 둘을 전체 데이터 2,000회 순회나 동일한 마르코프 궤적으로 해석하지 않는다.



## 정리하며

2차원 DDPM은 시간 조건 노이즈 회귀와 사후분산 역전이를 연결한다. 업데이트 수와 전체 순회, 시점별 주변 샘플과 연속 궤적을 구분한다. 보존된 문자열 유도의 계수 오류와 실제 계산의 최종 식도 따로 확인해야 한다.
