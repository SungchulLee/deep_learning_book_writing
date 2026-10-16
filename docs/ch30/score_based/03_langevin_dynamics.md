# 랑주뱅 동역학

랑주뱅 동역학은 로그 밀도의 그래디언트에 따른 이동과 가우시안 잡음을 결합한다. 정규화 상수를 모르는 목표 분포에도 적용할 수 있지만 연속시간 확산, 보정 없는 이산 반복, Metropolis 보정이 있는 방법을 구분해야 한다. 이 페이지는 1차원 혼합분포와 학습한 2차원 스코어를 사용하는 예제를 살펴본다.

## 1. 샘플링 비교와 학습 코드

??? note "코드 (606줄)"

    ```python
    """
    뽑기를 위한 랑주뱅 액션
    ==========================================

    어려움: 중간
    시간: 2~3시간
    미리 알 것: 단원 01-02, 기본 마르코프 사슬 몬테카를로 이해

    학습 목표:
    -------------------
    1. 랑주뱅 마르코프 사슬 몬테카를로를 그래디언트 바탕 뽑기로 이해한다
    2. 베이즈 추론의 메트로폴리스-헤이스팅스와 잇는다
    3. 점수 바탕 뽑기를 위한 랑주뱅 액션을 짠다
    4. 모임의 성질을 살핀다
    5. 확산의 뒤 과정과의 이음을 이해한다

    수학 바탕:
    -----------------------
    랑주뱅 액션(랑주뱅 1908, 이후 여러 사람이 넓힘):

    dx_t = ∇_x log p(x_t) dt + √(2)dW_t

    띄엄띄엄한 때의 꼴(랑주뱅 MCMC):

    x_{t+1} = x_t + ε * ∇_x log p(x_t) + √(2ε) * z_t

    where z_t ~ N(0, I)

    핵심 눈썰미:
    - 떠돎 항: ε * ∇_x log p(x_t)이 확률 높은 쪽으로 옮긴다
    - 확산 항: √(2ε) * z_t이 살펴보기 잡음을 더한다
    - 캐내기(그래디언트)와 살펴보기(잡음)의 균형을 잡는다
    - 목표 분포 p(x)으로 모인다!

    점수 맞추기와의 이음:
    s(x) = ∇_x log p(x)이므로 배운 점수로 표본을 뽑을 수 있다!

    x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

    이것이 확산 모델이 표본을 만드는 방식이다!

    지은이: 이성철 @ 연세대학교
    날짜: 2025년 11월
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    plt.style.use('seaborn-v0_8-darkgrid')
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 80)
    print("MODULE 03: LANGEVIN DYNAMICS")
    print("=" * 80)

    # ============================================================================
    # 마디 1: 메트로폴리스-헤이스팅스에서 랑주뱅으로
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: Evolution from Metropolis-Hastings to Langevin")
    print("=" * 80)

    print("""
    떠올리기: 메트로폴리스-헤이스팅스(베이즈 미룸에서)
    ----------------------------------------------------
    1. Propose: x' ~ q(x'|x_t)  (e.g., x' = x_t + N(0, σ²))
    2. 확률 α = min(1, p(x')/p(x_t))로 받아들인다
    3. If accepted: x_{t+1} = x', else x_{t+1} = x_t

    문제:
    - 받아들임 비율이 낮을 수 있다(물리치는 제안이 많다)
    - 고르게 맞추기까지 담아 p(x)을 셈해야 한다
    - 아무 걸음 제안은 효율이 나쁘다

    랑주뱅 마르코프 사슬 몬테카를로: 그래디언트에 바탕한 개선
    ----------------------------------------
    핵심 통찰: 그래디언트 앎을 쓴다!

    마구잡이 걸음 대신:
    x' = x_t + N(0, σ²)

    앎을 담은 제안을 쓴다.
    x' = x_t + ε * ∇log p(x_t) + N(0, 2ε)

    이점:
    - 제안이 확률이 높은 쪽으로 움직인다(그래디언트 표류)
    - 그러면서도 살펴본다(가우스 잡음)
    - ε→0의 끝에서는 제안이 늘 받아들여진다!
    - 온전한 확률이 아니라 점수만 필요하다!

    이것이 고치지 않은 랑주뱅 알고리즘(ULA)이다.
    """)

    # 차이를 보여 준다
    def metropolis_hastings_1d(target_logpdf, n_samples=5000, sigma=0.5):
        """아무 걸음을 쓴 여느 메트로폴리스-헤이스팅스"""
        samples = [0.0]  # 원점에서 시작
        x = 0.0
        n_accepted = 0

        for _ in range(n_samples):
            # 내놓기
            x_prop = x + np.random.randn() * sigma

            # 받아들이거나 물리치기
            log_alpha = target_logpdf(x_prop) - target_logpdf(x)
            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                n_accepted += 1

            samples.append(x)

        acceptance_rate = n_accepted / n_samples
        return np.array(samples), acceptance_rate

    def langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000, epsilon=0.01):
        """그래디언트 앎을 쓴 랑주뱅 마르코프 사슬 몬테카를로"""
        samples = [0.0]
        x = 0.0

        for _ in range(n_samples):
            # 랑주뱅 고침
            x = x + epsilon * score_fn(x) + np.sqrt(2 * epsilon) * np.random.randn()
            samples.append(x)

        return np.array(samples)

    # 목표: 정규 분포 섞기
    def target_logpdf(x):
        """섞임 분포의 로그 확률밀도: 0.3*N(-2,0.5²) + 0.7*N(2,0.5²)"""
        from scipy.stats import norm
        p1 = 0.3 * norm.pdf(x, -2, 0.5)
        p2 = 0.7 * norm.pdf(x, 2, 0.5)
        return np.log(p1 + p2 + 1e-10)

    def score_fn(x):
        """섞기의 점수"""
        from scipy.stats import norm
        p1 = 0.3 * norm.pdf(x, -2, 0.5)
        p2 = 0.7 * norm.pdf(x, 2, 0.5)

        s1 = -(x + 2) / 0.25  # N(-2, 0.5²)의 점수
        s2 = -(x - 2) / 0.25  # N(2, 0.5²)의 점수

        total_p = p1 + p2 + 1e-10
        return (p1 * s1 + p2 * s2) / total_p

    print("\nComparison: Metropolis-Hastings vs Langevin MCMC")
    print("-" * 80)

    # 둘 다 돌린다
    mh_samples, acc_rate = metropolis_hastings_1d(target_logpdf, n_samples=5000)
    langevin_samples = langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000)

    print(f"Metropolis-Hastings acceptance rate: {acc_rate:.2%}")
    print(f"Langevin MCMC: No rejection (always accepts in continuous limit)")

    # 시각화한다
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 참 분포
    x_plot = np.linspace(-4, 4, 1000)
    true_pdf = np.exp([target_logpdf(x) for x in x_plot])
    true_pdf /= np.trapz(true_pdf, x_plot)   # NumPy 2.0의 np.trapezoid와 같은 함수

    # 그림 1: 메트로폴리스-헤이스팅스 자취
    axes[0, 0].plot(mh_samples[:500], alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Metropolis-Hastings Trajectory', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(r'$x$')
    axes[0, 0].grid(True, alpha=0.3)

    # 그림 2: 랑주뱅 자취
    axes[0, 1].plot(langevin_samples[:500], alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title('Langevin MCMC Trajectory', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'$x$')
    axes[0, 1].grid(True, alpha=0.3)

    # 그림 3: 메트로폴리스-헤이스팅스 히스토그램
    axes[1, 0].hist(mh_samples[1000:], bins=50, density=True, alpha=0.6, label='MH samples')
    axes[1, 0].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
    axes[1, 0].set_title(f'Metropolis-Hastings\n(Acc. rate: {acc_rate:.1%})', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 그림 4: 랑주뱅 히스토그램
    axes[1, 1].hist(langevin_samples[1000:], bins=50, density=True, alpha=0.6, label='Langevin samples')
    axes[1, 1].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
    axes[1, 1].set_title('Langevin MCMC\n(Gradient-guided)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_mh_vs_langevin.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: 03_mh_vs_langevin.png")
    plt.close()

    print("""
    핵심 살핌:
    - 랑주뱅은 그래디언트 앎을 써서 더 효율 좋게 살핀다
    - 둘 다 목표 분포로 모인다
    - 랑주뱅은 복잡하고 봉우리 여럿인 분포를 더 잘 다룬다
    - M-H의 받아들임 비율 절충은 랑주뱅에는 들어맞지 않는다
    """)

    # ============================================================================
    # 마디 2: 랑주뱅 액션 이론
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Theory of Langevin Dynamics")
    print("=" * 80)

    print("""
    이어진 때의 랑주뱅 액션:
    ---------------------------------
    확률 미분 방정식(SDE):

    dx_t = ∇_x log p(x_t) dt + √2 dW_t

    여기서 W_t은 여느 브라운 액션이다.

    느낌으로 보면:
    - 정해진 떠돎: ∇log p(x)이 확률 높은 쪽으로 끈다
    - 확률 확산: √2 dW_t이 공간을 살핀다
    - 균형이 p(x)로 모여듦을 보장한다

    포커-플랑크 방정식:
    ----------------------
    x_t의 분포 p_t(x)는 다음과 같이 바뀐다.

    ∂p_t/∂t = -∇·(p_t ∇log p) + Δp_t
            = ∇·(p_t ∇log(p/p_t))

    At equilibrium (∂p_t/∂t = 0), we have p_t = p!

    띄엄띄엄한 때의 꼴:
    --------------------
    오일러-마루야마 띄엄띄엄하게 하기:

    x_{t+1} = x_t + ε ∇_x log p(x_t) + √(2ε) z_t

    여기서 z_t ~ N(0, I)이고 ε은 걸음 크기이다.

    CONVERGENCE:
    - ε→0이면 띄엄띄엄한 과정 → 이어진 랑주뱅 확률 미분 방정식
    - ε이 넉넉히 작으면 목표 분포로 모인다
    - 모여드는 빠르기는 p(x)의 성질에 달렸다

    실제로 살필 점:
    - 걸음 크기 ε: 너무 크면 흔들리고 너무 작으면 느리게 모여든다
    - 걸음 수: 많을수록 표본이 좋아지지만 느려진다
    - 첫자리: 어떤 분포에서 비롯해도 된다(보기: N(0,I))

    점수 맞추기와의 이음:
    ----------------------------
    자료에서 s_θ(x) ≈ ∇log p(x)를 배웠으므로,
    다음으로 표본을 뽑을 수 있다.

    x_{t+1} = x_t + ε s_θ(x_t) + √(2ε) z_t

    이것이 점수 바탕 뽑기이다!
    고르게 맞춘 확률이 필요 없다!
    """)

    # ============================================================================
    # 마디 3: 랑주뱅 뽑기 짜기
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Langevin Sampling with Learned Scores")
    print("=" * 80)

    print("\nExample: 2D Swiss Roll with Learned Score Network")
    print("-" * 80)

    # 스위스 롤을 만든다
    def make_swiss_roll(n_samples=2000, noise=0.1):
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        X = np.stack([x, y], axis=1)
        X += noise * np.random.randn(n_samples, 2)
        return X

    swiss_data = make_swiss_roll(2000)

    # 점수 뉴럴 넷트웍을 익힌다(단원 02을 단순하게 만든 것)
    class ScoreNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 2)
            )

        def forward(self, x):
            return self.net(x)

    def train_score_dsm(data, sigma=0.5, n_epochs=1000):
        """잡음 없애는 점수 맞추기로 점수 뉴럴 넷트웍을 익힌다"""
        model = ScoreNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data_tensor = torch.FloatTensor(data)

        for epoch in range(n_epochs):
            # 잡음 없애는 점수 맞추기 로스
            noise = torch.randn_like(data_tensor)
            noisy_data = data_tensor + sigma * noise
            predicted_score = model(noisy_data)
            target_score = -noise / sigma

            loss = torch.mean((predicted_score - target_score) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

        return model

    print("Training score network...")
    score_net = train_score_dsm(swiss_data, sigma=0.5, n_epochs=1000)
    print("✓ Training complete!")

    # 랑주뱅 뽑기 함수
    def langevin_sampling(score_model, n_samples=500, n_steps=1000, epsilon=0.01, 
                         init_x=None):
        """
        배운 점수로 랑주뱅 액션을 써서 뽑는다.

        인수:
            score_model: 익힌 점수 뉴럴 넷트웍
            n_samples: 만들 표본의 개수
            n_steps: 랑주뱅 걸음 수
            epsilon: 걸음 크기
            init_x: 첫 표본(None이면 N(0, I)을 쓴다)

        반환값:
            samples: 만들어 낸 표본 [n_samples, dim]
            trajectory: 뽑기의 온 자취 [n_steps, n_samples, dim]
        """
        if init_x is None:
            x = torch.randn(n_samples, 2) * 3  # 더 넓은 분포에서 첫자리매김
        else:
            x = init_x.clone()

        trajectory = [x.clone().detach().numpy()]

        for step in range(n_steps):
            with torch.no_grad():
                score = score_model(x)

            # 랑주뱅 고침
            x = x + epsilon * score + np.sqrt(2 * epsilon) * torch.randn_like(x)

            if step % 100 == 0:
                trajectory.append(x.clone().detach().numpy())

        samples = x.detach().numpy()
        return samples, trajectory

    print("\nGenerating samples via Langevin dynamics...")
    samples, trajectory = langevin_sampling(score_net, n_samples=500, n_steps=1000, epsilon=0.01)
    print("✓ Sampling complete!")

    # 결과를 그려 본다
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 그림 1: 참 자료
    axes[0].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.5, c='blue')
    axes[0].set_title('True Data Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(r'$x_1$')
    axes[0].set_ylabel(r'$x_2$')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # 그림 2: 만든 표본
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
    axes[1].set_title('Generated via Langevin Sampling', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(r'$x_1$')
    axes[1].set_ylabel(r'$x_2$')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)

    # 그림 3: 견주기
    axes[2].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.3, c='blue', label='True')
    axes[2].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='red', label='Generated')
    axes[2].set_title('Overlay Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xlabel(r'$x_1$')
    axes[2].set_ylabel(r'$x_2$')
    axes[2].legend()
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_langevin_sampling_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_langevin_sampling_results.png")
    plt.close()

    # 자취가 바뀌는 모습을 그려 본다
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (ax, traj_snapshot) in enumerate(zip(axes, trajectory)):
        step = idx * 100
        ax.scatter(traj_snapshot[:, 0], traj_snapshot[:, 1], s=1, alpha=0.5, c='purple')
        ax.scatter(swiss_data[:, 0], swiss_data[:, 1], s=0.5, alpha=0.1, c='blue')
        ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Langevin Sampling Evolution Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_langevin_trajectory.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_langevin_trajectory.png")
    plt.close()

    print("""
    ✓ 랑주뱅 뽑기가 스위스 롤에서 표본을 잘 만들었다!

    눈여겨볼 점은 다음과 같다.
    - 아무 정규 잡음에서 시작했다
    - 점수 이끌기로 자료 분포 쪽으로 차츰 옮겨 갔다
    - 마지막 표본이 참 자료 분포와 잘 맞는다
    - 이것이 바로 확산 모델이 만들어 내는 방식이다!
    """)

    # ============================================================================
    # 마디 4: 확산 모델과의 이음
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Connection to Diffusion Models")
    print("=" * 80)

    print("""
    확산 모델은 여러 자의 랑주뱅 액션이다
    -----------------------------------------------

    핵심 통찰: 여러 잡음 수준에서 랑주뱅 뽑기를 쓴다!

    앞으로 가는 흐름(확산):
    x_0 → x_1 → x_2 → ... → x_T
    x_T ~ N(0, I)이 될 때까지 잡음을 차츰 더한다

    거꾸로 가는 흐름(만들어 내기):
    x_T ← x_{T-1} ← ... ← x_1 ← x_0
    배운 점수로 차츰 잡음을 없앤다!

    거꾸로 걸음:
    x_{t-1} = x_t + ε * s_θ(x_t, t) + √(2ε) * z_t

    여기서 s_θ(x_t, t)은 잡음 수준 t의 점수이다.

    이것이 바로 랑주뱅 액션이다!
    다만 때에 따라 달라지는 점수를 쓸 뿐이다!

    달군 랑주뱅 액션:
    --------------------------
    잡음 레이어의 이음을 쓴다: σ_1 > σ_2 > ... > σ_L

    레이어 i마다:
    1. Learn score s_θ(x, σ_i)
    2. 랑주뱅을 돌린다: x ← x + ε * s_θ(x, σ_i) + √(2ε) * z

    잡음이 큰 데서 비롯한다(뽑기 쉬우나 성기다)
    잡음을 차츰 줄인다(뽑기 어려우나 더 촘촘하다)

    이것이 다음의 밑바탕이다.
    - 점수 바탕 만들어 내는 모델(Song & Ermon, 2019)
    - 잡음 지우기 확산 확률 모델(Ho 외, 2020)
    - 요즘의 모든 확산 모델이다!

    여태 배운 것:
    ------------------------
    ✓ 점수 함수란 무엇인가?
    ✓ 어떻게 배우는가(점수 맞추기)?
    ✓ 어떻게 쓰는가(랑주뱅 뽑기)?

    다음: 실제 쓰임새를 위한 여러 잣수 틀!
    """)

    # ============================================================================
    # 간추리기와 익힘
    # ============================================================================
    print("\n" + "=" * 80)
    print("MODULE SUMMARY")
    print("=" * 80)

    print("""
    배운 것:
    ---------------
    1. 랑주뱅 액션은 그래디언트 바탕 MCMC 뽑기다
    2. 점수 함수로 뽑기를 이끈다
    3. 아무 걸음 메트로폴리스-헤이스팅스보다 효율이 좋다
    4. 점수 맞추기로 배운 점수와 함께 통한다
    5. 확산 모델 뽑기의 바탕이다!

    핵심 식:
    ------------
    1. 이어진 랑주뱅 SDE:
       dx_t = ∇log p(x_t) dt + √2 dW_t

    2. 띄엄띄엄한 랑주뱅 MCMC:
       x_{t+1} = x_t + ε * ∇log p(x_t) + √(2ε) * z_t

    3. 점수 바탕 뽑기:
       x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

    4. 달군 랑주뱅(미리 보기):
       잡음 레이어 σ_1 > ... > σ_L에서 점수를 쓴다

    만들어진 파일:
    ---------------
    1. 03_mh_vs_langevin.png - 메트로폴리스-헤이스팅스와의 견줌
    2. 03_langevin_sampling_results.png - 만들어 낸 표본
    3. 03_langevin_trajectory.png - 때에 따른 바뀜
    """)

    print("\n" + "=" * 80)
    print("EXERCISES")
    print("=" * 80)

    print("""
    익힘 1: 모임 살피기
    -------------------------------
    For 1D Gaussian N(0,1):
    a) 걸음 크기 ε을 달리해 랑주뱅을 돌린다
    b) 모여드는 빠르기를 재어라(타깃까지의 KL 갈림)
    c) ε에 따른 모임을 그린다
    d) 가장 좋은 걸음 크기를 찾는다

    익힘 2: 봉우리 여럿인 분포
    -----------------------------------
    가우스 4개(정사각형의 꼭짓점)로 2차원 섞임 분포를 만들어라.
    a) 잡음 없애는 점수 맞추기로 점수 뉴럴 넷트웍을 익힌다
    b) 랑주뱅 뽑기를 돌린다
    c) 모든 봉우리를 들르는가?
    d) 여러 첫자리매김과 걸음 크기를 시험한다

    익힘 3: 메트로폴리스로 고친 랑주뱅
    ---------------------------------------
    MALA를 짜라(메트로폴리스 받아들임을 더한다).
    a) 랑주뱅으로 제안한다: x' = x + ε*∇log p(x) + √(2ε)*z
    b) 메트로폴리스-헤이스팅스 확률로 받아들인다
    c) 고치지 않은 랑주뱅과 견준다
    d) 메트로폴리스로 고친 랑주뱅은 언제 도움이 되는가?

    익힘 4: 점수의 어긋남
    -----------------------
    점수 넷트웍에 어긋남이 있어 s_θ(x) ≠ ∇log p(x)이면:
    a) 이것이 표본에 어떤 영향을 주는가?
    b) 일부러 점수를 치우치게 하여 짠다
    c) 분포의 어긋남을 잰다
    d) 확산 모델에 어떤 뜻이 있는가?

    익힘 5: 식힘 차례표
    -----------------------------
    달군 랑주뱅을 짜라.
    a) 잡음 짜임 σ_1 > σ_2 > ... > σ_L을 뜻매김하여라
    b) 레이어마다 점수를 익혀라(2단원을 보라)
    c) 큰 잡음에서 시작해 뽑는다
    d) 수준 하나의 랑주뱅과 견준다

    도전 익힘: 3차원 뽑기
    ------------------------------
    Extend to 3D:
    a) 3차원 스위스 롤이나 나선 자료를 만든다
    b) 3차원 점수 뉴럴 넷트웍을 익힌다
    c) 랑주뱅 뽑기를 짠다
    d) 3차원 자취를 그려 본다
    e) 차원이 모임에 어떤 영향을 주는가?
    """)

    print("\n" + "=" * 80)
    print("NEXT: Multi-Scale Score Modeling")
    print("=" * 80)
    print("""
    이제 벽돌이 모두 갖추어졌다.
    ✓ 점수 함수(1단원)
    ✓ 그것을 배우는 점수 맞추기(2단원)
    ✓ 표본을 뽑는 랑주뱅 액션(3단원)

    다음: 실제 자료(그림 등)에서 이것이 통하게 하려면?

    답: 여러 잡음 수준에서 점수를 배운다!
    이것이 온전한 확산 틀로 이어진다!
    """)

    print("\n✓ Module 03 complete!")
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```

??? note "전체 출력 (289줄)"

    ```
    ================================================================================
    MODULE 03: LANGEVIN DYNAMICS
    ================================================================================

    ================================================================================
    SECTION 1: Evolution from Metropolis-Hastings to Langevin
    ================================================================================

    떠올리기: 메트로폴리스-헤이스팅스(베이즈 미룸에서)
    ----------------------------------------------------
    1. Propose: x' ~ q(x'|x_t)  (e.g., x' = x_t + N(0, σ²))
    2. 확률 α = min(1, p(x')/p(x_t))로 받아들인다
    3. If accepted: x_{t+1} = x', else x_{t+1} = x_t

    문제:
    - 받아들임 비율이 낮을 수 있다(물리치는 제안이 많다)
    - 고르게 맞추기까지 담아 p(x)을 셈해야 한다
    - 아무 걸음 제안은 효율이 나쁘다

    랑주뱅 마르코프 사슬 몬테카를로: 그래디언트에 바탕한 개선
    ----------------------------------------
    핵심 통찰: 그래디언트 앎을 쓴다!

    마구잡이 걸음 대신:
    x' = x_t + N(0, σ²)

    앎을 담은 제안을 쓴다.
    x' = x_t + ε * ∇log p(x_t) + N(0, 2ε)

    이점:
    - 제안이 확률이 높은 쪽으로 움직인다(그래디언트 표류)
    - 그러면서도 살펴본다(가우스 잡음)
    - ε→0의 끝에서는 제안이 늘 받아들여진다!
    - 온전한 확률이 아니라 점수만 필요하다!

    이것이 고치지 않은 랑주뱅 알고리즘(ULA)이다.


    Comparison: Metropolis-Hastings vs Langevin MCMC
    --------------------------------------------------------------------------------
    Metropolis-Hastings acceptance rate: 72.10%
    Langevin MCMC: No rejection (always accepts in continuous limit)

    ✓ Saved: 03_mh_vs_langevin.png

    핵심 살핌:
    - 랑주뱅은 그래디언트 앎을 써서 더 효율 좋게 살핀다
    - 둘 다 목표 분포로 모인다
    - 랑주뱅은 복잡하고 봉우리 여럿인 분포를 더 잘 다룬다
    - M-H의 받아들임 비율 절충은 랑주뱅에는 들어맞지 않는다


    ================================================================================
    SECTION 2: Theory of Langevin Dynamics
    ================================================================================

    이어진 때의 랑주뱅 액션:
    ---------------------------------
    확률 미분 방정식(SDE):

    dx_t = ∇_x log p(x_t) dt + √2 dW_t

    여기서 W_t은 여느 브라운 액션이다.

    느낌으로 보면:
    - 정해진 떠돎: ∇log p(x)이 확률 높은 쪽으로 끈다
    - 확률 확산: √2 dW_t이 공간을 살핀다
    - 균형이 p(x)로 모여듦을 보장한다

    포커-플랑크 방정식:
    ----------------------
    x_t의 분포 p_t(x)는 다음과 같이 바뀐다.

    ∂p_t/∂t = -∇·(p_t ∇log p) + Δp_t
            = ∇·(p_t ∇log(p/p_t))

    At equilibrium (∂p_t/∂t = 0), we have p_t = p!

    띄엄띄엄한 때의 꼴:
    --------------------
    오일러-마루야마 띄엄띄엄하게 하기:

    x_{t+1} = x_t + ε ∇_x log p(x_t) + √(2ε) z_t

    여기서 z_t ~ N(0, I)이고 ε은 걸음 크기이다.

    CONVERGENCE:
    - ε→0이면 띄엄띄엄한 과정 → 이어진 랑주뱅 확률 미분 방정식
    - ε이 넉넉히 작으면 목표 분포로 모인다
    - 모여드는 빠르기는 p(x)의 성질에 달렸다

    실제로 살필 점:
    - 걸음 크기 ε: 너무 크면 흔들리고 너무 작으면 느리게 모여든다
    - 걸음 수: 많을수록 표본이 좋아지지만 느려진다
    - 첫자리: 어떤 분포에서 비롯해도 된다(보기: N(0,I))

    점수 맞추기와의 이음:
    ----------------------------
    자료에서 s_θ(x) ≈ ∇log p(x)를 배웠으므로,
    다음으로 표본을 뽑을 수 있다.

    x_{t+1} = x_t + ε s_θ(x_t) + √(2ε) z_t

    이것이 점수 바탕 뽑기이다!
    고르게 맞춘 확률이 필요 없다!


    ================================================================================
    SECTION 3: Langevin Sampling with Learned Scores
    ================================================================================

    Example: 2D Swiss Roll with Learned Score Network
    --------------------------------------------------------------------------------
    Training score network...
      Epoch 0: Loss = 4.015207
      Epoch 200: Loss = 3.909409
      Epoch 400: Loss = 3.900789
      Epoch 600: Loss = 3.759978
      Epoch 800: Loss = 3.511452
    ✓ Training complete!

    Generating samples via Langevin dynamics...
    ✓ Sampling complete!
    ✓ Saved: 03_langevin_sampling_results.png
    ✓ Saved: 03_langevin_trajectory.png

    ✓ 랑주뱅 뽑기가 스위스 롤에서 표본을 잘 만들었다!

    눈여겨볼 점은 다음과 같다.
    - 아무 정규 잡음에서 시작했다
    - 점수 이끌기로 자료 분포 쪽으로 차츰 옮겨 갔다
    - 마지막 표본이 참 자료 분포와 잘 맞는다
    - 이것이 바로 확산 모델이 만들어 내는 방식이다!


    ================================================================================
    SECTION 4: Connection to Diffusion Models
    ================================================================================

    확산 모델은 여러 자의 랑주뱅 액션이다
    -----------------------------------------------

    핵심 통찰: 여러 잡음 수준에서 랑주뱅 뽑기를 쓴다!

    앞으로 가는 흐름(확산):
    x_0 → x_1 → x_2 → ... → x_T
    x_T ~ N(0, I)이 될 때까지 잡음을 차츰 더한다

    거꾸로 가는 흐름(만들어 내기):
    x_T ← x_{T-1} ← ... ← x_1 ← x_0
    배운 점수로 차츰 잡음을 없앤다!

    거꾸로 걸음:
    x_{t-1} = x_t + ε * s_θ(x_t, t) + √(2ε) * z_t

    여기서 s_θ(x_t, t)은 잡음 수준 t의 점수이다.

    이것이 바로 랑주뱅 액션이다!
    다만 때에 따라 달라지는 점수를 쓸 뿐이다!

    달군 랑주뱅 액션:
    --------------------------
    잡음 레이어의 이음을 쓴다: σ_1 > σ_2 > ... > σ_L

    레이어 i마다:
    1. Learn score s_θ(x, σ_i)
    2. 랑주뱅을 돌린다: x ← x + ε * s_θ(x, σ_i) + √(2ε) * z

    잡음이 큰 데서 비롯한다(뽑기 쉬우나 성기다)
    잡음을 차츰 줄인다(뽑기 어려우나 더 촘촘하다)

    이것이 다음의 밑바탕이다.
    - 점수 바탕 만들어 내는 모델(Song & Ermon, 2019)
    - 잡음 지우기 확산 확률 모델(Ho 외, 2020)
    - 요즘의 모든 확산 모델이다!

    여태 배운 것:
    ------------------------
    ✓ 점수 함수란 무엇인가?
    ✓ 어떻게 배우는가(점수 맞추기)?
    ✓ 어떻게 쓰는가(랑주뱅 뽑기)?

    다음: 실제 쓰임새를 위한 여러 잣수 틀!


    ================================================================================
    MODULE SUMMARY
    ================================================================================

    배운 것:
    ---------------
    1. 랑주뱅 액션은 그래디언트 바탕 MCMC 뽑기다
    2. 점수 함수로 뽑기를 이끈다
    3. 아무 걸음 메트로폴리스-헤이스팅스보다 효율이 좋다
    4. 점수 맞추기로 배운 점수와 함께 통한다
    5. 확산 모델 뽑기의 바탕이다!

    핵심 식:
    ------------
    1. 이어진 랑주뱅 SDE:
       dx_t = ∇log p(x_t) dt + √2 dW_t

    2. 띄엄띄엄한 랑주뱅 MCMC:
       x_{t+1} = x_t + ε * ∇log p(x_t) + √(2ε) * z_t

    3. 점수 바탕 뽑기:
       x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

    4. 달군 랑주뱅(미리 보기):
       잡음 레이어 σ_1 > ... > σ_L에서 점수를 쓴다

    만들어진 파일:
    ---------------
    1. 03_mh_vs_langevin.png - 메트로폴리스-헤이스팅스와의 견줌
    2. 03_langevin_sampling_results.png - 만들어 낸 표본
    3. 03_langevin_trajectory.png - 때에 따른 바뀜


    ================================================================================
    EXERCISES
    ================================================================================

    익힘 1: 모임 살피기
    -------------------------------
    For 1D Gaussian N(0,1):
    a) 걸음 크기 ε을 달리해 랑주뱅을 돌린다
    b) 모여드는 빠르기를 재어라(타깃까지의 KL 갈림)
    c) ε에 따른 모임을 그린다
    d) 가장 좋은 걸음 크기를 찾는다

    익힘 2: 봉우리 여럿인 분포
    -----------------------------------
    가우스 4개(정사각형의 꼭짓점)로 2차원 섞임 분포를 만들어라.
    a) 잡음 없애는 점수 맞추기로 점수 뉴럴 넷트웍을 익힌다
    b) 랑주뱅 뽑기를 돌린다
    c) 모든 봉우리를 들르는가?
    d) 여러 첫자리매김과 걸음 크기를 시험한다

    익힘 3: 메트로폴리스로 고친 랑주뱅
    ---------------------------------------
    MALA를 짜라(메트로폴리스 받아들임을 더한다).
    a) 랑주뱅으로 제안한다: x' = x + ε*∇log p(x) + √(2ε)*z
    b) 메트로폴리스-헤이스팅스 확률로 받아들인다
    c) 고치지 않은 랑주뱅과 견준다
    d) 메트로폴리스로 고친 랑주뱅은 언제 도움이 되는가?

    익힘 4: 점수의 어긋남
    -----------------------
    점수 넷트웍에 어긋남이 있어 s_θ(x) ≠ ∇log p(x)이면:
    a) 이것이 표본에 어떤 영향을 주는가?
    b) 일부러 점수를 치우치게 하여 짠다
    c) 분포의 어긋남을 잰다
    d) 확산 모델에 어떤 뜻이 있는가?

    익힘 5: 식힘 차례표
    -----------------------------
    달군 랑주뱅을 짜라.
    a) 잡음 짜임 σ_1 > σ_2 > ... > σ_L을 뜻매김하여라
    b) 레이어마다 점수를 익혀라(2단원을 보라)
    c) 큰 잡음에서 시작해 뽑는다
    d) 수준 하나의 랑주뱅과 견준다

    도전 익힘: 3차원 뽑기
    ------------------------------
    Extend to 3D:
    a) 3차원 스위스 롤이나 나선 자료를 만든다
    b) 3차원 점수 뉴럴 넷트웍을 익힌다
    c) 랑주뱅 뽑기를 짠다
    d) 3차원 자취를 그려 본다
    e) 차원이 모임에 어떤 영향을 주는가?


    ================================================================================
    NEXT: Multi-Scale Score Modeling
    ================================================================================

    이제 벽돌이 모두 갖추어졌다.
    ✓ 점수 함수(1단원)
    ✓ 그것을 배우는 점수 맞추기(2단원)
    ✓ 표본을 뽑는 랑주뱅 액션(3단원)

    다음: 실제 자료(그림 등)에서 이것이 통하게 하려면?

    답: 여러 잡음 수준에서 점수를 배운다!
    이것이 온전한 확산 틀로 이어진다!


    ✓ Module 03 complete!
    ================================================================================
    ```

## 2. Metropolis–Hastings와 정규화 상수

대칭 랜덤 워크 제안의 Metropolis–Hastings 수락 확률은 $\min(1,p(x')/p(x))$다. 목표의 같은 정규화 상수는 비에서 약분되므로 **MH도 정규화된 밀도를 요구하지 않는다**. 코드 출력의 반대 설명은 잘못이다. 비대칭 제안에서는 제안 밀도의 정역 비도 필요하다.

`metropolis_hastings_1d`는 0에서 시작해 가우시안 제안을 5,000번 만든다. 초기 상태까지 반환하므로 길이는 5,001이다. 수락률은 이동 성공 비율이지 독립 샘플 비율이나 분포 전체 탐색의 보증이 아니다. 높은 수락률도 작은 이동과 강한 상관을 동반할 수 있다.

`langevin_mcmc_1d`는 이름에 target_logpdf를 받지만 실제 반복에는 score_fn만 사용한다. 수락·거절 단계가 없는 **ULA**다. 거절이 없다는 사실을 모든 제안이 이론적으로 정확히 수락된다는 뜻으로 해석하지 않는다. 두 방법의 분포 적합도와 계산비용은 조건별로 비교해야 하며 랑주뱅이 다봉 분포에서 항상 더 효율적인 것도 아니다.

## 3. 연속시간과 이산시간의 차이

이 페이지의 스트라이드 관례는 다음과 같다.

$$dX_t=\nabla\log p(X_t)dt+\sqrt2\,dW_t$$

밀도를 ρt라 하면 Fokker–Planck 방정식은 다음과 같다.

$$\partial_t\rho_t=-\nabla\cdot(\rho_t\nabla\log p)+\Delta\rho_t
=\nabla\cdot\left(\rho_t\nabla\log\frac{\rho_t}{p}\right)$$

보존된 출력의 마지막 비가 p/ρt인 식은 **부호가 반대**다. 적절한 경계·정칙 조건에서 ρ=p를 대입하면 시간 미분이 0이다. 불변분포임을 확인하는 것과 임의 초기화에서 유일하게 그 분포로 수렴한다는 것은 다르며, 후자에는 에르고딕성 등 추가 조건이 필요하다.

오일러 이산화는 다음과 같다.

$$x_{n+1}=x_n+\epsilon s(x_n)+\sqrt{2\epsilon}\,\xi_n$$

앞 페이지의 ε/2·√ε 관례와는 스트라이드가 두 배 차이 난다. 표준 정규 목표에서는 AR 계수가 1−ε이고, 0<ε<2에서 정상분산은 다음과 같다.

$$\operatorname{Var}(x_\infty)=\frac{2\epsilon}{1-(1-\epsilon)^2}=\frac1{1-\epsilon/2}$$

고정 스트라이드이면 이산화 바이어스가 남는다. 작은 스트라이드와 긴 실행의 관계, 초기 상태의 영향, 표본 상관을 따로 검토해야 한다. 이산화의 연속시간 극한에는 스트라이드만 줄이는 것이 아니라 비교하는 물리적 시간과 반복 수도 맞추어야 한다.

## 4. 1차원 혼합분포 예제의 수치 보정

목표는 평균 −2와 2, 공통 분산 0.25, 혼합 비율 0.3과 0.7인 가우시안 혼합으로 설정되어 있다. 그러나 `target_logpdf`는 밀도에 10⁻¹⁰을 더한 뒤 로그를 취하고 `score_fn`도 같은 보정 분모를 사용한다.

전 실수축에서 p(x)+10⁻¹⁰은 적분이 무한하므로 **정규화 가능한 목표 밀도가 아니다**. 보정의 영향이 작은 중심 구간에서는 원래 혼합분포와 가깝게 계산할 수 있지만 전역 목표를 정확히 정의하는 방법은 아니다. 꼬리의 안정적인 로그 혼합 계산과 그에 일치하는 스코어가 필요하다. 그림은 −4~4의 격자에서만 밀도를 다시 정규화하므로 이 전역 문제를 해결하지 않는다.

각 히스토그램은 앞쪽 1,000개 상태를 제외한 한 궤적이다. 두 모드를 충분히 오갔는지와 올바른 혼합 비율을 재현하는지 확인하지 않고 모양만으로 수렴을 선언해서는 안 된다. 같은 반복 수는 목표·그래디언트 평가비용까지 같다는 뜻도 아니다.

## 5. 학습한 스코어를 사용하는 예제

`ScoreNetwork`는 2→128→128→2 tanh MLP다. 2,000개 나선 데이터에 σ=0.5 노이즈를 추가해 −noise/σ를 회귀하고, Adam으로 1,000번 전체 배치를 갱신한다. 이 DSM의 모집단 최적 대상은 원래 데이터가 아니라 **노이즈로 평활화한 주변 밀도의 스코어**다. 유한 데이터와 유한 학습의 근사 오차도 남는다.

샘플링은 500개의 2차원 상태를 표준편차 3인 가우시안에서 시작해 ε=0.01로 1,000번 갱신한다. 상태·스코어·잡음은 모두 500×2이고 반환 최종 샘플도 같은 모양이다. 학습된 스코어가 정확하더라도 고정 σ의 평활화와 ULA 이산화 오차가 있다. 일반적인 벡터 네트워크가 항상 정규화 가능한 밀도의 정확한 스코어인 것도 아니다.

함수는 CPU 텐서의 `.numpy()` 변환을 사용하며 eval로 전환하지 않는다. 이 MLP에는 배치 정규화·드롭아웃이 없지만 임의의 다른 구조로 바꿀 때는 모드와 장치를 맞춰야 한다.

trajectory에는 초기 상태와 갱신 1, 101, …, 901회 시점이 저장된다. 그림은 처음 여덟 기록만 사용하고 제목을 0,100,…,700으로 붙이므로 **실제 저장 시점과 한 단계씩 어긋나며 최종 상태도 표시하지 않는다**. 최종 샘플 비교 그림과 시간별 그림을 구분해 읽는다.

## 6. 확산 모델과의 연결

노이즈 수준별 스코어로 목표를 차례로 바꾸는 어닐링된 랑주뱅 방법은 단일 평활화 수준의 탐색을 보완할 수 있다. 하지만 모든 DDPM·역시간 SDE·확률 흐름 ODE가 이 페이지의 동일한 ULA 식인 것은 아니다. 일반적인 역시간 확산에는 원래 드리프트, 시간별 확산 계수와 스코어가 함께 들어간다.

이 코드에는 다중 노이즈 조건 학습이나 DDPM 역과정이 없다. '모든 확산 모델은 시간에 따라 스코어만 바뀌는 랑주뱅 반복'이라는 출력 설명은 연결을 단순화한 것으로 읽어야 한다. 관련 배경은 [노이즈 조건 스코어 모델](https://arxiv.org/abs/1907.05600)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표준 정규 목표에서 x′=(1−ε)x+√(2ε)ξ의 안정 조건과 정상분산을 구하라.

</div>

??? success "연습문제 1 풀이"

    AR 계수의 절댓값이 1보다 작아야 하므로 0<ε<2다. 정상분산 v는 v=(1−ε)²v+2ε를 만족해 v=1/(1−ε/2)다. 목표 분산 1과의 차이는 초기 상태를 오래 버려도 남는 이산화 바이어스다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
랑주뱅 제안에 Metropolis 보정을 추가할 때 목표 밀도비만으로 충분한가?

</div>

??? success "연습문제 2 풀이"

    아니다. 제안 $q(y|x)=N(y;x+εs(x),2εI)$는 일반적으로 비대칭이다. 수락 확률은 min(1,p(y)q(x|y)/(p(x)q(y|x)))이며 정규화 상수는 약분된다. 올바른 정역 제안 밀도와 적분 가능한 목표가 필요하다. 보정 없는 현재 반복과 구분한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
학습 스코어의 오차, DSM의 평활화, ULA의 이산화 바이어스를 구분하라.

</div>

??? success "연습문제 3 풀이"

    학습 오차는 추정 스코어와 목표 노이즈 주변 스코어의 차이다. 평활화는 고정 σ>0에서 목표 분포가 원래 데이터와 달라지는 것이다. 이산화 바이어스는 정확한 목표 스코어를 써도 고정 스트라이드 ULA에 남을 수 있다. 노이즈 스케줄을 바꾼다는 말만으로 세 문제를 동시에 해결할 수는 없다.



## 정리하며

랑주뱅 샘플링에서는 정규화 상수의 불필요성과 정확한 목표 보존을 구분한다. MH도 상수 없이 밀도비를 사용하며, ULA에는 스트라이드 바이어스와 혼합 문제가 남는다. 학습 스코어의 대상, 수치 보정과 기록 시점을 확인하고 단일 수준 반복을 모든 확산 생성 과정과 동일시하지 않는다.
