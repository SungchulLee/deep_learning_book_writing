# 스코어 함수 기초

스코어 함수(score function)는 데이터 좌표에 대한 로그 확률 밀도의 그래디언트다. 밀도의 정규화 상수를 몰라도 계산할 수 있어, 분포를 학습하거나 샘플링하는 방법의 출발점이 된다. 이 페이지는 해석적으로 알려진 가우시안 스코어와 간단한 랑주뱅 반복을 살펴본다. 뉴럴 넷트웍 학습이나 노이즈 스케줄 최적화는 수행하지 않는다.

## 1. 가우시안 스코어와 시각화 코드

아래 코드는 NumPy·SciPy로 밀도와 스코어를 계산한다. 1차원 밀도·로그 밀도·스코어, 상관된 2차원 가우시안의 벡터장, 랑주뱅 궤적을 그려 파일로 저장하는 예제다. 코드와 기존 출력은 보존하며 여기서 새 그림이나 샘플을 생성하지 않는다.

??? note "코드 (516줄)"

    ```python
    """
    점수 함수 바탕
    ==================================

    어려움: 처음
    시간: 2~3시간
    미리 알아 둘 것: 01_Bayesian_Inference(특히 켤레 앞분포와 최대 사후 어림)

    학습 목표:
    -------------------
    1. 점수 함수가 무엇이며 왜 중요한지 이해한다
    2. 점수 함수를 베이즈 사후 추론과 잇는다
    3. 단순한 분포의 점수를 닫힌 꼴로 셈한다
    4. 2차원에서 점수 마당을 그려 본다
    5. 점수가 고르게 맞추는 상수를 피하는 까닭을 이해한다

    수학 바탕:
    -----------------------
    Definition:
        점수 함수 s(x)는 로그 확률의 그래디언트다.

        s(x) = ∇_x log p(x)

    베이즈 미룸과의 이음:
        베이즈 미룸에서 우리는 다음을 배웠다.
        p(θ|D) = p(D|θ)p(θ) / p(D)

        분모 p(D) = ∫ p(D|θ)p(θ) dθ은 셈할 수 없다!

        그러나 뒷분포의 점수는 다음과 같다.
        ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)] - ∇_θ log p(D)
                        = ∇_θ log[p(D|θ)p(θ)]    (constant w.r.t. θ!)

        점수는 고르개 상수가 필요 없다!

    핵심 눈썰미:
        점수를 쓰면 고르지 않은 분포를 그대로 다룰 수 있다,
        이것이 바로 베이즈 추론에 필요한 것이다!

    지은이: 이성철 @ 연세대학교
    날짜: 2025년 11월
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.stats import norm, multivariate_normal
    from matplotlib import cm

    # 그리기 결결이를 정한다
    plt.style.use('seaborn-v0_8-darkgrid')
    np.random.seed(42)

    print("=" * 80)
    print("MODULE 01: SCORE FUNCTIONS BASICS")
    print("=" * 80)

    # ============================================================================
    # 마디 1: 뜻매김과 직관
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: What is a Score Function?")
    print("=" * 80)

    print("""
    정의:
    -----------
    확률 분포 p(x)의 점수 함수는 다음과 같다.

        s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

    느낌으로 보면:
    ---------
    - 점수는 확률이 더 높은 자리를 가리킨다
    - p(x)의 봉우리에서 점수는 0이다
    - 크기 ||s(x)||는 확률이 얼마나 가파르게 바뀌는지 알려 준다
    - 점수에는 고르게 맞추는 상수가 필요 없다!

    베이즈 미룸과의 이음:
    --------------------------------
    01_Bayesian_Inference 단원에서 배운 것을 떠올려라.
    - Posterior: p(θ|D) ∝ p(D|θ)p(θ)
    - We couldn't compute p(D) = ∫ p(D|θ)p(θ) dθ

    점수를 쓰면:
    - ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]
    - ∫이 필요 없다! 점수는 고르게 맞추지 않은 분포에서도 통한다!
    """)

    # ============================================================================
    # 마디 2: 단순한 보기 - 1차원 정규 분포
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: Score of 1D Gaussian Distribution")
    print("=" * 80)

    print("""
    Example: Gaussian N(μ, σ²)
    --------------------------
    PDF: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

    로그 확률밀도: log p(x) = -log(√(2πσ²)) - (x-μ)²/(2σ²)

    Score: s(x) = d/dx log p(x) = -(x-μ)/σ²

    눈여겨볼 점은 다음과 같다.
    1. 상수 마디 -log(√(2πσ²))는 사라진다!(상수의 미분은 0)
    2. 점수는 x에 대해 선형이다
    3. x=μ(최빈값)에서 점수는 0이다
    4. 점수는 평균 μ 쪽을 가리킨다
    """)

    # 셈으로 보여 준다
    mu, sigma = 0.0, 1.0
    x_vals = np.linspace(-4, 4, 1000)

    # 확률 밀도
    pdf_vals = norm.pdf(x_vals, mu, sigma)

    # 로그 확률 밀도
    log_pdf_vals = norm.logpdf(x_vals, mu, sigma)

    # 점수(닫힌 꼴)
    score_vals = -(x_vals - mu) / (sigma ** 2)

    # 시각화 만들기
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 그림 1: 확률 밀도
    axes[0].plot(x_vals, pdf_vals, 'b-', linewidth=2, label='p(x)')
    axes[0].fill_between(x_vals, pdf_vals, alpha=0.3)
    axes[0].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
    axes[0].set_ylabel('p(x)', fontsize=12)
    axes[0].set_title('Probability Density Function (PDF)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 그림 2: 로그 확률 밀도
    axes[1].plot(x_vals, log_pdf_vals, 'g-', linewidth=2, label='log p(x)')
    axes[1].axvline(mu, color='r', linestyle='--', label=f'μ = {mu}')
    axes[1].set_ylabel('log p(x)', fontsize=12)
    axes[1].set_title('Log Probability', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 그림 3: 점수 함수
    axes[2].plot(x_vals, score_vals, 'purple', linewidth=2, label=r's(x) = \nabla\log p(x)')
    axes[2].axhline(0, color='k', linestyle='-', alpha=0.3)
    axes[2].axvline(mu, color='r', linestyle='--', label=fr'$\mu$ = {mu} (score=0)')
    axes[2].fill_between(x_vals, score_vals, alpha=0.3, color='purple')

    # 방향을 보이는 화살표를 더한다
    for x_point in [-2, -1, 1, 2]:
        score_at_x = -(x_point - mu) / (sigma ** 2)
        axes[2].arrow(x_point, score_at_x, 0.3 * np.sign(score_at_x), 0,
                      head_width=0.3, head_length=0.15, fc='darkred', ec='darkred', linewidth=2)

    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('s(x)', fontsize=12)
    axes[2].set_title(r'Score Function $s(x) = \nabla\log p(x)$\n(Points toward mean $\mu$)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '01_score_1d_gaussian.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("\n✓ Saved: 01_score_1d_gaussian.png")
    plt.close()

    print("\nKEY OBSERVATIONS:")
    print("  1. Score is ZERO at the mode (x = μ)")
    print("  2. Score POINTS TOWARD the mode:")
    print("     - For x < μ: score is POSITIVE (points right)")
    print("     - For x > μ: score is NEGATIVE (points left)")
    print("  3. Magnitude increases with distance from mode")
    print("  4. NO normalization constant √(2πσ²) in the score!")

    # ============================================================================
    # 마디 3: 2차원 정규 분포 - 점수 마당 그려 보기
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: Score Field for 2D Gaussian")
    print("=" * 80)

    print("""
    다변량 가우스 N(μ, Σ)에서는:
    ---------------------------------
    Score: s(x) = ∇_x log p(x) = -Σ^(-1)(x - μ)

    이는 평균 μ 쪽을 가리키는 선형 함수이다!

    해석:
    - 점수 벡터는 확률 높은 자리를 가리킨다
    - 평균 μ에서 점수는 0이다
    - 함께 흩어짐 Σ이 방향과 크기를 빚는다
    """)

    # 2차원 정규 분포 파라메터
    mu_2d = np.array([0, 0])
    Sigma_2d = np.array([[1.0, 0.5], [0.5, 1.0]])  # 서로 이어짐
    Sigma_inv = np.linalg.inv(Sigma_2d)

    # 격자 생성
    x1 = np.linspace(-3, 3, 30)
    x2 = np.linspace(-3, 3, 30)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    # 확률 밀도를 셈한다
    rv = multivariate_normal(mu_2d, Sigma_2d)
    pdf_2d = rv.pdf(pos)

    # 점마다 점수를 셈한다
    # s(x) = -Σ^(-1)(x - μ)
    score_field = np.zeros((len(x2), len(x1), 2))
    for i in range(len(x2)):
        for j in range(len(x1)):
            x = np.array([X1[i, j], X2[i, j]])
            score_field[i, j] = -Sigma_inv @ (x - mu_2d)

    # 시각화 만들기
    fig = plt.figure(figsize=(16, 6))

    # 그림 1: 등고선을 갖춘 확률 밀도
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.8)
    ax1.contour(X1, X2, pdf_2d, levels=10, colors='white', alpha=0.4, linewidths=0.5)
    ax1.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$')
    plt.colorbar(contour, ax=ax1, label=r'$p(x)$')
    ax1.set_xlabel(r'$x_1$', fontsize=12)
    ax1.set_ylabel(r'$x_2$', fontsize=12)
    ax1.set_title('2D Gaussian PDF', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 그림 2: 점수 마당
    ax2 = fig.add_subplot(122)
    # 바탕: 확률 밀도 등고선
    ax2.contourf(X1, X2, pdf_2d, levels=20, cmap='viridis', alpha=0.3)
    ax2.contour(X1, X2, pdf_2d, levels=10, colors='gray', alpha=0.3, linewidths=0.5)

    # 점수 벡터
    skip = 2  # 또렷하게 하려 일부만 뽑는다
    ax2.quiver(X1[::skip, ::skip], X2[::skip, ::skip],
               score_field[::skip, ::skip, 0], score_field[::skip, ::skip, 1],
               color='red', alpha=0.8, scale=20, width=0.004)
    ax2.plot(mu_2d[0], mu_2d[1], 'r*', markersize=20, label=r'Mean $\mu$ (score=0)')
    ax2.set_xlabel(r'$x_1$', fontsize=12)
    ax2.set_ylabel(r'$x_2$', fontsize=12)
    ax2.set_title(r'Score Function $s(x)=\nabla \log p(x)$' r"\n(Points toward mean $\mu$)", 
                      fontsize=14, fontweight='bold')

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '01_score_2d_gaussian_field.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("\n✓ Saved: 01_score_2d_gaussian_field.png")
    plt.close()

    print("\nKEY OBSERVATIONS:")
    print("  1. Score vectors point toward the mean (high probability)")
    print("  2. Score is zero at the mean")
    print("  3. The covariance structure affects score directions")
    print("  4. Longer arrows = steeper probability gradient")

    # ============================================================================
    # 마디 4: 뽑기에서 점수가 중요한 까닭
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: Why Score Functions Enable Sampling")
    print("=" * 80)

    print("""
    바탕이 되는 문제(베이즈 미룸에서):
    -------------------------------------------------
    Given: Posterior p(θ|D) ∝ p(D|θ)p(θ)
    바라는 것: p(θ|D)에서 뽑은 표본
    문제: 고르개 상수 p(D)를 셈할 수 없다!

    점수 풀이:
    ------------------
    핵심 눈썰미: ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]

    점수는 p(D)가 필요 없다!

    랑주뱅 액션(3단원 미리 보기):
    ----------------------------------------
    점수 s(x) = ∇_x log p(x)를 알면 다음으로 표본을 뽑을 수 있다.

        x_{t+1} = x_t + (ε/2) * s(x_t) + √ε * z_t

    여기서 z_t ~ N(0, I)은 정규 잡음이다.

    직관:
    - 떠돎 항 (ε/2)*s(x_t)이 확률 높은 쪽으로 옮긴다
    - 확산 항 √ε*z_t이 마구잡이를 더한다
    - 둘이 함께 분포를 살핀다!

    이것이 확산 모델이 도는 방식이다!
    """)

    # 1차원 정규 분포의 단순한 랑주뱅 뽑기를 보여 준다
    print("\nDemonstration: Langevin Sampling from 1D Gaussian")
    print("-" * 80)

    # 랑주뱅 액션 파라메터
    n_steps = 1000
    epsilon = 0.1
    x_current = -3.0  # 평균에서 멀리 시작

    # 저장 공간
    trajectory = [x_current]

    # 랑주뱅 액션을 돌린다
    for step in range(n_steps):
        # 지금 자리의 점수
        score = -(x_current - mu) / (sigma ** 2)

        # 랑주뱅 고침
        x_current = x_current + (epsilon / 2) * score + np.sqrt(epsilon) * np.random.randn()
        trajectory.append(x_current)

    trajectory = np.array(trajectory)

    # 시각화한다
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 그림 1: 자취
    axes[0].plot(trajectory, 'b-', alpha=0.6, linewidth=1)
    axes[0].axhline(mu, color='r', linestyle='--', linewidth=2, label=f'Target mean μ={mu}')
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('x', fontsize=12)
    axes[0].set_title('Langevin Dynamics Trajectory\n(Using score to sample from N(0,1))', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 그림 2: 히스토그램과 참 분포
    axes[1].hist(trajectory[100:], bins=50, density=True, alpha=0.6, color='blue', 
                 label='Langevin samples')
    axes[1].plot(x_vals, pdf_vals, 'r-', linewidth=3, label='True N(0,1)')
    axes[1].set_xlabel(r'$x$', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Samples vs True Distribution\n(Langevin correctly samples from target!)', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '01_langevin_preview.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("\n✓ Saved: 01_langevin_preview.png")
    plt.close()

    print("\n✓ Langevin dynamics successfully sampled from target distribution!")
    print("  This preview shows why scores are powerful:")
    print("  - Only need score, not normalization")
    print("  - Works for any distribution")
    print("  - Foundation for diffusion models!")

    # ============================================================================
    # 마디 5: 확산 모델과의 이음(미리 보기)
    # ============================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: Preview - Connection to Diffusion Models")
    print("=" * 80)

    print("""
    큰 그림:
    ---------------

    1. 베이즈 미룸(01_Bayesian_Inference 단원):
       - p(θ|D) ∝ p(D|θ)p(θ)
       - 고르개 상수를 셈할 수 없다
       - 뽑기 방법이 필요하다

    2. 점수 함수(이 단원):
       - s(x) = ∇_x log p(x)
       - 고르게 맞추지 않아도 통한다
       - 랑주뱅 뽑기를 가능하게 한다

    3. 점수 맞추기(다음 단원):
       - 자료에서 점수를 배운다
       - 잡음 없애기와의 이음

    4. 여러 자의 점수(뒤 단원):
       - 잡음 수준마다 점수를 배운다
       - σ₁ > σ₂ > ... > σ_L

    5. 확산 모델(마지막 단원):
       - 앞으로: 잡음을 차츰 더한다
       - 뒤로: 배운 점수로 잡음을 없앤다
       - 뽑기로 만들어 낸다!

    확산을 위한 핵심 눈썰미:
    -------------------------
    잡음 섞인 자료의 잡음 지우기가 곧 베이즈 뒷분포 미룸이다!

    주어진 것: x_noisy = x_clean + σ * noise
    구할 것: p(x_clean | x_noisy)

    이 사후 분포의 점수가 바로 확산 모델이 배우는 것이다!

      ∇_x log p(x|x_noisy) ← 이것을 뉴럴 넷트웍이 배운다!
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
    1. 점수 함수: s(x) = ∇_x log p(x)
    2. 점수는 확률 높은 자리를 가리킨다
    3. 점수는 고르개 상수가 필요 없다
    4. 베이즈 사후 추론과의 이음
    5. 점수는 랑주뱅 액션으로 뽑기를 가능하게 한다
    6. 확산 모델의 바탕

    핵심 식:
    ------------
    1. 점수의 뜻매김:
       s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

    2. 가우스 점수:
       For N(μ, σ²): s(x) = -(x-μ)/σ²
       For N(μ, Σ): s(x) = -Σ^(-1)(x-μ)

    3. 뒷분포 점수(베이즈):
       ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]  (p(D)가 필요 없다!)

    4. 랑주뱅 액션(미리 보기):
       x_{t+1} = x_t + (ε/2)*s(x_t) + √ε*z_t

    만들어진 파일:
    ---------------
    1. 01_score_1d_gaussian.png - 1차원 정규 분포의 점수
    2. 01_score_2d_gaussian_field.png - 2차원 점수 마당
    3. 01_langevin_preview.png - 점수를 쓴 뽑기
    """)

    print("\n" + "=" * 80)
    print("EXERCISES")
    print("=" * 80)

    print("""
    익힘 1: 닫힌 꼴 점수 셈하기
    ---------------------------------------
    다음의 점수 함수를 셈하여라.
    a) 지수 분포: x ≥ 0에서 p(x) = λ exp(-λx)
    b) 라플라스 분포: p(x) = (1/2b) exp(-|x-μ|/b)
    c) 정규 분포 둘 섞기

    익힘 2: 점수의 성질
    ---------------------------
    다음을 밝혀라.
    a) 봉우리에서 점수는 0이다
    b) ∫ p(x) s(x) dx = 0 (점수의 평균은 0이다)
    c) 정규 분포에서 점수는 선형이다

    익힘 3: 짜기
    -------------------------
    다음의 점수 셈을 짜라.
    a) 정규 분포 셋의 2차원 섞기
    b) 점수 마당을 그려 본다
    c) 랑주뱅 액션을 돌려 뽑는다

    익힘 4: 베이즈 추론과의 이음
    -------------------------------------------
    베타-이항 켤레 짝에서(01_Bayesian_Inference):
    a) 사후 분포의 점수를 이끌어 낸다
    b) 고르개 상수가 필요 없음을 보여라
    c) 곧바로 셈한 사후 분포와 견준다

    익힘 5: 점수 맞추기 미리 보기
    ---------------------------------
    식 없이 p(x)에서 뽑은 표본만 있다면:
    a) s(x) = ∇_x log p(x)를 곧바로 셈할 수 있는가? 왜 안 되는가?
    b) 잡음 없애기를 쓴 다른 방식을 내놓는다
    c) 이것이 다음 단원의 까닭이 된다!

    도전 익힘: 봉우리 여럿인 분포
    ------------------------------------------
    봉우리가 여럿인 2차원 "바둑판" 분포를 만들어라.
    a) p(x)을 격자 위 정규 분포 섞기로 뜻매김한다
    b) 점수 마당을 셈해 그려 본다
    c) 랑주뱅 액션을 돌린다 - 모든 봉우리를 살피는가?
    d) 스텝 크기 ε을 달리하면 어떻게 되는가?
    """)

    print("\n" + "=" * 80)
    print("NEXT MODULE: 02_score_matching_theory.py")
    print("=" * 80)
    print("""
    다음 단원에서는 핵심 물음을 다룬다.

      "자료만으로 점수 함수를 어떻게 배우는가?"

    이것이 요즘 만들어 내는 모델의 바탕인 점수 맞추기로 이어진다!
    """)

    print("\n✓ Module 01 complete! Generated 3 visualizations.")
    print("  Ready for Module 02: Score Matching Theory")
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```

??? note "전체 출력 (267줄)"

    ```
    ================================================================================
    MODULE 01: SCORE FUNCTIONS BASICS
    ================================================================================

    ================================================================================
    SECTION 1: What is a Score Function?
    ================================================================================

    정의:
    -----------
    확률 분포 p(x)의 점수 함수는 다음과 같다.

        s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

    느낌으로 보면:
    ---------
    - 점수는 확률이 더 높은 자리를 가리킨다
    - p(x)의 봉우리에서 점수는 0이다
    - 크기 ||s(x)||는 확률이 얼마나 가파르게 바뀌는지 알려 준다
    - 점수에는 고르게 맞추는 상수가 필요 없다!

    베이즈 미룸과의 이음:
    --------------------------------
    01_Bayesian_Inference 단원에서 배운 것을 떠올려라.
    - Posterior: p(θ|D) ∝ p(D|θ)p(θ)
    - We couldn't compute p(D) = ∫ p(D|θ)p(θ) dθ

    점수를 쓰면:
    - ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]
    - ∫이 필요 없다! 점수는 고르게 맞추지 않은 분포에서도 통한다!


    ================================================================================
    SECTION 2: Score of 1D Gaussian Distribution
    ================================================================================

    Example: Gaussian N(μ, σ²)
    --------------------------
    PDF: p(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

    로그 확률밀도: log p(x) = -log(√(2πσ²)) - (x-μ)²/(2σ²)

    Score: s(x) = d/dx log p(x) = -(x-μ)/σ²

    눈여겨볼 점은 다음과 같다.
    1. 상수 마디 -log(√(2πσ²))는 사라진다!(상수의 미분은 0)
    2. 점수는 x에 대해 선형이다
    3. x=μ(최빈값)에서 점수는 0이다
    4. 점수는 평균 μ 쪽을 가리킨다


    ✓ Saved: 01_score_1d_gaussian.png

    KEY OBSERVATIONS:
      1. Score is ZERO at the mode (x = μ)
      2. Score POINTS TOWARD the mode:
         - For x < μ: score is POSITIVE (points right)
         - For x > μ: score is NEGATIVE (points left)
      3. Magnitude increases with distance from mode
      4. NO normalization constant √(2πσ²) in the score!

    ================================================================================
    SECTION 3: Score Field for 2D Gaussian
    ================================================================================

    다변량 가우스 N(μ, Σ)에서는:
    ---------------------------------
    Score: s(x) = ∇_x log p(x) = -Σ^(-1)(x - μ)

    이는 평균 μ 쪽을 가리키는 선형 함수이다!

    해석:
    - 점수 벡터는 확률 높은 자리를 가리킨다
    - 평균 μ에서 점수는 0이다
    - 함께 흩어짐 Σ이 방향과 크기를 빚는다


    ✓ Saved: 01_score_2d_gaussian_field.png

    KEY OBSERVATIONS:
      1. Score vectors point toward the mean (high probability)
      2. Score is zero at the mean
      3. The covariance structure affects score directions
      4. Longer arrows = steeper probability gradient

    ================================================================================
    SECTION 4: Why Score Functions Enable Sampling
    ================================================================================

    바탕이 되는 문제(베이즈 미룸에서):
    -------------------------------------------------
    Given: Posterior p(θ|D) ∝ p(D|θ)p(θ)
    바라는 것: p(θ|D)에서 뽑은 표본
    문제: 고르개 상수 p(D)를 셈할 수 없다!

    점수 풀이:
    ------------------
    핵심 눈썰미: ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]

    점수는 p(D)가 필요 없다!

    랑주뱅 액션(3단원 미리 보기):
    ----------------------------------------
    점수 s(x) = ∇_x log p(x)를 알면 다음으로 표본을 뽑을 수 있다.

        x_{t+1} = x_t + (ε/2) * s(x_t) + √ε * z_t
        
    여기서 z_t ~ N(0, I)은 정규 잡음이다.

    직관:
    - 떠돎 항 (ε/2)*s(x_t)이 확률 높은 쪽으로 옮긴다
    - 확산 항 √ε*z_t이 마구잡이를 더한다
    - 둘이 함께 분포를 살핀다!

    이것이 확산 모델이 도는 방식이다!


    Demonstration: Langevin Sampling from 1D Gaussian
    --------------------------------------------------------------------------------

    ✓ Saved: 01_langevin_preview.png

    ✓ Langevin dynamics successfully sampled from target distribution!
      This preview shows why scores are powerful:
      - Only need score, not normalization
      - Works for any distribution
      - Foundation for diffusion models!

    ================================================================================
    SECTION 5: Preview - Connection to Diffusion Models
    ================================================================================

    큰 그림:
    ---------------

    1. 베이즈 미룸(01_Bayesian_Inference 단원):
       - p(θ|D) ∝ p(D|θ)p(θ)
       - 고르개 상수를 셈할 수 없다
       - 뽑기 방법이 필요하다

    2. 점수 함수(이 단원):
       - s(x) = ∇_x log p(x)
       - 고르게 맞추지 않아도 통한다
       - 랑주뱅 뽑기를 가능하게 한다

    3. 점수 맞추기(다음 단원):
       - 자료에서 점수를 배운다
       - 잡음 없애기와의 이음

    4. 여러 자의 점수(뒤 단원):
       - 잡음 수준마다 점수를 배운다
       - σ₁ > σ₂ > ... > σ_L

    5. 확산 모델(마지막 단원):
       - 앞으로: 잡음을 차츰 더한다
       - 뒤로: 배운 점수로 잡음을 없앤다
       - 뽑기로 만들어 낸다!

    확산을 위한 핵심 눈썰미:
    -------------------------
    잡음 섞인 자료의 잡음 지우기가 곧 베이즈 뒷분포 미룸이다!

    주어진 것: x_noisy = x_clean + σ * noise
    구할 것: p(x_clean | x_noisy)

    이 사후 분포의 점수가 바로 확산 모델이 배우는 것이다!

      ∇_x log p(x|x_noisy) ← 이것을 뉴럴 넷트웍이 배운다!


    ================================================================================
    MODULE SUMMARY
    ================================================================================

    배운 것:
    ---------------
    1. 점수 함수: s(x) = ∇_x log p(x)
    2. 점수는 확률 높은 자리를 가리킨다
    3. 점수는 고르개 상수가 필요 없다
    4. 베이즈 사후 추론과의 이음
    5. 점수는 랑주뱅 액션으로 뽑기를 가능하게 한다
    6. 확산 모델의 바탕

    핵심 식:
    ------------
    1. 점수의 뜻매김:
       s(x) = ∇_x log p(x) = (1/p(x)) * ∇_x p(x)

    2. 가우스 점수:
       For N(μ, σ²): s(x) = -(x-μ)/σ²
       For N(μ, Σ): s(x) = -Σ^(-1)(x-μ)

    3. 뒷분포 점수(베이즈):
       ∇_θ log p(θ|D) = ∇_θ log[p(D|θ)p(θ)]  (p(D)가 필요 없다!)

    4. 랑주뱅 액션(미리 보기):
       x_{t+1} = x_t + (ε/2)*s(x_t) + √ε*z_t

    만들어진 파일:
    ---------------
    1. 01_score_1d_gaussian.png - 1차원 정규 분포의 점수
    2. 01_score_2d_gaussian_field.png - 2차원 점수 마당
    3. 01_langevin_preview.png - 점수를 쓴 뽑기


    ================================================================================
    EXERCISES
    ================================================================================

    익힘 1: 닫힌 꼴 점수 셈하기
    ---------------------------------------
    다음의 점수 함수를 셈하여라.
    a) 지수 분포: x ≥ 0에서 p(x) = λ exp(-λx)
    b) 라플라스 분포: p(x) = (1/2b) exp(-|x-μ|/b)
    c) 정규 분포 둘 섞기

    익힘 2: 점수의 성질
    ---------------------------
    다음을 밝혀라.
    a) 봉우리에서 점수는 0이다
    b) ∫ p(x) s(x) dx = 0 (점수의 평균은 0이다)
    c) 정규 분포에서 점수는 선형이다

    익힘 3: 짜기
    -------------------------
    다음의 점수 셈을 짜라.
    a) 정규 분포 셋의 2차원 섞기
    b) 점수 마당을 그려 본다
    c) 랑주뱅 액션을 돌려 뽑는다

    익힘 4: 베이즈 추론과의 이음
    -------------------------------------------
    베타-이항 켤레 짝에서(01_Bayesian_Inference):
    a) 사후 분포의 점수를 이끌어 낸다
    b) 고르개 상수가 필요 없음을 보여라
    c) 곧바로 셈한 사후 분포와 견준다

    익힘 5: 점수 맞추기 미리 보기
    ---------------------------------
    식 없이 p(x)에서 뽑은 표본만 있다면:
    a) s(x) = ∇_x log p(x)를 곧바로 셈할 수 있는가? 왜 안 되는가?
    b) 잡음 없애기를 쓴 다른 방식을 내놓는다
    c) 이것이 다음 단원의 까닭이 된다!

    도전 익힘: 봉우리 여럿인 분포
    ------------------------------------------
    봉우리가 여럿인 2차원 "바둑판" 분포를 만들어라.
    a) p(x)을 격자 위 정규 분포 섞기로 뜻매김한다
    b) 점수 마당을 셈해 그려 본다
    c) 랑주뱅 액션을 돌린다 - 모든 봉우리를 살피는가?
    d) 스텝 크기 ε을 달리하면 어떻게 되는가?


    ================================================================================
    NEXT MODULE: 02_score_matching_theory.py
    ================================================================================

    다음 단원에서는 핵심 물음을 다룬다.

      "자료만으로 점수 함수를 어떻게 배우는가?"

    이것이 요즘 만들어 내는 모델의 바탕인 점수 맞추기로 이어진다!


    ✓ Module 01 complete! Generated 3 visualizations.
      Ready for Module 02: Score Matching Theory
    ================================================================================
    ```

## 2. 무엇에 대해 미분하는가?

양수이고 미분 가능한 밀도 p(x)에 대해 정의한다.

$$s(x)=\nabla_x\log p(x)=\frac{\nabla_xp(x)}{p(x)}$$

여기서 미분 변수는 **데이터 x**다. 통계학에서 파라메터 θ에 대한 $\nabla_\theta\log p_\theta(x)$도 스코어라고 부르므로 어느 변수를 미분하는지 구분한다. p(x)=q(x)/Z에서 Z가 x와 무관하면 $s(x)=\nabla_x\log q(x)$다. Z가 미분 변수에 의존하는 문제까지 무조건 생략할 수 있는 것은 아니다.

스코어는 로그 밀도를 국소적으로 가장 빠르게 증가시키는 방향이다. 길이는 밀도 자체가 아니라 **로그 밀도의 변화율**을 나타낸다. 작은 이동의 일차 근사에 관한 설명이며, 큰 스트라이드로 이동하면 실제 밀도가 줄 수도 있다. 양의 밀도를 가진 내부의 미분 가능한 극값에서 스코어가 0이지만 극소점이나 안장점에서도 0일 수 있다. 경계의 최빈값이나 미분 불가능한 점에 같은 조건을 적용하지 않는다.

## 3. 1차원과 다변량 가우시안

분산 σ²>0인 가우시안에서는 다음과 같다.

$$p(x)=\frac1{\sqrt{2\pi\sigma^2}}\exp\!\left[-\frac{(x-\mu)^2}{2\sigma^2}\right],\qquad
s(x)=-\frac{x-\mu}{\sigma^2}$$

μ에서 0이고 평균의 반대편에서 각각 부호가 바뀐다. 같은 거리라도 분산이 작을수록 스코어의 크기가 크다. 1차원 그림에서는 밀도의 봉우리, 로그 밀도의 곡률과 스코어의 선형 그래디언트를 대응해 읽는다.

양의 정부호 공분산 Σ인 다변량 가우시안에서는 다음과 같다.

$$s(x)=-\Sigma^{-1}(x-\mu)$$

Σ가 등방적이지 않으면 벡터가 평균으로 향하는 직선 방향과 평행하지 않을 수 있다. 다만 $(x-\mu)^Ts(x)=-(x-\mu)^T\Sigma^{-1}(x-\mu)<0$이므로 평균 쪽 성분을 갖는다. 2차원 그림의 화살표는 로그 밀도 등고선에 수직인 방향과 공분산에 따른 비등방성을 보여 준다. 길이를 단순히 확률 밀도 그래디언트의 크기라고 부르는 출력 설명은 두 그래디언트의 차이를 생략한 표현이다.

## 4. 랑주뱅 반복과 이산화 바이어스

코드는 다음과 같은 보정 없는 랑주뱅 반복을 1,000회 수행한다.

$$x_{n+1}=x_n+\frac\epsilon2s(x_n)+\sqrt\epsilon\,\xi_n,\qquad \xi_n\sim N(0,I)$$

스코어에 의한 이동과 잡음을 결합하는 방식이다. 적절한 정칙·적분 가능성 조건에서 연속시간 랑주뱅 확산은 목표 밀도를 불변분포로 가질 수 있지만, 고정 ε의 이산 반복이 목표를 정확히 보존한다는 뜻은 아니다. 안정성과 혼합도 분포·스트라이드에 의존한다.

표준 정규분포에서 반복은 $x_{n+1}=(1-\epsilon/2)x_n+\sqrt\epsilon\xi_n$이다. 0<ε<4이면 안정적인 AR(1) 과정이며 정상분산은 다음과 같다.

$$\operatorname{Var}(x_\infty)=\frac{\epsilon}{1-(1-\epsilon/2)^2}=\frac1{1-\epsilon/4}$$

코드의 ε=0.1에서는 약 1.0256으로 목표 분산 1과 다르다. 앞쪽 100개 상태를 버려도 이산화 바이어스나 연속 표본 간 상관은 사라지지 않는다. 보존된 그림 제목과 출력의 '정확한 샘플링', '어떤 분포에서도 작동'은 이러한 조건을 생략한 과장이다. 다봉 분포에서는 모드 사이 이동도 느릴 수 있다.

## 5. 베이즈 추론과 확산 모델의 연결

베이즈 사후분포에서 상태 변수를 θ로 보면, 증거 p(D)가 θ와 무관하므로 다음과 같다.

$$\nabla_\theta\log p(\theta\mid D)=\nabla_\theta[\log p(D\mid\theta)+\log p(\theta)]$$

여기서는 θ가 샘플링할 **상태 좌표**라는 점에 주의한다. 다른 하이퍼파라미터에 대해 증거를 미분하는 문제와 다르다.

가우시안 노이즈를 더한 $\widetilde x=x_0+\sigma\xi$의 주변 밀도를 pσ라 하면, 잡음 조건 스코어 모델은 일반적으로 $\nabla_{\widetilde x}\log p_\sigma(\widetilde x)$를 학습한다. 출력에 적힌 '깨끗한 x에 대한 사후분포의 스코어를 그대로 학습한다'는 설명과 다르다. 적절한 조건에서 두 대상은 사후평균 관계로 연결된다.

$$\nabla_{\widetilde x}\log p_\sigma(\widetilde x)
=\frac{\mathbb E[x_0\mid\widetilde x]-\widetilde x}{\sigma^2}$$

단일한 고정 분포에 대한 랑주뱅 반복과, 시간·노이즈 수준별 주변 스코어를 이용하는 확산 생성 과정도 구분한다. 이 예제는 그 출발점이지 완성된 확산 모델 학습·생성 구현은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
정규화하지 않은 밀도 q(x)=exp(−(x−μ)²/(2σ²))에서 스코어를 구하라. 다변량에서는 무엇이 달라지는가?

</div>

??? success "연습문제 1 풀이"

    로그를 취한 뒤 x로 미분하면 −(x−μ)/σ²다. x와 무관한 정규화 상수의 미분은 0이다. 다변량 양의 정부호 Σ에서는 −Σ⁻¹(x−μ)이며 일반적으로 평균으로 향하는 직선과 평행하지는 않다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
스코어의 평균이 항상 0인가? 지수분포를 예로 경계 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"

    적분과 미분이 적절하고 경계항이 사라지는 정칙 밀도에서는 ∫p(x)s(x)dx=∫∇p(x)dx=0이다. 하지만 x>0에서 지수분포 λexp(−λx)의 데이터 스코어는 −λ라 평균도 −λ다. 0에서의 경계항이 사라지지 않는다. 따라서 코드 출력의 평균 0 주장을 경계가 있는 모든 분포에 적용하면 안 된다. 라플라스 분포도 최빈값에서 미분 가능성을 따로 다룬다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
표준 정규 목표에서 ε=0.1인 랑주뱅 반복의 정상분산을 구하고 워밍업과의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"

    AR 계수는 0.95이고 잡음 분산은 0.1이므로 정상분산은 0.1/(1−0.95²)≈1.02564다. 초기 상태의 영향을 줄이는 워밍업과 이산화 바이어스는 다른 문제다. 충분히 오래 반복해도 고정 스트라이드의 정상분산은 1이 아니며 연속 상태 사이의 상관도 남는다. 이 페이지에는 뉴럴 넷트웍 로스나 노이즈 스케줄 비교 실험이 없다.



## 정리하며

스코어는 데이터 좌표에 대한 로그 밀도의 그래디언트이며 좌표와 무관한 정규화 상수를 제거할 수 있다. 가우시안 스코어로 방향과 스케일을 확인하되, 경계 조건과 고정 스트라이드 랑주뱅의 바이어스를 구분한다. 확산 모델이 배우는 잡음 주변 스코어는 깨끗한 데이터의 사후 스코어와 같은 대상이 아니다.
