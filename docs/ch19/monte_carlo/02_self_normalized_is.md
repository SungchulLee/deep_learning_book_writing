# 자기정규화 중요도 샘플링

미정규화 밀도에서 SNIS 평균을 계산하고 정규 평균 모형의 해석적 사후분포와 비교한다. 가중치 집중도와 반복 추정 오차를 구분하는 예제다.

## 1. 코드

??? note "코드 (505줄)"

    ```python
    """
    02_self_normalized_IS.py

    첫걸음 단계: 스스로 고르게 하는 중요도 표집

    이 단원은 스스로 고르게 하는 중요도 표집을 소개한다. 이는
    이는 뒤확률의 고르게 하는 상수를 모르는 베이즈 추론에 꼭 필요하다.

    수학적 바탕:
    --------------------
    문제: 우리는 상수배를 빼고만 π(θ)을 안다:
        π(θ) = γ(θ)/Z, 여기서 Z = ∫γ(θ)dθ은 알 수 없다

    베이즈 추론에서:
        γ(θ) = p(y|θ)p(θ)  (가능도 × 앞확률)
        Z = p(y) = ∫p(y|θ)p(θ)dθ  (주변 가능도, 흔히 다룰 수 없다)

    풀이: 스스로 고르게 하는 중요도 표집(SNIS)

    고르게 하지 않은 무게: w̃ᵢ = γ(θᵢ)/q(θᵢ)

    스스로 고르게 한 어림꼴:
        Ê[h(θ)] = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]
                 = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]

    성질:
    - 치우쳤지만 한결같다
    - 치우침 = O(1/n)
    - 흔히 고르게 한 중요도 표집보다 흩어짐이 작다
    - Z을 알 필요가 없다

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    import os

    np.random.seed(42)
    sns.set_style("whitegrid")


    def self_normalized_importance_sampling(unnormalized_target, proposal_dist, 
                                            h_function, n_samples):
        """
        고르게 하지 않은 과녁 분포를 위한, 스스로 고르게 하는 중요도 표집.

        매개변수:
        -----------
        unnormalized_target : callable
            π(θ) = γ(θ)/Z일 때의 함수 γ(θ)
            베이즈에서는: γ(θ) = p(y|θ)p(θ)
        proposal_dist : scipy.stats distribution
            제안 분포 q(θ)
        h_function : callable
            기댓값을 구하려는 함수
        n_samples : int
            표본의 개수

        반환값:
        --------
        estimate : float
            스스로 고르게 한 중요도 표집 어림값
        samples : array
            제안에서 뽑은 표본
        normalized_weights : array
            고르게 한 중요도 무게
        unnormalized_weights : array
            고르게 하지 않은 중요도 무게

        알고리즘:
        ---------
        1. i=1,...,n에 대해 θᵢ ~ q(θ) 표집
        2. 고르게 하지 않은 무게 셈하기: w̃ᵢ = γ(θᵢ)/q(θᵢ)
        3. 무게 고르게 하기: wᵢ = w̃ᵢ / Σⱼw̃ⱼ
        4. 어림하기: Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
        """
        # 걸음 1: 제안에서 표본 뽑기
        samples = proposal_dist.rvs(size=n_samples)

        # 걸음 2: 고르게 하지 않은 과녁 γ(θ) 값 매기기
        gamma_values = unnormalized_target(samples)

        # 걸음 3: 제안 밀도 q(θ) 값 매기기
        q_values = proposal_dist.pdf(samples)

        # 걸음 4: 고르게 하지 않은 무게 w̃ᵢ = γ(θᵢ)/q(θᵢ) 셈하기
        unnormalized_weights = gamma_values / (q_values + 1e-300)

        # 걸음 5: 무게 고르게 하기
        # wᵢ = w̃ᵢ / Σⱼw̃ⱼ
        # 무게를 그대로 더하면, 과녁 값이 모두 0에 가깝게 가라앉았을 때
        # 합이 0이 되어 0/0 = NaN 이 된다. 가장 큰 무게로 먼저 나누면
        # 비율이 그대로 유지되면서 이 문제가 사라진다.
        max_weight = np.max(unnormalized_weights)
        if max_weight > 0:
            scaled = unnormalized_weights / max_weight
            normalized_weights = scaled / np.sum(scaled)
        else:
            # 쓸 만한 무게가 하나도 없다. 고르게 나누고 진단에서 걸러지게 둔다.
            normalized_weights = np.full_like(
                unnormalized_weights, 1.0 / len(unnormalized_weights))
        weight_sum = np.sum(unnormalized_weights)

        # 걸음 6: 표본 점에서 함수 h 값 매기기
        h_values = h_function(samples)

        # 걸음 7: 스스로 고르게 한 어림값 셈하기
        # Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
        estimate = np.sum(normalized_weights * h_values)

        return estimate, samples, normalized_weights, unnormalized_weights


    def compute_ess(normalized_weights):
        """
        실효 표본 크기(ESS) 셈하기.

        ESS = 1 / Σᵢwᵢ²

        고르게 하지 않은 무게를 쓰는 다른 공식:
        ESS = (Σᵢw̃ᵢ)² / Σᵢw̃ᵢ²

        해석:
        - ESS ≈ n: 표본의 무게가 엇비슷하다(좋음)
        - ESS << n: 몇몇 표본이 판친다(나쁨)
        - ESS / n은 중요도 표집의 "효율"이다
        """
        ess = 1.0 / np.sum(normalized_weights**2)
        return ess


    # 보기 1: 고르게 하는 상수를 모르는 단순 가우스
    # ==========================================================
    print("=" * 70)
    print("EXAMPLE 1: Self-Normalized IS for π(θ) = γ(θ)/Z")
    print("=" * 70)

    # 고르게 하지 않은 과녁 정하기: γ(θ) = exp(-0.5(θ-3)²)
    # 이는 N(3, 1)에 비례하지만 고르게 하는 상수가 없다
    def gamma_function(theta):
        """
        고르게 하지 않은 가우스: γ(θ) = exp(-0.5(θ-μ)²/σ²)
        1/√(2πσ²) 인수가 빠졌다
        """
        mu, sigma = 3.0, 1.0
        return np.exp(-0.5 * ((theta - mu) / sigma)**2)

    # 참으로 고르게 한 분포(확인용)
    target_dist = stats.norm(3, 1)

    # 제안 분포
    proposal_dist = stats.norm(0, 2)

    # 어림할 함수: h(θ) = θ²
    h_function = lambda theta: theta**2

    # 참 기댓값
    true_expectation = 3**2 + 1**2  # N(3,1)의 E[θ²]

    print(f"\nTrue E[θ²]: {true_expectation:.6f}")

    # 스스로 고르게 하는 중요도 표집 돌리기
    n_samples = 1000
    estimate, samples, norm_weights, unnorm_weights = self_normalized_importance_sampling(
        gamma_function, proposal_dist, h_function, n_samples
    )

    ess = compute_ess(norm_weights)
    efficiency = ess / n_samples * 100

    print(f"\nSelf-Normalized IS Results (n={n_samples}):")
    print(f"  Estimate: {estimate:.6f}")
    print(f"  Error: {abs(estimate - true_expectation):.6f}")
    print(f"  ESS: {ess:.1f}")
    print(f"  Efficiency: {efficiency:.1f}%")

    # 시각화한다
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 칸 1: 고르게 하지 않은 과녁과 고르게 한 과녁
    x = np.linspace(-5, 8, 1000)
    ax = axes[0, 0]
    ax.plot(x, gamma_function(x), 'b-', linewidth=2, 
            label='Unnormalized γ(θ)')
    ax.plot(x, target_dist.pdf(x), 'r--', linewidth=2, 
            label='Normalized π(θ)')
    ax.plot(x, proposal_dist.pdf(x), 'g:', linewidth=2, 
            label='Proposal q(θ)')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density (arbitrary scale)', fontsize=12)
    ax.set_title('Unnormalized Target vs Normalized', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 2: 무게 분포
    ax = axes[0, 1]
    ax.hist(norm_weights, bins=50, density=True, alpha=0.7, 
            color='purple', edgecolor='black')
    ax.set_xlabel('Normalized Weight wᵢ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of Normalized Weights\nESS = {ess:.1f} ({efficiency:.1f}%)', 
                 fontsize=13, fontweight='bold')
    uniform_weight = 1.0 / n_samples
    ax.axvline(uniform_weight, color='red', linestyle='--', linewidth=2,
               label=f'Uniform = {uniform_weight:.4f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 3: 쌓인 무게 분포
    sorted_weights = np.sort(norm_weights)[::-1]  # 내림차순 정렬
    cumulative_weights = np.cumsum(sorted_weights)
    ax = axes[1, 0]
    ax.plot(np.arange(1, len(sorted_weights)+1), cumulative_weights, 
            'b-', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, 
               label='50% of total weight')
    ax.axhline(0.9, color='orange', linestyle='--', linewidth=2,
               label='90% of total weight')
    ax.set_xlabel('Number of Samples (sorted by weight)', fontsize=12)
    ax.set_ylabel('Cumulative Weight', fontsize=12)
    ax.set_title('Cumulative Weight Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 무게의 50%과 90%을 차지하는 표본 수 찾기
    n_50 = np.searchsorted(cumulative_weights, 0.5) + 1
    n_90 = np.searchsorted(cumulative_weights, 0.9) + 1
    ax.text(0.05, 0.95, f'{n_50} samples = 50% weight\n{n_90} samples = 90% weight',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 칸 4: 무게로 색칠한 표본
    ax = axes[1, 1]
    scatter = ax.scatter(samples, h_function(samples), c=norm_weights, 
                         cmap='hot', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Sample θ', fontsize=12)
    ax.set_ylabel('h(θ) = θ²', fontsize=12)
    ax.set_title('Samples Colored by Weight', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Normalized Weight')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__),"example1_self_normalized.png")
    plt.savefig(fig_path, 
                dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: example1_self_normalized.png")


    # 보기 2: 베이즈 추론 - 흩어짐을 모르는 정규 평균
    # ================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Bayesian Inference for Normal Mean")
    print("=" * 70)

    # 자료: σ = 1이 알려진 y ~ N(θ, σ²)
    # 앞확률: θ ~ N(μ₀, τ²)
    # 뒤확률: θ|y ~ N(μₙ, τₙ²), 여기서
    #   τₙ² = 1/(1/τ² + n/σ²)
    #   μₙ = τₙ²(μ₀/τ² + Σyᵢ/σ²)

    # 합성 데이터 생성
    true_theta = 5.0
    sigma = 1.0
    n_obs = 20
    data = np.random.normal(true_theta, sigma, n_obs)

    print(f"\nData: n={n_obs}, sample mean={np.mean(data):.3f}")

    # 앞확률 매개변수
    mu_0 = 0.0
    tau = 2.0

    # 뒤확률 매개변수(손으로 구함, 확인용)
    tau_n_sq = 1.0 / (1.0/tau**2 + n_obs/sigma**2)
    mu_n = tau_n_sq * (mu_0/tau**2 + np.sum(data)/sigma**2)

    posterior_dist = stats.norm(mu_n, np.sqrt(tau_n_sq))

    print(f"\nPosterior (analytical): N({mu_n:.3f}, {np.sqrt(tau_n_sq):.3f})")

    # 고르게 하지 않은 뒤확률 정하기: γ(θ) = p(y|θ)p(θ)
    def unnormalized_posterior(theta):
        """
        γ(θ) = p(y|θ)p(θ)
             = ∏ᵢ N(yᵢ|θ,σ²) × N(θ|μ₀,τ²)
             ∝ exp(-Σ(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ²)
        """
        # 로그 가능도: log p(y|θ)
        log_likelihood = -0.5 * np.sum((data[:, None] - theta)**2) / sigma**2

        # 로그 앞확률: log p(θ)
        log_prior = -0.5 * (theta - mu_0)**2 / tau**2

        # 고르게 하지 않은 뒤확률 돌려주기(수치 안정을 위해 로그 공간에서)
        return np.exp(log_likelihood + log_prior)

    # 앞확률을 제안으로 쓰기(단순한 고름)
    proposal_prior = stats.norm(mu_0, tau)

    # 뒤확률 평균 어림하기: E[θ|y]
    h_identity = lambda theta: theta
    n_samples = 5000

    estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
        unnormalized_posterior, proposal_prior, h_identity, n_samples
    )

    ess = compute_ess(norm_weights)

    # 참 뒤확률 평균
    true_post_mean = mu_n

    print(f"\nPosterior Mean E[θ|y]:")
    print(f"  True value: {true_post_mean:.6f}")
    print(f"  SNIS estimate: {estimate:.6f}")
    print(f"  Error: {abs(estimate - true_post_mean):.6f}")
    print(f"  ESS: {ess:.1f} ({ess/n_samples*100:.1f}%)")

    # 뒤확률 흩어짐 어림하기: Var[θ|y]
    h_centered_square = lambda theta: (theta - estimate)**2
    var_estimate, _, _, _ = self_normalized_importance_sampling(
        unnormalized_posterior, proposal_prior, h_centered_square, n_samples
    )

    true_post_var = tau_n_sq

    print(f"\nPosterior Variance Var[θ|y]:")
    print(f"  True value: {true_post_var:.6f}")
    print(f"  SNIS estimate: {var_estimate:.6f}")
    print(f"  Error: {abs(var_estimate - true_post_var):.6f}")


    # 보기 3: 제안 분포 견주기
    # =========================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Effect of Proposal Choice on ESS")
    print("=" * 70)

    # 보기 1과 같은 차림
    proposals = {
        'Prior N(0,2)': stats.norm(0, 2),
        'Close to posterior N(5,1.5)': stats.norm(5, 1.5),
        'Posterior (oracle) N(μₙ,τₙ)': posterior_dist,
        'Too narrow N(5,0.5)': stats.norm(5, 0.5),
        'Too wide N(0,4)': stats.norm(0, 4),
    }

    n_samples = 2000
    print(f"\nComparing proposals (n={n_samples}):")
    print("-" * 70)

    results = []
    for name, proposal in proposals.items():
        estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
            unnormalized_posterior, proposal, h_identity, n_samples
        )
        ess = compute_ess(norm_weights)
        efficiency = ess / n_samples * 100
        error = abs(estimate - true_post_mean)

        results.append({
            'name': name,
            'estimate': estimate,
            'ess': ess,
            'efficiency': efficiency,
            'error': error
        })

        print(f"{name:30s}: ESS={ess:6.1f} ({efficiency:5.1f}%), Error={error:.4f}")

    # ESS 견줌 그려 보기
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    colors = ['blue' if 'oracle' not in n.lower() else 'red' for n in names]

    bars = ax.bar(range(len(names)), efficiencies, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=12)
    ax.set_title('Proposal Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label='Perfect efficiency')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__),"example3_proposal_comparison.png")
    plt.savefig(fig_path,
                dpi=300, bbox_inches='tight')


    # 보기 4: 치우침과 표본 크기
    # ==============================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Bias of Self-Normalized IS")
    print("=" * 70)

    # 스스로 고르게 하는 중요도 표집은 치우쳤지만 한결같다
    # 치우침 = O(1/n)

    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    n_replications = 200

    print(f"\nInvestigating bias ({n_replications} replications):")
    print("-" * 70)

    biases = []
    std_errors = []

    for n in sample_sizes:
        estimates = []
        for _ in range(n_replications):
            est, _, _, _ = self_normalized_importance_sampling(
                unnormalized_posterior, proposal_prior, h_identity, n
            )
            estimates.append(est)

        mean_estimate = np.mean(estimates)
        bias = mean_estimate - true_post_mean
        std_error = np.std(estimates)

        biases.append(bias)
        std_errors.append(std_error)

        print(f"n={n:5d}: Bias={bias:+.6f}, Std Error={std_error:.6f}")

    # 표본 크기에 따른 치우침 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(sample_sizes, biases, 'bo-', linewidth=2, markersize=8, label='Observed bias')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Bias', fontsize=12)
    ax.set_title('Bias vs Sample Size (Self-Normalized IS)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sample_sizes, std_errors, 'go-', linewidth=2, markersize=8, 
            label='Standard error')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.set_title('Standard Error vs Sample Size', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # 기준선 더하기: 표준 오차 ~ 1/√n
    ax.plot(sample_sizes, 0.5/np.sqrt(sample_sizes), 'r--', linewidth=2, 
            label='O(1/√n) reference')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__),"example4_bias_analysis.png")
    plt.savefig(fig_path,
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 스스로 고르게 하는 중요도 표집은 고르게 하지 않은 과녁 분포를 다루며,
       그래서 p(y)을 모르는 베이즈 추론에 딱 맞다.

    2. 스스로 고르게 한 어림꼴은 다음과 같다:
       wᵢ = w̃ᵢ/Σⱼw̃ⱼ일 때 Ê[h(θ)] = Σᵢ wᵢh(θᵢ)

    3. 성질:
       - 치우쳤지만 한결같다(n → ∞이면 치우침 → 0)
       - 치우침 = O(1/n)
       - 흔히 고르게 한 중요도 표집보다 흩어짐이 작다

    4. 실효 표본 크기(ESS)는 제안의 질을 잰다:
       - ESS = 1/Σᵢwᵢ²
       - ESS ≈ n: 아주 좋은 제안
       - ESS << n: 나쁜 제안, 몇몇 표본이 판친다
       - 효율 = ESS/n × 100%

    5. 베이즈 추론에서:
       - 고르게 하지 않은 뒤확률: γ(θ) = p(y|θ)p(θ)
       - 앞확률은 단순한 제안 선택이 된다
       - 더 나은 제안(이를테면 라플라스 어림)이 ESS을 높인다

    6. 좋은 제안이 결정적이다:
       - 뒤확률과 잘 겹쳐야 한다
       - 뒤확률보다 꼬리가 두꺼워야 한다
       - 주고받음: 셈 값과 ESS 나아짐

    7. 무게 진단이 꼭 필요하다:
       - ESS 살피기
       - 무게 분포 살펴보기
       - 판치는 표본 살펴보기(무게 몰림)
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (95줄)"

    ```
    ======================================================================
    EXAMPLE 1: Self-Normalized IS for π(θ) = γ(θ)/Z
    ======================================================================

    True E[θ²]: 10.000000

    Self-Normalized IS Results (n=1000):
      Estimate: 10.335240
      Error: 0.335240
      ESS: 177.3
      Efficiency: 17.7%

    Visualization saved to: example1_self_normalized.png

    ======================================================================
    EXAMPLE 2: Bayesian Inference for Normal Mean
    ======================================================================

    Data: n=20, sample mean=5.434

    Posterior (analytical): N(5.367, 0.222)

    Posterior Mean E[θ|y]:
      True value: 5.366742
      SNIS estimate: -0.018585
      Error: 5.385328
      ESS: 5000.0 (100.0%)

    Posterior Variance Var[θ|y]:
      True value: 0.049383
      SNIS estimate: 4.020022
      Error: 3.970639

    ======================================================================
    EXAMPLE 3: Effect of Proposal Choice on ESS
    ======================================================================

    Comparing proposals (n=2000):
    ----------------------------------------------------------------------
    Prior N(0,2)                  : ESS=2000.0 (100.0%), Error=5.3755
    Close to posterior N(5,1.5)   : ESS=2000.0 (100.0%), Error=0.2897
    Posterior (oracle) N(μₙ,τₙ)   : ESS=2000.0 (100.0%), Error=0.0024
    Too narrow N(5,0.5)           : ESS=2000.0 (100.0%), Error=0.3596
    Too wide N(0,4)               : ESS=2000.0 (100.0%), Error=5.3654

    ======================================================================
    EXAMPLE 4: Bias of Self-Normalized IS
    ======================================================================

    Investigating bias (200 replications):
    ----------------------------------------------------------------------
    n=   10: Bias=-5.393068, Std Error=0.613604
    n=   50: Bias=-5.387002, Std Error=0.274807
    n=  100: Bias=-5.362901, Std Error=0.203625
    n=  500: Bias=-5.361557, Std Error=0.087201
    n= 1000: Bias=-5.372228, Std Error=0.061818
    n= 5000: Bias=-5.371045, Std Error=0.028169
    n=10000: Bias=-5.366182, Std Error=0.019181

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 스스로 고르게 하는 중요도 표집은 고르게 하지 않은 과녁 분포를 다루며,
       그래서 p(y)을 모르는 베이즈 추론에 딱 맞다.

    2. 스스로 고르게 한 어림꼴은 다음과 같다:
       wᵢ = w̃ᵢ/Σⱼw̃ⱼ일 때 Ê[h(θ)] = Σᵢ wᵢh(θᵢ)

    3. 성질:
       - 치우쳤지만 한결같다(n → ∞이면 치우침 → 0)
       - 치우침 = O(1/n)
       - 흔히 고르게 한 중요도 표집보다 흩어짐이 작다

    4. 실효 표본 크기(ESS)는 제안의 질을 잰다:
       - ESS = 1/Σᵢwᵢ²
       - ESS ≈ n: 아주 좋은 제안
       - ESS << n: 나쁜 제안, 몇몇 표본이 판친다
       - 효율 = ESS/n × 100%

    5. 베이즈 추론에서:
       - 고르게 하지 않은 뒤확률: γ(θ) = p(y|θ)p(θ)
       - 앞확률은 단순한 제안 선택이 된다
       - 더 나은 제안(이를테면 라플라스 어림)이 ESS을 높인다

    6. 좋은 제안이 결정적이다:
       - 뒤확률과 잘 겹쳐야 한다
       - 뒤확률보다 꼬리가 두꺼워야 한다
       - 주고받음: 셈 값과 ESS 나아짐

    7. 무게 진단이 꼭 필요하다:
       - ESS 살피기
       - 무게 분포 살펴보기
       - 판치는 표본 살펴보기(무게 몰림)

    ```


## 2. 논의

### 추정량과 수치 안정성

표본은 q에서 생성하고 원시 가중치 γ/q를 합으로 나누어 함수값을 평균한다. 목표 π=γ/Z에서 적분 가능성과 지지집합 조건이 맞으면 일치성을 얻지만 일반적으로 유한 표본에서는 편향된다. O(1/n) 편향 전개와 중심극한정리에는 추가 적률 조건이 필요하며, 표준 IS보다 항상 분산이 작은 것은 아니다.

첫 목표는 N(3,1²)이므로 E[θ²]=10이다. 코드는 밀도비를 계산한 뒤 최대값으로 나눈다. 이 단계는 이미 발생한 exp의 언더플로·오버플로를 복구하지 못한다. q에 10⁻³⁰⁰을 더하면 정확한 밀도비도 변하며, 전부 0인 가중치를 균등하게 바꾸는 분기는 실패를 감춘다. 로그밀도에서 log-sum-exp로 정규화하고 유효하지 않은 입력을 따로 진단해야 한다.

### 정규 평균 예제의 축 오류

알려진 관측분산 σ²에서 사후분산과 평균은 다음과 같다.

$$\tau_n^2=(\tau^{-2}+n\sigma^{-2})^{-1},\qquad \mu_n=\tau_n^2(\mu_0/\tau^2+\sum_i y_i/\sigma^2).$$

그러나 unnormalized_posterior는 관측×제안 표본 배열의 제곱 오차를 축 지정 없이 모두 합한다. 따라서 각 θ별 우도를 반환하지 않고 모든 제안에 공통인 스칼라를 만든다. 의도한 계산은 관측축만 합하는 것이며, 현재 정규 모형의 SNIS 비교·편향 실험을 이론 검증으로 해석할 수 없다.

별도 표본에서 이전 평균 추정치를 중심으로 계산한 이차 모멘트는 그 중심 오차의 제곱도 포함한다. 또한 σ=1, n=20, τ=2에서 사후 표준편차는 약 0.222다. 이름이 Too narrow인 표준편차 0.5 제안은 실제 사후보다 넓으므로 라벨만으로 제안을 판단하지 않는다.

### 진단의 의미

ESS=1/∑w̄²는 관측한 가중치 집중도다. 같은 표본으로 여러 관심 함수를 계산할 수 있지만 함수마다 오차는 다르다. 반복 추정량의 표준편차는 한 실행의 몬테카를로 표준오차를 평가하며 반복 평균으로 추정한 편향에도 표본 오차가 있다. 꼬리가 목표보다 반드시 두꺼워야 한다는 문구는 유한 이차 적률 조건을 대신하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SNIS에서 미지 정규화 상수가 사라지는 이유는?

</div>

??? success "연습문제 1 풀이"

    γ를 상수 c>0배 하면 모든 원시 가중치가 c배 되고 합으로 나누는 과정에서 소거된다. 다만 원시 가중치 평균을 주변우도로 해석할 때는 γ의 누락된 정규화 상수가 중요하다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현의 입력과 실패 상태를 어떻게 점검해야 하는가?

</div>

??? success "연습문제 2 풀이"

    양의 정수 표본 수, 표본과 함수값의 형태, 목표·제안의 지지집합과 유한 로그 가중치를 확인한다. 가중치 전부 0·무한대·NaN이면 균등 가중치로 대체해 성공으로 처리하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규 사후 예제에서 이론과 구현이 어긋나는 핵심을 설명하라.

</div>

??? success "연습문제 3 풀이"

    우도의 제곱 오차를 관측축만 합해야 표본별 값이 남는다. 현재 전체 합은 θ별 우도 정보를 없앤다. 이 상태에서는 표본 수를 늘려도 의도한 사후분포로 수렴한다는 주장을 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 신경망 형태의 시험 조각이 이 SNIS 함수에 맞는지 평가하라.

</div>

??? success "연습문제 4 풀이"

    표시된 클래스와 backward 인터페이스는 이 NumPy 함수에 없다. 적절한 확인은 상수배 불변성, h=상수의 정확성, 합이 1인 가중치, 알려진 적분 및 우도 반환 형태다. 신경망 시험 조각은 이 구현의 검증으로 사용할 수 없다.

    ```python
    def test_self normalized is():
        model = Self Normalized IS(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

미정규화 밀도에서 SNIS 평균을 계산하고 정규 평균 모형의 해석적 사후분포와 비교한다. 가중치 집중도와 반복 추정 오차를 구분하는 예제다. 가중치 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
