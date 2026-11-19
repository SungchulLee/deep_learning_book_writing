# 간단한 베이즈 예제

베타-이항, 정규-정규, 감마-푸아송의 켤레 사후분포를 기준으로 중요도 샘플링을 비교한다. 사전 제안의 웨이트, 사후 모멘트와 예측확률의 차이를 설명한다.

## 1. 코드

??? note "코드 (442줄)"

    ```python
    """
    03_bayesian_simple_examples.py

    입문 단계: 중요도 샘플링을 쓴 단순한 베이즈 추론 예제

    이 단원은 켤레 베이즈 모델에 중요도 샘플링을 적용한다.
    켤레 모델에는 결과를 확인할 수 있는 해석해가 있다.

    다루는 모델:
    1. 베타-이항(베르누이 관측)
    2. 정규-정규(분산이 알려짐)
    3. 감마-푸아송(푸아송 관측)

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")


    def plot_bayesian_update(prior_dist, posterior_dist, data, param_name='θ',
                             title='Bayesian Update'):
        """베이즈 갱신을 시각화하는 보조 함수."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 분포 그리기
        ax = axes[0]
        x = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 1000)
        if hasattr(posterior_dist, 'ppf'):
            x_post = np.linspace(posterior_dist.ppf(0.001), posterior_dist.ppf(0.999), 1000)
            x = np.union1d(x, x_post)

        ax.plot(x, prior_dist.pdf(x), 'b--', linewidth=2, label='Prior', alpha=0.7)
        ax.plot(x, posterior_dist.pdf(x), 'r-', linewidth=2, label='Posterior', alpha=0.7)
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 데이터 그리기
        ax = axes[1]
        if len(data) < 50:
            ax.hist(data, bins=min(len(np.unique(data)), 20), alpha=0.7, 
                    edgecolor='black', color='green')
        else:
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('Observed Data', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Data (n={len(data)})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


    # 예제 1: 베타-이항 모델
    # ============================
    print("=" * 70)
    print("EXAMPLE 1: Beta-Binomial Model (Bernoulli Trials)")
    print("=" * 70)

    print("""
    모델:
      가능도: y ~ Bernoulli(θ), n번 시도에 성공 s번 관측
      사전분포: θ ~ Beta(α₀, β₀)
      사후분포: θ|y ~ Beta(α₀+s, β₀+n-s)

    과제: 중요도 샘플링으로 사후분포의 평균과 분산 추정하기
    """)

    # 합성 데이터 만들기: 동전 던지기
    true_theta = 0.7  # 참 성공 확률
    n_trials = 50
    data = np.random.binomial(1, true_theta, n_trials)
    successes = np.sum(data)
    failures = n_trials - successes

    print(f"\nData: {successes} successes out of {n_trials} trials")
    print(f"Sample proportion: {successes/n_trials:.3f}")

    # 사전분포: Beta(2, 2) - 약한 정보를 주고 중심이 0.5
    alpha_0, beta_0 = 2, 2
    prior_dist = stats.beta(alpha_0, beta_0)

    print(f"\nPrior: Beta({alpha_0}, {beta_0})")
    print(f"  Prior mean: {alpha_0/(alpha_0+beta_0):.3f}")

    # 해석적으로 구한 사후분포
    alpha_n = alpha_0 + successes
    beta_n = beta_0 + failures
    posterior_dist = stats.beta(alpha_n, beta_n)

    print(f"\nPosterior (analytical): Beta({alpha_n}, {beta_n})")
    print(f"  Posterior mean: {alpha_n/(alpha_n+beta_n):.6f}")
    print(f"  Posterior variance: {posterior_dist.var():.6f}")

    # 정규화하지 않은 사후분포: γ(θ) = p(y|θ)p(θ)
    def unnormalized_posterior_beta(theta):
        """
        γ(θ) = θˢ(1-θ)ⁿ⁻ˢ × θ^(α₀-1)(1-θ)^(β₀-1)
             = θ^(α₀+s-1)(1-θ)^(β₀+n-s-1)

        이는 Beta(α₀+s, β₀+n-s)에 비례한다
        """
        # 경계에서의 수치 문제 방지
        theta = np.clip(theta, 1e-10, 1-1e-10)

        # 수치 안정을 위한 로그 공간
        log_likelihood = successes * np.log(theta) + failures * np.log(1 - theta)
        log_prior = (alpha_0-1) * np.log(theta) + (beta_0-1) * np.log(1 - theta)

        return np.exp(log_likelihood + log_prior)

    # 사전분포를 제안으로 쓴 중요도 샘플링
    n_samples = 5000

    # 사전분포에서 샘플링
    samples = prior_dist.rvs(size=n_samples)

    # 정규화하지 않은 웨이트 계산하기
    gamma_values = unnormalized_posterior_beta(samples)
    q_values = prior_dist.pdf(samples)
    unnorm_weights = gamma_values / q_values

    # 웨이트 정규화
    weights = unnorm_weights / np.sum(unnorm_weights)

    # 사후 평균 추정하기
    h_mean = lambda theta: theta
    posterior_mean_is = np.sum(weights * h_mean(samples))

    # 사후 분산 추정하기
    h_var = lambda theta: (theta - posterior_mean_is)**2
    posterior_var_is = np.sum(weights * h_var(samples))

    # ESS
    ess = 1.0 / np.sum(weights**2)

    print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess:.1f}):")
    print(f"  Posterior mean: {posterior_mean_is:.6f} (error: {abs(posterior_mean_is - posterior_dist.mean()):.6f})")
    print(f"  Posterior variance: {posterior_var_is:.6f} (error: {abs(posterior_var_is - posterior_dist.var()):.6f})")

    # 시각화한다
    fig = plot_bayesian_update(prior_dist, posterior_dist, data, 'θ', 
                               'Beta-Binomial: Prior vs Posterior')
    plt.savefig('example1_beta_binomial.png', 
                dpi=300, bbox_inches='tight')


    # 예제 2: 정규-정규 모델
    # ============================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Normal-Normal Model (Known Variance)")
    print("=" * 70)

    print("""
    모델:
      가능도: y ~ N(θ, σ²), σ²은 알려져 있음
      Prior: θ ~ N(μ₀, τ₀²)
      사후분포: θ|y ~ N(μₙ, τₙ²), 여기서
        τₙ² = 1/(1/τ₀² + n/σ²)
        μₙ = τₙ²(μ₀/τ₀² + Σyᵢ/σ²)

    과제: 중요도 샘플링으로 사후분포 추정하기
    """)

    # 데이터를 생성한다
    true_theta = 8.0
    sigma = 2.0  # 알려진 관측 잡음
    n_obs = 30
    data_normal = np.random.normal(true_theta, sigma, n_obs)

    print(f"\nData: n={n_obs}, sample mean={np.mean(data_normal):.3f}, σ={sigma}")

    # 사전분포
    mu_0 = 5.0
    tau_0 = 3.0
    prior_normal = stats.norm(mu_0, tau_0)

    print(f"\nPrior: N({mu_0}, {tau_0}²)")

    # 해석적으로 구한 사후분포
    precision_0 = 1.0 / tau_0**2
    precision_n = precision_0 + n_obs / sigma**2
    tau_n = 1.0 / np.sqrt(precision_n)
    mu_n = (precision_0 * mu_0 + np.sum(data_normal) / sigma**2) / precision_n
    posterior_normal = stats.norm(mu_n, tau_n)

    print(f"\nPosterior (analytical): N({mu_n:.6f}, {tau_n:.6f}²)")

    # 정규화하지 않은 사후분포
    def unnormalized_posterior_normal(theta):
        """
        γ(θ) = ∏ᵢ exp(-(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ₀²)
        """
        log_likelihood = -0.5 * np.sum((data_normal[:, None] - theta)**2) / sigma**2
        log_prior = -0.5 * (theta - mu_0)**2 / tau_0**2
        return np.exp(log_likelihood + log_prior)

    # 사전분포를 제안으로 쓴 중요도 샘플링
    n_samples = 5000
    samples_normal = prior_normal.rvs(size=n_samples)

    # 정규화하지 않은 웨이트
    gamma_values_normal = unnormalized_posterior_normal(samples_normal)
    q_values_normal = prior_normal.pdf(samples_normal)
    unnorm_weights_normal = gamma_values_normal / q_values_normal
    weights_normal = unnorm_weights_normal / np.sum(unnorm_weights_normal)

    # 추정값
    posterior_mean_normal_is = np.sum(weights_normal * samples_normal)
    posterior_var_normal_is = np.sum(weights_normal * (samples_normal - posterior_mean_normal_is)**2)
    ess_normal = 1.0 / np.sum(weights_normal**2)

    print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_normal:.1f}):")
    print(f"  Posterior mean: {posterior_mean_normal_is:.6f} (true: {mu_n:.6f})")
    print(f"  Posterior std: {np.sqrt(posterior_var_normal_is):.6f} (true: {tau_n:.6f})")

    # 신용구간: 95%
    sorted_samples = samples_normal[np.argsort(weights_normal)[::-1]]
    sorted_weights = np.sort(weights_normal)[::-1]
    cumsum_weights = np.cumsum(sorted_weights)
    n_95 = np.searchsorted(cumsum_weights, 0.95) + 1
    credible_95_is = np.percentile(sorted_samples[:n_95], [2.5, 97.5])

    # 참 신용구간
    credible_95_true = posterior_normal.ppf([0.025, 0.975])

    print(f"\n95% Credible Interval:")
    print(f"  IS: [{credible_95_is[0]:.3f}, {credible_95_is[1]:.3f}]")
    print(f"  True: [{credible_95_true[0]:.3f}, {credible_95_true[1]:.3f}]")

    # 시각화한다
    fig = plot_bayesian_update(prior_normal, posterior_normal, data_normal, 'θ',
                               'Normal-Normal: Prior vs Posterior')
    plt.savefig('example2_normal_normal.png',
                dpi=300, bbox_inches='tight')


    # 예제 3: 감마-푸아송 모델
    # ============================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gamma-Poisson Model")
    print("=" * 70)

    print("""
    모델:
      가능도: y ~ Poisson(λ), 관측한 횟수
      사전분포: λ ~ Gamma(α₀, β₀)
      사후분포: λ|y ~ Gamma(α₀+Σyᵢ, β₀+n)

    과제: 중요도 샘플링으로 사후분포 추정하기
    """)

    # 계수 데이터를 생성한다
    true_lambda = 4.5
    n_counts = 40
    data_poisson = np.random.poisson(true_lambda, n_counts)
    sum_counts = np.sum(data_poisson)

    print(f"\nData: n={n_counts}, Σyᵢ={sum_counts}, sample mean={np.mean(data_poisson):.3f}")

    # 사전분포: Gamma(2, 0.5)
    # 평균 = α/β = 2/0.5 = 4
    alpha_0_gamma = 2.0
    beta_0_gamma = 0.5
    prior_gamma = stats.gamma(alpha_0_gamma, scale=1.0/beta_0_gamma)

    print(f"\nPrior: Gamma({alpha_0_gamma}, {beta_0_gamma})")
    print(f"  Prior mean: {alpha_0_gamma/beta_0_gamma:.3f}")

    # 해석적으로 구한 사후분포
    alpha_n_gamma = alpha_0_gamma + sum_counts
    beta_n_gamma = beta_0_gamma + n_counts
    posterior_gamma = stats.gamma(alpha_n_gamma, scale=1.0/beta_n_gamma)

    print(f"\nPosterior (analytical): Gamma({alpha_n_gamma}, {beta_n_gamma})")
    print(f"  Posterior mean: {alpha_n_gamma/beta_n_gamma:.6f}")

    # 정규화하지 않은 사후분포
    def unnormalized_posterior_gamma(lam):
        """
        γ(λ) = ∏ᵢ λ^yᵢ exp(-λ) × λ^(α₀-1) exp(-β₀λ)
             = λ^(α₀+Σyᵢ-1) exp(-(β₀+n)λ)

        이는 Gamma(α₀+Σyᵢ, β₀+n)에 비례한다
        """
        # 안정을 위한 로그 공간
        log_likelihood = sum_counts * np.log(lam + 1e-10) - n_counts * lam
        log_prior = (alpha_0_gamma-1) * np.log(lam + 1e-10) - beta_0_gamma * lam
        return np.exp(log_likelihood + log_prior)

    # 중요도 샘플링
    n_samples = 5000
    samples_gamma = prior_gamma.rvs(size=n_samples)

    # 웨이트
    gamma_values_poisson = unnormalized_posterior_gamma(samples_gamma)
    q_values_gamma = prior_gamma.pdf(samples_gamma)
    unnorm_weights_gamma = gamma_values_poisson / q_values_gamma
    weights_gamma = unnorm_weights_gamma / np.sum(unnorm_weights_gamma)

    # 추정값
    posterior_mean_gamma_is = np.sum(weights_gamma * samples_gamma)
    posterior_var_gamma_is = np.sum(weights_gamma * (samples_gamma - posterior_mean_gamma_is)**2)
    ess_gamma = 1.0 / np.sum(weights_gamma**2)

    print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_gamma:.1f}):")
    print(f"  Posterior mean: {posterior_mean_gamma_is:.6f} (true: {alpha_n_gamma/beta_n_gamma:.6f})")
    print(f"  Posterior variance: {posterior_var_gamma_is:.6f} (true: {posterior_gamma.var():.6f})")

    # 사후예측분포: P(ỹ|y)
    # 새 관측 ỹ에 대해
    print("\nPosterior Predictive Distribution for new observation:")

    # 감마-푸아송의 참 사후예측분포는 음이항분포다
    # ỹ|y ~ NB(α_n, β_n/(β_n+1))
    post_pred_true = stats.nbinom(alpha_n_gamma, beta_n_gamma/(beta_n_gamma+1))

    # 중요도 샘플링으로 추정하기
    # E[P(ỹ|λ)|y] = E[Poisson(ỹ|λ)|y]
    y_new_values = np.arange(0, 15)
    post_pred_is = []

    for y_new in y_new_values:
        # h(λ) = P(ỹ=y_new|λ) = λ^y_new exp(-λ) / y_new!
        h_poisson = lambda lam: stats.poisson.pmf(y_new, lam)
        prob_is = np.sum(weights_gamma * h_poisson(samples_gamma))
        post_pred_is.append(prob_is)

    post_pred_true_probs = post_pred_true.pmf(y_new_values)

    print(f"\nPosterior Predictive P(ỹ|y):")
    print("y_new  True     IS")
    print("-" * 25)
    for y, p_true, p_is in zip(y_new_values[:8], post_pred_true_probs[:8], post_pred_is[:8]):
        print(f"{y:3d}   {p_true:.4f}  {p_is:.4f}")

    # 시각화한다
    fig = plot_bayesian_update(prior_gamma, posterior_gamma, data_poisson, 'λ',
                               'Gamma-Poisson: Prior vs Posterior')
    plt.savefig('example3_gamma_poisson.png',
                dpi=300, bbox_inches='tight')


    # 비교 분석
    # ===================
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS: ESS Across Models")
    print("=" * 70)

    models = ['Beta-Binomial', 'Normal-Normal', 'Gamma-Poisson']
    ess_values = [ess, ess_normal, ess_gamma]
    efficiencies = [e/n_samples*100 for e in ess_values]

    print("\nEffective Sample Size Summary:")
    print("-" * 50)
    for model, ess_val, eff in zip(models, ess_values, efficiencies):
        print(f"{model:20s}: ESS = {ess_val:6.1f} ({eff:5.1f}%)")

    print("""
    \n관측:
    1. 사전분포를 제안으로 쓰는 것은 단순하지만 늘 효율적이지는 않다
    2. ESS는 데이터가 사전분포를 얼마나 바꾸는지에 달렸다
    3. 데이터의 정보가 많으면(n이 크거나 관측이 극단이면) → ESS가 낮다
    4. 사전분포와 사후분포의 어긋남이 커질수록 ESS가 줄어든다
    5. 가능도의 정보가 많을수록 → 사전분포보다 나은 제안이 필요하다
    """)

    # 마지막 그림: ESS 견주기
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(models, efficiencies, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=13)
    ax.set_title('Importance Sampling Efficiency: Prior as Proposal', 
                 fontsize=14, fontweight='bold')
    ax.axhline(100, color='red', linestyle='--', linewidth=2, 
               label='Perfect efficiency', alpha=0.5)
    ax.set_ylim([0, max(efficiencies)*1.2])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)

    # 막대에 값 라벨을 추가한다
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparative_ess.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 켤레 모델에서는 중요도 샘플링을 해석해로 검증할 수 있다:
       - 이진 데이터에는 베타-이항
       - 연속 데이터에는 정규-정규(분산을 알 때)
       - 계수형 데이터에는 감마-푸아송

    2. 사전분포를 제안으로 쓰는 것은 단순하지만 한계가 있다:
       - 데이터가 적을 때(n이 작을 때) 잘 작동한다
       - 데이터의 정보가 많으면 효율이 낮아진다
       - 가능도가 사전분포를 압도할수록 ESS가 줄어든다

    3. 중요도 샘플링의 좋은 점:
       - 번인(burn-in) 기간이 없다(MCMC와 달리)
       - 표본이 독립이다
       - 같은 표본에서 여러 양을 추정할 수 있다
       - 사후예측분포를 쉽게 계산할 수 있다

    4. 진단 지표로서의 ESS:
       - 독립 표본의 유효 개수를 잰다
       - ESS << n이면 더 나은 제안이 필요하다는 뜻이다
       - 제안을 객관적으로 견줄 수 있다

    5. 실전에서 살필 점:
       - 샘플링한 뒤 ESS를 늘 살펴라
       - ESS가 낮다 → 몇몇 표본이 지배한다
       - ESS가 낮으면 적응적 방법이나 순차적 방법 생각해 보기
       - 복잡한 사후분포에는 더 정교한 제안이 필요하다

    6. MCMC와의 관계:
       - 중요도 샘플링은 MCMC를 보완한다
       - 중요도 샘플링을 쓸 때: 표본이 독립이고 좋은 제안을 알 때
       - MCMC를 쓸 때: 사후분포가 복잡하고 좋은 제안을 모를 때
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (147줄)"

    ```
    ======================================================================
    EXAMPLE 1: Beta-Binomial Model (Bernoulli Trials)
    ======================================================================

    모델:
      가능도: y ~ Bernoulli(θ), n번 시도에 성공 s번 관측
      사전분포: θ ~ Beta(α₀, β₀)
      사후분포: θ|y ~ Beta(α₀+s, β₀+n-s)

    과제: 중요도 샘플링으로 사후분포의 평균과 분산 추정하기


    Data: 39 successes out of 50 trials
    Sample proportion: 0.780

    Prior: Beta(2, 2)
      Prior mean: 0.500

    Posterior (analytical): Beta(41, 13)
      Posterior mean: 0.759259
      Posterior variance: 0.003323

    Importance Sampling Estimates (n=5000, ESS=1112.9):
      Posterior mean: 0.756897 (error: 0.002362)
      Posterior variance: 0.003287 (error: 0.000037)

    ======================================================================
    EXAMPLE 2: Normal-Normal Model (Known Variance)
    ======================================================================

    모델:
      가능도: y ~ N(θ, σ²), σ²은 알려져 있음
      Prior: θ ~ N(μ₀, τ₀²)
      사후분포: θ|y ~ N(μₙ, τₙ²), 여기서
        τₙ² = 1/(1/τ₀² + n/σ²)
        μₙ = τₙ²(μ₀/τ₀² + Σyᵢ/σ²)

    과제: 중요도 샘플링으로 사후분포 추정하기


    Data: n=30, sample mean=7.737, σ=2.0

    Prior: N(5.0, 3.0²)

    Posterior (analytical): N(7.696626, 0.362473²)

    Importance Sampling Estimates (n=5000, ESS=nan):
      Posterior mean: nan (true: 7.696626)
      Posterior std: nan (true: 0.362473)

    95% Credible Interval:
      IS: [6.311, 6.311]
      True: [6.986, 8.407]

    ======================================================================
    EXAMPLE 3: Gamma-Poisson Model
    ======================================================================

    모델:
      가능도: y ~ Poisson(λ), 관측한 횟수
      사전분포: λ ~ Gamma(α₀, β₀)
      사후분포: λ|y ~ Gamma(α₀+Σyᵢ, β₀+n)

    과제: 중요도 샘플링으로 사후분포 추정하기


    Data: n=40, Σyᵢ=211, sample mean=5.275

    Prior: Gamma(2.0, 0.5)
      Prior mean: 4.000

    Posterior (analytical): Gamma(213.0, 40.5)
      Posterior mean: 5.259259

    Importance Sampling Estimates (n=5000, ESS=624.4):
      Posterior mean: 5.280833 (true: 5.259259)
      Posterior variance: 0.119884 (true: 0.129858)

    Posterior Predictive Distribution for new observation:

    Posterior Predictive P(ỹ|y):
    y_new  True     IS
    -------------------------
      0   0.0055  0.0054
      1   0.0284  0.0279
      2   0.0733  0.0723
      3   0.1267  0.1254
      4   0.1648  0.1640
      5   0.1723  0.1723
      6   0.1509  0.1515
      7   0.1138  0.1146

    ======================================================================
    COMPARATIVE ANALYSIS: ESS Across Models
    ======================================================================

    Effective Sample Size Summary:
    --------------------------------------------------
    Beta-Binomial       : ESS = 1112.9 ( 22.3%)
    Normal-Normal       : ESS =    nan (  nan%)
    Gamma-Poisson       : ESS =  624.4 ( 12.5%)


    관측:
    1. 사전분포를 제안으로 쓰는 것은 단순하지만 늘 효율적이지는 않다
    2. ESS는 데이터가 사전분포를 얼마나 바꾸는지에 달렸다
    3. 데이터의 정보가 많으면(n이 크거나 관측이 극단이면) → ESS가 낮다
    4. 사전분포와 사후분포의 어긋남이 커질수록 ESS가 줄어든다
    5. 가능도의 정보가 많을수록 → 사전분포보다 나은 제안이 필요하다


    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 켤레 모델에서는 중요도 샘플링을 해석해로 검증할 수 있다:
       - 이진 데이터에는 베타-이항
       - 연속 데이터에는 정규-정규(분산을 알 때)
       - 계수형 데이터에는 감마-푸아송

    2. 사전분포를 제안으로 쓰는 것은 단순하지만 한계가 있다:
       - 데이터가 적을 때(n이 작을 때) 잘 작동한다
       - 데이터의 정보가 많으면 효율이 낮아진다
       - 가능도가 사전분포를 압도할수록 ESS가 줄어든다

    3. 중요도 샘플링의 좋은 점:
       - 번인(burn-in) 기간이 없다(MCMC와 달리)
       - 표본이 독립이다
       - 같은 표본에서 여러 양을 추정할 수 있다
       - 사후예측분포를 쉽게 계산할 수 있다

    4. 진단 지표로서의 ESS:
       - 독립 표본의 유효 개수를 잰다
       - ESS << n이면 더 나은 제안이 필요하다는 뜻이다
       - 제안을 객관적으로 견줄 수 있다

    5. 실전에서 살필 점:
       - 샘플링한 뒤 ESS를 늘 살펴라
       - ESS가 낮다 → 몇몇 표본이 지배한다
       - ESS가 낮으면 적응적 방법이나 순차적 방법 생각해 보기
       - 복잡한 사후분포에는 더 정교한 제안이 필요하다

    6. MCMC와의 관계:
       - 중요도 샘플링은 MCMC를 보완한다
       - 중요도 샘플링을 쓸 때: 표본이 독립이고 좋은 제안을 알 때
       - MCMC를 쓸 때: 사후분포가 복잡하고 좋은 제안을 모를 때

    ```


## 2. 논의

### 켤레 사후분포와 웨이트

성공 s, 시행 n에서 Beta(a,b) 사전은 Beta(a+s,b+n−s) 사후가 된다. 감마 사전의 두 번째 인자를 rate b로 쓰면 푸아송 관측 후 shape=a+∑y, rate=b+n이다. SciPy gamma에는 scale=1/rate를 넘긴다.

사전을 제안으로 사용하면 가능도(likelihood, 우도라고도 한다)에 비례하는 웨이트를 얻는다. 세 예제는 웨이트를 합으로 나누므로 SNIS다. 사후평균과 표본 평균을 중심으로 한 가중 분산은 유한 표본에서 일반적으로 편향되며, 같은 가중 표본으로 여러 양을 계산할 수 있다. 사전·사후의 어긋남이 커지면 흔히 웨이트가 집중하지만 n만으로 ESS를 결정할 수는 없다.

### 정규 가능도와 구간 계산의 한계

정규 예제의 unnormalized_posterior_normal은 관측×제안 표본의 제곱 오차를 축 없이 모두 합한다. 그 결과 각 제안에 대응하는 가능도가 사라져 의도한 사후 웨이트가 계산되지 않는다. 수식으로 제시된 해석적 사후분포와 이 추정량의 일치를 보장할 수 없다.

코드의 IS 95% 구간은 웨이트가 큰 표본을 누적 95%까지 고른 뒤 그 부분집합의 비가중 분위수를 구한다. 이는 사후분포의 가중 2.5%·97.5% 분위수가 아니다. 가중 분위수는 θ값을 오름차순 정렬한 뒤 그 순서의 누적 웨이트가 각 확률에 도달하는 위치에서 구한다. 사후 신용구간과 추정량의 몬테카를로 신뢰구간도 구분해야 한다.

### 사후예측과 수치 범위

감마-푸아송 사후예측은 shape a_n, 성공확률 b_n/(b_n+1)의 음이항분포다. 코드의 가중 평균은 각 새 정수 y에 대해 Poisson(y|λ)를 적분한다. 표시 범위 0,…,14는 전체 지지집합이 아니므로 그 확률 합이 반드시 1일 필요는 없다.

직접 exp·pdf를 사용하는 웨이트에는 언더플로와 오버플로가 가능하다. 베타의 clip과 감마의 작은 상수 추가는 경계의 수치 보정이며 원래 밀도의 정확한 정의와 구분한다. ESS는 웨이트 집중도이고, 단 한 번의 작은 오차나 큰 ESS가 구현의 정확성을 증명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 예제에서 켤레 모델을 사용하는 이유는?

</div>

??? success "풀이"

    해석적 사후 모멘트와 예측확률을 독립 기준으로 사용할 수 있다. 제안이 사전일 때 웨이트가 가능도에 비례하는 관계도 확인할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
베타와 감마 모델에서 유효한 입력은 무엇인가?

</div>

??? success "풀이"

    베타의 shape 두 개와 감마의 shape·rate는 양수이며 성공 수는 0≤s≤n인 정수다. 푸아송 데이터는 음이 아닌 정수이고 노출 시간이 모두 1이라는 가능도 가정을 확인한다. 표본 수·배열 형태·유한 웨이트 합도 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사전 강도와 제안 선택이 추론에 미치는 영향을 구분하라.

</div>

??? success "풀이"

    사전을 바꾸면 통계적 목표인 사후분포가 바뀐다. 사후를 고정하고 제안만 바꾸면 수치 추정의 오차와 비용이 바뀐다. 표본 수 증가는 잘못된 가능도 축이나 잘못된 분위수 계산을 고치지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 시험 조각 대신 이 예제를 검증할 수 있는 기준을 설명하라.

</div>

??? success "풀이"

    해석적 베타·정규·감마 사후의 모멘트, 정규 가능도의 표본별 반환 형태, 가중 누적분포의 분위수와 음이항 예측확률을 비교한다. 아래 가상 뉴럴 네트워크 클래스·그래디언트 시험은 이 NumPy/SciPy 코드와 맞지 않는다.

    ```python
    def test_bayesian simple examples():
        model = Bayesian Simple Examples(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

베타-이항, 정규-정규, 감마-푸아송의 켤레 사후분포를 기준으로 중요도 샘플링을 비교한다. 사전 제안의 웨이트, 사후 모멘트와 예측확률의 차이를 설명한다. 웨이트 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
