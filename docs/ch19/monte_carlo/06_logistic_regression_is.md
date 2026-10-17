# 로지스틱 회귀 중요도 샘플링

가우스 사전과 베르누이 로지스틱 우도의 사후분포를 SNIS로 근사한다. 사전 제안과 MAP의 라플라스 제안을 비교하고 계수 추론과 사후예측을 구분한다.

## 1. 코드

??? note "코드 (520줄)"

    ```python
    """
    06_logistic_regression_IS.py

    중급 단계: 중요도 표집으로 하는 베이즈 로지스틱 회귀

    이 단원은 베이즈 추론을 위한 중요도 표집을 보인다
    로지스틱 회귀에서 다룬다. 이는 뒤확률이 닫힌 꼴로 주어지지 않는
    켤레가 아닌 모형이다.

    모형:
    ------
    가능도: yᵢ ~ Bernoulli(π(xᵢ'β))
                여기서 π(z) = 1/(1 + exp(-z))은 로지스틱 함수이다

    앞확률: β ~ N(μ₀, Σ₀)

    Posterior: p(β|y,X) ∝ ∏ᵢ π(xᵢ'β)^yᵢ (1-π(xᵢ'β))^(1-yᵢ) × N(β|μ₀,Σ₀)

    이는 켤레가 아닌 모형이다. 곧 뒤확률이 가우스가 아니다!

    제안 전략:
    1. 앞확률을 제안으로(단순하지만 흔히 효율이 낮다)
    2. 라플라스 어림(봉우리에서의 가우스)
    3. 변분 어림
    4. 맞춰 가는 제안

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.special import expit  # 로지스틱 함수
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")


    def logistic(z):
        """로지스틱 함수: π(z) = 1/(1 + exp(-z))"""
        return expit(z)


    def log_likelihood_logistic(beta, X, y):
        """
        로지스틱 회귀의 로그 가능도.

        log p(y|X,β) = Σᵢ [yᵢ log π(xᵢ'β) + (1-yᵢ) log(1-π(xᵢ'β))]
        """
        eta = X @ beta  # 선형 미리봄 xᵢ'β

        # 수치로 안정된 로그 가능도
        # log π(η) = -log(1 + exp(-η))
        # log(1-π(η)) = -log(1 + exp(η))

        log_lik = np.sum(
            y * (-np.log1p(np.exp(-eta))) +
            (1 - y) * (-np.log1p(np.exp(eta)))
        )

        return log_lik


    def log_prior_gaussian(beta, mu_0, Sigma_0_inv):
        """
        로그 앞확률: log p(β) = log N(β|μ₀, Σ₀)
        """
        diff = beta - mu_0
        log_prior = -0.5 * (diff.T @ Sigma_0_inv @ diff)
        return log_prior


    def log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv):
        """
        고르게 하지 않은 로그 뒤확률: log p(β|y,X) ∝ log p(y|X,β) + log p(β)
        """
        return (log_likelihood_logistic(beta, X, y) +
                log_prior_gaussian(beta, mu_0, Sigma_0_inv))


    def find_map_estimate(X, y, mu_0, Sigma_0_inv):
        """
        최적화로 MAP(최대 뒤확률) 어림값 찾기.

        MAP = argmax_β p(β|y,X)
        """
        # 로그 뒤확률의 음수(최소화를 위해)
        def neg_log_post(beta):
            return -log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv)

        # 경사
        def grad(beta):
            eta = X @ beta
            prob = logistic(eta)
            grad_ll = X.T @ (y - prob)
            grad_prior = -Sigma_0_inv @ (beta - mu_0)
            return -(grad_ll + grad_prior)

        # 최적화
        result = minimize(neg_log_post, mu_0, jac=grad, method='BFGS')

        return result.x


    def laplace_approximation(X, y, mu_0, Sigma_0_inv):
        """
        뒤확률의 라플라스 어림 셈하기.

        MAP에 가운데를 맞춘 가우스로 뒤확률을 어림한다:
        p(β|y,X) ≈ N(β|β_MAP, H⁻¹)

        여기서 H은 MAP에서 로그 뒤확률의 음수의 헤세 행렬이다.

        반환값:
        --------
        beta_map : MAP 어림값
        Sigma_laplace : 라플라스 어림의 공분산 행렬
        """
        # MAP 찾기
        beta_map = find_map_estimate(X, y, mu_0, Sigma_0_inv)

        # MAP에서 헤세 행렬 셈하기
        eta = X @ beta_map
        prob = logistic(eta)

        # 로그 가능도의 헤세 행렬
        W = np.diag(prob * (1 - prob))  # 가중값 행렬
        H_ll = -X.T @ W @ X

        # 로그 뒤확률의 헤세 행렬
        H = H_ll - Sigma_0_inv

        # 공분산은 헤세 행렬의 음수의 역행렬이다
        Sigma_laplace = np.linalg.inv(-H)

        return beta_map, Sigma_laplace


    # 보기 1: 단순한 1차원 로지스틱 회귀
    # =======================================
    print("=" * 70)
    print("EXAMPLE 1: 1D Logistic Regression")
    print("=" * 70)

    # 합성 데이터 생성
    np.random.seed(42)
    n_obs = 100

    # 참 파라메터
    beta_true = np.array([0.5, 2.0])  # [절편, 그래디언트]

    # 피처: [1, x]
    x_raw = np.random.uniform(-2, 2, n_obs)
    X = np.column_stack([np.ones(n_obs), x_raw])

    # 두 값 결과 만들기
    eta_true = X @ beta_true
    prob_true = logistic(eta_true)
    y = np.random.binomial(1, prob_true)

    print(f"\nGenerated {n_obs} observations")
    print(f"True β = {beta_true}")
    print(f"Observed: {np.sum(y)} successes, {n_obs - np.sum(y)} failures")

    # 앞확률: N(0, 10I) - 정보가 약함
    mu_0 = np.zeros(2)
    Sigma_0 = 10 * np.eye(2)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    print(f"\nPrior: β ~ N(0, 10I)")

    # 라플라스 어림 셈하기
    beta_map, Sigma_laplace = laplace_approximation(X, y, mu_0, Sigma_0_inv)

    print(f"\nMAP estimate: {beta_map}")
    print(f"Laplace std: {np.sqrt(np.diag(Sigma_laplace))}")


    # 제안을 달리한 중요도 표집
    # ------------------------------------------

    # 제안 1: 앞확률(단순하지만 효율이 낮음)
    print("\n" + "-" * 70)
    print("PROPOSAL 1: Using Prior as Proposal")
    print("-" * 70)

    n_samples = 5000
    prior_dist = stats.multivariate_normal(mu_0, Sigma_0)

    # 앞확률에서 표집
    samples_prior = prior_dist.rvs(size=n_samples)

    # 중요도 무게 셈하기
    log_weights_prior = np.array([
        log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
        prior_dist.logpdf(beta)
        for beta in samples_prior
    ])

    # 무게 고르게 하기
    log_weights_prior_norm = log_weights_prior - np.max(log_weights_prior)
    weights_prior_unnorm = np.exp(log_weights_prior_norm)
    weights_prior = weights_prior_unnorm / np.sum(weights_prior_unnorm)

    # ESS
    ess_prior = 1.0 / np.sum(weights_prior**2)

    # 어림값
    beta_est_prior = np.sum(weights_prior[:, None] * samples_prior, axis=0)

    print(f"ESS: {ess_prior:.1f} ({ess_prior/n_samples:.1%})")
    print(f"Estimated β: {beta_est_prior}")
    print(f"Error: {np.linalg.norm(beta_est_prior - beta_true):.4f}")


    # 제안 2: 라플라스 어림(훨씬 나음)
    print("\n" + "-" * 70)
    print("PROPOSAL 2: Using Laplace Approximation as Proposal")
    print("-" * 70)

    laplace_dist = stats.multivariate_normal(beta_map, Sigma_laplace)

    # 라플라스 어림에서 표집
    samples_laplace = laplace_dist.rvs(size=n_samples)

    # 중요도 무게 셈하기
    log_weights_laplace = np.array([
        log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
        laplace_dist.logpdf(beta)
        for beta in samples_laplace
    ])

    # 무게 고르게 하기
    log_weights_laplace_norm = log_weights_laplace - np.max(log_weights_laplace)
    weights_laplace_unnorm = np.exp(log_weights_laplace_norm)
    weights_laplace = weights_laplace_unnorm / np.sum(weights_laplace_unnorm)

    # ESS
    ess_laplace = 1.0 / np.sum(weights_laplace**2)

    # 어림값
    beta_est_laplace = np.sum(weights_laplace[:, None] * samples_laplace, axis=0)

    print(f"ESS: {ess_laplace:.1f} ({ess_laplace/n_samples:.1%})")
    print(f"Estimated β: {beta_est_laplace}")
    print(f"Error: {np.linalg.norm(beta_est_laplace - beta_true):.4f}")

    print(f"\nImprovement: {ess_laplace/ess_prior:.1f}x better ESS with Laplace proposal")


    # 결과를 그려 본다
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 칸 1: 자료와 맞춘 곡선
    ax = axes[0, 0]
    sorted_idx = np.argsort(x_raw)
    x_plot = x_raw[sorted_idx]

    # 참 확률
    X_plot = np.column_stack([np.ones(len(x_plot)), x_plot])
    prob_true_plot = logistic(X_plot @ beta_true)

    # MAP 어림값
    prob_map = logistic(X_plot @ beta_map)

    # 중요도 표집 어림값(라플라스 제안 사용)
    prob_is = logistic(X_plot @ beta_est_laplace)

    ax.scatter(x_raw[y==0], y[y==0], c='red', alpha=0.5, s=50, label='y=0')
    ax.scatter(x_raw[y==1], y[y==1], c='blue', alpha=0.5, s=50, label='y=1')
    ax.plot(x_plot, prob_true_plot, 'k-', linewidth=3, label='True', alpha=0.7)
    ax.plot(x_plot, prob_map, 'g--', linewidth=2, label='MAP')
    ax.plot(x_plot, prob_is, 'r:', linewidth=2, label='IS (Laplace)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('P(y=1|x)', fontsize=12)
    ax.set_title('Logistic Regression Fit', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 2: 뒤확률 표본(앞확률 제안)
    ax = axes[0, 1]
    scatter = ax.scatter(samples_prior[:, 0], samples_prior[:, 1],
                        c=weights_prior*n_samples, cmap='viridis',
                        s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
    ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
    ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
    ax.set_xlabel('β₀ (intercept)', fontsize=11)
    ax.set_ylabel('β₁ (slope)', fontsize=11)
    ax.set_title(f'Prior Proposal: ESS={ess_prior:.0f}', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Weight × n')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 3: 뒤확률 표본(라플라스 제안)
    ax = axes[1, 0]
    scatter = ax.scatter(samples_laplace[:, 0], samples_laplace[:, 1],
                        c=weights_laplace*n_samples, cmap='viridis',
                        s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
    ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
    ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
    ax.set_xlabel('β₀ (intercept)', fontsize=11)
    ax.set_ylabel('β₁ (slope)', fontsize=11)
    ax.set_title(f'Laplace Proposal: ESS={ess_laplace:.0f}', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Weight × n')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 4: 무게 분포 견주기
    ax = axes[1, 1]
    ax.hist(weights_prior * n_samples, bins=50, alpha=0.5, density=True,
            color='blue', edgecolor='black', label='Prior proposal')
    ax.hist(weights_laplace * n_samples, bins=50, alpha=0.5, density=True,
            color='green', edgecolor='black', label='Laplace proposal')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
    ax.set_xlabel('Weight × n', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weight Distributions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])

    plt.tight_layout()
    plt.savefig('logistic_regression_1d.png',
                dpi=300, bbox_inches='tight')


    # 보기 2: 차원이 더 높은 경우
    # =================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Higher-Dimensional Logistic Regression")
    print("=" * 70)

    # 피처 p=5개인 자료 만들기
    n_obs_2 = 200
    p = 5

    # 참된 파라메터
    beta_true_2 = np.array([1.0, -0.5, 0.8, -0.3, 0.6])

    # 피처
    X_2 = np.random.randn(n_obs_2, p)

    # 결과
    eta_true_2 = X_2 @ beta_true_2
    prob_true_2 = logistic(eta_true_2)
    y_2 = np.random.binomial(1, prob_true_2)

    print(f"\nData: n={n_obs_2}, p={p} features")
    print(f"True β = {beta_true_2}")

    # 앞확률
    mu_0_2 = np.zeros(p)
    Sigma_0_2 = 5 * np.eye(p)
    Sigma_0_inv_2 = np.linalg.inv(Sigma_0_2)

    # 라플라스 어림
    beta_map_2, Sigma_laplace_2 = laplace_approximation(X_2, y_2, mu_0_2, Sigma_0_inv_2)

    print(f"\nMAP estimate: {beta_map_2}")

    # 라플라스 제안을 쓴 중요도 표집
    n_samples_2 = 10000
    laplace_dist_2 = stats.multivariate_normal(beta_map_2, Sigma_laplace_2)
    samples_2 = laplace_dist_2.rvs(size=n_samples_2)

    log_weights_2 = np.array([
        log_posterior_logistic(beta, X_2, y_2, mu_0_2, Sigma_0_inv_2) -
        laplace_dist_2.logpdf(beta)
        for beta in samples_2
    ])

    log_weights_2_norm = log_weights_2 - np.max(log_weights_2)
    weights_2_unnorm = np.exp(log_weights_2_norm)
    weights_2 = weights_2_unnorm / np.sum(weights_2_unnorm)

    ess_2 = 1.0 / np.sum(weights_2**2)

    # 뒤확률 평균
    beta_post_mean = np.sum(weights_2[:, None] * samples_2, axis=0)

    # 뒤확률 표준편차
    beta_post_std = np.sqrt(np.sum(weights_2[:, None] * (samples_2 - beta_post_mean)**2, axis=0))

    print(f"\nImportance Sampling Results (Laplace proposal):")
    print(f"  n_samples: {n_samples_2}")
    print(f"  ESS: {ess_2:.1f} ({ess_2/n_samples_2:.1%})")

    print("\nPosterior Estimates:")
    print(f"{'Feature':<10} {'True':>8} {'MAP':>8} {'IS Mean':>8} {'IS Std':>8}")
    print("-" * 50)
    for i in range(p):
        print(f"β{i:<9} {beta_true_2[i]:8.3f} {beta_map_2[i]:8.3f} "
              f"{beta_post_mean[i]:8.3f} {beta_post_std[i]:8.3f}")

    # 믿음 구간
    credible_intervals = []
    for i in range(p):
        sorted_samples = samples_2[:, i][np.argsort(weights_2)[::-1]]
        sorted_weights = np.sort(weights_2)[::-1]
        cumsum = np.cumsum(sorted_weights)
        n_95 = np.searchsorted(cumsum, 0.95) + 1
        ci = np.percentile(sorted_samples[:n_95], [2.5, 97.5])
        credible_intervals.append(ci)

    print("\n95% Credible Intervals:")
    for i, ci in enumerate(credible_intervals):
        contains = ci[0] <= beta_true_2[i] <= ci[1]
        status = "✓" if contains else "✗"
        print(f"β{i}: [{ci[0]:6.3f}, {ci[1]:6.3f}] {status}")


    # 보기 3: 미리봄
    # ===================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Posterior Predictive Distribution")
    print("=" * 70)

    # 새 시험 점
    x_new = np.array([0.5, -0.3, 0.2, 0.1, -0.4])

    # 뒤확률 미리봄: P(y_new=1|x_new, data)
    # = ∫ P(y_new=1|x_new, β) p(β|data) dβ
    # ≈ Σᵢ wᵢ × logistic(x_new'βᵢ)

    pred_probs = logistic(samples_2 @ x_new)
    posterior_pred_prob = np.sum(weights_2 * pred_probs)

    # 참 확률
    true_pred_prob = logistic(x_new @ beta_true_2)

    # MAP에 바탕을 둔 미리봄
    map_pred_prob = logistic(x_new @ beta_map_2)

    print(f"\nPredictive probability P(y=1|x_new):")
    print(f"  True: {true_pred_prob:.4f}")
    print(f"  MAP: {map_pred_prob:.4f}")
    print(f"  Posterior mean (IS): {posterior_pred_prob:.4f}")

    # 뒤확률 미리봄 분포
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pred_probs, bins=50, weights=weights_2, density=True,
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(true_pred_prob, color='red', linestyle='--', linewidth=2,
               label=f'True: {true_pred_prob:.3f}')
    ax.axvline(posterior_pred_prob, color='green', linestyle='-', linewidth=2,
               label=f'Posterior mean: {posterior_pred_prob:.3f}')
    ax.set_xlabel('P(y=1|x_new)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Posterior Predictive Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logistic_predictive.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 로지스틱 회귀는 켤레가 아니다:
       - 뒤확률이 닫힌 꼴이 아니다
       - 손으로 푼 풀이가 없다
       - 중요도 표집을 쓸 수 있다

    2. 제안 전략:
       - 앞확률: 단순하지만 흔히 효율이 낮다(ESS이 낮다)
       - 라플라스 어림: MAP에서의 가우스
         * ESS이 훨씬 낫다(흔히 10배에서 100배 나아짐)
         * 차원이 알맞을 때 좋다
       - 많은 경우 효율 30-60%을 이룰 수 있다

    3. 라플라스 어림:
       - 최적화로 MAP 찾기
       - MAP에서 헤세 행렬 셈하기
       - N(β_MAP, H⁻¹)을 제안으로 쓰기
       - 뒤확률이 거의 가우스일 때 잘 듣는다

    4. 차원의 효과:
       - 앞확률 제안의 ESS은 차원에 따라 지수로 나빠진다
       - 라플라스 제안이 규모를 훨씬 잘 견딘다
       - p > 10-20이면 맞춰 가는 방법이나 MCMC가 필요할 수 있다

    5. 추론 과제:
       - 뒤확률의 평균과 흩어짐
       - 믿음 구간
       - 뒤확률 미리봄 분포
       - 무게 표본으로 모두 얻을 수 있다

    6. 실전에서 살필 점:
       - ESS을 늘 살펴라
       - 라플라스 어림이 대개 앞확률보다 훨씬 낫다
       - 차원이 높으면(p > 20) MCMC를 생각해 보라
       - 수치 안정: 무게는 로그 공간에서 다룬다

    7. 로지스틱 회귀에서 중요도 표집의 좋은 점:
       - 독립 표본(자기상관 없음)
       - 태우기 기간이 없다
       - 나란히 하기 쉽다
       - 같은 표본에서 여러 양을 셈할 수 있다

    8. 중요도 표집이 잘 듣는 때:
       - 알맞은 차원(p < 20)
       - 좋은 제안을 쓸 수 있을 때(라플라스, 변분)
       - 자료가 극단으로 갈리지 않을 때
       - 넉넉한 표본 크기(n > 100)

    9. MCMC와 견주기:
       - 중요도 표집: 좋은 제안이 있을 때 더 낫다
       - MCMC: 차원이 높거나 뒤확률이 복잡할 때 낫다
       - 중요도 표집: 진단하기가 더 쉽다(ESS)
       - MCMC: 제안을 고르는 데 더 너그럽다
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (124줄)"

    ```
    ======================================================================
    EXAMPLE 1: 1D Logistic Regression
    ======================================================================

    Generated 100 observations
    True β = [0.5 2. ]
    Observed: 54 successes, 46 failures

    Prior: β ~ N(0, 10I)

    MAP estimate: [0.77833412 2.40529185]
    Laplace std: [0.35420946 0.44247453]

    ----------------------------------------------------------------------
    PROPOSAL 1: Using Prior as Proposal
    ----------------------------------------------------------------------
    ESS: 93.0 (1.9%)
    Estimated β: [0.77389778 2.52963072]
    Error: 0.5963

    ----------------------------------------------------------------------
    PROPOSAL 2: Using Laplace Approximation as Proposal
    ----------------------------------------------------------------------
    ESS: 3470.7 (69.4%)
    Estimated β: [0.8239311  2.56019708]
    Error: 0.6471

    Improvement: 37.3x better ESS with Laplace proposal

    ======================================================================
    EXAMPLE 2: Higher-Dimensional Logistic Regression
    ======================================================================

    Data: n=200, p=5 features
    True β = [ 1.  -0.5  0.8 -0.3  0.6]

    MAP estimate: [ 1.36782169 -0.70025869  1.25013984 -0.19096629  0.37297892]

    Importance Sampling Results (Laplace proposal):
      n_samples: 10000
      ESS: 8381.1 (83.8%)

    Posterior Estimates:
    Feature        True      MAP  IS Mean   IS Std
    --------------------------------------------------
    β0            1.000    1.368    1.427    0.233
    β1           -0.500   -0.700   -0.733    0.210
    β2            0.800    1.250    1.309    0.253
    β3           -0.300   -0.191   -0.200    0.185
    β4            0.600    0.373    0.388    0.203

    95% Credible Intervals:
    β0: [ 1.034,  1.827] ✗
    β1: [-1.116, -0.350] ✓
    β2: [ 0.887,  1.733] ✗
    β3: [-0.547,  0.141] ✓
    β4: [ 0.003,  0.772] ✓

    ======================================================================
    EXAMPLE 3: Posterior Predictive Distribution
    ======================================================================

    Predictive probability P(y=1|x_new):
      True: 0.6318
      MAP: 0.7263
      Posterior mean (IS): 0.7334

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 로지스틱 회귀는 켤레가 아니다:
       - 뒤확률이 닫힌 꼴이 아니다
       - 손으로 푼 풀이가 없다
       - 중요도 표집을 쓸 수 있다

    2. 제안 전략:
       - 앞확률: 단순하지만 흔히 효율이 낮다(ESS이 낮다)
       - 라플라스 어림: MAP에서의 가우스
         * ESS이 훨씬 낫다(흔히 10배에서 100배 나아짐)
         * 차원이 알맞을 때 좋다
       - 많은 경우 효율 30-60%을 이룰 수 있다

    3. 라플라스 어림:
       - 최적화로 MAP 찾기
       - MAP에서 헤세 행렬 셈하기
       - N(β_MAP, H⁻¹)을 제안으로 쓰기
       - 뒤확률이 거의 가우스일 때 잘 듣는다

    4. 차원의 효과:
       - 앞확률 제안의 ESS은 차원에 따라 지수로 나빠진다
       - 라플라스 제안이 규모를 훨씬 잘 견딘다
       - p > 10-20이면 맞춰 가는 방법이나 MCMC가 필요할 수 있다

    5. 추론 과제:
       - 뒤확률의 평균과 흩어짐
       - 믿음 구간
       - 뒤확률 미리봄 분포
       - 무게 표본으로 모두 얻을 수 있다

    6. 실전에서 살필 점:
       - ESS을 늘 살펴라
       - 라플라스 어림이 대개 앞확률보다 훨씬 낫다
       - 차원이 높으면(p > 20) MCMC를 생각해 보라
       - 수치 안정: 무게는 로그 공간에서 다룬다

    7. 로지스틱 회귀에서 중요도 표집의 좋은 점:
       - 독립 표본(자기상관 없음)
       - 태우기 기간이 없다
       - 나란히 하기 쉽다
       - 같은 표본에서 여러 양을 셈할 수 있다

    8. 중요도 표집이 잘 듣는 때:
       - 알맞은 차원(p < 20)
       - 좋은 제안을 쓸 수 있을 때(라플라스, 변분)
       - 자료가 극단으로 갈리지 않을 때
       - 넉넉한 표본 크기(n > 100)

    9. MCMC와 견주기:
       - 중요도 표집: 좋은 제안이 있을 때 더 낫다
       - MCMC: 차원이 높거나 뒤확률이 복잡할 때 낫다
       - 중요도 표집: 진단하기가 더 쉽다(ESS)
       - MCMC: 제안을 고르는 데 더 너그럽다

    ```


## 2. 논의

### 사후 기하와 라플라스 제안

설계행렬 X와 0/1 반응 y에 대해 η=Xβ, p=σ(η)이며 로그우도는 ∑[yη−log(1+exp η)]다. 양의 정부호 가우스 사전의 정밀도를 Λ₀라 하면 음의 로그사후 헤세 행렬은 Xᵀdiag(p(1−p))X+Λ₀다. 코드의 공분산은 이 행렬을 MAP에서 평가한 역행렬이다.

적절한 가우스 사전은 계수 방향의 곡률을 제공하지만 최적화 결과의 성공·경사와 행렬 조건수를 확인해야 한다. 라플라스는 국소 가우스 근사다. 제안으로 사용한 뒤 정확한 로그사후/제안 비를 가중하면 근사분포 자체의 바이어스를 보정하려는 SNIS가 된다. 이 과정에도 유한 표본 바이어스와 꼬리 적률 문제가 남는다.

### 수치 안정성과 비교 기준

log_weights에서 최대값을 빼는 것은 유한 로그값의 정규화에 유용하다. 그러나 log1p(exp(±η))의 exp는 큰 |η|에서 넘칠 수 있고 0×∞가 NaN을 만들 수 있다. 로지스틱 로그우도는 logaddexp 같은 안정적인 softplus 표현으로 평가해야 하며, 사후 상수를 생략한 현재 값은 주변우도 계산에 그대로 사용할 수 없다.

첫 예제는 입력 피처 하나에 절편을 더한 계수 2차원 문제이고 두 번째는 절편 열을 별도로 추가하지 않은 계수 5차원 문제다. 출력의 Error는 생성에 쓴 참 계수와 사후평균의 거리다. 이는 유한 자료의 통계 오차를 포함하며 같은 사후평균에 대한 순수 몬테카를로 오차가 아니다. 라플라스가 항상 10~100배 좋다거나 특정 차원·표본 수에서 방법을 바꿔야 한다는 보편적 경계는 없다.

### 구간과 예측확률

계수의 95% 구간은 웨이트 상위 표본을 누적 95%까지 선택한 뒤 비가중 분위수를 사용한다. 이 계산은 가중 사후 분위수가 아니다. 계수값을 정렬하고 그 순서의 누적 웨이트에서 2.5%·97.5%를 구해야 한다.

새 피처 x에서 사후예측 성공확률은 E[σ(xᵀβ)|자료]다. 일반적으로 σ(xᵀE[β|자료])와 다르다. 마지막 히스토그램은 성공확률 σ(xᵀβ)의 사후 불확실성을 보여 주며, 실제 새 이진 반응의 예측분포는 그 평균을 성공확률로 갖는 베르누이분포다. ESS는 함수별 예측 오차를 직접 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
라플라스 제안의 평균과 공분산을 설명하라.

</div>

??? success "풀이"

    평균은 MAP, 공분산은 MAP에서 음의 로그사후 헤세 행렬의 역행렬이다. 양의 정부호 사전 정밀도와 유효한 설계 형태를 확인하고 최적화 수렴을 점검한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
로지스틱 IS 입력과 수치 문제를 어떻게 확인하는가?

</div>

??? success "풀이"

    X는 n×p, y는 길이 n의 0/1 값, 사전 평균은 길이 p이고 공분산은 양의 정부호여야 한다. 극단적인 선형 예측값의 로그우도·경사와 유한 로그 웨이트를 확인한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
라플라스 오차와 SNIS 오차를 구분하라.

</div>

??? success "풀이"

    라플라스분포 자체를 사후로 사용하면 국소 근사 오차가 있다. 그 분포를 제안으로 쓰고 원래 목표 밀도비로 가중하면 수치 적분의 문제로 바뀌지만 유한 표본 바이어스·분산과 꼬리 조건이 남는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 넷트웍 시험 대신 어떤 수학적 기준으로 확인할 수 있는가?

</div>

??? success "풀이"

    로그우도 경사·헤세 행렬의 형태, MAP의 작은 경사, 저차원 수치 적분과 사후 모멘트, 가중 분위수·사후예측의 정의를 확인한다. 아래 가상 클래스와 backward 조각은 이 SciPy 구현에 맞지 않는다.

    ```python
    def test_logistic regression is():
        model = Logistic Regression IS(...)
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

가우스 사전과 베르누이 로지스틱 우도의 사후분포를 SNIS로 근사한다. 사전 제안과 MAP의 라플라스 제안을 비교하고 계수 추론과 사후예측을 구분한다. 웨이트 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
