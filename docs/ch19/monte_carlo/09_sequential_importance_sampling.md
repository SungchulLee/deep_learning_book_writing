# 순차 중요도 샘플링

고정된 파라메터 입자의 웨이트를 새 관측의 가능도(likelihood, 우도라고도 한다)로 갱신한다. 재표집이 웨이트 집중을 줄이는 효과와 입자 다양성을 잃는 효과를 구분하는 순차 베이즈 예제다.

## 1. 코드

??? note "코드 (555줄)"

    ```python
    """
    09_sequential_importance_sampling.py

    고급 단계: 순차 중요도 샘플링(SIS)

    이 단원은 순차 중요도 샘플링을 구현한다.
    데이터를 순차적으로 처리하고 순차 베이즈 추론을 하는 데 쓴다.

    수학적 기초:
    ---------------------
    순차 중요도 샘플링은 새 데이터가 올 때마다 중요도 웨이트를 갱신한다:

    시간 t에 데이터 y₁:t = (y₁, ..., y_t)가 있다

    사후분포: p(θ|y₁:t) ∝ p(y₁:t|θ)p(θ)

    순차 인수분해:
        p(θ|y₁:t) ∝ p(yt|θ, y₁:t₋₁) × p(θ|y₁:t₋₁)

    중요도 웨이트 갱신:
        w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ, y₁:t₋₁) / q_t(θ|y₁:t, θ₁:t₋₁)

    고정 제안 q(θ)에서(적응하지 않을 때):
        w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ)

    웨이트가 시간에 따라 곱해져 다음 문제에 이른다:

    웨이트 퇴화(weight degeneracy) 문제:
    - 웨이트가 점점 더 불균등해진다
    - 시간이 갈수록 ESS가 줄어든다
    - 결국 몇몇 입자가 지배한다

    해결책:
    1. 재표집(입자 필터)
    2. 적응 제안
    3. 방어적 혼합

    응용:
    - 순차 베이즈 갱신
    - 시계열 분석
    - 온라인 학습
    - 입자 필터

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.special import logsumexp
    import seaborn as sns

    np.random.seed(42)
    sns.set_style("whitegrid")


    class SequentialImportanceSampler:
        """
        재표집을 붙일 수 있는 순차 중요도 샘플링.
        """

        def __init__(self, prior_dist, proposal_dist):
            """
            파라메터:
            -----------
            prior_dist : scipy.stats distribution
                사전분포 p(θ)
            proposal_dist : scipy.stats distribution
                제안 분포 q(θ)
            """
            self.prior_dist = prior_dist
            self.proposal_dist = proposal_dist

            # 저장 공간
            self.particles = None
            self.log_weights = None
            self.weights = None
            self.ess_history = []
            self.t = 0

        def initialize(self, n_particles: int):
            """
            제안에서 입자 초기화하기.
            """
            self.n_particles = n_particles
            self.particles = self.proposal_dist.rvs(size=n_particles)

            # 초기 웨이트: w₀(θ) = p(θ)/q(θ)
            self.log_weights = (
                self.prior_dist.logpdf(self.particles) -
                self.proposal_dist.logpdf(self.particles)
            )

            # 정규화
            self._normalize_weights()
            self.t = 0

            # ESS 기록하기
            ess = self.compute_ess()
            self.ess_history.append(ess)

        def update(self, likelihood_fn, y_new):
            """
            새 관측으로 웨이트 갱신하기.

            파라메터:
            -----------
            likelihood_fn : 호출 가능 객체
                함수: likelihood_fn(theta, y) -> 가능도 값
            y_new : 관측
                새 데이터 점
            """
            self.t += 1

            # 로그 웨이트 갱신: log w_t = log w_{t-1} + log p(y_t|θ)
            log_likelihoods = np.array([
                likelihood_fn(theta, y_new) for theta in self.particles
            ])

            self.log_weights += log_likelihoods

            # 정규화
            self._normalize_weights()

            # ESS 기록하기
            ess = self.compute_ess()
            self.ess_history.append(ess)

            return ess

        def _normalize_weights(self):
            """안정을 위해 log-sum-exp로 웨이트 정규화하기."""
            self.log_weights = self.log_weights - logsumexp(self.log_weights)
            self.weights = np.exp(self.log_weights)

        def compute_ess(self):
            """유효 표본 크기 계산하기."""
            return 1.0 / np.sum(self.weights**2)

        def resample(self, threshold=0.5):
            """
            ESS가 임계값 아래로 떨어지면 입자 재표집하기.

            다항 재표집: 웨이트에 따라 입자를
            복원 추출한다.
            """
            ess = self.compute_ess()
            rel_ess = ess / self.n_particles

            if rel_ess < threshold:
                # 다항 재표집
                indices = np.random.choice(
                    self.n_particles,
                    size=self.n_particles,
                    replace=True,
                    p=self.weights
                )

                self.particles = self.particles[indices]

                # 웨이트를 균등하게 되돌리기
                self.log_weights = np.zeros(self.n_particles)
                self.weights = np.ones(self.n_particles) / self.n_particles

                return True  # 재표집함

            return False  # 재표집 없음

        def estimate(self, h_function):
            """
            E[h(θ)|y₁:t]의 가중 추정값 계산하기.
            """
            return np.sum(self.weights * h_function(self.particles))

        def credible_interval(self, alpha=0.95):
            """
            가중 분위수로 신용구간 계산하기.
            """
            sorted_indices = np.argsort(self.particles)
            sorted_particles = self.particles[sorted_indices]
            sorted_weights = self.weights[sorted_indices]

            cumsum = np.cumsum(sorted_weights)

            lower_idx = np.searchsorted(cumsum, (1-alpha)/2)
            upper_idx = np.searchsorted(cumsum, (1+alpha)/2)

            return sorted_particles[lower_idx], sorted_particles[upper_idx]


    # 보기 1: 순차 베이즈 갱신(정규 평균)
    # ===================================================
    print("=" * 70)
    print("EXAMPLE 1: Sequential Bayesian Updating")
    print("=" * 70)

    print("""
    모형: σ² = 1을 아는 yᵢ ~ N(θ, σ²)
    사전분포: θ ~ N(0, 4)
    데이터를 순차적으로 처리하며 ESS가 나빠지는 것 관찰하기.
    """)

    # 참 파라메터
    theta_true = 2.5
    sigma = 1.0

    # 순차 데이터 생성하기
    n_obs = 50
    data_seq = np.random.normal(theta_true, sigma, n_obs)

    print(f"\nTrue θ = {theta_true}")
    print(f"Generated {n_obs} observations sequentially")

    # 사전분포와 제안
    prior = stats.norm(0, 2)
    proposal = stats.norm(0, 2)  # 사전분포를 제안으로 씀

    # 가능도 함수
    def likelihood_normal(theta, y):
        """관측 하나의 로그 가능도."""
        return stats.norm.logpdf(y, theta, sigma)

    # SIS 초기화하기
    n_particles = 2000
    sis = SequentialImportanceSampler(prior, proposal)
    sis.initialize(n_particles)

    print(f"\nInitialized {n_particles} particles")
    print(f"Initial ESS: {sis.ess_history[0]:.1f}")

    # 견주기 위한 해석적 사후분포
    def analytical_posterior(y_data, sigma, mu_0, tau_0):
        """분산을 알 때의 해석적 사후분포 N(μₙ, τₙ²)."""
        n = len(y_data)
        tau_n_sq = 1.0 / (1.0/tau_0**2 + n/sigma**2)
        mu_n = tau_n_sq * (mu_0/tau_0**2 + np.sum(y_data)/sigma**2)
        return mu_n, np.sqrt(tau_n_sq)

    # 데이터를 순차적으로 처리하기
    print("\nSequential Processing:")
    print(f"{'t':>3} {'y_t':>8} {'Post Mean':>10} {'ESS':>8} {'Rel ESS':>8}")
    print("-" * 50)

    posterior_means = []
    posterior_stds = []

    for t, y_t in enumerate(data_seq):
        # 새 관측으로 갱신하기
        ess = sis.update(likelihood_normal, y_t)

        # 사후 평균 추정하기
        post_mean = sis.estimate(lambda x: x)
        posterior_means.append(post_mean)

        # 해석적 사후분포
        mu_n, tau_n = analytical_posterior(data_seq[:t+1], sigma, 0, 2)
        posterior_stds.append(tau_n)

        # 관측 5개마다 출력하기
        if (t+1) % 5 == 0 or t == 0:
            print(f"{t+1:3d} {y_t:8.3f} {post_mean:10.4f} {ess:8.1f} "
                  f"{ess/n_particles:8.1%}")

    # 마지막 비교
    final_analytical_mean, final_analytical_std = analytical_posterior(
        data_seq, sigma, 0, 2
    )

    print(f"\nFinal Estimates:")
    print(f"  True θ: {theta_true:.4f}")
    print(f"  SIS estimate: {posterior_means[-1]:.4f}")
    print(f"  Analytical: {final_analytical_mean:.4f}")
    print(f"  Final ESS: {sis.ess_history[-1]:.1f} ({sis.ess_history[-1]/n_particles:.1%})")

    # ESS 감소 시각화하기
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 칸 1: 시간에 따른 ESS
    ax = axes[0, 0]
    ax.plot(range(len(sis.ess_history)), sis.ess_history, 'b-',
            linewidth=2, label='ESS')
    ax.axhline(n_particles, color='red', linestyle='--', linewidth=2,
               label='n particles', alpha=0.7)
    ax.axhline(n_particles * 0.5, color='orange', linestyle='--', linewidth=1.5,
               label='50% threshold', alpha=0.7)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('Weight Degeneracy Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 2: 사후 평균의 변화
    ax = axes[0, 1]
    ax.plot(range(1, len(posterior_means)+1), posterior_means, 'b-',
            linewidth=2, label='SIS estimate')
    analytical_means = [analytical_posterior(data_seq[:i+1], sigma, 0, 2)[0]
                        for i in range(len(data_seq))]
    ax.plot(range(1, len(analytical_means)+1), analytical_means, 'r--',
            linewidth=2, label='Analytical', alpha=0.7)
    ax.axhline(theta_true, color='green', linestyle=':', linewidth=2,
               label='True θ', alpha=0.7)
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Posterior Mean', fontsize=12)
    ax.set_title('Posterior Mean Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 3: 시점에 따른 웨이트 분포
    ax = axes[1, 0]
    times_to_plot = [0, 10, 25, 49]
    colors_weights = ['blue', 'green', 'orange', 'red']

    # 정해진 시점의 웨이트를 얻으려 다시 실행하기
    sis_temp = SequentialImportanceSampler(prior, proposal)
    sis_temp.initialize(n_particles)
    weights_over_time = [sis_temp.weights.copy()]

    for t, y_t in enumerate(data_seq):
        sis_temp.update(likelihood_normal, y_t)
        if t+1 in times_to_plot:
            weights_over_time.append(sis_temp.weights.copy())

    for idx, (t, weights, color) in enumerate(zip([0] + times_to_plot,
                                                    weights_over_time,
                                                    colors_weights)):
        ax.hist(weights * n_particles, bins=30, alpha=0.4, color=color,
                label=f't={t}', density=True)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=2,
               label='Uniform', alpha=0.5)
    ax.set_xlabel('Weight × n', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Weight Distribution Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 칸 4: 현재의 입자 분포
    ax = axes[1, 1]
    x_range = np.linspace(-2, 6, 1000)

    # 입자 그리기
    ax.hist(sis.particles, bins=50, weights=sis.weights, density=True,
            alpha=0.6, color='steelblue', edgecolor='black',
            label='Weighted particles')

    # 참 사후분포
    true_post_dist = stats.norm(final_analytical_mean, final_analytical_std)
    ax.plot(x_range, true_post_dist.pdf(x_range), 'r-', linewidth=2,
            label='True posterior')

    # 사전분포
    ax.plot(x_range, prior.pdf(x_range), 'g--', linewidth=2,
            label='Prior', alpha=0.5)

    ax.axvline(theta_true, color='orange', linestyle=':', linewidth=2,
               label='True θ')
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Final Posterior Approximation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sis_weight_degeneracy.png',
                dpi=300, bbox_inches='tight')


    # 보기 2: 재표집을 붙인 SIS(입자 필터)
    # ==============================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Sequential IS with Resampling")
    print("=" * 70)

    print("\nResampling when ESS drops below 50% threshold")

    # 새 SIS 초기화하기
    sis_resample = SequentialImportanceSampler(prior, proposal)
    sis_resample.initialize(n_particles)

    resampling_times = []

    print(f"\n{'t':>3} {'ESS':>8} {'Rel ESS':>8} {'Resampled?'}")
    print("-" * 40)

    for t, y_t in enumerate(data_seq):
        sis_resample.update(likelihood_normal, y_t)
        ess = sis_resample.ess_history[-1]

        # 필요하면 재표집하기
        did_resample = sis_resample.resample(threshold=0.5)

        if did_resample:
            resampling_times.append(t+1)

        if (t+1) % 5 == 0 or did_resample:
            resample_str = "YES" if did_resample else ""
            print(f"{t+1:3d} {ess:8.1f} {ess/n_particles:8.1%} {resample_str}")

    print(f"\nResampled {len(resampling_times)} times at t = {resampling_times}")

    # 재표집할 때와 아닐 때의 ESS 견주기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(len(sis.ess_history)), sis.ess_history, 'b-',
            linewidth=2, label='Without resampling')
    ax.plot(range(len(sis_resample.ess_history)), sis_resample.ess_history, 'r-',
            linewidth=2, label='With resampling')
    ax.axhline(n_particles * 0.5, color='green', linestyle='--', linewidth=1.5,
               label='Resample threshold', alpha=0.7)

    for t in resampling_times:
        ax.axvline(t, color='red', linestyle=':', linewidth=1, alpha=0.3)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('ESS: With vs Without Resampling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 웨이트 분포 견주기
    ax = axes[1]
    ax.hist(sis.weights * n_particles, bins=50, alpha=0.5, density=True,
            color='blue', edgecolor='black', label='Without resampling')
    ax.hist(sis_resample.weights * n_particles, bins=50, alpha=0.5, density=True,
            color='red', edgecolor='black', label='With resampling')
    ax.axvline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Weight × n', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Final Weight Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sis_with_resampling.png',
                dpi=300, bbox_inches='tight')


    # 보기 3: 온라인 파라메터 추정
    # ====================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Online Credible Intervals")
    print("=" * 70)

    # 시간에 따른 신용구간 기록하기
    sis_ci = SequentialImportanceSampler(prior, proposal)
    sis_ci.initialize(n_particles)

    credible_intervals = []
    posterior_means_ci = []

    for y_t in data_seq:
        sis_ci.update(likelihood_normal, y_t)
        sis_ci.resample(threshold=0.5)  # 재표집을 붙임

        mean = sis_ci.estimate(lambda x: x)
        ci_lower, ci_upper = sis_ci.credible_interval(alpha=0.95)

        posterior_means_ci.append(mean)
        credible_intervals.append((ci_lower, ci_upper))

    # 시간에 따른 신용구간 그리기
    fig, ax = plt.subplots(figsize=(12, 6))

    t_vals = range(1, len(credible_intervals)+1)
    ci_lower = [ci[0] for ci in credible_intervals]
    ci_upper = [ci[1] for ci in credible_intervals]

    ax.fill_between(t_vals, ci_lower, ci_upper, alpha=0.3, color='blue',
                    label='95% CI')
    ax.plot(t_vals, posterior_means_ci, 'b-', linewidth=2,
            label='Posterior mean')
    ax.axhline(theta_true, color='red', linestyle='--', linewidth=2,
               label='True θ', alpha=0.7)
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('θ', fontsize=12)
    ax.set_title('Sequential Credible Intervals', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sis_credible_intervals.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 순차 중요도 샘플링은 데이터를 조금씩 처리한다:
       - 웨이트가 시간에 따라 곱해진다: w_t = w_{t-1} × p(y_t|θ)
       - 데이터를 모두 다시 처리할 필요가 없다
       - 온라인 응용에 유용하다

    2. 웨이트 퇴화가 근본 문제이다:
       - 시간이 갈수록 ESS가 줄어든다
       - 결국 몇몇 입자가 지배한다
       - 입자가 고정되어 있으면 피할 수 없다

    3. 퇴화 속도:
       - 많은 경우 지수적이다
       - 정보가 많은 데이터에서 더 빠르다
       - 좋은 제안이 있을 때는 더 느리다

    4. 재표집은 퇴화를 완화한다:
       - 웨이트가 큰 입자 복제하기
       - 웨이트가 작은 입자 버리기
       - 웨이트를 균등하게 되돌리기
       - 입자의 다양성을 잃게 한다

    5. 재표집 전략:
       - 임계값 방식: ESS < 임계값이면 재표집
       - 주기적: k스텝마다 재표집
       - 적응적: ESS나 다른 기준에 기반

    6. 재표집의 트레이드오프:
       - 이점: 유효 표본 크기를 유지한다
       - 비용: 입자의 다양성을 잃는다(표본 빈곤화)
       - 비용: 샘플링 변동이 더 생긴다

    7. 입자 필터 = SIS + 재표집:
       - 시계열에서 널리 쓰인다
       - 상태 공간 모형
       - 추적 응용
       - 로봇 공학과 내비게이션

    8. 순차 중요도 샘플링을 언제 쓰나:
       - 데이터가 순차적으로 들어온다
       - 온라인 학습 상황
       - 계산 제약(데이터를 모두 다시 처리할 수 없다)
       - 시계열 분석

    9. 실전에서의 고려 사항:
       - ESS를 계속 모니터링하기
       - ESS < 0.5n(또는 0.3n)이면 재표집하기
       - 될 수 있으면 적응 제안 쓰기
       - 아주 긴 시퀀스에는 입자 MCMC 고려하기

    10. 한계:
        - 재표집에 따른 표본 빈곤화
        - 고정된 입자는 새 영역을 탐색할 수 없다
        - 배치 방법만큼 강건하지는 않다
        - 재표집 임계값을 세심하게 조정해야 한다

    11. 대안과 개선:
        - 적응 제안(고정 제안보다 낫다)
        - 보조 입자 필터
        - 입자 MCMC(재활성화)
        - 라오-블랙웰화(할 수 있을 때)
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (129줄)"

    ```
    ======================================================================
    EXAMPLE 1: Sequential Bayesian Updating
    ======================================================================

    모형: σ² = 1을 아는 yᵢ ~ N(θ, σ²)
    사전분포: θ ~ N(0, 4)
    데이터를 순차적으로 처리하며 ESS가 나빠지는 것 관찰하기.


    True θ = 2.5
    Generated 50 observations sequentially

    Initialized 2000 particles
    Initial ESS: 2000.0

    Sequential Processing:
      t      y_t  Post Mean      ESS  Rel ESS
    --------------------------------------------------
      1    2.997     2.4225    554.9    27.7%
      5    2.266     2.8416    226.4    11.3%
     10    3.043     2.9023    155.9     7.8%
     15    0.775     2.4519    166.6     8.3%
     20    1.088     2.2886    160.4     8.0%
     25    1.956     2.3017    143.4     7.2%
     30    2.208     2.2843    134.1     6.7%
     35    3.323     2.3369    119.3     6.0%
     40    2.697     2.2638    120.9     6.0%
     45    1.021     2.2682    114.7     5.7%
     50    0.737     2.2619    110.5     5.5%

    Final Estimates:
      True θ: 2.5000
      SIS estimate: 2.2619
      Analytical: 2.2632
      Final ESS: 110.5 (5.5%)

    ======================================================================
    EXAMPLE 2: Sequential IS with Resampling
    ======================================================================

    Resampling when ESS drops below 50% threshold

      t      ESS  Rel ESS Resampled?
    ----------------------------------------
      1    511.2    25.6% YES
      5   1140.0    57.0% 
      7    932.4    46.6% YES
     10   1893.7    94.7% 
     15    720.5    36.0% YES
     20   1382.7    69.1% 
     25   1437.7    71.9% 
     30   1284.5    64.2% 
     35   1451.4    72.6% 
     40   1067.9    53.4% 
     45   1039.0    52.0% 
     46    988.5    49.4% YES
     50   1993.4    99.7% 

    Resampled 4 times at t = [1, 7, 15, 46]

    ======================================================================
    EXAMPLE 3: Online Credible Intervals
    ======================================================================

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 순차 중요도 샘플링은 데이터를 조금씩 처리한다:
       - 웨이트가 시간에 따라 곱해진다: w_t = w_{t-1} × p(y_t|θ)
       - 데이터를 모두 다시 처리할 필요가 없다
       - 온라인 응용에 유용하다

    2. 웨이트 퇴화가 근본 문제이다:
       - 시간이 갈수록 ESS가 줄어든다
       - 결국 몇몇 입자가 지배한다
       - 입자가 고정되어 있으면 피할 수 없다

    3. 퇴화 속도:
       - 많은 경우 지수적이다
       - 정보가 많은 데이터에서 더 빠르다
       - 좋은 제안이 있을 때는 더 느리다

    4. 재표집은 퇴화를 완화한다:
       - 웨이트가 큰 입자 복제하기
       - 웨이트가 작은 입자 버리기
       - 웨이트를 균등하게 되돌리기
       - 입자의 다양성을 잃게 한다

    5. 재표집 전략:
       - 임계값 방식: ESS < 임계값이면 재표집
       - 주기적: k스텝마다 재표집
       - 적응적: ESS나 다른 기준에 기반

    6. 재표집의 트레이드오프:
       - 이점: 유효 표본 크기를 유지한다
       - 비용: 입자의 다양성을 잃는다(표본 빈곤화)
       - 비용: 샘플링 변동이 더 생긴다

    7. 입자 필터 = SIS + 재표집:
       - 시계열에서 널리 쓰인다
       - 상태 공간 모형
       - 추적 응용
       - 로봇 공학과 내비게이션

    8. 순차 중요도 샘플링을 언제 쓰나:
       - 데이터가 순차적으로 들어온다
       - 온라인 학습 상황
       - 계산 제약(데이터를 모두 다시 처리할 수 없다)
       - 시계열 분석

    9. 실전에서의 고려 사항:
       - ESS를 계속 모니터링하기
       - ESS < 0.5n(또는 0.3n)이면 재표집하기
       - 될 수 있으면 적응 제안 쓰기
       - 아주 긴 시퀀스에는 입자 MCMC 고려하기

    10. 한계:
        - 재표집에 따른 표본 빈곤화
        - 고정된 입자는 새 영역을 탐색할 수 없다
        - 배치 방법만큼 강건하지는 않다
        - 재표집 임계값을 세심하게 조정해야 한다

    11. 대안과 개선:
        - 적응 제안(고정 제안보다 낫다)
        - 보조 입자 필터
        - 입자 MCMC(재활성화)
        - 라오-블랙웰화(할 수 있을 때)

    ```


## 2. 논의

### 고정 파라메터의 순차 갱신

초기 입자 θ_i~q에 p(θ_i)/q(θ_i)를 부여하고, 조건부 독립 관측마다 이전 웨이트에 p(y_t|θ_i)를 곱한다. likelihood_fn이라는 이름과 달리 update는 로그가능도를 반환하는 함수를 요구한다. 코드의 정규화는 누적 로그 웨이트에서 log-sum-exp를 빼므로 양의 상수배를 제거한다.

이 구현에는 동적 상태의 전이·이동 단계가 없다. 상태공간 모형의 일반적인 입자 필터 전체 구현이 아니라 정적 파라메터의 순차 재가중·재표집 예제다. 사전 N(0,4)는 분산 4, SciPy norm(0,2)는 표준편차 2로 같은 분포를 뜻한다.

### 재표집과 실제 정보량

ESS가 임계값 아래이면 multinomial 재표집으로 입자를 복제하고 웨이트를 1/N으로 되돌린다. 직후 웨이트 ESS는 N이지만 복제된 입자가 독립적인 새 정보를 주는 것은 아니다. 재표집은 없는 영역의 입자를 새로 만들 수 없고 정적 파라메터에서는 다양성 손실이 누적될 수 있다. 사후를 보존하는 이동 단계나 별도의 새 입자 제안을 고려할 수 있다.

ess_history는 update 직후이면서 resample 이전 값을 저장한다. 따라서 그래프의 ESS와 실제 재표집 직후 웨이트 ESS는 다를 수 있다. initialize는 history를 지우지 않으므로 같은 객체의 재실행 기록에도 주의한다. 데이터를 더 관측할 때 ESS가 항상 단조 감소하거나 고정된 지수 속도로 감소하는 것은 아니다.

### 구간과 표시 범위

credible_interval은 입자값을 정렬하고 그 순서의 누적 웨이트에서 분위수를 찾는다. 이는 입자 근사분포의 등꼬리 신용구간이며 평균 추정의 몬테카를로 신뢰구간이 아니다. 0<alpha<1, 유효한 웨이트와 누적합의 반올림 경계를 확인해야 한다. 해석적 정규 사후를 기준으로 평균·분산·구간을 비교할 수 있다.

웨이트 변화 그림은 관측 수와 0부터 시작하는 인덱스를 혼용한다. 저장한 웨이트와 zip의 t 라벨이 어긋나므로 표시된 모든 시점이 정확하다고 해석하지 않는다. 0.5N 재표집 기준은 선택한 휴리스틱이며 수렴·정확도 보증이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로그 웨이트 갱신식을 설명하라.

</div>

??? success "풀이"

    초기 log w=log p−log q에 관측별 log p(y_t|θ)를 더하고 공통 log-sum-exp를 뺀다. 조건부 독립성과 정적인 θ라는 모형 가정을 사용한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
update와 credible_interval의 입력 조건은 무엇인가?

</div>

??? success "풀이"

    입자를 먼저 초기화하고 likelihood_fn은 각 θ·관측에 로그가능도를 반환해야 한다. 입자 수는 양수, 구간 질량 alpha는 0과 1 사이이며 로그 웨이트가 모두 −∞인 경우를 처리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
재표집 직후 ESS=N이면 정보 손실이 회복되는가?

</div>

??? success "풀이"

    웨이트만 균등해진다. 복제 입자는 같고 새 영역은 탐색하지 못하므로 독립 정보 N개가 생긴 것이 아니다. 고유 입자 수와 함수별 오차도 살펴야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 네트워크 시험 대신 순차 샘플러의 검증 기준을 제시하라.

</div>

??? success "풀이"

    같은 데이터의 배치·순차 로그가능도 웨이트 일치, 해석적 정규 사후, 정렬된 가중 분위수와 재표집 전후 웨이트·입자 중복을 확인한다. 아래 model(x)·backward 인터페이스는 이 클래스에 없다.

    ```python
    def test_sequentialimportancesampler():
        model = SequentialImportanceSampler(...)
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

고정된 파라메터 입자의 웨이트를 새 관측의 가능도로 갱신한다. 재표집이 웨이트 집중을 줄이는 효과와 입자 다양성을 잃는 효과를 구분하는 순차 베이즈 예제다. 웨이트 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
