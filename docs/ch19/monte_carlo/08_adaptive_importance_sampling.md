# 적응적 중요도 샘플링

이 예제는 앞 단계의 가중 표본으로 다음 혼합 제안을 만드는 적응적 중요도 샘플링을 구현한다. 여기서 AIS는 adaptive importance sampling이며 annealed importance sampling과 구분한다.

## 1. 코드

??? note "코드 (586줄)"

    ```python
    """
    08_adaptive_importance_sampling.py

    고급 단계: 적응적 중요도 샘플링(AIS)

    이 단원은 적응적 중요도 샘플링을 구현한다. 여기서 제안
    분포는 앞서 뽑은 표본에 따라 반복적으로 갱신된다.

    수학적 기초:
    ---------------------
    표준 중요도 샘플링은 고정 제안 q(θ)를 쓴다. AIS는 q를 반복적으로 개선한다:

    알고리즘(집단 몬테카를로 - PMC):
    1. 초기화: q₀(θ) 선택
    2. t = 1, 2, ..., T에 대해:
       가) i=1,...,n에 대해 θᵢᵗ ~ qₜ₋₁(θ) 샘플링
       나) 웨이트 wᵢᵗ = π(θᵢᵗ)/qₜ₋₁(θᵢᵗ) 계산하기
       다) 제안 갱신하기: {θᵢᵗ, wᵢᵗ}에 바탕을 둔 qₜ(θ)
    3. 중요도 웨이트를 붙여 최종 표본 반환하기

    흔한 갱신 전략:
    1. 웨이트가 큰 표본에 중심을 맞춘 가우스 혼합
    2. 가중 커널 밀도 추정
    3. 파라메터 적합(이를테면 평균과 공분산을 갱신한 가우스)

    이점:
    - 타깃 분포에 자동으로 적응한다
    - 봉우리가 여럿인 구조를 찾아낼 수 있다
    - 대체로 고정 제안보다 ESS가 높다

    어려움:
    - 웨이트를 올바르게 계산하려면 주의해야 한다
    - 국소 모드에 갇힐 수 있다
    - 제안을 갱신하는 계산 비용

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.special import logsumexp
    import seaborn as sns
    from typing import Tuple, List, Callable

    np.random.seed(42)
    sns.set_style("whitegrid")


    class AdaptiveImportanceSampler:
        """
        혼합 제안을 쓰는 적응적 중요도 샘플링.
        """

        def __init__(self, target_log_density: Callable, dim: int,
                     n_components: int = 5, initial_scale: float = 2.0):
            """
            파라메터:
            -----------
            target_log_density : function
                정규화하지 않은 타깃 밀도의 로그
            dim : int
                파라메터 공간의 차원
            n_components : int
                혼합 성분의 개수
            initial_scale : float
                제안 성분의 초기 스케일
            """
            self.target_log_density = target_log_density
            self.dim = dim
            self.n_components = n_components
            self.initial_scale = initial_scale

            # 반복 기록을 담을 저장 공간
            self.samples_history = []
            self.weights_history = []
            self.ess_history = []

        def initialize_proposal(self, initial_mean: np.ndarray = None):
            """
            initial_mean에 중심을 맞춘 N(μ, σ²I)로 혼합 제안 초기화하기.
            """
            if initial_mean is None:
                initial_mean = np.zeros(self.dim)

            # 넓은 가우스 하나로 시작
            self.mixture_means = [initial_mean.copy()]
            self.mixture_covs = [np.eye(self.dim) * self.initial_scale**2]
            self.mixture_weights = [1.0]

        def proposal_log_density(self, theta: np.ndarray) -> float:
            """
            theta에서 현재 혼합 제안의 로그 밀도 평가.

            q(θ) = Σⱼ αⱼ N(θ|μⱼ, Σⱼ)
            """
            # 성분마다 평가하기
            log_densities = []
            for mean, cov, weight in zip(self.mixture_means, self.mixture_covs, 
                                          self.mixture_weights):
                # 다변량 정규의 로그 밀도
                component = stats.multivariate_normal(mean, cov)
                log_densities.append(np.log(weight + 1e-300) + component.logpdf(theta))

            # 수치 안정을 위한 log-sum-exp
            return logsumexp(log_densities)

        def sample_from_proposal(self, n_samples: int) -> np.ndarray:
            """
            현재 혼합 제안에서 샘플링하기.
            """
            samples = []

            # 성분 배정 샘플링
            component_probs = np.array(self.mixture_weights)
            component_probs /= component_probs.sum()
            components = np.random.choice(len(self.mixture_means), 
                                         size=n_samples, p=component_probs)

            # 배정된 성분에서 샘플링
            for i in range(n_samples):
                comp_idx = components[i]
                mean = self.mixture_means[comp_idx]
                cov = self.mixture_covs[comp_idx]
                sample = np.random.multivariate_normal(mean, cov)
                samples.append(sample)

            return np.array(samples)

        def update_proposal(self, samples: np.ndarray, weights: np.ndarray, 
                           method: str = 'resample'):
            """
            가중 표본에 따라 혼합 제안 갱신하기.

            방법:
            --------
            'resample': 재표집한 입자에 혼합을 적합한다
            'weighted_means': 웨이트가 큰 표본에 성분을 배치한다
            """
            # 웨이트 정규화하기
            normalized_weights = weights / np.sum(weights)

            if method == 'resample':
                # 웨이트에 따라 재표집하기
                indices = np.random.choice(len(samples), size=self.n_components,
                                          replace=True, p=normalized_weights)
                selected_samples = samples[indices]

                # 모든 표본의 경험 공분산 계산하기
                # 변수가 하나뿐이면 np.cov가 0차원 배열을 반환하므로
                # 아래의 += np.eye(dim) 가 브로드캐스트되지 못한다. 2차원으로 맞춰 둔다.
                weighted_cov = np.atleast_2d(
                    np.cov(samples.T, aweights=normalized_weights))

                # 수치 안정을 위해 작은 정칙화 항 더하기
                weighted_cov += np.eye(self.dim) * 1e-4

                # 탐색-활용 트레이드오프를 위해 공분산 축소하기
                shrinkage = 0.7
                weighted_cov *= shrinkage

                # 혼합 성분 갱신하기
                self.mixture_means = [s for s in selected_samples]
                self.mixture_covs = [weighted_cov for _ in range(self.n_components)]
                self.mixture_weights = [1.0/self.n_components] * self.n_components

            elif method == 'weighted_means':
                # 웨이트가 큰 표본 고르기
                top_indices = np.argsort(normalized_weights)[-self.n_components:]

                # 선택한 표본을 평균으로 쓰기
                self.mixture_means = [samples[i] for i in top_indices]

                # 적응 공분산 계산하기
                # 변수가 하나뿐이면 np.cov가 0차원 배열을 반환하므로
                # 아래의 += np.eye(dim) 가 브로드캐스트되지 못한다. 2차원으로 맞춰 둔다.
                weighted_cov = np.atleast_2d(
                    np.cov(samples.T, aweights=normalized_weights))
                weighted_cov += np.eye(self.dim) * 1e-4
                weighted_cov *= 0.5  # 안정을 위해 축소

                self.mixture_covs = [weighted_cov for _ in range(self.n_components)]

                # 중요도 웨이트에 비례하는 웨이트
                selected_weights = normalized_weights[top_indices]
                self.mixture_weights = (selected_weights / selected_weights.sum()).tolist()

        def run(self, n_samples: int, n_iterations: int, 
                update_method: str = 'resample',
                initial_mean: np.ndarray = None,
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
            """
            적응적 중요도 샘플링 실행하기.

            반환값:
            --------
            samples : 형태가 (n_iterations * n_samples, dim)인 배열
            weights : 정규화한 웨이트의 배열
            """
            self.initialize_proposal(initial_mean)

            all_samples = []
            all_log_weights = []

            if verbose:
                print(f"\nRunning Adaptive IS: {n_iterations} iterations, "
                      f"{n_samples} samples each")
                print("=" * 60)

            for t in range(n_iterations):
                # 현재 제안에서 샘플링
                samples = self.sample_from_proposal(n_samples)

                # 로그 중요도 웨이트 계산하기
                log_weights = np.array([
                    self.target_log_density(s) - self.proposal_log_density(s)
                    for s in samples
                ])

                # 표본과 웨이트 저장
                all_samples.append(samples)
                all_log_weights.append(log_weights)

                # 이번 반복의 웨이트 정규화하기
                log_weights_normalized = log_weights - logsumexp(log_weights)
                weights = np.exp(log_weights_normalized)

                # 진단을 위한 ESS 계산하기
                ess = 1.0 / np.sum(weights**2)
                self.ess_history.append(ess)

                if verbose:
                    print(f"Iteration {t+1:3d}: ESS = {ess:7.1f} ({ess/n_samples:5.1%})")

                # 제안 갱신하기(마지막 반복은 빼고)
                if t < n_iterations - 1:
                    self.update_proposal(samples, np.exp(log_weights), 
                                        method=update_method)

                self.samples_history.append(samples)
                self.weights_history.append(weights)

            # 표본을 모두 합치고 최종 웨이트 다시 계산하기
            all_samples = np.vstack(all_samples)
            all_log_weights = np.concatenate(all_log_weights)

            # 최종 정규화한 웨이트
            final_log_weights = all_log_weights - logsumexp(all_log_weights)
            final_weights = np.exp(final_log_weights)

            return all_samples, final_weights


    # 보기 1: 봉우리 둘인 1차원 타깃
    # ==========================
    print("=" * 70)
    print("EXAMPLE 1: Adaptive IS for 1D Bimodal Distribution")
    print("=" * 70)

    # 타깃: 가우스 둘의 혼합
    def target_log_density_1d(theta):
        """봉우리 둘인 타깃: 0.3*N(-3,1) + 0.7*N(4,1.5)"""
        if theta.ndim == 0 or len(theta) == 1:
            theta = np.atleast_1d(theta)

        # 첫 봉우리: N(-3, 1)
        log_p1 = stats.norm.logpdf(theta[0], -3, 1) + np.log(0.3)

        # 둘째 봉우리: N(4, 1.5)
        log_p2 = stats.norm.logpdf(theta[0], 4, 1.5) + np.log(0.7)

        return logsumexp([log_p1, log_p2])

    # 그리기 위한 참 타깃
    def true_target_1d(x):
        return 0.3 * stats.norm.pdf(x, -3, 1) + 0.7 * stats.norm.pdf(x, 4, 1.5)

    # 적응적 중요도 샘플링 실행하기
    sampler_1d = AdaptiveImportanceSampler(
        target_log_density=target_log_density_1d,
        dim=1,
        n_components=3,
        initial_scale=3.0
    )

    samples_1d, weights_1d = sampler_1d.run(
        n_samples=200,
        n_iterations=10,
        update_method='resample',
        initial_mean=np.array([0.0])
    )

    final_ess = 1.0 / np.sum(weights_1d**2)
    print(f"\nFinal ESS: {final_ess:.1f} out of {len(samples_1d)} samples")
    print(f"Efficiency: {final_ess/len(samples_1d):.1%}")

    # 적응 과정 시각화하기
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()

    x_plot = np.linspace(-8, 10, 1000)
    iterations_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for idx, iter_num in enumerate(iterations_to_plot):
        ax = axes[idx]

        # 참 타깃 그리기
        ax.plot(x_plot, true_target_1d(x_plot), 'r-', linewidth=2, 
                label='Target', alpha=0.7)

        # 이번 반복의 표본 그리기
        if iter_num < len(sampler_1d.samples_history):
            samples_iter = sampler_1d.samples_history[iter_num]
            weights_iter = sampler_1d.weights_history[iter_num]

            # 표본의 히스토그램
            ax.hist(samples_iter.flatten(), bins=30, density=True, alpha=0.5,
                    color='steelblue', edgecolor='black', linewidth=0.5)

            # 웨이트로 색칠한 표본 산점도
            y_pos = np.zeros(len(samples_iter))
            scatter = ax.scatter(samples_iter.flatten(), y_pos, 
                               c=weights_iter*len(weights_iter), 
                               cmap='hot', s=50, alpha=0.6, 
                               edgecolors='black', linewidth=0.5)

            ess = sampler_1d.ess_history[iter_num]
            ax.set_title(f'Iteration {iter_num+1}: ESS={ess:.0f}',
                        fontsize=11, fontweight='bold')

        ax.set_xlim([-8, 10])
        ax.set_ylim([0, 0.3])
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('adaptive_1d_evolution.png',
                dpi=300, bbox_inches='tight')


    # 보기 2: 바나나 형태 2차원 분포
    # ======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Adaptive IS for 2D Banana Distribution")
    print("=" * 70)

    def target_log_density_banana(theta):
        """
        바나나 형태 분포(로젠브록과 비슷함).

        p(θ₁, θ₂) ∝ exp(-0.5[(θ₁-2)²/4 + (θ₂-θ₁²)²])
        """
        theta = np.atleast_1d(theta)
        theta1, theta2 = theta[0], theta[1]

        term1 = -0.5 * (theta1 - 2)**2 / 4.0
        term2 = -0.5 * (theta2 - theta1**2)**2

        return term1 + term2

    # 적응적 중요도 샘플링 실행하기
    sampler_banana = AdaptiveImportanceSampler(
        target_log_density=target_log_density_banana,
        dim=2,
        n_components=8,
        initial_scale=3.0
    )

    samples_banana, weights_banana = sampler_banana.run(
        n_samples=300,
        n_iterations=15,
        update_method='resample',
        initial_mean=np.array([0.0, 0.0])
    )

    final_ess_banana = 1.0 / np.sum(weights_banana**2)
    print(f"\nFinal ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")

    # 2차원 적응 과정 시각화하기
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    iterations_2d = [0, 1, 2, 4, 6, 8, 10, 14]

    # 참 밀도를 위한 격자 만들기
    x1_grid = np.linspace(-2, 6, 100)
    x2_grid = np.linspace(-5, 25, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = np.exp(target_log_density_banana(np.array([X1[i,j], X2[i,j]])))

    for idx, iter_num in enumerate(iterations_2d):
        ax = axes[idx]

        # 참 타깃의 등고선
        ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)

        # 표본 그리기
        if iter_num < len(sampler_banana.samples_history):
            samples_iter = sampler_banana.samples_history[iter_num]
            weights_iter = sampler_banana.weights_history[iter_num]

            scatter = ax.scatter(samples_iter[:, 0], samples_iter[:, 1],
                               c=weights_iter*len(weights_iter), cmap='viridis',
                               s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

            ess = sampler_banana.ess_history[iter_num]
            ax.set_title(f'Iter {iter_num+1}: ESS={ess:.0f}',
                        fontsize=11, fontweight='bold')

        ax.set_xlabel('θ₁', fontsize=10)
        ax.set_ylabel('θ₂', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_2d_banana.png',
                dpi=300, bbox_inches='tight')


    # 보기 3: 고정 제안과 견주기
    # =======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Adaptive vs Fixed Proposal IS")
    print("=" * 70)

    # 고정 제안의 중요도 샘플링
    n_fixed_samples = 3000  # 적응 방식과 전체가 같음(반복 15번 × 표본 200개)
    fixed_proposal = stats.multivariate_normal([0, 0], [[9, 0], [0, 9]])
    samples_fixed = fixed_proposal.rvs(size=n_fixed_samples)

    # 고정 제안의 웨이트 계산하기
    log_weights_fixed = np.array([
        target_log_density_banana(s) - fixed_proposal.logpdf(s)
        for s in samples_fixed
    ])
    log_weights_fixed_norm = log_weights_fixed - logsumexp(log_weights_fixed)
    weights_fixed = np.exp(log_weights_fixed_norm)

    ess_fixed = 1.0 / np.sum(weights_fixed**2)

    print(f"\nFixed Proposal IS:")
    print(f"  ESS: {ess_fixed:.1f} out of {n_fixed_samples} samples")
    print(f"  Efficiency: {ess_fixed/n_fixed_samples:.1%}")

    print(f"\nAdaptive IS:")
    print(f"  ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")
    print(f"  Efficiency: {final_ess_banana/len(samples_banana):.1%}")

    print(f"\nImprovement: {final_ess_banana/ess_fixed:.2f}x better ESS")

    # 비교를 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 고정 제안
    ax = axes[0]
    ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
    scatter = ax.scatter(samples_fixed[:, 0], samples_fixed[:, 1],
                        c=weights_fixed*len(weights_fixed), cmap='viridis',
                        s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
    ax.set_title(f'Fixed Proposal: ESS={ess_fixed:.0f} ({ess_fixed/n_fixed_samples:.1%})',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    plt.colorbar(scatter, ax=ax, label='Weight × n')

    # 적응 제안
    ax = axes[1]
    ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
    scatter = ax.scatter(samples_banana[:, 0], samples_banana[:, 1],
                        c=weights_banana*len(weights_banana), cmap='viridis',
                        s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
    ax.set_title(f'Adaptive IS: ESS={final_ess_banana:.0f} ({final_ess_banana/len(samples_banana):.1%})',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    plt.colorbar(scatter, ax=ax, label='Weight × n')

    plt.tight_layout()
    plt.savefig('fixed_vs_adaptive.png',
                dpi=300, bbox_inches='tight')


    # 보기 4: 반복에 따른 ESS의 변화
    # ======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Tracking ESS Improvement")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1차원 경우
    ax = axes[0]
    iterations = np.arange(1, len(sampler_1d.ess_history) + 1)
    ess_values = sampler_1d.ess_history
    ax.plot(iterations, ess_values, 'o-', linewidth=2, markersize=8,
            color='steelblue', label='ESS')
    ax.axhline(200, color='red', linestyle='--', linewidth=2, 
              label='n per iteration', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('1D Bimodal: ESS Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # 2차원 경우
    ax = axes[1]
    iterations_2d = np.arange(1, len(sampler_banana.ess_history) + 1)
    ess_values_2d = sampler_banana.ess_history
    ax.plot(iterations_2d, ess_values_2d, 'o-', linewidth=2, markersize=8,
            color='darkgreen', label='ESS')
    ax.axhline(300, color='red', linestyle='--', linewidth=2,
              label='n per iteration', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('2D Banana: ESS Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('ess_evolution.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 적응적 중요도 샘플링은 제안을 반복적으로 개선한다:
       - 넓은 초기 제안으로 시작하기
       - 가중 표본에 따라 갱신하기
       - 확률이 높은 영역으로 수렴한다

    2. 혼합 제안이 효과적이다:
       - 성분이 여럿이면 복잡한 형태를 덮는다
       - 봉우리 여럿을 잡아낼 수 있다
       - 탐색과 활용의 균형

    3. 갱신 전략:
       - 재표집: 웨이트 분포에서 뽑기
       - 가중 평균: 웨이트가 큰 표본에 중심 맞추기
       - 둘 다 형태를 잡는 데 경험 공분산을 쓴다

    4. ESS는 보통 반복할수록 나아진다:
       - 초반: ESS가 낮고 탐색 중
       - 중반: 제안이 적응하면서 ESS가 커진다
       - 후반: ESS가 가장 좋은 값에서 정체된다

    5. 고정 제안보다 나은 점:
       - 좋은 제안을 미리 알 필요가 없다
       - 타깃의 형태에 자동으로 적응한다
       - ESS를 훨씬 높일 수 있다
       - 봉우리가 여럿인 타깃을 다룬다

    6. 계산 측면의 고려 사항:
       - 고정 제안 중요도 샘플링보다 표본마다 더 비싸다
       - 그러나 같은 정확도에 표본이 더 적게 든다
       - 트레이드오프: 적응 비용과 ESS 개선

    7. 적응적 중요도 샘플링을 언제 쓰나:
       - 타깃의 형태를 모를 때
       - 복잡하거나 봉우리가 여럿인 분포
       - 고정 제안의 ESS가 나쁠 때
       - 순차 의사결정 문제

    8. 실전 요령:
       - 넓은 초기 제안으로 시작하기
       - 혼합 성분 5개에서 10개 쓰기
       - ESS의 수렴 모니터링하기
       - 과신을 피하려고 공분산 축소하기
       - 보통 ESS > 0.3n이면 아주 좋다

    9. MCMC와 견주기:
       - AIS: 독립 표본, 번인(burn-in) 없음
       - MCMC: 표본이 서로 상관되어 있고 번인이 필요하다
       - 좋은 제안을 학습할 수 있으면 AIS가 낫다
       - 차원이 아주 높으면 MCMC가 낫다
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (116줄)"

    ```
    ======================================================================
    EXAMPLE 1: Adaptive IS for 1D Bimodal Distribution
    ======================================================================

    Running Adaptive IS: 10 iterations, 200 samples each
    ============================================================
    Iteration   1: ESS =    90.1 (45.0%)
    Iteration   2: ESS =    55.6 (27.8%)
    Iteration   3: ESS =    78.2 (39.1%)
    Iteration   4: ESS =   116.8 (58.4%)
    Iteration   5: ESS =    26.5 (13.2%)
    Iteration   6: ESS =    52.2 (26.1%)
    Iteration   7: ESS =   125.4 (62.7%)
    Iteration   8: ESS =   114.4 (57.2%)
    Iteration   9: ESS =    86.0 (43.0%)
    Iteration  10: ESS =   107.3 (53.7%)

    Final ESS: 658.5 out of 2000 samples
    Efficiency: 32.9%

    ======================================================================
    EXAMPLE 2: Adaptive IS for 2D Banana Distribution
    ======================================================================

    Running Adaptive IS: 15 iterations, 300 samples each
    ============================================================
    Iteration   1: ESS =     9.6 ( 3.2%)
    Iteration   2: ESS =    60.2 (20.1%)
    Iteration   3: ESS =    66.3 (22.1%)
    Iteration   4: ESS =     4.3 ( 1.4%)
    Iteration   5: ESS =    64.2 (21.4%)
    Iteration   6: ESS =   103.9 (34.6%)
    Iteration   7: ESS =   109.9 (36.6%)
    Iteration   8: ESS =    95.5 (31.8%)
    Iteration   9: ESS =    31.4 (10.5%)
    Iteration  10: ESS =    84.8 (28.3%)
    Iteration  11: ESS =   108.4 (36.1%)
    Iteration  12: ESS =    53.1 (17.7%)
    Iteration  13: ESS =    62.8 (20.9%)
    Iteration  14: ESS =   125.9 (42.0%)
    Iteration  15: ESS =     1.6 ( 0.5%)

    Final ESS: 23.7 out of 4500 samples

    ======================================================================
    EXAMPLE 3: Adaptive vs Fixed Proposal IS
    ======================================================================

    Fixed Proposal IS:
      ESS: 207.6 out of 3000 samples
      Efficiency: 6.9%

    Adaptive IS:
      ESS: 23.7 out of 4500 samples
      Efficiency: 0.5%

    Improvement: 0.11x better ESS

    ======================================================================
    EXAMPLE 4: Tracking ESS Improvement
    ======================================================================

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 적응적 중요도 샘플링은 제안을 반복적으로 개선한다:
       - 넓은 초기 제안으로 시작하기
       - 가중 표본에 따라 갱신하기
       - 확률이 높은 영역으로 수렴한다

    2. 혼합 제안이 효과적이다:
       - 성분이 여럿이면 복잡한 형태를 덮는다
       - 봉우리 여럿을 잡아낼 수 있다
       - 탐색과 활용의 균형

    3. 갱신 전략:
       - 재표집: 웨이트 분포에서 뽑기
       - 가중 평균: 웨이트가 큰 표본에 중심 맞추기
       - 둘 다 형태를 잡는 데 경험 공분산을 쓴다

    4. ESS는 보통 반복할수록 나아진다:
       - 초반: ESS가 낮고 탐색 중
       - 중반: 제안이 적응하면서 ESS가 커진다
       - 후반: ESS가 가장 좋은 값에서 정체된다

    5. 고정 제안보다 나은 점:
       - 좋은 제안을 미리 알 필요가 없다
       - 타깃의 형태에 자동으로 적응한다
       - ESS를 훨씬 높일 수 있다
       - 봉우리가 여럿인 타깃을 다룬다

    6. 계산 측면의 고려 사항:
       - 고정 제안 중요도 샘플링보다 표본마다 더 비싸다
       - 그러나 같은 정확도에 표본이 더 적게 든다
       - 트레이드오프: 적응 비용과 ESS 개선

    7. 적응적 중요도 샘플링을 언제 쓰나:
       - 타깃의 형태를 모를 때
       - 복잡하거나 봉우리가 여럿인 분포
       - 고정 제안의 ESS가 나쁠 때
       - 순차 의사결정 문제

    8. 실전 요령:
       - 넓은 초기 제안으로 시작하기
       - 혼합 성분 5개에서 10개 쓰기
       - ESS의 수렴 모니터링하기
       - 과신을 피하려고 공분산 축소하기
       - 보통 ESS > 0.3n이면 아주 좋다

    9. MCMC와 견주기:
       - AIS: 독립 표본, 번인(burn-in) 없음
       - MCMC: 표본이 서로 상관되어 있고 번인이 필요하다
       - 좋은 제안을 학습할 수 있으면 AIS가 낫다
       - 차원이 아주 높으면 MCMC가 낫다

    ```


## 2. 논의

### 적응 단계와 웨이트

각 반복에서 과거에 결정한 q_t로 표본을 생성하고 γ/q_t를 계산한 뒤 다음 제안을 갱신한다. resample 방식은 가중 재표집한 위치를 성분 중심으로 쓰며, weighted_means 방식은 이름과 달리 가중 평균이 아니라 웨이트가 큰 표본 위치를 선택한다. 두 방식 모두 가중 공분산에 작은 대각 보정을 더한다.

최종 결과는 각 시점의 원시 로그 웨이트를 합친 후 한 번에 정규화한다. 단계마다 정규화한 웨이트를 단순 연결하는 방식과 다르다. 단, 제안이 과거 표본에 의존하므로 모든 반복의 표본을 무조건 독립인 동일분포 표본으로 볼 수 없고, 고정 제안용 분산·ESS 해석을 그대로 적용해서도 안 된다. 일치성에는 적응 규칙과 적률 등에 맞는 조건이 필요하다.

### 안정성과 탐색

run의 정규화는 log-sum-exp를 사용하지만 제안 갱신에는 exp(log_weights)를 직접 넘긴다. 이 부분에서 언더플로·오버플로가 발생할 수 있다. 표본 수·차원·성분 수, 유효한 갱신 방식, 가중 공분산과 유한한 로그 웨이트를 확인해야 한다. 같은 객체를 다시 run해도 history 목록은 초기화되지 않아 이전 실행이 섞일 수 있다.

공분산에 0.7·0.5를 곱하는 것은 탐색 폭을 줄이는 휴리스틱이다. 과신을 피하려고 축소한다는 출력 설명은 부정확하며, 방문하지 못한 모드를 더 놓칠 수 있다. ESS가 반복마다 증가하거나 모든 모드를 찾는다는 보장은 없다. 초기 폭·방어적 성분·최소 공분산과 여러 시작점을 검토할 수 있다.

### 예제의 해석과 비용

1차원 혼합 목표의 정확한 평균은 0.3×(−3)+0.7×4=1.9다. 바나나 목표는 θ₁~N(2,2²), θ₂|θ₁~N(θ₁²,1)이므로 평균은 (2,8)이다. 가중 평균을 이런 독립 기준과 비교하면 집중도 외의 오차를 살필 수 있다.

바나나 비교는 고정 제안 3000개와 적응 제안 4500개를 사용하며 제안 갱신 비용도 다르다. 원시 ESS 비율만으로 속도 향상을 단정하지 않는다. 실린 출력에서는 오히려 적응 쪽의 최종 ESS가 23.7로 고정 제안(207.6)의 0.11배이고, 마지막 반복 하나의 ESS가 1.6까지 떨어진다. 코드 주석의 「반복 15번 × 표본 200개」도 실제 설정(300개씩, 4500개)과 다르다. 표시된 히스토그램은 각 제안의 비가중 표본 분포이므로 목표 밀도와 일치할 필요가 없다. 이 코드는 NumPy/SciPy의 제안 적응이며 PyTorch 트레인 루프가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
적응 표본을 모두 독립이라고 볼 수 있는가?

</div>

??? success "풀이"

    각 단계의 제안이 과거 표본으로 결정되므로 단계 간 의존성이 있다. 과거를 조건으로 한 올바른 밀도비 계산과 무조건적 독립성은 다른 주장이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
두 갱신 방식과 공분산 축소의 차이를 설명하라.

</div>

??? success "풀이"

    resample은 가중 확률로 중심을 재표집하고 weighted_means는 상위 웨이트 표본을 중심으로 선택한다. 공분산 축소는 국소 집중을 늘리지만 탐색 범위를 줄여 누락 모드를 악화시킬 수 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
고정 제안과 적응 제안을 공정하게 비교하려면 무엇을 맞춰야 하는가?

</div>

??? success "풀이"

    전체 표본 수 또는 총 계산 시간, 목표 평가 비용과 갱신 비용을 맞추고 같은 함수의 반복 오차를 비교한다. 3000개 대 4500개의 ESS 비만으로 계산 효율을 결론내리지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 조각이 이 적응 샘플러의 종료 기준으로 적절한가?

</div>

??? success "풀이"

    샘플러에는 model·val_loader·state_dict 인터페이스가 없으므로 직접 적용할 수 없다. 예산, 제안 안정성, 함수별 몬테카를로 오차와 탐색을 평가해야 한다. 조각의 state_dict().copy()도 텐서 값을 독립 복제하지 않는 얕은 복사다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 트레인 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


## 정리하며

이 예제는 앞 단계의 가중 표본으로 다음 혼합 제안을 만드는 적응적 중요도 샘플링을 구현한다. 여기서 AIS는 adaptive importance sampling이며 annealed importance sampling과 구분한다. 웨이트 진단과 실제 계산식을 함께 확인하고, 실린 출력 한 번의 결론을 일반적인 보장으로 받아들이지 않는다.
