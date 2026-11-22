# 혼합 제안분포

혼합 제안은 여러 영역에 표본을 배분하고 전체 혼합밀도로 가중한다. 두 봉우리·세 봉우리 목표와 방어적 혼합을 비교하면서 성분 수와 안전 성분의 실제 보장 조건을 살핀다.

## 1. 코드

??? note "코드 (552줄)"

    ```python
    """
    07_mixture_proposals.py

    중급 단계: 중요도 샘플링 제안으로서의 혼합 분포

    이 단원은 혼합 분포를 제안으로 쓰는 방법을 다룬다
    이는 특히 다음에 효과적이다:
    - 봉우리가 여럿인 타깃 분포
    - 복잡한 비표준 형태
    - 커버리지와 ESS가 개선됨

    수학적 기초:
    ---------------------
    혼합 제안:
        q(θ) = Σⱼ αⱼ qⱼ(θ)

    여기서 각 기호는 다음과 같다.
    - qⱼ(θ)는 성분 분포이다
    - αⱼ는 혼합 웨이트이다(Σⱼ αⱼ = 1, αⱼ ≥ 0)
    - K는 성분의 개수이다

    혼합에서 샘플링하기:
    1. 확률 αⱼ로 성분 j 샘플링
    2. θ ~ qⱼ(θ) 샘플링

    밀도 평가:
        q(θ) = Σⱼ αⱼ qⱼ(θ)

    이점:
    - 복잡한 형태를 근사할 수 있다
    - 봉우리가 여럿인 타깃을 자연스럽게 다룬다
    - 유연하고 표현력이 좋다
    - 성분마다 단순해도 된다(이를테면 가우스)

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.special import logsumexp
    import seaborn as sns
    from typing import List, Callable

    np.random.seed(42)
    sns.set_style("whitegrid")


    class MixtureProposal:
        """
        중요도 샘플링을 위한 혼합 분포 제안.
        """

        def __init__(self, components: List, weights: np.ndarray):
            """
            파라메터:
            -----------
            components : list of scipy.stats distributions
                성분 분포
            weights : array
                혼합 웨이트(정규화된다)
            """
            self.components = components
            self.weights = np.array(weights) / np.sum(weights)
            self.n_components = len(components)

        def rvs(self, size: int = 1) -> np.ndarray:
            """
            혼합 분포에서 샘플링하기.

            알고리즘:
            1. 표본마다:
               가) 확률 αⱼ로 성분 j 뽑기
               나) qⱼ에서 표본 뽑기
            """
            # 성분 인덱스 샘플링
            component_indices = np.random.choice(
                self.n_components,
                size=size,
                p=self.weights
            )

            # 선택한 성분에서 샘플링
            samples = []
            for idx in component_indices:
                sample = self.components[idx].rvs()
                samples.append(sample)

            return np.array(samples)

        def pdf(self, x: np.ndarray) -> np.ndarray:
            """
            혼합의 밀도 평가: q(x) = Σⱼ αⱼ qⱼ(x)
            """
            x = np.atleast_1d(x)
            density = np.zeros(len(x))

            for weight, component in zip(self.weights, self.components):
                density += weight * component.pdf(x)

            return density

        def logpdf(self, x: np.ndarray) -> np.ndarray:
            """
            안정을 위해 log-sum-exp를 써서 혼합의 로그 밀도 평가.

            log q(x) = log(Σⱼ αⱼ qⱼ(x))
                     = log-sum-exp(log αⱼ + log qⱼ(x))
            """
            x = np.atleast_1d(x)

            log_densities = []
            for weight, component in zip(self.weights, self.components):
                log_densities.append(np.log(weight) + component.logpdf(x))

            return logsumexp(log_densities, axis=0)


    def importance_sampling_mixture(target_density: Callable,
                                     proposal: MixtureProposal,
                                     h_function: Callable,
                                     n_samples: int) -> tuple:
        """
        혼합 제안을 쓰는 중요도 샘플링.

        반환값:
        --------
        estimate : float
        samples : array
        weights : 배열(정규화됨)
        ess : float
        """
        # 혼합 제안에서 샘플링
        samples = proposal.rvs(size=n_samples)

        # 중요도 웨이트 계산하기
        target_vals = target_density(samples)
        proposal_vals = proposal.pdf(samples)

        unnorm_weights = target_vals / (proposal_vals + 1e-300)
        weights = unnorm_weights / np.sum(unnorm_weights)

        # 추정하기
        estimate = np.sum(weights * h_function(samples))

        # ESS
        ess = 1.0 / np.sum(weights**2)

        return estimate, samples, weights, ess


    # 보기 1: 봉우리 둘인 타깃
    # =======================
    print("=" * 70)
    print("EXAMPLE 1: Bimodal Target Distribution")
    print("=" * 70)

    # 타깃: 가우스 둘의 혼합
    # 0.3 N(-2, 0.8) + 0.7 N(3, 1.2)
    def bimodal_target(x):
        """봉우리 둘인 타깃 밀도"""
        return (0.3 * stats.norm.pdf(x, -2, 0.8) +
                0.7 * stats.norm.pdf(x, 3, 1.2))

    # 추정할 함수: E[θ]
    h_mean = lambda x: x

    # 참 기댓값
    x_grid = np.linspace(-6, 8, 10000)
    true_mean = np.trapz(x_grid * bimodal_target(x_grid), x_grid)

    print(f"\nTarget: 0.3 N(-2, 0.8) + 0.7 N(3, 1.2)")
    print(f"True mean: {true_mean:.6f}")

    # 제안 1: 넓은 가우스 하나(봉우리 둘에는 나쁨)
    proposal_single = stats.norm(0, 3)

    n_samples = 3000

    samples_single = proposal_single.rvs(size=n_samples)
    weights_single_unnorm = bimodal_target(samples_single) / proposal_single.pdf(samples_single)
    weights_single = weights_single_unnorm / np.sum(weights_single_unnorm)
    estimate_single = np.sum(weights_single * h_mean(samples_single))
    ess_single = 1.0 / np.sum(weights_single**2)

    print(f"\nSingle Gaussian Proposal N(0, 3):")
    print(f"  Estimate: {estimate_single:.6f}")
    print(f"  Error: {abs(estimate_single - true_mean):.6f}")
    print(f"  ESS: {ess_single:.1f} ({ess_single/n_samples:.1%})")

    # 제안 2: 타깃의 구조에 맞춘 혼합
    components_matched = [
        stats.norm(-2, 1.0),   # 왼쪽 봉우리 덮기
        stats.norm(3, 1.5),    # 오른쪽 봉우리 덮기
    ]
    weights_matched = [0.3, 0.7]  # 타깃의 웨이트에 맞춤

    proposal_mixture_matched = MixtureProposal(components_matched, weights_matched)

    estimate_matched, samples_matched, weights_matched_norm, ess_matched = \
        importance_sampling_mixture(bimodal_target, proposal_mixture_matched, 
                                    h_mean, n_samples)

    print(f"\nMixture Proposal (matched to target):")
    print(f"  Components: 0.3 N(-2,1) + 0.7 N(3,1.5)")
    print(f"  Estimate: {estimate_matched:.6f}")
    print(f"  Error: {abs(estimate_matched - true_mean):.6f}")
    print(f"  ESS: {ess_matched:.1f} ({ess_matched/n_samples:.1%})")

    # 제안 3: 웨이트가 같은 혼합
    components_equal = [
        stats.norm(-2, 1.0),
        stats.norm(3, 1.5),
    ]
    weights_equal = [0.5, 0.5]  # 같은 웨이트(가장 좋지는 않음)

    proposal_mixture_equal = MixtureProposal(components_equal, weights_equal)

    estimate_equal, samples_equal, weights_equal_norm, ess_equal = \
        importance_sampling_mixture(bimodal_target, proposal_mixture_equal,
                                    h_mean, n_samples)

    print(f"\nMixture Proposal (equal weights):")
    print(f"  Components: 0.5 N(-2,1) + 0.5 N(3,1.5)")
    print(f"  Estimate: {estimate_equal:.6f}")
    print(f"  Error: {abs(estimate_equal - true_mean):.6f}")
    print(f"  ESS: {ess_equal:.1f} ({ess_equal/n_samples:.1%})")

    print(f"\nImprovement over single Gaussian:")
    print(f"  Matched mixture: {ess_matched/ess_single:.2f}x better ESS")
    print(f"  Equal mixture: {ess_equal/ess_single:.2f}x better ESS")

    # 시각화한다
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 칸 1: 타깃과 제안들
    ax = axes[0, 0]
    x_plot = np.linspace(-6, 8, 1000)
    ax.plot(x_plot, bimodal_target(x_plot), 'k-', linewidth=3,
            label='Target', alpha=0.7)
    ax.plot(x_plot, proposal_single.pdf(x_plot), 'r--', linewidth=2,
            label='Single N(0,3)')
    ax.plot(x_plot, proposal_mixture_matched.pdf(x_plot), 'b:', linewidth=2,
            label='Mixture (matched)')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Target vs Proposals', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 칸 2: 가우스 하나에서 뽑은 표본
    ax = axes[0, 1]
    ax.hist(samples_single, bins=50, density=True, alpha=0.5,
            color='steelblue', edgecolor='black')
    ax.plot(x_plot, bimodal_target(x_plot), 'r-', linewidth=2,
            label='Target')
    scatter = ax.scatter(samples_single, np.zeros(len(samples_single)),
                        c=weights_single*n_samples, cmap='hot',
                        s=20, alpha=0.6)
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Single Gaussian: ESS={ess_single:.0f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # 칸 3: 혼합에서 뽑은 표본(구조에 맞춤)
    ax = axes[1, 0]
    ax.hist(samples_matched, bins=50, density=True, alpha=0.5,
            color='green', edgecolor='black')
    ax.plot(x_plot, bimodal_target(x_plot), 'r-', linewidth=2,
            label='Target')
    scatter = ax.scatter(samples_matched, np.zeros(len(samples_matched)),
                        c=weights_matched_norm*n_samples, cmap='hot',
                        s=20, alpha=0.6)
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Mixture (matched): ESS={ess_matched:.0f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # 칸 4: ESS 견주기
    ax = axes[1, 1]
    proposals = ['Single\nGaussian', 'Equal\nMixture', 'Matched\nMixture']
    ess_values = [ess_single, ess_equal, ess_matched]
    colors = ['red', 'orange', 'green']
    bars = ax.bar(proposals, ess_values, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('ESS Comparison', fontsize=13, fontweight='bold')
    ax.axhline(n_samples, color='blue', linestyle='--', linewidth=2,
              label='n samples', alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # 값 라벨을 추가한다
    for bar, ess in zip(bars, ess_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ess:.0f}\n({ess/n_samples:.1%})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('mixture_bimodal.png',
                dpi=300, bbox_inches='tight')


    # 보기 2: 봉우리 셋인 타깃
    # ========================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Trimodal Target Distribution")
    print("=" * 70)

    # 타깃: 봉우리 셋
    def trimodal_target(x):
        """잘 분리된 봉우리 셋"""
        return (0.2 * stats.norm.pdf(x, -4, 0.6) +
                0.5 * stats.norm.pdf(x, 0, 0.8) +
                0.3 * stats.norm.pdf(x, 5, 0.7))

    # 참 평균
    true_mean_tri = np.trapz(x_grid * trimodal_target(x_grid), x_grid)

    print(f"\nTarget: 0.2 N(-4,0.6) + 0.5 N(0,0.8) + 0.3 N(5,0.7)")
    print(f"True mean: {true_mean_tri:.6f}")

    # 혼합 성분 개수를 달리해 견주기
    n_samples_tri = 4000

    # 성분 1개(가우스 하나)
    proposal_k1 = stats.norm(0, 4)
    samples_k1 = proposal_k1.rvs(size=n_samples_tri)
    weights_k1_unnorm = trimodal_target(samples_k1) / proposal_k1.pdf(samples_k1)
    weights_k1 = weights_k1_unnorm / np.sum(weights_k1_unnorm)
    ess_k1 = 1.0 / np.sum(weights_k1**2)

    # 성분 2개
    components_k2 = [stats.norm(-4, 1.0), stats.norm(2.5, 3.0)]
    weights_k2_mix = [0.3, 0.7]
    proposal_k2 = MixtureProposal(components_k2, weights_k2_mix)
    _, _, weights_k2, ess_k2 = importance_sampling_mixture(
        trimodal_target, proposal_k2, h_mean, n_samples_tri
    )

    # 성분 3개(구조에 맞춤)
    components_k3 = [
        stats.norm(-4, 0.8),
        stats.norm(0, 1.0),
        stats.norm(5, 0.9),
    ]
    weights_k3_mix = [0.2, 0.5, 0.3]
    proposal_k3 = MixtureProposal(components_k3, weights_k3_mix)
    _, _, weights_k3, ess_k3 = importance_sampling_mixture(
        trimodal_target, proposal_k3, h_mean, n_samples_tri
    )

    # 성분 5개(파라메터가 너무 많음)
    components_k5 = [
        stats.norm(-4, 0.8),
        stats.norm(-2, 0.8),
        stats.norm(0, 1.0),
        stats.norm(3, 1.0),
        stats.norm(5, 0.9),
    ]
    weights_k5_mix = [0.15, 0.15, 0.4, 0.15, 0.15]
    proposal_k5 = MixtureProposal(components_k5, weights_k5_mix)
    _, _, weights_k5, ess_k5 = importance_sampling_mixture(
        trimodal_target, proposal_k5, h_mean, n_samples_tri
    )

    print(f"\nESS vs Number of Components (n={n_samples_tri}):")
    print(f"{'K':>3} {'ESS':>8} {'Efficiency':>12} {'Description'}")
    print("-" * 55)
    print(f"  1 {ess_k1:8.1f} {ess_k1/n_samples_tri:11.1%} Single Gaussian")
    print(f"  2 {ess_k2:8.1f} {ess_k2/n_samples_tri:11.1%} Two components")
    print(f"  3 {ess_k3:8.1f} {ess_k3/n_samples_tri:11.1%} Three (matched)")
    print(f"  5 {ess_k5:8.1f} {ess_k5/n_samples_tri:11.1%} Five (overfit)")

    # 시각화한다
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x_plot_tri = np.linspace(-7, 8, 1000)

    for ax, k, proposal, ess in [(axes[0,0], 1, proposal_k1, ess_k1),
                                  (axes[0,1], 2, proposal_k2, ess_k2),
                                  (axes[1,0], 3, proposal_k3, ess_k3),
                                  (axes[1,1], 5, proposal_k5, ess_k5)]:
        ax.plot(x_plot_tri, trimodal_target(x_plot_tri), 'r-',
                linewidth=3, label='Target', alpha=0.7)

        if k == 1:
            ax.plot(x_plot_tri, proposal.pdf(x_plot_tri), 'b--',
                    linewidth=2, label='Proposal')
        else:
            ax.plot(x_plot_tri, proposal.pdf(x_plot_tri), 'b--',
                    linewidth=2, label='Mixture proposal')

        ax.set_xlabel('θ', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'K={k} components: ESS={ess:.0f} ({ess/n_samples_tri:.1%})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mixture_components.png',
                dpi=300, bbox_inches='tight')


    # 보기 3: 방어적 혼합(강건성)
    # =======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Defensive Mixture for Robustness")
    print("=" * 70)

    print("""
    방어적 중요도 샘플링은 집중 제안에
    커버리지를 보장하려고 넓은 "안전" 성분을 섞는다:

        q_def(θ) = α q_target(θ) + (1-α) q_safe(θ)

    여기서 각 기호는 다음과 같다.
    - q_target: 확률이 높은 영역에 집중
    - q_safe: 넓은 커버리지(이를테면 사전분포나 평평한 분포)
    - α ∈ (0,1): 트레이드오프 파라메터(보통 α ≈ 0.7-0.9)
    """)

    # 타깃: 봉우리 둘(보기 1과 같음)
    alpha_defensive = 0.8  # 집중 성분의 웨이트

    # 집중 성분: 봉우리 둘을 다 덮는 혼합
    components_targeted = [stats.norm(-2, 0.8), stats.norm(3, 1.2)]
    weights_targeted = [0.3, 0.7]
    proposal_targeted = MixtureProposal(components_targeted, weights_targeted)

    # 안전 성분: 넓은 가우스
    proposal_safe = stats.norm(0, 5)

    # 방어적 혼합
    components_defensive = [proposal_targeted, proposal_safe]

    # 한 성분을 그 자체로 혼합으로 다루는 사용자 정의 혼합
    class DefensiveMixture:
        def __init__(self, targeted_mixture, safe_dist, alpha):
            self.targeted = targeted_mixture
            self.safe = safe_dist
            self.alpha = alpha

        def rvs(self, size):
            samples = []
            for _ in range(size):
                if np.random.rand() < self.alpha:
                    samples.append(self.targeted.rvs(1)[0])
                else:
                    samples.append(self.safe.rvs())
            return np.array(samples)

        def pdf(self, x):
            return self.alpha * self.targeted.pdf(x) + (1 - self.alpha) * self.safe.pdf(x)

    proposal_defensive = DefensiveMixture(proposal_targeted, proposal_safe,
                                          alpha_defensive)

    estimate_def, samples_def, weights_def, ess_def = \
        importance_sampling_mixture(bimodal_target, proposal_defensive,
                                    h_mean, n_samples)

    print(f"\nDefensive Mixture (α={alpha_defensive}):")
    print(f"  Estimate: {estimate_def:.6f}")
    print(f"  Error: {abs(estimate_def - true_mean):.6f}")
    print(f"  ESS: {ess_def:.1f} ({ess_def/n_samples:.1%})")

    # 순수 집중 제안과 견주기
    estimate_pure, _, weights_pure, ess_pure = \
        importance_sampling_mixture(bimodal_target, proposal_targeted,
                                    h_mean, n_samples)

    print(f"\nPure Targeted Mixture:")
    print(f"  ESS: {ess_pure:.1f} ({ess_pure/n_samples:.1%})")

    print(f"\nTrade-off:")
    print(f"  Defensive ESS: {ess_def:.1f} ({ess_def/ess_pure:.0%} of pure)")
    print(f"  But guarantees minimum ESS even if target shape is wrong")

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 혼합 제안은 복잡한 타깃에 강력하다:
       - 임의의 형태를 근사할 수 있다
       - 봉우리가 여럿인 분포에 자연스럽다
       - 유연하고 표현력이 좋다

    2. 혼합 설계:
       - 타깃의 봉우리에 성분 배치하기
       - 혼합의 웨이트를 봉우리의 웨이트에 (대체로) 맞추기
       - 타깃의 봉우리보다 조금 넓은 성분 쓰기
       - 보통 K = 봉우리의 수가 필요하다

    3. 성분의 개수:
       - 너무 적으면: 중요한 영역을 놓치고 ESS가 낮다
       - 알맞은 정도: 봉우리를 모두 덮고 ESS가 높다
       - 너무 많으면: 계산 오버헤드가 커지고 ESS는 조금밖에 안 는다
       - K = 예상되는 봉우리 수로 시작한 뒤 조정하기

    4. 혼합 웨이트:
       - 타깃 봉우리의 웨이트와 대체로 맞아야 한다
       - 같은 웨이트도 되지만 가장 좋지는 않을 수 있다
       - 예비 표본으로 추정할 수 있다

    5. 성분 배치:
       - 추정한 봉우리 위치에 성분의 중심을 맞추기
       - 최적화나 군집화로 봉우리를 찾을 수 있다
       - 사전 탐색이 도움이 된다

    6. 방어적 혼합:
       - 집중 성분과 넓은 안전 성분 섞기
       - q(θ) = α q_target(θ) + (1-α) q_safe(θ)
       - 보통 α ≈ 0.7-0.9
       - 강건성을 얻으려고 ESS를 내준다
       - 최소 ESS를 보장한다

    7. 장점:
       - 봉우리가 여럿이면 성분 하나보다 훨씬 낫다
       - ESS를 5배에서 10배 높일 수 있다
       - 복잡한 형태에도 충분히 유연하다
       - 성분은 단순해도 된다(이를테면 가우스)

    8. 실전 요령:
       - 성분 두세 개로 시작하기
       - 탐색용 표본으로 봉우리 위치 찾기
       - 성분 하나일 때보다 ESS가 나아지는지 확인하기
       - 웨이트 집중 모니터링하기
       - 강건성을 위해 방어적 변형 고려하기

    9. 언제 쓰나:
       - 봉우리가 여럿인 타깃(꼭 필요하다)
       - 복잡한 비표준 형태
       - 성분 하나로는 ESS가 나쁠 때
       - 봉우리가 여럿인 베이즈 추론

    10. 계산 비용:
        - 샘플링: O(K)의 오버헤드
        - 밀도 평가: O(K) 비용
        - ESS가 나아지므로 대개 가치가 있다
        - 성분 샘플링을 병렬화할 수 있다
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (133줄)"

    ```
    ======================================================================
    EXAMPLE 1: Bimodal Target Distribution
    ======================================================================

    Target: 0.3 N(-2, 0.8) + 0.7 N(3, 1.2)
    True mean: 1.499911

    Single Gaussian Proposal N(0, 3):
      Estimate: 1.495669
      Error: 0.004242
      ESS: 1639.3 (54.6%)

    Mixture Proposal (matched to target):
      Components: 0.3 N(-2,1) + 0.7 N(3,1.5)
      Estimate: 1.449600
      Error: 0.050311
      ESS: 2795.4 (93.2%)

    Mixture Proposal (equal weights):
      Components: 0.5 N(-2,1) + 0.5 N(3,1.5)
      Estimate: 1.490874
      Error: 0.009038
      ESS: 2423.4 (80.8%)

    Improvement over single Gaussian:
      Matched mixture: 1.71x better ESS
      Equal mixture: 1.48x better ESS

    ======================================================================
    EXAMPLE 2: Trimodal Target Distribution
    ======================================================================

    Target: 0.2 N(-4,0.6) + 0.5 N(0,0.8) + 0.3 N(5,0.7)
    True mean: 0.700506

    ESS vs Number of Components (n=4000):
      K      ESS   Efficiency Description
    -------------------------------------------------------
      1   2000.7       50.0% Single Gaussian
      2   1905.9       47.6% Two components
      3   3684.0       92.1% Three (matched)
      5   2830.4       70.8% Five (overfit)

    ======================================================================
    EXAMPLE 3: Defensive Mixture for Robustness
    ======================================================================

    방어적 중요도 샘플링은 집중 제안에
    커버리지를 보장하려고 넓은 "안전" 성분을 섞는다:

        q_def(θ) = α q_target(θ) + (1-α) q_safe(θ)

    여기서 각 기호는 다음과 같다.
    - q_target: 확률이 높은 영역에 집중
    - q_safe: 넓은 커버리지(이를테면 사전분포나 평평한 분포)
    - α ∈ (0,1): 트레이드오프 파라메터(보통 α ≈ 0.7-0.9)


    Defensive Mixture (α=0.8):
      Estimate: 1.535051
      Error: 0.035140
      ESS: 2778.8 (92.6%)

    Pure Targeted Mixture:
      ESS: 3000.0 (100.0%)

    Trade-off:
      Defensive ESS: 2778.8 (93% of pure)
      But guarantees minimum ESS even if target shape is wrong

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 혼합 제안은 복잡한 타깃에 강력하다:
       - 임의의 형태를 근사할 수 있다
       - 봉우리가 여럿인 분포에 자연스럽다
       - 유연하고 표현력이 좋다

    2. 혼합 설계:
       - 타깃의 봉우리에 성분 배치하기
       - 혼합의 웨이트를 봉우리의 웨이트에 (대체로) 맞추기
       - 타깃의 봉우리보다 조금 넓은 성분 쓰기
       - 보통 K = 봉우리의 수가 필요하다

    3. 성분의 개수:
       - 너무 적으면: 중요한 영역을 놓치고 ESS가 낮다
       - 알맞은 정도: 봉우리를 모두 덮고 ESS가 높다
       - 너무 많으면: 계산 오버헤드가 커지고 ESS는 조금밖에 안 는다
       - K = 예상되는 봉우리 수로 시작한 뒤 조정하기

    4. 혼합 웨이트:
       - 타깃 봉우리의 웨이트와 대체로 맞아야 한다
       - 같은 웨이트도 되지만 가장 좋지는 않을 수 있다
       - 예비 표본으로 추정할 수 있다

    5. 성분 배치:
       - 추정한 봉우리 위치에 성분의 중심을 맞추기
       - 최적화나 군집화로 봉우리를 찾을 수 있다
       - 사전 탐색이 도움이 된다

    6. 방어적 혼합:
       - 집중 성분과 넓은 안전 성분 섞기
       - q(θ) = α q_target(θ) + (1-α) q_safe(θ)
       - 보통 α ≈ 0.7-0.9
       - 강건성을 얻으려고 ESS를 내준다
       - 최소 ESS를 보장한다

    7. 장점:
       - 봉우리가 여럿이면 성분 하나보다 훨씬 낫다
       - ESS를 5배에서 10배 높일 수 있다
       - 복잡한 형태에도 충분히 유연하다
       - 성분은 단순해도 된다(이를테면 가우스)

    8. 실전 요령:
       - 성분 두세 개로 시작하기
       - 탐색용 표본으로 봉우리 위치 찾기
       - 성분 하나일 때보다 ESS가 나아지는지 확인하기
       - 웨이트 집중 모니터링하기
       - 강건성을 위해 방어적 변형 고려하기

    9. 언제 쓰나:
       - 봉우리가 여럿인 타깃(꼭 필요하다)
       - 복잡한 비표준 형태
       - 성분 하나로는 ESS가 나쁠 때
       - 봉우리가 여럿인 베이즈 추론

    10. 계산 비용:
        - 샘플링: O(K)의 오버헤드
        - 밀도 평가: O(K) 비용
        - ESS가 나아지므로 대개 가치가 있다
        - 성분 샘플링을 병렬화할 수 있다

    ```


## 2. 논의

### 전체 혼합밀도와 추정량

q(x)=∑α_kq_k(x), α_k≥0, ∑α_k=1에서 성분을 선택한 뒤 해당 분포에서 표본을 생성한다. 현재 구현의 웨이트 분모는 선택된 성분 밀도만이 아니라 전체 혼합밀도다. importance_sampling_mixture는 웨이트를 합으로 나누므로 SNIS이며 이름만으로 불편한 표준 IS로 해석하면 안 된다.

첫 목표의 정확한 평균은 0.3×(−2)+0.7×3=1.5, 세 봉우리 목표는 0.2×(−4)+0.5×0+0.3×5=0.7이다. 코드가 True mean이라고 부른 값은 유한 격자 [−6,8]의 수치 적분이므로 절단·적분 오차가 있다. SciPy 정규 인자의 두 번째 수는 표준편차다.

### 성분 수와 비용

여러 봉우리에 성분을 배치하면 방문 범위를 개선할 수 있지만 성분 수가 봉우리 수와 같아야 하는 것은 아니다. 추가 성분이 항상 ESS를 늘리지도 않는다. 5개 성분은 고정해 지정했으므로 코드의 overfit 라벨을 트레인 데이터에 대한 과적합의 실증으로 볼 수 없다.

MixtureProposal은 1차원 표본과 정수 size를 가정한다. 비어 있지 않은 성분 목록, 같은 길이의 유한 비음수 비중, 양의 비중 합을 확인해야 한다. pdf는 K개 밀도를 더하며 logpdf는 log-sum-exp를 사용하지만 실제 IS 함수는 pdf 비를 사용하므로 로그밀도 구현의 안정성을 자동으로 얻지 못한다.

### 방어적 혼합의 조건부 보장

q_def=αq_target+(1−α)q_safe에서는 q_def≥(1−α)q_safe다. 따라서 안전 제안이 필요한 지지집합을 덮으면 혼합도 덮는다. p/q_safe≤M 같은 유한한 비율 상계가 있으면 p/q_def≤M/(1−α)도 얻을 수 있다. 단지 분산이 큰 정규분포를 넣었다는 이유로 모든 목표의 꼬리나 유한 분산이 보장되지는 않는다.

최소 ESS를 보장한다는 출력은 추가 조건 없이 성립하지 않는다. 이 예제의 순수 targeted 제안은 첫 목표와 정확히 같은 혼합분포이므로 이론적으로 원시 웨이트가 모두 1이고 ESS=n이다. 안전 성분을 더하면 이미 완벽한 웨이트 집중도는 개선할 수 없다. 다른 목표에서의 안정성과 실제 비용을 별도로 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
혼합 제안에서 웨이트 분모가 전체 혼합밀도인 이유는?

</div>

??? success "풀이"

    실제로 생성한 표본의 주변분포가 ∑α_kq_k이기 때문이다. 현재 추정 방식에서 선택된 성분의 밀도만 쓰면 일반적으로 다른 추정량이 된다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
혼합 제안의 입력 조건을 설명하라.

</div>

??? success "풀이"

    성분 수와 비중 길이가 같고 비중은 유한·비음수이며 합이 양수여야 한다. 각 성분은 유효한 표본·밀도 인터페이스와 일관된 차원을 가져야 하고 표본 수는 양의 정수다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
방어적 성분이 모든 목표에서 최소 ESS를 보장하는가?

</div>

??? success "풀이"

    아니다. 지지집합 포함과 목표/안전 제안의 비율·관심 함수의 적률 같은 조건이 필요하다. 넓다는 직관만으로 꼬리 안정성이나 유한 실행의 특정 ESS 하한을 보장할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 네트워크 시험 대신 혼합 구현을 확인할 기준을 제시하라.

</div>

??? success "풀이"

    전체 pdf의 성분 합, logpdf와 pdf의 일치 가능한 범위, 비중에 따른 표본 빈도와 알려진 혼합 모멘트를 확인한다. 아래 호출·parameters·backward 시험은 이 클래스에 존재하지 않는 인터페이스다.

    ```python
    def test_mixtureproposal():
        model = MixtureProposal(...)
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

혼합 제안은 여러 영역에 표본을 배분하고 전체 혼합밀도로 가중한다. 두 봉우리·세 봉우리 목표와 방어적 혼합을 비교하면서 성분 수와 안전 성분의 실제 보장 조건을 살핀다. 웨이트 진단과 실제 계산식을 함께 확인하고, 실린 출력 한 번의 결론을 일반적인 보장으로 받아들이지 않는다.
