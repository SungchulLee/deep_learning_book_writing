# 제안분포 진단

제안분포별 웨이트 집중도·모멘트·꼬리와 다봉 영역의 방문을 진단한다. 수치 요약이 추정하는 대상과 관측되지 않은 영역에 대한 한계를 함께 해석한다.

## 1. 코드

??? note "코드 (566줄)"

    ```python
    """
    05_proposal_diagnostics.py

    중급 단계: 제안의 질을 종합적으로 진단하기

    이 단원은 중요도 샘플링에서 제안 분포의 질을 평가하는
    도구와 방법을 다룬다.

    핵심 진단 지표:
    1. 유효 표본 크기(ESS)
    2. 웨이트의 분산과 변동계수
    3. 엔트로피와 퍼플렉서티
    4. 쿨백-라이블러 발산 추정값
    5. 커버리지 진단
    6. χ² 발산 지표

    수학적 배경:
    ---------------------
    좋은 제안 q는 다음을 만족해야 한다:
    - π의 지지집합(support)을 덮을 것(중요함: 반드시)
    - π와 형태가 비슷할 것(분산에 영향을 준다)
    - π보다 꼬리가 두꺼울 것(분산이 무한해지는 것을 막는다)
    - 샘플링하기 쉬울 것(계산 비용의 문제)

    가장 좋은 제안(분산을 가장 작게 하려면):
        q*(θ) = |h(θ)|π(θ) / ∫|h(θ)|π(θ)dθ

    실전에서는 q가 q*에 얼마나 가까운지 진단으로 살핀다.

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.special import logsumexp
    import seaborn as sns
    from typing import Tuple, Dict, Callable

    np.random.seed(42)
    sns.set_style("whitegrid")


    class ProposalDiagnostics:
        """
        중요도 샘플링 제안을 종합적으로 진단하는 클래스.
        """

        def __init__(self, target_density: Callable, proposal_dist,
                     name: str = "Proposal"):
            """
            파라메터:
            -----------
            target_density : callable
                타깃 밀도 함수(정규화하지 않아도 된다)
            proposal_dist : scipy.stats distribution
                제안 분포
            name : str
                출력에 쓰는 이름
            """
            self.target_density = target_density
            self.proposal_dist = proposal_dist
            self.name = name

        def compute_all_diagnostics(self, n_samples: int = 5000) -> Dict:
            """
            진단 통계량을 종합적으로 계산하기.

            모든 지표를 담은 딕셔너리를 반환한다.
            """
            # 표본 추출
            samples = self.proposal_dist.rvs(size=n_samples)

            # 웨이트 계산하기
            target_vals = self.target_density(samples)
            proposal_vals = self.proposal_dist.pdf(samples)

            unnorm_weights = target_vals / (proposal_vals + 1e-300)
            weights = unnorm_weights / np.sum(unnorm_weights)

            # 진단을 모두 계산하기
            diagnostics = {
                'name': self.name,
                'n_samples': n_samples,
            }

            # 1. ESS 지표
            diagnostics.update(self._compute_ess_metrics(weights))

            # 2. 웨이트 통계량
            diagnostics.update(self._compute_weight_statistics(weights))

            # 3. 커버리지 지표
            diagnostics.update(self._compute_coverage_metrics(samples, weights))

            # 4. 발산 추정값
            diagnostics.update(self._compute_divergence_metrics(samples, weights))

            # 그리기 위해 표본과 웨이트 저장
            diagnostics['samples'] = samples
            diagnostics['weights'] = weights

            return diagnostics

        def _compute_ess_metrics(self, weights: np.ndarray) -> Dict:
            """유효 표본 크기 관련 지표."""
            n = len(weights)

            # 표준 ESS
            ess = 1.0 / np.sum(weights**2)

            # 상대 ESS
            rel_ess = ess / n

            # 퍼플렉서티(엔트로피의 지수)
            entropy = -np.sum(weights * np.log(weights + 1e-300))
            perplexity = np.exp(entropy)

            return {
                'ess': ess,
                'relative_ess': rel_ess,
                'entropy': entropy,
                'perplexity': perplexity,
            }

        def _compute_weight_statistics(self, weights: np.ndarray) -> Dict:
            """웨이트 분포의 통계량."""
            n = len(weights)

            # 기본 통계량
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)

            # 변동계수
            cv = std_weight / (mean_weight + 1e-10)

            # 분위수
            quantiles = np.percentile(weights, [25, 50, 75, 90, 95, 99])

            # 웨이트 집중
            sorted_weights = np.sort(weights)[::-1]
            cumsum = np.cumsum(sorted_weights)

            n_for_50pct = np.searchsorted(cumsum, 0.50) + 1
            n_for_90pct = np.searchsorted(cumsum, 0.90) + 1

            return {
                'max_weight': max_weight,
                'min_weight': min_weight,
                'mean_weight': mean_weight,
                'std_weight': std_weight,
                'cv_weights': cv,
                'weight_q25': quantiles[0],
                'weight_q50': quantiles[1],
                'weight_q75': quantiles[2],
                'weight_q90': quantiles[3],
                'weight_q95': quantiles[4],
                'weight_q99': quantiles[5],
                'n_for_50pct_weight': n_for_50pct,
                'n_for_90pct_weight': n_for_90pct,
                'pct_for_50pct_weight': n_for_50pct / n * 100,
                'pct_for_90pct_weight': n_for_90pct / n * 100,
            }

        def _compute_coverage_metrics(self, samples: np.ndarray, 
                                       weights: np.ndarray) -> Dict:
            """
            제안이 타깃을 얼마나 잘 덮는지 평가하기.
            """
            # 표본의 유효 범위
            weighted_mean = np.sum(weights * samples)
            weighted_var = np.sum(weights * (samples - weighted_mean)**2)
            weighted_std = np.sqrt(weighted_var)

            # 커버리지 확인: 웨이트가 큰 표본이 널리 퍼져 있는가?
            # 웨이트로 정렬
            sorted_indices = np.argsort(weights)[::-1]
            top_10pct_idx = sorted_indices[:int(0.1 * len(samples))]
            top_samples = samples[top_10pct_idx]

            # 상위 표본의 확산
            top_spread = np.std(top_samples)

            return {
                'weighted_mean': weighted_mean,
                'weighted_std': weighted_std,
                'top_10pct_spread': top_spread,
            }

        def _compute_divergence_metrics(self, samples: np.ndarray,
                                         weights: np.ndarray) -> Dict:
            """
            타깃과 제안 사이의 발산 추정하기.

            주의: 이는 거친 추정값이지 정확한 값이 아니다.
            """
            n = len(samples)

            # 중요도 웨이트로 추정한 KL(π||q)
            # KL(π||q) ≈ E_π[log(π/q)] ≈ mean(log(w))
            log_weights = np.log(weights * n + 1e-300)
            kl_estimate = np.mean(log_weights)

            # 로그 웨이트의 분산(KL 발산과 관련됨)
            var_log_weights = np.var(log_weights)

            # χ² 발산 추정값
            # χ²(π||q) = E_π[(π/q - 1)²] = E_π[(w*n - 1)²]
            chi2_estimate = np.mean((weights * n - 1)**2)

            return {
                'kl_estimate': kl_estimate,
                'var_log_weights': var_log_weights,
                'chi2_estimate': chi2_estimate,
            }

        def plot_diagnostics(self, diagnostics: Dict, save_path: str = None):
            """
            종합 진단 그림 만들기.
            """
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            samples = diagnostics['samples']
            weights = diagnostics['weights']
            n = len(samples)

            # 패널 1: 웨이트 히스토그램
            ax = axes[0, 0]
            ax.hist(weights * n, bins=50, density=True, alpha=0.7,
                    color='steelblue', edgecolor='black')
            ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
                       label='Uniform weight')
            ax.set_xlabel('Normalized Weight × n', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{self.name}: Weight Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 패널 2: 누적 웨이트
            ax = axes[0, 1]
            sorted_weights = np.sort(weights)[::-1]
            cumsum = np.cumsum(sorted_weights)
            ax.plot(np.arange(1, len(sorted_weights)+1), cumsum,
                    linewidth=2, color='darkblue')
            ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
                       label='50% of weight')
            ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5,
                       label='90% of weight')
            ax.set_xlabel('Number of Samples (sorted)', fontsize=11)
            ax.set_ylabel('Cumulative Weight', fontsize=11)
            ax.set_title('Weight Concentration', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 패널 3: 표본 산점도
            ax = axes[0, 2]
            scatter = ax.scatter(samples, weights * n, c=weights * n,
                                cmap='hot', s=30, alpha=0.6,
                                edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Sample Value', fontsize=11)
            ax.set_ylabel('Weight × n', fontsize=11)
            ax.set_title('Samples vs Weights', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax)
            ax.grid(True, alpha=0.3)

            # 패널 4: 웨이트의 Q-Q 그림
            ax = axes[1, 0]
            theoretical_quantiles = np.linspace(0, 1, n)
            sample_quantiles = np.sort(weights * n)
            ax.plot(theoretical_quantiles, sample_quantiles, 'o',
                    markersize=3, alpha=0.5)
            ax.plot([0, 1], [1, 1], 'r--', linewidth=2, label='Uniform')
            ax.set_xlabel('Theoretical Quantile', fontsize=11)
            ax.set_ylabel('Sample Quantile (Weight × n)', fontsize=11)
            ax.set_title('Q-Q Plot: Weights vs Uniform', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 패널 5: 진단 지표 텍스트
            ax = axes[1, 1]
            ax.axis('off')
            metrics_text = f"""
            진단 요약
            ══════════════════════════════
            ESS: {diagnostics['ess']:.1f}
            상대 ESS: {diagnostics['relative_ess']:.1%}
            퍼플렉서티: {diagnostics['perplexity']:.1f}

            웨이트의 변동계수: {diagnostics['cv_weights']:.3f}
            최대 웨이트: {diagnostics['max_weight']:.6f}

            웨이트 50%에 든 표본: {diagnostics['n_for_50pct_weight']}개
                           ({diagnostics['pct_for_50pct_weight']:.1f}%)

            웨이트 90%에 든 표본: {diagnostics['n_for_90pct_weight']}개
                           ({diagnostics['pct_for_90pct_weight']:.1f}%)

            KL 추정값: {diagnostics['kl_estimate']:.4f}
            χ² estimate: {diagnostics['chi2_estimate']:.4f}
            """
            ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # 패널 6: 로그 웨이트 히스토그램
            ax = axes[1, 2]
            log_weights = np.log(weights * n + 1e-10)
            ax.hist(log_weights, bins=50, density=True, alpha=0.7,
                    color='darkgreen', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2,
                       label='log(1) = 0')
            ax.set_xlabel('log(Weight × n)', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title('Log-Weight Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            return fig


    # 예제 1: 여러 제안 견주기
    # ======================================
    print("=" * 70)
    print("EXAMPLE 1: Systematic Proposal Comparison")
    print("=" * 70)

    # 타깃 분포: N(5, 1.5)
    target_mean, target_std = 5.0, 1.5
    target = stats.norm(target_mean, target_std)

    # 견줄 제안 여럿 정하기
    proposals = {
        'Perfect': stats.norm(5.0, 1.5),          # 타깃과 같음
        'Good': stats.norm(5.0, 1.8),             # 조금 더 넓음
        'Decent': stats.norm(4.5, 2.0),           # 평균을 옮기고 더 넓힘
        'Mediocre': stats.norm(3.0, 2.5),         # 더 많이 옮김
        'Poor': stats.norm(5.0, 0.8),             # 너무 좁음
        'Bad': stats.norm(0.0, 1.5),              # 틀린 위치
    }

    print("\nAnalyzing proposals for target N(5, 1.5):\n")

    results = []
    for name, proposal in proposals.items():
        diagnostics_obj = ProposalDiagnostics(target.pdf, proposal, name)
        diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
        results.append(diag)

    # 비교 표 출력
    print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'KL Est':>10} "
          f"{'50% in':>8}")
    print("-" * 70)

    for diag in results:
        print(f"{diag['name']:<12} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
              f"{diag['cv_weights']:8.3f} {diag['kl_estimate']:10.4f} "
              f"{diag['pct_for_50pct_weight']:7.1f}%")

    # 고른 제안에 대한 자세한 그림 만들기
    selected_proposals = ['Perfect', 'Decent', 'Poor']
    for name in selected_proposals:
        diag = [d for d in results if d['name'] == name][0]
        prop = proposals[name]

        diagnostics_obj = ProposalDiagnostics(target.pdf, prop, name)
        diagnostics_obj.plot_diagnostics(
            diag,
            save_path=f'diagnostics_{name.lower()}.png'
        )
        print(f"\nSaved diagnostics plot for {name} proposal")


    # 예제 2: 점진적 성능 저하
    # ==================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: How Proposals Degrade")
    print("=" * 70)

    # 타깃: N(0, 1)
    target_ex2 = stats.norm(0, 1)

    # 어긋남이 커지는 제안 만들기
    mean_shifts = [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    scale = 1.2  # 타깃보다 조금 더 넓음

    print(f"\nProposal: N(μ, {scale}²), varying μ from 0 to 5")
    print(f"Target: N(0, 1)\n")

    print(f"{'Mean Shift':>12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Max Wt':>10}")
    print("-" * 60)

    ess_values = []
    for shift in mean_shifts:
        proposal = stats.norm(shift, scale)

        diagnostics_obj = ProposalDiagnostics(target_ex2.pdf, proposal, f"μ={shift}")
        diag = diagnostics_obj.compute_all_diagnostics(n_samples=2000)

        ess_values.append(diag['ess'])

        print(f"{shift:12.1f} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
              f"{diag['cv_weights']:8.3f} {diag['max_weight']:10.6f}")

    # ESS 저하 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mean_shifts, ess_values, 'o-', linewidth=2, markersize=10,
            color='steelblue')
    ax.set_xlabel('Proposal Mean Shift', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('ESS Degradation with Proposal-Target Mismatch',
                 fontsize=13, fontweight='bold')
    ax.axhline(2000, color='red', linestyle='--', linewidth=2,
               label='n samples', alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('ess_degradation.png',
                dpi=300, bbox_inches='tight')


    # 예제 3: 꼬리 커버리지 진단
    # =================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Tail Coverage Assessment")
    print("=" * 70)

    print("""
    꼬리 두꺼운 타깃에서는 제안이 꼬리를 넉넉히 덮어야 한다.
    스튜던트 t 타깃을 꼬리 두께가 다른 제안들과 견준다.
    """)

    # 타깃: 자유도 3인 스튜던트 t(두꺼운 꼬리)
    target_t = stats.t(df=3, loc=0, scale=1)

    # 꼬리 두께가 서로 다른 제안
    tail_proposals = {
        'Heavy: t(3)': stats.t(df=3, loc=0, scale=1.2),
        'Medium: t(5)': stats.t(df=5, loc=0, scale=1.2),
        'Light: Normal': stats.norm(0, 1.5),
    }

    print("\nTarget: Student-t(df=3)")
    print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'χ² Est':>10} {'Status'}")
    print("-" * 65)

    for name, proposal in tail_proposals.items():
        diagnostics_obj = ProposalDiagnostics(target_t.pdf, proposal, name)
        diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)

        # 수상하게 큰 웨이트가 있는지 살피기(꼬리 커버리지 문제)
        max_weight_ratio = diag['max_weight'] * diag['n_samples']
        status = "✓ Good" if max_weight_ratio < 10 else "⚠ Poor tails"

        print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
              f"{diag['chi2_estimate']:10.2f} {status}")

    print("\nKey insight: Proposals with lighter tails than target can fail!")


    # 예제 4: 다봉(multimodal) 타깃의 진단
    # ======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Diagnosing Coverage for Multimodal Target")
    print("=" * 70)

    # 쌍봉 타깃: 0.5*N(-3,1) + 0.5*N(3,1)
    def bimodal_density(x):
        return 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)

    # 방식이 서로 다른 제안
    multimodal_proposals = {
        'Single wide': stats.norm(0, 4),          # 성분 하나, 아주 넓음
        'Single narrow': stats.norm(0, 1.5),      # 성분 하나, 너무 좁음
        'Centered on mode': stats.norm(-3, 1.5),  # 모드 하나만 덮음
    }

    print("\nTarget: 0.5*N(-3,1) + 0.5*N(3,1) [bimodal]")
    print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'50% Wt%':>10} {'Coverage'}")
    print("-" * 70)

    for name, proposal in multimodal_proposals.items():
        diagnostics_obj = ProposalDiagnostics(bimodal_density, proposal, name)
        diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)

        # 모드 커버리지 확인
        samples = diag['samples']
        left_mode = np.sum((samples < -1) & (samples > -5))
        right_mode = np.sum((samples < 5) & (samples > 1))

        if left_mode > 50 and right_mode > 50:
            coverage = "✓ Both modes"
        else:
            coverage = "⚠ Missing mode"

        print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
              f"{diag['pct_for_50pct_weight']:9.1f}% {coverage}")

    print("\nKey insight: Single-component proposals struggle with multimodal targets!")

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 꼭 필요한 진단:
       - ESS와 상대 ESS(가장 중요하다)
       - 웨이트의 변동계수
       - 웨이트 집중(50%, 90% 백분위수)
       - 최대 웨이트 값

    2. 좋은 제안의 표지:
       - 상대 ESS > 0.1(되도록 > 0.3)
       - 웨이트의 변동계수 < 3(낮을수록 좋다)
       - 웨이트의 50%를 차지하는 표본이 10% 미만
       - 어느 웨이트도 혼자 지배하지 않는다

    3. 경고 신호:
       - ESS << n(n의 1% 미만)
       - 몇몇 표본이 웨이트의 대부분을 진다
       - 변동계수가 아주 높음(> 5-10)
       - KL이나 χ² 발산 추정값이 크다

    4. 꼬리 커버리지:
       - 꼬리 두꺼운 타깃에 결정적이다
       - 최대 웨이트 비 살피기
       - 제안은 꼬리가 더 두꺼워야 한다
       - 꼬리 얇은 제안은 크게 무너질 수 있다

    5. 다봉 타깃:
       - 성분 하나로는 흔히 모드를 놓친다
       - 표본이 모드를 모두 덮는지 확인하기
       - 대개 혼합 제안이 필요하다
       - 아니면 적응적 방법 쓰기

    6. 성능 저하 패턴:
       - 제안과 타깃의 어긋남이 커질수록 ESS가 떨어진다
       - 거리에 따라 흔히 지수적으로 나빠진다
       - 작게 옮겨도 큰 효과가 있을 수 있다

    7. 실전 작업 흐름:
       가) 결과를 믿기 전에 늘 진단을 계산하라
       나) 될 수 있으면 여러 제안 견주기
       다) 경고 신호 살펴보기(ESS, 변동계수, 집중)
       라) 웨이트 분포 그려 보기
       마) 중요한 영역의 커버리지 확인하기

    8. 진단이 나쁠 때의 대처:
       - 제안 파라메터 조정하기(평균, 분산)
       - 꼬리가 더 두꺼운 제안 분포족 쓰기
       - 혼합 제안으로 바꾸기
       - 적응적 중요도 샘플링 생각해 보기
       - 아니면 MCMC를 대신 쓰기
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (121줄)"

    ```
    ======================================================================
    EXAMPLE 1: Systematic Proposal Comparison
    ======================================================================

    Analyzing proposals for target N(5, 1.5):

    Proposal          ESS  Rel ESS       CV     KL Est   50% in
    ----------------------------------------------------------------------
    Perfect        3000.0   100.0%    0.000     0.0000    50.0%
    Good           2850.8    95.0%    0.229    -0.0392    42.5%
    Decent         2570.6    85.7%    0.409    -0.1640    36.0%
    Mediocre       1544.2    51.5%    0.971    -1.2717    19.4%
    Poor            828.1    27.6%    1.619    -0.2358    22.3%
    Bad               1.2     0.0%   49.076    -6.9731     0.0%

    Saved diagnostics plot for Perfect proposal

    Saved diagnostics plot for Decent proposal

    Saved diagnostics plot for Poor proposal

    ======================================================================
    EXAMPLE 2: How Proposals Degrade
    ======================================================================

    Proposal: N(μ, 1.2²), varying μ from 0 to 5
    Target: N(0, 1)

      Mean Shift      ESS  Rel ESS       CV     Max Wt
    ------------------------------------------------------------
             0.0   1904.1    95.2%    0.224   0.000599
             0.5   1662.7    83.1%    0.450   0.000797
             1.0   1146.6    57.3%    0.863   0.001816
             2.0    237.1    11.9%    2.727   0.026472
             3.0     53.2     2.7%    6.048   0.060927
             4.0      5.7     0.3%   18.714   0.295564
             5.0      4.0     0.2%   22.472   0.445987

    ======================================================================
    EXAMPLE 3: Tail Coverage Assessment
    ======================================================================

    꼬리 두꺼운 타깃에서는 제안이 꼬리를 넉넉히 덮어야 한다.
    스튜던트 t 타깃을 꼬리 두께가 다른 제안들과 견준다.


    Target: Student-t(df=3)
    Proposal                  ESS  Rel ESS     χ² Est Status
    -----------------------------------------------------------------
    Heavy: t(3)            2912.6    97.1%       0.03 ✓ Good
    Medium: t(5)           2903.2    96.8%       0.03 ✓ Good
    Light: Normal          2744.2    91.5%       0.09 ✓ Good

    Key insight: Proposals with lighter tails than target can fail!

    ======================================================================
    EXAMPLE 4: Diagnosing Coverage for Multimodal Target
    ======================================================================

    Target: 0.5*N(-3,1) + 0.5*N(3,1) [bimodal]
    Proposal                  ESS  Rel ESS    50% Wt% Coverage
    ----------------------------------------------------------------------
    Single wide            1554.2    51.8%      19.7% ✓ Both modes
    Single narrow           363.4    12.1%       4.2% ✓ Both modes
    Centered on mode         93.1     3.1%      22.2% ⚠ Missing mode

    Key insight: Single-component proposals struggle with multimodal targets!

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 꼭 필요한 진단:
       - ESS와 상대 ESS(가장 중요하다)
       - 웨이트의 변동계수
       - 웨이트 집중(50%, 90% 백분위수)
       - 최대 웨이트 값

    2. 좋은 제안의 표지:
       - 상대 ESS > 0.1(되도록 > 0.3)
       - 웨이트의 변동계수 < 3(낮을수록 좋다)
       - 웨이트의 50%를 차지하는 표본이 10% 미만
       - 어느 웨이트도 혼자 지배하지 않는다

    3. 경고 신호:
       - ESS << n(n의 1% 미만)
       - 몇몇 표본이 웨이트의 대부분을 진다
       - 변동계수가 아주 높음(> 5-10)
       - KL이나 χ² 발산 추정값이 크다

    4. 꼬리 커버리지:
       - 꼬리 두꺼운 타깃에 결정적이다
       - 최대 웨이트 비 살피기
       - 제안은 꼬리가 더 두꺼워야 한다
       - 꼬리 얇은 제안은 크게 무너질 수 있다

    5. 다봉 타깃:
       - 성분 하나로는 흔히 모드를 놓친다
       - 표본이 모드를 모두 덮는지 확인하기
       - 대개 혼합 제안이 필요하다
       - 아니면 적응적 방법 쓰기

    6. 성능 저하 패턴:
       - 제안과 타깃의 어긋남이 커질수록 ESS가 떨어진다
       - 거리에 따라 흔히 지수적으로 나빠진다
       - 작게 옮겨도 큰 효과가 있을 수 있다

    7. 실전 작업 흐름:
       가) 결과를 믿기 전에 늘 진단을 계산하라
       나) 될 수 있으면 여러 제안 견주기
       다) 경고 신호 살펴보기(ESS, 변동계수, 집중)
       라) 웨이트 분포 그려 보기
       마) 중요한 영역의 커버리지 확인하기

    8. 진단이 나쁠 때의 대처:
       - 제안 파라메터 조정하기(평균, 분산)
       - 꼬리가 더 두꺼운 제안 분포족 쓰기
       - 혼합 제안으로 바꾸기
       - 적응적 중요도 샘플링 생각해 보기
       - 아니면 MCMC를 대신 쓰기

    ```


## 2. 논의

### 집중도와 방문 범위

ProposalDiagnostics는 제안에서 표본을 생성하고 SNIS 웨이트를 계산한다. ESS·상대 ESS·CV·엔트로피·최대 웨이트와 누적 질량에 필요한 표본 수를 반환한다. 집중도 진단은 모두 관측된 표본에 의존하므로 방문하지 못한 중요한 모드를 자동으로 찾아내지 못한다.

출력의 좋은 제안 조건 중 웨이트 50%를 표본 10% 미만이 차지한다는 문구는 방향이 반대다. 이는 오히려 질량이 적은 수의 표본에 집중된 상황이다. 구간별 방문 수 50·최대 웨이트 비 10 같은 기준은 임의 휴리스틱이며 꼬리의 안전성이나 충분한 정확도를 보장하지 않는다.

### KL 표기의 방향

정규화된 목표 p와 제안 q에서 n w̄_i가 p(X_i)/q(X_i)에 가까워진다고 하자. 코드의 비가중 평균 mean(log(nw̄))는 적절한 적분 조건에서 E_q log(p/q)=−KL(q‖p)에 접근한다. KL(p‖q)의 양의 추정값이 아니며 작은 보정항을 무시하면 Jensen 부등식으로 유한 표본에서도 0 이하이다.

mean((nw̄−1)²)는 χ²(p‖q)의 플러그인 추정으로 볼 수 있다. 정확한 정규화 웨이트에는 n/ESS−1과 같으므로 독립적인 새 진단 정보가 아니다. 목표를 상수배해도 정규화 웨이트는 같지만 잘못된 지지집합이나 무한 모멘트는 이 계산으로 해결되지 않는다.

### 꼬리와 그림 해석

자유도 3의 t 목표에 정규 제안을 사용하면 목표 밀도비의 이차 모멘트가 발산한다. 자유도 5의 t 제안도 이 목표의 이차 모멘트에서 경계적으로 발산한다. 유한 실행에서 최대 웨이트가 작다고 이 문제를 배제할 수 없다. 관심 함수가 포함되면 필요한 모멘트 조건은 다시 달라진다.

Q-Q Plot이라는 그림은 확률 격자와 정렬한 웨이트를 그린 것이다. 연속 균등분포와의 일반적인 Q-Q 검정이 아니며, 이상적인 동일 웨이트는 1에 집중된 점질량이다. top_10pct_spread 역시 웨이트 상위 10% 표본의 비가중 표준편차로 전체 목표의 덮임이나 신용구간을 나타내지 않는다. 직접 pdf 비의 수치 범위와 빈 입력·모두 0인 웨이트를 별도로 점검한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드의 KL 추정값이 음수일 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    목표·제안이 정규화되고 필요한 적분이 존재하면 비가중 q 평균 log(p/q)는 −KL(q‖p)다. 코드의 정규화 웨이트 n w̄의 로그 평균도 작은 보정항을 제외하면 0 이하이다. 양의 KL(p‖q)라고 읽으면 방향과 부호가 틀린다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
진단 클래스에 필요한 입력 검사를 정리하라.

</div>

??? success "풀이"

    양의 정수 표본 수, 1차원 표본과 밀도 반환 길이, 음이 아닌 목표·양의 제안 밀도와 유한한 양의 웨이트 합을 확인한다. 상위 10%가 빈 집합이 되는 작은 n도 다뤄야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ESS가 높고 양쪽 모드에서 표본을 관측하면 목표를 충분히 덮었다고 할 수 있는가?

</div>

??? success "풀이"

    방문하지 않은 더 먼 영역과 웨이트의 모집단 모멘트를 알 수 없으므로 충분조건이 아니다. 함수별 오차·독립 반복·대안 제안과 이론적 꼬리 조건을 함께 평가한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 뉴럴 네트워크 시험이 ProposalDiagnostics의 인터페이스와 일치하는가?

</div>

??? success "풀이"

    이 클래스는 compute_all_diagnostics와 plot_diagnostics를 제공하며 model(x)·parameters·backward가 없다. 같은 목표·제안에서 균등 웨이트, 집중도 지표의 대수 관계, 잘못된 입력 처리를 점검하는 방식이 적절하다.

    ```python
    def test_proposaldiagnostics():
        model = ProposalDiagnostics(...)
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

제안분포별 웨이트 집중도·모멘트·꼬리와 다봉 영역의 방문을 진단한다. 수치 요약이 추정하는 대상과 관측되지 않은 영역에 대한 한계를 함께 해석한다. 웨이트 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
