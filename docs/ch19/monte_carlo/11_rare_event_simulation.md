# 희귀 사건 시뮬레이션

희귀 사건의 확률 적분과 손실 분위수를 중요도 샘플링으로 근사한다. 꼬리에 표본을 집중하는 제안과 SNIS 분모의 안정성, 표준오차 계산을 함께 살펴야 한다.

## 1. 코드

??? note "코드 (570줄)"

    ```python
    """
    11_rare_event_simulation.py

    앞선 단계: 드문 일 흉내내기를 위한 중요도 표집

    이 단원은 확률을 어림하는 중요도 표집을 보인다
    여기서는 어수룩한 몬테카를로의 효율이 몹시 낮다.

    수학적 바탕:
    ---------------------
    드문 일: ε << 1일 때 P(X ∈ A) = ε(이를테면 ε = 10⁻⁶)

    어수룩한 몬테카를로:
    - 그럴듯한 정확도에 표본이 n ≈ 1/ε²개 필요하다
    - ε = 10⁻⁶이면 표본이 n ≈ 10¹²개 필요하다(감당할 수 없다!)

    중요도 표집 방식:
    - 드문 구역 A에 확률을 더 두는 제안 q 고르기
    - P(X ∈ A) = E_q[I(X ∈ A) × w(X)]
    - 흩어짐을 지수로 줄일 수 있다

    가장 좋은 제안:
    P(X ∈ A)을 어림할 때:
        q*(x) ∝ I(x ∈ A) × p(x)

    실전에서는 p을 A 쪽으로 기울여 어림한다.

    흔한 기법:
    1. 지수 기울이기(꼬리 얇은 분포에)
    2. 평균 옮기기(가우스에서)
    3. 섞음 제안
    4. 교차 엔트로피 방법

    쓰임새:
    - 위험 분석(금융, 보험)
    - 신뢰성 공학
    - 그물 성능(줄 넘침)
    - 안전이 결정적인 체계

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.optimize import minimize_scalar
    import seaborn as sns
    import time

    np.random.seed(42)
    sns.set_style("whitegrid")


    def estimate_rare_event_mc(target_dist, threshold, n_samples):
        """
        P(X > threshold)의 어수룩한 몬테카를로 어림.
        """
        samples = target_dist.rvs(size=n_samples)
        indicator = (samples > threshold).astype(float)

        estimate = np.mean(indicator)
        std_error = np.std(indicator) / np.sqrt(n_samples)

        return estimate, std_error


    def estimate_rare_event_is(target_dist, proposal_dist, threshold, n_samples):
        """
        중요도 표집으로 P(X > threshold) 어림하기.
        """
        samples = proposal_dist.rvs(size=n_samples)

        # 중요도 무게
        weights_unnorm = target_dist.pdf(samples) / proposal_dist.pdf(samples)
        weights = weights_unnorm / np.sum(weights_unnorm)

        # 지시 함수
        indicator = (samples > threshold).astype(float)

        # 중요도 표집 어림값
        estimate = np.sum(weights * indicator)

        # ESS
        ess = 1.0 / np.sum(weights**2)

        # 어림 표준 오차
        std_error = np.sqrt(np.sum(weights**2 * indicator) - estimate**2)

        return estimate, std_error, ess


    # 보기 1: 가우스의 꼬리 확률
    # ======================================
    print("=" * 70)
    print("EXAMPLE 1: Gaussian Tail Probability")
    print("=" * 70)

    # 과녁: N(0, 1)
    target_gaussian = stats.norm(0, 1)

    # 드문 일: P(X > threshold)
    thresholds = [3.0, 4.0, 5.0, 6.0]

    print("\nEstimating P(X > threshold) for X ~ N(0,1)")
    print(f"{'Threshold':>10} {'True Prob':>12} {'MC Estimate':>12} "
          f"{'IS Estimate':>12} {'Variance Reduction'}")
    print("-" * 75)

    for threshold in thresholds:
        # 참 확률
        true_prob = 1 - target_gaussian.cdf(threshold)

        # 어수룩한 몬테카를로
        n_mc = 100000
        mc_estimate, mc_se = estimate_rare_event_mc(target_gaussian, threshold, n_mc)
        mc_rel_error = mc_se / (true_prob + 1e-10)

        # 옮긴 가우스를 쓴 중요도 표집
        # 가장 좋은 옮김: μ* = 문턱값(조건부 기댓값)
        proposal_shifted = stats.norm(threshold, 1)

        n_is = 10000  # 훨씬 적은 표본이 필요함
        is_estimate, is_se, ess = estimate_rare_event_is(
            target_gaussian, proposal_shifted, threshold, n_is
        )
        is_rel_error = is_se / (true_prob + 1e-10)

        # 흩어짐 줄임 배수
        variance_reduction = (mc_se**2 * n_is) / (is_se**2 * n_mc + 1e-20)

        print(f"{threshold:10.1f} {true_prob:12.2e} {mc_estimate:12.2e} "
              f"{is_estimate:12.2e} {variance_reduction:18.1f}x")

    print("\nKey insight: Exponential variance reduction for rare events!")


    # 보기 2: 차근차근 견주기
    # ==============================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Detailed Analysis for P(X > 4)")
    print("=" * 70)

    threshold_ex2 = 4.0
    true_prob_ex2 = 1 - target_gaussian.cdf(threshold_ex2)

    print(f"\nTrue probability: {true_prob_ex2:.8f} (very rare!)")

    # 서로 다른 제안
    proposals_ex2 = {
        'Naive (N(0,1))': stats.norm(0, 1),
        'Shifted N(4,1)': stats.norm(4, 1),
        'Shifted N(4,1.2)': stats.norm(4, 1.2),
        'Shifted N(5,1)': stats.norm(5, 1),
    }

    n_samples_ex2 = 5000
    n_replications = 500

    print(f"\nComparing proposals ({n_replications} replications, {n_samples_ex2} samples each):")
    print(f"{'Proposal':<18} {'Mean Est':>12} {'Bias':>10} {'RMSE':>10} {'Mean ESS':>10}")
    print("-" * 70)

    for name, proposal in proposals_ex2.items():
        estimates = []
        ess_values = []

        for _ in range(n_replications):
            est, _, ess = estimate_rare_event_is(target_gaussian, proposal,
                                                 threshold_ex2, n_samples_ex2)
            estimates.append(est)
            ess_values.append(ess)

        mean_est = np.mean(estimates)
        bias = mean_est - true_prob_ex2
        rmse = np.sqrt(np.mean((np.array(estimates) - true_prob_ex2)**2))
        mean_ess = np.mean(ess_values)

        print(f"{name:<18} {mean_est:12.8f} {bias:+10.2e} {rmse:10.2e} {mean_ess:10.1f}")


    # 보기 3: 지수 기울이기
    # ============================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Exponential Tilting")
    print("=" * 70)

    print("""
    드문 일 흉내내기를 위한 지수 기울이기:
    X ~ N(μ, σ²)일 때 분포를 기울이면:
        q(x) ∝ p(x) exp(λx)

    이러면 q = N(μ + λσ², σ²)이 된다

    가장 좋은 λ은 흩어짐을 가장 작게 한다.
    """)

    # 가우스에서는 문턱값으로 옮기는 것이 가장 좋다
    threshold_ex3 = 5.0
    true_prob_ex3 = 1 - target_gaussian.cdf(threshold_ex3)

    print(f"\nTarget: N(0,1)")
    print(f"Rare event: P(X > {threshold_ex3})")
    print(f"True probability: {true_prob_ex3:.8f}")

    # 서로 다른 기울임 매개변수 시도하기
    tilts = [0, 1, 2, 3, 4, 5, 6]
    n_samples_ex3 = 3000

    print(f"\n{'Tilt λ':>8} {'Proposal':>15} {'ESS':>10} {'Rel ESS':>10} {'Est Error':>12}")
    print("-" * 65)

    for tilt in tilts:
        # 기울인 제안: N(λ, 1)
        proposal_tilt = stats.norm(tilt, 1)

        # 여러 번 돌리기
        ess_values = []
        estimates = []

        for _ in range(100):
            est, _, ess = estimate_rare_event_is(target_gaussian, proposal_tilt,
                                                 threshold_ex3, n_samples_ex3)
            ess_values.append(ess)
            estimates.append(est)

        mean_ess = np.mean(ess_values)
        mean_est = np.mean(estimates)
        est_error = abs(mean_est - true_prob_ex3)

        print(f"{tilt:8.1f} N({tilt:2.0f}, 1){'':<6} {mean_ess:10.1f} "
              f"{mean_ess/n_samples_ex3:10.1%} {est_error:12.2e}")

    print(f"\nOptimal tilt ≈ {threshold_ex3} (equals threshold)")


    # 보기 4: 금융 위험 - 위험 가치(VaR)
    # =============================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Financial Application - Value at Risk")
    print("=" * 70)

    print("""
    포트폴리오 수익: R ~ N(μ, σ²)
    수준 α의 위험 가치: VaR_α = -quantile(R, α)

    α = 0.001(99.9%)이면 이는 드문 일 어림 문제이다.
    """)

    # 포트폴리오 매개변수
    mu_portfolio = 0.05  # 하루 기댓값 수익(5%)
    sigma_portfolio = 0.20  # 변동성(20%)

    portfolio_dist = stats.norm(mu_portfolio, sigma_portfolio)

    # VaR 수준
    alpha_var = 0.001  # 99.9% VaR
    true_var = -portfolio_dist.ppf(alpha_var)

    print(f"\nPortfolio: μ = {mu_portfolio:.2%}, σ = {sigma_portfolio:.2%}")
    print(f"Estimating 99.9% VaR (probability of extreme loss = {alpha_var:.1%})")
    print(f"True VaR: {true_var:.4f}")

    # 어수룩한 몬테카를로
    n_mc_var = 1000000  # 드문 일에는 표본이 엄청나게 많이 필요함
    start_time = time.time()
    samples_mc = portfolio_dist.rvs(size=n_mc_var)
    losses = -samples_mc
    var_mc = np.quantile(losses, 1-alpha_var)
    time_mc = time.time() - start_time

    print(f"\nNaive MC ({n_mc_var:,} samples):")
    print(f"  VaR estimate: {var_mc:.4f}")
    print(f"  Error: {abs(var_mc - true_var):.4f}")
    print(f"  Time: {time_mc:.2f} seconds")

    # 꼬리에 가운데를 맞춘 제안의 중요도 표집
    proposal_var = stats.norm(mu_portfolio - 3*sigma_portfolio, sigma_portfolio)

    n_is_var = 10000  # 표본이 훨씬 적음
    start_time = time.time()
    samples_is = proposal_var.rvs(size=n_is_var)
    losses_is = -samples_is

    # 무게 셈하기
    weights_var = portfolio_dist.pdf(samples_is) / proposal_var.pdf(samples_is)
    weights_var /= np.sum(weights_var)

    # 무게 분위수
    sorted_indices = np.argsort(losses_is)
    sorted_losses = losses_is[sorted_indices]
    sorted_weights = weights_var[sorted_indices]
    cumsum_weights = np.cumsum(sorted_weights)

    var_is_idx = np.searchsorted(cumsum_weights, 1-alpha_var)
    var_is = sorted_losses[var_is_idx]
    time_is = time.time() - start_time

    print(f"\nIS ({n_is_var:,} samples):")
    print(f"  VaR estimate: {var_is:.4f}")
    print(f"  Error: {abs(var_is - true_var):.4f}")
    print(f"  Time: {time_is:.2f} seconds")
    print(f"  Speedup: {time_mc/time_is:.1f}x faster")
    print(f"  Sample efficiency: {n_mc_var/n_is_var:.0f}x fewer samples")


    # 보기 5: 문턱값 여럿을 한꺼번에
    # ===========================================
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Estimating Multiple Tail Probabilities")
    print("=" * 70)

    print("""
    중요도 표집의 강점: 꼬리 확률 여럿을 어림할 수 있다
    같은 표본 배치에서 중요도 무게를 다시 매겨 얻는다.
    """)

    thresholds_multi = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    true_probs_multi = [1 - target_gaussian.cdf(t) for t in thresholds_multi]

    # 중앙값 문턱값에 가운데를 맞춘 제안으로 중요도 표집 한 번 돌리기
    median_threshold = np.median(thresholds_multi)
    proposal_multi = stats.norm(median_threshold, 1.2)

    n_samples_multi = 20000
    samples_multi = proposal_multi.rvs(size=n_samples_multi)
    weights_multi = target_gaussian.pdf(samples_multi) / proposal_multi.pdf(samples_multi)
    weights_multi /= np.sum(weights_multi)

    print(f"\nUsing single IS run with {n_samples_multi:,} samples")
    print(f"Proposal: N({median_threshold}, 1.2)")
    print(f"\n{'Threshold':>10} {'True Prob':>12} {'IS Estimate':>12} {'Rel Error':>12}")
    print("-" * 55)

    for threshold, true_prob in zip(thresholds_multi, true_probs_multi):
        indicator = (samples_multi > threshold).astype(float)
        is_estimate = np.sum(weights_multi * indicator)
        rel_error = abs(is_estimate - true_prob) / (true_prob + 1e-10)

        print(f"{threshold:10.1f} {true_prob:12.2e} {is_estimate:12.2e} {rel_error:12.1%}")

    print("\nKey insight: Reuse samples for multiple rare event estimates!")


    # 시각화
    # =============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 칸 1: 드문 일에 쓰는 제안
    ax = axes[0, 0]
    x_plot = np.linspace(-1, 7, 1000)
    threshold_vis = 4.0

    ax.plot(x_plot, target_gaussian.pdf(x_plot), 'k-', linewidth=3,
            label='Target N(0,1)', alpha=0.7)
    ax.plot(x_plot, stats.norm(0, 1).pdf(x_plot), 'b--', linewidth=2,
            label='Naive proposal', alpha=0.5)
    ax.plot(x_plot, stats.norm(4, 1).pdf(x_plot), 'r--', linewidth=2,
            label='IS proposal (shifted)')

    # 드문 일 구역 색칠하기
    ax.fill_between(x_plot[x_plot > threshold_vis], 0,
                    target_gaussian.pdf(x_plot[x_plot > threshold_vis]),
                    alpha=0.3, color='orange', label=f'Rare region (x>{threshold_vis})')

    ax.axvline(threshold_vis, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Proposal Design for Rare Events', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 칸 2: 문턱값에 따른 흩어짐 줄임
    ax = axes[1, 0]
    thresholds_vr = np.linspace(2, 6, 20)
    variance_reductions = []

    for thresh in thresholds_vr:
        # 몬테카를로 흩어짐(베르누이 흩어짐 p(1-p) 사용)
        p = 1 - target_gaussian.cdf(thresh)
        var_mc = p * (1 - p)

        # 중요도 표집의 흩어짐(어림)
        # 문턱값으로 가장 좋게 옮기면 흩어짐 줄임 ≈ exp(threshold²/2)
        var_is_approx = var_mc * np.exp(-thresh**2/2)

        vr = var_mc / var_is_approx if var_is_approx > 0 else 1
        variance_reductions.append(vr)

    ax.semilogy(thresholds_vr, variance_reductions, 'b-', linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Variance Reduction Factor', fontsize=12)
    ax.set_title('Variance Reduction vs Event Rarity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # 칸 3: 표본 효율
    ax = axes[0, 1]
    thresholds_eff = [3, 4, 5, 6]
    mc_samples_needed = []
    is_samples_needed = []

    for thresh in thresholds_eff:
        p = 1 - target_gaussian.cdf(thresh)
        # 상대 오차 10%에 대해
        target_rel_error = 0.1

        # 몬테카를로: n ≈ (1-p)/(p * rel_error²)이 필요
        n_mc_needed = (1-p) / (p * target_rel_error**2)

        # 중요도 표집: 대략 100배에서 1000배 낫다
        n_is_needed = n_mc_needed / 500  # 어림

        mc_samples_needed.append(n_mc_needed)
        is_samples_needed.append(n_is_needed)

    x_pos = np.arange(len(thresholds_eff))
    width = 0.35

    ax.bar(x_pos - width/2, np.log10(mc_samples_needed), width,
           label='MC', alpha=0.7, color='blue', edgecolor='black')
    ax.bar(x_pos + width/2, np.log10(is_samples_needed), width,
           label='IS', alpha=0.7, color='green', edgecolor='black')

    ax.set_ylabel('log₁₀(Samples needed)', fontsize=11)
    ax.set_title('Sample Efficiency: MC vs IS\n(for 10% relative error)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'P(X>{t})' for t in thresholds_eff])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 칸 4: 모임 견주기
    ax = axes[1, 1]
    threshold_conv = 4.5
    true_prob_conv = 1 - target_gaussian.cdf(threshold_conv)

    sample_sizes = [100, 500, 1000, 5000, 10000, 50000]

    mc_errors = []
    is_errors = []

    for n_samp in sample_sizes:
        # 몬테카를로
        errors_mc = []
        for _ in range(50):
            est_mc, _ = estimate_rare_event_mc(target_gaussian, threshold_conv, n_samp)
            errors_mc.append(abs(est_mc - true_prob_conv))
        mc_errors.append(np.mean(errors_mc))

        # 중요도 표집
        errors_is = []
        for _ in range(50):
            est_is, _, _ = estimate_rare_event_is(target_gaussian,
                                                   stats.norm(threshold_conv, 1),
                                                   threshold_conv, n_samp)
            errors_is.append(abs(est_is - true_prob_conv))
        is_errors.append(np.mean(errors_is))

    ax.loglog(sample_sizes, mc_errors, 'bo-', linewidth=2, markersize=8,
              label='Naive MC')
    ax.loglog(sample_sizes, is_errors, 'g^-', linewidth=2, markersize=8,
              label='IS')

    # 기준선: 1/√n
    ax.loglog(sample_sizes, 0.01 / np.sqrt(sample_sizes), 'k--',
              linewidth=1.5, alpha=0.5, label='O(1/√n)')

    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Convergence for P(X > {threshold_conv})',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('rare_event_analysis.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 드문 일 문제:
       - ε << 1일 때 P(X ∈ A) = ε(이를테면 ε = 10⁻⁶)
       - 어수룩한 몬테카를로는 효율이 몹시 낮다. 표본이 O(1/ε²)개 필요하다
       - ε = 10⁻⁶이면 표본이 10¹²개쯤 필요하다(불가능하다!)

    2. 중요도 표집은 지수만큼 빠르게 해 준다:
       - 흩어짐 줄임: 10²에서 10⁶배
       - 표본 효율: 표본이 100배에서 1000배 적다
       - 셈이 빨라짐: 10배에서 100배

    3. 가장 좋은 제안 짜기:
       - 드문 일이 일어나는 곳에 확률 두기
       - P(X > threshold)에서는 분포를 문턱값 쪽으로 옮긴다
       - 가우스에서는 가장 좋은 옮김이 μ* = 문턱값이다
       - 일반 원리: 드문 구역 쪽으로 기울인다

    4. 지수 기울이기:
       - q(x) ∝ p(x) exp(λx)
       - 확률을 드문 구역 쪽으로 옮기도록 λ 고르기
       - 가우스에서는 옮긴 가우스가 된다
       - λ을 손으로나 수치로 최적화할 수 있다

    5. 문턱값 여럿:
       - 중요도 표집을 한 번 돌려 확률 여럿을 어림할 수 있다
       - 문턱값이 다를 때 표본의 무게 다시 매기기
       - 몬테카를로를 따로 돌리는 것보다 훨씬 효율적이다
       - 민감도 분석에 쓸모 있다

    6. 금융에서의 쓰임새:
       - 위험 가치(VaR) 어림
       - 신용 위험(부도 확률)
       - 극단 손실 어림
       - 옵션 값매김(깊은 외가격)

    7. 신뢰성 공학:
       - 체계가 무너질 확률
       - 부품 수명 넘김
       - 안전 여유
       - 극한 시험

    8. 실전 지침:
       - 제안의 평균을 드문 구역으로 옮기기
       - 과녁보다 조금 큰 흩어짐 쓰기
       - 제안을 확인하려고 ESS 살피기
       - 손으로 구한 경계로 어림값 확인하기

    9. 어려움:
       - 좋은 제안 찾기(최적화가 필요할 수 있다)
       - 차원 높은 드문 일(더 어렵다)
       - 서로 떨어진 드문 구역 여럿
       - 아주 극단인 확률(< 10⁻¹⁰)

    10. 이론의 바탕:
        - 지수로 기울인 분포
        - 큰 벗어남 이론
        - 중요도 표집 무게
        - 흩어짐의 경계

    11. 셈의 효율:
        - 중요도 표집: 보통 표본 O(10³-10⁴)개면 넉넉하다
        - 몬테카를로: 표본 O(10⁶-10⁹)개가 필요하다
        - 시간 절약: 100배에서 1000배
        - 실시간 위험 살피기를 가능하게 한다

    12. 확인:
        - (있으면) 손으로 구한 결과와 견주기
        - ESS 살피기(그럴듯해야 하고 아주 작으면 안 된다)
        - 제안 매개변수에 예민함
        - 서로 다른 제안으로 교차 확인하기

    13. 중요도 표집이 꼭 필요할 때:
        - 확률 < 10⁻³
        - 계산 예산이 적을 때
        - 실시간 요구
        - 차원 높은 문제
        - 금융이나 안전이 결정적인 쓰임새

    14. 다른 길과 견주기:
        - 몬테카를로: 단순하지만 드문 일에는 효율이 낮다
        - 손으로 풀기: 흔히 불가능하다
        - 점근 어림: 덜 정확하다
        - 중요도 표집: 정확도와 효율의 균형이 가장 좋다
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (193줄)"

    ```
    ======================================================================
    EXAMPLE 1: Gaussian Tail Probability
    ======================================================================

    Estimating P(X > threshold) for X ~ N(0,1)
     Threshold    True Prob  MC Estimate  IS Estimate Variance Reduction
    ---------------------------------------------------------------------------
           3.0     1.35e-03     1.31e-03     1.64e-03                nanx
           4.0     3.17e-05     1.00e-05     4.19e-05                nanx
           5.0     2.87e-07     0.00e+00     1.59e-06                nanx
           6.0     9.87e-10     0.00e+00     2.87e-08                nanx

    Key insight: Exponential variance reduction for rare events!

    ======================================================================
    EXAMPLE 2: Detailed Analysis for P(X > 4)
    ======================================================================

    True probability: 0.00003167 (very rare!)

    Comparing proposals (500 replications, 5000 samples each):
    Proposal               Mean Est       Bias       RMSE   Mean ESS
    ----------------------------------------------------------------------
    Naive (N(0,1))       0.00002960  -2.07e-06   7.44e-05     5000.0
    Shifted N(4,1)       0.00007011  +3.84e-05   5.45e-05       10.5
    Shifted N(4,1.2)     0.00004093  +9.26e-06   1.95e-05       15.3
    Shifted N(5,1)       0.00027453  +2.43e-04   3.38e-04        5.5

    ======================================================================
    EXAMPLE 3: Exponential Tilting
    ======================================================================

    드문 일 흉내내기를 위한 지수 기울이기:
    X ~ N(μ, σ²)일 때 분포를 기울이면:
        q(x) ∝ p(x) exp(λx)
        
    이러면 q = N(μ + λσ², σ²)이 된다

    가장 좋은 λ은 흩어짐을 가장 작게 한다.


    Target: N(0,1)
    Rare event: P(X > 5.0)
    True probability: 0.00000029

      Tilt λ        Proposal        ESS    Rel ESS    Est Error
    -----------------------------------------------------------------
         0.0 N( 0, 1)           3000.0     100.0%     2.87e-07
         1.0 N( 1, 1)           1112.4      37.1%     3.53e-08
         2.0 N( 2, 1)            128.5       4.3%     2.27e-08
         3.0 N( 3, 1)             25.1       0.8%     9.73e-08
         4.0 N( 4, 1)              8.5       0.3%     4.12e-07
         5.0 N( 5, 1)              5.4       0.2%     2.92e-06
         6.0 N( 6, 1)              3.8       0.1%     3.83e-05

    Optimal tilt ≈ 5.0 (equals threshold)

    ======================================================================
    EXAMPLE 4: Financial Application - Value at Risk
    ======================================================================

    포트폴리오 수익: R ~ N(μ, σ²)
    수준 α의 위험 가치: VaR_α = -quantile(R, α)

    α = 0.001(99.9%)이면 이는 드문 일 어림 문제이다.


    Portfolio: μ = 5.00%, σ = 20.00%
    Estimating 99.9% VaR (probability of extreme loss = 0.1%)
    True VaR: 0.5680

    Naive MC (1,000,000 samples):
      VaR estimate: 0.5665
      Error: 0.0016
      Time: 0.25 seconds

    IS (10,000 samples):
      VaR estimate: 0.5601
      Error: 0.0079
      Time: 0.00 seconds
      Speedup: 119.0x faster
      Sample efficiency: 100x fewer samples

    ======================================================================
    EXAMPLE 5: Estimating Multiple Tail Probabilities
    ======================================================================

    중요도 표집의 강점: 꼬리 확률 여럿을 어림할 수 있다
    같은 표본 배치에서 중요도 무게를 다시 매겨 얻는다.


    Using single IS run with 20,000 samples
    Proposal: N(4.5, 1.2)

     Threshold    True Prob  IS Estimate    Rel Error
    -------------------------------------------------------
           3.0     1.35e-03     1.88e-03        39.6%
           3.5     2.33e-04     3.27e-04        40.7%
           4.0     3.17e-05     4.38e-05        38.2%
           4.5     3.40e-06     4.58e-06        34.7%
           5.0     2.87e-07     3.89e-07        35.7%
           5.5     1.90e-08     2.57e-08        35.2%
           6.0     9.87e-10     1.33e-09        32.0%

    Key insight: Reuse samples for multiple rare event estimates!

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 드문 일 문제:
       - ε << 1일 때 P(X ∈ A) = ε(이를테면 ε = 10⁻⁶)
       - 어수룩한 몬테카를로는 효율이 몹시 낮다. 표본이 O(1/ε²)개 필요하다
       - ε = 10⁻⁶이면 표본이 10¹²개쯤 필요하다(불가능하다!)

    2. 중요도 표집은 지수만큼 빠르게 해 준다:
       - 흩어짐 줄임: 10²에서 10⁶배
       - 표본 효율: 표본이 100배에서 1000배 적다
       - 셈이 빨라짐: 10배에서 100배

    3. 가장 좋은 제안 짜기:
       - 드문 일이 일어나는 곳에 확률 두기
       - P(X > threshold)에서는 분포를 문턱값 쪽으로 옮긴다
       - 가우스에서는 가장 좋은 옮김이 μ* = 문턱값이다
       - 일반 원리: 드문 구역 쪽으로 기울인다

    4. 지수 기울이기:
       - q(x) ∝ p(x) exp(λx)
       - 확률을 드문 구역 쪽으로 옮기도록 λ 고르기
       - 가우스에서는 옮긴 가우스가 된다
       - λ을 손으로나 수치로 최적화할 수 있다

    5. 문턱값 여럿:
       - 중요도 표집을 한 번 돌려 확률 여럿을 어림할 수 있다
       - 문턱값이 다를 때 표본의 무게 다시 매기기
       - 몬테카를로를 따로 돌리는 것보다 훨씬 효율적이다
       - 민감도 분석에 쓸모 있다

    6. 금융에서의 쓰임새:
       - 위험 가치(VaR) 어림
       - 신용 위험(부도 확률)
       - 극단 손실 어림
       - 옵션 값매김(깊은 외가격)

    7. 신뢰성 공학:
       - 체계가 무너질 확률
       - 부품 수명 넘김
       - 안전 여유
       - 극한 시험

    8. 실전 지침:
       - 제안의 평균을 드문 구역으로 옮기기
       - 과녁보다 조금 큰 흩어짐 쓰기
       - 제안을 확인하려고 ESS 살피기
       - 손으로 구한 경계로 어림값 확인하기

    9. 어려움:
       - 좋은 제안 찾기(최적화가 필요할 수 있다)
       - 차원 높은 드문 일(더 어렵다)
       - 서로 떨어진 드문 구역 여럿
       - 아주 극단인 확률(< 10⁻¹⁰)

    10. 이론의 바탕:
        - 지수로 기울인 분포
        - 큰 벗어남 이론
        - 중요도 표집 무게
        - 흩어짐의 경계

    11. 셈의 효율:
        - 중요도 표집: 보통 표본 O(10³-10⁴)개면 넉넉하다
        - 몬테카를로: 표본 O(10⁶-10⁹)개가 필요하다
        - 시간 절약: 100배에서 1000배
        - 실시간 위험 살피기를 가능하게 한다

    12. 확인:
        - (있으면) 손으로 구한 결과와 견주기
        - ESS 살피기(그럴듯해야 하고 아주 작으면 안 된다)
        - 제안 매개변수에 예민함
        - 서로 다른 제안으로 교차 확인하기

    13. 중요도 표집이 꼭 필요할 때:
        - 확률 < 10⁻³
        - 계산 예산이 적을 때
        - 실시간 요구
        - 차원 높은 문제
        - 금융이나 안전이 결정적인 쓰임새

    14. 다른 길과 견주기:
        - 몬테카를로: 단순하지만 드문 일에는 효율이 낮다
        - 손으로 풀기: 흔히 불가능하다
        - 점근 어림: 덜 정확하다
        - 중요도 표집: 정확도와 효율의 균형이 가장 좋다

    ```


## 2. 논의

### 상대 오차와 표본 수

직접 몬테카를로의 사건 지시변수 평균은 분산 p(1−p)/n을 갖는다. 상대 표준오차 r에는 n≈(1−p)/(pr²)가 필요하므로 고정 r에서 O(1/p)이며 출력의 O(1/p²)는 틀리다. 이 식은 고확률 상대 오차 보증과 다르다. 사건이 한 번도 나오지 않아 경험적 SE가 0이어도 확률을 정확히 안다는 뜻은 아니다.

### 표준 IS와 현재 SNIS

정규화된 p를 알면 표준 IS는 n⁻¹∑[p(X_i)/q(X_i)]1_A(X_i)다. 최적 제안은 사건에 조건부인 p이고 이를 직접 알면 분산이 0이지만 그 정규화 상수 자체가 미지 확률이다. 정규 평균을 문턱값 근처로 옮기는 것은 실용적인 근사이며 문턱값과 정확히 같은 이동이 항상 분산 최소값은 아니다.

현재 함수는 가중치를 합으로 나누므로 SNIS다. 오른쪽 꼬리만 주로 방문하면 분모에 필요한 목표 중심을 거의 방문하지 않아 비율 추정이 불안정할 수 있다. 가중치 ESS 최대화와 희귀 지시함수의 분산 최소화도 같은 목적이 아니다.

### 표준오차와 효율 그래프의 한계

코드의 sqrt(∑w̄²1_A−Î²)는 올바른 SNIS 표준오차가 아니며 제곱근 안이 음수가 될 수 있다. 적절한 조건에서 SNIS 점근 분산을 추정하는 기본식은 ∑w̄_i²(1_A(X_i)−Î)²이며, 작은 표본의 안정성·편향은 별도로 검토해야 한다. 이 잘못된 SE로 만든 표의 분산 감소율은 근거로 사용할 수 없다.

서로 다른 표본 수에서 표본당 분산을 비교하려면 n_MC·SE_MC²와 n_IS·SE_IS²를 비교해야 하는데 현재 식은 표본 수 비율도 반대로 적용한다. 그림의 지수 분산 감소 곡선은 가정한 공식을 그렸고 IS 필요 표본 수는 MC 값을 임의로 500으로 나눴다. 실제 방법의 검증 결과가 아니다. 일반적인 100배·1000배 속도 향상이나 고정 표본 수 보장으로 읽지 않는다.

### 지수 틸팅과 손실 분위수

정규 p=N(μ,σ²)의 지수 틸팅 q∝p exp(λx)는 N(μ+λσ²,σ²)다. 일반 분포에는 적률생성함수가 유한한 λ가 필요하다. 동일 가중치 배치로 여러 문턱값을 계산할 수 있지만 각 사건의 오차는 다르고 추정값들도 서로 의존한다.

손실 L=−R의 99.9% 분위수는 연속분포에서 −Q_R(0.001)이다. 코드의 가중 분위수는 손실값을 정렬한 누적 가중치로 계산한다. 주어진 정규분포 안에서의 수치 예제이며 실제 손실 모형의 타당성이나 규제 충족을 검증한 것은 아니다. 꼬리확률의 1−cdf 계산도 극단값에서는 survival function보다 정밀도가 떨어질 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
확률 p의 상대 표준오차를 10%로 만들 직접 MC 표본 수를 유도하라.

</div>

??? success "연습문제 1 풀이"

    sqrt(p(1−p)/n)/p=0.1에서 n≈100(1−p)/p다. p=10⁻⁶이면 약 10⁸이며 10¹²라는 출력 주장은 이 상대 SE 기준과 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
희귀 사건 IS의 입력 조건과 실패를 설명하라.

</div>

??? success "연습문제 2 풀이"

    양의 표본 수와 유효한 분포·문턱값, 사건의 필요한 지지집합을 확인한다. SNIS에는 전체 목표의 정규화 분모도 필요하다. 미관측 사건·모두 0 또는 비유한 가중치와 잘못된 SE를 성공으로 처리하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
꼬리로 이동한 제안에서 낮은 가중치 ESS가 곧 나쁜 확률 추정을 뜻하는가?

</div>

??? success "연습문제 3 풀이"

    표준 IS의 관심량은 가중 사건 지시변수이므로 전체 가중치 ESS와 오차가 반드시 같은 방향은 아니다. 현재 SNIS는 분모까지 추정하므로 중심 미방문의 불안정성도 추가로 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 신경망 시험 대신 희귀 사건 추정을 확인할 기준을 제시하라.

</div>

??? success "연습문제 4 풀이"

    정규 survival function 기준값, 표준 IS·SNIS 구분, 반복 분산과 표준오차의 일치, 가중 분위수의 정의를 확인한다. 아래 가상 클래스·backward 시험은 이 확률 적분 함수와 맞지 않는다.

    ```python
    def test_rare event simulation():
        model = Rare Event Simulation(...)
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

희귀 사건의 확률 적분과 손실 분위수를 중요도 샘플링으로 근사한다. 꼬리에 표본을 집중하는 제안과 SNIS 분모의 안정성, 표준오차 계산을 함께 살펴야 한다. 가중치 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
