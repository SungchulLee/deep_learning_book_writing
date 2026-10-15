# 유효 표본 크기

여러 제안분포에서 가중치 ESS·엔트로피·누적 집중도를 비교한다. ESS와 실제 함수 추정 오차의 관계를 반복 실험으로 살피되 보편적인 분산 공식으로 해석하지 않는다.

## 1. 코드

??? note "코드 (504줄)"

    ```python
    """
    04_effective_sample_size.py

    중급 단계: 실효 표본 크기(ESS) - 이론과 진단

    이 단원은 ESS을 두루 다룬다:
    - 수학의 바탕
    - ESS의 여러 가지 표현
    - 진단과 풀이
    - 흩어짐과의 관계

    수학적 바탕:
    ---------------------
    ESS은 독립 표본의 "실효" 개수를 잰다
    중요도 표집으로 얻은.

    표준 정의:
        ESS = 1 / Σᵢ wᵢ²

    여기서 wᵢ은 고르게 한 무게이다.

    다른 표현(고르게 하지 않은 무게 사용):
        ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²

    성질:
    - 1 ≤ ESS ≤ n
    - 무게가 모두 같으면 ESS = n이다(완벽한 표집)
    - 무게 하나가 판치면 ESS = 1이다(찌부러짐)
    - ESS/n은 "상대 효율"이다

    흩어짐과의 관계:
        Var[Ê] ≈ Var_π[h(θ)] / ESS

    따라서 ESS이 낮을수록 → 어림값의 흩어짐이 커진다.

    지은이: 베이즈 추론 교육 자료
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    from typing import Tuple, Dict

    np.random.seed(42)
    sns.set_style("whitegrid")


    def compute_ess_normalized(weights: np.ndarray) -> float:
        """
        고르게 한 무게를 써서 ESS 셈하기.

        ESS = 1 / Σᵢ wᵢ²

        매개변수:
        -----------
        weights : 고르게 한 무게(합이 1)

        반환값:
        --------
        ess : float
        """
        return 1.0 / np.sum(weights**2)


    def compute_ess_unnormalized(unnormalized_weights: np.ndarray) -> float:
        """
        고르게 하지 않은 무게를 써서 ESS 셈하기.

        ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²

        이는 고르게 한 판과 같지만 수치로 더 안정하다
        무게가 아주 작을 때.

        매개변수:
        -----------
        unnormalized_weights : 고르게 하지 않은 무게

        반환값:
        --------
        ess : float
        """
        sum_weights = np.sum(unnormalized_weights)
        sum_weights_squared = np.sum(unnormalized_weights**2)
        return sum_weights**2 / sum_weights_squared


    def compute_weight_statistics(weights: np.ndarray) -> Dict:
        """
        중요도 무게의 통계량을 두루 셈하기.

        여러 진단을 담은 사전을 돌려준다.
        """
        n = len(weights)
        ess = compute_ess_normalized(weights)

        # 무게의 변이 계수
        cv = np.std(weights) / (np.mean(weights) + 1e-10)

        # 최대 무게
        max_weight = np.max(weights)

        # 무게 분포의 엔트로피
        # 엔트로피가 높을수록 → 무게가 더 고르다
        entropy = -np.sum(weights * np.log(weights + 1e-300))
        max_entropy = np.log(n)  # 고른 분포
        normalized_entropy = entropy / max_entropy

        # 헷갈림도(ESS과 이어진 또 하나의 잣대)
        perplexity = np.exp(entropy)

        # 위쪽 표본이 차지하는 무게의 백분율
        sorted_weights = np.sort(weights)[::-1]
        cumsum_weights = np.cumsum(sorted_weights)
        top_10_pct = np.searchsorted(cumsum_weights, 0.10) + 1
        top_50_pct = np.searchsorted(cumsum_weights, 0.50) + 1
        top_90_pct = np.searchsorted(cumsum_weights, 0.90) + 1

        return {
            'ess': ess,
            'relative_ess': ess / n,
            'cv': cv,
            'max_weight': max_weight,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'perplexity': perplexity,
            'top_10_pct': top_10_pct,
            'top_50_pct': top_50_pct,
            'top_90_pct': top_90_pct,
            'n_samples': n
        }


    def estimate_variance_inflation(weights: np.ndarray) -> float:
        """
        흩어짐 부풂 배수 어림하기.

        Var[중요도 표집 어림꼴] ≈ Var[몬테카를로 어림꼴] × (1 + Var[w])

        여기서 w은 고르게 하지 않은 중요도 비이다.

        반환값:
        --------
        inflation : float
            흩어짐이 부푸는 배수
        """
        ess = compute_ess_normalized(weights)
        n = len(weights)
        # 흩어짐 부풂 ≈ n/ESS
        return n / ess


    # 보기 1: 제안 분포에 따른 ESS
    # =================================================
    print("=" * 70)
    print("EXAMPLE 1: ESS Depends on Proposal Quality")
    print("=" * 70)

    # 과녁 분포: N(5, 1)
    target = stats.norm(5, 1)

    # 여러 가지 제안 분포
    proposals = {
        'Perfect': stats.norm(5, 1),           # 과녁과 같음
        'Good': stats.norm(5, 1.2),            # 가까움
        'Okay': stats.norm(4.5, 1.5),          # 그럴듯함
        'Poor': stats.norm(3, 2),              # 어긋남
        'Bad': stats.norm(5, 0.5),             # 너무 좁음
        'Terrible': stats.norm(0, 1),          # 아주 멂
    }

    n_samples = 2000
    print(f"\nAnalyzing ESS for {n_samples} samples:\n")
    print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Entropy':>8} {'Top 10%':>8}")
    print("-" * 70)

    results_dict = {}

    for name, proposal in proposals.items():
        # 제안에서 표집
        samples = proposal.rvs(size=n_samples)

        # 무게 셈하기
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)

        # 통계를 셈한다
        stats_dict = compute_weight_statistics(weights)
        results_dict[name] = stats_dict

        print(f"{name:<12} {stats_dict['ess']:8.1f} {stats_dict['relative_ess']:8.2%} "
              f"{stats_dict['cv']:8.2f} {stats_dict['normalized_entropy']:8.2%} "
              f"{stats_dict['top_10_pct']:8d}")


    # 가중치 분포 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (name, proposal) in enumerate(proposals.items()):
        samples = proposal.rvs(size=n_samples)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)

        ax = axes[idx]
        ax.hist(weights * n_samples, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5)

        ess = compute_ess_normalized(weights)
        ax.set_title(f'{name}: ESS={ess:.1f} ({ess/n_samples:.1%})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Weight × n', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Uniform')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig('ess_proposal_comparison.png',
                dpi=300, bbox_inches='tight')


    # 보기 2: ESS과 흩어짐의 관계
    # ==============================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: ESS and Estimation Variance")
    print("=" * 70)

    # 과녁: N(5, 1)
    # E[θ²] 어림하기
    h_function = lambda x: x**2
    true_value = 5**2 + 1**2  # N(5,1)의 E[θ²]

    # ESS이 다른 여러 제안
    proposals_var = {
        'High ESS': stats.norm(5, 1.1),
        'Medium ESS': stats.norm(4, 1.5),
        'Low ESS': stats.norm(2, 2),
    }

    n_samples = 1000
    n_replications = 500

    print(f"\nEstimating E[θ²] = {true_value:.3f}")
    print(f"Replications: {n_replications}, Samples per replication: {n_samples}\n")
    print(f"{'Proposal':<12} {'ESS':>8} {'Bias':>10} {'Std Dev':>10} {'RMSE':>10}")
    print("-" * 60)

    for name, proposal in proposals_var.items():
        estimates = []
        ess_values = []

        for _ in range(n_replications):
            samples = proposal.rvs(size=n_samples)
            unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
            weights = unnorm_weights / np.sum(unnorm_weights)

            estimate = np.sum(weights * h_function(samples))
            estimates.append(estimate)
            ess_values.append(compute_ess_normalized(weights))

        mean_ess = np.mean(ess_values)
        bias = np.mean(estimates) - true_value
        std_dev = np.std(estimates)
        rmse = np.sqrt(np.mean((np.array(estimates) - true_value)**2))

        print(f"{name:<12} {mean_ess:8.1f} {bias:+10.4f} {std_dev:10.4f} {rmse:10.4f}")

    # 흩어짐과 ESS 그려 보기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 흩뿌림 그림: ESS과 표준편차(경험값)
    ess_vs_std = []
    for name, proposal in proposals_var.items():
        estimates = []
        ess_values = []

        for _ in range(100):
            samples = proposal.rvs(size=n_samples)
            unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
            weights = unnorm_weights / np.sum(unnorm_weights)

            estimate = np.sum(weights * h_function(samples))
            estimates.append(estimate)
            ess_values.append(compute_ess_normalized(weights))

        for e, s in zip(ess_values, estimates):
            ess_vs_std.append((e, s))

    ess_array = np.array([x[0] for x in ess_vs_std])
    std_array = np.array([x[1] for x in ess_vs_std])

    ax = axes[0]
    ax.scatter(ess_array, std_array, alpha=0.5, s=30, color='steelblue', edgecolors='black')
    ax.set_xlabel('ESS', fontsize=12)
    ax.set_ylabel('Estimate', fontsize=12)
    ax.set_title('ESS vs Estimate Distribution', fontsize=13, fontweight='bold')
    ax.axhline(true_value, color='red', linestyle='--', linewidth=2, label='True value')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 이론의 흩어짐 부풂
    ax = axes[1]
    ess_range = np.linspace(10, n_samples, 100)
    variance_inflation = n_samples / ess_range

    ax.plot(ess_range, variance_inflation, 'b-', linewidth=2,
            label='Variance inflation = n/ESS')
    ax.set_xlabel('ESS', fontsize=12)
    ax.set_ylabel('Variance Inflation Factor', fontsize=12)
    ax.set_title('Theoretical Variance Inflation', fontsize=13, fontweight='bold')
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='No inflation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('ess_variance_relationship.png',
                dpi=300, bbox_inches='tight')


    # 보기 3: 표본 크기의 함수로 본 ESS
    # =======================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: How ESS Scales with Sample Size")
    print("=" * 70)

    # 붙박인 제안에서 ESS은 n에 따라 어떻게 자라나?
    proposal_fixed = stats.norm(4, 1.5)
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

    print("\nSample Size vs ESS (averaged over 100 runs):\n")
    print(f"{'n':>8} {'Mean ESS':>10} {'ESS/n':>10} {'Std ESS':>10}")
    print("-" * 42)

    ess_by_n = []
    for n in sample_sizes:
        ess_list = []
        for _ in range(100):
            samples = proposal_fixed.rvs(size=n)
            unnorm_weights = target.pdf(samples) / proposal_fixed.pdf(samples)
            weights = unnorm_weights / np.sum(unnorm_weights)
            ess_list.append(compute_ess_normalized(weights))

        mean_ess = np.mean(ess_list)
        std_ess = np.std(ess_list)
        ess_by_n.append((n, mean_ess, std_ess))

        print(f"{n:8d} {mean_ess:10.1f} {mean_ess/n:10.3f} {std_ess:10.1f}")

    # n에 따른 ESS 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ns = [x[0] for x in ess_by_n]
    ess_means = [x[1] for x in ess_by_n]
    ess_stds = [x[2] for x in ess_by_n]
    rel_ess = [e/n for e, n in zip(ess_means, ns)]

    ax = axes[0]
    ax.errorbar(ns, ess_means, yerr=ess_stds, fmt='o-', linewidth=2,
                markersize=8, capsize=5, color='steelblue', label='ESS')
    ax.plot(ns, ns, 'r--', linewidth=2, label='Perfect (ESS=n)')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('ESS vs Sample Size', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ns, rel_ess, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Sample Size n', fontsize=12)
    ax.set_ylabel('Relative ESS (ESS/n)', fontsize=12)
    ax.set_title('Relative ESS vs Sample Size', fontsize=13, fontweight='bold')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('ess_vs_sample_size.png',
                dpi=300, bbox_inches='tight')


    # 보기 4: 무게 몰림 진단
    # ========================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Weight Concentration Analysis")
    print("=" * 70)

    def analyze_weight_concentration(weights: np.ndarray, name: str):
        """
        무게가 어떻게 몰리는지 자세히 살피기.
        """
        n = len(weights)
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)

        # 백분위수 찾기
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        n_for_percentile = []

        for p in percentiles:
            idx = np.searchsorted(cumsum, p/100.0)
            n_for_percentile.append(idx + 1)

        print(f"\n{name}:")
        print(f"  Total samples: {n}")
        print(f"  ESS: {compute_ess_normalized(weights):.1f}")
        print("\n  Weight concentration:")
        for p, n_samples in zip(percentiles, n_for_percentile):
            pct_samples = n_samples / n * 100
            print(f"    {n_samples:5d} samples ({pct_samples:5.1f}%) account for {p}% of weight")

        return sorted_weights, cumsum

    # 제안 셋 견주기
    test_proposals = {
        'Good (ESS high)': stats.norm(5, 1.1),
        'Medium (ESS mid)': stats.norm(4, 1.5),
        'Poor (ESS low)': stats.norm(2, 2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, proposal) in enumerate(test_proposals.items()):
        samples = proposal.rvs(size=2000)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)

        sorted_weights, cumsum = analyze_weight_concentration(weights, name)

        ax = axes[idx]
        ax.plot(np.arange(1, len(sorted_weights)+1), cumsum, 
                linewidth=2, color='steelblue')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Number of Top Samples', fontsize=11)
        ax.set_ylabel('Cumulative Weight', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('weight_concentration.png',
                dpi=300, bbox_inches='tight')

    plt.show()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. 실효 표본 크기(ESS)는 중요도 표집의 질을 재어 준다:
       - ESS = 1/Σᵢwᵢ²(고르게 한 무게)
       - ESS = (Σᵢw̃ᵢ)²/Σᵢw̃ᵢ²(고르게 하지 않은 무게)
       - 범위: 1 ≤ ESS ≤ n

    2. 풀이:
       - ESS/n ≈ 1: 아주 좋음, 무게가 거의 고르다
       - ESS/n ≈ 0.5: 좋음, 실효 표본이 절반
       - ESS/n < 0.1: 나쁨, 더 나은 제안을 생각해 보라
       - ESS/n << 0.01: 나쁨, 몇몇 표본이 판친다

    3. 흩어짐과의 관계:
       - 흩어짐 부풂 ≈ n/ESS
       - ESS이 낮을수록 → 어림값의 흩어짐이 커진다
       - 제안을 낫게 하지 않고 n만 늘려서는 고칠 수 없다

    4. 무게 몰림:
       - ESS이 낮다 → 몇몇 표본이 무게의 대부분을 진다
       - 살피기: 무게의 50%에 표본이 몇 개나 드나?
       - 되도록 여러 표본에 널리 퍼져 있어야 한다

    5. 늘 살펴야 할 진단:
       - ESS과 상대 ESS(ESS/n)
       - 무게의 변이 계수
       - 무게 몰림(위쪽 10%, 50%, 90%)
       - 최대 무게 값
       - 무게 엔트로피

    6. ESS이 낮을 때:
       - n만 늘리지 마라(별 도움이 안 된다)
       - 제안 분포 낫게 하기
       - 맞춰 가는 중요도 표집 생각해 보기
       - 아니면 MCMC 방법으로 바꾸기

    7. ESS은 n에 비례해 커진다:
       - 표본 n개에서 ESS/n ≈ c이면
       - 그러면 아무 n에 대해서도 (대체로) ESS/n ≈ c이다
       - 상대 효율은 대체로 일정하다

    8. 실전 규칙:
       - ESS > 1000: 대개 웬만한 쓰임새에 넉넉하다
       - ESS/n > 0.1: 받아들일 만한 효율
       - ESS/n < 0.01: 반드시 더 나은 제안이 필요하다
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (134줄)"

    ```
    ======================================================================
    EXAMPLE 1: ESS Depends on Proposal Quality
    ======================================================================

    Analyzing ESS for 2000 samples:

    Proposal          ESS  Rel ESS       CV  Entropy  Top 10%
    ----------------------------------------------------------------------
    Perfect        2000.0  100.00%     0.00  100.00%      200
    Good           1903.9   95.20%     0.22   99.61%      167
    Okay           1536.2   76.81%     0.55   97.48%      121
    Poor            728.4   36.42%     1.32   88.85%       52
    Bad              90.7    4.53%     4.59   88.58%        2
    Terrible          3.1    0.15%    25.51   23.02%        1

    ======================================================================
    EXAMPLE 2: ESS and Estimation Variance
    ======================================================================

    Estimating E[θ²] = 26.000
    Replications: 500, Samples per replication: 1000

    Proposal          ESS       Bias    Std Dev       RMSE
    ------------------------------------------------------------
    High ESS        984.8    -0.0116     0.2906     0.2908
    Medium ESS      624.6    -0.0176     0.3579     0.3584
    Low ESS         183.2    -0.0133     0.7018     0.7019

    ======================================================================
    EXAMPLE 3: How ESS Scales with Sample Size
    ======================================================================

    Sample Size vs ESS (averaged over 100 runs):

           n   Mean ESS      ESS/n    Std ESS
    ------------------------------------------
         100       62.5      0.625        3.7
         500      314.0      0.628        7.6
        1000      626.1      0.626       11.2
        2000     1248.5      0.624       15.2
        5000     3127.0      0.625       22.9
       10000     6248.6      0.625       35.4

    ======================================================================
    EXAMPLE 4: Weight Concentration Analysis
    ======================================================================

    Good (ESS high):
      Total samples: 2000
      ESS: 1967.5

      Weight concentration:
          182 samples (  9.1%) account for 10% of weight
          455 samples ( 22.8%) account for 25% of weight
          918 samples ( 45.9%) account for 50% of weight
         1407 samples ( 70.3%) account for 75% of weight
         1729 samples ( 86.5%) account for 90% of weight
         1849 samples ( 92.5%) account for 95% of weight
         1960 samples ( 98.0%) account for 99% of weight

    Medium (ESS mid):
      Total samples: 2000
      ESS: 1254.2

      Weight concentration:
           92 samples (  4.6%) account for 10% of weight
          231 samples ( 11.6%) account for 25% of weight
          488 samples ( 24.4%) account for 50% of weight
          819 samples ( 40.9%) account for 75% of weight
         1144 samples ( 57.2%) account for 90% of weight
         1329 samples ( 66.5%) account for 95% of weight
         1630 samples ( 81.5%) account for 99% of weight

    Poor (ESS low):
      Total samples: 2000
      ESS: 358.9

      Weight concentration:
           22 samples (  1.1%) account for 10% of weight
           57 samples (  2.9%) account for 25% of weight
          128 samples (  6.4%) account for 50% of weight
          238 samples ( 11.9%) account for 75% of weight
          380 samples ( 19.0%) account for 90% of weight
          489 samples ( 24.4%) account for 95% of weight
          716 samples ( 35.8%) account for 99% of weight

    ======================================================================
    KEY TAKEAWAYS
    ======================================================================

    1. 실효 표본 크기(ESS)는 중요도 표집의 질을 재어 준다:
       - ESS = 1/Σᵢwᵢ²(고르게 한 무게)
       - ESS = (Σᵢw̃ᵢ)²/Σᵢw̃ᵢ²(고르게 하지 않은 무게)
       - 범위: 1 ≤ ESS ≤ n

    2. 풀이:
       - ESS/n ≈ 1: 아주 좋음, 무게가 거의 고르다
       - ESS/n ≈ 0.5: 좋음, 실효 표본이 절반
       - ESS/n < 0.1: 나쁨, 더 나은 제안을 생각해 보라
       - ESS/n << 0.01: 나쁨, 몇몇 표본이 판친다

    3. 흩어짐과의 관계:
       - 흩어짐 부풂 ≈ n/ESS
       - ESS이 낮을수록 → 어림값의 흩어짐이 커진다
       - 제안을 낫게 하지 않고 n만 늘려서는 고칠 수 없다

    4. 무게 몰림:
       - ESS이 낮다 → 몇몇 표본이 무게의 대부분을 진다
       - 살피기: 무게의 50%에 표본이 몇 개나 드나?
       - 되도록 여러 표본에 널리 퍼져 있어야 한다

    5. 늘 살펴야 할 진단:
       - ESS과 상대 ESS(ESS/n)
       - 무게의 변이 계수
       - 무게 몰림(위쪽 10%, 50%, 90%)
       - 최대 무게 값
       - 무게 엔트로피

    6. ESS이 낮을 때:
       - n만 늘리지 마라(별 도움이 안 된다)
       - 제안 분포 낫게 하기
       - 맞춰 가는 중요도 표집 생각해 보기
       - 아니면 MCMC 방법으로 바꾸기

    7. ESS은 n에 비례해 커진다:
       - 표본 n개에서 ESS/n ≈ c이면
       - 그러면 아무 n에 대해서도 (대체로) ESS/n ≈ c이다
       - 상대 효율은 대체로 일정하다

    8. 실전 규칙:
       - ESS > 1000: 대개 웬만한 쓰임새에 넉넉하다
       - ESS/n > 0.1: 받아들일 만한 효율
       - ESS/n < 0.01: 반드시 더 나은 제안이 필요하다

    ```


## 2. 논의

### 정의와 입력 조건

유한한 음이 아닌 가중치의 합이 양수일 때 ESS_w=(∑w)²/∑w²이며 정규화하면 1/∑w̄²다. 범위는 [1,n]이다. normalized 함수는 입력 합이 1임을 가정하며 임의 원시 가중치에 그대로 적용하면 안 된다. 음수·NaN·무한대·모두 0인 입력을 확인해야 한다.

모집단 표준편차 방식(ddof=0)의 표본 CV에 대해서는 ESS=n/(1+CV²)가 대수적으로 성립한다. 코드가 평균 분모에 더한 작은 상수는 이 정확한 등식을 근사로 바꾼다. 엔트로피 H=−∑w̄log w̄와 perplexity=exp(H)는 집중도의 다른 요약이며 ESS≤exp(H)≤n이다. n=1에서는 H/log n의 정규화 엔트로피가 정의되지 않는다.

### 표의 열과 추정 분산

Top 10%는 가장 큰 표본 10%가 차지하는 질량이 아니라 누적 가중치 10%에 도달하는 데 필요한 표본 수다. 그림은 표와 다른 난수 표본을 새로 생성하므로 개별 숫자가 일치할 필요는 없다.

함수 estimate_variance_inflation이 반환하는 n/ESS는 집중도에서 유도한 휴리스틱이다. 일반적인 SNIS의 실제 분산비는 관심 함수와 가중치의 결합에 의존하며 이 값과 정확히 같지 않다. 여기서는 N(5,1²)의 E[θ²]=26을 반복 추정한 편향·표준편차·RMSE로 함수별 성능을 별도로 비교한다.

### 표본 수와 제안의 꼬리

고정 제안의 독립 표본에서 E_qW²가 유한하면 ESS/n은 (E_qW)²/E_qW²로 수렴한다. 이 조건 없이 모든 n에서 상대 ESS가 일정하다고 단정할 수 없다. 목표 표준편차 1에 제안 표준편차 0.5인 정규쌍은 밀도비의 이차 적률이 발산하므로 유한 실행의 ESS만으로 안정적인 분산을 결론내리면 안 된다.

표본 수 증가는 적절한 조건에서 실제 오차를 줄인다. 낮은 상대 ESS가 표본 수 증가를 무의미하게 만드는 것은 아니다. 다만 제안을 개선하는 편이 비용 효율적일 수 있다. ESS>1000·ESS/n>0.1 같은 출력 기준은 수렴이나 원하는 오차의 보증이 아니며 함수별 몬테카를로 표준오차와 누락된 영역을 함께 살펴야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
원시·정규화 가중치의 ESS가 같은 이유는?

</div>

??? success "연습문제 1 풀이"

    원시 가중치를 그 합으로 나눠 제곱합의 역수를 취하면 (∑w)²/∑w²가 된다. 공통 양의 상수배에도 값이 변하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
n=1과 가중치 전부 0인 경우를 구분하라.

</div>

??? success "연습문제 2 풀이"

    양의 가중치 하나이면 ESS=1이고 엔트로피는 0이다. 정규화 엔트로피 H/log n은 0/0이다. 전부 0인 가중치는 정규화할 수 없어 ESS 계산을 실패로 처리해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ESS/n이 같으면 두 제안의 모든 함수 추정 오차도 같은가?

</div>

??? success "연습문제 3 풀이"

    아니다. SNIS 점근 분산에는 E_q[W²(h−I)²]가 들어간다. 함수가 큰 영역과 가중치가 큰 영역이 어떻게 겹치는지가 중요하며 ESS는 그 정보를 모두 담지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 신경망 시험 대신 ESS의 어떤 성질을 확인해야 하는가?

</div>

??? success "연습문제 4 풀이"

    동일 가중치에서 n, 한 가중치에 집중할 때 1, 순열·양의 상수배 불변성, 두 계산식의 일치와 잘못된 입력의 처리를 확인한다. 아래 가상 클래스·backward 조각은 ESS 함수의 검증이 아니다.

    ```python
    def test_effective sample size():
        model = Effective Sample Size(...)
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

여러 제안분포에서 가중치 ESS·엔트로피·누적 집중도를 비교한다. ESS와 실제 함수 추정 오차의 관계를 반복 실험으로 살피되 보편적인 분산 공식으로 해석하지 않는다. 가중치 진단과 실제 계산식을 함께 확인하고, 보존된 출력의 고정 결론을 일반적인 보장으로 받아들이지 않는다.
