# 층 모형

층 베이즈 모형은 매개변수가 무리마다 달라지도록 하면서도 어울림을 얼마쯤 두어 무리끼리 통계적 힘을 나눈다. 이 기법은 여러 무리를 다룰 때의 근본 긴장을 다룬다. 곧 무리마다 따로 어림할 것인가(어울림 없음), 아니면 모든 무리가 같다고 볼 것인가(온전한 어울림) 하는 것이다. 층 모형은 자료에 따라 오그라들기의 정도를 스스로 맞추는 우아한 가운뎃길을 준다.

## 코드

```python
"""
베이즈 추론 — 모듈 7: 층 베이즈 모형
수준: 나아간 단계
주제: 층 모형, 얼마쯤 어울림, 오그라들기, 여러 층 추론

층 모형은 매개변수가 무리마다 달라지도록 하면서도 얼마쯤 어울림으로
무리끼리 통계적 힘을 나눈다.

지은이: 연세대학교 이성철 교수
전자우편: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ========================================================================
# 메인
# ========================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

"""
층 모형의 짜임:

층 1(자료): y_ij | θ_i ~ N(θ_i, σ²)
층 2(무리): θ_i | μ, τ ~ N(μ, τ²)
층 3(웃매개변수): μ ~ N(μ₀, σ₀²), τ ~ 반코시(눈금)

이렇게 하면 다음이 만들어진다:
- 어울림 없음: θ_i마다 따로 어림한다(무리 짜임을 무시한다)
- 온전한 어울림: 모든 θ_i = μ이라고 놓는다(무리 차이를 무시한다)
- 얼마쯤 어울림: θ_i을 μ 쪽으로 오그라뜨린다(두 극단의 균형을 잡는다)

쓰임새:
- 학교 안의 학생
- 병원 안의 환자
- 갈래 안의 제품
- 피험자 안의 되풀이 측정
"""

def demonstrate_pooling():
    """
    어울림 없음, 온전한 어울림, 얼마쯤 어울림을 견주어 보인다.
    """
    print("="*70)
    print("HIERARCHICAL MODELS: POOLING STRATEGIES")
    print("="*70)
    
    # 자료 흉내내기: 표본 크기가 다른 학교 8곳
    np.random.seed(42)
    true_mu = 8.0
    true_tau = 5.0
    
    n_schools = 8
    true_effects = np.random.normal(true_mu, true_tau, n_schools)
    sample_sizes = np.array([28, 8, 23, 20, 12, 44, 6, 11])
    sigma = 15.0  # 아는 표준 오차
    
    observed_means = []
    for i, n in enumerate(sample_sizes):
        obs = np.random.normal(true_effects[i], sigma/np.sqrt(n))
        observed_means.append(obs)
    observed_means = np.array(observed_means)
    
    print(f"\nTrue population mean: {true_mu:.2f}")
    print(f"True between-school std: {true_tau:.2f}")
    print(f"Within-school std: {sigma:.2f}")
    
    # 어울림 없음: 따로따로 어림
    no_pool_estimates = observed_means
    
    # 온전한 어울림: 큰 평균
    complete_pool_estimate = np.mean(observed_means)
    complete_pool_estimates = np.full(n_schools, complete_pool_estimate)
    
    # 얼마쯤 어울림: 큰 평균 쪽으로 오그라뜨리기
    # 무게 = n_i / (n_i + σ²/τ²)
    tau_est = np.std(observed_means)  # 단순 어림값
    weights = sample_sizes / (sample_sizes + sigma**2 / tau_est**2)
    partial_pool_estimates = (weights * observed_means + 
                             (1 - weights) * complete_pool_estimate)
    
    # 결과 보이기
    print("\n" + "-"*70)
    print(f"{'School':<8} {'n':<5} {'True':<8} {'Observed':<12} {'No Pool':<12} {'Partial Pool':<15} {'Complete Pool':<12}")
    print("-"*70)
    for i in range(n_schools):
        print(f"{i+1:<8} {sample_sizes[i]:<5} {true_effects[i]:<8.2f} {observed_means[i]:<12.2f} "
              f"{no_pool_estimates[i]:<12.2f} {partial_pool_estimates[i]:<15.2f} {complete_pool_estimates[i]:<12.2f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 그림 1: 어림값 견줌
    x = np.arange(n_schools) + 1
    axes[0].plot(x, true_effects, 'ko-', label='True effects', linewidth=2, markersize=8)
    axes[0].plot(x, observed_means, 'b^--', label='Observed (No pooling)', linewidth=2, markersize=8, alpha=0.7)
    axes[0].plot(x, partial_pool_estimates, 'ro-', label='Partial pooling', linewidth=2, markersize=8)
    axes[0].axhline(complete_pool_estimate, color='green', linestyle=':', linewidth=2, label='Complete pooling')
    axes[0].set_xlabel('School', fontsize=12)
    axes[0].set_ylabel('Effect Estimate', fontsize=12)
    axes[0].set_title('Comparing Pooling Strategies', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 그림 2: 오그라들기 그려 보기
    for i in range(n_schools):
        axes[1].plot([observed_means[i], partial_pool_estimates[i]], 
                    [i+1, i+1], 'r-', linewidth=2, alpha=0.7)
        axes[1].plot(observed_means[i], i+1, 'bo', markersize=10, label='Observed' if i==0 else '')
        axes[1].plot(partial_pool_estimates[i], i+1, 'ro', markersize=10, label='Partial pool' if i==0 else '')
    axes[1].axvline(complete_pool_estimate, color='green', linestyle=':', linewidth=2, label='Grand mean')
    axes[1].set_xlabel('Estimate', fontsize=12)
    axes[1].set_ylabel('School', fontsize=12)
    axes[1].set_title('Shrinkage Toward Grand Mean', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_yticks(range(1, n_schools+1))
    
    plt.tight_layout()
    plt.savefig('hierarchical_pooling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Insight:")
    print("  - Schools with smaller samples are shrunk more toward the grand mean")
    print("  - Partial pooling 'borrows strength' across groups")
    print("  - Provides better estimates than complete or no pooling")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 7: HIERARCHICAL MODELS")
    print("="*70)
    
    demonstrate_pooling()
    
    print("\n" + "="*70)
    print("MODULE 7 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Hierarchical models share information across groups")
    print("2. Partial pooling balances group-specific and population estimates")
    print("3. Small groups benefit most from pooling")
    print("4. Shrinkage is automatic and data-driven")
    print("\nNext: Module 8 - Empirical Bayes")
    print("="*70)
```

## 논의

A hierarchical Bayesian model introduces multiple levels of uncertainty. At the lowest level, data within each group are generated from group-specific parameters $\theta_i$. At the next level, these group parameters are drawn from a shared population distribution governed by hyperparameters $\mu$ and $\tau$. This structure creates a natural mechanism for partial pooling: each group's estimate is pulled toward the population mean by an amount that depends on both the group's sample size and the between-group variability.

The shrinkage weight for group $i$ is approximately $w_i = n_i / (n_i + \sigma^2 / \tau^2)$, where $n_i$ is the sample size, $\sigma^2$ is the within-group variance, and $\tau^2$ is the between-group variance. Groups with smaller samples have weights closer to zero, meaning their estimates are pulled more strongly toward the grand mean. This is precisely the behavior we want: when we have little data for a group, we rely more on the population-level information.

코드는 이를 흉내 낸 "여덟 학교" 상황으로 보인다. 어울림 없음(학교마다 따로 어림), 온전한 어울림(모든 학교에 큰 평균을 줌), 얼마쯤 어울림(층 오그라들기)을 견주면 얼마쯤 어울림이 참된 효과에 더 가까운 어림값을 내놓음을 본다. 이 길은 교육 연구, 임상 시험, 스포츠 분석을 비롯해 자료가 저절로 무리 지어지는 어느 분야에서나 널리 쓰인다.

## 연습문제

**연습문제 1.**
$p = 0.4$인 불공평한 놀이(노름꾼에게 불리하게 치우친)를 쓰도록 흉내내기를 고쳐라. 똑같이 무리 8개와 표본 크기로 얼마쯤 어울림 어림값을 셈하고, 세 전략의 평균 제곱 오차(MSE)를 참된 효과에 견주어 보아라.

??? success "연습문제 1 풀이"
    ```python
    np.random.seed(42)
    true_mu = 8.0
    true_tau = 5.0
    n_schools = 8
    true_effects = np.random.normal(true_mu, true_tau, n_schools)
    sample_sizes = np.array([28, 8, 23, 20, 12, 44, 6, 11])
    sigma = 15.0

    observed_means = np.array([
        np.random.normal(true_effects[i], sigma / np.sqrt(n))
        for i, n in enumerate(sample_sizes)
    ])

    grand_mean = np.mean(observed_means)
    tau_est = np.std(observed_means)
    weights = sample_sizes / (sample_sizes + sigma**2 / tau_est**2)
    partial = weights * observed_means + (1 - weights) * grand_mean

    mse_no_pool = np.mean((observed_means - true_effects)**2)
    mse_complete = np.mean((grand_mean - true_effects)**2)
    mse_partial = np.mean((partial - true_effects)**2)

    print(f"MSE No Pooling:       {mse_no_pool:.4f}")
    print(f"MSE Complete Pooling: {mse_complete:.4f}")
    print(f"MSE Partial Pooling:  {mse_partial:.4f}")
    ```
    얼마쯤 어울림은 낱낱의 차이를 지키면서도 모든 무리에서 힘을 빌려 오므로 보통 MSE이 가장 낮다.

---

**연습문제 2.**
Explain why the shrinkage weight $w_i = n_i / (n_i + \sigma^2 / \tau^2)$ approaches 1 as $n_i \to \infty$ and approaches 0 as $n_i \to 0$. What does this mean practically for estimation?

??? success "연습문제 2 풀이"
    When $n_i \to \infty$, the ratio becomes $n_i / n_i = 1$, so the group estimate equals the observed mean with no shrinkage. This makes sense: with infinite data, the group-specific estimate is perfectly reliable and needs no borrowing from other groups.

    When $n_i \to 0$, the ratio becomes $0 / (\sigma^2/\tau^2) = 0$, so the estimate collapses to the grand mean. With no data for a group, the best prediction is the population average.

    실전으로 보면 층 모형은 무리마다의 자료를 믿는 쪽과 모집단 정보에 기대는 쪽 사이를 스스로 맞춘다는 뜻이다. 관측이 많은 무리는 더 낱낱으로 다뤄지고, 자료가 성긴 무리는 함께 쓰는 모집단 어림값에 크게 기댄다.

---

**연습문제 3.**
Extend the code to compute 95% credible intervals for each school's effect under partial pooling. Assume the posterior for each $\theta_i$ is approximately normal with mean equal to the partial pooling estimate and variance $(1/n_i + 1/\tau^2)^{-1} \cdot \sigma^2 / n_i$. Compare interval widths across schools.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    true_mu, true_tau, sigma = 8.0, 5.0, 15.0
    n_schools = 8
    true_effects = np.random.normal(true_mu, true_tau, n_schools)
    sample_sizes = np.array([28, 8, 23, 20, 12, 44, 6, 11])
    observed_means = np.array([
        np.random.normal(true_effects[i], sigma / np.sqrt(n))
        for i, n in enumerate(sample_sizes)
    ])

    grand_mean = np.mean(observed_means)
    tau_est = np.std(observed_means)
    weights = sample_sizes / (sample_sizes + sigma**2 / tau_est**2)
    partial = weights * observed_means + (1 - weights) * grand_mean

    for i in range(n_schools):
        posterior_var = 1.0 / (sample_sizes[i] / sigma**2 + 1.0 / tau_est**2)
        posterior_std = np.sqrt(posterior_var)
        ci_low = partial[i] - 1.96 * posterior_std
        ci_high = partial[i] + 1.96 * posterior_std
        width = ci_high - ci_low
        print(f"School {i+1} (n={sample_sizes[i]}): "
              f"[{ci_low:.2f}, {ci_high:.2f}], width={width:.2f}")
    ```
    표본 크기가 작은 학교는 믿음 구간이 더 넓어 더 큰 불확실함을 드러낸다. 얼마쯤 어울림이 흩어짐을 줄이므로 층 짜임은 어울림 없음 어림값에 견주어 구간을 좁힌다.
