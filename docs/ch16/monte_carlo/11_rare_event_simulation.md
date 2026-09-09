# 드문 일 흉내내기

11_rare_event_simulation.py 나아간 수준: 드문 일 흉내내기를 위한 중요도 표집

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

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
같은 표본 묶음에서 중요도 무게를 다시 매겨 얻는다.
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
plt.savefig('/home/claude/03_Importance_Sampling/rare_event_analysis.png',
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

## 2. 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 표집과 어림 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 몬테카를로 방법에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 몬테카를로 방법 구현의 자리에서 치우침과 흩어짐의 주고받음을 설명하여라. 핵심 웃매개변수가 이 주고받음에 어떻게 영향을 주는가?

??? success "연습문제 3 풀이"
    몬테카를로 방법에서 치우침과 흩어짐의 주고받음은 모형의 복잡함과 표본 크기로 드러난다. 더 복잡한 모형(이를테면 섞음 성분이 더 많거나 층이 더 깊은 모형)은 치우침을 줄이지만 흩어짐을 키우며, 자료가 적을 때 특히 그렇다. 핵심 웃매개변수가 이를 다스린다. 앞확률의 세기가 벌주기 노릇을 하고(센 앞확률은 흩어짐을 줄이지만 치우침을 키울 수 있다), 표본 크기가 어림의 정확도에 영향을 주며(표본이 많을수록 흩어짐이 줄고), 모형의 복잡함이 유연함을 정한다. 가장 좋은 균형은 쓸 수 있는 자료의 양과 바탕 분포의 참된 복잡함에 달렸다.

---

**연습문제 4.**
드문 일 흉내내기 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 드문 일 흉내내기

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
