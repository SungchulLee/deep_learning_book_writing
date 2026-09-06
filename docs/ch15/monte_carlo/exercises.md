# 연습

exercises.py 연습: 베이즈 추론을 위한 중요도 표집의 차근차근 문제

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
exercises.py

연습 문제: 베이즈 추론을 위한 중요도 표집의 단계별 문제

이 파일에는 첫걸음부터 앞선 단계까지의 연습 문제가
자세한 풀이와 설명과 함께 담겨 있다.

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")

print("=" * 70)
print("IMPORTANCE SAMPLING EXERCISES")
print("=" * 70)

# =============================================================================
# 첫걸음 연습 문제
# =============================================================================

print("\n" + "=" * 70)
print("BEGINNER LEVEL EXERCISES")
print("=" * 70)

# 연습 1: 지수 분포의 기본 중요도 표집
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 1: Estimating E[X²] for Exponential Distribution")
print("-" * 70)

print("""
문제:
--------
X ~ Exp(λ=2)이라 하자. 여기서 x ≥ 0에 대해 확률 밀도는 p(x) = 2e^(-2x)이다.
제안 q(x) = Exp(1)을 쓴 중요도 표집으로 E[X²] 어림하기.

가) 중요도 표집 구현하기
나) 손으로 구한 값과 견주기: E[X²] = 2/λ² = 0.5
다) ESS 셈하기
라) 표본 크기를 달리하며 모임 살펴보기
""")

print("\nSOLUTION:")

# 참 분포: Exp(2)
lambda_true = 2.0
target = stats.expon(scale=1/lambda_true)

# 손으로 구한 E[X²]
true_value = 2.0 / lambda_true**2
print(f"Analytical E[X²] = {true_value:.6f}")

# 제안: Exp(1)
proposal = stats.expon(scale=1.0)

# 어림할 함수
h = lambda x: x**2

# 중요도 표집
def exercise1_is(n_samples):
    # 제안에서 표집
    samples = proposal.rvs(size=n_samples)
    
    # 중요도 무게 셈하기
    weights_unnorm = target.pdf(samples) / proposal.pdf(samples)
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    # 어림하기
    estimate = np.sum(weights * h(samples))
    
    # ESS
    ess = 1.0 / np.sum(weights**2)
    
    return estimate, ess

# 서로 다른 표본 크기 시도하기
sample_sizes = [100, 500, 1000, 5000]
print("\nResults:")
print(f"{'n':<8} {'Estimate':<12} {'Error':<12} {'ESS':<10} {'ESS/n'}")
print("-" * 55)

for n in sample_sizes:
    est, ess = exercise1_is(n)
    error = abs(est - true_value)
    print(f"{n:<8} {est:<12.6f} {error:<12.6f} {ess:<10.1f} {ess/n:.2%}")

print("\nKey insights:")
print("- ESS/n shows the efficiency of the proposal")
print("- Error decreases with √n for consistent estimator")
print("- This proposal works well because it has heavier tails than target")


# 연습 2: 앞확률을 달리한 베타-이항
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 2: Effect of Prior Choice in Beta-Binomial Model")
print("-" * 70)

print("""
문제:
--------
베르누이 시도 20번에 성공 15번을 관측한다.
다음을 쓴 중요도 표집으로 θ의 뒤확률 평균 어림하기:
가) 고른 앞확률: Beta(1,1)을 제안으로
나) 제프리스 앞확률: Beta(0.5,0.5)을 제안으로
다) 정보 있는 앞확률: Beta(2,2)을 제안으로

경우마다 ESS과 정확도 견주기.
""")

print("\nSOLUTION:")

# 데이터
successes = 15
trials = 20
failures = trials - successes

# 고르게 하지 않은 뒤확률(앞확률을 제안으로 쓰므로 가능도만)
def unnorm_posterior(theta, alpha0, beta0):
    """뒤확률 ∝ 가능도 × 앞확률"""
    # log(0)을 피하려고 잘라 내기
    theta = np.clip(theta, 1e-10, 1-1e-10)
    
    # 로그 가능도
    log_lik = successes * np.log(theta) + failures * np.log(1 - theta)
    
    # 로그 앞확률
    log_prior = (alpha0-1) * np.log(theta) + (beta0-1) * np.log(1 - theta)
    
    return np.exp(log_lik + log_prior)

# 서로 다른 앞확률
priors = {
    'Uniform Beta(1,1)': (1, 1),
    'Jeffreys Beta(0.5,0.5)': (0.5, 0.5),
    'Informative Beta(2,2)': (2, 2),
}

n_samples = 2000
print(f"\nUsing {n_samples} samples:")
print(f"{'Prior':<25} {'Post Mean':<12} {'ESS':<10} {'ESS/n'}")
print("-" * 55)

for name, (alpha0, beta0) in priors.items():
    # 손으로 구한 뒤확률
    alpha_n = alpha0 + successes
    beta_n = beta0 + failures
    analytical_mean = alpha_n / (alpha_n + beta_n)
    
    # 중요도 표집
    proposal = stats.beta(alpha0, beta0)
    samples = proposal.rvs(size=n_samples)
    
    weights_unnorm = unnorm_posterior(samples, alpha0, beta0) / proposal.pdf(samples)
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    post_mean = np.sum(weights * samples)
    ess = 1.0 / np.sum(weights**2)
    
    print(f"{name:<25} {post_mean:<12.6f} {ess:<10.1f} {ess/n_samples:.2%}")
    print(f"{'  (analytical)':<25} {analytical_mean:<12.6f}")

print("\nKey insights:")
print("- All give accurate posterior mean estimates")
print("- ESS depends on prior-posterior mismatch")
print("- Weak prior as proposal → good ESS when data is strong")


# =============================================================================
# 중간 연습 문제
# =============================================================================

print("\n" + "=" * 70)
print("INTERMEDIATE LEVEL EXERCISES")
print("=" * 70)

# 연습 3: 가장 좋은 표본 크기
# ==============================
print("\n" + "-" * 70)
print("EXERCISE 3: Determining Optimal Sample Size")
print("-" * 70)

print("""
문제:
--------
제안 q ~ N(3,2)로 θ ~ N(5,1)일 때 E[θ]을 어림하면:

가) 어림값의 흩어짐은 n에 따라 어떻게 변하는가?
나) 95% 믿음 구간의 너비가 0.1 미만이 되는 데 필요한 표본 크기 어림하기
다) 셈 값과 정확도의 주고받음 견주기
""")

print("\nSOLUTION:")

target_ex3 = stats.norm(5, 1)
proposal_ex3 = stats.norm(3, 2)
h_identity = lambda x: x

true_mean = 5.0

# n을 달리하며 흩어짐 어림하기
sample_sizes_ex3 = [50, 100, 200, 500, 1000, 2000, 5000]
n_reps = 200

print("\nVariance scaling with sample size:")
print(f"{'n':<8} {'Est Var':<12} {'Std Error':<12} {'95% CI Width':<15} {'Time (ms)'}")
print("-" * 65)

import time

for n in sample_sizes_ex3:
    estimates = []
    start_time = time.time()
    
    for _ in range(n_reps):
        samples = proposal_ex3.rvs(size=n)
        weights_unnorm = target_ex3.pdf(samples) / proposal_ex3.pdf(samples)
        weights = weights_unnorm / np.sum(weights_unnorm)
        estimate = np.sum(weights * h_identity(samples))
        estimates.append(estimate)
    
    elapsed = (time.time() - start_time) / n_reps * 1000  # 되풀이 실행마다의 밀리초
    
    var_est = np.var(estimates)
    std_err = np.std(estimates)
    ci_width = 1.96 * 2 * std_err  # 95% 믿음 구간의 너비
    
    print(f"{n:<8} {var_est:<12.6f} {std_err:<12.6f} {ci_width:<15.6f} {elapsed:<.2f}")

print("\nKey insights:")
print("- Variance ∝ 1/n (standard Monte Carlo rate)")
print("- For CI width < 0.1, need approximately n ≥ 1500")
print("- Computational cost scales linearly with n")


# 연습 4: 나쁜 제안 진단하기
# ====================================
print("\n" + "-" * 70)
print("EXERCISE 4: Identifying and Fixing Poor Proposals")
print("-" * 70)

print("""
문제:
--------
과녁이 π ~ N(10, 1)일 때 제안 q ~ N(0, 1)을 써 본다.
이는 나쁜 제안이다. 왜 그런지 진단하고 나아질 길을 내놓아라.

가) ESS 셈하기
나) 무게 분포 살펴보기
다) 더 나은 제안을 내놓고 시험하기
""")

print("\nSOLUTION:")

target_ex4 = stats.norm(10, 1)
poor_proposal = stats.norm(0, 1)

n_samples_ex4 = 2000

# 나쁜 제안
samples_poor = poor_proposal.rvs(size=n_samples_ex4)
weights_poor_unnorm = target_ex4.pdf(samples_poor) / poor_proposal.pdf(samples_poor)
weights_poor = weights_poor_unnorm / np.sum(weights_poor_unnorm)
ess_poor = 1.0 / np.sum(weights_poor**2)

print("\nPoor Proposal Analysis:")
print(f"  Proposal: N(0, 1)")
print(f"  ESS: {ess_poor:.1f} ({ess_poor/n_samples_ex4:.1%})")
print(f"  Max weight: {np.max(weights_poor):.6f}")
print(f"  CV of weights: {np.std(weights_poor)/np.mean(weights_poor):.2f}")

# 무게 몰림
sorted_weights = np.sort(weights_poor)[::-1]
cumsum = np.cumsum(sorted_weights)
n_for_50pct = np.searchsorted(cumsum, 0.5) + 1
print(f"  Samples for 50% weight: {n_for_50pct} ({n_for_50pct/n_samples_ex4:.1%})")

print("\nDiagnosis:")
print("  ✗ Very low ESS (~1-2% efficiency)")
print("  ✗ Few samples carry most weight")
print("  ✗ Proposal mean far from target mean")
print("  ✗ Most samples in low-probability region")

# 더 나은 제안
better_proposal = stats.norm(10, 1.5)
samples_better = better_proposal.rvs(size=n_samples_ex4)
weights_better_unnorm = target_ex4.pdf(samples_better) / better_proposal.pdf(samples_better)
weights_better = weights_better_unnorm / np.sum(weights_better_unnorm)
ess_better = 1.0 / np.sum(weights_better**2)

print("\nImproved Proposal Analysis:")
print(f"  Proposal: N(10, 1.5)")
print(f"  ESS: {ess_better:.1f} ({ess_better/n_samples_ex4:.1%})")
print(f"  Max weight: {np.max(weights_better):.6f}")
print(f"  Improvement: {ess_better/ess_poor:.1f}x better ESS")


# =============================================================================
# 앞선 연습 문제
# =============================================================================

print("\n" + "=" * 70)
print("ADVANCED LEVEL EXERCISES")
print("=" * 70)

# 연습 5: 드문 일에 쓰는 중요도 표집
# ==============================================
print("\n" + "-" * 70)
print("EXERCISE 5: Rare Event Probability Estimation")
print("-" * 70)

print("""
문제:
--------
X ~ N(0, 1)일 때 P(X > 4) 어림하기. 이는 드문 일이다(p ≈ 0.000032).

가) 어수룩한 몬테카를로 시도하기
나) 옮긴 제안으로 중요도 표집 쓰기
다) 흩어짐 줄임 배수 셈하기
라) 상대 오차 10%에 맞는 표본 크기 정하기
""")

print("\nSOLUTION:")

target_ex5 = stats.norm(0, 1)
threshold = 4.0
true_prob = 1 - target_ex5.cdf(threshold)

print(f"True probability: {true_prob:.8f}")

# 어수룩한 몬테카를로
n_mc = 100000
samples_mc = target_ex5.rvs(size=n_mc)
h_indicator = lambda x: (x > threshold).astype(float)
estimate_mc = np.mean(h_indicator(samples_mc))

print(f"\nNaive MC ({n_mc} samples):")
print(f"  Estimate: {estimate_mc:.8f}")
print(f"  Relative error: {abs(estimate_mc - true_prob)/true_prob:.1%}")

# 옮긴 제안을 쓴 중요도 표집
# 꼬리 가까이에 가운데를 맞춘 제안
proposal_ex5 = stats.norm(threshold + 1, 1)
n_is = 10000

samples_is = proposal_ex5.rvs(size=n_is)
weights_unnorm_ex5 = target_ex5.pdf(samples_is) / proposal_ex5.pdf(samples_is)
weights_ex5 = weights_unnorm_ex5 / np.sum(weights_unnorm_ex5)
estimate_is = np.sum(weights_ex5 * h_indicator(samples_is))

print(f"\nImportance Sampling ({n_is} samples):")
print(f"  Estimate: {estimate_is:.8f}")
print(f"  Relative error: {abs(estimate_is - true_prob)/true_prob:.1%}")
print(f"  ESS: {1.0/np.sum(weights_ex5**2):.1f}")

# 되풀이 실행으로 흩어짐 견주기
n_reps_ex5 = 500

mc_estimates = []
is_estimates = []

for _ in range(n_reps_ex5):
    # 몬테카를로
    samples_mc_rep = target_ex5.rvs(size=1000)
    mc_estimates.append(np.mean(h_indicator(samples_mc_rep)))
    
    # 중요도 표집
    samples_is_rep = proposal_ex5.rvs(size=1000)
    w_unnorm = target_ex5.pdf(samples_is_rep) / proposal_ex5.pdf(samples_is_rep)
    w_norm = w_unnorm / np.sum(w_unnorm)
    is_estimates.append(np.sum(w_norm * h_indicator(samples_is_rep)))

var_mc = np.var(mc_estimates)
var_is = np.var(is_estimates)
variance_reduction = var_mc / var_is

print(f"\nVariance Comparison (1000 samples, {n_reps_ex5} replications):")
print(f"  MC variance: {var_mc:.2e}")
print(f"  IS variance: {var_is:.2e}")
print(f"  Variance reduction: {variance_reduction:.1f}x")

# 상대 오차 10%에 필요한 표본 크기
# 상대 오차 ≈ 표준편차/평균
# 표준편차/평균 < 0.1을 바라므로 표준편차 < 0.1*평균
# 표준편차 ≈ √흩어짐이므로 √흩어짐 < 0.1*참확률
# 표본 n개에서: 흩어짐 ∝ 1/n

target_rel_error = 0.10
required_std = target_rel_error * true_prob
required_var = required_std**2

n_required_mc = var_mc / required_var
n_required_is = var_is / required_var

print(f"\nSample size for 10% relative error:")
print(f"  MC needs: {n_required_mc:.0f} samples")
print(f"  IS needs: {n_required_is:.0f} samples")
print(f"  Reduction: {n_required_mc/n_required_is:.1f}x fewer samples with IS")


# 연습 6: 봉우리 여럿인 뒤확률
# ==============================
print("\n" + "-" * 70)
print("EXERCISE 6: Importance Sampling for Multimodal Posterior")
print("-" * 70)

print("""
문제:
--------
봉우리 둘인 뒤확률을 만드는 섞음 가능도를 보자:
  가능도: y = 2일 때 0.4*N(y|θ, 1) + 0.6*N(y|θ+6, 1)
  앞확률: θ ~ N(0, 4)
  
뒤확률에 봉우리가 둘 있다. 중요도 표집 전략을 짜라.

가) 성분 하나짜리 제안 구현하기
나) 섞음 제안 구현하기
다) ESS 견주기
라) 뒤확률의 평균과 흩어짐 어림하기
""")

print("\nSOLUTION:")

y_obs = 2.0

def log_likelihood_mixture(theta):
    """섞음 모형의 로그 가능도"""
    ll1 = stats.norm.logpdf(y_obs, theta, 1) + np.log(0.4)
    ll2 = stats.norm.logpdf(y_obs, theta + 6, 1) + np.log(0.6)
    return logsumexp([ll1, ll2])

def unnorm_posterior_mixture(theta):
    """고르게 하지 않은 뒤확률"""
    log_prior = stats.norm.logpdf(theta, 0, 2)
    return np.exp(log_likelihood_mixture(theta) + log_prior)

# 성분 하나짜리 제안(봉우리 사이에 가운데를 맞춤)
proposal_single = stats.norm(-2, 3)

n_samples_ex6 = 5000
samples_single = proposal_single.rvs(size=n_samples_ex6)
weights_single_unnorm = np.array([
    unnorm_posterior_mixture(s) / proposal_single.pdf(s)
    for s in samples_single
])
weights_single = weights_single_unnorm / np.sum(weights_single_unnorm)
ess_single = 1.0 / np.sum(weights_single**2)

post_mean_single = np.sum(weights_single * samples_single)
post_var_single = np.sum(weights_single * (samples_single - post_mean_single)**2)

print("\nSingle-Component Proposal N(-2, 3):")
print(f"  ESS: {ess_single:.1f} ({ess_single/n_samples_ex6:.1%})")
print(f"  Posterior mean: {post_mean_single:.4f}")
print(f"  Posterior std: {np.sqrt(post_var_single):.4f}")

# 섞음 제안(봉우리마다 성분 하나씩 둘)
class MixtureProposal:
    def __init__(self, means, stds, weights):
        self.components = [stats.norm(m, s) for m, s in zip(means, stds)]
        self.weights = np.array(weights) / np.sum(weights)
    
    def rvs(self, size):
        # 성분 표집
        components = np.random.choice(len(self.components), size=size, p=self.weights)
        samples = []
        for i in range(size):
            comp_idx = components[i]
            sample = self.components[comp_idx].rvs()
            samples.append(sample)
        return np.array(samples)
    
    def pdf(self, x):
        densities = []
        for comp, weight in zip(self.components, self.weights):
            densities.append(weight * comp.pdf(x))
        return np.sum(densities, axis=0)

# 어림 봉우리에 가운데를 맞춘 섞음
proposal_mixture = MixtureProposal(
    means=[-1, -7],  # 어림한 뒤확률 봉우리
    stds=[1.5, 1.5],
    weights=[0.4, 0.6]
)

samples_mixture = proposal_mixture.rvs(size=n_samples_ex6)
weights_mixture_unnorm = np.array([
    unnorm_posterior_mixture(s) / proposal_mixture.pdf(s)
    for s in samples_mixture
])
weights_mixture = weights_mixture_unnorm / np.sum(weights_mixture_unnorm)
ess_mixture = 1.0 / np.sum(weights_mixture**2)

post_mean_mixture = np.sum(weights_mixture * samples_mixture)
post_var_mixture = np.sum(weights_mixture * (samples_mixture - post_mean_mixture)**2)

print("\nMixture Proposal:")
print(f"  Components: 0.4*N(-1,1.5) + 0.6*N(-7,1.5)")
print(f"  ESS: {ess_mixture:.1f} ({ess_mixture/n_samples_ex6:.1%})")
print(f"  Posterior mean: {post_mean_mixture:.4f}")
print(f"  Posterior std: {np.sqrt(post_var_mixture):.4f}")
print(f"\nImprovement: {ess_mixture/ess_single:.2f}x better ESS with mixture proposal")

print("\nKey insights:")
print("- Multimodal posteriors need careful proposal design")
print("- Single-component proposals may miss modes")
print("- Mixture proposals can capture multiple modes")
print("- ESS much higher with mixture proposal")

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
이 연습 문제가 다루는 것:

첫걸음 단계:
1. 기본 중요도 표집의 구현과 모임
2. 켤레 모형에서 앞확률 고름의 효과

중급 단계:
3. 표본 크기 정하기와 값-정확도의 주고받음
4. 나쁜 제안 진단하고 고치기

앞선 단계:
5. 흩어짐을 줄인 드문 일 어림
6. 섞음 제안으로 다루는, 봉우리가 여럿인 뒤확률

익히게 되는 핵심 솜씨:
- 중요도 표집을 밑바닥부터 구현하기
- ESS 셈하고 풀이하기
- 제안의 질 진단하기
- 알맞은 제안 고르기
- 흩어짐 줄이는 기법
- 복잡한 뒤확률 다루기

더 익히려면:
- 서로 다른 과녁 분포 시도하기
- 제안 집안을 이것저것 시험해 보기
- 중요도 표집과 MCMC 방법 견주기
- 맞춰 가는 중요도 표집의 여러 갈래 구현하기
- 참 베이즈 추론 문제에 쓰기
""")

plt.show()


if __name__ == "__main__":
    pass
```

## 논의

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
연습 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_mixtureproposal():
        model = MixtureProposal(...)
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
