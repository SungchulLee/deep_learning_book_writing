# 단순한 베이즈 보기

03_bayesian_simple_examples.py 첫걸음 수준: 중요도 표집을 쓴 단순한 베이즈 추론 보기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
03_bayesian_simple_examples.py

첫걸음 단계: 중요도 표집을 쓴 단순한 베이즈 추론 보기

이 단원은 켤레 베이즈 모형에 쓰는 중요도 표집을 보인다
여기서는 확인할 수 있는, 손으로 구한 풀이가 있다.

다루는 모형:
1. 베타-이항(베르누이 관측)
2. 정규-정규(흩어짐을 앎)
3. 감마-푸아송(푸아송 관측)

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


def plot_bayesian_update(prior_dist, posterior_dist, data, param_name='θ',
                         title='Bayesian Update'):
    """베이즈 새로 고치기를 그려 보는 도움 함수."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 분포 그리기
    ax = axes[0]
    x = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 1000)
    if hasattr(posterior_dist, 'ppf'):
        x_post = np.linspace(posterior_dist.ppf(0.001), posterior_dist.ppf(0.999), 1000)
        x = np.union1d(x, x_post)
    
    ax.plot(x, prior_dist.pdf(x), 'b--', linewidth=2, label='Prior', alpha=0.7)
    ax.plot(x, posterior_dist.pdf(x), 'r-', linewidth=2, label='Posterior', alpha=0.7)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 자료 그리기
    ax = axes[1]
    if len(data) < 50:
        ax.hist(data, bins=min(len(np.unique(data)), 20), alpha=0.7, 
                edgecolor='black', color='green')
    else:
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Observed Data', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Data (n={len(data)})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# 보기 1: 베타-이항 모형
# ============================
print("=" * 70)
print("EXAMPLE 1: Beta-Binomial Model (Bernoulli Trials)")
print("=" * 70)

print("""
모형:
  가능도: y ~ Bernoulli(θ), n번 시도에 성공 s번 관측
  앞확률: θ ~ Beta(α₀, β₀)
  뒤확률: θ|y ~ Beta(α₀+s, β₀+n-s)

과제: 중요도 표집으로 뒤확률의 평균과 흩어짐 어림하기
""")

# 인공 자료 만들기: 동전 던지기
true_theta = 0.7  # 참 성공 확률
n_trials = 50
data = np.random.binomial(1, true_theta, n_trials)
successes = np.sum(data)
failures = n_trials - successes

print(f"\nData: {successes} successes out of {n_trials} trials")
print(f"Sample proportion: {successes/n_trials:.3f}")

# 앞확률: Beta(2, 2) - 살짝 정보가 있고 가운데가 0.5
alpha_0, beta_0 = 2, 2
prior_dist = stats.beta(alpha_0, beta_0)

print(f"\nPrior: Beta({alpha_0}, {beta_0})")
print(f"  Prior mean: {alpha_0/(alpha_0+beta_0):.3f}")

# 손으로 구한 뒤확률
alpha_n = alpha_0 + successes
beta_n = beta_0 + failures
posterior_dist = stats.beta(alpha_n, beta_n)

print(f"\nPosterior (analytical): Beta({alpha_n}, {beta_n})")
print(f"  Posterior mean: {alpha_n/(alpha_n+beta_n):.6f}")
print(f"  Posterior variance: {posterior_dist.var():.6f}")

# 고르게 하지 않은 뒤확률: γ(θ) = p(y|θ)p(θ)
def unnormalized_posterior_beta(theta):
    """
    γ(θ) = θˢ(1-θ)ⁿ⁻ˢ × θ^(α₀-1)(1-θ)^(β₀-1)
         = θ^(α₀+s-1)(1-θ)^(β₀+n-s-1)
    
    이는 Beta(α₀+s, β₀+n-s)에 비례한다
    """
    # 경계에서의 수치 말썽 막기
    theta = np.clip(theta, 1e-10, 1-1e-10)
    
    # 수치 안정을 위한 로그 공간
    log_likelihood = successes * np.log(theta) + failures * np.log(1 - theta)
    log_prior = (alpha_0-1) * np.log(theta) + (beta_0-1) * np.log(1 - theta)
    
    return np.exp(log_likelihood + log_prior)

# 앞확률을 제안으로 쓴 중요도 표집
n_samples = 5000

# 앞확률에서 표집
samples = prior_dist.rvs(size=n_samples)

# 고르게 하지 않은 무게 셈하기
gamma_values = unnormalized_posterior_beta(samples)
q_values = prior_dist.pdf(samples)
unnorm_weights = gamma_values / q_values

# 무게 고르게 하기
weights = unnorm_weights / np.sum(unnorm_weights)

# 뒤확률 평균 어림하기
h_mean = lambda theta: theta
posterior_mean_is = np.sum(weights * h_mean(samples))

# 뒤확률 흩어짐 어림하기
h_var = lambda theta: (theta - posterior_mean_is)**2
posterior_var_is = np.sum(weights * h_var(samples))

# ESS
ess = 1.0 / np.sum(weights**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess:.1f}):")
print(f"  Posterior mean: {posterior_mean_is:.6f} (error: {abs(posterior_mean_is - posterior_dist.mean()):.6f})")
print(f"  Posterior variance: {posterior_var_is:.6f} (error: {abs(posterior_var_is - posterior_dist.var()):.6f})")

# 시각화한다
fig = plot_bayesian_update(prior_dist, posterior_dist, data, 'θ', 
                           'Beta-Binomial: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example1_beta_binomial.png', 
            dpi=300, bbox_inches='tight')


# 보기 2: 정규-정규 모형
# ============================
print("\n" + "=" * 70)
print("EXAMPLE 2: Normal-Normal Model (Known Variance)")
print("=" * 70)

print("""
모형:
  가능도: y ~ N(θ, σ²), σ²을 앎
  Prior: θ ~ N(μ₀, τ₀²)
  뒤확률: θ|y ~ N(μₙ, τₙ²), 여기서
    τₙ² = 1/(1/τ₀² + n/σ²)
    μₙ = τₙ²(μ₀/τ₀² + Σyᵢ/σ²)

과제: 중요도 표집으로 뒤확률 분포 어림하기
""")

# 데이터를 생성한다
true_theta = 8.0
sigma = 2.0  # 알려진 관측 잡음
n_obs = 30
data_normal = np.random.normal(true_theta, sigma, n_obs)

print(f"\nData: n={n_obs}, sample mean={np.mean(data_normal):.3f}, σ={sigma}")

# 앞확률
mu_0 = 5.0
tau_0 = 3.0
prior_normal = stats.norm(mu_0, tau_0)

print(f"\nPrior: N({mu_0}, {tau_0}²)")

# 손으로 구한 뒤확률
precision_0 = 1.0 / tau_0**2
precision_n = precision_0 + n_obs / sigma**2
tau_n = 1.0 / np.sqrt(precision_n)
mu_n = (precision_0 * mu_0 + np.sum(data_normal) / sigma**2) / precision_n
posterior_normal = stats.norm(mu_n, tau_n)

print(f"\nPosterior (analytical): N({mu_n:.6f}, {tau_n:.6f}²)")

# 고르게 하지 않은 뒤확률
def unnormalized_posterior_normal(theta):
    """
    γ(θ) = ∏ᵢ exp(-(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ₀²)
    """
    log_likelihood = -0.5 * np.sum((data_normal[:, None] - theta)**2) / sigma**2
    log_prior = -0.5 * (theta - mu_0)**2 / tau_0**2
    return np.exp(log_likelihood + log_prior)

# 앞확률을 제안으로 쓴 중요도 표집
n_samples = 5000
samples_normal = prior_normal.rvs(size=n_samples)

# 고르게 하지 않은 무게
gamma_values_normal = unnormalized_posterior_normal(samples_normal)
q_values_normal = prior_normal.pdf(samples_normal)
unnorm_weights_normal = gamma_values_normal / q_values_normal
weights_normal = unnorm_weights_normal / np.sum(unnorm_weights_normal)

# 어림값
posterior_mean_normal_is = np.sum(weights_normal * samples_normal)
posterior_var_normal_is = np.sum(weights_normal * (samples_normal - posterior_mean_normal_is)**2)
ess_normal = 1.0 / np.sum(weights_normal**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_normal:.1f}):")
print(f"  Posterior mean: {posterior_mean_normal_is:.6f} (true: {mu_n:.6f})")
print(f"  Posterior std: {np.sqrt(posterior_var_normal_is):.6f} (true: {tau_n:.6f})")

# 믿음 구간: 95%
sorted_samples = samples_normal[np.argsort(weights_normal)[::-1]]
sorted_weights = np.sort(weights_normal)[::-1]
cumsum_weights = np.cumsum(sorted_weights)
n_95 = np.searchsorted(cumsum_weights, 0.95) + 1
credible_95_is = np.percentile(sorted_samples[:n_95], [2.5, 97.5])

# 참 믿음 구간
credible_95_true = posterior_normal.ppf([0.025, 0.975])

print(f"\n95% Credible Interval:")
print(f"  IS: [{credible_95_is[0]:.3f}, {credible_95_is[1]:.3f}]")
print(f"  True: [{credible_95_true[0]:.3f}, {credible_95_true[1]:.3f}]")

# 시각화한다
fig = plot_bayesian_update(prior_normal, posterior_normal, data_normal, 'θ',
                           'Normal-Normal: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example2_normal_normal.png',
            dpi=300, bbox_inches='tight')


# 보기 3: 감마-푸아송 모형
# ============================
print("\n" + "=" * 70)
print("EXAMPLE 3: Gamma-Poisson Model")
print("=" * 70)

print("""
모형:
  가능도: y ~ Poisson(λ), 관측한 셈 수
  앞확률: λ ~ Gamma(α₀, β₀)
  뒤확률: λ|y ~ Gamma(α₀+Σyᵢ, β₀+n)

과제: 중요도 표집으로 뒤확률 어림하기
""")

# 계수 데이터를 생성한다
true_lambda = 4.5
n_counts = 40
data_poisson = np.random.poisson(true_lambda, n_counts)
sum_counts = np.sum(data_poisson)

print(f"\nData: n={n_counts}, Σyᵢ={sum_counts}, sample mean={np.mean(data_poisson):.3f}")

# 앞확률: Gamma(2, 0.5)
# 평균 = α/β = 2/0.5 = 4
alpha_0_gamma = 2.0
beta_0_gamma = 0.5
prior_gamma = stats.gamma(alpha_0_gamma, scale=1.0/beta_0_gamma)

print(f"\nPrior: Gamma({alpha_0_gamma}, {beta_0_gamma})")
print(f"  Prior mean: {alpha_0_gamma/beta_0_gamma:.3f}")

# 손으로 구한 뒤확률
alpha_n_gamma = alpha_0_gamma + sum_counts
beta_n_gamma = beta_0_gamma + n_counts
posterior_gamma = stats.gamma(alpha_n_gamma, scale=1.0/beta_n_gamma)

print(f"\nPosterior (analytical): Gamma({alpha_n_gamma}, {beta_n_gamma})")
print(f"  Posterior mean: {alpha_n_gamma/beta_n_gamma:.6f}")

# 고르게 하지 않은 뒤확률
def unnormalized_posterior_gamma(lam):
    """
    γ(λ) = ∏ᵢ λ^yᵢ exp(-λ) × λ^(α₀-1) exp(-β₀λ)
         = λ^(α₀+Σyᵢ-1) exp(-(β₀+n)λ)
    
    이는 Gamma(α₀+Σyᵢ, β₀+n)에 비례한다
    """
    # 안정을 위한 로그 공간
    log_likelihood = sum_counts * np.log(lam + 1e-10) - n_counts * lam
    log_prior = (alpha_0_gamma-1) * np.log(lam + 1e-10) - beta_0_gamma * lam
    return np.exp(log_likelihood + log_prior)

# 중요도 표집
n_samples = 5000
samples_gamma = prior_gamma.rvs(size=n_samples)

# 무게
gamma_values_poisson = unnormalized_posterior_gamma(samples_gamma)
q_values_gamma = prior_gamma.pdf(samples_gamma)
unnorm_weights_gamma = gamma_values_poisson / q_values_gamma
weights_gamma = unnorm_weights_gamma / np.sum(unnorm_weights_gamma)

# 어림값
posterior_mean_gamma_is = np.sum(weights_gamma * samples_gamma)
posterior_var_gamma_is = np.sum(weights_gamma * (samples_gamma - posterior_mean_gamma_is)**2)
ess_gamma = 1.0 / np.sum(weights_gamma**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_gamma:.1f}):")
print(f"  Posterior mean: {posterior_mean_gamma_is:.6f} (true: {alpha_n_gamma/beta_n_gamma:.6f})")
print(f"  Posterior variance: {posterior_var_gamma_is:.6f} (true: {posterior_gamma.var():.6f})")

# 뒤확률 미리봄: P(ỹ|y)
# 새 관측 ỹ에 대해
print("\nPosterior Predictive Distribution for new observation:")

# 감마-푸아송의 참 뒤확률 미리봄은 음이항이다
# ỹ|y ~ NB(α_n, β_n/(β_n+1))
post_pred_true = stats.nbinom(alpha_n_gamma, beta_n_gamma/(beta_n_gamma+1))

# 중요도 표집으로 어림하기
# E[P(ỹ|λ)|y] = E[Poisson(ỹ|λ)|y]
y_new_values = np.arange(0, 15)
post_pred_is = []

for y_new in y_new_values:
    # h(λ) = P(ỹ=y_new|λ) = λ^y_new exp(-λ) / y_new!
    h_poisson = lambda lam: stats.poisson.pmf(y_new, lam)
    prob_is = np.sum(weights_gamma * h_poisson(samples_gamma))
    post_pred_is.append(prob_is)

post_pred_true_probs = post_pred_true.pmf(y_new_values)

print(f"\nPosterior Predictive P(ỹ|y):")
print("y_new  True     IS")
print("-" * 25)
for y, p_true, p_is in zip(y_new_values[:8], post_pred_true_probs[:8], post_pred_is[:8]):
    print(f"{y:3d}   {p_true:.4f}  {p_is:.4f}")

# 시각화한다
fig = plot_bayesian_update(prior_gamma, posterior_gamma, data_poisson, 'λ',
                           'Gamma-Poisson: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example3_gamma_poisson.png',
            dpi=300, bbox_inches='tight')


# 견주어 살피기
# ===================
print("\n" + "=" * 70)
print("COMPARATIVE ANALYSIS: ESS Across Models")
print("=" * 70)

models = ['Beta-Binomial', 'Normal-Normal', 'Gamma-Poisson']
ess_values = [ess, ess_normal, ess_gamma]
efficiencies = [e/n_samples*100 for e in ess_values]

print("\nEffective Sample Size Summary:")
print("-" * 50)
for model, ess_val, eff in zip(models, ess_values, efficiencies):
    print(f"{model:20s}: ESS = {ess_val:6.1f} ({eff:5.1f}%)")

print("""
\n관측:
1. 앞확률을 제안으로 쓰는 것은 단순하지만 늘 효율적이지는 않다
2. ESS은 자료가 앞확률을 얼마나 바꾸는지에 달렸다
3. 자료가 세면(n이 크거나 관측이 극단이면) → ESS이 낮다
4. 앞확률과 뒤확률의 어긋남이 커질수록 ESS이 줄어든다
5. 가능도의 정보가 많을수록 → 앞확률보다 나은 제안이 필요하다
""")

# 마지막 그림: ESS 견주기
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(models, efficiencies, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=2)
ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=13)
ax.set_title('Importance Sampling Efficiency: Prior as Proposal', 
             fontsize=14, fontweight='bold')
ax.axhline(100, color='red', linestyle='--', linewidth=2, 
           label='Perfect efficiency', alpha=0.5)
ax.set_ylim([0, max(efficiencies)*1.2])
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

# 막대에 값 이름표를 추가한다
for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/comparative_ess.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 켤레 모형은 중요도 표집을 손으로 확인할 길을 준다:
   - 두 값 자료에는 베타-이항
   - 이어진 자료에는 정규-정규(흩어짐을 알 때)
   - 세는 자료에는 감마-푸아송

2. 앞확률을 제안으로 쓰는 것은 단순하지만 한계가 있다:
   - 자료가 약할 때(n이 작을 때) 잘 듣는다
   - 자료가 세면 효율이 낮아진다
   - 가능도가 앞확률을 누를수록 ESS이 줄어든다

3. 중요도 표집의 좋은 점:
   - 태우기 기간이 없다(MCMC과 달리)
   - 표본이 독립이다
   - 같은 표본에서 여러 양을 어림할 수 있다
   - 뒤확률 미리봄을 쉽게 셈할 수 있다

4. 진단 잣대로서의 ESS:
   - 독립 표본의 실효 개수를 잰다
   - ESS << n이면 더 나은 제안이 필요하다는 뜻이다
   - 제안을 객관적으로 견줄 수 있다

5. 실전에서 살필 점:
   - 표집한 뒤 ESS을 늘 살펴라
   - ESS이 낮다 → 몇몇 표본이 판친다
   - ESS이 낮으면 맞춰 가거나 차례차례 하는 방법 생각해 보기
   - 복잡한 뒤확률에는 더 똑똑한 제안이 필요하다

6. MCMC과의 이음:
   - 중요도 표집은 MCMC을 보완한다
   - 중요도 표집을 쓸 때: 표본이 독립이고 좋은 제안을 알 때
   - MCMC을 쓸 때: 뒤확률이 복잡하고 좋은 제안을 모를 때
""")


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
단순한 베이즈 보기 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_bayesian simple examples():
        model = Bayesian Simple Examples(...)
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
