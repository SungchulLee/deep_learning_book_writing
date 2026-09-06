# 잇단 중요도 표집

09_sequential_importance_sampling.py 나아간 수준: 잇단 중요도 표집(SIS)

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
09_sequential_importance_sampling.py

앞선 단계: 차례차례 중요도 표집(SIS)

이 단원은 차례차례 중요도 표집을 구현한다.
자료를 차례로 다루고 차례차례 베이즈 추론을 하는 데 쓴다.

수학적 바탕:
---------------------
차례차례 중요도 표집은 새 자료가 올 때마다 중요도 무게를 새로 고친다:

시간 t에 자료 y₁:t = (y₁, ..., y_t)이 있다

뒤확률: p(θ|y₁:t) ∝ p(y₁:t|θ)p(θ)

차례차례 인수 나누기:
    p(θ|y₁:t) ∝ p(yt|θ, y₁:t₋₁) × p(θ|y₁:t₋₁)

중요도 무게 새로 고치기:
    w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ, y₁:t₋₁) / q_t(θ|y₁:t, θ₁:t₋₁)

붙박인 제안 q(θ)에서(맞춰 가지 않을 때):
    w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ)

무게가 시간에 따라 곱해져 다음에 이른다:

무게 찌부러짐 문제:
- 무게가 점점 더 고르지 않게 된다
- 시간이 갈수록 ESS이 줄어든다
- 끝내 몇몇 알갱이가 판친다

풀이:
1. 다시 표집(알갱이 거르개)
2. 맞춰 가는 제안
3. 지킴 섞기

쓰임새:
- 차례차례 베이즈 새로 고치기
- 시계열 분석
- 흐름 속 배움
- 알갱이 거르개

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
    다시 표집을 붙일 수 있는 차례차례 중요도 표집.
    """
    
    def __init__(self, prior_dist, proposal_dist):
        """
        매개변수:
        -----------
        prior_dist : scipy.stats distribution
            앞확률 분포 p(θ)
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
        제안에서 알갱이 첫값 잡기.
        """
        self.n_particles = n_particles
        self.particles = self.proposal_dist.rvs(size=n_particles)
        
        # 첫 무게: w₀(θ) = p(θ)/q(θ)
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
        새 관측으로 무게 새로 고치기.
        
        매개변수:
        -----------
        likelihood_fn : 호출 가능 객체
            함수: likelihood_fn(theta, y) -> 가능도 값
        y_new : 관측
            새 자료 점
        """
        self.t += 1
        
        # 로그 무게 새로 고치기: log w_t = log w_{t-1} + log p(y_t|θ)
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
        """안정을 위해 log-sum-exp으로 무게 고르게 하기."""
        self.log_weights = self.log_weights - logsumexp(self.log_weights)
        self.weights = np.exp(self.log_weights)
    
    def compute_ess(self):
        """실효 표본 크기 셈하기."""
        return 1.0 / np.sum(self.weights**2)
    
    def resample(self, threshold=0.5):
        """
        ESS이 문턱값 아래로 떨어지면 알갱이 다시 표집하기.
        
        다항 다시 표집: 알갱이를 되돌려 놓으며 뽑는다
        무게에 따라.
        """
        ess = self.compute_ess()
        rel_ess = ess / self.n_particles
        
        if rel_ess < threshold:
            # 다항 다시 표집
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                replace=True,
                p=self.weights
            )
            
            self.particles = self.particles[indices]
            
            # 무게를 고르게 되돌리기
            self.log_weights = np.zeros(self.n_particles)
            self.weights = np.ones(self.n_particles) / self.n_particles
            
            return True  # 다시 표집함
        
        return False  # 다시 표집 없음
    
    def estimate(self, h_function):
        """
        E[h(θ)|y₁:t]의 무게 어림값 셈하기.
        """
        return np.sum(self.weights * h_function(self.particles))
    
    def credible_interval(self, alpha=0.95):
        """
        무게 분위수로 믿음 구간 셈하기.
        """
        sorted_indices = np.argsort(self.particles)
        sorted_particles = self.particles[sorted_indices]
        sorted_weights = self.weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        
        lower_idx = np.searchsorted(cumsum, (1-alpha)/2)
        upper_idx = np.searchsorted(cumsum, (1+alpha)/2)
        
        return sorted_particles[lower_idx], sorted_particles[upper_idx]


# 보기 1: 차례차례 베이즈 새로 고치기(정규 평균)
# ===================================================
print("=" * 70)
print("EXAMPLE 1: Sequential Bayesian Updating")
print("=" * 70)

print("""
모형: σ² = 1을 아는 yᵢ ~ N(θ, σ²)
앞확률: θ ~ N(0, 4)
자료를 차례로 다루며 ESS이 나빠지는 것 지켜보기.
""")

# 참 매개변수
theta_true = 2.5
sigma = 1.0

# 차례 있는 자료 만들기
n_obs = 50
data_seq = np.random.normal(theta_true, sigma, n_obs)

print(f"\nTrue θ = {theta_true}")
print(f"Generated {n_obs} observations sequentially")

# 앞확률과 제안
prior = stats.norm(0, 2)
proposal = stats.norm(0, 2)  # 앞확률을 제안으로 씀

# 가능도 함수
def likelihood_normal(theta, y):
    """관측 하나의 로그 가능도."""
    return stats.norm.logpdf(y, theta, sigma)

# SIS 첫값 잡기
n_particles = 2000
sis = SequentialImportanceSampler(prior, proposal)
sis.initialize(n_particles)

print(f"\nInitialized {n_particles} particles")
print(f"Initial ESS: {sis.ess_history[0]:.1f}")

# 견주기 위한, 손으로 구한 뒤확률
def analytical_posterior(y_data, sigma, mu_0, tau_0):
    """흩어짐을 알 때 손으로 구한 뒤확률 N(μₙ, τₙ²)."""
    n = len(y_data)
    tau_n_sq = 1.0 / (1.0/tau_0**2 + n/sigma**2)
    mu_n = tau_n_sq * (mu_0/tau_0**2 + np.sum(y_data)/sigma**2)
    return mu_n, np.sqrt(tau_n_sq)

# 자료를 차례로 다루기
print("\nSequential Processing:")
print(f"{'t':>3} {'y_t':>8} {'Post Mean':>10} {'ESS':>8} {'Rel ESS':>8}")
print("-" * 50)

posterior_means = []
posterior_stds = []

for t, y_t in enumerate(data_seq):
    # 새 관측으로 새로 고치기
    ess = sis.update(likelihood_normal, y_t)
    
    # 뒤확률 평균 어림하기
    post_mean = sis.estimate(lambda x: x)
    posterior_means.append(post_mean)
    
    # 손으로 구한 뒤확률
    mu_n, tau_n = analytical_posterior(data_seq[:t+1], sigma, 0, 2)
    posterior_stds.append(tau_n)
    
    # 관측 5개마다 찍기
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

# ESS 나빠짐 그려 보기
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

# 칸 2: 뒤확률 평균의 흘러감
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

# 칸 3: 때에 따른 무게 분포
ax = axes[1, 0]
times_to_plot = [0, 10, 25, 49]
colors_weights = ['blue', 'green', 'orange', 'red']

# 정해진 때의 무게를 얻으려 다시 돌리기
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

# 칸 4: 지금의 알갱이 분포
ax = axes[1, 1]
x_range = np.linspace(-2, 6, 1000)

# 알갱이 그리기
ax.hist(sis.particles, bins=50, weights=sis.weights, density=True,
        alpha=0.6, color='steelblue', edgecolor='black',
        label='Weighted particles')

# 참 뒤확률
true_post_dist = stats.norm(final_analytical_mean, final_analytical_std)
ax.plot(x_range, true_post_dist.pdf(x_range), 'r-', linewidth=2,
        label='True posterior')

# 앞확률
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
plt.savefig('/home/claude/03_Importance_Sampling/sis_weight_degeneracy.png',
            dpi=300, bbox_inches='tight')


# 보기 2: 다시 표집을 붙인 SIS(알갱이 거르개)
# ==============================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Sequential IS with Resampling")
print("=" * 70)

print("\nResampling when ESS drops below 50% threshold")

# 새 SIS 첫값 잡기
sis_resample = SequentialImportanceSampler(prior, proposal)
sis_resample.initialize(n_particles)

resampling_times = []

print(f"\n{'t':>3} {'ESS':>8} {'Rel ESS':>8} {'Resampled?'}")
print("-" * 40)

for t, y_t in enumerate(data_seq):
    sis_resample.update(likelihood_normal, y_t)
    ess = sis_resample.ess_history[-1]
    
    # 필요하면 다시 표집하기
    did_resample = sis_resample.resample(threshold=0.5)
    
    if did_resample:
        resampling_times.append(t+1)
    
    if (t+1) % 5 == 0 or did_resample:
        resample_str = "YES" if did_resample else ""
        print(f"{t+1:3d} {ess:8.1f} {ess/n_particles:8.1%} {resample_str}")

print(f"\nResampled {len(resampling_times)} times at t = {resampling_times}")

# 다시 표집할 때와 아닐 때의 ESS 견주기
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

# 무게 분포 견주기
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
plt.savefig('/home/claude/03_Importance_Sampling/sis_with_resampling.png',
            dpi=300, bbox_inches='tight')


# 보기 3: 흐름 속 매개변수 어림
# ====================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Online Credible Intervals")
print("=" * 70)

# 시간에 따른 믿음 구간 기록하기
sis_ci = SequentialImportanceSampler(prior, proposal)
sis_ci.initialize(n_particles)

credible_intervals = []
posterior_means_ci = []

for y_t in data_seq:
    sis_ci.update(likelihood_normal, y_t)
    sis_ci.resample(threshold=0.5)  # 다시 표집을 붙임
    
    mean = sis_ci.estimate(lambda x: x)
    ci_lower, ci_upper = sis_ci.credible_interval(alpha=0.95)
    
    posterior_means_ci.append(mean)
    credible_intervals.append((ci_lower, ci_upper))

# 시간에 따른 믿음 구간 그리기
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
plt.savefig('/home/claude/03_Importance_Sampling/sis_credible_intervals.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 차례차례 중요도 표집은 자료를 조금씩 다룬다:
   - 무게가 시간에 따라 곱해진다: w_t = w_{t-1} × p(y_t|θ)
   - 자료를 모두 다시 다룰 필요가 없다
   - 흐름 속 쓰임새에 쓸모 있다

2. 무게 찌부러짐이 근본 문제이다:
   - 시간이 갈수록 ESS이 줄어든다
   - 끝내 몇몇 알갱이가 판친다
   - 알갱이가 붙박이면 피할 수 없다

3. 찌부러짐 빠르기:
   - 많은 경우 지수꼴이다
   - 정보가 많은 자료에서 더 빠르다
   - 좋은 제안이 있을 때는 더 느리다

4. 다시 표집은 찌부러짐에 맞선다:
   - 무게 큰 알갱이 복제하기
   - 무게 작은 알갱이 버리기
   - 무게를 고르게 되돌리기
   - 알갱이의 다양함을 잃게 한다

5. 다시 표집 전략:
   - 문턱값 방식: ESS < 문턱값이면 다시 표집
   - 주기적: k걸음마다 다시 표집
   - 맞춰 감: ESS이나 다른 잣대에 바탕

6. 다시 표집의 주고받음:
   - 이로움: 실효 표본 크기를 지킨다
   - 값: 알갱이의 다양함을 잃는다(표본 메마름)
   - 값: 표집의 흔들림이 더 생긴다

7. 알갱이 거르개 = SIS + 다시 표집:
   - 시계열에서 널리 쓰인다
   - 상태 공간 모형
   - 뒤쫓기 쓰임새
   - 로봇 공학과 길 찾기

8. 차례차례 중요도 표집을 언제 쓰나:
   - 자료가 차례로 온다
   - 흐름 속 배움의 상황
   - 셈의 제약(자료를 모두 다시 다룰 수 없다)
   - 시계열 분석

9. 실전에서 살필 점:
   - ESS을 끊임없이 지켜보기
   - ESS < 0.5n(또는 0.3n)이면 다시 표집하기
   - 될 수 있으면 맞춰 가는 제안 쓰기
   - 아주 긴 늘어놓음에는 알갱이 MCMC 생각해 보기

10. 한계:
    - 다시 표집에 따른 표본 메마름
    - 붙박인 알갱이는 새 구역을 살펴볼 수 없다
    - 묶음 방법만큼 튼튼하지는 않다
    - 다시 표집 문턱값을 꼼꼼히 맞춰야 한다

11. 다른 길과 개선:
    - 맞춰 가는 제안(붙박인 것보다 낫다)
    - 도움 알갱이 거르개
    - 알갱이 MCMC(되살리기)
    - 라오-블랙웰화(할 수 있을 때)
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
잇단 중요도 표집 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_sequentialimportancesampler():
        model = SequentialImportanceSampler(...)
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
