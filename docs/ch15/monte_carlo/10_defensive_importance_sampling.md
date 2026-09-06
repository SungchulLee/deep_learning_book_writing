# 방어적 중요도 표집

10_defensive_importance_sampling.py 나아간 수준: 방어적 중요도 표집

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

```python
"""
10_defensive_importance_sampling.py

앞선 단계: 지킴 중요도 표집

이 단원은 지킴 중요도 표집을 구현한다. 이는 튼튼한 길로,
겨냥 제안에 지킴 성분을 섞어 흩어짐이 묶임을 보장한다.
지킴 성분을 섞는다.

수학적 바탕:
---------------------
표준 중요도 표집은 제안의 꼬리가 과녁보다 얇거나 중요한 구역을 놓치면
묶이지 않거나 무한한 흩어짐을 가질 수 있다.

지킴 섞음:
    q_def(θ) = α q(θ) + (1-α) m(θ)

여기서 각 기호는 다음과 같다.
- q(θ): 겨냥 제안(확률 높은 구역에 모임)
- m(θ): 지킴 성분(넓고 안전한 덮음)
- α ∈ (0,1): 섞음 매개변수

보통 m(θ)은 다음처럼 고른다:
- 앞확률 p(θ)
- 받침 위에서 고름
- 아주 넓은 가우스나 스튜던트 t

이론의 보장:
어떤 c > 0에 대해 m(θ) ≥ c·π(θ)이면(곧 m의 꼬리가 더 두꺼우면),
그러면 Var[h(θ)w_def(θ)]이 묶인다.

주고받음:
- α이 높으면 ESS은 낫지만 덜 튼튼하다
- α이 낮으면 ESS은 나쁘지만 더 튼튼하다
- 흔한 고름: α ∈ [0.7, 0.9]

핵심 강점: 흩어짐이 묶임을 보장한다

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


class DefensiveMixture:
    """
    튼튼한 중요도 표집을 위한 지킴 섞음 제안.
    """
    
    def __init__(self, targeted_dist, defensive_dist, alpha):
        """
        매개변수:
        -----------
        targeted_dist : scipy.stats distribution
            주된 확률 덩어리를 겨냥한 제안
        defensive_dist : scipy.stats distribution
            튼튼한 덮음을 위한 넓은 제안
        alpha : float in (0,1)
            겨냥 성분의 무게
        """
        self.targeted = targeted_dist
        self.defensive = defensive_dist
        self.alpha = alpha
        
    def rvs(self, size=1):
        """섞음에서 표집하기."""
        samples = []
        for _ in range(size):
            if np.random.rand() < self.alpha:
                samples.append(self.targeted.rvs())
            else:
                samples.append(self.defensive.rvs())
        return np.array(samples)
    
    def pdf(self, x):
        """섞음의 밀도 값 매기기."""
        return (self.alpha * self.targeted.pdf(x) +
                (1 - self.alpha) * self.defensive.pdf(x))
    
    def logpdf(self, x):
        """섞음의 로그 밀도 값 매기기."""
        log_targeted = np.log(self.alpha) + self.targeted.logpdf(x)
        log_defensive = np.log(1 - self.alpha) + self.defensive.logpdf(x)
        return logsumexp([log_targeted, log_defensive], axis=0)


def compare_proposals(target_density, proposals_dict, h_function,
                     n_samples=5000, n_replications=100):
    """
    되풀이 시도로 여러 제안 견주기.
    
    통계량을 돌려준다: 평균 ESS, ESS 표준편차, 평균 어림값, 어림값 표준편차
    """
    results = {}
    
    for name, proposal in proposals_dict.items():
        ess_list = []
        estimates = []
        
        for _ in range(n_replications):
            # 뽑기
            samples = proposal.rvs(size=n_samples)
            
            # 무게
            weights_unnorm = target_density(samples) / proposal.pdf(samples)
            weights = weights_unnorm / np.sum(weights_unnorm)
            
            # ESS
            ess = 1.0 / np.sum(weights**2)
            ess_list.append(ess)
            
            # 어림하기
            estimate = np.sum(weights * h_function(samples))
            estimates.append(estimate)
        
        results[name] = {
            'mean_ess': np.mean(ess_list),
            'std_ess': np.std(ess_list),
            'min_ess': np.min(ess_list),
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
        }
    
    return results


# 보기 1: 꼬리 두꺼운 과녁
# ============================
print("=" * 70)
print("EXAMPLE 1: Heavy-Tailed Target Distribution")
print("=" * 70)

print("""
과녁: 스튜던트 t(자유도=3) - 두꺼운 꼬리
제안:
1. 가우스(위험함 - 꼬리가 얇다)
2. 스튜던트 t(겨냥하지만 여전히 위험하다)
3. 지킴 섞음: α × t(3) + (1-α) × t(1)(코시)
""")

# 과녁: 자유도 3인 스튜던트 t
target_t3 = stats.t(df=3, loc=0, scale=1)

# 어림할 함수: E[θ²]
h_square = lambda x: x**2

# 참값
x_grid = np.linspace(target_t3.ppf(0.001), target_t3.ppf(0.999), 10000)
true_value = np.trapz(x_grid**2 * target_t3.pdf(x_grid), x_grid)

print(f"\nTrue E[θ²] = {true_value:.6f}")

# 제안들
proposals_ex1 = {
    'Gaussian (risky)': stats.norm(0, 1.5),
    'Student-t(3) (risky)': stats.t(df=3, loc=0, scale=1.2),
    'Defensive α=0.8': DefensiveMixture(
        stats.t(df=3, loc=0, scale=1.2),  # 겨냥함
        stats.t(df=1, loc=0, scale=2),     # 코시(꼬리가 아주 두꺼움)
        alpha=0.8
    ),
    'Defensive α=0.9': DefensiveMixture(
        stats.t(df=3, loc=0, scale=1.2),
        stats.t(df=1, loc=0, scale=2),
        alpha=0.9
    ),
}

# 제안 견주기
print("\nComparing proposals (100 replications, 3000 samples each):")
results_ex1 = compare_proposals(target_t3.pdf, proposals_ex1, h_square,
                                 n_samples=3000, n_replications=100)

print(f"\n{'Proposal':<22} {'Mean ESS':>10} {'Min ESS':>10} {'Std Est':>10} {'Robust?'}")
print("-" * 70)

for name, stats_dict in results_ex1.items():
    # 최소 ESS이 그럴듯한지 살피기(n의 1% 초과)
    robust = "✓" if stats_dict['min_ess'] > 30 else "✗"
    
    print(f"{name:<22} {stats_dict['mean_ess']:10.1f} "
          f"{stats_dict['min_ess']:10.1f} {stats_dict['std_estimate']:10.4f} {robust}")

print("\nKey insight: Defensive proposals have higher minimum ESS!")


# 보기 2: 잘못 잡은 제안
# ==============================
print("\n" + "=" * 70)
print("EXAMPLE 2: Robustness to Proposal Misspecification")
print("=" * 70)

print("""
상황: 과녁이 N(0,1)인 줄 알았는데 사실은 N(3,1)이다
          (제안을 잘못 잡음 - 자리가 틀림)

지킴 섞기가 우리를 완전한 무너짐에서 구해 준다.
""")

# 참 과녁(우리는 모름)
target_true = stats.norm(3, 1)

# 과녁에 대한 (틀린) 우리의 믿음
target_belief = stats.norm(0, 1)

# 틀린 믿음에 바탕을 둔 제안
proposals_ex2 = {
    'Wrong belief N(0,1)': stats.norm(0, 1.2),
    'Defensive α=0.7': DefensiveMixture(
        stats.norm(0, 1.2),      # 틀린 믿음에 바탕
        stats.norm(0, 5),         # 아주 넓은 안전망
        alpha=0.7
    ),
    'Defensive α=0.9': DefensiveMixture(
        stats.norm(0, 1.2),
        stats.norm(0, 5),
        alpha=0.9
    ),
}

# 참 평균
true_mean = 3.0

print(f"\nTrue target: N(3, 1)")
print(f"Our belief: N(0, 1) [WRONG!]")
print(f"True mean: {true_mean:.1f}")

# 비교
results_ex2 = compare_proposals(target_true.pdf, proposals_ex2,
                                lambda x: x, n_samples=2000,
                                n_replications=200)

print(f"\n{'Proposal':<22} {'Mean Est':>10} {'Bias':>10} {'RMSE':>10} {'Min ESS':>10}")
print("-" * 70)

for name, stats_dict in results_ex2.items():
    bias = stats_dict['mean_estimate'] - true_mean
    rmse = np.sqrt(bias**2 + stats_dict['std_estimate']**2)
    
    print(f"{name:<22} {stats_dict['mean_estimate']:10.4f} "
          f"{bias:+10.4f} {rmse:10.4f} {stats_dict['min_ess']:10.1f}")

print("\nKey insight: Defensive proposals are more robust to misspecification!")


# 보기 3: 알파 바꿔 보기
# ======================
print("\n" + "=" * 70)
print("EXAMPLE 3: Effect of Mixture Parameter α")
print("=" * 70)

# 과녁: 봉우리 둘
def bimodal(x):
    return 0.4 * stats.norm.pdf(x, -2, 0.8) + 0.6 * stats.norm.pdf(x, 3, 1)

# 겨냥 제안(봉우리 하나만 덮음 - 일부러 나쁘게)
targeted_poor = stats.norm(3, 1.2)

# 지킴 성분(봉우리 둘을 다 덮음)
defensive_broad = stats.norm(0, 4)

# 서로 다른 α 값 시도하기
alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

print(f"\nVarying α (mixture weight on targeted component):")
print(f"{'α':>6} {'Mean ESS':>10} {'Min ESS':>10} {'Std ESS':>10} {'Efficiency':>12}")
print("-" * 60)

ess_by_alpha = []
for alpha in alphas:
    proposal = DefensiveMixture(targeted_poor, defensive_broad, alpha)
    
    # 여러 번 시도
    ess_trials = []
    for _ in range(50):
        samples = proposal.rvs(size=2000)
        weights_unnorm = bimodal(samples) / proposal.pdf(samples)
        weights = weights_unnorm / np.sum(weights_unnorm)
        ess = 1.0 / np.sum(weights**2)
        ess_trials.append(ess)
    
    mean_ess = np.mean(ess_trials)
    min_ess = np.min(ess_trials)
    std_ess = np.std(ess_trials)
    
    ess_by_alpha.append((alpha, mean_ess, min_ess, std_ess))
    
    print(f"{alpha:6.2f} {mean_ess:10.1f} {min_ess:10.1f} "
          f"{std_ess:10.1f} {mean_ess/2000:11.1%}")

# 주고받음 그려 보기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

alphas_plot = [x[0] for x in ess_by_alpha]
mean_ess_plot = [x[1] for x in ess_by_alpha]
min_ess_plot = [x[2] for x in ess_by_alpha]

ax = axes[0]
ax.plot(alphas_plot, mean_ess_plot, 'bo-', linewidth=2, markersize=8,
        label='Mean ESS')
ax.plot(alphas_plot, min_ess_plot, 'r^--', linewidth=2, markersize=8,
        label='Min ESS (over 50 trials)')
ax.set_xlabel('α (weight on targeted component)', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS vs Mixture Parameter α', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axvline(0.8, color='green', linestyle=':', linewidth=2,
           label='Typical choice', alpha=0.7)

# α에 따른 제안 그려 보기
ax = axes[1]
x_plot = np.linspace(-6, 8, 1000)
ax.plot(x_plot, bimodal(x_plot), 'k-', linewidth=3,
        label='Target', alpha=0.7)

for alpha_vis in [0.5, 0.8, 0.95]:
    proposal_vis = DefensiveMixture(targeted_poor, defensive_broad, alpha_vis)
    ax.plot(x_plot, proposal_vis.pdf(x_plot), '--', linewidth=2,
            label=f'α={alpha_vis}', alpha=0.7)

ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Proposals for Different α', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/defensive_alpha_tradeoff.png',
            dpi=300, bbox_inches='tight')


# 보기 4: 지킴 중요도 표집을 쓴 베이즈 추론
# =============================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Defensive IS for Bayesian Inference")
print("=" * 70)

print("""
모형: y ~ N(θ, 1), 관측 n=30개
앞확률: θ ~ N(0, 5) [흐릿함]
제안: 라플라스 어림(겨냥) + 앞확률(지킴)
""")

# 데이터를 생성한다
theta_true_ex4 = 4.0
sigma_ex4 = 1.0
n_obs_ex4 = 30
data_ex4 = np.random.normal(theta_true_ex4, sigma_ex4, n_obs_ex4)

print(f"\nTrue θ = {theta_true_ex4}")
print(f"Data: n={n_obs_ex4}, sample mean = {np.mean(data_ex4):.3f}")

# 앞확률
prior_ex4 = stats.norm(0, 5)

# 고르게 하지 않은 뒤확률
def log_posterior_ex4(theta):
    log_lik = -0.5 * np.sum((data_ex4 - theta)**2) / sigma_ex4**2
    log_prior = prior_ex4.logpdf(theta)
    return log_lik + log_prior

def posterior_ex4(theta):
    return np.exp(log_posterior_ex4(theta))

# 라플라스 어림(겨냥 성분)
# 이 경우 뒤확률은 가우스이다
tau_n_sq = 1.0 / (1.0/25 + n_obs_ex4/sigma_ex4**2)
mu_n = tau_n_sq * (0/25 + np.sum(data_ex4)/sigma_ex4**2)

laplace_ex4 = stats.norm(mu_n, np.sqrt(tau_n_sq))

print(f"\nLaplace approximation: N({mu_n:.3f}, {np.sqrt(tau_n_sq):.3f})")

# 지킴 제안
alpha_ex4 = 0.8
defensive_proposal = DefensiveMixture(
    laplace_ex4,    # 겨냥: 라플라스 어림
    prior_ex4,      # 지킴: 앞확률
    alpha=alpha_ex4
)

# 순수 라플라스와 견주기
proposals_ex4 = {
    'Laplace only': laplace_ex4,
    f'Defensive α={alpha_ex4}': defensive_proposal,
}

results_ex4 = compare_proposals(posterior_ex4, proposals_ex4,
                                lambda x: x, n_samples=3000,
                                n_replications=100)

print(f"\n{'Proposal':<20} {'Mean ESS':>10} {'Min ESS':>10} {'Std Est':>10}")
print("-" * 55)

for name, stats_dict in results_ex4.items():
    print(f"{name:<20} {stats_dict['mean_ess']:10.1f} "
          f"{stats_dict['min_ess']:10.1f} {stats_dict['std_estimate']:10.4f}")

print("\nDefensive mixing provides insurance against model misspecification!")


# 간추린 그림
# ===================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 칸 1: 개념 그림
ax = axes[0, 0]
x_concept = np.linspace(-6, 10, 1000)

# 본뜬 과녁
target_concept = 0.7 * stats.norm.pdf(x_concept, 2, 1)

# 겨냥 제안(너무 좁음)
targeted_concept = stats.norm(2, 0.8)

# 지킴 성분(넓음)
defensive_concept = stats.norm(0, 4)

# 지킴 섞음
alpha_concept = 0.8
defensive_mix = DefensiveMixture(targeted_concept, defensive_concept, alpha_concept)

ax.plot(x_concept, target_concept, 'r-', linewidth=3, label='Target', alpha=0.7)
ax.plot(x_concept, targeted_concept.pdf(x_concept), 'b--', linewidth=2,
        label='Targeted (risky)')
ax.plot(x_concept, defensive_mix.pdf(x_concept), 'g:', linewidth=2,
        label=f'Defensive (α={alpha_concept})')
ax.fill_between(x_concept, 0, defensive_concept.pdf(x_concept),
                alpha=0.2, color='orange', label='Defensive component')

ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Defensive Mixture Concept', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 칸 2: ESS 견주기(보기 1에서)
ax = axes[0, 1]
proposal_names = list(results_ex1.keys())
mean_ess_values = [results_ex1[name]['mean_ess'] for name in proposal_names]
min_ess_values = [results_ex1[name]['min_ess'] for name in proposal_names]

x_pos = np.arange(len(proposal_names))
width = 0.35

ax.bar(x_pos - width/2, mean_ess_values, width, label='Mean ESS',
       alpha=0.7, color='steelblue', edgecolor='black')
ax.bar(x_pos + width/2, min_ess_values, width, label='Min ESS',
       alpha=0.7, color='orange', edgecolor='black')

ax.set_ylabel('ESS', fontsize=11)
ax.set_title('Robustness: Mean vs Minimum ESS', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names],
                   fontsize=8, rotation=0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 칸 3: 치우침 견주기(보기 2에서)
ax = axes[1, 0]
proposal_names_ex2 = list(results_ex2.keys())
biases = [results_ex2[name]['mean_estimate'] - true_mean
          for name in proposal_names_ex2]
colors_bias = ['red' if abs(b) > 0.5 else 'green' for b in biases]

ax.bar(range(len(proposal_names_ex2)), biases, color=colors_bias,
       alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Bias', fontsize=11)
ax.set_title('Bias Under Misspecification', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(proposal_names_ex2)))
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names_ex2],
                   fontsize=9, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 칸 4: 흩어짐 견주기
ax = axes[1, 1]
std_values = [results_ex1[name]['std_estimate'] for name in proposal_names]

ax.bar(range(len(proposal_names)), std_values,
       color='purple', alpha=0.7, edgecolor='black')
ax.set_ylabel('Standard Deviation of Estimates', fontsize=11)
ax.set_title('Estimation Variance', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(proposal_names)))
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names],
                   fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/defensive_summary.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 지킴 중요도 표집은 튼튼함을 준다:
   - 겨냥 제안과 넓은 지킴 성분 섞기
   - q_def(θ) = α q(θ) + (1-α) m(θ)
   - 너그러운 조건에서 흩어짐이 묶임을 보장한다

2. 지킴 성분 고르기:
   - 과녁보다 꼬리가 두꺼워야 한다
   - 흔한 고름: 앞확률, 고른 분포, 코시, 아주 넓은 가우스
   - 과녁의 받침 전체를 덮어야 한다

3. 섞음 매개변수 α:
   - α이 높으면(0.9-0.95) ESS은 낫지만 덜 튼튼하다
   - α이 낮으면(0.5-0.7) ESS은 나쁘지만 더 튼튼하다
   - 보통: α ∈ [0.7, 0.9]
   - 효율과 튼튼함의 주고받음

4. 이론의 보장:
   - 어떤 c > 0에 대해 m(θ) ≥ c·π(θ)이면 흩어짐이 묶인다
   - 순수 겨냥 제안에는 그런 보장이 없다
   - 표준 중요도 표집보다 나은 핵심 강점

5. 지킴 중요도 표집을 언제 쓰나:
   - 과녁의 꼴이 확실하지 않을 때
   - 꼬리 두꺼운 과녁
   - 제안을 잘못 잡을 위험
   - 보장된 성능이 필요할 때
   - 살펴보기 위한 분석

6. 튼튼함의 잣대:
   - 최소 ESS(여러 번 돌린 것 가운데)
   - 최악의 경우 성능
   - 평균 ESS보다 더 중요하다

7. ESS의 주고받음:
   - 지킴 중요도 표집: 평균 ESS이 낮다
   - 그러나 최소 ESS이 훨씬 높다
   - 최악의 경우 성능이 더 낫다
   - 여러 시나리오에 걸쳐 더 안정하다

8. 실전에서의 이로움:
   - "한 번 정하고 잊기" - 맞출 것이 적다
   - 잘못 잡아도 서서히 나빠질 뿐이다
   - 실제 운영 체계에서 마음이 놓인다
   - 자동화에서 특히 값지다

9. 순수 겨냥 제안과 견주기:
   - 순수 겨냥: 잘될 때는 아주 좋지만 최악일 때는 재앙이다
   - 지킴: 잘될 때는 좋고 최악일 때도 받아들일 만하다
   - 지킴은 무너짐에 대비한 "보험"이다

10. 구현 요령:
    - 기본값으로 α = 0.8부터 시작하기
    - 앞확률이나 아주 넓은 분포를 지킴 성분으로 쓰기
    - 평균 ESS과 최소 ESS을 함께 지켜보기
    - 안정 요구에 따라 α 다듬기
    - ESS에 따라 α을 맞춰 가는 것을 생각해 보기

11. 한계:
    - 잘 맞춘 순수 제안보다 평균 ESS이 낮다
    - 셈 값이 더 든다(섞음의 값 매기기)
    - 그래도 그럴듯한 지킴 성분이 필요하다

12. 쓰임새:
    - 실제 운영하는 베이즈 추론 체계
    - 저절로 하는 매개변수 어림
    - 안전이 결정적인 쓰임새
    - 제안 맞추기가 어려울 때
    - 자료를 살펴보는 분석
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
방어적 중요도 표집 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_defensivemixture():
        model = DefensiveMixture(...)
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
