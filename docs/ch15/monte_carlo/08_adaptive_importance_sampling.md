# 알아서 맞추는 중요도 표집

08_adaptive_importance_sampling.py 나아간 수준: 알아서 맞추는 중요도 표집(AIS)

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
08_adaptive_importance_sampling.py

앞선 단계: 맞춰 가는 중요도 표집(AIS)

이 단원은 맞춰 가는 중요도 표집을 구현한다. 여기서 제안
분포는 앞서 뽑은 표본에 따라 되풀이하며 다듬어진다.

수학적 바탕:
---------------------
표준 중요도 표집은 붙박인 제안 q(θ)을 쓴다. AIS은 q을 되풀이하며 낫게 한다:

알고리즘(무리 몬테카를로 - PMC):
1. 첫값 잡기: q₀(θ) 고르기
2. t = 1, 2, ..., T에 대해:
   가) i=1,...,n에 대해 θᵢᵗ ~ qₜ₋₁(θ) 표집
   나) 무게 wᵢᵗ = π(θᵢᵗ)/qₜ₋₁(θᵢᵗ) 셈하기
   다) 제안 새로 고치기: {θᵢᵗ, wᵢᵗ}에 바탕을 둔 qₜ(θ)
3. 중요도 무게를 붙여 마지막 표본 돌려주기

흔한 새로 고침 전략:
1. 무게 큰 표본에 가운데를 맞춘 가우스 섞음
2. 무게 알맹이 밀도 어림
3. 매개변수 맞추기(이를테면 평균과 공분산을 새로 고친 가우스)

이점:
- 과녁 분포에 저절로 맞춰 간다
- 봉우리 여럿인 짜임을 찾아낼 수 있다
- 대체로 붙박인 제안보다 ESS이 높다

어려움:
- 무게를 제대로 셈하려면 조심해야 한다
- 그 자리 봉우리에 갇힐 수 있다
- 제안을 새로 고치는 셈 값

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from typing import Tuple, List, Callable

np.random.seed(42)
sns.set_style("whitegrid")


class AdaptiveImportanceSampler:
    """
    섞음 제안을 쓰는 맞춰 가는 중요도 표집.
    """
    
    def __init__(self, target_log_density: Callable, dim: int,
                 n_components: int = 5, initial_scale: float = 2.0):
        """
        매개변수:
        -----------
        target_log_density : function
            고르게 하지 않은 과녁 밀도의 로그
        dim : int
            매개변수 공간의 차원
        n_components : int
            섞음 성분의 개수
        initial_scale : float
            제안 성분의 첫 규모
        """
        self.target_log_density = target_log_density
        self.dim = dim
        self.n_components = n_components
        self.initial_scale = initial_scale
        
        # 되풀이를 담을 저장 공간
        self.samples_history = []
        self.weights_history = []
        self.ess_history = []
        
    def initialize_proposal(self, initial_mean: np.ndarray = None):
        """
        initial_mean에 가운데를 맞춘 N(μ, σ²I)으로 섞음 제안 첫값 잡기.
        """
        if initial_mean is None:
            initial_mean = np.zeros(self.dim)
        
        # 넓은 가우스 하나로 시작
        self.mixture_means = [initial_mean.copy()]
        self.mixture_covs = [np.eye(self.dim) * self.initial_scale**2]
        self.mixture_weights = [1.0]
        
    def proposal_log_density(self, theta: np.ndarray) -> float:
        """
        theta에서 지금 섞음 제안의 로그 밀도 값 매기기.
        
        q(θ) = Σⱼ αⱼ N(θ|μⱼ, Σⱼ)
        """
        # 성분마다 값 매기기
        log_densities = []
        for mean, cov, weight in zip(self.mixture_means, self.mixture_covs, 
                                      self.mixture_weights):
            # 다변량 정규의 로그 밀도
            component = stats.multivariate_normal(mean, cov)
            log_densities.append(np.log(weight + 1e-300) + component.logpdf(theta))
        
        # 수치 안정을 위한 로그-합-지수
        return logsumexp(log_densities)
    
    def sample_from_proposal(self, n_samples: int) -> np.ndarray:
        """
        지금 섞음 제안에서 표집하기.
        """
        samples = []
        
        # 성분 배정 표집
        component_probs = np.array(self.mixture_weights)
        component_probs /= component_probs.sum()
        components = np.random.choice(len(self.mixture_means), 
                                     size=n_samples, p=component_probs)
        
        # 배정된 성분에서 표집
        for i in range(n_samples):
            comp_idx = components[i]
            mean = self.mixture_means[comp_idx]
            cov = self.mixture_covs[comp_idx]
            sample = np.random.multivariate_normal(mean, cov)
            samples.append(sample)
        
        return np.array(samples)
    
    def update_proposal(self, samples: np.ndarray, weights: np.ndarray, 
                       method: str = 'resample'):
        """
        무게 표본에 따라 섞음 제안 새로 고치기.
        
        방법:
        --------
        'resample': 다시 표집한 알갱이에 섞음을 맞춘다
        'weighted_means': 무게가 큰 표본에 성분을 놓는다
        """
        # 무게 고르게 하기
        normalized_weights = weights / np.sum(weights)
        
        if method == 'resample':
            # 무게에 따라 다시 표집하기
            indices = np.random.choice(len(samples), size=self.n_components,
                                      replace=True, p=normalized_weights)
            selected_samples = samples[indices]
            
            # 모든 표본의 경험 공분산 셈하기
            weighted_cov = np.cov(samples.T, aweights=normalized_weights)
            
            # 수치 안정을 위해 작은 벌주기 더하기
            weighted_cov += np.eye(self.dim) * 1e-4
            
            # 살펴보기-써먹기 주고받음을 위해 공분산 오그라뜨리기
            shrinkage = 0.7
            weighted_cov *= shrinkage
            
            # 섞음 성분 새로 고치기
            self.mixture_means = [s for s in selected_samples]
            self.mixture_covs = [weighted_cov for _ in range(self.n_components)]
            self.mixture_weights = [1.0/self.n_components] * self.n_components
            
        elif method == 'weighted_means':
            # 무게가 큰 표본 고르기
            top_indices = np.argsort(normalized_weights)[-self.n_components:]
            
            # 고른 표본을 평균으로 쓰기
            self.mixture_means = [samples[i] for i in top_indices]
            
            # 맞춰 가는 공분산 셈하기
            weighted_cov = np.cov(samples.T, aweights=normalized_weights)
            weighted_cov += np.eye(self.dim) * 1e-4
            weighted_cov *= 0.5  # 안정을 위해 오그라뜨림
            
            self.mixture_covs = [weighted_cov for _ in range(self.n_components)]
            
            # 중요도 무게에 비례하는 무게
            selected_weights = normalized_weights[top_indices]
            self.mixture_weights = (selected_weights / selected_weights.sum()).tolist()
    
    def run(self, n_samples: int, n_iterations: int, 
            update_method: str = 'resample',
            initial_mean: np.ndarray = None,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        맞춰 가는 중요도 표집 돌리기.
        
        반환값:
        --------
        samples : 꼴이 (n_iterations * n_samples, dim)인 배열
        weights : 고르게 한 무게의 배열
        """
        self.initialize_proposal(initial_mean)
        
        all_samples = []
        all_log_weights = []
        
        if verbose:
            print(f"\nRunning Adaptive IS: {n_iterations} iterations, "
                  f"{n_samples} samples each")
            print("=" * 60)
        
        for t in range(n_iterations):
            # 지금 제안에서 표집
            samples = self.sample_from_proposal(n_samples)
            
            # 로그 중요도 무게 셈하기
            log_weights = np.array([
                self.target_log_density(s) - self.proposal_log_density(s)
                for s in samples
            ])
            
            # 표본과 무게 저장
            all_samples.append(samples)
            all_log_weights.append(log_weights)
            
            # 이번 되풀이의 무게 고르게 하기
            log_weights_normalized = log_weights - logsumexp(log_weights)
            weights = np.exp(log_weights_normalized)
            
            # 진단을 위한 ESS 셈하기
            ess = 1.0 / np.sum(weights**2)
            self.ess_history.append(ess)
            
            if verbose:
                print(f"Iteration {t+1:3d}: ESS = {ess:7.1f} ({ess/n_samples:5.1%})")
            
            # 제안 새로 고치기(마지막 되풀이는 빼고)
            if t < n_iterations - 1:
                self.update_proposal(samples, np.exp(log_weights), 
                                    method=update_method)
            
            self.samples_history.append(samples)
            self.weights_history.append(weights)
        
        # 표본을 모두 합치고 마지막 무게 다시 셈하기
        all_samples = np.vstack(all_samples)
        all_log_weights = np.concatenate(all_log_weights)
        
        # 마지막 고르게 한 무게
        final_log_weights = all_log_weights - logsumexp(all_log_weights)
        final_weights = np.exp(final_log_weights)
        
        return all_samples, final_weights


# 보기 1: 봉우리 둘인 1차원 과녁
# ==========================
print("=" * 70)
print("EXAMPLE 1: Adaptive IS for 1D Bimodal Distribution")
print("=" * 70)

# 과녁: 가우스 둘의 섞음
def target_log_density_1d(theta):
    """봉우리 둘인 과녁: 0.3*N(-3,1) + 0.7*N(4,1.5)"""
    if theta.ndim == 0 or len(theta) == 1:
        theta = np.atleast_1d(theta)
    
    # 첫 봉우리: N(-3, 1)
    log_p1 = stats.norm.logpdf(theta[0], -3, 1) + np.log(0.3)
    
    # 둘째 봉우리: N(4, 1.5)
    log_p2 = stats.norm.logpdf(theta[0], 4, 1.5) + np.log(0.7)
    
    return logsumexp([log_p1, log_p2])

# 그리기 위한 참 과녁
def true_target_1d(x):
    return 0.3 * stats.norm.pdf(x, -3, 1) + 0.7 * stats.norm.pdf(x, 4, 1.5)

# 맞춰 가는 중요도 표집 돌리기
sampler_1d = AdaptiveImportanceSampler(
    target_log_density=target_log_density_1d,
    dim=1,
    n_components=3,
    initial_scale=3.0
)

samples_1d, weights_1d = sampler_1d.run(
    n_samples=200,
    n_iterations=10,
    update_method='resample',
    initial_mean=np.array([0.0])
)

final_ess = 1.0 / np.sum(weights_1d**2)
print(f"\nFinal ESS: {final_ess:.1f} out of {len(samples_1d)} samples")
print(f"Efficiency: {final_ess/len(samples_1d):.1%}")

# 맞춰 가는 과정 그려 보기
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.ravel()

x_plot = np.linspace(-8, 10, 1000)
iterations_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for idx, iter_num in enumerate(iterations_to_plot):
    ax = axes[idx]
    
    # 참 과녁 그리기
    ax.plot(x_plot, true_target_1d(x_plot), 'r-', linewidth=2, 
            label='Target', alpha=0.7)
    
    # 이번 되풀이의 표본 그리기
    if iter_num < len(sampler_1d.samples_history):
        samples_iter = sampler_1d.samples_history[iter_num]
        weights_iter = sampler_1d.weights_history[iter_num]
        
        # 표본의 막대그림
        ax.hist(samples_iter.flatten(), bins=30, density=True, alpha=0.5,
                color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 무게로 색칠한 표본 흩뿌리기
        y_pos = np.zeros(len(samples_iter))
        scatter = ax.scatter(samples_iter.flatten(), y_pos, 
                           c=weights_iter*len(weights_iter), 
                           cmap='hot', s=50, alpha=0.6, 
                           edgecolors='black', linewidth=0.5)
        
        ess = sampler_1d.ess_history[iter_num]
        ax.set_title(f'Iteration {iter_num+1}: ESS={ess:.0f}',
                    fontsize=11, fontweight='bold')
    
    ax.set_xlim([-8, 10])
    ax.set_ylim([0, 0.3])
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/adaptive_1d_evolution.png',
            dpi=300, bbox_inches='tight')


# 보기 2: 바나나 꼴 2차원 분포
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Adaptive IS for 2D Banana Distribution")
print("=" * 70)

def target_log_density_banana(theta):
    """
    바나나 꼴 분포(로젠브록과 비슷함).
    
    p(θ₁, θ₂) ∝ exp(-0.5[(θ₁-2)²/4 + (θ₂-θ₁²)²])
    """
    theta = np.atleast_1d(theta)
    theta1, theta2 = theta[0], theta[1]
    
    term1 = -0.5 * (theta1 - 2)**2 / 4.0
    term2 = -0.5 * (theta2 - theta1**2)**2
    
    return term1 + term2

# 맞춰 가는 중요도 표집 돌리기
sampler_banana = AdaptiveImportanceSampler(
    target_log_density=target_log_density_banana,
    dim=2,
    n_components=8,
    initial_scale=3.0
)

samples_banana, weights_banana = sampler_banana.run(
    n_samples=300,
    n_iterations=15,
    update_method='resample',
    initial_mean=np.array([0.0, 0.0])
)

final_ess_banana = 1.0 / np.sum(weights_banana**2)
print(f"\nFinal ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")

# 2차원 맞춰 가기 그려 보기
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

iterations_2d = [0, 1, 2, 4, 6, 8, 10, 14]

# 참 밀도를 위한 격자 만들기
x1_grid = np.linspace(-2, 6, 100)
x2_grid = np.linspace(-5, 25, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = np.exp(target_log_density_banana(np.array([X1[i,j], X2[i,j]])))

for idx, iter_num in enumerate(iterations_2d):
    ax = axes[idx]
    
    # 참 과녁의 등고선
    ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
    
    # 표본 그리기
    if iter_num < len(sampler_banana.samples_history):
        samples_iter = sampler_banana.samples_history[iter_num]
        weights_iter = sampler_banana.weights_history[iter_num]
        
        scatter = ax.scatter(samples_iter[:, 0], samples_iter[:, 1],
                           c=weights_iter*len(weights_iter), cmap='viridis',
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ess = sampler_banana.ess_history[iter_num]
        ax.set_title(f'Iter {iter_num+1}: ESS={ess:.0f}',
                    fontsize=11, fontweight='bold')
    
    ax.set_xlabel('θ₁', fontsize=10)
    ax.set_ylabel('θ₂', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/adaptive_2d_banana.png',
            dpi=300, bbox_inches='tight')


# 보기 3: 붙박인 제안과 견주기
# =======================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Adaptive vs Fixed Proposal IS")
print("=" * 70)

# 붙박인 제안의 중요도 표집
n_fixed_samples = 3000  # 맞춰 가는 방식과 전체가 같음(되풀이 15번 × 표본 200개)
fixed_proposal = stats.multivariate_normal([0, 0], [[9, 0], [0, 9]])
samples_fixed = fixed_proposal.rvs(size=n_fixed_samples)

# 붙박인 제안의 무게 셈하기
log_weights_fixed = np.array([
    target_log_density_banana(s) - fixed_proposal.logpdf(s)
    for s in samples_fixed
])
log_weights_fixed_norm = log_weights_fixed - logsumexp(log_weights_fixed)
weights_fixed = np.exp(log_weights_fixed_norm)

ess_fixed = 1.0 / np.sum(weights_fixed**2)

print(f"\nFixed Proposal IS:")
print(f"  ESS: {ess_fixed:.1f} out of {n_fixed_samples} samples")
print(f"  Efficiency: {ess_fixed/n_fixed_samples:.1%}")

print(f"\nAdaptive IS:")
print(f"  ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")
print(f"  Efficiency: {final_ess_banana/len(samples_banana):.1%}")

print(f"\nImprovement: {final_ess_banana/ess_fixed:.2f}x better ESS")

# 견줌을 그려 본다
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 붙박인 제안
ax = axes[0]
ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
scatter = ax.scatter(samples_fixed[:, 0], samples_fixed[:, 1],
                    c=weights_fixed*len(weights_fixed), cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.set_title(f'Fixed Proposal: ESS={ess_fixed:.0f} ({ess_fixed/n_fixed_samples:.1%})',
            fontsize=12, fontweight='bold')
ax.set_xlabel('θ₁', fontsize=11)
ax.set_ylabel('θ₂', fontsize=11)
plt.colorbar(scatter, ax=ax, label='Weight × n')

# 맞춰 가는 제안
ax = axes[1]
ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
scatter = ax.scatter(samples_banana[:, 0], samples_banana[:, 1],
                    c=weights_banana*len(weights_banana), cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.set_title(f'Adaptive IS: ESS={final_ess_banana:.0f} ({final_ess_banana/len(samples_banana):.1%})',
            fontsize=12, fontweight='bold')
ax.set_xlabel('θ₁', fontsize=11)
ax.set_ylabel('θ₂', fontsize=11)
plt.colorbar(scatter, ax=ax, label='Weight × n')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/fixed_vs_adaptive.png',
            dpi=300, bbox_inches='tight')


# 보기 4: 되풀이에 따른 ESS의 흘러감
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Tracking ESS Improvement")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1차원 경우
ax = axes[0]
iterations = np.arange(1, len(sampler_1d.ess_history) + 1)
ess_values = sampler_1d.ess_history
ax.plot(iterations, ess_values, 'o-', linewidth=2, markersize=8,
        color='steelblue', label='ESS')
ax.axhline(200, color='red', linestyle='--', linewidth=2, 
          label='n per iteration', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('1D Bimodal: ESS Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# 2차원 경우
ax = axes[1]
iterations_2d = np.arange(1, len(sampler_banana.ess_history) + 1)
ess_values_2d = sampler_banana.ess_history
ax.plot(iterations_2d, ess_values_2d, 'o-', linewidth=2, markersize=8,
        color='darkgreen', label='ESS')
ax.axhline(300, color='red', linestyle='--', linewidth=2,
          label='n per iteration', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('2D Banana: ESS Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_evolution.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 맞춰 가는 중요도 표집은 제안을 되풀이하며 다듬는다:
   - 넓은 첫 제안으로 시작하기
   - 무게 표본에 따라 새로 고치기
   - 확률 높은 구역으로 모인다

2. 섞음 제안이 잘 듣는다:
   - 성분이 여럿이면 복잡한 꼴을 덮는다
   - 봉우리 여럿을 잡아낼 수 있다
   - 살펴보기와 써먹기의 균형

3. 새로 고치는 전략:
   - 다시 표집: 무게 분포에서 뽑기
   - 무게 평균: 무게 큰 표본에 가운데 맞추기
   - 둘 다 꼴을 잡는 데 경험 공분산을 쓴다

4. ESS은 보통 되풀이할수록 나아진다:
   - 앞부분: ESS이 낮고 살펴보는 중
   - 가운데: 제안이 맞춰 가면서 ESS이 커진다
   - 뒷부분: ESS이 가장 좋은 값에서 평평해진다

5. 붙박인 제안보다 나은 점:
   - 좋은 제안을 미리 알 필요가 없다
   - 과녁의 꼴에 저절로 맞춰 간다
   - ESS을 훨씬 높일 수 있다
   - 봉우리가 여럿인 과녁을 다룬다

6. 셈에서 살필 점:
   - 붙박인 중요도 표집보다 표본마다 더 비싸다
   - 그러나 같은 정확도에 표본이 더 적게 든다
   - 주고받음: 맞춰 가는 값과 ESS 나아짐

7. 맞춰 가는 중요도 표집을 언제 쓰나:
   - 과녁의 꼴을 모를 때
   - 복잡하거나 봉우리가 여럿인 분포
   - 붙박인 제안의 ESS이 나쁠 때
   - 차례차례 결정하는 문제

8. 실전 요령:
   - 넓은 첫 제안으로 시작하기
   - 섞음 성분 5개에서 10개 쓰기
   - ESS의 모임 지켜보기
   - 지나친 자신을 피하려고 공분산 오그라뜨리기
   - 보통 ESS > 0.3n이면 아주 좋다

9. MCMC와 견주기:
   - AIS: 독립 표본, 태우기 없음
   - MCMC: 표본이 서로 얽혀 있고 태우기가 필요하다
   - 좋은 제안을 배울 수 있으면 AIS이 낫다
   - 차원이 아주 높으면 MCMC가 낫다
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 표집과 어림 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 몬테카를로 방법 구현의 자리에서 치우침과 흩어짐의 주고받음을 설명하여라. 핵심 웃매개변수가 이 주고받음에 어떻게 영향을 주는가?

??? success "연습문제 3 풀이"
    몬테카를로 방법에서 치우침과 흩어짐의 주고받음은 모형의 복잡함과 표본 크기로 드러난다. 더 복잡한 모형(이를테면 섞음 성분이 더 많거나 층이 더 깊은 모형)은 치우침을 줄이지만 흩어짐을 키우며, 자료가 적을 때 특히 그렇다. 핵심 웃매개변수가 이를 다스린다. 앞확률의 세기가 벌주기 노릇을 하고(센 앞확률은 흩어짐을 줄이지만 치우침을 키울 수 있다), 표본 크기가 어림의 정확도에 영향을 주며(표본이 많을수록 흩어짐이 줄고), 모형의 복잡함이 유연함을 정한다. 가장 좋은 균형은 쓸 수 있는 자료의 양과 바탕 분포의 참된 복잡함에 달렸다.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — 알아서 맞추는 중요도 표집

학습 루프는 표준적인 PyTorch 패턴을 따른다.

고갱이 갈래는 `AdaptiveImportanceSampler`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
