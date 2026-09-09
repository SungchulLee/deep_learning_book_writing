# 스스로 고르게 하는 중요도 표집

02_self_normalized_IS.py 첫걸음 수준: 스스로 고르게 하는 중요도 표집

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 몬테카를로 방법의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
02_self_normalized_IS.py

첫걸음 단계: 스스로 고르게 하는 중요도 표집

이 단원은 스스로 고르게 하는 중요도 표집을 소개한다. 이는
이는 뒤확률의 고르게 하는 상수를 모르는 베이즈 추론에 꼭 필요하다.

수학적 바탕:
--------------------
문제: 우리는 상수배를 빼고만 π(θ)을 안다:
    π(θ) = γ(θ)/Z, 여기서 Z = ∫γ(θ)dθ은 알 수 없다

베이즈 추론에서:
    γ(θ) = p(y|θ)p(θ)  (가능도 × 앞확률)
    Z = p(y) = ∫p(y|θ)p(θ)dθ  (주변 가능도, 흔히 다룰 수 없다)

풀이: 스스로 고르게 하는 중요도 표집(SNIS)

고르게 하지 않은 무게: w̃ᵢ = γ(θᵢ)/q(θᵢ)

스스로 고르게 한 어림꼴:
    Ê[h(θ)] = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]
             = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]

성질:
- 치우쳤지만 한결같다
- 치우침 = O(1/n)
- 흔히 고르게 한 중요도 표집보다 흩어짐이 작다
- Z을 알 필요가 없다

지은이: 베이즈 추론 교육 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

np.random.seed(42)
sns.set_style("whitegrid")


def self_normalized_importance_sampling(unnormalized_target, proposal_dist, 
                                        h_function, n_samples):
    """
    고르게 하지 않은 과녁 분포를 위한, 스스로 고르게 하는 중요도 표집.
    
    매개변수:
    -----------
    unnormalized_target : callable
        π(θ) = γ(θ)/Z일 때의 함수 γ(θ)
        베이즈에서는: γ(θ) = p(y|θ)p(θ)
    proposal_dist : scipy.stats distribution
        제안 분포 q(θ)
    h_function : callable
        기댓값을 구하려는 함수
    n_samples : int
        표본의 개수
        
    반환값:
    --------
    estimate : float
        스스로 고르게 한 중요도 표집 어림값
    samples : array
        제안에서 뽑은 표본
    normalized_weights : array
        고르게 한 중요도 무게
    unnormalized_weights : array
        고르게 하지 않은 중요도 무게
        
    알고리즘:
    ---------
    1. i=1,...,n에 대해 θᵢ ~ q(θ) 표집
    2. 고르게 하지 않은 무게 셈하기: w̃ᵢ = γ(θᵢ)/q(θᵢ)
    3. 무게 고르게 하기: wᵢ = w̃ᵢ / Σⱼw̃ⱼ
    4. 어림하기: Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
    """
    # 걸음 1: 제안에서 표본 뽑기
    samples = proposal_dist.rvs(size=n_samples)
    
    # 걸음 2: 고르게 하지 않은 과녁 γ(θ) 값 매기기
    gamma_values = unnormalized_target(samples)
    
    # 걸음 3: 제안 밀도 q(θ) 값 매기기
    q_values = proposal_dist.pdf(samples)
    
    # 걸음 4: 고르게 하지 않은 무게 w̃ᵢ = γ(θᵢ)/q(θᵢ) 셈하기
    unnormalized_weights = gamma_values / (q_values + 1e-300)
    
    # 걸음 5: 무게 고르게 하기
    # wᵢ = w̃ᵢ / Σⱼw̃ⱼ
    weight_sum = np.sum(unnormalized_weights)
    normalized_weights = unnormalized_weights / weight_sum
    
    # 걸음 6: 표본 점에서 함수 h 값 매기기
    h_values = h_function(samples)
    
    # 걸음 7: 스스로 고르게 한 어림값 셈하기
    # Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
    estimate = np.sum(normalized_weights * h_values)
    
    return estimate, samples, normalized_weights, unnormalized_weights


def compute_ess(normalized_weights):
    """
    실효 표본 크기(ESS) 셈하기.
    
    ESS = 1 / Σᵢwᵢ²
    
    고르게 하지 않은 무게를 쓰는 다른 공식:
    ESS = (Σᵢw̃ᵢ)² / Σᵢw̃ᵢ²
    
    해석:
    - ESS ≈ n: 표본의 무게가 엇비슷하다(좋음)
    - ESS << n: 몇몇 표본이 판친다(나쁨)
    - ESS / n은 중요도 표집의 "효율"이다
    """
    ess = 1.0 / np.sum(normalized_weights**2)
    return ess


# 보기 1: 고르게 하는 상수를 모르는 단순 가우스
# ==========================================================
print("=" * 70)
print("EXAMPLE 1: Self-Normalized IS for π(θ) = γ(θ)/Z")
print("=" * 70)

# 고르게 하지 않은 과녁 정하기: γ(θ) = exp(-0.5(θ-3)²)
# 이는 N(3, 1)에 비례하지만 고르게 하는 상수가 없다
def gamma_function(theta):
    """
    고르게 하지 않은 가우스: γ(θ) = exp(-0.5(θ-μ)²/σ²)
    1/√(2πσ²) 인수가 빠졌다
    """
    mu, sigma = 3.0, 1.0
    return np.exp(-0.5 * ((theta - mu) / sigma)**2)

# 참으로 고르게 한 분포(확인용)
target_dist = stats.norm(3, 1)

# 제안 분포
proposal_dist = stats.norm(0, 2)

# 어림할 함수: h(θ) = θ²
h_function = lambda theta: theta**2

# 참 기댓값
true_expectation = 3**2 + 1**2  # N(3,1)의 E[θ²]

print(f"\nTrue E[θ²]: {true_expectation:.6f}")

# 스스로 고르게 하는 중요도 표집 돌리기
n_samples = 1000
estimate, samples, norm_weights, unnorm_weights = self_normalized_importance_sampling(
    gamma_function, proposal_dist, h_function, n_samples
)

ess = compute_ess(norm_weights)
efficiency = ess / n_samples * 100

print(f"\nSelf-Normalized IS Results (n={n_samples}):")
print(f"  Estimate: {estimate:.6f}")
print(f"  Error: {abs(estimate - true_expectation):.6f}")
print(f"  ESS: {ess:.1f}")
print(f"  Efficiency: {efficiency:.1f}%")

# 시각화한다
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 칸 1: 고르게 하지 않은 과녁과 고르게 한 과녁
x = np.linspace(-5, 8, 1000)
ax = axes[0, 0]
ax.plot(x, gamma_function(x), 'b-', linewidth=2, 
        label='Unnormalized γ(θ)')
ax.plot(x, target_dist.pdf(x), 'r--', linewidth=2, 
        label='Normalized π(θ)')
ax.plot(x, proposal_dist.pdf(x), 'g:', linewidth=2, 
        label='Proposal q(θ)')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density (arbitrary scale)', fontsize=12)
ax.set_title('Unnormalized Target vs Normalized', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 2: 무게 분포
ax = axes[0, 1]
ax.hist(norm_weights, bins=50, density=True, alpha=0.7, 
        color='purple', edgecolor='black')
ax.set_xlabel('Normalized Weight wᵢ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Distribution of Normalized Weights\nESS = {ess:.1f} ({efficiency:.1f}%)', 
             fontsize=13, fontweight='bold')
uniform_weight = 1.0 / n_samples
ax.axvline(uniform_weight, color='red', linestyle='--', linewidth=2,
           label=f'Uniform = {uniform_weight:.4f}')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 3: 쌓인 무게 분포
sorted_weights = np.sort(norm_weights)[::-1]  # 내림차순 정렬
cumulative_weights = np.cumsum(sorted_weights)
ax = axes[1, 0]
ax.plot(np.arange(1, len(sorted_weights)+1), cumulative_weights, 
        'b-', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, 
           label='50% of total weight')
ax.axhline(0.9, color='orange', linestyle='--', linewidth=2,
           label='90% of total weight')
ax.set_xlabel('Number of Samples (sorted by weight)', fontsize=12)
ax.set_ylabel('Cumulative Weight', fontsize=12)
ax.set_title('Cumulative Weight Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 무게의 50%과 90%을 차지하는 표본 수 찾기
n_50 = np.searchsorted(cumulative_weights, 0.5) + 1
n_90 = np.searchsorted(cumulative_weights, 0.9) + 1
ax.text(0.05, 0.95, f'{n_50} samples = 50% weight\n{n_90} samples = 90% weight',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 칸 4: 무게로 색칠한 표본
ax = axes[1, 1]
scatter = ax.scatter(samples, h_function(samples), c=norm_weights, 
                     cmap='hot', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Sample θ', fontsize=12)
ax.set_ylabel('h(θ) = θ²', fontsize=12)
ax.set_title('Samples Colored by Weight', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Normalized Weight')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example1_self_normalized.png")
plt.savefig(fig_path, 
            dpi=300, bbox_inches='tight')
print("\nVisualization saved to: example1_self_normalized.png")


# 보기 2: 베이즈 추론 - 흩어짐을 모르는 정규 평균
# ================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Bayesian Inference for Normal Mean")
print("=" * 70)

# 자료: σ = 1이 알려진 y ~ N(θ, σ²)
# 앞확률: θ ~ N(μ₀, τ²)
# 뒤확률: θ|y ~ N(μₙ, τₙ²), 여기서
#   τₙ² = 1/(1/τ² + n/σ²)
#   μₙ = τₙ²(μ₀/τ² + Σyᵢ/σ²)

# 합성 데이터 생성
true_theta = 5.0
sigma = 1.0
n_obs = 20
data = np.random.normal(true_theta, sigma, n_obs)

print(f"\nData: n={n_obs}, sample mean={np.mean(data):.3f}")

# 앞확률 매개변수
mu_0 = 0.0
tau = 2.0

# 뒤확률 매개변수(손으로 구함, 확인용)
tau_n_sq = 1.0 / (1.0/tau**2 + n_obs/sigma**2)
mu_n = tau_n_sq * (mu_0/tau**2 + np.sum(data)/sigma**2)

posterior_dist = stats.norm(mu_n, np.sqrt(tau_n_sq))

print(f"\nPosterior (analytical): N({mu_n:.3f}, {np.sqrt(tau_n_sq):.3f})")

# 고르게 하지 않은 뒤확률 정하기: γ(θ) = p(y|θ)p(θ)
def unnormalized_posterior(theta):
    """
    γ(θ) = p(y|θ)p(θ)
         = ∏ᵢ N(yᵢ|θ,σ²) × N(θ|μ₀,τ²)
         ∝ exp(-Σ(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ²)
    """
    # 로그 가능도: log p(y|θ)
    log_likelihood = -0.5 * np.sum((data[:, None] - theta)**2) / sigma**2
    
    # 로그 앞확률: log p(θ)
    log_prior = -0.5 * (theta - mu_0)**2 / tau**2
    
    # 고르게 하지 않은 뒤확률 돌려주기(수치 안정을 위해 로그 공간에서)
    return np.exp(log_likelihood + log_prior)

# 앞확률을 제안으로 쓰기(단순한 고름)
proposal_prior = stats.norm(mu_0, tau)

# 뒤확률 평균 어림하기: E[θ|y]
h_identity = lambda theta: theta
n_samples = 5000

estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
    unnormalized_posterior, proposal_prior, h_identity, n_samples
)

ess = compute_ess(norm_weights)

# 참 뒤확률 평균
true_post_mean = mu_n

print(f"\nPosterior Mean E[θ|y]:")
print(f"  True value: {true_post_mean:.6f}")
print(f"  SNIS estimate: {estimate:.6f}")
print(f"  Error: {abs(estimate - true_post_mean):.6f}")
print(f"  ESS: {ess:.1f} ({ess/n_samples*100:.1f}%)")

# 뒤확률 흩어짐 어림하기: Var[θ|y]
h_centered_square = lambda theta: (theta - estimate)**2
var_estimate, _, _, _ = self_normalized_importance_sampling(
    unnormalized_posterior, proposal_prior, h_centered_square, n_samples
)

true_post_var = tau_n_sq

print(f"\nPosterior Variance Var[θ|y]:")
print(f"  True value: {true_post_var:.6f}")
print(f"  SNIS estimate: {var_estimate:.6f}")
print(f"  Error: {abs(var_estimate - true_post_var):.6f}")


# 보기 3: 제안 분포 견주기
# =========================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Effect of Proposal Choice on ESS")
print("=" * 70)

# 보기 1과 같은 차림
proposals = {
    'Prior N(0,2)': stats.norm(0, 2),
    'Close to posterior N(5,1.5)': stats.norm(5, 1.5),
    'Posterior (oracle) N(μₙ,τₙ)': posterior_dist,
    'Too narrow N(5,0.5)': stats.norm(5, 0.5),
    'Too wide N(0,4)': stats.norm(0, 4),
}

n_samples = 2000
print(f"\nComparing proposals (n={n_samples}):")
print("-" * 70)

results = []
for name, proposal in proposals.items():
    estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
        unnormalized_posterior, proposal, h_identity, n_samples
    )
    ess = compute_ess(norm_weights)
    efficiency = ess / n_samples * 100
    error = abs(estimate - true_post_mean)
    
    results.append({
        'name': name,
        'estimate': estimate,
        'ess': ess,
        'efficiency': efficiency,
        'error': error
    })
    
    print(f"{name:30s}: ESS={ess:6.1f} ({efficiency:5.1f}%), Error={error:.4f}")

# ESS 견줌 그려 보기
fig, ax = plt.subplots(figsize=(12, 6))
names = [r['name'] for r in results]
efficiencies = [r['efficiency'] for r in results]
colors = ['blue' if 'oracle' not in n.lower() else 'red' for n in names]

bars = ax.bar(range(len(names)), efficiencies, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=12)
ax.set_title('Proposal Efficiency Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha='right')
ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, 
           label='Perfect efficiency')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example3_proposal_comparison.png")
plt.savefig(fig_path,
            dpi=300, bbox_inches='tight')


# 보기 4: 치우침과 표본 크기
# ==============================
print("\n" + "=" * 70)
print("EXAMPLE 4: Bias of Self-Normalized IS")
print("=" * 70)

# 스스로 고르게 하는 중요도 표집은 치우쳤지만 한결같다
# 치우침 = O(1/n)

sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
n_replications = 200

print(f"\nInvestigating bias ({n_replications} replications):")
print("-" * 70)

biases = []
std_errors = []

for n in sample_sizes:
    estimates = []
    for _ in range(n_replications):
        est, _, _, _ = self_normalized_importance_sampling(
            unnormalized_posterior, proposal_prior, h_identity, n
        )
        estimates.append(est)
    
    mean_estimate = np.mean(estimates)
    bias = mean_estimate - true_post_mean
    std_error = np.std(estimates)
    
    biases.append(bias)
    std_errors.append(std_error)
    
    print(f"n={n:5d}: Bias={bias:+.6f}, Std Error={std_error:.6f}")

# 표본 크기에 따른 치우침 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(sample_sizes, biases, 'bo-', linewidth=2, markersize=8, label='Observed bias')
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Bias', fontsize=12)
ax.set_title('Bias vs Sample Size (Self-Normalized IS)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(sample_sizes, std_errors, 'go-', linewidth=2, markersize=8, 
        label='Standard error')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Standard Error', fontsize=12)
ax.set_title('Standard Error vs Sample Size', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
# 기준선 더하기: 표준 오차 ~ 1/√n
ax.plot(sample_sizes, 0.5/np.sqrt(sample_sizes), 'r--', linewidth=2, 
        label='O(1/√n) reference')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example4_bias_analysis.png")
plt.savefig(fig_path,
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. 스스로 고르게 하는 중요도 표집은 고르게 하지 않은 과녁 분포를 다루며,
   그래서 p(y)을 모르는 베이즈 추론에 딱 맞다.

2. 스스로 고르게 한 어림꼴은 다음과 같다:
   wᵢ = w̃ᵢ/Σⱼw̃ⱼ일 때 Ê[h(θ)] = Σᵢ wᵢh(θᵢ)

3. 성질:
   - 치우쳤지만 한결같다(n → ∞이면 치우침 → 0)
   - 치우침 = O(1/n)
   - 흔히 고르게 한 중요도 표집보다 흩어짐이 작다

4. 실효 표본 크기(ESS)는 제안의 질을 잰다:
   - ESS = 1/Σᵢwᵢ²
   - ESS ≈ n: 아주 좋은 제안
   - ESS << n: 나쁜 제안, 몇몇 표본이 판친다
   - 효율 = ESS/n × 100%

5. 베이즈 추론에서:
   - 고르게 하지 않은 뒤확률: γ(θ) = p(y|θ)p(θ)
   - 앞확률은 단순한 제안 선택이 된다
   - 더 나은 제안(이를테면 라플라스 어림)이 ESS을 높인다

6. 좋은 제안이 결정적이다:
   - 뒤확률과 잘 겹쳐야 한다
   - 뒤확률보다 꼬리가 두꺼워야 한다
   - 주고받음: 셈 값과 ESS 나아짐

7. 무게 진단이 꼭 필요하다:
   - ESS 살피기
   - 무게 분포 살펴보기
   - 판치는 표본 살펴보기(무게 몰림)
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
스스로 고르게 하는 중요도 표집 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_self normalized is():
        model = Self Normalized IS(...)
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

**다룬 것** — 스스로 고르게 하는 중요도 표집

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
