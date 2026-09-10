# 스스로 고르게 하는 중요도 표집

스스로 고르게 하는 중요도 표집(SNIS)은 베이즈 추론의 일꾼 방법이다. 베이즈 추론에서는 뒤확률 분포가 고르게 하는 상수까지만 알려져 있다. 표준 중요도 표집과 달리 SNIS은 같은 중요도 표본으로 분자와 분모를 함께 어림해 고르게 하지 않은 과녁 밀도를 다룬다.

---

## 1. 고르게 하지 않은 과녁 문제

### 베이즈 뒤확률의 짜임

베이즈 추론에서 뒤확률 분포는 다음과 같다:

$$
\pi(\theta) = p(\theta|y) = \frac{p(y|\theta) p(\theta)}{p(y)}
$$

여기서 각 기호는 다음과 같다.

- $p(y|\theta)$: 가능도(안다)
- $p(\theta)$: 앞확률(안다)
- $p(y) = \int p(y|\theta) p(\theta) d\theta$: 주변 가능도(보통 다룰 수 없다)

우리는 **고르게 하지 않은 뒤확률**의 값을 매길 수 있다:

$$
\gamma(\theta) = p(y|\theta) p(\theta)
$$

그러나 고르게 하는 상수 $Z = p(y)$은 그럴 수 없다.

### 일반적인 판 벌이기

더 넓게 보면 다음과 같다:

$$
\pi(\theta) = \frac{\gamma(\theta)}{Z}, \quad Z = \int \gamma(\theta) d\theta \text{ (unknown)}
$$

표준 중요도 표집은 $\pi(\theta)$의 값을 정확히 매겨야 하는데, $Z$을 모르면 그럴 수 없다.

---

## 2. 스스로 고르게 하는 어림자

### 유도

기댓값을 비로 적는다:

$$
I = \mathbb{E}_\pi[h(\theta)] = \int h(\theta) \pi(\theta) d\theta = \frac{\int h(\theta) \gamma(\theta) d\theta}{\int \gamma(\theta) d\theta}
$$

두 적분 모두 고르게 하지 않은 무게로 중요도 표집해 어림할 수 있다:

$$
\tilde{w}(\theta) = \frac{\gamma(\theta)}{q(\theta)}
$$

**분자 어림값:**

$$
\widehat{\text{Num}} = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \tilde{w}(\theta_i)
$$

**분모 어림값:**

$$
\widehat{\text{Den}} = \frac{1}{n} \sum_{i=1}^n \tilde{w}(\theta_i)
$$

### SNIS 어림자

$$
\boxed{\hat{I}_{\text{SNIS}} = \frac{\sum_{i=1}^n h(\theta_i) \tilde{w}_i}{\sum_{i=1}^n \tilde{w}_i} = \sum_{i=1}^n h(\theta_i) \bar{w}_i}
$$

여기서 **고르게 한 무게**는 다음과 같다:

$$
\bar{w}_i = \frac{\tilde{w}_i}{\sum_{j=1}^n \tilde{w}_j}, \quad \sum_{i=1}^n \bar{w}_i = 1
$$

이는 무게의 합이 1인 $h(\theta_i)$의 **무게 준 평균**이다.

---

## 3. SNIS의 성질

### 치우침

표준 중요도 표집과 달리 SNIS은 **치우쳐 있다**:

$$
\mathbb{E}[\hat{I}_{\text{SNIS}}] \neq I \quad \text{for finite } n
$$

기댓값의 비가 비의 기댓값과 같지 않기 때문이다:

$$
\mathbb{E}\left[\frac{\hat{\text{Num}}}{\hat{\text{Den}}}\right] \neq \frac{\mathbb{E}[\hat{\text{Num}}]}{\mathbb{E}[\hat{\text{Den}}]}
$$

### 일치성

치우쳐 있어도 SNIS은 **일관적**이다:

$$
\hat{I}_{\text{SNIS}} \xrightarrow{a.s.} I \quad \text{as } n \to \infty
$$

이는 다음에서 따라 나온다:

$$
\frac{1}{n} \sum_{i=1}^n \tilde{w}_i \xrightarrow{a.s.} \mathbb{E}_q\left[\frac{\gamma(\theta)}{q(\theta)}\right] = \int \gamma(\theta) d\theta = Z
$$

### 치우침의 성격

참값 둘레에서 테일러로 펼치면 다음과 같다:

$$
\text{Bias}(\hat{I}_{\text{SNIS}}) = O(1/n)
$$

치우침은 $1/n$의 속도로 사라지며, 이는 $O(1/\sqrt{n})$인 표준 오차보다 빠르다.

### 흩어짐(어림)

$n$이 크면 흩어짐은 대략 다음과 같다:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{1}{n} \text{Var}_\pi\left[(h(\theta) - I) \cdot \frac{\pi(\theta)}{q(\theta)}\right]
$$

---

## 4. 견줌: 표준 중요도 표집과 SNIS

| 성질 | 표준 중요도 표집 | 스스로 고르게 하는 중요도 표집 |
|----------|-------------|-------------------|
| 고르게 한 $\pi$이 필요한가 | 예 | 아니오 |
| 치우침 | 치우침 없음 | 치우침 있음, $O(1/n)$ |
| 일관성 | 예 | 예 |
| 무게의 합 | 꼭 1은 아님 | 정확히 1 |
| 쓰임새 | 고르게 하는 상수를 아는 경우 | 베이즈 뒤확률 |
| 흩어짐 | 흔히 더 큼 | 흔히 더 작음 |

!!! info "SNIS이 흔히 흩어짐이 더 작다"
    놀랍게도 고르게 하는 상수를 알 때조차 SNIS이 표준 중요도 표집보다 흩어짐이 **더 작은** 경우가 많다. 고르게 하기가 어림자를 안정시킨다.

---

## 5. PyTorch 구현

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def self_normalized_importance_sampling(h_function, unnormalized_log_target, 
                                        proposal_dist, n_samples, 
                                        return_diagnostics=False):
    """
    고르게 하지 않은 과녁을 위한, 스스로 고르게 하는 중요도 표집.
    
    매개변수
    ----------
    h_function : callable
        기댓값을 구하려는 함수 h(θ)
    unnormalized_log_target : callable
        고르게 하지 않은 과녁의 로그: π(θ) = γ(θ)/Z일 때 log γ(θ)
    proposal_dist : torch.distributions.Distribution
        제안 분포 q(θ)
    n_samples : int
        표본의 개수
    return_diagnostics : bool
        진단 정보를 돌려줄지 여부
        
    반환값
    -------
    estimate : torch.Tensor
        E_π[h(θ)]의 SNIS 어림값
    diagnostics : dict, optional
        표본, 무게, ESS, 그 밖의 진단
    
    수학의 바탕
    -----------------------
    Î_SNIS = Σᵢ h(θᵢ) w̄ᵢ
    
    여기서 각 기호는 다음과 같다.
    - w̃ᵢ = γ(θᵢ)/q(θᵢ)(고르게 하지 않은 무게)
    - w̄ᵢ = w̃ᵢ / Σⱼw̃ⱼ(고르게 한 무게, 합이 1)
    - θᵢ ~ q(θ)
    """
    # 걸음 1: 제안에서 표집
    samples = proposal_dist.sample((n_samples,))
    
    # 걸음 2: 고르게 하지 않은 로그 무게 셈하기
    # log w̃(θ) = log γ(θ) - log q(θ)
    log_gamma = unnormalized_log_target(samples)
    log_q = proposal_dist.log_prob(samples)
    log_unnorm_weights = log_gamma - log_q
    
    # 걸음 3: 무게 고르게 하기(수치 안정을 위해 log-sum-exp 사용)
    log_sum_weights = torch.logsumexp(log_unnorm_weights, dim=0)
    log_norm_weights = log_unnorm_weights - log_sum_weights
    norm_weights = torch.exp(log_norm_weights)
    
    # 걸음 4: 함수 값 매기기
    h_values = h_function(samples)
    
    # 걸음 5: SNIS 어림값 셈하기: Σᵢ w̄ᵢ h(θᵢ)
    estimate = torch.sum(norm_weights * h_values)
    
    if return_diagnostics:
        # 실효 표본 크기: ESS = 1/Σᵢw̄ᵢ²
        ess = 1.0 / torch.sum(norm_weights**2)
        
        # 살피기 위한, 고르게 하지 않은 무게
        unnorm_weights = torch.exp(log_unnorm_weights)
        
        diagnostics = {
            'samples': samples,
            'unnorm_weights': unnorm_weights,
            'norm_weights': norm_weights,
            'h_values': h_values,
            'log_unnorm_weights': log_unnorm_weights,
            'ess': ess,
            'ess_ratio': ess / n_samples,
            'n_samples': n_samples,
            'estimate_normalizing_constant': torch.exp(log_sum_weights) / n_samples
        }
        return estimate, diagnostics
    
    return estimate

def compute_ess(weights):
    """
    고르게 한 무게로 실효 표본 크기 셈하기.
    
    ESS = 1 / Σᵢ wᵢ²
    
    해석:
    - ESS ≈ n: 무게가 거의 고르다(아주 좋음)
    - ESS << n: 몇몇 표본이 판친다(나쁨)
    """
    return 1.0 / torch.sum(weights**2)

def compute_ess_unnormalized(unnorm_weights):
    """
    고르게 하지 않은 무게로 ESS 셈하기.
    
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²
    """
    sum_w = torch.sum(unnorm_weights)
    sum_w_sq = torch.sum(unnorm_weights**2)
    return sum_w**2 / sum_w_sq

# 보기: 정규 평균의 베이즈 추론
# 앞확률: θ ~ N(μ₀, τ₀²)
# 가능도: 관측마다 y ~ N(θ, σ²)
# 뒤확률: θ|y ~ N(μₙ, τₙ²)

# 합성 데이터 생성
torch.manual_seed(42)
true_theta = 5.0
sigma = 1.0  # 알려진 관측 잡음
n_obs = 20
data = torch.normal(true_theta, sigma, size=(n_obs,))

print(f"Data: n={n_obs}, sample mean={data.mean().item():.3f}")

# 앞확률 매개변수
mu_0 = 0.0
tau_0 = 2.0

# 손으로 구한 뒤확률의 매개변수
precision_0 = 1.0 / tau_0**2
precision_n = precision_0 + n_obs / sigma**2
tau_n = 1.0 / (precision_n**0.5)
mu_n = (precision_0 * mu_0 + n_obs * data.mean() / sigma**2) / precision_n

print(f"\nPrior: N({mu_0}, {tau_0}²)")
print(f"Posterior (analytical): N({mu_n:.4f}, {tau_n:.4f}²)")

# 고르게 하지 않은 로그 뒤확률 정하기
def unnormalized_log_posterior(theta):
    """
    log γ(θ) = log p(y|θ) + log p(θ)
             = -Σ(yᵢ-θ)²/2σ² - (θ-μ₀)²/2τ₀²
    """
    # 로그 가능도: -Σ(yᵢ-θ)²/2σ²
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
    
    # 꼴: (n_samples,) 또는 (n_samples, 1)
    theta_expanded = theta.unsqueeze(-1) if theta.dim() == 1 else theta
    data_expanded = data.unsqueeze(0)
    
    log_likelihood = -0.5 * torch.sum((data_expanded - theta_expanded)**2, dim=-1) / sigma**2
    
    # 로그 앞확률: -(θ-μ₀)²/2τ₀²
    log_prior = -0.5 * (theta - mu_0)**2 / tau_0**2
    
    return log_likelihood + log_prior

# 앞확률을 제안으로 쓰기
proposal = dist.Normal(mu_0, tau_0)

# SNIS 어림하기
n_samples = 5000

estimate, diagnostics = self_normalized_importance_sampling(
    h_function=lambda x: x,  # 뒤확률 평균
    unnormalized_log_target=unnormalized_log_posterior,
    proposal_dist=proposal,
    n_samples=n_samples,
    return_diagnostics=True
)

print(f"\n{'='*60}")
print("Self-Normalized IS Results")
print(f"{'='*60}")
print(f"Posterior mean E[θ|y]:")
print(f"  True: {mu_n:.6f}")
print(f"  SNIS: {estimate.item():.6f}")
print(f"  Error: {abs(estimate.item() - mu_n):.6f}")
print(f"\nEffective Sample Size:")
print(f"  ESS: {diagnostics['ess'].item():.1f}")
print(f"  Efficiency: {diagnostics['ess_ratio'].item():.1%}")

# 뒤확률 흩어짐 어림하기
var_estimate, _ = self_normalized_importance_sampling(
    h_function=lambda x: (x - estimate.item())**2,
    unnormalized_log_target=unnormalized_log_posterior,
    proposal_dist=proposal,
    n_samples=n_samples
)

print(f"\nPosterior variance Var[θ|y]:")
print(f"  True: {tau_n**2:.6f}")
print(f"  SNIS: {var_estimate.item():.6f}")

# 고르게 하는 상수 어림하기(주변 가능도)
print(f"\nMarginal likelihood estimate: {diagnostics['estimate_normalizing_constant'].item():.4e}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 칸 1: 앞확률, 제안, 뒤확률
x = torch.linspace(-5, 10, 500)
ax = axes[0, 0]

prior = dist.Normal(mu_0, tau_0)
posterior = dist.Normal(mu_n, tau_n)

ax.plot(x.numpy(), prior.log_prob(x).exp().numpy(), 
        'g--', linewidth=2, label='Prior', alpha=0.7)
ax.plot(x.numpy(), posterior.log_prob(x).exp().numpy(), 
        'b-', linewidth=2, label='Posterior (true)')

# 표본의 무게 막대그림
samples = diagnostics['samples']
weights = diagnostics['norm_weights']
ax.hist(samples.numpy(), bins=50, density=True, weights=weights.numpy(),
        alpha=0.5, color='red', label='SNIS approximation')

ax.axvline(true_theta, color='black', linestyle=':', linewidth=2, 
           label=f'True θ = {true_theta}')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Bayesian Posterior Estimation via SNIS', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 칸 2: 무게 분포
ax = axes[0, 1]
ax.hist(weights.numpy() * n_samples, bins=50, density=True, 
        alpha=0.7, color='purple', edgecolor='black')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, 
           label='Uniform weight')
ax.set_xlabel('Normalized Weight × n', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Weight Distribution (ESS = {diagnostics["ess"].item():.1f}, '
             f'{diagnostics["ess_ratio"].item():.1%})', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 칸 3: 쌓인 무게
sorted_weights = torch.sort(weights, descending=True)[0]
cumsum_weights = torch.cumsum(sorted_weights, dim=0)

ax = axes[1, 0]
ax.plot(torch.arange(1, n_samples+1).numpy(), cumsum_weights.numpy(), 
        'b-', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='50% of weight')
ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label='90% of weight')
ax.set_xlabel('Number of Samples (sorted by weight)', fontsize=12)
ax.set_ylabel('Cumulative Weight', fontsize=12)
ax.set_title('Weight Concentration', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 몰림 찾기
n_50 = (cumsum_weights < 0.5).sum().item() + 1
n_90 = (cumsum_weights < 0.9).sum().item() + 1
ax.text(0.95, 0.05, f'{n_50} samples = 50% weight\n{n_90} samples = 90% weight',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 칸 4: 치우침 살피기
sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
n_reps = 200

biases = []
std_devs = []

for n in sample_sizes:
    estimates = []
    for _ in range(n_reps):
        est, _ = self_normalized_importance_sampling(
            h_function=lambda x: x,
            unnormalized_log_target=unnormalized_log_posterior,
            proposal_dist=proposal,
            n_samples=n
        )
        estimates.append(est.item())
    
    estimates = torch.tensor(estimates)
    biases.append((estimates.mean() - mu_n).item())
    std_devs.append(estimates.std().item())

ax = axes[1, 1]
ax.plot(sample_sizes, biases, 'bo-', linewidth=2, markersize=8, 
        label='Bias')
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Bias', fontsize=12)
ax.set_title('SNIS Bias Decreases as O(1/n)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('self_normalized_is.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. 실효 표본 크기(ESS)

### 정의

**실효 표본 크기**는 중요도 표본의 질을 잰다:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^n \bar{w}_i^2}
$$

마찬가지로 고르게 하지 않은 무게를 쓰면 다음과 같다:

$$
\text{ESS} = \frac{\left(\sum_{i=1}^n \tilde{w}_i\right)^2}{\sum_{i=1}^n \tilde{w}_i^2}
$$

### 성질

| ESS 값 | 풀이 |
|-----------|----------------|
| ESS ≈ n | 아주 좋음: 무게가 거의 고름 |
| ESS ≈ n/2 | 좋음: 무게가 알맞게 들쭉날쭉함 |
| ESS ≈ n/10 | 넉넉함: 무게가 얼마쯤 몰림 |
| ESS << n/10 | 나쁨: 몇몇 표본이 좌우함 |
| ESS ≈ 1 | 주저앉음: 표본 하나가 좌우함 |

### 흩어짐과의 관계

대략 다음과 같다:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{\text{Var}_\pi(h(\theta))}{\text{ESS}}
$$

ESS이 낮으면 흩어짐이 크다는 뜻이다. 곧 실효 독립 표본의 개수가 적다.

이끌어 내기, 진단, 실전 길잡이를 아우르는 ESS의 두루 갖춘 다룸은 따로 마련한 [실효 표본 크기](ess.md) 마당을 보아라.

### 진단으로서의 ESS

```python
def diagnose_weights(norm_weights, name=""):
    """
    무게 두루 진단하기.
    """
    n = len(norm_weights)
    
    # ESS
    ess = 1.0 / torch.sum(norm_weights**2)
    
    # 변이 계수
    cv = torch.std(norm_weights) / torch.mean(norm_weights)
    
    # 최대 무게
    max_w = torch.max(norm_weights)
    
    # 무게 몰림
    sorted_w = torch.sort(norm_weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    # 엔트로피
    entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-10))
    max_entropy = torch.log(torch.tensor(float(n)))
    
    print(f"\nWeight Diagnostics {name}")
    print("-" * 50)
    print(f"  n samples: {n}")
    print(f"  ESS: {ess.item():.1f} ({ess.item()/n:.1%} efficiency)")
    print(f"  CV of weights: {cv.item():.3f}")
    print(f"  Max weight: {max_w.item():.6f} (uniform = {1/n:.6f})")
    print(f"  {n_for_50} samples ({n_for_50/n:.1%}) account for 50% of weight")
    print(f"  {n_for_90} samples ({n_for_90/n:.1%}) account for 90% of weight")
    print(f"  Normalized entropy: {entropy.item()/max_entropy.item():.3f}")
    
    return {
        'ess': ess.item(),
        'cv': cv.item(),
        'max_weight': max_w.item(),
        'n_for_50': n_for_50,
        'n_for_90': n_for_90
    }

# 진단 돌리기
diagnostics = diagnose_weights(diagnostics['norm_weights'], "(Prior as Proposal)")
```

---

## 7. 제안 분포 견주기

### 제안 고르기의 영향

```python
# 여러 제안 견주기
proposals = {
    'Prior N(0, 2)': dist.Normal(0.0, 2.0),
    'Close N(μₙ, 1)': dist.Normal(mu_n, 1.0),
    'Posterior (oracle)': dist.Normal(mu_n, tau_n),
    'Too narrow N(μₙ, 0.3)': dist.Normal(mu_n, 0.3),
    'Too wide N(0, 5)': dist.Normal(0.0, 5.0),
}

print("\nProposal Comparison")
print("=" * 70)
print(f"{'Proposal':<25} {'Estimate':>12} {'Error':>10} {'ESS':>10} {'Efficiency':>10}")
print("-" * 70)

for name, proposal in proposals.items():
    estimate, diag = self_normalized_importance_sampling(
        h_function=lambda x: x,
        unnormalized_log_target=unnormalized_log_posterior,
        proposal_dist=proposal,
        n_samples=5000,
        return_diagnostics=True
    )
    
    print(f"{name:<25} {estimate.item():12.4f} {abs(estimate.item()-mu_n):10.4f} "
          f"{diag['ess'].item():10.1f} {diag['ess_ratio'].item():10.1%}")
```

제안 고르기 전략을 체계로 다룬 것은 [제안 분포 설계](proposal_design.md)를 보아라.

---

## 8. 여러 기댓값 셈하기

### 표본 다시 쓰기

중요도 표집의 핵심 이점은 같은 무게 준 표본으로 여러 기댓값을 어림할 수 있다는 것이다:

```python
def compute_multiple_expectations(samples, weights, functions):
    """
    같은 표본으로 기댓값 여럿 어림하기.
    
    각 k에 대해 E_π[hₖ(θ)] ≈ Σᵢ w̄ᵢ hₖ(θᵢ)
    """
    results = {}
    for name, h in functions.items():
        h_values = h(samples)
        estimate = torch.sum(weights * h_values)
        results[name] = estimate.item()
    return results

# 관심 있는 함수 여럿 정하기
functions = {
    'E[θ]': lambda x: x,
    'E[θ²]': lambda x: x**2,
    'E[θ³]': lambda x: x**3,
    'Var[θ]': lambda x: (x - estimate.item())**2,  # 뒤확률 평균에 가운데를 맞춤
    'P(θ > 3)': lambda x: (x > 3).float(),
}

results = compute_multiple_expectations(
    diagnostics['samples'], 
    diagnostics['norm_weights'],
    functions
)

print("\nMultiple Expectations from Same Samples")
print("-" * 40)
for name, value in results.items():
    print(f"  {name}: {value:.6f}")
```

---

## 9. 뒤확률 예측 분포

### 정의

뒤확률 예측 분포는 다음과 같다:

$$
p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta
$$

### SNIS 어림

```python
def posterior_predictive_pmf(y_values, samples, weights, likelihood):
    """
    띄엄띄엄한 결과의 뒤확률 미리봄 P(ỹ|y) 어림하기.
    """
    probs = []
    for y in y_values:
        # P(ỹ|y) ≈ Σᵢ w̄ᵢ P(ỹ|θᵢ)
        p_y_given_theta = likelihood(y, samples)
        prob = torch.sum(weights * p_y_given_theta)
        probs.append(prob.item())
    return probs
```

---

## 10. 계량 금융에서의 쓰임새

### 자산 수익률의 베이즈 매개변수 어림

SNIS은 금융의 베이즈 어림 문제에 특히 자연스럽다. 그런 문제에서는 모형 매개변수의 뒤확률을 닫힌 꼴로 얻는 일이 드물다. 흔한 상황은 가능도는 다룰 수 있지만 고르게 하는 상수는 그럴 수 없는 확률 변동성 모형이나 국면 전환 모형의 매개변수를 어림하는 것이다.

```python
def bayesian_return_model_snis(returns, proposal_dist, n_samples=10000):
    """
    수익 분포 매개변수의 베이즈 어림을 위한 SNIS.
    
    모형: r_t ~ N(μ, σ²)
    앞확률: μ ~ N(0, 0.1²), log σ ~ N(log(0.02), 0.5²)
    
    매개변수
    ----------
    returns : torch.Tensor
        관측한 날마다의 로그 수익
    proposal_dist : callable
        (μ, log_σ)에 대한 결합 제안
    n_samples : int
        중요도 표집 표본의 개수
    """
    n_obs = len(returns)
    
    def unnormalized_log_posterior(params):
        """params: 열이 [μ, log_σ]인 (n_samples, 2)"""
        mu = params[:, 0]
        log_sigma = params[:, 1]
        sigma = torch.exp(log_sigma)
        
        # 로그 가능도: Σ log N(r_t; μ, σ²)
        log_lik = -0.5 * n_obs * torch.log(2 * torch.pi * sigma**2)
        log_lik -= 0.5 * torch.sum((returns.unsqueeze(0) - mu.unsqueeze(1))**2, dim=1) / sigma**2
        
        # 로그 앞확률
        log_prior_mu = dist.Normal(0.0, 0.1).log_prob(mu)
        log_prior_logsig = dist.Normal(torch.log(torch.tensor(0.02)), 0.5).log_prob(log_sigma)
        
        return log_lik + log_prior_mu + log_prior_logsig
    
    # 제안에서 표집
    samples = proposal_dist.sample((n_samples,))
    
    # SNIS 무게 셈하기
    log_gamma = unnormalized_log_posterior(samples)
    log_q = proposal_dist.log_prob(samples).sum(dim=-1)  # 독립인 성분
    log_weights = log_gamma - log_q
    
    # 정규화
    log_sum = torch.logsumexp(log_weights, dim=0)
    norm_weights = torch.exp(log_weights - log_sum)
    
    # 뒤확률의 적률
    post_mu = torch.sum(norm_weights * samples[:, 0])
    post_sigma = torch.sum(norm_weights * torch.exp(samples[:, 1]))
    
    # ESS
    ess = 1.0 / torch.sum(norm_weights**2)
    
    return {
        'posterior_mu': post_mu.item(),
        'posterior_sigma': post_sigma.item(),
        'ess': ess.item(),
        'ess_ratio': (ess / n_samples).item()
    }
```

### 모형 고르기를 위한 주변 가능도

SNIS은 고르게 하는 상수 $Z = p(y)$의 어림값을 자연스럽게 준다. 이것이 베이즈 모형 견줌에 쓰는 주변 가능도이다. 겨루는 인자 모형이나 변동성 설정을 견주는 데 값지다:

$$
\hat{Z} = \frac{1}{n} \sum_{i=1}^n \tilde{w}_i = \frac{1}{n} \sum_{i=1}^n \frac{\gamma(\theta_i)}{q(\theta_i)}
$$

그러면 모형 $\mathcal{M}_1$과 $\mathcal{M}_2$ 사이의 로그 베이즈 인자는 다음과 같다:

$$
\log B_{12} = \log \hat{Z}_1 - \log \hat{Z}_2
$$

---

## 11. 핵심 정리

!!! success "SNIS을 언제 쓰나"

    - 고르게 하는 상수를 모르는 뒤확률 추론
    - 같은 표본에서 여러 기댓값 구하기
    - 뒤확률 예측 셈하기
    - 잇단 중요도 표집 알고리즘

!!! warning "치우침에서 살필 점"

    - $n$이 끝이 있으면 SNIS은 치우쳐 있다
    - 치우침 = $O(1/n)$으로 표준 오차보다 빨리 줄어든다
    - 실전에서는 치우침이 보통 하찮다
    - 표본의 질이 넉넉한지 늘 ESS으로 지켜보아라

!!! info "ESS 길잡이"

    - ESS > 1000: 대부분의 쓰임새에 보통 넉넉하다
    - ESS/n > 0.1: 넉넉한 효율
    - ESS/n < 0.01: 제안을 낫게 할 것을 생각해 보아라
    - 어림값과 함께 늘 ESS을 알려라

---

## 연습문제

### 연습 1: 치우침 확인하기
SNIS의 치우침이 $O(1/n)$임을 겪어 보고 확인하여라. 표본 크기 $n = 100, 200, 400, 800, 1600$으로 여러 번 되풀이하여라. 로그-로그 눈금에서 치우침을 $n$에 대해 그리고 기울기가 $\approx -1$임을 확인하여라.

### 연습 2: ESS 풀이하기
제안을 붙박아 두고 크기 $n = 1000, 2000, 5000, 10000$의 표본을 뽑아라. ESS/n이 대략 상수로 남는가? 왜 그런지 또는 왜 그렇지 않은지 설명하여라.

### 연습 3: 뒤확률 예측
정규-정규 모형의 뒤확률 예측 어림을 구현하여라. SNIS 어림값을 해석으로 구한 결과와 견주어라.

### 연습 4: 여러 제안
앞확률을 제안으로 쓸 때와 라플라스 어림(최빈값을 가운데로 하는 가우스)을 제안으로 쓸 때의 효율을 견주어라. 어느 것이 ESS이 더 높은가?

### 연습 5: 주변 가능도 어림하기
앞확률을 제안으로 쓴 SNIS으로 베이즈 선형 회귀 모형의 주변 가능도를 어림하여라. 켤레 정규-역감마 모형에서 해석으로 얻는 결과와 중요도 표집 어림값을 견주어라. ESS은 $\hat{Z}$ 어림값의 정확도와 어떻게 이어지는가?

## 정리하며

이 마당은 고르게 하지 않은 과녁 문제、스스로 고르게 하는 어림자、SNIS의 성질、견줌: 표준 중요도 표집과 SNIS을 차례로 짚었다.

**참고 문헌**

1. Geweke, J. (1989). "Bayesian inference in econometric models using Monte Carlo integration." *Econometrica*, 57(6), 1317-1339.

2. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 9.4절: 스스로 고르게 하는 중요도 표집.

3. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. 3.3절.

4. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

5. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.
