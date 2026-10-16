# 자기정규화 중요도 샘플링

자기정규화 중요도 샘플링(SNIS)은 미정규화 목표밀도의 가중 합을 그 웨이트 합으로 나누어 기댓값을 추정한다. 알 수 없는 곱셈 상수는 비에서 약분된다. 대신 일반적으로 유한 표본 바이어스가 생기며, 일관성·분산·증거 추정의 조건을 따로 확인해야 한다.

## 1. 미정규화 목표밀도

목표분포가 $\pi(\theta)=\gamma(\theta)/Z$, $Z=\int\gamma(\theta)d\theta$이고 0<Z<∞라고 하자. 베이즈 추론에서는 γ가 우도와 사전밀도의 곱일 수 있다.

π를 직접 평가할 수 없어도 γ의 로그값을 상수 차이까지 계산할 수 있다면 SNIS의 상대 웨이트를 구할 수 있다. 다만 모델 증거 Z의 절대값이 필요하면 우도·사전밀도의 모든 정규화 상수를 알아야 한다. 기댓값 계산에서 약분되는 상수가 모델 비교에서는 약분되지 않을 수 있다.

## 2. 정규화 가중 평균

독립 표본 θ_i∼q에서 비정규화 웨이트를 $W_i=\gamma(\theta_i)/q(\theta_i)$라 두면 다음과 같다.

$$\hat I=\frac{\frac1n\sum_iW_i h(\theta_i)}{\frac1n\sum_iW_i}=\sum_i\bar w_i h(\theta_i),\qquad\bar w_i=\frac{W_i}{\sum_jW_j}$$

q는 γ의 지지집합을 덮어야 하고 웨이트 합은 양수·유한해야 한다. log W_i에서 log-sum-exp를 빼면 큰 공통 상수에도 안정적으로 정규화할 수 있다. 모든 로그 웨이트가 −∞이거나 NaN인 경우에는 이 계산도 유효하지 않다. 표본은 q에서 나온 것이며 정규화만으로 목표분포의 독립 표본이 되지는 않는다.

## 3. SNIS의 성질

### 바이어스와 일관성

분자·분모가 각각 불편이어도 비는 일반적으로 불편하지 않다. 다만 h가 상수이거나 q=π인 경우처럼 편향이 없는 특별한 경우도 있으므로 모든 유한 n에서 반드시 E[Î]≠I라고 단정하지 않는다.

독립 표본과 E_π|h|<∞, 적절한 지지집합 아래 두 표본평균의 대수의 법칙에서 Î→I가 거의 확실하게 성립한다. 일반적인 O(1/n) 바이어스 전개에는 추가 모멘트와 분모 꼬리의 제어 등이 필요하다. 모든 일관적인 SNIS에서 같은 바이어스 속도나 무시할 수 있는 바이어스가 보장되는 것은 아니다.

### 점근 분산

적절한 중심극한정리·델타 방법 조건 아래 점근 분산은 다음과 같다.

$$\sigma_{SN}^2=\frac{E_q[W^2(h-I)^2]}{Z^2}=\int\frac{(h(\theta)-I)^2\pi(\theta)^2}{q(\theta)}d\theta$$

따라서 큰 n에서 Var(Î)≈σ_SN²/n이다. 원래 식처럼 π 아래에서 가중항의 분산을 구하는 것이 아니라 **q 아래**의 이차 적률을 사용한다. 이 적분의 유한성은 지지집합 조건과 별개다. 실용적인 점근 분산 추정은 ∑_i w̄_i²(h_i−Î)²를 사용할 수 있지만 희귀 큰 웨이트를 놓친 유한 표본에서는 과신할 수 있다.

## 4. 비교: 표준 중요도 샘플링과 SNIS

| 항목 | 표준 IS | SNIS |
|---|---|---|
| 필요한 목표값 | 정규화된 π | 미정규화 γ, 공통 곱셈 상수는 약분 |
| 유한 표본 바이어스 | 조건 아래 불편 | 일반적으로 바이어스, 특별한 무바이어스 경우도 있음 |
| 일관성·CLT | 각각 적절한 적률 필요 | 각각 적절한 적률·비율 조건 필요 |
| 추정 형태 | n⁻¹∑h_i π_i/q_i | 정규화 가중 평균 |
| 분산 순서 | 보편적인 우열 없음 | 보편적인 우열 없음 |

정규화가 특정 적분을 안정화할 수 있지만 언제나 분산을 줄이지는 않는다. 예를 들어 h≥0의 표준 IS 최적 제안은 hπ에 비례하고 분산 0을 낼 수 있으나 동일 제안의 SNIS는 일반적으로 그렇지 않다. SNIS의 점근 최적 제안은 |h−I|π에 비례하는 다른 형태다.

## 5. PyTorch 구현

핵심 함수는 표본마다 스칼라 log γ·log q·h값이 하나인 배열을 가정한다. 반환 진단의 원시 웨이트와 Z 추정값은 exp로 변환하므로 오버플로할 수 있다. 주 계산의 로그 정규화가 안정적이어도 진단 전체가 안정적인 것은 아니다.

정규 평균 예제의 unnormalized_log_posterior는 우도·사전의 정규화 상수를 생략한다. 사후평균 계산에는 무관하지만 출력 Marginal likelihood는 실제 p(y)가 아니라 생략 상수에 따른 적분이다. 사후분산은 별도 새 표본에서 이전 평균 주위 제곱을 추정하므로 평균 추정 오차가 섞인다.

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def self_normalized_importance_sampling(h_function, unnormalized_log_target, 
                                        proposal_dist, n_samples, 
                                        return_diagnostics=False):
    """
    고르게 하지 않은 타깃을 위한, 스스로 고르게 하는 중요도 표집.
    
    파라메터
    ----------
    h_function : callable
        기댓값을 구하려는 함수 h(θ)
    unnormalized_log_target : callable
        고르게 하지 않은 타깃의 로그: π(θ) = γ(θ)/Z일 때 log γ(θ)
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
    # 스텝 1: 제안에서 표집
    samples = proposal_dist.sample((n_samples,))
    
    # 스텝 2: 고르게 하지 않은 로그 무게 셈하기
    # log w̃(θ) = log γ(θ) - log q(θ)
    log_gamma = unnormalized_log_target(samples)
    log_q = proposal_dist.log_prob(samples)
    log_unnorm_weights = log_gamma - log_q
    
    # 스텝 3: 무게 고르게 하기(수치 안정을 위해 log-sum-exp 사용)
    log_sum_weights = torch.logsumexp(log_unnorm_weights, dim=0)
    log_norm_weights = log_unnorm_weights - log_sum_weights
    norm_weights = torch.exp(log_norm_weights)
    
    # 스텝 4: 함수 값 매기기
    h_values = h_function(samples)
    
    # 스텝 5: SNIS 어림값 셈하기: Σᵢ w̄ᵢ h(θᵢ)
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

# 앞확률 파라메터
mu_0 = 0.0
tau_0 = 2.0

# 손으로 구한 뒤확률의 파라메터
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
    n_samples=n_samples,
    return_diagnostics=True
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
            return_diagnostics=True,
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

**출력:**

```
Data: n=20, sample mean=5.216

Prior: N(0.0, 2.0²)
Posterior (analytical): N(5.1518, 0.2222²)

============================================================
Self-Normalized IS Results
============================================================
Posterior mean E[θ|y]:
  True: 5.151762
  SNIS: 5.234283
  Error: 0.082521

Effective Sample Size:
  ESS: 22.4
  Efficiency: 0.4%

Posterior variance Var[θ|y]:
  True: 0.049383
  SNIS: 0.048890

Marginal likelihood estimate: 1.4309e-06
```

---

## 6. 유효 표본 크기(ESS)

정규화 웨이트에 기반한 진단량은 다음과 같다.

$$ESS_w=\frac1{\sum_i\bar w_i^2}=\frac{(\sum_iW_i)^2}{\sum_iW_i^2}$$

1≤ESS_w≤n이며 균등 웨이트이면 n, 하나에 전부 몰리면 1이다. 이는 관측한 웨이트의 집중도이지 모든 h에 공통인 독립 표본 수가 아니다. Var_π(h)/ESS_w를 보편적인 SNIS 분산식으로 사용할 수 없다. 실제 점근 분산에는 W와 (h−I)의 결합이 들어간다. 높은 ESS도 누락된 모드·꼬리나 지지집합 문제를 배제하지 못한다. [ESS의 해석](ess.md)을 참고한다.

### 웨이트 진단 예제

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
# 표집기가 돌려준 diagnostics를 덮어쓰면 뒤에서 쓸 'samples' 같은 열쇠가
# 사라진다. 진단 결과는 따로 받는다.
weight_report = diagnose_weights(diagnostics['norm_weights'], "(Prior as Proposal)")
```

**출력:**

```
Weight Diagnostics (Prior as Proposal)
--------------------------------------------------
  n samples: 5000
  ESS: 22.4 (0.4% efficiency)
  CV of weights: 14.894
  Max weight: 0.064764 (uniform = 0.000200)
  9 samples (0.2%) account for 50% of weight
  22 samples (0.4%) account for 90% of weight
  Normalized entropy: 0.386
```

---

## 7. 제안 분포 비교

### 제안 고르기의 영향

Posterior (oracle)는 정확한 사후분포를 이미 아는 기준값이다. 예제에서 사후 표준편차는 약 0.222이므로 이름이 Too narrow인 표준편차 0.3 제안은 실제로 사후보다 넓다. 등급 이름만으로 판단하지 않는다.

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

**출력:**

```
Proposal Comparison
======================================================================
Proposal                      Estimate      Error        ESS Efficiency
----------------------------------------------------------------------
Prior N(0, 2)                   5.1653     0.0136       20.0       0.4%
Close N(μₙ, 1)                  5.1554     0.0037     1564.7      31.3%
Posterior (oracle)              5.1481     0.0036     5000.0     100.0%
Too narrow N(μₙ, 0.3)           5.1495     0.0022     4443.6      88.9%
Too wide N(0, 5)                5.1661     0.0144      196.5       3.9%
```

제안 고르기 전략을 체계로 다룬 것은 [제안 분포 설계](proposal_design.md)를 보아라.

---

## 8. 여러 기댓값 계산

### 표본 다시 쓰기

같은 가중 표본으로 여러 함수를 평가할 수 있지만 정확도는 함수마다 다르다. 아래 Var 항의 중심 estimate는 앞 제안 비교 루프에서 덮어쓴 별도 추정값이며 해당 가중 표본의 평균과 다를 수 있다. 그 결과를 정확한 가중 표본분산으로 읽지 않는다.

같은 가중 표본에서 여러 관심 함수를 계산할 수 있다.

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

**출력:**

```
Multiple Expectations from Same Samples
----------------------------------------
  E[θ]: 5.234283
  E[θ²]: 27.457577
  E[θ³]: 144.345505
  Var[θ]: 0.064493
  P(θ > 3): 1.000000
```

---

## 9. 사후예측분포

### 정의

사후예측분포는 다음과 같다.

$$
p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta
$$

### SNIS 추정

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

평균 μ와 log σ를 좌표로 하는 정규 수익률 모형의 사후평균 계산 예제다. log σ에 직접 정규 사전을 두므로 이 좌표의 밀도로 웨이트를 계산한다. σ에 사전을 둔 뒤 로그 좌표로 옮기는 경우와 야코비안 처리가 다르다.

proposal_dist.log_prob(samples).sum(dim=−1)은 좌표별 로그확률을 반환하는 독립 Normal 배치 분포를 가정한다. MultivariateNormal이나 Independent가 이미 사건 차원을 합산해 길이 n의 log_prob를 반환하면 다시 합해 잘못된 웨이트가 된다. 제안 인터페이스와 표본 형태 n×2를 명시해야 한다.

### 수익률 모형 예제

```python
def bayesian_return_model_snis(returns, proposal_dist, n_samples=10000):
    """
    수익 분포 파라메터의 베이즈 어림을 위한 SNIS.
    
    모형: r_t ~ N(μ, σ²)
    앞확률: μ ~ N(0, 0.1²), log σ ~ N(log(0.02), 0.5²)
    
    파라메터
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

### 주변 우도 추정의 추가 조건

우도와 사전밀도의 절대 정규화 상수가 포함된 γ에서 q가 정규화된 제안이면 원시 웨이트 평균 $\hat Z=n^{-1}\sum_iW_i$로 증거를 추정할 수 있다. 정규화 웨이트만 남기면 Z의 크기를 복원할 수 없다. 이 계산은 SNIS 비율 자체가 아니라 같은 표본의 표준 IS 적분이다.

모델 간 로그 BF의 추정값은 log Ẑ₁−log Ẑ₂다. 로그와 비의 비선형성 때문에 Ẑ가 불편이어도 이 로그비가 불편하다는 뜻은 아니다. 각 증거의 수치 오차와 사전 가정을 함께 보고한다.

## 11. 핵심 정리

SNIS는 미지 곱셈 상수를 약분해 기댓값을 구한다. 일반적인 바이어스와 조건부 일관성, 점근 분산의 유한성을 구분한다. ESS의 고정 임계값으로 모든 평균·꼬리·분위수의 정확도를 인증하지 않는다.

같은 가중 표본을 재사용해도 함수별 오차가 다르다. 여러 제안·반복 실행과 지지집합·꼬리를 점검하며, 증거가 필요하면 γ의 절대 정규화 상수를 복원해야 한다.

## 연습문제

1. q=π 또는 상수 h에서 SNIS가 불편인 이유를 설명하고, 일반적인 비율 추정의 O(1/n) 바이어스 전개가 추가 조건을 요구함을 논하라.
2. 고정된 q·π와 유한 E_q[W²]에서 ESS_w/n의 극한을 구하라. 이 비율이 각 유한 표본에서 일정하거나 모든 함수의 분산 비율인 것은 아닌 이유를 설명하라.
3. 알려진 분산의 정규-정규 모형에서 사후예측 평균·분산을 유도하고 SNIS 가중 적분과 비교할 절차를 설계하라. 새 관측의 잡음분산과 파라메터 사후분산을 구분하라.
4. 사전·라플라스·정확한 사후 제안의 표본 생성 비용과 꼬리를 비교하라. 관심 함수가 바뀌면 ESS 순위가 실제 오차 순위를 보장하지 않는 이유를 설명하라.
5. 우도·사전의 상수를 포함한 원시 웨이트 평균으로 증거를 추정하라. 미정규화 γ에 상수 c를 곱할 때 SNIS 기대값·ESS·Z 추정값이 각각 어떻게 변하는지 설명하라.

## 정리하며

SNIS는 정규화 가중 평균이며 목표밀도의 공통 곱셈 상수에 불변이다. 일반적인 유한 표본 바이어스와 함수별 점근 분산을 고려하고, 웨이트 ESS를 정확도 보장으로 읽지 않는다. 주변 우도는 원시 웨이트와 절대 밀도 상수가 있어야 별도로 추정할 수 있다.

**참고 문헌**

1. Geweke, J. (1989). "Bayesian inference in econometric models using Monte Carlo integration." *Econometrica*, 57(6), 1317-1339.

2. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 10.4절: 자기정규화 중요도 샘플링.

3. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. 3.3절.

4. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

5. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.
