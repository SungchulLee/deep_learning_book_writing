# 중요도 표집의 바탕
## 개요

중요도 표집은 어떤 분포 아래의 기댓값을 다른 분포에서 표집해 셈하게 해 주는 흩어짐 줄이기 기법이다. 이 방법은 베이즈 추론에 근본이 되는데, 베이즈 추론에서는 곧바로 표집하기 어렵거나 불가능한 복잡한 뒤확률 분포에 대해 적분해야 할 때가 많기 때문이다.

## 지난 이야기

중요도 표집은 1940년대 후반 로스앨러모스의 몬테카를로 연구에서 나왔다. "중요도 표집"이라는 말과 그 현대적인 꼴은 1950년 무렵 중성자 이동 문제를 연구하던 허먼 칸, T. E. 해리스와 동료들의 일에서 나타났다. 이 방법은 해머슬리와 핸즈콤(1964)이 체계로 세웠고, 나중에 클록과 판데이크(1978)의 일을 거쳐 베이즈 통계에 받아들여졌다.

## 근본 문제

### 판 벌이기

우리는 과녁 분포 $\pi(\theta)$ 아래의 기댓값을 셈하려 한다:

$$
I = \mathbb{E}_{\pi}[h(\theta)] = \int h(\theta) \, \pi(\theta) \, d\theta
$$

**어려움**: $\pi(\theta)$에서 곧바로 표집하기가 어렵거나 불가능할 수 있다.

### 흔한 상황

1. **베이즈 추론**: $\pi(\theta) = p(\theta|y)$은 뒤확률이며 비례 상수까지만 알려져 있다
2. **드문 일 어림하기**: $\pi$ 아래에서 확률이 $< 10^{-6}$인 일
3. **복잡한 밀도**: 표준 표집 알고리즘이 없다

## 측도 바꾸기

### 핵심 항등식

제안 분포 $q(\theta)$을 들여와 $1 = \frac{q(\theta)}{q(\theta)}$을 곱한다:

$$
I = \int h(\theta) \, \pi(\theta) \, d\theta = \int h(\theta) \frac{\pi(\theta)}{q(\theta)} \, q(\theta) \, d\theta
$$

**중요도 무게**를 다음과 같이 정한다:

$$
w(\theta) = \frac{\pi(\theta)}{q(\theta)}
$$

그러면 다음과 같다.

$$
I = \int h(\theta) \, w(\theta) \, q(\theta) \, d\theta = \mathbb{E}_q[h(\theta) \cdot w(\theta)]
$$

### 받침 조건

!!! danger "꼭 지켜야 할 조건"
    제안 $q$은 과녁 $\pi$을 덮어야 한다:
    
    $$\pi(\theta) > 0 \implies q(\theta) > 0$$
    
    $\pi(\theta) > 0$인 곳에서 $q(\theta) = 0$이면 무게 $w(\theta)$이 정해지지 않아 **끝없는 치우침**이 생긴다.

## 중요도 표집 어림자

### 몬테카를로 어림

표본 $\theta_1, \ldots, \theta_n \stackrel{\text{i.i.d.}}{\sim} q(\theta)$을 뽑아 다음과 같이 어림한다:

$$
\hat{I}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \, w(\theta_i) = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \frac{\pi(\theta_i)}{q(\theta_i)}
$$

### 성질

**치우침 없음:**

$$
\mathbb{E}_q[\hat{I}_{\text{IS}}] = \mathbb{E}_q\left[\frac{1}{n} \sum_{i=1}^n h(\theta_i) w(\theta_i)\right] = \mathbb{E}_q[h(\theta) w(\theta)] = I
$$

**흩어짐:**

$$
\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n} \text{Var}_q(h(\theta) w(\theta)) = \frac{1}{n}\left(\mathbb{E}_q[h^2(\theta) w^2(\theta)] - I^2\right)
$$

## 직관: 중요한 곳에서 표집하라

### 핵심 생각

$\pi$에서 소박하게 몬테카를로를 하면:

- 많은 표본이 $|h(\theta)|$이 작은 구역에 떨어져 보태는 바가 적다
- $|h(\theta)|$이 큰 곳에는 표본이 적게 떨어져 흩어짐이 크다

$q$에서 중요도 표집을 하면:

- $|h(\theta)\pi(\theta)|$이 큰 중요한 구역을 **넘치게 표집**한다
- 무게 $w(\theta) = \pi(\theta)/q(\theta)$으로 넘치게 표집한 것을 **바로잡는다**

표어로 하면 **"중요한 곳에서 표집하고 무게로 바로잡아라."**

### 그림으로 보는 직관

```
Target π(θ):        [.....|XXXXXX|.....]
                          ↑ high density

Naive MC:           draws mostly from middle, few from tails

Important region:   [.....|..XXXX|.....]
(for some h)              ↑ where |h·π| is large

Good proposal q:    [.....|..XXXX|.....] 
                    puts extra mass where it matters
```

## 흩어짐 분석

### 일반 흩어짐 공식

$$
\text{Var}_q(h(\theta) w(\theta)) = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta - I^2
$$

### 이차 적률

다음과 같이 정한다:

$$
J(q) = \mathbb{E}_q[h^2(\theta) w^2(\theta)] = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta
$$

그러면 $\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n}(J(q) - I^2)$이다

### 중요도 표집이 흩어짐을 줄일 때

다음일 때 중요도 표집이 흩어짐을 줄인다:

$$
\mathbb{E}_q[h^2(\theta) w^2(\theta)] < \mathbb{E}_\pi[h^2(\theta)]
$$

$q$이 $|h(\theta)\pi(\theta)|$이 큰 곳에 표본을 모을 때 이렇게 된다.

### 중요도 표집이 흩어짐을 키울 때

$q$을 잘못 고르면 중요도 표집이 흩어짐을 *키운다*:

- $q$이 너무 좁다: 중요한 구역을 놓쳐 어떤 무게가 터진다
- $q$이 $\pi$에서 비켜 있다: 무게 대부분이 0에 가깝고 몇몇만 아주 크다

!!! warning "무게 주저앉음"
    $q$이 $\pi$과 잘 맞지 않으면:
    
    - 몇몇 표본의 무게가 엄청나게 크다
    - 표본 대부분의 무게가 하찮다
    - 실효 표본 크기가 주저앉는다
    - 흩어짐이 터진다

## 가장 좋은 제안 분포

### 변분법으로 이끌어 내기

제약 $\int q(\theta) d\theta = 1$ 아래 $J(q) = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta$을 가장 작게 한다.

라그랑주 곱수를 쓰면 가장 좋은 $q^*$은 다음을 만족한다:

$$
\frac{\partial}{\partial q}\left[h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} + \lambda q(\theta)\right] = 0
$$

$$
-h^2(\theta) \frac{\pi^2(\theta)}{q^2(\theta)} + \lambda = 0 \implies q(\theta) \propto |h(\theta)| \pi(\theta)
$$

### 가장 좋은 제안

$$
\boxed{q^*(\theta) = \frac{|h(\theta)| \pi(\theta)}{\int |h(\theta')| \pi(\theta') d\theta'}}
$$

**왜 절댓값인가?** 제안은 음이 아니어야 한다. $h$이 음일 수 있으면 $|h| \pi$에 비례해 표집하고 부호는 어림자 안의 $h$이 진다.

### 가장 좋을 때의 흩어짐

$h(\theta) \geq 0$일 때:

$$
q^*(\theta) = \frac{h(\theta) \pi(\theta)}{I}
$$

그러면 다음과 같다.

$$
\text{Var}_{q^*}(\hat{I}_{\text{IS}}) = \frac{1}{n}(I^2 - I^2) = 0
$$

**가장 좋은 제안은 흩어짐 0을 이룬다.** 곧 피적분 함수 자체에서 표집하는 셈이다!

### 실전에서 뜻하는 바

가장 좋은 $q^*$은 모르는 값 $I$에 기대므로 곧바로 쓸 수 없다. 그러나 이 이끌어 내기는 좋은 제안이 어떤 모습이어야 하는지를 드러낸다:

1. **모양**: $|h(\theta)|\pi(\theta)$을 따른다
2. **받침**: $\pi(\theta) > 0$인 곳을 모두 덮는다
3. **꼬리**: 적어도 $\pi$만큼은 무거워야 한다

## PyTorch 구현

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def importance_sampling(h_function, target_log_pdf, proposal_dist, 
                        n_samples, return_diagnostics=False):
    """
    고르게 하는 상수를 아는 표준 중요도 표집.
    
    매개변수
    ----------
    h_function : callable
        기댓값을 셈하려는 함수 h(θ)
    target_log_pdf : callable  
        과녁 π(θ)의 로그 밀도
    proposal_dist : torch.distributions.Distribution
        제안 분포 q(θ)
    n_samples : int
        뽑을 표본의 개수
    return_diagnostics : bool
        진단 정보를 돌려줄지 여부
        
    반환값
    -------
    estimate : torch.Tensor
        E_π[h(θ)]의 중요도 표집 어림값
    se : torch.Tensor
        어림한 표준 오차
    diagnostics : dict, optional
        표본, 무게, 그 밖의 진단
    
    수학의 바탕
    -----------------------
    Î_IS = (1/n) Σᵢ h(θᵢ) w(θᵢ)
    
    여기서 w(θ) = π(θ)/q(θ)이고 θᵢ ~ q(θ)이다
    """
    # 걸음 1: 제안 q(θ)에서 표본 뽑기
    samples = proposal_dist.sample((n_samples,))
    
    # 걸음 2: 로그 밀도 값 매기기
    log_target = target_log_pdf(samples)
    log_proposal = proposal_dist.log_prob(samples)
    
    # 걸음 3: 중요도 무게 셈하기(안정을 위해 로그 공간에서)
    # w(θ) = π(θ)/q(θ)  →  log w(θ) = log π(θ) - log q(θ)
    log_weights = log_target - log_proposal
    weights = torch.exp(log_weights)
    
    # 걸음 4: 표본 점에서 함수 h 값 매기기
    h_values = h_function(samples)
    
    # 걸음 5: 중요도 표집 어림값 셈하기: (1/n) Σᵢ h(θᵢ) w(θᵢ)
    weighted_h = h_values * weights
    estimate = torch.mean(weighted_h)
    
    # 걸음 6: 표준 오차 어림하기
    variance = torch.var(weighted_h, unbiased=True)
    se = torch.sqrt(variance / n_samples)
    
    if return_diagnostics:
        diagnostics = {
            'samples': samples,
            'weights': weights,
            'h_values': h_values,
            'log_weights': log_weights,
            'weighted_h': weighted_h,
            'n_samples': n_samples
        }
        return estimate, se, diagnostics
    
    return estimate, se


# 보기: π = N(3, 1)일 때 E_π[θ²] 셈하기
# 제안 q = N(0, 2) 사용

# 참값: E[θ²] = μ² + σ² = 9 + 1 = 10
true_value = 10.0

# 과녁과 제안 정하기
target_mean, target_std = 3.0, 1.0
proposal_mean, proposal_std = 0.0, 2.0

# 과녁의 로그 밀도(고르게 함)
def target_log_pdf(theta):
    return dist.Normal(target_mean, target_std).log_prob(theta)

# 제안 분포
proposal = dist.Normal(proposal_mean, proposal_std)

# 관심 있는 함수
h = lambda theta: theta**2

# 중요도 표집 돌리기
estimate, se, diagnostics = importance_sampling(
    h, target_log_pdf, proposal, 
    n_samples=10000, return_diagnostics=True
)

print("Importance Sampling: E_π[θ²] where π = N(3,1), q = N(0,2)")
print("=" * 60)
print(f"True value: {true_value:.6f}")
print(f"IS estimate: {estimate.item():.6f}")
print(f"Standard error: {se.item():.6f}")
print(f"Error: {abs(estimate.item() - true_value):.6f}")

# 무게 진단
weights = diagnostics['weights']
print(f"\nWeight Statistics:")
print(f"  Mean: {weights.mean().item():.4f}")
print(f"  Std: {weights.std().item():.4f}")
print(f"  Max: {weights.max().item():.4f}")
print(f"  Min: {weights.min().item():.4f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 칸 1: 과녁 분포와 제안 분포
x = torch.linspace(-5, 8, 500)
ax = axes[0, 0]
ax.plot(x.numpy(), torch.exp(target_log_pdf(x)).numpy(), 
        'b-', linewidth=2, label=f'Target π = N({target_mean},{target_std})')
ax.plot(x.numpy(), proposal.log_prob(x).exp().numpy(), 
        'r--', linewidth=2, label=f'Proposal q = N({proposal_mean},{proposal_std})')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Target vs Proposal Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 칸 2: 무게 분포
ax = axes[0, 1]
ax.hist(weights.numpy(), bins=50, density=True, alpha=0.7, 
        color='purple', edgecolor='black')
ax.axvline(weights.mean().item(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean = {weights.mean().item():.3f}')
ax.set_xlabel('Importance Weight w(θ)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Importance Weights', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 칸 3: 무게로 색칠한 표본
samples = diagnostics['samples']
ax = axes[1, 0]
scatter = ax.scatter(samples.numpy(), h(samples).numpy(), 
                     c=weights.numpy(), cmap='hot', alpha=0.6, 
                     s=30, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Sample θ', fontsize=12)
ax.set_ylabel('h(θ) = θ²', fontsize=12)
ax.set_title('Samples Colored by Importance Weight', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight')
ax.grid(True, alpha=0.3)

# 칸 4: 모임
n_values = torch.arange(100, 10001, 100)
estimates = []
for n in n_values:
    est, _ = importance_sampling(h, target_log_pdf, proposal, int(n))
    estimates.append(est.item())

ax = axes[1, 1]
ax.plot(n_values.numpy(), estimates, 'b-', linewidth=1.5, alpha=0.7)
ax.axhline(true_value, color='red', linestyle='--', linewidth=2, 
           label='True Value')
ax.fill_between(n_values.numpy(), true_value - 0.2, true_value + 0.2,
                alpha=0.2, color='red', label='±0.2 band')
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('IS Estimate', fontsize=12)
ax.set_title('Convergence of IS Estimate', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('importance_sampling_fundamentals.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 제안 분포 견주기

### 제안을 고르는 것의 효과

```python
def compare_proposals(h_function, target_log_pdf, proposals, 
                      n_samples=5000, n_replications=100):
    """
    여러 제안 분포 견주기.
    """
    results = []
    
    for name, proposal in proposals.items():
        estimates = []
        ess_values = []
        
        for _ in range(n_replications):
            est, se, diag = importance_sampling(
                h_function, target_log_pdf, proposal, 
                n_samples, return_diagnostics=True
            )
            estimates.append(est.item())
            
            # ESS 셈하기
            w = diag['weights']
            w_normalized = w / w.sum()
            ess = 1.0 / (w_normalized**2).sum()
            ess_values.append(ess.item())
        
        results.append({
            'name': name,
            'mean': torch.tensor(estimates).mean().item(),
            'std': torch.tensor(estimates).std().item(),
            'mean_ess': torch.tensor(ess_values).mean().item(),
            'ess_ratio': torch.tensor(ess_values).mean().item() / n_samples
        })
    
    return results

# 질이 다른 제안들 정하기
proposals = {
    'Good: N(3, 1.2)': dist.Normal(3.0, 1.2),
    'Okay: N(2, 1.5)': dist.Normal(2.0, 1.5),
    'Poor: N(0, 2)': dist.Normal(0.0, 2.0),
    'Bad: N(3, 0.5)': dist.Normal(3.0, 0.5),  # 너무 좁음
}

results = compare_proposals(h, target_log_pdf, proposals)

print("\nProposal Comparison:")
print("-" * 70)
print(f"{'Proposal':<20} {'Mean Est':>12} {'Std Dev':>12} {'ESS':>12} {'ESS/n':>12}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<20} {r['mean']:12.4f} {r['std']:12.4f} "
          f"{r['mean_ess']:12.1f} {r['ess_ratio']:12.2%}")
```

## 흩어짐 줄이기 보기: 꼬리 확률

### 드문 일 문제

$X \sim \mathcal{N}(0, 1)$일 때 $\mathbb{P}_\pi(X > 4)$을 어림하여라.

```python
# 참 확률
true_prob = 1 - dist.Normal(0, 1).cdf(torch.tensor(4.0))
print(f"True P(X > 4): {true_prob.item():.2e}")

# 지시 함수
h_indicator = lambda x: (x > 4).float()

# 과녁에서 뽑는 어수룩한 몬테카를로
target = dist.Normal(0, 1)
n_samples = 100000

# 어수룩한 몬테카를로
samples_naive = target.sample((n_samples,))
naive_estimate = h_indicator(samples_naive).mean()
naive_se = h_indicator(samples_naive).std() / (n_samples**0.5)

print(f"\nNaive MC (n={n_samples}):")
print(f"  Estimate: {naive_estimate.item():.6f}")
print(f"  SE: {naive_se.item():.6f}")

# 옮긴 제안을 쓴 중요도 표집
shifted_proposal = dist.Normal(4.0, 1.5)  # 꼬리에 가운데를 맞춤

def target_log_pdf_std(x):
    return dist.Normal(0, 1).log_prob(x)

is_estimate, is_se, is_diag = importance_sampling(
    h_indicator, target_log_pdf_std, shifted_proposal,
    n_samples=10000, return_diagnostics=True
)

print(f"\nImportance Sampling (n=10000):")
print(f"  Estimate: {is_estimate.item():.6f}")
print(f"  SE: {is_se.item():.6f}")

# 흩어짐 줄임 배수
variance_reduction = (naive_se / is_se)**2
print(f"\nVariance reduction factor: {variance_reduction.item():.1f}x")
```

## 계량 금융에서의 쓰임새

### 드문 일 위험 어림하기

중요도 표집은 계량 금융에서 꼬리 위험 잣대를 어림하는 데 널리 쓰인다. 극단 분위수(이를테면 99.9%)에서 위험 값(VaR)과 기대 부족액(ES)을 셈하려면 드문 손실 일을 효율적으로 어림해야 하는데, 이는 위에서 세운 중요도 표집 얼개의 자연스러운 쓰임새이다.

```python
def estimate_var_es_importance_sampling(
    loss_function, target_dist, proposal_dist, 
    alpha=0.999, n_samples=10000
):
    """
    중요도 표집으로 VaR과 기댓값 모자람 어림하기.
    
    매개변수
    ----------
    loss_function : callable
        위험 인자를 포트폴리오 손실로 잇는다
    target_dist : torch.distributions.Distribution
        P 아래에서 위험 인자의 분포
    proposal_dist : torch.distributions.Distribution
        꼬리 쪽으로 치우친 제안
    alpha : float
        믿음 수준(이를테면 0.999)
    n_samples : int
        중요도 표집 표본의 개수
    """
    # 제안에서 표집
    samples = proposal_dist.sample((n_samples,))
    
    # 중요도 무게 셈하기
    log_weights = target_dist.log_prob(samples) - proposal_dist.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
    
    # 손실 셈하기
    losses = loss_function(samples)
    
    # 무게 분위수를 어림하려고 손실로 정렬
    sorted_indices = torch.argsort(losses)
    sorted_losses = losses[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # VaR을 위한 무게 누적분포함수
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    var_idx = (cumulative_weights >= alpha).nonzero(as_tuple=True)[0][0]
    var_estimate = sorted_losses[var_idx]
    
    # 기댓값 모자람: E[L | L > VaR]
    tail_mask = losses > var_estimate
    if tail_mask.sum() > 0:
        tail_weights = weights[tail_mask]
        tail_weights = tail_weights / tail_weights.sum()
        es_estimate = torch.sum(tail_weights * losses[tail_mask])
    else:
        es_estimate = var_estimate
    
    # ESS 진단
    ess = 1.0 / torch.sum(weights**2)
    
    return {
        'var': var_estimate.item(),
        'es': es_estimate.item(),
        'ess': ess.item(),
        'ess_ratio': (ess / n_samples).item()
    }

# 보기: 꼬리 두꺼운 포트폴리오 손실
torch.manual_seed(42)

# 위험 인자 분포(보통 시장 상황)
target = dist.Normal(0.0, 1.0)

# 손실 꼬리 쪽으로 옮긴 제안
proposal = dist.Normal(3.0, 1.5)

# 단순한 손실 함수: L = exp(0.5 * X) - 1
loss_fn = lambda x: torch.exp(0.5 * x) - 1.0

results = estimate_var_es_importance_sampling(
    loss_fn, target, proposal, alpha=0.999, n_samples=50000
)

print("Tail Risk Estimation via Importance Sampling")
print("=" * 50)
print(f"  VaR(99.9%): {results['var']:.4f}")
print(f"  ES(99.9%):  {results['es']:.4f}")
print(f"  ESS:        {results['ess']:.1f} ({results['ess_ratio']:.1%})")
```

## 베이즈 추론에서의 쓰임새

### 뒤확률 기댓값

베이즈 추론에서 우리는 뒤확률 아래의 기댓값을 구하려 한다:

$$
\mathbb{E}[h(\theta)|y] = \int h(\theta) \, p(\theta|y) \, d\theta
$$

여기서 $p(\theta|y) \propto p(y|\theta) p(\theta)$이다.

뒤확률은 보통 비례 상수까지만 알려져 있으며, 그래서 **스스로 고르게 하는 중요도 표집**이 나온다([스스로 고르게 하는 중요도 표집](self_normalized.md)에서 다룬다).

### 모형 증거와의 이음

중요도 표집으로 주변 가능도도 어림할 수 있다:

$$
p(y) = \int p(y|\theta) p(\theta) d\theta
$$

앞확률을 제안으로 쓰면 $q(\theta) = p(\theta)$이다

$$
\hat{p}(y) = \frac{1}{n} \sum_{i=1}^n p(y|\theta_i), \quad \theta_i \sim p(\theta)
$$

이것이 조화 평균 어림자이다(다만 흩어짐이 끝없을 수 있다. 나아간 주제를 보아라).

## 핵심 정리

!!! success "중요도 표집을 언제 쓰나"

    - 과녁 분포에서 표집하기 어려울 때
    - 좋은 제안 분포를 쓸 수 있을 때
    - 드문 일에서 흩어짐을 줄여야 할 때
    - 여러 기댓값에 표본을 다시 쓸 때

!!! warning "중요도 표집이 무너질 수 있을 때"

    - 제안을 꼼꼼히 짜지 않은 채 차원이 높을 때
    - 과녁의 꼬리가 제안보다 무거울 때
    - 봉우리가 여럿인 과녁에 성분이 하나뿐인 제안을 쓸 때
    - 과녁과 제안이 아주 크게 어긋날 때

!!! info "좋은 버릇"

    1. 무게 진단(흩어짐, 최대 무게, ESS)을 늘 살펴라
    2. 제안의 꼬리가 과녁보다 무거워야 한다
    3. 과녁의 받침 전체를 덮어라
    4. 복잡한 과녁에는 알아서 맞추는 방법을 생각해 보아라

## 참고 문헌

1. Kahn, H., & Harris, T. E. (1951). "Estimation of particle transmission by random sampling." *National Bureau of Standards Applied Mathematics Series*, 12, 27-30.

2. Hammersley, J. M., & Handscomb, D. C. (1964). *Monte Carlo Methods*. Methuen.

3. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 9장: 중요도 표집.

4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. 3장.

5. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2장.

6. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. 4-5장.

## 연습문제

### 연습 1: 흩어짐 견주기
$X \sim \mathcal{N}(0,1)$일 때 $\mathbb{E}[e^{2X}]$을 어림하는 소박한 몬테카를로와 중요도 표집의 흩어짐을 견주어라. 제안으로 $q = \mathcal{N}(0,1)$, $q = \mathcal{N}(1,1)$, $q = \mathcal{N}(2,1)$을 써라. 어느 것이 흩어짐이 가장 작으며 왜 그런가?

### 연습 2: 받침 덮기
제안이 과녁의 받침을 덮지 않으면 어떻게 되는지 보여라. $\pi = \mathcal{N}(0,1)$, $q = \text{Uniform}(-2, 2)$이라 하자. $\mathbb{E}[X^2]$을 어림하고 치우침을 설명하여라.

### 연습 3: 가장 좋은 제안 어림하기
$h(\theta) = \theta^2$이고 $\pi = \mathcal{N}(3,1)$일 때 가장 좋은 제안은 $q^* \propto \theta^2 \cdot \mathcal{N}(3,1)$이다. $|\theta| \cdot \mathcal{N}(3,1)$에 가우스를 맞춰 이를 어림하고, 앞확률을 제안으로 쓸 때와 흩어짐을 견주어라.

### 연습 4: 중요도 표집으로 옵션 값 매기기
블랙-숄즈 모형 아래에서 깊은 외가격 유럽식 콜 옵션의 값을 중요도 표집으로 매겨라. 소박한 몬테카를로(위험 중립 측도 아래에서 경로를 표집)와 제안을 행사가 쪽으로 옮긴 중요도 표집의 흩어짐을 견주어라. 흩어짐이 줄어든 배수를 셈하여라.
