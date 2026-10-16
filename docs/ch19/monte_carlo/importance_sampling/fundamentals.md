# 중요도 샘플링 기초

중요도 샘플링은 제안분포에서 얻은 표본에 밀도비 웨이트를 부여해 목표분포의 기댓값을 추정한다. 직접 표본 생성이 어려운 분포나 희귀 사건 적분에 유용하지만, 제안분포가 부적절하면 분산이 커지거나 무한해질 수 있다. 정규화된 목표의 표준 추정량과 미정규화 목표의 자기정규화 추정량을 구분한다.

## 1. 지난 이야기

중요도 샘플링은 입자 수송을 비롯한 초기 몬테카를로 연구와 통계적 적분에서 발전했다. 이 절은 역사적 우선권보다 밀도비 항등식·불편성·분산 조건에 초점을 둔다. 아래 참고 문헌의 Kahn–Harris 연구와 현대 몬테카를로 교재는 서로 다른 응용과 계산 관점을 제공한다.

## 2. 근본 문제

### 문제 설정

우리는 목표분포 $\pi(\theta)$ 아래의 기댓값을 셈하려 한다:

$$
I = \mathbb{E}_{\pi}[h(\theta)] = \int h(\theta) \, \pi(\theta) \, d\theta
$$

**어려움**: $\pi(\theta)$에서 곧바로 표집하기가 어렵거나 불가능할 수 있다.

### 흔한 상황

1. **베이즈 추론**: $\pi(\theta) = p(\theta|y)$은 사후분포가며 비례 상수까지만 알려져 있다
2. **희귀 사건 추정하기**: $\pi$ 아래에서 확률이 $< 10^{-6}$인 일
3. **복잡한 밀도**: 표준 샘플링 알고리즘이 없다

---

## 3. 측도 바꾸기

### 핵심 항등식

제안분포 $q(\theta)$를 들여와 $1 = \frac{q(\theta)}{q(\theta)}$을 곱한다:

$$
I = \int h(\theta) \, \pi(\theta) \, d\theta = \int h(\theta) \frac{\pi(\theta)}{q(\theta)} \, q(\theta) \, d\theta
$$

**중요도 웨이트**를 다음과 같이 정한다:

$$
w(\theta) = \frac{\pi(\theta)}{q(\theta)}
$$

그러면 다음과 같다.

$$
I = \int h(\theta) \, w(\theta) \, q(\theta) \, d\theta = \mathbb{E}_q[h(\theta) \cdot w(\theta)]
$$

### 지지집합 조건

이 적분 하나에서는 h(θ)π(θ)≠0인 곳에서 q(θ)>0이어야 한다. 여러 함수 또는 자기정규화 추정을 위해서는 일반적으로 π의 지지집합 전체를 덮는 제안을 선택한다. q=0인 누락 영역의 hπ 적분은 추정할 수 없어 바이어스가 생긴다. 그 바이어스가 반드시 무한하다는 뜻은 아니다. 지지집합을 덮어도 가중 이차 적률은 무한할 수 있다.

## 4. 중요도 샘플링 추정량

### 몬테카를로 추정

표본 $\theta_1, \ldots, \theta_n \stackrel{\text{i.i.d.}}{\sim} q(\theta)$을 뽑아 다음과 같이 추정한다:

$$
\hat{I}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \, w(\theta_i) = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \frac{\pi(\theta_i)}{q(\theta_i)}
$$

### 성질

π는 정규화된 목표밀도이며 h가 절대 적분 가능하고 제안표본은 독립·동일분포라고 가정한다. 이 경우 표준 IS는 불편하다. 아래 유한 분산과 일반적인 √n 중심극한정리에는 ∫h²π²/q<∞가 추가로 필요하다. 미정규화 목표를 그대로 넣으면 기대값의 미지 상수가 함께 곱해지므로 같은 추정량이 아니다.

**바이어스 없음:**

$$
\mathbb{E}_q[\hat{I}_{\text{IS}}] = \mathbb{E}_q\left[\frac{1}{n} \sum_{i=1}^n h(\theta_i) w(\theta_i)\right] = \mathbb{E}_q[h(\theta) w(\theta)] = I
$$

**분산:**

$$
\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n} \text{Var}_q(h(\theta) w(\theta)) = \frac{1}{n}\left(\mathbb{E}_q[h^2(\theta) w^2(\theta)] - I^2\right)
$$

---

## 5. 직관: 중요한 곳에서 표집하라

### 핵심 생각

$\pi$에서 소박하게 몬테카를로를 하면:

- 많은 표본이 $|h(\theta)|$이 작은 구역에 떨어져 보태는 바가 적다
- $|h(\theta)|$이 큰 곳에는 표본이 적게 떨어져 분산이 크다

$q$에서 중요도 샘플링을 하면:

- $|h(\theta)\pi(\theta)|$이 큰 중요한 구역을 **넘치게 샘플링**한다
- 웨이트 $w(\theta) = \pi(\theta)/q(\theta)$으로 넘치게 표집한 것을 **바로잡는다**

표어로 하면 **"중요한 곳에서 샘플링하고 웨이트로 바로잡아라."**

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

---

## 6. 분산 분석

### 일반 분산 공식

$$
\text{Var}_q(h(\theta) w(\theta)) = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta - I^2
$$

### 이차 적률

다음과 같이 정한다:

$$
J(q) = \mathbb{E}_q[h^2(\theta) w^2(\theta)] = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta
$$

그러면 $\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n}(J(q) - I^2)$이다

### 중요도 샘플링이 분산을 줄일 때

다음일 때 중요도 샘플링이 분산을 줄인다:

$$
\mathbb{E}_q[h^2(\theta) w^2(\theta)] < \mathbb{E}_\pi[h^2(\theta)]
$$

피적분 기여가 큰 영역에 적절히 확률을 배분하면 이 조건을 충족할 수 있다. 단순히 목표분포의 중심에 가까운 제안이라는 사실만으로는 충분하지 않으며 꼬리와 관심 함수 h도 영향을 준다.

### 중요도 샘플링이 분산을 키울 때

$q$를 잘못 고르면 중요도 샘플링이 분산을 *키운다*:

- $q$가 너무 좁다: 중요한 구역을 놓쳐 어떤 웨이트가 터진다
- $q$가 $\pi$에서 비켜 있다: 웨이트 대부분이 0에 가깝고 몇몇만 아주 크다

!!! warning "웨이트 퇴화"
    $q$가 $\pi$와 잘 맞지 않으면:
    
    - 몇몇 표본의 웨이트가 엄청나게 크다
    - 표본 대부분의 웨이트가 하찮다
    - 유효 표본 크기가 감소한다
    - 분산이 터진다

---

## 7. 가장 좋은 제안 분포

0<A=∫|h(θ)|π(θ)dθ<∞라 하자. 정규화된 q에 대한 Cauchy–Schwarz 부등식에서 다음을 얻는다.

$$A^2=\left(\int\frac{|h|\pi}{\sqrt q}\sqrt q\right)^2\leq\int\frac{h^2\pi^2}{q}$$

따라서 이 적분 하나의 표준 IS 분산을 최소화하는 제안은 다음과 같다.

$$q^*(\theta)=\frac{|h(\theta)|\pi(\theta)}A,\qquad\operatorname{Var}(\hat I_{q^*})=\frac{A^2-I^2}{n}$$

h≥0이고 I>0이면 A=I여서 hπ/q*=I가 상수가 되고 분산은 0이다. h가 부호를 바꾸면 일반적으로 이 최솟값은 0이 아니다. q*가 h=0인 영역을 덮지 않아도 이 적분에는 문제가 없지만, 다른 함수·자기정규화에는 같은 제안을 자동 적용하지 않는다. [Owen의 중요도 샘플링 교재](https://artowen.su.domains/mc/Ch-var-is.pdf)를 참고한다.

실제로는 미지 정규화 상수와 표본 생성 때문에 q*를 그대로 사용하기 어렵다. |h|π의 모양을 근사하되 필요한 지지집합과 가중 이차 적률을 확인한다. 제안의 꼬리가 목표보다 항상 무거워야 한다는 조건은 필요조건이 아니며, 정확한 기준은 해당 적분의 이차 적률 유한성이다.

## 8. PyTorch 구현

아래 함수는 표본 하나마다 스칼라 h와 정규화된 log π·log q가 나오는 1차원 예제다. 다변량 분포의 log_prob가 좌표별 값을 반환하면 사건 차원에 대해 합해야 하며 그대로 mean하면 다른 양이 된다. 불편 표본분산에는 n_samples≥2가 필요하다.

로그 웨이트를 직접 exp하므로 오버플로·언더플로에 유의한다. log π에 상수만큼의 미지 오차가 있으면 표준 IS 추정값에도 그 상수가 곱해진다. 이 예제의 정규분포 두 번째 인자는 분산이 아니라 **표준편차**다.

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def importance_sampling(h_function, target_log_pdf, proposal_dist, 
                        n_samples, return_diagnostics=False):
    """
    고르게 하는 상수를 아는 표준 중요도 표집.
    
    파라메터
    ----------
    h_function : callable
        기댓값을 셈하려는 함수 h(θ)
    target_log_pdf : callable  
        타깃 π(θ)의 로그 밀도
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

# 타깃과 제안 정하기
target_mean, target_std = 3.0, 1.0
proposal_mean, proposal_std = 0.0, 2.0

# 타깃의 로그 밀도(고르게 함)
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

# 칸 1: 타깃 분포와 제안 분포
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

**출력:**

```
Importance Sampling: E_π[θ²] where π = N(3,1), q = N(0,2)
============================================================
True value: 10.000000
IS estimate: 9.982215
Standard error: 0.299109
Error: 0.017785

Weight Statistics:
  Mean: 0.9874
  Std: 2.1072
  Max: 8.9634
  Min: 0.0000
```

---

## 9. 제안 분포 비교

### 제안을 고르는 것의 효과

N(3,1) 목표에서 N(3,0.5) 제안은 지지집합은 맞지만 h(θ)=θ²의 가중 이차 적률이 무한하다. 유한 반복에서 표본분산이 작게 보이더라도 드문 큰 웨이트를 놓쳤을 수 있다. 웨이트 기반 ESS는 관심 함수별 적분 오차를 보장하지 않는다.

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

**출력:**

```
Proposal Comparison:
----------------------------------------------------------------------
Proposal                 Mean Est      Std Dev          ESS        ESS/n
----------------------------------------------------------------------
Good: N(3, 1.2)            9.9992       0.0833       4761.7       95.23%
Okay: N(2, 1.5)           10.0069       0.1702       3126.1       62.52%
Poor: N(0, 2)              9.8984       0.4012        914.3       18.29%
Bad: N(3, 0.5)            10.2465       4.9791        691.2       13.82%
```

---

## 10. 분산 줄이기 보기: 꼬리 확률

### 희귀 사건 문제

직접 MC는 100,000개, IS는 10,000개 표본을 사용하므로 출력의 SE 제곱 비는 동일 표본 수의 분산 감소율이 아니다. 같은 함수 평가 비용과 표본 수에서 비교해야 한다. 희귀 꼬리의 참값을 1−CDF로 계산하면 자료형에 따라 상쇄 오차가 생길 수 있다.

$X \sim \mathcal{N}(0, 1)$일 때 $\mathbb{P}_\pi(X > 4)$을 추정하여라.

```python
# 참 확률
true_prob = 1 - dist.Normal(0, 1).cdf(torch.tensor(4.0))
print(f"True P(X > 4): {true_prob.item():.2e}")

# 지시 함수
h_indicator = lambda x: (x > 4).float()

# 타깃에서 뽑는 어수룩한 몬테카를로
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

**출력:**

```
True P(X > 4): 3.17e-05

Naive MC (n=100000):
  Estimate: 0.000030
  SE: 0.000017

Importance Sampling (n=10000):
  Estimate: 0.000030
  SE: 0.000001

Variance reduction factor: 444.8x
```

---

## 11. 계량 금융에서의 쓰임새

### 희귀 사건 위험 추정하기

아래 함수는 자기정규화 가중 경험 누적분포의 분위수로 VaR를 추정한다. 로스가 연속이면 상위 꼬리의 가중 평균으로 ES를 근사할 수 있지만 이산 질량이 있으면 losses>VaR만의 평균이 일반적인 Expected Shortfall과 다르다. 꼬리 표본이 없을 때 VaR를 ES로 대체하는 처리는 올바른 ES를 보장하지 않는다. α·표본 수·웨이트 유한성과 관심 꼬리의 유효 정보를 확인해야 한다.

중요도 샘플링은 계량 금융에서 꼬리 위험 잣대를 추정하는 데 널리 쓰인다. 극단 분위수(이를테면 99.9%)에서 위험 값(VaR)과 기대 부족액(ES)을 셈하려면 드문 로스 일을 효율적으로 추정해야 하는데, 이는 위에서 세운 중요도 샘플링 구조의 자연스러운 쓰임새이다.

```python
def estimate_var_es_importance_sampling(
    loss_function, target_dist, proposal_dist, 
    alpha=0.999, n_samples=10000
):
    """
    중요도 표집으로 VaR과 기댓값 모자람 어림하기.
    
    파라메터
    ----------
    loss_function : callable
        위험 인자를 포트폴리오 로스로 잇는다
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
    
    # 로스 셈하기
    losses = loss_function(samples)
    
    # 무게 분위수를 어림하려고 로스로 정렬
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

# 보기: 꼬리 두꺼운 포트폴리오 로스
torch.manual_seed(42)

# 위험 인자 분포(보통 시장 상황)
target = dist.Normal(0.0, 1.0)

# 로스 꼬리 쪽으로 옮긴 제안
proposal = dist.Normal(3.0, 1.5)

# 단순한 로스 함수: L = exp(0.5 * X) - 1
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

**출력:**

```
Tail Risk Estimation via Importance Sampling
==================================================
  VaR(99.9%): 3.6712
  ES(99.9%):  4.4195
  ESS:        3213.9 (6.4%)
```

---

## 12. 베이즈 추론에서의 쓰임새

### 사후 기댓값

베이즈 추론에서 우리는 사후분포 아래의 기댓값을 구하려 한다:

$$
\mathbb{E}[h(\theta)|y] = \int h(\theta) \, p(\theta|y) \, d\theta
$$

여기서 $p(\theta|y) \propto p(y|\theta) p(\theta)$이다.

사후분포는 보통 비례 상수까지만 알려져 있으며, 그래서 **자기정규화 중요도 샘플링**이 나온다([자기정규화 중요도 샘플링](self_normalized.md)에서 다룬다).

### 모델 증거와의 이음

중요도 샘플링으로 주변 우도도 추정할 수 있다:

$$
p(y) = \int p(y|\theta) p(\theta) d\theta
$$

사전분포를 제안으로 쓰면 $q(\theta) = p(\theta)$이다

$$
\hat{p}(y) = \frac{1}{n} \sum_{i=1}^n p(y|\theta_i), \quad \theta_i \sim p(\theta)
$$

이것은 **사전 표본 우도의 산술평균**으로 구한 주변 우도 추정량이다. 사후표본의 역우도를 평균한 뒤 역수를 취하는 조화평균 방법과 다르다. 사전분포가 사후의 중요한 영역을 거의 방문하지 못하면 이 산술평균도 매우 비효율적일 수 있다.

---

## 13. 핵심 정리

표준 IS는 정규화된 목표와 적절한 지지집합 조건에서 불편하지만 유한 분산은 별도 이차 적률 조건이다. 목표와 비슷한 제안이라도 관심 함수와 꼬리에 따라 성능이 달라진다.

웨이트 분포·최대 웨이트·ESS와 함께 반복 실험 및 관심 함수의 기여를 살핀다. 높은 ESS나 유한한 표본분산만으로 누락 영역·무한 분산을 배제하지 못한다. 제안 선택과 적응에 자료를 사용하면 어떤 표본·밀도에 웨이트를 대응시키는지도 명시해야 한다.

## 연습문제

### 연습 1: 분산 비교
$X \sim \mathcal{N}(0,1)$일 때 $\mathbb{E}[e^{2X}]$을 추정하는 소박한 몬테카를로와 중요도 샘플링의 분산을 비교하여라. 제안으로 $q = \mathcal{N}(0,1)$, $q = \mathcal{N}(1,1)$, $q = \mathcal{N}(2,1)$을 써라. 어느 것이 분산이 가장 작으며 왜 그런가?

### 연습 2: 지지집합 포함
제안이 목표분포의 지지집합을 덮지 않으면 어떻게 되는지 보여라. $\pi = \mathcal{N}(0,1)$, $q = \text{Uniform}(-2, 2)$이라 하자. $\mathbb{E}[X^2]$을 추정하고 바이어스를 설명하여라.

### 연습 3: 최적 제안 근사
$h(\theta) = \theta^2$이고 $\pi = \mathcal{N}(3,1)$일 때 가장 좋은 제안은 $q^* \propto \theta^2 \cdot \mathcal{N}(3,1)$이다. 정규분포의 평균·분산을 조정해 θ²·N(3,1)의 정규화된 형태를 근사하라. |θ|가 아니라 θ²로 가중한 모멘트임을 확인하고, 원래 목표 N(3,1)을 제안으로 쓸 때와 가중 이차 적률·분산을 비교하라.

### 연습 4: 중요도 샘플링으로 옵션 값 매기기
블랙-숄즈 모델 아래에서 깊은 외가격 유럽식 콜 옵션의 값을 중요도 샘플링으로 매겨라. 소박한 몬테카를로(위험 중립 측도 아래에서 경로를 샘플링)와 제안을 행사가 쪽으로 옮긴 중요도 샘플링의 분산을 비교하여라. 분산이 줄어든 배수를 계산하여라.

## 정리하며

중요도 샘플링은 제안표본의 밀도비로 목표 기댓값을 추정한다. 지지집합·적분 가능성·가중 이차 적률을 나누어 확인한다. 최적 제안은 관심 함수에 의존하며 웨이트 ESS만으로 정확도를 보장하지 않는다. 미정규화 목표에는 [자기정규화 중요도 샘플링](self_normalized.md)을 구분해 사용한다.

**참고 문헌**

1. Kahn, H., & Harris, T. E. (1951). "Estimation of particle transmission by random sampling." *National Bureau of Standards Applied Mathematics Series*, 12, 27-30.

2. Hammersley, J. M., & Handscomb, D. C. (1964). *Monte Carlo Methods*. Methuen.

3. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. 9장: 중요도 샘플링.

4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. 3장.

5. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2장.

6. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. 4-5장.
