# 실효 표본 크기
## 개요

실효 표본 크기(ESS)는 중요도 표집의 질을 재는 근본 진단이다. 우리가 가진 무게 준 표본이 과녁 분포의 독립 표본 몇 개에 맞먹는지를 잰다. ESS은 제안의 질을 객관적으로 재어 주며 중요도 표집 어림자의 흩어짐과 곧바로 이어진다.

## 수학적 정의

### 고르게 한 무게에 대한 정의

$\sum_i \bar{w}_i = 1$인 고르게 한 중요도 무게 $\bar{w}_1, \ldots, \bar{w}_n$이 주어졌을 때:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^n \bar{w}_i^2}
$$

### 고르게 하지 않은 무게에 대한 정의

고르게 하지 않은 무게 $\tilde{w}_1, \ldots, \tilde{w}_n$에 대해:

$$
\text{ESS} = \frac{\left(\sum_{i=1}^n \tilde{w}_i\right)^2}{\sum_{i=1}^n \tilde{w}_i^2}
$$

두 정의는 같다. 곧 $\bar{w}_i = \tilde{w}_i / \sum_j \tilde{w}_j$이다.

### 성질

**한계:**

$$
1 \leq \text{ESS} \leq n
$$

**극단적인 경우:**

- **최댓값**: 무게가 모두 같을 때 ESS $= n$이다. 곧 $\bar{w}_i = 1/n$이다
- **최솟값**: 무게 하나가 1이고 나머지가 0일 때 ESS $= 1$이다

## 해석

### 직관적인 이해

ESS은 이렇게 답한다. "우리가 가진 무게 준 표본 $n$개는 $\pi$의 독립 표본 몇 개에 맞먹는가?"

| ESS | 풀이 | 뜻하는 바 |
|-----|----------------|-------------|
| $\text{ESS} = n$ | 완벽한 표집 | 무게가 고르며 $\pi$에서 표집한 것과 같다 |
| $\text{ESS} = n/2$ | 효율 50% | 표본의 절반이 "버려진다" |
| $\text{ESS} = n/10$ | 효율 10% | 같은 정밀도에 표본이 10배 필요하다 |
| $\text{ESS} \ll n$ | 심한 주저앉음 | 몇몇 표본이 좌우하며 어림값이 미덥지 않다 |
| $\text{ESS} \approx 1$ | 온통 무너짐 | 사실상 표본 하나라 쓸모없다 |

### 흩어짐과의 이음

스스로 고르게 하는 중요도 표집 어림자의 흩어짐은 대략 다음과 같다:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{\text{Var}_\pi(h(\theta))}{\text{ESS}}
$$

독립 표본 $n$개를 쓴 표준 몬테카를로와 견주어 보자:

$$
\text{Var}(\hat{I}_{\text{MC}}) = \frac{\text{Var}_\pi(h(\theta))}{n}
$$

**흩어짐 부풀림 인자:**

$$
\text{Variance Inflation} = \frac{n}{\text{ESS}}
$$

ESS $= n/10$이면 흩어짐이 완벽한 표집에 견주어 10배로 부푼다.

## ESS 이끌어 내기

### 무게의 흩어짐에서

무게의 변동 계수에서 시작하자:

$$
\text{CV}^2(\tilde{w}) = \frac{\text{Var}(\tilde{w})}{[\mathbb{E}(\tilde{w})]^2}
$$

ESS은 다음과 같이 쓸 수 있다:

$$
\text{ESS} = \frac{n}{1 + \text{CV}^2(\tilde{w})}
$$

**증명:**

$$
\text{ESS} = \frac{(\sum_i \tilde{w}_i)^2}{\sum_i \tilde{w}_i^2} = \frac{n^2 \bar{\tilde{w}}^2}{n \cdot (\text{Var}(\tilde{w}) + \bar{\tilde{w}}^2)}
$$

여기서 $\bar{\tilde{w}} = \frac{1}{n}\sum_i \tilde{w}_i$이다. 간단히 하면 다음과 같다:

$$
\text{ESS} = \frac{n}{1 + \text{Var}(\tilde{w})/\bar{\tilde{w}}^2} = \frac{n}{1 + \text{CV}^2}
$$

### 엔트로피에서

ESS은 무게 분포의 **혼란도**(엔트로피의 지수)와 이어져 있다:

$$
\text{Perplexity} = \exp(H(\bar{w})) = \exp\left(-\sum_i \bar{w}_i \log \bar{w}_i\right)
$$

고른 분포에서는 $H = \log n$, 혼란도 $= n$이다.

ESS과 혼란도는 이어져 있지만 같지는 않다:

- ESS은 $L_2$ 노름을 쓴다. 곧 $1/\|\bar{w}\|_2^2$이다
- 혼란도는 엔트로피를 쓴다(로그 무게의 $L_1$과 이어져 있다)

둘 다 무게가 몰린 정도를 재지만 강조하는 바가 다르다.

## PyTorch 구현

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def compute_ess_normalized(weights):
    """
    고르게 한 무게로 ESS 셈하기.
    
    ESS = 1 / sum_i w_i^2
    
    매개변수
    ----------
    weights : torch.Tensor
        고르게 한 무게(합이 1)
        
    반환값
    -------
    ess : float
        실효 표본 크기
    """
    return 1.0 / torch.sum(weights**2)


def compute_ess_unnormalized(unnorm_weights):
    """
    고르게 하지 않은 무게로 ESS 셈하기.
    
    ESS = (sum_i w_i)^2 / sum_i w_i^2
    
    먼저 고르게 하는 것보다 수치로 더 안정하다.
    """
    sum_w = torch.sum(unnorm_weights)
    sum_w_sq = torch.sum(unnorm_weights**2)
    return sum_w**2 / sum_w_sq


def compute_ess_log_weights(log_weights):
    """
    로그 무게로 ESS 셈하기(수치로 가장 안정하다).
    
    무게가 여러 자릿수에 걸쳐 있을 때 쓸모 있다.
    """
    # 로그 공간에서 고르게 하기
    log_sum = torch.logsumexp(log_weights, dim=0)
    log_norm_weights = log_weights - log_sum
    
    # ESS = exp(-log(sum_i exp(2 log w_i)))
    log_sum_sq = torch.logsumexp(2 * log_norm_weights, dim=0)
    log_ess = -log_sum_sq
    
    return torch.exp(log_ess)


def weight_diagnostics(weights, n_samples=None, name=""):
    """
    무게와 ESS 두루 진단하기.
    
    매개변수
    ----------
    weights : torch.Tensor
        중요도 무게(고르게 했거나 안 했거나)
    n_samples : int, optional
        표본의 전체 개수(None이면 무게에서 미루어 안다)
    name : str
        찍기 위한 이름표
        
    반환값
    -------
    dict
        진단 통계량의 사전
    """
    if n_samples is None:
        n_samples = len(weights)
    
    # 필요하면 고르게 하기
    if not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        norm_weights = weights / weights.sum()
    else:
        norm_weights = weights
    
    # ESS
    ess = compute_ess_normalized(norm_weights)
    ess_ratio = ess / n_samples
    
    # 무게 통계량
    max_weight = norm_weights.max()
    min_weight = norm_weights.min()
    uniform_weight = 1.0 / n_samples
    
    # 변이 계수
    cv = norm_weights.std() / norm_weights.mean()
    
    # 무게 몰림
    sorted_weights = torch.sort(norm_weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    
    n_for_10 = (cumsum < 0.1).sum().item() + 1
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    # 엔트로피와 헷갈림도
    entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-10))
    max_entropy = torch.log(torch.tensor(float(n_samples)))
    perplexity = torch.exp(entropy)
    
    # 흩어짐 부풂
    variance_inflation = n_samples / ess
    
    diagnostics = {
        'n_samples': n_samples,
        'ess': ess.item(),
        'ess_ratio': ess_ratio.item(),
        'variance_inflation': variance_inflation.item(),
        'cv': cv.item(),
        'max_weight': max_weight.item(),
        'max_weight_ratio': max_weight.item() / uniform_weight,
        'n_for_10_pct': n_for_10,
        'n_for_50_pct': n_for_50,
        'n_for_90_pct': n_for_90,
        'entropy': entropy.item(),
        'normalized_entropy': (entropy / max_entropy).item(),
        'perplexity': perplexity.item()
    }
    
    if name:
        print(f"\n{'='*60}")
        print(f"Weight Diagnostics: {name}")
        print(f"{'='*60}")
        print(f"  Total samples: {n_samples}")
        print(f"  ESS: {ess.item():.1f} ({ess_ratio.item():.1%} efficiency)")
        print(f"  Variance inflation: {variance_inflation.item():.1f}x")
        print(f"  CV of weights: {cv.item():.3f}")
        print(f"  Max weight: {max_weight.item():.6f} "
              f"({diagnostics['max_weight_ratio']:.1f}x uniform)")
        print(f"\n  Weight Concentration:")
        print(f"    10% weight in top {n_for_10} samples "
              f"({n_for_10/n_samples:.1%})")
        print(f"    50% weight in top {n_for_50} samples "
              f"({n_for_50/n_samples:.1%})")
        print(f"    90% weight in top {n_for_90} samples "
              f"({n_for_90/n_samples:.1%})")
        print(f"\n  Entropy: {entropy.item():.3f} "
              f"(normalized: {diagnostics['normalized_entropy']:.3f})")
        print(f"  Perplexity: {perplexity.item():.1f}")
        
        # 질 살피기
        if ess_ratio.item() > 0.5:
            quality = "EXCELLENT"
        elif ess_ratio.item() > 0.2:
            quality = "GOOD"
        elif ess_ratio.item() > 0.05:
            quality = "ACCEPTABLE"
        elif ess_ratio.item() > 0.01:
            quality = "POOR"
        else:
            quality = "FAILURE"
        
        print(f"\n  Overall Quality: {quality}")
    
    return diagnostics


# 보기: 제안의 질에 따른 ESS
torch.manual_seed(42)

# 과녁: N(5, 1)
target = dist.Normal(5.0, 1.0)

# 여러 가지 제안
proposals = {
    'Perfect: N(5, 1)': dist.Normal(5.0, 1.0),
    'Good: N(5, 1.2)': dist.Normal(5.0, 1.2),
    'Decent: N(4.5, 1.5)': dist.Normal(4.5, 1.5),
    'Poor: N(3, 2)': dist.Normal(3.0, 2.0),
    'Bad: N(5, 0.5)': dist.Normal(5.0, 0.5),  # 너무 좁음
    'Terrible: N(0, 1)': dist.Normal(0.0, 1.0),  # 틀린 자리
}

n_samples = 5000

print("ESS Comparison for Different Proposals")
print("Target: N(5, 1)")
print("=" * 70)
print(f"{'Proposal':<25} {'ESS':>10} {'ESS/n':>10} "
      f"{'CV':>10} {'Max/Uniform':>12}")
print("-" * 70)

results = {}
for name, proposal in proposals.items():
    # 표집하고 무게 셈하기
    samples = proposal.sample((n_samples,))
    log_weights = target.log_prob(samples) - proposal.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
    
    # 진단
    diag = weight_diagnostics(weights, n_samples)
    results[name] = diag
    
    print(f"{name:<25} {diag['ess']:10.1f} {diag['ess_ratio']:10.1%} "
          f"{diag['cv']:10.2f} {diag['max_weight_ratio']:12.1f}x")
```

## ESS과 표본 크기 늘리기

### ESS은 n에 따라 커지는가?

**물음**: $n$을 두 배로 하면 ESS도 두 배가 되는가?

**답**: 대체로 그렇다. 제안과 과녁의 짝이 붙박이면 ESS 비(ESS/n)가 대략 상수로 남는다.

**증명 얼개:**

$$
\text{ESS} = \frac{n}{1 + \text{CV}^2(\tilde{w})}
$$

$n$이 크면 (큰 수의 법칙에 따라) $\text{CV}^2(\tilde{w})$이 상수로 모이므로 다음과 같다:

$$
\frac{\text{ESS}}{n} \to \frac{1}{1 + \text{CV}^2_\infty}
$$

**뜻하는 바**: $n$을 늘리는 것만으로 나쁜 제안을 "고칠" 수는 없다. 효율(ESS/n)은 제안과 과녁이 얼마나 맞는지가 정한다.

```python
# n에 따른 ESS 변화 보이기
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

# 붙박인 제안-과녁 짝
target = dist.Normal(5.0, 1.0)
proposal = dist.Normal(3.0, 2.0)  # 일부러 어긋나게 함

print("\nESS Scaling with Sample Size")
print(f"Target: N(5, 1), Proposal: N(3, 2)")
print("-" * 50)
print(f"{'n':>10} {'ESS':>12} {'ESS/n':>12}")
print("-" * 50)

ess_ratios = []
for n in sample_sizes:
    samples = proposal.sample((n,))
    log_weights = target.log_prob(samples) - proposal.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
    
    ess = compute_ess_normalized(weights)
    ratio = ess / n
    ess_ratios.append(ratio.item())
    
    print(f"{n:10d} {ess.item():12.1f} {ratio.item():12.3f}")

print(f"\nESS/n converges to approximately "
      f"{sum(ess_ratios[-3:])/3:.3f}")
```

## 흩어짐과 ESS의 관계

### 겪어 보고 확인하기

```python
def verify_variance_ess_relationship(
    target_log_prob, proposal, h_function,
    true_value, n_samples=5000, n_reps=500
):
    """
    Var(어림꼴) ~ Var_pi(h)/ESS임을 경험으로 확인하기
    """
    estimates = []
    ess_values = []
    
    for _ in range(n_reps):
        samples = proposal.sample((n_samples,))
        log_weights = (target_log_prob(samples)
                       - proposal.log_prob(samples))
        weights = torch.exp(
            log_weights - torch.logsumexp(log_weights, 0)
        )
        
        estimate = torch.sum(weights * h_function(samples))
        ess = compute_ess_normalized(weights)
        
        estimates.append(estimate.item())
        ess_values.append(ess.item())
    
    estimates = torch.tensor(estimates)
    ess_values = torch.tensor(ess_values)
    
    empirical_var = estimates.var().item()
    mean_ess = ess_values.mean().item()
    mse = ((estimates - true_value)**2).mean().item()
    bias = (estimates.mean() - true_value).item()
    
    # 이론의 미리봄
    good_proposal = dist.Normal(5.0, 1.1)
    samples = good_proposal.sample((50000,))
    log_w = (target_log_prob(samples)
             - good_proposal.log_prob(samples))
    w = torch.exp(log_w - torch.logsumexp(log_w, 0))
    h_vals = h_function(samples)
    weighted_mean = torch.sum(w * h_vals)
    var_h = torch.sum(w * (h_vals - weighted_mean)**2)
    
    predicted_var = var_h.item() / mean_ess
    
    print(f"\nVariance-ESS Relationship Verification")
    print("=" * 50)
    print(f"  Mean ESS: {mean_ess:.1f}")
    print(f"  Estimated Var_pi(h): {var_h.item():.4f}")
    print(f"  Predicted Var(estimator): {predicted_var:.6f}")
    print(f"  Empirical Var(estimator): {empirical_var:.6f}")
    print(f"  Ratio (empirical/predicted): "
          f"{empirical_var/predicted_var:.2f}")
    print(f"  Bias: {bias:.6f}")
    print(f"  RMSE: {mse**0.5:.6f}")

# 확인
target_log_prob = lambda x: dist.Normal(5.0, 1.0).log_prob(x)
proposal = dist.Normal(3.0, 2.0)
h = lambda x: x**2
true_value = 5**2 + 1**2  # N(5,1)의 E[X^2]

verify_variance_ess_relationship(
    target_log_prob, proposal, h, true_value
)
```

## 실전에서의 ESS

### 최소 ESS 요구

| 쓰임새 | 최소 ESS | 까닭 |
|-------------|-------------|-----------|
| 점 어림값 | 100-500 | 기본 중심 극한 정리를 쓸 수 있음 |
| 뒤확률 분위수 | 500-1000 | 꼬리에도 밀도가 필요함 |
| 믿음 구간 | 1000 이상 | 꼬리를 정확히 덮어야 함 |
| 모형 견줌 | 1000 이상 | 무게 분포에 민감함 |
| 논문 수준 | 5000 이상 | 몬테카를로 오차가 작아야 함 |

### ESS이 낮을 때

**증상:**

- 어림값의 흩어짐이 크다
- 돌릴 때마다 결과가 흔들린다
- 무게 값이 극단적이다
- 몇몇 표본이 무게 대부분을 진다

**풀이:**

1. **제안 낫게 하기**: 더 나은 자리, 눈금, 또는 갈래
2. **알아서 맞추는 방법 쓰기**: 알고리즘이 좋은 제안을 찾게 한다
3. **MCMC으로 바꾸기**: 그 문제에 더 알맞을 수 있다
4. **$n$ 늘리기**: ESS/n이 그럭저럭일 때만 도움이 된다

**그냥 n을 늘리면 안 될 때:**

- ESS/n < 0.01이면 $n$을 두 배로 해도 ESS만 두 배가 된다
- 제안을 낫게 해 ESS/n > 0.1을 얻는 편이 낫다

### 시간에 따라 ESS 지켜보기

잇단 알고리즘이나 알아서 맞추는 알고리즘에서는 ESS의 흐름을 좇아라:

```python
def track_ess_over_iterations(
    log_target, initial_proposal, n_per_iter, n_iters
):
    """
    표본을 쌓거나 제안을 맞춰 가며 ESS 기록하기.
    """
    ess_history = []
    cumulative_ess_history = []
    
    all_samples = []
    all_log_weights = []
    
    current_proposal = initial_proposal
    
    for t in range(n_iters):
        # 표본 뽑기
        samples = current_proposal.sample((n_per_iter,))
        log_weights = (log_target(samples)
                       - current_proposal.log_prob(samples))
        
        all_samples.append(samples)
        all_log_weights.append(log_weights)
        
        # 이번 되풀이의 ESS
        weights = torch.exp(
            log_weights - torch.logsumexp(log_weights, 0)
        )
        ess_iter = compute_ess_normalized(weights)
        ess_history.append(ess_iter.item())
        
        # 쌓인 ESS(지금까지의 모든 표본)
        all_log_w = torch.cat(all_log_weights)
        all_w = torch.exp(
            all_log_w - torch.logsumexp(all_log_w, 0)
        )
        cumulative_ess = compute_ess_normalized(all_w)
        cumulative_ess_history.append(cumulative_ess.item())
        
        # 단순한 맞춰 가기: 무게 표본에 가우스 맞추기
        if t > 0 and t % 5 == 0:
            all_s = torch.cat(all_samples)
            weighted_mean = torch.sum(
                all_w.unsqueeze(-1) * all_s, dim=0
            )
            weighted_var = torch.sum(
                all_w * (all_s - weighted_mean)**2
            )
            current_proposal = dist.Normal(
                weighted_mean, 1.2 * torch.sqrt(weighted_var)
            )
        
        print(f"Iter {t+1}: ESS = {ess_iter.item():.1f}, "
              f"Cumulative ESS = {cumulative_ess.item():.1f}")
    
    return ess_history, cumulative_ess_history
```

## 계량 금융에서의 ESS

### 위험 잣대 어림을 위한 ESS

계량 금융에서 ESS은 위험 잣대 어림값이 얼마나 미더운지에 곧바로 이어진다. ESS과 VaR 및 기대 부족액 어림값의 정밀도 사이의 관계가 몬테카를로 흉내내기의 셈 예산을 이끈다:

| 위험 잣대 | 최소 ESS | 규제 자리 |
|-------------|-------------|-------------------|
| VaR(99%) | 1,000 이상 | 바젤 III 내부 모형 |
| ES(97.5%) | 2,000 이상 | FRTB 표준 방식 |
| 신용 VaR(99.9%) | 5,000 이상 | 경제적 자본 모형 |
| 스트레스 시험 | 500 이상 | CCAR/DFAST 상황 |

**실제 제품 얼개를 위한 ESS과 모임 지켜보기:**

```python
def risk_estimation_with_ess_monitoring(
    loss_simulator, proposal_dist, target_dist,
    ess_threshold=1000, max_batches=10, batch_size=10000
):
    """
    ESS을 지켜보다가 멈추는 맞춰 가는 위험 어림
    실효 표본이 넉넉히 모이면 멈춘다.
    
    매개변수
    ----------
    loss_simulator : callable
        위험 인자 표본을 포트폴리오 손실로 잇는다
    proposal_dist : distribution
        중요도 표집 제안
    target_dist : distribution
        위험 인자의 과녁(실제) 분포
    ess_threshold : float
        미더운 어림값에 필요한 최소 ESS
    """
    all_samples = []
    all_log_weights = []
    
    for batch in range(max_batches):
        # 표본 묶음 뽑기
        samples = proposal_dist.sample((batch_size,))
        log_w = (target_dist.log_prob(samples)
                 - proposal_dist.log_prob(samples))
        
        all_samples.append(samples)
        all_log_weights.append(log_w)
        
        # 쌓인 ESS 셈하기
        combined_log_w = torch.cat(all_log_weights)
        combined_w = torch.exp(
            combined_log_w
            - torch.logsumexp(combined_log_w, 0)
        )
        ess = compute_ess_normalized(combined_w)
        
        total_n = (batch + 1) * batch_size
        print(f"Batch {batch+1}: n={total_n}, "
              f"ESS={ess.item():.1f} "
              f"({ess.item()/total_n:.1%})")
        
        if ess.item() >= ess_threshold:
            print(f"ESS threshold {ess_threshold} reached.")
            break
    
    # 마지막 무게 표본으로 위험 잣대 셈하기
    combined_samples = torch.cat(all_samples)
    losses = loss_simulator(combined_samples)
    
    return losses, combined_w, ess.item()
```

## 시각화

```python
def plot_ess_diagnostics(weights, samples, name=""):
    """
    ESS을 두루 그려 보기.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n = len(weights)
    
    if not torch.isclose(
        weights.sum(), torch.tensor(1.0), atol=1e-6
    ):
        weights = weights / weights.sum()
    
    ess = compute_ess_normalized(weights)
    
    # 칸 1: 무게 막대그림
    ax = axes[0, 0]
    ax.hist(weights.numpy() * n, bins=50, density=True, 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--',
               linewidth=2, label='Uniform')
    ax.set_xlabel('Normalized Weight x n', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(
        f'Weight Distribution '
        f'(ESS={ess.item():.1f}, {ess.item()/n:.1%})', 
        fontsize=12, fontweight='bold'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 칸 2: 쌓인 무게 곡선
    sorted_w = torch.sort(weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    
    ax = axes[0, 1]
    ax.plot(torch.arange(1, n+1).numpy(),
            cumsum.numpy(), 'b-', linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--',
               alpha=0.7, label='50%')
    ax.axhline(0.9, color='orange', linestyle='--',
               alpha=0.7, label='90%')
    ax.plot(torch.arange(1, n+1).numpy(),
            torch.arange(1, n+1).numpy()/n, 
            'g:', linewidth=2, alpha=0.7,
            label='Ideal (uniform)')
    ax.set_xlabel('Number of Top Samples', fontsize=11)
    ax.set_ylabel('Cumulative Weight', fontsize=11)
    ax.set_title('Weight Concentration',
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 칸 3: 로그 무게 막대그림
    log_weights = torch.log(weights * n)
    
    ax = axes[1, 0]
    ax.hist(log_weights.numpy(), bins=50, density=True,
            alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2,
               label='log(1) = 0 (uniform)')
    ax.set_xlabel('Log(Normalized Weight x n)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Log Weight Distribution',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 칸 4: n에 따른 ESS 흉내내기
    ns = [100, 200, 500, 1000, 2000, 5000]
    ratios = []
    
    target = dist.Normal(5.0, 1.0)
    proposal = dist.Normal(3.0, 2.0)
    
    for nn in ns:
        s = proposal.sample((nn,))
        lw = target.log_prob(s) - proposal.log_prob(s)
        w = torch.exp(lw - torch.logsumexp(lw, 0))
        ratios.append(
            (compute_ess_normalized(w) / nn).item()
        )
    
    ax = axes[1, 1]
    ax.plot(ns, ratios, 'bo-', linewidth=2, markersize=8)
    ax.axhline(
        sum(ratios)/len(ratios), color='red',
        linestyle='--', linewidth=2, alpha=0.7,
        label=f'Mean = {sum(ratios)/len(ratios):.3f}'
    )
    ax.set_xlabel('Sample Size n', fontsize=11)
    ax.set_ylabel('ESS/n', fontsize=11)
    ax.set_title('ESS Ratio is Constant in n',
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(name, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
```

## 핵심 정리

!!! success "ESS이 알려 주는 것"

    - **ESS ~ n**: 아주 좋은 제안이며 무게가 거의 고르다
    - **ESS/n > 0.2**: 효율이 좋고 결과가 미덥다
    - **ESS/n < 0.05**: 효율이 나쁘니 제안을 낫게 하는 것을 생각해 보아라
    - **ESS/n < 0.01**: 어림값이 미덥지 않으니 제안을 반드시 고쳐야 한다

!!! warning "ESS이 알려 주지 않는 것"

    - 제안이 봉우리를 모두 덮는지(ESS이 높아도 봉우리를 놓칠 수 있다)
    - 표본이 받침 전체를 살펴보는지
    - 받침이 어긋나 생기는 치우침(흩어짐만 알려 준다)

!!! info "ESS을 쓰는 좋은 버릇"

    1. 중요도 표집 어림값과 함께 늘 ESS을 셈해 알려라
    2. 날 ESS보다 ESS/n이 알아보기 쉽다
    3. ESS이 낮으면 n을 늘리기보다 제안을 낫게 하여라
    4. 제안을 객관적으로 견주는 데 ESS을 써라
    5. 알아서 맞추는 방법에서는 되풀이마다 ESS을 지켜보아라

## 참고 문헌

1. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2.5절.

3. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.

4. Elvira, V., Martino, L., & Robert, C. P. (2019). "Rethinking the effective sample size." *International Statistical Review*, 87(3), 591-616.

5. Vehtari, A., Gelman, A., & Gabry, J. (2017). "Pareto smoothed importance sampling." *arXiv preprint arXiv:1507.02646*.

## 연습문제

### 연습 1: ESS의 한계
$1 \leq \text{ESS} \leq n$임을 증명하여라. 어떤 조건에서 그 한계에 이르는가?

### 연습 2: ESS과 혼란도
같은 무게 분포에 대해 ESS과 혼란도를 견주어라. 언제 제안의 질에 서로 다른 차례를 매기는가?

### 연습 3: ESS 어림의 흩어짐
ESS 자체도 표본에서 어림한 값이다. 돌릴 때마다 ESS이 얼마나 흔들리는가? ESS 어림값의 표준 오차를 셈하여라.

### 연습 4: 봉우리가 여럿일 때의 ESS
봉우리를 놓쳤는데도 ESS이 높을 수 있는가? 보기를 짓고 그것이 뜻하는 바를 이야기하여라.

### 연습 5: 위험 모형의 ESS 예산
어떤 위험 모형이 규제를 지키려면 ESS이 적어도 2000이어야 한다. 지금 제안에서 ESS/n이 대략 0.15로 관측된다. 표본이 모두 몇 개($n$) 필요한가? 제안을 낫게 해 ESS/n이 대략 0.4가 되면 필요한 $n$은 어떻게 바뀌는가? 위험 인자가 500개인 포트폴리오에서 셈 값이 어떻게 되는지 이야기하여라.
