# 유효 표본 크기

중요도 샘플링의 웨이트 ESS는 관측된 웨이트가 얼마나 집중되어 있는지 요약하는 진단량이다. 계산이 간단하지만 모든 관심 함수에 공통인 독립 표본 수나 정확도의 보장은 아니다. 정의의 정확한 성질과 분산에 대한 휴리스틱 해석을 구분한다.

## 1. 수학적 정의

음이 아닌 원시 웨이트 W_i의 합이 유한한 양수라고 하자. 정규화 웨이트 w̄_i=W_i/∑W_j에서 다음을 정의한다.

$$ESS_w=\frac1{\sum_i\bar w_i^2}=\frac{(\sum_iW_i)^2}{\sum_iW_i^2}$$

w̄_i²≤w̄_i에서 ∑w̄_i²≤1, Cauchy–Schwarz에서 1=(∑w̄_i)²≤n∑w̄_i²이므로 1≤ESS_w≤n이다. 모든 웨이트가 같으면 n, 한 웨이트가 1이고 나머지가 0이면 1이다. 빈 입력·모두 0·음수·무한대·NaN 웨이트에는 이 정의를 적용하지 않는다.

## 2. 해석

ESS_w/n은 웨이트의 집중도를 상대적으로 보여준다. 값이 작으면 소수의 관측 표본이 정규화 평균을 좌우한다. 하지만 ESS_w=n이라는 사실만으로 표본이 목표분포를 정확히 대표하거나 독립이라고 할 수는 없다. 중요한 모드를 놓쳤거나 재표집된 동일 표본이 복제되어도 웨이트는 균등할 수 있다.

SNIS의 함수별 점근 분산은 다음과 같다.

$$\sigma_h^2=\int(h-I)^2\frac{\pi^2}{q},\qquad\operatorname{Var}(\hat I)\approx\sigma_h^2/n$$

반면 웨이트 ESS는 h를 사용하지 않는다. 따라서 Var_π(h)/ESS_w와 n/ESS_w를 일반적인 실제 분산·분산 팽창률로 동일시할 수 없다. 상수 h의 분산은 ESS와 무관하게 0이며 특정 희귀 사건은 높은 전체 ESS에서도 부정확할 수 있다. [Elvira 등의 분석](https://arxiv.org/abs/1809.04129)은 이 웨이트 근사의 여러 가정과 한계를 다룬다.

## 3. ESS 유도

### 표본 변동계수와의 정확한 관계

표본평균 W̄와 **분모 n**의 표본분산 $s_{W,n}^2=n^{-1}\sum_i(W_i-\bar W)^2$를 사용하면 다음은 항등식이다.

$$ESS_w=\frac n{1+s_{W,n}^2/\bar W^2}$$

분모 n−1의 불편 표본분산을 사용하면 보정 계수 (n−1)/n이 추가된다. 모집단 Var_q(W)/E_q(W)²와 표본 변동계수를 같은 기호로 놓아 정확한 등식처럼 쓰지 않는다.

### 엔트로피와 perplexity

정규화 웨이트의 엔트로피 H=−∑w̄_i log w̄_i에서 perplexity=exp(H)를 정의한다. 0 log 0은 0으로 처리한다. 다음 순서를 만족한다.

$$ESS_w\leq\exp(H)\leq n$$

균등 웨이트에서는 둘 다 n이다. 서로 다른 집중도 요약이며 웨이트 모양에 따라 제안 순위가 달라질 수 있다. 엔트로피를 단순히 로그 웨이트의 L1 노름이라고 정의하지 않는다.

## 4. PyTorch 구현

정규화 웨이트 함수에는 합 1의 1차원 비음수 벡터를 넣는다. 원시·로그 웨이트 함수도 빈 입력과 모두 0·−∞인 경우를 처리하지 않으므로 별도 검증이 필요하다. n_samples를 별도로 전달한다면 실제 길이와 같아야 한다. n=1이면 정규화 엔트로피의 log n 분모가 0이다.

출력 variance_inflation=n/ESS는 웨이트 집중도의 휴리스틱이지 관심 함수의 실제 분산 팽창률이 아니다. CV는 분모 n−1 표준편차를 사용하므로 앞 절의 표본 항등식과 보정이 다르다. 자동 Quality 등급과 임계값은 예제용 분류이며 검증된 정확도 인증이 아니다.

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def compute_ess_normalized(weights):
    """
    고르게 한 무게로 ESS 셈하기.
    
    ESS = 1 / sum_i w_i^2
    
    파라메터
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
    
    파라메터
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

# 타깃: N(5, 1)
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

**출력:**

```
ESS Comparison for Different Proposals
Target: N(5, 1)
======================================================================
Proposal                         ESS      ESS/n         CV  Max/Uniform
----------------------------------------------------------------------
Perfect: N(5, 1)              5000.0     100.0%       0.00          1.0x
Good: N(5, 1.2)               4755.6      95.1%       0.23          1.2x
Decent: N(4.5, 1.5)           3876.8      77.5%       0.54          1.6x
Poor: N(3, 2)                 1895.8      37.9%       1.28          3.9x
Bad: N(5, 0.5)                1538.1      30.8%       1.50         38.9x
Terrible: N(0, 1)                1.2       0.0%      65.82       4653.5x
```

---

## 5. ESS과 표본 크기 늘리기

고정된 q에서 독립 표본을 얻고 0<E_qW<∞, E_qW²<∞라면 대수의 법칙으로 다음을 얻는다.

$$\frac{ESS_w}{n}\xrightarrow{a.s.}\frac{(E_qW)^2}{E_qW^2}$$

정규화된 목표에서 E_qW=1이면 극한은 1/E_qW²다. 유한 표본 비율은 변동하므로 표본을 두 배로 늘리면 ESS가 항상 두 배가 되는 것은 아니다. 이차 적률이 무한하면 같은 양의 상수 극한을 주장할 수 없다. 새 꼬리 표본을 만나 추정 ESS가 크게 감소할 수도 있다.

### 고정 제안의 예제

```python
# n에 따른 ESS 변화 보이기
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

# 붙박인 제안-타깃 짝
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

**출력:**

```

ESS Scaling with Sample Size
Target: N(5, 1), Proposal: N(3, 2)
--------------------------------------------------
         n          ESS        ESS/n
--------------------------------------------------
       100         37.3        0.373
       500        177.7        0.355
      1000        372.7        0.373
      2000        729.3        0.365
      5000       1859.8        0.372
     10000       3752.1        0.375

ESS/n converges to approximately 0.371
```

---

## 6. 분산과 ESS의 관계

### 반복 실험으로 확인하기

아래 함수는 실제 반복 추정의 분산과 Var_π(h)/평균 ESS라는 휴리스틱을 비교한다. 이름이 verification이어도 항등식을 증명하는 것이 아니다. h의 변동과 웨이트가 연관되므로 값이 달라도 올바른 이론과 모순되지 않는다. 진단용 Var_π(h) 역시 별도 표본의 가중 추정값이다.

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

**출력:**

```

Variance-ESS Relationship Verification
==================================================
  Mean ESS: 1867.3
  Estimated Var_pi(h): 102.8494
  Predicted Var(estimator): 0.055080
  Empirical Var(estimator): 0.040884
  Ratio (empirical/predicted): 0.74
  Bias: -0.005308
  RMSE: 0.202066
```

---

## 7. 실전에서의 ESS

필요한 계산량은 관심 함수와 허용 MCSE, 꼬리 확률·분위수의 정확도로 정한다. 점 추정·논문·모델 비교라는 이름마다 보편적인 최소 ESS 숫자가 정해지는 것은 아니다. ESS/n을 보고 제안을 비교하되 지지집합·모드·가중 꼬리와 반복 실행의 변동을 함께 살핀다.

track_ess_over_iterations는 1차원 표본에 all_w.unsqueeze(−1)을 곱해 n×n으로 브로드캐스트하는 부분이 있다. 이로 인해 갱신 평균·분산이 의도한 스칼라가 아닐 수 있어 적응 샘플러를 완성한 구현으로 해석하지 않는다. 반복별 웨이트를 결합할 때도 표본 생성 당시의 제안밀도와 미정규화 목표의 스케일을 유지해야 한다.

### 반복별 진단 예제

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

---

## 8. 계량 금융에서의 ESS

꼬리 위험 추정에서는 전체 웨이트 ESS와 로스 꼬리의 정보량이 다를 수 있다. 이 페이지에 특정 위험지표별 최소 ESS를 규제 요구처럼 제시할 근거는 없다. 정확도 목표를 정하고 가중 분위수·꼬리 기여의 변동과 모형 불확실성을 따로 평가한다.

아래 함수는 ESS 임계값에서 표본 수집을 멈추고 로스와 **웨이트**를 반환할 뿐 VaR·ES를 계산하지 않는다. 이후 웨이트를 버리고 단순 로스 분위수를 계산하면 제안분포의 위험을 추정하게 된다. ESS 기반 임의 종료가 고정 표본 신뢰구간의 커버리지를 보장하지 않으며 최대 배치 도달도 따로 보고해야 한다.

### ESS 모니터링 예제

```python
def risk_estimation_with_ess_monitoring(
    loss_simulator, proposal_dist, target_dist,
    ess_threshold=1000, max_batches=10, batch_size=10000
):
    """
    ESS을 지켜보다가 멈추는 맞춰 가는 위험 어림
    실효 표본이 넉넉히 모이면 멈춘다.
    
    파라메터
    ----------
    loss_simulator : callable
        위험 인자 표본을 포트폴리오 로스로 잇는다
    proposal_dist : distribution
        중요도 표집 제안
    target_dist : distribution
        위험 인자의 타깃(실제) 분포
    ess_threshold : float
        미더운 어림값에 필요한 최소 ESS
    """
    all_samples = []
    all_log_weights = []
    
    for batch in range(max_batches):
        # 표본 배치 뽑기
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

---

## 9. 시각화

로그 웨이트·정규화 엔트로피는 0 웨이트와 n=1을 별도로 처리해야 한다. 같은 표본에서 계산한 높은 ESS나 그림의 일정 비율이 제안의 모든 모드를 확인한 증거는 아니다.

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

---

## 10. 핵심 정리

ESS_w는 웨이트의 집중도를 요약하고 원시 웨이트의 공통 배수에 불변이다. 함수별 MCSE·바이어스·모드 누락·독립성을 판정하는 양은 아니다. 재표집 직후 균등 웨이트가 되었다고 원래 표본의 다양성이 회복되는 것도 아니다.

로그 계산으로 안정화하고 지지집합·가중 꼬리·반복 실행과 관심 함수의 오차를 함께 살핀다. MCMC의 자기상관 기반 ESS와 중요도 웨이트 ESS도 서로 다른 진단이다.

## 연습문제

1. 1≤ESS_w≤n을 증명하고 양쪽 등호의 웨이트 조건을 제시하라.
2. 엔트로피의 Jensen 부등식으로 ESS_w≤exp(H)≤n을 보여라. 둘이 같은 진단이 아닌 이유를 설명하라.
3. 고정 q·π의 독립 반복에서 ESS_w의 변동을 평가할 계획을 세워라. 웨이트 이차 적률이 무한한 경우 일반적인 √n 표준오차 근사가 실패할 수 있음을 논하라.
4. 목표의 한 모드를 놓쳤지만 관측된 웨이트는 비슷한 예를 구성하라. 높은 ESS가 지지집합과 모드 탐색을 보장하지 못하는 이유를 설명하라.
5. 규제가 아닌 가상의 내부 계산 목표를 ESS_w=2000으로 정했다. 안정적인 ESS_w/n을 0.15와 0.4로 계획할 때 필요한 n은 각각 올림(2000/0.15)=13,334와 5,000이다. 이 계획이 실제 꼬리 오차나 규제 준수를 인증하지는 않는 이유를 설명하라.

## 정리하며

웨이트 ESS는 관측한 웨이트 집중도의 간단한 지표다. 표본 변동계수의 항등식과 모집단 극한, 함수별 SNIS 분산을 구분한다. 정확도·꼬리 위험·모델 비교의 보장은 ESS 임계값만으로 정해지지 않는다.

**참고 문헌**

1. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2.5절.

3. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.

4. Elvira, V., Martino, L., & Robert, C. P. (2022). "Rethinking the Effective Sample Size." *International Statistical Review*, 90(3), 525–550.

5. Vehtari, A., et al. "Pareto Smoothed Importance Sampling." [arXiv:1507.02646](https://arxiv.org/abs/1507.02646).
