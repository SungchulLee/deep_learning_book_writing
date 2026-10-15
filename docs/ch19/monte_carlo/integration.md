# 몬테카를로 적분

몬테카를로 적분은 표본평균으로 기댓값을 추정한다. 독립 표본과 적분 가능성은 일관성을, 유한 분산은 일반적인 표준오차와 중심극한정리를 뒷받침한다. 차원에 대한 수렴 지수와 실제 분산·표본 생성 비용을 구분하고, 분산 감소 기법의 조건을 살펴본다.

## 1. 격자 방법이 왜 무너지나: 차원의 저주

### 1.1 격자 근사 되짚기

차원이 낮은 문제에서는 **격자 근사**이 정해진 수치 적분으로 사후 기댓값을 계산한다. 매개변수 공간 위의 균등 격자 $\{\theta_1, \ldots, \theta_n\}$이 주어지면 다음과 같다:

$$\mathbb{E}[f(\theta) \mid \mathcal{D}] \approx \sum_{i=1}^{n} f(\theta_i)\, p(\theta_i \mid \mathcal{D})\, \Delta\theta$$

격자를 끝없이 촘촘하게 하면 이는 정확해진다. 매개변수가 1-2개면 격자 근사이 단순하고 잘 듣는다.

### 1.2 지수로 커지기

근본 문제는 차원마다 점이 $G$개인 격자가 $d$차원에서 값 매기기를 모두 $G^d$번 요구한다는 것이다:

| 차원 | 격자점($G = 100$) | 기억 공간(float64) | 할 만한가 |
|------------|------------------------|-------------------|-------------|
| 1 | $10^2$ | 0.8 KB | 시시하다 |
| 2 | $10^4$ | 80 KB | 쉽다 |
| 3 | $10^6$ | 8 MB | 할 만하다 |
| 5 | $10^{10}$ | 80 GB | 실전에서 못 쓴다 |
| 10 | $10^{20}$ | — | 불가능하다 |
| 100 | $10^{200}$ | — | 터무니없다 |

### 1.3 수렴 속도의 무너짐

차원마다 G개 점인 d차원 텐서 격자의 전체 점 수를 n=G^d라 하자. 유계 영역에서 충분히 매끄러운 피적분 함수와 r차 구적법을 가정하면, G에 대한 O(G^(−r)) 오차를 전체 평가 수 n에 대해 O(n^(−r/d))로 표현할 수 있다. 무한 영역의 절단 오차·함수 구조·상수도 고려해야 한다.

$$\text{Error}_{\text{grid}} = O(n^{-r/d})$$

여기서 $r$은 매끄러움의 차수이다(보통 $r = 2$). 수렴 속도는 차원이 커지면 빠르게 무너진다:

| 차원 | 수렴 속도 | 오차 $10^{-3}$에 필요한 점 |
|-----------|-----------------|---------------------------|
| 1 | $O(n^{-2})$ | 약 32 |
| 2 | $O(n^{-1})$ | 약 1,000 |
| 5 | $O(n^{-2/5})$ | 약 $10^{7.5}$ |
| 10 | $O(n^{-1/5})$ | 약 $10^{15}$ |

이 표는 정해진 구적법과 숨긴 오차 상수를 가정한 규모 비교다. 모든 적분법의 근본적인 차원 하한은 아니다. 희소 격자·준몬테카를로·문제의 낮은 유효 차원 등으로 더 좋은 결과를 얻을 수 있다.

---

## 2. 몬테카를로 원리

### 2.1 기본 생각

$X_1, X_2, \ldots, X_n \overset{\text{i.i.d.}}{\sim} p(x)$이면 표본 평균이 모집단 기댓값으로 모인다:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(X_i) \xrightarrow{a.s.} \mathbb{E}_p[f(X)] = I$$

$\mathbb{E}_p[|f(X)|] < \infty$이기만 하면 **큰 수의 강한 법칙**이 이를 보장한다.

### 2.2 편향 없음

몬테카를로 추정량는 표본 크기가 얼마든 편향이 없다:

$$\mathbb{E}[\hat{I}_n] = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}[f(X_i)] = \mathbb{E}_p[f(X)] = I$$

### 2.3 분산

독립 표본이며 Var_p(f(X))=σ_f²<∞이면 표본평균의 분산은 다음과 같다. 유한한 절대 1차 적률만으로 이 분산이나 √n 중심극한정리가 보장되지는 않는다.

$$\text{Var}(\hat{I}_n) = \frac{\sigma_f^2}{n}$$

여기서 $\sigma_f^2 = \text{Var}_p(f(X)) = \mathbb{E}_p[f(X)^2] - I^2$이다.

그러므로 표준 오차는 다음과 같다:

$$\text{SE}(\hat{I}_n) = \frac{\sigma_f}{\sqrt{n}}$$

### 2.4 중심 극한 정리

독립·동일분포 표본과 유한 분산 조건에서 다음 중심극한정리가 성립한다. 극단적으로 비대칭이거나 드문 사건에서는 유한 표본의 정규 근사가 느릴 수 있다.

$$\sqrt{n}\,(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$$

이는 근사 $(1-\alpha)$ 신뢰 구간을 준다:

$$\hat{I}_n \pm z_{1-\alpha/2} \cdot \frac{\hat{\sigma}_f}{\sqrt{n}}$$

여기서 $\hat{\sigma}_f^2 = \frac{1}{n-1} \sum_{i=1}^{n} (f(X_i) - \hat{I}_n)^2$은 표본 분산이다.

### 2.5 차원과 상관없는 수렴 속도

몬테카를로의 수렴 속도는 **차원과 상관없이** $O(n^{-1/2})$이다. 이것이 격자 방법에 견준 핵심 이점이다:

| 방법 | 수렴 속도 | 10차원 문제 | 100차원 문제 |
|--------|-----------------|-------------|--------------|
| 격자(사다리꼴) | $O(n^{-2/d})$ | $O(n^{-0.2})$ | $O(n^{-0.02})$ |
| 격자(심프슨) | $O(n^{-4/d})$ | $O(n^{-0.4})$ | $O(n^{-0.04})$ |
| **몬테카를로** | $O(n^{-1/2})$ | $O(n^{-0.5})$ | $O(n^{-0.5})$ |

유한 분산인 독립 표본평균의 RMSE 지수는 차원을 직접 포함하지 않는다. 하지만 σ_f²와 표본 생성·함수 평가 비용은 차원에 따라 크게 증가할 수 있다. 특정 차원 이상에서 항상 몬테카를로가 구적법보다 우수하다는 보장은 없다.

---

## 3. 구현

### 3.1 기본 몬테카를로 추정량

아래 함수는 표본마다 스칼라 f값 하나인 길이 n_samples 벡터와 독립 표본을 가정한다. 다차원 결과를 그대로 mean하면 표본·출력 축을 함께 평균하고 분모도 맞지 않을 수 있다. 불편 표본분산에는 적어도 두 표본이 필요하다. MCMC의 상관 표본에는 이 독립 표본 SE를 그대로 사용하지 않는다.

```python
import torch
import torch.distributions as dist

def monte_carlo_estimate(
    f: callable,
    sampler: callable,
    n_samples: int,
    return_diagnostics: bool = False,
) -> tuple:
    """
    몬테카를로 적분으로 E_p[f(X)] 어림하기.

    매개변수
    ----------
    f : callable
        적분할 함수, f: Tensor -> Tensor.
    sampler : callable
        p에서 독립 동일 분포 표본 n개를 뽑는 함수, sampler: int -> Tensor.
    n_samples : int
        몬테카를로 표본의 개수.
    return_diagnostics : bool
        True이면 진단을 더 돌려준다.

    반환값
    -------
    estimate : Tensor
        E_p[f(X)]의 몬테카를로 어림값.
    se : Tensor
        어림한 표준 오차.
    diagnostics : dict (optional)
        표본, 함수값, 흩어짐 어림값.
    """
    samples = sampler(n_samples)
    f_values = f(samples)

    estimate = torch.mean(f_values)
    variance = torch.var(f_values, unbiased=True)
    se = torch.sqrt(variance / n_samples)

    if return_diagnostics:
        return estimate, se, {
            "samples": samples,
            "f_values": f_values,
            "variance": variance,
        }
    return estimate, se
```

### 3.2 보기: 표준 정규 아래에서 E[X²] 근사하기

```python
torch.manual_seed(42)

# X ~ N(0,1)일 때의 E[X²]: 참값 = 1.0
normal = dist.Normal(0.0, 1.0)
f = lambda x: x ** 2
sampler = lambda n: normal.sample((n,))

estimate, se = monte_carlo_estimate(f, sampler, n_samples=10_000)
print(f"MC estimate of E[X²]: {estimate.item():.6f}")
print(f"Standard error:       {se.item():.6f}")
print(f"True value:           1.000000")
print(f"95% CI: [{estimate.item() - 1.96*se.item():.6f}, "
      f"{estimate.item() + 1.96*se.item():.6f}]")
```

**출력:**

```
MC estimate of E[X²]: 1.008879
Standard error:       0.014036
True value:           1.000000
95% CI: [0.981368, 1.036390]
```

```
MC estimate of E[X²]: 0.993754
Standard error:       0.014068
True value:           1.000000
95% CI: [0.966181, 1.021327]
```

### 3.3 수렴 보여 주기

```python
import numpy as np

torch.manual_seed(42)

sample_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
true_value = 1.0

print(f"{'n':>10} {'Estimate':>12} {'SE':>10} {'|Error|':>10} {'n*SE²':>10}")
print("-" * 56)

for n in sample_sizes:
    est, se = monte_carlo_estimate(f, sampler, n)
    error = abs(est.item() - true_value)
    # n * SE²은 거의 상수여야 한다(= σ²)
    print(f"{n:>10} {est.item():>12.6f} {se.item():>10.6f} "
          f"{error:>10.6f} {n * se.item()**2:>10.4f}")
```

**출력:**

```
         n     Estimate         SE    |Error|      n*SE²
--------------------------------------------------------
       100     0.966770   0.115516   0.033230     1.3344
       500     0.900160   0.059029   0.099840     1.7422
      1000     1.010788   0.043040   0.010788     1.8524
      5000     1.014046   0.019779   0.014046     1.9560
     10000     0.990151   0.014165   0.009849     2.0065
     50000     1.003258   0.006366   0.003258     2.0266
    100000     0.997795   0.004472   0.002205     2.0003
```

```
         n     Estimate         SE     |Error|      n*SE²
--------------------------------------------------------
       100     1.011693   0.145832   0.011693     2.1267
       500     0.975553   0.061479   0.024447     1.8898
      1000     1.021448   0.045076   0.021448     2.0319
      5000     1.003217   0.019920   0.003217     1.9841
     10000     0.993754   0.014068   0.006246     1.9791
     50000     1.001812   0.006323   0.001812     1.9999
    100000     0.999421   0.004474   0.000579     2.0014
```

곱 $n \cdot \text{SE}^2 \approx \sigma_f^2 \approx 2$이 상수로 남아 $O(n^{-1/2})$ 수렴을 확인해 준다.

---

## 4. 베이즈 사후 기댓값

### 4.1 사후 기댓값 추정

베이즈 추론에서 우리는 다음을 셈하려 한다:

$$\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

표본 $\theta^{(1)}, \ldots, \theta^{(n)} \sim p(\theta \mid \mathcal{D})$을 뽑을 수 있으면 다음과 같다:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(\theta^{(i)}) \approx \mathbb{E}[f(\theta) \mid \mathcal{D}]$$

아래 함수 선택으로 사후 요약량을 추정한다. 사후분산은 미지 사후평균을 표본평균으로 대체하므로 단순한 고정 f의 평균과는 다른 추정량이다. 독립 표본의 불편 표본분산, 누적분포의 지시함수 평균과 그 역함수인 분위수의 차이도 구분한다.

| $f(\theta)$ | 근사하는 양 |
|-------------|-------------------|
| $\theta$ | 사후평균 |
| $(\theta - \bar{\theta})^2$ | 사후분산 |
| $\mathbb{1}(\theta \in A)$ | 영역 A의 사후확률 |
| $p(x_{\text{new}} \mid \theta)$ | 사후예측밀도 |

### 4.2 근본적인 어려움

기본 몬테카를로 방법은 **목표 $p(\theta \mid \mathcal{D})$에서 곧바로 표집하기**를 요구한다. 다음일 때 이것이 말썽이 된다:

1. 사후분포에 다룰 수 없는 정규화 상수 $p(\mathcal{D})$이 있을 때
2. 사후분포의 함수 꼴에 맞는 표준 샘플링 알고리즘이 없을 때
3. 사후분포의 봉우리가 여럿이거나 기하가 복잡할 때

기각·중요도 샘플링과 MCMC는 다른 분포에서 표본을 만들거나 커널을 구성해 이 계산을 돕는다. 정규화 상수를 모른다는 사실만으로 샘플링이 언제나 불가능한 것은 아니며 각 방법의 지지집합·포락·기약성과 계산 조건이 중요하다. 중요도 샘플링은 제안표본에 가중치를 주는 적분법이므로 표본 자체가 목표분포의 독립 표본으로 바뀌는 것은 아니다.

### 4.3 보기: 격자 샘플링으로 구하는 사후평균

아래 표본은 정확한 연속 사후분포가 아니라 [0.001,0.999] 격자 위 이산 근사에서 나온다. 표본 수 증가로 몬테카를로 오차를 줄여도 범위 절단과 격자 오차는 남는다.

(차원이 낮아) 격자 위에서 사후분포를 얻을 수 있으면 그 이산 근사에서 표본을 뽑을 수 있다:

```python
import torch
import numpy as np
from scipy.stats import beta, binom

torch.manual_seed(42)

# 베타-이항 모형: 10번 던져 앞면 7번, 고른 앞확률
n_flips, n_heads = 10, 7
n_grid = 1000

theta_grid = torch.linspace(0.001, 0.999, n_grid)
grid_width = (theta_grid[1] - theta_grid[0]).item()

# 격자 위에서 뒤확률 셈하기
prior = torch.ones(n_grid)  # 고른 앞확률
log_lik = n_heads * torch.log(theta_grid) + (n_flips - n_heads) * torch.log(1 - theta_grid)
log_lik -= log_lik.max()
likelihood = torch.exp(log_lik)

unnorm_posterior = prior * likelihood
posterior = unnorm_posterior / (unnorm_posterior.sum() * grid_width)

# 격자 뒤확률에서 몬테카를로 표본 뽑기
weights = unnorm_posterior / unnorm_posterior.sum()
sample_indices = torch.multinomial(weights, num_samples=10_000, replacement=True)
theta_samples = theta_grid[sample_indices]

# 몬테카를로로 구한 뒤확률 간추림
mc_mean = theta_samples.mean()
mc_std = theta_samples.std()
analytical_mean = (1 + n_heads) / (2 + n_flips)  # Beta(8, 4)의 평균

print(f"MC Posterior Mean:        {mc_mean.item():.6f}")
print(f"Analytical Posterior Mean: {analytical_mean:.6f}")
print(f"MC Posterior Std:         {mc_std.item():.6f}")
print(f"95% Credible Interval:   [{torch.quantile(theta_samples, 0.025).item():.4f}, "
      f"{torch.quantile(theta_samples, 0.975).item():.4f}]")
```

**출력:**

```
MC Posterior Mean:        0.668122
Analytical Posterior Mean: 0.666667
MC Posterior Std:         0.130086
95% Credible Interval:   [0.3936, 0.8881]
```

```
MC Posterior Mean:        0.666834
Analytical Posterior Mean: 0.666667
MC Posterior Std:         0.130212
95% Credible Interval:   [0.3934, 0.8919]
```

---

## 5. 분산 감소 기법

$O(n^{-1/2})$ 수렴 속도는 차원과 상관없지만 그 자체로는 느릴 수 있다. 정밀도를 10배 높이려면 표본이 100배 필요하다. 분산 감소 기법은 표본 개수를 늘리지 않고 $\sigma_f^2$을 줄여 효율을 낫게 한다.

### 5.1 제어변량

E[g]는 실제 사용하는 동일 분포 아래의 알려진 기댓값이어야 한다. 아래 구현은 c를 같은 표본에서 추정하므로 일반적으로 유한 표본의 불편성이 자동 유지되지 않으며 잔차 SE도 계수 추정의 영향을 정확히 반영하지 않는다. 독립 파일럿 표본이나 교차 적합을 고려할 수 있다. 공분산은 n, 분산은 n−1로 나누어 c의 분모 관례도 다르다.

$f$과 얽힌 어떤 함수 $g$에 대해 $\mathbb{E}_p[g(X)] = \mu_g$을 정확히 알면 **제어변량 추정량**는 다음과 같다:

$$\hat{I}_{\text{CV}} = \frac{1}{n} \sum_{i=1}^{n} \bigl[f(X_i) - c\,(g(X_i) - \mu_g)\bigr]$$

유한한 이차 적률과 Var(g)>0에서 고정 계수의 최적값은 $c^*=\operatorname{Cov}(f,g)/\operatorname{Var}(g)$다. 이 값을 정확히 알고 사용하는 경우 다음 분산을 얻는다.

$$\text{Var}(\hat{I}_{\text{CV}}) = \frac{\sigma_f^2(1 - \rho_{fg}^2)}{n}$$

여기서 $\rho_{fg}$은 $f(X)$과 $g(X)$ 사이의 상관이다.

```python
def control_variate_estimate(
    f: callable,
    g: callable,
    g_mean: float,
    sampler: callable,
    n_samples: int,
) -> tuple:
    """
    다스림 변량으로 흩어짐을 줄인 몬테카를로 어림값.

    매개변수
    ----------
    f : callable
        적분할 과녁 함수.
    g : callable
        평균 g_mean을 아는 다스림 변량 함수.
    g_mean : float
        알려진 기댓값 E_p[g(X)].
    sampler : callable
        p에서 독립 동일 분포 표본 n개를 뽑는다.
    n_samples : int
        표본의 개수.

    반환값
    -------
    estimate : Tensor
        E_p[f(X)]의 다스림 변량 어림값.
    se : Tensor
        어림한 표준 오차.
    """
    samples = sampler(n_samples)
    f_vals = f(samples)
    g_vals = g(samples)

    # 가장 좋은 계수
    cov_fg = torch.mean((f_vals - f_vals.mean()) * (g_vals - g_vals.mean()))
    var_g = torch.var(g_vals, unbiased=True)
    c_star = cov_fg / var_g

    # 다듬은 값
    adjusted = f_vals - c_star * (g_vals - g_mean)
    estimate = torch.mean(adjusted)
    se = torch.sqrt(torch.var(adjusted, unbiased=True) / n_samples)

    return estimate, se
```

```python
torch.manual_seed(42)

# X ~ N(0,1)일 때 E[exp(X)] 어림하기, 참값 = exp(0.5) ≈ 1.6487
normal = dist.Normal(0.0, 1.0)
f = lambda x: torch.exp(x)
g = lambda x: x  # 다스림 변량: N(0,1)에서 E[X] = 0
sampler = lambda n: normal.sample((n,))

# 표준 몬테카를로
est_mc, se_mc = monte_carlo_estimate(f, sampler, 10_000)

# 다스림 변량 몬테카를로
est_cv, se_cv = control_variate_estimate(f, g, g_mean=0.0, sampler=sampler, n_samples=10_000)

print(f"True value:    {np.exp(0.5):.6f}")
print(f"Standard MC:   {est_mc.item():.6f} (SE = {se_mc.item():.6f})")
print(f"Control Var:   {est_cv.item():.6f} (SE = {se_cv.item():.6f})")
print(f"Variance reduction: {(1 - (se_cv/se_mc)**2).item():.1%}")
```

**출력:**

```
True value:    1.648721
Standard MC:   1.661973 (SE = 0.021584)
Control Var:   1.635681 (SE = 0.012891)
Variance reduction: 64.3%
```

```
True value:    1.648721
Standard MC:   1.649844 (SE = 0.023118)
Control Var:   1.648553 (SE = 0.013247)
Variance reduction: 67.2%
```

### 5.2 대조변량

대칭 분포에서는 표본 $X_i$마다 거울상 $-X_i$과 짝짓는다:

$$\hat{I}_{\text{AV}} = \frac{1}{2n} \sum_{i=1}^{n} \bigl[f(X_i) + f(-X_i)\bigr]$$

분산은 다음과 같다:

$$\text{Var}(\hat{I}_{\text{AV}}) = \frac{\sigma_f^2 + \text{Cov}(f(X), f(-X))}{2n}$$

f가 단조이고 X와 −X가 같은 분포라면 공분산은 0 이하이므로 같은 2n회 함수 평가의 독립 MC보다 분산이 크지 않다. 퇴화·상수 함수에서는 등호일 수 있다. 짝 안의 상관을 반영해 표준오차는 독립인 n개 짝평균의 표본분산에서 구한다. 임의의 비단조 함수에서는 오히려 분산이 커질 수 있다.

```python
def antithetic_estimate(
    f: callable,
    sampler: callable,
    n_pairs: int,
) -> tuple:
    """
    대립 변량을 쓴 몬테카를로 어림값.

    매개변수
    ----------
    f : callable
        적분할 함수.
    sampler : callable
        대칭 분포에서 독립 동일 분포 표본 n개를 뽑는다.
    n_pairs : int
        표본 짝의 개수(전체 표본 = 2 * n_pairs).

    반환값
    -------
    estimate : Tensor
        대립 변량 어림값.
    se : Tensor
        어림한 표준 오차.
    """
    samples = sampler(n_pairs)
    pair_means = 0.5 * (f(samples) + f(-samples))

    estimate = torch.mean(pair_means)
    se = torch.sqrt(torch.var(pair_means, unbiased=True) / n_pairs)
    return estimate, se
```

### 5.3 층화 샘플링

표본 공간을 확률이 $p_k = P(X \in S_k)$인 층 $S_1, \ldots, S_K$ $K$개로 나눈다. 층마다 표본 $n_k$개를 뽑는다:

$$\hat{I}_{\text{strat}} = \sum_{k=1}^{K} p_k \hat{I}_k, \quad \text{where } \hat{I}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} f(X_i^{(k)})$$

층 내 독립 표본과 알려진 층 확률에서 분산은 $\sum_kp_k^2\sigma_k^2/n_k$다. 총 n개를 n_k=np_k로 비례 배분하면 같은 비용의 단순 MC보다 분산이 크지 않으며 층별 평균이 모두 같을 때 등호다. 임의의 배분에서는 이 보장이 없고, σ_k를 안다면 n_k∝p_kσ_k의 Neyman 배분을 고려할 수 있다.

### 5.4 분산 감소 방법 간추림

| 방법 | 같은 비용 비교의 핵심 | 조건 |
|---|---|---|
| 제어변량 | 최적 고정 계수에서 분산 비율 1−ρ², 감소율 ρ² | 알려진 E[g], 유한 이차 적률 |
| 대조변량 | 짝 공분산이 음이면 개선, 이상적으로 100% 감소 가능 | 같은 주변분포의 짝과 평가 비용 일치 |
| 층화 샘플링 | 비례 배분 또는 적절한 최적 배분 | 층 확률·조건부 표본·정수 배분 고려 |
| 중요도 샘플링 | 적분에 맞는 제안에서 개선 가능, 악화도 가능 | 지지집합·가중 이차 적률 |

---

## 6. 수렴 진단

### 몬테카를로 오차

독립 표본과 유한 분산 조건에서 표본분산으로 표준오차를 추정한다. 정규 근사 신뢰구간은 수치 적분값의 반복 표본 오차에 대한 구간이며, 매개변수의 사후 신용구간과 다르다.

```python
def mc_diagnostics(f_values: torch.Tensor, target_se: float = None) -> dict:
    """
    몬테카를로 진단 셈하기.

    매개변수
    ----------
    f_values : Tensor
        값을 매긴 함수값 f(X_1), ..., f(X_n).
    target_se : float, optional
        바라는 표준 오차. 더 필요한 표본 수를 돌려준다.

    반환값
    -------
    어림값, 표준 오차, 믿음 구간, 그리고 표본 크기 권고를 담은 사전.
    """
    n = len(f_values)
    estimate = torch.mean(f_values)
    variance = torch.var(f_values, unbiased=True)
    se = torch.sqrt(variance / n)

    result = {
        "estimate": estimate.item(),
        "se": se.item(),
        "ci_95": (
            estimate.item() - 1.96 * se.item(),
            estimate.item() + 1.96 * se.item(),
        ),
        "n_samples": n,
        "relative_se": abs(se.item() / estimate.item()) if estimate.item() != 0 else float("inf"),
    }

    if target_se is not None:
        n_required = int(np.ceil(variance.item() / target_se ** 2))
        result["n_additional_needed"] = max(0, n_required - n)

    return result
```


### 표본 수 계획

평균 I≠0에서 상대 표준오차를 ε 이내로 계획하려면 대략 n≥σ_f²/(ε²I²)가 필요하다. 희귀 사건 확률 p의 직접 지시함수 평균은 상대 SE가 √((1−p)/(np))이므로 ε=0.1일 때 n≥100(1−p)/p 정도다. 이는 95% 상대 오차 반폭 10%와는 다른 기준이며 후자에는 정규 분위수 제곱도 곱한다.

표본 수를 두 배로 하면 SE는 약 1/√2배이고 절반으로 줄이려면 네 배가 필요하다. 사건이 한 번도 관측되지 않아 표본 SE가 0이어도 참확률이 0이라는 뜻은 아니다. 단순 고정 표본 신뢰구간을 반복 감시하며 임의 종료할 때의 커버리지도 별도 검토한다.

## 7. 수치 적분을 위한 몬테카를로

베이즈 사후분포를 넘어 몬테카를로 적분은 어떤 적분에도 쓸 수 있다. "맞히거나 빗나가거나" 꼴이 기본 원리를 보여 준다.

### 7.1 몬테카를로로 원주율 계산

```python
torch.manual_seed(42)

def estimate_pi(n_samples: int) -> tuple:
    """단위 정사각형 위의 몬테카를로 적분으로 π 어림하기."""
    x = torch.rand(n_samples)
    y = torch.rand(n_samples)

    # 지시: 점이 단위원의 사분면 안에 떨어짐
    inside = (x ** 2 + y ** 2 <= 1.0).float()

    # 사분원 넓이 = π/4, 단위 정사각형 넓이 = 1
    pi_estimate = 4.0 * inside.mean()
    se = 4.0 * inside.std() / np.sqrt(n_samples)
    return pi_estimate.item(), se.item()

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    pi_est, se = estimate_pi(n)
    print(f"n={n:>10}: π ≈ {pi_est:.6f}  (SE={se:.6f}, error={abs(pi_est - np.pi):.6f})")
```

**출력:**

```
n=       100: π ≈ 3.120000  (SE=0.166533, error=0.021593)
n=      1000: π ≈ 3.120000  (SE=0.052425, error=0.021593)
n=     10000: π ≈ 3.137600  (SE=0.016450, error=0.003993)
n=    100000: π ≈ 3.136960  (SE=0.005203, error=0.004633)
n=   1000000: π ≈ 3.144672  (SE=0.001640, error=0.003079)
```

```
n=       100: π ≈ 3.120000  (SE=0.164966, error=0.021593)
n=     1,000: π ≈ 3.164000  (SE=0.052196, error=0.022407)
n=    10,000: π ≈ 3.130800  (SE=0.016419, error=0.010793)
n=   100,000: π ≈ 3.142480  (SE=0.005201, error=0.000887)
n= 1,000,000: π ≈ 3.142240  (SE=0.001644, error=0.000647)
```

### 7.2 일반 적분

$I = \int_a^b g(x)\, dx$을 셈하려면 $[a, b]$ 위 균등 분포 아래의 기댓값으로 고쳐 쓴다:

$$I = (b - a)\, \mathbb{E}_{U(a,b)}[g(X)]$$

그리고 다음으로 근사한다:

$$\hat{I}_n = \frac{b - a}{n} \sum_{i=1}^{n} g(X_i), \quad X_i \overset{\text{i.i.d.}}{\sim} U(a, b)$$

---

## 8. 한계와 앞으로 갈 길

기본 몬테카를로에는 평가할 기대값에 맞는 표본 생성과 유한한 적률이 필요하다. 정규화 상수를 몰라도 기각 샘플링의 포락 조건이나 MCMC의 전이비처럼 다른 구조를 이용할 수 있지만, 그 구조를 찾는 것이 어려울 수 있다.

자기정규화 중요도 샘플링은 미정규화 목표분포의 가중 합 비로 기대값을 추정하며 유한 표본에서 일반적으로 편향이 있다. MCMC는 불변 커널과 적절한 에르고드성 아래 상관 표본의 평균을 사용한다. HMC·랑주뱅은 경사를 활용하는 선택지이며 임의의 고차원·다봉 문제를 자동 해결하지 않는다. 방법의 이름보다 지지집합·꼬리·상관과 실제 비용을 살펴야 한다.

## 9. 금융에서의 쓰임새

옵션 가격은 지정한 위험중립 모형 아래 할인된 지급액의 기댓값으로 계산할 수 있다. 경로별 지급액의 표본을 사용하며, 위험중립 측도와 실제 손실 확률을 평가하는 측도는 구분한다. 분산 감소에는 동일한 경로에서 상관된 지급액을 구성하고 그 기대값을 알아야 한다.

VaR는 손실분포의 분위수이고, Expected Shortfall은 분위수의 꼬리 평균으로 정의한다. 이산 손실의 VaR 지점에 질량이 있으면 단순히 L>VaR 또는 L≥VaR인 표본만 평균하는 계산과 다를 수 있다. 희귀 사건의 상대 오차와 꼬리의 표본 부족을 확인한다.

베이즈 의사결정은 사후예측분포 아래 기대 효용을 최적화한다. 각 매개변수에서 구한 최적 비중 w*(θ)를 단순 평균하는 것과는 일반적으로 다르다. 확률 변동성의 잠재 경로에는 MCMC·입자 방법 등을 사용할 수 있지만 모형과 계산 오차가 남는다.

## 10. 핵심 되새김

독립 표본평균은 적분 가능성 아래 일관적이고 유한 분산 아래 RMSE가 1/√n 크기다. 차원이 직접 지수에 나타나지 않아도 분산과 표본 생성 비용은 커질 수 있다. 표준오차·신뢰구간은 모멘트와 표본 독립·정규 근사 조건을 확인한 뒤 해석한다.

제어변량·대조변량·층화·중요도 방법은 적절한 조건에서 분산을 줄인다. 같은 함수 평가 비용으로 비교하고, 계수 추정·배분·짝 상관·지지집합을 명시한다. 격자 오차·모형 오차·몬테카를로 오차를 분리해야 한다.

## 연습문제

1. X∼N(0,1), f(X)=X²에서 E[f]=1, Var(f)=2를 이용해 n별 이론 SE를 구하라. 표본 수를 두 배·네 배로 늘릴 때 변화를 설명하고 한 번의 실현 오차가 단조 감소할 필요는 없음을 논하라.
2. 옵션의 경로별 지급액 Y와, 같은 난수 경로에서 계산한 다른 행사가 지급액 G를 제어변량으로 사용하는 절차를 설계하라. 블랙-숄즈 공식이 주는 것은 E[G]라는 상수이며 G 자체는 확률변수임을 구분하라.
3. d차원 정규분포의 E[||X||²]=d에서 격자와 MC의 비용을 비교할 때 범위 절단·차원별 격자 수·전체 평가 수·분산 상수를 어떻게 맞출지 설명하라.
4. X∼N(0,1)에서 exp(X)와 exp(−X)의 대조 짝을 구성하라. 같은 2n회 함수 평가의 독립 MC와 비교한 분산 비율을 공분산으로 유도하라.
5. 사건확률 p≈0.00135의 지시함수 평균에서 상대 SE 10%와 정규 근사 95% 상대 반폭 10%에 필요한 n을 각각 계획하라. 관측 사건이 0개인 경우 Wald 구간의 문제와 적절한 이항 구간의 필요성을 설명하라.

## 정리하며

몬테카를로 적분의 일관성·분산·신뢰구간에는 각각 적률과 표본 조건이 있다. √n의 오차율을 모든 고차원 계산의 쉬움으로 해석하지 않는다. 분산 감소의 조건과 실제 함수 평가 비용을 맞추고, 수치 적분의 신뢰구간과 매개변수 신용구간을 구분한다.

**참고 문헌**

1. Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer. 3-4장.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2-3장.
3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. 10장.
4. Owen, A. B. (2013). *Monte Carlo Theory, Methods, and Examples*. https://artowen.su.domains/mc/.
5. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
6. McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. 9장.
