# 몬테카를로 적분
## 개요

몬테카를로 적분은 해석으로 풀 수 없거나 정해진 방법으로는 셈이 너무 비싼 적분을 무작위 표집으로 어림한다. 이 마당은 몬테카를로 어림의 수학 바탕을 세운다. 큰 수의 법칙이 주는 보장에서 시작해 오차를 재는 중심 극한 정리까지 나아가고, 높은 차원에서 격자 기반 길을 쓸모없게 만드는 차원의 저주를 몬테카를로 방법이 어떻게 넘어서는지 보인다. 어림의 효율을 크게 끌어올리는 흩어짐 줄이기 기법도 다룬다.

베이즈 추론에서 우리가 궁금해하는 양은 거의 모두 뒤확률 아래의 기댓값이다:

$$\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

뒤확률 평균, 흩어짐, 믿음 구간, 예측 분포를 셈하는 일이 모두 이 꼴의 적분으로 줄어든다. 몬테카를로 적분이 그것을 셈하는 엔진을 준다.

---

## 1. 격자 방법이 왜 무너지나: 차원의 저주

### 1.1 격자 어림 되짚기

차원이 낮은 문제에서는 **격자 어림**이 정해진 수치 적분으로 뒤확률 기댓값을 셈한다. 매개변수 공간 위의 고른 격자 $\{\theta_1, \ldots, \theta_n\}$이 주어지면 다음과 같다:

$$\mathbb{E}[f(\theta) \mid \mathcal{D}] \approx \sum_{i=1}^{n} f(\theta_i)\, p(\theta_i \mid \mathcal{D})\, \Delta\theta$$

격자를 끝없이 촘촘하게 하면 이는 정확해진다. 매개변수가 1-2개면 격자 어림이 단순하고 잘 듣는다.

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

### 1.3 모임 속도의 무너짐

차원마다 점이 $n$개인 고른 격자와 두 번 미분할 수 있는 피적분 함수에서는 다음과 같다:

$$\text{Error}_{\text{grid}} = O(n^{-r/d})$$

여기서 $r$은 매끄러움의 차수이다(보통 $r = 2$). 모임 속도는 차원이 커지면 빠르게 무너진다:

| 차원 | 모임 속도 | 오차 $10^{-3}$에 필요한 점 |
|-----------|-----------------|---------------------------|
| 1 | $O(n^{-2})$ | 약 32 |
| 2 | $O(n^{-1})$ | 약 1,000 |
| 5 | $O(n^{-2/5})$ | 약 $10^{7.5}$ |
| 10 | $O(n^{-1/5})$ | 약 $10^{15}$ |

이는 그저 실전에서의 불편이 아니라 **수학의 근본 한계**이며, 그래서 다른 길이 꼭 필요하다.

---

## 2. 몬테카를로 원리

### 2.1 기본 생각

$X_1, X_2, \ldots, X_n \overset{\text{i.i.d.}}{\sim} p(x)$이면 표본 평균이 모집단 기댓값으로 모인다:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(X_i) \xrightarrow{a.s.} \mathbb{E}_p[f(X)] = I$$

$\mathbb{E}_p[|f(X)|] < \infty$이기만 하면 **큰 수의 강한 법칙**이 이를 보장한다.

### 2.2 치우침 없음

몬테카를로 어림자는 표본 크기가 얼마든 치우침이 없다:

$$\mathbb{E}[\hat{I}_n] = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}[f(X_i)] = \mathbb{E}_p[f(X)] = I$$

### 2.3 흩어짐

몬테카를로 어림자의 흩어짐은 다음과 같다:

$$\text{Var}(\hat{I}_n) = \frac{\sigma_f^2}{n}$$

여기서 $\sigma_f^2 = \text{Var}_p(f(X)) = \mathbb{E}_p[f(X)^2] - I^2$이다.

그러므로 표준 오차는 다음과 같다:

$$\text{SE}(\hat{I}_n) = \frac{\sigma_f}{\sqrt{n}}$$

### 2.4 중심 극한 정리

$n$이 넉넉히 크면 어림자는 대략 가우스이다:

$$\sqrt{n}\,(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$$

이는 어림 $(1-\alpha)$ 신뢰 구간을 준다:

$$\hat{I}_n \pm z_{1-\alpha/2} \cdot \frac{\hat{\sigma}_f}{\sqrt{n}}$$

여기서 $\hat{\sigma}_f^2 = \frac{1}{n-1} \sum_{i=1}^{n} (f(X_i) - \hat{I}_n)^2$은 표본 흩어짐이다.

### 2.5 차원과 상관없는 모임 속도

몬테카를로의 모임 속도는 **차원과 상관없이** $O(n^{-1/2})$이다. 이것이 격자 방법에 견준 핵심 이점이다:

| 방법 | 모임 속도 | 10차원 문제 | 100차원 문제 |
|--------|-----------------|-------------|--------------|
| 격자(사다리꼴) | $O(n^{-2/d})$ | $O(n^{-0.2})$ | $O(n^{-0.02})$ |
| 격자(심프슨) | $O(n^{-4/d})$ | $O(n^{-0.4})$ | $O(n^{-0.04})$ |
| **몬테카를로** | $O(n^{-1/2})$ | $O(n^{-0.5})$ | $O(n^{-0.5})$ |

차원이 5쯤을 넘는 문제라면 몬테카를로가 정해진 구적법을 누른다.

---

## 3. 구현

### 3.1 기본 몬테카를로 어림자

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

### 3.2 보기: 표준 정규 아래에서 E[X²] 어림하기

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

```
MC estimate of E[X²]: 0.993754
Standard error:       0.014068
True value:           1.000000
95% CI: [0.966181, 1.021327]
```

### 3.3 모임 보여 주기

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

곱 $n \cdot \text{SE}^2 \approx \sigma_f^2 \approx 2$이 상수로 남아 $O(n^{-1/2})$ 모임을 확인해 준다.

---

## 4. 베이즈 뒤확률 기댓값

### 4.1 베이즈 몬테카를로의 판 벌이기

베이즈 추론에서 우리는 다음을 셈하려 한다:

$$\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] = \int f(\theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

표본 $\theta^{(1)}, \ldots, \theta^{(n)} \sim p(\theta \mid \mathcal{D})$을 뽑을 수 있으면 다음과 같다:

$$\hat{I}_n = \frac{1}{n} \sum_{i=1}^{n} f(\theta^{(i)}) \approx \mathbb{E}[f(\theta) \mid \mathcal{D}]$$

흔히 고르는 $f$은 표준 뒤확률 간추림을 준다:

| $f(\theta)$ | 어림하는 양 |
|-------------|-------------------|
| $\theta$ | 뒤확률 평균 |
| $(\theta - \bar{\theta})^2$ | 뒤확률 흩어짐 |
| $\mathbb{1}(\theta \in A)$ | 구역 $A$의 뒤확률 |
| $p(x_{\text{new}} \mid \theta)$ | 뒤확률 예측 밀도 |

### 4.2 근본적인 어려움

기본 몬테카를로 방법은 **과녁 $p(\theta \mid \mathcal{D})$에서 곧바로 표집하기**를 요구한다. 다음일 때 이것이 말썽이 된다:

1. 뒤확률에 다룰 수 없는 고르게 하는 상수 $p(\mathcal{D})$이 있을 때
2. 뒤확률의 함수 꼴에 맞는 표준 표집 알고리즘이 없을 때
3. 뒤확률의 봉우리가 여럿이거나 기하가 복잡할 때

이런 한계 때문에 이 장의 나머지에서 다루는 더 정교한 방법이 나온다. 곧 **물리치기 표집**(18.2절), **중요도 표집**(18.2절), **MCMC**(18.3절)이며, 이들은 곧바른 표집 없이도 복잡한 과녁 분포에서 표본을 만든다.

### 4.3 보기: 격자 표집으로 구하는 뒤확률 평균

(차원이 낮아) 격자 위에서 뒤확률을 얻을 수 있으면 그 이산 어림에서 표본을 뽑을 수 있다:

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

```
MC Posterior Mean:        0.666834
Analytical Posterior Mean: 0.666667
MC Posterior Std:         0.130212
95% Credible Interval:   [0.3934, 0.8919]
```

---

## 5. 흩어짐 줄이기 기법

$O(n^{-1/2})$ 모임 속도는 차원과 상관없지만 그 자체로는 느릴 수 있다. 정밀도를 10배 높이려면 표본이 100배 필요하다. 흩어짐 줄이기 기법은 표본 개수를 늘리지 않고 $\sigma_f^2$을 줄여 효율을 낫게 한다.

### 5.1 다스림 변량

$f$과 얽힌 어떤 함수 $g$에 대해 $\mathbb{E}_p[g(X)] = \mu_g$을 정확히 알면 **다스림 변량 어림자**는 다음과 같다:

$$\hat{I}_{\text{CV}} = \frac{1}{n} \sum_{i=1}^{n} \bigl[f(X_i) - c\,(g(X_i) - \mu_g)\bigr]$$

가장 좋은 계수는 $c^* = \text{Cov}(f, g) / \text{Var}(g)$이며 흩어짐을 다음처럼 줄인다:

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

```
True value:    1.648721
Standard MC:   1.649844 (SE = 0.023118)
Control Var:   1.648553 (SE = 0.013247)
Variance reduction: 67.2%
```

### 5.2 맞선 변량

대칭 분포에서는 표본 $X_i$마다 거울상 $-X_i$과 짝짓는다:

$$\hat{I}_{\text{AV}} = \frac{1}{2n} \sum_{i=1}^{n} \bigl[f(X_i) + f(-X_i)\bigr]$$

흩어짐은 다음과 같다:

$$\text{Var}(\hat{I}_{\text{AV}}) = \frac{\sigma_f^2 + \text{Cov}(f(X), f(-X))}{2n}$$

$f$이 단조이고 $X$이 대칭이면 $\text{Cov}(f(X), f(-X)) < 0$이므로 흩어짐이 줄어든다.

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

### 5.3 층 나눠 표집하기

표본 공간을 확률이 $p_k = P(X \in S_k)$인 층 $S_1, \ldots, S_K$ $K$개로 나눈다. 층마다 표본 $n_k$개를 뽑는다:

$$\hat{I}_{\text{strat}} = \sum_{k=1}^{K} p_k \hat{I}_k, \quad \text{where } \hat{I}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} f(X_i^{(k)})$$

층 나누기는 $\text{Var}(\hat{I}_{\text{strat}}) \leq \text{Var}(\hat{I}_{\text{MC}})$을 보장하며, 층 안 평균이 모두 같을 때만 등호가 성립한다.

### 5.4 흩어짐 줄이기 방법 간추림

| 방법 | 흩어짐 줄임 | 필요 조건 | 복잡함 |
|--------|-------------------|--------------|------------|
| 다스림 변량 | 최대 $100(1-\rho^2)\%$ | 아는 $\mathbb{E}[g]$, $f$과 얽힌 $g$ | 낮음 |
| 맞선 변량 | 최대 50% | 대칭 분포, 단조인 $f$ | 낮음 |
| 층 나눠 표집 | 층 나누기에 달렸다 | 공간의 자연스러운 나눔 | 보통 |
| 중요도 표집 | 한계 없이 줄일 수도 있다 | 좋은 제안 분포 | 18.2절을 보아라 |

---

## 6. 모임 진단

### 6.1 몬테카를로 오차 살피기

중심 극한 정리가 점근 정규성을 주므로 신뢰 구간을 짓고 표본을 넉넉히 뽑았는지 살필 수 있다:

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

### 6.2 어림 규칙

미더운 몬테카를로 어림을 위해:

- 점 어림값의 **상대 표준 오차 < 1%**: $n \geq 10{,}000 \cdot (\sigma_f / I)^2$이 필요하다
- **꼬리 확률**: 참 확률이 $p$일 때 $P(\theta > c)$을 상대 정확도 10%로 어림하려면 대략 $n \geq 10{,}000 / p$이 필요하다
- **표본을 두 배로 하면 표준 오차가 절반이 된다**: $\sqrt{n}$ 관계는 얻는 것이 점점 줄어든다는 뜻이다

---

## 7. 수치 적분을 위한 몬테카를로

베이즈 뒤확률을 넘어 몬테카를로 적분은 어떤 적분에도 쓸 수 있다. "맞히거나 빗나가거나" 꼴이 기본 원리를 보여 준다.

### 7.1 몬테카를로로 원주율 셈하기

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

```
n=       100: π ≈ 3.120000  (SE=0.164966, error=0.021593)
n=     1,000: π ≈ 3.164000  (SE=0.052196, error=0.022407)
n=    10,000: π ≈ 3.130800  (SE=0.016419, error=0.010793)
n=   100,000: π ≈ 3.142480  (SE=0.005201, error=0.000887)
n= 1,000,000: π ≈ 3.142240  (SE=0.001644, error=0.000647)
```

### 7.2 일반 적분

$I = \int_a^b g(x)\, dx$을 셈하려면 $[a, b]$ 위 고른 분포 아래의 기댓값으로 고쳐 쓴다:

$$I = (b - a)\, \mathbb{E}_{U(a,b)}[g(X)]$$

그리고 다음으로 어림한다:

$$\hat{I}_n = \frac{b - a}{n} \sum_{i=1}^{n} g(X_i), \quad X_i \overset{\text{i.i.d.}}{\sim} U(a, b)$$

---

## 8. 한계와 앞으로 갈 길

### 8.1 기본 몬테카를로로 모자랄 때

기본 몬테카를로는 **과녁 분포에서 곧바로 표집하기**를 요구한다. 세 가지 상황이 이 요구를 무너뜨린다:

1. **고르게 하지 않은 뒤확률**: 고르게 하는 상수 $p(\mathcal{D})$을 모르므로 $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$에서 곧바로 표집할 수 없다
2. **복잡한 분포**: 과녁의 함수 꼴에서 표본을 만드는 표준 알고리즘이 없다
3. **드문 일**: $p$ 아래에서 확률이 아주 낮은 일은 정확히 어림하려면 표본이 엄청나게 많이 필요하다

### 8.2 방법마다 고르게 하기를 어떻게 다루나

모든 베이즈 셈하기를 떠받치는 핵심 통찰은 $p(\mathcal{D})$을 드러내 놓고 쓸 일이 드물다는 것이다:

| 방법 | 고르게 하기를 어떻게 다루나 |
|--------|------------------------------|
| 격자 어림 | 격자점에 걸쳐 합한다 |
| 중요도 표집 | 무게 준 합의 비 |
| MCMC | 받아들임 비에서 고르게 하는 상수가 지워진다 |
| 점수 기반 방법 | $\nabla_\theta \log p(\theta)$을 배운다. 기울기에서 상수가 사라진다 |

### 8.3 방법이 나아가는 길

이 장의 방법은 하나하나 앞선 방법의 한계를 넘어선다:

$$\text{Grid} \xrightarrow{\text{dimension}} \text{Rejection/IS} \xrightarrow{\text{intractable target}} \text{MCMC} \xrightarrow{\text{gradients}} \text{HMC/Langevin}$$

- **격자 → 몬테카를로**: 차원의 저주를 넘어선다($O(n^{-1/2})$ 대 $O(n^{-2/d})$)
- **기본 몬테카를로 → 물리치기/중요도 표집**: 곧바로 표집할 수 없는 분포를 다룬다
- **중요도 표집 → MCMC**: 높은 차원에서의 무게 주저앉음을 다룬다
- **무작위 걸음 MCMC → HMC/랑주뱅**: 효율적으로 살펴보려고 기울기 정보를 쓴다

다음 마당은 물리치기 표집과 [중요도 표집](importance_sampling/fundamentals.md)을 다루는데, 이들은 MCMC 사슬 없이도 복잡한 분포에서 표집한다.

---

## 9. 금융에서의 쓰임새

몬테카를로 적분은 계량 금융 곳곳에 퍼져 있다:

| 쓰임새 | 어림하는 적분 | 방법에 대한 메모 |
|-------------|------------------------|--------------|
| 옵션 값 매기기 | 위험 중립 측도 아래의 $\mathbb{E}[\text{할인된 지급액}]$ | 경로에 기대는 옵션에는 맞선 변량이 표준이다 |
| 위험 값 | $P(\text{손실} > \text{VaR}_\alpha)$ 분위수 어림 | 꼬리 정확도에는 큰 $n$이 필요하다 |
| 포트폴리오 위험 | $\mathbb{E}[\text{손실} \mid \text{손실} > \text{VaR}_\alpha]$(CVaR) | 델타-정규 어림을 쓴 다스림 변량 |
| 베이즈 포트폴리오 | 기대 수익률의 뒤확률 아래 $\mathbb{E}[w^*(\theta)]$ | 튼튼한 배분을 위해 뒤확률을 온전히 퍼뜨린다 |
| 신용 위험 | 얽힌 인자 모형 아래 $P(\text{부도})$ | 드문 부도 일에는 중요도 표집 |
| 확률 변동성 | 숨은 변동성 경로 $\mathbb{E}[\sigma_t^2 \mid \text{수익률}]$ | 알갱이 거르기(잇단 몬테카를로) |

---

## 10. 핵심 되새김

1. **몬테카를로의 모임은 차원과 상관없이 $O(n^{-1/2})$**이며, 매개변수가 3개쯤을 넘으면 격자 방법을 쓸모없게 만드는 차원의 저주를 넘어선다.

2. **중심 극한 정리가 불확실함을 저절로 재어 준다.** 곧 몬테카를로 어림값마다 값을 더 치르지 않고도 표준 오차와 신뢰 구간이 딸려 온다.

3. **흩어짐 줄이기 기법**(다스림 변량, 맞선 변량, 층 나눠 표집)은 표본 개수를 늘리지 않고도 효율을 크게 끌어올릴 수 있다.

4. **기본 몬테카를로는 과녁 분포에서 표집하기를 요구**하는데, 고르게 하는 상수를 모르는 베이즈 뒤확률에서는 흔히 불가능하다. 그래서 물리치기 표집, 중요도 표집, MCMC가 나온다.

5. **베이즈 셈하기는 모두** 뒤확률 아래 기댓값을 셈하는 일로 줄어든다. 뒤확률 평균, 흩어짐, 믿음 구간, 예측 분포가 모두 몬테카를로로 어림할 수 있는 적분이다.

---

## 참고 문헌

1. Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer. 3-4장.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. 2-3장.
3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. 10장.
4. Owen, A. B. (2013). *Monte Carlo Theory, Methods, and Examples*. https://artowen.su.domains/mc/.
5. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
6. McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. 9장.

## 연습문제

### 연습 1: 모임 속도 확인하기

$\mathcal{N}(0, 1)$에서 표본 $n$개를 만들고 $n \in \{100, 500, 1000, 5000, 10000, 50000\}$에 대해 $\mathbb{E}[X^2] = 1$을 어림하여라. 로그-로그 눈금에서 표준 오차를 $n$에 대해 그리고 기울기가 $O(n^{-1/2})$임을 확인하여라.

### 연습 2: 옵션 값 매기기를 위한 다스림 변량

기하 브라운 운동 경로를 쓴 몬테카를로로 유럽식 콜 옵션의 값을 매겨라. 블랙-숄즈 공식을 (행사가나 변동성을 살짝 달리해) 다스림 변량으로 써라. 흩어짐이 얼마나 줄었는지 재어라.

### 연습 3: 여러 차원에서 격자와 몬테카를로 견주기

$d$차원 표준 가우스에서 $d \in \{1, 2, 3, 5\}$에 대해 격자 어림과 몬테카를로 둘 다로 $\mathbb{E}[\|X\|^2] = d$을 어림하여라. 상대 오차 1%에 필요한 함수 값 매기기 횟수를 견주어라.

### 연습 4: 맞선 변량

$X \sim \mathcal{N}(0, 1)$일 때 $\mathbb{E}[\exp(X)]$을 어림하는 맞선 변량을 구현하여라. 표준 몬테카를로와 흩어짐을 견주고 왜 줄어드는지 설명하여라.

### 연습 5: 꼬리 확률 어림하기

$X \sim \mathcal{N}(0, 1)$일 때 $P(X > 3)$을 어림하여라(참값 ≈ 0.00135). 95% 믿음 구간이 0을 빼려면 표본이 몇 개 필요한가? 이것이 왜 기본 몬테카를로에 어려운 문제이며 중요도 표집이 어떻게 도울 수 있는가?

---
