# 믿음 구간
## 개요

믿음 구간은 매개변수 어림값의 불확실성을 베이즈식으로 수에 담는다. 이 모듈은 양 꼬리가 같은 구간과 최고 뒤확률 밀도(HPD) 구간을 세우고, 베이즈 믿음 구간과 빈도주의 신뢰 구간의 근본 차이를 또렷이 한다.

---

## 1. 믿음 구간: 정의와 풀이

### 1.1 정의

매개변수 $\theta$의 **$(1-\alpha) \times 100\%$ 믿음 구간** $[L, U]$은 다음을 만족한다.

$$
\boxed{P(L \leq \theta \leq U \mid D) = 1 - \alpha}
$$

### 1.2 풀이

관찰한 데이터 $D$이 주어졌을 때 참 매개변수가 그 구간 안에 있을 확률이 $(1-\alpha) \times 100\%$이다.

**보기:** 95% 믿음 구간 $[0.55, 0.82]$은 다음을 뜻한다.

> "우리 데이터와 앞확률이 주어졌을 때 $\theta$이 0.55와 0.82 사이에 있을 확률이 95%이다."

이는 매개변수 자체에 대한 곧은 확률 진술이며, 우리가 정말로 알고 싶은 양이다.

---

## 2. 믿음 구간의 종류

### 2.1 양 꼬리가 같은 구간

**양 꼬리가 같은 구간**은 양쪽 꼬리에 확률 질량($\alpha/2$)을 똑같이 둔다.

$$
[L, U] = [q_{\alpha/2}, q_{1-\alpha/2}]
$$

여기서 $q_p$은 뒤확률 분포의 $p$번째 분위수이다.

**성질:**

- 분위수로 셈하기 쉽다
- 확률로는 대칭이다(너비까지 그렇지는 않다)
- 가장 짧은 구간이 아닐 수 있다

**구현:**

```python
def compute_equal_tailed_interval(posterior_dist, alpha=0.05):
    """
    양끝이 같은 믿음 구간을 셈한다.
    
    매개변수
    ----------
    posterior_dist : scipy.stats 분포
        뒤확률 분포 객체
    alpha : float
        유의수준(95% 구간이면 0.05)
    
    반환값
    -------
    interval : tuple
        (아래_경계, 위_경계)
    """
    lower = posterior_dist.ppf(alpha / 2)
    upper = posterior_dist.ppf(1 - alpha / 2)
    return (lower, upper)
```

### 2.2 최고 뒤확률 밀도(HPD) 구간

**HPD 구간**은 뒤확률 질량의 $(1-\alpha) \times 100\%$을 담는 가장 짧은 구간이다.

**정의하는 성질:** HPD 구간 안의 점은 모두 바깥의 어떤 점보다 뒤확률 밀도가 높다.

$$
\{θ : p(θ|D) \geq k\}
$$

여기서 $k$은 $P(\theta \in \text{HPD} \mid D) = 1 - \alpha$이 되도록 고른다.

**성질:**

- 주어진 덮음률에서 가장 짧은 구간
- 봉우리가 하나인 뒤확률에 가장 알맞다
- 셈하기가 더 복잡하다
- 대칭 분포에서는 HPD = 양 꼬리가 같은 구간

**구현(표본에서):**

```python
import numpy as np

def compute_hpd_interval(samples, alpha=0.05):
    """
    뒤확률 표본에서 HPD 구간을 셈한다.
    
    매개변수
    ----------
    samples : array-like
        뒤확률 분포에서 뽑은 표본
    alpha : float
        유의수준
    
    반환값
    -------
    interval : tuple
        (아래_경계, 위_경계)
    """
    samples_sorted = np.sort(samples)
    n = len(samples)
    
    # 담을 표본의 개수
    n_included = int(np.ceil((1 - alpha) * n))
    
    # 이 크기의 모든 구간 찾기
    n_intervals = n - n_included + 1
    interval_widths = samples_sorted[n_included-1:] - samples_sorted[:n_intervals]
    
    # 너비가 가장 작은 구간 고르기
    min_idx = np.argmin(interval_widths)
    hpd_lower = samples_sorted[min_idx]
    hpd_upper = samples_sorted[min_idx + n_included - 1]
    
    return (hpd_lower, hpd_upper)
```

### 2.3 견줌

| 성질 | 양 꼬리가 같음 | HPD |
|----------|--------------|-----|
| 셈하기 | 단순함(분위수) | 더 복잡함 |
| 최적성 | 가장 짧음이 보장되지 않음 | 가장 짧은 구간 |
| 대칭 뒤확률 | 같음 | 같음 |
| 기운 뒤확률 | 더 김 | 더 짧음 |
| 풀이 | 양 꼬리 확률이 같음 | 밀도가 가장 높은 자리 |

---

## 3. 보기: 베타 뒤확률

### 3.1 얼개

동전 던지기에서 나온 베타 뒤확률을 생각해 보자.

- 데이터: 앞면 15번, 뒷면 5번
- 앞확률: Beta$(1, 1)$(고른 분포)
- 뒤확률: Beta$(16, 6)$

### 3.2 구간 셈하기

```python
from scipy import stats

# 뒤확률 분포
posterior = stats.beta(16, 6)

# 95% 양끝 같은 구간
et_lower = posterior.ppf(0.025)  # 0.5765
et_upper = posterior.ppf(0.975)  # 0.8875
# 너비: 0.311

# 95% HPD 구간(표본 100,000개에서)
samples = posterior.rvs(100000)
hpd_lower, hpd_upper = compute_hpd_interval(samples, alpha=0.05)
# 대략 [0.571, 0.885]
# 너비: 0.314
```

### 3.3 결과

| 구간 종류 | 아래 | 위 | 너비 |
|---------------|-------|-------|-------|
| 양 꼬리가 같음 | 0.577 | 0.888 | 0.311 |
| HPD | 0.571 | 0.885 | 0.314 |

살짝 기운 이 Beta$(16, 6)$ 뒤확률에서는 두 구간이 비슷하다. 크게 기운 뒤확률일수록 차이가 뚜렷해진다.

---

## 4. 믿음 구간과 신뢰 구간

### 4.1 근본 차이

| 갈래 | 믿음 구간(베이즈) | 신뢰 구간(빈도주의) |
|--------|------------------------------|-----------------------------------|
| **확률 진술** | 매개변수에 대해 | 절차에 대해 |
| **데이터** | 붙박임(관찰됨) | 확률적임(가상의 되풀이) |
| **매개변수** | 확률적임(분포를 갖는다) | 붙박임(모르는 상수) |
| **풀이** | "θ이 여기 있을 확률이 95%" | "그런 구간의 95%가 θ을 담는다" |

### 4.2 풀이 견줌

**95% 믿음 구간 $[0.55, 0.82]$:**

> "우리가 관찰한 데이터가 주어졌을 때 참 매개변수 $\theta$이 0.55와 0.82 사이에 있을 확률이 95%이다."

**95% 신뢰 구간 $[0.55, 0.82]$:**

> "이 실험을 여러 번 되풀이하며 그때마다 구간을 셈한다면 그 구간의 95%가 참 매개변수 $\theta$을 담을 것이다."

### 4.3 빈도주의의 단서

신뢰 구간은 매개변수가 그 구간 안에 있을 확률이 95%라는 뜻이 **아니다**. 빈도주의 통계에서 매개변수는 붙박인(모르지만) 상수여서 구간 안에 있거나 없거나 둘 중 하나이다. 확률 진술은 붙박인 매개변수가 아니라 확률적인 구간에 대한 것이다.

---

## 5. 흉내 내기 연구

### 5.1 덮음률 견줌

흉내 내기로 구간의 성질을 확인할 수 있다.

```python
import numpy as np
from scipy import stats

def coverage_simulation(true_p=0.7, n_trials=20, n_experiments=1000):
    """믿음 구간과 신뢰 구간의 덮음률을 견준다."""
    
    np.random.seed(42)
    
    credible_coverage = 0
    confidence_coverage = 0
    
    for _ in range(n_experiments):
        # 데이터를 생성한다
        successes = np.random.binomial(n_trials, true_p)
        
        # 베이즈 95% 믿음 구간(고른 앞확률)
        post = stats.beta(1 + successes, 1 + n_trials - successes)
        cred_lower, cred_upper = post.ppf(0.025), post.ppf(0.975)
        
        if cred_lower <= true_p <= cred_upper:
            credible_coverage += 1
        
        # 빈도주의 95% 신뢰 구간(왈드)
        p_hat = successes / n_trials
        se = np.sqrt(p_hat * (1 - p_hat) / n_trials)
        conf_lower = max(0, p_hat - 1.96 * se)
        conf_upper = min(1, p_hat + 1.96 * se)
        
        if conf_lower <= true_p <= conf_upper:
            confidence_coverage += 1
    
    return {
        'credible': credible_coverage / n_experiments,
        'confidence': confidence_coverage / n_experiments
    }
```

### 5.2 흔한 결과

$p_{\text{true}} = 0.7$, $n = 20$으로 실험 1000번을 하면 다음과 같다.

| 구간 종류 | 관찰된 덮음률 |
|---------------|-------------------|
| 95% 믿음 구간 | 94~96% 남짓 |
| 95% 신뢰 구간(발드) | 90~94% 남짓 |

발드 신뢰 구간은 표본이 작거나 확률이 극단적일 때 **덜 덮는** 일이 많다. 베이즈 믿음 구간이 눈금을 더 잘 맞추는 편이다.

---

## 6. 실전에서 살필 것

### 6.1 어느 구간을 언제 쓸까

| 상황 | 권하는 구간 |
|-----------|---------------------|
| 대칭 뒤확률 | 아무거나(같다) |
| 기운 뒤확률 | HPD(더 짧다) |
| 빠른 셈 | 양 꼬리가 같은 구간 |
| 가장 좋은 결정 | HPD |
| 통계 전문가가 아닌 사람에게 알릴 때 | 믿음 구간(더 직관적이다) |

### 6.2 믿음 구간 알리기

보통 다음을 알린다.

1. **점 어림값**(뒤확률의 평균이나 최대 뒤확률)
2. **95% 믿음 구간**(양 꼬리가 같은 것이나 HPD)
3. **뒤확률 분포**(가능할 때)

**예:**

> "어림한 성공 확률은 0.73이다(95% 믿음 구간: 0.58~0.89)"

### 6.3 다변량 믿음 영역

매개변수가 여럿이면 구간 대신 **믿음 영역**을 셈한다.

$$
P(\theta \in \mathcal{R} \mid D) = 1 - \alpha
$$

2차원 매개변수라면 (가우스 뒤확률에서) **믿음 타원**이나 더 복잡한 영역이 나온다.

---

## 7. 핵심 요점

1. **믿음 구간**은 매개변수에 대한 곧은 확률 진술을 한다. 대부분의 사람이 바라는 직관적인 풀이이다.

2. **양 꼬리가 같은 구간**은 분위수로 셈하기 쉽고, **HPD 구간**은 가장 짧은 구간이다.

3. **대칭 뒤확률에서는** 양 꼬리가 같은 구간과 HPD 구간이 일치한다. 기운 뒤확률에서는 HPD가 더 짧다.

4. **믿음 구간 ≠ 신뢰 구간**: 둘은 서로 다른 물음에 답하고 풀이도 다르다.
   - 믿음 구간: "θ이 여기 있을 확률이 95%"(매개변수에 대한 확률)
   - 신뢰 구간: "그런 절차의 95%가 θ을 잡는다"(절차에 대한 확률)

5. 실전에서 베이즈 믿음 구간은 빈도주의 신뢰 구간보다 **덮음 성질**이 나을 때가 많으며, 특히 표본이 작을 때 그렇다.

---

## 8. 연습문제

### 연습문제 1: 기운 뒤확률
Gamma$(2, 1)$ 뒤확률에 대해 양 꼬리가 같은 구간과 HPD 구간을 모두 셈하라. HPD가 더 짧은지 확인하라.

### 연습문제 2: 대칭 뒤확률
대칭 분포(이를테면 정규)에서 양 꼬리가 같은 구간과 HPD 구간이 같음을 해석적으로(또는 흉내 내기로) 보여라.

### 연습문제 3: 다변량 영역
2차원 정규 뒤확률의 95% 믿음 타원을 구현하라. 공분산 짜임에 따라 영역이 어떻게 바뀌는지 그려 보라.

### 연습문제 4: 덮음률 연구
작은 표본 크기($n = 5, 10, 20$)와 극단적인 참 확률($p = 0.05, 0.5, 0.95$)에서 믿음 구간과 신뢰 구간의 실제 덮음률을 견주어라.

---

## 참고 문헌

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2장
- Kruschke, J. *Doing Bayesian Data Analysis* (2nd ed.), 12장
- Hoff, P. *A First Course in Bayesian Statistical Methods*, 3장

## 연습문제

**연습문제 1.**
믿음 구간을 정의하고 빈도주의 신뢰 구간과 견주어라.

??? success "연습문제 1 풀이"
    95% 믿음 구간 $[a, b]$은 $P(\theta \in [a,b] | D) = 0.95$을 뜻한다. 곧 데이터가 주어졌을 때 매개변수가 이 구간에 있을 확률이 95%이다. 95% 신뢰 구간은 실험을 여러 번 되풀이하면 그렇게 만든 구간의 95%가 참 매개변수를 담는다는 뜻이다. 베이즈의 풀이가 더 직관적이다.

---

**연습문제 2.**
Beta(10, 30) 뒤확률의 95% 믿음 구간을 셈하라.

??? success "연습문제 2 풀이"
    분위수를 쓰면 아래 = Beta.ppf(0.025, 10, 30) = 0.134, 위 = Beta.ppf(0.975, 10, 30) = 0.394이다. 95% 믿음 구간은 [0.134, 0.394]이다. 뒤확률의 평균은 10/40 = 0.25이다.

---

**연습문제 3.**
최고 뒤확률 밀도(HPD) 구간이란 무엇이며 양 꼬리가 같은 구간과 언제 갈리는가?

??? success "연습문제 3 풀이"
    HPD: 뒤확률 95%를 담는 가장 짧은 구간이다. 안의 점은 모두 바깥의 어떤 점보다 밀도가 높다. 양 꼬리가 같은 구간: 꼬리마다 확률이 2.5%이다. 기운 뒤확률에서 둘이 갈리며 HPD가 더 짧고 알맹이가 많다. 대칭 뒤확률(이를테면 정규)에서는 둘이 일치한다.

---

**연습문제 4.**
MCMC 표본에서 HPD 구간을 셈하는 함수를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    def hpd_interval(samples, credibility=0.95):
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_size = int(np.ceil(credibility * n))
        widths = sorted_samples[interval_size:] - sorted_samples[:n-interval_size]
        best = np.argmin(widths)
        return sorted_samples[best], sorted_samples[best + interval_size]
    ```
