# 최대 뒤확률(MAP) 어림

최대 뒤확률 어림은 뒤확률 분포의 최빈값을 찾아, 앞선 정보를 아우른 점 어림값을 준다. 이 마당은 최대 뒤확률을 최대 가능도, 뒤확률의 평균과 견주고, 최대 뒤확률을 위한 수치 최적화 방법을 세우며, 최대 뒤확률 어림과 벌주기 사이의 근본적인 이음을 밝힌다.

---

## 1. 베이즈 추론의 점 어림값 셋

뒤확률 분포 하나에서 값 하나를 뽑아내는 길은 여럿이다. 널리 쓰이는 세 가지는 가능도만 보는 것, 뒤확률의 최빈값을 보는 것, 뒤확률의 평균을 보는 것이다.

**최대 가능도 어림값(MLE)**은 앞선 정보를 아랑곳하지 않고 가능도 함수를 가장 크게 한다.

$$
\boxed{\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \; p(D|\theta)}
$$

**최대 뒤확률(MAP)**은 뒤확률 분포를 가장 크게 한다.

$$
\boxed{\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \; p(\theta|D) = \underset{\theta}{\arg\max} \; p(D|\theta) \, p(\theta)}
$$

**뒤확률의 평균**은 뒤확률 아래에서의 기댓값이다.

$$
\boxed{\hat{\theta}_{\text{Mean}} = \mathbb{E}[\theta|D] = \int \theta \, p(\theta|D) \, d\theta}
$$

### 정리 1. 세 가지 점 어림값 — 손실 함수가 어림기를 고른다

앞확률 $p(\theta)$과 가능도 $p(D|\theta)$가 주어졌다고 하자. 그러면 다음이 성립한다.

1. 앞확률이 고르면 $\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{MLE}}$이다.
2. 0-1 손실 아래에서 베이즈 어림기는 $\hat{\theta}_{\text{MAP}}$이다.
3. 제곱 오차 손실 아래에서 베이즈 어림기는 $\hat{\theta}_{\text{Mean}}$이다.

*밝힘.* (1) 고른 앞확률에서는 $\log p(\theta)$가 상수이므로 $\arg\max_\theta [\log p(D|\theta) + \log p(\theta)] = \arg\max_\theta \log p(D|\theta)$이다. (3) 기대 손실 $\mathbb{E}[(\theta - a)^2 | D]$을 $a$에 대해 미분해 0으로 두면 $a = \mathbb{E}[\theta|D]$을 얻는다. (2)는 손실 폭을 0으로 보내는 끝값에서 최빈값이 남는 데서 나온다. $\square$

| 어림기 | 정의 | 가장 작게 하는 손실 함수 | 앞확률 씀 |
|-----------|------------|------------------------|------------|
| 최대 가능도 | 가능도의 최빈값 | — | 아니오 |
| 최대 뒤확률 | 뒤확률의 최빈값 | 0-1 손실 | 예 |
| 뒤확률의 평균 | 뒤확률의 평균 | 제곱 오차 | 예 |

---

## 2. 베타-이항 모형의 닫힌 꼴

켤레 앞확률을 쓰면 세 어림값을 모두 손으로 적을 수 있어, 앞확률이 어림값을 어느 쪽으로 끌어당기는지 눈으로 볼 수 있다.

### 정리 2. 베타-이항의 닫힌 꼴 — 앞확률은 유사 관측으로 더해진다

앞확률이 $\text{Beta}(\alpha, \beta)$이고 데이터가 앞면 $k$번, 뒷면 $n-k$번이면 뒤확률은 $\text{Beta}(\alpha + k,\; \beta + n - k)$이고, $\alpha + k > 1$이며 $\beta + n - k > 1$일 때 다음이 성립한다.

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2}, \qquad
\hat{\theta}_{\text{MLE}} = \frac{k}{n}, \qquad
\hat{\theta}_{\text{Mean}} = \frac{\alpha + k}{\alpha + \beta + n}
$$

*밝힘.* 뒤확률은 $p(\theta|D) \propto \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$이므로 $\text{Beta}(\alpha+k, \beta+n-k)$이다. $\text{Beta}(a,b)$의 최빈값은 $a,b>1$일 때 $(a-1)/(a+b-2)$이고 평균은 $a/(a+b)$이다. 여기에 $a = \alpha+k$, $b = \beta+n-k$을 넣으면 된다. $\square$

곧 앞확률 $\text{Beta}(\alpha,\beta)$은 앞면 $\alpha$번, 뒷면 $\beta$번을 미리 본 것과 같은 몫을 한다.

**셈 보기.** 데이터가 앞면 7번, 뒷면 3번($k=7$, $n=10$)이고 앞확률이 $\text{Beta}(2, 2)$일 때는 다음과 같다.

| 어림기 | 식 | 값 |
|-----------|---------|-------|
| 최대 가능도 | \$7/10$ | 0.700 |
| 최대 뒤확률 | $(2+7-1)/(2+2+10-2) = 8/12$ | 0.667 |
| 뒤확률의 평균 | $(2+7)/(2+2+10) = 9/14$ | 0.643 |

앞확률 $\text{Beta}(2, 2)$이 어림값을 0.5 쪽으로 끌어당긴다.

```python
import numpy as np
from scipy import stats

def map_vs_mle_beta_binomial(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """베타-이항에서 MAP, MLE, 뒤확률 평균을 견준다."""
    
    n_total = n_heads + n_tails
    
    # MLE
    mle = n_heads / n_total if n_total > 0 else 0.5
    
    # 뒤확률 매개변수
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # 뒤확률 평균
    posterior_mean = post_alpha / (post_alpha + post_beta)
    
    # MAP(베타 분포의 최빈값)
    if post_alpha > 1 and post_beta > 1:
        map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        map_estimate = posterior_mean  # 최빈값이 없으면 평균을 쓴다
    
    return {
        'mle': mle,
        'map': map_estimate,
        'posterior_mean': posterior_mean
    }
```

---

## 3. 수치 최적화로 하는 최대 뒤확률

켤레가 아닌 모형에서는 닫힌 꼴이 없다. 그래도 최대 뒤확률은 늘 최적화 문제로 적을 수 있어서, 뒤확률 전체를 다루지 않고도 풀 수 있다.

### 정리 3. 최대 뒤확률의 최적화 꼴 — 로그가능도와 로그앞확률의 합

증거 $p(D)$은 $\theta$에 달리지 않으므로 다음이 성립한다.

$$
\hat{\theta}_{\text{MAP}}
= \underset{\theta}{\arg\max} \left[ \log p(D|\theta) + \log p(\theta) \right]
= \underset{\theta}{\arg\min} \left[ -\log p(D|\theta) - \log p(\theta) \right]
$$

*밝힘.* $p(\theta|D) = p(D|\theta)p(\theta)/p(D)$이고 $\log$는 단조 증가하므로 $\arg\max_\theta \log p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta) - \log p(D)]$이다. 마지막 항은 $\theta$과 무관하므로 떨어져 나간다. $\square$

곧 다루기 어려운 적분 $p(D) = \int p(D|\theta)p(\theta)\,d\theta$을 아예 셈하지 않아도 된다는 것이 최대 뒤확률의 값어치다.

**보기: 평균과 흩어짐을 모르는 정규.**

- 데이터: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$
- 평균의 앞확률: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$
- 정밀도의 앞확률: $\tau = 1/\sigma^2 \sim \text{Gamma}(\alpha, \beta)$

음의 로그 뒤확률은 다음과 같다.

$$
-\log p(\mu, \tau | D) = -\sum_{i=1}^n \log p(x_i|\mu, \tau) - \log p(\mu) - \log p(\tau) + \text{const}
$$

```python
from scipy import optimize

def map_normal_unknown_mean_variance(data, prior_mean_mu=0, prior_std_mu=10,
                                     prior_shape_tau=2, prior_rate_tau=1):
    """평균과 흩어짐을 모르는 정규에 대한 MAP 어림."""
    
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    
    def neg_log_posterior(params):
        mu, log_tau = params
        tau = np.exp(log_tau)  # 제약 없는 최적화를 위해 매개변수 바꾸기
        
        # 로그 가능도
        log_lik = np.sum(stats.norm(mu, np.sqrt(1/tau)).logpdf(data))
        
        # 로그 앞확률
        log_prior_mu = stats.norm(prior_mean_mu, prior_std_mu).logpdf(mu)
        log_prior_tau = stats.gamma(prior_shape_tau, 
                                    scale=1/prior_rate_tau).logpdf(tau)
        
        return -(log_lik + log_prior_mu + log_prior_tau)
    
    # 최적화
    initial_guess = [sample_mean, np.log(1/sample_var)]
    result = optimize.minimize(neg_log_posterior, initial_guess, method='BFGS')
    
    map_mu = result.x[0]
    map_tau = np.exp(result.x[1])
    map_sigma = np.sqrt(1/map_tau)
    
    return {'map_mu': map_mu, 'map_sigma': map_sigma}
```

| 기법 | 목적 |
|-----------|---------|
| 양의 매개변수에 로그 변환 | 제약 없는 최적화 |
| (뒤확률이 아니라) 로그 뒤확률 쓰기 | 수치 안정성 |
| 여러 곳에서 초기화 | 국소 최적을 피함 |
| 기울기 기반 방법(BFGS, L-BFGS) | 매끄러운 뒤확률에 효율적 |

---

## 4. 최대 뒤확률 어림과 벌주기

정리 3의 최적화 꼴을 다시 보면 $-\log p(\theta)$이 벌 항의 자리에 그대로 앉아 있다. 이것이 벌주기와 베이즈를 잇는 다리다.

### 정리 4. 앞확률과 벌주기의 대응 — 가우스는 능선, 라플라스는 라소

선형 회귀에서 계수마다 서로 독립인 앞확률을 두면 다음이 성립한다.

1. $\theta_j \sim \mathcal{N}(0, \sigma_\theta^2)$이면 최대 뒤확률은 능선(L2) 회귀와 같다.

$$
\boxed{\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 \right]}
$$

2. $\theta_j \sim \text{Laplace}(0, b)$이면 최대 뒤확률은 라소(L1) 회귀와 같다.

$$
\boxed{\hat{\beta}_{\text{Lasso}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right]}
$$

*밝힘.* (1) $\log p(\theta) \propto -\frac{1}{2\sigma_\theta^2}\sum_j \theta_j^2$이므로 정리 3에 넣으면 $\lambda = 1/(2\sigma_\theta^2)$인 L2 벌 항이 된다. (2) $\log p(\theta) \propto -\frac{1}{b}\sum_j |\theta_j|$이므로 같은 방식으로 $\lambda = 1/b$인 L1 벌 항이 된다. $\square$

앞확률의 흩어짐이 벌주기의 세기를 다스린다. 앞확률이 좁을수록 벌이 세다.

| 앞확률 분포 | 벌주기 | 벌 항 | 효과 |
|-------------------|----------------|--------------|--------|
| 고른 분포 | 없음(최대 가능도) | — | 오그라듦 없음 |
| 가우스 $\mathcal{N}(0, \sigma^2)$ | 능선(L2) | $\lambda\|\theta\|_2^2$ | 계수를 오그라뜨린다 |
| 라플라스$(0, b)$ | 라소(L1) | $\lambda\|\theta\|_1$ | 성긴 해 |
| 편자(horseshoe) | 맞추어 가는 오그라듦 | 복잡함 | 강한 성김 |

```python
import numpy as np
from sklearn.linear_model import Lasso

def demonstrate_map_regularization(n_samples=50, noise_std=1.0):
    """선형 회귀에서 MAP = 벌주기임을 보인다."""
    
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y = 2.0 - 0.5 * X + np.random.normal(0, noise_std, n_samples)
    
    # 다항 특징(지나친 맞춤에 빠지기 쉽다)
    X_poly = np.column_stack([X**i for i in range(6)])
    
    # 1. MLE(벌주기 없음)
    beta_mle = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    
    # 2. 가우스 앞확률을 쓴 MAP = 능선
    lambda_ridge = 10.0
    beta_ridge = np.linalg.solve(
        X_poly.T @ X_poly + lambda_ridge * np.eye(6),
        X_poly.T @ y
    )
    
    # 3. 라플라스 앞확률을 쓴 MAP = 라소
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_poly, y)
    beta_lasso = np.concatenate([[lasso.intercept_], lasso.coef_[1:]])
    
    return beta_mle, beta_ridge, beta_lasso
```

**핵심 관찰:**

1. **최대 가능도**는 차수가 높은 다항식에서 지나치게 맞춘다(계수가 커진다)
2. **능선(가우스 앞확률)**은 모든 계수를 0 쪽으로 오그라뜨린다
3. **라소(라플라스 앞확률)**는 일부 계수를 딱 0으로 몬다(성김)

---

## 5. 최빈값과 평균은 언제 갈리는가

최대 뒤확률과 뒤확률의 평균은 늘 다른 값이 아니다. 언제 같고 언제 갈리는지는 뒤확률의 모양이 정한다.

### 정리 5. 최빈값과 평균의 갈림 — 대칭이면 일치한다

뒤확률 분포가 어떤 점 $c$을 중심으로 대칭이고 그 점에서 봉우리가 하나이면 $\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{Mean}} = c$이다. 대칭이 깨지면 둘은 갈린다.

*밝힘.* $p(c+u|D) = p(c-u|D)$이면 $\mathbb{E}[\theta - c|D] = \int u\,p(c+u|D)\,du = 0$이므로 평균은 $c$이다. 봉우리가 $c$ 하나이므로 최빈값도 $c$이다. $\square$

| 분포 | 최빈값 | 평균 | 관계 |
|--------------|------|------|--------------|
| Gamma$(\alpha, \beta)$ | $(\alpha-1)/\beta$ | $\alpha/\beta$ | 평균 > 최빈값 |
| Beta$(\alpha, \beta)$, $\alpha > \beta$ | $(\alpha-1)/(\alpha+\beta-2)$ | $\alpha/(\alpha+\beta)$ | 매개변수에 달렸다 |

| 잣대 | 나은 어림기 |
|-----------|---------------------|
| 제곱 오차를 가장 작게 | 뒤확률의 평균 |
| 가장 그럴듯한 값 | 최대 뒤확률 |
| 셈의 단순함 | 최대 뒤확률(최적화) |
| 불확실성을 온전히 나타내기 | 뒤확률 전체 |

---

## 6. 정리하며

최대 뒤확률은 뒤확률 분포의 최빈값이다. 정리 1이 보였듯 이는 0-1 손실 아래의 베이즈 어림기이며, 앞확률이 고르면 최대 가능도로 돌아간다. 정리 2는 켤레 앞확률에서 이 값이 닫힌 꼴로 나오고 앞확률이 유사 관측처럼 더해짐을 보였다.

이 마당의 고갱이는 정리 3과 정리 4다. 최대 뒤확률은 다루기 어려운 증거 적분을 건너뛰고 로그가능도와 로그앞확률의 합을 가장 크게 하는 최적화 문제가 된다. 그리고 그 로그앞확률 항이 바로 벌 항이다. 가우스 앞확률은 능선 회귀를, 라플라스 앞확률은 라소 회귀를 낳는다. 곧 **벌주기는 곧 베이즈이다**. 우리가 쓰는 모든 벌 항 뒤에는 매개변수 위의 앞확률 분포가 숨어 있다.

정리 5는 이 점 어림값의 한계를 짚는다. 뒤확률이 대칭이면 최빈값과 평균이 일치하지만, 기울면 갈린다. 그리고 어느 쪽을 고르든 점 어림값은 불확실성을 버린다. 뒤확률 전체가 필요한 자리에서는 [믿음 구간](credible_intervals.md)과 [MCMC](../../ch15/mcmc/gibbs_sampling.md)로 넘어가야 한다.

**참고 문헌**

- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 7장
- Bishop, C. *Pattern Recognition and Machine Learning*, 3장
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 5장

---

## 연습문제

**연습문제 1.**
베타-이항 모형에서 고른 앞확률($\text{Beta}(1,1)$)일 때 최대 뒤확률이 최대 가능도와 같음을 해석적으로 보여라.

??? success "연습문제 1 풀이"
    정리 2에서 $\alpha = \beta = 1$을 넣으면 $\hat{\theta}_{\text{MAP}} = (1+k-1)/(1+1+n-2) = k/n$이 되어 $\hat{\theta}_{\text{MLE}}$과 같다. 정리 1의 (1)이 말하는 바를 닫힌 꼴로 확인한 것이다. 다만 뒤확률의 평균은 $(1+k)/(2+n)$으로 여전히 다르며, 이는 고른 앞확률조차 관측 두 번의 몫을 한다는 뜻이다.

---

**연습문제 2.**
앞확률의 세기를 키울 때(베타-이항에서 $\alpha, \beta$을 대칭으로 키울 때) 최대 뒤확률 어림값이 어떻게 바뀌는지 보여라. 데이터를 붙박아 두고 앞확률 세기에 따른 최대 뒤확률을 그려라.

??? success "연습문제 2 풀이"
    $\alpha = \beta = c$으로 두면 정리 2에서 $\hat{\theta}_{\text{MAP}} = (c+k-1)/(2c+n-2)$이다. $c \to \infty$이면 이 값은 $1/2$으로 간다. 곧 앞확률이 셀수록 데이터를 눌러 이기고 어림값이 앞확률의 최빈값으로 끌려간다. $c = 1$에서는 최대 가능도 $k/n$과 같고, $c$가 커질수록 그 사이를 매끄럽게 지나간다.

---

**연습문제 3.**
계수에 가우스 앞확률을 둔 로지스틱 회귀의 최대 뒤확률 어림을 구현하라. 벌주기 없는 최대 가능도와 견주어라.

??? success "연습문제 3 풀이"
    정리 3의 꼴에 따라 $-\sum_i [y_i \log \sigma(x_i^\top\beta) + (1-y_i)\log(1-\sigma(x_i^\top\beta))] + \frac{\lambda}{2}\|\beta\|_2^2$을 가장 작게 한다. `scipy.optimize.minimize`에 이 목표와 그 기울기 $X^\top(\sigma(X\beta) - y) + \lambda\beta$을 넘기면 된다. 데이터가 선형으로 갈리는 경우 최대 가능도는 계수가 무한대로 뻗지만, 가우스 앞확률을 두면 유한한 해가 남는다. 이것이 정리 4가 말하는 능선 벌주기의 효과다.

---

**연습문제 4.**
능선 회귀가 계수마다 서로 독립인 가우스 앞확률을 둔 최대 뒤확률 어림과 같음을 수학으로 증명하라.

??? success "연습문제 4 풀이"
    정리 4의 (1)이 그 증명이다. 가능도가 $y|X,\beta \sim \mathcal{N}(X\beta, \sigma^2 I)$이면 $-\log p(D|\beta) = \frac{1}{2\sigma^2}\|y-X\beta\|_2^2 + \text{const}$이고, 앞확률이 $\beta_j \sim \mathcal{N}(0,\sigma_\beta^2)$이면 $-\log p(\beta) = \frac{1}{2\sigma_\beta^2}\|\beta\|_2^2 + \text{const}$이다. 둘을 더해 $2\sigma^2$을 곱하면 $\|y-X\beta\|_2^2 + \lambda\|\beta\|_2^2$이며 $\lambda = \sigma^2/\sigma_\beta^2$이다.

---

**연습문제 5.**
어떤 뒤확률 분포에서 최대 뒤확률과 뒤확률의 평균이 가장 많이 갈리는가? 감마와 베타 분포로 그 갈림을 보여 주는 보기를 만들어라.

??? success "연습문제 5 풀이"
    정리 5에 따라 대칭이 크게 깨질수록 갈림이 커진다. $\text{Gamma}(\alpha,\beta)$에서 평균과 최빈값의 차는 $\alpha/\beta - (\alpha-1)/\beta = 1/\beta$으로 붙박여 있으나, 값의 자로 나눈 상대 갈림 $1/\alpha$은 $\alpha \to 1^+$에서 커진다. 곧 모양 매개변수가 작아 크게 기운 뒤확률에서 갈림이 가장 두드러진다. $\text{Beta}(1.2, 8)$처럼 한쪽 끝에 몰린 경우도 마찬가지다.

---

**연습문제 6.**
최대 뒤확률은 뒤확률 전체에 견주어 어떤 정보를 잃는가? 온전한 베이즈 추론보다 최대 뒤확률이 나을 때는 언제인가?

??? success "연습문제 6 풀이"
    최대 뒤확률은 점 어림값(최빈값)만 주고 뒤확률의 불확실성에 대한 정보를 모두 버린다. 곧 우리가 얼마나 자신하는지, 분포가 얼마나 넓은지, 봉우리가 여럿인지 같은 것이다. 뒤확률 전체는 불확실성 재기, 예측 분포, 주변 가능도를 거친 모형 견줌을 준다. 그래도 다음일 때는 최대 뒤확률이 낫다. (1) 뒤확률이 거의 가우스일 때(최대 뒤확률에 라플라스 어림을 곁들이면 넉넉하다), (2) 셈 자원이 빠듯할 때(MCMC가 필요 없다), (3) 점 예측만 필요할 때, (4) 모형이 아주 클 때(깊은 망에서는 온전한 베이즈 추론을 다룰 수 없다).

---

## 정리하며

이 마당은 베이즈 추론의 점 어림값 셋、베타-이항 모형의 닫힌 꼴、수치 최적화로 하는 최대 뒤확률、최대 뒤확률 어림과 벌주기을 차례로 짚었다.
