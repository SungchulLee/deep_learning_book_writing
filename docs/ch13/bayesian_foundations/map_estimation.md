# 최대 뒤확률(MAP) 어림
## 개요

최대 뒤확률 어림은 뒤확률 분포의 최빈값을 찾아, 앞선 정보를 아우른 점 어림값을 준다. 이 모듈은 최대 뒤확률을 최대 가능도, 뒤확률의 평균과 견주고, 최대 뒤확률을 위한 수치 최적화 방법을 세우며, 최대 뒤확률 어림과 벌주기 사이의 근본적인 이음을 밝힌다.

---

## 1. 베이즈 추론의 점 어림값 셋

### 1.1 최대 가능도 어림값(MLE)

최대 가능도 어림값은 앞선 정보를 아랑곳하지 않고 가능도 함수를 가장 크게 한다.

$$
\boxed{\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \; p(D|\theta)}
$$

- 빈도주의 방법
- 앞선 정보를 아우르지 않는다
- 데이터가 적으면 지나치게 맞출 수 있다

### 1.2 최대 뒤확률(MAP)

최대 뒤확률 어림값은 뒤확률 분포를 가장 크게 한다.

$$
\boxed{\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \; p(\theta|D) = \underset{\theta}{\arg\max} \; p(D|\theta) \, p(\theta)}
$$

- 앞선 정보를 아우른다
- 뒤확률 분포의 최빈값이다
- 앞확률이 고르면 최대 가능도와 같다

### 1.3 뒤확률의 평균

뒤확률의 평균은 뒤확률 아래에서의 기댓값이다.

$$
\boxed{\hat{\theta}_{\text{Mean}} = \mathbb{E}[\theta|D] = \int \theta \, p(\theta|D) \, d\theta}
$$

- 기대 제곱 오차를 가장 작게 한다(이차 손실 아래의 베이즈 어림기)
- 기운 뒤확률에서는 최대 뒤확률과 다를 때가 많다
- 적분이 필요하다(해석적이든 수치적이든)

### 1.4 견줌 간추림

| 어림기 | 정의 | 가장 작게 하는 손실 함수 | 앞확률 씀 |
|-----------|------------|------------------------|------------|
| 최대 가능도 | 가능도의 최빈값 | — | 아니오 |
| 최대 뒤확률 | 뒤확률의 최빈값 | 0-1 손실 | 예 |
| 뒤확률의 평균 | 뒤확률의 평균 | 제곱 오차 | 예 |

---

## 2. 베타-이항 모형의 최대 뒤확률

### 2.1 닫힌 꼴의 해

앞확률 Beta$(\alpha, \beta)$과 데이터 $(k, n-k)$을 갖춘 베타-이항 모형에서는 다음과 같다.

**뒤확률:** Beta$(\alpha + k, \beta + n - k)$

**최대 뒤확률 어림값**(베타 분포의 최빈값):

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2} \quad \text{for } \alpha + k > 1, \beta + n - k > 1
$$

**최대 가능도:**

$$
\hat{\theta}_{\text{MLE}} = \frac{k}{n}
$$

**뒤확률의 평균:**

$$
\hat{\theta}_{\text{Mean}} = \frac{\alpha + k}{\alpha + \beta + n}
$$

### 2.2 셈 보기

**데이터:** 앞면 7번, 뒷면 3번($k=7$, $n=10$)

**앞확률:** Beta$(2, 2)$

| 어림기 | 식 | 값 |
|-----------|---------|-------|
| 최대 가능도 | \$7/10$ | 0.700 |
| 최대 뒤확률 | $(2+7-1)/(2+2+10-2) = 8/12$ | 0.667 |
| 뒤확률의 평균 | $(2+7)/(2+2+10) = 9/14$ | 0.643 |

앞확률 Beta$(2, 2)$이 어림값을 0.5 쪽으로 끌어당긴다.

### 2.3 구현

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

### 3.1 닫힌 꼴의 해가 없을 때

복잡한 모형에서는 로그 뒤확률을 수치로 가장 크게 한다.

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \left[ \log p(D|\theta) + \log p(\theta) \right]
$$

같은 말로 음의 로그 뒤확률을 가장 작게 한다.

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\min} \left[ -\log p(D|\theta) - \log p(\theta) \right]
$$

### 3.2 보기: 평균과 흩어짐을 모르는 정규

**모형:**

- 데이터: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$
- 평균의 앞확률: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$
- 정밀도의 앞확률: $\tau = 1/\sigma^2 \sim \text{Gamma}(\alpha, \beta)$

**음의 로그 뒤확률:**

$$
-\log p(\mu, \tau | D) = -\sum_{i=1}^n \log p(x_i|\mu, \tau) - \log p(\mu) - \log p(\tau) + \text{const}
$$

### 3.3 구현

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

### 3.4 최적화 요령

| 기법 | 목적 |
|-----------|---------|
| 양의 매개변수에 로그 변환 | 제약 없는 최적화 |
| (뒤확률이 아니라) 로그 뒤확률 쓰기 | 수치 안정성 |
| 여러 곳에서 초기화 | 국소 최적을 피함 |
| 기울기 기반 방법(BFGS, L-BFGS) | 매끄러운 뒤확률에 효율적 |

---

## 4. 최대 뒤확률 어림과 벌주기

### 4.1 근본적인 이음

특정 앞확률을 쓴 최대 뒤확률 어림은 벌주기를 곁들인 최대 가능도와 같다.

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\min} \left[ -\log p(D|\theta) + \lambda \cdot R(\theta) \right]
$$

여기서 $R(\theta)$은 앞확률이 정하는 벌주기 항이다.

### 4.2 가우스 앞확률 ↔ 능선 회귀(L2)

**앞확률:** $\theta_j \sim \mathcal{N}(0, \sigma_\theta^2)$이며 서로 독립이다

**로그 앞확률:** $\log p(\theta) \propto -\frac{1}{2\sigma_\theta^2} \sum_j \theta_j^2 = -\frac{\lambda}{2} \|\theta\|_2^2$

**선형 회귀의 최대 뒤확률 목표:**

$$
\boxed{\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 \right]}
$$

### 4.3 라플라스 앞확률 ↔ 라소 회귀(L1)

**앞확률:** $\theta_j \sim \text{Laplace}(0, b)$이며 서로 독립이다

**로그 앞확률:** $\log p(\theta) \propto -\frac{1}{b} \sum_j |\theta_j| = -\lambda \|\theta\|_1$

**선형 회귀의 최대 뒤확률 목표:**

$$
\boxed{\hat{\beta}_{\text{Lasso}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right]}
$$

### 4.4 앞확률과 벌주기의 대응 간추림

| 앞확률 분포 | 벌주기 | 벌 항 | 효과 |
|-------------------|----------------|--------------|--------|
| 고른 분포 | 없음(최대 가능도) | — | 오그라듦 없음 |
| 가우스 $\mathcal{N}(0, \sigma^2)$ | 능선(L2) | $\lambda\|\theta\|_2^2$ | 계수를 오그라뜨린다 |
| 라플라스$(0, b)$ | 라소(L1) | $\lambda\|\theta\|_1$ | 성긴 해 |
| 편자(horseshoe) | 맞추어 가는 오그라듦 | 복잡함 | 강한 성김 |

### 4.5 보여 주기

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

## 5. 최대 뒤확률과 뒤확률의 평균은 언제 갈리는가?

### 5.1 대칭 뒤확률

**대칭**인 뒤확률 분포(이를테면 정규)에서는 최빈값과 평균이 일치한다.

$$
\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{Mean}}
$$

### 5.2 기운 뒤확률

**기운** 뒤확률(이를테면 감마, 로그 정규, $\alpha \neq \beta$인 베타)에서는 둘이 갈린다.

| 분포 | 최빈값 | 평균 | 관계 |
|--------------|------|------|--------------|
| Gamma$(\alpha, \beta)$ | $(\alpha-1)/\beta$ | $\alpha/\beta$ | 평균 > 최빈값 |
| Beta$(\alpha, \beta)$, $\alpha > \beta$ | $(\alpha-1)/(\alpha+\beta-2)$ | $\alpha/(\alpha+\beta)$ | 매개변수에 달렸다 |

### 5.3 어림기 고르기

| 잣대 | 나은 어림기 |
|-----------|---------------------|
| 제곱 오차를 가장 작게 | 뒤확률의 평균 |
| 가장 그럴듯한 값 | 최대 뒤확률 |
| 셈의 단순함 | 최대 뒤확률(최적화) |
| 불확실성을 온전히 나타내기 | 뒤확률 전체 |

---

## 6. 핵심 요점

1. **최대 뒤확률 = 뒤확률의 최빈값**: 데이터와 앞확률이 주어졌을 때 가장 그럴듯한 매개변수 값 하나를 찾는다.

2. 최대 가능도와 달리 **최대 뒤확률은 앞선 정보를 아우른다**. 앞확률이 고르면 최대 뒤확률은 최대 가능도가 된다.

3. 대칭 뒤확률에서는 **최대 뒤확률 ≈ 뒤확률의 평균**이지만 기운 분포에서는 크게 갈릴 수 있다.

4. **가우스 앞확률을 쓴 최대 뒤확률 = 능선 벌주기(L2)**: 앞확률의 흩어짐이 벌주기의 세기를 다스린다.

5. **라플라스 앞확률을 쓴 최대 뒤확률 = 라소 벌주기(L1)**: 계수를 딱 0으로 몰아 성긴 해를 북돋운다.

6. **벌주기는 곧 베이즈이다**: 벌주기 항마다 매개변수 위의 앞확률 분포에 대응한다.

---

## 7. 연습문제

### 연습문제 1: 고른 앞확률
베타-이항 모형에서 고른 앞확률(Beta$(1,1)$)일 때 최대 뒤확률이 최대 가능도와 같음을 해석적으로 보여라.

### 연습문제 2: 앞확률의 세기
앞확률의 세기를 키울 때(베타-이항에서 $\alpha, \beta$을 대칭으로 키울 때) 최대 뒤확률 어림값이 어떻게 바뀌는지 보여라. 데이터를 붙박아 두고 앞확률 세기에 따른 최대 뒤확률을 그려라.

### 연습문제 3: 로지스틱 회귀
계수에 가우스 앞확률을 둔 로지스틱 회귀의 최대 뒤확률 어림을 구현하라. 벌주기 없는 최대 가능도와 견주어라.

### 연습문제 4: 능선 회귀 끌어내기
능선 회귀가 계수마다 서로 독립인 가우스 앞확률을 둔 최대 뒤확률 어림과 같음을 수학으로 증명하라.

### 연습문제 5: 기운 뒤확률
어떤 뒤확률 분포에서 최대 뒤확률과 뒤확률의 평균이 가장 많이 갈리는가? 감마와 베타 분포로 그 갈림을 보여 주는 보기를 만들어라.

---

## 참고 문헌

- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 7장
- Bishop, C. *Pattern Recognition and Machine Learning*, 3장
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 5장

## 연습문제

**연습문제 1.**
최대 뒤확률 어림을 끌어내고 고른 앞확률에서 최대 가능도가 됨을 보여라.

??? success "연습문제 1 풀이"
    최대 뒤확률: $\theta_{\text{MAP}} = \arg\max_\theta p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$. 고른 앞확률에서는 $\log p(\theta) = \text{const}$이므로 최대 뒤확률 = 최대 가능도이다. 가우스 앞확률에서는 최대 뒤확률 = L2 벌주기 최대 가능도이다. 라플라스 앞확률에서는 최대 뒤확률 = L1 벌주기 최대 가능도이다.

---

**연습문제 2.**
가우스 앞확률 $\mathcal{N}(0, \sigma^2)$을 쓴 최대 뒤확률이 능선 회귀와 같음을 보여라.

??? success "연습문제 2 풀이"
    최대 뒤확률 목표: $\arg\min_\theta [-\log p(D|\theta) - \log p(\theta)] = \arg\min_\theta [\text{NLL} + \frac{\|\theta\|^2}{2\sigma^2}]$. $\lambda = 1/(2\sigma^2)$으로 두면 능선 회귀 $\text{NLL} + \lambda\|\theta\|^2$이 나온다.

---

**연습문제 3.**
최대 뒤확률은 뒤확률 전체에 견주어 어떤 정보를 잃는가?

??? success "연습문제 3 풀이"
    최대 뒤확률은 점 어림값(최빈값)만 주고 뒤확률의 불확실성에 대한 정보를 모두 버린다. 곧 우리가 얼마나 자신하는지, 분포가 얼마나 넓은지, 봉우리가 여럿인지 같은 것이다. 뒤확률 전체는 불확실성 재기, 예측 분포, 주변 가능도를 거친 모형 견줌을 준다.

---

**연습문제 4.**
온전한 베이즈 추론보다 최대 뒤확률 어림이 나을 때는 언제인가?

??? success "연습문제 4 풀이"
    다음일 때 최대 뒤확률이 낫다. (1) 뒤확률이 거의 가우스일 때(최대 뒤확률에 라플라스 어림을 곁들이면 넉넉하다), (2) 셈 자원이 빠듯할 때(MCMC가 필요 없다), (3) 점 예측만 필요할 때(불확실성을 나타낼 필요가 없다), (4) 모형이 아주 클 때(깊은 망에서는 온전한 베이즈 추론을 다룰 수 없다).
