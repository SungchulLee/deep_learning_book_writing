# 최대 뒤확률(MAP) 어림

최대 뒤확률 어림은 뒤확률 분포의 최빈값을 찾아, 앞선 정보를 아우른 점 어림값을 준다. 이 마당은 최대 뒤확률을 최대 가능도, 뒤확률의 평균과 견주고, 최대 뒤확률을 위한 수치 최적화 방법을 세우며, 최대 뒤확률 어림과 벌주기 사이의 근본적인 이음을 밝힌다.

## 1. 베이즈 추론의 점 어림값 셋

**정의 1.** [세 가지 점 어림값]

뒤확률 분포 $p(\theta \mid D)$ 하나에서 값 하나를 뽑아내는 길은 여럿이다. 널리 쓰이는 세 가지는 다음과 같다.

$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \; p(D \mid \theta), \qquad
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \; p(\theta \mid D), \qquad
\hat{\theta}_{\text{Mean}} = \mathbb{E}[\theta \mid D]
$$

첫째는 앞확률을 아랑곳하지 않고 가능도만 보고, 둘째는 뒤확률의 **최빈값**을, 셋째는 뒤확률의 **평균**을 본다.

### 정리 1. 세 가지 점 어림값 — 손실 함수가 어림기를 고른다 { .thm }

앞확률 $p(\theta)$ 와 가능도 $p(D \mid \theta)$ 가 주어졌다고 하자. 그러면 다음이 성립한다.

1. 앞확률이 고르면 $\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{MLE}}$ 이다.
2. 0-1 손실 아래에서 베이즈 어림기는 $\hat{\theta}_{\text{MAP}}$ 이다.
3. 제곱 오차 손실 아래에서 베이즈 어림기는 $\hat{\theta}_{\text{Mean}}$ 이다.

??? proof "증명"

    (1) 고른 앞확률에서는 $\log p(\theta)$ 가 상수이므로

    $$
    \underset{\theta}{\arg\max}\,[\log p(D \mid \theta) + \log p(\theta)]
    = \underset{\theta}{\arg\max}\,\log p(D \mid \theta)
    $$

    이다.

    (3) 기대 손실 $\mathbb{E}[(\theta - a)^2 \mid D]$ 를 $a$ 에 대해 미분하여 $0$ 으로 두면

    $$
    -2\,\mathbb{E}[\theta - a \mid D] = 0 \;\Longrightarrow\; a = \mathbb{E}[\theta \mid D]
    $$

    를 얻는다.

    (2) 손실 $L_\epsilon(\theta, a) = \mathbf{1}[\,|\theta - a| > \epsilon\,]$ 의 기대값을 가장 작게 하는 $a$ 는 폭 $2\epsilon$ 의 구간이 담는 뒤확률을 가장 크게 하는 자리이다. $\epsilon \to 0^{+}$ 으로 보내면 그 자리는 밀도가 가장 높은 점, 곧 최빈값으로 간다.

!!! note "쓰임새"
    어느 어림값을 고를지는 취향이 아니라 **어떤 손실을 치를 것인가**로 정해진다. 제곱 오차로 평가받는 예측이면 뒤확률의 평균을, 맞았는지 틀렸는지로만 평가받는 판정이면 최대 뒤확률을 쓴다.

| 어림기 | 정의 | 가장 작게 하는 손실 함수 | 앞확률 씀 |
|-----------|------------|------------------------|------------|
| 최대 가능도 | 가능도의 최빈값 | — | 아니오 |
| 최대 뒤확률 | 뒤확률의 최빈값 | 0-1 손실 | 예 |
| 뒤확률의 평균 | 뒤확률의 평균 | 제곱 오차 | 예 |

## 2. 베타-이항 모형의 닫힌 꼴

**정의 2.** [켤레 앞확률]

가능도 $p(D \mid \theta)$ 에 대해 앞확률 $p(\theta)$ 와 뒤확률 $p(\theta \mid D)$ 가 같은 분포족에 들면 그 앞확률을 **켤레 앞확률**이라 한다. 이항 가능도의 켤레 앞확률은 베타 분포이다.

### 정리 2. 베타-이항의 닫힌 꼴 — 앞확률은 유사 관측으로 더해진다 { .thm }

앞확률이 $\text{Beta}(\alpha, \beta)$ 이고 데이터가 앞면 $k$ 번, 뒷면 $n-k$ 번이면 뒤확률은 $\text{Beta}(\alpha + k,\; \beta + n - k)$ 이고, $\alpha + k > 1$ 이며 $\beta + n - k > 1$ 일 때 다음이 성립한다.

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2}, \qquad
\hat{\theta}_{\text{MLE}} = \frac{k}{n}, \qquad
\hat{\theta}_{\text{Mean}} = \frac{\alpha + k}{\alpha + \beta + n}
$$

??? proof "증명"

    뒤확률의 꼴은

    $$
    p(\theta \mid D) \propto \theta^{k}(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
    = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}
    $$

    이므로 $\text{Beta}(\alpha+k,\ \beta+n-k)$ 이다. $\text{Beta}(a,b)$ 의 최빈값은 $a,b>1$ 일 때 $(a-1)/(a+b-2)$ 이고 평균은 $a/(a+b)$ 이다. 여기에 $a = \alpha+k$, $b = \beta+n-k$ 를 넣으면 된다.

!!! note "쓰임새"
    앞확률 $\text{Beta}(\alpha,\beta)$ 는 **앞면 $\alpha$ 번, 뒷면 $\beta$ 번을 미리 본 것**과 같은 몫을 한다. 그래서 $\alpha,\beta$ 를 크게 잡을수록 데이터가 어림값을 덜 움직인다.

**보기 1.** <span class="diff easy" title="쉬움"></span> 데이터가 앞면 $7$ 번, 뒷면 $3$ 번이고 앞확률이 $\text{Beta}(2, 2)$ 일 때 세 어림값을 구하시오.

??? success "풀이"

    정리 2에 $\alpha=\beta=2$, $k=7$, $n=10$ 을 넣으면 다음과 같다.

    | 어림기 | 식 | 값 |
    |-----------|---------|-------|
    | 최대 가능도 | $7/10$ | $0.700$ |
    | 최대 뒤확률 | $(2+7-1)/(2+2+10-2) = 8/12$ | $0.667$ |
    | 뒤확률의 평균 | $(2+7)/(2+2+10) = 9/14$ | $0.643$ |

    앞확률 $\text{Beta}(2,2)$ 가 어림값을 $0.5$ 쪽으로 끌어당긴다. 앞확률이 더할 유사 관측이 $4$ 번뿐이라 데이터 $10$ 번에 견주어 끌어당기는 힘은 약하다.

**문제 1.** <span class="diff med" title="중간"></span> 위 보기에서 앞확률만 $\text{Beta}(10, 10)$ 으로 바꾸면 세 어림값이 어떻게 달라지는지 구하고, 그 까닭을 설명하시오.

??? success "풀이"

    $\hat{\theta}_{\text{MLE}} = 0.700$ 은 그대로이고,

    $$
    \hat{\theta}_{\text{MAP}} = \frac{10+7-1}{10+10+10-2} = \frac{16}{28} \approx 0.571,\qquad
    \hat{\theta}_{\text{Mean}} = \frac{10+7}{10+10+10} = \frac{17}{30} \approx 0.567
    $$

    이다. 앞확률이 유사 관측 $20$ 번을 더하므로 데이터 $10$ 번을 눌러 이기고, 두 베이즈 어림값이 모두 $0.5$ 에 훨씬 가까워진다.

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

## 3. 수치 최적화로 하는 최대 뒤확률

켤레가 아닌 모형에서는 닫힌 꼴이 없다. 그래도 최대 뒤확률은 늘 최적화 문제로 적을 수 있어서, 뒤확률 전체를 다루지 않고도 풀 수 있다.

### 정리 3. 최대 뒤확률의 최적화 꼴 — 로그가능도와 로그앞확률의 합 { .thm }

증거 $p(D)$ 는 $\theta$ 에 달리지 않으므로 다음이 성립한다.

$$
\hat{\theta}_{\text{MAP}}
= \underset{\theta}{\arg\max} \left[ \log p(D \mid \theta) + \log p(\theta) \right]
= \underset{\theta}{\arg\min} \left[ -\log p(D \mid \theta) - \log p(\theta) \right]
$$

??? proof "증명"

    베이즈 정리에서 $p(\theta \mid D) = p(D \mid \theta)\,p(\theta) / p(D)$ 이고 $\log$ 는 단조 증가하므로

    $$
    \underset{\theta}{\arg\max}\,\log p(\theta \mid D)
    = \underset{\theta}{\arg\max}\,[\log p(D \mid \theta) + \log p(\theta) - \log p(D)]
    $$

    이다. 마지막 항 $\log p(D)$ 는 $\theta$ 와 무관한 상수이므로 $\arg\max$ 에서 떨어져 나간다.

!!! note "쓰임새"
    다루기 어려운 적분 $p(D) = \int p(D \mid \theta)p(\theta)\,d\theta$ 를 **아예 셈하지 않아도 된다**는 것이 최대 뒤확률의 값어치다. 뒤확률 전체가 필요하면 이 적분을 피할 수 없지만, 최빈값 하나만 필요하면 피할 수 있다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 데이터가 $x_1,\dots,x_n \sim \mathcal{N}(\mu, \sigma^2)$ 이고 앞확률이 $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$, $\tau = 1/\sigma^2 \sim \text{Gamma}(\alpha, \beta)$ 일 때 가장 작게 할 목적함수를 적으시오.

??? success "풀이"

    정리 3에 따라 음의 로그 뒤확률을 적으면 다음과 같다.

    $$
    -\log p(\mu, \tau \mid D) = -\sum_{i=1}^n \log p(x_i \mid \mu, \tau) - \log p(\mu) - \log p(\tau) + \text{const}
    $$

    $\tau > 0$ 이라는 제약은 $\tau = e^{s}$ 로 바꾸어 $s$ 에 대해 제약 없이 최적화하면 사라진다.

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

## 4. 최대 뒤확률 어림과 벌주기

정리 3의 최적화 꼴을 다시 보면 $-\log p(\theta)$ 가 벌 항의 자리에 그대로 앉아 있다. 이것이 벌주기와 베이즈를 잇는 다리다.

### 정리 4. 앞확률과 벌주기의 대응 — 가우스는 능선, 라플라스는 라소 { .thm }

선형 회귀에서 계수마다 서로 독립인 앞확률을 두면 다음이 성립한다.

1. $\theta_j \sim \mathcal{N}(0, \sigma_\theta^2)$ 이면 최대 뒤확률은 능선(L2) 회귀와 같다.

$$
\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 \right]
$$

2. $\theta_j \sim \text{Laplace}(0, b)$ 이면 최대 뒤확률은 라소(L1) 회귀와 같다.

$$
\hat{\beta}_{\text{Lasso}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right]
$$

??? proof "증명"

    가능도가 $y \mid X, \beta \sim \mathcal{N}(X\beta, \sigma^2 I)$ 이면

    $$
    -\log p(D \mid \beta) = \frac{1}{2\sigma^2}\|y - X\beta\|_2^2 + \text{const}
    $$

    이다.

    (1) 가우스 앞확률에서는 $-\log p(\beta) = \frac{1}{2\sigma_\theta^2}\|\beta\|_2^2 + \text{const}$ 이다. 정리 3의 목적함수에 넣고 $2\sigma^2$ 을 곱하면 $\|y-X\beta\|_2^2 + \lambda\|\beta\|_2^2$ 이며 $\lambda = \sigma^2/\sigma_\theta^2$ 이다.

    (2) 라플라스 앞확률에서는 $-\log p(\beta) = \frac{1}{b}\|\beta\|_1 + \text{const}$ 이므로 같은 방식으로 $\lambda = 2\sigma^2/b$ 인 L1 벌 항이 나온다.

!!! note "쓰임새"
    **벌주기는 곧 베이즈이다.** 우리가 쓰는 모든 벌 항 뒤에는 매개변수 위의 앞확률 분포가 숨어 있고, 앞확률의 흩어짐이 벌주기의 세기 $\lambda$ 를 정한다. 앞확률이 좁을수록 벌이 세다.

| 앞확률 분포 | 벌주기 | 벌 항 | 효과 |
|-------------------|----------------|--------------|--------|
| 고른 분포 | 없음(최대 가능도) | — | 오그라듦 없음 |
| 가우스 $\mathcal{N}(0, \sigma^2)$ | 능선(L2) | $\lambda\|\theta\|_2^2$ | 계수를 오그라뜨린다 |
| 라플라스$(0, b)$ | 라소(L1) | $\lambda\|\theta\|_1$ | 성긴 해 |
| 편자(horseshoe) | 맞추어 가는 오그라듦 | 복잡함 | 강한 성김 |

**문제 2.** <span class="diff hard" title="어려움"></span> 라소가 계수를 **딱 $0$** 으로 모는 반면 능선은 그러지 못하는 까닭을, 두 앞확률의 밀도가 원점에서 보이는 차이로 설명하시오.

??? success "풀이"

    라플라스 밀도 $\propto e^{-|\theta|/b}$ 는 원점에서 뾰족한 모서리를 가져 미분할 수 없고, 벌 항 $|\theta|$ 의 열미분이 $\pm\lambda$ 로 뛴다. 그래서 데이터가 주는 기울기의 크기가 $\lambda$ 보다 작으면 $\theta = 0$ 이 최적이 되어 **정확히 $0$** 이 해로 남는다.

    가우스 밀도 $\propto e^{-\theta^2/2\sigma^2}$ 는 원점에서 매끄럽고 벌 항 $\theta^2$ 의 미분이 원점에서 $0$ 이다. 따라서 데이터의 기울기가 아무리 작아도 최적점은 $0$ 에 가까워질 뿐 결코 $0$ 에 닿지 않는다.

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

## 5. 최빈값과 평균은 언제 갈리는가

최대 뒤확률과 뒤확률의 평균은 늘 다른 값이 아니다. 언제 같고 언제 갈리는지는 뒤확률의 모양이 정한다.

### 정리 5. 최빈값과 평균의 갈림 — 대칭이면 일치한다 { .thm }

뒤확률 분포가 어떤 점 $c$ 를 중심으로 대칭이고 그 점에서 봉우리가 하나이면

$$
\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{Mean}} = c
$$

이다. 대칭이 깨지면 둘은 갈린다.

??? proof "증명"

    대칭이란 $p(c+u \mid D) = p(c-u \mid D)$ 를 뜻한다. 그러면

    $$
    \mathbb{E}[\theta - c \mid D] = \int u\,p(c+u \mid D)\,du = 0
    $$

    이다. 피적분함수가 홀함수이기 때문이다. 따라서 평균은 $c$ 이다. 봉우리가 $c$ 하나뿐이므로 최빈값도 $c$ 이다.

| 분포 | 최빈값 | 평균 | 관계 |
|--------------|------|------|--------------|
| $\text{Gamma}(\alpha, \beta)$ | $(\alpha-1)/\beta$ | $\alpha/\beta$ | 평균 $>$ 최빈값 |
| $\text{Beta}(\alpha, \beta)$, $\alpha > \beta$ | $(\alpha-1)/(\alpha+\beta-2)$ | $\alpha/(\alpha+\beta)$ | 매개변수에 달렸다 |

**문제 3.** <span class="diff med" title="중간"></span> $\text{Gamma}(\alpha, \beta)$ 뒤확률에서 평균과 최빈값의 차를 구하고, 어떤 $\alpha$ 에서 상대 갈림이 가장 큰지 밝히시오.

??? success "풀이"

    차는

    $$
    \frac{\alpha}{\beta} - \frac{\alpha-1}{\beta} = \frac{1}{\beta}
    $$

    로 $\alpha$ 와 무관하게 붙박여 있다. 그러나 평균으로 나눈 상대 갈림은

    $$
    \frac{1/\beta}{\alpha/\beta} = \frac{1}{\alpha}
    $$

    이므로 $\alpha \to 1^{+}$ 에서 가장 커진다. 곧 모양 매개변수가 작아 크게 기운 뒤확률일수록 두 어림값이 크게 갈린다.

| 잣대 | 나은 어림기 |
|-----------|---------------------|
| 제곱 오차를 가장 작게 | 뒤확률의 평균 |
| 가장 그럴듯한 값 | 최대 뒤확률 |
| 셈의 단순함 | 최대 뒤확률(최적화) |
| 불확실성을 온전히 나타내기 | 뒤확률 전체 |

## 연습문제

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 베타-이항 모형에서 고른 앞확률($\text{Beta}(1,1)$)일 때 최대 뒤확률이 최대 가능도와 같음을 해석적으로 보이시오.

??? success "풀이"

    정리 2에 $\alpha = \beta = 1$ 을 넣으면

    $$
    \hat{\theta}_{\text{MAP}} = \frac{1+k-1}{1+1+n-2} = \frac{k}{n} = \hat{\theta}_{\text{MLE}}
    $$

    이다. 정리 1의 (1)을 닫힌 꼴로 확인한 것이다. 다만 뒤확률의 평균은 $(1+k)/(2+n)$ 으로 여전히 다르며, 이는 고른 앞확률조차 관측 두 번의 몫을 한다는 뜻이다.

**연습문제 2.** <span class="diff med" title="중간"></span> 앞확률의 세기를 대칭으로 키울 때($\alpha = \beta = c$) 최대 뒤확률 어림값이 어디로 가는지 밝히시오.

??? success "풀이"

    정리 2에서

    $$
    \hat{\theta}_{\text{MAP}} = \frac{c+k-1}{2c+n-2} \;\xrightarrow[c \to \infty]{}\; \frac12
    $$

    이다. 앞확률이 셀수록 데이터를 눌러 이기고 어림값이 앞확률의 최빈값으로 끌려간다. $c = 1$ 에서는 최대 가능도 $k/n$ 과 같고, $c$ 가 커질수록 그 사이를 매끄럽게 지나간다.

**연습문제 3.** <span class="diff med" title="중간"></span> 계수에 가우스 앞확률을 둔 로지스틱 회귀의 최대 뒤확률 어림을 구현하고, 벌주기 없는 최대 가능도와 견주시오.

??? success "풀이"

    정리 3의 꼴에 따라

    $$
    -\sum_i \left[ y_i \log \sigma(x_i^\top\beta) + (1-y_i)\log(1-\sigma(x_i^\top\beta)) \right] + \frac{\lambda}{2}\|\beta\|_2^2
    $$

    를 가장 작게 한다. 기울기는 $X^\top(\sigma(X\beta) - y) + \lambda\beta$ 이다. 데이터가 선형으로 갈리는 경우 최대 가능도는 계수가 무한대로 뻗지만, 가우스 앞확률을 두면 유한한 해가 남는다. 이것이 정리 4가 말하는 능선 벌주기의 효과다.

**연습문제 4.** <span class="diff hard" title="어려움"></span> 능선 회귀가 계수마다 서로 독립인 가우스 앞확률을 둔 최대 뒤확률 어림과 같음을 증명하고, $\lambda$ 와 앞확률의 흩어짐 사이의 관계를 적으시오.

??? success "풀이"

    정리 4의 (1)이 그 증명이다. 결론만 다시 적으면

    $$
    \lambda = \frac{\sigma^2}{\sigma_\beta^2}
    $$

    이다. 곧 잡음이 클수록, 그리고 앞확률이 좁을수록 벌이 세진다. $\sigma_\beta \to \infty$ 이면 $\lambda \to 0$ 이 되어 최대 가능도로 돌아간다.

**연습문제 5.** <span class="diff med" title="중간"></span> 최대 뒤확률은 뒤확률 전체에 견주어 어떤 정보를 잃는가? 그럼에도 최대 뒤확률이 나을 때는 언제인지 밝히시오.

??? success "풀이"

    최대 뒤확률은 점 어림값(최빈값)만 주고 뒤확률의 불확실성을 모두 버린다. 곧 우리가 얼마나 자신하는지, 분포가 얼마나 넓은지, 봉우리가 여럿인지를 알 수 없다. 뒤확률 전체는 불확실성 재기, 예측 분포, 주변 가능도를 거친 모형 견줌을 준다.

    그래도 다음일 때는 최대 뒤확률이 낫다. (1) 뒤확률이 거의 가우스일 때(최대 뒤확률에 라플라스 어림을 곁들이면 넉넉하다), (2) 셈 자원이 빠듯할 때(MCMC가 필요 없다), (3) 점 예측만 필요할 때, (4) 모형이 아주 클 때(깊은 망에서는 온전한 베이즈 추론을 다룰 수 없다).

## 정리하며

최대 뒤확률은 뒤확률 분포의 **최빈값**이다.

1. 어느 점 어림값을 쓸지는 치를 손실이 정한다. 0-1 손실이면 최대 뒤확률, 제곱 오차면 뒤확률의 평균이다(정리 1).
2. 켤레 앞확률에서는 닫힌 꼴이 나오고, 앞확률은 유사 관측처럼 데이터에 더해진다(정리 2).
3. 최대 뒤확률은 다루기 어려운 증거 적분을 건너뛰고 로그가능도와 로그앞확률의 합을 가장 크게 하는 최적화 문제가 된다(정리 3).
4. 그 로그앞확률 항이 바로 벌 항이다. 가우스 앞확률은 능선을, 라플라스 앞확률은 라소를 낳는다(정리 4).

곧 **벌주기는 곧 베이즈이다.** 다만 대칭이 깨진 뒤확률에서는 최빈값과 평균이 갈리고(정리 5), 어느 쪽을 고르든 점 어림값은 불확실성을 버린다. 뒤확률 전체가 필요한 자리에서는 「[믿음 구간](credible_intervals.md)」과 「[깁스 표집](../../ch16/mcmc/gibbs_sampling.md)」으로 넘어가야 한다.

**참고 문헌**

- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 7장
- Bishop, C. *Pattern Recognition and Machine Learning*, 3장
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 5장
