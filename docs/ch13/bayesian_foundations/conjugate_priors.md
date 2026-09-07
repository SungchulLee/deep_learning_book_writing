# 켤레 앞확률
## 개요

켤레 앞확률은 수치 적분이나 표집 없이도 베이즈 추론 문제를 해석적으로 풀 수 있게 한다. 이 모듈은 켤레성의 이론을 세우고 근본이 되는 켤레족 셋, 곧 베타-이항, 감마-푸아송, 정규-정규를 내보인다.

---

## 1. 켤레 앞확률의 이론

### 1.1 정의

뒤확률 분포 $p(\theta|D)$이 앞확률과 같은 모수족에 들면, 앞확률 분포 $p(\theta)$은 가능도 함수 $p(D|\theta)$의 **켤레**이다.

정식으로, $\mathcal{F}$이 분포족이고 다음이 성립하면

$$
p(\theta) \in \mathcal{F} \implies p(\theta|D) \in \mathcal{F} \quad \text{for all data } D
$$

$\mathcal{F}$은 그 가능도의 **켤레족**이다.

### 1.2 켤레 앞확률의 이점

| 이점 | 설명 |
|-----------|-------------|
| **해석적 뒤확률** | 수치 셈 없이 닫힌 꼴의 해 |
| **셈 효율** | 수치 적분이나 MCMC 표집이 필요 없다 |
| **풀이할 수 있는 갱신** | 뜻이 또렷한 단순한 매개변수 변환 |
| **차례 갱신** | 온라인 학습이 쉽다 |
| **수학적 우아함** | 추론의 짜임에 대한 깊은 통찰 |

### 1.3 흔한 켤레족

| 앞확률 | 가능도 | 뒤확률 | 쓰임새 |
|-------|------------|-----------|----------|
| 베타 | 이항/베르누이 | 베타 | 이진 결과, 비율 |
| 감마 | 푸아송 | 감마 | 세는 데이터, 비율 |
| 감마 | 지수 | 감마 | 기다리는 시간, 수명 |
| 정규 | 정규($\sigma^2$을 알 때) | 정규 | 이어진 측정값 |
| 정규-역감마 | 정규 | 정규-역감마 | 평균과 흩어짐을 모를 때 |
| 디리클레 | 다항 | 디리클레 | 갈래 결과 |

---

## 2. 베타-이항 모형

### 2.1 모형 명세

**앞확률:** $\theta \sim \text{Beta}(\alpha, \beta)$

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

**가능도:** $k | n, \theta \sim \text{Binomial}(n, \theta)$

$$
p(k|n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

**뒤확률:** $\theta | k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$

### 2.2 켤레 갱신

갱신 규칙에는 우아한 풀이가 있다.

$$
\boxed{\alpha_{\text{post}} = \alpha_{\text{prior}} + k, \quad \beta_{\text{post}} = \beta_{\text{prior}} + (n-k)}
$$

여기서 각 기호는 다음과 같다.

- $k$ = 관찰한 성공 횟수
- $n - k$ = 관찰한 실패 횟수
- $\alpha - 1$ = 앞확률의 가짜 성공 횟수
- $\beta - 1$ = 앞확률의 가짜 실패 횟수

### 2.3 뒤확률 예측 분포

앞으로의 시행 $m$번에서 성공 $y$번에 대한 뒤확률 예측은 **베타-이항 분포**이다.

$$
P(y | D) = \int_0^1 \text{Binomial}(y | m, \theta) \cdot \text{Beta}(\theta | \alpha', \beta') \, d\theta
$$

$$
= \binom{m}{y} \frac{B(y + \alpha', m - y + \beta')}{B(\alpha', \beta')}
$$

### 2.4 구현

```python
from scipy import stats
import numpy as np

class BetaBinomialModel:
    """이진 자료를 위한 베타-이항 켤레 모형."""
    
    def __init__(self, alpha=1, beta=1):
        """Beta(alpha, beta) 앞확률로 첫걸음을 잡는다."""
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data_history = []
    
    def update(self, successes, trials):
        """관측 자료로 뒤확률을 새로 고친다."""
        failures = trials - successes
        self.posterior_alpha += successes
        self.posterior_beta += failures
        self.data_history.append((successes, trials))
    
    def posterior_predictive(self, n_trials=1):
        """뒤확률 예측 확률을 셈한다."""
        y_values = np.arange(n_trials + 1)
        probs = []
        
        for y in y_values:
            prob = (stats.binom.comb(n_trials, y) * 
                   stats.beta.beta_func(y + self.posterior_alpha, 
                                       n_trials - y + self.posterior_beta) / 
                   stats.beta.beta_func(self.posterior_alpha, 
                                       self.posterior_beta))
            probs.append(prob)
        
        return np.array(probs)
    
    def summary(self):
        """간추린 통계량을 찍는다."""
        post_dist = stats.beta(self.posterior_alpha, self.posterior_beta)
        
        print(f"Posterior: Beta({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean: {post_dist.mean():.4f}")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 2.5 보기: 동전 던지기

**앞확률:** Beta$(2, 2)$ — 공정하다는 약한 믿음

**데이터:** 20번 가운데 17번 성공

**뒤확률:** Beta$(2 + 17, 2 + 3) = $ Beta$(19, 5)$

| 통계량 | 값 |
|-----------|-------|
| 뒤확률의 평균 | \$19/24 = 0.792$ |
| 뒤확률의 최빈값 | \$18/22 = 0.818$ |
| 95% 믿음 구간 | $[0.60, 0.93]$ |

---

## 3. 감마-푸아송 모형

### 3.1 모형 명세

**앞확률:** $\lambda \sim \text{Gamma}(\alpha, \beta)$

$$
p(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda}
$$

**가능도:** $x_i | \lambda \sim \text{Poisson}(\lambda)$이며 서로 독립이다

$$
p(x_1, \ldots, x_n | \lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
$$

**뒤확률:** $\lambda | x_1, \ldots, x_n \sim \text{Gamma}(\alpha + \sum x_i, \beta + n)$

### 3.2 켤레 갱신

$$
\boxed{\alpha_{\text{post}} = \alpha_{\text{prior}} + \sum_{i=1}^n x_i, \quad \beta_{\text{post}} = \beta_{\text{prior}} + n}
$$

**해석:**

- $\alpha$ = 앞확률의 가짜 세기(전체 사건 수)
- $\beta$ = 앞확률의 가짜 관찰 수(기간의 개수)
- 갱신: 실제 전체 세기를 $\alpha$에, 관찰 수를 $\beta$에 더한다

### 3.3 앞확률과 뒤확률의 통계량

| 통계량 | 앞확률 | 뒤확률 |
|-----------|-------|-----------|
| 평균 | $\alpha/\beta$ | $(\alpha + \sum x_i)/(\beta + n)$ |
| 흩어짐 | $\alpha/\beta^2$ | $(\alpha + \sum x_i)/(\beta + n)^2$ |
| 최빈값 | $(\alpha-1)/\beta$ | $(\alpha + \sum x_i - 1)/(\beta + n)$ |

### 3.4 뒤확률 예측 분포

앞으로의 세기에 대한 뒤확률 예측은 **음이항 분포**이다.

$$
P(x_{\text{new}} | D) = \text{NegBinom}\left(x_{\text{new}} \,\Big|\, \alpha', \frac{\beta'}{\beta' + 1}\right)
$$

### 3.5 구현

```python
class GammaPoissonModel:
    """세기 자료를 위한 감마-푸아송 켤레 모형."""
    
    def __init__(self, alpha=1, beta=1):
        """Gamma(alpha, beta) 앞확률로 첫걸음을 잡는다."""
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.posterior_alpha = alpha
        self.posterior_beta = beta
        self.data = []
    
    def update(self, counts):
        """관측된 세기로 뒤확률을 새로 고친다."""
        counts = np.asarray(counts)
        self.posterior_alpha += np.sum(counts)
        self.posterior_beta += len(counts)
        self.data.extend(counts)
    
    def posterior_predictive(self):
        """뒤확률 예측 분포(음이항)를 되돌린다."""
        n = self.posterior_alpha
        p = self.posterior_beta / (self.posterior_beta + 1)
        return stats.nbinom(n, p)
    
    def summary(self):
        """간추린 통계량을 찍는다."""
        post_dist = stats.gamma(self.posterior_alpha, 
                                scale=1/self.posterior_beta)
        
        print(f"Posterior: Gamma({self.posterior_alpha}, {self.posterior_beta})")
        print(f"  Mean (rate): {post_dist.mean():.4f}")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 3.6 보기: 누리집 방문

**앞확률:** Gamma$(2, 1)$ — 기간마다 사건이 2번쯤 있으리라 본다

**데이터:** 날마다의 방문 $[5, 3, 7, 4, 6, 5, 8, 3, 4, 6]$(합 = 51, n = 10)

**뒤확률:** Gamma$(2 + 51, 1 + 10) = $ Gamma$(53, 11)$

| 통계량 | 값 |
|-----------|-------|
| 뒤확률의 평균 | \$53/11 = 4.82$ |
| 표본 평균 | \$51/10 = 5.10$ |
| 95% 믿음 구간 | $[3.61, 6.22]$ |

---

## 4. 정규-정규 모형(흩어짐을 알 때)

### 4.1 모형 명세

**앞확률:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**가능도:** $x_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$이며 서로 독립이고 $\sigma^2$은 안다

**뒤확률:** $\mu | x_1, \ldots, x_n \sim \mathcal{N}(\mu_n, \sigma_n^2)$

### 4.2 켤레 갱신(정밀도 꼴)

**정밀도**를 흩어짐의 역수로 정의한다: $\tau = 1/\sigma^2$

$$
\tau_n = \tau_0 + n\tau_{\text{data}}
$$

$$
\boxed{\mu_n = \frac{\tau_0 \mu_0 + n\tau_{\text{data}} \bar{x}}{\tau_n}, \quad \sigma_n^2 = \frac{1}{\tau_n}}
$$

여기서 $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$은 표본 평균이다.

### 4.3 풀이

뒤확률의 평균은 **정밀도로 무게 준 평균**이다.

$$
\mu_n = w_{\text{prior}} \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}
$$

여기서 각 기호는 다음과 같다.

- $w_{\text{prior}} = \tau_0 / \tau_n$ — 앞확률 평균의 무게
- $w_{\text{data}} = n\tau_{\text{data}} / \tau_n$ — 표본 평균의 무게

$n \to \infty$이면 $w_{\text{data}} \to 1$이고 $\mu_n \to \bar{x}$이다.

### 4.4 구현

```python
class NormalNormalModel:
    """정규-정규 켤레 모형(흩어짐을 아는 경우)."""
    
    def __init__(self, prior_mean=0, prior_std=1, known_std=1):
        """N(prior_mean, prior_std^2) 앞확률로 첫걸음을 잡는다."""
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.known_std = known_std
        
        self.posterior_mean = prior_mean
        self.posterior_std = prior_std
        self.data = []
    
    def update(self, observations):
        """새 관측으로 뒤확률을 새로 고친다."""
        observations = np.asarray(observations)
        n = len(observations)
        x_bar = np.mean(observations)
        
        # 정밀도 셈하기
        prior_precision = 1 / (self.prior_std ** 2)
        data_precision = n / (self.known_std ** 2)
        posterior_precision = prior_precision + data_precision
        
        # 매개변수 갱신
        self.posterior_mean = ((prior_precision * self.prior_mean + 
                               data_precision * x_bar) / posterior_precision)
        self.posterior_std = np.sqrt(1 / posterior_precision)
        
        # 잇단 새로 고치기용
        self.prior_mean = self.posterior_mean
        self.prior_std = self.posterior_std
        
        self.data.extend(observations)
    
    def summary(self):
        """간추린 통계량을 찍는다."""
        post_dist = stats.norm(self.posterior_mean, self.posterior_std)
        
        print(f"Posterior: N({self.posterior_mean:.4f}, {self.posterior_std:.4f})")
        print(f"  95% CI: [{post_dist.ppf(0.025):.4f}, {post_dist.ppf(0.975):.4f}]")
```

### 4.5 보기: 측정값

**앞확률:** $\mathcal{N}(100, 10^2)$ — 참값에 대한 앞선 믿음

**아는 데이터 표준편차:** $\sigma = 5$

**데이터:** 측정값 $[102, 98, 105, 101, 99, 103, 97, 104, 100, 102]$

표본 통계량: $n = 10$, $\bar{x} = 101.1$

**뒤확률 셈하기:**

- 앞확률의 정밀도: $\tau_0 = 1/100 = 0.01$
- 데이터의 정밀도: $n\tau = 10/25 = 0.4$
- 전체 정밀도: $\tau_n = 0.41$

$$
\mu_n = \frac{0.01 \times 100 + 0.4 \times 101.1}{0.41} \approx 101.07
$$

$$
\sigma_n = \sqrt{1/0.41} \approx 1.56
$$

| 무게 | 값 |
|--------|-------|
| 앞확률 | $0.01/0.41 = 2.4\%$ |
| 데이터 | $0.40/0.41 = 97.6\%$ |

---

## 5. 켤레 갱신 규칙 간추림

### 5.1 빠른 참고 표

| 모형 | 앞확률의 매개변수 | 갱신 규칙 |
|-------|------------------|-------------|
| **베타-이항** | $\alpha, \beta$ | $\alpha' = \alpha + k$, $\beta' = \beta + (n-k)$ |
| **감마-푸아송** | $\alpha, \beta$ | $\alpha' = \alpha + \sum x_i$, $\beta' = \beta + n$ |
| **정규-정규** | $\mu_0, \sigma_0$ | $\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau}$, $\sigma_n^2 = \frac{1}{\tau_0 + n\tau}$ |

### 5.2 초매개변수의 풀이

| 모형 | 초매개변수 | 풀이 |
|-------|-----------------|----------------|
| **베타** | $\alpha, \beta$ | 가짜 성공 $\alpha - 1$번, 가짜 실패 $\beta - 1$번 |
| **감마** | $\alpha, \beta$ | 가짜 세기 $\alpha$, 가짜 관찰 $\beta$ |
| **정규** | $\mu_0, \tau_0$ | 앞확률의 평균, 앞확률의 정밀도(자신감) |

---

## 6. 핵심 요점

1. **켤레 앞확률**은 앞확률과 같은 족의 해석적 뒤확률 분포를 내어 닫힌 꼴의 베이즈 추론을 가능하게 한다.

2. **베타-이항**은 이진 데이터와 비율 데이터의 일꾼이다. 베타의 초매개변수는 가짜 세기 노릇을 한다.

3. **감마-푸아송**은 세는 데이터를 자연스럽게 다룬다. 뒤확률 예측은 음이항 분포이다.

4. **정규-정규**(흩어짐을 알 때)는 정밀도로 무게 준 평균을 낸다. 뒤확률의 평균은 앞확률의 평균과 표본 평균 사이를 메운다.

5. 켤레 앞확률에서는 **차례 갱신**이 셈으로 효율적이다. 한 묶음의 뒤확률이 다음 묶음의 앞확률이 된다.

---

## 7. 연습문제

### 연습문제 1: 켤레성 확인
베이즈 정리를 짚어 가며 베타가 이항의 켤레임을 수학으로 확인하라. $p(\theta|k, n)$이 베타 분포임을 드러내 보여라.

### 연습문제 2: 앞확률 끌어내기
동전이 공정하다고 믿지만 확신하지는 못한다. 이를 Beta$(\alpha, \beta)$으로 나타내라. 공정함에 대한 "약한 믿음"과 "강한 믿음"은 각각 어떤 값으로 담기는가?

### 연습문제 3: 감마-지수
(기다리는 시간과 수명을 위한) 감마-지수 켤레 쌍을 살펴보라. 이 쌍에 대해 `GammaPoissonModel`과 비슷한 클래스를 구현하라.

### 연습문제 4: 디리클레-다항
디리클레-다항으로 베타-이항을 여러 갈래로 넓혀라. 결과가 세 가지인 주사위 굴리기 문제의 추론을 구현하라.

### 연습문제 5: 켤레가 아닌 앞확률
앞확률이 켤레가 아니면 어떻게 되는가? 해석적 베타-이항을, 켤레가 아닌 앞확률(이를테면 $\log(\theta)$ 위의 고른 분포)로 격자 수치 적분한 결과와 견주어라.

---

## 참고 문헌

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3장
- Hoff, P. *A First Course in Bayesian Statistical Methods*, 3~5장

---

# 덧붙임: 켤레 모형의 자세한 끌어내기

뒤이은 절들은 앞서 들여온 켤레족마다 온전한 끌어내기, 구현, 분석을 준다.

---

# 베르누이-베타 켤레 모형

베르누이-베타 모형은 베이즈 켤레 분석의 가장 단순하고 가르침이 많은 보기이다. 이진 데이터에서 확률을 추론하는, 해석적으로 다룰 수 있는 온전한 틀을 준다. 이 본보기 모형은 더 복잡한 상황으로도 이어지는 베이즈의 근본 개념을 잘 보여 준다.

---

## 문제 설정

### 추론 문제

이진 결과 $x_1, x_2, \ldots, x_n \in \{0, 1\}$(이를테면 동전 던지기, 성공과 실패, 누름과 안 누름)을 관찰하고 그 밑에 깔린 성공 확률 $\theta \in [0, 1]$을 미루어 알고자 한다.

**빈도주의 방법**: 점 어림값 $\hat{\theta} = \bar{x} = k/n$이며 $k = \sum_i x_i$이다.

**베이즈 방법**: $\theta$에 대한 불확실성을 수로 나타내는 온전한 뒤확률 분포 $p(\theta \mid x_1, \ldots, x_n)$.

### 베르누이 가능도

관찰마다 베르누이 분포를 따른다.

$$
x_i \mid \theta \sim \text{Bernoulli}(\theta)
$$

$$
p(x_i \mid \theta) = \theta^{x_i}(1-\theta)^{1-x_i}
$$

성공이 $k$번인 서로 독립인 관찰 $n$개에 대해 다음과 같다.

$$
p(x_1, \ldots, x_n \mid \theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}
$$

**핵심 관찰**: 가능도는 데이터에 대해 오직 $k$과 $n$을 거쳐서만 달라진다. 충분 통계량은 $(k, n)$이다.

---

## 베타 앞확률

### 왜 베타인가?

$[0, 1]$ 위의 앞확률 $p(\theta)$이 필요하다. **베타 분포**가 자연스러운 선택인 까닭은 다음과 같다.

1. **받침**: $[0, 1]$ 위에서 정의되어 매개변수 공간과 맞는다
2. **융통성**: 여러 앞선 믿음(고른 것, U자 모양, 한쪽으로 기운 것)을 나타낼 수 있다
3. **켤레성**: 뒤확률도 베타여서 닫힌 꼴로 고칠 수 있다

### 베타 분포의 정의

$$
\theta \sim \text{Beta}(\alpha, \beta)
$$

$$
p(\theta \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1} = \frac{1}{B(\alpha, \beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

여기서 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$은 베타 함수이다.

### 베타 분포의 성질

**적률**:

$$
\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}
$$

$$
\text{Var}[\theta] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
$$

$$
\text{Mode}[\theta] = \frac{\alpha - 1}{\alpha + \beta - 2} \quad \text{(for } \alpha, \beta > 1\text{)}
$$

**매개변수의 풀이**:

- $\alpha - 1$: 앞확률의 성공 "가짜 세기"
- $\beta - 1$: 앞확률의 실패 "가짜 세기"
- $\alpha + \beta$: "앞확률의 표본 크기" 또는 몰림 정도

### 흔한 베타 앞확률

| 앞확률 | $\alpha$ | $\beta$ | 평균 | 모양 | 쓰임새 |
|-------|----------|---------|------|-------|----------|
| 고른 분포 | 1 | 1 | 0.5 | 평평 | 최대 무지 |
| 제프리스 | 0.5 | 0.5 | 0.5 | U자 | 참조 앞확률 |
| 홀데인 | 0 | 0 | — | 제대로 되지 않음 | 극한의 정보 없는 앞확률 |
| 대칭 | $a$ | $a$ | 0.5 | 뾰족하거나 U자 | 어느 쪽도 편들지 않음 |
| 정보 있는 앞확률 | 10 | 2 | 0.83 | 오른쪽으로 기움 | $\theta$이 높다는 앞선 믿음 |

### 베타 앞확률 그려 보기

```
α=1, β=1 (Uniform)        α=0.5, β=0.5 (Jeffreys)    α=2, β=5 (Informative)
    ___________               ∪                           /\
   |           |             / \                         /  \____
   |___________|            /   \                       /        \
   0           1           0     1                     0          1
```

---

## 켤레 뒤확률 끌어내기

### 켤레성

**정의**: 뒤확률 $p(\theta \mid \mathcal{D})$이 앞확률과 같은 분포족에 들면 앞확률 $p(\theta)$은 가능도 $p(\mathcal{D} \mid \theta)$의 **켤레**이다.

### 유도

**앞확률**: $\theta \sim \text{Beta}(\alpha, \beta)$

$$
p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

**가능도**: 시행 $n$번 가운데 성공 $k$번

$$
p(\mathcal{D} \mid \theta) = \theta^k(1-\theta)^{n-k}
$$

**뒤확률**(베이즈 정리로):

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \cdot p(\theta)
$$

$$
p(\theta \mid \mathcal{D}) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

$$
p(\theta \mid \mathcal{D}) \propto \theta^{k + \alpha - 1}(1-\theta)^{n-k+\beta-1}
$$

이것은 베타 분포의 알맹이이다! 그러므로 다음과 같다.

$$
\boxed{\theta \mid \mathcal{D} \sim \text{Beta}(\alpha + k, \beta + n - k)}
$$

### 갱신 규칙

| 양 | 앞확률 | 뒤확률 |
|----------|-------|-----------|
| 분포 | $\text{Beta}(\alpha, \beta)$ | $\text{Beta}(\alpha + k, \beta + n - k)$ |
| "성공" | $\alpha - 1$ | $\alpha + k - 1$ |
| "실패" | $\beta - 1$ | $\beta + n - k - 1$ |
| "표본 크기" | $\alpha + \beta$ | $\alpha + \beta + n$ |

**직관**: 관찰한 성공을 $\alpha$에, 관찰한 실패를 $\beta$에 더하기만 하면 된다. 앞확률은 앞선 실험에서 온 "가짜 데이터" 노릇을 한다.

---

## 뒤확률 분석

### 뒤확률의 평균

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha + k}{\alpha + \beta + n}
$$

이는 **무게 준 평균**으로 다시 쓸 수 있다.

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{k}{n}}_{\text{MLE}}
$$

데이터의 무게를 $w = \frac{n}{\alpha + \beta + n}$이라 하자. 그러면 다음과 같다.

$$
\mathbb{E}[\theta \mid \mathcal{D}] = (1-w) \cdot \text{prior mean} + w \cdot \text{MLE}
$$

**핵심 통찰**: 뒤확률의 평균은 앞확률의 평균과 최대 가능도 어림값 사이를 메우며, 그 무게는 서로의 "표본 크기"가 정한다.

### 뒤확률의 최빈값(최대 뒤확률 어림값)

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2} \quad \text{(for } \alpha + k > 1, \beta + n - k > 1\text{)}
$$

### 뒤확률의 흩어짐

$$
\text{Var}[\theta \mid \mathcal{D}] = \frac{(\alpha + k)(\beta + n - k)}{(\alpha + \beta + n)^2(\alpha + \beta + n + 1)}
$$

$n \to \infty$이면 다음과 같다.

$$
\text{Var}[\theta \mid \mathcal{D}] \approx \frac{\hat{\theta}(1-\hat{\theta})}{n} \to 0
$$

뒤확률이 참값 둘레로 몰린다.

### 믿음 구간

$(1-\alpha)$ **양 꼬리가 같은 믿음 구간**은 다음과 같다.

$$
\left[F^{-1}_{\text{Beta}}\left(\frac{\alpha}{2}\right), F^{-1}_{\text{Beta}}\left(1 - \frac{\alpha}{2}\right)\right]
$$

여기서 $F^{-1}_{\text{Beta}}$은 $\text{Beta}(\alpha + k, \beta + n - k)$의 분위수 함수이다.

---

## 차례 갱신

### 온라인 학습

켤레 모형의 힘 있는 성질 하나는 차례 갱신이 효율적이라는 것이다. 관찰이 흐르듯 들어오면 뒤확률을 조금씩 고쳐 간다.

$$
\text{Beta}(\alpha_0, \beta_0) \xrightarrow{x_1} \text{Beta}(\alpha_1, \beta_1) \xrightarrow{x_2} \text{Beta}(\alpha_2, \beta_2) \xrightarrow{x_3} \cdots
$$

여기서 각 기호는 다음과 같다.

$$
\alpha_{t+1} = \alpha_t + x_{t+1}, \quad \beta_{t+1} = \beta_t + (1 - x_{t+1})
$$

### 차례와 무관함

마지막 뒤확률은 **관찰의 차례와 무관하다**.

$$
p(\theta \mid x_1, x_2, \ldots, x_n) = p(\theta \mid x_{\pi(1)}, x_{\pi(2)}, \ldots, x_{\pi(n)})
$$

어떤 순열 $\pi$에 대해서도 그렇다. 이는 베르누이 가능도의 맞바꿈 가능성에서 따라 나온다.

---

## 뒤확률 예측 분포

### 다음 관찰 맞히기

관찰한 데이터가 주어졌을 때 다음 관찰이 성공일 확률은 얼마인가?

$$
p(x_{n+1} = 1 \mid \mathcal{D}) = \int_0^1 p(x_{n+1} = 1 \mid \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

$$
= \int_0^1 \theta \cdot p(\theta \mid \mathcal{D}) \, d\theta = \mathbb{E}[\theta \mid \mathcal{D}]
$$

$$
\boxed{p(x_{n+1} = 1 \mid \mathcal{D}) = \frac{\alpha + k}{\alpha + \beta + n}}
$$

이것이 **라플라스의 계승 규칙**이다. 예측 확률은 뒤확률의 평균과 같다.

### 앞으로의 관찰 여럿 맞히기

앞으로의 시행 $m$번에서 성공 횟수 $k'$은 **베타-이항** 분포를 따른다.

$$
p(k' \mid m, \mathcal{D}) = \binom{m}{k'} \frac{B(\alpha + k + k', \beta + n - k + m - k')}{B(\alpha + k, \beta + n - k)}
$$

이는 표집의 흔들림과 매개변수의 불확실성을 모두 셈에 넣는다.

---

## 특별한 경우와 이음

### 고른 앞확률(alpha = beta = 1)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(1 + k, 1 + n - k)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k + 1}{n + 2}
$$

이것이 **라플라스의 규칙**이다. 관찰한 세기에 성공 하나와 실패 하나를 더한다.

### 제프리스 앞확률(alpha = beta = 1/2)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(k + 1/2, n - k + 1/2)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k + 1/2}{n + 1}
$$

제프리스 앞확률은 $\theta \leftrightarrow 1 - \theta$의 다시 매개변수화와 단조 변환에 흔들리지 않는 유일한 앞확률이다.

### 홀데인 앞확률(alpha = beta = 0)

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(k, n - k)
$$

$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{k}{n} = \text{MLE}
$$

**주의**: 홀데인 앞확률은 제대로 된 확률이 아니며, $k = 0$이거나 $k = n$이면 뒤확률도 제대로 되지 않는다.

### 최대 가능도와의 이음

앞확률의 세기가 $\to 0$이거나 $n \to \infty$이면 다음과 같다.

$$
\hat{\theta}_{\text{Bayes}} \to \hat{\theta}_{\text{MLE}} = \frac{k}{n}
$$

베이즈 어림값과 빈도주의 어림값은 점근적으로 하나로 모인다.

---

## 실용적인 고려

### 앞확률의 매개변수 고르기

**방법 1: 앞확률의 평균과 "맞먹는 표본 크기"**

앞확률의 평균이 $\mu_0$이고 앞확률이 관찰 $n_0$개에 맞먹는다고 믿으면 다음과 같다.

$$
\alpha = \mu_0 \cdot n_0, \quad \beta = (1 - \mu_0) \cdot n_0
$$

**방법 2: 앞확률의 평균과 흩어짐**

앞확률의 평균 $\mu$과 흩어짐 $\sigma^2$이 주어지면 다음과 같다.

$$
\alpha = \mu \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right), \quad \beta = (1-\mu) \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right)
$$

**방법 3: 앞확률의 믿음 구간**

$\theta \in [a, b]$일 확률이 95%라고 믿으면 다음을 만족하는 $(\alpha, \beta)$을 수치로 풀어라.

$$
F_{\text{Beta}}(b; \alpha, \beta) - F_{\text{Beta}}(a; \alpha, \beta) = 0.95
$$

### 앞확률 민감도 분석

앞확률을 달리했을 때 결론이 어떻게 바뀌는지 늘 살펴라.

| 앞확률 | 매개변수 | 뒤확률의 평균(k=7, n=10) |
|-------|------------|---------------------------|
| 고른 분포 | (1, 1) | 0.667 |
| 제프리스 | (0.5, 0.5) | 0.682 |
| 미더워하지 않음 | (1, 9) | 0.400 |
| 낙관적 | (9, 1) | 0.800 |

그럴듯한 앞확률들에 걸쳐 결론이 흔들리지 않으면 더 미덥다.

---

## 파이썬 구현

```python
"""
베르누이-베타 켤레 모형: 온전한 구현

이 모듈은 베타-베르누이 켤레 짝을 써서 이진 자료에 대한 베이즈 추론을
두루 갖춰 구현한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import beta as beta_func
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class BetaPosterior:
    """
    베타 뒤확률 분포를 나타낸다.
    
    속성
    ----------
    alpha : float
        첫째 모양 매개변수(가짜 성공 + 1)
    beta : float
        둘째 모양 매개변수(가짜 실패 + 1)
    n_successes : int
        관측된 성공 횟수
    n_trials : int
        관측된 시도 횟수
    """
    alpha: float
    beta: float
    n_successes: int = 0
    n_trials: int = 0
    
    @property
    def mean(self) -> float:
        """뒤확률 평균 E[θ|D]."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def mode(self) -> Optional[float]:
        """뒤확률 최빈값(MAP 어림값)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif self.alpha <= 1 and self.beta > 1:
            return 0.0
        elif self.alpha > 1 and self.beta <= 1:
            return 1.0
        else:
            return None  # 쌍봉이거나 정해지지 않음
    
    @property
    def variance(self) -> float:
        """뒤확률 흩어짐 Var[θ|D]."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))
    
    @property
    def std(self) -> float:
        """뒤확률 표준편차."""
        return np.sqrt(self.variance)
    
    def pdf(self, theta: np.ndarray) -> np.ndarray:
        """뒤확률 밀도의 값을 매긴다."""
        return stats.beta.pdf(theta, self.alpha, self.beta)
    
    def cdf(self, theta: float) -> float:
        """뒤확률 누적분포함수의 값을 매긴다."""
        return stats.beta.cdf(theta, self.alpha, self.beta)
    
    def quantile(self, p: float) -> float:
        """뒤확률 분위수를 셈한다."""
        return stats.beta.ppf(p, self.alpha, self.beta)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        양끝이 같은 믿음 구간을 셈한다.
        
        매개변수
        ----------
        level : float
            믿음 수준(95% 구간이면 기본값 0.95)
        
        반환값
        -------
        tuple
            (아래, 위) 경계
        """
        alpha_level = 1 - level
        lower = self.quantile(alpha_level / 2)
        upper = self.quantile(1 - alpha_level / 2)
        return (lower, upper)
    
    def hpd_interval(self, level: float = 0.95, n_points: int = 1000) -> Tuple[float, float]:
        """
        최고 뒤확률 밀도 구간을 셈한다.
        
        주어진 확률 질량을 담는 가장 짧은 구간.
        """
        # HPD을 찾는 격자 뒤지기
        theta_grid = np.linspace(0.001, 0.999, n_points)
        pdf_vals = self.pdf(theta_grid)
        
        # 밀도로 정렬(내림차순)
        sorted_idx = np.argsort(pdf_vals)[::-1]
        sorted_theta = theta_grid[sorted_idx]
        sorted_pdf = pdf_vals[sorted_idx]
        
        # 확률 질량 쌓기
        cumsum = np.cumsum(sorted_pdf) * (theta_grid[1] - theta_grid[0])
        cutoff_idx = np.searchsorted(cumsum, level)
        
        # HPD 구역 경계
        hpd_theta = sorted_theta[:cutoff_idx + 1]
        return (hpd_theta.min(), hpd_theta.max())
    
    def sample(self, n_samples: int) -> np.ndarray:
        """뒤확률에서 표본을 뽑는다."""
        return stats.beta.rvs(self.alpha, self.beta, size=n_samples)
    
    def predictive_prob(self) -> float:
        """다음 관측이 성공일 확률(라플라스의 규칙)."""
        return self.mean
    
    def __repr__(self) -> str:
        return f"Beta({self.alpha:.2f}, {self.beta:.2f})"

class BetaBernoulliModel:
    """
    온전한 베타-베르누이 켤레 모형.
    
    매개변수
    ----------
    prior_alpha : float
        앞확률 α 매개변수
    prior_beta : float
        앞확률 β 매개변수
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._reset()
    
    def _reset(self):
        """앞확률 상태로 되돌린다."""
        self.current_alpha = self.prior_alpha
        self.current_beta = self.prior_beta
        self.n_successes = 0
        self.n_trials = 0
    
    @property
    def prior(self) -> BetaPosterior:
        """앞확률 분포를 되돌린다."""
        return BetaPosterior(self.prior_alpha, self.prior_beta)
    
    @property
    def posterior(self) -> BetaPosterior:
        """지금의 뒤확률 분포를 되돌린다."""
        return BetaPosterior(
            self.current_alpha, 
            self.current_beta,
            self.n_successes,
            self.n_trials
        )
    
    def update(self, successes: int, trials: int) -> BetaPosterior:
        """
        새 관측으로 뒤확률을 새로 고친다.
        
        매개변수
        ----------
        successes : int
            관측된 성공 횟수
        trials : int
            관측된 시도 횟수
        
        반환값
        -------
        BetaPosterior
            새로 고친 뒤확률 분포
        """
        self.current_alpha += successes
        self.current_beta += (trials - successes)
        self.n_successes += successes
        self.n_trials += trials
        return self.posterior
    
    def update_single(self, outcome: int) -> BetaPosterior:
        """
        관측 하나로 새로 고친다.
        
        매개변수
        ----------
        outcome : int
            0 또는 1
        
        반환값
        -------
        BetaPosterior
            새로 고친 뒤확률
        """
        return self.update(outcome, 1)
    
    def update_sequence(self, outcomes: List[int]) -> List[BetaPosterior]:
        """
        차례대로 새로 고치며 뒤확률의 자취를 되돌린다.
        
        매개변수
        ----------
        outcomes : list
            0/1 관측의 늘어놓음
        
        반환값
        -------
        list
            새로 고칠 때마다의 뒤확률 분포 목록
        """
        history = [self.posterior]
        for outcome in outcomes:
            self.update_single(outcome)
            history.append(self.posterior)
        return history
    
    def log_marginal_likelihood(self) -> float:
        """
        로그 주변 가능도(로그 증거)를 셈한다.
        
        log p(D) = log B(α + k, β + n - k) - log B(α, β)
        
        반환값
        -------
        float
            로그 주변 가능도
        """
        from scipy.special import betaln
        
        prior_term = betaln(self.prior_alpha, self.prior_beta)
        posterior_term = betaln(self.current_alpha, self.current_beta)
        
        return posterior_term - prior_term
    
    def predictive_distribution(self, m: int) -> np.ndarray:
        """
        앞으로의 시도 m번에 대한 베타-이항 예측 분포를 셈한다.
        
        매개변수
        ----------
        m : int
            앞으로의 시도 횟수
        
        반환값
        -------
        array
            성공 k' = 0, 1, ..., m의 확률
        """
        from scipy.special import comb, betaln
        
        a, b = self.current_alpha, self.current_beta
        k_vals = np.arange(m + 1)
        
        log_probs = (
            np.log(comb(m, k_vals, exact=False)) +
            betaln(a + k_vals, b + m - k_vals) -
            betaln(a, b)
        )
        
        return np.exp(log_probs)

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_beta_distribution(
    alpha: float, 
    beta: float, 
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: str = 'steelblue',
    fill: bool = True
) -> plt.Axes:
    """베타 분포를 그린다."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    theta = np.linspace(0.001, 0.999, 500)
    pdf = stats.beta.pdf(theta, alpha, beta)
    
    if fill:
        ax.fill_between(theta, pdf, alpha=0.3, color=color)
    ax.plot(theta, pdf, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_prior_posterior(
    model: BetaBernoulliModel,
    true_theta: Optional[float] = None,
    title: str = "Bayesian Update"
) -> plt.Figure:
    """앞확률, 가능도, 뒤확률을 그려 본다."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    theta = np.linspace(0.001, 0.999, 500)
    
    # 앞확률
    prior_pdf = stats.beta.pdf(theta, model.prior_alpha, model.prior_beta)
    ax.plot(theta, prior_pdf / prior_pdf.max(), 'b--', 
            linewidth=2, label=f'Prior: Beta({model.prior_alpha}, {model.prior_beta})')
    
    # 가능도(그려 보려고 고르게 함)
    if model.n_trials > 0:
        k, n = model.n_successes, model.n_trials
        likelihood = theta**k * (1 - theta)**(n - k)
        ax.plot(theta, likelihood / likelihood.max(), 'g:', 
                linewidth=2, label=f'Likelihood ({k}/{n} successes)')
    
    # 뒤확률
    post = model.posterior
    posterior_pdf = post.pdf(theta)
    ax.fill_between(theta, posterior_pdf / posterior_pdf.max(), 
                    alpha=0.3, color='red')
    ax.plot(theta, posterior_pdf / posterior_pdf.max(), 'r-', 
            linewidth=2, label=f'Posterior: {post}')
    
    # 참값
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle='--', 
                   linewidth=2, label=f'True θ = {true_theta}')
    
    # 뒤확률 평균
    ax.axvline(post.mean, color='red', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_sequential_update(
    outcomes: List[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    true_theta: Optional[float] = None
) -> plt.Figure:
    """차례대로 베이즈 새로 고치기를 그려 본다."""
    
    model = BetaBernoulliModel(prior_alpha, prior_beta)
    history = model.update_sequence(outcomes)
    
    n_steps = len(history)
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    theta = np.linspace(0.001, 0.999, 200)
    
    for i, (ax, post) in enumerate(zip(axes[:n_steps], history)):
        pdf = post.pdf(theta)
        ax.fill_between(theta, pdf, alpha=0.4, color='steelblue')
        ax.plot(theta, pdf, 'b-', linewidth=2)
        
        if true_theta is not None:
            ax.axvline(true_theta, color='red', linestyle='--', linewidth=1.5)
        
        ax.axvline(post.mean, color='green', linestyle=':', linewidth=1.5)
        
        if i == 0:
            ax.set_title(f'Prior\nE[θ]={post.mean:.3f}')
        else:
            cumsum = sum(outcomes[:i])
            ax.set_title(f'After {i} obs ({cumsum}/{i})\nE[θ]={post.mean:.3f}')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('θ')
    
    # 안 쓰는 축 감추기
    for ax in axes[n_steps:]:
        ax.set_visible(False)
    
    plt.suptitle('Sequential Bayesian Updating', fontsize=14)
    plt.tight_layout()
    
    return fig

# =============================================================================
# 보여 주기
# =============================================================================

def demo_basic_inference():
    """기본 베타-베르누이 추론을 보인다."""
    
    print("=" * 60)
    print("BASIC BETA-BERNOULLI INFERENCE")
    print("=" * 60)
    
    # 준비
    true_theta = 0.7
    n_trials = 20
    np.random.seed(42)
    data = np.random.binomial(1, true_theta, n_trials)
    k = data.sum()
    
    print(f"\nTrue θ: {true_theta}")
    print(f"Data: {k} successes in {n_trials} trials")
    print(f"MLE: {k/n_trials:.4f}")
    
    # 서로 다른 앞확률
    priors = [
        ("Uniform", 1, 1),
        ("Jeffreys", 0.5, 0.5),
        ("Informative (pessimistic)", 2, 8),
        ("Informative (optimistic)", 8, 2),
    ]
    
    print("\nPosterior summaries under different priors:")
    print("-" * 60)
    
    for name, alpha, beta in priors:
        model = BetaBernoulliModel(alpha, beta)
        model.update(k, n_trials)
        post = model.posterior
        ci = post.credible_interval(0.95)
        
        print(f"\n{name} prior: Beta({alpha}, {beta})")
        print(f"  Posterior: Beta({post.alpha:.1f}, {post.beta:.1f})")
        print(f"  Mean: {post.mean:.4f}")
        print(f"  Mode: {post.mode:.4f}" if post.mode else "  Mode: undefined")
        print(f"  Std:  {post.std:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  P(next success): {post.predictive_prob():.4f}")

def demo_sequential_learning():
    """차례대로 새로 고치기를 보인다."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL BAYESIAN LEARNING")
    print("=" * 60)
    
    true_theta = 0.6
    np.random.seed(123)
    outcomes = list(np.random.binomial(1, true_theta, 15))
    
    print(f"\nTrue θ: {true_theta}")
    print(f"Outcomes: {outcomes}")
    
    model = BetaBernoulliModel(1, 1)  # 고른 앞확률
    
    print("\nEvolution of posterior mean:")
    print("-" * 40)
    
    for i, outcome in enumerate(outcomes):
        model.update_single(outcome)
        post = model.posterior
        cumsum = sum(outcomes[:i+1])
        print(f"After obs {i+1:2d} (x={outcome}): "
              f"E[θ|D] = {post.mean:.4f}, "
              f"σ = {post.std:.4f}, "
              f"Data: {cumsum}/{i+1}")
    
    # 시각화 만들기
    fig = plot_sequential_update(outcomes, true_theta=true_theta)
    fig.savefig('sequential_beta_update.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: sequential_beta_update.png")

def demo_predictive_distribution():
    """뒤확률 예측 분포를 보인다."""
    
    print("\n" + "=" * 60)
    print("POSTERIOR PREDICTIVE DISTRIBUTION")
    print("=" * 60)
    
    # 관측 자료
    k, n = 7, 10
    
    model = BetaBernoulliModel(1, 1)
    model.update(k, n)
    
    print(f"\nObserved: {k} successes in {n} trials")
    print(f"Posterior: Beta({model.current_alpha}, {model.current_beta})")
    
    # 다음 시도 m번 미리 알기
    m = 10
    predictive = model.predictive_distribution(m)
    
    print(f"\nPredictive distribution for next {m} trials:")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    k_vals = np.arange(m + 1)
    ax.bar(k_vals, predictive, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 기댓값
    expected = np.sum(k_vals * predictive)
    ax.axvline(expected, color='red', linestyle='--', linewidth=2,
               label=f'E[k\'] = {expected:.2f}')
    
    ax.set_xlabel('Number of successes in next 10 trials', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Beta-Binomial Posterior Predictive Distribution', fontsize=14)
    ax.legend()
    ax.set_xticks(k_vals)
    
    plt.tight_layout()
    plt.savefig('predictive_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Expected successes in next {m}: {expected:.2f}")
    print(f"Most likely outcome: {k_vals[np.argmax(predictive)]} successes")
    print("\nSee: predictive_distribution.png")

if __name__ == "__main__":
    demo_basic_inference()
    demo_sequential_learning()
    demo_predictive_distribution()
```

---

## 요약

| 갈래 | 식 |
|--------|---------|
| **앞확률** | $\theta \sim \text{Beta}(\alpha, \beta)$ |
| **가능도** | $p(\mathcal{D} \mid \theta) = \theta^k(1-\theta)^{n-k}$ |
| **뒤확률** | $\theta \mid \mathcal{D} \sim \text{Beta}(\alpha + k, \beta + n - k)$ |
| **뒤확률의 평균** | $\frac{\alpha + k}{\alpha + \beta + n}$ |
| **뒤확률의 최빈값** | $\frac{\alpha + k - 1}{\alpha + \beta + n - 2}$ |
| **예측** | $p(x_{n+1}=1 \mid \mathcal{D}) = \frac{\alpha + k}{\alpha + \beta + n}$ |

### 핵심 통찰

1. **켤레성**: 베타 앞확률 + 베르누이 가능도 → 베타 뒤확률
2. **가짜 세기**: 앞확률의 매개변수가 관찰을 더한 것처럼 굴러간다
3. **무게 준 평균**: 뒤확률의 평균이 앞확률의 평균과 최대 가능도 어림값 사이를 메운다
4. **차례 갱신**: 성공은 $\alpha$에, 실패는 $\beta$에 더한다
5. **라플라스의 규칙**: 예측 확률은 뒤확률의 평균과 같다
6. **점근적 최대 가능도**: $n \to \infty$이면 베이즈 어림값이 빈도주의 최대 가능도 어림값으로 간다

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 앞확률-가능도-뒤확률 | 13장: 바탕 | 일반 틀 |
| 가우스 켤레 | 13장: 가우스 모형 | 이어진 값에서의 대응물 |
| 켤레 앞확률 | 13장: 켤레 앞확률 | 일반 이론 |
| 모형 견줌 | 13장: 베이즈 인자 | 증거 셈하기 |
| BNN 분류 | 13장: BNN | 여러 층으로 넓히기 |

### 주요 참고 문헌

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). 2장.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. 3장.
- Hoff, P. D. (2009). *A First Course in Bayesian Statistical Methods*. Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. 2.1절.

---

# 흩어짐을 알 때의 가우스 추론

흩어짐을 아는 가우스 분포의 평균에 대한 베이즈 추론은 이어진 값에 대한 켤레 모형의 바탕이다. 베이즈 갱신이 **정밀도로 무게 준 평균 내기**를 거쳐 앞선 정보와 데이터를 어떻게 어우르는지 가장 또렷이 보여 준다. 이 우아한 결과는 다변량 상황으로도 이어지며 더 복잡한 베이즈 모형을 이해하는 바탕이 된다.

---

## 문제 설정

### 추론 문제

**평균 $\mu$은 모르고 흩어짐 $\sigma^2$은 아는** 가우스 분포에서 나왔다고 놓는 이어진 측정값 $x_1, x_2, \ldots, x_n \in \mathbb{R}$을 관찰한다.

$$
x_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)
$$

목표는 뒤확률 분포 $p(\mu \mid x_1, \ldots, x_n)$을 미루어 아는 것이다.

**흩어짐을 아는 때는 언제인가?**

- 정밀도가 눈금 맞춰진 측정 장치
- 오랜 기간에 걸친 변동성의 지난 어림값
- 이론적 제약(이를테면 양자 잡음 한계)
- 가르치기 위해 단순하게 놓는 가정

### 가우스 가능도

관찰 하나에 대해 다음과 같다.

$$
p(x_i \mid \mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

서로 독립인 관찰 $n$개에 대해 다음과 같다.

$$
p(x_1, \ldots, x_n \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

$$
= (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2\right)
$$

### 충분 통계량

가능도는 충분 통계량 $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$으로 다시 쓸 수 있다.

$$
\sum_{i=1}^{n}(x_i - \mu)^2 = \sum_{i=1}^{n}(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2
$$

첫 항은 $\mu$에 대해 상수이므로 다음과 같다.

$$
p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n(\bar{x} - \mu)^2}{2\sigma^2}\right)
$$

**핵심 통찰**: 가능도는 데이터에 대해 오직 $(\bar{x}, n)$을 거쳐서만 달라진다. 표본 평균 $\bar{x}$이 $\mu$에 충분하다.

---

## 정밀도: 자연스러운 매개변수화

### 정의

**정밀도**는 흩어짐의 역수이다.

$$
\tau = \frac{1}{\sigma^2}
$$

정밀도는 **정보의 알맹이**를 잰다. 정밀도가 높을수록 알맹이가 많고 덜 아리송한 측정이다.

### 왜 정밀도인가?

정밀도는 가우스 정보를 어우르는 데 자연스러운 매개변수이다.

- **정밀도는 더해진다**: 서로 독립인 정보를 어우를 때 정밀도가 합쳐진다
- **흩어짐은 그렇게 단순히 더해지지 않는다**: $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$은 $X, Y$이 독립일 때만 성립한다
- **선형 갱신**: 베이즈 갱신이 정밀도에 대해 선형이 된다

| 양 | 흩어짐 꼴 | 정밀도 꼴 |
|----------|---------------|----------------|
| 관찰 하나 | $\sigma^2$ | $\tau = 1/\sigma^2$ |
| 관찰 $n$개의 표본 평균 | $\sigma^2/n$ | $n\tau$ |
| 앞확률 | $\sigma_0^2$ | $\tau_0 = 1/\sigma_0^2$ |
| 뒤확률 | $\sigma_n^2$ | $\tau_n = \tau_0 + n\tau$ |

---

## 가우스 앞확률

### 켤레 앞확률

켤레가 되도록 $\mu$에 가우스 앞확률을 쓴다.

$$
\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)
$$

또는 같은 말로 정밀도 $\tau_0 = 1/\sigma_0^2$을 쓰면 다음과 같다.

$$
p(\mu) = \sqrt{\frac{\tau_0}{2\pi}} \exp\left(-\frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

### 앞확률 매개변수의 풀이

| 매개변수 | 기호 | 풀이 |
|-----------|--------|----------------|
| 앞확률의 평균 | $\mu_0$ | 데이터를 보기 전의 가장 나은 어림 |
| 앞확률의 흩어짐 | $\sigma_0^2$ | 앞선 믿음의 아리송함 |
| 앞확률의 정밀도 | $\tau_0$ | 앞선 믿음에 대한 자신감 |
| 앞확률의 "실효 표본 크기" | $n_0 = \tau_0/\tau = \sigma^2/\sigma_0^2$ | 앞확률이 관찰 $n_0$개에 맞먹는다 |

### 흔한 앞확률 선택

**정보 있는 앞확률**: $\mu_0$과 $\sigma_0^2$이 참된 앞선 앎을 비춘다

$$
\mu \sim \mathcal{N}(100, 5^2) \quad \text{("Mean is around 100, ± 10")}
$$

**약하게 정보 있는 앞확률**: 넓지만 제대로 된 확률

$$
\mu \sim \mathcal{N}(0, 100^2) \quad \text{("Probably not astronomically large")}
$$

**제대로 되지 않은 평평한 앞확률**: 극한의 경우 $\sigma_0^2 \to \infty$

$$
p(\mu) \propto 1 \quad \text{(improper, but yields proper posterior)}
$$

---

## 켤레 뒤확률 끌어내기

### 끌어내기

**앞확률**:

$$
p(\mu) \propto \exp\left(-\frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

**가능도**:

$$
p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n\tau}{2}(\mu - \bar{x})^2\right)
$$

여기서 $\tau = 1/\sigma^2$은 아는 데이터의 정밀도이다.

**뒤확률**(베이즈 정리로):

$$
p(\mu \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mu) \cdot p(\mu)
$$

$$
\propto \exp\left(-\frac{n\tau}{2}(\mu - \bar{x})^2 - \frac{\tau_0}{2}(\mu - \mu_0)^2\right)
$$

### 완전제곱 만들기

지수를 펼치면 다음과 같다.

$$
-\frac{1}{2}\left[n\tau(\mu^2 - 2\mu\bar{x} + \bar{x}^2) + \tau_0(\mu^2 - 2\mu\mu_0 + \mu_0^2)\right]
$$

$$
= -\frac{1}{2}\left[(n\tau + \tau_0)\mu^2 - 2\mu(n\tau\bar{x} + \tau_0\mu_0) + \text{const}\right]
$$

$$
= -\frac{n\tau + \tau_0}{2}\left[\mu^2 - 2\mu\frac{n\tau\bar{x} + \tau_0\mu_0}{n\tau + \tau_0}\right] + \text{const}
$$

$$
= -\frac{\tau_n}{2}\left(\mu - \mu_n\right)^2 + \text{const}
$$

이것은 가우스 분포의 알맹이이다! 그러므로 다음과 같다.

$$
\boxed{\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)}
$$

### 갱신 식

**뒤확률의 정밀도**(정밀도는 더해진다):

$$
\boxed{\tau_n = \tau_0 + n\tau}
$$

$$
\sigma_n^2 = \frac{1}{\tau_n} = \frac{1}{\tau_0 + n\tau} = \frac{\sigma^2\sigma_0^2}{n\sigma_0^2 + \sigma^2}
$$

**뒤확률의 평균**(정밀도로 무게 준 평균):

$$
\boxed{\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau} = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_n}}
$$

또는 흩어짐 꼴로 쓰면 다음과 같다.

$$
\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}
$$

---

## 정밀도로 무게 준 평균 내기

### 근본 통찰

뒤확률의 평균은 앞확률의 평균과 데이터 평균의 **무게 준 평균**이다.

$$
\mu_n = w_0 \cdot \mu_0 + w_{\text{data}} \cdot \bar{x}
$$

여기서 무게는 **정밀도에 비례한다**.

$$
w_0 = \frac{\tau_0}{\tau_0 + n\tau}, \quad w_{\text{data}} = \frac{n\tau}{\tau_0 + n\tau}
$$

$w_0 + w_{\text{data}} = 1$임에 유의하라.

### 해석

- **앞확률이 더 정밀하면**($\tau_0$이 크면) → 앞확률의 평균에 무게가 더 실린다
- **데이터가 많으면**($n$이 크면) → 표본 평균에 무게가 더 실린다
- **측정이 더 정밀하면**($\tau$이 크면) → 관찰마다의 몫이 커진다

### 맞먹는 표본 크기

앞확률의 **맞먹는 표본 크기**를 정의한다.

$$
n_0 = \frac{\tau_0}{\tau} = \frac{\sigma^2}{\sigma_0^2}
$$

그러면 다음과 같다.

$$
\mu_n = \frac{n_0 \cdot \mu_0 + n \cdot \bar{x}}{n_0 + n}
$$

앞확률은 데이터 정밀도에서 관찰 $n_0$개의 값어치를 지닌다.

---

## 뒤확률 분석

### 점 어림값

가우스 뒤확률에서는 흔한 점 어림값 셋이 모두 일치한다.

$$
\mathbb{E}[\mu \mid \mathcal{D}] = \text{Mode}[\mu \mid \mathcal{D}] = \text{Median}[\mu \mid \mathcal{D}] = \mu_n
$$

이는 대칭이고 봉우리가 하나인 분포만의 성질이다.

### 뒤확률의 흩어짐과 표준 오차

$$
\text{Var}[\mu \mid \mathcal{D}] = \sigma_n^2 = \frac{1}{\tau_0 + n\tau}
$$

뒤확률의 표준편차(베이즈판 "표준 오차")는 다음과 같다.

$$
\sigma_n = \frac{1}{\sqrt{\tau_0 + n\tau}}
$$

### 믿음 구간

가우스 뒤확률에서 $(1-\alpha)$ 믿음 구간은 다음과 같다.

$$
\mu_n \pm z_{\alpha/2} \cdot \sigma_n
$$

여기서 $z_{\alpha/2}$은 표준 정규 분위수이다.

**95% 믿음 구간**:

$$
\left[\mu_n - 1.96\sigma_n, \mu_n + 1.96\sigma_n\right]
$$

가우스 뒤확률에서는 대칭 덕분에 양 꼬리가 같은 구간과 최고 뒤확률 밀도 구간이 일치한다.

### 오그라들기

뒤확률의 평균은 최대 가능도 어림값을 앞확률의 평균 쪽으로 "오그라뜨린다".

$$
\mu_n - \mu_0 = \frac{n\tau}{\tau_0 + n\tau}(\bar{x} - \mu_0)
$$

오그라듦 인자 $\frac{n\tau}{\tau_0 + n\tau} < 1$이 어림값을 앞확률 쪽으로 끌어당긴다.

---

## 점근 거동

### 큰 표본의 극한

$n \to \infty$이면 다음과 같다.

**뒤확률의 평균**:

$$
\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_0 + n\tau} \to \bar{x} \to \mu_{\text{true}}
$$

**뒤확률의 흩어짐**:

$$
\sigma_n^2 = \frac{1}{\tau_0 + n\tau} \sim \frac{1}{n\tau} = \frac{\sigma^2}{n} \to 0
$$

**모이는 속도**:

$$
\sigma_n = O(n^{-1/2})
$$

뒤확률은 여느 때와 같은 $\sqrt{n}$ 속도로 참 평균 둘레에 몰린다.

### 앞확률이 씻겨 나감

앞확률의 영향이 사라진다.

$$
\text{Prior weight} = \frac{\tau_0}{\tau_0 + n\tau} \to 0 \quad \text{as } n \to \infty
$$

데이터가 넉넉하면 그럴듯한 앞확률들은 똑같은 뒤확률을 낸다.

### 베른슈타인-폰 미제스

이 모형은 베른슈타인-폰 미제스 정리를 꼭 만족한다. 뒤확률은 최대 가능도 어림값을 중심으로 점근적으로 정규 분포이며 흩어짐은 피셔 정보의 역수와 같다.

$$
p(\mu \mid \mathcal{D}_n) \xrightarrow{d} \mathcal{N}\left(\bar{x}, \frac{\sigma^2}{n}\right)
$$

---

## 차례 갱신

### 온라인 베이즈 학습

켤레 앞확률에서는 데이터를 모두 담아 두지 않고도 차례로 고칠 수 있다.

$$
\mathcal{N}(\mu_0, \sigma_0^2) \xrightarrow{x_1} \mathcal{N}(\mu_1, \sigma_1^2) \xrightarrow{x_2} \mathcal{N}(\mu_2, \sigma_2^2) \xrightarrow{x_3} \cdots
$$

**갱신 식**(관찰 $x$ 하나):

$$
\tau_{t+1} = \tau_t + \tau
$$

$$
\mu_{t+1} = \frac{\tau_t \mu_t + \tau x}{\tau_{t+1}}
$$

### 되돌이 꼴

같은 말로 다음과 같다.

$$
\mu_{t+1} = \mu_t + \frac{\tau}{\tau_{t+1}}(x - \mu_t) = \mu_t + K_t(x - \mu_t)
$$

여기서 $K_t = \frac{\tau}{\tau_t + \tau} = \frac{\sigma_t^2}{\sigma_t^2 + \sigma^2}$은 **칼만 이득**이다.

이는 가장 단순한 칼만 거르개이다. 곧 관찰 잡음을 아는 스칼라 상태이다.

---

## 뒤확률 예측 분포

### 새 관찰 맞히기

관찰한 데이터가 주어졌을 때 새 관찰 $x_{n+1}$의 분포는 무엇인가?

$$
p(x_{n+1} \mid \mathcal{D}) = \int p(x_{n+1} \mid \mu) \, p(\mu \mid \mathcal{D}) \, d\mu
$$

두 분포가 모두 가우스이므로 예측 분포도 가우스이다.

$$
x_{n+1} \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)
$$

### 예측 흩어짐 쪼개기

$$
\text{Var}[x_{n+1} \mid \mathcal{D}] = \underbrace{\sigma^2}_{\text{aleatoric}} + \underbrace{\sigma_n^2}_{\text{epistemic}}
$$

- **우연의 불확실성**($\sigma^2$): 관찰에 깃든 무작위성(줄일 수 없다)
- **앎의 불확실성**($\sigma_n^2$): $\mu$에 대한 아리송함(데이터가 늘면 줄어든다)

$n \to \infty$이면 앎의 불확실성이 사라지고 예측 흩어짐은 $\sigma^2$에 다가간다.

---

## 빈도주의 추론과의 이음

### 비교표

| 갈래 | 베이즈 | 빈도주의 |
|--------|----------|-------------|
| 점 어림값 | $\mu_n$(뒤확률의 평균) | $\bar{x}$(최대 가능도) |
| 구간 | $\mu_n \pm z_{\alpha/2}\sigma_n$(믿음 구간) | $\bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$(신뢰 구간) |
| 풀이 | "$\mu$이 구간 안에 있을 확률이 95%" | "그런 구간의 95%가 $\mu$을 담는다" |
| 앞확률 필요 | 예 | 아니오 |
| 앞선 정보 아우름 | 드러내 놓고 | 곧바로는 아님 |

### 둘이 맞아떨어질 때

평평한 앞확률($\tau_0 \to 0$)에서는 다음과 같다.

$$
\mu_n \to \bar{x}, \quad \sigma_n^2 \to \frac{\sigma^2}{n}
$$

베이즈 믿음 구간이 빈도주의 신뢰 구간과 같아진다.

### 둘이 갈릴 때

정보 있는 앞확률에서는 베이즈 어림값이 앞확률의 평균 쪽으로 "벌을 받아" 끌린다. 그래서 다음을 준다.

- 앞확률이 그럴듯하면 **작은 표본에서 더 나은 거동**
- 앞확률을 크게 잘못 잡으면 **더 나쁜 어림값**

---

## 다변량으로 넓히기

### 준비

공분산 $\boldsymbol{\Sigma}$을 아는 $\boldsymbol{x}_i \in \mathbb{R}^d$에 대해 다음과 같다.

$$
\boldsymbol{x}_i \mid \boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

앞확률:

$$
\boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)
$$

### 뒤확률

$$
\boldsymbol{\mu} \mid \mathcal{D} \sim \mathcal{N}(\boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n)
$$

**뒤확률의 정밀도**(정밀도 행렬은 더해진다):

$$
\boldsymbol{\Lambda}_n = \boldsymbol{\Lambda}_0 + n\boldsymbol{\Lambda}
$$

여기서 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$이고 $\boldsymbol{\Lambda}_0 = \boldsymbol{\Sigma}_0^{-1}$이다.

**뒤확률의 평균**:

$$
\boldsymbol{\mu}_n = \boldsymbol{\Sigma}_n(\boldsymbol{\Lambda}_0\boldsymbol{\mu}_0 + n\boldsymbol{\Lambda}\bar{\boldsymbol{x}})
$$

똑같이 정밀도로 무게 준 평균 내기이며 이제 행렬 꼴이다.

---

## 파이썬 구현

```python
"""
흩어짐을 아는 가우스 추론: 온전한 구현

이 모듈은 흩어짐을 알 때 가우스 분포의 평균에 대한 베이즈 추론을 주며,
정밀도로 무게를 준 평균 내기와 차례대로 새로 고치기를
보여 준다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class GaussianPosterior:
    """
    μ의 가우스 뒤확률 분포를 나타낸다.
    
    속성
    ----------
    mean : float
        뒤확률 평균 μₙ
    variance : float
        뒤확률 흩어짐 σₙ²
    n_observations : int
        담아 넣은 관측의 개수
    """
    mean: float
    variance: float
    n_observations: int = 0
    
    @property
    def precision(self) -> float:
        """뒤확률 정밀도 τₙ = 1/σₙ²."""
        return 1.0 / self.variance
    
    @property
    def std(self) -> float:
        """뒤확률 표준편차 σₙ."""
        return np.sqrt(self.variance)
    
    def pdf(self, mu: np.ndarray) -> np.ndarray:
        """뒤확률 밀도의 값을 매긴다."""
        return stats.norm.pdf(mu, self.mean, self.std)
    
    def cdf(self, mu: float) -> float:
        """뒤확률 누적분포함수의 값을 매긴다."""
        return stats.norm.cdf(mu, self.mean, self.std)
    
    def quantile(self, p: float) -> float:
        """뒤확률 분위수를 셈한다."""
        return stats.norm.ppf(p, self.mean, self.std)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        믿음 구간을 셈한다.
        
        가우스에서는 양끝이 같은 구간과 HPD 구간이 맞아떨어진다.
        """
        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha/2)
        return (self.mean - z * self.std, self.mean + z * self.std)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """뒤확률에서 표본을 뽑는다."""
        return np.random.normal(self.mean, self.std, n_samples)
    
    def __repr__(self) -> str:
        return f"N({self.mean:.4f}, {self.variance:.4f})"

class GaussianKnownVarianceModel:
    """
    흩어짐을 알 때 가우스 평균에 대한 베이즈 추론.
    
    매개변수
    ----------
    prior_mean : float
        앞확률 평균 μ₀
    prior_variance : float
        앞확률 흩어짐 σ₀²
    known_variance : float
        아는 자료 흩어짐 σ²
    """
    
    def __init__(
        self, 
        prior_mean: float, 
        prior_variance: float, 
        known_variance: float
    ):
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.known_variance = known_variance
        
        # 정밀도
        self.prior_precision = 1.0 / prior_variance
        self.data_precision = 1.0 / known_variance
        
        # 지금 상태
        self._reset()
    
    def _reset(self):
        """앞확률 상태로 되돌린다."""
        self.current_precision = self.prior_precision
        self.current_mean = self.prior_mean
        self.n_observations = 0
        self.sum_x = 0.0
    
    @property
    def prior(self) -> GaussianPosterior:
        """앞확률 분포를 되돌린다."""
        return GaussianPosterior(self.prior_mean, self.prior_variance, 0)
    
    @property
    def posterior(self) -> GaussianPosterior:
        """지금의 뒤확률 분포를 되돌린다."""
        return GaussianPosterior(
            self.current_mean,
            1.0 / self.current_precision,
            self.n_observations
        )
    
    def update(self, data: np.ndarray) -> GaussianPosterior:
        """
        새 관측으로 뒤확률을 새로 고친다.
        
        매개변수
        ----------
        data : array
            새 관측
        
        반환값
        -------
        GaussianPosterior
            새로 고친 뒤확률
        """
        data = np.atleast_1d(data)
        n = len(data)
        
        # 충분 통계량 새로 고치기
        self.n_observations += n
        self.sum_x += data.sum()
        
        # 정밀도 새로 고치기(정밀도는 더해진다)
        self.current_precision = self.prior_precision + self.n_observations * self.data_precision
        
        # 평균 새로 고치기(정밀도로 무게 준 평균)
        self.current_mean = (
            self.prior_precision * self.prior_mean + 
            self.data_precision * self.sum_x
        ) / self.current_precision
        
        return self.posterior
    
    def update_single(self, x: float) -> GaussianPosterior:
        """관측 하나로 새로 고친다."""
        return self.update(np.array([x]))
    
    def update_sequential(self, data: np.ndarray) -> List[GaussianPosterior]:
        """
        차례대로 새로 고치며 뒤확률의 자취를 되돌린다.
        
        매개변수
        ----------
        data : array
            관측의 늘어놓음
        
        반환값
        -------
        list
            관측마다의 뒤확률
        """
        self._reset()
        history = [self.posterior]
        
        for x in data:
            self.update_single(x)
            history.append(self.posterior)
        
        return history
    
    def predictive_distribution(self) -> Tuple[float, float]:
        """
        다음 관측의 뒤확률 예측 분포를 셈한다.
        
        반환값
        -------
        tuple
            (예측_평균, 예측_흩어짐)
        """
        pred_mean = self.current_mean
        pred_var = self.known_variance + 1.0 / self.current_precision
        return pred_mean, pred_var
    
    def log_marginal_likelihood(self, data: np.ndarray) -> float:
        """
        로그 주변 가능도(로그 증거)를 셈한다.
        
        log p(D) = log ∫ p(D|μ) p(μ) dμ
        
        가우스-가우스에서는 이것을 닫힌 꼴로 얻을 수 있다.
        """
        n = len(data)
        x_bar = data.mean()
        
        # 주변 분포는 흩어짐이 부푼 가우스
        marginal_var = self.prior_variance + self.known_variance / n
        
        # 앞확률 평균에서의 제곱 어긋남의 합
        ss_from_prior = np.sum((data - self.prior_mean)**2)
        
        # 로그 주변 가능도
        log_ml = (
            -0.5 * n * np.log(2 * np.pi * self.known_variance)
            - 0.5 * ss_from_prior / self.known_variance
            + 0.5 * np.log(self.prior_variance / (self.prior_variance + self.known_variance / n))
            + 0.5 * n**2 * (x_bar - self.prior_mean)**2 / 
              (self.known_variance * (n * self.prior_variance / self.known_variance + 1))
        )
        
        return log_ml
    
    def prior_weight(self) -> float:
        """앞확률 평균에 주는 무게를 셈한다."""
        return self.prior_precision / self.current_precision
    
    def data_weight(self) -> float:
        """자료 평균에 주는 무게를 셈한다."""
        return (self.n_observations * self.data_precision) / self.current_precision
    
    def equivalent_prior_samples(self) -> float:
        """앞확률을 맞먹는 관측 개수로 나타낸다."""
        return self.prior_precision / self.data_precision

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_precision_weighted_averaging(
    model: GaussianKnownVarianceModel,
    data: np.ndarray,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """정밀도로 무게를 준 평균 내기를 그려 본다."""
    
    model._reset()
    model.update(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 왼쪽: 앞확률, 가능도, 뒤확률
    ax = axes[0]
    
    x_bar = data.mean()
    mu_range = np.linspace(
        min(model.prior_mean, x_bar) - 3 * max(np.sqrt(model.prior_variance), np.sqrt(model.known_variance)),
        max(model.prior_mean, x_bar) + 3 * max(np.sqrt(model.prior_variance), np.sqrt(model.known_variance)),
        500
    )
    
    # 앞확률
    prior_pdf = stats.norm.pdf(mu_range, model.prior_mean, np.sqrt(model.prior_variance))
    ax.plot(mu_range, prior_pdf, 'b--', linewidth=2, 
            label=f'Prior: N({model.prior_mean}, {model.prior_variance})')
    
    # 가능도(그려 보려고 고르게 함)
    likelihood_var = model.known_variance / len(data)
    likelihood_pdf = stats.norm.pdf(mu_range, x_bar, np.sqrt(likelihood_var))
    ax.plot(mu_range, likelihood_pdf, 'g:', linewidth=2,
            label=f'Likelihood: centered at x̄={x_bar:.2f}')
    
    # 뒤확률
    post = model.posterior
    posterior_pdf = post.pdf(mu_range)
    ax.fill_between(mu_range, posterior_pdf, alpha=0.3, color='red')
    ax.plot(mu_range, posterior_pdf, 'r-', linewidth=2,
            label=f'Posterior: {post}')
    
    if true_mu is not None:
        ax.axvline(true_mu, color='black', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.axvline(post.mean, color='red', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bayesian Update: Precision-Weighted Averaging', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 오른쪽: 무게 그림
    ax = axes[1]
    
    weights = [model.prior_weight(), model.data_weight()]
    labels = [f'Prior\nμ₀ = {model.prior_mean}', f'Data\nx̄ = {x_bar:.2f}']
    colors = ['steelblue', 'forestgreen']
    
    bars = ax.bar(labels, weights, color=colors, edgecolor='black', linewidth=2)
    
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Weight in Posterior Mean', fontsize=12)
    ax.set_title(f'Weights (n={len(data)}, prior ≈ {model.equivalent_prior_samples():.1f} samples)', 
                 fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_sequential_updating(
    data: np.ndarray,
    prior_mean: float,
    prior_variance: float,
    known_variance: float,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """차례대로 베이즈 새로 고치기를 그려 본다."""
    
    model = GaussianKnownVarianceModel(prior_mean, prior_variance, known_variance)
    history = model.update_sequential(data)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 위: 뒤확률 평균과 믿음 구간의 흐름
    ax = axes[0]
    
    n_vals = np.arange(len(history))
    means = [h.mean for h in history]
    cis = [h.credible_interval(0.95) for h in history]
    lowers = [ci[0] for ci in cis]
    uppers = [ci[1] for ci in cis]
    
    ax.fill_between(n_vals, lowers, uppers, alpha=0.3, color='steelblue',
                    label='95% Credible Interval')
    ax.plot(n_vals, means, 'b-', linewidth=2, marker='o', markersize=4,
            label='Posterior Mean')
    
    if true_mu is not None:
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.axhline(prior_mean, color='gray', linestyle=':', linewidth=1.5,
               label=f'Prior Mean = {prior_mean}')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('μ', fontsize=12)
    ax.set_title('Sequential Bayesian Updating', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 아래: 뒤확률 표준편차의 흐름
    ax = axes[1]
    
    stds = [h.std for h in history]
    ax.plot(n_vals, stds, 'g-', linewidth=2, marker='s', markersize=4)
    
    # 이론상 점근
    asymptotic_std = np.sqrt(known_variance) / np.sqrt(np.maximum(n_vals, 1))
    asymptotic_std[0] = np.sqrt(prior_variance)
    ax.plot(n_vals, asymptotic_std, 'r--', linewidth=1.5, 
            label=r'Asymptotic: $\sigma/\sqrt{n}$')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Posterior Std Dev', fontsize=12)
    ax.set_title('Uncertainty Reduction', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_predictive_distribution(
    model: GaussianKnownVarianceModel,
    true_mu: Optional[float] = None
) -> plt.Figure:
    """뒤확률 예측 분포를 그려 본다."""
    
    pred_mean, pred_var = model.predictive_distribution()
    post = model.posterior
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_range = np.linspace(pred_mean - 4*np.sqrt(pred_var), 
                          pred_mean + 4*np.sqrt(pred_var), 500)
    
    # μ의 뒤확률
    posterior_pdf = post.pdf(x_range)
    ax.plot(x_range, posterior_pdf, 'b-', linewidth=2,
            label=f'Posterior for μ: N({post.mean:.2f}, {post.variance:.3f})')
    
    # x_{n+1}의 예측 분포
    predictive_pdf = stats.norm.pdf(x_range, pred_mean, np.sqrt(pred_var))
    ax.fill_between(x_range, predictive_pdf, alpha=0.3, color='orange')
    ax.plot(x_range, predictive_pdf, 'orange', linewidth=2,
            label=f'Predictive for x: N({pred_mean:.2f}, {pred_var:.3f})')
    
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'True μ = {true_mu}')
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Posterior vs Predictive Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 흩어짐 쪼개기 적어 넣기
    textstr = (f'Predictive Var = {pred_var:.3f}\n'
               f'  = Aleatoric ({model.known_variance:.3f})\n'
               f'  + Epistemic ({post.variance:.3f})')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

# =============================================================================
# 보여 주기
# =============================================================================

def demo_basic_inference():
    """기본 가우스 추론을 보인다."""
    
    print("=" * 60)
    print("GAUSSIAN INFERENCE WITH KNOWN VARIANCE")
    print("=" * 60)
    
    # 준비
    true_mu = 5.0
    known_var = 4.0  # σ² = 4, 그러므로 σ = 2
    
    np.random.seed(42)
    n = 10
    data = np.random.normal(true_mu, np.sqrt(known_var), n)
    
    print(f"\nTrue μ: {true_mu}")
    print(f"Known σ²: {known_var}")
    print(f"Sample: n = {n}, x̄ = {data.mean():.4f}")
    print(f"MLE: {data.mean():.4f}")
    
    # 서로 다른 앞확률
    priors = [
        ("Weak prior (σ₀² = 100)", 0.0, 100.0),
        ("Moderate prior", 3.0, 4.0),
        ("Strong prior (wrong)", 10.0, 1.0),
        ("Strong prior (right)", 5.0, 1.0),
    ]
    
    print("\nPosterior summaries under different priors:")
    print("-" * 60)
    
    for name, mu0, var0 in priors:
        model = GaussianKnownVarianceModel(mu0, var0, known_var)
        model.update(data)
        post = model.posterior
        ci = post.credible_interval(0.95)
        
        print(f"\n{name}")
        print(f"  Prior: N({mu0}, {var0})")
        print(f"  Posterior: {post}")
        print(f"  Prior weight: {model.prior_weight():.1%}")
        print(f"  Data weight: {model.data_weight():.1%}")
        print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

def demo_sequential_updating():
    """차례대로 새로 고치기를 보인다."""
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL BAYESIAN UPDATING")
    print("=" * 60)
    
    true_mu = 100.0
    known_var = 25.0  # σ = 5
    
    np.random.seed(123)
    data = np.random.normal(true_mu, np.sqrt(known_var), 20)
    
    # 틀린 앞확률로 시작
    prior_mean = 80.0
    prior_var = 100.0
    
    print(f"\nTrue μ: {true_mu}")
    print(f"Prior: N({prior_mean}, {prior_var}) [wrong!]")
    print(f"Known σ²: {known_var}")
    
    model = GaussianKnownVarianceModel(prior_mean, prior_var, known_var)
    
    print("\nPosterior evolution:")
    print("-" * 50)
    print(f"{'n':>4} {'x':>8} {'E[μ|D]':>10} {'σ_post':>10} {'Data Wt':>10}")
    print("-" * 50)
    
    for i, x in enumerate(data[:10]):
        model.update_single(x)
        print(f"{i+1:4d} {x:8.2f} {model.current_mean:10.3f} "
              f"{model.posterior.std:10.3f} {model.data_weight():10.1%}")
    
    # 시각화 만들기
    model._reset()
    fig = plot_sequential_updating(data, prior_mean, prior_var, known_var, true_mu)
    fig.savefig('gaussian_sequential_update.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: gaussian_sequential_update.png")

def demo_predictive():
    """예측 분포를 보인다."""
    
    print("\n" + "=" * 60)
    print("POSTERIOR PREDICTIVE DISTRIBUTION")
    print("=" * 60)
    
    true_mu = 50.0
    known_var = 16.0  # σ = 4
    
    np.random.seed(456)
    data = np.random.normal(true_mu, np.sqrt(known_var), 15)
    
    model = GaussianKnownVarianceModel(
        prior_mean=45.0,
        prior_variance=25.0,
        known_variance=known_var
    )
    model.update(data)
    
    pred_mean, pred_var = model.predictive_distribution()
    post = model.posterior
    
    print(f"\nObserved: {len(data)} observations")
    print(f"Posterior for μ: N({post.mean:.2f}, {post.variance:.4f})")
    print(f"\nPredictive for x_{len(data)+1}:")
    print(f"  Mean: {pred_mean:.2f}")
    print(f"  Variance: {pred_var:.4f}")
    print(f"    = Aleatoric ({known_var:.4f}) + Epistemic ({post.variance:.4f})")
    
    # 95% 예측 구간
    z = 1.96
    pi_lower = pred_mean - z * np.sqrt(pred_var)
    pi_upper = pred_mean + z * np.sqrt(pred_var)
    print(f"  95% Prediction Interval: [{pi_lower:.2f}, {pi_upper:.2f}]")
    
    fig = plot_predictive_distribution(model, true_mu)
    fig.savefig('gaussian_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSee: gaussian_predictive.png")

if __name__ == "__main__":
    demo_basic_inference()
    demo_sequential_updating()
    demo_predictive()
```

---

## 요약

| 갈래 | 식 |
|--------|---------|
| **앞확률** | $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$ |
| **가능도** | $p(\mathcal{D} \mid \mu) \propto \exp\left(-\frac{n(\bar{x}-\mu)^2}{2\sigma^2}\right)$ |
| **뒤확률** | $\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)$ |
| **뒤확률의 정밀도** | $\tau_n = \tau_0 + n\tau$ |
| **뒤확률의 평균** | $\mu_n = \frac{\tau_0\mu_0 + n\tau\bar{x}}{\tau_n}$ |
| **예측** | $x_{n+1} \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2)$ |

### 핵심 통찰

1. **정밀도는 더해진다**: 뒤확률의 정밀도 = 앞확률의 정밀도 + 데이터의 정밀도
2. **정밀도로 무게 준 평균 내기**: 뒤확률의 평균은 정밀도로 정보의 무게를 준다
3. **맞먹는 표본 크기**: 앞확률은 관찰 $n_0 = \sigma^2/\sigma_0^2$개의 값어치를 지닌다
4. **오그라들기**: 뒤확률의 평균은 최대 가능도 어림값을 앞확률의 평균 쪽으로 오그라뜨린다
5. **예측 흩어짐**: 우연의 불확실성과 앎의 불확실성으로 쪼개진다
6. **칼만 거르개**: 차례 갱신이 곧 가장 단순한 칼만 거르개이다

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 베타-베르누이 | 13장: 베르누이-베타 | 낱낱의 값에서의 대응물 |
| 흩어짐을 모를 때 | 13장: 흩어짐을 모르는 가우스 | 더 현실에 가까운 상황 |
| 다변량 | 13장: 베이즈 선형 회귀 | 회귀로 넓히기 |
| 칼만 거르개 | 16장: 상태 공간 모형 | 차례 추론 |
| BNN 앞확률 | 13장: BNN 앞확률 | 가중치 분포 설계 |

### 주요 참고 문헌

- DeGroot, M. H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). 2장.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. 4장.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. 2.3절.

---

# 흩어짐을 모를 때의 가우스 추론

가우스 분포의 평균 $\mu$과 흩어짐 $\sigma^2$을 모두 모르면 베이즈 추론에는 두 매개변수 위의 결합 앞확률이 필요하다. 켤레 앞확률은 **정규-역감마**(NIG) 분포이며, 이는 우아한 닫힌 꼴의 뒤확률로 이어진다. 이 상황은 흩어짐을 아는 경우보다 훨씬 현실에 가까우며 **귀찮은 매개변수를 주변화하기**라는 중요한 개념을 들여온다.

---

## 문제 설정

### 추론 문제

**평균 $\mu$과 흩어짐 $\sigma^2$을 모두 모르는** 가우스 분포에서 나왔다고 놓는 이어진 측정값 $x_1, x_2, \ldots, x_n \in \mathbb{R}$을 관찰한다.

$$
x_i \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)
$$

목표는 결합 뒤확률 분포 $p(\mu, \sigma^2 \mid \mathcal{D})$과, 무엇보다 $\mu$만의 **주변 뒤확률**을 미루어 아는 것이다.

$$
p(\mu \mid \mathcal{D}) = \int_0^\infty p(\mu, \sigma^2 \mid \mathcal{D}) \, d\sigma^2
$$

### 흩어짐을 모르는 것이 중요한 까닭

실제 응용에서는 대개 흩어짐을 모른다.

- **과학 실험**: 장치마다 측정 정밀도가 다르다
- **금융 데이터**: 변동성이 때에 따라 바뀐다
- **A/B 시험**: 효과 크기의 흔들림을 대개 모른다
- **기계 학습**: 모형의 불확실성 어림

흩어짐을 안다는 가정은 가르치기에는 쓸모 있지만 현실에서는 드물다.

### 가우스 가능도

서로 독립인 관찰 $n$개에 대해 다음과 같다.

$$
p(\mathcal{D} \mid \mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

$$
= (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2\right)
$$

### 충분 통계량

가능도는 데이터에 대해 오직 충분 통계량 둘을 거쳐서만 달라진다.

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \quad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

다음 쪼갬을 쓰면

$$
\sum_{i=1}^{n}(x_i - \mu)^2 = (n-1)s^2 + n(\bar{x} - \mu)^2
$$

가능도는 다음이 된다.

$$
p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{(n-1)s^2 + n(\bar{x} - \mu)^2}{2\sigma^2}\right)
$$

---

## 역감마 분포

### 정의

**역감마** 분포는 흩어짐 매개변수의 켤레 앞확률이다. $\sigma^2 \sim \text{Inv-Gamma}(\alpha, \beta)$이면 다음과 같다.

$$
p(\sigma^2) = \frac{\beta^\alpha}{\Gamma(\alpha)} (\sigma^2)^{-\alpha-1} \exp\left(-\frac{\beta}{\sigma^2}\right), \quad \sigma^2 > 0
$$

### 매개변수와 적률

| 매개변수 | 기호 | 풀이 |
|-----------|--------|----------------|
| 모양 | $\alpha$ | 몰림 정도를 다스린다(클수록 더 뾰족하다) |
| 눈금 | $\beta$ | 자리를 다스린다(클수록 흩어짐이 커진다) |

**적률**(각각 $\alpha > 1$과 $\alpha > 2$일 때):

$$
\mathbb{E}[\sigma^2] = \frac{\beta}{\alpha - 1}, \quad \text{Var}[\sigma^2] = \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}
$$

**최빈값**:

$$
\text{Mode}[\sigma^2] = \frac{\beta}{\alpha + 1}
$$

### 카이제곱과의 이음

$X \sim \chi^2_\nu$이면 다음과 같다.

$$
\frac{\nu s_0^2}{X} \sim \text{Inv-Gamma}\left(\frac{\nu}{2}, \frac{\nu s_0^2}{2}\right)
$$

이는 표본 흩어짐의 표집 분포와 이어진다.

### 왜 역감마인가?

역감마가 흩어짐에 자연스러운 까닭은 다음과 같다.

1. **받침**: $(0, \infty)$ 위에서 정의되어 흩어짐의 영역과 맞는다
2. **켤레성**: 닫힌 꼴의 뒤확률로 이어진다
3. **풀이 가능성**: 매개변수가 "앞선 관찰"과 이어진다

---

## 정규-역감마 앞확률

### 결합 앞확률 명세

$(\mu, \sigma^2)$의 켤레 앞확률은 **정규-역감마**(NIG) 분포이다.

$$
\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

$$
\mu \mid \sigma^2 \sim \mathcal{N}\left(\mu_0, \frac{\sigma^2}{\kappa_0}\right)
$$

이를 다음과 같이 쓴다.

$$
(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)
$$

### 결합 밀도

$$
p(\mu, \sigma^2) = p(\mu \mid \sigma^2) \cdot p(\sigma^2)
$$

$$
= \frac{1}{\sqrt{2\pi\sigma^2/\kappa_0}} \exp\left(-\frac{\kappa_0(\mu - \mu_0)^2}{2\sigma^2}\right) \cdot \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)} (\sigma^2)^{-\alpha_0-1} \exp\left(-\frac{\beta_0}{\sigma^2}\right)
$$

$$
\propto (\sigma^2)^{-\alpha_0-3/2} \exp\left(-\frac{1}{\sigma^2}\left[\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2\right]\right)
$$

### 앞확률 매개변수의 풀이

| 매개변수 | 기호 | 풀이 |
|-----------|--------|----------------|
| 앞확률 평균의 자리 | $\mu_0$ | 데이터를 보기 전 $\mu$에 대한 가장 나은 어림 |
| 앞확률 정밀도의 눈금 | $\kappa_0$ | 평균에 대한 "맞먹는 관찰 수" |
| 흩어짐의 모양 | $\alpha_0$ | "앞확률의 자유도"의 절반 |
| 흩어짐의 눈금 | $\beta_0$ | 앞확률의 흩어짐 어림값에 눈금을 준다 |

**가짜 관찰로 풀이하기**:

- $\kappa_0$: 앞확률이 $\mu$을 어림하는 데 관찰 $\kappa_0$개의 값어치를 지닌다
- $2\alpha_0$: 앞확률이 $\sigma^2$을 어림하는 데 관찰 $2\alpha_0$개의 값어치를 지닌다
- $\beta_0 / \alpha_0$: (최빈값에서) $\sigma^2$의 앞확률 어림값

### 흔한 앞확률 선택

**약하게 정보 있는 앞확률**:

$$
\mu_0 = 0, \quad \kappa_0 = 0.01, \quad \alpha_0 = 0.01, \quad \beta_0 = 0.01
$$

**제프리스 앞확률**(제대로 되지는 않지만 참조로 쓴다):

$$
p(\mu, \sigma^2) \propto \frac{1}{\sigma^2}
$$

이는 $\kappa_0 \to 0$, $\alpha_0 \to 0$, $\beta_0 \to 0$에 해당한다.

**데이터에 기댄 앞확률**(경험적 베이즈 방식):

$$
\mu_0 = \bar{x}_{\text{pilot}}, \quad \alpha_0 = 1, \quad \beta_0 = s^2_{\text{pilot}}
$$

---

## 켤레 뒤확률 끌어내기

### 끌어내기

**앞확률**:

$$
p(\mu, \sigma^2) \propto (\sigma^2)^{-\alpha_0-3/2} \exp\left(-\frac{1}{\sigma^2}\left[\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2\right]\right)
$$

**가능도**:

$$
p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\left[(n-1)s^2 + n(\bar{x} - \mu)^2\right]\right)
$$

**뒤확률**(베이즈 정리로):

$$
p(\mu, \sigma^2 \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mu, \sigma^2) \cdot p(\mu, \sigma^2)
$$

### 지수 합치기

$\sigma^2$의 지수는 다음이 된다.

$$
-\alpha_0 - \frac{3}{2} - \frac{n}{2} = -\left(\alpha_0 + \frac{n}{2}\right) - \frac{3}{2} = -\alpha_n - \frac{3}{2}
$$

여기서 $\alpha_n = \alpha_0 + n/2$이다.

지수 안의 항은 다음과 같다.

$$
\beta_0 + \frac{\kappa_0}{2}(\mu - \mu_0)^2 + \frac{(n-1)s^2}{2} + \frac{n}{2}(\bar{x} - \mu)^2
$$

### mu에 대해 완전제곱 만들기

$\mu$에 기댄 항은 다음과 같다.

$$
\frac{\kappa_0}{2}(\mu - \mu_0)^2 + \frac{n}{2}(\bar{x} - \mu)^2
$$

$$
= \frac{\kappa_0}{2}\left[\mu^2 - 2\mu\mu_0 + \mu_0^2\right] + \frac{n}{2}\left[\mu^2 - 2\mu\bar{x} + \bar{x}^2\right]
$$

$$
= \frac{\kappa_0 + n}{2}\mu^2 - \mu(\kappa_0\mu_0 + n\bar{x}) + \frac{\kappa_0\mu_0^2 + n\bar{x}^2}{2}
$$

완전제곱을 만들면 다음과 같다.

$$
= \frac{\kappa_n}{2}\left(\mu - \mu_n\right)^2 + \text{const}
$$

여기서 각 기호는 다음과 같다.

$$
\kappa_n = \kappa_0 + n, \quad \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_0 + n}
$$

### beta_n 갱신

($\mu$과 무관한) 상수 항이 $\beta_n$에 보태진다.

$$
\beta_n = \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2(\kappa_0 + n)}
$$

마지막 항은 완전제곱을 만들며 나오는데, 평균에 대한 "앞확률과 데이터의 부딪침"을 나타낸다.

### 뒤확률 분포

$$
\boxed{(\mu, \sigma^2) \mid \mathcal{D} \sim \text{NIG}(\mu_n, \kappa_n, \alpha_n, \beta_n)}
$$

갱신 식은 다음과 같다.

$$
\boxed{
\begin{aligned}
\kappa_n &= \kappa_0 + n \\
\mu_n &= \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_n} \\
\alpha_n &= \alpha_0 + \frac{n}{2} \\
\beta_n &= \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2\kappa_n}
\end{aligned}
}
$$

---

## 주변 뒤확률 분포

### 시그마 제곱의 주변 분포

$\mu$을 적분해 없애면 다음과 같다.

$$
p(\sigma^2 \mid \mathcal{D}) = \int_{-\infty}^{\infty} p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu
$$

$$
\sigma^2 \mid \mathcal{D} \sim \text{Inv-Gamma}(\alpha_n, \beta_n)
$$

**점 어림값**:

$$
\mathbb{E}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n - 1} \quad (\text{if } \alpha_n > 1)
$$

$$
\text{Mode}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n + 1}
$$

### mu의 주변 분포: 스튜던트 t 분포

$\sigma^2$을 적분해 없애면 다음과 같다.

$$
p(\mu \mid \mathcal{D}) = \int_0^\infty p(\mu, \sigma^2 \mid \mathcal{D}) \, d\sigma^2
$$

이 적분은 **스튜던트 t 분포**를 낸다.

$$
\boxed{\mu \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \frac{\beta_n}{\alpha_n \kappa_n}\right)}
$$

여기서 $t_\nu(\mu, \sigma^2)$은 자유도 $\nu$, 자리 $\mu$, 눈금 $\sigma$인 스튜던트 t를 뜻한다.

### 스튜던트 t 분포

자유도 $\nu$, 자리 $\mu$, 눈금 $\sigma$인 스튜던트 t의 확률밀도함수는 다음과 같다.

$$
p(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\sigma} \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}
$$

**성질**:

| 성질 | 식 |
|----------|---------|
| 평균 | $\mu$($\nu > 1$일 때) |
| 흩어짐 | $\frac{\nu}{\nu-2}\sigma^2$($\nu > 2$일 때) |
| 자유도 | $\nu = 2\alpha_n$ |

### 왜 스튜던트 t인가?

스튜던트 t가 나오는 까닭은 다음과 같다.

1. **$\sigma^2$에 대한 아리송함** 때문에 꼬리가 가우스보다 두꺼워진다
2. **데이터가 늘면** $\nu \to \infty$이고 $t_\nu \to \mathcal{N}$이다
3. **튼튼함**: 두꺼운 꼬리가 튀는 값을 받아 준다

---

## 뒤확률 분석

### mu의 점 어림값

**뒤확률의 평균**(스튜던트 t에서는 뒤확률의 최빈값과 같다):

$$
\mathbb{E}[\mu \mid \mathcal{D}] = \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_0 + n}
$$

이는 흩어짐을 아는 경우와 똑같이 정밀도로 무게 준 평균이다!

**뒤확률의 흩어짐**($\nu = 2\alpha_n > 2$일 때):

$$
\text{Var}[\mu \mid \mathcal{D}] = \frac{\nu}{\nu - 2} \cdot \frac{\beta_n}{\alpha_n \kappa_n} = \frac{\beta_n}{(\alpha_n - 1)\kappa_n}
$$

### 시그마 제곱의 점 어림값

**뒤확률의 평균**:

$$
\mathbb{E}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n - 1}
$$

**뒤확률의 최빈값**:

$$
\text{Mode}[\sigma^2 \mid \mathcal{D}] = \frac{\beta_n}{\alpha_n + 1}
$$

### 믿음 구간

**$\mu$에 대해**: 스튜던트 t 분위수를 쓴다

$$
\left[\mu_n - t_{\nu, \alpha/2} \cdot \sqrt{\frac{\beta_n}{\alpha_n\kappa_n}}, \; \mu_n + t_{\nu, \alpha/2} \cdot \sqrt{\frac{\beta_n}{\alpha_n\kappa_n}}\right]
$$

**$\sigma^2$에 대해**: 역감마 분위수를 쓴다(대칭이 아닌 구간)

---

## 빈도주의 추론과의 이음

### t 검정과의 이음

제프리스 앞확률($\kappa_0 \to 0$, $\alpha_0 \to 0$, $\beta_0 \to 0$)에서는 다음과 같다.

$$
\mu_n \to \bar{x}, \quad \kappa_n \to n, \quad \alpha_n \to \frac{n}{2}, \quad \beta_n \to \frac{(n-1)s^2}{2}
$$

$\mu$의 주변 뒤확률은 다음이 된다.

$$
\mu \mid \mathcal{D} \sim t_{n-1}\left(\bar{x}, \frac{s^2}{n}\right)
$$

이는 한 표본 $t$ 검정에서 쓰는 빈도주의 표집 분포와 꼭 맞는다!

### 비교표

| 갈래 | 베이즈(제프리스) | 빈도주의 |
|--------|---------------------|-------------|
| 점 어림값 | $\bar{x}$ | $\bar{x}$ |
| $\mu$의 구간 | $\bar{x} \pm t_{n-1,\alpha/2} \cdot \frac{s}{\sqrt{n}}$ | $\bar{x} \pm t_{n-1,\alpha/2} \cdot \frac{s}{\sqrt{n}}$ |
| 분포 | 뒤확률($\mu$에 대한 확률) | 표집 분포 |
| 풀이 | "$\mu$이 구간 안에 있을 확률이 95%" | "구간의 95%가 $\mu$을 담는다" |

### 놀라운 일치

제프리스 앞확률에서는 베이즈 믿음 구간이 빈도주의 신뢰 구간과 꼭 맞는다. 우연이 아니라 다음 둘 사이의 깊은 이음을 비추는 것이다.

- **제프리스 앞확률**: "객관적인" 베이즈 추론을 위해 설계되었다
- **최대 가능도**: 규칙성 조건 아래 점근적으로 효율적이다

---

## 차례 갱신

### 온라인 학습

NIG 켤레족은 차례 갱신을 가능하게 한다.

$$
\text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0) \xrightarrow{x_1} \text{NIG}(\mu_1, \kappa_1, \alpha_1, \beta_1) \xrightarrow{x_2} \cdots
$$

**관찰 하나에 대한 갱신**(관찰 $x$이 주어질 때):

$$
\begin{aligned}
\kappa_{t+1} &= \kappa_t + 1 \\
\mu_{t+1} &= \frac{\kappa_t \mu_t + x}{\kappa_{t+1}} \\
\alpha_{t+1} &= \alpha_t + \frac{1}{2} \\
\beta_{t+1} &= \beta_t + \frac{\kappa_t(x - \mu_t)^2}{2\kappa_{t+1}}
\end{aligned}
$$

### 해석

- $\kappa_t$과 $\alpha_t$은 관찰 수에 따라 선형으로 자란다
- $\mu_t$은 표본 평균으로 모인다
- $\beta_t$은 제곱 편차를 알맞게 눈금 맞추어 쌓는다

---

## 뒤확률 예측 분포

### 새 관찰 맞히기

뒤확률 예측은 모르는 두 매개변수 위에서 적분한다.

$$
p(x_{n+1} \mid \mathcal{D}) = \int_0^\infty \int_{-\infty}^\infty p(x_{n+1} \mid \mu, \sigma^2) \, p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu \, d\sigma^2
$$

**결과**: 예측 분포도 스튜던트 t이다.

$$
\boxed{x_{n+1} \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \frac{\beta_n(\kappa_n + 1)}{\alpha_n \kappa_n}\right)}
$$

### 예측 흩어짐 쪼개기

$\nu = 2\alpha_n > 2$일 때 다음과 같다.

$$
\text{Var}[x_{n+1} \mid \mathcal{D}] = \frac{\nu}{\nu - 2} \cdot \frac{\beta_n(\kappa_n + 1)}{\alpha_n \kappa_n}
$$

이 흩어짐은 세 성분으로 이루어진다.

1. **우연**: 관찰에 깃든 무작위성
2. **앎(평균)**: $\mu$에 대한 아리송함
3. **앎(흩어짐)**: $\sigma^2$에 대한 아리송함

$n \to \infty$이면 우연의 불확실성만 남는다.

---

## 점근 거동

### 큰 표본의 극한

$n \to \infty$이면 다음과 같다.

**$\mu$의 뒤확률**:

$$
\mu \mid \mathcal{D} \xrightarrow{d} \mathcal{N}\left(\bar{x}, \frac{s^2}{n}\right)
$$

자유도가 커지면 스튜던트 t는 가우스로 모인다.

**$\sigma^2$의 뒤확률**:

$$
\sigma^2 \mid \mathcal{D} \xrightarrow{p} s^2
$$

뒤확률은 표본 흩어짐 둘레에 몰린다.

### 앞확률이 씻겨 나감

데이터가 넉넉하면 그럴듯한 앞확률은 "씻겨 나간다".

$$
\frac{\kappa_0}{\kappa_n} \to 0, \quad \frac{\alpha_0}{\alpha_n} \to 0
$$

뒤확률은 가능도가 좌우한다.

---

## 수치적 안정성에 대한 고려

### beta_n 셈하기

$\beta_n$의 식은 수치 문제를 겪을 수 있다. 더 안정된 꼴은 다음과 같다.

$$
\beta_n = \beta_0 + \frac{1}{2}\left[\sum_{i=1}^n (x_i - \bar{x})^2 + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - \mu_0)^2\right]
$$

제곱합에 웰퍼드 알고리즘을 쓰면 수치가 안정된다.

### 로그 공간에서 셈하기

밀도를 셈할 때는 로그 공간에서 다룬다.

$$
\log p(\sigma^2 \mid \mathcal{D}) = \alpha_n \log \beta_n - \log\Gamma(\alpha_n) - (\alpha_n + 1)\log\sigma^2 - \frac{\beta_n}{\sigma^2}
$$

---

## 파이썬 구현

```python
"""
흩어짐을 모르는 가우스 추론: 온전한 구현

이 모듈은 정규-역감마 켤레 앞확률을 써서 가우스 분포의 평균과 흩어짐에 대한
베이즈 추론을 주며, 평균의 주변 뒤확률이 스튜던트 t임을
보여 준다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class NIGParameters:
    """
    정규-역감마 분포의 매개변수.
    
    NIG 분포는 다음과 같이 매개변수로 나타낸다:
        σ² ~ 역감마(α, β)
        μ | σ² ~ N(μ₀, σ²/κ)
    
    속성
    ----------
    mu : float
        위치 매개변수 μ₀
    kappa : float
        정밀도 눈금 κ(평균에 대한 실효 표본 크기)
    alpha : float
        흩어짐의 모양 매개변수 α
    beta : float
        흩어짐의 눈금 매개변수 β
    """
    mu: float
    kappa: float
    alpha: float
    beta: float
    
    def __post_init__(self):
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
    
    @property
    def variance_mean(self) -> Optional[float]:
        """α > 1이면 E[σ²] = β/(α-1)."""
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return None
    
    @property
    def variance_mode(self) -> float:
        """Mode[σ²] = β/(α+1)."""
        return self.beta / (self.alpha + 1)
    
    @property
    def degrees_of_freedom(self) -> float:
        """μ의 주변 t분포의 자유도."""
        return 2 * self.alpha
    
    @property
    def mu_scale(self) -> float:
        """μ의 주변 t분포의 눈금 매개변수."""
        return np.sqrt(self.beta / (self.alpha * self.kappa))
    
    def __repr__(self) -> str:
        return f"NIG(μ={self.mu:.4f}, κ={self.kappa:.4f}, α={self.alpha:.4f}, β={self.beta:.4f})"

class StudentTPosterior:
    """
    μ의 주변 스튜던트 t 뒤확률을 나타낸다.
    
    매개변수
    ----------
    loc : float
        위치 매개변수(뒤확률 평균)
    scale : float
        배율 매개변수
    df : float
        자유도
    """
    
    def __init__(self, loc: float, scale: float, df: float):
        self.loc = loc
        self.scale = scale
        self.df = df
        self._dist = stats.t(df=df, loc=loc, scale=scale)
    
    @property
    def mean(self) -> Optional[float]:
        """df > 1이면 평균이 있다."""
        return self.loc if self.df > 1 else None
    
    @property
    def variance(self) -> Optional[float]:
        """df > 2이면 흩어짐이 있다."""
        if self.df > 2:
            return (self.df / (self.df - 2)) * self.scale**2
        return None
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """뒤확률 밀도의 값을 매긴다."""
        return self._dist.pdf(x)
    
    def cdf(self, x: float) -> float:
        """뒤확률 누적분포함수의 값을 매긴다."""
        return self._dist.cdf(x)
    
    def quantile(self, p: float) -> float:
        """뒤확률 분위수를 셈한다."""
        return self._dist.ppf(p)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """양끝이 같은 믿음 구간을 셈한다."""
        alpha = 1 - level
        return (self.quantile(alpha/2), self.quantile(1 - alpha/2))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """뒤확률에서 표본을 뽑는다."""
        return self._dist.rvs(n_samples)
    
    def __repr__(self) -> str:
        return f"t_{self.df:.1f}({self.loc:.4f}, {self.scale:.4f})"

class InverseGammaPosterior:
    """
    σ²의 주변 역감마 뒤확률을 나타낸다.
    
    매개변수
    ----------
    alpha : float
        모양 매개변수
    beta : float
        배율 매개변수
    """
    
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.invgamma(a=alpha, scale=beta)
    
    @property
    def mean(self) -> Optional[float]:
        """α > 1이면 평균이 있다."""
        return self.beta / (self.alpha - 1) if self.alpha > 1 else None
    
    @property
    def mode(self) -> float:
        """최빈값 = β/(α+1)."""
        return self.beta / (self.alpha + 1)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """뒤확률 밀도의 값을 매긴다."""
        return self._dist.pdf(x)
    
    def cdf(self, x: float) -> float:
        """뒤확률 누적분포함수의 값을 매긴다."""
        return self._dist.cdf(x)
    
    def quantile(self, p: float) -> float:
        """뒤확률 분위수를 셈한다."""
        return self._dist.ppf(p)
    
    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """양끝이 같은 믿음 구간을 셈한다."""
        alpha = 1 - level
        return (self.quantile(alpha/2), self.quantile(1 - alpha/2))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """뒤확률에서 표본을 뽑는다."""
        return self._dist.rvs(n_samples)
    
    def __repr__(self) -> str:
        return f"Inv-Gamma({self.alpha:.4f}, {self.beta:.4f})"

class GaussianUnknownVarianceModel:
    """
    평균과 흩어짐을 모르는 가우스에 대한 베이즈 추론.
    
    정규-역감마 켤레 앞확률을 쓴다.
    
    매개변수
    ----------
    prior_mu : float
        앞확률 평균 위치 μ₀
    prior_kappa : float
        앞확률 정밀도 눈금 κ₀
    prior_alpha : float
        앞확률 모양 α₀
    prior_beta : float
        앞확률 눈금 β₀
    """
    
    def __init__(
        self,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        self.prior = NIGParameters(prior_mu, prior_kappa, prior_alpha, prior_beta)
        self._reset()
    
    def _reset(self):
        """앞확률 상태로 되돌린다."""
        self.current = NIGParameters(
            self.prior.mu, self.prior.kappa, 
            self.prior.alpha, self.prior.beta
        )
        self.n_observations = 0
        self._data_sum = 0.0
        self._data_sum_sq = 0.0
    
    @property
    def posterior_nig(self) -> NIGParameters:
        """지금의 NIG 뒤확률 매개변수를 되돌린다."""
        return self.current
    
    @property
    def posterior_mu(self) -> StudentTPosterior:
        """μ의 주변 뒤확률(스튜던트 t)을 되돌린다."""
        return StudentTPosterior(
            loc=self.current.mu,
            scale=self.current.mu_scale,
            df=self.current.degrees_of_freedom
        )
    
    @property
    def posterior_variance(self) -> InverseGammaPosterior:
        """σ²의 주변 뒤확률(역감마)을 되돌린다."""
        return InverseGammaPosterior(
            alpha=self.current.alpha,
            beta=self.current.beta
        )
    
    def update(self, data: np.ndarray) -> NIGParameters:
        """
        새 관측으로 뒤확률을 새로 고친다.
        
        매개변수
        ----------
        data : array
            새 관측
        
        반환값
        -------
        NIGParameters
            새로 고친 뒤확률 매개변수
        """
        data = np.atleast_1d(data).astype(float)
        n = len(data)
        
        if n == 0:
            return self.current
        
        # 충분 통계량 새로 고치기
        self.n_observations += n
        self._data_sum += data.sum()
        self._data_sum_sq += (data**2).sum()
        
        # 전체 표본 평균
        overall_mean = self._data_sum / self.n_observations
        
        # 표본 흩어짐 셈하기(모든 자료를 써서)
        if self.n_observations > 1:
            ss = self._data_sum_sq - self.n_observations * overall_mean**2
        else:
            ss = 0.0
        
        # NIG 새로 고치기 공식
        kappa_n = self.prior.kappa + self.n_observations
        mu_n = (self.prior.kappa * self.prior.mu + self._data_sum) / kappa_n
        alpha_n = self.prior.alpha + self.n_observations / 2
        
        # 베타 새로 고치기
        prior_data_sq = (self.prior.kappa * self.n_observations / kappa_n) * \
                        (overall_mean - self.prior.mu)**2
        beta_n = self.prior.beta + 0.5 * ss + 0.5 * prior_data_sq
        
        self.current = NIGParameters(mu_n, kappa_n, alpha_n, beta_n)
        return self.current
    
    def update_single(self, x: float) -> NIGParameters:
        """온라인 공식을 써서 관측 하나로 새로 고친다."""
        kappa_old = self.current.kappa
        mu_old = self.current.mu
        
        # 매개변수 갱신
        kappa_new = kappa_old + 1
        mu_new = (kappa_old * mu_old + x) / kappa_new
        alpha_new = self.current.alpha + 0.5
        beta_new = self.current.beta + (kappa_old * (x - mu_old)**2) / (2 * kappa_new)
        
        self.current = NIGParameters(mu_new, kappa_new, alpha_new, beta_new)
        self.n_observations += 1
        self._data_sum += x
        self._data_sum_sq += x**2
        
        return self.current
    
    def update_sequential(self, data: np.ndarray) -> List[NIGParameters]:
        """차례대로 새로 고치며 뒤확률의 자취를 되돌린다."""
        self._reset()
        history = [self.current]
        
        for x in data:
            self.update_single(x)
            history.append(self.current)
        
        return history
    
    def predictive_distribution(self) -> StudentTPosterior:
        """다음 관측의 뒤확률 예측 분포를 셈한다."""
        pred_scale = np.sqrt(
            self.current.beta * (self.current.kappa + 1) / 
            (self.current.alpha * self.current.kappa)
        )
        return StudentTPosterior(
            loc=self.current.mu,
            scale=pred_scale,
            df=self.current.degrees_of_freedom
        )
    
    def sample_posterior(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """뒤확률에서 (mu, sigma2) 결합 표본을 뽑는다."""
        # 먼저 역감마에서 σ²을 표집
        sigma2_samples = self.posterior_variance.sample(n_samples)
        
        # 그다음 정규에서 μ | σ²을 표집
        mu_std = np.sqrt(sigma2_samples / self.current.kappa)
        mu_samples = np.random.normal(self.current.mu, mu_std)
        
        return mu_samples, sigma2_samples
    
    def log_marginal_likelihood(self, data: np.ndarray) -> float:
        """로그 주변 가능도(모형 증거)를 셈한다."""
        data = np.atleast_1d(data)
        n = len(data)
        
        if n == 0:
            return 0.0
        
        # 뒤확률 매개변수 셈하기
        x_bar = data.mean()
        ss = ((data - x_bar)**2).sum() if n > 1 else 0.0
        
        kappa_n = self.prior.kappa + n
        alpha_n = self.prior.alpha + n / 2
        prior_data_sq = (self.prior.kappa * n / kappa_n) * (x_bar - self.prior.mu)**2
        beta_n = self.prior.beta + 0.5 * ss + 0.5 * prior_data_sq
        
        # 로그 주변 가능도
        log_ml = (
            gammaln(alpha_n) - gammaln(self.prior.alpha)
            + self.prior.alpha * np.log(self.prior.beta) - alpha_n * np.log(beta_n)
            + 0.5 * np.log(self.prior.kappa / kappa_n)
            - (n / 2) * np.log(2 * np.pi)
        )
        
        return log_ml

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_joint_posterior(
    model: GaussianUnknownVarianceModel,
    true_mu: Optional[float] = None,
    true_sigma2: Optional[float] = None,
    n_grid: int = 100
) -> plt.Figure:
    """결합 뒤확률과 주변 뒤확률을 그려 본다."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    post_mu = model.posterior_mu
    post_var = model.posterior_variance
    
    # 그릴 범위 정하기
    mu_std = post_mu.scale * np.sqrt(post_mu.df / (post_mu.df - 2)) if post_mu.df > 2 else post_mu.scale * 3
    mu_range = (post_mu.loc - 4*mu_std, post_mu.loc + 4*mu_std)
    
    var_mean = post_var.mean if post_var.mean is not None else post_var.mode
    var_range = (max(0.01, var_mean * 0.1), var_mean * 3)
    
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_grid)
    var_vals = np.linspace(var_range[0], var_range[1], n_grid)
    
    # 왼쪽 위: 결합 뒤확률 등고선
    ax = axes[0, 0]
    MU, VAR = np.meshgrid(mu_vals, var_vals)
    
    joint_log_pdf = np.zeros_like(MU)
    for i, v in enumerate(var_vals):
        mu_given_var = stats.norm(loc=model.current.mu, scale=np.sqrt(v / model.current.kappa))
        joint_log_pdf[i, :] = mu_given_var.logpdf(mu_vals) + post_var._dist.logpdf(v)
    
    joint_pdf = np.exp(joint_log_pdf - joint_log_pdf.max())
    
    contour = ax.contourf(MU, VAR, joint_pdf, levels=20, cmap='Blues')
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2)
    if true_sigma2 is not None:
        ax.axhline(true_sigma2, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Joint Posterior p(μ, σ² | D)', fontsize=14)
    plt.colorbar(contour, ax=ax)
    
    # 오른쪽 위: μ의 주변 분포
    ax = axes[0, 1]
    ax.plot(mu_vals, post_mu.pdf(mu_vals), 'b-', linewidth=2, label=f'{post_mu}')
    ax.fill_between(mu_vals, post_mu.pdf(mu_vals), alpha=0.3)
    if true_mu is not None:
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Marginal Posterior for μ (df = {post_mu.df:.0f})', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 왼쪽 아래: σ²의 주변 분포
    ax = axes[1, 0]
    ax.plot(var_vals, post_var.pdf(var_vals), 'b-', linewidth=2, label=f'{post_var}')
    ax.fill_between(var_vals, post_var.pdf(var_vals), alpha=0.3)
    if true_sigma2 is not None:
        ax.axvline(true_sigma2, color='red', linestyle='--', linewidth=2)
    ax.axvline(post_var.mode, color='green', linestyle=':', label=f'Mode = {post_var.mode:.3f}')
    ax.set_xlabel('σ²', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Marginal Posterior for σ²', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 오른쪽 아래: 뒤확률 표본
    ax = axes[1, 1]
    mu_samples, var_samples = model.sample_posterior(1000)
    ax.scatter(mu_samples, var_samples, alpha=0.3, s=10, c='steelblue')
    if true_mu is not None and true_sigma2 is not None:
        ax.scatter([true_mu], [true_sigma2], color='red', s=100, marker='*', zorder=5)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Posterior Samples', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_sequential_updating(
    data: np.ndarray,
    prior_mu: float,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
    true_mu: Optional[float] = None,
    true_sigma2: Optional[float] = None
) -> plt.Figure:
    """차례대로 베이즈 새로 고치기를 그려 본다."""
    
    model = GaussianUnknownVarianceModel(prior_mu, prior_kappa, prior_alpha, prior_beta)
    history = model.update_sequential(data)
    
    n_vals = np.arange(len(history))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 왼쪽 위: μ의 뒤확률 평균
    ax = axes[0, 0]
    mu_means = [h.mu for h in history]
    
    ci_lower, ci_upper = [], []
    for h in history:
        post = StudentTPosterior(h.mu, h.mu_scale, h.degrees_of_freedom)
        ci = post.credible_interval(0.95)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    
    ax.fill_between(n_vals, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(n_vals, mu_means, 'b-', linewidth=2, marker='o', markersize=4, label='E[μ|D]')
    if true_mu is not None:
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2, label=f'True μ = {true_mu}')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('μ', fontsize=12)
    ax.set_title('Posterior for Mean', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 오른쪽 위: σ²의 뒤확률 최빈값
    ax = axes[0, 1]
    var_modes = [h.variance_mode for h in history]
    ax.plot(n_vals, var_modes, 'g-', linewidth=2, marker='s', markersize=4, label='Mode[σ²|D]')
    if true_sigma2 is not None:
        ax.axhline(true_sigma2, color='red', linestyle='--', linewidth=2, label=f'True σ² = {true_sigma2}')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('σ²', fontsize=12)
    ax.set_title('Posterior for Variance', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 왼쪽 아래: 자유도
    ax = axes[1, 0]
    dfs = [h.degrees_of_freedom for h in history]
    ax.plot(n_vals, dfs, 'm-', linewidth=2, marker='d', markersize=4)
    ax.axhline(30, color='gray', linestyle=':', alpha=0.7, label='df=30 (≈Normal)')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('Degrees of Freedom', fontsize=12)
    ax.set_title('Student-t df (2αₙ)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 오른쪽 아래: κ과 α의 자람
    ax = axes[1, 1]
    kappas = [h.kappa for h in history]
    alphas = [h.alpha for h in history]
    ax.plot(n_vals, kappas, 'b-', linewidth=2, label='κₙ')
    ax.plot(n_vals, alphas, 'g-', linewidth=2, label='αₙ')
    ax.set_xlabel('Observations', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('NIG Parameter Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# 보여 주기
# =============================================================================

def demo_basic_inference():
    """흩어짐을 모를 때의 기본 추론을 보인다."""
    
    print("=" * 70)
    print("GAUSSIAN INFERENCE WITH UNKNOWN VARIANCE")
    print("=" * 70)
    
    true_mu, true_sigma2 = 5.0, 4.0
    np.random.seed(42)
    data = np.random.normal(true_mu, np.sqrt(true_sigma2), 20)
    
    print(f"\nTrue: μ = {true_mu}, σ² = {true_sigma2}")
    print(f"Data: n = {len(data)}, x̄ = {data.mean():.4f}, s² = {data.var(ddof=1):.4f}")
    
    model = GaussianUnknownVarianceModel(0.0, 0.1, 0.1, 0.1)
    model.update(data)
    
    print(f"\nPosterior NIG: {model.posterior_nig}")
    print(f"Marginal for μ: {model.posterior_mu}")
    print(f"Marginal for σ²: {model.posterior_variance}")

def demo_t_test_connection():
    """빈도주의 t검정과의 이음을 보인다."""
    
    print("\n" + "=" * 70)
    print("CONNECTION TO t-TEST")
    print("=" * 70)
    
    np.random.seed(456)
    data = np.random.normal(50, 10, 25)
    
    # 빈도주의
    x_bar, s = data.mean(), data.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=len(data)-1)
    freq_ci = (x_bar - t_crit * s/np.sqrt(len(data)), x_bar + t_crit * s/np.sqrt(len(data)))
    
    # 흐릿한 앞확률을 쓴 베이즈
    model = GaussianUnknownVarianceModel(0.0, 0.001, 0.001, 0.001)
    model.update(data)
    bayes_ci = model.posterior_mu.credible_interval(0.95)
    
    print(f"\nFrequentist 95% CI: [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}]")
    print(f"Bayesian 95% CI:    [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]")
    print(f"Difference: {abs(freq_ci[1] - bayes_ci[1]):.6f}")

if __name__ == "__main__":
    demo_basic_inference()
    demo_t_test_connection()
```

---

## 요약

| 갈래 | 식 |
|--------|---------|
| **앞확률** | $(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$ |
| **가능도** | $p(\mathcal{D} \mid \mu, \sigma^2) \propto (\sigma^2)^{-n/2} \exp\left(-\frac{(n-1)s^2 + n(\bar{x}-\mu)^2}{2\sigma^2}\right)$ |
| **뒤확률** | $(\mu, \sigma^2) \mid \mathcal{D} \sim \text{NIG}(\mu_n, \kappa_n, \alpha_n, \beta_n)$ |
| **$\mu$의 주변 분포** | $\mu \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \sqrt{\beta_n/(\alpha_n\kappa_n)}\right)$ |
| **$\sigma^2$의 주변 분포** | $\sigma^2 \mid \mathcal{D} \sim \text{Inv-Gamma}(\alpha_n, \beta_n)$ |
| **예측** | $x_{n+1} \mid \mathcal{D} \sim t_{2\alpha_n}\left(\mu_n, \sqrt{\beta_n(\kappa_n+1)/(\alpha_n\kappa_n)}\right)$ |

### 갱신 식

$$
\kappa_n = \kappa_0 + n, \quad \mu_n = \frac{\kappa_0\mu_0 + n\bar{x}}{\kappa_n}
$$

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \quad \beta_n = \beta_0 + \frac{(n-1)s^2}{2} + \frac{\kappa_0 n(\bar{x} - \mu_0)^2}{2\kappa_n}
$$

### 핵심 통찰

1. **결합 추론**: $\mu$과 $\sigma^2$을 함께 미루어 알아야 한다
2. **정규-역감마**: 두 매개변수의 켤레 앞확률
3. **스튜던트 t 주변 분포**: 흩어짐에 대한 아리송함이 꼬리를 두껍게 한다
4. **t 검정과의 이음**: 제프리스 앞확률이 빈도주의 구간을 낸다
5. **차례 갱신**: NIG족이 온라인 학습을 가능하게 한다
6. **점근 정규성**: $n \to \infty$이면 스튜던트 t가 가우스에 다가간다

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 흩어짐을 알 때 | 13장: 흩어짐을 아는 가우스 | 더 단순한 특별한 경우 |
| 베이즈 회귀 | 13장: 베이즈 선형 회귀 | 매개변수 여럿으로 넓히기 |
| 모형 견줌 | 13장: 모형 증거 | 주변 가능도 셈하기 |
| BNN 불확실성 | 13장: BNN 불확실성 | 앎의 불확실성과 우연의 불확실성 |
| 튼튼한 추론 | 8장: 튼튼한 방법 | 튼튼한 가능도로서의 스튜던트 t |

### 주요 참고 문헌

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). 3장.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. 4장.
- DeGroot, M. H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.
- Box, G. E. P., & Tiao, G. C. (1973). *Bayesian Inference in Statistical Analysis*.

## 연습문제

**연습문제 1.**
켤레 앞확률을 정의하고 보기 셋을 들어라.

??? success "연습문제 1 풀이"
    뒤확률이 앞확률과 같은 족에 들면 그 앞확률은 가능도의 켤레이다. 보기: (1) 베타-이항: $\text{Beta}(\alpha,\beta) + \text{Binomial} \to \text{Beta}(\alpha+k, \beta+n-k)$. (2) 정규-정규: $\mathcal{N}(\mu_0, \sigma_0^2) + \mathcal{N}(\mu, \sigma^2) \to \mathcal{N}(\mu_n, \sigma_n^2)$. (3) 감마-푸아송: $\text{Gamma}(\alpha,\beta) + \text{Poisson} \to \text{Gamma}(\alpha+\sum x_i, \beta+n)$.

---

**연습문제 2.**
흩어짐을 아는 정규 가능도와 정규 앞확률에서 뒤확률을 끌어내라.

??? success "연습문제 2 풀이"
    앞확률: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$. 데이터: $x_1,\ldots,x_n \sim \mathcal{N}(\mu, \sigma^2)$. 뒤확률: $\mu|x \sim \mathcal{N}(\mu_n, \sigma_n^2)$이며 $\sigma_n^2 = (1/\sigma_0^2 + n/\sigma^2)^{-1}$이고 $\mu_n = \sigma_n^2(\mu_0/\sigma_0^2 + n\bar{x}/\sigma^2)$이다. 뒤확률의 평균은 앞확률의 평균과 표본 평균을 정밀도로 무게 준 평균이다.

---

**연습문제 3.**
켤레 앞확률은 왜 셈에 편하면서도 때로 옥죄는가?

??? success "연습문제 3 풀이"
    편한 점: 닫힌 꼴의 뒤확률이 나오고 MCMC가 필요 없다. 옥죄는 점: 그 앞확률족이 실제 믿음을 담아내지 못할 수 있다. 이를테면 평균이 봉우리 둘이라고 믿는다면 정규 앞확률 하나로는 나타낼 수 없다. 켤레가 아닌 앞확률은 수치 방법(MCMC, 변분 추론)이 필요하지만 더 자유롭다.

---

**연습문제 4.**
앞확률의 '실효 표본 크기'라는 개념을 설명하라.

??? success "연습문제 4 풀이"
    앞확률 $\text{Beta}(\alpha, \beta)$의 실효 표본 크기는 $\alpha + \beta$이다. 곧 관찰 $\alpha + \beta$개만큼의 정보를 보탠다. 실제 관찰 $n$개를 얻은 뒤 뒤확률은 $\text{Beta}(\alpha+k, \beta+n-k)$이고 전체 실효 표본 크기는 $\alpha + \beta + n$이다. $n$이 커질수록 앞확률의 영향은 그만큼 줄어든다.
