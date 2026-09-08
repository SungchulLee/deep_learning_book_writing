# 켤레 앞확률

켤레 앞확률은 뒤확률이 앞확률과 같은 분포족에 머물게 하는 앞확률이다. 그 덕분에 적분 한 번 없이 매개변수 몇 개만 고쳐 쓰면 갱신이 끝난다. 이 마당은 켤레성이 왜 생기는지를 지수족으로 밝힌 뒤, 베타-이항·감마-푸아송·정규-정규·정규-역감마 네 모형에서 갱신 규칙을 손으로 끝까지 이끌어 낸다.

## 1. 켤레 앞확률이란 무엇인가

**정의 1.** [켤레족]

가능도족 $\{p(D \mid \theta)\}$ 에 대해 앞확률의 족 $\mathcal{F}$ 가 다음을 만족하면 $\mathcal{F}$ 를 그 가능도의 **켤레족**이라 한다.

$$
p(\theta) \in \mathcal{F} \;\Longrightarrow\; p(\theta \mid D) \in \mathcal{F}
$$

곧 갱신을 해도 분포의 **모양**은 그대로이고 매개변수만 바뀐다.

### 정리 1. 지수족의 켤레성 — 자연 매개변수에 대한 선형 갱신 { .thm }

가능도가 자연 매개변수 $\eta$ 와 충분 통계량 $T(x)$ 를 갖는 지수족

$$
p(x \mid \eta) = h(x)\,\exp\bigl(\eta^\top T(x) - A(\eta)\bigr)
$$

이라 하자. 그러면

$$
p(\eta \mid \nu, \chi) \;\propto\; \exp\bigl(\eta^\top \chi - \nu A(\eta)\bigr)
$$

꼴의 앞확률은 켤레이며, 관측 $x_1,\dots,x_n$ 을 본 뒤의 초매개변수는

$$
\chi_n = \chi + \sum_{i=1}^n T(x_i), \qquad \nu_n = \nu + n
$$

이다. 곧 갱신은 **충분 통계량을 더하고 관측 수를 세는 일**뿐이다.

??? proof "증명"

    비례 관계에 가능도와 앞확률을 넣으면

    $$
    p(\eta \mid D) \;\propto\; \Bigl[\textstyle\prod_i h(x_i)\Bigr]
    \exp\Bigl(\eta^\top \sum_i T(x_i) - nA(\eta)\Bigr)\cdot
    \exp\bigl(\eta^\top\chi - \nu A(\eta)\bigr)
    $$

    이다. $\prod_i h(x_i)$ 는 $\eta$ 에 달리지 않으므로 지워진다. 남은 지수를 모으면

    $$
    p(\eta\mid D) \;\propto\; \exp\Bigl(\eta^\top\bigl(\chi + \textstyle\sum_i T(x_i)\bigr) - (\nu+n)A(\eta)\Bigr)
    $$

    이고, 이는 초매개변수가 $(\chi_n, \nu_n)$ 인 같은 꼴의 분포이다.

!!! note "쓰임새"
    켤레성은 우연이 아니라 **지수족의 구조에서 저절로 나온다**. 아래 표의 짝들은 모두 이 정리의 특별한 경우이며, $\nu$ 는 "앞확률이 들고 있는 유사 관측의 수", $\chi$ 는 "그 유사 관측이 들고 있는 충분 통계량"으로 읽힌다.

| 가능도 | 켤레 앞확률 | 갱신되는 것 |
|--------|------------|------------|
| 베르누이 · 이항 | 베타 | 성공 횟수, 실패 횟수 |
| 푸아송 | 감마 | 사건 수의 합, 관측 시간 |
| 정규(흩어짐 앎) | 정규 | 정밀도, 정밀도로 무게 준 평균 |
| 정규(둘 다 모름) | 정규-역감마 | 위의 것에 제곱합을 더함 |
| 다항 | 디리클레 | 범주별 도수 |
| 지수 | 감마 | 관측 수, 값의 합 |

## 2. 베타-이항 모형

**정의 2.** [베타 분포]

$\alpha, \beta > 0$ 에 대해 $[0,1]$ 위의 밀도

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},
\qquad B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

를 갖는 분포를 $\text{Beta}(\alpha,\beta)$ 라 한다. 평균은 $\alpha/(\alpha+\beta)$, $\alpha,\beta>1$ 일 때 최빈값은 $(\alpha-1)/(\alpha+\beta-2)$ 이다.

### 정리 2. 베타-이항 켤레 갱신 — 성공과 실패를 세어 더한다 { .thm }

앞확률이 $\text{Beta}(\alpha,\beta)$ 이고 베르누이 시행 $n$ 번에서 성공 $k$ 번을 관측하면

$$
\theta \mid D \;\sim\; \text{Beta}(\alpha+k,\; \beta+n-k)
$$

이다. 뒤확률의 평균은 앞확률 평균과 최대 가능도의 무게 준 평균이며, 흩어짐은

$$
\operatorname{Var}[\theta \mid D] = \frac{\alpha_n\beta_n}{(\alpha_n+\beta_n)^2(\alpha_n+\beta_n+1)},
\qquad \alpha_n = \alpha+k,\ \ \beta_n=\beta+n-k
$$

이다.

??? proof "증명"

    베르누이는 $T(x) = x$, $\eta = \log\frac{\theta}{1-\theta}$ 인 지수족이므로 정리 1을 그대로 쓸 수 있다. 직접 셈해도 짧다.

    $$
    p(\theta\mid D) \propto \theta^{k}(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1}
    = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}
    $$

    이고 이는 $\text{Beta}(\alpha+k,\beta+n-k)$ 의 핵이다. 흩어짐 식은 베타 분포의 이차 적률에서 곧바로 나온다.

!!! note "쓰임새"
    $\alpha+\beta$ 가 앞확률의 **유사 관측 수**다. $n$ 이 이보다 훨씬 크면 앞확률의 자취는 거의 남지 않는다. 흩어짐 식의 분모에 $\alpha_n+\beta_n+1$ 이 있으므로 관측이 늘수록 뒤확률이 대략 $1/n$ 의 빠르기로 좁아진다.

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

**보기 1.** <span class="diff easy" title="쉬움"></span> 고른 앞확률 $\text{Beta}(1,1)$ 에서 동전을 $10$ 번 던져 앞면 $7$ 번을 보았다. 뒤확률과 그 평균·최빈값·표준편차를 구하시오.

??? success "풀이"

    정리 2에서 뒤확률은 $\text{Beta}(8, 4)$ 이다.

    $$
    \mathbb{E}[\theta\mid D] = \frac{8}{12} \approx 0.667,\qquad
    \text{최빈값} = \frac{7}{10} = 0.7
    $$

    $$
    \operatorname{Var} = \frac{8\cdot 4}{12^2\cdot 13} \approx 0.0171
    \;\Longrightarrow\; \text{표준편차} \approx 0.131
    $$

    최빈값이 최대 가능도 $0.7$ 과 같은 것은 고른 앞확률이 $\alpha=\beta=1$ 이어서 지수에 아무것도 더하지 않기 때문이다.

**문제 1.** <span class="diff med" title="중간"></span> 같은 데이터에 $\text{Beta}(10,10)$ 앞확률을 두면 뒤확률의 표준편차가 어떻게 달라지는가? 앞확률이 세면 왜 뒤확률이 좁아지는지 설명하시오.

??? success "풀이"

    뒤확률은 $\text{Beta}(17, 13)$ 이고

    $$
    \operatorname{Var} = \frac{17\cdot 13}{30^2\cdot 31}\approx 0.00792
    \;\Longrightarrow\;\text{표준편차}\approx 0.089
    $$

    로 $0.131$ 에서 $0.089$ 로 좁아진다. 앞확률이 유사 관측 $20$ 번을 들고 오므로 실제로는 $30$ 번을 본 셈이고, 흩어짐 분모의 $\alpha_n+\beta_n+1$ 이 $13$ 에서 $31$ 로 커졌기 때문이다.

## 3. 뒤확률 예측 분포와 이어짐 규칙

뒤확률 자체보다 "다음에 무엇이 나올까"가 궁금할 때가 많다. 이때는 뒤확률로 가능도를 평균 낸다.

### 정리 3. 라플라스의 이어짐 규칙 — 다음 관측의 확률 { .thm }

베타-이항 모형에서 $n$ 번 중 $k$ 번 성공을 본 뒤, 다음 시행이 성공일 확률은 뒤확률의 평균과 같다.

$$
P(x_{n+1}=1 \mid D) = \int_0^1 \theta\,p(\theta\mid D)\,d\theta = \frac{\alpha+k}{\alpha+\beta+n}
$$

특히 고른 앞확률 $\alpha=\beta=1$ 에서는

$$
P(x_{n+1}=1\mid D) = \frac{k+1}{n+2}
$$

이며, 이를 **라플라스의 이어짐 규칙**이라 한다.

??? proof "증명"

    예측 확률은 가능도를 뒤확률로 평균 낸 것이다.

    $$
    P(x_{n+1}=1\mid D) = \int_0^1 P(x_{n+1}=1\mid\theta)\,p(\theta\mid D)\,d\theta
    = \int_0^1 \theta\,p(\theta\mid D)\,d\theta
    $$

    이는 곧 뒤확률의 평균이고, 정리 2에서 뒤확률이 $\text{Beta}(\alpha+k,\beta+n-k)$ 이므로 그 평균은 $\dfrac{\alpha+k}{\alpha+\beta+n}$ 이다. $\alpha=\beta=1$ 을 넣으면 $(k+1)/(n+2)$ 이다.

!!! note "쓰임새"
    최대 가능도는 $k=0$ 일 때 확률을 $0$ 으로 못 박아, 한 번도 못 본 일은 **결코 일어나지 않는다**고 말한다. 이어짐 규칙은 $1/(n+2)$ 를 남겨 이 과신을 막는다. 언어 모형의 가산 매끄럽게 하기가 바로 이 셈이다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 새 백신을 $20$ 명에게 놓아 부작용이 한 번도 없었다. 다음 사람에게 부작용이 날 확률을 고른 앞확률로 어림하시오.

??? success "풀이"

    $n=20$, $k=0$ 이므로 부작용이 날 확률은

    $$
    1 - \frac{0+1}{20+2} = \frac{21}{22} \;\text{의 여사건},\qquad
    P(\text{부작용}) = \frac{1}{22} \approx 0.045
    $$

    이다. 최대 가능도라면 $0$ 이라고 답했을 자리에 약 $4.5\%$ 를 남긴다. 표본이 $20$ 명뿐이라는 사실을 정직하게 반영한 값이다.

## 4. 감마-푸아송 모형

### 정리 4. 감마-푸아송 켤레 갱신 — 예측 분포는 음이항 { .thm }

앞확률이 $\lambda \sim \text{Gamma}(a, b)$ (비율 매개변수 꼴)이고 $x_1,\dots,x_n \sim \text{Poisson}(\lambda)$ 이면

$$
\lambda \mid D \;\sim\; \text{Gamma}\Bigl(a + \sum_i x_i,\ \ b + n\Bigr)
$$

이고, 뒤확률 예측 분포는 음이항

$$
p(x_{n+1} \mid D) = \binom{x_{n+1}+a_n-1}{x_{n+1}}
\Bigl(\frac{b_n}{b_n+1}\Bigr)^{a_n}\Bigl(\frac{1}{b_n+1}\Bigr)^{x_{n+1}}
$$

이다. 여기서 $a_n = a+\sum_i x_i$, $b_n = b+n$ 이다.

??? proof "증명"

    가능도는 $\prod_i e^{-\lambda}\lambda^{x_i}/x_i! \propto \lambda^{\sum x_i}e^{-n\lambda}$ 이고 앞확률은 $\propto \lambda^{a-1}e^{-b\lambda}$ 이다. 곱하면

    $$
    p(\lambda\mid D) \propto \lambda^{a+\sum x_i-1}e^{-(b+n)\lambda}
    $$

    으로 $\text{Gamma}(a_n, b_n)$ 이다.

    예측 분포는 $\lambda$ 를 적분해 없앤 것이다.

    $$
    p(x'\mid D) = \int_0^\infty \frac{e^{-\lambda}\lambda^{x'}}{x'!}\cdot
    \frac{b_n^{a_n}}{\Gamma(a_n)}\lambda^{a_n-1}e^{-b_n\lambda}\,d\lambda
    $$

    적분 안을 $\lambda^{a_n+x'-1}e^{-(b_n+1)\lambda}$ 로 모으고 감마 적분 $\int_0^\infty \lambda^{s-1}e^{-r\lambda}d\lambda = \Gamma(s)/r^s$ 를 쓰면

    $$
    p(x'\mid D) = \frac{\Gamma(a_n+x')}{x'!\,\Gamma(a_n)}\cdot
    \frac{b_n^{a_n}}{(b_n+1)^{a_n+x'}}
    $$

    이고, 이는 위에 적은 음이항이다.

!!! note "쓰임새"
    푸아송은 평균과 흩어짐이 같아야 하는데 실제 개수 데이터는 흩어짐이 더 큰 일이 흔하다. 음이항 예측 분포는 $\lambda$ 의 불확실성까지 흩어짐에 얹으므로 이 **과대 흩어짐**을 저절로 담아낸다.

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

**보기 3.** <span class="diff easy" title="쉬움"></span> 누리집 방문이 하루 평균 몇 번인지 알고자 한다. 앞확률 $\text{Gamma}(2, 1)$ 에 $5$ 일 동안 $3, 7, 4, 6, 5$ 번을 관측했다. 뒤확률을 구하시오.

??? success "풀이"

    $\sum x_i = 25$, $n = 5$ 이므로 정리 4에서

    $$
    \lambda \mid D \sim \text{Gamma}(2+25,\ 1+5) = \text{Gamma}(27, 6)
    $$

    이다. 평균은 $27/6 = 4.5$, 표준편차는 $\sqrt{27}/6 \approx 0.87$ 이다. 표본 평균 $5$ 가 앞확률 평균 $2$ 쪽으로 조금 끌려갔다.

## 5. 정규-정규 모형(흩어짐을 알 때)

### 정리 5. 정밀도는 더해진다 — 뒤확률 평균은 정밀도로 무게 준 평균 { .thm }

$x_1,\dots,x_n\sim\mathcal{N}(\mu,\sigma^2)$ 에서 $\sigma$ 를 알고 앞확률이 $\mu\sim\mathcal{N}(\mu_0,\sigma_0^2)$ 이라 하자. 정밀도를 $\tau_0 = 1/\sigma_0^2$, $\tau = 1/\sigma^2$ 로 두면

$$
\mu \mid D \;\sim\; \mathcal{N}(\mu_n,\ 1/\tau_n),
\qquad
\tau_n = \tau_0 + n\tau,
\qquad
\mu_n = \frac{\tau_0\mu_0 + n\tau\bar x}{\tau_n}
$$

이다.

??? proof "증명"

    로그를 잡고 $\mu$ 와 얽힌 항만 남기면

    $$
    -\frac{\tau_0}{2}(\mu-\mu_0)^2 - \frac{n\tau}{2}(\mu-\bar x)^2
    $$

    이다. $\mu$ 에 대해 펼쳐 모으면 이차항의 계수가 $-(\tau_0+n\tau)/2$, 일차항의 계수가 $\tau_0\mu_0+n\tau\bar x$ 이다. 완전제곱으로 묶으면

    $$
    -\frac{\tau_n}{2}\Bigl(\mu - \frac{\tau_0\mu_0+n\tau\bar x}{\tau_n}\Bigr)^2 + \text{const}
    $$

    이 되어 결론의 정규와 같다.

!!! note "쓰임새"
    정밀도로 보면 베이즈 갱신이 **정보를 더하는 일**이다. 앞확률이 $\tau_0$, 데이터가 $n\tau$ 만큼 정보를 들고 오고 뒤확률은 그 합을 갖는다. 표준편차로 적으면 이 단순함이 가려진다.

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

**보기 4.** <span class="diff easy" title="쉬움"></span> 측정 잡음이 $\sigma = 2$ 임을 알고 앞확률이 $\mathcal{N}(10, 5^2)$ 이다. $n=9$ 개를 재어 평균 $\bar x = 12$ 를 얻었다. 뒤확률을 구하시오.

??? success "풀이"

    $\tau_0 = 1/25 = 0.04$, $\tau = 1/4 = 0.25$ 이므로

    $$
    \tau_9 = 0.04 + 9(0.25) = 2.29,\qquad
    \mu_9 = \frac{0.04(10) + 2.25(12)}{2.29} \approx 11.965
    $$

    이다. 뒤확률은 $\mathcal{N}(11.965,\ 1/2.29)$ 이고 표준편차는 약 $0.661$ 이다. 앞확률이 넓어서($\sigma_0=5$) 거의 데이터가 답을 정했다.

## 6. 정규-역감마 모형(둘 다 모를 때)

**정의 3.** [정규-역감마 앞확률]

평균과 흩어짐을 모두 모를 때의 켤레 앞확률은

$$
\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0),
\qquad
\mu \mid \sigma^2 \sim \mathcal{N}\bigl(\mu_0,\ \sigma^2/\kappa_0\bigr)
$$

이다. 이를 $\text{NIG}(\mu_0,\kappa_0,a_0,b_0)$ 로 적는다. $\mu$ 의 앞확률 흩어짐이 $\sigma^2$ 에 매여 있다는 점이 핵심이다.

### 정리 6. 정규-역감마 켤레 갱신 — 평균과 흩어짐을 한꺼번에 고친다 { .thm }

$x_1,\dots,x_n\sim\mathcal{N}(\mu,\sigma^2)$ 이고 앞확률이 $\text{NIG}(\mu_0,\kappa_0,a_0,b_0)$ 이면 뒤확률도 정규-역감마이며 매개변수는 다음과 같다.

$$
\kappa_n = \kappa_0+n, \qquad
\mu_n = \frac{\kappa_0\mu_0 + n\bar x}{\kappa_n}, \qquad
a_n = a_0 + \frac n2
$$

$$
b_n = b_0 + \frac12\sum_i(x_i-\bar x)^2 + \frac{\kappa_0 n(\bar x-\mu_0)^2}{2\kappa_n}
$$

??? proof "증명"

    가능도의 지수를 평균 언저리로 갈라 적으면

    $$
    \sum_i (x_i-\mu)^2 = \sum_i (x_i-\bar x)^2 + n(\bar x-\mu)^2
    $$

    이다. 앞확률의 $\mu$ 부분과 합치면 $\mu$ 에 대한 이차식

    $$
    \kappa_0(\mu-\mu_0)^2 + n(\mu-\bar x)^2
    = \kappa_n\bigl(\mu-\mu_n\bigr)^2 + \frac{\kappa_0 n(\bar x-\mu_0)^2}{\kappa_n}
    $$

    을 얻는다. 이는 완전제곱 묶기이며, 남은 상수항이 $b_n$ 의 마지막 항이다. $\mu$ 부분을 떼어 내면 $\sigma^2$ 에 대해 역감마 꼴이 남고 그 매개변수가 $(a_n, b_n)$ 이다.

!!! note "쓰임새"
    $b_n$ 의 세 항이 각각 **앞확률이 들고 온 흩어짐**, **데이터 안의 흩어짐**, **앞확률 평균과 표본 평균의 어긋남**이다. 셋째 항 때문에 앞확률이 데이터와 멀리 어긋나면 흩어짐 어림값이 커진다. 모형이 스스로 "뭔가 안 맞는다"고 말하는 셈이다.

## 7. 주변 뒤확률과 스튜던트 t

흩어짐을 모르면 평균에 대한 뒤확률은 더 이상 정규가 아니다.

### 정리 7. 평균의 주변 뒤확률 — 스튜던트 t가 나온다 { .thm }

정리 6의 뒤확률에서 $\sigma^2$ 를 적분해 없애면

$$
\mu \mid D \;\sim\; t_{2a_n}\Bigl(\mu_n,\ \frac{b_n}{a_n\kappa_n}\Bigr)
$$

이다. 곧 자유도 $2a_n$, 중심 $\mu_n$, 자 매개변수 $b_n/(a_n\kappa_n)$ 인 스튜던트 t 분포이다.

??? proof "증명"

    결합 뒤확률은

    $$
    p(\mu,\sigma^2\mid D) \propto (\sigma^2)^{-a_n-3/2}
    \exp\Bigl(-\frac{2b_n + \kappa_n(\mu-\mu_n)^2}{2\sigma^2}\Bigr)
    $$

    이다. $\sigma^2$ 에 대해 적분하면서 $s = 1/\sigma^2$ 로 바꾸면 감마 적분이 되어

    $$
    p(\mu\mid D) \propto \bigl(2b_n + \kappa_n(\mu-\mu_n)^2\bigr)^{-(a_n+1/2)}
    \propto \Bigl(1 + \frac{\kappa_n(\mu-\mu_n)^2}{2b_n}\Bigr)^{-\frac{2a_n+1}{2}}
    $$

    를 얻는다. 이는 자유도 $\nu = 2a_n$ 인 t 밀도 $\bigl(1+\frac{(\mu-m)^2}{\nu s^2}\bigr)^{-(\nu+1)/2}$ 의 꼴이며, 견주어 보면 $s^2 = b_n/(a_n\kappa_n)$ 이다.

!!! note "쓰임새"
    흩어짐을 모른다는 사실이 꼬리를 두껍게 만든다. 빈도주의에서 $\sigma$ 를 모를 때 $z$ 대신 $t$ 를 쓰는 것과 정확히 같은 까닭이며, $a_n = a_0 + n/2$ 이므로 $n$ 이 커지면 자유도가 커져 정규로 돌아간다.

**문제 2.** <span class="diff hard" title="어려움"></span> $a_0 \to 0$, $b_0\to 0$, $\kappa_0\to 0$ 으로 보내면 정리 7의 주변 뒤확률이 무엇이 되는지 구하고, 빈도주의 t 검정과 견주시오.

??? success "풀이"

    $\kappa_n \to n$, $\mu_n\to\bar x$, $a_n\to n/2$, $b_n\to \frac12\sum_i(x_i-\bar x)^2$ 이다. 자 매개변수는

    $$
    \frac{b_n}{a_n\kappa_n} \to \frac{\frac12\sum(x_i-\bar x)^2}{\frac n2\cdot n}
    = \frac{\sum(x_i-\bar x)^2}{n^2}
    $$

    이고 자유도는 $2a_n \to n$ 이다. 표본 흩어짐을 $s^2 = \sum(x_i-\bar x)^2/(n-1)$ 로 두면 이는 대략 $\bar x \pm t\cdot s/\sqrt n$ 꼴의 구간을 준다.

    빈도주의 t 구간은 자유도가 $n-1$ 인데 여기서는 $n$ 이다. 자유도 하나 차이는 이 변칙 앞확률이 정확히 $p(\mu,\sigma^2)\propto 1/\sigma^2$ 가 아니라 $1/\sigma^3$ 에 해당하기 때문이며, $p(\mu,\sigma^2)\propto 1/\sigma^2$ 를 쓰면 자유도가 정확히 $n-1$ 로 맞는다.

## 8. 특별한 경우와 점근 거동

### 정리 8. 앞확률의 씻김 — 데이터가 쌓이면 켤레 앞확률의 자취가 사라진다 { .thm }

베타-이항 모형에서 참값이 $\theta^\star \in (0,1)$ 이라 하자. 앞확률 $(\alpha,\beta)$ 를 붙박아 두고 $n\to\infty$ 로 보내면

$$
\mathbb{E}[\theta\mid D] \longrightarrow \theta^\star,
\qquad
\operatorname{Var}[\theta\mid D] = O\!\left(\frac1n\right)
$$

이며, 이는 $(\alpha,\beta)$ 를 어떻게 잡든 성립한다.

??? proof "증명"

    큰 수의 법칙에서 $k/n \to \theta^\star$ 이다. 정리 2의 평균을 다시 적으면

    $$
    \mathbb{E}[\theta\mid D] = \frac{\alpha+k}{\alpha+\beta+n}
    = \frac{\alpha/n + k/n}{(\alpha+\beta)/n + 1} \longrightarrow \frac{0+\theta^\star}{0+1} = \theta^\star
    $$

    이다. 흩어짐은 $\alpha_n+\beta_n = \alpha+\beta+n$ 이므로

    $$
    \operatorname{Var} = \frac{\alpha_n\beta_n}{(\alpha_n+\beta_n)^2(\alpha_n+\beta_n+1)}
    \le \frac{1}{4(\alpha+\beta+n+1)} = O(1/n)
    $$

    이다. 가운데 부등식은 $\alpha_n\beta_n \le \bigl(\frac{\alpha_n+\beta_n}{2}\bigr)^2$ 에서 나온다.

!!! note "쓰임새"
    앞확률이 씻겨 나가는 것은 $\theta^\star$ 언저리에 앞확률이 **양의 밀도를 줄 때만** 참이다. 홀데인 앞확률 $\text{Beta}(0,0)$ 처럼 변칙 앞확률을 쓰면 $k=0$ 이나 $k=n$ 에서 뒤확률이 아예 정의되지 않는 일이 생긴다.

**보기 5.** <span class="diff med" title="중간"></span> 고른·제프리스·홀데인 세 앞확률을 견주고, 각각이 무엇을 뜻하는지 밝히시오.

??? success "풀이"

    | 앞확률 | $(\alpha,\beta)$ | 유사 관측 | 뒤확률 평균 |
    |--------|------------------|-----------|-------------|
    | 고른 | $(1,1)$ | $2$ | $(k+1)/(n+2)$ |
    | 제프리스 | $(0.5,0.5)$ | $1$ | $(k+0.5)/(n+1)$ |
    | 홀데인 | $(0,0)$ | $0$ | $k/n$ |

    홀데인 앞확률은 유사 관측을 하나도 더하지 않아 뒤확률 평균이 최대 가능도와 같다. 그러나 $\text{Beta}(0,0)$ 은 적분할 수 없는 변칙 앞확률이고, $k=0$ 이면 뒤확률 $\text{Beta}(0,n)$ 도 변칙이라 쓸 수 없다. 제프리스가 그 사이에서 가장 무난한 절충이다.

**문제 3.** <span class="diff med" title="중간"></span> 앞확률 매개변수를 실제로 어떻게 고를지, 두 가지 길을 들어 설명하시오.

??? success "풀이"

    첫째는 **적률 맞추기**다. 앞선 지식으로 평균 $m$ 과 표준편차 $s$ 를 정한 뒤 베타의 적률 식을 풀어 $(\alpha,\beta)$ 를 얻는다.

    $$
    \alpha+\beta = \frac{m(1-m)}{s^2}-1,\qquad \alpha = m(\alpha+\beta)
    $$

    둘째는 **유사 관측으로 생각하기**다. "지금까지 본 것이 대략 $N$ 번쯤이고 그중 성공이 $m$ 몫이었다"고 여겨 $\alpha = mN$, $\beta=(1-m)N$ 으로 둔다. 정리 1이 보인 대로 $\nu$ 가 관측 수의 자리를 차지하므로 이 해석이 늘 통한다.

## 9. 온전한 짜보기

앞의 마당에서 이끌어 낸 갱신 규칙을 갈래로 묶은 것이다. 차례대로 베타-이항, 정규-정규, 정규-역감마이다.

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

## 연습문제

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 디리클레-다항이 켤레 짝임을 정리 1로 설명하고 갱신 규칙을 적으시오.

??? success "풀이"

    다항 가능도의 충분 통계량은 범주별 도수 $(n_1,\dots,n_K)$ 이다. 정리 1에 따라 켤레 앞확률은 $\exp(\eta^\top\chi - \nu A(\eta))$ 꼴이며 이것이 디리클레 $\text{Dir}(\alpha_1,\dots,\alpha_K)$ 이다.

    갱신은 도수를 더하는 일뿐이다.

    $$
    \text{Dir}(\alpha_1+n_1,\ \dots,\ \alpha_K+n_K)
    $$

    $K=2$ 이면 베타-이항으로 돌아간다.

**연습문제 2.** <span class="diff med" title="중간"></span> 지수 가능도의 켤레 앞확률이 감마임을 보이고 갱신 규칙을 이끌어 내시오.

??? success "풀이"

    $x_i\sim\text{Exp}(\lambda)$ 이면 가능도는 $\lambda^n e^{-\lambda\sum x_i}$ 이다. 앞확률을 $\text{Gamma}(a,b)\propto \lambda^{a-1}e^{-b\lambda}$ 로 두면

    $$
    p(\lambda\mid D)\propto \lambda^{a+n-1}e^{-(b+\sum x_i)\lambda}
    $$

    으로 $\text{Gamma}(a+n,\ b+\sum_i x_i)$ 이다. 충분 통계량이 $(n, \sum x_i)$ 이고 이 둘이 각각 더해지는 것이 정리 1의 결론과 맞는다.

**연습문제 3.** <span class="diff med" title="중간"></span> 정리 3의 이어짐 규칙을 언어 모형의 가산 매끄럽게 하기와 이어 설명하시오.

??? success "풀이"

    어휘 크기가 $V$ 인 다항 모형에서 각 낱말에 $\text{Dir}(\alpha,\dots,\alpha)$ 앞확률을 두면 예측 확률은

    $$
    P(w \mid D) = \frac{n_w + \alpha}{N + V\alpha}
    $$

    이다. $\alpha=1$ 이 라플라스 매끄럽게 하기, $\alpha<1$ 이 리드스톤 매끄럽게 하기다.

    한 번도 안 나온 낱말에 $0$ 이 아닌 확률을 주는 것이 핵심이며, 이는 연습문제 1의 디리클레 갱신에 정리 3의 논증을 그대로 얹은 것이다.

**연습문제 4.** <span class="diff hard" title="어려움"></span> 정리 7의 t 뒤확률에서 $n\to\infty$ 일 때 정규로 다가감을 보이시오.

??? success "풀이"

    자유도가 $2a_n = 2a_0+n \to\infty$ 이다. 스튜던트 t의 밀도에서

    $$
    \Bigl(1+\frac{z^2}{\nu}\Bigr)^{-\frac{\nu+1}{2}}
    = \exp\Bigl(-\frac{\nu+1}{2}\log\bigl(1+\tfrac{z^2}{\nu}\bigr)\Bigr)
    \longrightarrow e^{-z^2/2}
    $$

    이다. $\log(1+u)\approx u$ 를 썼다. 곧 t는 정규로 다가간다.

    자 매개변수도 $b_n/(a_n\kappa_n) \approx s^2/n$ 으로 가므로, 뒤확률은 $\mathcal{N}(\bar x,\ s^2/n)$ 에 다가간다. 흩어짐을 모른다는 사실의 값이 $n$ 이 커지면 사라지는 것이다.

**연습문제 5.** <span class="diff hard" title="어려움"></span> 앞확률 민감도 분석을 켤레 모형에서 어떻게 하면 값싸게 할 수 있는지 밝히시오.

??? success "풀이"

    켤레 모형에서는 앞확률을 바꾸어도 **다시 셈할 것이 없다**. 정리 2나 정리 6의 갱신 식이 닫힌 꼴이므로 $(\alpha,\beta)$ 나 $(\mu_0,\kappa_0,a_0,b_0)$ 를 격자로 훑으며 뒤확률 요약값을 바로 뽑을 수 있다.

    이는 MCMC를 다시 돌려야 하는 켤레 아닌 모형과 크게 다른 점이다. 켤레 모형을 먼저 세워 민감도를 살핀 뒤 더 복잡한 모형으로 옮겨 가는 것이 실전에서 값싼 길이다.

## 정리하며

켤레 앞확률은 편의를 위한 요령이 아니라 **지수족의 구조가 낳는 성질**이다.

1. 가능도가 지수족이면 자연 매개변수에 대한 켤레 앞확률이 있고, 갱신은 충분 통계량을 더하고 관측 수를 세는 일로 끝난다(정리 1).
2. 베타-이항에서 앞확률은 성공·실패의 유사 관측으로 더해진다(정리 2). 예측 확률은 뒤확률 평균이며 이것이 라플라스의 이어짐 규칙이다(정리 3).
3. 감마-푸아송의 예측 분포는 음이항이라 과대 흩어짐을 저절로 담는다(정리 4).
4. 정규-정규에서는 정밀도가 더해지고 평균은 정밀도로 무게 준 평균이 된다(정리 5).
5. 흩어짐까지 모르면 정규-역감마가 켤레이고(정리 6), 평균의 주변 뒤확률은 스튜던트 t가 된다(정리 7).
6. 참값 언저리에 앞확률이 양의 밀도를 주면 데이터가 쌓일수록 앞확률의 자취는 씻겨 나간다(정리 8).

켤레성이 깨지는 모형에서는 이 모든 닫힌 꼴을 잃는다. 그때는 「[깁스 표집](../../ch15/mcmc/gibbs_sampling.md)」처럼 표본을 뽑는 길로 가야 한다. 여기서 얻은 뒤확률을 점이나 구간으로 줄이려면 「[최대 뒤확률 어림](map_estimation.md)」과 「[믿음 구간](credible_intervals.md)」을 보라.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2~3장
- Bishop, C. *Pattern Recognition and Machine Learning*, 2장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3~5장
- Diaconis, P. & Ylvisaker, D. "Conjugate Priors for Exponential Families." *Annals of Statistics*, 1979
