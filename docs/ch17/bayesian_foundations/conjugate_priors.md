# 켤레 사전분포

켤레 사전분포를 사용하면 사후분포가 같은 분포족에 속하므로 하이퍼파라미터의 갱신만으로 추론을 정리할 수 있다. 지수족에서 이 구조가 나타나는 이유를 설명하고 베타-이항, 감마-푸아송, 정규-정규, 정규-역감마 모델의 갱신과 예측분포를 유도한다. 분포의 매개변수화와 적절성 조건을 함께 확인한다.

## 1. 켤레 사전분포란 무엇인가

**정의 1.** [켤레족]

우도족 $\{p(D \mid \theta)\}$ 에 대해 사전분포의 족 $\mathcal{F}$ 가 다음을 만족하면 $\mathcal{F}$ 를 그 우도의 **켤레족**이라 한다.

$$
p(\theta) \in \mathcal{F} \;\Longrightarrow\; p(\theta \mid D) \in \mathcal{F}
$$

갱신 후에도 같은 분포족에 속한다는 뜻이다. 매개변수가 달라지므로 집중도나 비대칭성 등 실제 분포의 모양은 달라질 수 있다.

### 정리 1. 지수족의 켤레성 — 자연 매개변수에 대한 선형 갱신 { .thm }

우도가 자연 매개변수 $\eta$ 와 충분 통계량 $T(x)$ 를 갖는 지수족

$$
p(x \mid \eta) = h(x)\,\exp\bigl(\eta^\top T(x) - A(\eta)\bigr)
$$

이라 하자. 그러면

$$
p(\eta \mid \nu, \chi) \;\propto\; \exp\bigl(\eta^\top \chi - \nu A(\eta)\bigr)
$$

꼴의 사전분포가 정규화 가능하고 관측들이 η를 조건으로 독립이면 켤레 갱신을 얻는다. 이 식은 자연 매개변수 공간의 고정된 기준측도에 대한 밀도다. 관측 $x_1,\dots,x_n$ 뒤의 하이퍼파라미터는

$$
\chi_n = \chi + \sum_{i=1}^n T(x_i), \qquad \nu_n = \nu + n
$$

이다. 곧 갱신은 **충분 통계량을 더하고 관측 수를 세는 일**뿐이다.

??? proof "증명"

    비례 관계에 우도와 사전분포를 넣으면

    $$
    p(\eta \mid D) \;\propto\; \Bigl[\textstyle\prod_i h(x_i)\Bigr]
    \exp\Bigl(\eta^\top \sum_i T(x_i) - nA(\eta)\Bigr)\cdot
    \exp\bigl(\eta^\top\chi - \nu A(\eta)\bigr)
    $$

    이다. $\prod_i h(x_i)$ 는 $\eta$ 에 달리지 않으므로 지워진다. 남은 지수를 모으면

    $$
    p(\eta\mid D) \;\propto\; \exp\Bigl(\eta^\top\bigl(\chi + \textstyle\sum_i T(x_i)\bigr) - (\nu+n)A(\eta)\Bigr)
    $$

    이고, 이는 하이퍼파라미터가 $(\chi_n, \nu_n)$ 인 같은 꼴의 분포이다.

!!! note "쓰임새"
    지수족의 우도와 위 사전분포는 자연 매개변수에 대한 지수가 같은 형태이므로 충분통계량을 더해 갱신된다. ν·χ는 사전 정보의 크기와 중심을 표현하지만 실제 관측 수와 일대일로 같지는 않다. 다른 매개변수 좌표에서 밀도를 적을 때는 야코비안을 포함해야 하며, 허용 가능한 하이퍼파라미터와 정규화 가능성도 확인해야 한다.

| 우도 | 켤레 사전분포 | 갱신되는 것 |
|--------|------------|------------|
| 베르누이 · 이항 | 베타 | 성공 횟수, 실패 횟수 |
| 푸아송 | 감마 | 사건 수의 합, 관측 시간 |
| 정규(분산 앎) | 정규 | 정밀도, 정밀도로 가중 평균 |
| 정규(둘 다 모름) | 정규-역감마 | 위의 것에 제곱합을 더함 |
| 다항 | 디리클레 | 범주별 도수 |
| 지수 | 감마 | 관측 수, 값의 합 |

## 2. 베타-이항 모델

**정의 2.** [베타 분포]

$\alpha, \beta > 0$ 에 대해 $[0,1]$ 위의 밀도

$$
p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},
\qquad B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$

를 갖는 분포를 $\text{Beta}(\alpha,\beta)$ 라 한다. 평균은 $\alpha/(\alpha+\beta)$, $\alpha,\beta>1$ 일 때 최빈값은 $(\alpha-1)/(\alpha+\beta-2)$ 이다.

### 정리 2. 베타-이항 켤레 갱신 — 성공과 실패를 세어 더한다 { .thm }

사전분포가 $\text{Beta}(\alpha,\beta)$ 이고 베르누이 시행 $n$ 번에서 성공 $k$ 번을 관측하면

$$
\theta \mid D \;\sim\; \text{Beta}(\alpha+k,\; \beta+n-k)
$$

이다. 사후분포의 평균은 사전평균과 최대 우도의 가중 평균이며, 분산은

$$
\operatorname{Var}[\theta \mid D] = \frac{\alpha_n\beta_n}{(\alpha_n+\beta_n)^2(\alpha_n+\beta_n+1)},
\qquad \alpha_n = \alpha+k,\ \ \beta_n=\beta+n-k
$$

이다.

??? proof "증명"

    베르누이는 T(x)=x, η=log(θ/(1−θ))인 지수족이다. 정리 1을 η 밀도에 적용한 뒤 θ 밀도로 변환할 때는 야코비안을 반영해야 한다. 여기서는 θ 좌표에서 직접 곱해 유도한다.

    $$
    p(\theta\mid D) \propto \theta^{k}(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1}
    = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}
    $$

    이고 이는 $\text{Beta}(\alpha+k,\beta+n-k)$ 의 커널이다. 분산 식은 베타 분포의 이차 적률에서 곧바로 나온다.

!!! note "쓰임새"
    α+β는 사후평균 가중치에서 사전분포의 유효 표본 크기로 해석할 수 있다. 실제 성공·실패 관측 수와 동일하다는 뜻은 아니다. 고정된 양의 사전 매개변수와 내부 참값을 가정하면 사후분산은 대략 1/n, 사후 표준편차는 1/√n 크기로 감소한다. 분산 감소가 모든 관측 단계에서 단조롭다는 보장은 없다.

??? note "코드 1 (43줄)"

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


**보기 1.** <span class="diff easy" title="쉬움"></span> 균등 사전분포 $\text{Beta}(1,1)$ 에서 동전을 $10$ 번 던져 앞면 $7$ 번을 보았다. 사후분포과 그 평균·최빈값·표준편차를 구하시오.

??? success "풀이"

    정리 2에서 사후분포는 $\text{Beta}(8, 4)$ 이다.

    $$
    \mathbb{E}[\theta\mid D] = \frac{8}{12} \approx 0.667,\qquad
    \text{최빈값} = \frac{7}{10} = 0.7
    $$

    $$
    \operatorname{Var} = \frac{8\cdot 4}{12^2\cdot 13} \approx 0.0171
    \;\Longrightarrow\; \text{표준편차} \approx 0.131
    $$

    최빈값이 최대 우도 $0.7$ 과 같은 것은 균등 사전분포가 $\alpha=\beta=1$ 이어서 지수에 아무것도 더하지 않기 때문이다.

**문제 1.** <span class="diff med" title="중간"></span> 같은 데이터에 $\text{Beta}(10,10)$ 사전분포를 두면 사후분포의 표준편차가 어떻게 달라지는가? 사전분포가 세면 왜 사후분포가 좁아지는지 설명하시오.

??? success "풀이"

    사후분포는 $\text{Beta}(17, 13)$ 이고

    $$
    \operatorname{Var} = \frac{17\cdot 13}{30^2\cdot 31}\approx 0.00792
    \;\Longrightarrow\;\text{표준편차}\approx 0.089
    $$

    이 예에서는 표준편차가 약 0.131에서 0.089로 감소한다. 총 집중도가 증가하고 사후평균도 바뀐 결과다. α+β만 커지면 어떤 사전 중심과 관측값에서도 사후분산이 반드시 감소한다고 일반화하지는 않는다.

## 3. 사후예측분포와 연속법칙

사후분포 자체보다 "다음에 무엇이 나올까"가 궁금할 때가 많다. 이때는 사후분포로 우도를 평균 낸다.

### 정리 3. 라플라스의 연속법칙 — 다음 관측의 확률 { .thm }

베타-이항 모델에서 $n$ 번 중 $k$ 번 성공을 본 뒤, 다음 시행이 성공일 확률은 사후분포의 평균과 같다.

$$
P(x_{n+1}=1 \mid D) = \int_0^1 \theta\,p(\theta\mid D)\,d\theta = \frac{\alpha+k}{\alpha+\beta+n}
$$

특히 균등 사전분포 $\alpha=\beta=1$ 에서는

$$
P(x_{n+1}=1\mid D) = \frac{k+1}{n+2}
$$

이며, 이를 **라플라스의 연속법칙**이라 한다.

??? proof "증명"

    예측 확률은 우도를 사후분포로 평균 낸 것이다.

    $$
    P(x_{n+1}=1\mid D) = \int_0^1 P(x_{n+1}=1\mid\theta)\,p(\theta\mid D)\,d\theta
    = \int_0^1 \theta\,p(\theta\mid D)\,d\theta
    $$

    이는 곧 사후분포의 평균이고, 정리 2에서 사후분포가 $\text{Beta}(\alpha+k,\beta+n-k)$ 이므로 그 평균은 $\dfrac{\alpha+k}{\alpha+\beta+n}$ 이다. $\alpha=\beta=1$ 을 넣으면 $(k+1)/(n+2)$ 이다.

!!! note "쓰임새"
    최대 우도는 $k=0$ 일 때 확률을 $0$ 으로 못 박아, 한 번도 못 본 일은 **결코 일어나지 않는다**고 말한다. 연속법칙은 $1/(n+2)$ 를 남겨 이 과신을 막는다. 언어 모델의 가산 평활화가 바로 이 계산이다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 같은 공정에서 생산한 부품 20개를 독립적으로 검사했는데 불량이 없었다. 불량 확률에 Beta(1,1) 사전분포를 둘 때 다음 부품의 불량 예측확률을 구하시오.

??? success "풀이"

    $n=20$, $k=0$ 이므로 부작용이 날 확률은

    $$
    P(\text{불량}) = \frac{1}{22} \approx 0.045
    $$

    이다. 이는 선택한 이항 모형과 균등 사전분포에 따른 예측확률이다. 다른 사전분포나 공정 변화 가정을 사용하면 값이 달라지며 모형과 무관한 보편적 불량률은 아니다.

## 4. 감마-푸아송 모델

### 정리 4. 감마-푸아송 켤레 갱신 — 예측 분포는 음이항 { .thm }

사전분포가 $\lambda \sim \text{Gamma}(a, b)$ (shape a>0, rate b>0)이고 $x_1,\dots,x_n$이 λ를 조건으로 독립인 $\text{Poisson}(\lambda)$ 관측이면

$$
\lambda \mid D \;\sim\; \text{Gamma}\Bigl(a + \sum_i x_i,\ \ b + n\Bigr)
$$

이고, 사후예측분포는 음이항

$$
p(x_{n+1} \mid D) = \binom{x_{n+1}+a_n-1}{x_{n+1}}
\Bigl(\frac{b_n}{b_n+1}\Bigr)^{a_n}\Bigl(\frac{1}{b_n+1}\Bigr)^{x_{n+1}}
$$

이다. 여기서 $a_n = a+\sum_i x_i$, $b_n = b+n$ 이다.

??? proof "증명"

    우도는 $\prod_i e^{-\lambda}\lambda^{x_i}/x_i! \propto \lambda^{\sum x_i}e^{-n\lambda}$ 이고 사전분포는 $\propto \lambda^{a-1}e^{-b\lambda}$ 이다. 곱하면

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
    다음 단위 시간의 예측분산은 E[λ|D]+Var(λ|D)로 예측평균보다 크다. 이는 공통 발생률 λ의 사후 불확실성 때문이다. 실제 시계열의 계절성·군집·서로 다른 발생률로 인한 과산포까지 이 공통 λ 모형이 자동으로 설명하는 것은 아니다. 관측 시간이 다르면 b에는 관측 개수가 아니라 노출 시간의 합을 더해야 한다.

??? note "코드 2 (32줄)"

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


**보기 3.** <span class="diff easy" title="쉬움"></span> 웹사이트 방문이 하루 평균 몇 번인지 알고자 한다. 사전분포 $\text{Gamma}(2, 1)$ 에 $5$ 일 동안 $3, 7, 4, 6, 5$ 번을 관측했다. 사후분포를 구하시오.

??? success "풀이"

    $\sum x_i = 25$, $n = 5$ 이므로 정리 4에서

    $$
    \lambda \mid D \sim \text{Gamma}(2+25,\ 1+5) = \text{Gamma}(27, 6)
    $$

    이다. 평균은 $27/6 = 4.5$, 표준편차는 $\sqrt{27}/6 \approx 0.87$ 이다. 표본 평균 $5$ 가 사전평균 $2$ 쪽으로 조금 끌려갔다.

## 5. 정규-정규 모델(분산을 알 때)

### 정리 5. 정밀도는 더해진다 — 사후평균은 정밀도로 가중 평균 { .thm }

$x_1,\dots,x_n$이 조건부 독립 정규 관측이고 $\sigma$ 를 알고 사전분포가 $\mu\sim\mathcal{N}(\mu_0,\sigma_0^2)$ 이라 하자. 정밀도를 $\tau_0 = 1/\sigma_0^2$, $\tau = 1/\sigma^2$ 로 두면

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
    정밀도로 보면 베이즈 갱신이 **정보를 더하는 일**이다. 사전분포가 $\tau_0$, 데이터가 $n\tau$ 만큼 정보를 들고 오고 사후분포는 그 합을 갖는다. 표준편차로 적으면 이 단순함이 가려진다.

??? note "코드 3 (41줄)"

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


**보기 4.** <span class="diff easy" title="쉬움"></span> 측정 잡음이 $\sigma = 2$ 임을 알고 사전분포가 $\mathcal{N}(10, 5^2)$ 이다. $n=9$ 개를 재어 평균 $\bar x = 12$ 를 얻었다. 사후분포를 구하시오.

??? success "풀이"

    $\tau_0 = 1/25 = 0.04$, $\tau = 1/4 = 0.25$ 이므로

    $$
    \tau_9 = 0.04 + 9(0.25) = 2.29,\qquad
    \mu_9 = \frac{0.04(10) + 2.25(12)}{2.29} \approx 11.965
    $$

    이다. 사후분포는 $\mathcal{N}(11.965,\ 1/2.29)$ 이고 표준편차는 약 $0.661$ 이다. 사전분포가 넓어서($\sigma_0=5$) 거의 데이터가 답을 정했다.

## 6. 정규-역감마 모델(둘 다 모를 때)

**정의 3.** [정규-역감마 사전분포]

평균과 분산을 모두 모를 때의 켤레 사전분포는

$$
\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0),
\qquad
\mu \mid \sigma^2 \sim \mathcal{N}\bigl(\mu_0,\ \sigma^2/\kappa_0\bigr)
$$

이다. κ₀,a₀,b₀>0이며 역감마 밀도는 v=σ²에 대해 v^(−a₀−1)exp(−b₀/v)에 비례한다. 이를 $\text{NIG}(\mu_0,\kappa_0,a_0,b_0)$로 적는다. $\mu$ 의 사전분산이 $\sigma^2$ 에 매여 있다는 점이 핵심이다.

### 정리 6. 정규-역감마 켤레 갱신 — 평균과 분산을 한꺼번에 고친다 { .thm }

$x_1,\dots,x_n$이 (μ,σ²)를 조건으로 독립인 정규 관측이고 사전분포가 $\text{NIG}(\mu_0,\kappa_0,a_0,b_0)$ 이면 사후분포도 정규-역감마이며 매개변수는 다음과 같다.

$$
\kappa_n = \kappa_0+n, \qquad
\mu_n = \frac{\kappa_0\mu_0 + n\bar x}{\kappa_n}, \qquad
a_n = a_0 + \frac n2
$$

$$
b_n = b_0 + \frac12\sum_i(x_i-\bar x)^2 + \frac{\kappa_0 n(\bar x-\mu_0)^2}{2\kappa_n}
$$

??? proof "증명"

    우도의 지수를 평균 언저리로 갈라 적으면

    $$
    \sum_i (x_i-\mu)^2 = \sum_i (x_i-\bar x)^2 + n(\bar x-\mu)^2
    $$

    이다. 사전분포의 $\mu$ 부분과 합치면 $\mu$ 에 대한 이차식

    $$
    \kappa_0(\mu-\mu_0)^2 + n(\mu-\bar x)^2
    = \kappa_n\bigl(\mu-\mu_n\bigr)^2 + \frac{\kappa_0 n(\bar x-\mu_0)^2}{\kappa_n}
    $$

    을 얻는다. 이는 완전제곱 묶기이며, 남은 상수항이 $b_n$ 의 마지막 항이다. $\mu$ 부분을 떼어 내면 $\sigma^2$ 에 대해 역감마 꼴이 남고 그 매개변수가 $(a_n, b_n)$ 이다.

!!! note "쓰임새"
    b_n은 역감마 분포의 스케일 매개변수이며 분산 자체가 아니다. 각 항은 사전 스케일, 자료 내 제곱편차의 절반, 사전평균·표본평균 불일치의 기여다. 다른 조건을 고정하면 마지막 항이 커질수록 σ²의 사후평균도 커진다. 다만 이 평균은 a_n>1일 때만 존재하며, 이 항만으로 모형 부적합을 진단할 수는 없다.

## 7. 주변 사후분포과 스튜던트 t

분산을 모르면 평균에 대한 사후분포는 더 이상 정규가 아니다.

### 정리 7. 평균의 주변 사후분포 — 스튜던트 t가 나온다 { .thm }

정리 6의 사후분포에서 $\sigma^2$ 를 적분해 없애면

$$
\mu \mid D \;\sim\; t_{2a_n}\Bigl(\mu_n,\ \sqrt{\frac{b_n}{a_n\kappa_n}}\Bigr)
$$

이다. 여기서는 t의 두 번째 인자를 스케일 s로 정의한다. 따라서 s²=b_n/(a_n κ_n)이며 분산은 아니다. 평균은 a_n>1/2, 유한 분산은 a_n>1일 때 존재하고 분산은 b_n/[κ_n(a_n−1)]이다.

??? proof "증명"

    결합 사후분포는

    $$
    p(\mu,\sigma^2\mid D) \propto (\sigma^2)^{-a_n-3/2}
    \exp\Bigl(-\frac{2b_n + \kappa_n(\mu-\mu_n)^2}{2\sigma^2}\Bigr)
    $$

    이다. $\sigma^2$ 에 대해 적분하면서 $s = 1/\sigma^2$ 로 바꾸면 감마 적분이 되어

    $$
    p(\mu\mid D) \propto \bigl(2b_n + \kappa_n(\mu-\mu_n)^2\bigr)^{-(a_n+1/2)}
    \propto \Bigl(1 + \frac{\kappa_n(\mu-\mu_n)^2}{2b_n}\Bigr)^{-\frac{2a_n+1}{2}}
    $$

    를 얻는다. 이는 자유도 $\nu = 2a_n$ 인 t 밀도 $\bigl(1+\frac{(\mu-m)^2}{\nu s^2}\bigr)^{-(\nu+1)/2}$ 의 꼴이며, 비교하여 보면 $s^2 = b_n/(a_n\kappa_n)$ 이다.

!!! note "쓰임새"
    정규 평균의 조건부 사후분포를 역감마 분산에 대해 적분하면 꼬리가 두꺼운 t분포가 된다. 빈도주의 t통계량과 형태는 비슷하지만 임의의 사전분포가 정확히 같은 구간을 주지는 않는다. 자유도가 커지면 표준화한 t분포는 표준정규분포에 가까워진다.

**문제 2.** <span class="diff hard" title="어려움"></span> $a_0 \to 0$, $b_0\to 0$, $\kappa_0\to 0$ 으로 보내면 정리 7의 주변 사후분포가 무엇이 되는지 구하고, 빈도주의 t 검정과 견주시오.

??? success "풀이"

    $\kappa_n \to n$, $\mu_n\to\bar x$, $a_n\to n/2$, $b_n\to \frac12\sum_i(x_i-\bar x)^2$ 이다. 스케일의 제곱은

    $$
    \frac{b_n}{a_n\kappa_n} \to \frac{\frac12\sum(x_i-\bar x)^2}{\frac n2\cdot n}
    = \frac{\sum(x_i-\bar x)^2}{n^2}
    $$

    이 극한에서 자유도는 n이고 스케일 제곱은 잔차제곱합/n²다. 따라서 고전적인 자유도 n−1, 스케일 s/√n의 t구간과 유한 표본에서 다르다. 이 극한의 부적절 결합 사전 밀도는 dμ d(σ²)에 대해 (σ²)^(−3/2)에 비례한다. 같은 기준측도에서 p(μ,σ²)∝1/σ²를 사용하면 자유도 n−1의 t 사후분포를 얻는다. 후자는 n≥2이고 잔차제곱합이 양수일 때 적절하며, 부적절 사전분포의 밀도를 어느 변수에 대해 적는지 명시해야 한다.

    빈도주의 t 구간은 자유도가 $n-1$ 인데 여기서는 $n$ 이다. 자유도 하나 차이는 이 부적절 사전분포가 정확히 $p(\mu,\sigma^2)\propto 1/\sigma^2$ 가 아니라 $1/\sigma^3$ 에 해당하기 때문이며, $p(\mu,\sigma^2)\propto 1/\sigma^2$ 를 쓰면 자유도가 정확히 $n-1$ 로 맞는다.

## 8. 특별한 경우와 점근적 성질

### 정리 8. 고정된 베타 사전분포 아래의 사후 일관성 { .thm }

베타-이항 모델에서 참값이 $\theta^\star \in (0,1)$ 이라 하자. 유한한 양의 사전 매개변수 $(\alpha,\beta)$를 고정하고 $n\to\infty$ 로 보내면

$$
\mathbb{E}[\theta\mid D] \longrightarrow \theta^\star,
\qquad
\operatorname{Var}[\theta\mid D] = O\!\left(\frac1n\right)
$$

이다. 관측은 고정된 θ⋆ 아래 독립 베르누이 시행이며 평균 수렴은 거의 확실한 수렴이다. 사전 매개변수가 n에 따라 증가하거나 모형이 달라지면 같은 결론을 자동 적용할 수 없다.

??? proof "증명"

    큰 수의 법칙에서 $k/n \to \theta^\star$ 이다. 정리 2의 평균을 다시 적으면

    $$
    \mathbb{E}[\theta\mid D] = \frac{\alpha+k}{\alpha+\beta+n}
    = \frac{\alpha/n + k/n}{(\alpha+\beta)/n + 1} \longrightarrow \frac{0+\theta^\star}{0+1} = \theta^\star
    $$

    이다. 분산은 $\alpha_n+\beta_n = \alpha+\beta+n$ 이므로

    $$
    \operatorname{Var} = \frac{\alpha_n\beta_n}{(\alpha_n+\beta_n)^2(\alpha_n+\beta_n+1)}
    \le \frac{1}{4(\alpha+\beta+n+1)} = O(1/n)
    $$

    이다. 가운데 부등식은 $\alpha_n\beta_n \le \bigl(\frac{\alpha_n+\beta_n}{2}\bigr)^2$ 에서 나온다.

!!! note "쓰임새"
    일반적인 일관성에는 참값 주변에 사전확률이 양수라는 조건과 모형의 식별성 등이 중요하다. 정확한 정규성 조건은 모형에 따라 다르다. 홀데인 사전은 p(θ)∝1/[θ(1−θ)]인 부적절 밀도이며 양쪽 결과가 모두 관측되지 않으면 사후분포도 정규화되지 않는다.

**보기 5.** <span class="diff med" title="중간"></span> 균등·제프리스·홀데인 세 사전분포를 견주고, 각각이 무엇을 뜻하는지 밝히시오.

??? success "풀이"

    | 사전분포 | $(\alpha,\beta)$ | 유사 관측 | 사후평균 |
    |--------|------------------|-----------|-------------|
    | 균등 | $(1,1)$ | $2$ | $(k+1)/(n+2)$ |
    | 제프리스 | $(0.5,0.5)$ | $1$ | $(k+0.5)/(n+1)$ |
    | 홀데인 | $(0,0)$ | $0$ | $k/n$ |

    홀데인 사전은 0<k<n인 경우에만 적절한 Beta(k,n−k) 사후분포를 주고 평균이 k/n과 같다. k=0 또는 k=n이면 사후분포가 부적절하다. 균등·제프리스 사전은 양의 베타 매개변수로 적절하지만 어느 하나가 모든 목적에서 최선인 것은 아니다.

**문제 3.** <span class="diff med" title="중간"></span> 사전분포 매개변수를 실제로 어떻게 고를지, 두 가지 길을 들어 설명하시오.

??? success "풀이"

    첫째는 **적률 맞추기**다. 앞선 지식으로 평균 $m$ 과 표준편차 $s$ 를 정한 뒤 베타의 적률 식을 풀어 $(\alpha,\beta)$ 를 얻는다.

    $$
    \alpha+\beta = \frac{m(1-m)}{s^2}-1,\qquad \alpha = m(\alpha+\beta)
    $$

    둘째는 사후평균의 유효 표본 크기 N>0와 중심 0<m<1을 정해 α=mN, β=(1−m)N으로 두는 방법이다. 이는 평균 가중치 해석이며 실제 과거 성공·실패 수를 그대로 재현한다는 뜻은 아니다. 적률법에는 0<s²<m(1−m) 조건이 필요하고, 선택한 사전분포의 예측과 민감도를 점검한다.

## 9. 전체 구현 예제

전체 예제는 베타-베르누이, 알려진 분산의 정규, 정규-역감마 추론을 각각 구현한다. 공통으로 양의 분포 매개변수와 유한한 입력을 확인해야 한다. 베타 갱신에는 0≤성공 수≤시행 수의 정수가 필요하며, 단순 정규 예제는 빈 배열을 처리하지 않는다.

코드 1의 posterior_predictive에 있는 stats.binom.comb와 stats.beta.beta_func는 공식 SciPy API가 아니다. 조합과 베타 함수는 [scipy.special](https://docs.scipy.org/doc/scipy/reference/special.html)에 있으며 아래의 더 긴 베타 예제는 comb·betaln을 사용한다. 코드 1을 완성된 예측 구현으로 해석하지 않는다.

긴 베타 예제의 log_marginal_likelihood는 순서가 지정된 베르누이 관측열의 확률을 계산한다. 성공 횟수만 관측한 이항 자료의 주변 우도에는 조합 계수가 추가된다. HPD 함수는 [0.001,0.999] 격자에서 높은 밀도를 고른 뒤 최소·최대값을 반환하므로 경계 질량과 분리된 HPD 집합을 정확히 처리하지 못한다. mode가 0인 경우에도 시연의 조건식은 undefined를 출력할 수 있다.

알려진 분산의 긴 정규 예제에서 사후 갱신·예측 공식과 log_marginal_likelihood를 구분한다. 후자의 정규화 항과 제곱항은 정규 적분의 주변 우도 공식과 일치하지 않는다. 모델 비교에 그대로 사용하지 않는다. 반면 사후 예측분산은 알려진 관측분산과 평균의 사후분산의 합이다.

정규-역감마 예제의 t 스케일과 역감마 스케일은 서로 다른 양이다. 제곱합에서 큰 비슷한 수를 빼는 계산은 반올림 오차를 일으킬 수 있다. update_sequential은 기존 상태를 초기화하며, 위치 매개변수를 사후평균으로 해석하려면 t평균의 존재 조건이 필요하다. 자유도 30이라는 표시도 모든 꼬리 확률에서 정규 근사가 충분하다는 기준은 아니다.

아래 출력과 그림은 기존 예시를 보존했다. 작은 양의 NIG 하이퍼파라미터를 사용한 t구간 비교는 자유도 n−1의 고전적 구간과 정확히 같은 계산은 아니다.

??? note "코드 4 (527줄)"

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


**출력:**

```
============================================================
BASIC BETA-BERNOULLI INFERENCE
============================================================

True θ: 0.7
Data: 14 successes in 20 trials
MLE: 0.7000

Posterior summaries under different priors:
------------------------------------------------------------

Uniform prior: Beta(1, 1)
  Posterior: Beta(15.0, 7.0)
  Mean: 0.6818
  Mode: 0.7000
  Std:  0.0971
  95% CI: [0.4782, 0.8541]
  P(next success): 0.6818

Jeffreys prior: Beta(0.5, 0.5)
  Posterior: Beta(14.5, 6.5)
  Mean: 0.6905
  Mode: 0.7105
  Std:  0.0986
  95% CI: [0.4828, 0.8639]
  P(next success): 0.6905

Informative (pessimistic) prior: Beta(2, 8)
  Posterior: Beta(16.0, 14.0)
  Mean: 0.5333
  Mode: 0.5357
  Std:  0.0896
  95% CI: [0.3569, 0.7055]
  P(next success): 0.5333

Informative (optimistic) prior: Beta(8, 2)
  Posterior: Beta(22.0, 8.0)
  Mean: 0.7333
  Mode: 0.7500
  Std:  0.0794
  95% CI: [0.5646, 0.8727]
  P(next success): 0.7333

============================================================
SEQUENTIAL BAYESIAN LEARNING
============================================================

True θ: 0.6
Outcomes: [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]

Evolution of posterior mean:
----------------------------------------
After obs  1 (x=0): E[θ|D] = 0.3333, σ = 0.2357, Data: 0/1
After obs  2 (x=1): E[θ|D] = 0.5000, σ = 0.2236, Data: 1/2
After obs  3 (x=1): E[θ|D] = 0.6000, σ = 0.2000, Data: 2/3
After obs  4 (x=1): E[θ|D] = 0.6667, σ = 0.1782, Data: 3/4
After obs  5 (x=0): E[θ|D] = 0.5714, σ = 0.1750, Data: 3/5
After obs  6 (x=1): E[θ|D] = 0.6250, σ = 0.1614, Data: 4/6
After obs  7 (x=0): E[θ|D] = 0.5556, σ = 0.1571, Data: 4/7
After obs  8 (x=0): E[θ|D] = 0.5000, σ = 0.1508, Data: 4/8
After obs  9 (x=1): E[θ|D] = 0.5455, σ = 0.1437, Data: 5/9
After obs 10 (x=1): E[θ|D] = 0.5833, σ = 0.1367, Data: 6/10
After obs 11 (x=1): E[θ|D] = 0.6154, σ = 0.1300, Data: 7/11
After obs 12 (x=0): E[θ|D] = 0.5714, σ = 0.1278, Data: 7/12
After obs 13 (x=1): E[θ|D] = 0.6000, σ = 0.1225, Data: 8/13
After obs 14 (x=1): E[θ|D] = 0.6250, σ = 0.1174, Data: 9/14
After obs 15 (x=1): E[θ|D] = 0.6471, σ = 0.1126, Data: 10/15

See: sequential_beta_update.png

============================================================
POSTERIOR PREDICTIVE DISTRIBUTION
============================================================

Observed: 7 successes in 10 trials
Posterior: Beta(8, 4)

Predictive distribution for next 10 trials:
----------------------------------------
Expected successes in next 10: 6.67
Most likely outcome: 7 successes

See: predictive_distribution.png
```

??? note "코드 5 (554줄)"

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


**출력:**

```
============================================================
GAUSSIAN INFERENCE WITH KNOWN VARIANCE
============================================================

True μ: 5.0
Known σ²: 4.0
Sample: n = 10, x̄ = 5.8961
MLE: 5.8961

Posterior summaries under different priors:
------------------------------------------------------------

Weak prior (σ₀² = 100)
  Prior: N(0.0, 100.0)
  Posterior: N(5.8726, 0.3984)
  Prior weight: 0.4%
  Data weight: 99.6%
  95% CI: [4.636, 7.110]

Moderate prior
  Prior: N(3.0, 4.0)
  Posterior: N(5.6328, 0.3636)
  Prior weight: 9.1%
  Data weight: 90.9%
  95% CI: [4.451, 6.815]

Strong prior (wrong)
  Prior: N(10.0, 1.0)
  Posterior: N(7.0687, 0.2857)
  Prior weight: 28.6%
  Data weight: 71.4%
  95% CI: [6.021, 8.116]

Strong prior (right)
  Prior: N(5.0, 1.0)
  Posterior: N(5.6401, 0.2857)
  Prior weight: 28.6%
  Data weight: 71.4%
  95% CI: [4.592, 6.688]

============================================================
SEQUENTIAL BAYESIAN UPDATING
============================================================

True μ: 100.0
Prior: N(80.0, 100.0) [wrong!]
Known σ²: 25.0

Posterior evolution:
--------------------------------------------------
   n        x     E[μ|D]     σ_post    Data Wt
--------------------------------------------------
   1    94.57     91.657      4.472      80.0%
   2   104.99     97.582      3.333      88.9%
   3   101.41     98.761      2.774      92.3%
   4    92.47     97.280      2.425      94.1%
   5    97.11     97.247      2.182      95.2%
   6   108.26     99.009      2.000      96.0%
   7    87.87     97.472      1.857      96.6%
   8    97.86     97.519      1.741      97.0%
   9   106.33     98.471      1.644      97.3%
  10    95.67     98.197      1.562      97.6%

See: gaussian_sequential_update.png

============================================================
POSTERIOR PREDICTIVE DISTRIBUTION
============================================================

Observed: 15 observations
Posterior for μ: N(49.47, 1.0230)

Predictive for x_16:
  Mean: 49.47
  Variance: 17.0230
    = Aleatoric (16.0000) + Epistemic (1.0230)
  95% Prediction Interval: [41.39, 57.56]

See: gaussian_predictive.png
```

??? note "코드 6 (583줄)"

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


**출력:**

```
======================================================================
GAUSSIAN INFERENCE WITH UNKNOWN VARIANCE
======================================================================

True: μ = 5.0, σ² = 4.0
Data: n = 20, x̄ = 4.6574, s² = 3.6866

Posterior NIG: NIG(μ=4.6342, κ=20.1000, α=10.1000, β=36.2020)
Marginal for μ: t_20.2(4.6342, 0.4223)
Marginal for σ²: Inv-Gamma(10.1000, 36.2020)

======================================================================
CONNECTION TO t-TEST
======================================================================

Frequentist 95% CI: [47.8735, 56.2946]
Bayesian 95% CI:    [47.9633, 56.2007]
Difference: 0.093895
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 디리클레-다항이 켤레 짝임을 정리 1로 설명하고 갱신 규칙을 적으시오.

</div>

??? success "풀이"

    조건부 독립 다항 관측의 범주별 도수를 n_j라 하면 우도는 범주 확률 π_j에 대해 $\prod_j\pi_j^{n_j}$에 비례한다. 디리클레 사전 밀도 $\prod_j\pi_j^{\alpha_j-1}$를 곱하면 지수가 α_j+n_j−1로 갱신된다. α_j>0이고 π는 확률 단체에 속한다. 자연 매개변수에서 정리 1을 사용할 때는 좌표 변환의 야코비안을 반영한다.

    갱신은 도수를 더하는 일뿐이다.

    $$
    \text{Dir}(\alpha_1+n_1,\ \dots,\ \alpha_K+n_K)
    $$

    $K=2$ 이면 베타-이항으로 돌아간다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 지수 우도의 켤레 사전분포가 감마임을 보이고 갱신 규칙을 이끌어 내시오.

</div>

??? success "풀이"

    $x_i\sim\text{Exp}(\lambda)$ 이면 우도는 $\lambda^n e^{-\lambda\sum x_i}$ 이다. 사전분포를 $\text{Gamma}(a,b)\propto \lambda^{a-1}e^{-b\lambda}$ 로 두면

    $$
    p(\lambda\mid D)\propto \lambda^{a+n-1}e^{-(b+\sum x_i)\lambda}
    $$

    으로 $\text{Gamma}(a+n,\ b+\sum_i x_i)$ 이다. 충분 통계량이 $(n, \sum x_i)$ 이고 이 둘이 각각 더해지는 것이 정리 1의 결론과 맞는다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 정리 3의 연속법칙을 언어 모델의 가산 평활화와 이어 설명하시오.

</div>

??? success "풀이"

    어휘 크기가 $V$ 인 다항 모델에서 각 단어에 $\text{Dir}(\alpha,\dots,\alpha)$ 사전분포를 두면 예측 확률은

    $$
    P(w \mid D) = \frac{n_w + \alpha}{N + V\alpha}
    $$

    이다. $\alpha=1$ 이 라플라스 평활화, $\alpha<1$ 이 리드스톤 평활화다.

    한 번도 안 나온 단어에 $0$ 이 아닌 확률을 주는 것이 핵심이며, 이는 연습문제 1의 디리클레 갱신에 정리 3의 논증을 그대로 얹은 것이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> 정리 7에서 표준화한 t 사후분포가 큰 표본에서 표준정규분포로 가까워지는 과정을 설명하시오.

</div>

??? success "풀이"

    자유도가 $2a_n = 2a_0+n \to\infty$ 이다. 스튜던트 t의 밀도에서

    $$
    \Bigl(1+\frac{z^2}{\nu}\Bigr)^{-\frac{\nu+1}{2}}
    = \exp\Bigl(-\frac{\nu+1}{2}\log\bigl(1+\tfrac{z^2}{\nu}\bigr)\Bigr)
    \longrightarrow e^{-z^2/2}
    $$

    이다. $\log(1+u)\approx u$ 를 썼다. 곧 t는 정규로 다가간다.

    고정된 적절한 사전분포와 정확한 독립 정규 관측 모형 아래 스케일의 제곱 b_n/(a_n κ_n)은 큰 표본에서 s²/n과 가까워진다. 사후분포를 표본평균으로 중심화하고 적절히 스케일하면 정규 근사가 가능하다. 원래 스케일에서는 사후분포가 참평균 주변으로 집중한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 사전분포 민감도 분석을 켤레 모델에서 어떻게 하면 값싸게 할 수 있는지 밝히시오.

</div>

??? success "풀이"

    켤레 모델은 충분통계량을 저장한 뒤 사전 매개변수별로 갱신식과 관심 요약량을 다시 계산하면 된다. MCMC를 매번 수행할 필요가 없어 민감도 비교가 빠르다. 계산이 전혀 필요 없다는 뜻은 아니며 비켤레 모델에서도 중요도 재가중 등으로 일부 계산을 재사용할 수 있다.

    이는 MCMC를 다시 돌려야 하는 켤레 아닌 모델과 크게 다른 점이다. 켤레 모델을 먼저 세워 민감도를 살핀 뒤 더 복잡한 모델로 옮겨 가는 것이 실전에서 값싼 길이다.

## 정리하며

켤레성은 사후분포가 같은 분포족에 속하는 성질이다. 지수족의 자연 매개변수 사전은 충분통계량을 더해 갱신할 수 있지만 정규화 가능성과 좌표별 밀도를 확인해야 한다. 예측분포는 매개변수의 불확실성을 적분하므로 매개변수 사후분포와 다르다.

베타·감마·정규·정규-역감마 모델의 매개변수화와 유효 범위를 명시하면 갱신·예측·민감도 비교가 간단해진다. 비켤레 모델에도 다른 해석적 해나 수치 적분·라플라스 근사·변분 추론·MCMC 등의 선택지가 있다. [MAP 추정](map_estimation.md)과 [베이즈 신용구간](credible_intervals.md)은 사후분포의 서로 다른 요약 방법이다.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2~3장
- Bishop, C. *Pattern Recognition and Machine Learning*, 2장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3~5장
- Diaconis, P. & Ylvisaker, D. "Conjugate Priors for Exponential Families." *Annals of Statistics*, 1979
