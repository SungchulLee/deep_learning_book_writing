# 사전분포, 우도, 사후분포

베이즈 추론은 사전분포와 우도를 결합해 사후분포를 구한다. 주변 우도는 이 분포를 정규화하고 모델 비교에도 사용된다. 각 개념을 정의한 뒤 베타-이항과 정규-정규 모델의 사후분포를 유도하고, 순차 갱신과 수치 근사의 조건을 살펴본다.

## 1. 베이즈 정리: 네 조각

**정의 1.** [사전분포, 우도, 사후분포, 증거]

파라메터 θ와 관측 데이터 D에 대해 다음과 같이 정의한다. 연속형 변수에서는 확률 자체와 확률밀도를 구분하며, p는 문맥에 따라 확률질량 또는 밀도를 나타낸다.

- **사전분포** $p(\theta)$: 데이터를 관측하기 전 파라메터의 불확실성을 나타낸다.
- **우도** $p(D\mid\theta)$: 관측한 D를 고정하고 θ의 함수로 보는 확률질량 또는 밀도다.
- **사후분포** $p(\theta\mid D)$: 관측 데이터를 반영한 파라메터의 분포다.
- **주변 우도** $p(D)$: 우도를 사전분포에 대해 평균한 값이며 증거라고도 한다.

### 정리 1. 베이즈 정리 — 사후분포는 우도와 사전분포의 곱에 비례한다 { .thm }

$0<p(D)<∞$이면 다음이 성립한다. 이산형 파라메터에서는 적분을 합으로 바꾼다.

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\,p(\theta)}{p(D)},
\qquad
p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta
$$

특히 $p(D)$ 는 $\theta$ 에 달리지 않으므로

$$
p(\theta \mid D) \;\propto\; p(D \mid \theta)\,p(\theta)
$$

이다.

??? proof "증명"

    조건부확률의 정의에서 두 가지로 결합밀도를 적으면

    $$
    p(\theta, D) = p(\theta \mid D)\,p(D) = p(D \mid \theta)\,p(\theta)
    $$

    이다. $p(D) > 0$ 이므로 양변을 $p(D)$ 로 나누면 첫 식을 얻는다.

    둘째 식은 결합밀도를 $\theta$ 에 대해 주변화한 것이다.

    $$
    p(D) = \int p(\theta, D)\,d\theta = \int p(D \mid \theta)\,p(\theta)\,d\theta
    $$

!!! note "쓰임새"
    비례식은 정규화 가능한 사후분포의 형태를 정한다. MAP처럼 밀도의 최댓값만 필요할 때는 θ와 무관한 주변 우도를 계산하지 않아도 된다. 다만 적분이 유한하고 양수인지, 최댓값이 존재하는지는 별도로 확인해야 한다.

## 2. 사전분포

**정의 2.** [정보를 담은 사전분포과 담지 않은 사전분포]

사전분포가 특정 범위에 얼마나 집중하는지를 기준으로 정보적 사전분포와 약한 정보 사전분포를 구분한다. 완전히 중립적인 사전분포가 하나로 정해지는 것은 아니며 변수의 스케일과 관측 모형에 따라 영향이 달라진다. 우도와 결합한 사후분포가 같은 분포족에 속하면 켤레 사전분포라 한다.

| 종류 | 예 | 해석과 조건 |
|------|------|-------|
| 균등 사전분포 | Beta(1,1) | 성공 확률 θ의 원래 좌표에서 균등하다. |
| 부적절 사전분포 | 실수 전체에서 p(μ)∝1 | 적분이 무한하므로 확률분포가 아니다. 사후분포의 적절성을 확인한다. |
| 제프리스 사전분포 | 베르누이 모형의 Beta(0.5,0.5) | 일대일 리파라메리제이션에서 같은 확률측도를 정의한다. |
| 약한 정보의 예 | Beta(2,2) | 이 모형에서는 0·1 근처를 완화한다. 영향은 데이터 양에 달려 있다. |
| 집중된 사전분포의 예 | Beta(10,10) | θ≈0.5라는 상대적으로 강한 가정을 반영한다. |

### 정리 2. 사전예측분포 — 관측 전의 데이터 분포 { .thm }

데이터를 보기 전에 어떤 데이터가 나올지 묻는 분포는 우도를 사전분포로 평균 낸 것이며, 이는 곧 증거이다.

$$
p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta
$$

베타-이항 모델에서 $n$ 번 중 $k$ 번 성공할 사전예측확률은 베타-이항 분포

$$
p(k) = \binom{n}{k}\,\frac{B(\alpha+k,\ \beta+n-k)}{B(\alpha,\beta)}
$$

이다. 여기서 $B$ 는 베타 함수이다.

??? proof "증명"

    이항 우도와 베타 사전분포를 넣으면

    $$
    p(k) = \int_0^1 \binom{n}{k}\theta^{k}(1-\theta)^{n-k}\,
    \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}\,d\theta
    $$

    이다. 적분 안을 모으면 $\theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$ 이고, 베타 함수의 정의

    $$
    \int_0^1 \theta^{a-1}(1-\theta)^{b-1}\,d\theta = B(a,b)
    $$

    를 쓰면 곧바로 결론이 나온다.

!!! note "쓰임새"
    사전예측분포는 사전분포와 관측 모형의 조합을 점검하는 데 사용한다. 현실적으로 불가능한 자료에 큰 확률을 주면 사전분포뿐 아니라 우도의 스케일과 관측 가정도 검토한다. 주변 우도를 확률분포로 해석하려면 적절한 사전분포가 필요하다.

**보기 1.** <span class="diff easy" title="쉬움"></span> $\text{Beta}(1,1)$ 사전분포에서 동전을 $n$ 번 던질 때 앞면이 $k$ 번 나올 사전예측확률을 구하시오.

??? success "풀이"

    $\alpha=\beta=1$ 이므로 $B(1,1)=1$ 이고

    $$
    p(k) = \binom{n}{k} B(1+k,\ 1+n-k)
    = \binom{n}{k}\frac{k!\,(n-k)!}{(n+1)!} = \frac{1}{n+1}
    $$

    이다. 곧 $k = 0,1,\dots,n$ 이 모두 **동일한 확률을 갖는다**. 균등 사전분포가 뜻하는 바가 "성공 횟수에 대해 아무 선호도 없다"임을 보여 준다.

**문제 1.** <span class="diff med" title="중간"></span> 제프리스 사전분포 $\text{Beta}(0.5, 0.5)$ 가 파라메터 바꾸기에 흔들리지 않는다는 것이 무슨 뜻인지 설명하고, 균등 사전분포가 왜 그 성질을 갖지 못하는지 보이시오.

??? success "풀이"

    $\theta$ 에 균등 사전분포를 두면 $\eta = \log\frac{\theta}{1-\theta}$ 같은 다른 파라메터에서는 고르지 않다. 야코비안 $\bigl|d\theta/d\eta\bigr| = \theta(1-\theta)$ 가 붙기 때문이다. 곧 "아무것도 모른다"는 진술이 어떤 자로 재느냐에 따라 달라진다.

    제프리스 사전분포는 $p(\theta) \propto \sqrt{I(\theta)}$ 로 잡는데, 피셔 정보 $I$ 가 파라메터 바꾸기에서 야코비안의 제곱으로 바뀌므로 $\sqrt{I}$ 는 야코비안 한 겹으로 바뀐다. 이는 밀도가 바뀌는 방식과 정확히 같아서, 어느 자로 재든 같은 사전분포를 준다. 베르누이에서 $I(\theta) = 1/(\theta(1-\theta))$ 이므로 $p(\theta)\propto \theta^{-1/2}(1-\theta)^{-1/2}$, 곧 $\text{Beta}(0.5,0.5)$ 이다.

## 3. 우도 함수

**정의 3.** [우도와 로그우도]

데이터 $D$ 를 고정하고 $\theta$ 의 함수로 본 $p(D \mid \theta)$ 를 **우도 함수** $L(\theta)$ 라 하고, 그 로그를 **로그우도** $\ell(\theta) = \log L(\theta)$ 라 한다. 관측이 θ를 조건으로 서로 독립이면

$$
L(\theta) = \prod_{i=1}^n p(x_i \mid \theta),
\qquad
\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)
$$

이다. 우도는 θ에 대한 확률밀도로 정의되지 않으므로 θ에 대한 적분이 1일 필요가 없다. 데이터 공간에 대한 정규화와 파라메터 공간에 대한 정규화는 다르다.

### 정리 3. 인수분해 정리 — 충분 통계량이 우도를 요약한다 { .thm }

공통 지배측도 아래 밀도를 갖는 통계 모형에서, 통계량 $T(D)$에 대해 음이 아닌 함수 $g,h$가 있어

$$
p(D \mid \theta) = g\bigl(T(D),\, \theta\bigr)\, h(D)
$$

로 적히는 것과 $T$ 가 $\theta$ 에 대한 **충분 통계량**인 것은 같은 말이다. 이때 사후분포는 $T(D)$ 에만 달린다.

$$
p(\theta \mid D) \;\propto\; g\bigl(T(D), \theta\bigr)\, p(\theta)
$$

??? proof "증명"

    인수분해가 성립한다고 하자. 그러면

    $$
    p(\theta \mid D) = \frac{g(T(D),\theta)h(D)p(\theta)}{\int g(T(D),\theta')h(D)p(\theta')\,d\theta'}
    = \frac{g(T(D),\theta)p(\theta)}{\int g(T(D),\theta')p(\theta')\,d\theta'}
    $$

    이다. 관측한 자료에서 정규화가 가능하면 h(D)가 약분되므로 사후분포는 D에 대해 T(D)를 통해서만 달라진다. 이는 인수분해가 베이즈 갱신을 요약한다는 설명이며, 특정 사전분포 하나에서의 사후 관계만으로 빈도주의 충분성을 역으로 증명한 것은 아니다.

    이산형 자료에서는 충분성의 정의에 따라 D의 T 조건부 분포가 θ에 의존하지 않는다. 따라서 g(T,θ)=p(T|θ), h(D)=p(D|T(D))로 인수분해할 수 있다. 연속형 자료의 일반적인 필요충분성은 조건부 밀도와 지배측도에 대한 인수분해 정리로 이해한다.

!!! note "쓰임새"
    베르누이 시행 $n$ 번에서 충분 통계량은 성공 횟수 $\sum x_i$ 하나뿐이다. 어느 시행에서 성공했는지 하는 순서 정보는 $\theta$ 에 대해 아무것도 말해 주지 않는다. 그래서 데이터 $n$ 개를 수 하나로 줄여도 손해가 없다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 정규 $\mathcal{N}(\mu, \sigma^2)$ 에서 $\sigma$ 를 알 때 충분 통계량을 구하시오.

??? success "풀이"

    로그우도를 펼치면

    $$
    \ell(\mu) = -\frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2 + \text{const}
    = -\frac{1}{2\sigma^2}\Bigl(\sum_i x_i^2 - 2\mu\sum_i x_i + n\mu^2\Bigr) + \text{const}
    $$

    이다. $\mu$ 와 얽힌 항은 $\sum_i x_i$ 뿐이므로 $T(D) = \sum_i x_i$, 같은 말로 표본 평균 $\bar x$ 가 충분 통계량이다.

**문제 2.** <span class="diff med" title="중간"></span> **우도 원리**란 두 실험이 비례하는 우도를 주면 같은 추론을 해야 한다는 것이다. 동전을 $12$ 번 던져 앞면 $9$ 번을 본 경우와, 앞면이 $9$ 번 나올 때까지 던져 $12$ 번 만에 멈춘 경우를 비교하여 이 원리를 설명하시오.

??? success "풀이"

    앞의 것은 이항, 뒤의 것은 음이항 실험이다. 우도는 각각

    $$
    \binom{12}{9}\theta^9(1-\theta)^3,
    \qquad
    \binom{11}{8}\theta^9(1-\theta)^3
    $$

    으로 $\theta$ 에 대해 **비례한다**. 상수 $\binom{12}{9}$ 와 $\binom{11}{8}$ 은 $\theta$ 에 달리지 않으므로 정리 1의 비례 관계에서 지워진다.

    따라서 같은 사전분포와 올바른 관측 모형을 사용하면 두 경우의 사후분포가 같다. 반면 빈도주의 $p$ 값은 "더 극단적인 결과"를 어떻게 셈하느냐가 실험 설계에 달려 있어 서로 다른 값이 나온다. 이것이 우도 원리를 둘러싼 오랜 다툼의 자리다.

## 4. 증거와 사후분포 계산

**정의 4.** [증거]

$p(D) = \int p(D\mid\theta)p(\theta)\,d\theta$ 를 **증거** 또는 **주변 우도**라 한다. 사후분포를 정규화하는 상수이면서, 동시에 그 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 값이다.

### 정리 4. 증거의 두 얼굴 — 정규화 상수이자 모델 비교의 자 { .thm }

두 모델 $M_1, M_2$ 에 대해 베이즈 인자를

$$
\text{BF}_{12} = \frac{p(D \mid M_1)}{p(D \mid M_2)}
$$

로 두면, 모델에 대한 사후 오즈은 사전 오즈에 베이즈 인자를 곱한 것이다.

$$
\frac{p(M_1 \mid D)}{p(M_2 \mid D)} = \text{BF}_{12}\cdot\frac{p(M_1)}{p(M_2)}
$$

??? proof "증명"

    모델 하나하나에 베이즈 정리를 쓰면

    $$
    p(M_i \mid D) = \frac{p(D \mid M_i)\,p(M_i)}{p(D)}
    $$

    이다. 두 식의 비를 잡으면 공통 분모 $p(D)$ 가 지워져

    $$
    \frac{p(M_1\mid D)}{p(M_2\mid D)} = \frac{p(D\mid M_1)}{p(D\mid M_2)}\cdot\frac{p(M_1)}{p(M_2)}
    $$

    를 얻는다.

!!! note "쓰임새"
    파라메터의 정규화된 사후분포를 구할 때 주변 우도는 정규화 상수다. 모델 비교에서는 각 모델의 적절한 파라메터 사전분포로 평균한 우도를 사용한다. 잘 맞는 좁은 영역만 있는 모델은 넓은 사전 범위 때문에 평균 우도가 낮아질 수 있지만, 파라메터가 많다는 이유만으로 주변 우도가 반드시 감소하지는 않는다. 부적절 사전분포의 임의 상수는 베이즈 인자를 정의하지 못하게 할 수 있다.

저차원에서는 격자 수치 적분으로 근사할 수 있다. 아래 함수에는 같은 길이의 1차원 격자·밀도·우도 값을 전달해야 하며 격자는 오름차순이고 적분값은 유한한 양수여야 한다. 격자 밖 질량과 곱셈의 언더플로도 확인해야 한다.

```python
import numpy as np

def posterior_continuous(theta_grid, prior, likelihood, normalize=True):
    """
    격자 위에서 뒤확률 분포를 셈한다.
    
    파라메터
    ----------
    theta_grid : array-like
        파라메터 값의 격자
    prior : array-like
        theta_grid에서 값을 매긴 앞확률 밀도
    likelihood : array-like
        theta_grid에서 값을 매긴 가능도
    
    반환값
    -------
    posterior : numpy array
        theta_grid에서의 뒤확률 밀도
    """
    theta_grid = np.asarray(theta_grid)
    prior = np.asarray(prior)
    likelihood = np.asarray(likelihood)
    
    # 고르게 하지 않은 뒤확률
    posterior = prior * likelihood
    
    if normalize:
        # 수치 적분(사다리꼴 규칙)
        evidence = np.trapz(posterior, theta_grid)
        posterior = posterior / evidence
    
    return posterior
```

**보기 3.** <span class="diff easy" title="쉬움"></span> 위 함수가 왜 사다리꼴 적분만으로 사후분포를 얻는지 설명하시오.

??? success "풀이"

    정리 1의 비례 관계에 따라 격자 위에서 `prior * likelihood` 를 셈하면 정규화하지 않은 사후분포가 곧바로 나온다. 남은 일은 그 곡선 아래 넓이로 나누어 적분이 $1$ 이 되게 하는 것뿐이며, 그 넓이가 바로 증거 $p(D)$ 의 수치 추정값이다.

    격자가 촘촘할수록 정확해지지만 차원이 늘면 격자점 수가 지수로 늘어난다. 그래서 파라메터가 몇 개만 넘어가도 격자 대신 샘플링을 쓴다.

## 5. 베타-이항 켤레 모델

이항 우도와 베타 사전분포를 결합하면 사후분포가 다시 베타가 된다. 켤레성이 주는 가장 단순하고 중요한 보기다.

### 정리 5. 베타-이항 켤레성 — 사전분포는 유사 관측으로 더해진다 { .thm }

α,β>0이고 n번의 조건부 독립 시행 중 k번 성공을 관측하면 사후분포는

$$
\theta \mid D \;\sim\; \text{Beta}(\alpha + k,\ \beta + n - k)
$$

이다. 따라서 사후분포의 평균은 사전평균과 최대 우도 추정값의 가중 평균이다. 이 표현은 n>0일 때 사용하며 n=0이면 사후분포는 사전분포와 같다.

$$
\mathbb{E}[\theta \mid D]
= \frac{\alpha+\beta}{\alpha+\beta+n}\cdot\frac{\alpha}{\alpha+\beta}
\;+\; \frac{n}{\alpha+\beta+n}\cdot\frac{k}{n}
$$

??? proof "증명"

    비례 관계에서

    $$
    p(\theta\mid D) \propto \theta^{k}(1-\theta)^{n-k}\cdot\theta^{\alpha-1}(1-\theta)^{\beta-1}
    = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}
    $$

    이고 이는 $\text{Beta}(\alpha+k,\ \beta+n-k)$ 의 커널이다.

    평균은 $\dfrac{\alpha+k}{\alpha+\beta+n}$ 이며, 분자를 $\alpha + k$ 로 쪼개어

    $$
    \frac{\alpha}{\alpha+\beta+n} + \frac{k}{\alpha+\beta+n}
    $$

    로 적은 뒤 각 항에 $\dfrac{\alpha+\beta}{\alpha+\beta}$ 와 $\dfrac nn$ 을 곱해 정리하면 결론의 꼴이 된다.

!!! note "쓰임새"
    α+β는 사후평균의 웨이트에서 사전분포의 유효 표본 크기로 해석할 수 있다. 이를 실제 성공·실패 관측 수라고 단정하지는 않는다. 베타 밀도의 지수는 α−1, β−1이므로 최빈값에서의 해석과도 다르다. 고정된 유효 표본 크기에 비해 n이 커지면 사후평균에서 사전평균의 웨이트가 작아진다.

아래 함수의 사후 최빈값 공식은 갱신한 두 베타 파라메터가 모두 1보다 클 때의 내부 최빈값에 해당한다. 경계 최빈값이나 균등분포는 별도 처리가 필요하며 관측 수가 0이면 MLE 계산도 정의되지 않는다.

```python
from scipy import stats

def beta_binomial_inference(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    베타-이항 모형을 써서 동전 던지기 확률에 대한 베이즈 추론.
    
    파라메터
    ----------
    n_heads : int
        관측된 앞면의 횟수
    n_tails : int
        관측된 뒷면의 횟수
    prior_alpha, prior_beta : float
        베타 앞확률의 파라메터
    
    반환값
    -------
    posterior_dist : scipy.stats.beta
        뒤확률 베타 분포
    """
    # 켤레 새로 고치기
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # 분포 만들기
    prior_dist = stats.beta(prior_alpha, prior_beta)
    posterior_dist = stats.beta(post_alpha, post_beta)
    
    # 통계
    prior_mean = prior_dist.mean()
    post_mean = posterior_dist.mean()
    post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2)
    
    # 최대 가능도 어림값
    mle = n_heads / (n_heads + n_tails)
    
    return posterior_dist
```

```python
def compare_priors(n_heads, n_tails):
    """서로 다른 앞확률 아래의 뒤확률을 견준다."""
    
    priors = {
        'Uniform': (1, 1),
        'Jeffreys': (0.5, 0.5),
        'Weak Fair': (2, 2),
        'Strong Fair': (10, 10),
        'Skeptical': (2, 8),
    }
    
    mle = n_heads / (n_heads + n_tails)
    
    for name, (alpha, beta) in priors.items():
        post_alpha = alpha + n_heads
        post_beta = beta + n_tails
        post_mean = post_alpha / (post_alpha + post_beta)
        print(f"{name}: posterior mean = {post_mean:.4f}")
```

**보기 4.** <span class="diff easy" title="쉬움"></span> 앞면 $15$ 번, 뒷면 $5$ 번을 보았을 때 사전분포를 달리하면 사후평균이 어떻게 달라지는지 견주시오.

??? success "풀이"

    최대 우도는 $15/20 = 0.75$ 이다. 정리 5를 쓰면 다음과 같다.

    | 사전분포 | 유사 관측 $\alpha+\beta$ | 사후평균 |
    |--------|------|------------|
    | $\text{Beta}(1,1)$ | $2$ | $16/22 \approx 0.727$ |
    | $\text{Beta}(0.5,0.5)$ | $1$ | $15.5/21 \approx 0.738$ |
    | $\text{Beta}(2,2)$ | $4$ | $17/24 \approx 0.708$ |
    | $\text{Beta}(10,10)$ | $20$ | $25/40 = 0.625$ |
    | $\text{Beta}(2,8)$ | $10$ | $17/30 \approx 0.567$ |

    유사 관측이 데이터 $20$ 번에 비교하여 작을수록 사후평균이 $0.75$ 에 가깝다. 회의적인 $\text{Beta}(2,8)$ 은 $0.2$ 쪽으로 끌어당기는 힘이 세어 답이 크게 달라진다.

**문제 3.** <span class="diff hard" title="어려움"></span> 사전분포과 데이터가 정면으로 부딪칠 때 사후분포가 어떤 모양이 되는지 밝히시오. 이를테면 $\text{Beta}(20,2)$ 사전분포에 $n=20$, $k=2$ 인 데이터를 주면 어떻게 되는가?

??? success "풀이"

    사후분포는 $\text{Beta}(22, 20)$ 으로 평균이 $22/42 \approx 0.524$ 이다. 사전평균 $20/22\approx 0.909$ 와 최대 우도 $2/20 = 0.1$ 의 **가운데 어딘가**이며, 어느 쪽도 지지하지 않는 값이다.

    이 사례의 갱신 파라메터는 둘 다 1보다 크므로 사후분포는 내부에 하나의 최빈값을 갖는다. 베타분포 전체가 단봉인 것은 아니며 두 파라메터가 1보다 작으면 U자 모양도 가능하다. 사전분포와 자료의 불일치는 모형 점검의 계기지만, 두 개의 분리된 가설이 실제로 존재한다는 뜻은 아니다. 다봉 구조가 필요하면 혼합분포 등의 모형을 명시적으로 고려한다.

## 6. 정규-정규 켤레 모델

### 정리 6. 정밀도로 가중한 평균 — 정밀도는 더해진다 { .thm }

$x_1,\dots,x_n$이 μ를 조건으로 독립인 $\mathcal{N}(\mu,\sigma^2)$ 관측이고 $\sigma$ 를 알고 사전분포가 $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$ 이면 사후분포는 정규이고, 정밀도 $\tau = 1/\sigma^2$ 로 적을 때

$$
\tau_n = \tau_0 + n\tau,
\qquad
\mu_n = \frac{\tau_0\,\mu_0 + n\tau\,\bar x}{\tau_0 + n\tau}
$$

이다. 곧 **정밀도는 더해지고, 평균은 정밀도로 가중한 평균이 된다**.

??? proof "증명"

    로그를 잡고 $\mu$ 와 얽힌 항만 남기면

    $$
    \log p(\mu \mid D) = -\frac{\tau_0}{2}(\mu-\mu_0)^2 - \frac{n\tau}{2}(\mu-\bar x)^2 + \text{const}
    $$

    이다. $\mu$ 에 대해 펼쳐 이차항과 일차항을 모으면

    $$
    -\frac{\tau_0 + n\tau}{2}\,\mu^2 + (\tau_0\mu_0 + n\tau\bar x)\,\mu + \text{const}
    $$

    이다. 이는 정밀도가 $\tau_0 + n\tau$ 이고 평균이 $(\tau_0\mu_0+n\tau\bar x)/(\tau_0+n\tau)$ 인 정규분포의 로그밀도와 같은 꼴이다. 완전제곱으로 묶으면 곧바로 결론이 나온다.

!!! note "쓰임새"
    정밀도로 생각하면 베이즈 갱신이 **정보를 더하는 일**로 보인다. 사전분포가 $\tau_0$ 만큼, 데이터가 $n\tau$ 만큼 정보를 들고 오고, 사후분포는 그 합만큼을 갖는다. 데이터가 늘수록 $n\tau$ 가 커져 사전분포의 몫이 줄어든다.

```python
def normal_normal_inference(data, prior_mean, prior_std, known_std):
    """
    흩어짐을 아는 정규 평균에 대한 베이즈 추론.
    
    파라메터
    ----------
    data : array-like
        관측된 자료점
    prior_mean : float
        앞확률 분포의 평균
    prior_std : float
        앞확률의 표준편차
    known_std : float
        아는 자료의 표준편차
    
    반환값
    -------
    posterior_dist : scipy.stats.norm
        뒤확률 정규 분포
    """
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    
    # 정밀도
    prior_precision = 1 / prior_std**2
    data_precision = n / known_std**2
    posterior_precision = prior_precision + data_precision
    
    # 뒤확률 파라메터
    posterior_mean = (prior_precision * prior_mean + 
                      data_precision * sample_mean) / posterior_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    return stats.norm(posterior_mean, posterior_std)
```

**보기 5.** <span class="diff easy" title="쉬움"></span> $\sigma = 1$ 인 관측 $n = 4$ 개의 평균이 $\bar x = 3$ 이고 사전분포가 $\mathcal{N}(0, 1)$ 일 때 사후분포를 구하시오.

??? success "풀이"

    $\tau_0 = 1$, $\tau = 1$ 이므로 정리 6에서

    $$
    \tau_4 = 1 + 4 = 5, \qquad
    \mu_4 = \frac{1\cdot 0 + 4\cdot 3}{5} = 2.4
    $$

    이다. 곧 사후분포는 $\mathcal{N}(2.4,\ 1/5)$ 이다. 사전분포가 관측 한 번의 웨이트를 가지므로 표본 평균 $3$ 이 $0$ 쪽으로 $1/5$ 만큼 끌려갔다.

## 7. 순차 갱신과 점근적 성질

### 정리 7. 순차 갱신의 순서 무관성 — 한꺼번에 갱신한 것과 같다 { .thm }

관측이 $\theta$ 가 주어졌을 때 서로 독립이면, 데이터를 하나씩 넣으며 사후분포를 거듭 고쳐도 마지막 결과는 모두를 한꺼번에 넣은 것과 같고, 넣는 **순서에도 달리지 않는다**.

$$
p(\theta \mid x_1, x_2) \;\propto\; p(x_2 \mid \theta)\,p(\theta \mid x_1)
\;\propto\; p(x_1\mid\theta)\,p(x_2\mid\theta)\,p(\theta)
$$

??? proof "증명"

    $x_1$ 을 넣은 사후분포는 $p(\theta\mid x_1) \propto p(x_1\mid\theta)p(\theta)$ 이다. 이를 사전분포로 삼아 $x_2$ 를 넣으면

    $$
    p(\theta \mid x_1,x_2) \propto p(x_2 \mid \theta,x_1)\,p(\theta\mid x_1)
    = p(x_2\mid\theta)\,p(\theta\mid x_1)
    $$

    이다. 가운데 등식에서 조건부 독립을 썼다. 여기에 $p(\theta\mid x_1)$ 의 꼴을 넣으면

    $$
    p(\theta\mid x_1,x_2) \propto p(x_1\mid\theta)\,p(x_2\mid\theta)\,p(\theta)
    $$

    이고, 이 식은 $x_1$ 과 $x_2$ 를 맞바꾸어도 그대로이므로 순서에 달리지 않는다.

!!! note "쓰임새"
    이 성질 덕분에 데이터를 모두 쌓아 두지 않고 **흘려 보내며** 추론할 수 있다. 켤레 모델에서는 파라메터 몇 개만 들고 있으면 되므로 메모리가 데이터 양과 무관하다. 정리 5의 베타-이항이라면 $(\alpha,\beta)$ 두 수가 전부다.

```python
def sequential_beta_binomial(flip_sequence, prior_alpha=1, prior_beta=1):
    """
    베타-이항 모형으로 차례대로 베이즈 새로 고치기.
    
    파라메터
    ----------
    flip_sequence : str의 list
        'H'과 'T' 관측의 늘어놓음
    prior_alpha, prior_beta : float
        첫 앞확률 파라메터
    
    반환값
    -------
    alpha_history, beta_history : list
        새로 고침에 따른 파라메터의 흐름
    """
    alpha_history = [prior_alpha]
    beta_history = [prior_beta]
    
    current_alpha = prior_alpha
    current_beta = prior_beta
    
    for flip in flip_sequence:
        if flip == 'H':
            current_alpha += 1
        else:
            current_beta += 1
        
        alpha_history.append(current_alpha)
        beta_history.append(current_beta)
    
    return alpha_history, beta_history
```

**보기 6.** <span class="diff easy" title="쉬움"></span> 위 함수에서 앞면과 뒷면의 순서를 뒤섞어도 마지막 $(\alpha,\beta)$ 가 같음을 확인하시오.

??? success "풀이"

    함수는 H이면 current_alpha를, 그 외 모든 입력이면 current_beta를 1씩 올린다. 따라서 사용할 입력을 H·T로 제한해야 한다. 마지막 값은 앞면의 수와 뒷면의 수에만 달리고 순서에는 달리지 않는다. 이는 정리 7의 결론을 코드로 그대로 옮긴 것이며, 정리 3이 말한 "성공 횟수가 충분 통계량"이라는 사실과도 맞물린다.

**문제 4.** <span class="diff med" title="중간"></span> $n \to \infty$ 일 때 사후분포가 어떻게 되는지 정리 6의 식으로 밝히고, 사전분포가 언제 씻겨 나가지 않는지 말하시오.

??? success "풀이"

    정리 6에서 $\tau_n = \tau_0 + n\tau \to \infty$ 이므로 사후분포의 분산 $1/\tau_n \to 0$ 이고,

    $$
    \mu_n = \frac{\tau_0\mu_0 + n\tau\bar x}{\tau_0+n\tau} \longrightarrow \bar x
    $$

    이다. 정확한 독립 정규 관측 모형과 고정된 유한 사전 정밀도 아래 μ_n−표본평균은 0으로 가고, 표본평균이 참평균에 수렴하므로 사후분포도 참평균 주변으로 집중한다. 집중 자체와, 적절히 중심화·스케일한 사후분포가 정규분포에 가까워진다는 베른슈타인–폰 미제스 정리는 구분해야 한다.

    다만 참값의 모든 주변에 사전확률이 양수여야 한다는 등의 지지집합 조건이 중요하다. 연속 분포에서 한 점의 확률이 0이라는 사실은 문제가 아니다. 참값 주변의 영역 전체를 사전분포가 배제하면 그 영역은 갱신 후에도 배제된다. 모형 오지정이나 식별성 문제도 일관성을 방해할 수 있다.

## 8. 전체 구현 예제

전체 코드는 1차원 격자 사후분포, 요약 통계와 구간 근사를 결합한다. 정규 예제의 범위는 사전평균 ±5 표준편차로 고정되고 베타 예제는 [0.001,0.999]를 사용한다. 따라서 범위 밖 사후 질량이 크면 정규화된 결과도 부정확하다. likelihood의 직접 곱은 언더플로를 일으킬 수 있으며 로그에 더하는 작은 상수도 정확한 0을 보존하지 않는다.

구간 함수는 균등 격자를 가정한다. hpd_interval은 높은 밀도의 격자점들을 선택한 뒤 최소·최대값을 반환하므로 다봉분포의 분리된 HPD 집합을 하나의 구간으로 메울 수 있다. 그림 함수는 각 곡선의 최대값을 1로 조정하므로 곡선의 높이가 서로 비교 가능한 정규화 밀도는 아니다. compute_evidence의 n_points 인자는 실제 적분에 사용되지 않는다.

출력 예시는 아래에 보존했다. 반복 시연은 서로 다른 난수 표본을 사용하므로 표본 수에 따른 변화가 한 중첩 표본 경로를 의미하지는 않는다.

```python
"""
앞확률, 가능도, 뒤확률: 베이즈의 핵심 장치

이 모듈은 베이즈 추론의 근본 구성 요소를 그려 보기와 실전 보여 주기와 함께
구현한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# 핵심 클래스
# =============================================================================

@dataclass
class BayesianModel:
    """
    앞확률과 가능도를 가진 베이즈 모형을 감싼다.
    
    속성
    ----------
    prior : callable
        앞확률 밀도 p(θ)
    likelihood : callable
        가능도 함수 p(D|θ)
    theta_range : tuple
        파라메터 θ의 쓸 수 있는 범위
    """
    prior: Callable[[np.ndarray], np.ndarray]
    likelihood: Callable[[np.ndarray, np.ndarray], np.ndarray]
    theta_range: Tuple[float, float]
    
    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """로그 앞확률 밀도."""
        return np.log(self.prior(theta) + 1e-300)
    
    def log_likelihood(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """로그 가능도."""
        return np.log(self.likelihood(theta, data) + 1e-300)
    
    def unnormalized_posterior(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """고르게 하지 않은 뒤확률: 가능도 × 앞확률."""
        return self.likelihood(theta, data) * self.prior(theta)
    
    def log_unnormalized_posterior(self, theta: np.ndarray, data: np.ndarray) -> np.ndarray:
        """고르게 하지 않은 로그 뒤확률."""
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def compute_evidence(self, data: np.ndarray, n_points: int = 1000) -> float:
        """
        수치 적분으로 증거 p(D)을 셈한다.
        
        파라메터
        ----------
        data : array
            관측 자료
        n_points : int
            구적점의 개수
        
        반환값
        -------
        float
            증거(주변 가능도)
        """
        def integrand(theta):
            return self.unnormalized_posterior(np.array([theta]), data)[0]
        
        evidence, _ = quad(integrand, self.theta_range[0], self.theta_range[1])
        return evidence
    
    def posterior_grid(
        self, 
        data: np.ndarray, 
        n_points: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        격자 위에서 고르게 한 뒤확률을 셈한다.
        
        반환값
        -------
        theta_grid : array
            파라메터 값의 격자
        posterior : array
            격자점마다 고르게 한 뒤확률 밀도
        """
        theta_grid = np.linspace(self.theta_range[0], self.theta_range[1], n_points)
        unnorm_post = self.unnormalized_posterior(theta_grid, data)
        
        # 사다리꼴 적분으로 고르게 하기
        evidence = np.trapz(unnorm_post, theta_grid)
        posterior = unnorm_post / evidence
        
        return theta_grid, posterior

# =============================================================================
# 뒤확률 간추림
# =============================================================================

def posterior_mean(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """뒤확률 평균 E[θ|D]을 셈한다."""
    return np.trapz(theta_grid * posterior, theta_grid)

def posterior_variance(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """뒤확률 흩어짐 Var[θ|D]을 셈한다."""
    mean = posterior_mean(theta_grid, posterior)
    return np.trapz((theta_grid - mean)**2 * posterior, theta_grid)

def posterior_mode(theta_grid: np.ndarray, posterior: np.ndarray) -> float:
    """뒤확률 최빈값(MAP 어림값)을 셈한다."""
    return theta_grid[np.argmax(posterior)]

def credible_interval(
    theta_grid: np.ndarray, 
    posterior: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    양끝이 같은 믿음 구간을 셈한다.
    
    파라메터
    ----------
    alpha : float
        유의수준(95% 구간이면 기본값 0.05)
    
    반환값
    -------
    tuple
        믿음 구간의 (아래, 위) 경계
    """
    # 누적분포함수 셈하기
    cdf = np.cumsum(posterior) * (theta_grid[1] - theta_grid[0])
    cdf = cdf / cdf[-1]  # 1에서 끝나도록 보장
    
    # 분위수 찾기
    lower_idx = np.searchsorted(cdf, alpha / 2)
    upper_idx = np.searchsorted(cdf, 1 - alpha / 2)
    
    return theta_grid[lower_idx], theta_grid[min(upper_idx, len(theta_grid)-1)]

def hpd_interval(
    theta_grid: np.ndarray, 
    posterior: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    최고 뒤확률 밀도(HPD) 구간을 셈한다.
    
    (1-alpha)의 확률 질량을 담는 가장 짧은 구간.
    """
    # 밀도로 정렬(내림차순)
    sorted_indices = np.argsort(posterior)[::-1]
    sorted_posterior = posterior[sorted_indices]
    sorted_theta = theta_grid[sorted_indices]
    
    # (1-alpha) 질량에 이를 때까지 쌓기
    cumsum = np.cumsum(sorted_posterior) * (theta_grid[1] - theta_grid[0])
    cutoff_idx = np.searchsorted(cumsum, 1 - alpha)
    
    # HPD 구역은 밀도가 임계값을 넘는 모든 theta이다
    hpd_theta = sorted_theta[:cutoff_idx+1]
    
    return hpd_theta.min(), hpd_theta.max()

# =============================================================================
# 보기: 이항-베타 모형
# =============================================================================

def create_beta_binomial_model(alpha_prior: float, beta_prior: float) -> BayesianModel:
    """
    베타-이항 베이즈 모형을 만든다.
    
    앞확률: θ ~ Beta(α, β)
    가능도: k | θ ~ 이항(n, θ)
    """
    def prior(theta):
        return stats.beta.pdf(theta, alpha_prior, beta_prior)
    
    def likelihood(theta, data):
        k, n = data  # 시도 n번에서 성공 k번
        return stats.binom.pmf(k, n, theta)
    
    return BayesianModel(
        prior=prior,
        likelihood=likelihood,
        theta_range=(0.001, 0.999)
    )

# =============================================================================
# 보기: 가우스 모형(흩어짐을 아는 경우)
# =============================================================================

def create_gaussian_model(
    prior_mean: float, 
    prior_var: float, 
    known_var: float
) -> BayesianModel:
    """
    흩어짐을 아는 가우스 베이즈 모형을 만든다.
    
    앞확률: μ ~ N(μ₀, σ₀²)
    가능도: x | μ ~ N(μ, σ²), 여기서 σ²은 안다
    """
    def prior(mu):
        return stats.norm.pdf(mu, prior_mean, np.sqrt(prior_var))
    
    def likelihood(mu, data):
        # 가우스 가능도의 곱
        result = np.ones_like(mu, dtype=float)
        for x in data:
            result *= stats.norm.pdf(x, mu, np.sqrt(known_var))
        return result
    
    # 앞확률 ± 표준편차 4배에 자료 범위를 더해 범위 잡기
    return BayesianModel(
        prior=prior,
        likelihood=likelihood,
        theta_range=(prior_mean - 5*np.sqrt(prior_var), 
                     prior_mean + 5*np.sqrt(prior_var))
    )

# =============================================================================
# 시각화
# =============================================================================

def plot_bayesian_update(
    model: BayesianModel,
    data: np.ndarray,
    true_theta: Optional[float] = None,
    title: str = "Bayesian Update"
):
    """
    앞확률, 가능도, 뒤확률을 그려 본다.
    """
    theta_grid = np.linspace(model.theta_range[0], model.theta_range[1], 1000)
    
    # 구성 요소 셈하기
    prior_vals = model.prior(theta_grid)
    likelihood_vals = model.likelihood(theta_grid, data)
    _, posterior_vals = model.posterior_grid(data)
    
    # 그려 보려고 고르게 하기
    prior_vals = prior_vals / np.max(prior_vals)
    likelihood_vals = likelihood_vals / np.max(likelihood_vals)
    posterior_vals = posterior_vals / np.max(posterior_vals)
    
    # 그림
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(theta_grid, prior_vals, alpha=0.3, color='blue', label='Prior')
    ax.plot(theta_grid, prior_vals, 'b-', linewidth=2)
    
    ax.fill_between(theta_grid, likelihood_vals, alpha=0.3, color='green', label='Likelihood')
    ax.plot(theta_grid, likelihood_vals, 'g-', linewidth=2)
    
    ax.fill_between(theta_grid, posterior_vals, alpha=0.3, color='red', label='Posterior')
    ax.plot(theta_grid, posterior_vals, 'r-', linewidth=2)
    
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle='--', linewidth=2, 
                   label=f'True θ = {true_theta}')
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density (normalized)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_posterior_summaries(
    theta_grid: np.ndarray,
    posterior: np.ndarray,
    true_theta: Optional[float] = None
):
    """
    점 어림값과 믿음 구간을 곁들여 뒤확률을 그려 본다.
    """
    mean = posterior_mean(theta_grid, posterior)
    mode = posterior_mode(theta_grid, posterior)
    var = posterior_variance(theta_grid, posterior)
    ci = credible_interval(theta_grid, posterior, 0.05)
    hpd = hpd_interval(theta_grid, posterior, 0.05)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 뒤확률
    ax.fill_between(theta_grid, posterior, alpha=0.4, color='steelblue')
    ax.plot(theta_grid, posterior, 'b-', linewidth=2, label='Posterior')
    
    # 점 어림값
    ax.axvline(mean, color='red', linestyle='-', linewidth=2, 
               label=f'Mean = {mean:.3f}')
    ax.axvline(mode, color='orange', linestyle='--', linewidth=2, 
               label=f'Mode (MAP) = {mode:.3f}')
    
    # 믿음 구간
    ax.axvspan(ci[0], ci[1], alpha=0.2, color='green', 
               label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    
    # 참값
    if true_theta is not None:
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2,
                   label=f'True θ = {true_theta}')
    
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title(f'Posterior Summary (σ = {np.sqrt(var):.3f})', fontsize=14)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# 보여 주기: 앞확률과 자료의 효과
# =============================================================================

def demonstrate_prior_data_tradeoff():
    """
    뒤확률이 앞확률과 자료의 균형을 어떻게 잡는지 보인다.
    """
    np.random.seed(42)
    
    # 참 파라메터
    true_theta = 0.7
    
    # 데이터를 생성한다
    n_obs = 10
    data = np.random.binomial(1, true_theta, n_obs)
    k = data.sum()  # 성공
    
    print(f"Data: {k} successes in {n_obs} trials")
    print(f"MLE: {k/n_obs:.3f}")
    print()
    
    # 서로 다른 앞확률
    priors = [
        ("Uniform (α=1, β=1)", 1, 1),
        ("Skeptical (α=2, β=8)", 2, 8),  # 앞확률은 낮은 θ을 믿는다
        ("Confident (α=8, β=2)", 8, 2),  # 앞확률은 높은 θ을 믿는다
        ("Strong (α=20, β=20)", 20, 20),  # 0.5에 놓인 센 앞확률
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, (name, alpha, beta) in zip(axes.flat, priors):
        model = create_beta_binomial_model(alpha, beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n_obs]))
        
        # 앞확률 그리기
        prior_vals = stats.beta.pdf(theta_grid, alpha, beta)
        prior_vals = prior_vals / np.max(prior_vals)
        ax.plot(theta_grid, prior_vals, 'b--', linewidth=2, label='Prior', alpha=0.7)
        
        # 뒤확률 그리기
        posterior_norm = posterior / np.max(posterior)
        ax.fill_between(theta_grid, posterior_norm, alpha=0.4, color='red')
        ax.plot(theta_grid, posterior_norm, 'r-', linewidth=2, label='Posterior')
        
        # 참값과 MLE
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2, label=f'True θ')
        ax.axvline(k/n_obs, color='green', linestyle='--', linewidth=2, label='MLE')
        
        # 뒤확률 평균
        post_mean = posterior_mean(theta_grid, posterior)
        ax.axvline(post_mean, color='red', linestyle=':', linewidth=1.5)
        
        ax.set_title(f'{name}\nPosterior mean = {post_mean:.3f}', fontsize=11)
        ax.set_xlabel('θ')
        ax.set_ylabel('Density (normalized)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prior-Data Tradeoff ({k}/{n_obs} successes, true θ = {true_theta})', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('prior_data_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()

def demonstrate_data_overwhelming_prior():
    """
    자료가 늘어나면 앞확률을 어떻게 눌러 버리는지 보인다.
    """
    np.random.seed(42)
    
    true_theta = 0.7
    
    # 엉뚱한 자리의 센 앞확률
    prior_alpha, prior_beta = 10, 40  # 앞확률 평균 = 0.2
    
    sample_sizes = [1, 5, 20, 100, 500]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, n in zip(axes, sample_sizes):
        # 데이터를 생성한다
        data = np.random.binomial(1, true_theta, n)
        k = data.sum()
        
        model = create_beta_binomial_model(prior_alpha, prior_beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n]))
        
        # 앞확률
        prior_vals = stats.beta.pdf(theta_grid, prior_alpha, prior_beta)
        ax.plot(theta_grid, prior_vals / np.max(prior_vals), 'b--', 
                linewidth=2, label='Prior', alpha=0.7)
        
        # 뒤확률
        ax.fill_between(theta_grid, posterior / np.max(posterior), 
                        alpha=0.4, color='red')
        ax.plot(theta_grid, posterior / np.max(posterior), 'r-', 
                linewidth=2, label='Posterior')
        
        # 참값
        ax.axvline(true_theta, color='black', linestyle=':', linewidth=2)
        
        post_mean = posterior_mean(theta_grid, posterior)
        ax.set_title(f'n = {n}\nPost. mean = {post_mean:.3f}')
        ax.set_xlabel('θ')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    
    plt.suptitle(f'Data Overwhelming Prior (Prior mean = 0.2, True θ = {true_theta})', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('data_overwhelming_prior.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Prior: Beta(10, 40) with mean 0.2")
    print(f"True θ: {true_theta}")
    print("\nAs n increases, posterior mean approaches true value:")
    for n in sample_sizes:
        data = np.random.binomial(1, true_theta, n)
        k = data.sum()
        model = create_beta_binomial_model(prior_alpha, prior_beta)
        theta_grid, posterior = model.posterior_grid(np.array([k, n]))
        print(f"  n = {n:4d}: posterior mean = {posterior_mean(theta_grid, posterior):.4f}")

# =============================================================================
# 주된 보여 주기
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEMONSTRATION: PRIOR, LIKELIHOOD, AND POSTERIOR")
    print("=" * 60)
    
    print("\n1. Basic Bayesian Update (Beta-Binomial)")
    print("-" * 40)
    
    # 단순한 보기
    model = create_beta_binomial_model(alpha_prior=2, beta_prior=2)
    data = np.array([7, 10])  # 시도 10번에서 성공 7번
    
    theta_grid, posterior = model.posterior_grid(data)
    
    print(f"Prior: Beta(2, 2)")
    print(f"Data: 7 successes in 10 trials")
    print(f"Posterior: Beta(9, 5)")  # 켤레 새로 고치기
    print(f"\nPosterior summaries:")
    print(f"  Mean: {posterior_mean(theta_grid, posterior):.4f}")
    print(f"  Mode: {posterior_mode(theta_grid, posterior):.4f}")
    print(f"  Std:  {np.sqrt(posterior_variance(theta_grid, posterior)):.4f}")
    print(f"  95% CI: {credible_interval(theta_grid, posterior)}")
    
    # 증거
    evidence = model.compute_evidence(data)
    print(f"  Evidence p(D): {evidence:.6f}")
    
    print("\n2. Effect of Different Priors")
    print("-" * 40)
    demonstrate_prior_data_tradeoff()
    print("See: prior_data_tradeoff.png")
    
    print("\n3. Data Overwhelming Prior")
    print("-" * 40)
    demonstrate_data_overwhelming_prior()
    print("See: data_overwhelming_prior.png")
```

**출력:**

```
============================================================
DEMONSTRATION: PRIOR, LIKELIHOOD, AND POSTERIOR
============================================================

1. Basic Bayesian Update (Beta-Binomial)
----------------------------------------
Prior: Beta(2, 2)
Data: 7 successes in 10 trials
Posterior: Beta(9, 5)

Posterior summaries:
  Mean: 0.6429
  Mode: 0.6663
  Std:  0.1237
  95% CI: (0.38561461461461466, 0.8611381381381382)
  Evidence p(D): 0.111888

2. Effect of Different Priors
----------------------------------------
Data: 6 successes in 10 trials
MLE: 0.600

See: prior_data_tradeoff.png

3. Data Overwhelming Prior
----------------------------------------
Prior: Beta(10, 40) with mean 0.2
True θ: 0.7

As n increases, posterior mean approaches true value:
  n =    1: posterior mean = 0.2157
  n =    5: posterior mean = 0.2182
  n =   20: posterior mean = 0.4000
  n =  100: posterior mean = 0.5533
  n =  500: posterior mean = 0.6436
See: data_overwhelming_prior.png
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 강한 사전분포과 약한 사전분포가 같은 데이터에서 어떻게 다른 사후분포를 주는지 정리 5로 설명하시오.

</div>

??? success "풀이"

    정리 5에서 사후평균의 웨이트는 $\dfrac{\alpha+\beta}{\alpha+\beta+n}$ 와 $\dfrac{n}{\alpha+\beta+n}$ 이다. 강한 사전분포는 $\alpha+\beta$ 가 커서 첫 웨이트가 크고, 약한 사전분포는 작아서 둘째 웨이트가 크다.

    $n$ 을 고정하고 $\alpha+\beta$ 를 키우면 사후평균이 사전평균 쪽으로 매끄럽게 움직인다. 보기 4의 표가 그 흐름을 수로 보여 준다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 사전분포 민감도 분석이란 무엇이며 왜 해야 하는지 밝히시오.

</div>

??? success "풀이"

    여러 개의 합리적인 사전분포를 사용해 관심 결론이 얼마나 달라지는지 비교한다. 결과가 비슷하더라도 시도한 사전분포의 범위가 좁았을 수 있다. 큰 차이는 자료가 해당 관심량을 충분히 식별하지 못하거나 사전분포와 자료가 충돌한다는 신호일 수 있으므로 모형 점검과 함께 해석한다.

    문제 3처럼 사전분포과 데이터가 부딪치는 경우에는 반드시 해야 한다. 알릴 때도 하나의 사후분포가 아니라 사전분포 몇 가지에서 나온 결과를 나란히 보이는 것이 정직하다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 순차 갱신에서 사후분포가 참값으로 모여드는 모습을 시뮬레이션으로 확인하시오.

</div>

??? success "풀이"

    참값 $\theta^\star$ 로 베르누이 표본을 길게 만들고 `sequential_beta_binomial` 을 돌려 $(\alpha_t,\beta_t)$ 의 흐름을 얻은 뒤, 각 걸음의 사후평균 $\alpha_t/(\alpha_t+\beta_t)$ 와 표준편차를 그린다.

    고정된 양의 베타 사전 파라메터와 독립 베르누이 관측, 내부 참값 0<θ⋆<1을 가정하면 사후평균이 참값에 수렴하고 사후 표준편차는 대략 t의 제곱근에 반비례한다. 한 시뮬레이션 경로에서 오차가 매 단계 감소하는 것은 아니며 경계 참값에서는 같은 정규 근사를 그대로 적용하지 않는다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> 정규 모델에서 평균과 분산을 **둘 다** 모를 때의 켤레 사전분포인 정규-역감마를 적고, 사후분포의 파라메터를 이끌어 내시오.

</div>

??? success "풀이"

    켤레 사전분포는 $\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0)$ 이고 그 조건 아래 $\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0,\ \sigma^2/\kappa_0)$ 이다. 사후분포도 같은 족이며 파라메터는 다음과 같다.

    $$
    \kappa_n = \kappa_0 + n, \qquad
    \mu_n = \frac{\kappa_0\mu_0 + n\bar x}{\kappa_0+n}, \qquad
    a_n = a_0 + \frac n2
    $$

    $$
    b_n = b_0 + \frac12\sum_i (x_i-\bar x)^2 + \frac{\kappa_0 n(\bar x - \mu_0)^2}{2(\kappa_0+n)}
    $$

    $\mu_n$ 이 정리 6과 같은 가중 평균 꼴임에 주목하라. $b_n$ 의 마지막 항은 사전평균과 표본 평균이 얼마나 어긋났는지에 대한 벌이며, 둘이 멀수록 분산을 크게 잡는다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 실제 데이터를 하나 골라 사전분포를 세우고, 사후분포를 셈한 뒤, 사전예측분포로 모델을 시험하시오.

</div>

??? success "풀이"

    절차는 이렇다. (1) 분야 지식으로 사전분포를 세운다. (2) 정리 2로 사전예측분포를 뽑아 실제로 볼 법한 데이터가 나오는지 본다. 여기서 어긋나면 사전분포로 돌아간다. (3) 데이터를 넣어 사후분포를 얻는다. (4) 연습문제 2의 민감도 분석을 한다. (5) 사후예측분포를 실제 데이터와 비교하여 모델이 맞는지 다시 본다.

    (2)와 (5)를 건너뛰고 사후분포만 알리는 것이 가장 흔한 잘못이다.

## 정리하며

사전분포와 우도의 곱을 정규화하면 사후분포를 얻는다. 주변 우도는 모델 비교에서 중요하며 적절한 사전분포와 동일한 관측 자료가 필요하다. 베타-이항은 성공·실패 수로, 알려진 분산의 정규 모형은 정밀도로 간단히 갱신된다. 조건부 독립 모형의 순차 갱신은 일괄 갱신과 같지만, 사후 일관성과 수치 근사의 정확성은 별도의 조건을 요구한다.

점 추정은 [MAP 추정](map_estimation.md), 구간 요약은 [베이즈 신용구간](credible_intervals.md), 비켤레 추론의 예는 [깁스 샘플링](../../ch19/mcmc/gibbs_sampling.md)을 참고한다.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 1~3장
- Bishop, C. *Pattern Recognition and Machine Learning*, 2장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3~5장
