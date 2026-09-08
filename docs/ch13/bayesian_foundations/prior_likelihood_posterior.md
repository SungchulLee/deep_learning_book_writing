# 앞확률, 가능도, 뒤확률

베이즈 추론은 앞확률·가능도·뒤확률·증거라는 네 조각으로 이루어진다. 이 마당은 네 조각을 하나씩 뜻매김하고, 이들이 베이즈 정리로 어떻게 묶이는지 밝힌 뒤, 켤레 앞확률을 쓰는 두 대표 모형(베타-이항과 정규-정규)에서 뒤확률을 손으로 끝까지 셈해 본다.

## 1. 베이즈 정리: 네 조각

**정의 1.** [앞확률, 가능도, 뒤확률, 증거]

매개변수 $\theta$ 와 데이터 $D$ 에 대해 다음 네 가지를 뜻매김한다.

- **앞확률** $p(\theta)$ — 데이터를 보기 앞의 믿음
- **가능도** $p(D \mid \theta)$ — $\theta$ 가 참일 때 이 데이터가 나올 그럴듯함
- **뒤확률** $p(\theta \mid D)$ — 데이터를 본 뒤로 고쳐 잡은 믿음
- **증거** $p(D)$ — 앞확률에 대해 평균 낸 가능도, 곧 주변 가능도

### 정리 1. 베이즈 정리 — 뒤확률은 가능도와 앞확률의 곱에 비례한다 { .thm }

$p(D) > 0$ 이면 다음이 성립한다.

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

    조건부확률의 뜻매김에서 두 가지로 결합밀도를 적으면

    $$
    p(\theta, D) = p(\theta \mid D)\,p(D) = p(D \mid \theta)\,p(\theta)
    $$

    이다. $p(D) > 0$ 이므로 양변을 $p(D)$ 로 나누면 첫 식을 얻는다.

    둘째 식은 결합밀도를 $\theta$ 에 대해 주변화한 것이다.

    $$
    p(D) = \int p(\theta, D)\,d\theta = \int p(D \mid \theta)\,p(\theta)\,d\theta
    $$

!!! note "쓰임새"
    비례 관계만으로도 뒤확률의 **모양**은 완전히 정해진다. 고르개 상수 $p(D)$ 는 마지막에 적분 한 번으로 채워 넣으면 된다. 최대 뒤확률처럼 최빈값만 필요하면 아예 셈하지 않아도 된다.

## 2. 앞확률 분포

**정의 2.** [정보를 담은 앞확률과 담지 않은 앞확률]

앞확률이 어떤 값 쪽으로 뚜렷이 무게를 실으면 **정보를 담은 앞확률**, 되도록 판단을 미루면 **정보를 담지 않은 앞확률**이라 한다. 뒤확률이 앞확률과 같은 분포족에 남게 하는 앞확률은 **켤레 앞확률**이다.

| 갈래 | 보기 | 쓸 때 |
|------|------|-------|
| 고른 앞확률 | $\text{Beta}(1,1)$, $p(\mu)\propto 1$ | 아는 바가 거의 없을 때 |
| 제프리스 앞확률 | $\text{Beta}(0.5,0.5)$ | 매개변수 바꾸기에 흔들리지 않기를 바랄 때 |
| 약한 정보 앞확률 | $\text{Beta}(2,2)$ | 극단값만 눌러 두고 싶을 때 |
| 센 정보 앞확률 | $\text{Beta}(10,10)$ | 앞선 연구가 넉넉할 때 |

### 정리 2. 앞확률 예측 분포 — 관측 앞의 데이터 분포 { .thm }

데이터를 보기 전에 어떤 데이터가 나올지 묻는 분포는 가능도를 앞확률로 평균 낸 것이며, 이는 곧 증거이다.

$$
p(D) = \int p(D \mid \theta)\,p(\theta)\,d\theta
$$

베타-이항 모형에서 $n$ 번 중 $k$ 번 성공할 앞확률 예측 확률은 베타-이항 분포

$$
p(k) = \binom{n}{k}\,\frac{B(\alpha+k,\ \beta+n-k)}{B(\alpha,\beta)}
$$

이다. 여기서 $B$ 는 베타 함수이다.

??? proof "증명"

    이항 가능도와 베타 앞확률을 넣으면

    $$
    p(k) = \int_0^1 \binom{n}{k}\theta^{k}(1-\theta)^{n-k}\,
    \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}\,d\theta
    $$

    이다. 적분 안을 모으면 $\theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$ 이고, 베타 함수의 뜻매김

    $$
    \int_0^1 \theta^{a-1}(1-\theta)^{b-1}\,d\theta = B(a,b)
    $$

    를 쓰면 곧바로 결론이 나온다.

!!! note "쓰임새"
    앞확률 예측 분포는 **모형을 시험하는 자리**다. 여기서 나온 분포가 실제로 볼 법한 데이터와 동떨어져 있다면 앞확률이 잘못 잡힌 것이다.

**보기 1.** <span class="diff easy" title="쉬움"></span> $\text{Beta}(1,1)$ 앞확률에서 동전을 $n$ 번 던질 때 앞면이 $k$ 번 나올 앞확률 예측 확률을 구하시오.

??? success "풀이"

    $\alpha=\beta=1$ 이므로 $B(1,1)=1$ 이고

    $$
    p(k) = \binom{n}{k} B(1+k,\ 1+n-k)
    = \binom{n}{k}\frac{k!\,(n-k)!}{(n+1)!} = \frac{1}{n+1}
    $$

    이다. 곧 $k = 0,1,\dots,n$ 이 모두 **똑같이 그럴듯하다**. 고른 앞확률이 뜻하는 바가 "성공 횟수에 대해 아무 선호도 없다"임을 보여 준다.

**문제 1.** <span class="diff med" title="중간"></span> 제프리스 앞확률 $\text{Beta}(0.5, 0.5)$ 가 매개변수 바꾸기에 흔들리지 않는다는 것이 무슨 뜻인지 설명하고, 고른 앞확률이 왜 그 성질을 갖지 못하는지 보이시오.

??? success "풀이"

    $\theta$ 에 고른 앞확률을 두면 $\eta = \log\frac{\theta}{1-\theta}$ 같은 다른 매개변수에서는 고르지 않다. 야코비 $\bigl|d\theta/d\eta\bigr| = \theta(1-\theta)$ 가 붙기 때문이다. 곧 "아무것도 모른다"는 진술이 어떤 자로 재느냐에 따라 달라진다.

    제프리스 앞확률은 $p(\theta) \propto \sqrt{I(\theta)}$ 로 잡는데, 피셔 정보 $I$ 가 매개변수 바꾸기에서 야코비의 제곱으로 바뀌므로 $\sqrt{I}$ 는 야코비 한 겹으로 바뀐다. 이는 밀도가 바뀌는 방식과 정확히 같아서, 어느 자로 재든 같은 앞확률을 준다. 베르누이에서 $I(\theta) = 1/(\theta(1-\theta))$ 이므로 $p(\theta)\propto \theta^{-1/2}(1-\theta)^{-1/2}$, 곧 $\text{Beta}(0.5,0.5)$ 이다.

## 3. 가능도 함수

**정의 3.** [가능도와 로그가능도]

데이터 $D$ 를 붙박아 두고 $\theta$ 의 함수로 본 $p(D \mid \theta)$ 를 **가능도 함수** $L(\theta)$ 라 하고, 그 로그를 **로그가능도** $\ell(\theta) = \log L(\theta)$ 라 한다. 관측이 서로 독립이면

$$
L(\theta) = \prod_{i=1}^n p(x_i \mid \theta),
\qquad
\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)
$$

이다. 가능도는 $\theta$ 에 대한 확률밀도가 아니므로 $\theta$ 에 대해 적분해도 $1$ 이 되지 않는다.

### 정리 3. 인수분해 정리 — 충분 통계량이 가능도를 요약한다 { .thm }

통계량 $T(D)$ 에 대해, 어떤 함수 $g, h$ 가 있어

$$
p(D \mid \theta) = g\bigl(T(D),\, \theta\bigr)\, h(D)
$$

로 적히는 것과 $T$ 가 $\theta$ 에 대한 **충분 통계량**인 것은 같은 말이다. 이때 뒤확률은 $T(D)$ 에만 달린다.

$$
p(\theta \mid D) \;\propto\; g\bigl(T(D), \theta\bigr)\, p(\theta)
$$

??? proof "증명"

    인수분해가 성립한다고 하자. 그러면

    $$
    p(\theta \mid D) = \frac{g(T(D),\theta)h(D)p(\theta)}{\int g(T(D),\theta')h(D)p(\theta')\,d\theta'}
    = \frac{g(T(D),\theta)p(\theta)}{\int g(T(D),\theta')p(\theta')\,d\theta'}
    $$

    이다. $h(D)$ 가 분자와 분모에서 지워지므로 뒤확률은 $D$ 를 오직 $T(D)$ 를 거쳐서만 본다. 곧 $T$ 를 알면 $D$ 를 더 알아도 $\theta$ 에 대한 믿음이 달라지지 않으니 $T$ 는 충분하다.

    거꾸로 $T$ 가 충분하면 $p(D \mid T, \theta)$ 가 $\theta$ 에 달리지 않으므로 $h(D) = p(D \mid T(D))$, $g(T,\theta) = p(T \mid \theta)$ 로 두면 인수분해가 나온다.

!!! note "쓰임새"
    베르누이 시행 $n$ 번에서 충분 통계량은 성공 횟수 $\sum x_i$ 하나뿐이다. 어느 시행에서 성공했는지 하는 순서 정보는 $\theta$ 에 대해 아무것도 말해 주지 않는다. 그래서 데이터 $n$ 개를 수 하나로 줄여도 손해가 없다.

**보기 2.** <span class="diff easy" title="쉬움"></span> 정규 $\mathcal{N}(\mu, \sigma^2)$ 에서 $\sigma$ 를 알 때 충분 통계량을 구하시오.

??? success "풀이"

    로그가능도를 펼치면

    $$
    \ell(\mu) = -\frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2 + \text{const}
    = -\frac{1}{2\sigma^2}\Bigl(\sum_i x_i^2 - 2\mu\sum_i x_i + n\mu^2\Bigr) + \text{const}
    $$

    이다. $\mu$ 와 얽힌 항은 $\sum_i x_i$ 뿐이므로 $T(D) = \sum_i x_i$, 같은 말로 표본 평균 $\bar x$ 가 충분 통계량이다.

**문제 2.** <span class="diff med" title="중간"></span> **가능도 원리**란 두 실험이 비례하는 가능도를 주면 같은 추론을 해야 한다는 것이다. 동전을 $12$ 번 던져 앞면 $9$ 번을 본 경우와, 앞면이 $9$ 번 나올 때까지 던져 $12$ 번 만에 멈춘 경우를 견주어 이 원리를 설명하시오.

??? success "풀이"

    앞의 것은 이항, 뒤의 것은 음이항 실험이다. 가능도는 각각

    $$
    \binom{12}{9}\theta^9(1-\theta)^3,
    \qquad
    \binom{11}{8}\theta^9(1-\theta)^3
    $$

    으로 $\theta$ 에 대해 **비례한다**. 상수 $\binom{12}{9}$ 와 $\binom{11}{8}$ 은 $\theta$ 에 달리지 않으므로 정리 1의 비례 관계에서 지워진다.

    따라서 베이즈 추론은 두 경우에 똑같은 뒤확률을 준다. 반면 빈도주의 $p$ 값은 "더 극단적인 결과"를 어떻게 셈하느냐가 실험 설계에 달려 있어 서로 다른 값이 나온다. 이것이 가능도 원리를 둘러싼 오랜 다툼의 자리다.

## 4. 증거와 뒤확률 셈하기

**정의 4.** [증거]

$p(D) = \int p(D\mid\theta)p(\theta)\,d\theta$ 를 **증거** 또는 **주변 가능도**라 한다. 뒤확률을 고르게 만드는 상수이면서, 동시에 그 모형이 데이터를 얼마나 잘 설명하는지를 나타내는 값이다.

### 정리 4. 증거의 두 얼굴 — 고르개 상수이자 모형 견줌의 자 { .thm }

두 모형 $M_1, M_2$ 에 대해 베이즈 인자를

$$
\text{BF}_{12} = \frac{p(D \mid M_1)}{p(D \mid M_2)}
$$

로 두면, 모형에 대한 뒤확률 승산은 앞확률 승산에 베이즈 인자를 곱한 것이다.

$$
\frac{p(M_1 \mid D)}{p(M_2 \mid D)} = \text{BF}_{12}\cdot\frac{p(M_1)}{p(M_2)}
$$

??? proof "증명"

    모형 하나하나에 베이즈 정리를 쓰면

    $$
    p(M_i \mid D) = \frac{p(D \mid M_i)\,p(M_i)}{p(D)}
    $$

    이다. 두 식의 비를 잡으면 공통 분모 $p(D)$ 가 지워져

    $$
    \frac{p(M_1\mid D)}{p(M_2\mid D)} = \frac{p(D\mid M_1)}{p(D\mid M_2)}\cdot\frac{p(M_1)}{p(M_2)}
    $$

    를 얻는다.

!!! note "쓰임새"
    매개변수 추론만 할 때는 $p(D)$ 를 셈할 까닭이 없다. 그러나 **모형을 견줄** 때는 바로 그 $p(D)$ 가 주인공이 된다. 매개변수가 많은 모형은 앞확률이 넓게 퍼져 $p(D)$ 가 작아지므로, 증거는 저절로 지나친 복잡함에 벌을 준다.

닫힌 꼴이 없으면 격자 위에서 수치로 셈한다.

```python
import numpy as np

def posterior_continuous(theta_grid, prior, likelihood, normalize=True):
    """
    격자 위에서 뒤확률 분포를 셈한다.
    
    매개변수
    ----------
    theta_grid : array-like
        매개변수 값의 격자
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

**보기 3.** <span class="diff easy" title="쉬움"></span> 위 함수가 왜 사다리꼴 적분만으로 뒤확률을 얻는지 설명하시오.

??? success "풀이"

    정리 1의 비례 관계에 따라 격자 위에서 `prior * likelihood` 를 셈하면 고르게 하지 않은 뒤확률이 곧바로 나온다. 남은 일은 그 곡선 아래 넓이로 나누어 적분이 $1$ 이 되게 하는 것뿐이며, 그 넓이가 바로 증거 $p(D)$ 의 수치 어림값이다.

    격자가 촘촘할수록 정확해지지만 차원이 늘면 격자점 수가 지수로 늘어난다. 그래서 매개변수가 몇 개만 넘어가도 격자 대신 표집을 쓴다.

## 5. 베타-이항 켤레 모형

이항 가능도와 베타 앞확률을 짝지으면 뒤확률이 다시 베타가 된다. 켤레성이 주는 가장 단순하고 중요한 보기다.

### 정리 5. 베타-이항 켤레성 — 앞확률은 유사 관측으로 더해진다 { .thm }

앞확률이 $\text{Beta}(\alpha,\beta)$ 이고 $n$ 번 중 $k$ 번 성공을 관측하면 뒤확률은

$$
\theta \mid D \;\sim\; \text{Beta}(\alpha + k,\ \beta + n - k)
$$

이다. 따라서 뒤확률의 평균은 앞확률 평균과 최대 가능도 어림값의 무게 준 평균이다.

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

    이고 이는 $\text{Beta}(\alpha+k,\ \beta+n-k)$ 의 핵이다.

    평균은 $\dfrac{\alpha+k}{\alpha+\beta+n}$ 이며, 분자를 $\alpha + k$ 로 쪼개어

    $$
    \frac{\alpha}{\alpha+\beta+n} + \frac{k}{\alpha+\beta+n}
    $$

    로 적은 뒤 각 항에 $\dfrac{\alpha+\beta}{\alpha+\beta}$ 와 $\dfrac nn$ 을 곱해 정리하면 결론의 꼴이 된다.

!!! note "쓰임새"
    $\alpha+\beta$ 는 **유사 관측의 수**로 읽힌다. 데이터가 $n \gg \alpha+\beta$ 이면 무게가 최대 가능도 쪽으로 쏠려 앞확률의 자취가 씻겨 나가고, 반대면 앞확률이 답을 지배한다.

```python
from scipy import stats

def beta_binomial_inference(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    베타-이항 모형을 써서 동전 던지기 확률에 대한 베이즈 추론.
    
    매개변수
    ----------
    n_heads : int
        관측된 앞면의 횟수
    n_tails : int
        관측된 뒷면의 횟수
    prior_alpha, prior_beta : float
        베타 앞확률의 매개변수
    
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

**보기 4.** <span class="diff easy" title="쉬움"></span> 앞면 $15$ 번, 뒷면 $5$ 번을 보았을 때 앞확률을 달리하면 뒤확률 평균이 어떻게 달라지는지 견주시오.

??? success "풀이"

    최대 가능도는 $15/20 = 0.75$ 이다. 정리 5를 쓰면 다음과 같다.

    | 앞확률 | 유사 관측 $\alpha+\beta$ | 뒤확률 평균 |
    |--------|------|------------|
    | $\text{Beta}(1,1)$ | $2$ | $16/22 \approx 0.727$ |
    | $\text{Beta}(0.5,0.5)$ | $1$ | $15.5/21 \approx 0.738$ |
    | $\text{Beta}(2,2)$ | $4$ | $17/24 \approx 0.708$ |
    | $\text{Beta}(10,10)$ | $20$ | $25/40 = 0.625$ |
    | $\text{Beta}(2,8)$ | $10$ | $17/30 \approx 0.567$ |

    유사 관측이 데이터 $20$ 번에 견주어 작을수록 뒤확률 평균이 $0.75$ 에 가깝다. 회의적인 $\text{Beta}(2,8)$ 은 $0.2$ 쪽으로 끌어당기는 힘이 세어 답이 크게 달라진다.

**문제 3.** <span class="diff hard" title="어려움"></span> 앞확률과 데이터가 정면으로 부딪칠 때 뒤확률이 어떤 모양이 되는지 밝히시오. 이를테면 $\text{Beta}(20,2)$ 앞확률에 $n=20$, $k=2$ 인 데이터를 주면 어떻게 되는가?

??? success "풀이"

    뒤확률은 $\text{Beta}(22, 20)$ 으로 평균이 $22/42 \approx 0.524$ 이다. 앞확률 평균 $20/22\approx 0.909$ 와 최대 가능도 $2/20 = 0.1$ 의 **가운데 어딘가**이며, 어느 쪽도 지지하지 않는 값이다.

    베타 족은 봉우리가 하나뿐이라 이 부딪침을 "둘 중 하나"로 나타내지 못하고 가운데로 뭉갠다. 이것이 켤레 앞확률의 한계다. 두 가설이 정말로 갈린다면 봉우리가 둘인 섞음 앞확률을 쓰거나, 애초에 앞확률이 틀렸는지 되물어야 한다.

## 6. 정규-정규 켤레 모형

### 정리 6. 정밀도로 무게 준 평균 — 정밀도는 더해진다 { .thm }

$x_1,\dots,x_n \sim \mathcal{N}(\mu, \sigma^2)$ 에서 $\sigma$ 를 알고 앞확률이 $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$ 이면 뒤확률은 정규이고, 정밀도 $\tau = 1/\sigma^2$ 로 적을 때

$$
\tau_n = \tau_0 + n\tau,
\qquad
\mu_n = \frac{\tau_0\,\mu_0 + n\tau\,\bar x}{\tau_0 + n\tau}
$$

이다. 곧 **정밀도는 더해지고, 평균은 정밀도로 무게 준 평균이 된다**.

??? proof "증명"

    로그를 잡고 $\mu$ 와 얽힌 항만 남기면

    $$
    \log p(\mu \mid D) = -\frac{\tau_0}{2}(\mu-\mu_0)^2 - \frac{n\tau}{2}(\mu-\bar x)^2 + \text{const}
    $$

    이다. $\mu$ 에 대해 펼쳐 이차항과 일차항을 모으면

    $$
    -\frac{\tau_0 + n\tau}{2}\,\mu^2 + (\tau_0\mu_0 + n\tau\bar x)\,\mu + \text{const}
    $$

    이다. 이는 정밀도가 $\tau_0 + n\tau$ 이고 평균이 $(\tau_0\mu_0+n\tau\bar x)/(\tau_0+n\tau)$ 인 정규의 로그밀도와 같은 꼴이다. 완전제곱으로 묶으면 곧바로 결론이 나온다.

!!! note "쓰임새"
    정밀도로 생각하면 베이즈 갱신이 **정보를 더하는 일**로 보인다. 앞확률이 $\tau_0$ 만큼, 데이터가 $n\tau$ 만큼 정보를 들고 오고, 뒤확률은 그 합만큼을 갖는다. 데이터가 늘수록 $n\tau$ 가 커져 앞확률의 몫이 줄어든다.

```python
def normal_normal_inference(data, prior_mean, prior_std, known_std):
    """
    흩어짐을 아는 정규 평균에 대한 베이즈 추론.
    
    매개변수
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
    
    # 뒤확률 매개변수
    posterior_mean = (prior_precision * prior_mean + 
                      data_precision * sample_mean) / posterior_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    return stats.norm(posterior_mean, posterior_std)
```

**보기 5.** <span class="diff easy" title="쉬움"></span> $\sigma = 1$ 인 관측 $n = 4$ 개의 평균이 $\bar x = 3$ 이고 앞확률이 $\mathcal{N}(0, 1)$ 일 때 뒤확률을 구하시오.

??? success "풀이"

    $\tau_0 = 1$, $\tau = 1$ 이므로 정리 6에서

    $$
    \tau_4 = 1 + 4 = 5, \qquad
    \mu_4 = \frac{1\cdot 0 + 4\cdot 3}{5} = 2.4
    $$

    이다. 곧 뒤확률은 $\mathcal{N}(2.4,\ 1/5)$ 이다. 앞확률이 관측 한 번의 무게를 가지므로 표본 평균 $3$ 이 $0$ 쪽으로 $1/5$ 만큼 끌려갔다.

## 7. 차례 갱신과 점근 거동

### 정리 7. 차례 갱신의 순서 무관성 — 한꺼번에 갱신한 것과 같다 { .thm }

관측이 $\theta$ 가 주어졌을 때 서로 독립이면, 데이터를 하나씩 넣으며 뒤확률을 거듭 고쳐도 마지막 결과는 모두를 한꺼번에 넣은 것과 같고, 넣는 **순서에도 달리지 않는다**.

$$
p(\theta \mid x_1, x_2) \;\propto\; p(x_2 \mid \theta)\,p(\theta \mid x_1)
\;\propto\; p(x_1\mid\theta)\,p(x_2\mid\theta)\,p(\theta)
$$

??? proof "증명"

    $x_1$ 을 넣은 뒤확률은 $p(\theta\mid x_1) \propto p(x_1\mid\theta)p(\theta)$ 이다. 이를 앞확률로 삼아 $x_2$ 를 넣으면

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
    이 성질 덕분에 데이터를 모두 쌓아 두지 않고 **흘려 보내며** 추론할 수 있다. 켤레 모형에서는 매개변수 몇 개만 들고 있으면 되므로 기억 자리가 데이터 양과 무관하다. 정리 5의 베타-이항이라면 $(\alpha,\beta)$ 두 수가 전부다.

```python
def sequential_beta_binomial(flip_sequence, prior_alpha=1, prior_beta=1):
    """
    베타-이항 모형으로 차례대로 베이즈 새로 고치기.
    
    매개변수
    ----------
    flip_sequence : str의 list
        'H'과 'T' 관측의 늘어놓음
    prior_alpha, prior_beta : float
        첫 앞확률 매개변수
    
    반환값
    -------
    alpha_history, beta_history : list
        새로 고침에 따른 매개변수의 흐름
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

    함수는 `'H'` 를 볼 때마다 `current_alpha` 를, `'T'` 를 볼 때마다 `current_beta` 를 $1$ 씩 올린다. 마지막 값은 앞면의 수와 뒷면의 수에만 달리고 순서에는 달리지 않는다. 이는 정리 7의 결론을 코드로 그대로 옮긴 것이며, 정리 3이 말한 "성공 횟수가 충분 통계량"이라는 사실과도 맞물린다.

**문제 4.** <span class="diff med" title="중간"></span> $n \to \infty$ 일 때 뒤확률이 어떻게 되는지 정리 6의 식으로 밝히고, 앞확률이 언제 씻겨 나가지 않는지 말하시오.

??? success "풀이"

    정리 6에서 $\tau_n = \tau_0 + n\tau \to \infty$ 이므로 뒤확률의 흩어짐 $1/\tau_n \to 0$ 이고,

    $$
    \mu_n = \frac{\tau_0\mu_0 + n\tau\bar x}{\tau_0+n\tau} \longrightarrow \bar x
    $$

    이다. 곧 뒤확률이 참값 둘레로 모여들고 앞확률의 자취는 씻겨 나간다. 이것이 베른슈타인-폰 미제스 정리가 말하는 바의 가장 단순한 꼴이다.

    다만 앞확률이 참값 언저리에 확률 $0$ 을 주면 이야기가 다르다. 비례 관계에서 $p(\theta)=0$ 인 자리는 데이터가 아무리 쌓여도 뒤확률이 $0$ 이다. 앞확률을 뾰족하게 잡을 때 늘 조심해야 하는 까닭이다.

## 8. 온전한 짜보기

앞의 마당에서 나눠 본 조각들을 하나의 갈래로 묶은 것이다. 격자 어림, 뒤확률 간추리기, 믿음 구간, 그림 그리기를 모두 담는다.

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
        매개변수 θ의 쓸 수 있는 범위
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
        
        매개변수
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
            매개변수 값의 격자
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
    
    매개변수
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
    
    # HPD 구역은 밀도가 문턱값을 넘는 모든 theta이다
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
    
    # 참 매개변수
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

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 센 앞확률과 약한 앞확률이 같은 데이터에서 어떻게 다른 뒤확률을 주는지 정리 5로 설명하시오.

??? success "풀이"

    정리 5에서 뒤확률 평균의 무게는 $\dfrac{\alpha+\beta}{\alpha+\beta+n}$ 와 $\dfrac{n}{\alpha+\beta+n}$ 이다. 센 앞확률은 $\alpha+\beta$ 가 커서 첫 무게가 크고, 약한 앞확률은 작아서 둘째 무게가 크다.

    $n$ 을 붙박아 두고 $\alpha+\beta$ 를 키우면 뒤확률 평균이 앞확률 평균 쪽으로 매끄럽게 움직인다. 보기 4의 표가 그 흐름을 수로 보여 준다.

**연습문제 2.** <span class="diff med" title="중간"></span> 앞확률 민감도 분석이란 무엇이며 왜 해야 하는지 밝히시오.

??? success "풀이"

    앞확률을 여러 개 두고 결론이 얼마나 바뀌는지 보는 일이다. 결론이 앞확률에 거의 달리지 않으면 데이터가 충분히 말하고 있다는 뜻이고, 크게 달라지면 결론이 사실은 **가정에서 나온 것**이라는 뜻이다.

    문제 3처럼 앞확률과 데이터가 부딪치는 경우에는 반드시 해야 한다. 알릴 때도 하나의 뒤확률이 아니라 앞확률 몇 가지에서 나온 결과를 나란히 보이는 것이 정직하다.

**연습문제 3.** <span class="diff med" title="중간"></span> 차례 갱신에서 뒤확률이 참값으로 모여드는 모습을 흉내 내어 확인하시오.

??? success "풀이"

    참값 $\theta^\star$ 로 베르누이 표본을 길게 만들고 `sequential_beta_binomial` 을 돌려 $(\alpha_t,\beta_t)$ 의 흐름을 얻은 뒤, 각 걸음의 뒤확률 평균 $\alpha_t/(\alpha_t+\beta_t)$ 와 표준편차를 그린다.

    평균은 $\theta^\star$ 로 다가가고 표준편차는 대략 $1/\sqrt t$ 로 줄어든다. 문제 4에서 정규 모형으로 본 것과 같은 거동이다.

**연습문제 4.** <span class="diff hard" title="어려움"></span> 정규 모형에서 평균과 흩어짐을 **둘 다** 모를 때의 켤레 앞확률인 정규-역감마를 적고, 뒤확률의 매개변수를 이끌어 내시오.

??? success "풀이"

    켤레 앞확률은 $\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0)$ 이고 그 조건 아래 $\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0,\ \sigma^2/\kappa_0)$ 이다. 뒤확률도 같은 족이며 매개변수는 다음과 같다.

    $$
    \kappa_n = \kappa_0 + n, \qquad
    \mu_n = \frac{\kappa_0\mu_0 + n\bar x}{\kappa_0+n}, \qquad
    a_n = a_0 + \frac n2
    $$

    $$
    b_n = b_0 + \frac12\sum_i (x_i-\bar x)^2 + \frac{\kappa_0 n(\bar x - \mu_0)^2}{2(\kappa_0+n)}
    $$

    $\mu_n$ 이 정리 6과 같은 무게 준 평균 꼴임에 주목하라. $b_n$ 의 마지막 항은 앞확률 평균과 표본 평균이 얼마나 어긋났는지에 대한 벌이며, 둘이 멀수록 흩어짐을 크게 잡는다.

**연습문제 5.** <span class="diff hard" title="어려움"></span> 실제 데이터를 하나 골라 앞확률을 세우고, 뒤확률을 셈한 뒤, 앞확률 예측 분포로 모형을 시험하시오.

??? success "풀이"

    절차는 이렇다. (1) 분야 지식으로 앞확률을 세운다. (2) 정리 2로 앞확률 예측 분포를 뽑아 실제로 볼 법한 데이터가 나오는지 본다. 여기서 어긋나면 앞확률로 돌아간다. (3) 데이터를 넣어 뒤확률을 얻는다. (4) 연습문제 2의 민감도 분석을 한다. (5) 뒤확률 예측 분포를 실제 데이터와 견주어 모형이 맞는지 다시 본다.

    (2)와 (5)를 건너뛰고 뒤확률만 알리는 것이 가장 흔한 잘못이다.

## 정리하며

베이즈 추론은 네 조각이 하나의 식으로 묶이는 이야기다.

1. 뒤확률은 가능도와 앞확률의 곱에 비례하고, 증거는 고르개 상수일 뿐이다(정리 1).
2. 앞확률은 관측 앞의 데이터 분포를 낳으며, 이것으로 앞확률이 터무니없는지 미리 시험할 수 있다(정리 2).
3. 가능도는 충분 통계량으로 요약되고, 뒤확률도 그 통계량에만 달린다(정리 3).
4. 증거는 매개변수 추론에서는 지워지지만 모형을 견줄 때는 주인공이 된다(정리 4).
5. 베타-이항에서 앞확률은 유사 관측으로 더해지고(정리 5), 정규-정규에서는 정밀도가 더해진다(정리 6).
6. 관측이 조건부 독립이면 데이터를 하나씩 흘려 넣어도 결과가 같다(정리 7).

데이터가 쌓이면 뒤확률은 참값 둘레로 모여들고 앞확률의 자취는 씻겨 나간다. 다만 앞확률이 어떤 자리에 확률 $0$ 을 주면 그 자리는 끝까지 되살아나지 않는다.

여기서 얻은 뒤확률을 점 하나로 줄이려면 「[최대 뒤확률 어림](map_estimation.md)」을, 구간으로 알리려면 「[믿음 구간](credible_intervals.md)」을, 켤레가 아니어서 손으로 풀 수 없으면 「[깁스 표집](../../ch15/mcmc/gibbs_sampling.md)」을 보라.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 1~3장
- Bishop, C. *Pattern Recognition and Machine Learning*, 2장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3~5장
