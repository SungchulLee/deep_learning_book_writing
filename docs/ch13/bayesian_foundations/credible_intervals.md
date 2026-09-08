# 믿음 구간

믿음 구간은 매개변수 어림값의 불확실성을 베이즈식으로 수에 담는다. 이 마당은 양 꼬리가 같은 구간과 최고 뒤확률 밀도(HPD) 구간을 세우고, 베이즈 믿음 구간과 빈도주의 신뢰 구간의 근본 차이를 또렷이 한다.

## 1. 믿음 구간이란 무엇인가

**정의 1.** [믿음 구간]

매개변수 $\theta$ 의 뒤확률 분포가 $p(\theta \mid D)$ 일 때, 다음을 만족하는 구간 $[L, U]$ 를 **$(1-\alpha) \times 100\%$ 믿음 구간**이라 한다.

$$
P(L \leq \theta \leq U \mid D) = 1 - \alpha
$$

이는 관찰한 데이터 $D$ 가 주어졌을 때 참 매개변수가 그 구간 안에 있을 확률이 $(1-\alpha) \times 100\%$ 라는 뜻이다. 매개변수 자체에 대한 곧은 확률 진술이며, 우리가 정말로 알고 싶은 양이다.

### 정리 1. 믿음 구간의 비유일성 — 같은 덮음률의 구간은 무수히 많다 { .thm }

뒤확률 분포가 이어져 있고 그 누적분포함수 $F$ 가 순증가라 하자. 그러면 각 $t \in [0, \alpha]$ 마다

$$
[L_t, U_t] = \bigl[\,F^{-1}(t),\; F^{-1}(1-\alpha+t)\,\bigr]
$$

는 모두 $(1-\alpha) \times 100\%$ 믿음 구간이다. 곧 믿음 구간은 하나로 정해지지 않는다.

??? proof "증명"

    $F$ 가 순증가이고 이어져 있으므로

    $$
    P(L_t \leq \theta \leq U_t \mid D) = F(U_t) - F(L_t) = (1-\alpha+t) - t = 1-\alpha
    $$

    이다. $t$ 는 $[0,\alpha]$ 안에서 아무 값이나 될 수 있으므로 그런 구간은 무수히 많다.

!!! note "쓰임새"
    덮음률만으로는 구간이 정해지지 않으니 **추가 잣대**가 있어야 한다. 널리 쓰는 두 잣대가 다음 마당의 "양 꼬리를 똑같이 둔다"와 "길이를 가장 짧게 한다"이다.

**보기 1.** <span class="diff easy" title="쉬움"></span> 뒤확률이 $\text{Beta}(16,6)$ 일 때 $t=0$, $t=\alpha/2$, $t=\alpha$ 로 얻는 세 $95\%$ 믿음 구간을 견주시오.

??? success "풀이"

    $\alpha = 0.05$ 이다. $t=0$ 이면 $[F^{-1}(0), F^{-1}(0.95)]$ 로 왼쪽 끝이 $0$ 인 한쪽 구간, $t=\alpha$ 이면 $[F^{-1}(0.05), F^{-1}(1)]$ 로 오른쪽 끝이 $1$ 인 한쪽 구간이다. $t = \alpha/2 = 0.025$ 이면 양 꼬리가 같은 구간 $[0.577, 0.888]$ 이 된다.

    셋 모두 뒤확률 $0.95$ 를 담지만 길이는 크게 다르다. 한쪽 구간은 쓸모가 적고, 가운데 것이 가장 자연스럽다.

## 2. 양 꼬리가 같은 구간과 HPD 구간

**정의 2.** [양 꼬리가 같은 구간과 HPD 구간]

**양 꼬리가 같은 구간**은 양쪽 꼬리에 확률 질량 $\alpha/2$ 씩을 똑같이 두는 구간이다.

$$
[L, U] = [\,q_{\alpha/2},\; q_{1-\alpha/2}\,]
$$

여기서 $q_p$ 는 뒤확률 분포의 $p$ 번째 분위수이다. **최고 뒤확률 밀도(HPD) 구간**은 어떤 문턱값 $k$ 에 대해

$$
\{\,\theta : p(\theta \mid D) \geq k\,\}
$$

꼴인 집합이며, $k$ 는 그 집합이 담는 뒤확률이 $1-\alpha$ 가 되도록 고른다.

### 정리 2. HPD 구간의 가장 짧음 — 밀도가 높은 자리를 모으면 가장 짧다 { .thm }

뒤확률 밀도 $p(\cdot \mid D)$ 가 이어져 있다고 하자. 덮음률이 $1-\alpha$ 인 모든 구간 가운데 HPD 구간의 길이가 가장 짧다.

??? proof "증명"

    $A = \{\theta : p(\theta\mid D) \geq k\}$ 를 HPD 집합, $B$ 를 덮음률이 같은 다른 집합이라 하자. 곧 $\int_A p = \int_B p = 1-\alpha$ 이다. 이때

    $$
    \int_{A \setminus B} p = \int_A p - \int_{A \cap B} p
    = \int_B p - \int_{A \cap B} p = \int_{B \setminus A} p
    $$

    이다. 그런데 $A \setminus B$ 위에서는 $p \geq k$ 이고 $B \setminus A$ 위에서는 $p < k$ 이므로

    $$
    k \cdot |A \setminus B| \leq \int_{A\setminus B} p = \int_{B \setminus A} p < k \cdot |B \setminus A|
    $$

    이다. 여기서 $|\cdot|$ 은 길이이다. 양변을 $k>0$ 으로 나누면 $|A \setminus B| \leq |B \setminus A|$ 이고, 양쪽에 $|A \cap B|$ 를 더하면 $|A| \leq |B|$ 를 얻는다.

!!! note "쓰임새"
    HPD는 **가장 짧다**는 좋은 성질을 주지만 분위수 하나로 얻어지지 않아 셈이 번거롭다. 봉우리가 여럿인 뒤확률에서는 HPD 집합이 이어진 구간이 아니라 **떨어진 구간 여러 개**가 되기도 한다.

| 성질 | 양 꼬리가 같음 | HPD |
|----------|--------------|-----|
| 셈하기 | 단순함(분위수) | 더 복잡함 |
| 최적성 | 가장 짧음이 보장되지 않음 | 가장 짧은 구간 |
| 대칭 뒤확률 | 같음 | 같음 |
| 기운 뒤확률 | 더 김 | 더 짧음 |
| 풀이 | 양 꼬리 확률이 같음 | 밀도가 가장 높은 자리 |

```python
import numpy as np

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

**보기 2.** <span class="diff easy" title="쉬움"></span> 앞면 $15$ 번, 뒷면 $5$ 번을 관찰하고 고른 앞확률 $\text{Beta}(1,1)$ 을 두었다. 두 구간을 셈하여 견주시오.

??? success "풀이"

    뒤확률은 $\text{Beta}(16, 6)$ 이다.

    ```python
    from scipy import stats

    posterior = stats.beta(16, 6)
    et_lower, et_upper = posterior.ppf(0.025), posterior.ppf(0.975)
    samples = posterior.rvs(100000)
    hpd_lower, hpd_upper = compute_hpd_interval(samples, alpha=0.05)
    ```

    | 구간 종류 | 아래 | 위 | 너비 |
    |---------------|-------|-------|-------|
    | 양 꼬리가 같음 | $0.577$ | $0.888$ | $0.311$ |
    | HPD | $0.571$ | $0.885$ | $0.314$ |

    살짝 기운 이 뒤확률에서는 두 구간이 거의 같다. 표본으로 얻은 HPD의 너비가 되레 조금 큰 것은 표집 오차 탓이며, 정리 2는 참 밀도에 대한 진술이다.

**문제 1.** <span class="diff med" title="중간"></span> 봉우리가 둘인 뒤확률에서 HPD 집합이 이어진 구간이 아닐 수 있음을 보이고, 그때 "구간"을 알리는 것이 왜 오해를 낳는지 설명하시오.

??? success "풀이"

    밀도가 $\theta = 0.2$ 와 $\theta = 0.8$ 에서 두 봉우리를 이루고 가운데 $0.5$ 언저리가 낮다고 하자. 문턱값 $k$ 를 두 봉우리 사이 골짜기보다 높게 잡으면 $\{p \geq k\}$ 는 두 봉우리를 감싸는 **떨어진 두 구간**이 된다.

    이때 둘을 아우르는 하나의 구간 $[0.15, 0.85]$ 를 알리면 밀도가 거의 $0$ 인 가운데 값들까지 그럴듯한 것처럼 보이게 된다. 봉우리가 여럿이면 구간 대신 **뒤확률 전체를 그려서** 알려야 한다.

## 3. 대칭일 때의 일치

두 구간이 언제 같아지는지는 뒤확률의 대칭성이 정한다.

### 정리 3. 대칭 뒤확률에서의 일치 — 두 잣대가 같은 답을 준다 { .thm }

뒤확률 밀도가 어떤 점 $c$ 를 중심으로 대칭이고 $c$ 에서 봉우리가 하나이면, 양 꼬리가 같은 구간과 HPD 구간이 일치한다.

??? proof "증명"

    대칭과 단봉이므로 밀도는 $c$ 에서 멀어질수록 줄어든다. 곧 $|\theta_1 - c| < |\theta_2 - c|$ 이면 $p(\theta_1 \mid D) > p(\theta_2 \mid D)$ 이다.

    따라서 집합 $\{p \geq k\}$ 는 $c$ 를 중심으로 하는 구간 $[c-d, c+d]$ 꼴이고, 이 구간이 담는 확률이 $1-\alpha$ 가 되도록 $d$ 를 잡은 것이 HPD이다.

    한편 대칭성에서 $P(\theta < c-d \mid D) = P(\theta > c+d \mid D)$ 이고 둘의 합이 $\alpha$ 이므로 각각 $\alpha/2$ 이다. 이는 곧 $[c-d, c+d]$ 가 양 꼬리가 같은 구간이라는 뜻이다.

!!! note "쓰임새"
    가우스 뒤확률이나 자유도가 큰 $t$ 뒤확률처럼 대칭에 가까운 경우에는 굳이 HPD를 셈할 까닭이 없다. 분위수 두 번이면 끝난다.

## 4. 믿음 구간과 신뢰 구간

**정의 3.** [신뢰 구간]

빈도주의에서 매개변수 $\theta$ 는 모르지만 붙박인 상수이고 데이터가 확률적이다. 데이터의 함수인 구간 $[\hat L(D), \hat U(D)]$ 가 모든 $\theta$ 에 대해

$$
P_{D \mid \theta}\bigl(\hat L(D) \leq \theta \leq \hat U(D)\bigr) = 1-\alpha
$$

를 만족하면 이를 **$(1-\alpha)\times 100\%$ 신뢰 구간**이라 한다. 확률은 $\theta$ 가 아니라 $D$ 에 대해 잡힌다.

### 정리 4. 두 구간의 갈림 — 수는 같아도 뜻이 다르다 { .thm }

$x_1,\dots,x_n \sim \mathcal{N}(\mu, \sigma^2)$ 이고 $\sigma$ 를 안다고 하자. 평균에 고른 앞확률(변칙 앞확률) $p(\mu) \propto 1$ 을 두면 뒤확률은 $\mathcal{N}(\bar x,\ \sigma^2/n)$ 이고, $95\%$ 믿음 구간과 $95\%$ 신뢰 구간은 **수치로 똑같이**

$$
\bar x \pm 1.96\,\frac{\sigma}{\sqrt n}
$$

이 된다. 그러나 두 진술이 뜻하는 바는 다르다. 앞의 것은 $\mu$ 에 대한 확률이고, 뒤의 것은 절차에 대한 확률이다.

??? proof "증명"

    뒤확률은 $p(\mu \mid D) \propto p(D \mid \mu)\cdot 1 \propto \exp\!\bigl(-\tfrac{n}{2\sigma^2}(\mu-\bar x)^2\bigr)$ 이므로 $\mathcal{N}(\bar x, \sigma^2/n)$ 이다. 대칭이므로 정리 3에 따라 $95\%$ 믿음 구간은 $\bar x \pm 1.96\,\sigma/\sqrt n$ 이다.

    한편 빈도주의에서는 $\bar X \sim \mathcal{N}(\mu, \sigma^2/n)$ 이므로 $P(|\bar X - \mu| \leq 1.96\,\sigma/\sqrt n) = 0.95$ 이고, 이를 $\mu$ 에 대해 풀면 같은 구간이 나온다. 두 셈이 같은 수를 주는 것은 고른 앞확률이 가능도를 그대로 두기 때문이다.

!!! note "쓰임새"
    수가 같다고 뜻이 같지는 않다. 신뢰 구간은 매개변수가 그 구간 안에 있을 확률이 $95\%$ 라는 뜻이 **아니다**. 빈도주의에서 $\theta$ 는 붙박인 상수여서 구간 안에 있거나 없거나 둘 중 하나이고, 확률은 되풀이하는 절차에 붙는다.

| 갈래 | 믿음 구간(베이즈) | 신뢰 구간(빈도주의) |
|--------|------------------------------|-----------------------------------|
| **확률 진술** | 매개변수에 대해 | 절차에 대해 |
| **데이터** | 붙박임(관찰됨) | 확률적임(가상의 되풀이) |
| **매개변수** | 확률적임(분포를 갖는다) | 붙박임(모르는 상수) |
| **풀이** | "$\theta$ 가 여기 있을 확률이 $95\%$" | "그런 구간의 $95\%$ 가 $\theta$ 를 담는다" |

**문제 2.** <span class="diff med" title="중간"></span> 앞확률을 $\mathcal{N}(\mu_0, \tau^2)$ 으로 바꾸면 정리 4의 두 구간이 더는 일치하지 않음을 보이고, $\tau \to \infty$ 일 때 어떻게 되는지 밝히시오.

??? success "풀이"

    켤레 갱신에서 뒤확률의 평균과 흩어짐은

    $$
    \mu_n = \frac{\tau^{-2}\mu_0 + n\sigma^{-2}\bar x}{\tau^{-2} + n\sigma^{-2}},
    \qquad
    \sigma_n^2 = \bigl(\tau^{-2} + n\sigma^{-2}\bigr)^{-1}
    $$

    이다. 믿음 구간은 $\mu_n \pm 1.96\,\sigma_n$ 으로 중심이 $\bar x$ 에서 $\mu_0$ 쪽으로 끌려가고 폭도 $\sigma/\sqrt n$ 보다 좁아진다. 반면 신뢰 구간은 앞확률을 쓰지 않으므로 그대로다.

    $\tau \to \infty$ 이면 $\tau^{-2} \to 0$ 이므로 $\mu_n \to \bar x$, $\sigma_n^2 \to \sigma^2/n$ 이 되어 정리 4의 경우로 돌아간다.

## 5. 덮음률

믿음 구간이 빈도주의 잣대로 보아도 쓸 만한지는 덮음률로 따진다.

### 정리 5. 베이즈 구간의 평균 덮음 — 앞확률이 맞으면 정확하다 { .thm }

$\theta$ 가 앞확률 $p(\theta)$ 에서 뽑히고 그 $\theta$ 에서 데이터가 나온다고 하자. 각 데이터마다 $(1-\alpha)$ 믿음 구간 $C(D)$ 를 만들면, 앞확률에 대해 평균 낸 덮음률은 정확히 $1-\alpha$ 이다.

$$
\mathbb{E}_{\theta \sim p(\theta)}\bigl[\,P_{D \mid \theta}(\theta \in C(D))\,\bigr] = 1-\alpha
$$

??? proof "증명"

    기댓값의 탑 성질을 데이터 쪽으로 조건 지어 쓰면

    $$
    \mathbb{E}\bigl[\mathbf{1}[\theta \in C(D)]\bigr]
    = \mathbb{E}_{D}\Bigl[\,\mathbb{E}\bigl[\mathbf{1}[\theta \in C(D)] \;\big|\; D\bigr]\Bigr]
    = \mathbb{E}_{D}\bigl[\,P(\theta \in C(D) \mid D)\,\bigr]
    $$

    이다. 안쪽 값은 믿음 구간의 뜻매김에 따라 어떤 $D$ 에서도 $1-\alpha$ 이므로, 바깥 기댓값도 $1-\alpha$ 이다.

!!! note "쓰임새"
    이 등식은 **앞확률에 대해 평균 낸** 덮음이다. 특정한 하나의 $\theta$ 에서는 덮음률이 $1-\alpha$ 보다 크거나 작을 수 있다. 그래도 실제로는 베이즈 구간이 발드 신뢰 구간보다 눈금이 잘 맞는 편이다.

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

**보기 3.** <span class="diff easy" title="쉬움"></span> $p_{\text{true}} = 0.7$, $n = 20$ 으로 실험 $1000$ 번을 돌린 결과를 읽으시오.

??? success "풀이"

    | 구간 종류 | 관찰된 덮음률 |
    |---------------|-------------------|
    | $95\%$ 믿음 구간 | $94$~$96\%$ 남짓 |
    | $95\%$ 신뢰 구간(발드) | $90$~$94\%$ 남짓 |

    발드 신뢰 구간은 표본이 작거나 확률이 극단적일 때 **덜 덮는다**. $\hat p$ 가 $0$ 이나 $1$ 에 가까우면 표준오차 $\sqrt{\hat p(1-\hat p)/n}$ 이 $0$ 으로 무너져 구간이 지나치게 좁아지기 때문이다. 베이즈 구간은 앞확률이 $\hat p$ 를 끝에서 떼어 놓아 이 무너짐을 겪지 않는다.

## 6. 실전에서 살필 것

어느 구간을 쓸지, 무엇을 함께 알릴지에 대한 지침이다.

| 상황 | 권하는 구간 |
|-----------|---------------------|
| 대칭 뒤확률 | 아무거나(정리 3에 따라 같다) |
| 기운 뒤확률 | HPD(더 짧다) |
| 빠른 셈 | 양 꼬리가 같은 구간 |
| 가장 좋은 결정 | HPD |
| 통계 전문가가 아닌 사람에게 알릴 때 | 믿음 구간(더 직관적이다) |

알릴 때는 보통 **점 어림값**(뒤확률의 평균이나 최대 뒤확률), **$95\%$ 믿음 구간**, 그리고 가능하면 **뒤확률 분포**를 함께 적는다.

> "어림한 성공 확률은 $0.73$ 이다($95\%$ 믿음 구간: $0.58$~$0.89$)"

매개변수가 여럿이면 구간 대신 **믿음 영역** $\mathcal{R}$ 을 셈한다.

$$
P(\theta \in \mathcal{R} \mid D) = 1 - \alpha
$$

$2$ 차원 매개변수라면 가우스 뒤확률에서 **믿음 타원**이 나온다.

## 연습문제

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 믿음 구간을 정의하고 빈도주의 신뢰 구간과 견주시오.

??? success "풀이"

    $95\%$ 믿음 구간 $[a,b]$ 는 $P(\theta \in [a,b] \mid D) = 0.95$ 를 뜻한다. 곧 데이터가 주어졌을 때 매개변수가 이 구간에 있을 확률이 $95\%$ 이다.

    $95\%$ 신뢰 구간은 실험을 여러 번 되풀이하면 그렇게 만든 구간의 $95\%$ 가 참 매개변수를 담는다는 뜻이다. 정리 4가 보였듯 특별한 경우에는 두 구간의 수가 같아지지만 뜻은 끝까지 다르다.

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $\text{Beta}(10, 30)$ 뒤확률의 $95\%$ 믿음 구간을 셈하시오.

??? success "풀이"

    분위수를 쓰면 아래는 $F^{-1}(0.025) = 0.134$, 위는 $F^{-1}(0.975) = 0.394$ 이므로 구간은 $[0.134,\ 0.394]$ 이다. 뒤확률의 평균은 $10/40 = 0.25$ 이다.

**연습문제 3.** <span class="diff med" title="중간"></span> $\text{Gamma}(2, 1)$ 뒤확률에 대해 양 꼬리가 같은 구간과 HPD 구간을 모두 셈하고, HPD가 더 짧은지 확인하시오.

??? success "풀이"

    $\text{Gamma}(2,1)$ 은 오른쪽으로 기울어 있으므로 정리 3의 조건을 만족하지 않는다. 양 꼬리가 같은 구간은 대략 $[0.242,\ 5.572]$ 로 너비가 $5.33$ 이고, HPD는 대략 $[0.052,\ 4.744]$ 로 너비가 $4.69$ 이다.

    HPD가 더 짧다. 기울어진 쪽에서는 왼쪽 꼬리를 $2.5\%$ 씩 잘라 내는 것이 낭비이기 때문이며, 이것이 정리 2가 말하는 바다.

**연습문제 4.** <span class="diff med" title="중간"></span> MCMC 표본에서 HPD 구간을 셈하는 함수를 구현하시오.

??? success "풀이"

    표본을 줄 세운 뒤 길이가 같은 모든 창을 훑어 가장 좁은 것을 고른다.

    ```python
    def hpd_interval(samples, credibility=0.95):
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_size = int(np.ceil(credibility * n))
        widths = sorted_samples[interval_size:] - sorted_samples[:n-interval_size]
        best = np.argmin(widths)
        return sorted_samples[best], sorted_samples[best + interval_size]
    ```

    이는 정리 2의 밀도 수준 집합을 표본으로 어림한 것이다. 봉우리가 여럿이면 이 방법은 가장 큰 봉우리 하나만 잡으므로 문제 1에서 본 함정에 빠진다.

**연습문제 5.** <span class="diff hard" title="어려움"></span> $2$ 차원 정규 뒤확률의 $95\%$ 믿음 타원을 구현하고, 공분산 짜임에 따라 영역이 어떻게 바뀌는지 밝히시오.

??? success "풀이"

    $\theta \sim \mathcal{N}(\mu, \Sigma)$ 이면 $(\theta-\mu)^\top \Sigma^{-1}(\theta-\mu) \sim \chi^2_2$ 이므로 믿음 영역은

    $$
    \mathcal{R} = \bigl\{\theta : (\theta-\mu)^\top \Sigma^{-1}(\theta-\mu) \leq \chi^2_{2,\,0.95}\bigr\}
    $$

    이고 $\chi^2_{2,\,0.95} \approx 5.991$ 이다. 이는 밀도 수준 집합이므로 정리 2에 따라 넓이가 가장 작은 영역이다.

    타원의 축 방향은 $\Sigma$ 의 고유벡터, 축 길이는 $\sqrt{5.991\,\lambda_i}$ 이다. 두 성분의 상관이 커질수록 타원이 대각선 쪽으로 길게 눕는다.

**연습문제 6.** <span class="diff hard" title="어려움"></span> 작은 표본 크기($n = 5, 10, 20$)와 극단적인 참 확률($p = 0.05, 0.5, 0.95$)에서 믿음 구간과 신뢰 구간의 실제 덮음률을 견주시오.

??? success "풀이"

    `coverage_simulation` 을 격자로 돌리면 $p = 0.5$ 부근에서는 둘이 비슷하지만, $p = 0.05$ 나 $0.95$ 이고 $n$ 이 작을 때 발드 구간의 덮음률이 $80\%$ 아래로 떨어지는 일이 흔하다. 보기 3에서 짚은 표준오차 무너짐 때문이다.

    믿음 구간은 $90\%$ 언저리를 지키는데, 이는 정리 5가 보장하는 것이 **앞확률에 대한 평균** 덮음이지 각 $\theta$ 에서의 덮음이 아니기 때문이다. 곧 극단적인 $p$ 에서는 베이즈 구간도 이름값에 못 미칠 수 있다.

## 정리하며

믿음 구간은 매개변수에 대한 **곧은 확률 진술**이다.

1. 덮음률만으로는 구간이 하나로 정해지지 않는다(정리 1). 잣대를 하나 더 얹어야 한다.
2. "양 꼬리를 똑같이"는 셈이 쉽고, "가장 짧게"는 HPD를 준다. HPD는 밀도 수준 집합이며 길이가 가장 짧다(정리 2).
3. 대칭이고 봉우리가 하나이면 두 잣대가 같은 답을 준다(정리 3). 가우스 뒤확률에서 HPD를 따로 셈할 까닭이 없는 이유다.
4. 믿음 구간과 신뢰 구간은 특별한 경우 수가 같아지지만(정리 4) 뜻은 끝까지 다르다. 앞의 것은 $\theta$ 에, 뒤의 것은 절차에 확률을 붙인다.
5. 앞확률에 대해 평균 내면 베이즈 구간의 덮음률은 정확히 $1-\alpha$ 이다(정리 5). 다만 특정한 $\theta$ 에서의 덮음은 보장하지 않는다.

봉우리가 여럿인 뒤확률에서는 구간 하나로 알리는 것이 오해를 낳으니(문제 1) 뒤확률 전체를 보여야 한다. 점 어림값이 필요하면 「[최대 뒤확률 어림](map_estimation.md)」을, 뒤확률을 표본으로 얻어야 하면 「[깁스 표집](../../ch15/mcmc/gibbs_sampling.md)」을 보라.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2장
- Kruschke, J. *Doing Bayesian Data Analysis* (2nd ed.), 12장
- Hoff, P. *A First Course in Bayesian Statistical Methods*, 3장
