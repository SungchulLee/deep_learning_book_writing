# 베이즈 신용구간

베이즈 신용구간은 관측 데이터와 모형·사전분포가 주어진 상태에서 파라메터의 사후 불확실성을 요약한다. 동일꼬리 구간과 최고사후밀도(HPD) 집합을 비교하고, 빈도주의 신뢰구간의 반복 표본추출 해석과 구분한다.

## 1. 신용구간이란 무엇인가

**정의 1.** [베이즈 신용구간]

파라메터 $\theta$ 의 사후확률 분포가 $p(\theta \mid D)$ 일 때, 다음을 만족하는 구간 $[L, U]$ 를 **$(1-\alpha) \times 100\%$ 신용구간**이라 한다.

$$
P(L \leq \theta \leq U \mid D) = 1 - \alpha
$$

이 확률은 관측한 데이터 D와 선택한 모형·사전분포에 조건부인 사후확률이다. 구간의 명목 확률만으로 모형의 타당성이나 반복 실험에서 특정한 참값을 포함하는 빈도를 보장하지 않는다.

### 정리 1. 신용구간의 비유일성 { .thm }

사후확률 분포가 연속적이어 있고 그 누적분포함수 $F$ 가 순증가라 하자. 그러면 각 $t \in [0, \alpha]$ 마다

$$
[L_t, U_t] = \bigl[\,F^{-1}(t),\; F^{-1}(1-\alpha+t)\,\bigr]
$$

는 모두 $(1-\alpha) \times 100\%$ 신용구간이다. 곧 신용구간은 하나로 정해지지 않는다.

??? proof "증명"

    $F$ 가 순증가이고 연속적이어 있으므로

    $$
    P(L_t \leq \theta \leq U_t \mid D) = F(U_t) - F(L_t) = (1-\alpha+t) - t = 1-\alpha
    $$

    이다. $t$ 는 $[0,\alpha]$ 안에서 아무 값이나 될 수 있으므로 그런 구간은 무수히 많다.

!!! note "쓰임새"
    포함확률만으로는 구간이 정해지지 않으니 **추가 잣대**가 있어야 한다. 널리 쓰는 두 잣대가 다음 마당의 "양 꼬리를 똑같이 둔다"와 "길이를 가장 짧게 한다"이다.

**보기 1.** <span class="diff easy" title="쉬움"></span> 사후확률이 $\text{Beta}(16,6)$ 일 때 $t=0$, $t=\alpha/2$, $t=\alpha$ 로 얻는 세 $95\%$ 신용구간을 견주시오.

??? success "풀이"

    α=0.05다. t=0은 [0,F⁻¹(0.95)], t=0.05는 [F⁻¹(0.05),1], t=0.025는 [F⁻¹(0.025),F⁻¹(0.975)]를 준다. 여기서 F는 Beta(16,6)의 누적분포함수다. 각 구간은 0.95의 사후확률을 가지지만 위치와 길이가 다르다.

    한쪽 구간과 동일꼬리 구간 중 어느 쪽이 적절한지는 질문에 따라 다르다. 모수의 상한만 필요한 경우 등에는 한쪽 구간도 유용하다.

## 2. 동일꼬리 구간과 HPD 구간

**정의 2.** [동일꼬리 구간과 HPD 구간]

**동일꼬리 구간**은 양쪽 꼬리에 확률 질량 $\alpha/2$ 씩을 똑같이 두는 구간이다.

$$
[L, U] = [\,q_{\alpha/2},\; q_{1-\alpha/2}\,]
$$

여기서 $q_p$ 는 사후확률 분포의 $p$ 번째 분위수이다. **최고사후밀도(HPD) 구간**은 어떤 임계값 $k$ 에 대해

$$
\{\,\theta : p(\theta \mid D) \geq k\,\}
$$

꼴인 집합이며, $k$ 는 그 집합이 담는 사후확률이 $1-\alpha$ 가 되도록 고른다.

### 정리 2. HPD 집합의 최소 길이 성질 { .thm }

연속 사후밀도에서 양의 임계값 k로 정의한 HPD 집합이 정확히 1−α의 확률을 담을 수 있다고 하자. 그러면 같은 확률을 담는 측정 가능한 집합 가운데 HPD 집합의 전체 길이가 최소다. 단봉 분포에서 이 집합이 연결되어 있으면 최단 구간이 된다. 다봉 분포에서는 분리된 여러 구간일 수 있다.

??? proof "증명"

    $A = \{\theta : p(\theta\mid D) \geq k\}$ 를 HPD 집합, $B$ 를 포함확률이 같은 다른 집합이라 하자. 곧 $\int_A p = \int_B p = 1-\alpha$ 이다. 이때

    $$
    \int_{A \setminus B} p = \int_A p - \int_{A \cap B} p
    = \int_B p - \int_{A \cap B} p = \int_{B \setminus A} p
    $$

    이다. 그런데 $A \setminus B$ 위에서는 $p \geq k$ 이고 $B \setminus A$ 위에서는 $p < k$ 이므로

    $$
    k \cdot |A \setminus B| \leq \int_{A\setminus B} p = \int_{B \setminus A} p \leq k \cdot |B \setminus A|
    $$

    이다. 여기서 $|\cdot|$ 은 길이이다. 양변을 $k>0$ 으로 나누면 $|A \setminus B| \leq |B \setminus A|$ 이고, 양쪽에 $|A \cap B|$ 를 더하면 $|A| \leq |B|$ 를 얻는다.

!!! note "쓰임새"
    HPD는 **가장 짧다**는 좋은 성질을 주지만 분위수 하나로 얻어지지 않아 계산이 번거롭다. 봉우리가 여럿인 사후확률에서는 HPD 집합이 이어진 구간이 아니라 **떨어진 구간 여러 개**가 되기도 한다.

| 성질 | 동일꼬리 구간 | HPD 집합 |
|---|---|---|
| 기준 | 양쪽 꼬리 확률을 같게 설정 | 높은 밀도 영역을 선택 |
| 계산 | 두 분위수 | 밀도 수준 또는 별도 최적화 |
| 모양 | 연결된 구간 | 다봉이면 분리될 수 있음 |
| 길이 | 최소라고 보장되지 않음 | 조건 아래 전체 길이 최소 |
| 변환 | 단조 변환과 분위수가 대응 | 밀도·파라메터화에 의존 |

```python
import numpy as np

def compute_equal_tailed_interval(posterior_dist, alpha=0.05):
    """
    양끝이 같은 믿음 구간을 셈한다.
    
    파라메터
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
    
    파라메터
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

**보기 2.** <span class="diff easy" title="쉬움"></span> 앞면 $15$ 번, 뒷면 $5$ 번을 관찰하고 균등 사전분포 $\text{Beta}(1,1)$ 을 두었다. 두 구간을 셈하여 견주시오.

??? success "풀이"

    사후확률은 $\text{Beta}(16, 6)$ 이다.

    ```python
    from scipy import stats

    posterior = stats.beta(16, 6)
    et_lower, et_upper = posterior.ppf(0.025), posterior.ppf(0.975)
    samples = posterior.rvs(100000)
    hpd_lower, hpd_upper = compute_hpd_interval(samples, alpha=0.05)
    ```

    동일꼬리 구간은 Beta(16,6)의 0.025·0.975 분위수로 구한다. 아래 코드의 HPD 추정은 난수 표본에서 가장 좁은 창을 찾는 근사이므로 결과가 달라질 수 있다. 수치가 맞는지는 같은 사후분포의 확률 질량과 길이를 확인한다.

    샘플 기반 최단 창은 참 밀도의 HPD 경계를 정확히 계산한 값과 다르다. 유한 표본에서 얻은 길이를 정확한 최소 길이 성질과 혼동하지 않는다.

**문제 1.** <span class="diff med" title="중간"></span> 봉우리가 둘인 사후확률에서 HPD 집합이 이어진 구간이 아닐 수 있음을 보이고, 그때 "구간"을 알리는 것이 왜 오해를 낳는지 설명하시오.

??? success "풀이"

    밀도가 $\theta = 0.2$ 와 $\theta = 0.8$ 에서 두 봉우리를 이루고 가운데 $0.5$ 언저리가 낮다고 하자. 임계값 $k$ 를 두 봉우리 사이 골짜기보다 높게 잡으면 $\{p \geq k\}$ 는 두 봉우리를 감싸는 **떨어진 두 구간**이 된다.

    이때 둘을 아우르는 하나의 구간 $[0.15, 0.85]$ 를 알리면 밀도가 거의 $0$ 인 가운데 값들까지 그럴듯한 것처럼 보이게 된다. 봉우리가 여럿이면 구간 대신 **사후확률 전체를 그려서** 알려야 한다.

## 3. 대칭일 때의 일치

두 구간이 언제 같아지는지는 사후확률의 대칭성이 정한다.

### 정리 3. 대칭 사후확률에서의 일치 — 두 잣대가 같은 답을 준다 { .thm }

사후밀도가 c를 중심으로 대칭이고 c에서 멀어질수록 엄격히 감소하는 단봉 분포라면 동일꼬리 구간과 HPD 구간이 일치한다. 평탄한 부분이 있는 분포에서는 HPD 선택이 유일하지 않을 수 있다.

??? proof "증명"

    대칭과 단봉이므로 밀도는 $c$ 에서 멀어질수록 줄어든다. 곧 $|\theta_1 - c| < |\theta_2 - c|$ 이면 $p(\theta_1 \mid D) > p(\theta_2 \mid D)$ 이다.

    따라서 집합 $\{p \geq k\}$ 는 $c$ 를 중심으로 하는 구간 $[c-d, c+d]$ 꼴이고, 이 구간이 담는 확률이 $1-\alpha$ 가 되도록 $d$ 를 잡은 것이 HPD이다.

    한편 대칭성에서 $P(\theta < c-d \mid D) = P(\theta > c+d \mid D)$ 이고 둘의 합이 $\alpha$ 이므로 각각 $\alpha/2$ 이다. 이는 곧 $[c-d, c+d]$ 가 동일꼬리 구간이라는 뜻이다.

!!! note "쓰임새"
    정규분포와 모든 유효한 자유도의 Student-t 분포는 대칭 단봉이므로 동일꼬리 구간과 HPD 구간이 일치한다. 자유도가 커야만 이 성질을 갖는 것은 아니다.

## 4. 신용구간과 신뢰 구간

**정의 3.** [신뢰 구간]

빈도주의에서는 θ를 고정된 미지수로 두고 데이터에 따라 달라지는 구간 절차를 평가한다. 아래처럼 반복 표본추출의 포함확률이 정확히 1−α이면 정확 신뢰구간이다. 일반적으로는 적어도 1−α를 만족하는 보수적 구간이나 근사 구간도 사용한다.

$$
P_{D \mid \theta}\bigl(\hat L(D) \leq \theta \leq \hat U(D)\bigr) = 1-\alpha
$$

를 만족하면 이를 **$(1-\alpha)\times 100\%$ 신뢰 구간**이라 한다. 확률은 $\theta$ 가 아니라 $D$ 에 대해 잡힌다.

### 정리 4. 두 구간의 갈림 — 수는 같아도 뜻이 다르다 { .thm }

$x_1,\dots,x_n \sim \mathcal{N}(\mu, \sigma^2)$ 이고 $\sigma$ 를 안다고 하자. 평균에 균등 사전분포(부적절 사전분포) $p(\mu) \propto 1$ 을 두면 사후확률은 $\mathcal{N}(\bar x,\ \sigma^2/n)$ 이고, $95\%$ 신용구간과 $95\%$ 신뢰 구간은 **수치로 똑같이**

$$
\bar x \pm 1.96\,\frac{\sigma}{\sqrt n}
$$

이 된다. 그러나 두 진술이 뜻하는 바는 다르다. 앞의 것은 $\mu$ 에 대한 확률이고, 뒤의 것은 절차에 대한 확률이다.

??? proof "증명"

    사후확률은 $p(\mu \mid D) \propto p(D \mid \mu)\cdot 1 \propto \exp\!\bigl(-\tfrac{n}{2\sigma^2}(\mu-\bar x)^2\bigr)$ 이므로 $\mathcal{N}(\bar x, \sigma^2/n)$ 이다. 대칭이므로 정리 3에 따라 $95\%$ 신용구간은 $\bar x \pm 1.96\,\sigma/\sqrt n$ 이다.

    한편 빈도주의에서는 $\bar X \sim \mathcal{N}(\mu, \sigma^2/n)$ 이므로 $P(|\bar X - \mu| \leq 1.96\,\sigma/\sqrt n) \approx 0.95$ 이고, 이를 $\mu$ 에 대해 풀면 같은 구간이 나온다. 두 계산이 같은 수를 주는 것은 균등 사전분포가 우도를 그대로 두기 때문이다.

!!! note "쓰임새"
    수가 같다고 뜻이 같지는 않다. 신뢰 구간은 파라메터가 그 구간 안에 있을 확률이 $95\%$ 라는 뜻이 **아니다**. 빈도주의에서 $\theta$ 는 고정된 상수여서 구간 안에 있거나 없거나 둘 중 하나이고, 확률은 되풀이하는 절차에 붙는다.

| 클래스 | 신용구간(베이즈) | 신뢰 구간(빈도주의) |
|--------|------------------------------|-----------------------------------|
| **확률 진술** | 파라메터에 대해 | 절차에 대해 |
| **데이터** | 고정됨(관찰됨) | 확률적임(가상의 되풀이) |
| **파라메터** | 확률적임(분포를 갖는다) | 고정됨(모르는 상수) |
| **풀이** | "$\theta$ 가 여기 있을 확률이 $95\%$" | "그런 구간의 $95\%$ 가 $\theta$ 를 담는다" |

**문제 2.** <span class="diff med" title="중간"></span> 사전확률을 $\mathcal{N}(\mu_0, \tau^2)$ 으로 바꾸면 정리 4의 두 구간이 더는 일치하지 않음을 보이고, $\tau \to \infty$ 일 때 어떻게 되는지 밝히시오.

??? success "풀이"

    켤레 갱신에서 사후확률의 평균과 분산은

    $$
    \mu_n = \frac{\tau^{-2}\mu_0 + n\sigma^{-2}\bar x}{\tau^{-2} + n\sigma^{-2}},
    \qquad
    \sigma_n^2 = \bigl(\tau^{-2} + n\sigma^{-2}\bigr)^{-1}
    $$

    이다. 신용구간은 $\mu_n \pm 1.96\,\sigma_n$ 으로 중심이 $\bar x$ 에서 $\mu_0$ 쪽으로 끌려가고 폭도 $\sigma/\sqrt n$ 보다 좁아진다. 반면 신뢰 구간은 사전확률을 쓰지 않으므로 그대로다.

    $\tau \to \infty$ 이면 $\tau^{-2} \to 0$ 이므로 $\mu_n \to \bar x$, $\sigma_n^2 \to \sigma^2/n$ 이 되어 정리 4의 경우로 돌아간다.

## 5. 포함확률

신용구간이 빈도주의 잣대로 보아도 쓸 만한지는 포함확률로 따진다.

### 정리 5. 베이즈 구간의 평균 덮음 — 사전확률이 맞으면 정확하다 { .thm }

$\theta$ 가 사전확률 $p(\theta)$ 에서 뽑히고 그 $\theta$ 에서 데이터가 나온다고 하자. 각 데이터마다 $(1-\alpha)$ 신용구간 $C(D)$ 를 만들면, 사전확률에 대해 평균 낸 포함확률은 정확히 $1-\alpha$ 이다.

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

    이다. 안쪽 값은 신용구간의 정의에 따라 어떤 $D$ 에서도 $1-\alpha$ 이므로, 바깥 기댓값도 $1-\alpha$ 이다.

!!! note "쓰임새"
    이 등식은 정규화된 사전분포와 올바른 우도를 가정한 사전 평균 포함확률이다. 특정 고정 θ에서의 빈도주의 포함확률을 보장하지 않는다. 다른 방법보다 항상 보정이 좋다는 주장도 여기서 도출되지 않는다.

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

    이 시뮬레이션은 고정 true_p에서의 반복 포함 비율을 추정한다. 구간의 명목 확률 0.95와 시뮬레이션의 몬테카를로 오차를 함께 고려한다. 사전 평균 포함확률 정리와 서로 다른 양이며, 실행하지 않은 고정 수치 범위를 결과로 제시하지 않는다.

    Wald 구간은 성공률 추정이 0 또는 1이면 표준오차가 0이 되는 문제가 있다. 이 때문에 작은 표본·극단 확률에서 포함확률이 나빠질 수 있다. 베타 사후 분위수 구간은 이와 다른 방식으로 구성되지만 모든 고정 참값에서 95%를 보장하지는 않는다.

## 6. 실전에서 살필 것

구간과 함께 사전분포·우도·확률 수준·구성 방법을 보고한다. 동일꼬리 구간은 분위수로 계산하기 쉽고 단조 변환에도 대응한다. HPD는 선택한 파라메터 좌표에서 높은 밀도 영역을 요약하지만 리파라메리제이션에 따라 달라질 수 있다. 최단 구간이라는 성질만으로 어떤 결정 문제에도 최적이라고 볼 수는 없다.

다봉 분포에서는 분리된 HPD 집합이나 사후분포 그림을 함께 제공한다. 여러 파라메터의 개별 신용구간이 각각 95%라고 해서 동시 영역도 95%가 되지는 않는다. 결합 사후분포에서 다음 영역을 따로 정의해야 한다.

$$P(\theta\in\mathcal R\mid D)=1-\alpha$$

정규 결합 사후분포에서는 공분산에 따른 신용타원으로 나타낼 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 신용구간을 정의하고 빈도주의 신뢰 구간과 견주시오.

</div>

??? success "풀이"

    $95\%$ 신용구간 $[a,b]$ 는 $P(\theta \in [a,b] \mid D) = 0.95$ 를 뜻한다. 곧 데이터가 주어졌을 때 파라메터가 이 구간에 있을 확률이 $95\%$ 이다.

    $95\%$ 신뢰 구간은 실험을 여러 번 되풀이하면 그렇게 만든 구간의 $95\%$ 가 참 파라메터를 담는다는 뜻이다. 정리 4가 보였듯 특별한 경우에는 두 구간의 수가 같아지지만 뜻은 끝까지 다르다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $\text{Beta}(10, 30)$ 사후확률의 $95\%$ 신용구간을 셈하시오.

</div>

??? success "풀이"

    동일꼬리 구간은 Beta(10,30)의 0.025와 0.975 분위수로 정의한다. 평균은 10/40=0.25이며, 평균만으로 구간 경계를 정하지 않는다. 사후분포의 정확한 분위수를 사용해야 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\text{Gamma}(2, 1)$ 사후확률에 대해 동일꼬리 구간과 HPD 구간을 모두 계산하고, HPD가 더 짧은지 확인하시오.

</div>

??? success "풀이"

    Gamma(2,1)은 shape=2, rate=1로 정의한다. 동일꼬리 구간은 [F⁻¹(0.025),F⁻¹(0.975)]다. HPD의 두 내부 경계 L,U는 F(U)−F(L)=0.95와 f(L)=f(U)를 함께 만족한다. 이 단봉 비대칭 분포에서는 HPD가 더 짧으며 정확 경계는 이 조건으로 계산한다.

    동일꼬리 기준은 확률을, HPD 기준은 밀도 수준을 맞춘다. 비대칭 분포에서는 같은 꼬리 확률이 최단 길이를 주지 않을 수 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> MCMC 표본에서 HPD 구간을 셈하는 함수를 구현하시오.

</div>

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

    이 함수는 정렬 표본에서 하나의 연결된 최단 창을 찾는다. 다봉 사후분포의 분리된 HPD 집합을 복원하지 않으며, 한 봉우리의 질량이 부족하면 여러 봉우리와 골짜기를 함께 포함할 수 있다. 위의 compute_hpd_interval과는 인덱스 차이도 있어 이 함수의 끝점 창에는 ceil(credibility×n)+1개 순서통계량이 포함된다. 빈 표본·범위 밖 확률·충분한 유효 표본 수도 검사해야 한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> $2$ 차원 정규 사후확률의 $95\%$ 신용타원을 구현하고, 공분산 구조에 따라 영역이 어떻게 바뀌는지 밝히시오.

</div>

??? success "풀이"

    $\theta \sim \mathcal{N}(\mu, \Sigma)$ 이면 $(\theta-\mu)^\top \Sigma^{-1}(\theta-\mu) \sim \chi^2_2$ 이므로 신용영역은

    $$
    \mathcal{R} = \bigl\{\theta : (\theta-\mu)^\top \Sigma^{-1}(\theta-\mu) \leq \chi^2_{2,\,0.95}\bigr\}
    $$

    이고 $\chi^2_{2,\,0.95} \approx 5.991$ 이다. 이는 밀도 수준 집합이므로 정리 2에 따라 넓이가 가장 작은 영역이다.

    공분산이 양의 정부호일 때 축 방향은 고유벡터, 반축 길이는 √(5.991λ_i)다. 상관의 부호·분산 크기에 따라 그래디언트와 길이가 달라지므로 항상 같은 대각선 방향으로 눕는 것은 아니다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span> 작은 표본 크기($n = 5, 10, 20$)와 극단적인 참 확률($p = 0.05, 0.5, 0.95$)에서 신용구간과 신뢰 구간의 실제 포함확률을 견주시오.

</div>

??? success "풀이"

    각 (n,p) 조건에서 성공 횟수의 가능한 값 k=0~n을 구하고, 해당 구간이 p를 포함하는지 확인한 뒤 Binomial(n,p)의 확률로 합산하면 정확한 반복 포함확률을 계산할 수 있다. 시뮬레이션은 이를 근사하며 반복 수와 오차를 함께 보고한다.

    베이즈 구간도 특정 고정 p에서 포함확률이 낮거나 높을 수 있다. 사전 평균의 보장은 모든 고정 참값의 보장이 아니다. 따라서 90% 근처 등 고정된 범위를 일반적인 결과로 제시하지 않는다.

## 정리하며

신용구간은 모형·사전분포와 관측 데이터에 조건부인 사후확률을 요약한다. 동일꼬리 구간과 HPD 집합은 서로 다른 기준이며 다봉 분포에서는 HPD가 분리될 수 있다. 빈도주의 반복 포함확률과 사후확률을 구분하고, 단일 구간·동시 영역·샘플 기반 근사의 범위를 명시한다.

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2장
- Kruschke, J. *Doing Bayesian Data Analysis* (2nd ed.), 12장
- Hoff, P. *A First Course in Bayesian Statistical Methods*, 3장
