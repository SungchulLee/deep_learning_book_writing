# 체르노프 한계

마구잡이 알고리즘을 살필 때는 아무 변수가 그 평균 가까이 모인다는 것, 곧 크게 벗어나는 일이 그저 드문 정도가 아니라 *지수로* 드물다는 것을 보여야 할 때가 많다. 마르코프 부등식과 체비쇼프 부등식은 다항식 꼬리 한계를 준다. 그러나 서로 매이지 않은 아무 변수의 합에서는 **체르노프 한계** 재주가 지수로 줄어드는 꼬리 확률을 준다. 이 한계는 알고리즘과 자료 짜임의 확률 살피기에서 일꾼 노릇을 한다.

---

## 1. 적률 만들개 함수 방법

체르노프 한계 재주는 적률 만들개 함수가 유한한 아무 변수 $X$에 다 쓸 수 있다. 아무 $t > 0$에 대해,

$$
\Pr[X \geq a] = \Pr[e^{tX} \geq e^{ta}] \leq \frac{E[e^{tX}]}{e^{ta}}
$$

첫 걸음은 음이 아닌 아무 변수 $e^{tX}$에 마르코프 부등식을 쓴 것이다. $t > 0$에 대해 가장 좋게 하면 가장 빡빡한 한계가 나온다.

---

## 2. 서로 매이지 않은 베르누이 변수 합의 체르노프 한계

$X_1, X_2, \ldots, X_n$을 $\Pr[X_i = 1] = p_i$인 서로 매이지 않은 베르누이 아무 변수라 하자. $X = \sum_{i=1}^{n} X_i$이고 $\mu = E[X] = \sum_{i=1}^{n} p_i$이라 하자.

**위 꼬리.** 아무 $\delta > 0$에 대해,

$$
\Pr[X \geq (1 + \delta)\mu] \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu
$$

**아래 꼬리.** 아무 $0 < \delta < 1$에 대해,

$$
\Pr[X \leq (1 - \delta)\mu] \leq \left(\frac{e^{-\delta}}{(1-\delta)^{(1-\delta)}}\right)^\mu
$$

---

## 3. 단순하게 만든 꼴

정확한 체르노프 한계는 쓰기 쉽도록 흔히 단순하게 만든다.

**위 꼬리(단순한 꼴).** $\delta \in (0, 1]$에 대해,

$$
\Pr[X \geq (1 + \delta)\mu] \leq e^{-\mu \delta^2 / 3}
$$

$\delta > 1$(크게 벗어남)에 대해,

$$
\Pr[X \geq (1 + \delta)\mu] \leq e^{-\mu \delta / 3}
$$

**아래 꼬리(단순한 꼴).** $0 < \delta < 1$에 대해,

$$
\Pr[X \leq (1 - \delta)\mu] \leq e^{-\mu \delta^2 / 2}
$$

**양쪽 한계.** 두 꼬리를 아우르면,

$$
\Pr[|X - \mu| \geq \delta \mu] \leq 2e^{-\mu \delta^2 / 3}
$$

!!! tip "어떤 꼴을 쓸까"
    웬만큼 벗어날 때($\delta \leq 1$)는 단순한 꼴 $e^{-\mu\delta^2/3}$을 쓰라. 크게 벗어날 때($\delta > 1$)는 한계 $e^{-\mu\delta/3}$이 더 빡빡하다. 가장 날카로운 결과를 얻으려면 정확한 꼴을 쓰고 $t$에 대해 가장 좋게 하라.

---

## 4. 밝힘 밑그림(위 꼬리)

$\Pr[X_i = 1] = p_i$인 서로 매이지 않은 $X_i$에 대해,

$$
E[e^{tX}] = \prod_{i=1}^{n} E[e^{tX_i}] = \prod_{i=1}^{n} (1 - p_i + p_i e^t)
$$

$1 + x \leq e^x$을 쓰면,

$$
E[e^{tX}] \leq \prod_{i=1}^{n} e^{p_i(e^t - 1)} = e^{\mu(e^t - 1)}
$$

따라서,

$$
\Pr[X \geq (1+\delta)\mu] \leq \frac{e^{\mu(e^t - 1)}}{e^{t(1+\delta)\mu}}
$$

$t = \ln(1 + \delta)$으로 두면 오른쪽이 가장 작아져 다음이 나온다.

$$
\Pr[X \geq (1+\delta)\mu] \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu
$$

$\square$

---

## 5. 알고리즘 살피기에서의 쓰임새

### 마구잡이 짐 고르게 나누기

공 $n$개를 통 $n$개에 던지면 어느 통이든 짐은 $\mu = 1$인 $X \sim \text{Binomial}(n, 1/n)$이다. 체르노프 한계에 따라,

$$
\Pr[X \geq c \ln n] \leq \left(\frac{e}{c \ln n}\right)^{c \ln n}
$$

$c = 3/\ln\ln n$으로 두고 통 $n$개에 합집합 한계를 쓰면 최대 짐 한계 $O(\ln n / \ln \ln n)$이 다시 나온다.

### 뽑기와 어림

믿음 $1 - \delta$으로 상대 어긋남 $\epsilon$ 안에서 확률 $p$을 어림하려면 표본을 $n = O(\frac{1}{\epsilon^2 p} \ln(1/\delta))$개 뽑아라. 체르노프 한계는 표본 평균이 적어도 $1 - \delta$의 확률로 $(1 \pm \epsilon)p$ 안에 있음을 보장한다.

### 그물에서 길 잡기

초입방체 위의 마구잡이 길 잡기에서 꾸러미마다 서로 매이지 않게 아무 중간 마디를 고른다. 체르노프 한계는 어느 모서리든 최대 붐빔이 높은 확률로 $O(\sqrt{n \log n})$임을 보인다.

---

## 6. 꼬리 한계 견주기

| 한계 | 꼬리 줄어듦 | 조건 |
|---|---|---|
| 마르코프 | $O(1/a)$ | 음이 아닌 $X$ |
| 체비쇼프 | $O(1/a^2)$ | 유한한 흩어짐 |
| 체르노프 | $e^{-\Omega(a)}$ | 서로 매이지 않음, 가둔 변수 |
| 회프딩 | $e^{-\Omega(a^2/n)}$ | 서로 매이지 않음, 가둔 범위 |

체르노프 한계는 가장 센 보장을 주지만 가장 센 가정(서로 매이지 않음과 가둔 변수)이 필요하다.

---

## 7. 회프딩 부등식

가까이 이어진 한계가 (베르누이일 필요는 없는) 가둔 서로 매이지 않은 아무 변수에 들어맞는다. $X_i \in [a_i, b_i]$이 서로 매이지 않았다면,

$$
\Pr\left[\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| \geq t\right] \leq 2\exp\left(\frac{-2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)
$$

여기서 $\mu = E[\frac{1}{n}\sum X_i]$이다.

---

## 8. 구현

```python
"""
체르노프 한계: 이론의 한계와 겪어 본 꼬리 확률.

이론의 내다봄과 베르누이 합의 몬테카를로 흉내 내기를 견주어
체르노프 한계가 얼마나 빡빡한지 보인다.
"""

import random
import math

# === 체르노프 한계 공식 ===

def chernoff_upper(mu, delta):
    """정확한 체르노프 위쪽 꼬리 한계: Pr[X >= (1+delta)*mu].

    인수:
        mu: X의 기댓값.
        delta: 상대 벗어남(delta > 0).

    반환값:
        꼬리 확률의 위 한계.
    """
    if delta <= 0:
        return 1.0
    exponent = mu * (delta - (1 + delta) * math.log(1 + delta))
    return math.exp(exponent)

def chernoff_upper_simplified(mu, delta):
    """단출한 체르노프 위쪽 꼬리 한계: exp(-mu * delta^2 / 3)."""
    return math.exp(-mu * delta ** 2 / 3)

def chernoff_lower_simplified(mu, delta):
    """단출한 체르노프 아래쪽 꼬리 한계: exp(-mu * delta^2 / 2)."""
    if delta <= 0 or delta >= 1:
        return 1.0
    return math.exp(-mu * delta ** 2 / 2)

# === 몬테카를로 어림 ===

def estimate_tail_prob(n, p, threshold, trials=100000):
    """몬테카를로 흉내내기로 Pr[X >= threshold]을 어림한다.

    X = sum of n independent Bernoulli(p) random variables.
    """
    count = 0
    for _ in range(trials):
        x = sum(1 for _ in range(n) if random.random() < p)
        if x >= threshold:
            count += 1
    return count / trials

# === 메인 ===

if __name__ == "__main__":
    random.seed(42)

    n = 100
    p = 0.3
    mu = n * p  # = 30

    print(f"X ~ Binomial(n={n}, p={p}), mu = {mu}")
    print(f"{'delta':>8} {'threshold':>10} {'Chernoff':>10} "
          f"{'Simplified':>12} {'Empirical':>10}")
    print("-" * 55)

    for delta in [0.2, 0.4, 0.6, 0.8, 1.0]:
        threshold = (1 + delta) * mu
        bound_exact = chernoff_upper(mu, delta)
        bound_simple = chernoff_upper_simplified(mu, delta)
        empirical = estimate_tail_prob(n, p, threshold)

        print(f"{delta:8.1f} {threshold:10.0f} {bound_exact:10.6f} "
              f"{bound_simple:12.6f} {empirical:10.5f}")
```

**출력:**
```
X ~ Binomial(n=100, p=0.3), mu = 30.0
   delta  문턱   체르노프   단순한 꼴  겪어 본 값
-------------------------------------------------------
     0.2         36   0.234960     0.670320    0.12540
     0.4         42   0.026826     0.188876    0.00672
     0.6         48   0.001558     0.022091    0.00009
     0.8         54   0.000048     0.001069    0.00000
     1.0         60   0.000001     0.000022    0.00000
```

---

## 연습문제

**연습문제 1.**
체르노프 한계의 핵심 마구잡이 재주와 그것이 정해진 방식보다 나은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    체르노프 한계은 마구잡이를 써서 정해진 알고리즘이 마주칠 수 있는 가장 나쁜 들임을 피한다. 아무렇게나 고르므로 알고리즘의 솜씨가 들임의 짜임이 아니라 제 동전 던지기에 달린다. 그래서 모든 들임에 대해 참인 센 기댓값 시간이나 높은 확률의 보장을 흔히 얻으며, 짓궂거나 병리적인 경우를 걱정할 까닭이 없어진다. $\square$

---

**연습문제 2.**
체르노프 한계의 기댓값 시간 복잡도는 얼마인가? 가장 나쁜 경우의 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    기댓값 시간 복잡도는 흔히 $O(n)$이나 $O(n \log n)$이며 높은 확률로 이룬다. 가장 나쁜 경우는 다항식만큼 더 나쁠 수 있지만(예컨대 $O(n^2)$) 그럴 확률은 무시할 만큼 작다. 기댓값과 가장 나쁜 경우의 틈이 마구잡이의 값이며, 가장 나쁜 움직임이 일어날 확률은 들임 크기에 따라 지수로 줄어든다. $\square$

---

**연습문제 3.**
체르노프 한계은 라스베이거스 알고리즘인가 몬테카를로 알고리즘인가? 그 차이를 설명하라.

??? success "연습문제 3 풀이"
    **라스베이거스**: 늘 옳은 결과를 내며 도는 시간이 아무 변수이다(기댓값이 다항식). **몬테카를로**: 늘 다항식 시간에 돌지만 결과가 어떤 가둔 확률로 틀릴 수 있다. 체르노프 한계은 옳음을 보장하느냐 도는 시간을 보장하느냐에 따라 이 가운데 하나에 든다. 이 가름이 어긋날 확률을 어떻게 다룰지 정한다. $\square$

---

**연습문제 4.**
체르노프 한계에서 마구잡이를 없애거나 솜씨가 나쁠 확률을 줄이는 법을 설명하라.

??? success "연습문제 4 풀이"
    방책은 다음과 같다. (1) **거듭 해 보기**: 알고리즘을 여러 번 돌려 가장 좋거나 많은 쪽 결과를 택하면 어긋날 확률이 지수로 줄어든다. (2) **마구잡이 없애기**: 조건부 기댓값이나 흩는 함수 무리로 아무 고르기를 정해진 고르기로 바꾼다. (3) **키우기**: 몬테카를로 알고리즘에서는 $k$번 되풀이해 어긋남을 $2^{-k}$으로 줄인다. (4) **비슷 마구잡이 만들개**: 알고리즘이 보기에 "마구잡이처럼 보이는" 정해진 차례를 쓴다. $\square$

## 정리하며

이 마당은 적률 만들개 함수 방법、서로 매이지 않은 베르누이 변수 합의 체르노프 한계、단순하게 만든 꼴、밝힘 밑그림(위 꼬리)을 차례로 짚었다.

**참고 문헌**

- Motwani, R. & Raghavan, P. *Randomized Algorithms*. Cambridge University Press, 1995.
- Mitzenmacher, M. & Upfal, E. *Probability and Computing*. Cambridge University Press, 2017.
