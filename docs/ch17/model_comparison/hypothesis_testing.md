# 베이즈 가설 검정과 모델 비교

베이즈 가설 검정은 각 가설의 파라메터 사전분포로 평균한 우도, 즉 주변 우도를 비교한다. 베이즈 인자의 방향과 모델 사전확률을 구분하고, 동전 예제로 직접 계산과 새비지-디키 밀도비의 관계를 살펴본다.

## 1. 베이즈 인자

두 가설 H₁·H₂의 베이즈 인자는 다음과 같다.

$$BF_{12}=\frac{p(D\mid H_1)}{p(D\mid H_2)},\qquad p(D\mid H_i)=\int p(D\mid\theta,H_i)p(\theta\mid H_i)\,d\theta$$

같은 관측 자료를 비교하고 각 파라메터 사전분포를 명시해야 한다. 주변 우도가 양수·유한이면 BF₂₁=1/BF₁₂이고, 동일한 모델 정의를 사용한 BF₁₃=BF₁₂ BF₂₃이다.

BF₁₂>1이면 상대적으로 H₁을, <1이면 H₂를 지지한다. 그 자체가 가설의 사후확률은 아니다. 관행적인 증거 등급은 해석 보조 수단이며 오류율이나 모델의 절대 적합도를 보장하지 않는다. BF는 모델 사전 오즈와는 별개지만 **모델 내부 파라메터의 사전분포**에는 의존한다.

## 2. 사후확률 승산

베이즈 정리에서 모델의 사후 오즈는 다음과 같다.

$$\frac{p(H_1\mid D)}{p(H_2\mid D)}=BF_{12}\frac{p(H_1)}{p(H_2)}$$

두 가설만 고려하고 사전확률이 합해서 1이면 다음과 같다.

$$p(H_1\mid D)=\frac{BF_{12}p(H_1)}{BF_{12}p(H_1)+p(H_2)}$$

같은 사전확률과 BF₁₂=10이면 사후확률은 10/11≈0.909다. 다른 모델 후보나 사전확률을 사용하면 값이 달라진다. 두 모델이 모두 부적절해도 상대 비교는 한쪽을 지지할 수 있다.

## 3. 동전이 공정한지 검정하기

### 3.1 문제의 구조

**가설:**

- $H_0$: $\theta = 0.5$(공정한 동전)
- $H_1$: $\theta \neq 0.5$(치우친 동전이며 파라메터 사전분포는 $\theta \sim \text{Beta}(\alpha, \beta)$)

**데이터:** $n$번 던져 앞면 $k$번

### 3.2 주변 우도

**$H_0$ 아래**(점 가설):

$$
p(D|H_0) = \binom{n}{k} (0.5)^n
$$

**$H_1$ 아래**(베타-이항):

$$
p(D|H_1) = \binom{n}{k} \frac{B(k + \alpha, n - k + \beta)}{B(\alpha, \beta)}
$$

여기서 $B(\cdot, \cdot)$은 베타 함수이다.

### 3.3 베이즈 인자 계산

$$
BF_{10} = \frac{p(D|H_1)}{p(D|H_0)} = \frac{B(k + \alpha, n - k + \beta)}{B(\alpha, \beta) \cdot (0.5)^n}
$$

### 3.4 구현

α,β>0이고 성공·실패 수가 음이 아닌 정수여야 한다. 아래 함수의 stats.binom.comb는 공식 API가 아니므로 이 코드를 그대로 완성된 계산기로 사용할 수 없다. 조합 함수는 [scipy.special.comb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html)에 있다. 큰 표본에서는 로그 주변 우도 차이로 계산해 언더플로를 피한다.

```python
import numpy as np
from scipy import stats
from scipy.special import beta as beta_func

def bayes_factor_coin_fairness(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    H1(치우침) 대 H0(공평한 동전)의 베이즈 인자를 셈한다.
    
    파라메터
    ----------
    n_heads, n_tails : int
        관측 자료
    prior_alpha, prior_beta : float
        H1 아래의 베타 앞확률 파라메터
    
    반환값
    -------
    bf : float
        베이즈 인자(H1 대 H0)
    """
    n = n_heads + n_tails
    
    # H0 아래 증거: θ = 0.5
    evidence_h0 = stats.binom.pmf(n_heads, n, 0.5)
    
    # H1 아래 증거: 베타-이항 주변 가능도
    evidence_h1 = (beta_func(n_heads + prior_alpha, n_tails + prior_beta) / 
                   beta_func(prior_alpha, prior_beta))
    evidence_h1 *= stats.binom.comb(n, n_heads)
    
    bf_h1_vs_h0 = evidence_h1 / evidence_h0
    
    return bf_h1_vs_h0
```

### 3.5 계산 예

**데이터:** 앞면 17번, 뒷면 3번. 아래 표는 H₁의 Beta(1,1) 사전분포에 대한 수식 계산이다.

```python
bf = bayes_factor_coin_fairness(n_heads=17, n_tails=3)
# bf ≈ 17.5
```

| 양 | 값 |
|----------|-------|
| $H_0$ 아래의 증거 | 약 0.00108719 |
| $H_1$ 아래의 증거 | 1/21≈0.04761905 |
| 베이즈 인자 $BF_{10}$ | 약 43.8002 |
| 같은 모델 사전확률에서 $p(H_1|D)$ | 약 0.97768 |

---

## 4. 새비지-디키 밀도비

### 4.1 방법

**점 귀무가설** H₀:θ=θ₀가 H₁에 포함되고 우도가 호환되는 경우, 다음 밀도비를 사용할 수 있다. H₁의 사전밀도는 θ₀에서 유한한 양수여야 한다. 추가 nuisance 파라메터 ψ가 있으면 H₀의 사전분포가 H₁에서 θ=θ₀를 조건으로 한 ψ의 사전분포와 일치하는 등의 조건이 필요하다. 단순히 모델이 중첩된다는 사실만으로 성립하지 않는다.

$$
\boxed{BF_{01} = \frac{p(\theta_0|D, H_1)}{p(\theta_0|H_1)}}
$$

이는 귀무가설 값 $\theta_0$에서 잰 **사후밀도와 사전밀도의 비**이다.

### 4.2 직관

- $\theta_0$에서 사후밀도가 사전밀도보다 **낮으면** → 데이터가 확률 질량을 $\theta_0$에서 멀리 옮긴 것 → $H_0$에 맞서는 증거
- $\theta_0$에서 사후밀도가 사전밀도보다 **높으면** → 데이터가 확률 질량을 $\theta_0$ 쪽으로 옮긴 것 → $H_0$을 편드는 증거

### 4.3 구현

아래 베타 예제는 추가 nuisance 파라메터가 없는 경우다. null_value는 0과 1 사이에 두며 밀도 비가 유한한지 확인한다. 극단적인 표본에서는 밀도 대신 로그밀도의 차이를 사용해야 한다.

```python
from scipy import stats

def savage_dickey_bf(n_heads, n_tails, prior_alpha=1, prior_beta=1, null_value=0.5):
    """
    새비지-디키 밀도 비로 베이즈 인자를 셈한다.
    
    BF_{01}(귀무 대 대립)과 BF_{10}(대립 대 귀무)을 되돌린다.
    """
    # 앞확률과 뒤확률 분포
    prior = stats.beta(prior_alpha, prior_beta)
    posterior = stats.beta(prior_alpha + n_heads, prior_beta + n_tails)
    
    # 귀무값에서 밀도 값 매기기
    prior_density = prior.pdf(null_value)
    posterior_density = posterior.pdf(null_value)
    
    # 새비지-디키 비
    bf_h0_vs_h1 = posterior_density / prior_density
    bf_h1_vs_h0 = 1 / bf_h0_vs_h1
    
    return bf_h0_vs_h1, bf_h1_vs_h0
```

### 4.4 보기

**데이터:** 앞면 17번, 뒷면 3번. 아래 표는 H₁의 Beta(1,1) 사전분포에 대한 수식 계산이다.
**검정:** $H_0: \theta = 0.5$

```python
bf_01, bf_10 = savage_dickey_bf(n_heads=17, n_tails=3)
```

| 양 | 값 |
|----------|-------|
| θ=0.5에서의 사전밀도 | 1 |
| θ=0.5에서의 사후밀도 | 약 0.02283096 |
| BF₀₁ | 약 0.02283096 |
| BF₁₀ | 약 43.8002 |

$\theta = 0.5$에서 사후밀도가 사전밀도보다 훨씬 낮은데, 이는 데이터가 확률 질량을 공정한 동전 가설에서 멀리 옮겼음을 뜻한다.

---

## 5. 베이즈 검정과 빈도주의 검정

베이즈 인자는 지정한 두 모델의 주변 우도 비를 계산한다. p값은 귀무가설 아래 지정한 검정통계량이 관측값 이상으로 극단적일 확률이다. 어느 것도 가설이 참일 확률로 바로 읽을 수 없다. 모델 사후확률에는 별도의 모델 사전확률이 필요하다.

큰 p값이 귀무가설의 진실을 증명하지는 않는다. 다만 빈도주의에도 동등성 검정·비열등성 검정이나 순차 설계가 있으므로 모든 대안을 고정 표본 t검정으로 한정해서는 안 된다. 베이즈 인자 역시 대립가설 사전분포와 비교 대상에 민감하다.

적절한 사전분포와 무시 가능한 중단 규칙 아래 사후 추론이 중단 계획에 의존하지 않을 수 있지만, 임의의 BF나 사후확률 기준이 고정 귀무가설에서 거짓 양성률을 자동 통제하지는 않는다. 큰 표본에서 작은 p값과 귀무가설을 지지하는 BF가 함께 나오는 린들리 현상은 서로 다른 비교 기준과 고정된 대립 사전 스케일을 고려해 해석한다.

## 6. 실전에서 살필 것

파라메터 사전분포와 모델 사전확률을 구분해 보고한다. 대립가설이 허용하는 효과 크기·스케일이 자료의 단위와 목적에 맞는지 살피고 민감도 분석을 한다. 부적절 사전분포의 임의 정규화 상수는 일반적인 베이즈 인자를 정의하지 못하게 한다. 제프리스 사전이라는 이름만으로 모델 비교에 적절하다고 판단해서는 안 된다.

주변 우도는 해석적 적분, 중요도·브리지 샘플링, 열역학 적분 등으로 계산할 수 있다. 가역 점프 MCMC는 모델 간 이동을 통해 사후 모델확률을 추정하는 방법이다. 각 계산의 수렴과 수치 오차를 확인한다.

이 동전 예제는 Beta(1,1) 대립 사전에서 BF₁₀≈43.80이며 동일한 모델 사전확률이면 H₁의 사후확률은 약 0.978이다. 이것은 공정성의 절대적 증명이나 임의의 다른 대립 사전에 대한 결과가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
앞면 17번·뒷면 3번에서 균등 대립 사전 아래 두 주변 우도를 구하라.

</div>

??? success "연습문제 1 풀이"
    횟수 자료이므로 $p(D|H_0)=\binom{20}{17}2^{-20}\approx0.00108719$, $p(D|H_1)=1/21$이다. BF₁₀≈43.8002다. 관측열의 순서까지 고정하면 두 우도에서 동일한 조합 계수가 빠지므로 비는 같다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 모델 사전확률에서 BF₁₀=43.8002를 사후확률로 변환하라.

</div>

??? success "연습문제 2 풀이"
    두 모델만 고려하면 p(H₁|D)=BF₁₀/(1+BF₁₀)≈0.97768이다. 사전확률이 다르면 사전 오즈를 먼저 곱해야 하며 BF 자체를 97.8%라고 읽지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
새비지-디키 비를 nuisance 파라메터가 있는 중첩 모형에 자동 적용할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 귀무모형의 nuisance 사전분포가 대립모형에서 귀무값을 조건으로 한 사전분포와 호환되는지 등을 확인해야 한다. 그렇지 않으면 보정 또는 직접 주변 우도 계산이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트와 위 직접 계산 함수만으로 올바른 BF 계산을 보장할 수 있는가?

</div>

??? success "연습문제 4 풀이"
    아니다. 임포트만으로 분석이 수행되지 않고 직접 계산 함수에는 공식 API가 아닌 조합 함수 호출이 있다. 파라메터 유효성·방향·관측 자료의 정의와 수치 안정성을 확인하고 수식 또는 독립 계산과 일치하는지 검증해야 한다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

베이즈 인자는 주변 우도의 비이며 사후 오즈를 갱신하는 배수다. 파라메터 사전분포, 모델 사전확률, BF의 방향을 구분한다. 새비지-디키 비는 사전분포 호환 조건이 필요하며, 베이즈 비교와 p값은 서로 다른 질문에 답한다.

**참고 문헌**

- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values. *Psychonomic Bulletin & Review*, 14(5), 779-804.
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 7장
