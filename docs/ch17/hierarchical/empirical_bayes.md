# 경험적 베이즈

경험적 베이즈는 여러 그룹의 자료에서 공통 사전분포의 하이퍼파라미터를 추정한 뒤 그 값을 대입해 그룹별 추론을 수행한다. 관련된 집단의 정보를 공유하는 실용적인 방법이지만, 하이퍼파라미터 자체의 불확실성을 적분하는 계층 베이즈 추론과는 구분해야 한다.

## 1. 경험적 베이즈의 틀

그룹 i의 관측과 파라메터에 다음 모형을 둔다.

$$y_i\mid\theta_i\sim p(y_i\mid\theta_i),\qquad \theta_i\mid\eta\sim p(\theta_i\mid\eta)$$

고정된 η를 사용하는 베이즈 분석은 이를 사전에 정한다. 경험적 베이즈는 자료로 η를 추정하고 조건부 사후분포에 대입한다. 완전한 계층 베이즈 분석은 η에 하이퍼사전분포를 두고 그 사후 불확실성도 적분한다.

최대 주변 우도 방식은 η 아래 그룹들이 독립이라는 가정에서 다음 값을 구한다.

$$\hat\eta=\arg\max_\eta\prod_i\int p(y_i\mid\theta_i)p(\theta_i\mid\eta)\,d\theta_i$$

그런 다음 $p(\theta_i\mid y_i,\hat\eta)$로 추론한다. 이 분포는 η의 추정 오차를 포함하지 않는다.

## 2. 근사 방법

### 2.1 최대 주변 우도

잠재 파라메터를 적분한 우도를 최적화한다. 켤레성 덕분에 적분식이 명시적으로 주어져도 하이퍼파라미터의 최적값까지 닫힌 형태로 구해지는 것은 아니다. 경계해와 식별성, 수치 최적화의 수렴을 확인해야 한다.

### 2.2 적률법

잠재 확률 θ 자체를 관측한 것처럼 취급하면 베타분포의 평균 m, 분산 v에서 다음 식을 얻는다.

$$\hat\alpha=m\left(\frac{m(1-m)}v-1\right),\qquad \hat\beta=(1-m)\left(\frac{m(1-m)}v-1\right)$$

유효한 양의 파라메터를 얻으려면 $0<m<1$, $0<v<m(1-m)$이어야 한다. 하지만 실제 타율 y_i/n_i에는 이항 표본 잡음이 포함된다. 같은 타수 n을 가정해도 관측 타율의 분산은 다음과 같다.

$$\operatorname{Var}(y_i/n)=\operatorname{Var}(\theta_i)+\mathbb E[\theta_i(1-\theta_i)]/n$$

관측 타율의 분산을 그대로 v에 대입하면 잠재 확률의 분산과 잡음을 혼동한다. 타수가 서로 다르면 각 n_i를 반영한 베타-이항 주변 우도 등의 접근이 더 적절하다.

## 3. 야구 타율 예제

### 3.1 문제의 구조

경험적 베이즈의 고전적인 쓰임새로, 시즌 초 데이터에서 참된 타격 능력을 추정하는 일이다.

**모델:**

- $y_i$ = 선수 $i$의 안타 수
- $n_i$ = 선수 $i$의 타수
- $\theta_i$ = 참된 타격 능력(안타를 칠 확률)

**계층 구조:**

$$
y_i | \theta_i \sim \text{Binomial}(n_i, \theta_i)
$$

$$
\theta_i \sim \text{Beta}(\alpha, \beta)
$$

### 3.2 최대 우도의 문제

타수가 양수일 때 선수별 최대 우도 추정값은 다음과 같다.

$$
\hat{\theta}_i^{\text{MLE}} = \frac{y_i}{n_i}
$$

**문제:** $n_i$가 작으면 이 추정값이 몹시 흔들린다. 10타수 3안타(0.300)인 선수와 100타수 25안타(0.250)인 선수의 참된 능력이 같을 수도 있다.

### 3.3 경험적 베이즈의 해

**1단계:** 관측 타율에 베타 적률 공식을 적용한다. 아래 함수는 관측 잡음을 분리하지 않는 간단한 대입 예제다. 빈 입력, 분산 0, 평균 0·1, 비양수 추정 파라메터를 처리하지 않는다.

```python
import numpy as np

def estimate_beta_prior(observed_avgs):
    """적률법으로 베타 앞확률 파라메터를 어림한다."""
    mean_obs = np.mean(observed_avgs)
    var_obs = np.var(observed_avgs)
    
    # alpha, beta 풀기
    common_factor = mean_obs * (1 - mean_obs) / var_obs - 1
    alpha_eb = mean_obs * common_factor
    beta_eb = (1 - mean_obs) * common_factor
    
    return alpha_eb, beta_eb
```

**2단계:** 유효한 양의 추정 파라메터를 조건으로 베타-이항 사후평균을 계산한다.

$$
\hat{\theta}_i^{\text{EB}} = \frac{y_i + \hat{\alpha}}{n_i + \hat{\alpha} + \hat{\beta}}
$$

### 3.4 구현

각 안타 수는 0 이상 타수 이하의 정수이고 타수는 양수여야 한다. 두 배열은 같은 형태여야 한다. 추정한 사전분포를 대입하는 아래 함수는 하이퍼파라미터의 사후분포를 계산하지 않는다.

```python
def empirical_bayes_batting(hits, at_bats):
    """
    타율에 대한 경험 베이즈 어림.
    
    파라메터
    ----------
    hits : array
        선수마다의 안타 수
    at_bats : array
        선수마다의 타수
    
    반환값
    -------
    MLE와 EB 어림값을 담은 dict
    """
    # MLE 어림값
    mle_estimates = hits / at_bats
    
    # 베타 앞확률 어림하기(적률법)
    alpha_eb, beta_eb = estimate_beta_prior(mle_estimates)
    
    # 경험 베이즈 어림값(뒤확률 평균)
    eb_estimates = (hits + alpha_eb) / (at_bats + alpha_eb + beta_eb)
    
    return {
        'mle': mle_estimates,
        'eb': eb_estimates,
        'alpha': alpha_eb,
        'beta': beta_eb,
        'prior_mean': alpha_eb / (alpha_eb + beta_eb)
    }
```

### 3.5 결과 해석

수축은 표본 타율의 분산을 줄이는 대신 공통 중심 쪽 바이어스를 도입한다. 실제 MSE를 비교하려면 참 확률, 타수 분포와 반복 실험 조건이 필요하다. 위 함수만으로 특정 개선율이나 모든 선수에서의 성능 우위를 주장할 수 없다.

---

## 4. 수축 효과

추정한 사전분포의 평균을 $\hat\mu=\hat\alpha/(\hat\alpha+\hat\beta)$라 하면 다음과 같다.

$$\hat\theta_i^{\mathrm{EB}}=w_i\frac{y_i}{n_i}+(1-w_i)\hat\mu,\qquad w_i=\frac{n_i}{n_i+\hat\alpha+\hat\beta}$$

w_i는 **데이터 웨이트**, 1−w_i는 **수축 비율**이다. 추정한 사전 파라메터를 고정하면 타수가 작을수록 수축 비율이 커진다. 같은 타수에서 관측 타율이 중심에서 멀수록 절대 이동량은 크지만 수축 비율 자체는 같다.

모형이 적절하면 편향과 분산의 균형으로 평균적인 추정 오차를 줄일 수 있다. 공통 사전분포가 맞지 않는 그룹에는 수축이 오히려 해로울 수 있다.

## 5. 제임스-스타인 추정과의 이음

독립인 $Y_i\sim N(\theta_i,\sigma^2)$에서 공통 분산 σ²가 알려져 있고 p≥3이라 하자. 고정된 목표 벡터 a로 수축하는 고전적인 추정량은 다음과 같다.

$$\hat{\boldsymbol\theta}^{\mathrm{JS}}=\boldsymbol a+\left(1-\frac{(p-2)\sigma^2}{\|\boldsymbol Y-\boldsymbol a\|^2}\right)(\boldsymbol Y-\boldsymbol a)$$

이는 총 제곱 오차의 기댓값인 위험에서 Y를 개선한다. 모든 표본이나 각 좌표에서 오차가 작다는 뜻은 아니다. 분모가 0인 점은 확률 0이며 별도 값을 정할 수 있다.

목표를 같은 자료의 평균 $\bar Y\boldsymbol1$로 추정하면 잔차 공간의 차원이 p−1이므로 표준적인 계수는 p−3이고 비자명한 개선은 p≥4에서 다룬다. 고정 목표의 p−2 공식을 그대로 적용해서는 안 된다.

정규 계층 모형의 사후평균도 유사한 수축 형태를 가진다. 다만 단순히 하이퍼파라미터의 최대 우도 추정값을 대입하면 항상 위 공식이 나온다는 뜻은 아니다. 어떤 추정 방식과 위험 기준을 사용하는지 명시해야 한다.

## 6. 이점과 한계

관련된 파라메터가 많을 때 공통 분포를 자료에서 추정해 정보를 공유할 수 있다. 켤레 모형에서는 그룹별 사후 계산이 간단하며, 반복 표본에서의 위험을 줄이는 유용한 수축 추정량을 만들 수 있다. 일반 모형에서는 주변 적분이나 최적화에 상당한 계산이 필요할 수도 있다.

하이퍼파라미터 추정값을 고정하면 그 불확실성을 전달하지 않아 구간이 지나치게 좁아질 수 있다. 같은 자료가 공통 분포 추정과 그룹 추론에 사용된다는 사실 자체가 방법을 무효로 만들지는 않는다. 다만 고정 사전분포를 전제로 한 보장을 그대로 적용할 수는 없다. 사전분포의 형태와 그룹의 교환가능성 가정도 여전히 선택해야 한다.

## 7. 온전한 베이즈와의 비교

경험적 베이즈는 추정한 η를 조건으로 추론하고, 완전한 계층 베이즈는 η의 사후분포에 대해 적분한다. 후자도 모형·사전분포와 수치 계산이 부적절하면 좋은 추론을 보장하지 않는다.

관련 그룹이 충분히 많고 식별성과 정규성 조건이 적절하면 η의 불확실성이 작아져 두 결과가 가까워질 수 있다. 그룹 수가 적거나 분산이 경계에 가까운 경우에는 차이가 클 수 있다. 계산 방식의 이름만으로 구간의 실제 커버리지나 예측 성능을 판단하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
공통 사전 파라메터를 고정할 때 타수와 수축 비율은 어떤 관계인가?

</div>

??? success "연습문제 1 풀이"
    수축 비율은 (α+β)/(n_i+α+β)이며 타수가 증가하면 감소한다. 중심으로부터의 절대 이동량은 이 비율과 관측 타율·사전평균 차이의 곱이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
관측 타율의 표본분산을 베타 사전분포의 분산으로 사용하면 무엇을 혼동하는가?

</div>

??? success "연습문제 2 풀이"
    선수 간 참 확률의 변동과 각 선수 타율의 이항 표본 잡음을 혼동한다. 특히 타수가 적거나 다르면 잡음 기여가 크거나 다르므로 각 타수를 반영한 추정이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
제임스-스타인의 위험 개선은 각 선수의 추정값이 항상 정확하다는 뜻인가?

</div>

??? success "연습문제 3 풀이"
    아니다. 독립 정규 관측·알려진 공통 분산과 차원 조건 아래 총 제곱 오차의 반복 표본 기댓값을 비교하는 결과다. 각 좌표나 개별 표본의 오차 개선을 보장하지 않으며 베타-이항 문제에 그대로 적용할 수도 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트만 있는 풀이와 위 타율 함수를 완전한 분석으로 만들려면 어떤 검증이 필요한가?

</div>

??? success "연습문제 4 풀이"
    관측 수·형태·범위를 검증하고 잡음을 반영한 하이퍼파라미터 추정법을 정한다. 추정값의 유효성과 경계해를 확인하고, 구간을 제시한다면 하이퍼파라미터 불확실성을 어떻게 다루는지 명시한다. 임포트만으로는 추론이나 검증이 수행되지 않는다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

경험적 베이즈는 공통 하이퍼파라미터를 추정해 그룹별 사후 추론에 대입한다. 데이터 웨이트와 수축 비율을 구분하고, 관측 잡음을 잠재 집단 변동과 혼동하지 않아야 한다. 수축의 효과는 모형과 위험 기준에 달려 있으며 하이퍼파라미터 불확실성의 생략도 고려해야 한다.

**참고 문헌**

- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*
- Efron, B., & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *JASA*, 70(350), 311-319.
- Casella, G. (1985). An introduction to empirical Bayes data analysis. *The American Statistician*, 39(2), 83-87.
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 5장
