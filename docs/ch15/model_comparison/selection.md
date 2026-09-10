# 모형 증거(주변 가능도)
**모형 증거**는 **주변 가능도**라고도 하며, 매개변수를 모두 적분해 없앤 뒤 어떤 모형 아래에서 관찰한 데이터가 나올 확률이다. 이는 베이즈 모형 견줌의 주춧돌이며, 설명하려는 데이터에 견주어 지나치게 복잡한 모형에 벌을 주어 오컴의 면도날을 자연스럽게 구현한다.

---

## 1. 왜 필요한가: 왜 모형 증거인가?

### 모형 견줌 문제

데이터 $\mathcal{D}$과 겨루는 모형 $\mathcal{M}_1, \mathcal{M}_2, \ldots$이 주어졌을 때 어느 모형이 데이터를 가장 잘 설명하는지 어떻게 정하는가?

**빈도주의의 길**:

- 가능도비 검정(겹친 모형에서만)
- 교차 검증(셈이 값비싸다)
- 정보 기준(AIC, BIC) — 어림값이다

**베이즈의 길**:

- 데이터가 주어졌을 때 모형마다의 확률을 셈한다
- 모형 수준에서 베이즈 정리를 쓴다

### 매개변수 추론에서 모형 추론으로

보통의 베이즈 추론은 모형을 붙박아 두고 매개변수를 미루어 안다.

$$
p(\theta \mid \mathcal{D}, \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M})}{p(\mathcal{D} \mid \mathcal{M})}
$$

모형 견줌에는 그 분모, 곧 **모형 증거**가 필요하다.

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

이 적분은 앞확률로 무게를 주어 가능한 모든 매개변수 값에 걸쳐 가능도를 평균 낸다.

---

## 2. 정의와 풀이

### 형식적 정의

매개변수가 $\theta$인 모형 $\mathcal{M}$의 **모형 증거**(주변 가능도)는 다음과 같다.

$$
\boxed{p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta}
$$

**부품**:

- $p(\mathcal{D} \mid \theta, \mathcal{M})$: 가능도 함수
- $p(\theta \mid \mathcal{M})$: 앞확률 분포
- 적분: 매개변수 공간 전체에 걸쳐

### 여러 가지 풀이

**1. 평균 가능도**: 앞확률 아래의 기대 가능도

$$
p(\mathcal{D} \mid \mathcal{M}) = \mathbb{E}_{\theta \sim p(\theta \mid \mathcal{M})}[p(\mathcal{D} \mid \theta, \mathcal{M})]
$$

**2. 예측 확률**: 모형이 데이터를 보기 전에 그것을 얼마나 잘 맞혔는가

**3. 고르는 상수**: 베이즈 정리의 분모

$$
p(\theta \mid \mathcal{D}, \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M})}{p(\mathcal{D} \mid \mathcal{M})}
$$

**4. 앞확률 예측**: 앞확률 예측 분포가 $\mathcal{D}$에 주는 확률

### 왜 "주변" 가능도인가?

"주변"이라는 말은 매개변수 위에서 주변화(적분)한다는 뜻이다.

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D}, \theta \mid \mathcal{M}) \, d\theta = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

이는 특정 매개변수 값에서 잰 **조건부 가능도** $p(\mathcal{D} \mid \hat{\theta}, \mathcal{M})$과 대비된다.

---

## 3. 베이즈 모형 견줌

### 모형의 뒤확률

모형 위의 앞확률 $p(\mathcal{M}_k)$이 주어지면 다음과 같다.

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathcal{M}_k) \, p(\mathcal{M}_k)}{\sum_j p(\mathcal{D} \mid \mathcal{M}_j) \, p(\mathcal{M}_j)}
$$

모형 증거가 데이터가 우리의 모형 믿음을 얼마나 고치는지를 곧바로 정한다.

### 뒤확률 승산

두 모형 $\mathcal{M}_1$과 $\mathcal{M}_2$에 대해 다음과 같다.

$$
\underbrace{\frac{p(\mathcal{M}_1 \mid \mathcal{D})}{p(\mathcal{M}_2 \mid \mathcal{D})}}_{\text{Posterior odds}} = \underbrace{\frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}}_{\text{Bayes factor}} \times \underbrace{\frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}}_{\text{Prior odds}}
$$

앞선 승산이 같으면 베이즈 인자가 뒤확률 승산과 같다.

### 모형 평균 내기

모형 하나를 고르는 대신 예측을 평균 낼 수 있다.

$$
p(y^* \mid x^*, \mathcal{D}) = \sum_k p(y^* \mid x^*, \mathcal{D}, \mathcal{M}_k) \, p(\mathcal{M}_k \mid \mathcal{D})
$$

이는 모형의 아리송함을 셈에 넣으며 예측 성능을 좋게 할 때가 많다.

---

## 4. 오컴의 면도날: 저절로 주어지는 복잡도 벌

### 베이즈판 오컴의 면도날

모형 증거는 복잡함에 자연스럽게 벌을 준다. 다음 적분을 보자.

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**단순한 모형**: 앞확률이 몰려 있어 데이터가 맞으면 증거가 높다
**복잡한 모형**: 앞확률이 얇게 퍼져 있어 자유로움의 "값을 치러야" 한다

### 기하학적 직관

앞확률을 매개변수 공간을 덮어야 하는 "확률 예산"이라고 생각해 보라.

$$
\int p(\theta) \, d\theta = 1
$$

복잡한 모형은 이 예산을 더 넓은 공간에 퍼뜨리므로 자리마다 확률 질량을 적게 받는다. 그 여분의 자유로움이 데이터를 설명하는 데 꼭 필요하지 않다면 복잡한 모형은 예산을 헛되이 쓰는 셈이다.

### 보기 예

데이터에 다항식을 맞춘다고 해 보자.

| 모형 | 매개변수 | 앞확률 부피 | 흔한 가능도 | 증거 |
|-------|------------|--------------|-------------------|----------|
| 선형 | 2 | 작음 | 보통 | 높음(선형 추세라면) |
| 삼차 | 4 | 보통 | 더 높음 | 보통 |
| 10차 | 11 | 큼 | 가장 높음 | 낮음(지나친 맞춤) |

10차 다항식은 가능도가 가장 높지만 앞확률이 너무 얇게 퍼져 증거는 가장 낮다.

### 수학적 쪼갬

로그 증거는 다음과 같이 쪼갤 수 있다.

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \underbrace{\log p(\mathcal{D} \mid \hat{\theta})}_{\text{Best fit}} - \underbrace{D_{KL}(p(\theta \mid \mathcal{D}) \| p(\theta))}_{\text{Complexity penalty}}
$$

여기서 $\hat{\theta}$은 최대 뒤확률 어림값이고 $D_{KL}$은 뒤확률이 앞확률과 얼마나 다른지를 잰다.

**풀이**:

- 첫 항: 모형이 데이터에 얼마나 잘 맞을 수 있는가(잘 맞음)
- 둘째 항: 모형이 데이터에서 얼마나 많이 "배워야" 했는가(복잡도)

---

## 5. 켤레 모형의 해석적 해

### 일반 원리

켤레 모형에서는 다음 덕분에 증거를 닫힌 꼴로 얻을 수 있다.

$$
p(\mathcal{D} \mid \mathcal{M}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\theta \mid \mathcal{D})}
$$

세 분포를 모두 해석적으로 아니 그 비를 셈할 수 있다.

### 베타-베르누이 모형

**얼개**: $x_i \sim \text{Bernoulli}(\theta)$, $\theta \sim \text{Beta}(\alpha_0, \beta_0)$

**증거**:

$$
p(\mathcal{D}) = \frac{B(\alpha_0 + s, \beta_0 + f)}{B(\alpha_0, \beta_0)}
$$

여기서 $s = \sum x_i$(성공), $f = n - s$(실패)이고 $B(\cdot, \cdot)$은 베타 함수이다.

**로그 증거**:

$$
\log p(\mathcal{D}) = \log B(\alpha_n, \beta_n) - \log B(\alpha_0, \beta_0)
$$

$$
= \log\Gamma(\alpha_n) + \log\Gamma(\beta_n) - \log\Gamma(\alpha_n + \beta_n) - \log\Gamma(\alpha_0) - \log\Gamma(\beta_0) + \log\Gamma(\alpha_0 + \beta_0)
$$

### 흩어짐을 아는 가우스

**얼개**: $x_i \sim \mathcal{N}(\mu, \sigma^2)$($\sigma^2$을 앎), $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**증거**:

$$
p(\mathcal{D}) = (2\pi\sigma^2)^{-n/2} \cdot \sqrt{\frac{\sigma_0^2}{\sigma_n^2}} \cdot \exp\left(-\frac{1}{2\sigma^2}\sum_i(x_i - \bar{x})^2 - \frac{(\bar{x} - \mu_0)^2}{2(\sigma^2/n + \sigma_0^2)}\right)
$$

**로그 증거**:

$$
\log p(\mathcal{D}) = -\frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2}\log\frac{\tau_0}{\tau_n} - \frac{1}{2\sigma^2}\sum_i(x_i - \bar{x})^2 - \frac{\tau_0 n \tau}{2\tau_n}(\bar{x} - \mu_0)^2
$$

여기서 $\tau = 1/\sigma^2$, $\tau_0 = 1/\sigma_0^2$, $\tau_n = \tau_0 + n\tau$이다.

### 흩어짐을 모르는 가우스(NIG 앞확률)

**얼개**: $x_i \sim \mathcal{N}(\mu, \sigma^2)$, $(\mu, \sigma^2) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$

**로그 증거**:

$$
\log p(\mathcal{D}) = \log\Gamma(\alpha_n) - \log\Gamma(\alpha_0) + \alpha_0\log\beta_0 - \alpha_n\log\beta_n + \frac{1}{2}\log\frac{\kappa_0}{\kappa_n} - \frac{n}{2}\log(2\pi)
$$

여기서 $\kappa_n = \kappa_0 + n$, $\alpha_n = \alpha_0 + n/2$이고 $\beta_n$은 NIG 갱신 식을 따른다.

---

## 6. 어림 방법

켤레가 아닌 모형에서는 정확한 셈을 다룰 수 없을 때가 많다. 여러 어림이 있다.

### 라플라스 어림

최대 뒤확률 어림값 $\hat{\theta}$ 둘레에서 뒤확률을 가우스로 어림한다.

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|H|
$$

여기서 $d$은 매개변수의 차원이고 $H$은 $\hat{\theta}$에서 음의 로그 뒤확률의 헤세 행렬이다.

**좋은 점**: 빠르고 최적화만 하면 된다
**나쁜 점**: 뒤확률이 거의 가우스라고 놓는다

### BIC 어림

베이즈 정보 기준은 로그 증거를 어림한다.

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) - \frac{d}{2}\log n
$$

여기서 $d$은 매개변수의 개수이고 $n$은 표본 크기이다.

**끌어내기**: 단위 정보 앞확률을 놓고 라플라스 어림에서 얻는다.

### 조화 평균 어림기

뒤확률 표본 $\theta^{(1)}, \ldots, \theta^{(S)}$이 주어지면 다음과 같다.

$$
p(\mathcal{D})^{-1} = \mathbb{E}_{p(\theta \mid \mathcal{D})}\left[\frac{1}{p(\mathcal{D} \mid \theta)}\right] \approx \frac{1}{S}\sum_{s=1}^S \frac{1}{p(\mathcal{D} \mid \theta^{(s)})}
$$

**주의**: 이 어림기는 흩어짐이 무한하며 못 미덥기로 이름났다!

### 중요도 표집

제안 분포 $q(\theta)$을 고른다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta = \int \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{q(\theta)} \, q(\theta) \, d\theta
$$

$$
\approx \frac{1}{S}\sum_{s=1}^S \frac{p(\mathcal{D} \mid \theta^{(s)}) \, p(\theta^{(s)})}{q(\theta^{(s)})}, \quad \theta^{(s)} \sim q
$$

**좋은 제안 분포**: 어림한 뒤확률(이를테면 라플라스 어림에서 얻은 것)

### 담금질 중요도 표집(AIS)

앞확률에서 뒤확률까지 사이를 메우는 분포의 열을 만든다.

$$
p_t(\theta) \propto p(\theta) \, p(\mathcal{D} \mid \theta)^{\beta_t}, \quad 0 = \beta_0 < \beta_1 < \cdots < \beta_T = 1
$$

증거는 중간의 고르는 상수들의 곱으로 어림한다.

### 겹 표집

증거 적분을 앞확률 질량 위의 1차원 적분으로 바꾼다.

$$
p(\mathcal{D}) = \int_0^1 L(X) \, dX
$$

여기서 $X(\lambda) = \int_{p(\mathcal{D} \mid \theta) > \lambda} p(\theta) \, d\theta$은 가능도가 $\lambda$을 넘는 앞확률 질량이다.

**널리 쓰이는 구현**: MultiNest, dynesty

---

## 7. 잇단 데이터에서의 증거

### 온라인 증거 셈하기

잇단 관찰 $x_1, x_2, \ldots, x_n$에 대해 다음과 같다.

$$
p(x_1, \ldots, x_n \mid \mathcal{M}) = \prod_{t=1}^n p(x_t \mid x_1, \ldots, x_{t-1}, \mathcal{M})
$$

인수마다 **한 걸음 앞 예측**이다.

$$
p(x_t \mid x_{1:t-1}) = \int p(x_t \mid \theta) \, p(\theta \mid x_{1:t-1}) \, d\theta
$$

**로그 증거**:

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \sum_{t=1}^n \log p(x_t \mid x_{1:t-1})
$$

이 쪼갬은 다음에 쓸모 있다.

- 온라인 모형 견줌
- 모형이 무너지는 것 알아채기(예측 확률이 떨어질 때)
- 예측 차례 검증

### 예측 차례로 풀이하기

로그 증거는 로그 예측 점수의 합과 같다.

$$
\log p(\mathcal{D}) = \sum_{t=1}^n \text{LogScore}_t
$$

이는 증거를 예측 성능과 잇는다. 증거가 높은 모형이 평균으로 더 잘 맞혔다는 뜻이다.

---

## 8. 앞확률을 어떻게 정하느냐에 대한 민감도

### 증거에서 앞확률이 하는 몫

(데이터가 넉넉하면 앞확률 선택에 흔들리지 않을 때가 많은) 뒤확률 추론과 달리, 증거는 앞확률에 **몹시 민감하다**.

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

앞확률을 바꾸면 증거가 곧바로 바뀌며 점근적으로도 그렇다.

### 제대로 되지 않은 앞확률

**결정적인 문제**: 제대로 되지 않은 앞확률에서는 증거가 정의되지 않는다!

$\int p(\theta) \, d\theta = \infty$이면 $p(\mathcal{D} \mid \mathcal{M})$은 제멋대로인 상수 배까지만 정의된다.

**결과**: 제대로 되지 않은 앞확률로는 모형을 견줄 수 없다(베이즈 인자에 뜻이 없다).

### 흐릿하지만 제대로 된 앞확률

제대로 되었더라도 아주 퍼진 앞확률은 말썽을 일으킨다.

$$
p(\theta) = \mathcal{N}(0, 10^6) \quad \text{(variance } 10^6 \text{)}
$$

이 앞확률은 그럴듯한 매개변수 자리에 하찮은 확률만 주어 모형에 억지로 벌을 준다.

### 앞확률 민감도 분석

앞확률에 따라 증거가 어떻게 바뀌는지 늘 살펴라.

1. 그럴듯한 앞확률 여럿에서 증거를 셈한다
2. 결론이 흔들리지 않으면 자신 있게 나아간다
3. 민감하다면 결론의 범위를 알린다

### 분수 베이즈 인자

한 가지 해법은 데이터의 일부로 "학습" 뒤확률을 정한 다음 나머지에서 증거를 셈하는 것이다.

$$
\text{FBF}_{12} = \frac{p(\mathcal{D}^{\text{test}} \mid \mathcal{D}^{\text{train}}, \mathcal{M}_1)}{p(\mathcal{D}^{\text{test}} \mid \mathcal{D}^{\text{train}}, \mathcal{M}_2)}
$$

이는 앞확률 민감도를 줄이지만 데이터를 쪼개야 한다.

---

## 9. 파이썬 구현

```python
"""
모형 증거(주변 가능도): 온전한 구현

이 모듈은 여러 베이즈 모형의 모형 증거 셈하기를 주며, 켤레인 경우의 정확한
풀이와 일반적인 모형의 어림을
보여 준다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize
from typing import Tuple, List, Optional, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# 베이즈 모형의 추상 바탕 클래스
# =============================================================================

class BayesianModel(ABC):
    """증거 셈하기를 갖춘 모형의 추상 바탕 클래스."""
    
    @abstractmethod
    def log_evidence(self, data: np.ndarray) -> float:
        """로그 주변 가능도를 셈한다."""
        pass
    
    @abstractmethod
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """주어진 매개변수에서 로그 가능도를 셈한다."""
        pass
    
    @abstractmethod
    def log_prior(self, params: np.ndarray) -> float:
        """로그 앞확률 밀도를 셈한다."""
        pass
    
    def log_posterior_unnorm(self, data: np.ndarray, params: np.ndarray) -> float:
        """고르게 하지 않은 로그 뒤확률을 셈한다."""
        return self.log_likelihood(data, params) + self.log_prior(params)

# =============================================================================
# 정확한 증거를 갖는 켤레 모형
# =============================================================================

@dataclass
class BetaBernoulliModel(BayesianModel):
    """
    정확한 증거 셈하기를 갖춘 베타-베르누이 모형.
    
    모형: x_i | θ ~ 베르누이(θ), θ ~ Beta(α₀, β₀)
    
    매개변수
    ----------
    alpha0 : float
        앞확률 alpha 매개변수
    beta0 : float
        앞확률 beta 매개변수
    """
    alpha0: float = 1.0
    beta0: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """
        로그 증거를 해석적으로 셈한다.
        
        log p(D) = log B(α_n, β_n) - log B(α₀, β₀)
        """
        data = np.atleast_1d(data)
        s = data.sum()  # 성공
        f = len(data) - s  # 실패
        
        alpha_n = self.alpha0 + s
        beta_n = self.beta0 + f
        
        # log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b)
        log_B_prior = gammaln(self.alpha0) + gammaln(self.beta0) - gammaln(self.alpha0 + self.beta0)
        log_B_post = gammaln(alpha_n) + gammaln(beta_n) - gammaln(alpha_n + beta_n)
        
        return log_B_post - log_B_prior
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """베르누이 로그 가능도를 셈한다."""
        theta = params[0]
        if theta <= 0 or theta >= 1:
            return -np.inf
        s = data.sum()
        f = len(data) - s
        return s * np.log(theta) + f * np.log(1 - theta)
    
    def log_prior(self, params: np.ndarray) -> float:
        """베타 로그 앞확률을 셈한다."""
        theta = params[0]
        return stats.beta.logpdf(theta, self.alpha0, self.beta0)
    
    def sequential_log_evidence(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """
        예측 확률로 증거를 차례대로 셈한다.
        
        전체 로그 증거와 예측 로그 확률의 목록을 되돌린다.
        """
        data = np.atleast_1d(data)
        alpha, beta = self.alpha0, self.beta0
        log_probs = []
        
        for x in data:
            # 예측 확률
            if x == 1:
                p = alpha / (alpha + beta)
            else:
                p = beta / (alpha + beta)
            log_probs.append(np.log(p))
            
            # 갱신
            if x == 1:
                alpha += 1
            else:
                beta += 1
        
        return sum(log_probs), log_probs

@dataclass
class GaussianKnownVarianceModel(BayesianModel):
    """
    흩어짐을 알고 증거를 정확히 셈하는 가우스 모형.
    
    모형: x_i | μ ~ N(μ, σ²), μ ~ N(μ₀, σ₀²)
    
    매개변수
    ----------
    mu0 : float
        앞확률 평균
    sigma0_sq : float
        앞확률 흩어짐
    sigma_sq : float
        아는 자료 흩어짐
    """
    mu0: float = 0.0
    sigma0_sq: float = 1.0
    sigma_sq: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """로그 증거를 해석적으로 셈한다."""
        data = np.atleast_1d(data)
        n = len(data)
        x_bar = data.mean()
        
        tau = 1 / self.sigma_sq
        tau0 = 1 / self.sigma0_sq
        tau_n = tau0 + n * tau
        
        # 표본 평균에서의 제곱 어긋남의 합
        ss = ((data - x_bar) ** 2).sum()
        
        # 로그 증거
        log_ev = (
            -0.5 * n * np.log(2 * np.pi * self.sigma_sq)  # 가능도 고르게 하기
            + 0.5 * np.log(tau0 / tau_n)  # 앞확률/뒤확률 정밀도 비
            - 0.5 * tau * ss  # 자료의 들쭉날쭉함
            - 0.5 * tau0 * n * tau / tau_n * (x_bar - self.mu0) ** 2  # 앞확률과 자료의 어긋남
        )
        
        return log_ev
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """가우스 로그 가능도를 셈한다."""
        mu = params[0]
        return stats.norm.logpdf(data, mu, np.sqrt(self.sigma_sq)).sum()
    
    def log_prior(self, params: np.ndarray) -> float:
        """가우스 로그 앞확률을 셈한다."""
        mu = params[0]
        return stats.norm.logpdf(mu, self.mu0, np.sqrt(self.sigma0_sq))
    
    def sequential_log_evidence(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """증거를 차례대로 셈한다."""
        data = np.atleast_1d(data)
        
        mu_t = self.mu0
        tau_t = 1 / self.sigma0_sq
        tau = 1 / self.sigma_sq
        
        log_probs = []
        
        for x in data:
            # 예측 분포: N(μ_t, σ² + 1/τ_t)
            pred_var = self.sigma_sq + 1 / tau_t
            log_p = stats.norm.logpdf(x, mu_t, np.sqrt(pred_var))
            log_probs.append(log_p)
            
            # 갱신
            tau_new = tau_t + tau
            mu_t = (tau_t * mu_t + tau * x) / tau_new
            tau_t = tau_new
        
        return sum(log_probs), log_probs

@dataclass
class NormalInverseGammaModel(BayesianModel):
    """
    평균과 흩어짐을 모르는 가우스 모형.
    
    모형: x_i | μ, σ² ~ N(μ, σ²)
           (μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀)
    """
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    
    def log_evidence(self, data: np.ndarray) -> float:
        """로그 증거를 해석적으로 셈한다."""
        data = np.atleast_1d(data)
        n = len(data)
        
        if n == 0:
            return 0.0
        
        x_bar = data.mean()
        ss = ((data - x_bar) ** 2).sum()
        
        # 뒤확률 매개변수
        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + n / 2
        beta_n = (self.beta0 + 0.5 * ss + 
                  0.5 * self.kappa0 * n / kappa_n * (x_bar - self.mu0) ** 2)
        
        # 로그 증거
        log_ev = (
            gammaln(alpha_n) - gammaln(self.alpha0)
            + self.alpha0 * np.log(self.beta0) - alpha_n * np.log(beta_n)
            + 0.5 * np.log(self.kappa0 / kappa_n)
            - 0.5 * n * np.log(2 * np.pi)
        )
        
        return log_ev
    
    def log_likelihood(self, data: np.ndarray, params: np.ndarray) -> float:
        """가우스 로그 가능도를 셈한다."""
        mu, sigma_sq = params
        if sigma_sq <= 0:
            return -np.inf
        return stats.norm.logpdf(data, mu, np.sqrt(sigma_sq)).sum()
    
    def log_prior(self, params: np.ndarray) -> float:
        """NIG 로그 앞확률을 셈한다."""
        mu, sigma_sq = params
        if sigma_sq <= 0:
            return -np.inf
        
        # p(σ²) = 역감마(α₀, β₀)
        log_p_sigma = stats.invgamma.logpdf(sigma_sq, self.alpha0, scale=self.beta0)
        
        # p(μ | σ²) = N(μ₀, σ²/κ₀)
        log_p_mu = stats.norm.logpdf(mu, self.mu0, np.sqrt(sigma_sq / self.kappa0))
        
        return log_p_sigma + log_p_mu

# =============================================================================
# 증거 어림 방법
# =============================================================================

def laplace_approximation(
    model: BayesianModel,
    data: np.ndarray,
    init_params: np.ndarray,
    param_bounds: Optional[List[Tuple]] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    라플라스 어림으로 로그 증거를 셈한다.
    
    log p(D) ≈ log p(D|θ̂) + log p(θ̂) + (d/2)log(2π) - (1/2)log|H|
    
    매개변수
    ----------
    model : BayesianModel
        log_likelihood과 log_prior 메서드를 갖춘 모형
    data : array
        관측 자료
    init_params : array
        최적화의 첫 매개변수 값
    param_bounds : 튜플의 list, 있어도 되고 없어도 됨
        매개변수마다의 경계
    
    반환값
    -------
    log_evidence : float
        로그 증거의 라플라스 어림
    map_params : array
        MAP 매개변수 어림값
    hessian : array
        MAP에서 음의 로그 뒤확률의 헤세 행렬
    """
    def neg_log_posterior(params):
        return -model.log_posterior_unnorm(data, params)
    
    # MAP 어림값 찾기
    result = minimize(
        neg_log_posterior,
        init_params,
        method='L-BFGS-B',
        bounds=param_bounds
    )
    
    map_params = result.x
    
    # 헤세 행렬을 수치로 셈하기
    d = len(map_params)
    eps = 1e-5
    hessian = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            params_pp = map_params.copy()
            params_pm = map_params.copy()
            params_mp = map_params.copy()
            params_mm = map_params.copy()
            
            params_pp[i] += eps
            params_pp[j] += eps
            params_pm[i] += eps
            params_pm[j] -= eps
            params_mp[i] -= eps
            params_mp[j] += eps
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hessian[i, j] = (
                neg_log_posterior(params_pp) - neg_log_posterior(params_pm)
                - neg_log_posterior(params_mp) + neg_log_posterior(params_mm)
            ) / (4 * eps ** 2)
    
    # 라플라스 어림
    log_posterior_at_map = model.log_posterior_unnorm(data, map_params)
    sign, log_det_H = np.linalg.slogdet(hessian)
    
    if sign <= 0:
        # 헤세 행렬이 양의 정부호가 아니다. 어림이 나쁠 수 있다
        log_det_H = np.log(np.abs(np.linalg.det(hessian)) + 1e-10)
    
    log_evidence = (
        log_posterior_at_map 
        + 0.5 * d * np.log(2 * np.pi) 
        - 0.5 * log_det_H
    )
    
    return log_evidence, map_params, hessian

def importance_sampling_evidence(
    model: BayesianModel,
    data: np.ndarray,
    proposal_samples: np.ndarray,
    proposal_log_pdf: Callable[[np.ndarray], float]
) -> Tuple[float, float]:
    """
    중요도 표집으로 로그 증거를 어림한다.
    
    매개변수
    ----------
    model : BayesianModel
        log_likelihood과 log_prior 메서드를 갖춘 모형
    data : array
        관측 자료
    proposal_samples : array
        제안 분포에서 뽑은 표본, 꼴 (n_samples, d)
    proposal_log_pdf : callable
        제안의 로그 밀도를 셈하는 함수
    
    반환값
    -------
    log_evidence : float
        어림한 로그 증거
    log_evidence_std : float
        어림한 표준 오차(로그 눈금에서)
    """
    n_samples = len(proposal_samples)
    log_weights = np.zeros(n_samples)
    
    for i, params in enumerate(proposal_samples):
        log_num = model.log_posterior_unnorm(data, params)
        log_denom = proposal_log_pdf(params)
        log_weights[i] = log_num - log_denom
    
    # 수치 안정을 위한 로그-합-지수
    log_evidence = logsumexp(log_weights) - np.log(n_samples)
    
    # 흩어짐 어림하기
    weights = np.exp(log_weights - log_weights.max())
    ess = weights.sum() ** 2 / (weights ** 2).sum()  # 실효 표본 크기
    
    # 어림 표준 오차
    log_evidence_std = np.std(log_weights) / np.sqrt(ess)
    
    return log_evidence, log_evidence_std

def bic_approximation(
    log_likelihood_at_mle: float,
    n_params: int,
    n_samples: int
) -> float:
    """
    로그 증거의 BIC 어림을 셈한다.
    
    log p(D) ≈ log p(D|θ̂_MLE) - (d/2) log(n)
    
    매개변수
    ----------
    log_likelihood_at_mle : float
        MLE에서의 로그 가능도
    n_params : int
        모형 매개변수의 개수
    n_samples : int
        자료점의 개수
    
    반환값
    -------
    float
        로그 증거의 BIC 어림
    """
    return log_likelihood_at_mle - 0.5 * n_params * np.log(n_samples)

# =============================================================================
# 모형 견줌 도구
# =============================================================================

def compute_model_probabilities(
    log_evidences: List[float],
    prior_probs: Optional[List[float]] = None
) -> np.ndarray:
    """
    로그 증거에서 뒤확률 모형 확률을 셈한다.
    
    매개변수
    ----------
    log_evidences : list
        모형마다의 로그 증거
    prior_probs : list, 있어도 되고 없어도 됨
        모형마다의 앞확률(None이면 고름)
    
    반환값
    -------
    array
        뒤확률 모형 확률
    """
    log_evidences = np.array(log_evidences)
    n_models = len(log_evidences)
    
    if prior_probs is None:
        log_priors = np.zeros(n_models) - np.log(n_models)
    else:
        log_priors = np.log(prior_probs)
    
    log_posteriors = log_evidences + log_priors
    log_posteriors -= logsumexp(log_posteriors)
    
    return np.exp(log_posteriors)

def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """
    베이즈 인자 BF₁₂ = p(D|M₁) / p(D|M₂)을 셈한다.
    
    수치 안정을 위해 베이즈 인자를 로그 눈금으로 되돌린다.
    """
    return log_evidence_1 - log_evidence_2

def interpret_bayes_factor(log_bf: float) -> str:
    """
    캐스와 래프터리(1995)의 눈금으로 베이즈 인자를 풀이한다.
    
    매개변수
    ----------
    log_bf : float
        로그 베이즈 인자(자연로그)
    
    반환값
    -------
    str
        증거 세기의 풀이
    """
    bf = np.exp(log_bf)
    
    if log_bf > 0:
        model = "Model 1"
        abs_log_bf = log_bf
    else:
        model = "Model 2"
        abs_log_bf = -log_bf
    
    # 캐스와 래프터리 눈금을 위해 밑 10 로그로 바꾸기
    log10_bf = abs_log_bf / np.log(10)
    
    if log10_bf < 0.5:
        strength = "Not worth more than a bare mention"
    elif log10_bf < 1:
        strength = "Substantial"
    elif log10_bf < 2:
        strength = "Strong"
    else:
        strength = "Decisive"
    
    return f"{strength} evidence for {model} (log₁₀ BF = {log10_bf:.2f})"

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_evidence_comparison(
    models: Dict[str, BayesianModel],
    data: np.ndarray,
    prior_probs: Optional[Dict[str, float]] = None
) -> plt.Figure:
    """증거로 모형 견줌을 그려 본다."""
    
    model_names = list(models.keys())
    log_evidences = [models[name].log_evidence(data) for name in model_names]
    
    if prior_probs is None:
        priors = None
    else:
        priors = [prior_probs.get(name, 1/len(models)) for name in model_names]
    
    posteriors = compute_model_probabilities(log_evidences, priors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 로그 증거
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, log_evidences, color=colors)
    ax.set_ylabel('Log Evidence', fontsize=12)
    ax.set_title('Log Marginal Likelihood', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, log_evidences):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10)
    
    # 뒤확률
    ax = axes[1]
    bars = ax.bar(model_names, posteriors, color=colors)
    ax.set_ylabel('Posterior Probability', fontsize=12)
    ax.set_title('Model Posterior Probabilities', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, posteriors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # 가장 좋은 모형에 견준 베이즈 인자
    ax = axes[2]
    best_idx = np.argmax(log_evidences)
    log_bfs = [log_evidences[best_idx] - le for le in log_evidences]
    
    bars = ax.bar(model_names, log_bfs, color=colors)
    ax.set_ylabel('Log Bayes Factor (vs best)', fontsize=12)
    ax.set_title(f'Evidence Against (vs {model_names[best_idx]})', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(np.log(10), color='orange', linestyle='--', alpha=0.7, label='Strong (10:1)')
    ax.axhline(np.log(100), color='red', linestyle='--', alpha=0.7, label='Decisive (100:1)')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_sequential_evidence(
    models: Dict[str, BayesianModel],
    data: np.ndarray
) -> plt.Figure:
    """증거가 차례대로 어떻게 쌓이는지 그려 본다."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    model_names = list(models.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    
    # 모형마다 잇단 증거 셈하기
    sequential_log_probs = {}
    cumulative_log_evidence = {}
    
    for name, model in models.items():
        if hasattr(model, 'sequential_log_evidence'):
            total, probs = model.sequential_log_evidence(data)
            sequential_log_probs[name] = probs
            cumulative_log_evidence[name] = np.cumsum(probs)
    
    n = len(data)
    x = np.arange(1, n + 1)
    
    # 위: 누적 로그 증거
    ax = axes[0]
    for i, (name, cum_ev) in enumerate(cumulative_log_evidence.items()):
        ax.plot(x, cum_ev, label=name, color=colors[i], linewidth=2)
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('Cumulative Log Evidence', fontsize=12)
    ax.set_title('Evidence Accumulation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 아래: 잇단 로그 베이즈 인자
    ax = axes[1]
    if len(model_names) >= 2:
        name1, name2 = model_names[0], model_names[1]
        cum1 = cumulative_log_evidence[name1]
        cum2 = cumulative_log_evidence[name2]
        log_bf = cum1 - cum2
        
        ax.plot(x, log_bf, 'b-', linewidth=2)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(np.log(10), color='green', linestyle='--', alpha=0.7, label='Substantial for M1')
        ax.axhline(-np.log(10), color='red', linestyle='--', alpha=0.7, label='Substantial for M2')
        
        ax.fill_between(x, 0, log_bf, where=log_bf > 0, alpha=0.3, color='green')
        ax.fill_between(x, 0, log_bf, where=log_bf < 0, alpha=0.3, color='red')
        
        ax.set_xlabel('Number of Observations', fontsize=12)
        ax.set_ylabel(f'Log BF ({name1} vs {name2})', fontsize=12)
        ax.set_title('Sequential Bayes Factor', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_occam_razor_demo(sample_sizes: List[int] = [10, 50, 200]) -> plt.Figure:
    """다항 모형 견줌으로 오컴의 면도날을 보인다."""
    
    np.random.seed(42)
    
    # 참 모형: 이차
    def true_func(x):
        return 2 + 1.5 * x - 0.5 * x**2
    
    # 데이터를 생성한다
    x_full = np.linspace(-2, 2, 200)
    
    fig, axes = plt.subplots(len(sample_sizes), 2, figsize=(14, 4*len(sample_sizes)))
    
    for row, n in enumerate(sample_sizes):
        np.random.seed(42)
        x = np.random.uniform(-2, 2, n)
        y = true_func(x) + np.random.normal(0, 0.5, n)
        
        # 차수가 다른 다항식 맞추기
        degrees = [1, 2, 3, 5, 8]
        log_evidences = []
        
        for deg in degrees:
            # 증거 어림으로 BIC 쓰기
            X = np.vander(x, deg + 1, increasing=True)
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ coeffs
            residuals = y - y_pred
            
            # MLE 흩어짐 어림값
            sigma2_mle = (residuals ** 2).mean()
            
            # MLE에서의 로그 가능도
            log_lik = -0.5 * n * np.log(2 * np.pi * sigma2_mle) - 0.5 * n
            
            # BIC 어림
            log_ev = bic_approximation(log_lik, deg + 1, n)
            log_evidences.append(log_ev)
        
        # 왼쪽: 자료와 맞춤
        ax = axes[row, 0] if len(sample_sizes) > 1 else axes[0]
        ax.scatter(x, y, alpha=0.5, s=30, label='Data')
        
        for i, deg in enumerate([1, 2, 5]):
            X = np.vander(x_full, deg + 1, increasing=True)
            X_fit = np.vander(x, deg + 1, increasing=True)
            coeffs = np.linalg.lstsq(X_fit, y, rcond=None)[0]
            y_fit = X @ coeffs
            ax.plot(x_full, y_fit, label=f'Degree {deg}', linewidth=2)
        
        ax.plot(x_full, true_func(x_full), 'k--', linewidth=2, label='True')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Polynomial Fits (n = {n})', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 오른쪽: 증거 견줌
        ax = axes[row, 1] if len(sample_sizes) > 1 else axes[1]
        
        # 뒤확률로 고르게 하기
        probs = compute_model_probabilities(log_evidences)
        
        colors = ['red' if d != 2 else 'green' for d in degrees]
        bars = ax.bar([f'Deg {d}' for d in degrees], probs, color=colors, alpha=0.7)
        
        ax.set_ylabel('Model Probability', fontsize=12)
        ax.set_title(f'Model Evidence (n = {n})', fontsize=14)
        ax.set_ylim(0, 1.1)
        
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

# =============================================================================
# 보여 주기
# =============================================================================

def demo_conjugate_evidence():
    """켤레 모형의 정확한 증거를 보인다."""
    
    print("=" * 70)
    print("MODEL EVIDENCE: CONJUGATE MODELS")
    print("=" * 70)
    
    # 베타-베르누이
    print("\n--- Beta-Bernoulli Model ---")
    np.random.seed(42)
    true_theta = 0.7
    data = np.random.binomial(1, true_theta, 50)
    
    models = {
        'Uniform prior (α=β=1)': BetaBernoulliModel(1, 1),
        'Informative correct (α=7, β=3)': BetaBernoulliModel(7, 3),
        'Informative wrong (α=3, β=7)': BetaBernoulliModel(3, 7),
    }
    
    print(f"Data: {data.sum()} successes in {len(data)} trials (true θ = {true_theta})")
    
    for name, model in models.items():
        log_ev = model.log_evidence(data)
        _, seq_probs = model.sequential_log_evidence(data)
        print(f"\n{name}:")
        print(f"  Log evidence: {log_ev:.4f}")
        print(f"  Sequential check: {sum(seq_probs):.4f}")
    
    # 가우스 모형
    print("\n\n--- Gaussian Models ---")
    np.random.seed(123)
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, 30)
    
    print(f"Data: n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}")
    print(f"True: μ={true_mu}, σ={true_sigma}")
    
    # 흩어짐을 아는 모형
    model_known = GaussianKnownVarianceModel(mu0=0, sigma0_sq=10, sigma_sq=true_sigma**2)
    log_ev_known = model_known.log_evidence(data)
    print(f"\nKnown variance (σ²={true_sigma**2}):")
    print(f"  Log evidence: {log_ev_known:.4f}")
    
    # 흩어짐을 모르는 모형
    model_unknown = NormalInverseGammaModel(mu0=0, kappa0=0.1, alpha0=1, beta0=1)
    log_ev_unknown = model_unknown.log_evidence(data)
    print(f"\nUnknown variance (NIG prior):")
    print(f"  Log evidence: {log_ev_unknown:.4f}")

def demo_laplace_approximation():
    """증거의 라플라스 어림을 보인다."""
    
    print("\n" + "=" * 70)
    print("LAPLACE APPROXIMATION")
    print("=" * 70)
    
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 30)
    
    # NIG 모형 쓰기
    model = NormalInverseGammaModel(mu0=0, kappa0=0.1, alpha0=1, beta0=1)
    
    # 정확한 증거
    exact_log_ev = model.log_evidence(data)
    
    # 라플라스 어림
    init_params = np.array([data.mean(), data.var()])
    laplace_log_ev, map_params, hess = laplace_approximation(
        model, data, init_params,
        param_bounds=[(None, None), (0.01, None)]
    )
    
    print(f"\nExact log evidence: {exact_log_ev:.4f}")
    print(f"Laplace approximation: {laplace_log_ev:.4f}")
    print(f"Difference: {abs(exact_log_ev - laplace_log_ev):.4f}")
    print(f"\nMAP estimates: μ = {map_params[0]:.3f}, σ² = {map_params[1]:.3f}")

def demo_model_comparison():
    """증거로 모형을 견주어 보인다."""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 치우친 동전에서 자료 만들기
    true_theta = 0.65
    n = 100
    data = np.random.binomial(1, true_theta, n)
    
    print(f"Data: {data.sum()} successes in {n} trials")
    
    # 서로 다른 앞확률 견주기
    models = {
        'Fair coin (α=50, β=50)': BetaBernoulliModel(50, 50),
        'Slight bias allowed (α=5, β=5)': BetaBernoulliModel(5, 5),
        'Uniform (α=1, β=1)': BetaBernoulliModel(1, 1),
        'Biased prior (α=6, β=4)': BetaBernoulliModel(6, 4),
    }
    
    log_evidences = {}
    for name, model in models.items():
        log_ev = model.log_evidence(data)
        log_evidences[name] = log_ev
        print(f"\n{name}:")
        print(f"  Log evidence: {log_ev:.4f}")
    
    # 모형 확률 셈하기
    names = list(log_evidences.keys())
    log_evs = list(log_evidences.values())
    probs = compute_model_probabilities(log_evs)
    
    print("\n--- Posterior Model Probabilities ---")
    for name, prob in zip(names, probs):
        print(f"  {name}: {prob:.4f}")
    
    # 베이즈 인자 풀이
    best_idx = np.argmax(log_evs)
    print(f"\n--- Bayes Factors vs Best Model ({names[best_idx]}) ---")
    for i, name in enumerate(names):
        if i != best_idx:
            log_bf = log_evs[best_idx] - log_evs[i]
            interp = interpret_bayes_factor(log_bf)
            print(f"  vs {name}: {interp}")

def demo_occam_razor():
    """다항 회귀에서 오컴의 면도날을 보인다."""
    
    print("\n" + "=" * 70)
    print("OCCAM'S RAZOR DEMONSTRATION")
    print("=" * 70)
    
    fig = plot_occam_razor_demo([20, 100, 500])
    fig.savefig('occam_razor_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: occam_razor_demo.png")
    print("\nAs sample size increases, evidence concentrates on true model (degree 2)")

if __name__ == "__main__":
    demo_conjugate_evidence()
    demo_laplace_approximation()
    demo_model_comparison()
    demo_occam_razor()
```

---

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 모형 고르기을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

---

**연습문제 2.**
주된 수학적 결과를 끌어내거나 밝히고 그 뜻을 설명하라.

??? success "연습문제 2 풀이"
    핵심 결과는 앞선 정보가 베이즈 정리를 거쳐 관찰한 데이터와 어우러져 고쳐진 추론을 낳는 모습을 보여 준다. 이 결과가 뜻깊은 까닭은, 매개변수의 불확실성을 아랑곳하지 않는 점 어림 방법과 달리 불확실성을 셈에 넣으면서 데이터에서 배우는 앞뒤 맞는 틀을 주기 때문이다.

---

**연습문제 3.**
이 주제에서 베이즈 방법과 빈도주의 대안을 견주어라.

??? success "연습문제 3 풀이"
    베이즈 방법은 온전한 뒤확률 분포, 자연스러운 불확실성 재기, 앞선 앎을 아우르는 원칙 있는 길을 준다. 빈도주의 대안은 표집 분포에 기대고, 큰 표본 어림이 필요할 수 있으며, 매개변수를 붙박인 미지수로 다룬다. 표본이 작을 때는 앞확률의 벌주기 효과 덕분에 베이즈 방법이 더 나을 때가 많다.

---

**연습문제 4.**
이 개념의 간단한 보기를 파이토치나 넘파이로 파이썬에 구현하라.

??? success "연습문제 4 풀이"
    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

| 항목 | 설명 |
|--------|-------------|
| **정의** | $p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$ |
| **풀이** | 앞확률 아래의 평균 가능도이자 데이터의 예측 확률 |
| **몫** | 베이즈 정리의 고르는 상수이자 모형 견줌의 바탕 |
| **오컴의 면도날** | 쓸데없는 복잡함에 저절로 벌을 준다 |
| **쪼갬** | $\log p(\mathcal{D}) = \text{잘 맞음} - \text{복잡도}$ |

### 셈하는 방법

| 방법 | 쓸 수 있는 곳 | 정확도 | 비용 |
|--------|--------------|----------|------|
| 정확한 셈(켤레) | 지수족 | 정확함 | 낮음 |
| 라플라스 어림 | 매끄러운 뒤확률 | $n$이 크면 좋음 | 보통 |
| BIC | 큰 표본 | 점근적으로 옳음 | 낮음 |
| 중요도 표집 | 두루 쓰임 | 제안 분포에 달림 | 보통에서 높음 |
| 겹 표집 | 두루 쓰임 | 높음 | 높음 |

### 핵심 통찰

1. **매개변수에 대한 주변화**: 증거는 매개변수를 모두 적분해 없앤다
2. **오컴의 면도날**: 복잡한 모형에 저절로 벌이 주어진다
3. **앞확률 민감도**: (뒤확률과 달리) 증거는 앞확률에 크게 기댄다
4. **차례로 쪼개기**: $\log p(\mathcal{D}) = \sum_t \log p(x_t \mid x_{1:t-1})$
5. **모형 평균 내기**: 예측에 증거 무게를 쓴다
6. **제대로 되지 않은 앞확률**: 모형 견줌에 쓸 수 없다

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 베이즈 인자 | 13장: 베이즈 인자 | 증거의 비 |
| 정보 기준 | 13장: 정보 기준 | BIC가 증거를 어림한다 |
| 켤레 모형 | 13장: 분포 | 정확한 증거를 얻을 수 있다 |
| BNN 모형 고르기 | 13장: BNN | 구조 고르기 |
| 교차 검증 | 7장: 모형 고르기 | 증거의 대안 |

### 주요 참고 문헌

- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. 28장.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). 7장.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. 3.4절.
