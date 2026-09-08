# 앞확률, 가능도, 뒤확률
베이즈 추론은 근본이 되는 세 양, 곧 **앞확률**, **가능도**, **뒤확률**을 중심으로 돈다. 이들의 몫과 수학적 성질, 그리고 서로 어우러지는 모습을 이해하는 것이 베이즈 방법을 기계 학습에 쓰는 데 꼭 필요하다. 이 절은 각 부품과 그것들이 베이즈 정리로 어우러지는 모습을 빈틈없이 다룬다.

---

## 1. 베이즈의 세 기둥

### 베이즈 정리: 으뜸 식

매개변수 $\theta$과 관찰한 데이터 $\mathcal{D}$에 대해 베이즈 정리는 다음을 말한다.

$$
\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\overbrace{p(\mathcal{D} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}
$$

또는 (셈에서 더 쓸모 있을 때가 많은) 비례 꼴로 쓰면 다음과 같다.

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \cdot p(\theta)
$$

부품마다 저마다의 몫을 한다.

| 부품 | 기호 | 몫 | 답하는 물음 |
|-----------|--------|------|-------------------|
| 앞확률 | $p(\theta)$ | 데이터 이전의 믿음을 담는다 | "데이터를 보기 전 $\theta$에 대해 무엇을 믿는가?" |
| 가능도 | $p(\mathcal{D} \mid \theta)$ | 매개변수와 데이터를 잇는다 | "$\theta$이 참이라면 내 데이터는 얼마나 그럴듯한가?" |
| 뒤확률 | $p(\theta \mid \mathcal{D})$ | 데이터 이후의 고쳐진 믿음 | "데이터를 본 뒤 $\theta$에 대해 무엇을 믿어야 하는가?" |
| 증거 | $p(\mathcal{D})$ | 고르는 상수 | "가능한 모든 $\theta$에 걸쳐 내 데이터는 얼마나 그럴듯한가?" |

---

## 2. 앞확률 분포

### 정의와 풀이

**앞확률 분포** $p(\theta)$은 데이터를 관찰하기 **전** 매개변수 $\theta$에 대한 우리의 믿음을 나타낸다. 여기에는 다음이 담긴다.

- 전문가에게서 온 분야의 앎
- 앞선 실험의 결과
- 매개변수에 대한 물리적 제약
- 일부러 모른 체하기(정보 없는 앞확률)

### 앞확률의 종류

**1. 정보 있는 앞확률**

참된 앞선 앎을 담는다.

$$
\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)
$$

여기서 $\mu_0$과 $\sigma_0^2$은 자리와 아리송함에 대한 앞선 믿음을 비춘다.

*보기*: 사람의 체온(°C)이라면 정보 있는 앞확률로 $\theta \sim \mathcal{N}(37, 0.5^2)$을 쓸 수 있다.

**2. 약하게 정보 있는 앞확률**

센 가정 없이 부드럽게 벌을 준다.

$$
\theta \sim \mathcal{N}(0, 10^2)
$$

이는 다른 말은 거의 하지 않으면서 "매개변수가 아마 엄청나게 크지는 않을 것"이라고 말한다.

**3. 정보 없는(퍼진) 앞확률**

"데이터가 말하게" 하려는 것이다.

$$
p(\theta) \propto 1 \quad \text{(improper uniform)}
$$

**주의**: 제대로 되지 않은 앞확률(적분해서 1이 되지 않는 것)은 조심히 다루어야 한다. 뒤확률도 제대로 되지 않을 수 있다.

**4. 제프리스 앞확률**

다시 매개변수화해도 흔들리지 않는다. 피셔 정보가 $I(\theta)$인 매개변수 $\theta$에 대해 다음과 같다.

$$
p_J(\theta) \propto \sqrt{I(\theta)} = \sqrt{-\mathbb{E}\left[\frac{\partial^2 \log p(\mathcal{D} \mid \theta)}{\partial \theta^2}\right]}
$$

*보기*: 매개변수가 $\theta \in (0,1)$인 베르누이 가능도에서는 다음과 같다.

$$
p_J(\theta) \propto \theta^{-1/2}(1-\theta)^{-1/2} = \text{Beta}(1/2, 1/2)
$$

**5. 최대 엔트로피 앞확률**

밝힌 제약만 담고 그 밖에는 아무것도 놓지 않는다.

$$
p(\theta) = \arg\max_q H(q) \quad \text{subject to constraints}
$$

여기서 $H(q) = -\int q(\theta) \log q(\theta) \, d\theta$이다.

### 앞확률 끌어내기

전문가의 앎을 수학적 앞확률로 옮기는 일이다.

| 전문가의 말 | 수학적 앞확률 |
|------------------|-------------------|
| "아마 0과 10 사이" | $\theta \sim \text{Uniform}(0, 10)$ |
| "5쯤일 가능성이 크고 8을 넘는 일은 드묾" | 잘라낸 $\theta \sim \mathcal{N}(5, 1.5^2)$ |
| "어디든 될 수 있지만 극단적인 값은 드묾" | $\theta \sim \text{Cauchy}(0, 2.5)$ |
| "양수여야 하고 대개 작음" | $\theta \sim \text{Exponential}(\lambda)$ |
| "아마 성김(0이 많음)" | $\theta \sim \text{Laplace}(0, b)$ |

### 앞확률 예측 분포

앞확률 예측은 데이터를 하나도 보기 **전에** 앞확률이 관찰할 데이터에 대해 넌지시 이르는 바이다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) \, d\theta
$$

이는 **앞확률 예측 점검**에 쓸모 있다. $p(\mathcal{D})$이 그럴듯한 데이터 모양새에 하찮은 확률만 준다면 앞확률의 눈금이 잘못 맞았을 수 있다.

---

## 3. 가능도 함수

### 정의

**가능도 함수** $\mathcal{L}(\theta; \mathcal{D}) = p(\mathcal{D} \mid \theta)$은 매개변수 값 $\theta$이 관찰한 데이터 $\mathcal{D}$을 얼마나 잘 설명하는지 잰다.

**결정적인 구별**:

- ($\theta$을 붙박은 채) $\mathcal{D}$의 함수로 보면 이는 확률 분포이다
- ($\mathcal{D}$을 붙박은 채) $\theta$의 함수로 보면 이는 가능도이며 $\theta$ 위의 확률 분포가 **아니다**

$$
\int p(\mathcal{D} \mid \theta) \, d\mathcal{D} = 1 \quad \text{(integrates over data)}
$$

$$
\int p(\mathcal{D} \mid \theta) \, d\theta \neq 1 \quad \text{(does NOT integrate over } \theta \text{)}
$$

### 로그가능도

수치 안정성과 해석의 편의를 위해 대개 **로그 가능도**를 쓴다.

$$
\ell(\theta; \mathcal{D}) = \log p(\mathcal{D} \mid \theta)
$$

서로 독립인 관찰 $\mathcal{D} = \{x_1, \ldots, x_n\}$에 대해 다음과 같다.

$$
\ell(\theta; \mathcal{D}) = \sum_{i=1}^{n} \log p(x_i \mid \theta)
$$

### 가능도 원리

**가능도 원리**는 데이터에 담긴 $\theta$에 대한 정보가 모두 가능도 함수에 담긴다고 말한다. (상수 배를 빼고) 같은 가능도 함수를 내는 두 데이터셋은 $\theta$에 대해 똑같은 정보를 지닌다.

### 흔한 가능도 함수

**베르누이·이항**(이진 결과):

$$
p(\mathcal{D} \mid \theta) = \theta^k (1-\theta)^{n-k}
$$

여기서 $k$ = 시행 $n$번 가운데 성공 횟수이다.

**가우스**(흩어짐을 아는 이어진 측정값):

$$
p(\mathcal{D} \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

**푸아송**(세는 데이터):

$$
p(\mathcal{D} \mid \lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}
$$

**갈래·다항**(여러 부류):

$$
p(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{k=1}^{K} \theta_k^{n_k}
$$

여기서 $n_k$ = 갈래 $k$의 개수이다.

### 충분 통계량

가능도가 다음과 같이 인수분해되면 통계량 $T(\mathcal{D})$은 $\theta$에 **충분하다**.

$$
p(\mathcal{D} \mid \theta) = g(T(\mathcal{D}), \theta) \cdot h(\mathcal{D})
$$

뒤확률은 $\mathcal{D}$에 대해 오직 $T(\mathcal{D})$을 거쳐서만 달라진다.

$$
p(\theta \mid \mathcal{D}) = p(\theta \mid T(\mathcal{D}))
$$

| 분포 | 충분 통계량 |
|--------------|---------------------|
| 베르누이 | $\sum_i x_i$(성공 횟수) |
| 가우스($\sigma^2$을 알 때) | $\bar{x} = \frac{1}{n}\sum_i x_i$(표본 평균) |
| 가우스($\mu, \sigma^2$을 모를 때) | $(\bar{x}, \sum_i (x_i - \bar{x})^2)$ |
| 푸아송 | $\sum_i x_i$(전체 세기) |
| 지수 | $\sum_i x_i$(전체 기다린 시간) |

---

## 4. 뒤확률 분포

### 정의

**뒤확률 분포** $p(\theta \mid \mathcal{D})$은 데이터 $\mathcal{D}$을 관찰한 뒤 $\theta$에 대한 고쳐진 믿음을 나타낸다.

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

뒤확률은 앞선 앎과 데이터의 증거를 가장 알맞게 아우른다.

### 뒤확률 셈하기

**해석적(닫힌 꼴)**: 켤레 앞확률-가능도 쌍에서 가능하다(다음 장을 보라).

**격자 어림**: $\theta$을 낱낱으로 쪼개고 격자에서 고르지 않은 뒤확률을 셈한 뒤 고른다.

**몬테카를로 방법**: 뒤확률에서 뽑기 위해 MCMC(메트로폴리스-헤이스팅스, HMC, 깁스)을 쓴다.

**변분 추론**: 다루기 쉬운 분포족으로 뒤확률을 어림한다.

### 뒤확률 간추리기

뒤확률을 얻으면 쓸모 있는 간추림을 뽑아낸다.

**점 어림값**:

| 어림값 | 정의 | 성질 |
|----------|------------|----------|
| 뒤확률의 평균 | $\mathbb{E}[\theta \mid \mathcal{D}] = \int \theta \, p(\theta \mid \mathcal{D}) \, d\theta$ | 제곱 오차 손실을 가장 작게 한다 |
| 뒤확률의 중앙값 | $P(\theta \leq \theta^* \mid \mathcal{D}) = 0.5$인 $\theta^*$ | 절대 오차 손실을 가장 작게 한다 |
| 뒤확률의 최빈값(MAP) | $\arg\max_\theta p(\theta \mid \mathcal{D})$ | 0-1 손실을 가장 작게 한다 |

**불확실성 재기**:

- **뒤확률의 흩어짐**: $\text{Var}[\theta \mid \mathcal{D}]$
- **믿음 구간**: $P(\theta \in C \mid \mathcal{D}) = 1 - \alpha$인 영역 $C$

**믿음 구간의 종류**:

1. **양 꼬리가 같음**: $P(\theta < a \mid \mathcal{D}) = P(\theta > b \mid \mathcal{D}) = \alpha/2$
2. **최고 뒤확률 밀도(HPD)**: 확률 질량 $1-\alpha$을 담는 가장 짧은 구간

### 뒤확률 구간과 신뢰 구간

| 갈래 | 베이즈 믿음 구간 | 빈도주의 신뢰 구간 |
|--------|---------------------------|--------------------------------|
| 풀이 | "$\theta$이 구간 안에 있을 확률이 95%" | "그런 구간의 95%가 참 $\theta$을 담는다" |
| 붙박인 양 | 구간 | 매개변수 $\theta$ |
| 확률적인 양 | 매개변수 $\theta$ | 구간의 끝 |
| 앞확률에 기댐 | 예 | 아니오 |
| 표본 하나에서의 타당성 | 예 | 덮음 보장으로서만 |

---

## 5. 증거(주변 가능도)

### 정의

**증거** 또는 **주변 가능도** $p(\mathcal{D})$은 고르는 상수이다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

이는 앞확률로 무게를 주어 가능한 모든 매개변수 값에 걸쳐 평균 낸 데이터의 확률을 나타낸다.

### 베이즈 추론에서의 몫

**매개변수 추론에서**: 흔히 무시한다(고르지 않은 뒤확률로 다룬다).

**모형 견줌에서**: 결정적이다! 증거는 모형(앞확률 + 가능도)이 관찰한 데이터를 얼마나 잘 맞히는지 잰다.

### 증거 셈하기

증거 적분은 흔히 다룰 수 없다. 어림 방법으로는 다음이 있다.

| 방법 | 설명 | 쓰임새 |
|--------|-------------|----------|
| 해석적 | 켤레 모형의 닫힌 꼴 | 단순한 모형 |
| 조화 평균 | $\hat{p}(\mathcal{D}) = \left[\frac{1}{S}\sum_{s=1}^S \frac{1}{p(\mathcal{D} \mid \theta^{(s)})}\right]^{-1}$ | **피하라!** 흩어짐이 크다 |
| 중요도 표집 | 제안 분포에서 뽑은 무게 준 표본 | 차원이 그리 높지 않을 때 |
| 겹 표집 | 차례로 눌러 담기 | 모형 견줌 |
| 열역학 적분 | 앞확률에서 뒤확률로 가는 길 | 빈틈없지만 값비싸다 |
| 변분 한계(ELBO) | $\log p(\mathcal{D}) \geq \text{ELBO}$ | 어림이지만 규모를 키울 수 있다 |

---

## 6. 고쳐 가는 얼개

### 앞확률에서 뒤확률로의 흐름

믿음이 어떻게 바뀌는지 그려 본다.

```
Prior p(θ)          ×    Likelihood p(D|θ)     ∝    Posterior p(θ|D)
     │                         │                         │
     ▼                         ▼                         ▼
[Broad/uncertain]    ×    [Peaked at MLE]     =    [Refined beliefs]
```

### 정밀도로 무게 준 평균 내기(가우스인 경우)

가우스 앞확률 $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$과 흩어짐 $\sigma^2$을 아는 가우스 가능도에 대해 다음과 같다.

$$
p(\theta \mid \mathcal{D}) = \mathcal{N}(\mu_n, \sigma_n^2)
$$

여기서 각 기호는 다음과 같다.

$$
\mu_n = \frac{\frac{1}{\sigma_0^2}\mu_0 + \frac{n}{\sigma^2}\bar{x}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{\tau_0 \mu_0 + \tau_{\text{data}} \bar{x}}{\tau_0 + \tau_{\text{data}}}
$$

$$
\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} = \frac{1}{\tau_0 + \tau_{\text{data}}}
$$

여기서 $\tau = 1/\sigma^2$은 **정밀도**(흩어짐의 역수)이다.

**핵심 통찰**: 뒤확률의 평균은 앞확률의 평균과 데이터 평균을 **정밀도로 무게 준 평균**이다. 더 정밀한(흩어짐이 작은) 쪽이 더 큰 무게를 받는다.

### 점근 거동

**베른슈타인-폰 미제스 정리**: 규칙성 조건 아래 $n \to \infty$이면 다음과 같다.

$$
p(\theta \mid \mathcal{D}_n) \xrightarrow{d} \mathcal{N}\left(\hat{\theta}_{\text{MLE}}, \frac{1}{n I(\theta_0)}\right)
$$

여기서 $I(\theta_0)$은 참 매개변수에서의 피셔 정보이다.

**뜻하는 바**:

1. 뒤확률이 참 매개변수 둘레에 몰린다
2. 데이터가 넉넉하면 앞확률이 "씻겨 나간다"
3. 베이즈 추론과 빈도주의 추론이 점근적으로 맞아떨어진다
4. 뒤확률의 흩어짐이 $O(1/n)$으로 줄어든다

---

## 7. 파이썬 구현

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

---

## 8. 개요

이 모듈은 베이즈 추론을 이어진 매개변수 공간으로 넓힌다. 거기서는 낱낱의 확률 대신 확률밀도함수를 다룬다. 동전 던지기를 위한 베타-이항 켤레 모형을 세우고, 앞확률을 달리했을 때의 효과를 살피며, 이어진 데이터의 추론을 위한 정규-정규 모형을 들여온다.

---

## 9. 이어진 매개변수에 대한 베이즈 정리

### 1.1 이어진 값에서의 정식화

매개변수 $\theta$이 이어진 공간의 값을 가지면 베이즈 정리는 다음이 된다.

$$
\boxed{p(\theta|D) = \frac{p(D|\theta) \, p(\theta)}{p(D)}}
$$

여기서 각 기호는 다음과 같다.

| 항 | 이름 | 설명 |
|------|------|-------------|
| $p(\theta\|D)$ | 뒤확률 밀도 | 데이터가 주어졌을 때 $\theta$의 확률 밀도 |
| $p(D\|\theta)$ | 가능도 함수 | 매개변수 값이 주어졌을 때 데이터의 확률 |
| $p(\theta)$ | 앞확률 밀도 | 매개변수에 대한 처음 믿음 |
| $p(D)$ | 증거 | 주변 가능도(고르는 상수) |

### 1.2 증거 적분

증거는 매개변수 공간 위에서 적분하여 셈한다.

$$
p(D) = \int p(D|\theta) \, p(\theta) \, d\theta
$$

실전에서는 흔히 **비례 꼴**로 다룬다.

$$
p(\theta|D) \propto p(D|\theta) \, p(\theta)
$$

그런 다음 $\int p(\theta|D) \, d\theta = 1$이 되도록 고른다.

### 1.3 격자 어림

수치로 셈하려면 이어진 매개변수 공간을 낱낱으로 쪼갤 수 있다.

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

---

## 10. 베타-이항 모형

### 2.1 문제의 얼개

던진 결과를 바탕으로 동전이 앞면일 확률 $\theta$을 어림하려 한다. 이어진 매개변수를 다루는 베이즈 추론의 본보기 보기이다.

### 2.2 베타 앞확률

**베타 분포**는 확률 매개변수 $\theta \in [0, 1]$의 자연스러운 앞확률이다.

$$
p(\theta) = \text{Beta}(\theta | \alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}
$$

여기서 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$은 베타 함수이다.

**Beta$(\alpha, \beta)$의 성질:**

| 성질 | 식 |
|----------|---------|
| 평균 | $\displaystyle\frac{\alpha}{\alpha + \beta}$ |
| 최빈값 | $\displaystyle\frac{\alpha - 1}{\alpha + \beta - 2}$($\alpha, \beta > 1$일 때) |
| 흩어짐 | $\displaystyle\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

**초매개변수의 풀이:**

- $\alpha = \beta = 1$: **고른 앞확률**(모든 값이 똑같이 그럴듯하다)
- $\alpha = \beta = 0.5$: **제프리스 앞확률**(정보가 없고 다시 매개변수화해도 흔들리지 않는다)
- $\alpha = \beta > 1$: 확률이 $\theta = 0.5$ 가까이에 몰린다
- $\alpha > \beta$: 앞면이 더 그럴듯하다는 앞선 믿음
- $\alpha + \beta$: **몰림 정도** — 값이 클수록 앞선 확신이 세다

### 2.3 이항 가능도

동전을 $n$번 던져 앞면이 $k$번 나왔다면 가능도는 다음과 같다.

$$
p(k | n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

(상수를 빼고) $\theta$의 함수로 보면 다음과 같다.

$$
\mathcal{L}(\theta) \propto \theta^k (1-\theta)^{n-k}
$$

### 2.4 켤레 뒤확률

베타 앞확률은 이항 가능도의 **켤레**이므로 뒤확률도 베타 분포이다.

$$
\boxed{p(\theta | k, n) = \text{Beta}(\theta | \alpha + k, \beta + n - k)}
$$

**켤레 갱신 규칙:**

$$
\alpha_{\text{post}} = \alpha_{\text{prior}} + k \quad \text{(add number of heads)}
$$

$$
\beta_{\text{post}} = \beta_{\text{prior}} + (n - k) \quad \text{(add number of tails)}
$$

초매개변수는 **가짜 세기**로 풀이할 수 있다. 곧 앞확률의 앞면 $\alpha - 1$번과 뒷면 $\beta - 1$번이다.

### 2.5 구현

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

### 2.6 보기: 고른 앞확률에 앞면 15번, 뒷면 5번

Beta$(1, 1)$(고른 앞확률)에서 시작하면 다음과 같다.

$$
\text{Posterior} = \text{Beta}(1 + 15, 1 + 5) = \text{Beta}(16, 6)
$$

| 어림값 | 값 |
|----------|-------|
| 최대 가능도 | \$15/20 = 0.750$ |
| 뒤확률의 평균 | $16/22 \approx 0.727$ |
| 뒤확률의 최빈값 | \$15/20 = 0.750$ |

고른 앞확률 때문에 뒤확률의 평균은 최대 가능도 어림값에 견주어 **0.5 쪽으로 오그라든다**.

---

## 11. 앞확률을 달리했을 때의 효과

### 3.1 앞확률 민감도 분석

표본이 작으면 어떤 앞확률을 고르느냐가 뒤확률을 크게 좌우한다. 10번 던져 앞면 7번을 보았다고 하자.

| 앞확률 | $(\alpha, \beta)$ | 뒤확률의 평균 | 최대 가능도와의 차이 |
|-------|-------------------|----------------|---------------|
| 고른 분포 | $(1, 1)$ | 0.667 | 0.033 |
| 제프리스 | $(0.5, 0.5)$ | 0.682 | 0.018 |
| 약한 공정 | $(2, 2)$ | 0.643 | 0.057 |
| 강한 공정 | $(10, 10)$ | 0.567 | 0.133 |
| 미더워하지 않음 | $(2, 8)$ | 0.450 | 0.250 |

최대 가능도 어림값은 어느 경우에나 \$7/10 = 0.70$이다.

### 3.2 앞확률과 데이터의 부딪침

앞선 믿음이 관찰한 데이터와 크게 부딪칠 때는 다음과 같다.

- **작은 표본**: 앞확률이 우세하여 뒤확률이 앞선 믿음을 비춘다
- **큰 표본**: 가능도가 우세하여 앞확률과 상관없이 뒤확률이 하나로 모인다

이는 **베른슈타인-폰 미제스 정리**로 정식화된다. (규칙성 조건 아래) $n \to \infty$이면 앞확률과 상관없이 뒤확률이 참 매개변수 값 둘레에 몰린다.

### 3.3 구현

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

---

## 12. 차례 갱신

### 4.1 온라인 베이즈 학습

베이즈 추론의 핵심 성질 하나는 **차례 갱신이 한꺼번에 갱신하는 것과 같다**는 것이다. 관찰을 하나씩 다루어도 한꺼번에 다룰 때와 같은 뒤확률이 나온다.

베타-이항에서는 다음과 같다.

$$
\text{Beta}(\alpha, \beta) \xrightarrow{\text{observe H}} \text{Beta}(\alpha + 1, \beta) \xrightarrow{\text{observe T}} \text{Beta}(\alpha + 1, \beta + 1)
$$

이는 다음과 같다.

$$
\text{Beta}(\alpha, \beta) \xrightarrow{\text{observe 1H, 1T}} \text{Beta}(\alpha + 1, \beta + 1)
$$

### 4.2 구현

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

### 4.3 모임 거동

데이터가 쌓이면 다음과 같다.

1. **뒤확률이 몰린다**: 흩어짐이 $O(1/n)$으로 줄어든다
2. **앞확률이 씻겨 나간다**: 뒤확률의 평균이 참 매개변수로 모인다
3. **불확실성이 수로 담긴다**: 데이터가 늘수록 믿음 구간이 좁아진다

---

## 13. 정규-정규 모형

### 5.1 문제의 얼개

관찰 $x_1, \ldots, x_n$ $n$개가 주어졌을 때 **흩어짐 $\sigma^2$을 아는** 정규 분포의 평균 $\mu$을 어림하려 한다.

### 5.2 켤레 짜임

**앞확률:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**가능도:** $x_i | \mu \sim \mathcal{N}(\mu, \sigma^2)$이며 서로 독립이다

**뒤확률:** $\mu | x_1, \ldots, x_n \sim \mathcal{N}(\mu_n, \sigma_n^2)$

### 5.3 뒤확률의 매개변수

뒤확률의 매개변수는 우아한 닫힌 꼴로 나타난다.

$$
\boxed{\mu_n = \frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2}}
$$

$$
\boxed{\sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}}
$$

여기서 $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$은 표본 평균이다.

### 5.4 정밀도로 무게 준 꼴

**정밀도**를 흩어짐의 역수($\tau = 1/\sigma^2$)로 정의하면 다음과 같다.

$$
\tau_n = \tau_0 + n\tau_{\text{data}}
$$

$$
\mu_n = \frac{\tau_0 \mu_0 + n\tau_{\text{data}} \bar{x}}{\tau_n}
$$

뒤확률의 평균은 앞확률의 평균과 표본 평균을 **정밀도로 무게 준 평균**이다.

### 5.5 구현

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

### 5.6 보기

다음이 주어졌다고 하자.

- 앞확률: $\mathcal{N}(0, 5^2)$
- 아는 데이터 표준편차: $\sigma = 2$
- 관찰: $\bar{x} = 4.8$인 표본 $n = 20$개

무게를 셈하면 다음과 같다.

- 앞확률의 정밀도: $\tau_0 = 1/25 = 0.04$
- 데이터의 정밀도: $n\tau = 20/4 = 5$
- 전체 정밀도: $\tau_n = 5.04$

뒤확률:

- 평균: $\mu_n = \frac{0.04 \times 0 + 5 \times 4.8}{5.04} \approx 4.76$
- 표준편차: $\sigma_n = \sqrt{1/5.04} \approx 0.445$

관찰이 20개이면 데이터가 앞확률을 압도한다(무게 $\approx 99.2\%$).

---

## 연습문제

### 연습문제 1: 센 앞확률과 약한 앞확률
10번 던져 앞면 5번을 보았을 때 Beta$(1,1)$ 앞확률과 Beta$(100,100)$ 앞확률을 견주어라. 센 앞확률이 뒤확률에 얼마나 영향을 주는지 수로 나타내라.

### 연습문제 2: 앞확률 민감도
1모듈의 의료 검사 보기를 베타 앞확률로 이어진 검사 정확도 매개변수를 두어 다시 세워라. 추론이 어떻게 달라지는가?

### 연습문제 3: 모임
공정한 동전으로 동전 던지기를 길게 만들어라. 앞확률을 달리하며 뒤확률의 평균과 95% 믿음 구간이 참 매개변수로 어떻게 모이는지 그려라.

### 연습문제 4: 정규-역감마
평균과 흩어짐을 모두 모르는 경우의 정규-역감마 켤레족을 살펴보라. 이 모형의 추론을 구현하라.

### 연습문제 5: 실제 데이터에 쓰기
이진 결과를 담은 데이터셋(이를테면 고객 전환, 전자우편 클릭률)을 찾아라. 베타-이항 모형으로 성공률을 어림하고 95% 믿음 구간을 셈하라.

---

**연습문제 1.**
베이즈 정리를 밝히고 부품마다 설명하라.

??? success "연습문제 1 풀이"
    $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$. 앞확률 $p(\theta)$: 데이터 이전의 믿음. 가능도 $p(D|\theta)$: 매개변수가 주어졌을 때 데이터의 확률. 뒤확률 $p(\theta|D)$: 데이터 이후의 고쳐진 믿음. 증거 $p(D) = \int p(D|\theta)p(\theta)d\theta$: 고르는 상수.

---

**연습문제 2.**
베타-이항 모형의 뒤확률을 셈하라.

??? success "연습문제 2 풀이"
    앞확률: $\theta \sim \text{Beta}(\alpha, \beta)$. 데이터: 시행 $n$번 가운데 성공 $k$번. 뒤확률: $\theta|k \sim \text{Beta}(\alpha+k, \beta+n-k)$. 베타 앞확률은 이항 가능도의 켤레여서 닫힌 꼴의 뒤확률을 준다.

---

**연습문제 3.**
확률에 대한 베이즈의 풀이와 빈도주의의 풀이가 어떻게 다른지 설명하라.

??? success "연습문제 3 풀이"
    빈도주의: 확률은 사건이 오랜 기간에 걸쳐 나타나는 빈도이고 매개변수는 붙박인 미지수이다. 베이즈: 확률은 믿음의 정도를 나타내고 매개변수는 아리송함을 비추는 분포를 갖는다. 핵심 결과: 베이즈주의자는 매개변수에 대해 확률 진술을 할 수 있지만(믿음 구간) 빈도주의자는 할 수 없다(신뢰 구간은 풀이가 다르다).

---

**연습문제 4.**
$n \to \infty$이면 뒤확률이 왜 최대 가능도 어림값 둘레에 몰리는가?

??? success "연습문제 4 풀이"
    베른슈타인-폰 미제스 정리에 따르면 뒤확률은 최대 가능도 어림값을 중심으로 흩어짐이 $1/(nI(\theta))$인 정규 분포로 모이며 $I$은 피셔 정보이다. $n$이 커질수록 가능도가 앞확률을 압도하고 뒤확률이 몰린다. 이는 베이즈 어림값과 빈도주의 어림값이 점근적으로 맞아떨어짐을 보여 준다.

## 정리하며

| 부품 | 정의 | 몫 |
|-----------|------------|------|
| **앞확률** $p(\theta)$ | 데이터 이전의 믿음 | 분야의 앎을 담는다 |
| **가능도** $p(\mathcal{D} \mid \theta)$ | 매개변수가 주어졌을 때 데이터의 확률 | 모형과 관찰을 잇는다 |
| **뒤확률** $p(\theta \mid \mathcal{D})$ | 데이터 이후의 믿음 | 앞확률과 데이터를 가장 알맞게 아우름 |
| **증거** $p(\mathcal{D})$ | 주변 데이터 확률 | 고르기, 모형 견줌 |

### 핵심 통찰

1. **뒤확률 ∝ 가능도 × 앞확률**: 근본이 되는 갱신 식
2. **충분 통계량**: 정보를 잃지 않고 데이터를 간추리기
3. **정밀도로 무게 준 평균 내기**: 더 정밀한 쪽이 더 큰 무게를 받는다
4. **점근적 우세**: 데이터가 넉넉하면 (그럴듯한) 앞확률과 상관없이 뒤확률이 참으로 모인다
5. **믿음 구간**: 매개변수에 대한 곧은 확률 진술

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 믿음으로서의 확률 | 13장: 믿음으로서의 확률 | 철학적 바탕 |
| 켤레 앞확률 | 13장: 켤레 앞확률 | 해석으로 다루기 쉬움 |
| 베타-이항 | 13장: 베르누이-베타 | 자세한 켤레 보기 |
| 가우스 추론 | 13장: 가우스 모형 | 이어진 매개변수 어림 |
| 증거 셈하기 | 13장: 모형 증거 | 주변 가능도 방법 |
| MCMC | 13장: MCMC | 뒤확률 표집 |
| 변분 추론 | 13장: VI | 어림 뒤확률 |

### 주요 참고 문헌

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. 1~2장.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. 3장, 5장.
- Robert, C. P. (2007). *The Bayesian Choice* (2nd ed.). Springer.

---

# 덧붙임: 이어진 값에서의 베이즈 추론

뒤이은 절은 앞의 낱낱의 보기를 이어진 매개변수 공간으로 넓혀, 베타-이항 모형과 정규-정규 모형을 온전한 뒤확률 분석과 함께 세운다.

---

# 이어진 값에서의 베이즈 추론

1. **이어진 매개변수**에는 낱낱의 확률이 아니라 확률 밀도가 필요하다. 뒤확률은 매개변수 공간 위의 밀도 함수이다.

2. **켤레 앞확률**은 같은 분포족의 뒤확률을 내어 닫힌 꼴로 고칠 수 있게 한다.
   - 베타-이항: $\text{Beta}(\alpha, \beta) \to \text{Beta}(\alpha + k, \beta + n - k)$
   - 정규-정규: $\mathcal{N}(\mu_0, \sigma_0^2) \to \mathcal{N}(\mu_n, \sigma_n^2)$

3. 표본이 작을 때는 **앞확률 민감도**가 중요하다. 센 앞확률이 뒤확률을 압도할 수 있다. 표본이 크면 가능도가 압도한다.

4. **차례 갱신**은 한꺼번에 갱신하는 것과 같다. 온라인 학습을 가능하게 하는 근본 성질이다.

5. **정밀도로 무게 주기**가 직관을 준다. 뒤확률의 평균은 무게가 정밀도(흩어짐의 역수)에 비례하는 무게 준 평균이다.

---

**참고 문헌**

- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), 2~3장
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, 3장
- Hoff, P. *A First Course in Bayesian Statistical Methods*
