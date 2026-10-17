# 국면 탐지와 베이즈 전략 평가

시장 상태를 확률적으로 추정하는 필터링과 전략 수익률을 비교하는 사후 추론을 살펴본다. 두 예제 모두 관측·의존성 가정을 명시해야 하며, 사후확률을 계산했다는 사실만으로 실시간 거래 성과나 반복 검정의 오류율이 보장되지는 않는다.

## 1. 베이즈 국면 필터링

낮은 변동성과 높은 변동성처럼 서로 다른 관측 분포를 갖는 두 잠재 상태를 가정한다.

$$r_t\mid z_t=k\sim N(\mu_k,\sigma_k^2),\qquad P(z_t=j\mid z_{t-1}=i)=A_{ij}$$

전이행렬의 각 행은 확률분포다. 고정된 상태 파라메터 아래 현재까지 관측한 자료로 필터 확률을 갱신한다.

$$P(z_t=k\mid r_{1:t})\propto p(r_t\mid z_t=k)\sum_j A_{jk}P(z_{t-1}=j\mid r_{1:t-1})$$

예측 단계에서 이전 상태 확률에 전이행렬을 적용하고, 새 관측의 우도를 곱해 정규화한다. 미래 자료까지 사용하는 스무딩과 구분한다. [은닉 마르코프 모델](../../ch19/markov_chains/hmm.md)을 참고한다.

### 구현의 범위

아래 코드는 μ·σ·A를 주어진 값으로 사용하는 두 상태 필터다. 이 파라메터의 사후분포를 추론하지 않는다. 초기 상태 확률은 [0.5,0.5]로 고정되며 run_filter를 다시 호출하면 이전 상태에서 이어진다. σ>0, 유효한 전이행렬과 장치·자료형 일치를 확인한다. 작은 우도로 정규화 합이 0이 될 수 있어 안정적인 로그 계산이 필요할 수 있다. 상태는 모형의 잠재 범주이므로 곧바로 상승장·하락장의 객관적 정답으로 간주하지 않는다.

```python
import torch

class BayesianRegimeDetector:
    """
    시장 자료를 위한 온라인 베이즈 국면 찾기.
    
    가우스 방출과 마르코프 옮김을 갖는 두 국면 모형.
    """
    
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, 
                 transition_matrix: torch.Tensor):
        """
        파라메터
        ----------
        mu : (2,) 국면 평균
        sigma : (2,) 국면 표준편차
        transition_matrix : (2, 2) 옮김 확률
        """
        self.mu = mu
        self.sigma = sigma
        self.A = transition_matrix
        self.filtered_prob = torch.tensor([0.5, 0.5])  # 첫 믿음
    
    def update(self, r_t: float) -> torch.Tensor:
        """새 수익률 관측이 주어졌을 때 국면 확률을 새로 고친다."""
        # 예측 스텝
        predicted = self.A.T @ self.filtered_prob
        
        # 국면마다의 가능도
        likelihood = torch.exp(
            -0.5 * ((r_t - self.mu) / self.sigma) ** 2
        ) / self.sigma
        
        # 새로 고치기 스텝
        unnormalized = likelihood * predicted
        self.filtered_prob = unnormalized / unnormalized.sum()
        
        return self.filtered_prob.clone()
    
    def run_filter(self, returns: torch.Tensor) -> torch.Tensor:
        """수익률 계열 전체에 거르개를 돌린다."""
        T = len(returns)
        probs = torch.zeros(T, 2)
        
        for t in range(T):
            probs[t] = self.update(returns[t].item())
        
        return probs
```

---

## 2. 전략 비교를 위한 베이즈 추론

서로 다른 전략의 평균수익률 μ_A·μ_B를 비교하자. 각 전략에서 관측이 조건부 독립 정규분포를 따르고, 두 전략의 파라메터 사전분포도 독립이라면 별도의 정규-역감마 사후분포를 구할 수 있다.

$$P(\mu_B>\mu_A\mid D)\approx\frac1S\sum_{s=1}^S\mathbf1[\mu_B^{(s)}>\mu_A^{(s)}]$$

같은 날짜·같은 시장에서 얻은 전략 수익률은 보통 상관될 수 있다. 이 경우 독립 사후표본을 결합하는 대신 짝지은 수익률 차이나 결합 모형으로 의존성을 반영해야 한다. 과거 백테스트의 비교는 무작위 배정한 A/B 실험과 같은 설계가 아니다. 평균수익률 우위와 샤프 비율 우위도 서로 다른 관심량이다.

### 구현의 범위

코드는 각 전략의 평균·분산을 별도로 추론하고 독립 사후표본으로 차이를 계산한다. prior_var의 역수는 κ₀로 사용되므로 prior_var 자체가 평균 μ의 주변 사전분산은 아니다. 조건부로 Var(μ|σ²)=σ²/κ₀다. 빈 배열은 처리하지 않으며 유효한 양의 사전 파라메터가 필요하다.

prob_practically_better의 기준 0.001은 입력 수익률의 한 기간에 대한 차이다. 일별·월별 자료에 동일한 경제적 의미를 갖지 않으므로 거래비용과 비교 기간에 맞춰 정의한다. 사후표본 개수는 수치 오차를 줄이지만 관측 자료의 정보량을 늘리지 않는다.

```python
import torch
from scipy import stats
import numpy as np

class BayesianABTest:
    """
    전략 견줌을 위한 베이즈 A/B 시험.
    
    켤레 정규-역감마 모형과 뒤확률 표집을 써서
    우월할 확률을 셈한다.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_var: float = 100.0,
                 prior_shape: float = 1.0, prior_scale: float = 1.0):
        self.mu_0 = prior_mean
        self.kappa_0 = 1.0 / prior_var  # 평균에 대한 앞확률 정밀도
        self.alpha_0 = prior_shape
        self.beta_0 = prior_scale
    
    def posterior_params(self, data: np.ndarray) -> dict:
        """정규-역감마 뒤확률 파라메터를 셈한다."""
        n = len(data)
        x_bar = data.mean()
        s2 = data.var(ddof=1) if n > 1 else 1.0
        
        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / kappa_n
        alpha_n = self.alpha_0 + n / 2.0
        beta_n = (self.beta_0 + 0.5 * (n - 1) * s2 + 
                  0.5 * self.kappa_0 * n * (x_bar - self.mu_0)**2 / kappa_n)
        
        return {
            'mu_n': mu_n, 'kappa_n': kappa_n,
            'alpha_n': alpha_n, 'beta_n': beta_n
        }
    
    def sample_posterior_mean(self, params: dict, n_samples: int = 10000):
        """뒤확률 평균 수익률을 표집한다."""
        # sigma^2 ~ 역감마(alpha_n, beta_n)
        sigma2_samples = stats.invgamma(
            a=params['alpha_n'], scale=params['beta_n']
        ).rvs(n_samples)
        
        # mu | sigma^2 ~ 정규(mu_n, sigma^2 / kappa_n)
        mu_samples = stats.norm(
            loc=params['mu_n'],
            scale=np.sqrt(sigma2_samples / params['kappa_n'])
        ).rvs()
        
        return mu_samples
    
    def compare(self, returns_a: np.ndarray, returns_b: np.ndarray,
                n_samples: int = 50000) -> dict:
        """
        전략 둘을 견준다.
        
        반환값
        -------
        다음을 담은 dict:
            prob_b_better: P(mu_B > mu_A | 자료)
            expected_difference: E[mu_B - mu_A | 자료]
            credible_interval: mu_B - mu_A의 95% 믿음 구간
        """
        params_a = self.posterior_params(returns_a)
        params_b = self.posterior_params(returns_b)
        
        samples_a = self.sample_posterior_mean(params_a, n_samples)
        samples_b = self.sample_posterior_mean(params_b, n_samples)
        
        diff = samples_b - samples_a
        
        return {
            'prob_b_better': float((diff > 0).mean()),
            'expected_difference': float(diff.mean()),
            'credible_interval': (
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5))
            ),
            'prob_practically_better': float(
                (diff > 0.001).mean()  # 달마다 10bp 넘음
            )
        }
```

### 사후확률과 반복 검정

베이즈 추론은 지정한 모형·사전분포 아래 평균 차이의 사후확률과 크기를 요약한다. 빈도주의 검정은 귀무가설 아래 통계량의 표본분포와 오류율을 다룬다. 빈도주의에도 순차 검정과 사전 정보 활용 방법이 있으며 고정 표본 t검정만이 대안은 아니다.

적절한 사전분포와 무시 가능한 중단 규칙에서는 같은 관측 자료의 사후분포가 중단 계획에 의존하지 않을 수 있다. 그러나 사후확률이 기준을 넘을 때마다 멈추는 전략이 고정된 참값에서의 거짓 양성률까지 자동 통제하는 것은 아니다. [중단 규칙의 영향에 대한 연구](https://arxiv.org/abs/1807.09077)는 우도 원리·사후 보정·빈도주의 오류 보장을 구분한다. 여러 전략을 탐색한 선택 과정과 시간 의존성도 반영해야 한다.

---

## 3. 시장 신호의 온라인 갱신

아래 추적기는 관측 전 상태 분산을 var/decay로 증가시킨 뒤 정규 관측의 칼만 갱신을 적용한다. 0<decay≤1, 양의 초기·관측 분산을 가정한다. decay=1이면 일정한 평균에 대한 갱신이며, decay<1이면 이전 정보를 할인하는 동적 근사다. 고정된 파라메터에 대한 정확한 순차 베이즈 갱신과 같은 해석을 자동 적용하지 않는다.

칼만 이득은 관측과 예측의 상대적 정밀도를 반영한다. decay나 관측분산을 자료에 맞춰 선택한 경우 그 추정 불확실성은 아래 두 상태 값에 포함되지 않는다. 시장 변화에 빠르게 반응하는 것과 잡음을 추적하는 것 사이의 균형을 검증한다.

```python
class BayesianSignalTracker:
    """차례대로 베이즈 새로 고치기로 시간에 따라 바뀌는 신호를 좇는다."""
    
    def __init__(self, prior_mean=0.0, prior_var=1.0, obs_var=1.0, 
                 decay=0.99):
        self.mu = prior_mean
        self.var = prior_var
        self.obs_var = obs_var
        self.decay = decay
    
    def update(self, observation: float) -> tuple:
        """
        새 관측으로 신호 어림값을 새로 고친다.
        
        시간에 따라 바뀌는 신호를 담으려고 지수 사그라듦을 쓴다.
        """
        # 앞확률 흩어짐 부풀리기(잊음 인자)
        predicted_var = self.var / self.decay
        
        # 칼만식 새로 고치기
        kalman_gain = predicted_var / (predicted_var + self.obs_var)
        self.mu = self.mu + kalman_gain * (observation - self.mu)
        self.var = (1 - kalman_gain) * predicted_var
        
        return self.mu, self.var
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
필터링·스무딩과 상태 파라메터 추론을 구분하라.

</div>

??? success "풀이"
    필터링은 현재까지 자료로 현재 상태를 추정하고 스무딩은 미래 자료도 사용한다. 아래 필터는 주어진 μ·σ·A를 사용하므로 이 파라메터들의 사후 불확실성을 추론하는 완전한 계층 분석과 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 날짜의 두 전략을 독립 사후분포로 비교하면 어떤 문제가 생길 수 있는가?

</div>

??? success "풀이"
    전략 수익률의 공분산을 생략한다. 차이의 분산은 두 분산의 합에서 공분산의 두 배를 뺀 값이므로 짝지은 차이나 결합 모형으로 의존성을 반영해야 한다. 시계열 자기상관도 별도 고려한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사후 우위 확률이 0.95를 넘으면 멈추는 규칙이 거짓 양성률 5%를 보장하는가?

</div>

??? success "풀이"
    일반적으로 그렇지 않다. 사후확률은 모형·사전분포 아래의 조건부 진술이며 고정 귀무가설에서 반복 실행한 오류율과 다르다. 원하는 보장에 맞는 순차 설계와 탐색·중단 과정의 평가가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트만 있는 풀이와 위 추적기에서 어떤 조건을 점검해야 하는가?

</div>

??? success "풀이"
    임포트는 분석을 수행하지 않는다. 필터의 초기화·장치·우도 정규화, 전략 자료의 짝과 시간 의존성, 수익률 단위와 경제적 차이 기준을 정의해야 한다. 추적기의 할인율은 0보다 크고 1 이하여야 하며 갱신 분산의 의미를 명시한다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 스텝: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

상태 필터링은 고정된 모형 아래 국면 확률을 갱신하고, 전략 비교는 관심 수익률 차이의 사후분포를 계산한다. 미래 정보를 사용하는 검증 누수, 전략 간·시점 간 의존성과 탐색 과정을 고려한다. 사후확률, 경제적 유의성, 순차 검정의 오류율은 구분해야 한다.

**참고 문헌**

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. *Journal of Experimental Psychology: General*, 142(2), 573.
