# 국면 알아채기와 베이즈식 전략 평가

이 절은 계량 금융에서 베이즈 방법의 결정적인 쓰임새 둘을 다룬다. 곧 온라인 베이즈 갱신으로 시장 국면의 바뀜을 알아채는 일과, 베이즈 A/B 시험으로 거래 전략을 평가하는 일이다. 둘 다 베이즈 정리의 차례 갱신 성질을 살려 쓰며 관심 있는 양에 대한 곧은 확률 진술을 준다.

---

## 1. 베이즈식 국면 알아채기

### 시장 국면

금융 시장은 뚜렷이 다른 거동의 국면을 보인다. 오름장과 내림장, 변동성이 크고 작은 기간, 추세와 평균 회귀의 움직임 같은 것이다. 베이즈 방법은 다음을 위한 원칙 있는 틀을 준다.

1. **온라인 국면 추론**: 새 데이터가 들어오는 대로 국면에 대한 믿음을 고친다
2. **국면 확률 어림**: 국면마다 그 안에 있을 곧은 뒤확률
3. **국면을 조건 지은 예보**: 국면의 아리송함을 셈에 넣은 예측

### 단순한 두 국면 모형

국면이 둘(이를테면 낮은 변동성과 높은 변동성)인 모형을 생각해 보자.

$$
r_t \mid z_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2), \quad k \in \{1, 2\}
$$

넘어갈 확률은 다음과 같다.

$$
P(z_t = j \mid z_{t-1} = i) = A_{ij}
$$

### 베이즈식 거르기

때 걸음마다 국면 $k$에 있을 **거른 확률**을 고친다.

$$
P(z_t = k \mid r_{1:t}) \propto p(r_t \mid z_t = k) \sum_j A_{jk} \, P(z_{t-1} = j \mid r_{1:t-1})
$$

이는 숨은 마르코프 모형 거르개의 앞먹임이다([18장: 숨은 마르코프 모형](../../ch15/markov_chains/hmm.md)을 보라).

### 구현

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
        매개변수
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
        # 예측 걸음
        predicted = self.A.T @ self.filtered_prob
        
        # 국면마다의 가능도
        likelihood = torch.exp(
            -0.5 * ((r_t - self.mu) / self.sigma) ** 2
        ) / self.sigma
        
        # 새로 고치기 걸음
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

## 2. 전략 평가를 위한 베이즈 A/B 시험

베이즈 A/B 시험은 거래 전략을 견주는 자연스러운 틀을 주며, 어느 전략이 나은지에 대한 곧은 확률 진술과 원칙 있는 일찍 멈추기를 가능하게 한다.

### 문제의 얼개

두 전략(또는 전략과 잣대)을 견준다.

- **전략 A**(이를테면 기존 전략): 수익률 $r^A_1, \ldots, r^A_{n_A}$
- **전략 B**(이를테면 새 전략): 수익률 $r^B_1, \ldots, r^B_{n_B}$

**물음**: 전략 B가 전략 A보다 나을 확률은 얼마인가?

### 베이즈 모형

수익률이 가우스라고 놓는다(또는 샤프 비율 위의 뒤확률을 쓴다).

$$
r^A_i \sim \mathcal{N}(\mu_A, \sigma_A^2), \quad r^B_i \sim \mathcal{N}(\mu_B, \sigma_B^2)
$$

켤레 정규-역감마 앞확률에서 전략마다 평균 수익률의 뒤확률은 $t$ 분포이다.

### P(mu_B > mu_A | 데이터) 셈하기

뒤확률 표본을 쓰면 다음과 같다.

$$
P(\mu_B > \mu_A \mid \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \mathbf{1}[\mu_B^{(s)} > \mu_A^{(s)}]
$$

여기서 $\mu_A^{(s)}, \mu_B^{(s)}$은 저마다의 뒤확률 분포에서 뽑는다.

### 구현

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
        """정규-역감마 뒤확률 매개변수를 셈한다."""
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

### 빈도주의 검정보다 나은 점

| 갈래 | 베이즈 A/B 시험 | 빈도주의 t 검정 |
|--------|-------------------|-------------------|
| **출력** | $P(\text{B가 낫다} \mid \text{데이터})$ | p값(잘못 풀이될 때가 많다) |
| **일찍 멈추기** | 원칙 있음(뒤확률을 살핀다) | 거짓 양성률을 부풀린다 |
| **앞선 정보** | 정식으로 아우른다 | 무시한다 |
| **결정 틀** | 결정에 쓸 곧은 확률 | 기각과 기각 못 함의 이분법 |
| **표본 크기** | 자유로움(쉼 없이 살핀다) | 붙박임(미리 정한다) |

---

## 3. 시장 신호를 위한 베이즈 온라인 갱신

베이즈식 차례 갱신은 실시간 신호 다루기를 가능하게 한다.

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

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 국면을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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

| 쓰임새 | 베이즈 도구 | 핵심 이득 |
|-------------|---------------|-------------|
| **국면 알아채기** | HMM 거르기 | 실시간 국면 확률 |
| **전략 견줌** | A/B 시험 | 어느 쪽이 나은지에 대한 곧은 확률 |
| **신호 좇기** | 차례 갱신 | 불확실성을 곁들여 맞추어 가는 어림 |
| **일찍 멈추기** | 뒤확률 살피기 | p값 조작 없이 원칙 있게 멈추기 |

---

**참고 문헌**

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. *Journal of Experimental Psychology: General*, 142(2), 573.
