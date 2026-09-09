# 금융에서의 매개변수 불확실성

금융 모형은 기대 수익률, 변동성, 상관, 요인 적재값처럼 어림에 큰 아리송함이 딸린 매개변수에 기댄다. 베이즈 방법은 이 아리송함을 수로 나타내고 그것을 결정까지 퍼뜨려, 더 튼튼한 포트폴리오 짜기와 위험 다스리기, 전략 평가를 이끈다.

---

## 1. 어림 위험

### 문제

달마다의 관찰 $T$개로 자산의 기대 수익률 $\mu$을 어림한다고 하자. 표본 평균의 표준 오차는 $\sigma / \sqrt{T}$이다. 연 변동성이 $\sigma \approx 0.30$(달로는 $\approx 0.087$)인 흔한 주식이라면 다음과 같다.

| 어림 창 | 표준 오차(연 환산) | 95% 구간 너비 |
|-------------------|-----------------------------|---------------|
| 5년(60달) | 3.87% | ±7.6% |
| 10년(120달) | 2.74% | ±5.4% |
| 20년(240달) | 1.94% | ±3.8% |

기대 수익률 어림값은 흔한 위험 프리미엄(연 3~8%)에 견주어 **몹시 잡음이 많다**. 이것이 계량 투자의 근본 어려움이다.

### 베이즈식 다룸

퍼진 앞확률을 쓴 정규-정규 모형에서 평균 수익률의 베이즈 뒤확률은 다음과 같다.

$$
\mu \mid \mathcal{D} \sim t_{T-1}\left(\bar{r}, \frac{s^2}{T}\right)
$$

앞으로의 수익률에 대한 예측 분포는 꼬리가 더 두껍다.

$$
r_{\text{new}} \mid \mathcal{D} \sim t_{T-1}\left(\bar{r}, s^2\left(1 + \frac{1}{T}\right)\right)
$$

더해진 흩어짐 $s^2/T$은 **어림 위험**, 곧 참 평균 자체에 대한 아리송함을 비춘다.

---

## 2. 뒤확률 예측 수익률

### 자산 하나

수익률 관찰 $\{r_1, \ldots, r_T\}$ $T$개를 가진 자산 하나에 대해 뒤확률 예측 분포는 매개변수의 아리송함 위에서 적분한다.

$$
p(r_{\text{new}} \mid \mathcal{D}) = \int p(r_{\text{new}} \mid \mu, \sigma^2) \, p(\mu, \sigma^2 \mid \mathcal{D}) \, d\mu \, d\sigma^2
$$

켤레 정규-역감마 모형에서는 자유도 $T - 1$인 $t$ 분포가 나온다.

### 포트폴리오 수준

자산 $N$개로 이루어진 포트폴리오에서 예측 공분산 행렬은 다음과 같다.

$$
\tilde{\boldsymbol{\Sigma}} = \hat{\boldsymbol{\Sigma}} \cdot \frac{T-1}{T-N-2} \cdot \left(1 + \frac{1}{T}\right)
$$

부풀림 인자 $\frac{T-1}{T-N-2}$은 $N$이 $T$에 가까워질수록 크게 자라며, 공분산 어림에서의 차원의 저주를 비춘다.

---

## 3. 오그라뜨리는 어림기

베이즈식 매개변수 불확실성은 자연스럽게 **오그라들기**, 곧 어림값을 짜임새 있는 앞확률 쪽으로 끌어당기는 일로 이어진다.

### 르두아-볼프 오그라뜨리기

오그라뜨리는 공분산 어림기는 다음과 같다.

$$
\hat{\boldsymbol{\Sigma}}_{\text{shrink}} = \delta \mathbf{F} + (1-\delta) \hat{\boldsymbol{\Sigma}}_{\text{sample}}
$$

여기서 $\mathbf{F}$은 짜임새 있는 목표(이를테면 한 요인 모형)이고 $\delta$은 가장 좋은 오그라듦의 세기이다.

여기에는 베이즈식 풀이가 있다. 짜임새 있는 목표 $\mathbf{F}$은 앞확률이고, 표본 공분산은 데이터이며, $\delta$은 서로의 정밀도를 비춘다.

### 수익률을 위한 제임스-스타인 오그라뜨리기

제임스-스타인 어림기는 기대 수익률을 공통 평균 쪽으로 오그라뜨린다.

$$
\hat{\mu}_i^{\text{JS}} = \bar{\mu} + (1 - \hat{c}) (\hat{\mu}_i - \bar{\mu})
$$

이는 정규 층층 모형 아래의 경험적 베이즈 해이다([경험적 베이즈](../hierarchical/empirical_bayes.md)를 보라).

---

## 4. PyTorch 구현

```python
import torch
import torch.distributions as dist

class BayesianReturnEstimator:
    """
    불확실함을 퍼뜨리는 수익률 매개변수의 베이즈 어림.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_precision: float = 0.01):
        self.mu_0 = prior_mean
        self.kappa_0 = prior_precision  # 앞확률 평균의 정밀도
    
    def fit(self, returns: torch.Tensor):
        """
        다변량 수익률의 뒤확률 매개변수를 셈한다.
        
        매개변수
        ----------
        returns : 수익률 관측의 (T, N) 텐서
        """
        T, N = returns.shape
        
        self.T = T
        self.N = N
        self.sample_mean = returns.mean(dim=0)
        self.sample_cov = torch.cov(returns.T)
        
        # 뒤확률 평균(정밀도로 무게 준)
        posterior_precision = self.kappa_0 + T
        self.posterior_mean = (
            self.kappa_0 * self.mu_0 + T * self.sample_mean
        ) / posterior_precision
        
        # 예측 공분산(어림 불확실함을 담아)
        if T > N + 2:
            inflation = (T - 1) / (T - N - 2) * (1 + 1/T)
        else:
            inflation = 2.0  # 작은 표본을 위한 물러섬
        self.predictive_cov = self.sample_cov * inflation
        
        return self
    
    def predictive_sharpe(self, weights: torch.Tensor) -> dict:
        """
        포트폴리오 샤프 비율의 뒤확률 분포를 셈한다.
        """
        port_mean = weights @ self.posterior_mean
        port_var = weights @ self.predictive_cov @ weights
        port_std = port_var.sqrt()
        
        sharpe = port_mean / port_std
        
        # 샤프 비율의 어림 표준 오차
        sharpe_se = ((1 + 0.5 * sharpe**2) / self.T).sqrt()
        
        return {
            'sharpe': sharpe.item(),
            'sharpe_se': sharpe_se.item(),
            'prob_positive': dist.Normal(sharpe, sharpe_se).cdf(
                torch.tensor(0.0)).item()
        }
```

---

## 5. 실전을 위한 핵심 통찰

1. **기대 수익률은 변동성이나 상관보다 훨씬 덜 정밀하게 어림된다.** 이 치우침이 모형 설계에 길잡이가 되어야 한다. 수익률 예보는 미더워하지 말고 위험 어림값은 더 믿어라.

2. **베이즈 예측 분포는 꼬리가 더 두껍다.** 점 어림값을 끼워 넣은 가우스 모형보다 그러하며, 어림 위험을 자연스럽게 셈에 넣어 더 조심스러운 VaR과 CVaR 어림값을 낸다.

3. **짜임새 있는 목표(요인 모형, 같은 무게)로 오그라뜨리는 것**은 임시방편의 벌주기 요령이 아니라, 매개변수 불확실성에 대한 가장 알맞은 베이즈식 응답이다.

4. **차원이 높아질수록 필요한 표본 크기가 커진다.** 자산 $N$개와 관찰 $T$개에서 $N/T$이 커질수록 어림의 질이 빠르게 나빠진다.

---

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 매개변수의 불확실성을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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

이 마당은 어림 위험、뒤확률 예측 수익률、오그라뜨리는 어림기、PyTorch 구현을 차례로 짚었다.

**참고 문헌**

- Barberis, N. (2000). Investing for the long run when returns are predictable. *Journal of Finance*, 55(1), 225-264.
- Kan, R., & Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621-656.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
