# 금융에서의 파라메터 불확실성

기대수익률·변동성·상관관계·요인 노출은 유한한 자료에서 추정하므로 불확실하다. 베이즈 분석은 파라메터의 사후분포를 미래 수익률의 예측과 의사결정에 반영한다. 결과의 신뢰도는 관측 모형과 사전분포, 시계열 의존성에 대한 가정에 달려 있다.

## 1. 어림 위험

### 기대수익률의 추정 오차

독립이고 분산이 일정한 월 수익률 T개의 평균은 표준오차 σ_month/√T를 갖는다. 월 평균을 12배 하여 연 기대수익률을 추정하면 표준오차도 12배다. 연 변동성이 30%이고 T=12Y라면 다음과 같다.

$$\operatorname{SE}(12\bar r)=\frac{0.30}{\sqrt Y}$$

| 관측 기간 | 연 기대수익률 추정의 표준오차 | 정규 근사 95% 구간의 반폭 |
|---|---:|---:|
| 5년 | 13.42%포인트 | 약 26.3%포인트 |
| 10년 | 9.49%포인트 | 약 18.6%포인트 |
| 20년 | 6.71%포인트 | 약 13.1%포인트 |

이는 연 기대수익률의 추정 오차이며 연 수익률 자체의 변동성과 다르다. 수익률의 자기상관·이분산이나 시간에 따른 평균 변화가 있으면 이 계산도 수정해야 한다.

### 평균과 미래 관측의 차이

독립 정규 수익률에서 평균과 분산을 모두 모르고 $p(\mu,\sigma^2)\propto1/\sigma^2$라는 부적절 사전을 사용하자. T≥2이고 잔차제곱합이 양수이면 다음 사후분포와 예측분포가 적절하다. 여기서 t의 두 번째 인자는 **스케일**이고 s²는 불편 표본분산이다.

$$\mu\mid D\sim t_{T-1}(\bar r,s/\sqrt T)$$

$$r_{new}\mid D\sim t_{T-1}(\bar r,s\sqrt{1+1/T})$$

스케일 제곱은 t분포의 분산 자체가 아니다. 자유도가 2보다 클 때 분산은 스케일 제곱에 (T−1)/(T−3)을 곱한 값이다. 알려진 분산의 정규-정규 모형과 이 미지 분산 모형을 구분한다.

## 2. 사후확률 예측 수익률

### 단일 자산

사후예측분포는 관측 모형을 파라메터 사후분포에 대해 적분한다.

$$p(r_{new}\mid D)=\int p(r_{new}\mid\mu,\sigma^2)p(\mu,\sigma^2\mid D)\,d\mu\,d\sigma^2$$

적절한 정규-역감마 모형에서는 자유도 2a_n의 t분포가 나온다. T−1이라는 자유도는 앞 절의 특정 부적절 사전에 해당하며 일반적인 켤레 사전의 결과는 아니다.

### 여러 자산

공분산의 불확실성까지 다루려면 사전분포의 파라메터화를 명시해야 한다. 예를 들어 사후분포가 $\Sigma\mid D\sim IW(\Psi_n,\nu_n)$, $\mu\mid\Sigma,D\sim N(m_n,\Sigma/\kappa_n)$인 N차원 모형에서, ν_n>N+1이면 다음과 같다.

$$\operatorname{Cov}(r_{new}\mid D)=\frac{\kappa_n+1}{\kappa_n}\frac{\Psi_n}{\nu_n-N-1}$$

이는 관측 잡음의 평균과 평균 파라메터의 사후 변동을 합한 값이다. 자료 수·자산 수만의 단일 팽창 계수를 어떤 사전분포에도 적용할 수는 없다. 자유도 조건이 충족되지 않으면 예측분포가 존재하더라도 유한한 공분산은 없을 수 있다.

## 3. 오그라뜨리는 어림기

### 공분산 수축

표본 공분산 S와 구조화한 목표 F를 결합할 수 있다.

$$\hat\Sigma_{shrink}=\delta F+(1-\delta)S,\qquad0\leq\delta\leq1$$

[Ledoit–Wolf 추정](https://www.ledoit.net/Well-conditioned2004.pdf)은 특정 로스와 점근 조건 아래 수축 강도를 자료에서 추정한다. 행렬의 조건수를 개선하는 빈도주의 방법이며 F를 사전분포라고 부르거나 δ를 베이즈 정밀도 웨이트로 동일시해서는 안 된다. 특정 계층 모형의 사후평균이 유사한 형태를 가질 수 있다는 해석과 구분한다.

### 기대수익률 수축

추정한 평균 벡터를 공통 중심이나 요인 모형의 값 쪽으로 이동시킬 수 있다. 정규 계층 모형의 사후평균은 한 예다. 제임스-스타인의 고전적 위험 개선에는 독립 정규 관측·알려진 공통 분산·차원 조건이 필요하다. 상관·이분산인 자산 수익률에 단순 공식을 그대로 적용해 같은 보장을 주장할 수 없다. [경험적 베이즈](../hierarchical/empirical_bayes.md)를 참고한다.

## 4. PyTorch 구현

아래 추정기는 평균을 스칼라 사전 중심으로 수축하고 표본 공분산에 경험적인 팽창 계수를 곱한다. 완전한 정규-역위샤트 갱신은 아니며 T≤N+2에서 계수를 2로 바꾸는 처리는 이론적 사후 공분산을 복구하지 않는다. prior_precision은 κ₀ 역할의 상대적 강도이며 고정된 절대 평균 정밀도와 다르다.

입력은 T×N 수익률 배열이며 표본 공분산·포트폴리오 분산이 유효해야 한다. predictive_sharpe는 예측 평균·표준편차의 비와 근사 표준오차를 반환할 뿐 샤프 비율의 완전한 사후분포를 구하지 않는다. prob_positive 필드는 실제로 0에서의 CDF, 즉 이 근사에서 비양수일 확률을 계산한다. 무위험수익률 차감과 연율화도 별도로 정의해야 한다.

```python
import torch
import torch.distributions as dist

class BayesianReturnEstimator:
    """
    불확실함을 퍼뜨리는 수익률 파라메터의 베이즈 어림.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_precision: float = 0.01):
        self.mu_0 = prior_mean
        self.kappa_0 = prior_precision  # 앞확률 평균의 정밀도
    
    def fit(self, returns: torch.Tensor):
        """
        다변량 수익률의 뒤확률 파라메터를 셈한다.
        
        파라메터
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

추정 오차와 수익률 자체의 위험을 구분하고 같은 단위·기간에서 비교한다. 두꺼운 예측 꼬리는 선택한 정규 혼합 모형의 결과이며 모든 사전분포에서 VaR·CVaR가 더 보수적이라는 뜻은 아니다.

고차원 공분산에서는 구조화한 추정과 충분한 자료가 중요하다. 수축 강도·목표를 선택한 뒤 시간 순서를 지킨 외부 검증으로 비용과 제약을 포함해 평가해야 하며, 베이즈라는 이유만으로 더 안정적인 투자 성과가 보장되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
월 평균을 연 기대수익률로 환산할 때 표준오차는 어떻게 변하는가?

</div>

??? success "연습문제 1 풀이"
    월 평균에 12를 곱하면 표준오차도 12배가 된다. 독립·정상 수익률과 T=12Y를 가정하면 연 기대수익률의 표준오차는 연 변동성/√Y다. 월 변동성을 √T로 나눈 값을 그대로 연 단위라고 표시하면 안 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
t분포의 스케일 제곱과 분산을 구분하라.

</div>

??? success "연습문제 2 풀이"
    자유도 ν와 스케일 s_t인 t분포의 분산은 ν>2일 때 ν s_t²/(ν−2)다. ν≤2에서는 유한 분산이 없다. 사후평균 파라메터의 분포와 새 수익률의 예측분포도 서로 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
공분산 수축과 완전한 베이즈 공분산 추론은 같은가?

</div>

??? success "연습문제 3 풀이"
    아니다. 수축은 점 추정량의 구조를 설명하며 베이즈·빈도주의 모두 사용할 수 있다. 베이즈 추론은 사전분포와 우도를 통해 행렬의 사후분포를 정하며 예측분포에 그 불확실성을 반영할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트와 위 추정기로 샤프 비율의 사후 양수 확률을 얻었다고 할 수 있는가?

</div>

??? success "연습문제 4 풀이"
    임포트는 분석을 수행하지 않는다. 위 함수도 샤프 비율의 사후표본을 구하지 않고 정규 근사 CDF를 계산하며 필드 이름과 확률 방향이 반대다. 완전한 분석에는 평균·공분산의 결합 사후표본과 초과수익률·기간의 정의가 필요하다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 스텝: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

추정 불확실성은 사후예측분포를 통해 반영할 수 있다. 수익률의 단위와 연율화, t 스케일과 분산, 공분산 수축과 사후분포를 구분한다. 간단한 팽창 계수나 샤프 비율 근사를 완전한 베이즈 추론으로 해석하지 않는다.

**참고 문헌**

- Barberis, N. (2000). Investing for the long run when returns are predictable. *Journal of Finance*, 55(1), 225-264.
- Kan, R., & Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621-656.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
