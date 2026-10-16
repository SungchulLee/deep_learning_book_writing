# 베이즈 포트폴리오 최적화

표본 평균과 공분산을 대입한 평균-분산 최적화는 추정 오차에 민감할 수 있다. 베이즈 접근은 파라메터 불확실성을 예측분포에 반영하고 이를 목적함수와 제약 아래 의사결정에 사용한다. 블랙-리터먼 모델은 균형 기대수익률과 투자자의 견해를 정규 모형으로 결합하는 한 방법이다.

## 1. 고전적 최적화의 어림 문제

위험 회피 계수 γ>0와 양의 정부호 공분산 Σ를 가정하고, 초과수익률 μ에 대해 다음 **제약 없는 평균-분산 목적함수**를 최대화하자.

$$\max_w\ w^\top\mu-\frac\gamma2w^\top\Sigma w$$

미분하면 다음 해를 얻는다.

$$w^*=\frac1\gamma\Sigma^{-1}\mu$$

이는 위험자산 비중의 합이 1이라는 예산 제약이나 공매도 제한을 포함하지 않는다. 그런 조건이 있으면 제약 최적화를 풀어야 한다. Σ의 작은 고유값과 평균 추정 오차가 결합하면 포지션이 크게 변할 수 있다. 추정수익률이 가장 큰 한 자산에 언제나 몰리는 것은 아니며 상관구조와 제약도 영향을 준다.

## 2. 베이즈식 방법

미래 수익률의 사후예측분포는 다음과 같다.

$$p(r_{new}\mid D)=\int p(r_{new}\mid\mu,\Sigma)p(\mu,\Sigma\mid D)\,d\mu\,d\Sigma$$

적절한 정규-역위샤트 모형은 다변량 t 예측분포를 주며, 필요한 모멘트가 존재하는지 확인한다. 예측 평균 m과 공분산 C로 평균-분산 목적함수 $w^\top m-\gamma w^\top Cw/2$를 정하면 제약 없는 해는 γ⁻¹C⁻¹m이다.

이를 **이차 효용의 기대값**과 혼동해서는 안 된다. U(x)=x−γx²/2의 기대값에는 $E[rr^\top]=C+mm^\top$가 들어가므로 제약 없는 해는 다음과 다르다.

$$w_U^*=\frac1\gamma(C+mm^\top)^{-1}m$$

어떤 목적함수·위험 정의·제약을 쓰는지 먼저 정해야 한다. 베이즈 사후예측을 사용해도 성과나 분산투자가 자동으로 개선되지는 않는다.

## 3. 블랙-리터먼 모델

### 균형 사전분포

고정된 수익률 공분산 Σ와 시장 비중 w_mkt, 위험 회피 계수 γ에서 균형 초과수익률 π를 정의한다.

$$\pi=\gamma\Sigma w_{mkt},\qquad \mu\sim N(\pi,\tau\Sigma),\quad\tau>0$$

### 투자자의 견해

K개의 견해를 q|μ∼N(Pμ,Ω)로 표현한다. P는 K×N의 선형 결합 행렬이며 절대 수익률과 상대 수익률 견해 모두 담을 수 있다. Ω는 견해 오차의 공분산이다. 모든 수익률·공분산은 같은 기간과 초과수익률 기준을 사용한다.

### 사후 평균과 공분산

Σ와 Ω가 양의 정부호이면 다음과 같다.

$$V=[(\tau\Sigma)^{-1}+P^\top\Omega^{-1}P]^{-1}$$

$$m=V[(\tau\Sigma)^{-1}\pi+P^\top\Omega^{-1}q]$$

V는 **기대수익률 파라메터 μ의 사후 공분산**이다. 관측 모형 r_new|μ∼N(μ,Σ)와 독립인 견해 오차를 가정하면 미래 수익률의 예측 공분산은 Σ+V다. 이 구분은 [공식 구현 설명](https://www.mathworks.com/help/finance/black-litterman-portfolio-optimization.html)에서도 확인할 수 있다.

### PyTorch 구현

아래 함수는 m과 V를 반환하며 포트폴리오 비중 최적화는 수행하지 않는다. 반환된 sigma_bl을 수익률 공분산 Σ와 혼동하지 않는다. 입력 행렬의 차원·장치·자료형, τ와 γ의 양수 조건 및 역행렬의 수치 안정성을 확인한다. 견해를 매우 확신하는 Ω나 작은 고유값은 수치 문제를 일으킬 수 있다.

```python
import torch

def black_litterman(
    sigma: torch.Tensor,
    w_mkt: torch.Tensor,
    P: torch.Tensor,
    q: torch.Tensor,
    omega: torch.Tensor,
    gamma: float = 2.5,
    tau: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    블랙-리터만 뒤확률 수익률.
    
    파라메터
    ----------
    sigma : (N, N) 공분산 행렬
    w_mkt : (N,) 시가총액 무게
    P : 견해 K개를 고르는 (K, N) 행렬
    q : (K,) 견해가 기대하는 수익률
    omega : (K, K) 견해의 불확실함
    gamma : 위험 꺼림 계수
    tau : 앞확률 불확실함의 눈금 인자
    
    반환값
    -------
    mu_bl : (N,) 뒤확률 기대 수익률
    sigma_bl : (N, N) 뒤확률 공분산
    """
    # 평형 수익률
    pi = gamma * sigma @ w_mkt
    
    # 앞확률 정밀도
    tau_sigma_inv = torch.linalg.inv(tau * sigma)
    omega_inv = torch.linalg.inv(omega)
    
    # 뒤확률 정밀도와 평균
    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
    sigma_bl = torch.linalg.inv(posterior_precision)
    mu_bl = sigma_bl @ (tau_sigma_inv @ pi + P.T @ omega_inv @ q)
    
    return mu_bl, sigma_bl
```


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
γ⁻¹Σ⁻¹μ라는 비중 공식에 포함되지 않은 제약은 무엇인가?

</div>

??? success "연습문제 1 풀이"
    비중 합이 1인 예산 제약, 공매도·레버리지·거래비용 제한이 포함되지 않는다. 공분산이 양의 정부호인 제약 없는 평균-분산 목적함수의 해이므로 실제 제약을 추가하면 다시 최적화해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기대 이차 효용과 평균-분산 목적함수의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    U(x)=x−γx²/2이면 E[x²]=Var(x)+(E[x])²가 들어간다. 평균-분산 목적함수는 평균의 제곱 항을 포함하지 않으므로 두 목적함수의 최적 비중을 같은 공식으로 쓸 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
블랙-리터먼의 V와 미래 수익률 공분산은 왜 다른가?

</div>

??? success "연습문제 3 풀이"
    V는 불확실한 평균 μ의 사후 공분산이다. 알려진 조건부 관측 공분산 Σ를 가정하면 미래 수익률 공분산은 총분산 법칙에 따라 Σ+V다. 코드는 V만 반환한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 임포트와 위 함수만으로 투자 비중을 얻을 수 있는가?

</div>

??? success "연습문제 4 풀이"
    임포트는 분석을 수행하지 않으며 black_litterman도 기대수익률 사후평균과 그 공분산만 반환한다. 예측 위험·목적함수·제약을 정하고 별도의 최적화를 수행해야 한다. 견해나 사전 강도가 잘못되면 안정적인 성과를 보장하지 못한다.

    ```python
    import numpy as np
    # 구현은 주제에 따라 달라진다.
    # 켤레 모형: 닫힌 꼴 뒤확률 새로 고치기.
    # 켤레가 아닌 모형: MCMC 또는 변분 추론.
    # 핵심 걸음: 앞확률 정하기, 가능도 셈하기, 뒤확률 이끌어 내기/어림하기.
    ```

## 정리하며

베이즈 포트폴리오 분석은 예측분포와 의사결정 목적함수를 연결한다. 평균-분산 목적함수와 기대 이차 효용, 평균의 사후 공분산과 수익률 예측 공분산을 구분한다. 블랙-리터먼은 균형 수익률과 견해를 결합하지만 비중 제약과 실제 성과 검증은 별도로 필요하다.

**참고 문헌**

- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
- Avramov, D., & Zhou, G. (2010). Bayesian portfolio analysis. *Annual Review of Financial Economics*, 2, 25-47.
