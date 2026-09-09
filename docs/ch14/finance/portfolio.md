# 베이즈 포트폴리오 최적화

고전적인 평균-분산 포트폴리오 최적화(마코위츠, 1952)는 어림한 매개변수를 아는 상수처럼 다루어 어림의 아리송함을 아랑곳하지 않는다. 베이즈 포트폴리오 방법은 매개변수의 불확실성을 최적화에 곧바로 아울러 더 튼튼하고 잘 흩뜨린 포트폴리오를 낸다.

---

## 1. 고전적 최적화의 어림 문제

평균-분산으로 가장 좋은 포트폴리오는 다음을 푼다.

$$
\mathbf{w}^* = \frac{1}{\gamma} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}
$$

여기서 $\boldsymbol{\mu}$과 $\boldsymbol{\Sigma}$은 기대 수익률과 공분산 행렬이다. 이를 표본 어림값 $\hat{\boldsymbol{\mu}}$과 $\hat{\boldsymbol{\Sigma}}$으로 갈아 끼우면 그렇게 나온 "끼워 넣기" 포트폴리오는 다음을 겪는다.

- **어림 오차 부풀림**: $\hat{\boldsymbol{\Sigma}}^{-1}$이 $\hat{\boldsymbol{\mu}}$의 오차를 부풀린다
- **극단적인 포지션**: 어림 수익률이 가장 큰 자산에 포트폴리오가 몰린다
- **흔들림**: 데이터가 조금만 바뀌어도 배분이 크게 달라진다

---

## 2. 베이즈식 방법

### 예측 수익률 분포

점 어림값을 쓰는 대신 베이즈 방법은 매개변수의 아리송함 위에서 적분한다.

$$
p(\mathbf{r}_{\text{new}} \mid \mathcal{D}) = \int p(\mathbf{r}_{\text{new}} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, p(\boldsymbol{\mu}, \boldsymbol{\Sigma} \mid \mathcal{D}) \, d\boldsymbol{\mu} \, d\boldsymbol{\Sigma}
$$

켤레 정규-역위샤트 앞확률에서 예측 분포는 가우스보다 꼬리가 두꺼운 다변량 $t$ 분포이며, 어림 위험을 자연스럽게 셈에 넣는다.

### 베이즈로 가장 좋은 포트폴리오

베이즈 포트폴리오는 **예측** 분포 아래에서 기대 효용을 가장 크게 한다.

$$
\mathbf{w}^*_{\text{Bayes}} = \arg\max_{\mathbf{w}} \mathbb{E}_{p(\mathbf{r} \mid \mathcal{D})}[U(\mathbf{w}^\top \mathbf{r})]
$$

이차 효용에서는 다음이 나온다.

$$
\mathbf{w}^*_{\text{Bayes}} = \frac{1}{\gamma} \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\boldsymbol{\mu}}
$$

여기서 $\tilde{\boldsymbol{\mu}}$과 $\tilde{\boldsymbol{\Sigma}}$은 예측 분포의 평균과 공분산이다.

---

## 3. 블랙-리터먼 모형

블랙-리터먼(1992) 모형은 가장 널리 쓰이는 베이즈 포트폴리오 틀이다. 시장 균형 앞확률과 투자자의 견해를 어우른다.

### 앞확률: 시장 균형 수익률

$$
\boldsymbol{\pi} = \gamma \boldsymbol{\Sigma} \mathbf{w}_{\text{mkt}}
$$

여기서 $\mathbf{w}_{\text{mkt}}$은 시가총액 무게이다.

### 견해

투자자의 견해는 다음과 같이 나타낸다.

$$
\mathbf{P} \boldsymbol{\mu} = \mathbf{q} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})
$$

여기서 $\mathbf{P}$은 고르기 행렬, $\mathbf{q}$은 견해가 내다보는 수익률 벡터, $\boldsymbol{\Omega}$은 견해의 아리송함을 나타낸다.

### 뒤확률

$$
\hat{\boldsymbol{\mu}}_{\text{BL}} = \left[(\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\pi} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{q}\right]
$$

이는 균형 앞확률과 투자자 견해를 정밀도로 무게 준 평균이며, 16.1절의 켤레 정규-정규 갱신과 꼭 같다.

### PyTorch 구현

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
    
    매개변수
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

---

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 포트폴리오을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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

| 방법 | 수익률 어림값 | 불확실성 | 흩뜨리기 |
|----------|-----------------|-------------|-----------------|
| 마코위츠(끼워 넣기) | 표본 평균 | 무시함 | 나쁨 |
| 베이즈(퍼진 앞확률) | 뒤확률 예측 평균 | 뒤확률 전체 | 더 나음 |
| 블랙-리터먼 | 균형 + 견해 | 견해에 대한 자신감 | 가장 좋음 |
| 튼튼한 최적화 | 불확실성 집합 | 최악의 경우 | 조심스러움 |

---

**참고 문헌**

- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28-43.
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
- Avramov, D., & Zhou, G. (2010). Bayesian portfolio analysis. *Annual Review of Financial Economics*, 2, 25-47.
