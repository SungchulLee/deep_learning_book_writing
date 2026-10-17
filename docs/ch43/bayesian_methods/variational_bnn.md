# 변분 베이즈 뉴럴 넷트웍
**변분 베이즈 뉴럴 넷트웍**(변이 BNN)은 다룰 수 없는 베이즈 추론 문제를 최적화 문제로 바꾸어 근사 사후분포 추론을 이치에 닿게 이룬다. 모델 증거 아래끝(ELBO)을 가장 크게 하여 네트워크 웨이트에 대한 다룰 수 있는 근사 분포를 배우고, 딥러닝에서 크게 늘릴 수 있는 불확실성 재기를 이룬다.

---

## 1. 왜 하는가: 크게 늘릴 수 있는 베이즈 추론

### 사후분포 추론의 어려움

뉴럴 넷트웍 웨이트에 대한 참 사후분포는 다룰 수 없다.

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

**어려움**:

- 모델 증거 $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta$에는 닫힌 형태가 없다
- 파라메터 밭의 차수가 높다(파라메터 $10^6$~$10^9$개)
- 사후분포의 터가 얽히고 봉우리가 여럿이다
- MCMC 방법은 큰 네트워크에 너무 더디다

### 변이의 길

**핵심 깨침**: 다룰 수 없는 사후분포를 다룰 수 있는 분포로 추정한다.

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

여기서 $q_\phi(\theta)$은 파라메터 $\phi$을 지닌 다룰 수 있는 클래스(가우시안 따위)에서 온다.

**최적화 목표**: KL 갈림을 가장 작게 하는 $\phi$을 찾는다.

$$
\phi^* = \arg\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

### 변분 추론의 나은 점

| 나은 점 | 풀이 |
|-----------|-------------|
| **크게 늘리기** | 추론을 최적화로 바꾼다 |
| **너그러움** | 근사 클래스를 고를 수 있다 |
| **잘 듦** | 확률로 최적화에 잘 맞는다 |
| **어울림** | 여느 딥러닝 연장과 함께 쓴다 |

---

## 2. 모델 증거 아래끝(ELBO)

### 유도

로그 모델 증거에서 비롯한다.

$$
\log p(\mathcal{D}) = \log \int p(\mathcal{D} \mid \theta) p(\theta) d\theta
$$

변이 분포 $q_\phi(\theta)$을 들인다.

$$
\log p(\mathcal{D}) = \log \int \frac{q_\phi(\theta)}{q_\phi(\theta)} p(\mathcal{D} \mid \theta) p(\theta) d\theta
$$

옌센 부등식을 쓴다.

$$
\log p(\mathcal{D}) \geq \int q_\phi(\theta) \log \frac{p(\mathcal{D} \mid \theta) p(\theta)}{q_\phi(\theta)} d\theta = \mathcal{L}(\phi)
$$

이 아래끝 $\mathcal{L}(\phi)$이 **모델 증거 아래끝(ELBO)**이다.

### ELBO 쪼개기

ELBO는 이렇게 적을 수 있다.

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

**풀이**:

- **첫째 항**: 바라는 로그 우도(데이터 맞춤)
- **둘째 항**: 사전분포에 대한 KL 갈림(번거로움 벌)

**다른 꼴**:

$$
\mathcal{L}(\phi) = \log p(\mathcal{D}) - \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

$\text{KL} \geq 0$이므로 ELBO를 가장 크게 하는 일은 사후분포에 대한 KL을 가장 작게 하는 일과 같다.

### 로스 함수로서의 ELBO

뉴럴 넷트웍을 익힐 때는 음수 ELBO를 가장 작게 한다.

$$
\boxed{\mathcal{L}_{\text{VI}}(\phi) = -\mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)] + \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

가우시안 우도를 쓰는 회귀에서

$$
\mathcal{L}_{\text{VI}}(\phi) = \frac{1}{2\sigma^2} \mathbb{E}_{q_\phi}\left[\sum_{i=1}^N (y_i - f_\theta(x_i))^2\right] + \text{KL}(q_\phi \| p)
$$

클래스 우도를 쓰는 분류에서

$$
\mathcal{L}_{\text{VI}}(\phi) = -\mathbb{E}_{q_\phi}\left[\sum_{i=1}^N \sum_c y_{ic} \log \text{softmax}(f_\theta(x_i))_c\right] + \text{KL}(q_\phi \| p)
$$

---

## 3. 평균장 변분 추론

### 곱으로 가른 근사

**평균장** 가정은 사후분포를 곱으로 가른다.

$$
q_\phi(\theta) = \prod_{j=1}^d q_{\phi_j}(\theta_j)
$$

파라메터마다 독립적인 제 분포를 지닌다.

### 가우시안 평균장

가장 흔히 고르는 것은 대각 가우시안이다.

$$
\boxed{q_\phi(\theta) = \prod_{j=1}^d \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)}
$$

**변이 파라메터**: $\phi = \{(\mu_j, \sigma_j)\}_{j=1}^d$

**온 파라메터**: $2d$(본디 네트워크의 두 배)

### 가우시안 분포의 KL 갈림

가우시안 사전분포 $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$에서

$$
\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^d \left[ \frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log \frac{\sigma_j^2}{\sigma_p^2} \right]
$$

**파라메터마다의 KL**:

$$
\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, \sigma_p^2)) = \frac{\mu^2 + \sigma^2}{2\sigma_p^2} - \frac{1}{2} - \log \frac{\sigma}{\sigma_p}
$$

### 평균장의 한계

**남남임 가정**: 웨이트끼리의 얽힘을 놓친다

- 웨이트끼리 주고받음을 담지 못한다
- 불확실성을 낮게 볼 수 있다
- 사후분포의 공분산이 대각이다

**봉우리 하나**: 가우시안 근사는 봉우리 하나만 담는다

- 봉우리가 여럿인 사후분포를 드러내지 못한다
- 중요한 사후분포 구조를 놓칠 수 있다

---

## 4. 파라메터 다시 잡기 재주

### 그래디언트 문제

ELBO를 가장 좋게 하려면 다음이 있어야 한다.

$$
\nabla_\phi \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)]
$$

**어려움**: 바람이 $\phi$에 매인 $q_\phi$에 대한 것이다.

### 파라메터 다시 잡기 풀이

**핵심 깨침**: $\theta$을 $\phi$과 잡음의 붙박인 함수로 적는다.

$$
\theta = g(\phi, \epsilon) = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

이제 바람은 $\phi$과 독립적인 $\epsilon$에 대한 것이다.

$$
\mathbb{E}_{q_\phi(\theta)}[f(\theta)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu + \sigma \odot \epsilon)]
$$

### 그래디언트 계산

그래디언트는 이렇게 된다.

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] = \mathbb{E}_{\epsilon}\left[\nabla_\phi f(\mu + \sigma \odot \epsilon)\right]
$$

표본 하나로 하는 **몬테카를로 근사**:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\theta)] \approx \nabla_\phi f(\mu + \sigma \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

### 참으로 짜기

**양이 되게 하는 파라메터 잡기**: $\sigma = \log(1 + e^\rho)$이 되는 $\rho$을 쓴다(소프트플러스)

**그래디언트 흐름**:

$$
\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial \theta}, \quad \frac{\partial \mathcal{L}}{\partial \rho} = \frac{\partial \mathcal{L}}{\partial \theta} \cdot \epsilon \cdot \frac{e^\rho}{1 + e^\rho}
$$

---

## 5. 되돌아가며 베이즈

### 알고리즘 두루 보기

**되돌아가며 베이즈**(블런델 등, 2015)은 변이 BNN을 익히는 기초 알고리즘이다.

**알고리즘: 되돌아가며 베이즈**

```
들임: 자료 꾸러미 D, 앞선 분포 p(θ), 넷트웍 구조
날임: 변이 파라메터 φ = {μ, ρ}

μ, ρ의 첫자리를 아무렇게나 잡는다
판마다:
    잔 배치 B마다:
        1. ε ~ N(0, I)을 뽑는다
        2. θ = μ + softplus(ρ) ⊙ ε을 셈한다
        3. 잃음을 셈한다:
           L = -log p(B|θ) + (1/M) * KL(q_φ || p)
           여기서 M = 잔 배치의 수
        4. 그래디언트 ∇_μ L, ∇_ρ L을 셈한다
        5. μ ← μ - α ∇_μ L으로 고친다
        6. ρ ← ρ - α ∇_ρ L으로 고친다
```

### 잔 배치 ELBO

잔 배치로 익힐 때는 ELBO의 잣대를 알맞게 맞춘다.

$$
\mathcal{L}(\phi) \approx \frac{N}{|B|} \sum_{i \in B} \log p(y_i \mid x_i, \theta) - \text{KL}(q_\phi \| p)
$$

**KL 웨이트 주기**: KL 항은 잔 배치마다 한 번 계산하고 $1/M$으로 잣대를 맞춘다. 여기서 $M$은 잔 배치의 수다.

### 웨이트의 불확실성

학습된 변이 분포는 다음을 준다.

$$
W_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)
$$

**풀이**:

- $\mu_{ij}$: 가장 그럴듯한 웨이트 값
- $\sigma_{ij}$: 웨이트 값의 불확실성
- $\sigma$이 크면 → 불확실성이 크고 → 추론을 덜 자신한다

---

## 6. 그 자리 파라메터 다시 잡기 재주

### 왜 하는가

여느 파라메터 다시 잡기는 웨이트를 뽑는다.

$$
W = \mu_W + \sigma_W \odot \epsilon_W
$$

그러고 나서 활성화를 계산한다.

$$
a = Wx
$$

**문제**: 네트워크가 너르면 그래디언트 근사의 분산이 크다.

### 그 자리 파라메터 다시 잡기

**핵심 깨침**: 선형 켜에서는 활성화를 곧바로 뽑는다.

$$
a = Wx = (\mu_W + \sigma_W \odot \epsilon_W)x
$$

$$
a \sim \mathcal{N}(\mu_W x, (\sigma_W^2 \odot x^2) \mathbf{1})
$$

**활성화 켜에서 파라메터를 다시 잡는다**:

$$
a = \mu_W x + \sqrt{\sigma_W^2 \odot x^2} \odot \epsilon_a
$$

여기서 $\epsilon_a \sim \mathcal{N}(0, I)$의 차수는 출력 크기와 같다.

### 나은 점

| 특성 | 여느 것 | 그 자리 |
|--------|----------|-------|
| 잡음의 차수 | $d$(웨이트) | $n$(활성화) |
| 그래디언트 분산 | 크다 | 작다 |
| 계산 비용 | 같다 | 같다 |
| 얽힘 | 웨이트끼리 얽힌다 | 활성화끼리 남남이다 |

### 짜기

웨이트 행렬이 $W \in \mathbb{R}^{m \times n}$인 켜에서

```python
# 여느 파라메터 다시 잡기
W = mu_W + sigma_W * eps_W  # eps_W: (m, n)
a = W @ x                    # a: (n,)

# 그 자리 파라메터 다시 잡기
a_mu = mu_W @ x                           # (n,)
a_var = (sigma_W**2) @ (x**2)            # (n,)
a = a_mu + sqrt(a_var) * eps_a           # eps_a: (n,)
```

---

## 7. KL 갈림을 다루는 꾀

### 정확한 KL(닫힌 형태)

가우시안과 가우시안 사이의 KL에서

$$
\text{KL}(q \| p) = \frac{1}{2}\left[\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q) - d + \log\frac{|\Sigma_p|}{|\Sigma_q|}\right]
$$

평균이 0인 사전분포를 지닌 대각 가우시안에서

$$
\text{KL} = \frac{1}{2}\sum_j \left[\frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log\frac{\sigma_j^2}{\sigma_p^2}\right]
$$

### 몬테카를로 KL 근사

짝이 맞지 않는 사전분포에서는 뽑아서 KL을 추정한다.

$$
\text{KL}(q \| p) = \mathbb{E}_{q}[\log q(\theta) - \log p(\theta)] \approx \frac{1}{S}\sum_{s=1}^S [\log q(\theta^{(s)}) - \log p(\theta^{(s)})]
$$

### KL 천천히 올리기

**문제**: 학습 초에 KL 항이 힘을 크게 써서 $q$이 사전분포로 주저앉을 수 있다.

**풀이**: KL 웨이트를 차츰 올린다.

$$
\mathcal{L}_t(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \beta_t \cdot \text{KL}(q_\phi \| p)
$$

**올리는 구조**:

**곧게**:

$$
\beta_t = \min(1, t / T_{\text{warmup}})
$$

**시그모이드**:

$$
\beta_t = \frac{1}{1 + \exp(-(t - T_{\text{mid}})/\tau)}
$$

**돌림**:

$$
\beta_t = \min(1, \text{mod}(t, T_{\text{cycle}}) / T_{\text{rise}})
$$

---

## 8. 평균장을 넘어

### 전체 공분산 가우시안

$$
q(\theta) = \mathcal{N}(\mu, \Sigma)
$$

**파라메터**: $d + d(d+1)/2$(평균 + 아래 세모 촐레스키)

**문제**: 파라메터 $O(d^2)$개와 계산 $O(d^3)$ — 큰 네트워크에서는 다룰 수 없다.

### 낮은 자리 근사

**낮은 자리 더하기 대각**:

$$
\Sigma = D + VV^\top
$$

여기서 $D$은 대각이고 $V \in \mathbb{R}^{d \times r}$이며 $r \ll d$이다.

**파라메터**: $d + dr$

**샘플링**: $\theta = \mu + D^{1/2}\epsilon_1 + V\epsilon_2$이고 여기서 $\epsilon_1 \in \mathbb{R}^d$, $\epsilon_2 \in \mathbb{R}^r$이다.

### 행렬 변수 가우시안

웨이트 행렬 $W \in \mathbb{R}^{m \times n}$에서

$$
q(W) = \mathcal{MN}(M, U, V)
$$

여기서 $U \in \mathbb{R}^{m \times m}$이고 $V \in \mathbb{R}^{n \times n}$이다.

**파라메터**: $mn + m^2 + n^2$($mn(mn+1)/2$보다 훨씬 적다)

### 정규화 흐름

되돌릴 수 있는 함수로 단순한 분포를 바꾼다.

$$
\theta = f_K \circ f_{K-1} \circ \cdots \circ f_1(z), \quad z \sim \mathcal{N}(0, I)
$$

**밀도**:

$$
q(\theta) = q_0(f^{-1}(\theta)) \left|\det \frac{\partial f^{-1}}{\partial \theta}\right|
$$

**즐겨 쓰는 흐름**:

- **판판한 흐름**: $f(z) = z + u \cdot \tanh(w^\top z + b)$
- **살 흐름**: $f(z) = z + \beta h(\alpha, r)(z - z_0)$
- **RealNVP**: 야코비안 행렬을 다룰 수 있는 짝 켜
- **IAF**: 역 제 회귀 흐름

---

## 9. 참으로 헤아릴 것

### 사전분포 고르기

사전분포 $p(\theta)$은 정규화와 불확실성에 함께 걸린다.

**여느 가우시안**:

$$
p(\theta) = \mathcal{N}(0, \sigma_p^2 I)
$$

**잣대 섞기**(든든하도록):

$$
p(\theta) = \pi \mathcal{N}(0, \sigma_1^2) + (1-\pi) \mathcal{N}(0, \sigma_2^2)
$$

**겪어 본 길잡이**:

- $\sigma_p = 1$에서 비롯한다
- 따짐 성능을 보고 맞춘다
- 절로 맞추려면 레이어진 사전분포를 헤아린다

### 첫자리 잡기

**평균의 첫자리**: 

- 여느 첫자리 잡기(자비에, 허)
- 또는 사전 학습된 웨이트

**분산의 첫자리**:

- 작은 첫 분산: $\sigma_{\text{init}} \approx 0.01$~$0.1$
- 학습 초가 붙박인 네트워크와 비슷해지게 한다

### 몬테카를로 표본의 수

**학습**: 흔히 표본 $S = 1$이면 넉넉하다(치우치지 않은 그래디언트)

**따지기**: 추론이 든든하도록 표본을 더 쓴다($S = 10$~$100$)

**트레이드오프**: 표본이 많을수록 → 근사는 좋아지고 값은 비싸진다

### 더 드는 계산

| 몫 | 여느 뉴럴 넷트웍 대비 값 |
|-----------|---------------------|
| 파라메터 | 2배(평균 + 분산) |
| 앞으로 스텝 | 약 1배(그 자리 다시 잡기를 쓰면) |
| 되돌아 스텝 | 약 1.5배 |
| 기억 | 2배 |
| 추론 | S배(표본 S개일 때) |

---

## 10. 변분 추론의 클래스

### 곱하는 정규화 흐름(MNF)

더 잘 드러내는 사후분포를 위한 도움 변수:

$$
q(\theta) = \int q(\theta \mid z) q(z) dz
$$

여기서 $q(\theta \mid z)$은 가우시안이고 $q(z)$은 정규화 흐름이다.

### 잡음 섞은 제 그래디언트(NNG)

더 잘 가장 좋게 하려고 잡음을 섞은 제 그래디언트를 쓴다.

$$
\theta_{t+1} = \theta_t - \alpha F^{-1} (\nabla \mathcal{L} + \epsilon)
$$

여기서 $F$은 피셔 소식 행렬이다.

### 살아 있는 변이 가우시안-뉴턴(VOGN)

제 그래디언트 변분 추론의 근사:

$$
\mu_{t+1} = \mu_t - \alpha \Sigma_t \nabla_\mu \mathcal{L}
$$

$$
\Sigma_{t+1}^{-1} = (1-\alpha)\Sigma_t^{-1} + \alpha \hat{F}
$$

### 함수 변분 추론

변이 분포를 함수 밭에 둔다.

$$
q(f) \approx p(f \mid \mathcal{D})
$$

**나은 점**:

- 사전분포를 풀이하기 쉽다
- 함수 밭에서 불확실성이 낫다
- 웨이트 공간의 뒤틀림을 비껴간다

---

## 11. 파이썬으로 짜기

```python
"""
변이 베이즈 뉴럴 넷트웍

이 배치는 뉴럴 넷트웍의 변이 미루어 봄을 온전히 짜 놓았다.
되돌아가며 베이즈, 그 자리 파라메터 다시 잡기, KL 천천히 올리기,
그리고 여러 뒷분포 어림이 들어 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from typing import Tuple, List, Optional, Dict, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# 변이 켜
# =============================================================================

class VariationalLayer(ABC):
    """변이 켜의 뼈대 클래스."""
    
    @abstractmethod
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """골라 뽑기를 곁들인 앞으로 스텝."""
        pass
    
    @abstractmethod
    def kl_divergence(self) -> float:
        """앞선 분포에 대한 KL 갈림을 셈한다."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """변이 파라메터를 얻는다."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, np.ndarray]):
        """변이 파라메터를 얹는다."""
        pass

class VariationalLinear(VariationalLayer):
    """
    가우스 짐을 지닌 변이 선형 켜.
    
    W_ij ~ N(mu_W_ij, sigma_W_ij^2)
    b_j ~ N(mu_b_j, sigma_b_j^2)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
        use_local_reparam: bool = True
    ):
        """
        Parameters
        ----------
        in_features : int
            들임 차수
        out_features : int
            날임 차수
        prior_sigma : float
            앞선 분포의 표준편차
        init_sigma : float
            뒷분포의 첫 표준편차
        use_local_reparam : bool
            그 자리 파라메터 다시 잡기 재주를 쓴다
        """
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.use_local_reparam = use_local_reparam
        
        # 변이 파라메터의 첫자리를 잡는다
        # 짐의 평균: 자비에 첫자리 잡기
        self.mu_W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        # 짐의 로그 흩어짐(rho으로 매긴다: sigma = softplus(rho))
        self.rho_W = np.full((in_features, out_features), np.log(np.exp(init_sigma) - 1))
        
        # 치우침의 평균과 로그 흩어짐
        self.mu_b = np.zeros(out_features)
        self.rho_b = np.full(out_features, np.log(np.exp(init_sigma) - 1))
        
        # 그래디언트 셈에 쓰려고 마지막에 뽑은 짐을 담는다
        self.last_eps_W = None
        self.last_eps_b = None
    
    @property
    def sigma_W(self) -> np.ndarray:
        """소프트플러스로 rho에서 sigma을 셈한다."""
        return np.log(1 + np.exp(self.rho_W))
    
    @property
    def sigma_b(self) -> np.ndarray:
        """소프트플러스로 rho에서 sigma을 셈한다."""
        return np.log(1 + np.exp(self.rho_b))
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """
        앞으로 스텝.
        
        Parameters
        ----------
        x : (batch_size, in_features) 꼴의 ndarray
            들임
        sample : bool
            True이면 짐을 뽑고, False이면 평균을 쓴다
        
        Returns
        -------
        (batch_size, out_features) 꼴의 ndarray
            날임
        """
        if not sample:
            # 붙박인 앞으로 스텝(평균을 쓴다)
            return x @ self.mu_W + self.mu_b
        
        if self.use_local_reparam:
            # 그 자리 파라메터 다시 잡기: 살림을 곧바로 뽑는다
            # a ~ N(x @ mu_W + mu_b, x^2 @ sigma_W^2 + sigma_b^2)
            
            mu_a = x @ self.mu_W + self.mu_b
            var_a = (x ** 2) @ (self.sigma_W ** 2) + self.sigma_b ** 2
            
            eps_a = np.random.randn(*mu_a.shape)
            return mu_a + np.sqrt(var_a + 1e-8) * eps_a
        
        else:
            # 여느 파라메터 다시 잡기: 짐을 뽑는다
            self.last_eps_W = np.random.randn(*self.mu_W.shape)
            self.last_eps_b = np.random.randn(*self.mu_b.shape)
            
            W = self.mu_W + self.sigma_W * self.last_eps_W
            b = self.mu_b + self.sigma_b * self.last_eps_b
            
            return x @ W + b
    
    def kl_divergence(self) -> float:
        """
        q(W)에서 앞선 분포 p(W)까지의 KL 갈림을 셈한다.
        
        KL(N(mu, sigma^2) || N(0, sigma_p^2)) = 
            0.5 * (mu^2/sigma_p^2 + sigma^2/sigma_p^2 - 1 - log(sigma^2/sigma_p^2))
        """
        prior_var = self.prior_sigma ** 2
        
        # 짐의 KL
        kl_W = 0.5 * np.sum(
            self.mu_W ** 2 / prior_var +
            self.sigma_W ** 2 / prior_var -
            1 -
            np.log(self.sigma_W ** 2 / prior_var + 1e-10)
        )
        
        # 치우침의 KL
        kl_b = 0.5 * np.sum(
            self.mu_b ** 2 / prior_var +
            self.sigma_b ** 2 / prior_var -
            1 -
            np.log(self.sigma_b ** 2 / prior_var + 1e-10)
        )
        
        return kl_W + kl_b
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """변이 파라메터를 얻는다."""
        return {
            'mu_W': self.mu_W.copy(),
            'rho_W': self.rho_W.copy(),
            'mu_b': self.mu_b.copy(),
            'rho_b': self.rho_b.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """변이 파라메터를 얹는다."""
        self.mu_W = params['mu_W'].copy()
        self.rho_W = params['rho_W'].copy()
        self.mu_b = params['mu_b'].copy()
        self.rho_b = params['rho_b'].copy()
    
    def n_params(self) -> int:
        """변이 파라메터의 수."""
        return 2 * (self.in_features * self.out_features + self.out_features)

# =============================================================================
# 변이 뉴럴 넷트웍
# =============================================================================

class VariationalMLP:
    """
    변이 여러 켜 퍼셉트론.
    
    평균 마당 가우스 뒷분포로 되돌아가며 베이즈를 짜 놓았다.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
        activation: str = 'relu',
        use_local_reparam: bool = True
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            [들임 차수, 숨은 켜1, ..., 날임 차수]
        prior_sigma : float
            앞선 분포의 표준편차
        init_sigma : float
            뒷분포의 첫 표준편차
        activation : str
            'relu' 또는 'tanh'
        use_local_reparam : bool
            그 자리 파라메터 다시 잡기를 쓴다
        """
        self.layer_sizes = layer_sizes
        self.prior_sigma = prior_sigma
        self.n_layers = len(layer_sizes) - 1
        
        # 살림 함수
        if activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
        elif activation == 'tanh':
            self.act_fn = np.tanh
        else:
            raise ValueError(f"모르는 살림 함수: {activation}")
        
        # 변이 켜를 만든다
        self.layers = []
        for i in range(self.n_layers):
            layer = VariationalLinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                prior_sigma=prior_sigma,
                init_sigma=init_sigma,
                use_local_reparam=use_local_reparam
            )
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """켜를 모두 지나는 앞으로 스텝."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, sample=sample)
            # 마지막 켜만 빼고 살림
            if i < self.n_layers - 1:
                h = self.act_fn(h)
        return h
    
    def kl_divergence(self) -> float:
        """켜 모두에 걸친 온 KL 갈림."""
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def predict(
        self,
        x: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        아리송함을 곁들여 미루어 본다.
        
        Returns
        -------
        mean : ndarray
            미루어 본 평균
        std : ndarray
            미루어 본 표준편차
        """
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x, sample=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
    
    def n_variational_params(self) -> int:
        """변이 파라메터의 온 수."""
        return sum(layer.n_params() for layer in self.layers)

# =============================================================================
# 익힘
# =============================================================================

class BayesByBackprop:
    """
    되돌아가며 베이즈 익힘 알고리즘.
    """
    
    def __init__(
        self,
        model: VariationalMLP,
        likelihood_sigma: float = 1.0,
        kl_weight: float = 1.0,
        lr: float = 0.001,
        lr_decay: float = 0.0
    ):
        """
        Parameters
        ----------
        model : VariationalMLP
            변이 뉴럴 넷트웍
        likelihood_sigma : float
            살핌 잡음의 표준편차
        kl_weight : float
            KL 항의 짐(천천히 올리기용)
        lr : float
            러닝 비율
        lr_decay : float
            판마다의 러닝 비율 줄이기
        """
        self.model = model
        self.likelihood_sigma = likelihood_sigma
        self.kl_weight = kl_weight
        self.lr = lr
        self.lr_decay = lr_decay
    
    def compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int,
        n_samples: int = 1
    ) -> Tuple[float, float, float]:
        """
        ELBO 잃음을 셈한다.
        
        Returns
        -------
        loss : float
            온 잃음(-ELBO)
        nll : float
            음수 로그 그럴듯함 항
        kl : float
            KL 갈림 항
        """
        batch_size = len(X)
        
        # 바라는 NLL의 몬테카를로 어림
        nll = 0.0
        for _ in range(n_samples):
            pred = self.model.forward(X, sample=True)
            # 가우스 NLL
            nll += 0.5 * np.sum((y - pred) ** 2) / (self.likelihood_sigma ** 2)
        nll /= n_samples
        
        # 온 자료 꾸러미에 맞게 잣대를 맞춘다
        nll *= (n_total / batch_size)
        
        # KL 갈림
        kl = self.model.kl_divergence()
        
        # 온 잃음
        loss = nll + self.kl_weight * kl
        
        return loss, nll, kl
    
    def compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int,
        eps: float = 1e-5
    ) -> List[Dict[str, np.ndarray]]:
        """
        그래디언트를 수로 셈한다(단순하게).
        
        참으로는 절로 미분하기를 쓴다.
        """
        gradients = []
        
        for layer in self.model.layers:
            params = layer.get_params()
            grads = {}
            
            for param_name, param_value in params.items():
                grad = np.zeros_like(param_value)
                
                for idx in np.ndindex(param_value.shape):
                    # 마디 있는 차를 셈한다
                    original = param_value[idx]
                    
                    param_value[idx] = original + eps
                    layer.set_params({**params, param_name: param_value})
                    loss_plus, _, _ = self.compute_loss(X, y, n_total)
                    
                    param_value[idx] = original - eps
                    layer.set_params({**params, param_name: param_value})
                    loss_minus, _, _ = self.compute_loss(X, y, n_total)
                    
                    param_value[idx] = original
                    layer.set_params(params)
                    
                    grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                
                grads[param_name] = grad
            
            gradients.append(grads)
        
        return gradients
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_total: int
    ) -> Tuple[float, float, float]:
        """
        익힘 한 스텝.
        
        Returns
        -------
        loss, nll, kl : float
            잃음의 몫
        """
        # 그래디언트를 셈한다
        gradients = self.compute_gradients(X, y, n_total)
        
        # 파라메터를 고친다
        for layer, grads in zip(self.model.layers, gradients):
            params = layer.get_params()
            for param_name in params:
                params[param_name] -= self.lr * grads[param_name]
            layer.set_params(params)
        
        # 마지막 잃음을 셈한다
        return self.compute_loss(X, y, n_total)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 100,
        batch_size: Optional[int] = None,
        kl_annealing: bool = False,
        annealing_epochs: int = 50,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        변이 BNN을 익힌다.
        
        Returns
        -------
        history : dict
            'loss', 'nll', 'kl'을 담은 익힘 자취
        """
        N = len(X)
        if batch_size is None:
            batch_size = min(N, 32)
        
        history = {'loss': [], 'nll': [], 'kl': []}
        
        for epoch in range(n_epochs):
            # KL 천천히 올리기
            if kl_annealing:
                self.kl_weight = min(1.0, epoch / annealing_epochs)
            
            # 러닝 비율 줄이기
            current_lr = self.lr * (1 - self.lr_decay) ** epoch
            
            # 자료를 뒤섞는다
            perm = np.random.permutation(N)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss, nll, kl = self.train_step(X_batch, y_batch, N)
                
                epoch_loss += loss
                epoch_nll += nll
                epoch_kl += kl
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_nll /= n_batches
            epoch_kl /= n_batches
            
            history['loss'].append(epoch_loss)
            history['nll'].append(epoch_nll)
            history['kl'].append(epoch_kl)
            
            if verbose and epoch % 10 == 0:
                print(f"{epoch}판: 잃음={epoch_loss:.4f}, "
                      f"NLL={epoch_nll:.4f}, KL={epoch_kl:.4f}")
        
        return history

# =============================================================================
# KL 천천히 올리는 짜임
# =============================================================================

def linear_annealing(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """곧게 KL을 올리는 짜임."""
    return min(1.0, epoch / warmup_epochs)

def sigmoid_annealing(epoch: int, total_epochs: int, midpoint: int, steepness: float = 0.1) -> float:
    """시그모이드로 KL을 올리는 짜임."""
    return 1.0 / (1.0 + np.exp(-steepness * (epoch - midpoint)))

def cyclical_annealing(epoch: int, cycle_length: int, ratio: float = 0.5) -> float:
    """돌림으로 KL을 올리는 짜임."""
    cycle_position = epoch % cycle_length
    rise_length = int(cycle_length * ratio)
    return min(1.0, cycle_position / rise_length)

# =============================================================================
# 잣대 섞기 앞선 분포
# =============================================================================

class ScaleMixturePrior:
    """
    가우스를 잣대로 섞은 앞선 분포.
    
    p(w) = pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2)
    """
    
    def __init__(
        self,
        pi: float = 0.5,
        sigma1: float = 1.0,
        sigma2: float = 0.1
    ):
        """
        Parameters
        ----------
        pi : float
            섞는 짐
        sigma1 : float
            첫째 몫의 표준편차
        sigma2 : float
            둘째 몫의 표준편차
        """
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
    
    def log_prob(self, w: np.ndarray) -> float:
        """로그 확률을 셈한다."""
        log_p1 = stats.norm.logpdf(w, 0, self.sigma1)
        log_p2 = stats.norm.logpdf(w, 0, self.sigma2)
        
        # 셈이 든든하도록 log-sum-exp
        log_mix = np.logaddexp(
            np.log(self.pi) + log_p1,
            np.log(1 - self.pi) + log_p2
        )
        
        return np.sum(log_mix)

class VariationalLinearMixturePrior(VariationalLayer):
    """
    잣대 섞기 앞선 분포를 지닌 변이 선형 켜.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior: ScaleMixturePrior,
        init_sigma: float = 0.1
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        
        # 변이 파라메터의 첫자리를 잡는다
        self.mu_W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.rho_W = np.full((in_features, out_features), np.log(np.exp(init_sigma) - 1))
        self.mu_b = np.zeros(out_features)
        self.rho_b = np.full(out_features, np.log(np.exp(init_sigma) - 1))
    
    @property
    def sigma_W(self) -> np.ndarray:
        return np.log(1 + np.exp(self.rho_W))
    
    @property
    def sigma_b(self) -> np.ndarray:
        return np.log(1 + np.exp(self.rho_b))
    
    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        if not sample:
            return x @ self.mu_W + self.mu_b
        
        eps_W = np.random.randn(*self.mu_W.shape)
        eps_b = np.random.randn(*self.mu_b.shape)
        
        W = self.mu_W + self.sigma_W * eps_W
        b = self.mu_b + self.sigma_b * eps_b
        
        return x @ W + b
    
    def kl_divergence(self, n_samples: int = 1) -> float:
        """
        KL 갈림의 몬테카를로 어림.
        
        KL(q||p) = E_q[log q - log p]
        """
        kl = 0.0
        
        for _ in range(n_samples):
            # 짐을 뽑는다
            eps_W = np.random.randn(*self.mu_W.shape)
            eps_b = np.random.randn(*self.mu_b.shape)
            
            W = self.mu_W + self.sigma_W * eps_W
            b = self.mu_b + self.sigma_b * eps_b
            
            # 로그 q(변이 뒷분포)
            log_q_W = np.sum(stats.norm.logpdf(W, self.mu_W, self.sigma_W))
            log_q_b = np.sum(stats.norm.logpdf(b, self.mu_b, self.sigma_b))
            
            # 로그 p(앞선 분포)
            log_p_W = self.prior.log_prob(W)
            log_p_b = self.prior.log_prob(b)
            
            kl += (log_q_W + log_q_b) - (log_p_W + log_p_b)
        
        return kl / n_samples
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            'mu_W': self.mu_W.copy(),
            'rho_W': self.rho_W.copy(),
            'mu_b': self.mu_b.copy(),
            'rho_b': self.rho_b.copy()
        }
    
    def set_params(self, params: Dict[str, np.ndarray]):
        self.mu_W = params['mu_W'].copy()
        self.rho_W = params['rho_W'].copy()
        self.mu_b = params['mu_b'].copy()
        self.rho_b = params['rho_b'].copy()

# =============================================================================
# 그리기
# =============================================================================

def plot_weight_distributions(
    model: VariationalMLP,
    layer_idx: int = 0,
    n_weights: int = 5
):
    """배운 짐 분포를 그린다."""
    
    layer = model.layers[layer_idx]
    
    fig, axes = plt.subplots(1, n_weights, figsize=(3*n_weights, 3))
    
    # 그리려고 짐을 편다
    mu_flat = layer.mu_W.flatten()
    sigma_flat = layer.sigma_W.flatten()
    
    # 짐을 아무렇게나 고른다
    indices = np.random.choice(len(mu_flat), n_weights, replace=False)
    
    x = np.linspace(-3, 3, 200)
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        mu = mu_flat[idx]
        sigma = sigma_flat[idx]
        
        # 뒷분포를 그린다
        posterior = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, posterior, 'b-', linewidth=2, label='뒷분포')
        
        # 앞선 분포를 그린다
        prior = stats.norm.pdf(x, 0, model.prior_sigma)
        ax.plot(x, prior, 'k--', linewidth=1, label='앞선 분포')
        
        ax.axvline(mu, color='red', linestyle=':', label=f'μ={mu:.2f}')
        ax.set_title(f'짐 {idx}\nσ={sigma:.3f}')
        ax.set_xlabel('짐 값')
        
        if i == 0:
            ax.legend()
    
    plt.suptitle(f'{layer_idx}번 켜의 짐 분포')
    plt.tight_layout()
    plt.show()

def plot_training_history(history: Dict[str, List[float]]):
    """익힘 굽이를 그린다."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 온 잃음
    axes[0].plot(history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('판')
    axes[0].set_ylabel('잃음')
    axes[0].set_title('온 잃음(-ELBO)')
    
    # NLL
    axes[1].plot(history['nll'], 'g-', linewidth=2)
    axes[1].set_xlabel('판')
    axes[1].set_ylabel('NLL')
    axes[1].set_title('음수 로그 그럴듯함')
    
    # KL
    axes[2].plot(history['kl'], 'r-', linewidth=2)
    axes[2].set_xlabel('판')
    axes[2].set_ylabel('KL')
    axes[2].set_title('KL 갈림')
    
    plt.tight_layout()
    plt.show()

def plot_predictions_with_uncertainty(
    model: VariationalMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    n_samples: int = 100
):
    """아리송함 띠를 곁들여 미루어 봄을 그린다."""
    
    mean, std = model.predict(X_test, n_samples=n_samples)
    
    plt.figure(figsize=(10, 6))
    
    # 아리송함 띠
    X_flat = X_test.flatten()
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    
    plt.fill_between(
        X_flat,
        mean_flat - 2*std_flat,
        mean_flat + 2*std_flat,
        alpha=0.3, color='blue', label='±2σ'
    )
    
    # 평균 미루어 봄
    plt.plot(X_flat, mean_flat, 'b-', linewidth=2, label='평균')
    
    # 참 함수
    if y_true is not None:
        plt.plot(X_flat, y_true.flatten(), 'k--', linewidth=1, label='참')
    
    # 익힘 자료
    plt.scatter(X_train.flatten(), y_train.flatten(), 
                c='red', s=30, zorder=5, label='자료')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('변이 BNN의 미루어 봄')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# 보여 주는 함수
# =============================================================================

def demo_variational_bnn():
    """변이 BNN 익히기를 보여 준다."""
    
    print("=" * 70)
    print("변이 베이즈 뉴럴 넷트웍")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 자료를 만든다
    N = 50
    X_train = np.random.uniform(-4, 4, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    y_true = np.sin(X_test)
    
    print(f"\n익힘 자료: {N}점")
    print(f"시험 자료: {len(X_test)}점")
    
    # 모형을 만든다
    model = VariationalMLP(
        layer_sizes=[1, 20, 20, 1],
        prior_sigma=1.0,
        init_sigma=0.1,
        activation='tanh',
        use_local_reparam=True
    )
    
    print(f"모형: {model.layer_sizes}")
    print(f"변이 파라메터: {model.n_variational_params()}")
    
    # 익힌다
    trainer = BayesByBackprop(
        model,
        likelihood_sigma=0.2,
        kl_weight=1.0,
        lr=0.01
    )
    
    print("\n익히는 중(좀 걸릴 수 있다)...")
    history = trainer.train(
        X_train, y_train,
        n_epochs=50,
        batch_size=N,
        kl_annealing=True,
        annealing_epochs=25,
        verbose=True
    )
    
    # 따진다
    mean, std = model.predict(X_test, n_samples=100)
    
    print(f"\n미루어 봄의 자:")
    print(f"  평균 표준편차(앎의): {np.mean(std):.4f}")
    print(f"  가장 큰 표준편차: {np.max(std):.4f}")
    
    return model, history

def demo_kl_annealing():
    """여러 KL 올리기 짜임을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("KL 천천히 올리는 짜임")
    print("=" * 70)
    
    n_epochs = 100
    epochs = np.arange(n_epochs)
    
    schedules = {
        '곧게 (몸풀기=30)': [linear_annealing(e, n_epochs, 30) for e in epochs],
        '시그모이드 (가운데=30)': [sigmoid_annealing(e, n_epochs, 30, 0.2) for e in epochs],
        '돌림 (돌림=40)': [cyclical_annealing(e, 40, 0.5) for e in epochs],
    }
    
    plt.figure(figsize=(10, 5))
    
    for name, values in schedules.items():
        plt.plot(epochs, values, linewidth=2, label=name)
    
    plt.xlabel('판')
    plt.ylabel('KL 짐 (β)')
    plt.title('KL 천천히 올리는 짜임')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nKL을 천천히 올리면 익힘 초에 뒷분포가 주저앉는 것을 막는다.")

def demo_local_reparameterization():
    """여느 파라메터 다시 잡기와 그 자리 다시 잡기를 견준다."""
    
    print("\n" + "=" * 70)
    print("그 자리 파라메터 다시 잡기 재주")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 켜 둘을 만든다
    layer_standard = VariationalLinear(10, 5, use_local_reparam=False)
    layer_local = VariationalLinear(10, 5, use_local_reparam=True)
    
    # 파라메터를 베낀다
    layer_local.mu_W = layer_standard.mu_W.copy()
    layer_local.rho_W = layer_standard.rho_W.copy()
    layer_local.mu_b = layer_standard.mu_b.copy()
    layer_local.rho_b = layer_standard.rho_b.copy()
    
    # 시험 들임
    x = np.random.randn(1, 10)
    
    # 앞으로 스텝 여러 번
    n_samples = 1000
    
    outputs_standard = np.array([layer_standard.forward(x).flatten() for _ in range(n_samples)])
    outputs_local = np.array([layer_local.forward(x).flatten() for _ in range(n_samples)])
    
    print("\n날임의 자(비슷해야 한다):")
    print(f"  여느 것 - 평균: {np.mean(outputs_standard, axis=0)[:3]}")
    print(f"  그 자리   - 평균: {np.mean(outputs_local, axis=0)[:3]}")
    print(f"  여느 것 - 표준편차:  {np.std(outputs_standard, axis=0)[:3]}")
    print(f"  그 자리   - 표준편차:  {np.std(outputs_local, axis=0)[:3]}")
    
    print("\n*** 그 자리 다시 잡기도 같은 분포를 준다")
    print("*** 다만 그래디언트 흩어짐이 더 작다")

def demo_uncertainty_quality():
    """변이 BNN의 아리송함 결을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("아리송함의 됨됨이")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 틈이 있는 자료를 만든다
    X_train = np.concatenate([
        np.random.uniform(-4, -1, 25),
        np.random.uniform(1, 4, 25)
    ]).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.15, (50, 1))
    
    X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
    
    print("자료에 [-1, 1] 자리의 틈이 있다")
    
    # 모형을 만들고 익힌다
    model = VariationalMLP(
        layer_sizes=[1, 30, 1],
        prior_sigma=1.0,
        init_sigma=0.05,
        activation='tanh'
    )
    
    trainer = BayesByBackprop(model, likelihood_sigma=0.15, lr=0.02)
    trainer.train(X_train, y_train, n_epochs=100, verbose=False)
    
    # 따진다
    mean, std = model.predict(X_test, n_samples=100)
    
    # 자리마다 아리송함을 살핀다
    in_gap = (X_test.flatten() > -1) & (X_test.flatten() < 1)
    near_data = ~in_gap & (np.abs(X_test.flatten()) < 4)
    extrapolation = np.abs(X_test.flatten()) > 4
    
    print(f"\n평균 아리송함(표준편차):")
    print(f"  틈 자리:      {np.mean(std[in_gap]):.4f}")
    print(f"  익힘 자료 가까이: {np.mean(std[near_data]):.4f}")
    print(f"  밖으로 늘림:      {np.mean(std[extrapolation]):.4f}")
    
    print("\n*** 아리송함은 틈과 밖으로 늘린 자리에서 더 커야 한다")

if __name__ == "__main__":
    model, history = demo_variational_bnn()
    demo_kl_annealing()
    demo_local_reparameterization()
    demo_uncertainty_quality()
```

**출력:**

```
======================================================================
변이 베이즈 뉴럴 넷트웍
======================================================================

익힘 자료: 50점
시험 자료: 200점
모형: [1, 20, 20, 1]
변이 파라메터: 962

익히는 중(좀 걸릴 수 있다)...
0판: 잃음=nan, NLL=inf, KL=nan
10판: 잃음=nan, NLL=nan, KL=nan
20판: 잃음=nan, NLL=nan, KL=nan
30판: 잃음=nan, NLL=nan, KL=nan
40판: 잃음=nan, NLL=nan, KL=nan

미루어 봄의 자:
  평균 표준편차(앎의): nan
  가장 큰 표준편차: nan

======================================================================
KL 천천히 올리는 짜임
======================================================================

KL을 천천히 올리면 익힘 초에 뒷분포가 주저앉는 것을 막는다.

======================================================================
그 자리 파라메터 다시 잡기 재주
======================================================================

날임의 자(비슷해야 한다):
  여느 것 - 평균: [-0.57822443  1.66707367 -0.85932269]
  그 자리   - 평균: [-0.57369844  1.64977453 -0.86525904]
  여느 것 - 표준편차:  [0.28735601 0.27301902 0.30143331]
  그 자리   - 표준편차:  [0.29015879 0.27977214 0.27965379]

*** 그 자리 다시 잡기도 같은 분포를 준다
*** 다만 그래디언트 흩어짐이 더 작다

======================================================================
아리송함의 됨됨이
======================================================================
자료에 [-1, 1] 자리의 틈이 있다

평균 아리송함(표준편차):
  틈 자리:      nan
  익힘 자료 가까이: nan
  밖으로 늘림:      nan

*** 아리송함은 틈과 밖으로 늘린 자리에서 더 커야 한다
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 두 켜 뉴럴 넷트웍에서, 이 마디에서 밝힌 방법에 따른 근사 사후분포의 꼴을 이끌어 내어라.

</div>

??? success "풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 근사 방법은 다룰 수 있는 꼴을 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 가운데로 삼고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 넌지시 세워진다. 근사마다 참 사후분포 모습의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 방법에서 얻은 불확실성 근사의 캘리브레이션을 MC 드롭아웃, 깊은 앙상블과 비교하는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

</div>

??? success "풀이"
    자: (1) 통 15개의 바라는 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음수 로그 우도(NLL), (4) 분포 외 알아내기의 AUROC. 그림: 방법마다 본 잦기를 예측한 신뢰도에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 데이터에서 캘리브레이션을, SVHN에서 분포 외 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 베이스라인으로 쓴다. 아무렇게나 하는 시드 5개에 걸친 평균과 표준편차를 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 넷트웍의 추론 분산이 인식적 불확실성과 우연적 불확실성으로 쪼개짐을 증명하여라. 학습 데이터 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

</div>

??? success "풀이"
    추론 분산은 온 분산 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 몫은 데이터를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 몫은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 우연적 불확실성만 남는다. 이 쪼갬은 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 마디의 불확실성 재기 방법을 거래 구조의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

</div>

??? success "풀이"
    판단 규칙: 자리 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측한 돌아옴, $\sigma_e^2$을 인식적 분산이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 위험 꺼림 값이다. 인식적 불확실성이 크면(낯선 시장 상황) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 인식적 불확실성 위끝을 두어 삼갈 수 있다. 이 틀은 모델의 신뢰도로 잣대를 잡은 켈리 잣대 특성의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

### 핵심 깨침

**변분 추론**은 다룰 수 없는 사후분포를 추정한다.

$$
p(\theta \mid \mathcal{D}) \approx q_\phi(\theta)
$$

**ELBO**(모델 증거 아래끝):

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi \| p)
$$

**파라메터 다시 잡기 재주**:

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### 평균장 가우시안

| 몫 | 식 |
|-----------|---------|
| **사후분포** | $q(\theta) = \prod_j \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)$ |
| **파라메터** | $\phi = \{\mu, \rho\}$이고 $\sigma = \text{softplus}(\rho)$ |
| **사전분포에 대한 KL** | $\frac{1}{2}\sum_j \left[\frac{\mu_j^2 + \sigma_j^2}{\sigma_p^2} - 1 - \log\frac{\sigma_j^2}{\sigma_p^2}\right]$ |

### 되돌아가며 베이즈 알고리즘

1. Sample $\epsilon \sim \mathcal{N}(0, I)$
2. $\theta = \mu + \text{softplus}(\rho) \odot \epsilon$을 계산한다
3. 로스를 계산한다: $\mathcal{L} = \text{NLL}(\theta) + \beta \cdot \text{KL}(q \| p)$
4. 되돌아가며 $\mu, \rho$을 고친다

### 참으로 헤아릴 것

| 특성 | 즐겨 쓸 길 |
|--------|----------------|
| **사전분포** | $\sigma_p = 1.0$(따짐 꾸러미로 맞춘다) |
| **첫 분산** | $\sigma_{\text{init}} = 0.01$~$0.1$ |
| **MC 표본(학습)** | 1(치우치지 않은 그래디언트) |
| **MC 표본(시험)** | 30~100 |
| **KL 천천히 올리기** | 20~50판에 걸쳐 곧게 워밍업 |
| **러닝레이트** | 0.001~0.01 |

### 평균장을 넘어

| 방법 | 리프리젠테이션 | 값 |
|--------|---------------|------|
| 평균장 | 낮음 | $O(d)$ |
| 낮은 자리 | 가운데 | $O(dr)$ |
| 행렬 변수 | 가운데 | $O(m^2 + n^2)$ |
| 정규화 흐름 | 높음 | $O(dK)$ |

### 나은 점과 한계

| 나은 점 | 한계 |
|------------|-------------|
| 큰 네트워크로 늘릴 수 있다 | 사후분포가 근사다 |
| 이치에 닿는 불확실성 | 불확실성을 낮게 볼 수 있다 |
| 되돌아가기와 함께 쓴다 | KL을 조심스레 다뤄야 한다 |
| 사전분포를 너그럽게 정한다 | 평균장은 얽힘을 놓친다 |

### 다른 장과의 이어짐

| 이야기 | 장 | 이어짐 |
|-------|---------|------------|
| 사전분포 정하기 | 43장: 웨이트의 사전분포 | KL 항 속의 사전분포 |
| 불확실성 | 43장: 불확실성 | 사후분포가 쪼갬을 이룬다 |
| MC 드롭아웃 | 43장: MC 드롭아웃 | 넌지시 하는 VI 갈음 |
| 사후분포 추론 | 43장: 사후분포 추론 | 추론 방법으로서의 VI |
| 모델 비교 | 43장: 모델 모델 증거 | ELBO가 모델 증거를 마디 짓는다 |

### 핵심 살펴볼 거리

- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
- Kingma, D. P., et al. (2015). Variational dropout and the local reparameterization trick. *NeurIPS*.
- Louizos, C., & Welling, M. (2017). Multiplicative normalizing flows for variational Bayesian neural networks. *ICML*.
- Zhang, G., et al. (2018). Noisy natural gradient as variational inference. *ICML*.
- Osawa, K., et al. (2019). Practical deep learning with Bayesian principles. *NeurIPS*.
