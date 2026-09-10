# 신경 그물 짐의 앞선 분포
신경 그물 짐에 대한 **앞선 분포**은 자료를 보기 앞서 그럴듯한 매개변수 값에 대한 믿음을 담는다. 베이즈 신경 그물에서 앞선 분포를 어떻게 고르느냐는 이끌려 나오는 함수 밭과 아리송함 어림에 깊이 걸린다. 이 장은 뒷분포 미루어 봄이 잘 되도록 앞선 분포를 이치에 닿게 정하는 길을 살핀다.

---

## 1. 왜 하는가: 앞선 분포가 걸리는 까닭

### 베이즈 신경 그물에서 앞선 분포가 하는 몫

여느 신경 그물에서 짐은 가장 좋게 하여 찾은 점 어림이다.

$$
\hat{\theta} = \arg\max_\theta \log p(\mathcal{D} \mid \theta)
$$

베이즈 신경 그물에서는 분포를 지닌다.

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

앞선 분포 $p(\theta)$은 다음을 정한다.

1. **정칙화의 셈**: 짐의 크기를 옭아맨다
2. **함수 밭의 결**: 매끄러움, 되풀이, 길이 잣대
3. **앎의 아리송함**: 자료에서 멀어질 때 아리송함이 어떻게 되는지
4. **뒷분포의 꼴**: 미루어 봄을 다룰 수 있는지에 걸린다

### 신경 그물에서 남다른 어려움

**높은 차수**: 요즘 그물은 매개변수가 수백만 개다

- 단순히 남남인 앞선 분포는 얽힌 매임을 담지 못할 수 있다
- 같은 켜의 짐끼리 얽힘이 종요로운 일이 잦다

**가려낼 수 없음**: 짐의 차림이 달라도 같은 함수가 나온다

- 자리 바꾸기 대칭: 숨은 낱자리를 맞바꿈
- 잣대 대칭: 켜 사이에서 짐의 잣대를 다시 잡음

**매개변수 넘침**: 자료보다 매개변수가 많음

- 뒷분포에서 앞선 분포가 크게 힘을 쓴다
- 함수가 이치에 닿게 움직이도록 이끄는 앞선 분포가 있어야 한다

### 짐 밭 대 함수 밭

**고갱이 깨침**: 우리가 마음 쓰는 것은 짐이 아니라 함수다.

$$
\text{Prior on weights } p(\theta) \implies \text{Prior on functions } p(f)
$$

짐에 얹은 단순한 앞선 분포도 함수 밭에서는 얽힌 움직임을 이끌 수 있다.

$$
f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x)))
$$

$p(f)$이 바라는 결을 지니도록 $p(\theta)$을 정하는 것이 어려운 대목이다.

---

## 2. 여느 가우스 앞선 분포

### 남남인 가우스 앞선 분포

가장 흔히 고르는 것은 남남인 가우스다.

$$
\boxed{p(\theta) = \prod_{l=1}^L \prod_{i,j} \mathcal{N}(w_{ij}^{(l)} \mid 0, \sigma_l^2)}
$$

행렬 꼴로 적으면 **같은 말이다**.

$$
p(W^{(l)}) = \mathcal{N}(\text{vec}(W^{(l)}) \mid 0, \sigma_l^2 I)
$$

### L2 정칙화와의 이어짐

가우스 앞선 분포를 쓴 MAP 어림은 L2으로 정칙화한 MLE와 같다.

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \log p(\mathcal{D} \mid \theta) - \frac{1}{2\sigma^2} \|\theta\|^2 \right]
$$

**정칙화의 셈**: $\lambda = 1/(2\sigma^2)$

| 앞선 분포 흩어짐 $\sigma^2$ | L2 벌 $\lambda$ | 미침 |
|---------------------------|---------------------|--------|
| 큼 | 작음 | 여린 정칙화 |
| 작음 | 큼 | 센 정칙화 |

### 켜마다의 흩어짐 잣대

켜마다 다른 앞선 분포 흩어짐이 있어야 할 수 있다.

$$
p(W^{(l)}) = \mathcal{N}(0, \sigma_l^2 I)
$$

**흔히 고르는 것**:

**1. 붙박인 흩어짐**:

$$
\sigma_l^2 = \sigma^2 \quad \forall l
$$

**2. 들임 갈래 잣대**(자비에/글로로 결):

$$
\sigma_l^2 = \frac{1}{n_{l-1}}
$$

**3. 날임 갈래 잣대**:

$$
\sigma_l^2 = \frac{1}{n_l}
$$

**4. 갈래 평균 잣대**:

$$
\sigma_l^2 = \frac{2}{n_{l-1} + n_l}
$$

여기서 $n_l$은 켜 $l$의 너비다.

### 살림 함수에 맞춘 흩어짐 잣대

앞선 분포의 흩어짐은 살림 함수를 헤아려야 한다.

**ReLU 살림**:

$$
\sigma_l^2 = \frac{2}{n_{l-1}}
$$

**Tanh/시그모이드 살림**:

$$
\sigma_l^2 = \frac{1}{n_{l-1}}
$$

**까닭**: 첫자리에서 켜에 걸쳐 살림의 흩어짐을 든든하게 지킨다.

---

## 3. 함수 밭의 결을 이끄는 앞선 분포

### 닐(1996)의 앞선 분포

닐은 어떤 앞선 분포를 지닌 끝없이 너른 신경 그물이 가우스 흐름으로 모임을 밝혔다.

**차림**: 숨은 낱자리가 $H$개인 숨은 켜 하나짜리 그물:

$$
f(x) = \sum_{h=1}^H v_h \, \sigma(w_h^\top x + b_h)
$$

**앞선 분포**:

$$
v_h \sim \mathcal{N}(0, \sigma_v^2/H), \quad w_h \sim \mathcal{N}(0, \sigma_w^2 I), \quad b_h \sim \mathcal{N}(0, \sigma_b^2)
$$

**결과**: $H \to \infty$일 때 $f(x) \to \mathcal{GP}(0, k(x, x'))$이고 여기서

$$
k(x, x') = \sigma_v^2 \, \mathbb{E}_{w, b}[\sigma(w^\top x + b) \, \sigma(w^\top x' + b)]
$$

### 신경 그물 가우스 흐름(NNGP) 알갱이

ReLU 살림에서는 알갱이가 닫힌 꼴을 지닌다.

$$
k(x, x') = \frac{\sigma_v^2}{\pi} \|x\| \|x'\| \left( \sin\phi + (\pi - \phi)\cos\phi \right)
$$

여기서 $\phi = \cos^{-1}\left(\frac{x^\top x'}{\|x\| \|x'\|}\right)$이다.

**뜻하는 바**:

- 짐의 앞선 분포가 함수의 매끄러움을 정한다
- 깊은 그물은 겹쳐 쌓은 알갱이를 이끈다
- 이치에 닿는 첫자리 잡기 길잡이를 준다

### 깊이와 앞선 분포의 흩어짐

깊은 그물에서는 흩어짐의 잣대를 조심스레 맞춰야 한다.

**잣대를 맞추지 않으면**: 흩어짐이 터지거나 사라진다

$$
\text{Var}[f(x)] \propto \prod_{l=1}^L \text{Var}[W^{(l)}]
$$

**잣대를 제대로 맞추면**: 흩어짐이 $O(1)$으로 남는다

$$
\sigma_l^2 = \frac{c}{n_{l-1}}
$$

여기서 $c$은 살림 함수에 매인다.

---

## 4. 성김을 이끄는 앞선 분포

### 성기게 하는 까닭

**성긴 그물의 나은 점**:

1. 지나치게 맞추기가 준다
2. 풀이하기가 나아진다
3. 셈이 잘 든다
4. 두루 미침이 나아진다

### 라플라스 앞선 분포(L1 정칙화)

$$
p(w) = \frac{\lambda}{2} \exp(-\lambda |w|)
$$

**결**:

- 봉우리가 0에 있다(MAP에서 꼭 0이 되게 이끈다)
- 가우스보다 꼬리가 두껍다
- MAP은 L1 정칙화(라소)와 같다

**한계**: 짝이 맞지 않아 미루어 봄이 얽힌다.

### 못과 널 앞선 분포

0에 놓인 점 무게와 이어지는 분포를 섞은 것이다.

$$
\boxed{p(w) = \pi \, \delta_0(w) + (1-\pi) \, \mathcal{N}(w \mid 0, \sigma^2)}
$$

**매개변수**:

- $\pi$: 짐이 꼭 0일 앞선 낌새
- $\sigma^2$: 0이 아닌 짐의 흩어짐

**미루어 봄**: 들고 남을 알리는 두 값 표시를 뽑아야 한다.

### 이어지게 풀어 쓰기

**말굽 앞선 분포**:

$$
w \mid \lambda \sim \mathcal{N}(0, \lambda^2), \quad \lambda \sim \text{Half-Cauchy}(0, \tau)
$$

**결**:

- 이어진다(점 무게가 없다)
- 두꺼운 꼬리가 큰 짐을 받아 준다
- 0 쪽으로 세게 오그린다
- 두루와 그 자리의 얼개: $\tau$은 두루, $\lambda$은 그 자리

**다독인 말굽**:

$$
w \mid \lambda, c \sim \mathcal{N}(0, \tilde{\lambda}^2), \quad \tilde{\lambda}^2 = \frac{c^2 \lambda^2}{c^2 + \lambda^2}
$$

가장 큰 흩어짐을 $c^2$으로 마디 짓는다.

### 걸림새 절로 가려내기(ARD)

들임마다 또는 결마다의 촘촘함:

$$
p(w_j \mid \alpha_j) = \mathcal{N}(w_j \mid 0, \alpha_j^{-1})
$$

$$
p(\alpha_j) = \text{Gamma}(\alpha_j \mid a_0, b_0)
$$

**미침**: $\alpha_j$이 큰 결은 사실상 쳐내진다.

---

## 5. 층진 앞선 분포

### 왜 하는가

**문제**: 앞선 분포의 하이퍼파라미터($\sigma^2$ 따위)를 고르기가 어렵다.

**풀이**: 하이퍼파라미터에 다시 앞선 분포를 얹고 자료가 정하게 한다.

### 두 층의 층짜임

$$
p(\theta \mid \sigma^2) = \mathcal{N}(\theta \mid 0, \sigma^2 I)
$$

$$
p(\sigma^2) = \text{Inv-Gamma}(\sigma^2 \mid \alpha_0, \beta_0)
$$

**가장자리 앞선 분포**($\sigma^2$을 적분해 없앤 뒤):

$$
p(\theta) = \int p(\theta \mid \sigma^2) \, p(\sigma^2) \, d\sigma^2 = \text{Student-}t(\theta \mid 0, \frac{\beta_0}{\alpha_0}, 2\alpha_0)
$$

**결**:

- 가우스보다 꼬리가 두껍다
- 튀는 값에 더 든든하다
- 자료에 맞추어 가는 정칙화

### 켜마다의 층진 앞선 분포

켜마다 다른 흩어짐:

$$
p(W^{(l)} \mid \sigma_l^2) = \mathcal{N}(0, \sigma_l^2 I)
$$

$$
p(\sigma_l^2) = \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

**나은 점**:

- 켜마다 알맞은 정칙화를 배운다
- 켜마다 다른 얽힘에 맞춘다
- 첫자리 잡기에 덜 예민해진다

### 무리마다의 앞선 분포

**신경 낱자리마다의 흩어짐**:

$$
p(w_{:j}^{(l)} \mid \sigma_{lj}^2) = \mathcal{N}(0, \sigma_{lj}^2 I)
$$

**거르개마다의 흩어짐**(CNN에서):

$$
p(W_k^{(l)} \mid \sigma_{lk}^2) = \mathcal{N}(0, \sigma_{lk}^2 I)
$$

이는 신경 낱자리나 거르개를 통째로 절로 쳐내게 한다.

---

## 6. 잣대 섞기와 꼬리 두꺼운 앞선 분포

### 가우스 잣대 섞기

쓸모 있는 앞선 분포는 흔히 이렇게 적을 수 있다.

$$
p(w) = \int \mathcal{N}(w \mid 0, \sigma^2) \, p(\sigma^2) \, d\sigma^2
$$

| 섞는 분포 $p(\sigma^2)$ | 나오는 $p(w)$ |
|-----------------------------------|------------------|
| 지수 | 라플라스 |
| 거꿀 감마 | 스튜던트 $t$ |
| $\sigma$의 반 코시 | 말굽 |
| 베르누이 잣대 | 못과 널 |

### 스튜던트 t 앞선 분포

$$
p(w) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi\sigma^2}} \left(1 + \frac{w^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}
$$

**자유도** $\nu$이 꼬리의 두께를 다스린다.

- $\nu = 1$: 코시(꼬리가 아주 두껍다)
- $\nu \to \infty$: 가우스에 다가간다
- $\nu = 3$~$7$: 알맞은 절충

**잣대 섞기로 보면**:

$$
w \mid \tau \sim \mathcal{N}(0, \tau), \quad \tau \sim \text{Inv-Gamma}(\nu/2, \nu\sigma^2/2)
$$

### 두꺼운 꼬리의 나은 점

1. **든든함**: 앞선 분포를 잘못 정해도 덜 흔들린다
2. **큰 짐**: 자료가 받쳐 주면 이따금 큰 값을 받아 준다
3. **절로 가려냄**: 두꺼운 꼬리와 뾰족한 가운데가 어울려 부드러운 성김을 이룬다

---

## 7. 얽힌 앞선 분포와 얼개를 지닌 앞선 분포

### 남남임을 넘어

남남인 앞선 분포는 얼개를 놓친다.

- 같은 낱자리에 이어지는 짐끼리 얽힐 수 있다
- 엮음 거르개의 자리 얼개
- 되돌이 그물의 때 얼개

### 행렬 변수 가우스

짐 행렬 $W \in \mathbb{R}^{m \times n}$에서

$$
p(W) = \mathcal{MN}(W \mid M, U, V)
$$

where:

- $M$은 평균 행렬
- $U \in \mathbb{R}^{m \times m}$은 줄끼리의 얽힘을 담는다
- $V \in \mathbb{R}^{n \times n}$은 기둥끼리의 얽힘을 담는다

**같은 말로**:

$$
\text{vec}(W) \sim \mathcal{N}(\text{vec}(M), V \otimes U)
$$

### 낮은 자리 앞선 분포

짐 행렬이 거의 낮은 자리가 되도록 이끈다.

$$
W = UV^\top + E
$$

여기서 $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $r \ll \min(m,n)$이다.

**앞선 분포**:

$$
p(U) = \mathcal{N}(0, I), \quad p(V) = \mathcal{N}(0, I), \quad p(E) = \mathcal{N}(0, \sigma_E^2 I)
$$

**나은 점**:

- 참으로 쓰이는 매개변수 수를 줄인다
- 눌러 담기를 이끈다
- 두루 미침을 낫게 할 수 있다

### 엮음 얼개

엮음 켜에서는 앞선 분포가 자리 얼개를 지킬 수 있다.

**그 자리의 매끄러움**: 가까운 거르개 짐끼리 비슷해야 한다

$$
p(W) \propto \exp\left(-\frac{1}{2\sigma^2} \sum_{i,j} (w_{ij} - w_{i+1,j})^2 + (w_{ij} - w_{i,j+1})^2 \right)
$$

**옮김에 따라 함께 움직임**: CNN 얼개에 이미 들어 있다.

---

## 8. 겪어 본 앞선 분포와 자료에 매인 앞선 분포

### 겪어 본 베이즈

자료로 앞선 분포의 하이퍼파라미터를 어림한다.

$$
\hat{\eta} = \arg\max_\eta \log p(\mathcal{D} \mid \eta) = \arg\max_\eta \log \int p(\mathcal{D} \mid \theta) \, p(\theta \mid \eta) \, d\theta
$$

**둘째 갈래 가장 큰 그럴듯함**: 가장자리 그럴듯함을 가장 크게 한다.

**좋은 점**:

- 자료에 맞추어 간다
- 뒷분포의 맞음을 높일 수 있다

**아쉬운 점**:

- 자료를 두 번 쓴다(앞선 분포와 뒷분포에)
- 아리송함을 낮게 볼 수 있다
- 셈이 비싸다

### 옮겨 배우기 앞선 분포

가까운 일에서 얻은 뒷분포를 앞선 분포로 쓴다.

$$
p(\theta) \approx p(\theta \mid \mathcal{D}_{\text{source}})
$$

**길**:

**1. 평균 마당 어림**:

$$
p(\theta) = \mathcal{N}(\theta \mid \mu_{\text{source}}, \sigma_{\text{source}}^2 I)
$$

**2. 미리 익힌 모형 섞기**:

$$
p(\theta) = \sum_{k} \pi_k \, p(\theta \mid \mathcal{D}_k)
$$

**3. 가운데 맞춘 앞선 분포**(L2-SP):

$$
p(\theta) = \mathcal{N}(\theta \mid \theta_{\text{pretrained}}, \sigma^2 I)
$$

### 함수 앞선 분포

함수 밭에서 앞선 분포를 곧바로 정한다.

$$
p(f) = \mathcal{GP}(m(x), k(x, x'))
$$

그러고 나서 이 $p(f)$을 거의 이끌어 내는 $p(\theta)$을 찾는다.

**어려움**:

- 마디 있는 그물에서는 $p(\theta) \to p(f)$의 닫힌 꼴이 없다
- 표본 뽑기에 기댄 길이 있어야 한다
- 한창인 연구 밭이다

---

## 9. 참으로 헤아릴 것

### 앞선 분포의 흩어짐 고르기

**어림 규칙**: 다음이 되도록 첫자리를 잡는다.

- 살림 앞 값의 흩어짐이 $\approx 1$
- 기울기의 흩어짐이 $\approx 1$

**온통 이은 켜에서는**:

$$
\sigma_W^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \quad \sigma_b^2 = 0.01
$$

알갱이 크기가 $k \times k$이고 갈래가 $c$개인 **엮음 켜에서는**:

$$
\sigma_W^2 = \frac{2}{k^2 \cdot c_{\text{in}}}
$$

### 치우침의 앞선 분포

**흔한 길**:

**1. 0을 가운데로 삼는 가우스**:

$$
p(b) = \mathcal{N}(0, \sigma_b^2)
$$

작은 $\sigma_b^2$을 쓴다($0.01$~$0.1$ 따위).

**2. 0으로 붙박기**:

$$
p(b) = \delta_0(b)
$$

매개변수가 준다. 때로는 잘 듣는다.

**3. 자료에 매인 첫자리 잡기**:
자료의 자를 보고 치우침의 가운데를 잡는다.

### 켜의 갈래마다 다루기

| 켜 갈래 | 앞선 분포에서 헤아릴 것 |
|------------|---------------------|
| **빽빽한 켜** | 여느 가우스, 들임/날임 갈래 잣대 |
| **엮음** | 거르개마다 또는 갈래마다의 흩어짐 |
| **묶음 잣대 잡기** | 흔히 붙박이. 잣대/옮김은 배울 수 있다 |
| **박아 넣기** | 박아 넣기마다 또는 묶인 흩어짐 |
| **눈길** | 열쇠-물음-값에 다른 잣대가 있어야 할 수 있다 |

### 앞선 분포로 미리 살펴보기

뽑아 보고 이끌려 나온 함수를 살펴 **앞선 분포를 따진다**.

```python
# 앞선 분포에서 짐을 뽑는다
theta_prior = sample_from_prior()

# 시험 들임에 대해 미루어 본다
f_prior = network(x_test, theta_prior)

# 살핌: 이 함수들이 이치에 닿는가?
```

**살펴볼 것**:

- 날임의 크기: 미루어 봄이 이치에 닿는 자리에 있는가?
- 매끄러움: 함수가 너무 꿈틀대거나 너무 납작한가?
- 밖으로 늘리기: 익힘 자리에서 멀어지면 어떻게 되는가?

---

## 10. 파이썬으로 짜기

```python
"""
신경 그물 짐의 앞선 분포

이 묶음은 베이즈 신경 그물에 쓰는 여러 앞선 분포를 짜 놓았다.
여느 가우스, 성김을 이끄는 앞선 분포, 층진 앞선 분포, 그리고 앞선 분포로
미리 살펴보는 데 쓰는 잔손질 함수가 들어 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, gamma
from typing import Tuple, List, Optional, Dict, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# 밑 앞선 분포 갈래
# =============================================================================

class Prior(ABC):
    """짐 앞선 분포의 뼈대 갈래."""
    
    @abstractmethod
    def log_prob(self, w: np.ndarray) -> float:
        """짐의 로그 낌새를 셈한다."""
        pass
    
    @abstractmethod
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 짐을 뽑는다."""
        pass
    
    def prob(self, w: np.ndarray) -> float:
        """낌새를 셈한다(밑으로 넘칠 수 있다)."""
        return np.exp(self.log_prob(w))

class GaussianPrior(Prior):
    """
    고루 퍼진 가우스 앞선 분포: w ~ N(0, sigma^2 I)
    """
    
    def __init__(self, sigma: float = 1.0, mean: float = 0.0):
        """
        Parameters
        ----------
        sigma : float
            앞선 분포의 잣대 어긋남
        mean : float
            앞선 분포의 평균(기본값 0)
        """
        self.sigma = sigma
        self.mean = mean
        self.var = sigma ** 2
    
    def log_prob(self, w: np.ndarray) -> float:
        """log p(w)을 셈한다."""
        w = np.asarray(w)
        n = w.size
        
        return (
            -0.5 * n * np.log(2 * np.pi * self.var)
            - 0.5 * np.sum((w - self.mean) ** 2) / self.var
        )
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 뽑는다."""
        return np.random.normal(self.mean, self.sigma, shape)
    
    def __repr__(self):
        return f"GaussianPrior(mean={self.mean}, sigma={self.sigma})"

class LaplacePrior(Prior):
    """
    라플라스 앞선 분포: p(w) = (lambda/2) * exp(-lambda * |w|)
    
    MAP 어림에서 L1 정칙화와 같다.
    """
    
    def __init__(self, scale: float = 1.0):
        """
        Parameters
        ----------
        scale : float
            잣대 매개변수(1/lambda)
        """
        self.scale = scale
        self.rate = 1.0 / scale
    
    def log_prob(self, w: np.ndarray) -> float:
        """log p(w)을 셈한다."""
        w = np.asarray(w)
        n = w.size
        
        return n * np.log(self.rate / 2) - self.rate * np.sum(np.abs(w))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 뽑는다."""
        return np.random.laplace(0, self.scale, shape)
    
    def __repr__(self):
        return f"LaplacePrior(scale={self.scale})"

class StudentTPrior(Prior):
    """
    자유도를 정해 주는 스튜던트 t 앞선 분포.
    
    가우스보다 꼬리가 두껍고 튀는 값에 든든하다.
    """
    
    def __init__(self, df: float = 3.0, scale: float = 1.0):
        """
        Parameters
        ----------
        df : float
            자유도(nu)
        scale : float
            잣대 매개변수
        """
        self.df = df
        self.scale = scale
    
    def log_prob(self, w: np.ndarray) -> float:
        """log p(w)을 셈한다."""
        w = np.asarray(w)
        return np.sum(stats.t.logpdf(w, df=self.df, scale=self.scale))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 뽑는다."""
        return stats.t.rvs(df=self.df, scale=self.scale, size=shape)
    
    def __repr__(self):
        return f"StudentTPrior(df={self.df}, scale={self.scale})"

# =============================================================================
# 성김을 이끄는 앞선 분포
# =============================================================================

class SpikeAndSlabPrior(Prior):
    """
    못과 널 앞선 분포: p(w) = pi * delta_0 + (1-pi) * N(0, sigma^2)
    
    기울기에 기댄 미루어 봄에 맞게 이어지도록 풀어 쓴 것이다.
    """
    
    def __init__(
        self,
        pi: float = 0.5,
        sigma_slab: float = 1.0,
        sigma_spike: float = 0.01
    ):
        """
        Parameters
        ----------
        pi : float
            못(0 언저리)에 들 앞선 낌새
        sigma_slab : float
            널 몫의 잣대 어긋남
        sigma_spike : float
            못 몫의 잣대 어긋남(작다)
        """
        self.pi = pi
        self.sigma_slab = sigma_slab
        self.sigma_spike = sigma_spike
    
    def log_prob(self, w: np.ndarray) -> float:
        """log-sum-exp으로 log p(w)을 셈한다."""
        w = np.asarray(w)
        
        # 몫마다의 로그 낌새
        log_spike = (
            np.log(self.pi)
            + stats.norm.logpdf(w, 0, self.sigma_spike)
        )
        log_slab = (
            np.log(1 - self.pi)
            + stats.norm.logpdf(w, 0, self.sigma_slab)
        )
        
        # 셈이 든든하도록 log-sum-exp
        log_probs = np.logaddexp(log_spike, log_slab)
        
        return np.sum(log_probs)
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 뽑는다."""
        # 어느 몫인지 표시를 뽑는다
        is_spike = np.random.random(shape) < self.pi
        
        # 알맞은 몫에서 뽑는다
        samples = np.where(
            is_spike,
            np.random.normal(0, self.sigma_spike, shape),
            np.random.normal(0, self.sigma_slab, shape)
        )
        
        return samples
    
    def __repr__(self):
        return f"SpikeAndSlabPrior(pi={self.pi}, sigma_spike={self.sigma_spike}, sigma_slab={self.sigma_slab})"

class HorseshoePrior(Prior):
    """
    말굽 앞선 분포: w | lambda ~ N(0, lambda^2), lambda ~ Half-Cauchy(0, tau)
    
    꼬리는 두껍고 0 쪽으로 세게 오그린다.
    """
    
    def __init__(self, tau: float = 1.0):
        """
        Parameters
        ----------
        tau : float
            두루 쓰는 잣대 매개변수
        """
        self.tau = tau
    
    def log_prob(self, w: np.ndarray) -> float:
        """
        로그 낌새의 어림(lambda을 적분해 없애기는 다룰 수 없다).
        어림을 쓴다: |w| >> 0에서 p(w) ≈ log(1 + 2*tau^2/w^2)
        """
        w = np.asarray(w)
        # 작은 엡실론을 더해 log(0)을 막는다
        eps = 1e-10
        
        # 가장자리 말굽 밀도의 어림
        log_prob = np.sum(np.log(np.log(1 + 2 * self.tau**2 / (w**2 + eps))))
        
        return log_prob
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """앞선 분포에서 뽑는다."""
        # 반 코시에서 그 자리 잣대를 뽑는다
        lambdas = np.abs(stats.cauchy.rvs(size=shape)) * self.tau
        
        # 짐을 뽑는다
        return np.random.normal(0, lambdas)
    
    def __repr__(self):
        return f"HorseshoePrior(tau={self.tau})"

# =============================================================================
# 층진 앞선 분포
# =============================================================================

class HierarchicalGaussianPrior(Prior):
    """
    흩어짐에 거꿀 감마를 얹은 층진 가우스 앞선 분포.
    
    w | sigma^2 ~ N(0, sigma^2)
    sigma^2 ~ Inv-Gamma(alpha, beta)
    
    적분해 없애면 스튜던트 t이 된다.
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            거꿀 감마의 꼴 매개변수
        beta : float
            거꿀 감마의 잣대 매개변수
        """
        self.alpha = alpha
        self.beta = beta
        
        # 가장자리 분포는 자유도 2*alpha인 스튜던트 t
        self.marginal_df = 2 * alpha
        self.marginal_scale = np.sqrt(beta / alpha)
    
    def log_prob(self, w: np.ndarray) -> float:
        """가장자리 분포(스튜던트 t)의 로그 낌새를 셈한다."""
        w = np.asarray(w)
        return np.sum(stats.t.logpdf(
            w, df=self.marginal_df, scale=self.marginal_scale
        ))
    
    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """가장자리 분포(스튜던트 t)에서 뽑는다."""
        return stats.t.rvs(
            df=self.marginal_df,
            scale=self.marginal_scale,
            size=shape
        )
    
    def sample_conditional(
        self,
        shape: Tuple[int, ...],
        return_variance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        층지게 뽑는다: 먼저 흩어짐, 그다음 짐.
        """
        # 거꿀 감마에서 흩어짐을 뽑는다
        sigma_sq = stats.invgamma.rvs(self.alpha, scale=self.beta)
        
        # 흩어짐이 주어졌을 때 짐을 뽑는다
        w = np.random.normal(0, np.sqrt(sigma_sq), shape)
        
        if return_variance:
            return w, sigma_sq
        return w
    
    def __repr__(self):
        return f"HierarchicalGaussianPrior(alpha={self.alpha}, beta={self.beta})"

class LayerWisePrior:
    """
    켜마다 다른 앞선 분포.
    """
    
    def __init__(self, layer_priors: Dict[str, Prior]):
        """
        Parameters
        ----------
        layer_priors : dict
            켜 이름을 Prior 물체에 맞춘 사전
        """
        self.layer_priors = layer_priors
    
    def log_prob(self, weights: Dict[str, np.ndarray]) -> float:
        """켜 모두에 걸친 온 로그 낌새를 셈한다."""
        total = 0.0
        for name, w in weights.items():
            if name in self.layer_priors:
                total += self.layer_priors[name].log_prob(w)
        return total
    
    def sample(self, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
        """켜 모두의 짐을 뽑는다."""
        return {
            name: self.layer_priors[name].sample(shape)
            for name, shape in shapes.items()
            if name in self.layer_priors
        }

# =============================================================================
# 흩어짐 잣대 잔손질
# =============================================================================

def compute_glorot_variance(fan_in: int, fan_out: int) -> float:
    """
    글로로/자비에 흩어짐 잣대.
    
    tanh/시그모이드에서 켜에 걸쳐 살림의 흩어짐을 지킨다.
    """
    return 2.0 / (fan_in + fan_out)

def compute_he_variance(fan_in: int) -> float:
    """
    허 흩어짐 잣대.
    
    ReLU 살림에서 흩어짐을 지킨다.
    """
    return 2.0 / fan_in

def compute_lecun_variance(fan_in: int) -> float:
    """
    르쿤 흩어짐 잣대.
    
    SELU 살림에서 흩어짐을 지킨다.
    """
    return 1.0 / fan_in

def create_scaled_gaussian_prior(
    layer_shapes: List[Tuple[int, int]],
    scaling: str = 'glorot'
) -> LayerWisePrior:
    """
    흩어짐 잣대를 제대로 맞춘 켜마다의 가우스 앞선 분포를 만든다.
    
    Parameters
    ----------
    layer_shapes : 짝들의 목록
        켜마다의 (fan_in, fan_out)
    scaling : str
        'glorot', 'he', 'lecun' 가운데 하나
    
    Returns
    -------
    LayerWisePrior
        켜에 맞는 흩어짐을 지닌 앞선 분포
    """
    priors = {}
    
    for i, (fan_in, fan_out) in enumerate(layer_shapes):
        if scaling == 'glorot':
            var = compute_glorot_variance(fan_in, fan_out)
        elif scaling == 'he':
            var = compute_he_variance(fan_in)
        elif scaling == 'lecun':
            var = compute_lecun_variance(fan_in)
        else:
            raise ValueError(f"모르는 잣대: {scaling}")
        
        priors[f'W{i}'] = GaussianPrior(sigma=np.sqrt(var))
        priors[f'b{i}'] = GaussianPrior(sigma=0.1)  # 작은 치우침 앞선 분포
    
    return LayerWisePrior(priors)

# =============================================================================
# 앞선 분포로 미리 뽑아 보기
# =============================================================================

class SimpleMLP:
    """앞선 분포로 미리 살펴보는 데 쓰는 단순 MLP."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'relu'
    ):
        """
        Parameters
        ----------
        layer_sizes : list
            [들임 차수, 숨은 켜1, 숨은 켜2, ..., 날임 차수]
        activation : str
            'relu', 'tanh', 'sigmoid' 가운데 하나
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # 살림 함수를 매긴다
        if activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
        elif activation == 'tanh':
            self.act_fn = np.tanh
        elif activation == 'sigmoid':
            self.act_fn = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            raise ValueError(f"모르는 살림 함수: {activation}")
    
    def forward(
        self,
        x: np.ndarray,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """주어진 짐으로 앞으로 걸음."""
        h = x
        n_layers = len(self.layer_sizes) - 1
        
        for i in range(n_layers):
            W = weights[f'W{i}']
            b = weights.get(f'b{i}', np.zeros(W.shape[1]))
            
            h = h @ W + b
            
            # 살림을 건다(마지막 켜는 뺀다)
            if i < n_layers - 1:
                h = self.act_fn(h)
        
        return h
    
    def get_weight_shapes(self) -> Dict[str, Tuple[int, int]]:
        """짐 행렬 모두의 꼴을 얻는다."""
        shapes = {}
        for i in range(len(self.layer_sizes) - 1):
            shapes[f'W{i}'] = (self.layer_sizes[i], self.layer_sizes[i + 1])
            shapes[f'b{i}'] = (self.layer_sizes[i + 1],)
        return shapes

def prior_predictive_check(
    model: SimpleMLP,
    prior: Union[Prior, LayerWisePrior],
    x_test: np.ndarray,
    n_samples: int = 100
) -> np.ndarray:
    """
    앞선 분포로 미리 보는 분포에서 함수를 뽑는다.
    
    Parameters
    ----------
    model : SimpleMLP
        신경 그물 얼개
    prior : Prior 또는 LayerWisePrior
        짐의 앞선 분포
    x_test : ndarray
        시험 들임
    n_samples : int
        뽑을 함수의 수
    
    Returns
    -------
    (n_samples, n_test_points, output_dim) 꼴의 ndarray
        뽑은 함수
    """
    shapes = model.get_weight_shapes()
    
    predictions = []
    
    for _ in range(n_samples):
        # 앞선 분포에서 짐을 뽑는다
        if isinstance(prior, LayerWisePrior):
            weights = prior.sample(shapes)
        else:
            weights = {name: prior.sample(shape) for name, shape in shapes.items()}
        
        # 앞으로 걸음
        y = model.forward(x_test, weights)
        predictions.append(y)
    
    return np.array(predictions)

# =============================================================================
# 그리는 함수
# =============================================================================

def plot_prior_comparison(
    priors: Dict[str, Prior],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 1000,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    여러 앞선 분포를 견준다.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # 밀도 함수 견주기
    ax = axes[0]
    for name, prior in priors.items():
        if isinstance(prior, GaussianPrior):
            pdf = stats.norm.pdf(x, prior.mean, prior.sigma)
        elif isinstance(prior, LaplacePrior):
            pdf = stats.laplace.pdf(x, 0, prior.scale)
        elif isinstance(prior, StudentTPrior):
            pdf = stats.t.pdf(x, prior.df, scale=prior.scale)
        else:
            # 수로 셈하는 어림
            pdf = np.array([prior.prob(np.array([xi])) for xi in x])
        
        ax.plot(x, pdf, label=name, linewidth=2)
    
    ax.set_xlabel('짐 값')
    ax.set_ylabel('밀도')
    ax.set_title('앞선 분포의 밀도')
    ax.legend()
    ax.set_ylim(0, None)
    
    # 로그 밀도 견주기(꼬리를 보려고)
    ax = axes[1]
    for name, prior in priors.items():
        if isinstance(prior, GaussianPrior):
            log_pdf = stats.norm.logpdf(x, prior.mean, prior.sigma)
        elif isinstance(prior, LaplacePrior):
            log_pdf = stats.laplace.logpdf(x, 0, prior.scale)
        elif isinstance(prior, StudentTPrior):
            log_pdf = stats.t.logpdf(x, prior.df, scale=prior.scale)
        else:
            log_pdf = np.array([prior.log_prob(np.array([xi])) for xi in x])
        
        ax.plot(x, log_pdf, label=name, linewidth=2)
    
    ax.set_xlabel('짐 값')
    ax.set_ylabel('로그 밀도')
    ax.set_title('앞선 분포의 로그 밀도(꼬리 결이 보인다)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_prior_predictive(
    predictions: np.ndarray,
    x_test: np.ndarray,
    title: str = "앞선 분포로 미리 보는 분포",
    n_show: int = 20
):
    """
    앞선 분포로 미리 뽑은 것을 그린다.
    
    Parameters
    ----------
    predictions : (n_samples, n_points) 꼴의 ndarray
        뽑은 함수
    x_test : ndarray
        시험 들임(1차)
    title : str
        그림의 이름
    n_show : int
        그릴 표본의 수
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 뽑은 함수
    ax = axes[0]
    for i in range(min(n_show, len(predictions))):
        ax.plot(x_test, predictions[i], alpha=0.3, color='blue')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'{title}\n(표본 {n_show}개를 보임)')
    
    # 평균과 아리송함
    ax = axes[1]
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    ax.fill_between(x_test.flatten(), mean - 2*std, mean + 2*std,
                    alpha=0.3, label='±2σ')
    ax.plot(x_test, mean, 'b-', linewidth=2, label='평균')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('앞선 분포로 미리 본 평균과 아리송함')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_sparsity_pattern(
    prior: Prior,
    n_samples: int = 10000,
    title: str = "성김의 결"
):
    """
    앞선 분포가 성김을 이끄는 결을 그린다.
    """
    samples = prior.sample((n_samples,))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 잦기 그림
    ax = axes[0]
    ax.hist(samples, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='0')
    ax.set_xlabel('짐 값')
    ax.set_ylabel('밀도')
    ax.set_title(f'{title}\n표본의 잦기 그림')
    ax.legend()
    
    # 0 언저리의 몫
    ax = axes[1]
    thresholds = np.logspace(-3, 0, 50)
    fractions = [np.mean(np.abs(samples) < t) for t in thresholds]
    
    ax.semilogx(thresholds, fractions, 'b-', linewidth=2)
    ax.set_xlabel('문턱 |w| < τ')
    ax.set_ylabel('짐의 몫')
    ax.set_title('0 언저리의 쌓인 몫')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{prior}의 성김 자:")
    print(f"  |w| < 0.01의 몫: {np.mean(np.abs(samples) < 0.01):.3f}")
    print(f"  |w| < 0.1의 몫:  {np.mean(np.abs(samples) < 0.1):.3f}")
    print(f"  |w| > 1.0의 몫:  {np.mean(np.abs(samples) > 1.0):.3f}")

# =============================================================================
# 보여 주는 함수
# =============================================================================

def demo_standard_priors():
    """여느 앞선 분포를 견준다."""
    
    print("=" * 70)
    print("여느 앞선 분포")
    print("=" * 70)
    
    priors = {
        'Gaussian (σ=1)': GaussianPrior(sigma=1.0),
        'Gaussian (σ=0.5)': GaussianPrior(sigma=0.5),
        'Laplace (scale=1)': LaplacePrior(scale=1.0),
        'Student-t (ν=3)': StudentTPrior(df=3.0, scale=1.0),
        'Student-t (ν=10)': StudentTPrior(df=10.0, scale=1.0),
    }
    
    print("\n앞선 분포 간추림:")
    for name, prior in priors.items():
        samples = prior.sample((10000,))
        print(f"  {name:25s}: mean={np.mean(samples):+.3f}, "
              f"std={np.std(samples):.3f}, "
              f"|w|>2: {np.mean(np.abs(samples) > 2):.3f}")
    
    return priors

def demo_sparsity_priors():
    """성김을 이끄는 앞선 분포를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("성김을 이끄는 앞선 분포")
    print("=" * 70)
    
    priors = {
        'Gaussian': GaussianPrior(sigma=1.0),
        'Laplace': LaplacePrior(scale=1.0),
        'Spike-and-Slab': SpikeAndSlabPrior(pi=0.8, sigma_spike=0.01, sigma_slab=1.0),
        'Horseshoe': HorseshoePrior(tau=1.0),
    }
    
    print("\n성김 견주기(0 언저리의 몫):")
    print(f"{'앞선 분포':<20} {'|w|<0.01':>10} {'|w|<0.1':>10} {'|w|>2':>10}")
    print("-" * 55)
    
    for name, prior in priors.items():
        samples = prior.sample((10000,))
        print(f"{name:<20} {np.mean(np.abs(samples) < 0.01):>10.3f} "
              f"{np.mean(np.abs(samples) < 0.1):>10.3f} "
              f"{np.mean(np.abs(samples) > 2):>10.3f}")
    
    return priors

def demo_hierarchical_prior():
    """층진 앞선 분포를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("층진 앞선 분포")
    print("=" * 70)
    
    # 여러 alpha 값을 견준다
    alphas = [1.0, 2.0, 5.0, 10.0]
    
    print("\n흩어짐에 Inv-Gamma(α, β=1)을 얹은 층진 가우스:")
    print(f"{'α':>5} {'가장자리 자유도':>12} {'표본 잣대 어긋남':>12} {'뾰족함':>12}")
    print("-" * 45)
    
    for alpha in alphas:
        prior = HierarchicalGaussianPrior(alpha=alpha, beta=1.0)
        samples = prior.sample((10000,))
        
        # 뾰족함(넘침, 가우스 = 0)
        kurtosis = stats.kurtosis(samples)
        
        print(f"{alpha:>5.1f} {prior.marginal_df:>12.1f} "
              f"{np.std(samples):>12.3f} {kurtosis:>12.3f}")
    
    print("\n*** α이 낮을수록 → 꼬리가 두껍다(뾰족함이 크다)")
    print("*** α → ∞이면 가장자리 분포가 가우스에 다가간다")

def demo_variance_scaling():
    """얼개마다의 흩어짐 잣대를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("흩어짐 잣대")
    print("=" * 70)
    
    # 보기 얼개
    layer_sizes = [784, 256, 128, 10]
    
    print(f"\n얼개: {layer_sizes}")
    print("\n즐겨 쓰는 앞선 분포 잣대 어긋남:")
    print(f"{'켜':<10} {'꼴':<15} {'글로로 σ':>12} {'허 σ':>12} {'르쿤 σ':>12}")
    print("-" * 65)
    
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        shape = f"({fan_in}, {fan_out})"
        
        glorot_var = compute_glorot_variance(fan_in, fan_out)
        he_var = compute_he_variance(fan_in)
        lecun_var = compute_lecun_variance(fan_in)
        
        print(f"W{i:<8} {shape:<15} {np.sqrt(glorot_var):>12.4f} "
              f"{np.sqrt(he_var):>12.4f} {np.sqrt(lecun_var):>12.4f}")
    
    print("\n*** 글로로: tanh/시그모이드에")
    print("*** 허: ReLU에")
    print("*** 르쿤: SELU에")

def demo_prior_predictive():
    """앞선 분포로 미리 살펴보기를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("앞선 분포로 미리 살펴보기")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 단순 MLP을 만든다
    model = SimpleMLP([1, 50, 50, 1], activation='tanh')
    
    # 시험 들임
    x_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    
    # 여러 앞선 분포
    prior_configs = {
        '너른 가우스 (σ=1)': GaussianPrior(sigma=1.0),
        '좁은 가우스 (σ=0.1)': GaussianPrior(sigma=0.1),
        '허 잣대': None,  # 알맞은 잣대를 쓴다
    }
    
    print("\n앞선 분포로 미리 본 자(시험 점에 걸쳐):")
    print(f"{'앞선 분포':<25} {'날임 평균':>12} {'날임 잣대 어긋남':>12} {'가장 큰 |y|':>12}")
    print("-" * 65)
    
    for name, prior in prior_configs.items():
        if prior is None:
            # 허 잣대 앞선 분포
            shapes = [(1, 50), (50, 50), (50, 1)]
            prior = create_scaled_gaussian_prior(shapes, scaling='he')
        
        predictions = prior_predictive_check(model, prior, x_test, n_samples=100)
        predictions = predictions.squeeze()
        
        print(f"{name:<25} {np.mean(predictions):>12.3f} "
              f"{np.std(predictions):>12.3f} {np.max(np.abs(predictions)):>12.3f}")
    
    print("\n*** 잣대를 제대로 맞추면 날임이 이치에 닿는 자리에 머문다")
    print("*** 너른 앞선 분포는 함수 값을 아주 크게 만들 수 있다")

def demo_l2_equivalence():
    """가우스 앞선 분포 ↔ L2 정칙화가 같음을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("가우스 앞선 분포 ↔ L2 정칙화")
    print("=" * 70)
    
    print("\n가우스 앞선 분포 N(0, σ²)을 쓴 MAP 어림은")
    print("벌 λ = 1/(2σ²)의 L2 정칙화 MLE와 같다")
    
    print(f"\n{'σ²':>10} {'σ':>10} {'λ = 1/(2σ²)':>15} {'풀이':>25}")
    print("-" * 65)
    
    variances = [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]
    
    for var in variances:
        sigma = np.sqrt(var)
        lam = 1 / (2 * var)
        
        if lam > 10:
            interp = "아주 센 정칙화"
        elif lam > 1:
            interp = "센 정칙화"
        elif lam > 0.1:
            interp = "가운데 정칙화"
        elif lam > 0.01:
            interp = "여린 정칙화"
        else:
            interp = "아주 여린 정칙화"
        
        print(f"{var:>10.2f} {sigma:>10.3f} {lam:>15.4f} {interp:>25}")

if __name__ == "__main__":
    demo_standard_priors()
    demo_sparsity_priors()
    demo_hierarchical_prior()
    demo_variance_scaling()
    demo_prior_predictive()
    demo_l2_equivalence()
```

---

## 연습문제

**연습문제 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "연습문제 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**연습문제 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**연습문제 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "연습문제 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**연습문제 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

### 여느 앞선 분포

| 앞선 분포 | 식 | MAP과 같은 것 | 결 |
|-------|---------|----------------|------------|
| **가우스** | $\mathcal{N}(0, \sigma^2)$ | L2 정칙화 | 매끄럽고 얌전하다 |
| **라플라스** | $\frac{\lambda}{2}e^{-\lambda\|w\|}$ | L1 정칙화 | 성김을 이끈다 |
| **스튜던트 $t$** | 꼬리가 두껍다 | 든든한 벌 | 튀는 값을 받아 준다 |

### 흩어짐 잣대

| 방법 | 식 | 잘 맞는 자리 |
|--------|---------|----------|
| **글로로/자비에** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Tanh, 시그모이드 |
| **허** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ | ReLU |
| **르쿤** | $\sigma^2 = \frac{1}{n_{\text{in}}}$ | SELU |

### 성김을 이끄는 앞선 분포

| 앞선 분포 | 고갱이 결 | 쓸 자리 |
|-------|-------------|----------|
| **못과 널** | 꼭 0이 됨 | 결 고르기 |
| **말굽** | 두꺼운 꼬리 + 오그리기 | 성긴 신호 |
| **ARD** | 결마다의 흩어짐 | 절로 쳐내기 |

### 층진 앞선 분포

$$
p(w \mid \sigma^2) = \mathcal{N}(0, \sigma^2), \quad p(\sigma^2) = \text{Inv-Gamma}(\alpha, \beta)
$$

**나은 점**:

- 자료에 맞추어 가는 정칙화
- 가우스보다 꼬리가 두껍다
- 하이퍼파라미터에 덜 예민하다

### 고갱이 꾸밈 원칙

1. **잣대를 알맞게**: 흩어짐을 켜 너비에 맞춘다
2. **함수 밭을 헤아려라**: 짐의 앞선 분포가 함수의 앞선 분포를 이끈다
3. **층진 앞선 분포를 써라**: 정칙화의 셈을 자료가 정하게 한다
4. **앞선 분포로 미리 살펴 따져라**: 익히기 앞서 뽑아 보고 그려 본다

### 다른 장과의 이어짐

| 이야기 | 장 | 이어짐 |
|-------|---------|------------|
| 뒷분포 미루어 봄 | 13장: 뒷분포 미루어 봄 | 앞선 분포가 뒷분포의 꼴에 걸린다 |
| 아리송함 | 13장: 아리송함 | 앞선 분포가 앎의 아리송함에 걸린다 |
| MC 드롭아웃 | 13장: MC 드롭아웃 | 앞선 분포를 넌지시 매긴다 |
| 변이 베이즈 신경 그물 | 13장: 변이 베이즈 신경 그물 | KL 갈림 속의 앞선 분포 |
| 정칙화 | 6장: 정칙화 | MAP ↔ 벌을 준 MLE |

### 고갱이 살펴볼 거리

- Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- Louizos, C., et al. (2017). Bayesian compression for deep learning. *NeurIPS*.
- Fortuin, V. (2022). Priors in Bayesian deep learning: A review. *International Statistical Review*.
- Wenzel, F., et al. (2020). How good is the Bayes posterior in deep neural networks really? *ICML*.
