# 신경 그물의 아리송함 재기
**아리송함 재기**은 여느 신경 그물의 종요로운 한계를 다룬다. 여느 그물은 자신함을 재는 자 없이 점 미루어 봄만 낸다. 베이즈 신경 그물은 모형이 자료에 대해 모르는 것(타고난 아리송함)과 제 스스로에 대해 모르는 것(앎의 아리송함)을 함께 재는 이치에 닿는 틀을 준다.

---

## 왜 하는가: 아리송함이 걸리는 까닭

### 지나친 자신함 문제

가장 큰 그럴듯함으로 익힌 여느 신경 그물은 점 어림을 낸다.

$$
\hat{y} = f_{\hat{\theta}}(x)
$$

**종요로운 문제**:

1. **자신함을 재는 자가 없음**: 그물은 미더움을 알리는 것 없이 미루어 봄 하나만 낸다

2. **지나치게 자신하는 밖으로 늘리기**: 그물은 익힘 자료에서 멀리서도 자신하는 미루어 봄을 하는 일이 잦다

3. **말없는 어그러짐**: 들임이 밖 분포일 때 모형은 알림 없이 어그러진다

4. **눈금 어긋난 낌새**: 소프트맥스 날임은 참 미루어 봄의 아리송함을 드러내지 못한다

### 참 세상에서의 뒤끝

**병 살피기**: 모형이 "착한 혹일 낌새 95%"라고 미루어 본다. 그런데 이는 다음 가운데 무엇 때문인가?

- 모형이 비슷한 자리를 많이 보았기 때문인가(아리송함이 작다)?
- 모형이 마구 밖으로 늘리고 있기 때문인가(아리송함이 크다)?

**스스로 몰기**: 알아보는 얼개는 사람이 끼어들도록 언제 아리송한지 알아야 한다.

**앎을 찾아내기**: 아리송함은 자료를 어디서 더 모을지 이끈다(살아 있는 배움).

### 베이즈 방법이 주는 것

베이즈의 길은 모형 매개변수에 대한 **뒷분포**을 지닌다.

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

이는 다음을 이룬다.

1. 점 미루어 봄 대신 **미루어 보는 분포**
2. 타고난 몫과 앎의 몫으로 **쪼개기**
3. 모형을 지나는 아리송함의 **이치에 닿는 퍼짐**
4. 모형 가정 아래서의 **절로 눈금 맞음**

---

## 아리송함의 갈래

### 갈래 나누기 두루 보기

$$
\boxed{\text{Total Uncertainty} = \text{Aleatoric Uncertainty} + \text{Epistemic Uncertainty}}
$$

| 갈래 | 다른 이름 | 밑동 | 줄일 수 있나? |
|------|-------------|--------|------------|
| **타고난** | 자료의 아리송함 | 살핌에 타고난 잡음 | 아니다 |
| **앎의** | 모형의 아리송함 | 자료가 적음, 모형이 모름 | 그렇다(자료를 더 모으면) |

### 타고난 아리송함

**뜻매김**: 자료를 낳는 흐름에 타고난 아리송함으로, 자료를 더 모아도 줄지 않는다.

**밑동**:

- 재기의 잡음
- 확률 흐름
- 온전하지 않은 살핌(숨은 변수)
- 가름에서 갈래가 겹침

**수학 꼴**(되돌이):

$$
y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2(x))
$$

잡음 흩어짐 $\sigma^2(x)$은 다음일 수 있다.

- **고른 흩어짐**: 들임 밭 어디서나 붙박이
- **다른 흩어짐**: 들임 $x$에 따라 달라짐

**보기**: 주식 값 미루어 보기

- 걸린 것을 모두 온전히 알아도 값에는 타고난 아무렇게나임이 있다
- 자료를 더 모아도 이 밑바탕의 못 미루어 봄은 없어지지 않는다

### 앎의 아리송함

**뜻매김**: 모형에 대한 앎이 모자라 생기는 아리송함으로, 자료를 더 모으면 줄일 수 있다.

**밑동**:

- 익힘 자료가 적음
- 모형을 잘못 정함
- 매개변수의 아리송함
- 밖 분포 들임

**수학 꼴**:

앎의 아리송함은 뒷분포 $p(\theta \mid \mathcal{D})$이 담는다.

- 좁은 뒷분포 → 앎의 아리송함이 작다(매개변수를 자신한다)
- 너른 뒷분포 → 앎의 아리송함이 크다(매개변수가 아리송하다)

**보기**: 새 동네의 집값 미루어 보기

- 살핌이 적으면 값과 결의 사이가 아리송하다
- 이 동네의 자료를 더 모으면 아리송함이 준다

### 고른 흩어짐 대 다른 흩어짐

**고른 흩어짐**(붙박인 잡음):

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)
$$

- 모든 들임에 잡음 매개변수 $\sigma^2$ 하나
- 모형으로 만들기가 더 쉽다
- 참 자리와 맞지 않는 일이 잦다

**다른 흩어짐**(들임에 매인 잡음):

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2_\theta(x))
$$

- 그물이 평균과 흩어짐을 함께 미루어 본다
- 더 너그럽고 참 자리에 가깝다
- 조심스러운 익힘이 있어야 한다

---

## 미루어 보는 분포

### 뜻매김

**뒷분포로 미루어 보는 분포**은 매개변수의 아리송함에 걸쳐 적분한다.

$$
\boxed{p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta}
$$

여기서

- $x^*$은 시험 들임
- $y^*$은 미루어 본 날임
- $p(y^* \mid x^*, \theta)$은 그럴듯함(매개변수가 주어졌을 때의 모형 미루어 봄)
- $p(\theta \mid \mathcal{D})$은 매개변수의 뒷분포

### 몬테카를로 어림

적분을 다룰 수 없으므로 뒷분포에서 뽑은 표본 $\{\theta^{(s)}\}_{s=1}^S$으로 어림한다.

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(y^* \mid x^*, \theta^{(s)})
$$

**되돌이에서는**(가우스 그럴듯함):

$$
\hat{\mu}(x^*) = \frac{1}{S} \sum_{s=1}^S f_{\theta^{(s)}}(x^*)
$$

$$
\hat{\sigma}^2(x^*) = \frac{1}{S} \sum_{s=1}^S \left[ f_{\theta^{(s)}}(x^*) - \hat{\mu}(x^*) \right]^2 + \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*)
$$

**가름에서는**:

$$
p(y^* = c \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \text{softmax}(f_{\theta^{(s)}}(x^*))_c
$$

### 미루어 본 평균과 흩어짐

다른 흩어짐 잡음을 지닌 되돌이에서 미루어 보는 분포는 다음을 지닌다.

**미루어 본 평균**:

$$
\mathbb{E}[y^* \mid x^*, \mathcal{D}] = \mathbb{E}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]
$$

**미루어 본 흩어짐**(온 흩어짐 법칙):

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \underbrace{\mathbb{E}_{\theta \mid \mathcal{D}}[\sigma^2_\theta(x^*)]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]}_{\text{Epistemic}}
$$

이 쪼갬은 아리송함의 밑동을 알아보는 데 밑바탕이 된다.

---

## 아리송함 쪼개기

### 온 흩어짐 법칙

이 쪼갬은 **온 흩어짐 법칙**에서 나온다.

$$
\text{Var}[Y] = \mathbb{E}[\text{Var}[Y \mid X]] + \text{Var}[\mathbb{E}[Y \mid X]]
$$

$Y = y^*$, $X = \theta$으로 두고 우리 자리에 쓰면

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \mathbb{E}_\theta[\text{Var}[y^* \mid x^*, \theta]] + \text{Var}_\theta[\mathbb{E}[y^* \mid x^*, \theta]]
$$

### 타고난 아리송함

$$
\boxed{\text{Aleatoric}(x^*) = \mathbb{E}_{\theta \mid \mathcal{D}}[\sigma^2_\theta(x^*)]}
$$

**풀이**: 매개변수의 아리송함에 걸쳐 고르게 한, 바라는 살핌 잡음

**어림**:

$$
\widehat{\text{Aleatoric}}(x^*) = \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*)
$$

**결**:

- 줄일 수 없다: 자료를 더 모아도 줄지 않는다
- 자료에 매인다: 들임 밭에 따라 달라진다
- 문제에 타고난 잡음을 담는다

### 앎의 아리송함

$$
\boxed{\text{Epistemic}(x^*) = \text{Var}_{\theta \mid \mathcal{D}}[\mu_\theta(x^*)]}
$$

**풀이**: 매개변수의 아리송함에서 오는 미루어 봄의 흩어짐

**어림**:

$$
\widehat{\text{Epistemic}}(x^*) = \frac{1}{S} \sum_{s=1}^S \left[\mu_{\theta^{(s)}}(x^*) - \bar{\mu}(x^*)\right]^2
$$

여기서 $\bar{\mu}(x^*) = \frac{1}{S} \sum_s \mu_{\theta^{(s)}}(x^*)$이다.

**결**:

- 줄일 수 있다: 자료를 더 모으면 준다
- 익힘 자료에서 먼 자리에서 크다
- 모형이 모름을 담는다

### 온 미루어 봄 아리송함

$$
\boxed{\text{Total}(x^*) = \text{Aleatoric}(x^*) + \text{Epistemic}(x^*)}
$$

**어림**:

$$
\widehat{\text{Total}}(x^*) = \frac{1}{S} \sum_{s=1}^S \sigma^2_{\theta^{(s)}}(x^*) + \frac{1}{S} \sum_{s=1}^S \left[\mu_{\theta^{(s)}}(x^*) - \bar{\mu}(x^*)\right]^2
$$

---

## 가름에서의 아리송함

### 미루어 본 엔트로피

가름에서 미루어 보는 분포는 갈래 분포다.

$$
p(y^* = c \mid x^*, \mathcal{D}) = \bar{p}_c = \frac{1}{S} \sum_{s=1}^S p(y^* = c \mid x^*, \theta^{(s)})
$$

엔트로피로 재는 **온 아리송함**:

$$
\boxed{\mathbb{H}[y^* \mid x^*, \mathcal{D}] = -\sum_{c=1}^C \bar{p}_c \log \bar{p}_c}
$$

### 서로 나눈 소식으로 쪼개기

온 엔트로피는 이렇게 쪼개진다.

$$
\underbrace{\mathbb{H}[y^* \mid x^*, \mathcal{D}]}_{\text{Total}} = \underbrace{\mathbb{I}[y^*; \theta \mid x^*, \mathcal{D}]}_{\text{Epistemic (MI)}} + \underbrace{\mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]]}_{\text{Aleatoric}}
$$

**타고난 아리송함**(바라는 엔트로피):

$$
\text{Aleatoric}(x^*) = \mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]] = -\frac{1}{S} \sum_{s=1}^S \sum_{c=1}^C p_{c,s} \log p_{c,s}
$$

여기서 $p_{c,s} = p(y^* = c \mid x^*, \theta^{(s)})$이다.

**앎의 아리송함**(서로 나눈 소식):

$$
\text{Epistemic}(x^*) = \mathbb{H}[y^* \mid x^*, \mathcal{D}] - \mathbb{E}_{\theta \mid \mathcal{D}}[\mathbb{H}[y^* \mid x^*, \theta]]
$$

### BALD: 어긋남으로 하는 베이즈 살아 있는 배움

서로 나눈 소식(앎의 아리송함)은 살아 있는 배움의 **BALD**에 쓰인다.

$$
\text{BALD}(x^*) = \mathbb{I}[y^*; \theta \mid x^*, \mathcal{D}] = \mathbb{H}[\bar{p}] - \frac{1}{S} \sum_{s=1}^S \mathbb{H}[p_s]
$$

**풀이**:

- 매개변수 표본끼리 어긋날 때 크다
- BALD이 큰 점은 배우는 데 알려 주는 바가 크다

**고르는 잣대**:

$$
x^*_{\text{next}} = \arg\max_{x \in \mathcal{X}_{\text{pool}}} \text{BALD}(x)
$$

---

## 눈금 맞음과 미더움

### 눈금 맞음이란

미루어 본 낌새가 겪은 잦기와 들어맞으면 모형은 **눈금이 잘 맞는다**.

$$
P(y = 1 \mid p(y = 1 \mid x) = q) = q \quad \forall q \in [0, 1]
$$

**보기**: 자신함 80%인 미루어 봄 가운데 80%이 맞아야 한다.

### 바라는 눈금 맞음 어긋남(ECE)

미루어 봄을 자신함에 따라 통 $M$개로 나눈다.

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

여기서

- $B_m$은 통 $m$에 든 보기의 모임
- $\text{acc}(B_m)$은 통 $m$의 맞음
- $\text{conf}(B_m)$은 통 $m$의 평균 자신함

### 미더움 그림

통마다 맞음을 자신함에 대고 그린다.

- **온전한 눈금 맞음**: 점이 대각선 위에 있다
- **지나친 자신함**: 점이 대각선 아래에 있다
- **머뭇거림**: 점이 대각선 위쪽에 있다

### 신경 그물의 눈금 어긋남

요즘 신경 그물은 흔히 **지나치게 자신한다**.

**까닭**:

1. **엇갈린 엔트로피 익힘**이 자신하는 미루어 봄을 이끈다
2. **ReLU 살림**은 마디 없는 로짓을 낸다
3. **묶음 잣대 잡기**가 눈금 맞음을 바꾼다
4. **모형이 담는 힘**이 자료의 얽힘을 넘어선다

**베이즈 풀이**: 매개변수에 걸쳐 적분하면 눈금 맞음이 절로 나아진다.

---

## 밖 분포 알아내기

### 밖 분포 문제

여느 신경 그물은 익힘 분포 밖에서 온 들임을 미덥게 짚어내지 못한다.

$$
x_{\text{OOD}} \notin \text{support}(p_{\text{train}}(x))
$$

**바라는 움직임**: 밖 분포 들임에서 아리송함이 크다

### 밖 분포 알아내기에 아리송함 쓰기

모형이 비슷한 자료를 본 적이 없으므로 밖 분포 들임에서는 **앎의 아리송함**이 커야 한다.

**알아내기 점수**:

$$
s(x) = \text{Epistemic}(x) \quad \text{or} \quad s(x) = \text{Total}(x)
$$

**판단 규칙**:

$$
\text{OOD if } s(x) > \tau
$$

### 따지는 자

**AUROC**: 밖 분포 알아내기의 ROC 굽이 아래 넓이

- 분포 안 보기: 아닌 갈래
- 밖 분포 보기: 맞는 갈래

**AUPRC**: 촘촘함-되불러옴 굽이 아래 넓이

**FPR@95**: 참 맞음 비율이 95%일 때의 헛 맞음 비율

### 어려움

1. **지나치게 자신하는 밖으로 늘리기**: ReLU 그물은 자료에서 멀리서도 자신할 수 있다
2. **결 주저앉음**: 깊은 그물이 밖 분포 들임을 분포 안의 결로 옮길 수 있다
3. **분포 옮겨감**: 차츰 옮겨가는 것은 뚜렷한 밖 분포보다 짚어내기 어렵다

---

## 다른 흩어짐 신경 그물

### 얼개

다른 흩어짐 그물은 평균과 흩어짐을 함께 미루어 본다.

$$
f_\theta(x) = [\mu_\theta(x), \log \sigma^2_\theta(x)]
$$

**왜 로그 흩어짐인가?**

- $\sigma^2 > 0$을 지킨다
- 셈이 든든하다
- 가장 좋게 하기가 쉽다

### 익힘 목표

가우스 살핌의 **음수 로그 그럴듯함**:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[ \frac{(y_i - \mu_\theta(x_i))^2}{2\sigma^2_\theta(x_i)} + \frac{1}{2}\log \sigma^2_\theta(x_i) \right]
$$

**풀이**:

- 첫째 항: 흩어짐의 거꿀로 짐을 준 미루어 봄 어긋남
- 둘째 항: 흩어짐을 다독인다(끝없는 흩어짐을 막는다)

### 참으로 헤아릴 것

**익힘의 든든함**:

- 흩어짐 미루어 봄의 첫자리를 조심스레 잡는다
- 평균 머리와 흩어짐 머리에 다른 배움 비율을 쓴다
- 셈 문제를 막으려 흩어짐을 자른다

**얼개 고르기**:

- 등뼈를 나누어 쓰고 날임 머리는 둘
- 따로 떨어진 그물(더 너그러우나 매개변수가 많다)
- 흩어짐은 결에 매일 수도, 들임에만 매일 수도 있다

---

## 파이썬으로 짜기

```python
"""
신경 그물의 아리송함 재기

이 묶음은 다음을 짜 놓았다:
- 타고난 아리송함과 앎의 아리송함 어림
- 다른 흩어짐 신경 그물
- 눈금 맞음 자와 미더움 그림
- 밖 분포 알아내기
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax, logsumexp
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
import warnings


# =============================================================================
# 아리송함 어림
# =============================================================================

def predictive_uncertainty_regression(
    predictions: np.ndarray,
    variances: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    되돌이의 아리송함 쪼갬을 셈한다.
    
    Parameters
    ----------
    predictions : (n_samples, n_points) 꼴의 ndarray
        뒷분포 표본마다의 평균 미루어 봄
    variances : (n_samples, n_points) 꼴의 ndarray, 골라 씀
        표본마다 미루어 본 흩어짐(타고난 것).
        None이면 흩어짐 0의 고른 흩어짐으로 본다.
    
    Returns
    -------
    mean : (n_points,) 꼴의 ndarray
        미루어 본 평균
    total_var : (n_points,) 꼴의 ndarray
        온 미루어 봄 흩어짐
    aleatoric : (n_points,) 꼴의 ndarray
        타고난(자료의) 아리송함
    epistemic : (n_points,) 꼴의 ndarray
        앎의(모형의) 아리송함
    """
    n_samples, n_points = predictions.shape
    
    # 미루어 본 평균
    mean = np.mean(predictions, axis=0)
    
    # 앎의: 평균들의 흩어짐
    epistemic = np.var(predictions, axis=0)
    
    # 타고난: 흩어짐들의 평균
    if variances is not None:
        aleatoric = np.mean(variances, axis=0)
    else:
        aleatoric = np.zeros(n_points)
    
    # 온 흩어짐
    total_var = epistemic + aleatoric
    
    return mean, total_var, aleatoric, epistemic


def predictive_uncertainty_classification(
    logits: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    가름의 아리송함 쪼갬을 셈한다.
    
    Parameters
    ----------
    logits : (n_samples, n_points, n_classes) 꼴의 ndarray
        뒷분포 표본마다의 로짓
    
    Returns
    -------
    mean_probs : (n_points, n_classes) 꼴의 ndarray
        평균 미루어 본 낌새
    total_entropy : (n_points,) 꼴의 ndarray
        온 미루어 봄 엔트로피
    aleatoric : (n_points,) 꼴의 ndarray
        타고난 아리송함(바라는 엔트로피)
    epistemic : (n_points,) 꼴의 ndarray
        앎의 아리송함(서로 나눈 소식)
    """
    n_samples, n_points, n_classes = logits.shape
    
    # 낌새로 옮긴다
    probs = softmax(logits, axis=2)  # (n_samples, n_points, n_classes)
    
    # 평균 낌새
    mean_probs = np.mean(probs, axis=0)  # (n_points, n_classes)
    
    # 온 엔트로피: H[E[p]]
    total_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    # 바라는 엔트로피: E[H[p]](타고난 것)
    sample_entropies = -np.sum(probs * np.log(probs + 1e-10), axis=2)
    aleatoric = np.mean(sample_entropies, axis=0)
    
    # 서로 나눈 소식(앎의 것)
    epistemic = total_entropy - aleatoric
    
    return mean_probs, total_entropy, aleatoric, epistemic


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """낌새 분포의 엔트로피를 셈한다."""
    return -np.sum(probs * np.log(probs + 1e-10), axis=axis)


def mutual_information(logits: np.ndarray) -> np.ndarray:
    """
    BALD에 쓸 서로 나눈 소식 I[y; theta | x, D]을 셈한다.
    
    Parameters
    ----------
    logits : (n_samples, n_points, n_classes) 꼴의 ndarray
    
    Returns
    -------
    mi : (n_points,) 꼴의 ndarray
        점마다의 서로 나눈 소식
    """
    _, _, _, mi = predictive_uncertainty_classification(logits)
    return mi


# =============================================================================
# 눈금 맞음 자
# =============================================================================

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    미더움 그림에 쓸 자료를 셈한다.
    
    Parameters
    ----------
    y_true : (n_samples,) 꼴의 ndarray
        참 이름표(둘 가름이면 0이나 1, 여러 갈래면 갈래 손가락질)
    y_prob : (n_samples,) 또는 (n_samples, n_classes) 꼴의 ndarray
        미루어 본 낌새
    n_bins : int
        통의 수
    
    Returns
    -------
    bin_centers : ndarray
        통마다의 가운데
    bin_accs : ndarray
        통마다의 맞음
    bin_confs : ndarray
        통마다의 평균 자신함
    """
    if y_prob.ndim == 1:
        # 둘 가름
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    else:
        # 여러 갈래: 가장 큰 낌새를 쓴다
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(accuracies[mask])
            bin_confs[i] = np.mean(confidences[mask])
        else:
            bin_accs[i] = np.nan
            bin_confs[i] = np.nan
    
    return bin_centers, bin_accs, bin_confs


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    바라는 눈금 맞음 어긋남(ECE)을 셈한다.
    
    Parameters
    ----------
    y_true : ndarray
        참 이름표
    y_prob : ndarray
        미루어 본 낌새
    n_bins : int
        통의 수
    
    Returns
    -------
    float
        ECE 값
    """
    if y_prob.ndim == 1:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
    else:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    
    accuracies = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        n_bin = np.sum(mask)
        
        if n_bin > 0:
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += (n_bin / len(y_true)) * np.abs(acc - conf)
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    가장 큰 눈금 맞음 어긋남(MCE)을 셈한다.
    """
    _, bin_accs, bin_confs = reliability_diagram_data(y_true, y_prob, n_bins)
    
    valid = ~np.isnan(bin_accs)
    if not np.any(valid):
        return 0.0
    
    return np.max(np.abs(bin_accs[valid] - bin_confs[valid]))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    낌새 미루어 봄의 브라이어 점수를 셈한다.
    
    작을수록 좋다. 둘 가름에서는 [0, 1] 자리.
    """
    if y_prob.ndim == 1:
        # 둘 가름
        return np.mean((y_prob - y_true) ** 2)
    else:
        # 여러 갈래: 참 이름표를 원핫으로 적는다
        n_classes = y_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))


# =============================================================================
# 밖 분포 알아내기
# =============================================================================

def ood_detection_metrics(
    in_scores: np.ndarray,
    out_scores: np.ndarray
) -> Dict[str, float]:
    """
    밖 분포 알아내기의 자를 셈한다.
    
    아리송함 점수가 클수록 밖 분포 보기여야 한다.
    
    Parameters
    ----------
    in_scores : ndarray
        분포 안 보기의 아리송함 점수
    out_scores : ndarray
        밖 분포 보기의 아리송함 점수
    
    Returns
    -------
    dict
        AUROC, AUPRC, FPR@95을 담은 사전
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    
    # 이름표: 분포 안이면 0, 밖 분포면 1
    y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_score)
    
    # AUPRC
    auprc = average_precision_score(y_true, y_score)
    
    # FPR@95: 참 맞음 비율이 95%일 때의 헛 맞음 비율
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = fpr[idx]
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'fpr_at_95': fpr_at_95
    }


def max_softmax_probability(logits: np.ndarray) -> np.ndarray:
    """
    밑금 밖 분포 알아내개: 음수 가장 큰 소프트맥스 낌새.
    
    값이 클수록 더 아리송하고 밖 분포일 낌새가 크다.
    """
    probs = softmax(logits, axis=-1)
    return -np.max(probs, axis=-1)


def predictive_entropy_score(logits: np.ndarray) -> np.ndarray:
    """
    밖 분포 알아내개: 미루어 본 엔트로피.
    
    모둠/베이즈에서: 로짓의 꼴은 (n_samples, n_points, n_classes)
    """
    if logits.ndim == 3:
        # 베이즈: 낌새를 고르게 한 뒤 엔트로피를 셈한다
        probs = softmax(logits, axis=2)
        mean_probs = np.mean(probs, axis=0)
        return entropy(mean_probs, axis=1)
    else:
        # 모형 하나
        probs = softmax(logits, axis=1)
        return entropy(probs, axis=1)


# =============================================================================
# 다른 흩어짐 그물
# =============================================================================

class HeteroscedasticLoss:
    """
    다른 흩어짐 가우스 음수 로그 그럴듯함 잃음.
    
    Loss = (y - mu)^2 / (2 * sigma^2) + 0.5 * log(sigma^2)
    """
    
    def __call__(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> float:
        """
        잃음을 셈한다.
        
        Parameters
        ----------
        y_true : ndarray
            참값
        mu : ndarray
            미루어 본 평균
        log_var : ndarray
            미루어 본 로그 흩어짐
        
        Returns
        -------
        float
            평균 잃음
        """
        var = np.exp(log_var)
        loss = 0.5 * ((y_true - mu) ** 2 / var + log_var)
        return np.mean(loss)
    
    def gradient(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        log_var: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        기울기를 셈한다.
        
        Returns
        -------
        grad_mu : ndarray
            미루어 본 평균에 대한 기울기
        grad_log_var : ndarray
            미루어 본 로그 흩어짐에 대한 기울기
        """
        var = np.exp(log_var)
        residual = y_true - mu
        
        grad_mu = -residual / var
        grad_log_var = 0.5 * (1 - residual ** 2 / var)
        
        return grad_mu, grad_log_var


class SimpleHeteroscedasticNetwork:
    """
    보여 주기용 단순 다른 흩어짐 신경 그물.
    
    등뼈를 나누어 쓰고 날임 머리는 둘이다:
    - 평균 머리: E[y|x]을 미루어 본다
    - 로그 흩어짐 머리: log Var[y|x]을 미루어 본다
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [50, 50],
        init_log_var: float = 0.0
    ):
        """
        그물의 첫자리를 잡는다.
        
        Parameters
        ----------
        input_dim : int
            들임 차수
        hidden_dims : list
            숨은 켜의 크기
        init_log_var : float
            첫 로그 흩어짐(조심스러운 비롯함)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.init_log_var = init_log_var
        
        # 짐의 첫자리를 잡는다(줄여 적음)
        self.weights = self._initialize_weights()
        self.loss_fn = HeteroscedasticLoss()
    
    def _initialize_weights(self) -> Dict:
        """자비에 첫자리 잡기."""
        weights = {}
        
        dims = [self.input_dim] + self.hidden_dims
        
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            weights[f'W{i}'] = np.random.randn(dims[i], dims[i + 1]) * scale
            weights[f'b{i}'] = np.zeros(dims[i + 1])
        
        # 날임 머리
        last_dim = dims[-1]
        
        # 평균 머리
        weights['W_mu'] = np.random.randn(last_dim, 1) * np.sqrt(2.0 / last_dim)
        weights['b_mu'] = np.zeros(1)
        
        # 로그 흩어짐 머리(조심스레 첫자리를 잡는다)
        weights['W_logvar'] = np.random.randn(last_dim, 1) * 0.01
        weights['b_logvar'] = np.full(1, self.init_log_var)
        
        return weights
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        앞으로 걸음.
        
        Returns
        -------
        mu : ndarray
            미루어 본 평균
        log_var : ndarray
            미루어 본 로그 흩어짐
        """
        h = X
        
        # 숨은 켜
        for i in range(len(self.hidden_dims)):
            h = h @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            h = np.maximum(h, 0)  # ReLU
        
        # 날임 머리
        mu = h @ self.weights['W_mu'] + self.weights['b_mu']
        log_var = h @ self.weights['W_logvar'] + self.weights['b_logvar']
        
        return mu.flatten(), log_var.flatten()
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        아리송함을 곁들여 미루어 본다.
        
        Returns
        -------
        mu : ndarray
            미루어 본 평균
        std : ndarray(return_std=True이면)
            미루어 본 잣대 어긋남(타고난 것만)
        """
        mu, log_var = self.forward(X)
        
        if return_std:
            std = np.sqrt(np.exp(log_var))
            return mu, std
        return mu, None


# =============================================================================
# 그리는 함수
# =============================================================================

def plot_uncertainty_decomposition(
    x: np.ndarray,
    mean: np.ndarray,
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = "아리송함 쪼개기"
):
    """
    미루어 봄의 아리송함을 쪼개어 그린다.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    total = np.sqrt(aleatoric + epistemic)
    aleatoric_std = np.sqrt(aleatoric)
    epistemic_std = np.sqrt(epistemic)
    
    # 온 아리송함
    ax = axes[0]
    ax.fill_between(x, mean - 2*total, mean + 2*total, 
                    alpha=0.3, label='온 ±2σ')
    ax.plot(x, mean, 'b-', label='평균 미루어 봄')
    if y_true is not None:
        ax.plot(x, y_true, 'k--', label='참 함수')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5, label='익힘 자료')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('온 아리송함')
    ax.legend()
    
    # 타고난 것
    ax = axes[1]
    ax.fill_between(x, mean - 2*aleatoric_std, mean + 2*aleatoric_std,
                    alpha=0.3, color='orange', label='타고난 ±2σ')
    ax.plot(x, mean, 'b-')
    if y_true is not None:
        ax.plot(x, y_true, 'k--')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5)
    ax.set_xlabel('x')
    ax.set_title('타고난 아리송함')
    
    # 앎의 것
    ax = axes[2]
    ax.fill_between(x, mean - 2*epistemic_std, mean + 2*epistemic_std,
                    alpha=0.3, color='green', label='앎의 ±2σ')
    ax.plot(x, mean, 'b-')
    if y_true is not None:
        ax.plot(x, y_true, 'k--')
    if x_train is not None:
        ax.scatter(x_train, y_train, c='red', s=20, zorder=5)
    ax.set_xlabel('x')
    ax.set_title('앎의 아리송함')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "미더움 그림"
):
    """
    눈금 맞음 자를 곁들여 미더움 그림을 그린다.
    """
    bin_centers, bin_accs, bin_confs = reliability_diagram_data(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 미더움 그림
    ax = axes[0]
    valid = ~np.isnan(bin_accs)
    
    ax.bar(bin_centers[valid], bin_accs[valid], width=0.08, alpha=0.7, 
           edgecolor='black', label='맞음')
    ax.plot([0, 1], [0, 1], 'k--', label='온전한 눈금 맞음')
    ax.set_xlabel('자신함')
    ax.set_ylabel('맞음')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.legend()
    
    # 자신함 잦기 그림
    ax = axes[1]
    if y_prob.ndim == 1:
        confidences = y_prob
    else:
        confidences = np.max(y_prob, axis=1)
    
    ax.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor='black')
    ax.set_xlabel('자신함')
    ax.set_ylabel('셈')
    ax.set_title('자신함의 분포')
    
    plt.tight_layout()
    plt.show()


def plot_ood_detection(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    title: str = "밖 분포 알아내기"
):
    """
    밖 분포 알아내기의 됨됨이를 그린다.
    """
    try:
        metrics = ood_detection_metrics(in_scores, out_scores)
    except ImportError:
        print("ROC 자에는 sklearn이 있어야 한다")
        metrics = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 점수의 분포
    ax = axes[0]
    ax.hist(in_scores, bins=50, alpha=0.7, label='분포 안', density=True)
    ax.hist(out_scores, bins=50, alpha=0.7, label='밖 분포', density=True)
    ax.set_xlabel('아리송함 점수')
    ax.set_ylabel('밀도')
    ax.set_title('점수의 분포')
    ax.legend()
    
    # ROC 굽이
    ax = axes[1]
    if metrics:
        from sklearn.metrics import roc_curve
        y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
        y_score = np.concatenate([in_scores, out_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'AUROC = {metrics["auroc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', label='아무렇게나')
        ax.set_xlabel('헛 맞음 비율')
        ax.set_ylabel('참 맞음 비율')
        ax.set_title('ROC 굽이')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'sklearn이 있어야 한다', ha='center', va='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    if metrics:
        print(f"\n밖 분포 알아내기 자:")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  FPR@95: {metrics['fpr_at_95']:.4f}")


# =============================================================================
# 보여 주는 함수
# =============================================================================

def demo_uncertainty_decomposition():
    """되돌이에서 아리송함 쪼개기를 보여 준다."""
    
    print("=" * 70)
    print("아리송함 쪼개기: 되돌이")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 다른 흩어짐 자료를 만든다
    n_train = 20
    x_train = np.sort(np.random.uniform(-3, 3, n_train))
    noise_std = 0.1 + 0.2 * np.abs(x_train)  # 다른 흩어짐 잡음
    y_train = np.sin(x_train) + np.random.normal(0, noise_std)
    
    # 시험 점
    x_test = np.linspace(-5, 5, 200)
    y_true = np.sin(x_test)
    
    # 모둠 미루어 봄을 흉내 낸다(BNN 어림을 줄여 적음)
    n_samples = 100
    predictions = np.zeros((n_samples, len(x_test)))
    variances = np.zeros((n_samples, len(x_test)))
    
    for s in range(n_samples):
        # 흔든 매개변수(뒷분포 표본을 흉내 냄)
        a = 1.0 + np.random.normal(0, 0.1)
        b = np.random.normal(0, 0.2)
        
        predictions[s] = a * np.sin(x_test + b)
        
        # 미루어 본 타고난 흩어짐
        variances[s] = (0.1 + 0.2 * np.abs(x_test)) ** 2
    
    # 쪼갬을 셈한다
    mean, total, aleatoric, epistemic = predictive_uncertainty_regression(
        predictions, variances
    )
    
    print(f"\n익힘 점: {n_train}")
    print(f"모둠 표본: {n_samples}")
    
    print("\n--- 아리송함 간추림 ---")
    print(f"평균 타고난 것(자료 자리):  {np.mean(aleatoric[50:150]):.4f}")
    print(f"평균 앎의 것(자료 자리):  {np.mean(epistemic[50:150]):.4f}")
    print(f"평균 앎의 것(밖으로 늘림): {np.mean(epistemic[:30]):.4f}")
    
    print("\n*** 앎의 아리송함은 익힘 자료 밖에서 크다")
    print("*** 타고난 아리송함은 타고난 잡음의 결을 드러낸다")
    
    return x_test, mean, aleatoric, epistemic


def demo_classification_uncertainty():
    """가름에서의 아리송함을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("아리송함 쪼개기: 가름")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_samples = 50  # 모둠 갈래
    n_points = 4
    n_classes = 3
    
    # 여러 형편을 흉내 낸다
    scenarios = {
        'confident_correct': np.tile([5.0, 0.0, 0.0], (n_samples, 1)),
        'confident_wrong': np.tile([0.0, 5.0, 0.0], (n_samples, 1)),
        'high_aleatoric': np.tile([0.5, 0.5, 0.0], (n_samples, 1)),
        'high_epistemic': np.random.randn(n_samples, n_classes) * 2
    }
    
    print("\n--- 형편 살피기 ---")
    print(f"{'형편':<20} {'온 H':>10} {'타고난':>10} {'앎의':>10}")
    print("-" * 55)
    
    for name, logits in scenarios.items():
        # 함수에 맞게 꼴을 바꾼다
        logits_shaped = logits.reshape(n_samples, 1, n_classes)
        
        mean_probs, total, aleatoric, epistemic = predictive_uncertainty_classification(
            logits_shaped
        )
        
        print(f"{name:<20} {total[0]:>10.4f} {aleatoric[0]:>10.4f} {epistemic[0]:>10.4f}")
    
    print("\n*** 자신하는 미루어 봄은 온 아리송함이 작다")
    print("*** 타고난 것이 크다 = 갈래가 타고나게 겹친다")
    print("*** 앎의 것이 크다 = 모형끼리 어긋난다(모둠의 흩어짐)")


def demo_calibration():
    """눈금 맞음 자와 미더움 그림을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("눈금 맞음 살피기")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1000
    
    # 눈금이 잘 맞은 미루어 봄을 만든다
    true_probs = np.random.uniform(0, 1, n)
    y_true_calibrated = (np.random.uniform(0, 1, n) < true_probs).astype(int)
    y_prob_calibrated = true_probs + np.random.normal(0, 0.05, n)
    y_prob_calibrated = np.clip(y_prob_calibrated, 0.01, 0.99)
    
    # 지나치게 자신하는 미루어 봄을 만든다
    y_prob_overconfident = np.where(
        y_prob_calibrated > 0.5,
        0.5 + (y_prob_calibrated - 0.5) * 1.5,
        0.5 - (0.5 - y_prob_calibrated) * 1.5
    )
    y_prob_overconfident = np.clip(y_prob_overconfident, 0.01, 0.99)
    
    # 자를 셈한다
    ece_calib = expected_calibration_error(y_true_calibrated, y_prob_calibrated)
    ece_over = expected_calibration_error(y_true_calibrated, y_prob_overconfident)
    
    brier_calib = brier_score(y_true_calibrated, y_prob_calibrated)
    brier_over = brier_score(y_true_calibrated, y_prob_overconfident)
    
    print("\n--- 눈금 맞음 자 ---")
    print(f"{'모형':<20} {'ECE':>10} {'브라이어':>10}")
    print("-" * 45)
    print(f"{'눈금 잘 맞음':<20} {ece_calib:>10.4f} {brier_calib:>10.4f}")
    print(f"{'지나친 자신함':<20} {ece_over:>10.4f} {brier_over:>10.4f}")
    
    print("\n*** ECE가 작을수록 눈금이 잘 맞는다")
    print("*** 브라이어 점수가 작을수록 낌새 미루어 봄이 좋다")


def demo_ood_detection():
    """밖 분포 알아내기를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("밖 분포 알아내기")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_samples = 50  # 모둠 갈래
    n_in = 500
    n_out = 500
    n_classes = 10
    
    # 분포 안: 자신하는 미루어 봄
    logits_in = np.random.randn(n_samples, n_in, n_classes)
    # 잣대를 키우고 갈래마다 치우침을 더해 자신하게 만든다
    confident_class = np.random.randint(0, n_classes, n_in)
    for i in range(n_in):
        logits_in[:, i, confident_class[i]] += 3.0
    
    # 밖 분포: 아리송한 미루어 봄(모둠이 어긋난다)
    logits_out = np.random.randn(n_samples, n_out, n_classes) * 0.5
    
    # 앎의 아리송함을 셈한다(서로 나눈 소식)
    _, _, _, epistemic_in = predictive_uncertainty_classification(logits_in)
    _, _, _, epistemic_out = predictive_uncertainty_classification(logits_out)
    
    print("\n--- 아리송함 자 ---")
    print(f"분포 안 앎의 것:  평균={np.mean(epistemic_in):.4f}, "
          f"std={np.std(epistemic_in):.4f}")
    print(f"밖 분포 앎의 것:              평균={np.mean(epistemic_out):.4f}, "
          f"std={np.std(epistemic_out):.4f}")
    
    # 알아내기 자를 셈한다
    try:
        metrics = ood_detection_metrics(epistemic_in, epistemic_out)
        print("\n--- 알아내기 됨됨이(앎의 아리송함으로) ---")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"FPR@95: {metrics['fpr_at_95']:.4f}")
    except ImportError:
        print("\n(자세한 자에는 sklearn이 있어야 한다)")
    
    print("\n*** 앎의 아리송함이 클수록 밖 분포 보기임을 뜻한다")
    print("*** 베이즈 방법은 밖 분포 알아내는 힘을 절로 준다")


def demo_heteroscedastic_loss():
    """다른 흩어짐 잃음 셈하기를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("다른 흩어짐 가우스 잃음")
    print("=" * 70)
    
    np.random.seed(42)
    
    loss_fn = HeteroscedasticLoss()
    
    # 형편 1: 좋은 미루어 봄에 맞는 아리송함
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu_good = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    log_var_good = np.log(np.array([0.1, 0.1, 0.1, 0.1, 0.1]))
    
    loss_good = loss_fn(y_true, mu_good, log_var_good)
    
    # 형편 2: 좋은 미루어 봄에 지나치게 큰 아리송함
    log_var_overest = np.log(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    loss_overest = loss_fn(y_true, mu_good, log_var_overest)
    
    # 형편 3: 나쁜 미루어 봄에 지나치게 작은 아리송함
    mu_bad = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    log_var_underest = np.log(np.array([0.01, 0.01, 0.01, 0.01, 0.01]))
    loss_underest = loss_fn(y_true, mu_bad, log_var_underest)
    
    # 형편 4: 나쁜 미루어 봄에 큰 아리송함(어긋남을 솔직히 밝힘)
    log_var_honest = np.log(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
    loss_honest = loss_fn(y_true, mu_bad, log_var_honest)
    
    print("\n--- 잃음 견주기 ---")
    print(f"{'형편':<45} {'잃음':>10}")
    print("-" * 60)
    print(f"{'좋은 미루어 봄, 맞는 아리송함':<45} {loss_good:>10.4f}")
    print(f"{'좋은 미루어 봄, 지나치게 큰 아리송함':<45} {loss_overest:>10.4f}")
    print(f"{'나쁜 미루어 봄, 지나치게 작은 아리송함':<45} {loss_underest:>10.4f}")
    print(f"{'나쁜 미루어 봄, 솔직한 큰 아리송함':<45} {loss_honest:>10.4f}")
    
    print("\n*** 잃음은 미루어 봄의 어긋남과 눈금 어긋난 아리송함을 함께 벌한다")
    print("*** 틀린 미루어 봄에서 아리송함을 낮게 보면 크게 벌한다")


if __name__ == "__main__":
    demo_uncertainty_decomposition()
    demo_classification_uncertainty()
    demo_calibration()
    demo_ood_detection()
    demo_heteroscedastic_loss()
```

---

## 간추림

### 아리송함의 갈래

| 갈래 | 뜻매김 | 밑동 | 줄일 수 있나? |
|------|------------|--------|------------|
| **타고난** | 자료에 타고난 잡음 | 재기, 확률성 | 아니다 |
| **앎의** | 모형이 모름 | 자료가 적음, 매개변수 | 그렇다 |
| **온** | 타고난 + 앎의 | 두 밑동을 아우름 | 얼마쯤 |

### 수학으로 쪼개기

**되돌이**(온 흩어짐 법칙):

$$
\text{Var}[y^* \mid x^*, \mathcal{D}] = \underbrace{\mathbb{E}_\theta[\sigma^2_\theta(x^*)]}_{\text{Aleatoric}} + \underbrace{\text{Var}_\theta[\mu_\theta(x^*)]}_{\text{Epistemic}}
$$

**가름**(서로 나눈 소식):

$$
\underbrace{\mathbb{H}[\bar{p}]}_{\text{Total}} = \underbrace{\mathbb{I}[y; \theta]}_{\text{Epistemic}} + \underbrace{\mathbb{E}_\theta[\mathbb{H}[p_\theta]]}_{\text{Aleatoric}}
$$

### 뒷분포 표본에서 어림하기

| 값 | 식 |
|----------|---------|
| 미루어 본 평균 | $\hat{\mu} = \frac{1}{S}\sum_s \mu_{\theta^{(s)}}$ |
| 앎의 것 | $\frac{1}{S}\sum_s (\mu_{\theta^{(s)}} - \hat{\mu})^2$ |
| 타고난 것 | $\frac{1}{S}\sum_s \sigma^2_{\theta^{(s)}}$ |
| 온 것 | 앎의 것 + 타고난 것 |

### 눈금 맞음 자

| 자 | 식 | 풀이 |
|--------|---------|----------------|
| **ECE** | $\sum_m \frac{|B_m|}{n}\|acc_m - conf_m\|$ | 바라는 눈금 맞음 어긋남 |
| **MCE** | $\max_m \|acc_m - conf_m\|$ | 가장 나쁜 자리의 눈금 맞음 |
| **브라이어** | $\frac{1}{n}\sum_i (p_i - y_i)^2$ | 낌새의 맞음 |

### 쓸 자리

| 쓰임 | 고갱이 아리송함 | 쓰는 길 |
|-------------|-----------------|-------|
| **살아 있는 배움** | 앎의 것(서로 나눈 소식/BALD) | 알려 주는 바가 큰 점 고르기 |
| **밖 분포 알아내기** | 앎의 것 | 낯선 들임에 표시하기 |
| **무릅씀 따지기** | 온 것 | 판단의 자신함 |
| **모형 벌레잡기** | 둘 다 | 어그러지는 결 짚어내기 |

### 다른 장과의 이어짐

| 이야기 | 장 | 이어짐 |
|-------|---------|------------|
| 앞선 분포 정하기 | 13장: 짐의 앞선 분포 | 앎의 아리송함에 걸린다 |
| 뒷분포 미루어 봄 | 13장: 뒷분포 미루어 봄 | 매개변수 표본의 밑동 |
| MC 드롭아웃 | 13장: MC 드롭아웃 | 앎의 아리송함의 어림 |
| 변이 베이즈 신경 그물 | 13장: 변이 베이즈 신경 그물 | 크게 늘릴 수 있는 아리송함 어림 |
| 모형 견주기 | 13장: 소식 잣대 | 모형 고르기의 아리송함 |

### 고갱이 살펴볼 거리

- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.
- Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Houlsby, N., et al. (2011). Bayesian active learning for classification and preference learning. *arXiv*.

## 익힘 문제

**익힘 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "익힘 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**익힘 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "익힘 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**익힘 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "익힘 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**익힘 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "익힘 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$
