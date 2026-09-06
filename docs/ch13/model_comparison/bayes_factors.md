# 베이즈 인자
**베이즈 인자**는 모형 증거의 비로, 데이터가 한 모형을 다른 모형보다 얼마나 더 받쳐 주는지를 원칙 있게 잰다. p값과 달리 베이즈 인자는 영가설에 맞서는 증거만이 아니라 어떤 가설을 편드는 증거도 수로 나타내며, 베이즈판 오컴의 면도날로 모형의 복잡도를 자연스럽게 셈에 넣는다.

---

## 왜 필요한가: 영가설 검정 너머로

### 고전적 가설 검정의 한계

고전적 가설 검정에는 근본 한계가 있다.

**비대칭**: 영가설을 기각하거나 기각하지 못할 뿐이다

- p값은 $P(\text{이만큼 극단적인 데이터} \mid H_0)$을 잰다
- $H_0$을 *편드는* 증거는 잴 수 없다
- "유의하지 않음" ≠ "영가설이 참임"

**표본 크기에 민감함**: 데이터가 넉넉하면 어떤 영가설이든 기각된다

- 하찮은 효과도 "유의해진다"
- 실질적 중요성과 통계적 유의성이 뒤섞인다

**증거의 세기를 수로 나타내지 못함**: 이분법적 결정만 있다

- $p = 0.049$과 $p = 0.0001$ 모두 "기각"으로 이어진다
- $p = 0.051$과 $p = 0.5$ 모두 "기각하지 못함"으로 이어진다

### 베이즈 인자가 주는 것

베이즈 인자는 이런 한계를 다룬다.

1. **대칭적인 견줌**: 어느 가설이든 편드는 증거
2. **이어진 재기**: 증거의 세기를 수로 나타낸다
3. **저절로 작동하는 오컴의 면도날**: 쓸데없는 복잡함에 벌을 준다
4. **앞뒤 맞는 확률 셈법**: 베이즈 정리로 믿음을 고친다

---

## 정의와 풀이

### 형식적 정의

모형 $\mathcal{M}_1$과 모형 $\mathcal{M}_2$을 견주는 **베이즈 인자**는 다음과 같다.

$$
\boxed{B_{12} = \frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}}
$$

여기서 $p(\mathcal{D} \mid \mathcal{M}_k)$은 **모형 증거**(주변 가능도)이다.

$$
p(\mathcal{D} \mid \mathcal{M}_k) = \int p(\mathcal{D} \mid \theta_k, \mathcal{M}_k) \, p(\theta_k \mid \mathcal{M}_k) \, d\theta_k
$$

### 뒤확률 승산과의 관계

베이즈 인자는 모형에 대한 앞선 믿음과 뒤의 믿음을 잇는다.

$$
\underbrace{\frac{p(\mathcal{M}_1 \mid \mathcal{D})}{p(\mathcal{M}_2 \mid \mathcal{D})}}_{\text{Posterior odds}} = \underbrace{B_{12}}_{\text{Bayes factor}} \times \underbrace{\frac{p(\mathcal{M}_1)}{p(\mathcal{M}_2)}}_{\text{Prior odds}}
$$

**핵심 통찰**: 베이즈 인자는 데이터가 우리의 상대적 믿음을 고치는 배수이다.

### 로그 베이즈 인자

수치 안정성을 위해 로그 베이즈 인자로 다룬다.

$$
\log B_{12} = \log p(\mathcal{D} \mid \mathcal{M}_1) - \log p(\mathcal{D} \mid \mathcal{M}_2)
$$

**성질**:

- $\log B_{12} = -\log B_{21}$(반대칭)
- $\log B_{13} = \log B_{12} + \log B_{23}$(추이적)
- 서로 독립인 데이터셋에 걸쳐 더해진다: $\log B_{12}^{\text{total}} = \sum_i \log B_{12}^{(i)}$

---

## 풀이 지침

### 카스와 라프터리(1995)의 눈금

널리 쓰이는 풀이 눈금이다.

| $\log_{10} B_{12}$ | $\log B_{12}$ | $B_{12}$ | $\mathcal{M}_1$을 편드는 증거 |
|-------------------|---------------|----------|------------------------------|
| 0 ~ 0.5 | 0 ~ 1.15 | 1 ~ 3.2 | 말할 값어치가 거의 없음 |
| 0.5 ~ 1 | 1.15 ~ 2.3 | 3.2 ~ 10 | 뚜렷함 |
| 1 ~ 2 | 2.3 ~ 4.6 | 10 ~ 100 | 강함 |
| > 2 | > 4.6 | > 100 | 결정적임 |

**대칭적인 풀이**: $B_{12} = 0.1$은 $\mathcal{M}_2$을 강하게 편드는 증거라는 뜻이다.

### 제프리스(1961)의 눈금

해럴드 제프리스의 본디 눈금이다($\log_{10}$을 쓴다).

| $\log_{10} B_{12}$ | 풀이 |
|-------------------|----------------|
| 0 | 증거 없음 |
| 0 ~ 0.5 | 말할 값어치가 거의 없음 |
| 0.5 ~ 1 | 뚜렷함 |
| 1 ~ 1.5 | 강함 |
| 1.5 ~ 2 | 매우 강함 |
| > 2 | 결정적임 |

### 뒤확률로 바꾸기

앞선 승산이 같으면 다음과 같다.

$$
p(\mathcal{M}_1 \mid \mathcal{D}) = \frac{B_{12}}{1 + B_{12}} = \frac{1}{1 + B_{21}}
$$

| $B_{12}$ | $p(\mathcal{M}_1 \mid \mathcal{D})$ |
|----------|-------------------------------------|
| 1 | 0.50 |
| 3 | 0.75 |
| 10 | 0.91 |
| 100 | 0.99 |
| 1000 | 0.999 |

### 풀이할 때 조심할 점

1. **가설 검정이 아니다**: 베이즈 인자는 "유의성"이 아니라 상대적 증거를 잰다
2. **눈금은 제멋대로이다**: 여러 눈금이 있으니 견주어 풀이하라
3. **앞확률에 기댄다**: (뒤확률 추론과 달리) 앞확률을 어떻게 정하느냐에 크게 기댄다
4. **모형이 알맞은가**: 베이즈 인자가 크다고 그 모형이 좋다는 뜻은 아니며, 대안보다 낫다는 뜻일 뿐이다

---

## 수학적 성질

### 대칭성과 추이성

**역수 관계**:

$$
B_{21} = \frac{1}{B_{12}}
$$

**추이성**(쌍으로 견줄 때):

$$
B_{13} = B_{12} \cdot B_{23}
$$

### 로그 증거의 더해짐

서로 독립인 데이터셋 $\mathcal{D}_1, \mathcal{D}_2$에 대해 다음과 같다.

$$
\log B_{12}^{\text{total}} = \log B_{12}^{(\mathcal{D}_1)} + \log B_{12}^{(\mathcal{D}_2)}
$$

이로써 모형 견줌을 차례로 고칠 수 있다.

### 일치성

규칙성 조건 아래에서 $\mathcal{M}_1$이 참이면 다음과 같다.

$$
\log B_{12} \xrightarrow{p} +\infty \quad \text{as } n \to \infty
$$

$\mathcal{M}_2$이 참이면 다음과 같다.

$$
\log B_{12} \xrightarrow{p} -\infty \quad \text{as } n \to \infty
$$

베이즈 인자는 **한결같다**. 끝내 참 모형을 편들게 된다.

### 점근 거동

매개변수가 $k$개 더 있는 겹친 모형에서는 다음과 같다.

$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) - \frac{k}{2} \log n + O(1)
$$

이는 BIC 어림과 이어진다.

---

## 흔한 검정의 베이즈 인자

### 점 영가설과 대립가설

**$H_0: \theta = \theta_0$과 $H_1: \theta \neq \theta_0$을 검정하기**

$H_0$ 아래: $p(\mathcal{D} \mid H_0) = p(\mathcal{D} \mid \theta_0)$

$H_1$ 아래: $p(\mathcal{D} \mid H_1) = \int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta$

**베이즈 인자**:

$$
B_{01} = \frac{p(\mathcal{D} \mid \theta_0)}{\int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta}
$$

### 새비지-디키 밀도비

$H_0: \theta = \theta_0$이 $H_1$의 특별한 경우인 겹친 모형에서는 다음과 같다.

$$
B_{01} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)}
$$

**풀이**: 영가설 값에서 뒤확률 밀도와 앞확률 밀도의 비이다.

**끌어내기**:

$$
B_{01} = \frac{p(\mathcal{D} \mid H_0)}{p(\mathcal{D} \mid H_1)} = \frac{p(\mathcal{D} \mid \theta_0)}{\int p(\mathcal{D} \mid \theta) \, p(\theta \mid H_1) \, d\theta}
$$

$H_1$에 베이즈 정리를 쓰면 다음과 같다.

$$
p(\theta_0 \mid \mathcal{D}, H_1) = \frac{p(\mathcal{D} \mid \theta_0) \, p(\theta_0 \mid H_1)}{p(\mathcal{D} \mid H_1)}
$$

정리하면 다음과 같다.

$$
\frac{p(\mathcal{D} \mid \theta_0)}{p(\mathcal{D} \mid H_1)} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)} = B_{01}
$$

### 두 표본 견줌

**평균이 같은지 검정하기**: $H_0: \mu_1 = \mu_2$과 $H_1: \mu_1 \neq \mu_2$

흩어짐을 아는 가우스 데이터에서는 베이즈 인자가 닫힌 꼴이다. 흩어짐을 모르면 수치 적분이나 어림이 필요하다.

### 분산분석: 여러 무리 견줌

**모두 같은지 검정하기**: $H_0: \mu_1 = \mu_2 = \cdots = \mu_K$

베이즈 인자는 다음을 견준다.

- $\mathcal{M}_0$: 모든 무리에 평균 하나
- $\mathcal{M}_1$: 무리마다 따로 평균

중간 모형(일부 무리만 같은 경우)에는 쌍으로 견주기가 필요하다.

---

## 린들리의 역설

### 역설의 진술

**린들리의 역설**(린들리, 1957): 유의수준 $\alpha$을 붙박고 표본 크기 $n$이 클 때 결과가 다음과 같을 수 있다.

- 통계적으로 유의하다(p값 $< \alpha$)
- 그런데도 베이즈 인자는 영가설을 강하게 편든다

### 수학으로 보이기

$\bar{x} \sim \mathcal{N}(\mu, \sigma^2/n)$에 대해 $H_0: \mu = 0$과 $H_1: \mu \neq 0$을 검정한다고 하자.

**p값 방법**: $|\bar{x}| > z_{\alpha/2} \cdot \sigma/\sqrt{n}$이면 기각한다

$H_1$ 아래 $\mu \sim \mathcal{N}(0, \tau^2)$일 때의 **베이즈 인자**:

$$
B_{01} = \sqrt{1 + n\tau^2/\sigma^2} \cdot \exp\left(-\frac{n\bar{x}^2}{2\sigma^2} \cdot \frac{n\tau^2/\sigma^2}{1 + n\tau^2/\sigma^2}\right)
$$

$\bar{x}$을 유의성 경계($|\bar{x}| = z_{\alpha/2} \cdot \sigma/\sqrt{n}$)에 붙박으면 다음과 같다.

$n \to \infty$이면 p값은 $\alpha$에 머물지만 $B_{01} \to \infty$이다(영가설을 편든다)!

### 풀림

이 역설이 생기는 까닭은 다음과 같다.

1. 잡음에 대한 효과 크기($|\bar{x}|/(\sigma/\sqrt{n})$)를 붙박아 둔다
2. $n \to \infty$이면 이는 실제 효과 $|\bar{x}|$이 0으로 오그라든다는 뜻이다
3. 베이즈 인자는 그 효과가 사라질 만큼 작다는 것을 옳게 알아챈다

**교훈**: 베이즈 인자와 p값은 서로 다른 물음에 답한다. 베이즈 인자는 "모형마다 이 데이터가 얼마나 그럴듯한가?"를 묻지, "영가설을 놓았을 때 이 데이터가 얼마나 극단적인가?"를 묻지 않는다.

---

## 앞확률을 어떻게 정하느냐에 대한 민감도

### 앞확률에 기댐

뒤확률 추론과 달리 베이즈 인자는 앞확률을 어떻게 정하느냐에 **몹시 민감하다**.

$$
B_{12} = \frac{\int p(\mathcal{D} \mid \theta_1) \, p(\theta_1 \mid \mathcal{M}_1) \, d\theta_1}{\int p(\mathcal{D} \mid \theta_2) \, p(\theta_2 \mid \mathcal{M}_2) \, d\theta_2}
$$

앞확률 $p(\theta_k \mid \mathcal{M}_k)$이 분자와 분모를 모두 곧바로 좌우한다.

### 제대로 되지 않은 앞확률

**결정적인 문제**: 제대로 되지 않은 앞확률에서는 베이즈 인자가 정의되지 않는다!

$\int p(\theta_k) \, d\theta_k = \infty$이면 $p(\mathcal{D} \mid \mathcal{M}_k)$은 제멋대로인 상수 배까지만 정해진다.

**결과**: 모형 견줌에 제대로 되지 않은 앞확률을 결코 쓰지 마라.

### 흐릿한 앞확률에서의 제프리스-린들리 역설

아주 퍼진 앞확률(흩어짐이 큰 것)을 쓰면 대립가설이 부당하게 벌을 받는다.

$$
p(\theta \mid H_1) = \mathcal{N}(0, 10^6) \quad \text{(very vague)}
$$

이 앞확률은 그럴듯한 효과 크기에 하찮은 확률만 주므로, 효과가 강해도 $H_0$을 편들게 된다.

### 권하는 버릇

**1. 기본 베이즈 인자**: 자리 잡힌 원칙 있는 앞확률을 쓰라

- $t$ 검정에는 JZS(제프리스-젤너-시오우)
- 거친 어림으로는 BIC

**2. 앞확률 민감도 분석**:

- 그럴듯한 앞확률 여럿에서 베이즈 인자를 셈하라
- 결론의 범위를 알려라

**3. 데이터에 기댄 앞확률**(조심해서 쓰라):

- 분수 베이즈 인자
- 내재 베이즈 인자
- 뒤확률 베이즈 인자

---

## 기본 베이즈 인자와 객관적 베이즈 인자

### JZS 베이즈 인자

$H_0: \delta = 0$과 $H_1: \delta \neq 0$(고른 효과 크기)을 견줄 때 쓴다.

**$H_1$ 아래의 앞확률**: 코시 앞확률(기본 눈금 $r = \sqrt{2}/2$)

$$
\delta \mid H_1 \sim \text{Cauchy}(0, r)
$$

**성질**:

- 두꺼운 꼬리가 큰 효과를 허용한다
- 눈금 매개변수 $r$이 기대 효과 크기를 다스린다
- 심리학 연구에 눈금이 잘 맞는다

### 내재 베이즈 인자

데이터의 일부로 제대로 된 참조 앞확률을 짓는다.

$$
B_{12}^I = \frac{p(\mathcal{D}^{(-)} \mid \mathcal{D}^{(*)}, \mathcal{M}_1)}{p(\mathcal{D}^{(-)} \mid \mathcal{D}^{(*)}, \mathcal{M}_2)}
$$

여기서 $\mathcal{D}^{(*)}$은 "학습" 부분집합이고 $\mathcal{D}^{(-)}$은 나머지이다.

### 분수 베이즈 인자

가능도의 일부로 앞확률을 정한다.

$$
p_b(\theta \mid \mathcal{M}) \propto p(\mathcal{D} \mid \theta, \mathcal{M})^b \, p_0(\theta \mid \mathcal{M})
$$

여기서 $b < 1$은 그 비율이고 $p_0$은 본디 앞확률이다.

**이점**: 앞확률을 어떻게 정하느냐에 대한 민감도를 줄인다.

---

## 여러 모형 견줌

### 모형의 뒤확률

앞확률이 $p(\mathcal{M}_k)$인 모형 $\mathcal{M}_1, \ldots, \mathcal{M}_K$ $K$개에 대해 다음과 같다.

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathcal{M}_k) \, p(\mathcal{M}_k)}{\sum_{j=1}^K p(\mathcal{D} \mid \mathcal{M}_j) \, p(\mathcal{M}_j)}
$$

### 베이즈 인자에서 뒤확률 셈하기

앞확률이 $p(\mathcal{M}_k) = 1/K$으로 같으면 다음과 같다.

$$
p(\mathcal{M}_k \mid \mathcal{D}) = \frac{B_{k1}}{\sum_{j=1}^K B_{j1}}
$$

여기서 $B_{j1}$은 모형 $j$을 모형 1(참조로 삼는 아무 모형)과 견준 베이즈 인자이다.

### 모형 평균 내기

모형 하나를 고르는 대신 예측을 평균 낸다.

$$
p(y^* \mid x^*, \mathcal{D}) = \sum_{k=1}^K p(y^* \mid x^*, \mathcal{D}, \mathcal{M}_k) \, p(\mathcal{M}_k \mid \mathcal{D})
$$

**좋은 점**:

- 모형의 아리송함을 셈에 넣는다
- 더 튼튼한 예측
- 눈금이 더 잘 맞을 때가 많다

### 쌍으로 견주기와 한꺼번에 견주기

**쌍으로**: 모형을 둘씩 견준다

- 셈하기 더 단순하다
- 추이성: $B_{13} = B_{12} \cdot B_{23}$

**한꺼번에**: 모든 모형을 한 번에 견준다

- 모든 모형에 대해 $p(\mathcal{M}_k)$을 정해야 한다
- 모형 평균 내기에는 더 원칙에 맞는다

---

## 셈하는 방법

### 정확한 셈(켤레 모형)

켤레 모형에서는 고르는 상수의 비를 쓴다.

$$
B_{12} = \frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_2)}
$$

증거마다 해석적으로 셈한다.

### 겹친 모형을 위한 새비지-디키

$H_1$ 안에서 $H_0: \theta = \theta_0$을 검정할 때 쓴다.

$$
B_{01} = \frac{p(\theta_0 \mid \mathcal{D}, H_1)}{p(\theta_0 \mid H_1)}
$$

한 점에서의 뒤확률 밀도와 앞확률 밀도만 있으면 된다.

### 다리 표집

두 뒤확률에서 뽑은 표본으로 베이즈 인자를 어림한다.

$$
\hat{B}_{12} = \frac{\frac{1}{n_2} \sum_{j=1}^{n_2} h(\theta_2^{(j)}) \, p(\mathcal{D} \mid \theta_2^{(j)}, \mathcal{M}_1) \, p(\theta_2^{(j)} \mid \mathcal{M}_1)}{\frac{1}{n_1} \sum_{i=1}^{n_1} h(\theta_1^{(i)}) \, p(\mathcal{D} \mid \theta_1^{(i)}, \mathcal{M}_2) \, p(\theta_1^{(i)} \mid \mathcal{M}_2)}
$$

여기서 $h(\cdot)$은 가장 좋은 다리 함수이다.

### 열역학 적분

모형 1에서 모형 2로 가는 길을 쓴다.

$$
p_t(\theta) \propto p(\theta \mid \mathcal{M}_1)^{1-t} \, p(\theta \mid \mathcal{M}_2)^t \, p(\mathcal{D} \mid \theta)
$$

로그 베이즈 인자는 다음과 같다.

$$
\log B_{12} = \int_0^1 \mathbb{E}_{p_t}\left[\log \frac{p(\theta \mid \mathcal{M}_1)}{p(\theta \mid \mathcal{M}_2)}\right] dt
$$

### 어림 방법

**라플라스 어림**:

$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) + \log p(\hat{\theta}_1) - \log p(\hat{\theta}_2) + \frac{d_1 - d_2}{2}\log(2\pi) - \frac{1}{2}\log\frac{|H_1|}{|H_2|}
$$

**BIC 어림**:

$$
\log B_{12} \approx \log p(\mathcal{D} \mid \hat{\theta}_1) - \log p(\mathcal{D} \mid \hat{\theta}_2) - \frac{d_1 - d_2}{2}\log n
$$

---

## 파이썬 구현

```python
"""
베이즈 인자: 온전한 구현

이 모듈은 모형 견줌을 위한 베이즈 인자의 셈하기와 풀이를 주며,
켤레 모형의 정확한 방법, 어림, 그리고 그려 보기 도구를
함께 담는다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.integrate import quad
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# 풀이 도구
# =============================================================================

def interpret_bayes_factor(log_bf: float, scale: str = 'kass_raftery') -> str:
    """
    표준 눈금으로 로그 베이즈 인자를 풀이한다.
    
    매개변수
    ----------
    log_bf : float
        베이즈 인자의 자연로그(ln B_12)
    scale : str
        'kass_raftery' 또는 'jeffreys'
    
    반환값
    -------
    str
        증거 세기의 풀이
    """
    # 표준 눈금을 위해 log10으로 바꾸기
    log10_bf = log_bf / np.log(10)
    
    if scale == 'kass_raftery':
        if log10_bf > 2:
            return f"Decisive evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 1:
            return f"Strong evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 0.5:
            return f"Substantial evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > 0:
            return f"Weak evidence for M1 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -0.5:
            return f"Weak evidence for M2 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -1:
            return f"Substantial evidence for M2 (log10 BF = {log10_bf:.2f})"
        elif log10_bf > -2:
            return f"Strong evidence for M2 (log10 BF = {log10_bf:.2f})"
        else:
            return f"Decisive evidence for M2 (log10 BF = {log10_bf:.2f})"
    
    elif scale == 'jeffreys':
        abs_log = abs(log10_bf)
        direction = "M1" if log10_bf > 0 else "M2"
        
        if abs_log > 2:
            strength = "Decisive"
        elif abs_log > 1.5:
            strength = "Very strong"
        elif abs_log > 1:
            strength = "Strong"
        elif abs_log > 0.5:
            strength = "Substantial"
        else:
            strength = "Barely worth mentioning"
        
        return f"{strength} evidence for {direction} (log10 BF = {log10_bf:.2f})"
    
    else:
        raise ValueError(f"Unknown scale: {scale}")

def log_bf_to_posterior_prob(log_bf: float, prior_odds: float = 1.0) -> float:
    """
    로그 베이즈 인자를 M1의 뒤확률로 바꾼다.
    
    매개변수
    ----------
    log_bf : float
        베이즈 인자 B_12의 자연로그
    prior_odds : float
        앞확률 승산 p(M1)/p(M2), 기본값 1(앞확률이 같음)
    
    반환값
    -------
    float
        뒤확률 p(M1 | D)
    """
    log_posterior_odds = log_bf + np.log(prior_odds)
    # p(M1|D) = 뒤확률_승산 / (1 + 뒤확률_승산)
    #         = 1 / (1 + 1/뒤확률_승산)
    #         = 1 / (1 + exp(-로그_뒤확률_승산))
    return 1.0 / (1.0 + np.exp(-log_posterior_odds))

def posterior_probs_from_log_evidences(
    log_evidences: np.ndarray,
    prior_probs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    로그 증거에서 뒤확률 모형 확률을 셈한다.
    
    매개변수
    ----------
    log_evidences : array-like
        모형마다의 로그 모형 증거
    prior_probs : array-like, 있어도 되고 없어도 됨
        앞확률(따로 정하지 않으면 고름)
    
    반환값
    -------
    ndarray
        뒤확률 모형 확률
    """
    log_evidences = np.asarray(log_evidences)
    K = len(log_evidences)
    
    if prior_probs is None:
        log_priors = np.zeros(K) - np.log(K)  # 고름
    else:
        log_priors = np.log(np.asarray(prior_probs))
    
    log_unnorm = log_evidences + log_priors
    log_norm = logsumexp(log_unnorm)
    
    return np.exp(log_unnorm - log_norm)

# =============================================================================
# 켤레 모형의 정확한 베이즈 인자
# =============================================================================

class BetaBernoulliModel:
    """정확한 증거 셈하기를 갖춘 베타-베르누이 모형."""
    
    def __init__(self, alpha: float, beta: float):
        """
        매개변수
        ----------
        alpha, beta : float
            베타 앞확률 매개변수
        """
        self.alpha0 = alpha
        self.beta0 = beta
    
    def log_evidence(self, data: np.ndarray) -> float:
        """로그 주변 가능도를 셈한다."""
        n = len(data)
        s = np.sum(data)  # 성공
        f = n - s         # 실패
        
        alpha_n = self.alpha0 + s
        beta_n = self.beta0 + f
        
        # log B(alpha_n, beta_n) - log B(alpha_0, beta_0)
        log_ev = (
            gammaln(alpha_n) + gammaln(beta_n) - gammaln(alpha_n + beta_n)
            - gammaln(self.alpha0) - gammaln(self.beta0) + gammaln(self.alpha0 + self.beta0)
        )
        
        return log_ev

def bayes_factor_beta_bernoulli(
    data: np.ndarray,
    model1_params: Tuple[float, float],
    model2_params: Tuple[float, float]
) -> float:
    """
    베타-베르누이 모형 둘의 베이즈 인자를 셈한다.
    
    매개변수
    ----------
    data : ndarray
        이진 관측
    model1_params : tuple
        모형 1의 (alpha, beta)
    model2_params : tuple
        모형 2의 (alpha, beta)
    
    반환값
    -------
    float
        로그 베이즈 인자 log(B_12)
    """
    model1 = BetaBernoulliModel(*model1_params)
    model2 = BetaBernoulliModel(*model2_params)
    
    return model1.log_evidence(data) - model2.log_evidence(data)

class GaussianKnownVarianceModel:
    """흩어짐을 알고 평균에 정규 앞확률을 준 가우스 모형."""
    
    def __init__(self, mu0: float, sigma0_sq: float, sigma_sq: float):
        """
        매개변수
        ----------
        mu0 : float
            앞확률 평균
        sigma0_sq : float
            앞확률 흩어짐
        sigma_sq : float
            아는 자료 흩어짐
        """
        self.mu0 = mu0
        self.sigma0_sq = sigma0_sq
        self.sigma_sq = sigma_sq
    
    def log_evidence(self, data: np.ndarray) -> float:
        """로그 주변 가능도를 셈한다."""
        n = len(data)
        x_bar = np.mean(data)
        
        # 뒤확률 정밀도
        tau0 = 1.0 / self.sigma0_sq
        tau = 1.0 / self.sigma_sq
        tau_n = tau0 + n * tau
        
        # 로그 증거
        log_ev = (
            -0.5 * n * np.log(2 * np.pi * self.sigma_sq)
            + 0.5 * np.log(tau0 / tau_n)
            - 0.5 / self.sigma_sq * np.sum((data - x_bar)**2)
            - 0.5 * tau0 * n * tau / tau_n * (x_bar - self.mu0)**2
        )
        
        return log_ev

def bayes_factor_one_sample_t(
    data: np.ndarray,
    null_value: float = 0.0,
    prior_scale: float = np.sqrt(2) / 2
) -> Tuple[float, float]:
    """
    한 표본 t검정의 JZS 베이즈 인자.
    
    H0: delta = 0(효과 크기가 0)
    H1: delta ~ 코시(0, prior_scale)
    
    매개변수
    ----------
    data : ndarray
        관측 자료
    null_value : float
        귀무가설 아래의 값
    prior_scale : float
        코시 앞확률의 눈금(기본값: sqrt(2)/2)
    
    반환값
    -------
    log_bf_01 : float
        귀무 쪽 로그 베이즈 인자
    log_bf_10 : float
        대립 쪽 로그 베이즈 인자
    """
    n = len(data)
    t_stat = (np.mean(data) - null_value) / (np.std(data, ddof=1) / np.sqrt(n))
    
    # 수치로 적분하기
    def integrand(g):
        """JZS 베이즈 인자의 피적분 함수."""
        if g <= 0:
            return 0
        return (
            (1 + g)**(-0.5)
            * (1 + t_stat**2 / ((1 + n * g) * (n - 1)))**(-(n) / 2)
            * (2 * np.pi)**(-0.5) * g**(-1.5)
            * np.exp(-1 / (2 * g * prior_scale**2))
        )
    
    # 수치 적분
    result, _ = quad(integrand, 0, np.inf)
    
    # 귀무와 견주기(g 없이 t분포만)
    null_density = stats.t.pdf(t_stat, df=n-1)
    
    log_bf_01 = np.log(null_density) - np.log(result) if result > 0 else np.inf
    log_bf_10 = -log_bf_01
    
    return log_bf_01, log_bf_10

# =============================================================================
# 새비지-디키 밀도 비
# =============================================================================

def savage_dickey_ratio(
    posterior_density_at_null: float,
    prior_density_at_null: float
) -> float:
    """
    새비지-디키 밀도 비로 베이즈 인자를 셈한다.
    
    B_01 = p(theta_0 | D, H1) / p(theta_0 | H1)
    
    매개변수
    ----------
    posterior_density_at_null : float
        귀무값에서의 뒤확률 밀도
    prior_density_at_null : float
        귀무값에서의 앞확률 밀도
    
    반환값
    -------
    float
        귀무 쪽 로그 베이즈 인자
    """
    return np.log(posterior_density_at_null) - np.log(prior_density_at_null)

def savage_dickey_gaussian(
    data: np.ndarray,
    null_value: float,
    prior_mean: float,
    prior_var: float,
    known_var: float
) -> float:
    """
    정규 앞확률을 쓴 가우스 모형의 새비지-디키.
    
    H0: mu = null_value 대 H1: mu ~ N(prior_mean, prior_var)을 검정한다
    
    매개변수
    ----------
    data : ndarray
        관측 자료
    null_value : float
        귀무가설 아래의 값
    prior_mean : float
        H1 아래의 앞확률 평균
    prior_var : float
        H1 아래의 앞확률 흩어짐
    known_var : float
        아는 자료 흩어짐
    
    반환값
    -------
    float
        로그 베이즈 인자 B_01
    """
    n = len(data)
    x_bar = np.mean(data)
    
    # 뒤확률 매개변수
    tau0 = 1.0 / prior_var
    tau = 1.0 / known_var
    tau_n = tau0 + n * tau
    mu_n = (tau0 * prior_mean + n * tau * x_bar) / tau_n
    var_n = 1.0 / tau_n
    
    # 귀무값에서의 밀도
    prior_density = stats.norm.pdf(null_value, prior_mean, np.sqrt(prior_var))
    posterior_density = stats.norm.pdf(null_value, mu_n, np.sqrt(var_n))
    
    return np.log(posterior_density) - np.log(prior_density)

# =============================================================================
# 베이즈 인자의 BIC 어림
# =============================================================================

def bic_bayes_factor(
    log_lik1: float,
    log_lik2: float,
    k1: int,
    k2: int,
    n: int
) -> float:
    """
    로그 베이즈 인자의 BIC 어림.
    
    log B_12 ≈ (log L1 - k1/2 * log n) - (log L2 - k2/2 * log n)
    
    매개변수
    ----------
    log_lik1, log_lik2 : float
        최대로 만든 로그 가능도
    k1, k2 : int
        매개변수의 개수
    n : int
        표본 크기
    
    반환값
    -------
    float
        어림 로그 베이즈 인자
    """
    bic1 = -2 * log_lik1 + k1 * np.log(n)
    bic2 = -2 * log_lik2 + k2 * np.log(n)
    
    # log B_12 ≈ -0.5 * (BIC1 - BIC2)
    return -0.5 * (bic1 - bic2)

# =============================================================================
# 린들리의 역설 보여 주기
# =============================================================================

def demonstrate_lindley_paradox(
    effect_size: float = 0.2,
    sample_sizes: np.ndarray = np.array([10, 50, 100, 500, 1000]),
    prior_var: float = 1.0,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    린들리의 역설을 보인다.
    
    매개변수
    ----------
    effect_size : float
        표준화한 효과 크기(코언의 d)
    sample_sizes : ndarray
        살펴볼 표본 크기
    prior_var : float
        H1 아래의 앞확률 흩어짐
    alpha : float
        유의수준
    
    반환값
    -------
    p_values : ndarray
        표본 크기마다의 p값
    bayes_factors : ndarray
        표본 크기마다의 로그 베이즈 인자 B_01
    decisions : ndarray
        고전적 결정(True = 귀무를 물리침)
    """
    np.random.seed(42)
    
    p_values = []
    log_bfs = []
    decisions = []
    
    for n in sample_sizes:
        # 붙박이 효과 크기로 자료 흉내내기
        true_mean = effect_size
        data = np.random.normal(true_mean, 1.0, n)
        
        # 고전 t검정
        t_stat, p_val = stats.ttest_1samp(data, 0)
        p_values.append(p_val)
        decisions.append(p_val < alpha)
        
        # 베이즈 인자(간추림, 흩어짐 = 1을 안다고 놓고)
        x_bar = np.mean(data)
        se = 1.0 / np.sqrt(n)
        
        # 점 귀무에 대한 B_01
        # H0 아래: 가능도는 N(0, 1/n)
        log_lik_h0 = stats.norm.logpdf(x_bar, 0, se)
        
        # N(0, 앞확률_흩어짐) 앞확률을 쓴 H1 아래: 주변 분포는 N(0, 앞확률_흩어짐 + 1/n)
        marginal_var = prior_var + se**2
        log_lik_h1 = stats.norm.logpdf(x_bar, 0, np.sqrt(marginal_var))
        
        log_bf_01 = log_lik_h0 - log_lik_h1
        log_bfs.append(log_bf_01)
    
    return np.array(p_values), np.array(log_bfs), np.array(decisions)

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_bayes_factor_interpretation(log_bfs: np.ndarray, labels: List[str]) -> plt.Figure:
    """
    풀이 구역과 함께 베이즈 인자를 그려 본다.
    
    매개변수
    ----------
    log_bfs : ndarray
        로그 베이즈 인자(자연로그)
    labels : list
        견줌마다의 이름표
    
    반환값
    -------
    그림
        Matplotlib 그림
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # log10으로 바꾸기
    log10_bfs = log_bfs / np.log(10)
    
    # 바탕 구역(캐스-래프터리 눈금)
    regions = [
        (-np.inf, -2, 'Decisive for M2', '#d62728', 0.2),
        (-2, -1, 'Strong for M2', '#ff7f0e', 0.2),
        (-1, -0.5, 'Substantial for M2', '#ffbb78', 0.2),
        (-0.5, 0.5, 'Inconclusive', '#d3d3d3', 0.3),
        (0.5, 1, 'Substantial for M1', '#98df8a', 0.2),
        (1, 2, 'Strong for M1', '#2ca02c', 0.2),
        (2, np.inf, 'Decisive for M1', '#1f77b4', 0.2),
    ]
    
    y_range = len(labels)
    for low, high, label, color, alpha in regions:
        low_plot = max(low, -4)
        high_plot = min(high, 4)
        ax.axvspan(low_plot, high_plot, alpha=alpha, color=color, label=label)
    
    # 베이즈 인자 그리기
    y_pos = np.arange(len(labels))
    colors = ['#2ca02c' if bf > 0 else '#d62728' for bf in log10_bfs]
    
    ax.barh(y_pos, log10_bfs, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('log₁₀ Bayes Factor', fontsize=12)
    ax.set_title('Bayes Factor Comparison', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlim(-4, 4)
    ax.grid(True, axis='x', alpha=0.3)
    
    # 값 적어 넣기
    for i, (bf, label) in enumerate(zip(log10_bfs, labels)):
        x_pos = bf + 0.1 if bf > 0 else bf - 0.1
        ha = 'left' if bf > 0 else 'right'
        ax.annotate(f'{bf:.2f}', (x_pos, i), va='center', ha=ha, fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_lindley_paradox() -> plt.Figure:
    """
    린들리의 역설을 그려 본다.
    
    반환값
    -------
    그림
        Matplotlib 그림
    """
    sample_sizes = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
    p_values, log_bfs, decisions = demonstrate_lindley_paradox(
        effect_size=0.15,
        sample_sizes=sample_sizes,
        prior_var=1.0
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 왼쪽: p값
    ax = axes[0]
    ax.semilogx(sample_sizes, p_values, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_title('Classical Test: P-values', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)
    
    # 오른쪽: 베이즈 인자
    ax = axes[1]
    log10_bfs = log_bfs / np.log(10)
    colors = ['#2ca02c' if bf > 0 else '#d62728' for bf in log10_bfs]
    ax.semilogx(sample_sizes, log10_bfs, 'o-', color='#ff7f0e', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='Substantial for H0')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7, label='Substantial for H1')
    ax.set_xlabel('Sample Size', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (B₀₁)', fontsize=12)
    ax.set_title('Bayesian Test: Bayes Factors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 주석 추가
    ax.annotate(
        "Paradox: p-value significant\nbut BF favors null",
        xy=(500, log10_bfs[4]),
        xytext=(800, -0.8),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray')
    )
    
    plt.tight_layout()
    return fig

def plot_prior_sensitivity(
    data: np.ndarray,
    prior_scales: np.ndarray = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
) -> plt.Figure:
    """
    베이즈 인자가 앞확률에 얼마나 민감한지 보인다.
    
    매개변수
    ----------
    data : ndarray
        관측 자료
    prior_scales : ndarray
        시험해 볼 앞확률 표준편차
    
    반환값
    -------
    그림
        Matplotlib 그림
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n = len(data)
    x_bar = np.mean(data)
    
    # 앞확률 눈금마다 베이즈 인자 셈하기
    log_bfs_01 = []
    for scale in prior_scales:
        prior_var = scale**2
        se = 1.0 / np.sqrt(n)  # 흩어짐 = 1을 안다고 놓고
        
        log_lik_h0 = stats.norm.logpdf(x_bar, 0, se)
        marginal_var = prior_var + se**2
        log_lik_h1 = stats.norm.logpdf(x_bar, 0, np.sqrt(marginal_var))
        
        log_bfs_01.append(log_lik_h0 - log_lik_h1)
    
    log_bfs_01 = np.array(log_bfs_01)
    log10_bfs = log_bfs_01 / np.log(10)
    
    # 왼쪽: 앞확률 밀도
    ax = axes[0]
    theta_range = np.linspace(-5, 5, 200)
    
    for i, scale in enumerate(prior_scales[::2]):  # 또렷하게 보이려고 하나 걸러 그린다
        prior_pdf = stats.norm.pdf(theta_range, 0, scale)
        ax.plot(theta_range, prior_pdf, label=f'σ = {scale}', linewidth=2)
    
    ax.axvline(x=x_bar, color='red', linestyle='--', label=f'x̄ = {x_bar:.2f}')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Prior Density', fontsize=12)
    ax.set_title('Prior Distributions for H₁', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 오른쪽: 앞확률 눈금에 따른 베이즈 인자
    ax = axes[1]
    ax.semilogx(prior_scales, log10_bfs, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='Substantial for H0')
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7, label='Substantial for H1')
    ax.set_xlabel('Prior Standard Deviation', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (B₀₁)', fontsize=12)
    ax.set_title('Prior Sensitivity of Bayes Factor', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 해석 덧붙이기
    ax.fill_between(prior_scales, -2, 0.5, alpha=0.1, color='red')
    ax.fill_between(prior_scales, 0.5, 3, alpha=0.1, color='green')
    
    plt.tight_layout()
    return fig

def plot_model_comparison_sequential(
    data: np.ndarray,
    model1: 'BetaBernoulliModel',
    model2: 'BetaBernoulliModel'
) -> plt.Figure:
    """
    모형 둘의 잇단 증거 쌓임을 그린다.
    
    매개변수
    ----------
    data : ndarray
        잇단 관측
    model1, model2 : BetaBernoulliModel
        견줄 모형
    
    반환값
    -------
    그림
        Matplotlib 그림
    """
    n = len(data)
    log_bfs = []
    
    for t in range(1, n + 1):
        data_t = data[:t]
        log_ev1 = model1.log_evidence(data_t)
        log_ev2 = model2.log_evidence(data_t)
        log_bfs.append(log_ev1 - log_ev2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    log10_bfs = np.array(log_bfs) / np.log(10)
    
    # 잇단 베이즈 인자 그리기
    ax.plot(range(1, n + 1), log10_bfs, 'o-', color='#1f77b4', 
            linewidth=2, markersize=4, alpha=0.7)
    
    # 기준선
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    
    # 구역 채우기
    ax.axhspan(0.5, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.axhspan(ax.get_ylim()[0], -0.5, alpha=0.1, color='red')
    
    ax.set_xlabel('Number of Observations', fontsize=12)
    ax.set_ylabel('log₁₀ Bayes Factor (M1 vs M2)', fontsize=12)
    ax.set_title('Sequential Evidence Accumulation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 주석
    ax.text(n * 0.9, 1.2, 'Strong for M1', fontsize=10, color='green')
    ax.text(n * 0.9, -1.2, 'Strong for M2', fontsize=10, color='red')
    
    plt.tight_layout()
    return fig

# =============================================================================
# 보여 주기
# =============================================================================

def demo_basic_bayes_factors():
    """기본 베이즈 인자 셈하기와 풀이를 보인다."""
    
    print("=" * 70)
    print("BAYES FACTORS: BASIC DEMONSTRATION")
    print("=" * 70)
    
    # 동전 던지기 자료 만들기
    np.random.seed(42)
    true_theta = 0.65
    n = 100
    data = np.random.binomial(1, true_theta, n)
    s = data.sum()
    
    print(f"\nData: {s} successes in {n} trials (true θ = {true_theta})")
    
    # 모델 비교
    print("\n--- Comparing Prior Beliefs ---")
    
    # 모형 1: 고른 앞확률(θ은 무엇이든 될 수 있다)
    # 모형 2: 공평한 동전 앞확률(θ이 0.5 언저리에 몰려 있다)
    model_uniform = BetaBernoulliModel(1, 1)
    model_fair = BetaBernoulliModel(50, 50)
    model_biased = BetaBernoulliModel(7, 3)  # θ ≈ 0.7을 기대한다
    
    log_ev_uniform = model_uniform.log_evidence(data)
    log_ev_fair = model_fair.log_evidence(data)
    log_ev_biased = model_biased.log_evidence(data)
    
    print(f"\nLog evidences:")
    print(f"  Uniform prior (α=β=1):     {log_ev_uniform:.4f}")
    print(f"  Fair coin (α=β=50):        {log_ev_fair:.4f}")
    print(f"  Biased prior (α=7, β=3):   {log_ev_biased:.4f}")
    
    # 베이즈 인자
    log_bf_ub = log_ev_uniform - log_ev_biased
    log_bf_fb = log_ev_fair - log_ev_biased
    log_bf_uf = log_ev_uniform - log_ev_fair
    
    print(f"\nBayes factors:")
    print(f"  Uniform vs Biased:  {interpret_bayes_factor(log_bf_ub)}")
    print(f"  Fair vs Biased:     {interpret_bayes_factor(log_bf_fb)}")
    print(f"  Uniform vs Fair:    {interpret_bayes_factor(log_bf_uf)}")
    
    # 뒤확률 모형 확률(앞확률 승산이 같음)
    log_evs = [log_ev_uniform, log_ev_fair, log_ev_biased]
    probs = posterior_probs_from_log_evidences(log_evs)
    
    print(f"\nPosterior model probabilities (equal priors):")
    print(f"  Uniform: {probs[0]:.4f}")
    print(f"  Fair:    {probs[1]:.4f}")
    print(f"  Biased:  {probs[2]:.4f}")

def demo_savage_dickey():
    """새비지-디키 밀도 비를 보인다."""
    
    print("\n" + "=" * 70)
    print("SAVAGE-DICKEY DENSITY RATIO")
    print("=" * 70)
    
    np.random.seed(123)
    
    # 데이터를 생성한다
    true_mu = 0.3  # 작은 효과
    n = 50
    data = np.random.normal(true_mu, 1.0, n)
    
    print(f"\nData: n={n}, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
    print(f"True μ = {true_mu}")
    
    # 앞확률 매개변수
    prior_mean = 0.0
    prior_var = 1.0
    known_var = 1.0
    null_value = 0.0
    
    # 새비지-디키 베이즈 인자 셈하기
    log_bf_01 = savage_dickey_gaussian(data, null_value, prior_mean, prior_var, known_var)
    
    print(f"\nTesting H0: μ = 0 vs H1: μ ~ N(0, 1)")
    print(f"Log B_01 = {log_bf_01:.4f}")
    print(f"Interpretation: {interpret_bayes_factor(log_bf_01)}")
    
    # 밀도 보이기
    x_bar = np.mean(data)
    tau0 = 1.0 / prior_var
    tau = 1.0 / known_var
    tau_n = tau0 + n * tau
    mu_n = (tau0 * prior_mean + n * tau * x_bar) / tau_n
    var_n = 1.0 / tau_n
    
    prior_at_null = stats.norm.pdf(null_value, prior_mean, np.sqrt(prior_var))
    posterior_at_null = stats.norm.pdf(null_value, mu_n, np.sqrt(var_n))
    
    print(f"\nDensities at null value (μ = 0):")
    print(f"  Prior density:     {prior_at_null:.6f}")
    print(f"  Posterior density: {posterior_at_null:.6f}")
    print(f"  Ratio (B_01):      {posterior_at_null / prior_at_null:.6f}")

def demo_bic_approximation():
    """베이즈 인자의 BIC 어림을 보인다."""
    
    print("\n" + "=" * 70)
    print("BIC APPROXIMATION TO BAYES FACTORS")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 다항 회귀 자료 만들기
    n = 100
    x = np.linspace(-2, 2, n)
    true_coeffs = [1, 0.5, -0.3]  # 이차
    y = true_coeffs[0] + true_coeffs[1] * x + true_coeffs[2] * x**2 + np.random.normal(0, 0.5, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.3x² + ε")
    print(f"Sample size: n = {n}")
    
    # 차수가 다른 모형 맞추기
    print("\n--- Model Comparison ---")
    
    results = []
    for degree in [1, 2, 3, 4, 5]:
        # 다항식 맞추기
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        
        # 로그 가능도 셈하기(σ² = 0.25을 안다고 놓고)
        residuals = y - y_pred
        sigma_sq = 0.25
        log_lik = -0.5 * n * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals**2) / sigma_sq
        
        # BIC
        k = degree + 1  # 매개변수의 개수
        bic = -2 * log_lik + k * np.log(n)
        
        results.append({
            'degree': degree,
            'k': k,
            'log_lik': log_lik,
            'bic': bic
        })
        
        print(f"Degree {degree}: log-lik = {log_lik:.2f}, BIC = {bic:.2f}")
    
    # 차수 2(참 모형)에 견준 어림 베이즈 인자 셈하기
    print("\n--- Approximate log Bayes factors vs Degree 2 ---")
    ref_idx = 1  # 차수 2
    
    for i, res in enumerate(results):
        if i != ref_idx:
            log_bf = bic_bayes_factor(
                results[ref_idx]['log_lik'], res['log_lik'],
                results[ref_idx]['k'], res['k'], n
            )
            print(f"Degree 2 vs Degree {res['degree']}: log B = {log_bf:.2f} "
                  f"({interpret_bayes_factor(log_bf).split('(')[0].strip()})")

def demo_lindley_paradox():
    """린들리의 역설을 보인다."""
    
    print("\n" + "=" * 70)
    print("LINDLEY'S PARADOX")
    print("=" * 70)
    
    print("\nEffect size: d = 0.15 (small)")
    print("Prior under H1: μ ~ N(0, 1)")
    
    sample_sizes = np.array([20, 50, 100, 500, 1000, 5000])
    p_values, log_bfs, decisions = demonstrate_lindley_paradox(
        effect_size=0.15,
        sample_sizes=sample_sizes
    )
    
    print("\n  n      p-value   Reject H0?   log₁₀ B₀₁   BF conclusion")
    print("-" * 65)
    
    for n, p, dec, log_bf in zip(sample_sizes, p_values, decisions, log_bfs):
        reject = "Yes" if dec else "No"
        log10_bf = log_bf / np.log(10)
        bf_interp = "H0" if log10_bf > 0.5 else "H1" if log10_bf < -0.5 else "Inconclusive"
        
        print(f"{n:5d}    {p:.4f}     {reject:4s}         {log10_bf:+.3f}        {bf_interp}")
    
    print("\n*** The paradox: Large samples show 'significant' p-values")
    print("    but Bayes factors favor the null hypothesis!")

def demo_sequential_evidence():
    """잇단 증거 쌓임을 보인다."""
    
    print("\n" + "=" * 70)
    print("SEQUENTIAL EVIDENCE ACCUMULATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 치우친 동전에서 자료 만들기
    true_theta = 0.65
    n = 200
    data = np.random.binomial(1, true_theta, n)
    
    print(f"\nTrue θ = {true_theta}")
    print(f"Comparing: M1 (uniform) vs M2 (fair coin, α=β=20)")
    
    model1 = BetaBernoulliModel(1, 1)        # 고름
    model2 = BetaBernoulliModel(20, 20)      # 공평한 동전이라는 믿음
    
    # 핵심 점에서 증거 좇기
    checkpoints = [10, 25, 50, 100, 150, 200]
    
    print("\nSequential Bayes factors (M1 vs M2):")
    print("  n      Successes   log₁₀ B₁₂   Interpretation")
    print("-" * 55)
    
    for t in checkpoints:
        data_t = data[:t]
        s = data_t.sum()
        log_bf = model1.log_evidence(data_t) - model2.log_evidence(data_t)
        log10_bf = log_bf / np.log(10)
        
        if log10_bf > 1:
            interp = "Strong for uniform"
        elif log10_bf > 0.5:
            interp = "Substantial for uniform"
        elif log10_bf < -1:
            interp = "Strong for fair"
        elif log10_bf < -0.5:
            interp = "Substantial for fair"
        else:
            interp = "Inconclusive"
        
        print(f"{t:4d}     {s:3d}         {log10_bf:+.3f}        {interp}")
    
    print("\n*** Evidence accumulates as more data arrives")

if __name__ == "__main__":
    demo_basic_bayes_factors()
    demo_savage_dickey()
    demo_bic_approximation()
    demo_lindley_paradox()
    demo_sequential_evidence()
```

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **정의** | $B_{12} = p(\mathcal{D} \mid \mathcal{M}_1) / p(\mathcal{D} \mid \mathcal{M}_2)$ |
| **풀이** | 데이터가 모형에 대한 상대적 믿음을 고치는 배수 |
| **로그 꼴** | $\log B_{12} = \log p(\mathcal{D} \mid \mathcal{M}_1) - \log p(\mathcal{D} \mid \mathcal{M}_2)$ |
| **뒤확률 승산** | $\text{뒤확률 승산} = B_{12} \times \text{앞확률 승산}$ |

### 풀이 눈금(카스와 라프터리)

| $\log_{10} B_{12}$ | $B_{12}$ | 증거 |
|-------------------|----------|----------|
| 0 ~ 0.5 | 1 ~ 3.2 | 말할 값어치가 거의 없음 |
| 0.5 ~ 1 | 3.2 ~ 10 | 뚜렷함 |
| 1 ~ 2 | 10 ~ 100 | 강함 |
| > 2 | > 100 | 결정적임 |

### 주요 성질

1. **대칭성**: $B_{21} = 1/B_{12}$
2. **추이성**: $B_{13} = B_{12} \cdot B_{23}$
3. **한결같음**: 끝내 참 모형을 편든다
4. **앞확률 민감도**: (뒤확률과 달리) 앞확률에 크게 기댄다
5. **오컴의 면도날**: 모형 증거에 이미 들어 있다

### 셈하는 방법

| 방법 | 쓸 수 있는 곳 | 필요한 것 |
|--------|--------------|--------------|
| 정확한 셈 | 켤레 모형 | 닫힌 꼴의 증거 |
| 새비지-디키 | 겹친 모형 | 영가설 값에서의 뒤확률 |
| BIC | 큰 표본 | 최대 가능도와 매개변수 개수 |
| 다리 표집 | 두루 쓰임 | 두 뒤확률에서 뽑은 표본 |

### 꼭 새길 단서

1. **린들리의 역설**: p값과 베이즈 인자가 어긋날 수 있다
2. **앞확률 민감도**: 결과가 앞확률을 어떻게 고르느냐에 크게 기댄다
3. **제대로 되지 않은 앞확률**: 모형 견줌에 쓸 수 없다
4. **모형이 알맞은가**: 베이즈 인자가 크다고 좋은 모형인 것은 아니고 대안보다 나을 뿐이다

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 모형 증거 | 13장: 모형 증거 | 베이즈 인자의 분자와 분모 |
| 정보 기준 | 13장: 정보 기준 | BIC이 로그 베이즈 인자를 어림한다 |
| 켤레 모형 | 13장: 분포 | 정확한 베이즈 인자 |
| 앞확률 고르기 | 13장: 바탕 | 앞확률 민감도 |
| BNN 견줌 | 13장: BNN | 구조 고르기 |

### 주요 참고 문헌

- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
- Jeffreys, H. (1961). *Theory of Probability* (3rd ed.). Oxford University Press.
- Rouder, J. N., et al. (2009). Bayesian t tests. *Psychonomic Bulletin & Review*, 16(2), 225-237.
- Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values. *Psychonomic Bulletin & Review*, 14(5), 779-804.

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 베이즈 인자을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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
