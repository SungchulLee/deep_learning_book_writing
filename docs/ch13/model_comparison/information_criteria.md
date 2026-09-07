# 정보 기준
**정보 기준**은 온전한 베이즈 증거를 셈하지 않고도 모형을 견줄 수 있는, 셈으로 다룰 만한 어림을 준다. 이 기준들은 잘 맞음과 모형의 복잡도 사이에서 균형을 잡아, 빈도주의 모형 고르기와 베이즈 원리를 잇는 원칙 있는 맞바꿈을 준다.

---

## 왜 필요한가: 실전의 모형 견줌

### 모형 증거의 어려움

주변 가능도(모형 증거)를 셈하는 일은 흔히 다룰 수 없다.

$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M}) \, p(\theta \mid \mathcal{M}) \, d\theta
$$

**셈의 어려움**:

- 차원이 높은 적분(닫힌 꼴이 없다)
- 제대로 된 앞확률이 있어야 한다(제대로 되지 않은 앞확률은 안 된다)
- 몬테카를로 어림값의 흩어짐이 크다
- 앞확률을 어떻게 정하느냐에 민감하다

**정보 기준**은 점 어림값과 점근 어림을 써서 이런 문제를 비껴간다.

### 정보 기준이 주는 것

1. **셈 효율**: 최대 가능도 어림값만 있으면 된다
2. **복잡도 벌**: 지나친 맞춤을 저절로 말린다
3. **풀이 가능성**: 잘 맞음과 복잡도로 또렷이 쪼개진다
4. **점근적 이음**: 큰 표본에서 베이즈의 양을 어림한다

---

## 아카이케 정보 기준(AIC)

### KL 벌어짐에서 끌어내기

**아카이케 정보 기준**은 참 데이터 생성 분포 $p_{\text{true}}$과 맞춘 모형 사이의 기대 쿨백-라이블러 벌어짐을 가장 작게 한다.

**얼개**: 모르는 $p_{\text{true}}$에서 온 데이터 $\mathcal{D} = \{x_1, \ldots, x_n\}$이 주어지면 다음과 같다.

$$
\text{KL}(p_{\text{true}} \| p_{\hat{\theta}}) = \int p_{\text{true}}(x) \log \frac{p_{\text{true}}(x)}{p(x \mid \hat{\theta})} \, dx
$$

$p_{\text{true}}$을 모르므로 표본 안의 로그 가능도로 어림한다.

$$
\hat{\ell}(\hat{\theta}) = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta})
$$

**핵심 통찰**: 표본 안의 로그 가능도는 표본 밖 성능을 *낙관적으로* 어림한다. 아카이케는 그 치우침이 대략 $k/n$임을 보였는데, $k$은 매개변수의 개수이다.

### 정의

$$
\boxed{\text{AIC} = -2 \log p(\mathcal{D} \mid \hat{\theta}) + 2k}
$$

여기서 각 기호는 다음과 같다.

- $\hat{\theta}$은 최대 가능도 어림값이다
- $k$은 어림한 매개변수의 개수이다
- AIC가 낮을수록 더 나은 모형이다

**같은 말로 쓰면 다음과 같다.**

$$
\text{AIC} = -2 \hat{\ell} + 2k
$$

여기서 $\hat{\ell} = \sum_{i=1}^n \log p(x_i \mid \hat{\theta})$은 가장 크게 한 로그 가능도이다.

### 끌어내기 자세히 보기

**1단계**: 관심 있는 양을 기대 로그 예측 밀도로 정의한다.

$$
\text{elpd} = \mathbb{E}_{p_{\text{true}}}\left[\log p(\tilde{x} \mid \hat{\theta})\right]
$$

**2단계**: 표본 안의 어림을 쓴다.

$$
\widehat{\text{elpd}}_{\text{in-sample}} = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta})
$$

**3단계**: 낙관 치우침을 바로잡는다. 규칙성 조건 아래에서는 다음과 같다.

$$
\mathbb{E}\left[\widehat{\text{elpd}}_{\text{in-sample}} - \text{elpd}\right] \approx \frac{k}{n}
$$

**4단계**: 바로잡은 어림값은 다음과 같다.

$$
\widehat{\text{elpd}}_{\text{AIC}} = \frac{1}{n} \sum_{i=1}^n \log p(x_i \mid \hat{\theta}) - \frac{k}{n}
$$

$-2n$을 곱하면 표준 AIC 식이 나온다.

### AIC의 성질

**1. 점근 효율성**: $n \to \infty$이면 AIC는 후보 모형 가운데 평균제곱 예측 오차를 가장 작게 하는 모형을 고른다.

**2. 한결같지 않음**: $n \to \infty$이어도 AIC는 참 모형으로 모이지 않는다(지나치게 복잡한 모형을 고르는 성향이 있다).

**3. 맞먹는 모형**: 가장 좋은 모형에서 AIC가 2 안쪽인 모형은 뚜렷한 뒷받침을 받는다.

모형 평균 내기를 위한 **AIC 무게**는 다음과 같다.

$$
w_i = \frac{\exp(-\Delta_i / 2)}{\sum_j \exp(-\Delta_j / 2)}
$$

여기서 $\Delta_i = \text{AIC}_i - \text{AIC}_{\min}$이다.

### 바로잡은 AIC(AICc)

표본이 작으면 AIC의 치우침 바로잡기가 모자라다. **AICc**은 이차 바로잡기를 준다.

$$
\boxed{\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}}
$$

**언제 쓸까**:

- AICc은 언제 써도 안전하다
- $n/k < 40$이면 꼭 써야 한다
- $n \to \infty$이면 AIC로 모인다

**끌어내기**: 가우스 오차를 쓰는 선형 회귀에서 치우침을 정확히 셈하면 다음이 나온다.

$$
\text{bias} = k + \frac{k(k+1)}{n - k - 1}
$$

---

## 베이즈 정보 기준(BIC)

### 모형 증거에서 끌어내기

**베이즈 정보 기준**은 라플라스 어림으로 로그 모형 증거를 어림한다.

**출발점**: 로그 주변 가능도

$$
\log p(\mathcal{D} \mid \mathcal{M}) = \log \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**라플라스 어림**: 최대 가능도 어림값 $\hat{\theta}$ 언저리에서 펼친다.

$$
\log p(\mathcal{D} \mid \theta) + \log p(\theta) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

여기서 $H = -\nabla^2_\theta [\log p(\mathcal{D} \mid \theta) + \log p(\theta)]|_{\hat{\theta}}$은 헤세 행렬이다.

**적분**: 가우스 적분을 쓰면 다음과 같다.

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) + \log p(\hat{\theta}) + \frac{k}{2}\log(2\pi) - \frac{1}{2}\log |H|
$$

**점근적 간추리기**: $n$이 크면 $H \approx n \cdot I(\hat{\theta})$이며 $I$은 피셔 정보이다. 퍼진 앞확률에서는 다음과 같다.

$$
\log p(\mathcal{D} \mid \mathcal{M}) \approx \log p(\mathcal{D} \mid \hat{\theta}) - \frac{k}{2}\log n + O(1)
$$

### 정의

$$
\boxed{\text{BIC} = -2 \log p(\mathcal{D} \mid \hat{\theta}) + k \log n}
$$

여기서 각 기호는 다음과 같다.

- $\hat{\theta}$은 최대 가능도 어림값이다
- $k$은 매개변수의 개수이다
- $n$은 표본 크기이다
- BIC가 낮을수록 더 나은 모형이다

**베이즈 인자와의 이음**: 모형 $\mathcal{M}_1$과 $\mathcal{M}_2$에 대해 다음과 같다.

$$
\log B_{12} \approx -\frac{1}{2}(\text{BIC}_1 - \text{BIC}_2)
$$

### BIC의 성질

**1. 한결같음**: $n \to \infty$이면 BIC는 (후보 가운데 있다면) 확률 1로 참 모형을 고른다.

**2. 더 센 벌**: $n \geq 8$이면 BIC가 AIC보다 복잡도에 더 세게 벌을 준다($\log n > 2$).

**3. 앞확률에 기댐**: BIC는 특정한 (단위 정보) 앞확률을 놓는다.

**AIC과의 견줌**:

| 갈래 | AIC | BIC |
|--------|-----|-----|
| 매개변수마다의 벌 | 2 | $\log n$ |
| 목표 | 가장 좋은 예측 | 참 모형 |
| 한결같음 | 아니오 | 예 |
| 복잡도 선호 | 더 복잡함 | 더 단순함 |
| 끌어내기 | KL 벌어짐 | 모형 증거 |

### BIC와 AIC가 어긋날 때

**다음일 때 BIC가 더 단순한 모형을 편든다.**

- 표본이 클 때($n > 8$)
- 참 모형이 후보 가운데 있을 때
- 목표가 참 모형을 짚어내는 것일 때

**다음일 때 AIC가 더 복잡한 모형을 편든다.**

- 예측 정확도에 초점이 있을 때
- 참 모형이 후보 가운데 없을 수 있을 때
- 어림해도 괜찮을 때

---

## 편차 정보 기준(DIC)

### 왜 필요한가: 베이즈 모형의 복잡도

층층 베이즈 모형에서는 매개변수를 세는 일이 아리송하다.

- 확률 효과도 "매개변수"인가?
- 오그라들기가 사실상 복잡도를 줄인다

**DIC**은 실효 매개변수 개수를 정의하여 이를 다룬다.

### 정의

**편차**: $D(\theta) = -2 \log p(\mathcal{D} \mid \theta)$

**실효 매개변수 개수**:

$$
p_D = \overline{D(\theta)} - D(\bar{\theta})
$$

여기서 각 기호는 다음과 같다.

- $\overline{D(\theta)} = \mathbb{E}_{\theta \mid \mathcal{D}}[D(\theta)]$은 뒤확률 평균 편차이다
- $D(\bar{\theta}) = D(\mathbb{E}_{\theta \mid \mathcal{D}}[\theta])$은 뒤확률의 평균에서의 편차이다

**DIC**:

$$
\boxed{\text{DIC} = \overline{D(\theta)} + p_D = 2\overline{D(\theta)} - D(\bar{\theta})}
$$

### p_D의 풀이

실효 매개변수 개수 $p_D$은 데이터가 뒤확률에 얼마나 많은 것을 알려 주는지를 잰다.

- 앞확률에 정보가 없으면 $p_D \approx k$이다
- 앞확률이 매개변수를 옥죄면(오그라들면) $p_D < k$이다
- $p_D$이 음수일 수도 있다(드물며 문제가 있다는 표시이다)

무리를 둔 **층층 모형**에서는 다음과 같다.

- 붙박인 효과는 저마다 1쯤을 보탠다
- 확률 효과는 (오그라듦 때문에) 저마다 1보다 적게 보탠다

### 다른 정의(겔먼)

**흩어짐에 바탕을 둔 $p_D$**:

$$
p_D = \frac{1}{2} \text{Var}_{\theta \mid \mathcal{D}}[D(\theta)]
$$

이는 흔히 더 안정적이고 늘 양수이다.

### DIC의 한계

1. **뒤확률의 평균이 대표가 아닐 수 있다**: 봉우리가 여럿인 뒤확률에서는 $\bar{\theta}$이 확률이 낮은 자리에 있을 수 있다.

2. **매개변수화에 흔들린다**: 매개변수화를 달리하면 DIC 값이 달라진다.

3. **또렷한 풀이가 없다**: AIC이나 BIC와 달리 DIC을 뒷받침하는 점근 이론이 없다.

4. **음수가 될 수 있다**: $p_D$이 음수가 될 수 있는데 이는 풀이할 수 없다.

---

## 널리 쓸 수 있는 정보 기준(WAIC)

### 왜 필요한가: 온전한 베이즈 대안

**WAIC**(와타나베-아카이케 정보 기준이라고도 한다)은 다음을 갖춘 온전한 베이즈 방법을 준다.

- 점 어림값만이 아니라 뒤확률 전체를 쓴다
- 표본 밖 예측 정확도를 어림한다
- BIC가 무너지는 특이 모형에서도 굴러간다

### 정의

**점마다의 로그 예측 밀도(lppd)**:

$$
\text{lppd} = \sum_{i=1}^n \log p(y_i \mid \mathcal{D}) = \sum_{i=1}^n \log \int p(y_i \mid \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

뒤확률 표본 $\{\theta^{(s)}\}_{s=1}^S$으로 어림한다.

$$
\widehat{\text{lppd}} = \sum_{i=1}^n \log \left( \frac{1}{S} \sum_{s=1}^S p(y_i \mid \theta^{(s)}) \right)
$$

**실효 매개변수 개수**:

$$
p_{\text{WAIC}} = \sum_{i=1}^n \text{Var}_{\theta \mid \mathcal{D}}[\log p(y_i \mid \theta)]
$$

다음과 같이 어림한다.

$$
\hat{p}_{\text{WAIC}} = \sum_{i=1}^n \widehat{\text{Var}}_s[\log p(y_i \mid \theta^{(s)})]
$$

**WAIC**:

$$
\boxed{\text{WAIC} = -2 \left( \widehat{\text{lppd}} - \hat{p}_{\text{WAIC}} \right)}
$$

### WAIC의 성질

**1. 교차 검증과 점근적으로 같다**: $n \to \infty$이면 WAIC $\approx$ 하나 빼기 교차 검증이다.

**2. 특이 모형에도 쓸 수 있다**: BIC가 무너지는 곳(이를테면 섞음 모형, 신경망)에서도 굴러간다.

**3. 온전한 베이즈**: 뒤확률의 아리송함을 셈에 넣는다.

**4. 점마다 셈하기**: 영향이 큰 관찰을 짚어낼 수 있다.

### DIC과의 견줌

| 갈래 | DIC | WAIC |
|--------|-----|------|
| 점 어림값 | 뒤확률의 평균 | 뒤확률 전체 |
| 복잡도 | $\bar{D} - D(\bar{\theta})$ | 흩어짐에 바탕을 둠 |
| 흔들리지 않음 | 매개변수화에 기댐 | 흔들리지 않음 |
| 특이 모형 | 무너질 수 있음 | 굴러감 |

---

## 하나 빼기 교차 검증(LOO-CV)

### 정의

예측을 재는 으뜸 잣대는 **하나 빼기 교차 검증**이다.

$$
\text{LOO-CV} = \sum_{i=1}^n \log p(y_i \mid \mathcal{D}_{-i})
$$

여기서 $\mathcal{D}_{-i}$은 관찰 $i$을 뺀 데이터를 뜻한다.

**정확히 셈하려면** 모형을 $n$개 맞추어야 하는데 값비싸다.

### 파레토로 매끄럽게 한 중요도 표집(PSIS-LOO)

**핵심 생각**: 중요도 표집으로 LOO을 어림한다.

$$
p(y_i \mid \mathcal{D}_{-i}) \approx \frac{\sum_s w_i^{(s)} p(y_i \mid \theta^{(s)})}{\sum_s w_i^{(s)}}
$$

여기서 $w_i^{(s)} \propto 1/p(y_i \mid \theta^{(s)})$은 중요도 무게이다.

**문제**: 중요도 무게의 흩어짐이 크다.

**해법**: **파레토로 매끄럽게 하기**가 무게를 안정시킨다.

1. 가장 큰 무게들에 일반화 파레토 분포를 맞춘다
2. 극단적인 무게를 기대 순서 통계량으로 갈아 끼운다
3. 안정시킨 무게로 어림한다

**진단**: 모양 매개변수 $\hat{k}$이 미더움을 알려 준다.

- $\hat{k} < 0.5$: 아주 좋다, LOO 어림값이 미덥다
- $0.5 < \hat{k} < 0.7$: 좋다, 약간의 치우침이 있을 수 있다
- $0.7 < \hat{k} < 1$: 그럭저럭이다, 이 점들에는 정확한 LOO을 생각해 보라
- $\hat{k} > 1$: 나쁘다, 중요도 표집이 무너진다

### 기대 로그 예측 밀도(ELPD)

모든 정보 기준은 **기대 로그 예측 밀도**를 어림한다.

$$
\text{elpd} = \sum_{i=1}^n \int p_{\text{true}}(\tilde{y}_i) \log p(\tilde{y}_i \mid \mathcal{D}) \, d\tilde{y}_i
$$

**관계**:

| 기준 | 무엇의 어림값인가 |
|-----------|-------------|
| AIC | $-2 \cdot \text{elpd}$(큰 표본) |
| BIC | $2 \log p(\mathcal{D} \mid \mathcal{M})$ |
| DIC | $-2 \cdot \text{elpd}$(베이즈식 끼워 넣기) |
| WAIC | $-2 \cdot \text{elpd}$(온전한 베이즈) |
| LOO-CV | elpd의 곧은 어림값 |

---

## 관계와 이음

### 점근 관계

**$n$이 큰 정규 모형에서는 다음과 같다.**

$$
\text{WAIC} \approx \text{LOO-CV} \approx \text{DIC}
$$

**BIC는 로그 베이즈 인자를 어림한다.**

$$
\text{BIC} \approx -2 \log p(\mathcal{D} \mid \mathcal{M}) + \text{constant}
$$

### 벌 견줌

매개변수가 $k$개이고 관찰이 $n$개인 모형에서는 다음과 같다.

| 기준 | 복잡도 벌 |
|-----------|-------------------|
| AIC | $2k$ |
| AICc | $2k + \frac{2k(k+1)}{n-k-1}$ |
| BIC | $k \log n$ |
| DIC | $2 p_D$ |
| WAIC | $2 p_{\text{WAIC}}$ |

**엇갈리는 점**: $n = e^2 \approx 7.4$일 때 AIC와 BIC의 벌이 같아진다.

### 기준마다 언제 쓸까

**다음일 때 AIC이나 AICc을 쓰라.**

- 목표가 예측일 때
- 참 모형이 후보 가운데 없을 수 있을 때
- 겹치지 않은 모형을 견주고 싶을 때

**다음일 때 BIC를 쓰라.**

- 목표가 참 모형을 짚어내는 것일 때
- 참 모형이 후보 가운데 있다고 믿을 때
- 베이즈 인자의 어림값을 얻고 싶을 때

**다음일 때 DIC을 쓰라.**

- 층층 베이즈 모형을 다룰 때
- 셈이 싼 기준이 필요할 때
- MCMC 표본이 이미 있을 때

**다음일 때 WAIC이나 LOO-CV을 쓰라.**

- 온전한 베이즈식 재기가 필요할 때
- 복잡하거나 특이한 모형을 다룰 때
- 미더운 불확실성 재기를 바랄 때

---

## 실용적인 고려

### 모형 고르기와 모형 평균 내기

**모형 고르기**: 가장 좋은 모형 하나를 고른다

- 모형의 아리송함을 아랑곳하지 않는다
- 풀이하기 더 단순하다
- 지나치게 자신할 수 있다

**모형 평균 내기**: 기준으로 모형에 무게를 준다

$$
p(\tilde{y} \mid \mathcal{D}) = \sum_k w_k \cdot p(\tilde{y} \mid \mathcal{D}, \mathcal{M}_k)
$$

AIC로 무게를 주면 다음과 같다.

$$
w_k = \frac{\exp(-\text{AIC}_k / 2)}{\sum_j \exp(-\text{AIC}_j / 2)}
$$

BIC로 무게를 주면(가짜 베이즈 인자) 다음과 같다.

$$
w_k = \frac{\exp(-\text{BIC}_k / 2)}{\sum_j \exp(-\text{BIC}_j / 2)}
$$

### 알리기와 풀이

**보통의 버릇**:

- 기준을 여럿 알려라(입맛대로 고르지 마라)
- 가장 좋은 모형과의 차이($\Delta$)를 알려라
- 기준 어림값의 아리송함을 살펴라

**$\Delta$에 대한 어림 규칙(AIC와 WAIC)**:

- $\Delta < 2$: 뚜렷한 뒷받침
- $2 < \Delta < 7$: 뒷받침이 꽤 약함
- $\Delta > 10$: 사실상 뒷받침 없음

### 흔히 빠지는 함정

**1. 견줄 수 없는 모형을 견주기**:

- 모든 견줌에 같은 데이터를 써야 한다
- 반응 변수가 다른 모형은 견줄 수 없다

**2. 불확실성을 아랑곳하지 않기**:

- 정보 기준의 차이에는 어림 오차가 있다
- 작은 차이는 뜻이 없을 수 있다

**3. 기준 하나에 지나치게 기대기**:

- 기준마다 서로 다른 물음에 답한다
- 민감도 분석이 중요하다

**4. 셈의 지름길을 잊기**:

- AIC는 최대 가능도 어림값만 있으면 된다
- BIC도 최대 가능도 어림값만 있으면 된다
- WAIC은 MCMC 표본이 필요하다

---

## 파이썬 구현

```python
"""
정보 기준: 온전한 구현

이 모듈은 모형 견줌을 위한 AIC, AICc, BIC, DIC, WAIC, LOO-CV의 셈하기와
그려 보기 및 모형 평균 내기 도구를 준다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp, gammaln
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# 핵심 정보 기준 함수
# =============================================================================

def aic(log_likelihood: float, k: int) -> float:
    """
    아카이케 정보 기준을 셈한다.
    
    매개변수
    ----------
    log_likelihood : float
        최대로 만든 로그 가능도
    k : int
        어림한 매개변수의 개수
    
    반환값
    -------
    float
        AIC 값(낮을수록 좋다)
    """
    return -2 * log_likelihood + 2 * k

def aicc(log_likelihood: float, k: int, n: int) -> float:
    """
    작은 표본을 위한 바로잡은 AIC(AICc)을 셈한다.
    
    매개변수
    ----------
    log_likelihood : float
        최대로 만든 로그 가능도
    k : int
        어림한 매개변수의 개수
    n : int
        표본 크기
    
    반환값
    -------
    float
        AICc 값(낮을수록 좋다)
    
    일으키는 예외
    ------
    ValueError
        n - k - 1 <= 0이면
    """
    if n - k - 1 <= 0:
        raise ValueError(f"Sample size n={n} too small for k={k} parameters")
    
    base_aic = aic(log_likelihood, k)
    correction = 2 * k * (k + 1) / (n - k - 1)
    
    return base_aic + correction

def bic(log_likelihood: float, k: int, n: int) -> float:
    """
    베이즈 정보 기준을 셈한다.
    
    매개변수
    ----------
    log_likelihood : float
        최대로 만든 로그 가능도
    k : int
        어림한 매개변수의 개수
    n : int
        표본 크기
    
    반환값
    -------
    float
        BIC 값(낮을수록 좋다)
    """
    return -2 * log_likelihood + k * np.log(n)

def bic_to_log_bayes_factor(bic1: float, bic2: float) -> float:
    """
    BIC 차이를 어림 로그 베이즈 인자로 바꾼다.
    
    매개변수
    ----------
    bic1, bic2 : float
        모형 1과 2의 BIC 값
    
    반환값
    -------
    float
        어림 log B_12(모형 1 대 모형 2)
    """
    return -0.5 * (bic1 - bic2)

# =============================================================================
# 베이즈 정보 기준(뒤확률 표본이 필요하다)
# =============================================================================

def dic(
    log_lik_samples: np.ndarray,
    log_lik_at_mean: float
) -> Tuple[float, float]:
    """
    이탈도 정보 기준을 셈한다.
    
    매개변수
    ----------
    log_lik_samples : ndarray
        뒤확률 표본마다 값을 매긴 로그 가능도
    log_lik_at_mean : float
        매개변수의 뒤확률 평균에서 값을 매긴 로그 가능도
    
    반환값
    -------
    dic : float
        DIC 값
    p_d : float
        실효 매개변수의 개수
    """
    # 평균 이탈도
    mean_deviance = -2 * np.mean(log_lik_samples)
    
    # 뒤확률 평균에서의 이탈도
    deviance_at_mean = -2 * log_lik_at_mean
    
    # 실효 매개변수
    p_d = mean_deviance - deviance_at_mean
    
    # DIC
    dic_value = mean_deviance + p_d
    
    return dic_value, p_d

def dic_alternative(log_lik_samples: np.ndarray) -> Tuple[float, float]:
    """
    흩어짐 기반 실효 매개변수를 써서 DIC을 셈한다(겔먼 판).
    
    매개변수
    ----------
    log_lik_samples : ndarray
        뒤확률 표본마다 값을 매긴 로그 가능도
    
    반환값
    -------
    dic : float
        DIC 값
    p_d : float
        실효 매개변수의 개수(흩어짐 기반)
    """
    # 평균 이탈도
    mean_deviance = -2 * np.mean(log_lik_samples)
    
    # 흩어짐 기반 실효 매개변수
    p_d = 0.5 * np.var(-2 * log_lik_samples)
    
    # DIC
    dic_value = mean_deviance + p_d
    
    return dic_value, p_d

def waic(
    log_lik_matrix: np.ndarray
) -> Tuple[float, float, float]:
    """
    널리 쓸 수 있는 정보 기준을 셈한다.
    
    매개변수
    ----------
    log_lik_matrix : 꼴 (n_samples, n_observations)의 ndarray
        뒤확률 표본과 관측마다의 로그 가능도
        항목 [s, i] = log p(y_i | theta^(s))
    
    반환값
    -------
    waic : float
        WAIC 값
    lppd : float
        점별 로그 예측 밀도
    p_waic : float
        실효 매개변수의 개수
    """
    S, n = log_lik_matrix.shape
    
    # 점별 로그 예측 밀도
    # lppd = sum_i log( mean_s p(y_i | theta^s) )
    lppd = np.sum(logsumexp(log_lik_matrix, axis=0) - np.log(S))
    
    # 실효 매개변수(점별 흩어짐)
    # p_waic = sum_i Var_s(log p(y_i | theta^s))
    p_waic = np.sum(np.var(log_lik_matrix, axis=0))
    
    # WAIC
    waic_value = -2 * (lppd - p_waic)
    
    return waic_value, lppd, p_waic

def waic_pointwise(
    log_lik_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    점별 WAIC 몫을 셈한다.
    
    매개변수
    ----------
    log_lik_matrix : 꼴 (n_samples, n_observations)의 ndarray
        뒤확률 표본과 관측마다의 로그 가능도
    
    반환값
    -------
    elpd_i : ndarray
        점별 기대 로그 예측 밀도
    p_waic_i : ndarray
        점별 실효 매개변수
    waic_i : ndarray
        점별 WAIC 몫
    """
    S, n = log_lik_matrix.shape
    
    # 점별 lppd
    lppd_i = logsumexp(log_lik_matrix, axis=0) - np.log(S)
    
    # 점별 p_waic
    p_waic_i = np.var(log_lik_matrix, axis=0)
    
    # 점별 WAIC
    elpd_i = lppd_i - p_waic_i
    waic_i = -2 * elpd_i
    
    return elpd_i, p_waic_i, waic_i

# =============================================================================
# 하나 빼기 교차 검증
# =============================================================================

def psis_loo(
    log_lik_matrix: np.ndarray,
    return_diagnostics: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    파레토로 매끄럽게 한 중요도 표집으로 LOO-CV을 셈한다.
    
    이것은 간추린 구현이다. 실제 제품에서는
    `arviz` 꾸러미를 생각해 보아라.
    
    매개변수
    ----------
    log_lik_matrix : 꼴 (n_samples, n_observations)의 ndarray
        뒤확률 표본과 관측마다의 로그 가능도
    return_diagnostics : bool
        파레토 k 진단을 되돌릴지 여부
    
    반환값
    -------
    loo : float
        LOO-CV 어림값(이탈도가 아니라 elpd 눈금)
    p_loo : float
        실효 매개변수의 개수
    k_hat : ndarray(return_diagnostics=True일 때)
        관측마다의 파레토 k 어림값
    """
    S, n = log_lik_matrix.shape
    
    elpd_loo = np.zeros(n)
    k_hat = np.zeros(n)
    
    for i in range(n):
        # 날 중요도 무게: 1 / p(y_i | theta)
        log_weights = -log_lik_matrix[:, i]
        
        # 안정시키고 파레토 매끄럽게 하기 쓰기
        # (간추림: 무게를 안정시키기만 한다)
        log_weights_centered = log_weights - np.max(log_weights)
        weights = np.exp(log_weights_centered)
        
        # 가장 큰 무게에 파레토 맞추기(간추림)
        M = max(int(np.sqrt(S)), 10)
        sorted_weights = np.sort(weights)[-M:]
        
        # 파레토 k 어림하기(간추린 적률 어림자)
        if sorted_weights[-1] > sorted_weights[0]:
            log_ratios = np.log(sorted_weights[-1] / sorted_weights[:-1])
            k_hat[i] = np.mean(log_ratios)
        else:
            k_hat[i] = 0
        
        # 무게 고르게 하기
        weights_normalized = weights / np.sum(weights)
        
        # LOO 예측 밀도
        log_lik_i = log_lik_matrix[:, i]
        elpd_loo[i] = np.log(np.sum(weights_normalized * np.exp(log_lik_i)))
    
    loo = np.sum(elpd_loo)
    
    # 실효 매개변수(어림)
    lppd = np.sum(logsumexp(log_lik_matrix, axis=0) - np.log(S))
    p_loo = lppd - loo
    
    if return_diagnostics:
        return loo, p_loo, k_hat
    return loo, p_loo

def exact_loo_cv(
    y: np.ndarray,
    X: np.ndarray,
    fit_func,
    predict_log_lik_func
) -> float:
    """
    다시 맞춰서 정확한 LOO-CV을 셈한다.
    
    매개변수
    ----------
    y : ndarray
        반응 변수
    X : ndarray
        설계 행렬
    fit_func : callable
        모형을 맞추고 매개변수를 되돌리는 함수
    predict_log_lik_func : callable
        Function(y_i, X_i, params) -> log p(y_i | X_i, params)
    
    반환값
    -------
    float
        LOO-CV(로그 예측 밀도의 합)
    """
    n = len(y)
    elpd = 0.0
    
    for i in range(n):
        # 관측 i 빼기
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        y_train = y[mask]
        X_train = X[mask] if X.ndim > 1 else X[mask]
        
        # 관측 n-1개로 맞추기
        params = fit_func(y_train, X_train)
        
        # 남겨 둔 관측에서 값 매기기
        elpd += predict_log_lik_func(y[i], X[i:i+1] if X.ndim > 1 else X[i], params)
    
    return elpd

# =============================================================================
# 모형 견줌 도구
# =============================================================================

@dataclass
class ModelComparison:
    """모형 견줌 결과를 담는 그릇."""
    names: List[str]
    aic: np.ndarray
    aicc: np.ndarray
    bic: np.ndarray
    n_params: np.ndarray
    log_lik: np.ndarray
    n_obs: int
    
    def summary(self) -> str:
        """꼴을 갖춘 견줌 표를 되돌린다."""
        lines = []
        lines.append("Model Comparison Summary")
        lines.append("=" * 70)
        lines.append(f"{'Model':<20} {'k':>5} {'LL':>12} {'AIC':>10} {'AICc':>10} {'BIC':>10}")
        lines.append("-" * 70)
        
        # AIC로 정렬
        order = np.argsort(self.aic)
        
        for i in order:
            lines.append(
                f"{self.names[i]:<20} {self.n_params[i]:>5} "
                f"{self.log_lik[i]:>12.2f} {self.aic[i]:>10.2f} "
                f"{self.aicc[i]:>10.2f} {self.bic[i]:>10.2f}"
            )
        
        lines.append("-" * 70)
        
        # 델타 값
        lines.append("\nDifferences from best model (AIC):")
        delta_aic = self.aic - np.min(self.aic)
        delta_bic = self.bic - np.min(self.bic)
        
        for i in order:
            lines.append(
                f"  {self.names[i]:<20} ΔAIC = {delta_aic[i]:>7.2f}, "
                f"ΔBIC = {delta_bic[i]:>7.2f}"
            )
        
        return "\n".join(lines)
    
    def weights(self, criterion: str = 'aic') -> np.ndarray:
        """
        모형 무게(아카이케 무게 또는 유사 베이즈 인자)를 셈한다.
        
        매개변수
        ----------
        criterion : str
            'aic' 또는 'bic'
        
        반환값
        -------
        ndarray
            모형 무게(합이 1)
        """
        if criterion == 'aic':
            values = self.aic
        elif criterion == 'bic':
            values = self.bic
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        delta = values - np.min(values)
        weights = np.exp(-0.5 * delta)
        
        return weights / np.sum(weights)

def compare_models(
    models: Dict[str, Tuple[float, int]],
    n_obs: int
) -> ModelComparison:
    """
    정보 기준으로 여러 모형을 견준다.
    
    매개변수
    ----------
    models : dict
        모형 이름을 (log_likelihood, n_params) 튜플로 잇는 사전
    n_obs : int
        관측의 개수
    
    반환값
    -------
    ModelComparison
        견줌 결과
    """
    names = list(models.keys())
    n_models = len(names)
    
    log_liks = np.array([models[name][0] for name in names])
    n_params = np.array([models[name][1] for name in names])
    
    aic_vals = np.array([aic(ll, k) for ll, k in zip(log_liks, n_params)])
    aicc_vals = np.array([aicc(ll, k, n_obs) for ll, k in zip(log_liks, n_params)])
    bic_vals = np.array([bic(ll, k, n_obs) for ll, k in zip(log_liks, n_params)])
    
    return ModelComparison(
        names=names,
        aic=aic_vals,
        aicc=aicc_vals,
        bic=bic_vals,
        n_params=n_params,
        log_lik=log_liks,
        n_obs=n_obs
    )

# =============================================================================
# 선형 회귀 보기
# =============================================================================

class LinearRegressionIC:
    """
    정보 기준 셈하기를 갖춘 선형 회귀.
    
    모형: y = X @ beta + epsilon, epsilon ~ N(0, sigma^2)
    """
    
    def __init__(self):
        self.beta_hat = None
        self.sigma_hat = None
        self.n = None
        self.k = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        MLE로 선형 회귀를 맞춘다.
        
        매개변수
        ----------
        X : 꼴 (n, p)의 ndarray
            설계 행렬
        y : 꼴 (n,)의 ndarray
            반응
        """
        self.n = len(y)
        self.k = X.shape[1] + 1  # sigma 몫으로 +1
        
        # beta의 OLS
        self.beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # sigma의 MLE
        residuals = y - X @ self.beta_hat
        self.sigma_hat = np.sqrt(np.sum(residuals**2) / self.n)
    
    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """MLE에서의 로그 가능도를 셈한다."""
        residuals = y - X @ self.beta_hat
        ll = (
            -0.5 * self.n * np.log(2 * np.pi)
            - self.n * np.log(self.sigma_hat)
            - 0.5 * np.sum(residuals**2) / self.sigma_hat**2
        )
        return ll
    
    def aic(self, X: np.ndarray, y: np.ndarray) -> float:
        """AIC를 셈한다."""
        return aic(self.log_likelihood(X, y), self.k)
    
    def aicc(self, X: np.ndarray, y: np.ndarray) -> float:
        """AICc을 셈한다."""
        return aicc(self.log_likelihood(X, y), self.k, self.n)
    
    def bic(self, X: np.ndarray, y: np.ndarray) -> float:
        """BIC를 셈한다."""
        return bic(self.log_likelihood(X, y), self.k, self.n)

# =============================================================================
# WAIC을 쓴 베이즈 선형 회귀
# =============================================================================

class BayesianLinearRegression:
    """
    켤레 앞확률을 쓴 베이즈 선형 회귀.
    
    모형: y = X @ beta + epsilon, epsilon ~ N(0, sigma^2)
    앞확률: beta | sigma^2 ~ N(0, sigma^2 * g * (X'X)^{-1})  [g-앞확률]
           sigma^2 ~ 역감마(a0/2, b0/2)
    """
    
    def __init__(self, g: float = 100.0, a0: float = 1.0, b0: float = 1.0):
        """
        매개변수
        ----------
        g : float
            g-앞확률 눈금 매개변수
        a0, b0 : float
            sigma^2의 역감마 앞확률 매개변수
        """
        self.g = g
        self.a0 = a0
        self.b0 = b0
        
        # 뒤확률 매개변수(맞춘 뒤 정해진다)
        self.beta_mean = None
        self.beta_cov = None
        self.a_n = None
        self.b_n = None
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """뒤확률 매개변수를 셈한다."""
        self.X = X
        self.y = y
        n, p = X.shape
        
        # OLS 어림값
        XtX = X.T @ X
        beta_ols = np.linalg.solve(XtX, X.T @ y)
        
        # beta의 뒤확률(sigma^2을 조건으로)
        # 평균: g/(1+g) * beta_ols
        self.beta_mean = self.g / (1 + self.g) * beta_ols
        
        # 잔차 제곱합
        residuals = y - X @ beta_ols
        SSR = np.sum(residuals**2)
        
        # sigma^2의 뒤확률
        self.a_n = self.a0 + n
        self.b_n = self.b0 + SSR + beta_ols.T @ XtX @ beta_ols / (1 + self.g)
        
        # beta의 뒤확률 공분산(sigma^2을 주변화하여)
        sigma2_mean = self.b_n / (self.a_n - 2) if self.a_n > 2 else self.b_n / self.a_n
        self.beta_cov = sigma2_mean * self.g / (1 + self.g) * np.linalg.inv(XtX)
    
    def sample_posterior(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        뒤확률 분포에서 표집한다.
        
        반환값
        -------
        beta_samples : 꼴 (n_samples, p)의 ndarray
        sigma2_samples : 꼴 (n_samples,)의 ndarray
        """
        p = len(self.beta_mean)
        
        # 역감마에서 sigma^2 표집
        sigma2_samples = stats.invgamma.rvs(
            self.a_n / 2,
            scale=self.b_n / 2,
            size=n_samples
        )
        
        # 다변량 정규에서 beta | sigma^2 표집
        XtX = self.X.T @ self.X
        beta_samples = np.zeros((n_samples, p))
        
        for s in range(n_samples):
            cov_s = sigma2_samples[s] * self.g / (1 + self.g) * np.linalg.inv(XtX)
            beta_samples[s] = np.random.multivariate_normal(self.beta_mean, cov_s)
        
        return beta_samples, sigma2_samples
    
    def compute_waic(self, n_samples: int = 1000) -> Tuple[float, float, float]:
        """
        이 모형의 WAIC을 셈한다.
        
        반환값
        -------
        waic_value : float
        lppd : float
        p_waic : float
        """
        beta_samples, sigma2_samples = self.sample_posterior(n_samples)
        
        n = len(self.y)
        log_lik_matrix = np.zeros((n_samples, n))
        
        for s in range(n_samples):
            mu = self.X @ beta_samples[s]
            sigma = np.sqrt(sigma2_samples[s])
            log_lik_matrix[s] = stats.norm.logpdf(self.y, mu, sigma)
        
        return waic(log_lik_matrix)

# =============================================================================
# 그려 보기 함수
# =============================================================================

def plot_ic_comparison(
    comparison: ModelComparison,
    criterion: str = 'all',
    figsize: Tuple[float, float] = (12, 5)
):
    """
    정보 기준 견줌을 그려 본다.
    
    매개변수
    ----------
    comparison : ModelComparison
        compare_models의 결과
    criterion : str
        'aic', 'bic', 'all'
    figsize : tuple
        그림 크기
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 왼쪽: 정보 기준 값
    ax = axes[0]
    x = np.arange(len(comparison.names))
    width = 0.25
    
    if criterion in ['aic', 'all']:
        ax.bar(x - width, comparison.aic, width, label='AIC', alpha=0.8)
    if criterion in ['bic', 'all']:
        ax.bar(x, comparison.bic, width, label='BIC', alpha=0.8)
    if criterion in ['aicc', 'all']:
        ax.bar(x + width, comparison.aicc, width, label='AICc', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.names, rotation=45, ha='right')
    ax.set_ylabel('Information Criterion')
    ax.legend()
    ax.set_title('Model Comparison')
    
    # 오른쪽: 모형 무게
    ax = axes[1]
    aic_weights = comparison.weights('aic')
    bic_weights = comparison.weights('bic')
    
    ax.bar(x - width/2, aic_weights, width, label='AIC weights', alpha=0.8)
    ax.bar(x + width/2, bic_weights, width, label='BIC weights', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.names, rotation=45, ha='right')
    ax.set_ylabel('Model Weight')
    ax.legend()
    ax.set_title('Model Weights')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def plot_penalty_comparison(
    max_params: int = 20,
    sample_sizes: List[int] = [10, 50, 100, 500]
):
    """
    표본 크기에 걸쳐 AIC와 BIC의 벌점 항을 견준다.
    
    매개변수
    ----------
    max_params : int
        그릴 매개변수의 최대 개수
    sample_sizes : list
        견줄 표본 크기
    """
    k = np.arange(1, max_params + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 왼쪽: 매개변수마다의 벌점
    ax = axes[0]
    ax.axhline(2, color='blue', linestyle='--', label='AIC: 2', linewidth=2)
    
    for n in sample_sizes:
        ax.axhline(np.log(n), linestyle='-', alpha=0.7, 
                   label=f'BIC (n={n}): {np.log(n):.2f}')
    
    ax.set_xlabel('(Reference)')
    ax.set_ylabel('Penalty per Parameter')
    ax.legend()
    ax.set_title('Penalty per Parameter')
    ax.set_ylim(0, 8)
    
    # 오른쪽: 벌점 합계
    ax = axes[1]
    ax.plot(k, 2 * k, 'b--', label='AIC: 2k', linewidth=2)
    
    for n in sample_sizes:
        ax.plot(k, k * np.log(n), label=f'BIC (n={n}): k·log({n})', alpha=0.7)
    
    ax.set_xlabel('Number of Parameters (k)')
    ax.set_ylabel('Total Complexity Penalty')
    ax.legend()
    ax.set_title('Total Penalty vs Model Complexity')
    
    plt.tight_layout()
    plt.show()

def plot_waic_diagnostics(
    elpd_i: np.ndarray,
    p_waic_i: np.ndarray,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    점별 WAIC 진단을 그린다.
    
    매개변수
    ----------
    elpd_i : ndarray
        점별 기대 로그 예측 밀도
    p_waic_i : ndarray
        점별 실효 매개변수
    figsize : tuple
        그림 크기
    """
    n = len(elpd_i)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 왼쪽: 점별 elpd
    ax = axes[0]
    ax.scatter(range(n), elpd_i, alpha=0.6, s=20)
    ax.axhline(np.mean(elpd_i), color='red', linestyle='--', 
               label=f'Mean: {np.mean(elpd_i):.2f}')
    
    # 말썽 있는 관측 도드라지게 하기
    threshold = np.percentile(elpd_i, 5)
    problematic = elpd_i < threshold
    ax.scatter(np.where(problematic)[0], elpd_i[problematic], 
               color='red', s=40, label=f'Bottom 5% (n={np.sum(problematic)})')
    
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Pointwise ELPD')
    ax.legend()
    ax.set_title('Pointwise Expected Log Predictive Density')
    
    # 오른쪽: 점별 p_waic
    ax = axes[1]
    ax.scatter(range(n), p_waic_i, alpha=0.6, s=20)
    ax.axhline(np.mean(p_waic_i), color='red', linestyle='--',
               label=f'Mean: {np.mean(p_waic_i):.3f}')
    
    # 영향이 큰 관측 도드라지게 하기
    threshold = np.percentile(p_waic_i, 95)
    high_influence = p_waic_i > threshold
    ax.scatter(np.where(high_influence)[0], p_waic_i[high_influence],
               color='red', s=40, label=f'Top 5% (n={np.sum(high_influence)})')
    
    ax.set_xlabel('Observation Index')
    ax.set_ylabel('Pointwise p_WAIC')
    ax.legend()
    ax.set_title('Pointwise Effective Parameters')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 시연 함수
# =============================================================================

def demo_basic_ic():
    """다항 회귀에서 기본 AIC/BIC 견줌을 보인다."""
    
    print("=" * 70)
    print("BASIC INFORMATION CRITERIA: POLYNOMIAL REGRESSION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 이차 모형에서 자료 만들기
    n = 100
    x = np.linspace(-3, 3, n)
    y_true = 1 + 0.5 * x - 0.3 * x**2
    y = y_true + np.random.normal(0, 0.5, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.3x² + ε, ε ~ N(0, 0.25)")
    print(f"Sample size: n = {n}")
    
    # 차수가 다른 모형 맞추기
    print("\n--- Fitting Polynomial Models ---")
    
    models = {}
    
    for degree in range(1, 7):
        # 설계 행렬 만들기
        X = np.column_stack([x**i for i in range(degree + 1)])
        
        # 모형 맞추기
        model = LinearRegressionIC()
        model.fit(X, y)
        
        ll = model.log_likelihood(X, y)
        k = degree + 2  # 계수 + sigma
        
        models[f'Degree {degree}'] = (ll, k)
        
        print(f"Degree {degree}: k={k}, log-lik={ll:.2f}")
    
    # 모델 비교
    comparison = compare_models(models, n)
    print("\n" + comparison.summary())
    
    # 무게
    print("\nModel weights:")
    aic_w = comparison.weights('aic')
    bic_w = comparison.weights('bic')
    
    for i, name in enumerate(comparison.names):
        print(f"  {name}: AIC weight = {aic_w[i]:.3f}, BIC weight = {bic_w[i]:.3f}")
    
    return comparison

def demo_aic_vs_bic():
    """AIC와 BIC가 엇갈리는 때를 보인다."""
    
    print("\n" + "=" * 70)
    print("AIC VS BIC: WHEN THEY DISAGREE")
    print("=" * 70)
    
    np.random.seed(123)
    
    # 참 모형은 단순하다(차수 1)
    true_degree = 1
    
    sample_sizes = [20, 50, 100, 500, 1000]
    
    print("\nTrue model: y = 1 + 0.5x + ε")
    print("Comparing degree 1 vs degree 3 polynomials")
    print("\n  n     AIC₁   AIC₃    BIC₁   BIC₃   AIC choice  BIC choice")
    print("-" * 65)
    
    for n in sample_sizes:
        x = np.linspace(-3, 3, n)
        y = 1 + 0.5 * x + np.random.normal(0, 0.5, n)
        
        results = []
        for degree in [1, 3]:
            X = np.column_stack([x**i for i in range(degree + 1)])
            model = LinearRegressionIC()
            model.fit(X, y)
            
            ll = model.log_likelihood(X, y)
            k = degree + 2
            
            results.append({
                'aic': aic(ll, k),
                'bic': bic(ll, k, n)
            })
        
        aic_choice = 1 if results[0]['aic'] < results[1]['aic'] else 3
        bic_choice = 1 if results[0]['bic'] < results[1]['bic'] else 3
        
        print(f"{n:5d}  {results[0]['aic']:6.1f} {results[1]['aic']:6.1f}  "
              f"{results[0]['bic']:6.1f} {results[1]['bic']:6.1f}   "
              f"Degree {aic_choice}      Degree {bic_choice}")
    
    print("\n*** BIC's stronger penalty correctly identifies the simpler true model")
    print("    even when AIC might prefer the more complex model")

def demo_waic():
    """베이즈 모형의 WAIC 셈하기를 보인다."""
    
    print("\n" + "=" * 70)
    print("WAIC: FULLY BAYESIAN MODEL COMPARISON")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 데이터를 생성한다
    n = 50
    x = np.linspace(-2, 2, n)
    y_true = 1 + 0.5 * x - 0.2 * x**2
    y = y_true + np.random.normal(0, 0.3, n)
    
    print(f"\nTrue model: y = 1 + 0.5x - 0.2x² + ε")
    print(f"Sample size: n = {n}")
    
    print("\n--- Computing WAIC for Different Polynomial Degrees ---")
    
    results = []
    
    for degree in [1, 2, 3, 4]:
        X = np.column_stack([x**i for i in range(degree + 1)])
        
        model = BayesianLinearRegression(g=n)
        model.fit(X, y)
        
        waic_val, lppd, p_waic = model.compute_waic(n_samples=2000)
        
        results.append({
            'degree': degree,
            'waic': waic_val,
            'lppd': lppd,
            'p_waic': p_waic
        })
        
        print(f"Degree {degree}: WAIC = {waic_val:.2f}, "
              f"lppd = {lppd:.2f}, p_WAIC = {p_waic:.2f}")
    
    # 가장 좋은 모델 찾기
    best = min(results, key=lambda x: x['waic'])
    print(f"\n*** Best model by WAIC: Degree {best['degree']}")
    
    return results

def demo_dic():
    """DIC 셈하기를 보인다."""
    
    print("\n" + "=" * 70)
    print("DIC: DEVIANCE INFORMATION CRITERION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 데이터를 생성한다
    n = 100
    x = np.linspace(0, 10, n)
    true_rate = 0.5
    y = np.random.poisson(np.exp(true_rate * x) * 0.1, n)
    
    print(f"\nPoisson regression example")
    print(f"Sample size: n = {n}")
    
    # 모형 둘의 뒤확률 표본 흉내내기
    # 모형 1: 일정한 비율
    # 모형 2: 선형 비율
    
    print("\n--- Model 1: Constant rate ---")
    
    # MCMC 표본 흉내내기(간추림)
    n_samples = 2000
    
    # 모형 1: lambda = exp(beta0)
    beta0_samples = np.random.normal(1.5, 0.1, n_samples)
    
    log_lik_1 = np.zeros(n_samples)
    for s in range(n_samples):
        rate = np.exp(beta0_samples[s]) * np.ones(n)
        log_lik_1[s] = np.sum(stats.poisson.logpmf(y, rate))
    
    # 뒤확률 평균에서의 로그 가능도
    rate_mean = np.exp(np.mean(beta0_samples)) * np.ones(n)
    log_lik_at_mean_1 = np.sum(stats.poisson.logpmf(y, rate_mean))
    
    dic_1, pd_1 = dic(log_lik_1, log_lik_at_mean_1)
    print(f"  DIC = {dic_1:.2f}, p_D = {pd_1:.2f}")
    
    print("\n--- Model 2: Linear rate ---")
    
    # 모형 2: lambda = exp(beta0 + beta1 * x)
    beta0_samples_2 = np.random.normal(0.5, 0.05, n_samples)
    beta1_samples = np.random.normal(0.1, 0.01, n_samples)
    
    log_lik_2 = np.zeros(n_samples)
    for s in range(n_samples):
        rate = np.exp(beta0_samples_2[s] + beta1_samples[s] * x)
        log_lik_2[s] = np.sum(stats.poisson.logpmf(y, rate))
    
    # 뒤확률 평균에서의 로그 가능도
    rate_mean = np.exp(np.mean(beta0_samples_2) + np.mean(beta1_samples) * x)
    log_lik_at_mean_2 = np.sum(stats.poisson.logpmf(y, rate_mean))
    
    dic_2, pd_2 = dic(log_lik_2, log_lik_at_mean_2)
    print(f"  DIC = {dic_2:.2f}, p_D = {pd_2:.2f}")
    
    print(f"\n*** Model comparison: ΔDIC = {dic_1 - dic_2:.2f}")
    if dic_1 < dic_2:
        print("    Constant rate model preferred")
    else:
        print("    Linear rate model preferred")

def demo_model_averaging():
    """정보 기준 무게로 모형 평균 내기를 보인다."""
    
    print("\n" + "=" * 70)
    print("MODEL AVERAGING WITH INFORMATION CRITERIA")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 데이터를 생성한다
    n = 50
    x = np.linspace(-2, 2, n)
    y_true = 1 + 0.3 * x - 0.15 * x**2 + 0.05 * x**3
    y = y_true + np.random.normal(0, 0.3, n)
    
    print(f"\nTrue model has small cubic term")
    print("Comparing models of degrees 1, 2, 3, 4")
    
    # 모형 맞추기
    models = {}
    predictions = {}
    
    x_new = np.linspace(-2.5, 2.5, 100)
    
    for degree in [1, 2, 3, 4]:
        X = np.column_stack([x**i for i in range(degree + 1)])
        X_new = np.column_stack([x_new**i for i in range(degree + 1)])
        
        model = LinearRegressionIC()
        model.fit(X, y)
        
        ll = model.log_likelihood(X, y)
        k = degree + 2
        
        models[f'Degree {degree}'] = (ll, k)
        predictions[degree] = X_new @ model.beta_hat
    
    # 무게 셈하기
    comparison = compare_models(models, n)
    weights = comparison.weights('aic')
    
    print("\nAIC weights:")
    for i, name in enumerate(comparison.names):
        print(f"  {name}: {weights[i]:.3f}")
    
    # 모형 평균 예측
    y_avg = np.zeros_like(x_new)
    for i, degree in enumerate([1, 2, 3, 4]):
        y_avg += weights[i] * predictions[degree]
    
    print(f"\n*** Model-averaged prediction incorporates uncertainty")
    print(f"    about which model is correct")
    
    return comparison, predictions, y_avg

def demo_criteria_consistency():
    """BIC의 일관성과 AIC의 효율을 견주어 보인다."""
    
    print("\n" + "=" * 70)
    print("CONSISTENCY VS EFFICIENCY: AIC VS BIC")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 참 모형: 단순 선형
    true_degree = 1
    
    sample_sizes = [20, 50, 100, 200, 500, 1000]
    n_simulations = 100
    
    print(f"\nTrue model: degree {true_degree}")
    print(f"Simulations per sample size: {n_simulations}")
    
    print("\n  n     AIC correct   BIC correct")
    print("-" * 40)
    
    for n in sample_sizes:
        aic_correct = 0
        bic_correct = 0
        
        for _ in range(n_simulations):
            x = np.linspace(-3, 3, n)
            y = 1 + 0.5 * x + np.random.normal(0, 0.5, n)
            
            models = {}
            for degree in [1, 2, 3, 4]:
                X = np.column_stack([x**i for i in range(degree + 1)])
                model = LinearRegressionIC()
                model.fit(X, y)
                
                ll = model.log_likelihood(X, y)
                k = degree + 2
                models[degree] = (ll, k)
            
            comparison = compare_models(
                {f'd{d}': models[d] for d in models},
                n
            )
            
            aic_best = np.argmin(comparison.aic) + 1
            bic_best = np.argmin(comparison.bic) + 1
            
            if aic_best == true_degree:
                aic_correct += 1
            if bic_best == true_degree:
                bic_correct += 1
        
        print(f"{n:5d}     {aic_correct/n_simulations*100:5.1f}%        "
              f"{bic_correct/n_simulations*100:5.1f}%")
    
    print("\n*** BIC is consistent: converges to true model as n → ∞")
    print("    AIC is efficient: minimizes prediction error but may overfit")

if __name__ == "__main__":
    comparison = demo_basic_ic()
    demo_aic_vs_bic()
    demo_waic()
    demo_dic()
    demo_model_averaging()
    demo_criteria_consistency()
```

---

## 요약

| 기준 | 식 | 주된 쓰임 |
|-----------|---------|-------------|
| **AIC** | $-2\hat{\ell} + 2k$ | 예측(KL 가장 작게 하기) |
| **AICc** | $\text{AIC} + \frac{2k(k+1)}{n-k-1}$ | 작은 표본의 예측 |
| **BIC** | $-2\hat{\ell} + k\log n$ | 모형 짚어내기 |
| **DIC** | $\bar{D} + p_D$ | 층층 베이즈 |
| **WAIC** | $-2(\text{lppd} - p_{\text{WAIC}})$ | 온전한 베이즈 |

### 주요 성질

| 성질 | AIC | BIC | DIC | WAIC |
|----------|-----|-----|-----|------|
| 한결같음 | 아니오 | 예 | 아니오 | 아니오 |
| 효율적 | 예 | 아니오 | — | 예 |
| MCMC 필요 | 아니오 | 아니오 | 예 | 예 |
| 특이 모형 | 무너짐 | 무너짐 | 무너질 수 있음 | 굴러감 |

### 실효 매개변수

| 기준 | 실효 매개변수 |
|-----------|---------------------|
| AIC와 BIC | $k$(개수) |
| DIC | $p_D = \bar{D} - D(\bar{\theta})$ |
| WAIC | $p_{\text{WAIC}} = \sum_i \text{Var}[\log p(y_i \mid \theta)]$ |

### 언제 쓸까

| 상황 | 권하는 기준 |
|-----------|----------------------|
| 예측에 초점 | AIC이나 AICc |
| 참 모형 짚어내기 | BIC |
| 층층 베이즈 모형 | DIC이나 WAIC |
| 복잡하거나 특이한 모형 | WAIC이나 LOO-CV |
| 작은 표본 | AICc |
| 모형 평균 내기 | AIC 무게나 BIC 무게 |

### 풀이 지침

**델타 값**(가장 좋은 모형과의 차이):

| $\Delta$ | 뒷받침 |
|----------|---------|
| 0~2 | 뚜렷함 |
| 2~7 | 꽤 약함 |
| > 10 | 사실상 없음 |

### 다른 장과의 이음

| 주제 | 장 | 이음 |
|-------|---------|------------|
| 베이즈 인자 | 13장: 베이즈 인자 | BIC가 로그 베이즈 인자를 어림한다 |
| 모형 증거 | 13장: 모형 증거 | 정보 기준은 온전한 적분을 피한다 |
| 앞확률 고르기 | 13장: 바탕 | BIC는 단위 정보 앞확률을 놓는다 |
| 뒤확률 추론 | 13장: 분포 | DIC과 WAIC은 뒤확률 표본을 쓴다 |
| BNN 견줌 | 13장: BNN | 구조 고르기 |

### 주요 참고 문헌

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Trans. Automatic Control*, 19(6), 716-723.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
- Spiegelhalter, D. J., et al. (2002). Bayesian measures of model complexity and fit. *JRSS B*, 64(4), 583-639.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and WAIC. *JMLR*, 11, 3571-3594.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

## 연습문제

**연습문제 1.**
이 쪽이 다루는 핵심 개념과 그것이 베이즈 통계에서 하는 몫을 설명하라.

??? success "연습문제 1 풀이"
    이 쪽은 베이즈 추론의 근본 부품인 정보 기준을(를) 다룬다. 이는 데이터로 믿음을 고치고, 불확실성을 수로 나타내며, 불확실함 속에서 결정을 내리는 더 넓은 틀과 이어진다. 베이즈의 눈은 앞선 앎을 아우르고 불확실성을 분석 전체로 퍼뜨리는 원칙 있는 길을 준다.

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
