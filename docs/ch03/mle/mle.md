# 최대가능도 추정
## 들어가며

최대가능도 추정(MLE)은 통계학과 기계 학습에서 매개변수를 추정하는 가장 기본적인 방법 중 하나이다. 통계 모델과 관측 데이터가 주어졌을 때, MLE는 우리가 실제로 관측한 데이터가 나올 확률을 최대로 만드는 매개변수 값을 찾는다.

!!! note "딥러닝에서 MLE가 중요한 이유"
    딥러닝의 거의 모든 손실 함수를 MLE의 원리에서 유도할 수 있으므로 MLE를 이해하는 것이 필수적이다. 교차 엔트로피 손실, 평균제곱오차를 비롯한 여러 목적 함수는 서로 다른 확률적 가정 아래의 음의 로그가능도일 뿐이다.

## 핵심 착상

### 직관적인 이해

동전을 100번 던져 앞면이 70번 나왔다고 하자. 앞면이 나올 확률로 가장 그럴듯한 추정값은 무엇인가? 직관적으로 0.70이라고 답할 것이다. MLE는 바로 이 직관을 수학적으로 정식화한 것이다.

MLE는 이렇게 묻는다. **"내 관측이 주어졌을 때, 어떤 매개변수 값이 이 관측을 가장 그럴듯하게 만들었을까?"**

### 형식적 정의

$\mathbf{X} = \{x_1, x_2, \ldots, x_n\}$을 미지의 매개변수 $\theta$을 갖는 확률분포에서 나온 $n$개의 독립적인 관측이라 하자. **가능도 함수**는 다음과 같이 정의된다.

$$
L(\theta | \mathbf{X}) = P(\mathbf{X} | \theta) = \prod_{i=1}^{n} p(x_i | \theta)
$$

**최대가능도 추정량**은 다음과 같다.

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} L(\theta | \mathbf{X})
$$

### 로그가능도

실무에서는 가능도 대신 거의 언제나 **로그가능도**를 다룬다.

$$
\ell(\theta | \mathbf{X}) = \log L(\theta | \mathbf{X}) = \sum_{i=1}^{n} \log p(x_i | \theta)
$$

!!! tip "왜 로그가능도인가?"

    1. **수치적 안정성**: 작은 확률을 여러 번 곱하면 아랫넘침으로 0이 될 수 있다
    2. **계산 효율**: 합이 곱보다 계산이 빠르다
    3. **수학적 편의**: 합의 미분이 곱의 미분보다 간단하다
    4. **단조성**: $\log$은 단조증가하므로 $\ell(\theta)$을 최대화하는 것은 $L(\theta)$을 최대화하는 것과 같다

## 수학적 틀

### 가능도 함수

모수 모델 $p(x|\theta)$에서 가능도 함수는 데이터를 고정된 것으로, 매개변수를 변수로 취급한다.

$$
L: \Theta \to \mathbb{R}^+, \quad \theta \mapsto \prod_{i=1}^{n} p(x_i | \theta)
$$

여기서 $\Theta$은 매개변수 공간이다. 유념할 핵심 성질은 다음과 같다. 가능도는 $\theta$에 대한 확률분포가 아니며, $\int L(\theta | \mathbf{X}) d\theta$은 일반적으로 1이 아니다. 가능도는 서로 다른 매개변수 값들의 상대적인 그럴듯함을 알려 준다.

### MLE 찾기

가능도 함수가 미분 가능하면 다음 과정으로 MLE를 찾는다.

1. 로그가능도를 $\theta$에 대해 **미분한다**
2. **0으로 둔다**: $\frac{\partial \ell}{\partial \theta} = 0$ (**점수 방정식**)
3. **$\theta$에 대해 푼다**
4. 최댓값인지 **확인한다** (이계도함수 판정)

### 점수 함수

**점수 함수**는 로그가능도의 경사이다.

$$
s(\theta) = \nabla_\theta \ell(\theta | \mathbf{X}) = \sum_{i=1}^{n} \nabla_\theta \log p(x_i | \theta)
$$

정칙 조건 아래에서 참 매개변수에서의 점수의 기댓값은 0이다.

$$
\mathbb{E}[s(\theta_0)] = 0
$$

## 풀이 예제: 베르누이 분포

가장 단순한 경우, 즉 베르누이 분포의 성공 확률 $p$을 추정하는 문제에서 MLE를 유도해 보자.

**설정**: 모델은 $X \sim \text{Bernoulli}(p)$, 데이터는 각 $x_i \in \{0, 1\}$인 $\mathbf{X} = \{x_1, \ldots, x_n\}$, 매개변수는 $p \in [0, 1]$이다.

**1단계 — 가능도.** 관측 하나에 대해 $p(x_i | p) = p^{x_i}(1-p)^{1-x_i}$이다. (독립을 가정하고) 모든 관측에 대해서는 다음과 같다.

$$
L(p | \mathbf{X}) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^{k}(1-p)^{n - k}
$$

여기서 $k = \sum_{i=1}^{n} x_i$은 성공의 횟수이다.

**2단계 — 로그가능도.**

$$
\ell(p) = k \log p + (n-k) \log(1-p)
$$

**3단계 — 점수 방정식.**

$$
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

**4단계 — 풀이.** 교차 곱하면 $k(1-p) = p(n-k)$이 되고, 정리하면 다음과 같다.

$$
\boxed{\hat{p}_{\text{MLE}} = \frac{k}{n} = \frac{\sum_{i=1}^{n} x_i}{n}}
$$

MLE는 그저 표본 비율이다. 직관이 말해 주는 바로 그것이다.

**5단계 — 최댓값 확인.** 이계도함수 $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$이므로 최댓값임이 확인된다.

## 흔한 분포들의 MLE

### 이산 분포

**이항분포** $X \sim \text{Binomial}(n, p)$ — 관측 $m$개 $x_1, \ldots, x_m$에 대해 다음과 같다.

$$
\hat{p} = \frac{\sum_{i=1}^{m} x_i}{mn} = \frac{\bar{x}}{n}
$$

**범주형** $X \sim \text{Categorical}(p_1, \ldots, p_K)$ — 제약 $\sum p_k = 1$에 라그랑주 승수를 쓰면 다음을 얻는다.

$$
\hat{p}_k = \frac{n_k}{n} = \frac{n_k}{\sum_{j=1}^{K} n_j}
$$

각 확률은 그 상대도수로 추정된다.

**포아송** $X \sim \text{Poisson}(\lambda)$ — 로그가능도는 $\ell(\lambda) = (\sum x_i) \log \lambda - n\lambda + \text{const}$이다. 도함수를 0으로 두면 다음과 같다.

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda} = \bar{x}
$$

**기하분포** $X \sim \text{Geometric}(p)$ (첫 성공까지의 시행 횟수) — 로그가능도는 $\ell(p) = (\sum x_i - n) \log(1-p) + n \log p$이다.

$$
\hat{p} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}
$$

### 연속 분포

**정규분포** $X \sim \mathcal{N}(\mu, \sigma^2)$ — 로그가능도는 다음과 같다.

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

$\mu$에 대해 미분하면 $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum(x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$이다.

$\sigma^2$에 대해 미분하면 $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \mu)^2 = 0$이고, 다음을 얻는다.

$$
\hat{\mu} = \bar{x}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

!!! warning "편향된 분산 추정량"
    분산의 MLE는 **편향되어 있다**. $\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2 < \sigma^2$이다. 불편 추정량은 분모에 $n-1$을 쓴다. 다만 MLE는 일치성을 가지므로 $n \to \infty$일 때 편향이 사라진다.

**지수분포** $X \sim \text{Exponential}(\lambda)$ — 확률밀도함수가 $p(x|\lambda) = \lambda e^{-\lambda x}$일 때 다음과 같다.

$$
\ell(\lambda) = n \log \lambda - \lambda \sum x_i \implies \hat{\lambda} = \frac{1}{\bar{x}}
$$

**균등분포** $X \sim \text{Uniform}(a, b)$ — 가능도는 $a \leq x_{(1)}$이고 $x_{(n)} \leq b$일 때 $L(a,b) = (b-a)^{-n}$이며 그 밖에서는 0이다. 최대화하려면 구간을 최대한 좁힌다.

$$
\hat{a} = x_{(1)} = \min_i x_i, \quad \hat{b} = x_{(n)} = \max_i x_i
$$

!!! note "정칙이 아닌 MLE"
    균등분포는 받침이 매개변수에 의존하므로 "정칙이 아닌" 경우이다. MLE는 존재하지만 통상적인 정칙 조건을 만족하지 않는다(예: 피셔 정보가 표준적인 방식으로는 잘 정의되지 않는다).

**감마분포** $X \sim \text{Gamma}(\alpha, \beta)$ — 두 매개변수 모두에 대한 닫힌 형태의 해는 없다. $\alpha$이 주어졌을 때 $\beta$은 $\hat{\beta} = \alpha / \bar{x}$이다. $\alpha$은 수치적으로 푼다.

$$
\log \alpha - \psi(\alpha) = \log \bar{x} - \overline{\log x}
$$

여기서 $\psi(\alpha) = \frac{d}{d\alpha}\log\Gamma(\alpha)$은 디감마 함수이다.

**다변량 정규분포** $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$
\hat{\boldsymbol{\mu}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i = \bar{\mathbf{x}}, \quad \hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T
$$

### 요약 표

| 분포 | 매개변수 | MLE 추정량 |
|-------------|------------|---------------|
| 베르누이 | $p$ | $\hat{p} = \bar{x}$ |
| 이항 | $p$ ($n$이 주어짐) | $\hat{p} = \bar{x}/n$ |
| 범주형 | $p_1, \ldots, p_K$ | $\hat{p}_k = n_k/n$ |
| 포아송 | $\lambda$ | $\hat{\lambda} = \bar{x}$ |
| 기하 | $p$ | $\hat{p} = 1/\bar{x}$ |
| 정규 | $\mu, \sigma^2$ | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ |
| 지수 | $\lambda$ | $\hat{\lambda} = 1/\bar{x}$ |
| 균등 | $a, b$ | $\hat{a} = \min(x_i)$, $\hat{b} = \max(x_i)$ |
| 감마 | $\alpha, \beta$ | 수치적 해가 필요하다 |

### 되풀이되는 형태 알아보기

MLE 유도에서 되풀이되는 형태에 주목하라. 위치 매개변수에는 표본평균이 자주 나타나고, MLE는 충분통계량을 통해서만 데이터에 의존하며, 확률 벡터의 제약은 라그랑주 승수로 다루고, 비율 매개변수의 MLE는 흔히 $1/\bar{x}$의 꼴을 갖는다.

## 피셔 정보

### 정의

점수 함수 $s(\theta) = \frac{\partial}{\partial \theta} \log p(X|\theta)$을 떠올리자. **피셔 정보**는 점수의 분산이다.

$$
I(\theta) = \text{Var}_\theta[s(\theta)] = \mathbb{E}_\theta\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]
$$

정칙 조건 아래에서(미분과 적분의 순서를 바꿀 수 있을 때) 다음의 동등한 공식이 성립한다.

$$
I(\theta) = -\mathbb{E}_\theta\left[\frac{\partial^2 \log p(X|\theta)}{\partial \theta^2}\right]
$$

이 두 번째 형태는 **피셔 정보가 로그가능도의 기대 곡률과 같음**을 보여준다.

### 직관적인 해석

피셔 정보가 크다는 것은 로그가능도가 $\theta$ 주변에서 날카로운 봉우리를 이룬다는 뜻이다. $\theta$이 조금만 바뀌어도 가능도가 크게 달라지므로 데이터가 $\theta$에 대해 많은 정보를 준다. 피셔 정보가 작다는 것은 로그가능도가 평평하다는 뜻이며, 데이터가 서로 다른 매개변수 값을 잘 구별하지 못한다.

### 흔한 분포들의 피셔 정보

**베르누이** $X \sim \text{Bernoulli}(p)$: $\frac{\partial^2 \log p}{\partial p^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}$을 계산하고 기댓값을 취하면 다음을 얻는다.

$$
\boxed{I(p) = \frac{1}{p(1-p)}}
$$

피셔 정보는 $p = 0.5$일 때(불확실성이 가장 클 때) 가장 크고 $p = 0$이나 $p = 1$ 근처에서 가장 작다.

분산을 아는 **정규분포** $X \sim \mathcal{N}(\mu, \sigma^2)$: $I(\mu) = 1/\sigma^2$이다. 평균을 아는 경우에는 $I(\sigma^2) = 1/(2\sigma^4)$이다.

**포아송** $X \sim \text{Poisson}(\lambda)$: $I(\lambda) = 1/\lambda$이다.

**지수분포** $X \sim \text{Exponential}(\lambda)$: $I(\lambda) = 1/\lambda^2$이다.

### 피셔 정보 행렬

매개변수 벡터 $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)^T$에 대해 **피셔 정보 행렬**은 다음과 같다.

$$
\mathbf{I}(\boldsymbol{\theta})_{ij} = -\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\right]
$$

$\boldsymbol{\theta} = (\mu, \sigma^2)$인 $X \sim \mathcal{N}(\mu, \sigma^2)$에 대해 다음과 같다.

$$
\mathbf{I}(\mu, \sigma^2) = \begin{pmatrix}
\frac{1}{\sigma^2} & 0 \\
0 & \frac{1}{2\sigma^4}
\end{pmatrix}
$$

대각 구조는 $\mu$과 $\sigma^2$이 **직교**함을 뜻한다. 한쪽에 대한 정보가 다른 쪽을 알려 주지 않는다.

### 성질

**가법성.** i.i.d. 관측 $n$개에 대해 $I_n(\theta) = n \cdot I(\theta)$이다. 데이터가 많을수록 추정이 정밀해진다.

**재매개화.** $\eta = g(\theta)$이 일대일 변환이면 $I_\eta(\eta) = I_\theta(\theta) \cdot (d\theta/d\eta)^2$이다. 매개변수가 여럿이면 $\mathbf{J}$을 야코비 행렬이라 할 때 $\mathbf{I}_\eta = \mathbf{J}^T \mathbf{I}_\theta \mathbf{J}$이다.

### 크라메르–라오 하한

피셔 정보는 어떤 불편 추정량도 넘을 수 없는 정밀도의 근본적인 한계를 정한다.

**정리.** $\theta$의 임의의 불편 추정량 $\hat{\theta}$에 대해 다음이 성립한다.

$$
\boxed{\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}}
$$

이 하한은 **효율 추정량**이 달성하며, MLE는 점근적으로 효율적이다.

i.i.d. 표본이 $n$개이면 하한은 $\text{Var}(\hat{\theta}) \geq 1/(nI(\theta))$이 된다.

### 관측 피셔 정보와 기대 피셔 정보

**기대** 피셔 정보 $I(\theta) = -\mathbb{E}[\partial^2 \ell / \partial \theta^2]$은 참 매개변수에서 계산하며 이론적 분석에 쓴다. **관측** 피셔 정보 $J(\hat{\theta}) = -\partial^2 \ell / \partial \theta^2 \big|_{\theta = \hat{\theta}}$은 MLE에서 계산하며 실무에서 신뢰구간을 구할 때 쓴다.

## MLE의 점근적 성질

정칙 조건 아래에서 MLE는 표본 크기가 커질수록 놀라운 성질을 갖는다. 이 성질들이 MLE가 통계적 추정의 일꾼인 이유와, 신경망 학습이 큰 데이터셋에서 잘 작동하는 이유를 설명해 준다.

!!! abstract "핵심 점근적 성질"
    정칙 조건 아래에서 $n \to \infty$일 때 다음이 성립한다.
    
    1. **일치성**: $\hat{\theta}_n \xrightarrow{p} \theta_0$
    2. **점근 정규성**: $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$
    3. **효율성**: MLE는 점근적으로 크라메르–라오 하한을 달성한다
    4. **불변성**: $g(\theta)$의 MLE는 $g(\hat{\theta})$이다

### 정칙 조건

점근적 성질은 다음 조건 아래에서 성립한다.

1. **식별 가능성**: $\theta_1 \neq \theta_2 \implies p(x|\theta_1) \neq p(x|\theta_2)$
2. **공통 받침**: $p(x|\theta)$의 받침이 $\theta$에 의존하지 않는다
3. **미분 가능성**: $\log p(x|\theta)$이 $\theta$에 대해 세 번 미분 가능하다
4. **유계인 도함수**: 삼계도함수가 적분 가능한 함수로 유계이다
5. **열린 매개변수 공간**: 참 매개변수 $\theta_0$이 $\Theta$의 내부에 있다

!!! warning "정칙성이 깨질 때"
    중요한 분포들 중 일부는 이 조건을 어긴다. 균등분포 $[0, \theta]$(받침이 $\theta$에 의존), 혼합 모델(국소 최댓값이 여럿), 경계 사례(매개변수가 $\Theta$의 경계에 있음) 등이다.

### 일치성

$n \to \infty$일 때 $\hat{\theta}_n \xrightarrow{p} \theta_0$이면 추정량 $\hat{\theta}_n$은 **일치성**을 갖는다. 핵심 통찰은 로그가능도를 최대화하는 것이 KL 발산을 최소화하는 것과 같다는 점이다. 큰 수의 법칙에 의해 다음이 성립한다.

$$
\frac{1}{n}\sum_{i=1}^{n} \log p(x_i | \theta) \xrightarrow{p} \mathbb{E}_{\theta_0}[\log p(X | \theta)]
$$

그리고 $\mathbb{E}_{\theta_0}[\log p(X | \theta)]$은 $\theta = \theta_0$에서 최대가 된다(정보 부등식).

### 점근 정규성

정칙 조건 아래에서 다음이 성립한다.

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, I(\theta_0)^{-1}\right)
$$

??? info "유도 개요"

    1. 점수를 $\theta_0$ 주변에서 **테일러 전개**한다: $s(\hat{\theta}) = s(\theta_0) + (\hat{\theta} - \theta_0) s'(\tilde{\theta})$
    2. **MLE에서** $s(\hat{\theta}) = 0$이므로 $\sqrt{n}(\hat{\theta} - \theta_0) = -\frac{\sqrt{n} \cdot s(\theta_0)/n}{s'(\tilde{\theta})/n}$이다
    3. **중심극한정리에 의해**: $\sqrt{n} \cdot \bar{s}(\theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0))$
    4. **큰 수의 법칙에 의해**: $s'(\tilde{\theta})/n \xrightarrow{p} -I(\theta_0)$
    5. **슬러츠키 정리에 의해**: 그 비는 $\mathcal{N}(0, I(\theta_0)^{-1})$으로 수렴한다

매개변수 벡터의 경우 $\sqrt{n}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}_0) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathbf{I}(\boldsymbol{\theta}_0)^{-1})$이다.

### 효율성

일치성과 점근 정규성을 갖는 모든 추정량 중에서 MLE는 **점근 분산이 가장 작다**.

$$
\text{Avar}(\hat{\theta}_{\text{MLE}}) = \frac{1}{I(\theta_0)} \leq \text{Avar}(\hat{\theta}_{\text{other}})
$$

**점근 상대 효율**(ARE)은 두 추정량을 비교한다. $\text{ARE}(\hat{\theta}_1, \hat{\theta}_2) = \text{Avar}(\hat{\theta}_2) / \text{Avar}(\hat{\theta}_1)$이다. 정규분포의 평균에 대해 표본 중앙값은 MLE(표본평균) 대비 ARE $\approx 2/\pi \approx 0.637$을 갖는다.

### 불변성

$\hat{\theta}$이 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $\widehat{g(\theta)} = g(\hat{\theta})$이다. 예를 들어 $\mu$의 MLE가 $\bar{x}$이면 $e^\mu$의 MLE는 $e^{\bar{x}}$이다.

!!! warning "불변성에서 오는 편향"
    불변성은 편리하지만, $\hat{\theta}$이 불편이더라도 $g(\hat{\theta})$은 편향될 수 있다. 예를 들어 $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$은 MLE인데도 편향되어 있다.

### 수렴 속도

정칙 조건 아래에서 $\|\hat{\theta}_n - \theta_0\| = O_p(n^{-1/2})$이다. 이는 추정 오차를 절반으로 줄이려면 데이터가 4배 필요하고, 정밀도를 10배로 높이려면 데이터가 100배 필요하다는 뜻이다.

### 신뢰구간

**왈드 구간** (점근 정규성으로부터):

$$
\hat{\theta} \pm z_{\alpha/2} \sqrt{\frac{1}{nI(\hat{\theta})}}
$$

**프로파일 가능도 구간** (가능도비 통계량으로부터): $\{\theta : 2[\ell(\hat{\theta}) - \ell(\theta)] \leq \chi^2_{1, \alpha}\}$이다. 표본이 작을 때는 프로파일 가능도 구간을 선호하는 경우가 많다.

## 딥러닝과의 관계

MLE와 딥러닝 손실 함수의 관계는 근본적이다.

$$
\text{Loss}(\theta) = -\ell(\theta | \mathbf{X}) = -\log L(\theta | \mathbf{X})
$$

즉 손실 최소화 = 가능도 최대화이고, 손실에 대한 경사 하강 = 로그가능도에 대한 경사 상승이며, 교차 엔트로피 손실 = 분류에서의 음의 로그가능도, MSE 손실 = 가우스 회귀에서의 음의 로그가능도이다.

!!! warning "딥러닝과의 연결"
    교차 엔트로피나 MSE를 최소화하여 신경망을 학습시킬 때 당신은 최대가능도 추정을 하고 있는 것이다. 유일한 차이는 모델 $p(y|x, \theta)$이 신경망으로 매개화되어 있다는 점뿐이다.

점근적 성질은 딥러닝의 현상들도 설명해 준다. 일치성은 큰 데이터셋을 쓰는 것을 정당화하고, 점근 정규성은 불확실성의 정량화를 가능케 하며, $\sqrt{n}$ 수렴 속도는 데이터 효율의 절충을 지배한다.

### 최적화에서의 응용

**자연 경사 하강법**은 피셔 정보 행렬을 써서 매개변수 공간의 기하를 고려한다.

$$
\theta \leftarrow \theta - \alpha \mathbf{I}(\theta)^{-1} \nabla_\theta L
$$

교차 엔트로피 손실로 학습한 신경망에서 피셔 정보 행렬은 헤세 행렬을 근사한다($\mathbf{I}(\theta) \approx \mathbf{H}(\theta)$). 이것이 효율적인 이차 최적화를 위한 K-FAC(크로네커 인수분해 근사 곡률) 같은 방법을 정당화한다.

## PyTorch 구현

### 해석적 MLE와 경사 기반 MLE

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_coin_flips(n_flips: int, true_p: float, seed: int = 42) -> torch.Tensor:
    """합성 베르누이 데이터(동전 던지기)를 생성한다."""
    torch.manual_seed(seed)
    return (torch.rand(n_flips) < true_p).float()

def compute_log_likelihood(data: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    베르누이 분포의 로그 가능도를 셈한다.
    
    ℓ(p) = Σ[x_i * log(p) + (1-x_i) * log(1-p)]
    """
    epsilon = 1e-8
    p = torch.clamp(p, epsilon, 1 - epsilon)
    return torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p))

def analytical_mle(data: torch.Tensor) -> float:
    """MLE를 해석적으로 계산한다: p̂ = k/n"""
    return data.mean().item()

def gradient_based_mle(data: torch.Tensor, 
                       lr: float = 0.1, 
                       n_iter: int = 500) -> tuple:
    """
    기울기 내림으로 최대가능도를 셈한다.
    
    최대가능도와 가장 좋게 하기 사이의 이음을 보인다.
    이것이 깊은 배움 전체를 받친다.
    """
    # 매개변수를 초기화한다 (제약 없는 최적화를 위한 시그모이드 매개화)
    logit_p = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([logit_p], lr=lr)
    
    history = []
    for i in range(n_iter):
        p = torch.sigmoid(logit_p)
        nll = -compute_log_likelihood(data, p)  # Minimize negative log-likelihood
        
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
        
        history.append(p.item())
    
    return torch.sigmoid(logit_p).item(), history

# --- 사용 예 ---
TRUE_P = 0.7
N_FLIPS = 100

data = generate_coin_flips(N_FLIPS, TRUE_P)
n_heads = int(data.sum().item())

print(f"Data: {n_heads} heads out of {N_FLIPS} flips")
print(f"True p: {TRUE_P}")
print(f"Analytical MLE: {analytical_mle(data):.4f}")

p_gradient, history = gradient_based_mle(data)
print(f"Gradient-based MLE: {p_gradient:.4f}")
```

### 흔한 분포들의 MLE

```python
def bernoulli_mle(data: torch.Tensor) -> float:
    """베르누이 매개변수 p의 MLE."""
    return data.mean().item()

def categorical_mle(data: torch.Tensor, num_categories: int) -> torch.Tensor:
    """범주형 분포의 MLE: p_k = n_k / n."""
    counts = torch.bincount(data.long(), minlength=num_categories).float()
    return counts / counts.sum()

def poisson_mle(data: torch.Tensor) -> float:
    """포아송 비율 매개변수 λ의 MLE."""
    return data.float().mean().item()

def normal_mle(data: torch.Tensor) -> tuple:
    """정규분포의 MLE (분산 추정량은 편향되어 있다)."""
    mu_hat = data.mean()
    sigma_hat = data.std(unbiased=False)
    return mu_hat.item(), sigma_hat.item()

def exponential_mle(data: torch.Tensor) -> float:
    """지수분포 비율 매개변수 λ의 MLE."""
    return 1.0 / data.mean().item()

def uniform_mle(data: torch.Tensor) -> tuple:
    """균등분포 매개변수의 MLE."""
    return data.min().item(), data.max().item()

def multivariate_normal_mle(data: torch.Tensor) -> tuple:
    """다변량 정규분포의 MLE: 평균 벡터와 공분산 행렬."""
    n = data.shape[0]
    mu_hat = data.mean(dim=0)
    centered = data - mu_hat
    sigma_hat = (centered.T @ centered) / n
    return mu_hat, sigma_hat
```

### 범용 경사 기반 MLE

```python
def gradient_mle(data: torch.Tensor, 
                 log_likelihood_fn: callable,
                 init_params: dict,
                 lr: float = 0.01,
                 n_iter: int = 1000) -> dict:
    """
    PyTorch 자동 미분을 쓰는 두루 쓰는 기울기 바탕 최대가능도.
    
    닫힌 꼴 최대가능도가 없는 분포나 배움 목적에 쓴다.
    """
    params = {k: v.clone().requires_grad_(True) for k, v in init_params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    
    for i in range(n_iter):
        nll = -log_likelihood_fn(data, **params)
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
    
    return {k: v.detach() for k, v in params.items()}
```

### 피셔 정보와 크라메르–라오 하한

```python
def compute_fisher_information_matrix(data: torch.Tensor,
                                      log_likelihood_fn: callable,
                                      params: torch.Tensor,
                                      eps: float = 1e-4) -> torch.Tensor:
    """
    피셔 정보 행렬을 수치로 셈한다.
    
    최대가능도에서의 관측 피셔 정보(음의 헤세)를 유한 차분으로 쓴다.
    I_ij = -∂²ℓ/∂θ_i∂θ_j
    """
    n_params = len(params)
    hessian = torch.zeros(n_params, n_params)
    
    for i in range(n_params):
        for j in range(n_params):
            params_pp = params.clone(); params_pp[i] += eps; params_pp[j] += eps
            params_pm = params.clone(); params_pm[i] += eps; params_pm[j] -= eps
            params_mp = params.clone(); params_mp[i] -= eps; params_mp[j] += eps
            params_mm = params.clone(); params_mm[i] -= eps; params_mm[j] -= eps
            
            ll_pp = log_likelihood_fn(data, params_pp)
            ll_pm = log_likelihood_fn(data, params_pm)
            ll_mp = log_likelihood_fn(data, params_mp)
            ll_mm = log_likelihood_fn(data, params_mm)
            
            hessian[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps**2)
    
    return -hessian

def verify_cramer_rao(true_p: float = 0.3, n: int = 100, n_simulations: int = 10000):
    """베르누이에서 MLE의 분산이 크라메르–라오 하한과 일치하는지 확인한다."""
    torch.manual_seed(42)
    
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_p).float()
        estimates.append(data.mean().item())
    
    empirical_var = np.var(estimates)
    cramer_rao = true_p * (1 - true_p) / n
    
    print(f"True p: {true_p}, Sample size: {n}")
    print(f"Cramér–Rao bound: {cramer_rao:.6f}")
    print(f"Empirical variance: {empirical_var:.6f}")
    print(f"Ratio (should be ≈ 1): {empirical_var / cramer_rao:.4f}")
```

### 점근적 성질 보여주기

```python
def demonstrate_consistency(true_theta: float = 0.7, 
                           sample_sizes: list = None,
                           n_simulations: int = 1000):
    """n이 커질수록 MLE가 참값 주위로 모여드는 것을 보여준다."""
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000, 5000]
    
    torch.manual_seed(42)
    
    print("MLE Consistency Demonstration")
    print("-" * 50)
    print(f"{'n':>8} {'Mean MLE':>12} {'Std MLE':>12} {'|Bias|':>12}")
    print("-" * 50)
    
    for n in sample_sizes:
        estimates = []
        for _ in range(n_simulations):
            data = (torch.rand(n) < true_theta).float()
            estimates.append(data.mean().item())
        
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        bias = abs(mean_est - true_theta)
        print(f"{n:>8} {mean_est:>12.6f} {std_est:>12.6f} {bias:>12.6f}")

def demonstrate_asymptotic_normality(true_theta: float = 0.7, 
                                     n: int = 100,
                                     n_simulations: int = 5000):
    """MLE의 경험적 분포를 이론적 정규 근사와 비교한다."""
    from scipy.stats import norm
    
    torch.manual_seed(42)
    
    fisher_info = 1 / (true_theta * (1 - true_theta))
    asymptotic_var = 1 / (n * fisher_info)
    asymptotic_std = np.sqrt(asymptotic_var)
    
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_theta).float()
        estimates.append(data.mean().item())
    
    estimates = np.array(estimates)
    
    print(f"Asymptotic Normality Check (n = {n}):")
    print(f"  Theoretical std: {asymptotic_std:.6f}")
    print(f"  Empirical std:   {np.std(estimates):.6f}")
    print(f"  Ratio (≈ 1):    {np.std(estimates)/asymptotic_std:.4f}")
```

### 가능도 함수 시각화하기

```python
def plot_likelihood_analysis(data: torch.Tensor, true_p: float):
    """가능도 함수, 정규화된 가능도, 경사 하강법의 수렴을 시각화한다."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    p_values = np.linspace(0.01, 0.99, 200)
    
    # 로그가능도
    log_liks = [compute_log_likelihood(data, torch.tensor(p)).item() for p in p_values]
    
    ax = axes[0]
    ax.plot(p_values, log_liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--', label=f'True p = {true_p}')
    ax.axvline(analytical_mle(data), color='red', linestyle='-', 
               label=f'MLE = {analytical_mle(data):.3f}')
    ax.set_xlabel('p'); ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood Function')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 정규화된 가능도
    liks = np.exp(np.array(log_liks) - max(log_liks))
    ax = axes[1]
    ax.plot(p_values, liks, 'b-', linewidth=2)
    ax.axvline(true_p, color='green', linestyle='--')
    ax.axvline(analytical_mle(data), color='red', linestyle='-')
    ax.fill_between(p_values, liks, alpha=0.3)
    ax.set_xlabel('p'); ax.set_ylabel('Normalized Likelihood')
    ax.set_title('Likelihood Function')
    ax.grid(True, alpha=0.3)
    
    # 경사 하강법의 수렴
    _, history = gradient_based_mle(data)
    ax = axes[2]
    ax.plot(history, 'b-', linewidth=2)
    ax.axhline(true_p, color='green', linestyle='--', label='True p')
    ax.axhline(analytical_mle(data), color='red', linestyle='-', label='MLE')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Estimated p')
    ax.set_title('Gradient Descent Convergence')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## 참고 문헌

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 1.2.4
- Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd Edition. Chapters 7, 10
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd Edition
- Lehmann, E. L. & Casella, G. (1998). *Theory of Point Estimation*, 2nd Edition
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. Chapter 4.2
- van der Vaart, A. W. (1998). *Asymptotic Statistics*

## 연습문제

**연습문제 1.**
관측 $n$개 중 성공이 $k$번일 때 베르누이 분포의 매개변수 $p$에 대한 MLE를 유도하라.

??? success "연습문제 1 풀이"
    가능도는 $L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^k(1-p)^{n-k}$이다.

    로그가능도: $\ell(p) = k\log p + (n-k)\log(1-p)$.

    $\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$으로 두고 풀면 $k(1-p) = (n-k)p$이 되어 표본 비율 $\hat{p}_{\text{MLE}} = k/n$을 얻는다.

    이계도함수가 $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$이므로 최댓값임이 확인된다. $\square$

---

**연습문제 2.**
정규분포의 분산에 대한 MLE $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2$이 편향되어 있음을 증명하고, 편향을 명시적으로 계산하라.

??? success "연습문제 2 풀이"
    $$
    \mathbb{E}[\hat{\sigma}^2] = \mathbb{E}\left[\frac{1}{n}\sum(x_i - \bar{x})^2\right] = \frac{1}{n}\mathbb{E}\left[\sum x_i^2 - n\bar{x}^2\right]
    $$

    $\mathbb{E}[x_i^2] = \sigma^2 + \mu^2$이고 $\mathbb{E}[\bar{x}^2] = \sigma^2/n + \mu^2$이므로 다음이 성립한다.

    $$
    \mathbb{E}[\hat{\sigma}^2] = \frac{1}{n}[n(\sigma^2+\mu^2) - n(\sigma^2/n + \mu^2)] = \frac{n-1}{n}\sigma^2
    $$

    편향은 $\mathbb{E}[\hat{\sigma}^2] - \sigma^2 = -\sigma^2/n$이며 $n \to \infty$일 때 사라진다. 불편 추정량은 분모에 $n-1$을 쓴다.

---

**연습문제 3.**
라플라스 분포 $p(x|\mu, b) = \frac{1}{2b}e^{-|x-\mu|/b}$에서 $\mu$의 MLE가 표본 중앙값임을 보여라.

??? success "연습문제 3 풀이"
    로그가능도는 $\ell(\mu) = -n\log(2b) - \frac{1}{b}\sum_{i=1}^n |x_i - \mu|$이다.

    $\ell$을 최대화하는 것은 절대 편차의 합 $\sum|x_i - \mu|$을 최소화하는 것과 같다. 열경사는 $\frac{\partial}{\partial\mu}\sum|x_i-\mu| = -\sum\text{sign}(x_i - \mu)$이다.

    0으로 두면 $x_i > \mu$인 개수와 $x_i < \mu$인 개수가 같아야 하는데, 이것이 바로 중앙값의 정의이다. $\square$

---

**연습문제 4.**
베르누이 분포의 피셔 정보를 유도하고, MLE가 크라메르–라오 하한을 달성함을 확인하라.

??? success "연습문제 4 풀이"
    점수 함수: $s(p) = \frac{x}{p} - \frac{1-x}{1-p}$.

    피셔 정보: $I(p) = \mathbb{E}[s(p)^2] = \mathbb{E}\left[\left(\frac{x-p}{p(1-p)}\right)^2\right] = \frac{\text{Var}(x)}{[p(1-p)]^2} = \frac{1}{p(1-p)}$.

    i.i.d. 관측 $n$개에 대한 크라메르–라오 하한은 $\text{Var}(\hat{p}) \geq \frac{1}{nI(p)} = \frac{p(1-p)}{n}$이다.

    MLE $\hat{p} = \bar{x}$의 분산은 $\text{Var}(\hat{p}) = p(1-p)/n$으로 하한을 정확히 달성한다. 따라서 MLE는 효율적이다. $\square$
