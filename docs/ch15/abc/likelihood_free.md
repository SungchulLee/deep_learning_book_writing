# 가능도 없는 추론
과학 모형 가운데 많은 것이 자료를 흉내 낼 수는 있어도 관측의 가능도를 셈하지는 못한다. 흉내 기반 추론이라고도 하는 가능도 없는 추론은 그런 자리에서 베이즈 추론을 하는 방법을 준다. 이 마당은 문제를 들여오고, ABC 얼개가 나온 까닭을 밝히며, 가능도 없는 방법의 갈래를 훑는다.

---

## 다룰 수 없는 가능도 문제

### 가능도를 쓸 수 없을 때

표준 베이즈 추론은 가능도 $p(\mathbf{y} | \theta)$의 값을 매겨야 한다:

$$
p(\theta | \mathbf{y}) = \frac{p(\mathbf{y} | \theta) p(\theta)}{p(\mathbf{y})}
$$

그러나 많은 모형이 **흉내내기 장치**로 정해진다. 곧 매개변수 $\theta$이 주어지면 흉내 낸 자료 $\mathbf{x} \sim p(\cdot | \theta)$을 만들 수 있지만, 관측 자료 $\mathbf{y}$에 대해 $p(\mathbf{y} | \theta)$의 값을 매기지는 못한다.

### 흉내내기 모형의 보기

**집단 유전학**: 합쳐짐 모형은 계보와 유전 변이를 흉내 내지만, 가능도는 있을 수 있는 모든 계보에 걸쳐 적분해야 해서 셈으로 다룰 수 없다.

**역학**: 행위자 기반 질병 모형은 낱낱의 감염과 접촉을 흉내 낸다. 가능도는 있을 수 있는 모든 옮김 내력에 걸쳐 합해야 한다.

**생태학**: 개체 기반 모형은 동물의 움직임, 태어남, 죽음, 서로 오감을 흉내 낸다. 닫힌 꼴 가능도가 없다.

**우주론**: N체 흉내내기는 물질 분포를 만든다. 관측한 은하 자리의 가능도는 다룰 수 없다.

**신경과학**: 생물물리 뉴런 모형은 스파이크 열을 흉내 낸다. 가능도는 관측하지 못한 속 상태에 걸쳐 적분해야 한다.

**경제학**: 행위자 기반 시장 모형은 거래자의 굴러감을 흉내 낸다. 가격 시계열의 가능도는 쓸 수 없다.

### 흉내내기 장치라는 추상

**흉내내기 장치**(낳는 모형)는 확률 함수이다:

$$
\mathbf{x} = f(\theta, \mathbf{u})
$$

여기서 각 기호는 다음과 같다.

- $\theta$은 우리가 궁금해하는 매개변수이다
- $\mathbf{u}$은 무작위 잡음이다(흔히 무작위 수가 많다)
- $\mathbf{x}$은 흉내 내어 나온 것이다

$\mathbf{u}$을 뽑고 $f$을 셈해 $\mathbf{x} | \theta$을 표집할 수 있지만 $p(\mathbf{x} | \theta)$은 셈할 수 없다.

---

## 표준 방법이 왜 무너지나

### MCMC는 가능도 값 매기기가 필요하다

메트로폴리스-헤이스팅스는 다음을 셈해야 한다:

$$
\alpha = \min\left(1, \frac{p(\mathbf{y} | \theta') p(\theta')}{p(\mathbf{y} | \theta) p(\theta)} \cdot \frac{q(\theta | \theta')}{q(\theta' | \theta)}\right)
$$

$p(\mathbf{y} | \theta)$이 없으면 $\alpha$을 셈할 수 없다.

### 중요도 표집은 가능도가 필요하다

중요도 표집의 무게는 다음과 같다:

$$
w(\theta) = \frac{p(\mathbf{y} | \theta) p(\theta)}{q(\theta)}
$$

여기서도 $p(\mathbf{y} | \theta)$이 필요하다.

### 변분 추론은 가능도가 필요하다

ELBO에는 다음이 들어간다:

$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{y} | \theta)] - D_{KL}(q(\theta) \| p(\theta))
$$

첫 항이 가능도를 필요로 한다.

---

## ABC의 생각

### 핵심 통찰

$p(\mathbf{y} | \theta)$을 셈할 수 없으면 거기서 **흉내 낼** 수는 있다. ABC의 길은 이렇다:

1. 앞확률(또는 제안)에서 $\theta$을 내놓는다
2. $\mathbf{x} \sim p(\cdot | \theta)$을 흉내 낸다
3. $\mathbf{x}$을 관측 $\mathbf{y}$과 견준다
4. $\mathbf{x} \approx \mathbf{y}$이면 $\theta$을 받아들인다

받아들인 $\theta$ 값이 어림 뒤확률 표본을 이룬다.

### 정확한 (실전에서는 못 쓰는) 판

$\mathbf{x} = \mathbf{y}$이 딱 맞을 때만 받아들이면:

$$
p(\theta | \mathbf{x} = \mathbf{y}) = p(\theta | \mathbf{y})
$$

정확한 뒤확률이 나온다! 그러나 이어진 자료에서는 $P(\mathbf{x} = \mathbf{y}) = 0$이다.

### 어림 판

$\mathbf{x}$이 $\mathbf{y}$에 "넉넉히 가까우면" 받아들인다:

$$
\rho(\mathbf{x}, \mathbf{y}) < \epsilon
$$

여기서 $\rho$은 거리 잣대이고 $\epsilon$은 너그러움이다.

이는 **ABC 뒤확률**을 겨냥한다:

$$
p_{ABC}(\theta | \mathbf{y}) \propto p(\theta) \int p(\mathbf{x} | \theta) \mathbf{1}[\rho(\mathbf{x}, \mathbf{y}) < \epsilon] \, d\mathbf{x}
$$

---

## ABC 뒤확률

### 해석

ABC 뒤확률은 흉내 낸 자료가 관측 자료에서 $\epsilon$ 안에 있다는 조건 아래의 뒤확률이다:

$$
p_{ABC}(\theta | \mathbf{y}) = p(\theta | \rho(\mathbf{X}, \mathbf{y}) < \epsilon)
$$

여기서 $\mathbf{X} \sim p(\cdot | \theta)$이다.

### 참 뒤확률과의 관계

$\epsilon \to 0$이면:

$$
p_{ABC}(\theta | \mathbf{y}) \to p(\theta | \mathbf{y})
$$

$\epsilon$이 끝이 있으면 ABC 뒤확률은 참 뒤확률을 매끄럽게 하고 넓힌 판이다.

### 치우침과 흩어짐의 주고받음

| 작은 $\epsilon$ | 큰 $\epsilon$ |
|------------------|------------------|
| 치우침이 작다 | 치우침이 크다 |
| 흩어짐이 크다(드물게 받아들인다) | 흩어짐이 작다(많이 받아들인다) |
| 참 뒤확률에 더 가깝다 | 참 뒤확률에서 더 멀다 |
| 셈이 비싸다 | 셈이 싸다 |

---

## 간추린 통계량

### 차원의 저주

차원이 높은 $\mathbf{x}$과 $\mathbf{y}$을 곧바로 견주는 것은 말썽이다:

- 무작위 $\mathbf{x}$이 $\mathbf{y}$에 가까운 일은 거의 없다
- 받아들임 비율이 사라질 만큼 작아진다
- 흉내내기가 지수만큼 많이 필요하다

### 간추린 통계량으로 차원 줄이기

날 자료를 **간추린 통계량** $S(\mathbf{x})$으로 바꾼다:

$$
\rho(S(\mathbf{x}), S(\mathbf{y})) < \epsilon
$$

이제 차원이 낮은 간추림끼리 견준다.

### 충분함의 물음

**충분 통계량**: $p(\mathbf{y} | \theta) = p(\mathbf{y} | S(\mathbf{y}), \theta) p(S(\mathbf{y}) | \theta)$이면 $S$은 $\theta$에 충분하다.

$S$이 충분하면:

$$
p_{ABC}(\theta | S(\mathbf{y})) = p(\theta | S(\mathbf{y})) = p(\theta | \mathbf{y})
$$

충분 통계량을 쓴 ABC은($\epsilon \to 0$과 함께) 정확한 뒤확률을 준다.

### 문제: 충분 통계량은 드물다

복잡한 모형 대부분에서는:

- 차원이 끝이 있는 충분 통계량이 없다
- 대충 충분하거나 어림짐작으로 만든 간추림을 써야 한다
- 정보를 잃는 일을 피할 수 없다

### 간추린 통계량 고르기

**분야 지식**: 관련 있는 특징을 담는 통계량.

**저절로 되는 방법**:

- 반자동 ABC(Fearnhead & Prangle, 2012)
- 신경망 묻힘
- 정보 이론으로 고르기

---

## 거리 함수

### 흔히 고르는 것

**유클리드 거리**:

$$
\rho(\mathbf{x}, \mathbf{y}) = \|S(\mathbf{x}) - S(\mathbf{y})\|_2
$$

**무게 준 유클리드**:

$$
\rho(\mathbf{x}, \mathbf{y}) = \sqrt{(S(\mathbf{x}) - S(\mathbf{y}))^T W (S(\mathbf{x}) - S(\mathbf{y}))}
$$

여기서 $W$은 눈금이 다른 것을 헤아린다.

**마할라노비스 거리**:

$$
\rho(\mathbf{x}, \mathbf{y}) = \sqrt{(S(\mathbf{x}) - S(\mathbf{y}))^T \Sigma^{-1} (S(\mathbf{x}) - S(\mathbf{y}))}
$$

여기서 $\Sigma$은 앞확률 예측 아래 $S(\mathbf{X})$의 공분산이다.

### 알맹이 ABC

딱딱한 문턱값을 부드러운 알맹이로 바꾼다:

$$
K_\epsilon(\mathbf{x}, \mathbf{y}) = K\left(\frac{\rho(\mathbf{x}, \mathbf{y})}{\epsilon}\right)
$$

흔한 알맹이:

- 고름: $K(u) = \mathbf{1}[u < 1]$
- 가우스: $K(u) = \exp(-u^2/2)$
- 에파네치니코프: $K(u) = (1 - u^2)\mathbf{1}[u < 1]$

---

## 이론의 바탕

### 일치성

규칙 조건 아래 ABC은 **일관적**이다. 곧 (자료 크기) $n \to \infty$이고 (알맞게) $\epsilon \to 0$이면:

$$
p_{ABC}(\theta | \mathbf{y}_n) \to \delta_{\theta_0}
$$

여기서 $\theta_0$은 참 매개변수이다.

### 수렴 속도

모임 속도는 다음에 기댄다:

- 간추린 통계량의 차원
- 모형의 매끄러움
- $\epsilon$ 일정을 고르는 법

차원이 $d$인 충분 통계량에서는:

$$
\epsilon_n \sim n^{-1/(d+4)}
$$

이 평균 제곱 오차를 가장 작게 한다.

### 점근 정규성

어떤 조건 아래 ABC 뒤확률은 점근으로 정규이다:

$$
p_{ABC}(\theta | \mathbf{y}_n) \approx \mathcal{N}(\hat{\theta}_n, V_n)
$$

여기서 $\hat{\theta}_n$은 일관 어림자이고 $V_n \to 0$이다.

---

## 기본 ABC 너머

### 가능도 없는 방법의 갈래

ABC은 한 갈래일 뿐이다. 더 넓게 보면 다음이 있다:

**ABC의 여러 판**:

- ABC 물리치기 표집
- ABC-MCMC
- ABC-SMC(잇단 몬테카를로)
- 회귀 조정

**신경망 기반 가능도 없는 방법**:

- 신경망 뒤확률 어림(NPE)
- 신경망 가능도 어림(NLE)
- 신경망 비 어림(NRE)

**다른 길**:

- 흉내 가능도
- 에두른 추론
- 가능도 없는 추론을 위한 베이즈 최적화

### 신경망 밀도 어림

다음을 어림하도록 신경망을 가르친다:

**뒤확률**(NPE):

$$
q_\phi(\theta | \mathbf{y}) \approx p(\theta | \mathbf{y})
$$

**가능도**(NLE):

$$
q_\phi(\mathbf{y} | \theta) \approx p(\mathbf{y} | \theta)
$$

**가능도 비**(NRE):

$$
r_\phi(\theta, \mathbf{y}) \approx \frac{p(\mathbf{y} | \theta)}{p(\mathbf{y})}
$$

이 방법들은 추론 비용을 나눠 문다. 곧 한번 가르치고 나면 새 관측의 뒤확률 표본을 싸게 얻는다.

### 흉내 가능도

간추린 통계량이 대략 가우스라고 놓는다:

$$
S(\mathbf{X}) | \theta \approx \mathcal{N}(\mu(\theta), \Sigma(\theta))
$$

흉내내기에서 $\mu(\theta)$과 $\Sigma(\theta)$을 어림한 뒤 이 가우스 가능도를 표준 MCMC에 쓴다.

---

## 가능도 없는 방법을 언제 쓰나

### 잘 맞는 경우

✓ 모형이 흉내내기 장치이다(만들 수는 있고 값은 못 매긴다)
✓ 모형이 과학적으로 뒷받침된다(그저 맞추기용이 아니다)
✓ 흉내내기가 그럭저럭 빠르다
✓ 알려 주는 바 있는 간추린 통계량이 있다
✓ 앞확률이 제대로 되어 있고 지나치게 퍼져 있지 않다

### 잘 맞지 않는 경우

✗ 가능도를 다룰 수 있다(표준 방법을 써라!)
✗ 흉내내기가 몹시 느리다
✗ 좋은 간추린 통계량이 알려져 있지 않다
✗ 매개변수 공간의 차원이 아주 높다
✗ 모형을 잘못 잡았다(쓰레기를 넣으면 쓰레기가 나온다)

### 셈에서 살필 점

| 요인 | 영향 |
|--------|--------|
| 흉내내기 값 | 도는 시간을 좌우한다 |
| 매개변수 차원 | 받아들임 비율에 영향을 준다 |
| 간추림의 차원 | 주고받음: 정보와 받아들임 |
| 자료 크기 | 자료가 많을수록 → $\epsilon$이 더 작아야 한다 |

---

## 실전 일머리

### 걸음 1: 모형 확인하기

추론에 앞서 흉내내기 장치를 확인하여라:

- 관측을 닮은 자료를 낼 수 있는가?
- 앞확률이 그럭저럭한가?
- 흉내내기 코드에 벌레가 있는가?

```python
def prior_predictive_check(simulator, prior, n_sims=100):
    """앞확률 미리봄 표본 만들기."""
    samples = []
    for _ in range(n_sims):
        theta = prior.sample()
        x = simulator(theta)
        samples.append({'theta': theta, 'x': x})
    
    # 그려 보기: 참 자료처럼 보이는 표본이 있는가?
    return samples
```

### 걸음 2: 간추린 통계량 고르기

분야에서 뒷받침되는 간추림에서 시작하여라:

```python
def summary_statistics(x):
    """보기: 시계열의 간추린 통계량."""
    return np.array([
        np.mean(x),
        np.std(x),
        np.corrcoef(x[:-1], x[1:])[0, 1],  # 뒤짐 1의 자기상관
        np.percentile(x, [25, 50, 75]),
    ]).flatten()
```

### 걸음 3: 너그러움 맞추기

거리 분포를 알아보려고 미리 흉내내기를 돌려라:

```python
def calibrate_epsilon(simulator, prior, summary_fn, y_obs, n_pilot=1000):
    """그럴듯한 엡실론 범위 정하기."""
    distances = []
    
    s_obs = summary_fn(y_obs)
    
    for _ in range(n_pilot):
        theta = prior.sample()
        x = simulator(theta)
        s_x = summary_fn(x)
        distances.append(np.linalg.norm(s_x - s_obs))
    
    # 앞확률 미리봄 거리의 분위수로 잡은 엡실론
    return {
        'q01': np.percentile(distances, 1),
        'q05': np.percentile(distances, 5),
        'q10': np.percentile(distances, 10),
    }
```

### 걸음 4: ABC 돌리기

물리치기 표집에서 시작하고 필요하면 MCMC이나 SMC로 옮겨라.

### 걸음 5: 결과 확인하기

뒤확률 예측을 살펴라:

```python
def posterior_predictive_check(simulator, posterior_samples, summary_fn, y_obs):
    """뒤확률이 관측을 되살릴 수 있는지 살피기."""
    s_obs = summary_fn(y_obs)
    
    for theta in posterior_samples:
        x = simulator(theta)
        s_x = summary_fn(x)
        # s_x과 s_obs 견주기
```

---

## 요약

| 개념 | 설명 |
|---------|-------------|
| **가능도 없음** | 흉내 낼 수는 있으나 가능도의 값은 못 매긴다 |
| **ABC의 생각** | 비슷한 자료를 내는 매개변수를 받아들인다 |
| **간추린 통계량** | 견줄 만하도록 차원을 줄인다 |
| **너그러움 $\epsilon$** | 주고받음: 치우침과 흩어짐 |
| **ABC 뒤확률** | 참 뒤확률의 어림 |
| **일관성** | $\epsilon \to 0$, $n \to \infty$이면 정확해진다 |

가능도 없는 추론은 전통적인 방법이 무너지는 복잡한 흉내내기 모형에서도 베이즈 분석을 가능하게 한다. ABC은 단순하고 널리 쓸 수 있는 얼개를 주고, 요즘의 신경망 방식은 추론을 되풀이하는 일감에서 효율을 끌어올린다.

---

## 참고 문헌

1. Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). "Approximate Bayesian Computation in Population Genetics." *Genetics*.
2. Marin, J.-M., Pudlo, P., Robert, C. P., & Ryder, R. J. (2012). "Approximate Bayesian Computational Methods." *Statistics and Computing*.
3. Sisson, S. A., Fan, Y., & Beaumont, M. A. (2018). *Handbook of Approximate Bayesian Computation*. CRC Press.
4. Cranmer, K., Brehmer, J., & Louppe, G. (2020). "The Frontier of Simulation-Based Inference." *PNAS*.
5. Fearnhead, P., & Prangle, D. (2012). "Constructing Summary Statistics for Approximate Bayesian Computation: Semi-Automatic Approximate Bayesian Computation." *JRSS-B*.

## 연습문제

1. **흉내내기 보기.** 단순한 생태 모형(이를테면 로트카-볼테라)을 흉내내기 장치로 구현하여라. 자료는 만들 수 있으나 가능도의 값은 매길 수 없음을 확인하여라.

2. **손으로 하는 ABC.** (가능도를 쓸 수 있는) 정규 평균 추론 문제에 ABC 물리치기 표집을 구현하여라. 여러 $\epsilon$에서 ABC 뒤확률을 참 뒤확률과 견주어라.

3. **간추린 통계량의 영향.** 마음대로 고른 모형에서 (a) 충분 통계량, (b) 충분하지는 않으나 알려 주는 바 있는 통계량, (c) 무작위 통계량을 쓴 ABC을 견주어라. 뒤확률이 어떻게 바뀌는가?

4. **너그러움 맞추기.** 위의 맞추기 절차를 구현하여라. 받아들임 비율이 $\epsilon$에 어떻게 기대는가? 그럭저럭한 고름은 무엇인가?

5. **정확한 것과 견주기.** ABC과 정확한 추론이 모두 가능한 모형에서 ABC 어림의 오차를 $\epsilon$의 함수로 재어라.

---
