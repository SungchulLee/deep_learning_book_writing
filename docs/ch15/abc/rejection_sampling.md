# ABC 물리치기 표집
ABC 물리치기 표집은 가장 단순하고 알아보기 쉬운 가능도 없는 추론 알고리즘이다. 이 마당은 그 알고리즘, 이론적 성질, 실전에서 살필 점, 그리고 더 나아간 ABC 방법이 나오게 된 한계를 보인다.

---

## 알고리즘

### 기본 ABC 물리치기 표집

**입력**: 앞확률 $p(\theta)$, 흉내내기 장치 $p(\mathbf{x}|\theta)$, 관측 자료 $\mathbf{y}$, 간추린 통계량 $S(\cdot)$, 거리 $\rho(\cdot, \cdot)$, 너그러움 $\epsilon$, 표본의 개수 $N$

**날 것**: 어림 뒤확률에서 뽑은 표본 $\{\theta_1, \ldots, \theta_N\}$

```
for i = 1 to N:
    repeat:
        Sample θ ~ p(θ)
        Simulate x ~ p(x | θ)
        Compute s = S(x)
    until ρ(s, S(y)) < ε
    
    Store θᵢ = θ

return {θ₁, ..., θₙ}
```

### 파이썬 구현

```python
import numpy as np

def abc_rejection(prior_sampler, simulator, summary_fn, y_obs, 
                  distance_fn, epsilon, n_samples):
    """
    ABC 물리치기 표집.
    
    인수:
        prior_sampler: 앞확률에서 표본 하나를 돌려주는 함수
        simulator: 자료를 흉내내는 함수 theta -> x
        summary_fn: 간추린 통계량을 셈하는 함수 x -> s
        y_obs: 관측한 자료
        distance_fn: 함수 (s1, s2) -> 거리
        epsilon: 받아들임 너그러움
        n_samples: 바라는 뒤확률 표본의 개수
    
    반환값:
        samples: 받아들인 매개변수 값의 배열
        acceptance_rate: Fraction of proposals accepted
    """
    s_obs = summary_fn(y_obs)
    
    samples = []
    n_attempts = 0
    
    while len(samples) < n_samples:
        # 앞확률에서 표집
        theta = prior_sampler()
        
        # 흉내내기
        x = simulator(theta)
        s_x = summary_fn(x)
        
        # 받아들이거나 물리치기
        n_attempts += 1
        if distance_fn(s_x, s_obs) < epsilon:
            samples.append(theta)
    
    acceptance_rate = n_samples / n_attempts
    return np.array(samples), acceptance_rate
```

---

## 이론적 성질

### 과녁 분포

ABC 물리치기 표집은 ABC 뒤확률을 겨냥한다:

$$
p_\epsilon(\theta | \mathbf{y}) = \frac{p(\theta) \int p(\mathbf{x}|\theta) K_\epsilon(S(\mathbf{x}), S(\mathbf{y})) d\mathbf{x}}{\int p(\theta') \int p(\mathbf{x}|\theta') K_\epsilon(S(\mathbf{x}), S(\mathbf{y})) d\mathbf{x} d\theta'}
$$

딱딱한 문턱값에서는 $K_\epsilon(s, s') = \mathbf{1}[\rho(s, s') < \epsilon]$이다.

### 정확함 결과

**명제**: $S$이 충분 통계량이고 $\epsilon \to 0$이면 다음이 성립한다:

$$
p_\epsilon(\theta | \mathbf{y}) \to p(\theta | \mathbf{y})
$$

ABC 뒤확률이 참 뒤확률로 모인다.

### 받아들임 확률

내놓은 $\theta$을 받아들일 확률은 다음과 같다:

$$
\alpha(\theta) = P(\rho(S(\mathbf{X}), S(\mathbf{y})) < \epsilon | \theta) = \int_{\{s: \rho(s, S(\mathbf{y})) < \epsilon\}} p(s|\theta) ds
$$

전체 받아들임 비율은 다음과 같다:

$$
\bar{\alpha} = \int \alpha(\theta) p(\theta) d\theta
$$

### 중요도 표집과의 관계

ABC 물리치기는 다음과 같은 중요도 표집으로 볼 수 있다:

- 제안: $q(\theta) = p(\theta)$(앞확률)
- 무게: $w(\theta) \propto \alpha(\theta)$

스스로 고르게 하는 중요도 표집 어림자는 다음과 같다:

$$
\mathbb{E}_{p(\theta|\mathbf{y})}[f(\theta)] \approx \frac{\sum_{i=1}^N w_i f(\theta_i)}{\sum_{i=1}^N w_i}
$$

물리치기 표집에서는 받아들인 표본의 무게가 모두 같다.

---

## 받아들임 비율 분석

### 받아들임 비율에 영향을 주는 것

| 요소 | 받아들임 비율에 주는 영향 |
|--------|--------------------------|
| $\epsilon$이 크다 | 받아들임이 높아진다 |
| 간추림의 차원이 낮다 | 받아들임이 높아진다 |
| 알려 주는 바 있는 앞확률 | 받아들임이 높아진다 |
| 복잡한 모형 | 받아들임이 낮아진다 |

### 받아들임 비율과 너그러움

분포가 거의 공 모양인 차원 $d$의 간추린 통계량에서는:

$$
\bar{\alpha} \approx V_d \cdot \epsilon^d \cdot C
$$

여기서 $V_d$은 단위 $d$차원 공의 부피이고 $C$은 모형에 기댄다.

**뜻하는 바**: 받아들임 비율은 간추림의 차원에 따라 지수로 줄어든다.

### 겪어 보고 받아들임 비율 어림하기

```python
def estimate_acceptance_rate(prior_sampler, simulator, summary_fn, 
                             y_obs, distance_fn, epsilon, n_trials=10000):
    """주어진 엡실론의 받아들임 비율 어림하기."""
    s_obs = summary_fn(y_obs)
    
    n_accepted = 0
    for _ in range(n_trials):
        theta = prior_sampler()
        x = simulator(theta)
        s_x = summary_fn(x)
        
        if distance_fn(s_x, s_obs) < epsilon:
            n_accepted += 1
    
    return n_accepted / n_trials
```

### 너그러움 고르기

**어림 규칙**: 받아들임 비율이 0.1%-1%가 되도록 $\epsilon$을 잡아라.

**알아서 맞추는 길**: 큰 $\epsilon$에서 시작해 받아들임 비율이 그럭저럭해질 때까지 줄여라.

```python
def find_epsilon(prior_sampler, simulator, summary_fn, y_obs, 
                 distance_fn, target_rate=0.01, n_pilot=10000):
    """목표 받아들임 비율에 맞는 엡실론 찾기."""
    s_obs = summary_fn(y_obs)
    
    # 예비 실행에서 거리 모으기
    distances = []
    for _ in range(n_pilot):
        theta = prior_sampler()
        x = simulator(theta)
        s_x = summary_fn(x)
        distances.append(distance_fn(s_x, s_obs))
    
    # 분위수로 잡은 엡실론
    epsilon = np.percentile(distances, target_rate * 100)
    return epsilon
```

---

## 간추린 통계량

### 좋은 간추림의 조건

1. **알려 주는 바 있음**: $\theta$에 대한 정보를 담는다
2. **차원이 낮음**: 받아들임 비율을 감당할 만하게 지킨다
3. **셈할 수 있음**: 흉내 낸 자료에서 셈할 수 있다
4. **튼튼함**: 잡음에 지나치게 민감하지 않다

### 흔한 간추린 통계량

**자리/눈금**:

- 평균, 중앙값, 최빈값
- 흩어짐, 표준편차, 사분위 범위
- 분위수

**기댐**:

- 상관, 공분산
- 자기상관(시계열에서)
- 엇상관

**모양**:

- 치우침도, 뾰족함도
- 막대그림 칸의 세기
- 경험 누적분포함수 값

**분야에 따른 것**:

- 집단 유전학: 대립유전자 빈도, 이형접합도, $F_{ST}$
- 역학: 마지막 규모, 정점 시각, 자람 비율
- 시계열: 스펙트럼 밀도, 주기도

### 보기: 정규 모형

$(\mu, \sigma^2)$을 모르는 $y_1, \ldots, y_n \sim \mathcal{N}(\mu, \sigma^2)$에서:

**충분 통계량**: $S(\mathbf{y}) = (\bar{y}, s^2)$이며 여기서 $\bar{y} = \frac{1}{n}\sum y_i$이고 $s^2 = \frac{1}{n-1}\sum(y_i - \bar{y})^2$이다.

이 간추림과 $\epsilon \to 0$을 쓴 ABC은 정확한 뒤확률을 준다.

### 보기: 시계열

AR(1) 과정 $y_t = \phi y_{t-1} + \epsilon_t$에서:

```python
def ar1_summaries(y):
    """AR(1) 모형의 간추린 통계량."""
    return np.array([
        np.mean(y),
        np.var(y),
        np.corrcoef(y[:-1], y[1:])[0, 1],  # 뒤짐 1의 자기상관
    ])
```

---

## 거리 함수

### 유클리드 거리

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \|\mathbf{s}_1 - \mathbf{s}_2\|_2 = \sqrt{\sum_j (s_{1j} - s_{2j})^2}
$$

단순하지만 모든 간추림을 똑같이 다룬다.

### 고르게 한 유클리드

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \sqrt{\sum_j \frac{(s_{1j} - s_{2j})^2}{\hat{\sigma}_j^2}}
$$

여기서 $\hat{\sigma}_j^2$은 앞확률 예측 아래 $j$번째 간추림의 흩어짐이다.

```python
def normalized_euclidean(s1, s2, sigma):
    """고르게 한 유클리드 거리."""
    return np.sqrt(np.sum(((s1 - s2) / sigma)**2))
```

### 마할라노비스 거리

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \sqrt{(\mathbf{s}_1 - \mathbf{s}_2)^T \Sigma^{-1} (\mathbf{s}_1 - \mathbf{s}_2)}
$$

간추림 사이의 상관을 헤아린다.

```python
def mahalanobis_distance(s1, s2, Sigma_inv):
    """마할라노비스 거리."""
    diff = s1 - s2
    return np.sqrt(diff @ Sigma_inv @ diff)
```

### 눈금 행렬 어림하기

```python
def estimate_summary_covariance(prior_sampler, simulator, summary_fn, n_sims=1000):
    """앞확률 미리봄에서 간추린 통계량의 공분산 어림하기."""
    summaries = []
    for _ in range(n_sims):
        theta = prior_sampler()
        x = simulator(theta)
        summaries.append(summary_fn(x))
    
    return np.cov(np.array(summaries).T)
```

---

## 실전 구현

### 온전한 보기: 정규 분포의 추론

```python
import numpy as np
from scipy import stats

# 참 매개변수와 관측한 자료
mu_true, sigma_true = 5.0, 2.0
n_obs = 100
y_obs = np.random.normal(mu_true, sigma_true, n_obs)

# 앞확률
def prior_sampler():
    mu = np.random.uniform(-10, 10)
    sigma = np.random.uniform(0.1, 10)
    return np.array([mu, sigma])

# 흉내내기 장치
def simulator(theta):
    mu, sigma = theta
    return np.random.normal(mu, sigma, n_obs)

# 간추린 통계량(이 모형에 넉넉함)
def summary_fn(x):
    return np.array([np.mean(x), np.std(x, ddof=1)])

# 거리
def distance_fn(s1, s2):
    return np.linalg.norm(s1 - s2)

# 엡실론 눈금 맞추기
s_obs = summary_fn(y_obs)
pilot_distances = []
for _ in range(10000):
    theta = prior_sampler()
    x = simulator(theta)
    pilot_distances.append(distance_fn(summary_fn(x), s_obs))

epsilon = np.percentile(pilot_distances, 1)  # 받아들임 1%
print(f"Epsilon: {epsilon:.4f}")

# ABC 돌리기
samples, acc_rate = abc_rejection(
    prior_sampler, simulator, summary_fn, y_obs,
    distance_fn, epsilon, n_samples=1000
)

print(f"Acceptance rate: {acc_rate:.4f}")
print(f"Posterior mean: mu={samples[:, 0].mean():.2f}, sigma={samples[:, 1].mean():.2f}")
print(f"True values: mu={mu_true}, sigma={sigma_true}")
```

### 병렬로 돌리기

ABC 물리치기는 민망할 만큼 병렬로 잘 돌아간다:

```python
from multiprocessing import Pool

def abc_rejection_parallel(prior_sampler, simulator, summary_fn, y_obs,
                          distance_fn, epsilon, n_samples, n_workers=4):
    """나란한 ABC 물리치기 표집."""
    s_obs = summary_fn(y_obs)
    
    def try_sample(_):
        while True:
            theta = prior_sampler()
            x = simulator(theta)
            if distance_fn(summary_fn(x), s_obs) < epsilon:
                return theta
    
    with Pool(n_workers) as pool:
        samples = pool.map(try_sample, range(n_samples))
    
    return np.array(samples)
```

### 나아감 지켜보기

```python
def abc_rejection_with_progress(prior_sampler, simulator, summary_fn, y_obs,
                                distance_fn, epsilon, n_samples, report_every=100):
    """나아감을 알려 주는 ABC."""
    s_obs = summary_fn(y_obs)
    samples = []
    n_attempts = 0
    
    while len(samples) < n_samples:
        theta = prior_sampler()
        x = simulator(theta)
        n_attempts += 1
        
        if distance_fn(summary_fn(x), s_obs) < epsilon:
            samples.append(theta)
            
            if len(samples) % report_every == 0:
                rate = len(samples) / n_attempts
                print(f"Accepted {len(samples)}/{n_samples}, "
                      f"rate: {rate:.4f}, "
                      f"attempts: {n_attempts}")
    
    return np.array(samples), n_samples / n_attempts
```

---

## 한계

### 낮은 받아들임 비율

복잡한 모형이나 작은 $\epsilon$에서는:

- 받아들임 비율이 $< 10^{-6}$일 수 있다
- 흉내내기가 수백만 번 필요하다
- 셈으로 감당할 수 없다

### 앞확률에 기댐

다음일 때 앞확률에서 표집하는 것이 비효율적이다:

- 앞확률이 퍼져 있다
- 뒤확률이 좁게 몰려 있다
- 앞확률과 뒤확률이 거의 겹치지 않는다

### 간추린 통계량 고르기

결과가 간추림에 결정적으로 기댄다:

- 충분하지 않은 간추림 → 치우친 뒤확률
- 간추림이 너무 많다 → 낮은 받아들임
- 체계적으로 고르는 길이 없다

### 가능도 어림이 없다

ABC 물리치기는 가능도 어림을 주지 않는다. 모형 견줌에 쓸모 있겠지만 여기서는 얻을 수 없다.

---

## ABC 물리치기를 언제 쓰나

### 잘 맞는 경우

✓ 흉내내기 장치가 빠르다(1초 미만)
✓ 매개변수 차원이 낮다(5 미만)
✓ 앞확률이 알려 주는 바가 있다
✓ 좋은 간추린 통계량을 쓸 수 있다
✓ 보통 정확도로 넉넉하다

### 나아간 방법으로 옮겨야 할 때

✗ 받아들임 비율이 너무 낮다(0.01% 미만)
✗ 앞확률이 알려 주는 바가 없다
✗ 뒤확률 표본이 많이 필요하다
✗ 더 높은 정확도가 필요하다
✗ 흉내내기 장치가 비싸다

**다음 걸음**: 더 잘 살펴보려면 ABC-MCMC, 너그러움을 알아서 맞추려면 ABC-SMC.

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **알고리즘** | 앞확률에서 뽑고, 흉내 내고, 가까우면 받아들인다 |
| **과녁** | ABC 뒤확률 $p_\epsilon(\theta \| \mathbf{y})$ |
| **받아들임 비율** | $\epsilon$과 간추림의 차원에 따라 줄어든다 |
| **좋은 점** | 단순하고, 병렬로 돌고, ABC 뒤확률의 정확한 표본을 준다 |
| **한계** | $\epsilon$이 작으면 비효율적이고 앞확률에 기댄다 |
| **핵심 고름** | 간추린 통계량, 거리 함수, 너그러움 |

ABC 물리치기 표집은 가능도 없는 추론의 바탕이다. 단순하지만 더 정교한 방법이 딛고 서는 핵심 생각을 세운다.

---

## 참고 문헌

1. Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A., & Feldman, M. W. (1999). "Population Growth of Human Y Chromosomes: A Study of Y Chromosome Microsatellites." *Molecular Biology and Evolution*.
2. Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). "Approximate Bayesian Computation in Population Genetics." *Genetics*.
3. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
4. Marin, J.-M., et al. (2012). "Approximate Bayesian Computational Methods." *Statistics and Computing*.

## 연습문제

1. **구현.** 푸아송 분포의 비율 매개변수 $\lambda$을 추론하는 ABC 물리치기를 구현하여라. 정확한 뒤확률과 견주어라.

2. **너그러움에 대한 민감함.** 정규 모형 보기에서 앞확률 예측 거리의 0.1%, 1%, 10% 분위수를 $\epsilon$으로 두고 ABC을 돌려라. 나온 뒤확률을 그려 견주어라.

3. **간추린 통계량 견주기.** 정규 매개변수를 추론할 때 (a) 평균과 흩어짐, (b) 중앙값과 사분위 범위, (c) 처음 네 적률을 쓴 ABC을 견주어라. 어느 것이 뒤확률을 가장 잘 어림하는가?

4. **커짐새 실험.** $\epsilon$을 붙박아 두고 간추린 통계량의 차원(간추림 1, 2, 5, 10개)에 따라 받아들임 비율이 어떻게 바뀌는지 재어라.

5. **병렬 잣대.** 일꾼 1, 2, 4, 8개로 차례대로 도는 ABC 물리치기와 병렬로 도는 것의 시간을 견주어라.

---
