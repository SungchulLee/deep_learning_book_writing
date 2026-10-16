# ABC 거절 샘플링

ABC 거절 샘플링은 독립적인 사전·시뮬레이션 쌍 중 요약 거리가 허용오차 안에 드는 파라메터를 남긴다. 고정된 ABC 목표의 직접 표본을 얻는 단순한 기준 방법이다.

## 1. 알고리즘

관측 요약 s_obs를 한 번 계산한 뒤 사전에서 θ를, 시뮬레이터에서 X|θ를 생성한다. ρ(S(X),s_obs)<ε이면 θ를 남기고 원하는 채택 수까지 반복한다.

사전·시뮬레이션이 독립이고 ABC 사건의 예측확률이 양수라면 채택값은 고정 ABC 사후의 독립 표본이다. 원래 사후의 정확 표본이라는 뜻은 아니다. 입력 표본 수·ε·유한 거리와 최대 시도·예산 종료를 확인한다. 현재 while 루프에는 시도 상한이 없어 확률 0인 사건에서 끝나지 않을 수 있다.

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
        samples: 받아들인 파라메터 값의 배열
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

## 2. 이론적 성질

A_ε(θ)=P_θ(ρ(S(X),s_obs)<ε)라 두면 π_ε(θ|y)∝p(θ)A_ε(θ)다. 전체 채택확률 a=E_prior A_ε가 양수일 때 채택 N개에 필요한 기대 시뮬레이션 수는 N/a다.

요약밀도의 정칙성 아래 ε→0은 p(θ|S(y))에 연결되고 S가 충분하면 원자료 사후와 같아질 수 있다. 유한 ε의 원사후 근사 오차는 별도다.

θ~prior에 웨이트 A_ε를 주는 IS 표현도 가능하지만 A_ε 자체는 보통 모른다. 한 번의 시뮬레이션으로 0/1 가중을 생성한 확장 공간의 거절 방법이라고 이해할 수 있다. 고정 시도 수에서 채택이 0이면 정규화 평균을 정의할 수 없다. 채택값의 평균이 모든 함수의 원사후 기대를 불편하게 추정하는 것도 아니다.

## 3. 채택률과 허용오차

요약 차원 d에서 사전예측 요약밀도가 s_obs 근처에 연속이고 양수이면 작은 유클리드 공의 채택확률은 대략 V_d ε^d p_S(s_obs)다. 작은 공 근사와 거리의 단위를 확인해야 하며 모든 요약분포가 등방이라는 가정은 필요하지 않다. 이 차수는 정확한 유한 ε 공식이 아니다.

ε를 늘리면 같은 거리의 사건은 커진다. 하지만 요약을 바꾸거나 사전을 더 집중시켰다는 이유만으로 채택률이 반드시 높아지는 것은 아니다. 관측과의 겹침이 중요하다.

아래 pilot 함수들은 고정 시도 수로 채택률 또는 거리 분위수를 추정한다. target_rate는 0과 1 사이여야 하고 난수 변동·거리의 동률 때문에 새 실행의 실제 비율이 같지는 않다. 0.1~1%는 보편적인 적정 정확도 기준이 아니며 허용오차 민감도와 계산 예산으로 정한다.

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

## 4. 요약통계량의 정보와 차원

유용한 요약은 관심 파라메터의 정보를 보존하면서 비교 차원·비용을 줄인다. 위치·척도에는 평균·중앙값·분산·분위수, 의존에는 상관·자기상관·스펙트럼 피처 등을 사용할 수 있다. 왜도·첨도 같은 고차 요약은 표본 변동에 민감할 수 있다.

독립 정규 모형의 평균·분산은 미지 μ·σ²에 충분한 요약이며 표준편차와의 일대일 변환도 같은 정보를 보존한다. AR(1)의 평균·분산·1차 상관은 편리한 요약이지만 초기분포·길이와 모수 설정에 무관한 일반적인 충분 통계량은 아니다. 짧거나 상수인 시계열에서는 상관이 정의되지 않을 수 있다.

```python
def ar1_summaries(y):
    """AR(1) 모형의 간추린 통계량."""
    return np.array([
        np.mean(y),
        np.var(y),
        np.corrcoef(y[:-1], y[1:])[0, 1],  # 뒤짐 1의 자기상관
    ])
```

## 5. 거리 함수

유클리드 거리는 원래 단위에 민감하다. 표준화 유클리드는 성분별 표준편차로 차이를 나누고, 마할라노비스는 요약 공분산의 역을 사용한다.

아래 normalized_euclidean의 sigma는 분산이 아니라 표준편차이며 각 값이 양수여야 한다. Sigma_inv는 대칭 양의 정부호이고 요약 차원에 맞아야 한다. 사전예측 공분산이 특이하거나 추정 잡음이 크면 정규화·차원 축소를 검토한다. 단순 스케일 선택이 충분성·식별성을 만들어 주지는 않는다.

```python
def normalized_euclidean(s1, s2, sigma):
    """고르게 한 유클리드 거리."""
    return np.sqrt(np.sum(((s1 - s2) / sigma)**2))
```
```python
def mahalanobis_distance(s1, s2, Sigma_inv):
    """마할라노비스 거리."""
    diff = s1 - s2
    return np.sqrt(diff @ Sigma_inv @ diff)
```
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

## 6. 실전 구현

정규 예제는 μ~Uniform(−10,10), σ~Uniform(0.1,10)와 n=100의 생성 모형을 사용한다. 사전은 σ에 균등하며 σ²에 균등한 사전과 다르다. 관측 평균·표본 표준편차를 충분 요약으로 사용해도 pilot에서 정한 양의 ε의 근사는 남는다.

추정치를 생성에 사용한 참값과 비교하는 것은 유한 자료의 통계 오차를 포함한다. 정확한 수치 사후나 독립 기준과 비교해야 ABC 수치 오차를 더 분리할 수 있다.

병렬 조각의 try_sample은 중첩 함수라 프로세스 직렬화에서 실패할 수 있다. 난수 스트림도 작업자별로 독립 관리해야 한다. 느린 시뮬레이션·직렬화·프로세스 시작 비용 때문에 작업자 수만큼 정확히 빨라지는 것은 아니다. progress 조각에도 양의 report_every와 예산 상한 등의 입력 처리가 필요하다.

```python
import numpy as np
from scipy import stats

# 참 파라메터와 관측한 자료
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
```
Epsilon: 0.7549
Acceptance rate: 0.0096
Posterior mean: mu=4.76, sigma=1.84
True values: mu=5.0, sigma=2.0
```
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

## 7. 한계

작은 채택률은 계산비용을 늘리고, 요약 정보 로스·허용오차는 목표를 바꾼다. 넓은 사전의 적은 영역에 사후가 몰리면 prior 기반 제안이 비효율적일 수 있다.

요약을 체계적으로 선택하는 방법이 전혀 없는 것은 아니다. 보조 모형·회귀·학습 기반 요약을 사용할 수 있지만 관심 정보를 보존하는지 평가해야 한다. θ에서 반복한 ABC 지시변수 평균은 A_ε(θ)의 불편 추정이므로 우도와 관련된 양을 전혀 추정할 수 없다는 설명도 부정확하다. 이것이 원자료 우도나 정당한 모형 비교 증거와 같은 것은 아니며 커널·요약의 정규화·충분성을 검토해야 한다.

## 8. 사용 범위와 다른 ABC 방법

고정 ABC 목표의 직접 독립 표본과 단순한 구현이 필요할 때 유용하다. 채택률·시뮬레이션 비용·요약 차원·필요 표본 수로 예산을 계산하고, 고정된 시간 1초·차원 5·비율 1%로 적용 가능성을 단정하지 않는다.

ABC-MCMC와 ABC-SMC는 제안을 개선할 수 있지만 초기화·혼합·중요도 가중 등의 비용과 문제가 추가된다. 같은 ε·요약·거리의 관심량 오차와 전체 시뮬레이션 비용을 비교한다.

## 연습문제

1. 채택확률 a에서 N개를 모으는 기대 시도 수를 구하라. 독립 반복이면 N/a이며 분산·예산 초과 가능성은 별도다.

2. 요약이 충분해도 양의 ε에서 정확 사후인가? 일반적으로 아니다. 요약 정보 로스는 없어도 허용오차 근사가 남는다.

3. 정규 예제의 사전이 σ²에 균등한가? 아니다. σ에 균등하므로 분산 v=σ²로 변환하면 야코비안 1/(2√v)가 들어간다.

4. 거리에서 표준편차로 나누는 이유를 설명하라. 성분별 단위를 맞추기 위한 선택이며 0인 척도와 상관·특이성 문제를 확인해야 한다.

5. 작은 거리 분위수만으로 정확도를 증명할 수 있는가? 채택비용의 경험적 기준일 뿐이다. 여러 ε·요약과 독립 기준의 사후 요약 차이를 확인해야 한다.

## 정리하며

ABC 거절은 고정 ABC 사후의 직접 표본을 얻지만 원사후 오차는 요약·허용오차에 남는다. 채택률과 시뮬레이션 예산, 거리 스케일과 실제 구현의 종료·병렬 조건을 함께 평가한다.

**참고 문헌**

1. Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A., & Feldman, M. W. (1999). "Population Growth of Human Y Chromosomes: A Study of Y Chromosome Microsatellites." *Molecular Biology and Evolution*.
2. Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). "Approximate Bayesian Computation in Population Genetics." *Genetics*.
3. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
4. Marin, J.-M., et al. (2012). "Approximate Bayesian Computational Methods." *Statistics and Computing*.
