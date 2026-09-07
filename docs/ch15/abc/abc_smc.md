# ABC-SMC(잇단 몬테카를로 ABC)
ABC-SMC은 어림 베이즈 셈하기와 잇단 몬테카를로 방법을 합쳐, 너그러움을 알아서 고르고 복잡한 뒤확률에서 효율적으로 표집하게 한다. 이 마당은 그 알고리즘과 여러 판, 그리고 실전 구현 길잡이를 보인다.

---

## 왜 필요한가

### 앞선 방법의 한계

**ABC 물리치기**:

- 앞확률에서 표집한다(비효율적이다)
- 너그러움이 붙박이다(고르기 어렵다)
- 알아서 맞추지 않는다

**ABC-MCMC**:

- 뒤확률이 좁게 몰려 있으면 물리치기보다 낫다
- 그러나 국소 봉우리에 갇힐 수 있다
- 여전히 $\epsilon$을 미리 골라야 한다

### SMC이라는 풀이

ABC-SMC은 다음으로 이 문제를 다룬다:

- 알갱이를 앞확률에서 뒤확률로 되풀이해 옮긴다
- 너그러움을 큰 값(쉬움)에서 작은 값(정확)으로 알아서 맞춘다
- 다시 표집하기와 돌연변이로 여러 갈래를 지킨다
- 자연스럽게 병렬로 돌릴 수 있게 한다

---

## ABC-SMC 알고리즘

### 훑어보기

ABC-SMC은 다음을 겨냥하는 알갱이 무리의 늘어놓음을 만든다:

$$
\pi_1(\theta) \to \pi_2(\theta) \to \cdots \to \pi_T(\theta) \approx p(\theta | \mathbf{y})
$$

여기서 $\pi_t$마다 너그러움이 $\epsilon_1 > \epsilon_2 > \cdots > \epsilon_T$으로 줄어드는 ABC 뒤확률이다.

### 알고리즘 진술

**입력**: 앞확률 $p(\theta)$, 흉내내기 장치 $p(\mathbf{x}|\theta)$, 관측 자료 $\mathbf{y}$, 간추린 통계량 $S(\cdot)$, 거리 $\rho$, 너그러움 일정 $\{\epsilon_1, \ldots, \epsilon_T\}$, 알갱이의 개수 $N$

**날 것**: 뒤확률을 어림하는 무게 준 알갱이 $\{(\theta_i^{(T)}, w_i^{(T)})\}_{i=1}^N$

```
# 세대 t = 1: 앞확률에서 표집
for i = 1 to N:
    repeat:
        Sample θᵢ⁽¹⁾ ~ p(θ)
        Simulate x ~ p(x | θᵢ⁽¹⁾)
    until ρ(S(x), S(y)) < ε₁
    
    Set wᵢ⁽¹⁾ = 1/N

# 세대 t = 2, ..., T
for t = 2 to T:
    # 흔들기를 위해 앞 세대의 흩어짐 셈하기
    Σₜ = 2 × Cov({θᵢ⁽ᵗ⁻¹⁾})
    
    for i = 1 to N:
        repeat:
            # 앞 세대에서 표집
            Sample θ* from {θⱼ⁽ᵗ⁻¹⁾} with probabilities {wⱼ⁽ᵗ⁻¹⁾}
            
            # 흔들기
            Sample θᵢ⁽ᵗ⁾ ~ K(θ | θ*) = N(θ*, Σₜ)
            
            # 흉내내기
            Simulate x ~ p(x | θᵢ⁽ᵗ⁾)
        until ρ(S(x), S(y)) < εₜ
        
        # 무게 셈하기
        wᵢ⁽ᵗ⁾ ∝ p(θᵢ⁽ᵗ⁾) / Σⱼ wⱼ⁽ᵗ⁻¹⁾ K(θᵢ⁽ᵗ⁾ | θⱼ⁽ᵗ⁻¹⁾)
    
    # 무게 고르게 하기
    Normalize {wᵢ⁽ᵗ⁾} to sum to 1

return {(θᵢ⁽ᵀ⁾, wᵢ⁽ᵀ⁾)}
```

---

## 구현

### 온전한 파이썬 구현

```python
import numpy as np
from scipy.stats import multivariate_normal

class ABCSMC:
    def __init__(self, prior_sampler, prior_logpdf, simulator, 
                 summary_fn, distance_fn, y_obs, n_particles=1000):
        """
        ABC-SMC 표집기.
        
        인수:
            prior_sampler: 앞확률에서 표본을 돌려주는 함수
            prior_logpdf: 함수 theta -> log p(theta)
            simulator: 함수 theta -> 흉내낸 자료
            summary_fn: 함수 x -> 간추린 통계량
            distance_fn: 함수 (s1, s2) -> 거리
            y_obs: 관측한 자료
            n_particles: 알갱이의 개수
        """
        self.prior_sampler = prior_sampler
        self.prior_logpdf = prior_logpdf
        self.simulator = simulator
        self.summary_fn = summary_fn
        self.distance_fn = distance_fn
        self.s_obs = summary_fn(y_obs)
        self.n_particles = n_particles
        
    def sample_initial_population(self, epsilon):
        """ABC으로 앞확률에서 첫 무리 표집하기."""
        particles = []
        distances = []
        
        while len(particles) < self.n_particles:
            theta = self.prior_sampler()
            x = self.simulator(theta)
            s = self.summary_fn(x)
            d = self.distance_fn(s, self.s_obs)
            
            if d < epsilon:
                particles.append(theta)
                distances.append(d)
        
        particles = np.array(particles)
        weights = np.ones(self.n_particles) / self.n_particles
        
        return particles, weights, distances
    
    def sample_next_population(self, prev_particles, prev_weights, epsilon):
        """중요도 표집으로 다음 무리 표집하기."""
        dim = prev_particles.shape[1]
        
        # 흔들기 공분산 셈하기
        cov = 2 * np.cov(prev_particles.T, aweights=prev_weights)
        if dim == 1:
            cov = np.array([[cov]])
        
        particles = []
        distances = []
        weights = []
        
        for i in range(self.n_particles):
            # 다시 표집하고 움직이기
            accepted = False
            while not accepted:
                # 앞 무리에서 표집
                idx = np.random.choice(
                    self.n_particles, p=prev_weights
                )
                theta_star = prev_particles[idx]
                
                # 흔들기
                theta = np.random.multivariate_normal(theta_star, cov)
                
                # 앞확률의 받침 살피기
                if self.prior_logpdf(theta) == -np.inf:
                    continue
                
                # 흉내내고 ABC 살피기
                x = self.simulator(theta)
                s = self.summary_fn(x)
                d = self.distance_fn(s, self.s_obs)
                
                if d < epsilon:
                    accepted = True
            
            particles.append(theta)
            distances.append(d)
            
            # 무게 셈하기
            kernel_sum = sum(
                prev_weights[j] * multivariate_normal.pdf(
                    theta, prev_particles[j], cov
                )
                for j in range(self.n_particles)
            )
            w = np.exp(self.prior_logpdf(theta)) / kernel_sum
            weights.append(w)
        
        particles = np.array(particles)
        weights = np.array(weights)
        weights /= weights.sum()  # 정규화
        
        return particles, weights, distances
    
    def run(self, epsilon_schedule, verbose=True):
        """
        주어진 너그러움 일정으로 ABC-SMC 돌리기.
        
        인수:
            epsilon_schedule: 줄어드는 너그러움의 목록
            verbose: 진행 상황 출력 여부
        
        반환값:
            particles: 마지막 알갱이 자리
            weights: 마지막 알갱이 무게
            history: 세대마다의 (알갱이, 무게) 목록
        """
        history = []
        
        # 첫 무리
        if verbose:
            print(f"Generation 1, ε={epsilon_schedule[0]:.4f}")
        
        particles, weights, distances = self.sample_initial_population(
            epsilon_schedule[0]
        )
        history.append((particles.copy(), weights.copy()))
        
        if verbose:
            print(f"  Accepted {self.n_particles} particles")
        
        # 뒤이은 무리
        for t, epsilon in enumerate(epsilon_schedule[1:], start=2):
            if verbose:
                print(f"Generation {t}, ε={epsilon:.4f}")
            
            particles, weights, distances = self.sample_next_population(
                particles, weights, epsilon
            )
            history.append((particles.copy(), weights.copy()))
            
            # 실효 표본 크기 살피기
            ess = 1 / np.sum(weights**2)
            if verbose:
                print(f"  ESS: {ess:.1f}")
        
        return particles, weights, history
```

### 너그러움 알아서 고르기

너그러움을 미리 못 박는 대신 알갱이의 거리에 기대어 알아서 맞춘다:

```python
def run_adaptive(self, epsilon_init, epsilon_final, alpha=0.5, 
                 max_generations=20, verbose=True):
    """
    맞춰 가는 너그러움으로 ABC-SMC 돌리기.
    
    인수:
        epsilon_init: 첫 너그러움(None이면 저절로 정함)
        epsilon_final: 목표로 하는 마지막 너그러움
        alpha: Quantile for tolerance selection (e.g., 0.5 = median)
        max_generations: 최대 세대 수
    """
    history = []
    
    # 첫 무리
    if epsilon_init is None:
        # 앞확률에서 50%쯤 받아들이도록 첫 엡실론 잡기
        epsilon_init = self.calibrate_initial_epsilon()
    
    particles, weights, distances = self.sample_initial_population(epsilon_init)
    history.append((particles.copy(), weights.copy()))
    
    epsilon = epsilon_init
    
    for t in range(2, max_generations + 1):
        # 맞춰 가는 엡실론: 지금 거리의 분위수
        epsilon_new = np.percentile(distances, alpha * 100)
        epsilon_new = max(epsilon_new, epsilon_final)
        
        if verbose:
            print(f"Generation {t}, ε: {epsilon:.4f} -> {epsilon_new:.4f}")
        
        if epsilon_new >= epsilon * 0.99:  # 나아감 없음
            if verbose:
                print("  Tolerance not decreasing, stopping")
            break
        
        epsilon = epsilon_new
        particles, weights, distances = self.sample_next_population(
            particles, weights, epsilon
        )
        history.append((particles.copy(), weights.copy()))
        
        ess = 1 / np.sum(weights**2)
        if verbose:
            print(f"  ESS: {ess:.1f}")
        
        if epsilon <= epsilon_final:
            if verbose:
                print("  Reached target tolerance")
            break
    
    return particles, weights, history
```

---

## 이론적 성질

### 과녁 분포

세대 $t$에서 ABC-SMC은 다음을 겨냥한다:

$$
\pi_t(\theta) \propto p(\theta) \cdot P(\rho(S(\mathbf{X}), S(\mathbf{y})) < \epsilon_t | \theta)
$$

$\epsilon_t \to 0$이면:

$$
\pi_t(\theta) \to p(\theta | \mathbf{y})
$$

### 무게 이끌어 내기

세대 $t$의 중요도 무게는 다음과 같다:

$$
w_i^{(t)} \propto \frac{\pi_t(\theta_i^{(t)})}{\sum_{j=1}^N w_j^{(t-1)} K_t(\theta_i^{(t)} | \theta_j^{(t-1)})}
$$

$\alpha_t$이 ABC 받아들임 확률일 때 $\pi_t(\theta) \propto p(\theta) \cdot \alpha_t(\theta)$이므로:

$$
w_i^{(t)} \propto \frac{p(\theta_i^{(t)})}{\sum_{j=1}^N w_j^{(t-1)} K_t(\theta_i^{(t)} | \theta_j^{(t-1)})}
$$

(ABC 받아들임은 표집 안에 속뜻으로 들어 있다.)

### 실효 표본 크기

무게 주저앉음을 알아내려고 ESS을 지켜본다:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^N (w_i^{(t)})^2}
$$

ESS이 너무 낮아지면 다시 표집해 무게를 고르게 되돌린다.

---

## 흔들기 알맹이

### 가우스 알맹이(표준)

$$
K(\theta' | \theta) = \mathcal{N}(\theta' | \theta, \Sigma_t)
$$

**공분산 고르기**:

- $\Sigma_t = 2 \times \text{Cov}(\{\theta_i^{(t-1)}\})$(무게 실은 표본 공분산의 두 배)
- 가장 좋은 선형 오그라들기 어림자
- 성분마다: $\Sigma_t = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$

### 가장 좋은 흔들기

가장 좋은 흔들기 알맹이는 기대 흉내내기 횟수를 가장 작게 한다. 가우스 과녁에서는:

$$
K^*(\theta' | \theta) = \mathcal{N}\left(\theta' \big| \theta, \frac{4}{(d+2)} \text{Cov}(\pi_t)\right)
$$

### 국소 선형 회귀 알맹이

알맹이를 그 자리의 기하에 맞춘다:

```python
def local_kernel(theta, particles, weights, k=50):
    """가장 가까운 이웃에 바탕을 둔 그 자리 공분산."""
    distances = np.linalg.norm(particles - theta, axis=1)
    nearest = np.argsort(distances)[:k]
    
    local_particles = particles[nearest]
    local_weights = weights[nearest]
    local_weights /= local_weights.sum()
    
    return 2 * np.cov(local_particles.T, aweights=local_weights)
```

---

## 다시 표집하기 전략

### 언제 다시 표집하나

ESS이 문턱값(이를테면 $N/2$) 아래로 떨어지면 다시 표집한다:

```python
def maybe_resample(particles, weights, threshold_ratio=0.5):
    """ESS이 문턱값 아래이면 다시 표집하기."""
    ess = 1 / np.sum(weights**2)
    
    if ess < threshold_ratio * len(weights):
        # 차근차근 다시 표집
        indices = systematic_resample(weights)
        particles = particles[indices]
        weights = np.ones(len(weights)) / len(weights)
    
    return particles, weights

def systematic_resample(weights):
    """차근차근 다시 표집."""
    n = len(weights)
    positions = (np.arange(n) + np.random.rand()) / n
    
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    
    return indices
```

### 다시 표집하기 방식

| 방식 | 흩어짐 | 복잡도 |
|--------|----------|------------|
| 다항 | 더 크다 | O(N) |
| 체계적 | 더 작다 | O(N) |
| 잔차 | 더 작다 | O(N) |
| 층 나눔 | 더 작다 | O(N) |

흩어짐이 작아 보통 체계적 다시 표집하기를 더 낫게 여긴다.

---

## 실용적인 고려

### 알갱이의 개수 고르기

**어림 규칙**: $N = 1000$에서 $10000$

**주고받음**:

- 알갱이가 많다 → 어림이 낫고 셈이 늘어난다
- 알갱이가 적다 → 빠르지만 흩어짐이 크다

### 너그러움 일정

**붙박이 일정**:
```python
epsilon_schedule = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
```

**알아서 맞추기**(권함):

- $\epsilon_t$을 이제 거리의 분위수(예: 중앙값)로 둔다
- 문제의 어려움에 저절로 맞춘다

### 멈추는 기준

1. **과녁 봐주는 값에 닿음**: $\epsilon_t \leq \epsilon_{target}$
2. **봐주는 값이 멈춤**: $\epsilon_t \approx \epsilon_{t-1}$
3. **최대 세대**: 안전 한도
4. **셈 예산**: 최대 흉내내기 횟수

### 병렬로 돌리기

ABC-SMC은 세대 안에서 자연스럽게 병렬로 돈다:

```python
from multiprocessing import Pool

def sample_particle_parallel(args):
    """알갱이 하나 표집하기(나란히 돌리기용)."""
    prev_particles, prev_weights, cov, epsilon, simulator, summary_fn, \
        distance_fn, s_obs, prior_logpdf = args
    
    while True:
        # 표집하고 흔들기
        idx = np.random.choice(len(prev_weights), p=prev_weights)
        theta = np.random.multivariate_normal(prev_particles[idx], cov)
        
        if prior_logpdf(theta) == -np.inf:
            continue
        
        x = simulator(theta)
        s = summary_fn(x)
        d = distance_fn(s, s_obs)
        
        if d < epsilon:
            return theta, d

def sample_population_parallel(prev_particles, prev_weights, epsilon, 
                               n_particles, n_workers=4, **kwargs):
    """무리를 나란히 표집하기."""
    cov = 2 * np.cov(prev_particles.T, aweights=prev_weights)
    
    args = [(prev_particles, prev_weights, cov, epsilon, 
             kwargs['simulator'], kwargs['summary_fn'],
             kwargs['distance_fn'], kwargs['s_obs'], 
             kwargs['prior_logpdf'])] * n_particles
    
    with Pool(n_workers) as pool:
        results = pool.map(sample_particle_parallel, args)
    
    particles = np.array([r[0] for r in results])
    distances = [r[1] for r in results]
    
    return particles, distances
```

---

## 변형

### 회귀 조정을 곁들인 ABC-SMC

알갱이를 회귀 조정으로 뒤처리한다:

```python
def regression_adjustment(particles, summaries, s_obs):
    """그 자리 선형 회귀로 알갱이 다듬기."""
    from sklearn.linear_model import LinearRegression
    
    adjusted = []
    for i, (theta, s) in enumerate(zip(particles, summaries)):
        # 그 자리 회귀
        reg = LinearRegression()
        reg.fit(summaries, particles)
        
        # 다듬기
        theta_adj = theta - reg.predict([s])[0] + reg.predict([s_obs])[0]
        adjusted.append(theta_adj)
    
    return np.array(adjusted)
```

### 여러 거리를 쓰는 ABC-SMC

간추림마다 다른 거리를 쓴다:

```python
def multi_distance(s1, s2, weights):
    """성분별 거리를 무게 두어 합친 것."""
    return np.sum(weights * np.abs(s1 - s2))
```

### 모집단 몬테카를로 ABC

일정이 필요 없는 판으로, 모든 것을 굴러가는 도중에 알아서 맞춘다.

---

## 보기: 생태 모형

```python
# 로트카-볼테라 잡이-먹이 모형
def lotka_volterra(theta, T=100, dt=0.1):
    """로트카-볼테라 모형 흉내내기."""
    alpha, beta, gamma, delta = theta
    
    prey = [50.0]
    predator = [20.0]
    
    for _ in range(int(T / dt)):
        x, y = prey[-1], predator[-1]
        
        dx = (alpha * x - beta * x * y) * dt + np.sqrt(max(x, 0)) * np.random.randn() * 0.1
        dy = (delta * x * y - gamma * y) * dt + np.sqrt(max(y, 0)) * np.random.randn() * 0.1
        
        prey.append(max(0, x + dx))
        predator.append(max(0, y + dy))
    
    return np.array([prey[::10], predator[::10]])  # 골라 뽑기

def lv_summaries(x):
    """로트카-볼테라 모형의 간추린 통계량."""
    prey, predator = x
    return np.array([
        np.mean(prey), np.std(prey),
        np.mean(predator), np.std(predator),
        np.corrcoef(prey, predator)[0, 1],
    ])

# 앞확률
def lv_prior_sample():
    return np.array([
        np.random.uniform(0.5, 1.5),   # alpha
        np.random.uniform(0.01, 0.1),  # beta
        np.random.uniform(0.5, 1.5),   # gamma
        np.random.uniform(0.01, 0.1),  # delta
    ])

# ABC-SMC 돌리기
sampler = ABCSMC(
    prior_sampler=lv_prior_sample,
    prior_logpdf=lambda t: 0 if all(t > 0) else -np.inf,
    simulator=lotka_volterra,
    summary_fn=lv_summaries,
    distance_fn=lambda s1, s2: np.linalg.norm(s1 - s2),
    y_obs=lotka_volterra([1.0, 0.05, 1.0, 0.05]),  # "참" 자료
    n_particles=1000
)

particles, weights, history = sampler.run_adaptive(
    epsilon_init=10.0, 
    epsilon_final=1.0
)
```

---

## 비교

| 방법 | 효율 | 알아서 맞추기 | 병렬성 | 복잡도 |
|--------|------------|------------|-------------|------------|
| ABC 물리치기 | 낮음 | 없음 | 높음 | 단순 |
| ABC-MCMC | 보통 | 없음 | 낮음 | 보통 |
| ABC-SMC | 높음 | 높음 | 높음 | 복잡 |

복잡한 문제에서는 대체로 ABC-SMC이 가장 효율적이지만 구현에 품이 더 든다.

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **알고리즘** | ABC을 쓴 잇단 중요도 표집 |
| **알아서 맞추기** | 세대를 거치며 너그러움이 줄어든다 |
| **효율** | 물리치기/MCMC보다 높다 |
| **병렬로 돌리기** | 세대 안에서 자연스럽다 |
| **날 것** | 뒤확률을 어림하는 무게 준 알갱이 |
| **핵심 맞추기** | 알갱이의 개수, 너그러움 일정/알아서 맞추기 |

ABC-SMC은 고전 ABC 방법 가운데 가장 앞선 것으로, 흉내내기 기반 모형에 효율적이고 알아서 맞추는 추론을 준다.

---

## 참고 문헌

1. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
2. Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., & Robert, C. P. (2009). "Adaptive Approximate Bayesian Computation." *Biometrika*.
3. Del Moral, P., Doucet, A., & Jasra, A. (2012). "An Adaptive Sequential Monte Carlo Method for Approximate Bayesian Computation." *Statistics and Computing*.
4. Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. (2009). "Approximate Bayesian Computation Scheme for Parameter Inference and Model Selection in Dynamical Systems." *JRSS-B*.

## 연습문제

1. **구현.** 정규 모형에 ABC-SMC을 구현하여라. 물리치기 및 MCMC과 효율을 견주어라.

2. **알아서 맞추기와 붙박이 일정.** 너그러움을 알아서 고르는 것과 붙박이 등비 일정을 견주어라. 어느 쪽이 더 효율적인가?

3. **알갱이 개수에 대한 민감함.** 알갱이가 $N = 100, 500, 1000, 5000$개일 때 결과가 어떻게 바뀌는지 살펴라.

4. **알맹이 견주기.** 공분산 어림자를 달리한 가우스 알맹이를 견주어라.

5. **생태 추론.** 로트카-볼테라 모형으로 실제 생태 자료 묶음에 ABC-SMC을 써라.

---
