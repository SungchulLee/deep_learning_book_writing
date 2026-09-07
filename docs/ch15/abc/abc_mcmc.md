# ABC-MCMC
ABC-MCMC는 어림 베이즈 셈하기와 마르코프 사슬 몬테카를로를 합쳐, 물리치기 표집보다 매개변수 공간을 더 효율적으로 살펴보게 한다. 이 마당은 그 알고리즘과 이론적 뒷받침, 그리고 실전 구현 길잡이를 보인다.

---

## 왜 필요한가

### ABC 물리치기의 한계

ABC 물리치기 표집에는 근본적인 비효율이 있다. 곧 앞확률에서 표집하는데 앞확률이 뒤확률과 거의 겹치지 않을 수 있다. 뒤확률이 앞확률에 견주어 좁게 몰려 있으면:

- 제안 대부분이 버려진다
- 받아들임 비율이 사라질 만큼 작아진다
- 셈 값이 터진다

### MCMC이라는 풀이

앞확률에서 따로따로 표집하는 대신 MCMC로 다음을 한다:

- 앞확률이 아니라 지금 자리에서 제안한다
- 뒤확률이 높은 구역에 표본을 모은다
- 뒤확률이 좁게 몰려 있어도 그럭저럭한 받아들임 비율을 얻는다

---

## ABC-MCMC 알고리즘

### 알고리즘 진술

**입력**: 첫 $\theta_0$, 앞확률 $p(\theta)$, 흉내내기 장치 $p(\mathbf{x}|\theta)$, 관측 자료 $\mathbf{y}$, 간추린 통계량 $S(\cdot)$, 거리 $\rho(\cdot, \cdot)$, 너그러움 $\epsilon$, 제안 $q(\theta'|\theta)$, 되풀이 $T$

**날 것**: ABC 뒤확률을 겨냥하는 사슬 $\{\theta_0, \theta_1, \ldots, \theta_T\}$

```
Simulate x₀ ~ p(x | θ₀)
Compute s₀ = S(x₀)

for t = 1 to T:
    # 내놓기
    θ' ~ q(θ' | θₜ₋₁)
    
    # 흉내내기
    x' ~ p(x | θ')
    s' = S(x')
    
    # ABC-MH 받아들임 확률
    if ρ(s', S(y)) < ε:
        α = min(1, [p(θ') q(θₜ₋₁ | θ')] / [p(θₜ₋₁) q(θ' | θₜ₋₁)])
    else:
        α = 0
    
    # 받아들이거나 물리치기
    if U(0,1) < α:
        θₜ = θ'
        sₜ = s'
    else:
        θₜ = θₜ₋₁
        sₜ = sₜ₋₁

return {θ₀, θ₁, ..., θₜ}
```

### 핵심 통찰

받아들임 확률에는 두 부분이 있다:

1. **ABC 기준**: 흉내 낸 자료가 넉넉히 가까운가?($\rho(s', S(\mathbf{y})) < \epsilon$)
2. **MH 비**: 제안과 앞확률의 균형을 잡는다

ABC 기준을 못 넘기면 바로 물리친다(α = 0).

---

## 이론적 바탕

### 과녁 분포

ABC-MCMC는 다음 결합 분포를 겨냥한다:

$$
\pi_\epsilon(\theta, \mathbf{s}) \propto p(\theta) p(\mathbf{s}|\theta) \mathbf{1}[\rho(\mathbf{s}, S(\mathbf{y})) < \epsilon]
$$

$\theta$에 걸친 주변 분포가 ABC 뒤확률이다:

$$
p_\epsilon(\theta | \mathbf{y}) = \int \pi_\epsilon(\theta, \mathbf{s}) d\mathbf{s}
$$

### 자세한 균형

ABC-MCMC 알맹이는 $\pi_\epsilon(\theta, \mathbf{s})$에 대해 자세한 균형을 만족한다.

**증명 얼개**:

- 제안 $q(\theta'|\theta) p(\mathbf{s}'|\theta')$이 새 $(\theta', \mathbf{s}')$을 내놓는다
- 받아들임 확률은 넓힌 공간에서의 표준 MH이다
- 지시 함수는 $(\theta, \mathbf{s})$과 $(\theta', \mathbf{s}')$에 대해 대칭이다

### 에르고드성

표준 조건(쪼갤 수 없음, 주기 없음) 아래 사슬이 모인다:

$$
\frac{1}{T}\sum_{t=1}^T f(\theta_t) \to \mathbb{E}_{p_\epsilon(\theta|\mathbf{y})}[f(\theta)]
$$

---

## 구현

### 기본 구현

```python
import numpy as np

def abc_mcmc(theta_init, prior_logpdf, simulator, summary_fn, y_obs,
             distance_fn, epsilon, proposal_fn, proposal_logpdf, n_iter):
    """
    ABC-MCMC 알고리즘.
    
    인수:
        theta_init: 첫 매개변수 값
        prior_logpdf: 함수 theta -> log p(theta)
        simulator: 함수 theta -> x
        summary_fn: 함수 x -> 간추린 통계량
        y_obs: 관측한 자료
        distance_fn: 함수 (s1, s2) -> 거리
        epsilon: ABC 너그러움
        proposal_fn: 함수 theta -> theta'(제안에서 표집)
        proposal_logpdf: 함수 (theta', theta) -> log q(theta' | theta)
        n_iter: 되풀이 횟수
    
    반환값:
        chain: 꼴이 (n_iter, dim)인 배열
        acceptance_rate: 전체 받아들임 비율
    """
    s_obs = summary_fn(y_obs)
    dim = len(theta_init)
    
    # 초기화한다
    theta = theta_init.copy()
    x = simulator(theta)
    s = summary_fn(x)
    
    # 첫 점이 쓸 만한지 살피기
    if distance_fn(s, s_obs) >= epsilon:
        raise ValueError("Initial point does not satisfy ABC criterion")
    
    chain = np.zeros((n_iter, dim))
    n_accepted = 0
    
    for t in range(n_iter):
        # 내놓기
        theta_prop = proposal_fn(theta)
        
        # 흉내내기
        x_prop = simulator(theta_prop)
        s_prop = summary_fn(x_prop)
        
        # ABC-MH 받아들임
        if distance_fn(s_prop, s_obs) < epsilon:
            # MH 비 셈하기
            log_alpha = (prior_logpdf(theta_prop) - prior_logpdf(theta) +
                        proposal_logpdf(theta, theta_prop) - 
                        proposal_logpdf(theta_prop, theta))
            
            if np.log(np.random.rand()) < log_alpha:
                theta = theta_prop
                s = s_prop
                n_accepted += 1
        
        chain[t] = theta
    
    return chain, n_accepted / n_iter
```

### 올바른 첫 점 찾기

사슬은 ABC 기준을 만족하는 점에서 시작해야 한다:

```python
def find_initial_point(prior_sampler, simulator, summary_fn, y_obs,
                       distance_fn, epsilon, max_attempts=100000):
    """ABC-MCMC의 쓸 만한 시작점 찾기."""
    s_obs = summary_fn(y_obs)
    
    for _ in range(max_attempts):
        theta = prior_sampler()
        x = simulator(theta)
        s = summary_fn(x)
        
        if distance_fn(s, s_obs) < epsilon:
            return theta
    
    raise RuntimeError(f"Could not find valid initial point in {max_attempts} attempts")
```

### 알아서 맞추는 제안

태우기 동안 제안 공분산을 알아서 맞춘다:

```python
class AdaptiveABCMCMC:
    def __init__(self, dim, target_rate=0.234):
        self.dim = dim
        self.target_rate = target_rate
        self.cov = np.eye(dim)
        self.scale = 2.4**2 / dim
        self.mean = np.zeros(dim)
        self.n = 0
        
    def propose(self, theta):
        return theta + np.random.multivariate_normal(
            np.zeros(self.dim), self.scale * self.cov
        )
    
    def adapt(self, theta, accepted):
        """사슬의 지난 내력에 따라 제안 새로 고치기."""
        self.n += 1
        
        # 달리는 평균과 공분산 새로 고치기
        delta = theta - self.mean
        self.mean += delta / self.n
        
        if self.n > 1:
            self.cov = ((self.n - 2) / (self.n - 1) * self.cov + 
                       delta.reshape(-1, 1) @ delta.reshape(1, -1) / self.n)
        
        # 받아들임에 따라 규모 맞추기
        if self.n > 100:
            if accepted:
                self.scale *= 1.01
            else:
                self.scale *= 0.99
```

---

## ABC 물리치기와의 견줌

### 효율 견주기

| 결 | ABC 물리치기 | ABC-MCMC |
|--------|---------------|----------|
| 제안이 나오는 곳 | 앞확률 | 지금 자리 둘레 |
| 독립성 | 표본이 서로 독립이다 | 표본이 서로 얽혀 있다 |
| 받아들임 비율 | 아주 낮을 수 있다 | 보통 더 높다 |
| 살펴보기 | 온전하다(앞확률에서) | 그 자리 둘레(섞임이 좋아야 한다) |
| 병렬로 돌리기 | 시시하다 | 더 복잡하다 |
| 태우기 | 필요 없다 | 필요하다 |

### ABC-MCMC가 도움이 될 때

다음일 때 ABC-MCMC가 낫다:

- 앞확률이 뒤확률보다 훨씬 넓을 때
- ABC 물리치기의 받아들임 비율이 0.1%보다 낮을 때
- 뒤확률 표본이 많이 필요할 때

### ABC 물리치기가 나을 수 있을 때

다음일 때 ABC 물리치기가 나을 수 있다:

- 받아들임 비율이 그럭저럭할 때(> 1%)
- 뒤확률의 봉우리가 여럿일 때(MCMC가 갇힐 수 있다)
- 병렬로 쓸 밑천이 넉넉할 때
- 표본이 서로 독립인 것이 중요할 때

---

## 실용적인 고려

### 제안 분포

**무작위 걸음**:

$$
q(\theta'|\theta) = \mathcal{N}(\theta, \Sigma)
$$

- 구현이 단순하다
- $\Sigma$을 맞춰야 한다
- 차원에 맞춰 눈금을 잡는다: $\Sigma = (2.4^2/d) \hat{\Sigma}$

**독립 제안**(ABC-MCMC에는 권하지 않는다):

- ABC 물리치기로 물러선다
- MCMC의 이점을 잃는다

### 제안 맞추기

**목표 받아들임 비율**: ABC-MCMC에서는 10-30%이다(ABC 물리침 때문에 표준 MH보다 낮다).

```python
def tune_proposal_scale(abc_mcmc_fn, theta_init, target_rate=0.2, 
                        n_tune=1000, n_test=500):
    """목표 받아들임 비율에 맞게 제안의 규모 맞추기."""
    scale = 1.0
    
    for _ in range(n_tune):
        _, acc_rate = abc_mcmc_fn(theta_init, scale, n_test)
        
        if acc_rate > target_rate + 0.05:
            scale *= 1.2
        elif acc_rate < target_rate - 0.05:
            scale *= 0.8
        else:
            break
    
    return scale
```

### 받아들임이 낮을 때 다루기

받아들임이 너무 낮으면:

1. $\epsilon$을 키운다
2. 제안의 흩어짐을 줄인다
3. 더 좋은 간추린 통계량을 쓴다
4. 대신 ABC-SMC를 생각해 본다

### 진단

표준 MCMC 진단을 쓴다:

```python
def abc_mcmc_diagnostics(chain):
    """ABC-MCMC 사슬의 진단 셈하기."""
    from statsmodels.tsa.stattools import acf
    
    n_samples, dim = chain.shape
    
    diagnostics = {}
    
    # 실효 표본 크기(차원마다)
    ess = []
    for d in range(dim):
        autocorr = acf(chain[:, d], nlags=100, fft=True)
        tau = 1 + 2 * np.sum(autocorr[1:])
        ess.append(n_samples / tau)
    diagnostics['ess'] = np.array(ess)
    
    # 자취 그림은 눈으로 살펴야 한다
    diagnostics['mean'] = np.mean(chain, axis=0)
    diagnostics['std'] = np.std(chain, axis=0)
    
    return diagnostics
```

---

## 변형

### 흉내내기를 여러 번 하는 ABC-MCMC

제안마다 자료 묶음을 여러 개 흉내 내어 흩어짐을 줄인다:

```python
def abc_mcmc_multiple_sims(theta, simulator, summary_fn, s_obs, 
                           distance_fn, epsilon, n_sims=10):
    """흉내내기를 여러 번 해서 정하는 ABC 받아들임."""
    n_close = 0
    for _ in range(n_sims):
        x = simulator(theta)
        s = summary_fn(x)
        if distance_fn(s, s_obs) < epsilon:
            n_close += 1
    
    return n_close / n_sims  # 어림한 받아들임 확률
```

### 시끄러운 ABC-MCMC

섞임을 낫게 하려고 받아들임 기준에 잡음을 더한다:

$$
\alpha = \min\left(1, \frac{p(\theta') K_\epsilon(s', s_{obs})}{p(\theta) K_\epsilon(s, s_{obs})} \cdot \frac{q(\theta|\theta')}{q(\theta'|\theta)}\right)
$$

여기서 $K_\epsilon$은 딱딱한 문턱값이 아니라 매끄러운 알맹이이다.

### 깁스-ABC

매개변수 덩이가 여럿인 모형에서는 덩이마다 따로 새로 고친다:

```python
def gibbs_abc(theta, blocks, simulators, summary_fns, s_obs, 
              distance_fns, epsilons, proposals):
    """매개변수 덩어리를 새로 고치는 깁스 방식 ABC-MCMC."""
    for i, block in enumerate(blocks):
        # 덩어리 i의 새 값 내놓기
        theta_prop = theta.copy()
        theta_prop[block] = proposals[i](theta[block])
        
        # 흉내내고 ABC 잣대 살피기
        x_prop = simulators[i](theta_prop)
        s_prop = summary_fns[i](x_prop)
        
        if distance_fns[i](s_prop, s_obs[i]) < epsilons[i]:
            # 이 덩어리의 MH 받아들임
            # ...
            theta = theta_prop
    
    return theta
```

---

## 보기: 확률 변동성 모형의 추론

```python
import numpy as np

# 확률 변동성 모형:
# y_t = exp(h_t/2) * eps_t,  eps_t ~ N(0,1)
# h_t = mu + phi*(h_{t-1} - mu) + sigma*eta_t,  eta_t ~ N(0,1)

def sv_simulator(theta, T=500):
    """확률 변동성 모형 흉내내기."""
    mu, phi, sigma = theta
    
    # 초기화한다
    h = np.zeros(T)
    h[0] = mu + sigma * np.random.randn() / np.sqrt(1 - phi**2)
    
    # 로그 변동성 흉내내기
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sigma * np.random.randn()
    
    # 수익 흉내내기
    y = np.exp(h / 2) * np.random.randn(T)
    
    return y

def sv_summaries(y):
    """확률 변동성 모형의 간추린 통계량."""
    log_y2 = np.log(y**2 + 1e-8)
    
    return np.array([
        np.mean(log_y2),
        np.std(log_y2),
        np.corrcoef(log_y2[:-1], log_y2[1:])[0, 1],
        np.corrcoef(log_y2[:-2], log_y2[2:])[0, 1],
        np.mean(np.abs(y)),
        np.std(np.abs(y)),
    ])

# 앞확률
def sv_prior_sample():
    mu = np.random.uniform(-2, 2)
    phi = np.random.uniform(0.8, 0.999)
    sigma = np.random.uniform(0.01, 0.5)
    return np.array([mu, phi, sigma])

def sv_prior_logpdf(theta):
    mu, phi, sigma = theta
    if not (-2 < mu < 2 and 0.8 < phi < 0.999 and 0.01 < sigma < 0.5):
        return -np.inf
    return 0  # 고른 앞확률

# ABC-MCMC 돌리기
y_obs = sv_simulator([0.0, 0.95, 0.2])  # 참 매개변수
s_obs = sv_summaries(y_obs)

# 첫 점을 찾고 사슬 돌리기...
```

---

## 요약

| 항목 | 설명 |
|--------|-------------|
| **알고리즘** | ABC 받아들임 기준을 쓴 MH |
| **과녁** | ABC 뒤확률 $p_\epsilon(\theta \| \mathbf{y})$ |
| **이점** | 앞확률이 퍼져 있을 때 물리치기보다 효율적이다 |
| **어려움** | 올바른 시작점과 꼼꼼한 맞추기가 필요하다 |
| **받아들임** | 두 단계: ABC 기준 뒤에 MH 비 |
| **진단** | 표준 MCMC 진단을 쓴다 |

ABC-MCMC는 단순한 물리치기 표집과 더 정교한 방법 사이의 틈을 이어, ABC 얼개의 단순함을 지키면서 효율을 끌어올린다.

---

## 참고 문헌

1. Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). "Markov Chain Monte Carlo Without Likelihoods." *PNAS*.
2. Sisson, S. A., & Fan, Y. (2011). "Likelihood-Free Markov Chain Monte Carlo." In *Handbook of Markov Chain Monte Carlo*.
3. Wegmann, D., Leuenberger, C., & Excoffier, L. (2009). "Efficient Approximate Bayesian Computation Coupled with Markov Chain Monte Carlo Without Likelihood." *Genetics*.
4. Bortot, P., Coles, S. G., & Sisson, S. A. (2007). "Inference for Stereological Extremes." *JASA*.

## 연습문제

1. **구현.** 정규 모형에 ABC-MCMC를 구현하여라. 흉내내기 한 번마다의 ESS으로 ABC 물리치기와 효율을 견주어라.

2. **알아서 맞추기.** 제안 공분산을 배우는 알아서 맞추는 ABC-MCMC를 구현하여라. 섞임이 나아짐을 보여라.

3. **첫걸음에 대한 민감함.** 첫걸음을 어떻게 잡느냐가 태우기 길이와 마지막 결과에 어떤 영향을 주는지 살펴라.

4. **제안 맞추기.** 문제를 붙박아 두고 가장 좋은 제안 눈금을 겪어 보고 찾아라. 표준 MH 길잡이와 견주면 어떠한가?

5. **봉우리 여럿.** 쌍봉 뒤확률을 만들어 ABC-MCMC가 갇힐 수 있음을 보여라. 풀이를 내놓아라.

---
