# 실전에서 방법 고르기
주어진 문제에 알맞은 MCMC 방법을 고르는 일은 반은 과학이고 반은 기예이다. 이 마당은 문제의 성격, 셈의 제약, 진단 결과에 기대어 방법을 고르는 실전 길잡이를 준다.

---

## 판단의 틀

### 핵심 물음

방법을 고르기에 앞서 다음 물음에 답하여라:

1. **과녁을 미분할 수 있는가?**
   - 그렇다 → 기울기 기반 방법(MALA, HMC)을 쓸 수 있다
   - 아니다 → 기울기 없는 방법(MH, 깁스)을 써야 한다

2. **차원이 얼마인가?**
   - $d < 20$ → 대부분의 방법이 잘 굴러간다
   - $20 < d < 200$ → MALA이나 HMC이 낫다
   - $d > 200$ → HMC/NUTS이 꼭 필요하다(또는 특화된 방법)

3. **온전한 조건부 분포를 다룰 수 있는가?**
   - 그렇다 → 깁스를 고를 수 있다
   - 얼마쯤 → 깁스 안의 메트로폴리스
   - 아니다 → 온전한 MH이나 기울기 기반

4. **과녁의 봉우리가 여럿인가?**
   - 그렇다 → 온도 다루기와 여러 사슬을 생각해 보아라
   - 아니다 → 표준 방법으로 넉넉할 듯하다

5. **셈의 제약은 무엇인가?**
   - 기울기가 비싸다 → 무작위 걸음 MH이나 깁스
   - 병렬 밑천이 있다 → 여러 사슬
   - 실시간이다 → 빨리 섞여야 한다

### 결정 나무

```
Start
  │
  ├─ Is target differentiable?
  │   │
  │   ├─ NO ──→ Are conditionals tractable?
  │   │          │
  │   │          ├─ YES ──→ Gibbs Sampling
  │   │          │
  │   │          └─ NO ───→ Random Walk MH
  │   │
  │   └─ YES ─→ Is dimension high (d > 20)?
  │              │
  │              ├─ NO ──→ Any method works; try MALA first
  │              │
  │              └─ YES ─→ Is gradient cheap?
  │                        │
  │                        ├─ YES ──→ HMC / NUTS
  │                        │
  │                        └─ NO ───→ MALA or Stochastic Gradient
```

---

## 방법마다의 됨됨이

### 무작위 걸음 메트로폴리스-헤이스팅스

**가장 알맞은 곳**:

- 낮은 차원($d < 20$)
- 미분할 수 없는 과녁
- 빠른 시제품 만들기
- 섞인 모형의 이산 성분

**피해야 할 때**:

- $d > 50$(섞임을 감당할 수 없다)
- 강한 상관(매개변수를 바꾸지 않았을 때)
- 실시간 쓰임새

**맞추기**:
```python
# 이것으로 시작해 받아들임에 따라 다듬기
sigma = 2.4 * np.std(initial_samples, axis=0) / np.sqrt(d)
# 목표: 차원이 높으면 받아들임 20-25%, 낮으면 50%까지
```

**빨간불**:

- 받아들임 비율이 10% 미만이거나 50% 초과
- 자취 그림이 "들러붙음"을 보인다
- 되풀이마다의 ESS이 0.01 미만

### 깁스 표집

**가장 알맞은 곳**:

- 켤레 모형(베이즈 선형 회귀, 가우스 섞음 모형)
- 조건부 분포가 표준 분포이다
- 켤레 앞확률을 갖는 층 모형
- 성긴 짜임의 높은 차원

**피해야 할 때**:

- 변수 사이의 강한 상관
- 조건부 분포가 표준이 아니다
- 한꺼번에 새로 고치는 편이 더 효율적이다

**맞추기**: 대체로 맞출 것이 없지만:
```python
# 무작위 훑기는 상관이 있을 때 도움이 된다
scan_order = np.random.permutation(d)  # 되풀이마다

# 서로 얽힌 묶음에 쓰는 덩어리 깁스
blocks = [[0, 1, 2], [3, 4, 5], ...]  # 서로 얽힌 변수 묶기
```

**빨간불**:

- 어떤 좌표에서 섞임이 몹시 느리다
- 받아들임이 100%인데도 자기상관이 높다
- 조건부 표집이 느리거나 복잡하다

### MALA(메트로폴리스 바로잡은 랑주뱅)

**가장 알맞은 곳**:

- 중간 차원($20 < d < 200$)
- 매끄럽고 미분할 수 있는 로그 밀도
- 기울기가 쌀 때
- 무작위 걸음 MH에서 한 걸음 올라설 때

**피해야 할 때**:

- 과녁에 끊긴 곳이 있다
- 기울기가 비싸거나 없다
- 차원이 아주 높다(HMC이 낫다)
- 꼬리가 무겁다(기울기가 미덥지 않다)

**맞추기**:
```python
# 첫 걸음 크기
epsilon = 1.0 / d**(1/6)

# 받아들임 57%쯤을 목표로 맞추기
def adapt_epsilon(epsilon, accept_rate, target=0.574):
    if accept_rate > target + 0.05:
        return epsilon * 1.1
    elif accept_rate < target - 0.05:
        return epsilon * 0.9
    return epsilon
```

**빨간불**:

- 받아들임 비율이 57%에서 멀다
- 기울기 값 매기기가 NaN이나 Inf을 되돌린다
- 무작위 걸음 MH보다 훨씬 나쁘다(기울기에 문제가 있다는 뜻이다)

### 해밀턴 몬테카를로

**가장 알맞은 곳**:

- 높은 차원($d > 50$)
- 매끄럽고 로그 오목한(또는 봉우리가 살짝 여럿인) 과녁
- 기울기 값이 감당할 만할 때
- 높은 ESS이 필요할 때

**피해야 할 때**:

- 과녁을 미분할 수 없다
- 이산 매개변수가 있다
- 봉우리가 여럿이고 벽이 높다
- 기울기가 아주 비싸다

**맞추기**:
```python
# L을 맞추지 않으려고 NUTS 쓰기
# 손수 하는 HMC에서는:
epsilon = 0.1 / d**(1/4)
L = int(np.ceil(np.sqrt(d)))

# 질량 행렬: 대각으로 시작하고 상관이 있으면 빽빽한 행렬 생각해 보기
M = np.diag(1.0 / np.var(warmup_samples, axis=0))
```

**빨간불**:

- 갈라져 나가는 옮김(에너지 오차 > 1000)
- (NUTS에서) 나무 깊이가 최대에 부딪힌다
- 낮은 E-BFMI(< 0.2)
- 받아들임 비율이 50-80% 밖이다

---

## 문제마다의 권함

### 베이즈 선형 회귀

$$
y = X\beta + \epsilon, \quad \beta \sim \mathcal{N}(0, \sigma_\beta^2 I), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

**권함**: **깁스 표집**

**왜**: 온전한 조건부 분포를 쓸 수 있다:

- $\beta | y, \sigma^2 \sim \mathcal{N}(\cdot, \cdot)$
- $\sigma^2 | y, \beta \sim \text{Inverse-Gamma}(\cdot, \cdot)$

```python
def gibbs_linear_regression(y, X, n_samples, prior_var=100):
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    
    # 초기화한다
    sigma2 = 1.0
    beta = np.zeros(p)
    
    samples = []
    for _ in range(n_samples):
        # beta | y, sigma2 표집
        V_post = np.linalg.inv(XtX / sigma2 + np.eye(p) / prior_var)
        m_post = V_post @ (Xty / sigma2)
        beta = np.random.multivariate_normal(m_post, V_post)
        
        # sigma2 | y, beta 표집
        resid = y - X @ beta
        a_post = n / 2
        b_post = np.sum(resid**2) / 2
        sigma2 = 1 / np.random.gamma(a_post, 1/b_post)
        
        samples.append({'beta': beta.copy(), 'sigma2': sigma2})
    
    return samples
```

### 베이즈 로지스틱 회귀

$$
y_i \sim \text{Bernoulli}(\sigma(x_i^T\beta)), \quad \beta \sim \mathcal{N}(0, \sigma_\beta^2 I)
$$

**권함**: **HMC / NUTS**

**왜**:

- 켤레가 아니다(깁스를 쓸 수 없다)
- 기울기가 싸다: $\nabla \log p(\beta|y) = X^T(y - \hat{y}) - \beta/\sigma_\beta^2$
- 차원이 중간에서 높음까지일 수 있다

```python
def logistic_regression_nuts(y, X, n_samples):
    n, p = X.shape
    
    def log_prob(beta):
        logits = X @ beta
        ll = np.sum(y * logits - np.log(1 + np.exp(logits)))
        prior = -0.5 * np.sum(beta**2) / 100  # 앞확률 흩어짐 100
        return ll + prior
    
    def grad_log_prob(beta):
        probs = 1 / (1 + np.exp(-X @ beta))
        grad_ll = X.T @ (y - probs)
        grad_prior = -beta / 100
        return grad_ll + grad_prior
    
    return nuts_sample(log_prob, grad_log_prob, np.zeros(p), n_samples)
```

### 가우스 섞음 모형

$$
p(x|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
$$

**권함**: **깁스 표집**(숨은 배정과 함께)

**왜**: 자료를 덧붙이면 조건부 분포를 다룰 수 있다.

```python
def gmm_gibbs(X, K, n_samples):
    n, d = X.shape
    
    # 초기화한다
    z = np.random.randint(0, K, n)  # 배정
    
    samples = []
    for _ in range(n_samples):
        # 배정 세기
        N_k = np.bincount(z, minlength=K)
        
        # pi | z 표집
        pi = np.random.dirichlet(1 + N_k)
        
        # mu_k, Sigma_k | z, X 표집
        mu, Sigma = [], []
        for k in range(K):
            X_k = X[z == k]
            if len(X_k) > d:
                mu_k = X_k.mean(axis=0)  # 간추림, 제대로 된 뒤확률을 쓸 것
                Sigma_k = np.cov(X_k.T)
            else:
                mu_k = np.zeros(d)
                Sigma_k = np.eye(d)
            mu.append(mu_k)
            Sigma.append(Sigma_k)
        
        # z | pi, mu, Sigma, X 표집
        for i in range(n):
            log_probs = np.log(pi) + np.array([
                multivariate_normal.logpdf(X[i], mu[k], Sigma[k]) 
                for k in range(K)
            ])
            probs = softmax(log_probs)
            z[i] = np.random.choice(K, p=probs)
        
        samples.append({'pi': pi, 'mu': mu, 'Sigma': Sigma, 'z': z.copy()})
    
    return samples
```

### 층 모형

$$
y_{ij} \sim \mathcal{N}(\mu + \alpha_j, \sigma^2), \quad \alpha_j \sim \mathcal{N}(0, \tau^2)
$$

**권함**: **가운데를 벗긴 매개변수화를 쓴 HMC** 또는 **깁스**

**가운데를 둔 것이 왜 무너지나**: "깔때기" 기하가 말썽을 일으킨다.

```python
# 가운데 맞춤(말썽 있음)
# alpha_j ~ N(0, tau^2)
# y_ij ~ N(mu + alpha_j, sigma^2)

# 가운데 벗김(더 나음)
# eta_j ~ N(0, 1)
# alpha_j = tau * eta_j
# y_ij ~ N(mu + tau * eta_j, sigma^2)

def hierarchical_hmc_noncentered(y, groups, n_samples):
    """층 모형을 위한, 가운데를 벗긴 매개변수화."""
    J = len(np.unique(groups))
    
    def log_prob(params):
        mu, log_sigma, log_tau, eta = unpack(params)
        sigma, tau = np.exp(log_sigma), np.exp(log_tau)
        alpha = tau * eta
        
        # 가능도
        means = mu + alpha[groups]
        ll = norm.logpdf(y, means, sigma).sum()
        
        # 앞확률
        lp = norm.logpdf(eta, 0, 1).sum()  # eta에 표준 정규
        lp += norm.logpdf(mu, 0, 10)
        lp += norm.logpdf(log_sigma, 0, 1)  # 로그 정규 앞확률
        lp += norm.logpdf(log_tau, 0, 1)
        
        return ll + lp
    
    return hmc_sample(log_prob, grad_log_prob, init_params, n_samples)
```

### 차원 높은 성긴 회귀

$$
y = X\beta + \epsilon, \quad \beta_j \sim (1-\pi)\delta_0 + \pi \mathcal{N}(0, \sigma_\beta^2)
$$

**권함**: **못과 판을 쓴 깁스** 또는 **변분 추론**

**왜**: 이산 포함 지시자가 깁스식 새로 고치기를 요구한다.

---

## 섞음 전략과 알아서 맞추는 전략

### 깁스 안의 메트로폴리스

(다룰 수 있는 조건부 분포에는) 깁스를, (나머지에는) MH을 섞어 쓴다:

```python
def metropolis_within_gibbs(current, conditionals, mh_params):
    """
    conditionals: 매개변수 이름을 표집 함수로 잇는 사전
    mh_params: 매개변수 이름을 MH 제안 매개변수로 잇는 사전
    """
    for param in current.keys():
        if param in conditionals:
            # 깁스 걸음
            current[param] = conditionals[param](current)
        else:
            # 메트로폴리스 걸음
            current[param] = mh_step(current, param, mh_params[param])
    
    return current
```

### 덩이 새로 고치기

얽힌 매개변수를 묶는다:

```python
def identify_blocks(correlation_matrix, threshold=0.5):
    """서로 얽힌 매개변수의 덩어리 가려내기."""
    import networkx as nx
    
    G = nx.Graph()
    d = correlation_matrix.shape[0]
    G.add_nodes_from(range(d))
    
    for i in range(d):
        for j in range(i+1, d):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)
    
    return list(nx.connected_components(G))
```

### 알아서 맞추는 MCMC

달굼 동안 제안 매개변수를 알아서 맞춘다:

```python
class AdaptiveMCMC:
    def __init__(self, log_prob, d, adapt_schedule=None):
        self.log_prob = log_prob
        self.d = d
        self.mu = np.zeros(d)
        self.Sigma = np.eye(d)
        self.n_adapt = 0
        
    def step(self, x, adapt=True):
        # 맞춰 간 분포에서 내놓기
        x_prop = np.random.multivariate_normal(x, 2.4**2 / self.d * self.Sigma)
        
        # 받아들이거나 물리치기
        log_alpha = self.log_prob(x_prop) - self.log_prob(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
        
        # 맞춰 가기
        if adapt:
            self.n_adapt += 1
            self.update_moments(x)
        
        return x
    
    def update_moments(self, x):
        """평균과 공분산의 흐름 속 새로 고치기."""
        n = self.n_adapt
        delta = x - self.mu
        self.mu += delta / n
        if n > 1:
            self.Sigma = (n-2)/(n-1) * self.Sigma + delta.reshape(-1,1) @ delta.reshape(1,-1) / n
```

---

## 진단에 기댄 고르기

### 방법을 언제 바꾸나

**다음이면 무작위 걸음 MH에서 MALA/HMC으로 바꿔라**:

- ESS/되풀이 < 0.01
- 섞임 시간 > 되풀이 10,000회
- 자취 그림이 강한 자기상관을 보인다

**다음이면 MALA에서 HMC으로 바꿔라**:

- ESS/기울기 < 0.1
- 가장 좋은 걸음 크기가 아주 작다
- 과녁의 조건이 나쁘다

**다음이면 HMC에서 다른 방법으로 바꿔라**:

- 갈라져 나가는 옮김이 많다
- 기울기 셈하기가 도는 시간을 좌우한다
- 이산 매개변수 때문에 기울기를 쓸 수 없다

### 진단 점검표

```python
def diagnose_chain(samples, target='hmc'):
    """사슬을 두루 진단하기."""
    diagnostics = {}
    
    # 기본 통계량
    diagnostics['mean'] = np.mean(samples, axis=0)
    diagnostics['std'] = np.std(samples, axis=0)
    
    # 실효 표본 크기
    diagnostics['ess'] = compute_ess(samples)
    diagnostics['ess_per_sample'] = diagnostics['ess'] / len(samples)
    
    # R-hat(사슬이 여럿이면)
    # diagnostics['rhat'] = compute_rhat(chains)
    
    # 방법마다 다름
    if target == 'hmc':
        diagnostics['target_accept'] = 0.65
    elif target == 'mala':
        diagnostics['target_accept'] = 0.574
    elif target == 'rwm':
        diagnostics['target_accept'] = 0.234
    
    # 권하는 바
    if diagnostics['ess_per_sample'] < 0.01:
        diagnostics['recommendation'] = 'Consider switching to HMC/NUTS'
    elif diagnostics['ess_per_sample'] > 0.5:
        diagnostics['recommendation'] = 'Excellent mixing'
    else:
        diagnostics['recommendation'] = 'Acceptable, but room for improvement'
    
    return diagnostics
```

---

## 빠른 참고 길잡이

### 문제 갈래별

| 문제 갈래 | 첫째 고름 | 둘째 고름 |
|--------------|--------------|---------------|
| 낮은 차원, 켤레 | 깁스 | 무작위 걸음 MH |
| 낮은 차원, 켤레 아님 | MALA | 무작위 걸음 MH |
| 높은 차원, 매끄러움 | HMC/NUTS | MALA |
| 높은 차원, 성김 | 깁스 + MH | 변분 |
| 층 모형 | NUTS(가운데 벗김) | 깁스 |
| 섞음 모형 | 깁스 | EM + 부트스트랩 |
| 이산 매개변수 | 깁스/MH | — |
| 봉우리 여럿 | 병렬 온도 다루기 | 여러 사슬 |

### 셈의 제약별

| 제약 | 권함 |
|------------|----------------|
| 기울기 없음 | 무작위 걸음 MH이나 깁스 |
| 비싼 기울기 | 무작위 걸음 MH, 깁스, 또는 작은 묶음 |
| 기억 공간 빠듯함 | 사슬 하나, 솎아내기 |
| 병렬 하드웨어 | 여러 사슬, 벡터로 만들기 |
| 실시간 | 미리 맞춘 HMC, 짧은 사슬 |

### 진단 결과별

| 진단 | 할 일 |
|------------|--------|
| 낮은 ESS | 표본을 늘리거나 방법을 바꾼다 |
| 높은 R-hat | 더 오래 돌리고 모임을 살핀다 |
| 갈라져 나감이 많음 | 걸음 크기를 줄이고 매개변수를 바꾼다 |
| 낮은 받아들임 | 걸음 크기를 줄인다 |
| 높은 받아들임 | 걸음 크기를 키운다 |

---

## 요약

**방법 고르기의 메타 알고리즘**:

1. **문제의 성격 밝히기**: 차원, 미분 가능함, 짜임
2. **단순하게 시작하기**: 쓸 수 있는 가장 단순한 방법을 해 본다
3. **진단하기**: ESS, R-hat, 자취 그림을 살핀다
4. **되풀이하기**: 진단에 기대어 방법을 바꾸거나 맞춘다
5. **확인하기**: 미덥지 않으면 여러 길을 견준다

**망설여지면**: $d > 20$인 이어지고 미분할 수 있는 과녁에는 **NUTS이 기본 고름**이다. 자취 길이를 스스로 맞추며 요즘의 확률 프로그래밍(Stan, PyMC, NumPyro)에서 표준이 되었다.

---

## 참고 문헌

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. Carpenter, B., et al. (2017). "Stan: A Probabilistic Programming Language." *Journal of Statistical Software*.
3. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.
5. Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.

## 연습문제

1. **결정 나무 써 보기.** 다음 문제마다 결정 나무로 방법을 골라라. (a) 매개변수 5개의 비선형 회귀, (b) 500차원 가우스 과정, (c) 성분 20개의 섞음 모형, (d) 10×10 격자의 이징 모형.

2. **방법 견주기.** 베이즈 선형 회귀에 무작위 걸음 MH, 깁스, MALA, HMC을 구현하여라. $d \in \{10, 50, 100\}$에서 초당 ESS을 견주어라.

3. **진단으로 맞추기.** 잘못 맞춘 HMC(걸음 크기와 자취 길이가 틀린 것)에서 시작하여라. ESS/되풀이 > 0.1이 될 때까지 진단으로 되풀이해 낫게 하여라.

4. **섞음 표집기.** 이어진 매개변수와 이산 매개변수를 함께 갖는 모형에 깁스 안의 메트로폴리스 표집기를 짜라. 순수 MH 길과 견주어라.

5. **무너지는 모습 가려내기.** 방법마다 무너지는 상황을 일부러 만들어라. (a) 높은 차원의 무작위 걸음 MH, (b) 강한 상관에서의 깁스, (c) 끊긴 곳이 있는 MALA, (d) 봉우리가 여럿일 때의 HMC.

---
