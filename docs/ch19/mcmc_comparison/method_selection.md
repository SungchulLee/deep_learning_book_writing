# 실전 알고리즘 선택

알고리즘은 목표의 구조와 필요한 정밀도·예산에 맞춰 고른다. 직접 적분·생성이 가능한지 먼저 확인하고, MCMC가 필요하면 조건부 생성·미분·블록·모드와 평가비용으로 후보를 정한 뒤 진단으로 비교한다.

## 1. 판단의 틀

목표의 적절성·지지집합·이산/연속 변수와 미분 가능한 부분, 정확 조건부 생성 여부를 확인한다. 다음으로 자료 수와 한 번의 밀도·경사·조건부 비용, 메모리·병렬 자원, 관심 평균·분위수의 허용오차를 정한다.

아래 결정 도식의 d>20 같은 경계와 어느 방법이나 가능하다는 문구는 설명용 출발점이지 사용 한계나 정확성 보증이 아니다. 낮은 차원도 다봉성에서 어렵고 큰 차원의 독립 구조는 쉬울 수 있다. 비싼 경사를 MALA가 사용하지 않는 것은 아니며 확률 경사로 대체하면 목표 보정·근사 조건이 달라진다.

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

## 2. 방법마다의 성능

RWM은 유효한 제안·밀도비만으로 시작하기 쉽지만 제안 스케일·공분산·블록이 중요하다. 깁스는 조건부 생성이 편리할 때 후보이며 스캔·블록·주변화와 강한 상관의 비용이 남는다. MALA는 정역 경사 기반 제안을, HMC는 여러 적분 단계와 질량을 조정한다.

아래 MALA 초기 ε∝d⁻¹/⁶는 표준편차 관례에서의 점근 참고값이다. 제안 분산 ε를 쓰는 MALA 클래스에 같은 수치를 넣으면 스텝 단위가 다르다. HMC 예제의 L=ceil(√d)는 보편적 최적 길이가 아니며 εL의 실제 시간과 주파수를 확인해야 한다.

수락률 0.234·0.574·0.65 또는 ESS/추출 0.01은 문제 조사에 참고할 수 있지만 필수 성능 경계는 아니다. MALA가 RWM보다 느리다는 사실만으로 경사 버그를 단정할 수 없고, NUTS 깊이 한계·발산·낮은 E-BFMI도 서로 다른 진단이다.

```python
# 이것으로 시작해 받아들임에 따라 다듬기
sigma = 2.4 * np.std(initial_samples, axis=0) / np.sqrt(d)
# 목표: 차원이 높으면 받아들임 20-25%, 낮으면 50%까지
```
```python
# 무작위 훑기는 상관이 있을 때 도움이 된다
scan_order = np.random.permutation(d)  # 되풀이마다

# 서로 얽힌 배치에 쓰는 덩어리 깁스
blocks = [[0, 1, 2], [3, 4, 5], ...]  # 서로 얽힌 변수 묶기
```
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
```python
# L을 맞추지 않으려고 NUTS 쓰기
# 손수 하는 HMC에서는:
epsilon = 0.1 / d**(1/4)
L = int(np.ceil(np.sqrt(d)))

# 질량 행렬: 대각으로 시작하고 상관이 있으면 빽빽한 행렬 생각해 보기
M = np.diag(1.0 / np.var(warmup_samples, axis=0))
```

## 3. 예제 코드와 실제 추론 목표

### 선형 회귀

가우스 우도·β~N(0,v₀I)에서 β|σ²는 가우스다. 아래 σ² 갱신의 shape n/2, scale RSS/2는 p(σ²)∝1/σ²라는 부적절 사전을 암묵적으로 사용한다. 제대로 정의된 사후인지 별도로 확인해야 하고 RSS=0이면 식이 실패할 수 있다. σ²에 독립인 β 사전과 σ²로 스케일한 켤레 사전의 식은 다르다.

### 로지스틱 회귀

매끄러운 적절한 사후에 HMC·MALA를 검토할 수 있고 Pólya–Gamma 보조변수 깁스도 가능하다. 아래 log(1+exp(logits))·직접 sigmoid는 큰 값에서 넘칠 수 있고 nuts_sample은 외부 정의다. 비켤레라는 이유만으로 깁스가 원리적으로 불가능하다고 하지 않는다.

### 가우스 혼합

완전한 베이즈 깁스에는 성분 매개변수의 조건부 사후에서도 무작위로 생성해야 한다. 아래 gmm_gibbs는 μ·Σ를 표본 평균·공분산으로 대입하고 적은 성분에는 0·I를 넣는다. 이 단계는 지정한 사전의 조건부 사후 생성이 아니므로 일반적인 베이즈 GMM 깁스 표본으로 소개할 수 없다. 라벨 교환·빈 성분·특이 공분산도 별도로 다룬다.

### 계층·희소 모형

비중심 표현 α_j=τ η_j는 약한 관측 정보의 깔때기 기하를 개선할 수 있지만 항상 중심 표현보다 낫지는 않다. 아래 조각은 log σ·log τ에 직접 정규 사전을 정의하므로 σ·τ에 대한 로그정규 사전이다. 이 정의에서는 그 좌표의 추가 야코비안이 필요 없지만 원래 σ·τ 사전을 변환하는 상황과 구분한다. unpack·grad_log_prob·init_params·hmc_sample 등은 외부 정의다. 그룹 인덱스도 0,…,J−1로 맞춰야 한다.

스파이크 앤 슬랩에는 이산 포함변수·블록·주변화·연속 부분 커널 등을 조합할 수 있다. 이산 포함변수 때문에 오직 깁스만 써야 하는 것은 아니며 변분 근사로 바꾸면 목표 근사도 달라진다.

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

## 4. 섞음 전략과 알아서 맞추는 전략

Metropolis-within-Gibbs는 각 블록의 완전조건부를 불변으로 보존하는 커널을 순차 합성한다. current 사전을 제자리 갱신하므로 순서와 최신 값의 의미를 확인한다. 조건부 함수·mh_step은 외부 정의다.

identify_blocks는 상관 임계값의 그래프 연결성분을 만든다. 한 블록 안의 모든 쌍이 그 임계값보다 크다는 뜻은 아니고 0.5는 선택한 휴리스틱이다. 비용·국소 기하와 상관 추정의 품질을 함께 평가한다.

AdaptiveMCMC는 adapt_schedule 인수를 저장·사용하지 않고 adapt=True면 계속 모멘트를 갱신한다. 공분산이 낮은 랭크로 축소될 수 있으며 대각 보정·유효 적응 조건을 구현하지 않는다. 각 시점의 대칭 가우스 비만으로 전체 적응 체인의 수렴이 자동 증명되지는 않는다.

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

## 5. 진단에 기댄 고르기

진단은 원인을 좁히는 데 사용한다. 작은 함수별 ESS·큰 MCSE에는 실행 길이·상관·탐색·목표 기하를 확인하고, 높은 R-hat에는 체인 차이·모드·비정상성을 조사한다. 발산과 최대 깊이 도달은 같은 오류가 아니다.

아래 diagnose_chain은 compute_ess 외부 함수를 사용하며 반환이 벡터라면 if의 불리언 판정이 모호해질 수 있다. target_accept는 목표값을 저장할 뿐 실제 수락률을 계산하지 않는다. ESS/추출 임계값으로 자동 출력하는 Excellent·Acceptable은 수렴·정확도 보증이 아니고, 해당 함수의 MCSE와 비용·경고를 함께 봐야 한다.

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

## 6. 빠른 참고 길잡이

직접 해석·생성이 어렵고 저렴한 정확 조건부가 있으면 깁스·블록 방법, 연속 매끄러운 목표와 경사가 있으면 MALA·HMC/NUTS, 이산·비미분 부분에는 적합한 MH·조건부·주변화를 후보로 삼는다. 모드 장벽에는 템퍼링·모드 제안 등 별도 탐색 전략을 검토한다.

EM+부트스트랩은 모형에 따른 점추정·빈도적 재표본 절차로 베이즈 사후 MCMC의 단순 대체품이 아니다. 실시간 요구가 짧은 체인의 정확성을 보증하지 않고 단일 체인·thinning은 저장을 줄이면서 진단 정보를 잃을 수 있다.

같은 목표·관심량·예산에서 복수의 설정과 독립 기준을 비교한다. 후보의 장점, 남은 초기·탐색·근사 오차와 실제 MCSE를 구분해 보고한다.

## 연습문제

1. 알려진 공분산의 가우스 문제에 먼저 확인할 것은? 직접 생성·해석적 적분이 가능한지다. MCMC를 기본 전제로 두지 않는다.

2. 선형 회귀 코드의 σ² 사전은 무엇인가? 조건부 IG(n/2,RSS/2)는 p(σ²)∝1/σ²와 독립 가우스 β 사전에 대응하며 사후 적절성이 필요하다.

3. gmm_gibbs가 성분 매개변수 사후를 생성하는가? 평균·공분산을 경험값으로 대입하므로 일반적인 베이즈 깁스가 아니다.

4. 상관 그래프의 연결성분 안에서 모든 쌍의 상관이 큰가? 경로로 연결되었을 뿐이므로 아니다. 블록 비용과 실제 의존성도 확인한다.

5. 자동 진단의 Excellent 라벨은 충분한가? 함수별 ESS·MCSE·모드 탐색과 경고·비용을 대체하지 못한다. 원래 목표와 코드 인터페이스도 검토해야 한다.

## 정리하며

알고리즘 선택은 직접 생성 가능성·조건부·미분·블록 구조와 원하는 오차·예산을 기준으로 한다. 코드 이름보다 실제 사전·갱신·반환·진단을 확인하고 고정 차원·수락률 경계로 정확성을 판정하지 않는다.

**참고 문헌**

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. Carpenter, B., et al. (2017). "Stan: A Probabilistic Programming Language." *Journal of Statistical Software*.
3. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.
5. Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.
