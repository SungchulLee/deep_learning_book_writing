# ABC-MCMC

ABC-MCMC는 시뮬레이션 요약을 보조변수로 포함한 MH 전이로 ABC 사후를 탐색한다. 사전 제안보다 유리할 수 있지만 국소 정체·초기화·작은 허용오차의 낮은 채택률 문제는 남는다.

## 1. 왜 필요한가

사전보다 ABC 사후가 좁으면 현재 위치 근처의 제안을 사용해 유효 영역을 더 자주 방문할 수 있다. 그러나 새 시뮬레이션이 ABC 조건을 만족할 확률 자체가 작으면 제안 폭만 줄여도 채택률이 충분히 높아지지 않을 수 있다.

ABC 거절은 고정 목표의 독립 채택값을 주고 ABC-MCMC는 일반적으로 상관된 체인을 준다. 같은 ε·요약·거리와 전체 시뮬레이션 예산에서 함수별 MCSE·ESS를 비교한다.

## 2. ABC-MCMC 알고리즘

상태는 (θ,s)이며 현재 요약은 ρ(s,s_obs)<ε를 만족한다고 하자. θ′~q(·|θ), s′~p_S(·|θ′)를 생성한다. 새 요약이 조건을 만족하지 않으면 현재 상태를 유지한다. 만족하면 다음 확률로 갱신한다.

$$\alpha=\min\left\{1,\frac{p(\theta')q(\theta|\theta')}{p(\theta)q(\theta'|\theta)}\right\}.$$

거절도 체인에 기록한다. simulator는 요약 생성에 사용되고 관측 우도는 직접 평가하지 않는다. 고정 ε를 중간에 바꾸면 같은 정상 목표의 체인이 아니므로 별도 적응·단계 설계가 필요하다.

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

## 3. 이론적 기초

확장 목표는 π_ε(θ,s)∝p(θ)p_S(s|θ)1{ρ(s,s_obs)<ε}다. 제안 q(θ′|θ)p_S(s′|θ′)의 MH 비에서 정·역 시뮬레이션 밀도가 소거된다. 현재 지시함수가 1이고 제안 지시함수가 0이면 거절하며, 둘 다 1이면 사전·제안 비만 남는다. 지시함수 자체가 두 상태에 무조건 대칭이라는 설명 대신 이 소거 과정을 확인한다.

θ의 주변은 p(θ)A_ε(θ)에 비례하는 ABC 사후다. 적절한 Harris 에르고드 조건과 f의 적분 가능성 아래 시간 평균이 이 목표 기댓값에 수렴한다. 일반 상태에서 단순한 기약·비주기라는 표현만으로 모든 수렴·CLT 조건을 대신하지 않는다. 요약·ε의 원사후 근사 오차도 별도로 남는다.

## 4. 구현

abc_mcmc는 초기 θ에서 새 데이터를 한 번 시뮬레이션해 실패하면 오류를 낸다. find_initial_point는 성공한 θ만 반환하므로 이 θ를 전달해도 abc_mcmc 안의 새로운 시뮬레이션이 다시 실패할 수 있다. 성공한 보조 요약까지 유지하거나 초기화 절차를 함께 설계해야 한다. 사전의 양의 지지집합도 확인한다.

proposal_logpdf(to,from) 인수 관례와 유한 로그비·배열 차원, 양의 n_iter를 확인한다. 사전 밖 제안도 먼저 시뮬레이션하므로 불필요하거나 유효하지 않은 모델 호출이 생길 수 있다.

AdaptiveABCMCMC는 별도 클래스이며 앞 함수에 연결되지 않는다. target_rate는 저장만 하고 adapt의 수락 여부에 따른 1.01배·0.99배 조정에 사용하지 않는다. 이는 0.234 목표 조정이 아니다. 공분산 초기 갱신은 낮은 랭크가 될 수 있고 양의 대각 보정이 없으며, 계속 적응의 수렴 조건도 구현하지 않는다.

```python
import numpy as np

def abc_mcmc(theta_init, prior_logpdf, simulator, summary_fn, y_obs,
             distance_fn, epsilon, proposal_fn, proposal_logpdf, n_iter):
    """
    ABC-MCMC 알고리즘.
    
    인수:
        theta_init: 첫 파라메터 값
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

## 5. 독립 표본과 체인의 비용 비교

MCMC는 prior 제안보다 좁은 유효 영역에 집중할 수 있지만 자기상관·워밍업·모드 정체가 추가된다. 거절 방법은 채택이 드물어도 독립 반복과 분산·병렬화를 단순하게 다룰 수 있다.

0.1%·1% 같은 비율만으로 승자를 정하지 않는다. ABC 사후·관심량·시뮬레이션 비용을 맞추고 초기화·조정 비용, 독립 반복의 변동과 모드 탐색을 비교한다. 높은 채택률만으로 MCMC가 더 정확한 것은 아니다.

## 6. 실용적인 고려

랜덤 워크 공분산은 파라메터 스케일과 지지집합에 맞춰 조정한다. 2.4²/d는 일부 점근 결과의 참고값이며 ABC 채택 요인이 추가된 상황의 보편적 최적 공식은 아니다. 잘 설계한 독립 제안도 가능하고 사전과 같을 때와 일반 독립 사후 근사 제안을 구분한다.

ε를 늘리거나 요약을 바꾸면 수치 설정만이 아니라 ABC 목표 자체가 달라진다. 제안 조정과 목표 근사 수준 변경을 따로 보고한다. tune_proposal_scale은 사용자가 정의한 래퍼 함수 인터페이스를 가정한다.

진단 함수는 지연 100까지의 잡음 많은 자기상관을 그대로 더하므로 τ가 비양수이거나 불안정한 ESS를 낼 수 있다. 상수열·짧은 체인, 안정적 절단·여러 체인·함수별 MCSE를 검토해야 한다. 이 값만으로 수렴을 확인하지 않는다.

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

## 7. 변형

한 θ에서 여러 독립 시뮬레이션의 지시함수 평균은 A_ε(θ)의 비음수 불편 추정이다. pseudo-marginal MH로 사용하려면 추정량을 포함한 상태를 유지하고 정·역 비를 일관되게 구성한다. 현재값도 매번 새로 추정해 비에 넣는 임의 방식은 일반적으로 같은 목표를 보장하지 않는다. 아래 함수는 확률 추정만 하며 전체 MH를 구현하지 않는다.

매끄러운 비음수 커널을 사용하면 확장 목표가 p(θ)p(s|θ)K_ε(s,s_obs)로 바뀌고 MH 비에 커널 비가 남는다. 이는 관측모형 오차를 별도로 정의하는 noisy ABC와 임의의 채택 잡음을 모두 같은 것으로 묶을 이유가 되지 않는다.

아래 gibbs_abc는 블록 제안의 사전·정역 제안 비와 공통 결합 목표의 다른 요약 조건을 확인하지 않는다. 일반적인 올바른 Gibbs-ABC 구현으로 소개할 수 없으며 특별한 조건이나 별도 보정이 필요하다.

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
```python
def gibbs_abc(theta, blocks, simulators, summary_fns, s_obs, 
              distance_fns, epsilons, proposals):
    """파라메터 덩어리를 새로 고치는 깁스 방식 ABC-MCMC."""
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

## 8. 보기: 확률 변동성 모델의 추론

예제는 잠재 log variance의 정상 AR(1)과 y_t=exp(h_t/2)η_t를 시뮬레이션한다. 초기식에는 |φ|<1, σ>0가 필요하다. prior는 μ∈(−2,2), φ∈(0.8,0.999), σ∈(0.01,0.5)의 균등 범위다.

log(y²+10⁻⁸)의 평균·분산과 여러 자기상관·절대수익 요약은 편리한 피처가지만 일반적으로 충분 통계량은 아니다. 작은 상수는 log 0을 피하면서 관측 변환도 바꾼다. 이 조각은 관측·요약 생성까지만 있고 실제 proposal·ε·abc_mcmc 실행은 연결하지 않는다. 실제 금융 자료의 성능을 검증한 예제도 아니다.

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
y_obs = sv_simulator([0.0, 0.95, 0.2])  # 참 파라메터
s_obs = sv_summaries(y_obs)

# 첫 점을 찾고 사슬 돌리기...
```

## 연습문제

1. 확장 MH 비에서 시뮬레이터 밀도가 왜 소거되는가? 목표와 제안에 동일한 p_S(s|θ)가 들어가 정·역의 곱이 소거되기 때문이다.

2. find_initial_point의 θ가 다음 함수의 초기 성공을 보장하는가? 성공한 요약을 전달하지 않고 새 시뮬레이션을 하므로 보장하지 않는다.

3. AdaptiveABCMCMC의 target_rate가 실제로 사용되는가? adapt에는 참조되지 않으며 고정 배율 수락/거절 조정은 0.234 목표 알고리즘이 아니다.

4. 여러 시뮬레이션 평균을 MH에 넣을 때 현재 추정량은 어떻게 해야 하는가? 정당한 pseudo-marginal 확장 상태의 일부로 유지해야 한다. 임의로 양쪽을 다시 생성하면 다른 전이가 될 수 있다.

5. gibbs_abc가 일반적으로 올바른가? 사전·제안 비와 전체 ABC 목표의 조건을 확인하지 않으므로 일반적인 정당화가 없다. 블록별 불변 커널을 명시해야 한다.

## 정리하며

ABC-MCMC는 보조 시뮬레이션을 포함한 확장 목표의 MH다. 초기 성공 요약·사전과 제안 비, 고정 ABC 목표와 적응 조건을 유지하며 상관·모드 탐색·원사후 근사 오차를 구분한다.

**참고 문헌**

1. Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). "Markov Chain Monte Carlo Without Likelihoods." *PNAS*.
2. Sisson, S. A., & Fan, Y. (2011). "Likelihood-Free Markov Chain Monte Carlo." In *Handbook of Markov Chain Monte Carlo*.
3. Wegmann, D., Leuenberger, C., & Excoffier, L. (2009). "Efficient Approximate Bayesian Computation Coupled with Markov Chain Monte Carlo Without Likelihood." *Genetics*.
4. Bortot, P., Coles, S. G., & Sisson, S. A. (2007). "Inference for Stereological Extremes." *JASA*.
