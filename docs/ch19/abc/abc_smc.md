# ABC-SMC(순차 몬테카를로)

ABC-SMC는 점차 작은 허용오차의 ABC 목표를 입자 집단으로 근사한다. 이전 집단을 제안에 활용하되 올바른 중요도 가중을 사용해야 하며, 적응 일정이나 많은 입자가 정확한 원사후를 자동으로 보장하지는 않는다.

## 1. 왜 필요한가

넓은 사전의 비효율과 단일 체인의 국소 정체를 줄이기 위해 여러 입자를 유지하고 이전 단계로 제안을 구성한다. 허용오차는 미리 정하거나 관측한 거리로 적응할 수 있다.

입자가 많다고 모든 모드를 유지하는 것은 아니고 재표집·국소 제안이 다양성을 잃을 수 있다. 최종 목표는 선택한 요약·거리·양의 ε의 ABC 사후이며 원자료 사후와의 차이는 별도다.

## 2. ABC-SMC 알고리즘

초기 집단은 prior 기반 ABC 거절로 생성하고 동일 웨이트를 준다. 이후에는 조상을 이전 웨이트로 고르고 K_t(θ|조상)로 이동한 뒤 새 시뮬레이션이 현재 ε_t를 만족하면 채택한다.

전체 주변 제안은 q_t(θ)=∑_j w_j^{t−1}K_t(θ|θ_j^{t−1})다. 채택한 입자는 단순한 π_t 표본이 아니므로 다음 중요도 가중을 사용한다.

$$\widetilde w_i^{(t)}=p(\theta_i^{(t)})/q_t(\theta_i^{(t)}),\qquad w_i=\widetilde w_i/\sum_k\widetilde w_k.$$

아래 의사코드는 이 모집단형 ABC 중요도 방식이다. 다른 ABC-SMC의 이동·재표집 구성에는 그 방식에 맞는 가중 유도가 필요하다.

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

## 3. 구현

ABCSMC 클래스는 run(epsilon_schedule)과 초기·다음 집단 생성 메서드를 제공한다. n_particles는 양수이고 파라메터는 배열, 공분산은 유효한 양의 정부호를 가정한다. 거절 루프에 최대 시도 상한이 없어 작은 ε·불가능한 관측에서 끝나지 않을 수 있다.

run_adaptive는 클래스 밖의 독립 함수로 제시되어 ABCSMC에 바인딩되지 않는다. 뒤의 sampler.run_adaptive 호출은 현재 클래스만으로 지원되지 않으며 epsilon_init=None일 때 호출하는 calibrate_initial_epsilon도 정의되지 않았다. 이 조각을 완결된 적응 샘플러로 볼 수 없다.

적응 함수의 분위수는 비가중 거리 기준이다. 이는 일정 선택의 휴리스틱이며 가중 목표의 정확한 분위수가 아니다. 목표 ε에 닿기 전에 정체·최대 세대로 중단할 수 있으므로 실제 마지막 ε를 확인해야 한다. 밀도 분모의 직접 pdf 합에는 언더플로·오버플로가 가능하다.

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

## 4. 이론적 성질

ABC 목표 π_t(θ)∝p(θ)A_t(θ)다. 그러나 시뮬레이션 조건을 통과한 제안의 분포는 q_t(θ)A_t(θ)에 비례한다. 따라서 두 분포의 비에서 A_t가 소거되어 웨이트 p/q_t를 얻는다. 채택 전 q_t에 π_t/q_t를 쓰다가 A_t를 이유 없이 지우는 유도가 아니다.

웨이트 ESS=1/∑w_i²는 [1,N]의 집중도 지표다. 재표집하면 N으로 돌아가도 복제 입자가 독립 정보를 늘리는 것은 아니다. 입자 근사·적응·가중 정밀도의 조건을 따로 검토한다.

ε→0의 한계는 정칙성 아래 p(θ|S(y))이며 충분성 등이 있어야 원자료 사후와 같다. 고정 집단 크기의 유한 단계에서는 시뮬레이션·적응·정보 로스 오차가 모두 남는다. [정확한 중요도 유도의 모집단 ABC 논문](https://arxiv.org/abs/0805.2256)을 참고한다.

## 5. 섭동 커널과 공분산

가우스 섭동 커널의 2×가중 공분산은 실용적인 선택이지 모든 목표의 기대 시뮬레이션 수를 최소화하는 보편적 정리가 아니다. 4/(d+2)×공분산을 모든 ABC의 최적 커널이라고 제시하지 않는다. 목표·기준과 허용오차에 따라 적절한 커널이 달라진다.

local_kernel은 가까운 k개 입자의 공분산을 계산할 뿐 국소 선형 회귀를 하지 않는다. k·거리 스케일·정규화와 양의 정부호를 확인한다. 조상마다 다른 공분산을 쓰면 웨이트 분모의 전체 혼합밀도에도 각 조상별 커널을 반영해야 한다.

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

## 6. 조상 선택과 재표집

현재 ABCSMC는 매 제안마다 가중 조상을 뽑지만 ESS 임계값의 별도 maybe_resample을 run에서 호출하지 않는다. 추가 재표집을 한다면 이후 가중·이동 단계의 명세를 일관되게 구성해야 한다.

아래 systematic_resample은 하나의 균등 난수로 누적 웨이트 격자를 만든다. 웨이트가 비음수·정규화되어 있고 끝 누적합의 반올림 경계를 처리해야 한다. 체계적 재표집이 모든 함수·입자 순서에서 다항 재표집보다 분산이 작다는 일반적인 순위는 없다. 재표집 변동·입자 중복·계보 다양성을 함께 본다.

```python
def maybe_resample(particles, weights, threshold_ratio=0.5):
    """ESS이 임계값 아래이면 다시 표집하기."""
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

## 7. 실용적인 고려

입자 수는 원하는 함수 정밀도와 전체 시뮬레이션 예산으로 정한다. N=1000~10000이라는 범위만으로 충분성을 보장하지 않는다. ε 일정·채택률·유효한 입자 수·모드 유지·실제 마지막 ε와 총 시도 수를 함께 기록한다.

단계별 웨이트 분모를 N개 혼합밀도로 계산하면 집단마다 O(N²)의 커널 평가가 필요하다. 이 비용은 시뮬레이터 자체 비용과 별개다. 병렬 조각은 채택한 입자·거리만 반환하고 중요도 웨이트를 계산하지 않으므로 전체 가중 ABC-SMC를 대체하지 않는다. 작업자 난수·직렬화와 종료 상한도 확인한다.

```python
epsilon_schedule = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
```
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

## 8. 변형

회귀 조정은 입자를 θ_adj=θ−m(S_sim)+m(S_obs)로 이동해 유한 ε 효과를 줄이려는 근사다. 아래 코드는 같은 비가중 회귀를 입자마다 다시 적합하며 ABC 가중·국소 커널을 반영하지 않는다. 조정값이 사전 지지집합 밖으로 갈 수도 있어 정확 사후 보정이라고 읽지 않는다.

multi_distance는 성분별 가중 절댓값의 합으로 하나의 거리를 정의한다. 웨이트의 부호·스케일과 정보 의미를 명시한다. Population Monte Carlo와 SMC의 이름만으로 일정·조정이 필요 없어지지 않으며 실제 사용한 제안과 가중식으로 방법을 구분한다.

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
```python
def multi_distance(s1, s2, weights):
    """성분별 거리를 무게 두어 합친 것."""
    return np.sum(weights * np.abs(s1 - s2))
```

## 9. 보기: 생태 모델

Lotka–Volterra 조각은 연속시간 반응 과정의 정확 시뮬레이션이 아니라 결정적 Euler 변화에 가우스 잡음을 더하고 음수를 0으로 자르는 장난감 모형이다. 잡음 계수에 √dt가 없으므로 dt를 바꿀 때 같은 SDE를 근사하는 표준 Euler–Maruyama 식이라고 볼 수 없다.

prior_sampler는 유한 직사각형 범위의 균등분포를 생성하지만 prior_logpdf는 양의 직교영역 전체에서 0을 반환한다. 지지집합이 일치하지 않고 후자는 전체 양의 영역에서 proper한 균등밀도가 아니다. 초기·후속 세대가 같은 사전 목표를 따른다는 정당화가 없다. 또한 run_adaptive는 클래스 메서드로 정의되지 않아 마지막 호출은 현재 조각 그대로 동작하지 않는다.

관측은 같은 장난감 시뮬레이터에서 생성한 합성 자료다. 평균·표준편차·상관은 일반적으로 충분 요약이 아니며 상수·소멸 경로에서 상관 NaN도 확인해야 한다. 실제 생태 자료의 검증 결과로 소개하지 않는다.

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

## 10. 비교

ABC 거절은 독립 채택·단순성을, ABC-MCMC는 국소 제안·상관 체인을, ABC-SMC는 집단 제안과 가중·단계 비용을 갖는다. 어느 방식이 항상 가장 효율적인지는 정해져 있지 않다.

같은 요약·거리·최종 ε와 관심량을 사용하고 총 시뮬레이션·커널 평가·워밍업 또는 pilot·병렬 비용을 맞춰 비교한다. 적응과 입자 수만으로 원사후의 정확성을 보장하지 않는다.

## 연습문제

1. 채택 후 중요도 가중에 A_ε가 왜 없는가? 채택 제안 자체가 q_t A_ε에 비례해 목표 p A_ε와의 비에서 소거되기 때문이다.

2. 새 입자를 비가중으로 평균해도 되는가? 후속 집단은 일반적으로 중요도 가중이 필요하며 p/q_t를 정규화한 평균을 사용한다.

3. local_kernel은 회귀인가? 가까운 입자의 공분산 계산이며 회귀를 수행하지 않는다. 조상별 커널 변화는 분모에도 반영해야 한다.

4. 생태 예제의 사전 두 인터페이스는 일치하는가? 유한 상자 샘플러와 전체 양의 영역 로그값이 다르므로 일치하지 않는다.

5. sampler.run_adaptive 호출이 지원되는가? 독립 함수로 정의되어 클래스에 바인딩되지 않았으므로 현재 클래스에는 없다. 출력·실행을 가정하기 전에 실제 인터페이스를 확인해야 한다.

## 정리하며

ABC-SMC에서는 채택을 포함한 제안분포에서 정확한 중요도 가중을 유도해야 한다. 요약·허용오차의 근사와 입자 오차를 구분하고 사전의 일관성, 실제 적응 메서드와 재표집·병렬 조각의 범위를 확인한다.

**참고 문헌**

1. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
2. Beaumont, M. A., Cornuet, J.-M., Marin, J.-M., & Robert, C. P. (2009). "Adaptive Approximate Bayesian Computation." *Biometrika*.
3. Del Moral, P., Doucet, A., & Jasra, A. (2012). "An Adaptive Sequential Monte Carlo Method for Approximate Bayesian Computation." *Statistics and Computing*.
4. Toni, T., Welch, D., Strelkowa, N., Ipsen, A., & Stumpf, M. P. (2009). "Approximate Bayesian Computation Scheme for Parameter Inference and Model Selection in Dynamical Systems." *Journal of the Royal Society Interface*, 6, 187–202.
