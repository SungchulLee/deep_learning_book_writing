# HMC 알고리즘

HMC의 한 전이는 운동량 생성, 고정된 수치 궤적과 에너지 차이의 MH 보정으로 구성한다. 이 절은 목표 불변성의 논리와 실제 교육용 구현의 범위를 구분한다.

## 1. 알고리즘 훑어보기

입력은 유효한 초기 위치 x₀, 표본 수 N, 스텝 크기 ε>0, 양의 정수 Leapfrog 수 L, 상수 양의 정부호 M이다. 목표 π(x)∝exp(−U(x))와 필요한 경사를 평가할 수 있어야 한다.

각 반복에서 p~N(0,M)를 뽑고 L단계 Leapfrog를 적용한 뒤 α=min(1,exp(H_old−H_new))로 위치를 갱신한다. 거절하면 현재 위치를 다음 표본으로 남긴다. 출력은 일반적으로 의존적인 체인이며 임의 초기값에서 즉시 목표의 독립 표본이 되는 것은 아니다. 아래 의사코드는 이 흐름을 보여 준다.

```
for t = 1 to N:
    # 스텝 1: 운동량 표집
    v ~ Normal(0, M)
    
    # 스텝 2: 첫 해밀턴 함수 값 셈하기
    H_current = U(x⁽ᵗ⁻¹⁾) + ½ vᵀ M⁻¹ v
    
    # 스텝 3: 개구리뜀 적분
    x_prop, v_prop = Leapfrog(x⁽ᵗ⁻¹⁾, v, ε, L)
    
    # 스텝 4: 운동량 부호 뒤집기(되돌릴 수 있게)
    v_prop = -v_prop
    
    # 스텝 5: 제안의 해밀턴 함수 값 셈하기
    H_prop = U(x_prop) + ½ v_propᵀ M⁻¹ v_prop
    
    # 스텝 6: 메트로폴리스-헤이스팅스 받아들임
    α = min(1, exp(H_current - H_prop))
    u ~ Uniform(0, 1)
    
    if u < α:
        x⁽ᵗ⁾ = x_prop    # 받아들임
    else:
        x⁽ᵗ⁾ = x⁽ᵗ⁻¹⁾    # 물리침
```

## 2. 스텝마다 자세히

운동량을 완전히 재생성하면 현재 위치가 목표를 따르는 경우 결합 상태가 π(x)N(p;0,M)를 따른다. 새 운동량은 에너지 면을 바꿀 기회를 주지만 에르고드성을 자동으로 보장하지는 않는다.

H=U+pᵀM⁻¹p/2를 시작·제안 상태에서 평가한다. L단계에는 시작 경사를 포함해 L+1회, 이미 캐시한 경우 새 경사 L회가 필요하다. T=εL는 적분시간이며 끝점 이동거리와 같지 않다.

운동량 반전 R과 Leapfrog 합성은 가역적인 자기 역 제안을 구성한다. 완전 재생성 후 위치만 기록할 때 반전을 생략해도 같은 위치 전이를 얻는 관례가 있지만, 부분 운동량 유지 방식에서는 거절·반전 규칙을 다시 설계해야 한다.

ΔH=H_new−H_old≤0이면 수락확률 1, 양수이면 exp(−ΔH)다. 아래 짧은 수락 코드는 U∈[0,1)과 비교해 이 규칙을 구현하지만 큰 음의 ΔH에서 exp가 넘칠 수 있다. 실제로는 유한 로그값을 확인한 뒤 로그 수락 비교가 안정적이다. 운동량 생성 조각의 두 줄은 대안 표현이며 둘을 모두 실행하면 두 번째가 첫 번째를 덮어쓴다.

```python
v = np.random.multivariate_normal(np.zeros(d), M)
# 또는 같은 말로:
v = L_M @ np.random.randn(d)  # 여기서 M = L_M @ L_M.T
```
```python
delta_H = H_prop - H_current
if np.random.rand() < np.exp(-delta_H):
    x = x_prop  # 받아들임
# 아니면 지금의 x 그대로 두기(물리침)
```

## 3. 확장 목표의 불변성

확장 상태 z=(x,p), 목표 ρ(z)∝exp(−H(z)), 제안 T=RΦ_ε^L를 생각하자. 고정 ε·L의 Leapfrog 가역성에서 T²=I, 체적 보존에서 |det DT|=1을 얻는다. R의 행렬식은 (−1)^d이지만 절댓값은 1이다.

자기 역 제안에 α(z)=min(1,ρ(Tz)/ρ(z))를 붙이면 흐름 밀도는 min{ρ(z),ρ(Tz)}로 대칭이다. 거절 시 원래 상태에 머무는 항도 포함한다. 운동량 재생성은 결합 목표를 보존하고, 이를 적분한 위치 커널도 목표 π를 보존한다.

불변성 증명은 독립 표본 생성이나 유한 반복의 정확도를 보장하지 않는다. 접근성·재귀성·비주기성 등 상태공간에 맞는 조건과 수치 구현을 검토해야 한다. 무작위 ε·L을 쓰려면 상태와 독립인 올바른 커널 혼합 등 해당 설계의 정당화가 필요하다.

## 4. 구현

아래 HMC 클래스는 상수 질량과 고정 ε·L을 사용하는 NumPy 교육용 구현이다. n_warmup번의 같은 step을 버릴 뿐 설정 적응은 하지 않는다. 입력 차원·유한 로그확률·경사, 양의 표본 수·스텝 수와 M의 대칭 양의 정부호를 확인한다.

step은 NaN 에너지 차이를 거절하지만 모든 비유한 상태를 진단·기록하지 않는다. exp(−delta_H)의 넘침도 로그 비교로 다룰 수 있다. 별도의 발산·에너지 기록 메서드는 클래스에 구현되어 있지 않다.

알려진 2차원 정규 평균·공분산은 확인 기준이지만 한 번의 표본 통계 일치가 모든 목표의 정확성 증명은 아니다. 신뢰할 수 있는 라이브러리 수준의 진단·적응·입력 처리를 갖춘 구현과 구분한다.

```python
import numpy as np

class HMC:
    def __init__(self, log_prob, grad_log_prob, dim, 
                 epsilon=0.1, L=10, M=None):
        """
        해밀턴 몬테카를로 표집기.
        
        인수:
            log_prob: (상수배를 빼고) log π(x)을 돌려주는 함수
            grad_log_prob: ∇ log π(x)을 돌려주는 함수
            dim: 타깃 분포의 차원
            epsilon: 개구리뜀 스텝 크기
            L: 개구리뜀 스텝 수
            M: 질량 행렬(기본값: 항등 행렬)
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.epsilon = epsilon
        self.L = L
        
        if M is None:
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)
            self.L_M = np.eye(dim)
        else:
            self.M = M
            self.M_inv = np.linalg.inv(M)
            self.L_M = np.linalg.cholesky(M)
    
    def kinetic_energy(self, v):
        """운동 에너지 K(v) = ½ vᵀ M⁻¹ v 셈하기"""
        return 0.5 * v @ self.M_inv @ v
    
    def hamiltonian(self, x, v):
        """전체 에너지 H(x,v) = U(x) + K(v) 셈하기"""
        U = -self.log_prob(x)
        K = self.kinetic_energy(v)
        return U + K
    
    def leapfrog(self, x, v):
        """개구리뜀 L스텝 밟기."""
        x = x.copy()
        v = v.copy()
        
        # 운동량의 반 스텝
        v = v + (self.epsilon / 2) * self.grad_log_prob(x)
        
        # 온 스텝
        for _ in range(self.L - 1):
            x = x + self.epsilon * self.M_inv @ v
            v = v + self.epsilon * self.grad_log_prob(x)
        
        # 마지막 자리 스텝
        x = x + self.epsilon * self.M_inv @ v
        
        # 운동량의 마지막 반 스텝
        v = v + (self.epsilon / 2) * self.grad_log_prob(x)
        
        return x, v
    
    def sample_momentum(self):
        """N(0, M)에서 운동량 표집하기."""
        return self.L_M @ np.random.randn(self.dim)
    
    def step(self, x):
        """HMC 되풀이 한 번 하기."""
        # 운동량 표집
        v = self.sample_momentum()
        
        # 지금의 해밀턴 함수 값
        H_current = self.hamiltonian(x, v)
        
        # 개구리뜀 적분
        x_prop, v_prop = self.leapfrog(x, v)
        
        # 제안의 해밀턴 함수 값(운동량 부호 뒤집기는 H에 영향 없음)
        H_prop = self.hamiltonian(x_prop, v_prop)
        
        # 메트로폴리스-헤이스팅스 받아들임
        delta_H = H_prop - H_current
        
        if np.isnan(delta_H):
            # 수치 말썽 - 물리치기
            return x, False
        
        if np.random.rand() < np.exp(-delta_H):
            return x_prop, True
        else:
            return x, False
    
    def sample(self, x0, n_samples, n_warmup=1000):
        """
        타깃 분포에서 표본 만들기.
        
        인수:
            x0: 첫 자리
            n_samples: 만들 표본의 개수
            n_warmup: 달굼 되풀이 횟수(버린다)
        
        반환값:
            samples: 꼴이 (n_samples, dim)인 배열
            accept_rate: 받아들임 비율
        """
        x = x0.copy()
        samples = np.zeros((n_samples, self.dim))
        n_accept = 0
        
        # 워밍업
        for _ in range(n_warmup):
            x, _ = self.step(x)
        
        # 표집
        for i in range(n_samples):
            x, accepted = self.step(x)
            samples[i] = x
            n_accept += accepted
        
        accept_rate = n_accept / n_samples
        return samples, accept_rate
```
```python
import numpy as np

# 타깃: 상관이 있는 2차원 가우스
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8], [0.8, 1]])
Sigma_inv = np.linalg.inv(Sigma)

def log_prob(x):
    return -0.5 * (x - mu) @ Sigma_inv @ (x - mu)

def grad_log_prob(x):
    return -Sigma_inv @ (x - mu)

# 표집기 만들기
hmc = HMC(log_prob, grad_log_prob, dim=2, epsilon=0.1, L=20)

# 표본 만들기
x0 = np.zeros(2)
samples, accept_rate = hmc.sample(x0, n_samples=10000, n_warmup=1000)

print(f"Acceptance rate: {accept_rate:.2%}")
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"Sample cov:\n{np.cov(samples.T)}")
```

## 5. 실용적인 고려

질량과 스텝 크기·적분시간은 서로 영향을 주므로 같은 관심량의 비용·오차를 기준으로 조정한다. 이 관례에서는 가우스 목표의 M≈Σ⁻¹가 주파수를 맞추며, 적응 단계의 공분산 추정과 정규화가 필요하다.

아래 비유한 에너지 거절·경사 clipping 조각은 클래스에 연결된 기능이 아니다. 위치에만 의존하는 잘린 힘의 전단은 가역적·체적 보존일 수 있어 원래 에너지로 올바르게 보정하면 반드시 상세균형이 깨지는 것은 아니다. 하지만 심플렉틱·에너지 정확도와 성능을 잃을 수 있으므로 임의의 경사 수정이 무조건 안전하다는 뜻도 아니다.

여러 체인은 서로 독립적인 난수·초기조건으로 실행할 수 있지만 체인 내부 표본은 의존한다. 아래 sample_parallel의 중첩 함수는 multiprocessing 직렬화 방식에 따라 pickle 오류를 낼 수 있다. 단순 호출만으로 모든 환경에서 병렬 실행된다고 가정하지 않는다. 같은 체인들이 모두 놓친 영역은 체인 간 진단으로도 알기 어려울 수 있다.

```python
if np.isnan(H_prop) or np.isinf(H_prop):
    return x, False  # 물리치고 이어 감
```
```python
grad = grad_log_prob(x)
grad_norm = np.linalg.norm(grad)
if grad_norm > max_grad:
    grad = grad * max_grad / grad_norm
```
```python
def sample_parallel(hmc, x0_list, n_samples, n_chains=4):
    from multiprocessing import Pool
    
    def sample_chain(args):
        x0, seed = args
        np.random.seed(seed)
        return hmc.sample(x0, n_samples)
    
    with Pool(n_chains) as pool:
        results = pool.map(sample_chain, 
                          [(x0_list[i], i) for i in range(n_chains)])
    
    return results
```

## 6. 진단

수락률은 적분 설정의 진단이며 높은 값·낮은 값만으로 성능을 단정하지 않는다. ΔH는 매 반복 기록해 큰 값·비유한 상태와 위치를 살핀다. 평균이 반드시 정확히 0이어야 하거나 모든 값이 [−1,1]에 있어야 하는 것은 아니다.

아래 step_with_diagnostics는 앞 클래스에 없는 메서드이고 발산 누적 조각도 별도 추가가 필요하다. 경고가 조각에 등장한다고 현재 클래스가 이를 보고하는 것은 아니다.

여러 체인의 트레이스·랭크 그림, R-hat·함수별 ESS와 MCSE를 함께 사용한다. 정상 CLT 아래 ESS는 N/(1+2∑ρ_f)로 점근 분산과 연결되며 함수별 값이다. 음의 상관에서는 N보다 클 수도 있다. 수락률이나 고정 ESS 임계값은 수렴 증명이 아니다.

```python
delta_H_values = []
for i in range(n_samples):
    x, accepted, delta_H = hmc.step_with_diagnostics(x)
    delta_H_values.append(delta_H)

plt.hist(delta_H_values, bins=50)
plt.xlabel('ΔH')
plt.title('Energy Change Distribution')
```
```python
if delta_H > 1000:
    n_divergent += 1
    divergent_points.append(x_prop)
```

## 7. 변형과 확장

부분 운동량 갱신 p_new=a p_old+√(1−a²)η, η~N(0,M)는 |a|<1에서 정상 가우스 운동량을 보존한다. 이를 사용할 때는 운동량을 상태로 유지하고 거절·반전 처리를 포함한 전체 전이 설계를 확인한다.

현재 상태와 독립으로 여러 고정 길이의 올바른 HMC 커널을 섞으면 목표 불변성을 보존하면서 공진을 완화할 수 있다. 경로를 보고 임의로 정지하는 규칙과는 다르다. 창 기반 선택·NUTS는 후보 선택과 보정의 명세가 필요하다.

리만 HMC는 위치 의존 M(x)와 log det M(x)/2 항을 포함하며 일반화된 적분·미분 비용을 지불한다. 위치마다 달라지는 기하를 다룰 수 있지만 항상 더 효율적인 것은 아니다.

## 8. 다른 MCMC와 비교

랜덤 워크 MH는 경사 없이 유연한 제안을 사용하고, MALA는 한 단계의 경사 기반 비대칭 제안을 보정한다. HMC는 운동량을 유지한 다단계 적분을 보정하며, 깁스는 정확한 조건부 생성으로 해당 갱신의 수락확률이 1이다.

0.234·0.574·0.65 같은 값은 특정 점근 분석과 비용 목적에 연결된 참고값이다. 방법별 보편적 순위나 모든 문제의 최적 수락률로 읽지 않는다. 비교는 같은 목표·함수·총 평가비용과 독립 반복의 변동을 고려한다.

## 연습문제

1. 완전 운동량 재생성과 고정 Leapfrog 제안·반전·MH 보정이 각각 어떤 분포를 보존하는지 설명하라. 결합 목표는 π(x)N(p;0,M)이며 위치를 주변화하면 원래 목표가 남는다.

2. 거절 상태를 저장하지 않고 수락한 위치만 모으면 왜 다른 표본이 되는가? 위치별 체류시간을 없애 전이 체인의 정상 가중을 바꾸기 때문이다.

3. 이 코드의 워밍업이 질량·스텝 크기를 배우는가? 아니다. 같은 고정 step을 반복해 버린다. 설정 적응은 별도 구현이 필요하다.

4. 부분 운동량 갱신에서 왜 반전·거절 처리를 다시 확인해야 하는가? 다음 반복에 운동량이 남아 확장 상태의 전이 법칙에 영향을 주므로 위치만 남기는 완전 재생성 관례를 그대로 적용할 수 없기 때문이다.

5. 현재 진단·병렬 조각의 실행 범위를 정리하라. step_with_diagnostics는 클래스에 없고 중첩 작업 함수는 프로세스 직렬화에 제약이 있다. 이들은 완결된 기능이 아니라 설명용 조각이다.

## 정리하며

HMC의 불변성은 운동량 생성, 가역적 체적 보존 제안과 MH 보정을 함께 사용해 구성한다. 거절 상태 보존, 실제 적응·진단의 구현 여부와 함수별 수치 오차를 확인한다.

**참고 문헌**

1. Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). "Hybrid Monte Carlo." *Physics Letters B*.
2. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
3. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
4. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
