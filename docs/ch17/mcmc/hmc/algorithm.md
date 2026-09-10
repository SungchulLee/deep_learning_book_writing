# HMC 알고리즘
이 절에서는 해밀턴 몬테카를로 알고리즘 전체를 보인다. 곧 온전한 절차, 옳음의 증명, 구현의 세부, 실제로 쓸 때 살필 점을 다룬다.

---

## 1. 알고리즘 훑어보기

### HMC 조리법

HMC는 $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$에서 표집하려고 재료 셋을 어우른다:

1. **운동량 덧붙이기**: 도움 운동량 $\mathbf{v}$으로 상태 공간을 넓힌다
2. **해밀턴 움직임**: 정해진 자취로 멀리 있는 상태를 내놓는다
3. **메트로폴리스-헤이스팅스 바로잡기**: 받아들이거나 물리쳐 정확함을 보장한다

### 온전한 알고리즘

**들임**: 첫 자리 $\mathbf{x}^{(0)}$, 표본 개수 $N$, 걸음 크기 $\epsilon$, 자취 길이 $L$, 질량 행렬 $\mathbf{M}$

**내임**: $\pi(\mathbf{x})$에서 뽑은 표본 $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}\}$

```
for t = 1 to N:
    # 걸음 1: 운동량 표집
    v ~ Normal(0, M)
    
    # 걸음 2: 첫 해밀턴 함수 값 셈하기
    H_current = U(x⁽ᵗ⁻¹⁾) + ½ vᵀ M⁻¹ v
    
    # 걸음 3: 개구리뜀 적분
    x_prop, v_prop = Leapfrog(x⁽ᵗ⁻¹⁾, v, ε, L)
    
    # 걸음 4: 운동량 부호 뒤집기(되돌릴 수 있게)
    v_prop = -v_prop
    
    # 걸음 5: 제안의 해밀턴 함수 값 셈하기
    H_prop = U(x_prop) + ½ v_propᵀ M⁻¹ v_prop
    
    # 걸음 6: 메트로폴리스-헤이스팅스 받아들임
    α = min(1, exp(H_current - H_prop))
    u ~ Uniform(0, 1)
    
    if u < α:
        x⁽ᵗ⁾ = x_prop    # 받아들임
    else:
        x⁽ᵗ⁾ = x⁽ᵗ⁻¹⁾    # 물리침
```

---

## 2. 걸음마다 자세히

### 걸음 1: 운동량 다시 표집하기

주변 분포에서 운동량을 새로 뽑는다:

$$
\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})
$$

**되풀이마다 왜 다시 표집하나?**

- 확률 요소를 불어넣는다(그러지 않으면 HMC는 정해진 것이 된다)
- 에르고드성을 보장한다(다른 에너지 면을 살펴볼 수 있게 한다)
- 결합 분포 $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})$을 지킨다

**구현**:
```python
v = np.random.multivariate_normal(np.zeros(d), M)
# 또는 같은 말로:
v = L_M @ np.random.randn(d)  # 여기서 M = L_M @ L_M.T
```

### 걸음 2: 첫 해밀턴 함수 값

시작점에서 전체 에너지를 셈한다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v}) = -\log \tilde{\pi}(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

이것이 받아들임을 정하는 기준이 된다.

### 걸음 3: 개구리뜀 적분

걸음 크기 $\epsilon$으로 해밀턴 움직임을 $L$걸음 흉내 낸다:

$$
(\mathbf{x}', \mathbf{v}') = \text{Leapfrog}(\mathbf{x}, \mathbf{v}, \epsilon, L)
$$

자취는 에너지를 거의 지키면서 위상 공간을 살펴본다.

**셈 값**: 기울기 값매김 $L$번(가장 큰 값이다).

### 걸음 4: 운동량 부호 뒤집기

마지막 운동량의 부호를 뒤집는다:

$$
\mathbf{v}' \leftarrow -\mathbf{v}'
$$

**왜 뒤집나?**

이렇게 하면 제안이 제 자신의 역이 된다. 곧 $(\mathbf{x}', -\mathbf{v}')$에서 시작해 개구리뜀을 돌리면 $(\mathbf{x}, -\mathbf{v})$으로 돌아온다.

**메모**: MH 비에서는 부호를 뒤집어도 운동 에너지가 그대로이므로($K(\mathbf{v}) = K(-\mathbf{v})$) 구현에서는 흔히 뺀다. 그러나 개념으로는 자세한 균형을 보장하는 한 부분이다.

### 걸음 5: 제안의 해밀턴 함수 값

제안에서 에너지를 셈한다:

$$
H(\mathbf{x}', \mathbf{v}') = U(\mathbf{x}') + K(\mathbf{v}')
$$

에너지 변화 $\Delta H = H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}, \mathbf{v})$이 받아들임을 정한다.

### 걸음 6: 메트로폴리스-헤이스팅스 받아들임

다음 확률로 받아들인다:

$$
\alpha = \min\left(1, \exp(-\Delta H)\right) = \min\left(1, \frac{\pi(\mathbf{x}', \mathbf{v}')}{\pi(\mathbf{x}, \mathbf{v})}\right)
$$

- $\Delta H \leq 0$이면(에너지가 줄었으면): 확률 1로 받아들인다
- $\Delta H > 0$이면(에너지가 늘었으면): 확률 $\exp(-\Delta H)$으로 받아들인다

**구현**:
```python
delta_H = H_prop - H_current
if np.random.rand() < np.exp(-delta_H):
    x = x_prop  # 받아들임
# 아니면 지금의 x 그대로 두기(물리침)
```

---

## 3. 옳음의 증명

### 무엇을 보여야 하나

옮김 알맹이 $K(\mathbf{x}' | \mathbf{x})$이 **자세한 균형**을 만족하면 HMC는 옳은 MCMC 알고리즘이다:

$$
\pi(\mathbf{x}) K(\mathbf{x}' | \mathbf{x}) = \pi(\mathbf{x}') K(\mathbf{x} | \mathbf{x}')
$$

### 옮김 알맹이

$\mathbf{x}$에서 $\mathbf{x}'$으로 가는 HMC의 옮김에는 다음이 들어 있다:

1. $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$을 표집한다
2. 개구리뜀과 부호 뒤집기로 정해진 사상 $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}', \mathbf{v}')$을 쓴다
3. 확률 $\alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')$으로 받아들인다

### 주요 성질

**성질 1: 대합**

(개구리뜀과 부호 뒤집기인) 사상 $T: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}', \mathbf{v}')$은 **대합**이다. 곧 $T \circ T = \text{identity}$이다.

*증명*: $(\mathbf{x}', \mathbf{v}')$에서 시작해 개구리뜀을 쓰면 (시간을 되돌릴 수 있으므로) $(\mathbf{x}, -\mathbf{v})$을 얻고, 부호를 뒤집으면 $(\mathbf{x}, \mathbf{v})$을 얻는다.

**성질 2: 부피 지킴**

개구리뜀 사상의 야코비 행렬식은 1이다(심플렉틱이다). 운동량 부호 뒤집기도 행렬식이 1이다(사실은 $(-1)^d$이지만 절댓값이 1이다).

**성질 3: 에너지 대칭**

운동 에너지가 $\mathbf{v}$에 대해 짝함수이므로 $H(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, -\mathbf{v})$이다.

### 자세한 균형의 증명

$(\mathbf{x}, \mathbf{v})$에서 $(\mathbf{x}', \mathbf{v}')$으로 가는 확률 흐름을 보자:

$$
\text{Flow}[(\mathbf{x}, \mathbf{v}) \to (\mathbf{x}', \mathbf{v}')] = \pi(\mathbf{x}, \mathbf{v}) \cdot \delta(T(\mathbf{x}, \mathbf{v}) - (\mathbf{x}', \mathbf{v}')) \cdot \alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')
$$

$(\mathbf{x}', \mathbf{v}')$에서 $(\mathbf{x}, \mathbf{v})$으로 가는 거꾸로 된 흐름은:

$$
\text{Flow}[(\mathbf{x}', \mathbf{v}') \to (\mathbf{x}, \mathbf{v})] = \pi(\mathbf{x}', \mathbf{v}') \cdot \delta(T(\mathbf{x}', \mathbf{v}') - (\mathbf{x}, \mathbf{v})) \cdot \alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})
$$

대합 성질에 따라 $T(\mathbf{x}', \mathbf{v}') = (\mathbf{x}, \mathbf{v})$이므로 델타 함수가 서로 맞는다.

MH 받아들임 비는 다음을 보장한다:

$$
\frac{\alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}')}{\alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})} = \frac{\pi(\mathbf{x}', \mathbf{v}')}{\pi(\mathbf{x}, \mathbf{v})}
$$

그러므로 흐름이 균형을 이룬다:

$$
\pi(\mathbf{x}, \mathbf{v}) \cdot \alpha(\mathbf{x}, \mathbf{v}, \mathbf{x}', \mathbf{v}') = \pi(\mathbf{x}', \mathbf{v}') \cdot \alpha(\mathbf{x}', \mathbf{v}', \mathbf{x}, \mathbf{v})
$$

(되풀이마다 다시 표집하는) 운동량을 적분해 없애면 $\mathbf{x}$의 주변 분포에 대한 자세한 균형을 얻는다.

---

## 4. 구현

### 온전한 파이썬 구현

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
            dim: 과녁 분포의 차원
            epsilon: 개구리뜀 걸음 크기
            L: 개구리뜀 걸음 수
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
        """개구리뜀 L걸음 밟기."""
        x = x.copy()
        v = v.copy()
        
        # 운동량의 반 걸음
        v = v + (self.epsilon / 2) * self.grad_log_prob(x)
        
        # 온 걸음
        for _ in range(self.L - 1):
            x = x + self.epsilon * self.M_inv @ v
            v = v + self.epsilon * self.grad_log_prob(x)
        
        # 마지막 자리 걸음
        x = x + self.epsilon * self.M_inv @ v
        
        # 운동량의 마지막 반 걸음
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
        과녁 분포에서 표본 만들기.
        
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

### 쓰는 보기

```python
# 과녁: 상관이 있는 2차원 가우스
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

---

## 5. 실용적인 고려

### 맞출 매개변수

| 매개변수 | 효과 | 맞추는 방법 |
|-----------|--------|-----------------|
| $\epsilon$(걸음 크기) | 받아들임 비율, 안정함 | 받아들임 65-80%을 목표로 맞춘다 |
| $L$(자취 길이) | 살펴보는 거리 | 얽힘이 풀릴 만큼 길게, NUTS이 저절로 해 준다 |
| $\mathbf{M}$(질량 행렬) | 기하에 맞춰 감 | 달굼 표본에서 어림한다 |

### 달굼 단계

달굼 단계는 여러 일을 한다:

1. **태우기**: 첫 자리에서 전형 집합으로 옮겨 간다
2. **맞춰 가기**: $\epsilon$, $L$, $\mathbf{M}$의 좋은 값을 배운다
3. **진단**: 말썽을 일찍 알아낸다

**흔한 달굼 전략**:

- 1단계: $\epsilon$을 빠르게 맞춘다
- 2단계: 표본 공분산에서 $\mathbf{M}$을 어림한다
- 3단계: $\epsilon$을 마지막으로 맞춘다

### 수치 말썽 다루기

**NaN과 Inf 알아내기**:
```python
if np.isnan(H_prop) or np.isinf(H_prop):
    return x, False  # 물리치고 이어 감
```

**갈라져 나가는 옮김**: $\Delta H$이 엄청나게 크면(이를테면 1000을 넘으면) 자취가 말썽 있는 구역에 들어갔을 가능성이 높다. 진단을 위해 기록해 두어라.

**기울기 잘라 내기**(조심해서 써라):
```python
grad = grad_log_prob(x)
grad_norm = np.linalg.norm(grad)
if grad_norm > max_grad:
    grad = grad * max_grad / grad_norm
```

### 여러 사슬

사슬 여럿을 나란히 돌리면:

- 모임 진단(R-hat)을 할 수 있다
- 독립인 표본을 얻는다
- 여러 코어의 하드웨어를 써먹는다

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

---

## 6. 진단

### 받아들임 비율

**목표**: 표준 HMC에서는 65-80%, NUTS에서는 80%쯤.

**너무 낮으면**(50% 미만): 걸음 크기가 너무 크고 에너지 오차가 너무 크다.

**너무 높으면**(95% 초과): 걸음 크기가 너무 작아 살펴보기가 효율이 낮다.

### 에너지 변화의 분포

되풀이에 걸친 $\Delta H$의 분포를 그려라:

- 가운데가 0 가까이에 있어야 한다
- 보통 범위: [-1, 1]
- 큰 벗어난 값은 말썽을 뜻한다

```python
delta_H_values = []
for i in range(n_samples):
    x, accepted, delta_H = hmc.step_with_diagnostics(x)
    delta_H_values.append(delta_H)

plt.hist(delta_H_values, bins=50)
plt.xlabel('ΔH')
plt.title('Energy Change Distribution')
```

### 갈라져 나가는 옮김

개구리뜀 자취가 수치로 흔들릴 때 **갈라져 나가는 옮김**이 일어나며, 보통 에너지 변화가 아주 큰 것으로 드러난다.

**알아내기**:
```python
if delta_H > 1000:
    n_divergent += 1
    divergent_points.append(x_prop)
```

**진단**: 갈라져 나감은 흔히 다음 가까이에서 일어난다:

- 제약이 있는 매개변수의 경계
- 굽음이 큰 구역
- 봉우리가 여럿인 짜임

### 실효 표본 크기

ESS은 독립 표본으로 치면 몇 개에 해당하는지를 잰다:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

여기서 $\rho_k$은 뒤짐 $k$에서의 자기상관이다.

**목표**: 초당 ESS을 가장 크게 해야 한다.

---

## 7. 변형과 확장

### 운동량 일부만 새로 하기

운동량을 통째로 다시 표집하는 대신 일부만 새로 한다:

$$
\mathbf{v}_{\text{new}} = \alpha \mathbf{v}_{\text{old}} + \sqrt{1 - \alpha^2} \boldsymbol{\eta}, \quad \boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})
$$

이러면 앞선 방향이 얼마쯤 남으며, 어떤 과녁에서는 도움이 된다.

### 무작위 자취 길이

$L$을 붙박아 두는 대신 $L \sim \text{Uniform}(1, L_{\max})$ 따위로 표집한다.

**이점**:

- 섞임을 나쁘게 하는 되풀이 굶을 피한다
- $L$을 잘못 골라도 더 튼튼하다

### 창을 쓴 HMC

알맞은 받아들임 확률을 써서 끝점만이 아니라 자취 위의 아무 점이나 받아들인다.

### 리만 HMC

그 자리 기하에 맞춰 가는, 자리에 달린 질량 행렬 $\mathbf{M}(\mathbf{x})$을 쓴다. 더 복잡하지만 조건이 나쁜 과녁에서 표집을 크게 낫게 할 수 있다.

---

## 8. 다른 MCMC와 견주기

| 방법 | 기울기 | 제안 | 받아들임 | 섞임 |
|--------|----------|-----------|------------|--------|
| 무작위 걸음 MH | 아니오 | 가까이 | 최적 23%쯤 | 느림(퍼짐꼴) |
| MALA | 예 | 가까이 | 최적 57%쯤 | 보통 |
| HMC | 예 | 멀리 | 65-80%쯤 | 빠름(탄도꼴) |
| 깁스 | 아니오 | 온전한 조건부 | 100% | 그때그때 다름 |

**HMC의 좋은 점**:

- 차원이 높아도 효율적으로 살펴본다
- 기울기 정보를 써먹는다
- 받아들임을 높게 지키면서도 큰 걸음을 뗄 수 있다

**HMC의 한계**:

- 미분할 수 있는 과녁이 필요하다
- 맞추기가 까다로울 수 있다
- 봉우리가 여럿이면 힘겨워한다

---

## 연습문제

1. **구현 확인하기**. HMC를 구현하고 알려진 분포(이를테면 2차원 가우스)에서 올바로 표집하는지 확인하여라. 표본 통계량을 참값과 견주어라.

2. **맞추기 실험**. 10차원 가우스에 대해 받아들임 비율과 초당 ESS을 $\epsilon$과 $L$의 함수로 그려라. 가장 좋은 설정을 찾아라.

3. **MALA과 견주기**. MALA을 구현하고 같은 과녁에서 HMC와 견주어라. 기울기 값매김당 ESS은 어떻게 다른가?

4. **갈라져 나감 파헤치기**. 깔때기 기하를 갖는 과녁(이를테면 닐의 깔때기)을 만들어라. HMC를 돌려 갈라져 나감이 어디서 일어나는지 찾아라.

5. **일부만 새로 하기**. 운동량을 일부만 새로 하는 방식을 구현하고 고른 과녁에서 통째로 새로 하는 방식과 견주어라.

---

## 정리하며

| 부품 | 하는 일 |
|-----------|---------|
| 운동량 다시 표집하기 | 확률 요소를 불어넣고 에르고드성을 보장한다 |
| 개구리뜀 적분 | 멀리 있는 상태를 효율적으로 내놓는다 |
| 운동량 부호 뒤집기 | 대합 성질을 보장한다 |
| MH 받아들임 | 띄엄띄엄 나눈 데서 오는 오차를 바로잡는다 |

HMC 알고리즘은 다음으로 효율적인 표집을 이룬다:

1. 기울기 정보를 써서 아는 바에 바탕을 둔 제안을 한다
2. 자취를 길게 잡아 멀리 있는 것을 내놓는다
3. 에너지 지킴으로 받아들임을 높게 지킨다
4. MH 바로잡기로 정확함을 지킨다

---

**참고 문헌**

1. Duane, S., Kennedy, A. D., Pendleton, B. J., & Roweth, D. (1987). "Hybrid Monte Carlo." *Physics Letters B*.
2. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
3. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
4. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
