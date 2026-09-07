# 유턴 없는 표집기(NUTS)
유턴 없는 표집기(NUTS)는 자취 길이 $L$을 저절로 맞추는, 맞춰 가는 HMC 넓힘이다. 자취가 "되돌아서기" 시작하는 때를 알아내어, NUTS은 가장 맞추기 어려운 매개변수 가운데 하나를 없애면서도 표집 효율을 지키고 흔히 더 낫게 한다.

---

## 왜 필요한가

### 자취 길이 문제

표준 HMC는 개구리뜀 걸음 수 $L$을 정해 주어야 한다. 이 고름은 결정적이다:

| $L$이 너무 작음 | $L$이 너무 큼 |
|---------------|---------------|
| 무작위 걸음처럼 굶 | 셈을 버림 |
| 높은 자기상관 | 자취가 되짚어 옴 |
| 나쁜 살펴보기 | 더 얻는 것이 없음 |

**진퇴양난**: 최적 $L$은 과녁 분포에 달렸고 매개변수 공간에 따라 달라지며 미리 정하기 어렵다.

### 유턴 직관

1차원 조화 떨개를 보자. 그 자취는 되풀이된다:

- $x_0$에서 떠나 알갱이가 멀어진다
- 가장 먼 곳에 이른 뒤 되돌아온다
- 마침내 $x_0$ 쪽으로 다시 온다

**핵심 통찰**: 자취가 출발점 쪽으로 돌아오기 시작하면 더 나아가도 얻을 것이 없다. "유턴"은 멈추라는 신호이다.

### NUTS의 풀이

NUTS은 다음처럼 자취 길이를 저절로 정한다:

1. 자취를 조금씩 키운다(곱절로 늘림)
2. 되돌아서기 시작하는 때를 알아낸다
3. 쓸 수 있는 자취에서 점 하나를 고른다

---

## 유턴 잣대

### 기본 정의

더 나아가면 시작점에 더 가까워질 때 자취가 **유턴**한다고 한다. 자취의 양 끝에 있는 자리 $\mathbf{x}^-$, $\mathbf{x}^+$과 운동량 $\mathbf{v}^-$, $\mathbf{v}^+$에 대해:

**유턴 조건**:

$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0 \quad \text{or} \quad (\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^- < 0
$$

**풀이**:

- $(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0$: 앞쪽 끝이 뒤로 움직이고 있다
- $(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^- < 0$: 뒤쪽 끝이 앞으로 움직이고 있다

둘 중 어느 조건이든 자취가 "되돌아서고" 있음을 뜻한다.

### 넓힌 잣대

질량 행렬 $\mathbf{M}$을 쓰는 자취에서는 잣대가 다음이 된다:

$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{M}^{-1}\mathbf{v}^+ < 0 \quad \text{or} \quad (\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{M}^{-1}\mathbf{v}^- < 0
$$

여기서는 운동량 $\mathbf{v}$ 대신 속도 $\mathbf{M}^{-1}\mathbf{v}$을 쓴다.

### 이것이 왜 되나

이차 퍼텐셜(가우스 과녁)에서 자취는 타원이다. 유턴 잣대는 자취가 한 바퀴의 반쯤을 돌았을 때, 곧 출발점에서 가장 멀어진 점을 알아낸다.

일반 과녁에서 이 잣대는 "넉넉히 살펴봄"에 대한 튼튼한 어림짐작이 된다.

---

## 나무 세우기 알고리즘

### 곱절 늘리기 얼개

NUTS은 자취를 **이진 나무**로 세운다:

1. **첫값 잡기**: 점 하나로 시작한다(깊이 0)
2. **곱절 늘리기**: 앞이나 뒤로 무작위로 넓힌다
3. **살피기**: 유턴이 잡히거나 다른 멈춤 조건이 되면 멈춘다
4. **되풀이하기**: 멈출 때까지 곱절 늘리기를 이어 간다

```
Depth 0:        [x₀]                    (1 point)
Depth 1:    [x₋₁, x₀, x₁]              (2 points added)
Depth 2: [x₋₃..x₋₁, x₀, x₁..x₃]        (4 points added)
   ...
Depth j: 2ʲ points total
```

### 왜 곱절인가?

**좋은 점**:

- 자취 길이가 지수로 자란다. 곧 깊이 $j$에서 걸음이 $2^j$개다
- 시간을 되돌릴 수 있음을 지킨다(대칭으로 넓힘)
- 효율적이다. 곧 자취 길이 $L$에 대해 깊이가 $O(\log L)$이다

**되돌릴 수 있음**: 곱절 늘릴 때마다 앞뒤를 무작위로 고르고 대칭인 나무 짜임을 쓰므로 NUTS은 자세한 균형을 지킨다.

### 나무 짜임

나무의 마디마다 다음을 담은 **부분 나무**를 나타낸다:

- 왼쪽 끝의 자리와 운동량: $(\mathbf{x}^-, \mathbf{v}^-)$
- 오른쪽 끝의 자리와 운동량: $(\mathbf{x}^+, \mathbf{v}^+)$
- 부분 나무에서 뽑은 후보 표본
- 부분 나무 안에서 쓸 수 있는 상태의 개수
- 멈춤 깃발

---

## NUTS 알고리즘

### 큰 틀의 가짜 코드

```
function NUTS(x₀, ε, M):
    # 운동량 표집
    v₀ ~ Normal(0, M)
    
    # 나무 첫값 잡기
    x⁻ = x⁺ = x₀
    v⁻ = v⁺ = v₀
    j = 0  # 나무 깊이
    n = 1  # 받아들일 수 있는 상태의 수
    s = 1  # 이어 감 깃발
    
    # 첫 후보
    x_sample = x₀
    
    while s == 1:
        # 방향을 고르게 고르기
        direction ~ Uniform({-1, +1})
        
        # 고른 방향으로 나무 세우기
        if direction == -1:
            x⁻, v⁻, _, _, x', n', s' = BuildTree(x⁻, v⁻, -1, j, ε)
        else:
            _, _, x⁺, v⁺, x', n', s' = BuildTree(x⁺, v⁺, +1, j, ε)
        
        # 새 표본을 받아들일 수도 있다
        if s' == 1 and Random() < n'/n:
            x_sample = x'
        
        # 개수를 새로 고치고 유턴 살피기
        n = n + n'
        s = s' AND NoUTurn(x⁻, x⁺, v⁻, v⁺)
        
        j = j + 1
    
    return x_sample
```

### BuildTree 함수

```
function BuildTree(x, v, direction, depth, ε):
    if depth == 0:
        # 바닥 경우: 개구리뜀 한 걸음
        x', v' = Leapfrog(x, v, direction * ε)
        
        # 에너지 살피기
        valid = (H(x', v') - H(x₀, v₀) < Δmax)
        
        return x', v', x', v', x', valid, valid
    else:
        # 되돌이: 왼쪽 부분 나무 세우기
        x⁻, v⁻, x⁺, v⁺, x', n', s' = BuildTree(x, v, direction, depth-1, ε)
        
        if s' == 1:
            # 오른쪽 부분 나무 세우기
            if direction == -1:
                x⁻, v⁻, _, _, x'', n'', s'' = BuildTree(x⁻, v⁻, direction, depth-1, ε)
            else:
                _, _, x⁺, v⁺, x'', n'', s'' = BuildTree(x⁺, v⁺, direction, depth-1, ε)
            
            # 표본을 새로 고칠 수도 있다
            if Random() < n''/(n' + n''):
                x' = x''
            
            # 부분 나무 안의 유턴 살피기
            s' = s'' AND NoUTurn(x⁻, x⁺, v⁻, v⁺)
            n' = n' + n''
        
        return x⁻, v⁻, x⁺, v⁺, x', n', s'
```

### 다항 표집

NUTS은 마지막 점만 돌려주지 않고 자취 위의 쓸 수 있는 점 모두에서 고르게 표집한다. 이는 다음으로 이룬다:

```
if Random() < n''/(n' + n''):
    x' = x''
```

이 "흘려보내며" 고르기는 점을 모두 저장하지 않고도 고른 표집을 보장한다.

---

## 멈춤 잣대

### 유턴으로 멈추기

으뜸가는 멈춤 조건은 자취가 유턴할 때 멈추는 것이다.

```python
def no_uturn(x_minus, x_plus, v_minus, v_plus, M_inv):
    delta_x = x_plus - x_minus
    return (np.dot(delta_x, M_inv @ v_plus) >= 0 and 
            np.dot(delta_x, M_inv @ v_minus) >= 0)
```

### 에너지로 멈추기

에너지 오차가 너무 커지면 멈춘다(수치가 흔들림):

$$
H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}_0, \mathbf{v}_0) > \Delta_{\max}
$$

보통 값: $\Delta_{\max} = 1000$.

### 최대 나무 깊이

최대 깊이를 두어 끝없는 되풀이를 막는다:

$$
j \leq j_{\max}
$$

보통 값: $j_{\max} = 10$(개구리뜀 걸음 최대 $2^{10} = 1024$개).

### 갈라져 나감 알아내기

자취가 수치 말썽을 만나면 **갈라져 나가는 옮김**이 일어난다:

- 에너지가 아주 크게 바뀜
- NaN이나 Inf 값
- 뒤확률의 기하에 말썽이 있음을 뜻함

```python
def is_divergent(H_new, H_old, delta_max=1000):
    return H_new - H_old > delta_max or np.isnan(H_new)
```

---

## 자세한 균형

### NUTS이 왜 옳은가

NUTS은 짜임을 꼼꼼히 만들어 자세한 균형을 지킨다:

1. **대칭으로 나무 세우기**: 곱절 늘릴 때마다 방향을 무작위로 고른다
2. **다항 표집**: 쓸 수 있는 상태에서 고르게 고른다
3. **한결같은 멈춤**: 유턴을 대칭으로 살핀다

### 조각 표집으로 보기

NUTS은 자취 공간에서의 **조각 표집**으로 볼 수 있다:

1. 받아들일 수 있는 상태의 "조각"을 정한다. 곧 $u \sim \text{Exp}(1)$일 때 $\{(\mathbf{x}, \mathbf{v}) : H(\mathbf{x}, \mathbf{v}) < H_0 + u\}$이다
2. 자취가 조각을 벗어나거나 유턴할 때까지 세운다
3. 조각 안의 자취에서 고르게 표집한다

이렇게 보면 다항 표집이 왜 옳은지가 뚜렷해진다.

### 받아들임 확률

표준 HMC와 달리 NUTS에는 드러난 MH 받아들임/물리침 걸음이 없다. 그 대신:

- 쓸 수 없는 상태(에너지가 높은 상태)는 고름에서 뺀다
- 쓸 수 있는 상태에서 고르게 표집한다
- 조각 표집 얼개가 옳음을 보장한다

NUTS에서 실효 "받아들임 비율"은 자취 가운데 쓸 수 있는 몫을 가리킨다.

---

## 구현

### 온전한 NUTS 구현

```python
import numpy as np

class NUTS:
    def __init__(self, log_prob, grad_log_prob, dim,
                 epsilon=0.1, M=None, max_depth=10, delta_max=1000):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.delta_max = delta_max
        
        if M is None:
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)
            self.L_M = np.eye(dim)
        else:
            self.M = M
            self.M_inv = np.linalg.inv(M)
            self.L_M = np.linalg.cholesky(M)
    
    def hamiltonian(self, x, v):
        U = -self.log_prob(x)
        K = 0.5 * v @ self.M_inv @ v
        return U + K
    
    def leapfrog(self, x, v, direction):
        eps = direction * self.epsilon
        x, v = x.copy(), v.copy()
        
        v = v + 0.5 * eps * self.grad_log_prob(x)
        x = x + eps * self.M_inv @ v
        v = v + 0.5 * eps * self.grad_log_prob(x)
        
        return x, v
    
    def check_uturn(self, x_minus, x_plus, v_minus, v_plus):
        delta_x = x_plus - x_minus
        return (np.dot(delta_x, self.M_inv @ v_plus) >= 0 and
                np.dot(delta_x, self.M_inv @ v_minus) >= 0)
    
    def build_tree(self, x, v, direction, depth, H0):
        if depth == 0:
            # 바닥 경우: 개구리뜀 한 걸음
            x_new, v_new = self.leapfrog(x, v, direction)
            H_new = self.hamiltonian(x_new, v_new)
            
            # 유효성을 확인한다
            valid = (H_new - H0) < self.delta_max
            divergent = (H_new - H0) > self.delta_max
            
            # 다항 표집을 위한 무게
            log_weight = -H_new if valid else -np.inf
            
            return (x_new, v_new, x_new, v_new, x_new, 
                    log_weight, valid, divergent, 1)
        else:
            # 되돌이
            (x_minus, v_minus, x_plus, v_plus, x_prime,
             log_weight, valid, divergent, n_steps) = \
                self.build_tree(x, v, direction, depth - 1, H0)
            
            if valid:
                if direction == -1:
                    (x_minus, v_minus, _, _, x_prime2,
                     log_weight2, valid2, divergent2, n_steps2) = \
                        self.build_tree(x_minus, v_minus, direction, 
                                       depth - 1, H0)
                else:
                    (_, _, x_plus, v_plus, x_prime2,
                     log_weight2, valid2, divergent2, n_steps2) = \
                        self.build_tree(x_plus, v_plus, direction,
                                       depth - 1, H0)
                
                # 다항 표집
                log_weight_sum = np.logaddexp(log_weight, log_weight2)
                if np.log(np.random.rand()) < log_weight2 - log_weight_sum:
                    x_prime = x_prime2
                
                # 갱신
                log_weight = log_weight_sum
                valid = valid2 and self.check_uturn(x_minus, x_plus, 
                                                     v_minus, v_plus)
                divergent = divergent or divergent2
                n_steps = n_steps + n_steps2
            
            return (x_minus, v_minus, x_plus, v_plus, x_prime,
                    log_weight, valid, divergent, n_steps)
    
    def step(self, x):
        # 운동량 표집
        v = self.L_M @ np.random.randn(self.dim)
        H0 = self.hamiltonian(x, v)
        
        # 나무 첫값 잡기
        x_minus = x_plus = x
        v_minus = v_plus = v
        depth = 0
        valid = True
        x_sample = x
        log_weight = -H0
        n_divergent = 0
        
        while valid and depth < self.max_depth:
            # 방향 고르기
            direction = 2 * (np.random.rand() < 0.5) - 1
            
            # 나무 세우기
            if direction == -1:
                (x_minus, v_minus, _, _, x_prime,
                 log_weight_subtree, valid_subtree, divergent, _) = \
                    self.build_tree(x_minus, v_minus, direction, depth, H0)
            else:
                (_, _, x_plus, v_plus, x_prime,
                 log_weight_subtree, valid_subtree, divergent, _) = \
                    self.build_tree(x_plus, v_plus, direction, depth, H0)
            
            n_divergent += divergent
            
            # 새 표본을 받아들일 수도 있다
            if valid_subtree:
                if np.log(np.random.rand()) < log_weight_subtree - log_weight:
                    x_sample = x_prime
                log_weight = np.logaddexp(log_weight, log_weight_subtree)
            
            # 전체 유턴 살피기
            valid = valid_subtree and self.check_uturn(x_minus, x_plus,
                                                        v_minus, v_plus)
            depth += 1
        
        return x_sample, depth, n_divergent > 0
    
    def sample(self, x0, n_samples, n_warmup=1000):
        x = x0.copy()
        samples = np.zeros((n_samples, self.dim))
        depths = []
        n_divergent = 0
        
        # 달굼(여기에 맞춰 가기를 넣을 수도 있다)
        for _ in range(n_warmup):
            x, _, _ = self.step(x)
        
        # 표집
        for i in range(n_samples):
            x, depth, divergent = self.step(x)
            samples[i] = x
            depths.append(depth)
            n_divergent += divergent
        
        return samples, np.mean(depths), n_divergent / n_samples
```

### 쓰는 보기

```python
# 서로 얽힌 2차원 가우스
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8], [0.8, 1]])
Sigma_inv = np.linalg.inv(Sigma)

def log_prob(x):
    return -0.5 * (x - mu) @ Sigma_inv @ (x - mu)

def grad_log_prob(x):
    return -Sigma_inv @ (x - mu)

# 뽑기
nuts = NUTS(log_prob, grad_log_prob, dim=2, epsilon=0.1)
samples, avg_depth, div_rate = nuts.sample(np.zeros(2), 5000)

print(f"Average tree depth: {avg_depth:.1f}")
print(f"Divergence rate: {div_rate:.2%}")
print(f"Sample mean: {samples.mean(0)}")
```

---

## 진단

### 나무 깊이

나무 깊이는 자취 길이를 나타낸다:

| 깊이 | 개구리뜀 걸음 | 풀이 |
|-------|----------------|----------------|
| 1-3 | 2-8 | 짧은 자취, 효율이 낮을 수 있음 |
| 4-6 | 16-64 | 보통 범위 |
| 7-10 | 128-1024 | 긴 자취, 말썽이 있을 수 있음 |
| 10(최대) | 1024 | 최대에 부딪힘, 늘리는 것을 생각해 보라 |

**평균 깊이**: 알맞아야 한다(보통 4-8). 아주 낮으면 걸음 크기가 너무 크다는 뜻이고, 아주 높으면 기하가 어렵다는 뜻이다.

### 갈라져 나감

갈라져 나가는 옮김은 심각한 말썽을 뜻한다:

- **갈라져 나감이 적음**(1% 미만): 대개 받아들일 만하다, 매개변수 공간을 살펴라
- **갈라져 나감이 많음**(5% 초과): 말썽이다, 손을 봐야 한다
- **어디서 일어나나?**: 갈라져 나간 점을 그려 진단하라

**흔한 까닭**:

- 걸음 크기가 너무 큼
- 깔때기 꼴 뒤확률
- 경계나 제약
- 규모가 여럿인 기하

### 빠진 정보의 에너지 베이즈 몫(E-BFMI)

BFMI은 운동량을 다시 표집하는 것이 에너지 층을 얼마나 잘 살펴보는지를 잰다:

$$
\text{E-BFMI} = \frac{\mathbb{E}[(E_n - E_{n-1})^2]}{\text{Var}(E_n)}
$$

여기서 $E_n = H(\mathbf{x}^{(n)}, \mathbf{v}^{(n)})$이다.

- **E-BFMI > 0.3**: 좋다
- **E-BFMI < 0.2**: 말썽이다(에너지를 잘 살펴보지 못한다)

---

## NUTS과 표준 HMC의 견줌

### NUTS의 좋은 점

1. **$L$을 맞출 필요 없음**: 자취 길이가 저절로 정해진다
2. **맞춰 감**: 그 자리 기하에 맞춘다
3. **튼튼함**: 규모가 달라져도 다룬다
4. **효율적임**: 흔히 $L$이 붙박인 HMC보다 효율이 높다

### 표준 HMC가 더 나을 수 있을 때

1. **최적 $L$을 알 때**: 알맞은 자취 길이를 이미 안다면
2. **셈 예산**: NUTS의 덧짐이 문제될 수 있다
3. **아주 높은 차원**: 나무 세우기에 덧짐이 있다
4. **단순한 과녁**: 맞춰 가는 복잡함이 필요 없을 수 있다

### 실험으로 견주기

대부분의 실전 문제에서 NUTS은 잘 맞춘 HMC와 같거나 그보다 낫다:

| 재는 잣대 | 표준 HMC | NUTS |
|--------|--------------|------|
| 맞추는 수고 | 큼($\epsilon$, $L$, $\mathbf{M}$) | 적음($\epsilon$, $\mathbf{M}$) |
| 기울기당 ESS | 맞추기에 달림 | 한결같이 좋음 |
| 튼튼함 | $L$에 예민함 | 튼튼함 |
| 구현 | 더 단순함 | 더 복잡함 |

---

## 실전 권고

### 걸음 크기 맞춰 가기

NUTS도 걸음 크기 $\epsilon$은 맞춰야 한다. 쌍대 평균내기를 써라:

```python
def adapt_step_size(epsilon, accept_stat, target=0.8, 
                    gamma=0.05, t0=10, kappa=0.75, iteration=None):
    # 걸음 크기 맞추기를 위한 쌍대 평균내기
    # (간추린 판)
    if accept_stat > target:
        return epsilon * 1.02
    else:
        return epsilon * 0.98
```

NUTS의 목표 받아들임 통계량: 0.8쯤(HMC의 0.65쯤보다 높다).

### 최대 나무 깊이

기본값 $j_{\max} = 10$이면 대개 넉넉하다. 다음이면 늘려라:

- 평균 깊이가 자주 최대에 부딪힌다
- 과녁의 규모가 아주 다르다
- 뒤확률에 멀리까지 미치는 상관이 있다

### NUTS에서의 질량 행렬

HMC와 같은 원리이다:

- 달굼 동안 맞춰 간다
- 대각만으로도 대개 넉넉하다
- 상관이 강하면 온전한 행렬을 쓴다

### 달굼 일정(Stan 방식)

1. **되풀이 1-75**: 걸음 크기를 빠르게 맞춘다
2. **되풀이 76-975**: 걸음 크기와 질량 행렬을 맞춘다
3. **되풀이 976-1000**: 걸음 크기를 마지막으로 맞춘다
4. **되풀이 1001 이상**: 표집한다(맞추지 않는다)

---

## 요약

| 부품 | 설명 |
|-----------|-------------|
| **유턴 잣대** | 자취가 방향을 뒤집으면 멈춘다 |
| **나무 곱절 늘리기** | 자취를 지수로 키운다 |
| **다항 표집** | 쓸 수 있는 상태에서 고르게 고른다 |
| **멈춤** | 유턴, 에너지 오차, 또는 최대 깊이 |

NUTS은 꼼꼼히 맞춰야 하던 HMC를 튼튼하고 거의 저절로 굴러가는 표집기로 바꾸었다. 자취 길이 매개변수를 없앰으로써 NUTS은 Stan이나 PyMC 같은 요즘 확률 프로그래밍 체계의 기본 알고리즘이 되었다.

---

## 참고 문헌

1. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *JMLR*, 15, 1593-1623.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Stan Development Team. "Stan Reference Manual."
4. Betancourt, M. (2016). "Diagnosing Biased Inference with Divergences." Stan Case Studies.

## 연습문제

1. **기본 NUTS 구현하기**. 2차원 가우스에 대해 NUTS을 구현하고 올바른 표본이 나오는지 확인하여라. 과녁의 기하에 따라 나무 깊이를 견주어라.

2. **유턴 그려 보기**. 2차원 과녁에 대해 유턴이 어디서 일어나는지를 보여 주는 NUTS 자취 여럿을 그려라. 이것이 과녁의 꼴과 어떻게 이어지는가?

3. **깊이 살피기**. 조건수가 다른 여러 과녁에 NUTS을 돌려라. 평균 나무 깊이를 조건수에 대해 그려라. 어떤 무늬가 드러나는가?

4. **갈라져 나감 파헤치기**. 닐의 깔때기($y \sim \mathcal{N}(0, 3)$, $x \sim \mathcal{N}(0, e^y)$)를 구현하여라. NUTS을 돌려 갈라져 나감이 어디서 일어나는지 찾아라. 걸음 크기가 갈라져 나가는 비율에 어떤 영향을 주는가?

5. **NUTS과 HMC 견주기**. 까다로운 과녁에서 NUTS을 여러 붙박이 $L$ 값의 HMC와 견주어라. 방법마다 기울기 값매김당 ESS을 그려라.

---
