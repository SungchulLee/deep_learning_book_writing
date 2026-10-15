# NUTS: 궤적 길이를 조절하는 HMC

NUTS(No-U-Turn Sampler)는 앞뒤로 확장한 해밀토니안 궤적의 유턴을 검사해 적분 길이를 선택한다. 스텝 크기·질량 행렬의 조정과 수치 진단은 여전히 필요하며, 유턴에서 단순히 멈추고 마지막 점을 반환하는 알고리즘과는 다르다.

## 1. 왜 필요한가

고정 길이 HMC는 스텝 크기 ε와 Leapfrog 수 L로 적분시간 εL을 정한다. 너무 짧으면 이동이 작고 너무 길면 되돌아오는 궤적에 계산을 쓸 수 있다. 알맞은 길이는 목표 기하와 위치·운동량에 따라 다르다.

NUTS는 무작위 방향의 배증 확장, 부분 트리의 유턴 검사와 적절한 후보 선택을 결합한다. 조화 진동자에서는 되돌아오는 직관을 쉽게 볼 수 있지만 일반적인 고차원 궤적이 정확히 주기적이거나 한 번의 유턴이 충분한 탐색을 보장하는 것은 아니다.

## 2. 운동량·속도와 유턴 조건

상수 양의 정부호 질량 행렬 M, 위치 양 끝 x⁻·x⁺, 운동량 p⁻·p⁺에서 속도는 M⁻¹p다. 기본적인 끝점 기준은 다음 둘 중 하나가 음수가 될 때 유턴으로 판정한다.

$$(x^+-x^-)^TM^{-1}p^+,\qquad (x^+-x^-)^TM^{-1}p^-.$$

아래 구현의 v는 이름과 달리 운동량이며 N(0,M)에서 생성한다. M=I일 때에만 운동량과 속도가 같다. 뒤쪽 끝은 음의 시간 방향으로 확장하므로 두 조건의 부호를 같은 방식으로 검사한다.

단일 1차원 조화계의 직관을 서로 다른 주파수를 갖는 모든 다변량 가우스에 그대로 적용하지 않는다. 실제 NUTS는 부분 트리 안의 유턴도 검사해야 하며 일반화된 기준과 후보 선택의 정확성은 해당 변형 전체의 증명에 달려 있다.

## 3. 배증 확장과 계산량

초기 상태 하나에서 시작해 새 확장마다 1,2,4,…개의 적분 단계를 추가한다. j회 완전 확장했다면 새 Leapfrog는 2ʲ−1개, 초기점을 포함한 방문점은 2ʲ개다. 아래 도식은 마디 수와 인덱스를 혼용하므로 정확한 단계 수 계산으로 사용하지 않는다.

재귀 깊이는 O(log L)일 수 있지만 실제 경사 평가 수는 O(L)다. 무작위 방향과 대칭 트리만으로 올바른 목표 보존이 자동으로 증명되지는 않는다. 부분 트리의 정지·후보 자격·선택 규칙을 함께 만족해야 한다. 트리에는 양 끝 상태, 후보, 후보 질량 또는 개수와 정지 정보를 유지한다.

```
Depth 0:        [x₀]                    (1 point)
Depth 1:    [x₋₁, x₀, x₁]              (2 points added)
Depth 2: [x₋₃..x₋₁, x₀, x₁..x₃]        (4 points added)
   ...
Depth j: 2ʲ points total
```

## 4. 후보 선택 방식의 구분

### 슬라이스 방식과 다항 방식

원래의 슬라이스 NUTS는 보조 슬라이스 변수로 후보 집합을 제한하고 그 집합 안의 상태를 균등하게 선택한다. 다항(multinomial) 방식은 해밀토니안 밀도 exp(−H)에 비례한 질량을 이용한다. 두 방식을 모두 쓸 수 있는 상태에서 균등 선택한다고 설명하면 안 된다.

아래 의사코드는 트리 확장의 구조를 보여 주는 생략된 조각이다. BuildTree의 에너지 차이 한계는 수치 실패 정지 기준이지 슬라이스 자격 조건을 대신하지 않는다. 슬라이스 변수와 필요한 인수·경계 처리가 빠져 있어 완결된 원래 알고리즘의 명세로 볼 수 없다.

같은 길이의 부분 트리 내부에서는 새 후보 질량을 기존·새 질량의 합으로 나누어 후보를 갱신할 수 있다. 바깥 배증 단계의 biased-progressive 선택은 새 질량/기존 질량을 사용하는 별도의 규칙일 수 있으므로 그 비가 1을 넘는다는 사실만으로 오류라고 단정하지 않는다. [Stan의 NUTS 설명](https://mc-stan.org/docs/reference-manual/mcmc.html)은 이 구분을 다룬다. 정확성은 선택·정지 규칙 전체에 대해 검토해야 한다.

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
```
if Random() < n''/(n' + n''):
    x' = x''
```

## 5. 유턴·발산·깊이 제한

유턴, 큰 에너지 오차·비유한 값, 최대 트리 깊이 등으로 확장을 끝낼 수 있다. 최대 깊이 10이면 이 페이지의 완전 배증 구조에서 새 Leapfrog가 최대 1023개이며 중간 정지 시 더 적다.

에너지 한계 Δ_max=1000은 코드가 선택한 수치 경고 기준이고 원하는 추정 오차의 허용범위가 아니다. 아래 is_divergent는 큰 양의 에너지 오차와 NaN을 검사하지만 모든 비유한 입력을 포괄하지 않는다. 뒤의 클래스는 별도의 비교식으로 판단하므로 NaN에서 valid=False이면서 divergent=False가 될 수 있다. 함수 이름만으로 모든 실패가 보고된다고 가정하지 않는다.

```python
def no_uturn(x_minus, x_plus, v_minus, v_plus, M_inv):
    delta_x = x_plus - x_minus
    return (np.dot(delta_x, M_inv @ v_plus) >= 0 and 
            np.dot(delta_x, M_inv @ v_minus) >= 0)
```
```python
def is_divergent(H_new, H_old, delta_max=1000):
    return H_new - H_old > delta_max or np.isnan(H_new)
```

## 6. 목표 보존과 수락 통계량

운동량을 포함한 확장 목표는 exp(−H(x,p))에 비례한다. 가역적인 체적 보존 적분, 트리 생성의 대칭성, 부분 트리의 자격·정지 조건과 후보 선택이 함께 목표 보존을 구성한다. 종료시점을 현재 궤적만 보고 임의로 정한 뒤 마지막 점을 반환하면 일반적으로 올바른 HMC 보정이 되지 않는다.

슬라이스 방식에서 E~Exp(1)를 사용하면 H(x,p)<H₀+E인 집합을 정의할 수 있다. 이는 다항 가중 선택과 구별되는 보조변수 표현이다. 후보 자격을 잃은 모든 순간에 무조건 적분을 멈춘다는 단순 규칙과도 같지 않다.

NUTS의 적응용 accept_stat은 보통 시도한 적분 단계의 MH 수락확률을 요약한 값이다. 단순히 유효 상태 수/전체 상태 수나 최종 위치가 바뀐 비율이라고 정의하지 않는다. 변형별 보정과 선택 방식은 [Hoffman·Gelman의 원 논문](https://www.jmlr.org/papers/v15/hoffman14a.html)과 사용한 구현의 명세를 확인한다.

## 7. 구현

아래 클래스는 상수 질량 행렬과 고정 ε를 사용하는 교육용 다항 가중 트리 예제다. sample의 n_warmup은 같은 step을 반복해 버리는 횟수이며 스텝 크기·질량 행렬을 학습하지 않는다. 완전한 자동 적응 NUTS나 검증된 라이브러리 구현으로 소개하지 않는다.

차원과 초기값·경사 형태, 양의 ε, 양의 정부호 M, 양의 정수 깊이와 유한한 로그확률을 확인해야 한다. 워밍업 중 발산은 반환 통계에 포함하지 않고 깊이는 평균으로만 보고하므로 개별 최대 깊이 도달률을 알 수 없다. NaN·에너지 오류와 트리 선택·정지 조건을 별도로 검토해야 한다.

2차원 정규 예제의 알려진 평균·공분산은 확인 기준이지만 한 번의 출력만으로 알고리즘 전체의 불변성·수렴을 검증할 수 없다. 표시된 출력과 구현은 보존된 예시다.

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
```
Average tree depth: 4.4
Divergence rate: 0.00%
Sample mean: [ 0.01095822 -0.00612379]
```

## 8. 진단

트리 깊이 자체에 보편적인 적정 범위 4~8이 있는 것은 아니다. 같은 깊이도 ε·목표 기하·질량 행렬에 따라 이동과 비용이 다르다. 최대 깊이 도달은 효율 제한을, 발산은 수치 탐색 문제를 의심하게 하므로 원인을 구분한다.

발산 1% 미만이면 괜찮다는 규칙은 없다. 위치·설정 민감도와 관심량의 변화를 조사한다. 발산이 없더라도 모드 누락은 가능하다. E-BFMI의 경험적 형태는 다음과 같다.

$$\widehat{\mathrm{E\! -\! BFMI}}=\frac{\operatorname{mean}_{t>1}(E_t-E_{t-1})^2}{\widehat{\operatorname{Var}}(E_t)}.$$

낮은 값은 에너지 탐색의 경고 기준이며 고정 임계값을 통과했다고 수렴이 보장되지는 않는다. 여러 체인의 R-hat·함수별 ESS·MCSE와 그림을 함께 평가한다.

## 9. NUTS과 표준 HMC의 비교

NUTS는 고정 L을 정하는 부담을 줄이지만 트리 구축·후보 선택 비용이 추가된다. 잘 맞춘 고정 길이 HMC가 더 효율적일 수도 있고 반대일 수도 있다. 차원 하나만으로 승자를 결정하지 않는다.

같은 관심량의 MCSE, 경사 평가당·시간당 ESS와 워밍업 비용을 비교한다. 스텝 크기·질량 행렬, 다봉성·조건수·초기상태의 차이도 확인한다. NUTS가 모든 문제에서 일관되게 더 좋은 효율을 보인다는 표는 근거로 사용할 수 없다.

## 10. 실전 권고

워밍업에서 스텝 크기와 질량 행렬을 조정하고 수집 단계의 설정을 고정하는 방식을 사용할 수 있다. 일정과 구간 길이는 구현·전체 워밍업 길이에 따라 달라지므로 특정 75·975·1000 구간을 일반적인 명세로 외우지 않는다.

아래 adapt_step_size는 수락 통계량에 따라 ε를 1.02배·0.98배 하는 휴리스틱이다. 함수 인수에 gamma·t0·kappa가 있어도 실제 갱신에는 사용하지 않으며 쌍대 평균 알고리즘을 구현하지 않는다. 주어진 NUTS 클래스도 이 함수를 호출하지 않는다.

목표 수락률 0.8은 흔한 출발점의 하나이며 보편적인 최적값이 아니다. 깊이 한계를 늘리기 전 작은 ε·나쁜 조건수·기하 문제를 구분한다. 대각·밀집 질량 행렬은 표현력과 추정·곱셈 비용을 함께 비교한다.

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

## 연습문제

1. 원래 슬라이스 NUTS와 다항 NUTS의 후보 자격·선택 비율을 구분하라. 슬라이스 방식은 보조 높이를 통과한 상태의 개수를, 다항 방식은 exp(−H) 질량을 사용한다. 정지 조건까지 같은 방식의 명세로 맞춰야 한다.

2. K=pᵀM⁻¹p/2일 때 유턴 식에서 p가 아닌 M⁻¹p를 쓰는 이유를 유도하라. 해밀토니안 방정식 ẋ=∂K/∂p=M⁻¹p가 실제 속도를 주기 때문이다.

3. 깊이 10의 완전 배증 트리에서 새 적분 수를 계산하라. 1+2+⋯+2⁹=1023이며 초기점까지 1024개다. 재귀 깊이 O(log L)과 경사 평가 O(L)를 구분한다.

4. 이 클래스의 워밍업과 별도 adapt_step_size 함수를 검토하라. 워밍업은 고정 step 반복이고 조정 함수를 호출하지 않는다. 별도 함수도 쌍대 평균이 아니라 고정 배율 휴리스틱이다.

5. HMC와 NUTS 비교를 설계할 때 통제할 항목을 정리하라. 같은 목표·관심량·예산·시드 변동을 고려하고 전체 워밍업 비용과 경사 평가당 오차, 발산·깊이 도달을 함께 평가한다.

## 정리하며

NUTS의 핵심은 유턴에서 궤적을 확장·정지하는 규칙과 목표를 보존하는 후보 선택의 결합이다. 슬라이스·다항 방식을 구분하고 자동 조정의 구현 여부, 에너지·깊이 경고와 함수별 오차를 확인한다.

**참고 문헌**

1. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *JMLR*, 15, 1593-1623.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Stan Development Team. "Stan Reference Manual."
4. Betancourt, M. (2016). "Diagnosing Biased Inference with Divergences." Stan Case Studies.
