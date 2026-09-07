# 질량 행렬
질량 행렬 $\mathbf{M}$은 운동량이 속도로 어떻게 바뀌는지를 다스리는, HMC의 결정적인 맞춤 매개변수이다. 이 절에서는 질량 행렬이 하는 일, 기하로 풀이하기, 어림하는 방법, 미리 다듬기와의 이음을 다룬다.

---

## 질량 행렬이 하는 일

### 정의와 기본 얼개

질량 행렬 $\mathbf{M}$은 운동 에너지에 나타난다:

$$
K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

그리고 해밀턴 방정식에도 나타난다:

$$
\frac{d\mathbf{x}}{dt} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\nabla U(\mathbf{x})
$$

질량 행렬은 다음을 정한다:

1. **운동량 분포**: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$
2. **운동량에서 나오는 속도**: $\dot{\mathbf{x}} = \mathbf{M}^{-1}\mathbf{v}$
3. 방향마다의 **실효 걸음 크기**

### 물리로 풀이하기

고전 역학에서 질량은 관성, 곧 물체가 빨라짐에 얼마나 버티는지를 정한다. 이와 마찬가지로:

- 방향 $i$의 **질량이 크면**: 힘에 느리게 반응하고 같은 운동량에서 속도가 작다
- 방향 $i$의 **질량이 작으면**: 빠르게 반응하고 같은 운동량에서 속도가 크다

표집에서는 방향마다 "질량"이 퍼텐셜의 "뻣뻣함"에 맞기를 바란다.

### 항등 질량 행렬

가장 단순한 고름인 $\mathbf{M} = \mathbf{I}$은 다음을 준다:

$$
K(\mathbf{v}) = \frac{1}{2}|\mathbf{v}|^2, \quad \mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

과녁 분포의 규모가 모든 방향에서 엇비슷하면 잘 듣는다. 다음일 때는 무너진다:

- 차원마다 흩어짐이 크게 다를 때
- 변수 사이에 강한 상관이 있을 때
- 뒤확률이 어떤 방향으로 "좁을" 때

---

## 기하학적 해석

### 자리 공간의 거리 재는 법

질량 행렬의 역행렬 $\mathbf{M}^{-1}$이 자리 공간의 **리만 계량**을 정한다. 운동 에너지는 다음이 된다:

$$
K(\mathbf{v}) = \frac{1}{2}\|\mathbf{v}\|_{\mathbf{M}^{-1}}^2
$$

여기서 $\|\mathbf{v}\|_{\mathbf{M}^{-1}}^2 = \mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$은 이 계량 아래의 노름의 제곱이다.

### 운동량 공간의 기하

질량 행렬 $\mathbf{M}$이 운동량 공간의 계량을 정한다:

$$
\|\mathbf{p}\|_{\mathbf{M}}^2 = \mathbf{p}^T \mathbf{M} \mathbf{p}
$$

이 계량 아래에서 자리 공간과 운동량 공간은 서로 **쌍대**이다.

### 에너지 면

이차 퍼텐셜 $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x}$에서 에너지 면은 타원체이다:

$$
H = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x} + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v} = E
$$

**가장 좋은 고름**: $\mathbf{M} = \mathbf{A}^{-1}$이면 크기를 알맞게 맞춘 좌표에서 에너지 면이 구면이 되고 움직임이 방향에 고르게 된다.

---

## 맞추기 원리

### 맞추기가 왜 중요한가

$x_1$의 흩어짐이 1이고 $x_2$의 흩어짐이 0.01인 2차원 과녁을 보자:

**$\mathbf{M} = \mathbf{I}$일 때**:

- 두 방향 모두 운동량의 흩어짐이 1이다
- 좁은 $x_2$ 방향에 견주어 속도 $\dot{x}_2 = v_2$이 "너무 빠르다"
- 걸음 크기가 크면 $x_2$에서 흔들린다
- 걸음 크기가 작으면 $x_1$에서 살펴보기가 느리다

**$\mathbf{M} = \text{diag}(1, 100)$일 때**:

- $x_2$의 운동량 흩어짐은 100이지만 속도 $\dot{x}_2 = v_2/100$은 작다
- 실효 걸음 크기가 규모에 맞는다
- 살펴보기가 고르게 된다

### 가장 좋은 질량 행렬

**정리**: 가우스 과녁 $\pi(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$에서 가장 좋은 질량 행렬은 다음과 같다:

$$
\mathbf{M}^* = \boldsymbol{\Sigma}^{-1}
$$

**왜 그런가?** 이렇게 고르면:

1. 해밀턴 함수가 $H = \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) + \frac{1}{2}\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v}$이 된다
2. 표준화한 좌표 $\mathbf{z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$에서 두 항이 모두 방향에 고르다
3. 자취가 모든 방향을 똑같이 효율적으로 살펴본다

### 가우스가 아닌 과녁에서

일반 과녁에서는 다음으로 둔다:

$$
\mathbf{M} \approx \text{Cov}[\mathbf{x}]^{-1} = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T]^{-1}
$$

이는 표본에서 어림한 **뒤확률 공분산의 역행렬**이다.

---

## 질량 행렬의 갈래

### 스칼라(방향에 고름)

$$
\mathbf{M} = m \mathbf{I}
$$

- 맞출 매개변수가 **1개**
- 모든 방향의 규모가 같다고 놓는다
- 실전에서 알맞은 경우가 드물다

### 대각

$$
\mathbf{M} = \text{diag}(m_1, \ldots, m_d)
$$

- 맞출 매개변수가 **$d$개**
- 주변 흩어짐이 다른 것에 맞춘다
- 상관을 무시한다
- 대부분의 소프트웨어(Stan, PyMC)에서 **기본 고름**이다

**어림하기**: 달굼 표본에서 어림한 $m_i = 1/\text{Var}[x_i]$을 쓴다.

### 온전한(빽빽한) 행렬

$$
\mathbf{M} = \text{any positive definite matrix}
$$

- 매개변수가 **$d(d+1)/2$개**
- 상관을 잡아낸다
- 더 비싸다. 곧 저장에 $O(d^2)$, 촐레스키에 $O(d^3)$이 든다
- 서로 얽힌 과녁에서 표집을 크게 낫게 할 수 있다

**어림하기**: $\mathbf{M} = \text{Cov}[\mathbf{x}]^{-1}$을 쓰지만 잘 어림하려면 표본이 많이 필요하다.

### 견줌

| 갈래 | 매개변수 | 상관을 잡나 | 걸음당 값 | 어디에 좋은가 |
|------|------------|----------------------|---------------|----------|
| 스칼라 | 1 | 아니오 | $O(d)$ | 방향에 고른 과녁 |
| 대각 | $d$ | 아니오 | $O(d)$ | 규모가 다를 때 |
| 온전함 | $d(d+1)/2$ | 예 | $O(d^2)$ | 서로 얽힌 과녁 |

---

## 달굼 동안 맞춰 가기

### 맞춰 가기 문제

$\mathbf{M} \approx \text{Cov}[\mathbf{x}]^{-1}$이 필요한데 표집을 해 봐야 $\text{Cov}[\mathbf{x}]$을 안다. 이 닭과 달걀 같은 문제는 **맞춰 가는 달굼**으로 푼다.

### 달굼 전략

**1단계: 첫 살펴보기**(되풀이 1–75)

- $\mathbf{M} = \mathbf{I}$을 쓴다
- 걸음 크기 $\epsilon$을 세게 맞춘다
- 목표: 전형 집합 찾기

**2단계: 공분산 어림하기**(되풀이 76–900)

- 표본에서 $\hat{\boldsymbol{\Sigma}}$을 어림한다
- $\mathbf{M} = \hat{\boldsymbol{\Sigma}}^{-1}$을 이따금 새로 고친다
- $\epsilon$을 계속 맞춰 간다

**3단계: 마지막 맞추기**(되풀이 901–1000)

- $\mathbf{M}$을 붙박아 둔다
- $\epsilon$을 마지막으로 맞춘다
- 안정한지 확인한다

**표집 단계**: (되풀이 1001 이상)

- 더는 맞추지 않는다
- 추론에 쓸 표본을 모은다

### 웰퍼드의 흐름 알고리즘

공분산을 흐름으로 효율적으로 어림하려면:

```python
class OnlineCovariance:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))
    
    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def covariance(self):
        if self.n < 2:
            return np.eye(len(self.mean))
        return self.M2 / (self.n - 1)
```

### 벌주기

표본이 적으면 $\hat{\boldsymbol{\Sigma}}$의 조건이 나쁘거나 특이할 수 있다. 벌주어 다듬어라:

$$
\hat{\boldsymbol{\Sigma}}_{\text{reg}} = (1 - \alpha)\hat{\boldsymbol{\Sigma}} + \alpha \cdot \text{diag}(\hat{\boldsymbol{\Sigma}})
$$

또는 능선 항을 더한다:

$$
\hat{\boldsymbol{\Sigma}}_{\text{reg}} = \hat{\boldsymbol{\Sigma}} + \lambda \mathbf{I}
$$

---

## 구현의 세부

### 촐레스키 분해

효율적으로 셈하려면 $\mathbf{M} = \mathbf{L}\mathbf{L}^T$인 촐레스키 인수 $\mathbf{L}$을 저장한다:

**운동량 표집하기**:
```python
v = L @ np.random.randn(d)  # v ~ N(0, M)
```

**운동 에너지 셈하기**:
```python
z = solve_triangular(L, v, lower=True)  # L z = v
K = 0.5 * np.dot(z, z)  # K = ½ vᵀ M⁻¹ v = ½ |z|²
```

**운동량에서 속도 얻기**:
```python
# M⁻¹ v = (L Lᵀ)⁻¹ v = L⁻ᵀ L⁻¹ v
z = solve_triangular(L, v, lower=True)
velocity = solve_triangular(L.T, z, lower=False)
```

### 대각 질량 행렬에서

$\mathbf{M} = \text{diag}(m_1, \ldots, m_d)$일 때:

```python
# 대각과 그 제곱근 저장
m_diag = np.array([m1, m2, ..., md])
sqrt_m = np.sqrt(m_diag)
inv_m = 1.0 / m_diag

# 운동량 표집
v = sqrt_m * np.random.randn(d)

# 운동 에너지
K = 0.5 * np.sum(v**2 * inv_m)

# 속도
velocity = v * inv_m
```

### 수치적 안정성

어림한 공분산의 역행렬을 구할 때:

```python
def safe_inverse(Sigma, min_var=1e-6):
    """공분산 행렬을 안전하게 뒤집기."""
    # 최소 흩어짐 지키기
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma, np.maximum(np.diag(Sigma), min_var))
    
    # 거의 특이한 행렬에는 유사 역행렬 쓰기
    try:
        M = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(Sigma)
    
    # 대칭 지키기
    M = 0.5 * (M + M.T)
    
    return M
```

---

## 미리 다듬기와의 이음

### 미리 다듬은 기울기 내리기로 본 HMC

개구리뜀의 운동량 새로 고치기는 다음과 같다:

$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{\epsilon}{2}\nabla \log \pi(\mathbf{x}_n)
$$

자리 새로 고치기는 다음과 같다:

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_{n+1/2}
$$

(한 걸음에 대해 어림해서) 합치면:

$$
\mathbf{x}_{n+1} \approx \mathbf{x}_n + \frac{\epsilon^2}{2}\mathbf{M}^{-1}\nabla \log \pi(\mathbf{x}_n)
$$

이는 미리 다듬는 행렬이 $\mathbf{M}^{-1}$인 **미리 다듬은 기울기 오르기**이다.

### 가장 좋은 미리 다듬기

최적화에서 가장 좋은 미리 다듬는 행렬은 헤세 행렬의 역행렬이다(뉴턴 방법):

$$
\mathbf{P}^* = (-\nabla^2 \log \pi)^{-1} = \boldsymbol{\Sigma}
$$

HMC 표집에서는 이것이 $\mathbf{M}^{-1} = \boldsymbol{\Sigma}$, 곧 $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$을 넌지시 일러 준다.

### 조건수

미리 다듬은 체계의 **조건수**가 모임을 정한다:

$$
\kappa = \frac{\lambda_{\max}(\mathbf{M}^{-1}\mathbf{A})}{\lambda_{\min}(\mathbf{M}^{-1}\mathbf{A})}
$$

여기서 $\mathbf{A} = -\nabla^2 \log \pi$은 헤세 행렬이다.

- **미리 다듬지 않으면**($\mathbf{M} = \mathbf{I}$): $\kappa = \lambda_{\max}(\mathbf{A})/\lambda_{\min}(\mathbf{A})$
- **가장 좋게 미리 다듬으면**($\mathbf{M} = \mathbf{A}$): $\kappa = 1$

조건수가 낮을수록 → 섞임이 빠르다.

---

## 리만 HMC(맛보기)

### 자리에 달린 질량

표준 HMC는 $\mathbf{M}$을 상수로 쓴다. **리만 HMC**은 $\mathbf{M}(\mathbf{x})$이 자리에 따라 달라지도록 한다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T\mathbf{M}(\mathbf{x})^{-1}\mathbf{v} + \frac{1}{2}\log|\mathbf{M}(\mathbf{x})|
$$

올바른 주변 분포를 얻으려면 로그 행렬식 항이 더 필요하다.

### 좋은 점

- 그 자리 기하에 맞춘다
- 굽음이 달라져도 다룬다
- 특히 깔때기 꼴 분포에 쓸모 있다

### 어려움

- 해밀턴 함수가 더 이상 나뉘지 않는다
- 개구리뜀을 그대로 쓸 수 없다
- 넌지시 푸는 적분기나 넓힌 적분기가 필요하다
- 걸음마다 더 비싸다

---

## 실전 권고

### 기본 전략

1. **대각으로 시작하기**: $\mathbf{M} = \text{diag}(1/\hat{\sigma}_1^2, \ldots, 1/\hat{\sigma}_d^2)$을 쓴다
2. **달굼 동안 맞춰 가기**: 표본에서 주변 흩어짐을 어림한다
3. 다음이면 **온전한 행렬을 생각해 보기**:
   - 강한 상관이 있으리라 여겨질 때
   - 대각으로 맞춰 가기가 나쁜 결과를 줄 때
   - 차원이 알맞을 때($d < 100$)

### 온전한 질량 행렬을 언제 쓰나

✓ **다음일 때 온전한 행렬을 써라**:

- 매개변수의 상관이 클 때(|ρ| > 0.8)
- 차원이 그리 크지 않을 때(d < 100-200)
- 달굼 표본이 넉넉할 때(> 100d)

✗ **다음일 때는 대각을 지켜라**:

- 차원이 높을 때(d > 200)
- 매개변수가 거의 독립일 때
- 계산 예산이 적을 때
- 달굼 표본이 적을 때

### 진단

**맞춰 가기가 잘되었는지 살피기**:
```python
# 달굼 뒤
estimated_cov = np.cov(warmup_samples.T)
M_inv = np.linalg.inv(M)

# 거의 항등 행렬이어야 한다
whitened = estimated_cov @ M  # 잘 맞춰졌으면 ≈ I
print("Whitening check:", np.diag(whitened))  # ≈ 1이어야 함
```

**조건이 나쁜지 살피기**:
```python
cond_number = np.linalg.cond(M)
if cond_number > 1e6:
    print("Warning: Mass matrix poorly conditioned")
```

---

## 요약

| 살필 점 | 권하는 바 |
|--------|----------------|
| **하는 일** | 과녁의 기하에 맞도록 움직임을 미리 다듬는다 |
| **가장 좋은 고름** | $\mathbf{M} = \text{Cov}[\mathbf{x}]^{-1}$ |
| **기본값** | 대각, 달굼 동안 맞춰 감 |
| **온전한 행렬** | 상관이 강하고 $d$이 알맞을 때 |
| **맞춰 가기** | 흐름 알고리즘으로 달굼 표본에서 어림한다 |
| **벌주기** | 안정을 위해 능선 항을 더하거나 대각과 섞는다 |

질량 행렬은 조건이 나쁜 과녁 앞에서 힘겨워하던 HMC를 복잡한 뒤확률 기하를 효율적으로 살펴보는 방법으로 바꾼다. 실전 성능에는 제대로 맞춰 가는 것이 꼭 필요하다.

---

## 참고 문헌

1. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
4. Stan Development Team. "Stan Reference Manual: HMC Algorithm Parameters."

## 연습문제

1. **질량 행렬의 효과**. 공분산이 $\begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix}$인 2차원 가우스에서 (가) $\mathbf{M} = \mathbf{I}$, (나) 대각 $\mathbf{M}$, (다) 온전한 $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$으로 표집하여라. 받아들임 비율과 ESS을 견주어라.

2. **흐름으로 맞춰 가기**. 공분산을 흐름으로 어림하는 달굼을 구현하여라. 되풀이가 늘면서 어림한 공분산이 참 공분산으로 어떻게 모이는지 그려라.

3. **조건수 실험**. 조건수가 여러 가지(1, 10, 100, 1000)인 과녁에 대해 질량 행렬을 맞춰 갈 때와 아닐 때의 HMC 성능을 견주어라.

4. **차원 높은 대각 행렬**. 주변 흩어짐이 서로 다른 100차원 과녁에 대해 대각 질량 행렬 맞춰 가기를 구현하고 표집 효율이 나아지는지 확인하여라.

5. **벌주기 견주기**. 표본이 적을 때 질량 행렬을 어림하는 여러 벌주기 전략(능선, 대각 쪽으로 오그라들기)을 견주어라.

---
