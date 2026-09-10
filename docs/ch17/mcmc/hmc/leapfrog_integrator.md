# 개구리뜀 적분기
개구리뜀 적분기는 해밀턴 몬테카를로의 수치 엔진이다. 이 절에서는 첫 원리부터 적분기를 펼친다. 곧 연산자 쪼개기, 슈퇴르머-베를레 얼개, 심플렉틱 짜임, 오차 살피기, 실전 구현에서 살필 점을 다룬다.

---

## 1. 수치 적분 문제

### 수치 적분이 왜 필요한가

해밀턴 방정식은 상미분방정식의 체계를 이룬다:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = -\nabla U(\mathbf{x})
$$

대부분의 과녁 분포에서 이 방정식에는 닫힌 꼴 풀이가 없다. 수치로 적분해야 한다.

### MCMC에 필요한 것

아무 수치 적분기나 되는 것이 아니다. 옳은 MCMC 표집에는 다음이 필요하다:

| 성질 | 요구 조건 | 왜 중요한가 |
|----------|-------------|----------------|
| **심플렉틱** | 위상 공간 부피를 지킴 | MH 비에 야코비 바로잡기가 필요 없음 |
| **되돌릴 수 있음** | 시간에 대칭 | 자세한 균형 |
| **정확함** | 에너지 오차가 묶임 | 높은 받아들임 비율 |
| **효율적임** | 기울기 값매김이 적음 | 셈 값 |

개구리뜀 적분기는 이 넷을 모두 이룬다.

### 흔한 방법이 무엇을 그르치나

**오일러 방법**:

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_n, \quad \mathbf{v}_{n+1} = \mathbf{v}_n - \epsilon \nabla U(\mathbf{x}_n)
$$

말썽:

- 심플렉틱이 아니다(부피가 늘거나 줄어든다)
- 되돌릴 수 없다
- 시간이 갈수록 에너지가 한쪽으로 쏠린다
- 자취가 길면 받아들임 비율이 0으로 떨어진다

**룽게-쿠타 방법**(RK4 등):

- 걸음마다 정확도가 높다
- 그러나 심플렉틱이 아니다
- 더 느리기는 해도 에너지가 쏠린다
- 긴 MCMC 자취에는 맞지 않다

---

## 2. 연산자 쪼개기

### 쪼개기 생각

해밀턴 함수 $H = U(\mathbf{x}) + K(\mathbf{v})$은 연산자 둘로 움직임을 낳는다:

**연산자 $\mathcal{A}$**(퍼텐셜 에너지 $U$에서):

$$
\frac{d\mathbf{x}}{dt} = 0, \quad \frac{d\mathbf{v}}{dt} = -\nabla U(\mathbf{x})
$$

**연산자 $\mathcal{B}$**(운동 에너지 $K$에서):

$$
\frac{d\mathbf{x}}{dt} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = 0
$$

연산자를 하나씩 보면 정확히 풀 수 있다:

$$
e^{\epsilon \mathcal{A}}: \quad \mathbf{v} \to \mathbf{v} - \epsilon \nabla U(\mathbf{x}), \quad \mathbf{x} \to \mathbf{x}
$$

$$
e^{\epsilon \mathcal{B}}: \quad \mathbf{x} \to \mathbf{x} + \epsilon \mathbf{M}^{-1}\mathbf{v}, \quad \mathbf{v} \to \mathbf{v}
$$

온전한 움직임 $e^{\epsilon(\mathcal{A} + \mathcal{B})}$은 두 효과를 섞으므로 정확히 셈할 수 없다.

### 베이커-캠벨-하우스도르프 공식

바꿔 곱할 수 없는 연산자에 대해 BCH 공식은 다음을 준다:

$$
e^{\epsilon \mathcal{A}} e^{\epsilon \mathcal{B}} = \exp\left(\epsilon(\mathcal{A} + \mathcal{B}) + \frac{\epsilon^2}{2}[\mathcal{A}, \mathcal{B}] + O(\epsilon^3)\right)
$$

여기서 $[\mathcal{A}, \mathcal{B}] = \mathcal{A}\mathcal{B} - \mathcal{B}\mathcal{A}$은 교환자이다.

**일차 쪼개기**(리-트로터):

$$
e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\epsilon \mathcal{A}} e^{\epsilon \mathcal{B}} + O(\epsilon^2)
$$

**이차 쪼개기**(스트랭):

$$
e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\frac{\epsilon}{2} \mathcal{A}} e^{\epsilon \mathcal{B}} e^{\frac{\epsilon}{2} \mathcal{A}} + O(\epsilon^3)
$$

스트랭 쪼개기의 대칭 짜임이 $O(\epsilon^2)$ 오차 항을 지운다.

---

## 3. 개구리뜀 알고리즘

### 스트랭 쪼개기에서 이끌어 내기

상태 $(\mathbf{x}_n, \mathbf{v}_n)$에 $e^{\frac{\epsilon}{2} \mathcal{A}} e^{\epsilon \mathcal{B}} e^{\frac{\epsilon}{2} \mathcal{A}}$을 씌우면:

**걸음 1** — 운동량 반 걸음 새로 고치기($e^{\frac{\epsilon}{2}\mathcal{A}}$):

$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n - \frac{\epsilon}{2} \nabla U(\mathbf{x}_n)
$$

**걸음 2** — 자리 온 걸음 새로 고치기($e^{\epsilon \mathcal{B}}$):

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1} \mathbf{v}_{n+1/2}
$$

**걸음 3** — 운동량 반 걸음 새로 고치기($e^{\frac{\epsilon}{2}\mathcal{A}}$):

$$
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} - \frac{\epsilon}{2} \nabla U(\mathbf{x}_{n+1})
$$

### "개구리뜀"이라는 이름

이 이름은 엇갈린 시간 점에서 왔다:

- 자리는 정수 시간에서 정해진다. 곧 $\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \ldots$이다
- 운동량은 반정수 시간에서 자리를 "뛰어넘는다". 곧 $\mathbf{v}_{1/2}, \mathbf{v}_{3/2}, \ldots$이다

```
Time:     0      ½      1      3/2     2
          |      |      |      |       |
Position: x₀ -------- x₁ -------- x₂ ---
Momentum: v₀ -- v½ -------- v3/2 -------
```

### 같은 뜻의 다른 꼴

**자리 먼저(PVP)**:

$$
\mathbf{x}_{n+1/2} = \mathbf{x}_n + \frac{\epsilon}{2}\mathbf{M}^{-1}\mathbf{v}_n, \quad
\mathbf{v}_{n+1} = \mathbf{v}_n - \epsilon \nabla U(\mathbf{x}_{n+1/2}), \quad
\mathbf{x}_{n+1} = \mathbf{x}_{n+1/2} + \frac{\epsilon}{2}\mathbf{M}^{-1}\mathbf{v}_{n+1}
$$

**운동량 먼저(VPV)** — 표준 꼴이다:

$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n - \frac{\epsilon}{2}\nabla U(\mathbf{x}_n), \quad
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_{n+1/2}, \quad
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} - \frac{\epsilon}{2}\nabla U(\mathbf{x}_{n+1})
$$

둘은 같은 것이며 이차 정확도를 갖는다.

---

## 4. 여러 개구리뜀 걸음

### 걸음 합성하기

개구리뜀 걸음 $L$개에서는 한 걸음 사상을 $L$번 합성한다. 경계의 반 걸음들은 하나로 합쳐진다:

```python
def leapfrog(x, v, epsilon, L, grad_U, M_inv):
    # 운동량의 첫 반 걸음
    v = v - (epsilon / 2) * grad_U(x)
    
    # 온 걸음
    for i in range(L - 1):
        x = x + epsilon * M_inv @ v
        v = v - epsilon * grad_U(x)
    
    # 마지막 자리 새로 고치기
    x = x + epsilon * M_inv @ v
    
    # 운동량의 마지막 반 걸음
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

**기울기 값매김**: 걸음 $L$개에는 기울기 값매김이 ($2L$개가 아니라) $L + 1$개 든다. 이웃한 반 걸음이 합쳐지기 때문이다.

### 자취 길이

전체 적분 시간은 $T = L\epsilon$이다. 이것이 자취가 위상 공간에서 얼마나 멀리 가는지를 정한다.

**주고받음**:

- $T$이 크면: 더 잘 살펴보지만 셈이 많아진다
- $T$이 작으면: 싸지만 표본이 더 얽힌다

보통 값: $L = 10$–$100$, $\epsilon = 0.01$–$0.1$이며 이때 $T = 0.1$–$10$이다.

---

## 5. 심플렉틱 성질

### 정의

사상 $\phi: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{X}, \mathbf{V})$의 야코비 행렬 $\mathbf{J}$이 다음을 만족하면 그 사상은 **심플렉틱**이다:

$$
\mathbf{J}^T \mathbf{\Omega} \mathbf{J} = \mathbf{\Omega}, \quad \text{where } \mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}
$$

### 개구리뜀이 왜 심플렉틱인가

낱낱의 새로 고치기가 모두 심플렉틱이다:

**운동량 새로 고치기** $\mathbf{v} \to \mathbf{v} - \epsilon \nabla U(\mathbf{x})$:

$$
\mathbf{J}_v = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ -\epsilon \nabla^2 U & \mathbf{I} \end{pmatrix}
$$

확인: $\mathbf{J}_v^T \mathbf{\Omega} \mathbf{J}_v = \mathbf{\Omega}$ ✓(밀기 바꿈)

**자리 새로 고치기** $\mathbf{x} \to \mathbf{x} + \epsilon \mathbf{M}^{-1}\mathbf{v}$:

$$
\mathbf{J}_x = \begin{pmatrix} \mathbf{I} & \epsilon \mathbf{M}^{-1} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}
$$

확인: $\mathbf{J}_x^T \mathbf{\Omega} \mathbf{J}_x = \mathbf{\Omega}$ ✓(밀기 바꿈)

**합성**: 심플렉틱 사상의 곱은 심플렉틱이다.

### 따라 나오는 것

1. **부피 지킴**: $|\det \mathbf{J}| = 1$

2. **MH 비에 야코비가 없음**: 제안 밀도의 비가 에너지 차이만으로 단순해진다

3. **오래도록 안정함**: 에너지가 묶인 채로 남는다(한쪽으로 쏠리지 않는다)

---

## 6. 시간을 되돌릴 수 있음

### 정의

운동량 뒤집기 $R: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$에 대해 다음이 성립하면 사상 $\phi$은 **시간을 되돌릴 수 있다**:

$$
\phi^{-1} = R \circ \phi \circ R
$$

같은 말로 $\phi$을 씌우고 운동량을 뒤집은 뒤 다시 $\phi$을 씌우고 운동량을 뒤집으면 → 처음으로 돌아온다.

### 개구리뜀에 대한 증명

VPV 개구리뜀은 $\phi = A_{1/2} \circ B \circ A_{1/2}$으로 쓸 수 있다. 여기서:

- $A_{1/2}$: 운동량 반 걸음 새로 고치기
- $B$: 자리 온 걸음 새로 고치기

$A_{1/2}$과 $B$은 모두 $R$ 켤레 아래에서 대칭이다:

- $R \circ A_{1/2} \circ R = A_{1/2}^{-1}$(운동량 새로 고치기가 부호를 뒤집는다)
- $R \circ B \circ R = B^{-1}$(운동량을 뒤집으면 자리 새로 고치기가 되돌아간다)

따라서 다음이 성립한다.

$$
R \circ \phi \circ R = R \circ A_{1/2} \circ B \circ A_{1/2} \circ R = A_{1/2}^{-1} \circ B^{-1} \circ A_{1/2}^{-1} = \phi^{-1}
$$

### MCMC에서의 중요함

시간을 되돌릴 수 있으면 MH 받아들임 걸음과 어우러져 **자세한 균형**이 보장된다. 제안 $(\mathbf{x}, \mathbf{v}) \to (\mathbf{x}', \mathbf{v}')$의 확률은 거꾸로 가는 $(\mathbf{x}', -\mathbf{v}') \to (\mathbf{x}, -\mathbf{v})$의 확률과 같다.

---

## 7. 오차 분석

### 그 자리에서 끊은 오차

개구리뜀 적분기는 정확한 흐름 $\phi_\epsilon^{\text{exact}}$을 다음 오차로 어림한다:

$$
\phi_\epsilon^{\text{leapfrog}} = \phi_\epsilon^{\text{exact}} + O(\epsilon^3)
$$

그 자리 오차는 걸음마다 $O(\epsilon^3)$이다(그 자리에서는 삼차 정확도).

### 전체 오차

시간 $T = L\epsilon$을 아우르는 걸음 $L$개 뒤에:

$$
\text{Global error} = O(L \cdot \epsilon^3) = O(T \cdot \epsilon^2)
$$

전체 오차는 $O(\epsilon^2)$이다(전체로는 이차 정확도).

### 에너지 오차

심플렉틱 적분기에서는 에너지가 흔들리기는 해도 쏠리지 않는다:

**거꾸로 오차 살피기**: 개구리뜀 적분기는 **고친 해밀턴 함수**를 정확히 푼다:

$$
\tilde{H}(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, \mathbf{v}) + \epsilon^2 H_2(\mathbf{x}, \mathbf{v}) + O(\epsilon^4)
$$

여기서 $H_2$은 바로잡는 항이다.

**따라 나오는 것**: 에너지 오차 $|H(\mathbf{x}_L, \mathbf{v}_L) - H(\mathbf{x}_0, \mathbf{v}_0)| = O(\epsilon^2)$이며 $L$에 상관없이 고르게 묶인다.

심플렉틱 적분기가 오랜 시간 적분에서 뛰어난 까닭이 이것이다. 곧 에너지 오차가 쌓이지 않는다.

### 에너지 오차와 받아들임

MH 받아들임 확률은 에너지 오차에 달려 있다:

$$
\alpha = \min(1, \exp(-\Delta H))
$$

여기서 $\Delta H = H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}, \mathbf{v})$이다.

| 에너지 오차 $\Delta H$ | 받아들임 $\alpha$ |
|------------------------|---------------------|
| 0 | 1.00 |
| 0.1 | 0.90 |
| 0.5 | 0.61 |
| 1.0 | 0.37 |
| 2.0 | 0.14 |

**목표**: 받아들임 비율 60–90%을 얻으려면 $\Delta H \approx 0.1$–$1.0$이어야 한다.

---

## 8. 걸음 크기 고르기

### 안정 한계

개구리뜀에는 **안정 한계**가 있다. 곧 $\epsilon$이 문턱값을 넘으면 자취가 지수로 갈라져 흩어진다.

$\mathbf{M} = \mathbf{I}$인 이차 퍼텐셜 $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x}$에 대해:

$$
\epsilon < \frac{2}{\sqrt{\lambda_{\max}(\mathbf{A})}}
$$

여기서 $\lambda_{\max}$은 ($U$의 헤세 행렬인) $\mathbf{A}$의 가장 큰 고윳값이다.

**풀이**: 걸음 크기는 체계에서 가장 빠른 흔들림을 잡아낼 만큼 작아야 한다.

### 최적 걸음 크기

**너무 작으면**: 걸음이 많이 들고 셈이 버려지며 자기상관이 높다.

**너무 크면**: 에너지 오차가 커지고 받아들임이 떨어지며 (물리쳐진 제안으로) 셈이 버려진다.

**가장 좋게**: 살펴보기와 받아들임을 저울질한다. 경험으로는 받아들임 비율의 목표를 65–80%쯤으로 잡는다.

**차원에 따른 크기 변화**: 조건이 좋은 문제에서 최적 $\epsilon \sim d^{-1/4}$이다.

### 맞춰 가는 걸음 크기

달굼 동안 목표 받아들임을 이루도록 $\epsilon$을 다듬는다:

```python
def adapt_step_size(epsilon, accept_rate, target=0.65):
    if accept_rate > target:
        epsilon *= 1.1  # 더 큰 걸음을 감당할 수 있음
    else:
        epsilon *= 0.9  # 더 작은 걸음이 필요함
    return epsilon
```

더 정교한 방법으로는 Stan에서 쓰는 쌍대 평균내기(네스테로프, 2009)가 있다.

---

## 9. 수치적 안정성

### 기울기 터짐

$\nabla U(\mathbf{x})$이 아주 커지면(이를테면 경계 가까이나 두꺼운 꼬리에서) 운동량 새로 고치기가 걷잡을 수 없이 지나칠 수 있다.

**증상**:

- NaN이나 Inf 값
- 에너지가 엄청나게 크게 바뀜
- 받아들임 비율이 0

**손보기**:

- 걸음 크기를 줄인다
- 기울기를 잘라 낸다(조심해야 한다. 자세한 균형이 깨진다)
- 모형의 매개변수를 바꾼다
- 제약이 있는 매개변수에는 소프트플러스 같은 바꿈을 쓴다

### 수치 정밀도

$\epsilon$이 아주 작거나 자취가 아주 길면:

**쌓임 오차**: 되풀이해 더하면 정밀도를 잃을 수 있다. 필요하면 (카한의) 메우는 합을 써라.

**헤세 행렬의 조건수**: $\kappa = \lambda_{\max}/\lambda_{\min}$이 크면 방향마다 걸음 크기가 달라야 한다. 질량 행렬이 이를 다룬다.

---

## 10. 구현

### 기본 구현

```python
import numpy as np

def leapfrog(x, v, epsilon, L, grad_U, M_inv=None):
    """
    HMC를 위한 개구리뜀 적분기.
    
    인수:
        x: 첫 자리 (d,)
        v: 첫 운동량 (d,)
        epsilon: 걸음 크기
        L: 개구리뜀 걸음 수
        grad_U: 퍼텐셜 에너지의 기울기를 돌려주는 함수
        M_inv: 질량 행렬의 역행렬(기본값: 항등 행렬)
    
    반환값:
        x_new, v_new: 마지막 자리와 운동량
    """
    if M_inv is None:
        M_inv = np.eye(len(x))
    
    x = x.copy()
    v = v.copy()
    
    # 운동량의 반 걸음
    v = v - (epsilon / 2) * grad_U(x)
    
    # 온 걸음을 번갈아 밟기
    for i in range(L - 1):
        x = x + epsilon * (M_inv @ v)
        v = v - epsilon * grad_U(x)
    
    # 자리의 마지막 온 걸음
    x = x + epsilon * (M_inv @ v)
    
    # 운동량의 반 걸음
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

### 진단을 붙여서

```python
def leapfrog_with_diagnostics(x, v, epsilon, L, H, grad_U, M_inv=None):
    """진단을 위해 에너지를 기록하는 개구리뜀."""
    if M_inv is None:
        M_inv = np.eye(len(x))
    
    x = x.copy()
    v = v.copy()
    
    energies = [H(x, v)]
    
    v = v - (epsilon / 2) * grad_U(x)
    
    for i in range(L - 1):
        x = x + epsilon * (M_inv @ v)
        v = v - epsilon * grad_U(x)
        energies.append(H(x, v))
    
    x = x + epsilon * (M_inv @ v)
    v = v - (epsilon / 2) * grad_U(x)
    energies.append(H(x, v))
    
    return x, v, np.array(energies)
```

### 여러 사슬을 위한 벡터 꼴

```python
def leapfrog_vectorized(x, v, epsilon, L, grad_U, M_inv=None):
    """
    사슬 여럿을 위한 벡터 꼴 개구리뜀.
    
    인수:
        x: 자리 (n_chains, d)
        v: 운동량 (n_chains, d)
        ...
    """
    if M_inv is None:
        M_inv = np.eye(x.shape[1])
    
    x = x.copy()
    v = v.copy()
    
    v = v - (epsilon / 2) * grad_U(x)
    
    for i in range(L - 1):
        x = x + epsilon * (v @ M_inv.T)
        v = v - epsilon * grad_U(x)
    
    x = x + epsilon * (v @ M_inv.T)
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

---

## 11. 더 높은 차수의 적분기

### 사차 방법

단계를 더 두면 정확도를 더 높일 수 있다. 사차 심플렉틱 적분기는 다음과 같다:

$$
\phi_\epsilon^{(4)} = \phi_{c_1\epsilon} \circ \phi_{c_2\epsilon} \circ \phi_{c_3\epsilon} \circ \phi_{c_2\epsilon} \circ \phi_{c_1\epsilon}
$$

여기서 계수 $c_1, c_2, c_3$은 꼼꼼히 고른다(이를테면 포레스트-루스, 요시다).

**주고받음**: 걸음마다 기울기 값매김이 늘지만 걸음 크기를 더 크게 할 수 있다.

### 차수가 높으면 좋을 때

- 기울기가 싼 아주 매끄러운 과녁
- 높은 정확도가 필요할 때
- 받아들임 비율은 이미 높은데 표본이 서로 얽혀 있을 때

**대개는 값어치가 없다**: 실전에서는 $\epsilon$을 잘 맞춘 표준 개구리뜀을 이기기 어렵다.

---

## 12. 다른 적분기와 견주기

| 적분기 | 차수 | 심플렉틱 | 되돌릴 수 있음 | 걸음당 기울기 |
|------------|-------|------------|------------|----------------|
| 오일러 | 1 | ✗ | ✗ | 1 |
| 심플렉틱 오일러 | 1 | ✓ | ✗ | 1 |
| 개구리뜀(슈퇴르머-베를레) | 2 | ✓ | ✓ | 1 |
| RK4 | 4 | ✗ | ✗ | 4 |
| 요시다(사차) | 4 | ✓ | ✓ | 3 |

개구리뜀 적분기가 딱 알맞은 자리에 있다. 곧 심플렉틱하고 되돌릴 수 있으며 이차이고 걸음마다 기울기가 하나만 든다.

---

## 연습문제

1. **구현하고 확인하기**. 1차원 조화 떨개 $U(x) = \frac{1}{2}x^2$에 대해 개구리뜀을 구현하여라. 자취가 타원이고 에너지가 거의 지켜짐을 확인하여라.

2. **안정성 살피기**. 조화 떨개에 대해 한 걸음 개구리뜀 사상의 고윳값을 손으로 셈하여라. $|\lambda| = 1$(안정)이 되려면 $\epsilon < 2$이어야 함을 보여라.

3. **차수 확인하기**. 걸음 크기 $\epsilon, \epsilon/2, \epsilon/4$에 대해 오차를 셈하고 오차가 4배씩 줄어드는지 살펴 개구리뜀이 이차임을 수치로 확인하여라.

4. **에너지 오차의 통계**. HMC 자취를 많이 돌리고 $\Delta H$의 분포를 그려라. 그 분포는 $\epsilon$과 $L$에 어떻게 달려 있는가?

5. **더 높은 차수의 적분기**. 사차 요시다 적분기를 구현하고 셈 값이 같을 때 개구리뜀과 에너지 오차를 견주어라.

---

## 정리하며

| 성질 | 개구리뜀 적분기 |
|----------|-------------------|
| **차수** | 이차(전체), 삼차(그 자리) |
| **심플렉틱** | 예 — 위상 공간 부피를 지킨다 |
| **되돌릴 수 있음** | 예 — 운동량을 뒤집으면 |
| **에너지 오차** | $O(\epsilon^2)$, 묶여 있고 쏠리지 않음 |
| **걸음당 기울기** | 1(반 걸음을 합친 뒤) |
| **안정 한계** | $\epsilon < 2/\sqrt{\lambda_{\max}}$ |

개구리뜀 적분기는 심플렉틱함, 되돌릴 수 있음, 효율을 함께 갖춘 덕분에 HMC의 표준 선택이 되었다. 에너지 오차가 묶여 있어 자취를 길게 하면서도 받아들임 비율을 높게 지킬 수 있다.

---

**참고 문헌**

1. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
2. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration* (2nd ed.). Springer.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. Blanes, S., Casas, F., & Sanz-Serna, J. M. (2014). "Numerical Integrators for the Hybrid Monte Carlo Method." *SIAM Journal on Scientific Computing*.
