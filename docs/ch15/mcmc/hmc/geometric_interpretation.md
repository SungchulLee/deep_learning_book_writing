# HMC의 기하 풀이
해밀턴 몬테카를로는 알고리즘이 왜 되는지를 밝혀 주고 나아갈 길을 일러 주는 풍성한 기하 풀이를 허락한다. 이 절에서는 미분 기하, 정보 기하, 물리 직관이라는 세 렌즈로 HMC를 살펴본다.

---

## 표집의 기하

### 기하가 왜 중요한가

분포 $\pi(\mathbf{x})$에서 표집하는 일은 본디 기하 문제이다:

- (확률의 대부분이 놓인) **전형 집합**은 기하 대상이다
- **살펴보기**는 이 기하를 효율적으로 헤쳐 나가는 일이다
- 로그 밀도의 **굽음**이 표집기의 굶에 영향을 준다
- 매개변수 공간의 **거리**는 통계의 거리를 담아야 한다

HMC의 힘은 이 기하를 지키고 써먹는 데서 나온다.

### 기하로 보는 세 눈

| 보는 눈 | 핵심 대상 | 통찰 |
|-------------|-----------|---------|
| **위상 공간** | 심플렉틱 다양체 | 부피 지킴, 에너지 지킴 |
| **정보 기하** | 피셔-라오 계량 | 확률 분포 위의 자연스러운 거리 |
| **물리** | 에너지 지형 | 움직임과 살펴보기에 대한 직관 |

---

## 위상 공간의 기하

### 심플렉틱 다양체

위상 공간 $(\mathbf{x}, \mathbf{v}) \in \mathbb{R}^{2d}$은 닫혀 있고 찌부러지지 않는 2형식인 **심플렉틱 짜임**을 지닌다:

$$
\omega = \sum_{i=1}^{d} dv_i \wedge dx_i
$$

이 짜임은 해밀턴 움직임이 지킨다:

$$
\phi_t^* \omega = \omega
$$

여기서 $\phi_t$은 시간 $t$의 흐름이다.

### 심플렉틱 짜임이 담고 있는 것

심플렉틱 형식은 다음을 담고 있다:

1. **푸아송 괄호**: $\{f, g\} = \omega(X_f, X_g)$
2. **해밀턴 방정식**: $\mathbf{J} = \omega^{-1}$일 때 $\dot{\mathbf{z}} = \mathbf{J} \nabla H$
3. **지킴 법칙**: $\{f, H\} = 0$인 함수는 지켜진다
4. **부피 요소**: $\omega^d$이 리우빌 측도를 준다

### 지킴의 기하 뜻

**에너지 지킴**: 자취가 등위 집합 $H = E$ 위에 놓인다.

**부피 지킴**: 그 흐름은 리우빌 측도의 등거리 사상이다.

**따라 나오는 것**: 이 성질들 덕분에 $\pi(\mathbf{x}, \mathbf{v}) \propto e^{-H}$에서 뽑은 표본으로 시작하면 올바른 분포에 머문다.

---

## 에너지 지형 그려 보기

### 퍼텐셜 에너지 면

퍼텐셜 에너지 $U(\mathbf{x}) = -\log \tilde{\pi}(\mathbf{x})$이 매개변수 공간 위의 면을 정한다:

- **골짜기**($U$이 낮음): 확률이 높은 구역
- **마루**($U$이 높음): 확률이 낮은 구역
- **그 자리 최솟값**: 분포의 봉우리
- **안장점**: 봉우리 사이를 잇는 구역

### 등고선 그림

2차원 분포에서 $U$이 일정한 등고선(곧 확률 밀도가 일정한 선)이 기하를 드러낸다:

```
           U increasing
               ↑
        ┌──────────────┐
        │    ╭───╮     │
        │  ╭─┤   ├─╮   │  Nested contours
        │ ╭┤ │ • │ ├╮  │  • = mode (minimum U)
        │  ╰─┤   ├─╯   │
        │    ╰───╯     │
        └──────────────┘
```

HMC 자취는 (에너지가 일정한 채로) 이 등고선을 거의 따라가며, 운동량이 등고선을 따라가는 "빠르기"를 정한다.

### 전형 집합

차원이 높으면 기하가 직관에 어긋난다:

**봉우리**: 밀도가 가장 큰 점($U$이 가장 작은 점).

**전형 집합**: 부피 × 밀도가 가장 커지는, $U$이 중간인 얇은 껍질.

$d$차원 가우스에서:

- 봉우리는 평균에 있다
- 전형 집합은 평균에서 반지름이 $\approx \sqrt{d}$인 껍질이다
- 껍질의 부피는 $d$에 따라 지수로 자란다

**뜻하는 바**: 좋은 표집기는 봉우리만 찾을 것이 아니라 전형 집합을 살펴봐야 한다.

---

## 정보 기하

### 피셔 정보 계량

확률 분포의 공간 위에서 **피셔 정보**가 자연스러운 계량을 정한다:

$$
g_{ij}(\boldsymbol{\theta}) = \mathbb{E}_{\pi_\theta}\left[\frac{\partial \log \pi_\theta}{\partial \theta_i} \frac{\partial \log \pi_\theta}{\partial \theta_j}\right] = -\mathbb{E}_{\pi_\theta}\left[\frac{\partial^2 \log \pi_\theta}{\partial \theta_i \partial \theta_j}\right]
$$

이는 로그 가능도의 음수의 **기댓값 헤세 행렬**이다.

### 피셔 계량과 HMC

$\pi(\mathbf{x})$에서 표집할 때 "매개변수"는 값 $\mathbf{x}$ 자체이다. 피셔 계량은 다음이 된다:

$$
\mathbf{G}(\mathbf{x}) = -\nabla^2 \log \pi(\mathbf{x}) = \nabla^2 U(\mathbf{x})
$$

곧 퍼텐셜 에너지의 헤세 행렬이다.

**질량 행렬과의 이음**: 가장 좋은 질량 행렬은 다음을 만족한다:

$$
\mathbf{M}^{-1} \approx \mathbb{E}[\mathbf{G}(\mathbf{x})] = \mathbb{E}[\nabla^2 U(\mathbf{x})]
$$

가우스 과녁에서는 $\mathbf{M}^{-1} = \boldsymbol{\Sigma}^{-1}$, 곧 $\mathbf{M} = \boldsymbol{\Sigma}$이 된다.

### 측지선과 표집

리만 기하에서 **측지선**은 길이가 가장 짧은 곡선이다. 정보 기하에서는:

- 측지선은 분포 사이의 가장 효율적인 길을 나타낸다
- (알맞은 질량 행렬을 쓴) HMC 자취는 측지선을 어림한다
- 이것이 HMC가 왜 효율적으로 살펴보는지를 말해 준다

### 리만 HMC

**리만 HMC**은 자리에 달린 계량 $\mathbf{G}(\mathbf{x})$을 쓴다:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{G}(\mathbf{x})^{-1} \mathbf{v} + \frac{1}{2}\log|\mathbf{G}(\mathbf{x})|
$$

이는 그 자리 굽음에 맞춰 가므로 기하가 달라지는 구역에서 표집을 낫게 할 수 있다.

---

## 자취와 궤도

### 돌림으로 본 해밀턴 흐름

$\mathbf{M} = \mathbf{I}$인 이차 퍼텐셜 $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x}$에서 해밀턴 방정식은 다음을 준다:

$$
\frac{d}{dt}\begin{pmatrix} \mathbf{x} \\ \mathbf{v} \end{pmatrix} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{A} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{x} \\ \mathbf{v} \end{pmatrix}
$$

그 풀이에는 위상 공간에서의 **돌림**이 들어 있다:

$$
\begin{pmatrix} \mathbf{x}(t) \\ \mathbf{v}(t) \end{pmatrix} = \exp\left(t\begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{A} & \mathbf{0} \end{pmatrix}\right) \begin{pmatrix} \mathbf{x}(0) \\ \mathbf{v}(0) \end{pmatrix}
$$

### 궤도의 기하

**1차원 조화 떨개**($U(x) = \frac{1}{2}\omega^2 x^2$):

자취는 $(x, v)$ 공간의 타원이다:

$$
\frac{\omega^2 x^2}{2E} + \frac{v^2}{2E} = 1
$$

**주기**: $T = 2\pi/\omega$이다. 시간 $T$ 뒤에 자취는 시작점으로 돌아온다.

**일반 이차식**: 알맞게 돌린 좌표에서 자취는 타원이며 주기는 $\mathbf{A}$의 고윳값이 정한다.

### 이차가 아닌 퍼텐셜

일반 $U(\mathbf{x})$에서 자취는 다음일 수 있다:

- **거의 되풀이됨**: 원환면을 빽빽이 채우지만 결코 똑같이 되풀이하지는 않는다
- **어지러움**: 첫 조건에 예민하게 달려 있다(흔한 뒤확률에서는 드물다)
- **복잡함**: 시간 규모가 여럿일 수 있다

NUTS의 유턴 잣대는 되풀이됨을 요구하지 않고도 자취가 "넉넉히" 살펴보았는지를 알아낸다.

---

## 그림자 해밀턴 함수

### 거꾸로 오차 살피기

개구리뜀 적분기는 $H$에 대한 해밀턴 방정식을 정확히 풀지 않는다. 그 대신 **고친 해밀턴 함수**를 **정확히** 푼다:

$$
\tilde{H}(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, \mathbf{v}) + \epsilon^2 H_2(\mathbf{x}, \mathbf{v}) + \epsilon^4 H_4(\mathbf{x}, \mathbf{v}) + \cdots
$$

이 $\tilde{H}$이 **그림자 해밀턴 함수**이다.

### 기하학적 해석

- 수치 자취는 $\tilde{H}$의 등위 집합 위에 정확히 놓인다
- 에너지 오차 $|H - \tilde{H}| = O(\epsilon^2)$은 묶여 있다
- 자취는 $\tilde{H} = \text{const}$인 면을 살펴보는데, 이는 $H = \text{const}$인 면과 $O(\epsilon^2)$만큼 가깝다

이것이 개구리뜀의 에너지 오차가 쏠림 없이 묶여 있는 까닭이다.

### 표집에 뜻하는 바

$\tilde{H} \approx H$이므로 개구리뜀 자취는 올바른 에너지 면을 거의 살펴본다. $O(\epsilon^2)$만큼의 어긋남은 MH 바로잡기가 헤아린다.

---

## 질량 행렬을 기하로 보기

### 계량으로 풀이하기

질량 행렬 $\mathbf{M}$이 운동량 공간의 계량을 정한다:

$$
\|\mathbf{v}\|_{\mathbf{M}}^2 = \mathbf{v}^T \mathbf{M} \mathbf{v}
$$

그 역행렬 $\mathbf{M}^{-1}$은 (자리 공간에 접하는) 속도 공간의 계량을 정한다:

$$
\|\dot{\mathbf{x}}\|_{\mathbf{M}^{-1}}^2 = \dot{\mathbf{x}}^T \mathbf{M}^{-1} \dot{\mathbf{x}}
$$

### 하얗게 만드는 바꿈

(가우스 과녁에 가장 좋은) $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$을 쓸 때 다음을 정한다:

$$
\tilde{\mathbf{x}} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu}), \quad \tilde{\mathbf{v}} = \boldsymbol{\Sigma}^{1/2}\mathbf{v}
$$

이 좌표에서는:

- 과녁이 $\mathcal{N}(\mathbf{0}, \mathbf{I})$이 된다
- 운동 에너지가 $\frac{1}{2}|\tilde{\mathbf{v}}|^2$이 된다
- 움직임이 방향에 고르게 된다

질량 행렬은 사실상 과녁 분포를 **하얗게 만든다**.

### 조건수

**조건수** $\kappa = \lambda_{\max}(\mathbf{M}^{-1}\mathbf{A})/\lambda_{\min}(\mathbf{M}^{-1}\mathbf{A})$은 실효 분포가 얼마나 "둥근지"를 잰다.

- $\kappa = 1$: 조건이 완벽하다(구면이다)
- $\kappa \gg 1$: 조건이 나쁘다(길쭉하다)

가장 좋은 질량 행렬은 $\kappa = 1$을 이룬다.

---

## 굽음과 표집의 어려움

### 가우스 굽음

2차원 분포에서 로그 밀도 면의 **가우스 굽음**은 다음과 같다:

$$
K = \frac{\det(\mathbf{H})}{\left(1 + |\nabla \log \pi|^2\right)^2}
$$

여기서 $\mathbf{H} = \nabla^2 \log \pi$은 헤세 행렬이다.

### 굽음이 HMC에 주는 영향

| 굽음 | 기하 | HMC의 굶 |
|-----------|----------|--------------|
| 고르게 양수 | 사발 꼴 | 쉽고 안정된 자취 |
| 양수인데 달라짐 | 달라지는 사발 | 맞춰 가는 방법이 필요할 수 있음 |
| 부호가 섞임 | 안장 구역 | 자취가 갈라져 나갈 수 있음 |
| 0에 가까움 | 평평한 구역 | 느린 섞임 |

### 깔때기 기하

**깔때기**는 굽음이 크게 달라지는 병든 기하이다:

$$
y \sim \mathcal{N}(0, \sigma_y^2), \quad x | y \sim \mathcal{N}(0, e^{y})
$$

- $y$이 크면: $x$ 쪽으로 넓고 굽음이 작다
- $y$이 작으면: $x$ 쪽으로 좁고 굽음이 크다

**말썽**: 걸음 크기 하나로는 어디서나 잘 들을 수 없다. 좁은 구역에 맞춰 $\epsilon$을 작게 하면 넓은 구역을 살펴보기가 아주 느려진다.

**풀이**: 리만 HMC, 매개변수 바꾸기, 또는 꼼꼼한 맞추기.

---

## 그려 보는 방법

### 위상 그림(2차원)

$d = 1$일 때 $(x, v)$ 공간에 자취를 그린다:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait(U, grad_U, x_range, v_range, n_traj=10):
    # 에너지 등고선
    x = np.linspace(*x_range, 100)
    v = np.linspace(*v_range, 100)
    X, V = np.meshgrid(x, v)
    H = U(X) + 0.5 * V**2
    
    plt.contour(X, V, H, levels=20, alpha=0.5)
    
    # 자취 표집
    for _ in range(n_traj):
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        
        # 자취 흉내내기
        xs, vs = [x0], [v0]
        x_curr, v_curr = x0, v0
        
        for _ in range(200):
            # 개구리뜀 걸음
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            x_curr = x_curr + 0.1 * v_curr
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            xs.append(x_curr)
            vs.append(v_curr)
        
        plt.plot(xs, vs, 'b-', alpha=0.3, linewidth=0.5)
    
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('Phase Portrait')
```

### 자취를 따라가는 에너지

에너지 지킴을 눈으로 보려고 $H(t)$을 그린다:

```python
def plot_energy_trajectory(x0, v0, U, grad_U, n_steps=100, epsilon=0.1):
    x, v = x0.copy(), v0.copy()
    energies = [U(x) + 0.5 * np.dot(v, v)]
    
    for _ in range(n_steps):
        v = v - (epsilon/2) * grad_U(x)
        x = x + epsilon * v
        v = v - (epsilon/2) * grad_U(x)
        energies.append(U(x) + 0.5 * np.dot(v, v))
    
    plt.plot(energies)
    plt.xlabel('Leapfrog Step')
    plt.ylabel('Hamiltonian H')
    plt.title('Energy Conservation')
    plt.axhline(energies[0], color='r', linestyle='--', label='Initial')
```

### 매개변수 공간의 자취

$d = 2$일 때 자취를 $(x_1, x_2)$에 쏘아 내려 그린다:

```python
def plot_trajectory_2d(samples, target_log_prob=None):
    plt.plot(samples[:, 0], samples[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(samples[0, 0], samples[0, 1], 'go', markersize=10, label='Start')
    plt.plot(samples[-1, 0], samples[-1, 1], 'ro', markersize=10, label='End')
    
    if target_log_prob is not None:
        x = np.linspace(samples[:, 0].min() - 1, samples[:, 0].max() + 1, 100)
        y = np.linspace(samples[:, 1].min() - 1, samples[:, 1].max() + 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[target_log_prob(np.array([xi, yi])) 
                       for xi, yi in zip(x_row, y_row)]
                      for x_row, y_row in zip(X, Y)])
        plt.contour(X, Y, np.exp(Z), levels=10, alpha=0.5, colors='gray')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
```

---

## NUTS을 기하로 보기

### 기하 잣대로 본 유턴

유턴 조건은

$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0
$$

기하로 풀이할 수 있다. 곧 자취가 출발점 쪽으로 **되굽기** 시작했다는 뜻이다.

1차원의 $(x, v)$ 평면에서:

- 자취가 호를 그린다
- 호가 뒤로 굽기 시작할 때 유턴이 일어난다
- 이는 "궤도"의 대략 반에 해당한다

### 살펴보기로 본 나무 세우기

NUTS의 곱절 늘리는 나무는 자취를 차근차근 살펴본다:

```
        Depth 0         Depth 1              Depth 2
           •       →    •───•        →    •───•───•───•
                      backward           forward extension
```

곱절 늘릴 때마다 살펴본 위상 공간 구역이 넓어지며 여러 규모에서 유턴을 살핀다.

---

## 다른 방법과의 이음

### 랑주뱅 움직임

**과감쇠 랑주뱅**(일차):

$$
d\mathbf{x} = \nabla \log \pi(\mathbf{x}) \, dt + \sqrt{2} \, d\mathbf{W}
$$

**해밀턴 움직임**(이차):

$$
d\mathbf{x} = \mathbf{v} \, dt, \quad d\mathbf{v} = \nabla \log \pi(\mathbf{x}) \, dt
$$

핵심 차이는 이렇다. HMC에는 한 방향으로 이어지는 움직임을 주는 **운동량**이 있고, 랑주뱅에는 퍼짐을 일으키는 **잡음**이 있다.

기하로 말하면 HMC는 위상 공간에서 정해진 곡선을 따라가고, 랑주뱅은 자리 공간에서 확률로 흔들리는 길을 따라간다.

### 자연 기울기 내리기

**자연 기울기 내리기**는 피셔 계량을 쓴다:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta \mathbf{G}(\boldsymbol{\theta}_t)^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_t)
$$

가장 좋은 질량 행렬을 쓴 HMC는 이것의 확률 판을 이루되, 에너지 지킴 덕분에 한 점으로 모이지 않는다.

---

## 요약

| 기하 개념 | HMC에서 하는 일 |
|------------------|-------------|
| **심플렉틱 짜임** | 부피 지킴을 보장한다 |
| **에너지 면** | 자취를 옭아맨다 |
| **피셔 계량** | 가장 좋은 질량 행렬의 까닭이 된다 |
| **굽음** | 표집의 어려움을 정한다 |
| **그림자 해밀턴 함수** | 에너지 오차가 묶이는 까닭을 말해 준다 |
| **측지선** | HMC가 효율적인 길을 어림한다 |

기하로 보면 HMC가 자연스러운 알고리즘임이 드러난다. 곧 확률 분포의 본디 기하를 지키고, 물리에서 비롯한 움직임을 쓰며, 지킴 법칙을 써먹어 효율을 얻는다. 이 기하를 이해하면 이론 살피기와 실전 개선이 모두 길을 얻는다.

---

## 참고 문헌

1. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
2. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
3. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
4. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
5. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.

## 연습문제

1. **위상 그림**. 두 우물 퍼텐셜 $U(x) = (x^2 - 1)^2$의 위상 그림을 그려라. 우물 하나를 맴도는 것과 둘을 넘나드는 것의 경계인 가름선을 찾아라.

2. **하얗게 만들기 그려 보기**. 서로 얽힌 2차원 가우스에 대해 가장 좋은 질량 행렬이 이끌어 내는 하얗게 만드는 바꿈의 앞뒤 자취를 그려라.

3. **굽음 셈하기**. 2차원 가우스 섞음에 대해 헤세 행렬과 가우스 굽음을 셈하여라. 굽음은 매개변수 공간에서 어떻게 달라지는가?

4. **그림자 해밀턴 함수**. 개구리뜀 자취에 $\tilde{H} = H + \epsilon^2 H_2$을 맞추어 그림자 해밀턴 함수를 수치로 어림하여라. $\tilde{H}$이 $H$보다 더 잘 지켜지는지 확인하여라.

5. **깔때기 기하**. 깔때기 분포를 그리고 걸음 크기를 여러 가지로 하여 HMC를 돌려라. 갈라져 나감이 어디서 일어나는지 찾고 그 자리 굽음과 이어 보아라.

---
