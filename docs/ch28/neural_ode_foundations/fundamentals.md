# 상미분 방정식 바탕
## 학습 목표

이 마디를 마치면 다음을 하게 된다:

- 상미분 방정식을 이어진 때의 움직임 계로 이해한다
- 오일러와 룽게-쿠타를 담은 수치 적분 방법을 익힌다
- 남은 이음 신경망과 상미분 방정식을 띄엄띄엄하게 만들기 사이의 깊은 이음을 알아본다
- 파이토치에서 기본 상미분 방정식 풀개를 바닥부터 짠다
- 위상 그림과 풀이 자취를 그려 본다
- `torchdiffeq`으로 온전한 신경 상미분 방정식 모델을 세운다
- 가름과 되돌이 맞춤 일에 신경 상미분 방정식을 익힌다

## 미리 알아야 할 것

- 여러 변수 미적분(미분, 적분)
- 기본 선형 대수(벡터, 행렬)
- 파이토치 바탕(텐서, 자동 미분)
- 남은 이음 신경망에 익숙함(도움이 되지만 꼭 필요하지는 않다)

---

## 1. 상미분 방정식이란 무엇인가?

**상미분 방정식(ODE)**은 어떤 양이 때에 따라 어떻게 바뀌는지 적어 준다. "상(常)"이라는 말은 이를 편미분 방정식과 갈라 준다. 상미분 방정식은 변수 하나(보통 때)에 대한 미분만 담는다.

### 1.1 수학으로 적기

일반적인 일차 상미분 방정식은 다음 꼴이다:

$$\frac{dy}{dt} = f(y, t)$$

여기서 각 기호는 다음과 같다.

- $y(t) \in \mathbb{R}^d$은 때 $t$의 **상태**이다
- $f: \mathbb{R}^d \times \mathbb{R} \rightarrow \mathbb{R}^d$은 **움직임 함수**이다
- $\frac{dy}{dt}$은 순간 바뀜 빠르기를 나타낸다

**첫값 문제(IVP)**는 시작 조건을 더한다:

$$\frac{dy}{dt} = f(y, t), \quad y(t_0) = y_0$$

목표는 처음 상태 $y_0$이 주어질 때 모든 $t > t_0$에 대해 $y(t)$을 찾는 것이다.

### 1.2 홀로 도는 계와 그렇지 않은 계

**홀로 도는 상미분 방정식**은 움직임이 때에 매이지 않는다:

$$\frac{dy}{dt} = f(y)$$

벡터 마당 $f$은 지금 상태에만 매이고 언제 보는지에는 매이지 않는다. 많은 물리 계가 이 때 불변 성질을 보인다.

**홀로 돌지 않는 상미분 방정식**은 때에 드러나게 매인다:

$$\frac{dy}{dt} = f(y, t)$$

바깥의 밀어붙임, 때에 따라 바뀌는 매개변수, 정해진 개입이 홀로 돌지 않는 움직임을 낳는다.

!!! info "신경 상미분 방정식의 약속"
    신경 상미분 방정식은 보통 $f$이 신경망인 홀로 돌지 않는 적기를 쓴다. 신경망 얼개가 $t$을 드러나게 쓰지 않더라도 그것에 닿을 수 있으면 때에 매인 바꿈이 가능해지고 나타내기가 더 너그러워진다. `torchdiffeq`은 `f(y, t)`이 아니라 `f(t, y)` 꼴을 바란다는 점에 유의하라.

### 1.3 있음과 하나뿐임

**피카르-린델뢰프 정리**는 $f$이 다음이면 유일한 풀이가 있음을 보장한다:

1. 두 인자 모두에 대해 **이어짐**
2. $y$에 대해 **립시츠 이어짐**: $\|f(y_1, t) - f(y_2, t)\| \leq L\|y_1 - y_2\|$

립시츠 조건은 풀이가 "터지거나" 서로 가로지르는 것을 막는다. 가둬진 깨움(tanh, 시그모이드)을 쓴 신경망은 이 조건을 자연스럽게 만족하지만 가둬지지 않은 깨움(ReLU)은 어길 수 있다. 이는 신경 상미분 방정식 얼개에서 깨움 함수를 고르는 데 곧바로 뜻이 있다:

| 깨움 | 립시츠 | 좋은 점 | 나쁜 점 |
|------------|-----------|------|------|
| **Tanh** | ✓ 가둬짐 | 안정된 상미분 방정식 움직임, 은근한 규칙 세우기 | 포화, 사라지는 기울기 |
| **Softplus** | ✗ 가둬지지 않음 | 매끄럽고 포화하지 않음 | 자취가 터질 수 있음 |
| **ReLU** | ✗ 매끄럽지 않음 | 셈이 빠름 | 립시츠가 아니어서 상미분 방정식 이론을 어김 |
| **GELU/SiLU** | ✓ 가둬짐 | 매끄럽고 나타냄 힘이 큼 | 매김마다 셈이 더 듦 |

---

## 2. 고전 상미분 방정식 보기

고전 상미분 방정식을 이해하면 신경망 움직임에 대한 직관이 생긴다.

### 2.1 지수 자람과 사그라짐

가장 단순한 상미분 방정식:

$$\frac{dy}{dt} = ky, \quad y(0) = y_0$$

**닫힌 꼴 풀이:** $y(t) = y_0 e^{kt}$

- $k > 0$: 지수 자람(인구 움직임, 복리)
- $k < 0$: 지수 사그라짐(방사성 붕괴, 식음)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def exponential_ode(y: torch.Tensor, t: float, k: float = 1.0) -> torch.Tensor:
    """
    지수 늘어남과 줄어듦: dy/dt = k*y
    
    인수:
        y: 이제 상태 (batch_size, dim)
        t: 이제 때(이 자율 상미분 방정식에서는 쓰지 않는다)
        k: 자람 빠르기 매개변수
        
    반환값:
        바뀜 빠르기 dy/dt
    """
    return k * y

# 확인을 위한 닫힌 꼴 풀이
def exponential_analytical(y0: torch.Tensor, t: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """해석 풀이: y(t) = y_0 * exp(k*t)"""
    return y0 * torch.exp(k * t)
```

### 2.2 조화 떨개

이차 상미분 방정식을 일차 계로 바꾼 것:

$$\frac{d^2x}{dt^2} + 2\zeta\omega_0\frac{dx}{dt} + \omega_0^2 x = 0$$

상태 벡터 $y = [x, \dot{x}]^T$(자리와 빠르기)을 뜻매김한다:

$$\frac{dy}{dt} = \begin{bmatrix} y_1 \\ -\omega_0^2 y_0 - 2\zeta\omega_0 y_1 \end{bmatrix}$$

여기서 $\omega_0$은 고유 진동수이고 $\zeta$은 감쇠비이다($\zeta < 1$: 덜 감쇠, $\zeta = 1$: 임계 감쇠, $\zeta > 1$: 지나친 감쇠).

```python
def damped_oscillator(y: torch.Tensor, t: float, 
                      omega_0: float = 2.0, zeta: float = 0.1) -> torch.Tensor:
    """
    일차 계로 나타낸 감쇠 조화 떨개.
    
    상태: y = [자리, 빠르기]
    
    인수:
        y: 상태 텐서 (batch_size, 2)
        t: 지금 때
        omega_0: 고유 진동수
        zeta: 감쇠비
        
    반환값:
        Derivative [dx/dt, dv/dt]
    """
    position = y[..., 0:1]
    velocity = y[..., 1:2]
    
    dxdt = velocity
    dvdt = -omega_0**2 * position - 2 * zeta * omega_0 * velocity
    
    return torch.cat([dxdt, dvdt], dim=-1)
```

### 2.3 로트카-볼테라 잡아먹개와 먹이

되풀이 움직임을 보이는 고전 비선형 계:

$$\frac{dx}{dt} = \alpha x - \beta xy$$

$$\frac{dy}{dt} = \delta xy - \gamma y$$

여기서 $x$은 먹이의 수이고 $y$은 잡아먹개의 수이다.

```python
def lotka_volterra(state: torch.Tensor, t: float,
                   alpha: float = 1.5, beta: float = 1.0,
                   gamma: float = 3.0, delta: float = 1.0) -> torch.Tensor:
    """
    로트카-볼테라 잡아먹개-먹이 움직임.
    
    인수:
        state: [먹이, 사냥꾼]의 수
        t: Time (unused)
        alpha: 먹이의 자람 빠르기
        beta: 잡아먹는 빠르기
        gamma: 잡아먹개의 죽음 빠르기
        delta: 잡아먹기 효율
        
    반환값:
        수의 미분
    """
    x, y = state[..., 0:1], state[..., 1:2]
    
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    
    return torch.cat([dxdt, dydt], dim=-1)
```

---

## 3. 수치 적분 방법

닫힌 꼴 풀이가 없으면(대부분이 그렇다) 수치 적분에 기댄다.

### 3.1 앞으로 오일러 방법

가장 단순한 수치 적분개는 미분을 앞 차분으로 어림한다:

$$y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$$

**이끌어 내기:** 테일러 펼침이 $y(t + \Delta t) = y(t) + \Delta t \cdot \frac{dy}{dt} + O(\Delta t^2)$을 준다. 일차에서 자르고 상미분 방정식을 넣으면 오일러 고침이 나온다.

```python
def euler_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    앞으로 오일러 방법의 한 걸음.
    
    수학으로 나타내기:
        y_{n+1} = y_n + dt * f(y_n, t_n)
    
    이는 일차 방법이다. 그 자리 어긋남 O(dt²), 온 어긋남 O(dt).
    
    인수:
        f: ODE function dy/dt = f(y, t)
        y: 지금 상태
        t: 지금 때
        dt: 때 걸음 크기
        
    반환값:
        Next state y_{n+1}
    """
    return y + dt * f(y, t)


def euler_integrate(f, y0: torch.Tensor, t_span: tuple, dt: float):
    """
    앞으로 오일러 방법으로 상미분 방정식을 적분한다.
    
    인수:
        f: 상미분 방정식 함수
        y0: 첫 상태 (batch_size, dim)
        t_span: (처음 때, 끝 때)
        dt: 때 걸음
        
    반환값:
        t_values: 때 점
        y_values: 때 점마다의 상태
    """
    t_start, t_end = t_span
    t_values = torch.arange(t_start, t_end + dt, dt)
    n_steps = len(t_values)
    
    # 자취 담을 곳을 첫자리매김한다
    y_values = torch.zeros(n_steps, *y0.shape)
    y_values[0] = y0
    
    # 적분 되풀이
    y = y0
    for i in range(n_steps - 1):
        y = euler_step(f, y, t_values[i].item(), dt)
        y_values[i + 1] = y
    
    return t_values, y_values
```

### 3.2 어긋남 살피기

**그 자리의 자름 어긋남:** 처음 조건이 완벽하다고 할 때 한 걸음에서 생기는 어긋남:

$$\text{LTE} = y(t + \Delta t) - \left[y(t) + \Delta t \cdot f(y(t), t)\right] = O(\Delta t^2)$$

**온 어긋남:** 온 적분에 걸쳐 쌓인 어긋남:

$$\text{Global Error} = O(\Delta t)$$

어긋남이 $O(1/\Delta t)$걸음에 걸쳐 쌓이므로 온 어긋남은 차수가 하나 낮다.

!!! warning "신경망에 대한 뜻"
    어긋남의 차수는 깊은 뜻을 지닌다. 걸음 크기 $\Delta t = 1$으로 층 $L$개를 쓴 남은 이음 신경망은 온 어긋남이 $O(1)$이다. 곧 띄엄띄엄하게 만든 어긋남이 깊이와 상관없이 *붙박여* 있다. 맞추어 가는 풀개를 쓴 신경 상미분 방정식은 걸음을 더 밟아 어긋남을 얼마든지 작게 할 수 있다.

### 3.3 안정성 살피기

모든 걸음 크기가 통하지는 않는다. $\lambda < 0$(사그라짐)인 시험 방정식 $\frac{dy}{dt} = \lambda y$을 살펴보자.

오일러는 $y_{n+1} = (1 + \lambda \Delta t) y_n$을 준다

안정하려면 $|1 + \lambda \Delta t| < 1$이어야 하고 이는 다음을 요구한다:

$$\Delta t < \frac{2}{|\lambda|}$$

$\lambda$이 크고 음수이면(뻣뻣한 계) 아주 작은 때 걸음이 필요하다. 그래서 맞추어 가는 풀개가 꼭 필요하다.

### 3.4 4차 룽게-쿠타(RK4)

RK4은 여러 점에서 $f$을 매겨 4차 정확도를 이룬다:

$$k_1 = f(y_n, t_n)$$

$$k_2 = f\left(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$

$$k_3 = f\left(y_n + \frac{\Delta t}{2}k_2, t_n + \frac{\Delta t}{2}\right)$$

$$k_4 = f(y_n + \Delta t \cdot k_3, t_n + \Delta t)$$

$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

```python
def rk4_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    4차 룽게-쿠타 방법의 한 걸음.
    
    Local error: O(dt^5)
    Global error: O(dt^4)
    
    인수:
        f: 상미분 방정식 함수
        y: 지금 상태
        t: 지금 때
        dt: 때 걸음
        
    반환값:
        다음 상태
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_integrate(f, y0: torch.Tensor, t_span: tuple, dt: float):
    """RK4 방법으로 상미분 방정식을 적분한다."""
    t_start, t_end = t_span
    t_values = torch.arange(t_start, t_end + dt, dt)
    n_steps = len(t_values)
    
    y_values = torch.zeros(n_steps, *y0.shape)
    y_values[0] = y0
    
    y = y0
    for i in range(n_steps - 1):
        y = rk4_step(f, y, t_values[i].item(), dt)
        y_values[i + 1] = y
    
    return t_values, y_values
```

### 3.5 맞추어 가는 걸음 크기 방법

실제 쓰는 신경 상미분 방정식 짜기는 `dopri5`(도맨드-프린스) 같은 **맞추어 가는 풀개**를 쓰며 이는:

1. 차수가 다른 풀이를 견주어 그 자리의 어긋남을 어림한다
2. 어긋남이 허용 오차를 넘으면 그 걸음을 물리친다
3. 걸음 크기를 저절로 맞춘다

```python
from torchdiffeq import odeint

# torchdiffeq의 맞추어 가는 풀개를 쓴다
def integrate_adaptive(f, y0, t_eval, method='dopri5', rtol=1e-7, atol=1e-9):
    """
    맞추어 가는 걸음 크기 다스리기로 상미분 방정식을 적분한다.
    
    인수:
        f: 상미분 방정식 함수(torchdiffeq은 (t, y) 차례로 받아야 한다)
        y0: 처음 상태
        t_eval: 풀이를 돌려줄 때
        method: 푸는 방법('dopri5', 'rk4', 'euler' 따위)
        rtol: 상대 허용 오차
        atol: 절대 허용 오차
        
    반환값:
        물어본 때의 풀이
    """
    return odeint(f, y0, t_eval, method=method, rtol=rtol, atol=atol)
```

---

## 4. 남은 이음 신경망과 상미분 방정식의 이음

이것이 신경 상미분 방정식의 까닭이 되는 핵심 통찰이다.

### 4.1 오일러로 띄엄띄엄하게 만든 것으로서의 남은 이음 신경망

**남은 이음 층**은 다음을 셈한다:

$$h_{l+1} = h_l + f_\theta(h_l)$$

이는 $\Delta t = 1$인 오일러 방법과 *꼭 같다*:

| 조각 | 오일러 방법 | 남은 이음 신경망 |
|-----------|-------------|--------|
| 상태 | $y_n$ | $h_l$ |
| 움직임 | $f(y_n, t_n)$ | $f_\theta(h_l)$ |
| 고침 | $y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$ | $h_{l+1} = h_l + f_\theta(h_l)$ |
| 걸음 크기 | $\Delta t$ | $1$(은근히) |

남은 이음이 배운 움직임 함수의 수치 적분을 짠다.

### 4.2 이어진 끝으로 가기

층의 개수 $L \to \infty$이고 걸음 크기 $\Delta t \to 0$이면:

$$\lim_{L \to \infty, \Delta t \to 0} h_L = h(T) \quad \text{where} \quad \frac{dh}{dt} = f_\theta(h(t), t)$$

띄엄띄엄한 층 번호가 이어진 때가 되고 층마다의 바꿈이 이어진 흐름이 된다.

```python
class ResNetBlock(nn.Module):
    """여느 ResNet 덩이: h_{l+1} = h_l + f(h_l)"""
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, h):
        return h + self.net(h)  # Euler step with dt=1


class ODEFunc(nn.Module):
    """ODE dynamics: dh/dt = f(h, t)"""
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),  # Bounded activation for Lipschitz guarantee
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, h):
        # 알림: torchdiffeq은 (t, y) 꼴을 바란다
        return self.net(h)
```

### 4.3 이어진 관점의 이점

| 면 | 남은 이음 신경망 | 신경 상미분 방정식 |
|--------|--------|------------|
| **깊이** | 붙박임(미리 고른다) | 맞추어 감(풀개가 정한다) |
| **매개변수** | 무게 묶음 $L$개 | 움직임 함수 하나 |
| **기억** | 층 $L$개에 $O(L)$ | 딸림 방법으로 $O(1)$ |
| **셈** | 들임마다 붙박임 | 들임마다 맞추어 감 |
| **되돌릴 수 있음** | 보장되지 않음 | 보장됨(뒤 상미분 방정식을 푼다) |

!!! tip "맞추어 가는 셈"
    이어진 적기는 **맞추어 가는 셈**을 가능하게 한다. 상미분 방정식 풀개는 움직임이 복잡한 곳에서 걸음을 더 밟고 단순한 곳에서 덜 밟는다. 이는 거래자가 예사롭지 않은 시장 상황을 살피는 데 시간을 더 쓰고 늘 있는 값 움직임에는 덜 쓰는 것과 비슷하다.

---

## 5. 위상 그림과 그려 보기

위상 그림은 움직임 계의 질적인 모습을 드러낸다.

### 5.1 벡터 마당

함수 $f(y, t)$은 **벡터 마당**을 뜻매김한다. 상태 공간의 점마다 바뀜의 방향과 크기를 가리키는 화살표가 있다.

```python
def plot_vector_field(f, xlim, ylim, n_points=20, ax=None):
    """
    2차원 홀로 도는 상미분 방정식의 벡터 마당을 그린다.
    
    인수:
        f: ODE function (y, t) -> dy/dt
        xlim, ylim: 축의 한계
        n_points: 격자 해상도
        ax: Matplotlib 축
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 격자 생성
    x = torch.linspace(xlim[0], xlim[1], n_points)
    y = torch.linspace(ylim[0], ylim[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 벡터 마당을 셈한다
    points = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    velocities = f(points, 0.0).reshape(n_points, n_points, 2)
    
    U = velocities[..., 0].numpy()
    V = velocities[..., 1].numpy()
    
    # 그려 보려고 고르게 하기
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm, V_norm = U / magnitude, V / magnitude
    
    ax.quiver(X.numpy(), Y.numpy(), U_norm, V_norm, magnitude, cmap='viridis', alpha=0.7)
    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    
    return ax


def plot_trajectories(f, initial_conditions, t_span, dt=0.01, ax=None):
    """
    여러 처음 조건에서 풀이 자취를 그린다.
    
    인수:
        f: 상미분 방정식 함수
        initial_conditions: (y1_0, y2_0) 짝의 목록
        t_span: 적분 구간
        dt: 때 걸음
        ax: Matplotlib 축
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(initial_conditions)))
    
    for i, y0 in enumerate(initial_conditions):
        y0_tensor = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
        t_vals, y_vals = rk4_integrate(f, y0_tensor, t_span, dt)
        
        trajectory = y_vals.squeeze().numpy()
        ax.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                color=colors[i], linewidth=2, alpha=0.8)
        ax.plot(y0[0], y0[1], 'o', color=colors[i], markersize=10,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'IC: {y0}')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
```

### 5.2 붙박이점과 안정성

**붙박이점**(평형)은 $f(y^*) = 0$인 곳에 생긴다. 거기서 시작하면 계가 가만히 머문다.

**안정성**은 붙박이점에서 매긴 야코비 $J = \frac{\partial f}{\partial y}$이 정한다:

- 고윳값의 실수부가 모두 음수 → **점근 안정**(끌개)
- 어떤 고윳값의 실수부가 양수 → **흔들림**(밀개)
- 복소 고윳값 → **소용돌이** 움직임

```python
def analyze_fixed_point(f, y_star, epsilon=1e-5):
    """
    선형화로 붙박이점의 안정성을 살핀다.
    
    인수:
        f: 상미분 방정식 함수
        y_star: 붙박이점의 자리
        epsilon: 수치 야코비를 위한 흔들림
        
    반환값:
        야코비의 고윳값과 고유 벡터
    """
    y_star = torch.tensor(y_star, dtype=torch.float32)
    dim = len(y_star)
    
    # 유한 차분으로 얻은 수치 야코비
    J = torch.zeros(dim, dim)
    for i in range(dim):
        e_i = torch.zeros(dim)
        e_i[i] = epsilon
        
        f_plus = f(y_star + e_i, 0.0)
        f_minus = f(y_star - e_i, 0.0)
        
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    
    # 고윳값 살피기
    eigenvalues, eigenvectors = torch.linalg.eig(J)
    
    print(f"Fixed point: {y_star.numpy()}")
    print(f"Jacobian eigenvalues: {eigenvalues.numpy()}")
    
    # 안정성 갈래 나눔
    real_parts = eigenvalues.real
    if torch.all(real_parts < 0):
        print("Classification: Asymptotically stable (attractor)")
    elif torch.any(real_parts > 0):
        print("Classification: Unstable")
    else:
        print("Classification: Marginally stable (requires further analysis)")
    
    return eigenvalues, eigenvectors
```

---

## 6. 신경 상미분 방정식 얼개와 `torchdiffeq`

상미분 방정식 풀개를 바닥부터 세워 보았으니 이제 실제로 쓸 신경 상미분 방정식 짜기에는 `torchdiffeq` 꾸러미를 쓴다.

### 6.1 설치와 기본 쓰임

```bash
pip install torchdiffeq
```

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    움직임 dh/dt = f(h, t)를 뜻매김한다.
    
    중요: torchdiffeq은 f(y, t)이 아니라 f(t, y) 꼴을 바란다!
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        self.nfe = 0  # Track function evaluations
    
    def forward(self, t, h):
        self.nfe += 1
        return self.net(h)


# 기본 앞먹임
dim = 10
batch_size = 32

func = ODEFunc(dim)
h0 = torch.randn(batch_size, dim)  # Initial hidden state
t = torch.tensor([0., 1.])  # Integration interval [0, 1]

# 상미분 방정식을 푼다: t에 적힌 때의 풀이를 돌려준다
h_trajectory = odeint(func, h0, t)

print(f"h0 shape: {h0.shape}")          # (32, 10)
print(f"trajectory shape: {h_trajectory.shape}")  # (2, 32, 10)
print(f"h(T) shape: {h_trajectory[-1].shape}")    # (32, 10)
```

### 6.2 쓸 수 있는 풀개

```python
# 드러난 룽게-쿠타 방법
y = odeint(func, y0, t, method='euler')      # 1st order
y = odeint(func, y0, t, method='midpoint')   # 2nd order  
y = odeint(func, y0, t, method='rk4')        # 4th order, fixed step
y = odeint(func, y0, t, method='dopri5')     # 4th/5th order, adaptive (DEFAULT)

# 은근한 방법(뻣뻣한 문제용)
y = odeint(func, y0, t, method='implicit_adams')

# 맞추어 가는 풀개 고르기
y = odeint(func, y0, t, method='dopri5',
           rtol=1e-7,    # Relative tolerance (default 1e-7)
           atol=1e-9)    # Absolute tolerance (default 1e-9)

# 붙박이 걸음 방법에는 step_size가 필요하다
y = odeint(func, y0, t, method='euler',
           options={'step_size': 0.1})
```

**풀개 고르기 지침:**

- `dopri5`: 기본 고르기이며 대부분의 문제에 좋다
- `rk4`: 셈 비용을 붙박이로 두고 싶을 때
- `euler`: 빠르지만 부정확하며 벌레 잡기에 좋다
- `implicit_adams`: 뻣뻣한 움직임에(신경 상미분 방정식에서는 드물다)

### 6.3 신경 상미분 방정식 덩이

```python
class NeuralODEBlock(nn.Module):
    """
    들임 h0을 내놓기 h(T)으로 바꾸는 신경 상미분 방정식 덩이.
    
    이는 남은 이음 덩이의 쌓기를 이어진 움직임으로 바꾼다.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, 
                 integration_time: float = 1.0,
                 solver: str = 'dopri5',
                 rtol: float = 1e-5,
                 atol: float = 1e-7):
        super().__init__()
        
        self.func = ODEFunc(dim, hidden_dim)
        self.integration_time = integration_time
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        
        # 적분 때를 (매개변수가 아니라) 버퍼로 등록한다
        self.register_buffer('t', torch.tensor([0., integration_time]))
    
    def forward(self, h0):
        """
        t=0에서 t=T까지 상미분 방정식을 적분한다.
        
        인수:
            h0: 첫 상태 (batch_size, dim)
            
        반환값:
            h(T): 마지막 상태 (batch_size, dim)
        """
        h_trajectory = odeint(
            self.func, h0, self.t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        return h_trajectory[-1]
    
    @property
    def nfe(self):
        """함수 매김 횟수(좇았다면)."""
        return getattr(self.func, 'nfe', None)
```

### 6.4 때에 매인 움직임

나타냄 힘이 더 큰 모델에서는 움직임이 때에 드러나게 매일 수 있다:

```python
class TimeVariantODEFunc(nn.Module):
    """
    Time-dependent dynamics: dh/dt = f(h, t).
    
    때를 들임에 이어 붙여 적분 구간의 자리마다
    다르게 굴 수 있게 한다.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, h):
        batch_size = h.shape[0]
        t_vec = t.expand(batch_size, 1)
        th = torch.cat([h, t_vec], dim=-1)
        return self.net(th)
```

### 6.5 윗신경망 바탕 때 조건 주기

더 힘센 방식은 윗신경망으로 때에 매인 무게를 만든다:

```python
class HypernetODEFunc(nn.Module):
    """
    윗신경망 때 조건 주기를 쓴 움직임.
    
    작은 그물이 때의 함수로 층의 무게를 만들어 낸다.
    드러난 이어 붙임 없이 때에 따라 매끄럽게 바뀌는 움직임을 가능하게 한다.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, hyper_dim: int = 16):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # 윗신경망이 때에서 무게를 만든다
        self.hypernet = nn.Sequential(
            nn.Linear(1, hyper_dim),
            nn.Tanh(),
            nn.Linear(hyper_dim, hidden_dim * dim + hidden_dim)
        )
        
        # 붙박인 층
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
    
    def forward(self, t, h):
        batch_size = h.shape[0]
        
        # 때에서 첫 층의 무게를 만든다
        t_input = t.view(1, 1) if t.dim() == 0 else t.view(-1, 1)
        hyper_out = self.hypernet(t_input)
        
        # 무게와 치우침을 뽑아낸다
        W = hyper_out[:, :self.hidden_dim * self.dim].view(self.hidden_dim, self.dim)
        b = hyper_out[:, self.hidden_dim * self.dim:].view(self.hidden_dim)
        
        # 만든 무게로 앞먹임
        h = torch.tanh(h @ W.T + b)
        h = torch.tanh(self.fc2(h))
        h = self.fc3(h)
        
        return h
```

---

## 7. 온전한 신경 상미분 방정식 가름개

### 7.1 그림 가름을 위한 얼개

```python
class NeuralODEClassifier(nn.Module):
    """
    그림 가름을 위한 온전한 신경 상미분 방정식 모델.
    
    구조:
        1. 줄이는 엮음(들임 → 특징)
        2. 신경 상미분 방정식 덩이(이어진 바꿈)
        3. 가름 머리(특징 → 로짓)
    """
    
    def __init__(self, in_channels: int = 1, 
                 num_classes: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        
        # 줄이기: (batch, 1, 28, 28) → (batch, hidden_dim)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 28 → 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 14 → 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim)
        )
        
        # 신경 상미분 방정식 덩이
        self.ode_block = NeuralODEBlock(
            dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            integration_time=1.0,
            solver='dopri5'
        )
        
        # 분류 머리
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        h0 = self.downsample(x)
        h_final = self.ode_block(h0)
        logits = self.classifier(h_final)
        return logits
```

### 7.2 익히기 되풀이

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_neural_ode_classifier():
    """신경 상미분 방정식 가름개의 온전한 익히기 흐름."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    
    # 데이터
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = NeuralODEClassifier(
        in_channels=1, num_classes=10, hidden_dim=64
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100. * correct / total
        
        # 평가
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return model
```

---

## 8. 익힐 때 살필 것

### 8.1 허용 오차와 정확도

상미분 방정식 풀개의 허용 오차가 모델의 움직임에 곧바로 영향을 준다:

```python
class NeuralODEWithAdaptiveTolerance(nn.Module):
    """익히기와 따지기에 허용 오차를 달리한 신경 상미분 방정식."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)
        self.register_buffer('t', torch.tensor([0., 1.]))
        
        self.train_rtol = 1e-3
        self.train_atol = 1e-5
        self.eval_rtol = 1e-5
        self.eval_atol = 1e-7
    
    def forward(self, h0):
        if self.training:
            rtol, atol = self.train_rtol, self.train_atol
        else:
            rtol, atol = self.eval_rtol, self.eval_atol
        
        return odeint(self.func, h0, self.t, 
                      rtol=rtol, atol=atol)[-1]
```

**권하는 바:**

- **익히기**: 빠르기를 위해 헐거운 허용 오차(보기로 `rtol=1e-3, atol=1e-5`)를 쓴다
- **따지기**: 정확한 헤아림을 위해 허용 오차를 빡빡하게 한다
- **기울기**: 딸림 방법의 허용 오차가 기울기 품질에 영향을 준다

### 8.2 규칙 세우기 재주

신경 상미분 방정식은 지나치게 복잡한 움직임을 배울 수 있다. 규칙 세우기가 도움이 된다.

**운동 에너지 규칙 세우기**는 움직임의 크기를 벌하여 더 단순한 자취를 이끈다:

$$\mathcal{L}_{\text{kinetic}} = \int_0^T \|f_\theta(h(t), t)\|^2 \, dt$$

```python
class RegularizedNeuralODE(nn.Module):
    """운동 에너지 규칙 세우기를 쓴 신경 상미분 방정식."""
    
    def __init__(self, dim, hidden_dim=64, kinetic_weight=0.01):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)
        self.kinetic_weight = kinetic_weight
        self.register_buffer('t', torch.tensor([0., 1.]))
    
    def forward(self, h0, return_regularization=False):
        def augmented_func(t, state):
            h = state[..., :-1]
            dhdt = self.func(t, h)
            
            # 운동 에너지: ||dh/dt||^2
            kinetic = (dhdt ** 2).sum(dim=-1, keepdim=True)
            
            return torch.cat([dhdt, kinetic], dim=-1)
        
        # 운동 에너지를 0으로 첫자리매김한다
        h0_aug = torch.cat([h0, torch.zeros(h0.shape[0], 1, device=h0.device)], dim=-1)
        
        trajectory = odeint(augmented_func, h0_aug, self.t)
        final_state = trajectory[-1]
        
        h_final = final_state[..., :-1]
        total_kinetic = final_state[..., -1].mean()
        
        if return_regularization:
            return h_final, self.kinetic_weight * total_kinetic
        return h_final
```

**야코비 프로베니우스 노름 규칙 세우기**는 움직임의 복잡함을 벌한다:

$$\mathcal{L}_{\text{jacobian}} = \int_0^T \left\| \frac{\partial f}{\partial h} \right\|_F^2 \, dt$$

이는 더 매끄러운 바꿈을 이끌며 이어진 고르게 맞추는 흐름(27.2절)에 쓰이는 대각합 셈하기와 가깝게 이어진다.

### 8.3 무게 첫자리매김

신경 상미분 방정식은 첫자리매김에 민감하다. 처음 무게가 크면 수치가 흔들리고 기울기가 터지며 함수 매김이 지나치게 많아질 수 있다.

```python
def init_neural_ode_weights(module):
    """
    안정된 익히기를 위해 신경 상미분 방정식의 무게를 첫자리매김한다.
    거의 항등인 바꿈으로 시작하도록 작은 무게를 쓴다.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

---

## 9. 나아간 무늬

### 9.1 여러 잣수 신경 상미분 방정식

때 잣수마다 따로 상미분 방정식 덩이로 다룬다:

```python
class MultiScaleNeuralODE(nn.Module):
    """
    때 잣수가 여럿인 신경 상미분 방정식.
    빠른 움직임과 느린 움직임이 함께 있는 문제에 쓸모 있다.
    """
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        
        self.fast_ode = NeuralODEBlock(dim, hidden_dim, integration_time=0.1)
        self.slow_ode = NeuralODEBlock(dim, hidden_dim, integration_time=1.0)
        self.combine = nn.Linear(dim * 2, dim)
    
    def forward(self, h0):
        h_fast = self.fast_ode(h0)
        h_slow = self.slow_ode(h0)
        
        combined = torch.cat([h_fast, h_slow], dim=-1)
        return self.combine(combined)
```

### 9.2 띄엄띄엄한 사건이 있는 신경 상미분 방정식

이어진 움직임과 띄엄띄엄한 뜀을 합친다:

```python
class HybridNeuralODE(nn.Module):
    """
    띄엄띄엄한 중간 바꿈이 있는 신경 상미분 방정식.
    어떤 바꿈이 본디 띄엄띄엄할 때 쓸모 있다
    (보기: 모으기, 눈여겨보기, 장 열고 닫는 일).
    """
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        
        self.ode1 = NeuralODEBlock(dim, hidden_dim, integration_time=0.5)
        self.ode2 = NeuralODEBlock(dim, hidden_dim, integration_time=0.5)
        
        # 상미분 방정식 덩이 사이의 띄엄띄엄한 바꿈
        self.discrete_transform = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, h0):
        h1 = self.ode1(h0)             # First continuous segment
        h2 = h1 + self.discrete_transform(h1)  # Discrete jump
        h3 = self.ode2(h2)             # Second continuous segment
        return h3
```

---

## 10. 신경 상미분 방정식의 벌레 잡기

```python
def debug_neural_ode(model, sample_input):
    """신경 상미분 방정식의 벌레를 잡는 진단 함수."""
    print("=" * 50)
    print("Neural ODE Diagnostics")
    print("=" * 50)
    
    # 매개변수에 NaN이 있는지 살핀다
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"WARNING: NaN in parameters: {nan_params}")
    else:
        print("✓ No NaN in parameters")
    
    # 앞먹임 살피기
    try:
        with torch.no_grad():
            output = model(sample_input)
        
        if torch.isnan(output).any():
            print("WARNING: NaN in forward pass output")
        else:
            print("✓ Forward pass successful")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
    
    # 함수 매김 횟수를 살핀다
    if hasattr(model, 'ode_block') and hasattr(model.ode_block.func, 'nfe'):
        model.ode_block.func.nfe = 0
        _ = model(sample_input)
        print(f"  Function evaluations: {model.ode_block.func.nfe}")
    
    # 기울기 살피기
    model.zero_grad()
    output = model(sample_input)
    loss = output.sum()
    
    try:
        loss.backward()
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()
        
        max_grad = max(grad_norms.values()) if grad_norms else 0
        min_grad = min(grad_norms.values()) if grad_norms else 0
        
        print(f"✓ Backward pass successful")
        print(f"  Gradient norm range: [{min_grad:.6f}, {max_grad:.6f}]")
        
        if max_grad > 100:
            print("WARNING: Potential gradient explosion")
        if min_grad < 1e-7:
            print("WARNING: Potential vanishing gradients")
            
    except Exception as e:
        print(f"ERROR in backward pass: {e}")
```

**흔한 문제와 풀이:**

- **NaN 기울기**: 배움 빠르기를 줄이고 허용 오차를 빡빡하게 하며 깨움이 터지는지 살핀다
- **아주 느린 익히기**: 움직임에 규칙을 세우고 허용 오차를 헐겁게 하며 붙박이 걸음 풀개를 생각해 본다
- **낮은 정확도**: 숨은 차원을 늘리고 더 오래 익히며 적분 시간을 맞춘다

---

## 11. 온전한 보여 주기

```python
torch.manual_seed(42)
np.random.seed(42)


class LearnableODE(nn.Module):
    """
    배울 수 있는 상미분 방정식 움직임 함수: dh/dt = f_theta(h, t).
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, time_dependent: bool = True):
        super().__init__()
        self.time_dependent = time_dependent
        
        input_dim = dim + 1 if time_dependent else dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # 안정을 위해 작은 무게로 첫자리매김한다
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, h):
        if self.time_dependent:
            if isinstance(t, float):
                t = torch.tensor([t])
            t_expanded = t.expand(h.shape[0], 1)
            inputs = torch.cat([h, t_expanded], dim=-1)
        else:
            inputs = h
        
        return self.net(inputs)


def demonstrate_ode_fundamentals():
    """상미분 방정식 개념의 온전한 보여 주기."""
    print("=" * 70)
    print("ODE FUNDAMENTALS DEMONSTRATION")
    print("=" * 70)
    
    # 1부: 수치 방법 견주기
    print("\n1. Comparing Numerical Methods on Exponential Growth")
    print("-" * 50)
    
    def exp_growth(y, t):
        return y  # dy/dt = y
    
    y0 = torch.tensor([[1.0]])
    t_span = (0.0, 2.0)
    
    step_sizes = [0.5, 0.1, 0.01]
    
    for dt in step_sizes:
        t_euler, y_euler = euler_integrate(exp_growth, y0, t_span, dt)
        t_rk4, y_rk4 = rk4_integrate(exp_growth, y0, t_span, dt)
        
        euler_error = abs(y_euler[-1, 0, 0].item() - np.exp(2.0))
        rk4_error = abs(y_rk4[-1, 0, 0].item() - np.exp(2.0))
        print(f"dt={dt}: Euler error={euler_error:.6f}, RK4 error={rk4_error:.2e}")
    
    # 2부: 위상 그림
    print("\n2. Phase Portrait: Damped Oscillator")
    print("-" * 50)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(damped_oscillator, (-3, 3), (-3, 3), ax=ax)
    
    initial_conditions = [(2, 0), (0, 2), (-2, 0), (1, 1)]
    plot_trajectories(damped_oscillator, initial_conditions, (0, 15), ax=ax)
    
    ax.set_title('Damped Harmonic Oscillator Phase Portrait')
    plt.savefig('damped_oscillator_phase.png', dpi=150)
    plt.show()
    
    analyze_fixed_point(damped_oscillator, [0.0, 0.0])
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    1. 상미분 방정식은 이어진 때의 움직임을 나타낸다: dy/dt = f(y, t)
    
    2. 수치 방법이 풀이를 어림한다.
       - 오일러: 온 어긋남 O(dt), 단순하지만 부정확하다
       - RK4: 온 어긋남 O(dt^4), 정확도와 비용의 균형이 좋다
       - 맞추어 가는 방법: 저절로 어긋남 다스리기
    
    3. ResNet은 dt=1인 오일러 띄엄띄엄 나눔 그 자체다.
       h_{l+1} = h_l + f(h_l)  ←→  y_{n+1} = y_n + dt·f(y_n, t_n)
    
    4. 신경 상미분 방정식은 이어진 끝값이다(L → ∞, dt → 0)
    
    5. 위상 그림이 움직임의 질적인 모습을 드러낸다
    """)


if __name__ == "__main__":
    demonstrate_ode_fundamentals()
```

---

## 12. 핵심 정리

1. **상미분 방정식은 이어진 때의 움직임을 적어 준다.** 식 $\frac{dy}{dt} = f(y, t)$으로 나타내며 움직임 함수 $f$이 상태가 어떻게 바뀌는지 정한다.

2. **닫힌 꼴 풀이가 없을 때 수치 적분이 풀이를 어림한다.** 오일러는 단순하지만 부정확하고, RK4은 균형이 좋으며, 맞추어 가는 방법은 뻣뻣한 계를 다룬다.

3. **남은 이음 신경망은 밑에 깔린 이어진 움직임을 오일러로 띄엄띄엄하게 만든 것이다.** 남은 이음 $h_{l+1} = h_l + f(h_l)$은 바로 $\Delta t = 1$인 오일러 한 걸음이다.

4. **신경 상미분 방정식은 이어진 끝을 잡아** 띄엄띄엄한 층을 상미분 방정식이 뜻매김하는 이어진 바꿈으로 바꾼다. 이는 맞추어 가는 셈, 보장된 되돌릴 수 있음, 기억을 아끼는 익히기를 가능하게 한다.

5. **`torchdiffeq`은 미분할 수 있는 상미분 방정식 풀개를 준다.** 여러 방법과 정할 수 있는 허용 오차를 갖춘다. 얼개 짜기에는 움직임 함수, 적분 시간, 풀개 고르기, 허용 오차 두기가 들어간다.

6. **익힐 때 살필 것은** 허용 오차 맞추기(익힐 때는 헐겁게, 따질 때는 빡빡하게), 규칙 세우기(운동 에너지, 야코비 노름), 조심스러운 무게 첫자리매김, 가둬진 깨움 함수이다.

---

## 13. 익힘

### 익힘 1: 중점 방법 짜기

**중점 방법**(RK2)은 다음과 같다:

$$k_1 = f(y_n, t_n)$$

$$k_2 = f\left(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$

$$y_{n+1} = y_n + \Delta t \cdot k_2$$

이 방법을 짜고 오일러 및 RK4과 정확도를 견주어라.

### 익힘 2: 안정 자리

시험 방정식 $\frac{dy}{dt} = \lambda y$에서 앞으로 오일러, 뒤로 오일러($y_{n+1} = y_n + \Delta t \cdot f(y_{n+1}, t_{n+1})$), RK4의 안정 자리를 이끌어 내라. 복소 $\lambda \Delta t$ 평면에 그 자리를 그려라.

### 익힘 3: 반 데르 폴 떨개

반 데르 폴 떨개는 비선형 계이다:

$$\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0$$

1. 일차 계로 바꾼다
2. 움직임 함수를 짠다
3. $\mu = 0.1, 1.0, 5.0$의 위상 그림을 만든다
4. $\mu$에 따라 움직임이 어떻게 바뀌는지 적는다

### 익힘 4: 소용돌이 가름

서로 엇갈린 소용돌이의 점을 가르도록 신경 상미분 방정식을 익혀라:

```python
def make_spiral_data(n_samples=1000, noise=0.1):
    t = torch.linspace(0, 4*np.pi, n_samples)
    x = t * torch.cos(t) + noise * torch.randn(n_samples)
    y = t * torch.sin(t) + noise * torch.randn(n_samples)
    return torch.stack([x, y], dim=1)
```

### 익힘 5: 허용 오차 살피기

`rtol`과 `atol`이 익히기 정확도, 함수 매김 횟수, 익히기 시간에 어떤 영향을 주는지 차근히 살펴라. 맞바꿈 곡선을 그려라.

### 익힘 6: 깊이 견주기

MNIST에서 신경 상미분 방정식(맞추어 가는 깊이)을 깊이 2, 4, 8, 16, 32의 남은 이음 신경망과 견주어라. 정확도, 익히기 시간, 신경 상미분 방정식의 실제 "깊이"를 살펴라.

---

## 참고 문헌

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I: Nonstiff Problems. Springer.
4. Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos. Westview Press.
5. Finlay, C., Jacobsen, J. H., Nurbekyan, L., & Oberman, A. M. (2020). How to Train Your Neural ODE. *ICML*.
6. torchdiffeq 문서: https://github.com/rtqichen/torchdiffeq

## 연습문제

**연습문제 1.**
딸림 민감도 방법을 이끌어 내라. 딸림 $a(t) = \partial L / \partial z(t)$이 $da/dt = -a^T (\partial f / \partial z)$을 만족함을 보이고 매개변수 기울기를 이끌어 내라.

??? success "연습문제 1 풀이"
    $a(t) = \partial L / \partial z(t)$이라 뜻매김한다. 아주 작은 흔들림에서 $z(t+\epsilon) \approx z(t) + \epsilon f_\theta(z(t), t)$이므로 $\partial z(t+\epsilon)/\partial z(t) = I + \epsilon(\partial f/\partial z)$이다. 딸림은 $da/dt = -a(t)^T (\partial f/\partial z)(z(t), t)$으로 바뀐다. 매개변수 기울기는 $dL/d\theta = -\int_T^0 a(t)^T (\partial f/\partial \theta)\,dt$이며 늘린 상미분 방정식 $[z, a, dL/d\theta]$을 $T$에서 $0$까지 뒤로 풀어 셈한다. $\square$

---

**연습문제 2.**
기억 비용을 견주어라. 딸림 방법이 깊이 $L$인 여느 남은 이음 신경망의 $O(L)$에 견주어 $O(1)$ 기억을 이룸을 보여라.

??? success "연습문제 2 풀이"
    남은 이음 신경망은 뒷걸음 퍼뜨리기를 위해 중간 깨움 $L$개를 모두 담아 기억이 $O(L \cdot d)$이다. 신경 상미분 방정식의 딸림 방법은 마지막 상태 $z(T)$만 있으면 되고 뒤로 풀며 중간 값을 다시 셈한다. 이는 적분 걸음 수와 상관없이 기억 $O(d)$만 든다. 맞바꿈은 셈이 두 배가 되는 것이다(앞으로 한 번, 뒤로 한 번). 깊은 신경망에서는 기억 아낌이 커서 기억이 아니라 셈만이 깊이를 제한하게 된다. $\square$

---

**연습문제 3.**
립시츠 이어진 $f_\theta$을 가진 신경 상미분 방정식이 위상 동형을 뜻매김함을 밝혀라. 이는 나타냄 힘에 어떤 뜻을 지니는가?

??? success "연습문제 3 풀이"
    피카르-린델뢰프에 따라 립시츠 이어짐이 유일한 풀이를 보장한다. 흐름 옮김 $\phi_t : z(0) \mapsto z(t)$은 일대일이고(하나뿐임) 이어져 있으며(처음 조건에 이어져 매임) 이어진 역 $\phi_{-t}$을 가진다. 따라서 $\phi_t$은 위상 동형이다. 곧 신경 상미분 방정식은 자료의 위상을 바꿀 수 없다. 이어진 조각을 가르거나 합칠 수 없다. 이는 남은 이음 신경망에 견주어 나타냄 힘을 제한한다. (차원을 더하는) 늘린 신경 상미분 방정식이 이를 넘어선다. $\square$

---

**연습문제 4.**
신경 상미분 방정식에서 앞 방향 미분이 딸림 방법보다 나은 때는 언제인가?

??? success "연습문제 4 풀이"
    앞 방향은 매개변수 개수 $p$과 상관없이 $O(d_{\text{out}})$ 시간에 방향 미분을 셈하고, 딸림은 내놓기 차원과 상관없이 $O(p)$이다. 앞 방향이 나은 때는 (1) 매개변수가 적을 때(보기로 물리 상수), (2) 온전한 야코비 $\partial z/\partial z_0$이 필요할 때, (3) 이차 방법이 야코비-벡터 곱을 요구할 때이다. 앞 방향은 또한 딸림의 어긋남이 쌓이는 뒤엉킨 움직임에서 뒤로 적분할 때의 수치 문제를 피한다. $\square$
