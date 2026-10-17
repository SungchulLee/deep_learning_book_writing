# 상미분방정식과 뉴럴 ODE의 기초

뉴럴 ODE는 은닉상태를 바꾸는 뉴럴 넷트웍을 “다음 레이어”이 아니라 시간 미분으로 정의한다. 그 미분을 수치 적분한 결과가 모델의 출력이다. 따라서 벡터장의 성질, 적분 구간·솔버와 미분 방식이 함께 모델의 동작을 결정한다. 먼저 고전 초기값 문제를 익힌 뒤 코드의 수치 근사와 학습 구조를 연결해 보자.

## 1. 초기값 문제와 해의 조건

상태 y(t)∈Rᵈ와 벡터장 f를 사용해 다음을 정의한다.

$$\frac{dy}{dt}=f(y,t),\qquad y(t_0)=y_0$$

t를 직접 사용하지 않는 f(y)는 자율계이고, t에 명시적으로 의존하면 비자율계다. 시간 인수를 받는 함수라도 실제 계산에서 t를 무시하면 자율계다. 이 페이지의 수제 솔버는 f(y,t), torchdiffeq는 **f(t,y)**를 사용한다.

시간에 연속이고 상태에 국소 Lipschitz인 f는 초기점 주변에서 해의 존재·유일성을 보장한다. 이 조건만으로 모든 미래 시간의 해를 보장하지는 않는다. 예를 들어 y′=y²는 국소 Lipschitz지만 양의 초기값에서 유한 시간에 발산한다. 유한 구간 전체의 해에는 전역 Lipschitz·성장 제어 등 추가 조건이 필요하다.

유한 웨이트의 affine 레이어와 Lipschitz 활성화를 합성한 MLP는 상태에 대해 Lipschitz다. **ReLU와 Softplus는 둘 다 1-Lipschitz**이며 출력이 유계일 필요는 없다. ReLU의 비미분점은 유일성 조건의 위반과 다르다. Tanh는 유계·매끄럽고, GELU·SiLU는 출력이 유계가 아니지만 도함수가 유계다. 어떤 활성화를 택해도 큰 웨이트의 수치 불안정성과 강성을 자동으로 막아 주지는 않는다.

## 2. 고전 벡터장 예제

지수 성장·감쇠 y′=ky의 해는 y(t)=y₀exp(k(t−t₀))다. 아래 해석 함수는 t₀=0을 가정한다.

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
        k: 자람 빠르기 파라메터
        
    반환값:
        바뀜 빠르기 dy/dt
    """
    return k * y

# 확인을 위한 닫힌 꼴 풀이
def exponential_analytical(y0: torch.Tensor, t: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """해석 풀이: y(t) = y_0 * exp(k*t)"""
    return y0 * torch.exp(k * t)
```

감쇠 진동자는 위치 x와 속도 v를 상태로 두어 x′=v, v′=−ω₀²x−2ζω₀v로 바꾼다. 양의 감쇠와 고유진동수에서 에너지가 소산되는 예다. ζ의 감쇠 구분은 보통 ζ≥0을 가정한다.

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

Lotka–Volterra는 먹이 x와 포식자 y 사이 상호작용을 표현한다. 양의 초기값과 양의 계수에서 연속 해의 양성 보존과 수치 근사의 양성 보존은 별개의 성질이다.

```python
def lotka_volterra(state: torch.Tensor, t: float,
                   alpha: float = 1.5, beta: float = 1.0,
                   gamma: float = 3.0, delta: float = 1.0) -> torch.Tensor:
    """
    로트카-볼테라 잡아먹개-먹이 액션.
    
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

## 3. 전진 오일러와 오차·안정성

한 스텝의 전진 오일러는 yₙ₊₁=yₙ+h f(yₙ,tₙ)다. 정확한 현재 상태에서 출발한 한 스텝 오차는 매끄러운 f에서 O(h²)이고, 고정된 유한 시간 구간의 전역 오차는 안정성 조건 아래 O(h)다.

```python
def euler_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    앞으로 오일러 방법의 한 스텝.
    
    수학으로 나타내기:
        y_{n+1} = y_n + dt * f(y_n, t_n)
    
    이는 일차 방법이다. 그 자리 어긋남 O(dt²), 온 어긋남 O(dt).
    
    인수:
        f: ODE function dy/dt = f(y, t)
        y: 지금 상태
        t: 지금 때
        dt: 때 스텝 크기
        
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
        dt: 때 스텝
        
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

여기서는 arange(t_start,t_end+dt,dt)를 사용하므로 구간 길이가 dt로 나누어떨어지지 않으면 끝점을 넘길 수 있다. 마지막 스텝을 자동으로 줄이지 않는다. dt는 양수·순방향 구간을 가정하고, 시간·결과 저장 텐서는 CPU 기본 dtype으로 만든다. 임의의 장치·dtype·역시간을 처리하는 범용 솔버는 아니다.

시험방정식 y′=λy에서 증폭 인자는 1+hλ다. 실수 λ<0의 감쇠를 유지하려면 |1+hλ|<1, 즉 0<h<2/|λ|가 필요하다. 양의 고유값에서 실제 해가 자라는 현상과 수치 불안정성도 구분한다. 적응형 명시적 솔버도 강성이 크면 매우 작은 스텝을 요구할 수 있다.

## 4. RK4와 적응형 적분

고전 RK4는 시작·중간 두 번·끝에서 네 그래디언트를 계산하고 (1,2,2,1)/6의 웨이트를 사용한다. 충분히 매끄러운 문제에서 한 스텝 오차 O(h⁵), 전역 오차 O(h⁴)를 얻는다.

```python
def rk4_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    4차 룽게-쿠타 방법의 한 스텝.
    
    Local error: O(dt^5)
    Global error: O(dt^4)
    
    인수:
        f: 상미분 방정식 함수
        y: 지금 상태
        t: 지금 때
        dt: 때 스텝
        
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

수제 RK4도 앞의 arange·저장 텐서 조건을 공유한다. 높은 차수가 모든 스텝에서 안정성을 보장하는 것은 아니다. 반올림 오차와 해의 정규성도 실제 수렴률에 영향을 준다.

```python
from torchdiffeq import odeint

# torchdiffeq의 맞추어 가는 솔버를 쓴다
def integrate_adaptive(f, y0, t_eval, method='dopri5', rtol=1e-7, atol=1e-9):
    """
    맞추어 가는 스텝 크기 다스리기로 상미분 방정식을 적분한다.
    
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

integrate_adaptive는 odeint를 그대로 호출하는 래퍼다. 앞의 exponential_ode나 damped_oscillator는 f(y,t) 순서이므로 **그대로 전달하면 인수 순서가 맞지 않는다**. odeint에 전달할 함수는 f(t,y)로 정의하거나 어댑터를 두어야 한다. 적응형 솔버의 rtol·atol은 국소 오차 추정을 제어하며 최종 전역 오차의 엄격한 보증값은 아니다.

## 5. ResNet과의 연결

잔차 블록 hₗ₊₁=hₗ+Fₗ(hₗ)은 형식적으로 단위 스텝 오일러와 닮았다. 하지만 레이어마다 임의의 다른 Fₗ을 가진 ResNet이 자동으로 하나의 고정 벡터장을 적분하는 것은 아니다. 고정 T에서 h=T/L로 줄이며 Fₗ≈h f(hₗ,tₗ)인 일관된 계열과 안정성 조건을 갖추어야 연속 극한을 논할 수 있다.

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

ODEFunc는 t를 받지만 계산에 사용하지 않아 자율계다. 이후에 같은 이름의 ODEFunc가 더 깊은 MLP로 다시 정의되는 점도 구분한다. “L을 늘리면 항상 같은 ODE로 수렴한다”거나 단위 스텝 잔차망의 오차를 깊이와 무관한 O(1)로 단정하는 것은 위 조건을 생략한 설명이다.

## 6. 위상도와 평형점

벡터장은 상태의 순간 이동 방향이며 적분 궤적은 이를 따라간다. 아래 벡터 화살표는 길이를 정규화하고 원래 크기를 색으로 표시하려는 코드다.

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
        dt: 때 스텝
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

plot 함수는 f(y,t)와 CPU 텐서를 가정한다. 뉴럴 넷트웍처럼 gradient를 추적하는 출력에 바로 numpy를 호출하면 변환할 수 없으며 no_grad 또는 detach 처리가 필요하다. 위상도는 수치 궤적과 함께 읽되 그림만으로 전역 안정성을 증명하지 않는다.

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
    
    # 안정성 클래스 나눔
    real_parts = eigenvalues.real
    if torch.all(real_parts < 0):
        print("Classification: Asymptotically stable (attractor)")
    elif torch.any(real_parts > 0):
        print("Classification: Unstable")
    else:
        print("Classification: Marginally stable (requires further analysis)")
    
    return eigenvalues, eigenvectors
```

analyze_fixed_point는 중심 차분으로 Jacobian을 근사하지만 먼저 f(y*)=0인지 검사하지 않는다. 자율계의 실제 평형에서 고유값 실수부가 모두 음수이면 국소 점근 안정, 하나라도 양수이면 불안정이라는 선형화 결론을 사용할 수 있다. 실수부가 0인 경우에는 **판정 불가**이며 코드 출력의 “marginally stable”만으로 안정성을 선언할 수 없다. 복소 고유값도 실수부에 따라 감쇠·성장 회전이 달라진다.

## 7. torchdiffeq 인터페이스

아래는 기존 설치 명령이다. 여기서는 설치·실행을 수행하지 않는다. API의 세부 옵션은 [공식 문서](https://github.com/rtqichen/torchdiffeq)와 실제 사용하는 버전을 확인한다.

```bash
pip install torchdiffeq
```

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    액션 dh/dt = f(h, t)를 뜻매김한다.
    
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

# 기본 피드포워드
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

h₀가 32×10이고 t=[0,1]이면 출력은 **2×32×10**이다. 두 출력 시점만 요청했다는 뜻이며 내부 스텝을 두 번만 수행했다는 뜻은 아니다. nfe는 벡터장 호출마다 누적되며 저절로 초기화되지 않는다.

```python
# 드러난 룽게-쿠타 방법
y = odeint(func, y0, t, method='euler')      # 1st order
y = odeint(func, y0, t, method='midpoint')   # 2nd order  
y = odeint(func, y0, t, method='rk4')        # 4th order, fixed step
y = odeint(func, y0, t, method='dopri5')     # 4th/5th order, adaptive (DEFAULT)

# 은근한 방법(뻣뻣한 문제용)
y = odeint(func, y0, t, method='implicit_adams')

# 맞추어 가는 솔버 고르기
y = odeint(func, y0, t, method='dopri5',
           rtol=1e-7,    # Relative tolerance (default 1e-7)
           atol=1e-9)    # Absolute tolerance (default 1e-9)

# 붙박이 스텝 방법에는 step_size가 필요하다
y = odeint(func, y0, t, method='euler',
           options={'step_size': 0.1})
```

이 선택 예시는 별도 y0·func·t가 준비되어 있어야 한다. 고정 스텝 솔버의 정확도는 step_size로 조절하고, 적응형 솔버는 오차 추정과 rtol·atol에 따라 내부 스텝을 선택한다. implicit_adams라는 이름만으로 강성 문제에 적합한 BDF 솔버와 같은 보장을 갖는다고 가정하지 않는다. 공식 torchdiffeq의 rk4는 3/8 규칙으로, 앞의 수제 고전 RK4와 계수는 다르지만 둘 다 4차다.

## 8. 은닉상태의 ODE 블록

```python
class NeuralODEBlock(nn.Module):
    """
    들임 h0을 내놓기 h(T)으로 바꾸는 뉴럴 상미분 방정식 덩이.
    
    이는 남은 이음 덩이의 쌓기를 이어진 액션으로 바꾼다.
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
        
        # 적분 때를 (파라메터가 아니라) 버퍼로 등록한다
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

NeuralODEBlock은 시간 벡터를 buffer로 등록하고 최종 요청 시점의 상태만 반환한다. integration_time 속성만 나중에 바꾸어도 이미 만들어진 self.t가 자동 갱신되지는 않는다. nfe 속성은 누적 호출 수를 읽으므로 전후 차이나 명시적 초기화로 측정한다.

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

TimeVariantODEFunc는 배치마다 같은 스칼라 t를 확장해 상태와 연결한다. h는 B×dim을 가정한다. 시간 정보를 받는 것과 실제로 사용하는 것이 다른 예다.

```python
class HypernetODEFunc(nn.Module):
    """
    윗뉴럴 넷트웍 때 조건 주기를 쓴 액션.
    
    작은 넷트웍이 때의 함수로 레이어의 무게를 만들어 낸다.
    드러난 이어 붙임 없이 때에 따라 매끄럽게 바뀌는 액션을 가능하게 한다.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, hyper_dim: int = 16):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # 윗뉴럴 넷트웍이 때에서 무게를 만든다
        self.hypernet = nn.Sequential(
            nn.Linear(1, hyper_dim),
            nn.Tanh(),
            nn.Linear(hyper_dim, hidden_dim * dim + hidden_dim)
        )
        
        # 붙박인 레이어
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
    
    def forward(self, t, h):
        batch_size = h.shape[0]
        
        # 때에서 첫 레이어의 무게를 만든다
        t_input = t.view(1, 1) if t.dim() == 0 else t.view(-1, 1)
        hyper_out = self.hypernet(t_input)
        
        # 무게와 치우침을 뽑아낸다
        W = hyper_out[:, :self.hidden_dim * self.dim].view(self.hidden_dim, self.dim)
        b = hyper_out[:, self.hidden_dim * self.dim:].view(self.hidden_dim)
        
        # 만든 무게로 피드포워드
        h = torch.tanh(h @ W.T + b)
        h = torch.tanh(self.fc2(h))
        h = self.fc3(h)
        
        return h
```

HypernetODEFunc는 스칼라 시간에서 첫 레이어의 W·b를 생성하고 나머지 두 레이어는 공유한다. 독립된 샘플별 시간 벡터를 임의로 받을 수 있는 코드는 아니다. hyper_out을 하나의 행렬로 view하는 모양이 스칼라 시간 호출에 맞추어져 있다.

## 9. MNIST 분류 구성

```python
class NeuralODEClassifier(nn.Module):
    """
    그림 가름을 위한 온전한 뉴럴 상미분 방정식 모델.
    
    구조:
        1. 줄이는 컨볼루션(들임 → 피처)
        2. 뉴럴 상미분 방정식 덩이(이어진 바꿈)
        3. 가름 머리(피처 → 로짓)
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
        
        # 뉴럴 상미분 방정식 덩이
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

CNN으로 28→14→7 해상도를 만든 뒤 hidden_dim=64 벡터로 옮기고, 0부터 1까지 ODE 블록을 통과시켜 10개 logits를 출력한다. BatchNorm은 CNN 앞단에 있고 ODE 벡터장은 Tanh MLP다. 이미지 크기와 첫 Linear의 고정 입력 차원을 함께 맞춰야 한다.

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_neural_ode_classifier():
    """뉴럴 상미분 방정식 분류기의 온전한 익히기 흐름."""
    
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

학습 함수는 Adam·교차 엔트로피를 사용하고 에포크마다 테스트 정확도를 계산한다. 60,000개·배치128에서 마지막96개를 포함해 469회, 10에포크에 4,690회 업데이트한다. 함수는 정의되어 있지만 맨 아래 main은 이 분류 학습을 호출하지 않는다. 보존된 출력은 뒤의 고전 ODE 데모이며 MNIST 정확도를 입증하지 않는다.

이 코드는 **odeint로 직접 역전파**한다. odeint_adjoint를 가져오지 않으므로 수치 스텝 수와 무관한 adjoint 메모리 절약을 이미 구현했다고 설명할 수 없다. 수치 솔버 내부 연산의 그래프를 통해 미분하는 방식과 연속 adjoint를 별도로 적분하는 방식을 구분한다.

## 10. 허용 오차와 정규화

```python
class NeuralODEWithAdaptiveTolerance(nn.Module):
    """익히기와 따지기에 허용 오차를 달리한 뉴럴 상미분 방정식."""
    
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

이 블록은 학습·평가 모드에서 허용 오차를 다르게 둔다. 같은 벡터장도 수치적인 입출력과 비용이 달라질 수 있으므로 더 엄격한 허용 오차가 언제나 분류 정확도를 높인다고 보장하지 않는다. 학습·평가 설정을 기록하고 성능의 민감도를 확인해야 한다.

```python
class RegularizedNeuralODE(nn.Module):
    """운동 에너지 규칙 세우기를 쓴 뉴럴 상미분 방정식."""
    
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

RegularizedNeuralODE는 상태에 누적 변수 하나를 추가해 ∫||f||²dt를 계산한다. return_regularization=True일 때 돌려주는 규제 항을 실제 학습 로스에 더해야 효과가 있다. 상태 경로가 부드러워질 수 있지만 Jacobian 규제 ∫||∂f/∂h||²_Fdt와는 다른 양이며 후자는 여기 구현되어 있지 않다.

```python
def init_neural_ode_weights(module):
    """
    안정된 익히기를 위해 뉴럴 상미분 방정식의 무게를 첫자리매김한다.
    거의 항등인 바꿈으로 시작하도록 작은 무게를 쓴다.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

이 함수는 작은 Xavier gain과 0 bias를 설정하는 초기화 도구다. 정의한 것만으로 기존 모델에 적용되는 것은 아니다. 초기화·정규화·허용 오차는 강성이나 비용을 줄이는 데 도움이 될 수 있으나 안정성을 보장하는 일괄 처방은 아니다.

## 11. 여러 경로와 이산 변환

```python
class MultiScaleNeuralODE(nn.Module):
    """
    때 잣수가 여럿인 뉴럴 상미분 방정식.
    빠른 액션과 느린 액션이 함께 있는 문제에 쓸모 있다.
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

MultiScaleNeuralODE는 같은 초기값에서 서로 다른 두 벡터장·적분 구간을 **병렬로** 계산하고 결합한다. 한 시스템의 빠른·느린 물리 변수에 대한 식별이나 강성 분해를 자동으로 구현한 것은 아니다.

```python
class HybridNeuralODE(nn.Module):
    """
    띄엄띄엄한 중간 바꿈이 있는 뉴럴 상미분 방정식.
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

HybridNeuralODE는 ODE 두 구간 사이에 잔차 변환을 넣는다. 각 블록의 자체 시간은 0부터 시작하며 사건 시각을 자동 탐지하는 하이브리드 솔버는 아니다. 중간 이산 변환은 단사성·가역성을 보장하지 않으므로 전체 모델을 순수 ODE 흐름과 동일시하지 않는다.

## 12. 진단 코드와 보존된 데모

```python
def debug_neural_ode(model, sample_input):
    """뉴럴 상미분 방정식의 벌레를 잡는 진단 함수."""
    print("=" * 50)
    print("Neural ODE Diagnostics")
    print("=" * 50)
    
    # 파라메터에 NaN이 있는지 살핀다
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"WARNING: NaN in parameters: {nan_params}")
    else:
        print("✓ No NaN in parameters")
    
    # 피드포워드 살피기
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
    
    # 그래디언트 살피기
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

debug_neural_ode는 여러 번 순방향을 호출하고 gradient를 계산해 모델 gradient를 변경한다. 기존 모드를 바꾸지 않아 훈련 모드의 BatchNorm 상태가 바뀔 수도 있다. NaN 검사만 하고 Inf를 별도로 검사하지 않으며, 임의 gradient 노름 임계값은 원인 진단의 참고값이다. 이 함수 호출도 모델 상태에 영향을 줄 수 있는 작업이다.

```python
torch.manual_seed(42)
np.random.seed(42)

class LearnableODE(nn.Module):
    """
    배울 수 있는 상미분 방정식 액션 함수: dh/dt = f_theta(h, t).
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
    1. 상미분 방정식은 이어진 때의 액션을 나타낸다: dy/dt = f(y, t)
    
    2. 수치 방법이 풀이를 어림한다.
       - 오일러: 온 어긋남 O(dt), 단순하지만 부정확하다
       - RK4: 온 어긋남 O(dt^4), 정확도와 비용의 균형이 좋다
       - 맞추어 가는 방법: 저절로 어긋남 다스리기
    
    3. ResNet은 dt=1인 오일러 띄엄띄엄 나눔 그 자체다.
       h_{l+1} = h_l + f(h_l)  ←→  y_{n+1} = y_n + dt·f(y_n, t_n)
    
    4. 뉴럴 상미분 방정식은 이어진 끝값이다(L → ∞, dt → 0)
    
    5. 위상 그림이 액션의 질적인 모습을 드러낸다
    """)

if __name__ == "__main__":
    demonstrate_ode_fundamentals()
```

**보존된 출력:**

```
======================================================================
ODE FUNDAMENTALS DEMONSTRATION
======================================================================

1. Comparing Numerical Methods on Exponential Growth
--------------------------------------------------
dt=0.5: Euler error=2.326556, RK4 error=5.09e-03
dt=0.1: Euler error=0.661556, RK4 error=9.43e-06
dt=0.01: Euler error=0.073038, RK4 error=1.07e-07

2. Phase Portrait: Damped Oscillator
--------------------------------------------------
Fixed point: [0. 0.]
Jacobian eigenvalues: [-0.20000005+1.9899749j -0.20000005-1.9899749j]
Classification: Asymptotically stable (attractor)

======================================================================
SUMMARY
======================================================================

    1. 상미분 방정식은 이어진 때의 액션을 나타낸다: dy/dt = f(y, t)
    
    2. 수치 방법이 풀이를 어림한다.
       - 오일러: 온 어긋남 O(dt), 단순하지만 부정확하다
       - RK4: 온 어긋남 O(dt^4), 정확도와 비용의 균형이 좋다
       - 맞추어 가는 방법: 저절로 어긋남 다스리기
    
    3. ResNet은 dt=1인 오일러 띄엄띄엄 나눔 그 자체다.
       h_{l+1} = h_l + f(h_l)  ←→  y_{n+1} = y_n + dt·f(y_n, t_n)
    
    4. 뉴럴 상미분 방정식은 이어진 끝값이다(L → ∞, dt → 0)
    
    5. 위상 그림이 액션의 질적인 모습을 드러낸다
```

main이 호출하는 데모는 지수 성장의 수제 오일러·RK4 비교와 감쇠 진동자의 위상도·평형 선형화다. LearnableODE는 정의하지만 이 데모에서 학습하지 않는다. float 시간 인수를 텐서로 바꾸는 경로는 CPU에서 생성하므로 다른 장치에서 사용할 때 입력 장치를 맞춰야 한다. 출력 문자열의 ResNet 극한·정확도 주장은 앞서 설명한 조건 아래에서 읽는다.

## 13. 추가 데이터 생성 코드의 범위

```python
def make_spiral_data(n_samples=1000, noise=0.1):
    t = torch.linspace(0, 4*np.pi, n_samples)
    x = t * torch.cos(t) + noise * torch.randn(n_samples)
    y = t * torch.sin(t) + noise * torch.randn(n_samples)
    return torch.stack([x, y], dim=1)
```

make_spiral_data는 t에 따라 반지름이 늘어나는 **하나의 나선**에 노이즈를 더해 N×2 좌표를 반환한다. 서로 얽힌 두 클래스나 레이블, 분류기·학습 루프를 제공하지 않는다. 따라서 이 함수만으로 두 나선 분류 실험을 완료한 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
최종 상태 로스 L=ℓ(z(T))에서 열벡터 adjoint a=∂L/∂z의 방정식과 파라미터 그래디언트를 적어라.

</div>

??? success "풀이"

    Jz=∂f/∂z, Jθ=∂f/∂θ라 하면 a′=−Jzᵀa, a(T)=∇ℓ(z(T))다. 초기값이 θ와 독립이고 로스에 직접 θ 항이 없으면 ∂L/∂θ=∫₀ᵀJθᵀa dt다. 역시간 적분에서는 부호와 적분 방향을 함께 바꾸어 같은 값을 얻는다. 중간 관측 로스에는 해당 시점의 adjoint 점프, θ에 의존하는 초기값에는 a(0)ᵀ∂z₀/∂θ가 추가된다. 수치적으로는 역적분·보간·허용 오차도 그래디언트에 영향을 준다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
직접 역전파와 adjoint의 메모리·계산 비용을 비교하라. 이 페이지가 실제로 쓰는 방식은 무엇인가?

</div>

??? success "풀이"

    직접 odeint 역전파는 내부 수치 연산 그래프를 저장한다. 연속 adjoint는 역방향에서 상태와 민감도를 다시 적분해 내부 스텝별 저장을 줄일 수 있다. O(1)은 내부 스텝 수에 대한 관례이며 요청 출력 시점·파라미터·상태·checkpoint의 메모리까지 모두 상수라는 뜻은 아니다. 역적분이 불안정하거나 NFE가 달라질 수 있어 비용도 정확히 두 배가 아니다. 이 페이지는 odeint를 가져오므로 직접 역전파다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
ODE 흐름의 가역성과 ReLU의 Lipschitz 성질을 설명하라.

</div>

??? success "풀이"

    충분한 유일성과 양방향으로 필요한 구간의 해가 존재하면 고정 시작·종료 시각의 흐름은 연속 역을 갖는다. 비자율계의 역은 종료 시각에서 시작 시각으로 푸는 두 시각 흐름이며 단순히 같은 시작 시각에서 −t를 넣는 것과 다르다. 수치 역적분은 정확한 역을 보장하지 않고 외부 차원 축소·분류 레이어도 가역일 필요가 없다. ReLU는 1-Lipschitz라 비미분점이 유일성을 깨지는 않는다. 국소 Lipschitz만으로 유한 시간 발산까지 막는 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
순방향 민감도와 adjoint의 비용이 유리한 경우를 방향 수 관점에서 비교하라.

</div>

??? success "풀이"

    순방향은 입력·파라미터의 각 관심 방향에 JVP를 전파하므로 적은 수의 파라미터나 방향 도함수가 필요할 때 유리할 수 있다. 역방향 adjoint는 스칼라 로스의 gradient를 많은 파라미터에 대해 구할 때 유리하다. 전체 Jacobian을 얻는 비용은 순방향의 입력 방향 수와 역방향의 출력 방향 수에 영향을 받는다. 둘 다 벡터장·선형화 연산의 비용이 있으며 단순히 출력 차원 또는 파라미터 수와 무관한 고정 비용이라고 쓸 수 없다.



## 정리하며

뉴럴 ODE는 벡터장과 수치 적분을 함께 설계하는 모델이다. 해의 국소·전역 조건, 활성화의 Lipschitz 성질과 수치 안정성을 구분한다. 함수 인수 순서·내부 스텝·요청 출력 시점·미분 방식을 맞추고 각 코드가 실제로 호출하는 학습·데모의 범위를 확인한다.

**참고 문헌**

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I: Nonstiff Problems. Springer.
4. Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos. Westview Press.
5. Finlay, C., Jacobsen, J. H., Nurbekyan, L., & Oberman, A. M. (2020). How to Train Your Neural ODE. *ICML*.
6. torchdiffeq 문서: https://github.com/rtqichen/torchdiffeq
