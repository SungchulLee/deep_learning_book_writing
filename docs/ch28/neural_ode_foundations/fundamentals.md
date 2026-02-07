# ODE Fundamentals

## Learning Objectives

By the end of this section, you will:

- Understand ordinary differential equations as continuous-time dynamical systems
- Master numerical integration methods including Euler and Runge-Kutta
- Recognize the deep connection between ResNets and ODE discretization
- Implement basic ODE solvers from scratch in PyTorch
- Visualize phase portraits and solution trajectories
- Build complete Neural ODE models using `torchdiffeq`
- Train Neural ODEs for classification and regression tasks

## Prerequisites

- Multivariable calculus (derivatives, integrals)
- Basic linear algebra (vectors, matrices)
- PyTorch fundamentals (tensors, autograd)
- Familiarity with residual networks (helpful but not required)

---

## 1. What Are Ordinary Differential Equations?

An **ordinary differential equation (ODE)** describes how a quantity changes over time. The term "ordinary" distinguishes these from partial differential equations—ODEs involve derivatives with respect to a single variable (typically time).

### 1.1 Mathematical Formulation

The general first-order ODE takes the form:

$$\frac{dy}{dt} = f(y, t)$$

where:

- $y(t) \in \mathbb{R}^d$ is the **state** at time $t$
- $f: \mathbb{R}^d \times \mathbb{R} \rightarrow \mathbb{R}^d$ is the **dynamics function**
- $\frac{dy}{dt}$ represents the instantaneous rate of change

An **initial value problem (IVP)** adds a starting condition:

$$\frac{dy}{dt} = f(y, t), \quad y(t_0) = y_0$$

The goal is to find $y(t)$ for all $t > t_0$ given the initial state $y_0$.

### 1.2 Autonomous vs Non-Autonomous Systems

**Autonomous ODEs** have dynamics independent of time:

$$\frac{dy}{dt} = f(y)$$

The vector field $f$ depends only on the current state, not when you're observing it. Many physical systems exhibit this time-invariance property.

**Non-autonomous ODEs** explicitly depend on time:

$$\frac{dy}{dt} = f(y, t)$$

External forcing, time-varying parameters, or scheduled interventions create non-autonomous behavior.

!!! info "Neural ODE Convention"
    Neural ODEs typically use non-autonomous formulations where $f$ is a neural network. Even if the network architecture doesn't explicitly use $t$, having access to it enables time-dependent transformations and provides additional modeling flexibility. Note that `torchdiffeq` expects the signature `f(t, y)`, not `f(y, t)`.

### 1.3 Existence and Uniqueness

The **Picard-Lindelöf theorem** guarantees a unique solution exists if $f$ is:

1. **Continuous** in both arguments
2. **Lipschitz continuous** in $y$: $\|f(y_1, t) - f(y_2, t)\| \leq L\|y_1 - y_2\|$

The Lipschitz condition prevents solutions from "blowing up" or crossing each other. Neural networks with bounded activations (tanh, sigmoid) naturally satisfy this condition, while unbounded activations (ReLU) can violate it. This has direct implications for activation function selection in Neural ODE architectures:

| Activation | Lipschitz | Pros | Cons |
|------------|-----------|------|------|
| **Tanh** | ✓ Bounded | Stable ODE dynamics, implicit regularization | Saturation, vanishing gradients |
| **Softplus** | ✗ Unbounded | Smooth, non-saturating | Can cause trajectory explosion |
| **ReLU** | ✗ Not smooth | Fast computation | Not Lipschitz, violates ODE theory |
| **GELU/SiLU** | ✓ Bounded | Smooth, expressive | More computation per evaluation |

---

## 2. Classic ODE Examples

Understanding classical ODEs builds intuition for neural network dynamics.

### 2.1 Exponential Growth/Decay

The simplest ODE:

$$\frac{dy}{dt} = ky, \quad y(0) = y_0$$

**Analytical solution:** $y(t) = y_0 e^{kt}$

- $k > 0$: Exponential growth (population dynamics, compound interest)
- $k < 0$: Exponential decay (radioactive decay, cooling)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def exponential_ode(y: torch.Tensor, t: float, k: float = 1.0) -> torch.Tensor:
    """
    Exponential growth/decay: dy/dt = k*y
    
    Args:
        y: Current state (batch_size, dim)
        t: Current time (unused in this autonomous ODE)
        k: Growth rate parameter
        
    Returns:
        Rate of change dy/dt
    """
    return k * y

# Analytical solution for verification
def exponential_analytical(y0: torch.Tensor, t: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Analytical solution: y(t) = y_0 * exp(k*t)"""
    return y0 * torch.exp(k * t)
```

### 2.2 Harmonic Oscillator

A second-order ODE converted to first-order system:

$$\frac{d^2x}{dt^2} + 2\zeta\omega_0\frac{dx}{dt} + \omega_0^2 x = 0$$

Define state vector $y = [x, \dot{x}]^T$ (position and velocity):

$$\frac{dy}{dt} = \begin{bmatrix} y_1 \\ -\omega_0^2 y_0 - 2\zeta\omega_0 y_1 \end{bmatrix}$$

where $\omega_0$ is the natural frequency and $\zeta$ is the damping ratio ($\zeta < 1$: underdamped, $\zeta = 1$: critically damped, $\zeta > 1$: overdamped).

```python
def damped_oscillator(y: torch.Tensor, t: float, 
                      omega_0: float = 2.0, zeta: float = 0.1) -> torch.Tensor:
    """
    Damped harmonic oscillator as first-order system.
    
    State: y = [position, velocity]
    
    Args:
        y: State tensor (batch_size, 2)
        t: Current time
        omega_0: Natural frequency
        zeta: Damping ratio
        
    Returns:
        Derivative [dx/dt, dv/dt]
    """
    position = y[..., 0:1]
    velocity = y[..., 1:2]
    
    dxdt = velocity
    dvdt = -omega_0**2 * position - 2 * zeta * omega_0 * velocity
    
    return torch.cat([dxdt, dvdt], dim=-1)
```

### 2.3 Lotka-Volterra Predator-Prey

A classic nonlinear system exhibiting periodic behavior:

$$\frac{dx}{dt} = \alpha x - \beta xy$$
$$\frac{dy}{dt} = \delta xy - \gamma y$$

where $x$ is prey population and $y$ is predator population.

```python
def lotka_volterra(state: torch.Tensor, t: float,
                   alpha: float = 1.5, beta: float = 1.0,
                   gamma: float = 3.0, delta: float = 1.0) -> torch.Tensor:
    """
    Lotka-Volterra predator-prey dynamics.
    
    Args:
        state: [prey, predator] populations
        t: Time (unused)
        alpha: Prey growth rate
        beta: Predation rate  
        gamma: Predator death rate
        delta: Predation efficiency
        
    Returns:
        Population derivatives
    """
    x, y = state[..., 0:1], state[..., 1:2]
    
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    
    return torch.cat([dxdt, dydt], dim=-1)
```

---

## 3. Numerical Integration Methods

When analytical solutions don't exist (which is most cases), we resort to numerical integration.

### 3.1 Forward Euler Method

The simplest numerical integrator approximates the derivative with a forward difference:

$$y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$$

**Derivation:** Taylor expansion gives $y(t + \Delta t) = y(t) + \Delta t \cdot \frac{dy}{dt} + O(\Delta t^2)$. Truncating at first order and substituting the ODE yields the Euler update.

```python
def euler_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    Single step of Forward Euler method.
    
    Mathematical formulation:
        y_{n+1} = y_n + dt * f(y_n, t_n)
    
    This is a first-order method: local error O(dt²), global error O(dt).
    
    Args:
        f: ODE function dy/dt = f(y, t)
        y: Current state
        t: Current time
        dt: Time step size
        
    Returns:
        Next state y_{n+1}
    """
    return y + dt * f(y, t)


def euler_integrate(f, y0: torch.Tensor, t_span: tuple, dt: float):
    """
    Integrate ODE using Forward Euler method.
    
    Args:
        f: ODE function
        y0: Initial state (batch_size, dim)
        t_span: (t_start, t_end)
        dt: Time step
        
    Returns:
        t_values: Time points
        y_values: States at each time point
    """
    t_start, t_end = t_span
    t_values = torch.arange(t_start, t_end + dt, dt)
    n_steps = len(t_values)
    
    # Initialize trajectory storage
    y_values = torch.zeros(n_steps, *y0.shape)
    y_values[0] = y0
    
    # Integration loop
    y = y0
    for i in range(n_steps - 1):
        y = euler_step(f, y, t_values[i].item(), dt)
        y_values[i + 1] = y
    
    return t_values, y_values
```

### 3.2 Error Analysis

**Local truncation error:** Error from a single step, assuming perfect initial condition:

$$\text{LTE} = y(t + \Delta t) - \left[y(t) + \Delta t \cdot f(y(t), t)\right] = O(\Delta t^2)$$

**Global error:** Accumulated error over the entire integration:

$$\text{Global Error} = O(\Delta t)$$

The global error is one order lower because errors accumulate over $O(1/\Delta t)$ steps.

!!! warning "Implications for Neural Networks"
    The error order has profound implications. A ResNet with $L$ layers using step size $\Delta t = 1$ has global error $O(1)$—the discretization error is *fixed* regardless of depth. Neural ODEs with adaptive solvers can achieve arbitrarily small error by taking more steps.

### 3.3 Stability Analysis

Not all step sizes work. Consider the test equation $\frac{dy}{dt} = \lambda y$ with $\lambda < 0$ (decay).

Euler gives: $y_{n+1} = (1 + \lambda \Delta t) y_n$

For stability, we need $|1 + \lambda \Delta t| < 1$, which requires:

$$\Delta t < \frac{2}{|\lambda|}$$

If $\lambda$ is large and negative (stiff systems), tiny time steps are needed—this is why adaptive solvers are essential.

### 3.4 Fourth-Order Runge-Kutta (RK4)

RK4 achieves fourth-order accuracy by evaluating $f$ at multiple points:

$$k_1 = f(y_n, t_n)$$
$$k_2 = f\left(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$
$$k_3 = f\left(y_n + \frac{\Delta t}{2}k_2, t_n + \frac{\Delta t}{2}\right)$$
$$k_4 = f(y_n + \Delta t \cdot k_3, t_n + \Delta t)$$
$$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

```python
def rk4_step(f, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    Single step of 4th-order Runge-Kutta method.
    
    Local error: O(dt^5)
    Global error: O(dt^4)
    
    Args:
        f: ODE function
        y: Current state
        t: Current time
        dt: Time step
        
    Returns:
        Next state
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_integrate(f, y0: torch.Tensor, t_span: tuple, dt: float):
    """Integrate ODE using RK4 method."""
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

### 3.5 Adaptive Step Size Methods

Production Neural ODE implementations use **adaptive solvers** like `dopri5` (Dormand-Prince) that:

1. Estimate local error by comparing solutions of different orders
2. Reject steps if error exceeds tolerance
3. Automatically adjust step size

```python
from torchdiffeq import odeint

# Using adaptive solver from torchdiffeq
def integrate_adaptive(f, y0, t_eval, method='dopri5', rtol=1e-7, atol=1e-9):
    """
    Integrate ODE with adaptive step size control.
    
    Args:
        f: ODE function (must accept (t, y) in that order for torchdiffeq)
        y0: Initial state
        t_eval: Times at which to return solution
        method: Solver method ('dopri5', 'rk4', 'euler', etc.)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Solution at requested times
    """
    return odeint(f, y0, t_eval, method=method, rtol=rtol, atol=atol)
```

---

## 4. The ResNet-ODE Connection

This is the key insight that motivates Neural ODEs.

### 4.1 ResNet as Euler Discretization

A **ResNet layer** computes:

$$h_{l+1} = h_l + f_\theta(h_l)$$

This is *exactly* Euler's method with $\Delta t = 1$:

| Component | Euler Method | ResNet |
|-----------|-------------|--------|
| State | $y_n$ | $h_l$ |
| Dynamics | $f(y_n, t_n)$ | $f_\theta(h_l)$ |
| Update | $y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$ | $h_{l+1} = h_l + f_\theta(h_l)$ |
| Step size | $\Delta t$ | $1$ (implicit) |

The residual connection implements numerical integration of a learned dynamics function.

### 4.2 Taking the Continuous Limit

As the number of layers $L \to \infty$ and step size $\Delta t \to 0$:

$$\lim_{L \to \infty, \Delta t \to 0} h_L = h(T) \quad \text{where} \quad \frac{dh}{dt} = f_\theta(h(t), t)$$

The discrete layer index becomes continuous time, and the layer-by-layer transformation becomes a continuous flow.

```python
class ResNetBlock(nn.Module):
    """Standard ResNet block: h_{l+1} = h_l + f(h_l)"""
    
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
        # Note: torchdiffeq expects (t, y) signature
        return self.net(h)
```

### 4.3 Advantages of the Continuous Perspective

| Aspect | ResNet | Neural ODE |
|--------|--------|------------|
| **Depth** | Fixed (chosen a priori) | Adaptive (solver decides) |
| **Parameters** | $L$ sets of weights | Single dynamics function |
| **Memory** | $O(L)$ for $L$ layers | $O(1)$ via adjoint method |
| **Computation** | Fixed per input | Adaptive per input |
| **Invertibility** | Not guaranteed | Guaranteed (solve reverse ODE) |

!!! tip "Adaptive Computation"
    The continuous formulation enables **adaptive computation**—the ODE solver takes more steps where the dynamics are complex and fewer where they're simple. This is analogous to how a trader spends more time analyzing unusual market conditions and less time on routine price movements.

---

## 5. Phase Portraits and Visualization

Phase portraits reveal the qualitative behavior of dynamical systems.

### 5.1 Vector Fields

The function $f(y, t)$ defines a **vector field**—at each point in state space, there's an arrow indicating the direction and magnitude of change.

```python
def plot_vector_field(f, xlim, ylim, n_points=20, ax=None):
    """
    Plot vector field for 2D autonomous ODE.
    
    Args:
        f: ODE function (y, t) -> dy/dt
        xlim, ylim: Axis limits
        n_points: Grid resolution
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = torch.linspace(xlim[0], xlim[1], n_points)
    y = torch.linspace(ylim[0], ylim[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Compute vector field
    points = torch.stack([X, Y], dim=-1).reshape(-1, 2)
    velocities = f(points, 0.0).reshape(n_points, n_points, 2)
    
    U = velocities[..., 0].numpy()
    V = velocities[..., 1].numpy()
    
    # Normalize for visualization
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm, V_norm = U / magnitude, V / magnitude
    
    ax.quiver(X.numpy(), Y.numpy(), U_norm, V_norm, magnitude, cmap='viridis', alpha=0.7)
    ax.set_xlabel('$y_1$')
    ax.set_ylabel('$y_2$')
    
    return ax


def plot_trajectories(f, initial_conditions, t_span, dt=0.01, ax=None):
    """
    Plot solution trajectories from multiple initial conditions.
    
    Args:
        f: ODE function
        initial_conditions: List of (y1_0, y2_0) tuples
        t_span: Integration interval
        dt: Time step
        ax: Matplotlib axis
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

### 5.2 Fixed Points and Stability

**Fixed points** (equilibria) occur where $f(y^*) = 0$. The system stays at rest if it starts there.

**Stability** is determined by the Jacobian $J = \frac{\partial f}{\partial y}$ evaluated at the fixed point:

- All eigenvalues have negative real parts → **asymptotically stable** (attractor)
- Any eigenvalue has positive real part → **unstable** (repeller)
- Complex eigenvalues → **spiral** behavior

```python
def analyze_fixed_point(f, y_star, epsilon=1e-5):
    """
    Analyze stability of a fixed point via linearization.
    
    Args:
        f: ODE function
        y_star: Fixed point location
        epsilon: Perturbation for numerical Jacobian
        
    Returns:
        eigenvalues, eigenvectors of Jacobian
    """
    y_star = torch.tensor(y_star, dtype=torch.float32)
    dim = len(y_star)
    
    # Numerical Jacobian via finite differences
    J = torch.zeros(dim, dim)
    for i in range(dim):
        e_i = torch.zeros(dim)
        e_i[i] = epsilon
        
        f_plus = f(y_star + e_i, 0.0)
        f_minus = f(y_star - e_i, 0.0)
        
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    
    # Eigenvalue analysis
    eigenvalues, eigenvectors = torch.linalg.eig(J)
    
    print(f"Fixed point: {y_star.numpy()}")
    print(f"Jacobian eigenvalues: {eigenvalues.numpy()}")
    
    # Stability classification
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

## 6. Neural ODE Architecture and `torchdiffeq`

Having built ODE solvers from scratch, we now use the `torchdiffeq` library for production Neural ODE implementations.

### 6.1 Installation and Basic Usage

```bash
pip install torchdiffeq
```

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    Defines the dynamics dh/dt = f(h, t).
    
    IMPORTANT: torchdiffeq expects the signature f(t, y), not f(y, t)!
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


# Basic forward pass
dim = 10
batch_size = 32

func = ODEFunc(dim)
h0 = torch.randn(batch_size, dim)  # Initial hidden state
t = torch.tensor([0., 1.])  # Integration interval [0, 1]

# Solve ODE: returns solution at times specified in t
h_trajectory = odeint(func, h0, t)

print(f"h0 shape: {h0.shape}")          # (32, 10)
print(f"trajectory shape: {h_trajectory.shape}")  # (2, 32, 10)
print(f"h(T) shape: {h_trajectory[-1].shape}")    # (32, 10)
```

### 6.2 Available Solvers

```python
# Explicit Runge-Kutta methods
y = odeint(func, y0, t, method='euler')      # 1st order
y = odeint(func, y0, t, method='midpoint')   # 2nd order  
y = odeint(func, y0, t, method='rk4')        # 4th order, fixed step
y = odeint(func, y0, t, method='dopri5')     # 4th/5th order, adaptive (DEFAULT)

# Implicit methods (for stiff problems)
y = odeint(func, y0, t, method='implicit_adams')

# Adaptive solver options
y = odeint(func, y0, t, method='dopri5',
           rtol=1e-7,    # Relative tolerance (default 1e-7)
           atol=1e-9)    # Absolute tolerance (default 1e-9)

# Fixed step methods need step_size
y = odeint(func, y0, t, method='euler',
           options={'step_size': 0.1})
```

**Solver Selection Guidelines:**

- `dopri5`: Default choice, good for most problems
- `rk4`: When you want fixed computation cost
- `euler`: Fast but inaccurate, good for debugging
- `implicit_adams`: For stiff dynamics (rare in Neural ODEs)

### 6.3 Neural ODE Block

```python
class NeuralODEBlock(nn.Module):
    """
    A Neural ODE block that transforms input h0 to output h(T).
    
    This replaces a stack of residual blocks with continuous dynamics.
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
        
        # Register integration times as buffer (not parameter)
        self.register_buffer('t', torch.tensor([0., integration_time]))
    
    def forward(self, h0):
        """
        Integrate ODE from t=0 to t=T.
        
        Args:
            h0: Initial state (batch_size, dim)
            
        Returns:
            h(T): Final state (batch_size, dim)
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
        """Number of function evaluations (if tracked)."""
        return getattr(self.func, 'nfe', None)
```

### 6.4 Time-Dependent Dynamics

For more expressive models, the dynamics can explicitly depend on time:

```python
class TimeVariantODEFunc(nn.Module):
    """
    Time-dependent dynamics: dh/dt = f(h, t).
    
    Concatenates time to input, allowing different behavior
    at different points in the integration interval.
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

### 6.5 Hypernetwork-Based Time Conditioning

A more powerful approach uses a hypernetwork to generate time-dependent weights:

```python
class HypernetODEFunc(nn.Module):
    """
    Dynamics with hypernetwork time conditioning.
    
    A small network generates layer weights as a function of time,
    enabling smooth time-varying dynamics without explicit concatenation.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, hyper_dim: int = 16):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Hypernet generates weights from time
        self.hypernet = nn.Sequential(
            nn.Linear(1, hyper_dim),
            nn.Tanh(),
            nn.Linear(hyper_dim, hidden_dim * dim + hidden_dim)
        )
        
        # Fixed layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
    
    def forward(self, t, h):
        batch_size = h.shape[0]
        
        # Generate first layer weights from time
        t_input = t.view(1, 1) if t.dim() == 0 else t.view(-1, 1)
        hyper_out = self.hypernet(t_input)
        
        # Extract weights and bias
        W = hyper_out[:, :self.hidden_dim * self.dim].view(self.hidden_dim, self.dim)
        b = hyper_out[:, self.hidden_dim * self.dim:].view(self.hidden_dim)
        
        # Forward pass with generated weights
        h = torch.tanh(h @ W.T + b)
        h = torch.tanh(self.fc2(h))
        h = self.fc3(h)
        
        return h
```

---

## 7. Complete Neural ODE Classifier

### 7.1 Architecture for Image Classification

```python
class NeuralODEClassifier(nn.Module):
    """
    Complete Neural ODE model for image classification.
    
    Architecture:
        1. Downsampling convolutions (input → features)
        2. Neural ODE block (continuous transformation)
        3. Classification head (features → logits)
    """
    
    def __init__(self, in_channels: int = 1, 
                 num_classes: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        
        # Downsampling: (batch, 1, 28, 28) → (batch, hidden_dim)
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
        
        # Neural ODE block
        self.ode_block = NeuralODEBlock(
            dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            integration_time=1.0,
            solver='dopri5'
        )
        
        # Classification head
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

### 7.2 Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_neural_ode_classifier():
    """Complete training pipeline for Neural ODE classifier."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    
    # Data
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
        
        # Evaluation
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

## 8. Training Considerations

### 8.1 Tolerances and Accuracy

The ODE solver tolerances directly affect model behavior:

```python
class NeuralODEWithAdaptiveTolerance(nn.Module):
    """Neural ODE with different tolerances for train/eval."""
    
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

**Recommendations:**

- **Training**: Use loose tolerances (e.g., `rtol=1e-3, atol=1e-5`) for speed
- **Evaluation**: Tighten tolerances for accurate predictions
- **Gradients**: Adjoint method tolerances affect gradient quality

### 8.2 Regularization Techniques

Neural ODEs can learn overly complex dynamics. Regularization helps.

**Kinetic Energy Regularization** penalizes the magnitude of the dynamics to encourage simpler trajectories:

$$\mathcal{L}_{\text{kinetic}} = \int_0^T \|f_\theta(h(t), t)\|^2 \, dt$$

```python
class RegularizedNeuralODE(nn.Module):
    """Neural ODE with kinetic energy regularization."""
    
    def __init__(self, dim, hidden_dim=64, kinetic_weight=0.01):
        super().__init__()
        self.func = ODEFunc(dim, hidden_dim)
        self.kinetic_weight = kinetic_weight
        self.register_buffer('t', torch.tensor([0., 1.]))
    
    def forward(self, h0, return_regularization=False):
        def augmented_func(t, state):
            h = state[..., :-1]
            dhdt = self.func(t, h)
            
            # Kinetic energy: ||dh/dt||^2
            kinetic = (dhdt ** 2).sum(dim=-1, keepdim=True)
            
            return torch.cat([dhdt, kinetic], dim=-1)
        
        # Initialize with zero kinetic energy
        h0_aug = torch.cat([h0, torch.zeros(h0.shape[0], 1, device=h0.device)], dim=-1)
        
        trajectory = odeint(augmented_func, h0_aug, self.t)
        final_state = trajectory[-1]
        
        h_final = final_state[..., :-1]
        total_kinetic = final_state[..., -1].mean()
        
        if return_regularization:
            return h_final, self.kinetic_weight * total_kinetic
        return h_final
```

**Jacobian Frobenius Norm Regularization** penalizes the complexity of the dynamics:

$$\mathcal{L}_{\text{jacobian}} = \int_0^T \left\| \frac{\partial f}{\partial h} \right\|_F^2 \, dt$$

This encourages smoother transformations and is closely related to the trace computations used in continuous normalizing flows (Section 27.2).

### 8.3 Weight Initialization

Neural ODEs are sensitive to initialization. Large initial weights can cause numerical instability, gradient explosion, and excessive function evaluations.

```python
def init_neural_ode_weights(module):
    """
    Initialize Neural ODE weights for stable training.
    Use small weights to start with near-identity transformation.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

---

## 9. Advanced Patterns

### 9.1 Multi-Scale Neural ODE

Process different time scales with separate ODE blocks:

```python
class MultiScaleNeuralODE(nn.Module):
    """
    Neural ODE with multiple time scales.
    Useful for problems with both fast and slow dynamics.
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

### 9.2 Neural ODE with Discrete Events

Combine continuous dynamics with discrete jumps:

```python
class HybridNeuralODE(nn.Module):
    """
    Neural ODE with discrete intermediate transformations.
    Useful when some transformations are naturally discrete
    (e.g., pooling, attention, or market open/close events).
    """
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        
        self.ode1 = NeuralODEBlock(dim, hidden_dim, integration_time=0.5)
        self.ode2 = NeuralODEBlock(dim, hidden_dim, integration_time=0.5)
        
        # Discrete transformation between ODE blocks
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

## 10. Debugging Neural ODEs

```python
def debug_neural_ode(model, sample_input):
    """Diagnostic function for Neural ODE debugging."""
    print("=" * 50)
    print("Neural ODE Diagnostics")
    print("=" * 50)
    
    # Check for NaN in parameters
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"WARNING: NaN in parameters: {nan_params}")
    else:
        print("✓ No NaN in parameters")
    
    # Forward pass check
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
    
    # Check function evaluations
    if hasattr(model, 'ode_block') and hasattr(model.ode_block.func, 'nfe'):
        model.ode_block.func.nfe = 0
        _ = model(sample_input)
        print(f"  Function evaluations: {model.ode_block.func.nfe}")
    
    # Gradient check
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

**Common Issues and Solutions:**

- **NaN gradients**: Reduce learning rate, use tighter tolerances, check for exploding activations
- **Very slow training**: Regularize dynamics, use looser tolerances, consider fixed-step solver
- **Poor accuracy**: Increase hidden dimension, train longer, adjust integration time

---

## 11. Complete Demonstration

```python
torch.manual_seed(42)
np.random.seed(42)


class LearnableODE(nn.Module):
    """
    A learnable ODE dynamics function: dh/dt = f_theta(h, t).
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
        
        # Initialize with small weights for stability
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
    """Complete demonstration of ODE concepts."""
    print("=" * 70)
    print("ODE FUNDAMENTALS DEMONSTRATION")
    print("=" * 70)
    
    # Part 1: Compare numerical methods
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
    
    # Part 2: Phase portrait
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
    1. ODEs describe continuous-time dynamics: dy/dt = f(y, t)
    
    2. Numerical methods approximate solutions:
       - Euler: O(dt) global error, simple but inaccurate
       - RK4: O(dt^4) global error, good balance of accuracy/cost
       - Adaptive methods: automatic error control
    
    3. ResNet IS Euler discretization with dt=1:
       h_{l+1} = h_l + f(h_l)  ←→  y_{n+1} = y_n + dt·f(y_n, t_n)
    
    4. Neural ODE is the continuous limit (L → ∞, dt → 0)
    
    5. Phase portraits reveal qualitative dynamics behavior
    """)


if __name__ == "__main__":
    demonstrate_ode_fundamentals()
```

---

## 12. Key Takeaways

1. **ODEs describe continuous-time dynamics** through the equation $\frac{dy}{dt} = f(y, t)$, with the dynamics function $f$ determining how the state evolves.

2. **Numerical integration approximates solutions** when analytical ones don't exist. Euler is simple but inaccurate; RK4 offers a good balance; adaptive methods handle stiff systems.

3. **ResNets are Euler discretizations** of an underlying continuous dynamics. The residual connection $h_{l+1} = h_l + f(h_l)$ is precisely one Euler step with $\Delta t = 1$.

4. **Neural ODEs take the continuous limit**, replacing discrete layers with a continuous transformation defined by an ODE. This enables adaptive computation, guaranteed invertibility, and memory-efficient training.

5. **`torchdiffeq` provides differentiable ODE solvers** with various methods and configurable tolerances. Architecture design involves the dynamics function, integration time, solver choice, and tolerance settings.

6. **Training considerations include** tolerance tuning (looser for training, tighter for evaluation), regularization (kinetic energy, Jacobian norm), careful weight initialization, and bounded activation functions.

---

## 13. Exercises

### Exercise 1: Implement Midpoint Method

The **midpoint method** (RK2) is:

$$k_1 = f(y_n, t_n)$$
$$k_2 = f\left(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$
$$y_{n+1} = y_n + \Delta t \cdot k_2$$

Implement this method and compare its accuracy to Euler and RK4.

### Exercise 2: Stability Region

For the test equation $\frac{dy}{dt} = \lambda y$, derive the stability region for Forward Euler, Backward Euler ($y_{n+1} = y_n + \Delta t \cdot f(y_{n+1}, t_{n+1})$), and RK4. Plot these regions in the complex $\lambda \Delta t$ plane.

### Exercise 3: Van der Pol Oscillator

The Van der Pol oscillator is a nonlinear system:

$$\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0$$

1. Convert to a first-order system
2. Implement the dynamics function
3. Create phase portraits for $\mu = 0.1, 1.0, 5.0$
4. Describe how the behavior changes with $\mu$

### Exercise 4: Spiral Classification

Train a Neural ODE to classify points from interleaved spirals:

```python
def make_spiral_data(n_samples=1000, noise=0.1):
    t = torch.linspace(0, 4*np.pi, n_samples)
    x = t * torch.cos(t) + noise * torch.randn(n_samples)
    y = t * torch.sin(t) + noise * torch.randn(n_samples)
    return torch.stack([x, y], dim=1)
```

### Exercise 5: Tolerance Study

Systematically study how `rtol` and `atol` affect training accuracy, number of function evaluations, and training time. Plot the trade-off curves.

### Exercise 6: Depth Comparison

Compare Neural ODE (adaptive depth) with ResNets of depth 2, 4, 8, 16, 32 on MNIST. Analyze accuracy, training time, and effective "depth" of the Neural ODE.

---

## References

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I: Nonstiff Problems. Springer.
4. Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos. Westview Press.
5. Finlay, C., Jacobsen, J. H., Nurbekyan, L., & Oberman, A. M. (2020). How to Train Your Neural ODE. *ICML*.
6. torchdiffeq documentation: https://github.com/rtqichen/torchdiffeq
