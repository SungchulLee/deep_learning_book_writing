# ODE Fundamentals

## Learning Objectives

By the end of this section, you will:

- Understand ordinary differential equations as continuous-time dynamical systems
- Master numerical integration methods including Euler and Runge-Kutta
- Recognize the deep connection between ResNets and ODE discretization
- Implement basic ODE solvers from scratch in PyTorch
- Visualize phase portraits and solution trajectories

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

> **Deep Insight:** Neural ODEs typically use non-autonomous formulations where $f$ is a neural network. Even if the network architecture doesn't explicitly use $t$, having access to it enables time-dependent transformations and provides additional modeling flexibility.

### 1.3 Existence and Uniqueness

The **Picard-Lindelöf theorem** guarantees a unique solution exists if $f$ is:

1. **Continuous** in both arguments
2. **Lipschitz continuous** in $y$: $\|f(y_1, t) - f(y_2, t)\| \leq L\|y_1 - y_2\|$

The Lipschitz condition prevents solutions from "blowing up" or crossing each other. Neural networks with bounded activations (tanh, sigmoid) naturally satisfy this condition, while unbounded activations (ReLU) can violate it.

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

where:

- $\omega_0$ is the natural frequency
- $\zeta$ is the damping ratio ($\zeta < 1$: underdamped, $\zeta = 1$: critically damped, $\zeta > 1$: overdamped)

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

where $x$ is prey population, $y$ is predator population.

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

> **Deep Insight:** The error order has profound implications for neural networks. A ResNet with $L$ layers using step size $\Delta t = 1$ has global error $O(1)$—the discretization error is *fixed* regardless of depth. Neural ODEs with adaptive solvers can achieve arbitrarily small error by taking more steps.

### 3.3 Stability Analysis

Not all step sizes work! Consider the test equation $\frac{dy}{dt} = \lambda y$ with $\lambda < 0$ (decay).

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

This is *exactly* Euler's method with $\Delta t = 1$!

Compare:
- Euler: $y_{n+1} = y_n + \Delta t \cdot f(y_n, t_n)$
- ResNet: $h_{l+1} = h_l + f_\theta(h_l)$

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
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, h):
        # Note: torchdiffeq expects (t, y) signature
        return self.net(h)


def compare_resnet_neural_ode():
    """
    Demonstrate equivalence between deep ResNet and Neural ODE.
    """
    dim = 2
    hidden_dim = 32
    
    # Create equivalent architectures
    # ResNet with L layers ≈ Neural ODE integrated from 0 to L with dt=1
    
    # Deep ResNet (many discrete layers)
    n_layers = 100
    resnet_blocks = nn.ModuleList([ResNetBlock(dim, hidden_dim) for _ in range(n_layers)])
    
    # Neural ODE (continuous)
    ode_func = ODEFunc(dim, hidden_dim)
    
    # Initial condition
    h0 = torch.randn(1, dim)
    
    # ResNet forward pass
    h_resnet = h0.clone()
    for block in resnet_blocks:
        h_resnet = block(h_resnet)
    
    # Neural ODE forward pass (integrate from 0 to n_layers)
    t = torch.tensor([0., float(n_layers)])
    h_ode = odeint(ode_func, h0, t, method='euler', options={'step_size': 1.0})
    
    print(f"ResNet output: {h_resnet}")
    print(f"Neural ODE output: {h_ode[-1]}")
    # Note: Outputs differ because parameters aren't shared,
    # but the computational structure is equivalent
```

### 4.3 Advantages of the Continuous Perspective

| Aspect | ResNet | Neural ODE |
|--------|--------|------------|
| **Depth** | Fixed (chosen a priori) | Adaptive (solver decides) |
| **Memory** | O(L) for L layers | O(1) via adjoint method |
| **Computation** | Fixed per input | Adaptive per input |
| **Expressivity** | Discrete transformations | Continuous flows |
| **Invertibility** | Not guaranteed | Guaranteed (solve reverse ODE) |

> **Deep Insight:** The continuous formulation enables **adaptive computation**—the ODE solver takes more steps where the dynamics are complex and fewer where they're simple. This is analogous to how humans spend more time on difficult problems.

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

## 6. PyTorch Implementation: Complete Example

Let's put everything together with a complete, runnable example.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)


class LearnableODE(nn.Module):
    """
    A learnable ODE dynamics function.
    
    This represents dh/dt = f_theta(h, t) where f_theta is a neural network.
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
        """
        Compute dh/dt at state h and time t.
        
        Args:
            t: Current time (scalar or tensor)
            h: Current state (batch_size, dim)
            
        Returns:
            dh/dt with same shape as h
        """
        if self.time_dependent:
            # Expand t to match batch dimension
            if isinstance(t, float):
                t = torch.tensor([t])
            t_expanded = t.expand(h.shape[0], 1)
            inputs = torch.cat([h, t_expanded], dim=-1)
        else:
            inputs = h
        
        return self.net(inputs)


def demonstrate_ode_fundamentals():
    """
    Complete demonstration of ODE concepts.
    """
    print("=" * 70)
    print("ODE FUNDAMENTALS DEMONSTRATION")
    print("=" * 70)
    
    # =========================================================================
    # Part 1: Compare numerical methods
    # =========================================================================
    print("\n1. Comparing Numerical Methods on Exponential Growth")
    print("-" * 50)
    
    def exp_growth(y, t):
        return y  # dy/dt = y
    
    y0 = torch.tensor([[1.0]])
    t_span = (0.0, 2.0)
    
    # Different step sizes
    step_sizes = [0.5, 0.1, 0.01]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, dt in enumerate(step_sizes):
        ax = axes[idx]
        
        # Euler integration
        t_euler, y_euler = euler_integrate(exp_growth, y0, t_span, dt)
        
        # RK4 integration
        t_rk4, y_rk4 = rk4_integrate(exp_growth, y0, t_span, dt)
        
        # Analytical solution
        t_analytical = torch.linspace(t_span[0], t_span[1], 100)
        y_analytical = exponential_analytical(y0, t_analytical.unsqueeze(-1))
        
        ax.plot(t_analytical, y_analytical.squeeze(), 'k-', linewidth=2, 
                label='Analytical')
        ax.plot(t_euler, y_euler.squeeze(), 'b--o', markersize=4,
                label='Euler')
        ax.plot(t_rk4, y_rk4.squeeze(), 'r--s', markersize=4,
                label='RK4')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('y(t)')
        ax.set_title(f'dt = {dt}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute errors
        euler_error = abs(y_euler[-1, 0, 0].item() - np.exp(2.0))
        rk4_error = abs(y_rk4[-1, 0, 0].item() - np.exp(2.0))
        print(f"dt={dt}: Euler error={euler_error:.6f}, RK4 error={rk4_error:.2e}")
    
    plt.tight_layout()
    plt.savefig('ode_numerical_methods.png', dpi=150)
    plt.show()
    
    # =========================================================================
    # Part 2: Phase portrait of damped oscillator
    # =========================================================================
    print("\n2. Phase Portrait: Damped Oscillator")
    print("-" * 50)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot vector field
    plot_vector_field(damped_oscillator, (-3, 3), (-3, 3), ax=ax)
    
    # Plot trajectories
    initial_conditions = [(2, 0), (0, 2), (-2, 0), (1, 1)]
    plot_trajectories(damped_oscillator, initial_conditions, (0, 15), ax=ax)
    
    ax.set_title('Damped Harmonic Oscillator Phase Portrait')
    plt.savefig('damped_oscillator_phase.png', dpi=150)
    plt.show()
    
    # Analyze fixed point
    print("\nFixed point analysis:")
    analyze_fixed_point(damped_oscillator, [0.0, 0.0])
    
    # =========================================================================
    # Part 3: ResNet vs Neural ODE visualization
    # =========================================================================
    print("\n3. ResNet → Neural ODE Transition")
    print("-" * 50)
    
    # Simple transformation for visualization
    def simple_dynamics(y, t):
        return torch.sin(y)
    
    y0_viz = torch.tensor([[0.5]])
    t_span_viz = (0.0, 3.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Different numbers of "layers" (Euler steps)
    n_layers_list = [3, 10, 30, 100]
    
    # Continuous solution for reference
    t_cont, y_cont = rk4_integrate(simple_dynamics, y0_viz, t_span_viz, dt=0.001)
    
    for idx, n_layers in enumerate(n_layers_list):
        ax = axes[idx // 2, idx % 2]
        
        # Discrete ResNet-like solution
        dt = (t_span_viz[1] - t_span_viz[0]) / n_layers
        t_discrete = torch.linspace(t_span_viz[0], t_span_viz[1], n_layers + 1)
        y_discrete = torch.zeros(n_layers + 1, 1, 1)
        y_discrete[0] = y0_viz
        
        for i in range(n_layers):
            y_discrete[i+1] = euler_step(simple_dynamics, y_discrete[i], 
                                          t_discrete[i].item(), dt)
        
        ax.plot(t_cont, y_cont.squeeze(), 'b-', linewidth=2,
                label='Continuous (Neural ODE)', alpha=0.7)
        ax.plot(t_discrete, y_discrete.squeeze(), 'ro-', markersize=8,
                label=f'Discrete (N={n_layers} layers)')
        
        ax.set_xlabel('Time t (or Layer Index / N)')
        ax.set_ylabel('State h')
        ax.set_title(f'N = {n_layers} Layers (dt = {dt:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ResNet (Discrete) → Neural ODE (Continuous): Increasing Depth',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('resnet_to_neural_ode.png', dpi=150)
    plt.show()
    
    print("\nKey Observation: As N increases, discrete updates approach continuous curve")
    
    # =========================================================================
    # Summary
    # =========================================================================
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
    
    Next: neural_ode.md - Implementing Neural ODEs with torchdiffeq
    """)


if __name__ == "__main__":
    demonstrate_ode_fundamentals()
```

---

## 7. Key Takeaways

1. **ODEs describe continuous-time dynamics** through the equation $\frac{dy}{dt} = f(y, t)$, with the dynamics function $f$ determining how the state evolves.

2. **Numerical integration approximates solutions** when analytical ones don't exist. Euler is simple but inaccurate; RK4 offers a good balance; adaptive methods handle stiff systems.

3. **ResNets are Euler discretizations** of an underlying continuous dynamics. The residual connection $h_{l+1} = h_l + f(h_l)$ is precisely one Euler step with $\Delta t = 1$.

4. **Neural ODEs take the continuous limit**, replacing discrete layers with a continuous transformation defined by an ODE. This enables adaptive computation, guaranteed invertibility, and memory-efficient training.

5. **Phase portraits visualize dynamics** through vector fields and trajectories, revealing fixed points, stability, and qualitative behavior.

---

## 8. Exercises

### Exercise 1: Implement Midpoint Method

The **midpoint method** (RK2) is:

$$k_1 = f(y_n, t_n)$$
$$k_2 = f\left(y_n + \frac{\Delta t}{2}k_1, t_n + \frac{\Delta t}{2}\right)$$
$$y_{n+1} = y_n + \Delta t \cdot k_2$$

Implement this method and compare its accuracy to Euler and RK4.

### Exercise 2: Stability Region

For the test equation $\frac{dy}{dt} = \lambda y$, derive the stability region for:

1. Forward Euler
2. Backward Euler: $y_{n+1} = y_n + \Delta t \cdot f(y_{n+1}, t_{n+1})$
3. RK4

Plot these regions in the complex $\lambda \Delta t$ plane.

### Exercise 3: Van der Pol Oscillator

The Van der Pol oscillator is a nonlinear system:

$$\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0$$

1. Convert to a first-order system
2. Implement the dynamics function
3. Create phase portraits for $\mu = 0.1, 1.0, 5.0$
4. Describe how the behavior changes with $\mu$

### Exercise 4: ResNet Depth Analysis

Empirically investigate how many ResNet layers are needed to approximate a Neural ODE solution to within tolerance $\epsilon$ for a given dynamics function. Plot the relationship between required depth and tolerance.

---

## References

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

3. Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I: Nonstiff Problems. Springer.

4. Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos. Westview Press.
