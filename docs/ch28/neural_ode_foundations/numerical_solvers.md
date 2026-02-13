# Numerical Solvers

## Overview

Neural ODEs require numerical integration of $dz/dt = f_\theta(z(t), t)$. The choice of solver directly affects accuracy, computational cost, and memory usage.

## Fixed-Step Solvers

### Euler Method
$$z_{n+1} = z_n + h \cdot f_\theta(z_n, t_n)$$

Simplest solver. First-order accurate ($O(h)$ local error). Fast but inaccurate unless $h$ is very small.

### Runge-Kutta 4 (RK4)
$$z_{n+1} = z_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_1 = f(z_n, t_n)$, $k_2 = f(z_n + hk_1/2, t_n + h/2)$, etc. Fourth-order accurate. Good balance of accuracy and cost for smooth dynamics.

## Adaptive Solvers

### Dormand-Prince (dopri5)
The default solver in `torchdiffeq`. Fifth-order method with embedded fourth-order estimate for step size control:

$$h_{\text{new}} = h \cdot \left(\frac{\text{tol}}{\text{err}}\right)^{1/5}$$

Adaptive solvers adjust step size automatically: take larger steps when the dynamics are smooth, smaller steps when they change rapidly.

```python
from torchdiffeq import odeint

z_T = odeint(dynamics_fn, z_0, t_span, method='dopri5',
             atol=1e-5, rtol=1e-5)
```

## Solver Selection Guidelines

| Solver | Order | NFE per Step | Use Case |
|--------|-------|-------------|----------|
| Euler | 1 | 1 | Quick prototyping |
| Midpoint | 2 | 2 | Moderate accuracy |
| RK4 | 4 | 4 | Fixed-step, smooth dynamics |
| Dormand-Prince | 5 | 6 | General purpose (default) |
| Adams-Bashforth | Variable | 1 (multistep) | Long integrations |

## Number of Function Evaluations (NFE)

NFE is the key efficiency metric for Neural ODEs. Each function evaluation requires a full forward pass through $f_\theta$. Adaptive solvers have variable NFE per sample, which complicates batching and GPU utilization.

## Stiff Dynamics

If $f_\theta$ has widely varying timescales (stiff ODE), explicit solvers require very small step sizes. Implicit solvers (backward Euler, BDF) handle stiffness but require solving a nonlinear system at each step, which is expensive for neural ODEs.
