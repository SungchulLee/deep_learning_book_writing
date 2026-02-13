# CNF Fundamentals

## Overview

Continuous Normalizing Flows (CNFs) replace the discrete sequence of transformations in standard normalizing flows with a continuous-time transformation defined by an ordinary differential equation (ODE). This eliminates architectural constraints while providing exact density evaluation.

## From Discrete to Continuous

Standard normalizing flows compose $K$ discrete transformations:

$$z_K = f_K \circ \cdots \circ f_1(z_0)$$

CNFs take the limit $K \to \infty$, defining the transformation as an ODE:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t)$$

where $f_\theta$ is a neural network parameterizing the velocity field.

## Density Evolution

The log-density evolves according to the **instantaneous change of variables formula**:

$$\frac{\partial \log p(z(t))}{\partial t} = -\text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right)$$

This is the continuous analog of the discrete change of variables formula $\log |\det J|$.

## Key Advantage: No Architectural Constraints

Discrete normalizing flows require invertible transformations with tractable Jacobian determinants, which severely constrains the architecture. CNFs have no such constraint: $f_\theta$ can be any neural network, because invertibility is guaranteed by the existence and uniqueness theorem for ODEs (assuming $f_\theta$ is Lipschitz continuous).

## Integration

The ODE is solved numerically using standard solvers:

```python
from torchdiffeq import odeint

def cnf_forward(z0, t_span, dynamics_fn):
    # z0: initial samples, t_span: [0, 1]
    z_t = odeint(dynamics_fn, z0, t_span, method='dopri5')
    return z_t[-1]
```

Adaptive solvers (Dormand-Prince, RK45) adjust step size for accuracy, but this makes the computational cost data-dependent and harder to predict.
