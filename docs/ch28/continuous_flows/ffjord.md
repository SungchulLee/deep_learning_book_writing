# FFJORD

## Overview

Free-Form Jacobian of Reversible Dynamics (FFJORD, Grathwohl et al., 2019) makes CNFs practical by replacing the expensive trace computation with a stochastic estimator, enabling unrestricted neural network architectures.

!!! note "See Also"
    For a comprehensive treatment of FFJORD including implementation, training, and comparison with discrete flows, see **Section 23.3 Continuous Normalizing Flows**.

## The Trace Problem

Computing the exact trace of the Jacobian $\text{tr}(\partial f / \partial z)$ requires $O(d)$ evaluations of $f_\theta$ (one per dimension) or $O(d^2)$ memory for the full Jacobian. For high-dimensional data, this is prohibitive.

## Hutchinson's Trace Estimator

FFJORD uses the identity:

$$\text{tr}(A) = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\epsilon^T A \epsilon]$$

where $\epsilon$ is a random vector with $\mathbb{E}[\epsilon\epsilon^T] = I$ (e.g., standard Gaussian or Rademacher).

This gives an unbiased estimate of the trace using a single vector-Jacobian product:

$$\text{tr}\left(\frac{\partial f}{\partial z}\right) \approx \epsilon^T \frac{\partial f}{\partial z} \epsilon$$

The vector-Jacobian product $\epsilon^T (\partial f / \partial z)$ is computed efficiently via reverse-mode autodiff in $O(d)$ time.

## Training Objective

FFJORD maximizes the log-likelihood:

$$\log p(x) = \log p(z_0) + \int_0^1 \text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right) dt$$

where $z_0 = z(0)$ is obtained by solving the ODE backward from $x = z(1)$.

## Implementation

```python
class FFJORD(nn.Module):
    def __init__(self, dynamics_net):
        super().__init__()
        self.dynamics = dynamics_net
    
    def forward(self, x):
        # Augment state with log-density
        # Solve [dz/dt, d(log p)/dt] jointly
        state0 = (x, torch.zeros(x.shape[0]))
        
        def augmented_dynamics(t, state):
            z, _ = state
            z.requires_grad_(True)
            dz_dt = self.dynamics(z, t)
            
            # Hutchinson trace estimator
            epsilon = torch.randn_like(z)
            vjp = torch.autograd.grad(dz_dt, z, epsilon, create_graph=True)[0]
            trace_est = (vjp * epsilon).sum(dim=-1)
            
            return dz_dt, -trace_est
        
        zT, delta_logp = odeint(augmented_dynamics, state0, 
                                 torch.tensor([0., 1.]))
        return zT[-1], delta_logp[-1]
```

## Advantages and Limitations

**Advantages**: free-form architecture, exact likelihood, memory-efficient training (via adjoint method), continuous-time dynamics.

**Limitations**: training is slow due to ODE solves, adaptive solvers make wall-clock time unpredictable, and sampling requires solving the ODE forward (no closed-form inverse).
