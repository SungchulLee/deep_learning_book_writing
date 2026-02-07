# Continuous Normalizing Flows

Continuous Normalizing Flows (CNFs) generalise discrete normalizing flows to **continuous-time transformations** defined by ordinary differential equations.  Instead of composing $K$ discrete invertible layers, a CNF defines an infinitely deep transformation through a Neural ODE.  This eliminates the need for restricted architectures—the dynamics network $f$ can be an arbitrary neural network—because ODE flows are always invertible (integrate backwards in time) and always have a well-defined, efficiently estimable change in log-density.

## From Discrete to Continuous

A residual-network flow layer computes $z_{k+1} = z_k + f(z_k, \theta_k)$.  With step size $\Delta t$ this becomes $z_{k+\Delta t} = z_k + \Delta t\,f(z_k, t, \theta)$.  Taking $\Delta t \to 0$ yields the ODE:

$$\frac{dz(t)}{dt} = f(z(t), t;\;\theta)$$

with $z(0) \sim p_Z$ (base distribution) and $z(T) = x$ (data).  The Neural ODE defines:

$$z(T) = z(0) + \int_0^T f(z(t), t;\;\theta)\,dt$$

Because the ODE is always reversible (integrate from $T$ to $0$), the transformation is invertible regardless of $f$'s architecture.

## Instantaneous Change of Variables

For discrete flows the density change uses $\log|\det J|$.  In continuous time:

$$\frac{\partial \log p(z(t))}{\partial t} = -\operatorname{tr}\!\left(\frac{\partial f}{\partial z}\right)$$

Integrating:

$$\log p(x) = \log p_Z(z(0)) - \int_0^T \operatorname{tr}\!\left(\frac{\partial f(z(t), t)}{\partial z}\right) dt$$

The trace replaces the determinant, and the integral replaces the sum over layers.

## The Scalability Problem

Computing the exact trace $\operatorname{tr}(J) = \sum_{i=1}^{D} \partial f_i / \partial z_i$ requires $D$ backward passes—**$O(D^2)$ total cost**—too expensive for high-dimensional data.

## Hutchinson's Trace Estimator

Hutchinson's estimator provides an unbiased estimate of the trace using a single random probe vector:

$$\operatorname{tr}(A) = \mathbb{E}_\epsilon\!\bigl[\epsilon^T A\,\epsilon\bigr]$$

where $\mathbb{E}[\epsilon] = 0$ and $\operatorname{Cov}(\epsilon) = I$.  Rademacher random vectors ($\epsilon_i \in \{-1, +1\}$ uniformly) have lower variance than Gaussians and are preferred in practice.

Computing $\epsilon^T (\partial f / \partial z)\,\epsilon$ requires a single vector-Jacobian product via reverse-mode autodiff, costing $O(D)$.  This reduces the per-step cost from $O(D^2)$ to $O(D)$.

## FFJORD

FFJORD (Free-Form Jacobian of Reversible Dynamics; Grathwohl et al., 2019) combines Neural ODEs with Hutchinson's estimator into a practical CNF.

### Joint ODE

FFJORD solves state and log-density jointly:

$$\frac{d}{dt}\begin{pmatrix} z(t) \\ \Delta\log p \end{pmatrix} = \begin{pmatrix} f(z(t), t;\;\theta) \\ -\epsilon^T \frac{\partial f}{\partial z}\,\epsilon \end{pmatrix}$$

### Implementation

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint


class CNFDynamics(nn.Module):
    """Dynamics network for FFJORD."""

    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim), nn.Tanh(),   # +1 for time
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, state):
        z, _ = state[..., :-1], state[..., -1:]
        t_vec = t.expand(z.shape[0], 1)
        dz = self.net(torch.cat([z, t_vec], dim=-1))

        # Hutchinson trace estimator
        epsilon = torch.randint(0, 2, z.shape, device=z.device).float() * 2 - 1
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            fz = self.net(torch.cat([z_req, t_vec], dim=-1))
            # Vector-Jacobian product
            vjp = torch.autograd.grad(fz, z_req, epsilon,
                                       create_graph=True)[0]
        trace_est = (vjp * epsilon).sum(dim=-1, keepdim=True)

        return torch.cat([dz, -trace_est], dim=-1)


class FFJORD(nn.Module):
    """Free-Form Jacobian of Reversible Dynamics."""

    def __init__(self, dim, hidden_dim=128, T=1.0):
        super().__init__()
        self.dynamics = CNFDynamics(dim, hidden_dim)
        self.T = T
        self.dim = dim

    def forward(self, z):
        """z -> x (base → data)."""
        init = torch.cat([z, torch.zeros(z.shape[0], 1, device=z.device)], -1)
        out = odeint(self.dynamics, init, torch.tensor([0.0, self.T]))
        x = out[-1, :, :self.dim]
        delta_logp = out[-1, :, -1]
        return x, delta_logp

    def inverse(self, x):
        """x -> z (data → base)."""
        init = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], -1)
        out = odeint(self.dynamics, init, torch.tensor([self.T, 0.0]))
        z = out[-1, :, :self.dim]
        delta_logp = out[-1, :, -1]
        return z, delta_logp

    def log_prob(self, x):
        z, delta_logp = self.inverse(x)
        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * 3.14159))).sum(-1)
        return log_pz + delta_logp
```

### The Adjoint Method

Training CNFs requires differentiating through the ODE solver.  The **adjoint method** avoids storing all intermediate states by solving an augmented ODE backwards in time, reducing memory from $O(L)$ (where $L$ is the number of solver steps) to $O(1)$.  This is critical for long integration horizons.

### Solver Choices

| Solver | Order | Adaptive | Use Case |
|---|---|---|---|
| Euler | 1 | No | Debugging |
| RK4 | 4 | No | Moderate precision |
| Dopri5 | 4/5 | Yes | General purpose |
| Adams | Variable | Yes | Smooth dynamics |

Adaptive solvers (Dopri5) automatically adjust step size, allocating more computation where the dynamics change rapidly.

### Regularisation

Unconstrained dynamics can develop stiff, hard-to-integrate trajectories.  Common regularisers include a kinetic-energy penalty $\lambda\int_0^T \|f(z(t),t)\|^2\,dt$ that encourages smooth, straight trajectories, and Jacobian Frobenius norm penalties that limit the complexity of the transformation.

## CNFs vs. Discrete Flows

| Aspect | Discrete Flows | Continuous Flows |
|---|---|---|
| Architecture | Constrained (invertible) | Unconstrained |
| Depth | Fixed $K$ layers | Adaptive (solver-dependent) |
| Log-density | Exact determinant | Stochastic trace estimate |
| Memory | $O(K \cdot d)$ | $O(d)$ with adjoint |
| Training speed | Faster per iteration | Slower (ODE solve) |

CNFs excel when architectural flexibility matters more than raw throughput.  They are particularly appealing for scientific applications (molecular dynamics, physics simulations) and for moderate-dimensional financial data where the ODE overhead is acceptable.

## Finance Applications

For quantitative finance, CNFs offer a compelling modelling approach for continuous-time dynamics of asset returns.  The ODE formulation naturally connects to stochastic differential equation models used in mathematical finance, and the ability to use unconstrained architectures allows the model to discover complex dependence structures without manual specification.

## Key References

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Finzi, I. & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Hutchinson, M. F. (1989). A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines. *Communications in Statistics*.
