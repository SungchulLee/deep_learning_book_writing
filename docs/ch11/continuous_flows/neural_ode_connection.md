# Connection to Neural ODEs

## Introduction

Continuous Normalizing Flows (CNFs) are built upon Neural Ordinary Differential Equations (Neural ODEs), a framework that parameterizes continuous-depth networks using differential equations. Understanding this connection illuminates why CNFs can use unconstrained architectures, how they achieve memory-efficient training, and their relationship to other continuous-time models in machine learning.

## Neural ODEs: Foundation

### From ResNets to ODEs

A residual network computes:

$$\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta_t)$$

As we add more layers with smaller steps:

$$\mathbf{h}_{t+\Delta t} = \mathbf{h}_t + \Delta t \cdot f(\mathbf{h}_t, t, \theta)$$

Taking $\Delta t \to 0$, we get an ODE:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$$

### The Neural ODE Framework

A Neural ODE defines a continuous transformation:

$$\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t, \theta) \, dt$$

where:
- $\mathbf{h}(t_0)$: Initial state (input)
- $\mathbf{h}(t_1)$: Final state (output)
- $f$: Neural network defining dynamics
- Integration performed by ODE solver

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class NeuralODE(nn.Module):
    """Basic Neural ODE layer."""
    
    def __init__(self, dynamics_net):
        super().__init__()
        self.dynamics = dynamics_net
    
    def forward(self, h0, t_span):
        """
        Integrate from t_span[0] to t_span[-1].
        
        Args:
            h0: Initial state (batch, dim)
            t_span: Time points [t0, t1] or [t0, t1, ..., tn]
        
        Returns:
            Solution at final time (or all times if len(t_span) > 2)
        """
        solution = odeint(
            self.dynamics,
            h0,
            t_span,
            method='dopri5'
        )
        return solution[-1]  # Return final state
```

## From Neural ODE to CNF

### Adding Density Tracking

A Neural ODE transforms points, but for a normalizing flow, we need to track how probability density changes. This requires the **instantaneous change of variables**:

$$\frac{d \log p(\mathbf{h}(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{h}}\right)$$

### The Augmented ODE

CNFs solve an augmented system jointly:

$$\frac{d}{dt}\begin{pmatrix} \mathbf{h}(t) \\ \log p(\mathbf{h}(t)) \end{pmatrix} = \begin{pmatrix} f(\mathbf{h}(t), t) \\ -\text{tr}\left(\frac{\partial f}{\partial \mathbf{h}}\right) \end{pmatrix}$$

```python
class CNFDynamics(nn.Module):
    """Augmented dynamics for CNF: state + log-density."""
    
    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.state_dim = state_dim
        self.net = self._build_net(state_dim, hidden_dims)
    
    def _build_net(self, dim, hidden_dims):
        layers = []
        prev = dim + 1  # +1 for time
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, dim))
        return nn.Sequential(*layers)
    
    def forward(self, t, state):
        """
        Compute d/dt [h, log_p].
        
        state: (batch, state_dim + 1) where last dim is log_p
        """
        h = state[:, :self.state_dim]
        
        with torch.enable_grad():
            h = h.requires_grad_(True)
            t_vec = t.expand(h.shape[0], 1)
            
            # Dynamics
            f = self.net(torch.cat([h, t_vec], dim=-1))
            
            # Trace of Jacobian (using Hutchinson estimator)
            trace = self._hutchinson_trace(f, h)
        
        # d(log_p)/dt = -tr(df/dh)
        d_log_p = -trace.unsqueeze(-1)
        
        return torch.cat([f, d_log_p], dim=-1)
    
    def _hutchinson_trace(self, f, h):
        """Estimate trace using Hutchinson's method."""
        eps = torch.randint(0, 2, h.shape, device=h.device).float() * 2 - 1
        f_eps = (f * eps).sum()
        grad_f_eps = torch.autograd.grad(f_eps, h, create_graph=True)[0]
        return (grad_f_eps * eps).sum(dim=-1)
```

## The Adjoint Method

### Memory-Efficient Backpropagation

Standard backpropagation through an ODE solver stores all intermediate states—memory grows with integration steps. The **adjoint method** computes gradients by solving another ODE backward in time.

### Adjoint Equations

Define the adjoint state:

$$\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{h}(t)}$$

The adjoint satisfies:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f}{\partial \mathbf{h}}$$

Parameter gradients:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f}{\partial \theta} \, dt$$

### Implementation

```python
class NeuralODEAdjoint(torch.autograd.Function):
    """Neural ODE with adjoint method for memory efficiency."""
    
    @staticmethod
    def forward(ctx, h0, t_span, dynamics, *params):
        with torch.no_grad():
            solution = odeint(dynamics, h0, t_span, method='dopri5')
        
        ctx.dynamics = dynamics
        ctx.t_span = t_span
        ctx.save_for_backward(solution)
        
        return solution[-1]
    
    @staticmethod
    def backward(ctx, grad_output):
        dynamics = ctx.dynamics
        t_span = ctx.t_span
        solution, = ctx.saved_tensors
        
        h1 = solution[-1]
        
        # Augmented state: [h, adjoint, param_gradients]
        def augmented_dynamics(t, aug_state):
            h = aug_state[:, :h1.shape[-1]]
            a = aug_state[:, h1.shape[-1]:2*h1.shape[-1]]
            
            with torch.enable_grad():
                h = h.requires_grad_(True)
                f = dynamics(t, h)
                
                # Adjoint dynamics: da/dt = -a^T df/dh
                a_f = (a * f).sum()
                dfh = torch.autograd.grad(a_f, h, create_graph=False)[0]
            
            return torch.cat([-f, dfh], dim=-1)  # Negative time direction
        
        # Initial adjoint is grad_output
        aug0 = torch.cat([h1, grad_output], dim=-1)
        
        # Integrate backward
        t_backward = t_span.flip(0)
        aug_solution = odeint(augmented_dynamics, aug0, t_backward)
        
        # Extract gradient w.r.t. initial state
        grad_h0 = aug_solution[-1][:, h1.shape[-1]:]
        
        return grad_h0, None, None, None
```

### Memory Comparison

| Method | Memory | Compute |
|--------|--------|---------|
| Backprop through solver | O(L × D) | O(L × D) |
| Adjoint method | O(D) | O(2 × L × D) |

where L = number of solver steps, D = state dimension.

## Why Unconstrained Architectures?

### Discrete Flows: Architectural Constraints

Discrete normalizing flows require:
1. **Invertibility**: Must compute $f^{-1}$
2. **Tractable Jacobian**: $\det(J)$ must be efficient

This limits architectures to coupling layers, autoregressive transforms, etc.

### CNFs: Freedom via ODEs

For CNFs, the transformation is defined implicitly:

$$\mathbf{x} = \mathbf{z} + \int_0^T f(\mathbf{h}(t), t) \, dt$$

**Key insight**: The ODE is always invertible (by integrating backward) regardless of $f$'s architecture!

```python
# Discrete flow: MUST use special architecture
class CouplingLayer(nn.Module):
    # Restricted to preserve invertibility
    pass

# CNF: ANY architecture works
class CNFDynamics(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Can use anything!
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.ReLU(),           # Non-invertible? No problem!
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),     # Stochastic? Still works!
            nn.Linear(128, dim)
        )
```

### Invertibility Guarantee

For any smooth $f$, the ODE flow is a diffeomorphism:
- **Existence**: Solutions exist for smooth $f$
- **Uniqueness**: Given initial condition, solution is unique
- **Invertibility**: Integrate forward or backward in time

## Instantaneous Change of Variables

### Discrete vs Continuous

**Discrete** (change of variables):
$$\log p(\mathbf{x}) = \log p(\mathbf{z}) - \log |\det J_f|$$

**Continuous** (instantaneous change):
$$\frac{d \log p(\mathbf{h}(t))}{dt} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{h}}\right)$$

### Derivation

Starting from the Fokker-Planck equation for deterministic dynamics:

$$\frac{\partial p}{\partial t} = -\nabla \cdot (p \cdot f)$$

For a point following the ODE:

$$\frac{d \log p(\mathbf{h}(t))}{dt} = -\nabla \cdot f = -\sum_i \frac{\partial f_i}{\partial h_i} = -\text{tr}(J)$$

### Why Trace Instead of Determinant?

| Operation | Discrete | Continuous |
|-----------|----------|------------|
| Formula | $\det(J)$ | $\text{tr}(J)$ |
| Exact cost | O(D³) | O(D²) |
| With structure | O(D) | O(D) |
| Hutchinson | N/A | O(D) |

The trace is fundamentally cheaper and admits efficient stochastic estimation.

## Connections to Other Models

### Score-Based Diffusion Models

Both CNFs and diffusion models use continuous-time dynamics:

| Aspect | CNF | Diffusion |
|--------|-----|-----------|
| Forward | Learned ODE | Fixed SDE (add noise) |
| Reverse | Reverse ODE | Learned SDE (denoise) |
| Training | Max likelihood | Score matching |
| Sampling | ODE solve | SDE solve |

**Connection**: The probability flow ODE of a diffusion model is a CNF!

```python
# Diffusion probability flow ODE
# dh/dt = f(h,t) - 0.5 * g(t)^2 * score(h,t)

# This is equivalent to a CNF with specific dynamics
```

### Optimal Transport

CNFs can be viewed as learning optimal transport maps:

- **Source**: Base distribution $p_0(\mathbf{z})$
- **Target**: Data distribution $p_1(\mathbf{x})$
- **Transport**: ODE trajectory from $\mathbf{z}$ to $\mathbf{x}$

With appropriate regularization, CNFs approximate the Wasserstein-2 optimal transport.

### Hamiltonian Dynamics

Hamiltonian Neural Networks use:

$$\frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}}$$

This is volume-preserving ($\text{tr}(J) = 0$), giving $\log|\det J| = 0$.

## Practical Considerations

### ODE Solver Choice

```python
# Adaptive solver (recommended for training)
solution = odeint(dynamics, h0, t_span, method='dopri5', rtol=1e-5, atol=1e-5)

# Fixed-step solver (faster, less accurate)
solution = odeint(dynamics, h0, t_span, method='euler', options={'step_size': 0.1})

# For stiff dynamics
solution = odeint(dynamics, h0, t_span, method='implicit_adams')
```

### Regularization for Faster Solving

Encourage simple dynamics for fewer solver steps:

```python
def compute_loss_with_reg(model, x, reg_weight=0.01):
    # Standard NLL
    z, log_det = model.forward(x)
    log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
    nll = -(log_pz + log_det).mean()
    
    # Kinetic energy regularization: ||f||^2
    # Encourages straight trajectories
    kinetic = model.compute_kinetic_energy(x)
    
    # Jacobian Frobenius norm: ||J||_F^2
    # Encourages simple dynamics
    jac_norm = model.compute_jacobian_norm(x)
    
    return nll + reg_weight * (kinetic + jac_norm)
```

### Integration Time

The integration interval $[0, T]$ affects expressiveness:

```python
class CNF(nn.Module):
    def __init__(self, dim, T=1.0):
        self.T = T  # Can be learned!
        # Longer T = more expressive but slower
        
    def forward(self, x):
        t_span = torch.tensor([0.0, self.T])
        # ...
```

## Summary

The Neural ODE foundation provides CNFs with:

1. **Unconstrained architectures**: Any smooth network defines valid dynamics
2. **Memory efficiency**: O(1) memory via adjoint method
3. **Adaptive computation**: Solver adjusts to dynamics complexity
4. **Theoretical grounding**: Connects to ODEs, optimal transport, diffusion

Key relationships:

$$\text{ResNet} \xrightarrow{\Delta t \to 0} \text{Neural ODE} \xrightarrow{+ \text{density tracking}} \text{CNF}$$

$$\text{CNF} \xleftrightarrow{\text{probability flow}} \text{Diffusion Model}$$

Understanding the Neural ODE connection reveals why CNFs are powerful: they inherit the continuous-depth flexibility of Neural ODEs while adding exact density computation through the instantaneous change of variables formula.

## References

1. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Pontryagin, L. S. (1962). Mathematical Theory of Optimal Processes.
4. Song, Y., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.
5. Finlay, C., et al. (2020). How to Train Your Neural ODE. *ICML*.
