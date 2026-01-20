# Continuous Normalizing Flows

## Introduction

Continuous Normalizing Flows (CNFs) generalize discrete normalizing flows to continuous-time transformations defined by ordinary differential equations (ODEs). Instead of composing a finite number of discrete layers, CNFs define an infinitely deep transformation through a neural ODE, enabling more flexible density modeling with unconstrained architectures.

## From Discrete to Continuous

### Discrete Flows Recap

A discrete normalizing flow composes K transformations:

$$\mathbf{z}_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0)$$

With log-likelihood:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}_0) - \sum_{k=1}^{K} \log \left|\det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right|$$

### The Continuous Limit

As K → ∞ with infinitesimal step sizes, the discrete updates become a continuous transformation:

$$\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t; \theta)$$

where:
- $\mathbf{z}(0) = \mathbf{z}_0$ is the initial state (base distribution sample)
- $\mathbf{z}(T) = \mathbf{x}$ is the final state (data)
- $f$ is a neural network defining the dynamics

## Mathematical Foundation

### The Neural ODE

A CNF defines a transformation through the initial value problem:

$$\mathbf{z}(t_0) = \mathbf{z}_0$$
$$\frac{d\mathbf{z}}{dt} = f(\mathbf{z}(t), t; \theta)$$

The solution at time $T$ gives:

$$\mathbf{z}(T) = \mathbf{z}_0 + \int_{t_0}^{T} f(\mathbf{z}(t), t; \theta) \, dt$$

### Instantaneous Change of Variables

For continuous transformations, the change of variables formula becomes:

$$\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right)$$

This is the **instantaneous change of variables** formula, involving the trace of the Jacobian rather than its determinant.

### Log-Likelihood Computation

Integrating over time:

$$\log p(\mathbf{z}(T)) = \log p(\mathbf{z}(0)) - \int_{t_0}^{T} \text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right) dt$$

Or equivalently for data likelihood:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}_0) - \int_{t_0}^{T} \text{tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right) dt$$

## Implementation

### Basic CNF Structure

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class CNFDynamics(nn.Module):
    """
    Neural network defining the ODE dynamics.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64]):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Linear(1, hidden_dims[0])
        
        # Main network
        layers = []
        prev_dim = dim + hidden_dims[0]  # Input + time embedding
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Tanh()
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, z):
        """
        Compute dz/dt.
        
        Args:
            t: Current time (scalar)
            z: Current state (batch, dim)
        
        Returns:
            dz/dt: Time derivative (batch, dim)
        """
        # Embed time
        t_embed = self.time_embed(t.view(1, 1)).expand(z.shape[0], -1)
        
        # Concatenate state and time
        z_t = torch.cat([z, t_embed], dim=-1)
        
        return self.net(z_t)


class CNF(nn.Module):
    """
    Continuous Normalizing Flow.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64], T=1.0):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.dynamics = CNFDynamics(dim, hidden_dims)
    
    def forward(self, z0, reverse=False):
        """
        Integrate ODE from z0.
        
        Args:
            z0: Initial state (batch, dim)
            reverse: If True, integrate backward (for sampling)
        
        Returns:
            zT: Final state
        """
        if reverse:
            t_span = torch.tensor([self.T, 0.0])
        else:
            t_span = torch.tensor([0.0, self.T])
        
        # Solve ODE
        solution = odeint(
            self.dynamics,
            z0,
            t_span,
            method='dopri5',
            rtol=1e-5,
            atol=1e-5
        )
        
        return solution[-1]  # Return final state
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples by integrating from noise."""
        z0 = torch.randn(n_samples, self.dim, device=device)
        return self.forward(z0, reverse=False)
```

### Computing Log-Likelihood with Trace

```python
class CNFWithLogProb(nn.Module):
    """
    CNF with exact log-probability computation.
    Uses augmented ODE to track log-det.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64], T=1.0):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.dynamics = CNFDynamics(dim, hidden_dims)
    
    def _augmented_dynamics(self, t, state):
        """
        Augmented dynamics for joint (z, log_det) integration.
        """
        z = state[:, :self.dim]
        
        # Enable gradient computation for trace
        with torch.enable_grad():
            z = z.requires_grad_(True)
            
            # Compute dynamics
            dz_dt = self.dynamics(t, z)
            
            # Compute trace of Jacobian (expensive: O(dim) backward passes)
            trace = 0.0
            for i in range(self.dim):
                trace += torch.autograd.grad(
                    dz_dt[:, i].sum(), z, create_graph=True
                )[0][:, i]
        
        # d(log_det)/dt = -trace(df/dz)
        d_log_det = -trace.unsqueeze(-1)
        
        return torch.cat([dz_dt, d_log_det], dim=-1)
    
    def forward(self, x):
        """
        Compute latent z and log-det for data x.
        Integrates backward: x (t=T) -> z (t=0)
        """
        batch_size = x.shape[0]
        
        # Initialize augmented state: [x, 0]
        state0 = torch.cat([x, torch.zeros(batch_size, 1, device=x.device)], dim=-1)
        
        # Integrate backward
        t_span = torch.tensor([self.T, 0.0])
        
        solution = odeint(
            self._augmented_dynamics,
            state0,
            t_span,
            method='dopri5'
        )
        
        final_state = solution[-1]
        z = final_state[:, :self.dim]
        log_det = final_state[:, -1]
        
        return z, log_det
    
    def log_prob(self, x):
        """Compute log p(x)."""
        z, log_det = self.forward(x)
        
        # Log-prob under base distribution
        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples."""
        z0 = torch.randn(n_samples, self.dim, device=device)
        
        # Initialize augmented state
        state0 = torch.cat([z0, torch.zeros(n_samples, 1, device=device)], dim=-1)
        
        # Integrate forward
        t_span = torch.tensor([0.0, self.T])
        
        solution = odeint(
            self._augmented_dynamics,
            state0,
            t_span,
            method='dopri5'
        )
        
        return solution[-1][:, :self.dim]
```

## Advantages of CNFs

### 1. Unconstrained Architectures

Unlike discrete flows, CNFs impose **no architectural constraints**:

```python
# Discrete flow: must be invertible
class DiscreteLayer(nn.Module):
    # Restricted architecture (coupling, autoregressive, etc.)
    pass

# CNF: any neural network works
class CNFDynamics(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Can use ANY architecture
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),  # Non-invertible activation is fine!
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )
```

### 2. Memory Efficiency

CNFs use **constant memory** regardless of depth via the adjoint method:

| Model | Memory | Effective Depth |
|-------|--------|-----------------|
| ResNet (100 layers) | O(100) | 100 |
| CNF | O(1) | ∞ |

### 3. Adaptive Computation

ODE solvers adapt step size based on dynamics complexity:

```python
# Solver automatically uses more steps where needed
solution = odeint(
    dynamics,
    z0,
    t_span,
    method='dopri5',  # Adaptive Runge-Kutta
    rtol=1e-5,        # Relative tolerance
    atol=1e-5         # Absolute tolerance
)
```

## The Trace Computation Problem

### Exact Trace is Expensive

Computing $\text{tr}(\partial f / \partial \mathbf{z})$ exactly requires:

$$\text{tr}(J) = \sum_{i=1}^{D} \frac{\partial f_i}{\partial z_i}$$

This needs D backward passes—**O(D²) total cost**.

### Solutions

1. **Hutchinson's trace estimator** (covered in next document)
2. **Restricting architecture** to have efficient trace
3. **Trace-free formulations** (flow matching)

## Training CNFs

### Maximum Likelihood

```python
def train_cnf(model, data, n_epochs=100, batch_size=128, lr=1e-3):
    """Train CNF via maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch, in loader:
            optimizer.zero_grad()
            
            # Compute negative log-likelihood
            loss = -model.log_prob(batch).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, NLL: {total_loss / len(data):.4f}")
```

### Regularization

CNFs benefit from regularization to control dynamics complexity:

```python
class RegularizedCNF(nn.Module):
    def compute_loss(self, x, reg_weight=0.01):
        """Loss with kinetic energy regularization."""
        z, log_det = self.forward(x)
        
        # NLL
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        nll = -(log_pz + log_det).mean()
        
        # Kinetic energy: encourages smooth trajectories
        # ||dz/dt||^2 integrated over time
        kinetic = self._compute_kinetic_energy(x)
        
        return nll + reg_weight * kinetic
```

## Connection to Other Models

### Neural ODEs

CNFs are normalizing flows built on Neural ODEs:

$$\text{CNF} = \text{Neural ODE} + \text{Change of Variables}$$

### Score-Based Models

CNFs connect to score-based diffusion:
- Both model continuous-time transformations
- Diffusion: fixed forward process, learned reverse
- CNF: learned bidirectional process

### Optimal Transport

CNFs can be viewed as learning optimal transport maps:
- Transform one distribution to another
- ODE defines the transport trajectory

## Summary

Continuous Normalizing Flows provide:

1. **Continuous-depth** transformations via ODEs
2. **Unconstrained architectures** for the dynamics network
3. **Memory-efficient** training via adjoint method
4. **Adaptive computation** through ODE solvers
5. **Instantaneous change of variables** with trace instead of determinant

The main challenge is efficient trace computation, addressed by Hutchinson's estimator in FFJORD.

## References

1. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
2. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
