# FFJORD: Free-form Continuous Dynamics

## Introduction

FFJORD (Free-Form Jacobian of Reversible Dynamics) makes Continuous Normalizing Flows practical by using Hutchinson's trace estimator to compute log-likelihoods efficiently. This reduces the cost from O(D²) to O(D), enabling CNFs to scale to high-dimensional data while maintaining exact likelihood computation.

## The Scalability Challenge

### Exact Trace is Expensive

For a CNF, the change in log-probability requires:

$$\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$$

Computing this trace exactly requires D partial derivatives:

$$\text{tr}(J) = \sum_{i=1}^{D} \frac{\partial f_i}{\partial z_i}$$

Each diagonal element needs one backward pass → **O(D²) total cost**.

### FFJORD's Solution

FFJORD uses **Hutchinson's stochastic trace estimator**:

$$\text{tr}(A) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}\right]$$

where $\boldsymbol{\epsilon}$ is a random vector with $\mathbb{E}[\boldsymbol{\epsilon}] = 0$ and $\text{Cov}(\boldsymbol{\epsilon}) = I$.

This reduces cost to **O(D)** per sample!

## Mathematical Foundation

### Hutchinson's Trace Estimator

For any matrix $A \in \mathbb{R}^{D \times D}$:

$$\text{tr}(A) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}\right]$$

**Proof**:
$$\mathbb{E}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] = \mathbb{E}\left[\sum_{i,j} \epsilon_i A_{ij} \epsilon_j\right] = \sum_{i,j} A_{ij} \mathbb{E}[\epsilon_i \epsilon_j] = \sum_i A_{ii} = \text{tr}(A)$$

### Applied to CNFs

For the Jacobian $J = \partial f / \partial \mathbf{z}$:

$$\text{tr}(J) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T \frac{\partial f}{\partial \mathbf{z}} \boldsymbol{\epsilon}\right] = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T \frac{\partial (f^T \boldsymbol{\epsilon})}{\partial \mathbf{z}}\right]$$

The key insight: $\boldsymbol{\epsilon}^T J \boldsymbol{\epsilon}$ can be computed with a **single backward pass** using vector-Jacobian products!

### Noise Distributions

Common choices for $\boldsymbol{\epsilon}$:

1. **Rademacher**: $\epsilon_i \in \{-1, +1\}$ with equal probability
2. **Gaussian**: $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$

Rademacher often has lower variance in practice.

## Implementation

### FFJORD Dynamics

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class FFJORDDynamics(nn.Module):
    """
    FFJORD dynamics with Hutchinson trace estimator.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64]):
        super().__init__()
        self.dim = dim
        
        # Dynamics network (can be any architecture)
        layers = []
        prev_dim = dim + 1  # dim + time
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Softplus()
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, state):
        """
        Compute augmented dynamics [dz/dt, d(log_det)/dt].
        
        Args:
            t: Time (scalar)
            state: [z, log_det, epsilon] concatenated
        """
        z = state[:, :self.dim]
        epsilon = state[:, self.dim + 1:]  # Noise for trace estimation
        
        with torch.enable_grad():
            z = z.requires_grad_(True)
            
            # Concatenate z and t
            t_vec = t.expand(z.shape[0], 1)
            z_t = torch.cat([z, t_vec], dim=-1)
            
            # Compute f(z, t)
            f = self.net(z_t)
            
            # Hutchinson trace estimator: tr(J) ≈ ε^T J ε
            # Compute ε^T J using vector-Jacobian product
            f_eps = (f * epsilon).sum()
            grad_f_eps = torch.autograd.grad(f_eps, z, create_graph=True)[0]
            trace_estimate = (grad_f_eps * epsilon).sum(dim=-1, keepdim=True)
        
        # d(log_det)/dt = -tr(df/dz)
        d_log_det = -trace_estimate
        
        # epsilon stays constant
        d_epsilon = torch.zeros_like(epsilon)
        
        return torch.cat([f, d_log_det, d_epsilon], dim=-1)


class FFJORD(nn.Module):
    """
    Free-Form Jacobian of Reversible Dynamics.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64], T=1.0):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.dynamics = FFJORDDynamics(dim, hidden_dims)
    
    def forward(self, x, reverse=False):
        """
        Transform x through the flow.
        
        Args:
            x: Input data (batch, dim)
            reverse: If True, integrate t: T -> 0 (for encoding)
        
        Returns:
            z: Transformed data
            log_det: Log determinant of transformation
        """
        batch_size = x.shape[0]
        
        # Sample noise for Hutchinson estimator (Rademacher)
        epsilon = torch.randint(0, 2, (batch_size, self.dim), device=x.device).float() * 2 - 1
        
        # Initial augmented state: [x, 0, epsilon]
        state0 = torch.cat([
            x,
            torch.zeros(batch_size, 1, device=x.device),
            epsilon
        ], dim=-1)
        
        # Integration direction
        if reverse:
            t_span = torch.tensor([self.T, 0.0], device=x.device)
        else:
            t_span = torch.tensor([0.0, self.T], device=x.device)
        
        # Solve ODE
        solution = odeint(
            self.dynamics,
            state0,
            t_span,
            method='dopri5',
            rtol=1e-5,
            atol=1e-5
        )
        
        final_state = solution[-1]
        z = final_state[:, :self.dim]
        log_det = final_state[:, self.dim]
        
        return z, log_det
    
    def log_prob(self, x):
        """
        Compute log p(x).
        Integrates backward: x (t=T) -> z (t=0)
        """
        z, log_det = self.forward(x, reverse=True)
        
        # Log-prob under standard Gaussian base
        log_pz = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        """
        Generate samples.
        Integrates forward: z (t=0) -> x (t=T)
        """
        z0 = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.forward(z0, reverse=False)
        return x
```

### Multiple Noise Samples for Lower Variance

```python
class FFJORDLowVariance(nn.Module):
    """FFJORD with multiple noise samples for lower variance trace estimates."""
    
    def __init__(self, dim, hidden_dims=[64, 64], T=1.0, n_trace_samples=1):
        super().__init__()
        self.dim = dim
        self.T = T
        self.n_trace_samples = n_trace_samples
        self.dynamics_net = self._build_net(dim, hidden_dims)
    
    def _build_net(self, dim, hidden_dims):
        layers = []
        prev_dim = dim + 1
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.Tanh()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, dim))
        return nn.Sequential(*layers)
    
    def _dynamics(self, t, z, epsilon_list):
        """Compute f and trace estimate with multiple noise samples."""
        with torch.enable_grad():
            z = z.requires_grad_(True)
            t_vec = t.expand(z.shape[0], 1)
            f = self.dynamics_net(torch.cat([z, t_vec], dim=-1))
            
            # Average trace estimate over multiple noise samples
            trace_estimates = []
            for epsilon in epsilon_list:
                f_eps = (f * epsilon).sum()
                grad_f_eps = torch.autograd.grad(f_eps, z, retain_graph=True)[0]
                trace_estimates.append((grad_f_eps * epsilon).sum(dim=-1))
            
            trace = torch.stack(trace_estimates).mean(dim=0)
        
        return f, -trace
    
    def log_prob(self, x):
        """Compute log-prob with averaged trace estimates."""
        batch_size = x.shape[0]
        
        # Multiple noise samples
        epsilon_list = [
            torch.randint(0, 2, (batch_size, self.dim), device=x.device).float() * 2 - 1
            for _ in range(self.n_trace_samples)
        ]
        
        # Manual integration with trace computation
        z = x.clone()
        log_det = torch.zeros(batch_size, device=x.device)
        
        n_steps = 100
        dt = -self.T / n_steps  # Backward integration
        t = self.T
        
        for _ in range(n_steps):
            f, d_log_det = self._dynamics(torch.tensor(t), z, epsilon_list)
            z = z + f * dt
            log_det = log_det + d_log_det * (-dt)  # Note sign
            t = t + dt
        
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
```

## Training FFJORD

### Basic Training Loop

```python
def train_ffjord(model, data, n_epochs=100, batch_size=128, lr=1e-3):
    """Train FFJORD via maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch, in loader:
            optimizer.zero_grad()
            
            loss = -model.log_prob(batch).mean()
            loss.backward()
            
            # Gradient clipping helps stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item() * len(batch)
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch+1}, NLL: {avg_loss:.4f}")
```

### Regularization Techniques

```python
class RegularizedFFJORD(FFJORD):
    """FFJORD with regularization for better training."""
    
    def compute_loss(self, x, kinetic_weight=0.01, jacobian_weight=0.01):
        """
        Loss with regularization terms.
        
        Regularizers:
        - Kinetic energy: ||f||^2 encourages smooth trajectories
        - Jacobian norm: ||J||_F^2 encourages simple dynamics
        """
        batch_size = x.shape[0]
        
        # Sample noise
        epsilon = torch.randint(0, 2, (batch_size, self.dim), 
                               device=x.device).float() * 2 - 1
        
        # Integration with regularization tracking
        z = x.clone()
        log_det = torch.zeros(batch_size, device=x.device)
        kinetic_energy = 0.0
        jacobian_norm = 0.0
        
        n_steps = 50
        dt = self.T / n_steps
        
        for step in range(n_steps):
            t = torch.tensor(self.T - step * dt)
            
            with torch.enable_grad():
                z_req = z.requires_grad_(True)
                t_vec = t.expand(batch_size, 1)
                f = self.dynamics.net(torch.cat([z_req, t_vec], dim=-1))
                
                # Trace estimate
                f_eps = (f * epsilon).sum()
                grad_f_eps = torch.autograd.grad(f_eps, z_req, create_graph=True)[0]
                trace = (grad_f_eps * epsilon).sum(dim=-1)
                
                # Kinetic energy: ||f||^2
                kinetic_energy += (f ** 2).sum(dim=-1).mean() * dt
                
                # Frobenius norm estimate: ||J||_F^2 ≈ E[||J ε||^2]
                jacobian_norm += (grad_f_eps ** 2).sum(dim=-1).mean() * dt
            
            z = z - f * dt  # Backward integration
            log_det = log_det + trace * dt
        
        # NLL
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        nll = -(log_pz + log_det).mean()
        
        # Total loss
        loss = nll + kinetic_weight * kinetic_energy + jacobian_weight * jacobian_norm
        
        return loss, {
            'nll': nll.item(),
            'kinetic': kinetic_energy.item(),
            'jacobian': jacobian_norm.item()
        }
```

## Architectural Choices

### Time Conditioning

Several ways to incorporate time:

```python
# 1. Concatenation (simple)
def forward_concat(self, z, t):
    return self.net(torch.cat([z, t], dim=-1))

# 2. Hypernetwork (expressive)
class HypernetDynamics(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.hyper = nn.Linear(1, hidden_dim * dim)
        self.output = nn.Linear(hidden_dim, dim)
    
    def forward(self, z, t):
        # Generate weights from time
        weights = self.hyper(t).view(-1, self.hidden_dim, self.dim)
        h = torch.bmm(weights, z.unsqueeze(-1)).squeeze(-1)
        return self.output(torch.tanh(h))

# 3. Concatenation at each layer
class ConcatSquashDynamics(nn.Module):
    def forward(self, z, t):
        h = z
        for layer in self.layers[:-1]:
            h = torch.cat([h, t.expand(h.shape[0], 1)], dim=-1)
            h = layer(h)
            h = torch.tanh(h)
        return self.layers[-1](h)
```

### Activation Functions

Smooth activations work better for ODEs:

```python
# Good choices
nn.Tanh()       # Smooth, bounded
nn.Softplus()   # Smooth, positive
nn.ELU()        # Smooth

# Avoid
nn.ReLU()       # Not smooth at 0
nn.LeakyReLU()  # Not smooth at 0
```

## Comparison with Discrete Flows

| Aspect | Discrete Flows | FFJORD |
|--------|---------------|--------|
| Architecture | Constrained | Free-form |
| Memory | O(depth) | O(1) |
| Computation | Fixed | Adaptive |
| Jacobian | Determinant | Trace |
| Exact likelihood | Yes | Yes (stochastic) |

## Summary

FFJORD enables practical continuous normalizing flows through:

1. **Hutchinson's estimator**: O(D) trace computation instead of O(D²)
2. **Vector-Jacobian products**: Single backward pass per trace estimate
3. **Free-form architecture**: No invertibility constraints
4. **Adaptive computation**: ODE solver adjusts to dynamics complexity

Key formula:
$$\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right) \approx \boldsymbol{\epsilon}^T \frac{\partial f}{\partial \mathbf{z}} \boldsymbol{\epsilon}$$

This estimator is unbiased, and variance can be reduced with multiple samples.

## References

1. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
2. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
3. Hutchinson, M. F. (1989). A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines. *Communications in Statistics*.
