# Continuous Normalizing Flows

## Learning Objectives

By the end of this section, you will:

- Understand how Neural ODEs enable continuous-time normalizing flows
- Derive the instantaneous change of variables formula
- Implement CNFs for density estimation and generative modeling
- Compare CNFs with discrete normalizing flows
- Apply FFJORD for tractable likelihood computation

## Prerequisites

- Neural ODE fundamentals and adjoint method
- Normalizing flows basics (change of variables, invertible transformations)
- Probability density functions and likelihood
- Basic information theory (KL divergence)

---

## 1. From Discrete to Continuous Flows

### 1.1 Discrete Normalizing Flows Review

A **normalizing flow** transforms a simple base distribution $p_0(z_0)$ into a complex target distribution through a sequence of invertible transformations:

$$z_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)$$

The density is computed via the **change of variables formula**:

$$\log p_K(z_K) = \log p_0(z_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|$$

**Challenges with discrete flows:**

1. Architecture constraints (must be invertible with tractable Jacobian)
2. Fixed number of transformations
3. Log-determinant computation can be expensive

### 1.2 The Continuous Perspective

What if instead of discrete transformations, we use a continuous transformation defined by an ODE?

$$\frac{dz}{dt} = f_\theta(z(t), t)$$

Starting from $z(0) \sim p_0$, we transform to $z(T) \sim p_T$ by integrating the ODE.

**Key insight:** The change of variables formula has a beautiful continuous analog!

---

## 2. Instantaneous Change of Variables

### 2.1 The Fundamental Result

**Theorem (Instantaneous Change of Variables):**

For a continuous transformation defined by $\frac{dz}{dt} = f(z, t)$, the log-density evolves as:

$$\frac{d \log p(z(t))}{dt} = -\text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

where $\text{Tr}(\cdot)$ denotes the trace (sum of diagonal elements).

### 2.2 Derivation

Starting from the continuity equation for probability flow:

$$\frac{\partial p}{\partial t} + \nabla \cdot (p f) = 0$$

Using the product rule: $\nabla \cdot (pf) = p(\nabla \cdot f) + f \cdot \nabla p$

And noting that $\frac{d \log p}{dt} = \frac{1}{p}\frac{dp}{dt}$ along a trajectory:

$$\frac{d \log p(z(t))}{dt} = \frac{\partial \log p}{\partial t} + f \cdot \nabla \log p = -\nabla \cdot f = -\text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

### 2.3 Integrated Form

Integrating from $t=0$ to $t=T$:

$$\log p(z(T)) = \log p(z(0)) - \int_0^T \text{Tr}\left(\frac{\partial f}{\partial z}(z(t), t)\right) dt$$

This is the **continuous normalizing flow (CNF) likelihood**.

> **Deep Insight:** Unlike discrete flows where we compute products of Jacobian determinants, CNFs integrate the *trace* of the Jacobian. The trace is just the sum of $d$ diagonal elements—much cheaper than a full determinant!

---

## 3. CNF Architecture

### 3.1 Basic CNF Implementation

```python
import torch
import torch.nn as nn
import torch.distributions as dist
from torchdiffeq import odeint

class CNFDynamics(nn.Module):
    """
    Neural network dynamics for CNF: dz/dt = f(z, t).
    
    Also computes the trace of the Jacobian for density estimation.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Time-conditioned network
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize small for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, state):
        """
        Compute dynamics and trace for augmented state.
        
        Args:
            t: Time
            state: Tuple of (z, log_p) where:
                   z: positions (batch, dim)
                   log_p: accumulated log probability change (batch, 1)
        
        Returns:
            (dz/dt, d(log_p)/dt)
        """
        z, log_p = state
        batch_size = z.shape[0]
        
        # Expand time
        t_vec = t.expand(batch_size, 1)
        
        with torch.enable_grad():
            z = z.requires_grad_(True)
            zt = torch.cat([z, t_vec], dim=-1)
            dzdt = self.net(zt)
            
            # Compute trace of Jacobian
            trace = self._compute_trace(dzdt, z)
        
        # d(log p)/dt = -Tr(df/dz)
        dlog_p_dt = -trace.view(-1, 1)
        
        return dzdt, dlog_p_dt
    
    def _compute_trace(self, f, z):
        """
        Compute trace of Jacobian df/dz.
        
        For exact trace: sum of d partial derivatives
        For approximate: use Hutchinson estimator
        """
        trace = 0.0
        for i in range(self.dim):
            # ∂f_i/∂z_i (diagonal of Jacobian)
            grad_i = torch.autograd.grad(
                f[:, i].sum(), z, 
                create_graph=True, 
                retain_graph=True
            )[0]
            trace = trace + grad_i[:, i]
        
        return trace


class ContinuousNormalizingFlow(nn.Module):
    """
    Complete CNF model for density estimation.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        self.dynamics = CNFDynamics(dim, hidden_dim)
        
        # Standard Gaussian base distribution
        self.base_dist = dist.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)
        )
        
        # Integration times
        self.register_buffer('t_span', torch.tensor([0., 1.]))
    
    def forward(self, z0, reverse=False):
        """
        Transform samples through the flow.
        
        Args:
            z0: Initial samples (batch, dim)
            reverse: If True, go from data to base distribution
        
        Returns:
            z_final: Transformed samples
            log_p_change: Change in log probability
        """
        batch_size = z0.shape[0]
        
        # Initial log probability change = 0
        log_p0 = torch.zeros(batch_size, 1, device=z0.device)
        
        # Set integration direction
        if reverse:
            t = torch.flip(self.t_span, dims=[0])
        else:
            t = self.t_span
        
        # Solve augmented ODE
        state0 = (z0, log_p0)
        states = odeint(self.dynamics, state0, t, method='dopri5',
                       rtol=1e-5, atol=1e-7)
        
        z_final = states[0][-1]
        log_p_change = states[1][-1]
        
        return z_final, log_p_change
    
    def log_prob(self, x):
        """
        Compute log p(x) for data samples.
        
        Process:
        1. Transform x -> z_base via reverse flow
        2. Compute log p(z_base) from base distribution
        3. Add log probability change from transformation
        """
        z_base, log_p_change = self.forward(x, reverse=True)
        log_p_base = self.base_dist.log_prob(z_base).view(-1, 1)
        
        return log_p_base + log_p_change
    
    def sample(self, n_samples):
        """
        Generate samples from the learned distribution.
        
        Process:
        1. Sample from base distribution
        2. Transform through forward flow
        """
        z_base = self.base_dist.sample((n_samples,))
        z_data, _ = self.forward(z_base, reverse=False)
        
        return z_data
```

---

## 4. FFJORD: Free-Form Jacobian

### 4.1 The Trace Computation Problem

Exact trace computation requires $d$ backward passes (one per dimension), which is $O(d^2)$ per sample.

For high-dimensional data, this is prohibitive.

### 4.2 Hutchinson's Trace Estimator

**Hutchinson's trick:** For any matrix $A$,

$$\text{Tr}(A) = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\epsilon^\top A \epsilon]$$

where $p(\epsilon)$ has $\mathbb{E}[\epsilon] = 0$ and $\text{Cov}(\epsilon) = I$.

Common choices:
- Rademacher: $\epsilon_i \in \{-1, +1\}$ with equal probability
- Gaussian: $\epsilon \sim \mathcal{N}(0, I)$

### 4.3 Applying to Jacobian Trace

$$\text{Tr}\left(\frac{\partial f}{\partial z}\right) \approx \epsilon^\top \frac{\partial f}{\partial z} \epsilon = \epsilon^\top \frac{\partial (f^\top \epsilon)}{\partial z}$$

The last equality uses the fact that $\epsilon^\top J \epsilon = \epsilon^\top \nabla_z (f^\top \epsilon)$.

This requires only **one** backward pass regardless of dimension!

```python
class FFJORDDynamics(nn.Module):
    """
    FFJORD dynamics with Hutchinson trace estimator.
    
    Key innovation: O(1) trace estimation instead of O(d).
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t, state):
        """
        Forward with Hutchinson trace estimator.
        
        State: (z, log_p, epsilon) where epsilon is the probe vector
        """
        z, log_p = state[0], state[1]
        
        # During training, use stored epsilon
        # During inference, epsilon doesn't matter (only sampling)
        if len(state) > 2:
            epsilon = state[2]
        else:
            epsilon = torch.randn_like(z)
        
        batch_size = z.shape[0]
        t_vec = t.expand(batch_size, 1)
        
        with torch.enable_grad():
            z = z.requires_grad_(True)
            zt = torch.cat([z, t_vec], dim=-1)
            dzdt = self.net(zt)
            
            # Hutchinson trace estimator
            # Tr(J) ≈ ε^T J ε = ε^T ∇_z(f^T ε)
            f_eps = (dzdt * epsilon).sum()
            grad_f_eps = torch.autograd.grad(f_eps, z, create_graph=True)[0]
            trace_estimate = (grad_f_eps * epsilon).sum(dim=-1)
        
        dlog_p_dt = -trace_estimate.view(-1, 1)
        
        return dzdt, dlog_p_dt, torch.zeros_like(epsilon)


class FFJORD(nn.Module):
    """
    Free-Form Jacobian of Reversible Dynamics.
    
    CNF with efficient Hutchinson trace estimation.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, n_trace_samples: int = 1):
        super().__init__()
        
        self.dim = dim
        self.n_trace_samples = n_trace_samples
        self.dynamics = FFJORDDynamics(dim, hidden_dim)
        
        self.base_dist = dist.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)
        )
        
        self.register_buffer('t_span', torch.tensor([0., 1.]))
    
    def forward(self, z0, reverse=False):
        batch_size = z0.shape[0]
        
        log_p0 = torch.zeros(batch_size, 1, device=z0.device)
        
        # Sample epsilon for trace estimation
        epsilon = torch.randn_like(z0)
        
        if reverse:
            t = torch.flip(self.t_span, dims=[0])
        else:
            t = self.t_span
        
        state0 = (z0, log_p0, epsilon)
        states = odeint(self.dynamics, state0, t, method='dopri5')
        
        z_final = states[0][-1]
        log_p_change = states[1][-1]
        
        return z_final, log_p_change
    
    def log_prob(self, x):
        """Compute log probability with multiple trace samples for variance reduction."""
        log_probs = []
        
        for _ in range(self.n_trace_samples):
            z_base, log_p_change = self.forward(x, reverse=True)
            log_p_base = self.base_dist.log_prob(z_base).view(-1, 1)
            log_probs.append(log_p_base + log_p_change)
        
        # Average over trace samples
        return torch.stack(log_probs).mean(dim=0)
    
    def sample(self, n_samples):
        z_base = self.base_dist.sample((n_samples,))
        z_data, _ = self.forward(z_base, reverse=False)
        return z_data
```

---

## 5. Training CNFs

### 5.1 Maximum Likelihood Training

The training objective is to maximize the log-likelihood:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}}[\log p_\theta(x)]$$

```python
def train_cnf(model, data_loader, n_epochs=100, lr=1e-3):
    """
    Train CNF with maximum likelihood.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            
            optimizer.zero_grad()
            
            # Compute negative log-likelihood
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: NLL = {avg_loss:.4f}")
    
    return losses
```

### 5.2 Training on 2D Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles

def train_cnf_2d_demo():
    """
    Demonstrate CNF training on 2D toy distributions.
    """
    # Generate data
    data_np, _ = make_moons(n_samples=5000, noise=0.05)
    data = torch.tensor(data_np, dtype=torch.float32)
    
    # Normalize
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0)
    data = (data - data_mean) / data_std
    
    # Create model
    model = FFJORD(dim=2, hidden_dim=64)
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train
    losses = train_cnf(model, loader, n_epochs=200, lr=1e-3)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data
    ax = axes[0, 0]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=5)
    ax.set_title('Original Data')
    ax.set_aspect('equal')
    
    # Generated samples
    with torch.no_grad():
        samples = model.sample(5000)
    
    ax = axes[0, 1]
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, c='red')
    ax.set_title('Generated Samples')
    ax.set_aspect('equal')
    
    # Learned density
    ax = axes[0, 2]
    x_grid = torch.linspace(-3, 3, 100)
    y_grid = torch.linspace(-3, 3, 100)
    xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    with torch.no_grad():
        log_probs = model.log_prob(grid)
        probs = torch.exp(log_probs).reshape(100, 100)
    
    ax.contourf(xx, yy, probs, levels=20, cmap='viridis')
    ax.set_title('Learned Density')
    ax.set_aspect('equal')
    
    # Base distribution
    ax = axes[1, 0]
    base_samples = model.base_dist.sample((5000,))
    ax.scatter(base_samples[:, 0], base_samples[:, 1], alpha=0.3, s=5, c='green')
    ax.set_title('Base Distribution')
    ax.set_aspect('equal')
    
    # Data -> Base transformation
    ax = axes[1, 1]
    with torch.no_grad():
        z_base, _ = model.forward(data[:1000], reverse=True)
    ax.scatter(z_base[:, 0], z_base[:, 1], alpha=0.3, s=5)
    ax.set_title('Data → Base')
    ax.set_aspect('equal')
    
    # Training loss
    ax = axes[1, 2]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnf_training_demo.png', dpi=150)
    plt.show()
    
    return model, losses

model, losses = train_cnf_2d_demo()
```

---

## 6. CNF vs Discrete Flows

### 6.1 Comparison Table

| Aspect | Discrete Flows | CNF |
|--------|----------------|-----|
| **Architecture** | Restricted (invertible) | Unrestricted |
| **Jacobian** | Full determinant | Only trace |
| **Computation** | $O(d^3)$ or special | $O(d)$ with Hutchinson |
| **Invertibility** | By construction | ODE solving |
| **Depth** | Fixed | Adaptive |
| **Sampling** | Fast (single pass) | Slow (ODE solve) |
| **Training** | Fast | Slower |

### 6.2 When to Use CNFs

**Advantages of CNFs:**

1. **Flexible architecture:** Any neural network can be the dynamics
2. **Cheaper Jacobian:** Trace vs determinant
3. **Continuous trajectories:** Useful for interpolation, visualization
4. **Memory efficient:** Adjoint method for training

**Disadvantages:**

1. **Slow sampling:** Must solve ODE for each sample
2. **Slow training:** ODE solve in both forward and backward passes
3. **Numerical issues:** ODE solver accuracy affects likelihood

```python
def compare_cnf_discrete_flow():
    """
    Compare CNF with discrete normalizing flow.
    """
    import time
    
    dim = 2
    n_samples = 1000
    
    # CNF
    cnf = FFJORD(dim=dim, hidden_dim=64)
    
    # Time sampling
    start = time.time()
    for _ in range(10):
        _ = cnf.sample(n_samples)
    cnf_sample_time = (time.time() - start) / 10
    
    # Time likelihood
    x = torch.randn(n_samples, dim)
    start = time.time()
    for _ in range(10):
        _ = cnf.log_prob(x)
    cnf_likelihood_time = (time.time() - start) / 10
    
    print(f"CNF Sampling: {cnf_sample_time:.3f}s for {n_samples} samples")
    print(f"CNF Likelihood: {cnf_likelihood_time:.3f}s for {n_samples} samples")
```

---

## 7. Applications

### 7.1 Density Estimation

CNFs excel at modeling complex probability distributions where:
- High flexibility is needed
- Memory is constrained
- Continuous trajectories are meaningful

### 7.2 Variational Inference

CNFs can be used as flexible approximate posteriors in VAEs:

$$q_\phi(z|x) = \text{CNF}(z; \phi, x)$$

This gives more expressive posteriors than simple Gaussians.

### 7.3 Generative Modeling

CNFs provide:
- Exact likelihood for evaluation
- Interpolation between samples (via intermediate ODE states)
- Controllable generation (via modified dynamics)

---

## 8. Key Takeaways

1. **CNFs use Neural ODEs for normalizing flows**, transforming base distributions continuously.

2. **The instantaneous change of variables** formula: $\frac{d \log p}{dt} = -\text{Tr}(\frac{\partial f}{\partial z})$

3. **FFJORD uses Hutchinson trace estimation** for $O(1)$ complexity regardless of dimension.

4. **CNFs trade computation for flexibility**: Slower than discrete flows but unrestricted architectures.

5. **Training uses maximum likelihood** with negative log-likelihood loss.

---

## 9. Exercises

### Exercise 1: Multi-Modal Distribution

Train a CNF on a mixture of 8 Gaussians arranged in a circle. Visualize the learned flow trajectories.

### Exercise 2: Variance of Hutchinson Estimator

Investigate how the number of Hutchinson samples affects:
- Gradient variance
- Training stability
- Final likelihood

### Exercise 3: Regularization

Implement kinetic energy regularization for CNFs and study its effect on the learned trajectories.

---

## References

1. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.

2. Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations. *NeurIPS*.

3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.

4. Hutchinson, M. F. (1989). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics*.
