# FFJORD: Free-Form Continuous Dynamics

## Learning Objectives

By the end of this section, you will:

- Understand how FFJORD makes continuous normalizing flows practical
- Implement the Hutchinson trace estimator for efficient log-density computation
- Train FFJORD models for high-dimensional density estimation
- Apply variance reduction techniques for stable training
- Compare FFJORD with constrained normalizing flow architectures

## Prerequisites

- Continuous normalizing flows and instantaneous change of variables (Section 27.2)
- Hutchinson's trace estimator (introduced in CNF section)
- Neural ODE training with adjoint method (Section 27.1)
- Maximum likelihood estimation

---

## 1. Motivation: Scaling CNFs

### 1.1 The Trace Bottleneck

The instantaneous change of variables formula requires the trace of the Jacobian:

$$\frac{d}{dt} \log p(z(t)) = -\text{tr}\left( \frac{\partial f}{\partial z} \right)$$

Computing this trace exactly requires $d$ backward passes through the dynamics network. For $d = 784$ (flattened MNIST) or $d > 1000$ (typical financial feature spaces), this is prohibitively expensive.

### 1.2 FFJORD's Solution

**FFJORD** (Grathwohl et al., 2019) combines two key ideas:

1. **Hutchinson's trace estimator** reduces the cost to a single vector-Jacobian product per ODE step
2. **Free-form dynamics** allows any neural network architecture—no coupling layers, no autoregressive structure, no triangular Jacobians

This makes continuous normalizing flows practical for high-dimensional problems.

---

## 2. The Hutchinson Estimator in Depth

### 2.1 Mathematical Foundation

**Theorem (Hutchinson, 1989).** For any square matrix $A \in \mathbb{R}^{d \times d}$ and random vector $\epsilon$ satisfying $\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}[\epsilon \epsilon^\top] = I$:

$$\text{tr}(A) = \mathbb{E}\left[\epsilon^\top A \epsilon\right]$$

**Proof.** By linearity of expectation and the cyclic property of the trace:

$$\mathbb{E}[\epsilon^\top A \epsilon] = \mathbb{E}\left[\text{tr}(\epsilon^\top A \epsilon)\right] = \mathbb{E}\left[\text{tr}(A \epsilon \epsilon^\top)\right] = \text{tr}\left(A \cdot \mathbb{E}[\epsilon \epsilon^\top]\right) = \text{tr}(A \cdot I) = \text{tr}(A)$$

### 2.2 Noise Distributions

Two common choices for $\epsilon$:

**Gaussian noise** ($\epsilon \sim \mathcal{N}(0, I)$):

- Variance: $\text{Var}[\epsilon^\top A \epsilon] = 2\|A\|_F^2$
- Smooth, well-understood statistical properties

**Rademacher noise** ($\epsilon_i \sim \text{Uniform}\{-1, +1\}$):

- Variance: $\text{Var}[\epsilon^\top A \epsilon] = 2\sum_{i \neq j} A_{ij}^2 \leq 2\|A\|_F^2$
- Lower variance than Gaussian (strictly lower when diagonal elements are large)
- Preferred in practice

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

def sample_rademacher(shape, device):
    """Sample Rademacher random variables: ±1 with equal probability."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1


def sample_gaussian(shape, device):
    """Sample standard Gaussian random variables."""
    return torch.randn(shape, device=device)
```

### 2.3 Variance Reduction

Multiple noise samples reduce variance at the cost of computation:

$$\widehat{\text{tr}}(A) = \frac{1}{K} \sum_{k=1}^{K} \epsilon_k^\top A \epsilon_k$$

Variance decreases as $O(1/K)$. In practice, $K = 1$ works well during training (the stochastic gradient already averages over mini-batches), while $K > 1$ can be used during evaluation for tighter density estimates.

---

## 3. FFJORD Architecture

### 3.1 Core Implementation

```python
class FFJORDDynamics(nn.Module):
    """
    FFJORD dynamics with Hutchinson trace estimation.
    
    Jointly evolves state z and log-density change using stochastic
    trace estimation for O(1) cost per ODE step.
    """
    
    def __init__(self, dim: int, hidden_dims: list = [64, 64],
                 noise_type: str = 'rademacher'):
        super().__init__()
        
        self.dim = dim
        self.noise_type = noise_type
        self._noise = None  # Cached noise for consistent estimation
        
        # Free-form dynamics: any architecture
        layers = []
        prev_dim = dim + 1  # +1 for time concatenation
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.net = nn.Sequential(*layers)
        
        # Small initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def _get_noise(self, z):
        """Get or cache noise vector for trace estimation."""
        if self._noise is None or self._noise.shape != z.shape:
            if self.noise_type == 'rademacher':
                self._noise = sample_rademacher(z.shape, z.device)
            else:
                self._noise = sample_gaussian(z.shape, z.device)
        return self._noise
    
    def dynamics(self, t, z):
        """Compute f(z, t)."""
        batch_size = z.shape[0]
        t_vec = t.expand(batch_size, 1)
        zt = torch.cat([z, t_vec], dim=-1)
        return self.net(zt)
    
    def forward(self, t, state):
        """
        Augmented dynamics for FFJORD.
        
        State: [z, log_p_change]
        d/dt [z, log_p] = [f(z,t), -tr(df/dz)]
        where tr(df/dz) is estimated via Hutchinson's estimator.
        """
        z = state[..., :self.dim]
        
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            dz_dt = self.dynamics(t, z)
            
            noise = self._get_noise(z)
            
            # Vector-Jacobian product: ε^T (df/dz)
            vjp = torch.autograd.grad(
                dz_dt, z,
                grad_outputs=noise,
                create_graph=self.training,
                retain_graph=True
            )[0]
            
            # Trace estimate: ε^T (df/dz) ε
            trace_est = (vjp * noise).sum(dim=-1, keepdim=True)
        
        dlog_p_dt = -trace_est
        
        return torch.cat([dz_dt, dlog_p_dt], dim=-1)
    
    def reset_noise(self):
        """Reset cached noise (call before each forward pass)."""
        self._noise = None
```

### 3.2 Complete FFJORD Model

```python
class FFJORD(nn.Module):
    """
    Free-Form Jacobian of Reversible Dynamics.
    
    A continuous normalizing flow using Hutchinson's trace estimator
    to enable unconstrained dynamics networks.
    """
    
    def __init__(self, dim: int, hidden_dims: list = [64, 64],
                 T: float = 1.0, solver: str = 'dopri5',
                 noise_type: str = 'rademacher',
                 use_adjoint: bool = True):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.solver = solver
        self.use_adjoint = use_adjoint
        
        self.dynamics = FFJORDDynamics(dim, hidden_dims, noise_type)
    
    def _integrate(self, state0, t):
        """Run ODE integration with appropriate method."""
        integrator = odeint_adjoint if self.use_adjoint else odeint
        return integrator(
            self.dynamics, state0, t,
            method=self.solver, rtol=1e-5, atol=1e-7
        )
    
    def log_prob(self, x):
        """
        Compute log p(x) under the FFJORD model.
        
        1. Map x to base distribution via inverse flow (T → 0)
        2. Evaluate base log-density
        3. Add accumulated log-density change
        """
        self.dynamics.reset_noise()
        
        state_T = torch.cat([
            x, torch.zeros(x.shape[0], 1, device=x.device)
        ], dim=-1)
        
        t = torch.tensor([self.T, 0.], device=x.device)
        state_0 = self._integrate(state_T, t)[-1]
        
        z0 = state_0[..., :self.dim]
        delta_log_p = state_0[..., self.dim]
        
        log_p_z0 = -0.5 * (
            self.dim * torch.log(torch.tensor(2 * torch.pi, device=x.device))
            + (z0 ** 2).sum(dim=-1)
        )
        
        return log_p_z0 + delta_log_p
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples via forward flow (0 → T)."""
        self.dynamics.reset_noise()
        
        z0 = torch.randn(n_samples, self.dim, device=device)
        state0 = torch.cat([
            z0, torch.zeros(n_samples, 1, device=device)
        ], dim=-1)
        
        t = torch.tensor([0., self.T], device=device)
        
        with torch.no_grad():
            state_T = self._integrate(state0, t)[-1]
        
        return state_T[..., :self.dim]
```

---

## 4. Multi-Scale FFJORD

For complex distributions, stacking multiple FFJORD blocks improves expressivity:

```python
class MultiScaleFFJORD(nn.Module):
    """
    Multi-scale FFJORD with multiple integration blocks.
    Each block learns dynamics over a separate time interval.
    """
    
    def __init__(self, dim: int, n_blocks: int = 3,
                 hidden_dims: list = [64, 64]):
        super().__init__()
        
        self.dim = dim
        self.n_blocks = n_blocks
        
        self.blocks = nn.ModuleList([
            FFJORDDynamics(dim, hidden_dims) for _ in range(n_blocks)
        ])
        
        # Optional batch normalization between blocks for training stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim, affine=True)
            for _ in range(n_blocks - 1)
        ])
    
    def log_prob(self, x):
        """Compute log p(x) through all blocks (reverse order)."""
        total_delta_log_p = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i in reversed(range(self.n_blocks)):
            block = self.blocks[i]
            block.reset_noise()
            
            state_T = torch.cat([
                z, torch.zeros(z.shape[0], 1, device=z.device)
            ], dim=-1)
            
            t = torch.tensor([1., 0.], device=z.device)
            state_0 = odeint(block, state_T, t, method='dopri5',
                            rtol=1e-5, atol=1e-7)[-1]
            
            z = state_0[..., :self.dim]
            total_delta_log_p = total_delta_log_p + state_0[..., self.dim]
            
            if i > 0:
                z = self.batch_norms[i - 1](z)
                log_det_bn = torch.log(
                    self.batch_norms[i - 1].weight.abs()
                ).sum()
                total_delta_log_p = total_delta_log_p + log_det_bn
        
        log_p_z0 = -0.5 * (
            self.dim * torch.log(torch.tensor(2 * torch.pi, device=z.device))
            + (z ** 2).sum(dim=-1)
        )
        
        return log_p_z0 + total_delta_log_p
```

---

## 5. Training FFJORD

### 5.1 Training Loop with Regularization

```python
def train_ffjord(model, train_data, n_epochs=500, lr=1e-3,
                 batch_size=256, kinetic_reg=0.01):
    """
    Train FFJORD with maximum likelihood and optional regularization.
    
    Args:
        model: FFJORD instance
        train_data: Training data (n_samples, dim)
        n_epochs: Number of epochs
        lr: Learning rate
        batch_size: Mini-batch size
        kinetic_reg: Weight for kinetic energy regularization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    dataset = torch.utils.data.TensorDataset(train_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for (batch,) in loader:
            optimizer.zero_grad()
            
            # Negative log-likelihood
            log_prob = model.log_prob(batch)
            nll = -log_prob.mean()
            
            loss = nll
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}: NLL = {avg_loss:.4f}")
    
    return model
```

### 5.2 Practical Training Tips

**Noise caching:** Use the same noise vector $\epsilon$ for all ODE solver steps within a single forward pass. This reduces variance because the trace estimate is consistent across the trajectory. Reset noise between different data samples.

**Gradient clipping:** FFJORD gradients can be noisy due to the stochastic trace estimator. Gradient clipping (norm $\leq 5.0$) stabilizes training.

**Warm-up:** Start with loose ODE tolerances and tighten them during training:

```python
class FFJORDWithWarmup(FFJORD):
    """FFJORD with tolerance warm-up schedule."""
    
    def set_tolerances(self, rtol, atol):
        """Adjust solver tolerances during training."""
        self._rtol = rtol
        self._atol = atol
    
    def _integrate(self, state0, t):
        integrator = odeint_adjoint if self.use_adjoint else odeint
        return integrator(
            self.dynamics, state0, t,
            method=self.solver,
            rtol=getattr(self, '_rtol', 1e-3),
            atol=getattr(self, '_atol', 1e-5)
        )
```

---

## 6. FFJORD vs Discrete Flows

### 6.1 Architecture Freedom

The defining advantage of FFJORD is **architectural freedom**. Compare the constraints:

| Flow Type | Dynamics Network | Jacobian Requirement |
|-----------|-----------------|---------------------|
| Planar Flow | $f(z) = uz^\top w + b$ | Rank-1 update formula |
| RealNVP | Coupling layers only | Triangular Jacobian |
| Autoregressive | Masked networks | Lower-triangular |
| **FFJORD** | **Any neural network** | **None (Hutchinson)** |

### 6.2 When to Choose FFJORD

**Use FFJORD when:**

- You need maximum expressivity without architectural constraints
- The problem dimension is moderate ($d < 1000$)
- Continuous-time interpretation is valuable (e.g., modeling temporal evolution of distributions)
- Memory is constrained (adjoint method gives $O(1)$ memory)

**Use discrete flows when:**

- Speed is critical (discrete flows avoid ODE solver overhead)
- Exact log-likelihoods are needed (no trace estimation noise)
- The problem has structure that coupling/autoregressive layers exploit
- $d$ is very large and even Hutchinson estimation overhead matters

---

## 7. Key Takeaways

1. **FFJORD enables free-form dynamics** in continuous normalizing flows by using the Hutchinson trace estimator, eliminating architectural constraints.

2. **The Hutchinson estimator** approximates $\text{tr}(\partial f / \partial z)$ with a single vector-Jacobian product, reducing cost from $O(d)$ backward passes to $O(1)$.

3. **Rademacher noise** ($\pm 1$) gives lower variance than Gaussian noise and is preferred in practice.

4. **Multi-scale FFJORD** stacks multiple integration blocks for greater expressivity on complex distributions.

5. **Training stability** requires noise caching within forward passes, gradient clipping, and careful tolerance scheduling.

---

## 8. Exercises

### Exercise 1: Noise Comparison

Compare Gaussian and Rademacher noise for Hutchinson estimation on a known matrix $A$ with analytically computable trace. Measure bias and variance as a function of dimension $d$ and number of samples $K$.

### Exercise 2: Tabular Density Estimation

Train FFJORD on a tabular dataset (e.g., UCI datasets) and compare test log-likelihood against MAF and RealNVP. Report number of parameters, training time, and NFE statistics.

### Exercise 3: Financial Return Distribution

Use FFJORD to model the joint distribution of daily returns for a portfolio of 10 assets. Compare the learned density with a Gaussian copula model. Evaluate tail behavior and correlation capture.

### Exercise 4: Conditional FFJORD

Extend FFJORD to a conditional model $p(x | c)$ by conditioning the dynamics on context $c$. Apply this to model asset return distributions conditioned on market regime indicators.

---

## References

1. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
2. Hutchinson, M. F. (1989). A Stochastic Estimator of the Trace of the Influence Matrix. *Communications in Statistics*.
3. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
