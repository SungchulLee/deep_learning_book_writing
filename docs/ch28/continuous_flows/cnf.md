# Continuous Normalizing Flows

## Learning Objectives

By the end of this section, you will:

- Understand how normalizing flows generalize from discrete to continuous transformations
- Derive the instantaneous change of variables formula
- Implement continuous normalizing flows (CNFs) in PyTorch
- Train CNFs for density estimation tasks
- Compare CNFs with discrete normalizing flows

## Prerequisites

- Neural ODE fundamentals and `torchdiffeq` (Section 27.1)
- Adjoint sensitivity method (Section 27.1)
- Normalizing flows and change of variables formula (Chapter 25)
- Probability distributions and likelihood-based training

---

## 1. From Discrete to Continuous Flows

### 1.1 Discrete Normalizing Flows Recap

A discrete normalizing flow transforms a simple base distribution $z_0 \sim p_0(z_0)$ through a sequence of invertible transformations:

$$z_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)$$

The density of $z_K$ is computed via the **change of variables formula**:

$$\log p_K(z_K) = \log p_0(z_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|$$

**Limitation:** Each transformation $f_k$ must be invertible with a tractable Jacobian determinant. This imposes severe architectural constraints—planar flows, radial flows, and coupling layers all sacrifice expressivity for tractability.

### 1.2 The Continuous Perspective

Recall from Section 27.1 that a ResNet block $z_{k+1} = z_k + f_\theta(z_k)$ is an Euler step of an ODE. Similarly, a chain of normalizing flow transformations can be viewed as discretizing a continuous transformation:

$$\frac{dz}{dt} = f_\theta(z(t), t), \quad z(0) = z_0$$

As the number of discrete steps $K \to \infty$ and step size $\to 0$, the flow becomes continuous. The key question is: what happens to the change of variables formula in the continuous limit?

### 1.3 Why Go Continuous?

| Aspect | Discrete Flows | Continuous Normalizing Flows |
|--------|---------------|------------------------------|
| **Architecture** | Constrained (coupling, autoregressive) | Free-form (any neural network) |
| **Jacobian** | Full determinant required | Only trace required |
| **Invertibility** | Must be designed in | Automatic (reverse ODE) |
| **Depth** | Fixed $K$ layers | Adaptive (solver decides) |
| **Memory** | $O(K)$ | $O(1)$ via adjoint |

---

## 2. The Instantaneous Change of Variables

### 2.1 Derivation

Consider a continuous transformation $z(t)$ governed by the ODE $\frac{dz}{dt} = f(z(t), t)$. At each instant, the transformation is infinitesimally close to the identity:

$$z(t + \epsilon) = z(t) + \epsilon \cdot f(z(t), t) + O(\epsilon^2)$$

The Jacobian of this infinitesimal step is:

$$\frac{\partial z(t+\epsilon)}{\partial z(t)} = I + \epsilon \frac{\partial f}{\partial z} + O(\epsilon^2)$$

Taking the log-determinant:

$$\log \left| \det \left( I + \epsilon \frac{\partial f}{\partial z} \right) \right| = \epsilon \cdot \text{tr}\left( \frac{\partial f}{\partial z} \right) + O(\epsilon^2)$$

where we used the identity $\log \det(I + \epsilon A) = \epsilon \cdot \text{tr}(A) + O(\epsilon^2)$.

Taking the limit $\epsilon \to 0$ gives the **instantaneous change of variables formula**:

$$\frac{d}{dt} \log p(z(t)) = -\text{tr}\left( \frac{\partial f}{\partial z} \right)$$

### 2.2 The Key Result

Integrating from $t = 0$ to $t = T$:

$$\boxed{\log p(z(T)) = \log p(z(0)) - \int_0^T \text{tr}\left( \frac{\partial f}{\partial z}(z(t), t) \right) dt}$$

This is remarkable: instead of computing the full $d \times d$ Jacobian determinant (cost $O(d^3)$), we only need the **trace** of the Jacobian (cost $O(d^2)$ or better with estimators).

### 2.3 Comparison with Discrete Formula

| | Discrete Flow | Continuous Flow |
|--|---------------|-----------------|
| **Density change** | $\sum_k \log |\det J_k|$ | $\int_0^T \text{tr}\left(\frac{\partial f}{\partial z}\right) dt$ |
| **Cost per step** | $O(d^3)$ for general $J$ | $O(d^2)$ for trace |
| **Architectural constraint** | Invertible with tractable det | Any Lipschitz network |
| **Accumulation** | Summation | Integration |

---

## 3. Computing the Trace

### 3.1 Exact Trace Computation

For a dynamics function $f: \mathbb{R}^d \to \mathbb{R}^d$, the trace of the Jacobian is:

$$\text{tr}\left( \frac{\partial f}{\partial z} \right) = \sum_{i=1}^{d} \frac{\partial f_i}{\partial z_i}$$

This requires $d$ backward passes through the network (one per diagonal element):

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

def exact_trace(f_output, z):
    """
    Compute exact trace of Jacobian df/dz.
    
    Cost: d backward passes (one per dimension).
    
    Args:
        f_output: Output of dynamics function f(z), shape (batch, d)
        z: Input to dynamics function, shape (batch, d)
        
    Returns:
        trace: (batch,)
    """
    d = z.shape[-1]
    trace = 0.0
    
    for i in range(d):
        # Compute df_i/dz_i (diagonal of Jacobian)
        grad_i = torch.autograd.grad(
            f_output[:, i].sum(), z,
            create_graph=True, retain_graph=True
        )[0]
        trace = trace + grad_i[:, i]
    
    return trace
```

This is $O(d)$ backward passes—expensive for high-dimensional states.

### 3.2 Hutchinson's Trace Estimator

The key to making CNFs scalable is **Hutchinson's trace estimator**:

$$\text{tr}(A) = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\epsilon^\top A \epsilon\right]$$

where $\epsilon$ is a random vector with $\mathbb{E}[\epsilon] = 0$ and $\mathbb{E}[\epsilon \epsilon^\top] = I$.

Common choices for $\epsilon$:

- **Standard normal:** $\epsilon \sim \mathcal{N}(0, I)$
- **Rademacher:** $\epsilon_i \sim \text{Uniform}\{-1, +1\}$ (lower variance in practice)

This requires only **one** vector-Jacobian product, computed via a single backward pass:

```python
def hutchinson_trace(f_output, z, noise=None):
    """
    Estimate trace of Jacobian using Hutchinson's estimator.
    
    Cost: 1 backward pass regardless of dimension d.
    
    tr(df/dz) = E[ε^T (df/dz) ε]
    
    Args:
        f_output: Output of dynamics function, shape (batch, d)
        z: Input, shape (batch, d)
        noise: Random vector for estimation (batch, d)
        
    Returns:
        trace_estimate: (batch,)
    """
    if noise is None:
        # Rademacher noise: lower variance than Gaussian
        noise = torch.randint(0, 2, z.shape, device=z.device).float() * 2 - 1
    
    # Vector-Jacobian product: ε^T (df/dz)
    vjp = torch.autograd.grad(
        f_output, z,
        grad_outputs=noise,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Dot product with ε gives trace estimate
    trace_estimate = (vjp * noise).sum(dim=-1)
    
    return trace_estimate
```

**Complexity comparison:**

| Method | Backward passes | Cost |
|--------|----------------|------|
| Exact determinant | $O(d^3)$ | Prohibitive for $d > 100$ |
| Exact trace | $d$ | Expensive for large $d$ |
| Hutchinson estimator | $1$ | Constant regardless of $d$ |

---

## 4. CNF Architecture

### 4.1 Augmented ODE for Density Tracking

A CNF jointly evolves the state $z(t)$ and its log-density by solving an augmented ODE:

$$\frac{d}{dt} \begin{bmatrix} z(t) \\ \log p(z(t)) \end{bmatrix} = \begin{bmatrix} f_\theta(z(t), t) \\ -\text{tr}\left(\frac{\partial f_\theta}{\partial z}\right) \end{bmatrix}$$

```python
class CNFDynamics(nn.Module):
    """
    Continuous Normalizing Flow dynamics.
    
    Jointly evolves state z and log-probability using the
    instantaneous change of variables formula.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, 
                 use_hutchinson: bool = True):
        super().__init__()
        
        self.dim = dim
        self.use_hutchinson = use_hutchinson
        
        # Dynamics network: any architecture works!
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize with small weights for near-identity initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def dynamics(self, t, z):
        """Compute f(z, t)."""
        batch_size = z.shape[0]
        t_vec = t.expand(batch_size, 1)
        zt = torch.cat([z, t_vec], dim=-1)
        return self.net(zt)
    
    def forward(self, t, state):
        """
        Augmented dynamics: evolve [z, log_p] jointly.
        
        Args:
            t: Current time
            state: Concatenated [z, log_p_z], shape (batch, dim + 1)
            
        Returns:
            d/dt [z, log_p_z], shape (batch, dim + 1)
        """
        z = state[..., :self.dim]
        # log_p is state[..., self.dim:] but we don't need it for dynamics
        
        # Enable gradient computation for trace
        z = z.requires_grad_(True)
        
        # Compute dynamics
        dz_dt = self.dynamics(t, z)
        
        # Compute trace of Jacobian
        if self.use_hutchinson:
            noise = torch.randint(0, 2, z.shape, device=z.device).float() * 2 - 1
            trace = hutchinson_trace(dz_dt, z, noise)
        else:
            trace = exact_trace(dz_dt, z)
        
        # Log-density change: d/dt log p = -tr(df/dz)
        dlog_p_dt = -trace.unsqueeze(-1)
        
        return torch.cat([dz_dt, dlog_p_dt], dim=-1)
```

### 4.2 Complete CNF Model

```python
class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow for density estimation.
    
    Transforms samples from a base distribution (standard normal)
    to match a target distribution by learning continuous dynamics.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64,
                 T: float = 1.0, solver: str = 'dopri5',
                 use_hutchinson: bool = True):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.solver = solver
        
        self.dynamics = CNFDynamics(dim, hidden_dim, use_hutchinson)
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_var', torch.ones(dim))
    
    def base_log_prob(self, z):
        """Log probability under base distribution (standard normal)."""
        return -0.5 * (
            self.dim * torch.log(torch.tensor(2 * torch.pi)) 
            + (z ** 2).sum(dim=-1)
        )
    
    def forward_flow(self, z0):
        """
        Transform from base distribution to data distribution.
        z0 ~ N(0, I) → x
        
        Args:
            z0: Samples from base distribution (batch, dim)
            
        Returns:
            x: Transformed samples (batch, dim)
            delta_log_p: Change in log-density (batch,)
        """
        # Initialize augmented state: [z, log_p_change=0]
        state0 = torch.cat([
            z0, 
            torch.zeros(z0.shape[0], 1, device=z0.device)
        ], dim=-1)
        
        t = torch.tensor([0., self.T], device=z0.device)
        
        # Solve augmented ODE
        state_T = odeint(
            self.dynamics, state0, t,
            method=self.solver, rtol=1e-5, atol=1e-7
        )[-1]
        
        x = state_T[..., :self.dim]
        delta_log_p = state_T[..., self.dim]
        
        return x, delta_log_p
    
    def inverse_flow(self, x):
        """
        Transform from data distribution to base distribution.
        x → z0 ~ N(0, I)
        
        This is the INVERSE of forward_flow, computed by solving
        the ODE backwards (from T to 0).
        
        Args:
            x: Data samples (batch, dim)
            
        Returns:
            z0: Base distribution samples (batch, dim)
            delta_log_p: Change in log-density (batch,)
        """
        state_T = torch.cat([
            x,
            torch.zeros(x.shape[0], 1, device=x.device)
        ], dim=-1)
        
        # Solve ODE backwards: T → 0
        t = torch.tensor([self.T, 0.], device=x.device)
        
        state_0 = odeint(
            self.dynamics, state_T, t,
            method=self.solver, rtol=1e-5, atol=1e-7
        )[-1]
        
        z0 = state_0[..., :self.dim]
        delta_log_p = state_0[..., self.dim]
        
        return z0, delta_log_p
    
    def log_prob(self, x):
        """
        Compute log p(x) via the change of variables formula.
        
        log p(x) = log p(z0) + delta_log_p
        
        where delta_log_p accounts for the volume change
        during the continuous transformation.
        """
        z0, delta_log_p = self.inverse_flow(x)
        log_p_z0 = self.base_log_prob(z0)
        
        # log p(x) = log p(z0) + accumulated log-density change
        return log_p_z0 + delta_log_p
    
    def sample(self, n_samples):
        """Generate samples from the learned distribution."""
        z0 = torch.randn(n_samples, self.dim, device=self.base_mean.device)
        x, _ = self.forward_flow(z0)
        return x
```

---

## 5. Training CNFs

### 5.1 Maximum Likelihood Training

```python
def train_cnf(model, data, n_epochs=500, lr=1e-3, batch_size=128):
    """
    Train CNF via maximum likelihood (minimize negative log-likelihood).
    
    Args:
        model: ContinuousNormalizingFlow instance
        data: Training data tensor (n_samples, dim)
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        for (batch,) in loader:
            optimizer.zero_grad()
            
            # Compute negative log-likelihood
            log_prob = model.log_prob(batch)
            loss = -log_prob.mean()  # Minimize NLL
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}: NLL = {avg_loss:.4f}")
    
    return model
```

### 5.2 Example: Learning a 2D Distribution

```python
def demo_cnf_2d():
    """Demonstrate CNF on a 2D mixture of Gaussians."""
    
    # Generate target distribution: two moons
    from sklearn.datasets import make_moons
    data_np, _ = make_moons(n_samples=2000, noise=0.1)
    data = torch.tensor(data_np, dtype=torch.float32)
    
    # Create and train model
    model = ContinuousNormalizingFlow(
        dim=2, hidden_dim=64, T=1.0,
        use_hutchinson=True
    )
    
    train_cnf(model, data, n_epochs=300, lr=1e-3)
    
    # Generate samples
    with torch.no_grad():
        samples = model.sample(1000).numpy()
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
    axes[1].set_title('CNF Samples')
    
    # Plot learned density
    xx, yy = torch.meshgrid(
        torch.linspace(-2, 3, 100), torch.linspace(-1.5, 2, 100), indexing='ij'
    )
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    with torch.no_grad():
        log_probs = model.log_prob(grid)
    
    axes[2].contourf(xx.numpy(), yy.numpy(), 
                     log_probs.reshape(100, 100).exp().numpy(), levels=50)
    axes[2].set_title('Learned Density')
    
    plt.tight_layout()
    plt.savefig('cnf_demo.png', dpi=150)
    plt.show()
    
    return model

demo_cnf_2d()
```

---

## 6. Visualizing Continuous Flows

One advantage of CNFs is the ability to visualize how samples continuously deform from the base distribution to the target:

```python
def visualize_flow_trajectory(model, n_samples=500, n_steps=20):
    """
    Visualize the continuous transformation from base to data distribution.
    """
    import matplotlib.pyplot as plt
    
    z0 = torch.randn(n_samples, model.dim)
    
    # Get trajectory at multiple time points
    t_eval = torch.linspace(0, model.T, n_steps)
    
    state0 = torch.cat([
        z0, torch.zeros(n_samples, 1)
    ], dim=-1)
    
    with torch.no_grad():
        trajectory = odeint(
            model.dynamics, state0, t_eval,
            method='dopri5', rtol=1e-5, atol=1e-7
        )
    
    # Plot snapshots
    n_show = min(6, n_steps)
    indices = torch.linspace(0, n_steps - 1, n_show).long()
    
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    
    for i, idx in enumerate(indices):
        z = trajectory[idx, :, :model.dim].numpy()
        t = t_eval[idx].item()
        
        axes[i].scatter(z[:, 0], z[:, 1], s=1, alpha=0.5)
        axes[i].set_title(f't = {t:.2f}')
        axes[i].set_xlim(-4, 4)
        axes[i].set_ylim(-4, 4)
        axes[i].set_aspect('equal')
    
    plt.suptitle('Continuous Flow: Base → Data', fontsize=14)
    plt.tight_layout()
    plt.savefig('flow_trajectory.png', dpi=150)
    plt.show()
```

---

## 7. Key Takeaways

1. **Continuous normalizing flows replace discrete invertible layers with a continuous ODE**, eliminating architectural constraints on the dynamics function.

2. **The instantaneous change of variables formula** $\frac{d}{dt} \log p(z(t)) = -\text{tr}\left(\frac{\partial f}{\partial z}\right)$ replaces expensive log-determinant computations with a Jacobian trace.

3. **Exact trace computation costs $O(d)$ backward passes**, but this can be reduced to $O(1)$ using Hutchinson's stochastic estimator—the basis of FFJORD (Section 27.2).

4. **CNFs are automatically invertible**: the forward flow (generation) is the forward ODE, and the inverse flow (density evaluation) solves the same ODE backwards.

5. **Training uses maximum likelihood**: minimize the negative log-probability under the model, with gradients computed via the adjoint method for $O(1)$ memory.

---

## 8. Exercises

### Exercise 1: Exact vs Estimated Trace

Compare exact trace computation with Hutchinson estimation (both Gaussian and Rademacher noise) on a CNF with $d = 2, 10, 50, 100$. Measure the bias, variance, and wall-clock time of each approach.

### Exercise 2: Flow Visualization

Train a CNF to transform a standard normal into each of the following 2D distributions: (a) a ring, (b) a checkerboard pattern, (c) a Swiss roll. Visualize the continuous deformation at 10 time snapshots. Which distributions are harder to learn?

### Exercise 3: Density Estimation Benchmark

Compare a CNF against a discrete normalizing flow (e.g., RealNVP or MAF) on a standard density estimation benchmark. Report test log-likelihood, number of parameters, and training time.

---

## References

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*. (Section 4: Continuous Normalizing Flows)
2. Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Hutchinson, M. F. (1989). A Stochastic Estimator of the Trace of the Influence Matrix. *Communications in Statistics - Simulation and Computation*.
4. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
