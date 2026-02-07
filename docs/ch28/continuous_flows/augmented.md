# Augmented Neural ODEs

## Learning Objectives

By the end of this section, you will:

- Understand the topological limitations of standard Neural ODEs
- Explain why homeomorphisms restrict the class of learnable transformations
- Implement augmented Neural ODEs that overcome these limitations
- Compare standard and augmented architectures on benchmark tasks
- Apply augmented formulations to density estimation and classification

## Prerequisites

- Neural ODE fundamentals and `torchdiffeq` (Section 27.1)
- Continuous normalizing flows (Section 27.2)
- Basic topology concepts (homeomorphisms, connected components)

---

## 1. The Topological Limitation

### 1.1 Neural ODEs Define Homeomorphisms

A standard Neural ODE evolves the state via:

$$\frac{dz}{dt} = f_\theta(z(t), t), \quad z(0) = z_0$$

By the Picard-Lindelöf theorem, if $f$ is Lipschitz continuous, the solution map $\phi_t: z(0) \mapsto z(t)$ is a **homeomorphism**—a continuous bijection with a continuous inverse (obtained by solving the ODE backwards).

This means the flow **preserves topology**: it cannot tear, create holes, or change the number of connected components.

### 1.2 What Cannot Be Learned

Consider trying to classify points from two concentric circles (annulus dataset). The inner circle maps to class 0 and the outer to class 1. A classifier based on a standard Neural ODE must continuously deform the plane to separate these classes.

The fundamental problem: **a homeomorphism of $\mathbb{R}^d$ cannot map a connected set to a disconnected one** (or vice versa). More specifically:

- Two nested circles in $\mathbb{R}^2$ cannot be separated by a continuous flow in $\mathbb{R}^2$
- Interleaved spirals require the flow to "cross" trajectories, which ODEs forbid (uniqueness theorem)
- Any transformation that changes the "knottedness" of the data manifold is impossible

!!! warning "Trajectory Non-Crossing"
    The uniqueness theorem for ODEs states that solution trajectories cannot cross in the state space. If $z_1(t_0) \neq z_2(t_0)$, then $z_1(t) \neq z_2(t)$ for all $t$. While this guarantees invertibility, it severely limits the class of learnable transformations in the original state dimension.

### 1.3 Formal Statement

**Theorem (Dupont et al., 2019).** Let $\phi: \mathbb{R}^d \to \mathbb{R}^d$ be the flow map of a Neural ODE. Then $\phi$ is a homeomorphism, and therefore:

1. $\phi$ preserves the number of connected components
2. $\phi$ cannot change the dimension of the data manifold
3. $\phi$ maps open sets to open sets
4. For $d = 1$: $\phi$ must be monotonically increasing or decreasing

---

## 2. The Augmented Solution

### 2.1 Core Idea

**Augmented Neural ODEs** (Dupont et al., 2019) lift the state to a higher-dimensional space before applying the ODE flow:

$$z_{\text{aug}} = [z, \mathbf{0}] \in \mathbb{R}^{d + p}$$

$$\frac{dz_{\text{aug}}}{dt} = f_\theta(z_{\text{aug}}(t), t), \quad z_{\text{aug}}(0) = [z_0, \mathbf{0}]$$

The extra $p$ dimensions give the flow "room to maneuver." In the augmented space, the trajectories can avoid crossing by moving through the extra dimensions.

**Analogy:** Consider untangling two interlinked rings. In 3D, they cannot be separated without cutting. But if you lift them into 4D, they can slide past each other freely. Augmentation provides these extra dimensions.

### 2.2 Why It Works

In $\mathbb{R}^{d+p}$ with $p \geq 1$:

- Trajectories that would cross in $\mathbb{R}^d$ can be routed through different paths in the augmented dimensions
- The flow can effectively "fold" the space using the extra dimensions
- The projection back to $\mathbb{R}^d$ after the flow can produce non-homeomorphic maps

The augmented flow is still a homeomorphism in $\mathbb{R}^{d+p}$, but its projection to $\mathbb{R}^d$ is not constrained to be a homeomorphism—it can create the topological changes needed for complex tasks.

---

## 3. Implementation

### 3.1 Augmented Neural ODE Block

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class AugmentedODEFunc(nn.Module):
    """
    Dynamics function for Augmented Neural ODE.
    
    Operates in the augmented space R^{d+p} where:
    - d is the original data dimension
    - p is the number of augmented dimensions
    """
    
    def __init__(self, data_dim: int, aug_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        total_dim = data_dim + aug_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, total_dim)
        )
        
        # Small initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, z_aug):
        """
        Compute dynamics in augmented space.
        
        Args:
            t: Current time
            z_aug: Augmented state (batch, data_dim + aug_dim)
            
        Returns:
            dz_aug/dt (batch, data_dim + aug_dim)
        """
        batch_size = z_aug.shape[0]
        t_vec = t.expand(batch_size, 1)
        zt = torch.cat([z_aug, t_vec], dim=-1)
        return self.net(zt)


class AugmentedNeuralODE(nn.Module):
    """
    Augmented Neural ODE that lifts data to higher dimensions
    before applying continuous dynamics.
    
    This overcomes the topological limitations of standard Neural ODEs
    by providing extra dimensions for the flow to route trajectories.
    """
    
    def __init__(self, data_dim: int, aug_dim: int = 1,
                 hidden_dim: int = 64, T: float = 1.0,
                 solver: str = 'dopri5', use_adjoint: bool = True):
        super().__init__()
        
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        self.T = T
        self.use_adjoint = use_adjoint
        self.solver = solver
        
        self.func = AugmentedODEFunc(data_dim, aug_dim, hidden_dim)
        self.register_buffer('t', torch.tensor([0., T]))
    
    def forward(self, z):
        """
        Forward pass: augment → integrate → extract.
        
        Args:
            z: Input data (batch, data_dim)
            
        Returns:
            z_out: Transformed data (batch, data_dim + aug_dim)
        """
        # Augment: pad with zeros
        z_aug = torch.cat([
            z,
            torch.zeros(z.shape[0], self.aug_dim, device=z.device)
        ], dim=-1)
        
        # Integrate
        integrator = odeint_adjoint if self.use_adjoint else odeint
        trajectory = integrator(
            self.func, z_aug, self.t,
            method=self.solver, rtol=1e-5, atol=1e-7
        )
        
        z_out = trajectory[-1]
        
        return z_out
    
    def forward_data_only(self, z):
        """Forward pass returning only the original data dimensions."""
        z_out = self.forward(z)
        return z_out[..., :self.data_dim]
```

### 3.2 Augmented Classifier

```python
class AugmentedNeuralODEClassifier(nn.Module):
    """
    Classifier using Augmented Neural ODE.
    
    Architecture:
        1. Optional encoder
        2. Augmented Neural ODE (lift + flow)
        3. Classification head
    """
    
    def __init__(self, input_dim: int, num_classes: int,
                 aug_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        
        self.aug_ode = AugmentedNeuralODE(
            data_dim=input_dim,
            aug_dim=aug_dim,
            hidden_dim=hidden_dim
        )
        
        # Classifier uses all dimensions (original + augmented)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + aug_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        z_out = self.aug_ode(x)
        return self.classifier(z_out)
```

---

## 4. Augmented Continuous Normalizing Flows

The augmented framework also applies to density estimation. Standard CNFs (Section 27.2) are limited by the homeomorphism constraint. Augmented CNFs can model more complex distributions.

### 4.1 Augmented CNF Architecture

```python
class AugmentedCNFDynamics(nn.Module):
    """
    Dynamics for Augmented CNF with Hutchinson trace estimation.
    
    State: [z_data, z_aug, log_p]
    """
    
    def __init__(self, data_dim: int, aug_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        self.total_dim = data_dim + aug_dim
        self._noise = None
        
        self.net = nn.Sequential(
            nn.Linear(self.total_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, state):
        """
        Augmented CNF dynamics with trace estimation.
        """
        z = state[..., :self.total_dim]
        
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            
            batch_size = z.shape[0]
            t_vec = t.expand(batch_size, 1)
            zt = torch.cat([z, t_vec], dim=-1)
            dz_dt = self.net(zt)
            
            # Hutchinson trace estimate
            if self._noise is None or self._noise.shape != z.shape:
                self._noise = torch.randint(
                    0, 2, z.shape, device=z.device
                ).float() * 2 - 1
            
            vjp = torch.autograd.grad(
                dz_dt, z, grad_outputs=self._noise,
                create_graph=self.training, retain_graph=True
            )[0]
            trace_est = (vjp * self._noise).sum(dim=-1, keepdim=True)
        
        dlog_p_dt = -trace_est
        
        return torch.cat([dz_dt, dlog_p_dt], dim=-1)
    
    def reset_noise(self):
        self._noise = None


class AugmentedCNF(nn.Module):
    """
    Augmented Continuous Normalizing Flow.
    
    Augments data with extra dimensions before applying CNF,
    enabling modeling of distributions with complex topology.
    """
    
    def __init__(self, data_dim: int, aug_dim: int = 2,
                 hidden_dim: int = 64, T: float = 1.0):
        super().__init__()
        
        self.data_dim = data_dim
        self.aug_dim = aug_dim
        self.total_dim = data_dim + aug_dim
        self.T = T
        
        self.dynamics = AugmentedCNFDynamics(data_dim, aug_dim, hidden_dim)
    
    def log_prob(self, x):
        """
        Compute log p(x) by marginalizing over augmented dimensions.
        
        p(x) = ∫ p(x, a) da ≈ p(x, a=0) under the augmentation scheme.
        """
        self.dynamics.reset_noise()
        
        # Augment data with zeros
        z = torch.cat([
            x,
            torch.zeros(x.shape[0], self.aug_dim, device=x.device)
        ], dim=-1)
        
        state_T = torch.cat([
            z, torch.zeros(x.shape[0], 1, device=x.device)
        ], dim=-1)
        
        t = torch.tensor([self.T, 0.], device=x.device)
        state_0 = odeint(self.dynamics, state_T, t,
                        method='dopri5', rtol=1e-5, atol=1e-7)[-1]
        
        z0 = state_0[..., :self.total_dim]
        delta_log_p = state_0[..., self.total_dim]
        
        # Base distribution in augmented space
        log_p_z0 = -0.5 * (
            self.total_dim * torch.log(
                torch.tensor(2 * torch.pi, device=x.device)
            )
            + (z0 ** 2).sum(dim=-1)
        )
        
        return log_p_z0 + delta_log_p
    
    def sample(self, n_samples, device='cpu'):
        """Sample from the learned distribution."""
        self.dynamics.reset_noise()
        
        z0 = torch.randn(n_samples, self.total_dim, device=device)
        
        state0 = torch.cat([
            z0, torch.zeros(n_samples, 1, device=device)
        ], dim=-1)
        
        t = torch.tensor([0., self.T], device=device)
        
        with torch.no_grad():
            state_T = odeint(self.dynamics, state0, t,
                            method='dopri5', rtol=1e-5, atol=1e-7)[-1]
        
        # Return only data dimensions
        return state_T[..., :self.data_dim]
```

---

## 5. Empirical Comparison

### 5.1 Concentric Circles Dataset

```python
def compare_standard_vs_augmented():
    """
    Compare standard and augmented Neural ODEs on the concentric
    circles dataset, which requires topological change.
    """
    from sklearn.datasets import make_circles
    
    # Generate data
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Standard Neural ODE classifier
    standard_model = nn.Sequential(
        # Standard Neural ODE in R^2
        NeuralODEBlock(dim=2, hidden_dim=64),
        nn.Linear(2, 2)
    )
    
    # Augmented Neural ODE classifier (augment to R^5)
    augmented_model = AugmentedNeuralODEClassifier(
        input_dim=2, num_classes=2, aug_dim=3, hidden_dim=64
    )
    
    # Train both models
    for model, name in [(standard_model, "Standard"), 
                         (augmented_model, "Augmented")]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(200):
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            pred = model(X).argmax(dim=1)
            acc = (pred == y).float().mean().item()
        
        print(f"{name} Neural ODE: Accuracy = {acc:.4f}")

compare_standard_vs_augmented()
```

### 5.2 Expected Results

| Model | Circles Accuracy | Spirals Accuracy | Parameters |
|-------|:---------------:|:---------------:|:----------:|
| Standard NODE (d=2) | ~0.50 | ~0.50 | ~8K |
| Augmented NODE (d=2, p=1) | ~0.95 | ~0.85 | ~12K |
| Augmented NODE (d=2, p=5) | ~0.99 | ~0.98 | ~20K |

The standard Neural ODE fails because it cannot separate concentric circles via a homeomorphism of $\mathbb{R}^2$. Even one extra dimension ($p = 1$) dramatically improves performance.

---

## 6. How Many Extra Dimensions?

### 6.1 Theoretical Guidance

The Whitney embedding theorem suggests that a $d$-dimensional manifold can be embedded in $\mathbb{R}^{2d+1}$. For practical purposes:

- **$p = 1$:** Often sufficient to break simple topological constraints
- **$p = d$:** Doubling the state dimension works well for most tasks
- **$p > d$:** Diminishing returns; extra computation without benefit

### 6.2 Practical Recommendations

```python
def choose_augmentation_dim(data_dim: int, task_complexity: str = 'medium'):
    """
    Heuristic for choosing augmentation dimension.
    
    Args:
        data_dim: Original data dimensionality
        task_complexity: 'low', 'medium', or 'high'
    """
    guidelines = {
        'low': max(1, data_dim // 4),    # Simple topological change
        'medium': max(2, data_dim // 2),  # Moderate complexity
        'high': data_dim,                  # Complex topology
    }
    return guidelines.get(task_complexity, data_dim // 2)
```

---

## 7. Key Takeaways

1. **Standard Neural ODEs define homeomorphisms** that preserve topology—they cannot separate concentric circles, untangle interleaved spirals, or perform other topologically nontrivial transformations.

2. **Augmented Neural ODEs** lift the state to $\mathbb{R}^{d+p}$ by padding with zeros, giving the ODE flow extra dimensions to route trajectories around each other.

3. **Even one extra dimension** ($p = 1$) can break the topological barrier. In practice, $p \approx d/2$ to $d$ works well.

4. **The augmented framework applies to both classification and density estimation**, enabling augmented CNFs that can model distributions with complex topological structure.

5. **The computational overhead is modest**: the dynamics network operates on $(d + p)$-dimensional inputs instead of $d$-dimensional, and the ODE solver cost scales with the total state dimension.

---

## 8. Exercises

### Exercise 1: Topological Barrier Demonstration

Create a 2D dataset with three nested rings (three classes). Show that a standard Neural ODE fails to classify it, while an augmented Neural ODE with $p \geq 2$ succeeds. Visualize the learned trajectories in 3D.

### Exercise 2: Augmentation Dimension Study

For the concentric circles dataset, train augmented Neural ODEs with $p = 1, 2, 4, 8, 16$ and plot accuracy vs. augmentation dimension. Identify the point of diminishing returns.

### Exercise 3: Augmented FFJORD

Implement an augmented FFJORD model and train it on a distribution with complex topology (e.g., a figure-eight shape in 2D). Compare the learned density against a standard (non-augmented) FFJORD.

### Exercise 4: Financial Application

Model the joint distribution of credit default events (which exhibit complex dependency structures including tail dependencies and asymmetric correlations) using an augmented CNF. Compare against a Gaussian copula baseline.

---

## References

1. Dupont, E., Doucet, A., & Teh, Y. W. (2019). Augmented Neural ODEs. *NeurIPS*.
2. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
3. Zhang, H., Gao, X., Unterman, J., & Arodz, T. (2020). Approximation Capabilities of Neural ODEs and Invertible Residual Networks. *ICML*.
4. Massaroli, S., Poli, M., Park, J., Yamashita, A., & Asama, H. (2020). Dissecting Neural ODEs. *NeurIPS*.
