# Energy-Based Portfolio Optimization

## Learning Objectives

After completing this section, you will be able to:

1. Frame portfolio optimization as energy minimization with constraints
2. Design energy functions that encode return objectives, risk penalties, and regulatory constraints
3. Implement energy-based portfolio optimization using Langevin dynamics
4. Compare energy-based approaches with traditional mean-variance optimization

## From Markowitz to Energy Functions

### Classical Portfolio Optimization

The Markowitz mean-variance framework seeks portfolios that maximize return for a given risk level:

$$\max_w \quad \mu^T w - \frac{\lambda}{2} w^T \Sigma w$$

subject to constraints like $\sum_i w_i = 1$ and $w_i \geq 0$. This is a quadratic program with a well-known closed-form solution for the unconstrained case.

### The Energy Formulation

We can recast portfolio optimization as energy minimization by defining an energy function over portfolio weights:

$$E(w) = -\mu^T w + \frac{\lambda}{2} w^T \Sigma w + \text{constraint penalties}$$

Low energy corresponds to portfolios with high expected return, controlled risk, and satisfied constraints. The Boltzmann distribution $p(w) \propto \exp(-E(w)/T)$ then assigns higher probability to better portfolios, with temperature $T$ controlling the concentration around the optimum.

### Why Energy-Based?

The energy formulation offers several advantages over direct optimization:

**Distributional output**: Instead of a single optimal portfolio, EBMs produce a distribution over "good" portfolios. At low temperature, this concentrates near the optimum; at higher temperature, it reveals the landscape of near-optimal alternatives.

**Flexible constraints**: Hard constraints, soft penalties, and complex objectives combine naturally as energy terms. Non-convex constraints that break classical optimizers are handled through MCMC sampling.

**Compositional structure**: Risk models, return models, and constraint modules can be trained and combined independently. Adding a new constraint is as simple as adding a new energy term.

**Regime-adaptive**: Neural energy functions can learn portfolio quality functions that adapt to market regimes, capturing nonlinear relationships that mean-variance misses.

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PortfolioEnergyFunction(nn.Module):
    """
    Energy function for portfolio optimization.
    
    E(w) = -return_term + risk_term + constraint_penalties
    """
    
    def __init__(self, n_assets: int, mu: torch.Tensor, 
                 sigma: torch.Tensor, risk_aversion: float = 1.0):
        super().__init__()
        self.n_assets = n_assets
        self.register_buffer('mu', mu)           # Expected returns
        self.register_buffer('sigma', sigma)      # Covariance matrix
        self.risk_aversion = risk_aversion
    
    def return_energy(self, w: torch.Tensor) -> torch.Tensor:
        """Negative expected return (lower is better)."""
        return -torch.mv(w, self.mu) if w.dim() == 2 else -w @ self.mu
    
    def risk_energy(self, w: torch.Tensor) -> torch.Tensor:
        """Portfolio variance penalty."""
        if w.dim() == 1:
            return 0.5 * self.risk_aversion * w @ self.sigma @ w
        return 0.5 * self.risk_aversion * torch.einsum(
            'bi,ij,bj->b', w, self.sigma, w
        )
    
    def budget_penalty(self, w: torch.Tensor, strength: float = 10.0) -> torch.Tensor:
        """Soft constraint: weights sum to 1."""
        deviation = (w.sum(dim=-1) - 1.0) ** 2
        return strength * deviation
    
    def long_only_penalty(self, w: torch.Tensor, strength: float = 5.0) -> torch.Tensor:
        """Soft constraint: no short positions."""
        violations = torch.clamp(-w, min=0) ** 2
        return strength * violations.sum(dim=-1)
    
    def concentration_penalty(self, w: torch.Tensor, max_weight: float = 0.3,
                              strength: float = 5.0) -> torch.Tensor:
        """Soft constraint: maximum position size."""
        violations = torch.clamp(w - max_weight, min=0) ** 2
        return strength * violations.sum(dim=-1)
    
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Total energy."""
        E = self.return_energy(w) + self.risk_energy(w)
        E = E + self.budget_penalty(w)
        E = E + self.long_only_penalty(w)
        E = E + self.concentration_penalty(w)
        return E


def optimize_portfolio_langevin(energy_fn, n_assets, n_samples=100,
                                 n_steps=1000, step_size=0.001,
                                 noise_scale=0.001, temperature=0.1):
    """
    Find optimal portfolio via Langevin dynamics on the energy landscape.
    
    Parameters
    ----------
    energy_fn : PortfolioEnergyFunction
        Portfolio energy function
    n_assets : int
        Number of assets
    n_samples : int
        Number of parallel chains
    n_steps : int
        Langevin steps
    step_size : float
        Gradient step size
    noise_scale : float
        Noise magnitude
    temperature : float
        Controls exploration vs exploitation
    """
    # Initialize with equal-weight portfolio + noise
    w = torch.ones(n_samples, n_assets) / n_assets
    w = w + 0.05 * torch.randn_like(w)
    
    energy_history = []
    
    for step in range(n_steps):
        w.requires_grad_(True)
        energy = energy_fn(w).sum()
        grad = torch.autograd.grad(energy, w)[0]
        
        noise = torch.randn_like(w) * noise_scale * np.sqrt(temperature)
        w = w.detach() - step_size * grad + noise
        
        # Project to simplex (approximately)
        w = torch.clamp(w, min=0)
        w = w / w.sum(dim=1, keepdim=True)
        
        if step % 100 == 0:
            with torch.no_grad():
                avg_energy = energy_fn(w).mean().item()
                energy_history.append(avg_energy)
    
    return w.detach(), energy_history


def portfolio_optimization_demo():
    """
    Demonstrate energy-based portfolio optimization.
    """
    # Market parameters
    n_assets = 10
    torch.manual_seed(42)
    
    # Expected returns (annualized)
    mu = torch.tensor([0.08, 0.12, 0.10, 0.15, 0.07,
                       0.09, 0.11, 0.14, 0.06, 0.13])
    
    # Correlation structure
    L = torch.randn(n_assets, n_assets) * 0.3
    sigma = L @ L.t() / n_assets
    # Add diagonal for stability
    sigma = sigma + 0.02 * torch.eye(n_assets)
    # Scale to realistic vol levels (15-30%)
    vols = torch.tensor([0.15, 0.25, 0.20, 0.30, 0.12,
                        0.18, 0.22, 0.28, 0.10, 0.24])
    D = torch.diag(vols)
    sigma = D @ torch.corrcoef(torch.randn(n_assets, 100)) @ D
    sigma = (sigma + sigma.t()) / 2 + 0.01 * torch.eye(n_assets)
    
    # Energy-based optimization
    energy_fn = PortfolioEnergyFunction(
        n_assets, mu, sigma, risk_aversion=2.0
    )
    
    # Run at different temperatures
    temperatures = [0.01, 0.1, 1.0]
    results = {}
    
    for T in temperatures:
        weights, history = optimize_portfolio_langevin(
            energy_fn, n_assets, n_samples=200,
            n_steps=2000, step_size=0.001,
            noise_scale=0.001, temperature=T
        )
        results[T] = weights
        
        # Report
        avg_w = weights.mean(dim=0)
        ret = (avg_w @ mu).item()
        risk = torch.sqrt(avg_w @ sigma @ avg_w).item()
        
        print(f"\nT={T}: Return={ret:.4f}, Risk={risk:.4f}, "
              f"Sharpe={ret/risk:.4f}")
        print(f"  Weights: {avg_w.numpy().round(3)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    asset_names = [f'Asset {i+1}' for i in range(n_assets)]
    
    for idx, T in enumerate(temperatures):
        avg_w = results[T].mean(dim=0).numpy()
        std_w = results[T].std(dim=0).numpy()
        
        axes[idx].bar(range(n_assets), avg_w, yerr=std_w, 
                     capsize=3, alpha=0.7)
        axes[idx].set_xlabel('Asset')
        axes[idx].set_ylabel('Weight')
        axes[idx].set_title(f'T = {T}')
        axes[idx].set_xticks(range(n_assets))
        axes[idx].set_xticklabels(range(1, n_assets+1))
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Portfolio Weights at Different Temperatures', fontsize=13)
    plt.tight_layout()
    plt.show()

portfolio_optimization_demo()
```

## Neural Energy Functions for Regime-Adaptive Portfolios

Beyond the analytical energy function above, we can learn energy functions from historical data:

```python
class LearnedPortfolioEnergy(nn.Module):
    """
    Neural energy function that learns portfolio quality
    conditioned on market features.
    
    E(w, z) where w = portfolio weights, z = market features
    """
    
    def __init__(self, n_assets: int, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets + n_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        w : torch.Tensor
            Portfolio weights (batch, n_assets)
        z : torch.Tensor
            Market features (batch, n_features)
        """
        combined = torch.cat([w, z], dim=-1)
        return self.net(combined).squeeze(-1)
```

The learned energy function can capture nonlinear relationships between market regimes and portfolio quality that are invisible to mean-variance optimization—for example, that momentum strategies have low energy (are desirable) in trending markets but high energy (are risky) in mean-reverting regimes.

## Key Takeaways

!!! success "Core Concepts"
    1. Portfolio optimization maps naturally to energy minimization: $E(w) = -\text{return} + \lambda \cdot \text{risk} + \text{penalties}$
    2. Langevin dynamics finds good portfolios while exploring the landscape of near-optimal alternatives
    3. Temperature controls the trade-off between finding the single best portfolio ($T \to 0$) and exploring diverse good portfolios ($T > 0$)
    4. Compositional energy terms enable modular constraint handling
    5. Neural energy functions can learn regime-adaptive portfolio quality measures from historical data

!!! warning "Practical Considerations"
    - Energy-based optimization is slower than quadratic programming for standard mean-variance problems
    - The approach shines when constraints are complex, non-convex, or regime-dependent
    - Constraint penalties must be tuned carefully—too weak and constraints are violated, too strong and optimization focuses on constraints rather than objectives
    - Temperature selection is critical: too low causes poor exploration, too high produces overly diverse (suboptimal) portfolios

## Exercises

1. **Efficient frontier**: Use energy-based optimization to trace the efficient frontier by varying the risk aversion parameter $\lambda$. Compare with the analytical solution.

2. **Transaction costs**: Add a transaction cost energy term $\|w - w_{\text{current}}\|_1$ and observe how the optimal portfolio changes. How does the energy landscape differ from the frictionless case?

3. **Regime conditioning**: Train a neural portfolio energy function on historical data from different market regimes (bull, bear, crisis). Test whether the learned energy function produces different optimal portfolios in each regime.

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
- Kolm, P. N., Tütüncü, R., & Fabozzi, F. J. (2014). 60 Years of portfolio optimization. *European Journal of Operational Research*.
- Du, Y., & Mordatch, I. (2019). Implicit Generation and Modeling with Energy Based Models. *NeurIPS*.
