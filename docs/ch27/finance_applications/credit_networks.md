# Energy-Based Credit Network Models

## Learning Objectives

After completing this section, you will be able to:

1. Model credit default dependencies using Boltzmann machine structure
2. Design energy functions that capture pairwise and higher-order default correlations
3. Implement credit risk estimation via Gibbs sampling from the energy model
4. Apply EBM-based anomaly detection for systemic risk monitoring

## Introduction

Credit risk modeling fundamentally involves understanding dependencies between default events. When one firm defaults, it affects its counterparties, suppliers, and competitors—creating cascading effects that are poorly captured by independent default models. The network structure of Boltzmann machines is a natural fit for this problem: binary units represent default/survival states of firms, and weighted connections encode the strength of default dependencies.

## Credit Networks as Boltzmann Machines

### Problem Setup

Consider a portfolio of $N$ firms. Each firm $i$ has a binary state:

$$s_i = \begin{cases} 1 & \text{if firm } i \text{ defaults} \\ 0 & \text{if firm } i \text{ survives} \end{cases}$$

The joint default distribution is modeled as a Boltzmann distribution:

$$P(\mathbf{s}) = \frac{1}{Z} \exp(-E(\mathbf{s}))$$

### Energy Function Design

The energy function encodes three types of information:

**Individual default propensity** (bias terms):
$$E_{\text{individual}}(\mathbf{s}) = -\sum_i \theta_i s_i$$

where $\theta_i$ encodes firm $i$'s standalone default probability. A more negative $\theta_i$ means higher default propensity.

**Pairwise default dependencies** (connection weights):
$$E_{\text{pairwise}}(\mathbf{s}) = -\sum_{i < j} w_{ij} s_i s_j$$

where $w_{ij} > 0$ means firms $i$ and $j$ tend to default together (positive correlation), and $w_{ij} < 0$ means defaults are substitutes (negative correlation).

**Sector-level factors** (hidden units):
$$E_{\text{sector}}(\mathbf{s}, \mathbf{h}) = -\sum_{i,k} W_{ik} s_i h_k - \sum_k b_k h_k$$

where hidden units $h_k$ represent latent sector or macro factors that drive correlated defaults.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class CreditNetworkEBM(nn.Module):
    """
    Boltzmann machine for credit default modeling.
    
    Models joint default distribution over a portfolio of firms
    with explicit pairwise dependencies and latent sector factors.
    
    Parameters
    ----------
    n_firms : int
        Number of firms (visible units)
    n_sectors : int
        Number of latent sector factors (hidden units)
    """
    
    def __init__(self, n_firms: int, n_sectors: int = 5):
        super().__init__()
        self.n_firms = n_firms
        self.n_sectors = n_sectors
        
        # Individual default propensities
        self.theta = nn.Parameter(torch.randn(n_firms) * 0.1 - 2.0)
        
        # Pairwise default dependencies
        W_vis = torch.randn(n_firms, n_firms) * 0.01
        W_vis = (W_vis + W_vis.t()) / 2
        W_vis.fill_diagonal_(0)
        self.W_visible = nn.Parameter(W_vis)
        
        # Firm-to-sector connections
        self.W_sector = nn.Parameter(torch.randn(n_firms, n_sectors) * 0.1)
        
        # Sector biases
        self.b_sector = nn.Parameter(torch.zeros(n_sectors))
    
    def energy(self, s: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        Compute energy of default configuration.
        
        Parameters
        ----------
        s : torch.Tensor
            Default states (batch, n_firms), values in {0, 1}
        h : torch.Tensor, optional
            Sector states (batch, n_sectors), values in {0, 1}
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        # Individual terms
        E = -torch.einsum('i,bi->b', self.theta, s)
        
        # Pairwise terms
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        W_sym.fill_diagonal_(0)
        E -= 0.5 * torch.einsum('bi,ij,bj->b', s, W_sym, s)
        
        # Sector terms
        if h is not None:
            E -= torch.einsum('bi,ij,bj->b', s, self.W_sector, h)
            E -= torch.einsum('j,bj->b', self.b_sector, h)
        
        return E
    
    def free_energy(self, s: torch.Tensor) -> torch.Tensor:
        """
        Free energy after marginalizing out hidden sector units.
        
        F(s) = -θᵀs - ½sᵀWs - Σ_k log(1 + exp(Ws_k + b_k))
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        # Visible terms
        E_vis = -torch.einsum('i,bi->b', self.theta, s)
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        W_sym.fill_diagonal_(0)
        E_vis -= 0.5 * torch.einsum('bi,ij,bj->b', s, W_sym, s)
        
        # Hidden terms (analytically marginalized)
        activation = s @ self.W_sector + self.b_sector
        E_hid = -torch.log(1 + torch.exp(activation)).sum(dim=1)
        
        return E_vis + E_hid
    
    def conditional_default_prob(self, firm_idx: int, 
                                 s: torch.Tensor) -> torch.Tensor:
        """
        P(s_i = 1 | s_{-i}) via mean-field with marginalized sectors.
        """
        # Field from other firms
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        field = self.theta[firm_idx] + (W_sym[firm_idx] * s).sum(dim=-1)
        
        # Field from sector units (mean-field approximation)
        sector_activation = s @ self.W_sector + self.b_sector
        sector_means = torch.sigmoid(sector_activation)
        field += (self.W_sector[firm_idx] * sector_means).sum(dim=-1)
        
        return torch.sigmoid(field)
    
    def gibbs_sample(self, n_samples: int = 1000, n_steps: int = 500,
                     initial_state: torch.Tensor = None) -> torch.Tensor:
        """
        Generate default scenarios via Gibbs sampling.
        """
        if initial_state is not None:
            s = initial_state.clone()
        else:
            # Initialize with independent defaults
            probs = torch.sigmoid(self.theta)
            s = torch.bernoulli(probs.unsqueeze(0).expand(n_samples, -1))
        
        for _ in range(n_steps):
            # Update each firm in random order
            order = torch.randperm(self.n_firms)
            for i in order:
                prob = self.conditional_default_prob(i, s)
                s[:, i] = torch.bernoulli(prob)
        
        return s
    
    def estimate_default_probabilities(self, n_samples: int = 10000) -> dict:
        """
        Estimate default probabilities and correlations from samples.
        """
        samples = self.gibbs_sample(n_samples=n_samples, n_steps=1000)
        
        # Marginal default probabilities
        pd = samples.mean(dim=0)
        
        # Default correlations
        corr = torch.corrcoef(samples.t())
        
        # Joint default probabilities (pairs)
        joint_pd = {}
        for i in range(self.n_firms):
            for j in range(i+1, self.n_firms):
                joint_pd[(i,j)] = (samples[:, i] * samples[:, j]).mean().item()
        
        # Expected number of defaults
        n_defaults = samples.sum(dim=1)
        
        return {
            'marginal_pd': pd.numpy(),
            'correlation': corr.numpy(),
            'joint_pd': joint_pd,
            'expected_defaults': n_defaults.mean().item(),
            'default_std': n_defaults.std().item(),
            'max_defaults_99': np.percentile(n_defaults.numpy(), 99)
        }


def credit_network_demo():
    """
    Demonstrate credit network modeling with EBMs.
    """
    # Create a small credit portfolio
    n_firms = 20
    n_sectors = 3
    
    model = CreditNetworkEBM(n_firms, n_sectors)
    
    # Set meaningful parameters
    with torch.no_grad():
        # Default propensities: most firms have low default prob
        model.theta.copy_(torch.tensor(
            [-3.0] * 5 +    # Low risk (PD ~ 5%)
            [-2.0] * 10 +   # Medium risk (PD ~ 12%)
            [-1.0] * 5      # High risk (PD ~ 27%)
        ))
        
        # Sector assignments (firms cluster into sectors)
        W_sector = torch.zeros(n_firms, n_sectors)
        W_sector[:7, 0] = 1.0    # Sector 1: firms 0-6
        W_sector[7:14, 1] = 1.0  # Sector 2: firms 7-13
        W_sector[14:, 2] = 1.0   # Sector 3: firms 14-19
        model.W_sector.copy_(W_sector * 0.5)
        
        # Some pairwise dependencies (supply chain links)
        W_vis = torch.zeros(n_firms, n_firms)
        W_vis[0, 7] = W_vis[7, 0] = 0.5   # Cross-sector link
        W_vis[3, 15] = W_vis[15, 3] = 0.3  # Cross-sector link
        model.W_visible.copy_(W_vis)
    
    # Estimate risk metrics
    print("Estimating default probabilities...")
    stats = model.estimate_default_probabilities(n_samples=20000)
    
    print(f"\nPortfolio Risk Metrics:")
    print(f"  Expected defaults: {stats['expected_defaults']:.2f}")
    print(f"  Default volatility: {stats['default_std']:.2f}")
    print(f"  99th percentile: {stats['max_defaults_99']:.0f}")
    
    print(f"\nMarginal default probabilities:")
    for i, pd in enumerate(stats['marginal_pd']):
        risk = "Low" if i < 5 else ("Medium" if i < 15 else "High")
        print(f"  Firm {i:2d} ({risk:6s}): {pd:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Marginal PDs
    colors = ['green']*5 + ['orange']*10 + ['red']*5
    axes[0].bar(range(n_firms), stats['marginal_pd'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Firm')
    axes[0].set_ylabel('Default Probability')
    axes[0].set_title('Marginal Default Probabilities')
    axes[0].grid(True, alpha=0.3)
    
    # Correlation matrix
    im = axes[1].imshow(stats['correlation'], cmap='RdBu_r', 
                        vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('Default Correlation Matrix')
    axes[1].set_xlabel('Firm')
    axes[1].set_ylabel('Firm')
    
    # Default count distribution
    samples = model.gibbs_sample(n_samples=10000, n_steps=500)
    n_defaults = samples.sum(dim=1).numpy()
    axes[2].hist(n_defaults, bins=range(int(n_defaults.max())+2), 
                density=True, alpha=0.7, edgecolor='black')
    axes[2].axvline(np.percentile(n_defaults, 99), color='red', 
                   linestyle='--', label='99th percentile')
    axes[2].set_xlabel('Number of Defaults')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Default Count Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

credit_network_demo()
```

## Systemic Risk Monitoring

### Energy as a Systemic Risk Indicator

The free energy of the observed market state provides a natural systemic risk indicator:

$$\text{Systemic Risk Score} = F(\mathbf{s}_{\text{current}}) - \mathbb{E}[F(\mathbf{s})]$$

A free energy significantly above the expected level signals that the current state of the credit system is unusual—potentially indicating elevated systemic risk.

```python
def systemic_risk_monitor(model, current_state, historical_states):
    """
    Monitor systemic risk using free energy anomaly.
    
    Parameters
    ----------
    model : CreditNetworkEBM
        Trained credit network
    current_state : torch.Tensor
        Current default/stress indicators
    historical_states : torch.Tensor
        Historical default patterns for calibration
    """
    with torch.no_grad():
        current_fe = model.free_energy(current_state).item()
        historical_fe = model.free_energy(historical_states).numpy()
    
    # Z-score relative to historical distribution
    z_score = (current_fe - historical_fe.mean()) / historical_fe.std()
    
    # Percentile
    percentile = (historical_fe < current_fe).mean() * 100
    
    return {
        'free_energy': current_fe,
        'z_score': z_score,
        'percentile': percentile,
        'alert': 'HIGH' if percentile > 95 else ('MEDIUM' if percentile > 80 else 'LOW')
    }
```

### Contagion Analysis

The energy function's pairwise weights reveal the channels through which defaults can propagate:

```python
def analyze_contagion(model, shocked_firm: int, n_samples: int = 5000):
    """
    Analyze default contagion from a specific firm's default.
    
    Compare default probabilities with and without the shock.
    """
    # Baseline: unconditional defaults
    baseline_samples = model.gibbs_sample(n_samples=n_samples)
    baseline_pd = baseline_samples.mean(dim=0)
    
    # Shocked: condition on firm defaulting
    shocked_samples = model.gibbs_sample(n_samples=n_samples)
    shocked_samples[:, shocked_firm] = 1.0
    # Re-equilibrate
    shocked_samples = model.gibbs_sample(
        n_samples=n_samples, n_steps=500,
        initial_state=shocked_samples
    )
    shocked_pd = shocked_samples.mean(dim=0)
    
    # Contagion effect
    contagion = shocked_pd - baseline_pd
    
    return {
        'baseline_pd': baseline_pd.numpy(),
        'shocked_pd': shocked_pd.numpy(),
        'contagion_effect': contagion.numpy()
    }
```

## Training from Historical Data

The credit network can be trained from historical default observations using contrastive divergence:

```python
def train_credit_network(model, default_history, n_epochs=100, lr=0.01):
    """
    Train credit network from historical default data.
    
    Parameters
    ----------
    model : CreditNetworkEBM
        Credit network model
    default_history : torch.Tensor
        Binary matrix (n_periods, n_firms) of observed defaults
    """
    n_periods = default_history.shape[0]
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_periods)
        epoch_loss = 0
        
        for i in range(0, n_periods, 32):
            batch = default_history[perm[i:i+32]]
            batch_size = batch.shape[0]
            
            # Positive phase: free energy on data
            pos_fe = model.free_energy(batch).mean()
            
            # Negative phase: free energy on model samples
            neg_samples = model.gibbs_sample(
                n_samples=batch_size, n_steps=50
            )
            neg_fe = model.free_energy(neg_samples).mean()
            
            # CD gradient
            loss = pos_fe - neg_fe
            
            # Manual gradient step (since we use custom sampling)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                        param.grad.zero_()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss = {epoch_loss:.4f}")
```

## Key Takeaways

!!! success "Core Concepts"
    1. Credit default networks map naturally to Boltzmann machine structure: binary default states as visible units, sector factors as hidden units
    2. Pairwise weights capture direct default dependencies; hidden units capture latent common factors
    3. Gibbs sampling generates correlated default scenarios respecting the learned dependency structure
    4. Free energy provides a natural systemic risk indicator—unusual market states have high free energy
    5. Contagion analysis is straightforward: condition on a shock and observe the propagation through the energy landscape

!!! warning "Limitations"
    - Boltzmann machine training on sparse default data is challenging—defaults are rare events
    - Binary default states are a simplification; real credit quality varies continuously
    - The model assumes equilibrium, but credit crises involve non-equilibrium dynamics
    - Calibration to historical data requires sufficient default observations across firms

## Exercises

1. **Sector structure**: Compare the default correlation matrix from a credit network with and without hidden sector units. How do hidden units improve the modeling of default clustering?

2. **Stress testing**: Implement a stress test that shocks an entire sector (set all firms in sector $k$ to default) and measures the cascade to other sectors.

3. **CDS pricing**: Use the credit network to estimate joint default probabilities for a basket CDS. Compare with the Gaussian copula model.

## References

- Dai Pra, P., & Tolotti, M. (2009). Heterogeneous credit portfolios and the dynamics of the aggregate losses. *Stochastic Processes and their Applications*.
- Giesecke, K., & Kim, B. (2011). Systemic Risk: What Defaults Are Telling Us. *Management Science*.
- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In *Parallel Distributed Processing*.
