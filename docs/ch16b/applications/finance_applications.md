# Markov Chains in Quantitative Finance

## Introduction

Markov chains have widespread applications in quantitative finance, from credit risk modeling to regime-switching market dynamics. This section explores key financial applications with practical PyTorch implementations.

## Credit Rating Transitions

### Overview

Credit rating agencies (Moody's, S&P, Fitch) assign ratings to bonds. Ratings transition over time, and this process is well-modeled as a Markov chain:

- **States**: Credit ratings (AAA, AA, A, BBB, BB, B, CCC, D)
- **Transitions**: Rating changes over one period (typically one year)
- **Absorbing State**: Default (D)

### Mathematical Framework

The **rating transition matrix** $P$ where:
$$P_{ij} = P(\text{Rating}_{t+1} = j \mid \text{Rating}_t = i)$$

Key quantities:
- **Cumulative default probability**: $P(\text{Default by time } T \mid \text{Rating}_0 = i)$
- **Expected time to default**: $E[T_D \mid \text{Rating}_0 = i]$
- **Credit migration risk**: Risk of rating downgrades

### PyTorch Implementation

```python
import torch
import torch.linalg as LA
from typing import Dict, List, Tuple

class CreditRatingModel:
    """
    Credit rating transition model using Markov chains.
    
    Models rating migrations and computes default probabilities.
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        ratings: List[str]
    ):
        """
        Initialize credit rating model.
        
        Args:
            transition_matrix: Annual rating transition matrix
            ratings: List of rating names (last should be 'D' for default)
        """
        self.P = transition_matrix.clone().double()
        self.ratings = ratings
        self.n_ratings = len(ratings)
        
        # Identify default state (assumed to be last)
        self.default_idx = self.n_ratings - 1
        
        # Validate that default is absorbing
        if not torch.isclose(self.P[self.default_idx, self.default_idx], 
                            torch.tensor(1.0, dtype=self.P.dtype)):
            raise ValueError("Default state must be absorbing (P[D,D] = 1)")
    
    def transition_matrix(self, horizon: int) -> torch.Tensor:
        """
        Get transition matrix for given time horizon.
        
        P(t) = P^t (matrix power)
        
        Args:
            horizon: Number of periods
            
        Returns:
            Multi-period transition matrix
        """
        return torch.linalg.matrix_power(self.P, horizon)
    
    def cumulative_default_prob(
        self,
        initial_rating: str,
        max_horizon: int = 10
    ) -> torch.Tensor:
        """
        Compute cumulative default probabilities.
        
        P(Default by time t | Rating_0 = initial_rating)
        
        Args:
            initial_rating: Starting rating
            max_horizon: Maximum time horizon
            
        Returns:
            Tensor of cumulative default probs for t=1,...,max_horizon
        """
        rating_idx = self.ratings.index(initial_rating)
        
        cum_probs = torch.zeros(max_horizon)
        
        for t in range(1, max_horizon + 1):
            P_t = self.transition_matrix(t)
            cum_probs[t-1] = P_t[rating_idx, self.default_idx]
        
        return cum_probs
    
    def marginal_default_prob(
        self,
        initial_rating: str,
        max_horizon: int = 10
    ) -> torch.Tensor:
        """
        Compute marginal (period-by-period) default probabilities.
        
        P(Default in year t | Survive to year t-1, Rating_0 = initial_rating)
        
        Args:
            initial_rating: Starting rating
            max_horizon: Maximum time horizon
            
        Returns:
            Tensor of marginal default probs
        """
        cum_probs = self.cumulative_default_prob(initial_rating, max_horizon)
        
        marginal = torch.zeros(max_horizon)
        marginal[0] = cum_probs[0]
        
        for t in range(1, max_horizon):
            # P(default in t | survive to t-1) = (cum[t] - cum[t-1]) / (1 - cum[t-1])
            survival_prob = 1 - cum_probs[t-1]
            if survival_prob > 1e-10:
                marginal[t] = (cum_probs[t] - cum_probs[t-1]) / survival_prob
        
        return marginal
    
    def expected_rating_distribution(
        self,
        initial_distribution: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Compute expected rating distribution at future time.
        
        Args:
            initial_distribution: Initial distribution over ratings
            horizon: Time horizon
            
        Returns:
            Expected distribution at horizon
        """
        P_t = self.transition_matrix(horizon)
        return initial_distribution @ P_t
    
    def credit_var(
        self,
        portfolio: Dict[str, float],
        horizon: int,
        lgd: float = 0.6,
        n_simulations: int = 10000
    ) -> Dict:
        """
        Compute Credit Value at Risk for a portfolio.
        
        Args:
            portfolio: Dict of rating -> exposure
            horizon: Time horizon
            lgd: Loss Given Default
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR statistics
        """
        losses = []
        
        for _ in range(n_simulations):
            total_loss = 0
            
            for rating, exposure in portfolio.items():
                rating_idx = self.ratings.index(rating)
                
                # Simulate rating evolution
                current_rating = rating_idx
                for t in range(horizon):
                    probs = self.P[current_rating].float()
                    current_rating = torch.multinomial(probs, num_samples=1).item()
                    
                    if current_rating == self.default_idx:
                        total_loss += exposure * lgd
                        break
            
            losses.append(total_loss)
        
        losses = torch.tensor(losses)
        
        return {
            'mean_loss': losses.mean().item(),
            'std_loss': losses.std().item(),
            'var_95': torch.quantile(losses, 0.95).item(),
            'var_99': torch.quantile(losses, 0.99).item(),
            'cvar_95': losses[losses >= torch.quantile(losses, 0.95)].mean().item()
        }


def demonstrate_credit_model():
    """
    Demonstrate credit rating transition model.
    """
    print("Credit Rating Transition Model")
    print("=" * 70)
    
    # Typical annual transition matrix (simplified)
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
    
    P = torch.tensor([
        [0.91, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],  # AAA
        [0.01, 0.90, 0.08, 0.01, 0.00, 0.00, 0.00, 0.00],  # AA
        [0.00, 0.02, 0.91, 0.05, 0.01, 0.01, 0.00, 0.00],  # A
        [0.00, 0.00, 0.04, 0.89, 0.05, 0.01, 0.01, 0.00],  # BBB
        [0.00, 0.00, 0.00, 0.06, 0.83, 0.08, 0.02, 0.01],  # BB
        [0.00, 0.00, 0.00, 0.00, 0.06, 0.82, 0.08, 0.04],  # B
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.06, 0.65, 0.28],  # CCC
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # D (absorbing)
    ])
    
    model = CreditRatingModel(P, ratings)
    
    # Display 1-year transition matrix
    print("\n1-Year Transition Matrix:")
    print("-" * 60)
    header = "       " + "  ".join(f"{r:>6}" for r in ratings)
    print(header)
    for i, rating in enumerate(ratings):
        row = f"{rating:6} " + "  ".join(f"{P[i,j]:6.2%}" for j in range(8))
        print(row)
    
    # Cumulative default probabilities
    print("\n" + "-" * 60)
    print("Cumulative Default Probabilities:")
    print("-" * 60)
    
    test_ratings = ['AAA', 'BBB', 'B', 'CCC']
    header = "Rating  " + "  ".join(f"Year {t}" for t in range(1, 6))
    print(header)
    
    for rating in test_ratings:
        cum_pds = model.cumulative_default_prob(rating, 5)
        row = f"{rating:6}  " + "  ".join(f"{pd:6.2%}" for pd in cum_pds)
        print(row)
    
    # 5-year transition matrix
    print("\n" + "-" * 60)
    print("5-Year Transition Matrix (Investment Grade):")
    P_5 = model.transition_matrix(5)
    
    print("       " + "  ".join(f"{r:>6}" for r in ratings[:5]))
    for i in range(4):  # Investment grade only
        row = f"{ratings[i]:6} " + "  ".join(f"{P_5[i,j]:6.2%}" for j in range(5))
        print(row)
    
    # Portfolio Credit VaR
    print("\n" + "-" * 60)
    print("Portfolio Credit VaR (1-year, LGD=60%):")
    
    portfolio = {
        'AAA': 10_000_000,
        'AA': 25_000_000,
        'A': 35_000_000,
        'BBB': 20_000_000,
        'BB': 8_000_000,
        'B': 2_000_000
    }
    
    total = sum(portfolio.values())
    print(f"\nPortfolio Total: ${total:,.0f}")
    for rating, exposure in portfolio.items():
        print(f"  {rating}: ${exposure:,.0f} ({exposure/total:.1%})")
    
    var_results = model.credit_var(portfolio, horizon=1, lgd=0.6, n_simulations=10000)
    
    print(f"\nCredit Risk Metrics:")
    print(f"  Expected Loss: ${var_results['mean_loss']:,.0f}")
    print(f"  Loss Std Dev:  ${var_results['std_loss']:,.0f}")
    print(f"  VaR (95%):     ${var_results['var_95']:,.0f}")
    print(f"  VaR (99%):     ${var_results['var_99']:,.0f}")
    print(f"  CVaR (95%):    ${var_results['cvar_95']:,.0f}")


demonstrate_credit_model()
```

## Regime-Switching Models

### Overview

Financial markets often exhibit distinct "regimes" characterized by different volatility levels, correlations, or trend behaviors. Regime-switching models use Markov chains to capture transitions between these market states.

### Mathematical Framework

A regime-switching model consists of:
1. **Hidden regime**: $S_t \in \{1, 2, \ldots, K\}$ following a Markov chain
2. **Observable process**: Conditional on regime, e.g., $r_t | S_t = k \sim N(\mu_k, \sigma_k^2)$

### Implementation

```python
class RegimeSwitchingModel:
    """
    Regime-switching model for asset returns.
    
    Returns are generated from regime-specific distributions,
    with regime transitions following a Markov chain.
    """
    
    def __init__(
        self,
        transition_matrix: torch.Tensor,
        regime_means: torch.Tensor,
        regime_stds: torch.Tensor,
        regime_names: List[str] = None
    ):
        """
        Initialize regime-switching model.
        
        Args:
            transition_matrix: Regime transition matrix
            regime_means: Mean return for each regime
            regime_stds: Return volatility for each regime
            regime_names: Names for regimes
        """
        self.P = transition_matrix.clone()
        self.n_regimes = self.P.shape[0]
        self.means = regime_means.clone()
        self.stds = regime_stds.clone()
        
        if regime_names is None:
            self.regime_names = [f"Regime_{i}" for i in range(self.n_regimes)]
        else:
            self.regime_names = regime_names
        
        # Compute stationary distribution
        self._compute_stationary()
    
    def _compute_stationary(self):
        """Compute stationary regime distribution."""
        eigenvalues, eigenvectors = LA.eig(self.P.T)
        idx = torch.argmin(torch.abs(eigenvalues.real - 1.0))
        pi = eigenvectors[:, idx].real
        self.stationary = torch.abs(pi) / torch.abs(pi).sum()
    
    def simulate(
        self,
        n_periods: int,
        initial_regime: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate returns and regimes.
        
        Args:
            n_periods: Number of periods
            initial_regime: Starting regime (random if None)
            
        Returns:
            (returns, regimes) tensors
        """
        if initial_regime is None:
            initial_regime = torch.multinomial(self.stationary.float(), 1).item()
        
        regimes = torch.zeros(n_periods, dtype=torch.long)
        returns = torch.zeros(n_periods)
        
        current_regime = initial_regime
        
        for t in range(n_periods):
            regimes[t] = current_regime
            
            # Generate return from regime-specific distribution
            returns[t] = torch.normal(
                self.means[current_regime],
                self.stds[current_regime]
            )
            
            # Transition to next regime
            probs = self.P[current_regime].float()
            current_regime = torch.multinomial(probs, 1).item()
        
        return returns, regimes
    
    def unconditional_moments(self) -> Dict:
        """
        Compute unconditional mean and variance of returns.
        
        E[r] = Σ_k π_k μ_k
        Var[r] = Σ_k π_k (σ_k² + μ_k²) - (E[r])²
        """
        # Unconditional mean
        mean = (self.stationary * self.means).sum()
        
        # Unconditional variance
        second_moment = (self.stationary * (self.stds**2 + self.means**2)).sum()
        var = second_moment - mean**2
        
        return {
            'mean': mean.item(),
            'variance': var.item(),
            'std': var.sqrt().item()
        }
    
    def regime_duration(self) -> Dict[str, float]:
        """
        Compute expected duration of each regime.
        
        E[duration of regime k] = 1 / (1 - P[k,k])
        """
        durations = {}
        for k in range(self.n_regimes):
            stay_prob = self.P[k, k].item()
            expected_duration = 1 / (1 - stay_prob) if stay_prob < 1 else float('inf')
            durations[self.regime_names[k]] = expected_duration
        
        return durations


def demonstrate_regime_switching():
    """
    Demonstrate regime-switching model.
    """
    print("\n" + "=" * 70)
    print("Regime-Switching Model for Market Returns")
    print("=" * 70)
    
    # Two-regime model: Bull and Bear markets
    regime_names = ['Bull', 'Bear']
    
    # Transition matrix: regimes are persistent
    P = torch.tensor([
        [0.95, 0.05],  # Bull: 95% stay, 5% switch to Bear
        [0.10, 0.90]   # Bear: 10% switch to Bull, 90% stay
    ])
    
    # Regime-specific parameters (daily returns)
    means = torch.tensor([0.0005, -0.0003])  # Bull positive, Bear negative
    stds = torch.tensor([0.01, 0.025])        # Bear more volatile
    
    model = RegimeSwitchingModel(P, means, stds, regime_names)
    
    print("\nRegime Parameters:")
    print("-" * 40)
    for k in range(2):
        print(f"{regime_names[k]}: μ = {means[k]*252*100:.2f}% ann, "
              f"σ = {stds[k]*np.sqrt(252)*100:.1f}% ann")
    
    print(f"\nStationary distribution:")
    print(f"  P(Bull) = {model.stationary[0]:.3f}")
    print(f"  P(Bear) = {model.stationary[1]:.3f}")
    
    durations = model.regime_duration()
    print(f"\nExpected regime durations:")
    for regime, duration in durations.items():
        print(f"  {regime}: {duration:.1f} days")
    
    moments = model.unconditional_moments()
    print(f"\nUnconditional moments:")
    print(f"  E[r] = {moments['mean']*252*100:.2f}% annualized")
    print(f"  σ[r] = {moments['std']*np.sqrt(252)*100:.1f}% annualized")
    
    # Simulate
    returns, regimes = model.simulate(n_periods=1000)
    
    print(f"\nSimulation (1000 days):")
    print(f"  Empirical mean:  {returns.mean()*252*100:.2f}% ann")
    print(f"  Empirical std:   {returns.std()*np.sqrt(252)*100:.1f}% ann")
    print(f"  Days in Bull:    {(regimes == 0).sum().item()}")
    print(f"  Days in Bear:    {(regimes == 1).sum().item()}")


import numpy as np
demonstrate_regime_switching()
```

## Summary

| Application | State Space | Key Outputs |
|-------------|-------------|-------------|
| **Credit Ratings** | Rating grades (AAA to D) | Default probabilities, migration risk |
| **Regime Switching** | Market regimes (Bull/Bear) | Return distributions, regime durations |
| **Interest Rates** | Rate levels | Term structure, rate dynamics |
| **Market Making** | Inventory states | Optimal quotes, inventory risk |

## References

1. Jarrow, R.A., Lando, D., & Turnbull, S.M. "A Markov Model for the Term Structure of Credit Risk Spreads." *Review of Financial Studies*, 1997.
2. Hamilton, J.D. "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 1989.
3. Lando, D. *Credit Risk Modeling*. Princeton University Press, 2004.
4. Duffie, D. & Singleton, K.J. *Credit Risk*. Princeton University Press, 2003.
