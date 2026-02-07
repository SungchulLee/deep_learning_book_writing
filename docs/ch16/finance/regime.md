# Regime Detection and Bayesian Strategy Evaluation

## Overview

This section covers two critical applications of Bayesian methods in quantitative finance: detecting market regime changes through online Bayesian updating, and evaluating trading strategies using Bayesian A/B testing. Both applications leverage the sequential updating property of Bayes' theorem and provide direct probability statements about quantities of interest.

---

## Bayesian Regime Detection

### Market Regimes

Financial markets exhibit distinct behavioral regimes — bull/bear markets, high/low volatility periods, trending/mean-reverting dynamics. Bayesian methods provide a principled framework for:

1. **Online regime inference**: Updating regime beliefs as new data arrives
2. **Regime probability estimation**: Direct posterior probability of being in each regime
3. **Regime-conditional forecasting**: Predictions that account for regime uncertainty

### Simple Two-Regime Model

Consider a model with two regimes (e.g., low and high volatility):

$$
r_t \mid z_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2), \quad k \in \{1, 2\}
$$

with transition probabilities:

$$
P(z_t = j \mid z_{t-1} = i) = A_{ij}
$$

### Bayesian Filtering

At each time step, the **filtered probability** of being in regime $k$ is updated:

$$
P(z_t = k \mid r_{1:t}) \propto p(r_t \mid z_t = k) \sum_j A_{jk} \, P(z_{t-1} = j \mid r_{1:t-1})
$$

This is the forward pass of the Hidden Markov Model filter (see [Ch18: Hidden Markov Models](../../ch18/markov_chains/hmm.md)).

### Implementation

```python
import torch


class BayesianRegimeDetector:
    """
    Online Bayesian regime detection for market data.
    
    Two-regime model with Gaussian emissions and Markov transitions.
    """
    
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, 
                 transition_matrix: torch.Tensor):
        """
        Parameters
        ----------
        mu : (2,) regime means
        sigma : (2,) regime standard deviations
        transition_matrix : (2, 2) transition probabilities
        """
        self.mu = mu
        self.sigma = sigma
        self.A = transition_matrix
        self.filtered_prob = torch.tensor([0.5, 0.5])  # initial belief
    
    def update(self, r_t: float) -> torch.Tensor:
        """Update regime probabilities given new return observation."""
        # Prediction step
        predicted = self.A.T @ self.filtered_prob
        
        # Likelihood under each regime
        likelihood = torch.exp(
            -0.5 * ((r_t - self.mu) / self.sigma) ** 2
        ) / self.sigma
        
        # Update step
        unnormalized = likelihood * predicted
        self.filtered_prob = unnormalized / unnormalized.sum()
        
        return self.filtered_prob.clone()
    
    def run_filter(self, returns: torch.Tensor) -> torch.Tensor:
        """Run filter over entire return series."""
        T = len(returns)
        probs = torch.zeros(T, 2)
        
        for t in range(T):
            probs[t] = self.update(returns[t].item())
        
        return probs
```

---

## Bayesian A/B Testing for Strategy Evaluation

Bayesian A/B testing provides a natural framework for comparing trading strategies, offering direct probability statements about strategy superiority and enabling principled early stopping.

### Problem Setup

Compare two strategies (or strategy vs benchmark):

- **Strategy A** (e.g., existing strategy): Returns $r^A_1, \ldots, r^A_{n_A}$
- **Strategy B** (e.g., new strategy): Returns $r^B_1, \ldots, r^B_{n_B}$

**Question**: What is the probability that Strategy B is better than Strategy A?

### Bayesian Model

Assume returns are Gaussian (or use posterior over Sharpe ratios):

$$
r^A_i \sim \mathcal{N}(\mu_A, \sigma_A^2), \quad r^B_i \sim \mathcal{N}(\mu_B, \sigma_B^2)
$$

With conjugate Normal-Inverse-Gamma priors, the posterior for each strategy's mean return is a $t$-distribution.

### Computing $P(\mu_B > \mu_A \mid \text{data})$

Using posterior samples:

$$
P(\mu_B > \mu_A \mid \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \mathbf{1}[\mu_B^{(s)} > \mu_A^{(s)}]
$$

where $\mu_A^{(s)}, \mu_B^{(s)}$ are drawn from their respective posterior distributions.

### Implementation

```python
import torch
from scipy import stats
import numpy as np


class BayesianABTest:
    """
    Bayesian A/B testing for strategy comparison.
    
    Uses conjugate Normal-Inverse-Gamma model with
    posterior sampling for probability of superiority.
    """
    
    def __init__(self, prior_mean: float = 0.0, prior_var: float = 100.0,
                 prior_shape: float = 1.0, prior_scale: float = 1.0):
        self.mu_0 = prior_mean
        self.kappa_0 = 1.0 / prior_var  # prior precision on mean
        self.alpha_0 = prior_shape
        self.beta_0 = prior_scale
    
    def posterior_params(self, data: np.ndarray) -> dict:
        """Compute Normal-Inverse-Gamma posterior parameters."""
        n = len(data)
        x_bar = data.mean()
        s2 = data.var(ddof=1) if n > 1 else 1.0
        
        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / kappa_n
        alpha_n = self.alpha_0 + n / 2.0
        beta_n = (self.beta_0 + 0.5 * (n - 1) * s2 + 
                  0.5 * self.kappa_0 * n * (x_bar - self.mu_0)**2 / kappa_n)
        
        return {
            'mu_n': mu_n, 'kappa_n': kappa_n,
            'alpha_n': alpha_n, 'beta_n': beta_n
        }
    
    def sample_posterior_mean(self, params: dict, n_samples: int = 10000):
        """Sample posterior mean returns."""
        # sigma^2 ~ Inverse-Gamma(alpha_n, beta_n)
        sigma2_samples = stats.invgamma(
            a=params['alpha_n'], scale=params['beta_n']
        ).rvs(n_samples)
        
        # mu | sigma^2 ~ Normal(mu_n, sigma^2 / kappa_n)
        mu_samples = stats.norm(
            loc=params['mu_n'],
            scale=np.sqrt(sigma2_samples / params['kappa_n'])
        ).rvs()
        
        return mu_samples
    
    def compare(self, returns_a: np.ndarray, returns_b: np.ndarray,
                n_samples: int = 50000) -> dict:
        """
        Compare two strategies.
        
        Returns
        -------
        dict with:
            prob_b_better: P(mu_B > mu_A | data)
            expected_difference: E[mu_B - mu_A | data]
            credible_interval: 95% CI for mu_B - mu_A
        """
        params_a = self.posterior_params(returns_a)
        params_b = self.posterior_params(returns_b)
        
        samples_a = self.sample_posterior_mean(params_a, n_samples)
        samples_b = self.sample_posterior_mean(params_b, n_samples)
        
        diff = samples_b - samples_a
        
        return {
            'prob_b_better': float((diff > 0).mean()),
            'expected_difference': float(diff.mean()),
            'credible_interval': (
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5))
            ),
            'prob_practically_better': float(
                (diff > 0.001).mean()  # >10bps monthly
            )
        }
```

### Advantages over Frequentist Testing

| Aspect | Bayesian A/B Test | Frequentist t-test |
|--------|-------------------|-------------------|
| **Output** | $P(\text{B better} \mid \text{data})$ | p-value (often misinterpreted) |
| **Early stopping** | Principled (monitor posterior) | Inflates false positive rate |
| **Prior information** | Formally incorporated | Ignored |
| **Decision framework** | Direct probability for decision | Reject/fail-to-reject dichotomy |
| **Sample size** | Flexible (continuous monitoring) | Fixed (pre-determined) |

---

## Bayesian Online Updating for Market Signals

Bayesian sequential updating enables real-time signal processing:

```python
class BayesianSignalTracker:
    """Track a time-varying signal using sequential Bayesian updating."""
    
    def __init__(self, prior_mean=0.0, prior_var=1.0, obs_var=1.0, 
                 decay=0.99):
        self.mu = prior_mean
        self.var = prior_var
        self.obs_var = obs_var
        self.decay = decay
    
    def update(self, observation: float) -> tuple:
        """
        Update signal estimate with new observation.
        
        Uses exponential decay to allow for time-varying signals.
        """
        # Inflate prior variance (forget factor)
        predicted_var = self.var / self.decay
        
        # Kalman-style update
        kalman_gain = predicted_var / (predicted_var + self.obs_var)
        self.mu = self.mu + kalman_gain * (observation - self.mu)
        self.var = (1 - kalman_gain) * predicted_var
        
        return self.mu, self.var
```

---

## Summary

| Application | Bayesian Tool | Key Benefit |
|-------------|---------------|-------------|
| **Regime detection** | HMM filtering | Real-time regime probabilities |
| **Strategy comparison** | A/B testing | Direct probability of superiority |
| **Signal tracking** | Sequential updating | Adaptive estimation with uncertainty |
| **Early stopping** | Posterior monitoring | Principled stopping without p-hacking |

---

## References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. *Journal of Experimental Psychology: General*, 142(2), 573.
