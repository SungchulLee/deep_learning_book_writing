# Effective Sample Size

The effective sample size (ESS) measures the information content of a set of weighted or correlated samples. It is a fundamental diagnostic that applies across Monte Carlo methods — from importance sampling to MCMC. This section develops ESS for independent weighted samples (as in importance sampling); for MCMC ESS based on autocorrelation, see [Diagnostics](../mcmc/diagnostics.md).

---

## The Problem: Not All Samples Are Equal

In importance sampling, we draw $N$ samples from a proposal $q(x)$ and assign weights $w_i = p(x_i)/q(x_i)$. If the weights are highly unequal — a few samples dominate — then the estimator is effectively using far fewer than $N$ samples.

In MCMC, successive samples are correlated. $N$ correlated samples contain less information than $N$ independent samples.

ESS quantifies these inefficiencies with a single number.

---

## ESS for Importance Sampling

### Definition

Given normalised importance weights $\bar{w}_i = w_i / \sum_j w_j$, the effective sample size is:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^N \bar{w}_i^2}
$$

### Properties

- **Range**: $1 \leq \text{ESS} \leq N$
- **Maximum** ($\text{ESS} = N$): All weights are equal ($\bar{w}_i = 1/N$), meaning $q = p$
- **Minimum** ($\text{ESS} = 1$): One weight is 1 and the rest are 0 (complete weight collapse)

### Derivation

The ESS is defined as the number of independent samples from $p$ that would give the same variance as the weighted estimator:

$$
\text{Var}\left[\hat{I}_{\text{IS}}\right] \approx \frac{\text{Var}_p[f(X)]}{\text{ESS}}
$$

For the self-normalized estimator, this leads to:

$$
\text{ESS} \approx \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} = \frac{1}{\sum_i \bar{w}_i^2}
$$

### Interpretation

| ESS / N | Quality | Interpretation |
|---------|---------|----------------|
| > 0.5 | Excellent | Proposal closely matches target |
| 0.1 - 0.5 | Good | Acceptable for most applications |
| 0.01 - 0.1 | Poor | Results may be unreliable |
| < 0.01 | Very poor | Effective weight collapse |

### PyTorch Implementation

```python
import torch

def importance_sampling_ess(log_weights: torch.Tensor) -> float:
    """
    Compute ESS from unnormalized log-weights.
    
    Args:
        log_weights: Tensor of shape (N,) containing log w_i
    
    Returns:
        ESS as a float
    """
    # Normalize in log space for numerical stability
    log_w_norm = log_weights - torch.logsumexp(log_weights, dim=0)
    
    # ESS = 1 / sum(w_bar^2) = exp(-logsumexp(2 * log_w_norm))
    ess = torch.exp(-torch.logsumexp(2 * log_w_norm, dim=0))
    
    return ess.item()


# Example
N = 1000
# Case 1: Good proposal (weights roughly equal)
log_w_good = torch.randn(N) * 0.5
print(f"Good proposal ESS: {importance_sampling_ess(log_w_good):.0f} / {N}")

# Case 2: Poor proposal (few large weights)
log_w_bad = torch.randn(N) * 5.0
print(f"Poor proposal ESS: {importance_sampling_ess(log_w_bad):.0f} / {N}")
```

---

## ESS for MCMC

### Definition

For correlated MCMC samples, ESS accounts for autocorrelation:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{K} \rho_k}
$$

where $\rho_k$ is the autocorrelation at lag $k$ and the sum is truncated when autocorrelations become noisy.

### Interpretation

- **High correlation**: $\text{ESS} \ll N$ (many samples needed per effective independent draw)
- **Low correlation**: $\text{ESS} \approx N$ (efficient mixing)
- **Independent samples**: $\text{ESS} = N$ (ideal, impossible in MCMC)

### Connection to Monte Carlo Error

The variance of MCMC estimators scales as:

$$
\text{Var}\left[\frac{1}{N}\sum_{t=1}^N f(X^{(t)})\right] = \frac{\sigma_f^2}{N_{\text{eff}}}
$$

This means ESS directly controls the precision of posterior estimates.

---

## Monitoring and Improving ESS

### For Importance Sampling

- **Choose better proposals**: Match the shape of the target
- **Use adaptive methods**: Update proposal based on previous iterations
- **Reduce dimension**: Lower-dimensional problems have better ESS

### For MCMC

- **Tune step size**: Optimal acceptance rates improve ESS
- **Use HMC/NUTS**: Lower autocorrelation than random walk
- **Reparameterize**: Non-centered parameterizations reduce correlations
- **Run longer**: ESS grows linearly with chain length

### ESS per Second

The most useful efficiency metric combines ESS with computational cost:

$$
\text{Efficiency} = \frac{\text{ESS}}{\text{wall-clock time}}
$$

A method with lower ESS per iteration but faster iterations may be more efficient overall.

---

## Summary

| Context | ESS Formula | Range |
|---------|-------------|-------|
| **Importance Sampling** | $1 / \sum \bar{w}_i^2$ | $[1, N]$ |
| **MCMC** | $N / (1 + 2\sum \rho_k)$ | $[1, N]$ |
| **Minimum acceptable** | — | > 100 (point estimates), > 400 (intervals) |

---

## References

1. Kong, A. (1992). A note on importance sampling using standardized weights. *Technical Report*, University of Chicago.
2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. Chapter 2.
3. Vehtari, A., et al. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
