# Monte Carlo Dropout Sample Convergence

## Overview

The quality of MC Dropout uncertainty estimates depends critically on the number of forward passes $T$. This document provides rigorous convergence analysis, practical bounds, and guidelines for selecting the number of samples.

## Theoretical Convergence Framework

### Monte Carlo Estimation

MC Dropout approximates the predictive distribution through sampling:

$$
\mathbb{E}_{q_\theta(\omega)}[f(\mathbf{x}; \omega)] \approx \hat{\mu}_T = \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)
$$

where $\hat{\omega}_t \sim q_\theta(\omega)$ are i.i.d. samples from the dropout distribution.

### Unbiasedness

The MC estimator is unbiased:

$$
\mathbb{E}[\hat{\mu}_T] = \mathbb{E}\left[\frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \hat{\omega}_t)\right] = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}[f(\mathbf{x}; \hat{\omega}_t)] = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]
$$

### Variance of the Mean Estimator

Let $\sigma^2_f = \text{Var}_{q_\theta}[f(\mathbf{x}; \omega)]$ be the variance of the network output under the dropout distribution. The variance of the MC mean estimator is:

$$
\text{Var}[\hat{\mu}_T] = \frac{\sigma^2_f}{T}
$$

**Standard error:**

$$
\text{SE}[\hat{\mu}_T] = \frac{\sigma_f}{\sqrt{T}}
$$

This decreases as $O(1/\sqrt{T})$, the standard Monte Carlo rate.

## Convergence Bounds

### Central Limit Theorem

For large $T$, by the Central Limit Theorem:

$$
\sqrt{T}(\hat{\mu}_T - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2_f)
$$

where $\mu = \mathbb{E}_{q_\theta}[f(\mathbf{x}; \omega)]$.

This gives asymptotic confidence intervals:

$$
P\left( \left| \hat{\mu}_T - \mu \right| \leq z_{\alpha/2} \frac{\sigma_f}{\sqrt{T}} \right) \approx 1 - \alpha
$$

For 95% confidence, $z_{0.025} \approx 1.96$.

### Finite-Sample Bounds (Hoeffding)

If the network output is bounded, $f(\mathbf{x}; \omega) \in [a, b]$, Hoeffding's inequality gives:

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq \epsilon \right) \leq 2 \exp\left( -\frac{2T\epsilon^2}{(b-a)^2} \right)
$$

**Inverting for required samples:**

To achieve $|\hat{\mu}_T - \mu| \leq \epsilon$ with probability at least $1 - \delta$:

$$
T \geq \frac{(b-a)^2 \ln(2/\delta)}{2\epsilon^2}
$$

### Finite-Sample Bounds (Chebyshev)

Without boundedness assumptions, Chebyshev's inequality gives:

$$
P\left( \left| \hat{\mu}_T - \mu \right| \geq k \frac{\sigma_f}{\sqrt{T}} \right) \leq \frac{1}{k^2}
$$

For 95% confidence ($\delta = 0.05$), need $k = \sqrt{20} \approx 4.47$.

## Variance Estimation Convergence

### Sample Variance Estimator

The epistemic uncertainty is estimated via sample variance:

$$
\hat{\sigma}^2_T = \frac{1}{T-1} \sum_{t=1}^{T} \left( f(\mathbf{x}; \hat{\omega}_t) - \hat{\mu}_T \right)^2
$$

### Distribution of Sample Variance

If $f(\mathbf{x}; \omega)$ is approximately Gaussian (often reasonable by CLT for deep networks), then:

$$
\frac{(T-1)\hat{\sigma}^2_T}{\sigma^2_f} \sim \chi^2_{T-1}
$$

**Mean and variance:**

$$
\mathbb{E}[\hat{\sigma}^2_T] = \sigma^2_f, \quad \text{Var}[\hat{\sigma}^2_T] = \frac{2\sigma^4_f}{T-1}
$$

### Relative Error in Variance Estimation

The coefficient of variation for the variance estimator:

$$
\text{CV}[\hat{\sigma}^2_T] = \frac{\sqrt{\text{Var}[\hat{\sigma}^2_T]}}{\mathbb{E}[\hat{\sigma}^2_T]} = \sqrt{\frac{2}{T-1}}
$$

| $T$ | Relative Error in Variance |
|-----|---------------------------|
| 10 | 47% |
| 30 | 26% |
| 50 | 20% |
| 100 | 14% |
| 500 | 6% |

**Implication:** Variance estimates require significantly more samples than mean estimates for comparable accuracy.

### Confidence Interval for Variance

Using the chi-squared distribution:

$$
P\left( \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, \alpha/2}} \leq \sigma^2_f \leq \frac{(T-1)\hat{\sigma}^2_T}{\chi^2_{T-1, 1-\alpha/2}} \right) = 1 - \alpha
$$

## Entropy and Mutual Information Convergence

### Predictive Entropy Estimation

Predictive entropy for classification:

$$
\mathbb{H}[\mathbf{y} | \mathbf{x}, \mathcal{D}] = -\sum_{c=1}^{C} p_c \log p_c
$$

where $p_c = \mathbb{E}_{q_\theta}[\text{softmax}(f(\mathbf{x}; \omega))_c]$.

The MC estimate uses $\hat{p}_c = \frac{1}{T} \sum_{t=1}^T \text{softmax}(f(\mathbf{x}; \hat{\omega}_t))_c$.

### Bias in Entropy Estimation

The entropy of the empirical distribution is a biased estimator:

$$
\mathbb{E}[\hat{\mathbb{H}}_T] = \mathbb{H}[p] - \frac{C - 1}{2T} + O(T^{-2})
$$

where $C$ is the number of classes. The bias is negative (entropy is underestimated).

**Miller-Madow correction:**

$$
\hat{\mathbb{H}}^{\text{MM}}_T = \hat{\mathbb{H}}_T + \frac{C - 1}{2T}
$$

### Mutual Information Convergence

Mutual information $\mathbb{I}[\mathbf{y}, \omega | \mathbf{x}, \mathcal{D}] = \mathbb{H}[\mathbf{y}] - \mathbb{E}[\mathbb{H}[\mathbf{y} | \omega]]$ involves:

1. Entropy of the averaged distribution (bias: $-\frac{C-1}{2T}$)
2. Average entropy of individual distributions (unbiased)

The biases partially cancel, but MI estimates can still be noisy for small $T$.

## Empirical Convergence Analysis

### Convergence Diagnostics

```python
import torch
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


def compute_convergence_diagnostics(
    model: torch.nn.Module,
    x: torch.Tensor,
    max_samples: int = 500,
    checkpoints: List[int] = None
) -> dict:
    """
    Compute convergence diagnostics for MC Dropout estimates.
    
    Returns running estimates of mean, variance, and their standard errors
    as a function of the number of samples.
    """
    if checkpoints is None:
        checkpoints = [5, 10, 20, 30, 50, 100, 200, 300, 500]
    checkpoints = [c for c in checkpoints if c <= max_samples]
    
    model.eval()
    model.enable_mc_dropout()
    
    # Collect all samples
    samples = []
    with torch.no_grad():
        for _ in range(max_samples):
            output = model(x)
            samples.append(output.cpu())
    
    samples = torch.stack(samples, dim=0)  # (T, B, D)
    
    results = {
        'checkpoints': checkpoints,
        'running_mean': [],
        'running_var': [],
        'mean_se': [],
        'var_se': []
    }
    
    for T in checkpoints:
        subset = samples[:T]
        
        # Running mean and variance
        mean_T = subset.mean(dim=0)
        var_T = subset.var(dim=0, unbiased=True)
        
        results['running_mean'].append(mean_T)
        results['running_var'].append(var_T)
        
        # Standard errors (using final estimate as ground truth proxy)
        # SE of mean = std / sqrt(T)
        results['mean_se'].append(subset.std(dim=0) / np.sqrt(T))
        
        # SE of variance ≈ var * sqrt(2/(T-1))
        results['var_se'].append(var_T * np.sqrt(2 / (T - 1)))
    
    return results


def plot_convergence(results: dict, output_idx: int = 0, batch_idx: int = 0):
    """Plot convergence of mean and variance estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    checkpoints = results['checkpoints']
    
    # Extract values for specific output
    means = [m[batch_idx, output_idx].item() for m in results['running_mean']]
    vars_ = [v[batch_idx, output_idx].item() for v in results['running_var']]
    mean_ses = [se[batch_idx, output_idx].item() for se in results['mean_se']]
    var_ses = [se[batch_idx, output_idx].item() for se in results['var_se']]
    
    # Mean convergence
    axes[0].errorbar(checkpoints, means, yerr=[1.96*se for se in mean_ses],
                     marker='o', capsize=3)
    axes[0].axhline(means[-1], color='r', linestyle='--', alpha=0.5,
                    label=f'Final estimate: {means[-1]:.4f}')
    axes[0].set_xlabel('Number of MC samples')
    axes[0].set_ylabel('Estimated mean')
    axes[0].set_title('Mean Convergence')
    axes[0].set_xscale('log')
    axes[0].legend()
    
    # Variance convergence
    axes[1].errorbar(checkpoints, vars_, yerr=[1.96*se for se in var_ses],
                     marker='o', capsize=3)
    axes[1].axhline(vars_[-1], color='r', linestyle='--', alpha=0.5,
                    label=f'Final estimate: {vars_[-1]:.4f}')
    axes[1].set_xlabel('Number of MC samples')
    axes[1].set_ylabel('Estimated variance')
    axes[1].set_title('Variance Convergence')
    axes[1].set_xscale('log')
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

### Effective Sample Size

When MC samples are not truly independent (e.g., correlated dropout masks across layers), the effective sample size may be less than $T$:

$$
T_{\text{eff}} = \frac{T}{1 + 2\sum_{k=1}^{\infty} \rho_k}
$$

where $\rho_k$ is the autocorrelation at lag $k$.

```python
def compute_effective_sample_size(samples: torch.Tensor) -> float:
    """
    Compute effective sample size accounting for autocorrelation.
    
    Args:
        samples: (T, ...) tensor of MC samples
        
    Returns:
        Effective sample size
    """
    T = samples.shape[0]
    samples_flat = samples.reshape(T, -1).mean(dim=1)  # Average over dimensions
    
    # Compute autocorrelation
    samples_centered = samples_flat - samples_flat.mean()
    var = (samples_centered ** 2).mean()
    
    if var < 1e-10:
        return float(T)
    
    # Autocorrelation function
    max_lag = min(T // 2, 100)
    rho_sum = 0
    
    for k in range(1, max_lag):
        rho_k = (samples_centered[:-k] * samples_centered[k:]).mean() / var
        if rho_k < 0.05:  # Cutoff for numerical stability
            break
        rho_sum += rho_k
    
    T_eff = T / (1 + 2 * rho_sum)
    return max(1.0, T_eff)
```

## Practical Sample Size Selection

### Guidelines by Task

| Task | Minimum $T$ | Recommended $T$ | Notes |
|------|-------------|-----------------|-------|
| Point prediction | 10-20 | 30 | Mean converges quickly |
| Confidence intervals | 30-50 | 100 | Need stable variance |
| Calibration curves | 100 | 200-500 | Binned statistics need precision |
| Active learning | 20-50 | 50-100 | Ranking is robust to noise |
| OOD detection | 50-100 | 100-200 | Tail behavior matters |
| Safety-critical | 200+ | 500+ | Conservative approach |

### Adaptive Sampling Strategy

```python
def adaptive_mc_sampling(
    model: torch.nn.Module,
    x: torch.Tensor,
    initial_samples: int = 20,
    max_samples: int = 500,
    tolerance: float = 0.01,
    patience: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Adaptive MC sampling that stops when estimates converge.
    
    Monitors the relative change in mean and variance estimates,
    stopping when both are stable.
    
    Args:
        model: MC Dropout model
        x: Input tensor
        initial_samples: Starting number of samples
        max_samples: Maximum samples to draw
        tolerance: Relative change threshold for convergence
        patience: Number of checks that must pass
        
    Returns:
        mean: Converged mean estimate
        std: Converged std estimate  
        n_samples: Actual number of samples used
    """
    model.eval()
    model.enable_mc_dropout()
    
    samples = []
    consecutive_stable = 0
    prev_mean, prev_var = None, None
    
    batch_size = 10  # Samples per iteration
    
    with torch.no_grad():
        # Initial samples
        for _ in range(initial_samples):
            samples.append(model(x))
        
        while len(samples) < max_samples:
            # Add more samples
            for _ in range(batch_size):
                if len(samples) >= max_samples:
                    break
                samples.append(model(x))
            
            # Compute current estimates
            stacked = torch.stack(samples, dim=0)
            curr_mean = stacked.mean(dim=0)
            curr_var = stacked.var(dim=0)
            
            # Check convergence
            if prev_mean is not None:
                mean_change = (curr_mean - prev_mean).abs() / (curr_mean.abs() + 1e-8)
                var_change = (curr_var - prev_var).abs() / (curr_var.abs() + 1e-8)
                
                mean_stable = (mean_change < tolerance).all()
                var_stable = (var_change < tolerance).all()
                
                if mean_stable and var_stable:
                    consecutive_stable += 1
                    if consecutive_stable >= patience:
                        break
                else:
                    consecutive_stable = 0
            
            prev_mean, prev_var = curr_mean, curr_var
    
    final_samples = torch.stack(samples, dim=0)
    mean = final_samples.mean(dim=0)
    std = final_samples.std(dim=0)
    
    return mean, std, len(samples)
```

### Batch Efficiency Considerations

```python
def estimate_optimal_batch_samples(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_time_ms: float = 100.0,
    warmup_runs: int = 5
) -> int:
    """
    Estimate optimal number of MC samples given time budget.
    
    Args:
        model: MC Dropout model
        x: Representative input
        target_time_ms: Time budget in milliseconds
        warmup_runs: Number of warmup forward passes
        
    Returns:
        Recommended number of MC samples
    """
    import time
    
    model.eval()
    model.enable_mc_dropout()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    # Time single forward pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    n_timing_runs = 20
    
    with torch.no_grad():
        for _ in range(n_timing_runs):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    time_per_sample = elapsed_ms / n_timing_runs
    
    optimal_samples = int(target_time_ms / time_per_sample)
    
    # Clamp to reasonable range
    return max(10, min(500, optimal_samples))
```

## Theoretical Lower Bounds

### Cramér-Rao Bound for MC Estimation

For estimating the mean $\mu$ with $T$ samples, the variance is bounded by:

$$
\text{Var}[\hat{\mu}_T] \geq \frac{\sigma^2_f}{T}
$$

MC estimation achieves this bound (is efficient) for i.i.d. samples.

### Information-Theoretic Perspective

The mutual information between the MC estimate and the true parameter:

$$
I(\hat{\mu}_T; \mu) = \frac{1}{2} \log\left(1 + \frac{T \sigma^2_\mu}{\sigma^2_f}\right)
$$

where $\sigma^2_\mu$ is the prior variance on $\mu$. Information grows logarithmically with $T$.

### Diminishing Returns

The marginal information gain from sample $T$ to $T+1$:

$$
\Delta I_T = I(\hat{\mu}_{T+1}; \mu) - I(\hat{\mu}_T; \mu) \approx \frac{\sigma^2_\mu}{2T\sigma^2_f} \quad \text{for large } T
$$

This $O(1/T)$ decay quantifies the diminishing returns of additional samples.

## Summary

**Key takeaways:**

1. **Mean estimates** converge at rate $O(1/\sqrt{T})$ — 100 samples gives ~10% standard error relative to $\sigma_f$

2. **Variance estimates** converge slower, requiring $T \approx 100$ for ~14% relative error

3. **For most applications**, $T = 50-100$ provides a good accuracy-efficiency tradeoff

4. **Safety-critical applications** should use $T \geq 200$ with convergence diagnostics

5. **Adaptive sampling** can reduce computation when estimates stabilize early

## References

1. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis*.

2. Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods. *Springer*.

3. Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. *Statistical Science*.
