# MCMC Diagnostics

Running an MCMC sampler produces a sequence of correlated samples, but there is no automatic guarantee that these samples faithfully represent the target posterior. Diagnostics are essential for assessing convergence, mixing quality, and the reliability of posterior summaries.

---

## The Fundamental Problem

MCMC provides **asymptotic** guarantees: as the number of samples $T \to \infty$, the empirical distribution converges to the target. But in practice we have finite samples, and we must assess whether:

1. The chain has **converged** to the stationary distribution (burn-in complete)
2. The chain is **mixing** well (exploring the full posterior efficiently)
3. We have **enough effective samples** for reliable inference

---

## Convergence Diagnostics

### Split-$\hat{R}$ (Gelman-Rubin)

The gold-standard convergence diagnostic compares **between-chain** and **within-chain** variance. Run $M$ chains from dispersed starting points:

$$
\hat{R} = \sqrt{\frac{\hat{\text{Var}}^+(\theta \mid \mathcal{D})}{W}}
$$

where $W$ is the within-chain variance and $\hat{\text{Var}}^+$ is the pooled variance estimate.

| $\hat{R}$ Value | Interpretation |
|------------------|---------------|
| < 1.01 | Converged (recommended threshold) |
| 1.01 - 1.05 | Marginal — run longer |
| > 1.05 | Not converged — do not use results |
| > 1.1 | Serious convergence failure |

**Split-$\hat{R}$** (recommended): Split each chain in half and treat the halves as separate chains, detecting within-chain non-stationarity.

### Implementation

```python
import torch


def split_rhat(chains: torch.Tensor) -> torch.Tensor:
    """
    Compute split-R-hat for convergence assessment.
    
    Parameters
    ----------
    chains : Tensor of shape (M, N, d)
        M chains, N samples each, d parameters
    
    Returns
    -------
    rhat : Tensor of shape (d,)
    """
    M, N, d = chains.shape
    
    # Split each chain in half
    N_half = N // 2
    split_chains = torch.cat([
        chains[:, :N_half, :],
        chains[:, N_half:2*N_half, :]
    ], dim=0)  # (2M, N_half, d)
    
    m = split_chains.shape[0]  # 2M
    n = N_half
    
    # Within-chain variance
    chain_vars = split_chains.var(dim=1)  # (2M, d)
    W = chain_vars.mean(dim=0)  # (d,)
    
    # Between-chain variance
    chain_means = split_chains.mean(dim=1)  # (2M, d)
    B = n * chain_means.var(dim=0)  # (d,)
    
    # Pooled variance estimate
    var_hat = (n - 1) / n * W + B / n
    
    rhat = torch.sqrt(var_hat / W)
    return rhat
```

---

## Effective Sample Size (ESS)

### Definition

Due to autocorrelation, $N$ MCMC samples contain the information of fewer independent samples:

$$
\text{ESS} = \frac{MN}{1 + 2\sum_{k=1}^{K} \hat{\rho}_k}
$$

where $\hat{\rho}_k$ is the estimated autocorrelation at lag $k$, and the sum is truncated when autocorrelations become noisy.

### Bulk and Tail ESS

- **Bulk ESS**: Effective samples for estimating the central posterior (mean, median)
- **Tail ESS**: Effective samples for estimating tail quantities (95% credible intervals)

Both should be monitored — it is possible to have good bulk ESS but poor tail ESS.

### Rules of Thumb

| ESS | Reliability |
|-----|-------------|
| > 400 per chain | Good for most summaries |
| > 100 per chain | Acceptable minimum |
| < 100 per chain | Unreliable — run longer or reparameterize |

### Implementation

```python
def effective_sample_size(samples: torch.Tensor, max_lag: int = None) -> float:
    """
    Estimate ESS using initial positive sequence estimator.
    
    Parameters
    ----------
    samples : Tensor of shape (N,) — single chain, single parameter
    """
    N = len(samples)
    if max_lag is None:
        max_lag = N // 2
    
    # Compute autocorrelation via FFT
    x = samples - samples.mean()
    fft_result = torch.fft.rfft(x, n=2 * N)
    acf = torch.fft.irfft(fft_result * fft_result.conj())[:N]
    acf = acf / acf[0]
    
    # Sum autocorrelations using initial positive sequence
    tau = 1.0
    for k in range(1, max_lag, 2):
        rho_pair = acf[k] + acf[k + 1] if k + 1 < max_lag else acf[k]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair
    
    return N / tau
```

---

## Visual Diagnostics

### Trace Plots

The most basic diagnostic — plot parameter values against iteration number.

| Pattern | Interpretation |
|---------|---------------|
| Stationary "hairy caterpillar" | Good mixing ✓ |
| Trends or drifts | Not converged ✗ |
| Flat regions (stuck) | Poor mixing ✗ |
| Multiple chains overlapping | Agreement ✓ |

### Autocorrelation Plots

Plot $\hat{\rho}_k$ vs lag $k$. Rapid decay indicates good mixing; slow decay indicates high autocorrelation.

### Rank Plots

Plot the histogram of within-chain ranks across chains. For converged chains, these should be approximately uniform.

### Pair Plots

For multivariate posteriors, scatter plots of parameter pairs reveal correlations and multimodality.

---

## NUTS-Specific Diagnostics

### Divergent Transitions

Divergences indicate the leapfrog integrator encountered problematic geometry. **Any divergences invalidate the inference.**

**Solutions:**
1. Increase `adapt_delta` (e.g., 0.95 → 0.99) — smaller step size
2. Reparameterize the model (non-centered parameterization)
3. Use stronger priors to eliminate pathological regions

### Tree Depth

If many iterations hit the maximum tree depth, NUTS is unable to find efficient trajectories. This often indicates high posterior curvature or funnel geometry.

### Energy Diagnostic (E-BFMI)

The Bayesian fraction of missing information compares the marginal energy distribution to the energy transition distribution. E-BFMI < 0.3 suggests the sampler has difficulty exploring the target.

---

## Practical Workflow

```
1. Run 4+ chains from dispersed starting points
2. Check split-R̂ < 1.01 for ALL parameters
3. Check bulk ESS > 400 and tail ESS > 400
4. Check 0 divergences (for NUTS/HMC)
5. Inspect trace plots for any pathology
6. If diagnostics fail:
   a. Reparameterize model
   b. Run chains longer
   c. Adjust sampler settings
   d. Simplify model if necessary
7. Only report results after all diagnostics pass
```

---

## Summary

| Diagnostic | What It Checks | Threshold |
|-----------|----------------|-----------|
| **Split-$\hat{R}$** | Between/within-chain agreement | < 1.01 |
| **Bulk ESS** | Effective samples for central summaries | > 400 |
| **Tail ESS** | Effective samples for interval estimates | > 400 |
| **Divergences** | Numerical integration failures | = 0 |
| **Tree depth** | Sampler efficiency | Not hitting max |
| **Trace plots** | Visual convergence and mixing | Stationary, well-mixed |

---

## References

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. Chapter 11.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.

---

## Additional Diagnostics

### Geweke Diagnostic

Compares means from early and late parts of the chain.

- First 10% of samples: $\bar{x}_A$
- Last 50% of samples: $\bar{x}_B$

**Z-score**:
$$
Z = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\text{Var}(\bar{x}_A) + \text{Var}(\bar{x}_B)}}
$$

$|Z| < 2$: No evidence of non-convergence. $|Z| > 2$: Chain still drifting.

### Heidelberger-Welch Test

Two-part test:

1. **Stationarity test**: Cramér-von Mises test on first 10%, 20%, ..., 50%
2. **Half-width test**: Is the MCMC error small enough? Pass if half-width of 95% CI is < 10% of estimate.

### Monte Carlo Standard Error (MCSE)

$$
\text{MCSE}(\bar{\theta}) = \frac{\sigma_\theta}{\sqrt{\text{ESS}}}
$$

**Rule of thumb**: Want $\text{MCSE} < 0.05 \cdot \sigma$ (5% of posterior SD).

---

## Sample Size Requirements

| Use Case | Minimum ESS |
|----------|-------------|
| Point estimates | 100 |
| Standard errors | 400 |
| Quantiles (median, 95% CI) | 1000 |
| Tail probabilities | 2000-4000 |
| High precision | 10,000+ |

---

## Thinning: Should You?

**Modern consensus**: **Don't thin** (usually).

- Wastes information
- Doesn't improve ESS (just reduces $N$ proportionally)
- Modern diagnostics account for autocorrelation

**When to thin**: Storage constraints, need uncorrelated samples for downstream analysis, computational cost of processing samples is high.

---

## Diagnosing Common Problems

### High $\hat{R}$ (> 1.1)

**Causes**: Burn-in too short, multiple modes (chains stuck in different modes), very slow mixing.

**Solutions**: Run longer, check trace plots, try different initial values, use better sampler (HMC instead of MH).

### Low ESS (< 100)

**Causes**: High autocorrelation, poor sampler tuning.

**Solutions**: Run longer, tune sampler (step size, mass matrix), reparameterize model.

### Multimodal Posterior

**Symptoms**: Different chains converge to different modes, high $\hat{R}$ even with long runs.

**Solutions**: Use parallel tempering, initialize chains near each mode, consider if multimodality is real or artifact.

---

## Burn-In Period

**Conservative approach**: Discard first 50% of samples.

**Modern practice** (Stan): Run warmup phase (adapt step size, mass matrix), discard entire warmup, only keep post-adaptation samples.

---

## Software Tools

**ArviZ** (Python):
```python
import arviz as az

# Comprehensive summary
az.summary(samples, hdi_prob=0.95)
# Includes: mean, sd, hdi, ess_bulk, ess_tail, r_hat
```

**Stan**:
```python
fit = stan.build(model, data=data)
fit.summary()  # Includes R-hat, ESS
fit.diagnose()  # Warnings for problems
```

---

## Extended Checklist

Before trusting your MCMC samples, verify:

- [ ] Ran at least 4 chains from dispersed initial values
- [ ] All $\hat{R} < 1.01$
- [ ] All ESS > 400 (preferably 1000+)
- [ ] Trace plots look stationary ("hairy caterpillar")
- [ ] Autocorrelation decays to near-zero within ~20 lags
- [ ] Running mean stabilizes
- [ ] No systematic trends or drift
- [ ] Bulk and tail ESS both sufficient
- [ ] MCSE < 5% of posterior SD
- [ ] 0 divergences (for NUTS/HMC)

**Golden rule**: When in doubt, run longer. Computation is cheap; wrong inferences are expensive.

---

## Further Reading

- Gelman & Rubin (1992): $\hat{R}$ statistic
- Geweke (1992): Time series diagnostics
- Vehtari et al. (2021): Rank-normalized $\hat{R}$, ESS recommendations
- Gelman et al. (2013): Bayesian Data Analysis (Chapter 11)
