# MCMC Convergence Diagnostics: How to Know When to Stop

## The Fundamental Challenge

MCMC produces a sequence $x^{(1)}, x^{(2)}, \ldots, x^{(N)}$ that **eventually** converges to the target distribution $\pi$.

**Questions**:
1. Has the chain converged yet?
2. How long is the burn-in period?
3. How many samples do I need for accurate estimates?
4. Are my samples effectively independent?

There is **no foolproof test** for convergence, but we have several useful diagnostics.

## Visual Diagnostics

### 1. Trace Plots

Plot the value of each parameter vs iteration number.

**Good trace plot**:
```
  x
  |     /\/\/\/\/\/\/\    "Hairy caterpillar"
  |    /\/\/\/\/\/\/\/
  |   /\/\/\/\/\/\/\/\
  |  /\/\/\/\/\/\/\/\/
  +-------------------→ iteration
```

- Rapid fluctuations (good mixing)
- Stationary (no trend)
- Covers the full range

**Bad trace plot (not converged)**:
```
  x
  |                 ----  "Stuck"
  |     ----
  |----
  +-------------------→ iteration
```

- Slow drift
- Stuck in one region
- Trend over time

**Bad trace plot (slow mixing)**:
```
  x
  |   -------    -------
  |  /       \  /       \  "Slow oscillation"
  | /         \/
  +-------------------→ iteration
```

- Slow transitions between regions
- Long periods in same area
- Low-frequency oscillations

### 2. Autocorrelation Plots

Plot autocorrelation $\rho_k = \text{Corr}(x^{(t)}, x^{(t+k)})$ vs lag $k$.

**Good autocorrelation**:
```
ρ
1 |█
  |█▓
  |█▓▒
  |█▓▒░
  |█▓▒░░░░░___________
0 +---------------------→ lag k
```

- Decays to zero quickly
- Samples become independent after ~10-20 lags

**Bad autocorrelation**:
```
ρ
1 |█
  |█
  |█▓
  |█▓▒
  |█▓▒▓▒▓▒▓▒▒▒░░___
0 +---------------------→ lag k
```

- Decays very slowly
- Samples highly correlated
- Need many more samples

### 3. Running Mean Plot

Plot cumulative mean $\bar{x}_n = \frac{1}{n}\sum_{i=1}^n x^{(i)}$ vs $n$.

**Converged**:
```
mean
  |      _________  Flat (stabilized)
  |     /
  |   /
  |  /
  +----------------→ iteration
```

**Not converged**:
```
mean
  |           /     Still changing
  |        /
  |     /
  |  /
  +----------------→ iteration
```

## Quantitative Diagnostics

### 1. Gelman-Rubin $\hat{R}$ Statistic

**Most important diagnostic!**

**Setup**: Run $M$ chains from different initial values.

**Between-chain variance**: $B = \frac{N}{M-1}\sum_{m=1}^M (\bar{x}_m - \bar{x})^2$

**Within-chain variance**: $W = \frac{1}{M}\sum_{m=1}^M s_m^2$

**Estimated marginal variance**: $\hat{V} = \frac{N-1}{N}W + \frac{1}{N}B$

**$\hat{R}$ statistic**:
$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

**Interpretation**:
- $\hat{R} \approx 1$: Chains have converged (within-chain ≈ between-chain variance)
- $\hat{R} > 1.1$: Chains have **not** converged (initial values still matter)
- $\hat{R} > 1.2$: **Serious problems** (do not trust samples)

**Rule of thumb**: Continue until $\hat{R} < 1.01$ for all parameters.

**Why it works**: If chains haven't converged, different starting points give different distributions → high between-chain variance.

**Example**:
```python
import arviz as az

# Run 4 chains
chains = [run_mcmc(init_i, n_samples=1000) for init_i in inits]

# Compute R-hat
rhat = az.rhat(np.array(chains))
print(f"R-hat: {rhat:.3f}")

if rhat < 1.01:
    print("Converged!")
else:
    print("Not converged, run longer")
```

### 2. Effective Sample Size (ESS)

Accounts for autocorrelation to estimate the **effective number of independent samples**.

**Formula**:
$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^K \rho_k}
$$

where:
- $N$ is the total number of samples
- $\rho_k$ is the autocorrelation at lag $k$
- Sum truncated when $\rho_k$ becomes negligible

**Interpretation**:
- $\text{ESS} \approx N$: Samples nearly independent (good!)
- $\text{ESS} \ll N$: Samples highly correlated (need more samples)
- $\text{ESS} < 100$: Too few effective samples for reliable inference

**Rule of thumb**: Aim for $\text{ESS} > 400$ per parameter (per chain).

**Why 400?** 
- Monte Carlo standard error scales as $1/\sqrt{\text{ESS}}$
- ESS = 400 → MCSE ≈ 5% of posterior SD
- Generally acceptable precision

**Example**:
```python
import arviz as az

ess = az.ess(samples)
print(f"Effective sample size: {ess:.0f} out of {len(samples)}")

ess_per_second = ess / runtime
print(f"Effective samples per second: {ess_per_second:.1f}")
```

### 3. Geweke Diagnostic

Compares means from early and late parts of the chain.

**Setup**: 
- First 10% of samples: $\bar{x}_A$
- Last 50% of samples: $\bar{x}_B$

**Z-score**:
$$
Z = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\text{Var}(\bar{x}_A) + \text{Var}(\bar{x}_B)}}
$$

(accounting for autocorrelation in variance estimates)

**Interpretation**:
- $|Z| < 2$: No evidence of non-convergence
- $|Z| > 2$: Chain still drifting (not stationary)

**Use case**: Check if burn-in was long enough.

### 4. Heidelberger-Welch Test

Two-part test:
1. **Stationarity test**: Is the chain stationary? (no trend)
2. **Half-width test**: Is the MCMC error small enough?

**Part 1** (Stationarity):
- Use Cramér-von Mises test on first 10%, 20%, ..., 50%
- If non-stationary, discard early portion

**Part 2** (Precision):
- Compute half-width of 95% CI: $h = 1.96 \cdot \text{SE}$
- Relative half-width: $r = h / |\bar{x}|$
- Pass if $r < 0.1$ (CI width < 10% of estimate)

**Interpretation**:
- Both pass: Chain converged with sufficient precision
- Stationarity fails: Need more burn-in
- Half-width fails: Need more samples

## Advanced Diagnostics

### 5. Monte Carlo Standard Error (MCSE)

Estimates the uncertainty in MCMC estimates due to finite sample size.

**For posterior mean**:
$$
\text{MCSE}(\bar{\theta}) = \frac{\sigma_\theta}{\sqrt{\text{ESS}}}
$$

where $\sigma_\theta$ is the posterior standard deviation.

**Rule of thumb**: Want $\text{MCSE} < 0.05 \cdot \sigma$ (5% of posterior SD).

**For quantiles** (e.g., median, 95% CI):
$$
\text{MCSE}(q_\alpha) \approx \frac{1}{\sqrt{\text{ESS}}} \cdot \frac{\sqrt{\alpha(1-\alpha)}}{p(q_\alpha)}
$$

More complex, but many packages compute automatically.

### 6. Split $\hat{R}$

Improved version of Gelman-Rubin:
- Split each chain in half
- Treat as $2M$ chains
- More robust to within-chain non-stationarity

**Modern recommendation**: Use split $\hat{R}$ instead of original.

### 7. Rank-Normalized $\hat{R}$

Even more robust version (Vehtari et al. 2021):
- Rank-normalize samples
- Less sensitive to outliers and heavy tails
- Default in modern software (Stan, ArviZ)

## Multivariate Diagnostics

For models with many parameters, check:

**Multivariate $\hat{R}$**:
- Compute $\hat{R}$ for all parameters
- Report maximum: $\max_j \hat{R}_j$
- All should be < 1.01

**Bulk ESS and Tail ESS**:
- Bulk ESS: For central 50% of distribution
- Tail ESS: For tails (5th and 95th percentiles)
- Need both to be sufficient

**Reason**: Chain might mix well in the center but poorly in the tails.

## Burn-In Period

How many initial samples to discard?

**Conservative approach**: Discard first 50% of samples.

**Diagnostic-based approach**:
1. Plot trace
2. Identify when chain enters stationary region
3. Discard samples before that point

**Multiple chains**: If using $\hat{R}$, burn-in less critical (chains must converge anyway).

**Modern practice** (Stan):
- Run warmup phase (adapt step size, mass matrix)
- Discard entire warmup
- Only keep samples from after adaptation

## Sample Size Requirements

**General guidelines**:

| Use Case | Minimum ESS |
|----------|-------------|
| Point estimates | 100 |
| Standard errors | 400 |
| Quantiles (median, 95% CI) | 1000 |
| Tail probabilities | 2000-4000 |
| High precision | 10,000+ |

**Trade-off**: More samples = better estimates but longer runtime.

**Practical advice**: Run until ESS > 400 for all parameters, then check if precision is adequate for your purposes.

## Thinning: Should You?

**Thinning**: Keep every $k$-th sample, discard the rest.

**Old advice**: Thin to reduce autocorrelation.

**Modern consensus**: **Don't thin** (usually).

**Why not thin?**
- Wastes information
- Doesn't improve ESS (just reduces $N$ proportionally)
- Modern diagnostics account for autocorrelation

**When to thin**:
- Storage constraints (millions of samples)
- Want uncorrelated samples for downstream analysis
- Computational cost of processing samples is high

## Diagnosing Common Problems

### Problem: High $\hat{R}$ (> 1.1)

**Possible causes**:
- Burn-in too short
- Multiple modes (chains stuck in different modes)
- Very slow mixing

**Solutions**:
- Run longer
- Check trace plots
- Try different initial values
- Use better sampler (HMC instead of MH)

### Problem: Low ESS (< 100)

**Possible causes**:
- High autocorrelation (slow mixing)
- Poor sampler tuning

**Solutions**:
- Run longer
- Tune sampler (step size, mass matrix)
- Use better sampler
- Reparameterize model

### Problem: Trace Shows Drift

**Possible causes**:
- Chain hasn't reached stationary distribution
- Non-stationary model (e.g., random walk prior)

**Solutions**:
- Longer burn-in
- Check model specification
- Use proper priors

### Problem: Multimodal Posterior

**Symptoms**:
- Different chains converge to different modes
- High $\hat{R}$ even with long runs
- Trace plot shows jumps between regions

**Solutions**:
- Use parallel tempering
- Initialize chains near each mode
- Consider if multimodality is real or artifact

## Practical Workflow

**Step 1: Initial Run**
```python
# Run 4 chains from dispersed initial values
chains = [run_mcmc(init, n_samples=1000) for init in inits]
```

**Step 2: Check Convergence**
```python
# Compute diagnostics
rhat = az.rhat(chains)
ess = az.ess(chains)

print(f"Max R-hat: {rhat.max():.3f}")
print(f"Min ESS: {ess.min():.0f}")
```

**Step 3: Visual Inspection**
```python
# Plot traces
az.plot_trace(chains)
plt.show()

# Plot autocorrelation
az.plot_autocorr(chains, max_lag=50)
plt.show()
```

**Step 4: Decide**
- If $\hat{R} < 1.01$ and $\text{ESS} > 400$ → converged
- If not → run longer or tune sampler

**Step 5: Run Production**
```python
# Run longer with tuned parameters
final_chains = [run_mcmc(init, n_samples=10000) for init in inits]
```

## Software Tools

**ArviZ** (Python):
```python
import arviz as az

# Comprehensive summary
az.summary(samples, hdi_prob=0.95)

# Includes: mean, sd, hdi, ess_bulk, ess_tail, r_hat
```

**Coda** (R):
```r
library(coda)

# Gelman-Rubin
gelman.diag(mcmc_list)

# Effective sample size
effectiveSize(mcmc_samples)
```

**Stan**:
```python
# Stan computes diagnostics automatically
fit = stan.build(model, data=data)
fit.summary()  # Includes R-hat, ESS

# Warnings for problems
fit.diagnose()
```

## Summary Checklist

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

**If any check fails**: Run longer, tune better, or use a different sampler.

**Golden rule**: When in doubt, run longer. Computation is cheap; wrong inferences are expensive.

## Further Reading

**Classic papers**:
- Gelman & Rubin (1992): $\hat{R}$ statistic
- Geweke (1992): Time series diagnostics
- Heidelberger & Welch (1983): Stationarity tests

**Modern standards**:
- Vehtari et al. (2021): Rank-normalized $\hat{R}$, ESS recommendations
- Gelman et al. (2013): Bayesian Data Analysis (Chapter 11)

**Software documentation**:
- ArviZ diagnostics guide
- Stan convergence diagnostics
- PyMC convergence docs

The field continues to develop better diagnostics — use the most recent recommendations from your chosen software!
