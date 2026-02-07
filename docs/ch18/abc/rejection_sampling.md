# ABC Rejection Sampling

ABC rejection sampling is the simplest and most intuitive likelihood-free inference algorithm. This section presents the algorithm, its theoretical properties, practical considerations, and limitations that motivate more advanced ABC methods.

---

## The Algorithm

### Basic ABC Rejection Sampling

**Input**: Prior $p(\theta)$, simulator $p(\mathbf{x}|\theta)$, observed data $\mathbf{y}$, summary statistics $S(\cdot)$, distance $\rho(\cdot, \cdot)$, tolerance $\epsilon$, number of samples $N$

**Output**: Samples $\{\theta_1, \ldots, \theta_N\}$ from approximate posterior

```
for i = 1 to N:
    repeat:
        Sample θ ~ p(θ)
        Simulate x ~ p(x | θ)
        Compute s = S(x)
    until ρ(s, S(y)) < ε
    
    Store θᵢ = θ

return {θ₁, ..., θₙ}
```

### Python Implementation

```python
import numpy as np

def abc_rejection(prior_sampler, simulator, summary_fn, y_obs, 
                  distance_fn, epsilon, n_samples):
    """
    ABC rejection sampling.
    
    Args:
        prior_sampler: Function that returns a sample from the prior
        simulator: Function theta -> x that simulates data
        summary_fn: Function x -> s that computes summary statistics
        y_obs: Observed data
        distance_fn: Function (s1, s2) -> distance
        epsilon: Acceptance tolerance
        n_samples: Number of posterior samples desired
    
    Returns:
        samples: Array of accepted parameter values
        acceptance_rate: Fraction of proposals accepted
    """
    s_obs = summary_fn(y_obs)
    
    samples = []
    n_attempts = 0
    
    while len(samples) < n_samples:
        # Sample from prior
        theta = prior_sampler()
        
        # Simulate
        x = simulator(theta)
        s_x = summary_fn(x)
        
        # Accept/reject
        n_attempts += 1
        if distance_fn(s_x, s_obs) < epsilon:
            samples.append(theta)
    
    acceptance_rate = n_samples / n_attempts
    return np.array(samples), acceptance_rate
```

---

## Theoretical Properties

### Target Distribution

ABC rejection sampling targets the ABC posterior:

$$
p_\epsilon(\theta | \mathbf{y}) = \frac{p(\theta) \int p(\mathbf{x}|\theta) K_\epsilon(S(\mathbf{x}), S(\mathbf{y})) d\mathbf{x}}{\int p(\theta') \int p(\mathbf{x}|\theta') K_\epsilon(S(\mathbf{x}), S(\mathbf{y})) d\mathbf{x} d\theta'}
$$

where $K_\epsilon(s, s') = \mathbf{1}[\rho(s, s') < \epsilon]$ for hard threshold.

### Exactness Result

**Proposition**: If $S$ is a sufficient statistic and $\epsilon \to 0$, then:
$$
p_\epsilon(\theta | \mathbf{y}) \to p(\theta | \mathbf{y})
$$

The ABC posterior converges to the true posterior.

### Acceptance Probability

The probability of accepting a proposed $\theta$ is:

$$
\alpha(\theta) = P(\rho(S(\mathbf{X}), S(\mathbf{y})) < \epsilon | \theta) = \int_{\{s: \rho(s, S(\mathbf{y})) < \epsilon\}} p(s|\theta) ds
$$

The overall acceptance rate is:

$$
\bar{\alpha} = \int \alpha(\theta) p(\theta) d\theta
$$

### Relationship to Importance Sampling

ABC rejection can be viewed as importance sampling with:
- Proposal: $q(\theta) = p(\theta)$ (the prior)
- Weights: $w(\theta) \propto \alpha(\theta)$

The self-normalized importance sampling estimator is:

$$
\mathbb{E}_{p(\theta|\mathbf{y})}[f(\theta)] \approx \frac{\sum_{i=1}^N w_i f(\theta_i)}{\sum_{i=1}^N w_i}
$$

With rejection sampling, all accepted samples have equal weight.

---

## Acceptance Rate Analysis

### Factors Affecting Acceptance Rate

| Factor | Effect on Acceptance Rate |
|--------|--------------------------|
| Larger $\epsilon$ | Higher acceptance |
| Lower summary dimension | Higher acceptance |
| Informative prior | Higher acceptance |
| Complex model | Lower acceptance |

### Acceptance Rate vs. Tolerance

For a $d$-dimensional summary statistic with roughly spherical distribution:

$$
\bar{\alpha} \approx V_d \cdot \epsilon^d \cdot C
$$

where $V_d$ is the volume of a unit $d$-ball and $C$ depends on the model.

**Implication**: Acceptance rate decreases exponentially with summary dimension.

### Empirical Acceptance Rate Estimation

```python
def estimate_acceptance_rate(prior_sampler, simulator, summary_fn, 
                             y_obs, distance_fn, epsilon, n_trials=10000):
    """Estimate acceptance rate for given epsilon."""
    s_obs = summary_fn(y_obs)
    
    n_accepted = 0
    for _ in range(n_trials):
        theta = prior_sampler()
        x = simulator(theta)
        s_x = summary_fn(x)
        
        if distance_fn(s_x, s_obs) < epsilon:
            n_accepted += 1
    
    return n_accepted / n_trials
```

### Choosing Tolerance

**Rule of thumb**: Set $\epsilon$ to achieve 0.1%–1% acceptance rate.

**Adaptive approach**: Start with large $\epsilon$, decrease until acceptance rate is acceptable.

```python
def find_epsilon(prior_sampler, simulator, summary_fn, y_obs, 
                 distance_fn, target_rate=0.01, n_pilot=10000):
    """Find epsilon for target acceptance rate."""
    s_obs = summary_fn(y_obs)
    
    # Collect distances from pilot run
    distances = []
    for _ in range(n_pilot):
        theta = prior_sampler()
        x = simulator(theta)
        s_x = summary_fn(x)
        distances.append(distance_fn(s_x, s_obs))
    
    # Epsilon as quantile
    epsilon = np.percentile(distances, target_rate * 100)
    return epsilon
```

---

## Summary Statistics

### Requirements for Good Summaries

1. **Informative**: Capture information about $\theta$
2. **Low-dimensional**: Keep acceptance rate tractable
3. **Computable**: Can be calculated from simulated data
4. **Robust**: Not overly sensitive to noise

### Common Summary Statistics

**Location/scale**:
- Mean, median, mode
- Variance, standard deviation, IQR
- Quantiles

**Dependence**:
- Correlations, covariances
- Autocorrelations (for time series)
- Cross-correlations

**Shape**:
- Skewness, kurtosis
- Histogram bin counts
- Empirical CDF values

**Domain-specific**:
- Population genetics: allele frequencies, heterozygosity, $F_{ST}$
- Epidemiology: final size, peak time, growth rate
- Time series: spectral density, periodogram

### Example: Normal Model

For $y_1, \ldots, y_n \sim \mathcal{N}(\mu, \sigma^2)$ with unknown $(\mu, \sigma^2)$:

**Sufficient statistics**: $S(\mathbf{y}) = (\bar{y}, s^2)$ where $\bar{y} = \frac{1}{n}\sum y_i$ and $s^2 = \frac{1}{n-1}\sum(y_i - \bar{y})^2$.

ABC with these summaries and $\epsilon \to 0$ gives the exact posterior.

### Example: Time Series

For an AR(1) process $y_t = \phi y_{t-1} + \epsilon_t$:

```python
def ar1_summaries(y):
    """Summary statistics for AR(1) model."""
    return np.array([
        np.mean(y),
        np.var(y),
        np.corrcoef(y[:-1], y[1:])[0, 1],  # Lag-1 autocorrelation
    ])
```

---

## Distance Functions

### Euclidean Distance

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \|\mathbf{s}_1 - \mathbf{s}_2\|_2 = \sqrt{\sum_j (s_{1j} - s_{2j})^2}
$$

Simple but treats all summaries equally.

### Normalized Euclidean

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \sqrt{\sum_j \frac{(s_{1j} - s_{2j})^2}{\hat{\sigma}_j^2}}
$$

where $\hat{\sigma}_j^2$ is the variance of the $j$-th summary under the prior predictive.

```python
def normalized_euclidean(s1, s2, sigma):
    """Normalized Euclidean distance."""
    return np.sqrt(np.sum(((s1 - s2) / sigma)**2))
```

### Mahalanobis Distance

$$
\rho(\mathbf{s}_1, \mathbf{s}_2) = \sqrt{(\mathbf{s}_1 - \mathbf{s}_2)^T \Sigma^{-1} (\mathbf{s}_1 - \mathbf{s}_2)}
$$

Accounts for correlations between summaries.

```python
def mahalanobis_distance(s1, s2, Sigma_inv):
    """Mahalanobis distance."""
    diff = s1 - s2
    return np.sqrt(diff @ Sigma_inv @ diff)
```

### Estimating the Scaling Matrix

```python
def estimate_summary_covariance(prior_sampler, simulator, summary_fn, n_sims=1000):
    """Estimate covariance of summaries under prior predictive."""
    summaries = []
    for _ in range(n_sims):
        theta = prior_sampler()
        x = simulator(theta)
        summaries.append(summary_fn(x))
    
    return np.cov(np.array(summaries).T)
```

---

## Practical Implementation

### Complete Example: Inference for Normal Distribution

```python
import numpy as np
from scipy import stats

# True parameters and observed data
mu_true, sigma_true = 5.0, 2.0
n_obs = 100
y_obs = np.random.normal(mu_true, sigma_true, n_obs)

# Prior
def prior_sampler():
    mu = np.random.uniform(-10, 10)
    sigma = np.random.uniform(0.1, 10)
    return np.array([mu, sigma])

# Simulator
def simulator(theta):
    mu, sigma = theta
    return np.random.normal(mu, sigma, n_obs)

# Summary statistics (sufficient for this model)
def summary_fn(x):
    return np.array([np.mean(x), np.std(x, ddof=1)])

# Distance
def distance_fn(s1, s2):
    return np.linalg.norm(s1 - s2)

# Calibrate epsilon
s_obs = summary_fn(y_obs)
pilot_distances = []
for _ in range(10000):
    theta = prior_sampler()
    x = simulator(theta)
    pilot_distances.append(distance_fn(summary_fn(x), s_obs))

epsilon = np.percentile(pilot_distances, 1)  # 1% acceptance
print(f"Epsilon: {epsilon:.4f}")

# Run ABC
samples, acc_rate = abc_rejection(
    prior_sampler, simulator, summary_fn, y_obs,
    distance_fn, epsilon, n_samples=1000
)

print(f"Acceptance rate: {acc_rate:.4f}")
print(f"Posterior mean: mu={samples[:, 0].mean():.2f}, sigma={samples[:, 1].mean():.2f}")
print(f"True values: mu={mu_true}, sigma={sigma_true}")
```

### Parallelization

ABC rejection is embarrassingly parallel:

```python
from multiprocessing import Pool

def abc_rejection_parallel(prior_sampler, simulator, summary_fn, y_obs,
                          distance_fn, epsilon, n_samples, n_workers=4):
    """Parallel ABC rejection sampling."""
    s_obs = summary_fn(y_obs)
    
    def try_sample(_):
        while True:
            theta = prior_sampler()
            x = simulator(theta)
            if distance_fn(summary_fn(x), s_obs) < epsilon:
                return theta
    
    with Pool(n_workers) as pool:
        samples = pool.map(try_sample, range(n_samples))
    
    return np.array(samples)
```

### Progress Monitoring

```python
def abc_rejection_with_progress(prior_sampler, simulator, summary_fn, y_obs,
                                distance_fn, epsilon, n_samples, report_every=100):
    """ABC with progress reporting."""
    s_obs = summary_fn(y_obs)
    samples = []
    n_attempts = 0
    
    while len(samples) < n_samples:
        theta = prior_sampler()
        x = simulator(theta)
        n_attempts += 1
        
        if distance_fn(summary_fn(x), s_obs) < epsilon:
            samples.append(theta)
            
            if len(samples) % report_every == 0:
                rate = len(samples) / n_attempts
                print(f"Accepted {len(samples)}/{n_samples}, "
                      f"rate: {rate:.4f}, "
                      f"attempts: {n_attempts}")
    
    return np.array(samples), n_samples / n_attempts
```

---

## Limitations

### Low Acceptance Rate

For complex models or small $\epsilon$:
- Acceptance rate can be $< 10^{-6}$
- Millions of simulations needed
- Computationally prohibitive

### Prior Dependence

Sampling from the prior is inefficient when:
- Prior is diffuse
- Posterior is concentrated
- Prior and posterior have little overlap

### Summary Statistic Choice

Results depend critically on summaries:
- Insufficient summaries → biased posterior
- Too many summaries → low acceptance
- No systematic way to choose

### No Likelihood Approximation

ABC rejection doesn't provide a likelihood approximation—useful for model comparison but not available here.

---

## When to Use ABC Rejection

### Good Candidates

✓ Simulator is fast (< 1 second)
✓ Parameter dimension is low (< 5)
✓ Prior is informative
✓ Good summary statistics are available
✓ Moderate accuracy is sufficient

### Move to Advanced Methods When

✗ Acceptance rate is too low (< 0.01%)
✗ Prior is uninformative
✗ Need many posterior samples
✗ Higher accuracy required
✗ Simulator is expensive

**Next steps**: ABC-MCMC for better exploration, ABC-SMC for adaptive tolerance.

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Algorithm** | Sample prior, simulate, accept if close |
| **Target** | ABC posterior $p_\epsilon(\theta \| \mathbf{y})$ |
| **Acceptance rate** | Decreases with $\epsilon$, summary dimension |
| **Advantages** | Simple, parallel, exact samples from ABC posterior |
| **Limitations** | Inefficient for small $\epsilon$, prior-dependent |
| **Key choices** | Summary statistics, distance function, tolerance |

ABC rejection sampling is the foundation of likelihood-free inference. While simple, it establishes the core ideas that more sophisticated methods build upon.

---

## Exercises

1. **Implementation**. Implement ABC rejection for inferring the rate parameter $\lambda$ of a Poisson distribution. Compare to the exact posterior.

2. **Tolerance sensitivity**. For the normal model example, run ABC with $\epsilon$ at the 0.1%, 1%, and 10% quantiles of prior predictive distances. Plot the resulting posteriors and compare.

3. **Summary statistic comparison**. For inferring normal parameters, compare ABC using (a) mean and variance, (b) median and IQR, (c) first four moments. Which gives the best posterior approximation?

4. **Scaling experiment**. Measure how acceptance rate scales with summary statistic dimension (1, 2, 5, 10 summaries) for fixed $\epsilon$.

5. **Parallelization benchmark**. Compare runtime of sequential vs. parallel ABC rejection with 1, 2, 4, 8 workers.

---

## References

1. Pritchard, J. K., Seielstad, M. T., Perez-Lezaun, A., & Feldman, M. W. (1999). "Population Growth of Human Y Chromosomes: A Study of Y Chromosome Microsatellites." *Molecular Biology and Evolution*.
2. Beaumont, M. A., Zhang, W., & Balding, D. J. (2002). "Approximate Bayesian Computation in Population Genetics." *Genetics*.
3. Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). "Sequential Monte Carlo Without Likelihoods." *PNAS*.
4. Marin, J.-M., et al. (2012). "Approximate Bayesian Computational Methods." *Statistics and Computing*.
