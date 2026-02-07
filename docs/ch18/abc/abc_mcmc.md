# ABC-MCMC

ABC-MCMC combines Approximate Bayesian Computation with Markov chain Monte Carlo, enabling more efficient exploration of the parameter space than rejection sampling. This section presents the algorithm, its theoretical justification, and practical implementation guidance.

---

## Motivation

### Limitations of ABC Rejection

ABC rejection sampling has a fundamental inefficiency: it samples from the prior, which may have little overlap with the posterior. When the posterior is concentrated relative to the prior:
- Most proposals are wasted
- Acceptance rate becomes vanishingly small
- Computational cost explodes

### The MCMC Solution

Instead of independent sampling from the prior, use MCMC to:
- Propose from the current location (not the prior)
- Concentrate samples in high-posterior regions
- Achieve reasonable acceptance rates even for concentrated posteriors

---

## The ABC-MCMC Algorithm

### Algorithm Statement

**Input**: Initial $\theta_0$, prior $p(\theta)$, simulator $p(\mathbf{x}|\theta)$, observed data $\mathbf{y}$, summary statistics $S(\cdot)$, distance $\rho(\cdot, \cdot)$, tolerance $\epsilon$, proposal $q(\theta'|\theta)$, iterations $T$

**Output**: Chain $\{\theta_0, \theta_1, \ldots, \theta_T\}$ targeting ABC posterior

```
Simulate x₀ ~ p(x | θ₀)
Compute s₀ = S(x₀)

for t = 1 to T:
    # Propose
    θ' ~ q(θ' | θₜ₋₁)
    
    # Simulate
    x' ~ p(x | θ')
    s' = S(x')
    
    # ABC-MH acceptance probability
    if ρ(s', S(y)) < ε:
        α = min(1, [p(θ') q(θₜ₋₁ | θ')] / [p(θₜ₋₁) q(θ' | θₜ₋₁)])
    else:
        α = 0
    
    # Accept/reject
    if U(0,1) < α:
        θₜ = θ'
        sₜ = s'
    else:
        θₜ = θₜ₋₁
        sₜ = sₜ₋₁

return {θ₀, θ₁, ..., θₜ}
```

### Key Insight

The acceptance probability has two components:
1. **ABC criterion**: Is the simulated data close enough? ($\rho(s', S(\mathbf{y})) < \epsilon$)
2. **MH ratio**: Balances proposal and prior

If the ABC criterion fails, we reject immediately (α = 0).

---

## Theoretical Foundation

### Target Distribution

ABC-MCMC targets the joint distribution:

$$
\pi_\epsilon(\theta, \mathbf{s}) \propto p(\theta) p(\mathbf{s}|\theta) \mathbf{1}[\rho(\mathbf{s}, S(\mathbf{y})) < \epsilon]
$$

The marginal over $\theta$ is the ABC posterior:

$$
p_\epsilon(\theta | \mathbf{y}) = \int \pi_\epsilon(\theta, \mathbf{s}) d\mathbf{s}
$$

### Detailed Balance

The ABC-MCMC kernel satisfies detailed balance with respect to $\pi_\epsilon(\theta, \mathbf{s})$.

**Proof sketch**: 
- The proposal $q(\theta'|\theta) p(\mathbf{s}'|\theta')$ proposes new $(\theta', \mathbf{s}')$
- The acceptance probability is standard MH on the extended space
- The indicator function is symmetric in $(\theta, \mathbf{s})$ and $(\theta', \mathbf{s}')$

### Ergodicity

Under standard conditions (irreducibility, aperiodicity), the chain converges:

$$
\frac{1}{T}\sum_{t=1}^T f(\theta_t) \to \mathbb{E}_{p_\epsilon(\theta|\mathbf{y})}[f(\theta)]
$$

---

## Implementation

### Basic Implementation

```python
import numpy as np

def abc_mcmc(theta_init, prior_logpdf, simulator, summary_fn, y_obs,
             distance_fn, epsilon, proposal_fn, proposal_logpdf, n_iter):
    """
    ABC-MCMC algorithm.
    
    Args:
        theta_init: Initial parameter value
        prior_logpdf: Function theta -> log p(theta)
        simulator: Function theta -> x
        summary_fn: Function x -> summary statistics
        y_obs: Observed data
        distance_fn: Function (s1, s2) -> distance
        epsilon: ABC tolerance
        proposal_fn: Function theta -> theta' (sample from proposal)
        proposal_logpdf: Function (theta', theta) -> log q(theta' | theta)
        n_iter: Number of iterations
    
    Returns:
        chain: Array of shape (n_iter, dim)
        acceptance_rate: Overall acceptance rate
    """
    s_obs = summary_fn(y_obs)
    dim = len(theta_init)
    
    # Initialize
    theta = theta_init.copy()
    x = simulator(theta)
    s = summary_fn(x)
    
    # Check initial point is valid
    if distance_fn(s, s_obs) >= epsilon:
        raise ValueError("Initial point does not satisfy ABC criterion")
    
    chain = np.zeros((n_iter, dim))
    n_accepted = 0
    
    for t in range(n_iter):
        # Propose
        theta_prop = proposal_fn(theta)
        
        # Simulate
        x_prop = simulator(theta_prop)
        s_prop = summary_fn(x_prop)
        
        # ABC-MH acceptance
        if distance_fn(s_prop, s_obs) < epsilon:
            # Compute MH ratio
            log_alpha = (prior_logpdf(theta_prop) - prior_logpdf(theta) +
                        proposal_logpdf(theta, theta_prop) - 
                        proposal_logpdf(theta_prop, theta))
            
            if np.log(np.random.rand()) < log_alpha:
                theta = theta_prop
                s = s_prop
                n_accepted += 1
        
        chain[t] = theta
    
    return chain, n_accepted / n_iter
```

### Finding a Valid Initial Point

The chain must start from a point satisfying the ABC criterion:

```python
def find_initial_point(prior_sampler, simulator, summary_fn, y_obs,
                       distance_fn, epsilon, max_attempts=100000):
    """Find a valid starting point for ABC-MCMC."""
    s_obs = summary_fn(y_obs)
    
    for _ in range(max_attempts):
        theta = prior_sampler()
        x = simulator(theta)
        s = summary_fn(x)
        
        if distance_fn(s, s_obs) < epsilon:
            return theta
    
    raise RuntimeError(f"Could not find valid initial point in {max_attempts} attempts")
```

### Adaptive Proposal

Adapt the proposal covariance during burn-in:

```python
class AdaptiveABCMCMC:
    def __init__(self, dim, target_rate=0.234):
        self.dim = dim
        self.target_rate = target_rate
        self.cov = np.eye(dim)
        self.scale = 2.4**2 / dim
        self.mean = np.zeros(dim)
        self.n = 0
        
    def propose(self, theta):
        return theta + np.random.multivariate_normal(
            np.zeros(self.dim), self.scale * self.cov
        )
    
    def adapt(self, theta, accepted):
        """Update proposal based on chain history."""
        self.n += 1
        
        # Update running mean and covariance
        delta = theta - self.mean
        self.mean += delta / self.n
        
        if self.n > 1:
            self.cov = ((self.n - 2) / (self.n - 1) * self.cov + 
                       delta.reshape(-1, 1) @ delta.reshape(1, -1) / self.n)
        
        # Adapt scale based on acceptance
        if self.n > 100:
            if accepted:
                self.scale *= 1.01
            else:
                self.scale *= 0.99
```

---

## Comparison with ABC Rejection

### Efficiency Comparison

| Aspect | ABC Rejection | ABC-MCMC |
|--------|---------------|----------|
| Proposals from | Prior | Local to current |
| Independence | Samples are independent | Samples are correlated |
| Acceptance rate | Can be very low | Usually higher |
| Exploration | Complete (from prior) | Local (needs good mixing) |
| Parallelization | Trivial | More complex |
| Burn-in | Not needed | Required |

### When ABC-MCMC Helps

ABC-MCMC is advantageous when:
- Prior is much wider than posterior
- ABC rejection acceptance rate is < 0.1%
- Many posterior samples are needed

### When ABC Rejection May Be Better

ABC rejection may be preferable when:
- Acceptance rate is reasonable (> 1%)
- Posterior is multimodal (MCMC may get stuck)
- Parallel resources are abundant
- Independence of samples is important

---

## Practical Considerations

### Proposal Distribution

**Random walk**:
$$
q(\theta'|\theta) = \mathcal{N}(\theta, \Sigma)
$$

- Simple to implement
- Requires tuning $\Sigma$
- Scale with dimension: $\Sigma = (2.4^2/d) \hat{\Sigma}$

**Independent proposal** (not recommended for ABC-MCMC):
- Falls back to ABC rejection
- Loses MCMC benefits

### Tuning the Proposal

**Target acceptance rate**: 10-30% for ABC-MCMC (lower than standard MH due to ABC rejections).

```python
def tune_proposal_scale(abc_mcmc_fn, theta_init, target_rate=0.2, 
                        n_tune=1000, n_test=500):
    """Tune proposal scale for target acceptance rate."""
    scale = 1.0
    
    for _ in range(n_tune):
        _, acc_rate = abc_mcmc_fn(theta_init, scale, n_test)
        
        if acc_rate > target_rate + 0.05:
            scale *= 1.2
        elif acc_rate < target_rate - 0.05:
            scale *= 0.8
        else:
            break
    
    return scale
```

### Handling Low Acceptance

If acceptance is too low:
1. Increase $\epsilon$
2. Reduce proposal variance
3. Use better summary statistics
4. Consider ABC-SMC instead

### Diagnostics

Standard MCMC diagnostics apply:

```python
def abc_mcmc_diagnostics(chain):
    """Compute diagnostics for ABC-MCMC chain."""
    from statsmodels.tsa.stattools import acf
    
    n_samples, dim = chain.shape
    
    diagnostics = {}
    
    # Effective sample size (per dimension)
    ess = []
    for d in range(dim):
        autocorr = acf(chain[:, d], nlags=100, fft=True)
        tau = 1 + 2 * np.sum(autocorr[1:])
        ess.append(n_samples / tau)
    diagnostics['ess'] = np.array(ess)
    
    # Trace plots should be examined visually
    diagnostics['mean'] = np.mean(chain, axis=0)
    diagnostics['std'] = np.std(chain, axis=0)
    
    return diagnostics
```

---

## Variants

### ABC-MCMC with Multiple Simulations

Reduce variance by simulating multiple datasets per proposal:

```python
def abc_mcmc_multiple_sims(theta, simulator, summary_fn, s_obs, 
                           distance_fn, epsilon, n_sims=10):
    """ABC acceptance based on multiple simulations."""
    n_close = 0
    for _ in range(n_sims):
        x = simulator(theta)
        s = summary_fn(x)
        if distance_fn(s, s_obs) < epsilon:
            n_close += 1
    
    return n_close / n_sims  # Estimated acceptance probability
```

### Noisy ABC-MCMC

Add noise to the acceptance criterion to improve mixing:

$$
\alpha = \min\left(1, \frac{p(\theta') K_\epsilon(s', s_{obs})}{p(\theta) K_\epsilon(s, s_{obs})} \cdot \frac{q(\theta|\theta')}{q(\theta'|\theta)}\right)
$$

where $K_\epsilon$ is a smooth kernel rather than hard threshold.

### Gibbs-ABC

For models with multiple parameter blocks, update blocks separately:

```python
def gibbs_abc(theta, blocks, simulators, summary_fns, s_obs, 
              distance_fns, epsilons, proposals):
    """Gibbs-style ABC-MCMC updating parameter blocks."""
    for i, block in enumerate(blocks):
        # Propose new value for block i
        theta_prop = theta.copy()
        theta_prop[block] = proposals[i](theta[block])
        
        # Simulate and check ABC criterion
        x_prop = simulators[i](theta_prop)
        s_prop = summary_fns[i](x_prop)
        
        if distance_fns[i](s_prop, s_obs[i]) < epsilons[i]:
            # MH acceptance for this block
            # ...
            theta = theta_prop
    
    return theta
```

---

## Example: Inference for a Stochastic Volatility Model

```python
import numpy as np

# Stochastic volatility model:
# y_t = exp(h_t/2) * eps_t,  eps_t ~ N(0,1)
# h_t = mu + phi*(h_{t-1} - mu) + sigma*eta_t,  eta_t ~ N(0,1)

def sv_simulator(theta, T=500):
    """Simulate stochastic volatility model."""
    mu, phi, sigma = theta
    
    # Initialize
    h = np.zeros(T)
    h[0] = mu + sigma * np.random.randn() / np.sqrt(1 - phi**2)
    
    # Simulate log-volatility
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sigma * np.random.randn()
    
    # Simulate returns
    y = np.exp(h / 2) * np.random.randn(T)
    
    return y

def sv_summaries(y):
    """Summary statistics for SV model."""
    log_y2 = np.log(y**2 + 1e-8)
    
    return np.array([
        np.mean(log_y2),
        np.std(log_y2),
        np.corrcoef(log_y2[:-1], log_y2[1:])[0, 1],
        np.corrcoef(log_y2[:-2], log_y2[2:])[0, 1],
        np.mean(np.abs(y)),
        np.std(np.abs(y)),
    ])

# Prior
def sv_prior_sample():
    mu = np.random.uniform(-2, 2)
    phi = np.random.uniform(0.8, 0.999)
    sigma = np.random.uniform(0.01, 0.5)
    return np.array([mu, phi, sigma])

def sv_prior_logpdf(theta):
    mu, phi, sigma = theta
    if not (-2 < mu < 2 and 0.8 < phi < 0.999 and 0.01 < sigma < 0.5):
        return -np.inf
    return 0  # Uniform prior

# Run ABC-MCMC
y_obs = sv_simulator([0.0, 0.95, 0.2])  # True parameters
s_obs = sv_summaries(y_obs)

# Find initial point and run chain...
```

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Algorithm** | MH with ABC acceptance criterion |
| **Target** | ABC posterior $p_\epsilon(\theta \| \mathbf{y})$ |
| **Advantage** | More efficient than rejection when prior is diffuse |
| **Challenge** | Requires valid starting point, careful tuning |
| **Acceptance** | Two-stage: ABC criterion then MH ratio |
| **Diagnostics** | Standard MCMC diagnostics apply |

ABC-MCMC bridges the gap between simple rejection sampling and more sophisticated methods, providing improved efficiency while maintaining the simplicity of the ABC framework.

---

## Exercises

1. **Implementation**. Implement ABC-MCMC for the normal model. Compare efficiency (ESS per simulation) to ABC rejection.

2. **Adaptation**. Implement adaptive ABC-MCMC with proposal covariance learning. Show that it improves mixing.

3. **Initialization sensitivity**. Study how different initialization strategies affect burn-in length and final results.

4. **Proposal tuning**. For a fixed problem, find the optimal proposal scale empirically. How does it compare to standard MH guidelines?

5. **Multimodality**. Create a bimodal posterior and show that ABC-MCMC can get stuck. Propose a solution.

---

## References

1. Marjoram, P., Molitor, J., Plagnol, V., & Tavaré, S. (2003). "Markov Chain Monte Carlo Without Likelihoods." *PNAS*.
2. Sisson, S. A., & Fan, Y. (2011). "Likelihood-Free Markov Chain Monte Carlo." In *Handbook of Markov Chain Monte Carlo*.
3. Wegmann, D., Leuenberger, C., & Excoffier, L. (2009). "Efficient Approximate Bayesian Computation Coupled with Markov Chain Monte Carlo Without Likelihood." *Genetics*.
4. Bortot, P., Coles, S. G., & Sisson, S. A. (2007). "Inference for Stereological Extremes." *JASA*.
