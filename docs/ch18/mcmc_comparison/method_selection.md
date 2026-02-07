# Practical Method Selection

Choosing the right MCMC method for a given problem is part science, part art. This section provides practical guidance for method selection based on problem characteristics, computational constraints, and diagnostic outcomes.

---

## Decision Framework

### The Key Questions

Before selecting a method, answer these questions:

1. **Is the target differentiable?**
   - Yes → Gradient-based methods (MALA, HMC) are available
   - No → Must use gradient-free (MH, Gibbs)

2. **What is the dimension?**
   - $d < 20$ → Most methods work
   - $20 < d < 200$ → MALA or HMC preferred
   - $d > 200$ → HMC/NUTS essential (or specialized methods)

3. **Are full conditionals tractable?**
   - Yes → Gibbs is an option
   - Partially → Metropolis-within-Gibbs
   - No → Full MH or gradient-based

4. **Is the target multimodal?**
   - Yes → Consider tempering, multiple chains
   - No → Standard methods likely sufficient

5. **What are the computational constraints?**
   - Gradient expensive → RWM or Gibbs
   - Parallel resources → Multiple chains
   - Real-time → Fast-mixing required

### Decision Tree

```
Start
  │
  ├─ Is target differentiable?
  │   │
  │   ├─ NO ──→ Are conditionals tractable?
  │   │          │
  │   │          ├─ YES ──→ Gibbs Sampling
  │   │          │
  │   │          └─ NO ───→ Random Walk MH
  │   │
  │   └─ YES ─→ Is dimension high (d > 20)?
  │              │
  │              ├─ NO ──→ Any method works; try MALA first
  │              │
  │              └─ YES ─→ Is gradient cheap?
  │                        │
  │                        ├─ YES ──→ HMC / NUTS
  │                        │
  │                        └─ NO ───→ MALA or Stochastic Gradient
```

---

## Method Profiles

### Random Walk Metropolis-Hastings

**Best for**:
- Low dimensions ($d < 20$)
- Non-differentiable targets
- Quick prototyping
- Discrete components in mixed models

**Avoid when**:
- $d > 50$ (mixing becomes prohibitive)
- Strong correlations (without reparameterization)
- Real-time applications

**Tuning**:
```python
# Start with this, adjust based on acceptance
sigma = 2.4 * np.std(initial_samples, axis=0) / np.sqrt(d)
# Target: 20-25% acceptance in high d, up to 50% in low d
```

**Red flags**:
- Acceptance rate < 10% or > 50%
- Trace plots show "stickiness"
- ESS per iteration < 0.01

### Gibbs Sampling

**Best for**:
- Conjugate models (Bayesian linear regression, GMMs)
- Conditionals are standard distributions
- Hierarchical models with conjugate priors
- High dimension with sparse structure

**Avoid when**:
- Strong correlations between variables
- Conditionals are non-standard
- All-at-once updates would be more efficient

**Tuning**: Generally parameter-free, but:
```python
# Random scan can help with correlations
scan_order = np.random.permutation(d)  # Each iteration

# Block Gibbs for correlated groups
blocks = [[0, 1, 2], [3, 4, 5], ...]  # Group correlated variables
```

**Red flags**:
- Extremely slow mixing in some coordinates
- High autocorrelation despite 100% acceptance
- Conditional sampling is slow/complex

### MALA (Metropolis-Adjusted Langevin)

**Best for**:
- Moderate dimensions ($20 < d < 200$)
- Smooth, differentiable log-densities
- When gradients are cheap
- As a step up from RWM

**Avoid when**:
- Target has discontinuities
- Gradient is expensive or unavailable
- Very high dimensions (HMC better)
- Heavy tails (gradient unreliable)

**Tuning**:
```python
# Initial step size
epsilon = 1.0 / d**(1/6)

# Adapt to target ~57% acceptance
def adapt_epsilon(epsilon, accept_rate, target=0.574):
    if accept_rate > target + 0.05:
        return epsilon * 1.1
    elif accept_rate < target - 0.05:
        return epsilon * 0.9
    return epsilon
```

**Red flags**:
- Acceptance rate far from 57%
- Gradient evaluations returning NaN/Inf
- Much worse than RWM (indicates gradient issues)

### Hamiltonian Monte Carlo

**Best for**:
- High dimensions ($d > 50$)
- Smooth, log-concave (or mildly multimodal) targets
- When gradient cost is acceptable
- When high ESS is needed

**Avoid when**:
- Target is non-differentiable
- Discrete parameters present
- Multimodal with high barriers
- Gradient is very expensive

**Tuning**:
```python
# Use NUTS to avoid tuning L
# For manual HMC:
epsilon = 0.1 / d**(1/4)
L = int(np.ceil(np.sqrt(d)))

# Mass matrix: start diagonal, consider dense for correlations
M = np.diag(1.0 / np.var(warmup_samples, axis=0))
```

**Red flags**:
- Divergent transitions (energy errors > 1000)
- Tree depth hitting maximum (in NUTS)
- Low E-BFMI (< 0.2)
- Acceptance rate outside 50-80%

---

## Problem-Specific Recommendations

### Bayesian Linear Regression

$$
y = X\beta + \epsilon, \quad \beta \sim \mathcal{N}(0, \sigma_\beta^2 I), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

**Recommendation**: **Gibbs Sampling**

**Why**: Full conditionals are available:
- $\beta | y, \sigma^2 \sim \mathcal{N}(\cdot, \cdot)$
- $\sigma^2 | y, \beta \sim \text{Inverse-Gamma}(\cdot, \cdot)$

```python
def gibbs_linear_regression(y, X, n_samples, prior_var=100):
    n, p = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Initialize
    sigma2 = 1.0
    beta = np.zeros(p)
    
    samples = []
    for _ in range(n_samples):
        # Sample beta | y, sigma2
        V_post = np.linalg.inv(XtX / sigma2 + np.eye(p) / prior_var)
        m_post = V_post @ (Xty / sigma2)
        beta = np.random.multivariate_normal(m_post, V_post)
        
        # Sample sigma2 | y, beta
        resid = y - X @ beta
        a_post = n / 2
        b_post = np.sum(resid**2) / 2
        sigma2 = 1 / np.random.gamma(a_post, 1/b_post)
        
        samples.append({'beta': beta.copy(), 'sigma2': sigma2})
    
    return samples
```

### Bayesian Logistic Regression

$$
y_i \sim \text{Bernoulli}(\sigma(x_i^T\beta)), \quad \beta \sim \mathcal{N}(0, \sigma_\beta^2 I)
$$

**Recommendation**: **HMC / NUTS**

**Why**: 
- No conjugacy (Gibbs not applicable)
- Gradient is cheap: $\nabla \log p(\beta|y) = X^T(y - \hat{y}) - \beta/\sigma_\beta^2$
- Dimension can be moderate to high

```python
def logistic_regression_nuts(y, X, n_samples):
    n, p = X.shape
    
    def log_prob(beta):
        logits = X @ beta
        ll = np.sum(y * logits - np.log(1 + np.exp(logits)))
        prior = -0.5 * np.sum(beta**2) / 100  # Prior variance 100
        return ll + prior
    
    def grad_log_prob(beta):
        probs = 1 / (1 + np.exp(-X @ beta))
        grad_ll = X.T @ (y - probs)
        grad_prior = -beta / 100
        return grad_ll + grad_prior
    
    return nuts_sample(log_prob, grad_log_prob, np.zeros(p), n_samples)
```

### Gaussian Mixture Model

$$
p(x|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
$$

**Recommendation**: **Gibbs Sampling** (with latent assignments)

**Why**: Conditionals are tractable with data augmentation.

```python
def gmm_gibbs(X, K, n_samples):
    n, d = X.shape
    
    # Initialize
    z = np.random.randint(0, K, n)  # Assignments
    
    samples = []
    for _ in range(n_samples):
        # Count assignments
        N_k = np.bincount(z, minlength=K)
        
        # Sample pi | z
        pi = np.random.dirichlet(1 + N_k)
        
        # Sample mu_k, Sigma_k | z, X
        mu, Sigma = [], []
        for k in range(K):
            X_k = X[z == k]
            if len(X_k) > d:
                mu_k = X_k.mean(axis=0)  # Simplified; use proper posterior
                Sigma_k = np.cov(X_k.T)
            else:
                mu_k = np.zeros(d)
                Sigma_k = np.eye(d)
            mu.append(mu_k)
            Sigma.append(Sigma_k)
        
        # Sample z | pi, mu, Sigma, X
        for i in range(n):
            log_probs = np.log(pi) + np.array([
                multivariate_normal.logpdf(X[i], mu[k], Sigma[k]) 
                for k in range(K)
            ])
            probs = softmax(log_probs)
            z[i] = np.random.choice(K, p=probs)
        
        samples.append({'pi': pi, 'mu': mu, 'Sigma': Sigma, 'z': z.copy()})
    
    return samples
```

### Hierarchical Model

$$
y_{ij} \sim \mathcal{N}(\mu + \alpha_j, \sigma^2), \quad \alpha_j \sim \mathcal{N}(0, \tau^2)
$$

**Recommendation**: **HMC with non-centered parameterization** or **Gibbs**

**Why centered fails**: The "funnel" geometry causes problems.

```python
# CENTERED (problematic)
# alpha_j ~ N(0, tau^2)
# y_ij ~ N(mu + alpha_j, sigma^2)

# NON-CENTERED (better)
# eta_j ~ N(0, 1)
# alpha_j = tau * eta_j
# y_ij ~ N(mu + tau * eta_j, sigma^2)

def hierarchical_hmc_noncentered(y, groups, n_samples):
    """Non-centered parameterization for hierarchical model."""
    J = len(np.unique(groups))
    
    def log_prob(params):
        mu, log_sigma, log_tau, eta = unpack(params)
        sigma, tau = np.exp(log_sigma), np.exp(log_tau)
        alpha = tau * eta
        
        # Likelihood
        means = mu + alpha[groups]
        ll = norm.logpdf(y, means, sigma).sum()
        
        # Priors
        lp = norm.logpdf(eta, 0, 1).sum()  # Standard normal on eta
        lp += norm.logpdf(mu, 0, 10)
        lp += norm.logpdf(log_sigma, 0, 1)  # Log-normal prior
        lp += norm.logpdf(log_tau, 0, 1)
        
        return ll + lp
    
    return hmc_sample(log_prob, grad_log_prob, init_params, n_samples)
```

### High-Dimensional Sparse Regression

$$
y = X\beta + \epsilon, \quad \beta_j \sim (1-\pi)\delta_0 + \pi \mathcal{N}(0, \sigma_\beta^2)
$$

**Recommendation**: **Gibbs with spike-and-slab** or **Variational inference**

**Why**: Discrete inclusion indicators require Gibbs-style updates.

---

## Hybrid and Adaptive Strategies

### Metropolis-within-Gibbs

Combine Gibbs (for tractable conditionals) with MH (for others):

```python
def metropolis_within_gibbs(current, conditionals, mh_params):
    """
    conditionals: dict mapping param names to sampling functions
    mh_params: dict mapping param names to MH proposal params
    """
    for param in current.keys():
        if param in conditionals:
            # Gibbs step
            current[param] = conditionals[param](current)
        else:
            # Metropolis step
            current[param] = mh_step(current, param, mh_params[param])
    
    return current
```

### Block Updates

Group correlated parameters:

```python
def identify_blocks(correlation_matrix, threshold=0.5):
    """Identify blocks of correlated parameters."""
    import networkx as nx
    
    G = nx.Graph()
    d = correlation_matrix.shape[0]
    G.add_nodes_from(range(d))
    
    for i in range(d):
        for j in range(i+1, d):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)
    
    return list(nx.connected_components(G))
```

### Adaptive MCMC

Adapt proposal parameters during warmup:

```python
class AdaptiveMCMC:
    def __init__(self, log_prob, d, adapt_schedule=None):
        self.log_prob = log_prob
        self.d = d
        self.mu = np.zeros(d)
        self.Sigma = np.eye(d)
        self.n_adapt = 0
        
    def step(self, x, adapt=True):
        # Propose from adapted distribution
        x_prop = np.random.multivariate_normal(x, 2.4**2 / self.d * self.Sigma)
        
        # Accept/reject
        log_alpha = self.log_prob(x_prop) - self.log_prob(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
        
        # Adapt
        if adapt:
            self.n_adapt += 1
            self.update_moments(x)
        
        return x
    
    def update_moments(self, x):
        """Online update of mean and covariance."""
        n = self.n_adapt
        delta = x - self.mu
        self.mu += delta / n
        if n > 1:
            self.Sigma = (n-2)/(n-1) * self.Sigma + delta.reshape(-1,1) @ delta.reshape(1,-1) / n
```

---

## Diagnostics-Based Selection

### When to Switch Methods

**Switch from RWM to MALA/HMC if**:
- ESS/iteration < 0.01
- Mixing time > 10,000 iterations
- Trace plots show strong autocorrelation

**Switch from MALA to HMC if**:
- ESS/gradient < 0.1
- Optimal step size is very small
- Target is ill-conditioned

**Switch from HMC to other methods if**:
- Many divergent transitions
- Gradient computation dominates runtime
- Discrete parameters prevent gradient use

### Diagnostic Checklist

```python
def diagnose_chain(samples, target='hmc'):
    """Comprehensive chain diagnostics."""
    diagnostics = {}
    
    # Basic statistics
    diagnostics['mean'] = np.mean(samples, axis=0)
    diagnostics['std'] = np.std(samples, axis=0)
    
    # Effective sample size
    diagnostics['ess'] = compute_ess(samples)
    diagnostics['ess_per_sample'] = diagnostics['ess'] / len(samples)
    
    # R-hat (if multiple chains)
    # diagnostics['rhat'] = compute_rhat(chains)
    
    # Method-specific
    if target == 'hmc':
        diagnostics['target_accept'] = 0.65
    elif target == 'mala':
        diagnostics['target_accept'] = 0.574
    elif target == 'rwm':
        diagnostics['target_accept'] = 0.234
    
    # Recommendations
    if diagnostics['ess_per_sample'] < 0.01:
        diagnostics['recommendation'] = 'Consider switching to HMC/NUTS'
    elif diagnostics['ess_per_sample'] > 0.5:
        diagnostics['recommendation'] = 'Excellent mixing'
    else:
        diagnostics['recommendation'] = 'Acceptable, but room for improvement'
    
    return diagnostics
```

---

## Quick Reference Guide

### By Problem Type

| Problem Type | First Choice | Second Choice |
|--------------|--------------|---------------|
| Low-d, conjugate | Gibbs | RWM |
| Low-d, non-conjugate | MALA | RWM |
| High-d, smooth | HMC/NUTS | MALA |
| High-d, sparse | Gibbs + MH | Variational |
| Hierarchical | NUTS (non-centered) | Gibbs |
| Mixture model | Gibbs | EM + bootstrap |
| Discrete params | Gibbs/MH | — |
| Multimodal | Parallel tempering | Multiple chains |

### By Computational Constraint

| Constraint | Recommendation |
|------------|----------------|
| No gradients | RWM or Gibbs |
| Expensive gradients | RWM, Gibbs, or mini-batch |
| Limited memory | Single chain, thinning |
| Parallel hardware | Multiple chains, vectorize |
| Real-time | Pre-tuned HMC, short chains |

### By Diagnostic Outcome

| Diagnostic | Action |
|------------|--------|
| Low ESS | Increase samples, switch method |
| High R-hat | Run longer, check convergence |
| Many divergences | Reduce step size, reparameterize |
| Low acceptance | Reduce step size |
| High acceptance | Increase step size |

---

## Summary

**The meta-algorithm for method selection**:

1. **Characterize the problem**: dimension, differentiability, structure
2. **Start simple**: Try the simplest applicable method
3. **Diagnose**: Check ESS, R-hat, trace plots
4. **Iterate**: Switch methods or tune based on diagnostics
5. **Validate**: Compare multiple approaches if unsure

**When in doubt**: For continuous, differentiable targets in $d > 20$, **NUTS is the default choice**. It automatically tunes trajectory length and has become the standard in modern probabilistic programming (Stan, PyMC, NumPyro).

---

## Exercises

1. **Decision tree application**. For each of the following problems, use the decision tree to select a method: (a) 5-parameter nonlinear regression, (b) 500-dimensional Gaussian process, (c) 20-component mixture model, (d) Ising model on a 10×10 grid.

2. **Method comparison**. Implement RWM, Gibbs, MALA, and HMC for Bayesian linear regression. Compare ESS per second for $d \in \{10, 50, 100\}$.

3. **Diagnostic-driven tuning**. Start with a poorly-tuned HMC (wrong step size, wrong trajectory length). Use diagnostics to iteratively improve until ESS/iteration > 0.1.

4. **Hybrid sampler**. Design a Metropolis-within-Gibbs sampler for a model with both continuous and discrete parameters. Compare to a pure MH approach.

5. **Failure mode identification**. Intentionally create scenarios where each method fails: (a) RWM in high-d, (b) Gibbs with strong correlations, (c) MALA with discontinuities, (d) HMC with multimodality.

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
2. Carpenter, B., et al. (2017). "Stan: A Probabilistic Programming Language." *Journal of Statistical Software*.
3. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*.
4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.
5. Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.
