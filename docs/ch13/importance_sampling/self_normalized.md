# Self-Normalized Importance Sampling

## Overview

Self-normalized importance sampling (SNIS) is the workhorse method for Bayesian inference, where the posterior distribution is known only up to a normalizing constant. Unlike standard IS, SNIS handles unnormalized target densities by estimating both numerator and denominator using the same importance samples.

## The Unnormalized Target Problem

### Bayesian Posterior Structure

In Bayesian inference, the posterior distribution is:

$$
\pi(\theta) = p(\theta|y) = \frac{p(y|\theta) p(\theta)}{p(y)}
$$

where:
- $p(y|\theta)$: likelihood (known)
- $p(\theta)$: prior (known)
- $p(y) = \int p(y|\theta) p(\theta) d\theta$: marginal likelihood (typically intractable)

We can evaluate the **unnormalized posterior**:

$$
\gamma(\theta) = p(y|\theta) p(\theta)
$$

but not the normalizing constant $Z = p(y)$.

### The General Setting

More broadly, we have:

$$
\pi(\theta) = \frac{\gamma(\theta)}{Z}, \quad Z = \int \gamma(\theta) d\theta \text{ (unknown)}
$$

Standard IS requires evaluating $\pi(\theta)$ exactly, which is impossible when $Z$ is unknown.

## Self-Normalized Estimator

### Derivation

Write the expectation as a ratio:

$$
I = \mathbb{E}_\pi[h(\theta)] = \int h(\theta) \pi(\theta) d\theta = \frac{\int h(\theta) \gamma(\theta) d\theta}{\int \gamma(\theta) d\theta}
$$

Both integrals can be estimated via IS with unnormalized weights:

$$
\tilde{w}(\theta) = \frac{\gamma(\theta)}{q(\theta)}
$$

**Numerator estimate:**
$$
\widehat{\text{Num}} = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \tilde{w}(\theta_i)
$$

**Denominator estimate:**
$$
\widehat{\text{Den}} = \frac{1}{n} \sum_{i=1}^n \tilde{w}(\theta_i)
$$

### The SNIS Estimator

$$
\boxed{\hat{I}_{\text{SNIS}} = \frac{\sum_{i=1}^n h(\theta_i) \tilde{w}_i}{\sum_{i=1}^n \tilde{w}_i} = \sum_{i=1}^n h(\theta_i) \bar{w}_i}
$$

where the **normalized weights** are:

$$
\bar{w}_i = \frac{\tilde{w}_i}{\sum_{j=1}^n \tilde{w}_j}, \quad \sum_{i=1}^n \bar{w}_i = 1
$$

This is a **weighted average** of $h(\theta_i)$ with weights summing to one.

## Properties of SNIS

### Bias

Unlike standard IS, SNIS is **biased**:

$$
\mathbb{E}[\hat{I}_{\text{SNIS}}] \neq I \quad \text{for finite } n
$$

This is because the ratio of expectations does not equal the expectation of the ratio:

$$
\mathbb{E}\left[\frac{\hat{\text{Num}}}{\hat{\text{Den}}}\right] \neq \frac{\mathbb{E}[\hat{\text{Num}}]}{\mathbb{E}[\hat{\text{Den}}]}
$$

### Consistency

Despite the bias, SNIS is **consistent**:

$$
\hat{I}_{\text{SNIS}} \xrightarrow{a.s.} I \quad \text{as } n \to \infty
$$

This follows from:

$$
\frac{1}{n} \sum_{i=1}^n \tilde{w}_i \xrightarrow{a.s.} \mathbb{E}_q\left[\frac{\gamma(\theta)}{q(\theta)}\right] = \int \gamma(\theta) d\theta = Z
$$

### Bias Characterization

Using a Taylor expansion around the true value:

$$
\text{Bias}(\hat{I}_{\text{SNIS}}) = O(1/n)
$$

The bias vanishes at rate $1/n$, faster than the standard error which is $O(1/\sqrt{n})$.

### Variance (Approximate)

For large $n$, the variance is approximately:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{1}{n} \text{Var}_\pi\left[(h(\theta) - I) \cdot \frac{\pi(\theta)}{q(\theta)}\right]
$$

## Comparison: Standard IS vs SNIS

| Property | Standard IS | Self-Normalized IS |
|----------|-------------|-------------------|
| Requires normalized $\pi$ | Yes | No |
| Bias | Unbiased | Biased, $O(1/n)$ |
| Consistency | Yes | Yes |
| Weights sum to | Not necessarily 1 | Exactly 1 |
| Use case | Known normalizing constant | Bayesian posterior |
| Variance | Often higher | Often lower |

!!! info "SNIS Often Has Lower Variance"
    Surprisingly, SNIS often has **lower** variance than standard IS even when the normalizing constant is known. The normalization stabilizes the estimator.

## PyTorch Implementation

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def self_normalized_importance_sampling(h_function, unnormalized_log_target, 
                                        proposal_dist, n_samples, 
                                        return_diagnostics=False):
    """
    Self-normalized importance sampling for unnormalized targets.
    
    Parameters
    ----------
    h_function : callable
        Function h(θ) whose expectation we want
    unnormalized_log_target : callable
        Log of unnormalized target: log γ(θ) where π(θ) = γ(θ)/Z
    proposal_dist : torch.distributions.Distribution
        Proposal distribution q(θ)
    n_samples : int
        Number of samples
    return_diagnostics : bool
        Whether to return diagnostic information
        
    Returns
    -------
    estimate : torch.Tensor
        SNIS estimate of E_π[h(θ)]
    diagnostics : dict, optional
        Samples, weights, ESS, and other diagnostics
    
    Mathematical Foundation
    -----------------------
    Î_SNIS = Σᵢ h(θᵢ) w̄ᵢ
    
    where:
    - w̃ᵢ = γ(θᵢ)/q(θᵢ) (unnormalized weights)
    - w̄ᵢ = w̃ᵢ / Σⱼw̃ⱼ (normalized weights, sum to 1)
    - θᵢ ~ q(θ)
    """
    # Step 1: Sample from proposal
    samples = proposal_dist.sample((n_samples,))
    
    # Step 2: Compute log unnormalized weights
    # log w̃(θ) = log γ(θ) - log q(θ)
    log_gamma = unnormalized_log_target(samples)
    log_q = proposal_dist.log_prob(samples)
    log_unnorm_weights = log_gamma - log_q
    
    # Step 3: Normalize weights (log-sum-exp for numerical stability)
    log_sum_weights = torch.logsumexp(log_unnorm_weights, dim=0)
    log_norm_weights = log_unnorm_weights - log_sum_weights
    norm_weights = torch.exp(log_norm_weights)
    
    # Step 4: Evaluate function
    h_values = h_function(samples)
    
    # Step 5: Compute SNIS estimate: Σᵢ w̄ᵢ h(θᵢ)
    estimate = torch.sum(norm_weights * h_values)
    
    if return_diagnostics:
        # Effective Sample Size: ESS = 1/Σᵢw̄ᵢ²
        ess = 1.0 / torch.sum(norm_weights**2)
        
        # Unnormalized weights for analysis
        unnorm_weights = torch.exp(log_unnorm_weights)
        
        diagnostics = {
            'samples': samples,
            'unnorm_weights': unnorm_weights,
            'norm_weights': norm_weights,
            'h_values': h_values,
            'log_unnorm_weights': log_unnorm_weights,
            'ess': ess,
            'ess_ratio': ess / n_samples,
            'n_samples': n_samples,
            'estimate_normalizing_constant': torch.exp(log_sum_weights) / n_samples
        }
        return estimate, diagnostics
    
    return estimate


def compute_ess(weights):
    """
    Compute Effective Sample Size from normalized weights.
    
    ESS = 1 / Σᵢ wᵢ²
    
    Interpretation:
    - ESS ≈ n: weights nearly uniform (excellent)
    - ESS << n: few samples dominate (poor)
    """
    return 1.0 / torch.sum(weights**2)


def compute_ess_unnormalized(unnorm_weights):
    """
    Compute ESS from unnormalized weights.
    
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²
    """
    sum_w = torch.sum(unnorm_weights)
    sum_w_sq = torch.sum(unnorm_weights**2)
    return sum_w**2 / sum_w_sq


# Example: Bayesian inference for Normal mean
# Prior: θ ~ N(μ₀, τ₀²)
# Likelihood: y ~ N(θ, σ²) for each observation
# Posterior: θ|y ~ N(μₙ, τₙ²)

# Generate synthetic data
torch.manual_seed(42)
true_theta = 5.0
sigma = 1.0  # Known observation noise
n_obs = 20
data = torch.normal(true_theta, sigma, size=(n_obs,))

print(f"Data: n={n_obs}, sample mean={data.mean().item():.3f}")

# Prior parameters
mu_0 = 0.0
tau_0 = 2.0

# Analytical posterior parameters
precision_0 = 1.0 / tau_0**2
precision_n = precision_0 + n_obs / sigma**2
tau_n = 1.0 / (precision_n**0.5)
mu_n = (precision_0 * mu_0 + n_obs * data.mean() / sigma**2) / precision_n

print(f"\nPrior: N({mu_0}, {tau_0}²)")
print(f"Posterior (analytical): N({mu_n:.4f}, {tau_n:.4f}²)")

# Define unnormalized log-posterior
def unnormalized_log_posterior(theta):
    """
    log γ(θ) = log p(y|θ) + log p(θ)
             = -Σ(yᵢ-θ)²/2σ² - (θ-μ₀)²/2τ₀²
    """
    # Log-likelihood: -Σ(yᵢ-θ)²/2σ²
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)
    
    # Shape: (n_samples,) or (n_samples, 1)
    theta_expanded = theta.unsqueeze(-1) if theta.dim() == 1 else theta
    data_expanded = data.unsqueeze(0)
    
    log_likelihood = -0.5 * torch.sum((data_expanded - theta_expanded)**2, dim=-1) / sigma**2
    
    # Log-prior: -(θ-μ₀)²/2τ₀²
    log_prior = -0.5 * (theta - mu_0)**2 / tau_0**2
    
    return log_likelihood + log_prior

# Use prior as proposal
proposal = dist.Normal(mu_0, tau_0)

# SNIS estimation
n_samples = 5000

estimate, diagnostics = self_normalized_importance_sampling(
    h_function=lambda x: x,  # Posterior mean
    unnormalized_log_target=unnormalized_log_posterior,
    proposal_dist=proposal,
    n_samples=n_samples,
    return_diagnostics=True
)

print(f"\n{'='*60}")
print("Self-Normalized IS Results")
print(f"{'='*60}")
print(f"Posterior mean E[θ|y]:")
print(f"  True: {mu_n:.6f}")
print(f"  SNIS: {estimate.item():.6f}")
print(f"  Error: {abs(estimate.item() - mu_n):.6f}")
print(f"\nEffective Sample Size:")
print(f"  ESS: {diagnostics['ess'].item():.1f}")
print(f"  Efficiency: {diagnostics['ess_ratio'].item():.1%}")

# Estimate posterior variance
var_estimate, _ = self_normalized_importance_sampling(
    h_function=lambda x: (x - estimate.item())**2,
    unnormalized_log_target=unnormalized_log_posterior,
    proposal_dist=proposal,
    n_samples=n_samples
)

print(f"\nPosterior variance Var[θ|y]:")
print(f"  True: {tau_n**2:.6f}")
print(f"  SNIS: {var_estimate.item():.6f}")

# Estimate normalizing constant (marginal likelihood)
print(f"\nMarginal likelihood estimate: {diagnostics['estimate_normalizing_constant'].item():.4e}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Prior, proposal, and posterior
x = torch.linspace(-5, 10, 500)
ax = axes[0, 0]

prior = dist.Normal(mu_0, tau_0)
posterior = dist.Normal(mu_n, tau_n)

ax.plot(x.numpy(), prior.log_prob(x).exp().numpy(), 
        'g--', linewidth=2, label='Prior', alpha=0.7)
ax.plot(x.numpy(), posterior.log_prob(x).exp().numpy(), 
        'b-', linewidth=2, label='Posterior (true)')

# Weighted histogram of samples
samples = diagnostics['samples']
weights = diagnostics['norm_weights']
ax.hist(samples.numpy(), bins=50, density=True, weights=weights.numpy(),
        alpha=0.5, color='red', label='SNIS approximation')

ax.axvline(true_theta, color='black', linestyle=':', linewidth=2, 
           label=f'True θ = {true_theta}')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Bayesian Posterior Estimation via SNIS', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Weight distribution
ax = axes[0, 1]
ax.hist(weights.numpy() * n_samples, bins=50, density=True, 
        alpha=0.7, color='purple', edgecolor='black')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, 
           label='Uniform weight')
ax.set_xlabel('Normalized Weight × n', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Weight Distribution (ESS = {diagnostics["ess"].item():.1f}, '
             f'{diagnostics["ess_ratio"].item():.1%})', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: Cumulative weight
sorted_weights = torch.sort(weights, descending=True)[0]
cumsum_weights = torch.cumsum(sorted_weights, dim=0)

ax = axes[1, 0]
ax.plot(torch.arange(1, n_samples+1).numpy(), cumsum_weights.numpy(), 
        'b-', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='50% of weight')
ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label='90% of weight')
ax.set_xlabel('Number of Samples (sorted by weight)', fontsize=12)
ax.set_ylabel('Cumulative Weight', fontsize=12)
ax.set_title('Weight Concentration', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Find concentration
n_50 = (cumsum_weights < 0.5).sum().item() + 1
n_90 = (cumsum_weights < 0.9).sum().item() + 1
ax.text(0.95, 0.05, f'{n_50} samples = 50% weight\n{n_90} samples = 90% weight',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 4: Bias analysis
sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
n_reps = 200

biases = []
std_devs = []

for n in sample_sizes:
    estimates = []
    for _ in range(n_reps):
        est, _ = self_normalized_importance_sampling(
            h_function=lambda x: x,
            unnormalized_log_target=unnormalized_log_posterior,
            proposal_dist=proposal,
            n_samples=n
        )
        estimates.append(est.item())
    
    estimates = torch.tensor(estimates)
    biases.append((estimates.mean() - mu_n).item())
    std_devs.append(estimates.std().item())

ax = axes[1, 1]
ax.plot(sample_sizes, biases, 'bo-', linewidth=2, markersize=8, 
        label='Bias')
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Bias', fontsize=12)
ax.set_title('SNIS Bias Decreases as O(1/n)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('self_normalized_is.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Effective Sample Size (ESS)

### Definition

The **Effective Sample Size** quantifies the quality of importance samples:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^n \bar{w}_i^2}
$$

Equivalently, using unnormalized weights:

$$
\text{ESS} = \frac{\left(\sum_{i=1}^n \tilde{w}_i\right)^2}{\sum_{i=1}^n \tilde{w}_i^2}
$$

### Properties

| ESS Value | Interpretation |
|-----------|----------------|
| ESS ≈ n | Excellent: weights nearly uniform |
| ESS ≈ n/2 | Good: moderate weight variation |
| ESS ≈ n/10 | Acceptable: some weight concentration |
| ESS << n/10 | Poor: few samples dominate |
| ESS ≈ 1 | Degenerate: single sample dominates |

### Variance Relationship

Approximately:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{\text{Var}_\pi(h(\theta))}{\text{ESS}}
$$

Low ESS implies high variance — the effective number of independent samples is small.

### ESS as a Diagnostic

```python
def diagnose_weights(norm_weights, name=""):
    """
    Comprehensive weight diagnostics.
    """
    n = len(norm_weights)
    
    # ESS
    ess = 1.0 / torch.sum(norm_weights**2)
    
    # Coefficient of variation
    cv = torch.std(norm_weights) / torch.mean(norm_weights)
    
    # Max weight
    max_w = torch.max(norm_weights)
    
    # Weight concentration
    sorted_w = torch.sort(norm_weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    # Entropy
    entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-10))
    max_entropy = torch.log(torch.tensor(float(n)))
    
    print(f"\nWeight Diagnostics {name}")
    print("-" * 50)
    print(f"  n samples: {n}")
    print(f"  ESS: {ess.item():.1f} ({ess.item()/n:.1%} efficiency)")
    print(f"  CV of weights: {cv.item():.3f}")
    print(f"  Max weight: {max_w.item():.6f} (uniform = {1/n:.6f})")
    print(f"  {n_for_50} samples ({n_for_50/n:.1%}) account for 50% of weight")
    print(f"  {n_for_90} samples ({n_for_90/n:.1%}) account for 90% of weight")
    print(f"  Normalized entropy: {entropy.item()/max_entropy.item():.3f}")
    
    return {
        'ess': ess.item(),
        'cv': cv.item(),
        'max_weight': max_w.item(),
        'n_for_50': n_for_50,
        'n_for_90': n_for_90
    }

# Run diagnostics
diagnostics = diagnose_weights(diagnostics['norm_weights'], "(Prior as Proposal)")
```

## Proposal Distribution Comparison

### Impact of Proposal Choice

```python
# Compare different proposals
proposals = {
    'Prior N(0, 2)': dist.Normal(0.0, 2.0),
    'Close N(μₙ, 1)': dist.Normal(mu_n, 1.0),
    'Posterior (oracle)': dist.Normal(mu_n, tau_n),
    'Too narrow N(μₙ, 0.3)': dist.Normal(mu_n, 0.3),
    'Too wide N(0, 5)': dist.Normal(0.0, 5.0),
}

print("\nProposal Comparison")
print("=" * 70)
print(f"{'Proposal':<25} {'Estimate':>12} {'Error':>10} {'ESS':>10} {'Efficiency':>10}")
print("-" * 70)

for name, proposal in proposals.items():
    estimate, diag = self_normalized_importance_sampling(
        h_function=lambda x: x,
        unnormalized_log_target=unnormalized_log_posterior,
        proposal_dist=proposal,
        n_samples=5000,
        return_diagnostics=True
    )
    
    print(f"{name:<25} {estimate.item():12.4f} {abs(estimate.item()-mu_n):10.4f} "
          f"{diag['ess'].item():10.1f} {diag['ess_ratio'].item():10.1%}")
```

## Computing Multiple Expectations

### Sample Reusability

A key advantage of importance sampling is that the same weighted samples can estimate multiple expectations:

```python
def compute_multiple_expectations(samples, weights, functions):
    """
    Estimate multiple expectations from the same samples.
    
    E_π[hₖ(θ)] ≈ Σᵢ w̄ᵢ hₖ(θᵢ)  for each k
    """
    results = {}
    for name, h in functions.items():
        h_values = h(samples)
        estimate = torch.sum(weights * h_values)
        results[name] = estimate.item()
    return results

# Define multiple functions of interest
functions = {
    'E[θ]': lambda x: x,
    'E[θ²]': lambda x: x**2,
    'E[θ³]': lambda x: x**3,
    'Var[θ]': lambda x: (x - estimate.item())**2,  # Centered at posterior mean
    'P(θ > 3)': lambda x: (x > 3).float(),
}

results = compute_multiple_expectations(
    diagnostics['samples'], 
    diagnostics['norm_weights'],
    functions
)

print("\nMultiple Expectations from Same Samples")
print("-" * 40)
for name, value in results.items():
    print(f"  {name}: {value:.6f}")
```

## Posterior Predictive Distribution

### Definition

The posterior predictive distribution is:

$$
p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta
$$

### SNIS Estimation

```python
def posterior_predictive_pmf(y_values, samples, weights, likelihood):
    """
    Estimate posterior predictive P(ỹ|y) for discrete outcomes.
    """
    probs = []
    for y in y_values:
        # P(ỹ|y) ≈ Σᵢ w̄ᵢ P(ỹ|θᵢ)
        p_y_given_theta = likelihood(y, samples)
        prob = torch.sum(weights * p_y_given_theta)
        probs.append(prob.item())
    return probs
```

## Key Takeaways

!!! success "When to Use SNIS"
    - Posterior inference with unknown normalizing constant
    - Multiple expectations from the same samples
    - Posterior predictive calculations
    - Sequential importance sampling algorithms

!!! warning "Bias Considerations"
    - SNIS is biased for finite $n$
    - Bias = $O(1/n)$, decreases faster than standard error
    - For practical purposes, bias is usually negligible
    - Always monitor ESS to ensure adequate sample quality

!!! info "ESS Guidelines"
    - ESS > 1000: Usually sufficient for most applications
    - ESS/n > 0.1: Acceptable efficiency
    - ESS/n < 0.01: Consider improving proposal
    - Always report ESS alongside estimates

## Exercises

### Exercise 1: Bias Verification
Verify empirically that SNIS bias is $O(1/n)$. Use sample sizes $n = 100, 200, 400, 800, 1600$ with many replications. Plot bias vs $n$ on a log-log scale and confirm slope $\approx -1$.

### Exercise 2: ESS Interpretation
For a fixed proposal, draw samples of size $n = 1000, 2000, 5000, 10000$. Does ESS/n remain approximately constant? Explain why or why not.

### Exercise 3: Posterior Predictive
Implement posterior predictive estimation for the Normal-Normal model. Compare SNIS estimates to analytical results.

### Exercise 4: Multiple Proposals
Compare the efficiency of using the prior versus a Laplace approximation (Gaussian centered at the mode) as proposals. Which achieves higher ESS?

## References

1. Geweke, J. (1989). "Bayesian inference in econometric models using Monte Carlo integration." *Econometrica*, 57(6), 1317-1339.

2. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. Chapter 9.4: Self-Normalized Importance Sampling.

3. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. Chapter 3.3.

4. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

5. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.
