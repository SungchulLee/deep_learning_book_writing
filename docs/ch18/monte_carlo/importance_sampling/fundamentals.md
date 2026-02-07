# Importance Sampling Fundamentals

## Overview

Importance sampling is a variance reduction technique that enables computing expectations under one distribution by sampling from another. This method is fundamental to Bayesian inference, where we often need to integrate with respect to complex posterior distributions that are difficult or impossible to sample from directly.

## Historical Context

The importance sampling method emerged from Monte Carlo research at Los Alamos in the late 1940s. The term "importance sampling" and its modern formulation appeared around 1950 in work by Herman Kahn, T. E. Harris, and colleagues studying neutron transport problems. The method was systematized by Hammersley and Handscomb (1964) and later adopted in Bayesian statistics through work by Kloek and van Dijk (1978).

## The Fundamental Problem

### Setting

We want to compute an expectation under a target distribution $\pi(\theta)$:

$$
I = \mathbb{E}_{\pi}[h(\theta)] = \int h(\theta) \, \pi(\theta) \, d\theta
$$

**Challenge**: Direct sampling from $\pi(\theta)$ may be difficult or impossible.

### Common Scenarios

1. **Bayesian inference**: $\pi(\theta) = p(\theta|y)$ is the posterior, known only up to proportionality
2. **Rare event estimation**: Events with probability $< 10^{-6}$ under $\pi$
3. **Complex densities**: No standard sampling algorithm exists

## The Change of Measure

### Key Identity

Introduce a proposal distribution $q(\theta)$ and multiply by $1 = \frac{q(\theta)}{q(\theta)}$:

$$
I = \int h(\theta) \, \pi(\theta) \, d\theta = \int h(\theta) \frac{\pi(\theta)}{q(\theta)} \, q(\theta) \, d\theta
$$

Define the **importance weight**:

$$
w(\theta) = \frac{\pi(\theta)}{q(\theta)}
$$

Then:

$$
I = \int h(\theta) \, w(\theta) \, q(\theta) \, d\theta = \mathbb{E}_q[h(\theta) \cdot w(\theta)]
$$

### Support Condition

!!! danger "Critical Requirement"
    The proposal $q$ must dominate the target $\pi$:
    
    $$\pi(\theta) > 0 \implies q(\theta) > 0$$
    
    If $q(\theta) = 0$ where $\pi(\theta) > 0$, the weight $w(\theta)$ is undefined, leading to **infinite bias**.

## The Importance Sampling Estimator

### Monte Carlo Approximation

Draw samples $\theta_1, \ldots, \theta_n \stackrel{\text{i.i.d.}}{\sim} q(\theta)$ and estimate:

$$
\hat{I}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \, w(\theta_i) = \frac{1}{n} \sum_{i=1}^n h(\theta_i) \frac{\pi(\theta_i)}{q(\theta_i)}
$$

### Properties

**Unbiasedness:**

$$
\mathbb{E}_q[\hat{I}_{\text{IS}}] = \mathbb{E}_q\left[\frac{1}{n} \sum_{i=1}^n h(\theta_i) w(\theta_i)\right] = \mathbb{E}_q[h(\theta) w(\theta)] = I
$$

**Variance:**

$$
\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n} \text{Var}_q(h(\theta) w(\theta)) = \frac{1}{n}\left(\mathbb{E}_q[h^2(\theta) w^2(\theta)] - I^2\right)
$$

## Intuition: Sample Where It Matters

### The Core Idea

Under naive Monte Carlo from $\pi$:

- Many samples fall in regions where $|h(\theta)|$ is small → contribute little
- Few samples land where $|h(\theta)|$ is large → high variance

Under importance sampling from $q$:

- **Oversample** important regions where $|h(\theta)\pi(\theta)|$ is large
- **Correct** for oversampling with weights $w(\theta) = \pi(\theta)/q(\theta)$

The slogan: **"Sample where it matters, correct with weights."**

### Visual Intuition

```
Target π(θ):        [.....|XXXXXX|.....]
                          ↑ high density

Naive MC:           draws mostly from middle, few from tails

Important region:   [.....|..XXXX|.....]
(for some h)              ↑ where |h·π| is large

Good proposal q:    [.....|..XXXX|.....] 
                    puts extra mass where it matters
```

## Variance Analysis

### General Variance Formula

$$
\text{Var}_q(h(\theta) w(\theta)) = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta - I^2
$$

### The Second Moment

Define:
$$
J(q) = \mathbb{E}_q[h^2(\theta) w^2(\theta)] = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta
$$

Then: $\text{Var}_q(\hat{I}_{\text{IS}}) = \frac{1}{n}(J(q) - I^2)$

### When IS Reduces Variance

IS reduces variance when:
$$
\mathbb{E}_q[h^2(\theta) w^2(\theta)] < \mathbb{E}_\pi[h^2(\theta)]
$$

This happens when $q$ concentrates samples where $|h(\theta)\pi(\theta)|$ is large.

### When IS Increases Variance

IS *increases* variance when $q$ is poorly chosen:

- $q$ too narrow: misses important regions → some weights explode
- $q$ shifted away from $\pi$: most weights near zero, few very large

!!! warning "Weight Degeneracy"
    If $q$ is badly matched to $\pi$:
    
    - A few samples have enormous weights
    - Most samples have negligible weights
    - Effective sample size collapses
    - Variance explodes

## Optimal Proposal Distribution

### Derivation via Calculus of Variations

We minimize $J(q) = \int h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} d\theta$ subject to $\int q(\theta) d\theta = 1$.

Using Lagrange multipliers, the optimal $q^*$ satisfies:

$$
\frac{\partial}{\partial q}\left[h^2(\theta) \frac{\pi^2(\theta)}{q(\theta)} + \lambda q(\theta)\right] = 0
$$

$$
-h^2(\theta) \frac{\pi^2(\theta)}{q^2(\theta)} + \lambda = 0 \implies q(\theta) \propto |h(\theta)| \pi(\theta)
$$

### The Optimal Proposal

$$
\boxed{q^*(\theta) = \frac{|h(\theta)| \pi(\theta)}{\int |h(\theta')| \pi(\theta') d\theta'}}
$$

**Why absolute value?** The proposal must be non-negative. If $h$ can be negative, we sample proportional to $|h| \pi$ and the sign is carried by $h$ in the estimator.

### Variance at Optimum

For $h(\theta) \geq 0$:

$$
q^*(\theta) = \frac{h(\theta) \pi(\theta)}{I}
$$

Then:
$$
\text{Var}_{q^*}(\hat{I}_{\text{IS}}) = \frac{1}{n}(I^2 - I^2) = 0
$$

**The optimal proposal achieves zero variance** — we're sampling from the integrand itself!

### Practical Implications

The optimal $q^*$ depends on the unknown $I$, so we can't use it directly. However, this derivation reveals what a good proposal should look like:

1. **Shape**: Follow $|h(\theta)|\pi(\theta)$
2. **Support**: Cover everywhere $\pi(\theta) > 0$
3. **Tails**: Should be at least as heavy as $\pi$

## PyTorch Implementation

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def importance_sampling(h_function, target_log_pdf, proposal_dist, 
                        n_samples, return_diagnostics=False):
    """
    Standard importance sampling with known normalizing constant.
    
    Parameters
    ----------
    h_function : callable
        Function h(θ) whose expectation we want to compute
    target_log_pdf : callable  
        Log density of target π(θ)
    proposal_dist : torch.distributions.Distribution
        Proposal distribution q(θ)
    n_samples : int
        Number of samples to draw
    return_diagnostics : bool
        Whether to return diagnostic information
        
    Returns
    -------
    estimate : torch.Tensor
        Importance sampling estimate of E_π[h(θ)]
    se : torch.Tensor
        Estimated standard error
    diagnostics : dict, optional
        Samples, weights, and other diagnostics
    
    Mathematical Foundation
    -----------------------
    Î_IS = (1/n) Σᵢ h(θᵢ) w(θᵢ)
    
    where w(θ) = π(θ)/q(θ) and θᵢ ~ q(θ)
    """
    # Step 1: Draw samples from proposal q(θ)
    samples = proposal_dist.sample((n_samples,))
    
    # Step 2: Evaluate log densities
    log_target = target_log_pdf(samples)
    log_proposal = proposal_dist.log_prob(samples)
    
    # Step 3: Compute importance weights (in log space for stability)
    # w(θ) = π(θ)/q(θ)  →  log w(θ) = log π(θ) - log q(θ)
    log_weights = log_target - log_proposal
    weights = torch.exp(log_weights)
    
    # Step 4: Evaluate function h at sample points
    h_values = h_function(samples)
    
    # Step 5: Compute IS estimate: (1/n) Σᵢ h(θᵢ) w(θᵢ)
    weighted_h = h_values * weights
    estimate = torch.mean(weighted_h)
    
    # Step 6: Estimate standard error
    variance = torch.var(weighted_h, unbiased=True)
    se = torch.sqrt(variance / n_samples)
    
    if return_diagnostics:
        diagnostics = {
            'samples': samples,
            'weights': weights,
            'h_values': h_values,
            'log_weights': log_weights,
            'weighted_h': weighted_h,
            'n_samples': n_samples
        }
        return estimate, se, diagnostics
    
    return estimate, se


# Example: Compute E_π[θ²] where π = N(3, 1)
# Using proposal q = N(0, 2)

# True value: E[θ²] = μ² + σ² = 9 + 1 = 10
true_value = 10.0

# Define target and proposal
target_mean, target_std = 3.0, 1.0
proposal_mean, proposal_std = 0.0, 2.0

# Target log-density (normalized)
def target_log_pdf(theta):
    return dist.Normal(target_mean, target_std).log_prob(theta)

# Proposal distribution
proposal = dist.Normal(proposal_mean, proposal_std)

# Function of interest
h = lambda theta: theta**2

# Run importance sampling
estimate, se, diagnostics = importance_sampling(
    h, target_log_pdf, proposal, 
    n_samples=10000, return_diagnostics=True
)

print("Importance Sampling: E_π[θ²] where π = N(3,1), q = N(0,2)")
print("=" * 60)
print(f"True value: {true_value:.6f}")
print(f"IS estimate: {estimate.item():.6f}")
print(f"Standard error: {se.item():.6f}")
print(f"Error: {abs(estimate.item() - true_value):.6f}")

# Weight diagnostics
weights = diagnostics['weights']
print(f"\nWeight Statistics:")
print(f"  Mean: {weights.mean().item():.4f}")
print(f"  Std: {weights.std().item():.4f}")
print(f"  Max: {weights.max().item():.4f}")
print(f"  Min: {weights.min().item():.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Target and proposal distributions
x = torch.linspace(-5, 8, 500)
ax = axes[0, 0]
ax.plot(x.numpy(), torch.exp(target_log_pdf(x)).numpy(), 
        'b-', linewidth=2, label=f'Target π = N({target_mean},{target_std})')
ax.plot(x.numpy(), proposal.log_prob(x).exp().numpy(), 
        'r--', linewidth=2, label=f'Proposal q = N({proposal_mean},{proposal_std})')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Target vs Proposal Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Weight distribution
ax = axes[0, 1]
ax.hist(weights.numpy(), bins=50, density=True, alpha=0.7, 
        color='purple', edgecolor='black')
ax.axvline(weights.mean().item(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean = {weights.mean().item():.3f}')
ax.set_xlabel('Importance Weight w(θ)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Importance Weights', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 3: Samples colored by weight
samples = diagnostics['samples']
ax = axes[1, 0]
scatter = ax.scatter(samples.numpy(), h(samples).numpy(), 
                     c=weights.numpy(), cmap='hot', alpha=0.6, 
                     s=30, edgecolors='black', linewidth=0.3)
ax.set_xlabel('Sample θ', fontsize=12)
ax.set_ylabel('h(θ) = θ²', fontsize=12)
ax.set_title('Samples Colored by Importance Weight', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight')
ax.grid(True, alpha=0.3)

# Panel 4: Convergence
n_values = torch.arange(100, 10001, 100)
estimates = []
for n in n_values:
    est, _ = importance_sampling(h, target_log_pdf, proposal, int(n))
    estimates.append(est.item())

ax = axes[1, 1]
ax.plot(n_values.numpy(), estimates, 'b-', linewidth=1.5, alpha=0.7)
ax.axhline(true_value, color='red', linestyle='--', linewidth=2, 
           label='True Value')
ax.fill_between(n_values.numpy(), true_value - 0.2, true_value + 0.2,
                alpha=0.2, color='red', label='±0.2 band')
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('IS Estimate', fontsize=12)
ax.set_title('Convergence of IS Estimate', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('importance_sampling_fundamentals.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Comparing Proposal Distributions

### Effect of Proposal Choice

```python
def compare_proposals(h_function, target_log_pdf, proposals, 
                      n_samples=5000, n_replications=100):
    """
    Compare multiple proposal distributions.
    """
    results = []
    
    for name, proposal in proposals.items():
        estimates = []
        ess_values = []
        
        for _ in range(n_replications):
            est, se, diag = importance_sampling(
                h_function, target_log_pdf, proposal, 
                n_samples, return_diagnostics=True
            )
            estimates.append(est.item())
            
            # Compute ESS
            w = diag['weights']
            w_normalized = w / w.sum()
            ess = 1.0 / (w_normalized**2).sum()
            ess_values.append(ess.item())
        
        results.append({
            'name': name,
            'mean': torch.tensor(estimates).mean().item(),
            'std': torch.tensor(estimates).std().item(),
            'mean_ess': torch.tensor(ess_values).mean().item(),
            'ess_ratio': torch.tensor(ess_values).mean().item() / n_samples
        })
    
    return results

# Define proposals with varying quality
proposals = {
    'Good: N(3, 1.2)': dist.Normal(3.0, 1.2),
    'Okay: N(2, 1.5)': dist.Normal(2.0, 1.5),
    'Poor: N(0, 2)': dist.Normal(0.0, 2.0),
    'Bad: N(3, 0.5)': dist.Normal(3.0, 0.5),  # Too narrow
}

results = compare_proposals(h, target_log_pdf, proposals)

print("\nProposal Comparison:")
print("-" * 70)
print(f"{'Proposal':<20} {'Mean Est':>12} {'Std Dev':>12} {'ESS':>12} {'ESS/n':>12}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<20} {r['mean']:12.4f} {r['std']:12.4f} "
          f"{r['mean_ess']:12.1f} {r['ess_ratio']:12.2%}")
```

## Variance Reduction Example: Tail Probabilities

### The Rare Event Problem

Estimate $\mathbb{P}_\pi(X > 4)$ where $X \sim \mathcal{N}(0, 1)$.

```python
# True probability
true_prob = 1 - dist.Normal(0, 1).cdf(torch.tensor(4.0))
print(f"True P(X > 4): {true_prob.item():.2e}")

# Indicator function
h_indicator = lambda x: (x > 4).float()

# Naive Monte Carlo from target
target = dist.Normal(0, 1)
n_samples = 100000

# Naive MC
samples_naive = target.sample((n_samples,))
naive_estimate = h_indicator(samples_naive).mean()
naive_se = h_indicator(samples_naive).std() / (n_samples**0.5)

print(f"\nNaive MC (n={n_samples}):")
print(f"  Estimate: {naive_estimate.item():.6f}")
print(f"  SE: {naive_se.item():.6f}")

# Importance sampling with shifted proposal
shifted_proposal = dist.Normal(4.0, 1.5)  # Centered in tail

def target_log_pdf_std(x):
    return dist.Normal(0, 1).log_prob(x)

is_estimate, is_se, is_diag = importance_sampling(
    h_indicator, target_log_pdf_std, shifted_proposal,
    n_samples=10000, return_diagnostics=True
)

print(f"\nImportance Sampling (n=10000):")
print(f"  Estimate: {is_estimate.item():.6f}")
print(f"  SE: {is_se.item():.6f}")

# Variance reduction factor
variance_reduction = (naive_se / is_se)**2
print(f"\nVariance reduction factor: {variance_reduction.item():.1f}x")
```

## Application to Quantitative Finance

### Rare Event Risk Estimation

Importance sampling is widely used in quantitative finance for estimating tail risk measures. Computing Value-at-Risk (VaR) and Expected Shortfall (ES) at extreme quantiles (e.g., 99.9%) requires efficient estimation of rare loss events — a natural application of the IS framework developed above.

```python
def estimate_var_es_importance_sampling(
    loss_function, target_dist, proposal_dist, 
    alpha=0.999, n_samples=10000
):
    """
    Estimate VaR and Expected Shortfall using importance sampling.
    
    Parameters
    ----------
    loss_function : callable
        Maps risk factors to portfolio loss
    target_dist : torch.distributions.Distribution
        Distribution of risk factors under P
    proposal_dist : torch.distributions.Distribution
        Proposal biased toward tail
    alpha : float
        Confidence level (e.g., 0.999)
    n_samples : int
        Number of IS samples
    """
    # Sample from proposal
    samples = proposal_dist.sample((n_samples,))
    
    # Compute importance weights
    log_weights = target_dist.log_prob(samples) - proposal_dist.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
    
    # Compute losses
    losses = loss_function(samples)
    
    # Sort by loss for weighted quantile estimation
    sorted_indices = torch.argsort(losses)
    sorted_losses = losses[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Weighted CDF for VaR
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    var_idx = (cumulative_weights >= alpha).nonzero(as_tuple=True)[0][0]
    var_estimate = sorted_losses[var_idx]
    
    # Expected Shortfall: E[L | L > VaR]
    tail_mask = losses > var_estimate
    if tail_mask.sum() > 0:
        tail_weights = weights[tail_mask]
        tail_weights = tail_weights / tail_weights.sum()
        es_estimate = torch.sum(tail_weights * losses[tail_mask])
    else:
        es_estimate = var_estimate
    
    # ESS diagnostic
    ess = 1.0 / torch.sum(weights**2)
    
    return {
        'var': var_estimate.item(),
        'es': es_estimate.item(),
        'ess': ess.item(),
        'ess_ratio': (ess / n_samples).item()
    }

# Example: Heavy-tailed portfolio loss
torch.manual_seed(42)

# Risk factor distribution (normal market conditions)
target = dist.Normal(0.0, 1.0)

# Proposal shifted toward loss tail
proposal = dist.Normal(3.0, 1.5)

# Simple loss function: L = exp(0.5 * X) - 1
loss_fn = lambda x: torch.exp(0.5 * x) - 1.0

results = estimate_var_es_importance_sampling(
    loss_fn, target, proposal, alpha=0.999, n_samples=50000
)

print("Tail Risk Estimation via Importance Sampling")
print("=" * 50)
print(f"  VaR(99.9%): {results['var']:.4f}")
print(f"  ES(99.9%):  {results['es']:.4f}")
print(f"  ESS:        {results['ess']:.1f} ({results['ess_ratio']:.1%})")
```

## Application to Bayesian Inference

### Posterior Expectations

In Bayesian inference, we want expectations under the posterior:

$$
\mathbb{E}[h(\theta)|y] = \int h(\theta) \, p(\theta|y) \, d\theta
$$

where $p(\theta|y) \propto p(y|\theta) p(\theta)$.

The posterior is typically known only up to proportionality, which motivates **self-normalized importance sampling** (covered in [Self-Normalized IS](self_normalized.md)).

### Connection to Model Evidence

Importance sampling can also estimate the marginal likelihood:

$$
p(y) = \int p(y|\theta) p(\theta) d\theta
$$

Using the prior as proposal: $q(\theta) = p(\theta)$

$$
\hat{p}(y) = \frac{1}{n} \sum_{i=1}^n p(y|\theta_i), \quad \theta_i \sim p(\theta)
$$

This is the harmonic mean estimator (though it can have infinite variance—see advanced topics).

## Key Takeaways

!!! success "When to Use Importance Sampling"
    - Target distribution is difficult to sample from
    - A good proposal distribution is available
    - Variance reduction is needed for rare events
    - Reusing samples for multiple expectations

!!! warning "When IS May Fail"
    - High dimensions without careful proposal design
    - Target has heavier tails than proposal
    - Multimodal targets with single-component proposals
    - Very large mismatch between target and proposal

!!! info "Best Practices"
    1. Always check weight diagnostics (variance, max weight, ESS)
    2. Proposal should have heavier tails than target
    3. Cover the full support of the target
    4. Consider adaptive methods for complex targets

## Exercises

### Exercise 1: Variance Comparison
Compare the variance of naive MC and IS for estimating $\mathbb{E}[e^{2X}]$ where $X \sim \mathcal{N}(0,1)$. Use proposals $q = \mathcal{N}(0,1)$, $q = \mathcal{N}(1,1)$, and $q = \mathcal{N}(2,1)$. Which achieves the lowest variance and why?

### Exercise 2: Support Coverage
Demonstrate what happens when the proposal doesn't cover the target's support. Let $\pi = \mathcal{N}(0,1)$ and $q = \text{Uniform}(-2, 2)$. Estimate $\mathbb{E}[X^2]$ and explain the bias.

### Exercise 3: Optimal Proposal Approximation
For $h(\theta) = \theta^2$ and $\pi = \mathcal{N}(3,1)$, the optimal proposal is $q^* \propto \theta^2 \cdot \mathcal{N}(3,1)$. Approximate this by fitting a Gaussian to $|\theta| \cdot \mathcal{N}(3,1)$ and compare the variance to using the prior as proposal.

### Exercise 4: Option Pricing via IS
Use importance sampling to price a deep out-of-the-money European call option under the Black-Scholes model. Compare the variance of naive MC (sampling paths under the risk-neutral measure) versus IS with the proposal shifted toward the strike price. Compute the variance reduction factor.

## References

1. Kahn, H., & Harris, T. E. (1951). "Estimation of particle transmission by random sampling." *National Bureau of Standards Applied Mathematics Series*, 12, 27-30.

2. Hammersley, J. M., & Handscomb, D. C. (1964). *Monte Carlo Methods*. Methuen.

3. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. Chapter 9: Importance Sampling.

4. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. Chapter 3.

5. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. Chapter 2.

6. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. Chapters 4-5.
