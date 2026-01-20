# Effective Sample Size

## Overview

The Effective Sample Size (ESS) is the fundamental diagnostic for importance sampling quality. It quantifies how many independent samples from the target distribution our weighted samples are equivalent to. ESS provides an objective measure of proposal quality and directly relates to the variance of importance sampling estimators.

## Mathematical Definition

### Definition for Normalized Weights

Given normalized importance weights $\bar{w}_1, \ldots, \bar{w}_n$ with $\sum_i \bar{w}_i = 1$:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^n \bar{w}_i^2}
$$

### Definition for Unnormalized Weights

For unnormalized weights $\tilde{w}_1, \ldots, \tilde{w}_n$:

$$
\text{ESS} = \frac{\left(\sum_{i=1}^n \tilde{w}_i\right)^2}{\sum_{i=1}^n \tilde{w}_i^2}
$$

Both definitions are equivalent: $\bar{w}_i = \tilde{w}_i / \sum_j \tilde{w}_j$.

### Properties

**Bounds:**
$$
1 \leq \text{ESS} \leq n
$$

**Extreme Cases:**
- **Maximum**: ESS $= n$ when all weights are equal: $\bar{w}_i = 1/n$
- **Minimum**: ESS $= 1$ when one weight equals 1 and rest equal 0

## Interpretation

### Intuitive Understanding

ESS answers: "How many independent samples from $\pi$ are our $n$ weighted samples equivalent to?"

| ESS | Interpretation | Implication |
|-----|----------------|-------------|
| $\text{ESS} = n$ | Perfect sampling | Weights are uniform, as if sampling from $\pi$ |
| $\text{ESS} = n/2$ | 50% efficiency | Half the samples are "wasted" |
| $\text{ESS} = n/10$ | 10% efficiency | Need 10× samples for same precision |
| $\text{ESS} \ll n$ | Severe degeneracy | Few samples dominate, estimates unreliable |
| $\text{ESS} \approx 1$ | Complete failure | Essentially one sample, useless |

### Connection to Variance

The variance of a self-normalized IS estimator is approximately:

$$
\text{Var}(\hat{I}_{\text{SNIS}}) \approx \frac{\text{Var}_\pi(h(\theta))}{\text{ESS}}
$$

Compare to standard MC with $n$ independent samples:

$$
\text{Var}(\hat{I}_{\text{MC}}) = \frac{\text{Var}_\pi(h(\theta))}{n}
$$

**Variance Inflation Factor:**

$$
\text{Variance Inflation} = \frac{n}{\text{ESS}}
$$

If ESS $= n/10$, the variance is inflated by 10× compared to perfect sampling.

## Derivation of ESS

### From Weight Variance

Start with the coefficient of variation of weights:

$$
\text{CV}^2(\tilde{w}) = \frac{\text{Var}(\tilde{w})}{[\mathbb{E}(\tilde{w})]^2}
$$

The ESS can be written as:

$$
\text{ESS} = \frac{n}{1 + \text{CV}^2(\tilde{w})}
$$

**Proof:**

$$
\text{ESS} = \frac{(\sum_i \tilde{w}_i)^2}{\sum_i \tilde{w}_i^2} = \frac{n^2 \bar{\tilde{w}}^2}{n \cdot (\text{Var}(\tilde{w}) + \bar{\tilde{w}}^2)}
$$

where $\bar{\tilde{w}} = \frac{1}{n}\sum_i \tilde{w}_i$. Simplifying:

$$
\text{ESS} = \frac{n}{1 + \text{Var}(\tilde{w})/\bar{\tilde{w}}^2} = \frac{n}{1 + \text{CV}^2}
$$

### From Entropy

The ESS is related to the **perplexity** (exponential of entropy) of the weight distribution:

$$
\text{Perplexity} = \exp(H(\bar{w})) = \exp\left(-\sum_i \bar{w}_i \log \bar{w}_i\right)
$$

For a uniform distribution: $H = \log n$, Perplexity $= n$.

ESS and perplexity are related but not identical:
- ESS uses the $L_2$ norm: $1/\|\bar{w}\|_2^2$
- Perplexity uses entropy (related to $L_1$ of log weights)

Both measure weight concentration but with different emphases.

## PyTorch Implementation

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

def compute_ess_normalized(weights):
    """
    Compute ESS from normalized weights.
    
    ESS = 1 / Σᵢ wᵢ²
    
    Parameters
    ----------
    weights : torch.Tensor
        Normalized weights (sum to 1)
        
    Returns
    -------
    ess : float
        Effective sample size
    """
    return 1.0 / torch.sum(weights**2)


def compute_ess_unnormalized(unnorm_weights):
    """
    Compute ESS from unnormalized weights.
    
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²
    
    More numerically stable than normalizing first.
    """
    sum_w = torch.sum(unnorm_weights)
    sum_w_sq = torch.sum(unnorm_weights**2)
    return sum_w**2 / sum_w_sq


def compute_ess_log_weights(log_weights):
    """
    Compute ESS from log weights (most numerically stable).
    
    Useful when weights span many orders of magnitude.
    """
    # Normalize in log space
    log_sum = torch.logsumexp(log_weights, dim=0)
    log_norm_weights = log_weights - log_sum
    
    # ESS = exp(-log(Σᵢ exp(2 log wᵢ)))
    log_sum_sq = torch.logsumexp(2 * log_norm_weights, dim=0)
    log_ess = -log_sum_sq
    
    return torch.exp(log_ess)


def weight_diagnostics(weights, n_samples=None, name=""):
    """
    Comprehensive weight and ESS diagnostics.
    
    Parameters
    ----------
    weights : torch.Tensor
        Importance weights (normalized or unnormalized)
    n_samples : int, optional
        Total number of samples (if None, inferred from weights)
    name : str
        Label for printing
        
    Returns
    -------
    dict
        Dictionary of diagnostic statistics
    """
    if n_samples is None:
        n_samples = len(weights)
    
    # Normalize if needed
    if not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        norm_weights = weights / weights.sum()
    else:
        norm_weights = weights
    
    # ESS
    ess = compute_ess_normalized(norm_weights)
    ess_ratio = ess / n_samples
    
    # Weight statistics
    max_weight = norm_weights.max()
    min_weight = norm_weights.min()
    uniform_weight = 1.0 / n_samples
    
    # Coefficient of variation
    cv = norm_weights.std() / norm_weights.mean()
    
    # Weight concentration
    sorted_weights = torch.sort(norm_weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    
    n_for_10 = (cumsum < 0.1).sum().item() + 1
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    # Entropy and perplexity
    entropy = -torch.sum(norm_weights * torch.log(norm_weights + 1e-10))
    max_entropy = torch.log(torch.tensor(float(n_samples)))
    perplexity = torch.exp(entropy)
    
    # Variance inflation
    variance_inflation = n_samples / ess
    
    diagnostics = {
        'n_samples': n_samples,
        'ess': ess.item(),
        'ess_ratio': ess_ratio.item(),
        'variance_inflation': variance_inflation.item(),
        'cv': cv.item(),
        'max_weight': max_weight.item(),
        'max_weight_ratio': max_weight.item() / uniform_weight,
        'n_for_10_pct': n_for_10,
        'n_for_50_pct': n_for_50,
        'n_for_90_pct': n_for_90,
        'entropy': entropy.item(),
        'normalized_entropy': (entropy / max_entropy).item(),
        'perplexity': perplexity.item()
    }
    
    if name:
        print(f"\n{'='*60}")
        print(f"Weight Diagnostics: {name}")
        print(f"{'='*60}")
        print(f"  Total samples: {n_samples}")
        print(f"  ESS: {ess.item():.1f} ({ess_ratio.item():.1%} efficiency)")
        print(f"  Variance inflation: {variance_inflation.item():.1f}x")
        print(f"  CV of weights: {cv.item():.3f}")
        print(f"  Max weight: {max_weight.item():.6f} ({diagnostics['max_weight_ratio']:.1f}x uniform)")
        print(f"\n  Weight Concentration:")
        print(f"    10% weight in top {n_for_10} samples ({n_for_10/n_samples:.1%})")
        print(f"    50% weight in top {n_for_50} samples ({n_for_50/n_samples:.1%})")
        print(f"    90% weight in top {n_for_90} samples ({n_for_90/n_samples:.1%})")
        print(f"\n  Entropy: {entropy.item():.3f} (normalized: {diagnostics['normalized_entropy']:.3f})")
        print(f"  Perplexity: {perplexity.item():.1f}")
        
        # Quality assessment
        if ess_ratio.item() > 0.5:
            quality = "EXCELLENT"
        elif ess_ratio.item() > 0.2:
            quality = "GOOD"
        elif ess_ratio.item() > 0.05:
            quality = "ACCEPTABLE"
        elif ess_ratio.item() > 0.01:
            quality = "POOR"
        else:
            quality = "FAILURE"
        
        print(f"\n  Overall Quality: {quality}")
    
    return diagnostics


# Example: ESS for different proposal qualities
torch.manual_seed(42)

# Target: N(5, 1)
target = dist.Normal(5.0, 1.0)

# Various proposals
proposals = {
    'Perfect: N(5, 1)': dist.Normal(5.0, 1.0),
    'Good: N(5, 1.2)': dist.Normal(5.0, 1.2),
    'Decent: N(4.5, 1.5)': dist.Normal(4.5, 1.5),
    'Poor: N(3, 2)': dist.Normal(3.0, 2.0),
    'Bad: N(5, 0.5)': dist.Normal(5.0, 0.5),  # Too narrow
    'Terrible: N(0, 1)': dist.Normal(0.0, 1.0),  # Wrong location
}

n_samples = 5000

print("ESS Comparison for Different Proposals")
print("Target: N(5, 1)")
print("=" * 70)
print(f"{'Proposal':<25} {'ESS':>10} {'ESS/n':>10} {'CV':>10} {'Max/Uniform':>12}")
print("-" * 70)

results = {}
for name, proposal in proposals.items():
    # Sample and compute weights
    samples = proposal.sample((n_samples,))
    log_weights = target.log_prob(samples) - proposal.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
    
    # Diagnostics
    diag = weight_diagnostics(weights, n_samples)
    results[name] = diag
    
    print(f"{name:<25} {diag['ess']:10.1f} {diag['ess_ratio']:10.1%} "
          f"{diag['cv']:10.2f} {diag['max_weight_ratio']:12.1f}x")
```

## ESS and Sample Size Scaling

### Does ESS Scale with n?

**Question**: If we double $n$, does ESS double?

**Answer**: Yes, approximately. The ESS ratio (ESS/n) remains roughly constant for a fixed proposal-target pair.

**Proof sketch:**

$$
\text{ESS} = \frac{n}{1 + \text{CV}^2(\tilde{w})}
$$

For large $n$, $\text{CV}^2(\tilde{w})$ converges to a constant (by LLN), so:

$$
\frac{\text{ESS}}{n} \to \frac{1}{1 + \text{CV}^2_\infty}
$$

**Implication**: You cannot "fix" a bad proposal by simply increasing $n$. The efficiency (ESS/n) is determined by the proposal-target match.

```python
# Demonstrate ESS scaling with n
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

# Fixed proposal-target pair
target = dist.Normal(5.0, 1.0)
proposal = dist.Normal(3.0, 2.0)  # Intentionally mismatched

print("\nESS Scaling with Sample Size")
print(f"Target: N(5, 1), Proposal: N(3, 2)")
print("-" * 50)
print(f"{'n':>10} {'ESS':>12} {'ESS/n':>12}")
print("-" * 50)

ess_ratios = []
for n in sample_sizes:
    samples = proposal.sample((n,))
    log_weights = target.log_prob(samples) - proposal.log_prob(samples)
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
    
    ess = compute_ess_normalized(weights)
    ratio = ess / n
    ess_ratios.append(ratio.item())
    
    print(f"{n:10d} {ess.item():12.1f} {ratio.item():12.3f}")

print(f"\nESS/n converges to approximately {sum(ess_ratios[-3:])/3:.3f}")
```

## Variance-ESS Relationship

### Empirical Verification

```python
def verify_variance_ess_relationship(target_log_prob, proposal, h_function, 
                                      true_value, n_samples=5000, n_reps=500):
    """
    Empirically verify that Var(estimator) ≈ Var_π(h)/ESS
    """
    estimates = []
    ess_values = []
    
    for _ in range(n_reps):
        samples = proposal.sample((n_samples,))
        log_weights = target_log_prob(samples) - proposal.log_prob(samples)
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
        
        estimate = torch.sum(weights * h_function(samples))
        ess = compute_ess_normalized(weights)
        
        estimates.append(estimate.item())
        ess_values.append(ess.item())
    
    estimates = torch.tensor(estimates)
    ess_values = torch.tensor(ess_values)
    
    empirical_var = estimates.var().item()
    mean_ess = ess_values.mean().item()
    mse = ((estimates - true_value)**2).mean().item()
    bias = (estimates.mean() - true_value).item()
    
    # Theoretical prediction
    # First estimate Var_π(h) using high-quality IS
    good_proposal = dist.Normal(5.0, 1.1)  # Close to target
    samples = good_proposal.sample((50000,))
    log_w = target_log_prob(samples) - good_proposal.log_prob(samples)
    w = torch.exp(log_w - torch.logsumexp(log_w, 0))
    h_vals = h_function(samples)
    weighted_mean = torch.sum(w * h_vals)
    var_h = torch.sum(w * (h_vals - weighted_mean)**2)
    
    predicted_var = var_h.item() / mean_ess
    
    print(f"\nVariance-ESS Relationship Verification")
    print("=" * 50)
    print(f"  Mean ESS: {mean_ess:.1f}")
    print(f"  Estimated Var_π(h): {var_h.item():.4f}")
    print(f"  Predicted Var(estimator): {predicted_var:.6f}")
    print(f"  Empirical Var(estimator): {empirical_var:.6f}")
    print(f"  Ratio (empirical/predicted): {empirical_var/predicted_var:.2f}")
    print(f"  Bias: {bias:.6f}")
    print(f"  RMSE: {mse**0.5:.6f}")

# Verify
target_log_prob = lambda x: dist.Normal(5.0, 1.0).log_prob(x)
proposal = dist.Normal(3.0, 2.0)
h = lambda x: x**2
true_value = 5**2 + 1**2  # E[X²] for N(5,1)

verify_variance_ess_relationship(target_log_prob, proposal, h, true_value)
```

## ESS in Practice

### Minimum ESS Requirements

| Application | Minimum ESS | Rationale |
|-------------|-------------|-----------|
| Point estimates | 100-500 | Basic CLT applicability |
| Posterior quantiles | 500-1000 | Need density in tails |
| Credible intervals | 1000+ | Require accurate tail coverage |
| Model comparison | 1000+ | Sensitive to weight distribution |
| Publication quality | 5000+ | Low Monte Carlo error |

### When ESS is Low

**Symptoms:**
- Large variance in estimates
- Unstable results across runs
- Extreme weight values
- Few samples carrying most weight

**Solutions:**
1. **Improve proposal**: Better location, scale, or family
2. **Use adaptive methods**: Let the algorithm find good proposal
3. **Switch to MCMC**: May be more appropriate for the problem
4. **Increase $n$**: Only helps if ESS/n is reasonable

**When NOT to just increase n:**
- If ESS/n < 0.01, doubling $n$ only doubles ESS
- Better to improve proposal and get ESS/n > 0.1

### Monitoring ESS Over Time

For sequential or adaptive algorithms, track ESS evolution:

```python
def track_ess_over_iterations(log_target, initial_proposal, n_per_iter, n_iters):
    """
    Track ESS as we accumulate samples or adapt proposal.
    """
    ess_history = []
    cumulative_ess_history = []
    
    all_samples = []
    all_log_weights = []
    
    current_proposal = initial_proposal
    
    for t in range(n_iters):
        # Draw samples
        samples = current_proposal.sample((n_per_iter,))
        log_weights = log_target(samples) - current_proposal.log_prob(samples)
        
        all_samples.append(samples)
        all_log_weights.append(log_weights)
        
        # ESS for this iteration
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
        ess_iter = compute_ess_normalized(weights)
        ess_history.append(ess_iter.item())
        
        # Cumulative ESS (all samples so far)
        all_log_w = torch.cat(all_log_weights)
        all_w = torch.exp(all_log_w - torch.logsumexp(all_log_w, 0))
        cumulative_ess = compute_ess_normalized(all_w)
        cumulative_ess_history.append(cumulative_ess.item())
        
        # Simple adaptation: fit Gaussian to weighted samples
        if t > 0 and t % 5 == 0:
            all_s = torch.cat(all_samples)
            weighted_mean = torch.sum(all_w.unsqueeze(-1) * all_s, dim=0)
            weighted_var = torch.sum(all_w * (all_s - weighted_mean)**2)
            current_proposal = dist.Normal(weighted_mean, 1.2 * torch.sqrt(weighted_var))
        
        print(f"Iter {t+1}: ESS = {ess_iter.item():.1f}, "
              f"Cumulative ESS = {cumulative_ess.item():.1f}")
    
    return ess_history, cumulative_ess_history
```

## Visualization

```python
def plot_ess_diagnostics(weights, samples, name=""):
    """
    Comprehensive ESS visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n = len(weights)
    
    if not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        weights = weights / weights.sum()
    
    ess = compute_ess_normalized(weights)
    
    # Panel 1: Weight histogram
    ax = axes[0, 0]
    ax.hist(weights.numpy() * n, bins=50, density=True, 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, 
               label='Uniform')
    ax.set_xlabel('Normalized Weight × n', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Weight Distribution (ESS={ess.item():.1f}, {ess.item()/n:.1%})', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative weight curve
    sorted_w = torch.sort(weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    
    ax = axes[0, 1]
    ax.plot(torch.arange(1, n+1).numpy(), cumsum.numpy(), 'b-', linewidth=2)
    
    # Reference lines
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='50%')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
    
    # Ideal curve (uniform weights)
    ax.plot(torch.arange(1, n+1).numpy(), torch.arange(1, n+1).numpy()/n, 
            'g:', linewidth=2, alpha=0.7, label='Ideal (uniform)')
    
    ax.set_xlabel('Number of Top Samples', fontsize=11)
    ax.set_ylabel('Cumulative Weight', fontsize=11)
    ax.set_title('Weight Concentration', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Log weights histogram
    log_weights = torch.log(weights * n)
    
    ax = axes[1, 0]
    ax.hist(log_weights.numpy(), bins=50, density=True,
            alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2,
               label='log(1) = 0 (uniform)')
    ax.set_xlabel('Log(Normalized Weight × n)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Log Weight Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: ESS vs n simulation
    # Show how ESS/n remains constant
    ns = [100, 200, 500, 1000, 2000, 5000]
    ratios = []
    
    # Use same proposal-target pair
    target = dist.Normal(5.0, 1.0)
    proposal = dist.Normal(3.0, 2.0)
    
    for nn in ns:
        s = proposal.sample((nn,))
        lw = target.log_prob(s) - proposal.log_prob(s)
        w = torch.exp(lw - torch.logsumexp(lw, 0))
        ratios.append((compute_ess_normalized(w) / nn).item())
    
    ax = axes[1, 1]
    ax.plot(ns, ratios, 'bo-', linewidth=2, markersize=8)
    ax.axhline(sum(ratios)/len(ratios), color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean = {sum(ratios)/len(ratios):.3f}')
    ax.set_xlabel('Sample Size n', fontsize=11)
    ax.set_ylabel('ESS/n', fontsize=11)
    ax.set_title('ESS Ratio is Constant in n', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(name, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
```

## Key Takeaways

!!! success "What ESS Tells You"
    - **ESS ≈ n**: Excellent proposal, weights nearly uniform
    - **ESS/n > 0.2**: Good efficiency, results reliable
    - **ESS/n < 0.05**: Poor efficiency, consider improving proposal
    - **ESS/n < 0.01**: Estimates unreliable, must fix proposal

!!! warning "What ESS Doesn't Tell You"
    - Whether the proposal covers all modes (can have high ESS but miss modes)
    - Whether samples explore the full support
    - Bias from support mismatch (only variance-related)

!!! info "ESS Best Practices"
    1. Always compute and report ESS with IS estimates
    2. ESS/n is more interpretable than raw ESS
    3. If ESS is low, improve proposal rather than increase n
    4. Use ESS to compare proposals objectively
    5. Monitor ESS over iterations in adaptive methods

## Exercises

### Exercise 1: ESS Bounds
Prove that $1 \leq \text{ESS} \leq n$. Under what conditions are the bounds achieved?

### Exercise 2: ESS vs Perplexity
Compare ESS and perplexity for the same weight distributions. When do they give different rankings of proposal quality?

### Exercise 3: ESS Estimation Variance
The ESS itself is estimated from samples. How variable is ESS across runs? Compute standard error of ESS estimates.

### Exercise 4: Multimodal ESS
Can ESS be high when missing a mode? Construct an example and discuss implications.

## References

1. Kong, A. (1992). "A note on importance sampling using standardized weights." University of Chicago Department of Statistics Technical Report 348.

2. Liu, J. S. (2001). *Monte Carlo Strategies in Scientific Computing*. Springer. Section 2.5.

3. Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing: Fifteen years later." *Handbook of Nonlinear Filtering*, 12, 656-704.

4. Elvira, V., Martino, L., & Robert, C. P. (2019). "Rethinking the effective sample size." *International Statistical Review*, 87(3), 591-616.

5. Vehtari, A., Gelman, A., & Gabry, J. (2017). "Pareto smoothed importance sampling." *arXiv preprint arXiv:1507.02646*.
