# Proposal Distribution Design

## Overview

The choice of proposal distribution is the single most important factor determining the success or failure of importance sampling. A well-designed proposal can achieve orders of magnitude variance reduction, while a poorly chosen proposal can render the estimator useless. This section covers the principles and practical strategies for designing effective proposal distributions.

## The Optimal Proposal

### Theoretical Result

For estimating $\mathbb{E}_\pi[h(\theta)]$, the variance-minimizing proposal is:

$$
q^*(\theta) = \frac{|h(\theta)| \pi(\theta)}{\int |h(\theta')| \pi(\theta') d\theta'}
$$

### Why This is Optimal

With $q^*$, the importance weight becomes:

$$
w(\theta) = \frac{\pi(\theta)}{q^*(\theta)} = \frac{\int |h(\theta')| \pi(\theta') d\theta'}{|h(\theta)|} \cdot \text{sign}(h(\theta))
$$

The product $h(\theta) w(\theta)$ has reduced variability because:
- Where $|h(\theta)|$ is large, $w(\theta)$ is small (compensating)
- Where $|h(\theta)|$ is small, $w(\theta)$ is large (but contribution is small anyway)

### The Impossibility Paradox

**Problem**: The optimal $q^*$ requires knowing $\int |h(\theta)| \pi(\theta) d\theta$ — the very quantity we're trying to estimate!

**Solution**: Use $q^*$ as a **guide** for what good proposals should look like:
1. Similar shape to $|h(\theta)| \pi(\theta)$
2. Heavier tails than $\pi(\theta)$
3. Cover the full support of $\pi(\theta)$

## Design Principles

### Principle 1: Support Coverage

!!! danger "Critical Requirement"
    $$\pi(\theta) > 0 \implies q(\theta) > 0$$
    
    Violation leads to **infinite bias**, not just high variance.

**Example of Failure:**
- Target: $\pi = \mathcal{N}(0, 1)$
- Proposal: $q = \text{Uniform}(-2, 2)$
- Problem: $q(\theta) = 0$ for $|\theta| > 2$, missing $\approx 5\%$ of $\pi$'s mass

### Principle 2: Tail Dominance

The proposal should have **heavier tails** than the target:

$$
\lim_{|\theta| \to \infty} \frac{q(\theta)}{\pi(\theta)} > 0
$$

**Why?** Lighter-tailed proposals can produce extreme weights in the tails, causing variance explosion.

**Practical Rule:**
- If $\pi$ is Gaussian, use $t$-distribution or wider Gaussian
- If $\pi$ is $t$-distributed, use $t$ with fewer degrees of freedom
- When in doubt, use heavier tails

### Principle 3: Shape Matching

The proposal should approximate the shape of the integrand:

$$
q(\theta) \approx c \cdot |h(\theta)| \pi(\theta)
$$

for some constant $c$.

**Strategies:**
1. Match location (mean) and scale (variance) to $\pi$
2. For tail probabilities, shift proposal toward the tail
3. For multimodal $\pi$, use mixture proposals

### Principle 4: Computational Feasibility

The proposal must be:
1. **Easy to sample from**: Efficient random number generation
2. **Easy to evaluate**: $q(\theta)$ computable in closed form
3. **Ideally, both**: Many standard distributions satisfy this

## Common Proposal Families

### Gaussian Proposals

**Form:** $q(\theta) = \mathcal{N}(\mu_q, \Sigma_q)$

**Advantages:**
- Simple to sample and evaluate
- Well-understood properties
- Works well for unimodal targets

**Parameter Selection:**
- $\mu_q$: Posterior mean estimate (or prior mean)
- $\Sigma_q$: Slightly inflated posterior covariance

```python
import torch
import torch.distributions as dist

def gaussian_proposal_from_laplace(theta_map, hessian_at_map, inflation=1.2):
    """
    Create Gaussian proposal from Laplace approximation.
    
    Parameters
    ----------
    theta_map : torch.Tensor
        Maximum a posteriori estimate
    hessian_at_map : torch.Tensor
        Hessian of negative log-posterior at MAP
    inflation : float
        Factor to inflate covariance for robustness
        
    Returns
    -------
    proposal : torch.distributions.Normal or MultivariateNormal
        Gaussian proposal centered at MAP
    """
    # Covariance = inverse of Hessian of negative log-posterior
    cov = torch.inverse(hessian_at_map)
    
    # Inflate covariance for robustness
    cov = inflation * cov
    
    if theta_map.dim() == 0 or theta_map.numel() == 1:
        return dist.Normal(theta_map, torch.sqrt(cov.squeeze()))
    else:
        return dist.MultivariateNormal(theta_map, cov)
```

### Student-t Proposals

**Form:** $q(\theta) = t_\nu(\mu_q, \Sigma_q)$

**Advantages:**
- Heavier tails than Gaussian
- Controlled via degrees of freedom $\nu$
- Reduces risk of weight explosion

**Parameter Selection:**
- $\nu = 3$ to $5$: Heavy tails
- $\nu > 30$: Approximately Gaussian
- $\mu_q, \Sigma_q$: Same as Gaussian proposals

```python
def student_t_proposal(location, scale, df=4):
    """
    Student-t proposal with specified degrees of freedom.
    
    Note: PyTorch has univariate StudentT.
    For multivariate, use scale mixture representation.
    """
    return dist.StudentT(df=df, loc=location, scale=scale)
```

### Mixture Proposals

**Form:** $q(\theta) = \sum_{k=1}^K \alpha_k q_k(\theta)$

**Advantages:**
- Handles multimodal targets
- Flexible shape approximation
- Each component can be simple

**Sampling:**
1. Sample component $k$ with probability $\alpha_k$
2. Sample $\theta \sim q_k(\theta)$

**Density Evaluation:**
$$q(\theta) = \sum_{k=1}^K \alpha_k q_k(\theta)$$

```python
class MixtureProposal:
    """
    Mixture of distributions as importance sampling proposal.
    """
    
    def __init__(self, components, weights):
        """
        Parameters
        ----------
        components : list of distributions
            Component distributions q_k
        weights : torch.Tensor
            Mixture weights α_k (will be normalized)
        """
        self.components = components
        self.weights = weights / weights.sum()
        self.n_components = len(components)
        
    def sample(self, n_samples):
        """Sample from mixture."""
        # Sample component indices
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # Sample from selected components
        samples = []
        for k in range(self.n_components):
            n_k = (indices == k).sum().item()
            if n_k > 0:
                samples_k = self.components[k].sample((n_k,))
                samples.append(samples_k)
        
        # Combine and shuffle
        samples = torch.cat(samples, dim=0)
        perm = torch.randperm(n_samples)
        return samples[perm]
    
    def log_prob(self, theta):
        """Evaluate log mixture density."""
        log_probs = []
        for k, (comp, alpha) in enumerate(zip(self.components, self.weights)):
            log_probs.append(torch.log(alpha) + comp.log_prob(theta))
        
        log_probs = torch.stack(log_probs, dim=-1)
        return torch.logsumexp(log_probs, dim=-1)


# Example: Bimodal target
# Create mixture proposal matching modes
mixture_proposal = MixtureProposal(
    components=[
        dist.Normal(-3.0, 1.2),  # Component at first mode
        dist.Normal(3.0, 1.2)   # Component at second mode
    ],
    weights=torch.tensor([0.5, 0.5])
)
```

### Prior as Proposal

**Form:** $q(\theta) = p(\theta)$

**Advantages:**
- No tuning required
- Always covers support
- Natural baseline

**Limitations:**
- Inefficient when likelihood is informative
- ESS can be very low with large datasets

**When to Use:**
- Quick sanity checks
- Small datasets (weak likelihood)
- Non-informative priors

```python
def prior_as_proposal_is(h_function, log_likelihood, prior, n_samples):
    """
    Importance sampling using prior as proposal.
    
    Target: π(θ) ∝ p(y|θ) p(θ)
    Proposal: q(θ) = p(θ)
    Weight: w(θ) = p(y|θ)
    """
    # Sample from prior
    samples = prior.sample((n_samples,))
    
    # Weights are just the likelihood values
    log_weights = log_likelihood(samples)
    
    # Normalize weights
    log_sum = torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights - log_sum)
    
    # SNIS estimate
    h_values = h_function(samples)
    estimate = torch.sum(weights * h_values)
    
    # ESS
    ess = 1.0 / torch.sum(weights**2)
    
    return estimate, ess, samples, weights
```

## Advanced Strategies

### Laplace Approximation

Approximate the posterior with a Gaussian at the mode:

$$
q(\theta) = \mathcal{N}(\hat{\theta}_{\text{MAP}}, H^{-1})
$$

where $H$ is the Hessian of the negative log-posterior at the MAP.

```python
import torch.autograd.functional as F

def laplace_approximation(log_posterior, init_theta, lr=0.1, n_steps=1000):
    """
    Compute Laplace approximation to posterior.
    
    Returns
    -------
    theta_map : torch.Tensor
        MAP estimate
    cov : torch.Tensor
        Approximate posterior covariance
    """
    theta = init_theta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    # Find MAP via gradient ascent
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = -log_posterior(theta)
        loss.backward()
        optimizer.step()
    
    theta_map = theta.detach()
    
    # Compute Hessian of negative log-posterior
    def neg_log_post(t):
        return -log_posterior(t)
    
    hessian = F.hessian(neg_log_post, theta_map)
    
    # Covariance is inverse Hessian
    cov = torch.inverse(hessian)
    
    return theta_map, cov


def create_laplace_proposal(theta_map, cov, inflation=1.5):
    """
    Create proposal from Laplace approximation.
    """
    inflated_cov = inflation * cov
    
    if theta_map.dim() == 0 or theta_map.numel() == 1:
        return dist.Normal(theta_map, torch.sqrt(inflated_cov.squeeze()))
    else:
        return dist.MultivariateNormal(theta_map, inflated_cov)
```

### Adaptive Importance Sampling

Iteratively improve the proposal based on samples:

**Algorithm (Population Monte Carlo):**
1. Initialize proposal $q_0$
2. For $t = 1, 2, \ldots, T$:
   a. Sample $\theta_i^t \sim q_{t-1}$
   b. Compute weights $w_i^t$
   c. Update proposal $q_t$ based on weighted samples
3. Return final weighted samples

```python
class AdaptiveImportanceSampler:
    """
    Adaptive importance sampling with Gaussian mixture proposal.
    """
    
    def __init__(self, log_target, dim, n_components=5):
        self.log_target = log_target
        self.dim = dim
        self.n_components = n_components
        
        # Initialize with broad Gaussian
        self.means = [torch.zeros(dim)]
        self.covs = [4.0 * torch.eye(dim)]
        self.mixture_weights = torch.tensor([1.0])
        
    def run(self, n_samples_per_iter, n_iterations):
        """Run adaptive IS."""
        all_samples = []
        all_weights = []
        
        for t in range(n_iterations):
            # Sample from current proposal
            samples = self._sample_mixture(n_samples_per_iter)
            
            # Compute weights
            log_target_vals = self.log_target(samples)
            log_proposal_vals = self._log_mixture_density(samples)
            log_weights = log_target_vals - log_proposal_vals
            weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
            
            all_samples.append(samples)
            all_weights.append(weights)
            
            # Update proposal
            self._update_proposal(samples, weights)
            
            # Report
            ess = 1.0 / torch.sum(weights**2)
            print(f"Iteration {t+1}: ESS = {ess.item():.1f} "
                  f"({ess.item()/n_samples_per_iter:.1%})")
        
        return torch.cat(all_samples), torch.cat(all_weights)
    
    def _sample_mixture(self, n):
        """Sample from current mixture proposal."""
        weights = self.mixture_weights / self.mixture_weights.sum()
        
        samples = []
        for _ in range(n):
            k = torch.multinomial(weights, 1).item()
            sample = dist.MultivariateNormal(
                self.means[k], self.covs[k]
            ).sample()
            samples.append(sample)
        
        return torch.stack(samples)
    
    def _log_mixture_density(self, samples):
        """Evaluate log mixture density."""
        log_probs = []
        weights = self.mixture_weights / self.mixture_weights.sum()
        
        for k, (mean, cov, w) in enumerate(zip(self.means, self.covs, weights)):
            comp = dist.MultivariateNormal(mean, cov)
            log_probs.append(torch.log(w) + comp.log_prob(samples))
        
        return torch.logsumexp(torch.stack(log_probs), dim=0)
    
    def _update_proposal(self, samples, weights):
        """Update mixture proposal based on weighted samples."""
        # Resample according to weights
        indices = torch.multinomial(weights, self.n_components, replacement=True)
        
        # New component means
        new_means = [samples[i] for i in indices]
        
        # Estimate global covariance
        weighted_mean = torch.sum(weights.unsqueeze(-1) * samples, dim=0)
        weighted_cov = torch.zeros(self.dim, self.dim)
        for s, w in zip(samples, weights):
            diff = s - weighted_mean
            weighted_cov += w * torch.outer(diff, diff)
        
        # Add regularization
        weighted_cov += 0.01 * torch.eye(self.dim)
        
        # Update
        self.means = new_means
        self.covs = [weighted_cov for _ in range(self.n_components)]
        self.mixture_weights = torch.ones(self.n_components) / self.n_components
```

## Diagnostics for Proposal Quality

### Weight-Based Diagnostics

```python
def proposal_diagnostics(weights, name=""):
    """
    Comprehensive proposal quality assessment.
    """
    n = len(weights)
    
    # Normalize weights if needed
    if not torch.isclose(weights.sum(), torch.tensor(1.0)):
        weights = weights / weights.sum()
    
    # ESS
    ess = 1.0 / torch.sum(weights**2)
    
    # Coefficient of variation
    cv = weights.std() / weights.mean()
    
    # Kurtosis of weights
    mean_w = weights.mean()
    kurtosis = ((weights - mean_w)**4).mean() / ((weights - mean_w)**2).mean()**2
    
    # Maximum weight ratio
    max_ratio = weights.max() * n
    
    # Weight concentration
    sorted_w = torch.sort(weights, descending=True)[0]
    cumsum = torch.cumsum(sorted_w, dim=0)
    n_for_50 = (cumsum < 0.5).sum().item() + 1
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    
    print(f"\nProposal Diagnostics: {name}")
    print("=" * 50)
    print(f"  ESS: {ess.item():.1f} / {n} ({ess.item()/n:.1%})")
    print(f"  CV of weights: {cv.item():.3f}")
    print(f"  Kurtosis: {kurtosis.item():.1f}")
    print(f"  Max weight / uniform: {max_ratio.item():.1f}x")
    print(f"  Samples for 50% weight: {n_for_50} ({n_for_50/n:.1%})")
    print(f"  Samples for 90% weight: {n_for_90} ({n_for_90/n:.1%})")
    
    # Quality assessment
    if ess.item() / n > 0.5:
        quality = "Excellent"
    elif ess.item() / n > 0.2:
        quality = "Good"
    elif ess.item() / n > 0.05:
        quality = "Acceptable"
    else:
        quality = "Poor - consider improving proposal"
    
    print(f"\n  Assessment: {quality}")
    
    return {'ess': ess.item(), 'ess_ratio': ess.item()/n, 'quality': quality}
```

## Practical Recommendations

### Decision Tree for Proposal Selection

```
Is the posterior approximately Gaussian?
├── Yes → Use Laplace approximation (inflated covariance)
└── No → Is it multimodal?
    ├── Yes → Use mixture proposal (components at each mode)
    └── No → Is it heavy-tailed?
        ├── Yes → Use Student-t proposal (df = 3-5)
        └── No → Start with inflated Gaussian, check ESS
```

### Rules of Thumb

| Situation | Recommended Proposal |
|-----------|---------------------|
| Quick baseline | Prior |
| Unimodal, well-behaved | Laplace approximation |
| Heavy tails suspected | Student-t ($\nu = 3-5$) |
| Multimodal | Mixture of Gaussians |
| High-dimensional | Variational approximation |
| No good initial guess | Adaptive IS |

### ESS Targets

| ESS/n | Quality | Action |
|-------|---------|--------|
| > 0.5 | Excellent | None needed |
| 0.2-0.5 | Good | Acceptable for most uses |
| 0.05-0.2 | Marginal | Consider improvement |
| < 0.05 | Poor | Must improve proposal |
| < 0.01 | Failure | Results unreliable |

## Key Takeaways

!!! success "Good Proposal Characteristics"
    - Covers full support of target
    - Heavier tails than target
    - Shape matches $|h(\theta)|\pi(\theta)$
    - Easy to sample and evaluate

!!! warning "Common Pitfalls"
    - Proposal tails lighter than target → weight explosion
    - Missing modes → infinite bias for multimodal targets
    - Too narrow proposal → poor coverage
    - Too broad proposal → low ESS (but safe)

!!! tip "Practical Workflow"
    1. Start with simple proposal (prior or Laplace)
    2. Check ESS and weight diagnostics
    3. If ESS too low, improve proposal
    4. Consider adaptive methods for complex targets

## Exercises

### Exercise 1: Tail Mismatch
Compare ESS for $\pi = t_3(0, 1)$ (Student-t with 3 df) using proposals: (a) $\mathcal{N}(0, 1.5)$, (b) $t_3(0, 1.5)$, (c) $t_5(0, 1.5)$. Explain the results.

### Exercise 2: Mode Discovery
Design a mixture proposal for $\pi = 0.3 \mathcal{N}(-5, 1) + 0.7 \mathcal{N}(3, 0.5)$. Compare ESS against a single Gaussian proposal.

### Exercise 3: Adaptive Refinement
Implement a simple adaptive scheme: (1) run IS with initial proposal, (2) fit Gaussian to weighted samples, (3) repeat. Track ESS improvement over iterations.

## References

1. Owen, A. B. (2013). *Monte Carlo theory, methods and examples*. Chapter 9.5: Proposal Distributions.

2. Cappé, O., Guillin, A., Marin, J. M., & Robert, C. P. (2004). "Population Monte Carlo." *Journal of Computational and Graphical Statistics*, 13(4), 907-929.

3. Cornuet, J. M., Marin, J. M., Mira, A., & Robert, C. P. (2012). "Adaptive multiple importance sampling." *Scandinavian Journal of Statistics*, 39(4), 798-812.

4. Bugallo, M. F., Elvira, V., Martino, L., Luengo, D., Miguez, J., & Djuric, P. M. (2017). "Adaptive importance sampling: The past, the present, and the future." *IEEE Signal Processing Magazine*, 34(4), 60-79.
