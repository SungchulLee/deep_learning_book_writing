# Asymptotic Properties of MLE

## Introduction

Maximum Likelihood Estimators possess remarkable properties when the sample size is large. These **asymptotic properties** explain why MLE is the workhorse of statistical estimation and why neural network training (which is essentially MLE) works so well with large datasets.

!!! abstract "Key Asymptotic Properties"
    Under regularity conditions, as $n \to \infty$:
    
    1. **Consistency**: $\hat{\theta}_n \xrightarrow{p} \theta_0$ (converges to true value)
    2. **Asymptotic Normality**: $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$
    3. **Efficiency**: MLE achieves the Cramér-Rao lower bound asymptotically
    4. **Invariance**: MLE of $g(\theta)$ is $g(\hat{\theta})$

## Regularity Conditions

The asymptotic properties of MLE hold under certain **regularity conditions**:

1. **Identifiability**: Different parameter values give different distributions
   $$\theta_1 \neq \theta_2 \implies p(x|\theta_1) \neq p(x|\theta_2)$$

2. **Common Support**: The support of $p(x|\theta)$ doesn't depend on $\theta$

3. **Differentiability**: $\log p(x|\theta)$ is three times differentiable in $\theta$

4. **Bounded Derivatives**: Third derivatives are bounded by an integrable function

5. **Open Parameter Space**: True parameter $\theta_0$ is in the interior of $\Theta$

!!! warning "When Regularity Fails"
    Some important distributions violate these conditions:
    
    - **Uniform$[0, \theta]$**: Support depends on $\theta$
    - **Mixture models**: May have multiple local maxima
    - **Boundary cases**: Parameter on boundary of $\Theta$

## Consistency

### Definition

An estimator $\hat{\theta}_n$ is **consistent** if it converges in probability to the true parameter:

$$
\hat{\theta}_n \xrightarrow{p} \theta_0 \quad \text{as } n \to \infty
$$

This means: $\forall \epsilon > 0, \; P(|\hat{\theta}_n - \theta_0| > \epsilon) \to 0$

### Why MLE is Consistent

The key insight is that maximizing the log-likelihood is equivalent to minimizing the Kullback-Leibler divergence:

$$
\hat{\theta}_n = \arg\max_\theta \frac{1}{n}\sum_{i=1}^{n} \log p(x_i | \theta)
$$

By the Law of Large Numbers:

$$
\frac{1}{n}\sum_{i=1}^{n} \log p(x_i | \theta) \xrightarrow{p} \mathbb{E}_{\theta_0}[\log p(X | \theta)]
$$

And $\mathbb{E}_{\theta_0}[\log p(X | \theta)]$ is maximized at $\theta = \theta_0$ (information inequality).

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def demonstrate_consistency(true_theta: float = 0.7, 
                           sample_sizes: List[int] = None,
                           n_simulations: int = 1000):
    """
    Demonstrate MLE consistency for Bernoulli parameter.
    
    Shows that as n increases, MLE concentrates around true value.
    """
    if sample_sizes is None:
        sample_sizes = [10, 50, 100, 500, 1000, 5000]
    
    torch.manual_seed(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, n in enumerate(sample_sizes):
        estimates = []
        for _ in range(n_simulations):
            # Generate data
            data = (torch.rand(n) < true_theta).float()
            # MLE
            theta_hat = data.mean().item()
            estimates.append(theta_hat)
        
        ax = axes[idx]
        ax.hist(estimates, bins=30, density=True, alpha=0.7, 
                edgecolor='black')
        ax.axvline(true_theta, color='red', linewidth=2, 
                   label=f'True θ = {true_theta}')
        ax.axvline(np.mean(estimates), color='blue', linewidth=2, 
                   linestyle='--', label=f'Mean MLE = {np.mean(estimates):.4f}')
        ax.set_xlabel('θ̂')
        ax.set_ylabel('Density')
        ax.set_title(f'n = {n}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('MLE Consistency: Distribution Concentrates as n → ∞', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print convergence statistics
    print("\nConvergence Statistics:")
    print("-" * 50)
    print(f"{'n':>8} {'Mean MLE':>12} {'Std MLE':>12} {'|Bias|':>12}")
    print("-" * 50)
    
    for n in sample_sizes:
        estimates = []
        for _ in range(n_simulations):
            data = (torch.rand(n) < true_theta).float()
            estimates.append(data.mean().item())
        
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        bias = abs(mean_est - true_theta)
        print(f"{n:>8} {mean_est:>12.6f} {std_est:>12.6f} {bias:>12.6f}")
```

## Asymptotic Normality

### The Central Limit Theorem for MLE

Under regularity conditions:

$$
\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, I(\theta_0)^{-1}\right)
$$

Equivalently:

$$
\hat{\theta}_n \stackrel{approx}{\sim} \mathcal{N}\left(\theta_0, \frac{1}{nI(\theta_0)}\right)
$$

### Derivation Sketch

1. **Taylor expand** the score function around $\theta_0$:
   $$s(\hat{\theta}) = s(\theta_0) + (\hat{\theta} - \theta_0) s'(\tilde{\theta})$$
   where $\tilde{\theta}$ is between $\hat{\theta}$ and $\theta_0$.

2. **At the MLE**, $s(\hat{\theta}) = 0$, so:
   $$0 = s(\theta_0) + (\hat{\theta} - \theta_0) s'(\tilde{\theta})$$

3. **Rearranging**:
   $$\sqrt{n}(\hat{\theta} - \theta_0) = -\frac{\sqrt{n} \cdot s(\theta_0)/n}{s'(\tilde{\theta})/n}$$

4. **By CLT**: $\sqrt{n} \cdot \bar{s}(\theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0))$

5. **By LLN**: $s'(\tilde{\theta})/n \xrightarrow{p} -I(\theta_0)$

6. **By Slutsky's theorem**: The ratio converges to $\mathcal{N}(0, I(\theta_0)^{-1})$

### Multivariate Case

For parameter vector $\boldsymbol{\theta}$:

$$
\sqrt{n}(\hat{\boldsymbol{\theta}}_n - \boldsymbol{\theta}_0) \xrightarrow{d} \mathcal{N}\left(\mathbf{0}, \mathbf{I}(\boldsymbol{\theta}_0)^{-1}\right)
$$

```python
def demonstrate_asymptotic_normality(true_theta: float = 0.7, 
                                     n: int = 100,
                                     n_simulations: int = 5000):
    """
    Demonstrate asymptotic normality of MLE.
    
    Compare empirical distribution to theoretical normal approximation.
    """
    torch.manual_seed(42)
    
    # Fisher Information for Bernoulli
    fisher_info = 1 / (true_theta * (1 - true_theta))
    asymptotic_var = 1 / (n * fisher_info)
    asymptotic_std = np.sqrt(asymptotic_var)
    
    # Generate MLEs
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_theta).float()
        estimates.append(data.mean().item())
    
    estimates = np.array(estimates)
    
    # Standardized estimates
    standardized = np.sqrt(n) * (estimates - true_theta)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original scale
    ax = axes[0]
    ax.hist(estimates, bins=50, density=True, alpha=0.7, label='Empirical')
    
    x = np.linspace(true_theta - 4*asymptotic_std, 
                    true_theta + 4*asymptotic_std, 200)
    from scipy.stats import norm
    theoretical = norm.pdf(x, true_theta, asymptotic_std)
    ax.plot(x, theoretical, 'r-', linewidth=2, label='Asymptotic Normal')
    ax.axvline(true_theta, color='green', linestyle='--', label='True θ')
    ax.set_xlabel('θ̂')
    ax.set_ylabel('Density')
    ax.set_title(f'MLE Distribution (n = {n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Standardized scale
    ax = axes[1]
    ax.hist(standardized, bins=50, density=True, alpha=0.7, 
            label='Standardized MLE')
    
    x_std = np.linspace(-4, 4, 200)
    theoretical_std = norm.pdf(x_std, 0, 1/np.sqrt(fisher_info))
    ax.plot(x_std, theoretical_std, 'r-', linewidth=2, 
            label=f'N(0, 1/I(θ)) = N(0, {1/fisher_info:.3f})')
    ax.set_xlabel('√n(θ̂ - θ₀)')
    ax.set_ylabel('Density')
    ax.set_title('Standardized MLE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAsymptotic Normality Check (n = {n}):")
    print("-" * 40)
    print(f"Theoretical asymptotic std: {asymptotic_std:.6f}")
    print(f"Empirical std of MLEs: {np.std(estimates):.6f}")
    print(f"Ratio (should be ≈ 1): {np.std(estimates)/asymptotic_std:.4f}")
```

## Efficiency

### Definition

An estimator is **efficient** if it achieves the Cramér-Rao lower bound:

$$
\text{Var}(\hat{\theta}) = \frac{1}{nI(\theta)}
$$

### MLE is Asymptotically Efficient

Among all consistent and asymptotically normal estimators, MLE has the **smallest asymptotic variance**:

$$
\text{Avar}(\hat{\theta}_{\text{MLE}}) = \frac{1}{I(\theta_0)} \leq \text{Avar}(\hat{\theta}_{\text{other}})
$$

This is why MLE is the "best" estimator in large samples.

### Relative Efficiency

The **asymptotic relative efficiency** (ARE) compares two estimators:

$$
\text{ARE}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Avar}(\hat{\theta}_2)}{\text{Avar}(\hat{\theta}_1)}
$$

If ARE > 1, then $\hat{\theta}_1$ is more efficient.

```python
def compare_estimator_efficiency():
    """
    Compare efficiency of different estimators for Normal mean.
    
    MLE (sample mean) vs. sample median vs. trimmed mean.
    """
    torch.manual_seed(42)
    
    true_mu = 5.0
    true_sigma = 2.0
    sample_sizes = [10, 50, 100, 500, 1000]
    n_simulations = 5000
    
    results = {n: {'mle': [], 'median': [], 'trimmed': []} 
               for n in sample_sizes}
    
    for n in sample_sizes:
        for _ in range(n_simulations):
            data = torch.randn(n) * true_sigma + true_mu
            
            # MLE (sample mean)
            results[n]['mle'].append(data.mean().item())
            
            # Median
            results[n]['median'].append(data.median().item())
            
            # 10% trimmed mean
            sorted_data = torch.sort(data)[0]
            trim = int(0.1 * n)
            if trim > 0:
                trimmed = sorted_data[trim:-trim]
            else:
                trimmed = sorted_data
            results[n]['trimmed'].append(trimmed.mean().item())
    
    # Calculate variances and efficiencies
    print("\nEstimator Efficiency Comparison (Normal Distribution)")
    print("=" * 70)
    print(f"{'n':>6} {'Var(MLE)':>12} {'Var(Median)':>12} {'Var(Trimmed)':>12} {'ARE(Med)':>10}")
    print("-" * 70)
    
    for n in sample_sizes:
        var_mle = np.var(results[n]['mle'])
        var_median = np.var(results[n]['median'])
        var_trimmed = np.var(results[n]['trimmed'])
        
        # Theoretical: Var(median) ≈ (π/2) * Var(mean) for Normal
        are_median = var_mle / var_median
        
        print(f"{n:>6} {var_mle:>12.6f} {var_median:>12.6f} {var_trimmed:>12.6f} {are_median:>10.4f}")
    
    print("-" * 70)
    print("ARE(Median) = Var(MLE)/Var(Median)")
    print("Theoretical ARE(Median) for Normal ≈ π/2 ≈ 0.637")
```

## Invariance Property

### The Principle

If $\hat{\theta}$ is the MLE of $\theta$, then for any function $g$:

$$
\widehat{g(\theta)} = g(\hat{\theta})
$$

### Example

For Normal distribution:
- MLE of $\mu$ is $\bar{x}$
- MLE of $\mu^2$ is $\bar{x}^2$
- MLE of $e^\mu$ is $e^{\bar{x}}$

!!! warning "Bias from Invariance"
    While invariance is convenient, note that $g(\hat{\theta})$ may be biased even if $\hat{\theta}$ is unbiased.
    
    Example: $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ is biased, even though it's the MLE.

## Confidence Intervals from Asymptotic Normality

### Wald Confidence Interval

Using asymptotic normality:

$$
\hat{\theta} \pm z_{\alpha/2} \sqrt{\frac{1}{nI(\hat{\theta})}}
$$

where $z_{\alpha/2}$ is the $(1-\alpha/2)$ quantile of standard normal.

### Profile Likelihood Confidence Interval

Based on the likelihood ratio statistic:

$$
\{\theta : 2[\ell(\hat{\theta}) - \ell(\theta)] \leq \chi^2_{1, \alpha}\}
$$

This is often preferred for small samples.

```python
def confidence_intervals_comparison(true_theta: float = 0.3, 
                                    n: int = 50,
                                    confidence: float = 0.95,
                                    n_simulations: int = 1000):
    """
    Compare Wald vs. Profile Likelihood confidence intervals.
    """
    from scipy import stats
    
    torch.manual_seed(42)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    chi2_crit = stats.chi2.ppf(confidence, df=1)
    
    wald_coverage = 0
    profile_coverage = 0
    
    wald_widths = []
    profile_widths = []
    
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_theta).float()
        theta_hat = data.mean().item()
        k = int(data.sum().item())
        
        # Wald interval
        if 0 < theta_hat < 1:
            se = np.sqrt(theta_hat * (1 - theta_hat) / n)
            wald_lower = max(0, theta_hat - z * se)
            wald_upper = min(1, theta_hat + z * se)
        else:
            # Edge cases
            wald_lower, wald_upper = theta_hat, theta_hat
        
        wald_widths.append(wald_upper - wald_lower)
        if wald_lower <= true_theta <= wald_upper:
            wald_coverage += 1
        
        # Profile likelihood interval (Wilson/score interval)
        # More accurate for binomial
        center = (k + z**2/2) / (n + z**2)
        margin = z * np.sqrt((theta_hat*(1-theta_hat) + z**2/(4*n)) / (n + z**2))
        profile_lower = max(0, center - margin)
        profile_upper = min(1, center + margin)
        
        profile_widths.append(profile_upper - profile_lower)
        if profile_lower <= true_theta <= profile_upper:
            profile_coverage += 1
    
    print(f"\nConfidence Interval Comparison (n = {n}, true θ = {true_theta})")
    print("-" * 50)
    print(f"Target coverage: {confidence*100}%")
    print(f"Wald CI coverage: {wald_coverage/n_simulations*100:.1f}%")
    print(f"Score CI coverage: {profile_coverage/n_simulations*100:.1f}%")
    print(f"Wald avg width: {np.mean(wald_widths):.4f}")
    print(f"Score avg width: {np.mean(profile_widths):.4f}")
```

## Convergence Rates

### Standard Parametric Rate

Under regularity conditions:

$$
\|\hat{\theta}_n - \theta_0\| = O_p(n^{-1/2})
$$

This means the error shrinks like $1/\sqrt{n}$.

### Implications

- To halve the estimation error, you need **4× more data**
- To achieve 10× more precision, you need **100× more data**

```python
def convergence_rate_demonstration():
    """
    Demonstrate the √n convergence rate of MLE.
    """
    true_theta = 0.6
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_simulations = 2000
    
    rmse_values = []
    
    for n in sample_sizes:
        errors = []
        for _ in range(n_simulations):
            data = (torch.rand(n) < true_theta).float()
            theta_hat = data.mean().item()
            errors.append((theta_hat - true_theta)**2)
        rmse = np.sqrt(np.mean(errors))
        rmse_values.append(rmse)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(sample_sizes, rmse_values, 'bo-', markersize=8, 
              linewidth=2, label='Empirical RMSE')
    
    # Theoretical: RMSE = sqrt(p(1-p)/n)
    theoretical_rmse = [np.sqrt(true_theta*(1-true_theta)/n) for n in sample_sizes]
    ax.loglog(sample_sizes, theoretical_rmse, 'r--', linewidth=2, 
              label='Theoretical: √(p(1-p)/n)')
    
    # Reference line for √n rate
    ref = rmse_values[0] * np.sqrt(sample_sizes[0]) / np.sqrt(np.array(sample_sizes))
    ax.loglog(sample_sizes, ref, 'g:', linewidth=2, label='1/√n reference')
    
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('RMSE')
    ax.set_title('MLE Convergence Rate: O(1/√n)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Practical Implications

### Sample Size Requirements

For margin of error $\epsilon$ with confidence $1-\alpha$:

$$
n \geq \frac{z_{\alpha/2}^2}{I(\theta) \cdot \epsilon^2}
$$

### Standard Errors

The standard error of the MLE is approximately:

$$
\text{SE}(\hat{\theta}) \approx \frac{1}{\sqrt{nI(\hat{\theta})}}
$$

This is used to construct confidence intervals and hypothesis tests.

## Summary Table

| Property | Statement | Requirement |
|----------|-----------|-------------|
| **Consistency** | $\hat{\theta}_n \xrightarrow{p} \theta_0$ | Identifiability |
| **Asymptotic Normality** | $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, I^{-1})$ | Regularity conditions |
| **Efficiency** | MLE has smallest asymptotic variance | Regularity conditions |
| **Invariance** | $\widehat{g(\theta)} = g(\hat{\theta})$ | Always holds |
| **Convergence Rate** | $O_p(n^{-1/2})$ | Regularity conditions |

## Connection to Deep Learning

These asymptotic properties explain several phenomena in deep learning:

1. **More data helps**: Consistency and $\sqrt{n}$ rate justify using large datasets

2. **Confidence estimates**: Asymptotic normality enables uncertainty quantification

3. **Hyperparameter transfer**: If one architecture works, scaled versions should too

4. **Overfitting transition**: As model capacity increases relative to data, MLE properties may fail

## Exercises

1. **Prove** that the MLE for uniform $[0, \theta]$ is consistent but not asymptotically normal (violates regularity)

2. **Calculate** the sample size needed to estimate a Bernoulli $p$ within $\pm 0.02$ with 95% confidence

3. **Implement** a simulation comparing Wald vs. profile likelihood intervals across different sample sizes

4. **Show** that for Normal mean estimation, the sample mean achieves exact (not just asymptotic) efficiency

## References

- Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd Edition, Chapter 10
- van der Vaart, A. W. (1998). *Asymptotic Statistics*
- Lehmann, E. L. & Casella, G. (1998). *Theory of Point Estimation*, 2nd Edition
