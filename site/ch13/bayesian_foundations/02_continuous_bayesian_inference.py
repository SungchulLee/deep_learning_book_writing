"""
Bayesian Inference - Module 2: Continuous Bayesian Inference
Level: Beginner-Intermediate
Topics: Continuous parameters, Beta-Binomial model, Normal inference, posterior visualization

This module extends Bayesian inference to continuous parameter spaces, where
we work with probability density functions instead of discrete probabilities.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# SECTION 1: CONTINUOUS BAYES' THEOREM
# ============================================================================

"""
BAYES' THEOREM FOR CONTINUOUS PARAMETERS:

When the parameter θ is continuous, Bayes' theorem becomes:

p(θ|D) = p(D|θ)p(θ) / p(D)

Where:
- p(θ|D) is the posterior density
- p(D|θ) is the likelihood function
- p(θ) is the prior density
- p(D) = ∫ p(D|θ)p(θ)dθ is the marginal likelihood (evidence)

Often we work with the proportional form:
p(θ|D) ∝ p(D|θ)p(θ)

and normalize afterward:
p(θ|D) = p(D|θ)p(θ) / ∫ p(D|θ')p(θ')dθ'
"""

def posterior_continuous(theta_grid, prior, likelihood, normalize=True):
    """
    Compute posterior distribution for continuous parameter on a grid.
    
    Parameters:
    -----------
    theta_grid : array-like
        Grid of parameter values to evaluate
    prior : array-like or callable
        Prior density evaluated at theta_grid, or a function p(theta)
    likelihood : array-like or callable
        Likelihood evaluated at theta_grid, or a function L(theta)
    normalize : bool
        Whether to normalize the posterior
    
    Returns:
    --------
    posterior : numpy array
        Posterior density evaluated at theta_grid
    """
    theta_grid = np.asarray(theta_grid)
    
    # Evaluate prior if it's a function
    if callable(prior):
        prior_values = np.array([prior(theta) for theta in theta_grid])
    else:
        prior_values = np.asarray(prior)
    
    # Evaluate likelihood if it's a function
    if callable(likelihood):
        likelihood_values = np.array([likelihood(theta) for theta in theta_grid])
    else:
        likelihood_values = np.asarray(likelihood)
    
    # Compute unnormalized posterior
    posterior = prior_values * likelihood_values
    
    # Normalize using numerical integration (trapezoidal rule)
    if normalize:
        evidence = np.trapz(posterior, theta_grid)
        if evidence > 0:
            posterior = posterior / evidence
        else:
            raise ValueError("Evidence is zero - cannot normalize")
    
    return posterior

# ============================================================================
# EXAMPLE 1: BETA-BINOMIAL MODEL (Coin Flipping with Continuous θ)
# ============================================================================

"""
BETA-BINOMIAL MODEL:

This is the canonical example of Bayesian inference with a continuous parameter.

Problem: Estimate the probability θ of heads for a coin.

Prior: Beta(α, β) distribution
  p(θ) = θ^(α-1)(1-θ)^(β-1) / B(α,β)
  
  where B(α,β) is the beta function.
  
  - α, β > 0 are hyperparameters
  - Mean: α/(α+β)
  - If α=β=1, this is a uniform prior (all θ values equally likely)
  - If α=β=0.5, this is the Jeffreys prior (non-informative)
  - If α=β>1, this concentrates probability near θ=0.5

Likelihood: Binomial
  p(k heads|n flips, θ) = C(n,k) θ^k (1-θ)^(n-k)

Posterior: Beta(α+k, β+n-k)
  This is a conjugate relationship - posterior is in the same family as prior!
  
  Updated parameters:
  - α_post = α_prior + k (number of heads)
  - β_post = β_prior + (n-k) (number of tails)
"""

def beta_binomial_inference(n_heads, n_tails, prior_alpha=1, prior_beta=1, visualize=True):
    """
    Perform Bayesian inference for coin flip probability using Beta-Binomial model.
    
    Parameters:
    -----------
    n_heads : int
        Number of observed heads
    n_tails : int
        Number of observed tails
    prior_alpha, prior_beta : float
        Parameters of Beta prior distribution
    visualize : bool
        Whether to create visualization
    
    Returns:
    --------
    posterior_dist : scipy.stats.beta
        Posterior Beta distribution
    """
    print("="*70)
    print("EXAMPLE 1: BETA-BINOMIAL MODEL FOR COIN FLIPPING")
    print("="*70)
    
    n_total = n_heads + n_tails
    print(f"\nObserved data: {n_heads} heads, {n_tails} tails (n={n_total})")
    print(f"Prior: Beta(α={prior_alpha}, β={prior_beta})")
    
    # Prior distribution
    prior_dist = stats.beta(prior_alpha, prior_beta)
    prior_mean = prior_dist.mean()
    prior_std = prior_dist.std()
    
    print(f"\nPrior statistics:")
    print(f"  Mean: {prior_mean:.4f}")
    print(f"  Std:  {prior_std:.4f}")
    print(f"  Mode: {(prior_alpha-1)/(prior_alpha+prior_beta-2):.4f}" if prior_alpha>1 and prior_beta>1 else "  Mode: undefined (uniform prior)")
    
    # Posterior parameters (conjugate update)
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # Posterior distribution
    posterior_dist = stats.beta(post_alpha, post_beta)
    post_mean = posterior_dist.mean()
    post_std = posterior_dist.std()
    post_mode = (post_alpha-1)/(post_alpha+post_beta-2) if post_alpha>1 and post_beta>1 else post_alpha/(post_alpha+post_beta)
    
    print(f"\nPosterior: Beta(α={post_alpha}, β={post_beta})")
    print(f"Posterior statistics:")
    print(f"  Mean: {post_mean:.4f}")
    print(f"  Std:  {post_std:.4f}")
    print(f"  Mode: {post_mode:.4f}")
    
    # Maximum Likelihood Estimate (for comparison)
    mle = n_heads / n_total
    print(f"\nMaximum Likelihood Estimate: {mle:.4f}")
    print(f"Posterior Mean (Bayesian estimate): {post_mean:.4f}")
    print(f"Difference: {abs(post_mean - mle):.4f}")
    
    if visualize:
        # Create grid for plotting
        theta = np.linspace(0, 1, 1000)
        
        # Evaluate distributions
        prior_pdf = prior_dist.pdf(theta)
        posterior_pdf = posterior_dist.pdf(theta)
        
        # Likelihood function (proportional to binomial)
        likelihood = theta**n_heads * (1-theta)**n_tails
        likelihood_normalized = likelihood / np.trapz(likelihood, theta)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Prior
        axes[0, 0].plot(theta, prior_pdf, 'b-', linewidth=2, label=f'Beta({prior_alpha}, {prior_beta})')
        axes[0, 0].axvline(prior_mean, color='b', linestyle='--', alpha=0.7, label=f'Mean = {prior_mean:.3f}')
        axes[0, 0].fill_between(theta, prior_pdf, alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('θ (Probability of Heads)', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title('Prior Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Likelihood
        axes[0, 1].plot(theta, likelihood_normalized, 'g-', linewidth=2, label=f'{n_heads}H, {n_tails}T')
        axes[0, 1].axvline(mle, color='g', linestyle='--', alpha=0.7, label=f'MLE = {mle:.3f}')
        axes[0, 1].fill_between(theta, likelihood_normalized, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('θ (Probability of Heads)', fontsize=11)
        axes[0, 1].set_ylabel('Relative Likelihood', fontsize=11)
        axes[0, 1].set_title('Likelihood Function', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Posterior
        axes[1, 0].plot(theta, posterior_pdf, 'r-', linewidth=2, label=f'Beta({post_alpha}, {post_beta})')
        axes[1, 0].axvline(post_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean = {post_mean:.3f}')
        axes[1, 0].axvline(post_mode, color='darkred', linestyle=':', alpha=0.7, label=f'Mode = {post_mode:.3f}')
        axes[1, 0].fill_between(theta, posterior_pdf, alpha=0.3, color='red')
        axes[1, 0].set_xlabel('θ (Probability of Heads)', fontsize=11)
        axes[1, 0].set_ylabel('Density', fontsize=11)
        axes[1, 0].set_title('Posterior Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: All together
        axes[1, 1].plot(theta, prior_pdf, 'b-', linewidth=2, alpha=0.7, label='Prior')
        axes[1, 1].plot(theta, likelihood_normalized, 'g-', linewidth=2, alpha=0.7, label='Likelihood')
        axes[1, 1].plot(theta, posterior_pdf, 'r-', linewidth=3, label='Posterior')
        axes[1, 1].axvline(post_mean, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('θ (Probability of Heads)', fontsize=11)
        axes[1, 1].set_ylabel('Density', fontsize=11)
        axes[1, 1].set_title('Prior, Likelihood, and Posterior', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'beta_binomial_{n_heads}H_{n_tails}T.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return posterior_dist

# ============================================================================
# EXAMPLE 2: EFFECT OF DIFFERENT PRIORS
# ============================================================================

def compare_priors(n_heads, n_tails):
    """
    Compare the effect of different prior beliefs on posterior inference.
    
    This demonstrates how prior information affects inference, especially
    with small sample sizes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: COMPARING DIFFERENT PRIORS")
    print("="*70)
    
    n_total = n_heads + n_tails
    mle = n_heads / n_total
    
    print(f"\nData: {n_heads} heads, {n_tails} tails (n={n_total})")
    print(f"MLE: {mle:.4f}")
    
    # Define different priors
    priors = {
        'Uniform (Uninformative)': (1, 1),
        'Jeffreys (Non-informative)': (0.5, 0.5),
        'Weak Belief (Fair)': (2, 2),
        'Strong Belief (Fair)': (10, 10),
        'Skeptical (Biased)': (2, 8),
    }
    
    theta = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    print("\nPosterior Means for Different Priors:")
    print("-" * 60)
    
    for idx, (name, (alpha, beta)) in enumerate(priors.items()):
        # Prior
        prior_dist = stats.beta(alpha, beta)
        prior_pdf = prior_dist.pdf(theta)
        
        # Posterior
        post_alpha = alpha + n_heads
        post_beta = beta + n_tails
        posterior_dist = stats.beta(post_alpha, post_beta)
        posterior_pdf = posterior_dist.pdf(theta)
        post_mean = posterior_dist.mean()
        
        print(f"{name:30s}: {post_mean:.4f} (diff from MLE: {abs(post_mean-mle):.4f})")
        
        # Plot
        if idx < len(axes):
            axes[idx].plot(theta, prior_pdf, 'b--', linewidth=2, alpha=0.6, label='Prior')
            axes[idx].plot(theta, posterior_pdf, 'r-', linewidth=2, label='Posterior')
            axes[idx].axvline(mle, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'MLE={mle:.3f}')
            axes[idx].axvline(post_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Post Mean={post_mean:.3f}')
            axes[idx].set_xlabel('θ', fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].set_title(f'{name}\nBeta({alpha}, {beta}) → Beta({post_alpha}, {post_beta})', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
    
    # Remove extra subplot
    if len(priors) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('prior_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Insight:")
    print("With small sample sizes, the prior has a strong effect on the posterior.")
    print("With large sample sizes, the likelihood dominates and different priors")
    print("converge to similar posteriors.")

# ============================================================================
# EXAMPLE 3: SEQUENTIAL UPDATING WITH CONTINUOUS PARAMETERS
# ============================================================================

def sequential_beta_binomial(flip_sequence, prior_alpha=1, prior_beta=1):
    """
    Demonstrate sequential Bayesian updating with Beta-Binomial model.
    
    The posterior from one update becomes the prior for the next update.
    This is equivalent to updating all at once with all the data.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: SEQUENTIAL UPDATING (CONTINUOUS)")
    print("="*70)
    
    print(f"\nFlip sequence: {' '.join(flip_sequence)}")
    print(f"Initial prior: Beta({prior_alpha}, {prior_beta})")
    
    # Track parameters over time
    alpha_history = [prior_alpha]
    beta_history = [prior_beta]
    
    current_alpha = prior_alpha
    current_beta = prior_beta
    
    # Process each flip
    for i, flip in enumerate(flip_sequence, 1):
        if flip == 'H':
            current_alpha += 1
        else:  # flip == 'T'
            current_beta += 1
        
        alpha_history.append(current_alpha)
        beta_history.append(current_beta)
        
        # Print update
        dist = stats.beta(current_alpha, current_beta)
        print(f"After flip {i} ({flip}): Beta({current_alpha}, {current_beta}), Mean = {dist.mean():.4f}")
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Evolution of posterior
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_history)))
    
    for i, (alpha, beta) in enumerate(zip(alpha_history, beta_history)):
        dist = stats.beta(alpha, beta)
        pdf = dist.pdf(theta)
        label = 'Initial Prior' if i == 0 else f'After {i} flips'
        alpha_val = 1.0 if i == 0 or i == len(alpha_history)-1 else 0.3
        linewidth = 3 if i == len(alpha_history)-1 else 1
        ax1.plot(theta, pdf, color=colors[i], alpha=alpha_val, linewidth=linewidth, label=label if i in [0, len(alpha_history)-1] else '')
    
    ax1.set_xlabel('θ (Probability of Heads)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Evolution of Posterior Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Posterior mean over time
    means = [stats.beta(a, b).mean() for a, b in zip(alpha_history, beta_history)]
    ax2.plot(range(len(means)), means, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Number of Flips', fontsize=12)
    ax2.set_ylabel('Posterior Mean of θ', fontsize=12)
    ax2.set_title('Convergence of Estimate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add flip labels
    ax2.set_xticks(range(len(means)))
    ax2.set_xticklabels(['Prior'] + list(flip_sequence), rotation=45)
    
    plt.tight_layout()
    plt.savefig('sequential_continuous.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return alpha_history, beta_history

# ============================================================================
# EXAMPLE 4: NORMAL-NORMAL MODEL (Known Variance)
# ============================================================================

"""
NORMAL-NORMAL MODEL:

Problem: Estimate the mean μ of a normal distribution with known variance σ².

Prior: N(μ₀, σ₀²)
Likelihood: N(μ, σ²) for n observations with sample mean x̄
Posterior: N(μₙ, σₙ²)

Where:
  μₙ = (σ²μ₀ + nσ₀²x̄) / (σ² + nσ₀²)
  σₙ² = σ²σ₀² / (σ² + nσ₀²)

This is another conjugate relationship. The posterior mean is a weighted
average of the prior mean and the sample mean, with weights proportional
to the precision (inverse variance).
"""

def normal_normal_inference(data, prior_mean, prior_std, known_std, visualize=True):
    """
    Bayesian inference for normal mean with known variance.
    
    Parameters:
    -----------
    data : array-like
        Observed data points
    prior_mean : float
        Mean of prior distribution
    prior_std : float
        Standard deviation of prior distribution
    known_std : float
        Known standard deviation of the data distribution
    visualize : bool
        Whether to create visualization
    
    Returns:
    --------
    posterior_dist : scipy.stats.norm
        Posterior normal distribution
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: NORMAL-NORMAL MODEL (Known Variance)")
    print("="*70)
    
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    print(f"\nObserved data: n = {n}")
    print(f"  Sample mean: {sample_mean:.4f}")
    print(f"  Sample std:  {sample_std:.4f}")
    print(f"\nPrior: N(μ={prior_mean:.2f}, σ={prior_std:.2f})")
    print(f"Known data std: σ={known_std:.2f}")
    
    # Prior
    prior_variance = prior_std ** 2
    data_variance = known_std ** 2
    
    # Posterior parameters (conjugate update)
    # Precision-weighted average
    prior_precision = 1 / prior_variance
    data_precision = n / data_variance
    
    posterior_precision = prior_precision + data_precision
    posterior_variance = 1 / posterior_precision
    posterior_std = np.sqrt(posterior_variance)
    
    posterior_mean = (prior_precision * prior_mean + data_precision * sample_mean) / posterior_precision
    
    print(f"\nPosterior: N(μ={posterior_mean:.4f}, σ={posterior_std:.4f})")
    print(f"\nWeight on prior mean: {prior_precision/posterior_precision:.4f}")
    print(f"Weight on sample mean: {data_precision/posterior_precision:.4f}")
    
    # Create distributions
    prior_dist = stats.norm(prior_mean, prior_std)
    posterior_dist = stats.norm(posterior_mean, posterior_std)
    
    if visualize:
        # Create grid for plotting
        x_min = min(prior_mean - 4*prior_std, sample_mean - 4*known_std)
        x_max = max(prior_mean + 4*prior_std, sample_mean + 4*known_std)
        x = np.linspace(x_min, x_max, 1000)
        
        # PDFs
        prior_pdf = prior_dist.pdf(x)
        posterior_pdf = posterior_dist.pdf(x)
        
        # Likelihood (proportional to N(sample_mean, known_std/sqrt(n)))
        likelihood_std = known_std / np.sqrt(n)
        likelihood = stats.norm(sample_mean, likelihood_std).pdf(x)
        likelihood_normalized = likelihood / np.max(likelihood) * np.max(prior_pdf)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Prior, Likelihood, Posterior
        ax1.plot(x, prior_pdf, 'b-', linewidth=2, label=f'Prior N({prior_mean:.1f}, {prior_std:.1f})')
        ax1.plot(x, likelihood_normalized, 'g-', linewidth=2, label=f'Likelihood (n={n})')
        ax1.plot(x, posterior_pdf, 'r-', linewidth=3, label=f'Posterior N({posterior_mean:.2f}, {posterior_std:.2f})')
        ax1.axvline(sample_mean, color='green', linestyle='--', alpha=0.5, label=f'Sample mean={sample_mean:.2f}')
        ax1.axvline(posterior_mean, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('μ', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Bayesian Inference for Normal Mean', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Data and distributions
        ax2.hist(data, bins=15, density=True, alpha=0.5, color='gray', label='Observed data')
        ax2.axvline(sample_mean, color='green', linestyle='--', linewidth=2, label=f'Sample mean={sample_mean:.2f}')
        ax2.axvline(posterior_mean, color='red', linestyle='--', linewidth=2, label=f'Posterior mean={posterior_mean:.2f}')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Data Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('normal_normal_inference.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return posterior_dist

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 2: CONTINUOUS BAYESIAN INFERENCE")
    print("="*70)
    
    # Example 1: Beta-Binomial with moderate data
    posterior1 = beta_binomial_inference(n_heads=15, n_tails=5, prior_alpha=1, prior_beta=1)
    
    # Example 2: Compare different priors
    compare_priors(n_heads=7, n_tails=3)
    
    # Example 3: Sequential updating
    flip_seq = ['H', 'H', 'T', 'H', 'H', 'H', 'T', 'T', 'H', 'H']
    alpha_hist, beta_hist = sequential_beta_binomial(flip_seq, prior_alpha=1, prior_beta=1)
    
    # Example 4: Normal-Normal inference
    # Simulate data from N(5, 2)
    np.random.seed(42)
    true_mean = 5.0
    true_std = 2.0
    data = np.random.normal(true_mean, true_std, size=20)
    
    posterior_normal = normal_normal_inference(
        data=data,
        prior_mean=0.0,
        prior_std=5.0,
        known_std=2.0
    )
    
    print("\n" + "="*70)
    print("MODULE 2 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Continuous parameters use probability densities, not probabilities")
    print("2. Beta-Binomial is conjugate: Beta prior + Binomial data → Beta posterior")
    print("3. Different priors affect inference differently with small samples")
    print("4. Sequential updating is equivalent to batch updating")
    print("5. Normal-Normal is conjugate: posterior mean is weighted average")
    print("\nNext: Module 3 - Conjugate Priors")
    print("="*70)

# ============================================================================
# EXERCISES FOR STUDENTS
# ============================================================================

"""
EXERCISE 1: Strong vs Weak Priors
Compare the effect of Beta(1,1) vs Beta(100,100) priors when you observe 
5 heads in 10 flips. How much does the strong prior influence the posterior?

EXERCISE 2: Prior Sensitivity
For the medical test example from Module 1, reformulate it with continuous
test accuracy parameter and use a Beta prior. How does this change the inference?

EXERCISE 3: Convergence
Generate a long sequence of coin flips from a fair coin. Show how the posterior
converges to the true parameter as n increases, for different prior beliefs.

EXERCISE 4: Normal-Normal with Unknown Variance
Research the Normal-Inverse-Gamma conjugate family for the case where both
mean and variance are unknown. Implement inference for this model.

EXERCISE 5: Real Data Application
Find a dataset with binary outcomes (e.g., customer conversion, email clicks).
Use Beta-Binomial model to estimate the success rate and compute credible intervals.
"""
