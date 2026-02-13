"""
01_gibbs.py

GIBBS SAMPLING: COMPREHENSIVE TUTORIAL
======================================

Learning Objectives:
-------------------
1. Understand Gibbs sampling as a special case of MCMC
2. Learn when Gibbs sampling is applicable and efficient
3. See connections to conditional distributions
4. Practice with 2D Gaussian and mixture models

Mathematical Foundation:
-----------------------
Goal:
    Sample from a joint distribution p(x₁, x₂, ..., xₙ) when direct sampling
    is difficult, but sampling from conditional distributions p(xᵢ | x₋ᵢ) is easy.

Gibbs Sampling Algorithm:
-------------------------
1. Initialize x⁽⁰⁾ = (x₁⁽⁰⁾, x₂⁽⁰⁾, ..., xₙ⁽⁰⁾)
2. For t = 1, 2, ...
   a. Sample x₁⁽ᵗ⁾ ~ p(x₁ | x₂⁽ᵗ⁻¹⁾, x₃⁽ᵗ⁻¹⁾, ..., xₙ⁽ᵗ⁻¹⁾)
   b. Sample x₂⁽ᵗ⁾ ~ p(x₂ | x₁⁽ᵗ⁾, x₃⁽ᵗ⁻¹⁾, ..., xₙ⁽ᵗ⁻¹⁾)
   c. ...
   d. Sample xₙ⁽ᵗ⁾ ~ p(xₙ | x₁⁽ᵗ⁾, x₂⁽ᵗ⁾, ..., xₙ₋₁⁽ᵗ⁾)

Why It Works:
-------------
- Gibbs is a special case of Metropolis-Hastings with acceptance probability = 1
- The proposal distribution is q(x' | x) = p(xᵢ' | x₋ᵢ)
- This choice ensures all proposals are accepted (!)
- Satisfies detailed balance, converges to target distribution

Advantages:
-----------
+ No tuning parameters needed
+ 100% acceptance rate
+ Works well for conjugate models
+ Efficient when conditional distributions are simple

Disadvantages:
--------------
- Requires tractable conditional distributions
- Can be slow if variables are highly correlated
- May get stuck in multimodal distributions

Connection to Diffusion Models:
-------------------------------
- Both use iterative refinement
- Gibbs updates one component at a time (like denoising one part)
- Understanding Gibbs helps with understanding conditional generation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, invgamma
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: GIBBS FOR 2D GAUSSIAN (BEGINNER)
# =============================================================================

def gibbs_2d_gaussian():
    """
    Example 1: Gibbs Sampling for 2D Gaussian
    =========================================
    
    Target: Bivariate normal distribution
    p(x₁, x₂) = N([μ₁, μ₂], Σ)
    
    Conditional distributions (key insight!):
    p(x₁ | x₂) = N(μ₁ + ρ(σ₁/σ₂)(x₂ - μ₂), σ₁²(1 - ρ²))
    p(x₂ | x₁) = N(μ₂ + ρ(σ₂/σ₁)(x₁ - μ₁), σ₂²(1 - ρ²))
    
    where ρ is the correlation coefficient.
    
    Algorithm:
    1. Start with x₁⁽⁰⁾, x₂⁽⁰⁾
    2. Sample x₁⁽ᵗ⁾ ~ p(x₁ | x₂⁽ᵗ⁻¹⁾)
    3. Sample x₂⁽ᵗ⁾ ~ p(x₂ | x₁⁽ᵗ⁾)
    4. Repeat
    
    Note: Each step is exact sampling, no acceptance/rejection needed!
    """
    print("=" * 70)
    print("EXAMPLE 1: Gibbs Sampling for 2D Gaussian")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target distribution parameters
    mu = np.array([2.0, 3.0])
    sigma = np.array([1.5, 2.0])
    rho = 0.7  # Correlation coefficient
    
    # Covariance matrix
    cov = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2]
    ])
    
    print(f"\nTarget distribution: N(μ={mu}, Σ)")
    print(f"Correlation: ρ = {rho}")
    
    # Conditional distribution parameters
    def conditional_x1(x2):
        """p(x₁ | x₂)"""
        mean = mu[0] + rho * (sigma[0] / sigma[1]) * (x2 - mu[1])
        std = sigma[0] * np.sqrt(1 - rho**2)
        return mean, std
    
    def conditional_x2(x1):
        """p(x₂ | x₁)"""
        mean = mu[1] + rho * (sigma[1] / sigma[0]) * (x1 - mu[0])
        std = sigma[1] * np.sqrt(1 - rho**2)
        return mean, std
    
    # Gibbs sampler
    def gibbs_sample(n_samples, x_init):
        """Run Gibbs sampling for 2D Gaussian"""
        samples = np.zeros((n_samples, 2))
        samples[0] = x_init
        
        for t in range(1, n_samples):
            # Sample x₁ given x₂
            mean_x1, std_x1 = conditional_x1(samples[t-1, 1])
            samples[t, 0] = np.random.normal(mean_x1, std_x1)
            
            # Sample x₂ given x₁
            mean_x2, std_x2 = conditional_x2(samples[t, 0])
            samples[t, 1] = np.random.normal(mean_x2, std_x2)
        
        return samples
    
    # Run Gibbs sampling
    n_samples = 5000
    x_init = np.array([0.0, 0.0])
    samples = gibbs_sample(n_samples, x_init)
    
    # Compute statistics
    sample_mean = samples[1000:].mean(axis=0)  # Discard burn-in
    sample_cov = np.cov(samples[1000:].T)
    
    print(f"\nGibbs sampling with {n_samples} iterations")
    print(f"\nEstimated mean: [{sample_mean[0]:.3f}, {sample_mean[1]:.3f}]")
    print(f"True mean:      [{mu[0]:.3f}, {mu[1]:.3f}]")
    print(f"\nEstimated correlation: {sample_cov[0,1] / (np.sqrt(sample_cov[0,0] * sample_cov[1,1])):.3f}")
    print(f"True correlation:      {rho:.3f}")
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 2D scatter with contours
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Show trajectory for first 100 samples
    n_show = 100
    ax1.plot(samples[:n_show, 0], samples[:n_show, 1], 
            'b-', alpha=0.3, linewidth=0.5, label='Trajectory')
    ax1.scatter(samples[:n_show, 0], samples[:n_show, 1], 
               c=np.arange(n_show), cmap='viridis', s=30, alpha=0.7)
    ax1.plot(samples[0, 0], samples[0, 1], 'ro', markersize=10, label='Start')
    
    # True distribution contours
    x1_grid = np.linspace(mu[0] - 3*sigma[0], mu[0] + 3*sigma[0], 100)
    x2_grid = np.linspace(mu[1] - 3*sigma[1], mu[1] + 3*sigma[1], 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    pos = np.dstack((X1, X2))
    rv = multivariate_normal(mu, cov)
    ax1.contour(X1, X2, rv.pdf(pos), levels=5, colors='red', alpha=0.6, linewidths=2)
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title(f'Gibbs Sampling Trajectory (first {n_show} samples)', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Marginal x₁
    ax2 = fig.add_subplot(gs[2, 0:2])
    ax2.hist(samples[1000:, 0], bins=50, density=True, alpha=0.7, 
            edgecolor='black', label='Gibbs samples')
    x1_range = np.linspace(mu[0] - 3*sigma[0], mu[0] + 3*sigma[0], 200)
    ax2.plot(x1_range, norm.pdf(x1_range, mu[0], sigma[0]), 
            'r-', linewidth=2, label='True p(x₁)')
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Marginal Distribution p(x₁)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Marginal x₂
    ax3 = fig.add_subplot(gs[0:2, 2])
    ax3.hist(samples[1000:, 1], bins=50, density=True, alpha=0.7,
            edgecolor='black', orientation='horizontal', label='Gibbs samples')
    x2_range = np.linspace(mu[1] - 3*sigma[1], mu[1] + 3*sigma[1], 200)
    ax3.plot(norm.pdf(x2_range, mu[1], sigma[1]), x2_range,
            'r-', linewidth=2, label='True p(x₂)')
    ax3.set_ylabel('x₂', fontsize=12)
    ax3.set_xlabel('Density', fontsize=12)
    ax3.set_title('Marginal Distribution p(x₂)', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Trace plots
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.plot(samples[:500, 0], linewidth=0.5, alpha=0.7, label='x₁')
    ax4.plot(samples[:500, 1], linewidth=0.5, alpha=0.7, label='x₂')
    ax4.axhline(mu[0], color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axhline(mu[1], color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Iteration', fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.set_title('Trace Plot (first 500)', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    
    fig_path = os.path.join(os.path.dirname(__file__), 'gibbs_2d_gaussian.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: gibbs_2d_gaussian.png")
    print("\nKey Observations:")
    print("  • 100% acceptance rate (no rejections!)")
    print("  • Samples converge to true distribution")
    print("  • Alternating updates create characteristic pattern")
    print("  • Efficient when conditionals are known")


# =============================================================================
# PART 2: GIBBS VS METROPOLIS COMPARISON
# =============================================================================

def compare_gibbs_metropolis():
    """
    Comparison: Gibbs vs Metropolis for Same Target
    ===============================================
    
    Shows:
    - Gibbs: 100% acceptance, updates one variable at a time
    - Metropolis: <100% acceptance, updates all variables simultaneously
    - Trade-offs in efficiency and applicability
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Gibbs vs Metropolis")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target: Same 2D Gaussian as before
    mu = np.array([1.0, 2.0])
    rho = 0.6
    sigma = np.array([1.0, 1.5])
    cov = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2]
    ])
    
    # Gibbs sampler (from before)
    def gibbs_sample(n_samples):
        samples = np.zeros((n_samples, 2))
        samples[0] = [0.0, 0.0]
        
        for t in range(1, n_samples):
            # x₁ | x₂
            mean_x1 = mu[0] + rho * (sigma[0] / sigma[1]) * (samples[t-1, 1] - mu[1])
            std_x1 = sigma[0] * np.sqrt(1 - rho**2)
            samples[t, 0] = np.random.normal(mean_x1, std_x1)
            
            # x₂ | x₁
            mean_x2 = mu[1] + rho * (sigma[1] / sigma[0]) * (samples[t, 0] - mu[0])
            std_x2 = sigma[1] * np.sqrt(1 - rho**2)
            samples[t, 1] = np.random.normal(mean_x2, std_x2)
        
        return samples
    
    # Metropolis sampler
    def metropolis_sample(n_samples, proposal_std):
        samples = np.zeros((n_samples, 2))
        samples[0] = [0.0, 0.0]
        n_accepted = 0
        
        rv = multivariate_normal(mu, cov)
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            x_proposal = x_current + np.random.randn(2) * proposal_std
            
            # Acceptance ratio
            alpha = min(1.0, rv.pdf(x_proposal) / rv.pdf(x_current))
            
            if np.random.rand() < alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Run both samplers
    n_samples = 5000
    gibbs_samples = gibbs_sample(n_samples)
    metro_samples, acc_rate = metropolis_sample(n_samples, 0.5)
    
    print(f"\nBoth samplers run for {n_samples} iterations")
    print(f"\nGibbs:")
    print(f"  Acceptance rate: 100%")
    print(f"  Mean: [{gibbs_samples[1000:].mean(axis=0)[0]:.3f}, {gibbs_samples[1000:].mean(axis=0)[1]:.3f}]")
    
    print(f"\nMetropolis:")
    print(f"  Acceptance rate: {acc_rate:.1%}")
    print(f"  Mean: [{metro_samples[1000:].mean(axis=0)[0]:.3f}, {metro_samples[1000:].mean(axis=0)[1]:.3f}]")
    
    print(f"\nTrue mean: [{mu[0]:.3f}, {mu[1]:.3f}]")
    
    # Compute autocorrelation
    def autocorr(x, lag):
        """Compute autocorrelation at given lag"""
        x = x - x.mean()
        return np.correlate(x, x, mode='full')[len(x)-1+lag] / np.correlate(x, x, mode='full')[len(x)-1]
    
    lags = range(0, 50)
    gibbs_autocorr = [autocorr(gibbs_samples[1000:, 0], lag) for lag in lags]
    metro_autocorr = [autocorr(metro_samples[1000:, 0], lag) for lag in lags]
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Gibbs trajectory
    n_show = 200
    axes[0, 0].plot(gibbs_samples[:n_show, 0], gibbs_samples[:n_show, 1], 
                   'b-', alpha=0.3, linewidth=0.5)
    axes[0, 0].scatter(gibbs_samples[:n_show, 0], gibbs_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='viridis', s=20)
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('x₂')
    axes[0, 0].set_title('Gibbs: Trajectory')
    axes[0, 0].grid(alpha=0.3)
    
    # Metropolis trajectory
    axes[0, 1].plot(metro_samples[:n_show, 0], metro_samples[:n_show, 1], 
                   'r-', alpha=0.3, linewidth=0.5)
    axes[0, 1].scatter(metro_samples[:n_show, 0], metro_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='plasma', s=20)
    axes[0, 1].set_xlabel('x₁')
    axes[0, 1].set_ylabel('x₂')
    axes[0, 1].set_title(f'Metropolis: Trajectory (accept={acc_rate:.1%})')
    axes[0, 1].grid(alpha=0.3)
    
    # Autocorrelation comparison
    axes[0, 2].plot(lags, gibbs_autocorr, 'b-', linewidth=2, label='Gibbs')
    axes[0, 2].plot(lags, metro_autocorr, 'r-', linewidth=2, label='Metropolis')
    axes[0, 2].set_xlabel('Lag')
    axes[0, 2].set_ylabel('Autocorrelation')
    axes[0, 2].set_title('Autocorrelation Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Trace plots
    axes[1, 0].plot(gibbs_samples[:500, 0], 'b-', linewidth=0.5, alpha=0.7)
    axes[1, 0].axhline(mu[0], color='r', linestyle='--')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('x₁')
    axes[1, 0].set_title('Gibbs: Trace of x₁')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(metro_samples[:500, 0], 'r-', linewidth=0.5, alpha=0.7)
    axes[1, 1].axhline(mu[0], color='b', linestyle='--')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('x₁')
    axes[1, 1].set_title('Metropolis: Trace of x₁')
    axes[1, 1].grid(alpha=0.3)
    
    # Histogram comparison
    axes[1, 2].hist(gibbs_samples[1000:, 0], bins=40, density=True, alpha=0.5, 
                   label='Gibbs', edgecolor='black')
    axes[1, 2].hist(metro_samples[1000:, 0], bins=40, density=True, alpha=0.5, 
                   label='Metropolis', edgecolor='black')
    x_range = np.linspace(mu[0] - 3*sigma[0], mu[0] + 3*sigma[0], 200)
    axes[1, 2].plot(x_range, norm.pdf(x_range, mu[0], sigma[0]), 
                   'r-', linewidth=2, label='True')
    axes[1, 2].set_xlabel('x₁')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Marginal Distribution Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'gibbs_vs_metropolis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: gibbs_vs_metropolis.png")
    print("\nKey Insights:")
    print("  • Gibbs: Axis-aligned moves (updates one variable at a time)")
    print("  • Metropolis: Diagonal moves (updates all variables together)")
    print("  • Gibbs: No rejections, but may be slower for correlated variables")
    print("  • Metropolis: Has rejections, but can explore more freely")


# =============================================================================
# PART 3: BAYESIAN LINEAR REGRESSION WITH GIBBS
# =============================================================================

def gibbs_bayesian_regression():
    """
    Example 3: Gibbs Sampling for Bayesian Linear Regression
    ========================================================
    
    Model:
    y = Xβ + ε, where ε ~ N(0, σ²I)
    
    Priors:
    β ~ N(0, σ²_β I)
    σ² ~ Inverse-Gamma(a, b)
    
    Conditionals:
    p(β | y, σ²) = N(μ_β, Σ_β)  where
        Σ_β = (XᵀX/σ² + I/σ²_β)⁻¹
        μ_β = Σ_β(Xᵀy/σ²)
    
    p(σ² | y, β) = Inverse-Gamma(a', b') where
        a' = a + n/2
        b' = b + (y - Xβ)ᵀ(y - Xβ)/2
    
    This shows the power of Gibbs: we can sample from these
    conditionals easily using standard distributions!
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Gibbs for Bayesian Linear Regression")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate synthetic data
    n = 100  # Number of observations
    p = 3    # Number of features (including intercept)
    
    X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
    beta_true = np.array([2.0, 1.5, -0.8])
    sigma_true = 0.5
    
    y = X @ beta_true + np.random.randn(n) * sigma_true
    
    print(f"\nData: n={n} observations, p={p} features")
    print(f"True β: {beta_true}")
    print(f"True σ: {sigma_true:.3f}")
    
    # Prior parameters
    sigma_beta = 10.0  # Prior std for β (vague prior)
    a_prior = 2.0      # Inverse-Gamma shape
    b_prior = 1.0      # Inverse-Gamma scale
    
    # Gibbs sampler
    def gibbs_linear_regression(X, y, n_samples, sigma_beta, a_prior, b_prior):
        """
        Gibbs sampling for Bayesian linear regression
        """
        n, p = X.shape
        
        # Storage
        beta_samples = np.zeros((n_samples, p))
        sigma2_samples = np.zeros(n_samples)
        
        # Initialize
        beta = np.zeros(p)
        sigma2 = 1.0
        
        # Precompute XtX for efficiency
        XtX = X.T @ X
        Xty = X.T @ y
        I_p = np.eye(p)
        
        for t in range(n_samples):
            # Sample β | y, σ²
            Sigma_beta_inv = XtX / sigma2 + I_p / (sigma_beta**2)
            Sigma_beta = np.linalg.inv(Sigma_beta_inv)
            mu_beta = Sigma_beta @ (Xty / sigma2)
            beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
            
            # Sample σ² | y, β
            residuals = y - X @ beta
            a_post = a_prior + n / 2
            b_post = b_prior + (residuals @ residuals) / 2
            sigma2 = invgamma.rvs(a_post, scale=b_post)
            
            # Store samples
            beta_samples[t] = beta
            sigma2_samples[t] = sigma2
        
        return beta_samples, np.sqrt(sigma2_samples)
    
    # Run Gibbs sampling
    n_samples = 5000
    beta_samples, sigma_samples = gibbs_linear_regression(
        X, y, n_samples, sigma_beta, a_prior, b_prior
    )
    
    # Burn-in and compute statistics
    burn_in = 1000
    beta_mean = beta_samples[burn_in:].mean(axis=0)
    beta_std = beta_samples[burn_in:].std(axis=0)
    sigma_mean = sigma_samples[burn_in:].mean()
    sigma_std = sigma_samples[burn_in:].std()
    
    print(f"\nGibbs sampling: {n_samples} iterations, burn-in: {burn_in}")
    print("\nPosterior Estimates:")
    print("Parameter | True  | Posterior Mean | Posterior Std")
    print("-" * 60)
    for i in range(p):
        print(f"    β[{i}]   | {beta_true[i]:5.2f} |     {beta_mean[i]:6.3f}     |    {beta_std[i]:5.3f}")
    print(f"     σ    | {sigma_true:5.2f} |     {sigma_mean:6.3f}     |    {sigma_std:5.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Trace plots for β
    for i in range(p):
        ax = axes[0, i]
        ax.plot(beta_samples[:, i], linewidth=0.5, alpha=0.7)
        ax.axhline(beta_true[i], color='r', linestyle='--', linewidth=2, label='True')
        ax.axhline(beta_mean[i], color='g', linestyle='--', linewidth=2, label='Posterior mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'β[{i}]')
        ax.set_title(f'Trace: β[{i}]')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Trace plot for σ
    ax = axes[0, 3]
    ax.plot(sigma_samples, linewidth=0.5, alpha=0.7)
    ax.axhline(sigma_true, color='r', linestyle='--', linewidth=2, label='True')
    ax.axhline(sigma_mean, color='g', linestyle='--', linewidth=2, label='Posterior mean')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('σ')
    ax.set_title('Trace: σ')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Posterior distributions for β
    for i in range(p):
        ax = axes[1, i]
        ax.hist(beta_samples[burn_in:, i], bins=50, density=True, 
               alpha=0.7, edgecolor='black')
        ax.axvline(beta_true[i], color='r', linestyle='--', linewidth=2, label='True')
        ax.axvline(beta_mean[i], color='g', linestyle='--', linewidth=2, label='Posterior mean')
        ax.set_xlabel(f'β[{i}]')
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: β[{i}]')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Posterior distribution for σ
    ax = axes[1, 3]
    ax.hist(sigma_samples[burn_in:], bins=50, density=True, 
           alpha=0.7, edgecolor='black')
    ax.axvline(sigma_true, color='r', linestyle='--', linewidth=2, label='True')
    ax.axvline(sigma_mean, color='g', linestyle='--', linewidth=2, label='Posterior mean')
    ax.set_xlabel('σ')
    ax.set_ylabel('Density')
    ax.set_title('Posterior: σ')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'gibbs_bayesian_regression.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: gibbs_bayesian_regression.png")
    print("\nKey Insights:")
    print("  • Gibbs efficiently samples from posterior")
    print("  • No need to tune proposal distributions")
    print("  • Works great for conjugate models")
    print("  • Each conditional is a standard distribution")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GIBBS SAMPLING: COMPREHENSIVE TUTORIAL")
    print("=" * 70)
    print("\nTopics covered:")
    print("  1. Gibbs sampling for 2D Gaussian (beginner)")
    print("  2. Comparison: Gibbs vs Metropolis")
    print("  3. Bayesian linear regression with Gibbs")
    print("\n" + "=" * 70)
    
    gibbs_2d_gaussian()
    compare_gibbs_metropolis()
    gibbs_bayesian_regression()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • gibbs_2d_gaussian.png")
    print("  • gibbs_vs_metropolis.png")
    print("  • gibbs_bayesian_regression.png")
    print("\nNext steps:")
    print("  • 02_metropolis.py - Metropolis algorithm with symmetric proposals")
    print("  • Try implementing Gibbs for mixture models")
    print("  • Explore blocked Gibbs sampling")
    print("\n" + "=" * 70)
