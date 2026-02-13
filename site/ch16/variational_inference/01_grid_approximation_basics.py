"""
01_grid_approximation_basics.py

GRID APPROXIMATION FOR BAYESIAN INFERENCE: BEGINNER LEVEL
==========================================================

Learning Objectives:
-------------------
1. Understand grid approximation as the simplest numerical posterior computation
2. Implement grid approximation for 1D parameter inference
3. Compare grid approximation with analytical solutions
4. Understand computational costs and limitations
5. See why we need more sophisticated sampling methods

Mathematical Foundation:
-----------------------
Bayes' Theorem:
    p(θ | D) = p(D | θ) p(θ) / p(D)

Central Difficulty:
-------------------
The term p(D), also called the *evidence* or *marginal likelihood*, is:
    p(D) = ∫ p(D | θ) p(θ) dθ

This integral is usually impossible to compute analytically.
But **the key insight is that we never need p(D) explicitly**.

Why p(D) Does NOT Matter:
-------------------------
Bayesian computation only requires the posterior **up to a constant**.
That is:
    p(θ | D) ∝ p(D | θ) p(θ)

In grid approximation, we compute:
    unnormalized_i = p(D | θ_i) * p(θ_i)

Then normalize by:
    p(θ_i | D) = unnormalized_i / Σ_j unnormalized_j

The denominator Σ_j unnormalized_j plays the role of p(D),
but we never compute p(D) symbolically — we only use it to normalize.
This idea generalizes to all Bayesian numerical methods:
    • Grid approx: denominator = sum over grid points
    • Importance sampling: denominator = sum of weights
    • MCMC: acceptance ratios cancel the normalizing constant
    • Diffusion/score-based models: learn ∇ log p(x) (normalizing constant cancels)

Grid Approximation Strategy:
---------------------------
1. Define grid of θ values: θ₁, θ₂, ..., θₙ
2. Compute prior at each point: p(θᵢ)
3. Compute likelihood at each point: p(D | θᵢ)
4. Compute unnormalized posterior: p̃(θᵢ) = p(D | θᵢ) p(θᵢ)
5. Normalize:
       p(θᵢ | D) = p̃(θᵢ) / Σⱼ p̃(θⱼ)

Why This Matters:
----------------
- Easiest numerical method for posterior computation
- Makes Bayes' theorem concrete and computational
- Demonstrates how we avoid computing p(D)
- Useful for teaching and simple 1D examples
- Reveals why high-dimensional inference is hard
- Prepares intuition for:
    • Importance sampling (weighted sums)
    • MCMC/Langevin dynamics (ratios, no normalizing constant)
    • Diffusion models (score = ∇ log p(x), normalizing constant cancels)

Connection to Diffusion Models:
------------------------------
Even advanced generative models rely on this same principle:
we only need the *unnormalized* log-density.

In diffusion:
    reverse drift = f(x,t) - g(t)^2 ∇_x log p_t(x)
Again, no need for p_t(x)'s normalizing constant.

Summary:
--------
Grid approximation teaches:
    Bayesian inference = prior × likelihood (unnormalized)
    Final posterior = normalized unnormalized values
    p(D) never needs to be computed explicitly

This insight is the backbone of all modern Bayesian computation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, beta, gamma, binom, poisson
from scipy.integrate import simpson
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: SIMPLEST EXAMPLE - COIN FLIP WITH GRID APPROXIMATION
# =============================================================================

def example_1_coin_flip_grid():
    """
    Example 1: Coin Flip Inference Using Grid Approximation
    =======================================================
    
    Problem: We flip a coin n times and observe k heads.
             Infer θ = P(heads) using grid approximation.
    
    Mathematical Setup:
    ------------------
    - Parameter: θ ∈ [0, 1]
    - Prior: p(θ) = Beta(α, β) or Uniform[0,1] = Beta(1,1)
    - Likelihood: p(k|θ,n) = Binomial(k; n, θ) = C(n,k) θᵏ (1-θ)ⁿ⁻ᵏ
    - Posterior: p(θ|k,n) ∝ p(k|θ,n) p(θ)
    
    Grid Approximation Algorithm:
    ----------------------------
    1. Create grid: θ = [0.001, 0.002, ..., 0.999]
    2. Evaluate prior: p(θᵢ) for each grid point
    3. Evaluate likelihood: p(k|θᵢ, n) for each grid point
    4. Multiply: unnormalized_posterior = prior × likelihood
    5. Normalize: posterior = unnormalized_posterior / sum(unnormalized_posterior)
    
    Why Start Here?
    --------------
    - We know the analytical answer: Beta(α+k, β+n-k)
    - Can verify our grid approximation works correctly
    - Builds intuition for numerical methods
    """
    print("=" * 70)
    print("EXAMPLE 1: Coin Flip with Grid Approximation")
    print("=" * 70)
    
    # Data: observed 7 heads in 10 flips
    n_flips = 10
    n_heads = 7
    
    # Prior: Beta(1, 1) = Uniform[0, 1]
    prior_alpha = 1
    prior_beta = 1
    
    print(f"\nData: {n_heads} heads in {n_flips} flips")
    print(f"Prior: Beta({prior_alpha}, {prior_beta}) [Uniform]")
    print(f"Analytical Posterior: Beta({prior_alpha + n_heads}, "
          f"{prior_beta + n_flips - n_heads})")
    
    # GRID APPROXIMATION
    # ==================
    
    # Step 1: Create a grid of θ values
    n_grid = 1000  # Number of grid points
    theta_grid = np.linspace(0.001, 0.999, n_grid)  # Avoid 0 and 1 exactly
    grid_width = theta_grid[1] - theta_grid[0]
    
    print(f"\nGrid Approximation Setup:")
    print(f"  Number of grid points: {n_grid}")
    print(f"  Grid range: [{theta_grid[0]:.3f}, {theta_grid[-1]:.3f}]")
    print(f"  Grid spacing: {grid_width:.6f}")
    
    # Step 2: Evaluate prior at each grid point
    # For Beta(1,1), this is just 1 (uniform)
    prior_grid = beta.pdf(theta_grid, prior_alpha, prior_beta)
    
    # Step 3: Evaluate likelihood at each grid point
    # Binomial likelihood: p(k|θ,n) = C(n,k) θᵏ (1-θ)ⁿ⁻ᵏ
    likelihood_grid = binom.pmf(n_heads, n_flips, theta_grid)
    
    # Step 4: Compute unnormalized posterior (prior × likelihood)
    unnormalized_posterior = prior_grid * likelihood_grid
    
    # Step 5: Normalize to get posterior
    # Discrete approximation to: ∫ p(θ|D) dθ = 1
    normalization_constant = np.sum(unnormalized_posterior) * grid_width
    posterior_grid = unnormalized_posterior / normalization_constant
    
    print(f"\nNormalization constant (approximate evidence): {normalization_constant:.6f}")
    
    # Compare with analytical solution
    analytical_posterior = beta.pdf(theta_grid, 
                                   prior_alpha + n_heads, 
                                   prior_beta + n_flips - n_heads)
    
    # Compute posterior mean and std from grid
    posterior_mean_grid = np.sum(theta_grid * posterior_grid * grid_width)
    posterior_var_grid = np.sum((theta_grid - posterior_mean_grid)**2 * 
                                posterior_grid * grid_width)
    posterior_std_grid = np.sqrt(posterior_var_grid)
    
    # Analytical posterior moments for Beta(α+k, β+n-k)
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_flips - n_heads
    analytical_mean = post_alpha / (post_alpha + post_beta)
    analytical_std = np.sqrt(post_alpha * post_beta / 
                            ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)))
    
    print(f"\nPosterior Mean:")
    print(f"  Grid approximation: {posterior_mean_grid:.6f}")
    print(f"  Analytical:         {analytical_mean:.6f}")
    print(f"  Error:              {abs(posterior_mean_grid - analytical_mean):.6e}")
    
    print(f"\nPosterior Std:")
    print(f"  Grid approximation: {posterior_std_grid:.6f}")
    print(f"  Analytical:         {analytical_std:.6f}")
    print(f"  Error:              {abs(posterior_std_grid - analytical_std):.6e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Prior
    ax = axes[0, 0]
    ax.plot(theta_grid, prior_grid, 'b-', linewidth=2, label='Prior: Beta(1,1)')
    ax.fill_between(theta_grid, prior_grid, alpha=0.3)
    ax.set_xlabel('θ (Probability of Heads)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Step 1: Prior Distribution\np(θ) = Beta(1,1) [Uniform]', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 2: Likelihood
    ax = axes[0, 1]
    ax.plot(theta_grid, likelihood_grid, 'g-', linewidth=2, 
           label=f'Likelihood: Binomial({n_heads}|{n_flips}, θ)')
    ax.fill_between(theta_grid, likelihood_grid, alpha=0.3, color='g')
    ax.set_xlabel('θ (Probability of Heads)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Step 2: Likelihood Function\np(k={n_heads}|n={n_flips}, θ)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 3: Unnormalized Posterior
    ax = axes[1, 0]
    ax.plot(theta_grid, unnormalized_posterior, 'orange', linewidth=2,
           label='Prior × Likelihood')
    ax.fill_between(theta_grid, unnormalized_posterior, alpha=0.3, color='orange')
    ax.set_xlabel('θ (Probability of Heads)', fontsize=12)
    ax.set_ylabel('Unnormalized Density', fontsize=12)
    ax.set_title('Step 3: Unnormalized Posterior\np(θ|D) ∝ p(D|θ)p(θ)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: Final Posterior (Grid vs Analytical)
    ax = axes[1, 1]
    ax.plot(theta_grid, posterior_grid, 'r-', linewidth=2, 
           label=f'Grid Approx (n={n_grid})', alpha=0.7)
    ax.plot(theta_grid, analytical_posterior, 'b--', linewidth=2,
           label='Analytical: Beta(8,4)', alpha=0.7)
    ax.fill_between(theta_grid, posterior_grid, alpha=0.2, color='r')
    ax.set_xlabel('θ (Probability of Heads)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Step 4: Final Posterior\np(θ|D) [Normalized]', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add text box with summary
    textstr = f'Grid: mean={posterior_mean_grid:.4f}, std={posterior_std_grid:.4f}\n'
    textstr += f'True: mean={analytical_mean:.4f}, std={analytical_std:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'example1_coin_flip_grid.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: example1_coin_flip_grid.png")
    print("\nKey Insights:")
    print("  ✓ Grid approximation reproduces analytical result accurately")
    print("  ✓ More grid points → better approximation")
    print("  ✓ Computational cost: O(n) for 1D problem")
    print("  ✓ Direct implementation of Bayes' theorem!")


# =============================================================================
# PART 2: EFFECT OF GRID RESOLUTION
# =============================================================================

def example_2_grid_resolution():
    """
    Example 2: How Grid Resolution Affects Accuracy
    ===============================================
    
    Investigates: What happens when we use too few/too many grid points?
    
    Key Questions:
    -------------
    1. How does accuracy change with grid resolution?
    2. What's the computational cost trade-off?
    3. When is grid approximation "good enough"?
    
    Mathematical Analysis:
    --------------------
    - Error in posterior mean: |E_grid[θ] - E_true[θ]|
    - Error decreases approximately as O(1/n) for grid size n
    - But computation time increases linearly with n
    
    Practical Lesson:
    ----------------
    - Too few points: Inaccurate, but fast
    - Too many points: Accurate, but slow
    - Need to balance accuracy and speed
    - In high dimensions, even "few" points is too many!
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Effect of Grid Resolution")
    print("=" * 70)
    
    # Same setup as Example 1
    n_flips = 10
    n_heads = 7
    prior_alpha, prior_beta = 1, 1
    
    # Analytical solution for comparison
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_flips - n_heads
    true_mean = post_alpha / (post_alpha + post_beta)
    true_std = np.sqrt(post_alpha * post_beta / 
                      ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)))
    
    # Test different grid resolutions
    grid_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    mean_errors = []
    std_errors = []
    computation_times = []
    
    print(f"\nTesting grid resolutions from {grid_sizes[0]} to {grid_sizes[-1]} points...")
    
    for n_grid in grid_sizes:
        import time
        start_time = time.time()
        
        # Grid approximation
        theta_grid = np.linspace(0.001, 0.999, n_grid)
        grid_width = theta_grid[1] - theta_grid[0]
        
        prior_grid = beta.pdf(theta_grid, prior_alpha, prior_beta)
        likelihood_grid = binom.pmf(n_heads, n_flips, theta_grid)
        unnormalized = prior_grid * likelihood_grid
        posterior = unnormalized / (np.sum(unnormalized) * grid_width)
        
        # Compute moments
        post_mean = np.sum(theta_grid * posterior * grid_width)
        post_var = np.sum((theta_grid - post_mean)**2 * posterior * grid_width)
        post_std = np.sqrt(post_var)
        
        # Track errors
        mean_errors.append(abs(post_mean - true_mean))
        std_errors.append(abs(post_std - true_std))
        computation_times.append(time.time() - start_time)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean Error vs Grid Size
    ax = axes[0, 0]
    ax.loglog(grid_sizes, mean_errors, 'bo-', linewidth=2, markersize=8, 
             label='Mean Error')
    # Add reference line: O(1/n)
    ref_line = mean_errors[0] * grid_sizes[0] / np.array(grid_sizes)
    ax.loglog(grid_sizes, ref_line, 'r--', linewidth=2, label='O(1/n) reference')
    ax.set_xlabel('Number of Grid Points', fontsize=12)
    ax.set_ylabel('Absolute Error in Mean', fontsize=12)
    ax.set_title('Posterior Mean: Error vs Resolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 2: Std Error vs Grid Size
    ax = axes[0, 1]
    ax.loglog(grid_sizes, std_errors, 'go-', linewidth=2, markersize=8,
             label='Std Error')
    ref_line_std = std_errors[0] * grid_sizes[0] / np.array(grid_sizes)
    ax.loglog(grid_sizes, ref_line_std, 'r--', linewidth=2, label='O(1/n) reference')
    ax.set_xlabel('Number of Grid Points', fontsize=12)
    ax.set_ylabel('Absolute Error in Std', fontsize=12)
    ax.set_title('Posterior Std: Error vs Resolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 3: Computation Time vs Grid Size
    ax = axes[1, 0]
    ax.plot(grid_sizes, np.array(computation_times) * 1000, 'mo-', 
           linewidth=2, markersize=8, label='Computation Time')
    # Linear reference
    linear_ref = computation_times[-1] * 1000 * np.array(grid_sizes) / grid_sizes[-1]
    ax.plot(grid_sizes, linear_ref, 'r--', linewidth=2, label='O(n) reference')
    ax.set_xlabel('Number of Grid Points', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Computational Cost vs Resolution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: Accuracy-Speed Trade-off
    ax = axes[1, 1]
    ax.loglog(np.array(computation_times) * 1000, mean_errors, 'ro-', 
             linewidth=2, markersize=8)
    for i, n in enumerate(grid_sizes[::2]):  # Label every other point
        ax.annotate(f'n={n}', 
                   (computation_times[i*2] * 1000, mean_errors[i*2]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Computation Time (ms)', fontsize=12)
    ax.set_ylabel('Absolute Error in Mean', fontsize=12)
    ax.set_title('Accuracy-Speed Trade-off\n(Lower left = Better)', fontsize=14)
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'grid_resolution_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: grid_resolution_analysis.png")
    print("\nKey Observations:")
    print(f"  • Error decreases approximately as O(1/n)")
    print(f"  • With n=1000: mean error = {mean_errors[6]:.2e}")
    print(f"  • With n=5000: mean error = {mean_errors[-1]:.2e}")
    print(f"  • Computation time grows linearly: O(n)")
    print(f"  • Sweet spot: n=500-1000 for 1D problems")


# =============================================================================
# PART 3: DIFFERENT PRIOR-LIKELIHOOD COMBINATIONS
# =============================================================================

def example_3_normal_inference():
    """
    Example 3: Inference for Normal Distribution Mean
    =================================================
    
    Problem: Observe data from N(μ, σ²) with known σ². Infer μ.
    
    Mathematical Setup:
    ------------------
    - Unknown: μ (mean)
    - Known: σ = 1 (standard deviation)
    - Data: x₁, ..., xₙ ~ N(μ, σ²)
    - Prior: μ ~ N(μ₀, τ²)
    - Likelihood: p(x₁,...,xₙ|μ) = ∏ N(xᵢ|μ, σ²)
    - Posterior: μ|D ~ N(μ_post, σ_post²)  [Conjugate!]
    
    Analytical Solution (Conjugate Gaussian):
    ----------------------------------------
    μ_post = (τ⁻² μ₀ + nσ⁻² x̄) / (τ⁻² + nσ⁻²)
    σ_post² = 1 / (τ⁻² + nσ⁻²)
    
    Why This Example?
    ----------------
    - Shows grid approximation on different problem type
    - Normal-Normal conjugacy is fundamental
    - Connects to Gaussian processes and diffusion models
    - Demonstrates grid approximation for continuous data
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Normal Distribution Mean Inference")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True parameter and data generation
    true_mu = 2.5
    sigma = 1.0  # Known standard deviation
    n_data = 20
    
    # Generate data
    data = np.random.normal(true_mu, sigma, n_data)
    data_mean = data.mean()
    
    # Prior: N(0, 3²)
    prior_mu = 0.0
    prior_tau = 3.0
    
    print(f"\nTrue mean: μ = {true_mu}")
    print(f"Data: n = {n_data}, x̄ = {data_mean:.3f}, σ = {sigma}")
    print(f"Prior: N({prior_mu}, {prior_tau}²)")
    
    # Analytical posterior (conjugate)
    precision_prior = 1 / prior_tau**2
    precision_likelihood = n_data / sigma**2
    precision_post = precision_prior + precision_likelihood
    
    post_mu = (precision_prior * prior_mu + precision_likelihood * data_mean) / precision_post
    post_sigma = np.sqrt(1 / precision_post)
    
    print(f"Analytical Posterior: N({post_mu:.4f}, {post_sigma:.4f})")
    
    # Grid Approximation
    n_grid = 1000
    mu_grid = np.linspace(data_mean - 3*sigma, data_mean + 3*sigma, n_grid)
    grid_width = mu_grid[1] - mu_grid[0]
    
    # Prior
    prior_grid = norm.pdf(mu_grid, prior_mu, prior_tau)
    
    # Likelihood: p(D|μ) = ∏ᵢ N(xᵢ|μ, σ²)
    # Log-likelihood for numerical stability
    log_likelihood_grid = np.zeros(n_grid)
    for x in data:
        log_likelihood_grid += norm.logpdf(x, mu_grid, sigma)
    likelihood_grid = np.exp(log_likelihood_grid - log_likelihood_grid.max())
    
    # Posterior
    unnormalized = prior_grid * likelihood_grid
    posterior_grid = unnormalized / (np.sum(unnormalized) * grid_width)
    
    # Analytical posterior for comparison
    analytical_posterior = norm.pdf(mu_grid, post_mu, post_sigma)
    
    # Compute posterior mean and std from grid
    grid_post_mu = np.sum(mu_grid * posterior_grid * grid_width)
    grid_post_var = np.sum((mu_grid - grid_post_mu)**2 * posterior_grid * grid_width)
    grid_post_sigma = np.sqrt(grid_post_var)
    
    print(f"\nGrid Approximation Posterior: N({grid_post_mu:.4f}, {grid_post_sigma:.4f})")
    print(f"Error in mean: {abs(grid_post_mu - post_mu):.6e}")
    print(f"Error in std:  {abs(grid_post_sigma - post_sigma):.6e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data
    ax = axes[0, 0]
    ax.hist(data, bins=15, density=True, alpha=0.7, color='skyblue',
           edgecolor='black', label='Observed Data')
    x_range = np.linspace(data.min()-1, data.max()+1, 200)
    ax.plot(x_range, norm.pdf(x_range, true_mu, sigma), 'r-', 
           linewidth=2, label=f'True: N({true_mu}, {sigma})')
    ax.axvline(data_mean, color='green', linestyle='--', linewidth=2,
              label=f'Sample mean: {data_mean:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Observed Data (n={n_data})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Prior and Likelihood
    ax = axes[0, 1]
    # Normalize likelihood for visualization
    likelihood_viz = likelihood_grid / likelihood_grid.max() * prior_grid.max()
    ax.plot(mu_grid, prior_grid, 'b-', linewidth=2, label='Prior', alpha=0.7)
    ax.fill_between(mu_grid, prior_grid, alpha=0.2)
    ax.plot(mu_grid, likelihood_viz, 'g-', linewidth=2, 
           label='Likelihood (scaled)', alpha=0.7)
    ax.fill_between(mu_grid, likelihood_viz, alpha=0.2, color='g')
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prior and Likelihood', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 3: Posterior Comparison
    ax = axes[1, 0]
    ax.plot(mu_grid, posterior_grid, 'r-', linewidth=2,
           label='Grid Approximation', alpha=0.7)
    ax.plot(mu_grid, analytical_posterior, 'b--', linewidth=2,
           label='Analytical', alpha=0.7)
    ax.fill_between(mu_grid, posterior_grid, alpha=0.2, color='r')
    ax.axvline(true_mu, color='green', linestyle=':', linewidth=2,
              label=f'True μ={true_mu}')
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Posterior Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: Prior to Posterior Update
    ax = axes[1, 1]
    ax.plot(mu_grid, prior_grid, 'b-', linewidth=2, label='Prior', alpha=0.5)
    ax.plot(mu_grid, posterior_grid, 'r-', linewidth=3, label='Posterior')
    ax.fill_between(mu_grid, prior_grid, alpha=0.2, color='b')
    ax.fill_between(mu_grid, posterior_grid, alpha=0.3, color='r')
    ax.axvline(prior_mu, color='b', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(grid_post_mu, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prior → Posterior Update\n(Uncertainty Reduction)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add text box
    textstr = f'Prior:     μ={prior_mu:.2f}, σ={prior_tau:.2f}\n'
    textstr += f'Posterior: μ={grid_post_mu:.2f}, σ={grid_post_sigma:.2f}\n'
    textstr += f'Uncertainty reduced by {(1 - grid_post_sigma/prior_tau)*100:.1f}%'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.5), family='monospace')
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'normal_mean_inference.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: normal_mean_inference.png")
    print("\nKey Insights:")
    print("  ✓ Grid approximation works for continuous data")
    print("  ✓ Log-likelihood trick prevents numerical underflow")
    print("  ✓ Posterior combines prior and data appropriately")
    print("  ✓ More data → posterior closer to likelihood, less to prior")


# =============================================================================
# PART 4: WHEN GRID APPROXIMATION STARTS TO FAIL
# =============================================================================

def example_4_curse_of_dimensionality_preview():
    """
    Example 4: Preview of the Curse of Dimensionality
    =================================================
    
    Problem: What happens when we go to 2D? 3D? Higher?
    
    The Curse:
    ----------
    - 1D: 1,000 grid points is fine
    - 2D: 1,000 × 1,000 = 1,000,000 points needed
    - 3D: 1,000³ = 1,000,000,000 points needed
    - 10D: 1,000¹⁰ = ∞ (practically impossible!)
    
    Computational Cost:
    ------------------
    For d dimensions with n points per dimension:
    - Total grid points: n^d
    - Memory: O(n^d)
    - Computation: O(n^d)
    
    This is why we need MCMC!
    -------------------------
    MCMC samples don't cover every grid point.
    Instead, they focus on high-probability regions.
    
    Example: For a 10D posterior:
    - Grid: Need 10^30 points (impossible!)
    - MCMC: 10,000 samples is often enough!
    
    Connection to Diffusion:
    -----------------------
    - Images are extremely high dimensional (e.g., 256×256×3 = 196,608 dimensions!)
    - Grid approximation is completely hopeless
    - Need score-based sampling methods (diffusion!)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Curse of Dimensionality")
    print("=" * 70)
    
    # Demonstrate computational requirements
    dimensions = np.arange(1, 11)
    points_per_dim = 100  # Conservative estimate
    
    total_points = points_per_dim ** dimensions
    memory_mb = total_points * 8 / 1e6  # 8 bytes per float64
    
    print(f"\nGrid Approximation Requirements (n={points_per_dim} points per dimension):")
    print("=" * 70)
    print(f"{'Dimension':<12} {'Total Points':<20} {'Memory (MB)':<20} {'Feasible?':<10}")
    print("-" * 70)
    
    for d, pts, mem in zip(dimensions, total_points, memory_mb):
        feasible = "✓ Yes" if mem < 1000 else "✗ No"
        print(f"{d:<12} {pts:<20,.0f} {mem:<20,.2f} {feasible:<10}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Total Points vs Dimension
    ax = axes[0, 0]
    ax.semilogy(dimensions, total_points, 'ro-', linewidth=2, markersize=8)
    ax.axhline(1e9, color='r', linestyle='--', linewidth=2, 
              label='1 billion (practical limit)')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Total Grid Points', fontsize=12)
    ax.set_title(f'Exponential Growth: n^d\n(n={points_per_dim})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 2: Memory Requirements
    ax = axes[0, 1]
    ax.semilogy(dimensions, memory_mb, 'bo-', linewidth=2, markersize=8)
    ax.axhline(1000, color='r', linestyle='--', linewidth=2,
              label='1 GB (reasonable)')
    ax.axhline(100000, color='orange', linestyle='--', linewidth=2,
              label='100 GB (impractical)')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Memory Required (MB)', fontsize=12)
    ax.set_title('Memory Requirements\n(8 bytes per number)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 3: Grid vs MCMC Comparison
    ax = axes[1, 0]
    mcmc_samples = np.full(len(dimensions), 10000)  # Constant MCMC samples
    
    ax.semilogy(dimensions, total_points, 'ro-', linewidth=2, 
               markersize=8, label='Grid Points Needed')
    ax.semilogy(dimensions, mcmc_samples, 'g--', linewidth=3,
               label='MCMC Samples (typical)')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Number of Points/Samples', fontsize=12)
    ax.set_title('Grid vs MCMC: Scaling Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 4: Text explanation
    ax = axes[1, 1]
    ax.axis('off')
    
    explanation = [
        ("Why Grid Approximation Fails", 0.95, 16, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("The Curse of Dimensionality:", 0.82, 13, 'bold', 'darkred'),
        ("• Grid points grow exponentially: n^d", 0.75, 11, 'normal', 'black'),
        ("• 1D: 100 points = 100 evaluations ✓", 0.70, 10, 'normal', 'darkgreen'),
        ("• 2D: 100² = 10,000 evaluations ✓", 0.66, 10, 'normal', 'darkgreen'),
        ("• 3D: 100³ = 1,000,000 evaluations ~", 0.62, 10, 'normal', 'orange'),
        ("• 10D: 100¹⁰ = impossible! ✗", 0.58, 10, 'normal', 'darkred'),
        ("", 0.52, 12, 'normal', 'black'),
        ("Why We Need MCMC:", 0.45, 13, 'bold', 'darkblue'),
        ("• Focuses on high-probability regions", 0.38, 11, 'normal', 'black'),
        ("• Samples scale gracefully with dimension", 0.34, 11, 'normal', 'black'),
        ("• 10,000 samples often sufficient", 0.30, 11, 'normal', 'black'),
        ("• Works in 100D, 1000D, even higher!", 0.26, 11, 'normal', 'black'),
        ("", 0.19, 12, 'normal', 'black'),
        ("For diffusion models:", 0.12, 12, 'bold', 'purple'),
        ("Images are 100,000+ dimensions!", 0.06, 11, 'normal', 'purple'),
        ("Grid approximation is hopeless.", 0.02, 11, 'normal', 'purple'),
    ]
    
    for text, y, size, weight, color in explanation:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'curse_of_dimensionality.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: curse_of_dimensionality.png")
    print("\nConclusion:")
    print("  Grid approximation is excellent for teaching and 1D/2D problems")
    print("  But it completely fails for high-dimensional inference")
    print("  → This motivates MCMC methods (next module!)")
    print("  → And eventually score-based diffusion (Part 3)")


# =============================================================================
# PART 5: PRACTICAL CONSIDERATIONS
# =============================================================================

def example_5_practical_tips():
    """
    Example 5: Practical Tips for Grid Approximation
    ================================================
    
    Important Practical Considerations:
    ----------------------------------
    1. Numerical Stability
       - Use log probabilities when possible
       - Subtract maximum before exp() to prevent overflow
       
    2. Grid Range
       - Cover the support of the posterior
       - Check that posterior probability near boundaries is ~0
       
    3. Grid Spacing
       - Uniform spacing is simplest
       - Adaptive grids can be more efficient
       
    4. Integration
       - Use proper numerical integration (not just sum!)
       - Trapezoidal rule: ∫ f(x)dx ≈ Δx Σ f(xᵢ)
       - Simpson's rule for higher accuracy
       
    5. When to Use Grid Approximation
       - 1D problems: Almost always fine
       - 2D problems: Usually OK
       - 3D+ problems: Consider MCMC instead
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Practical Tips and Numerical Stability")
    print("=" * 70)
    
    # Example: Numerical instability without log probabilities
    print("\nDemonstration: Why we need log probabilities")
    print("-" * 70)
    
    # Generate some data that leads to very small likelihoods
    np.random.seed(42)
    true_mean = 5.0
    sigma = 0.5
    n_data = 50  # Lots of data → small joint likelihood
    
    data = np.random.normal(true_mean, sigma, n_data)
    
    # Grid
    mu_grid = np.linspace(0, 10, 1000)
    
    # Method 1: Direct computation (will underflow!)
    print("\nMethod 1: Direct probability multiplication")
    likelihood_direct = np.ones(len(mu_grid))
    for x in data:
        likelihood_direct *= norm.pdf(x, mu_grid, sigma)
    
    print(f"  Min likelihood: {likelihood_direct.min()}")
    print(f"  Max likelihood: {likelihood_direct.max()}")
    print(f"  Problem: {np.sum(likelihood_direct == 0)} points became exactly 0!")
    
    # Method 2: Log probabilities (numerically stable)
    print("\nMethod 2: Log probability addition (CORRECT)")
    log_likelihood = np.zeros(len(mu_grid))
    for x in data:
        log_likelihood += norm.logpdf(x, mu_grid, sigma)
    
    # Subtract maximum before exponentiating
    log_likelihood_normalized = log_likelihood - log_likelihood.max()
    likelihood_stable = np.exp(log_likelihood_normalized)
    
    print(f"  Min log-likelihood: {log_likelihood.min():.2f}")
    print(f"  Max log-likelihood: {log_likelihood.max():.2f}")
    print(f"  After exp: min={likelihood_stable.min():.2e}, max={likelihood_stable.max():.2e}")
    print(f"  ✓ No underflow!")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Direct computation (problematic)
    ax = axes[0, 0]
    ax.plot(mu_grid, likelihood_direct, 'r-', linewidth=2)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Likelihood (direct)', fontsize=12)
    ax.set_title('Method 1: Direct Multiplication\n⚠ Numerical Underflow!', 
                fontsize=14, color='red')
    ax.grid(alpha=0.3)
    text = f'Most values = 0\ndue to underflow'
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=12, ha='center', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Log scale
    ax = axes[0, 1]
    ax.plot(mu_grid, log_likelihood, 'g-', linewidth=2)
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title('Method 2: Log Probabilities\n✓ Numerically Stable', 
                fontsize=14, color='green')
    ax.grid(alpha=0.3)
    
    # Plot 3: After exponentiation
    ax = axes[1, 0]
    ax.plot(mu_grid, likelihood_stable, 'b-', linewidth=2)
    ax.axvline(true_mean, color='r', linestyle='--', linewidth=2,
              label=f'True μ={true_mean}')
    ax.set_xlabel('μ', fontsize=12)
    ax.set_ylabel('Likelihood (stable)', fontsize=12)
    ax.set_title('Method 2: After Normalization\n✓ Correct Result', 
                fontsize=14, color='blue')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: Best practices summary
    ax = axes[1, 1]
    ax.axis('off')
    
    best_practices = [
        ("Best Practices for Grid Approximation", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("1. Numerical Stability:", 0.82, 12, 'bold', 'darkblue'),
        ("   ✓ Use log-probabilities", 0.77, 10, 'normal', 'black'),
        ("   ✓ Subtract max before exp()", 0.73, 10, 'normal', 'black'),
        ("   ✓ Check for underflow", 0.69, 10, 'normal', 'black'),
        ("", 0.64, 10, 'normal', 'black'),
        ("2. Grid Design:", 0.58, 12, 'bold', 'darkblue'),
        ("   ✓ Cover posterior support", 0.53, 10, 'normal', 'black'),
        ("   ✓ Check boundary values ≈ 0", 0.49, 10, 'normal', 'black'),
        ("   ✓ Use enough points (1000+)", 0.45, 10, 'normal', 'black'),
        ("", 0.40, 10, 'normal', 'black'),
        ("3. Integration:", 0.34, 12, 'bold', 'darkblue'),
        ("   ✓ Multiply by grid width Δx", 0.29, 10, 'normal', 'black'),
        ("   ✓ Use scipy.integrate if needed", 0.25, 10, 'normal', 'black'),
        ("", 0.20, 10, 'normal', 'black'),
        ("4. When to Use:", 0.14, 12, 'bold', 'darkred'),
        ("   ✓ 1D, 2D: Usually fine", 0.09, 10, 'normal', 'darkgreen'),
        ("   ✗ 3D+: Consider MCMC", 0.05, 10, 'normal', 'darkred'),
    ]
    
    for text, y, size, weight, color in best_practices:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'numerical_stability.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: numerical_stability.png")
    print("\nKey Takeaways:")
    print("  ✓ Always use log probabilities for products of many terms")
    print("  ✓ Subtract maximum log-probability before exponentiating")
    print("  ✓ Check your results make sense (no NaNs, Infs, or all zeros!)")
    print("  ✓ Grid approximation is great for learning, but limited to low dimensions")


# =============================================================================
# PART 6: LOOKING FORWARD
# =============================================================================

def preview_next_steps():
    """
    Preview: From Grid Approximation to MCMC
    ========================================
    
    What We've Learned:
    ------------------
    ✓ Grid approximation is the simplest numerical Bayesian inference method
    ✓ Direct implementation of Bayes' theorem
    ✓ Works great for 1D and 2D problems
    ✗ Fails catastrophically for high dimensions (curse of dimensionality)
    
    What's Next:
    -----------
    Module 03-06: MCMC Methods
    - Metropolis-Hastings algorithm
    - Gibbs sampling
    - Convergence diagnostics
    - Practical MCMC implementation
    
    Module 07-09: Langevin Dynamics
    - Gradient-based sampling
    - Score functions
    - Connections to diffusion
    
    Part 3: Diffusion Models
    - Score matching
    - Denoising diffusion probabilistic models
    - Iterative refinement
    
    The Journey:
    -----------
    Grid → MCMC → Langevin → Score Matching → Diffusion
    
    Each method solves the problems of the previous one!
    """
    print("\n" + "=" * 70)
    print("PREVIEW: From Grid Approximation to Advanced Methods")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    roadmap = [
        ("The Path from Grid Approximation to Diffusion", 0.96, 18, 'bold', 'black'),
        ("", 0.90, 12, 'normal', 'black'),
        
        ("✓ Module 02: Grid Approximation", 0.84, 14, 'bold', 'darkgreen'),
        ("  • Direct implementation of Bayes' theorem", 0.79, 11, 'normal', 'black'),
        ("  • Works perfectly for 1D-2D problems", 0.75, 11, 'normal', 'black'),
        ("  ✗ Exponential scaling kills it for high-D", 0.71, 11, 'normal', 'darkred'),
        ("", 0.66, 10, 'normal', 'black'),
        
        ("→ Module 03-06: MCMC Sampling", 0.60, 14, 'bold', 'darkblue'),
        ("  • Random walk through parameter space", 0.55, 11, 'normal', 'black'),
        ("  • Only visit high-probability regions", 0.51, 11, 'normal', 'black'),
        ("  • Works in any dimension!", 0.47, 11, 'normal', 'darkgreen'),
        ("  ~ But can be slow to converge", 0.43, 11, 'normal', 'orange'),
        ("", 0.38, 10, 'normal', 'black'),
        
        ("→ Module 07-09: Langevin Dynamics", 0.32, 14, 'bold', 'purple'),
        ("  • Use gradients to guide sampling", 0.27, 11, 'normal', 'black'),
        ("  • Follow the score: ∇log p(x)", 0.23, 11, 'normal', 'black'),
        ("  • Much faster than random MCMC", 0.19, 11, 'normal', 'darkgreen'),
        ("  • Bridge to diffusion models!", 0.15, 11, 'normal', 'darkgreen'),
        ("", 0.10, 10, 'normal', 'black'),
        
        ("→ Part 3: Diffusion Models", 0.04, 14, 'bold', 'red'),
        ("  • Iterative denoising via learned scores", 0.00, 11, 'normal', 'black'),
    ]
    
    for text, y, size, weight, color in roadmap:
        if text.startswith("→"):
            # Draw arrow
            ax.annotate('', xy=(0.15, y+0.01), xytext=(0.15, y+0.04),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    # Add summary box
    summary_box = [
        "Key Insight:",
        "",
        "Each method solves the limitations",
        "of the previous one!",
        "",
        "Grid: O(n^d) → MCMC: O(samples)",
        "MCMC slow → Langevin: use gradients",
        "Langevin → Diffusion: iterative refinement",
    ]
    
    box_text = '\n'.join(summary_box)
    ax.text(0.70, 0.30, box_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', 
                    edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'roadmap_to_diffusion.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: roadmap_to_diffusion.png")
    print("\nWhat You've Mastered:")
    print("  ✓ Implementing Bayesian inference numerically")
    print("  ✓ Understanding trade-offs (accuracy vs computation)")
    print("  ✓ Recognizing when grid approximation breaks down")
    print("  ✓ Numerical stability techniques")
    print("\nYou're ready for MCMC methods!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRID APPROXIMATION FOR BAYESIAN INFERENCE: BEGINNER TUTORIAL")
    print("=" * 70)
    print("\nThis tutorial covers:")
    print("  1. Basic grid approximation for coin flip inference")
    print("  2. Effect of grid resolution on accuracy")
    print("  3. Normal distribution mean inference")
    print("  4. Curse of dimensionality")
    print("  5. Practical considerations and numerical stability")
    print("  6. Preview of MCMC methods")
    print("\n" + "=" * 70)
    
    # Run all examples
    example_1_coin_flip_grid()
    example_2_grid_resolution()
    example_3_normal_inference()
    example_4_curse_of_dimensionality_preview()
    example_5_practical_tips()
    preview_next_steps()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • example1_coin_flip_grid.png")
    print("  • grid_resolution_analysis.png")
    print("  • normal_mean_inference.png")
    print("  • curse_of_dimensionality.png")
    print("  • numerical_stability.png")
    print("  • roadmap_to_diffusion.png")
    print("\nNext: 02_grid_approximation_intermediate.py")
    print("      (2D grids, contour plots, and computational optimization)")
