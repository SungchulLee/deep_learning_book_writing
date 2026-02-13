"""
02_grid_approximation_intermediate.py

GRID APPROXIMATION FOR BAYESIAN INFERENCE: INTERMEDIATE LEVEL
=============================================================

Learning Objectives:
-------------------
1. Implement 2D grid approximation for multivariate posteriors
2. Visualize posterior distributions with contour plots and 3D surfaces
3. Compute marginal and conditional distributions from joint posteriors
4. Understand computational optimization techniques
5. Compare grid approximation with analytical solutions in 2D

Prerequisites:
-------------
- 01_grid_approximation_basics.py (1D grid approximation)
- Basic understanding of multivariate distributions
- Familiarity with contour plots

Mathematical Foundation:
-----------------------
For parameters θ = (θ₁, θ₂):

Joint Posterior:
    p(θ₁, θ₂|D) ∝ p(D|θ₁, θ₂) p(θ₁, θ₂)

Marginal Posteriors:
    p(θ₁|D) = ∫ p(θ₁, θ₂|D) dθ₂
    p(θ₂|D) = ∫ p(θ₁, θ₂|D) dθ₁

Conditional Posteriors:
    p(θ₁|θ₂, D) = p(θ₁, θ₂|D) / p(θ₂|D)
    p(θ₂|θ₁, D) = p(θ₁, θ₂|D) / p(θ₁|D)

Connection to Diffusion:
-----------------------
- Multi-dimensional posteriors appear in diffusion denoising
- Understanding marginals helps interpret learned distributions
- Computational techniques transfer to high-D score estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, beta
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: 2D GRID APPROXIMATION - BIVARIATE NORMAL
# =============================================================================

def example_1_bivariate_normal_inference():
    """
    Example 1: Inference for 2D Normal Distribution
    ===============================================
    
    Problem: Observe data from N(μ, Σ) where μ = (μ₁, μ₂) is unknown
            and Σ is known. Infer μ using grid approximation.
    
    Mathematical Setup:
    ------------------
    - Parameters: μ = (μ₁, μ₂) ∈ ℝ²
    - Data: x₁, ..., xₙ ~ N(μ, Σ) where Σ is known
    - Prior: μ ~ N(μ₀, Σ₀) (conjugate)
    - Posterior: μ|D ~ N(μₙ, Σₙ) (analytical)
    
    Conjugate Update (Known):
    -------------------------
    Σₙ⁻¹ = Σ₀⁻¹ + n Σ⁻¹
    μₙ = Σₙ (Σ₀⁻¹ μ₀ + n Σ⁻¹ x̄)
    
    Grid Approximation:
    ------------------
    1. Create 2D grid: (μ₁,ᵢ, μ₂,ⱼ) for i=1..n₁, j=1..n₂
    2. Evaluate: p(μ₁,ᵢ, μ₂,ⱼ|D) ∝ p(D|μ₁,ᵢ, μ₂,ⱼ) p(μ₁,ᵢ, μ₂,ⱼ)
    3. Normalize: Sum over all grid points
    4. Compute marginals by integrating out one dimension
    
    Why This Example?
    ----------------
    - First taste of 2D posterior inference
    - Can verify against analytical solution
    - Demonstrates computational scaling: O(n₁ × n₂)
    - Shows importance of visualization
    """
    print("=" * 70)
    print("EXAMPLE 1: 2D Grid Approximation - Bivariate Normal")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True parameters
    true_mu = np.array([2.0, -1.0])
    true_Sigma = np.array([[1.0, 0.3],
                           [0.3, 0.5]])
    
    # Generate data
    n_data = 30
    data = np.random.multivariate_normal(true_mu, true_Sigma, n_data)
    data_mean = data.mean(axis=0)
    
    print(f"\nTrue mean: μ = {true_mu}")
    print(f"Data: n = {n_data}")
    print(f"Sample mean: {data_mean}")
    
    # Prior: Diffuse N(0, 5I)
    prior_mu = np.array([0.0, 0.0])
    prior_Sigma = np.eye(2) * 25  # Large variance = diffuse
    
    print(f"\nPrior: N({prior_mu}, diag([25, 25]))")
    
    # Analytical posterior (conjugate)
    prior_precision = np.linalg.inv(prior_Sigma)
    data_precision = n_data * np.linalg.inv(true_Sigma)
    post_precision = prior_precision + data_precision
    post_Sigma = np.linalg.inv(post_precision)
    post_mu = post_Sigma @ (prior_precision @ prior_mu + data_precision @ data_mean)
    
    print(f"\nAnalytical Posterior:")
    print(f"  Mean: {post_mu}")
    print(f"  Covariance:\n{post_Sigma}")
    
    # GRID APPROXIMATION IN 2D
    # ========================
    
    # Create 2D grid
    n_grid = 100  # 100 x 100 = 10,000 points
    
    # Define grid ranges (cover ±3σ of posterior)
    std1 = np.sqrt(post_Sigma[0, 0])
    std2 = np.sqrt(post_Sigma[1, 1])
    
    mu1_grid = np.linspace(post_mu[0] - 3*std1, post_mu[0] + 3*std1, n_grid)
    mu2_grid = np.linspace(post_mu[1] - 3*std2, post_mu[1] + 3*std2, n_grid)
    
    # Create meshgrid
    Mu1, Mu2 = np.meshgrid(mu1_grid, mu2_grid)
    grid_points = np.stack([Mu1.ravel(), Mu2.ravel()], axis=1)
    
    print(f"\nGrid Setup:")
    print(f"  Grid size: {n_grid} × {n_grid} = {n_grid**2:,} points")
    print(f"  μ₁ range: [{mu1_grid[0]:.2f}, {mu1_grid[-1]:.2f}]")
    print(f"  μ₂ range: [{mu2_grid[0]:.2f}, {mu2_grid[-1]:.2f}]")
    
    # Evaluate prior at each grid point
    prior_vals = multivariate_normal.pdf(grid_points, prior_mu, prior_Sigma)
    
    # Evaluate likelihood at each grid point
    # p(D|μ) = ∏ᵢ N(xᵢ|μ, Σ)
    log_likelihood = np.zeros(len(grid_points))
    for x in data:
        log_likelihood += multivariate_normal.logpdf(x, grid_points, true_Sigma)
    
    # Numerical stability: subtract max
    log_likelihood -= log_likelihood.max()
    likelihood_vals = np.exp(log_likelihood)
    
    # Compute unnormalized posterior
    unnormalized_posterior = prior_vals * likelihood_vals
    
    # Normalize
    grid_area = (mu1_grid[1] - mu1_grid[0]) * (mu2_grid[1] - mu2_grid[0])
    normalization = np.sum(unnormalized_posterior) * grid_area
    posterior_vals = unnormalized_posterior / normalization
    
    # Reshape to 2D for visualization
    posterior_grid = posterior_vals.reshape(n_grid, n_grid)
    
    # Analytical posterior for comparison
    analytical_vals = multivariate_normal.pdf(grid_points, post_mu, post_Sigma)
    analytical_grid = analytical_vals.reshape(n_grid, n_grid)
    
    # Compute posterior mean from grid
    grid_post_mu1 = np.sum(Mu1 * posterior_grid * grid_area)
    grid_post_mu2 = np.sum(Mu2 * posterior_grid * grid_area)
    grid_post_mu = np.array([grid_post_mu1, grid_post_mu2])
    
    print(f"\nPosterior Mean:")
    print(f"  Grid approximation: {grid_post_mu}")
    print(f"  Analytical:         {post_mu}")
    print(f"  Error: {np.linalg.norm(grid_post_mu - post_mu):.6e}")
    
    # VISUALIZATION
    # =============
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Contour plot of posterior
    ax1 = plt.subplot(2, 3, 1)
    levels = np.linspace(0, posterior_grid.max(), 20)
    contour = ax1.contourf(Mu1, Mu2, posterior_grid, levels=levels, cmap='viridis')
    ax1.contour(Mu1, Mu2, posterior_grid, levels=10, colors='white', 
               alpha=0.3, linewidths=0.5)
    ax1.plot(true_mu[0], true_mu[1], 'r*', markersize=20, 
            label='True μ', markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(grid_post_mu[0], grid_post_mu[1], 'wo', markersize=12,
            label='Posterior mean', markeredgecolor='black', markeredgewidth=1.5)
    ax1.scatter(data[:, 0], data[:, 1], c='cyan', s=30, alpha=0.6,
               edgecolor='black', linewidth=0.5, label='Data')
    ax1.set_xlabel('μ₁', fontsize=12)
    ax1.set_ylabel('μ₂', fontsize=12)
    ax1.set_title('Grid Approximation:\nPosterior p(μ₁,μ₂|D)', fontsize=13)
    ax1.legend(fontsize=9)
    plt.colorbar(contour, ax=ax1)
    
    # Plot 2: Comparison with analytical
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(Mu1, Mu2, analytical_grid, levels=levels, cmap='viridis')
    ax2.contour(Mu1, Mu2, analytical_grid, levels=10, colors='white',
               alpha=0.3, linewidths=0.5)
    ax2.plot(true_mu[0], true_mu[1], 'r*', markersize=20,
            label='True μ', markeredgecolor='white', markeredgewidth=1.5)
    ax2.plot(post_mu[0], post_mu[1], 'wo', markersize=12,
            label='Posterior mean', markeredgecolor='black', markeredgewidth=1.5)
    ax2.set_xlabel('μ₁', fontsize=12)
    ax2.set_ylabel('μ₂', fontsize=12)
    ax2.set_title('Analytical Solution:\nPosterior p(μ₁,μ₂|D)', fontsize=13)
    ax2.legend(fontsize=9)
    plt.colorbar(contour2, ax=ax2)
    
    # Plot 3: Difference (error)
    ax3 = plt.subplot(2, 3, 3)
    difference = np.abs(posterior_grid - analytical_grid)
    contour3 = ax3.contourf(Mu1, Mu2, difference, levels=20, cmap='Reds')
    ax3.set_xlabel('μ₁', fontsize=12)
    ax3.set_ylabel('μ₂', fontsize=12)
    ax3.set_title('Absolute Error:\n|Grid - Analytical|', fontsize=13)
    plt.colorbar(contour3, ax=ax3, label='Error')
    textstr = f'Max error: {difference.max():.2e}\nMean error: {difference.mean():.2e}'
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.8))
    
    # Plot 4: 3D surface of posterior
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    surf = ax4.plot_surface(Mu1, Mu2, posterior_grid, cmap='viridis',
                           alpha=0.8, edgecolor='none')
    ax4.set_xlabel('μ₁', fontsize=10)
    ax4.set_ylabel('μ₂', fontsize=10)
    ax4.set_zlabel('Density', fontsize=10)
    ax4.set_title('3D View: Posterior Surface', fontsize=13)
    ax4.view_init(elev=30, azim=45)
    
    # Plot 5: Marginal p(μ₁|D)
    ax5 = plt.subplot(2, 3, 5)
    # Compute marginal by integrating out μ₂
    marginal_mu1_grid = np.sum(posterior_grid, axis=0) * (mu2_grid[1] - mu2_grid[0])
    marginal_mu1_analytical = norm.pdf(mu1_grid, post_mu[0], np.sqrt(post_Sigma[0, 0]))
    
    ax5.plot(mu1_grid, marginal_mu1_grid, 'r-', linewidth=2,
            label='Grid (integrate out μ₂)')
    ax5.plot(mu1_grid, marginal_mu1_analytical, 'b--', linewidth=2,
            label='Analytical')
    ax5.axvline(true_mu[0], color='green', linestyle=':', linewidth=2,
               label='True μ₁')
    ax5.set_xlabel('μ₁', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Marginal Posterior: p(μ₁|D)', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Plot 6: Marginal p(μ₂|D)
    ax6 = plt.subplot(2, 3, 6)
    # Compute marginal by integrating out μ₁
    marginal_mu2_grid = np.sum(posterior_grid, axis=1) * (mu1_grid[1] - mu1_grid[0])
    marginal_mu2_analytical = norm.pdf(mu2_grid, post_mu[1], np.sqrt(post_Sigma[1, 1]))
    
    ax6.plot(mu2_grid, marginal_mu2_grid, 'r-', linewidth=2,
            label='Grid (integrate out μ₁)')
    ax6.plot(mu2_grid, marginal_mu2_analytical, 'b--', linewidth=2,
            label='Analytical')
    ax6.axvline(true_mu[1], color='green', linestyle=':', linewidth=2,
               label='True μ₂')
    ax6.set_xlabel('μ₂', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12)
    ax6.set_title('Marginal Posterior: p(μ₂|D)', fontsize=13)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '2d_bivariate_normal_grid.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: 2d_bivariate_normal_grid.png")
    print("\nKey Insights:")
    print("  ✓ 2D grid approximation works well with 100×100 = 10,000 points")
    print("  ✓ Can compute marginals by integrating out dimensions")
    print("  ✓ Contour plots and 3D surfaces reveal posterior structure")
    print("  ✓ Computational cost: O(n²) for n × n grid")


# =============================================================================
# PART 2: CORRELATED PARAMETERS - VISUALIZATION
# =============================================================================

def example_2_correlated_parameters():
    """
    Example 2: Understanding Parameter Correlation
    ==============================================
    
    Demonstrates how grid approximation reveals correlation structure
    in the posterior distribution.
    
    Key Concepts:
    ------------
    1. Positive correlation: θ₁ ↑ → θ₂ ↑
    2. Negative correlation: θ₁ ↑ → θ₂ ↓
    3. Independence: p(θ₁,θ₂) = p(θ₁)p(θ₂)
    
    Why Correlation Matters:
    -----------------------
    - Affects sampling efficiency (MCMC gets "stuck" in ellipses)
    - Important for uncertainty quantification
    - Common in regression problems
    
    Visualization Techniques:
    ------------------------
    - Contour plots show correlation ellipses
    - Scatter plots of samples
    - Marginal distributions
    - Conditional slices
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Visualizing Parameter Correlation")
    print("=" * 70)
    
    # Create posteriors with different correlations
    mean = np.array([0.0, 0.0])
    
    # Case 1: No correlation
    cov_independent = np.array([[1.0, 0.0],
                                [0.0, 1.0]])
    
    # Case 2: Positive correlation
    cov_positive = np.array([[1.0, 0.8],
                            [0.8, 1.0]])
    
    # Case 3: Negative correlation
    cov_negative = np.array([[1.0, -0.8],
                            [-0.8, 1.0]])
    
    # Create grid
    n_grid = 150
    x = np.linspace(-3, 3, n_grid)
    y = np.linspace(-3, 3, n_grid)
    X, Y = np.meshgrid(x, y)
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Evaluate densities
    Z_indep = multivariate_normal.pdf(pos, mean, cov_independent).reshape(n_grid, n_grid)
    Z_pos = multivariate_normal.pdf(pos, mean, cov_positive).reshape(n_grid, n_grid)
    Z_neg = multivariate_normal.pdf(pos, mean, cov_negative).reshape(n_grid, n_grid)
    
    # Generate samples
    np.random.seed(42)
    n_samples = 500
    samples_indep = np.random.multivariate_normal(mean, cov_independent, n_samples)
    samples_pos = np.random.multivariate_normal(mean, cov_positive, n_samples)
    samples_neg = np.random.multivariate_normal(mean, cov_negative, n_samples)
    
    # Visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    correlations = ['Independent (ρ=0)', 'Positive (ρ=0.8)', 'Negative (ρ=-0.8)']
    Z_list = [Z_indep, Z_pos, Z_neg]
    samples_list = [samples_indep, samples_pos, samples_neg]
    
    for i, (corr_name, Z, samples) in enumerate(zip(correlations, Z_list, samples_list)):
        # Contour plot
        ax = axes[i, 0]
        levels = np.linspace(0, Z.max(), 15)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('θ₁', fontsize=11)
        ax.set_ylabel('θ₂', fontsize=11)
        ax.set_title(f'{corr_name}\nContour Plot', fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax)
        
        # Scatter plot of samples
        ax = axes[i, 1]
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=20, c='blue',
                  edgecolor='none')
        ax.contour(X, Y, Z, levels=5, colors='red', alpha=0.5, linewidths=1)
        ax.set_xlabel('θ₁', fontsize=11)
        ax.set_ylabel('θ₂', fontsize=11)
        ax.set_title(f'{corr_name}\nSample Scatter', fontsize=12)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Marginal distributions
        ax = axes[i, 2]
        ax.hist(samples[:, 0], bins=30, density=True, alpha=0.7,
               color='blue', label='Marginal θ₁')
        ax.hist(samples[:, 1], bins=30, density=True, alpha=0.7,
               color='red', label='Marginal θ₂')
        x_range = np.linspace(-3, 3, 200)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'b-', linewidth=2, alpha=0.7)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Parameter value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{corr_name}\nMarginal Distributions', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'correlation_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: correlation_visualization.png")
    print("\nKey Observations:")
    print("  • Independent: Circular contours, θ₁ and θ₂ vary independently")
    print("  • Positive correlation: Diagonal ellipses (upper-left to lower-right)")
    print("  • Negative correlation: Diagonal ellipses (upper-right to lower-left)")
    print("  • Marginals are the same (N(0,1)) but joint structure differs!")
    print("  • This affects MCMC sampling efficiency (next modules)")


# =============================================================================
# PART 3: LINEAR REGRESSION WITH 2D GRID
# =============================================================================

def example_3_linear_regression_2d():
    """
    Example 3: Simple Linear Regression y = α + βx + ε
    ==================================================
    
    Problem: Infer slope β and intercept α from data
    
    Mathematical Setup:
    ------------------
    Model: yᵢ = α + β xᵢ + εᵢ,  εᵢ ~ N(0, σ²)
    Parameters: θ = (α, β)
    
    Likelihood:
        p(y|X, α, β, σ²) = ∏ᵢ N(yᵢ | α + βxᵢ, σ²)
    
    Prior:
        p(α, β) = N(α|0, σ_α²) N(β|0, σ_β²)  [Independent priors]
    
    Why This Example:
    ----------------
    - Classic regression problem
    - Natural 2D parameter space
    - Often shows correlation between α and β
    - Connects to Bayesian neural networks
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Linear Regression with 2D Grid")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True parameters
    true_alpha = 2.0  # intercept
    true_beta = 1.5   # slope
    sigma = 0.5       # noise std (known)
    
    # Generate data
    n_data = 25
    x = np.linspace(0, 5, n_data)
    y = true_alpha + true_beta * x + sigma * np.random.randn(n_data)
    
    print(f"\nTrue parameters:")
    print(f"  Intercept α = {true_alpha}")
    print(f"  Slope β = {true_beta}")
    print(f"  Noise σ = {sigma}")
    print(f"  Data points: n = {n_data}")
    
    # Priors (independent)
    prior_alpha_mean = 0.0
    prior_alpha_std = 5.0
    prior_beta_mean = 0.0
    prior_beta_std = 5.0
    
    print(f"\nPriors:")
    print(f"  α ~ N({prior_alpha_mean}, {prior_alpha_std}²)")
    print(f"  β ~ N({prior_beta_mean}, {prior_beta_std}²)")
    
    # Create 2D grid
    n_grid = 120
    alpha_range = np.linspace(-1, 5, n_grid)
    beta_range = np.linspace(-1, 4, n_grid)
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    
    # Vectorized computation for efficiency
    # Reshape for broadcasting
    alpha_vec = Alpha.ravel()
    beta_vec = Beta.ravel()
    
    # Evaluate prior
    prior_alpha_vals = norm.pdf(alpha_vec, prior_alpha_mean, prior_alpha_std)
    prior_beta_vals = norm.pdf(beta_vec, prior_beta_mean, prior_beta_std)
    prior_vals = prior_alpha_vals * prior_beta_vals  # Independent
    
    # Evaluate likelihood
    # For each grid point, compute p(y|x, α, β)
    log_likelihood = np.zeros(len(alpha_vec))
    for i, (xi, yi) in enumerate(zip(x, y)):
        y_pred = alpha_vec + beta_vec * xi
        log_likelihood += norm.logpdf(yi, y_pred, sigma)
    
    # Numerical stability
    log_likelihood -= log_likelihood.max()
    likelihood_vals = np.exp(log_likelihood)
    
    # Posterior
    unnormalized = prior_vals * likelihood_vals
    grid_area = (alpha_range[1] - alpha_range[0]) * (beta_range[1] - beta_range[0])
    posterior_vals = unnormalized / (np.sum(unnormalized) * grid_area)
    
    # Reshape for visualization
    posterior_grid = posterior_vals.reshape(n_grid, n_grid)
    
    # Compute posterior mean
    post_alpha_mean = np.sum(Alpha * posterior_grid * grid_area)
    post_beta_mean = np.sum(Beta * posterior_grid * grid_area)
    
    # Compute posterior std
    post_alpha_var = np.sum((Alpha - post_alpha_mean)**2 * posterior_grid * grid_area)
    post_beta_var = np.sum((Beta - post_beta_mean)**2 * posterior_grid * grid_area)
    post_alpha_std = np.sqrt(post_alpha_var)
    post_beta_std = np.sqrt(post_beta_var)
    
    # Compute correlation
    post_cov = np.sum((Alpha - post_alpha_mean) * (Beta - post_beta_mean) * 
                     posterior_grid * grid_area)
    post_corr = post_cov / (post_alpha_std * post_beta_std)
    
    print(f"\nPosterior Mean:")
    print(f"  α = {post_alpha_mean:.3f} ± {post_alpha_std:.3f}")
    print(f"  β = {post_beta_mean:.3f} ± {post_beta_std:.3f}")
    print(f"\nPosterior Correlation:")
    print(f"  Corr(α, β) = {post_corr:.3f}")
    
    # Compute marginals
    marginal_alpha = np.sum(posterior_grid, axis=0) * (beta_range[1] - beta_range[0])
    marginal_beta = np.sum(posterior_grid, axis=1) * (alpha_range[1] - alpha_range[0])
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Data and fits
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(x, y, c='blue', s=50, alpha=0.7, edgecolor='black',
               linewidth=1, label='Data', zorder=3)
    
    # True line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_true = true_alpha + true_beta * x_line
    ax1.plot(x_line, y_true, 'r-', linewidth=3, label='True line', zorder=2)
    
    # Posterior mean line
    y_post = post_alpha_mean + post_beta_mean * x_line
    ax1.plot(x_line, y_post, 'g--', linewidth=3, label='Posterior mean', zorder=2)
    
    # Uncertainty: sample 50 lines from posterior
    np.random.seed(42)
    n_samples = 50
    # Sample from grid (importance sampling)
    flat_posterior = posterior_vals / posterior_vals.sum()
    sample_indices = np.random.choice(len(alpha_vec), size=n_samples, p=flat_posterior)
    alpha_samples = alpha_vec[sample_indices]
    beta_samples = beta_vec[sample_indices]
    
    for a, b in zip(alpha_samples, beta_samples):
        y_sample = a + b * x_line
        ax1.plot(x_line, y_sample, 'gray', linewidth=0.5, alpha=0.3, zorder=1)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Data and Fitted Lines\n(Gray = posterior samples)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Joint posterior contours
    ax2 = plt.subplot(2, 3, 2)
    levels = np.linspace(0, posterior_grid.max(), 15)
    contour = ax2.contourf(Alpha, Beta, posterior_grid, levels=levels, cmap='viridis')
    ax2.contour(Alpha, Beta, posterior_grid, levels=8, colors='white',
               alpha=0.4, linewidths=1)
    ax2.plot(true_alpha, true_beta, 'r*', markersize=20,
            label='True (α,β)', markeredgecolor='white', markeredgewidth=2)
    ax2.plot(post_alpha_mean, post_beta_mean, 'wo', markersize=12,
            label='Posterior mean', markeredgecolor='black', markeredgewidth=2)
    ax2.set_xlabel('α (Intercept)', fontsize=12)
    ax2.set_ylabel('β (Slope)', fontsize=12)
    ax2.set_title('Joint Posterior p(α,β|D)', fontsize=13)
    ax2.legend(fontsize=10)
    plt.colorbar(contour, ax=ax2)
    
    # Plot 3: 3D surface
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    surf = ax3.plot_surface(Alpha, Beta, posterior_grid, cmap='viridis',
                           alpha=0.8, edgecolor='none')
    ax3.plot([true_alpha], [true_beta], [0], 'r*', markersize=15, zorder=10)
    ax3.set_xlabel('α', fontsize=10)
    ax3.set_ylabel('β', fontsize=10)
    ax3.set_zlabel('Density', fontsize=10)
    ax3.set_title('3D Posterior Surface', fontsize=13)
    ax3.view_init(elev=25, azim=45)
    
    # Plot 4: Marginal p(α|D)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(alpha_range, marginal_alpha, 'b-', linewidth=2,
            label='p(α|D)')
    ax4.fill_between(alpha_range, marginal_alpha, alpha=0.3)
    ax4.axvline(true_alpha, color='r', linestyle='--', linewidth=2,
               label=f'True α={true_alpha}')
    ax4.axvline(post_alpha_mean, color='g', linestyle=':', linewidth=2,
               label=f'Mean={post_alpha_mean:.2f}')
    ax4.set_xlabel('α (Intercept)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Marginal Posterior: α', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # Plot 5: Marginal p(β|D)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(beta_range, marginal_beta, 'b-', linewidth=2,
            label='p(β|D)')
    ax5.fill_between(beta_range, marginal_beta, alpha=0.3)
    ax5.axvline(true_beta, color='r', linestyle='--', linewidth=2,
               label=f'True β={true_beta}')
    ax5.axvline(post_beta_mean, color='g', linestyle=':', linewidth=2,
               label=f'Mean={post_beta_mean:.2f}')
    ax5.set_xlabel('β (Slope)', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Marginal Posterior: β', fontsize=13)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Plot 6: Correlation structure
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(alpha_samples, beta_samples, c='blue', alpha=0.5, s=30,
               edgecolor='none')
    ax6.plot(true_alpha, true_beta, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2, label='True')
    ax6.set_xlabel('α (Intercept)', fontsize=12)
    ax6.set_ylabel('β (Slope)', fontsize=12)
    ax6.set_title(f'Posterior Samples\n(Correlation = {post_corr:.3f})', fontsize=13)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    
    textstr = f'Note: Correlation arises from\nthe data structure and geometry'
    ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'linear_regression_2d_grid.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: linear_regression_2d_grid.png")
    print("\nKey Insights:")
    print("  ✓ Grid captures correlation between α and β")
    print("  ✓ Gray lines show posterior uncertainty")
    print("  ✓ Marginals give inference for individual parameters")
    print("  ✓ Correlation structure visible in contours and samples")


# =============================================================================
# PART 4: COMPUTATIONAL EFFICIENCY
# =============================================================================

def example_4_computational_efficiency():
    """
    Example 4: Optimizing 2D Grid Approximation
    ===========================================
    
    Techniques for Efficient Computation:
    ------------------------------------
    1. Vectorization (avoid loops!)
    2. Adaptive grid spacing
    3. Sparse grids for low posterior probability
    4. Parallel computation
    
    Performance Comparison:
    ----------------------
    - Naive loops: Slow
    - Vectorized operations: 10-100x faster
    - Adaptive grids: Can reduce points by 50-90%
    
    When to Optimize:
    ----------------
    - Large grids (> 100x100)
    - Complex likelihood functions
    - Multiple evaluations needed
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Computational Efficiency in 2D Grid")
    print("=" * 70)
    
    import time
    
    # Setup: Simple 2D Gaussian posterior
    true_mu = np.array([1.0, -0.5])
    true_Sigma = np.array([[1.0, 0.5],
                          [0.5, 1.0]])
    
    # Test different grid sizes
    grid_sizes = [50, 75, 100, 150, 200]
    
    print("\nComparing computation times for different grid sizes:")
    print("-" * 70)
    print(f"{'Grid Size':<12} {'Total Points':<15} {'Time (ms)':<12} {'Points/ms':<12}")
    print("-" * 70)
    
    times = []
    points_list = []
    
    for n_grid in grid_sizes:
        # Create grid
        x = np.linspace(-2, 4, n_grid)
        y = np.linspace(-3, 2, n_grid)
        X, Y = np.meshgrid(x, y)
        pos = np.stack([X.ravel(), Y.ravel()], axis=1)
        
        total_points = n_grid ** 2
        
        # Time the evaluation
        start = time.time()
        
        # Vectorized computation
        Z = multivariate_normal.pdf(pos, true_mu, true_Sigma)
        Z = Z.reshape(n_grid, n_grid)
        
        # Normalize
        grid_area = (x[1] - x[0]) * (y[1] - y[0])
        Z_normalized = Z / (Z.sum() * grid_area)
        
        elapsed = (time.time() - start) * 1000  # Convert to milliseconds
        
        times.append(elapsed)
        points_list.append(total_points)
        
        throughput = total_points / elapsed if elapsed > 0 else float('inf')
        
        print(f"{n_grid}x{n_grid:<5} {total_points:<15,} {elapsed:<12.2f} {throughput:<12,.0f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time vs Grid Size
    ax1 = axes[0]
    ax1.plot(grid_sizes, times, 'bo-', linewidth=2, markersize=8)
    # Fit quadratic (O(n²))
    coeffs = np.polyfit(grid_sizes, times, 2)
    fit_curve = np.polyval(coeffs, grid_sizes)
    ax1.plot(grid_sizes, fit_curve, 'r--', linewidth=2, label='O(n²) fit')
    ax1.set_xlabel('Grid Size (n×n)', fontsize=12)
    ax1.set_ylabel('Computation Time (ms)', fontsize=12)
    ax1.set_title('Scaling: Time vs Grid Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Throughput
    ax2 = axes[1]
    throughput = np.array(points_list) / np.array(times)
    ax2.bar(range(len(grid_sizes)), throughput, color='green', alpha=0.7,
           edgecolor='black')
    ax2.set_xticks(range(len(grid_sizes)))
    ax2.set_xticklabels([f'{n}×{n}' for n in grid_sizes])
    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_ylabel('Throughput (points/ms)', fontsize=12)
    ax2.set_title('Computational Throughput\n(Higher = Better)', fontsize=14)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'computational_efficiency.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: computational_efficiency.png")
    print("\nKey Insights:")
    print("  • Computation time scales as O(n²) for n×n grid")
    print("  • Vectorization is essential for good performance")
    print("  • NumPy operations are highly optimized")
    print("  • For n=200: 40,000 points evaluated in < 100ms")
    print("\nOptimization Tips:")
    print("  1. Always vectorize (no loops over grid points)")
    print("  2. Use log-probabilities for numerical stability")
    print("  3. Consider adaptive grids for complex posteriors")
    print("  4. Remember: 3D grids are 10x more expensive!")


# =============================================================================
# PART 5: LIMITATIONS AND NEXT STEPS
# =============================================================================

def preview_3d_and_beyond():
    """
    Preview: 3D Grid and the Path to MCMC
    =====================================
    
    What We've Learned:
    ------------------
    ✓ 2D grid approximation works well (up to ~200×200 points)
    ✓ Can visualize joint and marginal posteriors
    ✓ Understand correlation structure
    ✓ Computational cost: O(n²)
    
    The 3D Challenge:
    ----------------
    - 100×100×100 = 1,000,000 points
    - Visualization becomes difficult
    - Memory requirements grow quickly
    - Computation time becomes prohibitive
    
    Why We Need MCMC:
    ----------------
    1. Doesn't require full grid
    2. Focuses on high-probability regions
    3. Scales to arbitrary dimensions
    4. Provides samples for inference
    
    Next Steps:
    ----------
    Module 03: Metropolis-Hastings MCMC
    Module 04: Gibbs Sampling
    Module 05-06: Advanced MCMC techniques
    """
    print("\n" + "=" * 70)
    print("PREVIEW: 3D Grid and Beyond")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scaling comparison
    ax = axes[0, 0]
    dimensions = np.arange(1, 6)
    n_per_dim = 100
    grid_points = n_per_dim ** dimensions
    mcmc_samples = np.full(len(dimensions), 10000)
    
    ax.semilogy(dimensions, grid_points, 'ro-', linewidth=2,
               markersize=10, label='Grid points needed')
    ax.semilogy(dimensions, mcmc_samples, 'g--', linewidth=3,
               label='MCMC samples (typical)')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Number of Points/Samples', fontsize=12)
    ax.set_title('Grid vs MCMC Scaling\n(n=100 per dimension)', fontsize=14)
    ax.set_xticks(dimensions)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 2: Memory requirements
    ax = axes[0, 1]
    memory_mb = grid_points * 8 / 1e6  # 8 bytes per float64
    memory_mcmc = mcmc_samples * 8 / 1e6
    
    width = 0.35
    x_pos = np.arange(len(dimensions))
    ax.bar(x_pos - width/2, memory_mb, width, label='Grid', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, memory_mcmc, width, label='MCMC', color='green', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Number of Dimensions', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory Requirements', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dimensions)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both', axis='y')
    
    # Plot 3: Feasibility regions
    ax = axes[1, 0]
    ax.axis('off')
    
    feasibility = [
        ("Grid Approximation Feasibility", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("1D: ✓ Excellent", 0.82, 12, 'bold', 'darkgreen'),
        ("  • 1,000 points typical", 0.77, 10, 'normal', 'black'),
        ("  • Fast, accurate", 0.73, 10, 'normal', 'black'),
        ("", 0.68, 10, 'normal', 'black'),
        ("2D: ✓ Good", 0.62, 12, 'bold', 'green'),
        ("  • 100×100 = 10,000 points", 0.57, 10, 'normal', 'black'),
        ("  • Still manageable", 0.53, 10, 'normal', 'black'),
        ("", 0.48, 10, 'normal', 'black'),
        ("3D: ~ Challenging", 0.42, 12, 'bold', 'orange'),
        ("  • 100³ = 1,000,000 points", 0.37, 10, 'normal', 'black'),
        ("  • Memory/time intensive", 0.33, 10, 'normal', 'black'),
        ("", 0.28, 10, 'normal', 'black'),
        ("4D+: ✗ Impractical", 0.22, 12, 'bold', 'darkred'),
        ("  • Exponential explosion", 0.17, 10, 'normal', 'black'),
        ("  • Need MCMC!", 0.13, 10, 'normal', 'darkred'),
    ]
    
    for text, y, size, weight, color in feasibility:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    # Plot 4: Next steps
    ax = axes[1, 1]
    ax.axis('off')
    
    next_steps = [
        ("What's Next: MCMC Methods", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("Why MCMC?", 0.82, 13, 'bold', 'darkblue'),
        ("  • Works in any dimension", 0.76, 11, 'normal', 'black'),
        ("  • Explores high-probability regions", 0.71, 11, 'normal', 'black'),
        ("  • No exponential scaling", 0.66, 11, 'normal', 'black'),
        ("  • Provides samples for inference", 0.61, 11, 'normal', 'black'),
        ("", 0.55, 10, 'normal', 'black'),
        ("Coming Up:", 0.48, 13, 'bold', 'purple'),
        ("  → Module 03: Metropolis-Hastings", 0.42, 11, 'normal', 'black'),
        ("  → Module 04: Gibbs Sampling", 0.37, 11, 'normal', 'black'),
        ("  → Module 07-09: Langevin Dynamics", 0.32, 11, 'normal', 'black'),
        ("  → Part 3: Diffusion Models", 0.27, 11, 'normal', 'black'),
        ("", 0.20, 10, 'normal', 'black'),
        ("Grid → MCMC → Langevin → Diffusion", 0.12, 12, 'bold', 'red'),
        ("Each step solves previous limitations!", 0.05, 11, 'normal', 'red'),
    ]
    
    for text, y, size, weight, color in next_steps:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '3d_and_beyond.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: 3d_and_beyond.png")
    print("\nConclusion:")
    print("  Grid approximation is excellent for 1D and 2D")
    print("  But it's fundamentally limited by dimensionality")
    print("  → Time to learn MCMC methods!")
    print("  → And eventually gradient-based sampling (Langevin)")
    print("  → Leading to diffusion models!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRID APPROXIMATION: INTERMEDIATE TUTORIAL (2D)")
    print("=" * 70)
    print("\nThis tutorial covers:")
    print("  1. 2D grid approximation for bivariate normal")
    print("  2. Understanding parameter correlation")
    print("  3. Linear regression with 2D posteriors")
    print("  4. Computational efficiency and optimization")
    print("  5. Preview of 3D and path to MCMC")
    print("\n" + "=" * 70)
    
    # Run all examples
    example_1_bivariate_normal_inference()
    example_2_correlated_parameters()
    example_3_linear_regression_2d()
    example_4_computational_efficiency()
    preview_3d_and_beyond()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • 2d_bivariate_normal_grid.png")
    print("  • correlation_visualization.png")
    print("  • linear_regression_2d_grid.png")
    print("  • computational_efficiency.png")
    print("  • 3d_and_beyond.png")
    print("\nNext: 03_grid_approximation_advanced.py")
    print("      (Adaptive grids, importance sampling, and advanced topics)")
