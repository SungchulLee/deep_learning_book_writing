"""
03_grid_approximation_advanced.py

GRID APPROXIMATION FOR BAYESIAN INFERENCE: ADVANCED LEVEL
=========================================================

Learning Objectives:
-------------------
1. Understand adaptive grid methods
2. Implement importance sampling for efficient grid approximation
3. Explore sequential grid refinement
4. Study theoretical convergence properties
5. Connect to modern computational methods

Prerequisites:
-------------
- 01_grid_approximation_basics.py (1D grid methods)
- 02_grid_approximation_intermediate.py (2D grid methods)
- Understanding of numerical integration theory
- Familiarity with importance sampling

Theoretical Framework:
---------------------
Grid approximation is a quadrature method for computing:
    E[f(θ)|D] = ∫ f(θ) p(θ|D) dθ

Error Analysis:
--------------
For uniform grid with n points in d dimensions:
    Error = O(n^(-2/d)) for smooth functions
    
This shows the curse of dimensionality:
    - 1D: Error ~ 1/n²
    - 2D: Error ~ 1/n
    - 10D: Error ~ 1/n^(1/5) [very slow!]

Connection to Modern Methods:
-----------------------------
Grid approximation connects to:
1. Variational inference (grid-based optimization)
2. Particle filters (weighted point masses)
3. Quasi-Monte Carlo (low-discrepancy grids)
4. Tensor approximations (high-D structured grids)

Advanced Topics Covered:
-----------------------
1. Adaptive grid refinement
2. Importance-weighted grids
3. Hierarchical grid structures
4. Convergence analysis
5. Connections to MCMC and variational methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, beta
from scipy.integrate import quad, dblquad
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: ADAPTIVE GRID REFINEMENT
# =============================================================================

def example_1_adaptive_grid():
    """
    Example 1: Adaptive Grid Refinement
    ===================================
    
    Problem: Uniform grids waste points in low-probability regions
    Solution: Adapt grid density to posterior probability
    
    Algorithm:
    ---------
    1. Start with coarse uniform grid
    2. Evaluate posterior on grid
    3. Refine grid in high-probability regions
    4. Iterate until convergence
    
    Benefits:
    --------
    - Fewer total grid points needed
    - Better accuracy in important regions
    - Natural for multimodal posteriors
    
    Theoretical Foundation:
    ----------------------
    Adaptive quadrature minimizes:
        ∫ |error(x)|² p(x)dx
    
    By concentrating points where p(x) is large
    """
    print("=" * 70)
    print("EXAMPLE 1: Adaptive Grid Refinement")
    print("=" * 70)
    
    # Example: Bimodal posterior (mixture of two Gaussians)
    # This is where adaptive grids shine!
    
    def bimodal_posterior(x):
        """Mixture of two Gaussians"""
        return 0.6 * norm.pdf(x, -2, 0.5) + 0.4 * norm.pdf(x, 3, 0.8)
    
    # True integral (for error computation)
    true_integral, _ = quad(bimodal_posterior, -10, 10)
    true_mean = quad(lambda x: x * bimodal_posterior(x), -10, 10)[0] / true_integral
    
    print(f"\nTrue posterior mean: {true_mean:.4f}")
    
    # Method 1: Uniform grid
    n_uniform = 100
    x_uniform = np.linspace(-6, 6, n_uniform)
    p_uniform = bimodal_posterior(x_uniform)
    dx_uniform = x_uniform[1] - x_uniform[0]
    
    # Normalize
    norm_uniform = np.sum(p_uniform) * dx_uniform
    p_uniform_normalized = p_uniform / norm_uniform
    
    # Compute mean
    mean_uniform = np.sum(x_uniform * p_uniform_normalized * dx_uniform)
    error_uniform = abs(mean_uniform - true_mean)
    
    print(f"\nUniform Grid ({n_uniform} points):")
    print(f"  Estimated mean: {mean_uniform:.4f}")
    print(f"  Error: {error_uniform:.6f}")
    
    # Method 2: Adaptive grid
    # Start with coarse grid, refine where p(x) > threshold
    def create_adaptive_grid(func, x_min, x_max, n_initial=20, n_refine=80):
        """Create adaptive grid with more points in high-probability regions"""
        # Initial coarse grid
        x_coarse = np.linspace(x_min, x_max, n_initial)
        p_coarse = func(x_coarse)
        
        # Find high-probability regions (top 50%)
        threshold = np.percentile(p_coarse, 50)
        high_prob_indices = np.where(p_coarse > threshold)[0]
        
        # Create fine grid in high-probability regions
        x_adaptive = [x_coarse]
        for i in high_prob_indices:
            if i < len(x_coarse) - 1:
                # Add extra points between x_coarse[i] and x_coarse[i+1]
                x_fine = np.linspace(x_coarse[i], x_coarse[i+1], 10)[1:-1]
                x_adaptive.append(x_fine)
        
        x_adaptive = np.sort(np.concatenate(x_adaptive))
        return x_adaptive
    
    x_adaptive = create_adaptive_grid(bimodal_posterior, -6, 6, 20, 80)
    p_adaptive = bimodal_posterior(x_adaptive)
    
    # Approximate integral with variable grid spacing
    # Use trapezoidal rule
    integral_adaptive = 0
    for i in range(len(x_adaptive) - 1):
        dx = x_adaptive[i+1] - x_adaptive[i]
        integral_adaptive += 0.5 * (p_adaptive[i] + p_adaptive[i+1]) * dx
    
    p_adaptive_normalized = p_adaptive / integral_adaptive
    
    # Compute mean using trapezoidal rule
    mean_adaptive = 0
    for i in range(len(x_adaptive) - 1):
        dx = x_adaptive[i+1] - x_adaptive[i]
        mean_adaptive += 0.5 * (x_adaptive[i] * p_adaptive_normalized[i] + 
                               x_adaptive[i+1] * p_adaptive_normalized[i+1]) * dx
    
    error_adaptive = abs(mean_adaptive - true_mean)
    
    print(f"\nAdaptive Grid ({len(x_adaptive)} points):")
    print(f"  Estimated mean: {mean_adaptive:.4f}")
    print(f"  Error: {error_adaptive:.6f}")
    print(f"  Improvement: {error_uniform / error_adaptive:.1f}x better accuracy")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Posterior with both grids
    ax = axes[0, 0]
    x_fine = np.linspace(-6, 6, 1000)
    p_fine = bimodal_posterior(x_fine) / true_integral
    
    ax.plot(x_fine, p_fine, 'k-', linewidth=2, label='True posterior')
    ax.plot(x_uniform, p_uniform_normalized, 'bo', markersize=4,
           label=f'Uniform grid (n={n_uniform})', alpha=0.6)
    ax.plot(x_adaptive, p_adaptive_normalized, 'r^', markersize=4,
           label=f'Adaptive grid (n={len(x_adaptive)})', alpha=0.6)
    ax.axvline(true_mean, color='green', linestyle='--', linewidth=2,
              label=f'True mean={true_mean:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bimodal Posterior: Grid Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Grid point distribution
    ax = axes[0, 1]
    ax.hist(x_uniform, bins=20, alpha=0.5, label='Uniform', color='blue',
           edgecolor='black')
    ax.hist(x_adaptive, bins=30, alpha=0.5, label='Adaptive', color='red',
           edgecolor='black')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Number of grid points', fontsize=12)
    ax.set_title('Grid Point Distribution\n(Adaptive concentrates near modes)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Error comparison
    ax = axes[1, 0]
    methods = ['Uniform\n(100 pts)', 'Adaptive\n(100 pts)']
    errors = [error_uniform, error_adaptive]
    colors = ['blue', 'red']
    
    bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Posterior Mean: Error Comparison', fontsize=14)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.6f}',
               ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Key insights
    ax = axes[1, 1]
    ax.axis('off')
    
    insights = [
        ("Adaptive Grid Advantages", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("✓ Better accuracy with same # points", 0.82, 12, 'bold', 'darkgreen'),
        (f"  {error_uniform/error_adaptive:.1f}× improvement in this example", 0.77, 10, 'normal', 'black'),
        ("", 0.72, 10, 'normal', 'black'),
        ("✓ Efficient for multimodal posteriors", 0.66, 12, 'bold', 'darkgreen'),
        ("  Concentrates points near modes", 0.61, 10, 'normal', 'black'),
        ("", 0.56, 10, 'normal', 'black'),
        ("✓ Automatic region detection", 0.50, 12, 'bold', 'darkgreen'),
        ("  No need to guess grid range", 0.45, 10, 'normal', 'black'),
        ("", 0.40, 10, 'normal', 'black'),
        ("When to use:", 0.34, 12, 'bold', 'darkblue'),
        ("  • Complex posterior shapes", 0.29, 10, 'normal', 'black'),
        ("  • Unknown posterior structure", 0.25, 10, 'normal', 'black'),
        ("  • Need high accuracy", 0.21, 10, 'normal', 'black'),
        ("", 0.16, 10, 'normal', 'black'),
        ("Limitation:", 0.10, 12, 'bold', 'darkred'),
        ("  Still exponential in dimensions!", 0.05, 10, 'normal', 'darkred'),
    ]
    
    for text, y, size, weight, color in insights:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'adaptive_grid_refinement.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: adaptive_grid_refinement.png")
    print("\nKey Takeaways:")
    print("  ✓ Adaptive grids can be significantly more efficient")
    print("  ✓ Especially valuable for complex posterior shapes")
    print("  ✓ Still limited to low dimensions (1D-3D)")
    print("  ✓ Foundation for modern adaptive methods (MCMC, SMC)")


# =============================================================================
# PART 2: IMPORTANCE-WEIGHTED GRIDS
# =============================================================================

def example_2_importance_sampling_grid():
    """
    Example 2: Importance-Weighted Grid Approximation
    =================================================
    
    Idea: Sample from proposal q(θ), reweight by p(θ|D)/q(θ)
    
    Mathematical Foundation:
    -----------------------
    E[f(θ)|D] = ∫ f(θ) p(θ|D) dθ
              = ∫ f(θ) [p(θ|D)/q(θ)] q(θ) dθ
              ≈ (1/n) Σᵢ f(θᵢ) w(θᵢ)
              
    where θᵢ ~ q(θ) and w(θᵢ) = p(θᵢ|D)/q(θᵢ)
    
    Why This Matters:
    ----------------
    - Can use coarser grids
    - Focus computational effort on important regions
    - Connection to particle filters and SMC
    - Bridge between grid methods and Monte Carlo
    
    When q is good:
    - Variance reduction compared to uniform grid
    - Fewer points needed for same accuracy
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Importance-Weighted Grid Approximation")
    print("=" * 70)
    
    # Target: Mixture posterior (same as before)
    def target_posterior(x):
        return 0.6 * norm.pdf(x, -2, 0.5) + 0.4 * norm.pdf(x, 3, 0.8)
    
    # Compute true mean
    true_integral, _ = quad(target_posterior, -10, 10)
    true_mean = quad(lambda x: x * target_posterior(x), -10, 10)[0] / true_integral
    
    print(f"\nTrue posterior mean: {true_mean:.4f}")
    
    # Method 1: Uniform grid (standard)
    n_points = 100
    x_uniform = np.linspace(-6, 6, n_points)
    p_uniform = target_posterior(x_uniform)
    dx = x_uniform[1] - x_uniform[0]
    mean_uniform = np.sum(x_uniform * p_uniform) * dx / (np.sum(p_uniform) * dx)
    error_uniform = abs(mean_uniform - true_mean)
    
    print(f"\nUniform Grid ({n_points} points):")
    print(f"  Estimated mean: {mean_uniform:.4f}")
    print(f"  Error: {error_uniform:.6f}")
    
    # Method 2: Importance sampling grid
    # Proposal: Single Gaussian centered at data mean
    proposal_mean = 0.5  # Roughly between the two modes
    proposal_std = 3.0   # Wide enough to cover both modes
    
    np.random.seed(42)
    # Sample from proposal
    x_importance = np.random.normal(proposal_mean, proposal_std, n_points)
    x_importance = np.sort(x_importance)  # Sort for visualization
    
    # Compute importance weights
    p_target = target_posterior(x_importance)
    q_proposal = norm.pdf(x_importance, proposal_mean, proposal_std)
    weights = p_target / (q_proposal + 1e-10)  # Add small constant for stability
    
    # Normalize weights
    weights_normalized = weights / weights.sum()
    
    # Estimate mean
    mean_importance = np.sum(x_importance * weights_normalized)
    error_importance = abs(mean_importance - true_mean)
    
    print(f"\nImportance Sampling ({n_points} samples):")
    print(f"  Proposal: N({proposal_mean}, {proposal_std}²)")
    print(f"  Estimated mean: {mean_importance:.4f}")
    print(f"  Error: {error_importance:.6f}")
    print(f"  Effective sample size: {1/np.sum(weights_normalized**2):.1f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Target, proposal, and samples
    ax = axes[0, 0]
    x_fine = np.linspace(-6, 6, 1000)
    p_target_fine = target_posterior(x_fine) / true_integral
    q_proposal_fine = norm.pdf(x_fine, proposal_mean, proposal_std)
    
    ax.plot(x_fine, p_target_fine, 'b-', linewidth=2, label='Target p(θ|D)')
    ax.plot(x_fine, q_proposal_fine, 'g--', linewidth=2, label='Proposal q(θ)')
    ax.scatter(x_uniform, np.zeros(len(x_uniform)), c='blue', marker='|',
              s=100, alpha=0.5, label='Uniform grid')
    ax.scatter(x_importance, np.zeros(len(x_importance)), c='red', marker='|',
              s=100, alpha=0.5, label='IS samples')
    ax.axvline(true_mean, color='black', linestyle=':', linewidth=2,
              label=f'True mean={true_mean:.2f}')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Target, Proposal, and Sampling Locations', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Importance weights
    ax = axes[0, 1]
    ax.scatter(x_importance, weights_normalized, c=weights_normalized,
              cmap='viridis', s=50, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Normalized Weight', fontsize=12)
    ax.set_title('Importance Weights\n(Color intensity = weight)', fontsize=14)
    ax.grid(alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=weights_normalized.min(),
                                                vmax=weights_normalized.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Weight')
    
    # Plot 3: Error comparison
    ax = axes[1, 0]
    methods = ['Uniform\nGrid', 'Importance\nSampling']
    errors = [error_uniform, error_importance]
    colors = ['blue', 'red']
    
    bars = ax.bar(methods, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'Error Comparison (n={n_points})', fontsize=14)
    ax.grid(alpha=0.3, axis='y')
    
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.6f}',
               ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Theoretical insights
    ax = axes[1, 1]
    ax.axis('off')
    
    insights = [
        ("Importance Sampling Theory", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("Variance Reduction:", 0.82, 13, 'bold', 'darkblue'),
        ("  Var ∝ ∫ [p(θ)/q(θ)]² q(θ) dθ", 0.77, 10, 'normal', 'black'),
        ("", 0.72, 10, 'normal', 'black'),
        ("Optimal proposal:", 0.66, 12, 'bold', 'darkgreen'),
        ("  q*(θ) ∝ |f(θ)| p(θ|D)", 0.61, 10, 'normal', 'black'),
        ("  (not always feasible!)", 0.57, 9, 'italic', 'gray'),
        ("", 0.52, 10, 'normal', 'black'),
        ("Effective Sample Size:", 0.46, 12, 'bold', 'darkblue'),
        ("  ESS = 1 / Σ(wᵢ)²", 0.41, 10, 'normal', 'black'),
        (f"  Here: ESS = {1/np.sum(weights_normalized**2):.1f}/{n_points}", 0.37, 10, 'normal', 'black'),
        ("", 0.32, 10, 'normal', 'black'),
        ("Connection to modern methods:", 0.26, 12, 'bold', 'purple'),
        ("  • Particle filters", 0.21, 10, 'normal', 'black'),
        ("  • Sequential Monte Carlo", 0.17, 10, 'normal', 'black'),
        ("  • Annealed importance sampling", 0.13, 10, 'normal', 'black'),
        ("  • Variational inference", 0.09, 10, 'normal', 'black'),
    ]
    
    for text, y, size, weight, color in insights:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'importance_sampling_grid.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: importance_sampling_grid.png")
    print("\nKey Insights:")
    print("  ✓ Importance sampling bridges grid methods and Monte Carlo")
    print("  ✓ Can reduce variance with good proposal")
    print("  ✓ ESS indicates effective use of samples")
    print("  ✓ Foundation for particle methods and SMC")


# =============================================================================
# PART 3: CONVERGENCE ANALYSIS
# =============================================================================

def example_3_convergence_analysis():
    """
    Example 3: Convergence Rate Analysis
    ====================================
    
    Theoretical Question: How does error decrease with grid resolution?
    
    Theory:
    ------
    For d-dimensional grid with n points per dimension:
        Error ∝ n^(-r/d)
        
    where r depends on smoothness (r=2 for twice-differentiable)
    
    The Curse of Dimensionality:
    ---------------------------
    To achieve error ε:
        n ≥ ε^(-d/r)
        
    Example with ε = 0.01, r = 2:
        1D: n ≥ 100
        2D: n ≥ 10,000
        10D: n ≥ 10^10 (impossible!)
    
    This analysis shows why we MUST move to MCMC for high dimensions
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Convergence Rate Analysis")
    print("=" * 70)
    
    # Test posterior: 1D Gaussian
    true_mean = 2.0
    true_std = 1.0
    
    def posterior(x):
        return norm.pdf(x, true_mean, true_std)
    
    # Test grid sizes
    grid_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    
    print("\nConvergence analysis for 1D Gaussian posterior:")
    print("-" * 70)
    print(f"{'Grid Size':<12} {'Mean Error':<15} {'Std Error':<15}")
    print("-" * 70)
    
    mean_errors = []
    std_errors = []
    
    for n in grid_sizes:
        x_grid = np.linspace(true_mean - 4*true_std, true_mean + 4*true_std, n)
        p_grid = posterior(x_grid)
        dx = x_grid[1] - x_grid[0]
        
        # Normalize
        p_normalized = p_grid / (np.sum(p_grid) * dx)
        
        # Compute moments
        est_mean = np.sum(x_grid * p_normalized * dx)
        est_var = np.sum((x_grid - est_mean)**2 * p_normalized * dx)
        est_std = np.sqrt(est_var)
        
        mean_error = abs(est_mean - true_mean)
        std_error = abs(est_std - true_std)
        
        mean_errors.append(mean_error)
        std_errors.append(std_error)
        
        print(f"{n:<12} {mean_error:<15.2e} {std_error:<15.2e}")
    
    # Fit convergence rate
    log_n = np.log(grid_sizes)
    log_error = np.log(mean_errors)
    
    # Linear fit: log(error) = log(C) - α log(n)
    # So error ∝ n^(-α)
    coeffs = np.polyfit(log_n[3:], log_error[3:], 1)  # Use last points for fit
    alpha = -coeffs[0]
    
    print(f"\nEmpirical convergence rate: n^(-{alpha:.2f})")
    print(f"Theoretical rate for 1D: n^(-2) (quadratic)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error vs Grid Size (log-log)
    ax = axes[0, 0]
    ax.loglog(grid_sizes, mean_errors, 'bo-', linewidth=2, markersize=8,
             label='Mean error')
    ax.loglog(grid_sizes, std_errors, 'ro-', linewidth=2, markersize=8,
             label='Std error')
    
    # Add theoretical lines
    theoretical_mean = mean_errors[3] * (grid_sizes[3] / np.array(grid_sizes))**2
    ax.loglog(grid_sizes, theoretical_mean, 'b--', linewidth=2,
             label='n^(-2) reference', alpha=0.5)
    
    ax.set_xlabel('Grid Size (n)', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Convergence Rate: Error vs Grid Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 2: Convergence in different dimensions (theoretical)
    ax = axes[0, 1]
    n_range = np.linspace(10, 500, 100)
    
    # Error ∝ n^(-2/d) for different dimensions
    for d in [1, 2, 3, 4]:
        error_d = n_range**(-2/d)
        # Normalize to 1 at n=10
        error_d = error_d / error_d[0]
        ax.loglog(n_range, error_d, linewidth=2, label=f'{d}D: n^(-{2/d:.2f})')
    
    ax.set_xlabel('Grid Size per Dimension (n)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Curse of Dimensionality:\nError Scaling in Different Dimensions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 3: Points needed for fixed error
    ax = axes[1, 0]
    target_errors = np.logspace(-4, -1, 50)  # Errors from 0.0001 to 0.1
    
    for d in [1, 2, 3, 4, 5]:
        # n = error^(-d/2)
        points_per_dim = target_errors**(-d/2)
        total_points = points_per_dim ** d
        
        # Only plot feasible region (< 10^9 points)
        feasible = total_points < 1e9
        ax.loglog(target_errors[feasible], total_points[feasible],
                 linewidth=2, label=f'{d}D')
    
    ax.axhline(1e9, color='red', linestyle='--', linewidth=2,
              label='Practical limit (10^9)')
    ax.set_xlabel('Target Error', fontsize=12)
    ax.set_ylabel('Total Grid Points Needed', fontsize=12)
    ax.set_title('Computational Cost vs Accuracy\n(Exponential explosion!)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.invert_xaxis()  # Smaller error on right
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = [
        ("Convergence Theory Summary", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        ("Error Scaling:", 0.82, 13, 'bold', 'darkblue'),
        ("  Error ∝ n^(-2/d)", 0.77, 11, 'normal', 'black'),
        ("  where d = dimension", 0.73, 10, 'normal', 'gray'),
        ("", 0.68, 10, 'normal', 'black'),
        ("For error ε = 0.01:", 0.62, 12, 'bold', 'darkgreen'),
        ("  1D: need ~100 points ✓", 0.57, 10, 'normal', 'darkgreen'),
        ("  2D: need ~10,000 points ✓", 0.53, 10, 'normal', 'orange'),
        ("  3D: need ~10^6 points ~", 0.49, 10, 'normal', 'orange'),
        ("  10D: need ~10^10 points ✗", 0.45, 10, 'normal', 'darkred'),
        ("", 0.40, 10, 'normal', 'black'),
        ("The Fundamental Problem:", 0.34, 13, 'bold', 'darkred'),
        ("  Grid methods scale exponentially", 0.29, 11, 'normal', 'black'),
        ("  with dimension!", 0.25, 11, 'normal', 'black'),
        ("", 0.20, 10, 'normal', 'black'),
        ("Solution:", 0.14, 13, 'bold', 'purple'),
        ("  → MCMC: Polynomial scaling", 0.09, 11, 'normal', 'purple'),
        ("  → Works in 100+ dimensions", 0.05, 11, 'normal', 'purple'),
    ]
    
    for text, y, size, weight, color in summary:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'convergence_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: convergence_analysis.png")
    print("\nFundamental Conclusion:")
    print("  Grid approximation has EXPONENTIAL cost in dimension")
    print("  This is a fundamental mathematical limitation")
    print("  → MCMC is not just convenient, it's NECESSARY for high-D problems")


# =============================================================================
# PART 4: CONNECTION TO MODERN METHODS
# =============================================================================

def preview_modern_connections():
    """
    Preview: How Grid Approximation Connects to Modern Methods
    ==========================================================
    
    Grid approximation is the foundation for understanding:
    
    1. Markov Chain Monte Carlo (MCMC)
       - Replaces exhaustive grid with strategic sampling
       - Explores high-probability regions efficiently
       - Converges to correct distribution asymptotically
    
    2. Variational Inference
       - Approximates posterior with parametric family
       - Optimization over parameters (similar to grid search)
       - Trades accuracy for speed
    
    3. Particle Filters / Sequential Monte Carlo
       - Importance-weighted particles (like adaptive grid)
       - Evolves distribution through resampling
       - Handles sequential/dynamic problems
    
    4. Normalizing Flows
       - Transforms simple distribution to complex posterior
       - Learned via neural networks
       - Exact likelihood computation
    
    5. Diffusion Models
       - Iterative refinement (multi-step grid approximation)
       - Score-based methods (gradient of log-density)
       - Connects to Langevin dynamics
    
    The Journey:
    -----------
    Grid → MCMC → Langevin → Score Matching → Diffusion
    
    Each method addresses limitations of the previous!
    """
    print("\n" + "=" * 70)
    print("PREVIEW: Connections to Modern Methods")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Method comparison
    ax = axes[0, 0]
    methods = ['Grid\n(1D-2D)', 'Grid\n(3D)', 'MCMC', 'Variational\nInf.', 'Diffusion']
    dimensions = [2, 3, 100, 100, 100000]
    colors = ['green', 'orange', 'blue', 'purple', 'red']
    
    bars = ax.barh(methods, dimensions, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Max Practical Dimension', fontsize=12)
    ax.set_title('Method Scalability Comparison', fontsize=14)
    ax.set_xscale('log')
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 2: Accuracy vs Speed trade-off
    ax = axes[0, 1]
    
    # Conceptual positioning (not rigorous!)
    method_props = {
        'Grid (1D-2D)': (9, 5, 'green'),  # (accuracy, speed)
        'Grid (3D+)': (8, 1, 'orange'),
        'MCMC': (7, 6, 'blue'),
        'Variational': (5, 9, 'purple'),
        'Diffusion': (8, 7, 'red'),
    }
    
    for method, (acc, speed, color) in method_props.items():
        ax.scatter(speed, acc, s=200, c=color, alpha=0.7,
                  edgecolor='black', linewidth=2)
        ax.annotate(method, (speed, acc), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Speed →', fontsize=12)
    ax.set_ylabel('Accuracy →', fontsize=12)
    ax.set_title('Accuracy vs Speed Trade-off\n(Conceptual)', fontsize=14)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.3)
    
    # Add ideal region
    from matplotlib.patches import Rectangle
    ideal = Rectangle((7, 7), 3, 3, linewidth=2, edgecolor='gold',
                     facecolor='yellow', alpha=0.2)
    ax.add_patch(ideal)
    ax.text(8.5, 8.5, 'Ideal', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot 3: Evolution timeline
    ax = axes[1, 0]
    ax.axis('off')
    
    timeline = [
        ("Evolution of Bayesian Computation", 0.95, 15, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        
        ("1. Grid Approximation (1960s)", 0.82, 12, 'bold', 'green'),
        ("   • Direct computation", 0.77, 10, 'normal', 'black'),
        ("   • Limited to 1D-2D", 0.73, 10, 'normal', 'black'),
        ("   ↓", 0.69, 12, 'normal', 'gray'),
        
        ("2. MCMC (1980s-1990s)", 0.64, 12, 'bold', 'blue'),
        ("   • Metropolis-Hastings, Gibbs", 0.59, 10, 'normal', 'black'),
        ("   • Works in high dimensions", 0.55, 10, 'normal', 'black'),
        ("   ↓", 0.51, 12, 'normal', 'gray'),
        
        ("3. Hamiltonian MC (2000s)", 0.46, 12, 'bold', 'blue'),
        ("   • Uses gradients", 0.41, 10, 'normal', 'black'),
        ("   • Faster mixing", 0.37, 10, 'normal', 'black'),
        ("   ↓", 0.33, 12, 'normal', 'gray'),
        
        ("4. Variational Inference (2010s)", 0.28, 12, 'bold', 'purple'),
        ("   • Optimization-based", 0.23, 10, 'normal', 'black'),
        ("   • Fast but approximate", 0.19, 10, 'normal', 'black'),
        ("   ↓", 0.15, 12, 'normal', 'gray'),
        
        ("5. Score-Based/Diffusion (2020s)", 0.10, 12, 'bold', 'red'),
        ("   • Iterative denoising", 0.05, 10, 'normal', 'black'),
        ("   • State-of-the-art generation", 0.01, 10, 'normal', 'black'),
    ]
    
    for text, y, size, weight, color in timeline:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    # Plot 4: Key lessons
    ax = axes[1, 1]
    ax.axis('off')
    
    lessons = [
        ("What We Learned from Grid Methods", 0.95, 14, 'bold', 'black'),
        ("", 0.88, 12, 'normal', 'black'),
        
        ("✓ Understanding posteriors", 0.82, 12, 'bold', 'darkgreen'),
        ("  Direct visualization possible", 0.77, 10, 'normal', 'black'),
        ("  Intuition for Bayesian inference", 0.73, 10, 'normal', 'black'),
        ("", 0.68, 10, 'normal', 'black'),
        
        ("✓ Computational principles", 0.62, 12, 'bold', 'darkgreen'),
        ("  Numerical integration techniques", 0.57, 10, 'normal', 'black'),
        ("  Importance of normalization", 0.53, 10, 'normal', 'black'),
        ("  Curse of dimensionality", 0.49, 10, 'normal', 'black'),
        ("", 0.44, 10, 'normal', 'black'),
        
        ("✓ Foundation for advanced methods", 0.38, 12, 'bold', 'darkgreen'),
        ("  MCMC: Strategic sampling", 0.33, 10, 'normal', 'black'),
        ("  Variational: Optimization view", 0.29, 10, 'normal', 'black'),
        ("  Diffusion: Iterative refinement", 0.25, 10, 'normal', 'black'),
        ("", 0.20, 10, 'normal', 'black'),
        
        ("Next: MCMC Methods!", 0.12, 14, 'bold', 'purple'),
        ("The key to high-D inference", 0.05, 11, 'normal', 'purple'),
    ]
    
    for text, y, size, weight, color in lessons:
        ax.text(0.05, y, text, transform=ax.transAxes, fontsize=size,
               fontweight=weight, color=color)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'modern_connections.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: modern_connections.png")
    print("\nThe Big Picture:")
    print("  Grid approximation taught us:")
    print("    • How Bayesian inference works computationally")
    print("    • Why high dimensions are fundamentally hard")
    print("    • The need for smarter sampling strategies")
    print("  ")
    print("  This knowledge prepares us for:")
    print("    • MCMC methods (next module)")
    print("    • Variational inference")
    print("    • Modern diffusion models")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GRID APPROXIMATION: ADVANCED TUTORIAL")
    print("=" * 70)
    print("\nThis tutorial covers:")
    print("  1. Adaptive grid refinement techniques")
    print("  2. Importance-weighted grid approximation")
    print("  3. Convergence rate analysis and theory")
    print("  4. Connections to modern computational methods")
    print("\n" + "=" * 70)
    
    # Run all examples
    example_1_adaptive_grid()
    example_2_importance_sampling_grid()
    example_3_convergence_analysis()
    preview_modern_connections()
    
    print("\n" + "=" * 70)
    print("MODULE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • adaptive_grid_refinement.png")
    print("  • importance_sampling_grid.png")
    print("  • convergence_analysis.png")
    print("  • modern_connections.png")
    print("\nKey Takeaways:")
    print("  ✓ Grid approximation: Simple but limited to low dimensions")
    print("  ✓ Adaptive methods improve efficiency")
    print("  ✓ Importance sampling bridges grid and Monte Carlo")
    print("  ✓ Exponential scaling necessitates MCMC")
    print("\nYou're now ready for Module 03: Metropolis-Hastings MCMC!")
    print("\nFor further study:")
    print("  - Quasi-Monte Carlo methods")
    print("  - Sparse grid techniques")
    print("  - Tensor decomposition methods")
    print("  - Sequential Monte Carlo (particle filters)")
