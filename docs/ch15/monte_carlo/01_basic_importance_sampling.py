"""
01_basic_importance_sampling.py

BEGINNER LEVEL: Introduction to Importance Sampling

This module introduces the fundamental concept of importance sampling:
computing expectations under one distribution by sampling from another.

Mathematical Foundation:
--------------------
Goal: Compute E_π[h(θ)] = ∫ h(θ)π(θ)dθ

Importance Sampling Identity:
    E_π[h(θ)] = ∫ h(θ) π(θ) dθ
               = ∫ h(θ) [ π(θ) / q(θ) ] q(θ)dθ
               = E_q[ h(θ) · w(θ) ]

where w(θ) = π(θ)/q(θ) is the importance weight.

Monte Carlo Estimator:
    Sample θ₁, ..., θₙ ~ q(θ)
    Ê[h(θ)] = (1/n) Σᵢ h(θᵢ) w(θᵢ)

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

# Define target distribution: π(θ) = N(3, 1)
# This is the distribution we want to compute expectations under
target_mean, target_std = 3.0, 1.0
target_dist = stats.norm(loc=target_mean, scale=target_std)

# Define proposal distribution: q(θ) = N(0, 2)
# This is the distribution we will sample from
proposal_mean, proposal_std = 0.0, 2.0
proposal_dist = stats.norm(loc=proposal_mean, scale=proposal_std)

# Define function h(θ) = θ²
# We want to compute E_π[θ²]
h_function = lambda theta: theta**2

# Compute true expectation analytically
# For θ ~ N(μ, σ²): E[θ²] = μ² + σ²
true_expectation = target_mean**2 + target_std**2
print(f"\nTrue E_π[θ²] (analytical): {true_expectation:.6f}")


def basic_importance_sampling(target_dist, proposal_dist, h_function, n_samples):
    """
    Basic importance sampling algorithm.
    
    Parameters:
    -----------
    target_dist : scipy.stats distribution
        The target distribution π(θ) we want to compute expectations under
    proposal_dist : scipy.stats distribution
        The proposal distribution q(θ) we will sample from
    h_function : callable
        The function h(θ) whose expectation we want to compute
    n_samples : int
        Number of samples to draw from proposal
        
    Returns:
    --------
    estimate : float
        The importance sampling estimate of E_π[h(θ)]
    samples : array
        The samples drawn from the proposal distribution
    weights : array
        The importance weights for each sample
    
    Mathematical Steps:
    ------------------
    1. Draw samples: θᵢ ~ q(θ)
    2. Compute importance weights: wᵢ = π(θᵢ)/q(θᵢ)
    3. Compute weighted average: Ê[h(θ)] = (1/n) Σᵢ h(θᵢ)wᵢ
    """
    # Step 1: Draw samples from proposal distribution q(θ)
    samples = proposal_dist.rvs(size=n_samples)
    
    # Step 2: Evaluate target density π(θ) at each sample point
    # These are π(θᵢ) values
    target_density = target_dist.pdf(samples)
    
    # Step 3: Evaluate proposal density q(θ) at each sample point
    # These are q(θᵢ) values
    proposal_density = proposal_dist.pdf(samples)
    
    # Step 4: Compute importance weights w(θᵢ) = π(θᵢ)/q(θᵢ)
    # Add small epsilon to avoid division by zero
    weights = target_density / (proposal_density + 1e-300)
    
    # Step 5: Evaluate function h at each sample point
    h_values = h_function(samples)
    
    # Step 6: Compute importance sampling estimate
    # Ê[h(θ)] = (1/n) Σᵢ h(θᵢ)w(θᵢ)
    estimate = np.mean(h_values * weights)
    
    return estimate, samples, weights


def compute_true_expectation(target_dist, h_function, n_integration_points=100000):
    """
    Compute the true expectation by numerical integration (for validation).
    
    This is our ground truth for comparison.
    E_π[h(θ)] = ∫ h(θ)π(θ)dθ
    """
    # Generate a fine grid over the support of the target distribution
    x = np.linspace(target_dist.ppf(0.001), target_dist.ppf(0.999), 
                    n_integration_points)
    
    # Compute the integrand: h(θ)π(θ)
    integrand = h_function(x) * target_dist.pdf(x)
    
    # Numerical integration using trapezoidal rule
    true_value = np.trapz(integrand, x)
    
    return true_value


# Example 1: Simple Expectation with Normal Distributions
# ======================================================
def example_1():
    print("=" * 70)
    print("EXAMPLE 1: Computing E_π[θ²] where π = N(3, 1)")
    print("=" * 70)

    # Run importance sampling with different sample sizes
    sample_sizes = [10, 100, 1000, 10000]
    print("\nImportance Sampling Estimates:")
    print("-" * 50)

    for n in sample_sizes:
        estimate, samples, weights = basic_importance_sampling(
            target_dist, proposal_dist, h_function, n
        )
        error = abs(estimate - true_expectation)
        rel_error = error / true_expectation * 100
        
        print(f"n = {n:5d}: Estimate = {estimate:.6f}, "
            f"Error = {error:.6f} ({rel_error:.2f}%)")

    # Visualize the distributions and samples
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Target and Proposal Distributions
    x = np.linspace(-5, 8, 1000)
    ax = axes[0, 0]
    ax.plot(x, target_dist.pdf(x), 'b-', linewidth=2, label='Target π(θ) = N(3,1)')
    ax.plot(x, proposal_dist.pdf(x), 'r--', linewidth=2, label='Proposal q(θ) = N(0,2)')
    ax.set_xlabel('θ', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Target vs Proposal Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Importance Weights
    n_samples = 1000
    estimate, samples, weights = basic_importance_sampling(
        target_dist, proposal_dist, h_function, n_samples
    )
    ax = axes[0, 1]
    ax.hist(weights, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Importance Weight w(θ)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of Importance Weights (n={n_samples})', 
                fontsize=13, fontweight='bold')
    ax.axvline(np.mean(weights), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {np.mean(weights):.3f}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence of Estimate
    sample_sizes_conv = np.arange(10, 5001, 50)
    estimates = []
    for n in sample_sizes_conv:
        estimate, _, _ = basic_importance_sampling(
            target_dist, proposal_dist, h_function, n
        )
        estimates.append(estimate)

    ax = axes[1, 0]
    ax.plot(sample_sizes_conv, estimates, 'b-', linewidth=2, alpha=0.7, 
            label='IS Estimate')
    ax.axhline(true_expectation, color='red', linestyle='--', linewidth=2, 
            label='True Value')
    ax.fill_between(sample_sizes_conv, 
                    true_expectation - 0.1, 
                    true_expectation + 0.1,
                    alpha=0.2, color='red', label='±0.1 band')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Estimate of E[θ²]', fontsize=12)
    ax.set_title('Convergence of Importance Sampling Estimate', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: Samples vs Weights
    n_samples = 500
    estimate, samples, weights = basic_importance_sampling(
        target_dist, proposal_dist, h_function, n_samples
    )
    ax = axes[1, 1]
    scatter = ax.scatter(samples, weights, c=h_function(samples), 
                        cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Sample θ', fontsize=12)
    ax.set_ylabel('Weight w(θ)', fontsize=12)
    ax.set_title('Samples Colored by h(θ) = θ²', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='h(θ)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__),'example1_basic_IS.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: example1_basic_IS.png")


# Example 2: Tail Probability Estimation
# ======================================
def example_2():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Estimating Tail Probability P(θ > 5) for π = N(3, 1)")
    print("=" * 70)

    # Define indicator function h(θ) = I(θ > 5)
    # This computes P_π(θ > 5) = E_π[I(θ > 5)]
    threshold = 5.0
    h_indicator = lambda theta: (theta > threshold).astype(float)

    # True probability (analytical)
    true_prob = 1 - target_dist.cdf(threshold)
    print(f"\nTrue P(θ > {threshold}): {true_prob:.6f}")

    # Naive Monte Carlo from target (for comparison)
    n_samples = 10000
    samples_target = target_dist.rvs(size=n_samples)
    naive_estimate = np.mean(h_indicator(samples_target))
    print(f"Naive MC estimate: {naive_estimate:.6f}")

    # Importance Sampling with proposal shifted towards tail: q = N(5, 1.5)
    # This proposal puts more weight in the tail region
    proposal_tail = stats.norm(loc=5.0, scale=1.5)
    is_estimate, samples_is, weights = basic_importance_sampling(
        target_dist, proposal_tail, h_indicator, n_samples
    )
    print(f"IS estimate: {is_estimate:.6f}")

    # Compare standard errors (run multiple replications)
    n_replications = 100
    naive_estimates = []
    is_estimates = []

    print("\nRunning 100 replications to compare variance...")
    for _ in range(n_replications):
        # Naive MC
        samples_target = target_dist.rvs(size=n_samples)
        naive_estimates.append(np.mean(h_indicator(samples_target)))
        
        # Importance Sampling
        is_est, _, _ = basic_importance_sampling(
            target_dist, proposal_tail, h_indicator, n_samples
        )
        is_estimates.append(is_est)

    naive_std = np.std(naive_estimates)
    is_std = np.std(is_estimates)
    variance_reduction = naive_std / is_std

    print(f"\nStandard deviation of estimates:")
    print(f"  Naive MC: {naive_std:.6f}")
    print(f"  IS: {is_std:.6f}")
    print(f"  Variance reduction factor: {variance_reduction:.2f}x")


# Example 3: Multiple Functions Simultaneously
# ===========================================
def example_3():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Computing Multiple Expectations Simultaneously")
    print("=" * 70)

    n_samples = 5000
    estimate, samples, weights = basic_importance_sampling(
        target_dist, proposal_dist, lambda x: x, n_samples
    )

    # We can compute expectations of multiple functions using the same samples
    functions = {
        'E[θ]': lambda x: x,
        'E[θ²]': lambda x: x**2,
        'E[θ³]': lambda x: x**3,
        'E[exp(θ)]': lambda x: np.exp(x),
        'E[sin(θ)]': lambda x: np.sin(x)
    }

    print(f"\nUsing {n_samples} samples:")
    print("-" * 50)
    for name, func in functions.items():
        # Use same samples and weights for all functions
        estimate = np.mean(func(samples) * weights)
        print(f"{name:12s}: {estimate:.6f}")


# Example 4: Effect of Proposal Choice
# ====================================
def example_4():
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Impact of Proposal Distribution Choice")
    print("=" * 70)

    # Target: π = N(3, 1)
    # Try different proposals
    proposals = {
        'Good (N(3,1.2))': stats.norm(3, 1.2),      # Close to target
        'Okay (N(2,1.5))': stats.norm(2, 1.5),      # Reasonably close
        'Poor (N(0,2))': stats.norm(0, 2),          # Mismatched mean
        'Bad (N(3,0.5))': stats.norm(3, 0.5),       # Too narrow
    }

    h_function = lambda x: x**2
    n_samples = 1000
    n_replications = 50

    print(f"\nComparing proposals ({n_replications} replications, {n_samples} samples each):")
    print("-" * 70)

    for name, proposal in proposals.items():
        estimates = []
        for _ in range(n_replications):
            est, _, _ = basic_importance_sampling(
                target_dist, proposal, h_function, n_samples
            )
            estimates.append(est)
        
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        rmse = np.sqrt(np.mean((np.array(estimates) - true_expectation)**2))
        
        print(f"{name:20s}: Mean={mean_est:.4f}, Std={std_est:.4f}, RMSE={rmse:.4f}")

    plt.show()

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Importance sampling allows us to compute expectations under π(θ)
    by sampling from a different distribution q(θ).

    2. The importance weight w(θ) = π(θ)/q(θ) corrects for the mismatch
    between target and proposal distributions.

    3. For rare event estimation, IS can provide significant variance
    reduction compared to naive Monte Carlo.

    4. The choice of proposal distribution critically affects:
    - Accuracy (bias is zero, but finite-sample error varies)
    - Variance (poor proposals lead to high variance)
    - Numerical stability (very small/large weights are problematic)

    5. Good proposal distributions:
    - Cover the support of the target
    - Have similar shape to the target
    - Have slightly heavier tails than the target
    - Are easy to sample from

    6. We can compute expectations of multiple functions simultaneously
    using the same set of weighted samples (reusability).
    """)
