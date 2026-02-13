"""
02_self_normalized_IS.py

BEGINNER LEVEL: Self-Normalized Importance Sampling

This module introduces self-normalized importance sampling, which is essential
for Bayesian inference where the posterior normalizing constant is unknown.

Mathematical Foundation:
--------------------
Problem: We know π(θ) only up to a constant:
    π(θ) = γ(θ)/Z, where Z = ∫γ(θ)dθ is unknown

In Bayesian inference:
    γ(θ) = p(y|θ)p(θ)  (likelihood × prior)
    Z = p(y) = ∫p(y|θ)p(θ)dθ  (marginal likelihood, often intractable)

Solution: Self-Normalized Importance Sampling (SNIS)

Unnormalized weights: w̃ᵢ = γ(θᵢ)/q(θᵢ)

Self-normalized estimator:
    Ê[h(θ)] = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]
             = [Σᵢ h(θᵢ)w̃ᵢ] / [Σᵢ w̃ᵢ]

Properties:
- Biased but consistent
- Bias = O(1/n)
- Often lower variance than normalized IS
- Does not require knowing Z

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

np.random.seed(42)
sns.set_style("whitegrid")


def self_normalized_importance_sampling(unnormalized_target, proposal_dist, 
                                        h_function, n_samples):
    """
    Self-normalized importance sampling for unnormalized target distributions.
    
    Parameters:
    -----------
    unnormalized_target : callable
        Function γ(θ) where π(θ) = γ(θ)/Z
        In Bayesian: γ(θ) = p(y|θ)p(θ)
    proposal_dist : scipy.stats distribution
        Proposal distribution q(θ)
    h_function : callable
        Function whose expectation we want
    n_samples : int
        Number of samples
        
    Returns:
    --------
    estimate : float
        Self-normalized IS estimate
    samples : array
        Samples from proposal
    normalized_weights : array
        Normalized importance weights
    unnormalized_weights : array
        Unnormalized importance weights
        
    Algorithm:
    ---------
    1. Sample θᵢ ~ q(θ) for i=1,...,n
    2. Compute unnormalized weights: w̃ᵢ = γ(θᵢ)/q(θᵢ)
    3. Normalize weights: wᵢ = w̃ᵢ / Σⱼw̃ⱼ
    4. Estimate: Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
    """
    # Step 1: Draw samples from proposal
    samples = proposal_dist.rvs(size=n_samples)
    
    # Step 2: Evaluate unnormalized target γ(θ)
    gamma_values = unnormalized_target(samples)
    
    # Step 3: Evaluate proposal density q(θ)
    q_values = proposal_dist.pdf(samples)
    
    # Step 4: Compute unnormalized weights w̃ᵢ = γ(θᵢ)/q(θᵢ)
    unnormalized_weights = gamma_values / (q_values + 1e-300)
    
    # Step 5: Normalize weights
    # wᵢ = w̃ᵢ / Σⱼw̃ⱼ
    weight_sum = np.sum(unnormalized_weights)
    normalized_weights = unnormalized_weights / weight_sum
    
    # Step 6: Evaluate function h at sample points
    h_values = h_function(samples)
    
    # Step 7: Compute self-normalized estimate
    # Ê[h(θ)] = Σᵢ wᵢh(θᵢ)
    estimate = np.sum(normalized_weights * h_values)
    
    return estimate, samples, normalized_weights, unnormalized_weights


def compute_ess(normalized_weights):
    """
    Compute Effective Sample Size (ESS).
    
    ESS = 1 / Σᵢwᵢ²
    
    Alternative formula using unnormalized weights:
    ESS = (Σᵢw̃ᵢ)² / Σᵢw̃ᵢ²
    
    Interpretation:
    - ESS ≈ n: all samples have similar weights (good)
    - ESS << n: few samples dominate (poor)
    - ESS / n is the "efficiency" of importance sampling
    """
    ess = 1.0 / np.sum(normalized_weights**2)
    return ess


# Example 1: Simple Gaussian with Unknown Normalizing Constant
# ==========================================================
print("=" * 70)
print("EXAMPLE 1: Self-Normalized IS for π(θ) = γ(θ)/Z")
print("=" * 70)

# Define unnormalized target: γ(θ) = exp(-0.5(θ-3)²)
# This is proportional to N(3, 1) but without the normalizing constant
def gamma_function(theta):
    """
    Unnormalized Gaussian: γ(θ) = exp(-0.5(θ-μ)²/σ²)
    Missing the 1/√(2πσ²) factor
    """
    mu, sigma = 3.0, 1.0
    return np.exp(-0.5 * ((theta - mu) / sigma)**2)

# The true normalized distribution (for validation)
target_dist = stats.norm(3, 1)

# Proposal distribution
proposal_dist = stats.norm(0, 2)

# Function to estimate: h(θ) = θ²
h_function = lambda theta: theta**2

# True expectation
true_expectation = 3**2 + 1**2  # E[θ²] for N(3,1)

print(f"\nTrue E[θ²]: {true_expectation:.6f}")

# Run self-normalized IS
n_samples = 1000
estimate, samples, norm_weights, unnorm_weights = self_normalized_importance_sampling(
    gamma_function, proposal_dist, h_function, n_samples
)

ess = compute_ess(norm_weights)
efficiency = ess / n_samples * 100

print(f"\nSelf-Normalized IS Results (n={n_samples}):")
print(f"  Estimate: {estimate:.6f}")
print(f"  Error: {abs(estimate - true_expectation):.6f}")
print(f"  ESS: {ess:.1f}")
print(f"  Efficiency: {efficiency:.1f}%")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Unnormalized vs Normalized Target
x = np.linspace(-5, 8, 1000)
ax = axes[0, 0]
ax.plot(x, gamma_function(x), 'b-', linewidth=2, 
        label='Unnormalized γ(θ)')
ax.plot(x, target_dist.pdf(x), 'r--', linewidth=2, 
        label='Normalized π(θ)')
ax.plot(x, proposal_dist.pdf(x), 'g:', linewidth=2, 
        label='Proposal q(θ)')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density (arbitrary scale)', fontsize=12)
ax.set_title('Unnormalized Target vs Normalized', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Weight Distribution
ax = axes[0, 1]
ax.hist(norm_weights, bins=50, density=True, alpha=0.7, 
        color='purple', edgecolor='black')
ax.set_xlabel('Normalized Weight wᵢ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Distribution of Normalized Weights\nESS = {ess:.1f} ({efficiency:.1f}%)', 
             fontsize=13, fontweight='bold')
uniform_weight = 1.0 / n_samples
ax.axvline(uniform_weight, color='red', linestyle='--', linewidth=2,
           label=f'Uniform = {uniform_weight:.4f}')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: Cumulative Weight Distribution
sorted_weights = np.sort(norm_weights)[::-1]  # Sort descending
cumulative_weights = np.cumsum(sorted_weights)
ax = axes[1, 0]
ax.plot(np.arange(1, len(sorted_weights)+1), cumulative_weights, 
        'b-', linewidth=2)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, 
           label='50% of total weight')
ax.axhline(0.9, color='orange', linestyle='--', linewidth=2,
           label='90% of total weight')
ax.set_xlabel('Number of Samples (sorted by weight)', fontsize=12)
ax.set_ylabel('Cumulative Weight', fontsize=12)
ax.set_title('Cumulative Weight Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Find how many samples account for 50% and 90% of weight
n_50 = np.searchsorted(cumulative_weights, 0.5) + 1
n_90 = np.searchsorted(cumulative_weights, 0.9) + 1
ax.text(0.05, 0.95, f'{n_50} samples = 50% weight\n{n_90} samples = 90% weight',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 4: Samples colored by weight
ax = axes[1, 1]
scatter = ax.scatter(samples, h_function(samples), c=norm_weights, 
                     cmap='hot', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Sample θ', fontsize=12)
ax.set_ylabel('h(θ) = θ²', fontsize=12)
ax.set_title('Samples Colored by Weight', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Normalized Weight')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example1_self_normalized.png")
plt.savefig(fig_path, 
            dpi=300, bbox_inches='tight')
print("\nVisualization saved to: example1_self_normalized.png")


# Example 2: Bayesian Inference - Normal Mean with Unknown Variance
# ================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Bayesian Inference for Normal Mean")
print("=" * 70)

# Data: y ~ N(θ, σ²) with known σ = 1
# Prior: θ ~ N(μ₀, τ²)
# Posterior: θ|y ~ N(μₙ, τₙ²) where
#   τₙ² = 1/(1/τ² + n/σ²)
#   μₙ = τₙ²(μ₀/τ² + Σyᵢ/σ²)

# Generate synthetic data
true_theta = 5.0
sigma = 1.0
n_obs = 20
data = np.random.normal(true_theta, sigma, n_obs)

print(f"\nData: n={n_obs}, sample mean={np.mean(data):.3f}")

# Prior parameters
mu_0 = 0.0
tau = 2.0

# Posterior parameters (analytical, for validation)
tau_n_sq = 1.0 / (1.0/tau**2 + n_obs/sigma**2)
mu_n = tau_n_sq * (mu_0/tau**2 + np.sum(data)/sigma**2)

posterior_dist = stats.norm(mu_n, np.sqrt(tau_n_sq))

print(f"\nPosterior (analytical): N({mu_n:.3f}, {np.sqrt(tau_n_sq):.3f})")

# Define unnormalized posterior: γ(θ) = p(y|θ)p(θ)
def unnormalized_posterior(theta):
    """
    γ(θ) = p(y|θ)p(θ)
         = ∏ᵢ N(yᵢ|θ,σ²) × N(θ|μ₀,τ²)
         ∝ exp(-Σ(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ²)
    """
    # Log-likelihood: log p(y|θ)
    log_likelihood = -0.5 * np.sum((data[:, None] - theta)**2) / sigma**2
    
    # Log-prior: log p(θ)
    log_prior = -0.5 * (theta - mu_0)**2 / tau**2
    
    # Return unnormalized posterior (in log space for numerical stability)
    return np.exp(log_likelihood + log_prior)

# Use prior as proposal (simple choice)
proposal_prior = stats.norm(mu_0, tau)

# Estimate posterior mean: E[θ|y]
h_identity = lambda theta: theta
n_samples = 5000

estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
    unnormalized_posterior, proposal_prior, h_identity, n_samples
)

ess = compute_ess(norm_weights)

# True posterior mean
true_post_mean = mu_n

print(f"\nPosterior Mean E[θ|y]:")
print(f"  True value: {true_post_mean:.6f}")
print(f"  SNIS estimate: {estimate:.6f}")
print(f"  Error: {abs(estimate - true_post_mean):.6f}")
print(f"  ESS: {ess:.1f} ({ess/n_samples*100:.1f}%)")

# Estimate posterior variance: Var[θ|y]
h_centered_square = lambda theta: (theta - estimate)**2
var_estimate, _, _, _ = self_normalized_importance_sampling(
    unnormalized_posterior, proposal_prior, h_centered_square, n_samples
)

true_post_var = tau_n_sq

print(f"\nPosterior Variance Var[θ|y]:")
print(f"  True value: {true_post_var:.6f}")
print(f"  SNIS estimate: {var_estimate:.6f}")
print(f"  Error: {abs(var_estimate - true_post_var):.6f}")


# Example 3: Comparing Proposal Distributions
# =========================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Effect of Proposal Choice on ESS")
print("=" * 70)

# Same setup as Example 1
proposals = {
    'Prior N(0,2)': stats.norm(0, 2),
    'Close to posterior N(5,1.5)': stats.norm(5, 1.5),
    'Posterior (oracle) N(μₙ,τₙ)': posterior_dist,
    'Too narrow N(5,0.5)': stats.norm(5, 0.5),
    'Too wide N(0,4)': stats.norm(0, 4),
}

n_samples = 2000
print(f"\nComparing proposals (n={n_samples}):")
print("-" * 70)

results = []
for name, proposal in proposals.items():
    estimate, samples, norm_weights, _ = self_normalized_importance_sampling(
        unnormalized_posterior, proposal, h_identity, n_samples
    )
    ess = compute_ess(norm_weights)
    efficiency = ess / n_samples * 100
    error = abs(estimate - true_post_mean)
    
    results.append({
        'name': name,
        'estimate': estimate,
        'ess': ess,
        'efficiency': efficiency,
        'error': error
    })
    
    print(f"{name:30s}: ESS={ess:6.1f} ({efficiency:5.1f}%), Error={error:.4f}")

# Visualize ESS comparison
fig, ax = plt.subplots(figsize=(12, 6))
names = [r['name'] for r in results]
efficiencies = [r['efficiency'] for r in results]
colors = ['blue' if 'oracle' not in n.lower() else 'red' for n in names]

bars = ax.bar(range(len(names)), efficiencies, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=12)
ax.set_title('Proposal Efficiency Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=15, ha='right')
ax.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5, 
           label='Perfect efficiency')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example3_proposal_comparison.png")
plt.savefig(fig_path,
            dpi=300, bbox_inches='tight')


# Example 4: Bias vs. Sample Size
# ==============================
print("\n" + "=" * 70)
print("EXAMPLE 4: Bias of Self-Normalized IS")
print("=" * 70)

# Self-normalized IS is biased but consistent
# Bias = O(1/n)

sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
n_replications = 200

print(f"\nInvestigating bias ({n_replications} replications):")
print("-" * 70)

biases = []
std_errors = []

for n in sample_sizes:
    estimates = []
    for _ in range(n_replications):
        est, _, _, _ = self_normalized_importance_sampling(
            unnormalized_posterior, proposal_prior, h_identity, n
        )
        estimates.append(est)
    
    mean_estimate = np.mean(estimates)
    bias = mean_estimate - true_post_mean
    std_error = np.std(estimates)
    
    biases.append(bias)
    std_errors.append(std_error)
    
    print(f"n={n:5d}: Bias={bias:+.6f}, Std Error={std_error:.6f}")

# Plot bias vs sample size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(sample_sizes, biases, 'bo-', linewidth=2, markersize=8, label='Observed bias')
ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Bias', fontsize=12)
ax.set_title('Bias vs Sample Size (Self-Normalized IS)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(sample_sizes, std_errors, 'go-', linewidth=2, markersize=8, 
        label='Standard error')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Standard Error', fontsize=12)
ax.set_title('Standard Error vs Sample Size', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
# Add reference line: std error ~ 1/√n
ax.plot(sample_sizes, 0.5/np.sqrt(sample_sizes), 'r--', linewidth=2, 
        label='O(1/√n) reference')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__),"example4_bias_analysis.png")
plt.savefig(fig_path,
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. Self-normalized IS handles unnormalized target distributions,
   making it ideal for Bayesian inference where p(y) is unknown.

2. The self-normalized estimator is:
   Ê[h(θ)] = Σᵢ wᵢh(θᵢ), where wᵢ = w̃ᵢ/Σⱼw̃ⱼ

3. Properties:
   - Biased but consistent (bias → 0 as n → ∞)
   - Bias = O(1/n)
   - Often has lower variance than normalized IS

4. Effective Sample Size (ESS) measures proposal quality:
   - ESS = 1/Σᵢwᵢ²
   - ESS ≈ n: excellent proposal
   - ESS << n: poor proposal, few samples dominate
   - Efficiency = ESS/n × 100%

5. For Bayesian inference:
   - Unnormalized posterior: γ(θ) = p(y|θ)p(θ)
   - Prior makes a simple proposal choice
   - Better proposals (e.g., Laplace approx.) improve ESS

6. Good proposals are crucial:
   - Should overlap well with posterior
   - Should have heavier tails than posterior
   - Trade-off: computational cost vs. ESS improvement

7. Weight diagnostics are essential:
   - Check ESS
   - Examine weight distribution
   - Look for dominant samples (weight concentration)
""")
