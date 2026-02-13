"""
04_effective_sample_size.py

INTERMEDIATE LEVEL: Effective Sample Size (ESS) - Theory and Diagnostics

This module provides a comprehensive treatment of ESS:
- Mathematical foundations
- Multiple ESS formulations  
- Diagnostics and interpretation
- Relationship to variance

Mathematical Background:
---------------------
ESS measures the "effective" number of independent samples
obtained from importance sampling.

Standard definition:
    ESS = 1 / Σᵢ wᵢ²

where wᵢ are normalized weights.

Alternative formulation (using unnormalized weights):
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²

Properties:
- 1 ≤ ESS ≤ n
- ESS = n when all weights equal (perfect sampling)
- ESS = 1 when one weight dominates (degeneracy)
- ESS/n is the "relative efficiency"

Variance Relationship:
    Var[Ê] ≈ Var_π[h(θ)] / ESS

Thus lower ESS → higher variance in estimates.

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import Tuple, Dict

np.random.seed(42)
sns.set_style("whitegrid")


def compute_ess_normalized(weights: np.ndarray) -> float:
    """
    Compute ESS using normalized weights.
    
    ESS = 1 / Σᵢ wᵢ²
    
    Parameters:
    -----------
    weights : normalized weights (sum to 1)
    
    Returns:
    --------
    ess : float
    """
    return 1.0 / np.sum(weights**2)


def compute_ess_unnormalized(unnormalized_weights: np.ndarray) -> float:
    """
    Compute ESS using unnormalized weights.
    
    ESS = (Σᵢ w̃ᵢ)² / Σᵢ w̃ᵢ²
    
    This is equivalent to normalized version but more numerically stable
    when weights are very small.
    
    Parameters:
    -----------
    unnormalized_weights : unnormalized weights
    
    Returns:
    --------
    ess : float
    """
    sum_weights = np.sum(unnormalized_weights)
    sum_weights_squared = np.sum(unnormalized_weights**2)
    return sum_weights**2 / sum_weights_squared


def compute_weight_statistics(weights: np.ndarray) -> Dict:
    """
    Compute comprehensive statistics for importance weights.
    
    Returns dictionary with various diagnostics.
    """
    n = len(weights)
    ess = compute_ess_normalized(weights)
    
    # Coefficient of variation of weights
    cv = np.std(weights) / (np.mean(weights) + 1e-10)
    
    # Maximum weight
    max_weight = np.max(weights)
    
    # Entropy of weight distribution
    # Higher entropy → more uniform weights
    entropy = -np.sum(weights * np.log(weights + 1e-300))
    max_entropy = np.log(n)  # Uniform distribution
    normalized_entropy = entropy / max_entropy
    
    # Perplexity (another measure related to ESS)
    perplexity = np.exp(entropy)
    
    # Percentage of weight in top samples
    sorted_weights = np.sort(weights)[::-1]
    cumsum_weights = np.cumsum(sorted_weights)
    top_10_pct = np.searchsorted(cumsum_weights, 0.10) + 1
    top_50_pct = np.searchsorted(cumsum_weights, 0.50) + 1
    top_90_pct = np.searchsorted(cumsum_weights, 0.90) + 1
    
    return {
        'ess': ess,
        'relative_ess': ess / n,
        'cv': cv,
        'max_weight': max_weight,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'perplexity': perplexity,
        'top_10_pct': top_10_pct,
        'top_50_pct': top_50_pct,
        'top_90_pct': top_90_pct,
        'n_samples': n
    }


def estimate_variance_inflation(weights: np.ndarray) -> float:
    """
    Estimate variance inflation factor.
    
    Var[IS estimator] ≈ Var[MC estimator] × (1 + Var[w])
    
    where w are unnormalized importance ratios.
    
    Returns:
    --------
    inflation : float
        Multiplicative factor by which variance is inflated
    """
    ess = compute_ess_normalized(weights)
    n = len(weights)
    # Variance inflation ≈ n/ESS
    return n / ess


# Example 1: ESS for Different Proposal Distributions
# =================================================
print("=" * 70)
print("EXAMPLE 1: ESS Depends on Proposal Quality")
print("=" * 70)

# Target distribution: N(5, 1)
target = stats.norm(5, 1)

# Various proposal distributions
proposals = {
    'Perfect': stats.norm(5, 1),           # Same as target
    'Good': stats.norm(5, 1.2),            # Close
    'Okay': stats.norm(4.5, 1.5),          # Reasonable
    'Poor': stats.norm(3, 2),              # Mismatched
    'Bad': stats.norm(5, 0.5),             # Too narrow
    'Terrible': stats.norm(0, 1),          # Very far
}

n_samples = 2000
print(f"\nAnalyzing ESS for {n_samples} samples:\n")
print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Entropy':>8} {'Top 10%':>8}")
print("-" * 70)

results_dict = {}

for name, proposal in proposals.items():
    # Sample from proposal
    samples = proposal.rvs(size=n_samples)
    
    # Compute weights
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    # Compute statistics
    stats_dict = compute_weight_statistics(weights)
    results_dict[name] = stats_dict
    
    print(f"{name:<12} {stats_dict['ess']:8.1f} {stats_dict['relative_ess']:8.2%} "
          f"{stats_dict['cv']:8.2f} {stats_dict['normalized_entropy']:8.2%} "
          f"{stats_dict['top_10_pct']:8d}")


# Visualize weight distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (name, proposal) in enumerate(proposals.items()):
    samples = proposal.rvs(size=n_samples)
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    ax = axes[idx]
    ax.hist(weights * n_samples, bins=50, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=0.5)
    
    ess = compute_ess_normalized(weights)
    ax.set_title(f'{name}: ESS={ess:.1f} ({ess/n_samples:.1%})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Normalized Weight × n', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Uniform')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_proposal_comparison.png',
            dpi=300, bbox_inches='tight')


# Example 2: Relationship Between ESS and Variance
# ==============================================
print("\n" + "=" * 70)
print("EXAMPLE 2: ESS and Estimation Variance")
print("=" * 70)

# Target: N(5, 1)
# Estimate E[θ²]
h_function = lambda x: x**2
true_value = 5**2 + 1**2  # E[θ²] for N(5,1)

# Different proposals with varying ESS
proposals_var = {
    'High ESS': stats.norm(5, 1.1),
    'Medium ESS': stats.norm(4, 1.5),
    'Low ESS': stats.norm(2, 2),
}

n_samples = 1000
n_replications = 500

print(f"\nEstimating E[θ²] = {true_value:.3f}")
print(f"Replications: {n_replications}, Samples per replication: {n_samples}\n")
print(f"{'Proposal':<12} {'ESS':>8} {'Bias':>10} {'Std Dev':>10} {'RMSE':>10}")
print("-" * 60)

for name, proposal in proposals_var.items():
    estimates = []
    ess_values = []
    
    for _ in range(n_replications):
        samples = proposal.rvs(size=n_samples)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        estimate = np.sum(weights * h_function(samples))
        estimates.append(estimate)
        ess_values.append(compute_ess_normalized(weights))
    
    mean_ess = np.mean(ess_values)
    bias = np.mean(estimates) - true_value
    std_dev = np.std(estimates)
    rmse = np.sqrt(np.mean((np.array(estimates) - true_value)**2))
    
    print(f"{name:<12} {mean_ess:8.1f} {bias:+10.4f} {std_dev:10.4f} {rmse:10.4f}")

# Visualize variance vs ESS
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: ESS vs Std Dev (empirical)
ess_vs_std = []
for name, proposal in proposals_var.items():
    estimates = []
    ess_values = []
    
    for _ in range(100):
        samples = proposal.rvs(size=n_samples)
        unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        estimate = np.sum(weights * h_function(samples))
        estimates.append(estimate)
        ess_values.append(compute_ess_normalized(weights))
    
    for e, s in zip(ess_values, estimates):
        ess_vs_std.append((e, s))

ess_array = np.array([x[0] for x in ess_vs_std])
std_array = np.array([x[1] for x in ess_vs_std])

ax = axes[0]
ax.scatter(ess_array, std_array, alpha=0.5, s=30, color='steelblue', edgecolors='black')
ax.set_xlabel('ESS', fontsize=12)
ax.set_ylabel('Estimate', fontsize=12)
ax.set_title('ESS vs Estimate Distribution', fontsize=13, fontweight='bold')
ax.axhline(true_value, color='red', linestyle='--', linewidth=2, label='True value')
ax.grid(True, alpha=0.3)
ax.legend()

# Theoretical variance inflation
ax = axes[1]
ess_range = np.linspace(10, n_samples, 100)
variance_inflation = n_samples / ess_range

ax.plot(ess_range, variance_inflation, 'b-', linewidth=2,
        label='Variance inflation = n/ESS')
ax.set_xlabel('ESS', fontsize=12)
ax.set_ylabel('Variance Inflation Factor', fontsize=12)
ax.set_title('Theoretical Variance Inflation', fontsize=13, fontweight='bold')
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='No inflation')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_variance_relationship.png',
            dpi=300, bbox_inches='tight')


# Example 3: ESS as Function of Sample Size
# =======================================
print("\n" + "=" * 70)
print("EXAMPLE 3: How ESS Scales with Sample Size")
print("=" * 70)

# For a fixed proposal, how does ESS grow with n?
proposal_fixed = stats.norm(4, 1.5)
sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

print("\nSample Size vs ESS (averaged over 100 runs):\n")
print(f"{'n':>8} {'Mean ESS':>10} {'ESS/n':>10} {'Std ESS':>10}")
print("-" * 42)

ess_by_n = []
for n in sample_sizes:
    ess_list = []
    for _ in range(100):
        samples = proposal_fixed.rvs(size=n)
        unnorm_weights = target.pdf(samples) / proposal_fixed.pdf(samples)
        weights = unnorm_weights / np.sum(unnorm_weights)
        ess_list.append(compute_ess_normalized(weights))
    
    mean_ess = np.mean(ess_list)
    std_ess = np.std(ess_list)
    ess_by_n.append((n, mean_ess, std_ess))
    
    print(f"{n:8d} {mean_ess:10.1f} {mean_ess/n:10.3f} {std_ess:10.1f}")

# Plot ESS vs n
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ns = [x[0] for x in ess_by_n]
ess_means = [x[1] for x in ess_by_n]
ess_stds = [x[2] for x in ess_by_n]
rel_ess = [e/n for e, n in zip(ess_means, ns)]

ax = axes[0]
ax.errorbar(ns, ess_means, yerr=ess_stds, fmt='o-', linewidth=2,
            markersize=8, capsize=5, color='steelblue', label='ESS')
ax.plot(ns, ns, 'r--', linewidth=2, label='Perfect (ESS=n)')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS vs Sample Size', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(ns, rel_ess, 'o-', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel('Sample Size n', fontsize=12)
ax.set_ylabel('Relative ESS (ESS/n)', fontsize=12)
ax.set_title('Relative ESS vs Sample Size', fontsize=13, fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_vs_sample_size.png',
            dpi=300, bbox_inches='tight')


# Example 4: Weight Concentration Diagnostic
# ========================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Weight Concentration Analysis")
print("=" * 70)

def analyze_weight_concentration(weights: np.ndarray, name: str):
    """
    Detailed analysis of how weights are concentrated.
    """
    n = len(weights)
    sorted_weights = np.sort(weights)[::-1]
    cumsum = np.cumsum(sorted_weights)
    
    # Find percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    n_for_percentile = []
    
    for p in percentiles:
        idx = np.searchsorted(cumsum, p/100.0)
        n_for_percentile.append(idx + 1)
    
    print(f"\n{name}:")
    print(f"  Total samples: {n}")
    print(f"  ESS: {compute_ess_normalized(weights):.1f}")
    print("\n  Weight concentration:")
    for p, n_samples in zip(percentiles, n_for_percentile):
        pct_samples = n_samples / n * 100
        print(f"    {n_samples:5d} samples ({pct_samples:5.1f}%) account for {p}% of weight")
    
    return sorted_weights, cumsum

# Compare three proposals
test_proposals = {
    'Good (ESS high)': stats.norm(5, 1.1),
    'Medium (ESS mid)': stats.norm(4, 1.5),
    'Poor (ESS low)': stats.norm(2, 2),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, proposal) in enumerate(test_proposals.items()):
    samples = proposal.rvs(size=2000)
    unnorm_weights = target.pdf(samples) / proposal.pdf(samples)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    sorted_weights, cumsum = analyze_weight_concentration(weights, name)
    
    ax = axes[idx]
    ax.plot(np.arange(1, len(sorted_weights)+1), cumsum, 
            linewidth=2, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Number of Top Samples', fontsize=11)
    ax.set_ylabel('Cumulative Weight', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/weight_concentration.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. EFFECTIVE SAMPLE SIZE (ESS) quantifies importance sampling quality:
   - ESS = 1/Σᵢwᵢ² (normalized weights)
   - ESS = (Σᵢw̃ᵢ)²/Σᵢw̃ᵢ² (unnormalized weights)
   - Range: 1 ≤ ESS ≤ n

2. INTERPRETATION:
   - ESS/n ≈ 1: Excellent, weights nearly uniform
   - ESS/n ≈ 0.5: Good, half effective samples
   - ESS/n < 0.1: Poor, consider better proposal
   - ESS/n << 0.01: Bad, few samples dominate

3. VARIANCE RELATIONSHIP:
   - Variance inflation ≈ n/ESS
   - Lower ESS → higher variance in estimates
   - Can't be fixed by increasing n without improving proposal

4. WEIGHT CONCENTRATION:
   - ESS low → few samples carry most weight
   - Check: how many samples for 50% of weight?
   - Ideally spread across many samples

5. DIAGNOSTICS TO ALWAYS CHECK:
   - ESS and relative ESS (ESS/n)
   - Weight coefficient of variation
   - Weight concentration (top 10%, 50%, 90%)
   - Maximum weight value
   - Weight entropy

6. WHEN ESS IS LOW:
   - Don't just increase n (won't help much)
   - Improve proposal distribution
   - Consider adaptive importance sampling
   - Or switch to MCMC methods

7. ESS SCALES LINEARLY WITH n:
   - If ESS/n ≈ c for n samples
   - Then ESS/n ≈ c for any n (approximately)
   - The relative efficiency is roughly constant

8. PRACTICAL RULES:
   - ESS > 1000: usually sufficient for most applications
   - ESS/n > 0.1: acceptable efficiency
   - ESS/n < 0.01: definitely need better proposal
""")
