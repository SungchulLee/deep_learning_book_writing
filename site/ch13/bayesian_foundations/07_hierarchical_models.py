"""
Bayesian Inference - Module 7: Hierarchical Bayesian Models
Level: Advanced
Topics: Hierarchical models, partial pooling, shrinkage, multi-level inference

Hierarchical models allow parameters to vary by group while sharing statistical
strength across groups through partial pooling.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

"""
HIERARCHICAL MODEL STRUCTURE:

Level 1 (Data): y_ij | θ_i ~ N(θ_i, σ²)
Level 2 (Groups): θ_i | μ, τ ~ N(μ, τ²)
Level 3 (Hyperparameters): μ ~ N(μ₀, σ₀²), τ ~ HalfCauchy(scale)

This creates:
- NO POOLING: Estimate each θ_i independently (ignores group structure)
- COMPLETE POOLING: Assume all θ_i = μ (ignores group differences)
- PARTIAL POOLING: θ_i shrunk toward μ (balances the two extremes)

APPLICATIONS:
- Students within schools
- Patients within hospitals
- Products within categories
- Repeated measurements within subjects
"""

def demonstrate_pooling():
    """
    Demonstrate no pooling vs complete pooling vs partial pooling.
    """
    print("="*70)
    print("HIERARCHICAL MODELS: POOLING STRATEGIES")
    print("="*70)
    
    # Simulate data: 8 schools with different sample sizes
    np.random.seed(42)
    true_mu = 8.0
    true_tau = 5.0
    
    n_schools = 8
    true_effects = np.random.normal(true_mu, true_tau, n_schools)
    sample_sizes = np.array([28, 8, 23, 20, 12, 44, 6, 11])
    sigma = 15.0  # known standard error
    
    observed_means = []
    for i, n in enumerate(sample_sizes):
        obs = np.random.normal(true_effects[i], sigma/np.sqrt(n))
        observed_means.append(obs)
    observed_means = np.array(observed_means)
    
    print(f"\nTrue population mean: {true_mu:.2f}")
    print(f"True between-school std: {true_tau:.2f}")
    print(f"Within-school std: {sigma:.2f}")
    
    # No pooling: independent estimates
    no_pool_estimates = observed_means
    
    # Complete pooling: grand mean
    complete_pool_estimate = np.mean(observed_means)
    complete_pool_estimates = np.full(n_schools, complete_pool_estimate)
    
    # Partial pooling: shrink toward grand mean
    # Weight = n_i / (n_i + σ²/τ²)
    tau_est = np.std(observed_means)  # Simple estimate
    weights = sample_sizes / (sample_sizes + sigma**2 / tau_est**2)
    partial_pool_estimates = (weights * observed_means + 
                             (1 - weights) * complete_pool_estimate)
    
    # Display results
    print("\n" + "-"*70)
    print(f"{'School':<8} {'n':<5} {'True':<8} {'Observed':<12} {'No Pool':<12} {'Partial Pool':<15} {'Complete Pool':<12}")
    print("-"*70)
    for i in range(n_schools):
        print(f"{i+1:<8} {sample_sizes[i]:<5} {true_effects[i]:<8.2f} {observed_means[i]:<12.2f} "
              f"{no_pool_estimates[i]:<12.2f} {partial_pool_estimates[i]:<15.2f} {complete_pool_estimates[i]:<12.2f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Estimates comparison
    x = np.arange(n_schools) + 1
    axes[0].plot(x, true_effects, 'ko-', label='True effects', linewidth=2, markersize=8)
    axes[0].plot(x, observed_means, 'b^--', label='Observed (No pooling)', linewidth=2, markersize=8, alpha=0.7)
    axes[0].plot(x, partial_pool_estimates, 'ro-', label='Partial pooling', linewidth=2, markersize=8)
    axes[0].axhline(complete_pool_estimate, color='green', linestyle=':', linewidth=2, label='Complete pooling')
    axes[0].set_xlabel('School', fontsize=12)
    axes[0].set_ylabel('Effect Estimate', fontsize=12)
    axes[0].set_title('Comparing Pooling Strategies', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Shrinkage visualization
    for i in range(n_schools):
        axes[1].plot([observed_means[i], partial_pool_estimates[i]], 
                    [i+1, i+1], 'r-', linewidth=2, alpha=0.7)
        axes[1].plot(observed_means[i], i+1, 'bo', markersize=10, label='Observed' if i==0 else '')
        axes[1].plot(partial_pool_estimates[i], i+1, 'ro', markersize=10, label='Partial pool' if i==0 else '')
    axes[1].axvline(complete_pool_estimate, color='green', linestyle=':', linewidth=2, label='Grand mean')
    axes[1].set_xlabel('Estimate', fontsize=12)
    axes[1].set_ylabel('School', fontsize=12)
    axes[1].set_title('Shrinkage Toward Grand Mean', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_yticks(range(1, n_schools+1))
    
    plt.tight_layout()
    plt.savefig('hierarchical_pooling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Insight:")
    print("  - Schools with smaller samples are shrunk more toward the grand mean")
    print("  - Partial pooling 'borrows strength' across groups")
    print("  - Provides better estimates than complete or no pooling")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 7: HIERARCHICAL MODELS")
    print("="*70)
    
    demonstrate_pooling()
    
    print("\n" + "="*70)
    print("MODULE 7 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Hierarchical models share information across groups")
    print("2. Partial pooling balances group-specific and population estimates")
    print("3. Small groups benefit most from pooling")
    print("4. Shrinkage is automatic and data-driven")
    print("\nNext: Module 8 - Empirical Bayes")
    print("="*70)
