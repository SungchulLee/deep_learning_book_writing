"""
Bayesian Inference - Module 10: Advanced Applications
Level: Advanced
Topics: A/B testing, Bayesian optimization, change point detection, practical applications

This module covers practical applications of Bayesian inference in real-world scenarios.

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
BAYESIAN A/B TESTING:

Traditional A/B testing uses p-values and fixed sample sizes.
Bayesian A/B testing provides:
- Probability statements about which variant is better
- Ability to stop early when evidence is strong
- Direct interpretation of results
"""

def bayesian_ab_test(conversions_a, trials_a, conversions_b, trials_b):
    """
    Bayesian A/B test for conversion rates.
    """
    print("="*70)
    print("BAYESIAN A/B TESTING")
    print("="*70)
    
    print(f"\nVariant A: {conversions_a}/{trials_a} = {conversions_a/trials_a:.3f}")
    print(f"Variant B: {conversions_b}/{trials_b} = {conversions_b/trials_b:.3f}")
    
    # Beta posteriors (uniform prior)
    post_a = stats.beta(1 + conversions_a, 1 + trials_a - conversions_a)
    post_b = stats.beta(1 + conversions_b, 1 + trials_b - conversions_b)
    
    # Monte Carlo to compute P(B > A)
    n_samples = 100000
    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)
    prob_b_better = np.mean(samples_b > samples_a)
    
    print(f"\nP(B > A) = {prob_b_better:.4f}")
    print(f"P(A > B) = {1-prob_b_better:.4f}")
    
    # Expected lift
    lift = samples_b / samples_a - 1
    print(f"\nExpected lift (B vs A):")
    print(f"  Mean: {np.mean(lift)*100:.2f}%")
    print(f"  95% Credible Interval: [{np.percentile(lift, 2.5)*100:.2f}%, {np.percentile(lift, 97.5)*100:.2f}%]")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Posterior distributions
    p = np.linspace(0, 1, 1000)
    axes[0].plot(p, post_a.pdf(p), 'b-', linewidth=2, label=f'A ({conversions_a}/{trials_a})')
    axes[0].plot(p, post_b.pdf(p), 'r-', linewidth=2, label=f'B ({conversions_b}/{trials_b})')
    axes[0].axvline(post_a.mean(), color='blue', linestyle='--', alpha=0.7)
    axes[0].axvline(post_b.mean(), color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Conversion Rate', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Posterior Distributions', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Lift distribution
    axes[1].hist(lift * 100, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].axvline(np.mean(lift)*100, color='black', linestyle='-', linewidth=2, label=f'Mean lift={np.mean(lift)*100:.1f}%')
    axes[1].set_xlabel('Lift (% improvement)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Distribution of Lift (B vs A)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('bayesian_ab_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return prob_b_better

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 10: ADVANCED APPLICATIONS")
    print("="*70)
    
    # Example: A/B test
    prob_b_better = bayesian_ab_test(
        conversions_a=120, trials_a=1000,
        conversions_b=145, trials_b=1000
    )
    
    print("\n" + "="*70)
    print("MODULE 10 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayesian A/B testing provides probability statements")
    print("2. Can make early stopping decisions based on evidence")
    print("3. Direct interpretation: P(B better than A)")
    print("4. Naturally handles sequential testing")
    print("\nCongratulations! You've completed the Bayesian Inference curriculum.")
    print("="*70)
