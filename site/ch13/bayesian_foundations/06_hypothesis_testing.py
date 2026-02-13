"""
Bayesian Inference - Module 6: Hypothesis Testing and Model Comparison
Level: Intermediate-Advanced
Topics: Bayes factors, model comparison, hypothesis testing, Savage-Dickey ratio

Bayesian hypothesis testing uses Bayes factors to compare competing hypotheses,
providing a principled way to quantify evidence for one model over another.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ============================================================================
# THEORY: BAYESIAN HYPOTHESIS TESTING
# ============================================================================

"""
BAYES FACTORS:

For two competing hypotheses H₁ and H₂, the Bayes Factor is:

BF₁₂ = p(D|H₁) / p(D|H₂)

This is the ratio of marginal likelihoods (evidence) under each hypothesis.

INTERPRETATION:
- BF > 10: Strong evidence for H₁
- BF = 1-3: Weak evidence for H₁
- BF < 1: Evidence favors H₂
- BF < 0.1: Strong evidence for H₂

POSTERIOR ODDS:

The posterior odds are related to prior odds by:
Posterior Odds = Bayes Factor × Prior Odds

p(H₁|D) / p(H₂|D) = [p(D|H₁) / p(D|H₂)] × [p(H₁) / p(H₂)]

SAVAGE-DICKEY DENSITY RATIO:

For testing a point null hypothesis H₀: θ = θ₀ vs H₁: θ ≠ θ₀,
the Bayes Factor can be computed as:

BF₀₁ = p(θ₀|D) / p(θ₀)

This is the ratio of posterior to prior density at θ₀.
"""

def bayes_factor_beta_binomial(n_heads, n_tails, prior_alpha_h1=1, prior_beta_h1=1):
    """
    Compute Bayes Factor for testing H₀: θ = 0.5 vs H₁: θ ≠ 0.5
    using Beta-Binomial model.
    
    Parameters:
    -----------
    n_heads, n_tails : int
        Observed data
    prior_alpha_h1, prior_beta_h1 : float
        Beta prior parameters under H₁
    
    Returns:
    --------
    bf : float
        Bayes Factor (H₁ vs H₀)
    """
    n = n_heads + n_tails
    
    # Evidence under H₁: Beta-Binomial marginal likelihood
    from scipy.special import beta as beta_func
    evidence_h1 = (beta_func(n_heads + prior_alpha_h1, n_tails + prior_beta_h1) / 
                   beta_func(prior_alpha_h1, prior_beta_h1))
    evidence_h1 *= stats.binom.comb(n, n_heads)
    
    # Evidence under H₀: θ = 0.5
    evidence_h0 = stats.binom.pmf(n_heads, n, 0.5)
    
    # Bayes Factor
    bf_h1_vs_h0 = evidence_h1 / evidence_h0
    
    print("="*70)
    print("BAYES FACTOR: Testing Coin Fairness")
    print("="*70)
    print(f"\nH₀: θ = 0.5 (fair coin)")
    print(f"H₁: θ ≠ 0.5 (biased coin)")
    print(f"\nData: {n_heads} heads, {n_tails} tails")
    print(f"\nEvidence under H₀: {evidence_h0:.6f}")
    print(f"Evidence under H₁: {evidence_h1:.6f}")
    print(f"\nBayes Factor (H₁/H₀): {bf_h1_vs_h0:.4f}")
    
    # Interpret
    if bf_h1_vs_h0 > 10:
        interpretation = "Strong evidence for H₁ (biased)"
    elif bf_h1_vs_h0 > 3:
        interpretation = "Moderate evidence for H₁"
    elif bf_h1_vs_h0 > 1:
        interpretation = "Weak evidence for H₁"
    elif bf_h1_vs_h0 > 0.33:
        interpretation = "Inconclusive"
    elif bf_h1_vs_h0 > 0.1:
        interpretation = "Weak evidence for H₀"
    else:
        interpretation = "Strong evidence for H₀ (fair)"
    
    print(f"Interpretation: {interpretation}")
    
    return bf_h1_vs_h0

def savage_dickey_demo(n_heads, n_tails, prior_alpha=1, prior_beta=1, null_value=0.5):
    """
    Demonstrate Savage-Dickey density ratio for computing Bayes Factor.
    """
    print("\n" + "="*70)
    print("SAVAGE-DICKEY DENSITY RATIO")
    print("="*70)
    
    # Prior and posterior
    prior = stats.beta(prior_alpha, prior_beta)
    posterior = stats.beta(prior_alpha + n_heads, prior_beta + n_tails)
    
    # Evaluate densities at null value
    prior_density = prior.pdf(null_value)
    posterior_density = posterior.pdf(null_value)
    
    # Bayes Factor (H₀ vs H₁)
    bf_h0_vs_h1 = posterior_density / prior_density
    bf_h1_vs_h0 = 1 / bf_h0_vs_h1
    
    print(f"\nTesting H₀: θ = {null_value}")
    print(f"Data: {n_heads} heads, {n_tails} tails")
    print(f"\nPrior density at θ={null_value}: {prior_density:.4f}")
    print(f"Posterior density at θ={null_value}: {posterior_density:.4f}")
    print(f"\nBF₀₁ (Savage-Dickey): {bf_h0_vs_h1:.4f}")
    print(f"BF₁₀: {bf_h1_vs_h0:.4f}")
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    prior_pdf = prior.pdf(theta)
    post_pdf = posterior.pdf(theta)
    
    plt.figure(figsize=(12, 6))
    plt.plot(theta, prior_pdf, 'b--', linewidth=2, label='Prior')
    plt.plot(theta, post_pdf, 'r-', linewidth=2, label='Posterior')
    plt.axvline(null_value, color='green', linestyle=':', linewidth=2, label=f'H₀: θ={null_value}')
    plt.plot(null_value, prior_density, 'bo', markersize=10, label=f'Prior at θ₀={prior_density:.3f}')
    plt.plot(null_value, posterior_density, 'ro', markersize=10, label=f'Posterior at θ₀={posterior_density:.3f}')
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Savage-Dickey Density Ratio', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('savage_dickey.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return bf_h1_vs_h0

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 6: HYPOTHESIS TESTING")
    print("="*70)
    
    # Example 1: Bayes Factor for coin fairness
    bf = bayes_factor_beta_binomial(n_heads=17, n_tails=3)
    
    # Example 2: Savage-Dickey ratio
    bf_sd = savage_dickey_demo(n_heads=17, n_tails=3, prior_alpha=1, prior_beta=1)
    
    print("\n" + "="*70)
    print("MODULE 6 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayes Factors quantify evidence for competing hypotheses")
    print("2. BF combines with prior odds to give posterior odds")
    print("3. Savage-Dickey: ratio of posterior to prior at null value")
    print("4. Bayesian testing avoids p-values and arbitrary thresholds")
    print("\nNext: Module 7 - Hierarchical Models")
    print("="*70)

"""
EXERCISES:

1. Compare Bayesian hypothesis testing with frequentist p-values.
2. Compute Bayes Factors for different sample sizes to understand evidence accumulation.
3. Implement model comparison for nested linear regression models.
4. Use Savage-Dickey to test point nulls in Normal-Normal model.
"""
