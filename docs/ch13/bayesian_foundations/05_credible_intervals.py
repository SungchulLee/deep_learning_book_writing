"""
Bayesian Inference - Module 05: Credible Intervals
Level: Intermediate
Topics: Credible intervals, HPD intervals, comparison with confidence intervals

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
# THEORY: CREDIBLE INTERVALS VS CONFIDENCE INTERVALS
# ============================================================================

"""
CREDIBLE INTERVALS (Bayesian):

A 95% credible interval [L, U] for parameter θ satisfies:
P(L ≤ θ ≤ U | Data) = 0.95

Interpretation: Given the observed data, there is a 95% probability that
the true parameter lies in this interval.

TYPES OF CREDIBLE INTERVALS:

1. EQUAL-TAILED INTERVAL:
   - 2.5% probability in each tail
   - Symmetric quantile-based
   - Easy to compute: [q₀.₀₂₅, q₀.₉₇₅]

2. HIGHEST POSTERIOR DENSITY (HPD) INTERVAL:
   - Shortest possible interval
   - All points inside have higher density than points outside
   - Optimal for unimodal posteriors
   - More complex to compute

CONFIDENCE INTERVALS (Frequentist):

A 95% confidence interval means: If we repeat the experiment many times
and compute the interval each time, 95% of those intervals will contain
the true parameter.

KEY DIFFERENCE:
- Credible: probability statement about the parameter (fixed data)
- Confidence: probability statement about the procedure (random data)
"""

def compute_equal_tailed_interval(posterior_dist, alpha=0.05):
    """
    Compute equal-tailed credible interval.
    
    Parameters:
    -----------
    posterior_dist : scipy.stats distribution
        Posterior distribution object
    alpha : float
        Significance level (e.g., 0.05 for 95% interval)
    
    Returns:
    --------
    interval : tuple
        (lower_bound, upper_bound)
    """
    lower = posterior_dist.ppf(alpha / 2)
    upper = posterior_dist.ppf(1 - alpha / 2)
    return (lower, upper)

def compute_hpd_interval(samples, alpha=0.05):
    """
    Compute Highest Posterior Density (HPD) interval from samples.
    
    This is the shortest interval containing (1-alpha)*100% of the probability mass.
    
    Parameters:
    -----------
    samples : array-like
        Samples from posterior distribution
    alpha : float
        Significance level
    
    Returns:
    --------
    interval : tuple
        (lower_bound, upper_bound)
    """
    samples = np.asarray(samples)
    samples_sorted = np.sort(samples)
    n = len(samples)
    
    # Number of samples to include
    n_included = int(np.ceil((1 - alpha) * n))
    
    # Try all possible intervals of this size
    n_intervals = n - n_included + 1
    interval_widths = samples_sorted[n_included-1:] - samples_sorted[:n_intervals]
    
    # Choose interval with minimum width
    min_idx = np.argmin(interval_widths)
    hpd_lower = samples_sorted[min_idx]
    hpd_upper = samples_sorted[min_idx + n_included - 1]
    
    return (hpd_lower, hpd_upper)

def demonstrate_credible_intervals():
    """
    Demonstrate different types of credible intervals.
    """
    print("="*70)
    print("EXAMPLE: COMPARING EQUAL-TAILED AND HPD INTERVALS")
    print("="*70)
    
    # Example with Beta distribution (posterior from coin flips)
    n_heads, n_tails = 15, 5
    post_alpha, post_beta = 1 + n_heads, 1 + n_tails
    
    posterior = stats.beta(post_alpha, post_beta)
    
    print(f"\nPosterior: Beta({post_alpha}, {post_beta})")
    print(f"From {n_heads} heads and {n_tails} tails")
    
    # Equal-tailed interval
    et_interval = compute_equal_tailed_interval(posterior, alpha=0.05)
    print(f"\n95% Equal-Tailed Credible Interval:")
    print(f"  [{et_interval[0]:.4f}, {et_interval[1]:.4f}]")
    print(f"  Width: {et_interval[1] - et_interval[0]:.4f}")
    
    # HPD interval (using samples)
    samples = posterior.rvs(100000)
    hpd_interval = compute_hpd_interval(samples, alpha=0.05)
    print(f"\n95% HPD Credible Interval:")
    print(f"  [{hpd_interval[0]:.4f}, {hpd_interval[1]:.4f}]")
    print(f"  Width: {hpd_interval[1] - hpd_interval[0]:.4f}")
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    pdf = posterior.pdf(theta)
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Equal-tailed
    plt.subplot(1, 2, 1)
    plt.plot(theta, pdf, 'b-', linewidth=2)
    plt.axvline(et_interval[0], color='red', linestyle='--', linewidth=2, label=f'Lower = {et_interval[0]:.3f}')
    plt.axvline(et_interval[1], color='red', linestyle='--', linewidth=2, label=f'Upper = {et_interval[1]:.3f}')
    
    # Shade the interval
    mask = (theta >= et_interval[0]) & (theta <= et_interval[1])
    plt.fill_between(theta[mask], pdf[mask], alpha=0.3, color='red')
    
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Posterior Density', fontsize=12)
    plt.title('95% Equal-Tailed Credible Interval', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: HPD
    plt.subplot(1, 2, 2)
    plt.plot(theta, pdf, 'b-', linewidth=2)
    plt.axvline(hpd_interval[0], color='green', linestyle='--', linewidth=2, label=f'Lower = {hpd_interval[0]:.3f}')
    plt.axvline(hpd_interval[1], color='green', linestyle='--', linewidth=2, label=f'Upper = {hpd_interval[1]:.3f}')
    
    # Shade the interval
    mask = (theta >= hpd_interval[0]) & (theta <= hpd_interval[1])
    plt.fill_between(theta[mask], pdf[mask], alpha=0.3, color='green')
    
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Posterior Density', fontsize=12)
    plt.title('95% HPD Credible Interval (Shortest)', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('credible_intervals_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nNote: For symmetric posteriors, equal-tailed ≈ HPD")
    print(f"For skewed posteriors, HPD is shorter")

def credible_vs_confidence_simulation():
    """
    Simulation demonstrating the difference between credible and confidence intervals.
    """
    print("\n" + "="*70)
    print("SIMULATION: CREDIBLE VS CONFIDENCE INTERVALS")
    print("="*70)
    
    np.random.seed(42)
    true_p = 0.7  # True parameter
    n_trials = 20
    n_experiments = 100
    
    credible_coverage = 0
    confidence_coverage = 0
    
    credible_intervals = []
    confidence_intervals = []
    
    for exp in range(n_experiments):
        # Generate data
        successes = np.random.binomial(n_trials, true_p)
        
        # Bayesian credible interval (Beta posterior with uniform prior)
        post = stats.beta(1 + successes, 1 + n_trials - successes)
        cred_lower, cred_upper = post.ppf(0.025), post.ppf(0.975)
        credible_intervals.append((cred_lower, cred_upper))
        
        if cred_lower <= true_p <= cred_upper:
            credible_coverage += 1
        
        # Frequentist confidence interval (Wald method)
        p_hat = successes / n_trials
        se = np.sqrt(p_hat * (1 - p_hat) / n_trials)
        conf_lower = max(0, p_hat - 1.96 * se)
        conf_upper = min(1, p_hat + 1.96 * se)
        confidence_intervals.append((conf_lower, conf_upper))
        
        if conf_lower <= true_p <= conf_upper:
            confidence_coverage += 1
    
    print(f"\nTrue parameter: {true_p}")
    print(f"Number of experiments: {n_experiments}")
    print(f"Sample size per experiment: {n_trials}")
    
    print(f"\n95% Credible Intervals:")
    print(f"  Coverage: {credible_coverage}/{n_experiments} = {credible_coverage/n_experiments*100:.1f}%")
    
    print(f"\n95% Confidence Intervals:")
    print(f"  Coverage: {confidence_coverage}/{n_experiments} = {confidence_coverage/n_experiments*100:.1f}%")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot credible intervals
    for i, (lower, upper) in enumerate(credible_intervals[:30]):  # Plot first 30
        color = 'green' if lower <= true_p <= upper else 'red'
        ax1.plot([lower, upper], [i, i], color=color, linewidth=1.5, alpha=0.7)
        ax1.plot([lower, upper], [i, i], 'o', color=color, markersize=3)
    
    ax1.axvline(true_p, color='blue', linestyle='--', linewidth=2, label=f'True p = {true_p}')
    ax1.set_xlabel('Parameter Value', fontsize=11)
    ax1.set_ylabel('Experiment Number', fontsize=11)
    ax1.set_title('Bayesian Credible Intervals (95%)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot confidence intervals
    for i, (lower, upper) in enumerate(confidence_intervals[:30]):
        color = 'green' if lower <= true_p <= upper else 'red'
        ax2.plot([lower, upper], [i, i], color=color, linewidth=1.5, alpha=0.7)
        ax2.plot([lower, upper], [i, i], 'o', color=color, markersize=3)
    
    ax2.axvline(true_p, color='blue', linestyle='--', linewidth=2, label=f'True p = {true_p}')
    ax2.set_xlabel('Parameter Value', fontsize=11)
    ax2.set_ylabel('Experiment Number', fontsize=11)
    ax2.set_title('Frequentist Confidence Intervals (95%)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('credible_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nInterpretation:")
    print("  Credible: 'The probability that p is in this interval is 95%'")
    print("  Confidence: 'If we repeat this procedure, 95% of intervals contain p'")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 5: CREDIBLE INTERVALS")
    print("="*70)
    
    demonstrate_credible_intervals()
    credible_vs_confidence_simulation()
    
    print("\n" + "="*70)
    print("MODULE 5 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Credible intervals: probability statements about parameters")
    print("2. Equal-tailed: symmetric quantile-based")
    print("3. HPD: shortest interval containing specified probability")
    print("4. Credible ≠ Confidence intervals (different interpretations)")
    print("\nNext: Module 6 - Hypothesis Testing")
    print("="*70)

"""
EXERCISES:

1. Compute both equal-tailed and HPD intervals for a skewed Gamma posterior.
2. Show that for symmetric distributions, equal-tailed ≈ HPD.
3. Implement credible intervals for multivariate parameters (2D credible regions).
4. Compare coverage of credible vs confidence intervals for small sample sizes.
"""
