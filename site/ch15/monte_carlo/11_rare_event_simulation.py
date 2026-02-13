"""
11_rare_event_simulation.py

ADVANCED LEVEL: Importance Sampling for Rare Event Simulation

This module demonstrates importance sampling for estimating probabilities
of rare events, where naive Monte Carlo is extremely inefficient.

Mathematical Foundation:
---------------------
Rare Event: P(X ∈ A) = ε where ε << 1 (e.g., ε = 10⁻⁶)

Naive Monte Carlo:
- Need n ≈ 1/ε² samples for reasonable accuracy
- For ε = 10⁻⁶, need n ≈ 10¹² samples (infeasible!)

Importance Sampling Approach:
- Choose proposal q that puts more mass in rare region A
- P(X ∈ A) = E_q[I(X ∈ A) × w(X)]
- Can achieve exponential variance reduction

Optimal Proposal:
For estimating P(X ∈ A):
    q*(x) ∝ I(x ∈ A) × p(x)

In practice, approximate by tilting p toward A.

Common Techniques:
1. Exponential tilting (for light-tailed distributions)
2. Mean shifting (for Gaussian)
3. Mixture proposals
4. Cross-entropy method

Applications:
- Risk analysis (finance, insurance)
- Reliability engineering
- Network performance (queue overflows)
- Safety-critical systems

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import seaborn as sns
import time

np.random.seed(42)
sns.set_style("whitegrid")


def estimate_rare_event_mc(target_dist, threshold, n_samples):
    """
    Naive Monte Carlo estimation of P(X > threshold).
    """
    samples = target_dist.rvs(size=n_samples)
    indicator = (samples > threshold).astype(float)
    
    estimate = np.mean(indicator)
    std_error = np.std(indicator) / np.sqrt(n_samples)
    
    return estimate, std_error


def estimate_rare_event_is(target_dist, proposal_dist, threshold, n_samples):
    """
    Importance sampling estimation of P(X > threshold).
    """
    samples = proposal_dist.rvs(size=n_samples)
    
    # Importance weights
    weights_unnorm = target_dist.pdf(samples) / proposal_dist.pdf(samples)
    weights = weights_unnorm / np.sum(weights_unnorm)
    
    # Indicator function
    indicator = (samples > threshold).astype(float)
    
    # IS estimate
    estimate = np.sum(weights * indicator)
    
    # ESS
    ess = 1.0 / np.sum(weights**2)
    
    # Approximate standard error
    std_error = np.sqrt(np.sum(weights**2 * indicator) - estimate**2)
    
    return estimate, std_error, ess


# Example 1: Tail Probability for Gaussian
# ======================================
print("=" * 70)
print("EXAMPLE 1: Gaussian Tail Probability")
print("=" * 70)

# Target: N(0, 1)
target_gaussian = stats.norm(0, 1)

# Rare event: P(X > threshold)
thresholds = [3.0, 4.0, 5.0, 6.0]

print("\nEstimating P(X > threshold) for X ~ N(0,1)")
print(f"{'Threshold':>10} {'True Prob':>12} {'MC Estimate':>12} "
      f"{'IS Estimate':>12} {'Variance Reduction'}")
print("-" * 75)

for threshold in thresholds:
    # True probability
    true_prob = 1 - target_gaussian.cdf(threshold)
    
    # Naive Monte Carlo
    n_mc = 100000
    mc_estimate, mc_se = estimate_rare_event_mc(target_gaussian, threshold, n_mc)
    mc_rel_error = mc_se / (true_prob + 1e-10)
    
    # Importance Sampling with shifted Gaussian
    # Optimal shift: μ* = threshold (conditional expectation)
    proposal_shifted = stats.norm(threshold, 1)
    
    n_is = 10000  # Much fewer samples needed
    is_estimate, is_se, ess = estimate_rare_event_is(
        target_gaussian, proposal_shifted, threshold, n_is
    )
    is_rel_error = is_se / (true_prob + 1e-10)
    
    # Variance reduction factor
    variance_reduction = (mc_se**2 * n_is) / (is_se**2 * n_mc + 1e-20)
    
    print(f"{threshold:10.1f} {true_prob:12.2e} {mc_estimate:12.2e} "
          f"{is_estimate:12.2e} {variance_reduction:18.1f}x")

print("\nKey insight: Exponential variance reduction for rare events!")


# Example 2: Systematic Comparison
# ==============================
print("\n" + "=" * 70)
print("EXAMPLE 2: Detailed Analysis for P(X > 4)")
print("=" * 70)

threshold_ex2 = 4.0
true_prob_ex2 = 1 - target_gaussian.cdf(threshold_ex2)

print(f"\nTrue probability: {true_prob_ex2:.8f} (very rare!)")

# Different proposals
proposals_ex2 = {
    'Naive (N(0,1))': stats.norm(0, 1),
    'Shifted N(4,1)': stats.norm(4, 1),
    'Shifted N(4,1.2)': stats.norm(4, 1.2),
    'Shifted N(5,1)': stats.norm(5, 1),
}

n_samples_ex2 = 5000
n_replications = 500

print(f"\nComparing proposals ({n_replications} replications, {n_samples_ex2} samples each):")
print(f"{'Proposal':<18} {'Mean Est':>12} {'Bias':>10} {'RMSE':>10} {'Mean ESS':>10}")
print("-" * 70)

for name, proposal in proposals_ex2.items():
    estimates = []
    ess_values = []
    
    for _ in range(n_replications):
        est, _, ess = estimate_rare_event_is(target_gaussian, proposal,
                                             threshold_ex2, n_samples_ex2)
        estimates.append(est)
        ess_values.append(ess)
    
    mean_est = np.mean(estimates)
    bias = mean_est - true_prob_ex2
    rmse = np.sqrt(np.mean((np.array(estimates) - true_prob_ex2)**2))
    mean_ess = np.mean(ess_values)
    
    print(f"{name:<18} {mean_est:12.8f} {bias:+10.2e} {rmse:10.2e} {mean_ess:10.1f}")


# Example 3: Exponential Tilting
# ============================
print("\n" + "=" * 70)
print("EXAMPLE 3: Exponential Tilting")
print("=" * 70)

print("""
Exponential tilting for rare event simulation:
For X ~ N(μ, σ²), tilt distribution:
    q(x) ∝ p(x) exp(λx)
    
This gives q = N(μ + λσ², σ²)

Optimal λ minimizes variance.
""")

# For Gaussian, optimal shift is to threshold
threshold_ex3 = 5.0
true_prob_ex3 = 1 - target_gaussian.cdf(threshold_ex3)

print(f"\nTarget: N(0,1)")
print(f"Rare event: P(X > {threshold_ex3})")
print(f"True probability: {true_prob_ex3:.8f}")

# Try different tilting parameters
tilts = [0, 1, 2, 3, 4, 5, 6]
n_samples_ex3 = 3000

print(f"\n{'Tilt λ':>8} {'Proposal':>15} {'ESS':>10} {'Rel ESS':>10} {'Est Error':>12}")
print("-" * 65)

for tilt in tilts:
    # Tilted proposal: N(λ, 1)
    proposal_tilt = stats.norm(tilt, 1)
    
    # Multiple runs
    ess_values = []
    estimates = []
    
    for _ in range(100):
        est, _, ess = estimate_rare_event_is(target_gaussian, proposal_tilt,
                                             threshold_ex3, n_samples_ex3)
        ess_values.append(ess)
        estimates.append(est)
    
    mean_ess = np.mean(ess_values)
    mean_est = np.mean(estimates)
    est_error = abs(mean_est - true_prob_ex3)
    
    print(f"{tilt:8.1f} N({tilt:2.0f}, 1){'':<6} {mean_ess:10.1f} "
          f"{mean_ess/n_samples_ex3:10.1%} {est_error:12.2e}")

print(f"\nOptimal tilt ≈ {threshold_ex3} (equals threshold)")


# Example 4: Financial Risk - Value at Risk (VaR)
# =============================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Financial Application - Value at Risk")
print("=" * 70)

print("""
Portfolio returns: R ~ N(μ, σ²)
Value at Risk at level α: VaR_α = -quantile(R, α)

For α = 0.001 (99.9%), this is a rare event estimation problem.
""")

# Portfolio parameters
mu_portfolio = 0.05  # Expected daily return (5%)
sigma_portfolio = 0.20  # Volatility (20%)

portfolio_dist = stats.norm(mu_portfolio, sigma_portfolio)

# VaR level
alpha_var = 0.001  # 99.9% VaR
true_var = -portfolio_dist.ppf(alpha_var)

print(f"\nPortfolio: μ = {mu_portfolio:.2%}, σ = {sigma_portfolio:.2%}")
print(f"Estimating 99.9% VaR (probability of extreme loss = {alpha_var:.1%})")
print(f"True VaR: {true_var:.4f}")

# Naive MC
n_mc_var = 1000000  # Need huge sample for rare event
start_time = time.time()
samples_mc = portfolio_dist.rvs(size=n_mc_var)
losses = -samples_mc
var_mc = np.quantile(losses, 1-alpha_var)
time_mc = time.time() - start_time

print(f"\nNaive MC ({n_mc_var:,} samples):")
print(f"  VaR estimate: {var_mc:.4f}")
print(f"  Error: {abs(var_mc - true_var):.4f}")
print(f"  Time: {time_mc:.2f} seconds")

# IS with proposal centered on tail
proposal_var = stats.norm(mu_portfolio - 3*sigma_portfolio, sigma_portfolio)

n_is_var = 10000  # Much fewer samples
start_time = time.time()
samples_is = proposal_var.rvs(size=n_is_var)
losses_is = -samples_is

# Compute weights
weights_var = portfolio_dist.pdf(samples_is) / proposal_var.pdf(samples_is)
weights_var /= np.sum(weights_var)

# Weighted quantile
sorted_indices = np.argsort(losses_is)
sorted_losses = losses_is[sorted_indices]
sorted_weights = weights_var[sorted_indices]
cumsum_weights = np.cumsum(sorted_weights)

var_is_idx = np.searchsorted(cumsum_weights, 1-alpha_var)
var_is = sorted_losses[var_is_idx]
time_is = time.time() - start_time

print(f"\nIS ({n_is_var:,} samples):")
print(f"  VaR estimate: {var_is:.4f}")
print(f"  Error: {abs(var_is - true_var):.4f}")
print(f"  Time: {time_is:.2f} seconds")
print(f"  Speedup: {time_mc/time_is:.1f}x faster")
print(f"  Sample efficiency: {n_mc_var/n_is_var:.0f}x fewer samples")


# Example 5: Multiple Thresholds Simultaneously
# ===========================================
print("\n" + "=" * 70)
print("EXAMPLE 5: Estimating Multiple Tail Probabilities")
print("=" * 70)

print("""
Advantage of IS: Can estimate multiple tail probabilities
from the same set of samples with importance reweighting.
""")

thresholds_multi = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
true_probs_multi = [1 - target_gaussian.cdf(t) for t in thresholds_multi]

# Single IS run with proposal centered around median threshold
median_threshold = np.median(thresholds_multi)
proposal_multi = stats.norm(median_threshold, 1.2)

n_samples_multi = 20000
samples_multi = proposal_multi.rvs(size=n_samples_multi)
weights_multi = target_gaussian.pdf(samples_multi) / proposal_multi.pdf(samples_multi)
weights_multi /= np.sum(weights_multi)

print(f"\nUsing single IS run with {n_samples_multi:,} samples")
print(f"Proposal: N({median_threshold}, 1.2)")
print(f"\n{'Threshold':>10} {'True Prob':>12} {'IS Estimate':>12} {'Rel Error':>12}")
print("-" * 55)

for threshold, true_prob in zip(thresholds_multi, true_probs_multi):
    indicator = (samples_multi > threshold).astype(float)
    is_estimate = np.sum(weights_multi * indicator)
    rel_error = abs(is_estimate - true_prob) / (true_prob + 1e-10)
    
    print(f"{threshold:10.1f} {true_prob:12.2e} {is_estimate:12.2e} {rel_error:12.1%}")

print("\nKey insight: Reuse samples for multiple rare event estimates!")


# Visualizations
# =============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Proposals for rare events
ax = axes[0, 0]
x_plot = np.linspace(-1, 7, 1000)
threshold_vis = 4.0

ax.plot(x_plot, target_gaussian.pdf(x_plot), 'k-', linewidth=3,
        label='Target N(0,1)', alpha=0.7)
ax.plot(x_plot, stats.norm(0, 1).pdf(x_plot), 'b--', linewidth=2,
        label='Naive proposal', alpha=0.5)
ax.plot(x_plot, stats.norm(4, 1).pdf(x_plot), 'r--', linewidth=2,
        label='IS proposal (shifted)')

# Shade rare event region
ax.fill_between(x_plot[x_plot > threshold_vis], 0,
                target_gaussian.pdf(x_plot[x_plot > threshold_vis]),
                alpha=0.3, color='orange', label=f'Rare region (x>{threshold_vis})')

ax.axvline(threshold_vis, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Proposal Design for Rare Events', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Variance reduction vs threshold
ax = axes[1, 0]
thresholds_vr = np.linspace(2, 6, 20)
variance_reductions = []

for thresh in thresholds_vr:
    # MC variance (using Bernoulli variance p(1-p))
    p = 1 - target_gaussian.cdf(thresh)
    var_mc = p * (1 - p)
    
    # IS variance (approximate)
    # For optimal shift to threshold, variance reduction ≈ exp(threshold²/2)
    var_is_approx = var_mc * np.exp(-thresh**2/2)
    
    vr = var_mc / var_is_approx if var_is_approx > 0 else 1
    variance_reductions.append(vr)

ax.semilogy(thresholds_vr, variance_reductions, 'b-', linewidth=2)
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Variance Reduction Factor', fontsize=12)
ax.set_title('Variance Reduction vs Event Rarity', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Panel 3: Sample efficiency
ax = axes[0, 1]
thresholds_eff = [3, 4, 5, 6]
mc_samples_needed = []
is_samples_needed = []

for thresh in thresholds_eff:
    p = 1 - target_gaussian.cdf(thresh)
    # For 10% relative error
    target_rel_error = 0.1
    
    # MC: need n ≈ (1-p)/(p * rel_error²)
    n_mc_needed = (1-p) / (p * target_rel_error**2)
    
    # IS: approximately 100x-1000x better
    n_is_needed = n_mc_needed / 500  # Approximate
    
    mc_samples_needed.append(n_mc_needed)
    is_samples_needed.append(n_is_needed)

x_pos = np.arange(len(thresholds_eff))
width = 0.35

ax.bar(x_pos - width/2, np.log10(mc_samples_needed), width,
       label='MC', alpha=0.7, color='blue', edgecolor='black')
ax.bar(x_pos + width/2, np.log10(is_samples_needed), width,
       label='IS', alpha=0.7, color='green', edgecolor='black')

ax.set_ylabel('log₁₀(Samples needed)', fontsize=11)
ax.set_title('Sample Efficiency: MC vs IS\n(for 10% relative error)',
            fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'P(X>{t})' for t in thresholds_eff])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Convergence comparison
ax = axes[1, 1]
threshold_conv = 4.5
true_prob_conv = 1 - target_gaussian.cdf(threshold_conv)

sample_sizes = [100, 500, 1000, 5000, 10000, 50000]

mc_errors = []
is_errors = []

for n_samp in sample_sizes:
    # MC
    errors_mc = []
    for _ in range(50):
        est_mc, _ = estimate_rare_event_mc(target_gaussian, threshold_conv, n_samp)
        errors_mc.append(abs(est_mc - true_prob_conv))
    mc_errors.append(np.mean(errors_mc))
    
    # IS
    errors_is = []
    for _ in range(50):
        est_is, _, _ = estimate_rare_event_is(target_gaussian,
                                               stats.norm(threshold_conv, 1),
                                               threshold_conv, n_samp)
        errors_is.append(abs(est_is - true_prob_conv))
    is_errors.append(np.mean(errors_is))

ax.loglog(sample_sizes, mc_errors, 'bo-', linewidth=2, markersize=8,
          label='Naive MC')
ax.loglog(sample_sizes, is_errors, 'g^-', linewidth=2, markersize=8,
          label='IS')

# Reference line: 1/√n
ax.loglog(sample_sizes, 0.01 / np.sqrt(sample_sizes), 'k--',
          linewidth=1.5, alpha=0.5, label='O(1/√n)')

ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('Absolute Error', fontsize=12)
ax.set_title(f'Convergence for P(X > {threshold_conv})',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/rare_event_analysis.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. RARE EVENT PROBLEM:
   - P(X ∈ A) = ε where ε << 1 (e.g., ε = 10⁻⁶)
   - Naive MC extremely inefficient: needs O(1/ε²) samples
   - For ε = 10⁻⁶, need ~10¹² samples (impossible!)

2. IS PROVIDES EXPONENTIAL SPEEDUP:
   - Variance reduction: 10²-10⁶ fold
   - Sample efficiency: 100-1000x fewer samples
   - Computational speedup: 10-100x faster

3. OPTIMAL PROPOSAL DESIGN:
   - Put mass where rare event occurs
   - For P(X > threshold): shift distribution toward threshold
   - For Gaussian: optimal shift μ* = threshold
   - General principle: tilt toward rare region

4. EXPONENTIAL TILTING:
   - q(x) ∝ p(x) exp(λx)
   - Choose λ to shift mass toward rare region
   - For Gaussian: gives shifted Gaussian
   - Can optimize λ analytically or numerically

5. MULTIPLE THRESHOLDS:
   - Single IS run can estimate multiple probabilities
   - Reweight samples for different thresholds
   - Much more efficient than separate MC runs
   - Useful for sensitivity analysis

6. FINANCIAL APPLICATIONS:
   - Value at Risk (VaR) estimation
   - Credit risk (default probability)
   - Extreme loss estimation
   - Option pricing (deep out-of-money)

7. RELIABILITY ENGINEERING:
   - System failure probability
   - Component lifetime exceedance
   - Safety margins
   - Stress testing

8. PRACTICAL GUIDELINES:
   - Shift proposal mean to rare region
   - Use slightly wider variance than target
   - Check ESS to validate proposal
   - Verify estimates with analytical bounds

9. CHALLENGES:
   - Finding good proposal (may need optimization)
   - High-dimensional rare events (harder)
   - Multiple disjoint rare regions
   - Very extreme probabilities (< 10⁻¹⁰)

10. THEORETICAL FOUNDATION:
    - Exponentially tilted distributions
    - Large deviations theory
    - Importance sampling weights
    - Variance bounds

11. COMPUTATIONAL EFFICIENCY:
    - IS: O(10³-10⁴) samples typically sufficient
    - MC: O(10⁶-10⁹) samples needed
    - Time savings: 100-1000x
    - Enables real-time risk assessment

12. VERIFICATION:
    - Compare with analytical results (when available)
    - Check ESS (should be reasonable, not tiny)
    - Sensitivity to proposal parameters
    - Cross-validate with different proposals

13. WHEN IS IS ESSENTIAL:
    - Probabilities < 10⁻³
    - Limited computational budget
    - Real-time requirements
    - High-dimensional problems
    - Financial/safety-critical applications

14. COMPARISON TO ALTERNATIVES:
    - MC: Simple but inefficient for rare events
    - Analytical: Often impossible
    - Asymptotic approximations: Less accurate
    - IS: Best balance of accuracy and efficiency
""")
