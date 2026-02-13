#!/usr/bin/env python3
"""
================================================================================
CAPTURE-RECAPTURE MLE - Wildlife Population Estimation
================================================================================

DIFFICULTY: ⭐⭐ Medium (Level 2)

PROBLEM: Estimate the total population size of animals in a habitat using
the capture-recapture method.

METHODOLOGY:
1. Capture C animals, mark them, and release
2. Later, capture R animals
3. Observe T marked animals in the recapture

QUESTION: What is the total population N?

INTUITION: T/R ≈ C/N  =>  N ≈ (C × R) / T

This is the Lincoln-Petersen estimator, which is the MLE!

MATHEMATICAL MODEL:
The number of marked animals in recapture follows a hypergeometric distribution:
P(T | N) = C(C, T) × C(N-C, R-T) / C(N, R)

where C(n, k) is the binomial coefficient "n choose k"

MLE: Find N that maximizes P(T | N)

REAL APPLICATIONS:
- Wildlife population studies
- Epidemiology (estimating disease prevalence)
- Software testing (estimating number of bugs)
- Census correction (estimating undercounts)

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from typing import Tuple


def compute_hypergeometric_pmf(N: int, C: int, R: int, T: int) -> float:
    """
    Compute probability P(T | N, C, R) using hypergeometric distribution.
    
    P(T) = C(C, T) × C(N-C, R-T) / C(N, R)
    
    Parameters:
    -----------
    N : Total population size
    C : Number initially captured and marked
    R : Number in recapture sample
    T : Number of marked animals in recapture
    
    Returns:
    --------
    probability : Probability of observing T marked animals
    """
    # Check validity
    if T > C or T > R or R - T > N - C or N < C or N < R:
        return 0.0
    
    try:
        # Hypergeometric PMF
        numerator = comb(C, T, exact=True) * comb(N - C, R - T, exact=True)
        denominator = comb(N, R, exact=True)
        prob = numerator / denominator
        return prob
    except:
        return 0.0


def compute_log_likelihood(N: int, C: int, R: int, T: int) -> float:
    """Compute log-likelihood for population size N"""
    prob = compute_hypergeometric_pmf(N, C, R, T)
    if prob > 0:
        return np.log(prob)
    else:
        return -np.inf


def lincoln_petersen_estimator(C: int, R: int, T: int) -> float:
    """
    Compute the Lincoln-Petersen estimator (MLE approximation).
    
    N̂ = (C × R) / T
    
    This is the MLE for large populations and is very intuitive!
    
    Intuition: If T/R = C/N (proportion marked in sample = proportion in population)
    Then: N = (C × R) / T
    """
    if T == 0:
        return float('inf')  # Can't estimate if no recaptures
    return (C * R) / T


def find_mle_exact(C: int, R: int, T: int, max_N: int = 10000) -> Tuple[int, np.ndarray]:
    """
    Find MLE by computing likelihood for all possible N values.
    
    Returns:
    --------
    N_mle : Most likely population size
    likelihoods : Array of likelihoods for each N
    """
    # Minimum possible population
    min_N = max(C, R)
    
    # Compute likelihood for each possible N
    N_values = np.arange(min_N, max_N)
    log_likelihoods = np.array([compute_log_likelihood(N, C, R, T) for N in N_values])
    
    # Find maximum
    valid_mask = np.isfinite(log_likelihoods)
    if not np.any(valid_mask):
        return min_N, log_likelihoods
    
    max_idx = np.argmax(log_likelihoods[valid_mask])
    N_mle = N_values[valid_mask][max_idx]
    
    return N_mle, log_likelihoods


def visualize_results(C: int, R: int, T: int, N_true: int, 
                     N_mle: int, N_lp: float, log_likelihoods: np.ndarray):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========================================================================
    # Plot 1: Likelihood Function
    # ========================================================================
    ax = axes[0, 0]
    
    min_N = max(C, R)
    N_values = np.arange(min_N, min(min_N + len(log_likelihoods), N_true * 3))
    
    # Convert to regular likelihood for plotting
    # Normalize by subtracting max (for numerical stability)
    log_lik_plot = log_likelihoods[:len(N_values) - min_N]
    max_log_lik = np.max(log_lik_plot[np.isfinite(log_lik_plot)])
    likelihood = np.exp(log_lik_plot - max_log_lik)
    
    ax.plot(N_values, likelihood, 'b-', linewidth=2, label='Likelihood')
    ax.axvline(N_mle, color='r', linestyle='-', linewidth=2, label=f'MLE = {N_mle}')
    ax.axvline(N_lp, color='orange', linestyle='--', linewidth=2, 
              label=f'Lincoln-Petersen = {N_lp:.1f}')
    ax.axvline(N_true, color='g', linestyle='--', linewidth=2, label=f'True N = {N_true}')
    
    ax.set_xlabel('Population Size (N)', fontsize=12)
    ax.set_ylabel('Likelihood (normalized)', fontsize=12)
    ax.set_title('Likelihood Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Capture-Recapture Visualization
    # ========================================================================
    ax = axes[0, 1]
    ax.axis('off')
    
    # Create a visual representation
    from matplotlib.patches import Circle, FancyBboxPatch
    
    # Draw population
    ax.text(0.5, 0.95, 'Capture-Recapture Process', 
           ha='center', va='top', fontsize=14, fontweight='bold',
           transform=ax.transAxes)
    
    # Step 1: Initial capture
    box1 = FancyBboxPatch((0.1, 0.65), 0.35, 0.20, 
                          boxstyle="round,pad=0.01", 
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box1)
    ax.text(0.275, 0.75, f'Step 1: Capture & Mark\n{C} animals marked',
           ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Step 2: Release
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.46, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               transform=ax.transAxes)
    ax.text(0.48, 0.78, 'Release', ha='center', fontsize=9, transform=ax.transAxes)
    
    # Step 3: Recapture
    box2 = FancyBboxPatch((0.55, 0.65), 0.35, 0.20,
                          boxstyle="round,pad=0.01",
                          edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(box2)
    ax.text(0.725, 0.75, f'Step 2: Recapture\n{R} animals caught\n{T} are marked',
           ha='center', va='center', fontsize=10, transform=ax.transAxes)
    
    # Results
    box3 = FancyBboxPatch((0.2, 0.30), 0.6, 0.25,
                          boxstyle="round,pad=0.02",
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box3)
    
    results_text = f"""
Observations:
• Initially marked: C = {C}
• Recaptured: R = {R}  
• Marked in recapture: T = {T}

Estimates:
• True population: N = {N_true}
• MLE estimate: N̂ = {N_mle}
• L-P estimate: N̂ = {N_lp:.1f}
• Error: {abs(N_mle - N_true)} animals
"""
    ax.text(0.5, 0.425, results_text, ha='center', va='center',
           fontsize=9, family='monospace', transform=ax.transAxes)
    
    # ========================================================================
    # Plot 3: Error Analysis
    # ========================================================================
    ax = axes[1, 0]
    
    methods = ['True N', 'MLE', 'Lincoln-Petersen']
    values = [N_true, N_mle, N_lp]
    colors = ['green', 'red', 'orange']
    
    bars = ax.barh(methods, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f'  {val:.1f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Population Size', fontsize=12)
    ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # ========================================================================
    # Plot 4: Sampling Distribution Simulation
    # ========================================================================
    ax = axes[1, 1]
    
    # Simulate multiple studies to show sampling variability
    n_simulations = 1000
    estimates = []
    
    for _ in range(n_simulations):
        # Simulate recapture with T following hypergeometric
        possible_T = np.arange(max(0, R - (N_true - C)), min(R, C) + 1)
        probs = [compute_hypergeometric_pmf(N_true, C, R, t) for t in possible_T]
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        simulated_T = np.random.choice(possible_T, p=probs)
        if simulated_T > 0:
            N_est = lincoln_petersen_estimator(C, R, simulated_T)
            if N_est < 10000:  # Reasonable bound
                estimates.append(N_est)
    
    ax.hist(estimates, bins=50, density=True, alpha=0.7, edgecolor='black',
           label='Sampling distribution')
    ax.axvline(N_true, color='g', linestyle='--', linewidth=2, label=f'True N = {N_true}')
    ax.axvline(N_lp, color='r', linestyle='-', linewidth=2, label=f'Our estimate = {N_lp:.1f}')
    ax.axvline(np.mean(estimates), color='orange', linestyle=':', linewidth=2,
              label=f'Mean of estimates = {np.mean(estimates):.1f}')
    
    ax.set_xlabel('Estimated Population Size', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Sampling Variability (1000 simulations)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('capture_recapture_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'capture_recapture_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("CAPTURE-RECAPTURE MLE - Wildlife Population Estimation")
    print("=" * 80)
    
    # ========================================================================
    # Example: Estimating deer population
    # ========================================================================
    print("\n🦌 SCENARIO: Estimating Deer Population")
    print("-" * 80)
    
    # True population (unknown to us in real application)
    N_TRUE = 150
    
    # Study parameters
    C = 30  # Captured and marked in first session
    R = 40  # Captured in second session (recapture)
    T = 8   # Number of marked animals in recapture
    
    print(f"   • Step 1: Captured and marked C = {C} deer")
    print(f"   • Step 2: Recaptured R = {R} deer")
    print(f"   • Observed: T = {T} of them were marked")
    print(f"   • True population: N = {N_TRUE} (unknown in practice)")
    
    # ========================================================================
    # Method 1: Lincoln-Petersen Estimator
    # ========================================================================
    print("\n📐 Method 1: Lincoln-Petersen Estimator")
    print("-" * 80)
    
    N_lp = lincoln_petersen_estimator(C, R, T)
    print(f"   N̂ = (C × R) / T = ({C} × {R}) / {T} = {N_lp:.1f}")
    print(f"   Error: {abs(N_lp - N_TRUE):.1f} animals ({abs(N_lp - N_TRUE)/N_TRUE*100:.1f}%)")
    
    # ========================================================================
    # Method 2: Exact MLE
    # ========================================================================
    print("\n🎯 Method 2: Exact MLE (Hypergeometric)")
    print("-" * 80)
    print("   Computing likelihood for all possible population sizes...")
    
    N_mle, log_likelihoods = find_mle_exact(C, R, T, max_N=500)
    
    print(f"   MLE estimate: N̂ = {N_mle}")
    print(f"   Error: {abs(N_mle - N_TRUE)} animals ({abs(N_mle - N_TRUE)/N_TRUE*100:.1f}%)")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n📊 COMPARISON")
    print("-" * 80)
    print(f"   True population:     N = {N_TRUE}")
    print(f"   Lincoln-Petersen:    N̂ = {N_lp:.1f}")
    print(f"   Exact MLE:           N̂ = {N_mle}")
    print(f"   Difference (L-P vs MLE): {abs(N_lp - N_mle):.1f}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n📊 Creating visualizations...")
    visualize_results(C, R, T, N_TRUE, N_mle, N_lp, log_likelihoods)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print("   The capture-recapture method works!")
    print(f"   • We estimated {N_mle} animals")
    print(f"   • True population is {N_TRUE} animals")
    print(f"   • Estimation error: {abs(N_mle - N_TRUE)/N_TRUE*100:.1f}%")
    print("=" * 80)
    
    print("\n💡 KEY INSIGHTS:")
    print("   1. MLE provides population estimates from limited samples")
    print("   2. Lincoln-Petersen ≈ Exact MLE for large populations")
    print("   3. More captures → better estimates")
    print("   4. Assumes: closed population, equal catchability, marks don't fade")
    print("   5. Widely used in ecology, epidemiology, and software testing!")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. EASY: Try different values of C, R, T
   - What happens if T = 0 (no recaptures)?
   - How does increasing C and R improve accuracy?

2. MEDIUM: Add confidence intervals
   - Use likelihood-based confidence intervals
   - Find N values where likelihood drops by factor of exp(-1.92)

3. MEDIUM: Multiple recapture sessions
   - Extend to 3+ capture sessions
   - Schnabel method for multiple recaptures

4. CHALLENGING: Violations of assumptions
   - Simulate unequal catchability (trap-happy/trap-shy)
   - Population not closed (births, deaths, migration)
   - How robust is MLE to assumption violations?

5. CHALLENGING: Bayesian version
   - Add prior on N (e.g., Uniform or Geometric)
   - Compute posterior distribution
   - Compare Bayesian credible interval to MLE confidence interval
"""


if __name__ == "__main__":
    main()
