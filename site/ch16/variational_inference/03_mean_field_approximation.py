"""
Variational Inference - Module 3: Mean-Field Approximation
==========================================================

Learning Objectives:
-------------------
1. Understand the mean-field variational family
2. Derive the mean-field ELBO and optimal updates
3. Implement Coordinate Ascent Variational Inference (CAVI)
4. Apply mean-field VI to real problems
5. Understand limitations and trade-offs

Prerequisites:
-------------
- Module 01: Introduction to VI
- Module 02: ELBO Derivation
- Understanding of factorized distributions

Author: Prof. Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import digamma, gammaln
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# SECTION 1: Mean-Field Approximation Concept
# ============================================================================

def explain_mean_field():
    """
    Explain the mean-field approximation and its assumptions.
    
    MEAN-FIELD APPROXIMATION:
    ========================
    
    Idea: Approximate joint posterior with factorized distribution
    
    Full posterior: p(θ|D) over θ = (θ₁, θ₂, ..., θ_K)
    
    Mean-field family:
        Q_MF = {q: q(θ) = ∏ᵢ q_i(θᵢ)}
    
    Each parameter θᵢ has its own variational factor q_i(θᵢ), and
    all factors are mutually independent.
    
    WHY "MEAN-FIELD"?
    ----------------
    Term borrowed from physics: each particle/variable responds to
    the "mean field" created by all other variables, ignoring
    correlations and fluctuations.
    
    MATHEMATICAL FORMULATION:
    ------------------------
    Instead of minimizing KL(q(θ) || p(θ|D)) over all distributions,
    we restrict to:
    
        q(θ) = q_1(θ_1) × q_2(θ_2) × ... × q_K(θ_K)
    
    This dramatically simplifies optimization!
    
    ADVANTAGES:
    ----------
    1. Tractable optimization (coordinate ascent)
    2. Closed-form updates for many models
    3. Fast convergence
    4. Scalable to high dimensions
    
    LIMITATIONS:
    -----------
    1. Cannot capture posterior correlations
    2. Underestimates uncertainty
    3. May give poor approximations for highly coupled parameters
    4. Mode-seeking behavior (forward KL)
    
    WHEN TO USE:
    -----------
    - Parameters are weakly correlated in posterior
    - Speed is critical
    - Interpretability matters
    - Large-scale problems
    """
    
    print("=" * 80)
    print("MEAN-FIELD APPROXIMATION")
    print("=" * 80)
    
    explanation = """
THE MEAN-FIELD ASSUMPTION:
=========================

Given parameters θ = (θ₁, θ₂, ..., θ_K), we approximate:

    p(θ₁, θ₂, ..., θ_K | D) ≈ q(θ) = ∏ᵢ qᵢ(θᵢ)

Each qᵢ(θᵢ) is an independent marginal distribution.

VISUAL INTERPRETATION:
---------------------

True Posterior p(θ₁,θ₂|D):          Mean-Field q(θ₁)q(θ₂):
┌─────────────┐                     ┌─────────────┐
│    ╱╲       │                     │             │
│   ╱  ╲      │    becomes -->      │      ○      │
│  ╱ ○  ╲     │                     │             │
│ ╱      ╲    │                     │             │
└─────────────┘                     └─────────────┘
Correlated, tilted ellipse          Axis-aligned ellipse

The mean-field approximation CANNOT capture the correlation
(tilt) in the true posterior!

MATHEMATICAL CONSEQUENCE:
------------------------

True joint:  p(θ₁,θ₂) may have Cov(θ₁,θ₂) ≠ 0
Mean-field:  q(θ₁,θ₂) = q(θ₁)q(θ₂) forces Cov(θ₁,θ₂) = 0

This is the PRICE we pay for tractability.

OPTIMIZATION STRATEGY:
---------------------

Given the factorization q(θ) = ∏ᵢ qᵢ(θᵢ), we optimize each
factor q_j(θⱼ) while holding all others fixed:

    q*ⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ,D)]}

where q₋ⱼ = ∏ᵢ≠ⱼ qᵢ(θᵢ) represents all factors except j.

This is COORDINATE ASCENT VARIATIONAL INFERENCE (CAVI):
- Update q₁ holding q₂,...,q_K fixed
- Update q₂ holding q₁,q₃,...,q_K fixed
- ...
- Repeat until convergence
"""
    
    print(explanation)
    
    # Visualize mean-field vs full posterior
    visualize_mean_field_limitation()


def visualize_mean_field_limitation():
    """
    Visualize the limitation of mean-field: cannot capture correlations.
    """
    
    print("\n[Generating mean-field limitation visualization...]")
    
    # True posterior: correlated bivariate Gaussian
    mean_true = np.array([1.0, 2.0])
    cov_true = np.array([[1.0, 0.8],
                         [0.8, 1.5]])
    
    # Mean-field approximation: independent Gaussians
    # Optimal mean-field matches marginals
    var1_mf = cov_true[0, 0]
    var2_mf = cov_true[1, 1]
    cov_mf = np.array([[var1_mf, 0.0],
                       [0.0, var2_mf]])
    
    # Generate samples
    np.random.seed(42)
    n_samples = 1000
    samples_true = np.random.multivariate_normal(mean_true, cov_true, n_samples)
    samples_mf = np.random.multivariate_normal(mean_true, cov_mf, n_samples)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: True posterior
    ax = axes[0, 0]
    ax.scatter(samples_true[:, 0], samples_true[:, 1], alpha=0.3, s=10)
    
    # Add confidence ellipse
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_true)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ell = Ellipse(mean_true, width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(ell)
    
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    ax.set_title('(a) True Posterior p(θ₁,θ₂|D)\nCorrelation = 0.74', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Mean-field approximation
    ax = axes[0, 1]
    ax.scatter(samples_mf[:, 0], samples_mf[:, 1], alpha=0.3, s=10, color='green')
    
    # Add confidence ellipse (axis-aligned)
    for n_std in [1, 2]:
        width = 2 * n_std * np.sqrt(var1_mf)
        height = 2 * n_std * np.sqrt(var2_mf)
        ell = Ellipse(mean_true, width, height, angle=0,
                     facecolor='none', edgecolor='blue', linewidth=2)
        ax.add_patch(ell)
    
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    ax.set_title('(b) Mean-Field q(θ₁)q(θ₂)\nCorrelation = 0.00', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Overlay comparison
    ax = axes[0, 2]
    ax.scatter(samples_true[:, 0], samples_true[:, 1], alpha=0.2, s=10, 
              color='red', label='True')
    ax.scatter(samples_mf[:, 0], samples_mf[:, 1], alpha=0.2, s=10, 
              color='green', label='Mean-field')
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    ax.set_title('(c) Overlay Comparison', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 4: Marginal θ₁
    ax = axes[1, 0]
    ax.hist(samples_true[:, 0], bins=30, density=True, alpha=0.5, 
           color='red', label='True p(θ₁|D)')
    ax.hist(samples_mf[:, 0], bins=30, density=True, alpha=0.5, 
           color='green', label='MF q(θ₁)')
    
    # True marginal
    x1_range = np.linspace(-2, 4, 200)
    true_marg1 = stats.norm.pdf(x1_range, mean_true[0], np.sqrt(var1_mf))
    ax.plot(x1_range, true_marg1, 'r-', linewidth=2)
    ax.plot(x1_range, true_marg1, 'g--', linewidth=2)
    
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Marginal Distribution θ₁\n(Matched by mean-field)', 
                fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Marginal θ₂
    ax = axes[1, 1]
    ax.hist(samples_true[:, 1], bins=30, density=True, alpha=0.5, 
           color='red', label='True p(θ₂|D)')
    ax.hist(samples_mf[:, 1], bins=30, density=True, alpha=0.5, 
           color='green', label='MF q(θ₂)')
    
    x2_range = np.linspace(-1, 5, 200)
    true_marg2 = stats.norm.pdf(x2_range, mean_true[1], np.sqrt(var2_mf))
    ax.plot(x2_range, true_marg2, 'r-', linewidth=2)
    ax.plot(x2_range, true_marg2, 'g--', linewidth=2)
    
    ax.set_xlabel('θ₂', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(e) Marginal Distribution θ₂\n(Matched by mean-field)', 
                fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Information lost
    ax = axes[1, 2]
    metrics = ['Marginal\nVariance', 'Joint\nEntropy', 'Mutual\nInfo']
    
    # Compute metrics
    var1_ratio = var1_mf / var1_mf  # Always 1 (marginals match)
    var2_ratio = var2_mf / var2_mf  # Always 1
    
    # Entropy
    H_true = 0.5 * np.log((2*np.pi*np.e)**2 * np.linalg.det(cov_true))
    H_mf = 0.5 * np.log((2*np.pi*np.e)**2 * np.linalg.det(cov_mf))
    
    # Mutual information (0 for mean-field)
    MI_true = 0.5 * np.log(var1_mf * var2_mf / np.linalg.det(cov_true))
    MI_mf = 0.0
    
    true_vals = [1.0, H_true, MI_true]
    mf_vals = [1.0, H_mf, MI_mf]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, true_vals, width, label='True', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, mf_vals, width, label='Mean-field', color='green', alpha=0.7)
    
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(f) Information Comparison', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotations
    ax.text(0.05, 0.95, 'Mean-field:\n• Matches marginals ✓\n• Loses correlations ✗\n• Underestimates MI', 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/tmp/variational_inference/beginner/figures/09_mean_field_limitation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 09_mean_field_limitation.png]")


# ============================================================================
# SECTION 2: CAVI Algorithm - Coordinate Ascent Variational Inference
# ============================================================================

def derive_cavi_updates():
    """
    Derive the general CAVI update equations.
    
    CAVI ALGORITHM:
    ==============
    
    Goal: Optimize ELBO under mean-field assumption
        q(θ) = ∏ᵢ qᵢ(θᵢ)
    
    GENERAL UPDATE FORMULA:
    ----------------------
    For factor j, the optimal update is:
    
        log q*ⱼ(θⱼ) = E_{q₋ⱼ}[log p(θ, D)] + const
    
    Or equivalently:
        q*ⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
    
    where:
    - q₋ⱼ = ∏ᵢ≠ⱼ qᵢ(θᵢ) is the product of all other factors
    - The expectation is taken over all θᵢ for i ≠ j
    - The constant doesn't depend on θⱼ
    
    DERIVATION:
    ----------
    Start with ELBO:
        ELBO(q) = ∫ q(θ) log[p(θ,D)/q(θ)] dθ
    
    Substitute mean-field: q(θ) = ∏ᵢ qᵢ(θᵢ)
    
    Focusing on factor j:
        ELBO = ∫ qⱼ(θⱼ) [∫ q₋ⱼ(θ₋ⱼ) log p(θ,D) dθ₋ⱼ] dθⱼ - ∫ qⱼ(θⱼ) log qⱼ(θⱼ) dθⱼ + const
             = ∫ qⱼ(θⱼ) E_{q₋ⱼ}[log p(θ,D)] dθⱼ + H[qⱼ] + const
    
    This is maximized when:
        qⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ,D)]}
    
    ALGORITHM:
    ---------
    Initialize: q₁⁽⁰⁾, q₂⁽⁰⁾, ..., q_K⁽⁰⁾
    
    Repeat until convergence:
        For j = 1 to K:
            qⱼ⁽ᵗ⁺¹⁾(θⱼ) ∝ exp{E_{q₋ⱼ⁽ᵗ⁾}[log p(θ, D)]}
        Compute ELBO⁽ᵗ⁺¹⁾
        If |ELBO⁽ᵗ⁺¹⁾ - ELBO⁽ᵗ⁾| < ε: break
    
    KEY PROPERTIES:
    --------------
    1. ELBO increases monotonically
    2. Converges to (local) optimum
    3. Often has closed-form updates for exponential families
    4. Easy to implement
    """
    
    print("\n" + "=" * 80)
    print("COORDINATE ASCENT VARIATIONAL INFERENCE (CAVI)")
    print("=" * 80)
    
    explanation = """
THE CAVI UPDATE EQUATION:
========================

For each variational factor j:

    q*ⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}

INTUITION:
---------
- Each factor qⱼ is updated to match the "average" influence from all other factors
- The expectation E_{q₋ⱼ} averages over uncertainty in other parameters
- This is why it's called "mean field" - we respond to the mean/average field

STEPS TO APPLY CAVI:
-------------------

1) Write joint distribution:
   p(θ, D) = p(D|θ) × p(θ)

2) Take logarithm:
   log p(θ, D) = log p(D|θ) + log p(θ)

3) For each factor qⱼ:
   a) Take expectation w.r.t. all other factors
   b) Keep only terms involving θⱼ
   c) Recognize the resulting distribution
   
4) Iterate until ELBO converges

CONJUGACY HELPS:
---------------
If the model has conjugate structure, the CAVI updates often give
distributions in the same family as the prior!

Example: Gaussian likelihood + Gaussian prior → Gaussian q

This makes implementation very efficient.
"""
    
    print(explanation)


# ============================================================================
# SECTION 3: Example - Gaussian Mean and Variance
# ============================================================================

def gaussian_mean_variance_cavi():
    """
    Apply CAVI to infer both mean and variance of a Gaussian.
    
    MODEL:
    -----
    Data: x₁, ..., x_n ~ N(μ, τ⁻¹) where τ = 1/σ² is precision
    Prior on mean: μ ~ N(μ₀, (λ₀τ)⁻¹)
    Prior on precision: τ ~ Gamma(α₀, β₀)
    
    MEAN-FIELD APPROXIMATION:
    ------------------------
    q(μ, τ) = q_μ(μ) × q_τ(τ)
    
    CONJUGACY:
    ---------
    - q_μ(μ) will be Gaussian
    - q_τ(τ) will be Gamma
    
    CAVI UPDATES:
    ------------
    Derived from the general formula: q*ⱼ ∝ exp{E_{q₋ⱼ}[log p(θ,D)]}
    """
    
    print("\n" + "=" * 80)
    print("CAVI EXAMPLE: Gaussian Mean and Variance Estimation")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_data = 50
    mu_true = 5.0
    sigma_true = 2.0
    data = np.random.normal(mu_true, sigma_true, n_data)
    
    print(f"\nData Generation:")
    print(f"  True mean μ = {mu_true}")
    print(f"  True std σ = {sigma_true}")
    print(f"  Sample size n = {n_data}")
    print(f"  Sample mean = {np.mean(data):.3f}")
    print(f"  Sample std = {np.std(data, ddof=1):.3f}")
    
    # Prior parameters
    mu_0 = 0.0      # Prior mean for μ
    lambda_0 = 0.1  # Prior precision coefficient for μ
    alpha_0 = 2.0   # Prior shape for τ
    beta_0 = 2.0    # Prior rate for τ
    
    print(f"\nPrior Hyperparameters:")
    print(f"  μ ~ N(μ₀={mu_0}, (λ₀τ)⁻¹) with λ₀={lambda_0}")
    print(f"  τ ~ Gamma(α₀={alpha_0}, β₀={beta_0})")
    
    # Initialize variational parameters
    # q_μ(μ) = N(μ_n, λ_n⁻¹)
    mu_n = 0.0
    lambda_n = 1.0
    
    # q_τ(τ) = Gamma(α_n, β_n)
    alpha_n = alpha_0
    beta_n = beta_0
    
    # Storage for convergence monitoring
    elbo_history = []
    mu_history = []
    tau_mean_history = []
    
    max_iter = 100
    tolerance = 1e-6
    
    print(f"\nCAVI Optimization:")
    print(f"  Max iterations: {max_iter}")
    print(f"  Tolerance: {tolerance}")
    print("-" * 80)
    
    for iteration in range(max_iter):
        # Current expectations
        E_tau = alpha_n / beta_n
        E_log_tau = digamma(alpha_n) - np.log(beta_n)
        
        # ========================================
        # Update q_μ(μ)
        # ========================================
        # From: q*_μ(μ) ∝ exp{E_τ[log p(D|μ,τ) + log p(μ|τ)]}
        #
        # Result: q_μ(μ) = N(μ_n, λ_n⁻¹) where:
        #   λ_n = λ₀E[τ] + nE[τ]
        #   μ_n = (λ₀μ₀E[τ] + E[τ]∑xᵢ) / λ_n
        
        lambda_n_new = lambda_0 * E_tau + n_data * E_tau
        mu_n_new = (lambda_0 * mu_0 * E_tau + E_tau * np.sum(data)) / lambda_n_new
        
        # ========================================
        # Update q_τ(τ)
        # ========================================
        # From: q*_τ(τ) ∝ exp{E_μ[log p(D|μ,τ) + log p(μ|τ)] + log p(τ)}
        #
        # Result: q_τ(τ) = Gamma(α_n, β_n) where:
        #   α_n = α₀ + n/2 + 1/2
        #   β_n = β₀ + 0.5∑(xᵢ - E[μ])² + 0.5λ₀(E[μ] - μ₀)²
        
        alpha_n_new = alpha_0 + n_data / 2 + 0.5
        
        sum_sq_dev = np.sum((data - mu_n_new)**2) + n_data / lambda_n_new
        prior_term = lambda_0 * ((mu_n_new - mu_0)**2 + 1/lambda_n_new)
        beta_n_new = beta_0 + 0.5 * sum_sq_dev + 0.5 * prior_term
        
        # Compute ELBO
        elbo = compute_elbo_gaussian_mean_var(
            data, mu_n_new, lambda_n_new, alpha_n_new, beta_n_new,
            mu_0, lambda_0, alpha_0, beta_0
        )
        
        elbo_history.append(elbo)
        mu_history.append(mu_n_new)
        tau_mean_history.append(alpha_n_new / beta_n_new)
        
        # Check convergence
        if iteration > 0:
            elbo_change = abs(elbo - elbo_history[-2])
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: ELBO = {elbo:10.4f}, "
                      f"μ = {mu_n_new:6.3f}, E[τ] = {E_tau:6.3f}, "
                      f"ΔELBO = {elbo_change:.2e}")
            
            if elbo_change < tolerance:
                print(f"\n  Converged at iteration {iteration}")
                break
        elif iteration == 0:
            print(f"  Iter {iteration:3d}: ELBO = {elbo:10.4f}, "
                  f"μ = {mu_n_new:6.3f}, E[τ] = {E_tau:6.3f}")
        
        # Update parameters
        mu_n, lambda_n = mu_n_new, lambda_n_new
        alpha_n, beta_n = alpha_n_new, beta_n_new
    
    # Final results
    E_mu = mu_n
    E_tau = alpha_n / beta_n
    E_sigma = np.sqrt(1 / E_tau)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(f"\nVariational Posterior:")
    print(f"  q_μ(μ) = N({mu_n:.4f}, {1/lambda_n:.4f})")
    print(f"  q_τ(τ) = Gamma({alpha_n:.4f}, {beta_n:.4f})")
    
    print(f"\nPosterior Expectations:")
    print(f"  E[μ] = {E_mu:.4f}  (true: {mu_true:.4f})")
    print(f"  E[σ] = {E_sigma:.4f}  (true: {sigma_true:.4f})")
    
    print(f"\nComparison with Sample Statistics:")
    print(f"  Sample mean = {np.mean(data):.4f}")
    print(f"  Sample std = {np.std(data, ddof=1):.4f}")
    
    # Visualize results
    visualize_gaussian_cavi_results(
        data, mu_n, lambda_n, alpha_n, beta_n,
        mu_true, sigma_true, elbo_history, mu_history, tau_mean_history
    )
    
    return mu_n, lambda_n, alpha_n, beta_n


def compute_elbo_gaussian_mean_var(data, mu_n, lambda_n, alpha_n, beta_n,
                                   mu_0, lambda_0, alpha_0, beta_0):
    """
    Compute ELBO for Gaussian mean-variance model.
    
    ELBO = E_q[log p(D,μ,τ)] - E_q[log q(μ,τ)]
         = E_q[log p(D|μ,τ)] + E_q[log p(μ|τ)] + E_q[log p(τ)] 
           - E_q[log q(μ)] - E_q[log q(τ)]
    """
    
    n_data = len(data)
    
    # Current expectations
    E_mu = mu_n
    E_mu_sq = 1/lambda_n + mu_n**2
    E_tau = alpha_n / beta_n
    E_log_tau = digamma(alpha_n) - np.log(beta_n)
    
    # E[log p(D|μ,τ)]
    term1 = 0.5 * n_data * E_log_tau - 0.5 * n_data * np.log(2 * np.pi)
    term1 -= 0.5 * E_tau * np.sum((data - E_mu)**2 + 1/lambda_n)
    
    # E[log p(μ|τ)]
    term2 = 0.5 * E_log_tau + 0.5 * np.log(lambda_0 / (2*np.pi))
    term2 -= 0.5 * lambda_0 * E_tau * ((E_mu - mu_0)**2 + 1/lambda_n)
    
    # E[log p(τ)]
    term3 = alpha_0 * np.log(beta_0) - gammaln(alpha_0)
    term3 += (alpha_0 - 1) * E_log_tau - beta_0 * E_tau
    
    # -E[log q(μ)]
    term4 = -0.5 * np.log(lambda_n / (2*np.pi)) + 0.5
    
    # -E[log q(τ)]
    term5 = -alpha_n * np.log(beta_n) + gammaln(alpha_n)
    term5 -= (alpha_n - 1) * E_log_tau + beta_n * E_tau
    
    elbo = term1 + term2 + term3 + term4 + term5
    
    return elbo


def visualize_gaussian_cavi_results(data, mu_n, lambda_n, alpha_n, beta_n,
                                    mu_true, sigma_true, elbo_history, 
                                    mu_history, tau_mean_history):
    """
    Visualize CAVI results for Gaussian model.
    """
    
    print("\n[Generating CAVI results visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Data histogram with posterior predictive
    ax = axes[0, 0]
    ax.hist(data, bins=20, density=True, alpha=0.6, color='gray', edgecolor='black')
    
    # Posterior predictive: ∫∫ N(x|μ,τ⁻¹) q(μ) q(τ) dμ dτ
    # Approximate by sampling
    mu_samples = np.random.normal(mu_n, 1/np.sqrt(lambda_n), 1000)
    tau_samples = np.random.gamma(alpha_n, 1/beta_n, 1000)
    sigma_samples = 1/np.sqrt(tau_samples)
    
    x_range = np.linspace(data.min()-3, data.max()+3, 200)
    post_pred = np.zeros_like(x_range)
    for mu_s, sig_s in zip(mu_samples, sigma_samples):
        post_pred += stats.norm.pdf(x_range, mu_s, sig_s)
    post_pred /= len(mu_samples)
    
    ax.plot(x_range, post_pred, 'b-', linewidth=2.5, label='Posterior Predictive')
    ax.axvline(mu_true, color='red', linestyle='--', linewidth=2, label='True μ')
    ax.set_xlabel('Data Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(a) Data and Posterior Predictive', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Posterior over μ
    ax = axes[0, 1]
    mu_range = np.linspace(mu_n - 4/np.sqrt(lambda_n), mu_n + 4/np.sqrt(lambda_n), 200)
    q_mu = stats.norm.pdf(mu_range, mu_n, 1/np.sqrt(lambda_n))
    
    ax.plot(mu_range, q_mu, 'b-', linewidth=2.5, label='q(μ)')
    ax.fill_between(mu_range, q_mu, alpha=0.3, color='blue')
    ax.axvline(mu_true, color='red', linestyle='--', linewidth=2, label='True μ')
    ax.axvline(mu_n, color='blue', linestyle=':', linewidth=2, label='E[μ]')
    ax.set_xlabel('Mean μ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Posterior q(μ)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Posterior over τ (precision)
    ax = axes[0, 2]
    tau_range = np.linspace(0.01, alpha_n/beta_n * 2.5, 200)
    q_tau = stats.gamma.pdf(tau_range, alpha_n, scale=1/beta_n)
    
    ax.plot(tau_range, q_tau, 'g-', linewidth=2.5, label='q(τ)')
    ax.fill_between(tau_range, q_tau, alpha=0.3, color='green')
    tau_true = 1/sigma_true**2
    ax.axvline(tau_true, color='red', linestyle='--', linewidth=2, label='True τ')
    ax.axvline(alpha_n/beta_n, color='green', linestyle=':', linewidth=2, label='E[τ]')
    ax.set_xlabel('Precision τ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(c) Posterior q(τ)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: ELBO convergence
    ax = axes[1, 0]
    ax.plot(elbo_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(d) ELBO Convergence', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Mean parameter convergence
    ax = axes[1, 1]
    ax.plot(mu_history, 'b-', linewidth=2, label='E[μ]')
    ax.axhline(mu_true, color='red', linestyle='--', linewidth=2, label='True μ')
    ax.axhline(np.mean(data), color='orange', linestyle=':', linewidth=2, label='Sample mean')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean Parameter', fontsize=11)
    ax.set_title('(e) Mean Convergence', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Precision parameter convergence
    ax = axes[1, 2]
    ax.plot(tau_mean_history, 'g-', linewidth=2, label='E[τ]')
    ax.axhline(tau_true, color='red', linestyle='--', linewidth=2, label='True τ')
    ax.axhline(1/np.var(data), color='orange', linestyle=':', linewidth=2, label='Sample τ')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(f) Precision Convergence', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/variational_inference/beginner/figures/10_cavi_gaussian.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 10_cavi_gaussian.png]")


# ============================================================================
# SECTION 4: Summary
# ============================================================================

def print_summary():
    """
    Print module summary.
    """
    
    print("\n" + "=" * 80)
    print("MODULE SUMMARY: Mean-Field Approximation")
    print("=" * 80)
    
    summary = """
KEY CONCEPTS:
============

1. MEAN-FIELD APPROXIMATION:
   q(θ) = ∏ᵢ qᵢ(θᵢ)
   
   - Assumes posterior factors into independent marginals
   - Cannot capture correlations
   - Dramatically simplifies optimization

2. CAVI UPDATE FORMULA:
   q*ⱼ(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
   
   - Update each factor while holding others fixed
   - Coordinate ascent in function space
   - Guaranteed to increase ELBO

3. ALGORITHM PROPERTIES:
   - Monotonic ELBO improvement
   - Converges to local optimum
   - Often has closed-form updates
   - Fast and scalable

4. CONJUGATE MODELS:
   - If prior/likelihood are conjugate, q stays in same family
   - Updates reduce to hyperparameter updates
   - Very efficient implementation

5. TRADE-OFFS:
   Advantages:
   - Fast convergence
   - Tractable updates
   - Scales well
   - Easy to implement
   
   Limitations:
   - Cannot model correlations
   - Underestimates uncertainty
   - May give poor fits for coupled parameters
   - Local optima possible

6. WHEN TO USE:
   ✓ Parameters weakly correlated
   ✓ Large-scale problems
   ✓ Speed is critical
   ✓ Conjugate structure available
   
   ✗ Strong correlations important
   ✗ Uncertainty quantification critical
   ✗ Small problems where MCMC feasible

IMPLEMENTATION CHECKLIST:
========================

1. □ Write down joint distribution p(θ, D)
2. □ Choose factorization q(θ) = ∏ᵢ qᵢ(θᵢ)
3. □ Derive CAVI updates for each factor
4. □ Implement ELBO computation
5. □ Initialize variational parameters
6. □ Iterate updates until ELBO converges
7. □ Check convergence diagnostics
8. □ Validate against ground truth if available

NEXT STEPS:
==========
Module 04: Gaussian Mixture Model with Variational Inference
Module 05: Exponential Family and General CAVI Framework
"""
    
    print(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    
    print("=" * 80)
    print("VARIATIONAL INFERENCE - MODULE 3")
    print("Mean-Field Approximation and CAVI")
    print("=" * 80)
    print("\nAuthor: Prof. Sungchul")
    print("Institution: Yonsei University")
    print("Email: sungchulyonsei@gmail.com")
    print("=" * 80)
    
    # Create directory
    import os
    os.makedirs('/tmp/variational_inference/beginner/figures', exist_ok=True)
    
    # Run sections
    print("\n[1/3] Explaining mean-field approximation...")
    explain_mean_field()
    
    print("\n[2/3] Deriving CAVI algorithm...")
    derive_cavi_updates()
    
    print("\n[3/3] Applying CAVI to Gaussian model...")
    gaussian_mean_variance_cavi()
    
    print_summary()
    
    print("\n" + "=" * 80)
    print("MODULE COMPLETE!")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  • 09_mean_field_limitation.png")
    print("  • 10_cavi_gaussian.png")
    print("\nNext: Module 04 - Gaussian Mixture Models")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
