"""
Variational Inference - Module 2: Evidence Lower Bound (ELBO) Derivation
=========================================================================

Learning Objectives:
-------------------
1. Derive the Evidence Lower Bound (ELBO) from first principles
2. Understand the relationship between ELBO and marginal likelihood
3. Learn multiple formulations of ELBO
4. Implement ELBO computation for simple models
5. Visualize ELBO optimization

Prerequisites:
-------------
- Module 01: Introduction to VI
- Understanding of KL divergence
- Basic calculus and expectations

Author: Prof. Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# SECTION 1: Deriving the ELBO
# ============================================================================

def derive_elbo():
    """
    Step-by-step derivation of the Evidence Lower Bound (ELBO).
    
    MATHEMATICAL DERIVATION:
    =======================
    
    Goal: Approximate intractable posterior p(θ|D) with q(θ)
    
    Problem: We want to minimize KL(q(θ) || p(θ|D)), but this requires p(θ|D)
             which contains the intractable term p(D).
    
    STEP 1: Write out KL divergence
    --------------------------------
    KL(q(θ) || p(θ|D)) = ∫ q(θ) log[q(θ)/p(θ|D)] dθ
                       = E_q[log q(θ)] - E_q[log p(θ|D)]
    
    STEP 2: Apply Bayes' rule to posterior
    ---------------------------------------
    p(θ|D) = p(D|θ)p(θ) / p(D)
    
    Therefore:
    log p(θ|D) = log p(D|θ) + log p(θ) - log p(D)
    
    STEP 3: Substitute into KL divergence
    -------------------------------------
    KL(q(θ) || p(θ|D)) = E_q[log q(θ)] - E_q[log p(D|θ) + log p(θ) - log p(D)]
                       = E_q[log q(θ)] - E_q[log p(D|θ)] - E_q[log p(θ)] + E_q[log p(D)]
    
    Since log p(D) doesn't depend on θ:
    = E_q[log q(θ)] - E_q[log p(D|θ)] - E_q[log p(θ)] + log p(D)
    
    STEP 4: Rearrange to isolate log p(D)
    -------------------------------------
    log p(D) = KL(q(θ) || p(θ|D)) + E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
             = KL(q(θ) || p(θ|D)) + E_q[log p(D,θ)] - E_q[log q(θ)]
             = KL(q(θ) || p(θ|D)) + E_q[log p(D,θ) - log q(θ)]
    
    where p(D,θ) = p(D|θ)p(θ) is the joint probability.
    
    STEP 5: Define the ELBO
    -----------------------
    Define the Evidence Lower Bound:
    
        ELBO(q) = E_q[log p(D,θ)] - E_q[log q(θ)]
                = E_q[log p(D,θ) - log q(θ)]
    
    From Step 4:
        log p(D) = KL(q(θ) || p(θ|D)) + ELBO(q)
    
    Since KL ≥ 0:
        log p(D) ≥ ELBO(q)
    
    Hence the name "Evidence Lower Bound"!
    
    KEY INSIGHT:
    -----------
    Maximizing ELBO(q) ⟺ Minimizing KL(q(θ) || p(θ|D))
    
    And we can compute ELBO without knowing p(D)!
    """
    
    print("=" * 80)
    print("EVIDENCE LOWER BOUND (ELBO) DERIVATION")
    print("=" * 80)
    
    derivation = """
COMPLETE MATHEMATICAL DERIVATION:
=================================

Starting Point:
--------------
We want to minimize: KL(q(θ) || p(θ|D))

But p(θ|D) = p(D|θ)p(θ) / p(D) contains intractable p(D) = ∫ p(D|θ)p(θ) dθ

Step-by-Step Derivation:
-----------------------

1) KL(q || p) = E_q[log q(θ) - log p(θ|D)]

2) Apply Bayes' rule: log p(θ|D) = log p(D|θ) + log p(θ) - log p(D)

3) Substitute:
   KL(q || p) = E_q[log q(θ) - log p(D|θ) - log p(θ) + log p(D)]
              = E_q[log q(θ)] - E_q[log p(D|θ)] - E_q[log p(θ)] + log p(D)
              = E_q[log q(θ) - log p(D,θ)] + log p(D)

4) Rearrange:
   log p(D) = KL(q || p) + E_q[log p(D,θ) - log q(θ)]
            = KL(q || p) + ELBO(q)

5) Since KL(q || p) ≥ 0:
   log p(D) ≥ ELBO(q)

ALTERNATIVE FORMULATIONS OF ELBO:
=================================

Formulation 1 (Joint-based):
---------------------------
ELBO(q) = E_q[log p(D,θ)] - E_q[log q(θ)]
        = E_q[log p(D,θ) - log q(θ)]

Formulation 2 (Expectation-based):
---------------------------------
ELBO(q) = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
        = E_q[log p(D|θ)] - KL(q(θ) || p(θ))

Interpretation: Expected log-likelihood - KL divergence from prior

Formulation 3 (Entropy-based):
-----------------------------
ELBO(q) = E_q[log p(D|θ)] + E_q[log p(θ) - log q(θ)]
        = E_q[log p(D|θ)] - KL(q(θ) || p(θ))
        = E_q[log p(D|θ)] + H[q] - E_q[-log p(θ)]

where H[q] = -E_q[log q(θ)] is the entropy of q

KEY RELATIONSHIPS:
==================

1) Decomposition:
   log p(D) = ELBO(q) + KL(q(θ) || p(θ|D))

2) Optimization:
   max_q ELBO(q) ⟺ min_q KL(q(θ) || p(θ|D))

3) At optimum (if achievable):
   q*(θ) = p(θ|D)  ⟹  ELBO(q*) = log p(D)

4) Gap interpretation:
   log p(D) - ELBO(q) = KL(q(θ) || p(θ|D))
   
   The gap between log p(D) and ELBO equals the KL divergence!
"""
    
    print(derivation)
    
    # Visualize the decomposition
    visualize_elbo_decomposition()


def visualize_elbo_decomposition():
    """
    Visualize the relationship: log p(D) = ELBO(q) + KL(q || p)
    """
    
    print("\n[Generating visualization of ELBO decomposition...]")
    
    # Create a simple 1D example
    x = np.linspace(-4, 4, 1000)
    
    # True posterior (bimodal for illustration)
    p = 0.6 * stats.norm.pdf(x, -1, 0.5) + 0.4 * stats.norm.pdf(x, 2, 0.6)
    p = p / np.trapezoid(p, x)
    
    # Several approximations with different KL values
    qs = [
        (0.0, 1.0, 'Poor'),   # Far from true posterior
        (0.5, 1.2, 'Better'),  # Closer
        (0.3, 0.9, 'Best'),    # Best single Gaussian fit
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    log_p_D_true = 0  # We'll normalize this to 0 for visualization
    colors = ['red', 'orange', 'green']
    
    for idx, (mu_q, sig_q, label) in enumerate(qs):
        q = stats.norm.pdf(x, mu_q, sig_q)
        q = q / np.trapezoid(q, x)
        
        # Compute KL divergence numerically
        eps = 1e-10
        kl = np.trapezoid(q * np.log((q + eps) / (p + eps)), x)
        
        # For this visualization, set log p(D) = 0 (normalized)
        elbo = -kl  # Since log p(D) = ELBO + KL, and we set log p(D) = 0
        
        # Plot distributions
        ax = axes[0, 0] if idx == 0 else (axes[0, 1] if idx == 1 else axes[1, 0])
        ax.plot(x, p, 'b-', linewidth=2.5, label='True p(θ|D)', alpha=0.8)
        ax.fill_between(x, p, alpha=0.2, color='blue')
        ax.plot(x, q, color=colors[idx], linestyle='--', linewidth=2, 
                label=f'Approx q(θ) [{label}]')
        ax.fill_between(x, q, alpha=0.3, color=colors[idx])
        ax.set_xlabel('Parameter θ', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'({chr(97+idx)}) {label} Approximation\nKL = {kl:.3f}, ELBO = {elbo:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Summary bar plot
    ax = axes[1, 1]
    labels_list = [q[2] for q in qs]
    kls = [np.trapezoid(stats.norm.pdf(x, q[0], q[1])/np.trapezoid(stats.norm.pdf(x, q[0], q[1]), x) * 
                    np.log((stats.norm.pdf(x, q[0], q[1])/np.trapezoid(stats.norm.pdf(x, q[0], q[1]), x) + 1e-10) / 
                           (p + 1e-10)), x) for q in qs]
    elbos = [-kl for kl in kls]
    
    x_pos = np.arange(len(labels_list))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, kls, width, label='KL(q||p)', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, elbos, width, label='ELBO', color='green', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Approximation Quality', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('(d) ELBO vs KL Divergence', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation
    ax.text(0.5, 0.95, 'log p(D) = ELBO + KL(q||p)\n(normalized to 0)', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '06_elbo_decomposition.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 06_elbo_decomposition.png]")


# ============================================================================
# SECTION 2: Computing ELBO for Simple Models
# ============================================================================

def compute_elbo_gaussian():
    """
    Compute ELBO for the simple Gaussian example from Module 1.
    
    Model:
    -----
    - Data: x_i ~ N(θ, σ²) for i = 1, ..., n
    - Prior: θ ~ N(μ₀, σ₀²)
    - Variational approximation: q(θ) = N(m, s²)
    
    ELBO Computation:
    ----------------
    ELBO(q) = E_q[log p(D,θ)] - E_q[log q(θ)]
            = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
    
    For Gaussian model with Gaussian variational family:
    - E_q[log p(D|θ)] = sum of log-likelihoods under q
    - E_q[log p(θ)] = expected log-prior under q
    - E_q[log q(θ)] = negative entropy of q
    
    All these expectations have closed-form expressions!
    """
    
    print("\n" + "=" * 80)
    print("COMPUTING ELBO: Gaussian Mean Estimation")
    print("=" * 80)
    
    # Set up model
    np.random.seed(42)
    theta_true = 2.5
    sigma = 1.0
    n_data = 20
    
    # Prior
    mu_0 = 0.0
    sigma_0 = 2.0
    
    # Generate data
    data = np.random.normal(theta_true, sigma, n_data)
    data_mean = np.mean(data)
    
    print(f"\nModel:")
    print(f"  Data: {n_data} observations from N(θ, {sigma}²)")
    print(f"  Prior: θ ~ N({mu_0}, {sigma_0}²)")
    print(f"  Sample mean: {data_mean:.3f}")
    
    def compute_elbo(m, s):
        """
        Compute ELBO for variational approximation q(θ) = N(m, s²).
        
        ELBO = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]
        
        Each term:
        ---------
        1) E_q[log p(D|θ)]:
           For x_i ~ N(θ, σ²):
           log p(x_i|θ) = -0.5 log(2πσ²) - (x_i - θ)²/(2σ²)
           
           E_q[log p(D|θ)] = sum_i E_q[log p(x_i|θ)]
                           = -n/2 log(2πσ²) - 1/(2σ²) sum_i E_q[(x_i - θ)²]
                           = -n/2 log(2πσ²) - 1/(2σ²) sum_i [(x_i - m)² + s²]
        
        2) E_q[log p(θ)]:
           For θ ~ N(μ₀, σ₀²):
           E_q[log p(θ)] = -0.5 log(2πσ₀²) - E_q[(θ - μ₀)²]/(2σ₀²)
                         = -0.5 log(2πσ₀²) - [(m - μ₀)² + s²]/(2σ₀²)
        
        3) E_q[log q(θ)]:
           For q(θ) = N(m, s²):
           E_q[log q(θ)] = -0.5 log(2πs²) - E_q[(θ - m)²]/(2s²)
                         = -0.5 log(2πs²) - 0.5
                         = -0.5 log(2πs²e)
           
           Or equivalently, this is negative entropy: -H[q]
        """
        
        # Term 1: Expected log-likelihood
        term1 = -0.5 * n_data * np.log(2 * np.pi * sigma**2)
        term1 -= (1 / (2 * sigma**2)) * np.sum([(x - m)**2 + s**2 for x in data])
        
        # Term 2: Expected log-prior
        term2 = -0.5 * np.log(2 * np.pi * sigma_0**2)
        term2 -= ((m - mu_0)**2 + s**2) / (2 * sigma_0**2)
        
        # Term 3: Negative entropy of q
        term3 = -0.5 * np.log(2 * np.pi * np.e * s**2)
        
        elbo = term1 + term2 + term3
        
        return elbo, term1, term2, term3
    
    # Compute ELBO for several values of m and s
    print("\n" + "-" * 80)
    print("ELBO Computation for Different Variational Parameters:")
    print("-" * 80)
    
    test_params = [
        (0.0, 2.0, "Initial guess"),
        (1.0, 1.5, "Better approximation"),
        (data_mean, 0.5, "Close to optimum"),
    ]
    
    for m, s, description in test_params:
        elbo, t1, t2, t3 = compute_elbo(m, s)
        print(f"\nq(θ) = N(m={m:.2f}, s²={s**2:.2f}) - {description}")
        print(f"  E_q[log p(D|θ)] = {t1:.3f}")
        print(f"  E_q[log p(θ)]   = {t2:.3f}")
        print(f"  -H[q]           = {t3:.3f}")
        print(f"  ELBO            = {elbo:.3f}")
    
    # Optimize ELBO
    print("\n" + "-" * 80)
    print("Optimizing ELBO:")
    print("-" * 80)
    
    def neg_elbo(params):
        """Negative ELBO for minimization."""
        m, log_s = params
        s = np.exp(log_s)  # Ensure s > 0
        elbo, _, _, _ = compute_elbo(m, s)
        return -elbo
    
    # Initialize and optimize
    init_params = [0.0, np.log(1.0)]
    result = minimize(neg_elbo, init_params, method='L-BFGS-B')
    
    m_opt = result.x[0]
    s_opt = np.exp(result.x[1])
    elbo_opt, _, _, _ = compute_elbo(m_opt, s_opt)
    
    print(f"\nOptimized variational parameters:")
    print(f"  m* = {m_opt:.4f}")
    print(f"  s* = {s_opt:.4f}")
    print(f"  ELBO(q*) = {elbo_opt:.4f}")
    
    # Compare with exact posterior
    precision_0 = 1 / sigma_0**2
    precision_data = n_data / sigma**2
    precision_n = precision_0 + precision_data
    sigma_n = 1 / np.sqrt(precision_n)
    mu_n = (precision_0 * mu_0 + precision_data * data_mean) / precision_n
    
    print(f"\nExact posterior parameters:")
    print(f"  μₙ = {mu_n:.4f}")
    print(f"  σₙ = {sigma_n:.4f}")
    
    print(f"\nDifference:")
    print(f"  |m* - μₙ| = {abs(m_opt - mu_n):.6f}")
    print(f"  |s* - σₙ| = {abs(s_opt - sigma_n):.6f}")
    
    # Visualize ELBO surface
    visualize_elbo_surface(data, mu_0, sigma_0, sigma, mu_n, sigma_n)
    
    return m_opt, s_opt


def visualize_elbo_surface(data, mu_0, sigma_0, sigma, mu_exact, sigma_exact):
    """
    Visualize the ELBO as a function of variational parameters.
    """
    
    print("\n[Generating ELBO surface visualization...]")
    
    n_data = len(data)
    
    # Define grid
    m_range = np.linspace(-1, 4, 100)
    s_range = np.linspace(0.1, 2.5, 100)
    M, S = np.meshgrid(m_range, s_range)
    
    # Compute ELBO for each point
    ELBO = np.zeros_like(M)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            m, s = M[i, j], S[i, j]
            
            # Expected log-likelihood
            term1 = -0.5 * n_data * np.log(2 * np.pi * sigma**2)
            term1 -= (1 / (2 * sigma**2)) * np.sum([(x - m)**2 + s**2 for x in data])
            
            # Expected log-prior
            term2 = -0.5 * np.log(2 * np.pi * sigma_0**2)
            term2 -= ((m - mu_0)**2 + s**2) / (2 * sigma_0**2)
            
            # Negative entropy
            term3 = -0.5 * np.log(2 * np.pi * np.e * s**2)
            
            ELBO[i, j] = term1 + term2 + term3
    
    # Create visualization
    fig = plt.figure(figsize=(16, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(M, S, ELBO, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.scatter([mu_exact], [sigma_exact], [ELBO.max()], color='red', s=100, 
               marker='*', label='Exact posterior')
    ax1.set_xlabel('Mean (m)', fontsize=10)
    ax1.set_ylabel('Std Dev (s)', fontsize=10)
    ax1.set_zlabel('ELBO', fontsize=10)
    ax1.set_title('(a) ELBO Surface', fontsize=11, fontweight='bold')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(M, S, ELBO, levels=20, cmap='viridis')
    ax2.contour(M, S, ELBO, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax2.plot(mu_exact, sigma_exact, 'r*', markersize=15, label='Exact posterior')
    ax2.set_xlabel('Mean (m)', fontsize=10)
    ax2.set_ylabel('Std Dev (s)', fontsize=10)
    ax2.set_title('(b) ELBO Contours', fontsize=11, fontweight='bold')
    ax2.legend()
    fig.colorbar(contour, ax=ax2)
    ax2.grid(True, alpha=0.3)
    
    # Cross-sections
    ax3 = fig.add_subplot(133)
    
    # Fix s at optimal value, vary m
    idx_s_opt = np.argmin(np.abs(s_range - sigma_exact))
    elbo_vs_m = ELBO[idx_s_opt, :]
    ax3.plot(m_range, elbo_vs_m, 'b-', linewidth=2, label=f'ELBO vs m (s={sigma_exact:.2f})')
    ax3.axvline(mu_exact, color='red', linestyle='--', label='Optimal m')
    
    ax3.set_xlabel('Mean (m)', fontsize=10)
    ax3.set_ylabel('ELBO', fontsize=10)
    ax3.set_title('(c) ELBO Cross-section', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '07_elbo_surface.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 07_elbo_surface.png]")


# ============================================================================
# SECTION 3: ELBO Gradient and Optimization
# ============================================================================

def elbo_gradient_descent():
    """
    Optimize ELBO using gradient descent.
    
    For simple models, we can compute analytical gradients of ELBO
    with respect to variational parameters and use gradient-based optimization.
    
    Gradient of ELBO:
    ----------------
    ∂ELBO/∂m = ∂/∂m [E_q[log p(D,θ)] - E_q[log q(θ)]]
    ∂ELBO/∂s = ∂/∂s [E_q[log p(D,θ)] - E_q[log q(θ)]]
    
    For our Gaussian model, these have closed-form expressions.
    """
    
    print("\n" + "=" * 80)
    print("ELBO OPTIMIZATION: Gradient Descent")
    print("=" * 80)
    
    # Setup (same as before)
    np.random.seed(42)
    theta_true = 2.5
    sigma = 1.0
    n_data = 20
    mu_0 = 0.0
    sigma_0 = 2.0
    
    data = np.random.normal(theta_true, sigma, n_data)
    data_mean = np.mean(data)
    
    def compute_elbo_and_grad(m, s):
        """
        Compute ELBO and its gradients for q(θ) = N(m, s²).
        
        Gradients (derived analytically):
        --------------------------------
        ∂ELBO/∂m = [sum(x_i - m)]/σ² - (m - μ₀)/σ₀²
        ∂ELBO/∂s = -n*s/σ² - s/σ₀² - 1/s
        """
        
        # ELBO components
        term1 = -0.5 * n_data * np.log(2 * np.pi * sigma**2)
        term1 -= (1 / (2 * sigma**2)) * (np.sum((data - m)**2) + n_data * s**2)
        
        term2 = -0.5 * np.log(2 * np.pi * sigma_0**2)
        term2 -= ((m - mu_0)**2 + s**2) / (2 * sigma_0**2)
        
        term3 = -0.5 * np.log(2 * np.pi * np.e * s**2)
        
        elbo = term1 + term2 + term3
        
        # Gradients
        grad_m = np.sum(data - m) / sigma**2 - (m - mu_0) / sigma_0**2
        grad_s = -n_data * s / sigma**2 - s / sigma_0**2 - 1 / s
        
        return elbo, grad_m, grad_s
    
    # Initialize parameters
    m = 0.0
    s = 1.0
    learning_rate = 0.01
    n_iterations = 1000
    
    # Storage for visualization
    history_m = [m]
    history_s = [s]
    history_elbo = []
    
    print(f"\nInitial parameters: m={m:.3f}, s={s:.3f}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    
    # Gradient ascent loop (maximize ELBO)
    for iter in range(n_iterations):
        elbo, grad_m, grad_s = compute_elbo_and_grad(m, s)
        history_elbo.append(elbo)
        
        # Gradient ascent update (maximize ELBO)
        m += learning_rate * grad_m
        s += learning_rate * grad_s
        
        # Ensure s > 0
        s = max(s, 0.01)
        
        history_m.append(m)
        history_s.append(s)
        
        if iter % 100 == 0:
            print(f"Iter {iter:4d}: m={m:.4f}, s={s:.4f}, ELBO={elbo:.4f}")
    
    # Final results
    final_elbo, _, _ = compute_elbo_and_grad(m, s)
    print(f"\nFinal parameters: m={m:.4f}, s={s:.4f}, ELBO={final_elbo:.4f}")
    
    # Compare with exact
    precision_0 = 1 / sigma_0**2
    precision_data = n_data / sigma**2
    precision_n = precision_0 + precision_data
    sigma_n = 1 / np.sqrt(precision_n)
    mu_n = (precision_0 * mu_0 + precision_data * data_mean) / precision_n
    
    print(f"\nExact posterior: μₙ={mu_n:.4f}, σₙ={sigma_n:.4f}")
    print(f"Difference: Δm={abs(m - mu_n):.6f}, Δs={abs(s - sigma_n):.6f}")
    
    # Visualize optimization trajectory
    visualize_optimization_trajectory(history_m, history_s, history_elbo, 
                                     mu_n, sigma_n, data, mu_0, sigma_0, sigma)
    
    return m, s


def visualize_optimization_trajectory(history_m, history_s, history_elbo,
                                      mu_exact, sigma_exact, data, mu_0, sigma_0, sigma):
    """
    Visualize the gradient descent trajectory.
    """
    
    print("\n[Generating optimization trajectory visualization...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ELBO over iterations
    axes[0, 0].plot(history_elbo, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('ELBO', fontsize=11)
    axes[0, 0].set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter trajectory
    axes[0, 1].plot(history_m, history_s, 'b-', linewidth=1, alpha=0.5)
    axes[0, 1].plot(history_m[0], history_s[0], 'go', markersize=10, label='Start')
    axes[0, 1].plot(history_m[-1], history_s[-1], 'bo', markersize=10, label='End')
    axes[0, 1].plot(mu_exact, sigma_exact, 'r*', markersize=15, label='Exact')
    axes[0, 1].set_xlabel('Mean (m)', fontsize=11)
    axes[0, 1].set_ylabel('Std Dev (s)', fontsize=11)
    axes[0, 1].set_title('(b) Parameter Trajectory', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean convergence
    axes[1, 0].plot(history_m, 'b-', linewidth=2, label='m(t)')
    axes[1, 0].axhline(mu_exact, color='red', linestyle='--', linewidth=2, label='Exact μₙ')
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel('Mean', fontsize=11)
    axes[1, 0].set_title('(c) Mean Parameter Convergence', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Std dev convergence
    axes[1, 1].plot(history_s, 'b-', linewidth=2, label='s(t)')
    axes[1, 1].axhline(sigma_exact, color='red', linestyle='--', linewidth=2, label='Exact σₙ')
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel('Std Dev', fontsize=11)
    axes[1, 1].set_title('(d) Std Dev Parameter Convergence', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '08_optimization_trajectory.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[Figure saved: 08_optimization_trajectory.png]")


# ============================================================================
# SECTION 4: Summary
# ============================================================================

def print_summary():
    """
    Print module summary.
    """
    
    print("\n" + "=" * 80)
    print("MODULE SUMMARY: Evidence Lower Bound (ELBO)")
    print("=" * 80)
    
    summary = """
KEY CONCEPTS:
============

1. ELBO DEFINITION:
   ELBO(q) = E_q[log p(D,θ) - log q(θ)]
           = E_q[log p(D|θ)] + E_q[log p(θ)] - E_q[log q(θ)]

2. FUNDAMENTAL RELATIONSHIP:
   log p(D) = ELBO(q) + KL(q(θ) || p(θ|D))
   
   Since KL ≥ 0:
   log p(D) ≥ ELBO(q)  (hence "lower bound")

3. OPTIMIZATION EQUIVALENCE:
   max_q ELBO(q) ⟺ min_q KL(q(θ) || p(θ|D))

4. WHY ELBO WORKS:
   - Can be computed without knowing p(D)
   - Only requires: prior p(θ), likelihood p(D|θ), and variational q(θ)
   - Provides tractable objective for optimization

5. ELBO COMPONENTS:
   a) E_q[log p(D|θ)]: Expected log-likelihood (data fit)
   b) E_q[log p(θ)]: Expected log-prior (regularization)
   c) -E_q[log q(θ)]: Entropy of q (encourages uncertainty)
   
   Alternative view: Expected log-likelihood - KL(q||prior)

6. OPTIMIZATION METHODS:
   - Analytical (conjugate models)
   - Gradient descent / ascent
   - Coordinate ascent (next module)
   - Stochastic optimization (later modules)

PRACTICAL INSIGHTS:
==================

1. Gap Interpretation:
   The gap log p(D) - ELBO(q) measures approximation quality

2. Model Selection:
   ELBO can be used like BIC/AIC for comparing models

3. Convergence:
   Monitor ELBO to check if optimization has converged

4. Gradient Methods:
   For many models, ∇ELBO can be computed analytically or via
   automatic differentiation

NEXT STEPS:
==========
Module 03: Mean-Field Approximation and Coordinate Ascent VI (CAVI)
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
    print("VARIATIONAL INFERENCE - MODULE 2")
    print("Evidence Lower Bound (ELBO) Derivation")
    print("=" * 80)
    print("\nAuthor: Prof. Sungchul")
    print("Institution: Yonsei University")
    print("Email: sungchulyonsei@gmail.com")
    print("=" * 80)
    
    # Create directory for figures
    import os
    os.makedirs('/tmp/variational_inference/beginner/figures', exist_ok=True)
    
    # Run sections
    print("\n[1/4] Deriving the ELBO...")
    derive_elbo()
    
    print("\n[2/4] Computing ELBO for Gaussian model...")
    compute_elbo_gaussian()
    
    print("\n[3/4] Optimizing ELBO via gradient descent...")
    elbo_gradient_descent()
    
    print("\n[4/4] Summary...")
    print_summary()
    
    print("\n" + "=" * 80)
    print("MODULE COMPLETE!")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  • 06_elbo_decomposition.png")
    print("  • 07_elbo_surface.png")
    print("  • 08_optimization_trajectory.png")
    print("\nNext: Module 03 - Mean-Field Approximation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
