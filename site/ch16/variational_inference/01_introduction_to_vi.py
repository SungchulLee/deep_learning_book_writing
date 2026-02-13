"""
Variational Inference - Module 1: Introduction to Variational Inference
========================================================================

Learning Objectives:
-------------------
1. Understand the computational challenges in Bayesian inference
2. Learn why exact posterior computation is often intractable
3. Introduce the variational inference framework
4. Understand KL divergence and its role in VI
5. Visualize the concept of approximate distributions

Prerequisites:
-------------
- Basic probability theory
- Understanding of Bayesian inference
- Python and NumPy basics

Author: Prof. Sungchul Lee, Yonsei University
Email: sungchul@yonsei.ac.kr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
import seaborn as sns
import os

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# SECTION 1: The Challenge of Bayesian Inference
# ============================================================================

def demonstrate_intractable_posterior():
    """
    Demonstrate why computing exact posteriors can be intractable.
    
    Mathematical Background:
    -----------------------
    In Bayesian inference, we want to compute the posterior distribution:
    
        p(θ|D) = p(D|θ)p(θ) / p(D)
    
    where:
        - p(θ|D) is the posterior (what we want)
        - p(D|θ) is the likelihood (probability of data given parameters)
        - p(θ) is the prior (our initial beliefs)
        - p(D) is the marginal likelihood (evidence)
    
    The challenge: Computing p(D) requires integration over all possible θ:
    
        p(D) = ∫ p(D|θ)p(θ) dθ
    
    This integral is often:
    1. High-dimensional (many parameters)
    2. Non-conjugate (no closed-form solution)
    3. Computationally expensive to approximate numerically
    
    Example: Mixture of Gaussians
    -----------------------------
    Even simple models can have intractable posteriors.
    """
    
    print("=" * 80)
    print("DEMONSTRATION: The Intractability Challenge")
    print("=" * 80)
    
    # Example: Simple mixture model
    # True data generation process
    np.random.seed(42)
    
    # Generate data from a mixture of two Gaussians
    n_samples = 100
    true_weights = np.array([0.3, 0.7])
    true_means = np.array([-2.0, 3.0])
    true_stds = np.array([1.0, 1.5])
    
    # Generate mixture data
    components = np.random.choice([0, 1], size=n_samples, p=true_weights)
    data = np.zeros(n_samples)
    for i, comp in enumerate(components):
        data[i] = np.random.normal(true_means[comp], true_stds[comp])
    
    print("\nGenerated Data from Mixture Model:")
    print(f"  - Number of samples: {n_samples}")
    print(f"  - True mixture weights: {true_weights}")
    print(f"  - True means: {true_means}")
    print(f"  - True standard deviations: {true_stds}")
    
    # For this simple 2-component mixture, the posterior over parameters
    # (means, variances, weights) is high-dimensional and non-conjugate
    
    print("\nPosterior Complexity:")
    print(f"  - Parameters to infer: mixture weights (2), means (2), stds (2)")
    print(f"  - Total parameters: 6")
    print(f"  - Posterior p(θ|D) requires 6-dimensional integration")
    print(f"  - No closed-form solution available")
    
    # Visualize the data
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Data Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Observed Data Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Show true mixture components
    plt.subplot(1, 2, 2)
    x_range = np.linspace(-6, 8, 1000)
    component_1 = true_weights[0] * stats.norm.pdf(x_range, true_means[0], true_stds[0])
    component_2 = true_weights[1] * stats.norm.pdf(x_range, true_means[1], true_stds[1])
    mixture = component_1 + component_2
    
    plt.plot(x_range, component_1, 'r--', label='Component 1', linewidth=2)
    plt.plot(x_range, component_2, 'g--', label='Component 2', linewidth=2)
    plt.plot(x_range, mixture, 'b-', label='Mixture', linewidth=2)
    plt.hist(data, bins=30, density=True, alpha=0.3, color='gray')
    plt.xlabel('Data Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('True Mixture Components', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '01_intractable_posterior.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[Figure saved: 01_intractable_posterior.png]")
    
    return data


# ============================================================================
# SECTION 2: Introduction to Variational Inference
# ============================================================================

def introduction_to_vi():
    """
    Introduce the core idea of Variational Inference.
    
    The Main Idea:
    -------------
    Instead of computing the exact posterior p(θ|D), we:
    
    1. Choose a family of simpler distributions Q = {q(θ)}
    2. Find the member q*(θ) ∈ Q that is closest to p(θ|D)
    3. Use q*(θ) as an approximation to the true posterior
    
    "Closest" is measured using KL divergence:
    
        q*(θ) = argmin_{q∈Q} KL(q(θ) || p(θ|D))
    
    Why This Works:
    --------------
    - We convert an integration problem into an optimization problem
    - Optimization is often easier than integration
    - We can use gradient-based methods
    - We can scale to large datasets using stochastic optimization
    
    Trade-offs:
    ----------
    - Approximation: q(θ) ≠ p(θ|D) in general
    - Bias: VI can underestimate posterior uncertainty
    - Speed: Much faster than MCMC for large problems
    - Convergence: VI provides guarantees on convergence
    """
    
    print("\n" + "=" * 80)
    print("VARIATIONAL INFERENCE: Core Concept")
    print("=" * 80)
    
    print("\nThe Challenge:")
    print("  Computing p(θ|D) exactly is often impossible")
    
    print("\nThe Solution:")
    print("  Approximate p(θ|D) with a simpler distribution q(θ)")
    
    print("\nThe Method:")
    print("  1. Choose a family of distributions Q")
    print("  2. Find q*(θ) ∈ Q closest to p(θ|D)")
    print("  3. Measure 'closeness' using KL divergence")
    
    print("\nMathematical Formulation:")
    print("  q*(θ) = argmin_{q∈Q} KL(q(θ) || p(θ|D))")
    print("  where KL(q||p) = ∫ q(θ) log[q(θ)/p(θ|D)] dθ")
    
    # Visualize the concept
    visualize_vi_concept()


def visualize_vi_concept():
    """
    Create visualizations explaining the VI concept.
    
    We'll show:
    1. True posterior vs variational approximation
    2. Different approximating families
    3. The effect of choosing Q
    """
    
    # Define a bimodal "true" posterior (not normalized, but that's OK for visualization)
    x = np.linspace(-5, 5, 1000)
    
    # True posterior: mixture of two Gaussians
    true_posterior = (0.4 * stats.norm.pdf(x, -2, 0.5) + 
                      0.6 * stats.norm.pdf(x, 2, 0.7))
    # true_posterior = true_posterior / np.trapz(true_posterior, x)  # Normalize
    true_posterior = true_posterior / np.trapezoid(true_posterior, x)  # Normalize
    
    # Approximating distributions from different families
    # Family 1: Single Gaussian (mean-field)
    mean_q1, std_q1 = 1.0, 1.5
    q1 = stats.norm.pdf(x, mean_q1, std_q1)
    
    # Family 2: Narrower Gaussian (underestimates uncertainty)
    mean_q2, std_q2 = 1.0, 0.8
    q2 = stats.norm.pdf(x, mean_q2, std_q2)
    
    # Family 3: Wider Gaussian (overestimates uncertainty)
    mean_q3, std_q3 = 1.0, 2.5
    q3 = stats.norm.pdf(x, mean_q3, std_q3)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: True posterior
    axes[0, 0].fill_between(x, true_posterior, alpha=0.3, color='blue', label='True Posterior')
    axes[0, 0].plot(x, true_posterior, 'b-', linewidth=2, label='p(θ|D)')
    axes[0, 0].set_xlabel('Parameter θ', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('(a) True Posterior Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, 'Complex\nBimodal', transform=axes[0, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Good approximation
    axes[0, 1].fill_between(x, true_posterior, alpha=0.3, color='blue', label='True p(θ|D)')
    axes[0, 1].plot(x, true_posterior, 'b-', linewidth=2)
    axes[0, 1].fill_between(x, q1, alpha=0.3, color='red', label='Approximation q(θ)')
    axes[0, 1].plot(x, q1, 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Parameter θ', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('(b) Variational Approximation', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, 'Best fit\nGaussian', transform=axes[0, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 3: Underestimated uncertainty
    axes[1, 0].fill_between(x, true_posterior, alpha=0.3, color='blue')
    axes[1, 0].plot(x, true_posterior, 'b-', linewidth=2, label='True p(θ|D)')
    axes[1, 0].fill_between(x, q2, alpha=0.3, color='orange')
    axes[1, 0].plot(x, q2, 'orange', linestyle='--', linewidth=2, label='q(θ) - Too Narrow')
    axes[1, 0].set_xlabel('Parameter θ', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('(c) Underestimated Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, 'Too\nConfident', transform=axes[1, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Plot 4: Overestimated uncertainty
    axes[1, 1].fill_between(x, true_posterior, alpha=0.3, color='blue')
    axes[1, 1].plot(x, true_posterior, 'b-', linewidth=2, label='True p(θ|D)')
    axes[1, 1].fill_between(x, q3, alpha=0.3, color='green')
    axes[1, 1].plot(x, q3, 'g--', linewidth=2, label='q(θ) - Too Wide')
    axes[1, 1].set_xlabel('Parameter θ', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('(d) Overestimated Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, 'Too\nUncertain', transform=axes[1, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '02_vi_concept.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[Figure saved: 02_vi_concept.png]")


# ============================================================================
# SECTION 3: KL Divergence - Measuring Distribution Similarity
# ============================================================================

def kl_divergence_tutorial():
    """
    Deep dive into KL divergence and its properties.
    
    Definition:
    ----------
    The Kullback-Leibler (KL) divergence from distribution Q to P is:
    
        KL(P || Q) = ∫ p(x) log[p(x)/q(x)] dx
                   = E_p[log p(x) - log q(x)]
    
    For discrete distributions:
        KL(P || Q) = Σ p(x) log[p(x)/q(x)]
    
    Properties:
    ----------
    1. Non-negative: KL(P || Q) ≥ 0
    2. Zero if and only if P = Q almost everywhere
    3. Asymmetric: KL(P || Q) ≠ KL(Q || P) in general
    4. Not a distance metric (doesn't satisfy triangle inequality)
    
    Interpretation:
    --------------
    - Information lost when Q is used to approximate P
    - Expected log-likelihood ratio under P
    - Relative entropy
    
    In Variational Inference:
    ------------------------
    We minimize KL(q || p) where:
    - q is our variational approximation (what we control)
    - p is the true posterior (what we want to approximate)
    
    This is called "forward KL" or "inclusive KL"
    """
    
    print("\n" + "=" * 80)
    print("KL DIVERGENCE: The Heart of Variational Inference")
    print("=" * 80)
    
    # Example: KL divergence between two Gaussians
    print("\nExample: KL Divergence Between Gaussians")
    print("-" * 50)
    
    # Define two Gaussian distributions
    mu_p, sigma_p = 0.0, 1.0  # Target distribution
    mu_q, sigma_q = 0.5, 1.5  # Approximating distribution
    
    # Analytical KL divergence for Gaussians: KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
    # KL = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 1/2
    kl_analytical = (np.log(sigma_p / sigma_q) + 
                     (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5)
    
    print(f"\nTarget Distribution P: N(μ={mu_p}, σ²={sigma_p**2})")
    print(f"Approximation Q: N(μ={mu_q}, σ²={sigma_q**2})")
    print(f"\nKL(Q || P) = {kl_analytical:.4f}")
    
    # Numerical verification using Monte Carlo
    n_samples = 100000
    samples = np.random.normal(mu_q, sigma_q, n_samples)
    log_q = stats.norm.logpdf(samples, mu_q, sigma_q)
    log_p = stats.norm.logpdf(samples, mu_p, sigma_p)
    kl_numerical = np.mean(log_q - log_p)
    
    print(f"KL(Q || P) [Monte Carlo] = {kl_numerical:.4f}")
    print(f"Difference: {abs(kl_analytical - kl_numerical):.6f}")
    
    # Visualize KL divergence
    visualize_kl_divergence()
    
    # Demonstrate asymmetry
    demonstrate_kl_asymmetry()


def visualize_kl_divergence():
    """
    Visualize KL divergence and its properties.
    """
    
    x = np.linspace(-5, 5, 1000)
    
    # Target distribution (what we want to approximate)
    mu_p, sigma_p = 0.0, 1.0
    p = stats.norm.pdf(x, mu_p, sigma_p)
    
    # Various approximations
    mus_q = [-1.0, -0.5, 0.0, 0.5, 1.0]
    sigma_q = 1.2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    kl_values = []
    for i, mu_q in enumerate(mus_q):
        q = stats.norm.pdf(x, mu_q, sigma_q)
        
        # Calculate KL divergence
        kl = (np.log(sigma_p / sigma_q) + 
              (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5)
        kl_values.append(kl)
        
        # Plot
        axes[i].fill_between(x, p, alpha=0.3, color='blue', label=f'Target P')
        axes[i].plot(x, p, 'b-', linewidth=2)
        axes[i].fill_between(x, q, alpha=0.3, color='red', label=f'Approx Q (μ={mu_q})')
        axes[i].plot(x, q, 'r--', linewidth=2)
        axes[i].set_xlabel('x', fontsize=10)
        axes[i].set_ylabel('Density', fontsize=10)
        axes[i].set_title(f'KL(Q||P) = {kl:.4f}', fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    # Plot KL as a function of mean shift
    axes[5].plot(mus_q, kl_values, 'o-', linewidth=2, markersize=8, color='purple')
    axes[5].set_xlabel('Mean of Q (μ_q)', fontsize=10)
    axes[5].set_ylabel('KL(Q || P)', fontsize=10)
    axes[5].set_title('KL Divergence vs Mean Shift', fontsize=11, fontweight='bold')
    axes[5].grid(True, alpha=0.3)
    axes[5].axvline(mu_p, color='blue', linestyle='--', alpha=0.5, label='Target mean')
    axes[5].legend(fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '03_kl_divergence.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[Figure saved: 03_kl_divergence.png]")


def demonstrate_kl_asymmetry():
    """
    Demonstrate the asymmetry of KL divergence: KL(P||Q) ≠ KL(Q||P).
    
    This is crucial in VI because we use KL(q||p), not KL(p||q).
    The choice matters!
    
    KL(q||p) - "Forward KL" or "Exclusive KL":
        - q tries to cover the entire support of p
        - Leads to "mean-seeking" behavior
        - May overestimate uncertainty
        - Used in standard VI
    
    KL(p||q) - "Reverse KL" or "Inclusive KL":
        - q tries to be non-zero wherever p is non-zero
        - Leads to "mode-seeking" behavior
        - May underestimate uncertainty
        - Used in some VI variants (e.g., expectation propagation)
    """
    
    print("\n" + "=" * 80)
    print("KL DIVERGENCE ASYMMETRY: Forward vs Reverse KL")
    print("=" * 80)
    
    x = np.linspace(-6, 6, 1000)
    
    # True distribution: mixture of two well-separated Gaussians
    p = 0.5 * stats.norm.pdf(x, -2, 0.5) + 0.5 * stats.norm.pdf(x, 2, 0.5)
    # p = p / np.trapz(p, x)  # Normalize
    p = p / np.trapezoid(p, x)  # Normalize
    
    # Approximation: single Gaussian
    mu_q, sigma_q = 0.0, 2.0
    q = stats.norm.pdf(x, mu_q, sigma_q)
    
    # Calculate both KL divergences numerically
    # KL(q||p) = ∫ q(x) log[q(x)/p(x)] dx
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    #kl_q_p = np.trapz(q * np.log((q + eps) / (p + eps)), x)
    kl_q_p = np.trapezoid(q * np.log((q + eps) / (p + eps)), x)
    
    # KL(p||q) = ∫ p(x) log[p(x)/q(x)] dx
    #kl_p_q = np.trapz(p * np.log((p + eps) / (q + eps)), x)
    kl_p_q = np.trapezoid(p * np.log((p + eps) / (q + eps)), x)
    
    print(f"\nTrue distribution: Mixture of two Gaussians")
    print(f"Approximation: Single Gaussian N(μ={mu_q}, σ={sigma_q})")
    print(f"\nKL(q || p) [Forward/Exclusive] = {kl_q_p:.4f}")
    print(f"KL(p || q) [Reverse/Inclusive] = {kl_p_q:.4f}")
    print(f"Ratio: KL(p||q) / KL(q||p) = {kl_p_q/kl_q_p:.2f}")
    
    # Visualize the asymmetry
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Forward KL (what VI uses)
    axes[0].fill_between(x, p, alpha=0.3, color='blue', label='True p(x)')
    axes[0].plot(x, p, 'b-', linewidth=2)
    axes[0].fill_between(x, q, alpha=0.3, color='red', label='Approx q(x)')
    axes[0].plot(x, q, 'r--', linewidth=2)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title(f'(a) Forward KL: KL(q||p) = {kl_q_p:.4f}\n(Standard VI)', 
                     fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, 'q covers both\nmodes of p\n(mean-seeking)', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Reverse KL (for comparison)
    # For reverse KL, optimal q would concentrate on one mode
    mu_q2, sigma_q2 = 2.0, 0.6
    q2 = stats.norm.pdf(x, mu_q2, sigma_q2)
    #kl_p_q2 = np.trapz(p * np.log((p + eps) / (q2 + eps)), x)
    kl_p_q2 = np.trapezoid(p * np.log((p + eps) / (q2 + eps)), x)
    
    axes[1].fill_between(x, p, alpha=0.3, color='blue', label='True p(x)')
    axes[1].plot(x, p, 'b-', linewidth=2)
    axes[1].fill_between(x, q2, alpha=0.3, color='green', label='Approx q(x)')
    axes[1].plot(x, q2, 'g--', linewidth=2)
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title(f'(b) Reverse KL: KL(p||q) = {kl_p_q2:.4f}\n(Mode-seeking)', 
                     fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.05, 0.95, 'q focuses on\none mode\n(mode-seeking)', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 3: Comparison of behaviors
    axes[2].plot(x, p, 'b-', linewidth=2.5, label='True p(x)', alpha=0.8)
    axes[2].plot(x, q, 'r--', linewidth=2, label='Forward KL q(x)', alpha=0.8)
    axes[2].plot(x, q2, 'g--', linewidth=2, label='Reverse KL q(x)', alpha=0.8)
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('Density', fontsize=11)
    axes[2].set_title('(c) Comparison of KL Directions', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '04_kl_asymmetry.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[Figure saved: 04_kl_asymmetry.png]")
    
    print("\nKey Insight:")
    print("  Forward KL (q||p): q tries to cover all modes → mean-seeking")
    print("  Reverse KL (p||q): q focuses on one mode → mode-seeking")
    print("  Standard VI uses forward KL")


# ============================================================================
# SECTION 4: Simple VI Example - Gaussian with Unknown Mean
# ============================================================================

def simple_vi_example():
    """
    Work through a complete VI example with a simple model.
    
    Model:
    -----
    - Data: x_1, ..., x_n ~ N(θ, σ²) with known σ²
    - Prior: θ ~ N(μ_0, σ_0²)
    - Posterior: θ | x ~ N(μ_n, σ_n²) [this is exact/conjugate]
    
    VI Approximation:
    ----------------
    - Variational family: q(θ) = N(m, s²)
    - Objective: minimize KL(q(θ) || p(θ|x))
    - Optimal: q*(θ) = N(μ_n, σ_n²) [same as exact posterior!]
    
    This example shows that VI is exact when the variational family
    contains the true posterior.
    """
    
    print("\n" + "=" * 80)
    print("SIMPLE VI EXAMPLE: Gaussian Mean Estimation")
    print("=" * 80)
    
    # Set parameters
    np.random.seed(42)
    theta_true = 2.5  # True mean
    sigma = 1.0       # Known standard deviation
    n_data = 20       # Number of observations
    
    # Prior parameters
    mu_0 = 0.0        # Prior mean
    sigma_0 = 2.0     # Prior standard deviation
    
    # Generate data
    data = np.random.normal(theta_true, sigma, n_data)
    
    print(f"\nModel Setup:")
    print(f"  True parameter: θ = {theta_true}")
    print(f"  Data likelihood: x_i ~ N(θ, σ²={sigma**2})")
    print(f"  Number of observations: n = {n_data}")
    print(f"  Prior: θ ~ N(μ₀={mu_0}, σ₀²={sigma_0**2})")
    
    # Exact posterior (conjugate update)
    # Posterior precision = prior precision + data precision × n
    precision_0 = 1 / sigma_0**2
    precision_data = n_data / sigma**2
    precision_n = precision_0 + precision_data
    sigma_n = 1 / np.sqrt(precision_n)
    
    # Posterior mean = weighted average of prior mean and data mean
    mu_n = (precision_0 * mu_0 + precision_data * np.mean(data)) / precision_n
    
    print(f"\nExact Posterior (Conjugate Update):")
    print(f"  p(θ|x) = N(μₙ={mu_n:.3f}, σₙ²={sigma_n**2:.3f})")
    
    # Variational inference
    # For Gaussian family, the optimal q is also the exact posterior
    m_optimal = mu_n
    s_optimal = sigma_n
    
    print(f"\nVariational Approximation:")
    print(f"  q*(θ) = N(m={m_optimal:.3f}, s²={s_optimal**2:.3f})")
    
    # Verify they match
    print(f"\nVerification:")
    print(f"  |μₙ - m| = {abs(mu_n - m_optimal):.10f}")
    print(f"  |σₙ - s| = {abs(sigma_n - s_optimal):.10f}")
    print(f"  → VI is exact for this model!")
    
    # Visualize
    theta_range = np.linspace(-2, 6, 1000)
    
    # Prior
    prior = stats.norm.pdf(theta_range, mu_0, sigma_0)
    
    # Likelihood (for visualization, evaluate at true mean)
    likelihood = stats.norm.pdf(theta_range, np.mean(data), sigma / np.sqrt(n_data))
    
    # Posterior (exact)
    posterior = stats.norm.pdf(theta_range, mu_n, sigma_n)
    
    # Variational approximation
    q_approx = stats.norm.pdf(theta_range, m_optimal, s_optimal)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Data
    axes[0, 0].hist(data, bins=15, density=True, alpha=0.6, color='gray', edgecolor='black')
    axes[0, 0].axvline(theta_true, color='red', linestyle='--', linewidth=2, label='True θ')
    axes[0, 0].axvline(np.mean(data), color='blue', linestyle='--', linewidth=2, label='Sample mean')
    axes[0, 0].set_xlabel('Data Value', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('(a) Observed Data', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prior and Likelihood
    axes[0, 1].plot(theta_range, prior, 'g-', linewidth=2, label='Prior p(θ)')
    axes[0, 1].fill_between(theta_range, prior, alpha=0.2, color='green')
    axes[0, 1].plot(theta_range, likelihood, 'orange', linestyle='--', linewidth=2, 
                    label='Likelihood p(x|θ)')
    axes[0, 1].axvline(theta_true, color='red', linestyle=':', linewidth=1.5, label='True θ')
    axes[0, 1].set_xlabel('Parameter θ', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('(b) Prior and Likelihood', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Exact Posterior
    axes[1, 0].plot(theta_range, posterior, 'b-', linewidth=2.5, label='Exact Posterior p(θ|x)')
    axes[1, 0].fill_between(theta_range, posterior, alpha=0.3, color='blue')
    axes[1, 0].axvline(theta_true, color='red', linestyle='--', linewidth=2, label='True θ')
    axes[1, 0].axvline(mu_n, color='blue', linestyle=':', linewidth=2, label='Posterior mean')
    axes[1, 0].set_xlabel('Parameter θ', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('(c) Exact Posterior (Conjugate)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Comparison
    axes[1, 1].plot(theta_range, posterior, 'b-', linewidth=2.5, label='Exact p(θ|x)', alpha=0.7)
    axes[1, 1].plot(theta_range, q_approx, 'r--', linewidth=2.5, label='VI approx q(θ)', alpha=0.7)
    axes[1, 1].fill_between(theta_range, posterior, alpha=0.2, color='blue')
    axes[1, 1].axvline(theta_true, color='red', linestyle=':', linewidth=2, label='True θ')
    axes[1, 1].set_xlabel('Parameter θ', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('(d) VI Approximation vs Exact', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, 'Exact match!\nq* = p(θ|x)', 
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '05_simple_vi_example.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[Figure saved: 05_simple_vi_example.png]")


# ============================================================================
# SECTION 5: Summary and Key Takeaways
# ============================================================================

def print_summary():
    """
    Print comprehensive summary of key concepts.
    """
    
    print("\n" + "=" * 80)
    print("MODULE SUMMARY: Introduction to Variational Inference")
    print("=" * 80)
    
    summary = """
KEY CONCEPTS:
------------

1. THE PROBLEM:
   • Bayesian inference requires computing p(θ|D) = p(D|θ)p(θ) / p(D)
   • The marginal likelihood p(D) = ∫ p(D|θ)p(θ) dθ is often intractable
   • High-dimensional integrals, non-conjugate models make exact inference impossible

2. THE SOLUTION - VARIATIONAL INFERENCE:
   • Approximate p(θ|D) with a simpler distribution q(θ) from family Q
   • Convert integration problem → optimization problem
   • Find q*(θ) = argmin_{q∈Q} KL(q(θ) || p(θ|D))

3. KL DIVERGENCE:
   • Measures "distance" between distributions
   • KL(q || p) = ∫ q(θ) log[q(θ)/p(θ)] dθ
   • Non-negative, zero iff q = p
   • ASYMMETRIC: KL(q||p) ≠ KL(p||q)

4. FORWARD vs REVERSE KL:
   • Forward KL (VI uses this): KL(q || p)
     - Mean-seeking behavior
     - q covers all modes of p
     - May overestimate uncertainty
   
   • Reverse KL (alternative): KL(p || q)
     - Mode-seeking behavior
     - q focuses on one mode
     - May underestimate uncertainty

5. ADVANTAGES OF VI:
   • Faster than MCMC for large-scale problems
   • Provides lower bound on log p(D) (useful for model selection)
   • Deterministic (no MCMC randomness)
   • Scalable with stochastic optimization
   • Convergence guarantees

6. LIMITATIONS OF VI:
   • Approximation: q(θ) ≠ p(θ|D) in general
   • Limited by choice of variational family Q
   • May underestimate uncertainty (especially with mean-field)
   • Biased estimates (unlike MCMC which is asymptotically exact)

MATHEMATICAL FRAMEWORK:
----------------------

Posterior (what we want):
    p(θ|D) = p(D|θ)p(θ) / p(D)

Marginal likelihood (intractable):
    p(D) = ∫ p(D|θ)p(θ) dθ

Variational objective:
    q*(θ) = argmin_{q∈Q} KL(q(θ) || p(θ|D))

KL divergence:
    KL(q||p) = ∫ q(θ) log[q(θ)/p(θ|D)] dθ
             = E_q[log q(θ)] - E_q[log p(θ|D)]

NEXT STEPS:
----------
• Module 02: ELBO Derivation - Learn how to optimize the VI objective
• Module 03: Mean-Field Approximation - Factorized variational families
• Module 04: Practical VI Algorithms - CAVI, stochastic VI, and more

FURTHER READING:
---------------
• Blei et al. (2017) - "Variational Inference: A Review for Statisticians"
• Bishop (2006) - Pattern Recognition and Machine Learning, Chapter 10
• Murphy (2022) - Probabilistic Machine Learning: Advanced Topics
"""
    
    print(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all demonstrations.
    """
    
    print("\n" + "=" * 80)
    print("VARIATIONAL INFERENCE - MODULE 1")
    print("Introduction to Variational Inference")
    print("=" * 80)
    print("\nAuthor: Prof. Sungchul")
    print("Institution: Yonsei University")
    print("Email: sungchulyonsei@gmail.com")
    print("=" * 80)
    
    # Run all sections
    print("\n[1/5] Demonstrating computational challenges...")
    data = demonstrate_intractable_posterior()
    
    print("\n[2/5] Introducing variational inference...")
    introduction_to_vi()
    
    print("\n[3/5] Understanding KL divergence...")
    kl_divergence_tutorial()
    
    print("\n[4/5] Working through simple example...")
    simple_vi_example()
    
    print("\n[5/5] Summary...")
    print_summary()
    
    print("\n" + "=" * 80)
    print("MODULE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • 01_intractable_posterior.png")
    print("  • 02_vi_concept.png")
    print("  • 03_kl_divergence.png")
    print("  • 04_kl_asymmetry.png")
    print("  • 05_simple_vi_example.png")
    print("\nNext: Continue to Module 02 - ELBO Derivation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
