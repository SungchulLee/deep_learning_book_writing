"""
03_metropolis_hastings.py

METROPOLIS-HASTINGS ALGORITHM: COMPREHENSIVE TUTORIAL
====================================================

Learning Objectives:
-------------------
1. Understand the general Metropolis-Hastings algorithm
2. Learn about asymmetric proposal distributions
3. Explore independence samplers and adaptive methods
4. See connections to modern MCMC techniques

Mathematical Foundation:
-----------------------
Goal:
    Sample from a target distribution p(x) when we can evaluate p(x) only up
    to a constant (unnormalized density p̃(x) ∝ p(x)).

Metropolis-Hastings Algorithm (General Form):
---------------------------------------------
1. Choose an initial point x₀
2. For t = 1, 2, ...
   a. Propose a candidate x' ~ q(x' | x_t)
      where q(x' | x) is an ASYMMETRIC proposal distribution
      (i.e., q(x' | x) ≠ q(x | x') in general)
   
   b. Compute acceptance probability:
      α = min(1, [p(x') · q(x_t | x')] / [p(x_t) · q(x' | x_t)])
   
   c. With probability α, accept and set x_{t+1} = x'
      Otherwise, reject and set x_{t+1} = x_t

Why the Full Ratio Matters:
---------------------------
Unlike Metropolis (symmetric case), we CANNOT cancel the proposal terms!

The ratio q(x_t | x') / q(x' | x_t) corrects for asymmetry:
- If proposing x' from x is more likely than proposing x from x',
  we're biased toward x', so we need to down-weight the acceptance
- This correction ensures detailed balance still holds

Types of Asymmetric Proposals:
------------------------------
1. Independence Sampler: q(x' | x) = q(x')
   - Doesn't depend on current state
   - Good when q ≈ p

2. Langevin Dynamics: q(x' | x) includes gradient information
   - Drift toward high-probability regions
   - Foundation for modern methods (MALA, HMC)

3. Adaptive Proposals: q depends on past samples
   - Learn optimal proposal during sampling
   - Careful: must preserve ergodicity

4. Mixture Proposals: q = Σᵢ wᵢ qᵢ
   - Combine different move types
   - Robust to different distribution shapes

Connection to Diffusion Models:
-------------------------------
- MH with learned proposals → Neural samplers
- Langevin dynamics → Score-based models
- Reverse diffusion = iterative MH with learned transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, cauchy, t as student_t
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: INDEPENDENCE SAMPLER
# =============================================================================

def independence_sampler():
    """
    Example 1: Independence Sampler
    ===============================
    
    Proposal: q(x' | x) = q(x') - independent of current state!
    
    Acceptance ratio:
    α = min(1, [p(x') · q(x)] / [p(x) · q(x')])
    
    Strategy:
    - Choose q to approximate p
    - Heavy tails in q help exploration
    - Works well when q is close to p
    
    Example: Target is N(0,1), proposal is t-distribution
    """
    print("=" * 70)
    print("EXAMPLE 1: Independence Sampler")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target: Standard normal
    def target_pdf(x):
        return norm.pdf(x, 0, 1)
    
    # Proposal: t-distribution (heavier tails)
    df = 3  # degrees of freedom
    
    def proposal_pdf(x):
        return student_t.pdf(x, df)
    
    def proposal_sample():
        return student_t.rvs(df)
    
    # Independence Metropolis-Hastings
    def independence_mh(n_samples, x_init=0.0):
        """
        Independence sampler with MH acceptance
        
        Note: q(x'|x) = q(x'), so the acceptance ratio is:
        α = min(1, [p(x') q(x)] / [p(x) q(x')])
        """
        samples = np.zeros(n_samples)
        samples[0] = x_init
        n_accepted = 0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            
            # Propose from independent distribution
            x_proposal = proposal_sample()
            
            # MH acceptance ratio
            p_prop = target_pdf(x_proposal)
            p_curr = target_pdf(x_current)
            q_prop = proposal_pdf(x_proposal)
            q_curr = proposal_pdf(x_current)
            
            # Note the importance ratio structure
            alpha = min(1.0, (p_prop * q_curr) / (p_curr * q_prop))
            
            if np.random.rand() < alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Compare with random walk Metropolis
    def random_walk_metropolis(n_samples, proposal_std, x_init=0.0):
        samples = np.zeros(n_samples)
        samples[0] = x_init
        n_accepted = 0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            x_proposal = x_current + np.random.randn() * proposal_std
            
            alpha = min(1.0, target_pdf(x_proposal) / target_pdf(x_current))
            
            if np.random.rand() < alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Run both samplers
    n_samples = 5000
    indep_samples, indep_acc = independence_mh(n_samples)
    rw_samples, rw_acc = random_walk_metropolis(n_samples, 1.0)
    
    print(f"\nSampling from N(0,1) with {n_samples} iterations")
    print(f"\nIndependence Sampler (t-distribution proposal):")
    print(f"  Acceptance rate: {indep_acc:.2%}")
    print(f"  Sample mean: {indep_samples[1000:].mean():.4f}")
    print(f"  Sample std:  {indep_samples[1000:].std():.4f}")
    
    print(f"\nRandom Walk Metropolis (Gaussian proposal, σ=1.0):")
    print(f"  Acceptance rate: {rw_acc:.2%}")
    print(f"  Sample mean: {rw_samples[1000:].mean():.4f}")
    print(f"  Sample std:  {rw_samples[1000:].std():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Trace plots
    axes[0, 0].plot(indep_samples[:500], linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel(r'$x$')
    axes[0, 0].set_title(f'Independence Sampler\nAccept: {indep_acc:.1%}')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(rw_samples[:500], linewidth=0.5, alpha=0.7, color='orange')
    axes[0, 1].axhline(0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel(r'$x$')
    axes[0, 1].set_title(f'Random Walk\nAccept: {rw_acc:.1%}')
    axes[0, 1].grid(alpha=0.3)
    
    # Autocorrelation comparison
    def autocorr(x, lag):
        x = x - x.mean()
        return np.correlate(x, x, mode='full')[len(x)-1+lag] / np.correlate(x, x, mode='full')[len(x)-1]
    
    lags = range(0, 50)
    indep_acf = [autocorr(indep_samples[1000:], lag) for lag in lags]
    rw_acf = [autocorr(rw_samples[1000:], lag) for lag in lags]
    
    axes[0, 2].plot(lags, indep_acf, label='Independence', linewidth=2)
    axes[0, 2].plot(lags, rw_acf, label='Random Walk', linewidth=2)
    axes[0, 2].set_xlabel('Lag')
    axes[0, 2].set_ylabel('Autocorrelation')
    axes[0, 2].set_title('Autocorrelation Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Histograms
    x_grid = np.linspace(-4, 4, 200)
    
    axes[1, 0].hist(indep_samples[1000:], bins=50, density=True, alpha=0.7, 
                   edgecolor='black', label='Samples')
    axes[1, 0].plot(x_grid, target_pdf(x_grid), 'r-', linewidth=2, label='Target p(x)')
    axes[1, 0].plot(x_grid, proposal_pdf(x_grid), 'g--', linewidth=2, label='Proposal q(x)')
    axes[1, 0].set_xlabel(r'$x$')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Independence Sampler')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(rw_samples[1000:], bins=50, density=True, alpha=0.7, 
                   edgecolor='black', color='orange', label='Samples')
    axes[1, 1].plot(x_grid, target_pdf(x_grid), 'r-', linewidth=2, label='Target p(x)')
    axes[1, 1].set_xlabel(r'$x$')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Random Walk Metropolis')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Q-Q plot
    from scipy.stats import probplot
    probplot(indep_samples[1000:], dist='norm', plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot (Independence Sampler)')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'mh_independence_sampler.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: mh_independence_sampler.png")
    print("\nKey Insights:")
    print("  • Independence sampler can have lower autocorrelation")
    print("  • Heavy-tailed proposal (t-dist) helps exploration")
    print("  • Choice of q crucial for performance")


# =============================================================================
# PART 2: LANGEVIN DYNAMICS (MALA)
# =============================================================================

def langevin_mala():
    """
    Example 2: Metropolis-Adjusted Langevin Algorithm (MALA)
    ========================================================
    
    Idea: Use gradient information to drift toward high probability
    
    Proposal:
    x' = x + (ε²/2)∇log p(x) + ε·N(0, I)
    
    This is asymmetric! The reverse proposal is:
    q(x|x') ∝ exp(-||x - x' - (ε²/2)∇log p(x')||² / (2ε²))
    
    MH correction ensures detailed balance despite asymmetry.
    
    Advantages:
    - Uses gradient to guide exploration
    - More efficient than random walk
    - Foundation for Hamiltonian Monte Carlo
    
    Target: 2D Gaussian with correlation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Metropolis-Adjusted Langevin Algorithm")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target: 2D Gaussian with correlation
    mu = np.array([1.0, 2.0])
    rho = 0.7
    sigma = np.array([1.0, 1.5])
    cov = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2]
    ])
    cov_inv = np.linalg.inv(cov)
    
    def target_log_pdf(x):
        """Log pdf of target"""
        diff = x - mu
        return -0.5 * diff @ cov_inv @ diff
    
    def grad_log_pdf(x):
        """Gradient of log pdf"""
        return -cov_inv @ (x - mu)
    
    # MALA sampler
    def mala_sample(n_samples, epsilon, x_init=None):
        """
        MALA with gradient-based proposals
        
        Proposal: x' = x + (ε²/2)∇log p(x) + ε·Z, Z~N(0,I)
        """
        if x_init is None:
            x_init = np.zeros(2)
        
        samples = np.zeros((n_samples, 2))
        samples[0] = x_init
        n_accepted = 0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            
            # Compute gradient at current point
            grad_current = grad_log_pdf(x_current)
            
            # Langevin proposal: drift + diffusion
            drift = (epsilon**2 / 2) * grad_current
            diffusion = epsilon * np.random.randn(2)
            x_proposal = x_current + drift + diffusion
            
            # For MH, need reverse proposal probability
            grad_proposal = grad_log_pdf(x_proposal)
            
            # Forward proposal: q(x'|x)
            mean_forward = x_current + (epsilon**2 / 2) * grad_current
            diff_forward = x_proposal - mean_forward
            log_q_forward = -0.5 * (diff_forward @ diff_forward) / (epsilon**2)
            
            # Reverse proposal: q(x|x')
            mean_reverse = x_proposal + (epsilon**2 / 2) * grad_proposal
            diff_reverse = x_current - mean_reverse
            log_q_reverse = -0.5 * (diff_reverse @ diff_reverse) / (epsilon**2)
            
            # MH acceptance ratio
            log_alpha = (target_log_pdf(x_proposal) - target_log_pdf(x_current) +
                        log_q_reverse - log_q_forward)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Random walk for comparison
    def random_walk_2d(n_samples, step_size):
        samples = np.zeros((n_samples, 2))
        samples[0] = np.zeros(2)
        n_accepted = 0
        
        rv = multivariate_normal(mu, cov)
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            x_proposal = x_current + np.random.randn(2) * step_size
            
            alpha = min(1.0, np.exp(target_log_pdf(x_proposal) - target_log_pdf(x_current)))
            
            if np.random.rand() < alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Run both samplers
    n_samples = 5000
    epsilon = 0.3
    
    mala_samples, mala_acc = mala_sample(n_samples, epsilon)
    rw_samples, rw_acc = random_walk_2d(n_samples, 0.5)
    
    print(f"\nTarget: 2D Gaussian with correlation ρ={rho}")
    print(f"\nMALA (ε={epsilon}):")
    print(f"  Acceptance rate: {mala_acc:.2%}")
    print(f"  Sample mean: [{mala_samples[1000:].mean(axis=0)[0]:.3f}, {mala_samples[1000:].mean(axis=0)[1]:.3f}]")
    
    print(f"\nRandom Walk (σ=0.5):")
    print(f"  Acceptance rate: {rw_acc:.2%}")
    print(f"  Sample mean: [{rw_samples[1000:].mean(axis=0)[0]:.3f}, {rw_samples[1000:].mean(axis=0)[1]:.3f}]")
    
    print(f"\nTrue mean: [{mu[0]:.3f}, {mu[1]:.3f}]")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MALA trajectory
    n_show = 200
    axes[0, 0].plot(mala_samples[:n_show, 0], mala_samples[:n_show, 1], 
                   'b-', alpha=0.3, linewidth=0.5)
    axes[0, 0].scatter(mala_samples[:n_show, 0], mala_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='viridis', s=20)
    axes[0, 0].plot(mu[0], mu[1], 'r*', markersize=15, label='True mean')
    axes[0, 0].set_xlabel(r'$x_1$')
    axes[0, 0].set_ylabel(r'$x_2$')
    axes[0, 0].set_title(f'MALA Trajectory\nAccept: {mala_acc:.1%}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # RW trajectory
    axes[0, 1].plot(rw_samples[:n_show, 0], rw_samples[:n_show, 1], 
                   'r-', alpha=0.3, linewidth=0.5)
    axes[0, 1].scatter(rw_samples[:n_show, 0], rw_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='plasma', s=20)
    axes[0, 1].plot(mu[0], mu[1], 'b*', markersize=15, label='True mean')
    axes[0, 1].set_xlabel(r'$x_1$')
    axes[0, 1].set_ylabel(r'$x_2$')
    axes[0, 1].set_title(f'Random Walk Trajectory\nAccept: {rw_acc:.1%}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Autocorrelation
    def autocorr(x, lag):
        x = x - x.mean()
        return np.correlate(x, x, mode='full')[len(x)-1+lag] / np.correlate(x, x, mode='full')[len(x)-1]
    
    lags = range(0, 50)
    mala_acf = [autocorr(mala_samples[1000:, 0], lag) for lag in lags]
    rw_acf = [autocorr(rw_samples[1000:, 0], lag) for lag in lags]
    
    axes[0, 2].plot(lags, mala_acf, label='MALA', linewidth=2)
    axes[0, 2].plot(lags, rw_acf, label='Random Walk', linewidth=2)
    axes[0, 2].set_xlabel('Lag')
    axes[0, 2].set_ylabel('Autocorrelation')
    axes[0, 2].set_title(r'ACF Comparison ($x_1$)')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Trace plots
    axes[1, 0].plot(mala_samples[:500, 0], linewidth=0.5, alpha=0.7, label=r'$x_1$')
    axes[1, 0].plot(mala_samples[:500, 1], linewidth=0.5, alpha=0.7, label=r'$x_2$')
    axes[1, 0].axhline(mu[0], color='r', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(mu[1], color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('MALA Trace')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(rw_samples[:500, 0], linewidth=0.5, alpha=0.7, label=r'$x_1$')
    axes[1, 1].plot(rw_samples[:500, 1], linewidth=0.5, alpha=0.7, label=r'$x_2$')
    axes[1, 1].axhline(mu[0], color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(mu[1], color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Random Walk Trace')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Scatter comparison
    axes[1, 2].scatter(mala_samples[1000:, 0], mala_samples[1000:, 1], 
                      alpha=0.3, s=10, label='MALA')
    axes[1, 2].scatter(rw_samples[1000:, 0], rw_samples[1000:, 1], 
                      alpha=0.3, s=10, label='RW')
    axes[1, 2].plot(mu[0], mu[1], 'r*', markersize=15, label='True mean')
    axes[1, 2].set_xlabel(r'$x_1$')
    axes[1, 2].set_ylabel(r'$x_2$')
    axes[1, 2].set_title('Sample Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'mh_langevin_mala.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: mh_langevin_mala.png")
    print("\nKey Insights:")
    print("  • MALA uses gradient to guide proposals")
    print("  • More efficient mixing than random walk")
    print("  • Asymmetric proposal requires MH correction")
    print("  • Foundation for advanced methods like HMC")


# =============================================================================
# PART 3: ADAPTIVE METROPOLIS-HASTINGS
# =============================================================================

def adaptive_metropolis():
    """
    Example 3: Adaptive Metropolis Algorithm
    ========================================
    
    Idea: Learn optimal proposal covariance during sampling
    
    Proposal: N(x, λ·Σₜ) where
    - Σₜ = empirical covariance of samples so far
    - λ = 2.38²/d (optimal scaling in d dimensions)
    
    Note: Must be careful to preserve ergodicity
    - Adaptation must diminish over time OR
    - Use robust AM variants
    
    Advantage: Automatically adapts to target geometry
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Adaptive Metropolis")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target: 2D Gaussian with strong correlation
    mu = np.array([0.0, 0.0])
    rho = 0.9  # Strong correlation
    sigma = np.array([1.0, 3.0])
    cov = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2]
    ])
    
    def target_log_pdf(x):
        diff = x - mu
        return -0.5 * diff @ np.linalg.inv(cov) @ diff
    
    # Adaptive Metropolis
    def adaptive_mh(n_samples, adapt_interval=100):
        """
        Adaptive Metropolis with covariance learning
        
        Updates proposal covariance every adapt_interval iterations
        """
        d = 2  # dimension
        samples = np.zeros((n_samples, d))
        samples[0] = np.zeros(d)
        n_accepted = 0
        
        # Initial proposal covariance
        proposal_cov = np.eye(d) * 0.1
        
        # Optimal scaling constant
        s_d = 2.38**2 / d
        
        # Track acceptance over windows
        accept_history = []
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            
            # Propose with current covariance
            x_proposal = np.random.multivariate_normal(x_current, proposal_cov)
            
            # MH acceptance
            log_alpha = target_log_pdf(x_proposal) - target_log_pdf(x_current)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_proposal
                n_accepted += 1
                accept_history.append(1)
            else:
                samples[t] = x_current
                accept_history.append(0)
            
            # Adapt proposal covariance
            if t > 100 and t % adapt_interval == 0:
                # Compute empirical covariance
                sample_cov = np.cov(samples[:t].T)
                
                # Update proposal (with small regularization)
                proposal_cov = s_d * sample_cov + 1e-6 * np.eye(d)
        
        return samples, n_accepted / (n_samples - 1), accept_history
    
    # Standard Metropolis for comparison
    def standard_mh(n_samples, proposal_std):
        samples = np.zeros((n_samples, 2))
        samples[0] = np.zeros(2)
        n_accepted = 0
        accept_history = []
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            x_proposal = x_current + np.random.randn(2) * proposal_std
            
            log_alpha = target_log_pdf(x_proposal) - target_log_pdf(x_current)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_proposal
                n_accepted += 1
                accept_history.append(1)
            else:
                samples[t] = x_current
                accept_history.append(0)
        
        return samples, n_accepted / (n_samples - 1), accept_history
    
    # Run both
    n_samples = 5000
    adapt_samples, adapt_acc, adapt_hist = adaptive_mh(n_samples)
    std_samples, std_acc, std_hist = standard_mh(n_samples, 0.5)

    
    print(f"\nTarget: 2D Gaussian with strong correlation (ρ={rho})")
    print(f"\nAdaptive Metropolis:")
    print(f"  Final acceptance: {adapt_acc:.2%}")
    print(f"  Sample mean: [{adapt_samples[2000:].mean(axis=0)[0]:.3f}, {adapt_samples[2000:].mean(axis=0)[1]:.3f}]")
    
    print(f"\nStandard Metropolis (σ=0.5):")
    print(f"  Acceptance: {std_acc:.2%}")
    print(f"  Sample mean: [{std_samples[2000:].mean(axis=0)[0]:.3f}, {std_samples[2000:].mean(axis=0)[1]:.3f}]")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Trajectories
    n_show = 300
    axes[0, 0].plot(adapt_samples[:n_show, 0], adapt_samples[:n_show, 1], 
                   'b-', alpha=0.3, linewidth=0.5)
    axes[0, 0].scatter(adapt_samples[:n_show, 0], adapt_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='viridis', s=20)
    axes[0, 0].set_xlabel(r'$x_1$')
    axes[0, 0].set_ylabel(r'$x_2$')
    axes[0, 0].set_title(f'Adaptive MH\nAccept: {adapt_acc:.1%}')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(std_samples[:n_show, 0], std_samples[:n_show, 1], 
                   'r-', alpha=0.3, linewidth=0.5)
    axes[0, 1].scatter(std_samples[:n_show, 0], std_samples[:n_show, 1], 
                      c=np.arange(n_show), cmap='plasma', s=20)
    axes[0, 1].set_xlabel(r'$x_1$')
    axes[0, 1].set_ylabel(r'$x_2$')
    axes[0, 1].set_title(f'Standard MH\nAccept: {std_acc:.1%}')
    axes[0, 1].grid(alpha=0.3)
    
    # Acceptance rate over time
    window = 100
    def moving_avg(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')
    
    adapt_rate = moving_avg(adapt_hist, window)
    std_rate = moving_avg(std_hist, window)
    
    axes[0, 2].plot(adapt_rate, label='Adaptive', linewidth=2)
    axes[0, 2].plot(std_rate, label='Standard', linewidth=2)
    axes[0, 2].axhline(0.234, color='k', linestyle='--', label='Optimal (23.4%)')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Acceptance Rate')
    axes[0, 2].set_title(f'Acceptance Rate (window={window})')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Scatter plots of final samples
    axes[1, 0].scatter(adapt_samples[2000:, 0], adapt_samples[2000:, 1], 
                      alpha=0.2, s=5)
    axes[1, 0].set_xlabel(r'$x_1$')
    axes[1, 0].set_ylabel(r'$x_2$')
    axes[1, 0].set_title('Adaptive MH Samples')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    axes[1, 1].scatter(std_samples[2000:, 0], std_samples[2000:, 1], 
                      alpha=0.2, s=5, color='orange')
    axes[1, 1].set_xlabel(r'$x_1$')
    axes[1, 1].set_ylabel(r'$x_2$')
    axes[1, 1].set_title('Standard MH Samples')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    # ACF comparison
    def autocorr(x, lag):
        x = x - x.mean()
        return np.correlate(x, x, mode='full')[len(x)-1+lag] / np.correlate(x, x, mode='full')[len(x)-1]
    
    lags = range(0, 50)
    adapt_acf = [autocorr(adapt_samples[2000:, 0], lag) for lag in lags]
    std_acf = [autocorr(std_samples[2000:, 0], lag) for lag in lags]
    
    axes[1, 2].plot(lags, adapt_acf, label='Adaptive', linewidth=2)
    axes[1, 2].plot(lags, std_acf, label='Standard', linewidth=2)
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylabel('Autocorrelation')
    axes[1, 2].set_title('ACF Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'mh_adaptive.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: mh_adaptive.png")
    print("\nKey Insights:")
    print("  • Adaptive MH learns optimal proposal automatically")
    print("  • Better mixing for correlated targets")
    print("  • Acceptance rate converges to optimal ~23%")
    print("  • Useful when target geometry is unknown")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METROPOLIS-HASTINGS: COMPREHENSIVE TUTORIAL")
    print("=" * 70)
    print("\nTopics covered:")
    print("  1. Independence sampler (asymmetric proposal)")
    print("  2. Langevin dynamics / MALA (gradient-based)")
    print("  3. Adaptive Metropolis (learned proposals)")
    print("\n" + "=" * 70)
    
    independence_sampler()
    langevin_mala()
    adaptive_metropolis()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • mh_independence_sampler.png")
    print("  • mh_langevin_mala.png")
    print("  • mh_adaptive.png")
    print("\nNext steps:")
    print("  • Study Hamiltonian Monte Carlo (HMC)")
    print("  • Explore No-U-Turn Sampler (NUTS)")
    print("  • Learn about convergence diagnostics")
    print("  • Try parallel tempering / replica exchange")
    print("\n" + "=" * 70)
    print("\nCongratulations on completing the MCMC series!")
    print("You now understand:")
    print("  ✓ Gibbs sampling (01_gibbs.py)")
    print("  ✓ Metropolis algorithm (02_metropolis.py)")
    print("  ✓ Metropolis-Hastings (03_metropolis_hastings.py)")
    print("\n" + "=" * 70)
