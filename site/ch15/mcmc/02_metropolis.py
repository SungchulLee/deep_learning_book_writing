"""
02_metropolis.py

METROPOLIS ALGORITHM: COMPREHENSIVE TUTORIAL
============================================

Learning Objectives:
-------------------
1. Understand the Metropolis algorithm (symmetric proposals)
2. Learn about proposal distributions and tuning
3. See detailed balance in action
4. Practice with various target distributions

Mathematical Foundation:
-----------------------
Goal:
    Sample from a target distribution p(x) when we can evaluate p(x) only up
    to a constant (i.e., we know an unnormalized density p̃(x) ∝ p(x)).

Metropolis Algorithm (Symmetric Proposals):
-------------------------------------------
1. Choose an initial point x₀
2. For t = 1, 2, ...
   a. Propose a candidate x' ~ q(x' | x_t)
      where q is SYMMETRIC: q(x' | x) = q(x | x')
      Common choice: q(x' | x) = N(x, σ²I) (random walk)
   
   b. Compute acceptance probability:
      α = min(1, p(x') / p(x_t))
      
      Note: Because q is symmetric, q(x_t | x') = q(x' | x_t) cancels!
   
   c. With probability α, accept and set x_{t+1} = x'
      Otherwise, reject and set x_{t+1} = x_t

Why Symmetry Matters:
--------------------
The general Metropolis-Hastings acceptance ratio is:
    α = min(1, [p(x') q(x_t | x')] / [p(x_t) q(x' | x_t)])

When q is symmetric (q(x' | x) = q(x | x')), the ratio simplifies to:
    α = min(1, p(x') / p(x_t))

This simplification is the Metropolis algorithm!

Common Symmetric Proposals:
---------------------------
1. Random walk: q(x' | x) = N(x, σ²I)
2. Uniform ball: q(x' | x) = Uniform(B_r(x))
3. Independent: q(x' | x) = q(x') (symmetric distribution)

Tuning the Proposal:
-------------------
- σ too small → high acceptance, slow exploration
- σ too large → low acceptance, wasted iterations
- Target: 23-50% acceptance rate for optimal mixing
- Roberts & Rosenthal (2001): optimal rate ≈ 23.4% for high dimensions

Connection to Diffusion:
-----------------------
- Random walk proposal is like adding noise
- Acceptance/rejection is like a learned denoising step
- Understanding Metropolis helps grasp diffusion model dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, gamma, beta as beta_dist
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


# =============================================================================
# PART 1: METROPOLIS FOR 1D DISTRIBUTIONS
# =============================================================================

def metropolis_1d_gaussian():
    """
    Example 1: Metropolis Algorithm for 1D Gaussian
    ===============================================
    
    Target: p(x) = N(μ, σ²) (standard normal)
    Proposal: q(x'|x) = N(x, σ_prop²) (random walk)
    
    Since proposal is symmetric: q(x'|x) = q(x|x')
    Acceptance ratio simplifies: α = min(1, p(x') / p(x))
    
    Algorithm:
    1. Start at x₀
    2. Propose x' = x_t + N(0, σ_prop²)
    3. Accept if p(x') / p(x_t) > uniform(0,1)
    
    Key Parameters:
    - σ_prop (proposal std): Controls step size
    - Too small → slow mixing
    - Too large → low acceptance
    """
    print("=" * 70)
    print("EXAMPLE 1: Metropolis for 1D Gaussian")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target distribution
    target_mean = 0.0
    target_std = 1.0
    
    def target_pdf(x):
        """Target: Standard normal"""
        return norm.pdf(x, target_mean, target_std)
    
    # Metropolis algorithm
    def metropolis_sample(n_samples, proposal_std, x_init=0.0):
        """
        Simple Metropolis sampler with symmetric random walk proposal
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        proposal_std : float
            Standard deviation of proposal distribution
        x_init : float
            Initial value
        
        Returns:
        --------
        samples : ndarray
            Generated samples
        acceptance_rate : float
            Fraction of accepted proposals
        """
        samples = np.zeros(n_samples)
        samples[0] = x_init
        n_accepted = 0
        
        for t in range(1, n_samples):
            # Current state
            x_current = samples[t-1]
            
            # Propose new state (symmetric random walk)
            x_proposal = x_current + np.random.randn() * proposal_std
            
            # Compute acceptance ratio
            # For symmetric proposal: α = min(1, p(x')/p(x))
            alpha = min(1.0, target_pdf(x_proposal) / target_pdf(x_current))
            
            # Accept or reject
            if np.random.rand() < alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        acceptance_rate = n_accepted / (n_samples - 1)
        return samples, acceptance_rate
    
    # Try different proposal standard deviations
    n_samples = 5000
    proposal_stds = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\nRunning Metropolis with {n_samples} iterations")
    print("\nProposal σ | Acceptance Rate | Sample Mean | Sample Std")
    print("-" * 65)
    
    fig, axes = plt.subplots(3, len(proposal_stds), figsize=(20, 12))
    
    for idx, prop_std in enumerate(proposal_stds):
        samples, acc_rate = metropolis_sample(n_samples, prop_std)
        
        print(f"   {prop_std:4.1f}   |     {acc_rate:6.2%}      |    {samples.mean():6.3f}  |   {samples.std():5.3f}")
        
        # Trace plot
        ax = axes[0, idx]
        ax.plot(samples[:500], linewidth=0.5, alpha=0.7)
        ax.axhline(target_mean, color='r', linestyle='--', linewidth=2, label='True mean')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('x', fontsize=10)
        ax.set_title(f'Trace (σ={prop_std})\nAccept: {acc_rate:.1%}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Histogram vs true distribution
        ax = axes[1, idx]
        ax.hist(samples[1000:], bins=50, density=True, alpha=0.7, 
                edgecolor='black', label='MCMC samples')
        x_grid = np.linspace(-4, 4, 200)
        ax.plot(x_grid, target_pdf(x_grid), 'r-', linewidth=2, label='True p(x)')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Distribution (σ={prop_std})', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Autocorrelation
        ax = axes[2, idx]
        def autocorr(x, lag):
            x = x - x.mean()
            return np.correlate(x, x, mode='full')[len(x)-1+lag] / np.correlate(x, x, mode='full')[len(x)-1]
        
        lags = range(0, 50)
        acf = [autocorr(samples[1000:], lag) for lag in lags]
        ax.plot(lags, acf, 'b-', linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Lag', fontsize=10)
        ax.set_ylabel('Autocorrelation', fontsize=10)
        ax.set_title(f'ACF (σ={prop_std})', fontsize=12)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'metropolis_1d_tuning.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: metropolis_1d_tuning.png")
    print("\nKey Observations:")
    print("  • σ = 0.1: Very high acceptance (~99%), but slow exploration (high autocorr)")
    print("  • σ = 0.5-1.0: Optimal range (~40-70% acceptance), good mixing")
    print("  • σ = 5.0: Low acceptance (~15%), many wasted iterations")
    print("\nRule of thumb: Target 23-50% acceptance rate for efficient sampling")


# =============================================================================
# PART 2: DETAILED BALANCE AND STATIONARITY
# =============================================================================

def demonstrate_detailed_balance():
    """
    Demonstration: Why Metropolis Works (Detailed Balance)
    ======================================================
    
    Detailed Balance Condition:
    ---------------------------
    p(x) T(x'|x) = p(x') T(x|x')
    
    Where T(x'|x) is the transition probability from x to x'.
    
    For Metropolis with symmetric proposal:
    T(x'|x) = q(x'|x) · α(x'|x)
    
    where α(x'|x) = min(1, p(x')/p(x))
    
    Proof of Detailed Balance:
    --------------------------
    Case 1: p(x') ≥ p(x)
        α(x'|x) = 1, α(x|x') = p(x)/p(x')
        LHS = p(x) · q(x'|x) · 1 = p(x) · q(x'|x)
        RHS = p(x') · q(x|x') · p(x)/p(x') = p(x) · q(x|x')
        Since q is symmetric, LHS = RHS ✓
    
    Case 2: p(x') < p(x)
        α(x'|x) = p(x')/p(x), α(x|x') = 1
        Similar proof...
    
    Consequence:
    -----------
    Detailed balance ⟹ p(x) is stationary distribution
    Under mild conditions ⟹ chain converges to p(x)
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Detailed Balance for Metropolis")
    print("=" * 70)
    
    # Simple discrete example to show detailed balance numerically
    states = np.array([0, 1, 2, 3, 4])
    n_states = len(states)
    
    # Target distribution (arbitrary)
    p_unnorm = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    p = p_unnorm / p_unnorm.sum()
    
    print("\nTarget distribution p(x):")
    for i, prob in enumerate(p):
        print(f"  State {i}: {prob:.3f}")
    
    # Symmetric proposal: propose neighboring states with equal probability
    def proposal_prob(i, j):
        """Symmetric random walk on discrete states"""
        if abs(i - j) == 1:
            return 0.5
        elif i == j == 0 or i == j == n_states - 1:
            return 0.5  # Stay or move at boundaries
        return 0.0
    
    # Build transition matrix
    T = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            
            q_ij = proposal_prob(i, j)
            if q_ij > 0:
                # Metropolis acceptance
                alpha_ij = min(1.0, p[j] / p[i])
                T[i, j] = q_ij * alpha_ij
        
        # Diagonal: probability of staying
        T[i, i] = 1.0 - T[i, :].sum()
    
    print("\nTransition matrix T:")
    print(T)
    
    # Check detailed balance
    print("\nChecking detailed balance: p(i)·T(j|i) = p(j)·T(i|j)")
    print("-" * 60)
    max_violation = 0
    for i in range(n_states):
        for j in range(i+1, n_states):  # Only check upper triangle
            lhs = p[i] * T[j, i]
            rhs = p[j] * T[i, j]
            violation = abs(lhs - rhs)
            max_violation = max(max_violation, violation)
            if violation > 1e-10:
                print(f"  {i}↔{j}: p({i})T({j}|{i})={lhs:.6f}, p({j})T({i}|{j})={rhs:.6f}, diff={violation:.2e}")
    
    if max_violation < 1e-10:
        print("  ✓ Detailed balance holds for all state pairs!")
    
    # Simulate to verify convergence
    np.random.seed(42)
    n_steps = 10000
    state_counts = np.zeros(n_states)
    state = 0  # Start at state 0
    
    for step in range(n_steps):
        # Random walk proposal
        if state == 0:
            proposed = 1
        elif state == n_states - 1:
            proposed = n_states - 2
        else:
            proposed = state + np.random.choice([-1, 1])
        
        # Metropolis acceptance
        alpha = min(1.0, p[proposed] / p[state])
        if np.random.rand() < alpha:
            state = proposed
        
        state_counts[state] += 1
    
    empirical_dist = state_counts / state_counts.sum()
    
    print(f"\nEmpirical distribution after {n_steps} steps:")
    print("State | Target p(i) | Empirical | Error")
    print("-" * 45)
    for i in range(n_states):
        error = abs(p[i] - empirical_dist[i])
        print(f"  {i}   |   {p[i]:.4f}   |  {empirical_dist[i]:.4f}  | {error:.2e}")
    
    print("\n✓ Detailed balance ensures convergence to target distribution!")


# =============================================================================
# PART 3: METROPOLIS FOR 2D "BANANA" DISTRIBUTION
# =============================================================================

def metropolis_2d_banana():
    """
    Example 2: 2D Metropolis for \"Banana\" Distribution
    ==================================================
    
    Target: A non-Gaussian, banana-shaped distribution
    
    This example shows:
    - How Metropolis explores complex, non-standard distributions
    - The effect of correlation on mixing
    - Visualization of MCMC trajectory
    
    The banana distribution is defined by:
    log p(x₁, x₂) = -0.5 * (x₁² + (x₂ - x₁²)²)
    
    This creates a curved, banana-shaped region of high probability.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 2D Metropolis for Banana Distribution")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target: Banana-shaped distribution
    def target_log_pdf(x):
        """
        Log pdf of banana-shaped distribution
        
        The transformation y₂ = x₂ - x₁² creates curvature
        """
        x1, x2 = x[0], x[1]
        return -0.5 * (x1**2 + (x2 - x1**2)**2)
    
    def target_pdf(x):
        return np.exp(target_log_pdf(x))
    
    # Metropolis with 2D symmetric random walk
    def metropolis_2d(n_samples, proposal_std, x_init=None):
        """
        2D Metropolis sampler
        
        Proposal: x' = x + N(0, σ²I)
        """
        if x_init is None:
            x_init = np.array([0.0, 0.0])
        
        samples = np.zeros((n_samples, 2))
        samples[0] = x_init
        n_accepted = 0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            
            # Symmetric random walk proposal
            x_proposal = x_current + np.random.randn(2) * proposal_std
            
            # Acceptance ratio (use log for numerical stability)
            log_alpha = target_log_pdf(x_proposal) - target_log_pdf(x_current)
            
            # Accept/reject
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / (n_samples - 1)
    
    # Run MCMC
    n_samples = 10000
    proposal_std = 0.5
    
    samples, acc_rate = metropolis_2d(n_samples, proposal_std)
    
    print(f"\nMCMC: {n_samples} samples, proposal σ = {proposal_std}")
    print(f"Acceptance rate: {acc_rate:.2%}")
    print(f"Sample mean: [{samples[2000:].mean(axis=0)[0]:.3f}, {samples[2000:].mean(axis=0)[1]:.3f}]")
    print(f"Sample std:  [{samples[2000:].std(axis=0)[0]:.3f}, {samples[2000:].std(axis=0)[1]:.3f}]")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main: 2D scatter with trajectory
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Show trajectory for first 1000 samples
    n_show = 1000
    colors = np.arange(n_show)
    scatter = ax_main.scatter(samples[:n_show, 0], samples[:n_show, 1], 
                             c=colors, cmap='viridis', s=20, alpha=0.6)
    ax_main.plot(samples[:n_show, 0], samples[:n_show, 1], 
                'k-', alpha=0.2, linewidth=0.5)
    ax_main.plot(samples[0, 0], samples[0, 1], 'ro', markersize=10, 
                label='Start', zorder=5)
    
    # Add contours of true distribution
    x1_grid = np.linspace(-3, 3, 100)
    x2_grid = np.linspace(-2, 6, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = np.exp(target_log_pdf([X1[i, j], X2[i, j]]))
    ax_main.contour(X1, X2, Z, levels=10, colors='red', alpha=0.4, linewidths=1.5)
    
    plt.colorbar(scatter, ax=ax_main, label='Iteration')
    ax_main.set_xlabel('x₁', fontsize=12)
    ax_main.set_ylabel('x₂', fontsize=12)
    ax_main.set_title(f'MCMC Trajectory (first {n_show} samples)\nAcceptance: {acc_rate:.1%}', 
                     fontsize=14)
    ax_main.legend(fontsize=10)
    ax_main.grid(alpha=0.3)
    
    # Marginal x₁
    ax_marg1 = fig.add_subplot(gs[2, 0:2])
    ax_marg1.hist(samples[2000:, 0], bins=60, density=True, alpha=0.7, 
                 edgecolor='black')
    ax_marg1.set_xlabel('x₁', fontsize=12)
    ax_marg1.set_ylabel('Density', fontsize=12)
    ax_marg1.set_title('Marginal p(x₁)', fontsize=12)
    ax_marg1.grid(alpha=0.3)
    
    # Marginal x₂
    ax_marg2 = fig.add_subplot(gs[0:2, 2])
    ax_marg2.hist(samples[2000:, 1], bins=60, density=True, alpha=0.7,
                 edgecolor='black', orientation='horizontal')
    ax_marg2.set_ylabel('x₂', fontsize=12)
    ax_marg2.set_xlabel('Density', fontsize=12)
    ax_marg2.set_title('Marginal p(x₂)', fontsize=12)
    ax_marg2.grid(alpha=0.3)
    
    # Trace plots
    ax_trace = fig.add_subplot(gs[2, 2])
    ax_trace.plot(samples[:1000, 0], linewidth=0.5, alpha=0.7, label='x₁')
    ax_trace.plot(samples[:1000, 1], linewidth=0.5, alpha=0.7, label='x₂')
    ax_trace.set_xlabel('Iteration', fontsize=10)
    ax_trace.set_ylabel('Value', fontsize=10)
    ax_trace.set_title('Trace (first 1000)', fontsize=12)
    ax_trace.legend(fontsize=8)
    ax_trace.grid(alpha=0.3)
    
    fig_path = os.path.join(os.path.dirname(__file__), 'metropolis_2d_banana.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: metropolis_2d_banana.png")
    print("\nKey Observations:")
    print("  • MCMC successfully explores the banana-shaped region")
    print("  • Symmetric proposals work even for complex geometries")
    print("  • The chain gradually discovers the distribution's structure")


# =============================================================================
# PART 4: METROPOLIS FOR GAMMA DISTRIBUTION
# =============================================================================

def metropolis_gamma():
    """
    Example 3: Metropolis for Gamma Distribution
    ============================================
    
    Target: Gamma(α, β) distribution (positive support)
    Challenge: Distribution has support on (0, ∞)
    
    Solution approaches:
    1. Truncate proposals that go negative
    2. Use log-space (sample log(x) instead)
    3. Use asymmetric proposals (→ Metropolis-Hastings)
    
    Here we use approach #1 for simplicity.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Metropolis for Gamma Distribution")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Target parameters
    alpha = 3.0  # Shape
    beta = 2.0   # Rate
    
    def target_pdf(x):
        """Gamma(α, β) pdf"""
        if x <= 0:
            return 0.0
        return gamma.pdf(x, alpha, scale=1/beta)
    
    def metropolis_gamma(n_samples, proposal_std, x_init=1.0):
        """
        Metropolis for Gamma distribution
        
        Handles boundary at x=0 by rejecting negative proposals
        """
        samples = np.zeros(n_samples)
        samples[0] = x_init
        n_accepted = 0
        n_proposals = 0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            
            # Symmetric proposal
            x_proposal = x_current + np.random.randn() * proposal_std
            n_proposals += 1
            
            # Reject if proposal is negative
            if x_proposal <= 0:
                samples[t] = x_current
                continue
            
            # Metropolis acceptance
            alpha_accept = min(1.0, target_pdf(x_proposal) / target_pdf(x_current))
            
            if np.random.rand() < alpha_accept:
                samples[t] = x_proposal
                n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples, n_accepted / n_proposals
    
    # Run with different proposal stds
    n_samples = 5000
    proposal_stds = [0.3, 0.5, 1.0, 2.0]
    
    print(f"\nTarget: Gamma(α={alpha}, β={beta})")
    print(f"True mean: {alpha/beta:.3f}, True std: {np.sqrt(alpha)/beta:.3f}")
    print("\nProposal σ | Accept Rate | Sample Mean | Sample Std")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, len(proposal_stds), figsize=(18, 10))
    
    for idx, prop_std in enumerate(proposal_stds):
        samples, acc_rate = metropolis_gamma(n_samples, prop_std)
        
        print(f"   {prop_std:4.1f}   |   {acc_rate:6.2%}    |    {samples[1000:].mean():5.3f}   |   {samples[1000:].std():5.3f}")
        
        # Trace
        ax = axes[0, idx]
        ax.plot(samples[:500], linewidth=0.5, alpha=0.7)
        ax.axhline(alpha/beta, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('x')
        ax.set_title(f'Trace (σ={prop_std})\nAccept: {acc_rate:.1%}')
        ax.grid(alpha=0.3)
        
        # Histogram
        ax = axes[1, idx]
        ax.hist(samples[1000:], bins=50, density=True, alpha=0.7, edgecolor='black')
        x_grid = np.linspace(0, 6, 200)
        ax.plot(x_grid, gamma.pdf(x_grid, alpha, scale=1/beta), 
               'r-', linewidth=2, label='True')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution (σ={prop_std})')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'metropolis_gamma.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Plot saved: metropolis_gamma.png")
    print("\nKey Insight:")
    print("  • Boundary constraints reduce effective acceptance rate")
    print("  • Alternative: Use Metropolis-Hastings with asymmetric proposals")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METROPOLIS ALGORITHM: COMPREHENSIVE TUTORIAL")
    print("=" * 70)
    print("\nTopics covered:")
    print("  1. Metropolis for 1D Gaussian (with tuning)")
    print("  2. Detailed balance demonstration")
    print("  3. 2D Metropolis for banana distribution")
    print("  4. Metropolis for Gamma distribution (constrained support)")
    print("\n" + "=" * 70)
    
    metropolis_1d_gaussian()
    demonstrate_detailed_balance()
    metropolis_2d_banana()
    metropolis_gamma()
    
    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • metropolis_1d_tuning.png")
    print("  • metropolis_2d_banana.png")
    print("  • metropolis_gamma.png")
    print("\nNext: 03_metropolis_hastings.py")
    print("      (Asymmetric proposals for greater flexibility)")
    print("\n" + "=" * 70)
