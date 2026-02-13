"""
06_nuts.py

NO-U-TURN SAMPLER (NUTS): COMPREHENSIVE TUTORIAL
================================================

Learning Objectives:
-------------------
1. Understand the U-turn problem in HMC
2. Learn the NUTS algorithm for automatic trajectory length
3. Master dual averaging for step size adaptation
4. Compare NUTS with vanilla HMC
5. Apply to complex inference problems

Mathematical Foundation:
-----------------------
Goal:
    Eliminate HMC's main tuning challenge (number of leapfrog steps L) by
    automatically determining when to stop the trajectory.

The U-Turn Problem:
------------------
In vanilla HMC:
    - Too few steps L → proposals close to starting point
    - Too many steps L → trajectory may "U-turn" back toward start
    - Optimal L varies by region of parameter space!

What is a U-turn?
    A trajectory makes a U-turn when it starts moving back toward its
    starting point, wasting computation on exploring already-visited regions.

Mathematical Definition:
    Let x₀ be starting position, p₀ be starting momentum
    
    U-turn occurs when:
        (xₜ - x₀) · pₜ < 0  or  (xₜ - x₀) · p₀ < 0
    
    Intuition: Position is no longer "aligned" with momentum direction

NUTS Algorithm:
--------------
Key idea: Build trajectory recursively until a U-turn is detected

1. Start with current position x and momentum p ~ N(0, M)
2. Randomly choose direction: forward or backward
3. Double trajectory length until U-turn detected
4. Sample uniformly from all valid positions in trajectory
5. Accept/reject using slice sampling criterion

Recursive Doubling:
    - Start with single leapfrog step
    - Double: run trajectory forward AND backward
    - Continue until U-turn or max tree depth
    - Build a "tree" of candidate samples

Benefits:
    - No need to tune L (number of leapfrog steps)
    - Automatically adapts to local geometry
    - Uses all computed positions (via slice sampling)
    - Robust across different target distributions

Slice Sampling Criterion:
    Instead of standard Metropolis acceptance, NUTS uses:
        u ~ Uniform(0, exp(-H(x₀, p₀)))
    
    Accept any position (x', p') where:
        exp(-H(x', p')) ≥ u
    
    This creates a "slice" of valid samples!

Dual Averaging for Step Size:
-----------------------------
NUTS also adapts step size ε during warmup:
    
    Target: δ = 0.65 (65% acceptance rate)
    
    Update rule:
        ε_{m+1} = exp(μ - √m/γ · H̄ₘ)
    
    where H̄ₘ tracks deviation from target acceptance
    
    After warmup: fix ε and run production samples

Algorithm Outline:
-----------------
BuildTree(x, p, u, direction, depth, ε):
    if depth == 0:
        # Base case: single leapfrog step
        Take one step in direction
        Return (x', p', candidate set)
    else:
        # Recursive case: double tree
        Build subtree_1
        if no U-turn in subtree_1:
            Build subtree_2 extending subtree_1
        Combine subtrees, check U-turn
        Return combined tree

Advantages over HMC:
-------------------
+ No need to tune L (major benefit!)
+ Adapts step size automatically (dual averaging)
+ More efficient use of gradient evaluations
+ More robust across different posteriors
+ Default sampler in Stan and PyMC

Disadvantages:
-------------
- More complex implementation
- Slightly more gradient evaluations per sample
- Still requires differentiable target
- Can be slow for very stiff posteriors

Connection to Modern ML:
-----------------------
- Stan: Uses NUTS as default sampler
- PyMC: Default MCMC algorithm
- TensorFlow Probability: tfp.mcmc.NoUTurnSampler
- Normalizing flows: Related tree-building ideas
- Neural ODEs: Similar adaptive time stepping

Performance:
-----------
NUTS typically needs ~100-1000 samples (vs 10,000+ for RWM)
    - Effective sample size per gradient evaluation: 10-100x better
    - Well-suited for expensive likelihood evaluations
    - Scales to thousands of dimensions

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
import seaborn as sns
from collections import namedtuple
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Named tuple for tree state
TreeState = namedtuple('TreeState', 
    ['leftmost_x', 'leftmost_p', 'rightmost_x', 'rightmost_p',
     'x_proposals', 'n_proposals', 'accept_sum', 'n_steps'])


# =============================================================================
# PART 1: UNDERSTANDING U-TURNS (BEGINNER)
# =============================================================================

def visualize_uturn():
    """
    Example 1: Visualizing the U-Turn Problem
    =========================================
    
    Show how HMC trajectories can waste computation by making U-turns.
    """
    print("=" * 80)
    print("EXAMPLE 1: Visualizing U-Turns in HMC")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 2D Gaussian target
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    # Leapfrog integrator
    def leapfrog_step(x, p, epsilon):
        """Single leapfrog step"""
        p_half = p - (epsilon / 2) * grad_U(x)
        x_new = x + epsilon * p_half
        p_new = p_half - (epsilon / 2) * grad_U(x_new)
        return x_new, p_new
    
    # Run trajectory with different lengths
    x0 = np.array([1.5, 0.0])
    p0 = np.array([0.0, 2.0])  # Momentum pointing perpendicular
    
    epsilon = 0.2
    max_steps = 50
    
    # Collect trajectory
    trajectory_x = [x0.copy()]
    trajectory_p = [p0.copy()]
    
    # Compute U-turn criterion
    uturn_criterion = []
    
    x, p = x0.copy(), p0.copy()
    for i in range(max_steps):
        x, p = leapfrog_step(x, p, epsilon)
        trajectory_x.append(x.copy())
        trajectory_p.append(p.copy())
        
        # Check U-turn: (x - x0) · p < 0
        delta_x = x - x0
        uturn_val = np.dot(delta_x, p)
        uturn_criterion.append(uturn_val)
    
    trajectory_x = np.array(trajectory_x)
    trajectory_p = np.array(trajectory_p)
    uturn_criterion = np.array(uturn_criterion)
    
    # Find first U-turn
    uturn_idx = np.where(uturn_criterion < 0)[0]
    if len(uturn_idx) > 0:
        first_uturn = uturn_idx[0]
    else:
        first_uturn = len(uturn_criterion)
    
    print(f"U-turn detected at step {first_uturn}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trajectory in position space
    ax = axes[0]
    
    # Contours
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    
    # Before U-turn (good exploration)
    colors_before = plt.cm.Greens(np.linspace(0.3, 1, first_uturn))
    for i in range(min(first_uturn, len(trajectory_x)-1)):
        ax.plot(trajectory_x[i:i+2, 0], trajectory_x[i:i+2, 1],
               'o-', color=colors_before[i], markersize=4, linewidth=2)
    
    # After U-turn (wasted computation)
    if first_uturn < len(trajectory_x) - 1:
        colors_after = plt.cm.Reds(np.linspace(0.3, 1, 
                                   len(trajectory_x) - first_uturn - 1))
        for i in range(first_uturn, len(trajectory_x)-1):
            idx_color = i - first_uturn
            ax.plot(trajectory_x[i:i+2, 0], trajectory_x[i:i+2, 1],
                   'o-', color=colors_after[idx_color], 
                   markersize=4, linewidth=2)
    
    ax.plot(x0[0], x0[1], 'g*', markersize=25, label='Start', zorder=10)
    ax.plot(trajectory_x[first_uturn, 0], trajectory_x[first_uturn, 1],
           'r*', markersize=25, label='U-turn point', zorder=10)
    
    ax.set_xlabel('$x_1$', fontsize=13)
    ax.set_ylabel('$x_2$', fontsize=13)
    ax.set_title('HMC Trajectory: Before and After U-Turn\n' +
                'Green = useful exploration, Red = wasted',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # U-turn criterion over time
    ax = axes[1]
    
    steps = np.arange(len(uturn_criterion))
    ax.plot(steps, uturn_criterion, 'o-', linewidth=2, markersize=4)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
              label='U-turn threshold')
    ax.axvline(x=first_uturn, color='orange', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'First U-turn (step {first_uturn})')
    
    # Shade regions
    ax.fill_between(steps[:first_uturn], 
                    uturn_criterion[:first_uturn], 
                    alpha=0.3, color='green', label='Good exploration')
    if first_uturn < len(steps):
        ax.fill_between(steps[first_uturn:], 
                        uturn_criterion[first_uturn:],
                        alpha=0.3, color='red', label='Wasted computation')
    
    ax.set_xlabel('Leapfrog step', fontsize=13)
    ax.set_ylabel('$(x - x_0) \\cdot p$', fontsize=13)
    ax.set_title('U-Turn Criterion\n$(x - x_0) \\cdot p < 0$ indicates U-turn',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'nuts_uturn_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: nuts_uturn_visualization.png")
    plt.close()
    
    print("\nKey insight: NUTS stops at U-turn, avoiding wasted computation!")


# =============================================================================
# PART 2: SIMPLIFIED NUTS IMPLEMENTATION (INTERMEDIATE)
# =============================================================================

def nuts_simple():
    """
    Example 2: Simplified NUTS for 2D Gaussian
    ==========================================
    
    Implement a simplified version of NUTS (no tree building)
    to understand the core ideas.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Simplified NUTS Implementation")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target distribution
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    def leapfrog_step(x, p, epsilon):
        """Single leapfrog step"""
        p_half = p - (epsilon / 2) * grad_U(x)
        x_new = x + epsilon * p_half
        p_new = p_half - (epsilon / 2) * grad_U(x_new)
        return x_new, p_new
    
    def check_uturn(x_start, x_end, p_end):
        """Check if trajectory has made a U-turn"""
        # U-turn if (x_end - x_start) · p_end < 0
        return np.dot(x_end - x_start, p_end) < 0
    
    def nuts_single_iteration(x0, epsilon, max_depth=10):
        """
        Single NUTS iteration (simplified)
        
        Build trajectory until U-turn or max depth
        """
        # Sample momentum
        p0 = np.random.randn(len(x0))
        
        # Compute slice variable (slice sampling)
        u = np.random.uniform(0, np.exp(-U(x0) - 0.5 * np.sum(p0**2)))
        
        # Initialize trajectory
        x_left, p_left = x0.copy(), p0.copy()
        x_right, p_right = x0.copy(), p0.copy()
        
        # Collect all candidates
        candidates = [x0.copy()]
        
        depth = 0
        continue_building = True
        
        while continue_building and depth < max_depth:
            # Choose direction randomly
            direction = 2 * np.random.randint(2) - 1  # -1 or 1
            
            if direction == 1:
                # Extend to the right
                x_right, p_right = leapfrog_step(x_right, p_right, epsilon)
                candidates.append(x_right.copy())
            else:
                # Extend to the left
                x_left, p_left = leapfrog_step(x_left, p_left, -epsilon)
                candidates.append(x_left.copy())
            
            # Check if trajectory made a U-turn
            uturn = check_uturn(x_left, x_right, p_right) or \
                    check_uturn(x_left, x_right, p_left)
            
            if uturn:
                continue_building = False
            
            depth += 1
        
        # Select from candidates using slice sampling
        valid_candidates = []
        for x in candidates:
            # Recompute momentum needed for this x
            # (simplified: just check if x is in the slice)
            if np.exp(-U(x)) >= u:  # Simplified check
                valid_candidates.append(x)
        
        if len(valid_candidates) > 0:
            x_next = valid_candidates[np.random.randint(len(valid_candidates))]
        else:
            x_next = x0  # Reject
        
        return x_next, depth, len(candidates)
    
    # Run simplified NUTS
    n_samples = 1000
    x0 = np.array([2.0, 2.0])
    epsilon = 0.2
    
    samples = np.zeros((n_samples, 2))
    samples[0] = x0
    depths = []
    n_evals = []
    
    print("\nRunning simplified NUTS...")
    for i in range(1, n_samples):
        if i % 200 == 0:
            print(f"  Sample {i}/{n_samples}")
        
        x_new, depth, n_eval = nuts_single_iteration(samples[i-1], epsilon)
        samples[i] = x_new
        depths.append(depth)
        n_evals.append(n_eval)
    
    depths = np.array(depths)
    n_evals = np.array(n_evals)
    
    print(f"\nAverage tree depth: {depths.mean():.1f}")
    print(f"Average gradient evaluations per sample: {n_evals.mean():.1f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Samples
    ax = axes[0, 0]
    
    # Contours
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    ax.scatter(samples[200:, 0], samples[200:, 1], 
              alpha=0.2, s=10, color='blue')
    ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('NUTS Samples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Tree depth over iterations
    ax = axes[0, 1]
    ax.plot(depths, alpha=0.7, linewidth=0.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Tree depth', fontsize=12)
    ax.set_title(f'Adaptive Tree Depth\nMean = {depths.mean():.1f}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Histogram of depths
    ax = axes[1, 0]
    ax.hist(depths, bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Tree depth', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Tree Depths',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Gradient evaluations
    ax = axes[1, 1]
    ax.plot(n_evals, alpha=0.7, linewidth=0.5, color='green')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient evaluations', fontsize=12)
    ax.set_title(f'Gradient Evaluations per Sample\nMean = {n_evals.mean():.1f}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'nuts_simple_implementation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: nuts_simple_implementation.png")
    plt.close()
    
    print("\nNUTS automatically adapts trajectory length to local geometry!")


# =============================================================================
# PART 3: DUAL AVERAGING FOR STEP SIZE (ADVANCED)
# =============================================================================

def nuts_dual_averaging():
    """
    Example 3: Dual Averaging for Step Size Adaptation
    ==================================================
    
    Show how NUTS automatically tunes step size during warmup.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Dual Averaging for Step Size Adaptation")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target distribution (challenging!)
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.95],
                      [0.95, 1.0]])  # High correlation
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    # Dual averaging parameters
    delta_target = 0.65  # Target acceptance rate
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    
    # Find reasonable initial step size
    def find_reasonable_epsilon(x0):
        """Find reasonable initial epsilon using a simple heuristic"""
        epsilon = 1.0
        p0 = np.random.randn(len(x0))
        
        # Take one leapfrog step
        p_half = p0 - (epsilon / 2) * grad_U(x0)
        x_new = x0 + epsilon * p_half
        p_new = p_half - (epsilon / 2) * grad_U(x_new)
        
        # Compute change in Hamiltonian
        H0 = U(x0) + 0.5 * np.sum(p0**2)
        H_new = U(x_new) + 0.5 * np.sum(p_new**2)
        
        # Adjust epsilon
        a = 1.0 if (H_new - H0) < np.log(0.5) else -1.0
        
        while True:
            epsilon = epsilon * (2.0 ** a)
            
            p_half = p0 - (epsilon / 2) * grad_U(x0)
            x_new = x0 + epsilon * p_half
            p_new = p_half - (epsilon / 2) * grad_U(x_new)
            
            H_new = U(x_new) + 0.5 * np.sum(p_new**2)
            
            if a == 1.0 and (H_new - H0) > np.log(0.5):
                break
            if a == -1.0 and (H_new - H0) < np.log(0.5):
                break
            
            if epsilon > 1e10 or epsilon < 1e-10:
                break
        
        return epsilon
    
    # Initialize
    x0 = np.array([2.0, 2.0])
    epsilon = find_reasonable_epsilon(x0)
    mu_epsilon = np.log(10 * epsilon)
    epsilon_bar = 1.0
    H_bar = 0.0
    
    print(f"Initial epsilon: {epsilon:.4f}")
    
    # Warmup phase
    n_warmup = 500
    samples_warmup = np.zeros((n_warmup, 2))
    samples_warmup[0] = x0
    
    epsilon_history = [epsilon]
    acceptance_history = []
    
    print("\nWarmup phase (adapting step size)...")
    for m in range(1, n_warmup):
        if m % 100 == 0:
            print(f"  Warmup {m}/{n_warmup}, ε = {epsilon:.4f}")
        
        x = samples_warmup[m-1]
        
        # Simple HMC step to compute acceptance rate
        p = np.random.randn(2)
        H_current = U(x) + 0.5 * np.sum(p**2)
        
        # Leapfrog (simplified, fixed L=10)
        x_new, p_new = x.copy(), p.copy()
        L = 10
        for _ in range(L):
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
            x_new = x_new + epsilon * p_new
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
        
        H_new = U(x_new) + 0.5 * np.sum(p_new**2)
        
        # Accept/reject
        log_alpha = min(0, -H_new + H_current)
        alpha = np.exp(log_alpha)
        
        if np.log(np.random.rand()) < log_alpha:
            samples_warmup[m] = x_new
            accepted = 1
        else:
            samples_warmup[m] = x
            accepted = 0
        
        acceptance_history.append(alpha)
        
        # Dual averaging update
        H_bar = (1 - 1/(m + t0)) * H_bar + \
                (delta_target - alpha) / (m + t0)
        
        epsilon = np.exp(mu_epsilon - np.sqrt(m) / gamma * H_bar)
        
        epsilon_bar = np.exp(m**(-kappa) * np.log(epsilon) + \
                            (1 - m**(-kappa)) * np.log(epsilon_bar))
        
        epsilon_history.append(epsilon)
    
    # Use epsilon_bar for production
    epsilon_final = epsilon_bar
    print(f"\nFinal adapted epsilon: {epsilon_final:.4f}")
    
    # Production samples
    n_samples = 500
    samples = np.zeros((n_samples, 2))
    samples[0] = samples_warmup[-1]
    
    print("\nProduction phase (fixed step size)...")
    for i in range(1, n_samples):
        x = samples[i-1]
        p = np.random.randn(2)
        H_current = U(x) + 0.5 * np.sum(p**2)
        
        x_new, p_new = x.copy(), p.copy()
        L = 10
        for _ in range(L):
            p_new = p_new - (epsilon_final / 2) * grad_U(x_new)
            x_new = x_new + epsilon_final * p_new
            p_new = p_new - (epsilon_final / 2) * grad_U(x_new)
        
        H_new = U(x_new) + 0.5 * np.sum(p_new**2)
        
        if np.log(np.random.rand()) < (-H_new + H_current):
            samples[i] = x_new
        else:
            samples[i] = x
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Step size adaptation
    ax = axes[0, 0]
    ax.plot(epsilon_history, linewidth=1.5)
    ax.axhline(y=epsilon_final, color='red', linestyle='--',
              linewidth=2, label=f'Final ε = {epsilon_final:.4f}')
    ax.set_xlabel('Warmup iteration', fontsize=12)
    ax.set_ylabel('Step size ε', fontsize=12)
    ax.set_title('Dual Averaging: Step Size Adaptation',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Acceptance rate during warmup
    ax = axes[0, 1]
    
    # Moving average
    window = 50
    acceptance_smooth = np.convolve(acceptance_history, 
                                    np.ones(window)/window, mode='valid')
    
    ax.plot(acceptance_smooth, linewidth=1.5)
    ax.axhline(y=delta_target, color='red', linestyle='--',
              linewidth=2, label=f'Target = {delta_target}')
    ax.set_xlabel('Warmup iteration', fontsize=12)
    ax.set_ylabel('Acceptance rate (moving avg)', fontsize=12)
    ax.set_title('Convergence to Target Acceptance Rate',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Final samples
    ax = axes[1, 0]
    
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    ax.scatter(samples[:, 0], samples[:, 1], 
              alpha=0.3, s=15, color='blue')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Production Samples\n(with adapted step size)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Trace plots
    ax = axes[1, 1]
    ax.plot(samples[:, 0], alpha=0.7, linewidth=0.5, label='$x_1$')
    ax.plot(samples[:, 1], alpha=0.7, linewidth=0.5, label='$x_2$')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Trace Plots (production)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'nuts_dual_averaging.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: nuts_dual_averaging.png")
    plt.close()
    
    print("\nDual averaging automatically finds optimal step size!")


# =============================================================================
# PART 4: NUTS VS HMC COMPARISON (ADVANCED)
# =============================================================================

def nuts_vs_hmc_comparison():
    """
    Example 4: Comprehensive Comparison of NUTS vs HMC
    ==================================================
    
    Compare performance on a challenging multimodal distribution.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: NUTS vs HMC Comparison")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Banana-shaped distribution (challenging!)
    def banana_transform(x):
        """Transform to banana shape"""
        y = x.copy()
        y[1] = y[1] - y[0]**2
        return y
    
    def U(x):
        """Potential for banana distribution"""
        y = banana_transform(x)
        return 0.5 * (y[0]**2 + 10 * y[1]**2)
    
    def grad_U(x):
        """Gradient (via chain rule)"""
        y = banana_transform(x)
        grad_y = np.array([y[0], 10 * y[1]])
        
        # Jacobian of transform
        J = np.array([[1.0, 0.0],
                      [-2*x[0], 1.0]])
        
        return J.T @ grad_y
    
    # Fixed-L HMC
    def hmc_fixed_L(x0, n_samples, epsilon, L):
        """HMC with fixed number of leapfrog steps"""
        samples = np.zeros((n_samples, 2))
        samples[0] = x0
        n_accepted = 0
        n_grads = 0
        
        for i in range(1, n_samples):
            x = samples[i-1]
            p = np.random.randn(2)
            H_current = U(x) + 0.5 * np.sum(p**2)
            
            # Leapfrog
            x_new, p_new = x.copy(), p.copy()
            for _ in range(L):
                p_new = p_new - (epsilon / 2) * grad_U(x_new)
                x_new = x_new + epsilon * p_new
                p_new = p_new - (epsilon / 2) * grad_U(x_new)
                n_grads += 2  # Two gradient evaluations per step
            
            H_new = U(x_new) + 0.5 * np.sum(p_new**2)
            
            if np.log(np.random.rand()) < (-H_new + H_current):
                samples[i] = x_new
                n_accepted += 1
            else:
                samples[i] = x
        
        return samples, n_accepted / (n_samples - 1), n_grads / n_samples
    
    # Simplified NUTS
    def nuts_adaptive(x0, n_samples, epsilon, max_depth=8):
        """NUTS with adaptive trajectory length"""
        samples = np.zeros((n_samples, 2))
        samples[0] = x0
        total_depth = 0
        n_grads_total = 0
        
        for i in range(1, n_samples):
            x = samples[i-1]
            p = np.random.randn(2)
            
            # Build tree until U-turn
            x_left, p_left = x.copy(), p.copy()
            x_right, p_right = x.copy(), p.copy()
            
            depth = 0
            n_grads = 0
            
            while depth < max_depth:
                direction = 2 * np.random.randint(2) - 1
                
                if direction == 1:
                    p_right = p_right - (epsilon / 2) * grad_U(x_right)
                    x_right = x_right + epsilon * p_right
                    p_right = p_right - (epsilon / 2) * grad_U(x_right)
                    n_grads += 2
                else:
                    p_left = p_left - (epsilon / 2) * grad_U(x_left)
                    x_left = x_left - epsilon * p_left
                    p_left = p_left - (epsilon / 2) * grad_U(x_left)
                    n_grads += 2
                
                # Check U-turn
                if np.dot(x_right - x_left, p_right) < 0 or \
                   np.dot(x_right - x_left, p_left) < 0:
                    break
                
                depth += 1
            
            # Simple selection (just use one endpoint)
            samples[i] = x_right if direction == 1 else x_left
            total_depth += depth
            n_grads_total += n_grads
        
        return samples, total_depth / n_samples, n_grads_total / n_samples
    
    # Run both methods
    n_samples = 500
    x0 = np.array([0.0, 0.0])
    epsilon = 0.15
    
    print("\nRunning HMC (fixed L=10)...")
    samples_hmc, accept_hmc, grads_hmc = hmc_fixed_L(x0, n_samples, 
                                                      epsilon, L=10)
    
    print("Running NUTS (adaptive)...")
    samples_nuts, avg_depth, grads_nuts = nuts_adaptive(x0, n_samples, epsilon)
    
    print(f"\nResults:")
    print(f"HMC: {accept_hmc:.1%} acceptance, {grads_hmc:.1f} grads/sample")
    print(f"NUTS: avg depth = {avg_depth:.1f}, {grads_nuts:.1f} grads/sample")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create banana contours
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-2, 8, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    # HMC
    ax = axes[0]
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    ax.scatter(samples_hmc[100:, 0], samples_hmc[100:, 1],
              alpha=0.3, s=10, color='blue')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'HMC (Fixed L=10)\nAcceptance: {accept_hmc:.1%}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # NUTS
    ax = axes[1]
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    ax.scatter(samples_nuts[100:, 0], samples_nuts[100:, 1],
              alpha=0.3, s=10, color='green')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'NUTS (Adaptive)\nAvg depth: {avg_depth:.1f}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'nuts_vs_hmc.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: nuts_vs_hmc.png")
    plt.close()
    
    print("\nNUTS adapts to complex geometry automatically!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NO-U-TURN SAMPLER (NUTS) TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial covers advanced adaptive HMC:")
    print("1. Understanding U-turns in HMC")
    print("2. Simplified NUTS implementation")
    print("3. Dual averaging for step size adaptation")
    print("4. NUTS vs HMC comparison")
    print("\n" + "=" * 80 + "\n")
    
    # Run all examples
    visualize_uturn()
    nuts_simple()
    nuts_dual_averaging()
    nuts_vs_hmc_comparison()
    
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - nuts_uturn_visualization.png")
    print("  - nuts_simple_implementation.png")
    print("  - nuts_dual_averaging.png")
    print("  - nuts_vs_hmc.png")
    print("\nKey Takeaways:")
    print("  • NUTS eliminates the need to tune trajectory length L")
    print("  • Detects U-turns automatically to avoid wasted computation")
    print("  • Dual averaging adapts step size during warmup")
    print("  • Default algorithm in Stan and PyMC for good reason!")
    print("  • Robust across wide range of target distributions")
    print("  • Trades implementation complexity for ease of use")
    print("=" * 80 + "\n")
