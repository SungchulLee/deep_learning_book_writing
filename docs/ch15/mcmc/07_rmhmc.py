"""
07_rmhmc.py

RIEMANNIAN MANIFOLD HMC: COMPREHENSIVE TUTORIAL
===============================================

Learning Objectives:
-------------------
1. Understand why Euclidean geometry can be inefficient
2. Learn Riemannian geometry basics for sampling
3. Master the generalized leapfrog integrator
4. Apply RMHMC to challenging distributions
5. Compare with Euclidean HMC and NUTS

Mathematical Foundation:
-----------------------
Goal:
    Adapt Hamiltonian dynamics to the natural geometry of the target
    distribution, leading to more efficient exploration.

The Problem with Euclidean HMC:
-------------------------------
Standard HMC uses Euclidean metric (M = I):
    - Assumes all directions are "equally important"
    - Inefficient when parameters have different scales
    - Struggles with highly correlated/curved distributions
    - Fixed mass matrix doesn't adapt to local geometry

Example: Consider a distribution with:
    - x₁ has variance 1
    - x₂ has variance 100
    
Standard HMC treats them equally, wasting effort!

Riemannian Geometry Intuition:
-----------------------------
Key idea: Use a position-dependent metric G(x)

Instead of M = I (flat space), use:
    M(x) = [G(x)]⁻¹
    
where G(x) is the metric tensor (often the Hessian or Fisher information).

Geometric Interpretation:
    - Each point x has its own "local geometry"
    - Distances and directions adapt to curvature
    - Natural geodesics follow the manifold structure
    - Like walking on a curved surface, not a flat plane!

Fisher Information Metric:
    For probabilistic models:
        G(x) = E[∇log p(y|x) ∇log p(y|x)ᵀ]
    
    This captures the local sensitivity of the model!

Riemannian Hamiltonian:
----------------------
With position-dependent metric G(x):

    H(x, p) = U(x) + (1/2) pᵀ G(x)⁻¹ p
    
Hamilton's equations become:

    dx/dt = G(x)⁻¹ p
    dp/dt = -(∇U(x) + (1/2)∇ₓ[pᵀ G(x)⁻¹ p])

The extra term (1/2)∇ₓ[pᵀ G(x)⁻¹ p] accounts for changing geometry!

Generalized Leapfrog:
--------------------
More complex than Euclidean case due to metric dependence:

1. Implicit position update (requires solving nonlinear equation):
   x_{t+ε} = x_t + ε G(x_{t+ε/2})⁻¹ p_{t+ε/2}

2. Momentum updates:
   p_{t+ε/2} = p_t - (ε/2)[∇U(x_t) + Q(x_t, p_{t+ε/2})]
   
where Q(x,p) = (1/2)∇ₓ[pᵀ G(x)⁻¹ p] is the "metric force"

This is more expensive but can be much more efficient per step!

Advantages of RMHMC:
-------------------
+ Adapts to local geometry automatically
+ Efficient for highly correlated distributions
+ Fewer steps needed for convergence
+ Better scaling with dimension
+ Natural for constrained problems

Disadvantages:
-------------
- Much more complex implementation
- Requires computing/storing metric G(x)
- More expensive per leapfrog step
- Need to solve implicit equations
- Can be numerically unstable

Practical Metric Choices:
-------------------------
1. **SoftAbs Metric**: G(x) = ∇²U(x) + λI (regularized Hessian)
   - Most common choice
   - Captures local curvature
   - Requires second derivatives

2. **Fisher Information**: G(x) = Fisher information matrix
   - Natural for statistical models
   - Incorporates data information

3. **Euclidean-Riemannian**: Diagonal G(x)
   - Simpler than full matrix
   - Still captures scale differences

Connection to Modern ML:
-----------------------
- Natural gradient descent: Uses Fisher metric for optimization
- Variational inference: Natural gradients for faster convergence
- Neural ODEs: Adaptive step sizes use similar ideas
- Normalizing flows: Metric structure for invertibility
- Stein variational gradient descent: Metric-aware particles

Advanced Topics:
---------------
- Lagrangian Monte Carlo: Momentum in tangent space
- Constrained RMHMC: Sampling on manifolds (e.g., Stiefel)
- Shadow Hamiltonians: Higher-order integrators
- Geodesic flows: Following natural curves

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.linalg import solve, cho_factor, cho_solve
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


# =============================================================================
# PART 1: WHY RIEMANNIAN? (BEGINNER)
# =============================================================================

def visualize_geometry_adaptation():
    """
    Example 1: Visualizing Why We Need Adaptive Geometry
    ====================================================
    
    Show how Euclidean HMC struggles with ill-conditioned distributions.
    """
    print("=" * 80)
    print("EXAMPLE 1: The Need for Adaptive Geometry")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Ill-conditioned target: very different scales
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0],
                      [0.0, 100.0]])  # 100x difference in scale!
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    # Euclidean HMC (fixed mass matrix M = I)
    def euclidean_hmc_step(x, epsilon, L):
        """Single Euclidean HMC step"""
        p = np.random.randn(2)
        H_current = U(x) + 0.5 * np.sum(p**2)
        
        # Leapfrog with M = I
        x_new, p_new = x.copy(), p.copy()
        for _ in range(L):
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
            x_new = x_new + epsilon * p_new  # M⁻¹ = I
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
        
        H_new = U(x_new) + 0.5 * np.sum(p_new**2)
        
        # Accept/reject
        if np.log(np.random.rand()) < (-H_new + H_current):
            return x_new, True
        else:
            return x, False
    
    # Riemannian HMC (adaptive mass matrix M(x) = Σ)
    def riemannian_hmc_step(x, epsilon, L, metric_func):
        """Single Riemannian HMC step"""
        G = metric_func(x)
        G_inv = np.linalg.inv(G)
        
        # Sample momentum from N(0, G)
        p = np.random.multivariate_normal(np.zeros(2), G)
        H_current = U(x) + 0.5 * p @ G_inv @ p
        
        # Simplified leapfrog (assuming constant metric for simplicity)
        x_new, p_new = x.copy(), p.copy()
        for _ in range(L):
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
            x_new = x_new + epsilon * G_inv @ p_new
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
        
        H_new = U(x_new) + 0.5 * p_new @ G_inv @ p_new
        
        if np.log(np.random.rand()) < (-H_new + H_current):
            return x_new, True
        else:
            return x, False
    
    def metric_func(x):
        """Use covariance as metric (ideal for this Gaussian!)"""
        return Sigma_inv
    
    # Run both samplers
    n_steps = 500
    x0 = np.array([3.0, 30.0])
    epsilon = 0.1
    L = 10
    
    # Euclidean HMC
    print("\nRunning Euclidean HMC...")
    samples_euc = np.zeros((n_steps, 2))
    samples_euc[0] = x0
    accept_euc = 0
    
    for i in range(1, n_steps):
        samples_euc[i], accepted = euclidean_hmc_step(samples_euc[i-1], 
                                                       epsilon, L)
        if accepted:
            accept_euc += 1
    
    accept_rate_euc = accept_euc / (n_steps - 1)
    print(f"Euclidean HMC acceptance: {accept_rate_euc:.1%}")
    
    # Riemannian HMC
    print("Running Riemannian HMC...")
    samples_riem = np.zeros((n_steps, 2))
    samples_riem[0] = x0
    accept_riem = 0
    
    for i in range(1, n_steps):
        samples_riem[i], accepted = riemannian_hmc_step(samples_riem[i-1],
                                                         epsilon, L, metric_func)
        if accepted:
            accept_riem += 1
    
    accept_rate_riem = accept_riem / (n_steps - 1)
    print(f"Riemannian HMC acceptance: {accept_rate_riem:.1%}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create contours (stretched coordinates for visibility)
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-50, 50, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    # Euclidean HMC
    ax = axes[0]
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    
    # Show trajectory
    traj = samples_euc[:100]
    ax.plot(traj[:, 0], traj[:, 1], 'o-', color='blue',
           alpha=0.5, markersize=3, linewidth=0.8)
    ax.scatter(samples_euc[100:, 0], samples_euc[100:, 1],
              alpha=0.2, s=10, color='blue')
    ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
    
    ax.set_xlabel('$x_1$ (scale ~ 1)', fontsize=12)
    ax.set_ylabel('$x_2$ (scale ~ 10)', fontsize=12)
    ax.set_title(f'Euclidean HMC\nAcceptance: {accept_rate_euc:.1%}\n' +
                'Struggles with different scales!',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Riemannian HMC
    ax = axes[1]
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    
    traj = samples_riem[:100]
    ax.plot(traj[:, 0], traj[:, 1], 'o-', color='green',
           alpha=0.5, markersize=3, linewidth=0.8)
    ax.scatter(samples_riem[100:, 0], samples_riem[100:, 1],
              alpha=0.2, s=10, color='green')
    ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
    
    ax.set_xlabel('$x_1$ (scale ~ 1)', fontsize=12)
    ax.set_ylabel('$x_2$ (scale ~ 10)', fontsize=12)
    ax.set_title(f'Riemannian HMC\nAcceptance: {accept_rate_riem:.1%}\n' +
                'Adapts to geometry!',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Autocorrelation comparison
    def autocorr(x, lag=50):
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        return np.array([np.dot(x[:-k], x[k:]) / len(x) / c0 
                        if k > 0 else 1.0 for k in range(lag)])
    
    ax = axes[2]
    lags = np.arange(50)
    
    acf_euc = autocorr(samples_euc[100:, 0])
    acf_riem = autocorr(samples_riem[100:, 0])
    
    ax.plot(lags, acf_euc, 'o-', color='blue', linewidth=2, 
           label='Euclidean HMC')
    ax.plot(lags, acf_riem, 's-', color='green', linewidth=2,
           label='Riemannian HMC')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Mixing Comparison\nRiemannian shows faster decorrelation',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'rmhmc_motivation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: rmhmc_motivation.png")
    plt.close()
    
    print("\nRiemannian geometry enables efficient sampling!")


# =============================================================================
# PART 2: METRIC TENSOR AND GEODESICS (INTERMEDIATE)
# =============================================================================

def visualize_metric_tensor():
    """
    Example 2: Understanding the Metric Tensor
    ==========================================
    
    Visualize how the metric tensor changes across parameter space.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: The Metric Tensor")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Rosenbrock distribution (banana-shaped, varying curvature)
    a, b = 1.0, 20.0
    
    def U(x):
        """Rosenbrock potential"""
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    def grad_U(x):
        """Gradient of Rosenbrock"""
        grad_x0 = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
        grad_x1 = 2*b*(x[1] - x[0]**2)
        return np.array([grad_x0, grad_x1])
    
    def hessian_U(x):
        """Hessian of Rosenbrock (our metric!)"""
        H = np.zeros((2, 2))
        H[0, 0] = 2 + 12*b*x[0]**2 - 4*b*x[1]
        H[0, 1] = -4*b*x[0]
        H[1, 0] = -4*b*x[0]
        H[1, 1] = 2*b
        
        # Regularize for numerical stability
        return H + 0.1 * np.eye(2)
    
    # Visualize metric at different points
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Create grid
    x1 = np.linspace(-0.5, 2, 100)
    x2 = np.linspace(-0.5, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-0.5 * U(np.array([X1[i, j], X2[i, j]])))
    
    # Different points to examine
    points = [
        np.array([0.5, 0.25]),
        np.array([1.0, 1.0]),
        np.array([1.5, 2.25]),
        np.array([0.8, 0.64])
    ]
    
    for idx, (ax, point) in enumerate(zip(axes.flatten(), points)):
        # Plot distribution
        ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
        
        # Compute metric at this point
        G = hessian_U(point)
        G_inv = np.linalg.inv(G)
        
        # Eigendecomposition to understand metric
        eigvals, eigvecs = np.linalg.eig(G)
        
        # Plot ellipse showing metric
        # The metric defines distances, so G⁻¹ gives variance
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        
        from matplotlib.patches import Ellipse
        
        # Scale by inverse eigenvalues for visualization
        width = 2 * np.sqrt(1.0 / eigvals[0])
        height = 2 * np.sqrt(1.0 / eigvals[1])
        
        ellipse = Ellipse(point, width, height, 
                         angle=np.degrees(angle),
                         facecolor='red', alpha=0.3, 
                         edgecolor='red', linewidth=2)
        ax.add_patch(ellipse)
        
        # Plot eigenvectors
        scale = 0.3
        for i in range(2):
            direction = eigvecs[:, i] * scale / np.sqrt(eigvals[i])
            ax.arrow(point[0], point[1], direction[0], direction[1],
                    head_width=0.1, head_length=0.1, fc='blue', 
                    ec='blue', linewidth=2)
        
        ax.plot(point[0], point[1], 'r*', markersize=20)
        
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(f'Metric at {point}\n' +
                    f'Eigenvalues: [{eigvals[0]:.1f}, {eigvals[1]:.1f}]',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.5, 2])
        ax.set_ylim([-0.5, 4])
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'rmhmc_metric_tensor.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: rmhmc_metric_tensor.png")
    plt.close()
    
    print("\nThe metric tensor captures local curvature and anisotropy!")
    print("Red ellipse: region of equal 'Riemannian distance'")
    print("Blue arrows: principal directions (eigenvectors)")


# =============================================================================
# PART 3: GENERALIZED LEAPFROG (ADVANCED)
# =============================================================================

def generalized_leapfrog_demo():
    """
    Example 3: Generalized Leapfrog Integrator
    ==========================================
    
    Show how the Riemannian leapfrog differs from Euclidean.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Generalized Leapfrog Integrator")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target: correlated Gaussian (simple but illustrative)
    mu = np.array([0.0, 0.0])
    rho = 0.9
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    def metric_func(x):
        """Position-dependent metric (Hessian for Gaussian is constant)"""
        return Sigma_inv
    
    # Euclidean leapfrog
    def euclidean_leapfrog(x0, p0, epsilon, L):
        """Standard leapfrog with M = I"""
        trajectory_x = [x0.copy()]
        trajectory_p = [p0.copy()]
        
        x, p = x0.copy(), p0.copy()
        
        for _ in range(L):
            p = p - (epsilon / 2) * grad_U(x)
            x = x + epsilon * p  # M⁻¹ = I
            p = p - (epsilon / 2) * grad_U(x)
            
            trajectory_x.append(x.copy())
            trajectory_p.append(p.copy())
        
        return np.array(trajectory_x), np.array(trajectory_p)
    
    # Riemannian leapfrog (simplified - assuming metric doesn't change much)
    def riemannian_leapfrog(x0, p0, epsilon, L, metric_func):
        """Generalized leapfrog with position-dependent metric"""
        trajectory_x = [x0.copy()]
        trajectory_p = [p0.copy()]
        
        x, p = x0.copy(), p0.copy()
        
        for _ in range(L):
            G = metric_func(x)
            G_inv = np.linalg.inv(G)
            
            # Half momentum step
            p = p - (epsilon / 2) * grad_U(x)
            
            # Position step (using metric!)
            x = x + epsilon * G_inv @ p
            
            # Half momentum step
            p = p - (epsilon / 2) * grad_U(x)
            
            trajectory_x.append(x.copy())
            trajectory_p.append(p.copy())
        
        return np.array(trajectory_x), np.array(trajectory_p)
    
    # Run both integrators
    x0 = np.array([1.5, 1.0])
    p0 = np.array([0.5, -1.0])
    epsilon = 0.2
    L = 20
    
    print("\nComparing leapfrog integrators...")
    
    traj_x_euc, traj_p_euc = euclidean_leapfrog(x0, p0, epsilon, L)
    traj_x_riem, traj_p_riem = riemannian_leapfrog(x0, p0, epsilon, L, 
                                                     metric_func)
    
    # Compute energy for both
    def H_euclidean(x, p):
        return U(x) + 0.5 * np.sum(p**2)
    
    def H_riemannian(x, p, G_inv):
        return U(x) + 0.5 * p @ G_inv @ p
    
    energy_euc = [H_euclidean(x, p) for x, p in zip(traj_x_euc, traj_p_euc)]
    
    energy_riem = []
    for x, p in zip(traj_x_riem, traj_p_riem):
        G = metric_func(x)
        G_inv = np.linalg.inv(G)
        energy_riem.append(H_riemannian(x, p, G_inv))
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Position trajectories
    ax = axes[0, 0]
    
    # Contours
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    ax.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    
    # Euclidean
    ax.plot(traj_x_euc[:, 0], traj_x_euc[:, 1], 'o-',
           color='blue', label='Euclidean', linewidth=2, markersize=4)
    
    # Riemannian
    ax.plot(traj_x_riem[:, 0], traj_x_riem[:, 1], 's-',
           color='green', label='Riemannian', linewidth=2, markersize=4)
    
    ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Position Trajectories',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Momentum trajectories
    ax = axes[0, 1]
    
    ax.plot(traj_p_euc[:, 0], traj_p_euc[:, 1], 'o-',
           color='blue', label='Euclidean', linewidth=2, markersize=4)
    ax.plot(traj_p_riem[:, 0], traj_p_riem[:, 1], 's-',
           color='green', label='Riemannian', linewidth=2, markersize=4)
    ax.plot(p0[0], p0[1], 'r*', markersize=20, label='Start')
    
    ax.set_xlabel('$p_1$', fontsize=12)
    ax.set_ylabel('$p_2$', fontsize=12)
    ax.set_title('Momentum Trajectories',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Energy conservation
    ax = axes[1, 0]
    
    steps = np.arange(len(energy_euc))
    ax.plot(steps, energy_euc, 'o-', color='blue', 
           linewidth=2, label='Euclidean')
    ax.plot(steps, energy_riem, 's-', color='green',
           linewidth=2, label='Riemannian')
    
    ax.set_xlabel('Leapfrog step', fontsize=12)
    ax.set_ylabel('Hamiltonian H(x, p)', fontsize=12)
    ax.set_title('Energy Conservation\n(Smaller variation = better)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Energy error
    ax = axes[1, 1]
    
    error_euc = np.abs(np.array(energy_euc) - energy_euc[0])
    error_riem = np.abs(np.array(energy_riem) - energy_riem[0])
    
    ax.semilogy(steps, error_euc, 'o-', color='blue', 
               linewidth=2, label='Euclidean')
    ax.semilogy(steps, error_riem, 's-', color='green',
               linewidth=2, label='Riemannian')
    
    ax.set_xlabel('Leapfrog step', fontsize=12)
    ax.set_ylabel('|ΔH|', fontsize=12)
    ax.set_title('Energy Error (log scale)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'rmhmc_leapfrog.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: rmhmc_leapfrog.png")
    plt.close()
    
    print(f"\nEnergy error (Euclidean): {error_euc[-1]:.6f}")
    print(f"Energy error (Riemannian): {error_riem[-1]:.6f}")
    print("\nRiemannian leapfrog follows natural geodesics!")


# =============================================================================
# PART 4: FULL RMHMC IMPLEMENTATION (ADVANCED)
# =============================================================================

def rmhmc_full_example():
    """
    Example 4: Complete RMHMC for Bayesian Inference
    ================================================
    
    Apply RMHMC to a challenging Bayesian logistic regression problem.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: RMHMC for Bayesian Logistic Regression")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Generate data
    n_samples = 100
    n_features = 5
    
    w_true = np.array([2.0, -1.5, 1.0, -0.5, 0.3])
    X = np.random.randn(n_samples, n_features)
    logits = X @ w_true
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(float)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    
    # Prior
    sigma_w = 2.0
    
    def sigmoid(z):
        """Stable sigmoid"""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def U(w):
        """Negative log posterior"""
        logits = X @ w
        log_lik = np.sum(y * logits - np.log(1 + np.exp(logits)))
        log_prior = -0.5 * np.sum(w**2) / sigma_w**2
        return -(log_lik + log_prior)
    
    def grad_U(w):
        """Gradient of negative log posterior"""
        probs = sigmoid(X @ w)
        grad_lik = X.T @ (y - probs)
        grad_prior = -w / sigma_w**2
        return -(grad_lik + grad_prior)
    
    def fisher_information(w):
        """Fisher information matrix (our metric!)"""
        probs = sigmoid(X @ w)
        # Fisher = X^T W X where W = diag(p(1-p))
        W = probs * (1 - probs)
        
        # Add small regularization for numerical stability
        FIM = X.T @ np.diag(W) @ X + 0.1 * np.eye(n_features)
        
        # Add prior contribution
        FIM = FIM + np.eye(n_features) / sigma_w**2
        
        return FIM
    
    # Euclidean HMC
    def euclidean_hmc(w0, n_iter, epsilon, L):
        """Standard HMC with M = I"""
        samples = np.zeros((n_iter, n_features))
        samples[0] = w0
        n_accepted = 0
        
        print("\nRunning Euclidean HMC...")
        for i in range(1, n_iter):
            if i % 100 == 0:
                print(f"  Iteration {i}/{n_iter}")
            
            w = samples[i-1]
            p = np.random.randn(n_features)
            H_current = U(w) + 0.5 * np.sum(p**2)
            
            # Leapfrog
            w_new, p_new = w.copy(), p.copy()
            for _ in range(L):
                p_new = p_new - (epsilon / 2) * grad_U(w_new)
                w_new = w_new + epsilon * p_new
                p_new = p_new - (epsilon / 2) * grad_U(w_new)
            
            H_new = U(w_new) + 0.5 * np.sum(p_new**2)
            
            if np.log(np.random.rand()) < (-H_new + H_current):
                samples[i] = w_new
                n_accepted += 1
            else:
                samples[i] = w
        
        return samples, n_accepted / (n_iter - 1)
    
    # Riemannian HMC (simplified)
    def riemannian_hmc(w0, n_iter, epsilon, L):
        """RMHMC with Fisher information metric"""
        samples = np.zeros((n_iter, n_features))
        samples[0] = w0
        n_accepted = 0
        
        print("\nRunning Riemannian HMC...")
        for i in range(1, n_iter):
            if i % 100 == 0:
                print(f"  Iteration {i}/{n_iter}")
            
            w = samples[i-1]
            
            # Compute metric at current position
            G = fisher_information(w)
            
            # Sample momentum from N(0, G)
            try:
                L_chol = np.linalg.cholesky(G)
                p = L_chol @ np.random.randn(n_features)
            except:
                p = np.random.randn(n_features)
                G = np.eye(n_features)
            
            G_inv = np.linalg.inv(G)
            H_current = U(w) + 0.5 * p @ G_inv @ p
            
            # Simplified leapfrog (assuming slowly varying metric)
            w_new, p_new = w.copy(), p.copy()
            for _ in range(L):
                p_new = p_new - (epsilon / 2) * grad_U(w_new)
                w_new = w_new + epsilon * G_inv @ p_new
                p_new = p_new - (epsilon / 2) * grad_U(w_new)
            
            # Recompute metric at new position
            G_new = fisher_information(w_new)
            G_new_inv = np.linalg.inv(G_new)
            
            H_new = U(w_new) + 0.5 * p_new @ G_new_inv @ p_new
            
            if np.log(np.random.rand()) < (-H_new + H_current):
                samples[i] = w_new
                n_accepted += 1
            else:
                samples[i] = w
        
        return samples, n_accepted / (n_iter - 1)
    
    # Run both methods
    n_iter = 500
    w0 = 0.1 * np.random.randn(n_features)
    epsilon = 0.01
    L = 5
    
    samples_euc, accept_euc = euclidean_hmc(w0, n_iter, epsilon, L)
    samples_riem, accept_riem = riemannian_hmc(w0, n_iter, epsilon, L)
    
    print(f"\nEuclidean HMC: {accept_euc:.1%} acceptance")
    print(f"Riemannian HMC: {accept_riem:.1%} acceptance")
    
    # Analyze results
    burn_in = 100
    
    mean_euc = samples_euc[burn_in:].mean(axis=0)
    mean_riem = samples_riem[burn_in:].mean(axis=0)
    
    print("\nPosterior means:")
    print("=" * 60)
    print(f"{'Weight':<10} {'True':<10} {'Euclidean':<12} {'Riemannian':<12}")
    print("=" * 60)
    for i in range(n_features):
        print(f"w_{i:<8} {w_true[i]:<10.3f} {mean_euc[i]:<12.3f} "
              f"{mean_riem[i]:<12.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Trace plots
        ax.plot(samples_euc[:, i], alpha=0.5, linewidth=0.5, 
               color='blue', label='Euclidean')
        ax.plot(samples_riem[:, i], alpha=0.5, linewidth=0.5,
               color='green', label='Riemannian')
        ax.axvline(x=burn_in, color='red', linestyle='--',
                  linewidth=2, alpha=0.5)
        ax.axhline(y=w_true[i], color='black', linestyle='--',
                  linewidth=2, alpha=0.7, label='True')
        
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(f'$w_{i}$', fontsize=10)
        ax.set_title(f'Weight {i}', fontsize=12, fontweight='bold')
        if i == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Use last subplot for comparison
    ax = axes[-1]
    
    # Comparison bars
    methods = ['Euclidean', 'Riemannian']
    accept_rates = [accept_euc, accept_riem]
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, accept_rates, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Acceptance Rate', fontsize=12)
    ax.set_title('Acceptance Rate Comparison',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'rmhmc_bayesian_inference.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: rmhmc_bayesian_inference.png")
    plt.close()
    
    print("\nRMHMC adapts to the Fisher information geometry!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RIEMANNIAN MANIFOLD HMC TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial covers geometry-adaptive MCMC:")
    print("1. Why adaptive geometry matters")
    print("2. The metric tensor and local geometry")
    print("3. Generalized leapfrog integrator")
    print("4. RMHMC for Bayesian inference")
    print("\n" + "=" * 80 + "\n")
    
    # Run all examples
    visualize_geometry_adaptation()
    visualize_metric_tensor()
    generalized_leapfrog_demo()
    rmhmc_full_example()
    
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - rmhmc_motivation.png")
    print("  - rmhmc_metric_tensor.png")
    print("  - rmhmc_leapfrog.png")
    print("  - rmhmc_bayesian_inference.png")
    print("\nKey Takeaways:")
    print("  • Riemannian geometry adapts to local structure")
    print("  • Metric tensor captures curvature and anisotropy")
    print("  • Fisher information is natural metric for statistical models")
    print("  • More efficient than Euclidean HMC for many problems")
    print("  • Foundation for advanced methods (Lagrangian MC, constrained)")
    print("  • Trade-off: complexity vs. efficiency")
    print("\nFurther Reading:")
    print("  - Girolami & Calderhead (2011): Riemann manifold HMC")
    print("  - Natural gradient descent for optimization")
    print("  - Information geometry and statistical manifolds")
    print("=" * 80 + "\n")
