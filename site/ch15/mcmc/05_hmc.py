"""
05_hmc.py

HAMILTONIAN MONTE CARLO: COMPREHENSIVE TUTORIAL
===============================================

Learning Objectives:
-------------------
1. Understand Hamiltonian dynamics and its use in MCMC
2. Learn the leapfrog integrator for simulation
3. Master tuning: step size and number of leapfrog steps
4. Compare HMC vs MALA vs Random Walk
5. Apply HMC to real inference problems

Mathematical Foundation:
-----------------------
Goal:
    Use physics-inspired dynamics to make distant proposals that maintain high
    acceptance rates, enabling efficient exploration of high-dimensional spaces.

Hamiltonian Dynamics:
--------------------
Introduce auxiliary momentum variables p and define Hamiltonian:
    
    H(x, p) = U(x) + K(p)
    
where:
    - U(x) = -log p(x) is potential energy (negative log density)
    - K(p) = p^T M^{-1} p / 2 is kinetic energy (M is mass matrix)
    - H is total energy (conserved under continuous dynamics!)

Hamilton's Equations:
    dx/dt = ∂H/∂p = M^{-1}p
    dp/dt = -∂H/∂x = -∇U(x) = ∇log p(x)

These equations preserve H(x,p) and volume in phase space!

Key Insight:
    Hamiltonian dynamics naturally explores level sets of p(x) without
    random walk behavior. Proposals are DISTANT but have high acceptance!

Leapfrog Integration:
--------------------
Discretize Hamilton's equations with step size ε:

    p_{t+ε/2} = p_t + (ε/2) ∇log p(x_t)
    x_{t+ε} = x_t + ε M^{-1} p_{t+ε/2}
    p_{t+ε} = p_{t+ε/2} + (ε/2) ∇log p(x_{t+ε})

Properties:
    - Reversible: can run backwards
    - Volume-preserving: Jacobian determinant = 1
    - Symplectic: preserves phase space structure
    - Nearly preserves H (error = O(ε²))

HMC Algorithm:
-------------
1. Sample momentum: p ~ N(0, M)
2. Run L leapfrog steps with step size ε
3. Accept/reject with Metropolis:
   α = min(1, exp(-H(x', p') + H(x, p)))
4. Return x' (discard momentum p')

Why It Works:
    - Proposals x' are far from x (L leapfrog steps)
    - Energy approximately conserved → high acceptance
    - Momentum provides random direction
    - Gradient guides toward high probability regions

Advantages:
----------
+ Much faster mixing than random walk (especially high-D)
+ Intelligent exploration using gradient information
+ Fewer correlated samples
+ Scales well to high dimensions

Disadvantages:
-------------
- Requires gradient computation
- Needs tuning (ε and L)
- Can be sensitive to step size
- More complex than MALA

Optimal Tuning:
--------------
- Step size ε: 
  * Too large → energy errors, low acceptance
  * Too small → slow exploration
  * Target: 60-90% acceptance
  
- Number of steps L:
  * Too few → doesn't move far from starting point
  * Too many → U-turns waste computation
  * Heuristic: L ~ trajectory length / ε
  
- Mass matrix M:
  * M = I (simple, standard)
  * M = diag(σ²) (adapt to scales)
  * M = Σ (full adaptation, expensive)

Connection to Physics:
---------------------
- Particle in potential well U(x) = -log p(x)
- Momentum gives particle "inertia"
- Gradient creates "force" on particle
- Particle follows Newton's laws!
- Natural trajectories through probability landscape

Connection to Modern ML:
-----------------------
- Neural network training: Momentum-based optimizers (SGD + momentum)
- Normalizing flows: Continuous normalizing flows use ODEs
- Variational inference: Stein variational gradient descent
- Neural ODEs: Similar integration schemes

Advanced Extensions:
-------------------
- No-U-Turn Sampler (NUTS): Automatic L selection
- Riemannian HMC: Curved geometry
- Softabs HMC: Handling constraints
- Stochastic gradient HMC: Mini-batch gradients

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, gamma as gamma_dist
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


# =============================================================================
# PART 1: HMC FOR 1D DISTRIBUTION (BEGINNER)
# =============================================================================

def hmc_1d_gaussian():
    """
    Example 1: HMC for 1D Gaussian
    ==============================
    
    Target: p(x) ∝ exp(-x²/2)  (standard normal)
    
    This simple example shows how HMC makes distant proposals
    with high acceptance rates.
    
    Compare to:
    - Random walk: small steps, high acceptance
    - MALA: gradient-guided small steps
    - HMC: gradient-guided LARGE steps!
    """
    print("=" * 80)
    print("EXAMPLE 1: HMC for 1D Gaussian")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target distribution
    def U(x):
        """Potential energy: U(x) = -log p(x)"""
        return 0.5 * x**2
    
    def grad_U(x):
        """Gradient of potential: ∇U(x)"""
        return x
    
    def K(p, M_inv=1.0):
        """Kinetic energy: K(p) = p^T M^{-1} p / 2"""
        return 0.5 * M_inv * p**2
    
    def leapfrog(x, p, epsilon, L, grad_U_func, M_inv=1.0):
        """
        Leapfrog integrator for Hamiltonian dynamics
        
        Args:
            x: position
            p: momentum
            epsilon: step size
            L: number of steps
            grad_U_func: gradient of potential
            M_inv: inverse mass (default 1.0)
            
        Returns:
            x_new, p_new: new position and momentum
        """
        # Make a copy
        x_new = np.copy(x)
        p_new = np.copy(p)
        
        # Half step for momentum
        p_new = p_new - (epsilon / 2) * grad_U_func(x_new)
        
        # L-1 full steps
        for i in range(L):
            # Full step for position
            x_new = x_new + epsilon * M_inv * p_new
            
            # Full step for momentum (except at end)
            if i < L - 1:
                p_new = p_new - epsilon * grad_U_func(x_new)
        
        # Half step for momentum at end
        p_new = p_new - (epsilon / 2) * grad_U_func(x_new)
        
        # Negate momentum for reversibility
        p_new = -p_new
        
        return x_new, p_new
    
    # HMC sampler
    def hmc_sampler(x0, n_steps, epsilon, L):
        """
        Hamiltonian Monte Carlo sampler
        
        Args:
            x0: initial position
            n_steps: number of samples
            epsilon: leapfrog step size
            L: number of leapfrog steps
            
        Returns:
            samples, acceptance_rate
        """
        samples = np.zeros(n_steps)
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            
            # Sample momentum
            p = np.random.randn()
            
            # Current Hamiltonian
            H_current = U(x) + K(p)
            
            # Leapfrog integration
            x_prop, p_prop = leapfrog(x, p, epsilon, L, grad_U)
            
            # Proposed Hamiltonian
            H_prop = U(x_prop) + K(p_prop)
            
            # Accept or reject
            log_alpha = -H_prop + H_current  # Note: already negated
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
        
        acceptance_rate = n_accepted / (n_steps - 1)
        return samples, acceptance_rate
    
    # Run HMC with different parameters
    n_steps = 2000
    x0 = 3.0
    
    configs = [
        {'epsilon': 0.1, 'L': 10, 'label': 'ε=0.1, L=10'},
        {'epsilon': 0.2, 'L': 20, 'label': 'ε=0.2, L=20'},
        {'epsilon': 0.5, 'L': 10, 'label': 'ε=0.5, L=10'},
        {'epsilon': 0.3, 'L': 30, 'label': 'ε=0.3, L=30'},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        samples, accept_rate = hmc_sampler(
            x0, n_steps, config['epsilon'], config['L']
        )
        
        ax = axes[idx]
        
        # Plot trace
        ax.plot(samples, alpha=0.7, linewidth=0.5, color='blue')
        ax.axhline(y=0, color='red', linestyle='--', 
                   linewidth=2, label='True mean')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('x', fontsize=12)
        ax.set_title(f'{config["label"]}\n' +
                    f'Acceptance: {accept_rate:.1%}, ' +
                    f'Mean: {samples[500:].mean():.3f}, ' +
                    f'Std: {samples[500:].std():.3f}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'hmc_1d_gaussian.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: hmc_1d_gaussian.png")
    plt.close()
    
    print("\nKey observations:")
    print("- HMC makes large jumps (see trajectory length = ε × L)")
    print("- High acceptance rates despite distant proposals!")
    print("- Energy conservation enables efficient exploration")


# =============================================================================
# PART 2: HMC VS MALA VS RWM (INTERMEDIATE)
# =============================================================================

def compare_samplers_2d():
    """
    Example 2: Comparing HMC, MALA, and Random Walk Metropolis
    ==========================================================
    
    Target: 2D Gaussian with high correlation
    
    This shows HMC's superiority for correlated distributions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: HMC vs MALA vs Random Walk - 2D Correlated Gaussian")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target: highly correlated 2D Gaussian
    mu = np.array([0.0, 0.0])
    rho = 0.95  # High correlation!
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    target = multivariate_normal(mu, Sigma)
    
    def U(x):
        """Potential energy"""
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        """Gradient of potential"""
        return Sigma_inv @ (x - mu)
    
    # HMC sampler
    def hmc_2d(x0, n_steps, epsilon, L):
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            
            # Sample momentum
            p = np.random.randn(2)
            H_current = U(x) + 0.5 * np.sum(p**2)
            
            # Leapfrog
            x_new, p_new = x.copy(), p.copy()
            
            # Half step for momentum
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
            
            # L full steps
            for i in range(L):
                x_new = x_new + epsilon * p_new
                if i < L - 1:
                    p_new = p_new - epsilon * grad_U(x_new)
            
            # Half step for momentum
            p_new = p_new - (epsilon / 2) * grad_U(x_new)
            p_new = -p_new
            
            # Accept/reject
            H_prop = U(x_new) + 0.5 * np.sum(p_new**2)
            log_alpha = -H_prop + H_current
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_new
                n_accepted += 1
            else:
                samples[t] = x
        
        return samples, n_accepted / (n_steps - 1)
    
    # MALA sampler
    def mala_2d(x0, n_steps, epsilon):
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            
            # Propose
            grad = -grad_U(x)
            x_prop = x + (epsilon / 2) * grad + np.sqrt(epsilon) * np.random.randn(2)
            
            # Accept/reject (simplified for Gaussian)
            mean_forward = x + (epsilon / 2) * (-grad_U(x))
            mean_backward = x_prop + (epsilon / 2) * (-grad_U(x_prop))
            
            log_q_forward = -0.5 * np.sum((x_prop - mean_forward)**2) / epsilon
            log_q_backward = -0.5 * np.sum((x - mean_backward)**2) / epsilon
            
            log_alpha = (-U(x_prop) + log_q_backward) - (-U(x) + log_q_forward)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
        
        return samples, n_accepted / (n_steps - 1)
    
    # Random walk Metropolis
    def rwm_2d(x0, n_steps, sigma):
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            x_prop = x + sigma * np.random.randn(2)
            
            log_alpha = -U(x_prop) + U(x)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
        
        return samples, n_accepted / (n_steps - 1)
    
    # Run all samplers
    n_steps = 2000
    x0 = np.array([2.0, 2.0])
    
    samples_hmc, accept_hmc = hmc_2d(x0, n_steps, epsilon=0.2, L=20)
    samples_mala, accept_mala = mala_2d(x0, n_steps, epsilon=0.3)
    samples_rwm, accept_rwm = rwm_2d(x0, n_steps, sigma=0.5)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create contours
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    Z = target.pdf(pos)
    
    methods = [
        ('Random Walk Metropolis', samples_rwm, accept_rwm, 'blue'),
        ('MALA', samples_mala, accept_mala, 'orange'),
        ('HMC', samples_hmc, accept_hmc, 'green')
    ]
    
    for idx, (name, samples, accept, color) in enumerate(methods):
        ax = axes[idx]
        
        # Contours
        ax.contour(X1, X2, Z, levels=10, colors='gray', alpha=0.3)
        
        # Trajectory
        burn_in = 100
        traj = samples[burn_in:300]
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, 
                alpha=0.5, markersize=3, linewidth=0.8)
        
        # All samples
        ax.scatter(samples[500:, 0], samples[500:, 1], 
                  alpha=0.1, s=5, color=color)
        
        ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'{name}\nAcceptance: {accept:.1%}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'hmc_comparison_2d.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: hmc_comparison_2d.png")
    plt.close()
    
    # Autocorrelation comparison
    def autocorr(x, lag=50):
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        return np.array([np.dot(x[:-k], x[k:]) / len(x) / c0 
                        if k > 0 else 1.0 for k in range(lag)])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    lags = np.arange(50)
    
    for idx, (name, samples, _, color) in enumerate(methods):
        acf = autocorr(samples[500:, 0])
        axes[idx].plot(lags, acf, 'o-', color=color, linewidth=2)
        axes[idx].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[idx].set_xlabel('Lag', fontsize=12)
        axes[idx].set_ylabel('ACF', fontsize=12)
        axes[idx].set_title(f'{name}\nAutocorrelation',
                           fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'hmc_autocorrelation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: hmc_autocorrelation.png")
    plt.close()
    
    print(f"\nComparison:")
    print(f"Random Walk: {accept_rwm:.1%} acceptance")
    print(f"MALA:        {accept_mala:.1%} acceptance")
    print(f"HMC:         {accept_hmc:.1%} acceptance")
    print("\nHMC shows fastest decorrelation despite high correlation!")


# =============================================================================
# PART 3: HMC FOR BAYESIAN NEURAL NETWORK (ADVANCED)
# =============================================================================

def hmc_bayesian_nn():
    """
    Example 3: HMC for Bayesian Neural Network
    ==========================================
    
    Problem: Regression with uncertainty using a simple neural network
    
    Model:
        y = NN(x; w) + ε, ε ~ N(0, σ²)
        w ~ N(0, σ_w² I)
    
    This is a high-dimensional inference problem where HMC excels!
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: HMC for Bayesian Neural Network")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Generate synthetic data
    def true_function(x):
        """True underlying function"""
        return np.sin(3 * x) + 0.3 * x**2
    
    n_data = 50
    X_train = np.random.uniform(-2, 2, n_data)
    y_train = true_function(X_train) + 0.3 * np.random.randn(n_data)
    
    print(f"Generated {n_data} training points")
    
    # Simple neural network
    class SimpleNN:
        """Simple 1-hidden-layer neural network"""

        def __init__(self, n_hidden=10):
            self.n_hidden = n_hidden
            # Weight dimensions: (1->n_hidden) + (n_hidden->1)
            # W1: n_hidden, b1: n_hidden, W2: n_hidden, b2: 1  => total = 3*n_hidden + 1
            self.n_weights = (1 + 1) * n_hidden + (n_hidden + 1)

        def forward(self, x, weights):
            """Forward pass: x shape (n_data,)"""
            # Unpack weights
            idx = 0
            W1 = weights[idx:idx + self.n_hidden].reshape(-1, 1)   # (n_hidden, 1)
            idx += self.n_hidden
            b1 = weights[idx:idx + self.n_hidden]                  # (n_hidden,)
            idx += self.n_hidden
            W2 = weights[idx:idx + self.n_hidden]                  # (n_hidden,)
            idx += self.n_hidden
            b2 = weights[idx]                                      # scalar

            # Shapes: x -> (1, n_data), z1 -> (n_hidden, n_data)
            x = x.reshape(1, -1)                                   # (1, n_data)
            z1 = W1 @ x + b1.reshape(-1, 1)                        # (n_hidden, n_data)
            h = np.tanh(z1)                                        # (n_hidden, n_data)
            y = W2 @ h + b2                                        # (n_data,)
            return y.flatten()

        def gradient(self, x, y, weights, sigma_y, sigma_w):
            """Gradient of log posterior wrt weights"""
            # Unpack weights
            idx = 0
            W1 = weights[idx:idx + self.n_hidden].reshape(-1, 1)   # (n_hidden, 1)
            idx += self.n_hidden
            b1 = weights[idx:idx + self.n_hidden]                  # (n_hidden,)
            idx += self.n_hidden
            W2 = weights[idx:idx + self.n_hidden]                  # (n_hidden,)
            idx += self.n_hidden
            b2 = weights[idx]                                      # scalar

            # Forward with intermediates
            x = x.reshape(1, -1)                                   # (1, n_data)
            z1 = W1 @ x + b1.reshape(-1, 1)                        # (n_hidden, n_data)
            h = np.tanh(z1)                                        # (n_hidden, n_data)
            y_pred = (W2 @ h + b2).flatten()                       # (n_data,)

            # Residuals
            residual = y - y_pred                                  # (n_data,)

            # ---- Gradients of log-likelihood ----
            # Output layer
            grad_b2 = -np.sum(residual) / sigma_y**2               # scalar
            grad_W2 = -h @ residual.reshape(-1, 1) / sigma_y**2    # (n_hidden, 1)
            grad_W2 = grad_W2.flatten()                            # (n_hidden,)

            # Hidden layer
            # delta: derivative wrt pre-activation z1
            delta = -W2.reshape(-1, 1) * residual.reshape(1, -1) / sigma_y**2  # (n_hidden, n_data)
            delta = delta * (1 - h**2)                             # tanh'(z1)

            grad_W1 = (delta @ x.T).flatten()                      # (n_hidden,)
            grad_b1 = delta.sum(axis=1)                            # (n_hidden,)

            # Stack all grads
            grad_weights = np.concatenate([grad_W1, grad_b1, grad_W2, [grad_b2]])

            # ---- Add log-prior gradient (Gaussian prior) ----
            # log p(w) ∝ - ||w||^2 / (2 sigma_w^2) ⇒ ∂/∂w log p(w) = -w / sigma_w^2
            grad_weights = grad_weights - weights / sigma_w**2

            return grad_weights

    
    # Initialize network
    nn = SimpleNN(n_hidden=10)
    print(f"Neural network: {nn.n_weights} parameters")
    
    # Hyperparameters
    sigma_y = 0.3  # Noise std
    sigma_w = 1.0  # Prior std
    
    def U(weights):
        """Potential energy (negative log posterior)"""
        # Log likelihood
        y_pred = nn.forward(X_train, weights)
        log_lik = -0.5 * np.sum((y_train - y_pred)**2) / sigma_y**2
        
        # Log prior
        log_prior = -0.5 * np.sum(weights**2) / sigma_w**2
        
        return -(log_lik + log_prior)
    
    def grad_U(weights):
        """Gradient of potential"""
        return -nn.gradient(X_train, y_train, weights, sigma_y, sigma_w)
    
    # HMC sampler
    def hmc_nn(w0, n_steps, epsilon, L):
        samples = np.zeros((n_steps, len(w0)))
        samples[0] = w0
        n_accepted = 0
        
        print("\nRunning HMC...")
        for t in range(1, n_steps):
            if t % 200 == 0:
                print(f"  Step {t}/{n_steps}")
            
            w = samples[t-1]
            
            # Sample momentum
            p = np.random.randn(len(w))
            H_current = U(w) + 0.5 * np.sum(p**2)
            
            # Leapfrog
            w_new, p_new = w.copy(), p.copy()
            
            p_new = p_new - (epsilon / 2) * grad_U(w_new)
            
            for i in range(L):
                w_new = w_new + epsilon * p_new
                if i < L - 1:
                    p_new = p_new - epsilon * grad_U(w_new)
            
            p_new = p_new - (epsilon / 2) * grad_U(w_new)
            p_new = -p_new
            
            # Accept/reject
            H_prop = U(w_new) + 0.5 * np.sum(p_new**2)
            log_alpha = -H_prop + H_current
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = w_new
                n_accepted += 1
            else:
                samples[t] = w
        
        return samples, n_accepted / (n_steps - 1)
    
    # Run HMC
    w0 = 0.1 * np.random.randn(nn.n_weights)
    n_steps = 1000
    epsilon = 0.01
    L = 10
    
    samples, accept_rate = hmc_nn(w0, n_steps, epsilon, L)
    
    print(f"\nAcceptance rate: {accept_rate:.1%}")
    
    # Make predictions
    X_test = np.linspace(-3, 3, 100)
    burn_in = 200
    
    predictions = np.zeros((len(samples) - burn_in, len(X_test)))
    for i, w in enumerate(samples[burn_in:]):
        predictions[i] = nn.forward(X_test, w)
    
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Predictions
    ax = axes[0]
    ax.scatter(X_train, y_train, color='red', s=50, alpha=0.6, 
              label='Training data', zorder=5)
    ax.plot(X_test, true_function(X_test), 'k--', linewidth=2, 
           label='True function', alpha=0.7)
    ax.plot(X_test, pred_mean, 'b-', linewidth=2, 
           label='Posterior mean')
    ax.fill_between(X_test, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                    alpha=0.3, color='blue', label='95% credible interval')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Bayesian Neural Network Predictions\n(via HMC)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Sample trajectories
    ax = axes[1]
    # Plot first few weight traces
    for i in range(5):
        ax.plot(samples[:, i], alpha=0.7, linewidth=0.5, 
               label=f'$w_{i}$')
    ax.axvline(x=burn_in, color='red', linestyle='--', 
              linewidth=2, label='Burn-in', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Weight value', fontsize=12)
    ax.set_title('Weight Trace Plots',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'hmc_bayesian_nn.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: hmc_bayesian_nn.png")
    plt.close()
    
    print("\nHMC successfully sampled from 31-dimensional posterior!")


# =============================================================================
# PART 4: VISUALIZING HAMILTONIAN TRAJECTORIES (ADVANCED)
# =============================================================================

def visualize_hamiltonian_trajectory():
    """
    Example 4: Visualizing Hamiltonian Trajectories
    ===============================================
    
    Show how Hamiltonian dynamics explores the distribution
    by following constant energy contours.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Visualizing Hamiltonian Trajectories")
    print("=" * 80)
    
    np.random.seed(42)
    
    # 2D Gaussian target
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.7],
                      [0.7, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def U(x):
        return 0.5 * (x - mu) @ Sigma_inv @ (x - mu)
    
    def grad_U(x):
        return Sigma_inv @ (x - mu)
    
    def H(x, p):
        """Total Hamiltonian"""
        return U(x) + 0.5 * np.sum(p**2)
    
    # Run one HMC iteration with detailed trajectory
    x0 = np.array([1.5, 1.0])
    p0 = np.random.randn(2)
    
    epsilon = 0.15
    L = 30
    
    # Store trajectory
    trajectory_x = [x0.copy()]
    trajectory_p = [p0.copy()]
    energy = [H(x0, p0)]
    
    x, p = x0.copy(), p0.copy()
    
    # Half step for momentum
    p = p - (epsilon / 2) * grad_U(x)
    
    for i in range(L):
        # Full step for position
        x = x + epsilon * p
        trajectory_x.append(x.copy())
        
        # Full step for momentum (except at end)
        if i < L - 1:
            p = p - epsilon * grad_U(x)
            trajectory_p.append(p.copy())
            energy.append(H(x, p))
    
    # Final half step
    p = p - (epsilon / 2) * grad_U(x)
    trajectory_p.append(p.copy())
    energy.append(H(x, p))
    
    trajectory_x = np.array(trajectory_x)
    trajectory_p = np.array(trajectory_p)
    energy = np.array(energy)
    
    # Visualize
    fig = plt.figure(figsize=(18, 6))
    
    # Position space
    ax1 = fig.add_subplot(131)
    
    # Contours
    x1 = np.linspace(-2.5, 2.5, 100)
    x2 = np.linspace(-2.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[i, j] = np.exp(-U(np.array([X1[i, j], X2[i, j]])))
    
    ax1.contour(X1, X2, Z, levels=15, colors='gray', alpha=0.3)
    
    # Trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_x)))
    for i in range(len(trajectory_x)-1):
        ax1.plot(trajectory_x[i:i+2, 0], trajectory_x[i:i+2, 1], 
                'o-', color=colors[i], markersize=4, linewidth=1.5)
    
    ax1.plot(x0[0], x0[1], 'g*', markersize=20, label='Start')
    ax1.plot(trajectory_x[-1, 0], trajectory_x[-1, 1], 
            'r*', markersize=20, label='End')
    
    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_title('Position Space Trajectory\n(Hamiltonian Dynamics)',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Momentum space
    ax2 = fig.add_subplot(132)
    
    for i in range(len(trajectory_p)-1):
        ax2.plot(trajectory_p[i:i+2, 0], trajectory_p[i:i+2, 1], 
                'o-', color=colors[i], markersize=4, linewidth=1.5)
    
    ax2.plot(p0[0], p0[1], 'g*', markersize=20, label='Start')
    ax2.plot(trajectory_p[-1, 0], trajectory_p[-1, 1], 
            'r*', markersize=20, label='End')
    
    ax2.set_xlabel('$p_1$', fontsize=12)
    ax2.set_ylabel('$p_2$', fontsize=12)
    ax2.set_title('Momentum Space Trajectory',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Energy conservation
    ax3 = fig.add_subplot(133)
    
    ax3.plot(energy, 'o-', linewidth=2, markersize=4)
    ax3.axhline(y=energy[0], color='red', linestyle='--', 
               linewidth=2, label=f'Initial: {energy[0]:.3f}', alpha=0.7)
    ax3.set_xlabel('Leapfrog step', fontsize=12)
    ax3.set_ylabel('Hamiltonian H(x, p)', fontsize=12)
    ax3.set_title(f'Energy Conservation\nΔH = {energy[-1] - energy[0]:.4f}',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'hmc_trajectory_visualization.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: hmc_trajectory_visualization.png")
    plt.close()
    
    print(f"\nEnergy change: ΔH = {energy[-1] - energy[0]:.6f}")
    print("Small energy change → high acceptance probability!")
    print("Trajectory explores distant regions while maintaining energy!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HAMILTONIAN MONTE CARLO TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial covers physics-inspired MCMC:")
    print("1. HMC basics with 1D Gaussian")
    print("2. Comparison with MALA and Random Walk")
    print("3. Bayesian neural network inference")
    print("4. Visualizing Hamiltonian trajectories")
    print("\n" + "=" * 80 + "\n")
    
    # Run all examples
    hmc_1d_gaussian()
    compare_samplers_2d()
    hmc_bayesian_nn()
    visualize_hamiltonian_trajectory()
    
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - hmc_1d_gaussian.png")
    print("  - hmc_comparison_2d.png")
    print("  - hmc_autocorrelation.png")
    print("  - hmc_bayesian_nn.png")
    print("  - hmc_trajectory_visualization.png")
    print("\nKey Takeaways:")
    print("  • HMC uses physics (Hamiltonian dynamics) for efficient sampling")
    print("  • Makes distant proposals with high acceptance rates")
    print("  • Much faster mixing than random walk, especially in high-D")
    print("  • Energy conservation is key to performance")
    print("  • Foundation for NUTS and modern probabilistic programming")
    print("=" * 80 + "\n")
