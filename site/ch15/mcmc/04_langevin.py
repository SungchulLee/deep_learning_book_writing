"""
04_langevin.py

LANGEVIN DYNAMICS & MALA: COMPREHENSIVE TUTORIAL
================================================

Learning Objectives:
-------------------
1. Understand gradient-based MCMC methods
2. Learn Langevin dynamics and its discretization
3. Master MALA (Metropolis-Adjusted Langevin Algorithm)
4. Compare gradient-free vs. gradient-based sampling
5. Apply to high-dimensional inference problems

Mathematical Foundation:
-----------------------
Goal:
    Use gradient information ∇log p(x) to guide proposals toward high-probability
    regions, dramatically improving efficiency compared to random walk Metropolis.

Langevin Dynamics (Continuous Time):
------------------------------------
The Langevin diffusion is a stochastic differential equation:
    
    dx_t = ∇log p(x_t) dt + √(2) dW_t
    
where:
    - ∇log p(x) is the gradient (drift toward high probability)
    - dW_t is Brownian motion (noise for exploration)
    - At equilibrium, this converges to p(x)

Intuition:
    - Gradient pulls samples "uphill" toward modes
    - Noise prevents getting stuck in local modes
    - Balance between exploitation (gradient) and exploration (noise)

Unadjusted Langevin Algorithm (ULA):
-----------------------------------
Discretize the Langevin SDE with step size ε:

    x_{t+1} = x_t + (ε/2) ∇log p(x_t) + √ε · N(0, I)

Problem: Discretization introduces bias!
    - Not exact samples from p(x)
    - Bias decreases as ε → 0
    - But smaller ε means more iterations needed

MALA (Metropolis-Adjusted Langevin Algorithm):
----------------------------------------------
Add Metropolis-Hastings correction to ULA:

1. Propose: x' = x + (ε/2) ∇log p(x) + √ε · N(0, I)

2. Acceptance probability:
   α = min(1, [p(x') q(x|x')] / [p(x) q(x'|x)])
   
   where q(x'|x) = N(x + (ε/2)∇log p(x), εI)  [asymmetric!]

3. Accept/reject as in Metropolis-Hastings

Why MALA Works Better:
    - Corrects discretization bias (exact samples asymptotically)
    - Uses gradient for intelligent proposals
    - Much faster mixing than random walk (especially high-D)
    - Acceptance rate still informative for tuning ε

Key Differences from Metropolis:
--------------------------------
| Feature          | Metropolis        | MALA                    |
|------------------|-------------------|-------------------------|
| Proposal         | Random walk       | Gradient-guided         |
| Gradient needed? | No                | Yes                     |
| Tuning           | Step size σ       | Step size ε             |
| Efficiency       | O(d²) for d dims  | O(d^{5/4}) (better!)    |
| Best for         | Simple targets    | Smooth, high-D targets  |

Optimal Tuning:
--------------
- Target acceptance rate: ~57-60% (higher than Metropolis!)
- Roberts & Tweedie (1996): optimal scaling
- ε too large → rejections, poor approximation
- ε too small → slow progress, high autocorrelation

Connection to Modern ML:
-----------------------
- Stochastic Gradient Descent: ε∇log p(x) is like gradient ascent
- Diffusion Models: Reverse diffusion uses learned score ∇log p(x)
- Variational Inference: Natural gradient flows
- Langevin MCMC for Bayesian neural networks

Advantages:
----------
+ Much faster than random walk in high dimensions
+ Uses gradient information (if available)
+ Still general-purpose (any differentiable p(x))
+ Theoretical guarantees on convergence

Disadvantages:
-------------
- Requires gradient computation (auto-diff or numerical)
- Can be unstable with poor initialization
- Less effective if gradients are noisy
- Tuning ε is critical

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, gamma as gamma_dist
import seaborn as sns
from matplotlib.patches import Ellipse
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


# =============================================================================
# PART 1: UNADJUSTED LANGEVIN ALGORITHM (BEGINNER)
# =============================================================================

def ula_1d_gaussian():
    """
    Example 1: Unadjusted Langevin Algorithm for 1D Gaussian
    ========================================================
    
    Target: p(x) ∝ exp(-x²/2)  (standard normal)
    Gradient: ∇log p(x) = -x
    
    ULA Update:
    x_{t+1} = x_t + (ε/2)(-x_t) + √ε · N(0,1)
            = (1 - ε/2) x_t + √ε · N(0,1)
    
    This shows ULA has built-in "shrinkage" toward the mode!
    """
    print("=" * 80)
    print("EXAMPLE 1: Unadjusted Langevin Algorithm - 1D Gaussian")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target: standard normal
    def log_prob(x):
        """Log probability (up to constant)"""
        return -0.5 * x**2
    
    def grad_log_prob(x):
        """Gradient of log probability"""
        return -x
    
    # ULA sampler
    def ula_sampler(x0, n_steps, epsilon):
        """
        Unadjusted Langevin Algorithm
        
        Args:
            x0: Initial point
            n_steps: Number of steps
            epsilon: Step size
            
        Returns:
            samples: Array of samples
        """
        samples = np.zeros(n_steps)
        samples[0] = x0
        
        for t in range(1, n_steps):
            # Current position
            x = samples[t-1]
            
            # Gradient term (drift toward high probability)
            drift = (epsilon / 2) * grad_log_prob(x)
            
            # Noise term (exploration)
            noise = np.sqrt(epsilon) * np.random.randn()
            
            # ULA update
            samples[t] = x + drift + noise
            
        return samples
    
    # Run ULA with different step sizes
    n_steps = 5000
    x0 = 3.0  # Start far from mode
    epsilons = [0.01, 0.1, 0.5, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, eps in enumerate(epsilons):
        samples = ula_sampler(x0, n_steps, eps)
        
        ax = axes[idx]
        
        # Plot trace
        ax.plot(samples, alpha=0.7, linewidth=0.5, label='ULA samples')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                   label='True mean', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('x', fontsize=12)
        ax.set_title(f'ULA: ε = {eps}\n' + 
                    f'Sample mean = {samples[1000:].mean():.3f}, ' +
                    f'Sample std = {samples[1000:].std():.3f}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_ula_1d.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_ula_1d.png")
    plt.close()
    
    print("\nObservations:")
    print("- Small ε: Slow convergence but accurate")
    print("- Large ε: Fast convergence but discretization bias!")
    print("- ε = 1.0 shows clear bias (variance < 1)")
    print("- Need Metropolis correction → MALA")
    

# =============================================================================
# PART 2: MALA FOR 2D GAUSSIAN (INTERMEDIATE)
# =============================================================================

def mala_2d_gaussian():
    """
    Example 2: MALA for 2D Gaussian
    ===============================
    
    Target: Bivariate normal with correlation
    p(x) = N(μ, Σ)
    
    Gradient: ∇log p(x) = -Σ⁻¹(x - μ)
    
    Compare:
    - Random walk Metropolis (no gradient)
    - ULA (gradient but no correction)
    - MALA (gradient with correction)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MALA for 2D Correlated Gaussian")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target distribution
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[2.0, 1.5],
                      [1.5, 2.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    target_dist = multivariate_normal(mu, Sigma)
    
    def log_prob(x):
        """Log probability"""
        return target_dist.logpdf(x)
    
    def grad_log_prob(x):
        """Gradient of log probability"""
        return -Sigma_inv @ (x - mu)
    
    # MALA sampler
    def mala_sampler(x0, n_steps, epsilon):
        """
        Metropolis-Adjusted Langevin Algorithm
        
        The key difference from ULA: we add MH accept/reject step!
        """
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            
            # Propose using Langevin dynamics
            grad = grad_log_prob(x)
            drift = (epsilon / 2) * grad
            noise = np.sqrt(epsilon) * np.random.randn(2)
            x_prop = x + drift + noise
            
            # Compute acceptance probability
            # Need to account for asymmetric proposal!
            
            # Forward proposal density: q(x_prop | x)
            grad_x = grad_log_prob(x)
            mean_forward = x + (epsilon / 2) * grad_x
            log_q_forward = multivariate_normal.logpdf(
                x_prop, mean_forward, epsilon * np.eye(2)
            )
            
            # Backward proposal density: q(x | x_prop)
            grad_x_prop = grad_log_prob(x_prop)
            mean_backward = x_prop + (epsilon / 2) * grad_x_prop
            log_q_backward = multivariate_normal.logpdf(
                x, mean_backward, epsilon * np.eye(2)
            )
            
            # Metropolis-Hastings acceptance ratio
            log_alpha = (log_prob(x_prop) + log_q_backward) - \
                       (log_prob(x) + log_q_forward)
            
            # Accept or reject
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
                
        acceptance_rate = n_accepted / (n_steps - 1)
        return samples, acceptance_rate
    
    # Random walk Metropolis for comparison
    def rwm_sampler(x0, n_steps, sigma):
        """Random walk Metropolis (no gradient info)"""
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            x_prop = x + sigma * np.random.randn(2)
            
            log_alpha = log_prob(x_prop) - log_prob(x)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
                
        return samples, n_accepted / (n_steps - 1)
    
    # Run samplers
    n_steps = 3000
    x0 = np.array([0.0, 0.0])
    
    # MALA
    epsilon_mala = 0.3
    samples_mala, accept_mala = mala_sampler(x0, n_steps, epsilon_mala)
    
    # Random walk (tuned for similar acceptance)
    sigma_rwm = 0.5
    samples_rwm, accept_rwm = rwm_sampler(x0, n_steps, sigma_rwm)
    
    # ULA (no correction)
    def ula_sampler_2d(x0, n_steps, epsilon):
        samples = np.zeros((n_steps, 2))
        samples[0] = x0
        for t in range(1, n_steps):
            x = samples[t-1]
            drift = (epsilon / 2) * grad_log_prob(x)
            noise = np.sqrt(epsilon) * np.random.randn(2)
            samples[t] = x + drift + noise
        return samples
    
    samples_ula = ula_sampler_2d(x0, n_steps, epsilon_mala)
    
    # Visualize comparison
    fig = plt.figure(figsize=(18, 6))
    
    # Create grid for contours
    x1 = np.linspace(-3, 5, 100)
    x2 = np.linspace(-2, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    Z = target_dist.pdf(pos)
    
    methods = [
        ('Random Walk Metropolis', samples_rwm, accept_rwm, 'blue'),
        ('ULA (Unadjusted)', samples_ula, 1.0, 'orange'),
        ('MALA', samples_mala, accept_mala, 'green')
    ]
    
    for idx, (name, samples, accept, color) in enumerate(methods):
        ax = fig.add_subplot(1, 3, idx + 1)
        
        # True distribution contours
        ax.contour(X1, X2, Z, levels=10, colors='gray', alpha=0.3)
        
        # Sample trajectory (first 500 points)
        burn_in = 100
        traj = samples[burn_in:600]
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, 
                alpha=0.3, markersize=2, linewidth=0.5)
        
        # Scatter plot of all samples
        ax.scatter(samples[1000:, 0], samples[1000:, 1], 
                  alpha=0.1, s=10, color=color)
        
        # Mark start
        ax.plot(x0[0], x0[1], 'r*', markersize=20, label='Start')
        
        # Mark true mean
        ax.plot(mu[0], mu[1], 'k*', markersize=20, label='True mean')
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        
        if idx == 1:  # ULA
            title = f'{name}\nNo accept/reject'
        else:
            title = f'{name}\nAcceptance: {accept:.1%}'
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_mala_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_mala_comparison.png")
    plt.close()
    
    # Compute autocorrelations
    def autocorr(x, lag=50):
        """Compute autocorrelation"""
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        acf = np.array([np.dot(x[:-k], x[k:]) / len(x) / c0 
                       if k > 0 else 1.0 for k in range(lag)])
        return acf
    
    # Plot autocorrelations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    lags = np.arange(50)
    
    # RWM
    acf_rwm = autocorr(samples_rwm[1000:, 0])
    axes[0].plot(lags, acf_rwm, 'o-', color='blue', linewidth=2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_title('Random Walk Metropolis\nAutocorrelation', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Lag', fontsize=12)
    axes[0].set_ylabel('ACF', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # ULA
    acf_ula = autocorr(samples_ula[1000:, 0])
    axes[1].plot(lags, acf_ula, 'o-', color='orange', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('ULA\nAutocorrelation', 
                     fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Lag', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # MALA
    acf_mala = autocorr(samples_mala[1000:, 0])
    axes[2].plot(lags, acf_mala, 'o-', color='green', linewidth=2)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_title('MALA\nAutocorrelation', 
                     fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Lag', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_autocorrelation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_autocorrelation.png")
    plt.close()
    
    print(f"\nResults:")
    print(f"MALA acceptance rate: {accept_mala:.1%} (target: 57-60%)")
    print(f"RWM acceptance rate: {accept_rwm:.1%}")
    print(f"\nMALA shows faster decorrelation (lower autocorrelation)!")


# =============================================================================
# PART 3: MALA FOR BAYESIAN LOGISTIC REGRESSION (ADVANCED)
# =============================================================================

def mala_logistic_regression():
    """
    Example 3: MALA for Bayesian Logistic Regression
    ================================================
    
    Problem: Binary classification with uncertainty quantification
    
    Model:
        y_i ~ Bernoulli(σ(w^T x_i))
        w ~ N(0, σ_w² I)
    
    where σ(z) = 1/(1 + exp(-z)) is the sigmoid function.
    
    Posterior: p(w | data) ∝ p(data | w) p(w)
    
    Gradient:
        ∇log p(w) = X^T(y - σ(Xw)) - w/σ_w²
    
    This is a realistic application where MALA shines!
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: MALA for Bayesian Logistic Regression")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    n_features = 5
    
    # True weights
    w_true = np.array([2.0, -1.5, 1.0, -0.5, 0.8])
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels
    logits = X @ w_true
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(float)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Class balance: {y.mean():.2f}")
    
    # Prior variance
    sigma_w = 3.0
    
    def sigmoid(z):
        """Numerically stable sigmoid"""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def log_posterior(w):
        """Log posterior (unnormalized)"""
        # Log likelihood
        logits = X @ w
        log_lik = np.sum(y * logits - np.log(1 + np.exp(logits)))
        
        # Log prior
        log_prior = -0.5 * np.sum(w**2) / sigma_w**2
        
        return log_lik + log_prior
    
    def grad_log_posterior(w):
        """Gradient of log posterior"""
        # Gradient of log likelihood
        probs = sigmoid(X @ w)
        grad_lik = X.T @ (y - probs)
        
        # Gradient of log prior
        grad_prior = -w / sigma_w**2
        
        return grad_lik + grad_prior
    
    # MALA sampler
    n_steps = 5000
    w0 = np.zeros(n_features)
    epsilon = 0.01  # Small step size for stability
    
    samples = np.zeros((n_steps, n_features))
    samples[0] = w0
    n_accepted = 0
    
    print("\nRunning MALA...")
    for t in range(1, n_steps):
        if t % 1000 == 0:
            print(f"  Step {t}/{n_steps}")
        
        w = samples[t-1]
        
        # Propose using Langevin dynamics
        grad = grad_log_posterior(w)
        drift = (epsilon / 2) * grad
        noise = np.sqrt(epsilon) * np.random.randn(n_features)
        w_prop = w + drift + noise
        
        # Compute acceptance probability
        # Forward proposal
        mean_forward = w + (epsilon / 2) * grad_log_posterior(w)
        log_q_forward = -0.5 * np.sum((w_prop - mean_forward)**2) / epsilon
        
        # Backward proposal
        mean_backward = w_prop + (epsilon / 2) * grad_log_posterior(w_prop)
        log_q_backward = -0.5 * np.sum((w - mean_backward)**2) / epsilon
        
        # MH ratio
        log_alpha = (log_posterior(w_prop) + log_q_backward) - \
                   (log_posterior(w) + log_q_forward)
        
        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            samples[t] = w_prop
            n_accepted += 1
        else:
            samples[t] = w
    
    acceptance_rate = n_accepted / (n_steps - 1)
    print(f"\nAcceptance rate: {acceptance_rate:.1%}")
    
    # Analyze results
    burn_in = 1000
    samples_post = samples[burn_in:]
    
    # Posterior means and credible intervals
    w_mean = samples_post.mean(axis=0)
    w_std = samples_post.std(axis=0)
    
    print("\nPosterior Summary:")
    print("=" * 60)
    print(f"{'Weight':<10} {'True':<10} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
    print("=" * 60)
    for i in range(n_features):
        ci_lower = np.percentile(samples_post[:, i], 2.5)
        ci_upper = np.percentile(samples_post[:, i], 97.5)
        print(f"w_{i:<8} {w_true[i]:<10.3f} {w_mean[i]:<10.3f} "
              f"{w_std[i]:<10.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Trace plot
        ax.plot(samples[:, i], alpha=0.5, linewidth=0.5, color='blue')
        ax.axvline(x=burn_in, color='red', linestyle='--', 
                   label='Burn-in', alpha=0.7)
        ax.axhline(y=w_true[i], color='green', linestyle='--', 
                   linewidth=2, label='True value', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel(f'$w_{i}$', fontsize=11)
        ax.set_title(f'Weight {i}: True = {w_true[i]:.2f}, ' +
                    f'Posterior = {w_mean[i]:.2f} ± {w_std[i]:.2f}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Use last subplot for histogram of first weight
    ax = axes[-1]
    ax.hist(samples_post[:, 0], bins=50, density=True, 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=w_true[0], color='red', linestyle='--', 
               linewidth=2, label='True value')
    ax.axvline(x=w_mean[0], color='blue', linestyle='--', 
               linewidth=2, label='Posterior mean')
    ax.set_xlabel('$w_0$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Posterior Distribution of $w_0$', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_logistic_regression.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_logistic_regression.png")
    plt.close()
    
    # Prediction with uncertainty
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test = np.concatenate([X_test, np.random.randn(100, n_features-1)], axis=1)
    
    # Compute predictions for each posterior sample
    n_pred_samples = 500
    pred_samples = np.random.choice(len(samples_post), n_pred_samples, replace=False)
    
    preds = np.zeros((n_pred_samples, 100))
    for idx, sample_idx in enumerate(pred_samples):
        w_sample = samples_post[sample_idx]
        logits = X_test @ w_sample
        preds[idx] = sigmoid(logits)
    
    pred_mean = preds.mean(axis=0)
    pred_lower = np.percentile(preds, 2.5, axis=0)
    pred_upper = np.percentile(preds, 97.5, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X_test[:, 0], pred_mean, 'b-', linewidth=2, label='Posterior mean')
    ax.fill_between(X_test[:, 0], pred_lower, pred_upper, 
                    alpha=0.3, color='blue', label='95% credible interval')
    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_ylabel('P(y=1 | x)', fontsize=12)
    ax.set_title('Bayesian Logistic Regression Predictions with Uncertainty',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_predictions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_predictions.png")
    plt.close()


# =============================================================================
# PART 4: TUNING ANALYSIS (ADVANCED)
# =============================================================================

def mala_tuning_analysis():
    """
    Example 4: Analyzing MALA Performance vs Step Size
    ==================================================
    
    Study how step size ε affects:
    - Acceptance rate
    - Mixing (autocorrelation)
    - Effective sample size
    
    Find optimal ε for target distribution.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: MALA Tuning Analysis")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Target: 2D Gaussian
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    def log_prob(x):
        return multivariate_normal.logpdf(x, mu, Sigma)
    
    def grad_log_prob(x):
        return -Sigma_inv @ (x - mu)
    
    # Test different step sizes
    epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    n_steps = 3000
    
    results = []
    
    print("\nTesting different step sizes...")
    for eps in epsilons:
        print(f"  ε = {eps}")
        
        samples = np.zeros((n_steps, 2))
        samples[0] = np.array([2.0, 2.0])
        n_accepted = 0
        
        for t in range(1, n_steps):
            x = samples[t-1]
            
            # MALA proposal
            grad = grad_log_prob(x)
            drift = (eps / 2) * grad
            noise = np.sqrt(eps) * np.random.randn(2)
            x_prop = x + drift + noise
            
            # Acceptance
            mean_forward = x + (eps / 2) * grad_log_prob(x)
            log_q_forward = multivariate_normal.logpdf(
                x_prop, mean_forward, eps * np.eye(2)
            )
            
            mean_backward = x_prop + (eps / 2) * grad_log_prob(x_prop)
            log_q_backward = multivariate_normal.logpdf(
                x, mean_backward, eps * np.eye(2)
            )
            
            log_alpha = (log_prob(x_prop) + log_q_backward) - \
                       (log_prob(x) + log_q_forward)
            
            if np.log(np.random.rand()) < log_alpha:
                samples[t] = x_prop
                n_accepted += 1
            else:
                samples[t] = x
        
        acceptance_rate = n_accepted / (n_steps - 1)
        
        # Compute autocorrelation at lag 10
        burn_in = 500
        samples_post = samples[burn_in:, 0]
        samples_centered = samples_post - samples_post.mean()
        acf_lag10 = np.correlate(samples_centered[:-10], 
                                 samples_centered[10:], mode='valid')[0] / \
                    np.correlate(samples_centered, samples_centered, mode='valid')[0]
        
        # Effective sample size (simple estimate)
        ess = len(samples_post) / (1 + 2 * acf_lag10) if acf_lag10 > 0 else len(samples_post)
        
        results.append({
            'epsilon': eps,
            'acceptance_rate': acceptance_rate,
            'acf_lag10': acf_lag10,
            'ess': ess
        })
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epsilons_arr = np.array([r['epsilon'] for r in results])
    acceptance_arr = np.array([r['acceptance_rate'] for r in results])
    acf_arr = np.array([r['acf_lag10'] for r in results])
    ess_arr = np.array([r['ess'] for r in results])
    
    # Acceptance rate
    axes[0].plot(epsilons_arr, acceptance_arr, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=0.574, color='red', linestyle='--', 
                    linewidth=2, label='Optimal (57.4%)', alpha=0.7)
    axes[0].set_xlabel('Step size ε', fontsize=12)
    axes[0].set_ylabel('Acceptance rate', fontsize=12)
    axes[0].set_title('Acceptance Rate vs Step Size', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Autocorrelation at lag 10
    axes[1].plot(epsilons_arr, acf_arr, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Step size ε', fontsize=12)
    axes[1].set_ylabel('ACF(lag=10)', fontsize=12)
    axes[1].set_title('Autocorrelation vs Step Size\n(Lower is better)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Effective sample size
    axes[2].plot(epsilons_arr, ess_arr, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Step size ε', fontsize=12)
    axes[2].set_ylabel('Effective sample size', fontsize=12)
    axes[2].set_title('ESS vs Step Size\n(Higher is better)', 
                     fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'langevin_tuning_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print("✓ Saved: langevin_tuning_analysis.png")
    plt.close()
    
    # Find optimal epsilon
    optimal_idx = np.argmax(ess_arr)
    optimal_eps = epsilons_arr[optimal_idx]
    optimal_accept = acceptance_arr[optimal_idx]
    
    print(f"\nOptimal step size: ε = {optimal_eps}")
    print(f"Acceptance rate: {optimal_accept:.1%}")
    print(f"Effective sample size: {ess_arr[optimal_idx]:.0f}")
    print("\nNote: Optimal acceptance for MALA is ~57%, higher than Metropolis!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LANGEVIN DYNAMICS & MALA TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial covers gradient-based MCMC methods:")
    print("1. Unadjusted Langevin Algorithm (ULA)")
    print("2. Metropolis-Adjusted Langevin Algorithm (MALA)")
    print("3. Application to Bayesian inference")
    print("4. Tuning analysis")
    print("\n" + "=" * 80 + "\n")
    
    # Run all examples
    ula_1d_gaussian()
    mala_2d_gaussian()
    mala_logistic_regression()
    mala_tuning_analysis()
    
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - langevin_ula_1d.png")
    print("  - langevin_mala_comparison.png")
    print("  - langevin_autocorrelation.png")
    print("  - langevin_logistic_regression.png")
    print("  - langevin_predictions.png")
    print("  - langevin_tuning_analysis.png")
    print("\nKey Takeaways:")
    print("  • Gradient information dramatically improves MCMC efficiency")
    print("  • ULA is fast but biased; MALA corrects the bias")
    print("  • MALA optimal acceptance: ~57% (higher than Metropolis ~23%)")
    print("  • Essential for high-dimensional Bayesian inference")
    print("  • Foundation for understanding modern sampling methods (HMC, NUTS)")
    print("=" * 80 + "\n")
