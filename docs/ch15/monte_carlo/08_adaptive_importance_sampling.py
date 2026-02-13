"""
08_adaptive_importance_sampling.py

ADVANCED LEVEL: Adaptive Importance Sampling (AIS)

This module implements adaptive importance sampling, where the proposal
distribution is iteratively refined based on previously drawn samples.

Mathematical Foundation:
---------------------
Standard IS uses a fixed proposal q(θ). AIS iteratively improves q:

Algorithm (Population Monte Carlo - PMC):
1. Initialize: Choose q₀(θ)
2. For t = 1, 2, ..., T:
   a) Sample θᵢᵗ ~ qₜ₋₁(θ) for i=1,...,n
   b) Compute weights wᵢᵗ = π(θᵢᵗ)/qₜ₋₁(θᵢᵗ)
   c) Update proposal: qₜ(θ) based on {θᵢᵗ, wᵢᵗ}
3. Return final samples with importance weights

Common Update Strategies:
1. Mixture of Gaussians centered at high-weight samples
2. Weighted kernel density estimate
3. Parametric fit (e.g., Gaussian with updated mean/covariance)

Advantages:
- Automatically adapts to target distribution
- Can discover multimodal structure
- Generally achieves higher ESS than fixed proposal

Challenges:
- Proper weight calculation requires care
- Can get stuck in local modes
- Computational cost of updating proposal

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from typing import Tuple, List, Callable

np.random.seed(42)
sns.set_style("whitegrid")


class AdaptiveImportanceSampler:
    """
    Adaptive Importance Sampling with mixture proposal.
    """
    
    def __init__(self, target_log_density: Callable, dim: int,
                 n_components: int = 5, initial_scale: float = 2.0):
        """
        Parameters:
        -----------
        target_log_density : function
            Log of unnormalized target density
        dim : int
            Dimension of parameter space
        n_components : int
            Number of mixture components
        initial_scale : float
            Initial scale for proposal components
        """
        self.target_log_density = target_log_density
        self.dim = dim
        self.n_components = n_components
        self.initial_scale = initial_scale
        
        # Storage for iterations
        self.samples_history = []
        self.weights_history = []
        self.ess_history = []
        
    def initialize_proposal(self, initial_mean: np.ndarray = None):
        """
        Initialize mixture proposal as N(μ, σ²I) centered at initial_mean.
        """
        if initial_mean is None:
            initial_mean = np.zeros(self.dim)
        
        # Start with a single broad Gaussian
        self.mixture_means = [initial_mean.copy()]
        self.mixture_covs = [np.eye(self.dim) * self.initial_scale**2]
        self.mixture_weights = [1.0]
        
    def proposal_log_density(self, theta: np.ndarray) -> float:
        """
        Evaluate log density of current mixture proposal at theta.
        
        q(θ) = Σⱼ αⱼ N(θ|μⱼ, Σⱼ)
        """
        # Evaluate each component
        log_densities = []
        for mean, cov, weight in zip(self.mixture_means, self.mixture_covs, 
                                      self.mixture_weights):
            # Multivariate normal log density
            component = stats.multivariate_normal(mean, cov)
            log_densities.append(np.log(weight + 1e-300) + component.logpdf(theta))
        
        # Log-sum-exp for numerical stability
        return logsumexp(log_densities)
    
    def sample_from_proposal(self, n_samples: int) -> np.ndarray:
        """
        Sample from current mixture proposal.
        """
        samples = []
        
        # Sample component assignments
        component_probs = np.array(self.mixture_weights)
        component_probs /= component_probs.sum()
        components = np.random.choice(len(self.mixture_means), 
                                     size=n_samples, p=component_probs)
        
        # Sample from assigned components
        for i in range(n_samples):
            comp_idx = components[i]
            mean = self.mixture_means[comp_idx]
            cov = self.mixture_covs[comp_idx]
            sample = np.random.multivariate_normal(mean, cov)
            samples.append(sample)
        
        return np.array(samples)
    
    def update_proposal(self, samples: np.ndarray, weights: np.ndarray, 
                       method: str = 'resample'):
        """
        Update mixture proposal based on weighted samples.
        
        Methods:
        --------
        'resample': Fit mixture to resampled particles
        'weighted_means': Place components at high-weight samples
        """
        # Normalize weights
        normalized_weights = weights / np.sum(weights)
        
        if method == 'resample':
            # Resample according to weights
            indices = np.random.choice(len(samples), size=self.n_components,
                                      replace=True, p=normalized_weights)
            selected_samples = samples[indices]
            
            # Compute empirical covariance of all samples
            weighted_cov = np.cov(samples.T, aweights=normalized_weights)
            
            # Add small regularization for numerical stability
            weighted_cov += np.eye(self.dim) * 1e-4
            
            # Shrink covariance for exploration-exploitation trade-off
            shrinkage = 0.7
            weighted_cov *= shrinkage
            
            # Update mixture components
            self.mixture_means = [s for s in selected_samples]
            self.mixture_covs = [weighted_cov for _ in range(self.n_components)]
            self.mixture_weights = [1.0/self.n_components] * self.n_components
            
        elif method == 'weighted_means':
            # Select top-weighted samples
            top_indices = np.argsort(normalized_weights)[-self.n_components:]
            
            # Use selected samples as means
            self.mixture_means = [samples[i] for i in top_indices]
            
            # Compute adaptive covariance
            weighted_cov = np.cov(samples.T, aweights=normalized_weights)
            weighted_cov += np.eye(self.dim) * 1e-4
            weighted_cov *= 0.5  # Shrink for stability
            
            self.mixture_covs = [weighted_cov for _ in range(self.n_components)]
            
            # Weights proportional to importance weights
            selected_weights = normalized_weights[top_indices]
            self.mixture_weights = (selected_weights / selected_weights.sum()).tolist()
    
    def run(self, n_samples: int, n_iterations: int, 
            update_method: str = 'resample',
            initial_mean: np.ndarray = None,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run adaptive importance sampling.
        
        Returns:
        --------
        samples : array of shape (n_iterations * n_samples, dim)
        weights : array of normalized weights
        """
        self.initialize_proposal(initial_mean)
        
        all_samples = []
        all_log_weights = []
        
        if verbose:
            print(f"\nRunning Adaptive IS: {n_iterations} iterations, "
                  f"{n_samples} samples each")
            print("=" * 60)
        
        for t in range(n_iterations):
            # Sample from current proposal
            samples = self.sample_from_proposal(n_samples)
            
            # Compute log importance weights
            log_weights = np.array([
                self.target_log_density(s) - self.proposal_log_density(s)
                for s in samples
            ])
            
            # Store samples and weights
            all_samples.append(samples)
            all_log_weights.append(log_weights)
            
            # Normalize weights for this iteration
            log_weights_normalized = log_weights - logsumexp(log_weights)
            weights = np.exp(log_weights_normalized)
            
            # Compute ESS for diagnostics
            ess = 1.0 / np.sum(weights**2)
            self.ess_history.append(ess)
            
            if verbose:
                print(f"Iteration {t+1:3d}: ESS = {ess:7.1f} ({ess/n_samples:5.1%})")
            
            # Update proposal (except on last iteration)
            if t < n_iterations - 1:
                self.update_proposal(samples, np.exp(log_weights), 
                                    method=update_method)
            
            self.samples_history.append(samples)
            self.weights_history.append(weights)
        
        # Combine all samples and recompute final weights
        all_samples = np.vstack(all_samples)
        all_log_weights = np.concatenate(all_log_weights)
        
        # Final normalized weights
        final_log_weights = all_log_weights - logsumexp(all_log_weights)
        final_weights = np.exp(final_log_weights)
        
        return all_samples, final_weights


# Example 1: 1D Bimodal Target
# ==========================
print("=" * 70)
print("EXAMPLE 1: Adaptive IS for 1D Bimodal Distribution")
print("=" * 70)

# Target: mixture of two Gaussians
def target_log_density_1d(theta):
    """Bimodal target: 0.3*N(-3,1) + 0.7*N(4,1.5)"""
    if theta.ndim == 0 or len(theta) == 1:
        theta = np.atleast_1d(theta)
    
    # First mode: N(-3, 1)
    log_p1 = stats.norm.logpdf(theta[0], -3, 1) + np.log(0.3)
    
    # Second mode: N(4, 1.5)
    log_p2 = stats.norm.logpdf(theta[0], 4, 1.5) + np.log(0.7)
    
    return logsumexp([log_p1, log_p2])

# True target for plotting
def true_target_1d(x):
    return 0.3 * stats.norm.pdf(x, -3, 1) + 0.7 * stats.norm.pdf(x, 4, 1.5)

# Run adaptive IS
sampler_1d = AdaptiveImportanceSampler(
    target_log_density=target_log_density_1d,
    dim=1,
    n_components=3,
    initial_scale=3.0
)

samples_1d, weights_1d = sampler_1d.run(
    n_samples=200,
    n_iterations=10,
    update_method='resample',
    initial_mean=np.array([0.0])
)

final_ess = 1.0 / np.sum(weights_1d**2)
print(f"\nFinal ESS: {final_ess:.1f} out of {len(samples_1d)} samples")
print(f"Efficiency: {final_ess/len(samples_1d):.1%}")

# Visualize adaptation process
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.ravel()

x_plot = np.linspace(-8, 10, 1000)
iterations_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for idx, iter_num in enumerate(iterations_to_plot):
    ax = axes[idx]
    
    # Plot true target
    ax.plot(x_plot, true_target_1d(x_plot), 'r-', linewidth=2, 
            label='Target', alpha=0.7)
    
    # Plot samples for this iteration
    if iter_num < len(sampler_1d.samples_history):
        samples_iter = sampler_1d.samples_history[iter_num]
        weights_iter = sampler_1d.weights_history[iter_num]
        
        # Histogram of samples
        ax.hist(samples_iter.flatten(), bins=30, density=True, alpha=0.5,
                color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Scatter samples colored by weight
        y_pos = np.zeros(len(samples_iter))
        scatter = ax.scatter(samples_iter.flatten(), y_pos, 
                           c=weights_iter*len(weights_iter), 
                           cmap='hot', s=50, alpha=0.6, 
                           edgecolors='black', linewidth=0.5)
        
        ess = sampler_1d.ess_history[iter_num]
        ax.set_title(f'Iteration {iter_num+1}: ESS={ess:.0f}',
                    fontsize=11, fontweight='bold')
    
    ax.set_xlim([-8, 10])
    ax.set_ylim([0, 0.3])
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/adaptive_1d_evolution.png',
            dpi=300, bbox_inches='tight')


# Example 2: 2D Banana-Shaped Distribution
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Adaptive IS for 2D Banana Distribution")
print("=" * 70)

def target_log_density_banana(theta):
    """
    Banana-shaped distribution (Rosenbrock-like).
    
    p(θ₁, θ₂) ∝ exp(-0.5[(θ₁-2)²/4 + (θ₂-θ₁²)²])
    """
    theta = np.atleast_1d(theta)
    theta1, theta2 = theta[0], theta[1]
    
    term1 = -0.5 * (theta1 - 2)**2 / 4.0
    term2 = -0.5 * (theta2 - theta1**2)**2
    
    return term1 + term2

# Run adaptive IS
sampler_banana = AdaptiveImportanceSampler(
    target_log_density=target_log_density_banana,
    dim=2,
    n_components=8,
    initial_scale=3.0
)

samples_banana, weights_banana = sampler_banana.run(
    n_samples=300,
    n_iterations=15,
    update_method='resample',
    initial_mean=np.array([0.0, 0.0])
)

final_ess_banana = 1.0 / np.sum(weights_banana**2)
print(f"\nFinal ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")

# Visualize 2D adaptation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

iterations_2d = [0, 1, 2, 4, 6, 8, 10, 14]

# Create grid for true density
x1_grid = np.linspace(-2, 6, 100)
x2_grid = np.linspace(-5, 25, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i,j] = np.exp(target_log_density_banana(np.array([X1[i,j], X2[i,j]])))

for idx, iter_num in enumerate(iterations_2d):
    ax = axes[idx]
    
    # Contour of true target
    ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
    
    # Plot samples
    if iter_num < len(sampler_banana.samples_history):
        samples_iter = sampler_banana.samples_history[iter_num]
        weights_iter = sampler_banana.weights_history[iter_num]
        
        scatter = ax.scatter(samples_iter[:, 0], samples_iter[:, 1],
                           c=weights_iter*len(weights_iter), cmap='viridis',
                           s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ess = sampler_banana.ess_history[iter_num]
        ax.set_title(f'Iter {iter_num+1}: ESS={ess:.0f}',
                    fontsize=11, fontweight='bold')
    
    ax.set_xlabel('θ₁', fontsize=10)
    ax.set_ylabel('θ₂', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/adaptive_2d_banana.png',
            dpi=300, bbox_inches='tight')


# Example 3: Comparison with Fixed Proposal
# =======================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Adaptive vs Fixed Proposal IS")
print("=" * 70)

# Fixed proposal IS
n_fixed_samples = 3000  # Same total as adaptive (15 iters × 200 samples)
fixed_proposal = stats.multivariate_normal([0, 0], [[9, 0], [0, 9]])
samples_fixed = fixed_proposal.rvs(size=n_fixed_samples)

# Compute weights for fixed proposal
log_weights_fixed = np.array([
    target_log_density_banana(s) - fixed_proposal.logpdf(s)
    for s in samples_fixed
])
log_weights_fixed_norm = log_weights_fixed - logsumexp(log_weights_fixed)
weights_fixed = np.exp(log_weights_fixed_norm)

ess_fixed = 1.0 / np.sum(weights_fixed**2)

print(f"\nFixed Proposal IS:")
print(f"  ESS: {ess_fixed:.1f} out of {n_fixed_samples} samples")
print(f"  Efficiency: {ess_fixed/n_fixed_samples:.1%}")

print(f"\nAdaptive IS:")
print(f"  ESS: {final_ess_banana:.1f} out of {len(samples_banana)} samples")
print(f"  Efficiency: {final_ess_banana/len(samples_banana):.1%}")

print(f"\nImprovement: {final_ess_banana/ess_fixed:.2f}x better ESS")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fixed proposal
ax = axes[0]
ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
scatter = ax.scatter(samples_fixed[:, 0], samples_fixed[:, 1],
                    c=weights_fixed*len(weights_fixed), cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.set_title(f'Fixed Proposal: ESS={ess_fixed:.0f} ({ess_fixed/n_fixed_samples:.1%})',
            fontsize=12, fontweight='bold')
ax.set_xlabel('θ₁', fontsize=11)
ax.set_ylabel('θ₂', fontsize=11)
plt.colorbar(scatter, ax=ax, label='Weight × n')

# Adaptive proposal
ax = axes[1]
ax.contour(X1, X2, Z, levels=10, colors='red', alpha=0.5, linewidths=1.5)
scatter = ax.scatter(samples_banana[:, 0], samples_banana[:, 1],
                    c=weights_banana*len(weights_banana), cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.set_title(f'Adaptive IS: ESS={final_ess_banana:.0f} ({final_ess_banana/len(samples_banana):.1%})',
            fontsize=12, fontweight='bold')
ax.set_xlabel('θ₁', fontsize=11)
ax.set_ylabel('θ₂', fontsize=11)
plt.colorbar(scatter, ax=ax, label='Weight × n')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/fixed_vs_adaptive.png',
            dpi=300, bbox_inches='tight')


# Example 4: ESS Evolution Over Iterations
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Tracking ESS Improvement")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D case
ax = axes[0]
iterations = np.arange(1, len(sampler_1d.ess_history) + 1)
ess_values = sampler_1d.ess_history
ax.plot(iterations, ess_values, 'o-', linewidth=2, markersize=8,
        color='steelblue', label='ESS')
ax.axhline(200, color='red', linestyle='--', linewidth=2, 
          label='n per iteration', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('1D Bimodal: ESS Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# 2D case
ax = axes[1]
iterations_2d = np.arange(1, len(sampler_banana.ess_history) + 1)
ess_values_2d = sampler_banana.ess_history
ax.plot(iterations_2d, ess_values_2d, 'o-', linewidth=2, markersize=8,
        color='darkgreen', label='ESS')
ax.axhline(300, color='red', linestyle='--', linewidth=2,
          label='n per iteration', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('2D Banana: ESS Evolution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_evolution.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. ADAPTIVE IMPORTANCE SAMPLING iteratively refines proposal:
   - Start with broad initial proposal
   - Update based on weighted samples
   - Converge to high-probability regions

2. MIXTURE PROPOSALS are effective:
   - Multiple components cover complex shapes
   - Can capture multimodality
   - Balance exploration vs exploitation

3. UPDATE STRATEGIES:
   - Resample: Draw from weighted distribution
   - Weighted means: Center on high-weight samples
   - Both use empirical covariance for shape

4. ESS TYPICALLY IMPROVES over iterations:
   - Early: low ESS, exploring
   - Middle: ESS increases as proposal adapts
   - Late: ESS plateaus at optimal level

5. ADVANTAGES OVER FIXED PROPOSAL:
   - No need to know good proposal a priori
   - Automatically adapts to target shape
   - Can achieve much higher ESS
   - Handles multimodal targets

6. COMPUTATIONAL CONSIDERATIONS:
   - More expensive per sample than fixed IS
   - But fewer samples needed for same accuracy
   - Trade-off: adaptation cost vs ESS improvement

7. WHEN TO USE ADAPTIVE IS:
   - Target shape unknown
   - Complex/multimodal distributions
   - When fixed proposal gives poor ESS
   - Sequential decision problems

8. PRACTICAL TIPS:
   - Start with broad initial proposal
   - Use 5-10 mixture components
   - Monitor ESS convergence
   - Shrink covariance to avoid overconfidence
   - Regular ESS > 0.3n is excellent

9. COMPARISON TO MCMC:
   - AIS: Independent samples, no burn-in
   - MCMC: Correlated samples, needs burn-in
   - AIS better when good proposals learnable
   - MCMC better for very high dimensions
""")
