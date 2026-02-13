"""
09_sequential_importance_sampling.py

ADVANCED LEVEL: Sequential Importance Sampling (SIS)

This module implements sequential importance sampling for processing
data sequentially and for sequential Bayesian inference.

Mathematical Foundation:
---------------------
Sequential IS updates importance weights as new data arrives:

At time t, we have data y₁:t = (y₁, ..., y_t)

Posterior: p(θ|y₁:t) ∝ p(y₁:t|θ)p(θ)

Sequential factorization:
    p(θ|y₁:t) ∝ p(yt|θ, y₁:t₋₁) × p(θ|y₁:t₋₁)

Importance Weight Update:
    w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ, y₁:t₋₁) / q_t(θ|y₁:t, θ₁:t₋₁)

For fixed proposal q(θ) (not adaptive):
    w_t(θ) ∝ w_t₋₁(θ) × p(yt|θ)

The weights multiply over time, leading to:

Weight Degeneracy Problem:
- Weights become increasingly unequal
- ESS decreases over time
- Eventually dominated by few particles

Solutions:
1. Resampling (Particle Filter)
2. Adaptive proposals
3. Defensive mixing

Applications:
- Sequential Bayesian updating
- Time series analysis  
- Online learning
- Particle filters

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


class SequentialImportanceSampler:
    """
    Sequential Importance Sampling with optional resampling.
    """
    
    def __init__(self, prior_dist, proposal_dist):
        """
        Parameters:
        -----------
        prior_dist : scipy.stats distribution
            Prior distribution p(θ)
        proposal_dist : scipy.stats distribution
            Proposal distribution q(θ)
        """
        self.prior_dist = prior_dist
        self.proposal_dist = proposal_dist
        
        # Storage
        self.particles = None
        self.log_weights = None
        self.weights = None
        self.ess_history = []
        self.t = 0
        
    def initialize(self, n_particles: int):
        """
        Initialize particles from proposal.
        """
        self.n_particles = n_particles
        self.particles = self.proposal_dist.rvs(size=n_particles)
        
        # Initial weights: w₀(θ) = p(θ)/q(θ)
        self.log_weights = (
            self.prior_dist.logpdf(self.particles) -
            self.proposal_dist.logpdf(self.particles)
        )
        
        # Normalize
        self._normalize_weights()
        self.t = 0
        
        # Track ESS
        ess = self.compute_ess()
        self.ess_history.append(ess)
        
    def update(self, likelihood_fn, y_new):
        """
        Update weights with new observation.
        
        Parameters:
        -----------
        likelihood_fn : callable
            Function: likelihood_fn(theta, y) -> likelihood value
        y_new : observation
            New data point
        """
        self.t += 1
        
        # Update log weights: log w_t = log w_{t-1} + log p(y_t|θ)
        log_likelihoods = np.array([
            likelihood_fn(theta, y_new) for theta in self.particles
        ])
        
        self.log_weights += log_likelihoods
        
        # Normalize
        self._normalize_weights()
        
        # Track ESS
        ess = self.compute_ess()
        self.ess_history.append(ess)
        
        return ess
    
    def _normalize_weights(self):
        """Normalize weights using log-sum-exp for stability."""
        self.log_weights = self.log_weights - logsumexp(self.log_weights)
        self.weights = np.exp(self.log_weights)
    
    def compute_ess(self):
        """Compute Effective Sample Size."""
        return 1.0 / np.sum(self.weights**2)
    
    def resample(self, threshold=0.5):
        """
        Resample particles if ESS drops below threshold.
        
        Multinomial resampling: draw particles with replacement
        according to their weights.
        """
        ess = self.compute_ess()
        rel_ess = ess / self.n_particles
        
        if rel_ess < threshold:
            # Multinomial resampling
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                replace=True,
                p=self.weights
            )
            
            self.particles = self.particles[indices]
            
            # Reset weights to uniform
            self.log_weights = np.zeros(self.n_particles)
            self.weights = np.ones(self.n_particles) / self.n_particles
            
            return True  # Resampled
        
        return False  # No resampling
    
    def estimate(self, h_function):
        """
        Compute weighted estimate of E[h(θ)|y₁:t].
        """
        return np.sum(self.weights * h_function(self.particles))
    
    def credible_interval(self, alpha=0.95):
        """
        Compute credible interval using weighted quantiles.
        """
        sorted_indices = np.argsort(self.particles)
        sorted_particles = self.particles[sorted_indices]
        sorted_weights = self.weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        
        lower_idx = np.searchsorted(cumsum, (1-alpha)/2)
        upper_idx = np.searchsorted(cumsum, (1+alpha)/2)
        
        return sorted_particles[lower_idx], sorted_particles[upper_idx]


# Example 1: Sequential Bayesian Updating (Normal Mean)
# ===================================================
print("=" * 70)
print("EXAMPLE 1: Sequential Bayesian Updating")
print("=" * 70)

print("""
Model: yᵢ ~ N(θ, σ²) with known σ² = 1
Prior: θ ~ N(0, 4)
Process data sequentially and watch ESS degrade.
""")

# True parameter
theta_true = 2.5
sigma = 1.0

# Generate sequential data
n_obs = 50
data_seq = np.random.normal(theta_true, sigma, n_obs)

print(f"\nTrue θ = {theta_true}")
print(f"Generated {n_obs} observations sequentially")

# Prior and proposal
prior = stats.norm(0, 2)
proposal = stats.norm(0, 2)  # Use prior as proposal

# Likelihood function
def likelihood_normal(theta, y):
    """Log likelihood for single observation."""
    return stats.norm.logpdf(y, theta, sigma)

# Initialize SIS
n_particles = 2000
sis = SequentialImportanceSampler(prior, proposal)
sis.initialize(n_particles)

print(f"\nInitialized {n_particles} particles")
print(f"Initial ESS: {sis.ess_history[0]:.1f}")

# Analytical posterior for comparison
def analytical_posterior(y_data, sigma, mu_0, tau_0):
    """Analytical posterior N(μₙ, τₙ²) for known variance."""
    n = len(y_data)
    tau_n_sq = 1.0 / (1.0/tau_0**2 + n/sigma**2)
    mu_n = tau_n_sq * (mu_0/tau_0**2 + np.sum(y_data)/sigma**2)
    return mu_n, np.sqrt(tau_n_sq)

# Process data sequentially
print("\nSequential Processing:")
print(f"{'t':>3} {'y_t':>8} {'Post Mean':>10} {'ESS':>8} {'Rel ESS':>8}")
print("-" * 50)

posterior_means = []
posterior_stds = []

for t, y_t in enumerate(data_seq):
    # Update with new observation
    ess = sis.update(likelihood_normal, y_t)
    
    # Estimate posterior mean
    post_mean = sis.estimate(lambda x: x)
    posterior_means.append(post_mean)
    
    # Analytical posterior
    mu_n, tau_n = analytical_posterior(data_seq[:t+1], sigma, 0, 2)
    posterior_stds.append(tau_n)
    
    # Print every 5 observations
    if (t+1) % 5 == 0 or t == 0:
        print(f"{t+1:3d} {y_t:8.3f} {post_mean:10.4f} {ess:8.1f} "
              f"{ess/n_particles:8.1%}")

# Final comparison
final_analytical_mean, final_analytical_std = analytical_posterior(
    data_seq, sigma, 0, 2
)

print(f"\nFinal Estimates:")
print(f"  True θ: {theta_true:.4f}")
print(f"  SIS estimate: {posterior_means[-1]:.4f}")
print(f"  Analytical: {final_analytical_mean:.4f}")
print(f"  Final ESS: {sis.ess_history[-1]:.1f} ({sis.ess_history[-1]/n_particles:.1%})")

# Visualize ESS degradation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: ESS over time
ax = axes[0, 0]
ax.plot(range(len(sis.ess_history)), sis.ess_history, 'b-',
        linewidth=2, label='ESS')
ax.axhline(n_particles, color='red', linestyle='--', linewidth=2,
           label='n particles', alpha=0.7)
ax.axhline(n_particles * 0.5, color='orange', linestyle='--', linewidth=1.5,
           label='50% threshold', alpha=0.7)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('Weight Degeneracy Over Time', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Posterior mean evolution
ax = axes[0, 1]
ax.plot(range(1, len(posterior_means)+1), posterior_means, 'b-',
        linewidth=2, label='SIS estimate')
analytical_means = [analytical_posterior(data_seq[:i+1], sigma, 0, 2)[0]
                    for i in range(len(data_seq))]
ax.plot(range(1, len(analytical_means)+1), analytical_means, 'r--',
        linewidth=2, label='Analytical', alpha=0.7)
ax.axhline(theta_true, color='green', linestyle=':', linewidth=2,
           label='True θ', alpha=0.7)
ax.set_xlabel('Number of Observations', fontsize=12)
ax.set_ylabel('Posterior Mean', fontsize=12)
ax.set_title('Posterior Mean Convergence', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: Weight distribution at different times
ax = axes[1, 0]
times_to_plot = [0, 10, 25, 49]
colors_weights = ['blue', 'green', 'orange', 'red']

# Re-run to get weights at specific times
sis_temp = SequentialImportanceSampler(prior, proposal)
sis_temp.initialize(n_particles)
weights_over_time = [sis_temp.weights.copy()]

for t, y_t in enumerate(data_seq):
    sis_temp.update(likelihood_normal, y_t)
    if t+1 in times_to_plot:
        weights_over_time.append(sis_temp.weights.copy())

for idx, (t, weights, color) in enumerate(zip([0] + times_to_plot,
                                                weights_over_time,
                                                colors_weights)):
    ax.hist(weights * n_particles, bins=30, alpha=0.4, color=color,
            label=f't={t}', density=True)

ax.axvline(1.0, color='black', linestyle='--', linewidth=2,
           label='Uniform', alpha=0.5)
ax.set_xlabel('Weight × n', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Weight Distribution Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Current particle distribution
ax = axes[1, 1]
x_range = np.linspace(-2, 6, 1000)

# Plot particles
ax.hist(sis.particles, bins=50, weights=sis.weights, density=True,
        alpha=0.6, color='steelblue', edgecolor='black',
        label='Weighted particles')

# True posterior
true_post_dist = stats.norm(final_analytical_mean, final_analytical_std)
ax.plot(x_range, true_post_dist.pdf(x_range), 'r-', linewidth=2,
        label='True posterior')

# Prior
ax.plot(x_range, prior.pdf(x_range), 'g--', linewidth=2,
        label='Prior', alpha=0.5)

ax.axvline(theta_true, color='orange', linestyle=':', linewidth=2,
           label='True θ')
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Final Posterior Approximation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/sis_weight_degeneracy.png',
            dpi=300, bbox_inches='tight')


# Example 2: SIS with Resampling (Particle Filter)
# ==============================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Sequential IS with Resampling")
print("=" * 70)

print("\nResampling when ESS drops below 50% threshold")

# Initialize new SIS
sis_resample = SequentialImportanceSampler(prior, proposal)
sis_resample.initialize(n_particles)

resampling_times = []

print(f"\n{'t':>3} {'ESS':>8} {'Rel ESS':>8} {'Resampled?'}")
print("-" * 40)

for t, y_t in enumerate(data_seq):
    sis_resample.update(likelihood_normal, y_t)
    ess = sis_resample.ess_history[-1]
    
    # Resample if needed
    did_resample = sis_resample.resample(threshold=0.5)
    
    if did_resample:
        resampling_times.append(t+1)
    
    if (t+1) % 5 == 0 or did_resample:
        resample_str = "YES" if did_resample else ""
        print(f"{t+1:3d} {ess:8.1f} {ess/n_particles:8.1%} {resample_str}")

print(f"\nResampled {len(resampling_times)} times at t = {resampling_times}")

# Compare ESS with and without resampling
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(range(len(sis.ess_history)), sis.ess_history, 'b-',
        linewidth=2, label='Without resampling')
ax.plot(range(len(sis_resample.ess_history)), sis_resample.ess_history, 'r-',
        linewidth=2, label='With resampling')
ax.axhline(n_particles * 0.5, color='green', linestyle='--', linewidth=1.5,
           label='Resample threshold', alpha=0.7)

for t in resampling_times:
    ax.axvline(t, color='red', linestyle=':', linewidth=1, alpha=0.3)

ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS: With vs Without Resampling', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Weight distribution comparison
ax = axes[1]
ax.hist(sis.weights * n_particles, bins=50, alpha=0.5, density=True,
        color='blue', edgecolor='black', label='Without resampling')
ax.hist(sis_resample.weights * n_particles, bins=50, alpha=0.5, density=True,
        color='red', edgecolor='black', label='With resampling')
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Weight × n', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Final Weight Distributions', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/sis_with_resampling.png',
            dpi=300, bbox_inches='tight')


# Example 3: Online Parameter Estimation
# ====================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Online Credible Intervals")
print("=" * 70)

# Track credible intervals over time
sis_ci = SequentialImportanceSampler(prior, proposal)
sis_ci.initialize(n_particles)

credible_intervals = []
posterior_means_ci = []

for y_t in data_seq:
    sis_ci.update(likelihood_normal, y_t)
    sis_ci.resample(threshold=0.5)  # With resampling
    
    mean = sis_ci.estimate(lambda x: x)
    ci_lower, ci_upper = sis_ci.credible_interval(alpha=0.95)
    
    posterior_means_ci.append(mean)
    credible_intervals.append((ci_lower, ci_upper))

# Plot credible intervals over time
fig, ax = plt.subplots(figsize=(12, 6))

t_vals = range(1, len(credible_intervals)+1)
ci_lower = [ci[0] for ci in credible_intervals]
ci_upper = [ci[1] for ci in credible_intervals]

ax.fill_between(t_vals, ci_lower, ci_upper, alpha=0.3, color='blue',
                label='95% CI')
ax.plot(t_vals, posterior_means_ci, 'b-', linewidth=2,
        label='Posterior mean')
ax.axhline(theta_true, color='red', linestyle='--', linewidth=2,
           label='True θ', alpha=0.7)
ax.set_xlabel('Number of Observations', fontsize=12)
ax.set_ylabel('θ', fontsize=12)
ax.set_title('Sequential Credible Intervals', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/sis_credible_intervals.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. SEQUENTIAL IMPORTANCE SAMPLING processes data incrementally:
   - Weights multiply over time: w_t = w_{t-1} × p(y_t|θ)
   - No need to reprocess all data
   - Useful for online/streaming applications

2. WEIGHT DEGENERACY is fundamental problem:
   - ESS decreases over time
   - Eventually few particles dominate
   - Unavoidable with fixed particles

3. DEGENERACY RATE:
   - Exponential in many cases
   - Faster for informative data
   - Slower with good proposals

4. RESAMPLING combats degeneracy:
   - Replicate high-weight particles
   - Discard low-weight particles
   - Reset weights to uniform
   - Introduces particle diversity loss

5. RESAMPLING STRATEGIES:
   - Threshold-based: resample when ESS < threshold
   - Periodic: resample every k steps
   - Adaptive: based on ESS or other metrics

6. RESAMPLING TRADE-OFFS:
   - Benefit: Maintains effective sample size
   - Cost: Particle diversity loss (sample impoverishment)
   - Cost: Introduces additional sampling variability

7. PARTICLE FILTER = SIS + Resampling:
   - Widely used in time series
   - State space models
   - Tracking applications
   - Robotics/navigation

8. WHEN TO USE SEQUENTIAL IS:
   - Data arrives sequentially
   - Online learning scenarios
   - Computational constraints (can't reprocess all data)
   - Time series analysis

9. PRACTICAL CONSIDERATIONS:
   - Monitor ESS continuously
   - Resample when ESS < 0.5n (or 0.3n)
   - Use adaptive proposals if possible
   - Consider particle MCMC for very long sequences

10. LIMITATIONS:
    - Sample impoverishment with resampling
    - Fixed particles can't explore new regions
    - Not as robust as batch methods
    - Requires careful tuning of resampling threshold

11. ALTERNATIVES/IMPROVEMENTS:
    - Adaptive proposals (better than fixed)
    - Auxiliary particle filters
    - Particle MCMC (rejuvenation)
    - Rao-Blackwellization (when possible)
""")
