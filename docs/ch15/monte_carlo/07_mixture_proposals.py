"""
07_mixture_proposals.py

INTERMEDIATE LEVEL: Mixture Distributions as Importance Sampling Proposals

This module explores using mixture distributions as proposals for
importance sampling, particularly effective for:
- Multimodal target distributions
- Complex, non-standard shapes
- Improved coverage and ESS

Mathematical Foundation:
---------------------
Mixture proposal:
    q(θ) = Σⱼ αⱼ qⱼ(θ)

where:
- qⱼ(θ) are component distributions
- αⱼ are mixture weights (Σⱼ αⱼ = 1, αⱼ ≥ 0)
- K is the number of components

Sampling from mixture:
1. Sample component j with probability αⱼ
2. Sample θ ~ qⱼ(θ)

Evaluating density:
    q(θ) = Σⱼ αⱼ qⱼ(θ)

Advantages:
- Can approximate complex shapes
- Naturally handles multimodal targets
- Flexible and expressive
- Each component can be simple (e.g., Gaussian)

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from typing import List, Callable

np.random.seed(42)
sns.set_style("whitegrid")


class MixtureProposal:
    """
    Mixture distribution proposal for importance sampling.
    """
    
    def __init__(self, components: List, weights: np.ndarray):
        """
        Parameters:
        -----------
        components : list of scipy.stats distributions
            Component distributions
        weights : array
            Mixture weights (will be normalized)
        """
        self.components = components
        self.weights = np.array(weights) / np.sum(weights)
        self.n_components = len(components)
        
    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Sample from mixture distribution.
        
        Algorithm:
        1. For each sample:
           a) Draw component j with probability αⱼ
           b) Draw sample from qⱼ
        """
        # Sample component indices
        component_indices = np.random.choice(
            self.n_components,
            size=size,
            p=self.weights
        )
        
        # Sample from selected components
        samples = []
        for idx in component_indices:
            sample = self.components[idx].rvs()
            samples.append(sample)
        
        return np.array(samples)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate mixture density: q(x) = Σⱼ αⱼ qⱼ(x)
        """
        x = np.atleast_1d(x)
        density = np.zeros(len(x))
        
        for weight, component in zip(self.weights, self.components):
            density += weight * component.pdf(x)
        
        return density
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate log mixture density using log-sum-exp for stability.
        
        log q(x) = log(Σⱼ αⱼ qⱼ(x))
                 = log-sum-exp(log αⱼ + log qⱼ(x))
        """
        x = np.atleast_1d(x)
        
        log_densities = []
        for weight, component in zip(self.weights, self.components):
            log_densities.append(np.log(weight) + component.logpdf(x))
        
        return logsumexp(log_densities, axis=0)


def importance_sampling_mixture(target_density: Callable,
                                 proposal: MixtureProposal,
                                 h_function: Callable,
                                 n_samples: int) -> tuple:
    """
    Importance sampling with mixture proposal.
    
    Returns:
    --------
    estimate : float
    samples : array
    weights : array (normalized)
    ess : float
    """
    # Sample from mixture proposal
    samples = proposal.rvs(size=n_samples)
    
    # Compute importance weights
    target_vals = target_density(samples)
    proposal_vals = proposal.pdf(samples)
    
    unnorm_weights = target_vals / (proposal_vals + 1e-300)
    weights = unnorm_weights / np.sum(unnorm_weights)
    
    # Estimate
    estimate = np.sum(weights * h_function(samples))
    
    # ESS
    ess = 1.0 / np.sum(weights**2)
    
    return estimate, samples, weights, ess


# Example 1: Bimodal Target
# =======================
print("=" * 70)
print("EXAMPLE 1: Bimodal Target Distribution")
print("=" * 70)

# Target: mixture of two Gaussians
# 0.3 N(-2, 0.8) + 0.7 N(3, 1.2)
def bimodal_target(x):
    """Bimodal target density"""
    return (0.3 * stats.norm.pdf(x, -2, 0.8) +
            0.7 * stats.norm.pdf(x, 3, 1.2))

# Function to estimate: E[θ]
h_mean = lambda x: x

# True expectation
x_grid = np.linspace(-6, 8, 10000)
true_mean = np.trapz(x_grid * bimodal_target(x_grid), x_grid)

print(f"\nTarget: 0.3 N(-2, 0.8) + 0.7 N(3, 1.2)")
print(f"True mean: {true_mean:.6f}")

# Proposal 1: Single wide Gaussian (poor for bimodal)
proposal_single = stats.norm(0, 3)

n_samples = 3000

samples_single = proposal_single.rvs(size=n_samples)
weights_single_unnorm = bimodal_target(samples_single) / proposal_single.pdf(samples_single)
weights_single = weights_single_unnorm / np.sum(weights_single_unnorm)
estimate_single = np.sum(weights_single * h_mean(samples_single))
ess_single = 1.0 / np.sum(weights_single**2)

print(f"\nSingle Gaussian Proposal N(0, 3):")
print(f"  Estimate: {estimate_single:.6f}")
print(f"  Error: {abs(estimate_single - true_mean):.6f}")
print(f"  ESS: {ess_single:.1f} ({ess_single/n_samples:.1%})")

# Proposal 2: Mixture matching target structure
components_matched = [
    stats.norm(-2, 1.0),   # Cover left mode
    stats.norm(3, 1.5),    # Cover right mode
]
weights_matched = [0.3, 0.7]  # Match target weights

proposal_mixture_matched = MixtureProposal(components_matched, weights_matched)

estimate_matched, samples_matched, weights_matched_norm, ess_matched = \
    importance_sampling_mixture(bimodal_target, proposal_mixture_matched, 
                                h_mean, n_samples)

print(f"\nMixture Proposal (matched to target):")
print(f"  Components: 0.3 N(-2,1) + 0.7 N(3,1.5)")
print(f"  Estimate: {estimate_matched:.6f}")
print(f"  Error: {abs(estimate_matched - true_mean):.6f}")
print(f"  ESS: {ess_matched:.1f} ({ess_matched/n_samples:.1%})")

# Proposal 3: Equal-weighted mixture
components_equal = [
    stats.norm(-2, 1.0),
    stats.norm(3, 1.5),
]
weights_equal = [0.5, 0.5]  # Equal weights (not optimal)

proposal_mixture_equal = MixtureProposal(components_equal, weights_equal)

estimate_equal, samples_equal, weights_equal_norm, ess_equal = \
    importance_sampling_mixture(bimodal_target, proposal_mixture_equal,
                                h_mean, n_samples)

print(f"\nMixture Proposal (equal weights):")
print(f"  Components: 0.5 N(-2,1) + 0.5 N(3,1.5)")
print(f"  Estimate: {estimate_equal:.6f}")
print(f"  Error: {abs(estimate_equal - true_mean):.6f}")
print(f"  ESS: {ess_equal:.1f} ({ess_equal/n_samples:.1%})")

print(f"\nImprovement over single Gaussian:")
print(f"  Matched mixture: {ess_matched/ess_single:.2f}x better ESS")
print(f"  Equal mixture: {ess_equal/ess_single:.2f}x better ESS")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Target and proposals
ax = axes[0, 0]
x_plot = np.linspace(-6, 8, 1000)
ax.plot(x_plot, bimodal_target(x_plot), 'k-', linewidth=3,
        label='Target', alpha=0.7)
ax.plot(x_plot, proposal_single.pdf(x_plot), 'r--', linewidth=2,
        label='Single N(0,3)')
ax.plot(x_plot, proposal_mixture_matched.pdf(x_plot), 'b:', linewidth=2,
        label='Mixture (matched)')
ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Target vs Proposals', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Single Gaussian samples
ax = axes[0, 1]
ax.hist(samples_single, bins=50, density=True, alpha=0.5,
        color='steelblue', edgecolor='black')
ax.plot(x_plot, bimodal_target(x_plot), 'r-', linewidth=2,
        label='Target')
scatter = ax.scatter(samples_single, np.zeros(len(samples_single)),
                    c=weights_single*n_samples, cmap='hot',
                    s=20, alpha=0.6)
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Single Gaussian: ESS={ess_single:.0f}',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Panel 3: Mixture samples (matched)
ax = axes[1, 0]
ax.hist(samples_matched, bins=50, density=True, alpha=0.5,
        color='green', edgecolor='black')
ax.plot(x_plot, bimodal_target(x_plot), 'r-', linewidth=2,
        label='Target')
scatter = ax.scatter(samples_matched, np.zeros(len(samples_matched)),
                    c=weights_matched_norm*n_samples, cmap='hot',
                    s=20, alpha=0.6)
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Mixture (matched): ESS={ess_matched:.0f}',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Panel 4: ESS comparison
ax = axes[1, 1]
proposals = ['Single\nGaussian', 'Equal\nMixture', 'Matched\nMixture']
ess_values = [ess_single, ess_equal, ess_matched]
colors = ['red', 'orange', 'green']
bars = ax.bar(proposals, ess_values, color=colors, alpha=0.7,
              edgecolor='black', linewidth=2)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS Comparison', fontsize=13, fontweight='bold')
ax.axhline(n_samples, color='blue', linestyle='--', linewidth=2,
          label='n samples', alpha=0.7)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

# Add value labels
for bar, ess in zip(bars, ess_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{ess:.0f}\n({ess/n_samples:.1%})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/mixture_bimodal.png',
            dpi=300, bbox_inches='tight')


# Example 2: Trimodal Target
# ========================
print("\n" + "=" * 70)
print("EXAMPLE 2: Trimodal Target Distribution")
print("=" * 70)

# Target: three modes
def trimodal_target(x):
    """Three well-separated modes"""
    return (0.2 * stats.norm.pdf(x, -4, 0.6) +
            0.5 * stats.norm.pdf(x, 0, 0.8) +
            0.3 * stats.norm.pdf(x, 5, 0.7))

# True mean
true_mean_tri = np.trapz(x_grid * trimodal_target(x_grid), x_grid)

print(f"\nTarget: 0.2 N(-4,0.6) + 0.5 N(0,0.8) + 0.3 N(5,0.7)")
print(f"True mean: {true_mean_tri:.6f}")

# Compare different numbers of mixture components
n_samples_tri = 4000

# 1 component (single Gaussian)
proposal_k1 = stats.norm(0, 4)
samples_k1 = proposal_k1.rvs(size=n_samples_tri)
weights_k1_unnorm = trimodal_target(samples_k1) / proposal_k1.pdf(samples_k1)
weights_k1 = weights_k1_unnorm / np.sum(weights_k1_unnorm)
ess_k1 = 1.0 / np.sum(weights_k1**2)

# 2 components
components_k2 = [stats.norm(-4, 1.0), stats.norm(2.5, 3.0)]
weights_k2_mix = [0.3, 0.7]
proposal_k2 = MixtureProposal(components_k2, weights_k2_mix)
_, _, weights_k2, ess_k2 = importance_sampling_mixture(
    trimodal_target, proposal_k2, h_mean, n_samples_tri
)

# 3 components (matching structure)
components_k3 = [
    stats.norm(-4, 0.8),
    stats.norm(0, 1.0),
    stats.norm(5, 0.9),
]
weights_k3_mix = [0.2, 0.5, 0.3]
proposal_k3 = MixtureProposal(components_k3, weights_k3_mix)
_, _, weights_k3, ess_k3 = importance_sampling_mixture(
    trimodal_target, proposal_k3, h_mean, n_samples_tri
)

# 5 components (overparameterized)
components_k5 = [
    stats.norm(-4, 0.8),
    stats.norm(-2, 0.8),
    stats.norm(0, 1.0),
    stats.norm(3, 1.0),
    stats.norm(5, 0.9),
]
weights_k5_mix = [0.15, 0.15, 0.4, 0.15, 0.15]
proposal_k5 = MixtureProposal(components_k5, weights_k5_mix)
_, _, weights_k5, ess_k5 = importance_sampling_mixture(
    trimodal_target, proposal_k5, h_mean, n_samples_tri
)

print(f"\nESS vs Number of Components (n={n_samples_tri}):")
print(f"{'K':>3} {'ESS':>8} {'Efficiency':>12} {'Description'}")
print("-" * 55)
print(f"  1 {ess_k1:8.1f} {ess_k1/n_samples_tri:11.1%} Single Gaussian")
print(f"  2 {ess_k2:8.1f} {ess_k2/n_samples_tri:11.1%} Two components")
print(f"  3 {ess_k3:8.1f} {ess_k3/n_samples_tri:11.1%} Three (matched)")
print(f"  5 {ess_k5:8.1f} {ess_k5/n_samples_tri:11.1%} Five (overfit)")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_plot_tri = np.linspace(-7, 8, 1000)

for ax, k, proposal, ess in [(axes[0,0], 1, proposal_k1, ess_k1),
                              (axes[0,1], 2, proposal_k2, ess_k2),
                              (axes[1,0], 3, proposal_k3, ess_k3),
                              (axes[1,1], 5, proposal_k5, ess_k5)]:
    ax.plot(x_plot_tri, trimodal_target(x_plot_tri), 'r-',
            linewidth=3, label='Target', alpha=0.7)
    
    if k == 1:
        ax.plot(x_plot_tri, proposal.pdf(x_plot_tri), 'b--',
                linewidth=2, label='Proposal')
    else:
        ax.plot(x_plot_tri, proposal.pdf(x_plot_tri), 'b--',
                linewidth=2, label='Mixture proposal')
    
    ax.set_xlabel('θ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'K={k} components: ESS={ess:.0f} ({ess/n_samples_tri:.1%})',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/mixture_components.png',
            dpi=300, bbox_inches='tight')


# Example 3: Defensive Mixture (Robustness)
# =======================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Defensive Mixture for Robustness")
print("=" * 70)

print("""
Defensive importance sampling mixes a targeted proposal with a 
broad "safety" component to ensure coverage:

    q_def(θ) = α q_target(θ) + (1-α) q_safe(θ)

where:
- q_target: focused on high-probability regions
- q_safe: broad coverage (e.g., prior or flat distribution)
- α ∈ (0,1): trade-off parameter (typically α ≈ 0.7-0.9)
""")

# Target: bimodal (same as Example 1)
alpha_defensive = 0.8  # Weight on targeted component

# Targeted component: mixture covering both modes
components_targeted = [stats.norm(-2, 0.8), stats.norm(3, 1.2)]
weights_targeted = [0.3, 0.7]
proposal_targeted = MixtureProposal(components_targeted, weights_targeted)

# Safety component: broad Gaussian
proposal_safe = stats.norm(0, 5)

# Defensive mixture
components_defensive = [proposal_targeted, proposal_safe]

# Custom mixture that treats one component as a mixture itself
class DefensiveMixture:
    def __init__(self, targeted_mixture, safe_dist, alpha):
        self.targeted = targeted_mixture
        self.safe = safe_dist
        self.alpha = alpha
    
    def rvs(self, size):
        samples = []
        for _ in range(size):
            if np.random.rand() < self.alpha:
                samples.append(self.targeted.rvs(1)[0])
            else:
                samples.append(self.safe.rvs())
        return np.array(samples)
    
    def pdf(self, x):
        return self.alpha * self.targeted.pdf(x) + (1 - self.alpha) * self.safe.pdf(x)

proposal_defensive = DefensiveMixture(proposal_targeted, proposal_safe,
                                      alpha_defensive)

estimate_def, samples_def, weights_def, ess_def = \
    importance_sampling_mixture(bimodal_target, proposal_defensive,
                                h_mean, n_samples)

print(f"\nDefensive Mixture (α={alpha_defensive}):")
print(f"  Estimate: {estimate_def:.6f}")
print(f"  Error: {abs(estimate_def - true_mean):.6f}")
print(f"  ESS: {ess_def:.1f} ({ess_def/n_samples:.1%})")

# Compare with pure targeted
estimate_pure, _, weights_pure, ess_pure = \
    importance_sampling_mixture(bimodal_target, proposal_targeted,
                                h_mean, n_samples)

print(f"\nPure Targeted Mixture:")
print(f"  ESS: {ess_pure:.1f} ({ess_pure/n_samples:.1%})")

print(f"\nTrade-off:")
print(f"  Defensive ESS: {ess_def:.1f} ({ess_def/ess_pure:.0%} of pure)")
print(f"  But guarantees minimum ESS even if target shape is wrong")

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. MIXTURE PROPOSALS are powerful for complex targets:
   - Can approximate arbitrary shapes
   - Natural for multimodal distributions
   - Flexible and expressive

2. DESIGNING MIXTURES:
   - Place components at target modes
   - Match mixture weights to mode weights (approximately)
   - Use slightly wider components than target peaks
   - Typically need K = number of modes

3. NUMBER OF COMPONENTS:
   - Too few: Miss important regions, low ESS
   - Right amount: Cover all modes, high ESS
   - Too many: Computational overhead, marginal ESS gain
   - Start with K = suspected modes, then adjust

4. MIXTURE WEIGHTS:
   - Should roughly match target mode weights
   - Equal weights work but may be suboptimal
   - Can estimate from preliminary samples

5. COMPONENT PLACEMENT:
   - Center components at approximate mode locations
   - Can find modes via optimization or clustering
   - Preliminary exploration helpful

6. DEFENSIVE MIXTURES:
   - Mix targeted + broad safety component
   - q(θ) = α q_target(θ) + (1-α) q_safe(θ)
   - Typical α ≈ 0.7-0.9
   - Trades ESS for robustness
   - Guarantees minimum ESS

7. ADVANTAGES:
   - Much better than single component for multimodal
   - Can achieve 5-10x ESS improvement
   - Flexible enough for complex shapes
   - Components can be simple (e.g., Gaussians)

8. PRACTICAL TIPS:
   - Start with 2-3 components
   - Use exploratory samples to locate modes
   - Check that ESS improves vs single component
   - Monitor weight concentration
   - Consider defensive variant for robustness

9. WHEN TO USE:
   - Multimodal targets (essential)
   - Complex non-standard shapes
   - When single component gives poor ESS
   - Bayesian inference with multiple modes

10. COMPUTATIONAL COST:
    - Sampling: O(K) overhead
    - Density evaluation: O(K) cost
    - Usually worth it for ESS improvement
    - Can parallelize component sampling
""")
