"""
05_proposal_diagnostics.py

INTERMEDIATE LEVEL: Comprehensive Proposal Quality Diagnostics

This module provides tools and methods for assessing the quality of
proposal distributions in importance sampling.

Key Diagnostic Metrics:
1. Effective Sample Size (ESS)
2. Weight variance and coefficient of variation
3. Entropy and perplexity
4. Kullback-Leibler divergence estimates
5. Coverage diagnostics
6. χ² divergence measures

Mathematical Background:
---------------------
A good proposal q should:
- Cover the support of π (importance: absolute)
- Have similar shape to π (affects variance)
- Have heavier tails than π (prevents infinite variance)
- Be easy to sample from (computational)

Optimal proposal (for variance minimization):
    q*(θ) = |h(θ)|π(θ) / ∫|h(θ)|π(θ)dθ

In practice, we use diagnostics to assess how close q is to q*.

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from typing import Tuple, Dict, Callable

np.random.seed(42)
sns.set_style("whitegrid")


class ProposalDiagnostics:
    """
    Comprehensive diagnostic suite for importance sampling proposals.
    """
    
    def __init__(self, target_density: Callable, proposal_dist,
                 name: str = "Proposal"):
        """
        Parameters:
        -----------
        target_density : callable
            Target density function (can be unnormalized)
        proposal_dist : scipy.stats distribution
            Proposal distribution
        name : str
            Name for reporting
        """
        self.target_density = target_density
        self.proposal_dist = proposal_dist
        self.name = name
        
    def compute_all_diagnostics(self, n_samples: int = 5000) -> Dict:
        """
        Compute comprehensive diagnostic statistics.
        
        Returns dictionary with all metrics.
        """
        # Draw samples
        samples = self.proposal_dist.rvs(size=n_samples)
        
        # Compute weights
        target_vals = self.target_density(samples)
        proposal_vals = self.proposal_dist.pdf(samples)
        
        unnorm_weights = target_vals / (proposal_vals + 1e-300)
        weights = unnorm_weights / np.sum(unnorm_weights)
        
        # Compute all diagnostics
        diagnostics = {
            'name': self.name,
            'n_samples': n_samples,
        }
        
        # 1. ESS metrics
        diagnostics.update(self._compute_ess_metrics(weights))
        
        # 2. Weight statistics
        diagnostics.update(self._compute_weight_statistics(weights))
        
        # 3. Coverage metrics
        diagnostics.update(self._compute_coverage_metrics(samples, weights))
        
        # 4. Divergence estimates
        diagnostics.update(self._compute_divergence_metrics(samples, weights))
        
        # Store samples and weights for plotting
        diagnostics['samples'] = samples
        diagnostics['weights'] = weights
        
        return diagnostics
    
    def _compute_ess_metrics(self, weights: np.ndarray) -> Dict:
        """Effective Sample Size related metrics."""
        n = len(weights)
        
        # Standard ESS
        ess = 1.0 / np.sum(weights**2)
        
        # Relative ESS
        rel_ess = ess / n
        
        # Perplexity (exponential of entropy)
        entropy = -np.sum(weights * np.log(weights + 1e-300))
        perplexity = np.exp(entropy)
        
        return {
            'ess': ess,
            'relative_ess': rel_ess,
            'entropy': entropy,
            'perplexity': perplexity,
        }
    
    def _compute_weight_statistics(self, weights: np.ndarray) -> Dict:
        """Weight distribution statistics."""
        n = len(weights)
        
        # Basic statistics
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        # Coefficient of variation
        cv = std_weight / (mean_weight + 1e-10)
        
        # Quantiles
        quantiles = np.percentile(weights, [25, 50, 75, 90, 95, 99])
        
        # Weight concentration
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)
        
        n_for_50pct = np.searchsorted(cumsum, 0.50) + 1
        n_for_90pct = np.searchsorted(cumsum, 0.90) + 1
        
        return {
            'max_weight': max_weight,
            'min_weight': min_weight,
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'cv_weights': cv,
            'weight_q25': quantiles[0],
            'weight_q50': quantiles[1],
            'weight_q75': quantiles[2],
            'weight_q90': quantiles[3],
            'weight_q95': quantiles[4],
            'weight_q99': quantiles[5],
            'n_for_50pct_weight': n_for_50pct,
            'n_for_90pct_weight': n_for_90pct,
            'pct_for_50pct_weight': n_for_50pct / n * 100,
            'pct_for_90pct_weight': n_for_90pct / n * 100,
        }
    
    def _compute_coverage_metrics(self, samples: np.ndarray, 
                                   weights: np.ndarray) -> Dict:
        """
        Assess how well proposal covers the target.
        """
        # Effective range of samples
        weighted_mean = np.sum(weights * samples)
        weighted_var = np.sum(weights * (samples - weighted_mean)**2)
        weighted_std = np.sqrt(weighted_var)
        
        # Check coverage: are high-weight samples spread out?
        # Sort by weight
        sorted_indices = np.argsort(weights)[::-1]
        top_10pct_idx = sorted_indices[:int(0.1 * len(samples))]
        top_samples = samples[top_10pct_idx]
        
        # Spread of top samples
        top_spread = np.std(top_samples)
        
        return {
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'top_10pct_spread': top_spread,
        }
    
    def _compute_divergence_metrics(self, samples: np.ndarray,
                                     weights: np.ndarray) -> Dict:
        """
        Estimate divergences between target and proposal.
        
        Note: These are rough estimates, not exact values.
        """
        n = len(samples)
        
        # Estimate of KL(π||q) using importance weights
        # KL(π||q) ≈ E_π[log(π/q)] ≈ mean(log(w))
        log_weights = np.log(weights * n + 1e-300)
        kl_estimate = np.mean(log_weights)
        
        # Variance of log weights (related to KL divergence)
        var_log_weights = np.var(log_weights)
        
        # χ² divergence estimate
        # χ²(π||q) = E_π[(π/q - 1)²] = E_π[(w*n - 1)²]
        chi2_estimate = np.mean((weights * n - 1)**2)
        
        return {
            'kl_estimate': kl_estimate,
            'var_log_weights': var_log_weights,
            'chi2_estimate': chi2_estimate,
        }
    
    def plot_diagnostics(self, diagnostics: Dict, save_path: str = None):
        """
        Create comprehensive diagnostic plots.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        samples = diagnostics['samples']
        weights = diagnostics['weights']
        n = len(samples)
        
        # Panel 1: Weight histogram
        ax = axes[0, 0]
        ax.hist(weights * n, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2,
                   label='Uniform weight')
        ax.set_xlabel('Normalized Weight × n', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{self.name}: Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Cumulative weight
        ax = axes[0, 1]
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)
        ax.plot(np.arange(1, len(sorted_weights)+1), cumsum,
                linewidth=2, color='darkblue')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5,
                   label='50% of weight')
        ax.axhline(0.9, color='orange', linestyle='--', linewidth=1.5,
                   label='90% of weight')
        ax.set_xlabel('Number of Samples (sorted)', fontsize=11)
        ax.set_ylabel('Cumulative Weight', fontsize=11)
        ax.set_title('Weight Concentration', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Sample scatter
        ax = axes[0, 2]
        scatter = ax.scatter(samples, weights * n, c=weights * n,
                            cmap='hot', s=30, alpha=0.6,
                            edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Sample Value', fontsize=11)
        ax.set_ylabel('Weight × n', fontsize=11)
        ax.set_title('Samples vs Weights', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Q-Q plot of weights
        ax = axes[1, 0]
        theoretical_quantiles = np.linspace(0, 1, n)
        sample_quantiles = np.sort(weights * n)
        ax.plot(theoretical_quantiles, sample_quantiles, 'o',
                markersize=3, alpha=0.5)
        ax.plot([0, 1], [1, 1], 'r--', linewidth=2, label='Uniform')
        ax.set_xlabel('Theoretical Quantile', fontsize=11)
        ax.set_ylabel('Sample Quantile (Weight × n)', fontsize=11)
        ax.set_title('Q-Q Plot: Weights vs Uniform', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Diagnostic metrics text
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        DIAGNOSTIC SUMMARY
        ══════════════════════════════
        ESS: {diagnostics['ess']:.1f}
        Relative ESS: {diagnostics['relative_ess']:.1%}
        Perplexity: {diagnostics['perplexity']:.1f}
        
        CV of weights: {diagnostics['cv_weights']:.3f}
        Max weight: {diagnostics['max_weight']:.6f}
        
        50% weight in: {diagnostics['n_for_50pct_weight']} samples
                       ({diagnostics['pct_for_50pct_weight']:.1f}%)
        
        90% weight in: {diagnostics['n_for_90pct_weight']} samples
                       ({diagnostics['pct_for_90pct_weight']:.1f}%)
        
        KL estimate: {diagnostics['kl_estimate']:.4f}
        χ² estimate: {diagnostics['chi2_estimate']:.4f}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Panel 6: Weight log-histogram
        ax = axes[1, 2]
        log_weights = np.log(weights * n + 1e-10)
        ax.hist(log_weights, bins=50, density=True, alpha=0.7,
                color='darkgreen', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2,
                   label='log(1) = 0')
        ax.set_xlabel('log(Weight × n)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Log-Weight Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example 1: Comparing Multiple Proposals
# ======================================
print("=" * 70)
print("EXAMPLE 1: Systematic Proposal Comparison")
print("=" * 70)

# Target distribution: N(5, 1.5)
target_mean, target_std = 5.0, 1.5
target = stats.norm(target_mean, target_std)

# Define multiple proposals to compare
proposals = {
    'Perfect': stats.norm(5.0, 1.5),          # Same as target
    'Good': stats.norm(5.0, 1.8),             # Slightly wider
    'Decent': stats.norm(4.5, 2.0),           # Shifted mean, wider
    'Mediocre': stats.norm(3.0, 2.5),         # More shifted
    'Poor': stats.norm(5.0, 0.8),             # Too narrow
    'Bad': stats.norm(0.0, 1.5),              # Wrong location
}

print("\nAnalyzing proposals for target N(5, 1.5):\n")

results = []
for name, proposal in proposals.items():
    diagnostics_obj = ProposalDiagnostics(target.pdf, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    results.append(diag)

# Print comparison table
print(f"{'Proposal':<12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'KL Est':>10} "
      f"{'50% in':>8}")
print("-" * 70)

for diag in results:
    print(f"{diag['name']:<12} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['cv_weights']:8.3f} {diag['kl_estimate']:10.4f} "
          f"{diag['pct_for_50pct_weight']:7.1f}%")

# Create detailed plots for selected proposals
selected_proposals = ['Perfect', 'Decent', 'Poor']
for name in selected_proposals:
    diag = [d for d in results if d['name'] == name][0]
    prop = proposals[name]
    
    diagnostics_obj = ProposalDiagnostics(target.pdf, prop, name)
    diagnostics_obj.plot_diagnostics(
        diag,
        save_path=f'/home/claude/03_Importance_Sampling/diagnostics_{name.lower()}.png'
    )
    print(f"\nSaved diagnostics plot for {name} proposal")


# Example 2: Progressive Deterioration
# ==================================
print("\n" + "=" * 70)
print("EXAMPLE 2: How Proposals Degrade")
print("=" * 70)

# Target: N(0, 1)
target_ex2 = stats.norm(0, 1)

# Create proposals with increasing mismatch
mean_shifts = [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
scale = 1.2  # Slightly wider than target

print(f"\nProposal: N(μ, {scale}²), varying μ from 0 to 5")
print(f"Target: N(0, 1)\n")

print(f"{'Mean Shift':>12} {'ESS':>8} {'Rel ESS':>8} {'CV':>8} {'Max Wt':>10}")
print("-" * 60)

ess_values = []
for shift in mean_shifts:
    proposal = stats.norm(shift, scale)
    
    diagnostics_obj = ProposalDiagnostics(target_ex2.pdf, proposal, f"μ={shift}")
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=2000)
    
    ess_values.append(diag['ess'])
    
    print(f"{shift:12.1f} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['cv_weights']:8.3f} {diag['max_weight']:10.6f}")

# Plot ESS degradation
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mean_shifts, ess_values, 'o-', linewidth=2, markersize=10,
        color='steelblue')
ax.set_xlabel('Proposal Mean Shift', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS Degradation with Proposal-Target Mismatch',
             fontsize=13, fontweight='bold')
ax.axhline(2000, color='red', linestyle='--', linewidth=2,
           label='n samples', alpha=0.7)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/ess_degradation.png',
            dpi=300, bbox_inches='tight')


# Example 3: Tail Coverage Diagnostic
# =================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Tail Coverage Assessment")
print("=" * 70)

print("""
For heavy-tailed targets, proposals must have adequate tail coverage.
We compare Student-t target with different proposal tail behaviors.
""")

# Target: Student-t with 3 degrees of freedom (heavy tails)
target_t = stats.t(df=3, loc=0, scale=1)

# Proposals with different tail behavior
tail_proposals = {
    'Heavy: t(3)': stats.t(df=3, loc=0, scale=1.2),
    'Medium: t(5)': stats.t(df=5, loc=0, scale=1.2),
    'Light: Normal': stats.norm(0, 1.5),
}

print("\nTarget: Student-t(df=3)")
print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'χ² Est':>10} {'Status'}")
print("-" * 65)

for name, proposal in tail_proposals.items():
    diagnostics_obj = ProposalDiagnostics(target_t.pdf, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    
    # Check if any weights are suspiciously large (tail coverage issue)
    max_weight_ratio = diag['max_weight'] * diag['n_samples']
    status = "✓ Good" if max_weight_ratio < 10 else "⚠ Poor tails"
    
    print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['chi2_estimate']:10.2f} {status}")

print("\nKey insight: Proposals with lighter tails than target can fail!")


# Example 4: Multimodal Target Diagnostics
# ======================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Diagnosing Coverage for Multimodal Target")
print("=" * 70)

# Bimodal target: 0.5*N(-3,1) + 0.5*N(3,1)
def bimodal_density(x):
    return 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)

# Proposals with different strategies
multimodal_proposals = {
    'Single wide': stats.norm(0, 4),          # Single component, very wide
    'Single narrow': stats.norm(0, 1.5),      # Single component, too narrow
    'Centered on mode': stats.norm(-3, 1.5),  # Only covers one mode
}

print("\nTarget: 0.5*N(-3,1) + 0.5*N(3,1) [bimodal]")
print(f"{'Proposal':<20} {'ESS':>8} {'Rel ESS':>8} {'50% Wt%':>10} {'Coverage'}")
print("-" * 70)

for name, proposal in multimodal_proposals.items():
    diagnostics_obj = ProposalDiagnostics(bimodal_density, proposal, name)
    diag = diagnostics_obj.compute_all_diagnostics(n_samples=3000)
    
    # Check mode coverage
    samples = diag['samples']
    left_mode = np.sum((samples < -1) & (samples > -5))
    right_mode = np.sum((samples < 5) & (samples > 1))
    
    if left_mode > 50 and right_mode > 50:
        coverage = "✓ Both modes"
    else:
        coverage = "⚠ Missing mode"
    
    print(f"{name:<20} {diag['ess']:8.1f} {diag['relative_ess']:8.1%} "
          f"{diag['pct_for_50pct_weight']:9.1f}% {coverage}")

print("\nKey insight: Single-component proposals struggle with multimodal targets!")

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. ESSENTIAL DIAGNOSTICS:
   - ESS and relative ESS (most important)
   - Weight coefficient of variation
   - Weight concentration (50%, 90% percentiles)
   - Maximum weight value

2. GOOD PROPOSAL INDICATORS:
   - Relative ESS > 0.1 (ideally > 0.3)
   - CV of weights < 3 (lower is better)
   - < 10% of samples for 50% of weight
   - No individual weight dominates

3. WARNING SIGNS:
   - ESS << n (< 1% of n)
   - Few samples carry most weight
   - Very high CV (> 5-10)
   - Large KL or χ² divergence estimates

4. TAIL COVERAGE:
   - Critical for heavy-tailed targets
   - Check max weight ratios
   - Proposal should have heavier tails
   - Light-tailed proposals can fail badly

5. MULTIMODAL TARGETS:
   - Single components often miss modes
   - Check sample coverage of all modes
   - Mixture proposals usually needed
   - Or use adaptive methods

6. DETERIORATION PATTERNS:
   - ESS drops as proposal-target mismatch grows
   - Often exponential degradation with distance
   - Small shifts can have large effects

7. PRACTICAL WORKFLOW:
   a) Always compute diagnostics before trusting results
   b) Compare multiple proposals if possible
   c) Look for warning signs (ESS, CV, concentration)
   d) Visualize weight distribution
   e) Check coverage of important regions

8. REMEDIES FOR POOR DIAGNOSTICS:
   - Adjust proposal parameters (mean, variance)
   - Use heavier-tailed proposal family
   - Switch to mixture proposal
   - Consider adaptive IS
   - Or use MCMC instead
""")
