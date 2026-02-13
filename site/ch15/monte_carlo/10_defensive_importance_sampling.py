"""
10_defensive_importance_sampling.py

ADVANCED LEVEL: Defensive Importance Sampling

This module implements defensive importance sampling, a robust approach
that guarantees bounded variance by mixing a targeted proposal with
a defensive component.

Mathematical Foundation:
---------------------
Standard IS can have unbounded or infinite variance if the proposal
has lighter tails than the target or misses important regions.

Defensive Mixture:
    q_def(θ) = α q(θ) + (1-α) m(θ)

where:
- q(θ): targeted proposal (focused on high-probability regions)
- m(θ): defensive component (broad, safe coverage)
- α ∈ (0,1): mixture parameter

Typically m(θ) is chosen as:
- Prior p(θ)
- Uniform over support
- Very broad Gaussian/Student-t

Theoretical Guarantee:
If m(θ) ≥ c·π(θ) for some c > 0 (i.e., m has heavier tails),
then Var[h(θ)w_def(θ)] is bounded.

Trade-off:
- Higher α: better ESS, less robust
- Lower α: worse ESS, more robust
- Typical choice: α ∈ [0.7, 0.9]

Key Advantage: BOUNDED VARIANCE GUARANTEE

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


class DefensiveMixture:
    """
    Defensive mixture proposal for robust importance sampling.
    """
    
    def __init__(self, targeted_dist, defensive_dist, alpha):
        """
        Parameters:
        -----------
        targeted_dist : scipy.stats distribution
            Focused proposal for main probability mass
        defensive_dist : scipy.stats distribution
            Broad proposal for robust coverage
        alpha : float in (0,1)
            Weight on targeted component
        """
        self.targeted = targeted_dist
        self.defensive = defensive_dist
        self.alpha = alpha
        
    def rvs(self, size=1):
        """Sample from mixture."""
        samples = []
        for _ in range(size):
            if np.random.rand() < self.alpha:
                samples.append(self.targeted.rvs())
            else:
                samples.append(self.defensive.rvs())
        return np.array(samples)
    
    def pdf(self, x):
        """Evaluate mixture density."""
        return (self.alpha * self.targeted.pdf(x) +
                (1 - self.alpha) * self.defensive.pdf(x))
    
    def logpdf(self, x):
        """Evaluate log mixture density."""
        log_targeted = np.log(self.alpha) + self.targeted.logpdf(x)
        log_defensive = np.log(1 - self.alpha) + self.defensive.logpdf(x)
        return logsumexp([log_targeted, log_defensive], axis=0)


def compare_proposals(target_density, proposals_dict, h_function,
                     n_samples=5000, n_replications=100):
    """
    Compare multiple proposals via repeated trials.
    
    Returns statistics: mean ESS, std ESS, mean estimate, std estimate
    """
    results = {}
    
    for name, proposal in proposals_dict.items():
        ess_list = []
        estimates = []
        
        for _ in range(n_replications):
            # Sample
            samples = proposal.rvs(size=n_samples)
            
            # Weights
            weights_unnorm = target_density(samples) / proposal.pdf(samples)
            weights = weights_unnorm / np.sum(weights_unnorm)
            
            # ESS
            ess = 1.0 / np.sum(weights**2)
            ess_list.append(ess)
            
            # Estimate
            estimate = np.sum(weights * h_function(samples))
            estimates.append(estimate)
        
        results[name] = {
            'mean_ess': np.mean(ess_list),
            'std_ess': np.std(ess_list),
            'min_ess': np.min(ess_list),
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
        }
    
    return results


# Example 1: Heavy-Tailed Target
# ============================
print("=" * 70)
print("EXAMPLE 1: Heavy-Tailed Target Distribution")
print("=" * 70)

print("""
Target: Student-t(df=3) - heavy tails
Proposals:
1. Gaussian (risky - light tails)
2. Student-t (targeted but still risky)
3. Defensive mixture: α × t(3) + (1-α) × t(1) (Cauchy)
""")

# Target: Student-t with 3 df
target_t3 = stats.t(df=3, loc=0, scale=1)

# Function to estimate: E[θ²]
h_square = lambda x: x**2

# True value
x_grid = np.linspace(target_t3.ppf(0.001), target_t3.ppf(0.999), 10000)
true_value = np.trapz(x_grid**2 * target_t3.pdf(x_grid), x_grid)

print(f"\nTrue E[θ²] = {true_value:.6f}")

# Proposals
proposals_ex1 = {
    'Gaussian (risky)': stats.norm(0, 1.5),
    'Student-t(3) (risky)': stats.t(df=3, loc=0, scale=1.2),
    'Defensive α=0.8': DefensiveMixture(
        stats.t(df=3, loc=0, scale=1.2),  # Targeted
        stats.t(df=1, loc=0, scale=2),     # Cauchy (very heavy tails)
        alpha=0.8
    ),
    'Defensive α=0.9': DefensiveMixture(
        stats.t(df=3, loc=0, scale=1.2),
        stats.t(df=1, loc=0, scale=2),
        alpha=0.9
    ),
}

# Compare proposals
print("\nComparing proposals (100 replications, 3000 samples each):")
results_ex1 = compare_proposals(target_t3.pdf, proposals_ex1, h_square,
                                 n_samples=3000, n_replications=100)

print(f"\n{'Proposal':<22} {'Mean ESS':>10} {'Min ESS':>10} {'Std Est':>10} {'Robust?'}")
print("-" * 70)

for name, stats_dict in results_ex1.items():
    # Check if minimum ESS is reasonable (> 1% of n)
    robust = "✓" if stats_dict['min_ess'] > 30 else "✗"
    
    print(f"{name:<22} {stats_dict['mean_ess']:10.1f} "
          f"{stats_dict['min_ess']:10.1f} {stats_dict['std_estimate']:10.4f} {robust}")

print("\nKey insight: Defensive proposals have higher minimum ESS!")


# Example 2: Misspecified Proposal
# ==============================
print("\n" + "=" * 70)
print("EXAMPLE 2: Robustness to Proposal Misspecification")
print("=" * 70)

print("""
Scenario: We think target is N(0,1) but it's actually N(3,1)
          (proposal misspecified - wrong location)

Defensive mixing saves us from complete failure.
""")

# True target (unknown to us)
target_true = stats.norm(3, 1)

# Our (wrong) belief about target
target_belief = stats.norm(0, 1)

# Proposals based on wrong belief
proposals_ex2 = {
    'Wrong belief N(0,1)': stats.norm(0, 1.2),
    'Defensive α=0.7': DefensiveMixture(
        stats.norm(0, 1.2),      # Based on wrong belief
        stats.norm(0, 5),         # Very broad safety net
        alpha=0.7
    ),
    'Defensive α=0.9': DefensiveMixture(
        stats.norm(0, 1.2),
        stats.norm(0, 5),
        alpha=0.9
    ),
}

# True mean
true_mean = 3.0

print(f"\nTrue target: N(3, 1)")
print(f"Our belief: N(0, 1) [WRONG!]")
print(f"True mean: {true_mean:.1f}")

# Compare
results_ex2 = compare_proposals(target_true.pdf, proposals_ex2,
                                lambda x: x, n_samples=2000,
                                n_replications=200)

print(f"\n{'Proposal':<22} {'Mean Est':>10} {'Bias':>10} {'RMSE':>10} {'Min ESS':>10}")
print("-" * 70)

for name, stats_dict in results_ex2.items():
    bias = stats_dict['mean_estimate'] - true_mean
    rmse = np.sqrt(bias**2 + stats_dict['std_estimate']**2)
    
    print(f"{name:<22} {stats_dict['mean_estimate']:10.4f} "
          f"{bias:+10.4f} {rmse:10.4f} {stats_dict['min_ess']:10.1f}")

print("\nKey insight: Defensive proposals are more robust to misspecification!")


# Example 3: Varying Alpha
# ======================
print("\n" + "=" * 70)
print("EXAMPLE 3: Effect of Mixture Parameter α")
print("=" * 70)

# Target: bimodal
def bimodal(x):
    return 0.4 * stats.norm.pdf(x, -2, 0.8) + 0.6 * stats.norm.pdf(x, 3, 1)

# Targeted proposal (only covers one mode - deliberately poor)
targeted_poor = stats.norm(3, 1.2)

# Defensive component (covers both modes)
defensive_broad = stats.norm(0, 4)

# Try different α values
alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

print(f"\nVarying α (mixture weight on targeted component):")
print(f"{'α':>6} {'Mean ESS':>10} {'Min ESS':>10} {'Std ESS':>10} {'Efficiency':>12}")
print("-" * 60)

ess_by_alpha = []
for alpha in alphas:
    proposal = DefensiveMixture(targeted_poor, defensive_broad, alpha)
    
    # Multiple trials
    ess_trials = []
    for _ in range(50):
        samples = proposal.rvs(size=2000)
        weights_unnorm = bimodal(samples) / proposal.pdf(samples)
        weights = weights_unnorm / np.sum(weights_unnorm)
        ess = 1.0 / np.sum(weights**2)
        ess_trials.append(ess)
    
    mean_ess = np.mean(ess_trials)
    min_ess = np.min(ess_trials)
    std_ess = np.std(ess_trials)
    
    ess_by_alpha.append((alpha, mean_ess, min_ess, std_ess))
    
    print(f"{alpha:6.2f} {mean_ess:10.1f} {min_ess:10.1f} "
          f"{std_ess:10.1f} {mean_ess/2000:11.1%}")

# Visualize trade-off
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

alphas_plot = [x[0] for x in ess_by_alpha]
mean_ess_plot = [x[1] for x in ess_by_alpha]
min_ess_plot = [x[2] for x in ess_by_alpha]

ax = axes[0]
ax.plot(alphas_plot, mean_ess_plot, 'bo-', linewidth=2, markersize=8,
        label='Mean ESS')
ax.plot(alphas_plot, min_ess_plot, 'r^--', linewidth=2, markersize=8,
        label='Min ESS (over 50 trials)')
ax.set_xlabel('α (weight on targeted component)', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('ESS vs Mixture Parameter α', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axvline(0.8, color='green', linestyle=':', linewidth=2,
           label='Typical choice', alpha=0.7)

# Visualize proposals for different α
ax = axes[1]
x_plot = np.linspace(-6, 8, 1000)
ax.plot(x_plot, bimodal(x_plot), 'k-', linewidth=3,
        label='Target', alpha=0.7)

for alpha_vis in [0.5, 0.8, 0.95]:
    proposal_vis = DefensiveMixture(targeted_poor, defensive_broad, alpha_vis)
    ax.plot(x_plot, proposal_vis.pdf(x_plot), '--', linewidth=2,
            label=f'α={alpha_vis}', alpha=0.7)

ax.set_xlabel('θ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Proposals for Different α', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/defensive_alpha_tradeoff.png',
            dpi=300, bbox_inches='tight')


# Example 4: Bayesian Inference with Defensive IS
# =============================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Defensive IS for Bayesian Inference")
print("=" * 70)

print("""
Model: y ~ N(θ, 1), n=30 observations
Prior: θ ~ N(0, 5) [vague]
Proposal: Laplace approximation (targeted) + Prior (defensive)
""")

# Generate data
theta_true_ex4 = 4.0
sigma_ex4 = 1.0
n_obs_ex4 = 30
data_ex4 = np.random.normal(theta_true_ex4, sigma_ex4, n_obs_ex4)

print(f"\nTrue θ = {theta_true_ex4}")
print(f"Data: n={n_obs_ex4}, sample mean = {np.mean(data_ex4):.3f}")

# Prior
prior_ex4 = stats.norm(0, 5)

# Unnormalized posterior
def log_posterior_ex4(theta):
    log_lik = -0.5 * np.sum((data_ex4 - theta)**2) / sigma_ex4**2
    log_prior = prior_ex4.logpdf(theta)
    return log_lik + log_prior

def posterior_ex4(theta):
    return np.exp(log_posterior_ex4(theta))

# Laplace approximation (targeted component)
# Posterior is Gaussian in this case
tau_n_sq = 1.0 / (1.0/25 + n_obs_ex4/sigma_ex4**2)
mu_n = tau_n_sq * (0/25 + np.sum(data_ex4)/sigma_ex4**2)

laplace_ex4 = stats.norm(mu_n, np.sqrt(tau_n_sq))

print(f"\nLaplace approximation: N({mu_n:.3f}, {np.sqrt(tau_n_sq):.3f})")

# Defensive proposal
alpha_ex4 = 0.8
defensive_proposal = DefensiveMixture(
    laplace_ex4,    # Targeted: Laplace approximation
    prior_ex4,      # Defensive: prior
    alpha=alpha_ex4
)

# Compare with pure Laplace
proposals_ex4 = {
    'Laplace only': laplace_ex4,
    f'Defensive α={alpha_ex4}': defensive_proposal,
}

results_ex4 = compare_proposals(posterior_ex4, proposals_ex4,
                                lambda x: x, n_samples=3000,
                                n_replications=100)

print(f"\n{'Proposal':<20} {'Mean ESS':>10} {'Min ESS':>10} {'Std Est':>10}")
print("-" * 55)

for name, stats_dict in results_ex4.items():
    print(f"{name:<20} {stats_dict['mean_ess']:10.1f} "
          f"{stats_dict['min_ess']:10.1f} {stats_dict['std_estimate']:10.4f}")

print("\nDefensive mixing provides insurance against model misspecification!")


# Summary Visualization
# ===================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Concept illustration
ax = axes[0, 0]
x_concept = np.linspace(-6, 10, 1000)

# Stylized target
target_concept = 0.7 * stats.norm.pdf(x_concept, 2, 1)

# Targeted proposal (too narrow)
targeted_concept = stats.norm(2, 0.8)

# Defensive component (broad)
defensive_concept = stats.norm(0, 4)

# Defensive mixture
alpha_concept = 0.8
defensive_mix = DefensiveMixture(targeted_concept, defensive_concept, alpha_concept)

ax.plot(x_concept, target_concept, 'r-', linewidth=3, label='Target', alpha=0.7)
ax.plot(x_concept, targeted_concept.pdf(x_concept), 'b--', linewidth=2,
        label='Targeted (risky)')
ax.plot(x_concept, defensive_mix.pdf(x_concept), 'g:', linewidth=2,
        label=f'Defensive (α={alpha_concept})')
ax.fill_between(x_concept, 0, defensive_concept.pdf(x_concept),
                alpha=0.2, color='orange', label='Defensive component')

ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Defensive Mixture Concept', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: ESS comparison (from Example 1)
ax = axes[0, 1]
proposal_names = list(results_ex1.keys())
mean_ess_values = [results_ex1[name]['mean_ess'] for name in proposal_names]
min_ess_values = [results_ex1[name]['min_ess'] for name in proposal_names]

x_pos = np.arange(len(proposal_names))
width = 0.35

ax.bar(x_pos - width/2, mean_ess_values, width, label='Mean ESS',
       alpha=0.7, color='steelblue', edgecolor='black')
ax.bar(x_pos + width/2, min_ess_values, width, label='Min ESS',
       alpha=0.7, color='orange', edgecolor='black')

ax.set_ylabel('ESS', fontsize=11)
ax.set_title('Robustness: Mean vs Minimum ESS', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names],
                   fontsize=8, rotation=0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Bias comparison (from Example 2)
ax = axes[1, 0]
proposal_names_ex2 = list(results_ex2.keys())
biases = [results_ex2[name]['mean_estimate'] - true_mean
          for name in proposal_names_ex2]
colors_bias = ['red' if abs(b) > 0.5 else 'green' for b in biases]

ax.bar(range(len(proposal_names_ex2)), biases, color=colors_bias,
       alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Bias', fontsize=11)
ax.set_title('Bias Under Misspecification', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(proposal_names_ex2)))
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names_ex2],
                   fontsize=9, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Variance comparison
ax = axes[1, 1]
std_values = [results_ex1[name]['std_estimate'] for name in proposal_names]

ax.bar(range(len(proposal_names)), std_values,
       color='purple', alpha=0.7, edgecolor='black')
ax.set_ylabel('Standard Deviation of Estimates', fontsize=11)
ax.set_title('Estimation Variance', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(proposal_names)))
ax.set_xticklabels([name.replace(' ', '\n') for name in proposal_names],
                   fontsize=8, rotation=0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/defensive_summary.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. DEFENSIVE IMPORTANCE SAMPLING provides robustness:
   - Mix targeted proposal with broad defensive component
   - q_def(θ) = α q(θ) + (1-α) m(θ)
   - Guarantees bounded variance under mild conditions

2. DEFENSIVE COMPONENT CHOICE:
   - Must have heavier tails than target
   - Common choices: prior, uniform, Cauchy, very broad Gaussian
   - Should cover entire support of target

3. MIXTURE PARAMETER α:
   - Higher α (0.9-0.95): better ESS, less robust
   - Lower α (0.5-0.7): worse ESS, more robust
   - Typical: α ∈ [0.7, 0.9]
   - Trade-off between efficiency and robustness

4. THEORETICAL GUARANTEE:
   - If m(θ) ≥ c·π(θ) for some c > 0, variance is bounded
   - No such guarantee for pure targeted proposal
   - Key advantage over standard IS

5. WHEN TO USE DEFENSIVE IS:
   - Uncertain about target shape
   - Heavy-tailed targets
   - Risk of proposal misspecification
   - Need guaranteed performance
   - Exploratory analysis

6. ROBUSTNESS METRICS:
   - Minimum ESS (over multiple runs)
   - Worst-case performance
   - More important than average ESS

7. ESS TRADE-OFF:
   - Defensive IS: lower average ESS
   - But: much higher minimum ESS
   - Better worst-case performance
   - More stable across different scenarios

8. PRACTICAL BENEFITS:
   - "Set and forget" - less tuning needed
   - Graceful degradation under misspecification
   - Peace of mind for production systems
   - Especially valuable in automation

9. COMPARISON TO PURE TARGETED:
   - Pure targeted: best case very good, worst case disaster
   - Defensive: best case good, worst case acceptable
   - Defensive is "insurance" against failure

10. IMPLEMENTATION TIPS:
    - Start with α = 0.8 as default
    - Use prior or very broad distribution as defensive component
    - Monitor both mean and min ESS
    - Adjust α based on stability requirements
    - Consider adaptive α based on ESS

11. LIMITATIONS:
    - Lower average ESS than well-tuned pure proposal
    - More computational cost (evaluating mixture)
    - Still need reasonable defensive component

12. APPLICATIONS:
    - Production Bayesian inference systems
    - Automated parameter estimation
    - Safety-critical applications
    - When proposal tuning is difficult
    - Exploratory data analysis
""")
