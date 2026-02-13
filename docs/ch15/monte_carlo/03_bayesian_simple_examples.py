"""
03_bayesian_simple_examples.py

BEGINNER LEVEL: Simple Bayesian Inference Examples with Importance Sampling

This module demonstrates importance sampling for conjugate Bayesian models
where we have analytical solutions for validation.

Models Covered:
1. Beta-Binomial (Bernoulli observations)
2. Normal-Normal (known variance)
3. Gamma-Poisson (Poisson observations)

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


def plot_bayesian_update(prior_dist, posterior_dist, data, param_name='θ',
                         title='Bayesian Update'):
    """Helper function to visualize Bayesian updating."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot distributions
    ax = axes[0]
    x = np.linspace(prior_dist.ppf(0.001), prior_dist.ppf(0.999), 1000)
    if hasattr(posterior_dist, 'ppf'):
        x_post = np.linspace(posterior_dist.ppf(0.001), posterior_dist.ppf(0.999), 1000)
        x = np.union1d(x, x_post)
    
    ax.plot(x, prior_dist.pdf(x), 'b--', linewidth=2, label='Prior', alpha=0.7)
    ax.plot(x, posterior_dist.pdf(x), 'r-', linewidth=2, label='Posterior', alpha=0.7)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot data
    ax = axes[1]
    if len(data) < 50:
        ax.hist(data, bins=min(len(np.unique(data)), 20), alpha=0.7, 
                edgecolor='black', color='green')
    else:
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Observed Data', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Data (n={len(data)})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# Example 1: Beta-Binomial Model
# ============================
print("=" * 70)
print("EXAMPLE 1: Beta-Binomial Model (Bernoulli Trials)")
print("=" * 70)

print("""
Model:
  Likelihood: y ~ Bernoulli(θ), observed s successes in n trials
  Prior: θ ~ Beta(α₀, β₀)
  Posterior: θ|y ~ Beta(α₀+s, β₀+n-s)

Task: Estimate posterior mean and variance using importance sampling
""")

# Generate synthetic data: coin flips
true_theta = 0.7  # True success probability
n_trials = 50
data = np.random.binomial(1, true_theta, n_trials)
successes = np.sum(data)
failures = n_trials - successes

print(f"\nData: {successes} successes out of {n_trials} trials")
print(f"Sample proportion: {successes/n_trials:.3f}")

# Prior: Beta(2, 2) - slightly informative, centered at 0.5
alpha_0, beta_0 = 2, 2
prior_dist = stats.beta(alpha_0, beta_0)

print(f"\nPrior: Beta({alpha_0}, {beta_0})")
print(f"  Prior mean: {alpha_0/(alpha_0+beta_0):.3f}")

# Analytical posterior
alpha_n = alpha_0 + successes
beta_n = beta_0 + failures
posterior_dist = stats.beta(alpha_n, beta_n)

print(f"\nPosterior (analytical): Beta({alpha_n}, {beta_n})")
print(f"  Posterior mean: {alpha_n/(alpha_n+beta_n):.6f}")
print(f"  Posterior variance: {posterior_dist.var():.6f}")

# Unnormalized posterior: γ(θ) = p(y|θ)p(θ)
def unnormalized_posterior_beta(theta):
    """
    γ(θ) = θˢ(1-θ)ⁿ⁻ˢ × θ^(α₀-1)(1-θ)^(β₀-1)
         = θ^(α₀+s-1)(1-θ)^(β₀+n-s-1)
    
    This is proportional to Beta(α₀+s, β₀+n-s)
    """
    # Prevent numerical issues at boundaries
    theta = np.clip(theta, 1e-10, 1-1e-10)
    
    # Log-space for numerical stability
    log_likelihood = successes * np.log(theta) + failures * np.log(1 - theta)
    log_prior = (alpha_0-1) * np.log(theta) + (beta_0-1) * np.log(1 - theta)
    
    return np.exp(log_likelihood + log_prior)

# Importance sampling with prior as proposal
n_samples = 5000

# Sample from prior
samples = prior_dist.rvs(size=n_samples)

# Compute unnormalized weights
gamma_values = unnormalized_posterior_beta(samples)
q_values = prior_dist.pdf(samples)
unnorm_weights = gamma_values / q_values

# Normalize weights
weights = unnorm_weights / np.sum(unnorm_weights)

# Estimate posterior mean
h_mean = lambda theta: theta
posterior_mean_is = np.sum(weights * h_mean(samples))

# Estimate posterior variance
h_var = lambda theta: (theta - posterior_mean_is)**2
posterior_var_is = np.sum(weights * h_var(samples))

# ESS
ess = 1.0 / np.sum(weights**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess:.1f}):")
print(f"  Posterior mean: {posterior_mean_is:.6f} (error: {abs(posterior_mean_is - posterior_dist.mean()):.6f})")
print(f"  Posterior variance: {posterior_var_is:.6f} (error: {abs(posterior_var_is - posterior_dist.var()):.6f})")

# Visualize
fig = plot_bayesian_update(prior_dist, posterior_dist, data, 'θ', 
                           'Beta-Binomial: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example1_beta_binomial.png', 
            dpi=300, bbox_inches='tight')


# Example 2: Normal-Normal Model
# ============================
print("\n" + "=" * 70)
print("EXAMPLE 2: Normal-Normal Model (Known Variance)")
print("=" * 70)

print("""
Model:
  Likelihood: y ~ N(θ, σ²), σ² known
  Prior: θ ~ N(μ₀, τ₀²)
  Posterior: θ|y ~ N(μₙ, τₙ²) where
    τₙ² = 1/(1/τ₀² + n/σ²)
    μₙ = τₙ²(μ₀/τ₀² + Σyᵢ/σ²)

Task: Estimate posterior distribution using importance sampling
""")

# Generate data
true_theta = 8.0
sigma = 2.0  # Known observation noise
n_obs = 30
data_normal = np.random.normal(true_theta, sigma, n_obs)

print(f"\nData: n={n_obs}, sample mean={np.mean(data_normal):.3f}, σ={sigma}")

# Prior
mu_0 = 5.0
tau_0 = 3.0
prior_normal = stats.norm(mu_0, tau_0)

print(f"\nPrior: N({mu_0}, {tau_0}²)")

# Analytical posterior
precision_0 = 1.0 / tau_0**2
precision_n = precision_0 + n_obs / sigma**2
tau_n = 1.0 / np.sqrt(precision_n)
mu_n = (precision_0 * mu_0 + np.sum(data_normal) / sigma**2) / precision_n
posterior_normal = stats.norm(mu_n, tau_n)

print(f"\nPosterior (analytical): N({mu_n:.6f}, {tau_n:.6f}²)")

# Unnormalized posterior
def unnormalized_posterior_normal(theta):
    """
    γ(θ) = ∏ᵢ exp(-(yᵢ-θ)²/2σ²) × exp(-(θ-μ₀)²/2τ₀²)
    """
    log_likelihood = -0.5 * np.sum((data_normal[:, None] - theta)**2) / sigma**2
    log_prior = -0.5 * (theta - mu_0)**2 / tau_0**2
    return np.exp(log_likelihood + log_prior)

# Importance sampling with prior as proposal
n_samples = 5000
samples_normal = prior_normal.rvs(size=n_samples)

# Unnormalized weights
gamma_values_normal = unnormalized_posterior_normal(samples_normal)
q_values_normal = prior_normal.pdf(samples_normal)
unnorm_weights_normal = gamma_values_normal / q_values_normal
weights_normal = unnorm_weights_normal / np.sum(unnorm_weights_normal)

# Estimates
posterior_mean_normal_is = np.sum(weights_normal * samples_normal)
posterior_var_normal_is = np.sum(weights_normal * (samples_normal - posterior_mean_normal_is)**2)
ess_normal = 1.0 / np.sum(weights_normal**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_normal:.1f}):")
print(f"  Posterior mean: {posterior_mean_normal_is:.6f} (true: {mu_n:.6f})")
print(f"  Posterior std: {np.sqrt(posterior_var_normal_is):.6f} (true: {tau_n:.6f})")

# Credible interval: 95%
sorted_samples = samples_normal[np.argsort(weights_normal)[::-1]]
sorted_weights = np.sort(weights_normal)[::-1]
cumsum_weights = np.cumsum(sorted_weights)
n_95 = np.searchsorted(cumsum_weights, 0.95) + 1
credible_95_is = np.percentile(sorted_samples[:n_95], [2.5, 97.5])

# True credible interval
credible_95_true = posterior_normal.ppf([0.025, 0.975])

print(f"\n95% Credible Interval:")
print(f"  IS: [{credible_95_is[0]:.3f}, {credible_95_is[1]:.3f}]")
print(f"  True: [{credible_95_true[0]:.3f}, {credible_95_true[1]:.3f}]")

# Visualize
fig = plot_bayesian_update(prior_normal, posterior_normal, data_normal, 'θ',
                           'Normal-Normal: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example2_normal_normal.png',
            dpi=300, bbox_inches='tight')


# Example 3: Gamma-Poisson Model
# ============================
print("\n" + "=" * 70)
print("EXAMPLE 3: Gamma-Poisson Model")
print("=" * 70)

print("""
Model:
  Likelihood: y ~ Poisson(λ), observed counts
  Prior: λ ~ Gamma(α₀, β₀)
  Posterior: λ|y ~ Gamma(α₀+Σyᵢ, β₀+n)

Task: Estimate posterior using importance sampling
""")

# Generate count data
true_lambda = 4.5
n_counts = 40
data_poisson = np.random.poisson(true_lambda, n_counts)
sum_counts = np.sum(data_poisson)

print(f"\nData: n={n_counts}, Σyᵢ={sum_counts}, sample mean={np.mean(data_poisson):.3f}")

# Prior: Gamma(2, 0.5)
# Mean = α/β = 2/0.5 = 4
alpha_0_gamma = 2.0
beta_0_gamma = 0.5
prior_gamma = stats.gamma(alpha_0_gamma, scale=1.0/beta_0_gamma)

print(f"\nPrior: Gamma({alpha_0_gamma}, {beta_0_gamma})")
print(f"  Prior mean: {alpha_0_gamma/beta_0_gamma:.3f}")

# Analytical posterior
alpha_n_gamma = alpha_0_gamma + sum_counts
beta_n_gamma = beta_0_gamma + n_counts
posterior_gamma = stats.gamma(alpha_n_gamma, scale=1.0/beta_n_gamma)

print(f"\nPosterior (analytical): Gamma({alpha_n_gamma}, {beta_n_gamma})")
print(f"  Posterior mean: {alpha_n_gamma/beta_n_gamma:.6f}")

# Unnormalized posterior
def unnormalized_posterior_gamma(lam):
    """
    γ(λ) = ∏ᵢ λ^yᵢ exp(-λ) × λ^(α₀-1) exp(-β₀λ)
         = λ^(α₀+Σyᵢ-1) exp(-(β₀+n)λ)
    
    This is proportional to Gamma(α₀+Σyᵢ, β₀+n)
    """
    # Log-space for stability
    log_likelihood = sum_counts * np.log(lam + 1e-10) - n_counts * lam
    log_prior = (alpha_0_gamma-1) * np.log(lam + 1e-10) - beta_0_gamma * lam
    return np.exp(log_likelihood + log_prior)

# Importance sampling
n_samples = 5000
samples_gamma = prior_gamma.rvs(size=n_samples)

# Weights
gamma_values_poisson = unnormalized_posterior_gamma(samples_gamma)
q_values_gamma = prior_gamma.pdf(samples_gamma)
unnorm_weights_gamma = gamma_values_poisson / q_values_gamma
weights_gamma = unnorm_weights_gamma / np.sum(unnorm_weights_gamma)

# Estimates
posterior_mean_gamma_is = np.sum(weights_gamma * samples_gamma)
posterior_var_gamma_is = np.sum(weights_gamma * (samples_gamma - posterior_mean_gamma_is)**2)
ess_gamma = 1.0 / np.sum(weights_gamma**2)

print(f"\nImportance Sampling Estimates (n={n_samples}, ESS={ess_gamma:.1f}):")
print(f"  Posterior mean: {posterior_mean_gamma_is:.6f} (true: {alpha_n_gamma/beta_n_gamma:.6f})")
print(f"  Posterior variance: {posterior_var_gamma_is:.6f} (true: {posterior_gamma.var():.6f})")

# Posterior predictive: P(ỹ|y)
# For new observation ỹ
print("\nPosterior Predictive Distribution for new observation:")

# True posterior predictive for Gamma-Poisson is Negative Binomial
# ỹ|y ~ NB(α_n, β_n/(β_n+1))
post_pred_true = stats.nbinom(alpha_n_gamma, beta_n_gamma/(beta_n_gamma+1))

# Estimate via importance sampling
# E[P(ỹ|λ)|y] = E[Poisson(ỹ|λ)|y]
y_new_values = np.arange(0, 15)
post_pred_is = []

for y_new in y_new_values:
    # h(λ) = P(ỹ=y_new|λ) = λ^y_new exp(-λ) / y_new!
    h_poisson = lambda lam: stats.poisson.pmf(y_new, lam)
    prob_is = np.sum(weights_gamma * h_poisson(samples_gamma))
    post_pred_is.append(prob_is)

post_pred_true_probs = post_pred_true.pmf(y_new_values)

print(f"\nPosterior Predictive P(ỹ|y):")
print("y_new  True     IS")
print("-" * 25)
for y, p_true, p_is in zip(y_new_values[:8], post_pred_true_probs[:8], post_pred_is[:8]):
    print(f"{y:3d}   {p_true:.4f}  {p_is:.4f}")

# Visualize
fig = plot_bayesian_update(prior_gamma, posterior_gamma, data_poisson, 'λ',
                           'Gamma-Poisson: Prior vs Posterior')
plt.savefig('/home/claude/03_Importance_Sampling/example3_gamma_poisson.png',
            dpi=300, bbox_inches='tight')


# Comparative Analysis
# ===================
print("\n" + "=" * 70)
print("COMPARATIVE ANALYSIS: ESS Across Models")
print("=" * 70)

models = ['Beta-Binomial', 'Normal-Normal', 'Gamma-Poisson']
ess_values = [ess, ess_normal, ess_gamma]
efficiencies = [e/n_samples*100 for e in ess_values]

print("\nEffective Sample Size Summary:")
print("-" * 50)
for model, ess_val, eff in zip(models, ess_values, efficiencies):
    print(f"{model:20s}: ESS = {ess_val:6.1f} ({eff:5.1f}%)")

print("""
\nObservations:
1. Using the prior as proposal is simple but not always efficient
2. ESS depends on how much the data updates the prior
3. Strong data (large n or extreme observations) → lower ESS
4. ESS decreases as prior-posterior mismatch increases
5. More informative likelihood → need better proposal than prior
""")

# Final visualization: ESS comparison
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(models, efficiencies, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=2)
ax.set_ylabel('Efficiency (ESS/n × 100%)', fontsize=13)
ax.set_title('Importance Sampling Efficiency: Prior as Proposal', 
             fontsize=14, fontweight='bold')
ax.axhline(100, color='red', linestyle='--', linewidth=2, 
           label='Perfect efficiency', alpha=0.5)
ax.set_ylim([0, max(efficiencies)*1.2])
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

# Add value labels on bars
for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/comparative_ess.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. CONJUGATE MODELS provide analytical validation for IS:
   - Beta-Binomial for binary data
   - Normal-Normal for continuous data (known variance)
   - Gamma-Poisson for count data

2. PRIOR AS PROPOSAL is simple but has limitations:
   - Works well when data is weak (small n)
   - Becomes inefficient with strong data
   - ESS decreases as likelihood dominates prior

3. IMPORTANCE SAMPLING ADVANTAGES:
   - No burn-in period (unlike MCMC)
   - Samples are independent
   - Can estimate multiple quantities from same samples
   - Can compute posterior predictive easily

4. ESS AS DIAGNOSTIC:
   - Measures effective number of independent samples
   - ESS << n indicates need for better proposal
   - Can compare proposals objectively

5. PRACTICAL CONSIDERATIONS:
   - Always check ESS after sampling
   - Low ESS → few samples dominate
   - Consider adaptive or sequential methods for low ESS
   - For complex posteriors, need smarter proposals

6. CONNECTION TO MCMC:
   - IS complementary to MCMC
   - Use IS when: samples independent, known good proposal
   - Use MCMC when: complex posterior, unknown good proposal
""")
