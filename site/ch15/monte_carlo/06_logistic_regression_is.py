"""
06_logistic_regression_IS.py

INTERMEDIATE LEVEL: Bayesian Logistic Regression via Importance Sampling

This module demonstrates importance sampling for Bayesian inference
in logistic regression, a non-conjugate model where the posterior
is not available in closed form.

Model:
------
Likelihood: yᵢ ~ Bernoulli(π(xᵢ'β))
            where π(z) = 1/(1 + exp(-z)) is the logistic function

Prior: β ~ N(μ₀, Σ₀)

Posterior: p(β|y,X) ∝ ∏ᵢ π(xᵢ'β)^yᵢ (1-π(xᵢ'β))^(1-yᵢ) × N(β|μ₀,Σ₀)

This is a non-conjugate model: posterior is not Gaussian!

Proposal Strategies:
1. Prior as proposal (simple but often inefficient)
2. Laplace approximation (Gaussian at mode)
3. Variational approximation
4. Adaptive proposals

Author: Educational Materials for Bayesian Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit  # logistic function
import seaborn as sns

np.random.seed(42)
sns.set_style("whitegrid")


def logistic(z):
    """Logistic function: π(z) = 1/(1 + exp(-z))"""
    return expit(z)


def log_likelihood_logistic(beta, X, y):
    """
    Log-likelihood for logistic regression.
    
    log p(y|X,β) = Σᵢ [yᵢ log π(xᵢ'β) + (1-yᵢ) log(1-π(xᵢ'β))]
    """
    eta = X @ beta  # Linear predictor xᵢ'β
    
    # Numerically stable log-likelihood
    # log π(η) = -log(1 + exp(-η))
    # log(1-π(η)) = -log(1 + exp(η))
    
    log_lik = np.sum(
        y * (-np.log1p(np.exp(-eta))) +
        (1 - y) * (-np.log1p(np.exp(eta)))
    )
    
    return log_lik


def log_prior_gaussian(beta, mu_0, Sigma_0_inv):
    """
    Log-prior: log p(β) = log N(β|μ₀, Σ₀)
    """
    diff = beta - mu_0
    log_prior = -0.5 * (diff.T @ Sigma_0_inv @ diff)
    return log_prior


def log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv):
    """
    Unnormalized log-posterior: log p(β|y,X) ∝ log p(y|X,β) + log p(β)
    """
    return (log_likelihood_logistic(beta, X, y) +
            log_prior_gaussian(beta, mu_0, Sigma_0_inv))


def find_map_estimate(X, y, mu_0, Sigma_0_inv):
    """
    Find MAP (Maximum A Posteriori) estimate via optimization.
    
    MAP = argmax_β p(β|y,X)
    """
    # Negative log-posterior (for minimization)
    def neg_log_post(beta):
        return -log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv)
    
    # Gradient
    def grad(beta):
        eta = X @ beta
        prob = logistic(eta)
        grad_ll = X.T @ (y - prob)
        grad_prior = -Sigma_0_inv @ (beta - mu_0)
        return -(grad_ll + grad_prior)
    
    # Optimize
    result = minimize(neg_log_post, mu_0, jac=grad, method='BFGS')
    
    return result.x


def laplace_approximation(X, y, mu_0, Sigma_0_inv):
    """
    Compute Laplace approximation to posterior.
    
    Approximates posterior with Gaussian centered at MAP:
    p(β|y,X) ≈ N(β|β_MAP, H⁻¹)
    
    where H is the Hessian of negative log-posterior at MAP.
    
    Returns:
    --------
    beta_map : MAP estimate
    Sigma_laplace : Covariance matrix of Laplace approximation
    """
    # Find MAP
    beta_map = find_map_estimate(X, y, mu_0, Sigma_0_inv)
    
    # Compute Hessian at MAP
    eta = X @ beta_map
    prob = logistic(eta)
    
    # Hessian of log-likelihood
    W = np.diag(prob * (1 - prob))  # Weight matrix
    H_ll = -X.T @ W @ X
    
    # Hessian of log-posterior
    H = H_ll - Sigma_0_inv
    
    # Covariance is inverse of negative Hessian
    Sigma_laplace = np.linalg.inv(-H)
    
    return beta_map, Sigma_laplace


# Example 1: Simple 1D Logistic Regression
# =======================================
print("=" * 70)
print("EXAMPLE 1: 1D Logistic Regression")
print("=" * 70)

# Generate synthetic data
np.random.seed(42)
n_obs = 100

# True parameter
beta_true = np.array([0.5, 2.0])  # [intercept, slope]

# Features: [1, x]
x_raw = np.random.uniform(-2, 2, n_obs)
X = np.column_stack([np.ones(n_obs), x_raw])

# Generate binary outcomes
eta_true = X @ beta_true
prob_true = logistic(eta_true)
y = np.random.binomial(1, prob_true)

print(f"\nGenerated {n_obs} observations")
print(f"True β = {beta_true}")
print(f"Observed: {np.sum(y)} successes, {n_obs - np.sum(y)} failures")

# Prior: N(0, 10I) - weakly informative
mu_0 = np.zeros(2)
Sigma_0 = 10 * np.eye(2)
Sigma_0_inv = np.linalg.inv(Sigma_0)

print(f"\nPrior: β ~ N(0, 10I)")

# Compute Laplace approximation
beta_map, Sigma_laplace = laplace_approximation(X, y, mu_0, Sigma_0_inv)

print(f"\nMAP estimate: {beta_map}")
print(f"Laplace std: {np.sqrt(np.diag(Sigma_laplace))}")


# Importance Sampling with Different Proposals
# ------------------------------------------

# Proposal 1: Prior (simple but inefficient)
print("\n" + "-" * 70)
print("PROPOSAL 1: Using Prior as Proposal")
print("-" * 70)

n_samples = 5000
prior_dist = stats.multivariate_normal(mu_0, Sigma_0)

# Sample from prior
samples_prior = prior_dist.rvs(size=n_samples)

# Compute importance weights
log_weights_prior = np.array([
    log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
    prior_dist.logpdf(beta)
    for beta in samples_prior
])

# Normalize weights
log_weights_prior_norm = log_weights_prior - np.max(log_weights_prior)
weights_prior_unnorm = np.exp(log_weights_prior_norm)
weights_prior = weights_prior_unnorm / np.sum(weights_prior_unnorm)

# ESS
ess_prior = 1.0 / np.sum(weights_prior**2)

# Estimates
beta_est_prior = np.sum(weights_prior[:, None] * samples_prior, axis=0)

print(f"ESS: {ess_prior:.1f} ({ess_prior/n_samples:.1%})")
print(f"Estimated β: {beta_est_prior}")
print(f"Error: {np.linalg.norm(beta_est_prior - beta_true):.4f}")


# Proposal 2: Laplace approximation (much better)
print("\n" + "-" * 70)
print("PROPOSAL 2: Using Laplace Approximation as Proposal")
print("-" * 70)

laplace_dist = stats.multivariate_normal(beta_map, Sigma_laplace)

# Sample from Laplace approximation
samples_laplace = laplace_dist.rvs(size=n_samples)

# Compute importance weights
log_weights_laplace = np.array([
    log_posterior_logistic(beta, X, y, mu_0, Sigma_0_inv) -
    laplace_dist.logpdf(beta)
    for beta in samples_laplace
])

# Normalize weights
log_weights_laplace_norm = log_weights_laplace - np.max(log_weights_laplace)
weights_laplace_unnorm = np.exp(log_weights_laplace_norm)
weights_laplace = weights_laplace_unnorm / np.sum(weights_laplace_unnorm)

# ESS
ess_laplace = 1.0 / np.sum(weights_laplace**2)

# Estimates
beta_est_laplace = np.sum(weights_laplace[:, None] * samples_laplace, axis=0)

print(f"ESS: {ess_laplace:.1f} ({ess_laplace/n_samples:.1%})")
print(f"Estimated β: {beta_est_laplace}")
print(f"Error: {np.linalg.norm(beta_est_laplace - beta_true):.4f}")

print(f"\nImprovement: {ess_laplace/ess_prior:.1f}x better ESS with Laplace proposal")


# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Data and fitted curves
ax = axes[0, 0]
sorted_idx = np.argsort(x_raw)
x_plot = x_raw[sorted_idx]

# True probability
X_plot = np.column_stack([np.ones(len(x_plot)), x_plot])
prob_true_plot = logistic(X_plot @ beta_true)

# MAP estimate
prob_map = logistic(X_plot @ beta_map)

# IS estimates (using Laplace proposal)
prob_is = logistic(X_plot @ beta_est_laplace)

ax.scatter(x_raw[y==0], y[y==0], c='red', alpha=0.5, s=50, label='y=0')
ax.scatter(x_raw[y==1], y[y==1], c='blue', alpha=0.5, s=50, label='y=1')
ax.plot(x_plot, prob_true_plot, 'k-', linewidth=3, label='True', alpha=0.7)
ax.plot(x_plot, prob_map, 'g--', linewidth=2, label='MAP')
ax.plot(x_plot, prob_is, 'r:', linewidth=2, label='IS (Laplace)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('P(y=1|x)', fontsize=12)
ax.set_title('Logistic Regression Fit', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Posterior samples (Prior proposal)
ax = axes[0, 1]
scatter = ax.scatter(samples_prior[:, 0], samples_prior[:, 1],
                    c=weights_prior*n_samples, cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
ax.set_xlabel('β₀ (intercept)', fontsize=11)
ax.set_ylabel('β₁ (slope)', fontsize=11)
ax.set_title(f'Prior Proposal: ESS={ess_prior:.0f}', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight × n')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 3: Posterior samples (Laplace proposal)
ax = axes[1, 0]
scatter = ax.scatter(samples_laplace[:, 0], samples_laplace[:, 1],
                    c=weights_laplace*n_samples, cmap='viridis',
                    s=20, alpha=0.5, edgecolors='black', linewidth=0.3)
ax.plot(beta_true[0], beta_true[1], 'r*', markersize=20, label='True')
ax.plot(beta_map[0], beta_map[1], 'go', markersize=15, label='MAP')
ax.set_xlabel('β₀ (intercept)', fontsize=11)
ax.set_ylabel('β₁ (slope)', fontsize=11)
ax.set_title(f'Laplace Proposal: ESS={ess_laplace:.0f}', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Weight × n')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 4: Weight distributions comparison
ax = axes[1, 1]
ax.hist(weights_prior * n_samples, bins=50, alpha=0.5, density=True,
        color='blue', edgecolor='black', label='Prior proposal')
ax.hist(weights_laplace * n_samples, bins=50, alpha=0.5, density=True,
        color='green', edgecolor='black', label='Laplace proposal')
ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('Weight × n', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Weight Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])

plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/logistic_regression_1d.png',
            dpi=300, bbox_inches='tight')


# Example 2: Higher-Dimensional Case
# =================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Higher-Dimensional Logistic Regression")
print("=" * 70)

# Generate data with p=5 features
n_obs_2 = 200
p = 5

# True parameters
beta_true_2 = np.array([1.0, -0.5, 0.8, -0.3, 0.6])

# Features
X_2 = np.random.randn(n_obs_2, p)

# Outcomes
eta_true_2 = X_2 @ beta_true_2
prob_true_2 = logistic(eta_true_2)
y_2 = np.random.binomial(1, prob_true_2)

print(f"\nData: n={n_obs_2}, p={p} features")
print(f"True β = {beta_true_2}")

# Prior
mu_0_2 = np.zeros(p)
Sigma_0_2 = 5 * np.eye(p)
Sigma_0_inv_2 = np.linalg.inv(Sigma_0_2)

# Laplace approximation
beta_map_2, Sigma_laplace_2 = laplace_approximation(X_2, y_2, mu_0_2, Sigma_0_inv_2)

print(f"\nMAP estimate: {beta_map_2}")

# Importance sampling with Laplace proposal
n_samples_2 = 10000
laplace_dist_2 = stats.multivariate_normal(beta_map_2, Sigma_laplace_2)
samples_2 = laplace_dist_2.rvs(size=n_samples_2)

log_weights_2 = np.array([
    log_posterior_logistic(beta, X_2, y_2, mu_0_2, Sigma_0_inv_2) -
    laplace_dist_2.logpdf(beta)
    for beta in samples_2
])

log_weights_2_norm = log_weights_2 - np.max(log_weights_2)
weights_2_unnorm = np.exp(log_weights_2_norm)
weights_2 = weights_2_unnorm / np.sum(weights_2_unnorm)

ess_2 = 1.0 / np.sum(weights_2**2)

# Posterior mean
beta_post_mean = np.sum(weights_2[:, None] * samples_2, axis=0)

# Posterior std
beta_post_std = np.sqrt(np.sum(weights_2[:, None] * (samples_2 - beta_post_mean)**2, axis=0))

print(f"\nImportance Sampling Results (Laplace proposal):")
print(f"  n_samples: {n_samples_2}")
print(f"  ESS: {ess_2:.1f} ({ess_2/n_samples_2:.1%})")

print("\nPosterior Estimates:")
print(f"{'Feature':<10} {'True':>8} {'MAP':>8} {'IS Mean':>8} {'IS Std':>8}")
print("-" * 50)
for i in range(p):
    print(f"β{i:<9} {beta_true_2[i]:8.3f} {beta_map_2[i]:8.3f} "
          f"{beta_post_mean[i]:8.3f} {beta_post_std[i]:8.3f}")

# Credible intervals
credible_intervals = []
for i in range(p):
    sorted_samples = samples_2[:, i][np.argsort(weights_2)[::-1]]
    sorted_weights = np.sort(weights_2)[::-1]
    cumsum = np.cumsum(sorted_weights)
    n_95 = np.searchsorted(cumsum, 0.95) + 1
    ci = np.percentile(sorted_samples[:n_95], [2.5, 97.5])
    credible_intervals.append(ci)

print("\n95% Credible Intervals:")
for i, ci in enumerate(credible_intervals):
    contains = ci[0] <= beta_true_2[i] <= ci[1]
    status = "✓" if contains else "✗"
    print(f"β{i}: [{ci[0]:6.3f}, {ci[1]:6.3f}] {status}")


# Example 3: Prediction
# ===================
print("\n" + "=" * 70)
print("EXAMPLE 3: Posterior Predictive Distribution")
print("=" * 70)

# New test point
x_new = np.array([0.5, -0.3, 0.2, 0.1, -0.4])

# Posterior predictive: P(y_new=1|x_new, data)
# = ∫ P(y_new=1|x_new, β) p(β|data) dβ
# ≈ Σᵢ wᵢ × logistic(x_new'βᵢ)

pred_probs = logistic(samples_2 @ x_new)
posterior_pred_prob = np.sum(weights_2 * pred_probs)

# True probability
true_pred_prob = logistic(x_new @ beta_true_2)

# MAP-based prediction
map_pred_prob = logistic(x_new @ beta_map_2)

print(f"\nPredictive probability P(y=1|x_new):")
print(f"  True: {true_pred_prob:.4f}")
print(f"  MAP: {map_pred_prob:.4f}")
print(f"  Posterior mean (IS): {posterior_pred_prob:.4f}")

# Posterior predictive distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pred_probs, bins=50, weights=weights_2, density=True,
        alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(true_pred_prob, color='red', linestyle='--', linewidth=2,
           label=f'True: {true_pred_prob:.3f}')
ax.axvline(posterior_pred_prob, color='green', linestyle='-', linewidth=2,
           label=f'Posterior mean: {posterior_pred_prob:.3f}')
ax.set_xlabel('P(y=1|x_new)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/03_Importance_Sampling/logistic_predictive.png',
            dpi=300, bbox_inches='tight')

plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. LOGISTIC REGRESSION IS NON-CONJUGATE:
   - Posterior not in closed form
   - No analytical solution
   - Importance sampling is applicable

2. PROPOSAL STRATEGIES:
   - Prior: Simple but often inefficient (low ESS)
   - Laplace approximation: Gaussian at MAP
     * Much better ESS (often 10-100x improvement)
     * Good for moderate dimensions
   - Can achieve 30-60% efficiency in many cases

3. LAPLACE APPROXIMATION:
   - Find MAP via optimization
   - Compute Hessian at MAP
   - Use N(β_MAP, H⁻¹) as proposal
   - Works well when posterior is approximately Gaussian

4. DIMENSIONALITY EFFECTS:
   - Prior proposal ESS degrades exponentially with dimension
   - Laplace proposal scales much better
   - For p > 10-20, may need adaptive methods or MCMC

5. INFERENCE TASKS:
   - Posterior mean and variance
   - Credible intervals
   - Posterior predictive distributions
   - All accessible via weighted samples

6. PRACTICAL CONSIDERATIONS:
   - Always check ESS
   - Laplace approximation usually much better than prior
   - For high dimensions (p > 20), consider MCMC
   - Numerical stability: work in log-space for weights

7. ADVANTAGES OF IS FOR LOGISTIC REGRESSION:
   - Independent samples (no autocorrelation)
   - No burn-in period
   - Easy parallelization
   - Can compute multiple quantities from same samples

8. WHEN IS WORKS WELL:
   - Moderate dimensions (p < 20)
   - Good proposal available (Laplace, variational)
   - Non-extreme separation in data
   - Sufficient sample size (n > 100)

9. COMPARISON TO MCMC:
   - IS: Better when good proposal available
   - MCMC: Better for high dimensions or complex posteriors
   - IS: Easier to diagnose (ESS)
   - MCMC: More robust to proposal choice
""")
