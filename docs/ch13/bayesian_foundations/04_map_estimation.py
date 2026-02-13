"""
Bayesian Inference - Module 4: Maximum A Posteriori (MAP) Estimation
Level: Intermediate
Topics: MAP estimation, comparison with MLE, optimization, regularization connection

MAP estimation finds the mode of the posterior distribution, providing a
point estimate that incorporates prior information.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# THEORY: MAP vs MLE vs POSTERIOR MEAN
# ============================================================================

"""
THREE COMMON POINT ESTIMATES IN BAYESIAN INFERENCE:

1. MAXIMUM LIKELIHOOD ESTIMATE (MLE):
   θ_MLE = argmax_θ p(D|θ)
   - Ignores prior information
   - Frequentist approach
   
2. MAXIMUM A POSTERIORI (MAP):
   θ_MAP = argmax_θ p(θ|D) = argmax_θ p(D|θ)p(θ)
   - Incorporates prior information
   - Mode of posterior distribution
   - Equivalent to MLE with uniform prior
   
3. POSTERIOR MEAN:
   θ_MEAN = E[θ|D] = ∫ θ p(θ|D) dθ
   - Expected value of posterior
   - Minimizes expected squared error
   - Often different from MAP for skewed posteriors

CONNECTION TO REGULARIZATION:

MAP estimation with Gaussian prior ⟷ L2 regularization (Ridge)
MAP estimation with Laplace prior ⟷ L1 regularization (Lasso)

Specifically, for linear regression:
  MLE: min ||y - Xβ||²
  MAP (Gaussian prior): min ||y - Xβ||² + λ||β||²  (Ridge)
  MAP (Laplace prior): min ||y - Xβ||² + λ||β||₁  (Lasso)
"""

def map_vs_mle_beta_binomial(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """
    Compare MAP, MLE, and posterior mean for Beta-Binomial model.
    
    Parameters:
    -----------
    n_heads, n_tails : int
        Observed data
    prior_alpha, prior_beta : float
        Beta prior parameters
    
    Returns:
    --------
    estimates : dict
        Dictionary with MLE, MAP, and posterior mean
    """
    print("="*70)
    print("MAP vs MLE vs POSTERIOR MEAN: Beta-Binomial Example")
    print("="*70)
    
    n_total = n_heads + n_tails
    print(f"\nData: {n_heads} heads, {n_tails} tails (n={n_total})")
    print(f"Prior: Beta({prior_alpha}, {prior_beta})")
    
    # Maximum Likelihood Estimate
    mle = n_heads / n_total if n_total > 0 else 0.5
    
    # Posterior parameters
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # Posterior Mean (expected value)
    posterior_mean = post_alpha / (post_alpha + post_beta)
    
    # MAP (mode of posterior)
    # Mode of Beta(α, β) = (α-1)/(α+β-2) for α,β > 1
    if post_alpha > 1 and post_beta > 1:
        map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        map_estimate = posterior_mean  # Use mean if mode undefined
    
    print(f"\nPoint Estimates:")
    print(f"  MLE:            {mle:.4f}")
    print(f"  MAP:            {map_estimate:.4f}")
    print(f"  Posterior Mean: {posterior_mean:.4f}")
    
    # Visualization
    theta = np.linspace(0, 1, 1000)
    prior_pdf = stats.beta(prior_alpha, prior_beta).pdf(theta)
    post_pdf = stats.beta(post_alpha, post_beta).pdf(theta)
    likelihood = theta**n_heads * (1-theta)**n_tails
    likelihood_norm = likelihood / np.max(likelihood) * np.max(post_pdf)
    
    plt.figure(figsize=(12, 6))
    plt.plot(theta, prior_pdf, 'b--', linewidth=2, alpha=0.7, label='Prior')
    plt.plot(theta, likelihood_norm, 'g:', linewidth=2, alpha=0.7, label='Likelihood (scaled)')
    plt.plot(theta, post_pdf, 'r-', linewidth=2, label='Posterior')
    
    plt.axvline(mle, color='green', linestyle='--', linewidth=2, label=f'MLE = {mle:.3f}')
    plt.axvline(map_estimate, color='red', linestyle='--', linewidth=2, label=f'MAP = {map_estimate:.3f}')
    plt.axvline(posterior_mean, color='darkred', linestyle=':', linewidth=2, label=f'Post Mean = {posterior_mean:.3f}')
    
    plt.xlabel('θ (Probability of Heads)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Comparing MLE, MAP, and Posterior Mean', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('map_vs_mle_vs_mean.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'mle': mle, 'map': map_estimate, 'posterior_mean': posterior_mean}

# ============================================================================
# MAP ESTIMATION WITH NUMERICAL OPTIMIZATION
# ============================================================================

def map_normal_unknown_mean_variance(data, prior_mean_mu=0, prior_std_mu=10, 
                                     prior_shape_tau=2, prior_rate_tau=1):
    """
    MAP estimation for normal distribution with unknown mean and variance.
    
    This requires numerical optimization since there's no closed form.
    
    Prior:
      μ ~ N(μ₀, σ₀²)
      τ (precision = 1/σ²) ~ Gamma(α, β)
    
    Parameters:
    -----------
    data : array-like
        Observed data points
    prior_mean_mu, prior_std_mu : float
        Parameters of normal prior on mean
    prior_shape_tau, prior_rate_tau : float
        Parameters of gamma prior on precision
    """
    print("\n" + "="*70)
    print("MAP ESTIMATION: Normal with Unknown Mean and Variance")
    print("="*70)
    
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    
    print(f"\nData: n={n}")
    print(f"  Sample mean: {sample_mean:.4f}")
    print(f"  Sample var:  {sample_var:.4f}")
    print(f"\nPriors:")
    print(f"  μ ~ N({prior_mean_mu}, {prior_std_mu**2})")
    print(f"  τ ~ Gamma({prior_shape_tau}, {prior_rate_tau})")
    
    # Define negative log posterior (to minimize)
    def neg_log_posterior(params):
        mu, log_tau = params
        tau = np.exp(log_tau)  # Use log_tau for better optimization
        
        # Log likelihood: sum of log N(x_i | mu, 1/tau)
        log_lik = np.sum(stats.norm(mu, np.sqrt(1/tau)).logpdf(data))
        
        # Log prior on mu
        log_prior_mu = stats.norm(prior_mean_mu, prior_std_mu).logpdf(mu)
        
        # Log prior on tau
        log_prior_tau = stats.gamma(prior_shape_tau, scale=1/prior_rate_tau).logpdf(tau)
        
        # Log posterior (up to constant)
        log_post = log_lik + log_prior_mu + log_prior_tau
        
        return -log_post  # Negative because we minimize
    
    # Initial guess
    initial_guess = [sample_mean, np.log(1/sample_var)]
    
    # Optimize
    result = optimize.minimize(neg_log_posterior, initial_guess, method='BFGS')
    
    map_mu = result.x[0]
    map_tau = np.exp(result.x[1])
    map_sigma = np.sqrt(1/map_tau)
    
    print(f"\nMAP Estimates:")
    print(f"  μ_MAP: {map_mu:.4f}")
    print(f"  σ_MAP: {map_sigma:.4f}")
    
    print(f"\nFor comparison:")
    print(f"  Sample mean: {sample_mean:.4f}")
    print(f"  Sample std:  {np.sqrt(sample_var):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fitted distribution
    axes[0].hist(data, bins=15, density=True, alpha=0.5, color='gray', edgecolor='black', label='Data')
    x = np.linspace(min(data)-1, max(data)+1, 200)
    axes[0].plot(x, stats.norm(map_mu, map_sigma).pdf(x), 'r-', linewidth=2, label=f'MAP: N({map_mu:.2f}, {map_sigma:.2f}²)')
    axes[0].plot(x, stats.norm(sample_mean, np.sqrt(sample_var)).pdf(x), 'g--', linewidth=2, label=f'MLE: N({sample_mean:.2f}, {np.sqrt(sample_var):.2f}²)')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Data with MAP and MLE Fits', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Contour of log posterior
    mu_grid = np.linspace(sample_mean-3, sample_mean+3, 100)
    sigma_grid = np.linspace(max(0.1, map_sigma-2), map_sigma+2, 100)
    MU, SIGMA = np.meshgrid(mu_grid, sigma_grid)
    
    log_post_grid = np.zeros_like(MU)
    for i in range(len(mu_grid)):
        for j in range(len(sigma_grid)):
            tau = 1 / (SIGMA[j, i] ** 2)
            log_post_grid[j, i] = -neg_log_posterior([MU[j, i], np.log(tau)])
    
    contours = axes[1].contour(MU, SIGMA, log_post_grid, levels=20, cmap='viridis')
    axes[1].plot(map_mu, map_sigma, 'r*', markersize=20, label=f'MAP')
    axes[1].plot(sample_mean, np.sqrt(sample_var), 'go', markersize=10, label='MLE')
    axes[1].set_xlabel('μ', fontsize=12)
    axes[1].set_ylabel('σ', fontsize=12)
    axes[1].set_title('Log Posterior Contours', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.colorbar(contours, ax=axes[1], label='Log Posterior')
    plt.tight_layout()
    plt.savefig('map_normal_unknown_params.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'map_mu': map_mu, 'map_sigma': map_sigma}

# ============================================================================
# CONNECTION TO REGULARIZATION
# ============================================================================

def demonstrate_map_regularization_connection(n_samples=50, noise_std=1.0):
    """
    Demonstrate the connection between MAP estimation and regularization
    in linear regression.
    """
    print("\n" + "="*70)
    print("MAP ESTIMATION = REGULARIZATION")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    true_coef = [2.0, -0.5]
    y = true_coef[0] + true_coef[1] * X + np.random.normal(0, noise_std, n_samples)
    
    # Add polynomial features to demonstrate overfitting
    X_poly = np.column_stack([X**i for i in range(6)])
    
    # 1. MLE / Least Squares (no regularization)
    beta_mle = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    
    # 2. MAP with Gaussian prior = Ridge Regression
    lambda_ridge = 10.0  # regularization parameter
    beta_map_gaussian = np.linalg.solve(X_poly.T @ X_poly + lambda_ridge * np.eye(X_poly.shape[1]), 
                                        X_poly.T @ y)
    
    # 3. MAP with Laplace prior = Lasso (using sklearn for L1)
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_poly, y)
    beta_map_laplace = np.concatenate([[lasso.intercept_], lasso.coef_[1:]])
    
    print("\nCoefficient Estimates:")
    print(f"{'Degree':<10} {'MLE':<12} {'MAP(Gaussian)':<15} {'MAP(Laplace)':<15}")
    print("-" * 55)
    for i in range(len(beta_mle)):
        print(f"{i:<10} {beta_mle[i]:>11.4f} {beta_map_gaussian[i]:>14.4f} {beta_map_laplace[i]:>14.4f}")
    
    # Predictions
    X_test = np.linspace(-1, 11, 200)
    X_test_poly = np.column_stack([X_test**i for i in range(6)])
    
    y_pred_mle = X_test_poly @ beta_mle
    y_pred_ridge = X_test_poly @ beta_map_gaussian
    y_pred_lasso = X_test_poly @ beta_map_laplace
    
    # Visualization
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, s=50, color='gray', label='Data')
    plt.plot(X_test, y_pred_mle, 'b-', linewidth=2, label='MLE (No regularization)')
    plt.plot(X_test, y_pred_ridge, 'r-', linewidth=2, label='MAP (Gaussian prior) = Ridge')
    plt.plot(X_test, y_pred_lasso, 'g-', linewidth=2, label='MAP (Laplace prior) = Lasso')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Predictions: MLE vs MAP with Different Priors', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-1, 11])
    
    plt.subplot(1, 2, 2)
    degrees = range(len(beta_mle))
    width = 0.25
    plt.bar([d - width for d in degrees], np.abs(beta_mle), width, label='MLE', alpha=0.7)
    plt.bar([d for d in degrees], np.abs(beta_map_gaussian), width, label='Ridge (Gaussian)', alpha=0.7)
    plt.bar([d + width for d in degrees], np.abs(beta_map_laplace), width, label='Lasso (Laplace)', alpha=0.7)
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('|Coefficient|', fontsize=12)
    plt.title('Coefficient Magnitudes', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('map_regularization_connection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Insight:")
    print("  - MLE overfits with high-degree polynomial")
    print("  - MAP with Gaussian prior (Ridge) shrinks coefficients")
    print("  - MAP with Laplace prior (Lasso) promotes sparsity")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 4: MAP ESTIMATION")
    print("="*70)
    
    # Example 1: MAP vs MLE for Beta-Binomial
    estimates = map_vs_mle_beta_binomial(n_heads=7, n_tails=3, prior_alpha=2, prior_beta=2)
    
    # Example 2: MAP with numerical optimization
    np.random.seed(42)
    data_normal = np.random.normal(5.0, 2.0, size=30)
    map_params = map_normal_unknown_mean_variance(data_normal)
    
    # Example 3: MAP and regularization connection
    demonstrate_map_regularization_connection()
    
    print("\n" + "="*70)
    print("MODULE 4 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. MAP = mode of posterior distribution")
    print("2. MAP incorporates prior information, unlike MLE")
    print("3. MAP ≈ Posterior mean for symmetric distributions")
    print("4. MAP with Gaussian prior = Ridge regularization")
    print("5. MAP with Laplace prior = Lasso regularization")
    print("\nNext: Module 5 - Credible Intervals")
    print("="*70)

"""
EXERCISES:

EXERCISE 1: Demonstrate that with uniform prior, MAP = MLE for Beta-Binomial.

EXERCISE 2: Show how MAP estimate changes as you increase prior strength
            (increase α, β in Beta-Binomial).

EXERCISE 3: Implement MAP for logistic regression with Gaussian prior.

EXERCISE 4: Prove that Ridge regression is equivalent to MAP with Gaussian prior.

EXERCISE 5: For what posterior distributions are MAP and posterior mean most different?
"""
