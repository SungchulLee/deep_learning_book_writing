# Maximum A Posteriori (MAP) Estimation

## Overview

MAP estimation finds the mode of the posterior distribution, providing a point estimate that incorporates prior information. This module compares MAP with MLE and posterior mean, develops numerical optimization methods for MAP, and establishes the fundamental connection between MAP estimation and regularization.

---

## 1. Three Point Estimates in Bayesian Inference

### 1.1 Maximum Likelihood Estimate (MLE)

The MLE maximizes the likelihood function, ignoring prior information:

$$
\boxed{\hat{\theta}_{\text{MLE}} = \underset{\theta}{\arg\max} \; p(D|\theta)}
$$

- Frequentist approach
- No prior information incorporated
- Can overfit with limited data

### 1.2 Maximum A Posteriori (MAP)

The MAP estimate maximizes the posterior distribution:

$$
\boxed{\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \; p(\theta|D) = \underset{\theta}{\arg\max} \; p(D|\theta) \, p(\theta)}
$$

- Incorporates prior information
- Mode of the posterior distribution
- Equivalent to MLE when prior is uniform

### 1.3 Posterior Mean

The posterior mean is the expected value under the posterior:

$$
\boxed{\hat{\theta}_{\text{Mean}} = \mathbb{E}[\theta|D] = \int \theta \, p(\theta|D) \, d\theta}
$$

- Minimizes expected squared error (Bayes estimator under quadratic loss)
- Often different from MAP for skewed posteriors
- Requires integration (analytical or numerical)

### 1.4 Comparison Summary

| Estimator | Definition | Loss Function Minimized | Prior Used |
|-----------|------------|------------------------|------------|
| MLE | Mode of likelihood | — | No |
| MAP | Mode of posterior | 0-1 loss | Yes |
| Posterior Mean | Mean of posterior | Squared error | Yes |

---

## 2. MAP for Beta-Binomial Model

### 2.1 Closed-Form Solution

For the Beta-Binomial model with prior Beta$(\alpha, \beta)$ and data $(k, n-k)$:

**Posterior:** Beta$(\alpha + k, \beta + n - k)$

**MAP estimate** (mode of Beta distribution):

$$
\hat{\theta}_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2} \quad \text{for } \alpha + k > 1, \beta + n - k > 1
$$

**MLE:**

$$
\hat{\theta}_{\text{MLE}} = \frac{k}{n}
$$

**Posterior Mean:**

$$
\hat{\theta}_{\text{Mean}} = \frac{\alpha + k}{\alpha + \beta + n}
$$

### 2.2 Example Calculation

**Data:** 7 heads, 3 tails ($k=7$, $n=10$)

**Prior:** Beta$(2, 2)$

| Estimator | Formula | Value |
|-----------|---------|-------|
| MLE | $7/10$ | 0.700 |
| MAP | $(2+7-1)/(2+2+10-2) = 8/12$ | 0.667 |
| Posterior Mean | $(2+7)/(2+2+10) = 9/14$ | 0.643 |

The prior Beta$(2, 2)$ pulls estimates toward 0.5.

### 2.3 Implementation

```python
import numpy as np
from scipy import stats

def map_vs_mle_beta_binomial(n_heads, n_tails, prior_alpha=1, prior_beta=1):
    """Compare MAP, MLE, and posterior mean for Beta-Binomial."""
    
    n_total = n_heads + n_tails
    
    # MLE
    mle = n_heads / n_total if n_total > 0 else 0.5
    
    # Posterior parameters
    post_alpha = prior_alpha + n_heads
    post_beta = prior_beta + n_tails
    
    # Posterior Mean
    posterior_mean = post_alpha / (post_alpha + post_beta)
    
    # MAP (mode of Beta distribution)
    if post_alpha > 1 and post_beta > 1:
        map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        map_estimate = posterior_mean  # Use mean if mode undefined
    
    return {
        'mle': mle,
        'map': map_estimate,
        'posterior_mean': posterior_mean
    }
```

---

## 3. MAP with Numerical Optimization

### 3.1 When Closed-Form Solutions Don't Exist

For complex models, we maximize the log-posterior numerically:

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\max} \left[ \log p(D|\theta) + \log p(\theta) \right]
$$

Equivalently, minimize the negative log-posterior:

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\min} \left[ -\log p(D|\theta) - \log p(\theta) \right]
$$

### 3.2 Example: Normal with Unknown Mean and Variance

**Model:**
- Data: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$
- Prior on mean: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$
- Prior on precision: $\tau = 1/\sigma^2 \sim \text{Gamma}(\alpha, \beta)$

**Negative log-posterior:**

$$
-\log p(\mu, \tau | D) = -\sum_{i=1}^n \log p(x_i|\mu, \tau) - \log p(\mu) - \log p(\tau) + \text{const}
$$

### 3.3 Implementation

```python
from scipy import optimize

def map_normal_unknown_mean_variance(data, prior_mean_mu=0, prior_std_mu=10,
                                     prior_shape_tau=2, prior_rate_tau=1):
    """MAP estimation for Normal with unknown mean and variance."""
    
    data = np.asarray(data)
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    
    def neg_log_posterior(params):
        mu, log_tau = params
        tau = np.exp(log_tau)  # Reparametrize for unconstrained optimization
        
        # Log likelihood
        log_lik = np.sum(stats.norm(mu, np.sqrt(1/tau)).logpdf(data))
        
        # Log priors
        log_prior_mu = stats.norm(prior_mean_mu, prior_std_mu).logpdf(mu)
        log_prior_tau = stats.gamma(prior_shape_tau, 
                                    scale=1/prior_rate_tau).logpdf(tau)
        
        return -(log_lik + log_prior_mu + log_prior_tau)
    
    # Optimize
    initial_guess = [sample_mean, np.log(1/sample_var)]
    result = optimize.minimize(neg_log_posterior, initial_guess, method='BFGS')
    
    map_mu = result.x[0]
    map_tau = np.exp(result.x[1])
    map_sigma = np.sqrt(1/map_tau)
    
    return {'map_mu': map_mu, 'map_sigma': map_sigma}
```

### 3.4 Optimization Tips

| Technique | Purpose |
|-----------|---------|
| Log-transform positive parameters | Unconstrained optimization |
| Use log-posterior (not posterior) | Numerical stability |
| Multiple initializations | Avoid local optima |
| Gradient-based methods (BFGS, L-BFGS) | Efficient for smooth posteriors |

---

## 4. MAP Estimation and Regularization

### 4.1 The Fundamental Connection

MAP estimation with specific priors is equivalent to regularized maximum likelihood:

$$
\hat{\theta}_{\text{MAP}} = \underset{\theta}{\arg\min} \left[ -\log p(D|\theta) + \lambda \cdot R(\theta) \right]
$$

where $R(\theta)$ is a regularization penalty determined by the prior.

### 4.2 Gaussian Prior ↔ Ridge Regression (L2)

**Prior:** $\theta_j \sim \mathcal{N}(0, \sigma_\theta^2)$ independently

**Log-prior:** $\log p(\theta) \propto -\frac{1}{2\sigma_\theta^2} \sum_j \theta_j^2 = -\frac{\lambda}{2} \|\theta\|_2^2$

**MAP objective for linear regression:**

$$
\boxed{\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 \right]}
$$

### 4.3 Laplace Prior ↔ Lasso Regression (L1)

**Prior:** $\theta_j \sim \text{Laplace}(0, b)$ independently

**Log-prior:** $\log p(\theta) \propto -\frac{1}{b} \sum_j |\theta_j| = -\lambda \|\theta\|_1$

**MAP objective for linear regression:**

$$
\boxed{\hat{\beta}_{\text{Lasso}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right]}
$$

### 4.4 Summary of Prior-Regularization Correspondence

| Prior Distribution | Regularization | Penalty Term | Effect |
|-------------------|----------------|--------------|--------|
| Uniform | None (MLE) | — | No shrinkage |
| Gaussian $\mathcal{N}(0, \sigma^2)$ | Ridge (L2) | $\lambda\|\theta\|_2^2$ | Shrinks coefficients |
| Laplace$(0, b)$ | Lasso (L1) | $\lambda\|\theta\|_1$ | Sparse solutions |
| Horseshoe | Adaptive shrinkage | Complex | Strong sparsity |

### 4.5 Demonstration

```python
import numpy as np
from sklearn.linear_model import Lasso

def demonstrate_map_regularization(n_samples=50, noise_std=1.0):
    """Show MAP = regularization for linear regression."""
    
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    y = 2.0 - 0.5 * X + np.random.normal(0, noise_std, n_samples)
    
    # Polynomial features (prone to overfitting)
    X_poly = np.column_stack([X**i for i in range(6)])
    
    # 1. MLE (no regularization)
    beta_mle = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    
    # 2. MAP with Gaussian prior = Ridge
    lambda_ridge = 10.0
    beta_ridge = np.linalg.solve(
        X_poly.T @ X_poly + lambda_ridge * np.eye(6),
        X_poly.T @ y
    )
    
    # 3. MAP with Laplace prior = Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_poly, y)
    beta_lasso = np.concatenate([[lasso.intercept_], lasso.coef_[1:]])
    
    return beta_mle, beta_ridge, beta_lasso
```

**Key observations:**

1. **MLE** overfits with high-degree polynomials (large coefficients)
2. **Ridge (Gaussian prior)** shrinks all coefficients toward zero
3. **Lasso (Laplace prior)** drives some coefficients exactly to zero (sparsity)

---

## 5. When Do MAP and Posterior Mean Differ?

### 5.1 Symmetric Posteriors

For **symmetric** posterior distributions (e.g., Normal), the mode and mean coincide:

$$
\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{Mean}}
$$

### 5.2 Skewed Posteriors

For **skewed** posteriors (e.g., Gamma, Log-Normal, Beta with $\alpha \neq \beta$), they differ:

| Distribution | Mode | Mean | Relationship |
|--------------|------|------|--------------|
| Gamma$(\alpha, \beta)$ | $(\alpha-1)/\beta$ | $\alpha/\beta$ | Mean > Mode |
| Beta$(\alpha, \beta)$, $\alpha > \beta$ | $(\alpha-1)/(\alpha+\beta-2)$ | $\alpha/(\alpha+\beta)$ | Depends on params |

### 5.3 Choosing Between Estimators

| Criterion | Preferred Estimator |
|-----------|---------------------|
| Minimize squared error | Posterior Mean |
| Most probable value | MAP |
| Computational simplicity | MAP (optimization) |
| Full uncertainty quantification | Full posterior |

---

## 6. Key Takeaways

1. **MAP = mode of posterior**: It finds the single most probable parameter value given data and prior.

2. **MAP incorporates prior information**, unlike MLE. With a uniform prior, MAP reduces to MLE.

3. **MAP ≈ Posterior Mean** for symmetric posteriors, but they can differ substantially for skewed distributions.

4. **MAP with Gaussian prior = Ridge regularization (L2)**: The prior variance controls regularization strength.

5. **MAP with Laplace prior = Lasso regularization (L1)**: Promotes sparse solutions by driving coefficients to exactly zero.

6. **Regularization is Bayesian**: Every regularization penalty corresponds to a prior distribution on parameters.

---

## 7. Exercises

### Exercise 1: Uniform Prior
Demonstrate analytically that with a uniform prior (Beta$(1,1)$), MAP equals MLE for the Beta-Binomial model.

### Exercise 2: Prior Strength
Show how the MAP estimate changes as you increase prior strength (increase $\alpha, \beta$ symmetrically in Beta-Binomial). Plot MAP vs prior strength for fixed data.

### Exercise 3: Logistic Regression
Implement MAP estimation for logistic regression with a Gaussian prior on coefficients. Compare to unregularized MLE.

### Exercise 4: Ridge Derivation
Prove mathematically that Ridge regression is equivalent to MAP estimation with independent Gaussian priors on each coefficient.

### Exercise 5: Skewed Posteriors
For what posterior distributions are MAP and posterior mean most different? Generate examples with Gamma and Beta distributions showing the divergence.

---

## References

- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 7
- Bishop, C. *Pattern Recognition and Machine Learning*, Chapter 3
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 5
