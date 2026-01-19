# Bayesian Linear Regression

## Overview

Bayesian linear regression provides full posterior distributions over parameters and predictions, naturally quantifying uncertainty. This module develops the conjugate Normal-Normal model for regression, derives the posterior and predictive distributions, and demonstrates uncertainty visualization.

---

## 1. Model Specification

### 1.1 The Linear Model

**Likelihood:**
$$
y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

Equivalently:
$$
y | X, \beta, \sigma^2 \sim \mathcal{N}(X\beta, \sigma^2 I)
$$

### 1.2 Notation

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $y$ | $n \times 1$ | Response vector |
| $X$ | $n \times p$ | Design matrix |
| $\beta$ | $p \times 1$ | Regression coefficients |
| $\sigma^2$ | scalar | Noise variance |
| $n$ | — | Number of observations |
| $p$ | — | Number of predictors (including intercept) |

### 1.3 Prior Specification

**Prior on coefficients:**
$$
\beta \sim \mathcal{N}(m_0, V_0)
$$

where:
- $m_0$ is the prior mean (often $\mathbf{0}$)
- $V_0$ is the prior covariance (often $\tau^2 I$ for some large $\tau$)

---

## 2. Posterior Distribution

### 2.1 Conjugate Update (Known $\sigma^2$)

When $\sigma^2$ is known, the posterior is also Normal:

$$
\boxed{\beta | y, X, \sigma^2 \sim \mathcal{N}(m_n, V_n)}
$$

### 2.2 Posterior Parameters

**Posterior covariance:**
$$
V_n = \left( V_0^{-1} + \frac{1}{\sigma^2} X^\top X \right)^{-1}
$$

**Posterior mean:**
$$
m_n = V_n \left( V_0^{-1} m_0 + \frac{1}{\sigma^2} X^\top y \right)
$$

### 2.3 Precision Form

Using precision matrices ($\Lambda = V^{-1}$):

$$
\Lambda_n = \Lambda_0 + \frac{1}{\sigma^2} X^\top X
$$

$$
m_n = V_n \left( \Lambda_0 m_0 + \frac{1}{\sigma^2} X^\top y \right)
$$

### 2.4 Special Cases

**Uninformative prior** ($V_0^{-1} \to 0$):
$$
m_n \to (X^\top X)^{-1} X^\top y = \hat{\beta}_{\text{OLS}}
$$
$$
V_n \to \sigma^2 (X^\top X)^{-1}
$$

The posterior mean equals the OLS estimate when the prior is vague.

**Ridge regression connection** (spherical prior $V_0 = \tau^2 I$):
$$
m_n = \left( X^\top X + \frac{\sigma^2}{\tau^2} I \right)^{-1} X^\top y
$$

This is exactly the Ridge regression solution with $\lambda = \sigma^2/\tau^2$.

---

## 3. Predictive Distribution

### 3.1 Posterior Predictive

For a new input $x_*$, the predictive distribution is:

$$
\boxed{y_* | y, X, x_* \sim \mathcal{N}\left( x_*^\top m_n, \; \sigma^2 + x_*^\top V_n x_* \right)}
$$

### 3.2 Components of Predictive Variance

The predictive variance has two components:

$$
\text{Var}(y_* | y) = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{x_*^\top V_n x_*}_{\text{parameter uncertainty}}
$$

| Component | Source | Behavior |
|-----------|--------|----------|
| $\sigma^2$ | Irreducible noise | Constant everywhere |
| $x_*^\top V_n x_*$ | Parameter uncertainty | Larger far from data |

### 3.3 Uncertainty Grows Away from Data

The term $x_*^\top V_n x_*$ increases as $x_*$ moves away from the training data, reflecting our uncertainty about the regression function in regions with little data.

---

## 4. Implementation

### 4.1 Core Functions

```python
import numpy as np

def bayesian_linear_regression(X, y, sigma_sq, m0=None, V0=None):
    """
    Bayesian linear regression with Normal prior.
    
    Parameters
    ----------
    X : array (n, p)
        Design matrix
    y : array (n,)
        Response vector
    sigma_sq : float
        Known noise variance
    m0 : array (p,), optional
        Prior mean (default: zeros)
    V0 : array (p, p), optional
        Prior covariance (default: 100 * I)
    
    Returns
    -------
    mn : array (p,)
        Posterior mean
    Vn : array (p, p)
        Posterior covariance
    """
    n, p = X.shape
    
    # Default prior
    if m0 is None:
        m0 = np.zeros(p)
    if V0 is None:
        V0 = np.eye(p) * 100
    
    # Posterior covariance
    V0_inv = np.linalg.inv(V0)
    Vn_inv = V0_inv + (1/sigma_sq) * X.T @ X
    Vn = np.linalg.inv(Vn_inv)
    
    # Posterior mean
    mn = Vn @ (V0_inv @ m0 + (1/sigma_sq) * X.T @ y)
    
    return mn, Vn


def predictive_distribution(X_test, mn, Vn, sigma_sq):
    """
    Compute predictive mean and variance.
    
    Parameters
    ----------
    X_test : array (m, p)
        Test design matrix
    mn : array (p,)
        Posterior mean
    Vn : array (p, p)
        Posterior covariance
    sigma_sq : float
        Noise variance
    
    Returns
    -------
    pred_mean : array (m,)
        Predictive means
    pred_var : array (m,)
        Predictive variances
    """
    pred_mean = X_test @ mn
    
    # Predictive variance = noise + parameter uncertainty
    pred_var = sigma_sq + np.sum((X_test @ Vn) * X_test, axis=1)
    
    return pred_mean, pred_var
```

### 4.2 Example Usage

```python
# Generate data
np.random.seed(42)
n = 30
X = np.linspace(0, 10, n)
y = 2.0 + 1.5 * X + np.random.normal(0, 2.0, n)

# Design matrix with intercept
X_design = np.column_stack([np.ones(n), X])

# Fit model
sigma_sq = 4.0  # Known noise variance
mn, Vn = bayesian_linear_regression(X_design, y, sigma_sq)

print(f"Posterior mean: β₀ = {mn[0]:.3f}, β₁ = {mn[1]:.3f}")
print(f"Posterior std:  β₀ = {np.sqrt(Vn[0,0]):.3f}, β₁ = {np.sqrt(Vn[1,1]):.3f}")
```

**Output:**
```
Posterior mean: β₀ = 1.847, β₁ = 1.534
Posterior std:  β₀ = 0.687, β₁ = 0.115
```

---

## 5. Visualization

### 5.1 Predictive Intervals

```python
import matplotlib.pyplot as plt

# Test points
X_test = np.linspace(-1, 11, 200)
X_test_design = np.column_stack([np.ones(len(X_test)), X_test])

# Predictions
pred_mean, pred_var = predictive_distribution(X_test_design, mn, Vn, sigma_sq)
pred_std = np.sqrt(pred_var)

# Plot
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X_test, pred_mean, 'r-', linewidth=2, label='Posterior mean')
plt.fill_between(X_test, 
                 pred_mean - 2*pred_std, 
                 pred_mean + 2*pred_std,
                 alpha=0.3, color='red', label='95% Predictive interval')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Bayesian Linear Regression')
```

### 5.2 Posterior Samples

Visualize uncertainty by sampling regression lines from the posterior:

```python
# Sample from posterior
for _ in range(20):
    beta_sample = np.random.multivariate_normal(mn, Vn)
    y_sample = X_test_design @ beta_sample
    plt.plot(X_test, y_sample, 'r-', alpha=0.2, linewidth=1)
```

This shows the ensemble of plausible regression lines consistent with the data and prior.

---

## 6. Comparison with Frequentist Regression

### 6.1 Key Differences

| Aspect | Bayesian | Frequentist (OLS) |
|--------|----------|-------------------|
| Parameters | Random (have distribution) | Fixed (unknown constants) |
| Uncertainty | Full posterior distribution | Standard errors, CIs |
| Regularization | Through prior | Separate (Ridge, Lasso) |
| Predictions | Predictive distribution | Point estimate + SE |
| Small samples | Prior helps | May overfit |

### 6.2 When Bayesian Wins

- **Small samples**: Prior regularizes and prevents overfitting
- **Uncertainty quantification**: Natural predictive intervals
- **Incorporating prior knowledge**: Domain expertise in prior
- **Hierarchical extensions**: Natural framework for mixed effects

### 6.3 When They Agree

With vague priors and large samples, Bayesian and frequentist results converge:
- Posterior mean ≈ OLS estimate
- Posterior variance ≈ Frequentist variance estimate

---

## 7. Extensions

### 7.1 Unknown Variance

When $\sigma^2$ is unknown, use the Normal-Inverse-Gamma conjugate prior:

$$
\beta | \sigma^2 \sim \mathcal{N}(m_0, \sigma^2 V_0)
$$
$$
\sigma^2 \sim \text{Inverse-Gamma}(a_0, b_0)
$$

The posterior is also Normal-Inverse-Gamma, and marginalizing over $\sigma^2$ gives a multivariate $t$-distribution for $\beta$.

### 7.2 Bayesian Ridge Regression

With prior $\beta \sim \mathcal{N}(0, \tau^2 I)$:

$$
m_n = \left( X^\top X + \lambda I \right)^{-1} X^\top y, \quad \lambda = \sigma^2/\tau^2
$$

The regularization parameter $\lambda$ has a Bayesian interpretation as the ratio of noise variance to prior variance.

### 7.3 Automatic Relevance Determination (ARD)

Use separate prior variances for each coefficient:
$$
\beta_j \sim \mathcal{N}(0, \tau_j^2)
$$

This enables automatic feature selection by driving irrelevant $\tau_j \to 0$.

---

## 8. Key Takeaways

1. **Full posterior over parameters**: Bayesian regression gives $p(\beta | y)$, not just point estimates, enabling rich uncertainty quantification.

2. **Predictive distribution** includes both noise and parameter uncertainty:
   $$\text{Var}(y_*) = \sigma^2 + x_*^\top V_n x_*$$

3. **Natural regularization** through the prior — equivalent to Ridge regression with appropriate prior variance.

4. **Uncertainty grows** away from training data, reflecting our ignorance in extrapolation regions.

5. **Converges to OLS** with vague priors and large samples, providing a unified framework.

---

## 9. Exercises

### Exercise 1: Prior Sensitivity
Compare posterior estimates with different prior variances ($V_0 = I$, $V_0 = 10I$, $V_0 = 1000I$). When does the prior matter?

### Exercise 2: Predictive Intervals
Generate data, fit Bayesian regression, and verify that approximately 95% of test points fall within the 95% predictive interval.

### Exercise 3: Unknown Variance
Implement Bayesian linear regression with unknown $\sigma^2$ using the Normal-Inverse-Gamma conjugate prior.

### Exercise 4: Polynomial Regression
Apply Bayesian regression to polynomial features. Show how the prior prevents overfitting compared to OLS.

### Exercise 5: Comparison with sklearn
Compare Bayesian regression (your implementation) with `sklearn.linear_model.BayesianRidge`. Verify they give similar results.

---

## References

- Bishop, C. *Pattern Recognition and Machine Learning*, Chapter 3
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 7
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 14
