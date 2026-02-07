# Grid Approximation for Bayesian Inference

## Overview

Grid approximation is the simplest numerical method for computing posterior distributions. This module develops the algorithm, demonstrates its application to coin flipping and normal mean inference, analyzes the effect of grid resolution, and reveals why the curse of dimensionality necessitates more sophisticated methods like MCMC.

---

## 1. The Central Challenge of Bayesian Inference

### 1.1 Bayes' Theorem Revisited

$$
p(\theta | D) = \frac{p(D | \theta) \, p(\theta)}{p(D)}
$$

The denominator, called the **evidence** or **marginal likelihood**, is:

$$
p(D) = \int p(D | \theta) \, p(\theta) \, d\theta
$$

This integral is usually **impossible to compute analytically**.

### 1.2 The Key Insight: We Don't Need $p(D)$

Bayesian computation only requires the posterior **up to a constant**:

$$
\boxed{p(\theta | D) \propto p(D | \theta) \, p(\theta)}
$$

We can compute:
1. **Unnormalized posterior**: $\tilde{p}(\theta_i) = p(D | \theta_i) \cdot p(\theta_i)$
2. **Normalize**: $p(\theta_i | D) = \tilde{p}(\theta_i) / \sum_j \tilde{p}(\theta_j)$

The denominator $\sum_j \tilde{p}(\theta_j)$ plays the role of $p(D)$, but we never compute it symbolically.

### 1.3 This Principle Generalizes

| Method | How Normalization is Handled |
|--------|------------------------------|
| Grid approximation | Sum over grid points |
| Importance sampling | Sum of weights |
| MCMC | Acceptance ratios cancel normalizing constant |
| Score-based/Diffusion | Learn $\nabla \log p(x)$ (constant cancels in gradient) |

---

## 2. The Grid Approximation Algorithm

### 2.1 Algorithm Steps

1. **Define grid**: Create discrete values $\theta_1, \theta_2, \ldots, \theta_n$ covering the parameter space
2. **Evaluate prior**: Compute $p(\theta_i)$ at each grid point
3. **Evaluate likelihood**: Compute $p(D | \theta_i)$ at each grid point
4. **Compute unnormalized posterior**: $\tilde{p}(\theta_i) = p(D | \theta_i) \cdot p(\theta_i)$
5. **Normalize**: $p(\theta_i | D) = \tilde{p}(\theta_i) / \sum_j \tilde{p}(\theta_j) \cdot \Delta\theta$

### 2.2 Implementation

```python
import numpy as np
np.random.seed(42)  # For reproducibility

from scipy.stats import beta, binom

def grid_approximation(theta_grid, prior_func, likelihood_func):
    """
    Generic grid approximation for 1D Bayesian inference.
    
    Parameters
    ----------
    theta_grid : array
        Grid of parameter values
    prior_func : callable
        Function returning prior density at each theta
    likelihood_func : callable
        Function returning likelihood at each theta
    
    Returns
    -------
    posterior : array
        Normalized posterior density at each grid point
    """
    grid_width = theta_grid[1] - theta_grid[0]
    
    # Evaluate prior and likelihood
    prior = prior_func(theta_grid)
    likelihood = likelihood_func(theta_grid)
    
    # Unnormalized posterior
    unnormalized = prior * likelihood
    
    # Normalize (approximate integral)
    normalization = np.sum(unnormalized) * grid_width
    posterior = unnormalized / normalization
    
    return posterior
```

---

## 3. Example: Coin Flip Inference

### 3.1 Problem Setup

- **Data**: $k$ heads in $n$ flips
- **Parameter**: $\theta = P(\text{heads}) \in [0, 1]$
- **Prior**: $\theta \sim \text{Beta}(\alpha, \beta)$
- **Likelihood**: $p(k | n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}$
- **Analytical posterior**: $\text{Beta}(\alpha + k, \beta + n - k)$

### 3.2 Implementation

```python
import numpy as np
np.random.seed(42)

from scipy.stats import beta, binom

# Data: 7 heads in 10 flips
n_flips, n_heads = 10, 7

# Prior: Beta(1, 1) = Uniform
prior_alpha, prior_beta = 1, 1

# Create grid
n_grid = 1000
theta_grid = np.linspace(0.001, 0.999, n_grid)
grid_width = theta_grid[1] - theta_grid[0]

# Evaluate prior and likelihood
prior_grid = beta.pdf(theta_grid, prior_alpha, prior_beta)
likelihood_grid = binom.pmf(n_heads, n_flips, theta_grid)

# Compute posterior
unnormalized = prior_grid * likelihood_grid
posterior_grid = unnormalized / (np.sum(unnormalized) * grid_width)

# Compare with analytical solution
analytical = beta.pdf(theta_grid, prior_alpha + n_heads, prior_beta + n_flips - n_heads)

# Compute statistics
post_mean = np.sum(theta_grid * posterior_grid * grid_width)
post_var = np.sum((theta_grid - post_mean)**2 * posterior_grid * grid_width)
analytical_mean = (prior_alpha + n_heads) / (prior_alpha + prior_beta + n_flips)

print(f"Grid Posterior Mean: {post_mean:.6f}")
print(f"Analytical Mean:     {analytical_mean:.6f}")
print(f"Grid Posterior Std:  {np.sqrt(post_var):.6f}")
print(f"Max Error: {np.max(np.abs(posterior_grid - analytical)):.2e}")
```

**Output:**
```
Grid Posterior Mean: 0.666667
Analytical Mean:     0.666667
Grid Posterior Std:  0.130744
Max Error: 3.22e-11
```

### 3.3 Visualization

![Coin Flip Grid Approximation](figures/coin_flip_grid.png)

The four-panel figure shows:
- **Top-left**: Uniform prior Beta(1,1)
- **Top-right**: Binomial likelihood for 7 heads in 10 flips
- **Bottom-left**: Unnormalized posterior (prior × likelihood)
- **Bottom-right**: Normalized posterior vs analytical Beta(8,4)

### 3.4 Verification

| Quantity | Grid Approximation | Analytical |
|----------|-------------------|------------|
| Posterior Mean | 0.666667 | 0.666667 |
| Posterior Std | 0.130744 | 0.130744 |
| Error | ~$10^{-11}$ | — |

Grid approximation reproduces the analytical result to machine precision.

---

## 4. Effect of Grid Resolution

### 4.1 Accuracy vs Computation Trade-off

```python
grid_sizes = [10, 20, 50, 100, 200, 500, 1000]
```

**Results:**

| Grid Size | Mean Error | Std Error |
|-----------|------------|-----------|
| 10 | 4.02e-04 | 3.05e-04 |
| 20 | 2.50e-05 | 2.50e-05 |
| 50 | 5.60e-07 | 5.88e-07 |
| 100 | 2.73e-08 | 2.88e-08 |
| 200 | 4.80e-10 | 4.88e-10 |
| 500 | 5.13e-11 | 5.54e-11 |
| 1000 | 3.65e-12 | 3.94e-12 |

### 4.2 Error Scaling

![Grid Resolution Analysis](figures/grid_resolution_analysis.png)

The error decreases approximately as $O(1/n^2)$ where $n$ is the number of grid points:

$$
\text{Error} \approx \frac{C}{n^2}
$$

### 4.3 Practical Recommendation

For 1D problems:
- **500-1000 points**: Usually sufficient for ~$10^{-5}$ accuracy
- **Check convergence**: Verify results don't change with more points
- **Check boundaries**: Ensure posterior ≈ 0 at grid edges

---

## 5. Normal Mean Inference

### 5.1 Problem Setup

- **Data**: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma$ known
- **Prior**: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$
- **Posterior**: $\mu | D \sim \mathcal{N}(\mu_n, \sigma_n^2)$ (conjugate)

### 5.2 Analytical Posterior

$$
\mu_n = \frac{\tau^{-2} \mu_0 + n\sigma^{-2} \bar{x}}{\tau^{-2} + n\sigma^{-2}}, \quad \sigma_n^2 = \frac{1}{\tau^{-2} + n\sigma^{-2}}
$$

### 5.3 Grid Approximation with Log-Likelihood

For numerical stability with many data points:

```python
import numpy as np
np.random.seed(42)

from scipy.stats import norm

# Generate data
true_mu = 5.0
known_sigma = 2.0
n_data = 20
data = np.random.normal(true_mu, known_sigma, n_data)

# Prior
prior_mu = 0.0
prior_tau = 10.0  # prior std

# Grid
n_grid = 1000
data_mean = np.mean(data)
mu_grid = np.linspace(data_mean - 3*known_sigma, data_mean + 3*known_sigma, n_grid)
dmu = mu_grid[1] - mu_grid[0]

# Prior
prior_grid = norm.pdf(mu_grid, prior_mu, prior_tau)

# Log-likelihood (for numerical stability)
log_likelihood = np.zeros(n_grid)
for x in data:
    log_likelihood += norm.logpdf(x, mu_grid, known_sigma)

# Subtract max before exp to prevent overflow
log_likelihood -= log_likelihood.max()
likelihood_grid = np.exp(log_likelihood)

# Posterior
unnormalized = prior_grid * likelihood_grid
posterior = unnormalized / (np.sum(unnormalized) * dmu)

# Statistics
grid_mean = np.sum(mu_grid * posterior * dmu)
grid_std = np.sqrt(np.sum((mu_grid - grid_mean)**2 * posterior * dmu))

print(f"True μ: {true_mu}")
print(f"Sample mean: {data_mean:.4f}")
print(f"Grid Posterior Mean: {grid_mean:.4f}")
print(f"Grid Posterior Std: {grid_std:.4f}")
```

**Output:**
```
True μ: 5.0
Sample mean: 4.6574
Grid Posterior Mean: 4.6481
Grid Posterior Std: 0.4465
```

### 5.4 Visualization

![Normal Mean Inference](figures/normal_mean_inference.png)

---

## 6. The Curse of Dimensionality

### 6.1 Exponential Growth

For $d$ dimensions with $n$ points per dimension:

| Dimensions | Grid Points | Memory (float64) |
|------------|-------------|------------------|
| 1 | 100 | 0.8 KB |
| 2 | 10,000 | 80 KB |
| 3 | 1,000,000 | 8 MB |
| 4 | 100,000,000 | 800 MB |
| 5 | 10,000,000,000 | 80 GB |
| 6 | 1,000,000,000,000 | 8 TB |

$$
\text{Grid points} = n^d \quad \text{(exponential in dimension)}
$$

### 6.2 Visualization

![Curse of Dimensionality](figures/curse_of_dimensionality.png)

### 6.3 Why Grid Fails in High Dimensions

- **Memory**: Cannot store $n^d$ values
- **Computation**: Cannot evaluate likelihood $n^d$ times
- **Most volume is empty**: In high dimensions, most grid points have negligible posterior probability

### 6.4 Grid vs MCMC Scaling

| Dimension | Grid Points (n=100) | MCMC Samples (typical) |
|-----------|---------------------|------------------------|
| 1 | 100 | 1,000 |
| 2 | 10,000 | 1,000 |
| 5 | 10 billion | 10,000 |
| 10 | $10^{20}$ | 10,000 |
| 100 | $10^{200}$ | 50,000 |

**MCMC samples scale gracefully with dimension!**

### 6.5 Connection to Diffusion Models

Images are extremely high-dimensional:
- 256×256×3 RGB image = **196,608 dimensions**
- Grid approximation is completely hopeless
- This is why we need score-based methods and diffusion models

---

## 7. Numerical Stability

### 7.1 The Underflow Problem

When multiplying many small probabilities:

```python
# BAD: Direct multiplication
likelihood = 1.0
for x in data:
    likelihood *= norm.pdf(x, mu, sigma)  # Quickly becomes 0!
```

### 7.2 The Log-Probability Solution

```python
# GOOD: Log probabilities
log_likelihood = 0.0
for x in data:
    log_likelihood += norm.logpdf(x, mu, sigma)  # Stays finite

# Subtract max before exponentiating
log_likelihood -= log_likelihood.max()
likelihood = np.exp(log_likelihood)  # Now well-scaled
```

### 7.3 Demonstration

```python
data_sizes = [10, 50, 100, 200, 500, 1000]
```

**Results:**

| n | Direct Max | Log Method Max |
|---|------------|----------------|
| 10 | 9.70e-06 | 1.000000 |
| 50 | 2.74e-29 | 1.000000 |
| 100 | 3.48e-60 | 1.000000 |
| 200 | 2.64e-120 | 1.000000 |
| 500 | 1.64e-308 | 1.000000 |
| 1000 | **0.0 (underflow!)** | 1.000000 |

### 7.4 Visualization

![Numerical Stability](figures/numerical_stability.png)

### 7.5 Best Practices Summary

| Practice | Reason |
|----------|--------|
| Use log probabilities | Prevent underflow in likelihood products |
| Subtract max before exp() | Prevent overflow |
| Check boundary values ≈ 0 | Ensure grid covers posterior support |
| Use enough grid points (1000+) | Adequate resolution |
| Multiply by grid width $\Delta\theta$ | Proper numerical integration |

---

## 8. When to Use Grid Approximation

### 8.1 Appropriate Uses

✓ **1D problems**: Almost always fine  
✓ **2D problems**: Usually OK  
✓ **Teaching and visualization**: Excellent for building intuition  
✓ **Verifying other methods**: Quick sanity check  

### 8.2 When to Use Other Methods

✗ **3D+ problems**: Consider MCMC  
✗ **High-dimensional**: Use MCMC, variational inference, or score-based methods  
✗ **Complex posteriors**: Adaptive methods may be needed  

---

## 9. The Path Forward

Grid approximation teaches the fundamental principle:

$$
\text{Posterior} \propto \text{Prior} \times \text{Likelihood}
$$

But its limitations motivate more advanced methods:

| Method | Overcomes |
|--------|-----------|
| **MCMC** | Curse of dimensionality |
| **Langevin Dynamics** | Slow MCMC convergence (uses gradients) |
| **Variational Inference** | Computational cost of sampling |
| **Score Matching/Diffusion** | Learning unnormalized densities |

**The journey:**
$$
\text{Grid} \to \text{MCMC} \to \text{Langevin} \to \text{Score Matching} \to \text{Diffusion}
$$

Each method solves the limitations of the previous one.

---

## 10. Key Takeaways

1. **Grid approximation** is the simplest numerical Bayesian method: discretize parameter space, evaluate prior × likelihood, normalize.

2. **We never need $p(D)$ explicitly** — this insight underlies all Bayesian computation.

3. **Error scales as $O(1/n^2)$** where $n$ is grid size. 500-1000 points usually suffice for 1D.

4. **Use log probabilities** to prevent numerical underflow when computing likelihoods.

5. **Curse of dimensionality**: Grid points grow as $n^d$, making grid approximation useless for $d \geq 3$.

6. **Grid approximation prepares intuition** for MCMC, Langevin dynamics, and diffusion models.

---

## 11. Exercises

### Exercise 1: Beta-Binomial Verification
Implement grid approximation for different Beta priors (informative vs uninformative). Verify against analytical posteriors.

### Exercise 2: Resolution Analysis
For a coin flip problem, plot posterior mean error vs grid size on a log-log scale. Verify the $O(1/n^2)$ scaling.

### Exercise 3: 2D Grid
Implement grid approximation for a 2D problem (e.g., normal mean and variance). Visualize as a contour plot. Note the computational cost.

### Exercise 4: Numerical Stability
Generate data that causes underflow without log probabilities. Implement both methods and compare.

### Exercise 5: When Does Grid Fail?
Estimate the maximum dimension where grid approximation with n=100 points per dimension is feasible on your computer (memory < 8GB).

---

## References

- McElreath, R. *Statistical Rethinking* (2nd ed.), Chapter 2
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 3
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 5

---

# Grid Approximation: Intermediate (2D)

## Overview

This module extends grid approximation to two-dimensional parameter spaces, covering bivariate posteriors, marginal and conditional distributions, correlation visualization, and Bayesian linear regression. We also examine computational efficiency and the practical limits of grid methods.

---

## 1. 2D Grid Approximation Framework

### 1.1 Mathematical Foundation

For two parameters $\theta = (\theta_1, \theta_2)$:

**Joint Posterior:**
$$
p(\theta_1, \theta_2 | D) \propto p(D | \theta_1, \theta_2) \, p(\theta_1, \theta_2)
$$

**Marginal Posteriors:**
$$
p(\theta_1 | D) = \int p(\theta_1, \theta_2 | D) \, d\theta_2
$$
$$
p(\theta_2 | D) = \int p(\theta_1, \theta_2 | D) \, d\theta_1
$$

**Conditional Posteriors:**
$$
p(\theta_1 | \theta_2, D) = \frac{p(\theta_1, \theta_2 | D)}{p(\theta_2 | D)}
$$

### 1.2 Algorithm for 2D Grid

1. **Create 2D grid**: $(\theta_{1,i}, \theta_{2,j})$ for $i = 1, \ldots, n_1$ and $j = 1, \ldots, n_2$
2. **Evaluate prior**: $p(\theta_{1,i}, \theta_{2,j})$ at each point
3. **Evaluate likelihood**: $p(D | \theta_{1,i}, \theta_{2,j})$ at each point
4. **Compute unnormalized posterior**: prior × likelihood
5. **Normalize**: Divide by sum × grid area
6. **Compute marginals**: Sum over one dimension

### 1.3 Computational Cost

| Grid Size | Points | Memory (float64) | Typical Time |
|-----------|--------|------------------|--------------|
| 50 × 50 | 2,500 | 20 KB | < 10 ms |
| 100 × 100 | 10,000 | 80 KB | ~50 ms |
| 200 × 200 | 40,000 | 320 KB | ~200 ms |
| 500 × 500 | 250,000 | 2 MB | ~1 sec |

**Scaling**: $O(n^2)$ for an $n \times n$ grid.

---

## 2. Example: Bivariate Normal Inference

### 2.1 Problem Setup

- **Data**: $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \Sigma)$ with $\Sigma$ known
- **Unknown**: $\mu = (\mu_1, \mu_2)$
- **Prior**: $\mu \sim \mathcal{N}(\mu_0, \Sigma_0)$
- **Posterior**: $\mu | D \sim \mathcal{N}(\mu_n, \Sigma_n)$ (conjugate)

### 2.2 Analytical Solution

$$
\Sigma_n^{-1} = \Sigma_0^{-1} + n\Sigma^{-1}
$$
$$
\mu_n = \Sigma_n \left( \Sigma_0^{-1} \mu_0 + n\Sigma^{-1} \bar{x} \right)
$$

### 2.3 Implementation

```python
import numpy as np
np.random.seed(42)

from scipy.stats import multivariate_normal

# True parameters
true_mu = np.array([2.0, 3.0])
true_Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

# Generate data
n_data = 30
data = np.random.multivariate_normal(true_mu, true_Sigma, n_data)
data_mean = data.mean(axis=0)

# Prior
prior_mu = np.array([0.0, 0.0])
prior_Sigma = np.array([[10.0, 0.0], [0.0, 10.0]])

# Analytical posterior
prior_precision = np.linalg.inv(prior_Sigma)
data_precision = n_data * np.linalg.inv(true_Sigma)
post_precision = prior_precision + data_precision
post_Sigma = np.linalg.inv(post_precision)
post_mu = post_Sigma @ (prior_precision @ prior_mu + data_precision @ data_mean)

# Grid approximation
n_grid = 100
std1 = np.sqrt(post_Sigma[0, 0])
std2 = np.sqrt(post_Sigma[1, 1])
mu1_grid = np.linspace(post_mu[0] - 4*std1, post_mu[0] + 4*std1, n_grid)
mu2_grid = np.linspace(post_mu[1] - 4*std2, post_mu[1] + 4*std2, n_grid)
Mu1, Mu2 = np.meshgrid(mu1_grid, mu2_grid)
grid_points = np.stack([Mu1.ravel(), Mu2.ravel()], axis=1)

# Evaluate prior
prior_vals = multivariate_normal.pdf(grid_points, prior_mu, prior_Sigma)

# Evaluate likelihood (log for stability)
log_likelihood = np.zeros(len(grid_points))
for x in data:
    log_likelihood += multivariate_normal.logpdf(grid_points, x, true_Sigma)
log_likelihood -= log_likelihood.max()
likelihood_vals = np.exp(log_likelihood)

# Posterior
unnormalized = prior_vals * likelihood_vals
grid_area = (mu1_grid[1] - mu1_grid[0]) * (mu2_grid[1] - mu2_grid[0])
posterior_vals = unnormalized / (np.sum(unnormalized) * grid_area)
posterior_grid = posterior_vals.reshape(n_grid, n_grid)

# Compute grid statistics
grid_mean1 = np.sum(Mu1 * posterior_grid * grid_area)
grid_mean2 = np.sum(Mu2 * posterior_grid * grid_area)

print(f"True μ: [{true_mu[0]}, {true_mu[1]}]")
print(f"Sample mean: [{data_mean[0]:.4f}, {data_mean[1]:.4f}]")
print(f"Grid Post Mean: [{grid_mean1:.4f}, {grid_mean2:.4f}]")
print(f"Analytical Post Mean: [{post_mu[0]:.4f}, {post_mu[1]:.4f}]")
```

**Output:**
```
True μ: [2.0, 3.0]
Sample mean: [2.2074, 3.0423]
Grid Post Mean: [2.1951, 3.0285]
Analytical Post Mean: [2.1951, 3.0285]
```

### 2.4 Visualization

![Bivariate Normal Grid](figures/bivariate_normal_grid.png)

The six-panel figure shows:
- **Top-left**: Joint posterior contours (grid approximation)
- **Top-middle**: Joint posterior contours (analytical)
- **Top-right**: 3D surface of posterior density
- **Bottom-left**: Marginal distribution of $\mu_1$
- **Bottom-middle**: Marginal distribution of $\mu_2$
- **Bottom-right**: Data scatter plot

### 2.5 Computing Marginals

```python
def compute_marginals(posterior_grid, mu1_grid, mu2_grid):
    """Compute marginal distributions from joint posterior."""
    
    d_mu1 = mu1_grid[1] - mu1_grid[0]
    d_mu2 = mu2_grid[1] - mu2_grid[0]
    
    # p(μ₁|D) = ∫ p(μ₁,μ₂|D) dμ₂
    marginal_mu1 = np.sum(posterior_grid, axis=0) * d_mu2
    
    # p(μ₂|D) = ∫ p(μ₁,μ₂|D) dμ₁
    marginal_mu2 = np.sum(posterior_grid, axis=1) * d_mu1
    
    return marginal_mu1, marginal_mu2
```

---

## 3. Visualizing Parameter Correlation

### 3.1 Types of Correlation

| Correlation | Contour Shape | Interpretation |
|-------------|---------------|----------------|
| Independent ($\rho = 0$) | Circular | Parameters vary independently |
| Positive ($\rho > 0$) | Ellipse (↗ diagonal) | $\theta_1 \uparrow \Rightarrow \theta_2 \uparrow$ |
| Negative ($\rho < 0$) | Ellipse (↘ diagonal) | $\theta_1 \uparrow \Rightarrow \theta_2 \downarrow$ |

### 3.2 Why Correlation Matters

1. **MCMC efficiency**: Correlated posteriors cause samplers to explore slowly
2. **Uncertainty quantification**: Marginal uncertainties don't capture joint structure
3. **Parameter interpretation**: Correlated parameters have dependent effects
4. **Regression problems**: Intercept and slope are often negatively correlated

---

## 4. Linear Regression with 2D Grid

### 4.1 Model

$$
y_i = \alpha + \beta x_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

**Parameters**: $\theta = (\alpha, \beta)$ (intercept and slope)

### 4.2 Likelihood

$$
p(y | x, \alpha, \beta, \sigma^2) = \prod_{i=1}^n \mathcal{N}(y_i | \alpha + \beta x_i, \sigma^2)
$$

### 4.3 Implementation

```python
import numpy as np
np.random.seed(42)

from scipy.stats import norm

# True parameters
true_alpha = 1.5
true_beta = 2.0
sigma = 1.0

# Generate data
n_data = 20
x = np.random.uniform(0, 3, n_data)
y = true_alpha + true_beta * x + np.random.normal(0, sigma, n_data)

# Grid approximation
n_grid = 100
prior_std = 5.0
alpha_range = np.linspace(-1, 4, n_grid)
beta_range = np.linspace(0, 4, n_grid)
Alpha, Beta = np.meshgrid(alpha_range, beta_range)

alpha_vec = Alpha.ravel()
beta_vec = Beta.ravel()

# Prior (independent)
prior_vals = norm.pdf(alpha_vec, 0, prior_std) * norm.pdf(beta_vec, 0, prior_std)

# Likelihood
log_likelihood = np.zeros(len(alpha_vec))
for xi, yi in zip(x, y):
    y_pred = alpha_vec + beta_vec * xi
    log_likelihood += norm.logpdf(yi, y_pred, sigma)

log_likelihood -= log_likelihood.max()
likelihood_vals = np.exp(log_likelihood)

# Posterior
unnormalized = prior_vals * likelihood_vals
grid_area = (alpha_range[1] - alpha_range[0]) * (beta_range[1] - beta_range[0])
posterior_vals = unnormalized / (np.sum(unnormalized) * grid_area)
posterior_grid = posterior_vals.reshape(n_grid, n_grid)

# Statistics
alpha_mean = np.sum(Alpha * posterior_grid * grid_area)
beta_mean = np.sum(Beta * posterior_grid * grid_area)
alpha_var = np.sum((Alpha - alpha_mean)**2 * posterior_grid * grid_area)
beta_var = np.sum((Beta - beta_mean)**2 * posterior_grid * grid_area)
cov_ab = np.sum((Alpha - alpha_mean) * (Beta - beta_mean) * posterior_grid * grid_area)
corr_ab = cov_ab / (np.sqrt(alpha_var) * np.sqrt(beta_var))

print(f"True α: {true_alpha}, True β: {true_beta}")
print(f"Post Mean α: {alpha_mean:.4f} ± {np.sqrt(alpha_var):.4f}")
print(f"Post Mean β: {beta_mean:.4f} ± {np.sqrt(beta_var):.4f}")
print(f"Correlation(α, β): {corr_ab:.4f}")
```

**Output:**
```
True α: 1.5, True β: 2.0
Post Mean α: 1.4601 ± 0.4063
Post Mean β: 1.8283 ± 0.2477
Correlation(α, β): -0.8353
```

### 4.4 Visualization

![Linear Regression Grid](figures/linear_regression_grid.png)

The six-panel figure shows:
- **Top-left**: Data and regression lines (true vs posterior mean)
- **Top-middle**: Joint posterior contours showing negative correlation
- **Top-right**: 3D surface
- **Bottom-left**: Marginal of intercept $\alpha$
- **Bottom-middle**: Marginal of slope $\beta$
- **Bottom-right**: Posterior predictive (50 sample lines)

### 4.5 Key Observation: Intercept-Slope Correlation

In linear regression, $\alpha$ and $\beta$ are typically **negatively correlated** (here $\rho = -0.84$):
- If the slope is higher, the intercept must be lower to fit the same data
- This correlation increases with the mean of $x$

---

## 5. Computational Efficiency

### 5.1 Vectorization is Essential

```python
# BAD: Loop over grid points
posterior = np.zeros((n_grid, n_grid))
for i in range(n_grid):
    for j in range(n_grid):
        posterior[i, j] = compute_posterior(theta1[i], theta2[j])

# GOOD: Vectorized computation
theta1_vec = Theta1.ravel()
theta2_vec = Theta2.ravel()
posterior_vec = compute_posterior_vectorized(theta1_vec, theta2_vec)
posterior = posterior_vec.reshape(n_grid, n_grid)
```

### 5.2 Performance Tips

| Technique | Benefit |
|-----------|---------|
| **Vectorize** | 10-100x speedup |
| **Use log-probabilities** | Numerical stability |
| **Adaptive grid range** | Focus on posterior mass |
| **Sparse grids** | Memory reduction (advanced) |
| **Parallel computation** | Multi-core utilization |

### 5.3 Scaling Law

For an $n \times n$ grid:
- **Time complexity**: $O(n^2 \cdot m)$ where $m$ is data size
- **Space complexity**: $O(n^2)$

---

## 6. Limitations and Path Forward

### 6.1 The 3D Wall

| Dimension | Grid Size (n=100) | Feasibility |
|-----------|-------------------|-------------|
| 1D | 100 | ✓ Excellent |
| 2D | 10,000 | ✓ Good |
| 3D | 1,000,000 | ~ Challenging |
| 4D | 100,000,000 | ✗ Impractical |
| 5D+ | $10^{10+}$ | ✗ Impossible |

### 6.2 Why MCMC?

| Grid Approximation | MCMC |
|--------------------|------|
| Evaluates everywhere | Focuses on high-probability regions |
| $O(n^d)$ scaling | $O(\text{samples})$ scaling |
| Fixed grid resolution | Adapts to posterior shape |
| Works for $d \leq 2$ | Works for any $d$ |

### 6.3 The Path Forward

$$
\text{Grid} \xrightarrow{\text{dimension limits}} \text{MCMC} \xrightarrow{\text{use gradients}} \text{Langevin} \xrightarrow{\text{learn score}} \text{Diffusion}
$$

---

## 7. Key Takeaways

1. **2D grid approximation** works well with 100×100 = 10,000 points, enabling joint posterior visualization.

2. **Marginal distributions** are computed by integrating (summing) over the other dimension.

3. **Correlation structure** is revealed through contour plots and ellipse orientation. Correlated parameters affect MCMC sampling efficiency.

4. **Linear regression** naturally produces correlated posteriors for intercept and slope ($\rho \approx -0.84$ in our example).

5. **Vectorization is essential** — avoid loops over grid points.

6. **3D and beyond**: Grid approximation becomes impractical. This motivates MCMC methods.

---

## 8. Exercises

### Exercise 1: Different Correlation Strengths
Create 2D normal posteriors with correlations $\rho \in \{-0.9, -0.5, 0, 0.5, 0.9\}$. Visualize how the contour ellipses change.

### Exercise 2: Marginal vs Joint
For a correlated 2D posterior, show that knowing the marginals $p(\theta_1|D)$ and $p(\theta_2|D)$ is not enough to reconstruct the joint $p(\theta_1, \theta_2|D)$.

### Exercise 3: Conditional Distributions
Implement computation of conditional distributions $p(\theta_1 | \theta_2 = c, D)$ from the 2D grid. Visualize how the conditional changes as $c$ varies.

### Exercise 4: Regression Correlation
For linear regression, derive analytically why $\text{Corr}(\alpha, \beta)$ depends on $\bar{x}$. Verify with grid approximation.

### Exercise 5: 3D Attempt
Try to implement 3D grid approximation for a simple problem. Document the memory and time requirements. At what grid size does it become impractical?

---

## References

- McElreath, R. *Statistical Rethinking* (2nd ed.), Chapter 4
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 3
- Murphy, K. *Machine Learning: A Probabilistic Perspective*, Chapter 7

---

# Grid Approximation: Advanced Topics

## Overview

This module covers advanced grid approximation techniques: adaptive refinement for complex posteriors, importance-weighted grids bridging deterministic and Monte Carlo methods, rigorous convergence analysis, and connections to modern computational methods. We establish why MCMC becomes necessary for high-dimensional inference.

---

## 1. Theoretical Framework

### 1.1 Grid Approximation as Quadrature

Grid approximation computes posterior expectations via numerical integration:

$$
\mathbb{E}[f(\theta)|D] = \int f(\theta) \, p(\theta|D) \, d\theta \approx \sum_{i=1}^n f(\theta_i) \, p(\theta_i|D) \, \Delta\theta
$$

### 1.2 Error Analysis

For a uniform grid with $n$ points per dimension in $d$ dimensions:

$$
\boxed{\text{Error} = O(n^{-r/d})}
$$

where $r$ is the smoothness order (typically $r = 2$ for twice-differentiable functions).

**Implications:**
- **1D**: Error $\sim n^{-2}$ (quadratic convergence)
- **2D**: Error $\sim n^{-1}$ (linear convergence)
- **10D**: Error $\sim n^{-0.2}$ (very slow!)

### 1.3 Points Needed for Target Accuracy

To achieve error $\varepsilon$:

$$
n \geq \varepsilon^{-d/r}
$$

| Dimension | Points per Dim | Total Points | Feasible? |
|-----------|---------------|--------------|-----------|
| 1D | 100 | 100 | ✓ |
| 2D | 100 | 10,000 | ✓ |
| 3D | 100 | 1,000,000 | ~ |
| 5D | 100 | 10 billion | ✗ |
| 10D | 100 | $10^{20}$ | ✗ |

---

## 2. Adaptive Grid Refinement

### 2.1 The Problem with Uniform Grids

Uniform grids waste computational resources in low-probability regions while potentially under-sampling high-probability regions.

### 2.2 Adaptive Strategy

1. **Start with coarse uniform grid**
2. **Evaluate posterior** at all points
3. **Identify high-probability regions** (e.g., above median)
4. **Refine grid** in those regions
5. **Iterate** until convergence

### 2.3 Implementation

```python
import numpy as np
np.random.seed(42)

from scipy.stats import norm

def create_adaptive_grid(posterior_func, x_min, x_max, n_initial=20):
    """
    Create adaptive grid with more points in high-probability regions.
    """
    # Initial coarse grid
    x_coarse = np.linspace(x_min, x_max, n_initial)
    p_coarse = posterior_func(x_coarse)
    
    # Find high-probability regions (top 50%)
    threshold = np.percentile(p_coarse, 50)
    high_prob_indices = np.where(p_coarse > threshold)[0]
    
    # Add fine grid in high-probability regions
    x_adaptive = [x_coarse]
    for i in high_prob_indices:
        if i < len(x_coarse) - 1:
            x_fine = np.linspace(x_coarse[i], x_coarse[i+1], 10)[1:-1]
            x_adaptive.append(x_fine)
    
    return np.sort(np.concatenate(x_adaptive))
```

### 2.4 Example: Bimodal Posterior

For a mixture of two Gaussians:

$$
p(\theta|D) \propto 0.6 \cdot \mathcal{N}(\theta|-2, 0.5^2) + 0.4 \cdot \mathcal{N}(\theta|3, 0.8^2)
$$

**Output:**
```
True Mean: -0.000213
Uniform Grid (50 pts): -0.000120 (error = 0.000093)
Adaptive Grid (92 pts): -0.003602 (error = 0.003389)
```

### 2.5 Visualization

![Adaptive Grid Bimodal](figures/adaptive_grid_bimodal.png)

### 2.6 When to Use Adaptive Grids

✓ **Good for:**
- Multimodal posteriors
- Unknown posterior structure
- Need high accuracy with limited points

✗ **Limitations:**
- Still exponential in dimension
- Requires initial exploration
- More complex implementation

---

## 3. Importance-Weighted Grid Approximation

### 3.1 Mathematical Foundation

Instead of uniform sampling, sample from a proposal $q(\theta)$ and reweight:

$$
\mathbb{E}[f(\theta)|D] = \int f(\theta) \frac{p(\theta|D)}{q(\theta)} q(\theta) \, d\theta \approx \frac{1}{n} \sum_{i=1}^n f(\theta_i) \, w(\theta_i)
$$

where $\theta_i \sim q(\theta)$ and the **importance weight** is:

$$
w(\theta_i) = \frac{p(\theta_i|D)}{q(\theta_i)}
$$

### 3.2 Effective Sample Size

The **Effective Sample Size (ESS)** measures how many unweighted samples the importance samples are worth:

$$
\text{ESS} = \frac{1}{\sum_{i=1}^n w_i^2}
$$

where $w_i$ are normalized weights. ESS ranges from 1 (one dominant sample) to $n$ (uniform weights).

### 3.3 Connection to Modern Methods

| Method | Relationship to Importance Sampling |
|--------|-------------------------------------|
| **Particle Filters** | Sequential importance resampling |
| **SMC** | Importance sampling with tempering |
| **Annealed IS** | Bridge from prior to posterior |
| **Variational Inference** | Optimize proposal to minimize KL divergence |

---

## 4. Convergence Rate Analysis

### 4.1 Dimensional Scaling

The convergence rate degrades with dimension:

| Dimension | Convergence Rate | Implication |
|-----------|------------------|-------------|
| 1D | $O(n^{-2})$ | Fast convergence |
| 2D | $O(n^{-1})$ | Moderate |
| 3D | $O(n^{-2/3})$ | Slow |
| 10D | $O(n^{-1/5})$ | Very slow |
| 100D | $O(n^{-1/50})$ | Hopeless |

### 4.2 The Fundamental Limitation

**Theorem (Curse of Dimensionality):**

For grid-based methods with smooth integrands:
$$
\text{Total evaluations} = O(\varepsilon^{-d})
$$

to achieve error $\varepsilon$ in $d$ dimensions.

**This is why MCMC is not just convenient—it's mathematically necessary.**

---

## 5. Connections to Modern Methods

### 5.1 Evolution of Bayesian Computation

| Era | Method | Key Idea |
|-----|--------|----------|
| 1960s | Grid Approximation | Direct computation |
| 1980s-90s | MCMC | Strategic random sampling |
| 2000s | Hamiltonian MC | Gradient-guided sampling |
| 2010s | Variational Inference | Optimization-based approximation |
| 2020s | Diffusion Models | Iterative score-based refinement |

### 5.2 What Grid Methods Teach Us

| Concept | Modern Application |
|---------|-------------------|
| Numerical integration | Foundation for all posterior computation |
| Curse of dimensionality | Why we need efficient samplers |
| Importance weighting | Particle methods, SMC |
| Adaptive refinement | Adaptive MCMC |
| Normalization | Score functions avoid this |

### 5.3 The Path Forward

$$
\text{Grid} \xrightarrow{\text{dimension}} \text{MCMC} \xrightarrow{\text{gradients}} \text{HMC/Langevin} \xrightarrow{\text{learn score}} \text{Diffusion}
$$

Each method solves limitations of the previous:
- **Grid → MCMC**: Handles high dimensions
- **MCMC → HMC**: Uses gradients for faster mixing
- **HMC → Diffusion**: Learns score function from data

---

## 6. Key Takeaways

1. **Adaptive grids** can significantly improve efficiency for complex posteriors, especially multimodal distributions.

2. **Importance sampling** bridges grid methods and Monte Carlo, providing variance reduction with good proposals.

3. **Convergence rate** is $O(n^{-2/d})$ for $d$-dimensional problems — exponentially worse with dimension.

4. **The curse of dimensionality** is a fundamental mathematical limitation, not just a practical inconvenience.

5. **Grid methods lay the foundation** for understanding MCMC, variational inference, and modern diffusion models.

6. **MCMC is necessary** (not just convenient) for high-dimensional Bayesian inference.

---

## 7. Exercises

### Exercise 1: Adaptive Grid for Multimodal
Implement adaptive grid refinement for a mixture of 3 Gaussians. Compare accuracy with uniform grid.

### Exercise 2: ESS Behavior
Study how ESS varies with proposal width. What happens when the proposal is too narrow? Too wide?

### Exercise 3: Dimensional Scaling
Empirically verify the $O(n^{-2/d})$ convergence rate for $d = 1, 2, 3$. Plot error vs grid size.

### Exercise 4: Compare with MCMC
For a 3D problem, compare grid approximation with MCMC (Metropolis-Hastings). At what accuracy level does MCMC become more efficient?

---

## References

- Robert, C., & Casella, G. *Monte Carlo Statistical Methods* (2nd ed.), Chapters 3-4
- Liu, J. S. *Monte Carlo Strategies in Scientific Computing*, Chapters 2-3
- Gelman, A., et al. *Bayesian Data Analysis* (3rd ed.), Chapter 10

---

## Transition to Monte Carlo Methods

The curse of dimensionality demonstrated above motivates Monte Carlo integration. Instead of exhaustive grid evaluation, Monte Carlo methods draw samples from the target distribution and compute sample averages. The key insight: Monte Carlo convergence rate is $O(1/\sqrt{N})$ regardless of dimensionality, making it practical for the high-dimensional posteriors encountered in modern machine learning.

See [Rejection Sampling](rejection.md), [Importance Sampling](importance_sampling.md), and the MCMC methods in Section 18.3 for practical algorithms.

---

## 8. Monte Carlo Estimation

The curse of dimensionality motivates replacing deterministic grid evaluation with random sampling. Monte Carlo estimation leverages the Law of Large Numbers to approximate integrals.

### The Basic Principle

If $X_1, X_2, \ldots, X_n \stackrel{\text{i.i.d.}}{\sim} p(x)$, then:

$$
\hat{I}_{\text{MC}} = \frac{1}{n} \sum_{i=1}^n h(X_i) \xrightarrow{a.s.} \mathbb{E}_p[h(X)] = I
$$

### Unbiasedness and Variance

**Unbiasedness:**

$$
\mathbb{E}[\hat{I}_{\text{MC}}] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[h(X_i)] = I
$$

**Variance:**

$$
\text{Var}(\hat{I}_{\text{MC}}) = \frac{\sigma^2_h}{n}
$$

where $\sigma^2_h = \text{Var}_p(h(X)) = \mathbb{E}_p[h^2(X)] - I^2$.

### Central Limit Theorem

For sufficiently large $n$:

$$
\sqrt{n}(\hat{I}_{\text{MC}} - I) \xrightarrow{d} \mathcal{N}(0, \sigma^2_h)
$$

This yields the standard error $\text{SE}(\hat{I}_{\text{MC}}) = \sigma_h / \sqrt{n}$.

**Key insight**: The convergence rate is $O(n^{-1/2})$, **independent of dimension**.

### The Monte Carlo Advantage

| Method | Convergence Rate | 10D Problem | 100D Problem |
|--------|------------------|-------------|--------------|
| Grid (Simpson) | $O(n^{-4/d})$ | $O(n^{-0.4})$ | $O(n^{-0.04})$ |
| Monte Carlo | $O(n^{-1/2})$ | $O(n^{-0.5})$ | $O(n^{-0.5})$ |

Monte Carlo maintains its $O(n^{-1/2})$ convergence regardless of dimensionality.

### Estimating the Variance

Since $\sigma^2_h$ is typically unknown, we estimate it:

$$
\hat{\sigma}^2_h = \frac{1}{n-1} \sum_{i=1}^n (h(X_i) - \hat{I}_{\text{MC}})^2
$$

An approximate $(1-\alpha)$ confidence interval for $I$:

$$
\hat{I}_{\text{MC}} \pm z_{1-\alpha/2} \cdot \frac{\hat{\sigma}_h}{\sqrt{n}}
$$

### Limitations of Naive Monte Carlo

The basic Monte Carlo method requires sampling from $p(x)$. This becomes problematic when:

1. **Posterior distributions**: $p(\theta|y)$ has intractable normalising constant
2. **Complex distributions**: No standard sampling algorithm exists
3. **Rare events**: Events of interest have very low probability under $p$

These limitations motivate importance sampling and MCMC.

### PyTorch Implementation

```python
import torch

def monte_carlo_estimate(h_function, sampler, n_samples, return_diagnostics=False):
    """
    Basic Monte Carlo integration.
    
    Parameters
    ----------
    h_function : callable
        Function h(x) to integrate
    sampler : callable
        Function that returns n samples from distribution p
    n_samples : int
        Number of samples
        
    Returns
    -------
    estimate : torch.Tensor
        Monte Carlo estimate of E_p[h(X)]
    se : torch.Tensor
        Estimated standard error
    """
    samples = sampler(n_samples)
    h_values = h_function(samples)
    
    estimate = torch.mean(h_values)
    variance = torch.var(h_values, unbiased=True)
    se = torch.sqrt(variance / n_samples)
    
    if return_diagnostics:
        return estimate, se, {'samples': samples, 'h_values': h_values, 'variance': variance}
    return estimate, se


# Example: E[X²] where X ~ N(0, 1), true value = 1
import torch.distributions as dist

normal_dist = dist.Normal(0, 1)
h = lambda x: x**2
sampler = lambda n: normal_dist.sample((n,))

estimate, se = monte_carlo_estimate(h, sampler, n_samples=10000)
print(f"MC estimate: {estimate.item():.6f}")
print(f"Standard error: {se.item():.6f}")
print(f"95% CI: [{estimate.item() - 1.96*se.item():.6f}, {estimate.item() + 1.96*se.item():.6f}]")
```

!!! success "Monte Carlo Strengths"
    - **Dimension-independent convergence**: $O(n^{-1/2})$ regardless of $d$
    - **Unbiased estimation**: $\mathbb{E}[\hat{I}] = I$
    - **Easy uncertainty quantification**: CLT provides confidence intervals
    - **Trivially parallelizable**: Samples are i.i.d.

!!! warning "Monte Carlo Limitations"
    - **Requires sampling from target**: Not always possible
    - **Can have high variance**: Especially for rare events or heavy tails
    - **$O(n^{-1/2})$ convergence is slow**: Need 100× samples for 10× precision
