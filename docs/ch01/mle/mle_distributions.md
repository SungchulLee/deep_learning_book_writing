# MLE for Common Distributions

## Introduction

This section derives Maximum Likelihood Estimators for the most commonly encountered probability distributions. Understanding these derivations builds intuition for how MLE works and reveals patterns that extend to more complex models.

## Discrete Distributions

### Bernoulli Distribution

**Model**: $X \sim \text{Bernoulli}(p)$

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

**Log-Likelihood** (for $n$ observations with $k$ successes):

$$
\ell(p) = k \log p + (n-k) \log(1-p)
$$

**MLE**:

$$
\hat{p} = \frac{k}{n} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}
$$

**Interpretation**: The MLE is the sample proportion—the fraction of successes observed.

```python
import torch

def bernoulli_mle(data: torch.Tensor) -> float:
    """MLE for Bernoulli parameter p."""
    return data.mean().item()

def bernoulli_log_likelihood(data: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Log-likelihood for Bernoulli distribution."""
    eps = 1e-8
    p = torch.clamp(p, eps, 1 - eps)
    return torch.sum(data * torch.log(p) + (1 - data) * torch.log(1 - p))
```

### Binomial Distribution

**Model**: $X \sim \text{Binomial}(n, p)$ (single observation of $n$ trials)

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

**Log-Likelihood** (for $m$ observations $x_1, \ldots, x_m$):

$$
\ell(p) = \sum_{i=1}^{m} \left[ x_i \log p + (n - x_i) \log(1-p) \right] + \text{const}
$$

**MLE**:

$$
\hat{p} = \frac{\sum_{i=1}^{m} x_i}{mn} = \frac{\bar{x}}{n}
$$

### Categorical Distribution

**Model**: $X \sim \text{Categorical}(p_1, \ldots, p_K)$ where $\sum_{k=1}^{K} p_k = 1$

$$
P(X = k) = p_k, \quad k \in \{1, 2, \ldots, K\}
$$

**Log-Likelihood** (for counts $n_1, \ldots, n_K$):

$$
\ell(\mathbf{p}) = \sum_{k=1}^{K} n_k \log p_k
$$

**MLE** (using Lagrange multipliers for constraint $\sum p_k = 1$):

$$
\hat{p}_k = \frac{n_k}{n} = \frac{n_k}{\sum_{j=1}^{K} n_j}
$$

**Interpretation**: Each probability is estimated by its relative frequency.

```python
def categorical_mle(data: torch.Tensor, num_categories: int) -> torch.Tensor:
    """
    MLE for categorical distribution.
    
    Args:
        data: Integer tensor of observations in {0, 1, ..., K-1}
        num_categories: K, the number of categories
    
    Returns:
        Tensor of estimated probabilities [p_0, p_1, ..., p_{K-1}]
    """
    counts = torch.bincount(data.long(), minlength=num_categories).float()
    return counts / counts.sum()
```

### Poisson Distribution

**Model**: $X \sim \text{Poisson}(\lambda)$

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \ldots\}
$$

**Log-Likelihood**:

$$
\ell(\lambda) = \sum_{i=1}^{n} \left[ x_i \log \lambda - \lambda - \log(x_i!) \right]
= \left(\sum_{i=1}^{n} x_i\right) \log \lambda - n\lambda + \text{const}
$$

**Derivation**:

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0
$$

**MLE**:

$$
\hat{\lambda} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}
$$

**Interpretation**: The MLE for the Poisson rate parameter is the sample mean.

```python
def poisson_mle(data: torch.Tensor) -> float:
    """MLE for Poisson rate parameter λ."""
    return data.float().mean().item()

def poisson_log_likelihood(data: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Log-likelihood for Poisson distribution (up to constant)."""
    return torch.sum(data * torch.log(lam) - lam)
```

### Geometric Distribution

**Model**: $X \sim \text{Geometric}(p)$ (number of trials until first success)

$$
P(X = k) = (1-p)^{k-1} p, \quad k \in \{1, 2, 3, \ldots\}
$$

**Log-Likelihood**:

$$
\ell(p) = \sum_{i=1}^{n} \left[ (x_i - 1) \log(1-p) + \log p \right]
= \left(\sum x_i - n\right) \log(1-p) + n \log p
$$

**MLE**:

$$
\hat{p} = \frac{n}{\sum_{i=1}^{n} x_i} = \frac{1}{\bar{x}}
$$

## Continuous Distributions

### Normal (Gaussian) Distribution

**Model**: $X \sim \mathcal{N}(\mu, \sigma^2)$

$$
p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Log-Likelihood**:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

**Derivation for $\mu$**:

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
$$

$$
\sum x_i = n\mu \implies \hat{\mu} = \bar{x}
$$

**Derivation for $\sigma^2$**:

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \mu)^2 = 0
$$

$$
\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
$$

**MLE**:

$$
\hat{\mu} = \bar{x}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

!!! warning "Biased Variance Estimator"
    The MLE for variance is **biased**: $\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2 < \sigma^2$. The unbiased estimator uses $n-1$ in the denominator, but MLE is consistent—bias vanishes as $n \to \infty$.

```python
def normal_mle(data: torch.Tensor) -> tuple:
    """
    MLE for Normal distribution parameters.
    
    Returns:
        (mu_hat, sigma_hat): Estimated mean and standard deviation
    """
    mu_hat = data.mean()
    # MLE uses biased estimator (divides by n, not n-1)
    sigma_hat = data.std(unbiased=False)
    return mu_hat.item(), sigma_hat.item()

def normal_log_likelihood(data: torch.Tensor, 
                         mu: torch.Tensor, 
                         sigma: torch.Tensor) -> torch.Tensor:
    """Log-likelihood for Normal distribution."""
    import math
    n = len(data)
    sigma = torch.clamp(sigma, min=1e-6)
    
    ll = -n/2 * math.log(2 * math.pi)
    ll -= n * torch.log(sigma)
    ll -= torch.sum((data - mu)**2) / (2 * sigma**2)
    return ll
```

### Exponential Distribution

**Model**: $X \sim \text{Exponential}(\lambda)$

$$
p(x | \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**Log-Likelihood**:

$$
\ell(\lambda) = n \log \lambda - \lambda \sum_{i=1}^{n} x_i
$$

**MLE**:

$$
\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum x_i = 0 \implies \hat{\lambda} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}
$$

**Interpretation**: The MLE for the rate is the reciprocal of the sample mean.

```python
def exponential_mle(data: torch.Tensor) -> float:
    """MLE for Exponential rate parameter λ."""
    return 1.0 / data.mean().item()
```

### Uniform Distribution

**Model**: $X \sim \text{Uniform}(a, b)$

$$
p(x | a, b) = \frac{1}{b-a}, \quad a \leq x \leq b
$$

**Likelihood**:

$$
L(a, b) = \begin{cases}
\frac{1}{(b-a)^n} & \text{if } a \leq x_{(1)} \text{ and } x_{(n)} \leq b \\
0 & \text{otherwise}
\end{cases}
$$

where $x_{(1)} = \min_i x_i$ and $x_{(n)} = \max_i x_i$.

**MLE**:

$$
\hat{a} = x_{(1)} = \min_i x_i, \quad \hat{b} = x_{(n)} = \max_i x_i
$$

!!! note "Non-Regular MLE"
    The uniform distribution is a "non-regular" case because the support depends on the parameters. The MLE exists but doesn't satisfy the usual regularity conditions (e.g., Fisher information is not well-defined in the standard way).

```python
def uniform_mle(data: torch.Tensor) -> tuple:
    """MLE for Uniform distribution parameters."""
    return data.min().item(), data.max().item()
```

### Gamma Distribution

**Model**: $X \sim \text{Gamma}(\alpha, \beta)$

$$
p(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

**Log-Likelihood**:

$$
\ell(\alpha, \beta) = n\alpha \log \beta - n \log \Gamma(\alpha) + (\alpha-1)\sum \log x_i - \beta \sum x_i
$$

**MLE**:

No closed-form solution exists. For $\beta$ (given $\alpha$):

$$
\hat{\beta} = \frac{n\alpha}{\sum x_i} = \frac{\alpha}{\bar{x}}
$$

For $\alpha$, we solve numerically:

$$
\log \alpha - \psi(\alpha) = \log \bar{x} - \overline{\log x}
$$

where $\psi(\alpha) = \frac{d}{d\alpha}\log\Gamma(\alpha)$ is the digamma function.

```python
def gamma_mle(data: torch.Tensor, n_iter: int = 100) -> tuple:
    """
    MLE for Gamma distribution using Newton-Raphson.
    
    Returns:
        (alpha_hat, beta_hat): Shape and rate parameters
    """
    from scipy.special import digamma, polygamma
    import numpy as np
    
    data_np = data.numpy()
    n = len(data_np)
    
    # Sufficient statistics
    mean_x = np.mean(data_np)
    mean_log_x = np.mean(np.log(data_np))
    s = np.log(mean_x) - mean_log_x
    
    # Initial estimate using method of moments
    alpha = (3 - s + np.sqrt((s - 3)**2 + 24*s)) / (12*s)
    
    # Newton-Raphson for alpha
    for _ in range(n_iter):
        alpha_new = alpha - (np.log(alpha) - digamma(alpha) - s) / (1/alpha - polygamma(1, alpha))
        if abs(alpha_new - alpha) < 1e-8:
            break
        alpha = alpha_new
    
    beta = alpha / mean_x
    
    return float(alpha), float(beta)
```

## Multivariate Distributions

### Multivariate Normal Distribution

**Model**: $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

$$
p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

**MLE**:

$$
\hat{\boldsymbol{\mu}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i = \bar{\mathbf{x}}
$$

$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T
$$

```python
def multivariate_normal_mle(data: torch.Tensor) -> tuple:
    """
    MLE for Multivariate Normal distribution.
    
    Args:
        data: Tensor of shape (n_samples, n_features)
    
    Returns:
        (mu_hat, sigma_hat): Mean vector and covariance matrix
    """
    n = data.shape[0]
    mu_hat = data.mean(dim=0)
    centered = data - mu_hat
    sigma_hat = (centered.T @ centered) / n
    return mu_hat, sigma_hat
```

## Summary Table

| Distribution | Parameters | MLE Estimator |
|-------------|------------|---------------|
| Bernoulli | $p$ | $\hat{p} = \bar{x}$ |
| Binomial | $p$ (given $n$) | $\hat{p} = \bar{x}/n$ |
| Categorical | $p_1, \ldots, p_K$ | $\hat{p}_k = n_k/n$ |
| Poisson | $\lambda$ | $\hat{\lambda} = \bar{x}$ |
| Geometric | $p$ | $\hat{p} = 1/\bar{x}$ |
| Normal | $\mu, \sigma^2$ | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ |
| Exponential | $\lambda$ | $\hat{\lambda} = 1/\bar{x}$ |
| Uniform | $a, b$ | $\hat{a} = \min(x_i)$, $\hat{b} = \max(x_i)$ |
| Gamma | $\alpha, \beta$ | Numerical solution required |

## Pattern Recognition

Notice the recurring patterns in MLE derivations:

1. **Sample means appear frequently**: For location parameters, the MLE is often $\bar{x}$
2. **Sufficient statistics**: The MLE depends on data only through sufficient statistics
3. **Constraint handling**: For probability vectors, Lagrange multipliers enforce $\sum p_k = 1$
4. **Reciprocal relationships**: Rate parameters often have MLEs of form $1/\bar{x}$

## PyTorch Implementation: Gradient-Based MLE

For distributions without closed-form MLEs, or for learning purposes, we can use gradient-based optimization:

```python
def gradient_mle(data: torch.Tensor, 
                 log_likelihood_fn: callable,
                 init_params: dict,
                 lr: float = 0.01,
                 n_iter: int = 1000) -> dict:
    """
    Generic gradient-based MLE using PyTorch.
    
    Args:
        data: Observed data
        log_likelihood_fn: Function computing log-likelihood
        init_params: Dictionary of initial parameter tensors
        lr: Learning rate
        n_iter: Number of iterations
    
    Returns:
        Dictionary of estimated parameters
    """
    # Make parameters require gradients
    params = {k: v.clone().requires_grad_(True) for k, v in init_params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    
    for i in range(n_iter):
        # Compute negative log-likelihood
        nll = -log_likelihood_fn(data, **params)
        
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
    
    return {k: v.detach() for k, v in params.items()}

# Example: Fit Normal distribution
data = torch.randn(1000) * 2 + 5  # True: mu=5, sigma=2

def normal_ll(data, mu, log_sigma):
    sigma = torch.exp(log_sigma)  # Ensure positivity
    return normal_log_likelihood(data, mu, sigma)

init = {'mu': torch.tensor(0.0), 'log_sigma': torch.tensor(0.0)}
result = gradient_mle(data, normal_ll, init)
print(f"mu_hat = {result['mu'].item():.4f}")
print(f"sigma_hat = {torch.exp(result['log_sigma']).item():.4f}")
```

## Exercises

1. **Derive** the MLE for the negative binomial distribution
2. **Prove** that the MLE for normal variance is biased and calculate the bias
3. **Implement** gradient-based MLE for the Beta distribution and compare to the method of moments estimator
4. **Show** that for the Laplace distribution $p(x|\mu, b) = \frac{1}{2b}e^{-|x-\mu|/b}$, the MLE for $\mu$ is the sample median

## References

- Casella, G. & Berger, R. L. (2002). *Statistical Inference*, 2nd Edition
- Lehmann, E. L. & Casella, G. (1998). *Theory of Point Estimation*, 2nd Edition
