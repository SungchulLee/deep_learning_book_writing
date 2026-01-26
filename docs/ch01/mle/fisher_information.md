# Fisher Information

## Introduction

Fisher Information is a fundamental concept that quantifies how much information a random variable carries about an unknown parameter. It plays a central role in understanding the precision of maximum likelihood estimators and is deeply connected to the curvature of the likelihood function.

!!! note "Why Fisher Information Matters"
    Fisher Information determines:
    
    - The **precision** of MLE estimates
    - The **Cramér-Rao lower bound** on estimator variance
    - The **asymptotic variance** of MLEs
    - The geometry of statistical manifolds (information geometry)

## Definition

### Score Function

Recall that the **score function** is the gradient of the log-likelihood:

$$
s(\theta) = \frac{\partial}{\partial \theta} \log p(X | \theta) = \frac{\partial \ell(\theta)}{\partial \theta}
$$

Under regularity conditions, the score has zero mean:

$$
\mathbb{E}_\theta[s(\theta)] = 0
$$

### Fisher Information (Single Parameter)

The **Fisher Information** is defined as the variance of the score function:

$$
I(\theta) = \text{Var}_\theta[s(\theta)] = \mathbb{E}_\theta\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]
$$

Under regularity conditions (interchanging differentiation and integration), an equivalent formula is:

$$
I(\theta) = -\mathbb{E}_\theta\left[\frac{\partial^2 \log p(X|\theta)}{\partial \theta^2}\right]
$$

This second form shows that **Fisher Information equals the expected curvature of the log-likelihood**.

### Intuitive Interpretation

- **High Fisher Information**: The log-likelihood is sharply peaked around $\theta$, meaning small changes in $\theta$ cause large changes in likelihood. The data is very informative about $\theta$.

- **Low Fisher Information**: The log-likelihood is flat, meaning the data doesn't distinguish well between different parameter values.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_fisher_information():
    """Visualize how Fisher Information relates to likelihood curvature."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    theta_range = np.linspace(-3, 3, 200)
    
    # High Fisher Information: Sharp likelihood
    ax = axes[0]
    high_fi = np.exp(-5 * theta_range**2)
    ax.plot(theta_range, high_fi, 'b-', linewidth=2)
    ax.fill_between(theta_range, high_fi, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', label='True θ')
    ax.set_xlabel('θ')
    ax.set_ylabel('Likelihood')
    ax.set_title('High Fisher Information\n(Sharp peak, precise estimate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Low Fisher Information: Flat likelihood
    ax = axes[1]
    low_fi = np.exp(-0.3 * theta_range**2)
    ax.plot(theta_range, low_fi, 'b-', linewidth=2)
    ax.fill_between(theta_range, low_fi, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', label='True θ')
    ax.set_xlabel('θ')
    ax.set_ylabel('Likelihood')
    ax.set_title('Low Fisher Information\n(Flat peak, imprecise estimate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## Fisher Information for Common Distributions

### Bernoulli Distribution

For $X \sim \text{Bernoulli}(p)$:

$$
\log p(x|p) = x \log p + (1-x) \log(1-p)
$$

$$
\frac{\partial \log p}{\partial p} = \frac{x}{p} - \frac{1-x}{1-p}
$$

$$
\frac{\partial^2 \log p}{\partial p^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
$$

Taking expectation ($\mathbb{E}[X] = p$):

$$
I(p) = -\mathbb{E}\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{1}{p^2} \cdot p + \frac{1}{(1-p)^2} \cdot (1-p)
$$

$$
\boxed{I(p) = \frac{1}{p(1-p)}}
$$

!!! info "Interpretation"
    Fisher Information is highest when $p = 0.5$ (most uncertainty) and lowest when $p$ is near 0 or 1 (outcomes are predictable).

### Normal Distribution

For $X \sim \mathcal{N}(\mu, \sigma^2)$ with known variance:

$$
\log p(x|\mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

$$
\frac{\partial^2 \log p}{\partial \mu^2} = -\frac{1}{\sigma^2}
$$

$$
\boxed{I(\mu) = \frac{1}{\sigma^2}}
$$

For unknown variance with known mean:

$$
\boxed{I(\sigma^2) = \frac{1}{2\sigma^4}}
$$

### Poisson Distribution

For $X \sim \text{Poisson}(\lambda)$:

$$
\log p(x|\lambda) = x \log \lambda - \lambda - \log(x!)
$$

$$
\frac{\partial^2 \log p}{\partial \lambda^2} = -\frac{x}{\lambda^2}
$$

$$
\boxed{I(\lambda) = \frac{1}{\lambda}}
$$

### Exponential Distribution

For $X \sim \text{Exponential}(\lambda)$:

$$
\boxed{I(\lambda) = \frac{1}{\lambda^2}}
$$

## Fisher Information Matrix (Multiple Parameters)

For a parameter vector $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)^T$, the **Fisher Information Matrix** is:

$$
\mathbf{I}(\boldsymbol{\theta})_{ij} = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta_i} \cdot \frac{\partial \log p}{\partial \theta_j}\right]
$$

Or equivalently:

$$
\mathbf{I}(\boldsymbol{\theta})_{ij} = -\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\right]
$$

### Example: Normal Distribution (Both Parameters)

For $X \sim \mathcal{N}(\mu, \sigma^2)$ with $\boldsymbol{\theta} = (\mu, \sigma^2)$:

$$
\mathbf{I}(\mu, \sigma^2) = \begin{pmatrix}
\frac{1}{\sigma^2} & 0 \\
0 & \frac{1}{2\sigma^4}
\end{pmatrix}
$$

The diagonal structure indicates that $\mu$ and $\sigma^2$ are **orthogonal** (information about one doesn't inform the other).

```python
def compute_fisher_information_matrix(data: torch.Tensor,
                                      log_likelihood_fn: callable,
                                      params: torch.Tensor,
                                      eps: float = 1e-4) -> torch.Tensor:
    """
    Numerically compute Fisher Information Matrix using second derivatives.
    
    Uses the formula: I_ij = -E[∂²ℓ/∂θ_i∂θ_j]
    Approximated by negative Hessian at MLE (observed Fisher Information)
    """
    n_params = len(params)
    hessian = torch.zeros(n_params, n_params)
    
    for i in range(n_params):
        for j in range(n_params):
            # Compute second derivative using finite differences
            params_pp = params.clone()
            params_pp[i] += eps
            params_pp[j] += eps
            
            params_pm = params.clone()
            params_pm[i] += eps
            params_pm[j] -= eps
            
            params_mp = params.clone()
            params_mp[i] -= eps
            params_mp[j] += eps
            
            params_mm = params.clone()
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            ll_pp = log_likelihood_fn(data, params_pp)
            ll_pm = log_likelihood_fn(data, params_pm)
            ll_mp = log_likelihood_fn(data, params_mp)
            ll_mm = log_likelihood_fn(data, params_mm)
            
            hessian[i, j] = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * eps**2)
    
    return -hessian  # Fisher Information is negative Hessian


def fisher_information_autodiff(log_prob_fn: callable,
                                params: torch.Tensor,
                                n_samples: int = 10000) -> torch.Tensor:
    """
    Compute Fisher Information using score variance (definition).
    
    I(θ) = E[(∂log p / ∂θ)²] = Var(score)
    """
    # Sample from the distribution
    # This requires knowing how to sample from p(x|θ)
    # Here we use automatic differentiation to compute scores
    
    params = params.clone().requires_grad_(True)
    
    scores = []
    for _ in range(n_samples):
        # Sample x ~ p(x|θ)
        # Compute score for this sample
        pass  # Implementation depends on distribution
    
    # Fisher info = variance of scores
    return torch.var(torch.stack(scores), dim=0)
```

## Properties of Fisher Information

### Additivity for Independent Observations

For $n$ i.i.d. observations, the total Fisher Information is:

$$
I_n(\theta) = n \cdot I(\theta)
$$

This is why more data leads to more precise estimates.

### Reparameterization

If $\eta = g(\theta)$ is a one-to-one transformation, then:

$$
I_\eta(\eta) = I_\theta(\theta) \cdot \left(\frac{d\theta}{d\eta}\right)^2
$$

For multiple parameters:

$$
\mathbf{I}_\eta = \mathbf{J}^T \mathbf{I}_\theta \mathbf{J}
$$

where $\mathbf{J}$ is the Jacobian matrix $\frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\eta}}$.

## Cramér-Rao Lower Bound

The Fisher Information sets a fundamental limit on how precise any unbiased estimator can be.

### Theorem (Cramér-Rao Bound)

For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

The bound is achieved by **efficient estimators**, and MLE is asymptotically efficient.

### Implications

1. **Lower bound on variance**: No unbiased estimator can have variance smaller than $1/I(\theta)$
2. **More information = less variance**: High Fisher Information means potentially more precise estimates
3. **Sample size effect**: With $n$ samples, the bound becomes $1/(nI(\theta))$

```python
def cramer_rao_bound_demo():
    """Demonstrate Cramér-Rao bound for Bernoulli parameter."""
    import matplotlib.pyplot as plt
    
    # True parameter values to test
    p_values = np.linspace(0.05, 0.95, 50)
    sample_sizes = [10, 50, 100, 500]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in sample_sizes:
        # Cramér-Rao bound: Var >= 1/(n*I(p)) = p(1-p)/n
        cr_bound = p_values * (1 - p_values) / n
        ax.plot(p_values, cr_bound, label=f'n = {n}')
    
    ax.set_xlabel('True probability p')
    ax.set_ylabel('Cramér-Rao Lower Bound on Variance')
    ax.set_title('Cramér-Rao Bound for Bernoulli MLE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.03])
    
    plt.show()

def verify_cramer_rao(true_p: float = 0.3, n: int = 100, n_simulations: int = 10000):
    """Verify that MLE variance matches Cramér-Rao bound."""
    torch.manual_seed(42)
    
    # Run many simulations
    estimates = []
    for _ in range(n_simulations):
        data = (torch.rand(n) < true_p).float()
        p_hat = data.mean().item()
        estimates.append(p_hat)
    
    empirical_var = np.var(estimates)
    cramer_rao = true_p * (1 - true_p) / n
    
    print(f"True p: {true_p}")
    print(f"Sample size: {n}")
    print(f"Cramér-Rao bound: {cramer_rao:.6f}")
    print(f"Empirical variance: {empirical_var:.6f}")
    print(f"Ratio (should be ≈ 1): {empirical_var / cramer_rao:.4f}")
```

## Observed vs. Expected Fisher Information

### Expected Fisher Information

What we've discussed so far:

$$
I(\theta) = -\mathbb{E}_\theta\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]
$$

This is evaluated at the true parameter value.

### Observed Fisher Information

In practice, we compute Fisher Information at the MLE:

$$
J(\hat{\theta}) = -\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\theta = \hat{\theta}}
$$

This is the **observed** (or **empirical**) Fisher Information.

!!! tip "When to Use Which"
    - **Expected**: Theoretical analysis, prior to collecting data
    - **Observed**: Practical inference, constructing confidence intervals

## Applications in Deep Learning

### Natural Gradient Descent

In standard gradient descent, we update: $\theta \leftarrow \theta - \alpha \nabla_\theta L$

In **natural gradient descent**, we account for the geometry of the parameter space:

$$
\theta \leftarrow \theta - \alpha \mathbf{I}(\theta)^{-1} \nabla_\theta L
$$

This leads to updates that are more "natural" in the sense of information geometry.

### Connection to Second-Order Optimization

The Fisher Information Matrix approximates the Hessian for neural networks trained with cross-entropy loss:

$$
\mathbf{I}(\theta) \approx \mathbf{H}(\theta)
$$

This justifies methods like K-FAC (Kronecker-Factored Approximate Curvature) for efficient second-order optimization.

```python
class NaturalGradientOptimizer:
    """
    Simple natural gradient optimizer for demonstration.
    
    In practice, computing I(θ)^{-1} is expensive for neural networks,
    so approximations like K-FAC are used.
    """
    
    def __init__(self, params: list, lr: float = 0.01, damping: float = 1e-4):
        self.params = list(params)
        self.lr = lr
        self.damping = damping
    
    def step(self, fisher_info: torch.Tensor, gradients: list):
        """
        Perform natural gradient update.
        
        θ ← θ - lr * I(θ)^{-1} * ∇L
        """
        # Add damping for numerical stability
        fisher_damped = fisher_info + self.damping * torch.eye(fisher_info.shape[0])
        
        # Compute natural gradient
        grad_flat = torch.cat([g.flatten() for g in gradients])
        natural_grad = torch.linalg.solve(fisher_damped, grad_flat)
        
        # Update parameters
        idx = 0
        for param in self.params:
            numel = param.numel()
            param.data -= self.lr * natural_grad[idx:idx+numel].view(param.shape)
            idx += numel
```

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Score Function | $s(\theta) = \nabla_\theta \log p(X\|\theta)$ | Gradient of log-likelihood |
| Fisher Information | $I(\theta) = \text{Var}[s(\theta)]$ | Curvature of log-likelihood |
| Cramér-Rao Bound | $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$ | Lower bound on variance |
| Information Additivity | $I_n = nI$ | More data = more information |

## Exercises

1. **Derive** the Fisher Information for the Beta distribution $\text{Beta}(\alpha, \beta)$

2. **Prove** that the MLE for Bernoulli $p$ achieves the Cramér-Rao bound exactly (is efficient)

3. **Implement** a function that computes Fisher Information using automatic differentiation in PyTorch

4. **Show** that for the exponential family, Fisher Information equals the variance of the sufficient statistic

5. **Compare** convergence of standard gradient descent vs. natural gradient descent for a simple logistic regression problem

## References

- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd Edition
- Amari, S. (2016). *Information Geometry and Its Applications*
- Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method", JMLR
