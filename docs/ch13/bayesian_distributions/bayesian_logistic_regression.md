# Bayesian Logistic Regression

## Overview

Bayesian logistic regression extends the standard logistic regression model by placing prior distributions over the weight vector, yielding full posterior distributions over parameters and predictions. Unlike Bayesian linear regression, the posterior is **non-conjugate** and requires approximate inference methods — making it a natural bridge between the analytical results of Chapter 16 and the approximate inference techniques of Chapters 18-19.

---

## Model Specification

### Likelihood

For binary classification with observations $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $y_i \in \{0, 1\}$:

$$
p(y_i = 1 \mid \mathbf{x}_i, \mathbf{w}) = \sigma(\mathbf{w}^\top \mathbf{x}_i) = \frac{1}{1 + \exp(-\mathbf{w}^\top \mathbf{x}_i)}
$$

The full data likelihood:

$$
p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \prod_{i=1}^N \sigma(\mathbf{w}^\top \mathbf{x}_i)^{y_i} \left[1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)\right]^{1-y_i}
$$

### Prior

A Gaussian prior on the weights provides regularization:

$$
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I})
$$

where $\alpha$ controls prior precision (inverse variance). This corresponds to L2 regularization when performing MAP estimation.

### Posterior

The posterior has no closed-form solution:

$$
p(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \frac{p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})}{p(\mathbf{y} \mid \mathbf{X})} \propto p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) \, p(\mathbf{w})
$$

The product of a Gaussian prior and a Bernoulli likelihood (through the sigmoid) does not yield a distribution from any standard family.

---

## Laplace Approximation

The **Laplace approximation** fits a Gaussian to the posterior by matching the mode and curvature:

$$
p(\mathbf{w} \mid \mathcal{D}) \approx q(\mathbf{w}) = \mathcal{N}(\mathbf{w} \mid \mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})
$$

where $\mathbf{w}_{\text{MAP}}$ is the MAP estimate and $\mathbf{H}$ is the Hessian of the negative log-posterior at $\mathbf{w}_{\text{MAP}}$.

### Finding the MAP Estimate

The log-posterior is:

$$
\log p(\mathbf{w} \mid \mathcal{D}) = \sum_{i=1}^N \left[ y_i \log \sigma_i + (1-y_i) \log(1-\sigma_i) \right] - \frac{\alpha}{2} \|\mathbf{w}\|^2 + \text{const}
$$

where $\sigma_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$.

**Gradient:**

$$
\nabla_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\sigma}) - \alpha \mathbf{w}
$$

**Hessian:**

$$
\mathbf{H} = -\nabla^2_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D}) = \mathbf{X}^\top \mathbf{S} \mathbf{X} + \alpha \mathbf{I}
$$

where $\mathbf{S} = \text{diag}(\sigma_i(1-\sigma_i))$.

### Iteratively Reweighted Least Squares (IRLS)

The MAP estimate is found via Newton-Raphson iteration:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \mathbf{H}^{-1} \nabla_{\mathbf{w}} \log p(\mathbf{w}^{(t)} \mid \mathcal{D})
$$

which is equivalent to solving a sequence of weighted least-squares problems.

### Predictive Distribution

For a new input $\mathbf{x}_*$, the predictive distribution integrates over the approximate posterior:

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) = \int \sigma(\mathbf{w}^\top \mathbf{x}_*) \, q(\mathbf{w}) \, d\mathbf{w}
$$

This integral is approximated using the **probit approximation**:

$$
p(y_* = 1 \mid \mathbf{x}_*, \mathcal{D}) \approx \sigma\left(\frac{\mu_a}{\sqrt{1 + \pi \sigma_a^2 / 8}}\right)
$$

where $\mu_a = \mathbf{w}_{\text{MAP}}^\top \mathbf{x}_*$ and $\sigma_a^2 = \mathbf{x}_*^\top \mathbf{H}^{-1} \mathbf{x}_*$.

---

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F


class BayesianLogisticRegression:
    """
    Bayesian logistic regression with Laplace approximation.
    
    Prior: w ~ N(0, alpha^{-1} I)
    Likelihood: y | x, w ~ Bernoulli(sigmoid(w^T x))
    Posterior ≈ N(w_MAP, H^{-1})  via Laplace approximation
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w_map = None
        self.H_inv = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
            max_iter: int = 100, tol: float = 1e-6):
        """Find MAP estimate via IRLS."""
        N, D = X.shape
        w = torch.zeros(D, dtype=torch.float64)
        
        for iteration in range(max_iter):
            # Forward pass
            logits = X @ w
            sigma = torch.sigmoid(logits)
            
            # Gradient of log-posterior
            grad = X.T @ (y - sigma) - self.alpha * w
            
            # Hessian of negative log-posterior
            S = sigma * (1 - sigma)
            H = X.T @ (S.unsqueeze(1) * X) + self.alpha * torch.eye(D, dtype=torch.float64)
            
            # Newton step
            delta = torch.linalg.solve(H, grad)
            w = w + delta
            
            if torch.norm(delta) < tol:
                break
        
        self.w_map = w
        self.H_inv = torch.linalg.inv(H)
        return self
    
    def predict_proba(self, X_new: torch.Tensor) -> torch.Tensor:
        """
        Predictive probabilities with uncertainty integration.
        
        Uses probit approximation to integrate over posterior.
        """
        mu_a = X_new @ self.w_map
        sigma2_a = (X_new @ self.H_inv * X_new).sum(dim=1)
        
        # Probit approximation
        kappa = 1.0 / torch.sqrt(1.0 + torch.pi * sigma2_a / 8.0)
        return torch.sigmoid(kappa * mu_a)
    
    def predict_map(self, X_new: torch.Tensor) -> torch.Tensor:
        """Point predictions using MAP estimate (no uncertainty)."""
        return torch.sigmoid(X_new @ self.w_map)
```

---

## Comparison: MAP vs Full Bayesian Predictions

| Aspect | MAP Prediction | Bayesian Prediction |
|--------|---------------|---------------------|
| Formula | $\sigma(\mathbf{w}_{\text{MAP}}^\top \mathbf{x}_*)$ | $\int \sigma(\mathbf{w}^\top \mathbf{x}_*) q(\mathbf{w}) d\mathbf{w}$ |
| Uncertainty | None | Epistemic uncertainty via $\sigma_a^2$ |
| Far from data | Overconfident | Appropriately uncertain |
| Decision boundary | Sharp | Soft (wider transition region) |
| Calibration | Often poor | Generally better |

The Bayesian predictive is always **less confident** than the MAP prediction, especially for inputs far from the training data — a desirable property for risk-sensitive applications.

---

## Connection to Other Methods

| Method | Chapter | Relationship |
|--------|---------|-------------|
| MAP with Gaussian prior | This chapter | Equivalent to L2-regularized logistic regression |
| Variational inference | Ch19 | Alternative to Laplace, scales better |
| MCMC sampling | Ch18 | Exact posterior, more expensive |
| Bayesian neural networks | Ch19 (BNN) | Generalization to nonlinear models |

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 4.5.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 8.
- MacKay, D. J. C. (1992). The evidence framework applied to classification networks. *Neural Computation*, 4(5), 720-736.
