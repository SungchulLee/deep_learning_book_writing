# Gaussian Processes

## Overview

A **Gaussian Process (GP)** is a nonparametric Bayesian approach to regression and classification that defines a prior directly over functions rather than over a finite set of parameters. GPs provide exact posterior inference with closed-form predictive distributions, naturally quantifying uncertainty and adapting model complexity to the data.

---

## Intuition: Reasoning About Functions

Most machine learning methods estimate parameters from data—weights of a neural network, coefficients of a linear model. GPs take a fundamentally different approach: they reason directly about the **high-level properties** of functions that could fit the data. Rather than asking "what are the best weights?", a GP asks "what kinds of functions are consistent with what we've observed?"

Consider a time series of asset returns observed at irregular intervals. Before seeing any data, we might have prior beliefs: the function should be reasonably smooth, perhaps periodic (seasonal patterns), and we expect more uncertainty in regions where we have fewer observations. A GP lets us encode all of these assumptions directly through the choice of kernel function.

!!! note "The GP Workflow"
    1. **Specify a prior**: Choose a kernel that encodes assumptions about smoothness, periodicity, and scale
    2. **Condition on data**: Compute the posterior distribution over functions consistent with observations
    3. **Make predictions**: The posterior mean provides point estimates, while the posterior variance gives calibrated uncertainty

A key property of GPs is that **epistemic uncertainty** (uncertainty due to limited data) naturally grows in regions far from observed data points. This is particularly valuable in quantitative finance, where knowing what we *don't* know is as important as making accurate predictions.

---

## Definition

A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution. A GP is fully specified by its **mean function** $m(\mathbf{x})$ and **covariance (kernel) function** $k(\mathbf{x}, \mathbf{x}')$:

$$
f \sim \mathcal{GP}\bigl(m(\mathbf{x}), \, k(\mathbf{x}, \mathbf{x}')\bigr)
$$

For any finite set of inputs $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$:

$$
\mathbf{f} = \begin{pmatrix} f(\mathbf{x}_1) \\ \vdots \\ f(\mathbf{x}_N) \end{pmatrix} \sim \mathcal{N}\bigl(\boldsymbol{\mu}, \mathbf{K}\bigr)
$$

where $\mu_i = m(\mathbf{x}_i)$ and $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.

---

## Kernel Functions

The kernel encodes prior assumptions about the function: smoothness, periodicity, length scales, and amplitude.

### Common Kernels

**Squared Exponential (RBF):**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right)
$$

Produces infinitely differentiable (very smooth) functions. The **length scale** $\ell$ controls the correlation distance and **signal variance** $\sigma_f^2$ controls the amplitude.

**Matérn Kernel:**

$$
k_\nu(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\,r}{\ell}\right)
$$

where $r = \|\mathbf{x} - \mathbf{x}'\|$. Controls smoothness through $\nu$: $\nu = 1/2$ (Ornstein-Uhlenbeck), $\nu = 3/2$, $\nu = 5/2$ are common choices. As $\nu \to \infty$, recovers the RBF kernel.

**Periodic Kernel:**

$$
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi |\mathbf{x} - \mathbf{x}'|/p)}{\ell^2}\right)
$$

For functions with known periodicity $p$.

### Kernel Composition

Kernels can be combined to encode richer prior structure:

| Operation | Formula | Interpretation |
|-----------|---------|---------------|
| Sum | $k_1 + k_2$ | Superposition of independent patterns |
| Product | $k_1 \cdot k_2$ | Interaction of patterns |
| Scaling | $\sigma^2 k$ | Amplitude control |

---

## GP Regression

### Model

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with noise model $y_i = f(\mathbf{x}_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$:

$$
\mathbf{y} \sim \mathcal{N}\bigl(\mathbf{0}, \, \mathbf{K} + \sigma_n^2 \mathbf{I}\bigr)
$$

### Posterior Predictive

For test inputs $\mathbf{X}_*$, the predictive distribution is Gaussian:

$$
f_* \mid \mathbf{X}_*, \mathbf{X}, \mathbf{y} \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))
$$

$$
\bar{f}_* = \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
$$

$$
\text{cov}(f_*) = \mathbf{K}_{**} - \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{K}_*
$$

where $\mathbf{K}_* = k(\mathbf{X}, \mathbf{X}_*)$ and $\mathbf{K}_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$.

### Marginal Likelihood

The log marginal likelihood for hyperparameter optimization:

$$
\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2}\log|\mathbf{K}_y| - \frac{N}{2}\log(2\pi)
$$

where $\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I}$. This naturally balances data fit (first term) against model complexity (second term) — a Bayesian Occam's razor.

---

## PyTorch Implementation

```python
import torch


class GaussianProcessRegressor:
    """
    Gaussian Process regression with RBF kernel.
    
    Hyperparameters optimized via marginal likelihood maximization.
    """
    
    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0, 
                 noise_var: float = 0.1):
        self.log_length_scale = torch.tensor(
            [torch.log(torch.tensor(length_scale))], requires_grad=True)
        self.log_signal_var = torch.tensor(
            [torch.log(torch.tensor(signal_var))], requires_grad=True)
        self.log_noise_var = torch.tensor(
            [torch.log(torch.tensor(noise_var))], requires_grad=True)
    
    def rbf_kernel(self, X1, X2):
        """Compute RBF kernel matrix."""
        ell = torch.exp(self.log_length_scale)
        sf2 = torch.exp(self.log_signal_var)
        
        dist_sq = torch.cdist(X1 / ell, X2 / ell, p=2) ** 2
        return sf2 * torch.exp(-0.5 * dist_sq)
    
    def log_marginal_likelihood(self, X, y):
        """Compute log marginal likelihood."""
        sn2 = torch.exp(self.log_noise_var)
        K = self.rbf_kernel(X, X) + sn2 * torch.eye(len(X))
        
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze()
        
        lml = -0.5 * y @ alpha - L.diagonal().log().sum() - 0.5 * len(y) * torch.log(
            torch.tensor(2 * torch.pi))
        return lml
    
    def fit(self, X, y, n_iter=100, lr=0.1):
        """Optimize hyperparameters via marginal likelihood."""
        self.X_train = X
        self.y_train = y
        
        optimizer = torch.optim.Adam(
            [self.log_length_scale, self.log_signal_var, self.log_noise_var], lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = -self.log_marginal_likelihood(X, y)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X_new):
        """Posterior predictive mean and variance."""
        with torch.no_grad():
            sn2 = torch.exp(self.log_noise_var)
            K = self.rbf_kernel(self.X_train, self.X_train) + sn2 * torch.eye(
                len(self.X_train))
            K_star = self.rbf_kernel(self.X_train, X_new)
            K_ss = self.rbf_kernel(X_new, X_new)
            
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(
                self.y_train.unsqueeze(1), L).squeeze()
            
            mean = K_star.T @ alpha
            v = torch.linalg.solve_triangular(L, K_star, upper=False)
            var = K_ss.diag() - (v ** 2).sum(dim=0)
            
        return mean, var
```

---

## Computational Considerations

| Aspect | Standard GP | Approximation |
|--------|-------------|---------------|
| Training | $O(N^3)$ | Sparse GP: $O(NM^2)$ |
| Prediction | $O(N^2)$ per point | $O(M^2)$ per point |
| Memory | $O(N^2)$ | $O(NM)$ |
| Inducing points $M$ | — | $M \ll N$ |

For large-scale financial datasets, sparse GP approximations (FITC, VFE) or structured kernel interpolation (SKI) are necessary.

---

## Applications in Quantitative Finance

GPs are particularly well-suited for financial modeling due to their uncertainty quantification:

- **Volatility surface modeling**: Fitting implied volatility as a function of strike and maturity with uncertainty bands
- **Yield curve interpolation**: Smooth interpolation with principled extrapolation uncertainty
- **Alpha signal modeling**: Nonlinear factor models with epistemic uncertainty for position sizing
- **Bayesian optimization**: Hyperparameter tuning for trading strategies

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **GP definition** | Prior over functions specified by mean and kernel |
| **Kernel choice** | Encodes smoothness, periodicity, and other structural assumptions |
| **Posterior** | Exact Gaussian predictive distribution in closed form |
| **Marginal likelihood** | Natural model selection criterion (Bayesian Occam's razor) |
| **Scalability** | $O(N^3)$ limits direct application; sparse methods extend to larger datasets |

---

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 15.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 6.4.
