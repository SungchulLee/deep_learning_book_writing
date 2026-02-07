# Probabilistic PCA

A generative latent variable formulation of PCA.

---

## Overview

Classical PCA is an algebraic technique — it finds eigenvectors of a covariance matrix. **Probabilistic PCA (PPCA)**, introduced by Tipping and Bishop (1999), recasts PCA as a latent variable model with an explicit generative process. This reinterpretation provides several advantages: a proper likelihood function for model comparison, principled handling of missing data, a natural connection to variational autoencoders, and the ability to perform Bayesian inference over model parameters.

---

## The Generative Model

### Latent Variable Formulation

PPCA posits that each observed data point $\mathbf{x} \in \mathbb{R}^d$ is generated from a low-dimensional latent variable $\mathbf{z} \in \mathbb{R}^k$ through a linear mapping plus isotropic Gaussian noise:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_k)$$

$$\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(\mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \sigma^2 \mathbf{I}_d)$$

where:

- $\mathbf{z} \in \mathbb{R}^k$: latent variable (the "scores" in PCA terminology)
- $\mathbf{W} \in \mathbb{R}^{d \times k}$: loading matrix mapping latent space to observation space
- $\boldsymbol{\mu} \in \mathbb{R}^d$: data mean
- $\sigma^2 > 0$: isotropic noise variance (identical for all observed dimensions)

The generative process is: sample a latent code $\mathbf{z}$ from a standard normal, linearly transform it via $\mathbf{W}$, shift by the data mean $\boldsymbol{\mu}$, and add spherical Gaussian noise.

### Graphical Model

```
z ~ N(0, I)          [latent, k-dimensional]
    |
    | W (linear mapping)
    v
x | z ~ N(Wz + μ, σ²I)   [observed, d-dimensional]
```

---

## Marginal and Posterior Distributions

### Marginal Likelihood

Integrating out the latent variable gives the marginal distribution over observed data:

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z} = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \mathbf{C})$$

where the marginal covariance is:

$$\mathbf{C} = \mathbf{W}\mathbf{W}^T + \sigma^2 \mathbf{I}_d$$

This follows from the standard result that a linear transformation of a Gaussian plus Gaussian noise is Gaussian, with mean $\mathbf{W} \cdot \mathbf{0} + \boldsymbol{\mu} = \boldsymbol{\mu}$ and covariance $\mathbf{W}\mathbf{I}_k\mathbf{W}^T + \sigma^2\mathbf{I}_d = \mathbf{W}\mathbf{W}^T + \sigma^2\mathbf{I}_d$.

### Posterior Over Latents

Given an observation $\mathbf{x}$, the posterior over $\mathbf{z}$ is also Gaussian:

$$p(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \mathbf{m}, \boldsymbol{\Phi})$$

where:

$$\boldsymbol{\Phi} = (\mathbf{I}_k + \sigma^{-2}\mathbf{W}^T\mathbf{W})^{-1}$$

$$\mathbf{m} = \boldsymbol{\Phi} \, \mathbf{W}^T \sigma^{-2}(\mathbf{x} - \boldsymbol{\mu})$$

The posterior mean $\mathbf{m}$ is the probabilistic analog of the PCA score. Note that it involves a "shrinkage" toward zero through the matrix $\boldsymbol{\Phi}$, unlike classical PCA where the score is simply $\mathbf{W}^T(\mathbf{x} - \boldsymbol{\mu})$.

---

## Maximum Likelihood Estimation

### Log-Likelihood

For a dataset $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$, the log-likelihood is:

$$\ell(\mathbf{W}, \sigma^2) = -\frac{n}{2}\left[d \ln(2\pi) + \ln|\mathbf{C}| + \operatorname{tr}\left(\mathbf{C}^{-1}\mathbf{S}\right)\right]$$

where $\mathbf{S} = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - \boldsymbol{\mu})(\mathbf{x}^{(i)} - \boldsymbol{\mu})^T$ is the sample covariance matrix.

### Closed-Form ML Solution

Tipping and Bishop showed that the ML estimates have a closed-form solution in terms of the eigendecomposition of $\mathbf{S}$. Let $\lambda_1 \geq \cdots \geq \lambda_d$ be the eigenvalues of $\mathbf{S}$ with corresponding eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_d$.

**Noise variance:**

$$\hat{\sigma}^2 = \frac{1}{d - k}\sum_{j=k+1}^d \lambda_j$$

The ML noise variance is the average of the discarded eigenvalues — the "leftover" variance not captured by the $k$ principal components.

**Loading matrix:**

$$\hat{\mathbf{W}} = \mathbf{V}_k (\boldsymbol{\Lambda}_k - \hat{\sigma}^2 \mathbf{I}_k)^{1/2} \mathbf{R}$$

where:

- $\mathbf{V}_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$: top-$k$ eigenvectors of $\mathbf{S}$
- $\boldsymbol{\Lambda}_k = \operatorname{diag}(\lambda_1, \ldots, \lambda_k)$: corresponding eigenvalues
- $\mathbf{R}$: arbitrary $k \times k$ rotation matrix (the likelihood is invariant to rotations in latent space)

### Connection to Classical PCA

In the limit $\sigma^2 \to 0$, the PPCA model degenerates:

$$\lim_{\sigma^2 \to 0} \hat{\mathbf{W}} = \mathbf{V}_k \boldsymbol{\Lambda}_k^{1/2} \mathbf{R}$$

The posterior mean converges to the classical PCA projection:

$$\lim_{\sigma^2 \to 0} \mathbf{m} = \mathbf{W}^T(\mathbf{W}\mathbf{W}^T)^{-1}\mathbf{W} \cdot \mathbf{W}^T(\mathbf{x} - \boldsymbol{\mu})$$

In this limit, the noise model vanishes and PPCA reduces exactly to classical PCA (up to rotation ambiguity in the latent space).

---

## EM Algorithm for PPCA

When the closed-form solution is impractical (e.g., for very high-dimensional data where even forming $\mathbf{S}$ is expensive), an EM algorithm provides an iterative alternative.

### E-Step

Compute the posterior statistics of $\mathbf{z}$ given current parameters:

$$\boldsymbol{\Phi} = (\mathbf{I}_k + \sigma^{-2}\mathbf{W}^T\mathbf{W})^{-1}$$

$$\mathbb{E}[\mathbf{z}_i] = \boldsymbol{\Phi}\mathbf{W}^T\sigma^{-2}(\mathbf{x}^{(i)} - \boldsymbol{\mu})$$

$$\mathbb{E}[\mathbf{z}_i\mathbf{z}_i^T] = \boldsymbol{\Phi} + \mathbb{E}[\mathbf{z}_i]\mathbb{E}[\mathbf{z}_i]^T$$

### M-Step

Update $\mathbf{W}$ and $\sigma^2$ to maximize the expected complete log-likelihood:

$$\mathbf{W}_{\text{new}} = \left(\sum_i (\mathbf{x}^{(i)} - \boldsymbol{\mu})\mathbb{E}[\mathbf{z}_i]^T\right)\left(\sum_i \mathbb{E}[\mathbf{z}_i\mathbf{z}_i^T]\right)^{-1}$$

$$\sigma^2_{\text{new}} = \frac{1}{nd}\sum_i \left[\|\mathbf{x}^{(i)} - \boldsymbol{\mu}\|^2 - 2\mathbb{E}[\mathbf{z}_i]^T\mathbf{W}_{\text{new}}^T(\mathbf{x}^{(i)} - \boldsymbol{\mu}) + \operatorname{tr}\left(\mathbb{E}[\mathbf{z}_i\mathbf{z}_i^T]\mathbf{W}_{\text{new}}^T\mathbf{W}_{\text{new}}\right)\right]$$

### Advantages of EM

The E-step only requires $k \times k$ matrix inversions (not $d \times d$), making it efficient when $k \ll d$. Each iteration costs $O(ndk)$, compared to $O(nd^2 + d^3)$ for the full eigendecomposition.

---

## Implementation

```python
import numpy as np

class ProbabilisticPCA:
    """Probabilistic PCA with closed-form ML estimation.

    Implements the Tipping & Bishop (1999) model:
        z ~ N(0, I_k)
        x | z ~ N(Wz + mu, sigma^2 I_d)
    """

    def __init__(self, n_components):
        self.k = n_components
        self.W = None
        self.mu = None
        self.sigma2 = None

    def fit(self, X):
        """Fit PPCA using closed-form ML solution."""
        n, d = X.shape
        self.mu = X.mean(axis=0)
        X_centered = X - self.mu

        # Sample covariance
        S = X_centered.T @ X_centered / n

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # ML noise variance: average of discarded eigenvalues
        self.sigma2 = eigenvalues[self.k:].mean()

        # ML loading matrix
        Lambda_k = np.diag(eigenvalues[:self.k])
        V_k = eigenvectors[:, :self.k]
        self.W = V_k @ np.sqrt(Lambda_k - self.sigma2 * np.eye(self.k))

        return self

    def transform(self, X):
        """Compute posterior mean of latent variables (probabilistic scores)."""
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.k)
        M_inv = np.linalg.inv(M)
        return (X - self.mu) @ self.W @ M_inv.T

    def inverse_transform(self, Z):
        """Reconstruct observations from latent variables."""
        return Z @ self.W.T + self.mu

    def log_likelihood(self, X):
        """Compute per-sample average log-likelihood."""
        n, d = X.shape
        C = self.W @ self.W.T + self.sigma2 * np.eye(d)
        X_centered = X - self.mu
        S = X_centered.T @ X_centered / n

        sign, logdet = np.linalg.slogdet(C)
        C_inv = np.linalg.inv(C)

        ll = -0.5 * (d * np.log(2 * np.pi) + logdet + np.trace(C_inv @ S))
        return ll

    def sample(self, n_samples):
        """Generate samples from the fitted model."""
        d = self.W.shape[0]
        Z = np.random.randn(n_samples, self.k)
        noise = np.sqrt(self.sigma2) * np.random.randn(n_samples, d)
        return Z @ self.W.T + self.mu + noise
```

### EM Implementation

```python
class ProbabilisticPCA_EM:
    """PPCA fitted via EM — efficient for very high-dimensional data."""

    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape
        self.mu = X.mean(axis=0)
        X_c = X - self.mu

        # Initialize W randomly, sigma2 from data variance
        self.W = np.random.randn(d, self.k) * 0.1
        self.sigma2 = np.var(X_c)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            M = self.W.T @ self.W + self.sigma2 * np.eye(self.k)
            M_inv = np.linalg.inv(M)
            EZ = X_c @ self.W @ M_inv.T                  # [n, k]
            EZZ = n * self.sigma2 * M_inv + EZ.T @ EZ     # [k, k]

            # M-step
            self.W = (X_c.T @ EZ) @ np.linalg.inv(EZZ)   # [d, k]
            self.sigma2 = (
                np.sum(X_c ** 2)
                - 2 * np.trace(EZ.T @ X_c @ self.W)
                + np.trace(EZZ @ self.W.T @ self.W)
            ) / (n * d)

            # Check convergence
            ll = self.log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def transform(self, X):
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.k)
        return (X - self.mu) @ self.W @ np.linalg.inv(M).T

    def log_likelihood(self, X):
        n, d = X.shape
        C = self.W @ self.W.T + self.sigma2 * np.eye(d)
        X_c = X - self.mu
        S = X_c.T @ X_c / n
        _, logdet = np.linalg.slogdet(C)
        return -0.5 * (d * np.log(2 * np.pi) + logdet
                       + np.trace(np.linalg.solve(C, S)))
```

---

## Model Selection via Likelihood

A key advantage of PPCA is that the marginal likelihood provides a principled criterion for selecting the number of components $k$. Unlike the heuristic scree plot or variance-threshold methods of classical PCA, we can compare models of different $k$ by their log-likelihood on held-out data:

```python
import matplotlib.pyplot as plt

def select_components(X_train, X_val, max_k=20):
    """Select k by validation log-likelihood."""
    log_likelihoods = []
    for k in range(1, max_k + 1):
        ppca = ProbabilisticPCA(n_components=k).fit(X_train)
        ll = ppca.log_likelihood(X_val)
        log_likelihoods.append(ll)

    best_k = np.argmax(log_likelihoods) + 1

    plt.plot(range(1, max_k + 1), log_likelihoods, 'o-')
    plt.xlabel('Number of Components')
    plt.ylabel('Validation Log-Likelihood')
    plt.axvline(best_k, color='r', linestyle='--',
                label=f'Best k = {best_k}')
    plt.legend()
    plt.show()

    return best_k
```

---

## Handling Missing Data

PPCA naturally handles missing observations through the EM framework. In the E-step, the posterior over $\mathbf{z}_i$ conditions only on the **observed** dimensions of $\mathbf{x}^{(i)}$. In the M-step, sufficient statistics are accumulated from these partial posteriors. This makes PPCA a principled approach to PCA with incomplete data, unlike ad-hoc imputation methods.

```python
def e_step_missing(x_obs, obs_idx, W, mu, sigma2):
    """E-step for a single sample with missing values.

    Args:
        x_obs: Observed values
        obs_idx: Indices of observed dimensions
        W, mu, sigma2: Current model parameters
    """
    W_obs = W[obs_idx]
    mu_obs = mu[obs_idx]

    M = W_obs.T @ W_obs + sigma2 * np.eye(W.shape[1])
    M_inv = np.linalg.inv(M)

    Ez = M_inv @ W_obs.T @ (x_obs - mu_obs) / sigma2
    Ezz = sigma2 * M_inv + np.outer(Ez, Ez)

    return Ez, Ezz
```

---

## Connection to VAEs

Probabilistic PCA is the simplest instance of the latent variable model family that includes variational autoencoders (VAEs):

| Aspect | PPCA | VAE |
|--------|------|-----|
| **Prior** | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| **Decoder** | Linear: $\mathbf{W}\mathbf{z} + \boldsymbol{\mu}$ | Nonlinear neural network |
| **Noise model** | Isotropic: $\sigma^2\mathbf{I}$ | Learned (possibly heteroscedastic) |
| **Inference** | Exact posterior (Gaussian) | Approximate (amortized variational) |
| **Optimization** | Closed-form ML or EM | Stochastic gradient descent on ELBO |

PPCA can be viewed as a VAE with linear encoder and decoder, where the exact posterior replaces the variational approximation. Understanding PPCA thoroughly provides a strong foundation for the nonlinear extensions that VAEs introduce.

---

## Quantitative Finance Application

In quantitative finance, PPCA is valuable for factor modeling with incomplete data — a common scenario when assets have different listing dates or trading halts create gaps:

```python
# Factor model for equity returns
# X: [n_days, n_stocks] return matrix with missing values
ppca = ProbabilisticPCA(n_components=5)
ppca.fit(X_complete_subset)

# The loading matrix W represents factor exposures
# Columns of W are statistical risk factors
factor_loadings = ppca.W  # [n_stocks, n_factors]

# Noise variance sigma^2 estimates idiosyncratic risk
idio_var = ppca.sigma2

# Factor covariance: W W^T captures systematic risk
systematic_cov = ppca.W @ ppca.W.T
total_cov = systematic_cov + ppca.sigma2 * np.eye(d)
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Model** | $\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}$, $\;\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\;\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ |
| **Marginal** | $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{W}\mathbf{W}^T + \sigma^2\mathbf{I})$ |
| **ML loading matrix** | $\hat{\mathbf{W}} = \mathbf{V}_k(\boldsymbol{\Lambda}_k - \hat{\sigma}^2\mathbf{I})^{1/2}\mathbf{R}$ |
| **ML noise** | $\hat{\sigma}^2 = \frac{1}{d-k}\sum_{j>k}\lambda_j$ |
| **Limit $\sigma^2 \to 0$** | Recovers classical PCA |
| **Advantages** | Likelihood-based model selection, missing data, generative sampling, VAE connection |
| **Estimation** | Closed-form or EM (efficient when $k \ll d$) |
