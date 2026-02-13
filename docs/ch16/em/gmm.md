# Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are the canonical application of the EM algorithm and serve as a foundational model in unsupervised learning. This section provides a complete treatment: model specification, EM derivation, implementation details, and extensions relevant to quantitative finance.

---

## Model Specification

### The Generative Story

A Gaussian Mixture Model assumes that data is generated from a mixture of $K$ Gaussian distributions. For each observation $\mathbf{x}_i$:

1. **Draw component assignment**: $z_i \sim \text{Categorical}(\boldsymbol{\pi})$ where $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$
2. **Draw observation from selected component**: $\mathbf{x}_i | z_i = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

The latent variable $z_i \in \{1, \ldots, K\}$ indicates which component generated observation $i$.

### Joint Distribution

The joint distribution of observations and latent variables is:

$$
p(\mathbf{x}_i, z_i = k | \theta) = p(z_i = k | \boldsymbol{\pi}) \, p(\mathbf{x}_i | z_i = k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### Marginal Distribution (Mixture Density)

Marginalizing over the latent variable:

$$
p(\mathbf{x}_i | \theta) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

This is a **weighted sum of Gaussians**, capable of modeling complex, multimodal distributions.

### Parameters

The full parameter set is $\theta = \{\boldsymbol{\pi}, \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K\}$:

| Parameter | Dimension | Constraint |
|-----------|-----------|------------|
| $\pi_k$ (mixing weight) | Scalar | $\pi_k \geq 0$, $\sum_k \pi_k = 1$ |
| $\boldsymbol{\mu}_k$ (mean) | $d \times 1$ | None |
| $\boldsymbol{\Sigma}_k$ (covariance) | $d \times d$ | Positive definite, symmetric |

### Parameter Count

For a $K$-component GMM in $d$ dimensions:

| Covariance Type | Parameters per Component | Total Parameters |
|-----------------|-------------------------|------------------|
| Full | $d + d(d+1)/2$ | $K(d + d(d+1)/2) + (K-1)$ |
| Diagonal | $d + d = 2d$ | $K \cdot 2d + (K-1)$ |
| Spherical | $d + 1$ | $K(d+1) + (K-1)$ |
| Tied (shared) | $d$ | $Kd + d(d+1)/2 + (K-1)$ |

---

## Log-Likelihood Function

### Complete-Data Log-Likelihood

If we observed both $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^N$ and $\mathbf{Z} = \{z_i\}_{i=1}^N$:

$$
\ell_c(\theta) = \log p(\mathbf{X}, \mathbf{Z} | \theta) = \sum_{i=1}^{N} \sum_{k=1}^{K} \mathbb{1}[z_i = k] \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

This is **linear** in the indicator $\mathbb{1}[z_i = k]$ and easy to optimize.

### Marginal Log-Likelihood

The observed (incomplete) data log-likelihood:

$$
\ell(\theta) = \sum_{i=1}^{N} \log p(\mathbf{x}_i | \theta) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

The **log of sum** structure makes direct optimization intractable—this motivates EM.

### Why Direct MLE Fails

1. **No closed-form solution**: Setting $\nabla_\theta \ell = 0$ yields coupled nonlinear equations
2. **Multimodality**: The likelihood surface has many local maxima (at least $K!$ due to label permutations)
3. **Singularities**: Likelihood is unbounded when a component collapses to a single point

---

## EM for GMM: Complete Derivation

### E-Step: Computing Responsibilities

The E-step computes the posterior probability that observation $i$ belongs to component $k$:

$$
\gamma_{ik} = p(z_i = k | \mathbf{x}_i, \theta^{(t)}) = \frac{p(z_i = k) \, p(\mathbf{x}_i | z_i = k)}{\sum_{j=1}^{K} p(z_i = j) \, p(\mathbf{x}_i | z_i = j)}
$$

Substituting the model:

$$
\gamma_{ik} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

**Interpretation**: $\gamma_{ik}$ is the "responsibility" that component $k$ takes for explaining observation $i$.

### Expected Sufficient Statistics

Define the effective number of points assigned to component $k$:

$$
N_k = \sum_{i=1}^{N} \gamma_{ik}
$$

Note: $\sum_{k=1}^{K} N_k = N$ since responsibilities sum to 1 for each observation.

### M-Step: Updating Parameters

**Q-function**:

$$
Q(\theta | \theta^{(t)}) = \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_{ik} \left[ \log \pi_k + \log \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]
$$

Expanding the Gaussian:

$$
Q = \sum_{i,k} \gamma_{ik} \log \pi_k - \frac{1}{2} \sum_{i,k} \gamma_{ik} \left[ d \log(2\pi) + \log|\boldsymbol{\Sigma}_k| + (\mathbf{x}_i - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \right]
$$

**Mixing Proportions** (constrained optimization with $\sum_k \pi_k = 1$):

$$
\boxed{\pi_k^{(t+1)} = \frac{N_k}{N}}
$$

**Means** (weighted average):

$$
\boxed{\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \, \mathbf{x}_i}
$$

**Covariances** (weighted empirical covariance):

$$
\boxed{\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \, (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top}
$$

### Algorithm Summary

```
Input: Data X, number of components K
Output: Parameters θ = {π, μ, Σ}

1. Initialize parameters θ⁽⁰⁾
2. Repeat until convergence:
   
   # E-step: Compute responsibilities
   For each i, k:
       γ_ik ← π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)
   
   # M-step: Update parameters  
   For each k:
       N_k ← Σ_i γ_ik
       π_k ← N_k / N
       μ_k ← (1/N_k) Σ_i γ_ik x_i
       Σ_k ← (1/N_k) Σ_i γ_ik (x_i - μ_k)(x_i - μ_k)ᵀ
   
   # Check convergence
   If |ℓ(θ⁽ᵗ⁺¹⁾) - ℓ(θ⁽ᵗ⁾)| < tolerance: break

3. Return θ
```

---

## Initialization Strategies

### The Importance of Initialization

GMM likelihood surfaces are highly multimodal. Poor initialization can lead to:

- Convergence to suboptimal local maxima
- Slow convergence
- Component collapse or degeneracy

### K-Means Initialization

Use K-means clustering to initialize:

1. Run K-means on data to get cluster centers and assignments
2. Set $\boldsymbol{\mu}_k$ to K-means centers
3. Set $\boldsymbol{\Sigma}_k$ to empirical covariance of assigned points
4. Set $\pi_k$ to proportion of points in cluster $k$

**Advantages**: Fast, deterministic (given K-means seed), provides reasonable starting point

### K-Means++ Initialization

Improved center selection for K-means:

1. Choose first center uniformly at random from data
2. For each subsequent center, choose point with probability proportional to squared distance from nearest existing center
3. Proceed with K-means from these centers

This spreads initial centers and avoids placing them too close together.

### Random Initialization

Simple but requires multiple restarts:

1. $\boldsymbol{\mu}_k$: Random subset of data points, or sample from data range
2. $\boldsymbol{\Sigma}_k$: Identity matrix scaled by data variance
3. $\pi_k$: Uniform ($1/K$) or random from Dirichlet

### Hierarchical Initialization

Start with fewer components and split:

1. Fit GMM with $K' < K$ components
2. Split largest/most spread component
3. Repeat until $K$ components reached

---

## Model Selection

### Choosing the Number of Components

The optimal $K$ is unknown and must be selected. Common approaches:

### Information Criteria

**Bayesian Information Criterion (BIC)**:

$$
\text{BIC} = -2 \ell(\hat{\theta}) + p \log N
$$

where $p$ is the number of parameters. BIC penalizes complexity more heavily for large $N$.

**Akaike Information Criterion (AIC)**:

$$
\text{AIC} = -2 \ell(\hat{\theta}) + 2p
$$

AIC tends to select more complex models than BIC.

**Integrated Completed Likelihood (ICL)**:

$$
\text{ICL} = \text{BIC} - 2 \sum_{i=1}^{N} \sum_{k=1}^{K} \hat{\gamma}_{ik} \log \hat{\gamma}_{ik}
$$

ICL adds an entropy penalty, favoring models with more certain cluster assignments.

### Selection Procedure

1. Fit GMMs for $K = 1, 2, \ldots, K_{\max}$
2. Compute information criterion for each
3. Select $K$ that minimizes the criterion

```
K_values = range(1, K_max + 1)
bic_scores = []

for K in K_values:
    gmm = fit_gmm(X, K)
    n_params = K * (1 + d + d*(d+1)/2) - 1  # Full covariance
    bic = -2 * gmm.log_likelihood(X) + n_params * log(N)
    bic_scores.append(bic)

best_K = K_values[argmin(bic_scores)]
```

### Cross-Validation

Held-out log-likelihood:

1. Split data into training and validation sets
2. Fit GMM on training data
3. Evaluate log-likelihood on validation data
4. Select $K$ maximizing validation likelihood

---

## Covariance Constraints

### Full Covariance

Each component has its own unrestricted $d \times d$ positive definite covariance:

$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top
$$

**Pros**: Maximum flexibility, captures correlations
**Cons**: Most parameters, risk of singularity, requires more data

### Diagonal Covariance

Restrict to diagonal matrices $\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$:

$$
\sigma_{kj}^{2(t+1)} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (x_{ij} - \mu_{kj})^2
$$

**Pros**: Fewer parameters, more stable
**Cons**: Assumes features are uncorrelated within clusters

### Spherical Covariance

Single variance per component $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$:

$$
\sigma_k^{2(t+1)} = \frac{1}{N_k \cdot d} \sum_{i=1}^{N} \gamma_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

**Pros**: Fewest parameters, most stable
**Cons**: Assumes isotropic clusters

### Tied Covariance

All components share the same covariance $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$:

$$
\boldsymbol{\Sigma}^{(t+1)} = \frac{1}{N} \sum_{k=1}^{K} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top
$$

**Pros**: Regularization effect, stable estimation
**Cons**: Restrictive assumption

---

## Regularization and Singularities

### The Singularity Problem

If a component's covariance becomes singular (determinant approaches zero), the likelihood becomes unbounded:

$$
\mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \propto |\boldsymbol{\Sigma}_k|^{-1/2} \to \infty \text{ as } |\boldsymbol{\Sigma}_k| \to 0
$$

This occurs when:

- A component collapses to fit a single data point exactly
- Fewer than $d+1$ points are effectively assigned to a component
- Data lies in a lower-dimensional subspace

### Covariance Regularization

**Diagonal Loading** (Ridge regularization):

$$
\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \lambda \mathbf{I}
$$

Adds small positive value to diagonal, ensuring positive definiteness.

**Minimum Eigenvalue Constraint**:

$$
\boldsymbol{\Sigma}_k = \mathbf{U} \max(\boldsymbol{\Lambda}, \epsilon \mathbf{I}) \mathbf{U}^\top
$$

where $\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$ is the eigendecomposition.

**Shrinkage Estimator** (Ledoit-Wolf):

$$
\boldsymbol{\Sigma}_k^{\text{shrunk}} = (1 - \alpha) \boldsymbol{\Sigma}_k + \alpha \cdot \text{tr}(\boldsymbol{\Sigma}_k)/d \cdot \mathbf{I}
$$

### Bayesian Regularization

Place priors on parameters:

- **Inverse-Wishart prior** on $\boldsymbol{\Sigma}_k$: Acts as pseudo-observations
- **Dirichlet prior** on $\boldsymbol{\pi}$: Prevents components from having zero weight

The MAP estimate incorporates these priors:

$$
\boldsymbol{\Sigma}_k^{\text{MAP}} = \frac{N_k \boldsymbol{\Sigma}_k^{\text{MLE}} + \nu_0 \boldsymbol{\Psi}_0}{N_k + \nu_0 + d + 1}
$$

where $\nu_0, \boldsymbol{\Psi}_0$ are prior hyperparameters.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GaussianMixtureModel:
    """
    Gaussian Mixture Model with full EM implementation.
    
    Supports multiple covariance types and regularization.
    """
    
    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = 'full',
        reg_covar: float = 1e-6,
        init_method: str = 'kmeans',
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        """
        Args:
            n_components: Number of mixture components (K)
            n_features: Dimensionality of data (d)
            covariance_type: 'full', 'diagonal', 'spherical', or 'tied'
            reg_covar: Regularization added to covariance diagonal
            init_method: 'kmeans', 'random', or 'kmeans++'
            n_init: Number of initializations to try
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed
        """
        self.K = n_components
        self.d = n_features
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.init_method = init_method
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # Parameters (initialized in fit)
        self.weights_ = None      # (K,)
        self.means_ = None        # (K, d)
        self.covariances_ = None  # Shape depends on covariance_type
        
        # Fitting info
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = float('-inf')
        
    def _initialize_parameters(self, X: torch.Tensor):
        """Initialize GMM parameters."""
        N, d = X.shape
        
        if self.init_method == 'kmeans':
            # Simple K-means initialization
            indices = torch.randperm(N)[:self.K]
            self.means_ = X[indices].clone()
            
            # Run a few K-means iterations
            for _ in range(10):
                # Assign points to nearest center
                dists = torch.cdist(X, self.means_)
                assignments = dists.argmin(dim=1)
                
                # Update centers
                for k in range(self.K):
                    mask = assignments == k
                    if mask.sum() > 0:
                        self.means_[k] = X[mask].mean(dim=0)
            
            # Initialize covariances from K-means clusters
            self._initialize_covariances(X, assignments)
            
            # Initialize weights
            self.weights_ = torch.bincount(
                assignments, minlength=self.K
            ).float() / N
            
        elif self.init_method == 'random':
            # Random initialization
            self.means_ = X[torch.randperm(N)[:self.K]].clone()
            self.weights_ = torch.ones(self.K) / self.K
            self._initialize_covariances(X, None)
            
    def _initialize_covariances(
        self, X: torch.Tensor, assignments: Optional[torch.Tensor]
    ):
        """Initialize covariance matrices."""
        N, d = X.shape
        
        if self.covariance_type == 'full':
            self.covariances_ = torch.stack([
                torch.eye(d) * X.var() for _ in range(self.K)
            ])
        elif self.covariance_type == 'diagonal':
            self.covariances_ = torch.stack([
                X.var(dim=0) for _ in range(self.K)
            ])
        elif self.covariance_type == 'spherical':
            self.covariances_ = torch.ones(self.K) * X.var()
        elif self.covariance_type == 'tied':
            self.covariances_ = torch.eye(d) * X.var()
            
    def _compute_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x|z=k) for each observation and component.
        
        Returns:
            log_prob: (N, K) tensor of log-probabilities
        """
        N = X.shape[0]
        log_prob = torch.zeros(N, self.K)
        
        for k in range(self.K):
            if self.covariance_type == 'full':
                cov = self.covariances_[k]
            elif self.covariance_type == 'diagonal':
                cov = torch.diag(self.covariances_[k])
            elif self.covariance_type == 'spherical':
                cov = self.covariances_[k] * torch.eye(self.d)
            elif self.covariance_type == 'tied':
                cov = self.covariances_
                
            # Add regularization
            cov = cov + self.reg_covar * torch.eye(self.d)
            
            # Compute log probability
            diff = X - self.means_[k]
            
            # Cholesky for numerical stability
            try:
                L = torch.linalg.cholesky(cov)
                log_det = 2 * torch.log(torch.diag(L)).sum()
                solve = torch.linalg.solve_triangular(L, diff.T, upper=False)
                mahalanobis = (solve ** 2).sum(dim=0)
            except:
                # Fallback to direct computation
                log_det = torch.logdet(cov)
                cov_inv = torch.inverse(cov)
                mahalanobis = (diff @ cov_inv * diff).sum(dim=1)
            
            log_prob[:, k] = -0.5 * (
                self.d * torch.log(torch.tensor(2 * torch.pi)) +
                log_det +
                mahalanobis
            )
            
        return log_prob
    
    def _e_step(self, X: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        E-step: compute responsibilities and log-likelihood.
        
        Returns:
            responsibilities: (N, K) tensor
            log_likelihood: scalar
        """
        log_prob = self._compute_log_prob(X)  # (N, K)
        log_weights = torch.log(self.weights_)  # (K,)
        
        # Log responsibilities (unnormalized)
        log_resp = log_prob + log_weights  # (N, K)
        
        # Log-sum-exp for normalization
        log_resp_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)  # (N, 1)
        
        # Normalized log responsibilities
        log_resp = log_resp - log_resp_norm
        
        # Log-likelihood
        log_likelihood = log_resp_norm.sum().item()
        
        return torch.exp(log_resp), log_likelihood
    
    def _m_step(self, X: torch.Tensor, responsibilities: torch.Tensor):
        """M-step: update parameters given responsibilities."""
        N = X.shape[0]
        
        # Effective counts
        N_k = responsibilities.sum(dim=0) + 1e-10  # (K,)
        
        # Update weights
        self.weights_ = N_k / N
        
        # Update means
        self.means_ = (responsibilities.T @ X) / N_k.unsqueeze(1)  # (K, d)
        
        # Update covariances
        self._update_covariances(X, responsibilities, N_k)
        
    def _update_covariances(
        self,
        X: torch.Tensor,
        responsibilities: torch.Tensor,
        N_k: torch.Tensor
    ):
        """Update covariance matrices based on covariance_type."""
        N, d = X.shape
        
        if self.covariance_type == 'full':
            for k in range(self.K):
                diff = X - self.means_[k]  # (N, d)
                weighted_diff = responsibilities[:, k].unsqueeze(1) * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
                
        elif self.covariance_type == 'diagonal':
            for k in range(self.K):
                diff = X - self.means_[k]
                weighted_sq_diff = responsibilities[:, k].unsqueeze(1) * (diff ** 2)
                self.covariances_[k] = weighted_sq_diff.sum(dim=0) / N_k[k]
                
        elif self.covariance_type == 'spherical':
            for k in range(self.K):
                diff = X - self.means_[k]
                sq_dist = (diff ** 2).sum(dim=1)
                self.covariances_[k] = (responsibilities[:, k] * sq_dist).sum() / (N_k[k] * d)
                
        elif self.covariance_type == 'tied':
            self.covariances_ = torch.zeros(d, d)
            for k in range(self.K):
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k].unsqueeze(1) * diff
                self.covariances_ += weighted_diff.T @ diff
            self.covariances_ /= N
    
    def fit(self, X: torch.Tensor, verbose: bool = False) -> 'GaussianMixtureModel':
        """
        Fit the GMM using EM algorithm.
        
        Args:
            X: Data tensor of shape (N, d)
            verbose: Print progress
            
        Returns:
            self
        """
        best_ll = float('-inf')
        best_params = None
        
        for init in range(self.n_init):
            # Initialize
            self._initialize_parameters(X)
            
            prev_ll = float('-inf')
            
            for iteration in range(self.max_iter):
                # E-step
                responsibilities, ll = self._e_step(X)
                
                # Check convergence
                if abs(ll - prev_ll) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    break
                    
                if verbose and iteration % 10 == 0:
                    print(f"Init {init+1}, Iter {iteration}: LL = {ll:.4f}")
                
                # M-step
                self._m_step(X, responsibilities)
                
                prev_ll = ll
            
            # Keep best initialization
            if ll > best_ll:
                best_ll = ll
                best_params = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone() if isinstance(self.covariances_, torch.Tensor)
                    else [c.clone() for c in self.covariances_]
                )
        
        # Restore best parameters
        self.weights_, self.means_, self.covariances_ = best_params
        self.lower_bound_ = best_ll
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels."""
        responsibilities, _ = self._e_step(X)
        return responsibilities.argmax(dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster probabilities."""
        responsibilities, _ = self._e_step(X)
        return responsibilities
    
    def score(self, X: torch.Tensor) -> float:
        """Compute average log-likelihood."""
        _, ll = self._e_step(X)
        return ll / X.shape[0]
    
    def bic(self, X: torch.Tensor) -> float:
        """Compute Bayesian Information Criterion."""
        N = X.shape[0]
        _, ll = self._e_step(X)
        
        # Count parameters
        if self.covariance_type == 'full':
            n_params = self.K * (1 + self.d + self.d * (self.d + 1) / 2) - 1
        elif self.covariance_type == 'diagonal':
            n_params = self.K * (1 + 2 * self.d) - 1
        elif self.covariance_type == 'spherical':
            n_params = self.K * (1 + self.d + 1) - 1
        elif self.covariance_type == 'tied':
            n_params = self.K * (1 + self.d) + self.d * (self.d + 1) / 2 - 1
            
        return -2 * ll + n_params * torch.log(torch.tensor(N)).item()
    
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples from the fitted model."""
        # Sample component assignments
        component_counts = torch.multinomial(
            self.weights_, n_samples, replacement=True
        )
        
        samples = []
        labels = []
        
        for k in range(self.K):
            n_k = (component_counts == k).sum().item()
            if n_k > 0:
                if self.covariance_type == 'full':
                    cov = self.covariances_[k]
                elif self.covariance_type == 'diagonal':
                    cov = torch.diag(self.covariances_[k])
                elif self.covariance_type == 'spherical':
                    cov = self.covariances_[k] * torch.eye(self.d)
                elif self.covariance_type == 'tied':
                    cov = self.covariances_
                
                cov = cov + self.reg_covar * torch.eye(self.d)
                
                dist = torch.distributions.MultivariateNormal(
                    self.means_[k], cov
                )
                samples.append(dist.sample((n_k,)))
                labels.extend([k] * n_k)
        
        X = torch.cat(samples, dim=0)
        y = torch.tensor(labels)
        
        # Shuffle
        perm = torch.randperm(n_samples)
        return X[perm], y[perm]
```

---

## Applications in Quantitative Finance

### Return Distribution Modeling

Financial returns exhibit non-Gaussian features (fat tails, skewness) that GMMs can capture:

$$
p(r_t) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(r_t | \mu_k, \sigma_k^2)
$$

**Interpretation**: Each component represents a market regime (normal, volatile, crisis).

### Regime Detection

Use GMM to identify market regimes:

1. Fit GMM to return time series
2. Compute responsibilities at each time point
3. Assign regime based on maximum responsibility

Regime-dependent strategies can then condition on detected state.

### Risk Measurement

GMM-based Value at Risk:

$$
\text{VaR}_\alpha = \text{quantile}\left( \sum_{k=1}^{K} \pi_k \, F_k^{-1}(\alpha) \right)
$$

where $F_k$ is the CDF of component $k$. The mixture captures fat tails better than single Gaussian.

### Portfolio Clustering

Cluster assets by return characteristics:

1. Compute feature vectors (mean, volatility, skewness, correlations)
2. Fit GMM to feature space
3. Group assets by cluster assignment

This reveals hidden structure beyond traditional sector classifications.

---

## Summary

| Aspect | Description |
|--------|-------------|
| **Model** | Weighted sum of $K$ Gaussian distributions |
| **Parameters** | Mixing weights $\boldsymbol{\pi}$, means $\{\boldsymbol{\mu}_k\}$, covariances $\{\boldsymbol{\Sigma}_k\}$ |
| **E-step** | Compute responsibilities $\gamma_{ik}$ via Bayes' theorem |
| **M-step** | Weighted MLE updates for all parameters |
| **Initialization** | K-means, K-means++, or random with multiple restarts |
| **Model Selection** | BIC, AIC, or cross-validation to choose $K$ |
| **Regularization** | Diagonal loading, minimum eigenvalue, Bayesian priors |

GMMs remain a fundamental tool in machine learning and statistics, providing both a tractable density model and a principled approach to clustering. Their EM estimation procedure exemplifies the elegance of the algorithm and serves as a template for more complex latent variable models.
