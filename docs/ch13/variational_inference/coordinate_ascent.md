# Coordinate Ascent Variational Inference

## Learning Objectives

By the end of this section, you will be able to:

1. Understand coordinate ascent as an optimization strategy for ELBO
2. Derive CAVI updates for exponential family models
3. Implement efficient CAVI algorithms with convergence monitoring
4. Analyze convergence properties and computational complexity
5. Apply CAVI to Gaussian Mixture Models

## Coordinate Ascent Strategy

Coordinate Ascent Variational Inference (CAVI) is the classical algorithm for optimizing the ELBO under mean-field assumptions. Instead of optimizing all variational parameters simultaneously, CAVI updates one factor at a time while holding others fixed.

### The CAVI Principle

For a mean-field factorization $q(\theta) = \prod_{j=1}^K q_j(\theta_j)$, CAVI iterates:

$$
\text{For } j = 1, \ldots, K: \quad q_j^{(t+1)}(\theta_j) \propto \exp\left\{\mathbb{E}_{q_{-j}^{(t)}}[\log p(\theta, \mathcal{D})]\right\}
$$

where $q_{-j}^{(t)}$ uses the most recent updates for factors $1, \ldots, j-1$ and previous iteration values for factors $j+1, \ldots, K$.

### Algorithm Structure

```
Algorithm: Coordinate Ascent Variational Inference (CAVI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: 
  - Model: p(D, θ) = p(D|θ)p(θ)
  - Factorization: q(θ) = ∏ⱼ qⱼ(θⱼ)
  - Tolerance: ε
  - Max iterations: T

Output: Optimized variational factors {qⱼ*}

1. Initialize: Set initial variational parameters {φⱼ⁽⁰⁾}
2. Compute: ELBO⁽⁰⁾

3. For t = 1, 2, ..., T:
   a. For j = 1, 2, ..., K:
      i.   Compute: E_{q₋ⱼ}[log p(θ, D)]
      ii.  Update: qⱼ⁽ᵗ⁾(θⱼ) ∝ exp{E_{q₋ⱼ}[log p(θ, D)]}
      iii. Extract: New variational parameters φⱼ⁽ᵗ⁾
   
   b. Compute: ELBO⁽ᵗ⁾
   
   c. If |ELBO⁽ᵗ⁾ - ELBO⁽ᵗ⁻¹⁾| < ε:
      Return {qⱼ⁽ᵗ⁾}

4. Return {qⱼ⁽ᵀ⁾} (may not have converged)
```

## Convergence Properties

### Monotonicity Theorem

**Theorem**: Each CAVI update cannot decrease the ELBO.

**Proof**: Consider updating factor $q_j$ while holding all others fixed. The ELBO can be written as:

$$
\text{ELBO}(q_1, \ldots, q_K) = \mathbb{E}_q[\log p(\mathcal{D}, \theta)] - \sum_{k=1}^K \mathbb{E}_{q_k}[\log q_k(\theta_k)]
$$

The terms involving $q_j$ are:

$$
\text{ELBO}_j = \mathbb{E}_{q_j}\left[\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\right] - \mathbb{E}_{q_j}[\log q_j(\theta_j)]
$$

This is the negative KL divergence (up to a constant):

$$
\text{ELBO}_j = -\text{KL}\left(q_j(\theta_j) \| \tilde{p}_j(\theta_j)\right) + \text{const}
$$

where $\tilde{p}_j(\theta_j) \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\mathcal{D}, \theta)]\}$.

Since KL divergence is minimized (equals zero) when $q_j = \tilde{p}_j$, the CAVI update maximizes $\text{ELBO}_j$. Therefore, each update can only increase or maintain the ELBO.

### Convergence to Local Optimum

**Corollary**: CAVI converges to a local optimum of the ELBO.

The ELBO is bounded above by $\log p(\mathcal{D})$. Combined with monotonicity, the sequence of ELBO values must converge. At convergence, no single-factor update can improve the ELBO, which is the definition of a coordinate-wise optimum.

### Rate of Convergence

CAVI typically exhibits linear convergence:

$$
\text{ELBO}^* - \text{ELBO}^{(t)} \leq C \cdot \rho^t
$$

where $\rho < 1$ is the convergence rate and depends on the problem structure.

## CAVI for Exponential Family Models

When both the complete conditional distributions and variational factors belong to exponential families, CAVI updates take a particularly elegant form.

### Exponential Family Background

A distribution is in the exponential family if it can be written as:

$$
p(x | \eta) = h(x) \exp\left\{\eta^\top T(x) - A(\eta)\right\}
$$

where:
- $\eta$ are the **natural parameters**
- $T(x)$ are the **sufficient statistics**
- $A(\eta)$ is the **log-partition function**
- $h(x)$ is the **base measure**

### CAVI Update in Natural Parameters

For exponential family models, the CAVI update for factor $q_j$ reduces to updating natural parameters:

$$
\eta_j^{(t+1)} = \mathbb{E}_{q_{-j}^{(t)}}\left[\eta_j(\theta_{-j}, \mathcal{D})\right]
$$

where $\eta_j(\theta_{-j}, \mathcal{D})$ are the natural parameters of the complete conditional $p(\theta_j | \theta_{-j}, \mathcal{D})$.

### Common Exponential Family Updates

| Distribution | Natural Parameters | Update Rule |
|--------------|-------------------|-------------|
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\eta_1 = \mu/\sigma^2$, $\eta_2 = -1/(2\sigma^2)$ | Mean of expectations |
| Gamma $\text{Ga}(\alpha, \beta)$ | $\eta_1 = \alpha - 1$, $\eta_2 = -\beta$ | Sum of expected statistics |
| Categorical $\text{Cat}(\pi)$ | $\eta_k = \log \pi_k$ | Log of expected probabilities |
| Dirichlet $\text{Dir}(\alpha)$ | $\eta_k = \alpha_k - 1$ | Sum of expected log-probabilities |

## Example: Gaussian Mixture Model

The Gaussian Mixture Model (GMM) is a canonical example for CAVI.

### Model Specification

$$
\begin{aligned}
\text{Mixing weights: } & \pi \sim \text{Dir}(\alpha_0) \\
\text{Component means: } & \mu_k \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad k = 1, \ldots, K \\
\text{Cluster assignments: } & z_i \sim \text{Cat}(\pi) \quad i = 1, \ldots, N \\
\text{Observations: } & x_i | z_i, \{\mu_k\} \sim \mathcal{N}(\mu_{z_i}, \sigma^2)
\end{aligned}
$$

### Mean-Field Factorization

$$
q(\pi, \{\mu_k\}, \{z_i\}) = q(\pi) \prod_{k=1}^K q(\mu_k) \prod_{i=1}^N q(z_i)
$$

### CAVI Updates

**Update for $q(z_i)$ (cluster assignments):**

$$
q^*(z_i = k) \propto \exp\left\{\mathbb{E}[\log \pi_k] + \mathbb{E}\left[\log \mathcal{N}(x_i | \mu_k, \sigma^2)\right]\right\}
$$

Let $r_{ik} = q(z_i = k)$, then:

$$
r_{ik} \propto \exp\left\{\psi(\alpha_k) - \psi\left(\sum_j \alpha_j\right) - \frac{1}{2\sigma^2}\left[(x_i - \mathbb{E}[\mu_k])^2 + \text{Var}[\mu_k]\right]\right\}
$$

**Update for $q(\mu_k)$ (component means):**

$$
q^*(\mu_k) = \mathcal{N}(\mu_k | m_k, s_k^2)
$$

where:

$$
\begin{aligned}
s_k^2 &= \left(\frac{1}{\sigma_0^2} + \frac{N_k}{\sigma^2}\right)^{-1} \\
m_k &= s_k^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_i r_{ik} x_i}{\sigma^2}\right)
\end{aligned}
$$

and $N_k = \sum_i r_{ik}$ is the expected number of points in cluster $k$.

**Update for $q(\pi)$ (mixing weights):**

$$
q^*(\pi) = \text{Dir}(\alpha_1, \ldots, \alpha_K)
$$

where:

$$
\alpha_k = \alpha_0 + \sum_{i=1}^N r_{ik}
$$

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Categorical
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

class GMMCavi:
    """
    Coordinate Ascent VI for Gaussian Mixture Model.
    
    Model:
        π ~ Dir(α₀)
        μₖ ~ N(μ₀, σ₀²)
        zᵢ ~ Cat(π)
        xᵢ | zᵢ ~ N(μ_{zᵢ}, σ²)
    
    Variational family:
        q(π, μ, z) = q(π) ∏ₖ q(μₖ) ∏ᵢ q(zᵢ)
    """
    
    def __init__(self, n_components: int, 
                 alpha_0: float = 1.0,
                 mu_0: float = 0.0,
                 sigma_0: float = 10.0,
                 sigma: float = 1.0):
        """
        Initialize GMM-CAVI.
        
        Args:
            n_components: Number of mixture components K
            alpha_0: Dirichlet prior concentration
            mu_0: Prior mean for component means
            sigma_0: Prior std for component means
            sigma: Known observation noise std
        """
        self.K = n_components
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.sigma = sigma
        
        # Variational parameters (to be initialized with data)
        self.r = None       # N x K responsibility matrix
        self.alpha = None   # K Dirichlet parameters
        self.m = None       # K component means
        self.s2 = None      # K component variances
    
    def initialize(self, data: torch.Tensor, init_method: str = 'kmeans'):
        """
        Initialize variational parameters.
        """
        N = len(data)
        
        if init_method == 'random':
            # Random initialization
            self.r = torch.rand(N, self.K)
            self.r = self.r / self.r.sum(dim=1, keepdim=True)
        elif init_method == 'kmeans':
            # K-means++ style initialization
            indices = torch.randperm(N)[:self.K]
            centers = data[indices].clone()
            
            # Compute distances and assign
            dists = torch.cdist(data.unsqueeze(1), centers.unsqueeze(0).unsqueeze(2))
            self.r = F.softmax(-dists.squeeze() / self.sigma**2, dim=1)
        
        # Initialize other parameters based on responsibilities
        self._update_q_pi()
        self._update_q_mu(data)
    
    def _update_q_z(self, data: torch.Tensor) -> None:
        """
        Update q(zᵢ) for all data points.
        
        q*(zᵢ = k) ∝ exp{E[log πₖ] - (xᵢ - E[μₖ])²/(2σ²) - Var[μₖ]/(2σ²)}
        """
        N = len(data)
        
        # E[log πₖ] = ψ(αₖ) - ψ(Σⱼ αⱼ)
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        
        # Log responsibilities (unnormalized)
        log_r = torch.zeros(N, self.K)
        
        for k in range(self.K):
            # E[(xᵢ - μₖ)²] = (xᵢ - E[μₖ])² + Var[μₖ]
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            log_r[:, k] = E_log_pi[k] - 0.5 / self.sigma**2 * E_sq_diff
        
        # Normalize (softmax)
        self.r = F.softmax(log_r, dim=1)
    
    def _update_q_mu(self, data: torch.Tensor) -> None:
        """
        Update q(μₖ) for all components.
        
        q*(μₖ) = N(mₖ, sₖ²)
        """
        N_k = self.r.sum(dim=0)  # Expected count per component
        
        # Posterior precision
        precision = 1/self.sigma_0**2 + N_k / self.sigma**2
        self.s2 = 1 / precision
        
        # Posterior mean
        weighted_sum = (self.r * data.unsqueeze(1)).sum(dim=0)
        self.m = self.s2 * (self.mu_0 / self.sigma_0**2 + weighted_sum / self.sigma**2)
    
    def _update_q_pi(self) -> None:
        """
        Update q(π).
        
        q*(π) = Dir(α₁, ..., αₖ)
        αₖ = α₀ + Σᵢ rᵢₖ
        """
        N_k = self.r.sum(dim=0)
        self.alpha = self.alpha_0 + N_k
    
    def compute_elbo(self, data: torch.Tensor) -> float:
        """
        Compute the Evidence Lower Bound.
        """
        N = len(data)
        N_k = self.r.sum(dim=0)
        
        # E[log p(x|z,μ)] - Expected log-likelihood
        E_log_lik = 0.0
        for k in range(self.K):
            E_sq_diff = (data - self.m[k])**2 + self.s2[k]
            E_log_lik += (self.r[:, k] * (
                -0.5 * np.log(2 * np.pi * self.sigma**2)
                - 0.5 / self.sigma**2 * E_sq_diff
            )).sum()
        
        # E[log p(z|π)] - Expected log prior on z
        E_log_pi = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        E_log_p_z = (self.r * E_log_pi).sum()
        
        # E[log p(π)] - Expected log prior on π
        E_log_p_pi = (
            torch.lgamma(torch.tensor(self.K * self.alpha_0))
            - self.K * torch.lgamma(torch.tensor(self.alpha_0))
            + (self.alpha_0 - 1) * E_log_pi.sum()
        )
        
        # E[log p(μ)] - Expected log prior on μ
        E_log_p_mu = 0.0
        for k in range(self.K):
            E_log_p_mu += (
                -0.5 * np.log(2 * np.pi * self.sigma_0**2)
                - 0.5 / self.sigma_0**2 * ((self.m[k] - self.mu_0)**2 + self.s2[k])
            )
        
        # -E[log q(z)] - Entropy of q(z)
        H_q_z = -(self.r * torch.log(self.r + 1e-10)).sum()
        
        # -E[log q(π)] - Entropy of q(π)
        H_q_pi = (
            torch.lgamma(self.alpha.sum())
            - torch.lgamma(self.alpha).sum()
            + (self.alpha - 1).dot(
                torch.digamma(self.alpha.sum()) - torch.digamma(self.alpha)
            )
        )
        
        # -E[log q(μ)] - Entropy of q(μ)
        H_q_mu = 0.5 * self.K * (1 + np.log(2 * np.pi)) + 0.5 * torch.log(self.s2).sum()
        
        elbo = (E_log_lik + E_log_p_z + E_log_p_pi + E_log_p_mu 
                + H_q_z + H_q_pi + H_q_mu)
        
        return elbo.item()
    
    def fit(self, data: torch.Tensor, max_iter: int = 100,
            tol: float = 1e-6, verbose: bool = True) -> Dict:
        """
        Run CAVI algorithm.
        """
        # Initialize
        self.initialize(data)
        
        history = {
            'elbo': [],
            'r': [],
            'm': [],
            'alpha': []
        }
        
        elbo_prev = -float('inf')
        
        for iteration in range(max_iter):
            # CAVI updates
            self._update_q_z(data)
            self._update_q_mu(data)
            self._update_q_pi()
            
            # Compute ELBO
            elbo = self.compute_elbo(data)
            
            # Record history
            history['elbo'].append(elbo)
            history['r'].append(self.r.clone())
            history['m'].append(self.m.clone())
            history['alpha'].append(self.alpha.clone())
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ELBO = {elbo:.4f}, "
                      f"means = {self.m.numpy()}")
            
            # Check convergence
            if abs(elbo - elbo_prev) < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                break
            
            elbo_prev = elbo
        
        return history
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments.
        """
        self._update_q_z(data)
        return self.r.argmax(dim=1)


def visualize_gmm_cavi(model: GMMCavi, history: Dict, data: torch.Tensor,
                       true_labels: torch.Tensor = None):
    """Visualize GMM CAVI results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: ELBO convergence
    ax = axes[0, 0]
    ax.plot(history['elbo'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('(a) ELBO Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Component means convergence
    ax = axes[0, 1]
    means_history = torch.stack(history['m'])
    for k in range(model.K):
        ax.plot(means_history[:, k], linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(b) Component Means', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final clustering
    ax = axes[0, 2]
    predictions = model.r.argmax(dim=1)
    colors = plt.cm.tab10(predictions.numpy() / model.K)
    ax.scatter(data.numpy(), np.zeros_like(data.numpy()), c=colors, alpha=0.6, s=50)
    for k in range(model.K):
        ax.axvline(model.m[k].item(), color=plt.cm.tab10(k / model.K), 
                   linestyle='--', linewidth=2, label=f'μ_{k+1}')
    ax.set_xlabel('x', fontsize=11)
    ax.set_title('(c) Final Clustering', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Responsibility evolution
    ax = axes[1, 0]
    r_history = torch.stack(history['r'])
    sample_idx = 0  # Track first data point
    for k in range(model.K):
        ax.plot(r_history[:, sample_idx, k], linewidth=2, label=f'r_{sample_idx+1},{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Responsibility', fontsize=11)
    ax.set_title(f'(d) Responsibilities (point {sample_idx+1})', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Dirichlet parameters
    ax = axes[1, 1]
    alpha_history = torch.stack(history['alpha'])
    for k in range(model.K):
        ax.plot(alpha_history[:, k], linewidth=2, label=f'α_{k+1}')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Concentration', fontsize=11)
    ax.set_title('(e) Dirichlet Parameters', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final mixture density
    ax = axes[1, 2]
    x_range = torch.linspace(data.min() - 2, data.max() + 2, 500)
    
    E_pi = model.alpha / model.alpha.sum()
    mixture_pdf = torch.zeros_like(x_range)
    
    for k in range(model.K):
        component_pdf = E_pi[k] * torch.exp(
            -0.5 * (x_range - model.m[k])**2 / (model.sigma**2 + model.s2[k])
        ) / np.sqrt(2 * np.pi * (model.sigma**2 + model.s2[k]))
        mixture_pdf += component_pdf
        ax.plot(x_range.numpy(), component_pdf.numpy(), '--', 
                linewidth=1.5, alpha=0.7, label=f'Component {k+1}')
    
    ax.plot(x_range.numpy(), mixture_pdf.numpy(), 'b-', linewidth=2.5, 
            label='Mixture')
    ax.hist(data.numpy(), bins=30, density=True, alpha=0.4, color='gray')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(f) Fitted Mixture', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gmm_cavi.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic mixture data
    n_samples = 300
    true_means = torch.tensor([-3.0, 0.0, 4.0])
    true_weights = torch.tensor([0.3, 0.4, 0.3])
    sigma = 1.0
    
    # Sample from mixture
    z_true = torch.multinomial(true_weights, n_samples, replacement=True)
    data = torch.randn(n_samples) * sigma + true_means[z_true]
    
    print("=" * 60)
    print("Gaussian Mixture Model CAVI")
    print("=" * 60)
    print(f"\nTrue means: {true_means.numpy()}")
    print(f"True weights: {true_weights.numpy()}")
    
    # Fit GMM with CAVI
    model = GMMCavi(n_components=3, alpha_0=1.0, sigma=sigma)
    history = model.fit(data, max_iter=100, verbose=True)
    
    print(f"\nFinal estimates:")
    print(f"  Means: {model.m.numpy()}")
    print(f"  Expected weights: {(model.alpha / model.alpha.sum()).numpy()}")
    
    # Visualize
    visualize_gmm_cavi(model, history, data)
```

## Computational Complexity

### Per-Iteration Cost

For a model with $K$ factors, $N$ data points:

| Component | Complexity |
|-----------|------------|
| Update $q(z_i)$ for all $i$ | $O(NK)$ |
| Update $q(\mu_k)$ for all $k$ | $O(NK)$ |
| Update $q(\pi)$ | $O(K)$ |
| Compute ELBO | $O(NK)$ |
| **Total per iteration** | $O(NK)$ |

### Comparison with EM

CAVI for GMM has the same complexity as the EM algorithm, but provides full posterior distributions rather than point estimates.

## Summary

CAVI provides a systematic approach to optimizing the ELBO:

**Algorithm**:
1. Initialize variational parameters
2. Update each factor in turn: $q_j^* \propto \exp\{\mathbb{E}_{q_{-j}}[\log p(\theta, \mathcal{D})]\}$
3. Monitor ELBO until convergence

**Properties**:
- Monotonic ELBO improvement
- Converges to local optimum
- Closed-form updates for exponential families
- Same complexity as EM for mixture models

## Exercises

### Exercise 1: CAVI for Factor Analysis

Derive and implement CAVI for a probabilistic factor analysis model.

### Exercise 2: Convergence Analysis

Empirically measure the convergence rate of CAVI for GMM as a function of cluster separation.

### Exercise 3: Initialization Sensitivity

Study how different initialization strategies affect the final ELBO and cluster quality.

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.

2. Blei, D. M., & Jordan, M. I. (2006). "Variational Inference for Dirichlet Process Mixtures."

3. Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). "Stochastic Variational Inference."

4. Wainwright, M. J., & Jordan, M. I. (2008). "Graphical Models, Exponential Families, and Variational Inference."
