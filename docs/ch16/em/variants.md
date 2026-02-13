# EM Variants

The classical EM algorithm, while elegant, has limitations: slow convergence, sensitivity to initialization, and inapplicability when the E-step or M-step lack closed-form solutions. This section presents important variants that address these challenges, extending EM's reach to a broader class of problems.

---

## Generalized EM (GEM)

### Motivation

The standard M-step requires **maximizing** the Q-function:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})
$$

In many models, this maximization has no closed-form solution. Generalized EM relaxes this requirement.

### The GEM Principle

**Generalized EM** only requires that the M-step **increases** (rather than maximizes) the Q-function:

$$
Q(\theta^{(t+1)} | \theta^{(t)}) \geq Q(\theta^{(t)} | \theta^{(t)})
$$

Any improvement suffices—we need not find the global maximum.

### Theoretical Guarantee

**Theorem**: GEM retains the monotonic improvement property:

$$
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
$$

**Proof**: The same chain of inequalities holds:

$$
\ell(\theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \ell(\theta^{(t)})
$$

The second inequality requires only that $\theta^{(t+1)}$ improves the ELBO, not that it maximizes it.

### Implementation Strategies

**Gradient Ascent M-Step**: Take one or more gradient steps:

$$
\theta^{(t+1)} = \theta^{(t)} + \eta \nabla_\theta Q(\theta | \theta^{(t)})\big|_{\theta = \theta^{(t)}}
$$

**Conditional Maximization**: Maximize over parameter subsets sequentially (see ECM below).

**Constrained Optimization**: When parameters are constrained, projected gradient methods suffice.

### When to Use GEM

- No closed-form M-step solution
- Constrained parameter spaces
- Computational budget limits
- Complex likelihood structures

---

## Expectation Conditional Maximization (ECM)

### The Challenge of Coupled Parameters

In many models, jointly optimizing all parameters in the M-step is intractable, even though optimizing each parameter group separately is easy.

### ECM Algorithm

**ECM** replaces the M-step with a sequence of **conditional maximization (CM) steps**, each maximizing over a subset of parameters while holding others fixed.

Partition parameters as $\theta = (\theta_1, \theta_2, \ldots, \theta_S)$. The ECM iteration:

**E-step**: Compute $q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$

**CM-steps**: For $s = 1, \ldots, S$:

$$
\theta_s^{(t+1)} = \arg\max_{\theta_s} Q(\theta_1^{(t+1)}, \ldots, \theta_{s-1}^{(t+1)}, \theta_s, \theta_{s+1}^{(t)}, \ldots, \theta_S^{(t)} | \theta^{(t)})
$$

Each CM-step maximizes over $\theta_s$ while using updated values for $\theta_1, \ldots, \theta_{s-1}$ and old values for $\theta_{s+1}, \ldots, \theta_S$.

### Example: Factor Analysis

Parameters: loading matrix $\mathbf{W}$, noise variances $\boldsymbol{\Psi}$ (diagonal).

- **CM-step 1**: Update $\mathbf{W}$ given current $\boldsymbol{\Psi}$
- **CM-step 2**: Update $\boldsymbol{\Psi}$ given new $\mathbf{W}$

Each step has a closed form; joint optimization does not.

### Convergence Properties

**Theorem (Meng & Rubin, 1993)**: ECM inherits EM's convergence properties:

1. Monotonic improvement: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$
2. Convergence to stationary points under standard regularity conditions

The key insight: each CM-step increases (or maintains) the Q-function, which suffices for the GEM guarantee.

---

## ECME Algorithm

### Motivation

While ECM uses the Q-function for all CM-steps, sometimes maximizing the **actual log-likelihood** $\ell(\theta)$ directly is easier for certain parameter subsets.

### ECME = ECM + Direct Likelihood Steps

**ECME (Expectation/Conditional Maximization Either)** allows each CM-step to maximize either:

- The Q-function $Q(\theta | \theta^{(t)})$, **or**
- The log-likelihood $\ell(\theta)$ directly

whichever is more convenient or efficient.

### Algorithm Structure

For parameter partition $\theta = (\theta_1, \ldots, \theta_S)$:

**E-step**: Compute posterior $p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$

**CM-steps**: For $s = 1, \ldots, S$:

$$
\theta_s^{(t+1)} = \arg\max_{\theta_s} F_s(\theta_s)
$$

where $F_s$ is either:

- $Q(\theta_1^{(t+1)}, \ldots, \theta_s, \ldots, \theta_S^{(t)} | \theta^{(t)})$ (Q-step), or
- $\ell(\theta_1^{(t+1)}, \ldots, \theta_s, \ldots, \theta_S^{(t)})$ (L-step)

### Why ECME Can Be Faster

Direct likelihood maximization steps can:

1. Skip the E-step overhead for those parameters
2. Take larger steps in parameter space
3. Exploit problem structure unavailable to Q-function

### Convergence

**Theorem**: ECME maintains monotonic improvement:

- Q-steps increase $\mathcal{L}(q, \theta)$, which lower-bounds $\ell(\theta)$
- L-steps directly increase $\ell(\theta)$

Both ensure the sequence $\{\ell(\theta^{(t)})\}$ is non-decreasing.

---

## Parameter-Expanded EM (PX-EM)

### The Slow Convergence Problem

EM can converge slowly when the **fraction of missing information** is high (see Convergence Theory). PX-EM addresses this by temporarily expanding the parameter space.

### The Idea

1. **Expand**: Introduce auxiliary parameters that don't change the marginal model
2. **EM in expanded space**: Run EM with more degrees of freedom
3. **Reduce**: Project back to original parameter space

The expanded space often has better conditioning, leading to faster convergence.

### Formal Setup

Let $\theta \in \Theta$ be the original parameters. Introduce expanded parameters $\phi \in \Phi$ where $\dim(\Phi) > \dim(\Theta)$.

Define a **reduction function** $R: \Phi \to \Theta$ such that:

$$
p(\mathbf{X} | R(\phi)) = p(\mathbf{X} | \theta) \quad \text{for all } \phi \text{ with } R(\phi) = \theta
$$

The expanded model has the same marginal likelihood but more parameters.

### PX-EM Iteration

1. **E-step**: Compute expectations using current $\theta^{(t)}$
2. **M-step in expanded space**: Find $\phi^{(t+1)} = \arg\max_\phi Q_\phi(\phi | \theta^{(t)})$
3. **Reduction**: Set $\theta^{(t+1)} = R(\phi^{(t+1)})$

### Example: Covariance Parameter Expansion

For factor analysis with loading matrix $\mathbf{W}$ and latent covariance $\mathbf{I}$:

**Original model**: $p(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{W}\mathbf{z}, \boldsymbol{\Psi})$ with $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

**Expanded model**: Allow $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_z)$ for general positive definite $\boldsymbol{\Sigma}_z$

**Reduction**: Any $(\mathbf{W}', \boldsymbol{\Sigma}_z')$ satisfying $\mathbf{W}' \boldsymbol{\Sigma}_z' \mathbf{W}'^\top = \mathbf{W} \mathbf{W}^\top$ gives the same marginal model.

The expanded M-step can make larger moves that, after reduction, correspond to better standard EM updates.

### Convergence Rate Improvement

PX-EM typically achieves:

$$
\rho_{\text{PX-EM}} < \rho_{\text{EM}}
$$

The expansion effectively "fills in" some of the missing information, reducing the fraction lost.

---

## Monte Carlo EM (MCEM)

### When the E-Step is Intractable

For some models, the posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$ has no closed form, making exact computation of:

$$
Q(\theta | \theta^{(t)}) = \mathbb{E}_{p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

impossible.

### MCEM Approximation

**Monte Carlo EM** approximates the E-step expectation using samples from the posterior:

1. **Sample**: Draw $\mathbf{Z}^{(1)}, \ldots, \mathbf{Z}^{(M)} \sim p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$
2. **Approximate Q**: 
$$
\hat{Q}(\theta | \theta^{(t)}) = \frac{1}{M} \sum_{m=1}^{M} \log p(\mathbf{X}, \mathbf{Z}^{(m)} | \theta)
$$
3. **M-step**: $\theta^{(t+1)} = \arg\max_\theta \hat{Q}(\theta | \theta^{(t)})$

### Sampling Methods

Common approaches for sampling from the posterior:

- **Gibbs sampling**: When conditional distributions are tractable
- **Metropolis-Hastings**: General-purpose MCMC
- **Importance sampling**: With appropriate proposal distribution
- **Sequential Monte Carlo**: For sequential latent variable models

### Convergence Considerations

MCEM introduces **Monte Carlo noise**. To ensure convergence:

**1. Increasing Sample Size**: Let $M_t \to \infty$ as $t \to \infty$. Common schedules:

- $M_t = M_0$ (constant—may not converge to optimum)
- $M_t = t$ (linear growth)
- $M_t = M_0 \cdot c^t$ (exponential growth)

**2. Averaging**: Use running averages of parameter estimates to reduce variance.

**3. Ascent Condition**: Ensure sufficient accuracy in Q-approximation to guarantee ascent.

### Practical MCEM

```python
def mcem_iteration(X, theta, n_samples, sampler):
    """
    One iteration of Monte Carlo EM.
    
    Args:
        X: Observed data
        theta: Current parameters
        n_samples: Number of MC samples
        sampler: Function to sample Z from posterior
    
    Returns:
        Updated theta
    """
    # E-step: Sample from posterior
    Z_samples = [sampler(X, theta) for _ in range(n_samples)]
    
    # M-step: Maximize approximate Q
    def neg_Q(new_theta):
        return -sum(
            complete_log_likelihood(X, Z, new_theta) 
            for Z in Z_samples
        ) / n_samples
    
    result = minimize(neg_Q, theta)
    return result.x
```

---

## Stochastic EM (SEM)

### A Different Stochastic Approach

While MCEM uses multiple samples to approximate expectations, **Stochastic EM** uses a **single sample** from the posterior and treats it as if it were the true latent values.

### SEM Iteration

1. **S-step (Stochastic)**: Draw single sample $\mathbf{Z}^{(t)} \sim p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$
2. **M-step**: Maximize complete-data likelihood:
$$
\theta^{(t+1)} = \arg\max_\theta \log p(\mathbf{X}, \mathbf{Z}^{(t)} | \theta)
$$

### Properties

**Non-monotonic**: Unlike EM, SEM does not guarantee monotonic improvement—the likelihood can decrease at any step.

**Ergodic**: Under regularity conditions, the sequence $\{\theta^{(t)}\}$ forms a Markov chain that converges to a stationary distribution concentrated around the MLE.

**Exploration**: The stochasticity helps escape local optima—SEM explores the parameter space more than deterministic EM.

### SEM for Mixture Models

For GMMs:

1. **S-step**: For each observation, sample cluster assignment:
$$
z_i \sim \text{Categorical}(\gamma_{i1}, \ldots, \gamma_{iK})
$$

2. **M-step**: Compute standard MLE as if $\{z_i\}$ were observed:
$$
\pi_k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[z_i = k]
$$
$$
\boldsymbol{\mu}_k = \frac{\sum_{i: z_i = k} \mathbf{x}_i}{\sum_{i} \mathbb{1}[z_i = k]}
$$

### When to Use SEM

- **Multimodal likelihoods**: SEM's stochasticity aids exploration
- **Initial exploration**: Run SEM to find promising regions, then switch to EM
- **Bayesian sampling**: SEM as one component of MCMC for posterior inference
- **Large datasets**: Single samples are cheaper than full expectations

---

## Variational EM

### When the Posterior is Intractable

In complex models, the exact posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$ may have no closed form, preventing exact E-step computation.

### Variational Approximation

**Variational EM** replaces the exact posterior with an approximation $q(\mathbf{Z})$ from a tractable family $\mathcal{Q}$:

$$
q^*(\mathbf{Z}) = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)
$$

### The ELBO Perspective

Since exact posterior gives $\mathcal{L}(q, \theta) = \ell(\theta)$ (tight bound), a variational approximation gives:

$$
\mathcal{L}(q, \theta) = \ell(\theta) - D_{\mathrm{KL}}(q \| p(\mathbf{Z}|\mathbf{X}, \theta)) \leq \ell(\theta)
$$

Variational EM maximizes this lower bound instead of the true likelihood.

### Mean-Field Approximation

The most common choice is the **mean-field** family, which assumes latent variables factorize:

$$
q(\mathbf{Z}) = \prod_{j} q_j(z_j)
$$

This dramatically simplifies optimization while often providing good approximations.

### Variational E-Step

For mean-field, the optimal factors satisfy:

$$
\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \text{const}
$$

where $q_{-j}$ denotes all factors except $q_j$. This leads to **coordinate ascent variational inference (CAVI)**.

### Variational EM Algorithm

1. **Initialize**: $q^{(0)}(\mathbf{Z})$, $\theta^{(0)}$

2. **Repeat until convergence**:

   **VE-step**: Update variational distribution
   $$
   q^{(t+1)} = \arg\max_q \mathcal{L}(q, \theta^{(t)})
   $$
   (via CAVI or other variational methods)
   
   **M-step**: Update parameters
   $$
   \theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)
   $$

### Convergence

Variational EM converges to a **local optimum of the ELBO**, not necessarily of $\ell(\theta)$. However:

- The ELBO is often a good proxy for the likelihood
- Variational EM enables inference in models where exact EM is impossible
- The gap $\ell(\theta) - \mathcal{L}(q, \theta)$ can be monitored

---

## Incremental and Online EM

### Motivation

Standard EM requires a full pass through data for each E-step—prohibitive for large datasets or streaming data.

### Incremental EM

Process data in mini-batches, updating sufficient statistics incrementally:

**Standard EM sufficient statistics**:

$$
S_k = \sum_{i=1}^{N} \gamma_{ik}, \quad T_k = \sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i
$$

**Incremental update** with batch $\mathcal{B}$:

$$
S_k \leftarrow S_k + \sum_{i \in \mathcal{B}} \gamma_{ik}, \quad T_k \leftarrow T_k + \sum_{i \in \mathcal{B}} \gamma_{ik} \mathbf{x}_i
$$

M-step uses accumulated statistics.

### Online EM

For truly streaming data, use **stochastic approximation**:

**Online E-step** for observation $\mathbf{x}_t$:

$$
\gamma_{tk} = \frac{\pi_k^{(t-1)} \mathcal{N}(\mathbf{x}_t | \boldsymbol{\mu}_k^{(t-1)}, \boldsymbol{\Sigma}_k^{(t-1)})}{\sum_j \pi_j^{(t-1)} \mathcal{N}(\mathbf{x}_t | \boldsymbol{\mu}_j^{(t-1)}, \boldsymbol{\Sigma}_j^{(t-1)})}
$$

**Online M-step** with learning rate $\eta_t$:

$$
\hat{S}_k^{(t)} = (1 - \eta_t) \hat{S}_k^{(t-1)} + \eta_t \gamma_{tk}
$$

$$
\hat{T}_k^{(t)} = (1 - \eta_t) \hat{T}_k^{(t-1)} + \eta_t \gamma_{tk} \mathbf{x}_t
$$

$$
\boldsymbol{\mu}_k^{(t)} = \hat{T}_k^{(t)} / \hat{S}_k^{(t)}
$$

### Learning Rate Schedules

For convergence, $\eta_t$ should satisfy:

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

Common choices:

- $\eta_t = t^{-\alpha}$ for $\alpha \in (0.5, 1]$
- $\eta_t = (t + \tau)^{-\alpha}$ with delay $\tau$

### Mini-Batch Online EM

Combine mini-batch processing with online updates for reduced variance:

```python
def online_em_step(batch, theta, sufficient_stats, learning_rate):
    """Process one mini-batch in online EM."""
    
    # E-step on batch
    gamma = compute_responsibilities(batch, theta)
    
    # Compute batch sufficient statistics
    batch_S = gamma.sum(dim=0)
    batch_T = gamma.T @ batch
    batch_V = compute_weighted_outer_products(batch, gamma, theta['mu'])
    
    # Online update
    sufficient_stats['S'] = (1 - learning_rate) * sufficient_stats['S'] + learning_rate * batch_S * N_total / len(batch)
    sufficient_stats['T'] = (1 - learning_rate) * sufficient_stats['T'] + learning_rate * batch_T * N_total / len(batch)
    sufficient_stats['V'] = (1 - learning_rate) * sufficient_stats['V'] + learning_rate * batch_V * N_total / len(batch)
    
    # M-step from sufficient statistics
    theta['pi'] = sufficient_stats['S'] / sufficient_stats['S'].sum()
    theta['mu'] = sufficient_stats['T'] / sufficient_stats['S'].unsqueeze(1)
    theta['Sigma'] = sufficient_stats['V'] / sufficient_stats['S'].unsqueeze(1).unsqueeze(2)
    
    return theta, sufficient_stats
```

---

## EM Acceleration Methods

### The Convergence Rate Problem

Near convergence, EM's rate is governed by:

$$
\|\theta^{(t+1)} - \theta^*\| \approx \rho \|\theta^{(t)} - \theta^*\|
$$

where $\rho = \rho(I_{\text{comp}}^{-1} I_{\text{miss}})$ can be close to 1, causing slow convergence.

### Aitken Acceleration

Extrapolate the limit using observed convergence:

$$
\theta^*_{\text{est}} = \theta^{(t)} + \frac{\theta^{(t+1)} - \theta^{(t)}}{1 - \hat{\rho}}
$$

where $\hat{\rho} = \frac{\|\theta^{(t+1)} - \theta^{(t)}\|}{\|\theta^{(t)} - \theta^{(t-1)}\|}$.

**When to use**: After EM has entered the linear convergence regime (near optimum).

### SQUAREM (Squared Iterative Methods)

SQUAREM effectively "squares" the EM mapping, reducing rate from $\rho$ to $\rho^2$:

Let $\mathbf{r} = \theta^{(t+1)} - \theta^{(t)}$ and $\mathbf{v} = \theta^{(t+2)} - \theta^{(t+1)} - \mathbf{r}$

**Accelerated update**:

$$
\theta^{(t+3)} = \theta^{(t)} - 2\alpha \mathbf{r} + \alpha^2 \mathbf{v}
$$

where $\alpha = -\|\mathbf{r}\| / \|\mathbf{v}\|$.

### Quasi-Newton Acceleration

Use approximate second-order information:

$$
\theta^{(t+1)} = \theta^{(t)} - \mathbf{H}^{-1} \nabla_\theta \ell(\theta^{(t)})
$$

where $\mathbf{H}$ is approximated using BFGS or L-BFGS updates.

**Louis's identity** provides the gradient:

$$
\nabla_\theta \ell(\theta) = \nabla_\theta Q(\theta | \theta) - \nabla_{\theta'} Q(\theta | \theta')\big|_{\theta' = \theta}
$$

At a fixed point of EM, $\nabla_\theta Q(\theta^* | \theta^*) = 0$, so this simplifies.

---

## Comparison of EM Variants

| Variant | E-Step | M-Step | Key Advantage |
|---------|--------|--------|---------------|
| **Standard EM** | Exact | Exact maximization | Simplicity, monotonic improvement |
| **GEM** | Exact | Improvement only | Handles non-closed-form M-step |
| **ECM** | Exact | Conditional maximization | Decouples complex M-steps |
| **ECME** | Exact | Mixed Q/L maximization | Faster convergence |
| **PX-EM** | Exact | Expanded space | Accelerated convergence |
| **MCEM** | Monte Carlo | Exact or approximate | Intractable E-step |
| **SEM** | Stochastic sample | Complete-data MLE | Exploration, escape local optima |
| **Variational EM** | Variational approximation | Exact | Complex posteriors |
| **Online EM** | Incremental | Stochastic update | Large/streaming data |

### Selection Guidelines

| Scenario | Recommended Variant |
|----------|-------------------|
| Closed-form E and M steps | Standard EM |
| No closed-form M-step | GEM or ECM |
| Complex parameter coupling | ECM or ECME |
| Slow convergence | PX-EM, SQUAREM, or quasi-Newton |
| Intractable posterior | MCEM or Variational EM |
| Multimodal likelihood | SEM for exploration, then EM |
| Large dataset | Online or Incremental EM |
| Streaming data | Online EM |

---

## PyTorch Implementation: Variational EM for GMM

```python
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Dirichlet, Categorical

class VariationalGMM:
    """
    Variational EM for Gaussian Mixture Models.
    
    Uses mean-field approximation with Dirichlet prior on mixing weights
    and Normal-Wishart prior on component parameters.
    """
    
    def __init__(self, n_components: int, n_features: int, 
                 alpha_0: float = 1.0, beta_0: float = 1.0,
                 nu_0: float = None, reg_covar: float = 1e-6):
        """
        Args:
            n_components: Number of mixture components
            n_features: Data dimensionality
            alpha_0: Dirichlet concentration parameter
            beta_0: Prior precision scaling
            nu_0: Wishart degrees of freedom (default: n_features)
            reg_covar: Covariance regularization
        """
        self.K = n_components
        self.d = n_features
        self.reg_covar = reg_covar
        
        # Prior hyperparameters
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0 if nu_0 is not None else float(n_features)
        
        # Variational parameters (initialized in fit)
        self.alpha_ = None      # Dirichlet parameters for mixing weights
        self.beta_ = None       # Precision scaling
        self.m_ = None          # Mean parameters
        self.W_ = None          # Wishart scale matrices
        self.nu_ = None         # Wishart degrees of freedom
        
    def _initialize(self, X: torch.Tensor):
        """Initialize variational parameters."""
        N, d = X.shape
        
        # Initialize means with K-means
        indices = torch.randperm(N)[:self.K]
        self.m_ = X[indices].clone()
        
        # Initialize other parameters
        self.alpha_ = torch.ones(self.K) * self.alpha_0 + N / self.K
        self.beta_ = torch.ones(self.K) * self.beta_0 + N / self.K
        self.nu_ = torch.ones(self.K) * self.nu_0 + N / self.K
        self.W_ = torch.stack([torch.eye(d) for _ in range(self.K)])
        
    def _compute_responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """Variational E-step: compute expected responsibilities."""
        N = X.shape[0]
        
        # Expected log mixing weights: E[log π_k]
        log_pi = torch.digamma(self.alpha_) - torch.digamma(self.alpha_.sum())
        
        # Expected log precision determinant: E[log |Λ_k|]
        log_det_Lambda = torch.zeros(self.K)
        for k in range(self.K):
            log_det_Lambda[k] = (
                self.d * torch.log(torch.tensor(2.0)) +
                torch.logdet(self.W_[k]) +
                sum(torch.digamma((self.nu_[k] + 1 - i) / 2) for i in range(1, self.d + 1))
            )
        
        # Expected Mahalanobis distance: E[(x - μ_k)^T Λ_k (x - μ_k)]
        log_rho = torch.zeros(N, self.K)
        for k in range(self.K):
            diff = X - self.m_[k]
            E_Lambda = self.nu_[k] * self.W_[k]
            mahal = self.d / self.beta_[k] + self.nu_[k] * (diff @ self.W_[k] * diff).sum(dim=1)
            
            log_rho[:, k] = log_pi[k] + 0.5 * log_det_Lambda[k] - 0.5 * self.d * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * mahal
        
        # Normalize
        log_rho_norm = torch.logsumexp(log_rho, dim=1, keepdim=True)
        return torch.exp(log_rho - log_rho_norm)
    
    def _update_parameters(self, X: torch.Tensor, r: torch.Tensor):
        """Variational M-step: update variational parameters."""
        N = X.shape[0]
        
        # Sufficient statistics
        N_k = r.sum(dim=0) + 1e-10
        x_bar = (r.T @ X) / N_k.unsqueeze(1)
        
        # Update parameters
        self.alpha_ = self.alpha_0 + N_k
        self.beta_ = self.beta_0 + N_k
        self.nu_ = self.nu_0 + N_k
        
        # Mean prior (assume zero)
        m_0 = torch.zeros(self.d)
        self.m_ = (self.beta_0 * m_0 + N_k.unsqueeze(1) * x_bar) / self.beta_.unsqueeze(1)
        
        # Wishart scale matrix
        W_0_inv = torch.eye(self.d)
        for k in range(self.K):
            diff = X - x_bar[k]
            S_k = (r[:, k].unsqueeze(1) * diff).T @ diff
            
            diff_m = x_bar[k] - m_0
            W_k_inv = (
                W_0_inv + S_k + 
                (self.beta_0 * N_k[k]) / (self.beta_0 + N_k[k]) * 
                torch.outer(diff_m, diff_m)
            )
            self.W_[k] = torch.inverse(W_k_inv + self.reg_covar * torch.eye(self.d))
    
    def _compute_elbo(self, X: torch.Tensor, r: torch.Tensor) -> float:
        """Compute evidence lower bound."""
        N = X.shape[0]
        N_k = r.sum(dim=0) + 1e-10
        
        # This is a simplified ELBO computation
        # Full version includes KL divergences for all variational distributions
        
        # Expected log likelihood
        E_log_lik = 0.0
        for k in range(self.K):
            diff = X - self.m_[k]
            E_Lambda = self.nu_[k] * self.W_[k]
            mahal = self.d / self.beta_[k] + self.nu_[k] * (diff @ self.W_[k] * diff).sum(dim=1)
            
            log_det = self.d * torch.log(torch.tensor(2.0)) + torch.logdet(self.W_[k])
            log_det += sum(torch.digamma((self.nu_[k] + 1 - i) / 2) for i in range(1, self.d + 1))
            
            E_log_lik += (r[:, k] * (0.5 * log_det - 0.5 * self.d * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * mahal)).sum()
        
        # Entropy of responsibilities
        H_r = -(r * torch.log(r + 1e-10)).sum()
        
        return (E_log_lik + H_r).item()
    
    def fit(self, X: torch.Tensor, max_iter: int = 100, tol: float = 1e-4,
            verbose: bool = False) -> 'VariationalGMM':
        """Fit variational GMM using variational EM."""
        self._initialize(X)
        
        prev_elbo = float('-inf')
        
        for iteration in range(max_iter):
            # VE-step
            r = self._compute_responsibilities(X)
            
            # M-step
            self._update_parameters(X, r)
            
            # Compute ELBO
            elbo = self._compute_elbo(X, r)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: ELBO = {elbo:.4f}")
            
            if abs(elbo - prev_elbo) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_elbo = elbo
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster assignments."""
        r = self._compute_responsibilities(X)
        return r.argmax(dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster probabilities."""
        return self._compute_responsibilities(X)
```

---

## Summary

| Aspect | Key Points |
|--------|------------|
| **GEM** | Relax M-step to improvement only; enables gradient-based updates |
| **ECM/ECME** | Sequential conditional maximization; decouples complex M-steps |
| **PX-EM** | Expand parameter space for faster convergence |
| **MCEM** | Monte Carlo approximation for intractable E-steps |
| **SEM** | Single stochastic sample; aids exploration of multimodal surfaces |
| **Variational EM** | Approximate posterior for complex models |
| **Online EM** | Stochastic updates for large/streaming data |
| **Acceleration** | Aitken, SQUAREM, quasi-Newton methods speed convergence |

The flexibility of the EM framework—separating inference (E-step) from optimization (M-step)—enables these diverse extensions. Understanding when and how to apply each variant is essential for tackling real-world latent variable models.

### Key References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1), 1-38.
- Meng, X. L., & Rubin, D. B. (1993). Maximum likelihood estimation via the ECM algorithm. *Biometrika*, 80(2), 267-278.
- Liu, C., & Rubin, D. B. (1994). The ECME algorithm: A simple extension of EM and ECM with faster monotone convergence. *Biometrika*, 81(4), 633-648.
- Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the EM algorithm. *JASA*, 85(411), 699-704.
- Celeux, G., & Diebolt, J. (1985). The SEM algorithm: A probabilistic teacher algorithm derived from the EM algorithm. *Computational Statistics Quarterly*, 2, 73-82.
- Cappé, O., & Moulines, E. (2009). On-line expectation–maximization algorithm for latent data models. *JRSS-B*, 71(3), 593-613.
