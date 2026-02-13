# Deterministic Annealing for EM

Deterministic annealing applies the temperature concept to the Expectation-Maximization algorithm, providing a principled way to escape local optima in likelihood optimization. Unlike stochastic simulated annealing, this approach works with expected values directly, making it suitable for problems where expectations can be computed analytically.

---

## Motivation: The Local Optima Problem in EM

### EM and Local Maxima

The standard EM algorithm maximizes the log-likelihood:

$$
\ell(\theta) = \log p(\mathbf{X} | \theta) = \log \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}
$$

EM iterates between:

- **E-step**: Compute $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$
- **M-step**: Maximize $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$

**Problem**: EM only guarantees convergence to a *local* maximum. The final solution depends on initialization.

### Why Local Optima Occur

For mixture models, local optima arise from discrete assignment structure (each data point "belongs" to one component), symmetry breaking (multiple equivalent solutions exist), and multimodality (the likelihood surface has many peaks).

---

## The Annealing Idea

### Temperature-Scaled Distributions

Introduce inverse temperature $\beta \in [0, 1]$ and define the **tempered posterior**:

$$
q_\beta(\mathbf{Z} | \mathbf{X}, \theta) \propto p(\mathbf{X}, \mathbf{Z} | \theta)^\beta
$$

At different temperatures:

| $\beta$ | Temperature | Posterior Behavior |
|---------|-------------|-------------------|
| $\beta \to 0$ | $T \to \infty$ | Uniform over latent states |
| $\beta = 1$ | $T = 1$ | True posterior (standard EM) |
| $\beta \to \infty$ | $T \to 0$ | Concentrated on MAP assignment |

### How Temperature Smooths the Landscape

At high temperature ($\beta \to 0$), all latent configurations are equally likely, the effective likelihood becomes smooth (often convex), and there is typically a unique optimum. At low temperature ($\beta \to 1$), the true multimodal structure emerges and local optima appear—but the algorithm is already in a good basin.

Starting from $\beta = 0$ (easy problem) and gradually increasing to $\beta = 1$ (true problem), we trace a path through parameter space that avoids local optima.

---

## Mathematical Framework

### The Annealed Free Energy

Define the **annealed log-likelihood** at inverse temperature $\beta$:

$$
\ell_\beta(\theta) = \frac{1}{\beta} \log \int p(\mathbf{X}, \mathbf{Z} | \theta)^\beta \, d\mathbf{Z}
$$

**Properties**:

- $\lim_{\beta \to 1} \ell_\beta(\theta) = \ell(\theta)$ (recovers true log-likelihood)
- $\lim_{\beta \to 0} \ell_\beta(\theta) = \mathbb{E}_{\text{uniform}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$ (average over latent space)
- $\ell_\beta(\theta)$ varies continuously in $\beta$

### The Annealed ELBO

The variational lower bound at temperature $\beta$:

$$
\mathcal{L}_\beta(q, \theta) = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} | \theta)] - \frac{1}{\beta} H[q]
$$

where $H[q] = -\mathbb{E}_q[\log q(\mathbf{Z})]$ is the entropy.

**Interpretation**: Temperature scales the entropy term. High $T$ (low $\beta$) gives strong entropy regularization producing a spread-out $q$; low $T$ (high $\beta$) gives weak entropy regularization producing a concentrated $q$.

### Connection to Free Energy

In statistical physics:

$$
F_\beta = U - T S = \mathbb{E}_q[E] - \frac{1}{\beta} H[q]
$$

where $U$ is internal energy and $S$ is entropy. The annealed EM minimizes free energy at each temperature.

---

## The Annealed EM Algorithm

### Algorithm Structure

```
Input: Data X, number of components K, 
       temperature schedule β₁ < β₂ < ... < βₘ = 1

Initialize θ randomly

for β in [β₁, β₂, ..., βₘ]:
    repeat until convergence:
        # Annealed E-step
        q_β(Z) ∝ p(X, Z | θ)^β
        
        # M-step (unchanged)
        θ = argmax_θ E_q[log p(X, Z | θ)]
    
return θ
```

### Annealed E-Step

The key modification is in the E-step. For the standard posterior:

$$
p(z_n = k | \mathbf{x}_n, \theta) = \frac{\pi_k p(\mathbf{x}_n | \theta_k)}{\sum_j \pi_j p(\mathbf{x}_n | \theta_j)}
$$

The **annealed posterior** (responsibilities):

$$
r_{nk}^\beta = \frac{[\pi_k p(\mathbf{x}_n | \theta_k)]^\beta}{\sum_j [\pi_j p(\mathbf{x}_n | \theta_j)]^\beta}
$$

### Temperature Effects on Responsibilities

**At $\beta \to 0$** (high temperature): $r_{nk}^\beta \to 1/K$ — all components share responsibility equally.

**At $\beta = 1$** (normal temperature): $r_{nk}^1 = r_{nk}$ — standard EM responsibilities.

**At $\beta \to \infty$** (zero temperature):

$$
r_{nk}^\beta \to \begin{cases} 1 & k = \arg\max_j \pi_j p(\mathbf{x}_n | \theta_j) \\ 0 & \text{otherwise} \end{cases}
$$

Hard assignment (k-means like).

---

## Example: Gaussian Mixture Models

### Annealed E-Step for GMM

```python
def annealed_e_step(X, pi, mu, Sigma, beta):
    """Compute annealed responsibilities."""
    N, D = X.shape
    K = len(pi)
    
    # Compute log-likelihoods
    log_resp = np.zeros((N, K))
    for k in range(K):
        log_resp[:, k] = (np.log(pi[k]) + 
                         multivariate_normal.logpdf(X, mu[k], Sigma[k]))
    
    # Apply temperature
    log_resp_beta = beta * log_resp
    
    # Normalize (softmax)
    log_resp_beta -= logsumexp(log_resp_beta, axis=1, keepdims=True)
    resp = np.exp(log_resp_beta)
    
    return resp
```

### M-Step (Unchanged)

The M-step uses the annealed responsibilities but is otherwise standard:

```python
def m_step(X, resp):
    """Standard M-step using responsibilities."""
    N, D = X.shape
    K = resp.shape[1]
    
    N_k = resp.sum(axis=0)
    pi = N_k / N
    mu = np.array([resp[:, k] @ X / N_k[k] for k in range(K)])
    
    Sigma = []
    for k in range(K):
        diff = X - mu[k]
        Sigma_k = (resp[:, k:k+1] * diff).T @ diff / N_k[k]
        Sigma.append(Sigma_k)
    
    return pi, mu, np.array(Sigma)
```

### Complete Annealed EM for GMM

```python
def annealed_em_gmm(X, K, betas=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    max_iter=100, tol=1e-6):
    """Annealed EM for Gaussian Mixture Model."""
    N, D = X.shape
    
    # Random initialization
    pi = np.ones(K) / K
    indices = np.random.choice(N, K, replace=False)
    mu = X[indices].copy()
    Sigma = np.array([np.eye(D) for _ in range(K)])
    
    for beta in betas:
        prev_ll = -np.inf
        for iteration in range(max_iter):
            # Annealed E-step
            resp = annealed_e_step(X, pi, mu, Sigma, beta)
            
            # M-step
            pi, mu, Sigma = m_step(X, resp)
            
            # Check convergence
            ll = compute_log_likelihood(X, pi, mu, Sigma)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
    
    return pi, mu, Sigma
```

---

## Theoretical Analysis

### Bifurcation Theory

As $\beta$ increases, the parameter space undergoes **bifurcations**—points where the number or nature of critical points changes.

At low $\beta$, there is typically a single optimum. At critical values of $\beta$, the optimum splits into multiple optima (a pitchfork bifurcation). The algorithm follows one branch, determined by the data.

### Homotopy Continuation

Annealed EM can be viewed as **homotopy continuation**: a continuous deformation from an easy problem ($\beta \to 0$, convex landscape) to the hard problem ($\beta = 1$, multimodal landscape).

The path $\beta \mapsto \theta^*(\beta)$ traces a continuous curve through parameter space. If the path has no discontinuities, we reach the global optimum.

### Phase Transitions

At critical temperatures, the nature of the solution changes qualitatively:

- **Cluster emergence**: As $\beta$ increases, uniform responsibilities split into distinct clusters
- **Symmetry breaking**: Equivalent components become distinguishable
- **First-order transitions**: Abrupt jumps in parameters at specific $\beta$ values

---

## Temperature Schedule Design

### Linear Schedule

$$
\beta_m = \frac{m}{M}, \quad m = 1, 2, \ldots, M
$$

Simple but may miss important phase transitions.

### Geometric Schedule

$$
\beta_m = \beta_0 \cdot r^{m-1}, \quad r = (1/\beta_0)^{1/(M-1)}
$$

Finer resolution at low $\beta$ where transitions often occur.

### Adaptive Schedule

```python
def adaptive_annealed_em(X, K, beta_init=0.01, max_beta_steps=50):
    """Annealed EM with adaptive temperature schedule."""
    theta = random_init(X, K)
    beta = beta_init
    beta_increment = 0.05
    
    while beta < 1.0:
        theta, converged_iters = em_at_temperature(X, theta, beta)
        
        # Adapt temperature increment
        if converged_iters < 5:
            beta_increment *= 1.5  # Converged quickly, increase faster
        elif converged_iters > 20:
            beta_increment *= 0.7  # Slow convergence, increase more slowly
        
        beta = min(beta + beta_increment, 1.0)
    
    return theta
```

---

## Extensions and Variants

### Deterministic Annealing for Other Models

**Hidden Markov Models**: Anneal the forward-backward responsibilities, using $\alpha_t(k)^\beta$ and $\beta_t(k)^\beta$ in scaled form.

**Latent Dirichlet Allocation**: Anneal the topic assignments, normalizing $q(z_{dn} = k)^\beta$.

**Variational Autoencoders**: Anneal the KL divergence term (β-VAE connection):

$$
\mathcal{L} = \mathbb{E}[\log p(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}[q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
$$

### Connection to β-VAE

The β-VAE objective:

$$
\mathcal{L}_\beta = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}[q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
$$

can be viewed as annealed variational inference: $\beta < 1$ encourages more diverse latent representations, $\beta = 1$ gives the standard VAE, and $\beta > 1$ encourages disentanglement.

### Mean Field Annealing

For mean field variational inference with factorized $q(\mathbf{Z}) = \prod_i q_i(z_i)$:

**Standard update**: $\log q_i(z_i) = \mathbb{E}_{q_{-i}}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$

**Annealed update**: $\log q_i(z_i) = \beta \cdot \mathbb{E}_{q_{-i}}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$

---

## Practical Considerations

### Initialization at High Temperature

At $\beta \to 0$, the solution is insensitive to initialization. Any reasonable starting point works:

```python
def initialize_for_annealing(X, K):
    """Initialization doesn't matter much at high temperature."""
    N, D = X.shape
    pi = np.ones(K) / K
    mu = X[np.random.choice(N, K, replace=False)]
    Sigma = np.array([np.cov(X.T) for _ in range(K)])
    return pi, mu, Sigma
```

### Numerical Stability

At extreme temperatures, numerical issues can arise:

```python
def stable_annealed_responsibilities(log_prob, beta, eps=1e-10):
    """Numerically stable annealed responsibilities."""
    log_prob_scaled = beta * log_prob
    log_prob_scaled -= log_prob_scaled.max(axis=1, keepdims=True)
    
    prob = np.exp(log_prob_scaled)
    prob = np.clip(prob, eps, 1-eps)
    prob /= prob.sum(axis=1, keepdims=True)
    
    return prob
```

### Monitoring Convergence

Track the annealed log-likelihood at each temperature:

```python
def annealed_log_likelihood(X, pi, mu, Sigma, beta):
    """Compute log-likelihood at inverse temperature beta."""
    N, K = len(X), len(pi)
    
    log_probs = np.zeros((N, K))
    for k in range(K):
        log_probs[:, k] = (np.log(pi[k]) + 
                          multivariate_normal.logpdf(X, mu[k], Sigma[k]))
    
    return (1/beta) * np.sum(logsumexp(beta * log_probs, axis=1))
```

---

## Comparison with Other Approaches

### Annealed EM vs. Random Restarts

| Aspect | Annealed EM | Random Restarts |
|--------|-------------|-----------------|
| Runs | Single run | Multiple runs |
| Exploration | Systematic via temperature | Random via initialization |
| Computation | One long run | Many short runs |
| Theory | Homotopy continuation | Best of independent trials |
| Best for | Smooth landscapes | Rugged landscapes |

### Annealed EM vs. Simulated Annealing

| Aspect | Annealed EM | Simulated Annealing |
|--------|-------------|---------------------|
| Nature | Deterministic | Stochastic |
| Updates | Expected values | Sample-based |
| Per-iteration cost | Higher (full E-step) | Lower (single move) |
| Convergence | Smooth path | Noisy path |
| Applicability | When expectations tractable | General |

### When to Use Annealed EM

**Use annealed EM when**: Expectations can be computed analytically, the landscape has a few deep basins, standard EM is sensitive to initialization, and deterministic reproducible results are desired.

**Consider alternatives when**: Expectations are intractable (use SA or VI), the landscape is extremely rugged, or phase transitions are sharp.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Annealed E-step** | $q_\beta(\mathbf{Z}) \propto p(\mathbf{X}, \mathbf{Z} | \theta)^\beta$ |
| **Temperature effect** | $\beta \to 0$: uniform; $\beta = 1$: true posterior |
| **Landscape smoothing** | High temperature removes local optima |
| **Homotopy path** | Continuous path from easy to hard problem |
| **Schedule design** | Start low, increase gradually to $\beta = 1$ |

Deterministic annealing transforms EM from a local optimizer to a more global one by starting with a smoothed, convex-like problem, gradually revealing the true multimodal structure, and following a continuous path to a good solution.
