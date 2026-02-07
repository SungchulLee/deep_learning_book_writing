# KL Divergence

Kullback-Leibler divergence measures how one probability distribution diverges from another. It appears throughout deep learning as a regularization term in variational autoencoders, a training objective in knowledge distillation, and a theoretical tool for understanding model behavior. This section introduces the definition, core properties, and PyTorch interfaces, with dedicated sub-pages for [distance axioms](kl_distance_axioms.md), [Gaussian computation](kl_gaussian.md), and [Fisher information](kl_fisher_information.md).

## Definition

For discrete distributions $p$ and $q$ over the same sample space:

$$D_{\text{KL}}(p \| q) = \sum_i p_i \log\frac{p_i}{q_i} = \mathbb{E}_{p}\!\left[\log \frac{p_i}{q_i}\right]$$

For continuous distributions with densities $f$ and $g$:

$$D_{\text{KL}}(f \| g) = \int f(x)\log\frac{f(x)}{g(x)}\,dx = \mathbb{E}_{f}\!\left[\log \frac{f(x)}{g(x)}\right]$$

$D_{\text{KL}}(p \| q)$ can be interpreted as the expected number of extra bits needed to encode samples from $p$ using a code optimized for $q$. The convention $0 \log(0/q) = 0$ follows from continuity, and $D_{\text{KL}}$ is undefined when $q_i = 0$ for some $i$ where $p_i > 0$ (the support of $p$ must be contained in the support of $q$).

### Intuitive Interpretations

**Extra bits.** Design a code optimal for $q$ but use it to encode samples from $p$. The KL divergence is the expected number of additional bits required beyond the optimal code for $p$.

**Expected log evidence ratio.** $D_{\text{KL}}(p \| q) = \mathbb{E}_{p}[\log(p(x)/q(x))]$ measures how much more likely samples are under $p$ than under $q$, on average.

**Information gain.** $D_{\text{KL}}(p \| q)$ quantifies the information gained when updating from a prior $q$ to a posterior $p$.

## Non-Negativity (Gibbs' Inequality)

KL divergence is always non-negative. The proof uses Jensen's inequality applied to the concave function $\log$:

$$\begin{aligned}
D_{\text{KL}}(f \| g)
&= -\int f(x)\log\frac{g(x)}{f(x)}\,dx \\
&\geq -\log\int f(x)\frac{g(x)}{f(x)}\,dx \\
&= -\log\int g(x)\,dx \\
&= -\log 1 = 0
\end{aligned}$$

Equality holds if and only if $f = g$ almost everywhere. This result, known as **Gibbs' inequality**, guarantees that the divergence is minimized (at zero) when the two distributions are identical.

## Asymmetry

KL divergence is **not** symmetric:

$$D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p) \quad\text{in general}$$

```python
import numpy as np

np.random.seed(2)

p = np.random.uniform(0., 1., 3)
q = np.random.uniform(0., 1., 3)
p, q = p / p.sum(), q / q.sum()

KL_pq = np.sum(p * np.log(p / q))
KL_qp = np.sum(q * np.log(q / p))
print(f"KL(p||q) = {KL_pq:.6f}")
print(f"KL(q||p) = {KL_qp:.6f}")
print(f"Difference: {abs(KL_pq - KL_qp):.6f}")  # non-zero
```

### Additivity for Independent Distributions

If $p(x, y) = p(x)p(y)$ and $q(x, y) = q(x)q(y)$:

$$D_{\text{KL}}(p(x,y) \| q(x,y)) = D_{\text{KL}}(p(x) \| q(x)) + D_{\text{KL}}(p(y) \| q(y))$$

## Relationship to Cross-Entropy

Cross-entropy and KL divergence are related by:

$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

where $H(P) = -\sum_k P(k)\log P(k)$ is the entropy of $P$. Since $H(P)$ is constant with respect to $Q$, minimizing cross-entropy $H(P, Q)$ is equivalent to minimizing $D_{\text{KL}}(P \| Q)$. This is why cross-entropy loss and KL minimization are interchangeable as classification objectives.

## Forward vs Reverse KL

The asymmetry has critical consequences for how KL divergence is used in practice.

### Forward KL: $D_{\text{KL}}(p \| q)$

Minimizing forward KL over $q$ gives:

$$\min_q D_{\text{KL}}(p \| q) = \min_q \mathbb{E}_p[-\log q(x)] + \text{const}$$

This is **mean-seeking** (also called **mode-covering**): $q$ spreads out to cover all modes of $p$, penalizing $q$ for missing any mode where $p$ has mass.

**Typical use:** Maximum likelihood estimation (we know $p$ from data and optimize $q$).

### Reverse KL: $D_{\text{KL}}(q \| p)$

Minimizing reverse KL over $q$ gives:

$$\min_q D_{\text{KL}}(q \| p) = \min_q \mathbb{E}_q[\log q(x) - \log p(x)]$$

This is **mode-seeking**: $q$ concentrates on a single mode of $p$ rather than spreading across all modes. $q$ avoids placing mass where $p$ is small because those regions incur large $\log q(x) - \log p(x)$ penalties.

**Typical use:** Variational inference (we can evaluate $\log p$ up to a constant but cannot sample from $p$).

## Not a Metric

Despite being called a "divergence," KL divergence is not a distance metric. It satisfies non-negativity and identity of indiscernibles but violates symmetry and the triangle inequality. The detailed analysis with proofs and numerical counterexamples is in [KL Divergence and Distance Axioms](kl_distance_axioms.md).

## Local Behavior: Fisher Information

A Taylor expansion of $D_{\text{KL}}(f_{\theta_0} \| f_\theta)$ around $\theta = \theta_0$ reveals that locally, KL divergence behaves like a quadratic form:

$$D_{\text{KL}}(f_{\theta_0} \| f_\theta) \approx \frac{1}{2}(\theta - \theta_0)^T\, \mathbf{I}(\theta_0)\, (\theta - \theta_0)$$

where $\mathbf{I}(\theta_0)$ is the **Fisher information matrix**. Near the minimum, KL divergence behaves like a Mahalanobis distance, providing a natural Riemannian metric on the space of probability distributions. This underpins natural gradient methods, trust region optimization (TRPO, PPO), and variational inference. The full derivation is in [KL and Fisher Information](kl_fisher_information.md).

## KL Divergence for Gaussians

### Univariate Formula

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### General Multivariate Formula

For $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q, \Sigma_q)$ in $\mathbb{R}^d$:

$$D_{\text{KL}}(p \| q) = \frac{1}{2}\!\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + \operatorname{tr}\!\left(\Sigma_q^{-1}\Sigma_p\right) + (\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)\right]$$

### VAE Special Case

For encoder $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$ and prior $p(z) = \mathcal{N}(0, I)$:

$$D_{\text{KL}}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\!\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

The complete derivation with the full-covariance case is in [KL for Gaussians](kl_gaussian.md).

## Application in VAEs

In a Variational Autoencoder, the loss combines reconstruction error with KL regularization. Starting from the ELBO decomposition:

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\mathcal{L}_{\text{ELBO}}} + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))$$

Maximizing the ELBO is equivalent to minimizing:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

### The KL–Reconstruction Trade-off

The $\beta$-VAE introduces a weight on the KL term: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$.

| KL value | Meaning | Effect |
|----------|---------|--------|
| **High** | Encoder outputs far from $\mathcal{N}(0, I)$ | More information encoded; better reconstruction |
| **Low** | Encoder outputs close to $\mathcal{N}(0, I)$ | Less information encoded; smoother latent space |
| **Zero** | All inputs map to prior | No information encoded; random outputs |

## PyTorch Implementation

### Gaussian KL for VAEs

The encoder outputs $\mu$ and $\log\sigma^2$ for each latent dimension:

```python
import torch

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor,
                  reduction: str = 'sum') -> torch.Tensor:
    """KL divergence from q = N(mu, diag(exp(logvar))) to p = N(0, I).

    Formula: D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of q, shape (batch_size, latent_dim).
        logvar: Log variance of q, shape (batch_size, latent_dim).
        reduction: 'sum', 'mean', or 'none'.

    Returns:
        KL divergence with specified reduction.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == 'sum':
        return kl.sum()
    elif reduction == 'mean':
        return kl.mean()
    else:  # 'none' — sum over latent dims, keep batch
        return kl.sum(dim=1)
```

!!! tip "Why Log-Variance?"
    The encoder outputs $\log\sigma^2$ rather than $\sigma^2$ directly for numerical stability: `logvar` can be any real number (valid neural network output), `exp(logvar)` is always positive (valid variance), and `log(sigma^2) = logvar` avoids taking the log of small numbers.

### Full VAE Loss

```python
import torch.nn.functional as F

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple:
    """Complete VAE loss: Reconstruction + beta * KL.

    Args:
        recon_x: Reconstructed data, shape (batch_size, data_dim).
        x: Original data, shape (batch_size, data_dim).
        mu: Encoder mean, shape (batch_size, latent_dim).
        logvar: Encoder log-variance, shape (batch_size, latent_dim).
        beta: KL weight (beta-VAE).

    Returns:
        (total_loss, recon_loss, kl_loss) tuple.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
```

### Practical Techniques

**Free bits.** Prevent posterior collapse by enforcing a minimum KL per latent dimension:

```python
def kl_free_bits(mu, logvar, free_bits=0.1):
    """KL with free bits constraint."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1)
```

**KL annealing.** Gradually increase the KL weight during training:

```python
def get_beta(epoch, warmup_epochs=10, max_beta=1.0):
    """Linear KL annealing."""
    if epoch < warmup_epochs:
        return max_beta * epoch / warmup_epochs
    return max_beta
```

### `nn.KLDivLoss` for General Distributions

For non-Gaussian distributions (e.g., knowledge distillation), PyTorch provides `nn.KLDivLoss`. It expects **log-probabilities** as input and **probabilities** as target:

```python
import torch.nn as nn

kl_criterion = nn.KLDivLoss(reduction='batchmean')

# input: log-probabilities from the student model
log_probs = F.log_softmax(logits, dim=1)

# target: probability distribution from a teacher model
target_probs = F.softmax(teacher_logits, dim=1)

loss = kl_criterion(log_probs, target_probs)
```

!!! warning "Input Convention"
    `nn.KLDivLoss` expects the **input** in log-space and the **target** in probability space. This convention is opposite to what many users expect. Swapping them produces incorrect results silently.

The `reduction` parameter controls aggregation: `'batchmean'` (recommended, gives true per-sample KL), `'sum'` (raw sum), `'mean'` (divides by total elements—**not** true KL), and `'none'` (per-element).

### Analytical vs Monte Carlo KL

For Gaussian encoder and Gaussian prior, the analytical formula is exact and variance-free. For non-Gaussian posteriors or complex priors, KL must be estimated via Monte Carlo:

```python
def kl_monte_carlo(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Estimate D_KL(q||p) from samples z ~ q.

    D_KL = E_q[log q - log p] ≈ (1/N) sum_i (log q(z_i) - log p(z_i))
    """
    return (log_q - log_p).mean()
```

## Key Takeaways

KL divergence is the expected log-likelihood ratio between two distributions, measuring information loss when approximating $p$ with $q$. It is non-negative (Gibbs' inequality), asymmetric, and not a metric. Forward KL ($D_{\text{KL}}(p \| q)$) is mode-covering and underlies maximum likelihood; reverse KL ($D_{\text{KL}}(q \| p)$) is mode-seeking and underlies variational inference. For Gaussians, closed-form expressions enable efficient computation in VAEs. PyTorch provides `nn.KLDivLoss` for general discrete distributions (with the input-in-log-space convention) and the VAE KL term is implemented directly from the analytical formula.
