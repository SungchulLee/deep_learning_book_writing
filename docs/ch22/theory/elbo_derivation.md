# ELBO Derivation

Deriving the Evidence Lower Bound from first principles using multiple approaches.

---

## Learning Objectives

By the end of this section, you will be able to:

- Derive the ELBO using Jensen's inequality
- Derive the ELBO via the KL divergence decomposition
- Explain the fundamental identity connecting ELBO, log-likelihood, and posterior approximation gap
- Decompose the ELBO into its information-theoretic components

---

## The Problem: Intractable Marginal Likelihood

### Maximum Likelihood Objective

We want to learn a generative model $p_\theta(x)$ by maximizing the log-likelihood of observed data:

$$\max_\theta \mathbb{E}_{p_{\text{data}}(x)}[\log p_\theta(x)]$$

For a latent variable model, the marginal likelihood involves integrating over all latent codes:

$$p_\theta(x) = \int p_\theta(x|z) p(z) dz$$

### Why This Is Intractable

This integral requires evaluating the decoder $p_\theta(x|z)$ — parameterized by a neural network — at every possible $z$. For a latent space of dimension $d$, this is an integral over $\mathbb{R}^d$ with no closed-form solution. Monte Carlo estimation of $\log \int p_\theta(x|z) p(z) dz$ is also problematic because the log of an expectation cannot be efficiently estimated by sampling.

---

## Derivation 1: Jensen's Inequality

### Step-by-Step

Starting from the log marginal likelihood, we introduce the variational distribution $q_\phi(z|x)$:

$$\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int \frac{p_\theta(x, z)}{q_\phi(z|x)} q_\phi(z|x) dz$$

$$= \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

Applying Jensen's inequality ($\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ since $\log$ is concave):

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

### Expanding the Bound

$$\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p(z)}{q_\phi(z|x)}\right]$$

$$= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL divergence term}}$$

This is the **Evidence Lower Bound (ELBO)**:

$$\boxed{\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))}$$

---

## Derivation 2: KL Divergence Decomposition

### Starting from KL Divergence

Consider the KL divergence from the approximate posterior to the true posterior:

$$D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]$$

Applying Bayes' theorem: $p_\theta(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)}$:

$$= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x) \cdot p_\theta(x)}{p_\theta(x|z)p(z)}\right]$$

$$= \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)] - \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathbb{E}_{q_\phi(z|x)}[\log p(z)] + \log p_\theta(x)$$

$$= -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) \| p(z)) + \log p_\theta(x)$$

### Rearranging

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))}_{\mathcal{L}(\theta, \phi; x) \text{ (ELBO)}} + \underbrace{D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}_{\geq 0}$$

This is the **fundamental identity** of variational inference:

$$\boxed{\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}$$

### Key Insights

Since KL divergence is non-negative, $\mathcal{L} \leq \log p_\theta(x)$ always holds. The bound is tight when $q_\phi(z|x) = p_\theta(z|x)$, i.e., when the approximate posterior equals the true posterior. Maximizing the ELBO simultaneously improves the generative model (increasing $\log p_\theta(x)$) and tightens the approximation (decreasing $D_{KL}(q \| p_\theta(z|x))$).

---

## ELBO Components

### Reconstruction Term

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

This term measures how well the decoder reconstructs data from latent codes sampled via the encoder. Maximizing it encourages high reconstruction fidelity. In practice, this becomes a reconstruction loss (BCE or MSE) evaluated at sampled latent codes.

### KL Divergence Term

$$D_{KL}(q_\phi(z|x) \| p(z))$$

This term regularizes the encoder's output distribution to remain close to the prior. It prevents the encoder from placing all probability mass on a single point (which would reduce to a standard autoencoder) and ensures the latent space has the structure needed for generation.

### The Trade-off

These two terms create a fundamental tension:

- **Low reconstruction loss** requires $q_\phi(z|x)$ to be informative about $x$ (high mutual information $I(X; Z)$)
- **Low KL divergence** requires $q_\phi(z|x)$ to be close to $p(z) = \mathcal{N}(0, I)$ (low mutual information)

The ELBO optimum balances these competing objectives.

---

## Information-Theoretic Decomposition

### ELBO as Rate-Distortion

The ELBO can be interpreted through the lens of rate-distortion theory:

$$\mathcal{L} = \underbrace{-\mathbb{E}_q[-\log p(x|z)]}_{\text{Distortion (reconstruction error)}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{Rate (information transmitted)}}$$

The **distortion** measures how much information is lost during encoding, while the **rate** measures how much information the latent code carries about the input.

### Decomposing the KL Term

The expected KL over the data distribution decomposes as:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = \underbrace{I_q(X; Z)}_{\text{Mutual information}} + \underbrace{D_{KL}(q_\phi(z) \| p(z))}_{\text{Marginal KL}}$$

where $q_\phi(z) = \mathbb{E}_{p_{\text{data}}(x)}[q_\phi(z|x)]$ is the aggregated posterior.

The first term measures how much information about $x$ is encoded in $z$, and the second measures how well the aggregated posterior matches the prior. This decomposition is central to understanding β-VAE and disentanglement.

---

## Practical Implementation

### Monte Carlo Estimation

In practice, we estimate the ELBO using a single sample from $q_\phi(z|x)$:

$$\mathcal{L} \approx \log p_\theta(x|z) - D_{KL}(q_\phi(z|x) \| p(z)), \quad z \sim q_\phi(z|x)$$

The reconstruction term is estimated by sampling (enabled by the reparameterization trick), while the KL term has a closed-form solution for Gaussian distributions.

### PyTorch Loss Function

```python
import torch
import torch.nn.functional as F

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Negative ELBO loss for VAE training.
    
    Args:
        recon_x: Decoder output [batch_size, data_dim]
        x: Original data [batch_size, data_dim]
        mu: Encoder mean [batch_size, latent_dim]
        logvar: Encoder log-variance [batch_size, latent_dim]
        beta: KL weight (β=1 for standard VAE)
    
    Returns:
        Negative ELBO (to minimize), reconstruction loss, KL loss
    """
    # Reconstruction: -E_q[log p(x|z)]
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL: D_KL(q(z|x) || p(z)) for Gaussian q and standard normal p
    # = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Negative ELBO
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss
```

---

## Tighter Bounds: Importance Weighted ELBO

### IWAE Bound

The Importance Weighted Autoencoder (IWAE) provides a tighter bound using $K$ samples:

$$\mathcal{L}_K = \mathbb{E}_{z_1, \ldots, z_K \sim q(z|x)}\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}\right]$$

This satisfies $\mathcal{L}_1 \leq \mathcal{L}_K \leq \log p_\theta(x)$, with equality as $K \to \infty$.

```python
def iwae_loss(model, x, K=5):
    """Importance-weighted ELBO with K samples."""
    batch_size = x.size(0)
    
    # Encode once
    mu, logvar = model.encode(x)
    
    # Sample K latent vectors
    log_w = []
    for _ in range(K):
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z)
        
        # log p(x|z)
        log_p_x_z = -F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        # log p(z)
        log_p_z = -0.5 * z.pow(2).sum(dim=1)
        # log q(z|x)
        log_q_z_x = -0.5 * ((z - mu).pow(2) / logvar.exp() + logvar).sum(dim=1)
        
        log_w.append(log_p_x_z + log_p_z - log_q_z_x)
    
    # Log-sum-exp for numerical stability
    log_w = torch.stack(log_w, dim=0)  # [K, batch_size]
    iwae_elbo = torch.logsumexp(log_w, dim=0) - torch.log(torch.tensor(K, dtype=torch.float))
    
    return -iwae_elbo.mean()
```

---

## Summary

| Derivation | Approach | Key Result |
|------------|----------|------------|
| **Jensen's inequality** | Lower bound on $\log \mathbb{E}[X]$ | ELBO $\leq \log p(x)$ |
| **KL decomposition** | Decompose posterior gap | $\log p(x) = \text{ELBO} + D_{KL}(q \| p(z\|x))$ |
| **Rate-distortion** | Information-theoretic view | Trade-off between compression and fidelity |
| **IWAE** | Multiple samples | Tighter bound with $K$ importance samples |

---

## Exercises

### Exercise 1: Jensen's Inequality

Show that Jensen's inequality applied to $\log$ gives $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ by using the concavity of $\log$.

### Exercise 2: Tightness

Under what conditions is the ELBO exactly equal to $\log p_\theta(x)$? What does this imply about the expressiveness needed for $q_\phi(z|x)$?

### Exercise 3: IWAE Implementation

Implement the IWAE bound with $K \in \{1, 5, 50\}$ samples and compare the estimated log-likelihood on a test set.

---

## What's Next

The next section covers the [Reparameterization Trick](reparameterization.md), which enables gradient-based optimization of the ELBO despite the stochastic sampling operation.
