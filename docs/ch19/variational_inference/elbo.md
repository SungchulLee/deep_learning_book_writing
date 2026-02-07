# Evidence Lower Bound (ELBO)

The Evidence Lower Bound is the central quantity in variational inference, the EM algorithm, and VAE training. It provides a tractable optimisation objective that lower-bounds the intractable log marginal likelihood, and its gap from the true likelihood has a precise characterisation in terms of KL divergence. This section presents three complementary derivations, analyses the gap and tightness conditions, develops the key alternative formulations, and connects the ELBO to both the EM algorithm and variational autoencoders.

## The Problem: Intractable Marginal Likelihood

In latent-variable models the marginal likelihood (or *evidence*) involves an integral over latent variables $z$ (or parameters $\theta$, depending on the setting):

$$p(\mathbf{X} | \theta) = \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}$$

For neural network decoders or complex likelihood functions this integral is intractable: it cannot be evaluated in closed form or efficiently approximated. The ELBO provides a lower bound that *can* be optimised.

## Derivation 1: Jensen's Inequality

### Jensen's Inequality

For a concave function $\varphi$ (such as $\log$) and any random variable $Y$:

$$\varphi(\mathbb{E}[Y]) \geq \mathbb{E}[\varphi(Y)]$$

with equality if and only if $Y$ is constant almost surely.

!!! note "Proof sketch"
    For convex $\varphi$ there exists a supporting hyperplane at $\mu = \mathbb{E}[X]$ with slope $\alpha$ such that $\varphi(X) \geq \alpha(X - \mu) + \varphi(\mu)$. Taking expectations and noting $\mathbb{E}[X - \mu] = 0$ gives $\mathbb{E}[\varphi(X)] \geq \varphi(\mathbb{E}[X])$. For concave $\varphi$ the inequality reverses.

### Applying Jensen's Inequality

Introduce an arbitrary distribution $q(\mathbf{Z})$ over the latent variables by multiplying and dividing inside the integral:

$$\log p(\mathbf{X} | \theta) = \log \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} \, d\mathbf{Z} = \log \, \mathbb{E}_{q}\!\left[\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right]$$

Since $\log$ is concave, Jensen's inequality gives:

$$\log p(\mathbf{X} | \theta) \geq \mathbb{E}_{q}\!\left[\log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right] \;\equiv\; \mathcal{L}(q, \theta)$$

The right-hand side is the **Evidence Lower Bound (ELBO)**.

### When Equality Holds

Jensen's inequality is tight when the random variable inside the expectation is constant. This requires $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z}) = c$ for all $\mathbf{Z}$ in the support of $q$. Normalising both sides shows $c = p(\mathbf{X} | \theta)$ and therefore:

$$q(\mathbf{Z}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{p(\mathbf{X} | \theta)} = p(\mathbf{Z} | \mathbf{X}, \theta)$$

The bound is tight if and only if $q$ equals the true posterior.

## Derivation 2: KL Divergence Decomposition

Write the KL divergence from $q(\mathbf{Z})$ to the true posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$:

$$D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr) = \mathbb{E}_{q}\!\left[\log \frac{q(\mathbf{Z})}{p(\mathbf{Z} | \mathbf{X}, \theta)}\right]$$

Substitute Bayes' theorem $p(\mathbf{Z} | \mathbf{X}, \theta) = p(\mathbf{X}, \mathbf{Z} | \theta) / p(\mathbf{X} | \theta)$:

$$D_{\text{KL}} = \mathbb{E}_{q}[\log q(\mathbf{Z})] - \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \log p(\mathbf{X} | \theta)$$

Rearranging and recognising the ELBO:

$$\boxed{\log p(\mathbf{X} | \theta) = \mathcal{L}(q, \theta) + D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)}$$

Since $D_{\text{KL}} \geq 0$, we recover $\log p(\mathbf{X} | \theta) \geq \mathcal{L}(q, \theta)$. This decomposition is more informative than Jensen's approach because it identifies the gap exactly.

## Derivation 3: Importance Sampling Perspective

From an importance sampling viewpoint, the marginal likelihood is the expectation of the importance weight $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z})$:

$$p(\mathbf{X} | \theta) = \mathbb{E}_{q}\!\left[\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right]$$

The ELBO is the log of a lower bound on this expectation via Jensen's inequality. This perspective motivates the **Importance-Weighted Autoencoder (IWAE)** objective, which uses multiple samples to tighten the bound.

## The Fundamental Identity

The three derivations establish the same result from different angles:

$$\log p(\mathbf{X} | \theta) = \mathcal{L}(q, \theta) + D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)$$

This identity has three immediate consequences. First, since $D_{\text{KL}} \geq 0$, the ELBO is always a lower bound on the log-evidence. Second, the gap between the true log-likelihood and the ELBO is exactly the KL divergence from $q$ to the true posterior. Third, maximising the ELBO over $q$ is equivalent to minimising $D_{\text{KL}}(q \| p_{\text{posterior}})$, which is the variational inference objective.

## Alternative Formulations

The ELBO can be rewritten in several equivalent forms, each providing different insight.

### Joint Form

$$\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] - \mathbb{E}_{q}[\log q(\mathbf{Z})]$$

The first term is the expected complete-data log-likelihood; the second is the negative entropy of $q$.

### Reconstruction + Regularisation Form

Factoring the joint as $p(\mathbf{X}, \mathbf{Z} | \theta) = p(\mathbf{X} | \mathbf{Z}, \theta) \, p(\mathbf{Z})$:

$$\mathcal{L}(q, \theta) = \underbrace{\mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \theta)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z})\bigr)}_{\text{regularisation}}$$

The reconstruction term rewards accurate data modelling; the KL term penalises deviation from the prior. This is the standard form used in VAE training.

### Entropy Form

$$\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \theta)] + \mathbb{E}_{q}[\log p(\mathbf{Z})] + H[q]$$

where $H[q] = -\mathbb{E}_{q}[\log q(\mathbf{Z})]$. The entropy term encourages $q$ to be spread out, preventing collapse to a point mass.

### Negative Free Energy

In the physics literature the ELBO is the negative **variational free energy**: $\mathcal{F}(q) = -\mathcal{L}(q, \theta)$. Minimising free energy is equivalent to maximising the ELBO.

## Gap Analysis

### The Gap Is the KL Divergence

$$\text{Gap} = \log p(\mathbf{X} | \theta) - \mathcal{L}(q, \theta) = D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)$$

The gap is zero if and only if $q = p(\mathbf{Z} | \mathbf{X}, \theta)$.

### Factors Affecting the Gap

An expressive $q$ family makes the gap smaller; a complex or multimodal true posterior makes it larger. If the true posterior is multimodal but $q$ is restricted to unimodal distributions (e.g., diagonal Gaussian), a non-zero gap is inevitable.

### Tightening the Bound

Methods to reduce the gap include richer variational families (normalising flows, autoregressive posteriors), importance weighting (IWAE), and hierarchical latent structures with multiple stochastic layers.

## Connection to the EM Algorithm

The EM algorithm is a special case of ELBO maximisation where $q$ is set to the exact posterior.

**E-step.** Set $q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$. This makes $D_{\text{KL}} = 0$, so the bound is tight: $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(\mathbf{X} | \theta^{(t)})$.

**M-step.** Maximise $\mathcal{L}(q^{(t+1)}, \theta)$ over $\theta$:

$$\theta^{(t+1)} = \arg\max_\theta \, \mathbb{E}_{q^{(t+1)}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$$

Since the entropy $H[q^{(t+1)}]$ does not depend on $\theta$, this reduces to maximising the expected complete-data log-likelihood.

**Monotonic improvement.** After the M-step, $\theta$ changes to $\theta^{(t+1)}$, which may reintroduce a positive gap. But the likelihood cannot decrease:

$$\log p(\mathbf{X} | \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(\mathbf{X} | \theta^{(t)})$$

### Geometric Interpretation: Tangent Lower Bound

When $q$ is set to the posterior at $\theta^{(t)}$, the ELBO touches the log-likelihood curve at $\theta^{(t)}$ and lies below it elsewhere. The M-step moves to the peak of this lower bound. This is an instance of **Minorisation-Maximisation (MM)**.

### EM vs Variational Inference vs VAE

| Aspect | EM | Variational Inference | VAE |
|--------|----|-----------------------|-----|
| **Posterior** | Exact $p(\mathbf{Z} \| \mathbf{X}, \theta)$ | Restricted family $q \in \mathcal{Q}$ | Neural network $q_\phi(\mathbf{Z} \| \mathbf{X})$ |
| **Updates** | Alternating E/M | Coordinate ascent VI | Joint SGD over $(\theta, \phi)$ |
| **Inference** | Per data point | Per data point | Amortised across data |
| **Monotonic** | Yes | Yes (within family) | No guarantee |

## Connection to VAEs

In a VAE the encoder $q_\phi(\mathbf{Z} | \mathbf{X})$ and decoder $p_\theta(\mathbf{X} | \mathbf{Z})$ are jointly trained by maximising the ELBO in its reconstruction + regularisation form:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

The reconstruction term is estimated via Monte Carlo (often a single sample suffices), while the KL term admits a closed form when both $q_\phi$ and the prior $p(z)$ are Gaussian. Gradients flow through the sampling step via the **reparameterisation trick**: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.

### Decoder Choice and Reconstruction Loss

The reconstruction term $\mathbb{E}_q[\log p_\theta(x|z)]$ corresponds to different losses depending on the decoder distribution: a Gaussian decoder gives MSE loss ($-\|x - \hat{x}\|^2 / 2\sigma^2$ plus constants), while a Bernoulli decoder gives binary cross-entropy loss.

### The $\beta$-VAE Trade-off

The $\beta$-VAE objective $\mathcal{L} = \mathbb{E}_q[\log p_\theta(x|z)] - \beta \, D_{\text{KL}}$ interpolates between reconstruction quality ($\beta \to 0$) and latent space regularity ($\beta \to \infty$). The standard VAE corresponds to $\beta = 1$.

For a detailed treatment of KL divergence computation (including the Gaussian closed form), see [KL Divergence](../../ch02/loss/kl_divergence.md). For the reparameterisation trick and full VAE loss implementation, see the [VAE Loss Function](../../ch09/vae/vae_loss.md) page.

## ELBO for a Gaussian Model (Worked Example)

Consider a conjugate Gaussian model: prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$, likelihood $x_i | \theta \sim \mathcal{N}(\theta, \sigma^2)$ for $i = 1, \ldots, n$, and variational family $q(\theta) = \mathcal{N}(m, s^2)$.

**Expected log-likelihood:**

$$\mathbb{E}_q[\log p(\mathcal{D} | \theta)] = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\!\left[\sum_{i=1}^n (x_i - m)^2 + n s^2\right]$$

using $\mathbb{E}_q[(x_i - \theta)^2] = (x_i - m)^2 + s^2$.

**Expected log-prior:**

$$\mathbb{E}_q[\log p(\theta)] = -\frac{1}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\!\left[(m - \mu_0)^2 + s^2\right]$$

**Entropy:**

$$H[q] = \frac{1}{2}\log(2\pi e \, s^2)$$

The ELBO is the sum of these three terms and can be optimised analytically or via gradient ascent over $(m, s)$.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict


class GaussianELBO:
    """ELBO computation for Gaussian mean estimation with known variance."""

    def __init__(self, prior_mean: float, prior_std: float,
                 likelihood_std: float):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std

    def elbo(self, data: torch.Tensor, m: torch.Tensor,
             s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO = E_q[log p(D|theta)] - KL(q || prior).

        Returns:
            (elbo, reconstruction_term, kl_term)
        """
        n = len(data)

        # Reconstruction: E_q[log p(D|theta)]
        reconstruction = (
            -0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.sigma**2))
            - 0.5 / self.sigma**2 * (torch.sum((data - m)**2) + n * s**2)
        )

        # KL(N(m, s^2) || N(mu_0, sigma_0^2))
        kl = (
            torch.log(torch.tensor(self.sigma_0) / s)
            + (s**2 + (m - self.mu_0)**2) / (2 * self.sigma_0**2)
            - 0.5
        )

        return reconstruction - kl, reconstruction, kl


def optimize_elbo(data: torch.Tensor, elbo_computer: GaussianELBO,
                  n_iterations: int = 500,
                  learning_rate: float = 0.05) -> Dict:
    """Optimise ELBO via gradient ascent using PyTorch autograd."""
    m = torch.tensor([0.0], requires_grad=True)
    log_s = torch.tensor([0.0], requires_grad=True)  # log-scale for positivity

    optimizer = torch.optim.Adam([m, log_s], lr=learning_rate)
    history = {'elbo': [], 'm': [], 's': []}

    for _ in range(n_iterations):
        optimizer.zero_grad()
        s = torch.exp(log_s)
        elbo, _, _ = elbo_computer.elbo(data, m, s)
        (-elbo).backward()  # minimise negative ELBO
        optimizer.step()
        history['elbo'].append(elbo.item())
        history['m'].append(m.item())
        history['s'].append(s.item())

    return history, m.detach(), torch.exp(log_s).detach()
```

## ELBO as Model Selection Criterion

The optimised ELBO $\mathcal{L}(q^*, \theta^*)$ approximates the log model evidence $\log p(\mathcal{D} | \mathcal{M})$, which is the standard Bayesian quantity for model comparison. Unlike BIC and AIC, the ELBO uses the full posterior (approximated by $q$) rather than a point estimate, providing a richer penalty for model complexity.

## Summary

| Derivation | Key insight |
|------------|-------------|
| Jensen's inequality | $\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$; equality when $Y$ is constant |
| KL decomposition | $\log p(\mathbf{X}) = \mathcal{L} + D_{\text{KL}}(q \| p_{\text{post}})$ |
| Importance sampling | ELBO lower-bounds the log importance weight expectation |

| Formulation | Expression | Insight |
|-------------|------------|---------|
| Joint | $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z})] - \mathbb{E}_q[\log q]$ | Expected joint + entropy |
| Reconstruction + KL | $\mathbb{E}_q[\log p(\mathbf{X} \| \mathbf{Z})] - D_{\text{KL}}(q \| p(\mathbf{Z}))$ | Data fit vs prior |
| Entropy | $\mathbb{E}_q[\log p(\mathbf{X} \| \mathbf{Z})] + \mathbb{E}_q[\log p(\mathbf{Z})] + H[q]$ | Explicit entropy bonus |

## Exercises

### Exercise 1: Jensen's Equality Condition

Show that Jensen's inequality for the ELBO becomes an equality when $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)$ by verifying that the importance ratio $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z})$ is constant.

### Exercise 2: ELBO for Beta-Binomial

Derive the ELBO for: prior $\theta \sim \text{Beta}(\alpha_0, \beta_0)$, likelihood $x | \theta \sim \text{Binomial}(n, \theta)$, and variational family $q(\theta) = \text{Beta}(\alpha, \beta)$.

### Exercise 3: Numerical Tightness

For the Gaussian model in the worked example, verify numerically that the ELBO equals the log evidence when $q$ is set to the exact posterior.

### Exercise 4: EM vs VAE

Explain why EM guarantees monotonic improvement of the log-likelihood at each iteration but VAE training does not.

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians."
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10.
3. Hoffman, M. D., & Johnson, M. J. (2016). "ELBO Surgery: Yet Another Way to Carve up the Variational Evidence Lower Bound."
4. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."
