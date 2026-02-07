# Weight Uncertainty in Bayesian Neural Networks

## Overview

The central challenge of Bayesian neural networks is specifying meaningful priors over weights and performing inference over the resulting posterior. This section covers prior specification, the geometry of neural network weight spaces, and the key inference approaches.

!!! note "Complete Coverage"
    For comprehensive implementations and benchmarks, see **[Ch33: Prior Specification](../../ch33/model_uncertainty/bayesian_methods/prior_weights.md)** and **[Ch33: Posterior Inference](../../ch33/model_uncertainty/bayesian_methods/posterior_inference.md)**.

---

## Prior Distributions on Weights

### Standard Gaussian Prior

The simplest and most common choice:

$$
p(\theta) = \prod_{l=1}^L \prod_{i,j} \mathcal{N}(w_{ij}^{(l)} \mid 0, \sigma_l^2)
$$

**Connection to L2 regularization**: MAP estimation with Gaussian prior equals L2-regularized MLE with $\lambda = 1/(2\sigma^2)$.

### Variance Scaling

The prior variance should scale with layer width to maintain stable signal propagation:

| Method | Formula | Best For |
|--------|---------|----------|
| **Glorot/Xavier** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Tanh, Sigmoid |
| **He** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ | ReLU |

### Sparsity-Inducing Priors

**Laplace Prior** (L1 regularization):

$$
p(w) = \frac{1}{2b} \exp\left(-\frac{|w|}{b}\right)
$$

**Horseshoe Prior** (adaptive shrinkage):

$$
w_j \mid \lambda_j \sim \mathcal{N}(0, \lambda_j^2 \tau^2), \quad \lambda_j \sim \text{Half-Cauchy}(0, 1)
$$

The horseshoe aggressively shrinks small weights toward zero while leaving large weights relatively unaffected — ideal for sparse networks.

**Spike-and-Slab** (structured sparsity):

$$
p(w_j) = (1-\pi) \delta_0(w_j) + \pi \mathcal{N}(w_j \mid 0, \sigma_{\text{slab}}^2)
$$

### Hierarchical Priors

Place hyperpriors on the prior variance:

$$
w_j \sim \mathcal{N}(0, \sigma^2), \quad \sigma^2 \sim \text{Inverse-Gamma}(\alpha, \beta)
$$

This allows the data to determine the appropriate regularization strength — automatic relevance determination (ARD).

---

## Posterior Inference Methods

### The Inference Challenge

The posterior $p(\theta \mid \mathcal{D})$ for a neural network:

- Has no closed form
- Lives in a space of millions of dimensions
- Contains many modes (due to symmetries)
- Has complex, non-convex geometry

### MCMC Methods

**Hamiltonian Monte Carlo (HMC)**: Uses gradient information for efficient exploration. Gold standard for accuracy but prohibitively expensive for large networks.

**Stochastic Gradient Langevin Dynamics (SGLD)**: Adds calibrated noise to SGD updates:

$$
\theta_{t+1} = \theta_t + \frac{\eta_t}{2} \left(\nabla \log p(\theta_t) + \frac{N}{n} \sum_{i \in \text{batch}} \nabla \log p(x_i \mid \theta_t)\right) + \epsilon_t
$$

where $\epsilon_t \sim \mathcal{N}(0, \eta_t \mathbf{I})$. As $\eta_t \to 0$, samples converge to the true posterior.

### Variational Inference

Approximate the posterior with a tractable family (see [Bayes by Backprop](bayes_by_backprop.md)):

$$
q_\phi(\theta) \approx p(\theta \mid \mathcal{D})
$$

Optimize by maximizing the ELBO (see [Ch19: ELBO](../variational_inference/elbo.md)).

### Laplace Approximation

Fit a Gaussian at the MAP estimate:

$$
p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_{\text{MAP}}, \mathbf{H}^{-1})
$$

where $\mathbf{H} = -\nabla^2 \log p(\theta \mid \mathcal{D}) \big|_{\theta_{\text{MAP}}}$ is the Hessian.

For large networks, approximations to the Hessian are necessary: diagonal, KFAC, or low-rank representations.

### Implicit Methods

**MC Dropout**: Reinterpret dropout at test time as approximate variational inference (Gal & Ghahramani, 2016):

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y^* \mid x^*, \hat{w}_t)
$$

where $\hat{w}_t$ represents weights with random dropout masks applied.

**Deep Ensembles**: Train $M$ independent networks from different initializations:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y^* \mid x^*, \theta_m)
$$

Not formally Bayesian but provides excellent uncertainty estimates in practice.

---

## Method Comparison

| Method | Accuracy | Cost | Simplicity | Memory |
|--------|----------|------|------------|--------|
| HMC | ★★★★★ | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ |
| SGLD | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| Variational | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| Laplace | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| MC Dropout | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| Ensembles | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Prior choice** | Must balance informativeness with computational tractability |
| **Variance scaling** | Critical for training stability — He or Glorot initialization |
| **Sparsity priors** | Horseshoe and spike-and-slab for network compression |
| **MCMC** | Accurate but expensive; SGLD offers scalable approximation |
| **Variational** | Fast and scalable; may underestimate uncertainty |
| **MC Dropout** | Simplest practical approach; limited approximation quality |

---

## References

- Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Bluntschli, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
