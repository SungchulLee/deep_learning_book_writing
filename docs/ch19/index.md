# Chapter 19: Approximate Inference

## Overview

Exact Bayesian inference requires computing the posterior distribution $p(\theta \mid \mathcal{D})$, which involves an often-intractable integral over the parameter space. This chapter develops the three major families of approximate inference methods that make Bayesian computation practical for complex models.

**Variational inference** transforms the inference problem into optimization, finding the closest tractable approximation to the true posterior. The **EM algorithm** provides a special case for maximum likelihood estimation with latent variables. **Bayesian neural networks** extend these ideas to deep learning, enabling principled uncertainty quantification in neural networks.

---

## Chapter Structure

### 19.1 Variational Inference

The framework for converting intractable posterior computation into tractable optimization:

- **[VI Framework](../variational_inference/framework.md)** — Variational families, KL divergence, and the optimization perspective on inference
- **[ELBO](../variational_inference/elbo.md)** — Three derivations of the Evidence Lower Bound, tightness conditions, and alternative formulations
- **[Mean Field](../variational_inference/mean_field.md)** — The fully-factorized approximation, optimal factor derivation, and CAVI algorithm
- **[Amortized VI](../variational_inference/amortized.md)** — Inference networks, VAEs, and scaling VI to large datasets
- **[Reparameterization](../variational_inference/reparameterization.md)** — The reparameterization trick, score function estimators, and black-box VI

### 19.2 EM Algorithm

Maximum likelihood estimation in the presence of latent variables:

- **[EM Foundations](../em/foundations.md)** — Latent variable models, marginal likelihood intractability, and the lower-bound strategy
- **[E-Step M-Step](../em/e_step_m_step.md)** — Detailed derivation of both steps, computational mechanics, and practical considerations
- **[GMM with EM](../em/gmm.md)** — Complete treatment of Gaussian Mixture Models: specification, derivation, implementation, and finance applications
- **[EM Variants](../em/variants.md)** — Monte Carlo EM, variational EM, online EM, and stochastic EM for when classical EM is insufficient

### 19.3 Bayesian Neural Networks

Uncertainty quantification through weight distributions in neural networks:

- **[BNN Fundamentals](../bnn/fundamentals.md)** — Why Bayesian? Predictive distributions, uncertainty types, and the posterior over weights
- **[Weight Uncertainty](../bnn/weight_uncertainty.md)** — Prior specification, posterior challenges, and inference methods for neural network weights
- **[Bayes by Backprop](../bnn/bayes_by_backprop.md)** — Variational BNNs with the reparameterization trick and KL annealing

---

## The Inference Landscape

| Method | Output | Accuracy | Scalability | Use Case |
|--------|--------|----------|-------------|----------|
| **Exact (conjugate)** | Analytical posterior | Exact | Limited models | Ch16 conjugate models |
| **Grid approximation** | Discretized posterior | Good (low-$d$) | $O(G^d)$ | 1-2 parameters |
| **MCMC** | Posterior samples | Asymptotically exact | Moderate | Gold-standard inference (Ch18) |
| **Variational inference** | Approximate distribution | Biased | High | Large-scale models |
| **EM algorithm** | Point estimate + latent posterior | Point estimate | High | Latent variable models |
| **Laplace approximation** | Gaussian at MAP | Local | High | Quick approximation |

---

## Connections Between Methods

The methods in this chapter are deeply interconnected:

1. **EM as coordinate ascent on ELBO**: The E-step sets $q = p(Z \mid X, \theta)$ and the M-step optimizes $\theta$ — this is VI with the variational family being the exact conditional posterior
2. **VI generalizes EM**: When the E-step is intractable, variational EM uses an approximate $q$ from a restricted family
3. **ELBO connects everything**: The same objective appears in VI (maximized over $q$), EM (alternating over $q$ and $\theta$), and VAEs (jointly over encoder $q_\phi$ and decoder $p_\theta$)
4. **BNNs apply VI to neural networks**: Bayes by Backprop is amortized VI with the variational family being Gaussian distributions over weights

---

## Prerequisites

- Bayesian foundations: Bayes' theorem, posteriors, conjugacy (Ch16)
- Probability distributions (Ch3)
- Optimization and gradient descent (Ch5)
- Neural network basics (Ch20-21) for the BNN section

---

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *JASA*, 112(518), 859-877.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 10.
3. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *JRSS-B*, 39(1), 1-38.
4. Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks. *ICML*.
