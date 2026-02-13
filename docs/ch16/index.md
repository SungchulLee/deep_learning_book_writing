# Chapter 19: Approximate Inference

## Overview

Exact Bayesian inference requires computing the posterior distribution $p(\theta \mid \mathcal{D})$, which involves an often-intractable integral over the parameter space. For all but the simplest models, the normalization constant $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta$ has no closed-form solution. This chapter develops the three major families of approximate inference methods that make Bayesian computation practical for complex models.

**Variational inference** transforms the inference problem into optimization, finding the closest tractable approximation to the true posterior by maximizing the Evidence Lower Bound (ELBO). The **EM algorithm** provides a powerful special case for maximum likelihood estimation with latent variables, alternating between computing expected sufficient statistics and optimizing parameters. **Bayesian neural networks** extend these variational ideas to deep learning, placing distributions over weights to enable principled uncertainty quantification in neural network predictions.

Together, these methods complement the sampling-based approaches of Chapter 18 (MCMC), offering faster but approximate alternatives that scale to the high-dimensional parameter spaces encountered in modern machine learning and quantitative finance.

---

## Motivation

Consider a portfolio risk model with latent market regimes, or a credit scoring model where we need not just predictions but calibrated uncertainty over default probabilities. In both cases, we need the full posterior—not just a point estimate—but the posterior is intractable. The methods in this chapter address this challenge from complementary angles:

- **Variational inference** provides a distributional approximation that can be optimized efficiently, making it suitable for large-scale models where MCMC would be prohibitively slow
- **The EM algorithm** finds maximum likelihood estimates when the model involves unobserved latent variables, such as mixture components or hidden states
- **Bayesian neural networks** combine the representational power of deep learning with Bayesian uncertainty, enabling models that know what they don't know

---

## Chapter Structure

### 19.1 Variational Inference

The framework for converting intractable posterior computation into tractable optimization:

- **[VI Framework](variational_inference/framework.md)** — Variational families, KL divergence, and the optimization perspective on inference
- **[ELBO](variational_inference/elbo.md)** — Three derivations of the Evidence Lower Bound, tightness conditions, and alternative formulations
- **[Mean Field](variational_inference/mean_field.md)** — The fully-factorized approximation, optimal factor derivation, and CAVI algorithm
- **[Amortized VI](variational_inference/amortized.md)** — Inference networks, VAEs, and scaling VI to large datasets
- **[Reparameterization](variational_inference/reparameterization.md)** — The reparameterization trick, score function estimators, and black-box VI

### 19.2 EM Algorithm

Maximum likelihood estimation in the presence of latent variables:

- **[EM Foundations](em/foundations.md)** — Latent variable models, marginal likelihood intractability, and the lower-bound strategy
- **[E-Step M-Step](em/e_step_m_step.md)** — Detailed derivation of both steps, computational mechanics, and practical considerations
- **[GMM with EM](em/gmm.md)** — Complete treatment of Gaussian Mixture Models: specification, derivation, implementation, and finance applications
- **[EM Variants](em/variants.md)** — Monte Carlo EM, variational EM, online EM, and stochastic EM for when classical EM is insufficient

### 19.3 Bayesian Neural Networks

Uncertainty quantification through weight distributions in neural networks:

- **[BNN Fundamentals](bnn/fundamentals.md)** — Why Bayesian? Predictive distributions, uncertainty types, and the posterior over weights
- **[Weight Uncertainty](bnn/weight_uncertainty.md)** — Prior specification, posterior challenges, and inference methods for neural network weights
- **[Bayes by Backprop](bnn/bayes_by_backprop.md)** — Variational BNNs with the reparameterization trick and KL annealing

---

## The Core Idea: Optimization Instead of Integration

The fundamental insight behind variational inference is converting an integration problem into an optimization problem. Rather than computing

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}$$

directly—which requires the intractable marginal likelihood $p(\mathcal{D})$—we instead search for the member $q^*(\theta)$ of a tractable family $\mathcal{Q}$ that is closest to the true posterior:

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}}\; \text{KL}\bigl(q(\theta) \,\|\, p(\theta \mid \mathcal{D})\bigr)$$

Minimizing this KL divergence is equivalent to maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(q) = \mathbb{E}_{q(\theta)}\bigl[\log p(\mathcal{D} \mid \theta)\bigr] - \text{KL}\bigl(q(\theta) \,\|\, p(\theta)\bigr) \leq \log p(\mathcal{D})$$

This objective decomposes into a data-fit term (expected log-likelihood) and a regularization term (KL divergence from the prior), providing an elegant optimization target that appears throughout this chapter.

---

## The Inference Landscape

| Method | Output | Accuracy | Scalability | Use Case |
|--------|--------|----------|-------------|----------|
| **Exact (conjugate)** | Analytical posterior | Exact | Limited models | Ch 16 conjugate models |
| **Grid approximation** | Discretized posterior | Good (low-$d$) | $O(G^d)$ | 1–2 parameters |
| **MCMC** | Posterior samples | Asymptotically exact | Moderate | Gold-standard inference (Ch 18) |
| **Variational inference** | Approximate distribution | Biased | High | Large-scale models |
| **EM algorithm** | Point estimate + latent posterior | Point estimate | High | Latent variable models |
| **Laplace approximation** | Gaussian at MAP | Local | High | Quick approximation |

The key tradeoff is **accuracy versus scalability**. MCMC provides asymptotically exact samples but can be slow to converge for high-dimensional models. Variational methods sacrifice exactness for speed, finding approximate solutions through efficient optimization. The EM algorithm sits between the two, providing exact E-steps when possible but only yielding point estimates of model parameters.

---

## Connections Between Methods

The methods in this chapter are deeply interconnected:

1. **EM as coordinate ascent on ELBO**: The E-step sets $q(Z) = p(Z \mid X, \theta^{(t)})$ and the M-step optimizes $\theta$. This is variational inference with the variational family being the exact conditional posterior of the latent variables.

2. **VI generalizes EM**: When the E-step is intractable, variational EM replaces the exact conditional $p(Z \mid X, \theta)$ with an approximate $q(Z)$ from a restricted family, then optimizes the resulting ELBO over both $q$ and $\theta$.

3. **ELBO connects everything**: The same objective appears in VI (maximized over $q$), EM (alternating optimization over $q$ and $\theta$), and VAEs (jointly optimized over encoder $q_\phi$ and decoder $p_\theta$ using amortized inference).

4. **BNNs apply VI to neural networks**: Bayes by Backprop is stochastic variational inference where the latent variables are the neural network weights and the variational family consists of factorized Gaussian distributions.

```
                    ELBO: L(q, θ) = E_q[log p(X,Z|θ)] - E_q[log q(Z)]
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   │                      │                      │
            ┌──────▼──────┐        ┌──────▼──────┐       ┌──────▼──────┐
            │     VI      │        │     EM      │       │    VAE      │
            │ max_q L(q)  │        │ alt. max    │       │ max_{φ,θ}   │
            │             │        │ q then θ    │       │ L(q_φ, θ)   │
            └──────┬──────┘        └─────────────┘       └──────┬──────┘
                   │                                            │
            ┌──────▼──────┐                              ┌──────▼──────┐
            │ Mean Field  │                              │  Amortized  │
            │ q = Π q_i   │                              │  q_φ(z|x)  │
            └─────────────┘                              └──────┬──────┘
                                                                │
                                                         ┌──────▼──────┐
                                                         │    BNN      │
                                                         │  q_φ(w)    │
                                                         └─────────────┘
```

---

## Finance Applications

Approximate inference methods are essential in quantitative finance where uncertainty quantification directly impacts decision-making:

| Application | Method | Why |
|-------------|--------|-----|
| Regime detection | EM (GMM / HMM) | Identify latent market states from returns |
| Portfolio optimization | VI | Posterior over expected returns for robust allocation |
| Credit risk modeling | BNN | Calibrated uncertainty over default probabilities |
| Factor model estimation | Variational EM | Latent factors with intractable posteriors |
| Volatility surface modeling | Amortized VI | Fast posterior inference across strikes and maturities |
| Model uncertainty in pricing | BNN | Quantify model risk for exotic derivatives |

---

## Prerequisites

- Bayesian foundations: Bayes' theorem, posteriors, conjugacy (Ch 16)
- Probability distributions and expectations (Ch 3)
- Optimization and gradient descent (Ch 5)
- Information theory: KL divergence (Ch 4)
- Neural network basics (Ch 20–21) for the BNN section

---

## References

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapters 9–10.
3. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.
4. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
5. Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *Proceedings of the 2nd International Conference on Learning Representations (ICLR)*.
6. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An Introduction to Variational Methods for Graphical Models. *Machine Learning*, 37(2), 183–233.
7. Zhang, C., Bütepage, J., Kjellström, H., & Mandt, S. (2018). Advances in Variational Inference. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(8), 2008–2026.
