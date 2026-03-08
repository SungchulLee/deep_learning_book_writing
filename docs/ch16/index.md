# Chapter 16: Approximate Inference

This chapter covers the major families of approximate inference methods that make Bayesian computation tractable for complex models. Variational inference recasts posterior approximation as an optimization problem, the EM algorithm handles latent variable models through iterative expectation and maximization, and Bayesian neural networks extend these ideas to deep learning with principled uncertainty quantification.

---

## Variational Inference

- Overview -- Comprehensive introduction to variational inference and approximate posterior methods
- Theoretical Background -- Bayesian intractability, KL divergence, and the formulation of VI as optimization
- ELBO and KL Derivation -- Detailed derivation of the evidence lower bound and its relationship to KL divergence
- ELBO plus KL Identity -- Step-by-step proof of the decomposition of log-evidence into ELBO and KL divergence
- Approximating the Posterior -- Using parametric models to approximate the posterior distribution in deep VI settings
- [Variational Inference Framework](variational_inference/framework.md) -- Formulating VI as optimization, the relationship to KL divergence, and comparison with other methods
- [Evidence Lower Bound (ELBO)](variational_inference/elbo.md) -- Three complementary derivations, gap analysis, tightness conditions, and connections to EM and VAEs
- [Mean-Field Variational Inference](variational_inference/mean_field.md) -- The fully-factorized approximation, optimal factor derivation, and limitations of ignoring correlations
- [Reparameterization and Black-Box VI](variational_inference/reparameterization.md) -- Score function estimators, variance reduction, and the reparameterization trick for gradient-based VI
- [Amortized Variational Inference](variational_inference/amortized.md) -- Inference networks for scalable posterior approximation, VAEs, and the amortization gap

## Expectation-Maximization

- [EM Foundations](em/foundations.md) -- Latent variable models, the role of hidden variables, and motivation for the EM algorithm
- [E-Step and M-Step](em/e_step_m_step.md) -- Detailed mechanics of each step including posterior computation and expected sufficient statistics
- [Gaussian Mixture Models](em/gmm.md) -- The canonical EM application with full derivation, implementation, and financial extensions
- [EM Variants](em/variants.md) -- Generalized EM, variational EM, and extensions for models lacking closed-form E or M steps

## Bayesian Neural Networks

- [BNN Fundamentals](bnn/fundamentals.md) -- Placing probability distributions over weights for principled uncertainty quantification
- [Weight Uncertainty](bnn/weight_uncertainty.md) -- Prior specification over weights, the geometry of weight spaces, and key inference approaches
- [Bayes by Backprop](bnn/bayes_by_backprop.md) -- Training BNNs via variational inference using ELBO optimization and the reparameterization trick
- BNNs vs Deep Ensembles -- Comparing Bayesian and ensemble approaches to uncertainty quantification in theory and practice
- [Prior Selection](bnn/prior_selection.md) -- Gaussian, Laplace, horseshoe, and spike-and-slab priors with hierarchical and empirical Bayes strategies
- Scalability Challenges -- Computational bottlenecks, approximate inference complexity, and hybrid approaches for large-scale BNNs
