# BNN Fundamentals

## Overview

Bayesian neural networks (BNNs) extend standard neural networks by placing probability distributions over weights rather than learning point estimates. This principled approach enables uncertainty quantification, provides regularization through priors, and offers a coherent framework for reasoning about what the model knows and doesn't know.

## From Point Estimates to Distributions

Standard neural networks trained with maximum likelihood produce point estimates:

$$\hat{\theta} = \arg\max_\theta \log p(\mathcal{D} \mid \theta)$$

**Critical issues with point estimates**:

1. **No confidence measure**: The network outputs predictions with no indication of reliability
2. **Overconfident extrapolation**: Networks make confident predictions far from training data
3. **Silent failures**: Out-of-distribution inputs cause failure without warning
4. **Miscalibrated probabilities**: Softmax outputs don't reflect true predictive uncertainty

### The Bayesian Alternative

Instead of a single weight vector $\hat{\theta}$, maintain a posterior distribution:

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}$$

### What Bayesian Methods Provide

| Capability | Description |
|------------|-------------|
| **Predictive distributions** | Full distributions over outputs, not point predictions |
| **Uncertainty decomposition** | Separate aleatoric from epistemic uncertainty |
| **Principled regularization** | Priors encode inductive biases |
| **Automatic Occam's razor** | Model complexity penalized through marginal likelihood |
| **Coherent decision theory** | Integrate uncertainty into downstream decisions |

## The Predictive Distribution

The posterior predictive distribution integrates predictions over all plausible weight configurations:

$$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta$$

Approximated via Monte Carlo sampling from the posterior:

$$p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(y^* \mid x^*, \theta^{(s)})$$

## The Three Pillars of BNNs

### 1. Prior Specification

The prior $p(\theta)$ encodes beliefs about weights before observing data. Common choices include Gaussian priors (equivalent to L2 regularization), Laplace priors (sparsity-promoting), and hierarchical priors. See [Prior Specification](./priors.md).

### 2. Posterior Inference

Computing the exact posterior is intractable for neural networks. Available approximation methods span a wide range of accuracy-scalability trade-offs:

| Method | Approach | Accuracy | Scalability |
|--------|----------|----------|-------------|
| **HMC** | Exact MCMC | High | Low |
| **SGLD** | Stochastic MCMC | Medium-High | High |
| **Variational** | Optimization | Medium | High |
| **Laplace** | Gaussian at MAP | Medium | Medium-High |
| **MC Dropout** | Implicit VI | Low-Medium | Very High |
| **Deep Ensembles** | Multiple MAPs | Medium-High | Medium |

See [Posterior Inference](./posterior.md) and [Variational BNN](./variational_bnn.md).

### 3. Prediction with Uncertainty

Given posterior samples or approximation, compute point predictions (mean), uncertainty (variance or entropy), and confidence intervals (quantiles).

## BNN Method Taxonomy

```
Bayesian Neural Network Methods
├── Sampling Methods
│   ├── Hamiltonian Monte Carlo (HMC)
│   ├── Stochastic Gradient MCMC
│   │   ├── SGLD (Langevin Dynamics)
│   │   └── SGHMC (Hamiltonian)
│   └── Cyclical SGMC
│
├── Variational Inference
│   ├── Mean-Field VI (Bayes by Backprop)
│   ├── Natural Gradient VI
│   └── Normalizing Flows
│
├── Laplace Approximation
│   ├── Full Laplace
│   ├── Diagonal / KFAC Laplace
│   └── Last-Layer Laplace
│
└── Implicit / Approximate Methods
    ├── MC Dropout
    ├── Deep Ensembles
    └── SWAG
```

## Challenges in BNNs

### Computational Challenges

**High dimensionality**: Modern networks have $10^6$–$10^9$ parameters, making exact inference impossible.

**Non-conjugacy**: Neural network likelihoods aren't conjugate to standard priors.

**Intractable normalization**: The evidence $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta$ has no closed form.

### Structural Challenges

**Multimodality**: The posterior has many modes due to permutation symmetry, scaling symmetry, and multiple good solutions.

**Weight space vs function space**: Simple priors on weights can induce complex function-space behavior.

**Overparameterization**: With more parameters than data, the prior strongly influences the posterior.

## Method Comparison Summary

| Method | Posterior Form | Multimodal? | Correlations? | Training Cost | Inference Cost |
|--------|---------------|-------------|---------------|---------------|----------------|
| HMC | Exact samples | Yes | Yes | Very High | High |
| SGLD | Approximate samples | Partially | Yes | Medium | High |
| Mean-field VI | Factorized Gaussian | No | No | High | Medium |
| MC Dropout | Implicit (Bernoulli) | No | Limited | Low | Medium |
| Deep Ensembles | Mixture of points | Yes | No | High (M×) | Low (M×) |
| SWAG | Low-rank Gaussian | No | Partial | Low+ | Medium |
| Laplace | Gaussian at MAP | No | Partial | Low+ | Low |

## When to Use Each Method

**MC Dropout**: Quickest path to uncertainty from an existing trained model. Good for prototyping.

**Deep Ensembles**: Best uncertainty quality with straightforward implementation. Recommended default for production.

**Variational BNN**: When you want principled Bayesian treatment and can afford doubled parameter count.

**SWAG**: Good uncertainty from a single training run with minimal overhead.

**Laplace Approximation**: Post-hoc uncertainty for already-trained networks without retraining.

**HMC/SGLD**: Research settings requiring ground-truth posteriors for comparison.

## References

- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- Wilson, A. G. (2020). The case for Bayesian deep learning. *arXiv*.
