# Bayesian Neural Network Fundamentals

## Overview

**Bayesian neural networks (BNNs)** extend standard neural networks by placing probability distributions over weights rather than learning point estimates. This enables principled uncertainty quantification, distinguishing between what the model doesn't know about the data (aleatoric uncertainty) and what the model doesn't know about itself (epistemic uncertainty).

!!! note "Comprehensive Coverage"
    For full implementation details, benchmarks, and advanced methods (SWAG, Laplace, deep ensembles), see **[Ch33: Model Uncertainty — Bayesian Methods](../../ch33/model_uncertainty/bayesian_methods/overview.md)**.

---

## Why Bayesian Neural Networks?

Standard neural networks produce point predictions:

$$
\hat{y} = f_{\hat{\theta}}(x)
$$

Bayesian neural networks maintain a posterior distribution:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

This enables:

- **Predictive distributions** instead of point estimates
- **Uncertainty decomposition** (aleatoric vs epistemic)
- **Principled regularization** through priors
- **Better calibration** and out-of-distribution detection
- **Active learning** — knowing what to query next

### Predictive Distribution

The Bayesian predictive distribution integrates over all plausible parameter values:

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, p(\theta \mid \mathcal{D}) \, d\theta
$$

This integral is generally intractable for neural networks, motivating the approximate inference methods of this chapter.

---

## Types of Uncertainty

$$
\boxed{\text{Total Uncertainty} = \text{Aleatoric Uncertainty} + \text{Epistemic Uncertainty}}
$$

| Type | Source | Reducible? | Example |
|------|--------|------------|---------|
| **Aleatoric** | Inherent data noise | No | Measurement noise, market microstructure |
| **Epistemic** | Limited training data | Yes (with more data) | Novel market conditions, unseen asset classes |

### Uncertainty Decomposition (Regression)

$$
\underbrace{\text{Var}[y^*]}_{\text{Total}} = \underbrace{\mathbb{E}_\theta[\sigma^2_\theta(x^*)]}_{\text{Aleatoric}} + \underbrace{\text{Var}_\theta[\mu_\theta(x^*)]}_{\text{Epistemic}}
$$

### Uncertainty Decomposition (Classification)

Using mutual information:

$$
\underbrace{\mathbb{H}[\bar{p}]}_{\text{Total}} = \underbrace{\mathbb{I}[y; \theta]}_{\text{Epistemic}} + \underbrace{\mathbb{E}_\theta[\mathbb{H}[p_\theta]]}_{\text{Aleatoric}}
$$

---

## The Inference Challenge

For a neural network with $D$ parameters, the posterior is defined over a $D$-dimensional space with highly complex geometry:

1. **Non-convex loss landscape**: Multiple modes separated by high-loss barriers
2. **High dimensionality**: Modern networks have millions to billions of parameters
3. **Non-identifiability**: Weight space symmetries create equivalent parameterizations
4. **No conjugacy**: Neural network likelihoods are not conjugate to any standard prior

### Inference Method Landscape

| Method | Accuracy | Scalability | Simplicity |
|--------|----------|-------------|------------|
| HMC | High | Low | Low |
| SGLD | Medium-High | High | Medium |
| Variational (Bayes by Backprop) | Medium | High | Medium |
| Laplace Approximation | Medium | Medium-High | High |
| MC Dropout | Low-Medium | Very High | Very High |
| Deep Ensembles | Medium-High | Medium | High |

---

## Connection to Regularization

Bayesian inference with specific priors corresponds to familiar regularization techniques:

| Prior | Regularization | MAP Objective |
|-------|---------------|---------------|
| $\mathcal{N}(0, \sigma^2 I)$ | L2 / Weight decay | $\mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$ |
| Laplace$(0, b)$ | L1 | $\mathcal{L}(\theta) + \lambda\|\theta\|_1$ |
| Spike-and-Slab | Structured sparsity | Pruning |
| Dropout | Approximate variational inference | MC Dropout |

The key difference: MAP estimation gives a **point estimate** with regularization, while full Bayesian inference maintains the **entire posterior** distribution.

---

## Practical Considerations for Finance

BNNs are particularly valuable in quantitative finance where:

1. **Data is limited**: Financial time series are short relative to model complexity
2. **Non-stationarity**: Market regimes change, making uncertainty awareness critical
3. **Cost of errors is asymmetric**: Overconfident predictions lead to excessive position sizing
4. **Model risk management**: Regulators increasingly require uncertainty quantification

### Position Sizing with Uncertainty

A natural application: scale positions inversely with epistemic uncertainty:

$$
w_i \propto \frac{\hat{\alpha}_i}{\hat{\sigma}_i^{\text{epistemic}}}
$$

This automatically reduces exposure when the model is unsure — a principled risk management approach.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **BNN definition** | Distributions over weights instead of point estimates |
| **Aleatoric uncertainty** | Irreducible data noise |
| **Epistemic uncertainty** | Reducible model uncertainty (decreases with more data) |
| **Predictive distribution** | Integrates over posterior — requires approximate inference |
| **Connection to regularization** | MAP with prior = regularized MLE; full Bayesian goes further |

---

## References

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*.
- MacKay, D. J. C. (1992). A Practical Bayesian Framework for Backpropagation Networks. *Neural Computation*, 4(3), 448-472.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian Deep Learning and a Probabilistic Perspective of Generalization. *NeurIPS*.
