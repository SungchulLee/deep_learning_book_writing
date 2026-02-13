# Density Estimation Metrics

## Overview

Evaluating normalizing flows requires metrics that assess both the quality of the learned density and the quality of generated samples. Unlike GANs, flows provide exact likelihoods, enabling direct density evaluation.

## Log-Likelihood

The most direct metric: average log-probability on held-out test data:

$$\mathcal{L}_{\text{test}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \log p_\theta(x^{(i)}_{\text{test}})$$

Higher is better. Report in **nats** (natural log) or **bits per dimension** (log base 2, divided by dimensionality).

## Bits Per Dimension (BPD)

$$\text{BPD} = -\frac{\log_2 p(x)}{d} = -\frac{\log p(x)}{d \cdot \ln 2}$$

Enables comparison across datasets with different dimensionalities.

## KL Divergence

If the true density $p^*(x)$ is known (e.g., for synthetic benchmarks):

$$D_{\text{KL}}(p^* \| p_\theta) = \mathbb{E}_{p^*}\left[\log \frac{p^*(x)}{p_\theta(x)}\right]$$

Estimated by sampling from $p^*$ and evaluating $\log p_\theta$.

## Two-Sample Tests

Statistical tests comparing the distribution of generated samples to real data:

- **Maximum Mean Discrepancy (MMD)**: compares kernel mean embeddings
- **Kolmogorov-Smirnov test**: compares marginal CDFs (1D)
- **Energy distance**: generalization of KS to multiple dimensions

## Calibration Metrics

For flows used in uncertainty estimation, check whether the predicted probability integral transform (PIT) values are uniform:

$$u_i = F_\theta(x_i) \quad \text{should be} \quad u_i \sim \text{Uniform}(0,1)$$

## Benchmark Comparison

| Model | CIFAR-10 BPD | ImageNet 32×32 BPD |
|-------|-------------|-------------------|
| RealNVP | 3.49 | 4.28 |
| Glow | 3.35 | 4.09 |
| Flow++ | 3.08 | 3.86 |
| Residual Flow | 3.28 | — |
