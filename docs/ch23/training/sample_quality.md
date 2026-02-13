# Sample Quality Evaluation

## Overview

While normalizing flows provide exact likelihoods, high likelihood does not always correspond to high sample quality. Evaluating the visual and statistical quality of generated samples requires additional metrics.

## FID (Fréchet Inception Distance)

The standard metric for sample quality in image generation:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of Inception-v3 features for real and generated images. Lower FID is better.

## Precision and Recall

Decompose sample quality into:

- **Precision**: fraction of generated samples that fall within the real data manifold (quality)
- **Recall**: fraction of the real data manifold covered by generated samples (diversity)

Flows typically have high recall (good coverage due to MLE training) but may have lower precision than GANs (some samples may be blurry).

## Likelihood vs Sample Quality

High likelihood and high sample quality are not the same:

- A model can achieve high likelihood by placing mass on low-density regions near the data manifold without generating sharp samples
- Conversely, a model can generate sharp samples while assigning low likelihood to some real data points

This tension is particularly relevant for flows vs GANs: flows optimize likelihood (better density estimation), GANs optimize sample quality (sharper images).

## Temperature Scaling

At generation time, sampling from a sharper base distribution (lower temperature) trades diversity for quality:

$$z \sim \mathcal{N}(0, T^2 I), \quad T < 1$$

Lower temperature produces sharper but less diverse samples.

## Domain-Specific Metrics

For applications beyond images, use domain-appropriate metrics: statistical properties of generated time series (autocorrelation, volatility clustering), physical validity of generated molecules, distributional accuracy for risk modeling.
