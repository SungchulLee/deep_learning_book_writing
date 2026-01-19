# Score Function Definition

The **score function** is a fundamental concept in score-based generative modeling that provides an alternative approach to density estimation and sampling.

## Definition

For a probability distribution $p(x)$, the **score function** is defined as the gradient of the log-density:

$$
s(x) = \nabla_x \log p(x)
$$

This vector field points in the direction of increasing probability density at each point in space.

## Key Properties

### Relationship to Density

The score function encodes the same information as the density $p(x)$ up to a normalization constant. Since:

$$
\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}
$$

the score captures the relative rate of change of the density.

### Normalization-Free

A crucial advantage: the score function **does not depend on the normalization constant** of $p(x)$. If $p(x) = \frac{\tilde{p}(x)}{Z}$ where $Z = \int \tilde{p}(x) dx$, then:

$$
\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)
$$

This property makes score-based methods particularly useful when the normalization constant is intractable.

## Score Function in Different Contexts

### For Gaussian Distributions

For $p(x) = \mathcal{N}(x; \mu, \Sigma)$:

$$
s(x) = -\Sigma^{-1}(x - \mu)
$$

The score points toward the mean, with magnitude inversely proportional to variance.

### For Mixture Models

For a mixture $p(x) = \sum_k \pi_k p_k(x)$:

$$
s(x) = \frac{\sum_k \pi_k p_k(x) s_k(x)}{\sum_k \pi_k p_k(x)}
$$

The mixture score is a weighted average of component scores.

## Why Learn the Score?

### Langevin Dynamics Sampling

Given access to $s(x) = \nabla_x \log p(x)$, we can generate samples via **Langevin dynamics**:

$$
x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t, \quad z_t \sim \mathcal{N}(0, I)
$$

As $\epsilon \to 0$ and $t \to \infty$, samples converge to the target distribution.

### Connection to Diffusion Models

In diffusion models, the score function at different noise levels guides the reverse (denoising) process. The network $\epsilon_\theta(x_t, t)$ in DDPM is related to the score by:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

This establishes that diffusion models are fundamentally learning score functions at multiple noise scales.

## Score Networks

A **score network** $s_\theta(x)$ is a neural network trained to approximate $\nabla_x \log p(x)$. Unlike density estimation networks, score networks:

1. Output vectors (same dimension as input)
2. Do not need to integrate to 1
3. Can be trained without knowing the normalization constant

## Practical Considerations

### Challenges with Raw Score Estimation

In low-density regions, the score function is poorly defined and difficult to estimate. This motivates:

1. **Noise Conditional Score Networks (NCSN)**: Train separate scores for different noise levels
2. **Denoising Score Matching**: Estimate scores of noise-perturbed distributions
3. **Annealed Langevin Dynamics**: Sample through decreasing noise levels

## Summary

The score function $s(x) = \nabla_x \log p(x)$ provides a normalization-free representation of probability distributions. Learning scores enables sampling via Langevin dynamics and forms the theoretical foundation for diffusion models. The key insight is that predicting noise in diffusion models is equivalent to estimating score functions at multiple noise scales.
