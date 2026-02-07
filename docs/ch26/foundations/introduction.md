# Introduction to Diffusion Models

A **diffusion model** is a generative model that learns to create data by reversing a gradual noising process. This section establishes the mathematical framework, defines the key objects, and provides intuition for why the approach works.

## Generative Modelling via Denoising

The central insight is that destroying structure is easy—adding Gaussian noise requires no learning—while recovering structure from noise is hard and must be learned. Diffusion models decompose this hard problem into many small, tractable denoising steps.

Given real data $x_0$ drawn from an unknown distribution $p_{\text{data}}$:

1. **Forward process**: Gradually add Gaussian noise over $T$ steps until $x_T \approx \mathcal{N}(0, I)$.
2. **Training**: At each noise level, train a neural network to predict the noise that was added.
3. **Generation**: Start from $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise using the trained network.

The forward process is fixed and parameter-free. All learning happens in the reverse process, where the network $\epsilon_\theta(x_t, t)$ learns to approximate the true denoising operation.

## The Three Parameterisations

The denoising network can equivalently predict three targets, each offering different advantages:

| Parameterisation | Network output | Relationship |
|------------------|---------------|--------------|
| **Noise prediction** | $\epsilon_\theta(x_t, t) \approx \epsilon$ | DDPM default; simplest training |
| **Score prediction** | $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$ | Score-based framework |
| **Data prediction** | $\hat{x}_\theta(x_t, t) \approx x_0$ | Direct clean-data estimate |

These are related by:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}, \qquad \hat{x}_\theta = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

The noise-prediction parameterisation is standard because it yields a simple mean-squared-error training loss with approximately uniform gradient magnitudes across timesteps.

## Mathematical Setup

We work with continuous data $x \in \mathbb{R}^D$ drawn from $p_{\text{data}}(x)$. The forward process defines a sequence of distributions $q(x_t | x_0)$ indexed by a discrete timestep $t \in \{0, 1, \ldots, T\}$ or continuous time $t \in [0, 1]$. The noise schedule $\{\beta_t\}$ controls the rate of corruption.

Key notation used throughout this chapter:

| Symbol | Meaning |
|--------|---------|
| $x_0$ | Clean data sample |
| $x_t$ | Noisy sample at timestep $t$ |
| $\epsilon \sim \mathcal{N}(0, I)$ | Noise added during forward process |
| $\beta_t$ | Noise variance at step $t$ |
| $\alpha_t = 1 - \beta_t$ | Signal retention at step $t$ |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | Cumulative signal retention |
| $\epsilon_\theta(x_t, t)$ | Neural network noise predictor |
| $s_\theta(x_t, t)$ | Neural network score predictor |

## Why It Works: Intuition

The forward process creates a smooth bridge between the complex, unknown data distribution and a simple Gaussian. Each denoising step is a small perturbation—the distribution $q(x_{t-1}|x_t, x_0)$ is well-approximated by a Gaussian when $\beta_t$ is small. This means each reverse step only needs to learn a simple conditional Gaussian, even though the overall mapping from noise to data is highly complex.

The training objective decomposes into independent per-timestep losses. At each step, the network sees a noisy version of a training example and learns to predict the noise that was added. This is a standard regression problem—no adversarial dynamics, no intractable normalisation constants—which explains the training stability of diffusion models.

## Quantitative Finance Motivation

Diffusion models offer several properties valuable for quantitative finance:

**Distributional fidelity.** Unlike GANs, diffusion models cover the full support of the data distribution without mode collapse—critical for capturing tail risks in financial return distributions.

**Conditional generation.** The framework naturally supports conditioning on market regimes, macroeconomic variables, or partial observations, enabling scenario-conditional simulation.

**Density estimation.** The probability flow ODE formulation provides (approximate) log-likelihoods, useful for anomaly detection and model comparison.

**Temporal structure.** The iterative refinement process can be adapted to respect temporal dependencies in financial time series, producing realistic multi-step trajectories.

## What Follows

The next three sections develop the mathematical foundations: the forward process that defines how noise is added (§25.1.2), the reverse process that defines how noise is removed (§25.1.3), and the training objective that connects them (§25.1.4). Subsequent sections build on this foundation with score-based models (§25.2), the DDPM framework (§25.3), and practical extensions.
