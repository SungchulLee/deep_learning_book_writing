# Chapter 23: Normalizing Flows

Normalizing flows transform a simple base distribution—typically a standard Gaussian—into a complex data distribution through a sequence of invertible, differentiable transformations. Unlike VAEs (which optimise a lower bound) or GANs (which lack likelihoods entirely), flows provide **exact log-likelihood computation**, enabling principled density estimation, rigorous model comparison, and direct maximum-likelihood training.

## Why Normalizing Flows?

The name captures two ideas. *Normalizing* refers to the fact that the inverse transformation maps complex data back to a simple (normal) distribution. *Flow* describes the continuous transport of probability mass through the chain of bijections.

$$z_0 \;\xrightarrow{f_1}\; z_1 \;\xrightarrow{f_2}\; z_2 \;\xrightarrow{\;\cdots\;}\; z_K = x$$

The change-of-variables formula gives the density of the output:

$$\log p_X(x) = \log p_Z(z_0) + \sum_{k=1}^{K} \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|$$

This equation is the backbone of every normalizing flow. All architectural choices—coupling layers, autoregressive structures, spline transforms—are ultimately strategies for making this computation tractable.

## Generative Model Landscape

| Model | Exact Likelihood | Fast Sampling | Bijective Latent | Stable Training |
|---|---|---|---|---|
| **Normalizing Flows** | ✓ | ✓ | ✓ | ✓ |
| VAE | ✗ (ELBO) | ✓ | ✗ (stochastic) | ✓ |
| GAN | ✗ | ✓ | ✗ | ✗ (mode collapse) |
| Autoregressive | ✓ | ✗ (sequential) | ✗ | ✓ |
| Diffusion | ✗ (approx) | ✗ (iterative) | ✓ | ✓ |

Flows occupy a unique niche: they are the only family that simultaneously provides exact likelihoods, single-pass sampling, and deterministic bijective encoding.

## Chapter Organisation

**23.1 Flow Foundations** develops the mathematical machinery—change of variables, Jacobian determinants, invertibility requirements, and the composition principle that lets simple layers build expressive models.

**23.2 Flow Architectures** surveys the major architecture families in roughly chronological order: early simple flows (planar, radial, Sylvester), coupling-layer methods (RealNVP, Glow), neural spline flows, and autoregressive flows (MADE, MAF, IAF).

**23.3 Continuous Normalizing Flows** extends the framework from discrete layers to continuous-time ODE-based transformations, covering the instantaneous change-of-variables formula, FFJORD, and Hutchinson's trace estimator.

**23.4 Training and Evaluation** covers practical considerations: maximum-likelihood training, base distribution selection, dequantisation for discrete data, and density-estimation metrics.

**23.5 Finance Applications** applies normalizing flows to quantitative finance: return distribution modelling, risk measurement (VaR, CVaR), option pricing under learned risk-neutral measures, portfolio optimisation with non-Gaussian returns, and scenario generation for stress testing.

## When to Use Flows

Normalising flows are a strong choice when the application demands exact density evaluation together with fast sampling—risk measurement, anomaly detection, or model comparison in finance, for instance. They are less suited to very high-dimensional generation tasks (e.g. high-resolution images) where diffusion models currently dominate, or to settings where a compressed latent space is needed (where a VAE may be more natural). In practice flows combine well with other generative paradigms: flow priors in VAEs, flow matching as a training objective for diffusion, and flow-based surrogates for Monte Carlo simulation.

## Key References

1. Rezende, D. J. & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
2. Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Kingma, D. P. & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
4. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
5. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
6. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
