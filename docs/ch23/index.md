# Chapter 23: Normalizing Flows

Generative models aim to learn the probability distribution underlying observed data so that new, realistic samples can be drawn from it. **Normalizing flows** approach this problem by transforming a simple base distribution (typically a standard Gaussian) through a sequence of invertible, differentiable mappings until the transformed distribution matches the data. Because every transformation is invertible and its Jacobian determinant is tractable, normalizing flows provide two properties that many other generative models lack simultaneously: **exact log-likelihood computation** and **efficient single-pass sampling**.

The change-of-variables formula is the mathematical engine behind flows. If $z \sim p_Z(z)$ and $x = f(z)$ where $f$ is a diffeomorphism, then

$$
p_X(x) = p_Z\!\bigl(f^{-1}(x)\bigr)\,\left|\det \frac{\partial f^{-1}}{\partial x}\right|
$$

This chapter covers the mathematical foundations, discrete and continuous flow architectures, training methods, and applications to quantitative finance.

---

## Flow Foundations

- Introduction to Normalizing Flows -- Overview of normalizing flows as generative models with exact density estimation and efficient sampling
- Change of Variables -- The mathematical foundation describing how probability density transforms under invertible mappings
- Invertibility and Flow Composition -- Requirements for bijective transformations and how simple invertible layers compose into expressive models
- Jacobian Determinant -- Computational aspects of the Jacobian determinant, the central bottleneck in normalizing flows

## Flow Architectures

- Planar, Radial, and Simple Flows -- Early normalizing flow architectures using simple, analytically tractable transformations
- [RealNVP and Coupling Layers](flow_architectures/realnvp.md) -- Affine coupling layers that guarantee invertibility by construction with efficient Jacobian computation
- Autoregressive Flows -- MAF and IAF architectures exploiting the chain rule of probability for triangular Jacobians
- Glow -- Refinements to RealNVP including ActNorm, invertible 1x1 convolutions, and systematic multi-scale architecture
- Neural Spline Flows -- Monotonic rational quadratic splines that dramatically increase per-layer expressiveness beyond affine transforms

## Continuous Flows

- Continuous Normalizing Flows -- Generalization of discrete flows to continuous-time transformations defined by ordinary differential equations
- CNF Fundamentals -- Core theory of continuous normalizing flows including density evolution via the instantaneous change of variables
- Hutchinson Trace Estimator -- Stochastic method for estimating matrix traces that makes FFJORD and other continuous flows practical
- FFJORD: Free-Form Continuous Dynamics -- Making continuous normalizing flows practical with unrestricted architectures and trace estimation

## Training and Evaluation

- Training and Evaluation -- Maximum likelihood training, base distribution choices, dequantization, and evaluation metrics for flows
- MLE Training -- Maximum likelihood estimation objective and training loop for normalizing flows
- Density Estimation Metrics -- Metrics for evaluating learned densities including log-likelihood, bits per dimension, and comparisons
- Sample Quality Evaluation -- FID, precision-recall, and other metrics for assessing the quality of generated samples

## Finance Applications

- Finance Applications of Normalizing Flows -- Return distribution modeling, risk measurement, option pricing, and portfolio optimization using flows
- Density Estimation for Risk -- Exact density estimation for high-dimensional financial data enabling accurate risk assessment
- Tail Risk Modeling -- Modeling heavy-tailed financial return distributions for extreme loss probability estimation
- Scenario Generation -- Generating realistic multi-dimensional financial scenarios for risk management and stress testing
