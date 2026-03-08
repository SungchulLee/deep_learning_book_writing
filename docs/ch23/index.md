# Chapter Overview

<<<<<<< HEAD
This chapter covers **Suffix Structures**.

# Reference

[Algorithms on Strings, Trees, and Sequences (Gusfield)](https://www.amazon.com/Algorithms-Strings-Trees-Sequences-Computational/dp/0521585198)
=======
Normalizing flows are generative models that learn complex probability distributions by composing sequences of invertible transformations. They uniquely provide both exact, tractable log-likelihoods and single-pass sampling, making them powerful tools for density estimation, data generation, and probabilistic inference. This chapter covers the mathematical foundations, discrete and continuous architectures, training methods, and applications to quantitative finance.

---

## Flow Foundations

- [Introduction to Normalizing Flows](flow_foundations/introduction.md) -- Overview of normalizing flows as generative models with exact density estimation and efficient sampling.
- [Change of Variables](flow_foundations/change_of_variables.md) -- The mathematical foundation describing how probability density transforms under invertible mappings.
- [Invertibility and Flow Composition](flow_foundations/invertibility.md) -- Requirements for bijective transformations and how simple invertible layers compose into expressive models.
- [Jacobian Determinant](flow_foundations/jacobian.md) -- Computational aspects of the Jacobian determinant, the central bottleneck in normalizing flows.

## Flow Architectures

- [Planar, Radial, and Simple Flows](flow_architectures/planar.md) -- Early normalizing flow architectures using simple, analytically tractable transformations.
- [RealNVP and Coupling Layers](flow_architectures/realnvp.md) -- Affine coupling layers that guarantee invertibility by construction with efficient Jacobian computation.
- [Autoregressive Flows](flow_architectures/autoregressive.md) -- MAF and IAF architectures exploiting the chain rule of probability for triangular Jacobians.
- [Glow](flow_architectures/glow.md) -- Refinements to RealNVP including ActNorm, invertible 1x1 convolutions, and systematic multi-scale architecture.
- [Neural Spline Flows](flow_architectures/spline.md) -- Monotonic rational quadratic splines that dramatically increase per-layer expressiveness beyond affine transforms.

## Continuous Flows

- [Continuous Normalizing Flows](continuous_flows/cnf.md) -- Generalization of discrete flows to continuous-time transformations defined by ordinary differential equations.
- [CNF Fundamentals](continuous_flows/cnf_fundamentals.md) -- Core theory of continuous normalizing flows including density evolution via the instantaneous change of variables.
- [Hutchinson Trace Estimator](continuous_flows/hutchinson.md) -- Stochastic method for estimating matrix traces that makes FFJORD and other continuous flows practical.
- [FFJORD: Free-Form Continuous Dynamics](continuous_flows/ffjord.md) -- Making continuous normalizing flows practical with unrestricted architectures and trace estimation.

## Training and Evaluation

- [Training and Evaluation](training/training_evaluation.md) -- Maximum likelihood training, base distribution choices, dequantization, and evaluation metrics for flows.
- [MLE Training](training/mle_training.md) -- Maximum likelihood estimation objective and training loop for normalizing flows.
- [Density Estimation Metrics](training/density_metrics.md) -- Metrics for evaluating learned densities including log-likelihood, bits per dimension, and comparisons.
- [Sample Quality Evaluation](training/sample_quality.md) -- FID, precision-recall, and other metrics for assessing the quality of generated samples.

## Finance Applications

- [Finance Applications of Normalizing Flows](finance_applications/density.md) -- Return distribution modeling, risk measurement, option pricing, and portfolio optimization using flows.
- [Density Estimation for Risk](finance_applications/density_risk.md) -- Exact density estimation for high-dimensional financial data enabling accurate risk assessment.
- [Tail Risk Modeling](finance_applications/tail_risk.md) -- Modeling heavy-tailed financial return distributions for extreme loss probability estimation.
- [Scenario Generation](finance_applications/scenario_generation.md) -- Generating realistic multi-dimensional financial scenarios for risk management and stress testing.
>>>>>>> 96f31bd (...)
