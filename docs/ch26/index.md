# Chapter 26: Diffusion Models


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

Diffusion models are generative models that learn to create data by reversing a gradual noising process, achieving state-of-the-art results in image generation, audio synthesis, and beyond. By decomposing the difficult generation problem into many small denoising steps, these models produce remarkably high-quality samples with strong mode coverage. This chapter covers the mathematical foundations, score-based methods, DDPM, DDIM, SDE formulations, conditional generation, architectures, evaluation metrics, and finance applications.

---

## Foundations

- Introduction to Diffusion Models -- Mathematical framework for diffusion models: forward noising, reverse denoising, and three parameterizations.
- Forward Diffusion Process -- The fixed, parameter-free Markov chain that gradually corrupts data into Gaussian noise.
- [Reverse Process](foundations/reverse_process.md) -- The learned generative core that transforms noise into structured data by reversing the forward diffusion.
- Training Objective -- From the variational lower bound to the simplified noise-prediction loss and its connection to score matching.

## Score-Based Methods

- Score Function -- The gradient of log-density as the mathematical foundation of score-based generative modeling.
- Score Matching -- Training unnormalized models by matching score functions without computing the partition function.
- [Denoising Score Matching](score_based/denoising_score_matching.md) -- Replacing expensive Hessian computations with simple regression against known noise targets.
- [Sliced Score Matching](score_based/sliced_score_matching.md) -- Scalable alternative using random projections and Hutchinson's trace estimator.
- Score-Based Models Overview -- Comprehensive tutorial on score-based generative models covering theory and implementation.
- Score-Based Methods Leading to Diffusion -- Bridge from Bayesian inference and MCMC sampling to modern diffusion models.
- Quick Start Guide -- Quick start guide for the score-based methods tutorial package.

## DDPM

- DDPM Fundamentals -- Denoising Diffusion Probabilistic Models: the complete framework with forward process, reverse process, and simplified loss.
- [Noise Schedules](ddpm/noise_schedule.md) -- How the variance schedule controls information flow, training dynamics, and generation quality.
- DDPM Training -- Complete training pipeline including EMA, gradient clipping, mixed-precision, and production-ready implementation.
- [DDPM Sampling](ddpm/sampling.md) -- Iterative denoising from pure noise through the learned reverse process with guidance options.
- Diffusion Models Tutorial Overview -- Educational package overview for learning diffusion models from basics to advanced techniques.
- Diffusion Models Collection -- Collection of complete DDPM implementations from toy to expert level with utilities.
- Quick Start Guide -- Installation and quick start for the diffusion models tutorial.
- Theory Guide -- Intuitive theory guide covering forward and reverse diffusion processes for beginners.
- Getting Started -- Entry points for beginners and advanced practitioners in the diffusion models package.
- Quick Reference -- At-a-glance comparison of diffusion model implementations by complexity and capability.
- Package Summary -- Complete file listing and descriptions for the diffusion models educational package.

## DDIM

- [DDIM Fundamentals](ddim/fundamentals.md) -- Denoising Diffusion Implicit Models: faster sampling via non-Markovian processes without retraining.
- Deterministic Sampling -- Fully deterministic generation where the same noise always produces the same output.
- Accelerated Sampling -- Methods to reduce sampling steps from 1000 to 50 or fewer while maintaining quality.

## SDE Framework

- SDE Fundamentals -- Unified continuous-time perspective on diffusion models using stochastic differential equations.
- Variance Preserving SDE (VP-SDE) -- Continuous-time generalization of DDPM maintaining approximately unit variance throughout diffusion.
- Variance Exploding SDE (VE-SDE) -- Continuous-time generalization of NCSN with growing noise and no signal shrinkage.
- Probability Flow ODE -- Deterministic ODE sharing the same marginals as the reverse SDE, enabling exact likelihood and interpolation.

## Conditional Generation

- Classifier Guidance -- Steering diffusion sampling toward desired classes using an external classifier gradient.
- [Classifier-Free Guidance](conditional/classifier_free.md) -- Improving conditional generation quality without a separate classifier by jointly training conditional and unconditional models.
- Text-to-Image Generation -- Generating images from natural language descriptions by combining diffusion models with CLIP text encoders.

## Architectures

- U-Net Denoiser -- The standard encoder-decoder architecture with skip connections and timestep conditioning for diffusion models.
- Diffusion Transformer (DiT) -- Replacing U-Net with a Vision Transformer backbone, demonstrating transformer scaling laws for diffusion.
- Latent Diffusion Models -- Performing diffusion in compressed latent space for dramatic computational savings, underlying Stable Diffusion.

## Evaluation

- [FID for Diffusion Models](evaluation/fid.md) -- Frechet Inception Distance with diffusion-specific considerations including sampling steps and guidance scale.
- [Inception Score for Diffusion Models](evaluation/inception_score.md) -- IS as a secondary metric for diffusion models with practical usage guidelines.
- [Likelihood-Based Evaluation](evaluation/likelihood.md) -- NLL, bits per dimension, and perplexity for principled information-theoretic assessment.
- CLIP Score -- Measuring text-image alignment as the primary metric for text-to-image diffusion models.
- Human Evaluation -- Designing effective human evaluation studies as the gold standard for generative model assessment.

## Finance

- Synthetic Financial Data Generation -- Generating privacy-preserving synthetic datasets that preserve statistical properties of real financial data.
- Scenario Generation -- Producing plausible future market states for risk management, stress testing, and portfolio optimization.
- Time Series Generation -- Generating realistic financial time series preserving fat tails, volatility clustering, and cross-asset correlations.
