# Chapter 15: Autoregressive & Energy-based Models

## Chapter Overview

This chapter explores two foundational paradigms in generative modeling that complement the diffusion models covered in Chapter 14: **autoregressive models** and **energy-based models (EBMs)**. Both approaches offer distinct perspectives on learning and sampling from complex probability distributions, with unique trade-offs in expressiveness, computational efficiency, and sample quality.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the autoregressive factorization of probability distributions
- Implement and train autoregressive models for images and sequences
- Understand the energy-based perspective on generative modeling
- Compare and contrast different generative modeling paradigms
- Apply Neural ODE methods for continuous-time generative modeling
- Evaluate generative models using appropriate metrics

---

## Chapter Structure

### 15.1 Autoregressive Models

Autoregressive models decompose the joint distribution of data into a product of conditional distributions, generating data one element at a time:

$$
p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i \mid x_1, \ldots, x_{i-1})
$$

This section covers:

- **[Autoregressive Factorization](autoregressive/factorization.md)**: Mathematical foundations and ordering strategies
- **[PixelCNN](autoregressive/pixelcnn.md)**: Convolutional autoregressive models for images
- **[WaveNet](autoregressive/wavenet.md)**: Dilated causal convolutions for audio
- **[Autoregressive Transformers](autoregressive/transformers.md)**: Attention-based autoregressive generation

### 15.2 Probabilistic Graphical Models

Understanding the graphical structure underlying generative models:

- **[Directed Graphical Models](pgm/directed.md)**: Bayesian networks and belief propagation
- **[Undirected Graphical Models](pgm/undirected.md)**: Markov random fields
- **[Factor Graphs](pgm/factor_graphs.md)**: Unified representation for inference
- **[Inference Algorithms](pgm/inference.md)**: Exact and approximate inference methods

### 15.3 Energy-based Models

Energy-based models define probability distributions through an energy function:

$$
p(\mathbf{x}) = \frac{\exp(-E(\mathbf{x}))}{Z}, \quad Z = \int \exp(-E(\mathbf{x})) d\mathbf{x}
$$

This section covers:

- **[Energy Function Formulation](ebm/energy_function.md)**: Designing and parameterizing energy functions
- **[Boltzmann Distribution](ebm/boltzmann.md)**: Statistical mechanics foundations
- **[Contrastive Divergence](ebm/contrastive_divergence.md)**: Approximate gradient estimation
- **[Restricted Boltzmann Machines](ebm/rbm.md)**: Classic latent variable EBMs
- **[Modern EBMs](ebm/modern_ebm.md)**: Score-based and noise-contrastive approaches

### 15.4 Neural ODE Models

Continuous-time approaches to generative modeling:

- **[ODE Fundamentals](neural_ode/ode_fundamentals.md)**: Differential equations for deep learning
- **[Neural ODE](neural_ode/neural_ode.md)**: Continuous-depth networks
- **[Adjoint Sensitivity Method](neural_ode/adjoint_method.md)**: Memory-efficient backpropagation
- **[Continuous Normalizing Flows](neural_ode/cnf.md)**: Flow models with Neural ODEs
- **[Applications](neural_ode/applications.md)**: Time series and irregularly-sampled data

### 15.5 Generative Model Evaluation

Quantitative assessment of generative model quality:

- **[Inception Score](evaluation/inception_score.md)**: Quality and diversity metrics
- **[FID Score](evaluation/fid.md)**: Fréchet Inception Distance
- **[Precision and Recall](evaluation/precision_recall.md)**: Mode coverage analysis
- **[Likelihood-based Evaluation](evaluation/likelihood.md)**: Bits per dimension
- **[Human Evaluation](evaluation/human_evaluation.md)**: Perceptual studies

---

## Prerequisites

Before studying this chapter, ensure familiarity with:

- **Chapter 11**: [Normalizing Flows](../ch11/flow_foundations/generative_overview.md) - Change of variables, invertible transformations
- **Chapter 13**: [Bayesian Inference](../ch13/bayesian_foundations/probability_belief.md) - Probabilistic reasoning
- **Chapter 14**: [Diffusion Models](../ch14/index.md) - Score-based generative modeling

---

## Key Concepts Preview

### Autoregressive vs Other Generative Models

| Aspect | Autoregressive | VAE | GAN | Flow | Diffusion |
|--------|---------------|-----|-----|------|-----------|
| Exact likelihood | ✓ | ✗ (ELBO) | ✗ | ✓ | ✓ |
| Parallel sampling | ✗ | ✓ | ✓ | ✓ | ✗ |
| Parallel training | ✓* | ✓ | ✓ | ✓ | ✓ |
| Latent space | ✗ | ✓ | ✓ | ✓ | ✗ |
| Mode coverage | ✓ | ✓ | ✗ | ✓ | ✓ |

*Teacher forcing enables parallel training

### Energy-based Models: Key Properties

Energy-based models provide a flexible framework where:

1. **No normalization required during training**: Many EBM training methods avoid computing the partition function $Z$
2. **Flexible architectures**: Any function can serve as an energy function
3. **Connection to score matching**: $\nabla_\mathbf{x} \log p(\mathbf{x}) = -\nabla_\mathbf{x} E(\mathbf{x})$

---

## Connections to Other Chapters

This chapter bridges several important topics:

```
Chapter 14 (Diffusion) ──────────────────────────────┐
     │                                                │
     │ Score functions                                │
     ↓                                                │
Chapter 15.3 (EBMs) ←── Energy-based perspective ────┤
     │                                                │
     │ MCMC sampling                                  │
     ↓                                                │
Chapter 13 (Bayesian) ←── Probabilistic foundations ──┘
     │
     │ Graphical models
     ↓
Chapter 15.2 (PGMs)
     │
     │ Factorization
     ↓
Chapter 15.1 (Autoregressive)
```

---

## Practical Applications

### In Natural Language Processing

- **Language Models**: GPT-style autoregressive transformers
- **Text Generation**: Story writing, code completion
- **Machine Translation**: Autoregressive decoders

### In Computer Vision

- **Image Generation**: PixelCNN, ImageGPT
- **Image Inpainting**: Conditional autoregressive generation
- **Super-resolution**: Autoregressive upsampling

### In Quantitative Finance

- **Scenario Generation**: Sequential market dynamics modeling
- **Risk Modeling**: Energy-based density estimation for tail risks
- **Time Series Forecasting**: Neural ODE for irregular observations

---

## Chapter Summary

After completing this chapter, you will understand:

1. **Autoregressive models** decompose complex distributions into tractable conditionals, enabling exact likelihood computation at the cost of sequential sampling
2. **Energy-based models** offer maximum flexibility in distribution specification, with training challenges addressed by modern techniques
3. **Neural ODEs** provide continuous-time perspectives connecting discrete models to differential equations
4. **Evaluation metrics** each capture different aspects of generative quality—no single metric tells the complete story

---

## Getting Started

Begin with [Autoregressive Factorization](autoregressive/factorization.md) to understand the mathematical foundations, then proceed through PixelCNN and WaveNet for concrete implementations. The EBM sections build on score function concepts from Chapter 14, making that chapter valuable context.

For readers focused on financial applications, the [Neural ODE Applications](neural_ode/applications.md) section offers relevant content for time series modeling with irregular observations.
