# Autoregressive Factorization

## Overview

Autoregressive models decompose a joint distribution into a product of conditionals using the chain rule of probability. This factorization is exact and enables tractable density evaluation and sampling.

## Chain Rule Decomposition

For a $d$-dimensional random variable $x = (x_1, \ldots, x_d)$:

$$p(x) = \prod_{i=1}^{d} p(x_i \mid x_1, \ldots, x_{i-1})$$

Each conditional $p(x_i \mid x_{<i})$ is modeled by a neural network that takes the preceding variables as input.

## Comparison with Other Generative Models

| Model | Density | Sampling | Latent Variables |
|-------|---------|----------|-----------------|
| Autoregressive | Exact, tractable | Sequential | None |
| VAE | Lower bound (ELBO) | Parallel | Continuous |
| GAN | No density | Parallel | Continuous |
| Flow | Exact, tractable | Parallel | Deterministic mapping |
| Diffusion | Lower bound | Sequential | Noise levels |

## Key Properties

**Exact likelihood**: unlike VAEs (which optimize a lower bound) or GANs (which have no density), autoregressive models compute $\log p(x)$ exactly.

**Sequential sampling**: generating a sample requires $d$ sequential steps — one per dimension. This is the primary drawback: sampling is inherently slow and cannot be parallelized.

**Universal approximation**: with sufficient capacity, autoregressive models can represent any distribution, regardless of the ordering of variables.

## Connection to Language Models

Language models are autoregressive models over token sequences:

$$p(w_1, \ldots, w_T) = \prod_{t=1}^{T} p(w_t \mid w_1, \ldots, w_{t-1})$$

GPT and other decoder-only transformers are autoregressive models. The success of LLMs demonstrates the power of the autoregressive framework at scale.
