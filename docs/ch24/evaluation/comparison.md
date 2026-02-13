# Comparison with Other Generative Models

## Overview

Autoregressive models occupy a specific point in the tradeoff space of generative models. Understanding their strengths and weaknesses relative to alternatives guides model selection for specific applications.

## Comprehensive Comparison

| Property | AR Models | VAE | GAN | Flow | Diffusion |
|----------|-----------|-----|-----|------|-----------|
| Exact density | ✓ | ✗ (ELBO) | ✗ | ✓ | ✗ (ELBO) |
| Parallel sampling | ✗ | ✓ | ✓ | ✓ | ✗ |
| Parallel training | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mode coverage | High | High | Low | High | High |
| Sample quality | High | Medium | High | Medium | Very High |
| Architecture constraints | Masking | Encoder + Decoder | Generator + Discriminator | Invertible | U-Net |
| Training stability | High | Medium | Low | High | High |

## When to Use Autoregressive Models

**Best suited for:**
- Tasks requiring exact density evaluation (anomaly detection, compression)
- Sequential data with natural ordering (text, audio, time series)
- Tasks where sample quality matters more than sampling speed
- Multimodal generation with interleaved tokens

**Not ideal for:**
- Real-time generation requiring fast sampling
- High-resolution image generation (quadratic cost in pixels)
- Tasks where mode coverage is less important than per-sample quality

## The Sampling Speed Problem

The fundamental limitation of AR models is sequential sampling. For a sequence of length $T$:

- **AR model**: $T$ forward passes, each using the full model
- **Diffusion**: $T_{\text{steps}}$ forward passes (typically 20–1000)
- **GAN/VAE/Flow**: 1 forward pass

Techniques to accelerate AR sampling include speculative decoding, parallel generation with non-autoregressive refinement, and KV caching.
