<<<<<<< HEAD
# Chapter Overview

This chapter covers **Computational Geometry**.

# Reference

[Computational Geometry (de Berg et al.)](https://www.springer.com/gp/book/9783540779735)
=======
# Chapter 24: Autoregressive Models

Autoregressive models decompose joint probability distributions into products of conditional distributions using the chain rule of probability, enabling exact density evaluation and sequential generation. From PixelCNN for images to WaveNet for audio and GPT-style transformers for text, autoregressive models underpin many of the most successful generative systems in modern deep learning. This chapter covers the theoretical foundations, key architectures, training strategies, evaluation methods, and finance applications.

---

## Foundations

- [Autoregressive Models Overview](foundations/autoregressive_models_overview.md) -- Educational overview of autoregressive models with PyTorch implementations and examples.
- [Autoregressive Factorization](foundations/factorization.md) -- Chain rule decomposition of joint distributions into tractable conditional distributions.
- [Ordering Strategies](foundations/ordering.md) -- How the choice of variable ordering affects model complexity and natural orderings for different data types.
- [Density Estimation Perspective](foundations/density_estimation.md) -- Autoregressive models as density estimators with connections to compression and information theory.
- [Masking Approaches](foundations/masking.md) -- Mechanisms for enforcing the autoregressive property including causal masking in attention and weight masking.

## Factorization

- [Autoregressive Factorization (Extended)](factorization/factorization.md) -- In-depth treatment of chain rule factorization as the mathematical foundation for all autoregressive generative models.

## PixelCNN

- [PixelCNN: Autoregressive Image Generation](pixelcnn/pixelcnn.md) -- Landmark autoregressive model for images using masked convolutions with raster scan ordering.
- [PixelCNN Architecture](pixelcnn/architecture.md) -- Detailed architecture of PixelCNN with stacked masked convolutional layers.
- [Masked Convolutions](pixelcnn/masked_convolutions.md) -- Key building block that constrains convolutions to access only previously generated pixels.
- [Gated PixelCNN](pixelcnn/gated_pixelcnn.md) -- Addresses the blind spot problem and improves expressiveness with gated activations and two-stack architecture.
- [PixelCNN++](pixelcnn/pixelcnn_plus.md) -- Improvements including discretized logistic mixture likelihood and U-Net-like multi-resolution architecture.

## WaveNet

- [WaveNet: Autoregressive Audio Generation](wavenet/wavenet.md) -- Breakthrough autoregressive model for raw audio waveforms using dilated causal convolutions.
- [WaveNet Architecture](wavenet/architecture.md) -- Core architecture with stacked dilated causal convolutions and exponentially increasing dilation rates.
- [Dilated Causal Convolutions](wavenet/dilated_causal.md) -- Key building block enabling exponentially growing receptive fields without proportional increase in computation.
- [Conditioning Mechanisms](wavenet/conditioning.md) -- Global and local conditioning for controlling generated audio in text-to-speech and other applications.

## Transformers

- [Autoregressive Transformers](transformers/transformers.md) -- The transformer architecture as the dominant paradigm for autoregressive sequence modeling with causal masking.
- [Image GPT](transformers/image_gpt.md) -- Applying the GPT architecture directly to image generation by treating images as sequences of pixels.
- [Autoregressive Image Generation](transformers/ar_image_generation.md) -- Modern token-based approaches combining VQGAN with transformers for image generation.
- [Connection to LLMs](transformers/connection_to_llms.md) -- How large language models are autoregressive models and why techniques transfer across modalities.

## Training

- [Teacher Forcing](training/teacher_forcing.md) -- Standard training strategy using ground-truth inputs for efficient parallel training of autoregressive models.
- [Training Stability](training/stability.md) -- Practical techniques for stable training at scale including gradient clipping, warmup, and normalization.
- [Scheduled Sampling](training/scheduled_sampling.md) -- Bridging the gap between teacher forcing and free-running generation by gradually using model predictions.

## Evaluation

- [Comparison with Other Generative Models](evaluation/comparison.md) -- Strengths and weaknesses of autoregressive models relative to VAEs, GANs, flows, and diffusion models.
- [Likelihood Evaluation](evaluation/likelihood.md) -- Computing, reporting, and interpreting exact log-likelihood metrics for autoregressive models.
- [Sample Quality](evaluation/sample_quality.md) -- Sampling strategies including temperature scaling, top-k, and nucleus sampling for generation quality.

## Finance

- [Time Series Forecasting](finance/time_series.md) -- Autoregressive models for probabilistic financial time series forecasting with exact density evaluation.
- [Volatility Modeling](finance/volatility.md) -- Data-driven volatility modeling capturing clustering, leverage effects, and regime dependence.
- [Order Flow Modeling](finance/order_flow.md) -- Modeling sequences of buy and sell orders for market microstructure analysis and optimal execution.
>>>>>>> 96f31bd (...)
