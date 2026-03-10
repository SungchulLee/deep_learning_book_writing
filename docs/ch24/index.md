# Chapter 24: Autoregressive Models


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

Autoregressive models decompose joint probability distributions into products of conditional distributions using the chain rule of probability, enabling exact density evaluation and sequential generation. From PixelCNN for images to WaveNet for audio and GPT-style transformers for text, autoregressive models underpin many of the most successful generative systems in modern deep learning. This chapter covers the theoretical foundations, key architectures, training strategies, evaluation methods, and finance applications.

---

## Foundations

- Autoregressive Models Overview -- Educational overview of autoregressive models with PyTorch implementations and examples.
- Autoregressive Factorization -- Chain rule decomposition of joint distributions into tractable conditional distributions.
- Ordering Strategies -- How the choice of variable ordering affects model complexity and natural orderings for different data types.
- Density Estimation Perspective -- Autoregressive models as density estimators with connections to compression and information theory.
- Masking Approaches -- Mechanisms for enforcing the autoregressive property including causal masking in attention and weight masking.

## Factorization

- [Autoregressive Factorization (Extended)](factorization/factorization.md) -- In-depth treatment of chain rule factorization as the mathematical foundation for all autoregressive generative models.

## PixelCNN

- PixelCNN: Autoregressive Image Generation -- Landmark autoregressive model for images using masked convolutions with raster scan ordering.
- PixelCNN Architecture -- Detailed architecture of PixelCNN with stacked masked convolutional layers.
- Masked Convolutions -- Key building block that constrains convolutions to access only previously generated pixels.
- Gated PixelCNN -- Addresses the blind spot problem and improves expressiveness with gated activations and two-stack architecture.
- PixelCNN++ -- Improvements including discretized logistic mixture likelihood and U-Net-like multi-resolution architecture.

## WaveNet

- WaveNet: Autoregressive Audio Generation -- Breakthrough autoregressive model for raw audio waveforms using dilated causal convolutions.
- WaveNet Architecture -- Core architecture with stacked dilated causal convolutions and exponentially increasing dilation rates.
- Dilated Causal Convolutions -- Key building block enabling exponentially growing receptive fields without proportional increase in computation.
- Conditioning Mechanisms -- Global and local conditioning for controlling generated audio in text-to-speech and other applications.

## Transformers

- [Autoregressive Transformers](transformers/transformers.md) -- The transformer architecture as the dominant paradigm for autoregressive sequence modeling with causal masking.
- Image GPT -- Applying the GPT architecture directly to image generation by treating images as sequences of pixels.
- Autoregressive Image Generation -- Modern token-based approaches combining VQGAN with transformers for image generation.
- Connection to LLMs -- How large language models are autoregressive models and why techniques transfer across modalities.

## Training

- [Teacher Forcing](training/teacher_forcing.md) -- Standard training strategy using ground-truth inputs for efficient parallel training of autoregressive models.
- Training Stability -- Practical techniques for stable training at scale including gradient clipping, warmup, and normalization.
- [Scheduled Sampling](training/scheduled_sampling.md) -- Bridging the gap between teacher forcing and free-running generation by gradually using model predictions.

## Evaluation

- Comparison with Other Generative Models -- Strengths and weaknesses of autoregressive models relative to VAEs, GANs, flows, and diffusion models.
- Likelihood Evaluation -- Computing, reporting, and interpreting exact log-likelihood metrics for autoregressive models.
- Sample Quality -- Sampling strategies including temperature scaling, top-k, and nucleus sampling for generation quality.

## Finance

- Time Series Forecasting -- Autoregressive models for probabilistic financial time series forecasting with exact density evaluation.
- Volatility Modeling -- Data-driven volatility modeling capturing clustering, leverage effects, and regime dependence.
- Order Flow Modeling -- Modeling sequences of buy and sell orders for market microstructure analysis and optimal execution.
