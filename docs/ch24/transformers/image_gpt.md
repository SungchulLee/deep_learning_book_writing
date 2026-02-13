# Image GPT

## Overview

Image GPT (iGPT, Chen et al., 2020) applies the GPT architecture directly to image generation by treating images as sequences of pixels and training an autoregressive transformer to predict the next pixel.

## Approach

1. **Reduce resolution**: resize images to 32×32 or 48×48 or 64×64
2. **Quantize colors**: cluster pixel colors into 512 clusters using k-means
3. **Flatten to sequence**: raster scan order → sequence of 1024–4096 tokens
4. **Train GPT**: standard autoregressive transformer on pixel sequences

## Architecture

iGPT uses the same transformer architecture as GPT-2:

- Decoder-only transformer with causal attention
- No positional encoding modifications — standard learned positional embeddings
- Model sizes: iGPT-S (76M), iGPT-M (455M), iGPT-L (1.4B)

## Key Finding: Representations

The surprising finding of iGPT is not generation quality (which is moderate) but **representation quality**: features learned by iGPT transfer well to downstream tasks.

Linear probing on ImageNet using iGPT-L features achieves competitive accuracy with supervised methods, demonstrating that autoregressive pre-training on pixels alone learns semantically meaningful representations.

## Limitations

- **Resolution**: practical only at low resolution (32×32 to 64×64) due to quadratic attention cost
- **Color quantization**: 512 colors is a significant limitation
- **Sampling speed**: generating a 32×32 image requires 1024 sequential steps

## Significance

iGPT demonstrated that the autoregressive transformer framework scales beyond text to other modalities, foreshadowing multimodal models and the general trend of applying language model architectures to non-language domains.
