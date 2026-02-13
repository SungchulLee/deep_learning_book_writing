# BLIP: Bootstrapping Language-Image Pre-training

## Learning Objectives

By the end of this section, you will be able to:

- Understand BLIP's unified architecture for understanding and generation tasks
- Explain the captioner-filter bootstrapping mechanism for data cleaning
- Describe the multi-task pre-training objectives (ITC, ITM, LM)

## Overview

BLIP (Li et al., 2022) introduces a unified vision-language model that excels at both understanding tasks (retrieval, VQA) and generation tasks (captioning). Its key innovation is a bootstrapping method that generates and filters captions to improve training data quality.

## Architecture: Multimodal Mixture of Encoder-Decoder

BLIP uses a shared vision transformer with three text modules:

1. **Unimodal text encoder**: For Image-Text Contrastive (ITC) learning
2. **Image-grounded text encoder**: For Image-Text Matching (ITM) with cross-attention
3. **Image-grounded text decoder**: For Language Modeling (LM) / caption generation

```
Image → ViT → Image features
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
Text Encoder    Cross-Attention   Cross-Attention
  (ITC)        Text Encoder      Text Decoder
                 (ITM)             (LM)
```

## Captioning and Filtering (CapFilt)

BLIP improves noisy web data through a bootstrapping process:

1. **Train** initial model on noisy web data
2. **Generate** synthetic captions using the decoder (Captioner)
3. **Filter** both original and synthetic captions using the ITM head (Filter)
4. **Retrain** on the cleaned dataset

This iterative process progressively improves data quality and model performance.

## Pre-training Objectives

$$\mathcal{L} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{LM}}$$

- **ITC** (Image-Text Contrastive): Aligns unimodal representations (like CLIP)
- **ITM** (Image-Text Matching): Binary classification of matched vs. unmatched pairs
- **LM** (Language Modeling): Autoregressive caption generation conditioned on image

## Results

BLIP achieves state-of-the-art on multiple tasks including image-text retrieval, image captioning, and VQA, demonstrating the effectiveness of unified pre-training.

## Summary

BLIP's contributions:

1. **Unified architecture** for both understanding and generation
2. **CapFilt bootstrapping** for data quality improvement
3. **Three complementary objectives** that reinforce each other
4. Strong transfer across diverse vision-language tasks

## References

1. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML.
2. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.
