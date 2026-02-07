# Chapter 8: Transformers

## Overview

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), fundamentally changed sequence modeling by replacing recurrence with self-attention. This chapter covers the architecture from first principles through modern pretrained models and their application to both NLP and computer vision.

## Chapter Structure

### 8.1 Transformer Architecture

The foundational components of the Transformer:

- **Transformer Architecture** — The complete encoder-decoder model, including input embeddings, positional encoding, multi-head attention, feed-forward networks, and residual connections. Covers the original "Attention Is All You Need" design and its computational complexity.
- **Positional Encoding** — How Transformers inject sequence order information: sinusoidal encodings, learned embeddings, Rotary Position Embeddings (RoPE), and Attention with Linear Biases (ALiBi). Mathematical foundations and length extrapolation properties.
- **Encoder-Decoder Structure** — The three architectural paradigms (encoder-only, decoder-only, encoder-decoder), with detailed implementations of encoder blocks (bidirectional self-attention) and decoder blocks (causal self-attention, cross-attention, KV-caching).
- **Masked Attention** — Causal masking for autoregressive generation: mathematical formulation, mask construction, combining causal and padding masks, Flash Attention, prefix-LM, and sliding window variants.
- **Layer Normalization in Transformers** — Pre-norm vs post-norm placement, RMSNorm, and their effects on training stability and gradient flow.

### 8.2 Pretrained Models

The major pretrained Transformer families:

- **BERT** — Bidirectional encoder representations with masked language modeling and next sentence prediction. Architecture details, fine-tuning for classification, and practical deployment.
- **GPT** — Generative decoder-only models from GPT-1 through GPT-4. Autoregressive language modeling, text generation strategies (top-k, nucleus sampling, beam search), and in-context learning.
- **T5** — The text-to-text framework that unifies all NLP tasks. Encoder-decoder architecture, span corruption pretraining, and the landscape of pretraining objectives (CLM, MLM, ELECTRA, UL2).
- **Transformer Variants** — Architectural comparisons (Transformer vs RNN vs CNN), sparse attention patterns for long sequences, and scaling to billions of parameters.

### 8.3 Attention Visualization and Interpretability

Understanding what Transformers learn through attention map analysis:

- **Attention Map Visualization** — Encoder self-attention, decoder self-attention, and cross-attention maps. Per-head specialization, layer-wise pattern evolution, and limitations of attention-based interpretation.

### 8.4 Training and Inference

Practical aspects of training and deploying Transformers:

- **Training and Inference** — Teacher forcing, autoregressive generation, KV-caching, and the training-inference gap.
- **Training Optimization** — Learning rate warmup schedules, regularization, mixed-precision training, gradient accumulation, and memory optimization.

### 8.5 Vision Transformers

Applying Transformers to computer vision:

- **Vision Transformer (ViT)** — Treating images as sequences of patches, achieving state-of-the-art image classification.
- **Patch Embeddings** — The bridge from continuous images to discrete tokens.
- **Position Embeddings for Images** — Encoding spatial information for 2D patch sequences.
- **Swin Transformer** — Hierarchical vision Transformer with shifted window attention for linear complexity.
- **DeiT** — Data-efficient training through knowledge distillation.
- **CNN vs ViT** — Systematic comparison of inductive biases, data efficiency, and performance.
- **Hybrid Models** — Combining CNN local features with Transformer global attention.
- **Attention Visualization** — Interpreting what vision Transformers see.
- **Training Strategies** — Augmentation, regularization, and optimization for ViTs.

## Prerequisites

- Chapter 5: Neural Network Foundations (backpropagation, optimization)
- Chapter 6: Convolutional Neural Networks (for vision Transformer comparisons)
- Chapter 7: Attention Mechanisms (scaled dot-product attention, multi-head attention)

## Key Notation

| Symbol | Meaning |
|--------|---------|
| $d_{\text{model}}$ | Model/embedding dimension |
| $d_k, d_v$ | Key and value dimensions per head |
| $d_{ff}$ | Feed-forward hidden dimension (typically $4 \times d_{\text{model}}$) |
| $h$ | Number of attention heads |
| $N$ | Number of encoder/decoder layers |
| $L$ | Sequence length |
| $V$ | Vocabulary size |

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5." JMLR.
5. Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words." ICLR.
